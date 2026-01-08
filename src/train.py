import argparse
from dataclasses import asdict
import random
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set, Tuple

import torch
from torch import nn
from torch.optim import AdamW
import numpy as np

from sanitized_augment import build_sanitized_augmentor
from tinytransformer import RMSNorm, TinyTransformer, TinyTransformerConfig
from utils import (
    ARCExampleDataset,
    MAX_SEQ_LEN,
    create_dataloader,
    generate_task_color_mappings,
    generate_task_dihedral_orders,
    TaskDihedralAugmentor,
    TaskColorAugmentor,
    tokens_to_string,
)

DEFAULT_DATA_PATH = Path("assets/challenges.json")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    device_str = device_str.lower()
    if device_str == "cuda":
        if torch.cuda.is_available():
            # Prefer TF32 on capable CUDA hardware using the new fp32_precision API.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            return torch.device("cuda")
        print("CUDA not available, falling back to CPU.")
        return torch.device("cpu")
    if device_str == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        print("MPS not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cpu" if device_str not in {"cpu"} else "cpu")


def _capture_rng_state(device: torch.device) -> Dict[str, Any]:
    """Capture Python/numpy/torch RNG states so training can resume deterministically."""
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    state["torch"] = torch.get_rng_state()
    if torch.cuda.is_available() and device.type == "cuda":
        try:
            state["cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    if (
        hasattr(torch, "mps")
        and torch.backends.mps.is_available()
        and device.type == "mps"
    ):
        try:
            state["mps"] = torch.mps.get_rng_state()
        except Exception:
            pass
    return state


def _restore_rng_state(state: Optional[Dict[str, Any]], device: torch.device) -> None:
    """Restore RNG state saved in a checkpoint; safe to call with None."""
    if not state:
        return
    try:
        random.setstate(state["python"])
    except Exception:
        pass
    try:
        np.random.set_state(state["numpy"])
    except Exception:
        pass
    try:
        torch.set_rng_state(state["torch"])
    except Exception:
        pass
    if "cuda" in state and torch.cuda.is_available() and device.type == "cuda":
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except Exception:
            pass
    if (
        "mps" in state
        and hasattr(torch, "mps")
        and torch.backends.mps.is_available()
        and device.type == "mps"
    ):
        try:
            torch.mps.set_rng_state(state["mps"])
        except Exception:
            pass


def _select_color_aug_seed(args: argparse.Namespace, is_eval: bool) -> int:
    if is_eval:
        seed = getattr(args, "color_aug_seed_eval", None)
        if seed is None:
            seed = getattr(args, "color_aug_seed", None)
    else:
        seed = getattr(args, "color_aug_seed", None)
    if seed is None:
        seed = args.seed
    return int(seed)


def _select_dihedral_aug_seed(args: argparse.Namespace) -> int:
    seed = getattr(args, "dihedral_aug_seed", None)
    if seed is None:
        seed = args.seed
    return int(seed)


def _build_color_augmentor(
    args: argparse.Namespace, dataset: ARCExampleDataset, is_eval: bool
) -> Optional[TaskColorAugmentor]:
    flag_name = "enable_color_aug_eval" if is_eval else "enable_color_aug_train"
    max_name = "max_color_augments_eval" if is_eval else "max_color_augments_train"
    enabled = bool(getattr(args, flag_name, False))
    max_augments = int(getattr(args, max_name, 0) or 0)
    if not enabled or max_augments <= 0:
        return None
    seed = _select_color_aug_seed(args, is_eval)
    mappings_by_task = generate_task_color_mappings(
        dataset.task_input_colors, max_augments, seed
    )
    if not mappings_by_task:
        return None
    apply_to_test_split = True
    if not is_eval:
        apply_to_test_split = bool(
            getattr(args, "enable_color_on_aug_test_split_during_training", False)
        )
    return TaskColorAugmentor(
        mappings_by_task=mappings_by_task, apply_to_test_split=apply_to_test_split
    )


def _build_dihedral_augmentor(
    args: argparse.Namespace, dataset: ARCExampleDataset, is_eval: bool
) -> Optional[TaskDihedralAugmentor]:
    flag_name = "enable_dihedral_aug_eval" if is_eval else "enable_dihedral_aug_train"
    enabled = bool(getattr(args, flag_name, False))
    if not enabled:
        return None
    seed = _select_dihedral_aug_seed(args)
    orders_by_task = generate_task_dihedral_orders(dataset.task_ids, seed)
    if not orders_by_task:
        return None
    apply_to_test_split = True
    if not is_eval:
        apply_to_test_split = bool(
            getattr(args, "enable_dihedral_on_aug_test_split_during_training", False)
        )
    return TaskDihedralAugmentor(
        orders_by_task=orders_by_task, apply_to_test_split=apply_to_test_split
    )


def train_one_epoch(
    model: TinyTransformer,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    start_step: int = 0,
    log_train_strings: bool = False,
    log_train_limit: int = 0,
    log_file: Optional[Path] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> int:
    model.train()
    step = start_step
    total_loss = 0.0
    total_input_loss = 0.0
    total_output_loss = 0.0
    logged = 0
    color_augmentor = getattr(dataloader, "color_augmentor", None)
    color_aug_in_collate = bool(getattr(dataloader, "color_aug_in_collate", False))

    # Enable BF16 autocast only if on CUDA
    use_amp = device.type == "cuda"

    for batch in dataloader:
        step += 1
        # print(f"DEBUG: Step {step} sequence index: {batch['example_ids'][0].item()}")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        example_ids = batch["example_ids"].to(device)
        positions_3d = batch["positions_3d"].to(device)
        if (
            color_augmentor is not None
            and not color_aug_in_collate
            and color_augmentor.max_permutations > 0
        ):
            splits = batch.get("splits")
            task_ids = batch.get("task_ids")
            if splits and task_ids:
                mappings = [
                    color_augmentor.mapping_for_task(task_id, split)
                    for task_id, split in zip(task_ids, splits)
                ]
                active_mapping = next((m for m in mappings if m is not None), None)
                if active_mapping is not None:
                    vocab_size = active_mapping.size(0)
                    identity = torch.arange(vocab_size, dtype=torch.long)
                    batch_maps = torch.stack(
                        [m if m is not None else identity for m in mappings]
                    ).to(device)

                    # Apply map to all tokens: input_ids[b, t] = batch_maps[b, input_ids[b, t]]
                    input_ids = torch.gather(batch_maps, 1, input_ids)

        # (set_to_none is slightly faster)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
        ):
            outputs = model(
                input_ids,
                example_ids,
                attention_mask=attention_mask,
                positions_3d=positions_3d,
            )
            loss = outputs["loss"]
            inp_loss = outputs.get("input_loss")
            out_loss = outputs.get("output_loss")

        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_input_loss += inp_loss.item() if inp_loss is not None else 0.0
        total_output_loss += out_loss.item() if out_loss is not None else 0.0

        # Optional: log the exact serialized strings the model is trained on
        if log_train_strings and logged < log_train_limit:
            bs = input_ids.size(0)
            to_log = min(log_train_limit - logged, bs)
            task_ids = batch.get("task_ids", ["?"] * bs)
            splits = batch.get("splits", ["?"] * bs)
            for i in range(to_log):
                seq_len = int(attention_mask[i].sum().item())
                seq = input_ids[i, :seq_len].detach().cpu().tolist()
                print(
                    "[train string]",
                    f"step={step}",
                    f"split={splits[i]}",
                    f"task={task_ids[i]}",
                    "::",
                    tokens_to_string(seq),
                )
                logged += 1
                if logged >= log_train_limit:
                    break
        if step % 10 == 0:
            avg_loss = total_loss / 10
            avg_inp = total_input_loss / 10
            avg_out = total_output_loss / 10

            current_lr = (
                scheduler.get_last_lr()[0]
                if scheduler
                else optimizer.param_groups[0]["lr"]
            )

            log_msg = f"step={step} lr={current_lr:.2e} losses: avg={avg_loss:.4f} inp={avg_inp:.4f} out={avg_out:.4f}"
            print(log_msg)

            if log_file:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                with open(log_file, "a") as f:
                    f.write(log_msg + "\n")

            total_loss = 0.0
            total_input_loss = 0.0
            total_output_loss = 0.0
    return step


def _is_norm_module(module: nn.Module) -> bool:
    return isinstance(
        module,
        (
            nn.LayerNorm,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.GroupNorm,
            RMSNorm,
        ),
    )


def _is_positional_embedding_param(name: str) -> bool:
    lowered = name.lower()
    return (
        "pos_embed" in lowered
        or "position_embedding" in lowered
        or "positional_embedding" in lowered
    )


def _is_attention_param(name: str) -> bool:
    parts = name.split(".")
    return "attention" in parts


def _build_weight_decay_param_groups(
    model: nn.Module,
    weight_decay: float,
    attention_weight_decay: float,
    token_embedding_weight_decay: float,
    task_embedding_weight_decay: float,
) -> Any:
    """Split parameters into decay groups for linear, attention, and embeddings."""
    decay_params = []
    attention_params = []
    token_embed_params = []
    task_embed_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias"):
            no_decay_params.append(param)
            continue

        module_name = name.rsplit(".", 1)[0] if "." in name else ""
        module = model.get_submodule(module_name) if module_name else model

        if _is_norm_module(module) or _is_positional_embedding_param(name):
            no_decay_params.append(param)
            continue

        if isinstance(module, nn.Embedding):
            if name.startswith("token_embedding."):
                token_embed_params.append(param)
            elif name.startswith("example_embedding."):
                task_embed_params.append(param)
            else:
                no_decay_params.append(param)
            continue

        if _is_attention_param(name):
            attention_params.append(param)
            continue

        if isinstance(module, nn.Linear):
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": weight_decay})
    if attention_params:
        param_groups.append(
            {"params": attention_params, "weight_decay": attention_weight_decay}
        )
    if token_embed_params:
        param_groups.append(
            {
                "params": token_embed_params,
                "weight_decay": token_embedding_weight_decay,
            }
        )
    if task_embed_params:
        param_groups.append(
            {
                "params": task_embed_params,
                "weight_decay": task_embedding_weight_decay,
            }
        )
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return param_groups


def maybe_save_model(
    model: TinyTransformer,
    dataset: ARCExampleDataset,
    data_path: Path,
    save_path: Optional[Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    global_step: Optional[int] = None,
    epoch: Optional[int] = None,
    rng_state: Optional[Dict[str, Any]] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> None:
    if save_path is None:
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "config": asdict(model.config),
        "task_ids": list(dataset.task_ids),
        "data_path": str(data_path),
    }
    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:  # <--- SAVE SCHEDULER
        checkpoint["scheduler_state"] = scheduler.state_dict()
    if global_step is not None:
        checkpoint["global_step"] = int(global_step)
    if epoch is not None:
        checkpoint["epoch"] = int(epoch)
    if rng_state is not None:
        checkpoint["rng_state"] = rng_state
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")


def _normalize_checkpoint_epochs(
    checkpoint_epochs: Optional[object],
    total_epochs: int,
) -> Optional[Set[int]]:
    if checkpoint_epochs is None:
        return None
    if isinstance(checkpoint_epochs, bool):
        raise TypeError("checkpoint_epochs must be an int or list of ints, not bool.")
    if isinstance(checkpoint_epochs, int):
        interval = int(checkpoint_epochs)
        if interval <= 0:
            return None
        return set(range(interval, total_epochs + 1, interval))
    if isinstance(checkpoint_epochs, Sequence) and not isinstance(
        checkpoint_epochs, (str, bytes)
    ):
        epochs: Set[int] = set()
        for item in checkpoint_epochs:
            if isinstance(item, bool):
                raise TypeError(
                    "checkpoint_epochs must be an int or list of ints, not bool."
                )
            try:
                epoch = int(item)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "checkpoint_epochs must be an int or list of ints."
                ) from exc
            if epoch <= 0:
                raise ValueError("checkpoint_epochs entries must be positive.")
            if epoch <= total_epochs:
                epochs.add(epoch)
        return epochs or None
    raise TypeError("checkpoint_epochs must be an int or list of ints.")


def _checkpoint_path_for_epoch(
    save_path: Path,
    epoch: int,
    total_epochs: int,
) -> Path:
    width = max(2, len(str(total_epochs)))
    suffix = save_path.suffix
    stem = save_path.stem
    return save_path.with_name(f"{stem}.epoch{epoch:0{width}d}{suffix}")


def load_checkpoint(checkpoint_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if checkpoint_path is None:
        return None
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state" not in checkpoint:
        checkpoint = {"model_state": checkpoint}
    checkpoint["__path__"] = str(checkpoint_path)
    print(f"Loaded checkpoint from {checkpoint_path}")
    return checkpoint


def infer_num_examples_from_checkpoint(
    checkpoint: Optional[Dict[str, Any]],
) -> Optional[int]:
    if not checkpoint:
        return None
    config = checkpoint.get("config")
    if config and "num_examples" in config:
        return int(config["num_examples"])
    state_dict = checkpoint.get("model_state", {})
    weight = state_dict.get("example_embedding.weight")
    if weight is not None:
        return int(weight.shape[0])
    return None


def build_model_and_data(
    args: argparse.Namespace,
    checkpoint: Optional[Dict[str, Any]] = None,
    reuse_dataset: Optional[ARCExampleDataset] = None,
    is_eval: bool = False,
) -> Tuple[
    TinyTransformer, ARCExampleDataset, torch.utils.data.DataLoader, torch.device, Path
]:
    """Construct dataset, dataloader, and model for a given arg namespace.

    Shared by CLI entrypoints and notebooks so that training, evaluation,
    and inference can be orchestrated independently.
    """
    set_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint = (
        checkpoint if checkpoint is not None else load_checkpoint(args.checkpoint_path)
    )

    data_path = args.data_path
    if data_path is None:
        if checkpoint and "data_path" in checkpoint:
            data_path = Path(checkpoint["data_path"])
        else:
            raise ValueError(
                "--data-path is required when loading checkpoints that do not encode their source dataset. "
                "Please re-run with the same dataset used for training."
            )
    data_path = Path(data_path)

    checkpoint_num_examples = infer_num_examples_from_checkpoint(checkpoint)

    task_whitelist = None
    if checkpoint and "task_ids" in checkpoint:
        task_whitelist = checkpoint["task_ids"]

    if reuse_dataset is not None:
        print("Reusing existing dataset from RAM (skipping 3D pre-computation).")
        dataset = reuse_dataset
    else:
        dataset = ARCExampleDataset(
            json_path=data_path,
            splits=("train", "test"),
            include_outputs=True,
            max_seq_len=MAX_SEQ_LEN,
            task_whitelist=task_whitelist,
            color_aug_mode=getattr(args, "color_aug_mode", None),
        )

    use_sanitized = bool(
        getattr(args, "enable_sanitized_aug_train", False) and not is_eval
    )
    sanitized_augmentor = None
    color_augmentor = None
    dihedral_augmentor = None

    if use_sanitized:
        if getattr(args, "num_workers", 0) != 0:
            raise ValueError("Sanitized augmentation requires num_workers=0.")
        max_sanitized_augments = int(
            getattr(args, "max_sanitized_augments", 0) or 0
        )
        if max_sanitized_augments <= 0:
            max_sanitized_augments = int(
                getattr(args, "max_color_augments_train", 0) or 0
            )
        enable_color = bool(getattr(args, "enable_color_aug_train", False))
        enable_dihedral = bool(getattr(args, "enable_dihedral_aug_train", False))
        color_apply_to_test = bool(
            getattr(args, "enable_color_on_aug_test_split_during_training", False)
        )
        dihedral_apply_to_test = bool(
            getattr(args, "enable_dihedral_on_aug_test_split_during_training", False)
        )
        seed = getattr(args, "sanitized_aug_seed", None)
        if seed is None:
            seed = getattr(args, "color_aug_seed", None)
        if seed is None:
            seed = args.seed

        sanitized_augmentor = build_sanitized_augmentor(
            dataset.examples,
            dataset.task_input_colors,
            max_sanitized_augments=max_sanitized_augments,
            enable_color=enable_color,
            enable_dihedral=enable_dihedral,
            seed=int(seed),
            color_apply_to_test_split=color_apply_to_test,
            dihedral_apply_to_test_split=dihedral_apply_to_test,
        )
        dataset.sanitized_augmentor = sanitized_augmentor
    else:
        color_augmentor = _build_color_augmentor(args, dataset, is_eval=is_eval)
        if color_augmentor is not None:
            dataset.color_task_mappings = color_augmentor.mappings_by_task
            dataset.color_aug_apply_to_test = color_augmentor.apply_to_test_split

        dihedral_augmentor = _build_dihedral_augmentor(args, dataset, is_eval=is_eval)
        if dihedral_augmentor is not None:
            dataset.dihedral_orders_by_task = dihedral_augmentor.orders_by_task
            dataset.dihedral_aug_apply_to_test = dihedral_augmentor.apply_to_test_split

        if dihedral_augmentor is not None and getattr(args, "num_workers", 0) != 0:
            raise ValueError("Dihedral augmentation requires num_workers=0.")

    # We always recreate the dataloader because batch_size might have changed in args
    collate_color_mapper = None
    collate_dihedral_mapper = None
    collate_augment_selector = None
    if sanitized_augmentor is not None:
        collate_augment_selector = sanitized_augmentor.select_for_example
    else:
        collate_color_mapper = (
            color_augmentor.mapping_for_example
            if color_augmentor is not None and getattr(args, "num_workers", 0) == 0
            else None
        )
        collate_dihedral_mapper = (
            dihedral_augmentor.transform_index_for_example
            if dihedral_augmentor is not None
            else None
        )
    dataloader = create_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=not getattr(args, "eval_only", False),
        num_workers=args.num_workers,
        color_mapper=collate_color_mapper,
        dihedral_mapper=collate_dihedral_mapper,
        augment_selector=collate_augment_selector,
    )
    if color_augmentor is not None:
        dataloader.color_augmentor = color_augmentor
        dataloader.color_aug_in_collate = collate_color_mapper is not None
    if dihedral_augmentor is not None:
        dataloader.dihedral_augmentor = dihedral_augmentor
    if sanitized_augmentor is not None:
        dataloader.sanitized_augmentor = sanitized_augmentor

    if (
        checkpoint_num_examples is not None
        and dataset.num_examples != checkpoint_num_examples
    ):
        raise ValueError(
            "Dataset task-count mismatch: "
            f"checkpoint was trained with {checkpoint_num_examples} unique examples but the provided dataset "
            f"currently exposes {dataset.num_examples}. Pass the original --data-path or retrain."
        )

    if checkpoint and "config" in checkpoint:
        config = TinyTransformerConfig(**checkpoint["config"])
    else:
        num_examples = checkpoint_num_examples or max(1, dataset.num_examples)
        d_model = getattr(args, "d_model", 128)
        n_heads = getattr(args, "n_heads", 4)
        d_ff = getattr(args, "d_ff", 512)
        n_layers = getattr(args, "n_layers", 4)
        dropout = getattr(args, "dropout", 0.1)
        config = TinyTransformerConfig(
            num_examples=num_examples,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_layers,
            dropout=dropout,
        )

    if dataset.num_examples != config.num_examples:
        raise ValueError(
            f"Dataset provides {dataset.num_examples} examples but model expects "
            f"{config.num_examples}. Please ensure the dataset/task whitelist matches the checkpoint."
        )

    model = TinyTransformer(config).to(device)

    if checkpoint:
        state_dict = checkpoint["model_state"]
        model.load_state_dict(state_dict, strict=False)
        _restore_rng_state(checkpoint.get("rng_state"), device)

    # Stash checkpoint for downstream consumers (e.g., so train_model can restore the optimizer).
    model._loaded_checkpoint = checkpoint

    return model, dataset, dataloader, device, data_path


@torch.no_grad()
def validate_one_epoch(
    model: TinyTransformer,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Calculates validation loss (Output Loss) on the test set."""
    model.eval()
    total_loss_sum = 0.0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        example_ids = batch["example_ids"].to(device)
        positions_3d = batch["positions_3d"].to(device)

        # We only care about validation on examples that actually have outputs
        # (The val_dataset should be constructed such that has_output is True)
        if not any(batch["has_output"]):
            continue

        outputs = model(
            input_ids,
            example_ids,
            attention_mask=attention_mask,
            positions_3d=positions_3d,
        )

        # 'output_loss' is the specific loss on tokens AFTER the separator
        out_loss = outputs.get("output_loss")
        num_tokens = outputs.get("num_output_tokens")

        if out_loss is not None and num_tokens is not None:
            n = num_tokens.item()
            if n > 0:
                # Reconstruct sum: batch_avg * batch_count
                total_loss_sum += out_loss.item() * n
                total_tokens += n

    if total_tokens == 0:
        return 0.0

    return total_loss_sum / total_tokens


def train_model(
    args: argparse.Namespace,
    model: TinyTransformer,
    dataloader: torch.utils.data.DataLoader,
    dataset: ARCExampleDataset,
    device: torch.device,
    data_path: Path,
    checkpoint: Optional[Dict[str, Any]] = None,
) -> None:
    """Run the training loop only (no evaluation)."""
    if checkpoint is None:
        checkpoint = getattr(model, "_loaded_checkpoint", None)

    do_validate = getattr(args, "do_validate", True)
    val_dataloader = None

    if do_validate:
        val_batch_size = getattr(args, "val_batch_size", args.batch_size)
        print(f"Building validation dataloader (batch_size={val_batch_size})...")

        # Create a separate Validation Dataset/Loader that accesses solutions.json
        # We only include the 'test' split here to calculate validation loss.
        # STRICT SEPARATION: This is the ONLY place load_test_solutions=True is used.
        print("Building validation dataloader (reading hidden solutions)...")
        val_dataset = ARCExampleDataset(
            json_path=data_path,
            splits=("test",),  # Only test split for validation
            include_outputs=True,  # We need outputs to calculate loss
            load_test_solutions=True,  # <--- Loads solutions.json
            max_seq_len=MAX_SEQ_LEN,
            task_whitelist=dataset.task_ids,  # Keep ID mapping consistent
            color_aug_mode=getattr(args, "color_aug_mode", None),
        )

        val_dataloader = create_dataloader(
            dataset=val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        print(f"Validation dataset size: {len(val_dataset)}")
    else:
        print("Validation disabled (skipping solutions.json load).")

    # Extract log file from args if it exists
    log_file = getattr(args, "train_log_file", None)

    save_path = getattr(args, "save_path", None)
    if save_path is not None and not isinstance(save_path, Path):
        save_path = Path(save_path)
    checkpoint_schedule = _normalize_checkpoint_epochs(
        getattr(args, "checkpoint_epochs", None), args.epochs
    )

    attention_weight_decay = getattr(args, "attention_weight_decay", args.weight_decay)
    token_embedding_weight_decay = getattr(args, "token_embedding_weight_decay", 0.0)
    task_embedding_weight_decay = getattr(args, "task_embedding_weight_decay", 0.0)

    param_groups = _build_weight_decay_param_groups(
        model,
        args.weight_decay,
        attention_weight_decay,
        token_embedding_weight_decay,
        task_embedding_weight_decay,
    )

    # Fused AdamW is significantly faster on CUDA.
    use_fused = device.type == "cuda"
    optimizer = AdamW(param_groups, lr=args.lr, fused=use_fused)

    step = int(checkpoint.get("global_step", 0)) if checkpoint else 0

    start_epoch = 0
    if checkpoint:
        saved_epoch = checkpoint.get("epoch")
        if saved_epoch is None:
            saved_epoch = checkpoint.get("epochs_completed")
        if saved_epoch is not None:
            start_epoch = int(saved_epoch)
        elif step > 0:
            steps_per_epoch = len(dataloader)
            if steps_per_epoch > 0:
                start_epoch = step // steps_per_epoch
        if start_epoch > args.epochs:
            print(
                f"Checkpoint has {start_epoch} epochs completed; "
                f"configured epochs={args.epochs}. Nothing left to train."
            )
            start_epoch = args.epochs

    if checkpoint and step > 0:
        print(f"Resuming training from global_step={step}.")
    if checkpoint and start_epoch > 0:
        print(f"Resuming training from epoch {start_epoch + 1}/{args.epochs}.")

    # Restore optimizer state if available so momentum/adam moments resume.
    if checkpoint and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("Restored optimizer state from checkpoint.")

    # print(f"DEBUG CHECK: Optimizer state size = {len(optimizer.state)} (0 = Fresh/Reset, >0 = Restored)")

    # Linear warmup + WSD (warmup, stable, decay) schedule with linear decay tail.
    total_steps = len(dataloader) * args.epochs
    warmup_pct = float(getattr(args, "warmup_pct", 0.02))
    warmup_pct = max(0.0, min(1.0, warmup_pct))
    warmup_steps = int(total_steps * warmup_pct)
    min_lr_factor = float(getattr(args, "lr_floor", 0.01))
    min_lr_factor = max(0.0, min(1.0, min_lr_factor))
    decay_start_pct = float(getattr(args, "wsd_decay_start_pct", 0.8))
    decay_start_pct = max(0.0, min(1.0, decay_start_pct))
    decay_start_step = int(total_steps * decay_start_pct)
    decay_start_step = max(decay_start_step, warmup_steps)
    decay_start_step = min(decay_start_step, total_steps)
    decay_steps = max(1, total_steps - decay_start_step)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        if decay_start_step >= total_steps or current_step < decay_start_step:
            return 1.0
        progress = float(current_step - decay_start_step) / float(decay_steps)
        progress = max(0.0, min(1.0, progress))
        linear = 1.0 - progress
        return min_lr_factor + (1.0 - min_lr_factor) * linear

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Restore scheduler if resuming
    if checkpoint and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        print("Restored scheduler state from checkpoint.")
    elif step > 0:
        # If we didn't save scheduler state but have steps, fast-forward
        for _ in range(step):
            scheduler.step()

    # Compile a specific reference for training execution only.
    # We do this AFTER optimizer loading to ensure parameter consistency.
    # We check for CUDA because compile support on MPS/CPU can be flaky or slower.
    if hasattr(torch, "compile") and device.type == "cuda":
        print("Compiling model for training speedup...")
        training_model = torch.compile(model)
    else:
        training_model = model

    sanitized_augmentor = getattr(dataloader, "sanitized_augmentor", None)
    color_augmentor = getattr(dataloader, "color_augmentor", None)
    dihedral_augmentor = getattr(dataloader, "dihedral_augmentor", None)
    disable_color_aug_last_epochs = int(
        getattr(args, "disable_color_aug_last_epochs", 0) or 0
    )
    disable_color_aug_last_epochs = max(0, disable_color_aug_last_epochs)
    disable_color_aug_start = (
        max(0, args.epochs - disable_color_aug_last_epochs)
        if disable_color_aug_last_epochs > 0
        else None
    )
    disable_dihedral_aug_last_epochs = int(
        getattr(args, "disable_dihedral_aug_last_epochs", 0) or 0
    )
    disable_dihedral_aug_last_epochs = max(0, disable_dihedral_aug_last_epochs)
    disable_dihedral_aug_start = (
        max(0, args.epochs - disable_dihedral_aug_last_epochs)
        if disable_dihedral_aug_last_epochs > 0
        else None
    )
    prev_color_aug_enabled = None
    prev_dihedral_aug_enabled = None

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        color_aug_enabled = (
            True if disable_color_aug_start is None else epoch < disable_color_aug_start
        )
        if sanitized_augmentor is not None:
            if (
                prev_color_aug_enabled is None
                or color_aug_enabled != prev_color_aug_enabled
            ):
                sanitized_augmentor.set_color_enabled(color_aug_enabled)
                if prev_color_aug_enabled is not None or not color_aug_enabled:
                    state = "enabled" if color_aug_enabled else "disabled"
                    print(f"Color augmentation {state} for epoch {epoch + 1}.")
                prev_color_aug_enabled = color_aug_enabled
        elif color_augmentor is not None and hasattr(color_augmentor, "set_enabled"):
            if (
                prev_color_aug_enabled is None
                or color_aug_enabled != prev_color_aug_enabled
            ):
                color_augmentor.set_enabled(color_aug_enabled)
                if prev_color_aug_enabled is not None or not color_aug_enabled:
                    state = "enabled" if color_aug_enabled else "disabled"
                    print(f"Color augmentation {state} for epoch {epoch + 1}.")
                prev_color_aug_enabled = color_aug_enabled
        if (
            sanitized_augmentor is None
            and color_augmentor is not None
            and color_augmentor.max_permutations > 0
        ):
            color_augmentor.set_epoch(epoch)
            print(
                f"Using per-task color permutations (max {color_augmentor.max_permutations})."
            )
        dihedral_aug_enabled = (
            True
            if disable_dihedral_aug_start is None
            else epoch < disable_dihedral_aug_start
        )
        if sanitized_augmentor is not None:
            if (
                prev_dihedral_aug_enabled is None
                or dihedral_aug_enabled != prev_dihedral_aug_enabled
            ):
                sanitized_augmentor.set_dihedral_enabled(dihedral_aug_enabled)
                if prev_dihedral_aug_enabled is not None or not dihedral_aug_enabled:
                    state = "enabled" if dihedral_aug_enabled else "disabled"
                    print(f"Dihedral augmentation {state} for epoch {epoch + 1}.")
                prev_dihedral_aug_enabled = dihedral_aug_enabled
        elif dihedral_augmentor is not None and hasattr(
            dihedral_augmentor, "set_enabled"
        ):
            if (
                prev_dihedral_aug_enabled is None
                or dihedral_aug_enabled != prev_dihedral_aug_enabled
            ):
                dihedral_augmentor.set_enabled(dihedral_aug_enabled)
                if prev_dihedral_aug_enabled is not None or not dihedral_aug_enabled:
                    state = "enabled" if dihedral_aug_enabled else "disabled"
                    print(f"Dihedral augmentation {state} for epoch {epoch + 1}.")
                prev_dihedral_aug_enabled = dihedral_aug_enabled
        if (
            sanitized_augmentor is None
            and dihedral_augmentor is not None
            and dihedral_augmentor.max_transforms > 0
        ):
            dihedral_augmentor.set_epoch(epoch)
            print(
                f"Using per-task dihedral permutations (max {dihedral_augmentor.max_transforms})."
            )
        if sanitized_augmentor is not None:
            sanitized_augmentor.set_epoch(epoch)

        # Run Training
        step = train_one_epoch(
            model=training_model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
            start_step=step,
            log_train_strings=args.log_train_strings,
            log_train_limit=args.log_train_limit,
            log_file=log_file,
            scheduler=scheduler,
        )

        # Run Validation
        if val_dataloader is not None:
            val_loss = validate_one_epoch(
                model=model,  # Use the base model (not compiled) or compiled one, usually base is safer for eval switch
                dataloader=val_dataloader,
                device=device,
            )

            val_msg = (
                f"Epoch {epoch + 1} finished. Validation Output Loss: {val_loss:.4f}"
            )
            print(val_msg)

            if log_file:
                with open(log_file, "a") as f:
                    f.write(val_msg + "\n")

        if checkpoint_schedule and save_path is not None:
            epoch_num = epoch + 1
            if epoch_num in checkpoint_schedule:
                rng_state = _capture_rng_state(device)
                epoch_save_path = _checkpoint_path_for_epoch(
                    save_path, epoch_num, args.epochs
                )
                maybe_save_model(
                    model,
                    dataset,
                    data_path,
                    epoch_save_path,
                    optimizer=optimizer,
                    global_step=step,
                    epoch=epoch_num,
                    rng_state=rng_state,
                    scheduler=scheduler,
                )

    rng_state = _capture_rng_state(device)
    maybe_save_model(
        model,
        dataset,
        data_path,
        save_path,
        optimizer=optimizer,
        global_step=step,
        epoch=args.epochs,
        rng_state=rng_state,
        scheduler=scheduler,
    )
