import argparse
from dataclasses import asdict
import math
import numbers
import random
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set, Tuple

import torch
from torch import nn
from torch.optim import AdamW
import numpy as np

from augment import build_augmentor
from normuon import SingleDeviceNorMuonWithAuxAdam
from tinytransformer import RMSNorm, TinyTransformer, TinyTransformerConfig
from utils import (
    ARCExampleDataset,
    MAX_SEQ_LEN,
    create_dataloader,
)

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    device_str = str(device_str or "cuda").lower()
    if not device_str.startswith("cuda"):
        raise ValueError("Only CUDA is supported. Set device='cuda'.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")
    # Prefer TF32 on capable CUDA hardware using the new fp32_precision API.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return torch.device(device_str)


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


def train_one_epoch(
    model: TinyTransformer,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    gradient_accumulation_steps: int = 1,
    start_step: int = 0,
    log_file: Optional[Path] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    epoch: Optional[int] = None,
    steps_per_epoch: Optional[int] = None,
) -> int:
    model.train()
    step = start_step
    total_loss = 0.0
    total_input_loss = 0.0
    total_output_loss = 0.0
    # Enable BF16 autocast only if on CUDA
    use_amp = device.type == "cuda"

    if steps_per_epoch is None:
        steps_per_epoch = len(dataloader)
    if steps_per_epoch is not None and steps_per_epoch <= 0:
        steps_per_epoch = None

    accum_steps = max(1, int(gradient_accumulation_steps or 1))
    accum_index = 0
    accum_target = accum_steps
    dataloader_length = steps_per_epoch if steps_per_epoch is not None else None

    last_batch_idx = None
    for batch_idx, batch in enumerate(dataloader):
        last_batch_idx = batch_idx
        step += 1
        # print(f"DEBUG: Step {step} sequence index: {batch['example_ids'][0].item()}")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        example_ids = batch["example_ids"].to(device)
        positions_3d = batch["positions_3d"].to(device)
        if accum_index == 0:
            # (set_to_none is slightly faster)
            optimizer.zero_grad(set_to_none=True)
            if dataloader_length is not None:
                remaining = dataloader_length - batch_idx
                accum_target = min(accum_steps, remaining)
            else:
                accum_target = accum_steps

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

        loss_value = loss.item()
        inp_loss_value = inp_loss.item() if inp_loss is not None else 0.0
        out_loss_value = out_loss.item() if out_loss is not None else 0.0

        loss = loss / accum_target
        loss.backward()
        accum_index += 1

        if accum_index >= accum_target:
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None:
                if epoch is None or steps_per_epoch is None:
                    scheduler.step()
                else:
                    epoch_progress = float(epoch) + (batch_idx + 1) / float(
                        steps_per_epoch
                    )
                    scheduler.step(epoch_progress)
            accum_index = 0

        total_loss += loss_value
        total_input_loss += inp_loss_value
        total_output_loss += out_loss_value

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
    if accum_index > 0:
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            if epoch is None or steps_per_epoch is None or last_batch_idx is None:
                scheduler.step()
            else:
                epoch_progress = float(epoch) + (last_batch_idx + 1) / float(
                    steps_per_epoch
                )
                scheduler.step(epoch_progress)
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


def _is_muon_candidate(
    module: nn.Module, name: str, param: nn.Parameter
) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    if name.startswith("lm_head."):
        return False
    if not name.endswith(".weight"):
        return False
    return param.ndim == 2



def _normuon_supported(device: torch.device) -> Tuple[bool, str]:
    if device.type != "cuda":
        return False, "NorMuon requires CUDA."
    bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(bf16_supported) and not bf16_supported():
        return False, "NorMuon requires CUDA bfloat16 support"
    return True, ""


def _collect_param_groups(
    model: nn.Module, *, include_muon: bool
) -> Dict[str, list[nn.Parameter]]:
    groups: Dict[str, list[nn.Parameter]] = {
        "decay": [],
        "attention": [],
        "token_embed": [],
        "task_embed": [],
        "no_decay": [],
        "muon": [],
        "muon_attention": [],
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        module_name = name.rsplit(".", 1)[0] if "." in name else ""
        module = model.get_submodule(module_name) if module_name else model
        is_attention = _is_attention_param(name)

        if include_muon and _is_muon_candidate(module, name, param):
            if is_attention:
                groups["muon_attention"].append(param)
            else:
                groups["muon"].append(param)
            continue

        if name.endswith(".bias"):
            groups["no_decay"].append(param)
            continue

        if _is_norm_module(module) or _is_positional_embedding_param(name):
            groups["no_decay"].append(param)
            continue

        if isinstance(module, nn.Embedding):
            if name.startswith("token_embedding."):
                groups["token_embed"].append(param)
            elif name.startswith("example_embedding."):
                groups["task_embed"].append(param)
            else:
                groups["no_decay"].append(param)
            continue

        if is_attention:
            groups["attention"].append(param)
            continue

        if isinstance(module, nn.Linear):
            groups["decay"].append(param)
        else:
            groups["no_decay"].append(param)

    return groups


def _build_param_groups(
    model: nn.Module,
    weight_decay: float,
    attention_weight_decay: float,
    token_embedding_weight_decay: float,
    task_embedding_weight_decay: float,
) -> Tuple[Sequence[Dict[str, Any]], Sequence[Dict[str, Any]]]:
    """Split params for Muon (linear weights only) and AdamW (everything else)."""
    groups = _collect_param_groups(model, include_muon=True)

    muon_groups = []
    if groups["muon"]:
        muon_groups.append({"params": groups["muon"], "weight_decay": weight_decay})
    if groups["muon_attention"]:
        muon_groups.append(
            {
                "params": groups["muon_attention"],
                "weight_decay": attention_weight_decay,
            }
        )

    adamw_groups = []
    if groups["decay"]:
        adamw_groups.append({"params": groups["decay"], "weight_decay": weight_decay})
    if groups["attention"]:
        adamw_groups.append(
            {"params": groups["attention"], "weight_decay": attention_weight_decay}
        )
    if groups["token_embed"]:
        adamw_groups.append(
            {
                "params": groups["token_embed"],
                "weight_decay": token_embedding_weight_decay,
            }
        )
    if groups["task_embed"]:
        adamw_groups.append(
            {
                "params": groups["task_embed"],
                "weight_decay": task_embedding_weight_decay,
            }
        )
    if groups["no_decay"]:
        adamw_groups.append({"params": groups["no_decay"], "weight_decay": 0.0})

    return muon_groups, adamw_groups



def _move_optimizer_state(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

def _load_optimizer_state(
    optimizer: torch.optim.Optimizer,
    state_dict: Dict[str, Any],
    device: torch.device,
) -> bool:
    def _is_torch_state_dict(value: Any) -> bool:
        return isinstance(value, dict) and "param_groups" in value

    candidate_state: Optional[Dict[str, Any]] = None
    if _is_torch_state_dict(state_dict):
        candidate_state = state_dict
    else:
        state_key = optimizer.__class__.__name__.lower()
        sub_state = state_dict.get(state_key)
        if _is_torch_state_dict(sub_state):
            candidate_state = sub_state

    if candidate_state is None:
        return False

    try:
        optimizer.load_state_dict(candidate_state)
    except (KeyError, ValueError, RuntimeError) as exc:
        print(f"Skipping optimizer state restore ({exc}).")
        return False
    _move_optimizer_state(optimizer, device)
    return True


def _optimizer_identity(optimizer: torch.optim.Optimizer) -> str:
    return optimizer.__class__.__name__.lower()


_OPTIMIZER_HPARAMS_IGNORE_KEYS = {
    "params",
    "fused",
    "foreach",
    "capturable",
    "differentiable",
}


def _sanitize_optimizer_value(value: Any) -> Optional[Any]:
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Number):
        return float(value)
    if isinstance(value, str) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        sanitized = []
        for item in value:
            item_sanitized = _sanitize_optimizer_value(item)
            if item_sanitized is None and item is not None:
                return None
            sanitized.append(item_sanitized)
        return sanitized
    if isinstance(value, dict):
        sanitized_dict = {}
        for key, item in value.items():
            item_sanitized = _sanitize_optimizer_value(item)
            if item_sanitized is None and item is not None:
                return None
            sanitized_dict[key] = item_sanitized
        return sanitized_dict
    return None


def _param_groups_snapshot(
    param_groups: Sequence[Dict[str, Any]],
) -> Sequence[Dict[str, Any]]:
    snapshot = []
    for group in param_groups:
        entry: Dict[str, Any] = {}
        for key, value in group.items():
            if key in _OPTIMIZER_HPARAMS_IGNORE_KEYS:
                continue
            sanitized = _sanitize_optimizer_value(value)
            if sanitized is None and value is not None:
                continue
            entry[key] = sanitized
        snapshot.append(entry)
    return snapshot


def _optimizer_hparams_snapshot(optimizer: torch.optim.Optimizer) -> Any:
    return _param_groups_snapshot(optimizer.param_groups)


def _checkpoint_optimizer_hparams(checkpoint: Dict[str, Any]) -> Optional[Any]:
    state_dict = checkpoint.get("optimizer_state")
    if isinstance(state_dict, dict) and "param_groups" in state_dict:
        return _param_groups_snapshot(state_dict.get("param_groups", []))
    return None


def _optimizer_values_match(left: Any, right: Any) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-8)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False
        return all(_optimizer_values_match(lv, rv) for lv, rv in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return False
        return all(
            _optimizer_values_match(left[key], right[key]) for key in left.keys()
        )
    return left == right


def _optimizer_hparams_changed(
    optimizer: torch.optim.Optimizer, checkpoint: Dict[str, Any]
) -> bool:
    saved = checkpoint.get("optimizer_hparams")
    if saved is None:
        saved = _checkpoint_optimizer_hparams(checkpoint)
    if saved is None:
        return False
    current = _optimizer_hparams_snapshot(optimizer)
    return not _optimizer_values_match(current, saved)


def _apply_param_group_hparams(
    param_groups: Sequence[Dict[str, Any]],
    hparams: Optional[Sequence[Dict[str, Any]]],
) -> None:
    if not isinstance(hparams, Sequence):
        return
    for group, desired in zip(param_groups, hparams):
        if not isinstance(desired, dict):
            continue
        for key, value in desired.items():
            group[key] = value


def _apply_optimizer_hparams(optimizer: torch.optim.Optimizer, hparams: Any) -> None:
    _apply_param_group_hparams(optimizer.param_groups, hparams)


def _optimizer_switch_detected(
    optimizer: torch.optim.Optimizer,
    checkpoint: Dict[str, Any],
) -> bool:
    checkpoint_name = checkpoint.get("optimizer_name")
    if checkpoint_name:
        return str(checkpoint_name).lower() != _optimizer_identity(optimizer)
    # If we are loading an old checkpoint that had the Muon/AdamW split,
    # but we are now using a standard/Normuon optimizer, assume a switch occurred.
    state_dict = checkpoint.get("optimizer_state")
    if isinstance(state_dict, dict) and "muon" in state_dict and "adamw" in state_dict:
        return True
        
    return False


def _build_optimizer(
    args: argparse.Namespace,
    model: nn.Module,
    device: torch.device,
    attention_weight_decay: float,
    token_embedding_weight_decay: float,
    task_embedding_weight_decay: float,
) -> torch.optim.Optimizer:
    optimizer_name = str(getattr(args, "optimizer", "adamw")).lower()
    use_fused = device.type == "cuda"

    muon_groups, adamw_groups = _build_param_groups(
        model,
        args.weight_decay,
        attention_weight_decay,
        token_embedding_weight_decay,
        task_embedding_weight_decay,
    )

    if optimizer_name not in {"normuon"}:
        return AdamW(list(muon_groups) + list(adamw_groups), lr=args.lr, fused=use_fused)

    supported, reason = _normuon_supported(device)
    if not supported:
        print(f"NorMuon unavailable ({reason}); falling back to AdamW.")
        return AdamW(list(muon_groups) + list(adamw_groups), lr=args.lr, fused=use_fused)

    if not muon_groups:
        print("NorMuon requested but no eligible linear weights found; using AdamW.")
        return AdamW(list(adamw_groups), lr=args.lr, fused=use_fused)

    normuon_lr = getattr(args, "normuon_lr", None)
    normuon_lr = 0.02 if normuon_lr is None else float(normuon_lr)
    normuon_momentum = float(getattr(args, "normuon_momentum", 0.95))
    normuon_beta2 = float(getattr(args, "normuon_beta2", 0.95))

    adamw_lr = getattr(args, "adamw_lr", None)
    adamw_lr = args.lr if adamw_lr is None else float(adamw_lr)
    normuon_param_groups = []
    for group in muon_groups:
        normuon_group = dict(group)
        normuon_group["use_muon"] = True
        normuon_group["lr"] = normuon_lr
        normuon_group["momentum"] = normuon_momentum
        normuon_group["beta2"] = normuon_beta2
        normuon_param_groups.append(normuon_group)
    for group in adamw_groups:
        adam_group = dict(group)
        adam_group["use_muon"] = False
        adam_group["lr"] = adamw_lr
        normuon_param_groups.append(adam_group)
    return SingleDeviceNorMuonWithAuxAdam(normuon_param_groups)


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
        checkpoint["optimizer_name"] = _optimizer_identity(optimizer)
        checkpoint["optimizer_hparams"] = _optimizer_hparams_snapshot(optimizer)
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
    device = resolve_device(getattr(args, "device", "cuda"))
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
        )

    use_aug = bool(getattr(args, "enable_aug", False) and not is_eval)
    augmentor = None

    if use_aug:
        max_augments = int(getattr(args, "max_augments", 0) or 0)
        
        augmentor = build_augmentor(
            dataset.examples,
            dataset.task_input_colors,
            max_augments=max_augments,
            enable_color=bool(getattr(args, "enable_color_aug", False)),
            enable_dihedral=bool(getattr(args, "enable_dihedral_aug", False)),
            seed=args.seed,
            color_apply_to_test_split=bool(getattr(args, "color_apply_to_test", False)),
            dihedral_apply_to_test_split=bool(getattr(args, "dihedral_apply_to_test", False)),
        )
        dataset.augmentor = augmentor

    # We always recreate the dataloader because batch_size might have changed in args
    collate_augment_selector = None
    if augmentor is not None:
        collate_augment_selector = augmentor.select_for_example
    dataloader = create_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=not getattr(args, "eval_only", False),
        augment_selector=collate_augment_selector,
    )
    if augmentor is not None:
        dataloader.augmentor = augmentor

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
        )

        val_dataloader = create_dataloader(
            dataset=val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
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

    optimizer = _build_optimizer(
        args,
        model,
        device,
        attention_weight_decay,
        token_embedding_weight_decay,
        task_embedding_weight_decay,
    )
    desired_optimizer_hparams = _optimizer_hparams_snapshot(optimizer)

    batch_sampler = getattr(dataloader, "batch_sampler", None)
    if batch_sampler is not None and hasattr(batch_sampler, "drop_last"):
        batch_sampler.drop_last = True

    step = int(checkpoint.get("global_step", 0)) if checkpoint else 0
    steps_per_epoch = len(dataloader)

    start_epoch = 0
    if checkpoint:
        saved_epoch = checkpoint.get("epoch")
        if saved_epoch is None:
            saved_epoch = checkpoint.get("epochs_completed")
        if saved_epoch is not None:
            start_epoch = int(saved_epoch)
        elif step > 0:
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

    resume_warmup = False
    if checkpoint and _optimizer_switch_detected(optimizer, checkpoint):
        resume_warmup = True
        print("Detected optimizer change; resetting LR schedule.")
    optimizer_hparams_changed = False
    if checkpoint and _optimizer_hparams_changed(optimizer, checkpoint):
        optimizer_hparams_changed = True
        resume_warmup = True
        print("Detected optimizer hyperparameter change; applying new settings.")

    # Restore optimizer state if available so momentum/adam moments resume.
    if checkpoint and "optimizer_state" in checkpoint:
        restored = _load_optimizer_state(
            optimizer, checkpoint["optimizer_state"], device
        )
        if restored:
            if optimizer_hparams_changed:
                _apply_optimizer_hparams(optimizer, desired_optimizer_hparams)
                # Keep scheduler base LRs in sync with updated args after restore.
                for group in optimizer.param_groups:
                    if "initial_lr" in group and "lr" in group:
                        group["initial_lr"] = group.get("lr", group["initial_lr"])
            print("Restored optimizer state from checkpoint.")
        else:
            resume_warmup = True
            print("Skipping optimizer state restore (incompatible optimizer).")
    elif checkpoint:
        resume_warmup = True
        print("No optimizer state in checkpoint; starting with fresh optimizer.")

    # print(f"DEBUG CHECK: Optimizer state size = {len(optimizer.state)} (0 = Fresh/Reset, >0 = Restored)")

    # Linear warmup + WSD (warmup, stable, decay) schedule with linear decay tail.
    # Schedule uses epoch progress so it is insensitive to batch size changes.
    total_epochs = max(0.0, float(args.epochs))
    steps_per_epoch = max(1, steps_per_epoch)
    warmup_pct = float(getattr(args, "warmup_pct", 0.02))
    warmup_pct = max(0.0, min(1.0, warmup_pct))
    warmup_epochs = total_epochs * warmup_pct
    warmup_epochs = max(0.0, min(total_epochs, warmup_epochs))
    min_lr_factor = float(getattr(args, "lr_floor", 0.01))
    min_lr_factor = max(0.0, min(1.0, min_lr_factor))
    decay_start_pct = float(getattr(args, "wsd_decay_start_pct", 0.8))
    decay_start_pct = max(0.0, min(1.0, decay_start_pct))
    decay_start_epoch = total_epochs * decay_start_pct
    decay_start_epoch = max(decay_start_epoch, warmup_epochs)
    decay_start_epoch = min(decay_start_epoch, total_epochs)
    decay_epochs = max(1e-8, total_epochs - decay_start_epoch)

    def base_lr_lambda(current_epoch: float):
        if warmup_epochs > 0 and current_epoch < warmup_epochs:
            return float(current_epoch) / float(warmup_epochs)
        if decay_start_epoch >= total_epochs or current_epoch < decay_start_epoch:
            return 1.0
        progress = float(current_epoch - decay_start_epoch) / float(decay_epochs)
        progress = max(0.0, min(1.0, progress))
        linear = 1.0 - progress
        return min_lr_factor + (1.0 - min_lr_factor) * linear

    lr_lambda = base_lr_lambda
    resume_start_epoch = float(start_epoch)
    if resume_warmup and resume_start_epoch > 0:
        remaining_epochs = max(0.0, total_epochs - resume_start_epoch)
        if remaining_epochs > 0:
            resume_warmup_epochs = remaining_epochs * 0.02
            min_warmup_epochs = 1.0 / float(steps_per_epoch)
            resume_warmup_epochs = max(min_warmup_epochs, resume_warmup_epochs)
            resume_warmup_epochs = min(resume_warmup_epochs, remaining_epochs)
            resume_warmup_start = resume_start_epoch
            resume_warmup_end = resume_start_epoch + resume_warmup_epochs
            resume_target = base_lr_lambda(resume_warmup_end)

            def lr_lambda(current_epoch: float):
                if current_epoch < resume_warmup_start:
                    return base_lr_lambda(current_epoch)
                if current_epoch < resume_warmup_end:
                    progress = float(current_epoch - resume_warmup_start) / float(
                        resume_warmup_epochs
                    )
                    return resume_target * progress
                return base_lr_lambda(current_epoch)

            print(
                f"Applying resume LR warmup for {resume_warmup_epochs:.3f} epochs."
            )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Restore scheduler if resuming (unless we intentionally reset the schedule).
    scheduler_restored = False
    if checkpoint and "scheduler_state" in checkpoint and not resume_warmup:
        scheduler_state = checkpoint["scheduler_state"]
        last_epoch = (
            scheduler_state.get("last_epoch")
            if isinstance(scheduler_state, dict)
            else None
        )
        if isinstance(last_epoch, (int, float)) and abs(
            float(last_epoch) - float(start_epoch)
        ) > 1.0:
            print(
                "Skipping scheduler state restore; using epoch-based schedule from args."
            )
        else:
            scheduler.load_state_dict(scheduler_state)
            scheduler_restored = True
            print("Restored scheduler state from checkpoint.")
    elif checkpoint and "scheduler_state" in checkpoint and resume_warmup:
        print(
            "Skipping scheduler state restore; recomputing schedule from args."
        )
    if not scheduler_restored and start_epoch > 0:
        if resume_warmup:
            resume_base_factor = base_lr_lambda(resume_start_epoch)
            for base_lr, group in zip(scheduler.base_lrs, optimizer.param_groups):
                group["lr"] = base_lr * resume_base_factor
        else:
            scheduler.step(float(start_epoch))

    # Compile a specific reference for training execution only.
    # We do this AFTER optimizer loading to ensure parameter consistency.
    # Guard on CUDA because training is CUDA-only and compilation is backend-specific.
    if hasattr(torch, "compile") and device.type == "cuda":
        print("Compiling model for training speedup...")
        training_model = torch.compile(model)
    else:
        training_model = model


    augmentor = getattr(dataloader, "augmentor", None)

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        if augmentor is not None:
            augmentor.set_epoch(epoch)

        # Run Training
        step = train_one_epoch(
            model=training_model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
            gradient_accumulation_steps=getattr(args, "gradient_accumulation_steps", 1),
            start_step=step,
            log_file=log_file,
            scheduler=scheduler,
            epoch=epoch,
            steps_per_epoch=steps_per_epoch,
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
