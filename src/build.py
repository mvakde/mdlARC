"""Build model and data for training and evaluation.

Contains: build_model_and_data, checkpoint loading, and related utilities.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from common import (
    build_augmentor,
    ARCExampleDataset,
    MAX_SEQ_LEN,
    create_dataloader,
    restore_rng_state,
    resolve_device,
    set_seed,
)
from tinytransformer import TinyTransformer, TinyTransformerConfig


# =============================================================================
# Checkpoint Loading
# =============================================================================

def load_checkpoint(checkpoint_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Load a model checkpoint from disk."""
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
    """Extract the number of examples from checkpoint metadata."""
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


# =============================================================================
# Build Model and Data
# =============================================================================

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
    checkpoint = checkpoint if checkpoint is not None else load_checkpoint(args.checkpoint_path)

    data_path = args.data_path
    if data_path is None:
        if checkpoint and "data_path" in checkpoint:
            data_path = Path(checkpoint["data_path"])
        else:
            raise ValueError(
                "--data-path is required when loading checkpoints that do not encode "
                "their source dataset. Please re-run with the same dataset used for training."
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
        restore_rng_state(checkpoint.get("rng_state"), device)

    # Stash checkpoint for downstream consumers (e.g., so train_model can restore the optimizer).
    model._loaded_checkpoint = checkpoint

    return model, dataset, dataloader, device, data_path
