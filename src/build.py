"""Build model and data for training and evaluation.

Contains: build_model_and_data, checkpoint loading, and related utilities.
"""

import argparse
import hashlib
import json
import os
import random
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import distributed_training as distributed

from common import (
    build_augmentor,
    ARCExampleDataset,
    CombinedAugmentor,
    create_dataloader,
    load_challenges,
    recommended_num_workers,
    restore_rng_state,
    resolve_device,
    scan_nvarc_dir,
    set_seed,
    normalize_augmentation_sampling_mode,
    normalize_attention_pattern,
    sequence_format_for_attention,
)
from tinytransformer import (
    TinyTransformer,
    TinyTransformerConfig,
    normalize_flash_attention_version,
)
from variations import VariationTrainingDataset, create_variation_dataloader
from research_options import RESEARCH_DEFAULTS, research_parameter_shapes


def checkpoint_trained_capped_demos(checkpoint: Optional[Dict[str, Any]]) -> bool:
    """Whether a checkpoint was trained on balanced capped demo contexts.

    Recorded by maybe_save_model ("demo_selection"); family-sampled checkpoints
    (the loop-composition-v3 recipe) always used them.
    """
    if not checkpoint:
        return False
    if (checkpoint.get("demo_selection") or {}).get("training") == "balanced_capped":
        return True
    return ((checkpoint.get("task_variations") or {}).get("sampling") or {}).get("unit") == "family"


def _resolve_inference_max_demos(args: argparse.Namespace, checkpoint: Optional[Dict[str, Any]]) -> Optional[int]:
    """Decode-context demo cap: explicit, else 6 for capped-context checkpoints, else uncapped (None)."""
    cap = getattr(args, "inference_max_demos", None)
    if cap is not None:
        return int(cap)
    from demo_schedule import MAX_CONTEXT_DEMOS
    return MAX_CONTEXT_DEMOS if checkpoint_trained_capped_demos(checkpoint) else None


# =============================================================================
# Checkpoint Loading
# =============================================================================

def _validate_checkpoint_format(checkpoint: Dict[str, Any]) -> None:
    pattern = checkpoint.get("config", {}).get("attention_pattern", "dense")
    expected = sequence_format_for_attention(pattern)
    # These historical formats used identical serialization/roles, with the
    # original combined attention. Never reinterpret them as two lookup.
    legacy = {"all_demo_pair_group_target_roles_v2", "half_or_more_pair_group_target_roles_v1"}
    actual = checkpoint.get("sequence_format")
    if actual in legacy and pattern == "pair_group":
        if checkpoint["config"].get("split_target_demo_attention", False):
            raise ValueError("Legacy group format cannot use two-lookup attention")
        return
    if actual != expected:
        raise ValueError(
            "Checkpoint sequence format is incompatible; fresh training is required. Expected "
            f"sequence_format={expected!r}."
        )

def load_checkpoint(checkpoint_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Load a model checkpoint from disk."""
    if checkpoint_path is None:
        return None
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint {checkpoint_path} must be a checkpoint dictionary.")
    _validate_checkpoint_format(checkpoint)
    print(f"Loaded checkpoint from {checkpoint_path}")
    return checkpoint


def _alias_eval_checkpoint_tasks(
    checkpoint: Dict[str, Any], data_path: Path, prefix: str,
    selected_task_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Use trained namespaced task rows for native evaluation IDs, without changing weights."""
    if not isinstance(prefix, str) or not prefix or not prefix.endswith("/"):
        raise ValueError("checkpoint_task_id_prefix must be a nonempty namespace ending in '/'.")
    task_ids = set(load_challenges(data_path))
    if selected_task_ids is not None:
        task_ids.intersection_update(selected_task_ids)
    saved_tasks = set(checkpoint["task_ids"])
    saved_embeddings = set(checkpoint.get("embedding_task_ids", checkpoint["task_ids"]))
    missing = sorted(task_id for task_id in task_ids
                     if prefix + task_id not in saved_tasks or prefix + task_id not in saved_embeddings)
    if missing:
        raise ValueError(f"Evaluation tasks have no trained {prefix!r} checkpoint rows: {missing[:5]}")
    aliases = {prefix + task_id: task_id for task_id in task_ids}
    adapted = dict(checkpoint)
    for key in ("task_ids", "embedding_task_ids"):
        if key not in checkpoint:
            continue
        renamed = [aliases.get(task_id, task_id) for task_id in checkpoint[key]]
        if len(renamed) != len(set(renamed)):
            raise ValueError(f"checkpoint_task_id_prefix creates duplicate {key} identities")
        adapted[key] = renamed
    print(f"Matched {len(task_ids)} evaluation tasks to trained checkpoint rows under {prefix}")
    return adapted


def build_checkpoint_config(checkpoint: Dict[str, Any]) -> TinyTransformerConfig:
    config_data = checkpoint["config"]
    # Accept the saved R1 treatment, but never silently reinterpret a removed
    # ablation (or an old implicit 10000 role base) as the simplified model.
    version = config_data.get("architecture_version")
    if version not in (None, 2):
        raise ValueError(f"Unsupported checkpoint architecture_version: {version!r}.")
    retired_settings = {
        "residual_init": "xavier",
        "qk_norm": "none",
        "branch_norm": "none",
        "rope_axes": "3d",
        "grid_role_embeddings": False,
        "grid_boundary_positions": "row_end",
        # Intermediate-grid chains (loop-composition-v3 only) are not part of this model.
        "ordered_grid_positions": False,
        "predict_intermediate_grids": False,
    }
    for key, expected in retired_settings.items():
        if config_data.get(key, expected) != expected:
            raise ValueError(f"Checkpoint uses removed ablation {key}={config_data[key]!r}; "
                             f"the model now uses the R1 setting {expected!r}.")
    role_base = config_data.get("rope_role_base", 1000.0 if version == 2 else 10000.0)
    if role_base != 1000.0:
        raise ValueError(f"Checkpoint rope_role_base={role_base!r} is incompatible with "
                         "the fixed R1 role base of 1000.")
    valid_keys = {field.name for field in fields(TinyTransformerConfig) if field.init}
    config_kwargs = {
        key: value for key, value in config_data.items() if key in valid_keys
    }
    # Pre-restoration single-pair checkpoints have no dihedral parameters.
    # Keep their architecture and optimizer topology intact on resume/eval.
    config_kwargs.setdefault("dihedral_embeddings", False)
    # Spatial frequencies remain configurable. Preserve old explicit-role
    # checkpoints whose omitted spatial base meant 10000.
    if version is None:
        config_kwargs.setdefault("rope_spatial_base", 10000.0)
    return TinyTransformerConfig(**config_kwargs)


def _maybe_override_flash_attention_version(
    config: TinyTransformerConfig,
    args: argparse.Namespace,
) -> None:
    if not hasattr(args, "FA"):
        return
    config.FA = normalize_flash_attention_version(getattr(args, "FA"))


def get_checkpoint_mode(args: argparse.Namespace) -> Optional[str]:
    mode = getattr(args, "checkpoint_mode", None)
    if mode is None:
        return None
    mode = str(mode).strip().lower()
    if mode not in {"resume", "finetune"}:
        raise ValueError("checkpoint_mode must be 'resume' or 'finetune'.")
    return mode


def should_extend_checkpoint_task_ids(args: argparse.Namespace) -> bool:
    mode = get_checkpoint_mode(args)
    if mode is not None:
        return mode == "finetune"
    return bool(getattr(args, "extend_checkpoint_task_ids", False))


def should_inherit_checkpoint_rearc_data_path(args: argparse.Namespace) -> bool:
    mode = get_checkpoint_mode(args)
    if mode is not None:
        return mode == "resume"
    return bool(getattr(args, "inherit_checkpoint_rearc_data_path", True))


def should_reset_training_state(args: argparse.Namespace) -> bool:
    mode = get_checkpoint_mode(args)
    if mode is not None:
        return mode == "finetune"
    return bool(getattr(args, "reset_training_state", False))


def _configure_eval_loop_iterations(config: TinyTransformerConfig, iterations: Optional[int]) -> TinyTransformerConfig:
    """Decode a looped checkpoint at another iteration count (evaluation only; paper row A7).

    Deep supervision trains every iteration's exit through the shared coda, so a
    checkpoint whose loop parameters do not depend on K can be read out at any K >= 2.
    Unset (None) or the trained K leaves the checkpoint's configuration unchanged.
    """
    if iterations is None or iterations == config.loop_iterations:
        return config
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 2 or config.loop_iterations < 2:
        raise ValueError("eval_loop_iterations needs a looped checkpoint and an integer >= 2.")
    # Exits only shape the training loss; decoding reads the final iteration.
    updated = replace(config, loop_iterations=iterations, loop_exit_iterations=None)
    if research_parameter_shapes(updated) != research_parameter_shapes(config):
        raise ValueError("eval_loop_iterations: this checkpoint has loop parameters that depend on the "
                         "iteration count (streams, dense mixing, iteration embeddings, gates or norms).")
    print(f"Evaluating the loop at {iterations} iterations (trained with {config.loop_iterations}).")
    return updated


def _configure_checkpoint_mtp(
    config: TinyTransformerConfig,
    args: argparse.Namespace,
    *,
    is_eval: bool,
) -> TinyTransformerConfig:
    # Evaluation always uses the saved architecture. Unspecified training
    # options also inherit it, so old resume specs remain single-head runs.
    if is_eval or getattr(args, "eval_only", False):
        return _configure_eval_loop_iterations(config, getattr(args, "eval_loop_iterations", None))
    overrides = {
        key: getattr(args, key)
        for key in ("n_predict", "mtp_weight", "task_embeddings", "dihedral_embeddings", "input_shortcuts", "rope_spatial_base", "attention_pattern", *RESEARCH_DEFAULTS)
        if getattr(args, key, None) is not None
    }
    updated = replace(config, **overrides)
    architecture_keys = ("task_embeddings", "dihedral_embeddings", "input_shortcuts", "rope_spatial_base", "attention_pattern")
    if any(getattr(updated, key) != getattr(config, key) for key in architecture_keys):
        raise ValueError("Checkpoint training must preserve task_embeddings, dihedral_embeddings, input_shortcuts, rope_spatial_base and attention_pattern; "
                         "these options configure fresh models only.")
    if (
        (updated.n_predict, updated.mtp_weight) != (config.n_predict, config.mtp_weight)
        and not should_reset_training_state(args)
    ):
        raise ValueError(
            "MTP settings differ from the checkpoint. Resume must preserve n_predict "
            "and mtp_weight; use checkpoint_mode='finetune' (fresh optimizer/schedule) "
            "to change them."
        )
    if any(getattr(updated, key) != getattr(config, key) for key in RESEARCH_DEFAULTS) and not should_reset_training_state(args):
        raise ValueError("Looping/NextLat settings differ from the checkpoint. Use checkpoint_mode='finetune' "
                         "with a fresh optimizer/schedule to change them; resume preserves saved settings.")
    return updated


def _resolve_num_workers(value: Optional[object], *, max_workers: int) -> int:
    if value is None:
        return recommended_num_workers(max_workers=max_workers)
    return max(0, int(value))


def _merge_task_ids(
    checkpoint_task_ids: Tuple[str, ...], *sources: Optional[Path]
) -> list[str]:
    task_ids = list(checkpoint_task_ids)
    seen = set(task_ids)
    for source in sources:
        if source is None:
            continue
        source_task_ids = (
            scan_nvarc_dir(source).keys()
            if Path(source).is_dir()
            else load_challenges(source).keys()
        )
        for task_id in source_task_ids:
            if task_id not in seen:
                task_ids.append(task_id)
                seen.add(task_id)
    return task_ids


def _resolve_checkpoint_path_arg(
    args: argparse.Namespace,
    checkpoint: Optional[Dict[str, Any]],
    key: str,
) -> Optional[Path]:
    path = getattr(args, key, None)
    if (
        path is None
        and checkpoint
        and should_inherit_checkpoint_rearc_data_path(args)
        and key in checkpoint
    ):
        path = checkpoint[key]
    return None if path is None else Path(path)


def _load_model_state(
    model: TinyTransformer,
    checkpoint: Dict[str, Any],
    current_embedding_task_ids: Sequence[str],
    *,
    allow_mtp_resize: bool = False,
    allow_research_changes: bool = False,
) -> None:
    state_dict = dict(checkpoint["model_state"])
    expected = model.state_dict()
    if "embedding_task_ids" not in checkpoint:
        raise ValueError(
            "Checkpoint is missing embedding_task_ids and is no longer supported."
        )
    checkpoint_embedding_row_ids = tuple(
        str(task_id) for task_id in checkpoint["embedding_task_ids"]
    )

    def _align_task_factor_weight() -> None:
        key = "task_factor_embedding.weight"
        saved_weight = state_dict.get(key)
        current_weight = expected.get(key)
        if saved_weight is None or current_weight is None:
            return
        if (
            saved_weight.ndim != 2
            or current_weight.ndim != 2
            or saved_weight.size(1) != current_weight.size(1)
        ):
            raise ValueError(
                "Checkpoint task factors are incompatible with the current model shape: "
                f"{tuple(saved_weight.shape)} vs {tuple(current_weight.shape)}."
            )
        if len(checkpoint_embedding_row_ids) != saved_weight.size(0):
            raise ValueError(
                "Checkpoint embedding_task_ids length does not match "
                f"{key}: {len(checkpoint_embedding_row_ids)} ids vs "
                f"{saved_weight.size(0)} rows."
            )
        current_row_ids = tuple(str(task_id) for task_id in current_embedding_task_ids)
        if checkpoint_embedding_row_ids == current_row_ids and saved_weight.shape == current_weight.shape:
            # The common evaluation/resume case needs only load_state_dict's
            # bulk copy, including a single CPU-to-GPU transfer for this table.
            copied = len(current_row_ids)
        else:
            checkpoint_index_by_task_id = {
                task_id: idx for idx, task_id in enumerate(checkpoint_embedding_row_ids)
            }
            current_indices, checkpoint_indices = [], []
            for current_idx, task_id in enumerate(current_row_ids):
                checkpoint_idx = checkpoint_index_by_task_id.get(task_id)
                if checkpoint_idx is not None:
                    current_indices.append(current_idx)
                    checkpoint_indices.append(checkpoint_idx)
            copied = len(current_indices)
            if copied:
                source_indices = torch.tensor(checkpoint_indices, device=saved_weight.device, dtype=torch.long)
                rows = saved_weight.index_select(0, source_indices).to(
                    device=current_weight.device, dtype=current_weight.dtype,
                )
                resized_weight = current_weight.clone()
                target_indices = torch.tensor(current_indices, device=current_weight.device, dtype=torch.long)
                resized_weight.index_copy_(0, target_indices, rows)
                state_dict[key] = resized_weight
            else:
                # Fresh tasks (e.g. Kaggle) keep their exact initialized factors.
                state_dict[key] = current_weight
        print(
            f"Loaded task factors by task id: "
            f"copied {copied}/{len(current_embedding_task_ids)} rows."
        )

    _align_task_factor_weight()
    saved_config = build_checkpoint_config(checkpoint)
    saved_count = saved_config.n_predict
    saved_heads = {f"lm_head_mtp.{idx}.weight" for idx in range(saved_count - 1)}
    actual_heads = {key for key in state_dict if key.startswith("lm_head_mtp.")}
    if actual_heads != saved_heads:
        raise ValueError("Checkpoint MTP heads do not match its saved n_predict.")
    if allow_mtp_resize:
        current_heads = {f"lm_head_mtp.{idx}.weight" for idx in range(model.config.n_predict - 1)}
        for key in saved_heads - current_heads:
            # Missing saved heads still fail, even when dropping a head.
            state_dict.pop(key)
        for key in current_heads - saved_heads:
            state_dict[key] = expected[key]
        if current_heads != saved_heads:
            print(f"Changed prediction heads from {saved_count} to {model.config.n_predict}; "
                  "retained existing heads and initialized added heads.")
    saved_research_shapes = research_parameter_shapes(saved_config)
    actual_research = {key for key in state_dict if key.startswith(("loop_", "next_latent_dynamics."))}
    if actual_research != set(saved_research_shapes):
        raise ValueError("Checkpoint looping/NextLat parameters do not match its saved configuration.")
    for key, shape in saved_research_shapes.items():
        if tuple(state_dict[key].shape) != shape:
            raise ValueError(f"Checkpoint research parameter has an invalid shape: {key}.")
    if allow_research_changes:
        current_research_shapes = research_parameter_shapes(model.config)
        for key in saved_research_shapes.keys() - current_research_shapes.keys():
            state_dict.pop(key)
        for key, shape in current_research_shapes.items():
            if key not in state_dict or tuple(state_dict[key].shape) != shape:
                state_dict[key] = expected[key]
    model.load_state_dict(state_dict)


# =============================================================================
# Build Model and Data
# =============================================================================

def build_model_and_data(
    args: argparse.Namespace,
    checkpoint: Optional[Dict[str, Any]] = None,
    reuse_dataset: Optional[ARCExampleDataset] = None,
    is_eval: bool = False,
    eval_splits: Optional[Sequence[str]] = None,
    build_dataloader: bool = True,
    eval_task_ids: Optional[Sequence[str]] = None,
) -> Tuple[
    TinyTransformer,
    ARCExampleDataset,
    Optional[torch.utils.data.DataLoader],
    torch.device,
    Path,
]:
    """Construct dataset, dataloader, and model for a given arg namespace.

    Shared by CLI entrypoints and notebooks so that training, evaluation,
    and inference can be orchestrated independently. When ``build_dataloader``
    is False, the returned dataloader slot is None. Evaluation indexes only
    present tasks (and ``eval_task_ids`` when provided), while retaining the
    checkpoint's embedding row order and model initialization shape.
    """
    set_seed(args.seed)
    device = resolve_device(getattr(args, "device", "cuda"))
    if device.type == "cuda" and distributed.active() and not is_eval:
        device = torch.device("cuda", torch.cuda.current_device())
    checkpoint = checkpoint if checkpoint is not None else load_checkpoint(args.checkpoint_path)
    if checkpoint is not None:
        _validate_checkpoint_format(checkpoint)
    selected_mode = getattr(args, "augmentation_sampling_mode", None)
    mode = normalize_augmentation_sampling_mode(
        selected_mode if selected_mode is not None else (checkpoint or {}).get("augmentation_sampling_mode")
    )

    data_path = args.data_path
    if data_path is None and checkpoint is not None:
        data_path = Path(checkpoint["data_path"])
    if data_path is None:
        raise ValueError("--data-path is required when training without a checkpoint.")
    data_path = Path(data_path)
    task_id_prefix = getattr(args, "checkpoint_task_id_prefix", None)
    if task_id_prefix is not None:
        if not is_eval or checkpoint is None:
            raise ValueError("checkpoint_task_id_prefix requires checkpoint evaluation")
        checkpoint = _alias_eval_checkpoint_tasks(checkpoint, data_path, task_id_prefix, eval_task_ids)
    rearc_data_path = _resolve_checkpoint_path_arg(args, checkpoint, "rearc_data_path")
    nvarc_training_path = _resolve_checkpoint_path_arg(
        args, checkpoint, "nvarc_training_path"
    )
    nvarc_full_path = _resolve_checkpoint_path_arg(args, checkpoint, "nvarc_full_path")

    config = build_checkpoint_config(checkpoint) if checkpoint else None
    if config is not None:
        config = _configure_checkpoint_mtp(config, args, is_eval=is_eval)
    attention_pattern = config.attention_pattern if config is not None else normalize_attention_pattern(
        getattr(args, "attention_pattern", None)
    )
    task_whitelist = checkpoint["task_ids"] if checkpoint is not None else None
    embedding_task_whitelist = (
        checkpoint.get("embedding_task_ids") if checkpoint is not None else None
    )
    extend_checkpoint_task_ids = bool(
        checkpoint is not None and should_extend_checkpoint_task_ids(args)
    )
    if extend_checkpoint_task_ids:
        task_whitelist = _merge_task_ids(
            tuple(task_whitelist),
            data_path,
            None if is_eval else rearc_data_path,
            None if is_eval else nvarc_training_path,
            None if is_eval else nvarc_full_path,
        )

    if reuse_dataset is not None:
        print("Reusing existing dataset from RAM (skipping 3D pre-computation).")
        dataset = reuse_dataset
        if dataset.attention_pattern != attention_pattern:
            raise ValueError("Reused dataset attention_pattern does not match the model.")
        dataset.augmentation_sampling_mode = mode
    else:
        if is_eval:
            splits = tuple(eval_splits) if eval_splits is not None else ("test",)
            if not splits:
                raise ValueError("Evaluation requires at least one split.")
        else:
            splits = ("train",)
        training_demo_schedule = getattr(args, "training_demo_schedule", "all")
        inference_demo_policy = getattr(args, "inference_demo_policy", "all") if is_eval else "all"
        max_training_demos = getattr(args, "max_training_demos", None)
        inference_max_demos = (_resolve_inference_max_demos(args, checkpoint) if is_eval else None)
        if is_eval:
            # Train-split evaluation still excludes its target, but the two-demo
            # inference policy applies to both train and test prompts.
            training_demo_schedule = "two" if inference_demo_policy == "two" else "all"
            if training_demo_schedule != "two":
                max_training_demos = None
        variation_index_path = getattr(args, "variation_index_path", None)
        if not is_eval and variation_index_path is not None:
            dataset = VariationTrainingDataset(
                json_path=data_path,
                index_path=Path(variation_index_path),
                seed=int(args.seed),
                max_augments=int(getattr(args, "max_augments", 1) or 1),
                augmentation_sampling_mode=mode,
                attention_pattern=attention_pattern,
                training_demo_schedule=training_demo_schedule,
                max_training_demos=max_training_demos,
                pair_identity_vertices=config.pair_identity_vertices if config else getattr(args, "pair_identity_vertices", 0),
            )
            if checkpoint is not None:
                if get_checkpoint_mode(args) != "finetune" and checkpoint.get("task_variations") != dataset.checkpoint_metadata(
                    enable_aug=bool(getattr(args, "enable_aug", False)),
                ):
                    raise ValueError("Checkpoint task-variation dataset does not match the current run")
        else:
            dataset = ARCExampleDataset(
                json_path=data_path,
                rearc_json_path=None if is_eval else rearc_data_path,
                nvarc_training_dir=None if is_eval else nvarc_training_path,
                nvarc_full_dir=None if is_eval else nvarc_full_path,
                rearc_pairs_per_epoch=int(getattr(args, "rearc_pairs_per_epoch", 10)),
                nvarc_pairs_per_epoch=int(getattr(args, "nvarc_pairs_per_epoch", 10)),
                nvarc_puzzles_per_epoch=getattr(args, "nvarc_puzzles_per_epoch", None),
                splits=splits,
                include_outputs=True,
                max_seq_len=None,
                attention_pattern=attention_pattern,
                task_whitelist=task_whitelist,
                embedding_task_whitelist=embedding_task_whitelist,
                augmentation_sampling_mode=mode,
                allow_missing_tasks=bool(
                    getattr(args, "allow_missing_tasks", False)
                    or (is_eval and task_whitelist is not None)
                    or extend_checkpoint_task_ids
                ),
                seed=int(args.seed),
                max_augments=int(getattr(args, "max_augments", 1) or 1),
                index_present_tasks_only=is_eval,
                selected_task_ids=eval_task_ids if is_eval else None,
                training_demo_schedule=training_demo_schedule,
                inference_demo_policy=inference_demo_policy,
                max_training_demos=max_training_demos,
                inference_max_demos=inference_max_demos,
                pair_identity_vertices=config.pair_identity_vertices if config else getattr(args, "pair_identity_vertices", 0),
                scale_augmentation=not is_eval and bool(getattr(args, "scale_augmentation", False)),
                scale_max_specs=int(getattr(args, "scale_max_specs", 12)),
                scale_anisotropic=bool(getattr(args, "scale_anisotropic", True)),
                rearc_max_demos=int(getattr(args, "rearc_max_demos", 4)),
            )
            # Decode-time scale views (evaluate._build_inference_attempts); off by default.
            dataset.inference_scale_views = is_eval and bool(getattr(args, "inference_scale_views", False))

    manifest_path = getattr(args, "dataset_manifest_path", None)
    if not is_eval and manifest_path is not None:
        metadata = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if metadata.get("training_sha256") != hashlib.sha256(data_path.read_bytes()).hexdigest():
            raise ValueError("Training data does not match its dataset manifest")
        metadata.update(
            seed=int(args.seed), max_augments=int(getattr(args, "max_augments", 1) or 1),
            enable_aug=bool(getattr(args, "enable_aug", False)),
        )
        if checkpoint is not None and get_checkpoint_mode(args) != "finetune" and checkpoint.get("task_data") != metadata:
            raise ValueError("Checkpoint original-task dataset does not match the current run")
        dataset.training_data_metadata = metadata

    use_aug = bool(getattr(args, "enable_aug", False))
    augmentor = None
    if not is_eval and getattr(dataset, "scale_augmentation", False):
        # On by default: it needs the augmented regular-task stream, and only ever scales regular
        # ARC sequences (ReARC/NVARC examples are never scaled; see training_scale_factor).
        if not use_aug or getattr(dataset, "has_variation_sampling", False):
            dataset.scale_augmentation = False
            print("Scale augmentation off: it needs enable_aug and is not defined for variation sampling.")
        elif (checkpoint is not None and get_checkpoint_mode(args) == "resume"
              and "scale_augmentation" not in checkpoint and not getattr(args, "scale_augmentation_explicit", True)):
            dataset.scale_augmentation = False
            print("Scale augmentation off: resuming a checkpoint trained without it "
                  "(set scale_augmentation in the spec to change the recipe).")
        else:
            mixed = (getattr(dataset, "has_rearc_train_sampling", False)
                     or getattr(dataset, "has_nvarc_train_sampling", False))
            print(f"Scale augmentation: up to {dataset.scale_max_specs} block-upscale specs per sequence "
                  f"(identity included), anisotropic={dataset.scale_anisotropic}"
                  f"{'; regular ARC tasks only (ReARC/NVARC unscaled)' if mixed else ''}.")

    if (not is_eval and mode == "context_palette" and getattr(dataset, "has_variation_sampling", False)
            and dataset.variation_index.get("sampling", {}).get("unit") != "family"):
        raise ValueError("context_palette supports regular ARC, ReARC and family-sampled tasks only")
    # Variation datasets build their own augmentor in create_variation_dataloader.
    if (not is_eval and use_aug and mode == "context_palette"
            and not getattr(dataset, "has_variation_sampling", False)):
        from context_palette import ContextPaletteAugmentor
        augmentor = ContextPaletteAugmentor(
            dataset, seed=args.seed, max_augments=int(args.max_augments))
        dataset.augmentor = augmentor
    elif not is_eval and use_aug and not getattr(dataset, "has_variation_sampling", False):
        max_augments = int(getattr(args, "max_augments", 0) or 0)
        regular_examples = list(dataset.examples)
        regular_augmentor = None
        if regular_examples:
            regular_augmentor = build_augmentor(
                regular_examples,
                dataset.task_colors,
                max_augments=max_augments,
                seed=args.seed,
                augmentation_sampling_mode=mode,
                task_output_only_colors=dataset.task_output_only_colors,
                task_inputs={task_id: [pair["input"] for pair in
                             dataset.pairs_by_task.get(task_id, []) + dataset.test_pairs_by_task.get(task_id, [])]
                             for task_id in {example.task_id for example in regular_examples}}
                            if mode == "single_grid_prefix" else None,
            )

        if regular_augmentor is not None or dataset.has_lazy_train_sampling:
            augmentor = CombinedAugmentor(
                regular_augmentor,
                rearc_seed=int(args.seed),
                task_colors=dataset.task_colors,
                augmentation_sampling_mode=mode,
                task_output_only_colors=dataset.task_output_only_colors,
                max_augments=max_augments,
                dataset=dataset,
            )
            dataset.augmentor = augmentor

    dataloader = None
    if build_dataloader:
        # We always recreate the dataloader because batch_size might have changed in args.
        collate_augment_selector = None
        if augmentor is not None:
            collate_augment_selector = augmentor.select_for_example
        # Evaluation uses deterministic length bucketing; training uses random batching.
        use_length_bucketing = bool(is_eval or getattr(args, "eval_only", False))
        num_workers = _resolve_num_workers(
            getattr(args, "num_workers", None),
            max_workers=4 if is_eval else max(1, 8 // distributed.world_size()),
        )
        # DDP: each rank takes its shard of one deterministic global batch.
        global_batch_size = args.batch_size * (1 if is_eval else distributed.world_size())
        if getattr(dataset, "has_variation_sampling", False):
            dataloader = create_variation_dataloader(
                dataset,
                batch_size=global_batch_size,
                seed=int(args.seed),
                enable_aug=use_aug,
                max_augments=int(getattr(args, "max_augments", 1) or 1),
                num_workers=num_workers,
                pin_memory=getattr(args, "pin_memory", None),
                persistent_workers=getattr(args, "persistent_workers", None),
                prefetch_factor=getattr(args, "prefetch_factor", None),
                gradient_accumulation_steps=int(getattr(args, "gradient_accumulation_steps", 1)),
                num_epochs=int(getattr(args, "epochs", 1)),
            )
        else:
            dataloader = create_dataloader(
                dataset=dataset,
                batch_size=global_batch_size,
                shuffle=not (is_eval or getattr(args, "eval_only", False)),
                augment_selector=collate_augment_selector,
                use_length_bucketing=use_length_bucketing,
                seed=int(args.seed),
                augmentor=augmentor,
                num_workers=num_workers,
                pin_memory=getattr(args, "pin_memory", None),
                persistent_workers=getattr(args, "persistent_workers", None),
                prefetch_factor=getattr(args, "prefetch_factor", None),
                num_epochs=int(getattr(args, "epochs", 1)),
                gradient_accumulation_steps=int(
                    getattr(args, "gradient_accumulation_steps", 1)
                ),
            )
        if augmentor is not None:
            dataloader.augmentor = augmentor

    if config is None:
        num_examples = max(1, dataset.num_examples)
        d_model = getattr(args, "d_model", 128)
        n_heads = getattr(args, "n_heads", 4)
        d_ff = getattr(args, "d_ff", 512)
        n_layers = getattr(args, "n_layers", 4)
        dropout = getattr(args, "dropout", 0.1)
        embedding_dropout = getattr(args, "embedding_dropout", 0.0)
        attention_dropout = getattr(args, "attention_dropout", None)
        num_task_embedding_latents = getattr(args, "num_task_embedding_latents", None)
        flash_attention_version = normalize_flash_attention_version(
            getattr(args, "FA", 2)
        )
        if num_task_embedding_latents is None:
            num_task_embedding_latents = min(3072, num_examples)
        if attention_dropout is None:
            attention_dropout = dropout
        config = TinyTransformerConfig(
            attention_pattern=attention_pattern,
            split_target_demo_attention=attention_pattern == "pair_group",
            pair_identity_vertices=getattr(args, "pair_identity_vertices", 0),
            pair_identity_strength=getattr(args, "pair_identity_strength", 0.25),
            max_seq_len=dataset.max_seq_len,
            num_examples=num_examples,
            num_task_embedding_latents=num_task_embedding_latents,
            task_embeddings=True if getattr(args, "task_embeddings", None) is None else args.task_embeddings,
            dihedral_embeddings=False if getattr(args, "dihedral_embeddings", None) is None else args.dihedral_embeddings,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_layers,
            dropout=dropout,
            embedding_dropout=embedding_dropout,
            attention_dropout=attention_dropout,
            FA=flash_attention_version,
            n_predict=4 if getattr(args, "n_predict", None) is None else args.n_predict,
            mtp_weight=0.3 if getattr(args, "mtp_weight", None) is None else args.mtp_weight,
            input_shortcuts=getattr(args, "input_shortcuts", None) or "learned",
            rope_spatial_base=1000.0 if getattr(args, "rope_spatial_base", None) is None else args.rope_spatial_base,
            **{key: default if getattr(args, key, None) is None else getattr(args, key)
               for key, default in RESEARCH_DEFAULTS.items()},
        )
    if config is not None:
        _maybe_override_flash_attention_version(config, args)
        if config.attention_pattern == "pair_group":
            # Role RoPE has no sequence-length parameters; new tasks may have
            # more demonstrations than the tasks used to train the checkpoint.
            config.max_seq_len = max(config.max_seq_len, dataset.max_seq_len)

    if dataset.num_examples != config.num_examples:
        if extend_checkpoint_task_ids and dataset.num_examples > config.num_examples:
            config.num_examples = dataset.num_examples
        else:
            raise ValueError(
                "Dataset task-embedding count mismatch: "
                f"checkpoint/model expects {config.num_examples} unique task embeddings but the provided dataset "
                f"currently exposes {dataset.num_examples}. Please ensure the dataset/task whitelist matches."
            )
    model = TinyTransformer(config).to(device)
    model.stable_cross_entropy = bool(getattr(args, "stable_cross_entropy", False))
    if config.loop_iterations > 1:
        print(f"Depth loop: layers [{config.loop_start_layer}, {config.loop_end_layer}) "
              f"x {config.loop_iterations}; {config.n_layers} physical blocks, "
              f"{model.cache_layer_count} logical applications / independent KV caches.")
        extensions = {key: getattr(config, key) for key in (
            "loop_dense_mixing", "loop_iteration_embedding", "loop_gate", "loop_gate_init",
            "loop_ffn", "loop_iteration_norms", "loop_streams", "loop_stream_gate")}
        print("Loop extensions: " + ", ".join(f"{key}={value}" for key, value in extensions.items()))
    if config.next_latent_prediction:
        print(f"NextLat: horizon={config.next_latent_horizon} recursive causal output transitions; "
              f"Smooth L1 weight={config.next_latent_regression_weight}, "
              f"KL weight={config.next_latent_kl_weight}; "
              f"MTP heads={config.n_predict - 1}.")

    if checkpoint is None and isinstance(dataset, VariationTrainingDataset):
        dataset.initialize_task_factors(model.task_factor_embedding)

    if checkpoint:
        _load_model_state(
            model, checkpoint, dataset.embedding_task_ids,
            allow_mtp_resize=not is_eval and should_reset_training_state(args),
            allow_research_changes=not is_eval and should_reset_training_state(args),
        )
        if not should_reset_training_state(args):
            states = checkpoint.get("rng_state_by_rank") if not is_eval else None
            # Opt-in: resume across a different DDP world size (the d1024 e270 -> e440 continuation).
            reseed_ranks = (states is not None and len(states) != distributed.world_size()
                            and os.environ.get("MDLARC_RESUME_RESEED_RANKS") == "1")
            if states is not None and len(states) != distributed.world_size() and not reseed_ranks:
                raise ValueError("Resume must preserve the checkpoint's DDP world size; "
                                 "use checkpoint_mode='finetune' to change it "
                                 "(or MDLARC_RESUME_RESEED_RANKS=1 to re-derive per-rank RNG streams)")
            if reseed_ranks:
                # Model, optimizer, scheduler and epoch resume exactly; only per-rank RNG streams are re-derived.
                import numpy as np
                seed = 1_000_003 * (int(checkpoint.get("epoch", 0)) + 1) + distributed.rank()
                random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)
                print(f"Resume across DDP world size {len(states)} -> {distributed.world_size()}: "
                      f"rank {distributed.rank()} RNG reseeded ({seed}).", flush=True)
            else:
                rng_state = states[distributed.rank()] if states is not None else checkpoint.get("rng_state")
                if isinstance(rng_state, dict):
                    restore_rng_state(rng_state, device)

    # Stash checkpoint for downstream consumers (e.g., so train_model can restore the optimizer).
    model._loaded_checkpoint = checkpoint

    return model, dataset, dataloader, device, data_path
