"""Kaggle-specific input checks and checkpoint adaptation (no remote services)."""

from argparse import Namespace
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch


COMPETITION = "arc-prize-2026-arc-agi-2"
FINAL_CHECKPOINT_SIZE = 4364936923
FINAL_CHECKPOINT_SHA256 = "d5912ada04df6010369a59a7c5b6043261411825f95d5f3c304ad4461068a0dd"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint_file(path: Path, identity: dict) -> str:
    """Verify trusted pickle bytes before deserializing any checkpoint object."""
    if path.stat().st_size != identity["size"]:
        raise ValueError("Checkpoint size differs from the selected checkpoint profile.")
    observed = sha256_file(path)
    if observed != identity["sha256"]:
        raise ValueError("Checkpoint SHA-256 differs from the selected checkpoint profile.")
    return observed


def verify_checkpoint_metadata(checkpoint: dict, identity: dict) -> None:
    if checkpoint.get("epoch") != identity["epoch"]:
        raise ValueError(f"Expected pretraining epoch {identity['epoch']}.")
    if checkpoint.get("augmentation_sampling_mode") != identity["augmentation_sampling_mode"]:
        raise ValueError("Unexpected augmentation format for this checkpoint.")
    for key, expected in identity["config"].items():
        if checkpoint["config"].get(key) != expected:
            raise ValueError(f"Unexpected checkpoint configuration: {key}.")


def find_checkpoint(root: Path, selected=None) -> Path:
    if selected is not None:
        path = Path(selected)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    names = {"pretrain__tiny.pt", "tiny.pt"}
    matches = sorted(p for p in root.rglob("*.pt") if p.name in names)
    if len(matches) != 1:
        raise ValueError(
            "Attach the final checkpoint dataset and set CHECKPOINT_PATH to its exact path. "
            f"Found {len(matches)} candidates: {[str(p) for p in matches]}"
        )
    return matches[0]


def find_test_challenges(root: Path, selected=None) -> Path:
    names = {"arc-agi_test_challenges.json", "arc-agi_test-challenges.json"}
    if selected is not None:
        path = Path(selected)
        if not path.is_file():
            raise FileNotFoundError(f"Test challenges not found: {path}")
        # Only use the competition mount, whose contents Kaggle replaces at scoring.
        if path.name not in names or COMPETITION not in path.parts:
            raise ValueError("TEST_CHALLENGES must be the competition's test challenges file.")
        return path
    matches = sorted(p for p in root.rglob("*.json")
                     if p.name in names and COMPETITION in p.parts)
    if len(matches) != 1:
        raise ValueError(
            "Attach the ARC Prize 2026 ARC-AGI-2 competition data. "
            f"Found {len(matches)} test files: {[str(p) for p in matches]}"
        )
    return matches[0]


def verify_dataset_file(path: Path, identity: dict) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Validation file not found: {path}")
    if path.stat().st_size != identity["size"] or sha256_file(path) != identity["sha256"]:
        raise ValueError(f"Validation file differs from the pinned holdout: {path.name}")
    return path


def find_validation_challenges(root: Path, selected, identity: dict) -> Path:
    name = identity["challenges"]["filename"]
    matches = [Path(selected)] if selected is not None else sorted(Path(root).rglob(name))
    if len(matches) != 1 or matches[0].name != name:
        raise ValueError("Attach the selected private validation dataset; select its challenges file unambiguously.")
    return verify_dataset_file(matches[0], identity["challenges"])


def score_validation(predictions: dict, challenges: dict, solutions: dict) -> dict:
    """Post-generation scoring only; never adjusts predictions or model weights."""
    validate_submission(predictions, challenges)
    top1, top2, queries1, queries2, queries = 0.0, 0.0, 0, 0, 0
    per_task = {}
    for task_id, task in challenges.items():
        answers = solutions.get(task_id)
        if not isinstance(answers, list) or len(answers) != len(task["test"]) or not all(valid_grid(x) for x in answers):
            raise ValueError(f"{task_id}: invalid validation solutions or query count.")
        first = sum(guess["attempt_1"] == answer for guess, answer in zip(predictions[task_id], answers))
        either = sum(guess["attempt_1"] == answer or guess["attempt_2"] == answer
                     for guess, answer in zip(predictions[task_id], answers))
        count = len(answers)
        top1 += first / count
        top2 += either / count
        queries1 += first
        queries2 += either
        queries += count
        per_task[task_id] = {"queries": count, "top1_correct": first, "top2_correct": either}
    tasks = len(challenges)
    return {"tasks": tasks, "queries": queries,
            "metric": "Average query correctness within each task, then average across tasks.",
            "top1": {"score": top1, "maximum": tasks, "percent": 100 * top1 / tasks,
                     "correct_queries": queries1, "query_percent": 100 * queries1 / queries},
            "top2": {"score": top2, "maximum": tasks, "percent": 100 * top2 / tasks,
                     "correct_queries": queries2, "query_percent": 100 * queries2 / queries},
            "per_task": per_task}


def score_validation_groups(predictions, challenges, solutions, groups):
    """Keep original and synthetic accuracy distinct, including on smoke subsets."""
    ids = [task_id for group in groups.values() for task_id in group]
    if len(ids) != len(set(ids)) or not set(challenges) <= set(ids):
        raise ValueError("Validation groups must be disjoint and cover every decoded task.")
    result = {"combined_timing_workload": score_validation(predictions, challenges, solutions)}
    for name, group in groups.items():
        subset = {task_id: challenges[task_id] for task_id in group if task_id in challenges}
        result[name] = (score_validation({key: predictions[key] for key in subset}, subset, solutions)
                        if subset else {"tasks": 0, "queries": 0, "status": "No tasks from this group decoded."})
    return result


def valid_grid(grid) -> bool:
    return (
        isinstance(grid, list) and 1 <= len(grid) <= 30
        and isinstance(grid[0], list) and 1 <= len(grid[0]) <= 30
        and all(isinstance(row, list) and len(row) == len(grid[0])
                and all(type(color) is int and 0 <= color <= 9 for color in row)
                for row in grid)
    )


def prepare_challenges(source: Path, destination: Path, *, smoke_test=False,
                       smoke_task_count=2, smoke_strategy="first"):
    tasks = json.loads(Path(source).read_text())
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("Challenges must be a nonempty task dictionary.")
    cleaned = {}
    for task_id, task in tasks.items():
        if not isinstance(task_id, str) or not isinstance(task, dict):
            raise ValueError("Invalid task id or task object.")
        train_pairs, test_pairs = task.get("train"), task.get("test")
        if not isinstance(train_pairs, list) or not train_pairs:
            raise ValueError(f"{task_id}: missing train examples")
        if not isinstance(test_pairs, list) or not test_pairs:
            raise ValueError(f"{task_id}: missing test inputs")
        for pair in train_pairs:
            if not isinstance(pair, dict) or not all(valid_grid(pair.get(k)) for k in ("input", "output")):
                raise ValueError(f"{task_id}: invalid train pair")
        for pair in test_pairs:
            if not isinstance(pair, dict) or not valid_grid(pair.get("input")):
                raise ValueError(f"{task_id}: invalid test input")
        cleaned[task_id] = {
            "train": [{"input": p["input"], "output": p["output"]} for p in train_pairs],
            # Strip test outputs even if an interactive placeholder includes them.
            "test": [{"input": p["input"]} for p in test_pairs],
        }
    if smoke_test:
        if type(smoke_task_count) is not int or smoke_task_count < 1:
            raise ValueError("Smoke task count must be positive.")
        if smoke_strategy not in {"first", "largest"}:
            raise ValueError("Unknown smoke task selection strategy.")
        def cost(task_id):
            task = cleaned[task_id]
            train_size = sum(sum(len(row) + 1 for row in pair[key]) + 1
                             for pair in task["train"] for key in ("input", "output"))
            query_size = max(sum(len(row) + 1 for row in pair["input"]) for pair in task["test"])
            return (-train_size - query_size, task_id)
        ids = sorted(cleaned, key=cost if smoke_strategy == "largest" else None)[:smoke_task_count]
        cleaned = {key: cleaned[key] for key in ids}
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    Path(destination).write_text(json.dumps(cleaned))
    return cleaned


def adapt_checkpoint(checkpoint: dict) -> dict:
    """Retain shared weights, reset task factors, and discard pretraining state.

    build.build_model_and_data initializes fresh factors for the runtime tasks.
    An empty saved factor table lets its standard task-id alignment retain those
    initialized rows. The 3072-wide shared latent tables must never be resized.
    """
    from build import _validate_checkpoint_format, build_checkpoint_config

    _validate_checkpoint_format(checkpoint)
    build_checkpoint_config(checkpoint)
    state = dict(checkpoint["model_state"])
    factors = state["task_factor_embedding.weight"]
    saved_ids = checkpoint.get("embedding_task_ids")
    if not isinstance(saved_ids, (list, tuple)) or len(saved_ids) != factors.shape[0]:
        raise ValueError("Checkpoint task-factor rows do not match embedding_task_ids.")
    if factors.ndim != 2 or factors.shape[1] != checkpoint["config"]["num_task_embedding_latents"]:
        raise ValueError("Checkpoint task-factor latent width is inconsistent.")
    state["task_factor_embedding.weight"] = factors.new_empty((0, factors.shape[1]))
    config = dict(checkpoint["config"])
    # TinyTransformerConfig requires a positive count before runtime tasks extend it.
    config["num_examples"] = 1
    return {
        "sequence_format": checkpoint["sequence_format"],
        "augmentation_sampling_mode": checkpoint["augmentation_sampling_mode"],
        "config": config,
        "model_state": state,
        "task_ids": [],
        "embedding_task_ids": [],
    }


def finetune_config(defaults: dict, *, data_path: Path, save_path: Path,
                    epochs: int, batch_size: int, accumulation: int,
                    max_augments: int, num_workers: int,
                    normuon_lr: float, adamw_lr: float) -> Namespace:
    values = dict(defaults)
    # Architecture and MTP settings come from the saved checkpoint, without overrides.
    for key in ("n_predict", "mtp_weight", "dihedral_embeddings", "input_shortcuts", "rope_spatial_base"):
        values.pop(key, None)
    values.update(
        data_path=Path(data_path), checkpoint_path=None, checkpoint_mode="finetune",
        save_path=Path(save_path), train_log_file=Path(save_path).with_suffix(".log"),
        epochs=int(epochs), batch_size=int(batch_size),
        gradient_accumulation_steps=int(accumulation), num_workers=int(num_workers),
        pin_memory=True, persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        rearc_data_path=None, nvarc_training_path=None, nvarc_full_path=None,
        rearc_pairs_per_epoch=0, nvarc_pairs_per_epoch=0, nvarc_puzzles_per_epoch=None,
        variation_index_path=None, dataset_manifest_path=None,
        reset_training_state=True, inherit_checkpoint_rearc_data_path=False,
        distributed_training=False, distributed_inference=False,
        do_validate=False, eval_only=False, checkpoint_epochs=[],
        device="cuda:0", FA=2, enable_aug=True, max_augments=int(max_augments),
        normuon_lr=float(normuon_lr), adamw_lr=float(adamw_lr),
        log_location="both", train_log_mode="epoch",
        inference_speculative=False, mode="full-decode",
        depth_diagnostics_every=0,
    )
    if min(epochs, batch_size, accumulation, max_augments) < 1 or num_workers < 0:
        raise ValueError("Epochs, batch sizes, accumulation and augment counts must be positive.")
    if min(normuon_lr, adamw_lr) <= 0:
        raise ValueError("Learning rates must be positive.")
    return Namespace(**values)


def check_backward(model, dataloader, device) -> float:
    """Exercise the actual CUDA attention and backward before a long run."""
    from common import capture_rng_state, restore_rng_state
    from train import _move_batch_to_device

    rng = capture_rng_state(device)
    try:
        augmentor = getattr(dataloader, "augmentor", None)
        if hasattr(augmentor, "prepare_training_schedule"):
            sampler = dataloader.batch_sampler
            # Match train_model before persistent workers copy these plans.
            if hasattr(sampler, "drop_last"):
                sampler.drop_last = not getattr(sampler, "preserve_all_pairs", False)
            augmentor.prepare_training_schedule(sampler)
        batch = _move_batch_to_device(next(iter(dataloader)), device)
        model.train()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            output = model(
                batch["input_ids"], batch["example_ids"],
                dihedral_ids=batch.get("dihedral_ids"),
                attention_mask=batch.get("attention_mask"),
                sep_indices=batch["sep_indices"], positions_3d=batch["positions_3d"],
                cu_seqlens=batch.get("cu_seqlens"), max_seqlen=batch.get("max_seqlen"),
            )
        loss = output["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError("Preflight loss is not finite.")
        loss.backward()
        if not any(p.grad is not None for name, p in model.named_parameters()
                   if name.startswith("blocks.")):
            raise RuntimeError("Preflight did not produce backbone gradients.")
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for p in model.parameters()):
            raise RuntimeError("Preflight produced nonfinite gradients.")
        return float(loss.detach())
    finally:
        model.zero_grad(set_to_none=True)
        restore_rng_state(rng, device)


def validate_submission(submission: dict, challenges: dict) -> None:
    if not isinstance(submission, dict) or set(submission) != set(challenges):
        raise ValueError("Submission task ids must exactly match runtime challenge task ids.")
    for task_id, task in challenges.items():
        guesses = submission[task_id]
        if not isinstance(guesses, list) or len(guesses) != len(task["test"]):
            raise ValueError(f"{task_id}: wrong number of test predictions")
        for pair_idx, guess in enumerate(guesses):
            if not isinstance(guess, dict) or set(guess) != {"attempt_1", "attempt_2"}:
                raise ValueError(f"{task_id}/{pair_idx}: expected exactly two attempts")
            for key in ("attempt_1", "attempt_2"):
                if not valid_grid(guess[key]):
                    raise ValueError(f"{task_id}/{pair_idx}/{key}: invalid output grid")


def finalize_submission(source: Path, destination: Path, challenges: dict):
    submission = json.loads(Path(source).read_text())
    # Missing tasks/pairs indicate a pipeline error, rather than an invalid decode.
    if not isinstance(submission, dict) or set(submission) != set(challenges):
        raise ValueError("Decoder did not return exactly the runtime tasks.")
    repairs = []
    for task_id, task in challenges.items():
        guesses = submission[task_id]
        if not isinstance(guesses, list) or len(guesses) != len(task["test"]):
            raise ValueError(f"{task_id}: decoder missed one or more test inputs")
        for index, guess in enumerate(guesses):
            if not isinstance(guess, dict) or set(guess) != {"attempt_1", "attempt_2"}:
                raise ValueError(f"{task_id}/{index}: decoder missed an attempt")
            valid = [guess[k] for k in ("attempt_1", "attempt_2") if valid_grid(guess[k])]
            for key in ("attempt_1", "attempt_2"):
                if not valid_grid(guess[key]):
                    guess[key] = deepcopy(valid[0] if valid else task["test"][index]["input"])
                    repairs.append(f"{task_id}/{index}/{key}")
    validate_submission(submission, challenges)
    destination = Path(destination)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(submission, separators=(",", ":")))
    temporary.replace(destination)
    validate_submission(json.loads(destination.read_text()), challenges)
    return repairs
