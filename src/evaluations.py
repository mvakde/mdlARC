import argparse
import copy
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

import aaivr
import sanitized_eval
import train
from inference import DEFAULT_MAX_NEW_TOKENS, run_split_inference
from sanitized_augment import build_sanitized_augmentor
from tinytransformer import TinyTransformer
from utils import (
    END_TOKEN_ID,
    IO_SEPARATOR_TOKEN_ID,
    NEXT_LINE_TOKEN_ID,
    create_dataloader,
    generate_task_color_mappings,
    generate_task_dihedral_orders,
)


def _has_correct_shape(
    sequence: Sequence[int],
    predicted_tokens: Sequence[int],
    target_tokens: Sequence[int],
) -> bool:
    if not target_tokens:
        return False
    if len(predicted_tokens) != len(target_tokens):
        return False

    target_newlines = [
        idx for idx, tok in enumerate(target_tokens) if tok == NEXT_LINE_TOKEN_ID
    ]
    predicted_newlines = [
        idx for idx, tok in enumerate(predicted_tokens) if tok == NEXT_LINE_TOKEN_ID
    ]
    if target_newlines != predicted_newlines:
        return False

    try:
        sep_idx = sequence.index(IO_SEPARATOR_TOKEN_ID)
        end_idx = sequence.index(END_TOKEN_ID, sep_idx + 1)
    except ValueError:
        return False
    return (end_idx - (sep_idx + 1)) == len(target_tokens)


def _pixel_accuracy(
    predicted_tokens: Sequence[int], target_tokens: Sequence[int]
) -> Optional[float]:
    predicted_digits = [tok for tok in predicted_tokens if 0 <= tok <= 9]
    target_digits = [tok for tok in target_tokens if 0 <= tok <= 9]
    if not target_digits or len(predicted_digits) != len(target_digits):
        return None
    correct = sum(1 for p, t in zip(predicted_digits, target_digits) if p == t)
    return correct / len(target_digits)


def summarize_split_results(results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    num_shape_correct = 0
    num_fully_correct = 0
    accuracies: List[float] = []
    fully_correct_results: List[Dict[str, object]] = []

    for res in results:
        predicted_tokens = res.get("output_tokens", [])
        target_tokens = res.get("target_output_tokens", [])
        sequence = res.get("sequence", [])
        shape_ok = _has_correct_shape(sequence, predicted_tokens, target_tokens)
        res["shape_correct"] = shape_ok
        if not shape_ok:
            continue
        num_shape_correct += 1
        acc = _pixel_accuracy(predicted_tokens, target_tokens)
        if acc is not None:
            res["pixel_accuracy"] = acc
            accuracies.append(acc)
        if predicted_tokens == target_tokens:
            num_fully_correct += 1
            fully_correct_results.append(res)

    avg_pixel_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    return {
        "total_sequences": len(results),
        "num_shape_correct": num_shape_correct,
        "avg_pixel_accuracy": avg_pixel_accuracy,
        "num_fully_correct": num_fully_correct,
        "fully_correct_results": fully_correct_results,
    }


def evaluate_model_on_dataset(
    model: TinyTransformer,
    dataset,
    device: torch.device,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    batch_size: int = 16,
    splits: Sequence[str] = ("train", "test"),
    log_prompts: bool = False,
    color_mappings: Optional[Sequence[Sequence[int]]] = None,
    color_mappings_by_task: Optional[Dict[str, Sequence[Sequence[int]]]] = None,
    color_apply_fn: Optional[Callable[[str], bool]] = None,
    dihedral_orders_by_task: Optional[Dict[str, Sequence[int]]] = None,
    task_ids: Optional[Sequence[str]] = None,
    include_targets: bool = True,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
) -> Dict[str, Dict[str, object]]:
    evaluation: Dict[str, Dict[str, object]] = {}
    # Determine if we should look for targets for this specific split
    # Usually we want targets for 'train' (to debug) but maybe not for 'test' if doing blind submission
    split_include_targets = include_targets

    # If the dataset split itself doesn't have outputs (like test) and we forced include_targets=True,
    # run_split_inference will look for solutions.json. If we forced False, it won't.
    for split in splits:
        split_results = run_split_inference(
            model=model,
            dataset=dataset,
            split=split,
            device=device,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            log_prompts=log_prompts,
            include_targets=split_include_targets,
            color_mappings=color_mappings,
            color_mappings_by_task=color_mappings_by_task,
            color_apply_fn=color_apply_fn,
            dihedral_orders_by_task=dihedral_orders_by_task,
            task_ids=task_ids,
            temperature=temperature,
            top_k=top_k,
        )
        summary = summarize_split_results(split_results)
        evaluation[split] = {"results": split_results, "summary": summary}
    return evaluation


class _TaskDatasetView(Dataset):
    def __init__(self, dataset, indices: Sequence[int]) -> None:
        self._dataset = dataset
        self._indices = list(indices)
        self.sequence_lengths = [dataset.sequence_lengths[idx] for idx in self._indices]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        return self._dataset[self._indices[idx]]


def _task_ids_with_test_pairs(dataset, task_ids: Optional[Sequence[str]]) -> List[str]:
    test_task_ids = {ex.task_id for ex in dataset.iter_examples(split="test")}
    if task_ids is None:
        return sorted(test_task_ids)
    filtered = [task_id for task_id in task_ids if task_id in test_task_ids]
    missing = [task_id for task_id in task_ids if task_id not in test_task_ids]
    if missing:
        print(f"Skipping tasks without test pairs: {missing}")
    return filtered


def _build_task_sft_dataloader(
    dataset,
    indices: Sequence[int],
    *,
    batch_size: int,
    num_workers: int,
    sanitized_augmentor: Optional[object],
):
    if not indices:
        return None
    subset = _TaskDatasetView(dataset, indices)
    collate_augment_selector = (
        sanitized_augmentor.select_for_example if sanitized_augmentor else None
    )
    return create_dataloader(
        dataset=subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        augment_selector=collate_augment_selector,
    )


class TeeLogger:
    def __init__(self, filepath: Path) -> None:
        self.terminal = sys.stdout
        self.log = Path(filepath).open("w")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def close(self) -> None:
        self.log.close()


def _build_submission_from_aaivr(
    selections: Sequence[aaivr.AAIVRSelection],
) -> Dict[str, List[Dict[str, object]]]:
    submission_data: Dict[str, List[Dict[str, object]]] = {}
    temp_grouping: Dict[str, Dict[int, Dict[str, object]]] = {}

    for item in selections:
        task_id = item.task_id
        pair_idx = item.original_pair_index
        if task_id not in temp_grouping:
            temp_grouping[task_id] = {}

        top_grids = item.selected_outputs[:2]
        if not top_grids:
            top_grids = [[[0]]]

        pair_dict = {
            "attempt_1": top_grids[0],
            "attempt_2": top_grids[1] if len(top_grids) > 1 else top_grids[0],
        }
        temp_grouping[task_id][pair_idx] = pair_dict

    for task_id, pairs_map in temp_grouping.items():
        sorted_indices = sorted(pairs_map.keys())
        submission_data[task_id] = [pairs_map[idx] for idx in sorted_indices]

    return submission_data


def _compute_arc_style_score(
    selections: Sequence[aaivr.AAIVRSelection],
) -> Tuple[float, int, float]:
    tasks_map: Dict[str, List[aaivr.AAIVRSelection]] = {}
    for res in selections:
        tasks_map.setdefault(res.task_id, []).append(res)

    arc_score = 0.0
    total_tasks = len(tasks_map)

    for pairs in tasks_map.values():
        n_pairs = len(pairs)
        if n_pairs > 0:
            n_solved = sum(1 for p in pairs if p.pass_at_k)
            arc_score += n_solved / n_pairs

    max_score = total_tasks
    pct = (arc_score / max_score * 100) if max_score > 0 else 0.0
    return arc_score, max_score, pct


def run_evaluation_pipeline(
    cfg: argparse.Namespace,
    run_name: str,
    max_eval_augments: int,
    dataset_path: Path,
    *,
    eval_batch_size: int = 1300,
    splits: Sequence[str] = ("test",),
    checkpoint_path: Optional[Path] = None,
    include_targets: bool = False,
    task_ids: Optional[Sequence[str]] = None,
    log_correct_grids: bool = False,
    state: Optional[Dict[str, object]] = None,
) -> Tuple[
    Dict[str, Dict[str, object]],
    List[aaivr.AAIVRSelection],
    Path,
    Dict[str, object],
]:
    print(f"\n{'=' * 60}")
    print(
        f"STARTING PIPELINE: {run_name} "
        f"({'Sanitized' if getattr(cfg, 'enable_sanitized_aug_train', False) else 'Color'} "
        f"Augs: {max_eval_augments})"
    )
    print(f"{'=' * 60}\n")

    if state is None:
        state = {}
    dataset_path = Path(dataset_path)

    base_run_dir = Path("runs") / run_name
    base_run_dir.mkdir(parents=True, exist_ok=True)

    eval_log_path = base_run_dir / "eval_log.txt"
    aaivr_log_path = base_run_dir / "aaivr.txt"
    submission_path = base_run_dir / "submission.json"

    prev_checkpoint = getattr(cfg, "checkpoint_path", None)
    prev_data_path = getattr(cfg, "data_path", None)
    had_enable_color_aug = hasattr(cfg, "enable_color_aug_eval")
    prev_enable_color_aug = getattr(cfg, "enable_color_aug_eval", None)
    had_max_color_aug = hasattr(cfg, "max_color_augments_eval")
    prev_max_color_aug = getattr(cfg, "max_color_augments_eval", None)
    had_max_sanitized_aug = hasattr(cfg, "max_sanitized_augments")
    prev_max_sanitized_aug = getattr(cfg, "max_sanitized_augments", None)
    had_enable_dihedral_aug = hasattr(cfg, "enable_dihedral_aug_eval")
    prev_enable_dihedral_aug = getattr(cfg, "enable_dihedral_aug_eval", None)

    if checkpoint_path is None:
        checkpoint_path = getattr(cfg, "checkpoint_path", None)
    if checkpoint_path is None:
        raise ValueError("checkpoint_path must be provided for evaluation.")

    cfg.checkpoint_path = Path(checkpoint_path)
    cfg.data_path = dataset_path
    cfg.enable_color_aug_eval = max_eval_augments > 0
    cfg.max_color_augments_eval = max_eval_augments
    cfg.enable_dihedral_aug_eval = bool(
        getattr(cfg, "enable_dihedral_aug_eval", False)
    )
    use_sanitized = bool(getattr(cfg, "enable_sanitized_aug_train", False))
    if use_sanitized:
        cfg.max_sanitized_augments = int(max_eval_augments)

    reuse_dataset = None
    prior_dataset = state.get("dataset")
    if prior_dataset is not None:
        prior_path = getattr(prior_dataset, "source_path", None)
        if prior_path is not None and Path(prior_path) == dataset_path:
            reuse_dataset = prior_dataset

    checkpoint = state.get("checkpoint")
    if checkpoint is None or state.get("checkpoint_path") != str(checkpoint_path):
        checkpoint = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
        state["checkpoint"] = checkpoint
        state["checkpoint_path"] = str(checkpoint_path)

    print("Building model and dataloader for config...")
    model, dataset, _, device, _ = train.build_model_and_data(
        cfg, checkpoint=checkpoint, reuse_dataset=reuse_dataset
    )
    state["dataset"] = dataset

    def log_eval(msg: str) -> None:
        print(msg)
        with eval_log_path.open("a") as handle:
            handle.write(msg + "\n")

    color_mappings_eval = None
    color_apply_fn = None
    dihedral_orders_eval = None
    sanitized_color_mappings_by_split: Dict[str, Dict[str, List[List[int]]]] = {}
    sanitized_dihedral_by_split: Dict[str, bool] = {}

    if use_sanitized:
        max_sanitized_augments = int(
            getattr(cfg, "max_sanitized_augments", 0) or 0
        )
        if max_sanitized_augments <= 0:
            max_sanitized_augments = int(
                getattr(cfg, "max_color_augments_train", 0) or 0
            )
        sanitized_augmentor = getattr(dataset, "sanitized_augmentor", None)
        if (
            sanitized_augmentor is None
            or getattr(sanitized_augmentor, "max_sanitized_augments", None)
            != max_sanitized_augments
        ):
            enable_color_train = bool(getattr(cfg, "enable_color_aug_train", False))
            enable_dihedral_train = bool(
                getattr(cfg, "enable_dihedral_aug_train", False)
            )
            color_apply_to_test = bool(
                getattr(cfg, "enable_color_on_aug_test_split_during_training", False)
            )
            dihedral_apply_to_test = bool(
                getattr(cfg, "enable_dihedral_on_aug_test_split_during_training", False)
            )
            seed = getattr(cfg, "sanitized_aug_seed", None)
            if seed is None:
                seed = getattr(cfg, "color_aug_seed", None)
            if seed is None:
                seed = getattr(cfg, "seed", 42)
            sanitized_augmentor = build_sanitized_augmentor(
                dataset.examples,
                dataset.task_input_colors,
                max_sanitized_augments=max_sanitized_augments,
                enable_color=enable_color_train,
                enable_dihedral=enable_dihedral_train,
                seed=int(seed),
                color_apply_to_test_split=color_apply_to_test,
                dihedral_apply_to_test_split=dihedral_apply_to_test,
            )
            dataset.sanitized_augmentor = sanitized_augmentor
        evaluation: Dict[str, Dict[str, object]] = {}
        for split in splits:
            split_results, color_maps, dihedral_augmented = (
                sanitized_eval.run_split_inference_sanitized(
                    model=model,
                    dataset=dataset,
                    split=split,
                    device=device,
                    augmentor=sanitized_augmentor,
                    batch_size=eval_batch_size,
                    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                    task_ids=task_ids,
                    log_prompts=getattr(cfg, "log_inference_prompt", False),
                    include_targets=include_targets,
                    temperature=getattr(cfg, "inference_temperature", None),
                    top_k=getattr(cfg, "inference_top_k", None),
                )
            )
            summary = summarize_split_results(split_results)
            evaluation[split] = {"results": split_results, "summary": summary}
            sanitized_color_mappings_by_split[split] = color_maps
            sanitized_dihedral_by_split[split] = dihedral_augmented
        evaluation["_sanitized"] = {
            "color_mappings_by_split": sanitized_color_mappings_by_split,
            "dihedral_augmented_by_split": sanitized_dihedral_by_split,
        }
    else:
        if cfg.enable_color_aug_eval and cfg.max_color_augments_eval > 0:
            color_seed = getattr(cfg, "color_aug_seed_eval", None)
            if color_seed is None:
                color_seed = getattr(cfg, "color_aug_seed", None)
            if color_seed is None:
                color_seed = cfg.seed
            color_mappings_eval = generate_task_color_mappings(
                dataset.task_input_colors, cfg.max_color_augments_eval, int(color_seed)
            )
            color_apply_fn = lambda split: True

        if cfg.enable_dihedral_aug_eval:
            dihedral_seed = getattr(cfg, "dihedral_aug_seed", None)
            if dihedral_seed is None:
                dihedral_seed = getattr(cfg, "seed", 42)
            dihedral_orders_eval = generate_task_dihedral_orders(
                dataset.task_ids, int(dihedral_seed)
            )

        evaluation = evaluate_model_on_dataset(
            model=model,
            dataset=dataset,
            device=device,
            batch_size=eval_batch_size,
            log_prompts=getattr(cfg, "log_inference_prompt", False),
            temperature=getattr(cfg, "inference_temperature", None),
            top_k=getattr(cfg, "inference_top_k", None),
            splits=splits,
            color_mappings_by_task=color_mappings_eval,
            color_apply_fn=color_apply_fn,
            dihedral_orders_by_task=dihedral_orders_eval,
            task_ids=task_ids,
            include_targets=include_targets,
        )

    epochs = getattr(cfg, "epochs", None)
    epoch_label = f"{epochs}ep" if epochs is not None else "eval"
    label = "sanitized" if use_sanitized else "color"
    log_eval(f"\n-- {epoch_label} {max_eval_augments}{label} --\n")
    dihedral_augmented = bool(cfg.enable_dihedral_aug_eval)
    if use_sanitized:
        dihedral_augmented = sanitized_dihedral_by_split.get("test", False)

    for split in splits:
        summary = evaluation.get(split, {}).get("summary", {})
        total = summary.get("total_sequences", 0)
        shape_ok = summary.get("num_shape_correct", 0)
        fully_correct = summary.get("num_fully_correct", 0)
        avg_pixel_acc = summary.get("avg_pixel_accuracy", 0.0)

        log_eval(
            f"Split: {split} | Seq: {total} | Shape OK: {shape_ok} | Fully Correct: {fully_correct} | Pixel Acc: {avg_pixel_acc:.4f}"
        )

        if log_correct_grids and fully_correct > 0:
            log_eval(f"  [Correct Grids Details for {split}]")

            is_dihedral_split = (
                sanitized_dihedral_by_split.get(split, dihedral_augmented)
                if use_sanitized
                else dihedral_augmented
            )

            correct_results = summary.get("fully_correct_results", [])
            for res in correct_results:
                raw_idx = res.get("pair_index", 0)
                if is_dihedral_split:
                    pair_id = raw_idx // 8
                    dihedral_id = raw_idx % 8
                else:
                    pair_id = raw_idx
                    dihedral_id = 0

                color_id = res.get("color_permutation_index", 0)
                grid = res.get("output_grid", [])
                log_eval(
                    f"    T:{res.get('task_id')} | Pair:{pair_id} | Dihedral:{dihedral_id} | Color:{color_id} -> {grid}"
                )

    print(f"Running AAIVR for {run_name}...")
    if hasattr(sys.stdout, "log"):
        sys.stdout = sys.stdout.terminal
    sys.stdout = TeeLogger(aaivr_log_path)

    try:
        test_results = evaluation.get("test", {}).get("results", [])
        dataset_has_dihedral_augments = dihedral_augmented

        aaivr_results: List[aaivr.AAIVRSelection] = []
        if test_results:
            aaivr_color_mappings = color_mappings_eval
            if use_sanitized:
                aaivr_color_mappings = sanitized_color_mappings_by_split.get("test")
            aaivr_results = aaivr.run_aaivr_on_results(
                test_results,
                is_dihedral_augmented=dataset_has_dihedral_augments,
                color_mappings_by_task=aaivr_color_mappings,
                dihedral_orders_by_task=dihedral_orders_eval,
            )
        else:
            print("No test results for AAIVR.")

        aaivr.summarize_aaivr_pass_at_k(aaivr_results)
        arc_score, max_score, pct = _compute_arc_style_score(aaivr_results)
        print(
            f"Official ARC style scoring: {arc_score:.2f}/{max_score} ({pct:.2f}%)"
        )
    finally:
        if hasattr(sys.stdout, "terminal"):
            sys.stdout.close()
            sys.stdout = sys.stdout.terminal

    print(f"Generating submission.json for {run_name}...")
    submission_data = _build_submission_from_aaivr(aaivr_results)
    with submission_path.open("w") as handle:
        json.dump(submission_data, handle)

    print(f"Finished {run_name}. Submission saved to {submission_path}")

    cfg.checkpoint_path = prev_checkpoint
    cfg.data_path = prev_data_path
    if had_enable_color_aug:
        cfg.enable_color_aug_eval = prev_enable_color_aug
    else:
        delattr(cfg, "enable_color_aug_eval")
    if had_max_color_aug:
        cfg.max_color_augments_eval = prev_max_color_aug
    else:
        delattr(cfg, "max_color_augments_eval")
    if had_enable_dihedral_aug:
        cfg.enable_dihedral_aug_eval = prev_enable_dihedral_aug
    else:
        delattr(cfg, "enable_dihedral_aug_eval")
    if had_max_sanitized_aug:
        cfg.max_sanitized_augments = prev_max_sanitized_aug
    else:
        if hasattr(cfg, "max_sanitized_augments"):
            delattr(cfg, "max_sanitized_augments")

    return evaluation, aaivr_results, submission_path, state


def run_task_sft_evaluation_pipeline(
    cfg: argparse.Namespace,
    run_name: str,
    max_eval_augments: int,
    dataset_path: Path,
    *,
    eval_batch_size: int = 1300,
    splits: Sequence[str] = ("test",),
    checkpoint_path: Optional[Path] = None,
    include_targets: bool = False,
    task_ids: Optional[Sequence[str]] = None,
    log_correct_grids: bool = False,
    state: Optional[Dict[str, object]] = None,
) -> Tuple[
    Dict[str, Dict[str, object]],
    List[aaivr.AAIVRSelection],
    Path,
    Dict[str, object],
]:
    print(f"\n{'=' * 60}")
    print(
        f"STARTING TASK SFT PIPELINE: {run_name} "
        f"({'Sanitized' if getattr(cfg, 'enable_sanitized_aug_train', False) else 'Color'} "
        f"Augs: {max_eval_augments})"
    )
    print(f"{'=' * 60}\n")

    if state is None:
        state = {}
    dataset_path = Path(dataset_path)

    base_run_dir = Path("runs") / run_name
    base_run_dir.mkdir(parents=True, exist_ok=True)

    eval_log_path = base_run_dir / "eval_log.txt"
    aaivr_log_path = base_run_dir / "aaivr.txt"
    submission_path = base_run_dir / "submission.json"

    prev_checkpoint = getattr(cfg, "checkpoint_path", None)
    prev_data_path = getattr(cfg, "data_path", None)
    had_enable_color_aug = hasattr(cfg, "enable_color_aug_eval")
    prev_enable_color_aug = getattr(cfg, "enable_color_aug_eval", None)
    had_max_color_aug = hasattr(cfg, "max_color_augments_eval")
    prev_max_color_aug = getattr(cfg, "max_color_augments_eval", None)
    had_max_sanitized_aug = hasattr(cfg, "max_sanitized_augments")
    prev_max_sanitized_aug = getattr(cfg, "max_sanitized_augments", None)
    had_enable_dihedral_aug = hasattr(cfg, "enable_dihedral_aug_eval")
    prev_enable_dihedral_aug = getattr(cfg, "enable_dihedral_aug_eval", None)

    if checkpoint_path is None:
        checkpoint_path = getattr(cfg, "checkpoint_path", None)
    if checkpoint_path is None:
        raise ValueError("checkpoint_path must be provided for evaluation.")

    cfg.checkpoint_path = Path(checkpoint_path)
    cfg.data_path = dataset_path
    cfg.enable_color_aug_eval = max_eval_augments > 0
    cfg.max_color_augments_eval = max_eval_augments
    cfg.enable_dihedral_aug_eval = bool(
        getattr(cfg, "enable_dihedral_aug_eval", False)
    )
    use_sanitized = bool(getattr(cfg, "enable_sanitized_aug_train", False))
    if use_sanitized:
        cfg.max_sanitized_augments = int(max_eval_augments)

    reuse_dataset = None
    prior_dataset = state.get("dataset")
    if prior_dataset is not None:
        prior_path = getattr(prior_dataset, "source_path", None)
        if prior_path is not None and Path(prior_path) == dataset_path:
            reuse_dataset = prior_dataset

    checkpoint = state.get("checkpoint")
    if checkpoint is None or state.get("checkpoint_path") != str(checkpoint_path):
        checkpoint = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
        state["checkpoint"] = checkpoint
        state["checkpoint_path"] = str(checkpoint_path)

    print("Building model and dataloader for config...")
    model, dataset, _, device, _ = train.build_model_and_data(
        cfg, checkpoint=checkpoint, reuse_dataset=reuse_dataset
    )
    state["dataset"] = dataset

    def log_eval(msg: str) -> None:
        print(msg)
        with eval_log_path.open("a") as handle:
            handle.write(msg + "\n")

    train_indices_by_task: Dict[str, List[int]] = {}
    for idx, ex in enumerate(dataset.examples):
        if ex.split == "train" and ex.has_output:
            train_indices_by_task.setdefault(ex.task_id, []).append(idx)

    eval_task_ids = _task_ids_with_test_pairs(dataset, task_ids)

    color_mappings_eval = None
    color_apply_fn = None
    dihedral_orders_eval = None
    sanitized_color_mappings_by_split: Dict[str, Dict[str, List[List[int]]]] = {
        split: {} for split in splits
    }
    sanitized_dihedral_by_split: Dict[str, bool] = {split: False for split in splits}
    sanitized_augmentor = None

    if use_sanitized:
        max_sanitized_augments = int(
            getattr(cfg, "max_sanitized_augments", 0) or 0
        )
        if max_sanitized_augments <= 0:
            max_sanitized_augments = int(
                getattr(cfg, "max_color_augments_train", 0) or 0
            )
        sanitized_augmentor = getattr(dataset, "sanitized_augmentor", None)
        if (
            sanitized_augmentor is None
            or getattr(sanitized_augmentor, "max_sanitized_augments", None)
            != max_sanitized_augments
        ):
            enable_color_train = bool(getattr(cfg, "enable_color_aug_train", False))
            enable_dihedral_train = bool(
                getattr(cfg, "enable_dihedral_aug_train", False)
            )
            color_apply_to_test = bool(
                getattr(cfg, "enable_color_on_aug_test_split_during_training", False)
            )
            dihedral_apply_to_test = bool(
                getattr(cfg, "enable_dihedral_on_aug_test_split_during_training", False)
            )
            seed = getattr(cfg, "sanitized_aug_seed", None)
            if seed is None:
                seed = getattr(cfg, "color_aug_seed", None)
            if seed is None:
                seed = getattr(cfg, "seed", 42)
            sanitized_augmentor = build_sanitized_augmentor(
                dataset.examples,
                dataset.task_input_colors,
                max_sanitized_augments=max_sanitized_augments,
                enable_color=enable_color_train,
                enable_dihedral=enable_dihedral_train,
                seed=int(seed),
                color_apply_to_test_split=color_apply_to_test,
                dihedral_apply_to_test_split=dihedral_apply_to_test,
            )
            dataset.sanitized_augmentor = sanitized_augmentor
        if sanitized_augmentor is not None:
            sanitized_augmentor.set_epoch(0)
    else:
        if cfg.enable_color_aug_eval and cfg.max_color_augments_eval > 0:
            color_seed = getattr(cfg, "color_aug_seed_eval", None)
            if color_seed is None:
                color_seed = getattr(cfg, "color_aug_seed", None)
            if color_seed is None:
                color_seed = cfg.seed
            color_mappings_eval = generate_task_color_mappings(
                dataset.task_input_colors, cfg.max_color_augments_eval, int(color_seed)
            )
            color_apply_fn = lambda split: True

        if cfg.enable_dihedral_aug_eval:
            dihedral_seed = getattr(cfg, "dihedral_aug_seed", None)
            if dihedral_seed is None:
                dihedral_seed = getattr(cfg, "seed", 42)
            dihedral_orders_eval = generate_task_dihedral_orders(
                dataset.task_ids, int(dihedral_seed)
            )

    sft_epochs = int(getattr(cfg, "sft_epochs", 1) or 0)
    sft_batch_size = int(
        getattr(cfg, "sft_batch_size", getattr(cfg, "batch_size", 1)) or 1
    )
    sft_grad_clip = float(
        getattr(cfg, "sft_grad_clip", getattr(cfg, "grad_clip", 0.0) or 0.0)
    )
    sft_grad_accum = int(
        getattr(
            cfg,
            "sft_gradient_accumulation_steps",
            getattr(cfg, "gradient_accumulation_steps", 1),
        )
        or 1
    )
    sft_log_train_strings = bool(getattr(cfg, "sft_log_train_strings", False))
    sft_log_train_limit = int(getattr(cfg, "sft_log_train_limit", 0) or 0)

    attention_weight_decay = getattr(cfg, "attention_weight_decay", cfg.weight_decay)
    token_embedding_weight_decay = getattr(cfg, "token_embedding_weight_decay", 0.0)
    task_embedding_weight_decay = getattr(cfg, "task_embedding_weight_decay", 0.0)

    base_model = model
    if device.type == "cuda":
        base_model.to("cpu")
        torch.cuda.empty_cache()

    evaluation: Dict[str, Dict[str, object]] = {
        split: {"results": []} for split in splits
    }

    if not eval_task_ids:
        print("No tasks with test pairs found; skipping evaluation.")
    else:
        for task_id in eval_task_ids:
            train_indices = train_indices_by_task.get(task_id, [])
            sft_dataloader = _build_task_sft_dataloader(
                dataset,
                train_indices,
                batch_size=sft_batch_size,
                num_workers=getattr(cfg, "num_workers", 0),
                sanitized_augmentor=sanitized_augmentor,
            )

            task_model = copy.deepcopy(base_model)
            task_model.to(device)

            optimizer = None
            if sft_dataloader is not None and sft_epochs > 0:
                optimizer = train._build_optimizer(
                    cfg,
                    task_model,
                    device,
                    attention_weight_decay,
                    token_embedding_weight_decay,
                    task_embedding_weight_decay,
                )
                for _ in range(sft_epochs):
                    train.train_one_epoch(
                        model=task_model,
                        dataloader=sft_dataloader,
                        optimizer=optimizer,
                        device=device,
                        grad_clip=sft_grad_clip,
                        gradient_accumulation_steps=sft_grad_accum,
                        log_train_strings=sft_log_train_strings,
                        log_train_limit=sft_log_train_limit,
                        loss_key="output_loss",
                    )
            elif not train_indices:
                print(f"Skipping SFT for {task_id}: no train pairs.")

            for split in splits:
                if use_sanitized and sanitized_augmentor is not None:
                    split_results, color_maps, dihedral_augmented = (
                        sanitized_eval.run_split_inference_sanitized(
                            model=task_model,
                            dataset=dataset,
                            split=split,
                            device=device,
                            augmentor=sanitized_augmentor,
                            batch_size=eval_batch_size,
                            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                            task_ids=[task_id],
                            log_prompts=getattr(cfg, "log_inference_prompt", False),
                            include_targets=include_targets,
                            temperature=getattr(cfg, "inference_temperature", None),
                            top_k=getattr(cfg, "inference_top_k", None),
                        )
                    )
                    evaluation[split]["results"].extend(split_results)
                    if color_maps:
                        sanitized_color_mappings_by_split[split].update(color_maps)
                    if dihedral_augmented:
                        sanitized_dihedral_by_split[split] = True
                else:
                    split_results = run_split_inference(
                        model=task_model,
                        dataset=dataset,
                        split=split,
                        device=device,
                        batch_size=eval_batch_size,
                        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                        log_prompts=getattr(cfg, "log_inference_prompt", False),
                        include_targets=include_targets,
                        color_mappings_by_task=color_mappings_eval,
                        color_apply_fn=color_apply_fn,
                        dihedral_orders_by_task=dihedral_orders_eval,
                        task_ids=[task_id],
                        temperature=getattr(cfg, "inference_temperature", None),
                        top_k=getattr(cfg, "inference_top_k", None),
                    )
                    evaluation[split]["results"].extend(split_results)

            del task_model
            if optimizer is not None:
                del optimizer
            if sft_dataloader is not None:
                del sft_dataloader
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    for split in splits:
        summary = summarize_split_results(evaluation[split]["results"])
        evaluation[split]["summary"] = summary

    if use_sanitized:
        evaluation["_sanitized"] = {
            "color_mappings_by_split": sanitized_color_mappings_by_split,
            "dihedral_augmented_by_split": sanitized_dihedral_by_split,
        }

    epochs = getattr(cfg, "epochs", None)
    epoch_label = f"{epochs}ep" if epochs is not None else "eval"
    label = "sanitized" if use_sanitized else "color"
    log_eval(f"\n-- {epoch_label} {max_eval_augments}{label} --\n")
    dihedral_augmented = bool(cfg.enable_dihedral_aug_eval)
    if use_sanitized:
        dihedral_augmented = sanitized_dihedral_by_split.get("test", False)

    for split in splits:
        summary = evaluation.get(split, {}).get("summary", {})
        total = summary.get("total_sequences", 0)
        shape_ok = summary.get("num_shape_correct", 0)
        fully_correct = summary.get("num_fully_correct", 0)
        avg_pixel_acc = summary.get("avg_pixel_accuracy", 0.0)

        log_eval(
            f"Split: {split} | Seq: {total} | Shape OK: {shape_ok} | Fully Correct: {fully_correct} | Pixel Acc: {avg_pixel_acc:.4f}"
        )

        if log_correct_grids and fully_correct > 0:
            log_eval(f"  [Correct Grids Details for {split}]")

            is_dihedral_split = (
                sanitized_dihedral_by_split.get(split, dihedral_augmented)
                if use_sanitized
                else dihedral_augmented
            )

            correct_results = summary.get("fully_correct_results", [])
            for res in correct_results:
                raw_idx = res.get("pair_index", 0)
                if is_dihedral_split:
                    pair_id = raw_idx // 8
                    dihedral_id = raw_idx % 8
                else:
                    pair_id = raw_idx
                    dihedral_id = 0

                color_id = res.get("color_permutation_index", 0)
                grid = res.get("output_grid", [])
                log_eval(
                    f"    T:{res.get('task_id')} | Pair:{pair_id} | Dihedral:{dihedral_id} | Color:{color_id} -> {grid}"
                )

    print(f"Running AAIVR for {run_name}...")
    if hasattr(sys.stdout, "log"):
        sys.stdout = sys.stdout.terminal
    sys.stdout = TeeLogger(aaivr_log_path)

    try:
        test_results = evaluation.get("test", {}).get("results", [])
        dataset_has_dihedral_augments = dihedral_augmented

        aaivr_results: List[aaivr.AAIVRSelection] = []
        if test_results:
            aaivr_color_mappings = color_mappings_eval
            if use_sanitized:
                aaivr_color_mappings = sanitized_color_mappings_by_split.get("test")
            aaivr_results = aaivr.run_aaivr_on_results(
                test_results,
                is_dihedral_augmented=dataset_has_dihedral_augments,
                color_mappings_by_task=aaivr_color_mappings,
                dihedral_orders_by_task=dihedral_orders_eval,
            )
        else:
            print("No test results for AAIVR.")

        aaivr.summarize_aaivr_pass_at_k(aaivr_results)
        arc_score, max_score, pct = _compute_arc_style_score(aaivr_results)
        print(
            f"Official ARC style scoring: {arc_score:.2f}/{max_score} ({pct:.2f}%)"
        )
    finally:
        if hasattr(sys.stdout, "terminal"):
            sys.stdout.close()
            sys.stdout = sys.stdout.terminal

    print(f"Generating submission.json for {run_name}...")
    submission_data = _build_submission_from_aaivr(aaivr_results)
    with submission_path.open("w") as handle:
        json.dump(submission_data, handle)

    print(f"Finished {run_name}. Submission saved to {submission_path}")

    cfg.checkpoint_path = prev_checkpoint
    cfg.data_path = prev_data_path
    if had_enable_color_aug:
        cfg.enable_color_aug_eval = prev_enable_color_aug
    else:
        delattr(cfg, "enable_color_aug_eval")
    if had_max_color_aug:
        cfg.max_color_augments_eval = prev_max_color_aug
    else:
        delattr(cfg, "max_color_augments_eval")
    if had_enable_dihedral_aug:
        cfg.enable_dihedral_aug_eval = prev_enable_dihedral_aug
    else:
        delattr(cfg, "enable_dihedral_aug_eval")
    if had_max_sanitized_aug:
        cfg.max_sanitized_augments = prev_max_sanitized_aug
    else:
        if hasattr(cfg, "max_sanitized_augments"):
            delattr(cfg, "max_sanitized_augments")

    return evaluation, aaivr_results, submission_path, state


def run_evaluation_configs(
    cfg: argparse.Namespace,
    eval_configs: Sequence[Tuple[str, int, Path]],
    *,
    eval_batch_size: int = 1300,
    splits: Sequence[str] = ("test",),
    checkpoint_path: Optional[Path] = None,
    include_targets: bool = False,
    task_ids: Optional[Sequence[str]] = None,
    log_correct_grids: bool = False,
    timing_path: Path = Path("runs/timing.txt"),
) -> List[Tuple[str, Dict[str, Dict[str, object]], Path]]:
    timing_path = Path(timing_path)
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    state: Dict[str, object] = {}
    results: List[Tuple[str, Dict[str, Dict[str, object]], Path]] = []

    for config in eval_configs:
        if len(config) != 3:
            raise ValueError(
                "eval_configs entries must be (name, aug_count, data_path)."
            )
        name, aug_count, data_path = config
        t_start = perf_counter()
        evaluation, _, submission_path, state = run_evaluation_pipeline(
            cfg,
            name,
            aug_count,
            data_path,
            eval_batch_size=eval_batch_size,
            splits=splits,
            checkpoint_path=checkpoint_path,
            include_targets=include_targets,
            task_ids=task_ids,
            log_correct_grids=log_correct_grids,
            state=state,
        )
        t_duration = perf_counter() - t_start
        print(f"Run {name} took {t_duration:.2f}s")
        with timing_path.open("a") as handle:
            handle.write(f"Evaluation {name}: {t_duration:.4f} s\n")
        results.append((name, evaluation, submission_path))

    print("\nAll evaluation runs completed.")
    return results


def run_task_sft_evaluation_configs(
    cfg: argparse.Namespace,
    eval_configs: Sequence[Tuple[str, int, Path]],
    *,
    eval_batch_size: int = 1300,
    splits: Sequence[str] = ("test",),
    checkpoint_path: Optional[Path] = None,
    include_targets: bool = False,
    task_ids: Optional[Sequence[str]] = None,
    log_correct_grids: bool = False,
    timing_path: Path = Path("runs/timing.txt"),
) -> List[Tuple[str, Dict[str, Dict[str, object]], Path]]:
    timing_path = Path(timing_path)
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    state: Dict[str, object] = {}
    results: List[Tuple[str, Dict[str, Dict[str, object]], Path]] = []

    for config in eval_configs:
        if len(config) != 3:
            raise ValueError(
                "eval_configs entries must be (name, aug_count, data_path)."
            )
        name, aug_count, data_path = config
        t_start = perf_counter()
        evaluation, _, submission_path, state = run_task_sft_evaluation_pipeline(
            cfg,
            name,
            aug_count,
            data_path,
            eval_batch_size=eval_batch_size,
            splits=splits,
            checkpoint_path=checkpoint_path,
            include_targets=include_targets,
            task_ids=task_ids,
            log_correct_grids=log_correct_grids,
            state=state,
        )
        t_duration = perf_counter() - t_start
        print(f"Run {name} took {t_duration:.2f}s")
        with timing_path.open("a") as handle:
            handle.write(f"Task SFT evaluation {name}: {t_duration:.4f} s\n")
        results.append((name, evaluation, submission_path))

    print("\nAll task SFT evaluation runs completed.")
    return results
