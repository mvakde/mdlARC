"""Utility functions for visualization, scoring, and memory cleanup.

For tokenization, dataset classes, and other shared code, see common.py.
"""

import gc
import gzip
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np
except Exception:
    plt = None
    ListedColormap = None
    np = None


# =============================================================================
# Visualization
# =============================================================================

DEFAULT_COLORS = [
    "#000000",  # 0 black
    "#1e93ff",  # 1 blue
    "#f93c31",  # 2 red
    "#4eea3a",  # 3 green
    "#ffd84d",  # 4 yellow
    "#999999",  # 5 gray
    "#e53aa3",  # 6 magenta
    "#ff8a1f",  # 7 orange
    "#4fe3f0",  # 8 blue-light
    "#921231",  # 9 maroon
]


def plot_grids(
    grids: List[List[List[int]]], title: Optional[str] = None, figsize=(4, 4)
) -> None:
    """Plot one or more integer grids using a fixed 10-color palette."""
    if plt is None or ListedColormap is None:
        raise RuntimeError("matplotlib is not available in this environment.")

    n = max(1, len(grids))
    fig, axes = plt.subplots(1, n, figsize=(figsize[0] * n, figsize[1]))
    if n == 1:
        axes = [axes]
    cmap = ListedColormap(DEFAULT_COLORS)

    for ax, grid in zip(axes, grids):
        if not grid:
            ax.axis("off")
            continue
        arr = np.array(grid, dtype=int)
        ax.imshow(arr, cmap=cmap, vmin=0, vmax=9)
        ax.set_xticks(np.arange(-0.5, arr.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, arr.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def visualize_submissions(
    submission_file: Path,
    solutions_file: Optional[Path] = None,
    mode: str = "submission",
) -> None:
    """Visualize submission attempts, optionally comparing against solutions."""
    submission_path = Path(submission_file)
    if not submission_path.exists():
        print(f"Error: Could not find submission file: {submission_path}")
        return

    mode_normalized = "compare" if mode == "!" else mode
    if mode_normalized not in ("compare", "submission"):
        print(f"Error: Unknown visualization mode '{mode}'.")
        return

    if mode_normalized == "compare":
        if solutions_file is None:
            print("Error: Solutions file required for compare mode.")
            return
        solutions_path = Path(solutions_file)
        if not solutions_path.exists():
            print(
                f"Error: Could not find solutions file for compare mode:\n{solutions_path}"
            )
            return

        with submission_path.open("r") as handle:
            subs = json.load(handle)
        with solutions_path.open("r") as handle:
            sols = json.load(handle)

        print(f"Visualizing comparison for {len(subs)} tasks...")

        for task_id, attempts_list in subs.items():
            if task_id not in sols:
                print(f"Warning: Task {task_id} not found in solutions.json")
                continue

            gt_grids = sols[task_id]
            print(gt_grids)
            for i, attempts in enumerate(attempts_list):
                if i >= len(gt_grids):
                    break

                gt = gt_grids[i]
                att1 = attempts.get("attempt_1")
                att2 = attempts.get("attempt_2")

                pass1 = (att1 == gt) if att1 is not None else False
                pass2 = (att2 == gt) if att2 is not None else False

                if pass1 and pass2:
                    status = "Pass - both"
                elif pass1:
                    status = "Pass - 1"
                elif pass2:
                    status = "Pass - 2"
                else:
                    status = "Fail"

                grids_to_plot = [gt]
                if att1 is not None:
                    grids_to_plot.append(att1)
                if att2 is not None:
                    grids_to_plot.append(att2)

                header = f"Task: {task_id} | Pair: {i} | Status: {status}"
                print(f"Plotting {header}")

                try:
                    plot_grids(grids_to_plot, title=header)
                except Exception as exc:
                    print(f"Skipping plot for {task_id} due to error: {exc}")
    else:
        with submission_path.open("r") as handle:
            subs = json.load(handle)

        print(f"Visualizing submissions for {len(subs)} tasks (no solutions)...")

        for task_id, attempts_list in subs.items():
            for i, attempts in enumerate(attempts_list):
                att1 = attempts.get("attempt_1")
                att2 = attempts.get("attempt_2")

                grids_to_plot = []
                if att1 is not None:
                    grids_to_plot.append(att1)
                if att2 is not None:
                    grids_to_plot.append(att2)

                if not grids_to_plot:
                    print(f"Skipping {task_id} pair {i} (no attempts)")
                    continue

                header = f"Task: {task_id} | Pair: {i} | Status: submission-only"
                print(f"Plotting {header}")

                try:
                    plot_grids(grids_to_plot, title=header)
                except Exception as exc:
                    print(f"Skipping plot for {task_id} due to error: {exc}")


# =============================================================================
# Scoring
# =============================================================================

def score_arc_submission(
    solutions_file: Path, submission_file: Path
) -> Dict[str, object]:
    """Score a submission.json against ARC solutions.json."""
    solutions_path = Path(solutions_file)
    submission_path = Path(submission_file)

    if not solutions_path.exists():
        raise FileNotFoundError(f"Solutions file not found: {solutions_path}")
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission file not found: {submission_path}")

    with solutions_path.open("r") as handle:
        solutions = json.load(handle)

    with submission_path.open("r") as handle:
        submissions = json.load(handle)

    calc_score = 0.0
    max_total_score = len(solutions)
    fully_solved_tasks: List[str] = []

    for task_id, ground_truth_grids in solutions.items():
        if task_id not in submissions:
            continue

        task_attempts = submissions[task_id]
        num_pairs = len(ground_truth_grids)
        pairs_solved = 0

        for i in range(min(len(task_attempts), num_pairs)):
            truth = ground_truth_grids[i]
            attempts = task_attempts[i]
            if attempts.get("attempt_1") == truth or attempts.get("attempt_2") == truth:
                pairs_solved += 1

        if num_pairs > 0:
            calc_score += pairs_solved / num_pairs
            if pairs_solved == num_pairs:
                fully_solved_tasks.append(task_id)

    percentage = 100 * (calc_score / max_total_score) if max_total_score > 0 else 0.0
    print(f"Official ARC style scoring: {calc_score}/{max_total_score} ({percentage}%)")
    print(f"Fully correct tasks ({len(fully_solved_tasks)}):")
    for task_id in fully_solved_tasks:
        print(task_id)

    return {
        "score": calc_score,
        "max_score": max_total_score,
        "percentage": percentage,
        "fully_solved_tasks": fully_solved_tasks,
    }


# =============================================================================
# Memory Cleanup
# =============================================================================

def cleanup_memory(
    globals_dict: Optional[Dict[str, object]] = None,
    names: Sequence[str] = ("model", "dataset", "dataloader", "optimizer", "scheduler"),
) -> None:
    """Free common training objects and clear torch caches for inference."""
    if globals_dict is not None:
        for name in names:
            if name in globals_dict:
                del globals_dict[name]

    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print(f"GPU cleaned. Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# =============================================================================
# Oracle and top-2 scoring
# =============================================================================
# Every scored full decode writes oracle_score.json beside submission.json. Three metrics share
# score_arc_submission's aggregation (mean correctness over a task's test outputs, summed over
# tasks): "top2" = either voted output correct (the competition score); "oracle" = any of the N
# generated candidates (all augmentations, inverse-transformed) correct, before vote filters;
# "oracle_after_filters" = any candidate surviving the vote filters correct. References are read
# only after generation, and the submission's voting is unchanged.


METRICS = ("top2", "oracle", "oracle_after_filters")


def _keyed(selections: Iterable) -> Dict[tuple, object]:
    return {(s.task_id, s.original_pair_index): s for s in selections}


def oracle_report(raw_selections: Sequence, voted_selections: Sequence, solutions: dict,
                  expected_candidates: Optional[int] = None,
                  extra_top2: Optional[Dict[str, Sequence]] = None) -> dict:
    """Score raw (unfiltered) candidates, filtered candidates and the submitted top-2 against solutions.

    ``extra_top2`` names other selections (e.g. plain votes before guards or rescoring) whose
    top-2 is scored alongside, as additional metrics."""
    raw, voted = _keyed(raw_selections), _keyed(voted_selections)
    extras = {name: _keyed(selections) for name, selections in (extra_top2 or {}).items()}
    names = METRICS + tuple(extras)
    per_task = {name: {} for name in names}
    pairs = []
    for task, outputs in solutions.items():
        hits = {name: [] for name in names}
        for index, truth in enumerate(outputs):
            a, f = raw.get((task, index)), voted.get((task, index))
            raw_hit = bool(a) and any(c["grid"] == truth for c in a.ranked_candidates)
            filtered_hit = bool(f) and any(c["grid"] == truth for c in f.ranked_candidates)
            top2_hit = bool(f) and any(grid == truth for grid in f.selected_outputs[:2])
            hits["oracle"].append(raw_hit)
            hits["oracle_after_filters"].append(filtered_hit)
            hits["top2"].append(top2_hit)
            extra_hits = {}
            for name, keyed in extras.items():
                e = keyed.get((task, index))
                extra_hits[name] = bool(e) and any(grid == truth for grid in e.selected_outputs[:2])
                hits[name].append(extra_hits[name])
            correct = next((c for c in a.ranked_candidates if c["grid"] == truth), None) if a else None
            pairs.append({
                "task_id": task, "pair_index": index,
                "num_generated": a.num_generated if a else 0,
                "num_unique": len(a.ranked_candidates) if a else 0,
                "oracle_hit": raw_hit, "oracle_after_filters_hit": filtered_hit, "top2_hit": top2_hit,
                **{f"{name}_hit": hit for name, hit in extra_hits.items()},
                "correct_candidate_votes": correct["count"] if correct else 0,
                "correct_candidate_rank_after_filters": next(
                    (j + 1 for j, c in enumerate(f.ranked_candidates) if c["grid"] == truth), None) if f else None,
            })
        for name, values in hits.items():
            per_task[name][task] = sum(values) / len(values) if values else 0.0
    maximum = len(solutions)
    report = {
        "definition": "oracle: any generated candidate correct; oracle_after_filters: any candidate surviving "
                      "the vote filters (input copies, symbolic guards when on) correct; top2: either "
                      "submitted output correct; top2_votes: either plain-vote output correct",
        "aggregation": "mean correctness over each task's test outputs, summed over tasks (score_arc_submission)",
        "expected_candidates_per_test_output": expected_candidates,
        "num_tasks": maximum, "num_test_outputs": len(pairs), "metrics": {}, "pairs": pairs,
    }
    for name, values in per_task.items():
        score = sum(values.values())
        report["metrics"][name] = {
            "score": score, "max_score": maximum, "percentage": 100 * score / maximum if maximum else 0.0,
            "fully_solved_tasks": [task for task, value in values.items() if value == 1],
        }
    return report


def write_oracle_report(directory, raw_selections, voted_selections, solutions_path,
                        expected_candidates: Optional[int] = None,
                        extra_top2: Optional[Dict[str, Sequence]] = None) -> dict:
    solutions = json.loads(Path(solutions_path).read_text())
    report = oracle_report(raw_selections, voted_selections, solutions, expected_candidates, extra_top2)
    path = Path(directory) / "oracle_score.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    m = report["metrics"]
    print(f"Top-2: {m['top2']['score']:g}/{m['top2']['max_score']} ({m['top2']['percentage']:.2f}%) | "
          f"Oracle@{expected_candidates}: {m['oracle']['score']:g} ({m['oracle']['percentage']:.2f}%) | "
          f"after filters: {m['oracle_after_filters']['percentage']:.2f}%"
          + "".join(f" | {name}: {m[name]['score']:g} ({m[name]['percentage']:.2f}%)" for name in m if name not in METRICS),
          flush=True)
    return report


def write_candidate_dump(directory, raw_selections, test_inputs: Optional[dict] = None,
                         expected_candidates: Optional[int] = None, annotations: Optional[dict] = None) -> Path:
    """Every distinct de-augmented candidate with its vote count, per test output (candidates.json.gz).

    Lets candidates from several models be pooled or re-ranked offline without decoding again.
    `input_copy` marks candidates equal to the test input (the voting filter drops them). With
    ``annotations`` ({(task, pair): {grid tuple: fields}}), `guard` names the symbolic guard that
    rejects a candidate and `log_likelihood` is its rescored summed log-likelihood."""
    test_inputs = test_inputs or {}
    annotations = annotations or {}
    pairs = []
    for selection in raw_selections:
        original = test_inputs.get((str(selection.task_id), int(selection.original_pair_index)))
        notes = annotations.get((str(selection.task_id), int(selection.original_pair_index)), {})
        pairs.append({
            "task_id": selection.task_id, "pair_index": selection.original_pair_index,
            "num_generated": selection.num_generated, "num_valid": selection.num_valid,
            "discarded_non_rectangular": selection.discarded_non_rectangular,
            **({"discarded_scale_mismatch": selection.discarded_scale_mismatch}
               if getattr(selection, "discarded_scale_mismatch", 0) else {}),
            "candidates": [{"grid": entry["grid"], "count": entry["count"],
                            "input_copy": original is not None and entry["grid"] == original,
                            **notes.get(tuple(tuple(row) for row in entry["grid"]), {})}
                           for entry in selection.ranked_candidates],
        })
    path = Path(directory) / "candidates.json.gz"
    with gzip.open(path, "wt") as handle:
        json.dump({"format": 1, "expected_candidates_per_test_output": expected_candidates, "pairs": pairs}, handle)
    return path


def oracle_summary_fields(submission_path) -> dict:
    """Oracle fields for score.json, read from the oracle report beside a submission (if any)."""
    path = Path(submission_path).with_name("oracle_score.json")
    if not path.is_file():
        return {}
    metrics = json.loads(path.read_text())["metrics"]
    return {
        "top2_score": metrics["top2"]["score"], "top2_percentage": metrics["top2"]["percentage"],
        "oracle_score": metrics["oracle"]["score"], "oracle_percentage": metrics["oracle"]["percentage"],
        "oracle_after_filters_percentage": metrics["oracle_after_filters"]["percentage"],
        "oracle_fully_solved_tasks": len(metrics["oracle"]["fully_solved_tasks"]),
        **{f"{name}_score": metrics[name]["score"] for name in metrics if name not in METRICS},
        **{f"{name}_percentage": metrics[name]["percentage"] for name in metrics if name not in METRICS},
    }
