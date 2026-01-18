"""Utility functions for visualization, scoring, and memory cleanup.

For tokenization, dataset classes, and other shared code, see common.py.
"""

import gc
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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
    "#0074D9",  # 1 blue
    "#FF4136",  # 2 red
    "#2ECC40",  # 3 green
    "#FFDC00",  # 4 yellow
    "#AAAAAA",  # 5 gray
    "#F012BE",  # 6 fuchsia
    "#FF851B",  # 7 orange
    "#7FDBFF",  # 8 aqua
    "#B10DC9",  # 9 purple
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
# Re-exports from common.py for backward compatibility
# =============================================================================
# These allow existing code to import from utils without changes during transition

from common import (
    # Vocabulary
    VOCAB_SIZE,
    TOKEN_TO_ID,
    ID_TO_TOKEN,
    START_TOKEN_ID,
    NEXT_LINE_TOKEN_ID,
    IO_SEPARATOR_TOKEN_ID,
    END_TOKEN_ID,
    MAX_SEQ_LEN,
    IGNORE_INDEX,
    # Grid/Token conversion
    grid_to_tokens,
    tokens_to_grid,
    encode_example,
    extract_output_tokens,
    tokens_to_string,
    split_grids_from_tokens,
    load_challenges,
    # Transforms
    apply_dihedral_transform,
    apply_inverse_dihedral_transform,
    apply_color_permutation_to_tokens,
    apply_color_permutation_to_grid,
    extract_task_input_colors,
    # 3D positions
    compute_positions_3d,
    # Dataset classes
    SequenceExample,
    ARCExampleDataset,
    LengthBucketBatchSampler,
    collate_examples,
    create_dataloader,
    # Validation
    is_rectangular_grid,
)
