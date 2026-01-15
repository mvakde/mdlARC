import argparse
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import List, Optional

# Python port of interactive-run.ipynb for non-notebook runs.

# ---------------------------
# Config (edit in-place)
# ---------------------------

# Dataset building
BUILD_DATASETS = True
DATASET_NAMES = ["arc1", "conceptarc"]
DATASET_SPLITS = ["train", "eval"]
WITH_SOLUTIONS = True
DATASET_CLEANUP = (
    "none"  # "none" | "solutions" | other options supported by build_datasets.py
)

# Optional cleanup to mirror the notebook's sanitised environment step.
SANITIZE_ENV = False

# Training / evaluation flow
RUN_TRAINING = True
CLEANUP_BEFORE_EVAL = True
RUN_EVALUATION = True
RUN_VISUALIZATION = False
RUN_SCORING = True
RUN_AAIVR_FLOW_VIS = False

# Archive runs/ (zip + copy) for persistence.
ENABLE_ARCHIVE = True
SAVE_ARCHIVE_BEFORE_EVAL = True
UPDATE_ARCHIVE_AFTER_EVAL = True

# Archive paths (used only if ENABLE_ARCHIVE=True).
PROJECT_ROOT = Path(__file__).resolve().parent

# Calculate ROOT_FOLDER dynamically.
# utils.py expects: /{root_folder}/mdlARC/runs
# This requires the current folder to be named "mdlARC".
try:
    _repo_index = PROJECT_ROOT.parts.index("mdlARC")
    ROOT_FOLDER = str(Path(*PROJECT_ROOT.parts[:_repo_index])).lstrip("/")
except ValueError:
    raise ValueError("The project folder must be named 'mdlARC' for archiving to work.")

# Use a local archive folder instead of system /mnt
# When you move to Modal, you will change this back to "mnt/mithil-arc"
MOUNT_FOLDER = str(PROJECT_ROOT / "archives").lstrip("/")

# Model + training config (mirrors interactive-run.ipynb).
ARGS = {
    # run config
    "device": "cuda",  # CUDA only
    "do_validate": True,
    "name": "arc1-37M-bs32-101ep-100color-ccdb",
    "GPU": "A100",  # logging only
    # paths
    "train_log_file": Path("runs/training_log.txt"),
    "save_path": Path("runs/tiny.pt"),
    "checkpoint_path": None,  # Path("runs/tiny.pt") to resume
    "checkpoint_epochs": None,  # int N for every N epochs, or list [5, 10, 25]
    "data_path": Path("assets/challenges.json"),
    # hyperparameters
    "epochs": 101,
    "batch_size": 32,
    "gradient_accumulation_steps": 1,
    "val_batch_size": 300,
    "enable_aug": True,
    "max_augments": 100,
    "enable_color_aug": True,
    "color_apply_to_test": True,
    "enable_dihedral_aug": True,
    "dihedral_apply_to_test": True,
    "optimizer": "normuon",  # "adamw" | "normuon"
    "normuon_lr": 0.02,
    "normuon_momentum": 0.95,
    "normuon_beta2": 0.95,
    "lr": 3e-4,
    "warmup_pct": 0.02,
    "wsd_decay_start_pct": 0.8,  # 1.0 = no decay (start at last epoch)
    "lr_floor": 0.01,
    "weight_decay": 0.01,
    "attention_weight_decay": 0.01,
    "token_embedding_weight_decay": 0.0,
    "task_embedding_weight_decay": 0.0,
    "grad_clip": 1.0,
    "dropout": 0.1,
    "seed": 42,
    # Model Architecture
    "d_model": 768,
    "n_heads": 12,
    "d_ff": 3072,
    "n_layers": 4,
    "inference_temperature": None,
    "inference_top_k": None,
}

# Evaluation config
PATH_BOTH = ARGS["data_path"]
# aug_count = max_augments (0 = no aug)
EVAL_CONFIGS = [("eval_100aug_both", 100, PATH_BOTH)]
EVAL_BATCH_SIZE = 900
EVAL_SPLITS = ["test"]
EVAL_CHECKPOINT_PATH = ARGS["save_path"]
EVAL_SOLUTIONS_PRESENT = False
EVAL_TASK_IDS = None  # None = full dataset
EVAL_LOG_CORRECT_GRIDS = False

# Visualization config
EVAL_SUB_FOLDER = EVAL_CONFIGS[0][0]
VIS_MODE = "!"  # "!" = compare vs solutions, "submission" = attempts-only
VIS_SOLUTIONS_FILE = "assets/solutions.json"

# AAIVR flow visualization config (disabled by default)
AAIVR_FLOW_CONFIG_INDEX = 0
AAIVR_FLOW_TASK_ID = None  # e.g., "00576224"
AAIVR_FLOW_TASK_INDEX = 0  # 0-based index in evaluation pipeline order
AAIVR_FLOW_INPUT_INDEX = 0  # base pair index before dihedral aug

# Scoring config (mirrors final notebook cell)
SCORE_SOLUTIONS_FILE = Path("assets/solutions.json")
SCORE_SUBMISSION_FILE = Path(f"runs/{EVAL_SUB_FOLDER}/submission.json")


def _run_command(cmd: List[str], cwd: Optional[Path] = None) -> None:
    display = " ".join(str(part) for part in cmd)
    print(f"[run] {display}")
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def _build_datasets() -> None:
    cmd = [
        sys.executable,
        "dataset_building_scripts/build_datasets.py",
        "--datasets",
        *DATASET_NAMES,
        "--splits",
        *DATASET_SPLITS,
    ]
    if WITH_SOLUTIONS:
        cmd.append("--with-solutions")
    if DATASET_CLEANUP:
        cmd.extend(["--cleanup", DATASET_CLEANUP])
    _run_command(cmd)


def _sanitize_repo(project_root: Path) -> None:
    targets = [
        project_root / "interactive-run.ipynb",
        project_root / "clean-env-run.ipynb",
        project_root / "max-clean-env-run.ipynb",
        project_root / "readme.md",
        project_root / "img",
    ]
    for path in targets:
        if path.is_dir():
            shutil.rmtree(path)
            print(f"[sanitize] removed dir {path}")
        elif path.exists():
            path.unlink()
            print(f"[sanitize] removed file {path}")


def _prepare_environment() -> None:
    os.chdir(PROJECT_ROOT)
    src_dir = PROJECT_ROOT / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def main() -> None:
    _prepare_environment()

    if BUILD_DATASETS:
        _build_datasets()

    if SANITIZE_ENV:
        _sanitize_repo(PROJECT_ROOT)

    import evaluations
    import train
    import utils

    cfg = argparse.Namespace(**ARGS)

    runs_dir = Path("runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    with (runs_dir / "config.txt").open("w") as handle:
        for key, value in ARGS.items():
            handle.write(f"{key}: {value}\n")

    model = dataset = dataloader = device = data_path = None
    if RUN_TRAINING:
        model, dataset, dataloader, device, data_path = train.build_model_and_data(cfg)
        t_start = perf_counter()
        train.train_model(
            cfg,
            model=model,
            dataloader=dataloader,
            dataset=dataset,
            device=device,
            data_path=data_path,
        )
        t_duration = perf_counter() - t_start
        print(f"Training took {t_duration:.2f}s")
        with (runs_dir / "timing.txt").open("w") as handle:
            handle.write(f"Training: {t_duration:.4f} s\n")

    if CLEANUP_BEFORE_EVAL and RUN_TRAINING:
        del model, dataset, dataloader
        utils.cleanup_memory()

    archive_state = None
    if ENABLE_ARCHIVE and SAVE_ARCHIVE_BEFORE_EVAL:
        Path(f"/{MOUNT_FOLDER}").mkdir(parents=True, exist_ok=True)
        print(f"Archiving to local folder: /{MOUNT_FOLDER}")
        archive_state = utils.save_run_archive(
            cfg.name, ROOT_FOLDER, MOUNT_FOLDER, globals_dict=globals()
        )

    eval_results = None
    if RUN_EVALUATION:
        eval_results = evaluations.run_evaluation_configs(
            cfg,
            EVAL_CONFIGS,
            eval_batch_size=EVAL_BATCH_SIZE,
            splits=EVAL_SPLITS,
            checkpoint_path=EVAL_CHECKPOINT_PATH,
            include_targets=EVAL_SOLUTIONS_PRESENT,
            task_ids=EVAL_TASK_IDS,
            log_correct_grids=EVAL_LOG_CORRECT_GRIDS,
        )
        if eval_results:
            eval_results_path = runs_dir / "eval_results.pkl"
            eval_results_path.write_bytes(pickle.dumps(eval_results))
            print(f"Saved eval_results to {eval_results_path}")

    if RUN_EVALUATION and RUN_SCORING:
        utils.score_arc_submission(SCORE_SOLUTIONS_FILE, SCORE_SUBMISSION_FILE)

    if RUN_AAIVR_FLOW_VIS:
        if not eval_results:
            print("AAIVR flow visualization requires eval_results; enable RUN_EVALUATION.")
        else:
            import aaivr

            cfg_idx = AAIVR_FLOW_CONFIG_INDEX
            if cfg_idx < 0 or cfg_idx >= len(eval_results):
                raise ValueError(
                    f"AAIVR_FLOW_CONFIG_INDEX {cfg_idx} is out of range."
                )
            eval_data = eval_results[cfg_idx][1]
            test_results = eval_data.get("test", {}).get("results", [])
            eval_config = EVAL_CONFIGS[cfg_idx]
            dataset_path = eval_config[2]
            dihedral_enabled = False
            color_mappings_by_task = None

            aug_ctx = eval_data.get("_aug", {})
            if aug_ctx:
                color_mappings_by_task = aug_ctx.get("color_mappings_by_split", {}).get(
                    "test"
                )
                dihedral_enabled = aug_ctx.get("dihedral_augmented_by_split", {}).get(
                    "test", False
                )

            aaivr.visualize_aaivr_flow(
                test_results,
                dataset_path=dataset_path,
                input_index=AAIVR_FLOW_INPUT_INDEX,
                task_id=AAIVR_FLOW_TASK_ID,
                task_index=AAIVR_FLOW_TASK_INDEX,
                is_dihedral_augmented=dihedral_enabled,
                color_mappings_by_task=color_mappings_by_task,
            )

    if ENABLE_ARCHIVE and UPDATE_ARCHIVE_AFTER_EVAL:
        Path(f"/{MOUNT_FOLDER}").mkdir(parents=True, exist_ok=True)
        archive_state = utils.update_run_archive(
            cfg.name, ROOT_FOLDER, MOUNT_FOLDER, globals_dict=globals()
        )

    if RUN_VISUALIZATION:
        utils.visualize_eval_submissions(
            EVAL_SUB_FOLDER, mode=VIS_MODE, solutions_file=VIS_SOLUTIONS_FILE
        )

    if archive_state:
        print(f"Archive saved to: {archive_state.last_drive_zip}")


if __name__ == "__main__":
    main()
