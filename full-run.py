import argparse
import os
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
AUGMENT_DIHEDRAL = True
DATASET_NAMES = ["arc1", "conceptarc"]
DATASET_SPLITS = ["train", "eval"]
WITH_SOLUTIONS = True
DATASET_CLEANUP = "none"  # "none" | "solutions" | other options supported by build_datasets.py

# Optional cleanup to mirror the notebook's sanitised environment step.
SANITIZE_ENV = False

# Training / evaluation flow
RUN_TRAINING = True
CLEANUP_BEFORE_EVAL = True
RUN_EVALUATION = True
RUN_VISUALIZATION = False
RUN_SCORING = True

# Archive runs/ (zip + copy) for persistence.
ENABLE_ARCHIVE = False
SAVE_ARCHIVE_BEFORE_EVAL = True
UPDATE_ARCHIVE_AFTER_EVAL = True

# Archive paths (used only if ENABLE_ARCHIVE=True).
PROJECT_ROOT = Path(__file__).resolve().parent
ROOT_FOLDER = str(PROJECT_ROOT.parent).lstrip("/")
MOUNT_FOLDER = str((PROJECT_ROOT / "runs_archive").resolve()).lstrip("/")

# Model + training config (mirrors interactive-run.ipynb).
ARGS = {
    # run config
    "num_workers": 0,
    "device": "cuda",  # "cuda" | "mps" | "cpu"
    "do_validate": False,
    "name": "arc1-cleanenv-30M-vvwide-bs32-101ep-100color-ccdb-18dec0430",
    "GPU": "A100-noaugreg",  # logging only
    # paths
    "train_log_file": Path("runs/training_log.txt"),
    "save_path": Path("runs/tiny.pt"),
    "checkpoint_path": None,  # Path("runs/tiny.pt") to resume
    "data_path": Path("assets/challenges_dihedral_both.json"),
    "dihedral_augmented": True,
    # hyperparameters
    "epochs": 101,
    "batch_size": 32,
    "val_batch_size": 300,
    "enable_color_aug_train": True,
    "max_color_augments_train": 100,
    "color_aug_seed": 42,
    "color_aug_seed_eval": None,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "dropout": 0.1,
    "seed": 42,
    # Model Architecture
    "d_model": 768,
    "n_heads": 12,
    "d_ff": 3072,
    "n_layers": 4,
    # Visibility toggles
    "log_train_strings": False,
    "log_train_limit": 10,
    "log_inference_prompt": False,
    "inference_temperature": None,
    "inference_top_k": None,
}

# Evaluation config
PATH_BOTH = ARGS["data_path"]
EVAL_CONFIGS = [
    ("eval_100color_both", 100, PATH_BOTH, True),
]
EVAL_BATCH_SIZE = 1300
EVAL_SPLITS = ["test"]
EVAL_CHECKPOINT_PATH = ARGS["save_path"]
EVAL_SOLUTIONS_PRESENT = False
EVAL_TASK_IDS = None  # None = full dataset
EVAL_LOG_CORRECT_GRIDS = False

# Visualization config
EVAL_SUB_FOLDER = "eval_100color_both"
VIS_MODE = "!"  # "!" = compare vs solutions, "submission" = attempts-only
VIS_SOLUTIONS_FILE = "assets/solutions.json"

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
    if AUGMENT_DIHEDRAL:
        _run_command(
            [sys.executable, "dataset_building_scripts/augment_dataset_dihedral.py"]
        )


def _sanitize_repo(project_root: Path) -> None:
    targets = [
        project_root / "interactive-run.ipynb",
        project_root / "clean-env-run.ipynb",
        project_root / "max-clean-env-run.ipynb",
        project_root / "dataset_building_scripts",
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
        archive_state = utils.save_run_archive(
            cfg.name, ROOT_FOLDER, MOUNT_FOLDER, globals_dict=globals()
        )

    if RUN_EVALUATION:
        evaluations.run_evaluation_configs(
            cfg,
            EVAL_CONFIGS,
            eval_batch_size=EVAL_BATCH_SIZE,
            splits=EVAL_SPLITS,
            checkpoint_path=EVAL_CHECKPOINT_PATH,
            include_targets=EVAL_SOLUTIONS_PRESENT,
            task_ids=EVAL_TASK_IDS,
            log_correct_grids=EVAL_LOG_CORRECT_GRIDS,
        )

    if RUN_EVALUATION and RUN_SCORING:
        utils.score_arc_submission(SCORE_SOLUTIONS_FILE, SCORE_SUBMISSION_FILE)

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
