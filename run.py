"""
## Breaking Pareto Frontier

Steps to reproduce:

1. Upload this script to google colab or modal

2. (optional) If you want to save checkpoints and results, mount your google drive (colab) / your volume (modal)

3. Click run-all





Notes:

1. The config in this notebook has been tuned for an 80GB A100

2. Actual results were obtained by running this exact file in 2 phases.

    - Training on a 40GB A100

    - Take the final checkpoint, and run the inference on an 80GB A100



This will work on smaller GPUs too, but will take longer to train

For very constrained environments, disable the "do_validate" flag. This avoids checking the validation loss every epoch
"""

from pathlib import Path
import argparse
import importlib
import gc
import sys
import json
from datetime import datetime
from time import perf_counter
import shutil

import torch
import matplotlib.pyplot as plt

import utils
import tinytransformer
import train
import inference


# root_folder, mount_folder = "app", "mnt/transformer-arc"  # for modal
# root_folder, mount_folder = "content", "content/drive/MyDrive"  # for colab
root_folder, mount_folder = ".", "."  # for local


# Helper class for logging to file and console
class TeeLogger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def build_config():
    """Build and return the configuration namespace."""
    args = {
        # run config
        "num_workers": 0,
        "device": "cuda",  # 'cuda' | 'mps' | 'cpu'
        "do_validate": True,
        "name": "arc1-30M-vvwide-bs32-101ep-100color-ccdb-18dec0430",  # download file name
        "GPU": "A100-noaugreg",
        # paths - must pass as Path("<path_to_dir>")
        "train_log_file": Path("runs/training_log.txt"),
        "save_path": Path("runs/tiny.pt"),
        "checkpoint_path": None,  # Path("runs/tiny.pt"),  # or None to start from scratch
        # "data_path": Path("assets/script-tests/grouped-tasks-00576224/challenges.json"),
        # "data_path": Path("assets/ARC-2/grouped-tasks/concept_plus_combined_dihedral_train/challenges.json"),  # this dataset has dihedral augments only on the train sequences (use this for training)
        "data_path": Path(
            "assets/ARC-1/grouped-tasks/concept_plus_combined_dihedral_both/challenges.json"
        ),  # this has dihedral augments on train and test sequences (only use for evaluation)
        # hyperparameters
        "epochs": 101,
        "batch_size": 32,
        "val_batch_size": 300,
        "enable_color_aug_train": True,
        "max_color_augments_train": 100,
        "color_aug_seed": 42,
        "lr": 3e-4,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "dropout": 0.1,
        "seed": 42,
        # Model Architecture
        "d_model": 768,  # 128, 256, 512, 768 | 128, 384, 640
        "n_heads": 12,  # 4, 8, 8/16, 12 | 4, 12, 10
        "d_ff": 3072,  # 512, 1024, 2048, 3072 | 512, 1536, 2560
        "n_layers": 4,  # 4, 6, 16, 16 | 24, 28, 24
        # Loss masking
        "mask_input_loss": False,  # If True, only compute loss on output tokens (mask input loss)
        # Visibility toggles
        "log_train_strings": False,
        "log_train_limit": 10,
        "log_inference_prompt": False,
    }
    return argparse.Namespace(**args), args


def setup_directories(args):
    """Create necessary directories and save config."""
    runs_dir = Path("runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    with (runs_dir / "config.txt").open("w") as f:
        for k, v in args.items():
            f.write(f"{k}: {v}\n")


def run_training(cfg, model, dataset, dataloader, device, data_path):
    """Run the training phase."""
    # Training only

    t_start = perf_counter()

    # ---
    # direct
    train.train_model(
        cfg,
        model=model,
        dataloader=dataloader,
        dataset=dataset,
        device=device,
        data_path=data_path,
    )

    # # periodic checkpointing
    # cfg.save_path = Path(f"runs/tiny-{cfg.epochs}.pt")
    # for i in range(3):
    #   if i != 0:
    #     cfg.checkpoint_path = cfg.save_path
    #     cfg.save_path = Path(f"runs/tiny-{cfg.epochs*(i+1)}.pt")
    #   train.train_model(cfg, model=model, dataloader=dataloader, dataset=dataset, device=device, data_path=data_path)
    # ---

    t_duration = perf_counter() - t_start
    print(f"Training took {t_duration:.2f}s")

    with open(Path("runs/timing.txt"), "w") as f:
        f.write(f"Training: {t_duration:.4f} s\n")


def cleanup_memory():
    """Clean up memory to run inference."""
    # cleaning up memory to run inference

    # 1. Delete global references to free memory
    # Deleting 'model' ensures Cell 4 reloads a fresh instance from the checkpoint,
    # preventing memory fragmentation or leftover gradients from training.
    for name in ["model", "dataset", "dataloader", "optimizer", "scheduler"]:
        if name in globals():
            del globals()[name]

    # 2. Reset compiled graph caches (crucial if torch.compile was used)
    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()

    # 3. Force garbage collection and clear GPU memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(
            f"GPU cleaned. Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        )


def save_checkpoint(cfg, root_folder, mount_folder):
    """Save data immediately in case eval fails."""
    # save data immediately in case eval fails

    # Reusable paths (keep these for cell 2)
    SRC_DIR = Path(f"{root_folder}/runs") if root_folder != "." else Path("runs")
    ZIP_BASE = (
        Path(f"{root_folder}") if root_folder != "." else Path(".")
    ) / f"runs-{cfg.name}"  # no .zip here
    LOCAL_ZIP = ZIP_BASE.with_suffix(".zip")
    MOUNT_DIR = Path(f"{mount_folder}") if mount_folder != "." else Path(".")

    timestamp = datetime.now().strftime("%d%m%y-%H%M%S")
    LAST_DRIVE_ZIP = MOUNT_DIR / f"runs-{cfg.name}-{timestamp}.zip"

    # Clean local zip if it exists
    LOCAL_ZIP.unlink(missing_ok=True)

    print(f"Zipping {SRC_DIR} ...")
    shutil.make_archive(str(ZIP_BASE), "zip", str(SRC_DIR))

    print(f"Copying to Drive: {LAST_DRIVE_ZIP}")
    shutil.copy2(str(LOCAL_ZIP), str(LAST_DRIVE_ZIP))

    # Optional: delete local zip to save space
    LOCAL_ZIP.unlink(missing_ok=True)

    return SRC_DIR, ZIP_BASE, LOCAL_ZIP, MOUNT_DIR, LAST_DRIVE_ZIP


def run_evaluation_pipeline(
    cfg,
    args,
    run_name,
    max_color_augments,
    dataset_path,
    device,
    eval_batch_size,
    splits,
    checkpoint_path,
    solutions_present,
    eval_task_ids,
    log_correct_grids,
    model_ref=None,
):
    """Run the evaluation pipeline for a single configuration."""
    print(f"\n{'=' * 60}")
    print(f"STARTING PIPELINE: {run_name} (Color Augs: {max_color_augments})")
    print(f"{'=' * 60}\n")

    # 1. Setup Directories
    base_run_dir = Path("runs") / run_name
    base_run_dir.mkdir(parents=True, exist_ok=True)

    eval_log_path = base_run_dir / "eval_log.txt"
    aaivr_log_path = base_run_dir / "aaivr.txt"
    submission_path = base_run_dir / "submission.json"

    # 2. Update Config
    cfg.checkpoint_path = checkpoint_path
    cfg.data_path = dataset_path
    cfg.enable_color_aug_eval = max_color_augments > 0
    cfg.max_color_augments_eval = max_color_augments

    # 3. Build/Rebuild Model & Data
    # We rebuild the dataloader every time to handle the different color augmentation settings
    print("Building model and dataloader for config...")

    # Load checkpoint explicitly to pass to build function
    checkpoint = torch.load(
        cfg.checkpoint_path, map_location=device, weights_only=False
    )

    # Check if model exists to reuse weights, else create
    model = model_ref
    if model is not None:
        model.load_state_dict(
            checkpoint["model_state"] if "model_state" in checkpoint else checkpoint,
            strict=False,
        )
        model.eval()
        # Rebuild only dataset/loader
        _, dataset, dataloader, device, _ = train.build_model_and_data(
            cfg, checkpoint=checkpoint
        )
    else:
        model, dataset, dataloader, device, _ = train.build_model_and_data(cfg)

    # 4. Run Inference (Logic from old Cell 3)
    def log_eval(msg):
        print(msg)
        with open(eval_log_path, "a") as f:
            f.write(msg + "\n")

    color_mappings_eval = None
    color_apply_fn = None
    if cfg.enable_color_aug_eval and cfg.max_color_augments_eval > 0:
        color_seed = cfg.color_aug_seed or cfg.seed
        color_mappings_eval = utils.generate_color_mapping_tensors(
            cfg.max_color_augments_eval, color_seed
        )
        color_apply_fn = lambda split: True

    evaluation = inference.evaluate_model_on_dataset(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=eval_batch_size,
        log_prompts=args["log_inference_prompt"],
        splits=splits,
        color_mappings=color_mappings_eval,
        color_apply_fn=color_apply_fn,
        task_ids=eval_task_ids,
        include_targets=solutions_present,
    )

    # Log Inference Stats
    log_eval(f"\n-- {cfg.epochs}ep {max_color_augments}color --\n")
    data_path_str = str(cfg.data_path)
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

            # Determine if THIS split has dihedral augmentations
            # Train is augmented if "dihedral" is anywhere in the name
            # Test is augmented ONLY if "dihedral_both" is in the name
            is_dihedral_split = (split == "train" and "dihedral" in data_path_str) or (
                split == "test" and "dihedral_both" in data_path_str
            )

            correct_results = summary.get("fully_correct_results", [])
            for res in correct_results:
                raw_idx = res.get("pair_index", 0)

                # Decode indices based on split properties
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

    # 5. Run AAIVR (Logic from old Cell 4)
    print(f"Running AAIVR for {run_name}...")

    # Redirect stdout for AAIVR logging
    original_stdout = sys.stdout
    sys.stdout = TeeLogger(str(aaivr_log_path))

    aaivr_results = []
    try:
        test_results = evaluation.get("test", {}).get("results", [])
        dataset_has_dihedral_augments = "dihedral_both" in str(cfg.data_path)

        if test_results:
            aaivr_results = utils.run_aaivr_on_results(
                test_results,
                is_dihedral_augmented=dataset_has_dihedral_augments,
                color_aug_seed=cfg.color_aug_seed,
                max_color_augments=cfg.max_color_augments_eval,
            )

            # Print Stats (will go to console + aaivr.txt)
            utils.summarize_aaivr_pass_at_k(aaivr_results)
            if aaivr_results:
                tasks_map = {}
                for res in aaivr_results:
                    if res.task_id not in tasks_map:
                        tasks_map[res.task_id] = []
                    tasks_map[res.task_id].append(res)

                arc_score = 0.0
                total_tasks = len(tasks_map)

                for t_id, pairs in tasks_map.items():
                    n_pairs = len(pairs)
                    if n_pairs > 0:
                        n_solved = sum(1 for p in pairs if p.pass_at_k)
                        arc_score += n_solved / n_pairs

                max_score = total_tasks
                pct = (arc_score / max_score * 100) if max_score > 0 else 0.0
                print(
                    f"Official ARC style scoring: {arc_score:.2f}/{max_score} ({pct:.2f}%)"
                )
        else:
            print("No test results for AAIVR.")

    finally:
        # Always restore stdout
        if hasattr(sys.stdout, "terminal"):
            sys.stdout.close()
        sys.stdout = original_stdout

    # 6. Generate Submission (Logic from old Cell 5)
    print(f"Generating submission.json for {run_name}...")
    submission_data = {}
    temp_grouping = {}

    if aaivr_results:
        for item in aaivr_results:
            t_id = item.task_id
            p_idx = item.original_pair_index
            if t_id not in temp_grouping:
                temp_grouping[t_id] = {}

            top_grids = item.selected_outputs[:2]
            if not top_grids:
                top_grids = [[[0]]]  # Fallback

            pair_dict = {
                "attempt_1": top_grids[0],
                "attempt_2": top_grids[1] if len(top_grids) > 1 else top_grids[0],
            }
            temp_grouping[t_id][p_idx] = pair_dict

        for t_id, pairs_map in temp_grouping.items():
            sorted_indices = sorted(pairs_map.keys())
            submission_data[t_id] = [pairs_map[idx] for idx in sorted_indices]

    with open(submission_path, "w") as f:
        json.dump(submission_data, f)

    print(f"Finished {run_name}. Submission saved to {submission_path}")

    return model


def run_all_evaluations(cfg, args, device):
    """Run all evaluation configurations."""
    # Reload modules to pick up changes
    importlib.reload(tinytransformer)
    importlib.reload(inference)
    importlib.reload(utils)

    # Define your paths constants
    PATH_BOTH = Path(
        "assets/ARC-1/grouped-tasks/concept_plus_combined_dihedral_both/challenges.json"
    )
    PATH_TRAIN = Path(
        "assets/ARC-1/grouped-tasks/concept_plus_combined_dihedral_train/challenges.json"
    )

    # Config List: (Run Name, Max Color Augments, Dataset Path)
    EVAL_CONFIGS = [
        # ("eval_125color_both", 125, PATH_BOTH),
        ("eval_100color_both", 100, PATH_BOTH)
        # ("eval_10color_both", 10, PATH_BOTH),
        # ("eval_0color_both", 0, PATH_BOTH),
        # ("eval_0color_train", 0, PATH_TRAIN) # <--- Uses TRAIN path (No Geom TTA on Test)
    ]

    # Global settings shared across runs
    EVAL_BATCH_SIZE = 1300
    SPLITS = ["test"]
    CHECKPOINT_PATH = Path("runs/tiny.pt")
    SOLUTIONS_PRESENT = True
    EVAL_TASK_IDS = None  # Set to None to evaluate full dataset, or ["00576224", ...] for specific tasks
    LOG_CORRECT_GRIDS = False  # Print the actual grid, IDs, and augmentation indices for fully correct grids

    # --- Execute the Loop (Modified with Timing) ---
    timing_path = Path("runs/timing.txt")
    model = None

    for name, aug_count, d_path in EVAL_CONFIGS:  # <--- Unpack 3 items
        t_start = perf_counter()

        model = run_evaluation_pipeline(
            cfg,
            args,
            name,
            aug_count,
            d_path,
            device,
            EVAL_BATCH_SIZE,
            SPLITS,
            CHECKPOINT_PATH,
            SOLUTIONS_PRESENT,
            EVAL_TASK_IDS,
            LOG_CORRECT_GRIDS,
            model_ref=model,
        )

        t_duration = perf_counter() - t_start
        print(f"Run {name} took {t_duration:.2f}s")

        with open(timing_path, "a") as f:
            f.write(f"Evaluation {name}: {t_duration:.4f} s\n")

    print("\nAll evaluation runs completed.")


def save_final_results(cfg, root_folder, mount_folder, last_drive_zip=None):
    """Save final results after evaluation."""
    SRC_DIR = Path(f"{root_folder}/runs") if root_folder != "." else Path("runs")
    ZIP_BASE = (
        Path(f"{root_folder}") if root_folder != "." else Path(".")
    ) / f"runs-{cfg.name}"
    LOCAL_ZIP = ZIP_BASE.with_suffix(".zip")
    MOUNT_DIR = Path(f"{mount_folder}") if mount_folder != "." else Path(".")

    # If kernel restarted and LAST_DRIVE_ZIP isn't defined, fall back to "latest matching zip"
    if last_drive_zip is None:
        pattern = f"runs-{cfg.name}-*.zip"
        matches = sorted(
            MOUNT_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True
        )
        last_drive_zip = matches[0] if matches else None

    # Delete the previous zip on Drive (if any)
    if last_drive_zip and Path(last_drive_zip).exists():
        print(f"Deleting old Drive zip: {last_drive_zip}")
        Path(last_drive_zip).unlink()

    # Create a new timestamped zip
    timestamp = datetime.now().strftime("%d%m%y-%H%M%S")
    NEW_DRIVE_ZIP = MOUNT_DIR / f"runs-{cfg.name}-{timestamp}.zip"

    LOCAL_ZIP.unlink(missing_ok=True)

    print(f"Zipping UPDATED {SRC_DIR} ...")
    shutil.make_archive(str(ZIP_BASE), "zip", str(SRC_DIR))

    print(f"Copying NEW zip to Drive: {NEW_DRIVE_ZIP}")
    shutil.copy2(str(LOCAL_ZIP), str(NEW_DRIVE_ZIP))

    # Update pointer for the next run
    last_drive_zip = NEW_DRIVE_ZIP

    # Optional: delete local zip
    LOCAL_ZIP.unlink(missing_ok=True)

    return last_drive_zip


def run_visualization(cfg):
    """Run visualization of results."""
    # visualisation

    EVAL_SUB_FOLDER = "eval_100color_both"  # "eval_Ncolor" -> replace N

    submission_file = Path(f"runs/{EVAL_SUB_FOLDER}/submission.json")
    # solutions_file = Path("assets/ARC-1/grouped-tasks/evaluation/solutions.json")
    solutions_file = Path("assets/ARC-1/grouped-tasks/evaluation/solutions.json")

    if not submission_file.exists() or not solutions_file.exists():
        print(
            f"Error: Could not find one of the files:\n{submission_file}\n{solutions_file}"
        )
    else:
        # Load Data
        with open(submission_file, "r") as f:
            subs = json.load(f)
        with open(solutions_file, "r") as f:
            sols = json.load(f)

        print(f"Visualizing comparison for {len(subs)} tasks...")

        for task_id, attempts_list in subs.items():
            # Get Ground Truth (list of grids)
            if task_id not in sols:
                print(f"Warning: Task {task_id} not found in solutions.json")
                continue

            gt_grids = sols[task_id]
            print(gt_grids)
            for i, attempts in enumerate(attempts_list):
                if i >= len(gt_grids):
                    break

                # 1. Retrieve Grids
                gt = gt_grids[i]
                att1 = attempts.get("attempt_1")
                att2 = attempts.get("attempt_2")

                # 2. Check Correctness
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

                # 3. Visualize
                # Construct list: [Ground Truth, Attempt 1, Attempt 2]
                grids_to_plot = [gt]
                if att1 is not None:
                    grids_to_plot.append(att1)
                if att2 is not None:
                    grids_to_plot.append(att2)

                header = f"Task: {task_id} | Pair: {i} | Status: {status}"
                print(f"Plotting {header}")

                # utils.plot_grids handles the matplotlib figure creation
                try:
                    utils.plot_grids(grids_to_plot, title=header)
                except Exception as e:
                    print(f"Skipping plot for {task_id} due to error: {e}")


def main():
    """Main entry point for the training and evaluation pipeline."""
    # Build configuration
    cfg, args = build_config()

    # Setup directories
    setup_directories(args)

    # Build model and data
    model, dataset, dataloader, device, data_path = train.build_model_and_data(cfg)

    # Run training
    run_training(cfg, model, dataset, dataloader, device, data_path)

    # Cleanup memory
    cleanup_memory()

    # Save checkpoint
    SRC_DIR, ZIP_BASE, LOCAL_ZIP, MOUNT_DIR, LAST_DRIVE_ZIP = save_checkpoint(
        cfg, root_folder, mount_folder
    )

    # Run evaluations
    run_all_evaluations(cfg, args, device)

    # Save final results
    save_final_results(cfg, root_folder, mount_folder, LAST_DRIVE_ZIP)

    # Run visualization (optional - uncomment if needed)
    # run_visualization(cfg)

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


# --- Modal App Configuration ---
import modal


def modal_add_dep(img: modal.Image, local, remote, sync=False):
    remote = f"/app/{remote}"

    for d in ["pyproject.toml", "uv.lock"]:
        img = img.add_local_file(f"{local}/{d}", f"{remote}/{d}", copy=True)
    if sync:
        img = img.run_commands(f"cd {remote} && uv sync")
    return img.add_local_dir(local, remote, ignore=["*.venv"], copy=True)


img = modal.Image.debian_slim()
img = modal_add_dep(img, ".", ".", sync=True)
vol = {"/assets": modal.Volume.from_name("mdlarc-assets")}
app = modal.App("arc-agi", image=img, volumes=vol, include_source=False)


@app.function(gpu="H100")
def main_modal():
    main()


if __name__ == "__main__":
    main_modal.local()
