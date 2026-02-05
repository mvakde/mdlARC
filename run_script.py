import sys
import argparse
import json
from pathlib import Path
from time import perf_counter

# Choose whether scoring is enabled or not
SCORE_RESULTS = True

# 1. SETUP PATHS AND IMPORT MODULES
SRC_DIR = Path.cwd() / "src"
sys.path.insert(0, str(SRC_DIR))

import utils
import train
import build
import evaluate
import common
import tinytransformer
print("Modules imported successfully.")

# 2. DEFINE CONFIG
args_dict = {
    "name": "submission_run",
    "data_path": Path("assets/challenges.json"),
    "train_log_file": Path("runs/training_log.txt"),
    "save_path": Path("runs/tiny.pt"),
    "checkpoint_path": None, 
    "checkpoint_epochs": [], # No intermediate saves needed for short run
    
    # Hyperparameters
    "epochs": 240, 
    "batch_size": 32,
    "gradient_accumulation_steps": 1,
    "do_validate": False,
    "val_batch_size": 70,

    "enable_aug": True,
    "max_augments": 80,
    "enable_color_aug": True,
    "color_apply_to_test": True,
    "enable_dihedral_aug": True,
    "dihedral_apply_to_test": True,

    "optimizer": "normuon",
    "normuon_lr": 1.66e-3,
    "normuon_momentum": 0.95,
    "normuon_beta2": 0.95,
    "adamw_lr": 3e-4,

    "warmup_pct": 0.02,
    "wsd_decay_start_pct": 0.8,
    "lr_floor": 0.0,

    "weight_decay": 0.1,
    "attention_weight_decay": 0.01,
    "token_embedding_weight_decay": 0.01,
    "task_embedding_weight_decay": 0.01,

    "grad_clip": 1.0,
    "dropout": 0.1,
    "seed": 42,

    # Architecture
    "d_model": 768,
    "n_heads": 12,
    "d_ff": 3072,
    "n_layers": 8,

    "inference_temperature": None,
    "inference_top_k": None,
}
cfg = argparse.Namespace(**args_dict) # Convert dictionary to Namespace
Path("runs").mkdir(parents=True, exist_ok=True) # Create runs dir
TRAIN_LOG_FILE = Path(cfg.train_log_file)
TRAIN_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
with TRAIN_LOG_FILE.open("w") as handle:
    handle.write("Per-task training log\n")

# Discover task ids once (avoids full dataset build).
challenges = common.load_challenges(cfg.data_path)
task_ids = sorted(challenges.keys())
print(f"Discovered {len(task_ids)} tasks.")

# Aggregate per-task submissions into one final submission.
combined_submission = {}

base_batch_size = cfg.batch_size
data_path = Path(cfg.data_path)


def _reset_model_weights(model):
    common.set_seed(cfg.seed)
    model.apply(model._init_weights)
    for module in model.modules():
        if isinstance(module, tinytransformer.RMSNorm):
            module.weight.data.fill_(1.0)
    model._loaded_checkpoint = None


def _build_dataloader_for_dataset(dataset):
    use_aug = bool(getattr(cfg, "enable_aug", False))
    augmentor = None
    if use_aug:
        max_augments = int(getattr(cfg, "max_augments", 0) or 0)
        augmentor = common.build_augmentor(
            dataset.examples,
            dataset.task_input_colors,
            max_augments=max_augments,
            enable_color=bool(getattr(cfg, "enable_color_aug", False)),
            enable_dihedral=bool(getattr(cfg, "enable_dihedral_aug", False)),
            seed=cfg.seed,
            color_apply_to_test_split=bool(getattr(cfg, "color_apply_to_test", False)),
            dihedral_apply_to_test_split=bool(getattr(cfg, "dihedral_apply_to_test", False)),
        )
        dataset.augmentor = augmentor

    collate_augment_selector = augmentor.select_for_example if augmentor is not None else None
    dataloader = common.create_dataloader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=not getattr(cfg, "eval_only", False),
        augment_selector=collate_augment_selector,
    )
    if augmentor is not None:
        dataloader.augmentor = augmentor
    return dataloader

model = None
device = None

# 3-5. PER-TASK TRAIN + EVAL
for index, task_id in enumerate(task_ids, start=1):
    print(f"\n{'=' * 60}")
    print(f"TASK {index}/{len(task_ids)}: {task_id}")
    print(f"{'=' * 60}\n")

    with TRAIN_LOG_FILE.open("a") as handle:
        handle.write(f"\n=== TASK {task_id} ===\n")

    # Build per-task dataset (train+test, outputs hidden for test)
    dataset = common.ARCExampleDataset(
        json_path=cfg.data_path,
        splits=("train", "test"),
        include_outputs=True,
        task_whitelist=[task_id],
    )

    # Adjust batch size to avoid zero-step training
    task_batch_size = min(base_batch_size, len(dataset))
    if task_batch_size <= 0:
        task_batch_size = 1
    cfg.batch_size = task_batch_size

    # Per-task checkpoint path
    cfg.save_path = Path(f"runs/tiny_{task_id}.pt")
    cfg.checkpoint_path = None

    if model is None:
        print("Building model and data...")
        model, _, dataloader, device, data_path = build.build_model_and_data(
            cfg, reuse_dataset=dataset
        )
        model._force_full_seq_len = True
    else:
        _reset_model_weights(model)
        dataloader = _build_dataloader_for_dataset(dataset)

    print("Starting Training...")
    t_start = perf_counter()
    train.train_model(
        cfg,
        model=model,
        dataloader=dataloader,
        dataset=dataset,
        device=device,
        data_path=data_path,
    )
    print(f"Training finished in {perf_counter() - t_start:.2f}s")

    # EVALUATE / INFERENCE
    print("Starting Evaluation...")
    eval_result = evaluate.run_evaluation(
        cfg,
        run_name="_per_task_eval",
        max_augments=cfg.max_augments,
        data_path=cfg.data_path,
        checkpoint_path=None,
        batch_size=100,
        splits=["test"],
        task_ids=[task_id],
        model=model,
        dataset=dataset,
        device=device,
    )
    task_submission_path = Path(f"runs/{eval_result[0]}/submission.json")
    with task_submission_path.open("r") as handle:
        task_submission = json.load(handle)
    combined_submission.update(task_submission)
    print("Evaluation complete. submission.json updated in memory.")

    # Delete per-task checkpoint to save space
    if cfg.save_path.exists():
        cfg.save_path.unlink()

# 6. FINAL COMBINED SUBMISSION
final_run_dir = Path("runs/submission_eval")
final_run_dir.mkdir(parents=True, exist_ok=True)
SUBMISSION_FILE = final_run_dir / "submission.json"
with SUBMISSION_FILE.open("w") as handle:
    json.dump(combined_submission, handle)
print("Final combined submission.json generated.")

# 6. RESULTS: score the results (if enabled), then visualise
if SCORE_RESULTS: # scoring, if enabled
    SOLUTIONS_FILE = Path("assets/solutions.json")
    score = utils.score_arc_submission(SOLUTIONS_FILE, SUBMISSION_FILE)
    utils.visualize_submissions(SUBMISSION_FILE, SOLUTIONS_FILE, mode="!")
else:
    utils.visualize_submissions(SUBMISSION_FILE, mode="submission")
