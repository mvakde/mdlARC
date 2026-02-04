import sys
import argparse
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

# 3. BUILD
print("Building model and data...")
model, dataset, dataloader, device, data_path = build.build_model_and_data(cfg)

# 4. TRAIN
print("Starting Training...")
t_start = perf_counter()
train.train_model(
    cfg,
    model=model,
    dataloader=dataloader,
    dataset=dataset,
    device=device,
    data_path=data_path
)
print(f"Training finished in {perf_counter() - t_start:.2f}s")

# 5. EVALUATE / INFERENCE
print("Starting Evaluation...")
utils.cleanup_memory(globals()) # Force garbage collection before eval
eval_result = evaluate.run_evaluation(
    cfg,
    run_name="submission_eval",
    max_augments=cfg.max_augments,        
    data_path=cfg.data_path,
    checkpoint_path=cfg.save_path,
    batch_size=100,
    splits=["test"],          
    task_ids=None,
)
SUBMISSION_FILE = Path(f"runs/{eval_result[0]}/submission.json")
print("Evaluation complete. submission.json generated.")

# 6. RESULTS: score the results (if enabled), then visualise
if SCORE_RESULTS: # scoring, if enabled
    SOLUTIONS_FILE = Path("assets/solutions.json")
    score = utils.score_arc_submission(SOLUTIONS_FILE, SUBMISSION_FILE)
    utils.visualize_submissions(SUBMISSION_FILE, SOLUTIONS_FILE, mode="!")
else:
    utils.visualize_submissions(SUBMISSION_FILE, mode="submission")
