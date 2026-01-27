import sys
import argparse
import pickle
from pathlib import Path
from time import perf_counter
import torch

# 1. SETUP PATHS (As per your notebook structure)
PROJECT_ROOT = Path.cwd()
# Assuming files are in src/ or root. Your notebook adds 'src', checking for that:
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import your modules
try:
    import utils
    import tinytransformer
    import train
    import build
    import evaluate
    print("Modules imported successfully.")
except ImportError as e:
    # Fallback if src isn't used in repo, purely root
    print(f"Import warning: {e}. Assuming modules are in CWD.")
    import train
    import build
    import evaluate

# 2. DEFINE CONFIG (Matching your interactive.ipynb exactly)
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
    "lr": 3e-4,

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
    "n_layers": 4,

    "inference_temperature": None,
    "inference_top_k": None,
}

# Convert dictionary to Namespace (mimicking argparse)
cfg = argparse.Namespace(**args_dict)

# Create runs dir
Path("runs").mkdir(parents=True, exist_ok=True)

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

# Force garbage collection before eval
utils.cleanup_memory(globals())

# Reload checkpoint we just saved (best practice to ensure we use saved state)
# However, evaluate.py loads it internally based on path.

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

print("Evaluation complete. submission.json generated.")