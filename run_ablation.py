import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Sequence

import torch

# 1. SETUP PATHS AND IMPORT MODULES
SRC_DIR = Path.cwd() / "src"
sys.path.insert(0, str(SRC_DIR))

import build
import common
import evaluate
import train
import utils


def default_config() -> Dict[str, object]:
    return {
        "name": "per_task_ablation",
        "data_path": Path("assets/challenges.json"),
        "train_log_file": Path("runs/ablation/training_log.txt"),
        "save_path": None,
        "checkpoint_path": None,
        "checkpoint_epochs": [],
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
        "attention_dropout": None,
        "seed": 42,
        # Architecture
        "d_model": 768,
        "n_heads": 12,
        "d_ff": 3072,
        "n_layers": 8,
        "inference_temperature": None,
        "inference_top_k": None,
        # train logging
        "train_log_mode": "epoch",
        "log_location": "terminal",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run per-task-from-scratch ablation on ARC."
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="ablation_per_task",
        help="Base run folder under runs/.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("assets/challenges.json"),
        help="Path to ARC challenges.json.",
    )
    parser.add_argument(
        "--solutions-path",
        type=Path,
        default=Path("assets/solutions.json"),
        help="Path to ARC solutions.json for scoring.",
    )
    parser.add_argument(
        "--task-ids",
        nargs="*",
        default=None,
        help="Optional task IDs to run. If omitted, runs all tasks.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index into sorted task IDs.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Optional cap on number of tasks to run.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training epochs per task.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training batch size.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=100,
        help="Batch size for evaluation inference.",
    )
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="Skip final scoring even if solutions file exists.",
    )
    parser.add_argument(
        "--force-full-seq-len",
        action="store_true",
        default=True,
        help="Force decode cache length to model max_seq_len for compile stability.",
    )
    parser.add_argument(
        "--no-force-full-seq-len",
        dest="force_full_seq_len",
        action="store_false",
        help="Disable full-length decode cache forcing.",
    )
    return parser.parse_args()


def _build_training_dataloader(
    cfg: argparse.Namespace,
    dataset: common.ARCExampleDataset,
) -> torch.utils.data.DataLoader:
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
            seed=int(cfg.seed),
            color_apply_to_test_split=bool(getattr(cfg, "color_apply_to_test", False)),
            dihedral_apply_to_test_split=bool(
                getattr(cfg, "dihedral_apply_to_test", False)
            ),
        )
        dataset.augmentor = augmentor

    collate_augment_selector = (
        augmentor.select_for_example if augmentor is not None else None
    )
    dataloader = common.create_dataloader(
        dataset=dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        augment_selector=collate_augment_selector,
        use_length_bucketing=False,
    )
    if augmentor is not None:
        dataloader.augmentor = augmentor
    return dataloader


def _resolve_task_ids(
    all_task_ids: Sequence[str],
    explicit_task_ids: Optional[Sequence[str]],
    start_index: int,
    max_tasks: Optional[int],
) -> List[str]:
    if explicit_task_ids:
        known = set(all_task_ids)
        missing = [task_id for task_id in explicit_task_ids if task_id not in known]
        if missing:
            raise ValueError(f"Unknown task IDs: {missing}")
        task_ids = list(explicit_task_ids)
    else:
        task_ids = list(all_task_ids)
        if start_index < 0:
            raise ValueError("--start-index must be >= 0.")
        task_ids = task_ids[start_index:]
        if max_tasks is not None:
            if max_tasks <= 0:
                raise ValueError("--max-tasks must be positive when set.")
            task_ids = task_ids[:max_tasks]
    return task_ids


def main() -> None:
    cli = parse_args()

    args_dict = default_config()
    args_dict["name"] = cli.run_name
    args_dict["data_path"] = Path(cli.data_path)
    args_dict["train_log_file"] = Path("runs") / cli.run_name / "training_log.txt"
    if cli.epochs is not None:
        args_dict["epochs"] = int(cli.epochs)
    if cli.batch_size is not None:
        args_dict["batch_size"] = int(cli.batch_size)

    cfg = argparse.Namespace(**args_dict)
    run_dir = Path("runs") / cli.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.train_log_file.parent.mkdir(parents=True, exist_ok=True)

    challenges = common.load_challenges(cfg.data_path)
    all_task_ids = sorted(challenges.keys())
    task_ids = _resolve_task_ids(
        all_task_ids=all_task_ids,
        explicit_task_ids=cli.task_ids,
        start_index=int(cli.start_index),
        max_tasks=cli.max_tasks,
    )
    if not task_ids:
        raise ValueError("No tasks selected to run.")

    print(f"Running per-task ablation for {len(task_ids)} tasks.")

    combined_submission: Dict[str, List[Dict[str, object]]] = {}
    task_timings: List[Dict[str, object]] = []

    base_batch_size = int(cfg.batch_size)
    model = None
    device = None
    data_path = Path(cfg.data_path)
    initial_state = None

    ablation_start = perf_counter()
    for idx, task_id in enumerate(task_ids, start=1):
        print(f"\n{'=' * 60}")
        print(f"TASK {idx}/{len(task_ids)}: {task_id}")
        print(f"{'=' * 60}")

        common.set_seed(int(cfg.seed))
        train_dataset = common.ARCExampleDataset(
            json_path=cfg.data_path,
            splits=("train",),
            include_outputs=True,
            task_whitelist=[task_id],
        )

        if len(train_dataset) <= 0:
            raise RuntimeError(f"Task {task_id} has no train examples.")

        cfg.batch_size = max(1, min(base_batch_size, len(train_dataset)))

        if model is None:
            print("Building model...")
            model, _, dataloader, device, data_path = build.build_model_and_data(
                cfg, reuse_dataset=train_dataset
            )
            model._force_full_seq_len = bool(cli.force_full_seq_len)
            initial_state = {
                key: tensor.detach().clone()
                for key, tensor in model.state_dict().items()
            }
        else:
            if initial_state is None:
                raise RuntimeError("Initial model state was not captured.")
            model.load_state_dict(initial_state, strict=True)
            dataloader = _build_training_dataloader(cfg, train_dataset)

        task_train_start = perf_counter()
        train.train_model(
            cfg,
            model=model,
            dataloader=dataloader,
            dataset=train_dataset,
            device=device,
            data_path=data_path,
        )
        train_seconds = perf_counter() - task_train_start

        # Build eval dataset separately so test pairs are available while training
        # remains train-only.
        eval_dataset = common.ARCExampleDataset(
            json_path=cfg.data_path,
            splits=("train", "test"),
            include_outputs=True,
            task_whitelist=[task_id],
        )

        task_eval_start = perf_counter()
        eval_result = evaluate.run_evaluation(
            cfg,
            run_name=f"{cli.run_name}/tasks/{task_id}",
            max_augments=cfg.max_augments,
            data_path=cfg.data_path,
            checkpoint_path=None,
            batch_size=int(cli.eval_batch_size),
            splits=["test"],
            task_ids=[task_id],
            model=model,
            dataset=eval_dataset,
            device=device,
        )
        eval_seconds = perf_counter() - task_eval_start

        submission_path = Path(eval_result[2])
        with submission_path.open("r") as handle:
            task_submission = json.load(handle)
        combined_submission.update(task_submission)

        task_timings.append(
            {
                "task_id": task_id,
                "train_seconds": train_seconds,
                "eval_seconds": eval_seconds,
            }
        )
        print(
            f"Task {task_id} done. train={train_seconds:.2f}s eval={eval_seconds:.2f}s"
        )

    total_seconds = perf_counter() - ablation_start

    final_submission_path = run_dir / "submission.json"
    with final_submission_path.open("w") as handle:
        json.dump(combined_submission, handle)
    print(f"Final submission saved to {final_submission_path}")

    timing_path = run_dir / "task_timings.json"
    with timing_path.open("w") as handle:
        json.dump(
            {
                "total_seconds": total_seconds,
                "num_tasks": len(task_ids),
                "tasks": task_timings,
            },
            handle,
            indent=2,
        )
    print(f"Per-task timings saved to {timing_path}")
    print(f"Total ablation runtime: {total_seconds:.2f}s")

    if cli.no_score:
        utils.visualize_submissions(final_submission_path, mode="submission")
        return

    solutions_path = Path(cli.solutions_path)
    if solutions_path.exists():
        utils.score_arc_submission(solutions_path, final_submission_path)
        utils.visualize_submissions(final_submission_path, solutions_path, mode="!")
    else:
        print(f"Skipping score: solutions file not found at {solutions_path}")
        utils.visualize_submissions(final_submission_path, mode="submission")


if __name__ == "__main__":
    main()
