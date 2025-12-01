import argparse
from pathlib import Path

from inference import run_inference
from train import build_model_and_data, load_checkpoint, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and run inference with TinyTransformer."
    )
    parser.add_argument(
        "--data-path", type=Path, default=None, help="Path to the challenges.json file."
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", type=str, default="mps", help="cpu | cuda | mps (Apple Silicon)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Optional path to save the trained model.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Load weights before training/eval.",
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="Skip training and only run inference."
    )
    parser.add_argument("--inference-task-id", type=str, default=None)
    parser.add_argument("--inference-pair-index", type=int, default=0)
    parser.add_argument(
        "--log-train-strings",
        action="store_true",
        help="Print example training sequences (decoded token strings).",
    )
    parser.add_argument(
        "--log-train-limit",
        type=int,
        default=3,
        help="Max number of training examples to log per run.",
    )
    parser.add_argument(
        "--log-inference-prompt",
        action="store_true",
        help="Print the exact prompt sequence used for inference.",
    )
    parser.add_argument(
        "--plot-inference-grids",
        action="store_true",
        help="During single-example inference, plot input/output grids.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    checkpoint = load_checkpoint(args.checkpoint_path)
    model, dataset, dataloader, device, data_path = build_model_and_data(
        args, checkpoint=checkpoint
    )

    if not args.eval_only:
        train_model(
            args=args,
            model=model,
            dataloader=dataloader,
            dataset=dataset,
            device=device,
            data_path=data_path,
            checkpoint=checkpoint,
        )
    else:
        if args.inference_task_id:
            run_inference(
                model=model,
                dataset=dataset,
                task_id=args.inference_task_id,
                pair_index=args.inference_pair_index,
                device=device,
                log_prompt=args.log_inference_prompt,
                plot_grids_flag=args.plot_inference_grids,
            )
        else:
            raise ValueError(
                "In eval_only mode, you must provide --inference-task-id "
                "to run single-example inference."
            )


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
