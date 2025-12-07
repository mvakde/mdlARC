import argparse
from pathlib import Path
from inference import run_split_inference
from utils import tokens_to_string, split_grids_from_tokens, plot_grids
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
    parser.add_argument(
        "--enable-color-aug-train",
        action="store_true",
        help="Apply color permutations during training (train split only).",
    )
    parser.add_argument(
        "--max-color-augments-train",
        type=int,
        default=0,
        help="Max number of unique color permutations to cycle through during training.",
    )
    parser.add_argument(
        "--enable-color-aug-eval",
        action="store_true",
        help="Apply color permutations during evaluation/inference.",
    )
    parser.add_argument(
        "--max-color-augments-eval",
        type=int,
        default=0,
        help="Number of color permutations to run at evaluation time.",
    )
    parser.add_argument(
        "--color-aug-seed",
        type=int,
        default=None,
        help="Optional seed for color permutations (defaults to --seed).",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    checkpoint = load_checkpoint(args.checkpoint_path)
    model, dataset, dataloader, device, data_path = build_model_and_data(
        args, checkpoint=checkpoint, is_eval=args.eval_only
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
            # 1. CALCULATE: Use the generic batch runner
            results = run_split_inference(
                model=model,
                dataset=dataset,
                split="test",
                device=device,
                task_ids=[args.inference_task_id],
                pair_index=args.inference_pair_index,
                log_prompts=args.log_inference_prompt,
                include_targets=True,  # Optional: enables checking against ground truth
            )

            # 2. PRESENT: Handle printing and plotting here in the CLI
            if not results:
                print("No results found.")
                return

            result = results[0]  # We only asked for one task/pair
            full_sequence = result["sequence"]
            output_tokens = result["output_tokens"]
            predicted_grid = result["output_grid"]

            print(
                f"\nInference results for task {args.inference_task_id} pair {args.inference_pair_index}"
            )
            print("Generated (string):", tokens_to_string(output_tokens))

            if predicted_grid:
                print("Decoded grid:")
                for row in predicted_grid:
                    print(row)
            else:
                print("Decoded grid: <empty>")

            # 3. PLOT: Optional visualization
            if args.plot_inference_grids:
                try:
                    prompt_tokens = result.get("prompt_tokens", [])
                    # Reconstruct grid lists for plotting
                    prompt_grids = split_grids_from_tokens(prompt_tokens)
                    input_grid = prompt_grids[0] if prompt_grids else []

                    # Prepare list of grids to plot: [Input, Predicted]
                    to_plot = [input_grid, predicted_grid]

                    plot_grids(
                        to_plot,
                        title=f"Task {args.inference_task_id} Pair {args.inference_pair_index}",
                    )
                except Exception as e:
                    print(f"Plotting failed: {e}")
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
