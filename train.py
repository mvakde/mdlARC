import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.optim import AdamW

from tinytransformer import TinyTransformer, TinyTransformerConfig
from utils import (
    ARCExampleDataset,
    END_TOKEN_ID,
    IO_SEPARATOR_TOKEN_ID,
    MAX_SEQ_LEN,
    NEXT_LINE_TOKEN_ID,
    START_TOKEN_ID,
    create_dataloader,
    extract_output_tokens,
    tokens_to_string,
    tokens_to_grid,
    split_grids_from_tokens,
    plot_grids,
)

DEFAULT_DATA_PATH = Path("assets/ARC-2/grouped-tasks/training/challenges.json")


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
    parser.add_argument(
        "--max-steps", type=int, default=0, help="Optional cap on total training steps."
    )
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
    parser.add_argument("--max-new-tokens", type=int, default=512)
    # Visibility / logging options
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
        "--log-eval-strings",
        action="store_true",
        help="During evaluate, print prompt and generated strings for a few examples.",
    )
    parser.add_argument(
        "--log-eval-limit",
        type=int,
        default=3,
        help="Max number of eval examples to log.",
    )
    parser.add_argument(
        "--plot-inference-grids",
        action="store_true",
        help="During single-example inference, plot input/output grids.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    device_str = device_str.lower()
    if device_str == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA not available, falling back to CPU.")
        return torch.device("cpu")
    if device_str == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        print("MPS not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cpu" if device_str not in {"cpu"} else "cpu")


def train_one_epoch(
    model: TinyTransformer,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    max_steps: int,
    start_step: int = 0,
    log_train_strings: bool = False,
    log_train_limit: int = 0,
) -> int:
    model.train()
    step = start_step
    total_loss = 0.0
    logged = 0
    for batch in dataloader:
        step += 1
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        example_ids = batch["example_ids"].to(device)

        outputs = model(input_ids, example_ids, attention_mask=attention_mask)
        loss = outputs["loss"]

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        # Optional: log the exact serialized strings the model is trained on
        if log_train_strings and logged < log_train_limit:
            bs = input_ids.size(0)
            to_log = min(log_train_limit - logged, bs)
            task_ids = batch.get("task_ids", ["?"] * bs)
            splits = batch.get("splits", ["?"] * bs)
            for i in range(to_log):
                seq_len = int(attention_mask[i].sum().item())
                seq = input_ids[i, :seq_len].detach().cpu().tolist()
                print(
                    "[train string]",
                    f"step={step}",
                    f"split={splits[i]}",
                    f"task={task_ids[i]}",
                    "::",
                    tokens_to_string(seq),
                )
                logged += 1
                if logged >= log_train_limit:
                    break
        if step % 10 == 0:
            avg_loss = total_loss / 10
            print(f"step={step} avg_loss={avg_loss:.4f}")
            total_loss = 0.0
        if max_steps and step >= max_steps:
            break
    return step


def _build_weight_decay_param_groups(
    model: nn.Module, weight_decay: float
) -> Any:
    """Split parameters so only non-attention Linear weights use weight decay."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias"):
            no_decay_params.append(param)
            continue

        module_name = name.rsplit(".", 1)[0] if "." in name else ""
        module = model.get_submodule(module_name) if module_name else model

        if isinstance(module, nn.Linear) and "attention" not in module_name:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return param_groups


def _update_position_state(
    x: int, y: int, z: int, token_id: int, is_first: bool
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Update 3D grid position state for a single token.

    Mirrors TinyTransformer._compute_positions_3d for sequential decoding.
    """
    if is_first and token_id == START_TOKEN_ID:
        return (0, 0, 0), (x, y, z)

    if token_id == IO_SEPARATOR_TOKEN_ID:
        return (0, 0, 2), (0, 0, 3)

    if token_id == END_TOKEN_ID:
        return (0, 0, 4), (x, y, z)

    px = min(max(x, 0), 29)
    py = min(max(y, 0), 30)
    pos = (px, py, z)

    if token_id == NEXT_LINE_TOKEN_ID:
        x = 0
        y += 1
    else:
        x += 1

    return pos, (x, y, z)


@torch.no_grad()
def greedy_generate(
    model: TinyTransformer,
    prompt_tokens: torch.LongTensor,
    example_id: int,
    device: torch.device,
    max_new_tokens: int,
) -> torch.LongTensor:
    model.eval()
    generated = prompt_tokens.unsqueeze(0).to(device)
    example_ids_tensor = torch.tensor([example_id], dtype=torch.long, device=device)

    # Initialize 3D position state after consuming the prompt.
    x, y, z = 0, 0, 1
    prompt_list = prompt_tokens.tolist()
    for idx, tok in enumerate(prompt_list):
        _, (x, y, z) = _update_position_state(x, y, z, int(tok), is_first=(idx == 0))

    # First pass: run the full prompt to build KV cache and get initial logits.
    outputs = model.forward_generate(
        input_ids=generated,
        example_ids=example_ids_tensor,
        past_key_values=None,
    )
    logits = outputs["logits"]
    past_key_values = outputs["past_key_values"]

    for _ in range(max_new_tokens):
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        token_id = int(next_token.item())
        generated = torch.cat([generated, next_token], dim=1)

        if token_id == END_TOKEN_ID:
            break
        if generated.size(1) >= model.config.max_seq_len:
            print("Reached model max_seq_len during generation; stopping.")
            break

        pos, (x, y, z) = _update_position_state(
            x, y, z, token_id, is_first=False
        )
        pos_tensor = torch.tensor([[list(pos)]], dtype=torch.long, device=device)

        outputs = model.forward_generate(
            input_ids=next_token,
            example_ids=example_ids_tensor,
            past_key_values=past_key_values,
            positions_3d=pos_tensor,
        )
        logits = outputs["logits"]
        past_key_values = outputs["past_key_values"]

    return generated.squeeze(0).cpu()


def run_inference(
    model: TinyTransformer,
    dataset: ARCExampleDataset,
    task_id: str,
    pair_index: int,
    device: torch.device,
    max_new_tokens: int,
    log_prompt: bool = False,
    plot_grids_flag: bool = False,
) -> None:
    candidate = None
    for example in dataset.iter_examples(split="test", has_output=False):
        if example.task_id == task_id and example.pair_index == pair_index:
            candidate = example
            break

    if candidate is None:
        raise ValueError(
            f"No test example found for task_id={task_id} pair_index={pair_index}."
        )

    if log_prompt:
        print("\nInference prompt (string):", tokens_to_string(candidate.tokens.tolist()))

    generated = greedy_generate(
        model=model,
        prompt_tokens=candidate.tokens,
        example_id=candidate.example_id,
        device=device,
        max_new_tokens=max_new_tokens,
    )
    full_sequence = generated.tolist()
    output_tokens = extract_output_tokens(full_sequence)
    predicted_grid = tokens_to_grid(output_tokens)

    print(f"\nInference results for task {task_id} pair {pair_index}")
    print("Generated raw (string):", tokens_to_string(full_sequence))
    print("Generated (string):", tokens_to_string(output_tokens))
    if predicted_grid:
        print("Decoded grid:")
        for row in predicted_grid:
            print(row)
    else:
        print("Decoded grid: <empty>")

    if plot_grids_flag:
        try:
            # Plot two grids: input (before separator) and generated output
            prompt_grids = split_grids_from_tokens(candidate.tokens.tolist())
            gen_grids = split_grids_from_tokens([*candidate.tokens.tolist(), *output_tokens, END_TOKEN_ID])
            # Prefer showing exactly two: the input grid(s) first segment and the predicted output
            input_grid = prompt_grids[0] if prompt_grids else []
            output_grid = gen_grids[1] if len(gen_grids) > 1 else tokens_to_grid(output_tokens)
            to_plot = [input_grid, output_grid]
            plot_grids(to_plot, title=f"task {task_id} pair {pair_index}")
        except Exception as e:
            print(f"Plotting failed: {e}")


@torch.no_grad()
def evaluate_dataset(
    model: TinyTransformer,
    dataset: ARCExampleDataset,
    data_path: Path,
    device: torch.device,
    max_new_tokens: int,
    log_eval_strings: bool = False,
    log_eval_limit: int = 0,
) -> None:
    """Run greedy inference over all test pairs and compute exact-match accuracy.

    Uses the companion solutions.json alongside the provided challenges.json to
    compare predictions against ground truth for test pairs.
    """
    # Solutions are stored next to the challenges.json
    solutions_path = data_path.parent / "solutions.json"
    if not solutions_path.exists():
        print(f"solutions.json not found at {solutions_path}; skipping eval.")
        return

    # Load solutions mapping: task_id -> list of output grids for test pairs
    import json

    with solutions_path.open("r") as f:
        solutions = json.load(f)

    total = 0
    correct = 0
    logged = 0

    # Evaluate only on test pairs without provided outputs (i.e., challenge test pairs)
    for example in dataset.iter_examples(split="test", has_output=False):
        # Guard against missing solutions
        sols = solutions.get(example.task_id)
        if not sols:
            continue
        if example.pair_index >= len(sols):
            continue

        # Prepare prompt and generate
        # Optional: print prompt before generation
        if log_eval_strings and logged < log_eval_limit:
            print("\n[eval prompt]",
                  f"task={example.task_id}",
                  f"pair={example.pair_index}")
            print("str:", tokens_to_string(example.tokens.tolist()))

        generated = greedy_generate(
            model=model,
            prompt_tokens=example.tokens,
            example_id=example.example_id,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        full_sequence = generated.tolist()
        output_tokens = extract_output_tokens(full_sequence)
        predicted_grid = tokens_to_grid(output_tokens)
        reference_grid = sols[example.pair_index]

        # Optional: print generated sequence
        if log_eval_strings and logged < log_eval_limit:
            print("[eval generated raw]",
                  f"task={example.task_id}",
                  f"pair={example.pair_index}")
            print("str:", tokens_to_string(full_sequence))
            print("[eval generated]",
                  f"task={example.task_id}",
                  f"pair={example.pair_index}")
            print("str:", tokens_to_string(output_tokens))
            logged += 1

        is_match = predicted_grid == reference_grid
        total += 1
        correct += int(is_match)

    if total == 0:
        print("No evaluable test pairs found; skipping eval.")
        return

    acc = 100.0 * correct / total
    print(f"\nEval: {correct}/{total} correct ({acc:.2f}%)")


def maybe_save_model(
    model: TinyTransformer,
    dataset: ARCExampleDataset,
    data_path: Path,
    save_path: Optional[Path],
) -> None:
    if save_path is None:
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "config": asdict(model.config),
        "task_ids": list(dataset.task_ids),
        "data_path": str(data_path),
    }
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")


def load_checkpoint(checkpoint_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if checkpoint_path is None:
        return None
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state" not in checkpoint:
        checkpoint = {"model_state": checkpoint}
    checkpoint["__path__"] = str(checkpoint_path)
    print(f"Loaded checkpoint from {checkpoint_path}")
    return checkpoint


def infer_num_examples_from_checkpoint(
    checkpoint: Optional[Dict[str, Any]],
) -> Optional[int]:
    if not checkpoint:
        return None
    config = checkpoint.get("config")
    if config and "num_examples" in config:
        return int(config["num_examples"])
    state_dict = checkpoint.get("model_state", {})
    weight = state_dict.get("example_embedding.weight")
    if weight is not None:
        return int(weight.shape[0])
    return None


def build_model_and_data(
    args: argparse.Namespace,
) -> Tuple[
    TinyTransformer,
    ARCExampleDataset,
    torch.utils.data.DataLoader,
    torch.device,
    Path,
]:
    """Construct dataset, dataloader, and model for a given arg namespace.

    Shared by CLI entrypoints and notebooks so that training, evaluation,
    and inference can be orchestrated independently.
    """
    set_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint_path)

    data_path = args.data_path
    if data_path is None:
        if checkpoint and "data_path" in checkpoint:
            data_path = Path(checkpoint["data_path"])
        else:
            raise ValueError(
                "--data-path is required when loading checkpoints that do not encode their source dataset. "
                "Please re-run with the same dataset used for training."
            )
    data_path = Path(data_path)

    checkpoint_num_examples = infer_num_examples_from_checkpoint(checkpoint)

    task_whitelist = None
    if checkpoint and "task_ids" in checkpoint:
        task_whitelist = checkpoint["task_ids"]

    dataset = ARCExampleDataset(
        json_path=data_path,
        splits=("train", "test"),
        include_outputs=True,
        max_seq_len=MAX_SEQ_LEN,
        task_whitelist=task_whitelist,
    )
    dataloader = create_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=not args.eval_only,
        num_workers=args.num_workers,
    )

    if (
        checkpoint_num_examples is not None
        and dataset.num_examples != checkpoint_num_examples
    ):
        raise ValueError(
            "Dataset task-count mismatch: "
            f"checkpoint was trained with {checkpoint_num_examples} unique examples but the provided dataset "
            f"currently exposes {dataset.num_examples}. Pass the original --data-path or retrain."
        )

    if checkpoint and "config" in checkpoint:
        config = TinyTransformerConfig(**checkpoint["config"])
    else:
        num_examples = checkpoint_num_examples or max(1, dataset.num_examples)
        config = TinyTransformerConfig(num_examples=num_examples)

    if dataset.num_examples != config.num_examples:
        raise ValueError(
            f"Dataset provides {dataset.num_examples} examples but model expects "
            f"{config.num_examples}. Please ensure the dataset/task whitelist matches the checkpoint."
        )

    model = TinyTransformer(config).to(device)

    if checkpoint:
        state_dict = checkpoint["model_state"]
        model.load_state_dict(state_dict, strict=False)

    return model, dataset, dataloader, device, data_path


def train_model(
    args: argparse.Namespace,
    model: TinyTransformer,
    dataloader: torch.utils.data.DataLoader,
    dataset: ARCExampleDataset,
    device: torch.device,
    data_path: Path,
) -> None:
    """Run the training loop only (no evaluation)."""
    param_groups = _build_weight_decay_param_groups(model, args.weight_decay)
    optimizer = AdamW(param_groups, lr=args.lr)
    step = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        step = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
            max_steps=args.max_steps,
            start_step=step,
            log_train_strings=args.log_train_strings,
            log_train_limit=args.log_train_limit,
        )
        if args.max_steps and step >= args.max_steps:
            break
    maybe_save_model(model, dataset, data_path, args.save_path)


def evaluate_model(
    args: argparse.Namespace,
    model: TinyTransformer,
    dataset: ARCExampleDataset,
    device: torch.device,
    data_path: Path,
) -> None:
    """Evaluate a model on the ARC test split using solutions.json."""
    evaluate_dataset(
        model=model,
        dataset=dataset,
        data_path=data_path,
        device=device,
        max_new_tokens=args.max_new_tokens,
        log_eval_strings=args.log_eval_strings,
        log_eval_limit=args.log_eval_limit,
    )


def run(args: argparse.Namespace) -> None:
    model, dataset, dataloader, device, data_path = build_model_and_data(args)

    if not args.eval_only:
        # Train then auto-evaluate on test pairs from the same dataset
        # (using solutions.json). This mirrors the original behavior but
        # now delegates to dedicated train / eval helpers.
        train_model(
            args=args,
            model=model,
            dataloader=dataloader,
            dataset=dataset,
            device=device,
            data_path=data_path,
        )
        evaluate_model(
            args=args,
            model=model,
            dataset=dataset,
            device=device,
            data_path=data_path,
        )
    else:
        # Eval-only mode: if a specific task is provided, run single-example inference.
        # Otherwise, evaluate all test pairs using solutions.json.
        if args.inference_task_id:
            run_inference(
                model=model,
                dataset=dataset,
                task_id=args.inference_task_id,
                pair_index=args.inference_pair_index,
                device=device,
                max_new_tokens=args.max_new_tokens,
                log_prompt=args.log_inference_prompt,
                plot_grids_flag=args.plot_inference_grids,
            )
        else:
            evaluate_dataset(
                model=model,
                dataset=dataset,
                data_path=data_path,
                device=device,
                max_new_tokens=args.max_new_tokens,
                log_eval_strings=args.log_eval_strings,
                log_eval_limit=args.log_eval_limit,
            )


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
