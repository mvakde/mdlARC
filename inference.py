import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import torch

from tinytransformer import TinyTransformer
from utils import (
    END_TOKEN_ID,
    IO_SEPARATOR_TOKEN_ID,
    NEXT_LINE_TOKEN_ID,
    START_TOKEN_ID,
    compute_positions_3d,
    extract_output_tokens,
    grid_to_tokens,
    plot_grids,
    split_grids_from_tokens,
    tokens_to_grid,
    tokens_to_string,
)

DEFAULT_MAX_NEW_TOKENS = 931


class BatchGridState:
    """Vectorized tracker for 3D grid coordinates during generation."""

    def __init__(self, initial_state: torch.Tensor) -> None:
        if initial_state.dim() != 2 or initial_state.size(1) != 3:
            raise ValueError("initial_state must have shape [batch, 3].")
        self.state = initial_state.clone().long()

    def update(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Advance state with a batch of token ids and return positions for them."""
        token_ids = token_ids.view(-1).to(device=self.state.device)
        x, y, z = self.state.unbind(-1)

        pos_x = torch.clamp(x, min=0, max=30)
        pos_y = torch.clamp(y, min=0, max=29)
        pos_z = z

        is_start = token_ids == START_TOKEN_ID
        is_sep = token_ids == IO_SEPARATOR_TOKEN_ID
        is_end = token_ids == END_TOKEN_ID
        is_newline = token_ids == NEXT_LINE_TOKEN_ID

        zeros = torch.zeros_like(x)
        pos_x = torch.where(is_start | is_sep | is_end, zeros, pos_x)
        pos_y = torch.where(is_start | is_sep | is_end, zeros, pos_y)
        pos_z = torch.where(is_start, zeros, pos_z)
        pos_z = torch.where(is_sep, torch.full_like(pos_z, 2), pos_z)
        pos_z = torch.where(is_end, torch.full_like(pos_z, 4), pos_z)

        next_x = x + 1
        next_y = y
        next_z = z

        next_x = torch.where(is_newline, zeros, next_x)
        next_y = torch.where(is_newline, y + 1, next_y)

        next_x = torch.where(is_sep, zeros, next_x)
        next_y = torch.where(is_sep, zeros, next_y)
        next_z = torch.where(is_sep, torch.full_like(next_z, 3), next_z)

        next_x = torch.where(is_end | is_start, x, next_x)
        next_y = torch.where(is_end | is_start, y, next_y)
        next_z = torch.where(is_start, z, next_z)
        next_z = torch.where(is_end, z, next_z)

        self.state = torch.stack([next_x, next_y, next_z], dim=-1)
        positions = torch.stack([pos_x, pos_y, pos_z], dim=-1)
        return positions


def _left_pad_sequences(
    sequences: Sequence[Sequence[int]], pad_token_id: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(sequences)
    max_len = max(len(seq) for seq in sequences)
    input_ids = torch.full(
        (batch_size, max_len), pad_token_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
    for idx, seq in enumerate(sequences):
        seq_len = len(seq)
        start = max_len - seq_len
        input_ids[idx, start:] = torch.tensor(seq, dtype=torch.long, device=device)
        attention_mask[idx, start:] = True
    return input_ids, attention_mask


def _pad_cached_positions(
    cached_positions: Sequence[torch.Tensor], max_len: int, device: torch.device
) -> torch.Tensor:
    positions = torch.zeros(
        (len(cached_positions), max_len, 3), dtype=torch.long, device=device
    )
    for idx, pos in enumerate(cached_positions):
        seq_len = pos.size(0)
        start = max_len - seq_len
        positions[idx, start:] = pos.to(device=device, dtype=torch.long)
    return positions


def _derive_initial_state_from_prompt(
    input_ids: torch.Tensor, positions_3d: torch.Tensor, attention_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Since input_ids are left-padded by _left_pad_sequences, the last valid token
    # (the one determining the start state for generation) is always at the very end.
    last_tokens = input_ids[:, -1]
    last_positions = positions_3d[:, -1]

    x, y, z = last_positions.unbind(-1)

    next_x = x + 1
    next_y = y
    next_z = z

    next_x = torch.where(
        last_tokens == NEXT_LINE_TOKEN_ID, torch.zeros_like(next_x), next_x
    )
    next_y = torch.where(last_tokens == NEXT_LINE_TOKEN_ID, y + 1, next_y)

    next_x = torch.where(
        last_tokens == IO_SEPARATOR_TOKEN_ID, torch.zeros_like(next_x), next_x
    )
    next_y = torch.where(
        last_tokens == IO_SEPARATOR_TOKEN_ID, torch.zeros_like(next_y), next_y
    )
    next_z = torch.where(
        last_tokens == IO_SEPARATOR_TOKEN_ID, torch.full_like(next_z, 3), next_z
    )

    next_x = torch.where(last_tokens == END_TOKEN_ID, x, next_x)
    next_y = torch.where(last_tokens == END_TOKEN_ID, y, next_y)
    next_z = torch.where(last_tokens == END_TOKEN_ID, z, next_z)

    next_x = torch.where(
        last_tokens == START_TOKEN_ID, torch.zeros_like(next_x), next_x
    )
    next_y = torch.where(
        last_tokens == START_TOKEN_ID, torch.zeros_like(next_y), next_y
    )
    next_z = torch.where(last_tokens == START_TOKEN_ID, torch.ones_like(next_z), next_z)

    initial_state = torch.stack([next_x, next_y, next_z], dim=-1)
    finished = last_tokens == END_TOKEN_ID
    return initial_state, finished


@torch.inference_mode()
def batched_greedy_generate(
    model: TinyTransformer,
    prompts: Sequence[Sequence[int]],
    example_ids: Sequence[int],
    device: torch.device,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    cached_positions: Optional[Sequence[Optional[torch.Tensor]]] = None,
) -> List[List[int]]:
    if not prompts:
        raise ValueError("prompts must be non-empty.")
    if len(prompts) != len(example_ids):
        raise ValueError("prompts and example_ids must have the same length.")
    if cached_positions is not None and len(cached_positions) != len(prompts):
        raise ValueError(
            "cached_positions must be None or match the number of prompts."
        )

    model.eval()
    batch_size = len(prompts)
    max_prompt_len = max(len(seq) for seq in prompts)
    if max_prompt_len > model.config.max_seq_len:
        raise ValueError("Prompt length exceeds model max_seq_len; cannot generate.")

    example_ids_tensor = torch.tensor(example_ids, dtype=torch.long, device=device)
    input_ids, attention_mask = _left_pad_sequences(
        prompts, pad_token_id=END_TOKEN_ID, device=device
    )

    use_cached_positions = cached_positions is not None and all(
        pos is not None for pos in cached_positions
    )

    if use_cached_positions:
        prompt_positions = _pad_cached_positions(
            [pos for pos in cached_positions if pos is not None],
            max_prompt_len,
            device=device,
        )
    else:
        prompt_positions = compute_positions_3d(input_ids, attention_mask).to(
            device=device, dtype=torch.long
        )

    example_embeds = model.example_embedding(example_ids_tensor)
    initial_state, finished = _derive_initial_state_from_prompt(
        input_ids, prompt_positions, attention_mask
    )
    grid_state = BatchGridState(initial_state)
    current_len = input_ids.size(1)
    max_len = model.config.max_seq_len

    # 1. Initial Prompt Pass
    running_attention_mask = torch.zeros(
        (batch_size, max_len), dtype=torch.bool, device=device
    )
    running_attention_mask[:, :current_len] = attention_mask
    prompt_attention_mask = running_attention_mask[:, :current_len]
    outputs = model.forward_generate(
        input_ids=input_ids,
        example_ids=example_ids_tensor,
        past_key_values=None,
        positions_3d=prompt_positions,
        attention_mask=prompt_attention_mask,
        example_embeds=example_embeds,
    )
    logits = outputs["logits"]
    prompt_past_key_values = outputs["past_key_values"]

    # 2. Pre-allocate KV Cache
    # We create a buffer of (Batch, Heads, MaxSeqLen, Dim) and copy the prompt KV into it.
    past_key_values: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for k, v in prompt_past_key_values:
        # k, v are [Batch, Heads, PromptLen, Dim]
        # create buffer
        B, H, L, D = k.shape
        k_buffer = torch.zeros((B, H, max_len, D), dtype=k.dtype, device=k.device)
        v_buffer = torch.zeros((B, H, max_len, D), dtype=v.dtype, device=v.device)

        # Copy prompt data
        k_buffer[:, :, :L, :] = k
        v_buffer[:, :, :L, :] = v
        past_key_values.append((k_buffer, v_buffer))

    past_key_values = tuple(past_key_values)
    cache_position = current_len  # We start generating at this index

    max_steps_allowed = max(model.config.max_seq_len - input_ids.size(1), 0)
    steps_remaining = min(max_new_tokens, max_steps_allowed)

    # Instead of appending to python lists (CPU) every step, we write to this tensor.
    generated_tokens_buffer = torch.full(
        (batch_size, steps_remaining), END_TOKEN_ID, dtype=torch.long, device=device
    )

    if steps_remaining <= 0 or finished.all():
        return [list(seq) for seq in prompts]

    steps = 0
    while steps < steps_remaining and not finished.all():
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        next_token = torch.where(
            finished, torch.full_like(next_token, END_TOKEN_ID), next_token
        )

        should_append = ~finished
        token_positions = grid_state.update(next_token).unsqueeze(1)

        # Write to GPU buffer directly. No .item(), no .tolist(), no CPU sync.
        generated_tokens_buffer[:, steps] = next_token

        finished = finished | (next_token == END_TOKEN_ID)

        running_attention_mask[:, cache_position] = should_append
        attn_mask_view = running_attention_mask[:, : cache_position + 1]

        outputs = model.forward_generate(
            input_ids=next_token.unsqueeze(1),
            example_ids=example_ids_tensor,
            past_key_values=past_key_values,
            positions_3d=token_positions,
            attention_mask=attn_mask_view,
            cache_position=cache_position,
            example_embeds=example_embeds,
        )
        logits = outputs["logits"]
        # past_key_values = outputs["past_key_values"]

        steps += 1
        cache_position += 1

    generated_cpu = generated_tokens_buffer[:, :steps].tolist()

    results = []
    for i, prompt in enumerate(prompts):
        gen_seq = []
        # Extract valid tokens until the first END_TOKEN_ID
        for token in generated_cpu[i]:
            gen_seq.append(token)
            if token == END_TOKEN_ID:
                break
        results.append(list(prompt) + gen_seq)

    return results


def _build_prompt_from_tokens(tokens: Sequence[int]) -> List[int]:
    if IO_SEPARATOR_TOKEN_ID not in tokens:
        raise ValueError("Prompt sequence is missing <input_output_separator>.")
    sep_idx = tokens.index(IO_SEPARATOR_TOKEN_ID)
    return list(tokens[: sep_idx + 1])


def _prepare_examples_for_inference(
    examples: Sequence[object],
    include_targets: bool = False,
    solutions: Optional[Dict[Tuple[str, int], List[List[int]]]] = None,
) -> Tuple[
    List[List[int]],
    List[int],
    List[Dict[str, object]],
    List[Optional[torch.Tensor]],
    List[List[int]],
]:
    prompts: List[List[int]] = []
    example_ids: List[int] = []
    metadata: List[Dict[str, object]] = []
    cached_positions: List[Optional[torch.Tensor]] = []
    target_tokens: List[List[int]] = []

    for ex in examples:
        if not hasattr(ex, "tokens"):
            raise ValueError("Examples must provide a 'tokens' attribute.")
        tokens = ex.tokens.tolist()
        prompt_tokens = _build_prompt_from_tokens(tokens)
        prompts.append(prompt_tokens)
        example_ids.append(int(getattr(ex, "example_id", 0)))
        cached = getattr(ex, "cached_positions", None)
        if cached is not None:
            cached_positions.append(cached[: len(prompt_tokens)])
        else:
            cached_positions.append(None)

        targets: List[int] = []
        if include_targets and getattr(ex, "has_output", False):
            targets = extract_output_tokens(tokens)
        elif include_targets and solutions is not None:
            key = (getattr(ex, "task_id", None), getattr(ex, "pair_index", None))
            if key in solutions and solutions[key] is not None:
                targets = grid_to_tokens(solutions[key])
        target_tokens.append(targets)
        metadata.append(
            {
                "task_id": getattr(ex, "task_id", None),
                "pair_index": getattr(ex, "pair_index", None),
                "example_id": getattr(ex, "example_id", None),
                "split": getattr(ex, "split", None),
            }
        )

    return prompts, example_ids, metadata, cached_positions, target_tokens


def _build_generation_results(
    sequences: Sequence[Sequence[int]],
    metadata: Sequence[Dict[str, object]],
    prompts: Sequence[Sequence[int]],
    target_output_tokens: Sequence[Sequence[int]],
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for seq, meta, prompt, target in zip(
        sequences, metadata, prompts, target_output_tokens
    ):
        output_tokens = extract_output_tokens(seq)
        predicted_grid = tokens_to_grid(output_tokens)
        target_grid = tokens_to_grid(target) if target else []
        result = {
            "task_id": meta.get("task_id"),
            "pair_index": meta.get("pair_index"),
            "example_id": meta.get("example_id"),
            "split": meta.get("split"),
            "prompt_tokens": list(prompt),
            "sequence": list(seq),
            "output_tokens": output_tokens,
            "output_grid": predicted_grid,
            "target_output_tokens": list(target),
            "target_grid": target_grid,
        }
        results.append(result)
    return results


def _run_generation_batch(
    model: TinyTransformer,
    prompts: Sequence[Sequence[int]],
    example_ids: Sequence[int],
    metadata: Sequence[Dict[str, object]],
    cached_positions: Sequence[Optional[torch.Tensor]],
    device: torch.device,
    max_new_tokens: int,
    target_output_tokens: Optional[Sequence[Sequence[int]]] = None,
) -> List[Dict[str, object]]:
    sequences = batched_greedy_generate(
        model=model,
        prompts=prompts,
        example_ids=example_ids,
        device=device,
        max_new_tokens=max_new_tokens,
        cached_positions=cached_positions,
    )
    return _build_generation_results(
        sequences=sequences,
        metadata=metadata,
        prompts=prompts,
        target_output_tokens=target_output_tokens
        if target_output_tokens is not None
        else [[] for _ in prompts],
    )


def _select_inference_examples(
    dataset,
    task_ids: Sequence[str],
    split: str = "test",
    pair_index: Optional[int] = 0,
    require_outputs: bool = False,
    solutions: Optional[Dict[Tuple[str, int], List[List[int]]]] = None,
) -> Tuple[
    List[List[int]],
    List[int],
    List[Dict[str, object]],
    List[Optional[torch.Tensor]],
    List[List[int]],
]:
    selected = []
    for task_id in task_ids:
        candidate = None
        for example in dataset.iter_examples(split=split):
            if example.task_id != task_id:
                continue
            if pair_index is not None and example.pair_index != pair_index:
                continue
            has_solution = solutions is not None and (
                (example.task_id, example.pair_index) in solutions
            )
            if require_outputs and not example.has_output and not has_solution:
                continue
            candidate = example
            break
        if candidate is None:
            raise ValueError(
                f"No {split} example found for task_id={task_id} pair_index={pair_index}."
            )
        selected.append(candidate)

    prompts, example_ids, metadata, cached_positions, targets = (
        _prepare_examples_for_inference(
            selected, include_targets=require_outputs, solutions=solutions
        )
    )
    return prompts, example_ids, metadata, cached_positions, targets


@torch.no_grad()
def run_batched_inference(
    model: TinyTransformer,
    dataset,
    task_ids: Sequence[str],
    device: torch.device,
    split: str = "test",
    pair_index: int = 0,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    log_prompts: bool = False,
    include_targets: bool = False,
) -> List[Dict[str, object]]:
    solutions = _load_solutions_for_dataset(dataset) if include_targets else None
    (prompts, example_ids, metadata, cached_positions, target_output_tokens) = (
        _select_inference_examples(
            dataset,
            task_ids,
            split=split,
            pair_index=pair_index,
            require_outputs=include_targets,
            solutions=solutions,
        )
    )
    if log_prompts:
        for meta, prompt in zip(metadata, prompts):
            print(
                "[prompt]",
                f"task={meta['task_id']}",
                f"pair={meta['pair_index']}",
                "::",
                tokens_to_string(prompt),
            )

    sequences = batched_greedy_generate(
        model=model,
        prompts=prompts,
        example_ids=example_ids,
        device=device,
        max_new_tokens=max_new_tokens,
        cached_positions=cached_positions,
    )
    return _build_generation_results(
        sequences=sequences,
        metadata=metadata,
        prompts=prompts,
        target_output_tokens=target_output_tokens
        if include_targets
        else [[] for _ in prompts],
    )


def _load_solutions_for_dataset(dataset) -> Dict[Tuple[str, int], List[List[int]]]:
    """Load solutions.json located next to the dataset (used only for evaluation).

    Expected structure:
    {task_id: [<2d_test_grid_0>, <2d_test_grid_1>, ...]}
    """
    solutions_map: Dict[Tuple[str, int], List[List[int]]] = {}
    source_path = getattr(dataset, "source_path", None)
    if source_path is None:
        return solutions_map
    solutions_path = Path(source_path).with_name("solutions.json")
    if not solutions_path.exists():
        return solutions_map
    try:
        data = json.loads(solutions_path.read_text())
        for task_id, outputs in data.items():
            if not isinstance(outputs, list):
                continue
            for idx, grid in enumerate(outputs):
                if grid is not None:
                    solutions_map[(task_id, idx)] = grid
    except Exception:
        return solutions_map
    return solutions_map


def _gather_examples_for_split(
    dataset,
    split: str,
    task_ids: Optional[Sequence[str]] = None,
    pair_index: Optional[int] = None,
    require_outputs: bool = False,
    solutions: Optional[Dict[Tuple[str, int], List[List[int]]]] = None,
):
    examples = []
    for example in dataset.iter_examples(split=split):
        if task_ids is not None and example.task_id not in task_ids:
            continue
        if pair_index is not None and example.pair_index != pair_index:
            continue
        has_solution = solutions is not None and (
            (example.task_id, example.pair_index) in solutions
        )
        if require_outputs and not example.has_output and not has_solution:
            continue
        examples.append(example)
    return examples


@torch.inference_mode()
def run_split_inference(
    model: TinyTransformer,
    dataset,
    split: str,
    device: torch.device,
    batch_size: int = 16,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    task_ids: Optional[Sequence[str]] = None,
    pair_index: Optional[int] = None,
    log_prompts: bool = False,
    include_targets: bool = True,
) -> List[Dict[str, object]]:
    solutions = _load_solutions_for_dataset(dataset) if include_targets else None
    examples = _gather_examples_for_split(
        dataset,
        split=split,
        task_ids=task_ids,
        pair_index=pair_index,
        require_outputs=include_targets,
        solutions=solutions,
    )
    if not examples:
        return []

    # Sort by sequence length to keep padding overhead low, then restore order.
    indexed_examples = list(enumerate(examples))
    indexed_examples.sort(key=lambda pair: pair[1].seq_len, reverse=True)
    results_buffer: List[Optional[Dict[str, object]]] = [None] * len(examples)

    for start in range(0, len(indexed_examples), batch_size):
        chunk = indexed_examples[start : start + batch_size]
        batch_indices, batch_examples = zip(*chunk)
        (prompts, example_ids, metadata, cached_positions, target_output_tokens) = (
            _prepare_examples_for_inference(
                batch_examples, include_targets=include_targets, solutions=solutions
            )
        )
        if log_prompts:
            for meta, prompt in zip(metadata, prompts):
                print(
                    "[prompt]",
                    f"split={meta.get('split')}",
                    f"task={meta.get('task_id')}",
                    f"pair={meta.get('pair_index')}",
                    "::",
                    tokens_to_string(prompt),
                )
        batch_results = _run_generation_batch(
            model=model,
            prompts=prompts,
            example_ids=example_ids,
            metadata=metadata,
            cached_positions=cached_positions,
            device=device,
            max_new_tokens=max_new_tokens,
            target_output_tokens=target_output_tokens if include_targets else None,
        )
        for idx, res in zip(batch_indices, batch_results):
            results_buffer[idx] = res

    return [res for res in results_buffer if res is not None]


def _has_correct_shape(
    sequence: Sequence[int],
    predicted_tokens: Sequence[int],
    target_tokens: Sequence[int],
) -> bool:
    if not target_tokens:
        return False
    if len(predicted_tokens) != len(target_tokens):
        return False

    target_newlines = [
        idx for idx, tok in enumerate(target_tokens) if tok == NEXT_LINE_TOKEN_ID
    ]
    predicted_newlines = [
        idx for idx, tok in enumerate(predicted_tokens) if tok == NEXT_LINE_TOKEN_ID
    ]
    if target_newlines != predicted_newlines:
        return False

    try:
        sep_idx = sequence.index(IO_SEPARATOR_TOKEN_ID)
        end_idx = sequence.index(END_TOKEN_ID, sep_idx + 1)
    except ValueError:
        return False
    return (end_idx - (sep_idx + 1)) == len(target_tokens)


def _pixel_accuracy(
    predicted_tokens: Sequence[int], target_tokens: Sequence[int]
) -> Optional[float]:
    predicted_digits = [tok for tok in predicted_tokens if 0 <= tok <= 9]
    target_digits = [tok for tok in target_tokens if 0 <= tok <= 9]
    if not target_digits or len(predicted_digits) != len(target_digits):
        return None
    correct = sum(1 for p, t in zip(predicted_digits, target_digits) if p == t)
    return correct / len(target_digits)


def summarize_split_results(results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    num_shape_correct = 0
    num_fully_correct = 0
    accuracies: List[float] = []
    fully_correct_results: List[Dict[str, object]] = []

    for res in results:
        predicted_tokens = res.get("output_tokens", [])
        target_tokens = res.get("target_output_tokens", [])
        sequence = res.get("sequence", [])
        shape_ok = _has_correct_shape(sequence, predicted_tokens, target_tokens)
        res["shape_correct"] = shape_ok
        if not shape_ok:
            continue
        num_shape_correct += 1
        acc = _pixel_accuracy(predicted_tokens, target_tokens)
        if acc is not None:
            res["pixel_accuracy"] = acc
            accuracies.append(acc)
        if predicted_tokens == target_tokens:
            num_fully_correct += 1
            fully_correct_results.append(res)

    avg_pixel_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    return {
        "total_sequences": len(results),
        "num_shape_correct": num_shape_correct,
        "avg_pixel_accuracy": avg_pixel_accuracy,
        "num_fully_correct": num_fully_correct,
        "fully_correct_results": fully_correct_results,
    }


def evaluate_model_on_dataset(
    model: TinyTransformer,
    dataset,
    device: torch.device,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    batch_size: int = 16,
    splits: Sequence[str] = ("train", "test"),
    log_prompts: bool = False,
) -> Dict[str, Dict[str, object]]:
    evaluation: Dict[str, Dict[str, object]] = {}
    for split in splits:
        split_results = run_split_inference(
            model=model,
            dataset=dataset,
            split=split,
            device=device,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            log_prompts=log_prompts,
            include_targets=True,
        )
        summary = summarize_split_results(split_results)
        evaluation[split] = {"results": split_results, "summary": summary}
    return evaluation


def run_inference(
    model: TinyTransformer,
    dataset,
    task_id: str,
    pair_index: int,
    device: torch.device,
    split: str = "test",
    log_prompt: bool = False,
    plot_grids_flag: bool = False,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> None:
    start_time = time.perf_counter()
    results = run_batched_inference(
        model=model,
        dataset=dataset,
        task_ids=[task_id],
        device=device,
        split=split,
        pair_index=pair_index,
        max_new_tokens=max_new_tokens,
        log_prompts=log_prompt,
    )
    if not results:
        print("No inference results were produced.")
        return

    result = results[0]
    full_sequence = result["sequence"]
    output_tokens = result["output_tokens"]
    predicted_grid = result["output_grid"]
    elapsed = time.perf_counter() - start_time

    print(f"\nInference results for task {task_id} pair {pair_index} ({split} split)")
    print(
        f"Generation time: {elapsed:.3f}s for "
        f"{len(full_sequence) - len(result.get('prompt_tokens', []))} new tokens "
        f"(total length {len(full_sequence)})"
    )
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
            prompt_tokens = result.get("prompt_tokens", [])
            prompt_grids = split_grids_from_tokens(prompt_tokens)
            gen_grids = split_grids_from_tokens(
                [*prompt_tokens, *output_tokens, END_TOKEN_ID]
            )
            input_grid = prompt_grids[0] if prompt_grids else []
            output_grid = (
                gen_grids[1] if len(gen_grids) > 1 else tokens_to_grid(output_tokens)
            )
            to_plot = [input_grid, output_grid]
            plot_grids(to_plot, title=f"task {task_id} pair {pair_index}")
        except Exception as e:
            print(f"Plotting failed: {e}")


@torch.inference_mode()
def greedy_generate(
    model: TinyTransformer,
    prompt_tokens: torch.LongTensor,
    example_id: int,
    device: torch.device,
    cached_positions: Optional[torch.LongTensor] = None,
    log_time: bool = False,
) -> torch.LongTensor:
    prompts = [prompt_tokens.tolist()]
    cached = [cached_positions] if cached_positions is not None else None
    start_time = time.perf_counter() if log_time else None

    sequences = batched_greedy_generate(
        model=model,
        prompts=prompts,
        example_ids=[example_id],
        device=device,
        cached_positions=cached,
    )

    if start_time is not None:
        elapsed = time.perf_counter() - start_time
        print(
            f"Generation time: {elapsed:.3f}s for "
            f"{len(sequences[0]) - len(prompts[0])} new tokens"
        )

    return torch.tensor(sequences[0], dtype=torch.long)


def group_eval_sequences_by_task(
    evaluation: Dict[str, Dict[str, object]],
) -> Dict[str, Dict[str, List[List[int]]]]:
    """
    Restructures evaluation results into a format optimized for export:
    {
      "split_name": {
        "task_id": [ <sequence_pair_0>, <sequence_pair_1>, ... ]
      }
    }
    """
    output = {}
    for split, split_data in evaluation.items():
        results = split_data.get("results", [])

        # 1. Group by task_id -> pair_index -> sequence
        # We use a dict first to handle potential out-of-order processing
        task_map: Dict[str, Dict[int, List[int]]] = {}

        for res in results:
            task_id = res.get("task_id")
            pair_index = res.get("pair_index")
            sequence = res.get("sequence")

            # Basic validation
            if task_id is None or pair_index is None or sequence is None:
                continue

            if task_id not in task_map:
                task_map[task_id] = {}
            task_map[task_id][pair_index] = sequence

        # 2. Convert pair-maps to sorted lists
        output[split] = {}
        for task_id, pairs_dict in task_map.items():
            if not pairs_dict:
                output[split][task_id] = []
                continue

            # Determine list size based on the highest pair index found
            max_idx = max(pairs_dict.keys())

            # Pre-fill with empty lists to handle potential non-contiguous indices safely
            # (Though ARC data is usually contiguous 0..N)
            seq_list = [[] for _ in range(max_idx + 1)]

            for idx, seq in pairs_dict.items():
                seq_list[idx] = seq

            output[split][task_id] = seq_list

    return output
