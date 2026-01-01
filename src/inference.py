import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import torch

from tinytransformer import TinyTransformer
from utils import (
    END_TOKEN_ID,
    IO_SEPARATOR_TOKEN_ID,
    NEXT_LINE_TOKEN_ID,
    START_TOKEN_ID,
    apply_color_permutation_to_grid,
    apply_color_permutation_to_tokens,
    compute_positions_3d,
    extract_output_tokens,
    grid_to_tokens,
    tokens_to_grid,
    tokens_to_string,
)

DEFAULT_MAX_NEW_TOKENS = 931


# 1. COMPILED HELPER FOR GRID LOGIC
@torch.compile(mode="default", fullgraph=True)
def _compiled_grid_update(state, token_ids, start_id, sep_id, end_id, nl_id):
    x, y, z = state.unbind(-1)

    pos_x = torch.clamp(x, min=0, max=30)
    pos_y = torch.clamp(y, min=0, max=29)
    pos_z = z

    is_start = token_ids == start_id
    is_sep = token_ids == sep_id
    is_end = token_ids == end_id
    is_newline = token_ids == nl_id

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

    return torch.stack([next_x, next_y, next_z], dim=-1), torch.stack(
        [pos_x, pos_y, pos_z], dim=-1
    )


class BatchGridState:
    def __init__(self, initial_state: torch.Tensor) -> None:
        self.state = initial_state.clone().long()

    def update(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.view(-1).to(device=self.state.device)
        self.state, positions = _compiled_grid_update(
            self.state,
            token_ids,
            START_TOKEN_ID,
            IO_SEPARATOR_TOKEN_ID,
            END_TOKEN_ID,
            NEXT_LINE_TOKEN_ID,
        )
        # Clone to break potential cudagraph output reuse between steps
        self.state = self.state.clone()
        return positions.clone()


def _select_next_token(
    logits: torch.Tensor,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    last_logits = logits[:, -1, :]

    if (temperature is None and top_k is None) or (
        temperature is not None and temperature <= 0
    ):
        return torch.argmax(last_logits, dim=-1)

    if temperature is None:
        temperature = 1.0

    use_top_k = top_k is not None and top_k > 0
    if use_top_k:
        top_k = int(top_k)
        if top_k == 1:
            return torch.argmax(last_logits, dim=-1)
        top_k = min(top_k, last_logits.size(-1))
        top_values, top_indices = torch.topk(last_logits, top_k, dim=-1)
        scaled = (top_values / temperature).float()
        probs = torch.softmax(scaled, dim=-1)
        next_index = torch.multinomial(probs, num_samples=1)
        return top_indices.gather(-1, next_index).squeeze(-1)

    scaled = (last_logits / temperature).float()
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.inference_mode()
def batched_greedy_generate(
    model,
    prompts,
    example_ids,
    device,
    max_new_tokens=931,
    cached_positions=None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
):
    model.eval()
    model.to(dtype=torch.bfloat16)

    batch_size = len(prompts)

    # --- Setup Inputs ---
    example_ids_tensor = torch.tensor(example_ids, dtype=torch.long, device=device)
    # _left_pad_sequences and _pad_cached_positions from your original code...
    # Assuming they are available in scope or imported
    input_ids, attention_mask = _left_pad_sequences(prompts, END_TOKEN_ID, device)

    # --- 2. Calculate DYNAMIC Buffer Size ---
    current_len = input_ids.size(1)

    # We only need space for the prompt + new tokens
    batch_max_needed = current_len + max_new_tokens

    # OPTIONAL: Round up to nearest 64 to reduce torch.compile recompilation frequency
    batch_max_needed = (batch_max_needed + 127) // 128 * 128

    # Clamp to model capacity
    max_model_len = min(batch_max_needed, model.config.max_seq_len)

    # Pre-calculate 3D positions for prompt
    if cached_positions and all(p is not None for p in cached_positions):
        prompt_positions = _pad_cached_positions(
            [p for p in cached_positions if p is not None], input_ids.size(1), device
        )
    else:
        prompt_positions = compute_positions_3d(input_ids, attention_mask).to(
            device=device, dtype=torch.long
        )

    # Calculate initial grid state
    initial_state, finished = _derive_initial_state_from_prompt(
        input_ids, prompt_positions
    )
    grid_state = BatchGridState(initial_state)

    example_embeds = model.example_embedding(example_ids_tensor).to(
        dtype=torch.bfloat16
    )
    current_len = input_ids.size(1)

    # --- 1. PROMPT PASS (Fill Cache) ---
    # We create a FULL SIZED mask: [B, MaxLen]
    # We set future positions to False
    full_attention_mask = torch.zeros(
        (batch_size, max_model_len), dtype=torch.bool, device=device
    )
    full_attention_mask[:, :current_len] = attention_mask

    # Prompt pass uses the sliced view for computation, but returns KV cache we will size up
    outputs = model.forward_generate(
        input_ids=input_ids,
        example_ids=example_ids_tensor,
        past_key_values=None,
        positions_3d=prompt_positions,
        attention_mask=attention_mask,  # Prompt pass uses tight mask
        example_embeds=example_embeds,
    )
    logits = outputs["logits"]
    prompt_kvs = outputs["past_key_values"]

    # --- 2. SETUP STATIC KV CACHE ---
    # Convert the prompt KVs into a fixed size buffer [B, H, MaxLen, D]
    past_key_values = []
    for k, v in prompt_kvs:
        # k, v are [B, H, PromptLen, D]
        B, H, L, D = k.shape
        # Create full-sized buffer
        k_buf = torch.zeros(
            (B, H, max_model_len, D), dtype=torch.bfloat16, device=device
        )
        v_buf = torch.zeros(
            (B, H, max_model_len, D), dtype=torch.bfloat16, device=device
        )
        # Copy prompt history into buffer
        k_buf[:, :, :L, :] = k
        v_buf[:, :, :L, :] = v
        past_key_values.append((k_buf, v_buf))
    past_key_values = tuple(past_key_values)

    # --- 3. COMPILE GENERATION STEP ---
    # We compile the forward pass specifically for the decoding step
    if not hasattr(model, "_compiled_decode"):
        print("Compiling model for decoding step...")
        model._compiled_decode = torch.compile(
            model.forward_generate, mode="default", fullgraph=True
        )

    # --- 4. GENERATION LOOP ---
    cache_position = torch.tensor([current_len], dtype=torch.long, device=device)
    steps_remaining = min(max_new_tokens, max_model_len - current_len)

    # Output buffer
    generated_tokens_buffer = torch.full(
        (batch_size, steps_remaining), END_TOKEN_ID, dtype=torch.long, device=device
    )

    # Loop over steps
    # We use a Python loop, but the heavy lifting is inside the compiled regions
    for step_i in range(steps_remaining):
        if finished.all():
            break

        # Mark a new cudagraph step to avoid reusing outputs as inputs
        torch.compiler.cudagraph_mark_step_begin()

        # Greedy decode unless sampling is enabled via temperature/top_k
        next_token = _select_next_token(logits, temperature=temperature, top_k=top_k)
        next_token = torch.where(
            finished, torch.tensor(END_TOKEN_ID, device=device), next_token
        )

        # Save token
        generated_tokens_buffer[:, step_i] = next_token

        # Update finished status
        finished = finished | (next_token == END_TOKEN_ID)

        # Update Grid (Compiled)
        token_positions = grid_state.update(next_token).unsqueeze(1)

        # Update Mask (In-place on the full buffer)
        # Note: In static graph world, we pass the WHOLE mask.
        # We update the mask bit for the current position.
        full_attention_mask.index_fill_(1, cache_position, True)
        # However, we must ensure we handle finished sequences.
        # Actually, if we just keep attending to padding (END tokens), it's fine for shape consistency.
        # Ideally, we mask out finished ones, but for batching we usually just let them generate padding.
        # Specifically: The mask must be True for the *new* token index.
        # Since we use index_fill with a scalar tensor, it sets that column to True for ALL batch items.
        # This is acceptable for simple batching; masking finished rows is handled by "finished" logic.

        # Run Model (Compiled)
        # Note: We pass the FULL STATIC mask. No slicing.
        # We pass the cache_position as a TENSOR.
        outputs = model._compiled_decode(
            input_ids=next_token.unsqueeze(1),
            example_ids=example_ids_tensor,
            past_key_values=past_key_values,  # Passing the fixed buffers
            positions_3d=token_positions,
            attention_mask=full_attention_mask,  # Full [B, MaxLen]
            cache_position=cache_position,  # Tensor(1)
            example_embeds=example_embeds,
        )
        logits = outputs["logits"]

        # Increment position
        cache_position.add_(1)

    # --- 5. FINALIZE ---
    # (Existing code to convert buffer to lists)
    generated_cpu = generated_tokens_buffer.tolist()
    results = []
    for i, prompt in enumerate(prompts):
        gen_seq = []
        for token in generated_cpu[i]:
            gen_seq.append(token)
            if token == END_TOKEN_ID:
                break
        results.append(list(prompt) + gen_seq)
    return results


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
    input_ids: torch.Tensor, positions_3d: torch.Tensor
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


def _build_prompt_from_tokens(tokens: Sequence[int]) -> List[int]:
    if IO_SEPARATOR_TOKEN_ID not in tokens:
        raise ValueError("Prompt sequence is missing <input_output_separator>.")
    sep_idx = tokens.index(IO_SEPARATOR_TOKEN_ID)
    return list(tokens[: sep_idx + 1])


def _prepare_examples_for_inference(
    examples: Sequence[object],
    include_targets: bool = False,
    solutions: Optional[Dict[Tuple[str, int], List[List[int]]]] = None,
    color_mappings: Optional[Sequence[Optional[Sequence[int]]]] = None,
    color_apply_fn: Optional[Callable[[str], bool]] = None,
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

    for idx, ex in enumerate(examples):
        if not hasattr(ex, "tokens"):
            raise ValueError("Examples must provide a 'tokens' attribute.")
        raw_tokens = ex.tokens.tolist()
        split = getattr(ex, "split", None)

        # Select the specific mapping for this example
        mapping = color_mappings[idx] if color_mappings is not None else None

        should_color = mapping is not None and (
            color_apply_fn is None or color_apply_fn(split)
        )
        tokens = (
            apply_color_permutation_to_tokens(raw_tokens, mapping)
            if should_color
            else raw_tokens
        )
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
                target_grid = solutions[key]
                if should_color:
                    target_grid = apply_color_permutation_to_grid(target_grid, mapping)
                targets = grid_to_tokens(target_grid)
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
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    target_output_tokens: Optional[Sequence[Sequence[int]]] = None,
) -> List[Dict[str, object]]:
    sequences = batched_greedy_generate(
        model=model,
        prompts=prompts,
        example_ids=example_ids,
        device=device,
        max_new_tokens=max_new_tokens,
        cached_positions=cached_positions,
        temperature=temperature,
        top_k=top_k,
    )
    return _build_generation_results(
        sequences=sequences,
        metadata=metadata,
        prompts=prompts,
        target_output_tokens=target_output_tokens
        if target_output_tokens is not None
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
    color_mappings: Optional[Sequence[Sequence[int]]] = None,
    color_mappings_by_task: Optional[Dict[str, Sequence[Sequence[int]]]] = None,
    color_apply_fn: Optional[Callable[[str], bool]] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
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

    # Flatten the workload: Create a job for every (Example, ColorMapping) pair
    work_items = []
    global_mappings = list(color_mappings) if color_mappings is not None else None
    for ex in examples:
        task_mappings = None
        if color_mappings_by_task is not None:
            task_mappings = color_mappings_by_task.get(getattr(ex, "task_id", None))
        elif global_mappings is not None:
            task_mappings = global_mappings
        if not task_mappings:
            task_mappings = [None]
        for c_idx, mapping in enumerate(task_mappings):
            work_items.append((ex, c_idx, mapping))

    # Sort ALL work items by sequence length (descending) to minimize padding.
    # Note: Color permutation maps digits 1-to-1, so ex.seq_len is invariant.
    work_items.sort(key=lambda item: item[0].seq_len, reverse=True)

    all_results: List[Dict[str, object]] = []

    for start in range(0, len(work_items), batch_size):
        chunk = work_items[start : start + batch_size]

        # Unzip the batch components
        batch_examples = [item[0] for item in chunk]
        batch_c_indices = [item[1] for item in chunk]
        batch_mappings = [item[2] for item in chunk]

        (prompts, example_ids, metadata, cached_positions, target_output_tokens) = (
            _prepare_examples_for_inference(
                batch_examples,
                include_targets=include_targets,
                solutions=solutions,
                color_mappings=batch_mappings,  # Pass the batch-specific mappings
                color_apply_fn=color_apply_fn,
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
            temperature=temperature,
            top_k=top_k,
            target_output_tokens=target_output_tokens if include_targets else None,
        )

        print(
            f"[{split}] Finished batch {start // batch_size + 1} / {(len(work_items) + batch_size - 1) // batch_size}"
        )

        # Attach the correct color index to each result and collect
        for res, c_idx in zip(batch_results, batch_c_indices):
            if color_mappings_by_task is not None or color_mappings is not None:
                res["color_permutation_index"] = c_idx
            all_results.append(res)

    return all_results
