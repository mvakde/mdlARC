"""Bounded frozen-checkpoint parity and timing through the normal launcher."""
import argparse
import json
from pathlib import Path
from functools import partial
from statistics import median
from time import perf_counter


DEFAULTS = {"task_count": 8, "augmentation_attempts": 2, "batch_size": 8,
            "max_new_tokens": 128, "parity_cases": 4, "parity_tokens": 32, "repeats": 3}


def validate_config(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError("Invalid speculative_benchmark object or unknown keys.")
    config = {**DEFAULTS, **value}
    for key, number in config.items():
        if isinstance(number, bool) or not isinstance(number, int) or number < 1:
            raise ValueError(f"speculative_benchmark.{key} must be an integer >= 1.")
    return config


def _serial_prefix_forward(model, **kwargs):
    return model.forward_generate(**kwargs)


def _block_prefix_forward(model, **kwargs):
    return model.forward_verify(**kwargs)


def fixed_prefix_logits(model, attempt, device, token_limit, max_new_tokens, *, block_width):
    """Compare identical target prefixes, not divergent generated histories."""
    import torch
    import evaluate

    item = attempt.inference
    prefix_len = len(item.prompt)
    padded = evaluate._prompt_bucket_length(prefix_len, max_seq_len=model.config.max_seq_len,
                                            max_new_tokens=max_new_tokens)
    inputs, mask = evaluate._left_pad_sequences([item.prompt], evaluate.END_PAIR_TOKEN_ID, device,
                                               padded_length=padded)
    positions = evaluate._pad_cached_positions([item.positions], padded, device)
    ids = torch.tensor([item.task_embedding_ids], dtype=torch.int32, device=device)
    dihedrals = torch.tensor([item.dihedral_index], dtype=torch.int32, device=device)
    shared = dict(example_ids=ids, dihedral_ids=dihedrals,
                  conditioning_embeds=model.compute_conditioning_embeds(ids, dihedrals),
                  value_embeds=model.compute_value_embeds(ids))
    prefill = model.forward_generate(inputs, positions_3d=positions, attention_mask=mask, **shared)
    capacity = min(((padded + max_new_tokens + 127) // 128) * 128, model.config.max_seq_len)
    cache = tuple(tuple(torch.cat([v, v.new_zeros((*v.shape[:2], capacity - padded, v.shape[-1]))], dim=2)
                        for v in pair) for pair in prefill["past_key_values"])
    full_mask = torch.zeros(1, capacity, dtype=torch.bool, device=device)
    full_mask[:, :padded] = mask
    count = min(token_limit, len(attempt.tokens) - prefix_len, capacity - padded)
    logits = [prefill["logits"][0, -1:].float().cpu()]
    # Probe shapes must not consume the production decode compile caches.
    name = "_compiled_serial_probe" if block_width == 1 else "_compiled_block_probe"
    if not hasattr(model, name):
        entry = _serial_prefix_forward if block_width == 1 else _block_prefix_forward
        setattr(model, name, torch.compile(partial(entry, model), mode="default", fullgraph=True))
    decode = getattr(model, name)
    for offset in range(0, count - 1, block_width):
        width = min(block_width, count - 1 - offset)
        slots = torch.arange(padded + offset, padded + offset + width, device=device)
        slots = slots if block_width == 1 else slots[None]
        full_mask[:, padded + offset:padded + offset + width] = True
        block_mask = model.build_decode_block_mask_for_step(full_mask, slots, capacity, device) if device.type == "cuda" else None
        index = prefix_len + offset
        torch.compiler.cudagraph_mark_step_begin()
        output = decode(input_ids=attempt.tokens[None, index:index + width].to(device),
                        positions_3d=attempt.positions[None, index:index + width].to(device),
                        past_key_values=cache, attention_mask=full_mask, cache_position=slots,
                        decode_block_mask=block_mask, **shared)
        logits.append(output["logits"][0].float().cpu())
    return torch.cat(logits, dim=0)


def run_benchmark(plan):
    import random
    import torch
    import evaluate
    from build import build_model_and_data

    if not torch.cuda.is_available():
        raise RuntimeError("Checkpoint speed comparison requires CUDA.")
    config = plan["speculative_benchmark"]
    directory = plan["run_dir"] / "speculative_benchmark"
    directory.mkdir(parents=True, exist_ok=True)
    eval_cfg = next(iter(plan["evals"].values()))
    cfg = argparse.Namespace(**{**plan["shared"], **eval_cfg,
                               "checkpoint_path": plan["pretrain"]["checkpoint_path"]})
    model, dataset, _, device, _ = build_model_and_data(cfg, is_eval=True, eval_splits=("test",), build_dataloader=False)
    model._loaded_checkpoint = None
    model = evaluate._prepare_model_for_inference(model)
    if model.config.n_predict < 2:
        raise ValueError("Checkpoint has no trained auxiliary prediction heads.")
    if not all(block.attention._has_flex_attention for block in model.blocks):
        raise RuntimeError("FlexAttention is required to compare the existing CUDA decode path.")
    tasks = sorted({example.task_id for example in dataset.iter_examples(split="test")})
    random.Random(cfg.seed).shuffle(tasks)
    tasks = sorted(tasks[:config["task_count"]])
    work = evaluate._build_inference_attempts(dataset, "test", tasks, None, config["augmentation_attempts"], True)
    if not work:
        raise ValueError("No test attempts selected.")
    batches = evaluate._build_inference_batches(work, batch_size=config["batch_size"],
                                                max_seq_len=model.config.max_seq_len, max_new_tokens=config["max_new_tokens"])
    work = [item for batch in batches for item in batch.items]
    results = {"config": config, "checkpoint": str(cfg.checkpoint_path), "torch": torch.__version__,
               "gpu": torch.cuda.get_device_name(device), "n_predict": model.config.n_predict,
               "task_ids": tasks, "prefix_parity": [], "timings": [], "generated_disagreements": []}

    def save():
        path = directory / "results.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(results, indent=2) + "\n")
        temporary.replace(path)

    solutions = json.loads(Path(cfg.solutions_path).read_text())
    selected, seen = [], set()
    for item in work:
        identity = (item.task_id, item.pair_index)
        if identity not in seen:
            seen.add(identity)
            selected.append(item)
        if len(selected) >= config["parity_cases"]:
            break
    attempts = evaluate._build_accuracy_attempts(dataset, "test", selected, solutions, model.config.max_seq_len)
    with torch.inference_mode():
        for attempt in attempts:
            reference = fixed_prefix_logits(model, attempt, device, config["parity_tokens"], config["max_new_tokens"], block_width=1)
            block = fixed_prefix_logits(model, attempt, device, config["parity_tokens"], config["max_new_tokens"], block_width=model.config.n_predict)
            if not torch.isfinite(reference).all() or not torch.isfinite(block).all():
                raise RuntimeError("Nonfinite logits in checkpoint comparison.")
            results["prefix_parity"].append({"task_id": attempt.inference.task_id, "pair_index": attempt.inference.pair_index,
                "tokens": reference.size(0), "argmax_disagreements": int((reference.argmax(-1) != block.argmax(-1)).sum()),
                "max_abs_logit_delta": float((reference - block).abs().max())})
            save()
            print(f"Prefix comparison {len(results['prefix_parity'])}/{len(attempts)}: "
                  f"{results['prefix_parity'][-1]}", flush=True)

        def generate(speculative):
            sequences = []
            for batch in batches:
                items = batch.items
                sequences.extend(evaluate.batched_greedy_generate(
                    model=model, prompts=[item.prompt for item in items], example_ids=[item.task_embedding_ids for item in items],
                    dihedral_ids=[item.dihedral_index for item in items], cached_positions=[item.positions for item in items],
                    padded_prompt_length=batch.padded_prompt_length, device=device, max_new_tokens=config["max_new_tokens"],
                    temperature=0, speculative=speculative))
            return sequences

        # Warm both complete routes, including masks, heads, and acceptance.
        print("Warming ordinary decoding...", flush=True)
        reference = generate(False)
        print("Warming speculative decoding...", flush=True)
        speculative = generate(True)
        print("Warmup complete; comparing outputs and timing both routes...", flush=True)
        for index, (left, right) in enumerate(zip(reference, speculative)):
            if left != right:
                item = work[index]
                results["generated_disagreements"].append({"task_id": item.task_id, "pair_index": item.pair_index,
                    "ordinary": left, "speculative": right})
        save()
        for repeat in range(config["repeats"]):
            for mode in ([False, True] if repeat % 2 == 0 else [True, False]):
                torch.cuda.synchronize(device)
                started = perf_counter()
                sequences = generate(mode)
                torch.cuda.synchronize(device)
                elapsed = perf_counter() - started
                expected = speculative if mode else reference
                if sequences != expected:
                    raise RuntimeError("Repeated decoding changed generated tokens.")
                tokens = sum(len(seq) - len(item.prompt) for seq, item in zip(sequences, work))
                results["timings"].append({"mode": "speculative" if mode else "ordinary", "repeat": repeat,
                                           "seconds": elapsed, "tokens": tokens, "tokens_per_second": tokens / elapsed})
                save()
    timings = results["timings"]
    ordinary = median(row["seconds"] for row in timings if row["mode"] == "ordinary")
    speculative_time = median(row["seconds"] for row in timings if row["mode"] == "speculative")
    results["speedup"] = ordinary / speculative_time
    results["parity_passed"] = not results["generated_disagreements"] and all(row["argmax_disagreements"] == 0 for row in results["prefix_parity"])
    save()
    print(f"Speculative checkpoint comparison: parity={results['parity_passed']}, speedup={results['speedup']:.3f}x", flush=True)
    if not results["parity_passed"]:
        raise RuntimeError("Speculative decoding disagreed with ordinary decoding; inspect saved comparison.")
