"""Bounded frozen-checkpoint parity and timing through the normal launcher."""
import argparse
import json
import os
from pathlib import Path
from functools import partial
import signal
from statistics import median
import subprocess
import sys
from time import perf_counter


DEFAULTS = {"task_count": 8, "augmentation_attempts": 2, "batch_size": 8,
            "max_new_tokens": 128, "parity_cases": 4, "parity_tokens": 32, "repeats": 3,
            "compile": True, "compare_legacy": False, "compiled_probe": False,
            "max_seconds": None}


def validate_config(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError("Invalid speculative_benchmark object or unknown keys.")
    config = {**DEFAULTS, **value}
    for key, number in config.items():
        if key == "max_seconds" and number is None:
            continue
        if key in ("compile", "compare_legacy", "compiled_probe"):
            if not isinstance(number, bool):
                raise ValueError(f"speculative_benchmark.{key} must be boolean.")
            continue
        if isinstance(number, bool) or not isinstance(number, int) or number < 1:
            raise ValueError(f"speculative_benchmark.{key} must be an integer >= 1.")
    return config


def _serial_prefix_forward(model, **kwargs):
    return model.forward_generate(**kwargs)


def _block_prefix_forward(model, **kwargs):
    return model.forward_verify(**kwargs)


def fixed_prefix_logits(model, attempt, device, token_limit, max_new_tokens, *, block_width,
                        compile_decode=True, use_prepared_cache=True):
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
    identity = {}
    if model.config.pair_identity_vertices:
        pair_ids = torch.zeros_like(inputs)
        pair_ids[0, -prefix_len:] = torch.tensor(item.pair_identity_ids, device=device)
        identity['pair_identity_ids'] = pair_ids
    prefill = model.forward_generate(inputs, positions_3d=positions, attention_mask=mask, **identity, **shared)
    capacity = min(((padded + max_new_tokens + 127) // 128) * 128, model.config.max_seq_len)
    cache = tuple(tuple(torch.cat([v, v.new_zeros((*v.shape[:2], capacity - padded, v.shape[-1]))], dim=2)
                        for v in pair) for pair in prefill["past_key_values"])
    full_mask = torch.zeros(1, capacity, dtype=torch.bool, device=device)
    full_mask[:, :padded] = mask
    count = min(token_limit, len(attempt.tokens) - prefix_len, capacity - padded)
    start = padded
    if model.config.split_target_demo_attention:
        z = positions[..., 2]
        demos = mask & (z > 0) & (z < z[:, -1:])
        demo_ids = demos
        if use_prepared_cache:
            import target_demo_attention
            cache, prepared, full_mask, start = target_demo_attention.prepare_decode_cache(
                prefill['past_key_values'], demo_ids, mask, capacity-padded,
                [model.blocks[layer_idx].attention for layer_idx in model.layer_schedule])
            shared['target_decode_cache'] = tuple({'demo': c} for c in prepared)
        else:
            shared['demo_attention_mask'] = torch.nn.functional.pad(demo_ids, (0, capacity-padded))
    logits = [prefill["logits"][0, -1:].float().cpu()]
    # Probe shapes must not consume the production decode compile caches.
    name = "_compiled_serial_probe" if block_width == 1 else "_compiled_block_probe"
    entry = _serial_prefix_forward if block_width == 1 else _block_prefix_forward
    if compile_decode and not hasattr(model, name):
        setattr(model, name, evaluate._compile_inference(partial(entry, model)))
    decode = getattr(model, name) if compile_decode else partial(entry, model)
    for offset in range(0, count - 1, block_width):
        width = min(block_width, count - 1 - offset)
        slots = torch.arange(start + offset, start + offset + width, device=device)
        slots = slots if block_width == 1 else slots[None]
        full_mask[:, start + offset:start + offset + width] = True
        block_mask = evaluate._decode_mask(model, full_mask, slots, device)
        index = prefix_len + offset
        torch.compiler.cudagraph_mark_step_begin()
        output = decode(input_ids=attempt.tokens[None, index:index + width].to(device),
                        positions_3d=attempt.positions[None, index:index + width].to(device),
                        past_key_values=cache, attention_mask=full_mask, cache_position=slots,
                        decode_block_mask=block_mask, **shared)
        logits.append(output["logits"][0].float().cpu())
    return torch.cat(logits, dim=0)


def compare_logits(reference, actual):
    """Keep token changes and their reference margins alongside rounding drift."""
    import torch
    if reference.shape != actual.shape or not torch.isfinite(reference).all() or not torch.isfinite(actual).all():
        raise RuntimeError("Invalid or nonfinite logits in checkpoint comparison.")
    expected, observed = reference.argmax(-1), actual.argmax(-1)
    changed = (expected != observed).nonzero().flatten()
    top = reference.topk(2, dim=-1).values
    margins = top[:, 0] - top[:, 1]
    return dict(tokens=reference.size(0), argmax_disagreements=changed.numel(),
                max_abs_logit_delta=float((reference-actual).abs().max()),
                mean_abs_logit_delta=float((reference-actual).abs().mean()),
                minimum_reference_margin=float(margins.min()),
                disagreements=[dict(position=int(i), reference_token=int(expected[i]),
                                    actual_token=int(observed[i]), reference_margin=float(margins[i]))
                               for i in changed[:32]])


def _bounded_worker(command, seconds):
    """Kill this diagnostic's entire process group if compilation exceeds its budget."""
    process = subprocess.Popen(command, start_new_session=True)
    try:
        process.wait(timeout=seconds)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


def _run_bounded(plan, directory):
    deadline = perf_counter() + plan['speculative_benchmark']['max_seconds']
    summaries = {}
    for index, (name, cfg) in enumerate(plan['evals'].items()):
        output = directory if len(plan['evals']) == 1 else directory / name
        output.mkdir(parents=True, exist_ok=True)
        request = output / 'request.json'
        response = output / 'summary.json'
        request.write_text(json.dumps(dict(plan=plan, eval_cfg=cfg, output=output), default=str) + '\n')
        remaining = deadline - perf_counter()
        try:
            if remaining <= 0:
                summaries[name] = dict(status='not_run_time_budget')
            else:
                # Reserve each remaining checkpoint an equal share; unused time rolls over.
                seconds = remaining / (len(plan['evals']) - index)
                _bounded_worker([sys.executable, '-u', str(Path(__file__).resolve()), str(request)], seconds)
                summaries[name] = dict(status='completed', **json.loads(response.read_text()))
        except subprocess.TimeoutExpired:
            summaries[name] = dict(status='time_budget_exhausted')
        except subprocess.CalledProcessError as exc:
            summaries[name] = dict(status='worker_failed', returncode=exc.returncode)
        # Partial evidence survives timeout; an incomplete check never becomes a pass.
        partial_path = output / 'results.json'
        if partial_path.exists():
            partial_result = json.loads(partial_path.read_text())
            summaries[name]['partial_results'] = str(partial_path)
            if summaries[name]['status'] != 'completed':
                summaries[name]['parity_passed'] = False if partial_result.get('parity_passed') is False else None
        (directory / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
        print(f"Bounded checkpoint comparison {name}: {summaries[name]}", flush=True)
    return summaries


def run_benchmark(plan):
    directory = plan['run_dir'] / 'speculative_benchmark'
    directory.mkdir(parents=True, exist_ok=True)
    if plan['speculative_benchmark'].get('max_seconds') is not None:
        return _run_bounded(plan, directory)
    import torch
    summaries = {}
    for name, cfg in plan['evals'].items():
        summaries[name] = _run_checkpoint(plan, cfg, directory if len(plan['evals']) == 1 else directory / name)
        torch._dynamo.reset()
        torch.cuda.empty_cache()
    (directory / 'summary.json').write_text(json.dumps(summaries, indent=2)+'\n')


def _run_checkpoint(plan, eval_cfg, directory):
    import random
    import torch
    import evaluate
    from build import build_model_and_data

    if not torch.cuda.is_available():
        raise RuntimeError("Checkpoint speed comparison requires CUDA.")
    config = plan["speculative_benchmark"]
    directory.mkdir(parents=True, exist_ok=True)
    from launch_run import resolve_checkpoint_ref
    cfg = argparse.Namespace(**{**plan["shared"], **eval_cfg,
                               "checkpoint_path": resolve_checkpoint_ref(str(eval_cfg['checkpoint']), plan)})
    model, dataset, _, device, _ = build_model_and_data(cfg, is_eval=True, eval_splits=("test",), build_dataloader=False)
    model._loaded_checkpoint = None
    model = evaluate._prepare_model_for_inference(model)
    if model.config.n_predict < 2:
        raise ValueError("Checkpoint has no trained auxiliary prediction heads.")
    if not model.config.split_target_demo_attention and not all(block.attention._has_flex_attention for block in model.blocks):
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
               "task_ids": tasks, "prefix_parity": [], "legacy_prefix_parity": [],
               "timings": [], "generated_disagreements": []}

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
            reference = fixed_prefix_logits(model, attempt, device, config["parity_tokens"], config["max_new_tokens"], block_width=1, compile_decode=config['compile'])
            block = fixed_prefix_logits(model, attempt, device, config["parity_tokens"], config["max_new_tokens"], block_width=model.config.n_predict, compile_decode=config['compile'])
            results["prefix_parity"].append({"task_id": attempt.inference.task_id, "pair_index": attempt.inference.pair_index,
                **compare_logits(reference, block)})
            if config['compare_legacy']:
                legacy_logits = fixed_prefix_logits(model, attempt, device, config['parity_tokens'],
                    config['max_new_tokens'], block_width=1, compile_decode=False, use_prepared_cache=False)
                results['legacy_prefix_parity'].append(dict(task_id=attempt.inference.task_id,
                    pair_index=attempt.inference.pair_index, **compare_logits(legacy_logits, reference)))
            save()
            print(f"Prefix comparison {len(results['prefix_parity'])}/{len(attempts)}: "
                  f"{results['prefix_parity'][-1]}", flush=True)

        def generate(speculative, legacy=False):
            sequences = []
            for batch in batches:
                items = batch.items
                sequences.extend(evaluate.batched_greedy_generate(
                    model=model, prompts=[item.prompt for item in items], example_ids=[item.task_embedding_ids for item in items],
                    dihedral_ids=[item.dihedral_index for item in items], cached_positions=[item.positions for item in items],
                    pair_identity_ids=[item.pair_identity_ids for item in items],
                    padded_prompt_length=batch.padded_prompt_length, device=device, max_new_tokens=config["max_new_tokens"],
                    temperature=0, speculative=speculative, compile_decode=config['compile'],
                    use_prepared_cache=not legacy, compact_finished=not legacy))
            return sequences

        # Warm both complete routes, including masks, heads, and acceptance.
        print("Warming ordinary decoding...", flush=True)
        reference = generate(False)
        print("Warming speculative decoding...", flush=True)
        speculative = generate(True)
        legacy = generate(False, True) if config['compare_legacy'] else reference
        results['legacy_disagreements'] = sum(left != right for left, right in zip(legacy, reference))
        results['legacy_generated_disagreements'] = [dict(task_id=item.task_id, pair_index=item.pair_index,
            legacy=left, ordinary=right) for item, left, right in zip(work, legacy, reference) if left != right]
        results['generated_sequences_checked'] = len(reference)
        print("Warmup complete; comparing outputs and timing both routes...", flush=True)
        for index, (left, right) in enumerate(zip(reference, speculative)):
            if left != right:
                item = work[index]
                results["generated_disagreements"].append({"task_id": item.task_id, "pair_index": item.pair_index,
                    "ordinary": left, "speculative": right})
        save()
        # Save the useful original/cache/speculation evidence before potentially
        # expensive compiler work consumes a bounded diagnostic's remaining time.
        if config['compiled_probe']:
            print('Checking compiled serial and block cached decode graphs...', flush=True)
            attempt = attempts[0]
            reference_logits = fixed_prefix_logits(model, attempt, device, config['parity_tokens'],
                config['max_new_tokens'], block_width=1, compile_decode=config['compile'])
            compiled = fixed_prefix_logits(model, attempt, device, config['parity_tokens'],
                config['max_new_tokens'], block_width=1)
            results['compiled_probe'] = compare_logits(reference_logits, compiled)
            save()
            compiled_block = fixed_prefix_logits(model, attempt, device, config['parity_tokens'],
                config['max_new_tokens'], block_width=model.config.n_predict)
            results['compiled_block_probe'] = compare_logits(reference_logits, compiled_block)
            save()
        for repeat in range(config["repeats"]):
            modes = [('ordinary', False, False), ('speculative', True, False)]
            if config['compare_legacy']: modes.insert(0, ('legacy', False, True))
            for name, mode, old in (modes if repeat % 2 == 0 else reversed(modes)):
                torch.cuda.synchronize(device)
                started = perf_counter()
                sequences = generate(mode, old)
                torch.cuda.synchronize(device)
                elapsed = perf_counter() - started
                expected = legacy if old else speculative if mode else reference
                if sequences != expected:
                    raise RuntimeError("Repeated decoding changed generated tokens.")
                tokens = sum(len(seq) - len(item.prompt) for seq, item in zip(sequences, work))
                results["timings"].append({"mode": name, "repeat": repeat,
                                           "seconds": elapsed, "tokens": tokens, "tokens_per_second": tokens / elapsed})
                save()
                print(f'{name}: {elapsed:.3f}s, {tokens} tokens', flush=True)
    timings = results["timings"]
    ordinary = median(row["seconds"] for row in timings if row["mode"] == "ordinary")
    speculative_time = median(row["seconds"] for row in timings if row["mode"] == "speculative")
    results["speedup"] = ordinary / speculative_time
    results['cache_speedup'] = median(row['seconds'] for row in timings if row['mode'] == 'legacy') / ordinary if config['compare_legacy'] else None
    probes = [*results['prefix_parity'], *results['legacy_prefix_parity'],
              *([results['compiled_probe'], results['compiled_block_probe']] if config['compiled_probe'] else [])]
    results["parity_passed"] = not results['legacy_disagreements'] and not results["generated_disagreements"] and all(row["argmax_disagreements"] == 0 for row in probes)
    save()
    print(f"Speculative checkpoint comparison: parity={results['parity_passed']}, speedup={results['speedup']:.3f}x", flush=True)
    if not results["parity_passed"]:
        raise RuntimeError("Speculative decoding disagreed with ordinary decoding; inspect saved comparison.")
    return {key: results[key] for key in ('checkpoint', 'parity_passed', 'cache_speedup', 'speedup', 'timings')}


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root / 'src')]
    request = json.loads(Path(sys.argv[1]).read_text())
    request['plan']['run_dir'] = Path(request['plan']['run_dir'])
    output = Path(request['output'])
    summary = _run_checkpoint(request['plan'], request['eval_cfg'], output)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
