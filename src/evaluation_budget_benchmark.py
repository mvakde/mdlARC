"""Bounded frozen-checkpoint batching diagnostics through the normal launcher.

Timing and memory instrumentation live here, never in production evaluation.
Workers isolate compilation caches and release CUDA allocations between trials.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
from statistics import median
import subprocess
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    "variants": [
        {"token_budget": 65536, "batch_size": 100},
        {"token_budget": 262144, "batch_size": 100},
        {"token_budget": 65536, "batch_size": 512},
        {"token_budget": 262144, "batch_size": 512},
        {"token_budget": 393216, "batch_size": 512},
        {"token_budget": 524288, "batch_size": 512},
    ],
    "task_count": 48, "augmentation_attempts": 16, "max_new_tokens": 930,
    "repeats": 3, "trial_timeout_seconds": 1800, "headroom_fraction": 0.15,
}


def validate_config(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError("Invalid evaluation_budget_benchmark object or unknown keys.")
    config = {**DEFAULTS, **value}
    for key in set(DEFAULTS) - {"variants", "headroom_fraction"}:
        number = config[key]
        if isinstance(number, bool) or not isinstance(number, int) or number < 1:
            raise ValueError(f"evaluation_budget_benchmark.{key} must be a positive integer.")
    fraction = config["headroom_fraction"]
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
        raise ValueError("evaluation_budget_benchmark.headroom_fraction must be between 0 and 1.")
    variants = config["variants"]
    if not isinstance(variants, list) or not variants:
        raise ValueError("evaluation_budget_benchmark.variants must be a nonempty list.")
    seen = set()
    for variant in variants:
        if not isinstance(variant, dict) or set(variant) != {"token_budget", "batch_size"}:
            raise ValueError("Each budget variant requires token_budget and batch_size.")
        for number in variant.values():
            if isinstance(number, bool) or not isinstance(number, int) or number < 1:
                raise ValueError("Budget variants require positive integer values.")
        key = variant_key(variant)
        if key in seen:
            raise ValueError("Budget variants must be unique.")
        seen.add(key)
    return config


def variant_key(variant):
    return f"tokens{variant['token_budget']}-cap{variant['batch_size']}"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def compare_outputs(reference, actual):
    if reference["selection_sha256"] != actual["selection_sha256"]:
        raise ValueError("Budget trials used different inference attempts.")
    comparisons = {}
    for mode, left in reference["outputs"].items():
        right = actual["outputs"][mode]
        if len(left) != len(right):
            raise ValueError("Budget trial output counts differ.")
        differing = [index for index, (a, b) in enumerate(zip(left, right)) if a != b]
        comparisons[mode] = {"changed_attempts": len(differing), "attempt_indices": differing,
                             "total_attempts": len(left)}
    return comparisons


def run_benchmark(plan):
    config = plan["evaluation_budget_benchmark"]
    directory = plan["run_dir"] / "evaluation_budget_benchmark"
    directory.mkdir(parents=True, exist_ok=True)
    results = {"config": config, "variants": {}, "failures": {}, "comparisons": {}}
    for index, variant in enumerate(config["variants"]):
        key = variant_key(variant)
        output = directory / f"{key}.json"
        print(f"Evaluation budget diagnostic: {key}", flush=True)
        try:
            subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                            "--config", str(plan["spec_path"]), "--variant-index", str(index),
                            "--output", str(output)], cwd=ROOT, check=True,
                           timeout=config["trial_timeout_seconds"])
            result = json.loads(output.read_text())
            # Full token sequences remain in per-variant files for inspection.
            results["variants"][key] = {k: v for k, v in result.items() if k != "outputs"}
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            results["failures"][key] = str(exc)
        write_json(directory / "results.json", results)
    reference = config["variants"][0]
    ref_path = directory / f"{variant_key(reference)}.json"
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())
        if ref.get("outputs"):
            for variant in config["variants"][1:]:
                key = variant_key(variant)
                path = directory / f"{key}.json"
                if path.exists():
                    actual = json.loads(path.read_text())
                    if actual.get("median_seconds"):
                        results["comparisons"][key] = compare_outputs(ref, actual)
    write_json(directory / "results.json", results)
    print(json.dumps({key: {"memory_safe": row.get("memory_safe"), "median_seconds": row.get("median_seconds")}
                      for key, row in results["variants"].items()}, indent=2), flush=True)
    if results["failures"]:
        raise RuntimeError("Budget diagnostic incomplete; inspect saved results.")


def worker(plan, variant_index, output_path):
    import torch
    import evaluate
    from build import build_model_and_data
    from common import compute_positions_3d, encode_example

    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation budget diagnostics require CUDA.")
    config = plan["evaluation_budget_benchmark"]
    variant = config["variants"][variant_index]
    eval_cfg = next(iter(plan["evals"].values()))
    cfg = argparse.Namespace(**{**plan["shared"], **eval_cfg,
                               "checkpoint_path": plan["pretrain"]["checkpoint_path"]})
    model, dataset, _, device, _ = build_model_and_data(cfg, is_eval=True, eval_splits=("test",), build_dataloader=False)
    model._loaded_checkpoint = None
    model = evaluate._prepare_model_for_inference(model)
    if not all(block.attention._has_flex_attention for block in model.blocks):
        raise RuntimeError("The existing FlexAttention decode backend is required.")
    # Many intentionally different batch shapes; never change production limits.
    torch._dynamo.config.recompile_limit = 32
    tasks = sorted({example.task_id for example in dataset.iter_examples(split="test")})
    unaugmented = evaluate._build_inference_attempts(dataset, "test", tasks, None, 1, False)
    lengths = {task: max(len(item.prompt) for item in unaugmented if item.task_id == task) for task in tasks}
    longest = sorted(tasks, key=lambda task: (-lengths[task], task))[:config["task_count"] // 2]
    remaining = [task for task in tasks if task not in longest]
    random.Random(cfg.seed).shuffle(remaining)
    selected = sorted(longest + remaining[:config["task_count"] - len(longest)])
    work = evaluate._build_inference_attempts(dataset, "test", selected, None, config["augmentation_attempts"], True)
    if not work:
        raise ValueError("No inference attempts selected.")
    batches = evaluate._build_inference_batches(work, max_seq_len=model.config.max_seq_len,
                max_new_tokens=config["max_new_tokens"], **variant)
    indices = {id(item): index for index, item in enumerate(work)}
    identity = [{"task": w.task_id, "pair": w.pair_index, "prompt": w.prompt,
                 "ids": w.task_embedding_ids, "dihedral": w.dihedral_index, "mapping": w.color_mapping} for w in work]
    result = {"variant": variant, "selection_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
              "task_ids": selected, "prompt_lengths": {"min": min(len(w.prompt) for w in work), "max": max(len(w.prompt) for w in work)},
              "attempt_count": len(work), "batches": [{"count": len(b.items), "padded_prompt_length": b.padded_prompt_length} for b in batches],
              "environment": {"gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
                              "dtype": str(next(model.parameters()).dtype), "n_layers": model.config.n_layers,
                              "d_model": model.config.d_model, "max_seq_len": model.config.max_seq_len},
              "stress": [], "timings": [], "outputs": {}}
    save = lambda: write_json(output_path, result)

    def measure(fn):
        torch.cuda.synchronize(device)
        free, total = torch.cuda.mem_get_info(device)
        reserved = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = perf_counter()
        outputs = fn()
        torch.cuda.synchronize(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        peak_device = total - free + max(0, peak_reserved - reserved)
        row = {"seconds": perf_counter() - started, "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
               "peak_reserved_gib": peak_reserved / 2**30, "total_gib": total / 2**30,
               "estimated_device_headroom_gib": (total - peak_device) / 2**30,
               "headroom_passed": peak_device <= total * (1 - config["headroom_fraction"])}
        return outputs, row

    def generate(speculative, *, progress=False):
        outputs = [None] * len(work)
        for batch_index, batch in enumerate(batches):
            items = batch.items
            if progress:
                print(f"Warmup batch {batch_index + 1}/{len(batches)}: "
                      f"rows={len(items)}, prompt={batch.padded_prompt_length}", flush=True)
            sequences = evaluate.batched_greedy_generate(model=model, device=device,
                prompts=[w.prompt for w in items], example_ids=[w.task_embedding_ids for w in items],
                dihedral_ids=[w.dihedral_index for w in items], cached_positions=[w.positions for w in items],
                padded_prompt_length=batch.padded_prompt_length, max_new_tokens=config["max_new_tokens"],
                temperature=0, speculative=speculative)
            for item, sequence in zip(items, sequences):
                outputs[indices[id(item)]] = sequence[len(item.prompt):]
        return outputs

    with torch.inference_mode():
        try:
            # Exercise both buckets through every output/cache slot, even if
            # the checkpoint would normally emit END_PAIR early. Only the
            # benchmark's next-token selector is replaced; kernels/weights stay.
            for rows in (15, 30):
                prompt = encode_example([[0] * 30 for _ in range(rows)], include_output=False)
                padded = evaluate._prompt_bucket_length(len(prompt), max_seq_len=model.config.max_seq_len,
                                                        max_new_tokens=config["max_new_tokens"])
                count = min(variant["batch_size"], max(1, variant["token_budget"] // padded))
                positions = compute_positions_3d(torch.tensor(prompt)[None])[0]
                item = replace(work[0], prompt=prompt, positions=positions, dihedral_index=0)
                previous = evaluate._select_next_token
                evaluate._select_next_token = lambda logits, **kwargs: torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
                try:
                    print(f"Full-output stress: prompt={padded}, batch={count}, output={config['max_new_tokens']}", flush=True)
                    sequences, memory = measure(lambda: evaluate.batched_greedy_generate(
                        model=model, device=device, prompts=[prompt] * count,
                        example_ids=[item.task_embedding_ids] * count, dihedral_ids=[0] * count,
                        cached_positions=[positions] * count, padded_prompt_length=padded,
                        max_new_tokens=config["max_new_tokens"], temperature=0))
                    if any(len(seq) - len(prompt) != config["max_new_tokens"] for seq in sequences):
                        raise RuntimeError("Full-output stress did not consume every output slot.")
                    result["stress"].append({"padded_prompt_length": padded, "batch_size": count,
                                             "output_tokens_per_row": config["max_new_tokens"], **memory})
                    del sequences
                finally:
                    evaluate._select_next_token = previous
                save()
            modes = (False, True) if model.config.n_predict > 1 else (False,)
            for mode in modes:
                name = "speculative" if mode else "ordinary"
                print(f"Warming {name}: {len(work)} attempts, {len(batches)} batches", flush=True)
                result["outputs"][name], memory = measure(lambda: generate(mode, progress=True))
                result.setdefault("warmups", {})[name] = memory
                save()
            for repeat in range(config["repeats"]):
                for mode in (modes if repeat % 2 == 0 else tuple(reversed(modes))):
                    name = "speculative" if mode else "ordinary"
                    outputs, timing = measure(lambda: generate(mode))
                    if outputs != result["outputs"][name]:
                        raise RuntimeError("Repeated greedy generation changed outputs.")
                    tokens = sum(len(seq) for seq in outputs)
                    result["timings"].append({"mode": name, "repeat": repeat, "tokens": tokens,
                                              "tokens_per_second": tokens / timing["seconds"], **timing})
                    save()
                    print(f"{variant_key(variant)} {name}: {timing['seconds']:.3f}s, {tokens} tokens", flush=True)
            result["median_seconds"] = {name: median(row["seconds"] for row in result["timings"] if row["mode"] == name)
                                        for name in result["outputs"]}
            memories = result["stress"] + list(result["warmups"].values()) + result["timings"]
            result["memory_safe"] = all(row["headroom_passed"] for row in memories)
            result["ordinary_vs_speculative"] = sum(left != right for left, right in zip(
                result["outputs"]["ordinary"], result["outputs"].get("speculative", result["outputs"]["ordinary"])))
        except torch.cuda.OutOfMemoryError:
            result["memory_safe"] = False
            result["out_of_memory"] = True
            print(f"{variant_key(variant)} exhausted GPU memory; saved rejection evidence.", flush=True)
        save()


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    import launch_run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    worker(launch_run.build_plan(launch_run.load_spec(args.config), args.config), args.variant_index, args.output)
