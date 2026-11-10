"""Bounded training capacity probes, executed through the normal spec launcher.

Each candidate owns a fresh CUDA process. Measurements include the existing
compiled model, optimizer states, metrics, backward, clipping, and prefetch.
The fixed pre-collated batches measure device training, not augmentation speed.
"""

import argparse
import gzip
import json
import math
import re
import subprocess
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    "search": True,
    "memory_limit_gb": 70.0,  # Decimal GB; GiB are reported separately.
    "allocator_margin_gb": 1.0,
    "start_batch_size": 32,
    "max_batch_size": 4096,
    "batch_multiple": 8,
    "warmup_steps": 3,
    "measure_steps": 8,
    "trial_timeout_seconds": 900,
    "modes": ["sampled", "grid_20", "grid_30"],
    # Speed probes: cycle several pre-collated batches so compiled shapes vary
    # as in training, then record a bounded torch.profiler trace and summary.
    "distinct_batches": 1,
    "profile_steps": 0,
    # A fixed probe reruns its candidate in a second process by default; a speed
    # comparison on a box whose time is precious can skip that confirmation.
    "confirm": True,
}


def validate_config(value):
    if not isinstance(value, dict):
        raise ValueError("training_benchmark must be an object.")
    unknown = set(value) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown training_benchmark keys: {sorted(unknown)}")
    config = {**DEFAULTS, **value}
    if not isinstance(config["search"], bool):
        raise ValueError("training_benchmark.search must be a boolean.")
    if not isinstance(config["confirm"], bool):
        raise ValueError("training_benchmark.confirm must be a boolean.")
    if config["search"] and not config["confirm"]:
        raise ValueError("A capacity search always confirms its boundary; confirm=false is for fixed probes.")
    for key in ("memory_limit_gb", "allocator_margin_gb"):
        if isinstance(config[key], bool) or not isinstance(config[key], (int, float)) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"training_benchmark.{key} must be positive and finite.")
    if config["allocator_margin_gb"] >= config["memory_limit_gb"]:
        raise ValueError("Allocator margin must be smaller than the memory limit.")
    for key in ("start_batch_size", "max_batch_size", "batch_multiple", "warmup_steps", "measure_steps", "trial_timeout_seconds", "distinct_batches"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"training_benchmark.{key} must be a positive integer.")
    if isinstance(config["profile_steps"], bool) or not isinstance(config["profile_steps"], int) or config["profile_steps"] < 0:
        raise ValueError("training_benchmark.profile_steps must be a non-negative integer.")
    if config["start_batch_size"] > config["max_batch_size"]:
        raise ValueError("start_batch_size exceeds max_batch_size.")
    if not config["search"] and config["start_batch_size"] != config["max_batch_size"]:
        raise ValueError("A fixed probe requires equal start_batch_size and max_batch_size.")
    if any(config[key] % config["batch_multiple"] for key in ("start_batch_size", "max_batch_size")):
        raise ValueError("Batch bounds must be divisible by batch_multiple.")
    if not isinstance(config["modes"], list) or not config["modes"] or len(set(config["modes"])) != len(config["modes"]):
        raise ValueError("training_benchmark.modes must be a nonempty unique list.")
    if any(mode not in ("training", "sampled", "longest", "variation_longest", "grid_15", "grid_20", "grid_25", "grid_30") for mode in config["modes"]):
        raise ValueError("Unsupported training_benchmark batch mode.")
    if config["warmup_steps"] < 2 * config["distinct_batches"]:
        raise ValueError("warmup_steps must cover every distinct batch twice so compilation settles before timing.")
    return config


def next_candidate(low, high, multiple):
    """Return a new aligned midpoint, or None when the bracket is adjacent."""
    if high - low <= multiple:
        return None
    return ((low // multiple + high // multiple) // 2) * multiple


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def variation_profile(index_path, attention_pattern=None):
    """Scan the selected, verified corpus once; include all-demo context lengths."""
    import numpy as np
    from common import normalize_attention_pattern

    attention_pattern = normalize_attention_pattern(attention_pattern)
    index = json.loads(Path(index_path).read_text())
    lengths, longest_refs = [], []
    maximum, variant_count = 0, 0
    for parent_id, variants in index["parents"].items():
        for variant_index, item in enumerate(variants):
            task = json.loads((ROOT / item["path"]).read_text())
            pairs = task["train"] + task["test"]
            if len(pairs) != item["pair_count"]:
                raise ValueError("Variation pair count disagrees with the verified index.")
            variant_count += 1
            pair_lengths = [1 + sum(sum(len(row) for row in grid) + len(grid)
                                    for grid in (pair["input"], pair["output"])) for pair in pairs]
            context_length = 1 + sum(length - 1 for length in pair_lengths)
            for pair_index, pair in enumerate(pairs):
                grids = (pair["input"], pair["output"])
                length = context_length if attention_pattern == "pair_group" else pair_lengths[pair_index]
                lengths.append(length)
                if length > maximum:
                    maximum, longest_refs = length, []
                if length == maximum and len(longest_refs) < 32:
                    longest_refs.append({"parent_id": parent_id, "variant_index": variant_index,
                        "pair_index": pair_index,
                        "input_shape": [len(grids[0]), len(grids[0][0])],
                        "output_shape": [len(grids[1]), len(grids[1][0])]})
    if not lengths:
        raise ValueError("No selected task-variation pairs.")
    from common import MAX_SEQ_LEN
    if attention_pattern == "dense" and maximum > MAX_SEQ_LEN:
        raise ValueError(f"Variation length {maximum} exceeds the current model limit {MAX_SEQ_LEN}.")
    return {"snapshot": index["snapshot"], "archive_sha256": index["archive_sha256"],
        "attention_pattern": attention_pattern,
        "sequence_length_scope": "all_demo_context" if attention_pattern == "pair_group" else "target_pair",
        "parents": len(index["parents"]), "variants": variant_count, "solved_pairs": len(lengths),
        "sequence_length_percentiles": dict(zip(["min", "p50", "p90", "p95", "p99", "max"],
            np.percentile(lengths, [0, 50, 90, 95, 99, 100]).tolist())), "longest_refs": longest_refs}


def run_benchmark(plan):
    config = plan["training_benchmark"]
    output = plan["run_dir"] / "training_benchmark"
    output.mkdir(parents=True, exist_ok=True)
    results = {"config": config, "spec_path": str(plan["spec_path"]), "trials": [], "summary": {}}
    profile_path = None
    if plan["pretrain"].get("variation_index_path"):
        sys.path.insert(0, str(ROOT / "src"))
        profile_path = output / "variation_profile.json"
        print("Scanning all selected task-variation pairs for the real worst case...", flush=True)
        write_json(profile_path, variation_profile(plan["pretrain"]["variation_index_path"],
                                                   plan["pretrain"].get("attention_pattern")))

    def probe(mode, batch_size, repeat=False):
        label = f"{mode}-b{batch_size}" + ("-confirm" if repeat else "")
        request = output / f"{label}.request.json"
        response = output / f"{label}.json"
        write_json(request, {"spec_path": str(plan["spec_path"]), "mode": mode,
                             "batch_size": batch_size, "output": str(response),
                             "variation_profile": str(profile_path) if profile_path else None})
        print(f"Training capacity probe: {label}", flush=True)
        subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), str(request)],
                       check=True, timeout=config["trial_timeout_seconds"])
        record = json.loads(response.read_text())
        record["confirmation"] = repeat
        results["trials"].append(record)
        write_json(output / "results.json", results)
        print("CAPACITY_RESULT " + json.dumps(record), flush=True)
        return record

    for mode in config["modes"]:
        if not config["search"]:
            candidate = config["start_batch_size"]
            record = probe(mode, candidate)
            fits = record["status"] == "fits"
            confirmation = probe(mode, candidate, repeat=True) if fits and config["confirm"] else None
            if confirmation is not None and confirmation["status"] != "fits":
                raise RuntimeError(f"Fixed probe failed independent confirmation: {mode}/{candidate}")
            results["summary"][mode] = {
                "tested_batch": candidate, "status": record["status"],
                "largest_confirmed_batch": candidate if confirmation else None,
                "first_rejected_batch": None if fits else candidate,
                "confirmation": confirmation,
            }
            write_json(output / "results.json", results)
            continue
        low, high = 0, None
        candidate = config["start_batch_size"]
        while True:
            record = probe(mode, candidate)
            if record["status"] == "fits":
                low = candidate
                if candidate == config["max_batch_size"]:
                    break
                candidate = min(candidate * 2, config["max_batch_size"])
            elif record["status"] in ("memory_limit", "over_limit"):
                high = candidate
                break
            else:
                raise RuntimeError(f"Unexpected probe status: {record['status']}")
        if high is not None:
            while (candidate := next_candidate(low, high, config["batch_multiple"])) is not None:
                record = probe(mode, candidate)
                if record["status"] == "fits":
                    low = candidate
                else:
                    high = candidate
        if low == 0:
            raise RuntimeError(f"No batch fits in mode {mode}.")
        confirmation = probe(mode, low, repeat=True)
        while confirmation["status"] != "fits":
            # Allocator fragmentation can reject a boundary on repetition.
            # Back off rather than promoting an unstable batch or losing other modes.
            high = low
            low -= config["batch_multiple"]
            if low <= 0:
                raise RuntimeError(f"No batch survives independent confirmation: {mode}")
            record = probe(mode, low)
            if record["status"] != "fits":
                confirmation = record
                continue
            confirmation = probe(mode, low, repeat=True)
        fitting = [r for r in results["trials"] if r["mode"] == mode and r["status"] == "fits"]
        fastest = max(fitting, key=lambda r: r["examples_per_second"])
        results["summary"][mode] = {
            "largest_confirmed_batch": low, "first_rejected_batch": high,
            "search_resolution": config["batch_multiple"],
            "search_upper_bound_reached": high is None,
            "fastest_measured_batch": fastest["batch_size"],
            "fastest_examples_per_second": fastest["examples_per_second"],
            "confirmation": confirmation,
        }
        write_json(output / "results.json", results)
    print("TRAINING_BENCHMARK_SUMMARY " + json.dumps(results["summary"]), flush=True)


class CycledBatches:
    """Pre-collated CPU batches repeated in order; one batch is the fixed probe."""

    def __init__(self, batches, count):
        self.batches, self.count = list(batches), count

    def __iter__(self):
        for index in range(self.count):
            yield self.batches[index % len(self.batches)]

    def __len__(self):
        return self.count


# Device kernel names, matched in order; inductor's fused kernels come before the
# aten index/elementwise names they may embed.
KERNEL_CATEGORIES = (
    # FA2 names are flash_fwd_kernel/flash_bwd_*; FA3 (hopper) wraps FlashAttnFwdSm90/
    # FlashAttnBwdSm90 in cutlass::device_kernel, so it must match before "cutlass".
    ("attention", re.compile(r"flash_(fwd|bwd)|FlashAttn|fmha|attention", re.I)),
    ("gemm", re.compile(r"gemm|cutlass|xmma|cublas|nvjet|gemv|sm\d\d_", re.I)),
    ("memory", re.compile(r"Memcpy|Memset", re.I)),
    ("fused", re.compile(r"^triton_", re.I)),
    ("index", re.compile(r"index|gather|scatter", re.I)),
    ("scan", re.compile(r"cub::|DeviceScan|DeviceSelect|DeviceReduce|radix|cumsum|nonzero|unique|sort", re.I)),
    ("reduce", re.compile(r"reduce", re.I)),
    ("elementwise", re.compile(r"elementwise|vectorized|unrolled|fill|copy|Cat", re.I)),
    ("optimizer", re.compile(r"multi_tensor|foreach|adam", re.I)),
)
HOST_SYNC_NAMES = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaMemcpyAsync",
                   "cudaLaunchKernel", "cuLaunchKernel", "cuLaunchKernelEx",
                   "aten::item", "aten::_local_scalar_dense", "aten::nonzero", "aten::unique_consecutive")
LAUNCH_API_NAMES = ("cudaLaunchKernel", "cuLaunchKernel", "cuLaunchKernelEx")
MARKER_NAMES = ("packed_layout", "forward", "backward", "optimizer_step")


def categorize_kernel(name):
    for category, pattern in KERNEL_CATEGORIES:
        if pattern.search(name):
            return category
    return "other"


def _union_seconds(intervals):
    """Total covered time of possibly overlapping (start, end) microsecond intervals."""
    total, current = 0.0, None
    for start, end in sorted(intervals):
        if current is None or start > current[1]:
            if current is not None:
                total += current[1] - current[0]
            current = [start, end]
        else:
            current[1] = max(current[1], end)
    if current is not None:
        total += current[1] - current[0]
    return total / 1e6


def summarize_records(records, wall_seconds, steps):
    """Summarize (name, is_device, start_us, end_us) events from a profiled training window.

    Device events are compute kernels plus copies/memsets. ``gpu_busy`` is the
    union of kernel time only, so copy-engine time never masquerades as SM work;
    ``gpu_occupied`` adds copies and memsets, so ``1 - gpu_occupied_fraction`` is
    the time the device had nothing queued at all. Copies that run alone (the
    clone inside an out-of-place index_add, for example) show up as the gap
    between the two fractions, not as idle time.
    """
    categories, kernels, intervals, device_intervals, kernel_count = {}, {}, [], [], 0
    host = {name: {"count": 0, "seconds": 0.0} for name in (*HOST_SYNC_NAMES, *MARKER_NAMES)}
    for name, is_device, start, end in records:
        duration = (end - start) / 1e6
        if is_device:
            category = categorize_kernel(name)
            categories[category] = categories.get(category, 0.0) + duration
            entry = kernels.setdefault(name, {"seconds": 0.0, "count": 0})
            entry["seconds"] += duration
            entry["count"] += 1
            device_intervals.append((start, end))
            if category != "memory":
                intervals.append((start, end))
                kernel_count += 1
        elif name in host:
            host[name]["count"] += 1
            host[name]["seconds"] += duration
    busy = _union_seconds(intervals)
    occupied = _union_seconds(device_intervals)
    launch_calls = sum(host[name]["count"] for name in LAUNCH_API_NAMES)
    top = sorted(kernels.items(), key=lambda item: item[1]["seconds"], reverse=True)[:40]
    return {
        "steps": steps, "wall_seconds": wall_seconds, "seconds_per_step": wall_seconds / max(steps, 1),
        "gpu_busy_seconds": busy, "gpu_busy_fraction": busy / wall_seconds if wall_seconds else None,
        "gpu_occupied_seconds": occupied,
        "gpu_occupied_fraction": occupied / wall_seconds if wall_seconds else None,
        "gpu_idle_fraction": 1.0 - occupied / wall_seconds if wall_seconds else None,
        "kernel_seconds_by_category": dict(sorted(categories.items(), key=lambda item: item[1], reverse=True)),
        "kernels_per_step": kernel_count / max(steps, 1), "launch_api_calls_per_step": launch_calls / max(steps, 1),
        "host_events": {name: value for name, value in host.items() if value["count"]},
        "top_kernels": [{"name": name, "seconds": value["seconds"], "count": value["count"],
                         "mean_us": 1e6 * value["seconds"] / value["count"]} for name, value in top],
    }


def _is_annotation(name):
    """Kineto projects record_function ranges onto the GPU timeline; they are not kernels."""
    return name in MARKER_NAMES or "#" in name or name.startswith("ProfilerStep")


def profile_records(profiler):
    import torch
    device_type = getattr(torch.autograd, "DeviceType", None)
    records = []
    for event in profiler.events():
        is_device = device_type is not None and getattr(event, "device_type", None) == device_type.CUDA
        if is_device and _is_annotation(event.name):
            continue
        records.append((event.name, is_device, event.time_range.start, event.time_range.end))
    return records


def export_profile(profiler, output, label, wall_seconds, steps):
    """Write the gzipped Chrome trace and the category summary next to the trial record."""
    summary = summarize_records(profile_records(profiler), wall_seconds, steps)
    trace = output / f"{label}.trace.json"
    profiler.export_chrome_trace(str(trace))
    with trace.open("rb") as source, gzip.open(str(trace) + ".gz", "wb") as target:
        target.writelines(source)
    trace.unlink()
    summary["trace"] = trace.name + ".gz"
    write_json(output / f"{label}.profile.json", summary)
    print("PROFILE_SUMMARY " + json.dumps({key: summary[key] for key in (
        "seconds_per_step", "gpu_busy_fraction", "gpu_occupied_fraction", "gpu_idle_fraction",
        "kernels_per_step", "launch_api_calls_per_step",
        "kernel_seconds_by_category", "host_events")}), flush=True)
    return summary


def training_batches(dataloader, count):
    """The first ``count`` real epoch-0 batches, prepared exactly as the trainer prepares its loader."""
    batch_sampler = getattr(dataloader, "batch_sampler", None)
    if batch_sampler is not None and hasattr(batch_sampler, "drop_last"):
        batch_sampler.drop_last = True
    augmentor = getattr(dataloader, "augmentor", None)
    if hasattr(augmentor, "prepare_training_schedule"):
        augmentor.prepare_training_schedule(batch_sampler)
    if augmentor is not None:
        augmentor.set_epoch(0)
    if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(0)
    iterator = iter(dataloader)
    return [next(iterator) for _ in range(count)]


def _training_sequence_lengths(dataset):
    """Profile the serialized training episodes, not just their target pairs."""
    lengths = getattr(dataset, "benchmark_sequence_lengths", None)
    if lengths is None:
        lengths = [int(dataset.build_training_sequence(example, 0, None)[0].numel())
                   for example in dataset.examples]
        dataset.benchmark_sequence_lengths = lengths
    return lengths


def _synthetic_demo_count(dataset):
    if getattr(dataset, "attention_pattern", "dense") != "pair_group":
        return 2
    counts = [max(0, sum(pair.get("output") is not None for pair in pairs) - 1)
              for pairs in dataset.pairs_by_task.values()]
    counts.extend(max(0, int(item["pair_count"]) - 1)
                  for variants in getattr(dataset, "variation_parents", {}).values() for item in variants)
    return max(counts, default=0)


def make_batch(dataset, mode, batch_size, seed):
    import numpy as np
    import torch
    from common import collate_examples, compute_positions_3d, encode_context

    if mode == "variation_longest":
        profile = dataset.benchmark_variation_profile
        if profile.get("attention_pattern", "dense") != dataset.attention_pattern:
            raise ValueError("Variation benchmark profile attention_pattern does not match the dataset.")
        pool = profile["longest_refs"]
        selected = [dataset[("variation", item["parent_id"], item["variant_index"], item["pair_index"], 0)]
            for item in (pool[i % len(pool)] for i in range(batch_size))]
        augmentor = getattr(dataset, "augmentor", None)
        batch = collate_examples(selected, dataset,
            augment_selector=augmentor.select_for_example if augmentor else None)
    elif mode in ("sampled", "longest"):
        examples = dataset.examples
        if not examples:
            raise ValueError("The sampled probe requires original training examples.")
        if mode == "longest":
            lengths = _training_sequence_lengths(dataset)
            maximum = max(lengths)
            pool = [example for example, length in zip(examples, lengths) if length == maximum]
            selected = [pool[i % len(pool)] for i in range(batch_size)]
        else:
            rng = np.random.default_rng(seed)
            selected = [examples[i] for i in rng.integers(0, len(examples), size=batch_size)]
        augmentor = getattr(dataset, "augmentor", None)
        batch = collate_examples(selected, dataset,
            augment_selector=augmentor.select_for_example if augmentor else None)
    else:
        side = int(mode.split("_")[1])
        rng = np.random.default_rng(seed)
        input_grid = rng.integers(0, 10, (side, side)).tolist()
        output_grid = rng.integers(0, 10, (side, side)).tolist()
        pair = {"input": input_grid, "output": output_grid}
        tokens, separator = encode_context([pair] * _synthetic_demo_count(dataset), pair)
        length = tokens.numel()
        positions = compute_positions_3d(tokens.unsqueeze(0)).squeeze(0)
        ids = torch.arange(batch_size, dtype=torch.int32) % dataset.num_examples
        batch = {"input_ids": tokens.repeat(batch_size),
                 "cu_seqlens": torch.arange(batch_size + 1, dtype=torch.int32) * length,
                 "max_seqlen": length, "has_padding": False,
                 "example_ids": torch.stack((ids, torch.full_like(ids, -1)), dim=1),
                 "dihedral_ids": torch.arange(batch_size, dtype=torch.int32) % 8,
                 "sep_indices": torch.full((batch_size,), separator, dtype=torch.int32),
                 "positions_3d": positions.repeat(batch_size, 1)}
    return batch


def run_trial(request):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    import torch
    import build
    import train
    import launch_run

    spec_path = Path(request["spec_path"])
    plan = launch_run.build_plan(launch_run.load_spec(spec_path), spec_path)
    config = plan["training_benchmark"]
    record = {"mode": request["mode"], "batch_size": request["batch_size"], "status": "pending"}
    device = torch.device("cuda:0")
    torch.cuda.init()
    free, total = torch.cuda.mem_get_info(device)
    initial_external = total - free - torch.cuda.memory_reserved(device)
    limit = int(config["memory_limit_gb"] * 1e9)
    allocator_budget = limit - initial_external - int(config["allocator_margin_gb"] * 1e9)
    if allocator_budget <= 0 or limit >= total:
        raise ValueError("Benchmark ceiling must leave space below physical GPU capacity.")
    torch.cuda.set_per_process_memory_fraction(allocator_budget / total, device)
    record.update(gpu=torch.cuda.get_device_name(device), total_gpu_bytes=total,
                  memory_limit_bytes=limit, allocator_budget_bytes=allocator_budget,
                  precision="bfloat16_autocast_fp32_parameters", model_compiled=True)
    peaks = {"device_used": total - free, "external": initial_external}
    stop = threading.Event()

    def sample_memory():
        while not stop.wait(0.02):
            free_now, total_now = torch.cuda.mem_get_info(device)
            used = total_now - free_now
            peaks["device_used"] = max(peaks["device_used"], used)
            peaks["external"] = max(peaks["external"], used - torch.cuda.memory_reserved(device))

    monitor = threading.Thread(target=sample_memory, daemon=True)
    monitor.start()
    try:
        args = argparse.Namespace(**plan["pretrain"])
        args.batch_size = request["batch_size"]
        args.num_workers = 0
        training_mode = request["mode"] == "training"
        model, dataset, dataloader, device, _ = build.build_model_and_data(args, build_dataloader=training_mode)
        record["model_config"] = vars(model.config)
        lengths = np.asarray(_training_sequence_lengths(dataset))
        record["dataset_profile"] = {"regular_training_pairs": len(lengths), "task_embeddings": dataset.num_examples,
            "sequence_length_scope": "training_context", "sequence_length_geometry": "identity",
            "sequence_length_percentiles": dict(zip(["min", "p50", "p90", "p95", "p99", "max"],
                np.percentile(lengths, [0, 50, 90, 95, 99, 100]).tolist())) if len(lengths) else None}
        if request.get("variation_profile"):
            profile = json.loads(Path(request["variation_profile"]).read_text())
            dataset.benchmark_variation_profile = profile
            record["dataset_profile"]["task_variations"] = profile
            if args.enable_aug:
                from variations import VariationAugmentor
                dataset.augmentor = VariationAugmentor(dataset, seed=args.seed, max_augments=args.max_augments)
        if training_mode:
            batches = training_batches(dataloader, config["distinct_batches"])
        else:
            batches = [make_batch(dataset, request["mode"], args.batch_size, args.seed + index)
                       for index in range(config["distinct_batches"])]
        batch = batches[0]
        record["tokens_per_batch"] = int(batch["input_ids"].numel())
        record["max_sequence_length"] = int(batch["max_seqlen"])
        record["distinct_batches"] = [{"tokens": int(item["input_ids"].numel()), "max_seqlen": int(item["max_seqlen"])}
                                      for item in batches]
        optimizer = train._build_optimizer(args, model, device,
            args.attention_weight_decay, args.token_embedding_weight_decay, args.task_embedding_weight_decay)
        compiled_model, record["compile_options"] = train.compile_training_model(model, args)

        def steps(count, label):
            train.train_one_epoch(compiled_model, CycledBatches(batches, count), optimizer,
                device, args.grad_clip, gradient_accumulation_steps=1,
                train_log_mode="last_step", log_location="terminal", log_prefix=label)

        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        steps(config["warmup_steps"], "benchmark_warmup")
        torch.cuda.synchronize(device)
        record["warmup_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        steps(config["measure_steps"], "benchmark_measured")
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        measured_tokens = sum(record["distinct_batches"][index % len(batches)]["tokens"]
                              for index in range(config["measure_steps"]))
        record.update(measure_steps=config["measure_steps"], measured_seconds=elapsed,
                      seconds_per_step=elapsed / config["measure_steps"],
                      examples_per_second=args.batch_size * config["measure_steps"] / elapsed,
                      tokens_per_second=measured_tokens / elapsed)
        if config["profile_steps"]:
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
                start = time.perf_counter()
                steps(config["profile_steps"], "benchmark_profiled")
                torch.cuda.synchronize(device)
                profiled = time.perf_counter() - start
            record["profile"] = export_profile(profiler, Path(request["output"]).parent,
                                               Path(request["output"]).stem, profiled, config["profile_steps"])
        if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
            raise FloatingPointError("Nonfinite model parameters after the probe.")
        record["status"] = "fits"
    except torch.OutOfMemoryError:
        # Capacity rejection is a result; the next candidate gets a fresh process.
        record["status"] = "memory_limit"
    finally:
        stop.set()
        monitor.join(timeout=2)
        record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        record["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        record["sampled_peak_device_bytes"] = peaks["device_used"]
        record["peak_external_bytes"] = peaks["external"]
        record["conservative_peak_device_bytes"] = max(peaks["device_used"],
            record["peak_reserved_bytes"] + peaks["external"])
        if record["status"] == "fits" and record["conservative_peak_device_bytes"] >= limit:
            record["status"] = "over_limit"
        write_json(Path(request["output"]), record)


if __name__ == "__main__":
    run_trial(json.loads(Path(sys.argv[1]).read_text()))
