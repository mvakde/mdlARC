"""Bounded training capacity probes, executed through the normal spec launcher.

Each candidate owns a fresh CUDA process. Measurements include the existing
compiled model, optimizer states, metrics, backward, clipping, and prefetch.
The fixed pre-collated batches measure device training, not augmentation speed.
"""

import argparse
import json
import math
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
    "warmup_epochs": 1,
    "measure_epochs": 2,
    "trial_timeout_seconds": 900,
    "modes": ["sampled", "grid_20", "grid_30"],
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
    for key in ("memory_limit_gb", "allocator_margin_gb"):
        if isinstance(config[key], bool) or not isinstance(config[key], (int, float)) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"training_benchmark.{key} must be positive and finite.")
    if config["allocator_margin_gb"] >= config["memory_limit_gb"]:
        raise ValueError("Allocator margin must be smaller than the memory limit.")
    for key in ("start_batch_size", "max_batch_size", "batch_multiple", "warmup_steps", "measure_steps", "warmup_epochs", "measure_epochs", "trial_timeout_seconds"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"training_benchmark.{key} must be a positive integer.")
    if config["start_batch_size"] > config["max_batch_size"]:
        raise ValueError("start_batch_size exceeds max_batch_size.")
    if not config["search"] and config["start_batch_size"] != config["max_batch_size"]:
        raise ValueError("A fixed probe requires equal start_batch_size and max_batch_size.")
    if any(config[key] % config["batch_multiple"] for key in ("start_batch_size", "max_batch_size")):
        raise ValueError("Batch bounds must be divisible by batch_multiple.")
    if not isinstance(config["modes"], list) or not config["modes"] or len(set(config["modes"])) != len(config["modes"]):
        raise ValueError("training_benchmark.modes must be a nonempty unique list.")
    if any(mode not in ("epoch", "sampled", "longest", "variation_longest", "grid_15", "grid_20", "grid_25", "grid_30") for mode in config["modes"]):
        raise ValueError("Unsupported training_benchmark batch mode.")
    if "epoch" in config["modes"] and config["search"]:
        raise ValueError("Full-epoch timing requires a fixed batch size (search=false).")
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
    """Scan the corpus once; bound contexts by their six longest eligible demos."""
    import numpy as np
    from common import normalize_attention_pattern
    from demo_schedule import MAX_CONTEXT_DEMOS

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
            longest = sorted(range(len(pairs)), key=pair_lengths.__getitem__, reverse=True)[:MAX_CONTEXT_DEMOS + 1]
            for pair_index, pair in enumerate(pairs):
                grids = (pair["input"], pair["output"])
                length = pair_lengths[pair_index]
                if attention_pattern == "pair_group":
                    demos = [i for i in longest if i != pair_index][:MAX_CONTEXT_DEMOS]
                    length += sum(pair_lengths[i] - 1 for i in demos)
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
        "sequence_length_scope": "capped_demo_context_upper_bound" if attention_pattern == "pair_group" else "target_pair",
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
            confirmation = probe(mode, candidate, repeat=True) if record["status"] == "fits" else None
            if confirmation is not None and confirmation["status"] != "fits":
                raise RuntimeError(f"Fixed probe failed independent confirmation: {mode}/{candidate}")
            results["summary"][mode] = {
                "tested_batch": candidate, "status": record["status"],
                "largest_confirmed_batch": candidate if confirmation else None,
                "first_rejected_batch": None if confirmation else candidate,
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


class RepeatedBatch:
    def __init__(self, batch, count):
        self.batch, self.count = batch, count

    def __iter__(self):
        for _ in range(self.count):
            yield self.batch

    def __len__(self):
        return self.count


def _training_sequence_lengths(dataset):
    """Profile the serialized training episodes, not just their target pairs."""
    lengths = getattr(dataset, "benchmark_sequence_lengths", None)
    if lengths is None:
        lengths = [int(dataset.build_training_sequence(example, 0, None)[0].numel())
                   for example in dataset.examples]
        dataset.benchmark_sequence_lengths = lengths
    return lengths


def _synthetic_demo_count(dataset):
    from demo_schedule import MAX_CONTEXT_DEMOS
    if getattr(dataset, "attention_pattern", "dense") != "pair_group":
        return 2
    counts = [max(0, sum(pair.get("output") is not None for pair in pairs) - 1)
              for pairs in dataset.pairs_by_task.values()]
    counts.extend(max(0, int(item["pair_count"]) - 1)
                  for variants in getattr(dataset, "variation_parents", {}).values() for item in variants)
    return min(MAX_CONTEXT_DEMOS, max(counts, default=0))


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
        full_epoch = request["mode"] == "epoch"
        if not full_epoch:
            args.num_workers = 0
        model, dataset, dataloader, device, _ = build.build_model_and_data(args, build_dataloader=full_epoch)
        if full_epoch:
            # Match _train_single_model: science training drops the incomplete
            # final physical batch after constructing the dataloader.
            if hasattr(dataloader.batch_sampler, "drop_last"):
                dataloader.batch_sampler.drop_last = True
            augmentor = getattr(dataloader, "augmentor", None)
            if hasattr(augmentor, "prepare_training_schedule"):
                augmentor.prepare_training_schedule(dataloader.batch_sampler)
        record["model_config"] = vars(model.config)
        lengths = np.asarray([] if full_epoch else _training_sequence_lengths(dataset))
        record["dataset_profile"] = {"regular_training_pairs": len(dataset.examples) if full_epoch else len(lengths), "task_embeddings": dataset.num_examples,
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
        if not full_epoch:
            batch = make_batch(dataset, request["mode"], args.batch_size, args.seed)
            record["tokens_per_batch"] = int(batch["input_ids"].numel())
            record["max_sequence_length"] = int(batch["max_seqlen"])
        optimizer = train._build_optimizer(args, model, device,
            args.attention_weight_decay, args.token_embedding_weight_decay, args.task_embedding_weight_decay)
        train._enable_compile_scalar_capture()
        compiled_model = torch.compile(model)

        def steps(count, label):
            train.train_one_epoch(compiled_model, RepeatedBatch(batch, count), optimizer,
                device, args.grad_clip, gradient_accumulation_steps=1,
                train_log_mode="last_step", log_location="terminal", log_prefix=label)

        torch.cuda.reset_peak_memory_stats(device)
        if full_epoch:
            scheduler = train._FractionalLambdaLR(optimizer, train._base_lr_schedule(args, optimizer))
            if hasattr(optimizer, "configure_muon_schedule"):
                sampler = dataloader.batch_sampler
                if hasattr(sampler, 'set_epoch'):
                    total_steps = 0
                    for planned_epoch in range(args.epochs):
                        sampler.set_epoch(planned_epoch)
                        total_steps += len(dataloader)
                    sampler.set_epoch(0)
                else:
                    total_steps = args.epochs * len(dataloader)
                optimizer.configure_muon_schedule(total_steps=total_steps, start_step=0)
            record.update(timing_scope="full_training_epoch_including_cpu_collation_and_transfer",
                          steps_per_epoch=len(dataloader), epoch_timings=[])
            global_step = 0
            for epoch in range(config["warmup_epochs"] + config["measure_epochs"]):
                augmentor = getattr(dataloader, "augmentor", None)
                if augmentor is not None:
                    augmentor.set_epoch(epoch)
                if hasattr(dataloader.batch_sampler, "set_epoch"):
                    dataloader.batch_sampler.set_epoch(epoch)
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                global_step = train.train_one_epoch(compiled_model, dataloader, optimizer,
                    device, args.grad_clip, gradient_accumulation_steps=1, start_step=global_step,
                    epoch=epoch, steps_per_epoch=len(dataloader), total_epochs=args.epochs, scheduler=scheduler,
                    mtp_decay=args.mtp_decay, train_log_mode=args.train_log_mode,
                    log_location="terminal", log_prefix="epoch_benchmark")
                torch.cuda.synchronize(device)
                timing = dict(epoch=epoch + 1, seconds=time.perf_counter() - start,
                              warmup=epoch < config["warmup_epochs"])
                record["epoch_timings"].append(timing)
                print("EPOCH_TIMING " + json.dumps(timing), flush=True)
                write_json(Path(request["output"]), record)
            measured = [t["seconds"] for t in record["epoch_timings"] if not t["warmup"]]
            elapsed = sum(measured)
            record.update(warmup_seconds=sum(t["seconds"] for t in record["epoch_timings"] if t["warmup"]),
                seconds_per_epoch=float(np.median(measured)), measured_seconds=elapsed,
                measure_steps=len(dataloader) * len(measured),
                seconds_per_step=elapsed / (len(dataloader) * len(measured)),
                examples_per_second=args.batch_size * len(dataloader) * len(measured) / elapsed)
        else:
            start = time.perf_counter()
            steps(config["warmup_steps"], "benchmark_warmup")
            torch.cuda.synchronize(device)
            record["warmup_seconds"] = time.perf_counter() - start
            start = time.perf_counter()
            steps(config["measure_steps"], "benchmark_measured")
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            record.update(measure_steps=config["measure_steps"], measured_seconds=elapsed,
                          seconds_per_step=elapsed / config["measure_steps"],
                          examples_per_second=args.batch_size * config["measure_steps"] / elapsed,
                          tokens_per_second=record["tokens_per_batch"] * config["measure_steps"] / elapsed)
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
