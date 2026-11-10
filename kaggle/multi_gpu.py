"""Offline Kaggle workers: DDP adaptation, independent task-sharded decoding."""

import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

RUNTIME = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RUNTIME / "src"), str(RUNTIME / "kaggle")]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def task_shards(challenges, count):
    if type(count) is not int or count < 1:
        raise ValueError("Worker count must be positive.")
    tasks = sorted(challenges)
    return [tasks[rank::count] for rank in range(count)]


def worker_environment(*, gpu=None):
    env = dict(os.environ)
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    for key in list(env):
        if key.startswith("TORCHELASTIC_"):
            env.pop(key, None)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TORCHINDUCTOR_COMPILE_THREADS="1",
               MDLARC_REMOTE="1", NCCL_DEBUG="WARN", PYTHONUNBUFFERED="1")
    return env


def training_command(request_path, count):
    return [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone",
            "--nnodes=1", f"--nproc-per-node={count}", "--max-restarts=0",
            str(Path(__file__).resolve()), "--phase", "train", "--request", str(request_path)]


def inference_command(request_path, rank, count):
    return [sys.executable, "-u", str(Path(__file__).resolve()), "--phase", "decode",
            "--request", str(request_path), "--rank", str(rank), "--world-size", str(count)]


def run_processes(jobs, directory, *, progress_interval=30):
    """Relay logs on the notebook thread and never restart a failed execution."""
    if progress_interval <= 0:
        raise ValueError("Progress interval must be positive.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    processes, readers = [], []
    output = queue.Queue()
    started = time.monotonic()
    last_output = {}
    last_report = {}

    def read_output(process, label, path):
        with path.open("w") as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                output.put((label, line))

    def relay_output():
        # Jupyter associates this main-thread output with the running cell.
        # Reader threads only collect pipes and persist the raw worker logs.
        while True:
            try:
                label, line = output.get_nowait()
            except queue.Empty:
                break
            last_output[label] = time.monotonic()
            print(f"[{label}] {line}", end="", flush=True)

    try:
        for label, command, env in jobs:
            process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            processes.append((label, process))
            last_output[label] = time.monotonic()
            last_report[label] = last_output[label]
            print(f"[{label}] Started PID {process.pid}; log: {directory / (label + '.log')}", flush=True)
            reader = threading.Thread(target=read_output,
                                      args=(process, label, directory / f"{label}.log"), daemon=True)
            reader.start()
            readers.append(reader)
        while any(process.poll() is None for _, process in processes):
            relay_output()
            for label, process in processes:
                if process.poll() not in (None, 0):
                    raise RuntimeError(f"{label} failed; inspect {directory / (label + '.log')}. No retry started.")
            now = time.monotonic()
            for label, process in processes:
                quiet_seconds = now - last_output[label]
                if (process.poll() is None and quiet_seconds >= progress_interval
                        and now - last_report[label] >= progress_interval):
                    print(f"[{label}] Elapsed {now - started:.0f}s; no new worker output for "
                          f"{quiet_seconds:.0f}s. PID {process.pid} is running; "
                          "this does not confirm optimizer progress.", flush=True)
                    last_report[label] = now
            time.sleep(min(0.1, progress_interval / 2))
        for label, process in processes:
            if process.returncode:
                raise RuntimeError(f"{label} failed; inspect {directory / (label + '.log')}. No retry started.")
    finally:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for reader in readers:
            reader.join(timeout=5)
        relay_output()


def launch_training(request_path):
    request = json.loads(Path(request_path).read_text())
    count = request["num_gpus"]
    env = worker_environment()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(count)))
    env["MDLARC_REPORT_FIRST_UPDATE"] = "1"
    run_processes([("train-ddp", training_command(request_path, count), env)], request["run_dir"],
                  progress_interval=request.get("progress_interval_seconds", 30))
    if not Path(request["save_path"]).is_file():
        raise RuntimeError("DDP adaptation did not save its final checkpoint.")


def launch_inference(request_path):
    request = json.loads(Path(request_path).read_text())
    count = request["num_gpus"]
    jobs = [(f"decode-gpu{rank}", inference_command(request_path, rank, count),
             worker_environment(gpu=rank)) for rank in range(count)]
    run_processes(jobs, request["run_dir"],
                  progress_interval=request.get("progress_interval_seconds", 30))


def make_inference_checkpoint(source, destination, expected_epoch):
    """Drop optimizer buffers once, before four independent inference loads."""
    import torch

    original = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    if original.get("epoch") != expected_epoch:
        raise ValueError("Final adaptation checkpoint has an unexpected epoch.")
    keys = ("model_state", "config", "task_ids", "embedding_task_ids", "sequence_format",
            "augmentation_sampling_mode", "epoch")
    checkpoint = {key: original[key] for key in keys}
    destination = Path(destination)
    temporary = destination.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(destination)
    return destination


def merge_shards(request, challenges, destination):
    import submission_utils as ku

    merged, repairs, statistics = {}, [], []
    for rank, ids in enumerate(task_shards(challenges, request["num_gpus"])):
        base = Path(request["run_dir"])
        part = json.loads((base / f"shard-{rank}.json").read_text())
        ku.validate_submission(part, {key: challenges[key] for key in ids})
        if set(part) & set(merged):
            raise ValueError("Inference shards overlap.")
        merged.update(part)
        info = json.loads((base / f"decode-{rank}.json").read_text())
        repairs.extend(info["malformed_attempt_replacements"])
        statistics.append(info)
    ku.validate_submission(merged, challenges)
    write_json(destination, merged)
    return repairs, statistics


def _runtime(request):
    # Must run before imports that discover CUDA attention extensions.
    if request["attention_backend"] == "native":
        sys.modules["flash_attn"] = None
        sys.modules["flash_attn_interface"] = None
    if not request["enable_compile"]:
        os.environ["TORCH_COMPILE_DISABLE"] = "1"
    import torch
    import torch._dynamo
    torch.set_num_threads(1)
    for option in ("cache_size_limit", "recompile_limit"):
        if hasattr(torch._dynamo.config, option):
            setattr(torch._dynamo.config, option, 64)
    return torch


def _config(request):
    import submission_utils as ku

    cfg = ku.finetune_config(request["defaults"], data_path=Path(request["data_path"]),
                             save_path=Path(request["save_path"]), **request["finetune"])
    cfg.train_log_mode = request.get("train_log_mode", "step")
    if cfg.train_log_mode not in {"step", "10_steps", "epoch", "never"}:
        raise ValueError("Unknown training log mode.")
    return cfg


def memory_statistics(torch, device):
    if device.type != "cuda":
        return {}
    return {"peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}


def configure_trainable_blocks(model, first=None, last=None):
    """Freeze every parameter outside the selected edge transformer blocks.

    Frozen middle blocks still participate in autograd so early-block gradients
    propagate through them. Do not wrap these blocks in no_grad or detach them.
    """
    count = len(model.blocks)
    originally_active = {id(p) for p in model.parameters() if p.requires_grad}
    if first is None and last is None:
        selected = list(range(count))
    else:
        if any(type(value) is not int or value < 0 or value > count for value in (first, last)):
            raise ValueError("First/last trainable block counts must be integers within model depth.")
        selected = sorted(set(range(first)) | set(range(count - last, count)))
        if not selected:
            raise ValueError("At least one transformer block must be trainable.")
        model.requires_grad_(False)
        for index in selected:
            for parameter in model.blocks[index].parameters():
                parameter.requires_grad_(id(parameter) in originally_active)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"trainable_blocks": selected, "trainable_parameters": trainable,
            "total_parameters": total, "trainable_fraction": trainable / total}


def train_worker(request, *, device_override=None):
    torch = _runtime(request)
    import build
    import distributed_training as dt
    import submission_utils as ku
    import train

    cfg = _config(request)
    cfg.device = device_override or f"cuda:{int(os.environ['LOCAL_RANK'])}"
    cfg.distributed_training = dt.world_size() > 1
    if device_override is None:
        torch.cuda.reset_peak_memory_stats()
    print("Loading pretrained checkpoint and verifying metadata...", flush=True)
    checkpoint = torch.load(request["checkpoint_path"], map_location="cpu", weights_only=False, mmap=True)
    ku.verify_checkpoint_metadata(checkpoint, request["checkpoint_identity"])
    adapted = ku.adapt_checkpoint(checkpoint)
    del checkpoint
    print("Building model and adaptation dataset...", flush=True)
    model, dataset, loader, device, _ = build.build_model_and_data(cfg, checkpoint=adapted)
    trainable = configure_trainable_blocks(model, **request.get("trainable_blocks", {}))
    print("Fine-tuning parameters:", json.dumps(trainable), flush=True)
    model._loaded_checkpoint = None
    del adapted
    # Preflight uses this same sampler/augmentation plan before workers spawn.
    print("Checking one backward pass before training...", flush=True)
    loss = ku.check_backward(model, loader, device)
    print(f"GPU rank {dt.rank()}: backward preflight loss {loss:.5f}", file=sys.stderr, flush=True)
    if len(loader) < cfg.gradient_accumulation_steps:
        raise ValueError("Too few microbatches for the selected accumulation; lower accumulation.")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"Training ready: {len(loader)} microbatches per rank/epoch; "
          f"accumulation={cfg.gradient_accumulation_steps}; log mode={cfg.train_log_mode}. "
          "First forward/backward and optimizer updates may compile before step logs appear.", flush=True)
    epoch_times = []
    original = train.train_one_epoch
    original_optimizer = train._build_optimizer
    optimizer_updates = 0
    optimizer_hooks = []

    def record_update(optimizer, args, kwargs):
        nonlocal optimizer_updates
        optimizer_updates += 1

    def checked_optimizer(*args, **kwargs):
        optimizer = original_optimizer(*args, **kwargs)
        active = {id(p) for p in model.parameters() if p.requires_grad}
        optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
        if optimized != active:
            raise RuntimeError("Optimizer parameters differ from the selected trainable parameters.")
        optimizer_hooks.append(optimizer.register_step_post_hook(record_update))
        return optimizer

    def timed_epoch(*args, **kwargs):
        start = time.perf_counter()
        result = original(*args, **kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = torch.tensor(time.perf_counter() - start, device=device)
        if dt.active():
            torch.distributed.all_reduce(elapsed, op=torch.distributed.ReduceOp.MAX)
        epoch_times.append(float(elapsed))
        print(f"Epoch {len(epoch_times)}/{cfg.epochs} compute completed in {float(elapsed):.1f}s.", flush=True)
        if len(epoch_times) >= 3:
            average = sum(epoch_times[1:]) / (len(epoch_times) - 1)
            hours = (epoch_times[0] + average * (cfg.epochs - 1)) / 3600
            print(f"Estimated DDP adaptation: {hours:.2f} hours; decoding is additional.", flush=True)
            if not request["smoke_test"] and hours > request["max_finetune_hours"]:
                raise RuntimeError("Adaptation ETA exceeds MAX_FINETUNE_HOURS. Lower epochs after inspecting timing.")
        return result

    train.train_one_epoch = timed_epoch
    train._build_optimizer = checked_optimizer
    try:
        train.train_model(cfg, model=model, dataset=dataset, dataloader=loader,
                          device=device, data_path=Path(request["data_path"]))
    finally:
        train.train_one_epoch = original
        train._build_optimizer = original_optimizer
        for handle in optimizer_hooks:
            handle.remove()
    if optimizer_updates < 1 or len(epoch_times) != cfg.epochs:
        raise RuntimeError("Adaptation did not complete real optimizer updates and every requested epoch.")
    write_json(Path(request["run_dir"]) / f"train-{dt.rank()}.json",
               {"rank": dt.rank(), "world_size": dt.world_size(), "epochs": cfg.epochs,
                "optimizer_updates": optimizer_updates,
                "effective_batch_size": cfg.batch_size * cfg.gradient_accumulation_steps * dt.world_size(),
                "epoch_seconds": epoch_times,
                **trainable,
                **memory_statistics(torch, device)})


def decode_worker(request, rank, count, *, device_override=None):
    decode_path = Path(request.get("decode_data_path", request["data_path"]))
    challenges = json.loads(decode_path.read_text())
    ids = task_shards(challenges, count)[rank]
    directory = Path(request["run_dir"])
    stats = {"rank": rank, "task_ids": ids, "malformed_attempt_replacements": []}
    if not ids:
        write_json(directory / f"shard-{rank}.json", {})
        write_json(directory / f"decode-{rank}.json", stats)
        return
    torch = _runtime(request)
    import evaluate
    import submission_utils as ku

    cfg = _config(request)
    cfg.device = device_override or "cuda:0"  # This worker sees exactly one physical GPU.
    cfg.checkpoint_mode = None
    cfg.extend_checkpoint_task_ids = False
    cfg.reset_training_state = False
    cfg.inherit_checkpoint_rearc_data_path = False
    cfg.num_workers = 0
    cfg.inference_speculative = request.get("inference_speculative", False)
    cfg.prefill_token_budget = request.get("prefill_token_budget", 16384)
    stats.update(batch_size=request["eval_batch_size"], max_augments=request["eval_augments"],
                 inference_speculative=cfg.inference_speculative,
                 prefill_token_budget=cfg.prefill_token_budget)
    device = torch.device(cfg.device)
    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
    _, _, raw = evaluate.run_evaluation(
        cfg, run_name=f"{directory.name}/decode-{rank}", max_augments=request["eval_augments"],
        data_path=decode_path, checkpoint_path=Path(request["inference_checkpoint"]),
        batch_size=request["eval_batch_size"], splits=["test"], task_ids=ids,
        timing_path=directory / f"timing-{rank}.txt",
    )
    stats["malformed_attempt_replacements"] = ku.finalize_submission(
        raw, directory / f"shard-{rank}.json", {key: challenges[key] for key in ids})
    stats.update(memory_statistics(torch, device))
    write_json(directory / f"decode-{rank}.json", stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "decode"), required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    os.chdir(request["working"])
    if args.phase == "train":
        print(f"GPU rank {os.environ.get('LOCAL_RANK', '0')}: importing runtime...", flush=True)
        _runtime(request)
        import distributed_training as dt
        print(f"GPU rank {os.environ.get('LOCAL_RANK', '0')}: initializing DDP...", flush=True)
        with dt.training_process(request["num_gpus"]):
            train_worker(request)
    else:
        decode_worker(request, args.rank, args.world_size)


if __name__ == "__main__":
    main()
