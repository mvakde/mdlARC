"""Compare native adaptation recipes using an immutable notebook model runtime.

Executed by launch_run.py inside the ordinary Colab backup/recovery workflow.
Only the final scoring process opens query solutions. Training and decoding
use separate processes and a merged, ordinary model checkpoint for inference.
"""

import argparse
import ast
import base64
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    pending.replace(path)


def validate_config(config):
    allowed = {"inputs", "method", "last", "rank", "alpha", "epochs", "batch_size",
               "accumulation", "train_augments", "eval_augments", "eval_batch_size",
               "normuon_lr", "adamw_lr", "lora_lr", "lora_weight_decay", "checkpoint_epochs",
               "enable_compile", "attention_backend", "seed"}
    allowed |= {"decode_only", "input_epoch", "inference_speculative", "prefill_token_budget",
                "decode_probe", "probe_cases", "probe_tokens", "probe_max_seconds", "wait_for_decode_settings"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("Invalid checkpoint_adaptation configuration")
    if config.get("method") not in {"last_blocks", "lora"}:
        raise ValueError("checkpoint_adaptation.method must be last_blocks or lora")
    inputs = config.get("inputs", {})
    if not {"runtime", "challenges", "solutions"} <= set(inputs) or set(inputs) - {"runtime", "challenges", "solutions", "training_stats"}:
        raise ValueError("Pin runtime, challenges and solutions inputs")
    for identity in inputs.values():
        if not isinstance(identity, dict) or set(identity) != {"path", "size", "sha256"}:
            raise ValueError("Each adaptation input requires path, size and sha256")
        path = Path(identity["path"])
        if path.is_absolute() or ".." in path.parts or "\\" in str(path):
            raise ValueError("Adaptation input paths must be repository relative")
        if type(identity["size"]) is not int or identity["size"] < 1:
            raise ValueError("Invalid adaptation input size")
        if len(identity["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in identity["sha256"]):
            raise ValueError("Invalid adaptation input SHA-256")
    for key in ("last", "rank", "epochs", "batch_size", "accumulation", "train_augments", "eval_augments", "eval_batch_size"):
        value = config.get(key, 1)
        if type(value) is not int or value < 1:
            raise ValueError(f"checkpoint_adaptation.{key} must be positive")
    for key in ("alpha", "lora_lr", "normuon_lr", "adamw_lr"):
        value = config.get(key, 1)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"checkpoint_adaptation.{key} must be positive")
    if type(config.get("enable_compile", True)) is not bool:
        raise ValueError("enable_compile must be boolean")
    for key in ("decode_only", "decode_probe", "inference_speculative", "wait_for_decode_settings"):
        if type(config.get(key, False)) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key in ("input_epoch", "prefill_token_budget", "probe_cases", "probe_tokens", "probe_max_seconds"):
        value = config.get(key, 1)
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be positive")
    if config.get("attention_backend", "flash") not in {"flash", "native"}:
        raise ValueError("attention_backend must be flash or native")
    if any(type(e) is not int or not 1 <= e <= config.get("epochs", 90)
           for e in config.get("checkpoint_epochs", [])):
        raise ValueError("Checkpoint epochs must be within the adaptation duration")
    return dict(config)


def verified_path(root, identity):
    path = Path(root) / identity["path"]
    path.resolve().relative_to(Path(root).resolve())
    if path.is_symlink() or path.stat().st_size != identity["size"] or hashlib.sha256(path.read_bytes()).hexdigest() != identity["sha256"]:
        raise ValueError(f"Adaptation input changed: {identity['path']}")
    return path


def unpack_runtime(path, destination):
    values = {}
    for cell in json.loads(Path(path).read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Name):
                continue
            name = node.targets[0].id
            if name in {"RUNTIME_B64", "RUNTIME_SHA256", "CHECKPOINT_IDENTITY"}:
                values[name] = ast.literal_eval(node.value)
            elif name == "SAVED_DEFAULTS":
                call = node.value
                if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                        and isinstance(call.func.value, ast.Name) and call.func.value.id == "json"
                        and call.func.attr == "loads" and len(call.args) == 1 and not call.keywords):
                    raise ValueError("Unsupported saved notebook defaults")
                values[name] = json.loads(ast.literal_eval(call.args[0]))
    raw = base64.b64decode(values["RUNTIME_B64"], validate=True)
    if hashlib.sha256(raw).hexdigest() != values["RUNTIME_SHA256"]:
        raise ValueError("Notebook runtime checksum mismatch")
    destination = Path(destination)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate notebook runtime members")
        for name in names:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or relative.parts[0] not in {"src", "kaggle"} or relative.suffix != ".py":
                raise ValueError("Unsafe notebook runtime member")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            data = archive.read(name)
            if target.exists() and target.read_bytes() != data:
                raise ValueError("Preserved runtime cannot change in place")
            target.write_bytes(data)
    return {name: values[name] for name in ("RUNTIME_SHA256", "CHECKPOINT_IDENTITY", "SAVED_DEFAULTS")}


def run_adaptation(plan):
    directory = Path(plan["run_dir"]).resolve()
    # The spec may be in specs/temp specs; use the launcher's actual repository.
    import launch_run
    root = launch_run.REPO_ROOT
    config = plan["checkpoint_adaptation"]
    paths = {name: verified_path(root, identity) for name, identity in config["inputs"].items()}
    runtime = directory / "runtime"
    pinned = unpack_runtime(paths["runtime"], runtime)
    checkpoint_identity = pinned["CHECKPOINT_IDENTITY"]
    source = plan["drive_checkpoint"]
    if not config.get("decode_only") and source and (source["size"] != checkpoint_identity["size"] or source["sha256"] != checkpoint_identity["sha256"]):
        raise ValueError("Input checkpoint differs from the submission notebook")
    if config.get("decode_only"):
        checkpoint_identity = {key: checkpoint_identity[key] for key in ("config", "augmentation_sampling_mode") if key in checkpoint_identity}
        checkpoint_identity.update(epoch=config.get("input_epoch", config.get("epochs", 90)), **(source or {}))
    challenges = json.loads(paths["challenges"].read_text())
    if not challenges or any("output" in pair for task in challenges.values() for pair in task["test"]):
        raise ValueError("Adaptation challenges must contain input-only queries")
    data_path = directory / "data/challenges.json"
    write_json(data_path, challenges)
    request = {"config": config, "defaults": pinned["SAVED_DEFAULTS"],
               "checkpoint_identity": checkpoint_identity, "source_checkpoint": source,
               "base_checkpoint_identity": pinned["CHECKPOINT_IDENTITY"],
               "checkpoint_path": str(plan["pretrain"]["checkpoint_path"].resolve()),
               "run_dir": str(directory), "runtime": str(runtime), "data_path": str(data_path),
               "save_path": str(directory / "adaptation/tiny.pt"), "runtime_sha256": pinned["RUNTIME_SHA256"]}
    request_path = directory / "adaptation_request.json"
    if config.get("decode_only"):
        request["save_path"] = request["checkpoint_path"]
    write_json(request_path, request)
    environment = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       MDLARC_REPORT_FIRST_UPDATE="1")
    phases = {}
    stages = ([] if config.get("decode_only") else ["train"]) + (["probe"] if config.get("decode_probe") else []) + ["decode"]
    for phase in stages:
        if phase == "decode" and config.get("wait_for_decode_settings"):
            settings_path = directory / "decode_settings.json"
            print("Waiting for common decoding settings after both checkpoint probes.", flush=True)
            deadline = time.monotonic() + 1200
            while not settings_path.is_file():
                if time.monotonic() >= deadline:
                    raise TimeoutError("Common decoding settings were not supplied within 20 minutes")
                time.sleep(2)
            settings = json.loads(settings_path.read_text())
            if (set(settings) != {"eval_batch_size", "inference_speculative"}
                    or type(settings["eval_batch_size"]) is not int or settings["eval_batch_size"] < 1
                    or type(settings["inference_speculative"]) is not bool):
                raise ValueError("Invalid common decoding settings")
            request["config"].update(settings)
            write_json(request_path, request)
        started = time.perf_counter()
        subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--phase", phase,
                        "--request", str(request_path)], cwd=root, env=environment, check=True)
        phases[phase + "_seconds"] = time.perf_counter() - started
        write_json(directory / "phase_times.json", phases)
        if phase == "probe":
            result = json.loads((directory / "decoding_probe.json").read_text())
            request["config"] = {**config, "eval_batch_size": result["selected_batch_size"],
                                  "inference_speculative": result["selected_speculative"]}
            write_json(request_path, request)
    # Open query answers only after the decoding subprocess has succeeded.
    module_spec = importlib.util.spec_from_file_location("adaptation_scoring", runtime / "kaggle/submission_utils.py")
    scoring = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(scoring)
    predictions = json.loads((directory / "predictions.json").read_text())
    scores = scoring.score_validation(predictions, challenges, json.loads(paths["solutions"].read_text()))
    write_json(directory / "validation_scores.json", scores)
    if config.get("decode_only") and "training_stats" in paths:
        training = json.loads(paths["training_stats"].read_text())
        write_json(directory / "training_stats.json", training)
    else:
        training = json.loads((directory / "training_stats.json").read_text()) if not config.get("decode_only") else {}
    summary = {"config": request["config"], "checkpoint_identity": checkpoint_identity,
               "runtime_sha256": pinned["RUNTIME_SHA256"], **phases,
               "training": training,
               "decoding": json.loads((directory / "decoding_stats.json").read_text()), "scores": scores}
    write_json(directory / "validation_run_info.json", summary)
    print("VALIDATION COMPLETE", json.dumps(scores), flush=True)


def runtime_modules(request):
    config = request["config"]
    if not config.get("enable_compile", True):
        os.environ["TORCH_COMPILE_DISABLE"] = "1"
    if config.get("attention_backend", "flash") == "native":
        sys.modules["flash_attn"] = None
        sys.modules["flash_attn_interface"] = None
    import torch
    import torch._dynamo
    torch.set_num_threads(1)
    for option in ("cache_size_limit", "recompile_limit"):
        if hasattr(torch._dynamo.config, option):
            setattr(torch._dynamo.config, option, 64)
    sys.path[:0] = [str(Path(request["runtime"]) / "src"), str(Path(request["runtime"]) / "kaggle")]
    import submission_utils as ku
    cfg = ku.finetune_config(request["defaults"], data_path=Path(request["data_path"]),
        save_path=Path(request["save_path"]), epochs=config.get("epochs", 90),
        batch_size=config.get("batch_size", 1), accumulation=config.get("accumulation", 40),
        max_augments=config.get("train_augments", 80), num_workers=0,
        normuon_lr=config.get("normuon_lr", .0016), adamw_lr=config.get("adamw_lr", .00028))
    cfg.seed = config.get("seed", 42)
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.train_log_mode, cfg.log_location = "10_steps", "both"
    cfg.checkpoint_epochs = config.get("checkpoint_epochs", [])
    cfg.inference_speculative = config.get("inference_speculative", False)
    cfg.prefill_token_budget = config.get("prefill_token_budget", 16384)
    return torch, ku, cfg


def memory_stats(torch):
    if not torch.cuda.is_available():
        return {}
    return {"peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__}


def train_worker(request):
    # Import our reusable adaptation code before selecting the preserved runtime.
    import parameter_efficient as peft
    torch, ku, cfg = runtime_modules(request)
    import build
    import train
    import checkpoint_io
    config = request["config"]
    checkpoint = torch.load(request["checkpoint_path"], map_location="cpu", weights_only=False, mmap=True)
    ku.verify_checkpoint_metadata(checkpoint, request["checkpoint_identity"])
    adapted = ku.adapt_checkpoint(checkpoint)
    del checkpoint
    model, dataset, loader, device, _ = build.build_model_and_data(cfg, checkpoint=adapted)
    model._loaded_checkpoint = None
    del adapted
    report = peft.configure(model, config["method"], last=config.get("last", 4),
                            rank=config.get("rank", 16), alpha=config.get("alpha", 16))
    from common import set_seed
    set_seed(cfg.seed)
    print("Adaptation parameters:", json.dumps(report), flush=True)
    if config["method"] == "lora":
        cfg.optimizer, cfg.adamw_lr, cfg.lr_floor = "adamw", config.get("lora_lr", 1e-4), .01
        def optimizer(*args, **kwargs):
            parameters = [p for p in model.parameters() if p.requires_grad]
            return torch.optim.AdamW([{"params": parameters, "group_name": "adamw_decay"}],
                lr=cfg.adamw_lr, weight_decay=config.get("lora_weight_decay", .01), betas=(.9, .999))
        train._build_optimizer = optimizer
        original_save = train.maybe_save_model
        def save_model(model, dataset, data_path, save_path, **kwargs):
            if save_path is None:
                return
            adapter_path = Path(save_path).with_suffix(".adapters.pt")
            state = {"adapter_state": peft.adapter_state(model), "config": config,
                     "source_checkpoint": request["source_checkpoint"],
                     "runtime_sha256": request["runtime_sha256"],
                     **{key: value.state_dict() if key in {"optimizer", "scheduler"} and value is not None else value
                        for key, value in kwargs.items() if key not in {"checkpoint_keep_last", "checkpoint_base_path"}}}
            checkpoint_io.save_checkpoint_atomically(adapter_path, lambda path: torch.save(state, path))
            with peft.merged_state_dict(model):
                original_save(model, dataset, data_path, save_path,
                              **{**kwargs, "optimizer": None, "scheduler": None})
        train.maybe_save_model = save_model
    counters = {"optimizer_updates": 0}
    original_optimizer = train._build_optimizer
    def counted_optimizer(*args, **kwargs):
        result = original_optimizer(*args, **kwargs)
        def count_update(optimizer, args, kwargs):
            counters["optimizer_updates"] += 1
        result.register_step_post_hook(count_update)
        return result
    train._build_optimizer = counted_optimizer
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    loss = ku.check_backward(model, loader, device)
    print(f"Native adaptation backward passed: loss={loss:.6f}", flush=True)
    epoch_times = []
    original_epoch = train.train_one_epoch
    def timed_epoch(*args, **kwargs):
        started = time.perf_counter()
        result = original_epoch(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - started)
        write_json(Path(request["run_dir"]) / "training_progress.json",
            {"epochs_completed": len(epoch_times), "microbatch_steps": result, **counters,
             "epoch_seconds": epoch_times, **report, **memory_stats(torch)})
        print(f"Adaptation epoch {len(epoch_times)}: {epoch_times[-1]:.2f}s; "
              f"optimizer updates {counters['optimizer_updates']}; microbatch steps {result}", flush=True)
        return result
    train.train_one_epoch = timed_epoch
    started = time.perf_counter()
    train.train_model(cfg, model=model, dataset=dataset, dataloader=loader,
                      device=device, data_path=Path(request["data_path"]))
    write_json(Path(request["run_dir"]) / "training_stats.json",
        {**report, **memory_stats(torch), **counters, "epoch_seconds": epoch_times, "seconds": time.perf_counter() - started,
         "effective_batch_size": cfg.batch_size * cfg.gradient_accumulation_steps,
         "optimizer": cfg.optimizer, "adamw_lr": cfg.adamw_lr, "normuon_lr": cfg.normuon_lr})


def decode_worker(request):
    torch, ku, cfg = runtime_modules(request)
    import evaluate
    cfg.checkpoint_mode, cfg.reset_training_state, cfg.extend_checkpoint_task_ids = None, False, False
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    directory = Path(request["run_dir"])
    checkpoint = torch.load(request["save_path"], map_location="cpu", weights_only=False, mmap=True)
    ku.verify_checkpoint_metadata(checkpoint, request["checkpoint_identity"])
    del checkpoint
    started = time.perf_counter()
    _, candidates, raw = evaluate.run_evaluation(cfg, run_name=str(directory / "evals/final"),
        max_augments=request["config"].get("eval_augments", 80), data_path=Path(request["data_path"]),
        checkpoint_path=Path(request["save_path"]), batch_size=request["config"].get("eval_batch_size", 1),
        splits=["test"], task_ids=None, timing_path=directory / "timing.txt")
    repairs = ku.finalize_submission(raw, directory / "predictions.json", json.loads(Path(request["data_path"]).read_text()))
    with gzip.open(directory / "generated_candidates.json.gz", "wt") as handle:
        json.dump(candidates, handle)
    write_json(directory / "decoding_stats.json", {**memory_stats(torch),
        "seconds": time.perf_counter() - started, "malformed_attempt_replacements": repairs,
        "batch_size": request["config"].get("eval_batch_size", 1),
        "speculative": cfg.inference_speculative, "prefill_token_budget": cfg.prefill_token_budget})


def probe_worker(request):
    """Bounded, answer-free batching/speculation check on the longest contexts."""
    torch, ku, cfg = runtime_modules(request)
    import build
    import evaluate
    config = request["config"]
    cfg.checkpoint_path = Path(request["save_path"])
    cfg.checkpoint_mode, cfg.reset_training_state, cfg.extend_checkpoint_task_ids = None, False, False
    model, dataset, _, device, _ = build.build_model_and_data(
        cfg, is_eval=True, eval_splits=("test",), build_dataloader=False)
    model._loaded_checkpoint = None
    model = evaluate._prepare_model_for_inference(model)
    work = evaluate._build_inference_attempts(dataset, "test", None, None, 2, True)
    cases = config.get("probe_cases", 32)
    items = sorted(work, key=lambda item: len(item.prompt), reverse=True)[:cases]
    padded = evaluate._prompt_bucket_length(max(len(item.prompt) for item in items),
        max_seq_len=model.config.max_seq_len, max_new_tokens=config.get("probe_tokens", 64))
    results = {"cases": len(items), "max_prompt_tokens": max(len(item.prompt) for item in items),
               "tokens_limit": config.get("probe_tokens", 64), "timings": [],
               "selected_batch_size": config.get("eval_batch_size", 32), "selected_speculative": False}
    destination = Path(request["run_dir"]) / "decoding_probe.json"
    def generate(selected, speculative):
        return evaluate.batched_greedy_generate(model=model,
            prompts=[item.prompt for item in selected], example_ids=[item.task_embedding_ids for item in selected],
            dihedral_ids=[item.dihedral_index for item in selected], cached_positions=[item.positions for item in selected],
            padded_prompt_length=padded, device=device, max_new_tokens=config.get("probe_tokens", 64),
            temperature=0, speculative=speculative, compile_decode=config.get("enable_compile", True))
    def timed(selected, speculative, label):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        output = generate(selected, speculative)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        tokens = sum(len(sequence)-len(item.prompt) for sequence,item in zip(output,selected))
        row = {"mode": label, "batch_size": len(selected), "seconds": elapsed,
               "generated_tokens": tokens, "tokens_per_second": tokens/elapsed, **memory_stats(torch)}
        results["timings"].append(row)
        write_json(destination, results)
        print("Decode probe:", json.dumps(row), flush=True)
        return output, elapsed
    def deadline(signum, frame):
        raise TimeoutError("Bounded decoding probe reached its time limit")
    previous = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(config.get("probe_max_seconds", 900))
    try:
        reference = []
        for item in items[:4]:
            output, _ = timed([item], False, "serial_batch1_reference")
            reference.extend(output)
        normal, _ = timed(items, False, "batched_serial_warmup")
        results["batch_parity_disagreements"] = sum(a != b for a,b in zip(reference,normal[:4]))
        _, serial_time = timed(items, False, "batched_serial")
        if config.get("inference_speculative", False):
            speculative, _ = timed(items, True, "batched_speculative_warmup")
            results["speculative_parity_disagreements"] = sum(a != b for a,b in zip(normal,speculative))
            _, speculative_time = timed(items, True, "batched_speculative")
            results["speculative_speedup"] = serial_time/speculative_time
            results["selected_speculative"] = results["speculative_parity_disagreements"] == 0 and speculative_time < serial_time
        peak = memory_stats(torch).get("peak_reserved_gib", 0)
        results["selected_batch_size"] = config.get("eval_batch_size", 32) if peak < 65 else 16
        if results["batch_parity_disagreements"]:
            results["selected_batch_size"] = 1
            results["selected_speculative"] = False
    except TimeoutError as exc:
        results["timeout"] = str(exc)
        results["selected_speculative"] = False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        write_json(destination, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "decode", "probe"), required=True)
    parser.add_argument("--request", type=Path, required=True)
    arguments = parser.parse_args()
    request = json.loads(arguments.request.read_text())
    {"train": train_worker, "decode": decode_worker, "probe": probe_worker}[arguments.phase](request)
