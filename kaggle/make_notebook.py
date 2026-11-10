"""Rebuild the self-contained Kaggle notebook from this checkout's runtime.

Run with: venv/bin/python kaggle/make_notebook.py
This packages only an explicit list of source files; no datasets or credentials.
"""

import base64
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import textwrap
import zipfile
from string import Template

from checkpoint_profiles import PROFILES


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FILES = [
    "src/build.py", "src/checkpoint_io.py", "src/common.py", "src/evaluate.py",
    "src/group_attention.py",
    "src/context_palette.py",
    "src/grid_chain.py", "src/intermediate_sampling.py",
    "src/distributed_training.py",
    "src/demo_schedule.py", "src/pair_identity.py",
    "src/target_demo_attention.py",
    "src/stable_losses.py",
    "src/tinytransformer.py", "src/train.py", "src/utils.py", "src/variations.py",
    "kaggle/submission_utils.py",
    "kaggle/multi_gpu.py",
]

# Saved defaults of the exact 25-epoch public/composition run. Architecture fields
# are read from the checkpoint at runtime. Finetune LR overrides are visible below.
DEFAULTS = {
    "seed": 42, "optimizer": "normuon", "FA": 2,
    "normuon_momentum": 0.95, "normuon_momentum_min": 0.85,
    "normuon_warmup_steps": 0, "normuon_cooldown_steps": 0,
    "normuon_beta2": 0.95, "normuon_bucket_size": 0,
    "normuon_qkv_lr_multiplier": 1.0, "normuon_attention_lr_multiplier": 1.0,
    "normuon_fc_in_lr_multiplier": 1.0, "normuon_fc_out_lr_multiplier": 1.0,
    "normuon_example_value_latent_lr_multiplier": 1.0,
    "normuon_other_lr_multiplier": 1.0, "normuon_residual_output_lr_multiplier": 1.0,
    "adamw_decay_lr_multiplier": 1.0, "adamw_attention_lr_multiplier": 1.0,
    "adamw_token_embedding_lr_multiplier": 1.0, "adamw_task_factor_lr_multiplier": 1.0,
    "adamw_example_value_latent_lr_multiplier": 1.0, "adamw_no_decay_lr_multiplier": 1.0,
    "weight_decay": 0.1, "attention_weight_decay": 0.01,
    "token_embedding_weight_decay": 0.01, "task_embedding_weight_decay": 0.01,
    "grad_clip": 1.0, "warmup_pct": 0.005, "resume_warmup_pct": 0.0,
    "wsd_decay_start_pct": 0.8, "lr_floor": 0.0001,
    "mtp_decay": [0.8, 0.9, 1.0],
    "augmentation_sampling_mode": "single_grid_prefix",
    "inference_temperature": None, "inference_top_k": None,
}


def cell(kind, source):
    source = textwrap.dedent(source).strip() + "\n"
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def frozen_runtime(path):
    """Read a verified notebook payload without importing its model code."""
    notebook = json.loads(Path(path).read_text())
    values = {}
    for item in notebook["cells"]:
        if item["cell_type"] != "code":
            continue
        for statement in ast.parse("".join(item["source"])).body:
            if isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name):
                name = statement.targets[0].id
                if name in {"RUNTIME_B64", "RUNTIME_SHA256"}:
                    values[name] = ast.literal_eval(statement.value)
    raw = base64.b64decode(values["RUNTIME_B64"])
    if hashlib.sha256(raw).hexdigest() != values["RUNTIME_SHA256"]:
        raise ValueError("Source notebook runtime checksum mismatch.")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if not set(archive.namelist()) <= set(RUNTIME_FILES):
            raise ValueError("Source notebook contains files outside the runtime allowlist.")
        return {name: archive.read(name) for name in archive.namelist()}


def make_notebook(profile="public-v3-e25", destination=None, *, runtime_notebook=None,
                  refresh_runtime=False):
    selected = PROFILES[profile]
    if runtime_notebook is not None and refresh_runtime:
        raise ValueError("Choose a frozen runtime or refresh, not both.")
    existing = ROOT / "kaggle" / selected["filename"]
    if runtime_notebook is None and not refresh_runtime and selected.get("runtime_source"):
        runtime_notebook = ROOT / selected["runtime_source"]
    if selected.get("runtime_source_sha256"):
        if refresh_runtime:
            raise ValueError("This comparison profile requires its pinned model runtime.")
        if hashlib.sha256(Path(runtime_notebook).read_bytes()).hexdigest() != selected["runtime_source_sha256"]:
            raise ValueError("Last-four comparison source notebook differs from its pinned identity.")
    if runtime_notebook is None and not refresh_runtime and selected.get("preserve_runtime"):
        if not existing.is_file():
            raise FileNotFoundError("Preserved model runtime is missing; select --runtime-notebook or explicitly --refresh-runtime.")
        runtime_notebook = existing
    if runtime_notebook is not None:
        sources = frozen_runtime(runtime_notebook)
        # Kaggle controls may evolve while the trained model implementation stays pinned.
        for name in RUNTIME_FILES:
            if name.startswith("kaggle/"):
                sources[name] = (ROOT / name).read_bytes()
        print(f"Preserving packaged model source from {runtime_notebook}")
    else:
        sources = {name: (ROOT / name).read_bytes() for name in RUNTIME_FILES}
    defaults = dict(DEFAULTS, augmentation_sampling_mode=selected["augmentation_sampling_mode"],
                    training_demo_schedule=selected["training_demo_schedule"],
                    inference_demo_policy=selected["inference_demo_policy"],
                    prefill_token_budget=selected.get("prefill_token_budget", 16384))
    packed = io.BytesIO()
    with zipfile.ZipFile(packed, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, source in sources.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 10, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, source)
    raw = packed.getvalue()
    encoded = base64.b64encode(raw).decode("ascii")
    digest = hashlib.sha256(raw).hexdigest()
    cells = [cell("markdown", f"""
        # mdlARC: ARC Prize 2026 ARC-AGI-2 submission

        Uses **{selected['title']}**, the final pretraining checkpoint from
        `{selected['session']}`. Its file size, SHA-256, epoch, and architecture
        are checked before adaptation. Default adaptation: **{selected['epochs']} epochs**,
        effective batch **{selected['batch_size'] * selected['accumulation']}**,
        80 training and inference augmentations.

        1. Attach the competition data and a Kaggle dataset containing `pretrain__tiny.pt`.
        2. Choose **L4 × 4** and turn **Internet off**. This code uses **only GPU 0**.
        3. Attach a compatible FlashAttention 2 wheel, or select `ATTENTION_BACKEND="native"`.
        4. First run with `SMOKE_TEST=True`. Then restart, set it to **False**, and Run All.
        5. **Save Version → Save & Run All** with `SMOKE_TEST=False`. Submit that version's
           `/kaggle/working/submission.json` through **Submit to Competition**.

        The notebook contains its own mdlARC source; no source-code dataset is needed.
        It finetunes **all parameters** on the supplied train pairs, then generates two
        predictions for each runtime test input. It never reads test solutions.

        Full instructions: `kaggle/README_600m.md` for the 600M profile,
        or `kaggle/README.md` for the smaller checkpoint.
        Current rules: [Kaggle code requirements](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview/code-requirements).
        Kaggle currently allows up to 12 hours and requires Internet off.
        The full L4 run has to be timed in Kaggle; smoke mode checks execution, not accuracy.
    """), cell("code", Template('''
        # SETTINGS — normally only change the checkpoint/wheel paths and SMOKE_TEST.
        import os
        import sys
        from pathlib import Path

        if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
            if sys.modules["torch"].cuda.device_count() != 1:
                raise RuntimeError("Restart the session before running this notebook to select GPU 0.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        CHECKPOINT_PATH = None  # Auto-find pretrain__tiny.pt / tiny.pt, or paste its exact /kaggle/input/... path.
        TEST_CHALLENGES = None  # Auto-find only the competition's arc-agi_test_challenges.json.
        ATTENTION_BACKEND = "flash"  # "flash" = FlashAttention 2; "native" = PyTorch attention (no wheel).
        FLASH_ATTN_WHEEL = None  # Exact /kaggle/input/.../flash_attn....whl path; None tries an installed package first.
        ENABLE_COMPILE = True
        SMOKE_TEST = False  # True: 2 tasks, 2 epochs, 2 augs; writes smoke_submission.json only.

        EPOCHS = $epochs
        TRAIN_BATCH_SIZE = $batch_size
        GRADIENT_ACCUMULATION_STEPS = $accumulation
        EVAL_BATCH_SIZE = $eval_batch_size
        TRAIN_AUGMENTS = 80
        EVAL_AUGMENTS = 80
        NUM_WORKERS = 2
        NORMUON_LR = $normuon_lr
        ADAMW_LR = $adamw_lr
        MAX_FINETUNE_HOURS = 8.0  # Abort early if steady-state ETA exceeds this; reserve time for decoding.

        if ATTENTION_BACKEND not in {"flash", "native"}:
            raise ValueError('ATTENTION_BACKEND must be "flash" or "native".')
        if int(os.environ.get("WORLD_SIZE", "1")) != 1 or "LOCAL_RANK" in os.environ:
            raise RuntimeError("Use the notebook directly in a fresh session, without torchrun/DDP.")
        if not ENABLE_COMPILE:
            os.environ["TORCH_COMPILE_DISABLE"] = "1"
        else:
            os.environ.pop("TORCH_COMPILE_DISABLE", None)
        WORKING = Path("/kaggle/working")
        WORKING.mkdir(parents=True, exist_ok=True)
        os.chdir(WORKING)
        if (WORKING / "submission.json").exists():
            (WORKING / "submission.json").replace(WORKING / "previous_submission.json")
        # Prevent any inherited remote-backup environment from affecting local saves.
        os.environ.pop("MDLARC_CHECKPOINT_QUEUE", None)
        os.environ.pop("MDLARC_RUN_DIR", None)
        print(f"Mode: {'SMOKE TEST — do not submit' if SMOKE_TEST else 'FULL SUBMISSION'}")
    ''').substitute({key: selected[key] for key in (
        "epochs", "batch_size", "accumulation", "eval_batch_size", "normuon_lr", "adamw_lr"
    )})), cell("markdown", """
        ## Environment and attention wheel

        Run this cell before choosing a wheel. It prints the actual environment, rather
        than assuming the four-month-old wheel still works. The wheel needs compatible
        Python, PyTorch, CUDA, Linux and C++ ABI. L4 uses FlashAttention **2**, not the
        Hopper-only FlashAttention 3 package. No packages are fetched from the internet.

        If `flash` fails, attach a compatible wheel and restart, or explicitly select
        `native`. Native attention preserves the prefix attention mask but can use more
        memory/time. Keep the selected effective batch when changing the microbatch
        by adjusting accumulation. The 600M profile starts at microbatch one.
    """), cell("code", '''
        import importlib.util
        import platform
        import subprocess
        import torch

        # Available before extracting mdlARC, including in an Internet-off notebook.
        import hashlib
        import json
        import shutil

        def restore_wheel_filename(source):
            source = Path(source)
            manifest_path = source.parent / "wheel-manifest.json"
            if not manifest_path.is_file():
                return source
            manifest = json.loads(manifest_path.read_text())
            name = manifest["wheel"]
            if Path(name).name != name or not name.endswith(".whl"):
                raise ValueError("Invalid wheel filename in build manifest.")
            with source.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != manifest["wheel_sha256"]:
                raise ValueError("FlashAttention wheel checksum failed.")
            target = WORKING / name
            if source.resolve() != target.resolve():
                shutil.copyfile(source, target)
            return target

        print("Python:", sys.version)
        print("Platform:", platform.platform(), platform.machine())
        print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda)
        print("C++ ABI:", getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", "unknown"))
        if not torch.cuda.is_available():
            raise RuntimeError("Select GPU L4 × 4 in Kaggle Settings, then restart the session.")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("GPU selection was applied too late. Restart, then Run All.")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This optimizer needs a CUDA GPU with bfloat16 support; choose L4.")
        print("Using:", torch.cuda.get_device_name(0))
        print("GPU memory GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
        subprocess.run(["nvidia-smi"], check=True)
        for package in ("numpy", "numba"):
            if importlib.util.find_spec(package) is None:
                raise RuntimeError(f"Missing {package}; attach an offline wheel. Do not enable Internet for submission.")

        flash_probe = [sys.executable, "-c", "from flash_attn import flash_attn_varlen_func; import flash_attn; print(flash_attn.__version__)"]
        if ATTENTION_BACKEND == "flash":
            probe = subprocess.run(flash_probe, text=True, capture_output=True)
            if FLASH_ATTN_WHEEL is not None or probe.returncode:
                if FLASH_ATTN_WHEEL is None:
                    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
                    wheels = sorted(p for p in Path("/kaggle/input").rglob("*.whl")
                                    if p.name.startswith("flash_attn-") and tag in p.name)
                    if len(wheels) != 1:
                        raise RuntimeError(
                            f"No working FlashAttention import; found {len(wheels)} wheels for {tag}. "
                            "Attach a compatible wheel and set FLASH_ATTN_WHEEL, or set ATTENTION_BACKEND='native'.\\n"
                            + probe.stderr[-3000:]
                        )
                    FLASH_ATTN_WHEEL = wheels[0]
                wheel = Path(FLASH_ATTN_WHEEL)
                if not wheel.is_file() or wheel.suffix != ".whl":
                    raise FileNotFoundError(f"Offline wheel not found: {wheel}")
                wheel = restore_wheel_filename(wheel)
                for dependency in sorted(Path(FLASH_ATTN_WHEEL).parent.glob("einops-*.whl")):
                    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                                    "--no-deps", str(dependency)], check=True)
                print("Installing offline wheel:", wheel)
                subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                                "--force-reinstall", str(wheel)], check=True)
                probe = subprocess.run(flash_probe, text=True, capture_output=True)
            if probe.returncode:
                raise RuntimeError("FlashAttention wheel is incompatible with this environment.\\n" + probe.stderr[-5000:])
            print("FlashAttention:", probe.stdout.strip())
        else:
            print("Using native PyTorch attention; no FlashAttention wheel will be used.")
        import torch._dynamo
        for option in ("cache_size_limit", "recompile_limit"):
            if hasattr(torch._dynamo.config, option):
                setattr(torch._dynamo.config, option, 64)
    '''), cell("markdown", """
        ## Embedded source

        This payload contains only the mdlARC runtime and submission helpers. It includes
        the latent-width fix needed when loading the 3072-wide embeddings for fewer than
        3072 Kaggle tasks. Its checksum is verified before extraction.
    """), cell("code", f'''
        import base64
        import hashlib
        import io
        import zipfile

        RUNTIME_SHA256 = "{digest}"
        RUNTIME_B64 = "{encoded}"
        runtime_bytes = base64.b64decode(RUNTIME_B64)
        if hashlib.sha256(runtime_bytes).hexdigest() != RUNTIME_SHA256:
            raise RuntimeError("Embedded runtime checksum failed; re-upload the original notebook.")
        RUNTIME = WORKING / "mdlarc_runtime"
        with zipfile.ZipFile(io.BytesIO(runtime_bytes)) as archive:
            for entry in archive.infolist():
                target = (RUNTIME / entry.filename).resolve()
                target.relative_to(RUNTIME.resolve())
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(entry))
        sys.path.insert(0, str(RUNTIME / "src"))
        sys.path.insert(0, str(RUNTIME / "kaggle"))
        del runtime_bytes, RUNTIME_B64

        if ATTENTION_BACKEND == "native":
            # Ignore an incompatible installed extension as well as attached wheels.
            sys.modules["flash_attn"] = None
            sys.modules["flash_attn_interface"] = None
        import build
        import tinytransformer
        import train
        import submission_utils as ku
        if ATTENTION_BACKEND == "native":
            tinytransformer.flash_attn_varlen_func = None
            tinytransformer.flash_attn_varlen_qkvpacked_func = None
            tinytransformer.flash_attn_varlen_kvpacked_func = None
        elif tinytransformer.flash_attn_varlen_func is None:
            raise RuntimeError("FlashAttention was not imported into the notebook kernel. Restart and Run All.")
        print("Runtime SHA-256:", RUNTIME_SHA256)
    '''), cell("markdown", """
        ## Checkpoint and runtime challenges

        The selected profile pins the checkpoint's exact size, SHA-256, epoch, and
        architecture. The original checkpoint stays on the CPU. Old optimizer state
        and pretraining factor rows are discarded; fresh factors are created for runtime
        tasks. All shared transformer and latent weights are retained.

        Some public evaluation families were included in pretraining. Submit on the supplied
        **test** challenges; Kaggle replaces that file with its scoring tasks.
    """), cell("code", '''
        import json
        import gc
        from time import perf_counter

        CHECKPOINT_PATH = ku.find_checkpoint(Path("/kaggle/input"), CHECKPOINT_PATH)
        TEST_CHALLENGES = ku.find_test_challenges(Path("/kaggle/input"), TEST_CHALLENGES)
        print("Checkpoint:", CHECKPOINT_PATH)
        print("Challenges:", TEST_CHALLENGES)
        checkpoint_hash = ku.verify_checkpoint_file(CHECKPOINT_PATH, CHECKPOINT_IDENTITY)
        print("Checkpoint SHA-256 verified:", checkpoint_hash)

        data_path = WORKING / "runtime_challenges.json"
        challenges = ku.prepare_challenges(TEST_CHALLENGES, data_path, smoke_test=SMOKE_TEST)
        print("Tasks:", len(challenges), "train pairs:", sum(len(t["train"]) for t in challenges.values()),
              "test inputs:", sum(len(t["test"]) for t in challenges.values()))
        # mmap avoids eagerly copying multi-GB optimizer tensors into RAM.
        # weights_only=False is needed for this trusted, hash-verified legacy checkpoint.
        original = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False, mmap=True)
        ku.verify_checkpoint_metadata(original, CHECKPOINT_IDENTITY)
        adapted = ku.adapt_checkpoint(original)
        print("Saved epoch:", original.get("epoch"), "saved architecture:", original["config"])
        del original
        gc.collect()
    '''), cell("markdown", """
        ## Full-model finetuning

        Starts a **fresh optimizer and schedule**, rather than resuming pretraining.
        Gradient accumulation preserves the selected effective batch on GPU 0.
        All model parameters are trainable, including shared embeddings and MTP heads.
        The checkpoint determines width/depth/MTP/dropout; no 1B-era architecture is used.
        The 600M profile uses the completed 90-epoch development adaptation recipe:
        NorMuon 0.0016, AdamW 0.00028, effective batch 40, all other demos for training,
        and balanced leave-one-out demo contexts for inference. The small profile retains
        its original template. GPU/runtime differences still require a Kaggle smoke test.
    """), cell("code", f'''
        SAVED_DEFAULTS = json.loads({json.dumps(json.dumps(defaults))})
        save_path = WORKING / "runs" / ("smoke_finetuned.pt" if SMOKE_TEST else "kaggle_finetuned.pt")
        cfg = ku.finetune_config(
            SAVED_DEFAULTS, data_path=data_path, save_path=save_path,
            epochs=2 if SMOKE_TEST else EPOCHS,
            batch_size=1 if SMOKE_TEST else TRAIN_BATCH_SIZE,
            accumulation=1 if SMOKE_TEST else GRADIENT_ACCUMULATION_STEPS,
            max_augments=2 if SMOKE_TEST else TRAIN_AUGMENTS,
            num_workers=0 if SMOKE_TEST else NUM_WORKERS,
            normuon_lr=NORMUON_LR, adamw_lr=ADAMW_LR,
        )
        model, dataset, dataloader, device, _ = build.build_model_and_data(cfg, checkpoint=adapted)
        if len(dataloader) < cfg.gradient_accumulation_steps:
            raise ValueError("Too few minibatches for the effective batch. Lower batch size/accumulation.")
        if len(dataset.task_ids) != len(challenges) or len(dataset.embedding_task_ids) != len(challenges):
            raise RuntimeError("Unexpected task-factor mapping for runtime tasks.")
        if model.config.num_task_embedding_latents != adapted["config"]["num_task_embedding_latents"]:
            raise RuntimeError("Shared task latent width changed.")
        model.requires_grad_(True)
        model._loaded_checkpoint = None  # Release old CPU weights; optimizer starts fresh.
        del adapted
        gc.collect()
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        factors = model.task_factor_embedding.weight.numel()
        print(f"Trainable: {{trainable:,}} / {{total:,}} (100%); task factors: {{factors:,}}")
        print("Effective batch:", cfg.batch_size * cfg.gradient_accumulation_steps)
        print("CUDA forward/backward preflight loss:", ku.check_backward(model, dataloader, device))
        torch.cuda.empty_cache()

        epoch_times = []
        real_train_one_epoch = train.train_one_epoch
        def timed_train_one_epoch(*args, **kwargs):
            start = perf_counter()
            result = real_train_one_epoch(*args, **kwargs)
            torch.cuda.synchronize()
            elapsed = perf_counter() - start
            epoch_times.append(elapsed)
            # Skip the first epoch, which includes compilation warmup.
            if len(epoch_times) >= 3:
                seconds_per_epoch = sum(epoch_times[1:]) / (len(epoch_times) - 1)
                estimated_hours = (epoch_times[0] + seconds_per_epoch * (cfg.epochs - 1)) / 3600
                if len(epoch_times) in (3, 10, 25, 50, 100, 200):
                    print(f"Estimated full finetune: {{estimated_hours:.2f}} hours; decoding is additional.")
                if not SMOKE_TEST and estimated_hours > MAX_FINETUNE_HOURS:
                    raise RuntimeError(
                        f"Finetune ETA {{estimated_hours:.2f}}h exceeds {{MAX_FINETUNE_HOURS}}h budget. "
                        "Restart and lower EPOCHS, or try a compatible flash wheel / ENABLE_COMPILE=True. "
                        "Leave room for decoding under the competition's 12h limit."
                    )
            return result

        train.train_one_epoch = timed_train_one_epoch
        started = perf_counter()
        try:
            train.train_model(cfg, model=model, dataloader=dataloader, dataset=dataset,
                              device=device, data_path=data_path)
        finally:
            train.train_one_epoch = real_train_one_epoch
        print(f"Finetune completed in {{(perf_counter() - started) / 3600:.3f}} hours.")
        if not save_path.is_file():
            raise RuntimeError("Finetuning did not save a checkpoint.")
        del model, dataset, dataloader
        gc.collect()
        torch.cuda.empty_cache()
        torch._dynamo.reset()
    '''), cell("markdown", """
        ## Decode and write the submission

        Uses ordinary greedy decoding, 80 test-time augmentations, and the repository's
        AAIVR aggregation to select two attempts. Both finetuning and decoding use the
        same runtime challenge file. A malformed decoded grid is replaced with the other
        valid attempt, or the corresponding input grid if neither attempt is valid;
        replacements are reported. Missing tasks or test inputs cause an error.
    """), cell("code", '''
        from argparse import Namespace
        import evaluate

        eval_cfg = Namespace(**vars(cfg))
        eval_cfg.checkpoint_mode = None
        eval_cfg.extend_checkpoint_task_ids = False
        eval_cfg.inherit_checkpoint_rearc_data_path = False
        eval_cfg.reset_training_state = False
        eval_cfg.distributed_inference = False
        eval_cfg.inference_speculative = False
        eval_cfg.mode = "full-decode"
        started = perf_counter()
        run_name, eval_results, raw_submission = evaluate.run_evaluation(
            eval_cfg, run_name="smoke_decode" if SMOKE_TEST else "kaggle_decode",
            max_augments=2 if SMOKE_TEST else EVAL_AUGMENTS,
            data_path=data_path, checkpoint_path=save_path,
            batch_size=2 if SMOKE_TEST else EVAL_BATCH_SIZE, splits=["test"], task_ids=None,
            timing_path=WORKING / "runs" / "timing.txt",
        )
        output = WORKING / ("smoke_submission.json" if SMOKE_TEST else "submission.json")
        repairs = ku.finalize_submission(raw_submission, output, challenges)
        if repairs:
            print(f"Replaced {len(repairs)} malformed attempts:", repairs[:20])
        print(f"Decode completed in {(perf_counter() - started) / 3600:.3f} hours.")
        del eval_results
        gc.collect()

        summary = {
            "smoke_test": SMOKE_TEST, "checkpoint_sha256": checkpoint_hash,
            "runtime_sha256": RUNTIME_SHA256, "attention_backend": ATTENTION_BACKEND,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "epochs": cfg.epochs, "effective_batch_size": cfg.batch_size * cfg.gradient_accumulation_steps,
            "train_augments": cfg.max_augments, "eval_augments": 2 if SMOKE_TEST else EVAL_AUGMENTS,
            "tasks": len(challenges), "test_inputs": sum(len(t["test"]) for t in challenges.values()),
            "malformed_attempt_replacements": repairs,
        }
        (WORKING / "submission_run_info.json").write_text(json.dumps(summary, indent=2))
        if SMOKE_TEST:
            print("SMOKE TEST PASSED. Restart, set SMOKE_TEST=False, then Save & Run All. Do not submit smoke output.")
        else:
            # Check against the complete competition input again, independently of the cached dict.
            full_tasks = json.loads(TEST_CHALLENGES.read_text())
            ku.validate_submission(json.loads(output.read_text()), full_tasks)
            print(f"READY: {output} ({output.stat().st_size:,} bytes), all runtime test inputs covered.")
            print("Submit submission.json from this successfully saved notebook version.")
    ''')]
    if selected.get("num_gpus", 1) > 1:
        from notebook_multi_gpu import distributed_cells
        cells = distributed_cells(cells, selected, defaults, cell)
    if selected.get("recipe") == "last4":
        from notebook_last4 import last_four_cells
        cells = last_four_cells(cells, selected, defaults, cell)
    cells[1]["source"] = [
        f"CHECKPOINT_IDENTITY = {selected!r}\n",
        *cells[1]["source"],
    ]
    notebook = {
        "cells": cells, "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "kaggle": {"isInternetEnabled": False, "language": "python", "sourceType": "notebook"},
        },
    }
    for index, item in enumerate(cells):
        item["id"] = f"mdlarc-{index:02d}"
    destination = Path(destination) if destination is not None else ROOT / "kaggle" / selected["filename"]
    destination.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"Wrote {destination} ({destination.stat().st_size:,} bytes)")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="public-v3-e25")
    parser.add_argument("--output", type=Path)
    runtime = parser.add_mutually_exclusive_group()
    runtime.add_argument("--runtime-notebook", type=Path,
                         help="Keep this notebook's model source and update only Kaggle helpers")
    runtime.add_argument("--refresh-runtime", action="store_true",
                         help="Explicitly replace the packaged model implementation with this checkout")
    args = parser.parse_args()
    make_notebook(args.profile, args.output, runtime_notebook=args.runtime_notebook,
                  refresh_runtime=args.refresh_runtime)
