#!/usr/bin/env python3
"""Shared detached training supervisor, checkpoint uploader, and deadline guard."""

import argparse
import codecs
import importlib.util
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import select
import re
import signal
import subprocess
import sys
import sysconfig
import tempfile
import time
import zipfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workflows.vast.common import (CHECKPOINT_SUFFIXES, ROLLING_CHECKPOINT_MARKER, ROOT_FOLDER, atomic_json, hashes,
                                   keep_b300_running, lock, now, relative_file)
from workflows.vast.drive import Drive, service
from workflows.vast.snapshot import results_archive
from workflows.vast.vast_api import Vast
from src.checkpoint_io import checkpoint_upload_due, prune_backed_up_checkpoints, prune_unscheduled_checkpoints


PATH_LIST_ENV_VARS = (
    "PATH", "LIBRARY_PATH", "LD_LIBRARY_PATH", "COMPILER_PATH", "CPATH",
    "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "OBJC_INCLUDE_PATH",
    "PKG_CONFIG_PATH", "PYTHONPATH",
)

CUDA_PREFLIGHT_CODE = """import json, os, pathlib, sys
import torch
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable in the selected container')
if not torch.cuda.is_bf16_supported():
    raise RuntimeError('CUDA BF16 is unavailable in the selected container')
if not hasattr(torch, 'compile') or not hasattr(torch, 'compiler'):
    raise RuntimeError('The selected Torch runtime does not support compilation')
if not hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
    raise RuntimeError('The selected Torch runtime is missing the required cudagraph API')
from triton.runtime import driver
target = driver.active.get_current_target()
compiled = torch.compile(lambda tensor: tensor.sin() + 1, fullgraph=True)
smoke_input = torch.arange(8, device='cuda', dtype=torch.float32)
smoke_output = compiled(smoke_input)
torch.cuda.synchronize()
torch.testing.assert_close(smoke_output, smoke_input.sin() + 1)
from workflows.vast.common import atomic_json
path = pathlib.Path(sys.argv[1])
report = json.loads(path.read_text())
report.update(python=sys.version, torch=torch.__version__, torch_cuda=torch.version.cuda,
              gpu=torch.cuda.get_device_name(0), bf16=True,
              cuda_home=os.environ.get('CUDA_HOME'), cuda_include_path=os.environ.get('CPATH'),
              triton_ptxas_path=os.environ.get('TRITON_PTXAS_PATH'),
              triton_target=str(target), torch_compile_smoke=True)
atomic_json(path, report)
print(json.dumps(report), flush=True)
"""


def upload_verified(store, path, folder, name):
    for attempt in range(3):
        try:
            return store.upload(path, folder, name)
        except RuntimeError:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))


def pip_install_command(*requirements):
    command = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-input"]
    # NGC's disposable system Python carries PEP 668's marker. Keep the image's
    # Torch; install only the missing helpers and explicitly selected FA4.
    if sys.prefix == sys.base_prefix and (Path(sysconfig.get_path("stdlib")) / "EXTERNALLY-MANAGED").is_file():
        command.append("--break-system-packages")
    return [*command, *requirements]


def ensure_drive_dependencies(code, run_dir, plan, private):
    try:
        present = all(importlib.util.find_spec(name) is not None for name in ("googleapiclient", "google.auth", "google_auth_httplib2"))
    except ModuleNotFoundError:
        present = False
    if present:
        return
    def health():
        if time.time() >= plan["deadline_unix"] - plan["policy"]["backup_grace_seconds"]:
            raise RuntimeError("Setup exhausted the runtime allowance")
    run_training(pip_install_command("-r", str(code / "workflows/vast/requirements.txt")), code, clean_training_env(plan, private),
                 run_dir / "bootstrap.log", health)


def store_for(workspace, plan, role):
    private = workspace / "private"
    store = Drive(service(private / "drive-token.json"), plan["drive_config"], plan["branch"],
                  control=private / role, max_upload_gb=plan["policy"]["max_upload_gb"],
                  max_download_gb=plan["policy"]["max_download_gb"], transfer_file=private / "transfers.json")
    store.preflight()
    store.verify_run(plan["drive_folder_id"], plan["session"])
    return store


def package_and_upload(store, run_dir, workspace, plan, *, final=False):
    archive = workspace / ("results-final.tar.gz" if final else "results-live.tar.gz")
    digest = results_archive(run_dir, archive)
    name = ("results-without-checkpoints-" if final else "artifacts-") + digest["sha256"] + ".tar.gz"
    return upload_verified(store, archive, plan["drive_folder_id"], name)


def backup_source_inputs(store, code, run_dir, plan):
    backups = {}
    for name, identity in plan.get('source_inputs', {}).items():
        path = relative_file(code, identity['archive_path'])
        saved = upload_verified(store, path, plan['drive_folder_id'], 'input-' + name + '-' + identity['sha256'] + '.zip')
        if any(saved[key] != identity[key] for key in ('sha256', 'size')):
            raise ValueError('External input backup differs from the pinned spec')
        backups[name] = saved
    if backups:
        atomic_json(run_dir / 'source_input_backups.json', backups)
    return backups


def checkpoint_paths(run_dir):
    return {p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*")
            if p.is_file() and not p.is_symlink() and p.suffix.lower() in CHECKPOINT_SUFFIXES
            and not p.stem.endswith(ROLLING_CHECKPOINT_MARKER)
            and p.relative_to(run_dir).parts[0] != "inputs"}


def verify_checkpoint_ledger(run_dir, ledger):
    uploaded = {entry["path"] for entry in ledger["files"]}
    retained = {entry["path"] for entry in ledger["files"] if not entry.get("local_deleted")}
    if retained != checkpoint_paths(run_dir) or len(uploaded) != len(ledger["files"]):
        return False
    for entry in ledger["files"]:
        if entry.get("local_deleted"):
            continue
        digest = hashes(relative_file(run_dir, entry["path"]))
        if digest["sha256"] != entry["sha256"] or digest["size"] != entry["size"]:
            return False
    return True


def automatic_checkpoint_uploads(plan):
    return plan.get("policy", {}).get("checkpoint_upload_mode", "automatic") == "automatic"


def uploader(workspace, plan):
    run_dir = workspace / "code/runs" / plan["run_name"]
    private = workspace / "private"
    store = store_for(workspace, plan, "uploader-drive")
    ledger_path = run_dir / "checkpoint_backups.json"
    ledger = (json.loads(ledger_path.read_text()) if ledger_path.exists() else
              {"folder_id": plan["drive_folder_id"], "files": [], "errors": [], "complete": False})
    ledger["errors"] = []
    ledger["complete"] = False
    automatic = automatic_checkpoint_uploads(plan)
    upload_every = plan.get("policy", {}).get("checkpoint_upload_every_epochs")
    ledger["upload_mode"] = "automatic" if automatic else "manual"
    uploaded = {entry["path"] for entry in ledger["files"]}
    atomic_json(ledger_path, ledger)
    atomic_json(private / "uploader-ready.json", {"pid": os.getpid(), "at": now()})
    last_artifacts = 0
    while True:
        try:
            events = sorted((private / "queue").glob("*.json")) if automatic else []
            for event in events:
                value = json.loads(event.read_text())
                if not checkpoint_upload_due(value["path"], upload_every):
                    continue
                if value["path"] in uploaded:
                    continue
                path = relative_file(run_dir, value["path"])
                if path.suffix.lower() not in CHECKPOINT_SUFFIXES or path.stat().st_size != value["size"]:
                    raise ValueError("Checkpoint queue entry is invalid or changed after publication")
                if value["path"] not in uploaded:
                    identity = upload_verified(store, path, plan["drive_folder_id"], value["path"].replace("/", "__"))
                    ledger["files"].append({"path": value["path"], **identity})
                    uploaded.add(value["path"])
                    atomic_json(ledger_path, ledger)
                    print(f"Verified checkpoint backup: {value['path']}", flush=True)
            for event in events:
                value = json.loads(event.read_text())
                prune_backed_up_checkpoints(run_dir, value, ledger)
                prune_unscheduled_checkpoints(run_dir, value, upload_every)
            atomic_json(ledger_path, ledger)
            stopping = (private / "uploader-stop").exists()
            if stopping:
                if not automatic:
                    ledger["local_checkpoint_paths"] = sorted(checkpoint_paths(run_dir))
                    atomic_json(ledger_path, ledger)
                    return 0  # Records succeeded; checkpoints intentionally remain on the VM.
                # Cover completed checkpoints whose queue event was lost in a process crash.
                for relative in sorted(checkpoint_paths(run_dir) - uploaded):
                    identity = upload_verified(store, relative_file(run_dir, relative), plan["drive_folder_id"], relative.replace("/", "__"))
                    ledger["files"].append({"path": relative, **identity})
                    uploaded.add(relative)
                ledger["complete"] = verify_checkpoint_ledger(run_dir, ledger)
                atomic_json(ledger_path, ledger)
                return 0 if ledger["complete"] else 1
            if time.monotonic() - last_artifacts >= plan["policy"]["artifact_interval_seconds"]:
                identity = package_and_upload(store, run_dir, workspace, plan)
                pointer = run_dir / "latest_artifacts.json"
                atomic_json(pointer, {"at": now(), "archive": identity})
                # Immutable pointer snapshots let Drive-only recovery find the latest live archive.
                upload_verified(store, pointer, plan["drive_folder_id"], f"live-{hashes(pointer)['sha256']}.json")
                last_artifacts = time.monotonic()
            atomic_json(private / "uploader-heartbeat.json", {"at": now()})
        except Exception as exc:
            # Provider exceptions can contain URLs or secrets; record the type only.
            ledger["errors"].append({"at": now(), "error": type(exc).__name__})
            atomic_json(ledger_path, ledger)
            print(f"Backup needs recovery ({type(exc).__name__})", flush=True)
            return 1
        time.sleep(0.5)


def unpack_dataset(archive, destination):
    import shutil
    if destination.exists():
        raise FileExistsError("Dataset destination exists")
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            if member.filename.startswith("__MACOSX/") or member.filename.endswith("/.DS_Store"):
                continue
            path = Path(member.filename)
            if not path.parts or path.parts[0] != "exported-datasets" or (member.external_attr >> 16) & 0o170000 not in (0, 0o100000, 0o040000):
                raise ValueError("Invalid task-variation ZIP member")
            target = relative_file(destination.parent, member.filename)
            if member.is_dir():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as src, target.open("xb") as out:
                shutil.copyfileobj(src, out)
    if not destination.is_dir():
        raise ValueError("Dataset archive has no exported-datasets directory")


def prepare_inputs(store, workspace, plan, spec):
    code, run_dir = workspace / "code", workspace / "code/runs" / plan["run_name"]
    inputs = {}
    dataset = spec.get("dataset") or {}
    if ((dataset.get("pretrain_data") or {}).get("kind") == "task_variations"
            or dataset.get("finetune_source") == "task_variations_full"):
        identity = plan["drive_config"].get("task_variations")
        if not identity:
            raise ValueError("Configure the Drive task-variation archive")
        archive = code / "assets_tmp/task-variations-v1.zip"
        inputs["dataset"] = store.download(identity["drive_id"], archive, expected=identity, ancestor=ROOT_FOLDER)
        unpack_dataset(archive, code / "assets_tmp/exported-datasets")
    checkpoint = spec.get("checkpoint")
    reuse = plan.get("reuse_parent_inputs")
    if reuse:
        parent = plan["continuation_of"]
        parent_workspace = Path(parent["remote_workspace"])
        parent_plan = json.loads((parent_workspace / "launch.json").read_text())
        if any(parent_plan[key] != parent[key] for key in ("branch", "session", "instance_id")):
            raise ValueError("Cached input owner differs from the continuation parent")
        parent_run = parent_workspace / "code/runs" / parent_plan["run_name"]
        source = relative_file(parent_run, reuse["checkpoint_path"])
        digest = hashes(source)
        if any(digest[key] != checkpoint[key] for key in ("size", "sha256")):
            raise ValueError("On-VM checkpoint differs from its saved identity")
        store.assert_under(checkpoint["drive_id"], store.branch_id)
        metadata = store.metadata(checkpoint["drive_id"])
        store._verify_binary(metadata, metadata["parents"][0], checkpoint["name"], digest)
        destination = run_dir / "inputs/pretrain.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, destination)
        inputs["checkpoint"] = {"id": checkpoint["drive_id"], **digest, "reused_vm_path": str(source)}
        if spec.get("skip_data_build"):
            for entry in reuse["data_files"]:
                source = relative_file(parent_run / "data", entry["path"])
                if hashes(source) != entry["identity"]:
                    raise ValueError("Cached dataset changed before continuation")
                destination = relative_file(run_dir / "data", entry["path"])
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.link(source, destination)
            inputs["data"] = {"reused_vm_path": str(parent_run / "data"), "files": reuse["data_files"]}
            # Immutable family indexes contain repository-relative paths under
            # the original run name. Preserve those namespaces without editing
            # the index or changing its checkpoint identity.
            for alias in reuse.get("data_aliases", []):
                alias_root = relative_file(code, alias)
                if alias_root == run_dir / "data":
                    continue
                for entry in reuse["data_files"]:
                    source = relative_file(run_dir / "data", entry["path"])
                    destination = relative_file(alias_root, entry["path"])
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.link(source, destination)
        print("Reused and verified checkpoint already on this VM; no checkpoint download.", flush=True)
        return inputs
    if isinstance(checkpoint, dict):
        if not store.branch_id:
            raise ValueError("Checkpoint source branch is unavailable")
        inputs["checkpoint"] = store.download(checkpoint["drive_id"], run_dir / "inputs/pretrain.pt",
                                               expected=checkpoint, ancestor=store.branch_id)
    return inputs


def sanitize_path_environment(env, cwd):
    """Remove implicit/explicit cwd entries so GCC cannot load the repo's specs/ directory."""
    cwd = os.path.abspath(cwd)
    removed = {}
    for name in PATH_LIST_ENV_VARS:
        if name not in env:
            continue
        entries = env[name].split(os.pathsep)
        kept = [entry for entry in entries if os.path.abspath(os.path.join(cwd, entry)) != cwd]
        if len(kept) == len(entries):
            continue
        removed[name] = len(entries) - len(kept)
        if kept:
            env[name] = os.pathsep.join(kept)
        else:
            env.pop(name)
    return removed


def clean_training_env(plan, private, *, removed_paths=None):
    env = dict(os.environ)
    for key in list(env):
        if any(word in key for word in ("CONTAINER_API_KEY", "VAST_API_KEY", "RUNPOD_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS", "GMN_TOKEN", "GMAN_TOKEN")):
            env.pop(key, None)
    secrets = private / "training-secrets.json"
    if secrets.exists():
        env.update(json.loads(secrets.read_text()))
    env.update(MDLARC_REMOTE="1", MDLARC_RUN_BRANCH=plan["branch"],
               MDLARC_RUN_DIR=str(private.parent / "code/runs" / plan["run_name"]),
               MDLARC_REPORT_FIRST_UPDATE="1", PYTHONUNBUFFERED="1")
    # Family batches have variable token counts. Grow allocator segments instead
    # of leaving unusable fragments as their activation sizes change.
    if not env.get("PYTORCH_ALLOC_CONF") and not env.get("PYTORCH_CUDA_ALLOC_CONF"):
        env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    if plan.get("backend") in {"runpod", "sfcompute"}:
        # Keep disposable dataset/compiler files off the /workspace network volume.
        env["TMPDIR"] = "/tmp"
        if not plan["policy"].get("validate_model_on_start", False):
            env["MDLARC_SKIP_CUDA_VALIDATION"] = "1"
    if automatic_checkpoint_uploads(plan):
        env["MDLARC_CHECKPOINT_QUEUE"] = str(private / "queue")
    else:
        env.pop("MDLARC_CHECKPOINT_QUEUE", None)
    # SSH into NGC containers does not inherit the entrypoint's CUDA paths.
    # Triton needs cuda.h when building its native driver helper.
    cuda = Path(env.get("CUDA_HOME") or env.get("CUDA_PATH") or "/usr/local/cuda")
    if (cuda / "include/cuda.h").is_file():
        env.setdefault("CUDA_HOME", str(cuda))
        for name, suffix in (("TRITON_PTXAS_PATH", "bin/ptxas"),
                             ("TRITON_CUOBJDUMP_PATH", "bin/cuobjdump"),
                             ("TRITON_NVDISASM_PATH", "bin/nvdisasm"),
                             ("TRITON_LIBDEVICE_PATH", "nvvm/libdevice/libdevice.10.bc")):
            if (cuda / suffix).is_file():
                env.setdefault(name, str(cuda / suffix))
        for name, suffix in (("CPATH", "include"), ("LIBRARY_PATH", "lib64"),
                             ("LD_LIBRARY_PATH", "lib64"), ("PATH", "bin")):
            directory = str(cuda / suffix)
            if (cuda / suffix).is_dir() and directory not in env.get(name, "").split(os.pathsep):
                env[name] = directory + (os.pathsep + env[name] if env.get(name) else "")
    removed = sanitize_path_environment(env, private.parent / "code")
    if removed_paths is not None:
        removed_paths.update(removed)
    return env


def runtime_cache_identity(code, plan, env, selected):
    def package(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).is_file() else None
    def container_start():
        if not Path('/proc/1/stat').is_file():
            return None
        return {"start_ticks": Path('/proc/1/stat').read_text().rsplit(')', 1)[1].split()[19],
                "pid_namespace": Path('/proc/1/ns/pid').stat().st_ino}
    tools = {}
    for key in ('TRITON_PTXAS_PATH', 'TRITON_LIBDEVICE_PATH'):
        path = Path(env[key]) if env.get(key) else None
        if path and path.is_file():
            info = path.stat()
            tools[key] = [str(path.resolve()), info.st_size, info.st_mtime_ns]
    return {"schema_version": 1, "instance_id": plan["instance_id"], "image": plan["policy"]["image"],
            "python": sys.version, "executable": sys.executable, "attention_versions": sorted(selected),
            "num_gpus": plan['policy'].get('num_gpus'),
            "device_visibility": {key: env.get(key) for key in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'NVIDIA_VISIBLE_DEVICES')},
            "packages": {name: package(name) for name in
                ("torch", "triton", "flash-attn", "flash-attn-4", "nvidia-cutlass-dsl", "nvidia-cutlass-dsl-libs-cu13")},
            "container_start": container_start(), "boot": digest("/proc/sys/kernel/random/boot_id"),
            "driver": digest("/proc/driver/nvidia/version"),
            "cuda_preflight": hashlib.sha256(CUDA_PREFLIGHT_CODE.encode()).hexdigest(),
            "attention_preflight": digest(code / "src/attention_runtime.py"),
            "tools": tools,
            "compiler_environment": {key: env.get(key) for key in PATH_LIST_ENV_VARS},
            "tool_paths": {key: env.get(key) for key in ("CUDA_HOME", "TRITON_PTXAS_PATH", "TRITON_LIBDEVICE_PATH")}}


def preflight_runtime(code, run_dir, plan, private):
    from src.attention_runtime import FA4_REQUIREMENT, fa4_installed, selected_versions
    spec = (json.loads(relative_file(code, plan["spec_relative"]).read_text())
            if "spec_relative" in plan else {})
    selected = selected_versions(spec)
    removed = {}
    env = clean_training_env(plan, private, removed_paths=removed)
    if plan.get("skip_native_validation") is True:
        if 4 in selected and not fa4_installed():
            raise RuntimeError("Requested FA4 package is absent; cannot skip native setup")
        atomic_json(run_dir / "runtime_environment.json", {
            "at": now(), "validation_skipped_at_user_request": True,
            "torch_compile_smoke": False,
            "sanitized_current_directory_path_entries": removed})
        print("Skipping repeated native CUDA/FA4 tests at the user's explicit request.", flush=True)
        return
    cache_enabled = plan.get("backend") in {"runpod", "sfcompute"} and plan["policy"].get("cache_runtime_validation", True)
    def cache_location(identity):
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        return Path(env.get("TMPDIR") or tempfile.gettempdir()) / "mdlarc-runtime-cache" / (key + ".json")
    identity = runtime_cache_identity(code, plan, env, selected) if cache_enabled or plan.get('runtime_validated_from') else None
    if cache_enabled and identity["container_start"] and cache_location(identity).is_file():
        try:
            cached = json.loads(cache_location(identity).read_text())
            report, fa4 = cached["report"], cached.get("fa4")
            valid = (cached["identity"] == identity and report.get('runtime_identity') == identity
                     and report.get("torch_compile_smoke") and report.get("bf16")
                     and (4 not in selected or fa4 and fa4.get("rectangular_varlen_forward_backward")))
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            valid = False
        if valid:
            atomic_json(run_dir / "runtime_environment.json", {**report, "runtime_cache_reused": True})
            if fa4:
                (run_dir / "fa4_preflight.log").write_text(json.dumps(fa4) + "\n")
            print("Reusing validated container runtime; no CUDA/FA4 tests rerun.", flush=True)
            return
    previous = plan.get('runtime_validated_from')
    if previous:
        origin = Path(previous['remote_workspace'])
        saved = json.loads((origin / 'launch.json').read_text())
        if (saved['instance_id'] != plan['instance_id'] or saved['policy']['image'] != plan['policy']['image']):
            raise ValueError('Validated runtime belongs to another pod or image')
        records = origin / 'code/runs' / saved['run_name']
        report = json.loads((records / 'runtime_environment.json').read_text())
        fa4 = json.loads((records / 'fa4_preflight.log').read_text().splitlines()[-1])
        if report.get('runtime_identity') is None and plan.get('backend') == 'runpod':
            # Migrate legacy proof only inside its still-running container.
            # Reading versions and process age launches no CUDA validation.
            from datetime import datetime
            import ast
            validated_at = datetime.fromisoformat(report['at']).timestamp()
            uptime = float(Path('/proc/uptime').read_text().split()[0])
            start_ticks = int(identity['container_start']['start_ticks'])
            container_started = time.time() - uptime + start_ticks / os.sysconf('SC_CLK_TCK')
            packages = identity['packages']
            # NGC's wheel metadata adds a build suffix absent from torch.__version__.
            # Compare the installed version module used by the original proof.
            torch_version = Path(importlib.metadata.distribution('torch').locate_file('torch/version.py'))
            torch_values = {}
            for node in ast.parse(torch_version.read_text()).body:
                target = (node.targets[0] if isinstance(node, ast.Assign)
                          else node.target if isinstance(node, ast.AnnAssign) else None)
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                    torch_values[target.id] = node.value.value
            same_runtime = (container_started < validated_at
                and report.get('python') == sys.version
                and report.get('torch') == torch_values.get('__version__')
                and report.get('torch_cuda') == torch_values.get('cuda')
                and fa4.get('version') == packages['flash-attn-4']
                and report.get('cuda_home') == env.get('CUDA_HOME')
                and report.get('triton_ptxas_path') == env.get('TRITON_PTXAS_PATH')
                and hashes(origin / 'code/src/attention_runtime.py')['sha256'] == identity['attention_preflight'])
            evidence_time = min((records / 'runtime_environment.json').stat().st_mtime,
                                (records / 'fa4_preflight.log').stat().st_mtime)
            same_runtime = same_runtime and torch_version.stat().st_mtime <= evidence_time
            for package, version in packages.items():
                if version is not None:
                    metadata = importlib.metadata.distribution(package)
                    same_runtime = same_runtime and all(
                        Path(metadata.locate_file(item)).stat().st_mtime <= evidence_time
                        for item in metadata.files or () if str(item).endswith('.dist-info/METADATA'))
            if same_runtime:
                report = {**report, 'runtime_identity': identity,
                          'legacy_validation_migrated': True, 'container_started_at_unix': container_started}
        if report.get('runtime_identity') != identity:
            raise ValueError('Previous validation has no matching runtime fingerprint; omit --skip-cuda-validation for one native validation')
        if not report.get('torch_compile_smoke') or not report.get('bf16') or not fa4.get('rectangular_varlen_forward_backward'):
            raise ValueError('Previous CUDA/FA4 validation did not pass')
        atomic_json(run_dir / 'runtime_environment.json', {**report, 'reused_from': previous})
        (run_dir / 'fa4_preflight.log').write_text(json.dumps(fa4) + '\n')
        if cache_enabled:
            atomic_json(cache_location(identity), {'identity': identity, 'report': report, 'fa4': fa4})
        print('Reusing passed CUDA/FA4 validation on the same powered-on pod; no tests rerun.', flush=True)
        return
    report_path = run_dir / "runtime_environment.json"
    atomic_json(report_path, {"at": now(), "torch_compile_smoke": False,
                             "sanitized_current_directory_path_entries": removed})
    deadline = min(time.time() + 300, plan["deadline_unix"] - plan["policy"]["backup_grace_seconds"])
    def health():
        if time.time() >= deadline:
            raise RuntimeError("CUDA compilation preflight exhausted its time allowance")
    health()
    print("Checking CUDA, Triton, and torch.compile before preparing inputs...", flush=True)
    run_training([sys.executable, "-u", "-c", CUDA_PREFLIGHT_CODE, str(report_path)], code, env,
                 run_dir / "runtime_preflight.log", health)
    fa4 = None
    if 4 in selected:
        if not fa4_installed():
            run_training(pip_install_command(FA4_REQUIREMENT), code, env, run_dir / "fa4_install.log", health)
        # A fresh process validates the actual CUDA package before dataset setup
        # and optimizer work. Never replace FA4 with another backend on failure.
        run_training([sys.executable, "-u", str(code / "src/attention_runtime.py")],
                     code, env, run_dir / "fa4_preflight.log", health)
        fa4 = json.loads((run_dir / "fa4_preflight.log").read_text().splitlines()[-1])
    if cache_enabled:
        identity = runtime_cache_identity(code, plan, env, selected)
        report = {**json.loads(report_path.read_text()), 'runtime_identity': identity}
        atomic_json(report_path, report)
        cached = {"identity": identity, "report": report, "fa4": fa4}
        atomic_json(cache_location(identity), cached, private=True)


class LogRedactor:
    """Redact known training credentials even when a value spans output chunks."""
    def __init__(self, env):
        values = {value.encode() for key, value in env.items() if value and len(value) >= 4
                  and any(part in key.upper() for part in ("TOKEN", "SECRET", "KEY", "KAGGLE_USERNAME"))}
        self.pattern = re.compile(b"|".join(re.escape(x) for x in sorted(values, key=len, reverse=True))) if values else None
        self.hold = max((len(x) - 1 for x in values), default=0)
        self.pending = b""

    def feed(self, data, *, final=False):
        self.pending += data
        cut = len(self.pending) if final else max(0, len(self.pending) - self.hold)
        if self.pattern:
            for match in self.pattern.finditer(self.pending):
                if match.start() < cut < match.end():
                    cut = match.end()
        output, self.pending = self.pending[:cut], self.pending[cut:]
        return self.pattern.sub(b"[REDACTED]", output) if self.pattern else output


def run_training(command, cwd, env, log_path, health, on_output=None):
    process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True, bufsize=0)
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    redactor = LogRedactor(env)
    try:
        with log_path.open("a", encoding="utf-8") as out:
            while True:
                health()
                ready, _, _ = select.select([process.stdout], [], [], 1)
                if not ready:
                    continue
                data = os.read(process.stdout.fileno(), 65536)
                text = decoder.decode(redactor.feed(data, final=not data), final=not data)
                # Training never receives Drive or provider control credentials.
                out.write(text)
                out.flush()
                print(text, end="", flush=True)
                if on_output:
                    on_output(text)
                if not data:
                    break
        if process.wait():
            raise RuntimeError("Training process failed; consult console.log")
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def cleanup_action(status, policy, plan=None):
    if (plan or {}).get("backend") == "ssh":
        return "keep"  # An external SSH worker never owns the provider rental.
    if policy.get("checkpoint_upload_mode") == "manual":
        return "keep"  # The VM may hold the only copy of requested local checkpoints.
    if keep_b300_running(plan or {"policy": policy}):
        return "keep"
    if status.get("ok") and status.get("backups_complete") and status.get("finalized"):
        return policy["success_action"]
    return policy["failure_action"]


def instance_client(plan):
    if plan.get("backend") == "sfcompute":
        from workflows.sfcompute.api import MetadataClient
        return MetadataClient(plan["instance_id"], plan["label"])
    if plan.get("backend") == "ssh":
        return None
    if plan.get("backend") == "runpod":
        if plan.get("adopted_instance") and plan.get("preserve_instance") is True:
            return None  # This supervisor has no authority to control the rental.
        from workflows.runpod.api import CONTROL_FILE, Runpod
        values = dict(os.environ)
        path = Path(CONTROL_FILE)
        if path.is_file():
            values.update(dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line))
        if not values.get("RUNPOD_API_KEY") or values.get("RUNPOD_POD_ID") != plan["instance_id"]:
            raise RuntimeError("Pod-scoped Runpod control credentials are unavailable or mismatched")
        return Runpod(values["RUNPOD_API_KEY"], own_pod=plan["instance_id"])
    identity, key = os.environ.get("CONTAINER_ID"), os.environ.get("CONTAINER_API_KEY")
    path = Path("/root/.mdlarc-vast-instance.env")
    if (not identity or not key) and path.is_file():
        values = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
        identity, key = values.get("CONTAINER_ID"), values.get("CONTAINER_API_KEY")
    if not key or str(identity) != str(plan["instance_id"]):
        raise RuntimeError("Per-instance Vast control credentials are unavailable or mismatched")
    return Vast(key)


def recover_backups(workspace, plan):
    """Reconcile saved transfers and complete backups; never executes training."""
    with lock(workspace / "private/supervisor.lock"):
        status = json.loads((workspace / "status.json").read_text())
        run_dir = workspace / "code/runs" / plan["run_name"]
        (workspace / "private/uploader-stop").touch()
        if uploader(workspace, plan):
            raise RuntimeError("Checkpoint recovery is incomplete")
        store = store_for(workspace, plan, "supervisor-drive")
        upload_verified(store, workspace / "source.tar.gz", plan["drive_folder_id"], "source.tar.gz")
        upload_verified(store, workspace / "launch.json", plan["drive_folder_id"], "launch.json")
        backup_source_inputs(store, workspace / 'code', run_dir, plan)
        automatic = automatic_checkpoint_uploads(plan)
        status.update(backups_complete=automatic, record_backups_complete=True,
                      checkpoint_upload_mode="automatic" if automatic else "manual",
                      state="complete" if status["ok"] else "failed", finished_at=now())
        atomic_json(run_dir / "run_status.json", status)
        archive = package_and_upload(store, run_dir, workspace, plan, final=True)
        manifest = {**status, "finalized": True, "archive": archive, "drive_folder_id": plan["drive_folder_id"]}
        # If a verified failure manifest exists, recovery creates a separate immutable manifest.
        atomic_json(workspace / "completion.json", manifest)
        name = "completion-recovered-" + hashes(workspace / "completion.json")["sha256"] + ".json"
        store.upload(workspace / "completion.json", plan["drive_folder_id"], name)
        atomic_json(workspace / "status.json", manifest)
        action = cleanup_action(manifest, plan["policy"], plan)
        if action != "keep":
            getattr(instance_client(plan), action)(plan["instance_id"])
        return 0


def deadline_guard(workspace, plan):
    with lock(workspace / "private/deadline.lock"):
        if plan.get("backend") == "ssh":
            atomic_json(workspace / "deadline-status.json", {"at": now(), "state": "keep_external_instance"})
            return 0
        if keep_b300_running(plan):
            atomic_json(workspace / "deadline-status.json", {"at": now(), "state": "keep_b300_running"})
            return 0
        client = instance_client(plan)
        while time.time() < plan["deadline_unix"]:
            time.sleep(min(10, max(0.1, plan["deadline_unix"] - time.time())))
        atomic_json(workspace / "deadline-status.json", {"at": now(), "state": "stop_requested"})
        for _ in range(3):
            try:
                client.stop(plan["instance_id"])
                return 0
            except Exception:
                time.sleep(5)
        atomic_json(workspace / "deadline-status.json", {"at": now(), "state": "stop_unconfirmed"})
        return 1


def supervise(workspace, plan):
    private, code = workspace / "private", workspace / "code"
    run_dir = code / "runs" / plan["run_name"]
    automatic = automatic_checkpoint_uploads(plan)
    with lock(private / "supervisor.lock"):
        started = private / "started.json"
        if started.exists():
            print("This run was already submitted; observe or recover it instead of resubmitting.", flush=True)
            return 0
        atomic_json(started, {"at": now(), "pid": os.getpid()}, private=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        status = {"session": plan["session"], "branch": plan["branch"], "instance_id": plan["instance_id"],
                  "ok": False, "backups_complete": False, "record_backups_complete": False,
                  "checkpoint_upload_mode": "automatic" if automatic else "manual",
                  "finalized": False, "state": "starting", "started_at": now()}
        atomic_json(workspace / "status.json", status)
        startup = {"launch_requested_at_unix": plan.get('launch_requested_at_unix', plan['rented_at_unix']), "stages": {}}
        def mark_startup(stage, state=None):
            if stage in startup['stages']:
                return
            startup['stages'][stage] = {"at": now(), "elapsed_seconds": round(time.time() - startup['launch_requested_at_unix'], 3)}
            atomic_json(run_dir / 'startup_timing.json', startup)
            if state:
                status['state'] = state
                atomic_json(workspace / 'status.json', status)
        mark_startup('supervisor_started')
        uploader_process = uploader_log = store = None
        provenance_verified = False
        try:
            client = instance_client(plan)
            if not keep_b300_running(plan):
                with (workspace / "deadline.log").open("ab") as log:
                    subprocess.Popen([sys.executable, "-u", __file__, "--workspace", str(workspace), "--deadline-only"],
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            if time.time() >= plan["deadline_unix"] - plan["policy"]["backup_grace_seconds"]:
                raise RuntimeError("Runtime allowance is exhausted")
            if hashes(workspace / "source.tar.gz")["sha256"] != plan["source"]["sha256"]:
                raise ValueError("Uploaded source verification failed")
            spec_path = relative_file(code, plan["spec_relative"])
            if hashes(spec_path)["sha256"] != plan["spec_sha256"]:
                raise ValueError("Uploaded experiment spec verification failed")
            spec = json.loads(spec_path.read_text())
            ensure_drive_dependencies(code, run_dir, plan, private)
            store = store_for(workspace, plan, "supervisor-drive")
            upload_verified(store, workspace / "source.tar.gz", plan["drive_folder_id"], "source.tar.gz")
            upload_verified(store, workspace / "launch.json", plan["drive_folder_id"], "launch.json")
            source_input_backups = backup_source_inputs(store, code, run_dir, plan)
            provenance_verified = True
            mark_startup('provenance_backups_ready')
            status["state"] = "preflighting_runtime"
            atomic_json(workspace / "status.json", status)
            preflight_runtime(code, run_dir, plan, private)
            mark_startup('runtime_ready')
            status["state"] = "preparing_inputs"
            atomic_json(workspace / "status.json", status)
            inputs = prepare_inputs(store, workspace, plan, spec)
            if source_input_backups:
                inputs['source_inputs'] = source_input_backups
            details = {"backend": plan.get("backend", "vast"), "launch": plan, "spec": spec, "inputs": inputs, "started_at": status["started_at"],
                       "python": sys.version}
            # Existing image packages stay in place; FA4 is installed only when
            # explicitly selected and its pinned version is absent.
            (run_dir / "experiment.json").write_text(json.dumps(details, indent=2))
            upload_verified(store, run_dir / "experiment.json", plan["drive_folder_id"], "experiment.json")
            uploader_log = (run_dir / "drive_upload.log").open("ab", buffering=0)
            uploader_process = subprocess.Popen([sys.executable, "-u", __file__, "--workspace", str(workspace), "--uploader"],
                stdin=subprocess.DEVNULL, stdout=uploader_log, stderr=subprocess.STDOUT)
            ready_deadline = min(time.time() + 90, plan["deadline_unix"])
            while not (private / "uploader-ready.json").exists():
                if uploader_process.poll() is not None or time.time() > ready_deadline:
                    raise RuntimeError("Backup worker did not start; training was not started")
                time.sleep(0.2)
            progress_supported = all('MDLARC_STARTUP_EVENT' in (code / path).read_text()
                                     for path in ('launch_run.py', 'src/train.py'))
            status["state"] = "preparing_data" if progress_supported else "running"
            atomic_json(workspace / "status.json", status)
            progress_tail = ''
            def progress(text):
                nonlocal progress_tail
                progress_tail = (progress_tail + text)[-4096:]
                for match in re.finditer(r'MDLARC_STARTUP_EVENT (preparing_data|data_ready|model_setup|compiling_training|first_optimizer_step)\b', progress_tail):
                    stage = match.group(1)
                    mark_startup(stage, 'training' if stage == 'first_optimizer_step' else stage)
            def health():
                if uploader_process.poll() is not None:
                    raise RuntimeError("Backup worker stopped; training stopped for recovery")
                if time.time() >= plan["deadline_unix"] - plan["policy"]["backup_grace_seconds"]:
                    raise RuntimeError("Training time allowance reached; finalizing saved outputs")
            run_training([sys.executable, "-u", "launch_run.py", "--config", plan["spec_relative"]], code,
                         clean_training_env(plan, private), run_dir / "console.log", health, on_output=progress)
            status["ok"] = True
        except BaseException as exc:
            status["error"] = type(exc).__name__
            (run_dir / "failure.txt").write_text(f"Run failed ({type(exc).__name__}). Consult console.log and drive_upload.log.\n")
            print(f"Run failed ({type(exc).__name__}); finalizing available outputs", flush=True)
        finally:
            status["state"] = "finalizing"
            atomic_json(workspace / "status.json", status)
            if uploader_process:
                (private / "uploader-stop").touch()
                try:
                    allowance = max(1, min(plan["policy"]["backup_grace_seconds"], plan["deadline_unix"] - time.time() - 30))
                    result = uploader_process.wait(timeout=allowance)
                    ledger = json.loads((run_dir / "checkpoint_backups.json").read_text())
                    status["record_backups_complete"] = provenance_verified and result == 0
                    status["backups_complete"] = provenance_verified and result == 0 and ledger["complete"] and verify_checkpoint_ledger(run_dir, ledger)
                except Exception:
                    if uploader_process.poll() is None:
                        uploader_process.terminate()
                        try:
                            uploader_process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            uploader_process.kill()
                            uploader_process.wait()
                uploader_log.close()
            elif store:
                status["record_backups_complete"] = provenance_verified
                status["backups_complete"] = provenance_verified and not checkpoint_paths(run_dir)
            status["finished_at"] = now()
            required_backups = status["backups_complete"] if automatic else status["record_backups_complete"]
            status["state"] = "complete" if status["ok"] and required_backups else "needs_recovery"
            atomic_json(run_dir / "run_status.json", status)
            if store:
                try:
                    archive = package_and_upload(store, run_dir, workspace, plan, final=True)
                    manifest = {**status, "finalized": True, "archive": archive, "drive_folder_id": plan["drive_folder_id"]}
                    atomic_json(workspace / "completion.json", manifest)
                    upload_verified(store, workspace / "completion.json", plan["drive_folder_id"], "completion.json")
                    status["finalized"] = True
                    status["archive"] = archive
                except Exception as exc:
                    status.update(backups_complete=False, record_backups_complete=False,
                                  state="needs_recovery", backup_error=type(exc).__name__)
                    (workspace / "completion.json").unlink(missing_ok=True)
            atomic_json(workspace / "status.json", status)
            action = cleanup_action(status, plan["policy"], plan)
            atomic_json(workspace / "cleanup.json", {"at": now(), "action": action, "state": "requested"})
            if action != "keep":
                try:
                    client = instance_client(plan)
                    getattr(client, action)(plan["instance_id"])
                except Exception:
                    atomic_json(workspace / "cleanup.json", {"at": now(), "action": action, "state": "unconfirmed"})
                    print("Instance cleanup needs reconciliation; deadline guard remains active", flush=True)
        required_backups = status["backups_complete"] if automatic else status["record_backups_complete"]
        return 0 if status["ok"] and required_backups and status["finalized"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--uploader", action="store_true")
    mode.add_argument("--deadline-only", action="store_true")
    mode.add_argument("--recover-only", action="store_true")
    args = parser.parse_args(argv)
    workspace = args.workspace.resolve()
    plan = json.loads((workspace / "launch.json").read_text())
    if args.uploader:
        return uploader(workspace, plan)
    if args.deadline_only:
        return deadline_guard(workspace, plan)
    if args.recover_only:
        return recover_backups(workspace, plan)
    return supervise(workspace, plan)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Remote worker failed ({type(exc).__name__}); inspect the saved run state.", flush=True)
        raise SystemExit(1) from None
