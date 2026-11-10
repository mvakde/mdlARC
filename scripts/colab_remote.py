"""Code submitted once to the preserved Colab kernel to run a packaged working tree."""

import hashlib
import codecs
import json
import os
import platform
import signal
import select
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
import zipfile
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path("/content/mdlARC")
ARCHIVE = Path("/content/mdlARC-source.tar.gz")
RESULT_ARCHIVE = Path("/content/mdlARC-results.tar.gz")
MANIFEST = Path("/content/mdlARC-results-manifest.json")
CHUNK_SIZE = 16 * 1024 * 1024
DRIVE_TOKEN = Path("/content/.mdlarc-drive-token.json")
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}
# Training overwrites tiny.rolling.pt in place for local resumes; it is never backed up.
ROLLING_CHECKPOINT_MARKER = ".rolling"


def use_drive():
    return os.environ.get("MDLARC_DRIVE", "0") == "1"


def use_shared_account():
    return use_drive() and os.environ.get("MDLARC_DRIVE_AUTH") == "shared"


def configure_cuda_allocator():
    """Limit fragmentation across varying inference shapes; preserve overrides."""
    if not os.environ.get("PYTORCH_ALLOC_CONF") and not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


def is_small_result(path):
    return path.suffix not in CHECKPOINT_SUFFIXES


def flash_attn_wheel(torch):
    manifest_path = ROOT / "wheels" / "environment.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    wheel_name = manifest["wheel"]
    if not isinstance(wheel_name, str) or Path(wheel_name).name != wheel_name or not wheel_name.endswith(".whl"):
        raise ValueError(f"Invalid flash-attn wheel name: {wheel_name!r}")
    wheel = manifest_path.parent / wheel_name
    digest = hashlib.sha256()
    with wheel.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != manifest["sha256"]:
        raise RuntimeError("Flash-attn wheel SHA-256 does not match its manifest")
    required_python = tuple(int(part) for part in manifest["python"].split()[0].split(".")[:2])
    actual_abi = bool(torch._C._GLIBCXX_USE_CXX11_ABI)
    required_capability = tuple(manifest["compute_capability"])
    actual_capability = torch.cuda.get_device_capability(0)
    mismatches = []
    if sys.version_info[:2] != required_python:
        mismatches.append(f"Python {sys.version_info.major}.{sys.version_info.minor} != {required_python[0]}.{required_python[1]}")
    if torch.__version__ != manifest["torch"]:
        mismatches.append(f"PyTorch {torch.__version__} != {manifest['torch']}")
    if torch.version.cuda != manifest["torch_cuda"]:
        mismatches.append(f"PyTorch CUDA {torch.version.cuda} != {manifest['torch_cuda']}")
    if actual_abi != manifest["torch_cxx11_abi"]:
        mismatches.append(f"PyTorch C++ ABI {actual_abi} != {manifest['torch_cxx11_abi']}")
    if actual_capability != required_capability:
        mismatches.append(f"GPU compute capability {actual_capability} != {required_capability}")
    if sys.platform != "linux" or platform.machine() != "x86_64":
        mismatches.append(f"platform {sys.platform}/{platform.machine()} != linux/x86_64")
    if mismatches:
        raise RuntimeError("Bundled flash-attn wheel is incompatible with this runtime: " + "; ".join(mismatches))
    return wheel


def run(args, *, cwd=None, log_path=None, health_check=None):
    print("Running:", " ".join(args), flush=True)
    process = subprocess.Popen(
        args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=0, start_new_session=True,
    )
    assert process.stdout is not None
    log = log_path.open("a", encoding="utf-8") if log_path else None
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    try:
        while True:
            if health_check:
                health_check()
            readable, _, _ = select.select([process.stdout], [], [], 1)
            if not readable:
                continue
            chunk = os.read(process.stdout.fileno(), 64 * 1024)
            output = decoder.decode(chunk, final=not chunk)
            print(output, end="", flush=True)
            if log:
                log.write(output)
                log.flush()
            if not chunk:
                break
        returncode = process.wait()
    finally:
        if log:
            log.close()
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, args)


def unpack_source():
    ROOT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        for member in archive:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or not member.isfile():
                raise ValueError(f"Unsafe source member: {member.name}")
            target = ROOT / path
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Unreadable source member: {member.name}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def unpack_task_variations(archive_path: Path, destination: Path) -> int:
    """Unpack the reviewed ZIP into a new directory, ignoring macOS metadata."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Dataset destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".task-variations-", dir=destination.parent) as temp_name:
        staging = Path(temp_name)
        count = 0
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                name = member.filename
                if name.startswith("__MACOSX/") or name.endswith("/.DS_Store"):
                    continue
                path = Path(name)
                if (path.is_absolute() or ".." in path.parts or "\\" in name
                        or not path.parts or path.parts[0] != "exported-datasets"):
                    raise ValueError(f"Unsafe dataset ZIP member: {name}")
                if member.is_dir():
                    continue
                mode = (member.external_attr >> 16) & 0o170000
                if mode not in (0, 0o100000):
                    raise ValueError(f"Unsupported dataset ZIP member: {name}")
                target = staging / path
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                count += 1
        extracted = staging / "exported-datasets"
        if count == 0 or not extracted.is_dir():
            raise ValueError("Dataset ZIP contains no exported-datasets files")
        extracted.rename(destination)
    return count


def publish_results_manifest(manifest):
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest), encoding="utf-8")
    temporary.replace(MANIFEST)


def package_results(run_name, status, *, publish=True):
    status_path = Path("/content/mdlARC-run-status.json")
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    with tarfile.open(RESULT_ARCHIVE, "w:gz") as archive:
        archive.add(status_path, arcname="run_status.json")
        run_dir = ROOT / "runs" / run_name
        if run_dir.exists():
            for path in sorted(run_dir.rglob("*")):
                if path.is_file() and not path.is_symlink() and (not use_drive() or is_small_result(path)):
                    archive.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
    parts = []
    whole = hashlib.sha256()
    with RESULT_ARCHIVE.open("rb") as source:
        for index, data in enumerate(iter(lambda: source.read(CHUNK_SIZE), b"")):
            name = f"mdlARC-results.part{index:04d}"
            Path("/content", name).write_bytes(data)
            whole.update(data)
            parts.append({"name": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {"parts": parts, "sha256": whole.hexdigest(), "run_name": run_name,
                "session": os.environ.get("MDLARC_SESSION"), "finalized": True}
    if publish:
        publish_results_manifest(manifest)
    print(f"Packaged {len(parts)} result chunk(s).", flush=True)
    return manifest


def main():
    configure_cuda_allocator()
    # Publishing local recovery records uses this same persistent kernel.
    # Its one-shot mode must never leak into a subsequent managed run.
    os.environ.pop("MDLARC_PUBLISH_RECORDS", None)
    os.environ.pop("MDLARC_RECORDS_FOLDER", None)
    os.environ["MDLARC_REMOTE"] = "1"
    spec = os.environ["MDLARC_SPEC"]
    run_name = os.environ["MDLARC_RUN_NAME"]
    status = {"ok": False, "error": None, "backups_complete": True, "started_at": datetime.now(UTC).isoformat()}
    worker_process = worker_log = store = folder = dataset_info = input_checkpoint = None
    evaluation_inputs = []
    run_dir = ROOT / "runs" / run_name
    control = Path("/content/.mdlarc-drive-control")
    queue, stop = control / "queue", control / "stop"
    try:
        unpack_source()
        # A new managed run may reuse a terminal VM/kernel. Invalidate only
        # modules from its preserved source, so imports resolve the new upload.
        for module_name, module in list(sys.modules.items()):
            filename = getattr(module, "__file__", None)
            if isinstance(filename, str) and filename.startswith(str(ROOT) + "/"):
                sys.modules.pop(module_name, None)
        import importlib
        importlib.invalidate_caches()
        selected_spec = json.loads((ROOT / spec).read_text())
        sys.path.insert(0, str(ROOT))
        from launch_run import resolve_input_checkpoint, resolve_evaluation_checkpoints
        input_path, drive_checkpoint = resolve_input_checkpoint(selected_spec.get("checkpoint"), run_dir)
        evaluation_checkpoints = resolve_evaluation_checkpoints(selected_spec, run_dir)
        if evaluation_checkpoints and not use_shared_account():
            raise ValueError("Named evaluation inputs require shared-account Drive staging")
        if drive_checkpoint and (not use_shared_account() or os.environ.get("MDLARC_PRETRAIN_DRIVE_ID")
                                 or os.environ.get("MDLARC_FINETUNE_TARGET") or os.environ.get("MDLARC_EVAL_TARGET")):
            raise ValueError("A Drive checkpoint spec requires the shared VM account and no finetune-only CLI flags")
        run_dir.mkdir(parents=True, exist_ok=True)
        if use_drive():
            if not use_shared_account():
                DRIVE_TOKEN.chmod(0o600)
            run([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "scripts/requirements_colab_drive.txt")])
            sys.path.insert(0, str(ROOT / "scripts"))
            from colab_drive import DriveStore, SharedDriveStore, build_service, build_shared_service, load_config
            drive_config = ROOT / "scripts/colab_drive.json"
            config = load_config(drive_config)
            store = (SharedDriveStore(config, build_shared_service()) if use_shared_account()
                     else DriveStore(config, build_service(DRIVE_TOKEN)))
            task_variations = os.environ.get("MDLARC_TASK_VARIATIONS", "0") == "1"
            store.preflight(include_dataset=task_variations)
            if task_variations:
                dataset_zip = ROOT / "assets_tmp/task-variations-v1.zip"
                dataset_info = store.download_dataset_zip(dataset_zip)
                dataset_info["path"] = "assets_tmp/exported-datasets"
                dataset_info["files"] = unpack_task_variations(
                    dataset_zip, ROOT / dataset_info["path"])
                print(f"Verified and extracted {dataset_info['files']} task-variation files.", flush=True)
            checkpoint_id = os.environ.get("MDLARC_PRETRAIN_DRIVE_ID", "")
            checkpoint_mount = os.environ.get("MDLARC_CHECKPOINT_MOUNT", "")
            checkpoint_transfer = {"mounted_runs_root": Path(checkpoint_mount)} if checkpoint_mount else {}
            if drive_checkpoint or checkpoint_id:
                if not use_shared_account():
                    raise ValueError("Drive checkpoint staging requires the dedicated VM account")
                identity = drive_checkpoint or {
                    "drive_id": checkpoint_id, "name": os.environ["MDLARC_PRETRAIN_NAME"],
                    "size": int(os.environ["MDLARC_PRETRAIN_SIZE"]), "sha256": os.environ["MDLARC_PRETRAIN_SHA256"],
                }
                if not drive_checkpoint:
                    input_path = run_dir / "pretrain/tiny.pt"
                input_checkpoint = store.download_input_checkpoint(
                    identity["drive_id"], input_path,
                    expected_name=identity["name"], expected_size=identity["size"],
                    expected_sha256=None if identity["sha256"] == "drive" else identity["sha256"],
                    **({"source_branch": identity["source_branch"]} if "source_branch" in identity else {}),
                    **checkpoint_transfer,
                )
                input_checkpoint["staged_path"] = input_path.relative_to(run_dir).as_posix()
                print("Input pretrain checkpoint verified directly from Drive.", flush=True)
            folder = store.create_run_folder(os.environ["MDLARC_SESSION"])
            for name, identity in evaluation_checkpoints.items():
                path = run_dir / "inputs" / f"{name}.pt"
                record = store.download_input_checkpoint(identity["drive_id"], path,
                    expected_name=identity["name"], expected_size=identity["size"],
                    expected_sha256=None if identity["sha256"] == "drive" else identity["sha256"],
                    **({"source_branch": identity["source_branch"]} if "source_branch" in identity else {}),
                    **checkpoint_transfer)
                record["staged_path"] = path.relative_to(run_dir).as_posix()
                evaluation_inputs.append(record)
                print(f"Evaluation input {name} verified directly from Drive.", flush=True)
            status["drive_folder_id"] = folder
            status["drive_folder_url"] = f"https://drive.google.com/drive/folders/{folder}"
            sys.path.insert(0, str(ROOT))
            from src.attention_runtime import selected_versions
            if selected_versions(json.loads((ROOT / spec).read_text())) - {4}:
                wheel_manifest = ROOT / "wheels/environment.json"
                wheel_name = json.loads(wheel_manifest.read_text())["wheel"]
                if Path(wheel_name).name != wheel_name or not wheel_name.endswith(".whl"):
                    raise ValueError("Invalid wheel filename")
                store.download_wheel(wheel_manifest.parent / wheel_name, wheel_manifest)
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Colab did not provide a CUDA GPU")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name}; VRAM: {gpu_memory_gb:.1f} GiB; PyTorch: {torch.__version__}", flush=True)
        minimum_memory_gb = float(os.environ.get("MDLARC_MIN_GPU_MEMORY_GB", "0"))
        if gpu_memory_gb < minimum_memory_gb:
            raise RuntimeError(
                f"GPU has {gpu_memory_gb:.1f} GiB VRAM; "
                f"requested at least {minimum_memory_gb:g} GiB"
            )
        sys.path.insert(0, str(ROOT))
        from src.attention_runtime import FA4_REQUIREMENT, fa4_installed, selected_versions
        versions = selected_versions(json.loads((ROOT / spec).read_text()))
        if versions - {4}:
            wheel = flash_attn_wheel(torch)
            run([sys.executable, "-m", "pip", "install", "einops"])
            run([sys.executable, "-m", "pip", "install", "--no-deps", str(wheel)])
            run([sys.executable, "-c", "import flash_attn, flash_attn_2_cuda; print('FlashAttention', flash_attn.__version__)"])
        if 4 in versions:
            if not fa4_installed():
                run([sys.executable, "-m", "pip", "install", FA4_REQUIREMENT])
            run([sys.executable, "-u", str(ROOT / "src/attention_runtime.py")])
        details = {"branch": os.environ.get("MDLARC_RUN_BRANCH"),
                   "spec": json.loads((ROOT / spec).read_text()), "gpu": gpu_name,
                   "gpu_memory_gib": gpu_memory_gb, "python": sys.version,
                   "torch": torch.__version__, "cuda": torch.version.cuda,
                   "cuda_allocator_config": os.environ.get("PYTORCH_ALLOC_CONF") or os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                   "source_sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
                   "started_at": status["started_at"]}
        if store:
            status["backups_complete"] = False
            details["drive_run_parent_id"] = store.run_parent_id
            details["source_backup"] = store.upload_new(ARCHIVE, folder, "source.tar.gz")
            from scripts.run_records import pack_records, publish_bundle
            launch_records = run_dir / 'launch_records.tar.gz'
            pack_records(ROOT, launch_records, os.environ["MDLARC_SESSION"])
            details["launch_records_backup"] = publish_bundle(store, folder, launch_records)
            if dataset_info:
                details["task_variations"] = dataset_info
            if input_checkpoint:
                details["input_checkpoint"] = input_checkpoint
            if evaluation_inputs:
                details["evaluation_inputs"] = evaluation_inputs
            queue.mkdir(parents=True, exist_ok=True)
            os.environ["MDLARC_CHECKPOINT_QUEUE"] = str(queue)
            os.environ["MDLARC_RUN_DIR"] = str(run_dir)
            worker_log = (run_dir / "drive_upload.log").open("w", encoding="utf-8")
            worker_process = subprocess.Popen([
                sys.executable, "-u", str(ROOT / "scripts/colab_drive.py"),
                "--config", str(drive_config),
                *(["--shared-account"] if use_shared_account() else ["--credentials", str(DRIVE_TOKEN)]),
                "--folder", folder, "--run-dir", str(run_dir),
                "--queue", str(queue), "--stop", str(stop),
            ], stdout=worker_log, stderr=subprocess.STDOUT)
            status["backups_complete"] = False
            deadline = time.monotonic() + 90
            while not (run_dir / "checkpoint_backups.json").exists():
                if worker_process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("Drive backup process could not start; training was not started")
                time.sleep(0.1)
        (run_dir / "experiment.json").write_text(json.dumps(details, indent=2))
        if store:
            store.upload_new(run_dir / "experiment.json", folder, "experiment.json")
        def check_backup_worker():
            if worker_process and worker_process.poll() is not None:
                raise RuntimeError("Drive backup process exited unexpectedly; training stopped for recovery")
        finetune_target = os.environ.get("MDLARC_FINETUNE_TARGET", "")
        if finetune_target:
            if not input_checkpoint:
                raise ValueError("Finetune-only runs require a verified Drive input checkpoint")
            if details["spec"].get("skip_data_build", False):
                if not (run_dir / "data").is_dir():
                    raise ValueError("Finetune data is missing; use a spec that builds data on the VM")
            else:
                run([sys.executable, "-u", "launch_run.py", "--config", spec, "--prepare-data"],
                    cwd=ROOT, log_path=run_dir / "console.log", health_check=check_backup_worker)
            run([sys.executable, "-u", "launch_run.py", "--config", spec,
                 "--train-stage", f"finetune:{finetune_target}"],
                cwd=ROOT, log_path=run_dir / "console.log", health_check=check_backup_worker)
            eval_target = os.environ.get("MDLARC_EVAL_TARGET", "")
            if eval_target:
                run([sys.executable, "-u", "launch_run.py", "--config", spec,
                     "--eval-stage", eval_target],
                    cwd=ROOT, log_path=run_dir / "console.log", health_check=check_backup_worker)
        else:
            run([sys.executable, "-u", "launch_run.py", "--config", spec], cwd=ROOT,
                log_path=run_dir / "console.log", health_check=check_backup_worker)
        status["ok"] = True
    except BaseException as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        if run_dir.exists():
            (run_dir / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        if worker_process:
            stop.touch()
            try:
                # The worker drains all published checkpoints before the GPU is released.
                returncode = worker_process.wait(timeout=1800)
                ledger = json.loads((run_dir / "checkpoint_backups.json").read_text())
                status["backups_complete"] = returncode == 0 and ledger["complete"]
                expected = {path.relative_to(run_dir).as_posix() for path in run_dir.rglob("*.pt")
                            if not path.is_symlink() and not path.stem.endswith(ROLLING_CHECKPOINT_MARKER)}
                if input_checkpoint:
                    expected.discard(input_checkpoint["staged_path"])
                for record in evaluation_inputs:
                    expected.discard(record["staged_path"])
                uploaded = {item["path"] for item in ledger["files"] if not item.get("local_deleted")}
                status["backups_complete"] = status["backups_complete"] and expected == uploaded
            except Exception:
                status["backups_complete"] = False
                if worker_process.poll() is None:
                    worker_process.terminate()
                    try:
                        worker_process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        worker_process.kill()
                        worker_process.wait()
            finally:
                worker_log.close()
            if not status["backups_complete"]:
                status["ok"] = False
                status["backup_error"] = "Checkpoint backups are incomplete; keep the runtime for recovery"
                if status["error"] is None:
                    status["error"] = status["backup_error"]
        os.environ.pop("MDLARC_CHECKPOINT_QUEUE", None)
        os.environ.pop("MDLARC_RUN_DIR", None)
        if use_drive() and not use_shared_account():
            DRIVE_TOKEN.unlink(missing_ok=True)
        status["finished_at"] = datetime.now(UTC).isoformat()
        manifest = package_results(run_name, status, publish=False)
        if store and folder:
            try:
                store.upload_new(RESULT_ARCHIVE, folder, "results-without-checkpoints.tar.gz")
            except Exception:
                # The small summary is still downloaded locally. Do not discard
                # remote files after an unconfirmed Drive transfer.
                status["backups_complete"] = False
                status["ok"] = False
                status["backup_error"] = "Drive result-summary upload did not verify; recovery required"
                status["error"] = status["error"] or status["backup_error"]
                manifest = package_results(run_name, status, publish=False)
        # This is the completion signal for file-based monitors. Publish only
        # after the last Drive upload has either verified or recorded failure.
        publish_results_manifest(manifest)


if __name__ == "__main__":
    main()
