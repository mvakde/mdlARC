#!/usr/bin/env python3
"""Run a local mdlARC snapshot on a Colab GPU with the official Colab CLI."""

import argparse
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

if __package__:
    from . import colab_accounts, colab_connection, colab_run_docs, materialize_sweep_spec, run_records
else:
    import colab_accounts
    import colab_connection
    import colab_run_docs
    import materialize_sweep_spec
    import run_records


if __package__:
    from .run_layout import branch_name, results_root, runs_root
else:
    from run_layout import branch_name, results_root, runs_root


ROOT = Path(__file__).resolve().parents[1]
REMOTE_SOURCE = "/content/mdlARC-source.tar.gz"
REMOTE_MANIFEST = "/content/mdlARC-results-manifest.json"
WATCH_INTERVAL = 30
SOURCE_DIRS = ("src", "dataset_building_scripts", "assets")
GROUPED_DIRS = ("ARC-1", "ARC-2", "ConceptARC")
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git", "venv", "runs", "colab_runs"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".swp"}


def flash_attn_bundle(*, drive: bool = False) -> tuple[Path, ...]:
    manifest_path = ROOT / "wheels" / "environment.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    wheel_name = manifest["wheel"]
    if not isinstance(wheel_name, str) or Path(wheel_name).name != wheel_name or not wheel_name.endswith(".whl"):
        raise ValueError(f"Invalid flash-attn wheel name: {wheel_name!r}")
    wheel_path = manifest_path.parent / wheel_name
    expected_hash = manifest["sha256"]
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError("Invalid flash-attn wheel SHA-256 in manifest")
    if drive:
        return (manifest_path,)
    digest = hashlib.sha256()
    with wheel_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_hash = digest.hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(f"Flash-attn wheel SHA-256 mismatch: {wheel_path}")
    return manifest_path, wheel_path


def source_files(spec_path: Path, *, drive: bool = False, session: str | None = None) -> list[Path]:
    files = [ROOT / "launch_run.py", ROOT / "requirements.txt", ROOT / "scripts/run_layout.py"]
    from src.attention_runtime import selected_versions
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if "checkpoint_adaptation" in spec:
        from src.checkpoint_adaptation import validate_config, verified_path
        adaptation = validate_config(spec["checkpoint_adaptation"])
        files.extend(verified_path(ROOT, identity) for identity in adaptation["inputs"].values())
    prepared = (spec.get("dataset") or {}).get("prepared")
    if prepared:
        from dataset_building_scripts.intermediate_grids import validate_identity, sha
        validate_identity(prepared)
        package = ROOT / prepared["archive_path"]
        if package.is_symlink() or package.stat().st_size != prepared["size"] or sha(package.read_bytes()) != prepared["sha256"]:
            raise ValueError("Prepared dataset identity changed")
        files.append(package)
    if selected_versions(spec) - {4}:
        files.extend(flash_attn_bundle(drive=drive))
    if (ROOT / "requirements-fa4.txt").is_file():
        files.append(ROOT / "requirements-fa4.txt")
    if drive:
        files.extend(ROOT / "scripts" / name for name in (
            "colab_drive.py", "colab_drive.json", "requirements_colab_drive.txt",
            "colab_remote.py", "run_on_colab.py", "colab_run_docs.py",
            "materialize_sweep_spec.py",
            "colab_connection.py", "colab_accounts.py", "run_records.py",
            "colab_reuse.py",
        ))
    for name in SOURCE_DIRS:
        base = ROOT / name
        if not base.exists():
            continue
        for directory, subdirs, filenames in os.walk(base):
            subdirs[:] = sorted(d for d in subdirs if d not in SKIP_DIRS and not d.startswith("."))
            for filename in sorted(filenames):
                if filename.startswith(".") or Path(filename).suffix in SKIP_SUFFIXES:
                    continue
                files.append(Path(directory) / filename)
    for name in GROUPED_DIRS:
        files.extend(sorted((ROOT / "assets_tmp" / name).glob("*.json")))
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    for cfg in [spec.get("eval_defaults", {}), *spec.get("evals", {}).values()]:
        reference = cfg.get("reference_submission")
        if reference:
            path = Path(reference)
            if path.is_absolute() or ".." in path.parts or path.suffix != ".json":
                raise ValueError("reference_submission must be a repository-relative JSON file")
            files.append(ROOT / path)
    split_path = ((spec.get("dataset") or {}).get("evaluation") or {}).get("split_manifest")
    family_data = ((spec.get("dataset") or {}).get("pretrain_data") or {})
    if family_data.get("kind") == "family_sampled":
        from dataset_building_scripts.family_data import regular, file_sha
        identity = family_data.get("catalog")
        if not identity:
            raise ValueError("Prepare and pin a complete family catalog before snapshotting")
        package = regular(ROOT, identity["archive_path"])
        if package.stat().st_size != identity["size"] or file_sha(package) != identity["sha256"]:
            raise ValueError("Family catalog package identity changed")
        files.append(package)
    if split_path is not None:
        if (not isinstance(split_path, str) or not split_path or Path(split_path).is_absolute()
                or ".." in Path(split_path).parts or "\\" in split_path):
            raise ValueError("split_manifest must be a repository-relative file path")
        files.append(ROOT / split_path)
    files.append(spec_path)
    if session:
        base = results_root(ROOT)
        files.extend(path for path in (
            base / "experiments" / "notes" / f"{session}.md",
            base / "colab_runs" / "_active" / f"{session}.json",
        ) if path.is_file())
    unique = sorted(set(files))
    for path in unique:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Snapshot contains a missing or linked file: {path}")
        path.relative_to(ROOT)
    return unique


def create_snapshot(spec_path: Path, archive_path: Path, *, drive: bool = False, session: str | None = None) -> tuple[int, int]:
    files = source_files(spec_path, drive=drive, session=session)
    with tarfile.open(archive_path, "w:gz", compresslevel=6) as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
    return len(files), archive_path.stat().st_size


def preserve_launch_spec(session: str, archive_path: Path, relative_spec: Path,
                         expected_sha256: str,
                         trial: materialize_sweep_spec.MaterializedSpec | None = None) -> Path:
    """Save the selected spec bytes from the archive that will be uploaded."""
    with tarfile.open(archive_path, "r:gz") as archive:
        member = archive.extractfile(relative_spec.as_posix())
        if member is None:
            raise ValueError("Selected spec is missing from the source archive")
        spec_bytes = member.read()
    digest = hashlib.sha256(spec_bytes).hexdigest()
    if digest != expected_sha256 or (trial and spec_bytes != trial.spec_bytes):
        raise ValueError("Selected spec changed between validation and packaging")
    destination = results_root(ROOT) / "colab_runs" / session
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(archive_path, destination / "source.tar.gz")
    (destination / "spec.json").write_bytes(spec_bytes)
    provenance = {"config": relative_spec.as_posix(), "spec_sha256": digest}
    if trial:
        (destination / "trial.json").write_bytes(trial.trial_bytes)
        (destination / "template.json").write_bytes(trial.template_bytes)
        provenance.update(trial_id=trial.trial_id,
                          template=trial.template_path.relative_to(ROOT).as_posix(),
                          template_sha256=hashlib.sha256(trial.template_bytes).hexdigest(),
                          trial_sha256=hashlib.sha256(trial.trial_bytes).hexdigest())
    (destination / "spec_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return destination / "spec.json"


def spec_requires_task_variations(spec: dict) -> bool:
    dataset = spec.get("dataset") or {}
    return (
        isinstance(dataset, dict)
        and isinstance(dataset.get("pretrain_data"), dict)
        and dataset["pretrain_data"].get("kind") == "task_variations"
        and not (spec.get("eval_only", False) and spec.get("skip_data_build", False))
    )


def redact_runtime_credentials(value: object) -> str:
    """Colab/requests errors can include runtime credentials in a request URL."""
    text = str(value)
    text = re.sub(
        r"(?i)(\b(?:colab-runtime-proxy-token|access_token|refresh_token|id_token|token)(?:=|%3d))[^\s&\"'<>)]+",
        r"\1[REDACTED]", text,
    )
    return re.sub(r"(?i)(\bBearer\s+)[A-Za-z0-9_.~-]+", r"\1[REDACTED]", text)


def run_command(args: list[str], *, timeout: int | None = None, quiet: bool = False,
                log_path: Path | None = None, on_output=None) -> None:
    args = colab_accounts.route_command(args)
    if log_path is not None:
        if timeout is not None or quiet:
            raise ValueError("Streaming output cannot be combined with timeout or quiet mode")
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(args, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    line = redact_runtime_credentials(line)
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                    if on_output:
                        try:
                            on_output(line)
                        except Exception as exc:
                            print(f"Could not update live run notes: {redact_runtime_credentials(exc)}", file=sys.stderr)
                returncode = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
        if returncode:
            raise RuntimeError(f"Command failed ({returncode}): {redact_runtime_credentials(' '.join(args))}")
        return
    result = subprocess.run(
        args,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )
    if not quiet:
        if result.stdout:
            print(redact_runtime_credentials(result.stdout), end="", flush=True)
        if result.stderr:
            print(redact_runtime_credentials(result.stderr), end="", file=sys.stderr, flush=True)
    if result.returncode:
        detail = redact_runtime_credentials((result.stderr or result.stdout or "").strip()) if quiet else ""
        raise RuntimeError(f"Command failed ({result.returncode}): {redact_runtime_credentials(' '.join(args))}\n{detail}")


def require_unused_remote_workspace(colab: str, session: str) -> None:
    """Do not overwrite source or records belonging to an existing VM run."""
    result = subprocess.run(
        colab_accounts.cli_command(colab, "ls", "-s", session, "/content"), cwd=ROOT,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=30, check=False,
    )
    if result.returncode:
        raise RuntimeError("Could not inspect the Colab VM before upload; no files were transferred")
    names = set(result.stdout.splitlines())
    if names.intersection({"mdlARC/", "mdlARC-source.tar.gz", "mdlARC-results-manifest.json"}):
        raise RuntimeError("This Colab VM already contains mdlARC run files; use a new session")


def connection_identity(session: str) -> Path:
    return results_root(ROOT) / "colab_runs" / "_active" / f"{session}.connection.json"


def connection_command(colab: str, session: str, run_name: str, *arguments: str) -> list[str]:
    return [colab_connection.cli_python(colab), "-u", str(Path(__file__).with_name("colab_connection.py")),
            "--session", session, "--run-name", run_name, "--identity", str(connection_identity(session)),
            "--account", colab_accounts.selected_account(),
            "--parent-pid", str(os.getpid()), *arguments]


def session_assignment_available(colab: str, session: str) -> bool | None:
    """Only an authenticated assignment listing can confirm a missing VM."""
    try:
        result = subprocess.run([
            colab_connection.cli_python(colab), str(Path(__file__).with_name("colab_connection.py")),
            "--session", session, "--identity", str(connection_identity(session)), "--probe",
            "--account", colab_accounts.selected_account(),
        ], cwd=ROOT, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False)
        if result.returncode == 0:
            value = json.loads(result.stdout).get("available")
            if value is True or value is False:
                return value
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired):
        pass
    return None


def read_remote_snapshot(colab: str, session: str, run_name: str) -> tuple[str | None, dict | None]:
    """Use bounded file requests, independently of the execution WebSocket."""
    def request(arguments: list[str]) -> str:
        result = subprocess.run(colab_accounts.cli_command(colab, *arguments), cwd=ROOT, stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=30, check=False)
        if result.returncode:
            raise RuntimeError(redact_runtime_credentials((result.stderr or result.stdout or "VM file request failed").strip()))
        return result.stdout

    names = set(request(["ls", "-s", session, "/content"]).splitlines())
    with tempfile.TemporaryDirectory(prefix="mdlARC-watch-") as temp_name:
        local = Path(temp_name) / "snapshot"
        if Path(REMOTE_MANIFEST).name in names:
            request(["download", "-s", session, REMOTE_MANIFEST, str(local)])
            manifest = json.loads(local.read_text(encoding="utf-8"))
            if manifest.get("run_name", run_name) != run_name or manifest.get("session", session) != session:
                raise ValueError("Remote results belong to a different run or session")
            return None, manifest
        if "mdlARC/" not in names:
            return None, None
        remote_log = f"/content/mdlARC/runs/{run_name}/console.log"
        # The training console is created after environment/data setup. A missing
        # console is normal here; the next poll still checks the completion file.
        try:
            request(["download", "-s", session, remote_log, str(local)])
        except RuntimeError as exc:
            if "404" in str(exc) or "not found" in str(exc).lower():
                return None, None
            raise
        return local.read_text(encoding="utf-8", errors="replace"), None


def watch_remote_run(colab: str, session: str, run_name: str, *, command: list[str] | None = None,
                     poll_interval: float = WATCH_INTERVAL) -> dict:
    """Stream when available; poll saved logs/results even if the CLI hangs."""
    active = results_root(ROOT) / "colab_runs" / "_active"
    active.mkdir(parents=True, exist_ok=True)
    messages: queue.Queue[str | None] = queue.Queue()
    process = reader = None
    stream_log = None
    if command:
        stream_log = (active / f"{session}.log").open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                       start_new_session=True)
        except BaseException:
            stream_log.close()
            raise

        def read_stream():
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    messages.put(line)
            finally:
                messages.put(None)

        reader = threading.Thread(target=read_stream, daemon=True)
        reader.start()
    previous = ""
    next_poll = 0.0
    last_stream = time.monotonic()
    last_change = last_stream
    missing_checks = 0
    last_assignment_check = 0.0
    stream_ended_at = None
    fallback = command is None
    print(f"[colab-watch] Checking saved VM logs and results every {poll_interval:g}s. "
          "Connection failures will retry without restarting training.", flush=True)
    try:
        while True:
            now = time.monotonic()
            if process is not None and process.poll() is not None:
                stream_ended_at = stream_ended_at or now
                if any(Path(argument).name == "colab_connection.py" for argument in (command or [])):
                    status_path = connection_identity(session).with_suffix(".status.json")
                    state = json.loads(status_path.read_text()) if status_path.exists() else {}
                    if not connection_identity(session).exists() or state.get("submitted") is False:
                        raise RuntimeError("Kernel connection failed before a submitted job was recorded; no execution was retried")
                    if state.get("state") == "job_failed" and now - stream_ended_at >= 2 * poll_interval:
                        raise RuntimeError("Remote execution failed without publishing recoverable results; keep this VM for inspection")
            if not fallback and (process.poll() is not None or now - last_stream >= 2 * poll_interval):
                fallback = True
                print("[colab-watch] Terminal stream ended or went quiet; following the saved VM console.", flush=True)
            if now >= next_poll:
                try:
                    console, manifest = read_remote_snapshot(colab, session, run_name)
                    if manifest is not None:
                        print("[colab-watch] Remote results are ready; recovering them.", flush=True)
                        return manifest
                    colab_run_docs.update_journal(ROOT, session, last_vm_check_at=colab_run_docs.utc_now())
                    if console is not None:
                        (active / f"{session}.vm.log").write_text(console, encoding="utf-8")
                        # Keep an incomplete last line for the next snapshot.
                        complete = console[:console.rfind("\n") + 1]
                        appended = complete[len(previous):] if complete.startswith(previous) else complete
                        lines = appended.splitlines(keepends=True)
                        if fallback and lines:
                            for line in (lines if previous else lines[-20:]):
                                print(line, end="", flush=True)
                        changed = False
                        for line in lines:
                            changed = colab_run_docs.observe_output(ROOT, session, line) or changed
                        if changed:
                            colab_run_docs.render_session(ROOT, session)
                        if complete != previous:
                            last_change = time.monotonic()
                        previous = complete
                    if fallback:
                        detail = (f"last console change {time.monotonic() - last_change:.0f}s ago"
                                  if previous else "waiting for the training console")
                        print(f"[colab-watch] VM checked; {detail}. Waiting for results.", flush=True)
                except (OSError, RuntimeError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
                    available = session_assignment_available(colab, session)
                    last_assignment_check = time.monotonic()
                    # Do not turn Wi-Fi loss, authentication errors, or file 404s
                    # into a false disappearance. Require two successful server
                    # listings with the original assignment absent.
                    missing_checks = missing_checks + 1 if available is False else 0
                    if missing_checks >= 2:
                        colab_run_docs.update_journal(ROOT, session, state="unavailable", session_disposition="unavailable",
                                                     unavailable_at=colab_run_docs.utc_now())
                        raise colab_connection.SessionUnavailable(
                            "Colab confirmed the original VM is unavailable twice. Local records and Drive backups were preserved; no VM or training was restarted.") from None
                    print(f"[colab-watch] Could not read VM files; retrying in {poll_interval:g}s: {redact_runtime_credentials(exc)}", flush=True)
                next_poll = time.monotonic() + poll_interval
            # An absent training console can also be a normal setup phase. Probe
            # liveness independently so that this state cannot hide a lost VM.
            if now - last_assignment_check >= max(60, 2 * poll_interval):
                available = session_assignment_available(colab, session)
                missing_checks = missing_checks + 1 if available is False else 0
                last_assignment_check = time.monotonic()
                if missing_checks >= 2:
                    colab_run_docs.update_journal(ROOT, session, state="unavailable", session_disposition="unavailable",
                                                 unavailable_at=colab_run_docs.utc_now())
                    raise colab_connection.SessionUnavailable("Colab confirmed the original VM is unavailable twice; no job was restarted")
            delay = max(0.01, min(1.0, next_poll - time.monotonic()))
            if process is None:
                time.sleep(delay)
                continue
            try:
                line = messages.get(timeout=delay)
            except queue.Empty:
                continue
            if line is None:
                if not fallback:
                    print("[colab-watch] Terminal stream ended; following the saved VM console.", flush=True)
                fallback = True
            else:
                line = redact_runtime_credentials(line)
                print(line, end="", flush=True)
                stream_log.write(line)
                stream_log.flush()
                last_stream = time.monotonic()
    finally:
        # This closes only the local CLI. Never interrupt/re-execute the VM cell.
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            if reader is not None:
                reader.join(timeout=2)
            if process.stdout is not None:
                process.stdout.close()
        if stream_log is not None:
            stream_log.close()


def safe_extract(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts or not member.isfile():
                raise ValueError(f"Unsafe result archive member: {member.name}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Unreadable result archive member: {member.name}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def download_results(colab: str, session: str, destination: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="mdlARC-results-") as temp_name:
        temp = Path(temp_name)
        manifest_path = temp / "manifest.json"
        run_command([colab, "download", "-s", session, REMOTE_MANIFEST, str(manifest_path)], timeout=60)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        archive_path = temp / "results.tar.gz"
        combined_hash = hashlib.sha256()
        with archive_path.open("wb") as combined:
            for part in manifest["parts"]:
                name = part["name"]
                if not re.fullmatch(r"mdlARC-results\.part\d{4}", name):
                    raise ValueError(f"Unexpected result chunk name: {name}")
                local_part = temp / name
                run_command([colab, "download", "-s", session, f"/content/{name}", str(local_part)], timeout=60)
                data = local_part.read_bytes()
                if len(data) != part["size"] or hashlib.sha256(data).hexdigest() != part["sha256"]:
                    raise ValueError(f"Result chunk verification failed: {name}")
                combined.write(data)
                combined_hash.update(data)
                local_part.unlink()
        if combined_hash.hexdigest() != manifest["sha256"]:
            raise ValueError("Result archive verification failed")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as staging_name:
            staging = Path(staging_name)
            safe_extract(archive_path, staging)
            if destination.exists():
                # Launch records are saved before any results are downloaded.
                launch_record_names = {"source.tar.gz", "spec.json", "trial.json", "template.json",
                                       "spec_provenance.json", "reuse.json"}
                if {path.name for path in destination.iterdir()} - launch_record_names:
                    raise FileExistsError(f"Result destination already contains results: {destination}")
                if any((destination / path.name).exists() for path in staging.iterdir()):
                    raise FileExistsError(f"Result archive conflicts with local launch records: {destination}")
                for path in staging.iterdir():
                    path.rename(destination / path.name)
            else:
                staging.rename(destination)
    return json.loads((destination / "run_status.json").read_text(encoding="utf-8"))


def recover_results(colab: str, session: str, destination: Path) -> dict:
    for attempt in range(3):
        try:
            return download_results(colab, session, destination)
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            if attempt == 2:
                raise
            print(f"[colab-watch] Result recovery failed; retrying in 5s: {redact_runtime_credentials(exc)}", flush=True)
            time.sleep(5)
    raise AssertionError("unreachable")


def sync_records_to_drive(colab: str, session: str, status: dict) -> dict:
    """Publish recovered local records using only the original VM's credentials."""
    folder = status["drive_folder_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", folder):
        raise ValueError("Invalid Drive run folder")
    colab_run_docs.render_index(ROOT)
    with tempfile.TemporaryDirectory(prefix="mdlARC-records-") as temp_name:
        bundle = Path(temp_name) / "records.tar.gz"
        expected = run_records.pack_records(ROOT, bundle, session)
        run_command([colab, "upload", "-s", session, str(bundle), "/content/mdlARC-local-records.tar.gz"], timeout=120)
        run_command([colab, "exec", "-s", session, "--file", str(ROOT / "scripts/run_records.py"),
                     "--timeout", "120", "--env", "MDLARC_PUBLISH_RECORDS=1",
                     "--env", f"MDLARC_RUN_BRANCH={branch_name(ROOT)}",
                     "--env", f"MDLARC_RECORDS_FOLDER={folder}"], timeout=150)
        receipt = Path(temp_name) / "receipt.json"
        run_command([colab, "download", "-s", session, "/content/mdlARC-local-records-status.json", str(receipt)], timeout=60)
        result = json.loads(receipt.read_text())
        if any(result.get(key) != expected[key] for key in ("size", "sha256", "md5")):
            raise ValueError("Drive record backup did not verify")
        ledger = results_root(ROOT) / "drive_sync" / f"{session}.json"
        colab_run_docs._write_atomic(ledger, json.dumps(result, indent=2) + "\n")
        return result


def watch_existing_session(colab: str, session: str, *, stop_after_success: bool = False, account=None) -> int:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", session) or session in {".", ".."}:
        raise ValueError("--watch-session must be one safe name")
    lock = results_root(ROOT) / "colab_runs" / "_active" / f"{session}.watch.lock"
    owner = colab_accounts.run_account(ROOT, session, account)
    with colab_accounts.account_context(owner), colab_connection.session_lock(lock), colab_connection.prevent_idle_sleep():
        return _watch_existing_session(colab, session, stop_after_success=stop_after_success)


def _watch_existing_session(colab: str, session: str, *, stop_after_success: bool = False,
                            initial_command: list[str] | None = None) -> int:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", session) or session in {".", ".."}:
        raise ValueError("--watch-session must be one safe name")
    journal = json.loads((results_root(ROOT) / "colab_runs" / "_active" / f"{session}.json").read_text())
    run_name = journal["run_name"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name) or run_name in {".", ".."}:
        raise ValueError("Saved run_name must be one safe directory name")
    drive_backed = journal.get("drive_backed", journal.get("session_disposition") in
                               {"stop after success", "user-managed"})
    if drive_backed is not True:
        raise ValueError("--watch-session supports shared Drive runs only")
    destination = results_root(ROOT) / "colab_runs" / session
    if (destination / "run_status.json").exists():
        raise ValueError("This session already has recovered results; review its report instead")
    # The saved launch record supplies identity, never a subsequently edited spec.
    saved = destination / "spec.json"
    if hashlib.sha256(saved.read_bytes()).hexdigest() != journal["spec_sha256"]:
        raise ValueError("Saved launch spec does not match the session journal")
    stop_after_success = stop_after_success or journal.get("stop_after_success", False)
    status = None
    try:
        if initial_command is not None:
            manifest = watch_remote_run(colab, session, run_name, command=initial_command)
        elif connection_identity(session).exists():
            command = connection_command(colab, session, run_name)
            manifest = watch_remote_run(colab, session, run_name, command=command)
        else:
            print("[colab-watch] Historical run has no saved kernel identity; following saved files only.", flush=True)
            manifest = watch_remote_run(colab, session, run_name)
        colab_run_docs.update_journal(ROOT, session, state="downloading_results")
        status = recover_results(colab, session, destination)
        print(f"Results saved to {destination}", flush=True)
        success = status.get("ok") is True and status.get("backups_complete") is True
        colab_run_docs.update_journal(ROOT, session, state="completed" if success else "needs_recovery")
        if status.get("drive_folder_id"):
            # As with a new launch, verify the final local record bundle before
            # allowing automatic shutdown after attachment or safe first submission.
            try:
                sync_records_to_drive(colab, session, status)
            except Exception as exc:
                colab_run_docs.update_journal(ROOT, session, state="needs_recovery",
                    records_backup_error=f"{type(exc).__name__}: {redact_runtime_credentials(exc)}")
                raise
        if stop_after_success and success and manifest.get("finalized") is True:
            try:
                run_command([colab, "stop", "-s", session], timeout=30)
            except (RuntimeError, subprocess.TimeoutExpired):
                colab_run_docs.update_journal(ROOT, session, session_disposition="stop_failed")
                raise RuntimeError(f"Results recovered, but shutdown failed; run `{colab_accounts.cli_hint()} stop -s {session}`") from None
            colab_run_docs.update_journal(ROOT, session, session_disposition="stopped")
            print(f"Colab session {session} stopped.", flush=True)
        elif stop_after_success:
            colab_run_docs.update_journal(ROOT, session, session_disposition="retained for recovery")
            print("Session retained: verified success and a finalized completion marker are required for shutdown. "
                  "Older runs need a manual check of the final Drive summary before stopping.", flush=True)
        return 0 if success else 1
    except BaseException as exc:
        fields = {"launcher_error": f"{type(exc).__name__}: {redact_runtime_credentials(exc)}"}
        if isinstance(exc, colab_connection.SessionUnavailable):
            fields.update(state="unavailable", session_disposition="unavailable")
        elif status is None:
            fields["state"] = "needs_recovery"
        colab_run_docs.update_journal(ROOT, session, **fields)
        if isinstance(exc, colab_connection.SessionUnavailable):
            print("Session unavailable; preserve its local records and existing Drive backups.", file=sys.stderr)
        else:
            print(f"Session retained. Reattach with `venv/bin/python scripts/run_on_colab.py --watch-session {session}`.", file=sys.stderr)
        raise
    finally:
        colab_run_docs.render_index(ROOT)


def resume_unsubmitted_session(colab: str, session: str, *, expected_endpoint: str,
                               expected_kernel: str, stop_after_success: bool = False, account=None) -> int:
    """Finish a fresh pretrain/eval launch which never reached submission.

    Reuse the uploaded archive and the existing idle kernel. Never authorize,
    upload, create a kernel, overwrite identity, or retry ambiguous delivery.
    """
    for value in (session, expected_endpoint, expected_kernel):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) or value in {".", ".."}:
            raise ValueError("A saved session, VM endpoint and kernel ID are required")
    owner = colab_accounts.run_account(ROOT, session, account)
    active = results_root(ROOT) / "colab_runs" / "_active"
    with colab_accounts.account_context(owner), colab_connection.session_lock(active / f"{session}.watch.lock"), colab_connection.prevent_idle_sleep():
        if connection_identity(session).exists():
            raise ValueError("Execution identity exists; inspect or use --watch-session, never resubmit")
        journal = json.loads((active / f"{session}.json").read_text())
        status_path = connection_identity(session).with_suffix(".status.json")
        if (any(journal.get(key) for key in ("stage", "epoch", "last_progress_at"))
                or (status_path.exists() and json.loads(status_path.read_text()).get("submitted") is True)):
            raise ValueError("Previous execution evidence exists; inspect instead of submitting")
        destination = results_root(ROOT) / "colab_runs" / session
        if journal.get("drive_backed") is not True or journal.get("state") != "needs_recovery":
            raise ValueError("Only a retained Drive-backed, unsubmitted launch can be resumed")
        if (destination / "run_status.json").exists():
            raise ValueError("Recovered results already exist; refusing submission")
        archive = destination / "source.tar.gz"
        if hashlib.sha256(archive.read_bytes()).hexdigest() != journal["source_sha256"]:
            raise ValueError("Saved source archive does not match the launch journal")
        spec_bytes = (destination / "spec.json").read_bytes()
        if hashlib.sha256(spec_bytes).hexdigest() != journal["spec_sha256"]:
            raise ValueError("Saved spec does not match the launch journal")
        spec = json.loads(spec_bytes)
        if (spec.get("checkpoint") or spec.get("eval_only") or spec.get("enable_finetune") is not False
                or spec.get("run_name") != journal["run_name"]):
            raise ValueError("Unsubmitted recovery supports fresh pretraining/evaluation specs without launch-only overrides")
        config = Path(journal["config"])
        if config.is_absolute() or ".." in config.parts:
            raise ValueError("Invalid saved spec path")
        with tarfile.open(archive, "r:gz") as source:
            if source.extractfile(config.as_posix()).read() != spec_bytes:
                raise ValueError("Uploaded spec differs from the saved spec")
            remote_code = source.extractfile("scripts/colab_remote.py").read()
        with tempfile.TemporaryDirectory(prefix="mdlARC-unsubmitted-") as temporary:
            temp = Path(temporary)
            listing = subprocess.run(colab_accounts.cli_command(colab, "ls", "-s", session, "/content"),
                                     cwd=ROOT, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)
            if listing.returncode:
                raise RuntimeError("Could not inspect VM files; no execution was submitted")
            names = set(listing.stdout.splitlines())
            if ("mdlARC-source.tar.gz" not in names or "mdlARC/" in names
                    or any(name.startswith("mdlARC-results") for name in names)):
                raise ValueError("VM contains execution evidence or lacks the source; inspect instead of submitting")
            remote_archive = temp / "uploaded-source.tar.gz"
            run_command([colab, "download", "-s", session, REMOTE_SOURCE, str(remote_archive)], timeout=60)
            if hashlib.sha256(remote_archive.read_bytes()).hexdigest() != journal["source_sha256"]:
                raise ValueError("VM source archive differs from the original upload")
            code_path = temp / "colab_remote.py"
            code_path.write_bytes(remote_code)
            environment = {
                "MDLARC_SPEC": config.as_posix(), "MDLARC_RUN_NAME": journal["run_name"],
                "MDLARC_RUN_BRANCH": branch_name(ROOT), "MDLARC_REMOTE": "1",
                "MDLARC_MIN_GPU_MEMORY_GB": str(journal["min_gpu_memory_gb"]),
                "MDLARC_DRIVE": "1", "MDLARC_DRIVE_AUTH": "shared",
                "MDLARC_TASK_VARIATIONS": str(int(spec_requires_task_variations(spec))),
                "MDLARC_PRETRAIN_DRIVE_ID": "", "MDLARC_PRETRAIN_NAME": "pretrain__tiny.pt",
                "MDLARC_PRETRAIN_SIZE": "", "MDLARC_PRETRAIN_SHA256": "",
                "MDLARC_FINETUNE_TARGET": "", "MDLARC_EVAL_TARGET": "", "MDLARC_SESSION": session,
            }
            arguments = ["--file", str(code_path), "--timeout", "90000",
                         "--expected-endpoint", expected_endpoint, "--expected-kernel-id", expected_kernel]
            for key, value in environment.items():
                arguments.extend(["--env", f"{key}={value}"])
            attempt = {"session": session, "endpoint": expected_endpoint, "kernel_id": expected_kernel,
                       "source_sha256": journal["source_sha256"], "spec_sha256": journal["spec_sha256"],
                       "checked_at": colab_run_docs.utc_now(), "execution_identity_absent": True}
            colab_connection.atomic_json(active / f"{session}.resume-{uuid.uuid4().hex}.json", attempt)
            command = connection_command(colab, session, journal["run_name"], *arguments)
            colab_run_docs.update_journal(ROOT, session, state="running", launcher_error=None,
                resume_previous_launcher_error=journal.get("launcher_error"), resumed_at=attempt["checked_at"],
                session_disposition="stop after success" if stop_after_success or journal.get("stop_after_success") else "user-managed")
            print("[colab] Resuming the first submission on the original idle kernel; source and spec are unchanged.", flush=True)
            return _watch_existing_session(colab, session, initial_command=command, stop_after_success=stop_after_success)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", help="Branch owning the run records (default: current Git branch)")
    parser.add_argument("--account", choices=colab_accounts.ACCOUNTS,
                        help="Colab compute account: 1=your-colab-account@example.com (default), 2=your-drive-account@example.com; watchers use saved ownership")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--config", help="Path to a JSON experiment spec inside this repository")
    selection.add_argument("--watch-session", help="Follow and recover an existing shared Drive run without uploading or executing training")
    selection.add_argument("--resume-unsubmitted", help="Explicitly finish a never-submitted fresh pretrain/eval launch on its original idle kernel")
    parser.add_argument("--expected-endpoint", help="Original VM endpoint, required for --resume-unsubmitted")
    parser.add_argument("--expected-kernel-id", help="Original kernel ID, required for --resume-unsubmitted")
    provenance = parser.add_mutually_exclusive_group()
    provenance.add_argument("--trial-record", type=Path, help="Trial record used to generate --config; checked and saved with the run")
    provenance.add_argument("--standalone-spec", action="store_true", help="Use a manually authored temporary spec without a sweep trial record")
    parser.add_argument("--gpu", default="A100", choices=("T4", "L4", "G4", "H100", "A100"))
    parser.add_argument("--high-mem", action="store_true", help="Request Colab's high-memory machine shape")
    parser.add_argument("--min-gpu-memory-gb", type=float, default=0, help="Abort before setup if GPU VRAM is below this many GiB")
    parser.add_argument("--dry-run", action="store_true", help="Validate and package locally without allocating a VM")
    parser.add_argument("--local-artifacts", action="store_true", help="Legacy mode: upload the local wheel and download checkpoints to this Mac")
    parser.add_argument("--task-variations", action="store_true", help="Download and extract the versioned task-variations ZIP from restricted Drive on the Colab VM")
    parser.add_argument("--session", help="Existing Colab session authorized as your-drive-account@example.com; use scripts/colab_oauth/launch.py to authorize with the cached token")
    parser.add_argument("--reuse-session", help="Preserve a terminal, fully recovered run and use its idle VM for a fresh --session alias")
    parser.add_argument("--stop-after-success", action="store_true", help="Stop the shared Colab VM only after a successful run, verified Drive backups, and local result recovery")
    parser.add_argument("--pretrain-drive-id", help="Existing pretrain checkpoint in mdlARC/runs to stage directly on the VM")
    parser.add_argument("--pretrain-name", default="pretrain__tiny.pt", help="Expected Drive checkpoint filename")
    parser.add_argument("--pretrain-size", type=int, help="Expected checkpoint size in bytes")
    parser.add_argument("--pretrain-sha256", help="Expected checkpoint SHA-256, or 'drive' to verify against Drive metadata on the VM")
    parser.add_argument("--checkpoint-mount", help="Existing VM-side mounted mdlARC/runs directory; copy input checkpoints locally instead of downloading through the API")
    parser.add_argument("--finetune-target", help="Finetune stage to run from the staged pretrain checkpoint")
    parser.add_argument("--eval-target", help="Optional evaluation stage to run after that finetune")
    parser.add_argument("--intent", help="Run intent; provide with --changed-from-previous to skip the Mac dialogs")
    parser.add_argument("--changed-from-previous", help="What changed since the previous run; provide with --intent")
    args = parser.parse_args()
    if args.branch:
        os.environ["MDLARC_RUN_BRANCH"] = args.branch
    # Freeze ownership before subprocesses or a branch switch can change it.
    os.environ["MDLARC_RUN_BRANCH"] = branch_name(ROOT)
    if args.watch_session or args.resume_unsubmitted:
        if (args.session or args.dry_run or args.local_artifacts or args.trial_record or args.standalone_spec
                or args.intent or args.changed_from_previous or args.pretrain_drive_id
                or args.pretrain_size is not None or args.pretrain_sha256 or args.finetune_target
                or args.eval_target or args.task_variations or args.reuse_session or args.checkpoint_mount):
            parser.error("Recovery cannot be combined with launch options")
        colab = shutil.which("colab")
        if not colab:
            parser.error("Colab CLI is missing")
        if args.resume_unsubmitted:
            if not args.expected_endpoint or not args.expected_kernel_id:
                parser.error("--resume-unsubmitted requires --expected-endpoint and --expected-kernel-id")
            return resume_unsubmitted_session(colab, args.resume_unsubmitted,
                expected_endpoint=args.expected_endpoint, expected_kernel=args.expected_kernel_id,
                stop_after_success=args.stop_after_success, account=args.account)
        if args.expected_endpoint or args.expected_kernel_id:
            parser.error("Expected VM/kernel options require --resume-unsubmitted")
        return watch_existing_session(colab, args.watch_session, stop_after_success=args.stop_after_success, account=args.account)
    if args.expected_endpoint or args.expected_kernel_id:
        parser.error("Expected VM/kernel options require --resume-unsubmitted")
    os.environ[colab_accounts.ACCOUNT_ENV] = colab_accounts.selected_account(args.account)
    if args.reuse_session and (not args.session or args.local_artifacts or args.reuse_session == args.session):
        parser.error("--reuse-session requires a different fresh --session and shared Drive mode")
    spec_path = (ROOT / args.config).resolve()
    try:
        relative_spec = spec_path.relative_to(ROOT)
    except ValueError:
        parser.error("--config must point to a file inside this repository")
    if not spec_path.is_file():
        parser.error(f"Config does not exist: {spec_path}")
    trial = None
    if args.trial_record:
        try:
            trial = materialize_sweep_spec.materialize(ROOT, args.trial_record)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(f"Invalid trial record: {exc}")
        if spec_path.read_bytes() != trial.spec_bytes:
            parser.error("--config does not match --trial-record; materialize the trial again")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    sys.path.insert(0, str(ROOT))
    from launch_run import resolve_input_checkpoint
    try:
        _, drive_checkpoint = resolve_input_checkpoint(spec.get("checkpoint"), runs_root(ROOT) / str(spec.get("run_name") or spec_path.stem))
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if (relative_spec.parts[:2] == ("specs", "temp specs")
            and not args.trial_record and not drive_checkpoint and not args.standalone_spec):
        parser.error("Staged sweep specs require --trial-record; manually authored temp specs require --standalone-spec or a Drive checkpoint object")
    if drive_checkpoint and (args.pretrain_drive_id or args.pretrain_size is not None
                             or args.pretrain_sha256 is not None or args.pretrain_name != "pretrain__tiny.pt"
                             or args.finetune_target or args.eval_target):
        parser.error("A checkpoint object in the spec cannot be combined with checkpoint or finetune-only CLI flags")
    drive = not args.local_artifacts
    shared = drive
    if args.local_artifacts and (args.session or args.pretrain_drive_id or drive_checkpoint):
        parser.error("Local-artifacts mode cannot use a shared Drive session or Drive checkpoint")
    if args.local_artifacts and args.stop_after_success:
        parser.error("--stop-after-success is for shared Drive sessions; local-artifacts sessions already have automatic cleanup")
    if shared and not args.dry_run and not args.session:
        parser.error("Shared Drive runs require --session; create a fresh VM and launch through scripts/colab_oauth/launch.py (manual fallback: colab auth -s SESSION)")
    if args.pretrain_drive_id:
        if (not shared or not args.pretrain_size or not args.pretrain_sha256
                or not args.finetune_target
                or not re.fullmatch(r"[A-Za-z0-9_-]+", args.pretrain_drive_id)
                or (args.pretrain_sha256 != "drive" and not re.fullmatch(r"[0-9a-f]{64}", args.pretrain_sha256))
                or args.pretrain_size <= 0
                or Path(args.pretrain_name).name != args.pretrain_name
                or not args.pretrain_name.endswith(".pt")):
            parser.error("A shared checkpoint needs its Drive ID, .pt name, positive size, SHA-256 (or 'drive'), and finetune target")
    elif args.pretrain_size is not None or args.pretrain_sha256 is not None or args.finetune_target:
        parser.error("Checkpoint details and --finetune-target require --pretrain-drive-id")
    if args.eval_target and not args.finetune_target:
        parser.error("--eval-target requires --finetune-target")
    if args.finetune_target and args.finetune_target not in (spec.get("finetunes") or {}):
        parser.error("--finetune-target must name a finetune in the selected spec")
    if args.eval_target and args.eval_target not in (spec.get("evals") or {}):
        parser.error("--eval-target must name an evaluation in the selected spec")
    if ((args.pretrain_drive_id or drive_checkpoint) and spec.get("skip_data_build", False)
            and "checkpoint_adaptation" not in spec):
        parser.error("This spec skips data building; use a new spec that builds data on the VM")
    if args.checkpoint_mount:
        if (args.local_artifacts or not (args.pretrain_drive_id or drive_checkpoint or spec.get("evaluation_checkpoints"))):
            parser.error("--checkpoint-mount requires a Drive input checkpoint")
        mount = Path(args.checkpoint_mount)
        if (not mount.is_absolute() or ".." in mount.parts or "\\" in args.checkpoint_mount
                or any(ord(char) < 32 for char in args.checkpoint_mount)):
            parser.error("--checkpoint-mount must be an absolute VM directory without traversal")
    task_variations = args.task_variations or spec_requires_task_variations(spec)
    if task_variations and args.local_artifacts:
        parser.error("Task variations require the restricted Drive mode")
    run_name = str(spec.get("run_name") or spec_path.stem)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name) or run_name in {".", ".."}:
        parser.error("run_name must be a single safe directory name")
    if (args.intent is None) != (args.changed_from_previous is None):
        parser.error("Provide both --intent and --changed-from-previous, or neither for Mac dialogs")

    if args.dry_run:
        run_command([sys.executable, str(ROOT / "launch_run.py"), "--config", str(spec_path), "--preview"], quiet=True)
        with tempfile.TemporaryDirectory(prefix="mdlARC-source-") as temp_name:
            archive_path = Path(temp_name) / "source.tar.gz"
            count, size = create_snapshot(spec_path, archive_path, drive=not args.local_artifacts)
            print(f"Packaged {count} files ({size / 1024 / 1024:.1f} MiB) from the current working tree.", flush=True)
        return 0

    if args.intent is None:
        intent, changes = colab_run_docs.prompt_for_notes(run_name)
    else:
        intent, changes = args.intent, args.changed_from_previous
    session = args.session or f"mdlarc-{datetime.now(UTC):%Y%m%d-%H%M}-{uuid.uuid4().hex[:6]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", session) or session in {".", ".."}:
        parser.error("--session must be one safe name")
    if (results_root(ROOT) / "experiments" / "notes" / f"{session}.md").exists() or (results_root(ROOT) / "colab_runs" / session).exists():
        parser.error("Session already has local run records; choose a new session")
    note_path = colab_run_docs.create_note(ROOT, session, run_name, intent, changes)
    journal = colab_run_docs.start_journal(ROOT, session, config=relative_spec.as_posix(), run_name=run_name,
                                           gpu=args.gpu, high_mem=args.high_mem,
                                           min_gpu_memory_gb=args.min_gpu_memory_gb, spec_path=spec_path)
    journal = colab_run_docs.update_journal(ROOT, session, stop_after_success=args.stop_after_success,
                                          drive_backed=drive, colab_account=colab_accounts.selected_account(),
                                          colab_email=colab_accounts.ACCOUNTS[colab_accounts.selected_account()],
                                          checkpoint_mount=args.checkpoint_mount)
    colab_run_docs.render_index(ROOT)
    print(f"Run notes saved to {note_path}", flush=True)

    allocated = False
    recovered = False
    started = False
    status = None
    colab = None
    locks = ExitStack()
    try:
        locks.enter_context(colab_connection.session_lock(results_root(ROOT) / "colab_runs" / "_active" / f"{session}.watch.lock"))
        locks.enter_context(colab_connection.prevent_idle_sleep())
        run_command([sys.executable, str(ROOT / "launch_run.py"), "--config", str(spec_path), "--preview"], quiet=True)
        with tempfile.TemporaryDirectory(prefix="mdlARC-source-") as temp_name:
            archive_path = Path(temp_name) / "source.tar.gz"
            count, size = create_snapshot(spec_path, archive_path, drive=drive, session=session)
            print(f"Packaged {count} files ({size / 1024 / 1024:.1f} MiB) from the current working tree.", flush=True)
            preserve_launch_spec(session, archive_path, relative_spec,
                                 journal["spec_sha256"], trial)
            if archive_path.is_file():
                digest = hashlib.sha256()
                with archive_path.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
                colab_run_docs.update_journal(ROOT, session, source_sha256=digest.hexdigest())

            colab = shutil.which("colab")
            if not colab:
                raise RuntimeError("Colab CLI is missing. Install with: uv tool install google-colab-cli")
            try:
                run_command([colab, "sessions"], timeout=30, quiet=True)
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                account = colab_accounts.selected_account()
                raise RuntimeError(f"Colab login is not ready. Run `venv/bin/python scripts/colab_accounts.py --account {account} -- sessions` in your terminal and complete Google sign-in first.") from exc

            destination = results_root(ROOT) / "colab_runs" / session
            if shared:
                if args.reuse_session:
                    previous = args.reuse_session
                    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", previous) or previous in {".", ".."}:
                        raise ValueError("Invalid previous session name")
                    if colab_accounts.run_account(ROOT, previous, args.account) != colab_accounts.selected_account():
                        raise ValueError("Reuse requires the original compute account")
                    locks.enter_context(colab_connection.session_lock(results_root(ROOT) / "colab_runs/_active" / f"{previous}.watch.lock"))
                    previous_base = results_root(ROOT) / "colab_runs" / previous
                    run_command([colab_connection.cli_python(colab), str(ROOT / "scripts/colab_reuse.py"),
                        "--account", colab_accounts.selected_account(), "--previous", previous, "--session", session,
                        "--identity", str(connection_identity(previous)),
                        "--journal", str(results_root(ROOT) / "colab_runs/_active" / f"{previous}.json"),
                        "--status", str(previous_base / "run_status.json"),
                        "--output", str(destination / "reuse.json")], timeout=90)
                run_command([colab, "status", "-s", session], timeout=30, quiet=True)
                require_unused_remote_workspace(colab, session)
                colab_run_docs.update_journal(ROOT, session, state="uploading",
                                             session_disposition="stop after success" if args.stop_after_success else "user-managed")
            else:
                colab_run_docs.update_journal(ROOT, session, state="allocating")
                colab_run_docs.render_index(ROOT)
                print(f"Allocating {args.gpu} session {session}...", flush=True)
                new_command = [colab, "new", "-s", session, "--gpu", args.gpu]
                if args.high_mem:
                    new_command.append("--high-mem")
                run_command(new_command)
                allocated = True
                colab_run_docs.update_journal(ROOT, session, state="uploading", session_disposition="allocated")
            run_command([colab, "upload", "-s", session, str(archive_path), REMOTE_SOURCE])
            print("Running the snapshot on Colab...", flush=True)
            started = True
            colab_run_docs.update_journal(ROOT, session, state="running")
            colab_run_docs.render_index(ROOT)
            completion = watch_remote_run(colab, session, run_name, command=connection_command(
                colab, session, run_name, "--file", str(ROOT / "scripts" / "colab_remote.py"),
                "--timeout", "90000", "--env", f"MDLARC_SPEC={relative_spec.as_posix()}",
                "--env", f"MDLARC_RUN_NAME={run_name}",
                "--env", f"MDLARC_RUN_BRANCH={branch_name(ROOT)}",
                "--env", "MDLARC_REMOTE=1",
                "--env", f"MDLARC_MIN_GPU_MEMORY_GB={args.min_gpu_memory_gb}",
                "--env", f"MDLARC_DRIVE={int(drive)}",
                "--env", f"MDLARC_DRIVE_AUTH={'shared' if drive else 'none'}",
                "--env", f"MDLARC_TASK_VARIATIONS={int(task_variations)}",
                "--env", f"MDLARC_PRETRAIN_DRIVE_ID={args.pretrain_drive_id or ''}",
                "--env", f"MDLARC_PRETRAIN_NAME={args.pretrain_name}",
                "--env", f"MDLARC_PRETRAIN_SIZE={args.pretrain_size or ''}",
                "--env", f"MDLARC_PRETRAIN_SHA256={args.pretrain_sha256 or ''}",
                "--env", f"MDLARC_CHECKPOINT_MOUNT={args.checkpoint_mount or ''}",
                "--env", f"MDLARC_FINETUNE_TARGET={args.finetune_target or ''}",
                "--env", f"MDLARC_EVAL_TARGET={args.eval_target or ''}",
                "--env", f"MDLARC_SESSION={session}",
            ))
            if completion.get("finalized") is not True:
                raise RuntimeError("Run lacks a finalized completion marker; keep the VM for manual recovery")
            colab_run_docs.update_journal(ROOT, session, state="downloading_results")
            status = recover_results(colab, session, destination)
            recovered = True
            print(f"Results saved to {destination}", flush=True)
            if not status["ok"] or (drive and not status.get("backups_complete", False)):
                colab_run_docs.update_journal(ROOT, session, state="needs_recovery" if drive and not status.get("backups_complete", False) else "failed")
                print(f"Remote run failed: {status.get('error') or 'Drive backups are incomplete'}", file=sys.stderr)
                return 1
            colab_run_docs.update_journal(ROOT, session, state="completed")
            return 0
    except BaseException as exc:
        colab_run_docs.update_journal(
            ROOT, session, state="unavailable" if isinstance(exc, colab_connection.SessionUnavailable) else ("needs_recovery" if started else "launch_failed"),
            launcher_error=f"{type(exc).__name__}: {redact_runtime_credentials(exc)}",
        )
        raise
    finally:
        backup_safe = recovered and status is not None and status.get("backups_complete", not drive)
        stop_shared = shared and args.stop_after_success and backup_safe and status.get("ok", False)
        stop_error = None
        if recovered and status and status.get("drive_folder_id"):
            try:
                sync_records_to_drive(colab, session, status)
            except Exception as exc:
                stop_shared = False
                colab_run_docs.update_journal(ROOT, session, state="needs_recovery",
                    records_backup_error=f"{type(exc).__name__}: {redact_runtime_credentials(exc)}")
                print("Run records could not be verified on Drive; keep the VM for recovery.", file=sys.stderr)
                stop_error = exc
        if (allocated and (backup_safe or not started)) or stop_shared:
            try:
                run_command([colab, "stop", "-s", session], timeout=30)
                colab_run_docs.update_journal(ROOT, session, session_disposition="stopped")
                print(f"Colab session {session} stopped.", flush=True)
            except Exception as exc:
                colab_run_docs.update_journal(ROOT, session, session_disposition="stop_failed")
                print(f"Could not stop Colab session {session}: {redact_runtime_credentials(exc)}", file=sys.stderr)
                if stop_shared:
                    stop_error = exc
        elif isinstance(sys.exc_info()[1], colab_connection.SessionUnavailable):
            colab_run_docs.update_journal(ROOT, session, session_disposition="unavailable")
            print(f"Session {session} is unavailable. Existing Drive backups and local records were preserved.", file=sys.stderr)
        elif allocated:
            colab_run_docs.update_journal(ROOT, session, session_disposition="retained for recovery")
            print(
                f"Results or Drive backups need recovery. Session {session} was kept; "
                f"stop it with `{colab_accounts.cli_hint()} stop -s {session}` to end GPU usage.",
                file=sys.stderr,
            )
        elif shared and args.stop_after_success:
            colab_run_docs.update_journal(ROOT, session, session_disposition="retained for recovery")
            print(f"Session {session} was kept because the run did not succeed or artifact recovery did not verify.", file=sys.stderr)
        elif shared:
            print(f"Session {session} is user-managed; verify Drive backups before stopping it.", file=sys.stderr)
        try:
            colab_run_docs.render_index(ROOT)
        finally:
            locks.close()
        if stop_error is not None:
            raise RuntimeError(
                f"Automatic shutdown or record backup failed; inspect the journal and Drive records before stopping {session}."
            ) from stop_error


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        print(f"Colab launch failed: {redact_runtime_credentials(exc)}", file=sys.stderr)
        raise SystemExit(1) from None
