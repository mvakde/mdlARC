#!/usr/bin/env python3
"""Bounded, CPU-only live check of Colab reconnect, reattachment and recovery.

Creates its own fresh CPU VM, runs a three-minute diagnostic once, drops only
the local WebSocket, replaces the local watcher, verifies downloaded results,
then stops only that diagnostic VM. Does not train or access private Drive.
"""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime

try:
    from . import colab_connection, colab_run_docs, run_on_colab
except ImportError:
    import colab_connection
    import colab_run_docs
    import run_on_colab


if __package__:
    from .run_layout import branch_name, results_root, runs_root
else:
    from run_layout import branch_name, results_root, runs_root


ROOT = run_on_colab.ROOT


def diagnostic_source(run_name: str, seconds: int = 180) -> str:
    # Exercise the production archive/manifest functions without environment
    # installation, GPU allocation, training, or Drive authentication.
    source = (ROOT / "scripts/colab_remote.py").read_text().rsplit('if __name__ == "__main__":', 1)[0]
    return source + f'''
run_name = {run_name!r}
run_dir = ROOT / "runs" / run_name
run_dir.mkdir(parents=True, exist_ok=True)
counter = run_dir / "executions.txt"
executions = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(executions))
if executions != 1:
    raise RuntimeError("Diagnostic execution was submitted more than once")
pid = os.getpid()
ticks = {math.ceil(seconds / 5)}
with (run_dir / "console.log").open("w", buffering=1) as console:
    for tick in range(1, ticks + 1):
        # Small, real CPU work; this is a transport test, not a training run.
        digest = hashlib.sha256(b"Colab transport check" * 1024).hexdigest()
        record = json.dumps({{"tick": tick, "pid": pid, "run_name": run_name, "digest": digest}})
        console.write(record + "\\n")
        console.write(f"Epoch {{tick}}/{{ticks}}\\n")
        print(record, flush=True)
        time.sleep(5)
(run_dir / "diagnostic.json").write_text(json.dumps({{"pid": pid, "ticks": ticks, "executions": executions}}))
package_results(run_name, {{"ok": True, "backups_complete": True, "drive_backed": False,
                           "diagnostic": True, "finished_at": datetime.now(UTC).isoformat()}})
'''


def records(console: str) -> list[dict]:
    return [json.loads(line) for line in console.splitlines() if line.startswith('{"tick":')]


def stop_local(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def monitor(colab: str, session: str, run_name: str, job: Path | None) -> int:
    def close_local(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, close_local)
    arguments = [] if job is None else ["--file", str(job), "--env", f"MDLARC_SESSION={session}",
                                      "--test-disconnect-after", "20", "--test-disconnect-duration", "45"]
    command = run_on_colab.connection_command(colab, session, run_name, "--timeout", "300", *arguments)
    lock = results_root(ROOT) / "colab_runs" / "_active" / f"{session}.watch.lock"
    try:
        with colab_connection.session_lock(lock), colab_connection.prevent_idle_sleep():
            manifest = run_on_colab.watch_remote_run(colab, session, run_name, command=command, poll_interval=5)
            colab_connection.atomic_json(results_root(ROOT) / "colab_runs" / "_active" / f"{session}.check-ready.json", manifest)
        return 0
    except KeyboardInterrupt:
        return 130


def live_check(colab: str) -> int:
    session = f"connection-check-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    run_name = "connection_check"
    destination = results_root(ROOT) / "colab_runs" / session
    active = results_root(ROOT) / "colab_runs" / "_active"
    destination.mkdir(parents=True)
    active.mkdir(parents=True, exist_ok=True)
    spec = destination / "spec.json"
    spec.write_text(json.dumps({"run_name": run_name, "diagnostic": "connection", "seconds": 180}) + "\n")
    colab_run_docs.create_note(ROOT, session, run_name, "Validate Colab transport safety on a short CPU VM",
                               "Drop local socket, replace local watcher, recover results, stop this CPU VM")
    colab_run_docs.start_journal(ROOT, session, config=spec.relative_to(ROOT).as_posix(), run_name=run_name,
                               gpu="CPU", high_mem=False, min_gpu_memory_gb=0, spec_path=spec)
    colab_run_docs.update_journal(ROOT, session, diagnostic=True, drive_backed=False, state="allocating")
    colab_run_docs.update_journal(ROOT, session, colab_account=run_on_colab.colab_accounts.selected_account(),
                                 colab_email=run_on_colab.colab_accounts.ACCOUNTS[run_on_colab.colab_accounts.selected_account()])
    job = active / f"{session}.validation.py"
    job.write_text(diagnostic_source(run_name))
    identity_path = run_on_colab.connection_identity(session)
    status_path = identity_path.with_suffix(".status.json")
    vm_log = active / f"{session}.vm.log"
    ready = active / f"{session}.check-ready.json"
    evidence = {"session": session, "cpu_only": True, "drive_tested": False, "passed": False}
    processes, outputs = [], []
    allocated = False
    deadline = time.monotonic() + 420

    def saved_status():
        return json.loads(status_path.read_text()) if status_path.exists() else {}

    def saved_records():
        return records(vm_log.read_text()) if vm_log.exists() else []

    def wait_for(predicate, process):
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            if process.poll() is not None:
                raise RuntimeError("Diagnostic watcher ended early; inspect its saved monitor log")
            time.sleep(1)
        raise TimeoutError("CPU connection check exceeded its seven-minute bound")

    def spawn(job_path=None):
        log = (active / f"{session}.monitor-{len(processes) + 1}.log").open("w")
        outputs.append(log)
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--monitor", "--session", session,
                   "--run-name", run_name]
        if job_path is not None:
            command += ["--job", str(job_path)]
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(process)
        return process

    try:
        print(f"[connection-check] Creating CPU-only session {session}.", flush=True)
        run_on_colab.run_command([colab, "new", "-s", session], timeout=60, quiet=True)
        allocated = True
        colab_run_docs.update_journal(ROOT, session, state="running", session_disposition="diagnostic CPU VM")
        first = spawn(job)
        wait_for(lambda: saved_status().get("state") == "reconnecting", first)
        offline = wait_for(lambda: saved_records() if saved_records() and saved_records()[-1]["tick"] >= 9 else None, first)
        assert saved_status().get("connections") == 1, "Socket reconnected before offline file check"
        evidence["progress_during_socket_outage"] = offline[-1]
        print("[connection-check] Saved-file watcher observed CPU progress while the socket was disconnected.", flush=True)
        resumed = wait_for(lambda: saved_status() if saved_status().get("connections", 0) >= 2 else None, first)
        evidence["automatic_reconnect"] = resumed
        identity = json.loads(identity_path.read_text())
        before_close = saved_records()[-1]
        stop_local(first)
        time.sleep(10)
        console, manifest = run_on_colab.read_remote_snapshot(colab, session, run_name)
        continued = records(console or "")[-1]
        assert continued["tick"] > before_close["tick"] and continued["pid"] == before_close["pid"]
        evidence["progress_without_local_watcher"] = continued
        print("[connection-check] Remote job continued after the local watcher closed; attaching a new watcher.", flush=True)
        second = spawn()
        attached = wait_for(lambda: saved_status() if saved_status().get("state") == "connected" and
                            saved_status().get("local_pid") != resumed["local_pid"] else None, second)
        assert json.loads(identity_path.read_text()) == identity, "Reattachment changed execution identity"
        evidence["replacement_watcher"] = attached
        wait_for(lambda: ready.exists(), second)
        assert second.wait(timeout=10) == 0
        assert json.loads(ready.read_text())["finalized"] is True
        status = run_on_colab.recover_results(colab, session, destination)
        run_dir = destination / "runs" / run_name
        final = json.loads((run_dir / "diagnostic.json").read_text())
        recovered = records((run_dir / "console.log").read_text())
        assert status["ok"] and final["executions"] == int((run_dir / "executions.txt").read_text()) == 1
        assert [record["tick"] for record in recovered] == list(range(1, final["ticks"] + 1))
        assert {record["pid"] for record in recovered} == {offline[-1]["pid"]} == {final["pid"]}
        evidence.update(single_execution=True, same_kernel_pid=final["pid"], archive_checksums_verified=True,
                        all_ticks_recovered=final["ticks"], passed=True)
        colab_run_docs.update_journal(ROOT, session, state="completed")
        print("[connection-check] Same kernel, exactly one execution, complete progress and verified result checksums.", flush=True)
    except Exception as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        colab_run_docs.update_journal(ROOT, session, state="validation_failed", launcher_error=evidence["error"])
        print(f"[connection-check] Validation failed ({type(exc).__name__}); saved evidence: {destination}.", flush=True)
    finally:
        for process in processes:
            stop_local(process)
        for output in outputs:
            output.close()
        if allocated:
            # This VM contains only this disposable CPU diagnostic, no training
            # checkpoints or private Drive data. Never stop any other session.
            try:
                run_on_colab.run_command([colab, "stop", "-s", session], timeout=30, quiet=True)
                absent = run_on_colab.session_assignment_available(colab, session) is False
                evidence["shutdown_verified"] = absent
                if not absent:
                    raise RuntimeError("CPU VM shutdown could not be confirmed")
                colab_run_docs.update_journal(ROOT, session, session_disposition="stopped")
                print("[connection-check] Diagnostic CPU VM shutdown verified.", flush=True)
            except Exception as exc:
                evidence.update(passed=False, shutdown_error=type(exc).__name__)
                colab_run_docs.update_journal(ROOT, session, session_disposition="stop_failed")
                print(f"[connection-check] CPU VM shutdown needs attention: colab stop -s {session}", flush=True)
        colab_connection.atomic_json(destination / "validation.json", evidence)
        colab_run_docs.render_index(ROOT)
    print(f"[connection-check] {'PASS' if evidence['passed'] else 'FAIL'}: {destination / 'validation.json'}", flush=True)
    return 0 if evidence["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitor", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--account", choices=run_on_colab.colab_accounts.ACCOUNTS)
    parser.add_argument("--session", help=argparse.SUPPRESS)
    parser.add_argument("--run-name", help=argparse.SUPPRESS)
    parser.add_argument("--job", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.environ[run_on_colab.colab_accounts.ACCOUNT_ENV] = run_on_colab.colab_accounts.selected_account(args.account)
    colab = shutil.which("colab")
    if not colab:
        parser.error("Install the Google Colab CLI first")
    if args.monitor:
        if not args.session or not args.run_name:
            parser.error("Monitor mode needs session and run name")
        return monitor(colab, args.session, args.run_name, args.job)
    with colab_connection.prevent_idle_sleep():
        return live_check(colab)


if __name__ == "__main__":
    raise SystemExit(main())
