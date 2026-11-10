#!/usr/bin/env python3
"""Tiny, local-only Colab monitor. No framework, background polling or model calls."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import select
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import quote, urlsplit
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.colab_accounts import ACCOUNTS
from scripts.colab_monitor_data import FIELDS, compact_score, measure_speed, progress_value

UI = Path(__file__).with_name("colab_monitor_ui")
WORKER = Path(__file__).with_name("colab_monitor_backend.py")


class MonitorError(Exception):
    pass


def load_json(path, limit=2 * 1024 * 1024):
    try:
        if path.stat().st_size > limit:
            return {}
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def cli_interpreter():
    colab = shutil.which("colab")
    if not colab:
        raise MonitorError("Install the Colab CLI first; see scripts/README_colab.md.")
    first = Path(colab).resolve().read_text().splitlines()[0]
    python = first.removeprefix("#!")
    if not first.startswith("#!/") or not Path(python).is_file():
        raise MonitorError("The Colab CLI needs an absolute Python shebang.")
    return python


class Worker:
    """A lazy account process: at most one per account, exits after 20s idle."""

    def __init__(self, python, account):
        self.python, self.account = python, account
        self.process = None
        self.lock = threading.Lock()

    def close(self):
        process, self.process = self.process, None
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def call(self, request):
        with self.lock:
            # An idle worker can exit between poll() and write(). Read-only
            # operations may safely be retried in a new worker in that case.
            for attempt in range(2):
                if self.process is None or self.process.poll() is not None:
                    self.process = subprocess.Popen(
                        [self.python, "-u", str(WORKER), self.account], cwd=ROOT,
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL, text=True, bufsize=1)
                    # Block in waitpid, not a polling loop. Reap idle workers
                    # immediately so even defunct child entries do not linger.
                    threading.Thread(target=self.process.wait, daemon=True,
                                     name=f"colab-account-{self.account}-reaper").start()
                try:
                    self.process.stdin.write(json.dumps(request) + "\n")
                    self.process.stdin.flush()
                    if not select.select([self.process.stdout], [], [], 55)[0]:
                        self.close()
                        raise MonitorError("Colab request timed out. Retry when connectivity returns.")
                    line = self.process.stdout.readline(1024 * 1024)
                    if not line:
                        raise BrokenPipeError
                    result = json.loads(line)
                    if not result.get("ok"):
                        raise MonitorError(result.get("error", "Colab read failed."))
                    return result["data"]
                except (BrokenPipeError, OSError):
                    self.close()
                    if attempt:
                        raise MonitorError("The Colab account worker exited; retry the check.") from None
                except ValueError:
                    self.close()
                    raise MonitorError("The Colab worker returned an invalid response.") from None


def local_records(root):
    """Index only small journals/identities, never recurse through run artifacts."""
    records = {}
    for directory in (root / "results").glob("*/colab_runs/_active"):
        for path in directory.glob("*.json"):
            journal = load_json(path, 128 * 1024)
            session = journal.get("session")
            if not session or session != path.stem or not journal.get("run_name"):
                continue
            account = str(journal.get("colab_account", "1"))
            if account not in ACCOUNTS:
                continue
            identity = load_json(directory / f"{session}.connection.json", 128 * 1024)
            hint = {k: journal.get(k) for k in ("session", "run_name", "branch", "state")}
            hint["endpoint"] = identity.get("endpoint")
            hint["cache_dir"] = str(directory.parent / session)
            records[(account, session)] = hint
            if identity.get("endpoint"):
                records[(account, identity["endpoint"])] = hint
    return records


class Monitor:
    def __init__(self, root=ROOT, workers=None):
        self.root = root
        self.workers = workers or {a: Worker(cli_interpreter(), a) for a in ACCOUNTS}
        self.lock = threading.Lock()
        self.list_lock = threading.Lock()
        self.list_cache = None
        self.list_time = 0
        self.hints = {}
        self.samples = {}
        self.latest = {}
        self.references = {}

    def close(self):
        for worker in self.workers.values():
            worker.close()

    def listing(self, force=False):
        with self.list_lock:
            if not force and self.list_cache is not None and time.monotonic() - self.list_time < 15:
                return self.list_cache
            records = local_records(self.root)

            def one(account):
                try:
                    runs = self.workers[account].call({"action": "list"})
                    for run in runs:
                        alias = records.get((account, run["session"])) or {}
                        if alias.get("endpoint") and alias["endpoint"] != run["endpoint"]:
                            alias = {}
                        hint = records.get((account, run["endpoint"])) or alias
                        run.update(account=account, branch=hint.get("branch"), run_name=hint.get("run_name"),
                                   recorded_state=hint.get("state"), cached=self.latest.get((account, run["endpoint"]), {}))
                        self.hints[(account, run["endpoint"])] = hint
                    return {"account": account, "email": ACCOUNTS[account], "runs": runs}
                except MonitorError as exc:
                    # An unreachable account is unknown, never an empty success.
                    return {"account": account, "email": ACCOUNTS[account], "error": str(exc), "runs": []}

            with ThreadPoolExecutor(max_workers=2) as pool:
                accounts = list(pool.map(one, ACCOUNTS))
            live = {(a["account"], r["endpoint"]) for a in accounts for r in a["runs"]}
            with self.lock:
                pending = {(s["account"], s["endpoint"]) for s in self.samples.values()}
                keep = live | pending
                self.hints = {k: v for k, v in self.hints.items() if k in keep}
                self.latest = {k: v for k, v in self.latest.items() if k in keep}
            self.list_cache = {"accounts": accounts, "observed_at": datetime.now(UTC).isoformat()}
            self.list_time = time.monotonic()
            return self.list_cache

    def hint(self, account, endpoint):
        if account not in ACCOUNTS or (account, endpoint) not in self.hints:
            raise MonitorError("Refresh the VM list before checking this VM.")
        hint = self.hints[(account, endpoint)]
        spec = load_json(Path(hint["cache_dir"]) / "spec.json") if hint.get("cache_dir") else {}
        return {k: v for k, v in {**hint, "spec": spec}.items() if k != "cache_dir"}

    def compare(self, account, endpoint, result):
        hint = self.hints[(account, endpoint)]
        if not hint.get("cache_dir"):
            return []
        branch_root = Path(hint["cache_dir"]).parent.parent
        path = branch_root / "status_checks" / hint["session"] / "comparison_references.json"
        try:
            stamp = path.stat().st_mtime_ns
        except OSError:
            return []
        if path not in self.references or self.references[path][0] != stamp:
            source = load_json(path)
            # Cache only the few scalars needed by the UI, not full specs,
            # prediction lists, task IDs or raw console lines.
            rows = []
            for ref in source.get("references", []):
                rows.append({"label": ref.get("label", ref.get("session")),
                             "session": ref.get("session"),
                             "score": compact_score(ref.get("score") or {}),
                             "epoch_metrics": {str(e): {k: v[k] for k in FIELDS if k in v}
                                               for e, v in ref.get("epoch_metrics", {}).items()}})
            if len(self.references) >= 8:
                self.references.pop(next(iter(self.references)))
            self.references[path] = (stamp, rows, source.get("comparison_caveats", []))
        _, rows, caveats = self.references[path]
        epoch = result.get("progress", {}).get("epoch_metrics", {}).get("epoch")
        return {"epoch": epoch, "caveats": caveats,
                "rows": [{"label": r["label"], "session": r["session"], "score": r["score"],
                          "metrics": r["epoch_metrics"].get(str(epoch), {})} for r in rows]}

    def check(self, account, endpoint, kind):
        if kind not in {"performance", "metrics"}:
            raise MonitorError("Choose a performance or training-metrics check.")
        hint = self.hint(account, endpoint)
        before = time.monotonic()
        result = self.workers[account].call({"action": kind, "endpoint": endpoint, "hint": hint})
        after = time.monotonic()
        if kind == "metrics":
            result["comparisons"] = self.compare(account, endpoint, result)
        elif result.get("progress") and progress_value(result["progress"])[0] is not None:
            sample_id = secrets.token_hex(12)
            with self.lock:
                now = time.monotonic()
                self.samples = {k: v for k, v in self.samples.items() if now - v["time"] < 600}
                if len(self.samples) >= 64:
                    raise MonitorError("Too many active speed samples. Wait for one to finish.")
                self.samples[sample_id] = {"account": account, "endpoint": endpoint,
                                           "time": result.get("sample_time", (before + after) / 2),
                                           "progress": result["progress"]}
            result.update(sample_id=sample_id, sample_seconds=30)
        elif result.get("progress"):
            result["speed"] = {"note": "Speed needs one completed epoch or an evaluation batch counter; GPU snapshot is ready."}
        self.latest.setdefault((account, endpoint), {})[kind] = result
        return result

    def finish(self, sample_id):
        with self.lock:
            first = self.samples.get(sample_id)
        if first is None:
            raise MonitorError("Speed sample expired. Start a new check.")
        if time.monotonic() - first["time"] < 25:
            raise MonitorError("Speed measurement is still collecting its 30-second window.")
        account, endpoint = first["account"], first["endpoint"]
        before = time.monotonic()
        last = self.workers[account].call({"action": "progress", "endpoint": endpoint,
                                           "hint": self.hint(account, endpoint)})
        elapsed = last.get("sample_time", (before + time.monotonic()) / 2) - first["time"]
        speed = measure_speed(first["progress"], last.get("progress", {}), elapsed)
        with self.lock:
            self.samples.pop(sample_id, None)
        cached = self.latest.setdefault((account, endpoint), {}).setdefault("performance", {})
        cached.update(speed=speed, speed_observed_at=last.get("observed_at"))
        return {"speed": speed, "observed_at": last.get("observed_at")}


class Handler(BaseHTTPRequestHandler):
    server_version = "ColabMonitor"

    def log_message(self, *_):
        pass

    def respond(self, status, data, content_type="application/json"):
        if isinstance(data, (dict, list)):
            data = json.dumps(data, separators=(",", ":"), allow_nan=False).encode()
        elif isinstance(data, str):
            data = data.encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(data)

    def valid_origin(self):
        hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        host = self.headers.get("Host")
        origin = self.headers.get("Origin")
        return host in hosts and (not origin or origin in {f"http://{h}" for h in hosts})

    def do_GET(self):
        if not self.valid_origin():
            return self.respond(403, {"error": "Local access only."})
        path = urlsplit(self.path).path
        if path == "/api/health":
            return self.respond(200, {"app": "mdlarc-colab-monitor", "pid": os.getpid(), "root": str(ROOT)})
        files = {"/": ("index.html", "text/html; charset=utf-8"),
                 "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                 "/style.css": ("style.css", "text/css; charset=utf-8")}
        if path not in files:
            return self.respond(404, {"error": "Not found."})
        filename, mime = files[path]
        data = (UI / filename).read_text().replace("{{TOKEN}}", self.server.token)
        self.respond(200, data, mime)

    def do_POST(self):
        if not self.valid_origin() or not secrets.compare_digest(self.headers.get("X-Monitor-Token", ""), self.server.token):
            return self.respond(403, {"error": "Reload the local monitor page."})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 2 or length > 4096:
                raise MonitorError("Invalid request size.")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise MonitorError("Invalid request.")
            path = urlsplit(self.path).path
            if path == "/api/runs":
                result = self.server.monitor.listing(force=body.get("refresh") is True)
            elif path == "/api/check":
                result = self.server.monitor.check(body.get("account"), body.get("endpoint"), body.get("kind"))
            elif path == "/api/speed":
                result = self.server.monitor.finish(body.get("sample_id"))
            else:
                return self.respond(404, {"error": "Not found."})
            self.respond(200, result)
        except (MonitorError, ValueError, TypeError) as exc:
            self.respond(400, {"error": str(exc) if isinstance(exc, MonitorError) else "Invalid request."})
        except Exception:
            self.respond(500, {"error": "Monitor check failed. Retry; no VM state was changed."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--open", action="store_true", help="Open the monitor in your browser")
    args = parser.parse_args()
    monitor = Monitor()
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError:
        # Double-clicking the launcher twice should reuse this app, not leave
        # duplicate servers or steal a port from an unrelated program.
        from urllib.request import urlopen
        url = f"http://127.0.0.1:{args.port}"
        try:
            with urlopen(url + "/api/health", timeout=2) as response:
                existing = json.load(response)
            if existing.get("app") != "mdlarc-colab-monitor" or existing.get("root") != str(ROOT):
                raise ValueError
        except Exception:
            raise SystemExit(f"Port {args.port} is in use. Select another --port.") from None
        if args.open:
            webbrowser.open(url)
        print(f"Already running: {url}")
        return
    server.daemon_threads = True
    server.monitor, server.token = monitor, secrets.token_urlsafe(24)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Colab monitor: {url}\nOnly button presses contact Colab. Ctrl-C stops the local UI.", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.close()
        server.server_close()


if __name__ == "__main__":
    main()
