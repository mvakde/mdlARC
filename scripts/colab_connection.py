#!/usr/bin/env python3
"""Submit once or reconnect to an existing Colab kernel without re-execution.

Run this local helper with the Colab CLI's Python interpreter. Credentials stay
in the CLI's credential store; saved identities and status files contain none.
"""

import argparse
from contextlib import contextmanager
from datetime import UTC, datetime
import fcntl
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlencode, urlsplit, urlunsplit
import uuid

if __package__:
    from . import colab_accounts
else:
    import colab_accounts


class SessionUnavailable(RuntimeError):
    """The authenticated server no longer lists the original assignment."""


class KernelUnavailable(RuntimeError):
    """The original VM exists, but the original kernel does not."""


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".connection-", delete=False) as output:
            temporary = Path(output.name)
            os.chmod(temporary, 0o600)
            json.dump(value, output, indent=2)
            output.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def session_lock(path: Path):
    """Locks are released by the OS even if a watcher/terminal crashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another watcher is already using this session") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@contextmanager
def prevent_idle_sleep():
    """Keep the Mac awake only while this local launcher/watcher exists."""
    executable = shutil.which("caffeinate") if sys.platform == "darwin" else None
    process = None
    if executable:
        try:
            process = subprocess.Popen([executable, "-i", "-w", str(os.getpid())],
                                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, start_new_session=True)
            print("[colab-watch] Mac idle sleep is prevented while this watcher runs.", flush=True)
        except OSError:
            print("[colab-watch] Could not start caffeinate; keep this computer awake during monitoring.", flush=True)
    try:
        yield
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def cli_python(colab: str) -> str:
    launcher = Path(colab).resolve()
    first = launcher.read_text(encoding="utf-8").splitlines()[0]
    interpreter = Path(first.removeprefix("#!"))
    if not first.startswith("#!/") or not interpreter.is_file():
        raise RuntimeError("Colab CLI must have an absolute Python shebang; reinstall with uv tool install google-colab-cli")
    return str(interpreter)


def watch_parent(parent_pid: int, status_path: Path, *, interval: float = 1) -> None:
    """An abruptly closed launcher must not leave an orphan holding its socket.

Only this local helper exits. The OS releases its socket/lock; no interrupt or
shutdown request is sent to the remote kernel. Check parent identity rather
than merely testing whether a possibly reused PID exists.
"""
    def supervise():
        while True:
            time.sleep(interval)
            if os.getppid() != parent_pid:
                try:
                    status = json.loads(status_path.read_text()) if status_path.exists() else {}
                    atomic_json(status_path, {**status, "state": "watcher_closed", "local_pid": os.getpid(),
                                              "updated_at": datetime.now(UTC).isoformat()})
                finally:
                    os._exit(5)

    threading.Thread(target=supervise, daemon=True).start()


def validate_identity(identity: dict, session: str, run_name: str) -> None:
    if identity.get("version") != 1 or identity.get("session") != session or identity.get("run_name") != run_name:
        raise ValueError("Saved kernel identity belongs to a different run or session")
    for key in ("endpoint", "kernel_id", "ws_session_id", "execution_id"):
        if not isinstance(identity.get(key), str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identity[key]):
            raise ValueError(f"Invalid saved {key}")


class ColabBackend:
    """Only the initial launch may create a kernel; attaching never does."""

    def __init__(self, session: str):
        import requests
        from colab_cli.common import state

        self.state = state
        self.session = session
        self.http = requests.Session()
        # Bound credential refresh and assignment requests too, not just VM I/O.
        authenticated = state.client.session
        original_request = authenticated.request

        def bounded_request(*args, **kwargs):
            kwargs.setdefault("timeout", 15)
            return original_request(*args, **kwargs)

        authenticated.request = bounded_request

    def local_session(self):
        return self.state.store.get(self.session)

    def assignment(self, endpoint: str):
        assignment = next((item for item in self.state.client.list_assignments() if item.endpoint == endpoint), None)
        if assignment is None:
            raise SessionUnavailable("Colab no longer lists the original VM assignment")
        proxy = assignment.runtime_proxy_info
        if urlsplit(proxy.url).scheme != "https":
            raise ValueError("Colab returned a non-HTTPS runtime URL")
        return proxy

    def request(self, proxy, method: str, path: str, **kwargs):
        response = self.http.request(
            method, proxy.url.rstrip("/") + path,
            headers={"X-Colab-Runtime-Proxy-Token": proxy.token, "X-Colab-Client-Agent": "colab-cli"},
            params={"colab-runtime-proxy-token": proxy.token}, timeout=15, **kwargs,
        )
        if response.status_code == 404:
            raise KernelUnavailable("The original kernel is unavailable; no replacement was created")
        response.raise_for_status()
        return response.json()

    def existing_kernel(self, proxy, kernel_id: str) -> dict:
        """Retry only the read-only lookup, never kernel creation or execution."""
        import requests
        for attempt in range(3):
            try:
                return self.request(proxy, "GET", f"/api/kernels/{kernel_id}")
            except (requests.Timeout, requests.ConnectionError):
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    def create_identity(self, run_name: str, code_hash: str, *,
                        expected_endpoint: str | None = None,
                        expected_kernel: str | None = None) -> dict:
        local = self.local_session()
        if local is None:
            raise SessionUnavailable("No local Colab session record; create and authorize a new session before launch")
        if expected_endpoint is not None and (local.endpoint != expected_endpoint or local.kernel_id != expected_kernel):
            raise ValueError("Original VM or kernel changed; refusing the initial submission")
        proxy = self.assignment(local.endpoint)
        kernel_id = local.kernel_id
        if kernel_id:
            kernel = self.existing_kernel(proxy, kernel_id)
            if expected_endpoint is not None and (kernel.get("id") != expected_kernel or kernel.get("execution_state") != "idle"):
                raise ValueError("Original kernel is not idle; inspect before submission")
        else:
            kernel_id = self.request(proxy, "POST", "/api/kernels", json={"name": "python3"})["id"]
            local.kernel_id = kernel_id
            self.state.store.add(local)
        identity = {"version": 1, "session": self.session, "run_name": run_name,
                    "colab_account": colab_accounts.selected_account(),
                    "colab_email": colab_accounts.ACCOUNTS[colab_accounts.selected_account()],
                    "endpoint": local.endpoint, "kernel_id": kernel_id,
                    "ws_session_id": uuid.uuid4().hex, "execution_id": uuid.uuid4().hex,
                    "code_sha256": code_hash, "submitted": False,
                    "created_at": datetime.now(UTC).isoformat()}
        validate_identity(identity, self.session, run_name)
        return identity

    def connection(self, identity: dict) -> tuple[str, dict]:
        # Refresh from the server on every attach/reconnect, including after the
        # old CLI alias was pruned. Never assign(), restart(), or start_kernel().
        proxy = self.assignment(identity["endpoint"])
        kernel = self.request(proxy, "GET", f"/api/kernels/{identity['kernel_id']}")
        if kernel.get("id") != identity["kernel_id"]:
            raise KernelUnavailable("Kernel identity changed; refusing to connect")
        parsed = urlsplit(proxy.url.rstrip("/") + f"/api/kernels/{identity['kernel_id']}/channels")
        query = urlencode({"session_id": identity["ws_session_id"], "token": proxy.token,
                           "colab-runtime-proxy-token": proxy.token})
        url = urlunsplit(("wss", parsed.netloc, parsed.path, query, ""))
        return url, {"X-Colab-Runtime-Proxy-Token": proxy.token, "X-Colab-Client-Agent": "colab-cli"}


class KernelConnection:
    def __init__(self, backend, identity_path: Path, identity: dict, *, code: str | None = None,
                 socket_factory=None, timeout: float = 90000, max_retries: int = 8,
                 sleep=time.sleep, clock=time.monotonic, disconnect_after: float | None = None,
                 disconnect_duration: float = 0):
        if socket_factory is None:
            import websocket
            websocket.setdefaulttimeout(15)
            socket_factory = websocket.WebSocketApp
        self.backend, self.identity_path, self.identity = backend, identity_path, identity
        self.code, self.socket_factory = code, socket_factory
        self.timeout, self.max_retries, self.sleep, self.clock = timeout, max_retries, sleep, clock
        self.status_path = identity_path.with_suffix(".status.json")
        self.finished = self.failed = self.received = False
        self.connections = 0
        self.disconnect_after = disconnect_after
        self.disconnect_duration = disconnect_duration
        self.fault_used = False
        self.fault_pending = False
        self.last_error_type = None

    def status(self, state: str, **fields):
        atomic_json(self.status_path, {"state": state, "connections": self.connections,
                    "submitted": self.identity["submitted"], "local_pid": os.getpid(),
                    "updated_at": datetime.now(UTC).isoformat(),
                    "last_error_type": self.last_error_type, **fields})

    def on_open(self, ws):
        self.connections += 1
        self.status("connected")
        print("[colab-connection] Connected to the original kernel.", flush=True)
        if self.code is not None and not self.identity["submitted"]:
            # Persist BEFORE sending. A crash or failed send has ambiguous
            # delivery; attaching must observe it, never resubmit it.
            self.identity["submitted"] = True
            self.identity["submitted_at"] = datetime.now(UTC).isoformat()
            atomic_json(self.identity_path, self.identity)
            message = {"header": {"msg_id": self.identity["execution_id"],
                       "session": self.identity["ws_session_id"], "username": "mdlarc",
                       "date": datetime.now(UTC).isoformat(), "msg_type": "execute_request", "version": "5.3"},
                       "parent_header": {}, "metadata": {}, "channel": "shell",
                       "content": {"code": self.code, "silent": False, "store_history": True,
                                   "user_expressions": {}, "allow_stdin": False, "stop_on_error": True}, "buffers": []}
            ws.send(json.dumps(message))
            self.status("connected")

    def on_message(self, ws, raw):
        # This client negotiates the ordinary JSON Jupyter subprotocol.
        if isinstance(raw, bytes):
            return
        message = json.loads(raw)
        if message.get("parent_header", {}).get("msg_id") != self.identity["execution_id"]:
            return
        self.received = True
        kind = message.get("msg_type") or message.get("header", {}).get("msg_type")
        content = message.get("content", {})
        if kind == "stream":
            print(content.get("text", ""), end="", flush=True)
        elif kind == "error":
            # Avoid serializing full remote traceback/code/credentials.
            print(f"[colab-connection] Remote execution reported {content.get('ename', 'an error')}.", flush=True)
        elif kind == "execute_reply":
            self.failed = content.get("status") != "ok"
            self.finished = True
            self.status("job_failed" if self.failed else "completed")
            ws.close()

    def on_error(self, ws, error):
        # Exception strings may contain token-bearing URLs or HTTP headers.
        self.last_error_type = type(error).__name__
        self.status("disconnected", error_type=type(error).__name__)

    def run(self) -> int:
        started = self.clock()
        retries = 0
        while self.clock() - started < self.timeout:
            timer = fault = None
            self.received = False
            try:
                url, headers = self.backend.connection(self.identity)
                ws = self.socket_factory(url, header=headers, on_open=self.on_open,
                                         on_message=self.on_message, on_error=self.on_error)
                remaining = self.timeout - (self.clock() - started)
                timer = threading.Timer(remaining, ws.close)
                timer.daemon = True
                timer.start()
                if self.disconnect_after is not None and not self.fault_used:
                    self.fault_used = True
                    def sever_stream():
                        self.fault_pending = True
                        self.status("test_disconnect")
                        print("[colab-connection] Test: closing ONLY the local WebSocket.", flush=True)
                        ws.close()
                    fault = threading.Timer(self.disconnect_after, sever_stream)
                    fault.daemon = True
                    fault.start()
                ws.run_forever(ping_interval=20, ping_timeout=10, reconnect=0)
                if self.finished:
                    return 1 if self.failed else 0
                if self.received:
                    retries = 0
            except SessionUnavailable:
                self.status("unavailable")
                print("[colab-connection] Original VM assignment is absent. No new VM or job was started.", flush=True)
                return 2
            except KernelUnavailable:
                self.status("kernel_unavailable")
                print("[colab-connection] Original kernel is unavailable. Saved-file monitoring can still recover results.", flush=True)
                return 3
            except Exception as exc:
                self.last_error_type = type(exc).__name__
                self.status("disconnected", error_type=type(exc).__name__)
            finally:
                if timer is not None:
                    timer.cancel()
                if fault is not None:
                    fault.cancel()
            retries += 1
            if retries > self.max_retries:
                self.status("retries_exhausted")
                print("[colab-connection] Reconnect attempts exhausted. Continue through the saved-file watcher; training was not resubmitted.", flush=True)
                return 4
            delay = min(30, 2 ** (retries - 1), max(0, self.timeout - (self.clock() - started)))
            if self.fault_pending:
                delay = min(self.disconnect_duration, max(0, self.timeout - (self.clock() - started)))
                self.fault_pending = False
            self.status("reconnecting", retry=retries)
            print(f"[colab-connection] Connection lost; reconnecting to the same job in {delay:g}s (attempt {retries}/{self.max_retries}).", flush=True)
            self.sleep(delay)
        self.status("timeout")
        return 4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--account", choices=colab_accounts.ACCOUNTS)
    parser.add_argument("--run-name")
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--file", type=Path)
    parser.add_argument("--expected-endpoint")
    parser.add_argument("--expected-kernel-id")
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--timeout", type=float, default=90000)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--parent-pid", type=int, help="Exit this local helper if its launcher closes")
    parser.add_argument("--test-disconnect-after", type=float, help="Transport test only: close the local socket once after this many seconds")
    parser.add_argument("--test-disconnect-duration", type=float, default=0,
                        help="Transport test only: delay the first reconnect for this many seconds")
    args = parser.parse_args()
    if bool(args.expected_endpoint) != bool(args.expected_kernel_id) or (args.expected_endpoint and not args.file):
        parser.error("Expected VM and kernel must be supplied together for an initial submission")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.session) or args.session in {".", ".."}:
        parser.error("Invalid session name")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if (not math.isfinite(args.test_disconnect_duration) or args.test_disconnect_duration < 0 or
            (args.test_disconnect_after is not None and
             (not math.isfinite(args.test_disconnect_after) or args.test_disconnect_after <= 0))):
        parser.error("Invalid test disconnect timing")
    if args.parent_pid is not None:
        if args.parent_pid <= 0:
            parser.error("--parent-pid must be positive")
        watch_parent(args.parent_pid, args.identity.with_suffix(".status.json"))
    # Never enable the CLI's debug logger: it writes authentication headers.
    logging.getLogger().setLevel(logging.CRITICAL)
    saved_identity = json.loads(args.identity.read_text()) if args.identity.exists() else None
    account = colab_accounts.record_account(saved_identity) if saved_identity is not None else colab_accounts.selected_account(args.account)
    if saved_identity is not None and args.account is not None and colab_accounts.selected_account(args.account) != account:
        raise ValueError("Selected Colab account differs from the saved kernel identity")
    os.environ[colab_accounts.ACCOUNT_ENV] = account
    if account == "2":
        colab_accounts.configure_cli(account)
    backend = ColabBackend(args.session)
    if args.probe:
        identity = json.loads(args.identity.read_text()) if args.identity.exists() else {}
        local = backend.local_session()
        endpoint = identity.get("endpoint") or (local.endpoint if local else None)
        assignments = backend.state.client.list_assignments()
        available = any(item.endpoint == endpoint for item in assignments) if endpoint else (False if not assignments else None)
        print(json.dumps({"available": available}))
        return 0
    if not args.run_name:
        parser.error("--run-name is required for execution or attachment")
    with session_lock(args.identity.with_suffix(".socket.lock")):
        code = None
        if args.file:
            if args.identity.exists():
                raise ValueError("Execution identity already exists; attach without --file instead of submitting again")
            environment = {}
            for entry in args.env:
                key, separator, value = entry.partition("=")
                if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                    raise ValueError("Invalid --env entry")
                environment[key] = value
            code = "import os\nos.chdir('/content')\nos.environ.update(" + repr(environment) + ")\n" + args.file.read_text()
            identity = backend.create_identity(args.run_name, hashlib.sha256(code.encode()).hexdigest(),
                                               expected_endpoint=args.expected_endpoint,
                                               expected_kernel=args.expected_kernel_id)
            atomic_json(args.identity, identity)
        else:
            identity = json.loads(args.identity.read_text())
            validate_identity(identity, args.session, args.run_name)
            if not identity.get("submitted"):
                raise ValueError("No submitted execution is recorded; refusing to start a job during attachment")
        return KernelConnection(backend, args.identity, identity, code=code, timeout=args.timeout,
                                disconnect_after=args.test_disconnect_after,
                                disconnect_duration=args.test_disconnect_duration).run()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Never expose credential-bearing backend exception text.
        print(f"[colab-connection] Could not start/attach ({type(exc).__name__}); no execution was retried.", file=sys.stderr)
        raise SystemExit(1) from None
