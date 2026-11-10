import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import colab_connection as connection


def identity():
    return {"version": 1, "session": "test-session", "run_name": "test-run", "endpoint": "original-vm",
            "kernel_id": "original-kernel", "ws_session_id": "original-client", "execution_id": "original-execution",
            "submitted": False}


def message(kind, **content):
    return json.dumps({"header": {"msg_type": kind}, "parent_header": {"msg_id": "original-execution"},
                       "content": content})


def test_disconnect_refreshes_connection_and_submits_exactly_once(tmp_path, capsys):
    calls, sent = [], []

    class Backend:
        def connection(self, saved):
            assert saved["kernel_id"] == "original-kernel"
            calls.append(saved.copy())
            return f"wss://vm/channels?token=secret-{len(calls)}", {"token": f"secret-{len(calls)}"}

    class Socket:
        def __init__(self, url, **callbacks):
            self.callbacks = callbacks

        def send(self, raw):
            # Submission is saved before the first network send.
            assert json.loads((tmp_path / "identity.json").read_text())["submitted"] is True
            sent.append(json.loads(raw))

        def close(self):
            pass

        def run_forever(self, **options):
            assert options == {"ping_interval": 20, "ping_timeout": 10, "reconnect": 0}
            self.callbacks["on_open"](self)
            self.callbacks["on_message"](self, message("stream", text="progress\n"))
            if len(calls) == 1:
                self.callbacks["on_error"](self, RuntimeError("secret-1 token-bearing URL"))
            else:
                self.callbacks["on_message"](self, message("execute_reply", status="ok"))

    saved = identity()
    worker = connection.KernelConnection(Backend(), tmp_path / "identity.json", saved, code="train()",
                                         socket_factory=Socket, sleep=lambda seconds: None)
    assert worker.run() == 0
    assert len(calls) == 2
    assert len(sent) == 1
    assert sent[0]["header"]["msg_id"] == "original-execution"
    assert sent[0]["header"]["session"] == "original-client"
    assert sent[0]["content"]["code"] == "train()"
    status = (tmp_path / "identity.status.json").read_text()
    assert json.loads(status)["connections"] == 2
    assert "secret" not in status + capsys.readouterr().out


def test_ambiguous_send_is_never_repeated_and_attachment_never_executes(tmp_path):
    saved = identity()
    sent = []

    class Socket:
        def send(self, raw):
            sent.append(raw)
            raise OSError("delivery unknown")

    worker = connection.KernelConnection(None, tmp_path / "identity.json", saved, code="train()", socket_factory=object)
    with pytest.raises(OSError):
        worker.on_open(Socket())
    worker.on_open(Socket())
    assert len(sent) == 1
    attached = connection.KernelConnection(None, tmp_path / "identity.json", saved, socket_factory=object)
    attached.on_open(Socket())
    assert len(sent) == 1


@pytest.mark.parametrize("error,expected,state", [
    (connection.SessionUnavailable("gone"), 2, "unavailable"),
    (connection.KernelUnavailable("gone"), 3, "kernel_unavailable"),
])
def test_missing_vm_or_kernel_does_not_allocate_or_execute(tmp_path, error, expected, state):
    class Backend:
        def connection(self, saved):
            raise error

    worker = connection.KernelConnection(Backend(), tmp_path / "identity.json", identity(), code="train()",
                                         socket_factory=lambda *a, **k: pytest.fail("No socket or execution"))
    assert worker.run() == expected
    assert json.loads(worker.status_path.read_text())["state"] == state


def test_reconnect_exhaustion_is_bounded_and_never_logs_credentials(tmp_path, capsys):
    calls, delays = [], []

    class Backend:
        def connection(self, saved):
            calls.append(saved)
            raise RuntimeError("https://vm?token=very-secret")

    worker = connection.KernelConnection(Backend(), tmp_path / "identity.json", identity(), socket_factory=object,
                                         max_retries=3, sleep=delays.append)
    assert worker.run() == 4
    assert len(calls) == 4
    assert delays == [1, 2, 4]
    assert "very-secret" not in capsys.readouterr().out + worker.status_path.read_text()


def test_attach_uses_only_get_and_websocket_with_the_original_identifiers():
    backend = connection.ColabBackend.__new__(connection.ColabBackend)
    calls = []
    backend.assignment = lambda endpoint: SimpleNamespace(url="https://vm.example", token="fresh")
    backend.request = lambda proxy, method, path: calls.append((method, path)) or {"id": "original-kernel"}
    url, headers = backend.connection(identity())
    assert calls == [("GET", "/api/kernels/original-kernel")]
    assert "session_id=original-client" in url
    assert "fresh" in url
    assert headers["X-Colab-Runtime-Proxy-Token"] == "fresh"


def test_saved_identity_cannot_be_used_for_a_different_run():
    with pytest.raises(ValueError, match="different run"):
        connection.validate_identity(identity(), "test-session", "other-run")


def test_only_one_local_watcher_can_own_a_session(tmp_path):
    lock = tmp_path / "watch.lock"
    with connection.session_lock(lock):
        with pytest.raises(RuntimeError, match="Another watcher"):
            with connection.session_lock(lock):
                pytest.fail("Two watchers acquired the lock")
    with connection.session_lock(lock):
        pass


def test_orphan_socket_helper_exits_and_releases_its_lock(tmp_path):
    lock, status = tmp_path / "socket.lock", tmp_path / "status.json"
    code = ("import os,time; from pathlib import Path; from scripts.colab_connection import session_lock,watch_parent; "
            f"lock=Path({str(lock)!r}); status=Path({str(status)!r}); "
            # Simulate a different parent after abrupt watcher termination.
            "guard=session_lock(lock); guard.__enter__(); "
            "watch_parent(os.getppid()+1,status,interval=0.02); time.sleep(10)")
    result = subprocess.run([sys.executable, "-c", code], timeout=5, capture_output=True, text=True)
    assert result.returncode == 5
    assert json.loads(status.read_text())["state"] == "watcher_closed"
    with connection.session_lock(lock):
        pass


def test_mac_idle_guard_stops_only_its_own_caffeinate(monkeypatch):
    calls = []
    class Process:
        def poll(self):
            return None
        def terminate(self):
            calls.append("terminate")
        def wait(self, *, timeout):
            calls.append(("wait", timeout))
    def spawn(command, **kwargs):
        calls.append(command)
        return Process()
    monkeypatch.setattr(connection.sys, "platform", "darwin")
    monkeypatch.setattr(connection.shutil, "which", lambda name: "/usr/bin/caffeinate")
    monkeypatch.setattr(connection.subprocess, "Popen", spawn)
    with pytest.raises(RuntimeError), connection.prevent_idle_sleep():
        raise RuntimeError("watcher closed")
    assert calls == [["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())], "terminate", ("wait", 5)]
