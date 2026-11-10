import json
import stat
from types import SimpleNamespace

import pytest

from scripts import fix_colab_cli


RUNTIME_SOURCE = (
    "class Runtime:\n"
    "    def stdin_handler(self, original_stdin_hook=None):\n"
    + fix_colab_cli.BROKEN_HOOK
    + "        return wrapped_stdin_hook\n"
)


def runtime_with_transport():
    namespace = {}
    exec(compile(fix_colab_cli.repaired_source(RUNTIME_SOURCE), "runtime.py", "exec"), namespace)
    runtime = namespace["Runtime"]()
    replies, history = [], []
    runtime.kernel_client = SimpleNamespace(_manager=SimpleNamespace(
        client=SimpleNamespace(input=replies.append)))
    runtime.session_name = "test-session"
    runtime.history = SimpleNamespace(log_event=lambda *event: history.append(event))
    return runtime, replies, history


def test_verification_code_is_forwarded_once_and_never_logged(monkeypatch):
    runtime, replies, history = runtime_with_transport()
    prompts = []

    def hidden_input(prompt):
        prompts.append(prompt)
        return "test-verification-secret"

    monkeypatch.setattr("getpass.getpass", hidden_input)
    monkeypatch.setattr("builtins.input", lambda *args: pytest.fail("Verification code was echoed"))
    message = {"content": {"prompt": "Enter the verification code: ", "password": False}}
    runtime.stdin_handler()(message)
    assert prompts == [message["content"]["prompt"]]
    assert replies == ["test-verification-secret"]
    assert "test-verification-secret" not in json.dumps(history)
    assert history[-1] == ("test-session", "input_reply", {"sent": True})


def test_plain_input_uses_prompt_text_and_sends_reply(monkeypatch):
    runtime, replies, history = runtime_with_transport()
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or "answer")
    runtime.stdin_handler()({"content": {"prompt": "Enter a value: ", "password": False}})
    assert prompts == ["Enter a value: "]
    assert replies == ["answer"]
    assert "answer" not in json.dumps(history)


def test_password_input_is_hidden(monkeypatch):
    runtime, replies, history = runtime_with_transport()
    monkeypatch.setattr("getpass.getpass", lambda prompt: "test-password")
    monkeypatch.setattr("builtins.input", lambda *args: pytest.fail("Password was echoed"))
    runtime.stdin_handler()({"content": {"prompt": "Password: ", "password": True}})
    assert replies == ["test-password"]
    assert "test-password" not in json.dumps(history)


def test_custom_hook_is_preserved_without_double_reply():
    runtime, replies, history = runtime_with_transport()
    requests = []

    def custom_hook(message):
        requests.append(message)
        runtime.kernel_client._manager.client.input("custom-secret")
        return "custom-result"

    message = {"content": {"prompt": "Enter the verification code: "}}
    assert runtime.stdin_handler(custom_hook)(message) == "custom-result"
    assert requests == [message]
    assert replies == ["custom-secret"]
    assert "custom-secret" not in json.dumps(history)
    assert "custom-result" not in json.dumps(history)


def test_repair_is_atomic_idempotent_and_keeps_original(tmp_path):
    runtime_path = tmp_path / "runtime.py"
    runtime_path.write_text(RUNTIME_SOURCE)
    runtime_path.chmod(0o640)
    assert fix_colab_cli.repair_runtime(runtime_path, check=True)
    assert runtime_path.read_text() == RUNTIME_SOURCE
    assert list(tmp_path.iterdir()) == [runtime_path]
    assert fix_colab_cli.repair_runtime(runtime_path)
    assert runtime_path.with_name("runtime.py.mdlarc-stdin-backup").read_text() == RUNTIME_SOURCE
    assert stat.S_IMODE(runtime_path.stat().st_mode) == 0o640
    assert not fix_colab_cli.repair_runtime(runtime_path)
    assert not fix_colab_cli.repair_runtime(runtime_path, check=True)
    assert not list(tmp_path.glob(".mdlarc-stdin-*"))


def test_unknown_cli_version_is_left_unchanged(tmp_path):
    runtime_path = tmp_path / "runtime.py"
    runtime_path.write_text("print('unrecognized version')\n")
    with pytest.raises(ValueError, match="unrecognized stdin handler"):
        fix_colab_cli.repair_runtime(runtime_path)
    assert runtime_path.read_text() == "print('unrecognized version')\n"
    assert list(tmp_path.iterdir()) == [runtime_path]
