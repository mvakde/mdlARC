"""Account isolation and safe routing, with no live Colab allocations."""

from contextlib import nullcontext
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import colab_accounts as accounts, colab_connection, colab_run_docs, run_on_colab
from scripts.colab_oauth import authorize, launch
from workflows.vast import choose


@pytest.fixture(autouse=True)
def account_defaults(monkeypatch):
    monkeypatch.setenv(accounts.ACCOUNT_ENV, "1")
    monkeypatch.setenv("MDLARC_RUN_BRANCH", "account-tests")


def save_record(root, session="saved", account="2", suffix=".json"):
    path = root / "results/account-tests/colab_runs/_active" / (session + suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"session": session}
    if account is not None:
        record.update(colab_account=account, colab_email=accounts.ACCOUNTS[account])
    path.write_text(json.dumps(record))
    return path


def test_profiles_preserve_account_one_and_isolate_account_two(tmp_path, monkeypatch):
    monkeypatch.setattr(accounts.Path, "home", lambda: tmp_path)
    assert accounts.profile_dir("1") == tmp_path / ".config/colab-cli"
    assert accounts.profile_dir("2") == tmp_path / ".config/mdlarc-colab-accounts/2"
    assert accounts.selected_account() == "1"
    with accounts.account_context("2"):
        assert accounts.selected_account() == "2"
    assert os.environ[accounts.ACCOUNT_ENV] == "1"


@pytest.mark.parametrize("value", ["0", "3", "../2", "your-drive-account@example.com"])
def test_unknown_account_never_falls_back(value):
    with pytest.raises(ValueError, match="Colab account"):
        accounts.selected_account(value)


def test_cli_commands_route_only_colab_and_preserve_legacy(monkeypatch):
    monkeypatch.setattr(colab_connection, "cli_python", lambda path: "/cli/python")
    arguments = ["/bin/colab", "stop", "-s", "saved"]
    assert accounts.route_command(arguments) == arguments
    with accounts.account_context("2"):
        assert accounts.route_command(arguments) == [
            "/cli/python", str(Path(accounts.__file__).resolve()), "--account", "2", "--", *arguments[1:]]
        assert accounts.route_command([sys.executable, "launch_run.py"]) == [sys.executable, "launch_run.py"]


def test_vm_transport_operations_all_use_account_two(monkeypatch):
    monkeypatch.setenv(accounts.ACCOUNT_ENV, "2")
    monkeypatch.setattr(colab_connection, "cli_python", lambda path: "/cli/python")
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert command[2:5] == ["--account", "2", "--"]
        return SimpleNamespace(returncode=0, stdout="sample_data/\n", stderr="")
    monkeypatch.setattr(subprocess, "run", run)
    run_on_colab.run_command(["/bin/colab", "download", "-s", "saved", "remote", "local"])
    run_on_colab.require_unused_remote_workspace("/bin/colab", "saved")
    assert run_on_colab.read_remote_snapshot("/bin/colab", "saved", "run") == (None, None)
    authorize.cli_call(["/bin/colab", "upload", "-s", "saved", "local", "remote"])
    assert [command[5] for command in calls] == ["download", "ls", "ls", "upload"]


@pytest.mark.parametrize("saved", ["1", "2", None])
def test_saved_owner_wins_over_environment_and_rejects_explicit_switch(tmp_path, monkeypatch, saved):
    save_record(tmp_path, account=saved)
    monkeypatch.setenv(accounts.ACCOUNT_ENV, "2" if saved != "2" else "1")
    expected = saved or "1"
    assert accounts.run_account(tmp_path, "saved") == expected
    with pytest.raises(ValueError, match="belongs to Colab account"):
        accounts.run_account(tmp_path, "saved", "1" if expected == "2" else "2")


def test_conflicting_journal_and_kernel_ownership_is_rejected(tmp_path):
    save_record(tmp_path, account="2")
    save_record(tmp_path, account="1", suffix=".connection.json")
    with pytest.raises(ValueError, match="different accounts"):
        accounts.run_account(tmp_path, "saved")


def test_saved_email_cannot_disagree_with_selected_account():
    with pytest.raises(ValueError, match="email disagree"):
        accounts.record_account({"colab_account": "2", "colab_email": accounts.ACCOUNTS["1"]})


def test_cached_launch_selects_compute_account_without_changing_drive_token(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "ROOT", tmp_path)
    events = []
    monkeypatch.setattr(launch, "delegate", lambda arguments: events.append(
        ("delegate", arguments.copy(), accounts.selected_account())) or 0)
    monkeypatch.setattr(launch, "authorize_session", lambda session, token: events.append(
        ("authorize", session, token, accounts.selected_account())))
    arguments = ["--config", "specs/example.json", "--session", "fresh"]
    assert launch.main([*arguments, "--account", "2"]) == 0
    assert events[0] == ("delegate", [*arguments, "--account", "2", "--dry-run"], "2")
    assert events[1] == ("authorize", "fresh", launch.DEFAULT_TOKEN, "2")
    assert events[2] == ("delegate", [*arguments, "--account", "2"], "2")


def test_wrapper_watch_defers_to_saved_owner_without_authorization(monkeypatch):
    captured = []
    monkeypatch.setattr(launch, "delegate", lambda arguments: captured.append(arguments) or 0)
    monkeypatch.setattr(launch, "authorize_session", lambda *args: pytest.fail("Reinstalled credentials"))
    assert launch.main(["--watch-session", "saved", "--account", "2"]) == 0
    assert captured == [["--watch-session", "saved", "--account", "2"]]


def test_watcher_uses_saved_account_for_all_recovery_operations(tmp_path, monkeypatch):
    save_record(tmp_path)
    monkeypatch.setattr(run_on_colab, "ROOT", tmp_path)
    monkeypatch.setattr(colab_connection, "session_lock", lambda path: nullcontext())
    monkeypatch.setattr(colab_connection, "prevent_idle_sleep", nullcontext)
    operations = []
    monkeypatch.setattr(run_on_colab, "_watch_existing_session", lambda *args, **kwargs:
                        operations.append(accounts.selected_account()) or 0)
    assert run_on_colab.watch_existing_session("colab", "saved") == 0
    assert operations == ["2"]
    assert accounts.selected_account() == "1"
    with pytest.raises(ValueError, match="belongs to"):
        run_on_colab.watch_existing_session("colab", "saved", account="1")
    assert operations == ["2"]


def test_status_refresh_uses_owner_and_rejects_wrong_account_before_cli(tmp_path, monkeypatch):
    save_record(tmp_path)
    monkeypatch.setattr(colab_run_docs.shutil, "which", lambda name: "/bin/colab")
    monkeypatch.setattr(colab_connection, "cli_python", lambda path: "/cli/python")
    monkeypatch.setattr(colab_run_docs, "render_index", lambda root: None)
    calls = []
    monkeypatch.setattr(colab_run_docs.subprocess, "run", lambda command, **kwargs:
                        calls.append(command) or SimpleNamespace(returncode=0, stdout="available"))
    assert colab_run_docs.refresh_session(tmp_path, "saved")["session_check"]["result"] == "session visible to CLI"
    assert calls[0][-6:] == ["--account", "2", "--", "status", "-s", "saved"]
    with pytest.raises(ValueError, match="belongs to"):
        colab_run_docs.refresh_session(tmp_path, "saved", "1")
    assert len(calls) == 1


def test_connection_rejects_wrong_owner_before_constructing_backend(tmp_path, monkeypatch):
    identity = tmp_path / "identity.json"
    identity.write_text(json.dumps({"colab_account": "2", "colab_email": accounts.ACCOUNTS["2"]}))
    monkeypatch.setattr(sys, "argv", ["connection", "--session", "saved", "--identity", str(identity), "--probe", "--account", "1"])
    monkeypatch.setattr(colab_connection, "ColabBackend", lambda *args: pytest.fail("Contacted wrong account"))
    with pytest.raises(ValueError, match="differs from the saved"):
        colab_connection.main()


def test_connection_probe_restores_saved_owner_without_explicit_flag(tmp_path, monkeypatch):
    identity = tmp_path / "identity.json"
    identity.write_text(json.dumps({"colab_account": "2", "colab_email": accounts.ACCOUNTS["2"], "endpoint": "original"}))
    monkeypatch.setattr(sys, "argv", ["connection", "--session", "saved", "--identity", str(identity), "--probe"])
    events = []
    monkeypatch.setattr(accounts, "configure_cli", lambda owner: events.append(owner))
    backend = SimpleNamespace(local_session=lambda: None,
                             state=SimpleNamespace(client=SimpleNamespace(list_assignments=lambda: [SimpleNamespace(endpoint="original")])))
    monkeypatch.setattr(colab_connection, "ColabBackend", lambda *args: backend)
    assert colab_connection.main() == 0
    assert events == ["2"]


def test_native_chooser_offers_both_accounts_and_forwards_selection(monkeypatch):
    monkeypatch.setattr(choose.sys, "platform", "darwin")
    dialogs, calls = [], []
    def dialog(script):
        dialogs.append(script)
        if "Choose experiment workflow" in script:
            return "Colab"
        if "Choose Colab compute account" in script:
            return "Account 2: your-drive-account@example.com"
        if "choose file" in script:
            return str(choose.ROOT / "specs/example.json")
        assert "account 2 (your-drive-account@example.com)" in script
        return "fresh-account2"
    monkeypatch.setattr(choose, "dialog", dialog)
    monkeypatch.setattr(choose.subprocess, "call", lambda command, **kwargs: calls.append(command) or 0)
    assert choose.main([]) == 0
    assert all(email in dialogs[1] for email in accounts.ACCOUNTS.values())
    assert calls[0][-4:] == ["--session", "fresh-account2", "--account", "2"]


def test_cli_chooser_forwards_account_and_vast_rejects_it(monkeypatch):
    calls = []
    monkeypatch.setattr(choose.subprocess, "call", lambda command, **kwargs: calls.append(command) or 0)
    assert choose.main(["--backend", "colab", "--account", "2", "--", "--config", "example.json"]) == 0
    assert calls[0][-2:] == ["--account", "2"]
    with pytest.raises(SystemExit):
        choose.main(["--backend", "vast", "--account", "2", "--", "--config", "example.json"])
    assert len(calls) == 1


@pytest.mark.parametrize("email", ["your-drive-account@example.com", "your-colab-account@example.com", None])
def test_live_identity_verification_is_bounded_and_rejects_other_accounts(monkeypatch, email):
    import google.auth.transport.requests
    calls = []
    class Session:
        def __init__(self, credentials):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def get(self, url, *, timeout):
            calls.append((url, timeout))
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"email": email} if email else {})
    monkeypatch.setattr(google.auth.transport.requests, "AuthorizedSession", Session)
    if email == accounts.ACCOUNTS["2"]:
        accounts.verify_cli_account(object(), "2")
    else:
        with pytest.raises(accounts.AccountError, match="Choose your-drive-account"):
            accounts.verify_cli_account(object(), "2")
    assert calls == [("https://openidconnect.googleapis.com/v1/userinfo", 15)]


def test_installed_cli_account_two_uses_isolated_state_without_network(tmp_path):
    colab = shutil.which("colab")
    if not colab:
        pytest.skip("Optional installed-CLI compatibility check")
    # Exercise the real CLI callback and stores with an offline fake client.
    source = '''
import logging, sys
from pathlib import Path
from types import SimpleNamespace
from scripts import colab_accounts as accounts
from colab_cli import auth, common
from colab_cli import cli
cli.auto_update.run_background_check = lambda: None
directory = Path(sys.argv[1]) / "account2"
accounts.profile_dir = lambda account: directory
verified = []
accounts.verify_cli_account = lambda credentials, account: verified.append(account)
auth.get_credentials = lambda *args, **kwargs: SimpleNamespace(credentials="fake-credential")
class Client:
    def __init__(self, environment, credentials):
        assert verified == ["2"]
    def list_assignments(self):
        return []
common.Client = Client
try:
    accounts.main(["--account", "2", "--", "sessions"])
except SystemExit as result:
    assert result.code == 0
assert common.state.config_path == str(directory / "sessions.json")
assert auth.TOKEN_CONFIG_PATH == str(directory / "token.json")
assert common.state.history.log_dir == str(directory / "history")
assert common.state.settings_store.path == str(directory / "settings.json")
assert directory.stat().st_mode & 0o777 == 0o700
assert logging.getLogger().level == logging.CRITICAL
print("ISOLATED_CLI_OK")
'''
    result = subprocess.run([colab_connection.cli_python(colab), "-c", source, str(tmp_path)],
                            cwd=accounts.ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "ISOLATED_CLI_OK" in result.stdout


def test_installed_cli_wrong_consent_is_rejected_before_saving_token(tmp_path):
    colab = shutil.which("colab")
    if not colab:
        pytest.skip("Optional installed-CLI compatibility check")
    source = '''
import sys
from pathlib import Path
from types import SimpleNamespace
from scripts import colab_accounts as accounts
from colab_cli import auth
directory = Path(sys.argv[1]) / "account2"
accounts.profile_dir = lambda account: directory
secret = "FAKE-OAUTH-CREDENTIAL-MUST-NOT-BE-SAVED"
auth._run_remote_flow = lambda config: SimpleNamespace(to_json=lambda: secret)
def reject(credentials, account):
    raise accounts.AccountError("Choose your-drive-account@example.com")
accounts.verify_cli_account = reject
accounts.configure_cli("2")
try:
    auth.get_credentials()
except accounts.AccountError:
    pass
else:
    raise AssertionError("Wrong account accepted")
assert not (directory / "token.json").exists()
assert not list(directory.rglob("*.json"))
print("WRONG_CONSENT_REJECTED")
'''
    result = subprocess.run([colab_connection.cli_python(colab), "-c", source, str(tmp_path)],
                            cwd=accounts.ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "WRONG_CONSENT_REJECTED" in result.stdout
    assert "FAKE-OAUTH-CREDENTIAL" not in result.stdout + result.stderr
