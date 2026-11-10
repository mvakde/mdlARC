"""Offline security and integration checks for default cached Colab authentication."""

import hashlib
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

from scripts import run_on_colab, run_records
from scripts.colab_oauth import authorize, credentials, launch, remote_auth, setup


SENTINEL = "FAKE-TEST-CREDENTIAL-DO-NOT-LOG"


def token():
    return {"type": "authorized_user", "client_id": "test-client.apps.googleusercontent.com",
            "client_secret": SENTINEL + "-client", "refresh_token": SENTINEL + "-refresh",
            "token_uri": credentials.TOKEN_URI, "scopes": [credentials.DRIVE_SCOPE]}


@pytest.fixture
def private_file(tmp_path, monkeypatch):
    monkeypatch.setattr(credentials, "ROOT", tmp_path / "repo")
    path = tmp_path / "private/token.json"
    credentials.save_token(path, token())
    return path


def fake_google(monkeypatch, module, *, account=credentials.ACCOUNT, scopes=None, error=None):
    import google.oauth2.credentials
    import google.auth.transport.requests
    calls = []

    def refresh(self, request):
        if error:
            raise error
        self.token = SENTINEL + "-access"
        request(url="https://oauth2.googleapis.com/token", method="POST", body=b"fake")

    monkeypatch.setattr(google.oauth2.credentials.Credentials, "refresh", refresh)
    monkeypatch.setattr(google.auth.transport.requests, "Request",
                        lambda: lambda **kwargs: calls.append(kwargs))

    def response(request, *, timeout):
        assert timeout == 30
        url = request if isinstance(request, str) else request.full_url
        if "tokeninfo" in url:
            value = {"scope": scopes if scopes is not None else credentials.DRIVE_SCOPE}
        else:
            assert request.get_header("Authorization") == "Bearer " + SENTINEL + "-access"
            value = {"user": {"emailAddress": account}}
        return io.BytesIO(json.dumps(value).encode())

    monkeypatch.setattr(module, "urlopen", response)
    return calls


def staged(tmp_path, monkeypatch, *, data=None):
    monkeypatch.setattr(remote_auth, "BASE", tmp_path)
    remote_auth.prepare()
    path = tmp_path / ".mdlarc-experiment-oauth/stage-test.json"
    raw = (json.dumps(data if data is not None else token(), sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def test_private_storage_never_accepts_checkout_or_ignored_paths(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    monkeypatch.setattr(credentials, "ROOT", repo)
    for path in (repo / "token.json", repo / "results/private.json", repo / ".git/token.json"):
        with pytest.raises(credentials.AuthError, match="outside the repository"):
            credentials.save_token(path, token())
    alias = tmp_path / "alias"
    repo.mkdir()
    alias.symlink_to(repo, target_is_directory=True)
    with pytest.raises(credentials.AuthError, match="outside the repository"):
        credentials.outside_repository(alias / "private.json")


def test_private_permissions_and_atomic_renewal(private_file):
    assert private_file.stat().st_mode & 0o777 == 0o600
    assert private_file.parent.stat().st_mode & 0o777 == 0o700
    original = private_file.read_bytes()
    with pytest.raises(credentials.AuthError, match="already exists"):
        credentials.save_token(private_file, token())
    assert private_file.read_bytes() == original
    renewed = {**token(), "refresh_token": SENTINEL + "-renewed"}
    credentials.save_token(private_file, renewed, replace=True)
    assert credentials.read_token(private_file) == renewed
    assert private_file.stat().st_mode & 0o777 == 0o600
    assert list(private_file.parent.iterdir()) == [private_file]


def test_loose_permissions_and_symlinks_are_rejected(private_file, tmp_path):
    private_file.chmod(0o644)
    with pytest.raises(credentials.AuthError, match="permissions 600"):
        credentials.read_token(private_file)
    link = tmp_path / "linked.json"
    link.symlink_to(private_file)
    with pytest.raises(credentials.AuthError, match="symbolic links"):
        credentials.read_token(link)


def test_credentials_cannot_be_saved_in_another_git_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(credentials, "ROOT", tmp_path / "repo")
    other = tmp_path / "another-repo"
    (other / ".git").mkdir(parents=True)
    with pytest.raises(credentials.AuthError, match="every Git checkout"):
        credentials.save_token(other / "private/token.json", token())


@pytest.mark.parametrize("changed", [
    {"type": "service_account"}, {"token_uri": "https://untrusted.invalid/token"},
    {"scopes": [credentials.DRIVE_SCOPE, "https://www.googleapis.com/auth/gmail.readonly"]},
    {"refresh_token": ""}, {"client_secret": None},
])
def test_only_complete_drive_only_user_credentials_accepted(changed):
    with pytest.raises(credentials.AuthError):
        credentials.clean_credentials({**token(), **changed})


def test_refresh_verifies_actual_account_scope_and_bounds_requests(monkeypatch):
    calls = fake_google(monkeypatch, credentials)
    assert credentials.verify_credentials({**token(), "token": SENTINEL + "-discard"}) == token()
    assert calls[0]["timeout"] == 30


@pytest.mark.parametrize("account, scopes", [
    ("personal@example.com", credentials.DRIVE_SCOPE),
    (credentials.ACCOUNT, credentials.DRIVE_SCOPE + " https://www.googleapis.com/auth/cloud-platform"),
])
def test_personal_identity_or_additional_actual_scopes_rejected(monkeypatch, account, scopes):
    fake_google(monkeypatch, credentials, account=account, scopes=scopes)
    with pytest.raises(credentials.AuthError):
        credentials.verify_credentials(token())


def test_provider_failure_cannot_expose_credentials(monkeypatch, capsys):
    fake_google(monkeypatch, credentials, error=RuntimeError(SENTINEL))
    with pytest.raises(credentials.AuthError) as error:
        credentials.verify_credentials(token())
    assert SENTINEL not in str(error.value) + capsys.readouterr().out


def test_expired_testing_reauthorization_preserves_previous_token(private_file, monkeypatch):
    original = private_file.read_bytes()
    monkeypatch.setattr(setup, "authorize_client", lambda path: (_ for _ in ()).throw(
        credentials.AuthError("Google authorization did not complete.")))
    with pytest.raises(credentials.AuthError):
        setup.main(["--client-secrets", str(private_file.parent / "client.json"),
                    "--token", str(private_file), "--reauthorize"])
    assert private_file.read_bytes() == original


def test_desktop_flow_uses_loopback_pkce_offline_and_account_hint(tmp_path, monkeypatch):
    import google_auth_oauthlib.flow
    monkeypatch.setattr(credentials, "ROOT", tmp_path / "repo")
    client = tmp_path / "client.json"
    client.write_text(json.dumps({"installed": {"client_id": "test", "client_secret": "fake",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth", "token_uri": credentials.TOKEN_URI}}))
    calls = []

    class Flow:
        credentials = SimpleNamespace(to_json=lambda: json.dumps(token()))

        def run_local_server(self, **kwargs):
            calls.append(kwargs)

    def create(client, *, scopes, autogenerate_code_verifier):
        assert scopes == [credentials.DRIVE_SCOPE]
        assert autogenerate_code_verifier is True
        return Flow()

    monkeypatch.setattr(google_auth_oauthlib.flow.InstalledAppFlow, "from_client_config", create)
    monkeypatch.setattr(credentials, "verify_credentials", lambda data: data)
    assert credentials.authorize_client(client) == token()
    assert calls[0]["host"] == "127.0.0.1" and calls[0]["port"] == 0
    assert calls[0]["access_type"] == "offline"
    assert calls[0]["login_hint"] == credentials.ACCOUNT
    assert calls[0]["include_granted_scopes"] == "false"


def test_vm_install_has_private_refresh_credentials_outside_worktree(tmp_path, monkeypatch, capsys):
    path, digest = staged(tmp_path, monkeypatch)
    calls = fake_google(monkeypatch, remote_auth)
    remote_auth.install(path.name, digest)
    adc = tmp_path / ".adc/adc.json"
    assert json.loads(adc.read_text()) == token()
    assert adc.stat().st_mode & 0o777 == 0o600
    assert adc.parent.stat().st_mode & 0o777 == 0o700
    assert not path.exists()
    assert calls[0]["timeout"] == 30
    assert SENTINEL not in capsys.readouterr().out
    # The existing Drive code reads this location and Google's library receives
    # a refresh credential, not an access token requiring periodic Mac delivery.
    import google.auth
    restored, _ = google.auth.load_credentials_from_file(str(adc), scopes=[credentials.DRIVE_SCOPE])
    assert restored.refresh_token == token()["refresh_token"]


def test_original_drive_client_reads_installed_credentials(tmp_path, monkeypatch):
    from scripts import colab_drive
    import googleapiclient.discovery
    path, digest = staged(tmp_path, monkeypatch)
    fake_google(monkeypatch, remote_auth)
    remote_auth.install(path.name, digest)
    original_path = Path
    monkeypatch.setattr(colab_drive, "Path", lambda value: tmp_path / ".adc/adc.json"
                        if value == "/content/.adc/adc.json" else original_path(value))
    captured = {}

    def build(*args, **kwargs):
        captured.update(kwargs)
        return "drive-client"

    monkeypatch.setattr(googleapiclient.discovery, "build", build)
    assert colab_drive.build_shared_service() == "drive-client"
    assert captured["credentials"].refresh_token == token()["refresh_token"]


def test_legacy_vm_credentials_never_overwritten(tmp_path, monkeypatch):
    path, digest = staged(tmp_path, monkeypatch)
    adc = tmp_path / ".adc/adc.json"
    adc.parent.mkdir()
    adc.parent.chmod(0o755)
    adc.write_text("original manual credential")
    with pytest.raises(RuntimeError, match="old-style"):
        remote_auth.prepare()
    assert adc.parent.stat().st_mode & 0o777 == 0o755
    with pytest.raises(RuntimeError, match="old-style"):
        remote_auth.install(path.name, digest)
    assert adc.read_text() == "original manual credential"
    assert not path.exists()


def test_same_unused_vm_can_recover_lost_install_ack(tmp_path, monkeypatch):
    path, digest = staged(tmp_path, monkeypatch)
    fake_google(monkeypatch, remote_auth)
    remote_auth.install(path.name, digest)
    path, digest = staged(tmp_path, monkeypatch)
    remote_auth.install(path.name, digest)
    assert json.loads((tmp_path / ".adc/adc.json").read_text()) == token()


@pytest.mark.parametrize("failure", ["checksum", "personal", "scope", "expired"])
def test_failed_vm_checks_leave_no_installed_or_staged_secret(tmp_path, monkeypatch, capsys, failure):
    path, digest = staged(tmp_path, monkeypatch)
    fake_google(monkeypatch, remote_auth,
                account="personal@example.com" if failure == "personal" else credentials.ACCOUNT,
                scopes=credentials.DRIVE_SCOPE + " extra" if failure == "scope" else None,
                error=RuntimeError(SENTINEL) if failure == "expired" else None)
    with pytest.raises(SystemExit):
        remote_auth.safe_install(path.name, "0" * 64 if failure == "checksum" else digest)
    assert not (tmp_path / ".adc/adc.json").exists()
    assert not path.exists()
    assert SENTINEL not in capsys.readouterr().out


def test_existing_run_is_never_reauthenticated(tmp_path, monkeypatch):
    path, digest = staged(tmp_path, monkeypatch)
    (tmp_path / "mdlARC").mkdir()
    with pytest.raises(RuntimeError, match="fresh session"):
        remote_auth.install(path.name, digest)
    assert not (tmp_path / ".adc/adc.json").exists()
    assert not path.exists()


def test_changed_adc_or_linked_paths_are_not_replaced(tmp_path, monkeypatch):
    path, digest = staged(tmp_path, monkeypatch)
    fake_google(monkeypatch, remote_auth)
    remote_auth.install(path.name, digest)
    adc = tmp_path / ".adc/adc.json"
    adc.write_text("changed elsewhere")
    path.write_text(json.dumps(token()))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="changed outside"):
        remote_auth.install(path.name, digest)
    assert adc.read_text() == "changed elsewhere"


def test_two_fresh_vms_reuse_one_token_without_browser_login(private_file, tmp_path, monkeypatch):
    from contextlib import redirect_stdout
    import shutil
    import google.oauth2.credentials
    fake_google(monkeypatch, remote_auth)
    monkeypatch.setattr(authorize, "verify_credentials", lambda data: data)
    monkeypatch.setattr(authorize.shutil, "which", lambda name: "/fake/colab")
    monkeypatch.setattr(run_on_colab, "require_unused_remote_workspace", lambda *args: None)
    vm_paths = {name: tmp_path / name for name in ("one", "two")}
    for path in vm_paths.values():
        path.mkdir()
    code_source = Path(remote_auth.__file__).read_text()
    histories = []

    def call(arguments, *, code=None):
        session = arguments[arguments.index("-s") + 1]
        vm = vm_paths[session]
        histories.append((arguments, code))
        assert SENTINEL not in " ".join(arguments) + (code or "")
        if arguments[1] == "upload":
            destination = vm / Path(arguments[-1]).relative_to("/content")
            shutil.copyfile(arguments[-2], destination)
        if code:
            namespace = {}
            exec(compile(code_source, "remote-auth", "exec"), namespace)
            namespace["BASE"] = vm
            namespace["urlopen"] = remote_auth.urlopen
            output = io.StringIO()
            with redirect_stdout(output):
                exec(code[len(code_source):], namespace)
            return output.getvalue()
        return ""

    monkeypatch.setattr(authorize, "cli_call", call)
    for session in vm_paths:
        authorize.authorize_session(session, private_file)
        assert json.loads((vm_paths[session] / ".adc/adc.json").read_text()) == token()
        assert not list((vm_paths[session] / ".mdlarc-experiment-oauth").glob("stage-*.json"))
    assert len(histories) == 8


def test_authorize_transfer_never_embeds_credentials_in_code_or_argv(private_file, monkeypatch):
    monkeypatch.setattr(authorize, "verify_credentials", lambda data: data)
    monkeypatch.setattr(authorize.shutil, "which", lambda name: "/fake/colab")
    monkeypatch.setattr(run_on_colab, "require_unused_remote_workspace", lambda *args: None)
    calls, transfer_path = [], []

    def call(arguments, *, code=None):
        assert SENTINEL not in " ".join(arguments) + (code or "")
        calls.append((arguments, code))
        if arguments[1] == "upload":
            local = Path(arguments[-2])
            assert local.stat().st_mode & 0o777 == 0o600
            assert local.parent.stat().st_mode & 0o777 == 0o700
            assert SENTINEL in local.read_text()
            transfer_path.append(local)
        return "MDLARC_EXPERIMENT_OAUTH_READY\n" if code and "\nsafe_install(" in code else ""

    monkeypatch.setattr(authorize, "cli_call", call)
    authorize.authorize_session("test-session", private_file)
    assert [args[1] for args, _ in calls] == ["status", "exec", "upload", "exec"]
    assert not transfer_path[0].exists()


def test_ambiguous_transfer_cleans_only_staging_and_never_launches(private_file, monkeypatch):
    monkeypatch.setattr(authorize, "verify_credentials", lambda data: data)
    monkeypatch.setattr(authorize.shutil, "which", lambda name: "/fake/colab")
    monkeypatch.setattr(run_on_colab, "require_unused_remote_workspace", lambda *args: None)
    calls = []

    def call(arguments, *, code=None):
        calls.append((arguments, code))
        return ""

    monkeypatch.setattr(authorize, "cli_call", call)
    with pytest.raises(credentials.AuthError, match="ambiguous"):
        authorize.authorize_session("test-session", private_file)
    assert "\ndiscard_staging(" in calls[-1][1]
    assert all(args[1] not in ("new", "stop") for args, code in calls)


def test_cli_error_details_are_redacted(monkeypatch):
    monkeypatch.setattr(authorize.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=1, stdout=SENTINEL, stderr=SENTINEL))
    with pytest.raises(credentials.AuthError) as error:
        authorize.cli_call(["colab", "exec"], code="print('safe')")
    assert SENTINEL not in str(error.value)


@pytest.mark.parametrize("arguments", [
    ["--config", "specs/example.json", "--dry-run"],
    ["--watch-session", "running-session", "--stop-after-success"],
    ["--auth-mode", "manual", "--session", "manual-session", "--config", "specs/example.json"],
])
def test_manual_preview_and_watch_modes_never_read_or_upload_token(monkeypatch, arguments):
    delegated = []
    monkeypatch.setattr(launch, "delegate", lambda args: delegated.append(args) or 0)
    monkeypatch.setattr(launch, "authorize_session", lambda *args: pytest.fail("unexpected authentication"))
    assert launch.main(arguments) == 0
    expected = [arg for arg in arguments if arg not in ("--auth-mode", "manual")]
    assert delegated == [expected]


def test_cached_wrapper_validates_then_authenticates_then_uses_original_launcher(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "ROOT", tmp_path)
    monkeypatch.setenv("MDLARC_RUN_BRANCH", "newreason")
    events = []
    monkeypatch.setattr(launch, "delegate", lambda args: events.append(("launch", args)) or 0)
    monkeypatch.setattr(launch, "authorize_session", lambda *args: events.append(("auth", args)))
    args = ["--config", "specs/example.json", "--session", "fresh", "--stop-after-success"]
    assert launch.main(args) == 0
    assert events[0] == ("launch", [*args, "--dry-run"])
    assert events[1][0] == "auth" and events[1][1][0] == "fresh"
    assert events[2] == ("launch", args)


def test_invalid_spec_or_existing_branch_records_never_touch_vm(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "ROOT", tmp_path)
    monkeypatch.setenv("MDLARC_RUN_BRANCH", "other-branch")
    monkeypatch.setattr(launch, "authorize_session", lambda *args: pytest.fail("unexpected authentication"))
    monkeypatch.setattr(launch, "delegate", lambda args: 1)
    assert launch.main(["--config", "bad.json", "--session", "fresh"]) == 1
    monkeypatch.setattr(launch, "delegate", lambda args: 0)
    saved = tmp_path / "results/newreason/experiments/notes/fresh.md"
    saved.parent.mkdir(parents=True)
    saved.write_text("existing note")
    with pytest.raises(credentials.AuthError, match="already has run records"):
        launch.main(["--config", "good.json", "--session", "fresh", "--branch", "newreason"])


def test_oauth_files_never_enter_source_or_record_archives(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(run_on_colab, "ROOT", repo)
    monkeypatch.setenv("MDLARC_RUN_BRANCH", "newreason")
    files = {
        "launch_run.py": "# launch", "requirements.txt": "torch", "scripts/run_layout.py": "# layout",
        "specs/example.json": "{}", "wheels/environment.json": json.dumps({"wheel": "fake.whl", "sha256": "0" * 64}),
        "scripts/colab_oauth/private/token.json": SENTINEL,
        ".adc/adc.json": SENTINEL, ".mdlarc-experiment-oauth/stage.json": SENTINEL,
        "results/newreason/colab_runs/_active/fresh.connection.json": '{"kernel_id":"public-kernel"}',
        "results/newreason/experiments/notes/fresh.md": "test intent",
        "results/newreason/colab_runs/_active/fresh.json": '{"session":"fresh"}',
    }
    for relative, content in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    for name in ("colab_drive.py", "colab_drive.json", "requirements_colab_drive.txt", "colab_remote.py",
                 "run_on_colab.py", "colab_run_docs.py", "materialize_sweep_spec.py", "colab_connection.py", "colab_accounts.py", "run_records.py", "colab_reuse.py"):
        (repo / "scripts" / name).write_text("# existing source")
    source = tmp_path / "source.tar.gz"
    run_on_colab.create_snapshot(repo / "specs/example.json", source, drive=True, session="fresh")
    records = tmp_path / "records.tar.gz"
    run_records.pack_records(repo, records, "fresh")
    for path in (source, records):
        with tarfile.open(path) as archive:
            for member in archive.getmembers():
                assert "colab_oauth/private" not in member.name and ".adc/" not in member.name
                if member.isfile():
                    assert SENTINEL.encode() not in archive.extractfile(member).read()
