#!/usr/bin/env python3
"""Use two independent Colab CLI logins without changing HOME or the installed CLI."""

import argparse
from contextlib import contextmanager
import getpass
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys

ACCOUNTS = {"1": "your-colab-account@example.com", "2": "your-drive-account@example.com"}
ACCOUNT_ENV = "MDLARC_COLAB_ACCOUNT"
ROOT = Path(__file__).resolve().parents[1]


class AccountError(RuntimeError):
    """A safe, credential-free account-selection failure."""


def selected_account(account=None):
    value = str(account if account is not None else os.environ.get(ACCOUNT_ENV, "1"))
    if value not in ACCOUNTS:
        raise ValueError("Colab account must be 1 (your-colab-account@example.com) or 2 (your-drive-account@example.com)")
    return value


def profile_dir(account):
    account = selected_account(account)
    if account == "1":
        return Path.home() / ".config/colab-cli"
    return Path.home() / ".config/mdlarc-colab-accounts" / account


def record_account(record):
    account = selected_account(record.get("colab_account", "1"))
    if record.get("colab_email", ACCOUNTS[account]) != ACCOUNTS[account]:
        raise ValueError("Saved Colab account and email disagree")
    return account


def run_account(root, session, requested=None):
    """Old records belong to account 1; saved ownership wins over environment defaults."""
    if not session or Path(session).name != session or session in {".", ".."}:
        raise ValueError("Invalid Colab session name")
    if __package__:
        from .run_layout import results_root
    else:
        from run_layout import results_root
    active = results_root(root) / "colab_runs" / "_active"
    owners = set()
    for suffix in (".json", ".connection.json"):
        path = active / f"{session}{suffix}"
        if path.exists():
            owners.add(record_account(json.loads(path.read_text())))
    if len(owners) > 1:
        raise ValueError("Saved Colab journal and kernel belong to different accounts")
    saved = next(iter(owners), None)
    if saved and requested is not None and selected_account(requested) != saved:
        raise ValueError(f"Session {session} belongs to Colab account {saved} ({ACCOUNTS[saved]})")
    return saved or selected_account(requested)


@contextmanager
def account_context(account):
    previous = os.environ.get(ACCOUNT_ENV)
    os.environ[ACCOUNT_ENV] = selected_account(account)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(ACCOUNT_ENV, None)
        else:
            os.environ[ACCOUNT_ENV] = previous


def cli_command(colab, *arguments, account=None):
    account = selected_account(account)
    if account == "1":
        # Preserve the existing login and session aliases for historical runs.
        return [colab, *arguments]
    if __package__:
        from .colab_connection import cli_python
    else:
        from colab_connection import cli_python
    return [cli_python(shutil.which(colab) or colab), str(Path(__file__).resolve()), "--account", account, "--", *arguments]


def cli_hint():
    if selected_account() == "1":
        return "colab"
    return "venv/bin/python scripts/colab_accounts.py --account 2 --"


def route_command(arguments):
    if arguments and Path(arguments[0]).name == "colab":
        return cli_command(arguments[0], *arguments[1:])
    return arguments


def verify_cli_account(credentials, account):
    """Verify identity before an allocation, pruning, upload, or kernel lookup."""
    from google.auth.transport.requests import AuthorizedSession
    try:
        with AuthorizedSession(credentials) as session:
            response = session.get("https://openidconnect.googleapis.com/v1/userinfo", timeout=15)
            response.raise_for_status()
            email = response.json().get("email", "").lower()
    except Exception:
        raise AccountError("Could not verify the Colab login; no VM operation was performed") from None
    if email != ACCOUNTS[account]:
        raise AccountError(f"Choose {ACCOUNTS[account]} for Colab account {account}; no VM operation was performed")


def configure_cli(account):
    """Configure only this process; the dedicated Drive token is never read here."""
    account = selected_account(account)
    from colab_cli import auth, common
    from colab_cli.history import HistoryLogger
    from colab_cli.state import SettingsStore
    directory = profile_dir(account)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if account == "2":
        directory.chmod(0o700)
    auth.TOKEN_CONFIG_PATH = str(directory / "token.json")
    common.state.config_path = str(directory / "sessions.json")
    common.state._history = HistoryLogger(str(directory / "history"))
    common.state._settings_store = SettingsStore(str(directory / "settings.json"))
    original_flow = auth._run_remote_flow
    original_credentials = auth.get_credentials
    original_request = auth.Request

    def bounded_request():
        request = original_request()

        def call(*args, **kwargs):
            kwargs.setdefault("timeout", 15)
            return request(*args, **kwargs)

        return call

    def verified_flow(config):
        print(f"Sign in as {ACCOUNTS[account]} (Colab account {account}).", file=sys.stderr)
        # The CLI uses input() for the code; keep the code out of terminal logs.
        original_input = original_flow.__globals__.get("input")
        original_flow.__globals__["input"] = getpass.getpass
        try:
            credentials = original_flow(config)
        finally:
            if original_input is None:
                original_flow.__globals__.pop("input", None)
            else:
                original_flow.__globals__["input"] = original_input
        verify_cli_account(credentials, account)
        return credentials

    def verified_credentials(*args, **kwargs):
        # New consent is verified before the CLI saves it; cached tokens are
        # checked before its client can touch an account's assignments.
        previous_umask = os.umask(0o077)
        try:
            session = original_credentials(*args, **kwargs)
        finally:
            os.umask(previous_umask)
        verify_cli_account(session.credentials, account)
        token_path = directory / "token.json"
        if token_path.exists():
            token_path.chmod(0o600)
        return session

    auth._run_remote_flow = verified_flow
    auth.Request = bounded_request
    auth.get_credentials = verified_credentials
    common.get_credentials = verified_credentials
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog="Example: --account 2 -- new -s my-session --gpu A100")
    parser.add_argument("--account", choices=ACCOUNTS, default="1")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    if not arguments:
        parser.error("Provide a Colab CLI command after --")
    os.environ[ACCOUNT_ENV] = args.account
    try:
        import colab_cli  # noqa: F401
    except ImportError:
        colab = shutil.which("colab")
        if not colab:
            parser.error("Install the Google Colab CLI first")
        if __package__:
            from .colab_connection import cli_python
        else:
            from colab_connection import cli_python
        return subprocess.call([cli_python(colab), str(Path(__file__).resolve()), "--account", args.account, "--", *arguments])
    # Debug logging in the CLI can include authentication headers.
    if any(value in {"--logtostderr", "--config", "--auth"} or
           value.startswith(("--config=", "--auth=", "--logtostderr=")) for value in arguments):
        parser.error("Account routing owns the CLI config and OAuth provider; debug logging is disabled")
    logging.getLogger().setLevel(logging.CRITICAL)
    directory = configure_cli(args.account)
    from colab_cli import cli
    # The installed CLI normally enables a shared debug log in its callback.
    # Account routing never writes credential-bearing HTTP debug logs.
    cli.setup_logging = lambda *_: logging.getLogger().setLevel(logging.CRITICAL)
    # Some CLI operations use an unexpired runtime-proxy token directly and
    # would otherwise skip verifying the selected Google login altogether.
    if arguments[0] in {"new", "sessions", "status", "exec", "upload", "download", "ls", "rm", "stop", "auth", "ssh"}:
        from colab_cli.common import state
        _ = state.client  # Refresh and verify before any session can be pruned or used.
    # The CLI callback otherwise resets state.config_path to its default.
    cli.app(args=["--config", str(directory / "sessions.json"), *arguments])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        message = str(exc) if isinstance(exc, AccountError) else type(exc).__name__
        print(f"Colab account command failed: {message}", file=sys.stderr)
        raise SystemExit(1) from None
