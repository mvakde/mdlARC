#!/usr/bin/env python3
"""Install the verified experiment token separately from source on an unused VM."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts import run_on_colab
from scripts import colab_accounts
from scripts.colab_oauth import remote_auth
from scripts.colab_oauth.credentials import AuthError, DEFAULT_TOKEN, read_token, verify_credentials


def cli_call(arguments, *, code=None):
    arguments = colab_accounts.route_command(arguments)
    try:
        result = subprocess.run(arguments, input=code, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180 if code else 120,
            check=False, cwd=run_on_colab.ROOT)
    except Exception:
        raise AuthError("Colab authentication transport did not confirm. Inspect the session before retrying; no VM was stopped or replaced.") from None
    if result.returncode:
        raise AuthError("Colab authentication step failed. Existing runs and old-style credentials were not overwritten; use a fresh VM or check its console.")
    return result.stdout


def authorize_session(session, token_path=DEFAULT_TOKEN):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", session) or session in {".", ".."}:
        raise AuthError("--session must be one safe session name.")
    data = verify_credentials(read_token(token_path))
    colab = shutil.which("colab")
    if not colab:
        raise AuthError("Colab CLI is missing. Use the existing Colab CLI setup.")
    cli_call([colab, "status", "-s", session])
    run_on_colab.require_unused_remote_workspace(colab, session)
    source = Path(remote_auth.__file__).read_text(encoding="utf-8")
    # Only generic source and a checksum enter CLI/kernel history, never credential values.
    cli_call([colab, "exec", "-s", session], code=source + "\nprepare()\n")
    staged_name = f"stage-{uuid.uuid4().hex}.json"
    remote = f"/content/.mdlarc-experiment-oauth/{staged_name}"
    raw = (json.dumps(data, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        with tempfile.TemporaryDirectory(prefix="mdlarc-private-oauth-") as directory:
            os.chmod(directory, 0o700)
            local = Path(directory) / "credentials.json"
            descriptor = os.open(local, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as output:
                output.write(raw)
            cli_call([colab, "upload", "-s", session, str(local), remote])
        output = cli_call([colab, "exec", "-s", session], code=source +
                          f"\nsafe_install({staged_name!r}, {digest!r})\n")
        if "MDLARC_EXPERIMENT_OAUTH_READY" not in output.splitlines():
            raise AuthError("VM authentication completion is ambiguous. Inspect the same VM; no training was submitted.")
    except BaseException:
        try:
            cli_call([colab, "exec", "-s", session], code=source +
                     f"\ndiscard_staging({staged_name!r})\n")
        except Exception:
            pass  # A lost connection can leave private staging on this VM; never stop it.
        raise
    print(f"Experiment Drive authentication ready on {session}. Credentials were transferred separately from source.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session")
    parser.add_argument("--token", type=Path, default=DEFAULT_TOKEN)
    parser.add_argument("--check", action="store_true", help="Verify the local token without accessing or changing a VM")
    parser.add_argument("--account", choices=colab_accounts.ACCOUNTS, help="Select Colab compute account 1 or 2; Drive credentials stay dedicated")
    args = parser.parse_args(argv)
    os.environ[colab_accounts.ACCOUNT_ENV] = colab_accounts.selected_account(args.account)
    if args.check:
        verify_credentials(read_token(args.token))
        print("Dedicated Drive account and token verified. No VM was accessed.")
    else:
        if not args.session:
            parser.error("--session is required unless using --check")
        authorize_session(args.session, args.token)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc) if isinstance(exc, AuthError) else "Experiment authentication failed; no credential details were logged.", file=sys.stderr)
        raise SystemExit(1) from None
