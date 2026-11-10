#!/usr/bin/env python3
"""Authorize only the dedicated experiment account for the independent Vast workflow."""

import argparse
import json
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workflows.vast.common import ACCOUNT, DEFAULT_TOKEN, DRIVE_SCOPE
from workflows.vast.drive import Drive, service


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-secrets", type=Path, required=True)
    parser.add_argument("--token", type=Path, default=DEFAULT_TOKEN)
    parser.add_argument("--drive-config", type=Path, default=Path(__file__).with_name("drive.example.json"))
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    token, client_path = args.token.expanduser().resolve(), args.client_secrets.expanduser().resolve()
    if token.is_relative_to(repo) or client_path.is_relative_to(repo):
        parser.error("Keep the separate Vast OAuth client and token outside this repository")
    if token.exists():
        parser.error("Token exists; choose a new --token path for reauthorization")
    from google_auth_oauthlib.flow import InstalledAppFlow
    client = json.loads(client_path.read_text())
    if "installed" not in client:
        parser.error("Use a Google Desktop app OAuth client")
    flow = InstalledAppFlow.from_client_config(client, scopes=[DRIVE_SCOPE], autogenerate_code_verifier=True)
    auth = flow.run_local_server(host="127.0.0.1", port=0, timeout_seconds=600,
        login_hint=ACCOUNT, prompt="consent", access_type="offline", include_granted_scopes="false",
        authorization_prompt_message="Opening Google sign-in for the experiment account.")
    if auth.granted_scopes is not None and set(auth.granted_scopes) != {DRIVE_SCOPE}:
        raise RuntimeError("Google granted unexpected scopes; use a separate Drive-only OAuth client")
    data = json.loads(auth.to_json())
    data.update(type="authorized_user", scopes=[DRIVE_SCOPE])
    if not data.get("refresh_token"):
        raise RuntimeError("Google did not grant offline access")
    token.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(token, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(data, out)
        store = Drive(service(token), json.loads(args.drive_config.read_text()), "setup")
        store.preflight()  # Read-only; no folders or experiment records created.
    except Exception:
        token.unlink(missing_ok=True)
        raise
    print(f"Verified {ACCOUNT}. Separate Vast authorization saved at {token}.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Vast Drive setup failed ({type(exc).__name__}); check the account and client configuration.", file=sys.stderr)
        raise SystemExit(1) from None
