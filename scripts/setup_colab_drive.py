#!/usr/bin/env python3
"""One-time, file-picker authorization for the dedicated mdlARC Drive uploader."""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import secrets
import time
from urllib.parse import parse_qs, urlparse
import webbrowser

from colab_drive import DEFAULT_TOKEN, DRIVE_SCOPE, DriveStore, build_service, load_config


def picker_parameters(config: dict) -> dict:
    ids = {config["root_folder_id"], config["wheel_file_id"], config["environment_file_id"],
           config["dataset_folder_id"], config["dataset_file_id"], *config["wheel_ancestor_ids"]}
    return {"prompt": "consent", "trigger_onepick": "true", "allow_multiple": "true",
            "allow_folder_selection": "true", "file_ids": ",".join(sorted(ids)),
            "include_granted_scopes": "false", "access_type": "offline"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-secrets", type=Path, required=True, help="Dedicated Google Desktop OAuth client JSON outside this repository")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("colab_drive.json"))
    parser.add_argument("--token", type=Path, default=DEFAULT_TOKEN)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    token = args.token.expanduser().resolve()
    if token.is_relative_to(repo) or args.client_secrets.resolve().is_relative_to(repo):
        parser.error("Keep OAuth client secrets and tokens outside the repository")
    if token.exists():
        parser.error("Token already exists; use a new --token path for reauthorization")
    client = json.loads(args.client_secrets.read_text())
    if "installed" not in client:
        parser.error("Use a dedicated Desktop app OAuth client")
    config = load_config(args.config)
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        parser.error("Install scripts/requirements_colab_drive.txt in this Python environment")
    flow = InstalledAppFlow.from_client_config(client, scopes=[DRIVE_SCOPE], autogenerate_code_verifier=True)
    callback = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            values = parse_qs(urlparse(self.path).query)
            if "code" in values or "error" in values:
                callback.update(values)
                message = b"Return to your terminal to finish mdlARC authorization."
            else:
                message = b"Waiting for Google authorization."
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(message)

        def log_message(self, *args):
            pass  # OAuth callback URLs contain codes; never log them.

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        server.timeout = 1
        flow.redirect_uri = f"http://127.0.0.1:{server.server_port}/"
        url, state = flow.authorization_url(**picker_parameters(config))
        print("Select only mdlARC, its two wheel folders, the wheel, environment.json, datasets, and task-variations-v1.zip in Google's picker.")
        print("Opening Google sign-in. If no browser opens, visit this authorization URL:")
        print(url)
        webbrowser.open(url)
        deadline = time.monotonic() + 600
        while not callback and time.monotonic() < deadline:
            server.handle_request()
    if not callback or callback.get("error") or not secrets.compare_digest(callback.get("state", [""])[0], state):
        raise RuntimeError("Authorization cancelled, timed out, or failed state validation")
    expected = set(picker_parameters(config)["file_ids"].split(","))
    picked = set(callback.get("picked_file_ids", [""])[0].split(","))
    if picked != expected:
        raise RuntimeError("Select exactly the configured mdlARC folder, wheel files, and dataset ZIP; no token was saved")
    flow.fetch_token(code=callback["code"][0])
    data = json.loads(flow.credentials.to_json())
    data.update(type="authorized_user", scopes=[DRIVE_SCOPE])
    token.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(token, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump(data, output)
        store = DriveStore(config, build_service(token))
        store.preflight(include_dataset=True)
    except Exception:
        token.unlink(missing_ok=True)
        raise
    print(f"Restricted Drive access verified. Credentials saved privately to {token}")
    print("No Drive files were created, renamed, or deleted during setup.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Drive setup failed ({type(exc).__name__}). Check the dedicated client and selected files.")
        raise SystemExit(1) from None
