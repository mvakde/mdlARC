"""Short VM-side credential installation; contains no credential values."""

import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ACCOUNT = "your-drive-account@example.com"
SCOPE = "https://www.googleapis.com/auth/drive"
TOKEN_URI = "https://oauth2.googleapis.com/token"
BASE = Path("/content")


def private_directory(path):
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise RuntimeError("Unsafe VM authentication directory.")
    path.mkdir(mode=0o700, exist_ok=True)
    os.chmod(path, 0o700)


def unused_workspace():
    if any((BASE / name).exists() or (BASE / name).is_symlink()
           for name in ("mdlARC", "mdlARC-source.tar.gz", "mdlARC-results-manifest.json")):
        raise RuntimeError("VM contains experiment run files. Use a fresh session; existing runs were untouched.")


def prepare():
    unused_workspace()
    existing_credentials()
    private_directory(BASE / ".mdlarc-experiment-oauth")


def existing_credentials():
    """Check ownership markers before changing directories or uploading credentials."""
    directory = BASE / ".mdlarc-experiment-oauth"
    adc_directory = BASE / ".adc"
    adc = adc_directory / "adc.json"
    receipt = directory / "receipt.json"
    if (directory.is_symlink() or adc_directory.is_symlink()
            or (adc_directory.exists() and not adc_directory.is_dir())
            or adc.is_symlink() or receipt.is_symlink()):
        raise RuntimeError("Unsafe VM credential path.")
    if adc.exists():
        if not receipt.is_file():
            raise RuntimeError("VM already has old-style credentials. Use a fresh session; they were not overwritten.")
        previous = json.loads(receipt.read_text())
        if (previous.get("account") != ACCOUNT
                or previous.get("credential_sha256") != hashlib.sha256(adc.read_bytes()).hexdigest()):
            raise RuntimeError("VM credentials changed outside this workflow; they were not overwritten.")


def install(staged_name, expected_sha256):
    """Refresh and verify before installing; refuse old-style or modified ADC files."""
    if (Path(staged_name).name != staged_name or not staged_name.endswith(".json")
            or len(expected_sha256) != 64):
        raise RuntimeError("Invalid credential staging request.")
    directory = BASE / ".mdlarc-experiment-oauth"
    private_directory(directory)
    staged = directory / staged_name
    adc_directory = BASE / ".adc"
    adc = adc_directory / "adc.json"
    receipt = directory / "receipt.json"
    try:
        unused_workspace()
        existing_credentials()
        if staged.is_symlink() or not staged.is_file():
            raise RuntimeError("Credential transfer is missing or linked.")
        os.chmod(staged, 0o600)
        raw = staged.read_bytes()
        if len(raw) > 64 * 1024 or hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise RuntimeError("Credential transfer checksum failed.")
        data = json.loads(raw)
        if (data.get("type") != "authorized_user" or data.get("scopes") != [SCOPE]
                or data.get("token_uri") != TOKEN_URI
                or any(not isinstance(data.get(key), str) or not data[key]
                       for key in ("client_id", "client_secret", "refresh_token"))
                or set(data) != {"type", "scopes", "token_uri", "client_id", "client_secret", "refresh_token"}):
            raise RuntimeError("Only dedicated Drive-only user credentials are accepted.")
        from google.auth.transport.requests import Request as GoogleRequest
        from google.oauth2.credentials import Credentials
        credentials = Credentials.from_authorized_user_info(data, scopes=[SCOPE])
        request = GoogleRequest()

        def bounded_request(**kwargs):
            kwargs["timeout"] = 30
            return request(**kwargs)

        credentials.refresh(bounded_request)
        with urlopen("https://oauth2.googleapis.com/tokeninfo?" + urlencode(
                {"access_token": credentials.token}), timeout=30) as response:
            info = json.load(response)
        if set(info.get("scope", "").split()) != {SCOPE}:
            raise RuntimeError("Google granted additional scopes; VM authentication was not installed.")
        with urlopen(Request("https://www.googleapis.com/drive/v3/about?fields=user(emailAddress)",
                headers={"Authorization": "Bearer " + credentials.token}), timeout=30) as response:
            email = json.load(response).get("user", {}).get("emailAddress", "").casefold()
        if email != ACCOUNT:
            raise RuntimeError("Wrong Drive account; VM authentication was not installed.")
        private_directory(adc_directory)
        os.replace(staged, adc)
        os.chmod(adc, 0o600)
        temporary = directory / "receipt.tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "w") as output:
            json.dump({"account": ACCOUNT, "credential_sha256": expected_sha256}, output)
        temporary.replace(receipt)
        print("MDLARC_EXPERIMENT_OAUTH_READY", flush=True)
    finally:
        if staged.is_file() and not staged.is_symlink():
            staged.unlink()


def safe_install(staged_name, expected_sha256):
    try:
        install(staged_name, expected_sha256)
    except Exception:
        # Provider exceptions can include refresh tokens, access tokens, or OAuth codes.
        print("Experiment OAuth installation did not confirm. Inspect the VM; existing runs were untouched. No credential details were logged.", flush=True)
        raise SystemExit(1) from None


def discard_staging(staged_name):
    """Best-effort cleanup after a failed upload/installation; never remove installed ADC."""
    if Path(staged_name).name != staged_name or not staged_name.startswith("stage-"):
        raise RuntimeError("Invalid credential staging request.")
    directory = BASE / ".mdlarc-experiment-oauth"
    if not directory.is_symlink():
        staged = directory / staged_name
        if staged.is_file() and not staged.is_symlink():
            staged.unlink()
