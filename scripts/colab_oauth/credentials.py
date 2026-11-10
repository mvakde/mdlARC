"""Private credential storage and read-only verification. Never log provider errors."""

import json
import logging
import os
import stat
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
ACCOUNT = "your-drive-account@example.com"
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"
TOKEN_URI = "https://oauth2.googleapis.com/token"
DEFAULT_TOKEN = Path.home() / ".config/mdlarc-colab-oauth/your-drive-account/token.json"
FIELDS = ("client_id", "client_secret", "refresh_token")


class AuthError(RuntimeError):
    """An error with a safe, credential-free message."""


def outside_repository(path: Path) -> Path:
    path = path.expanduser()
    if path.is_symlink():
        raise AuthError("Credential files must not be symbolic links.")
    path = path.resolve()
    if path.is_relative_to(ROOT):
        raise AuthError("Keep OAuth client files and tokens outside the repository, even in ignored folders.")
    if any((parent / ".git").exists() for parent in path.parents):
        raise AuthError("Keep OAuth client files and tokens outside every Git checkout.")
    return path


def clean_credentials(data: dict) -> dict:
    if (not isinstance(data, dict) or data.get("type") != "authorized_user"
            or any(not isinstance(data.get(key), str) or not data[key].strip() for key in FIELDS)
            or data.get("token_uri", TOKEN_URI) != TOKEN_URI
            or data.get("scopes") != [DRIVE_SCOPE]):
        raise AuthError("Use the dedicated Drive-only token made by colab_oauth/setup.py.")
    # Do not persist access tokens, provider responses, or unrelated credential fields.
    return {"type": "authorized_user", **{key: data[key] for key in FIELDS},
            "token_uri": TOKEN_URI, "scopes": [DRIVE_SCOPE]}


def read_token(path: Path = DEFAULT_TOKEN) -> dict:
    path = outside_repository(path)
    try:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.getuid():
            raise AuthError("Token must be a file owned by you with permissions 600.")
        if info.st_size > 64 * 1024:
            raise AuthError("Unexpected credential file size.")
        return clean_credentials(json.loads(path.read_text(encoding="utf-8")))
    except AuthError:
        raise
    except Exception:
        raise AuthError("Could not read the private token. Run colab_oauth/setup.py first.") from None


def verify_credentials(data: dict) -> dict:
    """Refresh, confirm granted scopes, and check the Drive identity without writes."""
    data = clean_credentials(data)
    try:
        from google.auth.transport.requests import Request as GoogleRequest
        from google.oauth2.credentials import Credentials
        credentials = Credentials.from_authorized_user_info(data, scopes=[DRIVE_SCOPE])
        request = GoogleRequest()

        def bounded_request(**kwargs):
            kwargs["timeout"] = 30
            return request(**kwargs)

        credentials.refresh(bounded_request)
        # No credential-bearing URLs or response bodies are printed or included in errors.
        with urlopen(TOKEN_URI.replace("/token", "/tokeninfo") + "?" + urlencode(
                {"access_token": credentials.token}), timeout=30) as response:
            info = json.load(response)
        if set(info.get("scope", "").split()) != {DRIVE_SCOPE}:
            raise AuthError("Google granted additional scopes. Use a separate Drive-only OAuth client.")
        with urlopen(Request("https://www.googleapis.com/drive/v3/about?fields=user(emailAddress)",
                headers={"Authorization": "Bearer " + credentials.token}), timeout=30) as response:
            email = json.load(response).get("user", {}).get("emailAddress", "").casefold()
        if email != ACCOUNT:
            raise AuthError("Wrong Google account. Only your-drive-account@example.com is accepted; no token was saved or uploaded.")
        return data
    except AuthError:
        raise
    except ImportError:
        raise AuthError("Install scripts/requirements_colab_drive.txt in this Python environment.") from None
    except Exception:
        raise AuthError("Drive authorization could not be verified. Check connectivity; if Testing access expired, rerun setup.py. No VM was changed.") from None


def save_token(path: Path, data: dict, *, replace: bool = False) -> None:
    path = outside_repository(path)
    data = clean_credentials(data)
    if path.exists() and not replace:
        raise AuthError("Token already exists. Use --reauthorize to replace it after Google consent succeeds.")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=".token-", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump({**data, "account": ACCOUNT,
                       "authorized_at": datetime.now(UTC).isoformat(),
                       "testing_reauthorize_by": (datetime.now(UTC) + timedelta(days=7)).isoformat()}, output)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            # An exclusive link also prevents a concurrent setup from overwriting a token.
            os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def authorize_client(client_path: Path) -> dict:
    client_path = outside_repository(client_path)
    try:
        client = json.loads(client_path.read_text(encoding="utf-8"))
        installed = client.get("installed", {})
        if (not all(installed.get(key) for key in ("client_id", "client_secret"))
                or installed.get("auth_uri") != "https://accounts.google.com/o/oauth2/auth"
                or installed.get("token_uri") != TOKEN_URI):
            raise AuthError("Use a Google Desktop-app OAuth client JSON outside the repository.")
        from google_auth_oauthlib.flow import InstalledAppFlow
        flow = InstalledAppFlow.from_client_config(client, scopes=[DRIVE_SCOPE],
                                                   autogenerate_code_verifier=True)
        # The library's callback logger can contain the authorization code.
        logger = logging.getLogger("google_auth_oauthlib.flow")
        old_level = logger.level
        logger.setLevel(logging.CRITICAL)
        try:
            flow.run_local_server(host="127.0.0.1", port=0, open_browser=True,
                timeout_seconds=600, authorization_prompt_message="",
                success_message="Experiment authorization received. Return to the terminal.",
                access_type="offline", prompt="consent", login_hint=ACCOUNT,
                include_granted_scopes="false")
        finally:
            logger.setLevel(old_level)
        data = json.loads(flow.credentials.to_json())
        data["type"] = "authorized_user"
        return verify_credentials(data)
    except AuthError:
        raise
    except ImportError:
        raise AuthError("Install scripts/requirements_colab_drive.txt in this Python environment.") from None
    except Exception:
        raise AuthError("Google authorization did not complete. The existing token was preserved.") from None
