"""Scoped Drive transfers for mdlARC Colab runs.

New runs use the dedicated account authenticated on the VM. The historical
credential mode remains for recovery of runs made with the older launcher.
Neither mode deletes, replaces, moves, shares, or lists the whole Drive.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from urllib.parse import urlencode
from urllib.request import urlopen
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.checkpoint_io import prune_backed_up_checkpoints


ROOT_FOLDER_ID = "DRIVE_ID_05_PLACEHOLDER"
EXPERIMENT_ACCOUNT = "your-drive-account@example.com"
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.file"
FOLDER_MIME = "application/vnd.google-apps.folder"
FIELDS = "id,name,mimeType,parents,size,md5Checksum,sha256Checksum,trashed"
DEFAULT_TOKEN = Path.home() / ".config" / "mdlarc-drive" / "token.json"
CHECKPOINT_API_RANGE_BYTES = 256 * 1024 * 1024
CHECKPOINT_API_STREAM_BYTES = 1024 * 1024


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_config(path: Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("root_folder_id") != ROOT_FOLDER_ID:
        raise ValueError("Drive transfers are restricted to the configured mdlARC folder")
    ancestors = config.get("wheel_ancestor_ids", [])
    if not ancestors or ancestors[-1] != ROOT_FOLDER_ID or len(set(ancestors)) != len(ancestors):
        raise ValueError("Wheel ancestry must end at the mdlARC folder without cycles")
    for item in [config.get("runs_folder_id"), config.get("wheel_file_id"), config.get("environment_file_id"),
                 config.get("dataset_folder_id"), config.get("dataset_file_id"), *ancestors]:
        if not isinstance(item, str) or not item or not all(c.isalnum() or c in "_-" for c in item):
            raise ValueError("Invalid Drive file ID")
    if config.get("dataset_name") != "task-variations-v1.zip":
        raise ValueError("Unexpected Drive dataset filename")
    if not isinstance(config.get("dataset_size"), int) or config["dataset_size"] <= 0:
        raise ValueError("Invalid Drive dataset size")
    if not isinstance(config.get("dataset_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", config["dataset_sha256"]):
        raise ValueError("Invalid Drive dataset SHA-256")
    return config


def validate_token_data(data: dict) -> None:
    if data.get("type") != "authorized_user" or set(data.get("scopes", [])) != {DRIVE_SCOPE}:
        raise ValueError("Only dedicated drive.file credentials are accepted; run setup_colab_drive.py")
    if not all(data.get(key) for key in ("client_id", "client_secret", "refresh_token")):
        raise ValueError("Incomplete Drive credentials; run setup_colab_drive.py")
    if data.get("token_uri", "https://oauth2.googleapis.com/token") != "https://oauth2.googleapis.com/token":
        raise ValueError("Only Google's OAuth token endpoint is accepted")


def build_service(token_path: Path):
    """Check actual token scope, not just the scope label in a local JSON file."""
    data = json.loads(Path(token_path).read_text(encoding="utf-8"))
    validate_token_data(data)
    try:
        import httplib2
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        import google_auth_httplib2
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError("Install scripts/requirements_colab_drive.txt in this Python environment") from exc
    credentials = Credentials.from_authorized_user_info(data, scopes=[DRIVE_SCOPE])
    # Do not include provider exception bodies, which may contain sensitive details.
    try:
        credentials.refresh(Request())
        with urlopen("https://oauth2.googleapis.com/tokeninfo?" + urlencode({"access_token": credentials.token}), timeout=30) as response:
            scope = json.load(response).get("scope", "")
    except Exception:
        raise RuntimeError("Drive authorization could not be verified; repeat restricted setup") from None
    if set(scope.split()) != {DRIVE_SCOPE}:
        raise ValueError("Google granted broader access than drive.file; use a dedicated OAuth client")
    http = google_auth_httplib2.AuthorizedHttp(credentials, http=httplib2.Http(timeout=120))
    return build("drive", "v3", http=http, cache_discovery=False)


def build_shared_service():
    """Use credentials obtained interactively on this Colab VM, never the Mac token."""
    try:
        import google.auth
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError("Install scripts/requirements_colab_drive.txt on the Colab VM") from exc
    # colab auth installs its user credential here. Loading this exact VM file
    # avoids silently falling back to Colab's unrelated service credentials.
    adc_path = Path("/content/.adc/adc.json")
    if not adc_path.is_file():
        raise RuntimeError("VM Drive authorization is missing; run `colab auth -s SESSION` as the dedicated account")
    credentials, _ = google.auth.load_credentials_from_file(
        str(adc_path), scopes=["https://www.googleapis.com/auth/drive"])
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def file_hashes(path: Path) -> tuple[str, str]:
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


class DriveStore:
    def __init__(self, config: dict, service):
        if config["root_folder_id"] != ROOT_FOLDER_ID:
            raise ValueError("Wrong Drive root folder")
        self.config, self.service = config, service
        self.allowed = {ROOT_FOLDER_ID, config["wheel_file_id"], config["environment_file_id"],
                        config["dataset_folder_id"], config["dataset_file_id"], *config["wheel_ancestor_ids"]}
        self.run_folders = set()
        self.run_parent_id = ROOT_FOLDER_ID

    def metadata(self, file_id: str) -> dict:
        if file_id not in self.allowed:
            raise ValueError("Refusing access to an unapproved Drive file")
        result = self.service.files().get(fileId=file_id, fields=FIELDS).execute(num_retries=3)
        if result.get("trashed"):
            raise ValueError("Configured Drive file is in the trash")
        return result

    def preflight(self, *, include_dataset: bool = False) -> None:
        root = self.metadata(ROOT_FOLDER_ID)
        if root["mimeType"] != FOLDER_MIME:
            raise ValueError("mdlARC destination is not a folder")
        for file_id in (self.config["wheel_file_id"], self.config["environment_file_id"]):
            current = self.metadata(file_id)
            for ancestor in self.config["wheel_ancestor_ids"]:
                if current.get("parents") != [ancestor]:
                    raise ValueError("Wheel files must remain inside the approved mdlARC folder")
                current = self.metadata(ancestor)
                if current["mimeType"] != FOLDER_MIME:
                    raise ValueError("Wheel parent is not a folder")
        if include_dataset:
            folder = self.metadata(self.config["dataset_folder_id"])
            if folder.get("parents") != [ROOT_FOLDER_ID] or folder["mimeType"] != FOLDER_MIME:
                raise ValueError("Dataset folder must be directly inside mdlARC")
            dataset = self.metadata(self.config["dataset_file_id"])
            if (dataset.get("parents") != [self.config["dataset_folder_id"]]
                    or dataset.get("name") != self.config["dataset_name"]
                    or dataset.get("mimeType") != "application/zip"
                    or int(dataset.get("size", -1)) != self.config["dataset_size"]):
                raise ValueError("Configured dataset ZIP differs from the approved Drive file")

    def download_dataset_zip(self, destination: Path) -> dict:
        from googleapiclient.http import MediaIoBaseDownload
        self.preflight(include_dataset=True)
        if destination.name != self.config["dataset_name"] or destination.exists() or destination.is_symlink():
            raise ValueError("Dataset ZIP destination must be a new task-variations-v1.zip")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".partial")
        if temporary.exists() or temporary.is_symlink():
            raise FileExistsError(f"Dataset download staging path already exists: {temporary}")
        try:
            with temporary.open("xb") as target:
                request = self.service.files().get_media(fileId=self.config["dataset_file_id"])
                download = MediaIoBaseDownload(target, request, chunksize=8 * 1024 * 1024)
                done = False
                while not done:
                    _, done = download.next_chunk(num_retries=3)
            if (temporary.stat().st_size != self.config["dataset_size"]
                    or file_hashes(temporary)[1] != self.config["dataset_sha256"]):
                raise ValueError("Downloaded dataset ZIP failed size or SHA-256 verification")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return {"file_id": self.config["dataset_file_id"], "sha256": self.config["dataset_sha256"],
                "size": self.config["dataset_size"]}

    def create_run_folder(self, session: str) -> str:
        # New folder only: never look up or reuse a folder by name.
        result = self.service.files().create(body={
            "name": session, "mimeType": FOLDER_MIME, "parents": [self.run_parent_id],
        }, fields=FIELDS).execute()
        folder = result["id"]
        self.allowed.add(folder)
        self.accept_run_folder(folder)
        return folder

    def accept_run_folder(self, folder: str) -> None:
        # The separate worker receives only this newly created folder ID.
        self.allowed.add(folder)
        metadata = self.metadata(folder)
        if metadata.get("parents") != [self.run_parent_id] or metadata["mimeType"] != FOLDER_MIME:
            self.allowed.discard(folder)
            raise ValueError("Run folder must be directly inside the approved run parent")
        self.run_folders.add(folder)

    def download_wheel(self, wheel_path: Path, environment_path: Path) -> None:
        from googleapiclient.http import MediaIoBaseDownload
        import io
        # Compare the small Drive environment file with the reviewed local copy.
        buffer = io.BytesIO()
        request = self.service.files().get_media(fileId=self.config["environment_file_id"])
        download = MediaIoBaseDownload(buffer, request, chunksize=64 * 1024)
        done = False
        while not done:
            _, done = download.next_chunk(num_retries=3)
            if buffer.tell() > 64 * 1024:
                raise ValueError("Unexpectedly large wheel environment file")
        if json.loads(buffer.getvalue()) != json.loads(environment_path.read_text()):
            raise ValueError("Drive wheel environment differs from the reviewed local manifest")
        if self.metadata(self.config["wheel_file_id"])["name"] != wheel_path.name:
            raise ValueError("Configured Drive wheel filename differs from its manifest")
        temporary = wheel_path.with_suffix(".whl.partial")
        try:
            with temporary.open("wb") as target:
                download = MediaIoBaseDownload(target, self.service.files().get_media(fileId=self.config["wheel_file_id"]), chunksize=8 * 1024 * 1024)
                done = False
                while not done:
                    _, done = download.next_chunk(num_retries=3)
            expected = json.loads(environment_path.read_text())["sha256"]
            if file_hashes(temporary)[1] != expected:
                raise ValueError("Downloaded FlashAttention wheel failed SHA-256 verification")
            temporary.replace(wheel_path)
        finally:
            temporary.unlink(missing_ok=True)

    def upload_new(self, path: Path, folder: str, name: str) -> dict:
        from googleapiclient.http import MediaFileUpload
        if folder not in self.run_folders:
            raise ValueError("Uploads require a verified run folder inside mdlARC")
        # Recheck ancestry so moving a run folder cannot redirect an upload.
        self.accept_run_folder(folder)
        if path.is_symlink() or not path.is_file():
            raise ValueError("Refusing a linked or missing backup file")
        md5, sha256 = file_hashes(path)
        size = path.stat().st_size
        media = MediaFileUpload(str(path), mimetype="application/octet-stream", chunksize=8 * 1024 * 1024, resumable=True)
        request = self.service.files().create(body={"name": name, "parents": [folder]}, media_body=media, fields=FIELDS)
        result = None
        try:
            while result is None:
                _, result = request.next_chunk(num_retries=3)
        except Exception:
            # Do not restart files.create after an ambiguous completion: this
            # could create a duplicate. Leave the runtime for recovery instead.
            raise RuntimeError("Drive upload did not confirm completion; keep the runtime for recovery") from None
        self.allowed.add(result["id"])
        result = self.metadata(result["id"])
        if result.get("parents") != [folder] or int(result.get("size", -1)) != size or result.get("md5Checksum") != md5:
            raise RuntimeError("Drive backup verification failed; keep the runtime for recovery")
        return {"id": result["id"], "url": f"https://drive.google.com/file/d/{result['id']}/view", "size": size, "sha256": sha256, "md5": md5}


class SharedDriveStore(DriveStore):
    """The dedicated account can read mdlARC and write only below mdlARC/runs."""

    def preflight(self, *, include_dataset: bool = False) -> None:
        user = self.service.about().get(fields="user(emailAddress)").execute(num_retries=3)
        email = user.get("user", {}).get("emailAddress", "").casefold()
        if email != EXPERIMENT_ACCOUNT:
            raise ValueError(f"Colab Drive account must be {EXPERIMENT_ACCOUNT}; got {email or 'unknown'}")
        super().preflight(include_dataset=include_dataset)
        runs_id = self.config["runs_folder_id"]
        self.allowed.add(runs_id)
        folder = self.metadata(runs_id)
        if (folder.get("name") != "runs" or folder.get("parents") != [ROOT_FOLDER_ID]
                or folder.get("mimeType") != FOLDER_MIME):
            raise ValueError("Configured mdlARC/runs folder is missing or has moved")
        branch = os.environ.get("MDLARC_RUN_BRANCH")
        if branch:
            from urllib.parse import quote
            if branch in {".", ".."} or any(ord(char) < 32 for char in branch):
                raise ValueError("Invalid run branch")
            name = quote(branch, safe="-_.")
            response = self.service.files().list(
                q=f"'{runs_id}' in parents and trashed = false and mimeType = '{FOLDER_MIME}'",
                fields="files(id,name),nextPageToken", pageSize=1000,
            ).execute(num_retries=3)
            matches = [item for item in response["files"] if item["name"] == name]
            if response.get("nextPageToken"):
                raise ValueError("Too many branch folders; configure a specific branch folder ID")
            if len(matches) > 1:
                raise ValueError("Duplicate Drive branch folders; resolve before launching")
            if matches:
                parent_id = matches[0]["id"]
            else:
                parent_id = self.service.files().create(body={
                    "name": name, "mimeType": FOLDER_MIME, "parents": [runs_id],
                }, fields=FIELDS).execute()["id"]
            self.allowed.add(parent_id)
            parent = self.metadata(parent_id)
            if (parent.get("parents") != [runs_id] or parent.get("name") != name
                    or parent.get("mimeType") != FOLDER_MIME):
                raise ValueError("Drive branch folder differs from the selected branch")
            self.run_parent_id = parent_id
        else:
            # Recovery of historical snapshots without branch metadata retains
            # the original parent. New launchers always send MDLARC_RUN_BRANCH.
            self.run_parent_id = runs_id
        capabilities = self.service.files().get(
            fileId=self.run_parent_id, fields="capabilities(canAddChildren)").execute(num_retries=3)
        if not capabilities.get("capabilities", {}).get("canAddChildren", False):
            raise ValueError("Dedicated account cannot create folders inside the run branch")

    def _download_checkpoint_ranges(self, file_id: str, target, expected_size: int) -> None:
        """Stream large ranges using the already-verified API client's credential."""
        from google.auth.transport.requests import AuthorizedSession
        from requests import RequestException

        request = self.service.files().get_media(fileId=file_id)
        with AuthorizedSession(request.http.credentials, refresh_timeout=30) as session:
            while target.tell() < expected_size:
                last = min(target.tell() + CHECKPOINT_API_RANGE_BYTES, expected_size) - 1
                retries = 0
                while target.tell() <= last:
                    first = target.tell()
                    try:
                        with session.get(request.uri, headers={
                                "Range": f"bytes={first}-{last}", "Accept-Encoding": "identity",
                                }, stream=True, timeout=(15, 120)) as response:
                            if response.status_code in {408, 429, 500, 502, 503, 504}:
                                response.raise_for_status()
                            if response.status_code != 206:
                                raise ValueError(f"Checkpoint API expected partial content; got HTTP {response.status_code}")
                            if response.headers.get("Content-Range") != f"bytes {first}-{last}/{expected_size}":
                                raise ValueError("Checkpoint API returned an unexpected byte range")
                            if response.headers.get("Content-Encoding", "identity") != "identity":
                                raise ValueError("Checkpoint API returned compressed range content")
                            for block in response.iter_content(chunk_size=CHECKPOINT_API_STREAM_BYTES):
                                if target.tell() + len(block) > last + 1:
                                    raise ValueError("Checkpoint API response exceeded the requested range")
                                target.write(block)
                            if target.tell() != last + 1:
                                raise ValueError("Checkpoint API returned an incomplete byte range")
                    except RequestException:
                        # Reads can retry from the last byte written on this VM.
                        # Final size and hashes still gate checkpoint publication.
                        if target.tell() == last + 1:
                            break
                        if retries == 3:
                            raise RuntimeError("Checkpoint API transfer failed after bounded retries") from None
                        time.sleep(2 ** retries)
                        retries += 1
                print(f"Input checkpoint downloaded: {target.tell():,}/{expected_size:,} bytes", flush=True)

    def _mounted_checkpoint_source(self, runs_root: Path, run: dict,
                                   branch_id: str, filename: str) -> Path:
        """Resolve only the API-verified ancestors; never search the mount."""
        runs_root = Path(runs_root)
        if not runs_root.is_absolute() or not runs_root.is_dir():
            raise ValueError("Checkpoint mount must be an existing absolute mdlARC/runs directory on the VM")
        names = []
        if branch_id != self.config["runs_folder_id"]:
            branch = self.metadata(branch_id)
            if (branch.get("parents") != [self.config["runs_folder_id"]]
                    or branch.get("mimeType") != FOLDER_MIME):
                raise ValueError("Mounted checkpoint branch must be directly inside mdlARC/runs")
            names.append(branch.get("name"))
        names.extend([run.get("name"), filename])
        for name in names:
            if (not isinstance(name, str) or not name or name in {".", ".."}
                    or any(char in name for char in "/\\")
                    or any(ord(char) < 32 for char in name)):
                raise ValueError("Unsafe name in mounted checkpoint ancestry")
        source = runs_root.joinpath(*names)
        if (source.is_symlink() or not source.is_file()
                or not source.resolve().is_relative_to(runs_root.resolve())):
            raise ValueError("Verified checkpoint is missing or escapes the supplied Drive mount")
        return source

    def download_input_checkpoint(self, file_id: str, destination: Path, *,
                                  expected_name: str, expected_size: int, expected_sha256: str | None,
                                  source_branch: str | None = None,
                                  mounted_runs_root: Path | None = None) -> dict:
        if self.run_parent_id == ROOT_FOLDER_ID:
            raise ValueError("Preflight the shared runs folder before downloading a checkpoint")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", file_id):
            raise ValueError("Invalid checkpoint Drive ID")
        if (Path(expected_name).name != expected_name or not expected_name.endswith(".pt")
                or expected_size <= 0 or (expected_sha256 is not None
                                         and not re.fullmatch(r"[0-9a-f]{64}", expected_sha256))):
            raise ValueError("Invalid checkpoint identity")
        self.allowed.add(file_id)
        metadata = self.metadata(file_id)
        parents = metadata.get("parents", [])
        if (metadata.get("name") != expected_name or metadata.get("mimeType") == FOLDER_MIME
                or int(metadata.get("size", -1)) != expected_size or len(parents) != 1
                or not metadata.get("md5Checksum")):
            raise ValueError("Drive checkpoint metadata differs from the expected file")
        parent_id = parents[0]
        self.allowed.add(parent_id)
        parent = self.metadata(parent_id)
        source_parent = self.run_parent_id
        if source_branch is not None:
            from urllib.parse import quote
            if (not isinstance(source_branch, str) or not source_branch or source_branch in {".", ".."}
                    or any(ord(c) < 32 for c in source_branch) or len(parent.get("parents", [])) != 1):
                raise ValueError("Invalid explicit checkpoint source branch")
            source_parent = parent["parents"][0]
            self.allowed.add(source_parent)
            branch_folder = self.metadata(source_parent)
            if (branch_folder.get("parents") != [self.config["runs_folder_id"]]
                    or branch_folder.get("name") != quote(source_branch, safe="-_.")
                    or branch_folder.get("mimeType") != FOLDER_MIME):
                raise ValueError("Input checkpoint ancestry differs from its explicit source branch")
        if parent.get("parents") != [source_parent] or parent.get("mimeType") != FOLDER_MIME:
            raise ValueError("Input checkpoint must be in a run folder under mdlARC/runs for the selected branch")
        checksum_source = "supplied"
        if expected_sha256 is None:
            expected_sha256 = metadata.get("sha256Checksum", "")
            if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
                raise ValueError("Drive has no SHA-256 for this checkpoint; supply checkpoint.sha256 in the spec (or --pretrain-sha256 for finetune-only) from the original backup ledger")
            checksum_source = "drive_metadata"
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Checkpoint staging path already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".partial")
        if temporary.exists() or temporary.is_symlink():
            raise FileExistsError(f"Checkpoint staging path already exists: {temporary}")
        transfer_method = "mounted_copy" if mounted_runs_root is not None else "drive_api"
        started = time.perf_counter()
        partial_created = False
        try:
            if mounted_runs_root is not None:
                source = self._mounted_checkpoint_source(mounted_runs_root, parent, source_parent, expected_name)
                if source.stat().st_size != expected_size:
                    raise ValueError("Mounted checkpoint size differs from the verified Drive file")
                mounted_root = Path(mounted_runs_root).resolve()
                mount_boundary = next((path for path in (mounted_root, *mounted_root.parents)
                                       if path.parent != path and path.is_mount()), mounted_root)
                if destination.resolve().is_relative_to(mount_boundary):
                    raise ValueError("Checkpoint destination must be outside the Drive mount")
                print("Copying input checkpoint from mounted Drive to VM-local storage...", flush=True)
                with source.open("rb") as input_file, temporary.open("xb") as target:
                    partial_created = True
                    shutil.copyfileobj(input_file, target, length=16 * 1024 * 1024)
            else:
                print("Downloading input checkpoint with 256 MiB API ranges...", flush=True)
                with temporary.open("xb") as target:
                    partial_created = True
                    self._download_checkpoint_ranges(file_id, target, expected_size)
            transfer_seconds = time.perf_counter() - started
            verification_started = time.perf_counter()
            md5, sha256 = file_hashes(temporary)
            if (temporary.stat().st_size != expected_size or md5 != metadata["md5Checksum"]
                    or sha256 != expected_sha256):
                raise ValueError("Downloaded checkpoint failed size, MD5, or SHA-256 verification")
            temporary.replace(destination)
            verification_seconds = time.perf_counter() - verification_started
        finally:
            if partial_created:
                temporary.unlink(missing_ok=True)
        return {"file_id": file_id, "name": expected_name, "size": expected_size,
                "sha256": expected_sha256, "sha256_source": checksum_source, "source_run_folder_id": parent_id,
                "transfer_method": transfer_method, "transfer_seconds": transfer_seconds,
                "verification_seconds": verification_seconds,
                **({"api_range_bytes": CHECKPOINT_API_RANGE_BYTES,
                    "api_stream_bytes": CHECKPOINT_API_STREAM_BYTES,
                    "accept_encoding": "identity"} if mounted_runs_root is None else {})}


def process_checkpoint(event_path: Path, run_dir: Path, store, folder: str) -> dict:
    event = json.loads(event_path.read_text())
    relative = Path(event["path"])
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".pt":
        raise ValueError("Unsafe checkpoint queue entry")
    path = run_dir / relative
    path.resolve().relative_to(run_dir.resolve())
    if path.is_symlink() or path.stat().st_size != event["size"]:
        raise ValueError("Checkpoint changed after publication")
    return {"path": relative.as_posix(), **store.upload_new(path, folder, relative.as_posix().replace("/", "__"))}


def worker(store, folder: str, run_dir: Path, queue: Path, stop: Path) -> int:
    ledger_path = run_dir / "checkpoint_backups.json"
    ledger = {"folder_id": folder, "folder_url": f"https://drive.google.com/drive/folders/{folder}", "complete": False, "files": [], "errors": []}
    handled = set()
    atomic_json(ledger_path, ledger)
    while True:
        for event in sorted(queue.glob("*.json")):
            if event.name in handled:
                continue
            handled.add(event.name)
            try:
                result = process_checkpoint(event, run_dir, store, folder)
                ledger["files"].append(result)
                print(f"Backed up {result['path']} to Drive", flush=True)
            except Exception as exc:
                # Provider errors can contain request details; keep them out of
                # downloaded records and do not retry a whole create blindly.
                ledger["errors"].append({"event": event.name, "error": type(exc).__name__})
                print(f"Checkpoint backup failed ({type(exc).__name__}); recovery required", flush=True)
            atomic_json(ledger_path, ledger)
        for event in queue.glob("*.json"):
            prune_backed_up_checkpoints(run_dir, json.loads(event.read_text()), ledger)
        atomic_json(ledger_path, ledger)
        if stop.exists():
            # The parent writes this only after the training subprocess exits.
            if any(path.name not in handled for path in queue.glob("*.json")):
                continue
            ledger["complete"] = not ledger["errors"]
            atomic_json(ledger_path, ledger)
            return 0 if ledger["complete"] else 1
        time.sleep(0.25)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    auth = parser.add_mutually_exclusive_group(required=True)
    auth.add_argument("--credentials", type=Path)
    auth.add_argument("--shared-account", action="store_true")
    parser.add_argument("--folder", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--stop", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    store = (SharedDriveStore(config, build_shared_service()) if args.shared_account
             else DriveStore(config, build_service(args.credentials)))
    store.preflight()
    store.accept_run_folder(args.folder)
    return worker(store, args.folder, args.run_dir, args.queue, args.stop)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Drive backup process failed ({type(exc).__name__}); recovery required", file=sys.stderr)
        raise SystemExit(1) from None
