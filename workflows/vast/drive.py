"""Dedicated-account Drive OAuth, immutable verified backups, and direct recovery."""

import json
from pathlib import Path
import re
import time

try:
    from .common import ACCOUNT, DRIVE_SCOPE, ROOT_FOLDER, atomic_json, branch_folder, hashes
except ImportError:
    from common import ACCOUNT, DRIVE_SCOPE, ROOT_FOLDER, atomic_json, branch_folder, hashes

CHECKPOINT_API_RANGE_BYTES = 256 * 1024 * 1024
CHECKPOINT_API_STREAM_BYTES = 8 * 1024 * 1024

FOLDER = "application/vnd.google-apps.folder"
FIELDS = "id,name,mimeType,parents,size,md5Checksum,sha256Checksum,trashed,createdTime,appProperties,capabilities(canAddChildren)"


def file_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("Invalid Google Drive file ID")
    return value


def credentials(path):
    from google.oauth2.credentials import Credentials
    data = json.loads(Path(path).read_text())
    if (data.get("type") != "authorized_user" or set(data.get("scopes", [])) != {DRIVE_SCOPE}
            or not all(data.get(k) for k in ("client_id", "client_secret", "refresh_token"))
            or data.get("token_uri", "https://oauth2.googleapis.com/token") != "https://oauth2.googleapis.com/token"):
        raise ValueError("Use the separate Vast OAuth setup with Drive-only offline authorization")
    return Credentials.from_authorized_user_info(data, scopes=[DRIVE_SCOPE])


def service(token):
    import httplib2
    import google_auth_httplib2
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    auth = credentials(token)
    try:
        auth.refresh(Request())
        transport = httplib2.Http(timeout=60)
        # Drive uses 308 for resumable-upload progress, without a redirect URL.
        # Match googleapiclient.http.build_http() so large uploads reach next_chunk.
        transport.redirect_codes = transport.redirect_codes - {308}
        http = google_auth_httplib2.AuthorizedHttp(auth, http=transport)
        return build("drive", "v3", http=http, cache_discovery=False)
    except Exception:
        raise RuntimeError("Drive authorization failed; repeat Vast OAuth setup") from None


class Drive:
    def __init__(self, api, config, branch, *, control=None, max_upload_gb=None, max_download_gb=None, transfer_file=None):
        if config.get("root_folder_id") != ROOT_FOLDER:
            raise ValueError("Drive root must be the existing mdlARC folder")
        self.api, self.config, self.branch = api, config, branch
        file_id(config["runs_folder_id"])
        self.control = Path(control) if control else None
        self.max_upload = None if max_upload_gb is None else max_upload_gb * 1e9
        self.max_download = None if max_download_gb is None else max_download_gb * 1e9
        self.branch_id = None
        self.transfer_file = Path(transfer_file) if transfer_file else None
        self.transferred = {"upload": 0, "download": 0}
        self.journal = {"files": {}}
        if self.control:
            self.control.mkdir(parents=True, exist_ok=True, mode=0o700)
            if (self.control / "uploads.json").exists():
                self.journal = json.loads((self.control / "uploads.json").read_text())
            if (self.control / "transfers.json").exists():
                self.transferred = json.loads((self.control / "transfers.json").read_text())

    def save(self):
        if self.control:
            atomic_json(self.control / "uploads.json", self.journal, private=True)
            atomic_json(self.control / "transfers.json", self.transferred, private=True)

    def charge(self, direction, size):
        from .common import lock
        if self.transfer_file:
            with lock(self.transfer_file.with_suffix(".lock"), blocking=True):
                self.transferred = (json.loads(self.transfer_file.read_text()) if self.transfer_file.exists()
                                    else {"upload": 0, "download": 0})
                self._charge(direction, size)
                atomic_json(self.transfer_file, self.transferred, private=True)
            return
        self._charge(direction, size)

    def _charge(self, direction, size):
        maximum = self.max_upload if direction == "upload" else self.max_download
        if maximum is not None and self.transferred[direction] + size > maximum:
            raise RuntimeError("Drive transfer allowance exceeded; remote data retained")
        self.transferred[direction] += size
        self.save()

    def metadata(self, identity):
        result = self.api.files().get(fileId=file_id(identity), fields=FIELDS).execute(num_retries=3)
        if result.get("trashed"):
            raise ValueError("Drive file is in the trash")
        return result

    def children(self, parent, *, folders=False, name=None):
        query = f"'{file_id(parent)}' in parents and trashed = false"
        if folders:
            query += f" and mimeType = '{FOLDER}'"
        if name is not None:
            query += " and name = '" + name.replace("\\", "\\\\").replace("'", "\\'") + "'"
        result, page = [], None
        while True:
            response = self.api.files().list(q=query, fields=f"files({FIELDS}),nextPageToken",
                                             pageSize=1000, pageToken=page).execute(num_retries=3)
            result.extend(response.get("files", []))
            page = response.get("nextPageToken")
            if not page:
                return result

    def preflight(self, *, create_branch=False):
        email = self.api.about().get(fields="user(emailAddress),storageQuota").execute(num_retries=3).get("user", {}).get("emailAddress", "")
        if email.casefold() != ACCOUNT:
            raise ValueError(f"Drive authorization must belong to {ACCOUNT}")
        root = self.metadata(ROOT_FOLDER)
        runs = self.metadata(self.config["runs_folder_id"])
        if root.get("mimeType") != FOLDER or runs.get("mimeType") != FOLDER or runs.get("parents") != [ROOT_FOLDER] or runs.get("name") != "runs":
            raise ValueError("Configured mdlARC/runs folder is missing or moved")
        if not runs.get("capabilities", {}).get("canAddChildren"):
            raise ValueError("Experiment account cannot write to mdlARC/runs")
        name = branch_folder(self.branch)
        matches = self.children(runs["id"], folders=True, name=name)
        if len(matches) > 1:
            raise ValueError("Duplicate Drive branch folders; resolve before launching")
        if matches:
            self.branch_id = matches[0]["id"]
        elif create_branch:
            self.branch_id = self.create_folder(runs["id"], name)["id"]
        return self.branch_id

    def create_folder(self, parent, name, *, properties=None):
        # Preallocated IDs make ambiguous creation recoverable without duplicate folders.
        if not self.control:
            raise ValueError("Folder creation requires a persistent private control journal")
        if parent not in {self.config["runs_folder_id"], self.branch_id}:
            raise ValueError("Folder writes are restricted to mdlARC/runs and the selected branch")
        key = f"folder:{parent}:{name}"
        entry = self.journal["files"].get(key)
        if entry is None:
            identity = self.api.files().generateIds(count=1, space="drive").execute(num_retries=3)["ids"][0]
            entry = self.journal["files"][key] = {"id": identity}
            self.save()
        current = self._existing(entry["id"])
        if current is None:
            try:
                current = self.api.files().create(body={"id": entry["id"], "name": name, "mimeType": FOLDER,
                    "parents": [parent], "appProperties": properties or {}}, fields=FIELDS).execute(num_retries=0)
            except Exception:
                current = self._existing(entry["id"])
                if current is None:
                    raise RuntimeError("Drive folder creation is unconfirmed; reconcile the recorded ID") from None
        if current.get("parents") != [parent] or current.get("name") != name or current.get("mimeType") != FOLDER:
            raise ValueError("Drive folder identity changed")
        return current

    def _existing(self, identity):
        from googleapiclient.errors import HttpError
        try:
            return self.metadata(identity)
        except HttpError as exc:
            if exc.resp.status == 404:
                return None
            raise RuntimeError("Drive metadata lookup failed") from None

    def create_run(self, session):
        from .common import safe_name
        safe_name(session)
        self.preflight(create_branch=True)
        matches = self.children(self.branch_id, folders=True, name=session)
        key = f"folder:{self.branch_id}:{session}"
        recorded = self.journal["files"].get(key, {}).get("id")
        if matches and (len(matches) != 1 or matches[0]["id"] != recorded):
            raise ValueError("Session already exists on Drive; choose a new session or recover its saved record")
        return self.create_folder(self.branch_id, session, properties={"backend": "vast", "session": session})["id"]

    def verify_run(self, folder, session=None):
        if not self.branch_id:
            self.preflight()
        metadata = self.metadata(folder)
        if (not self.branch_id or metadata.get("parents") != [self.branch_id] or metadata.get("mimeType") != FOLDER
                or (session is not None and metadata.get("name") != session)):
            raise ValueError("Run folder is outside the selected branch or session")
        return metadata

    def upload(self, path, folder, name):
        from googleapiclient.http import MediaFileUpload
        self.verify_run(folder)
        path = Path(path)
        if path.is_symlink() or not path.is_file() or not self.control:
            raise ValueError("Backup requires a regular file and private persistent journal")
        digest = hashes(path)
        key = f"file:{folder}:{name}:{digest['sha256']}"
        entry = self.journal["files"].get(key)
        if entry is None:
            identity = self.api.files().generateIds(count=1, space="drive").execute(num_retries=3)["ids"][0]
            entry = self.journal["files"][key] = {"id": identity, **digest}
            self.save()
        metadata = self._existing(entry["id"])
        if metadata is None:
            if not entry.get("charged"):
                self.charge("upload", digest["size"])
                entry["charged"] = True
                self.save()
            request = self.api.files().create(body={"id": entry["id"], "name": name, "parents": [folder],
                "appProperties": {"sha256": digest["sha256"], "backend": "vast"}},
                media_body=MediaFileUpload(str(path), chunksize=8 * 1024 * 1024, resumable=True), fields=FIELDS)
            if entry.get("uri"):
                request.resumable_uri = entry["uri"]
                request.resumable_progress = entry.get("progress", 0)
            result = None
            try:
                while result is None:
                    _, result = request.next_chunk(num_retries=3)
                    entry.update(uri=request.resumable_uri, progress=request.resumable_progress)
                    self.save()
            except Exception:
                entry.update(uri=request.resumable_uri, progress=request.resumable_progress)
                self.save()
                # The same preallocated ID is checked; no second files.create with a new ID.
                metadata = self._existing(entry["id"])
                if metadata is None:
                    raise RuntimeError("Drive upload is incomplete; saved transfer can be resumed") from None
            metadata = metadata or self.metadata(entry["id"])
        self._verify_binary(metadata, folder, name, digest)
        if hashes(path) != digest:
            raise ValueError("Backup source changed while uploading")
        entry["verified"] = True
        entry.pop("uri", None)
        self.save()
        return {"id": entry["id"], "name": name, **digest}

    @staticmethod
    def _verify_binary(metadata, folder, name, digest):
        if (metadata.get("parents") != [folder] or metadata.get("name") != name
                or int(metadata.get("size", -1)) != digest["size"] or metadata.get("md5Checksum") != digest["md5"]
                or (metadata.get("sha256Checksum") and metadata["sha256Checksum"] != digest["sha256"])):
            raise ValueError("Drive backup checksum, size, or destination verification failed")

    def assert_under(self, identity, ancestor):
        current = self.metadata(identity)
        for _ in range(16):
            if current["id"] == ancestor:
                return
            if len(current.get("parents", [])) != 1:
                break
            current = self.metadata(current["parents"][0])
        raise ValueError("Drive input is outside the approved mdlARC folder")

    def _download_checkpoint_ranges(self, file_id: str, target, expected_size: int) -> None:
        """Stream large ranges using the already-verified API client's credential."""
        from google.auth.transport.requests import AuthorizedSession
        from requests import RequestException

        request = self.api.files().get_media(fileId=file_id)
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

    def download(self, identity, destination, *, expected=None, ancestor=None):
        from googleapiclient.http import MediaIoBaseDownload
        if ancestor:
            self.assert_under(identity, ancestor)
        metadata = self.metadata(identity)
        if metadata.get("mimeType") == FOLDER or not metadata.get("md5Checksum"):
            raise ValueError("Expected a checksummed binary Drive file")
        expected = expected or {}
        if (("name" in expected and expected["name"] != metadata["name"])
                or ("size" in expected and expected["size"] != int(metadata["size"]))):
            raise ValueError("Drive input metadata differs from its recorded identity")
        sha = expected.get("sha256")
        if sha == "drive":
            sha = metadata.get("sha256Checksum")
            if not sha:
                raise ValueError("Drive has no SHA-256; use the original checkpoint ledger's digest")
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("Input destination already exists")
        temporary = destination.with_name(destination.name + ".partial")
        try:
            self.charge("download", int(metadata["size"]))
            with temporary.open("xb") as out:
                if int(metadata["size"]) >= CHECKPOINT_API_RANGE_BYTES:
                    print("Downloading input with streamed 256 MiB API ranges...", flush=True)
                    self._download_checkpoint_ranges(identity, out, int(metadata["size"]))
                else:
                    request = MediaIoBaseDownload(out, self.api.files().get_media(fileId=identity), chunksize=8 * 1024 * 1024)
                    done = False
                    while not done:
                        _, done = request.next_chunk(num_retries=3)
            digest = hashes(temporary)
            if digest["size"] != int(metadata["size"]) or digest["md5"] != metadata["md5Checksum"] or (sha and digest["sha256"] != sha):
                raise ValueError("Drive download verification failed")
            temporary.replace(destination)
            return {"id": identity, "name": metadata["name"], **digest}
        finally:
            temporary.unlink(missing_ok=True)
