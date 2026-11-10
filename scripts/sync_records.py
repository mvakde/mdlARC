#!/usr/bin/env python3
"""Synchronize small research records across laptops through immutable Drive uploads.

This file is standalone so its installed copy survives source branch switches.
See README_record_sync.md for scope, conflict resolution, and authentication.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import uuid

ROOT_FOLDER = "DRIVE_ID_05_PLACEHOLDER"
RUNS_FOLDER = "DRIVE_ID_10_PLACEHOLDER"
SHARED_FOLDER = "shared-records"
ACCOUNT = "your-drive-account@example.com"
SCOPE = "https://www.googleapis.com/auth/drive"
DEFAULT_TOKEN = Path.home() / ".config/mdlarc-colab-oauth/your-drive-account/token.json"
CONTROL = "results/_shared/record_sync"
FOLDER_MIME = "application/vnd.google-apps.folder"
FIELDS = "id,name,mimeType,parents,size,md5Checksum,trashed,capabilities(canAddChildren)"
MAX_FILE = 4 * 1024 * 1024
MAX_BUNDLE = 32 * 1024 * 1024
REVISION = re.compile(r"[0-9a-f]{32}")
SECRET_KEYS = {"access_token", "refresh_token", "client_secret", "private_key",
               "api_key", "kaggle_api_token", "vast_api_key", "runpod_api_key", "authorization"}


class SyncError(RuntimeError):
    pass


def digest(data):
    return hashlib.sha256(data).hexdigest()


def selected(name):
    """Only portable text records, never live connection state or bulk artifacts."""
    path = PurePosixPath(name)
    parts = path.parts
    if (path.is_absolute() or path.as_posix() != name or not parts
            or any(p in {".", ".."} or p.startswith(".") for p in parts)
            or "\\" in name or "\x00" in name):
        return False
    if parts[:2] == ("specs", "temp specs"):
        return len(parts) >= 3 and path.suffix == ".json"
    if len(parts) < 4 or parts[0] != "results" or parts[1] == "_shared":
        return False
    # The pre-encoding layout results/codex/<branch>/ remains readable too.
    start = 3 if parts[1] == "codex" else 2
    tail = parts[start:]
    if not tail:
        return False
    if tail[0] == "experiments":
        return (len(tail) > 1 and path.suffix in {".md", ".json"}
                and (len(tail) == 2 or tail[1] in {"notes", "generated", "trials"})
                and not set(tail).intersection({"workspace", "source", "source_snapshot",
                                               "assets_tmp", "checkpoints", "recovered"}))
    if tail[0] not in {"colab_runs", "vast_runs", "runpod_runs", "sfcompute_runs"} or len(tail) < 3 or tail[1].startswith("_"):
        return False
    if len(tail) == 3:
        return tail[2] in {"spec.json", "run_status.json", "experiment.json"}
    if tail[2] == "runs" and len(tail) >= 5:
        return ((len(tail) == 5 and tail[4] in {"experiment.json", "checkpoint_backups.json"})
                or (len(tail) >= 7 and tail[4] == "evals" and tail[-1] == "score.json"))
    return False


def validate_text(name, data):
    if not selected(name) or len(data) > MAX_FILE:
        raise SyncError(f"Record is outside the text allowlist or too large: {name}")
    try:
        text = data.decode("utf-8")
        if "\x00" in text:
            raise ValueError()
        if name.endswith(".json"):
            def check(value):
                if isinstance(value, dict):
                    for key, item in value.items():
                        if key.casefold() in SECRET_KEYS:
                            raise SyncError(f"Credential field found; record was not uploaded: {name}")
                        check(item)
                elif isinstance(value, list):
                    for item in value:
                        check(item)
            check(json.loads(text))
        if re.search(r"-----BEGIN [A-Z ]*PRIVATE KEY-----|ya29\.[\w-]{20,}|gh[pousr]_[\w]{20,}", text):
            raise SyncError(f"Credential-like content found; record was not uploaded: {name}")
        if re.search(r'''(?i)(?:refresh_token|access_token|client_secret|api_key)["']?\s*[:=]\s*["']?[A-Za-z0-9_+./=-]{16,}''', text):
            raise SyncError(f"Credential-like content found; record was not uploaded: {name}")
    except (UnicodeError, ValueError, RecursionError):
        raise SyncError(f"Record must contain valid UTF-8 text/JSON: {name}") from None


def local_path(root, name):
    path = root
    for part in PurePosixPath(name).parts:
        if part in {".", ".."} or not part or PurePosixPath(name).is_absolute():
            raise SyncError("Unsafe local path")
        path /= part
        if path.is_symlink():
            raise SyncError(f"Refusing a linked record or destination: {name}")
    if not path.is_relative_to(root):
        raise SyncError("Local destination escaped the checkout")
    return path


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".record-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def save_json(path, value):
    atomic_write(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def collect(root):
    result = {}
    for base in (root / "results", root / "specs/temp specs"):
        if base.is_symlink():
            raise SyncError("Keep record directories inside the checkout; no symbolic links")
        for parent, dirs, files in os.walk(base, followlinks=False):
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in {
                "_shared", "workspace", "source", "source_snapshot", "__pycache__",
                "assets_tmp", "checkpoints", "recovered"} and not (Path(parent) / d).is_symlink())
            for filename in sorted(files):
                path = Path(parent) / filename
                name = path.relative_to(root).as_posix()
                if selected(name):
                    path = local_path(root, name)
                    if path.stat().st_size > MAX_FILE:
                        raise SyncError(f"Record exceeds the 4 MiB limit: {name}")
                    data = path.read_bytes()
                    validate_text(name, data)
                    result[name] = data
    return result


def pack(changes, parents, revision=None):
    revision = revision or uuid.uuid4().hex
    manifest = {"format": 1, "revision": revision,
                "created": datetime.now(timezone.utc).isoformat(), "changes": {}}
    for name, data in sorted(changes.items()):
        validate_text(name, data)
        manifest["changes"][name] = {"sha256": digest(data), "size": len(data),
                                      "parents": sorted(parents.get(name, []))}
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name, data in [("manifest.json", json.dumps(manifest, sort_keys=True).encode()),
                               *(("files/" + name, data) for name, data in sorted(changes.items()))]:
                entry = tarfile.TarInfo(name)
                entry.size = len(data)
                archive.addfile(entry, io.BytesIO(data))
    data = raw.getvalue()
    if len(data) > MAX_BUNDLE or sum(map(len, changes.values())) > MAX_BUNDLE:
        raise SyncError("Record batch exceeds 32 MiB; split large research outputs before syncing")
    return revision, data


def unpack(data):
    try:
        if len(data) > MAX_BUNDLE:
            raise ValueError()
        # Bound decompression before inspecting archive headers or payloads.
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as compressed:
            expanded = compressed.read(MAX_BUNDLE + 16 * 1024 * 1024 + 1)
        if len(expanded) > MAX_BUNDLE + 16 * 1024 * 1024:
            raise ValueError()
        with tarfile.open(fileobj=io.BytesIO(expanded), mode="r:") as archive:
            members = archive.getmembers()
            if (not members or len(members) > 10001
                    or sum(m.size for m in members) > MAX_BUNDLE + MAX_FILE
                    or any(not m.isfile() or m.size > MAX_FILE for m in members)
                    or len({m.name for m in members}) != len(members)
                    or members[0].name != "manifest.json"):
                raise ValueError()
            manifest = json.loads(archive.extractfile(members[0]).read())
            revision = manifest["revision"]
            if manifest.get("format") != 1 or not REVISION.fullmatch(revision):
                raise ValueError()
            changes = manifest["changes"]
            if not isinstance(changes, dict) or set(m.name for m in members[1:]) != {"files/" + n for n in changes}:
                raise ValueError()
            payload = {}
            for name, info in changes.items():
                blob = archive.extractfile("files/" + name).read()
                validate_text(name, blob)
                if (info["size"] != len(blob) or info["sha256"] != digest(blob)
                        or not isinstance(info["parents"], list)
                        or any(not isinstance(p, str) or not REVISION.fullmatch(p) or p == revision for p in info["parents"])):
                    raise ValueError()
                payload[name] = blob
        return manifest, payload
    except SyncError:
        raise
    except (ValueError, KeyError, TypeError, AttributeError, tarfile.TarError, OSError, EOFError):
        raise SyncError("Invalid or damaged Drive record bundle; local records were preserved") from None


def remote_heads(snapshots):
    versions = {}
    seen = set()
    for manifest, payload in snapshots:
        revision = manifest["revision"]
        if revision in seen:
            raise SyncError("Duplicate Drive record revision; inspect the shared folder")
        seen.add(revision)
        for name, info in manifest["changes"].items():
            versions.setdefault(name, {})[revision] = {**info, "data": payload[name]}
    result = {}
    for name, entries in versions.items():
        visited, visiting = set(), set()
        def visit(revision):
            if revision in visiting or revision not in entries:
                raise SyncError("Record history is incomplete or cyclic; retry after checking Drive")
            if revision not in visited:
                visiting.add(revision)
                for parent in entries[revision]["parents"]:
                    visit(parent)
                visiting.remove(revision)
                visited.add(revision)
        for revision in entries:
            visit(revision)
        replaced = {p for info in entries.values() for p in info["parents"]}
        result[name] = {r: info for r, info in entries.items() if r not in replaced}
    return result


def plan(local, heads, baseline, resolutions=()):
    uploads, downloads, conflicts, parents = {}, {}, {}, {}
    resolutions = set(resolutions)
    if resolutions - heads.keys():
        raise SyncError("--resolve must name an existing shared record")
    for name in sorted(local.keys() | heads.keys()):
        versions = heads.get(name, {})
        parents[name] = sorted(versions)
        data = local.get(name)
        local_sha = digest(data) if data is not None else None
        hashes = {info["sha256"] for info in versions.values()}
        if name in resolutions:
            if data is None:
                raise SyncError(f"Create the reviewed local resolution first: {name}")
            uploads[name] = data
        elif not versions:
            uploads[name] = data
        elif len(hashes) > 1:
            conflicts[name] = versions
        else:
            remote_sha = next(iter(hashes))
            remote = next(iter(versions.values()))["data"]
            base_sha = baseline.get(name)
            if data is None or (base_sha == local_sha and local_sha != remote_sha):
                downloads[name] = remote
            elif local_sha == remote_sha:
                pass
            elif base_sha == remote_sha:
                uploads[name] = data
            else:
                conflicts[name] = versions
    return uploads, downloads, conflicts, parents


@contextmanager
def checkout_lock(root):
    path = local_path(root, CONTROL + "/sync.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SyncError("Another record sync is active in this checkout") from None
        yield


def synchronize(root, store, *, dry_run=False, resolutions=()):
    state_path = local_path(root, CONTROL + "/baseline.json")
    baseline = json.loads(state_path.read_text()) if state_path.exists() else {}
    local = collect(root)
    snapshots = store.read_snapshots(dry_run=dry_run)
    heads = remote_heads(snapshots)
    uploads, downloads, conflicts, parents = plan(local, heads, baseline, resolutions)
    summary = {"local_records": len(local), "upload": len(uploads),
               "download": len(downloads), "conflicts": sorted(conflicts), "dry_run": dry_run}
    if dry_run:
        return summary
    if conflicts:
        for name, versions in conflicts.items():
            directory = local_path(root, CONTROL + "/conflicts/" + digest(name.encode()))
            save_json(directory / "record.json", {"path": name, "revisions": sorted(versions)})
            for revision, info in versions.items():
                atomic_write(directory / (revision + PurePosixPath(name).suffix), info["data"])
        raise SyncError("Conflicting edits; nothing was uploaded or replaced. Review "
                        + CONTROL + "/conflicts/ and rerun with --resolve PATH after editing the local file. Paths: "
                        + ", ".join(sorted(conflicts)))
    # Recheck the complete selection before publishing; files can be edited during a download.
    if collect(root) != local:
        raise SyncError("Local records changed during sync; retry after saving edits")
    if uploads:
        revision, blob = pack(uploads, parents)
        store.publish(revision, blob)
    for name, data in downloads.items():
        path = local_path(root, name)
        before = path.read_bytes() if path.exists() else None
        if before != local.get(name):
            raise SyncError(f"Local file changed during sync; it was preserved: {name}")
        atomic_write(path, data)
    # Missing files are restored rather than interpreted as deletion requests.
    merged = {**local, **downloads}
    save_json(state_path, {name: digest(data) for name, data in merged.items()})
    return summary


def drive_service(token, root):
    """Reuse only the separately authorized Drive token; never a compute credential."""
    token = token.expanduser()
    if token.is_symlink():
        raise SyncError("Drive credential must not be a symbolic link")
    token = token.resolve()
    if token.is_relative_to(root) or any((p / ".git").exists() for p in token.parents):
        raise SyncError("Keep the Drive credential outside all Git checkouts")
    try:
        info = token.stat()
        if (not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077
                or info.st_uid != os.getuid() or info.st_size > 65536):
            raise ValueError()
        data = json.loads(token.read_text())
        if (data.get("type") != "authorized_user" or data.get("scopes") != [SCOPE]
                or data.get("token_uri", "https://oauth2.googleapis.com/token") != "https://oauth2.googleapis.com/token"
                or not all(data.get(k) for k in ("client_id", "client_secret", "refresh_token"))):
            raise ValueError()
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        import google_auth_httplib2
        import httplib2
        from urllib.parse import urlencode
        from urllib.request import urlopen
        credentials = Credentials.from_authorized_user_info(data, scopes=[SCOPE])
        request = Request()
        credentials.refresh(lambda **kw: request(**{**kw, "timeout": 30}))
        with urlopen("https://oauth2.googleapis.com/tokeninfo?" + urlencode({"access_token": credentials.token}), timeout=30) as response:
            granted = json.load(response)
        if set(granted.get("scope", "").split()) != {SCOPE}:
            raise ValueError()
        transport = google_auth_httplib2.AuthorizedHttp(credentials, http=httplib2.Http(timeout=30))
        api = build("drive", "v3", http=transport, cache_discovery=False)
        if api.about().get(fields="user(emailAddress)").execute(num_retries=2)["user"]["emailAddress"].casefold() != ACCOUNT:
            raise ValueError()
        return api
    except Exception:
        raise SyncError("Dedicated Drive authorization failed. Check the connection or renew it with "
                        "scripts/colab_oauth/setup.py --reauthorize; no VM was accessed.") from None


class DriveStore:
    def __init__(self, api, root):
        self.api = api
        self.control = local_path(root, CONTROL)
        self.folder = None
        outer = self.metadata(ROOT_FOLDER)
        runs = self.metadata(RUNS_FOLDER)
        if (outer["mimeType"] != FOLDER_MIME or runs["mimeType"] != FOLDER_MIME
                or runs.get("parents") != [ROOT_FOLDER] or runs["name"] != "runs"
                or not runs.get("capabilities", {}).get("canAddChildren")):
            raise SyncError("The existing mdlARC/runs folder is missing, moved, or not writable")

    def metadata(self, identity, *, missing=False):
        from googleapiclient.errors import HttpError
        try:
            result = self.api.files().get(fileId=identity, fields=FIELDS).execute(num_retries=2)
        except HttpError as exc:
            if missing and exc.resp.status == 404:
                return None
            raise
        if result.get("trashed"):
            raise SyncError("A saved Drive record identity is in the trash; inspect it before retrying")
        return result

    def children(self, parent):
        result, page = [], None
        while True:
            response = self.api.files().list(q=f"'{parent}' in parents and trashed = false",
                fields=f"files({FIELDS}),nextPageToken", pageSize=1000, pageToken=page).execute(num_retries=2)
            result.extend(response.get("files", []))
            page = response.get("nextPageToken")
            if not page:
                return result

    def get_folder(self, *, create=False):
        matches = [f for f in self.children(RUNS_FOLDER) if f["name"] == SHARED_FOLDER]
        if len(matches) > 1 or (matches and matches[0]["mimeType"] != FOLDER_MIME):
            raise SyncError("Duplicate or invalid shared-records folder; inspect Drive before syncing")
        if matches:
            self.folder = matches[0]["id"]
        elif create:
            journal = self.control / "folder-create.json"
            identity = json.loads(journal.read_text())["id"] if journal.exists() else self.new_id()
            save_json(journal, {"id": identity})
            item = self.metadata(identity, missing=True)
            if item is None:
                try:
                    self.api.files().create(body={"id": identity, "name": SHARED_FOLDER,
                        "mimeType": FOLDER_MIME, "parents": [RUNS_FOLDER]}, fields=FIELDS).execute(num_retries=0)
                except Exception:
                    pass
                item = self.metadata(identity, missing=True)
            if not item or item.get("parents") != [RUNS_FOLDER] or item["name"] != SHARED_FOLDER or item["mimeType"] != FOLDER_MIME:
                raise SyncError("Shared folder creation is unconfirmed; retry using its saved identity")
            self.folder = identity
            # Catch concurrent first-time setup before publishing records into either folder.
            if {f["id"] for f in self.children(RUNS_FOLDER) if f["name"] == SHARED_FOLDER} != {identity}:
                raise SyncError("Concurrent shared folder setup; inspect Drive before syncing")
        return self.folder

    def new_id(self):
        return self.api.files().generateIds(count=1, space="drive").execute(num_retries=2)["ids"][0]

    def read_snapshots(self, *, dry_run=False):
        pending = self.control / "pending.json"
        if pending.exists() and not dry_run:
            entry = json.loads(pending.read_text())
            self.publish(entry["revision"], (self.control / "pending.tar.gz").read_bytes())
        folder = self.get_folder()
        if not folder:
            return []
        snapshots = []
        for item in self.children(folder):
            match = re.fullmatch(r"records-v1-([0-9a-f]{32})\.tar\.gz", item["name"])
            if not match:
                continue
            size = int(item.get("size", -1))
            if item.get("parents") != [folder] or not 0 < size <= MAX_BUNDLE or not item.get("md5Checksum"):
                raise SyncError("Invalid record backup metadata")
            cache = self.control / "cache" / (item["id"] + ".tar.gz")
            blob = cache.read_bytes() if cache.exists() else b""
            if len(blob) != size or hashlib.md5(blob).hexdigest() != item["md5Checksum"]:
                blob = self.api.files().get_media(fileId=item["id"]).execute(num_retries=2)
                if len(blob) != size or hashlib.md5(blob).hexdigest() != item["md5Checksum"]:
                    raise SyncError("Drive record download checksum failed")
                atomic_write(cache, blob)
            snapshot = unpack(blob)
            if snapshot[0]["revision"] != match[1]:
                raise SyncError("Drive record revision differs from its filename")
            snapshots.append(snapshot)
        return snapshots

    def publish(self, revision, blob):
        from googleapiclient.http import MediaIoBaseUpload
        folder = self.get_folder(create=True)
        pending = self.control / "pending.json"
        if pending.exists():
            entry = json.loads(pending.read_text())
            if entry["revision"] != revision or entry["sha256"] != digest(blob) or entry["folder"] != folder:
                raise SyncError("Reconcile the saved pending upload before publishing another revision")
        else:
            entry = {"id": self.new_id(), "revision": revision, "sha256": digest(blob), "folder": folder}
            atomic_write(self.control / "pending.tar.gz", blob)
            save_json(pending, entry)
        name = f"records-v1-{revision}.tar.gz"
        item = self.metadata(entry["id"], missing=True)
        if item is None:
            try:
                self.api.files().create(body={"id": entry["id"], "name": name,
                    "parents": [folder], "mimeType": "application/gzip"},
                    media_body=MediaIoBaseUpload(io.BytesIO(blob), mimetype="application/gzip", resumable=False),
                    fields=FIELDS).execute(num_retries=0)
            except Exception:
                pass
            item = self.metadata(entry["id"], missing=True)
        if (not item or item.get("parents") != [folder] or item["name"] != name
                or int(item.get("size", -1)) != len(blob)
                or item.get("md5Checksum") != hashlib.md5(blob).hexdigest()):
            raise SyncError("Upload not confirmed; retry to reconcile the same saved Drive file ID")
        recovered = self.api.files().get_media(fileId=entry["id"]).execute(num_retries=2)
        if digest(recovered) != digest(blob):
            raise SyncError("Uploaded record readback failed; pending identity was retained")
        atomic_write(self.control / "cache" / (entry["id"] + ".tar.gz"), recovered)
        pending.unlink()


def install():
    if sys.prefix == sys.base_prefix:
        raise SyncError("Install using venv/bin/python so the copy stays inside this project's environment")
    target = Path(sys.executable).absolute().parent / "mdlarc-records"
    source = Path(__file__).read_text().split("\n", 1)[1]
    atomic_write(target, (f"#!{sys.executable}\n" + source).encode())
    target.chmod(0o755)
    print(f"Installed {target}; it remains available when code branches change.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Checkout directory; default is the current Git checkout")
    parser.add_argument("--token", type=Path, default=DEFAULT_TOKEN)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("install")
    commands.add_parser("inventory")
    sync = commands.add_parser("sync")
    sync.add_argument("--dry-run", action="store_true")
    sync.add_argument("--resolve", action="append", default=[], metavar="PATH",
                      help="Publish the reviewed local file as the resolution of all current versions")
    args = parser.parse_args(argv)
    if args.command == "install":
        install()
        return
    root = args.root
    if root is None:
        response = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True)
        if response.returncode:
            raise SyncError("Run from the mdlARC checkout or pass --root")
        root = Path(response.stdout.strip())
    root = root.expanduser().resolve()
    if args.command == "inventory":
        records = collect(root)
        print(json.dumps({"records": len(records), "bytes": sum(map(len, records.values()))}, indent=2))
        return
    with checkout_lock(root):
        api = drive_service(args.token, root)
        store = DriveStore(api, root)
        result = synchronize(root, store, dry_run=args.dry_run, resolutions=args.resolve)
        if store.folder:
            result["drive_folder"] = "https://drive.google.com/drive/folders/" + store.folder
        print(json.dumps(result, indent=2))
        if result["conflicts"]:
            raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except SyncError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None
    except Exception:
        print("Record sync failed; local records and existing Drive backups were preserved. "
              "Check connectivity and authorization, then retry.", file=sys.stderr)
        raise SystemExit(1) from None
