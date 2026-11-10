"""Small, dependency-free helpers shared by the local and remote workflow."""

from contextlib import contextmanager
from datetime import UTC, datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
import uuid
from urllib.parse import quote

CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}
# Training overwrites tiny.rolling.pt in place for local resumes; it is never backed up.
ROLLING_CHECKPOINT_MARKER = ".rolling"
ACCOUNT = "your-drive-account@example.com"
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"
DEFAULT_TOKEN = Path.home() / ".config/mdlarc-vast/drive-token.json"
ROOT_FOLDER = "DRIVE_ID_05_PLACEHOLDER"
# Explicit user instruction: this scarce rental must never be shut down.
PROTECTED_B300_INSTANCES = frozenset({57431449})


def now():
    return datetime.now(UTC).isoformat()


def keep_b300_running(plan, instance=None):
    """Standing user rule: scarce B300 capacity must survive automatic cleanup."""
    names = [plan.get("gpu_name"), (plan.get("adopted_instance") or {}).get("gpu_name"),
             (instance or {}).get("gpu_name")]
    if not any(names):
        names = (plan.get("policy") or {}).get("gpu_names", [])
    identity = plan.get("instance_id", (instance or {}).get("id"))
    return str(identity) in {str(value) for value in PROTECTED_B300_INSTANCES} or plan.get("preserve_instance") is True or any(
        re.search(r"\bB300\b", str(name), re.IGNORECASE) for name in names if name)


def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", value):
        raise ValueError("Run/session names must be safe single directory names (maximum 160 characters)")
    return value


def branch_folder(value):
    if not value or value in {".", ".."} or any(ord(c) < 32 for c in value):
        raise ValueError("Invalid run branch")
    return quote(value, safe="-_.")


def atomic_json(path, value, *, private=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600 if private else 0o644)
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(value, out, indent=2, allow_nan=False)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def hashes(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with Path(path).open("rb") as src:
        for data in iter(lambda: src.read(8 * 1024 * 1024), b""):
            md5.update(data)
            sha.update(data)
    return {"size": Path(path).stat().st_size, "md5": md5.hexdigest(), "sha256": sha.hexdigest()}


def relative_file(root, value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "\\" in str(value) or not path.parts:
        raise ValueError("Unsafe relative file path")
    root = Path(root).resolve()
    target = root / path
    target.resolve().relative_to(root)
    if target.is_symlink() or any(p.is_symlink() for p in target.parents if p != root):
        raise ValueError("Linked paths are not permitted")
    return target


def extract_archive(archive_path, destination, *, exclude_checkpoints=False):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    seen = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or member.name in seen:
                raise ValueError("Archive contains a link, directory, or duplicate member")
            seen.add(member.name)
            target = relative_file(destination, member.name)
            if exclude_checkpoints and target.suffix.lower() in CHECKPOINT_SUFFIXES:
                raise ValueError("Result archive unexpectedly contains a checkpoint")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("Unreadable archive member")
            with source, target.open("xb") as out:
                import shutil
                shutil.copyfileobj(source, out)


@contextmanager
def lock(path, *, blocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            raise RuntimeError("Another worker/watcher already owns this run") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
