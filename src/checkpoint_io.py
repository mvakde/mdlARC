"""Publish complete checkpoints and notify an optional external backup process."""

import json
import os
from pathlib import Path
import re
import tempfile
import uuid

# A rolling resume checkpoint (tiny.rolling.pt) is overwritten in place and stays on the
# training machine: backup uploaders and completeness checks skip it.
ROLLING_CHECKPOINT_MARKER = ".rolling"


def is_rolling_checkpoint(path: Path) -> bool:
    return Path(path).stem.endswith(ROLLING_CHECKPOINT_MARKER)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_checkpoint_keep_last(value) -> None:
    if value is not None and (type(value) is not int or value < 1):
        raise ValueError("checkpoint_keep_last must be null or a positive integer.")


def _checkpoint_order(path: Path, base: Path):
    if path.parent != base.parent:
        return None
    if path.name == base.name:
        return float("inf")  # The final checkpoint replaces the last periodic one.
    match = re.fullmatch(re.escape(base.stem) + r"\.epoch(\d+)" + re.escape(base.suffix), path.name)
    return int(match[1]) if match else None


def _retention_candidates(paths, base: Path, keep_last: int):
    ordered = [(order, path) for path in paths if (order := _checkpoint_order(path, base)) is not None]
    return [path for _, path in sorted(ordered)[:-keep_last]]


def checkpoint_upload_due(path, every_epochs=None):
    """Final checkpoints always upload; periodic checkpoints follow epoch multiples."""
    if every_epochs is None:
        return True
    if type(every_epochs) is not int or every_epochs <= 0:
        raise ValueError("Checkpoint upload interval must be a positive integer")
    match = re.search(r"\.epoch(\d+)\.[^.]+$", Path(path).name)
    return match is None or int(match[1]) % every_epochs == 0


def prune_unscheduled_checkpoints(run_dir: Path, event: dict, every_epochs=None):
    """Retain the latest local save and protect scheduled uploads until verified."""
    policy = event.get("retention")
    if every_epochs is None or policy is None:
        return
    keep_last = policy["keep_last"]
    validate_checkpoint_keep_last(keep_last)
    relative = Path(policy["base_path"])
    if (keep_last is None or relative.is_absolute() or ".." in relative.parts
            or not relative.parts or relative.parts[0] == "inputs" or relative.suffix != ".pt"):
        raise ValueError("Unsafe checkpoint retention policy")
    base = run_dir / relative
    base.resolve().relative_to(run_dir.resolve())
    paths = (path for path in base.parent.iterdir() if path.is_file() and not path.is_symlink())
    for path in _retention_candidates(paths, base, keep_last):
        if not checkpoint_upload_due(path, every_epochs):
            path.unlink(missing_ok=True)


def prune_backed_up_checkpoints(run_dir: Path, event: dict, ledger: dict) -> None:
    """Called only by the uploader; keep Drive identities for locally pruned files."""
    policy = event.get("retention")
    if policy is None:
        return
    keep_last = policy["keep_last"]
    validate_checkpoint_keep_last(keep_last)
    base = Path(policy["base_path"])
    published = Path(event["path"])
    if (keep_last is None or base.is_absolute() or ".." in base.parts
            or not base.parts or base.parts[0] == "inputs" or base.suffix != ".pt"
            or _checkpoint_order(published, base) is None):
        raise ValueError("Unsafe checkpoint retention policy")
    entries = {entry["path"]: entry for entry in ledger["files"]}
    if event["path"] not in entries:
        return  # Never prune for a replacement that has not been backed up.
    candidates = _retention_candidates(map(Path, entries), base, keep_last)
    for relative in candidates:
        entry = entries[relative.as_posix()]
        if entry.get("local_deleted"):
            continue
        path = run_dir / relative
        path.resolve().relative_to(run_dir.resolve())
        if path.is_symlink():
            raise ValueError("Refusing to prune a checkpoint symlink")
        path.unlink(missing_ok=True)
        entry["local_deleted"] = True


def save_checkpoint_atomically(save_path: Path, save, *, checkpoint_keep_last=None,
                               checkpoint_base_path: Path | None = None) -> None:
    """The backup queue only sees completed, immutable checkpoint files."""
    save_path = Path(save_path)
    validate_checkpoint_keep_last(checkpoint_keep_last)
    base = Path(checkpoint_base_path) if checkpoint_base_path is not None else save_path
    if checkpoint_keep_last is not None and _checkpoint_order(save_path, base) is None:
        raise ValueError("Checkpoint does not belong to its retention group")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    queue = os.environ.get("MDLARC_CHECKPOINT_QUEUE")
    if queue:
        run_dir = Path(os.environ["MDLARC_RUN_DIR"]).resolve()
        save_path.resolve().relative_to(run_dir)
        base.resolve().relative_to(run_dir)
        if save_path.is_symlink() or save_path.exists():
            raise FileExistsError(f"Backup checkpoints must have unique names: {save_path}")
    descriptor, name = tempfile.mkstemp(prefix=f".{save_path.name}.", suffix=".tmp", dir=save_path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        save(temporary)
        if queue:
            # Atomic no-clobber publication on the same filesystem.
            os.link(temporary, save_path)
            temporary.unlink()
            event = {
                "path": save_path.resolve().relative_to(run_dir).as_posix(),
                "size": save_path.stat().st_size,
            }
            if checkpoint_keep_last is not None:
                event["retention"] = {"base_path": base.resolve().relative_to(run_dir).as_posix(),
                                      "keep_last": checkpoint_keep_last}
            atomic_json(Path(queue) / f"{uuid.uuid4().hex}.json", event)
        else:
            temporary.replace(save_path)
            if checkpoint_keep_last is not None:
                paths = (path for path in base.parent.iterdir() if path.is_file() and not path.is_symlink())
                for previous in _retention_candidates(paths, base, checkpoint_keep_last):
                    previous.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def save_rolling_checkpoint(save_path: Path, save) -> None:
    """Atomically replace a local resume checkpoint; it is never queued for backup."""
    save_path = Path(save_path)
    if not is_rolling_checkpoint(save_path):
        raise ValueError(f"Rolling checkpoints must be named *{ROLLING_CHECKPOINT_MARKER}.pt: {save_path}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{save_path.name}.", suffix=".tmp", dir=save_path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        save(temporary)
        temporary.replace(save_path)
    finally:
        temporary.unlink(missing_ok=True)
