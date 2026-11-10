"""Read fixed parent-task splits without generating a new split at run time."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def content_sha256(value: dict) -> str:
    """Hash parsed task content independently of whitespace and key order."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_parent_split(root: Path, relative_path: str) -> tuple[dict, dict]:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("split_manifest must be a repository-relative file path")
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts or "\\" in relative_path:
        raise ValueError("split_manifest must be a repository-relative file path")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()) or (root / path).is_symlink():
        raise ValueError("split_manifest must be an ordinary file inside the repository")
    contents = resolved.read_bytes()
    manifest = json.loads(contents)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Parent split requires schema_version=1")
    if manifest.get("source") not in {"arc2_train", "arc2_eval", "conceptarc", "arc1_filtered"}:
        raise ValueError("Parent split has an unsupported source")
    if not isinstance(manifest.get("name"), str) or not NAME.fullmatch(manifest["name"]):
        raise ValueError("Parent split requires a safe name")
    digest = manifest.get("source_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Parent split requires source_sha256")
    groups = manifest.get("eval_groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("Parent split requires nonempty eval_groups")
    if any(not isinstance(name, str) or not NAME.fullmatch(name) or name in {"all", "train", ".", ".."}
           for name in groups):
        raise ValueError("Parent split has an invalid evaluation group name")
    seen: set[str] = set()
    for name, ids in {"train": manifest.get("train"), **groups}.items():
        if (not isinstance(ids, list) or not ids
                or any(not isinstance(task_id, str) or not NAME.fullmatch(task_id)
                       or task_id in {".", ".."} for task_id in ids)):
            raise ValueError(f"Parent split group {name} requires a nonempty list of task IDs")
        if len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError("Parent split task IDs must be unique and disjoint across groups")
        seen.update(ids)
    identity = {
        "manifest": path.as_posix(),
        "sha256": hashlib.sha256(contents).hexdigest(),
        "source": manifest["source"],
        "source_sha256": digest,
    }
    return manifest, identity


def validate_split_tasks(manifest: dict, tasks: dict) -> None:
    if content_sha256(tasks) != manifest["source_sha256"]:
        raise ValueError("Parent split source content differs from source_sha256")
    roles = {task_id: "train" for task_id in manifest["train"]}
    for group, ids in manifest["eval_groups"].items():
        roles.update({task_id: group for task_id in ids})
    if set(roles) != set(tasks):
        raise ValueError("Parent split must cover every source task exactly once")
    # A shared solved example also needs to stay on one side of pretraining.
    seen_pairs: dict[str, tuple[str, bool]] = {}
    for task_id, task in tasks.items():
        is_train = roles[task_id] == "train"
        for pair in list(task.get("train", [])) + list(task.get("test", [])):
            digest = content_sha256(pair)
            previous = seen_pairs.get(digest)
            if previous is not None and previous[1] != is_train:
                raise ValueError(f"Solved pair crosses the parent split: {previous[0]} and {task_id}")
            seen_pairs[digest] = (task_id, is_train)
