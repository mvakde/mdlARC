"""Compact artifacts and readable digit-grid evidence for composition v2."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil


def jsonable(value):
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if callable(getattr(value, "model_dump", None)):
        return jsonable(value.model_dump(mode="json"))
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(jsonable(value), separators=(",", ":"), default=str) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy_file(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".json" or source.name == "manifest":
        write(destination, read(source))
    else:
        shutil.copyfile(source, destination)


def copy_tree(source, destination):
    for path in sorted(Path(source).rglob("*")):
        if path.is_file() and not any(p in (".git", "__pycache__") for p in path.parts):
            copy_file(path, Path(destination) / path.relative_to(source))


def hashes(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(Path(root).rglob("*"))
            if p.is_file() and not any(part in (".git", "__pycache__", "scratch")
                                      for part in p.relative_to(root).parts)}


def puzzle_text(puzzle, label=""):
    sections = [f"Puzzle: {label}"] if label else []
    for split in ("train", "test"):
        for i, pair in enumerate(jsonable(puzzle)[split], 1):
            lines = [f"{split.title()} pair {i}"]
            for role in ("input", "output"):
                if role in pair:
                    lines += [role.title() + ":"]
                    lines += ["".join(str(c) for c in row) for row in pair[role]]
            sections.append("\n".join(lines))
    return "\n\n".join(sections) + "\n"


def save_puzzle(path, puzzle, label=""):
    write(path, puzzle)
    Path(path).with_suffix(".txt").write_text(puzzle_text(puzzle, label))
