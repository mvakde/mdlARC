#!/usr/bin/env python3
"""Resolve one sweep trial from a reusable template and a small JSON record."""

import argparse
import copy
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRIAL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class MaterializedSpec:
    trial_id: str
    template_path: Path
    spec_bytes: bytes
    trial_bytes: bytes
    template_bytes: bytes


def _inside(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Path must be inside the repository: {path}") from exc
    return resolved


def _merge(base: dict, overrides: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def materialize(root: Path, trial_path: Path) -> MaterializedSpec:
    """Return exact serialized bytes; reject records that contain multiple trials."""
    trial_path = _inside(root, trial_path)
    trial_bytes = trial_path.read_bytes()
    record = json.loads(trial_bytes)
    if not isinstance(record, dict) or set(record) != {"trial_id", "template", "overrides"}:
        raise ValueError("Trial record needs trial_id, template, and overrides only")
    trial_id = record["trial_id"]
    if not isinstance(trial_id, str) or not TRIAL_ID.fullmatch(trial_id) or trial_id in {".", ".."}:
        raise ValueError("trial_id must be one safe name")
    if not isinstance(record["template"], str):
        raise ValueError("template must be a repository-relative specs/ path")
    template_path = _inside(root, root / record["template"])
    if not template_path.is_relative_to((root / "specs").resolve()):
        raise ValueError("template must be under specs/")
    template_bytes = template_path.read_bytes()
    template = json.loads(template_bytes)
    overrides = record["overrides"]
    if not isinstance(template, dict) or not isinstance(overrides, dict):
        raise ValueError("Template and overrides must be JSON objects")
    if template.get("finetunes") not in ({}, None) or template.get("evals") not in ({}, None):
        raise ValueError("Sweep template must not contain trial-specific finetunes or evals")
    finetunes = overrides.get("finetunes")
    evals = overrides.get("evals", {})
    if not isinstance(finetunes, dict) or set(finetunes) != {trial_id}:
        raise ValueError("overrides.finetunes must contain exactly trial_id")
    if not isinstance(finetunes[trial_id], dict) or not isinstance(evals, dict) or set(evals) not in (set(), {trial_id}):
        raise ValueError("Trial finetune must be an object; evals may contain only trial_id")
    if evals and not isinstance(evals[trial_id], dict):
        raise ValueError("Trial evaluation must be an object")
    spec = _merge(template, overrides)
    return MaterializedSpec(trial_id, template_path,
                            (json.dumps(spec, indent=2, ensure_ascii=False) + "\n").encode(),
                            trial_bytes, template_bytes)


def stage(root: Path, trial_path: Path, output_path: Path | None = None, *, replace: bool = False) -> Path:
    resolved = materialize(root, trial_path)
    output = _inside(root, output_path or root / "specs" / "temp specs" / f"{resolved.trial_id}.json")
    if not output.is_relative_to((root / "specs" / "temp specs").resolve()):
        raise ValueError("Generated specs must be staged under specs/temp specs/")
    if output.is_file() and output.read_bytes() != resolved.spec_bytes and not replace:
        raise FileExistsError(f"Staged spec differs: {output}; use --replace before launching")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=output.parent, prefix=f".{output.name}-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(resolved.spec_bytes)
    temporary.replace(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", required=True, type=Path, help="Compact JSON trial record")
    parser.add_argument("--output", type=Path, help="Output under specs/temp specs/ (default: trial_id.json)")
    parser.add_argument("--replace", action="store_true", help="Replace a changed staged spec before launching")
    args = parser.parse_args()
    print(stage(ROOT, args.trial, args.output, replace=args.replace).relative_to(ROOT))


if __name__ == "__main__":
    main()
