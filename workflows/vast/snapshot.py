"""Package the selected working tree without wheels, credentials, or old runs."""

import json
import os
from pathlib import Path
import tarfile

from .common import CHECKPOINT_SUFFIXES, hashes, relative_file, safe_name

SKIP_DIRS = {"__pycache__", ".git", "venv", ".pytest_cache", "runs", "results", "colab_runs"}
SKIP_SUFFIXES = CHECKPOINT_SUFFIXES | {".pyc", ".pyo", ".whl", ".pem", ".key", ".tmp", ".partial"}


def pinned_family_input(root, spec):
    family = (spec.get("dataset") or {}).get("pretrain_data") or {}
    if family.get("kind") != "family_sampled":
        return {}
    identity = family.get("catalog")
    if not identity:
        raise ValueError("Prepare and pin a complete family catalog before snapshotting")
    path = relative_file(Path(root), identity["archive_path"])
    actual = hashes(path)
    if any(actual[key] != identity[key] for key in ("size", "sha256")):
        raise ValueError("Family catalog package identity changed")
    return {"family_catalog": identity}


def source_files(root, spec_path, *, external_inputs=None, cached_inputs=False):
    root, spec_path = Path(root).resolve(), Path(spec_path).resolve()
    spec_path.relative_to(root)
    files = {root / "launch_run.py", root / "requirements.txt", root / "scripts/run_layout.py", spec_path}
    if (root / "requirements-fa4.txt").is_file():
        files.add(root / "requirements-fa4.txt")
    for name in ("src", "dataset_building_scripts", "assets", "workflows/vast", "workflows/runpod", "workflows/sfcompute"):
        directory = root / name
        for base, subdirs, names in os.walk(directory, followlinks=False):
            subdirs[:] = sorted(d for d in subdirs if d not in SKIP_DIRS and d != "tests" and not d.startswith(".")
                                and not (Path(base) / d).is_symlink())
            for name in names:
                path = Path(base) / name
                if not name.startswith(".") and path.suffix.lower() not in SKIP_SUFFIXES:
                    files.add(path)
    for name in ("ARC-1", "ARC-2", "ConceptARC"):
        files.update((root / "assets_tmp" / name).glob("*.json"))
    spec = json.loads(spec_path.read_text())
    safe_name(spec.get("run_name") or spec_path.stem)
    dataset = spec.get("dataset") or {}
    if dataset.get("prepared"):
        from dataset_building_scripts.intermediate_grids import validate_identity
        identity = dataset["prepared"]
        validate_identity(identity)
        archive = relative_file(root, identity["archive_path"])
        actual = hashes(archive)
        if any(actual[k] != identity[k] for k in ("size", "sha256")):
            raise ValueError("Prepared dataset differs from the selected spec")
        files.add(archive)
    if ((dataset.get("pretrain_data") or {}).get("kind") == "task_variations"
            or dataset.get("finetune_source") == "task_variations_full"):
        # The dataset builder uses this non-secret archive identity even on Vast.
        files.add(root / "scripts/colab_drive.json")
    compositions = (dataset.get("pretrain_data") or {}).get("compositions")
    family_data = dataset.get("pretrain_data") or {}
    if family_data.get("kind") == "family_sampled":
        identity = family_data.get("catalog")
        if not identity:
            raise ValueError("Prepare and pin a complete family catalog before snapshotting")
        if external_inputs:
            if external_inputs != {"family_catalog": identity}:
                raise ValueError("External family input differs from the pinned spec")
        else:
            pinned_family_input(root, spec)
            files.add(relative_file(root, identity["archive_path"]))
    elif external_inputs:
        raise ValueError("External inputs require a family-sampled spec")
    if compositions:
        archive = relative_file(root, compositions["archive_path"])
        identity = hashes(archive)
        if identity["size"] != compositions["size"] or identity["sha256"] != compositions["sha256"]:
            raise ValueError("Composition archive differs from the selected experiment spec")
        files.add(archive)
    split = ((spec.get("dataset") or {}).get("evaluation") or {}).get("split_manifest")
    if split:
        files.add(relative_file(root, split))
    if isinstance(spec.get("checkpoint"), str):
        raise ValueError("Vast continuation inputs must use a Drive checkpoint identity, so checkpoints never pass through the laptop")
    if spec.get("skip_data_build") and not spec.get("finetune_only") and not cached_inputs:
        raise ValueError("A fresh Vast workspace needs skip_data_build=false")
    for path in files:
        relative_file(root, path.relative_to(root))
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Source snapshot contains a missing or linked file: {path.relative_to(root)}")
        if path.name.casefold() in {"token.json", "drive-token.json", "credentials.json", "kaggle.json"} or path.name.casefold().startswith("client_secret"):
            raise ValueError("Credential-like file found in source snapshot")
    return sorted(files)


def create(root, spec_path, archive_path, *, external_inputs=None, cached_inputs=False):
    root = Path(root).resolve()
    files = source_files(root, spec_path, external_inputs=external_inputs, cached_inputs=cached_inputs)
    with tarfile.open(archive_path, "w:gz", compresslevel=6) as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    return {"files": len(files), **hashes(archive_path)}


def result_files(run_dir):
    run_dir = Path(run_dir)
    for path in sorted(run_dir.rglob("*")):
        relative = path.relative_to(run_dir)
        if (path.is_file() and not path.is_symlink() and path.suffix.lower() not in SKIP_SUFFIXES
                and not any(part.startswith(".") for part in relative.parts)):
            relative_file(run_dir, relative)
            yield path


def results_archive(run_dir, destination):
    """Copy live logs into an immutable archive; closed checkpoints stay excluded."""
    with tarfile.open(destination, "w:gz", compresslevel=3) as archive:
        for path in result_files(run_dir):
            archive.add(path, arcname=path.relative_to(run_dir).as_posix(), recursive=False)
    return hashes(destination)
