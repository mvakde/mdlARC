#!/usr/bin/env python3
"""Download, group, and build ARC datasets with granular split control."""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, TypeVar

TASK_ID_RE = re.compile(r"^[0-9a-f]{8}$", re.IGNORECASE)
RAW_SPLIT_DIR = {"train": "training", "eval": "evaluation"}


@dataclass(frozen=True)
class DatasetInfo:
    key: str
    name: str
    repo: str
    branch: str
    subdir: str
    folder: str
    prefix: str | None


DATASETS: Dict[str, DatasetInfo] = {
    "arc1": DatasetInfo(
        key="arc1",
        name="ARC-1",
        repo="fchollet/ARC-AGI",
        branch="master",
        subdir="data",
        folder="ARC-1",
        prefix="arc1",
    ),
    "arc2": DatasetInfo(
        key="arc2",
        name="ARC-2",
        repo="arcprize/ARC-AGI-2",
        branch="main",
        subdir="data",
        folder="ARC-2",
        prefix="arc2",
    ),
    "conceptarc": DatasetInfo(
        key="conceptarc",
        name="ConceptARC",
        repo="victorvikram/ConceptARC",
        branch="main",
        subdir="corpus",
        folder="ConceptARC",
        prefix=None,
    ),
}


def _normalize_dataset_key(key: str) -> str:
    cleaned = key.strip().lower().replace("-", "")
    if cleaned in {"arc1", "arc2"}:
        return cleaned
    if cleaned in {"concept", "conceptarc"}:
        return "conceptarc"
    raise KeyError(f"Unknown dataset key: {key}")


def _normalize_dataset_keys(keys: Iterable[str]) -> List[str]:
    expanded: List[str] = []
    for key in keys:
        cleaned = key.strip().lower()
        if cleaned == "all":
            expanded.extend(DATASETS.keys())
        else:
            expanded.append(_normalize_dataset_key(key))
    normalized: List[str] = []
    for key in expanded:
        if key not in normalized:
            normalized.append(key)
    return normalized


def _normalize_splits(splits: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    for split in splits:
        cleaned = split.strip().lower()
        if cleaned in {"train", "training"}:
            name = "train"
        elif cleaned in {"eval", "evaluation"}:
            name = "eval"
        else:
            raise KeyError(f"Unknown split: {split}")
        if name not in normalized:
            normalized.append(name)
    return normalized


def _normalize_steps(steps: Iterable[str]) -> List[str]:
    requested = {step.strip().lower() for step in steps}
    if "all" in requested:
        requested = {"download", "group", "build"}
    valid = {"download", "group", "build"}
    unknown = requested - valid
    if unknown:
        unknown_list = ", ".join(sorted(unknown))
        raise KeyError(f"Unknown step(s): {unknown_list}")
    return [step for step in ["download", "group", "build"] if step in requested]


def _validate_request(
    dataset_keys: List[str], splits: List[str], with_solutions: bool
) -> None:
    if not dataset_keys:
        raise ValueError("At least one dataset must be selected.")
    if not splits:
        raise ValueError("At least one split must be selected.")
    arc_keys = [key for key in dataset_keys if key in {"arc1", "arc2"}]
    if len(arc_keys) > 1:
        raise ValueError("Combining ARC-1 and ARC-2 is not supported yet.")
    if "conceptarc" in dataset_keys and "train" not in splits:
        raise ValueError("ConceptARC requires the train split.")
    if "eval" in splits and not arc_keys:
        raise ValueError("Eval split requires ARC-1 or ARC-2 in --datasets.")
    if with_solutions and "eval" not in splits:
        raise ValueError("--with-solutions requires the eval split.")
    if with_solutions and not arc_keys:
        raise ValueError("--with-solutions requires ARC-1 or ARC-2 in --datasets.")


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, dest.open("wb") as f:
        shutil.copyfileobj(resp, f)


def _find_single_dir(root: Path) -> Path:
    dirs = [path for path in root.iterdir() if path.is_dir()]
    if len(dirs) != 1:
        raise RuntimeError(f"Expected 1 top-level directory, found {len(dirs)}")
    return dirs[0]


def _extract_subdir(archive_path: Path, subdir: str, dest: Path, force: bool) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(tmp_root)
        repo_root = _find_single_dir(tmp_root)
        src = repo_root / subdir
        if not src.exists():
            raise FileNotFoundError(f"Missing expected path in archive: {subdir}")
        if dest.exists():
            if not force:
                print(f"Skipping {dest} (already exists).")
                return
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dest)


def download_datasets(assets_dir: Path, dataset_keys: Iterable[str], force: bool) -> None:
    for key in dataset_keys:
        dataset = DATASETS[key]
        dest = assets_dir / dataset.folder / "raw"
        if dest.exists() and not force:
            print(f"Skipping {dataset.name}; {dest} already exists.")
            continue
        url = f"https://github.com/{dataset.repo}/archive/refs/heads/{dataset.branch}.zip"
        print(f"Downloading {dataset.name} from {url}")
        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_path = Path(tmp_dir) / f"{dataset.key}.zip"
            _download(url, archive_path)
            _extract_subdir(archive_path, dataset.subdir, dest, force)
        print(f"Saved {dataset.name} to {dest}")


def _iter_task_files(split_dir: Path) -> List[Path]:
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")
    task_files = [
        path
        for path in split_dir.iterdir()
        if path.is_file() and path.suffix == ".json" and TASK_ID_RE.fullmatch(path.stem)
    ]
    return sorted(task_files)


def _iter_concept_task_files(root: Path, output_name: str) -> List[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Missing ConceptARC root directory: {root}")
    task_files = [
        path
        for path in root.rglob("*.json")
        if path.is_file() and path.name != output_name
    ]
    return sorted(task_files)


def _load_task(path: Path) -> dict:
    data = json.loads(path.read_text())
    if "train" not in data or "test" not in data:
        raise ValueError(f"Task {path.name} is missing train/test keys.")
    return data


def _split_task(task: dict) -> Tuple[dict, List[List[List[int]]]]:
    challenge = copy.deepcopy(task)
    solutions: List[List[List[int]]] = []
    for pair in challenge.get("test", []):
        if "output" not in pair:
            raise ValueError("Test pair missing output key.")
        solutions.append(pair.pop("output"))
    return challenge, solutions


def group_concept_tasks(assets_dir: Path) -> Path:
    concept_root = assets_dir / "ConceptARC"
    raw_root = concept_root / "raw"
    output_name = "concept_all.json"
    out_path = concept_root / output_name
    task_files = _iter_concept_task_files(raw_root, output_name)
    combined: Dict[str, dict] = {}
    for task_path in task_files:
        task_id = task_path.stem
        if task_id in combined:
            raise ValueError(f"Duplicate ConceptARC task id: {task_id}")
        combined[task_id] = json.loads(task_path.read_text())
    combined = {task_id: combined[task_id] for task_id in sorted(combined.keys())}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(combined, indent=2))
    print(f"Wrote {out_path.name}")
    return out_path


def group_arc_tasks(
    assets_dir: Path, dataset: DatasetInfo, splits: Iterable[str]
) -> List[Path]:
    raw_root = assets_dir / dataset.folder / "raw"
    dataset_root = raw_root.parent
    output_paths: List[Path] = []
    for split in splits:
        raw_split = RAW_SPLIT_DIR[split]
        split_dir = raw_root / raw_split
        task_files = _iter_task_files(split_dir)
        both: Dict[str, dict] = {}
        challenges: Dict[str, dict] = {}
        solutions: Dict[str, List[List[List[int]]]] = {}
        for task_path in task_files:
            task_id = task_path.stem
            task = _load_task(task_path)
            both[task_id] = task
            challenge, task_solutions = _split_task(task)
            challenges[task_id] = challenge
            solutions[task_id] = task_solutions

        ordered_ids = sorted(both.keys())
        both = {task_id: both[task_id] for task_id in ordered_ids}
        challenges = {task_id: challenges[task_id] for task_id in ordered_ids}
        solutions = {task_id: solutions[task_id] for task_id in ordered_ids}

        dataset_root.mkdir(parents=True, exist_ok=True)
        suffix = split
        both_path = dataset_root / f"{dataset.prefix}_{suffix}_both.json"
        challenges_path = dataset_root / f"{dataset.prefix}_{suffix}_challenges.json"
        solutions_path = dataset_root / f"{dataset.prefix}_{suffix}_solutions.json"

        both_path.write_text(json.dumps(both, indent=2))
        challenges_path.write_text(json.dumps(challenges, indent=2))
        solutions_path.write_text(json.dumps(solutions, indent=2))
        output_paths.extend([both_path, challenges_path, solutions_path])

        print(
            f"Wrote {both_path.name}, {challenges_path.name}, {solutions_path.name}"
        )
    return output_paths


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _move_test_to_train(task_map: Dict[str, dict]) -> Dict[str, dict]:
    moved: Dict[str, dict] = {}
    for task_id, task in task_map.items():
        task_copy = dict(task)
        task_copy.pop("name", None)
        train_pairs = list(task_copy.get("train", []))
        test_pairs = list(task_copy.get("test", []))
        task_copy["train"] = train_pairs + test_pairs
        task_copy.pop("test", None)
        moved[task_id] = task_copy
    return moved


def _strip_name_key(task_map: Dict[str, dict]) -> Dict[str, dict]:
    cleaned: Dict[str, dict] = {}
    for task_id, task in task_map.items():
        task_copy = dict(task)
        task_copy.pop("name", None)
        cleaned[task_id] = task_copy
    return cleaned


T = TypeVar("T")


def _merge_maps(*maps: Dict[str, T]) -> Dict[str, T]:
    merged: Dict[str, T] = {}
    for task_map in maps:
        for task_id, task in task_map.items():
            if task_id in merged:
                raise ValueError(f"Duplicate task id: {task_id}")
            merged[task_id] = task
    return {task_id: merged[task_id] for task_id in sorted(merged.keys())}


def _ensure_required(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_list = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required files: {missing_list}")


def _dataset_root(assets_dir: Path, dataset: DatasetInfo) -> Path:
    return assets_dir / dataset.folder


def _collect_grouped_inputs(
    assets_dir: Path, datasets: Iterable[DatasetInfo], splits: List[str]
) -> List[Path]:
    paths: List[Path] = []
    for dataset in datasets:
        dataset_root = _dataset_root(assets_dir, dataset)
        if dataset.key == "conceptarc":
            if "train" in splits:
                paths.append(dataset_root / "concept_all.json")
            continue
        if "train" in splits:
            paths.append(dataset_root / f"{dataset.prefix}_train_both.json")
        if "eval" in splits:
            paths.append(dataset_root / f"{dataset.prefix}_eval_challenges.json")
    return paths


def _solutions_source_path(assets_dir: Path, dataset: DatasetInfo) -> Path:
    dataset_root = _dataset_root(assets_dir, dataset)
    return dataset_root / f"{dataset.prefix}_eval_solutions.json"


def _collect_solution_sources(
    assets_dir: Path,
    datasets: Iterable[DatasetInfo],
    splits: List[str],
    with_solutions: bool,
) -> List[Path]:
    if not with_solutions or "eval" not in splits:
        return []
    sources: List[Path] = []
    for dataset in datasets:
        if dataset.key == "conceptarc":
            continue
        sources.append(_solutions_source_path(assets_dir, dataset))
    return sources


def _resolve_output_path(assets_dir: Path, output_path: str | None) -> Path:
    if output_path is None:
        return assets_dir / "challenges.json"
    return Path(output_path)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _clean_assets(assets_dir: Path, keep_paths: Iterable[Path]) -> None:
    keep_abs = {path.resolve() for path in keep_paths}
    for path in assets_dir.rglob("*"):
        if path.is_file() and path.resolve() not in keep_abs:
            path.unlink()


def _prune_empty_dirs(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def build_dataset(
    assets_dir: Path,
    datasets: List[DatasetInfo],
    splits: List[str],
    output_path: str | None,
    with_solutions: bool,
    cleanup: str,
    solutions_only: bool,
) -> Tuple[Path | None, Path | None]:
    assets_root = assets_dir
    output_path_resolved = _resolve_output_path(assets_root, output_path)

    grouped_inputs = []
    if not solutions_only:
        grouped_inputs = _collect_grouped_inputs(assets_root, datasets, splits)
    solutions_sources = _collect_solution_sources(
        assets_root, datasets, splits, with_solutions
    )
    required_paths = grouped_inputs + solutions_sources
    _ensure_required(required_paths)

    task_maps: List[Dict[str, dict]] = []
    if not solutions_only:
        for dataset in datasets:
            dataset_root = _dataset_root(assets_root, dataset)
            if dataset.key == "conceptarc":
                if "train" in splits:
                    concept_path = dataset_root / "concept_all.json"
                    task_maps.append(_move_test_to_train(_load_json(concept_path)))
                continue
            if "train" in splits:
                train_path = dataset_root / f"{dataset.prefix}_train_both.json"
                task_maps.append(_move_test_to_train(_load_json(train_path)))
            if "eval" in splits:
                eval_path = dataset_root / f"{dataset.prefix}_eval_challenges.json"
                task_maps.append(_strip_name_key(_load_json(eval_path)))

    output_path_written: Path | None = None
    if task_maps:
        combined = _merge_maps(*task_maps)
        output_path_resolved.write_text(json.dumps(combined, indent=2))
        output_path_written = output_path_resolved

    solutions_output: Path | None = None
    if solutions_sources:
        solutions_output = output_path_resolved.with_name("solutions.json")
        solution_maps = [_load_json(path) for path in solutions_sources]
        merged_solutions = _merge_maps(*solution_maps)
        solutions_output.write_text(json.dumps(merged_solutions, indent=2))

    if cleanup != "raw":
        keep_paths: List[Path] = []
        if cleanup == "grouped":
            keep_paths.extend(grouped_inputs)
            keep_paths.extend(solutions_sources)
        if output_path_written is not None and _is_within(output_path_written, assets_root):
            keep_paths.append(output_path_written)
        if solutions_output is not None and _is_within(solutions_output, assets_root):
            keep_paths.append(solutions_output)
        _clean_assets(assets_root, keep_paths)
        _prune_empty_dirs(assets_root)

    return output_path_written, solutions_output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download, group, and build ARC datasets with granular control."
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["download", "group", "build"],
        help="Pipeline steps to run: download group build (or all).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to process: arc1 arc2 conceptarc (or all).",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Single dataset to process (legacy).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "eval"],
        help="Splits to include: train eval.",
    )
    parser.add_argument(
        "--with-solutions",
        action="store_true",
        help="Write solutions.json from eval solutions (requires eval split).",
    )
    parser.add_argument(
        "--solutions-only",
        action="store_true",
        help="Only write solutions.json for the selected datasets and splits.",
    )
    parser.add_argument(
        "--cleanup",
        default="grouped",
        choices=["raw", "grouped", "none"],
        help=(
            "Cleanup mode: raw keeps everything, grouped removes unused files, "
            "none keeps only outputs."
        ),
    )
    parser.add_argument(
        "--assets-dir",
        default="assets",
        help="Root assets directory for raw and grouped data.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for challenges.json (defaults to assets/challenges.json).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing dataset folders when downloading.",
    )
    args = parser.parse_args()

    if args.datasets is not None and args.dataset is not None:
        raise ValueError("Use --datasets or --dataset, not both.")
    dataset_args = args.datasets
    if dataset_args is None:
        dataset_args = [args.dataset] if args.dataset is not None else ["arc1"]

    dataset_keys = _normalize_dataset_keys(dataset_args)
    datasets = [DATASETS[key] for key in dataset_keys]
    splits = _normalize_splits(args.splits)
    steps = _normalize_steps(args.steps)
    with_solutions = args.with_solutions or args.solutions_only
    _validate_request(dataset_keys, splits, with_solutions)

    assets_root = Path(args.assets_dir)

    if "download" in steps:
        download_datasets(assets_root, dataset_keys, args.force)
    if "group" in steps:
        for dataset in datasets:
            if dataset.key == "conceptarc":
                group_concept_tasks(assets_root)
            else:
                group_arc_tasks(assets_root, dataset, splits)
    if "build" in steps:
        output_path, solutions_path = build_dataset(
            assets_root,
            datasets,
            splits,
            output_path=args.output,
            with_solutions=with_solutions,
            cleanup=args.cleanup,
            solutions_only=args.solutions_only,
        )
        if output_path is not None:
            print(f"Wrote {output_path}")
        if solutions_path is not None:
            print(f"Wrote {solutions_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
