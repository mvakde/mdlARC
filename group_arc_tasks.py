#!/usr/bin/env python3
"""Group ARC-1/ARC-2 tasks into combined, challenge, and solution JSON files."""

from __future__ import annotations

import argparse
import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

TASK_ID_RE = re.compile(r"^[0-9a-f]{8}$", re.IGNORECASE)


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    folder: str
    prefix: str


DATASETS: Dict[str, DatasetConfig] = {
    "arc1": DatasetConfig(key="arc1", folder="ARC-1", prefix="arc1"),
    "arc2": DatasetConfig(key="arc2", folder="ARC-2", prefix="arc2"),
}

SPLIT_SUFFIX = {"training": "train", "evaluation": "eval"}


def _normalize_dataset_key(key: str) -> str:
    key = key.strip().lower().replace("-", "")
    if key == "arc1":
        return "arc1"
    if key == "arc2":
        return "arc2"
    if key in {"conceptarc", "concept"}:
        return "conceptarc"
    raise KeyError(f"Unknown dataset key: {key}")


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


def group_concept_tasks(
    assets_dir: Path | str = "assets", output_path: Path | str | None = None
) -> None:
    concept_root = Path(assets_dir) / "ConceptARC" / "raw"
    output_name = "concept_all.json"
    if output_path is None:
        out_path = concept_root / output_name
    else:
        out_path = Path(output_path)
        if out_path.suffix != ".json":
            out_path = out_path / output_name
    task_files = _iter_concept_task_files(concept_root, output_name)
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


def group_tasks(
    assets_dir: Path | str = "assets",
    datasets: Iterable[str] = ("arc1", "arc2", "conceptarc"),
    splits: Iterable[str] = ("training", "evaluation"),
    output_dir: Path | str | None = None,
) -> None:
    """Create grouped JSONs for ARC-1/2 and ConceptARC when requested."""
    assets_root = Path(assets_dir)
    normalized = [_normalize_dataset_key(key) for key in datasets]
    for dataset_key in normalized:
        if dataset_key == "conceptarc":
            group_concept_tasks(assets_dir=assets_root, output_path=output_dir)
            continue
        dataset = DATASETS[dataset_key]
        raw_dir = assets_root / dataset.folder / "raw"
        for split in splits:
            if split not in SPLIT_SUFFIX:
                raise KeyError(f"Unknown split: {split}")
            split_dir = raw_dir / split
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

            out_dir = split_dir if output_dir is None else Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            suffix = SPLIT_SUFFIX[split]
            both_path = out_dir / f"{dataset.prefix}_{suffix}_both.json"
            challenges_path = out_dir / f"{dataset.prefix}_{suffix}_challenges.json"
            solutions_path = out_dir / f"{dataset.prefix}_{suffix}_solutions.json"

            both_path.write_text(json.dumps(both, indent=2))
            challenges_path.write_text(json.dumps(challenges, indent=2))
            solutions_path.write_text(json.dumps(solutions, indent=2))

            print(
                f"Wrote {both_path.name}, {challenges_path.name}, {solutions_path.name}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Group ARC-1/ARC-2 tasks and optionally ConceptARC tasks."
    )
    parser.add_argument(
        "--assets-dir",
        default="assets",
        help="Root folder containing ARC-1/ARC-2 raw directories.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["arc1", "arc2", "conceptarc"],
        help="Datasets to process: arc1 arc2 conceptarc",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["training", "evaluation"],
        help="Splits to process: training evaluation",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory (ConceptARC writes concept_all.json there).",
    )
    args = parser.parse_args()
    group_tasks(
        assets_dir=args.assets_dir,
        datasets=args.datasets,
        splits=args.splits,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
