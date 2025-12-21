#!/usr/bin/env python3
"""Build combined ARC training/eval datasets with configurable clean setups."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class BuildConfig:
    name: str
    include_concept: bool
    keep_rel_paths: List[Path]
    required_rel_paths: List[Path]


CONCEPT_ALL = Path("ConceptARC/raw/concept_all.json")
ARC1_TRAIN_BOTH = Path("ARC-1/raw/training/arc1_train_both.json")
ARC1_EVAL_CHALLENGES = Path("ARC-1/raw/evaluation/arc1_eval_challenges.json")
ARC1_EVAL_SOLUTIONS = Path("ARC-1/raw/evaluation/arc1_eval_solutions.json")

CONFIGS: Dict[str, BuildConfig] = {
    "concept_arc1_clean": BuildConfig(
        name="concept_arc1_clean",
        include_concept=True,
        keep_rel_paths=[CONCEPT_ALL, ARC1_TRAIN_BOTH, ARC1_EVAL_CHALLENGES],
        required_rel_paths=[CONCEPT_ALL, ARC1_TRAIN_BOTH, ARC1_EVAL_CHALLENGES],
    ),
    "concept_arc1_train_eval": BuildConfig(
        name="concept_arc1_train_eval",
        include_concept=True,
        keep_rel_paths=[
            CONCEPT_ALL,
            ARC1_TRAIN_BOTH,
            ARC1_EVAL_CHALLENGES,
            ARC1_EVAL_SOLUTIONS,
        ],
        required_rel_paths=[CONCEPT_ALL, ARC1_TRAIN_BOTH, ARC1_EVAL_CHALLENGES],
    ),
    "arc1_clean": BuildConfig(
        name="arc1_clean",
        include_concept=False,
        keep_rel_paths=[ARC1_TRAIN_BOTH, ARC1_EVAL_CHALLENGES],
        required_rel_paths=[ARC1_TRAIN_BOTH, ARC1_EVAL_CHALLENGES],
    ),
}


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


def _merge_task_maps(*maps: Dict[str, dict]) -> Dict[str, dict]:
    merged: Dict[str, dict] = {}
    for task_map in maps:
        for task_id, task in task_map.items():
            if task_id in merged:
                raise ValueError(f"Duplicate task id: {task_id}")
            merged[task_id] = task
    return {task_id: merged[task_id] for task_id in sorted(merged.keys())}


def _clean_assets(assets_dir: Path, keep_paths: Iterable[Path]) -> None:
    keep_abs = {path.resolve() for path in keep_paths}
    for path in assets_dir.rglob("*"):
        if path.is_file() and path.resolve() not in keep_abs:
            path.unlink()


def _ensure_required(assets_dir: Path, required_paths: Iterable[Path]) -> None:
    missing = [path for path in required_paths if not (assets_dir / path).exists()]
    if missing:
        missing_list = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required files: {missing_list}")


def build_config(
    config: BuildConfig,
    assets_dir: Path | str = "assets",
    output_path: Path | str | None = None,
) -> Path:
    assets_root = Path(assets_dir)
    if output_path is None:
        output_path = assets_root / "challenges.json"
    else:
        output_path = Path(output_path)

    _ensure_required(assets_root, config.required_rel_paths)

    keep_paths = [assets_root / rel for rel in config.keep_rel_paths]
    _clean_assets(assets_root, keep_paths)

    arc1_train = _move_test_to_train(
        _load_json(assets_root / ARC1_TRAIN_BOTH)
    )
    arc1_eval = _strip_name_key(_load_json(assets_root / ARC1_EVAL_CHALLENGES))

    task_maps = [arc1_train, arc1_eval]
    if config.include_concept:
        concept_all = _move_test_to_train(
            _load_json(assets_root / CONCEPT_ALL)
        )
        task_maps.insert(0, concept_all)

    combined = _merge_task_maps(*task_maps)
    output_path.write_text(json.dumps(combined, indent=2))
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build combined training/eval datasets with clean configs."
    )
    parser.add_argument(
        "--config",
        default="concept_arc1_clean",
        choices=sorted(CONFIGS.keys()),
        help="Configuration to build.",
    )
    parser.add_argument(
        "--assets-dir",
        default="assets",
        help="Root assets directory.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for challenges.json (defaults to assets/challenges.json).",
    )
    args = parser.parse_args()

    output_path = build_config(
        CONFIGS[args.config], assets_dir=args.assets_dir, output_path=args.output
    )
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
