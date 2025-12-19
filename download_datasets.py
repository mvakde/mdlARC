#!/usr/bin/env python3
"""Download official ARC-1, ARC-2, and ConceptARC datasets from GitHub."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    repo: str
    branch: str
    subdir: str
    dest_rel: Path


DATASETS: Dict[str, DatasetSpec] = {
    "arc1": DatasetSpec(
        key="arc1",
        name="ARC-1",
        repo="fchollet/ARC-AGI",
        branch="master",
        subdir="data",
        dest_rel=Path("ARC-1/raw"),
    ),
    "arc2": DatasetSpec(
        key="arc2",
        name="ARC-2",
        repo="arcprize/ARC-AGI-2",
        branch="main",
        subdir="data",
        dest_rel=Path("ARC-2/raw"),
    ),
    "conceptarc": DatasetSpec(
        key="conceptarc",
        name="ConceptARC",
        repo="victorvikram/ConceptARC",
        branch="main",
        subdir="corpus",
        dest_rel=Path("ConceptARC/raw"),
    ),
}


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, dest.open("wb") as f:
        shutil.copyfileobj(resp, f)


def _find_single_dir(root: Path) -> Path:
    dirs = [p for p in root.iterdir() if p.is_dir()]
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


def _resolve_datasets(requested: Iterable[str]) -> List[DatasetSpec]:
    requested = list(requested)
    if not requested or "all" in requested:
        return list(DATASETS.values())
    specs: List[DatasetSpec] = []
    for key in requested:
        spec = DATASETS.get(key)
        if spec is None:
            raise KeyError(f"Unknown dataset key: {key}")
        specs.append(spec)
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download official ARC datasets into a local folder."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="Datasets to download: arc1, arc2, conceptarc, or all.",
    )
    parser.add_argument(
        "--output-dir",
        default="assets",
        help="Root folder to place dataset folders.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing dataset folders.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    specs = _resolve_datasets(args.datasets)

    for spec in specs:
        dest = output_dir / spec.dest_rel
        if dest.exists() and not args.force:
            print(f"Skipping {spec.name}; {dest} already exists.")
            continue
        url = f"https://github.com/{spec.repo}/archive/refs/heads/{spec.branch}.zip"
        print(f"Downloading {spec.name} from {url}")
        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_path = Path(tmp_dir) / f"{spec.key}.zip"
            _download(url, archive_path)
            _extract_subdir(archive_path, spec.subdir, dest, args.force)
        print(f"Saved {spec.name} to {dest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
