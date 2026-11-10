#!/usr/bin/env python3
import argparse
import json
import shutil
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS = SCRIPT_DIR.parent / "assets_tmp"

# Pin source revisions so upstream branch updates cannot change an experiment's data.
# ARC-1, ARC-2, and ConceptARC were checked against the existing grouped assets.
ARC1_COMMIT = "399030444e0ab0cc8b4e199870fb20b863846f34"
ARC2_COMMIT = "f3283f727488ad98fe575ea6a5ac981e4a188e49"
CONCEPTARC_COMMIT = "0e67da6af879e4bad3d7cd3c196e8d551b445725"
REARC_COMMIT = "e5b7f1d06362a76f9d3b8c25154ff1fafca897ce"

def _load_clean(path: Path) -> dict:
    """Loads a JSON and keeps ONLY 'train' and 'test' keys."""
    data = json.loads(path.read_text())
    return {
        "train": data["train"],
        "test": data["test"]
    }

def download_and_extract(url, extract_path, dest_folder, dest_subdir="raw"):
    print(f"Downloading {dest_folder}...")
    with urllib.request.urlopen(url) as resp:
        with zipfile.ZipFile(BytesIO(resp.read())) as zf:
            # Filter files to extract only the specific subdir
            for file in zf.namelist():
                if file.startswith(extract_path) and file.endswith(".json"):
                    zf.extract(file, ASSETS)
    
    # Move files from extracted nested dir to clean dest
    raw_path = ASSETS / extract_path
    final_path = ASSETS / dest_folder / dest_subdir
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists(): shutil.rmtree(final_path)
    shutil.move(str(raw_path), str(final_path))
    shutil.rmtree(ASSETS / extract_path.split("/")[0]) 

# The original datasets have each task in a separate json. We collect it into a single json like the official Kaggle datasets
def group_arc(name, prefix):
    print(f"Grouping {name}...")
    base = ASSETS / name
    for split, folder in [("train", "training"), ("eval", "evaluation")]:
        data = {"both": {}, "challenges": {}, "solutions": {}}
        
        for p in sorted((base / "raw" / folder).glob("*.json")):
            # _load_clean strips all metadata immediately
            task = _load_clean(p)
            
            data["both"][p.stem] = task
            # Create challenge by copying task and overwriting 'test' to remove outputs
            data["challenges"][p.stem] = {
                "train": task["train"], 
                "test": [{"input": t["input"]} for t in task["test"]]
            }
            data["solutions"][p.stem] = [t["output"] for t in task["test"]]

        for k, v in data.items():
            (base / f"{prefix}_{split}_{k}.json").write_text(json.dumps(v))

def group_concept():
    print("Grouping ConceptARC...")
    root = ASSETS / "ConceptARC"
    # Load and clean every file found
    tasks = {
        p.stem: _load_clean(p) 
        for p in sorted((root / "raw").rglob("*.json"))
    }
    (root / "concept_all.json").write_text(json.dumps(tasks))

def filter_overlaps():
    print("Filtering overlaps between ARC-1 and ARC-2...")
    
    def load_superset(name):
        data = {}
        # Loads both train_both.json and eval_both.json
        for p in (ASSETS / name).glob("*_both.json"):
            data.update(json.loads(p.read_text()))
        return data

    arc1 = load_superset("ARC-1")
    arc2 = load_superset("ARC-2")
    
    # Identify common keys (intersection)
    duplicates = set(arc1.keys()) & set(arc2.keys())

    # Save filtered versions (removing duplicates from both)
    (ASSETS / "ARC-1" / "arc1_filtered.json").write_text(json.dumps({k:v for k,v in arc1.items() if k not in duplicates}))
    (ASSETS / "ARC-2" / "arc2_filtered.json").write_text(json.dumps({k:v for k,v in arc2.items() if k not in duplicates}))
    
    print(f"  Removed {len(duplicates)} overlapping tasks from both datasets.")

def _download_and_group_conceptarc():
    download_and_extract(
        f"https://github.com/victorvikram/ConceptARC/archive/{CONCEPTARC_COMMIT}.zip",
        f"ConceptARC-{CONCEPTARC_COMMIT}/corpus",
        "ConceptARC",
    )
    group_concept()


def main():
    parser = argparse.ArgumentParser(description="Download and group ARC datasets.")
    parser.add_argument(
        "--conceptarc-only",
        action="store_true",
        help="Download and group only ConceptARC.",
    )
    args = parser.parse_args()

    if args.conceptarc_only:
        _download_and_group_conceptarc()
        return

    # 1. ARC-1
    download_and_extract(
        f"https://github.com/fchollet/ARC-AGI/archive/{ARC1_COMMIT}.zip",
        f"ARC-AGI-{ARC1_COMMIT}/data",
        "ARC-1",
    )
    group_arc("ARC-1", "arc1")

    # 2. ARC-2
    download_and_extract(
        f"https://github.com/arcprize/ARC-AGI-2/archive/{ARC2_COMMIT}.zip",
        f"ARC-AGI-2-{ARC2_COMMIT}/data",
        "ARC-2",
    )
    group_arc("ARC-2", "arc2")

    filter_overlaps() 

    # 3. ConceptARC
    _download_and_group_conceptarc()

    # 4. RE-ARC
    download_and_extract(
        f"https://raw.githubusercontent.com/michaelhodel/re-arc/{REARC_COMMIT}/re_arc.zip",
        "re_arc/tasks",
        "RE-ARC",
        dest_subdir="tasks",
    )

if __name__ == "__main__":
    main()
