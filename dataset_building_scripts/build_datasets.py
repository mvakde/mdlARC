#!/usr/bin/env python3
"""
Build final challenges.json and solutions.json from grouped assets.
"""

import argparse
import json
from pathlib import Path

ASSETS_TMP = Path("../assets_tmp")

def _load(filename):
    path = ASSETS_TMP / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")
    return json.loads(path.read_text())

def _flatten_to_train(tasks: dict) -> dict:
    """
    For training data, we want to learn from ALL examples.
    This moves 'test' pairs into 'train' and removes the 'test' key.
    """
    flat_tasks = {}
    for tid, data in tasks.items():
        # Copy to avoid mutating original if reused
        task = data.copy()
        # Merge test pairs into train
        task["train"] = task.get("train", []) + task.get("test", [])
        if "test" in task:
            del task["test"]
        flat_tasks[tid] = task
    return flat_tasks

def main():
    parser = argparse.ArgumentParser(description="Build ARC datasets.")
    parser.add_argument("dataset", choices=["arc1", "arc2"], help="Main dataset to build (ignored with --submission).")
    parser.add_argument("--submission", action="store_true", help="Build combined ARC1+ARC2+ConceptARC with private tasks.")
    parser.add_argument("--private-json", help="Path to private challenges JSON (unflattened). Required with --submission.")
    parser.add_argument("--add-conceptarc", action="store_true", help="Merge ConceptARC into the training set.")
    parser.add_argument("--output-dir", default="../assets", help="Directory to save output files.")
    parser.add_argument("--with-filtered", action="store_true", help="Merge filtered tasks from the other ARC dataset.")
    args = parser.parse_args()

    if args.submission and not args.private_json:
        raise SystemExit("--private-json is required when using --submission.")

    if args.submission:
        print("Building submission dataset (ARC-1 + ARC-2 + ConceptARC + private)")

        arc1 = {}
        arc1.update(_load("ARC-1/arc1_train_both.json"))
        arc1.update(_load("ARC-1/arc1_eval_both.json"))

        arc2 = {}
        arc2.update(_load("ARC-2/arc2_train_both.json"))
        arc2.update(_load("ARC-2/arc2_eval_both.json"))

        # Keep overlaps from ARC-2
        train_data = _flatten_to_train({**arc1, **arc2})

        try:
            concept_raw = _load("ConceptARC/concept_all.json")
            train_data.update(_flatten_to_train(concept_raw))
        except FileNotFoundError:
            print("Warning: ConceptARC file not found.")

        private = json.loads(Path(args.private_json).read_text())
        final_challenges = {**train_data, **private}

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "challenges.json").write_text(json.dumps(final_challenges))

        print(f"Done. Built {len(final_challenges)} tasks into 'challenges.json'.")
        return

    print(f"Building dataset: {args.dataset} (ConceptARC: {args.add_conceptarc})")

    # Define directory mapping based on download_and_group.py folders
    dataset_dirs = {"arc1": "ARC-1", "arc2": "ARC-2"}
    d_dir = dataset_dirs[args.dataset]

    # 1. Load Main Dataset
    # Train: Load 'both' (full data), then flatten test->train for maximum data
    train_raw = _load(f"{d_dir}/{args.dataset}_train_both.json")
    train_data = _flatten_to_train(train_raw)
    
    # Eval: Load 'challenges' (masked inputs) and 'solutions' (outputs)
    eval_challenges = _load(f"{d_dir}/{args.dataset}_eval_challenges.json")
    eval_solutions = _load(f"{d_dir}/{args.dataset}_eval_solutions.json")

    # 2. Load ConceptARC (Optional)
    if args.add_conceptarc:
        try:
            concept_raw = _load("ConceptARC/concept_all.json")
            # ConceptARC is pure training data, so we flatten it too
            concept_flat = _flatten_to_train(concept_raw)
            
            # Check for ID collisions (though unlikely between these datasets)
            collisions = set(train_data.keys()) & set(concept_flat.keys())
            if collisions:
                print(f"Warning: {len(collisions)} ID collisions detected with ConceptARC. Overwriting.")
            
            train_data.update(concept_flat)
        except FileNotFoundError:
            print("Warning: ConceptARC file not found.")
    
    if args.with_filtered:
        other_ds = "arc2" if args.dataset == "arc1" else "arc1"
        other_dir = dataset_dirs[other_ds]
        # Filename matches production in download_and_group.py
        f_name = f"{other_dir}/{other_ds}_filtered.json"
        
        try:
            filtered_raw = _load(f_name)
            # Flatten to ensure all pairs (train+test) are used for training
            filtered_flat = _flatten_to_train(filtered_raw)
            print(f"Merging {len(filtered_flat)} filtered tasks from {other_ds}...")
            train_data.update(filtered_flat)
        except FileNotFoundError:
            print(f"Warning: --with-filtered requested but '{f_name}' not found.")

    # 3. Merge for Output
    # challenges.json contains ALL training tasks (inputs and outputs) + ALL eval challenges (only inputs)
    final_challenges = {**train_data, **eval_challenges}
    
    # solutions.json contains ONLY eval solutions
    final_solutions = eval_solutions

    # 4. Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    (out_dir / "challenges.json").write_text(json.dumps(final_challenges))
    (out_dir / "solutions.json").write_text(json.dumps(final_solutions))

    print(f"Done. Built {len(final_challenges)} tasks into 'challenges.json'.")

if __name__ == "__main__":
    main()
