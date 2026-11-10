"""Package a hash-pinned public ARC holdout as separate challenge/solution JSONs."""

import argparse
import hashlib
import json
from pathlib import Path

from submission_utils import valid_grid


def encoded(value):
    return (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()


def identity(raw):
    return {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def package_holdout(source, split_path, destination, dataset_ref):
    source, split_path, destination = map(Path, (source, split_path, destination))
    split = json.loads(split_path.read_text())
    raw_source = source.read_bytes()
    expected_sha = split["source_manifests"]["arc2/evaluation"]["sha256"]
    if hashlib.sha256(raw_source).hexdigest() != expected_sha:
        raise ValueError("Source ARC2 evaluation data differs from the pinned split.")
    tasks = json.loads(raw_source)
    ids = split["eval_groups"]["development_validation"]
    if len(ids) != len(set(ids)) or set(ids) & set(split["train"]):
        raise ValueError("Holdout IDs must be unique and disjoint from pretraining IDs.")
    challenges, solutions = {}, {}
    for task_id in sorted(ids):
        task = tasks[task_id]
        if not task["train"] or not task["test"]:
            raise ValueError("Every held-out task needs demonstrations and queries.")
        for pair in [*task["train"], *task["test"]]:
            if not all(valid_grid(pair.get(key)) for key in ("input", "output")):
                raise ValueError(f"Invalid solved pair in {task_id}.")
        challenges[task_id] = {
            "train": [{key: pair[key] for key in ("input", "output")} for pair in task["train"]],
            "test": [{"input": pair["input"]} for pair in task["test"]],
        }
        solutions[task_id] = [pair["output"] for pair in task["test"]]
    counts = {"tasks": len(challenges),
              "demonstration_pairs": sum(len(task["train"]) for task in challenges.values()),
              "query_pairs": sum(len(task["test"]) for task in challenges.values())}
    expected = split["validation"]
    if counts != {"tasks": expected["challenge_tasks"],
                  "demonstration_pairs": expected["demonstration_pairs"], "query_pairs": expected["query_pairs"]}:
        raise ValueError("Packaged holdout counts differ from the pinned split.")
    payloads = {"arc2_val40_challenges.json": encoded(challenges),
                "arc2_val40_solutions.json": encoded(solutions)}
    manifest = {
        "schema": "mdlarc-arc2-val40-v1", "split": split["name"],
        "source_sha256": expected_sha, "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "task_ids": sorted(ids), **counts,
        "files": {name: identity(raw) for name, raw in payloads.items()},
        "adaptation": "Provided demonstration pairs only; query answers are used only after decoding.",
    }
    payloads["holdout_manifest.json"] = encoded(manifest)
    payloads["README.md"] = (
        "# mdlARC ARC2 development holdout\n\n"
        "40 tasks, 121 demonstration pairs, 56 query inputs.\n\n"
        "arc2_val40_challenges.json contains train input/output pairs and input-only test pairs.\n"
        "arc2_val40_solutions.json contains query output grids in matching test-input order.\n"
        "Use only demonstration pairs for adaptation. Read solutions only after predictions finish.\n"
        "This is the frozen arc2_dev80_val40_v1 split from public ARC-AGI-2 evaluation data.\n"
        "This dataset is for validation, not competition submission.\n"
    ).encode()
    payloads["dataset-metadata.json"] = encoded({
        "id": dataset_ref, "title": "mdlARC ARC2 Development Holdout 40 v1",
        "licenses": [{"name": "other"}],
        "description": "Exact arc2_dev80_val40_v1 development holdout: 40 public ARC2 tasks, "
                       "121 demonstration pairs and 56 query inputs. Challenges have no query answers. "
                       "Solutions are separate and intended only for post-generation scoring. "
                       "Source and file checksums are recorded in holdout_manifest.json.",
    })
    destination.mkdir(parents=True, exist_ok=True)
    if {path.name for path in destination.iterdir()} - set(payloads):
        raise ValueError("Package directory contains unexpected files.")
    for name, raw in payloads.items():
        path = destination / name
        if path.exists() and path.read_bytes() != raw:
            raise ValueError(f"Existing immutable package differs: {name}")
        path.write_bytes(raw)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-ref", required=True)
    args = parser.parse_args()
    print(json.dumps(package_holdout(args.source, args.split, args.output, args.dataset_ref), indent=2))
