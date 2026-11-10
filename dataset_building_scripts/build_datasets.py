#!/usr/bin/env python3
"""
Build final ARC dataset artifacts from grouped assets.
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ASSETS_TMP = REPO_ROOT / "assets_tmp"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "assets"
DATASET_DIRS = {"arc1": "ARC-1", "arc2": "ARC-2"}
DATASET_CHOICES = (*DATASET_DIRS, "conceptarc")
GENERATED_FILENAMES = (
    "challenges.json",
    "solutions.json",
    "priv_challenges.json",
    "rearc_challenges.json",
)
NVARC_DATASET = "sorokin/nvarc-synthetic-puzzles"
NVARC_DIR_NAMES = {
    "nvarc_training": ("nvarc_training", "nvarc-training"),
    "nvarc_full": ("nvarc_full", "nvarc-full"),
}
KNOWN_REARC_PAIR_EXCLUSIONS = {"e26a3af2": frozenset({2,12,18,26,28,32,55,60,62,77,88,106,109,112,113,116,122,141,147,151,153,154,155,156,158,169,172,173,180,183,201,207,211,220,227,228,236,246,255,280,286,302,311,318,325,329,335,353,356,361,376,378,381,393,400,421,431,444,445,451,453,473,475,491,493,495,508,509,512,514,516,528,539,543,560,564,568,572,591,595,600,605,607,611,612,623,626,627,636,647,649,651,654,661,679,688,697,699,701,705,718,726,728,753,757,759,763,785,796,799,800,807,829,830,834,839,847,849,857,865,889,890,896,901,915,966,970,974,987})}


def _load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")
    with path.open("r") as handle:
        return json.load(handle)


def _load_asset(filename: str):
    return _load_json(ASSETS_TMP / filename)


def _flatten_to_train(tasks: Mapping[str, Mapping[str, object]]) -> Dict[str, dict]:
    """Move test pairs into train and drop the test key."""
    flat_tasks: Dict[str, dict] = {}
    for task_id, data in tasks.items():
        task = dict(data)
        train_pairs = list(task.get("train", []))
        test_pairs = list(task.get("test", []))
        task["train"] = train_pairs + test_pairs
        task.pop("test", None)
        flat_tasks[task_id] = task
    return flat_tasks


def _rearc_count(value: str) -> int:
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--add-rearc expects an integer.") from exc

    if not 1 <= count <= 1000:
        raise argparse.ArgumentTypeError("--add-rearc must be between 1 and 1000.")
    return count


def _merge_sources(*named_sources: Tuple[str, Mapping[str, dict]]) -> Dict[str, dict]:
    merged: Dict[str, dict] = {}
    for source_name, tasks in named_sources:
        collisions = set(merged.keys()) & set(tasks.keys())
        if collisions:
            print(
                f"Warning: {len(collisions)} task id collisions while merging {source_name}; "
                "later sources override earlier ones."
            )
        merged.update(tasks)
    return merged


def _load_train_flat(dataset: str) -> Dict[str, dict]:
    dataset_dir = DATASET_DIRS[dataset]
    return _flatten_to_train(_load_asset(f"{dataset_dir}/{dataset}_train_both.json"))


def _load_eval_challenges(dataset: str) -> Dict[str, dict]:
    dataset_dir = DATASET_DIRS[dataset]
    return _load_asset(f"{dataset_dir}/{dataset}_eval_challenges.json")


def _load_eval_solutions(dataset: str) -> Dict[str, list]:
    dataset_dir = DATASET_DIRS[dataset]
    return _load_asset(f"{dataset_dir}/{dataset}_eval_solutions.json")


def _load_eval_flat_with_solutions(dataset: str) -> Dict[str, dict]:
    challenges = _load_eval_challenges(dataset)
    solutions = _load_eval_solutions(dataset)
    solved_challenges: Dict[str, dict] = {}

    for task_id, data in challenges.items():
        task = dict(data)
        test_pairs = list(task.get("test", []))
        task_solutions = solutions.get(task_id)
        if task_solutions is None:
            raise ValueError(f"Missing eval solutions for task {task_id}.")
        if len(test_pairs) != len(task_solutions):
            raise ValueError(
                f"Eval challenge/solution test pair mismatch for task {task_id}: "
                f"{len(test_pairs)} challenges, {len(task_solutions)} solutions."
            )

        task["test"] = [
            {**pair, "output": output}
            for pair, output in zip(test_pairs, task_solutions)
        ]
        solved_challenges[task_id] = task

    return _flatten_to_train(solved_challenges)


def _load_filtered_flat(dataset: str) -> Dict[str, dict]:
    dataset_dir = DATASET_DIRS[dataset]
    return _flatten_to_train(_load_asset(f"{dataset_dir}/{dataset}_filtered.json"))


def _load_conceptarc() -> Dict[str, dict]:
    return _load_asset("ConceptARC/concept_all.json")


def _load_conceptarc_flat() -> Dict[str, dict]:
    return _flatten_to_train(_load_conceptarc())


def _build_conceptarc_eval_outputs(
    tasks: Mapping[str, Mapping[str, object]],
) -> Tuple[Dict[str, dict], Dict[str, list]]:
    """Split solved ConceptARC tasks into challenge and solution artifacts."""
    challenges: Dict[str, dict] = {}
    solutions: Dict[str, list] = {}

    for task_id, data in tasks.items():
        test_challenges = []
        test_solutions = []
        for pair_index, pair in enumerate(data.get("test", [])):
            if "input" not in pair:
                raise ValueError(
                    f"Missing ConceptARC test input for {task_id} pair {pair_index}."
                )
            if "output" not in pair:
                raise ValueError(
                    f"Missing ConceptARC test output for {task_id} pair {pair_index}."
                )
            test_challenges.append({"input": pair["input"]})
            test_solutions.append(pair["output"])

        challenges[task_id] = {
            "train": list(data.get("train", [])),
            "test": test_challenges,
        }
        solutions[task_id] = test_solutions

    return challenges, solutions


def _sanitize_rearc_pairs(task_id: str, pairs: Sequence[dict]) -> list[dict]:
    excluded_indices = KNOWN_REARC_PAIR_EXCLUSIONS.get(task_id)
    if not excluded_indices:
        return list(pairs)

    filtered_pairs = [
        pair for index, pair in enumerate(pairs) if index not in excluded_indices
    ]
    removed_count = len(pairs) - len(filtered_pairs)
    if removed_count:
        print(
            f"Sanitized RE-ARC task {task_id}: removed {removed_count} known "
            "problematic pairs."
        )
    return filtered_pairs


def _load_rearc_pairs() -> Dict[str, Sequence[dict]]:
    rearc_dir = ASSETS_TMP / "RE-ARC" / "tasks"
    if not rearc_dir.is_dir():
        raise FileNotFoundError(f"Missing expected RE-ARC directory: {rearc_dir}")

    return {
        task_path.stem: _sanitize_rearc_pairs(task_path.stem, _load_json(task_path))
        for task_path in sorted(rearc_dir.glob("*.json"))
    }


def _build_rearc_tasks(
    arc1_train: Mapping[str, dict],
    *,
    protected_task_ids: Iterable[str],
    sample_count: Optional[int],
) -> Dict[str, dict]:
    rearc_pairs_by_task = _load_rearc_pairs()
    protected = set(protected_task_ids)
    rng = random.Random(0)

    rearc_tasks: Dict[str, dict] = {}
    skipped_protected = 0
    missing_rearc_pairs = 0
    truncated_tasks = 0
    total_pairs_added = 0

    for task_id, task in sorted(arc1_train.items()):
        if task_id in protected:
            skipped_protected += 1
            continue

        available_pairs = list(rearc_pairs_by_task.get(task_id, []))
        if not available_pairs:
            missing_rearc_pairs += 1

        if sample_count is None:
            selected_pairs = available_pairs
        else:
            selected_count = min(sample_count, len(available_pairs))
            if len(available_pairs) < sample_count:
                truncated_tasks += 1
            selected_pairs = rng.sample(available_pairs, selected_count)

        augmented_train = list(task.get("train", []))
        augmented_train.extend(selected_pairs)
        rearc_tasks[task_id] = {**task, "train": augmented_train}
        total_pairs_added += len(selected_pairs)

    print(f"Built {len(rearc_tasks)} ARC-1 RE-ARC tasks.")
    if sample_count is None:
        print(f"Appended all {total_pairs_added} available RE-ARC pairs.")
    else:
        print(
            f"Appended {total_pairs_added} total RE-ARC pairs "
            f"with sample size {sample_count}."
        )
    if skipped_protected:
        print(
            f"Excluded {skipped_protected} ARC-1 train tasks from RE-ARC because they "
            "overlap protected eval/private tasks."
        )
    if missing_rearc_pairs:
        print(
            f"Warning: {missing_rearc_pairs} ARC-1 train tasks had no matching RE-ARC file; "
            "keeping only their base ARC-1 train pairs."
        )
    if sample_count is not None and truncated_tasks:
        print(
            f"Warning: {truncated_tasks} tasks had fewer than {sample_count} RE-ARC pairs; "
            "used all available pairs for those tasks."
        )

    return rearc_tasks


def _pop_rearc_tasks_from_challenges(
    challenges: Dict[str, dict], rearc_tasks: Mapping[str, dict]
) -> None:
    removed = 0
    for task_id in rearc_tasks:
        if task_id in challenges:
            del challenges[task_id]
            removed += 1
    print(f"Removed {removed} RE-ARC task ids from challenges.json.")


def _write_outputs(output_dir: Path, outputs: Mapping[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    keep = set(outputs.keys())

    for filename in GENERATED_FILENAMES:
        output_path = output_dir / filename
        if filename in keep:
            output_path.write_text(json.dumps(outputs[filename]))
        elif output_path.exists():
            output_path.unlink()
            print(f"Removed stale {output_path.name}.")


def _find_extracted_nvarc_dir(stage_dir: Path, canonical_name: str) -> Path:
    for candidate in NVARC_DIR_NAMES[canonical_name]:
        match = next((path for path in stage_dir.rglob(candidate) if path.is_dir()), None)
        if match is not None:
            return match
    raise FileNotFoundError(
        f"Could not find extracted {canonical_name} folder after downloading {NVARC_DATASET}."
    )


def _resolve_kaggle_cli() -> Optional[str]:
    kaggle = shutil.which("kaggle")
    if kaggle is not None:
        return kaggle
    candidate = Path(sys.executable).parent / (
        "kaggle.exe" if os.name == "nt" else "kaggle"
    )
    return str(candidate) if candidate.exists() else None


def _kaggle_auth_env() -> dict[str, str]:
    env = dict(os.environ)
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if token:
        env["KAGGLE_API_TOKEN"] = token
        return env
    if env.get("KAGGLE_USERNAME", "").strip() and env.get("KAGGLE_KEY", "").strip():
        return env
    raise RuntimeError(
        "Missing Kaggle auth. Set KAGGLE_API_TOKEN or KAGGLE_USERNAME/KAGGLE_KEY "
        "before requesting NVARC datasets."
    )


def _ensure_nvarc_assets(output_dir: Path, args: argparse.Namespace) -> None:
    requested = {
        name: output_dir / name
        for name, enabled in (
            ("nvarc_training", args.add_nvarc_training),
            ("nvarc_full", args.add_nvarc_full),
        )
        if enabled
    }
    missing = {name: path for name, path in requested.items() if not path.exists()}
    if not missing:
        return

    kaggle = _resolve_kaggle_cli()
    if kaggle is None:
        raise RuntimeError(
            "Missing 'kaggle' CLI. Install it and configure Kaggle credentials before "
            "requesting NVARC datasets."
        )

    with tempfile.TemporaryDirectory(prefix="nvarc_") as tmpdir:
        stage_dir = Path(tmpdir)
        try:
            subprocess.run(
                [kaggle, "datasets", "download", "-d", NVARC_DATASET, "-p", str(stage_dir)],
                check=True,
                capture_output=True,
                text=True,
                env=_kaggle_auth_env(),
            )
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            raise RuntimeError(
                "Failed to download NVARC datasets via Kaggle. Ensure the Kaggle CLI is "
                f"installed and authenticated.\n{detail}"
            ) from exc

        zip_path = stage_dir / "nvarc-synthetic-puzzles.zip"
        try:
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(stage_dir)
        except Exception as exc:
            raise RuntimeError(f"Failed to extract {zip_path.name}: {exc}") from exc

        output_dir.mkdir(parents=True, exist_ok=True)
        for canonical_name, target_path in missing.items():
            shutil.move(
                str(_find_extracted_nvarc_dir(stage_dir, canonical_name)),
                str(target_path),
            )
            print(f"Staged {canonical_name} into {target_path}.")


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    if args.prefinesub:
        disallowed = []
        if args.submission:
            disallowed.append("--submission")
        if args.style is not None:
            disallowed.append("--style")
        if args.dataset is not None:
            disallowed.append("--dataset")
        if args.private_json is not None:
            disallowed.append("--private-json")
        if args.add_conceptarc:
            disallowed.append("--add-conceptarc")
        if args.keep_train_test_separate:
            disallowed.append("--keep-train-test-separate")
        if args.add_rearc is not None:
            disallowed.append("--add-rearc")
        if args.add_filtered:
            disallowed.append("--add-filtered")
        if args.add_nvarc_training:
            disallowed.append("--add-nvarc-training")
        if args.add_nvarc_full:
            disallowed.append("--add-nvarc-full")
        if disallowed:
            parser.error(
                "--prefinesub cannot be combined with "
                + ", ".join(disallowed)
                + "."
            )

        args.style = "pretrain"
        args.dataset = "arc2"
        args.add_conceptarc = True
        args.add_filtered = True
        args.add_rearc = 1000
        args.add_nvarc_training = True
        args.add_nvarc_full = True
        if args.output_dir is None:
            args.output_dir = DEFAULT_OUTPUT_DIR
    elif args.submission:
        disallowed = []
        if args.style is not None:
            disallowed.append("--style")
        if args.dataset is not None:
            disallowed.append("--dataset")
        if args.add_conceptarc:
            disallowed.append("--add-conceptarc")
        if args.keep_train_test_separate:
            disallowed.append("--keep-train-test-separate")
        if args.add_rearc is not None:
            disallowed.append("--add-rearc")
        if args.add_filtered:
            disallowed.append("--add-filtered")
        if args.add_nvarc_training:
            disallowed.append("--add-nvarc-training")
        if args.add_nvarc_full:
            disallowed.append("--add-nvarc-full")
        if disallowed:
            parser.error(
                "--submission cannot be combined with "
                + ", ".join(disallowed)
                + "."
            )
        if args.private_json is None:
            parser.error("--submission requires --private-json.")
        if args.output_dir is None:
            parser.error("--submission requires --output-dir.")
    else:
        if args.style is None or args.dataset is None:
            parser.error("--style and --dataset are required unless --submission is used.")
        if args.private_json is not None:
            parser.error("--private-json is only valid with --submission.")
        if args.dataset == "conceptarc":
            disallowed = []
            if args.style != "pretrain":
                disallowed.append("--style finetune")
            if args.add_conceptarc:
                disallowed.append("--add-conceptarc")
            if args.add_rearc is not None:
                disallowed.append("--add-rearc")
            if args.add_filtered:
                disallowed.append("--add-filtered")
            if args.add_nvarc_training:
                disallowed.append("--add-nvarc-training")
            if args.add_nvarc_full:
                disallowed.append("--add-nvarc-full")
            if disallowed:
                parser.error(
                    "ConceptARC-only builds require --style pretrain and cannot be "
                    "combined with " + ", ".join(disallowed) + "."
                )
            if not args.keep_train_test_separate:
                parser.error(
                    "ConceptARC-only builds require --keep-train-test-separate."
                )
        elif args.keep_train_test_separate:
            parser.error(
                "--keep-train-test-separate is only valid with --dataset conceptarc."
            )
        if args.output_dir is None:
            args.output_dir = DEFAULT_OUTPUT_DIR

    return args


def _build_style_outputs(args: argparse.Namespace) -> Dict[str, object]:
    train_dataset = args.dataset
    if train_dataset == "conceptarc":
        challenges, solutions = _build_conceptarc_eval_outputs(_load_conceptarc())
        return {
            "challenges.json": challenges,
            "solutions.json": solutions,
        }

    train_data = _load_train_flat(train_dataset)
    other_dataset = "arc2" if train_dataset == "arc1" else "arc1"

    train_sources = [("base train", train_data)]

    if args.add_conceptarc:
        concept_data = _load_conceptarc_flat()
        train_sources.append(("ConceptARC", concept_data))

    if args.add_filtered:
        filtered_data = _load_filtered_flat(other_dataset)
        train_sources.append((f"{other_dataset} filtered", filtered_data))

    if args.style == "pretrain":
        eval_dataset = train_dataset
        if getattr(args, "prefinesub", False):
            eval_challenges = _load_eval_flat_with_solutions(eval_dataset)
            outputs: Dict[str, object] = {}
        else:
            eval_challenges = _load_eval_challenges(eval_dataset)
            outputs = {
                "solutions.json": _load_eval_solutions(eval_dataset),
            }
        protected_ids = set(eval_challenges.keys())
        challenges = _merge_sources(*train_sources, (f"{eval_dataset} eval", eval_challenges))
    else:
        priv_challenges = _load_eval_challenges(train_dataset)
        outputs = {"priv_challenges.json": priv_challenges, "solutions.json": _load_eval_solutions(train_dataset)}
        protected_ids = set(priv_challenges.keys())
        challenges = _merge_sources(*train_sources)
    
    # Build RE-ARC last so ARC-1 train task ids can be removed from the final
    # public challenges split after all other additions are merged in.
    if args.add_rearc is not None:
        arc1_train = _load_train_flat("arc1")
        rearc_tasks = _build_rearc_tasks(arc1_train, protected_task_ids=protected_ids, sample_count=args.add_rearc)
        _pop_rearc_tasks_from_challenges(challenges, rearc_tasks)
        outputs["rearc_challenges.json"] = rearc_tasks

    outputs["challenges.json"] = challenges
    return outputs


def _build_submission_outputs(args: argparse.Namespace) -> Dict[str, object]:
    arc1_train = _load_train_flat("arc1")
    arc1_eval = _load_eval_challenges("arc1")
    arc2_train = _load_train_flat("arc2")
    arc2_eval = _load_eval_challenges("arc2")
    concept_data = _load_conceptarc_flat()
    private_tasks = _load_json(Path(args.private_json))

    protected_ids = set(arc1_eval.keys()) | set(arc2_eval.keys()) | set(private_tasks.keys())
    rearc_tasks = _build_rearc_tasks(
        arc1_train,
        protected_task_ids=protected_ids,
        sample_count=None,
    )

    challenges = _merge_sources(
        ("arc2 train", arc2_train),
        ("ConceptARC", concept_data),
        ("arc1 eval", arc1_eval),
        ("arc2 eval", arc2_eval),
        ("private", private_tasks),
    )
    _pop_rearc_tasks_from_challenges(challenges, rearc_tasks)

    return {
        "challenges.json": challenges,
        "rearc_challenges.json": rearc_tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ARC datasets.")
    parser.add_argument("--style", choices=["pretrain", "finetune"], help="Dataset-building mode.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, help="Primary ARC dataset.")
    parser.add_argument("--submission",action="store_true",help="Build the submission dataset with public, ConceptARC, private, and RE-ARC splits.")
    parser.add_argument("--prefinesub",action="store_true",help="Build the public-data pre-finetune preset: ARC-2 pretrain with add-ons, with ARC-2 eval answers folded into train.")
    parser.add_argument("--private-json",help="Path to the private challenge JSON. Required with --submission.")
    parser.add_argument("--add-conceptarc",action="store_true",help="Merge ConceptARC into challenges.json.")
    parser.add_argument("--keep-train-test-separate",action="store_true",help="Keep ConceptARC train and test pairs separate. Required for ConceptARC-only builds.")
    parser.add_argument("--add-rearc",type=_rearc_count,metavar="N",help="Create rearc_challenges.json from ARC-1 train tasks, augmented with N RE-ARC pairs.")
    parser.add_argument("--add-filtered",action="store_true",help="Merge filtered tasks from the other ARC dataset into challenges.json.")
    parser.add_argument("--add-nvarc-training",action="store_true",help="Download and stage the train-only NVARC training dataset into output_dir/nvarc_training.")
    parser.add_argument("--add-nvarc-full",action="store_true",help="Download and stage the train-only NVARC full dataset into output_dir/nvarc_full.")
    parser.add_argument("--output-dir",type=Path,help="Directory to save output files.")
    args = _validate_args(parser, parser.parse_args())

    output_dir = Path(args.output_dir)
    if args.submission:
        print("Building submission dataset.")
        outputs = _build_submission_outputs(args)
    else:
        if args.prefinesub:
            print("Building prefinesub dataset.")
        else:
            print(
                f"Building dataset with style={args.style}, dataset={args.dataset}, "
                f"ConceptARC={args.add_conceptarc}, filtered={args.add_filtered}, "
                f"RE-ARC={args.add_rearc!r}, NVARC training={args.add_nvarc_training}, "
                f"NVARC full={args.add_nvarc_full}."
            )
        outputs = _build_style_outputs(args)

    _ensure_nvarc_assets(output_dir, args)
    _write_outputs(output_dir, outputs)
    print(f"Done. Wrote {', '.join(sorted(outputs.keys()))} to {output_dir.resolve()}.")

if __name__ == "__main__":
    main()
