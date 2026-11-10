"""Build run-local artifacts from original tasks or versioned task variations."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import zipfile
import zlib
from pathlib import Path

from .task_splits import content_sha256, load_parent_split, validate_split_tasks

ROOT = Path(__file__).resolve().parents[1]
ASSETS_TMP = ROOT / "assets_tmp"
SOURCE_DIRS = {
    "arc2_train": "arc2 train",
    "arc2_eval": "arc2 eval",
    "conceptarc": "conceptarc",
    "arc1_filtered": "arc1 filtered",
}
ORIGINAL_FILES = {
    "arc2_train": "ARC-2/arc2_train_both.json",
    "arc2_eval": "ARC-2/arc2_eval_both.json",
    "conceptarc": "ConceptARC/concept_all.json",
    "arc1_filtered": "ARC-1/arc1_filtered.json",
}
SEED_NAME = re.compile(r"(.+)seed(\d+)\.json\Z")
EVALUATION_FILES = {
    "arc1_eval": ("ARC-1/arc1_eval_challenges.json", "ARC-1/arc1_eval_solutions.json"),
    "arc2_eval": ("ARC-2/arc2_eval_challenges.json", "ARC-2/arc2_eval_solutions.json"),
    "arc_heavy100": ("ARC-Heavy/heavy100_challenges.json", "ARC-Heavy/heavy100_solutions.json"),
    "arc_new80": ("ARC-New80/new80_challenges.json", "ARC-New80/new80_solutions.json"),
}


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _archive_identity() -> tuple[Path, str]:
    config = _load(ROOT / "scripts/colab_drive.json")
    archive = ASSETS_TMP / "task-variations-v1.zip"
    if not archive.is_file() or archive.is_symlink():
        raise FileNotFoundError(f"Variation archive is required: {archive}")
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if archive.stat().st_size != config["dataset_size"] or actual != config["dataset_sha256"]:
        raise ValueError("task-variations-v1.zip does not match the approved size and SHA-256")
    return archive, actual


def _ensure_extracted(archive: Path) -> Path:
    destination = ASSETS_TMP / "exported-datasets"
    if destination.is_dir() and not destination.is_symlink():
        return destination / "task-variation"
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Invalid variation extraction destination: {destination}")
    with tempfile.TemporaryDirectory(prefix=".task-variations-", dir=ASSETS_TMP) as temp_name:
        staging = Path(temp_name)
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                name = member.filename
                if member.is_dir() or name.startswith("__MACOSX/") or name.endswith("/.DS_Store"):
                    continue
                path = Path(name)
                if (path.is_absolute() or ".." in path.parts or "\\" in name
                        or not path.parts or path.parts[0] != "exported-datasets"):
                    raise ValueError(f"Unsafe variation archive member: {name}")
                mode = (member.external_attr >> 16) & 0o170000
                if mode not in (0, 0o100000):
                    raise ValueError(f"Unsupported variation archive member: {name}")
                target = staging / path
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as input_file, target.open("xb") as output_file:
                    shutil.copyfileobj(input_file, output_file)
        (staging / "exported-datasets").rename(destination)
    return destination / "task-variation"


def _composition_tasks(identity: dict, sources: list[str]) -> dict:
    """Verify the packaged, solved composition samples before lazy training."""
    relative = Path(identity["archive_path"])
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
        raise ValueError("Unsafe composition archive path")
    archive = ROOT / relative
    if archive.is_symlink() or not archive.is_file():
        raise ValueError("Composition archive is missing or linked")
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if archive.stat().st_size != identity["size"] or digest.hexdigest() != identity["sha256"]:
        raise ValueError("Composition archive differs from the approved experiment identity")
    destination = ASSETS_TMP / ("composition-cache-" + identity["sha256"])
    tasks = {}
    seen = set()
    with zipfile.ZipFile(archive) as source:
        if len(source.namelist()) != len(set(source.namelist())):
            raise ValueError("Duplicate composition archive members")
        manifest = json.loads(source.read("composition_manifest.json"))
        if manifest.get("schema_version") != 1 or not manifest.get("samples"):
            raise ValueError("Composition manifest is empty or unsupported")
        if set(source.namelist()) != {"composition_manifest.json", *(item["path"] for item in manifest["samples"])}:
            raise ValueError("Unexpected composition archive members")
        for item in manifest["samples"]:
            if item["source"] not in sources:
                raise ValueError("Composition lineage is outside the selected training sources")
            name = Path(item["path"])
            if (name.is_absolute() or ".." in name.parts or "\\" in str(name)
                    or not name.parts or name.parts[0] != "task-compositions" or name.suffix != ".json"):
                raise ValueError("Unsafe composition sample path")
            info = source.getinfo(item["path"])
            if (info.external_attr >> 16) & 0o170000 not in (0, 0o100000):
                raise ValueError("Linked composition sample")
            parent_id = item["composition_id"]
            if (not isinstance(parent_id, str) or not parent_id.startswith("composition/")
                    or not re.fullmatch(r"[A-Za-z0-9_./-]+", parent_id) or ".." in Path(parent_id).parts):
                raise ValueError("Unsafe composition task identity")
            seed = item["seed"]
            if type(seed) is not int or seed < 0 or (parent_id, seed) in seen:
                raise ValueError("Duplicate or invalid composition seed")
            seen.add((parent_id, seed))
            content = source.read(item["path"])
            if hashlib.sha256(content).hexdigest() != item["sha256"]:
                raise ValueError("Composition sample checksum changed")
            pairs = _solved_pairs(json.loads(content), item["path"])
            path = destination / name
            path.resolve().relative_to(destination.resolve())
            if path.is_symlink():
                raise ValueError("Linked extracted composition sample")
            if path.exists():
                if path.read_bytes() != content:
                    raise ValueError("Extracted composition sample changed")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            tasks.setdefault(parent_id, []).append({"seed": seed, "path": path.relative_to(ROOT).as_posix(),
                                                   "pair_count": len(pairs), "kind": "composition"})
    return {key: sorted(values, key=lambda item: item["seed"]) for key, values in sorted(tasks.items())}


def _solved_pairs(task: dict, label: str) -> list[dict]:
    pairs = list(task.get("train", [])) + list(task.get("test", []))
    if not pairs or any(not isinstance(pair, dict) or "input" not in pair or "output" not in pair for pair in pairs):
        raise ValueError(f"Missing solved input/output pairs: {label}")
    for pair in pairs:
        for key in ("input", "output"):
            grid = pair[key]
            if (not isinstance(grid, list) or not 1 <= len(grid) <= 30
                    or not isinstance(grid[0], list) or not 1 <= len(grid[0]) <= 30):
                raise ValueError(f"Invalid ARC grid in {label}: {key}")
            width = len(grid[0])
            if any(
                not isinstance(row, list) or len(row) != width
                or any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 9
                       for value in row)
                for row in grid
            ):
                raise ValueError(f"Invalid ARC grid in {label}: {key}")
    return pairs


def _evaluation_data(config: dict) -> tuple[dict, dict, dict, dict | None, dict | None]:
    evaluation = config.get("evaluation", {"source": "arc2_eval"})
    source = evaluation["source"]
    manifest = identity = None
    if evaluation.get("split_manifest"):
        manifest, identity = load_parent_split(ROOT, evaluation["split_manifest"])
        originals = _load(ASSETS_TMP / ORIGINAL_FILES[manifest["source"]])
        validate_split_tasks(manifest, originals)
        challenges, solutions, groups = {}, {}, {}
        for group, native_ids in manifest["eval_groups"].items():
            groups[group] = []
            for native_id in native_ids:
                task_id = f"{manifest['source']}/{native_id}"
                task = originals[native_id]
                _solved_pairs(task, task_id)
                if not task.get("train") or not task.get("test"):
                    raise ValueError(f"Held-out task requires original train and test pairs: {task_id}")
                challenges[task_id] = {
                    "train": task["train"],
                    "test": [{"input": pair["input"]} for pair in task["test"]],
                }
                solutions[task_id] = [pair["output"] for pair in task["test"]]
                groups[group].append(task_id)
        if evaluation.get("extra_tasks"):
            _add_extra_tasks(evaluation["extra_tasks"], challenges, solutions, groups)
    else:
        challenge_file, solution_file = EVALUATION_FILES[source]
        challenges = _load(ASSETS_TMP / challenge_file)
        solutions = _load(ASSETS_TMP / solution_file)
        groups = {}
    if set(challenges) != set(solutions):
        raise ValueError(f"{source} challenge and solution IDs differ")
    for task_id, task in challenges.items():
        if any("output" in pair for pair in task.get("test", [])):
            raise ValueError(f"Evaluation test output found in challenges: {task_id}")
        if len(task.get("test", [])) != len(solutions[task_id]):
            raise ValueError(f"Evaluation answer count mismatch: {task_id}")
    return challenges, solutions, groups, manifest, identity


def _add_extra_tasks(extra: dict, challenges: dict, solutions: dict, groups: dict) -> None:
    """Add the tasks of extra challenge/solution files to one evaluation group (adapted on and scored).

    Tasks already in that group (same native ID) must match it exactly; the others become "<prefix>/<native ID>".
    """
    group, prefix = extra["group"], extra["prefix"]
    if group not in groups:
        raise ValueError(f"extra_tasks.group is not an evaluation group: {group!r}")
    extra_challenges = _load(ROOT / extra["challenges"])
    extra_solutions = _load(ROOT / extra["solutions"])
    if set(extra_challenges) != set(extra_solutions):
        raise ValueError("extra_tasks challenge and solution IDs differ")
    existing = {task_id.rsplit("/", 1)[-1]: task_id for task_id in groups[group]}
    for native_id, task in extra_challenges.items():
        answers = extra_solutions[native_id]
        if any("output" in pair for pair in task.get("test", [])) or len(task.get("test", [])) != len(answers):
            raise ValueError(f"extra_tasks test pairs do not match their answers: {native_id}")
        _solved_pairs({"train": task.get("train", []),
                       "test": [{"input": pair["input"], "output": answer} for pair, answer in zip(task["test"], answers)]},
                      native_id)
        if native_id in existing:
            task_id = existing[native_id]
            if task != challenges[task_id] or answers != solutions[task_id]:
                raise ValueError(f"extra_tasks differs from its held-out original: {native_id}")
            continue
        task_id = f"{prefix}/{native_id}"
        if task_id in challenges:
            raise ValueError(f"Duplicate extra task: {task_id}")
        challenges[task_id] = {"train": task["train"], "test": [{"input": pair["input"]} for pair in task["test"]]}
        solutions[task_id] = answers
        groups[group].append(task_id)


def _split_contents(split_identity: dict | None) -> bytes | None:
    if split_identity is None:
        return None
    contents = (ROOT / split_identity["manifest"]).read_bytes()
    if hashlib.sha256(contents).hexdigest() != split_identity["sha256"]:
        raise ValueError("Parent split changed while building the dataset")
    return contents


def _write_outputs(output_dir: Path, outputs: dict, eval_groups: dict,
                   manifest_contents: bytes | None) -> None:
    challenges, solutions = outputs["priv_challenges.json"], outputs["solutions.json"]
    for group, ids in eval_groups.items():
        outputs[f"eval_{group}_challenges.json"] = {task_id: challenges[task_id] for task_id in ids}
        outputs[f"eval_{group}_solutions.json"] = {task_id: solutions[task_id] for task_id in ids}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in outputs.items():
        (output_dir / name).write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")
    if manifest_contents is not None:
        (output_dir / "split_manifest.json").write_bytes(manifest_contents)


def build_original_dataset(config: dict, output_dir: Path) -> dict:
    """Use solved original parents, with held-out tests reserved for evaluation."""
    settings = config["pretrain_data"]
    challenges, solutions, groups, manifest, identity = _evaluation_data(config)
    manifest_contents = _split_contents(identity)
    train_ids = set(manifest["train"]) if manifest else None
    protected_ids = {task_id.rsplit("/", 1)[-1] for task_id in challenges}
    tasks, source_hashes = {}, {}
    for source in settings["sources"]:
        originals = _load(ASSETS_TMP / ORIGINAL_FILES[source])
        source_hashes[source] = content_sha256(originals)
        for native_id, task in originals.items():
            if manifest is not None and source == manifest["source"] and native_id not in train_ids:
                continue
            if native_id in protected_ids:
                raise ValueError(f"Evaluation original cannot be added as a training source: {native_id}")
            task_id = f"{source}/{native_id}"
            tasks[task_id] = {"train": _solved_pairs(task, task_id)}
    original_count = len(tasks)
    include_support = settings["eval_support_repeats_per_epoch"] > 0
    if manifest is not None and include_support:
        raise ValueError("A parent split must keep evaluation support out of pretraining")
    if include_support:
        # Keep held-out test inputs and answers outside the pretrain dataset.
        tasks.update({task_id: {"train": task["train"]} for task_id, task in challenges.items()})
    if not tasks:
        raise ValueError("Original-task pretraining requires training tasks")
    metadata = {
        "schema_version": 1,
        "kind": "original_tasks",
        "sources": settings["sources"],
        "source_sha256": source_hashes,
        "evaluation_source": config["evaluation"]["source"],
        "eval_support_repeats_per_epoch": settings["eval_support_repeats_per_epoch"],
        "training_sha256": hashlib.sha256(json.dumps(tasks, separators=(",", ":")).encode("utf-8")).hexdigest(),
    }
    if identity is not None:
        metadata["parent_split"] = identity
    _write_outputs(output_dir, {
        "challenges.json": tasks,
        "priv_challenges.json": challenges,
        "solutions.json": solutions,
        "dataset_manifest.json": metadata,
    }, groups, manifest_contents)
    return {
        "original_tasks": original_count,
        "training_pairs": sum(len(task["train"]) for task in tasks.values()),
        "evaluation_tasks": len(challenges),
        "evaluation_groups": {group: len(ids) for group, ids in groups.items()},
    }


def build_variation_dataset(config: dict, output_dir: Path) -> dict:
    """Verify the snapshot and build training plus original evaluation artifacts."""
    archive, archive_sha256 = _archive_identity()
    root = _ensure_extracted(archive)
    if not root.is_dir():
        raise FileNotFoundError(f"Missing extracted variation tree: {root}")
    settings = config["pretrain_data"]
    public_pretraining = settings.get("public_pretraining", False)
    sources = settings["sources"]
    original_sources = settings["original_sources"]
    eval_challenges, eval_solutions, eval_groups, manifest, split_identity = _evaluation_data(config)
    manifest_contents = _split_contents(split_identity)
    protected_ids = {task_id.rsplit("/", 1)[-1] for task_id in eval_challenges}
    train_ids = set(manifest["train"]) if manifest else None

    def allowed_parent(source: str, native_id: str) -> bool:
        return manifest is None or source != manifest["source"] or native_id in train_ids

    with zipfile.ZipFile(archive) as source_zip:
        zip_info = {item.filename: item for item in source_zip.infolist()}

    parents: dict[str, list[dict]] = {}
    for source in sources:
        source_dir = root / SOURCE_DIRS[source]
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing variation source: {source_dir}")
        for task_dir in sorted(source_dir.iterdir()):
            if not task_dir.is_dir() or task_dir.is_symlink():
                raise ValueError(f"Unexpected variation task entry: {task_dir}")
            native_id = task_dir.name
            if not allowed_parent(source, native_id):
                continue
            if native_id in protected_ids and not public_pretraining:
                raise ValueError(f"Evaluation variation cannot be used in training: {native_id}")
            parent_id = f"{source}/{native_id}"
            variants = []
            for path in sorted(task_dir.glob("*.json")):
                match = SEED_NAME.fullmatch(path.name)
                if not match or match.group(1) != native_id or path.is_symlink():
                    raise ValueError(f"Unexpected variation file: {path}")
                seed = int(match.group(2))
                archive_name = path.relative_to(ASSETS_TMP).as_posix()
                member = zip_info.get(archive_name)
                content = path.read_bytes()
                if member is None or len(content) != member.file_size or zlib.crc32(content) != member.CRC:
                    raise ValueError(f"Extracted variation differs from approved ZIP: {path}")
                pair_count = len(_solved_pairs(json.loads(content), str(path)))
                variants.append({
                    "seed": seed,
                    "path": path.relative_to(ROOT).as_posix(),
                    "pair_count": pair_count,
                })
            if sorted(item["seed"] for item in variants) != list(range(42)):
                raise ValueError(f"Expected seeds 0..41 for {parent_id}")
            parents[parent_id] = sorted(variants, key=lambda item: item["seed"])
        if manifest is not None and source == manifest["source"]:
            actual_ids = {parent_id.split("/", 1)[1] for parent_id in parents if parent_id.startswith(f"{source}/")}
            if actual_ids != train_ids:
                raise ValueError("Variation tree does not contain every training parent in the split")

    static_tasks = {parent_id: {"train": []} for parent_id in parents}
    original_task_count = 0
    for source in original_sources:
        originals = _load(ASSETS_TMP / ORIGINAL_FILES[source])
        for native_id, task in originals.items():
            if not allowed_parent(source, native_id):
                continue
            if native_id in protected_ids and not public_pretraining:
                raise ValueError(f"Evaluation original cannot be added as a training source: {native_id}")
            parent_id = f"{source}/{native_id}"
            static_tasks[parent_id] = {"train": _solved_pairs(task, parent_id)}
            original_task_count += 1
    include_support = settings["eval_support_repeats_per_epoch"] > 0
    if manifest is not None and include_support:
        raise ValueError("A parent split must keep evaluation support out of pretraining")
    if include_support:
        static_tasks.update(eval_challenges)

    composition_parents = []
    composition_identity = settings.get("compositions")
    if composition_identity:
        composition_tasks = _composition_tasks(composition_identity, sources)
        for parent_id, variants in composition_tasks.items():
            if parent_id in parents:
                raise ValueError(f"Duplicate composition identity: {parent_id}")
            parents[parent_id] = variants
            static_tasks[parent_id] = {"train": []}
        composition_parents = sorted(composition_tasks)

    output_dir.mkdir(parents=True, exist_ok=True)
    index = {
        "snapshot": "task-variations-v1",
        "archive_sha256": archive_sha256,
        "sources": sources,
        "original_sources": original_sources,
        "sampling": settings["sampling"],
        "eval_support_repeats_per_epoch": settings["eval_support_repeats_per_epoch"],
        "eval_support_task_ids": sorted(eval_challenges) if include_support else [],
        "parents": parents,
    }
    if public_pretraining:
        index["public_pretraining"] = True
    if composition_identity:
        index["composition_parents"] = composition_parents
        index["composition_archive_sha256"] = composition_identity["sha256"]
    if split_identity is not None:
        index["parent_split"] = split_identity
    evaluation_source = config.get("evaluation", {"source": "arc2_eval"})["source"]
    if evaluation_source != "arc2_eval":
        index["evaluation_source"] = evaluation_source
    outputs = {
        "challenges.json": static_tasks,
        "priv_challenges.json": eval_challenges,
        "solutions.json": eval_solutions,
        "variation_index.json": index,
    }
    _write_outputs(output_dir, outputs, eval_groups, manifest_contents)
    return {
        "archive_sha256": archive_sha256,
        "parent_tasks": len(parents),
        "variation_files": sum(len(values) for key, values in parents.items() if key not in composition_parents),
        **({"composition_tasks": len(composition_parents),
            "composition_samples": sum(len(parents[key]) for key in composition_parents)} if composition_identity else {}),
        "original_tasks": original_task_count,
        "eval_support_pairs": (
            sum(len(task.get("train", [])) for task in eval_challenges.values())
            if include_support else 0
        ),
        "evaluation_tasks": len(eval_challenges),
        "evaluation_groups": {group: len(ids) for group, ids in eval_groups.items()},
    }
