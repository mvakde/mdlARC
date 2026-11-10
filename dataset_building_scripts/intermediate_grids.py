"""Explicit, resumable grid-only preparation using the sibling native DSL engine."""
import hashlib
import importlib.util
import json
import re
import random
import sqlite3
import time
import zipfile
from collections import Counter
from pathlib import Path

SELECTOR = "filtered-canvas-pool-v2"
SAMPLING = "ordered-stratified-per-visit-v1"
MEMBERS = {"challenges.json", "priv_challenges.json", "solutions.json", "dataset_manifest.json"}


def encoded(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pair_key(pair):
    return sha(encoded([pair["input"], pair["output"]]))


def recover_saved_originals(reader, catalog_path, catalog, banks, wanted, labels, pins):
    """Read full saved original runs behind catalog receipts, including ARC2 queries.

    The shared label catalog excludes official evaluation queries. The filtered
    ARC2 supplement is training data in this ARC1 experiment, so its existing
    original query programs are valid supervision. ARC1 held-out queries are
    never requested here. Canonical catalog contents remain unchanged.
    """
    owner = catalog_path.resolve().parents[2]
    receipts, archives, restored = {}, set(), []
    for bank_id in banks:
        bank = catalog["banks"][bank_id]
        archive = catalog["archives"][bank["archive"]] if "archive" in bank else catalog["archive"]
        identity = archive["sha256"]
        if identity in archives:
            continue
        archives.add(identity)
        files, archive_pins = reader.verified_archive(catalog_path, archive)
        pins.update(archive_pins)
        for name, raw in files.items():
            if not name.endswith("provenance.json"):
                continue
            provenance = json.loads(raw)
            if not isinstance(provenance, dict):
                continue
            for receipt in provenance.values():
                if not isinstance(receipt, dict):
                    continue
                relative = receipt.get("path", "")
                parts = Path(relative).parts
                if (relative.endswith("/runs/original/manifest.json") and len(parts) >= 4
                        and parts[-4] in wanted and receipt.get("manifest_sha256")):
                    receipts[relative] = receipt
    for relative, receipt in sorted(receipts.items()):
        manifest_path = owner / relative
        if not manifest_path.resolve().is_relative_to(owner.resolve()):
            raise ValueError("Original program receipt escapes its owner")
        manifest_raw = manifest_path.read_bytes()
        if sha(manifest_raw) != receipt["manifest_sha256"]:
            raise ValueError(f"Saved original manifest hash changed: {relative}")
        manifest = json.loads(manifest_raw)
        if manifest.get("status") != "verified" or manifest.get("puzzle_source") != "original":
            raise ValueError(f"Original receipt is not verified: {relative}")
        pins[str(manifest_path)] = sha(manifest_raw)
        label_path = manifest_path.parent / "primary-labels.unassigned.jsonl"
        raw = label_path.read_bytes()
        if sha(raw) != manifest["artifact_sha256"][label_path.name]:
            raise ValueError(f"Saved original label hash changed: {label_path}")
        pins[str(label_path)] = sha(raw)
        task = manifest_path.parts[-4]
        for line in raw.splitlines():
            row = json.loads(line)
            key = pair_key(row)
            if key not in wanted[task] or key in labels:
                continue
            if not row["task_id"].endswith("/" + task):
                raise ValueError("Saved original task identity mismatch")
            labels[key] = row["program"]
            restored.append({"task": task, "pair_id": row["pair_id"], "source": str(label_path)})
    return restored


def saved_repair_compilers(owner, wanted, pins):
    """Locate existing, release-pinned compilers for the remaining ARC2 originals."""
    result = {}
    for report_path in sorted((owner / "data/programs/releases").glob("*.json")):
        report_raw = report_path.read_bytes()
        report = json.loads(report_raw)
        for relative, expected in report.get("source_pins", {}).items():
            path = Path(relative)
            if path.suffix != ".py" or path.stem not in wanted:
                continue
            source = owner / path
            if not source.resolve().is_relative_to(owner.resolve()):
                raise ValueError("Repair compiler escapes its owner")
            if sha(source.read_bytes()) != expected:
                raise ValueError(f"Saved repair compiler hash changed: {relative}")
            pins[str(report_path)] = sha(report_raw)
            pins[str(source)] = expected
            result[path.stem] = source
    return result


def validate_identity(identity):
    if not isinstance(identity, dict) or set(identity) != {"archive_path", "sha256", "size"}:
        raise ValueError("prepared requires archive_path, sha256 and size")
    path = Path(identity["archive_path"])
    if (path.is_absolute() or ".." in path.parts or "\\" in str(path) or not path.parts
            or path.parts[0] != "assets_tmp" or path.suffix != ".zip"
            or type(identity["size"]) is not int or identity["size"] < 1
            or not isinstance(identity["sha256"], str)
            or not re.fullmatch("[0-9a-f]{64}", identity["sha256"])):
        raise ValueError("Invalid prepared dataset identity")


def candidate_grids(trace, input_grid, output_grid, *, counts=None):
    """Keep visible computation, excluding only demonstrably redundant states.

    Erasure, small crops, monochrome objects and partial constructions can all
    explain a rule. No size, entropy, changed-cell or output-distance threshold
    is used. A grid cycle is removed only when the *complete* executor state
    returns, so work retained in a clipboard, variable or selection survives.
    """
    counts = Counter() if counts is None else counts
    stack, states = [], {}
    for frame in trace:
        identity = sha(encoded(frame["state"]))
        if identity in states:
            stop = states[identity] + 1
            for previous in stack[stop:]:
                states.pop(previous[0])
            counts["reverted_or_noop_frames"] += len(stack) - stop + 1
            del stack[stop:]
        else:
            states[identity] = len(stack)
            stack.append((identity, frame))

    seen = {encoded(input_grid), encoded(output_grid)}
    candidates = []
    for _, frame in stack:
        grid = frame["state"]["grid"]
        key = encoded(grid)
        if key in seen:
            counts["duplicate_or_endpoint_frames"] += 1
            continue
        seen.add(key)
        # Native labels may include loop context and a variable binding, e.g.
        # "each 1/3 · patch = cut(...)". Match the actual call, not its prefix.
        call = re.search(r"([A-Za-z_]\w*)\s*\(", frame.get("label", "").lower())
        operation = call[1] if call else ""
        background = frame["state"].get("background", 0)
        blank = all(value == background for row in grid for value in row)
        if operation in {"canvas", "erase", "cut", "paint", "recolor", "fill"} and blank:
            counts["blank_reset_or_clear_frames"] += 1
            continue
        candidates.append(grid)
    counts["retained_candidates"] += len(candidates)
    return candidates


def build_prepared_dataset(identity, output_dir, *, root):
    """Consume a completed immutable asset; never execute programs at launch."""
    validate_identity(identity)
    archive = Path(root) / identity["archive_path"]
    if archive.is_symlink() or archive.stat().st_size != identity["size"] or sha(archive.read_bytes()) != identity["sha256"]:
        raise ValueError("Prepared dataset identity changed")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        if len(source.namelist()) != len(MEMBERS) or set(source.namelist()) != MEMBERS:
            raise ValueError("Unexpected prepared dataset members")
        data = {name: source.read(name) for name in MEMBERS}
    manifest = json.loads(data["dataset_manifest.json"])
    if manifest["training_sha256"] != sha(data["challenges.json"]):
        raise ValueError("Prepared training hash mismatch")
    for name, raw in data.items():
        path = output_dir / name
        if path.exists() and path.read_bytes() != raw:
            raise ValueError("Prepared run dataset cannot change in place")
        if not path.exists():
            pending = path.with_suffix(".pending")
            pending.write_bytes(raw)
            pending.replace(path)
    return manifest["summary"]


def reserve_diagnostic_pairs(identity, *, root, package, count=64, seed=42):
    """Reserve labeled evaluation demos globally by exact I/O, never queries.

    The original evaluation file remains byte-identical. A reserved pair is a
    normal leave-one-out demo target for the diagnostic, but occurs nowhere in
    pretraining, including any duplicate under a different task identity.
    """
    validate_identity(identity)
    root, package = Path(root), Path(package)
    source = root / identity["archive_path"]
    if source.stat().st_size != identity["size"] or sha(source.read_bytes()) != identity["sha256"]:
        raise ValueError("Parent prepared dataset identity changed")
    with zipfile.ZipFile(source) as archive:
        if len(archive.namelist()) != len(MEMBERS) or set(archive.namelist()) != MEMBERS:
            raise ValueError("Unexpected parent prepared dataset members")
        payloads = {name: archive.read(name) for name in MEMBERS}
    training, evaluation, manifest = (json.loads(payloads[name]) for name in
        ("challenges.json", "priv_challenges.json", "dataset_manifest.json"))
    if sha(payloads["challenges.json"]) != manifest["training_sha256"] or manifest.get("diagnostic_holdouts"):
        raise ValueError("Invalid or already-reserved parent dataset")
    eligible = {}
    for task, value in sorted(evaluation.items()):
        pairs = value["train"]
        if len(pairs) < 4 or not pairs[0].get("intermediate_grid_candidates", pairs[0].get("intermediate_grids")):
            continue
        if training.get(task) != {"train": pairs}:
            raise ValueError("Diagnostic parent must contain the exact evaluation demos")
        eligible.setdefault(pair_key(pairs[0]), task)
    if type(count) is not int or not 1 <= count <= len(eligible):
        raise ValueError(f"Request 1..{len(eligible)} distinct diagnostic pairs")
    selected = random.Random(seed).sample(sorted(eligible), count)
    holds = [{"task": eligible[key], "pair_index": 0, "io_sha256": key} for key in selected]
    removed = 0
    for task, value in list(training.items()):
        kept = [p for p in value["train"] if pair_key(p) not in selected]
        removed += len(value["train"]) - len(kept)
        if kept:
            value["train"] = kept
        else:
            del training[task]
    assert not {pair_key(p) for t in training.values() for p in t["train"]}.intersection(selected)
    manifest["source_preparation_summary"] = manifest["summary"]
    counts = Counter(min(2, len(p.get("intermediate_grid_candidates", p.get("intermediate_grids", []))))
                     for t in training.values() for p in t["train"])
    manifest.update(schema=3, prepared_parent=identity, diagnostic_holdouts=holds,
        diagnostic_seed=seed, summary={"training_tasks": len(training), "training_pairs": sum(counts.values()),
            "evaluation_tasks": len(evaluation), "evaluation_queries": sum(len(t["test"]) for t in evaluation.values()),
            "diagnostic_holdout_pairs": len(holds), "heldout_occurrences_removed": removed,
            **{f"verified_pairs_{n}_intermediates": counts[n] for n in range(3)}},
        protocol="pretraining on originals and non-reserved evaluation demos; diagnostic demo pairs withheld globally by exact I/O; official query inputs and answers excluded from pretraining; query answers scoring only")
    payloads["challenges.json"] = encoded(training)
    manifest["training_sha256"] = sha(payloads["challenges.json"])
    payloads["dataset_manifest.json"] = encoded(manifest)
    package.parent.mkdir(parents=True, exist_ok=True)
    pending = package.with_suffix(".pending.zip")
    with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(payloads.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, raw)
    if package.exists() and package.read_bytes() != pending.read_bytes():
        pending.unlink()
        raise ValueError("Diagnostic package exists with different bytes; use a new path")
    pending.replace(package)
    return {"archive_path": package.resolve().relative_to(root.resolve()).as_posix(),
            "size": package.stat().st_size, "sha256": sha(package.read_bytes())}


def prepare_originals(*, root, dsl_root, catalog_path, package, audit_dir, keep_unlabeled=False,
                      include_eval_demos=True):
    """ARC1 + filtered ARC2, optionally including ARC1 evaluation demos."""
    root, dsl_root, catalog_path = map(Path, (root, dsl_root, catalog_path))
    package, audit_dir = Path(package), Path(audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    # Import only the native bridge, avoiding sibling model/training imports.
    module_spec = importlib.util.spec_from_file_location("intermediate_native", dsl_root / "src/program_executor.py")
    native = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(native)
    owner = catalog_path.resolve().parents[2]
    reader_spec = importlib.util.spec_from_file_location("intermediate_catalog", owner / "tools/program_data.py")
    reader = importlib.util.module_from_spec(reader_spec)
    reader_spec.loader.exec_module(reader)
    catalog = json.loads(catalog_path.read_bytes())
    banks = [k for k in catalog["default_banks"] if catalog["banks"][k]["puzzle_source"] == "original"]
    labels, pins = {}, {}
    for source in reader.iter_sources(catalog_path, banks, pins=pins):
        for line in source["data"].splitlines():
            row = json.loads(line)
            labels[pair_key(row)] = row["program"]
    files = {"training": root / "assets_tmp/ARC-1/arc1_train_both.json",
             "filtered": root / "assets_tmp/ARC-2/arc2_filtered.json",
             "evaluation": root / "assets_tmp/ARC-1/arc1_eval_challenges.json",
             "solutions": root / "assets_tmp/ARC-1/arc1_eval_solutions.json"}
    contents = {k: p.read_bytes() for k, p in files.items()}
    tasks = {k: json.loads(v) for k, v in contents.items()}
    pins.update({str(p): sha(contents[k]) for k, p in files.items()})
    wanted = {}
    for task_id, task in tasks["filtered"].items():
        missing = {pair_key(p) for p in task["train"] + task.get("test", []) if pair_key(p) not in labels}
        if missing:
            wanted[task_id] = missing
    restored = recover_saved_originals(reader, catalog_path, catalog, banks, wanted, labels, pins)
    unresolved = {task: keys - labels.keys() for task, keys in wanted.items() if keys - labels.keys()}
    compiler_paths = saved_repair_compilers(owner, unresolved, pins)
    compilers, compiler_failures, compiled_labels = {}, [], []
    semantic = native.interpreter_hash()
    cache = sqlite3.connect(audit_dir / "execution.sqlite3")
    cache.execute("CREATE TABLE IF NOT EXISTS traces (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    counts, excluded, timing = Counter(), [], []
    training, evaluation, solutions = {}, {}, {}
    started = time.monotonic()
    with native.NativeExecutor() as executor:
        def annotate(pair, *, task_id, task, index, source):
            program = labels.get(pair_key(pair))
            if program is None and source == "filtered" and task_id in compiler_paths:
                if task_id not in compilers:
                    compiler_spec = importlib.util.spec_from_file_location("intermediate_repair_" + task_id, compiler_paths[task_id])
                    module = importlib.util.module_from_spec(compiler_spec)
                    compiler_spec.loader.exec_module(module)
                    compilers[task_id] = module
                demo_count = len(task["train"])
                role = "demo" if index < demo_count else "query"
                pair_id = f"{role}-{index if role == 'demo' else index - demo_count:03d}"
                try:
                    candidates = compilers[task_id].compile_pair(
                        seed=0, pair_id=pair_id, role=role, input_grid=pair["input"],
                        target_output=pair["output"], puzzle_context=task)
                    if len(candidates) != 1 or not isinstance(candidates[0].get("program"), str):
                        raise ValueError("Expected one canonical compiler program")
                    program = candidates[0]["program"]
                    compiled_labels.append({"task": task_id, "pair_id": pair_id,
                                            "source": str(compiler_paths[task_id])})
                except Exception as error:
                    reason = f"existing_compiler_failed: {type(error).__name__}: {error}"
                    compiler_failures.append({"task": task_id, "pair_id": pair_id, "reason": reason})
                    return None, reason
            if program is None:
                return None, "no_exact_program_after_catalog_and_saved_original_lookup"
            key = sha(encoded([semantic, SELECTOR, pair["input"], program]))
            hit = cache.execute("SELECT value FROM traces WHERE key=?", (key,)).fetchone()
            counts["cache_hits" if hit else "executions"] += 1
            before = time.monotonic()
            if hit:
                value = json.loads(hit[0])
            else:
                result = executor.execute(program, pair["input"], trace=True)
                if result.get("error"):
                    value = {"error": result["error"]}
                else:
                    filtered = Counter()
                    grids = candidate_grids(result["trace"], pair["input"], result["grid"], counts=filtered)
                    value = {"output_sha256": sha(encoded(result["grid"])),
                             "grids": grids, "filter_counts": dict(filtered)}
                cache.execute("INSERT OR REPLACE INTO traces VALUES (?,?)", (key, encoded(value).decode()))
                if counts["executions"] % 100 == 0:
                    cache.commit()
            if len(timing) < 100 and not hit:
                timing.append(time.monotonic() - before)
            if value.get("error"):
                return None, value["error"]
            if value["output_sha256"] != sha(encoded(pair["output"])):
                return None, "execution_output_mismatch"
            grids = value["grids"]
            counts.update(value["filter_counts"])
            counts[f"verified_pairs_{min(2, len(grids))}_intermediates"] += 1
            counts["pairs_with_resampling_choices"] += len(grids) > 2
            return {"input": pair["input"], "output": pair["output"],
                    "intermediate_grid_candidates": grids}, None

        for source in ("training", "filtered", "evaluation"):
            for task_id, task in sorted(tasks[source].items()):
                pairs = list(task["train"])
                if source != "evaluation":
                    pairs += task.get("test", [])
                annotated, errors = [], []
                for index, pair in enumerate(pairs):
                    changed, error = annotate(pair, task_id=task_id, task=task, index=index, source=source)
                    if error:
                        errors.append({"pair": index, "reason": error})
                    if changed is not None or keep_unlabeled:
                        annotated.append(changed if changed is not None else {"input": pair["input"], "output": pair["output"]})
                if errors:
                    excluded.append({"source": source, "task": task_id, "errors": errors})
                    if not annotated:
                        continue
                if source == "evaluation":
                    evaluation[task_id] = {"train": annotated, "test": [{"input": p["input"]} for p in task["test"]]}
                    solutions[task_id] = tasks["solutions"][task_id]
                    if include_eval_demos:
                        if task_id in training:
                            raise ValueError("Evaluation task collided with a training source")
                        training[task_id] = {"train": annotated}
                else:
                    # Match the existing ARC1 + filtered builder: filtered
                    # originals replace matching ARC1 task IDs.
                    training[task_id] = {"train": annotated}
                counts[source + "_tasks"] += 1
                counts[source + "_pairs"] += len(annotated)
            cache.commit()
            print(json.dumps({"completed": source, "counts": dict(counts)}, separators=(",", ":")), flush=True)
    elapsed = time.monotonic() - started
    cache.close()
    train_raw = encoded(training)
    summary = {**{k: v for k, v in counts.items() if k not in {"executions", "cache_hits"}}, "training_tasks": len(training),
               "training_pairs": sum(len(v["train"]) for v in training.values()),
               "evaluation_tasks": len(evaluation),
               "excluded_tasks": sum(not any(t["task"] in group for group in (training, evaluation)) for t in excluded),
               "tasks_with_excluded_pairs": len(excluded), "excluded_pairs": sum(len(t["errors"]) for t in excluded)}
    overlap = set(training) & set(evaluation)
    if overlap and not include_eval_demos:
        raise ValueError("Evaluation task appeared in pretraining")
    for task_id in overlap:
        if training[task_id] != {"train": evaluation[task_id]["train"]}:
            raise ValueError("Only evaluation demos may appear in pretraining")
    manifest = {"kind": "intermediate_grid_originals", "schema": 2, "selector": SELECTOR,
                "sampling": SAMPLING,
                "sources": ["arc1_training", "arc2_filtered"] + (["arc1_evaluation_demos"] if include_eval_demos else []),
                "evaluation_source": "arc1_evaluation", "include_eval_demos": include_eval_demos,
                "training_sha256": sha(train_raw), "native_semantics_sha256": semantic,
                "source_pins": pins, "summary": summary,
                "label_recovery": {"saved_original_pairs": len(restored),
                                   "existing_compiler_pairs": len(compiled_labels)},
                "protocol": "pretraining on solved originals and evaluation demos; two demo grid chains; target input to output; no target intermediates; evaluation query inputs and answers excluded from training; query answers scoring only"}
    payloads = {"challenges.json": train_raw, "priv_challenges.json": encoded(evaluation),
                "solutions.json": encoded(solutions), "dataset_manifest.json": encoded(manifest)}
    package.parent.mkdir(parents=True, exist_ok=True)
    pending = package.with_suffix(".pending.zip")
    with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(payloads.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, raw)
    if package.exists() and package.read_bytes() != pending.read_bytes():
        pending.unlink()
        raise ValueError("Prepared package exists with different bytes; choose a new path")
    pending.replace(package)
    identity = {"archive_path": package.resolve().relative_to(root.resolve()).as_posix(),
                "size": package.stat().st_size, "sha256": sha(package.read_bytes())}
    audit = {"summary": summary, "excluded": excluded, "package": identity,
             "restored_original_programs": restored, "existing_compiler_programs": compiled_labels,
             "existing_compiler_failures": compiler_failures,
             "executions": counts["executions"], "cache_hits": counts["cache_hits"],
             "elapsed_seconds": elapsed, "sample_100_execution_seconds": sum(timing),
             "retained_bytes": identity["size"] + (audit_dir / "execution.sqlite3").stat().st_size,
             "peak_package_staging_bytes": sum(map(len, payloads.values())) + identity["size"]}
    (audit_dir / "preparation.json").write_bytes(encoded(audit))
    print(json.dumps({k: v for k, v in audit.items() if k not in {
        "excluded", "restored_original_programs", "existing_compiler_programs", "existing_compiler_failures"}}, separators=(",", ":")), flush=True)
    return identity
