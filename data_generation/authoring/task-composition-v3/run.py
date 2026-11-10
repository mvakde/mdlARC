#!/usr/bin/env python3
"""Compose unique task pairs from a dataset with two model stages and bounded workers."""
from __future__ import annotations
import argparse
import copy
from contextlib import contextmanager
from collections import deque
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "task-variations/src"))
from v2_storage import copy_file, copy_tree, hashes, puzzle_text, read, save_puzzle, sha, write
import pair_policy
RESULTS = HERE.parents[1] / "results" / "task-composition-v3"
PARENTS = RESULTS.parent / "task-variations"
EXAMPLES = HERE.parent / "task-variations/examples/review"
PARENT_FILES = ("task.json", "summary.xml", "generator_v1.py", "review.md",
                "generator_v2.py", "review_coverage.md", "provenance.json")
STAGES = ("author", "review-improve")


def implementation_hashes():
    paths = list((HERE / "prompts").glob("*.md")) + [HERE / n for n in
            ("run.py", "pair_policy.py", "v2_storage.py", "verify.py", "parent_sample.py", "inspect_puzzles.py", "finish_review.py")]
    return {p.relative_to(HERE).as_posix(): sha(p) for p in sorted(paths)}


def choose_examples(parent_ids, seed, example_root=EXAMPLES):
    available = read(example_root / "manifest.json")["example_ids"]
    eligible = sorted(set(available) - set(parent_ids))
    if len(eligible) < 4:
        raise ValueError("Need four review examples distinct from both parents")
    return random.Random(seed).sample(eligible, 4)


def stage_examples(ids, destination, example_root=EXAMPLES):
    pinned = read(example_root / "manifest.json")
    for task_id in ids:
        for name, digest in pinned["files"].items():
            if name.split("/")[0] != task_id:
                continue
            source = example_root / name
            source.resolve().relative_to(example_root.resolve())
            if source.is_symlink() or sha(source) != digest:
                raise ValueError("Pinned review example changed: " + name)
            copy_file(source, destination / name)
            if source.name == "task.json" or source.parent.name in ("samples_before", "samples_after"):
                target = destination / name
                target.with_suffix(".txt").write_text(puzzle_text(read(target), name))
    (destination / "README.md").write_text(
        "Study these FOUR worked human reviews: " + ", ".join(ids) + ".\n"
        "Read task.txt, summary, generator_before.py, manual_review.md, generator_after.py, "
        "and before/after sample .txt files. They are review-style examples, not parents.\n")
    write(destination / "manifest.json", {"example_ids": ids, "files": hashes(destination)})


def sample_parent(source, destination, task_id, fallback_seed):
    result = subprocess.run([sys.executable, str(HERE / "parent_sample.py"),
        "--generator", str(destination / "generator_v2.py"), "--task-id", task_id,
        "--fallback-seed", str(fallback_seed), "--output", str(destination / "variation.json"),
        "--selection", str(destination / "variation_selection.json")],
        capture_output=True, text=True, timeout=300)
    if result.returncode:
        raise RuntimeError(f"Parent sampling failed for {task_id}: {result.stderr[-4000:]}")
    selection = read(destination / "variation_selection.json")
    selection["source_file_hashes"] = {name: sha(source / name) for name in PARENT_FILES}
    write(destination / "variation_selection.json", selection)
    (destination / "variation_selection.md").write_text(
        f"Seed {selection['seed']}, train/test counts 4/1.\n"
        f"Selection: {selection['method']}. {selection['reason']}.\n"
        f"Observed mode: {json.dumps(selection['observed_mode'], separators=(",", ":"))}\n")
    return selection


def pair_indices(rank, size):
    """Decode a lexicographic unordered pair rank without building N**2 pairs."""
    if not 0 <= rank < size * (size - 1) // 2:
        raise ValueError("Pair rank out of range")
    low, high = 0, size - 1
    while low + 1 < high:
        middle = (low + high) // 2
        start = middle * (2 * size - middle - 1) // 2
        if start <= rank:
            low = middle
        else:
            high = middle
    start = low * (2 * size - low - 1) // 2
    return low, low + 1 + rank - start


def dataset_pair_schedule(collection, dataset, candidates):
    """Persist dataset-defined ordering; count/workers/run seed cannot alter it."""
    ids = [p.name for p in candidates]
    identity = json.dumps({"algorithm": "round-robin-v1", "dataset": dataset, "task_ids": ids},
                          separators=(",", ":"))
    fingerprint = hashlib.sha256(identity.encode()).hexdigest()
    order = sorted(range(len(ids)), key=lambda i: hashlib.sha256(
        f"{fingerprint}:{ids[i]}".encode()).digest())
    schedule = {"algorithm": "round-robin-v1", "dataset": dataset, "task_ids": ids,
                "task_order": order, "fingerprint": fingerprint}
    marker = collection / ".composition-v3-pair-order.json"
    with (collection / ".composition-v3-pair-order.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another v3 prepare/run owns this dataset; resume it before starting another') from None
        if marker.exists():
            if read(marker) != schedule:
                raise ValueError("Dataset task membership changed since its pair order was saved")
        else:
            write(marker, schedule)
    return schedule


def round_robin_pair(index, size, order):
    """Every even-size prefix has task participation counts differing by at most 1.

    Odd-size collections use a rotating bye and differ by at most 2.
    Full traversal visits every unordered pair once; no pair list is materialized.
    """
    if not 0 <= index < size * (size - 1) // 2:
        raise ValueError("All unique pairs in this dataset have been reserved")
    ring_size = size + size % 2
    width = size // 2
    round_number, slot = divmod(index, width)
    # Rotate the within-round starting pair, retaining disjoint matching prefixes.
    slot = (slot + round_number) % width
    if size % 2:
        dummy_position = 1 + ((ring_size - 2 + round_number) % (ring_size - 1))
        skipped = min(dummy_position, ring_size - 1 - dummy_position)
        if slot >= skipped:
            slot += 1
    def at(position):
        return 0 if position == 0 else 1 + (position - 1 - round_number) % (ring_size - 1)
    a, b = at(slot), at(ring_size - 1 - slot)
    return tuple(sorted((order[a], order[b])))


def requested_stages(args_or_plan):
    if isinstance(args_or_plan, dict):
        return tuple(args_or_plan.get("run_stages", STAGES))
    return ("author",) if getattr(args_or_plan, "stage", "all") == "author" else STAGES


def stages_done(root, job, stages):
    return all(finished_stage(root / job["name"], stage) for stage in stages)


def check_options(args, plan):
    for key in ("model", "effort", "seed", "review_samples"):
        supplied = getattr(args, key)
        if supplied is not None and supplied != plan[key]:
            raise ValueError(f"Existing plan has {key}={plan[key]}; use a new --output-root")
    if args.dataset is not None and Path(args.dataset).as_posix() != plan["dataset"]:
        raise ValueError("Existing plan uses another dataset; use a new --output-root")
    if args.parents_root is not None and args.parents_root.resolve() != Path(plan["parents_root"]):
        raise ValueError("Existing plan uses another parents root; use a new --output-root")
    if implementation_hashes() != plan["implementation_hashes"]:
        raise ValueError("Pipeline changed since preparation; use a new --output-root")


def prepare(args, *, source_plan=None, source_root=None):
    root = args.output_root.resolve()
    publication = root / "llm-inputs/.pending-prepare/plan.json"
    if publication.exists():
        pending_plan = read(publication)
        check_options(args, pending_plan)
        publish_plan(root, pending_plan)
    existing = (root / "plan.json").exists()
    if existing:
        plan = read(root / "plan.json")
        check_options(args, plan)
        if getattr(args, "batch_size", None) is not None:
            count = len(plan["jobs"])
            if all(stages_done(root, job, requested_stages(args)) for job in plan["jobs"]):
                count += args.batch_size
        else:
            count = args.count if args.count is not None else len(plan["jobs"])
        if count < len(plan["jobs"]):
            raise ValueError("--count is the total target; cannot shrink an existing plan")
        if count == len(plan["jobs"]):
            with pair_policy.dataset_lock(plan):
                ledger = pair_policy.load_ledger(plan)
                pair_policy.check_plan(ledger, root, plan)
                pair_policy.check_single_queue(ledger, root)
            return plan
    elif source_plan is not None:
        check_options(args, source_plan)
        count = args.count
        plan = copy.deepcopy(source_plan)
        plan.update(jobs=[], source_run=str(source_root),
                    pair_capacity=len(plan['candidates']) * (len(plan['candidates']) - 1) // 2)
        for key in ('pair_range', 'fixed_batch', 'pair_selection', 'run_stages'):
            plan.pop(key, None)
    else:
        count = args.count or getattr(args, "batch_size", None)
        if count is None:
            raise ValueError("Supply --count or --batch-size for a new run")
        dataset = Path(args.dataset or "arc2/evaluation").as_posix()
        if Path(dataset).is_absolute() or ".." in Path(dataset).parts:
            raise ValueError("--dataset must be a relative folder within --parents-root")
        base = (args.parents_root or PARENTS).resolve()
        collection = (base / dataset).resolve()
        collection.relative_to(base)
        candidates = sorted(p for p in collection.glob("*") if p.is_dir()
                            and not p.is_symlink()
                            and all((p / name).is_file() and not (p / name).is_symlink()
                                    for name in PARENT_FILES))
        if len(candidates) < 2:
            raise ValueError(f"Need at least two complete task-variation parents under {collection}")
        capacity = len(candidates) * (len(candidates) - 1) // 2
        schedule = dataset_pair_schedule(collection, dataset, candidates)
        plan = {"schema_version": 3, "seed": args.seed if args.seed is not None else int(schedule["fingerprint"][:16], 16),
                "dataset": dataset, "source": dataset, "parents_root": str(base),
                "model": args.model or "gpt-6.1-sol", "effort": args.effort or "xhigh",
                "review_samples": args.review_samples or 5, "python": sys.executable,
                "implementation_hashes": implementation_hashes(), "jobs": [],
                "candidates": [{"id": p.name, "path": str(p),
                                "hashes": {name: sha(p / name) for name in PARENT_FILES}}
                               for p in candidates],
                "pair_capacity": capacity, "pair_schedule": schedule}
    if count > plan["pair_capacity"]:
        raise ValueError(f"Requested {count} families but dataset has only {plan['pair_capacity']} unique pairs")
    records = pair_policy.allocate(root, plan, count)
    plan['pair_policy'] = pair_policy.POLICY
    pending = root / "llm-inputs/.pending-prepare"
    if pending.exists():
        shutil.rmtree(pending)
    pending.mkdir(parents=True)
    evidence = {}
    for evidence_root, evidence_plan in ((root, plan), (source_root, source_plan)):
        if evidence_plan is not None:
            for job in evidence_plan['jobs']:
                for label, task_id in zip(('A', 'B'), job['parents']):
                    evidence.setdefault(task_id, (Path(evidence_root) / job['name'] / 'parents' / label, job, label))
    while len(plan["jobs"]) < count:
        index = len(plan["jobs"])
        record = records[index]
        ids = [p['id'] for p in plan['candidates']]
        a, b = sorted(ids.index(t) for t in record['parents'])
        rank = a * (2 * len(ids) - a - 1) // 2 + b - a - 1
        pair = [plan['candidates'][i] for i in (a, b)]
        name = record['name']
        rng = random.Random(f"{plan['seed']}:job:{name}")
        print(f"Prepare {index + 1}/{count}: {name}", flush=True)
        folder = pending / name
        selections = {}
        for label, parent in zip(("A", "B"), pair):
            source = Path(parent["path"])
            destination = folder / "parents" / label
            if parent['id'] in evidence:
                snapshot, old_job, old_label = evidence[parent['id']]
                prefix = f'parents/{old_label}/'
                for relative, digest in old_job['input_hashes'].items():
                    if relative.startswith(prefix) and sha(snapshot / relative[len(prefix):]) != digest:
                        raise ValueError(f'Prepared parent evidence changed: {relative}')
                copy_tree(snapshot, destination)
                selections[label] = read(destination / 'variation_selection.json')
                continue
            for filename, digest in parent["hashes"].items():
                if sha(source / filename) != digest:
                    raise ValueError(f"Pinned parent changed: {source / filename}")
                copy_file(source / filename, destination / filename)
            (destination / "task.txt").write_text(puzzle_text(read(destination / "task.json"), parent["id"]))
            selections[label] = sample_parent(source, destination, parent["id"], rng.getrandbits(64))
        ids = choose_examples([p["id"] for p in pair], rng.getrandbits(64))
        stage_examples(ids, folder / "llm-inputs/evidence/examples")
        checks = {stage: [0] + rng.sample(range(1000, 10**9), 41) for stage in STAGES}
        plan["jobs"].append({"name": name, "parents": [p["id"] for p in pair], "pair_rank": rank,
            'usage_before': record['counts_before'],
            "review_examples": ids, "review_sample_count": plan["review_samples"],
            "parent_samples": selections, "check_seeds": checks,
            "input_hashes": hashes(folder)})
    write(pending / "plan.json", plan)
    return publish_plan(root, plan)


def publish_plan(root, plan):
    """Recover publication/extension atomically without allocating new pairs."""
    pending = root / "llm-inputs/.pending-prepare"
    for job in plan["jobs"]:
        target = root / job["name"]
        if target.exists():
            if (pending / job["name"]).exists():
                raise FileExistsError(f"Unplanned job folder exists: {target}")
            for name, digest in job["input_hashes"].items():
                if not (target / name).is_file() or sha(target / name) != digest:
                    raise ValueError(f"Partially published evidence changed: {job['name']}/{name}")
        else:
            (pending / job["name"]).rename(target)
    write(root / "plan.json", plan)
    shutil.rmtree(pending)
    return plan


@contextmanager
def run_lock(root):
    """Prevent duplicate pair reservations/model calls from concurrent invocations."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another prepare/run owns this output root; stop it before resuming") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def workspace(path):
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "scratch/tmp").mkdir(parents=True, exist_ok=True)
    (path / "AGENTS.md").write_text(
        "Work only in this directory. Evidence and instructions are read-only. "
        "Use scratch/ for experiments. Do not inspect sibling or ancestor directories, "
        "use the network, or delegate to other agents. Follow instructions.md.\n"
        'All JSON and JSONL you create, including generated code, must be compact with no indentation. '
        'Use json.dumps(value, separators=(",", ":")) in Python and JSON.stringify(value) in JavaScript.\n'
        'Bound disk use: estimate bytes, file counts and peak temporary space across tasks, seeds, pairs, steps, workers, retries and retained runs. '
        'Keep scratch diagnostics and logs bounded; avoid duplicate reconstructible payloads. Preserve required evidence and hashes. '
        'Use the existing lossless trace codec for saved execution traces, keeping initial and exceptional before states. Generated code follows these rules too. '
        'Before storage optimization edits, report measured projected net savings, runtime and peak disk cost, including journals, indexes and backups.\n')


class TaskClients:
    """Own a separate permission-scoped Codex connection for each workspace."""
    def __enter__(self):
        from contextlib import ExitStack
        self.stack = ExitStack()
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def thread_start(self, **options):
        from openai_codex import Codex, CodexConfig
        from task_variations.common.isolation import overrides, codex_binary
        folder = Path(options["cwd"])
        connection = self.stack.enter_context(Codex(CodexConfig(
            codex_bin=codex_binary(), config_overrides=overrides(folder), cwd=str(folder))))
        return connection.thread_start(**options)


def invoke(client, folder, model, effort):
    from openai_codex import ApprovalMode
    thread = client.thread_start(
        model=model, config={"model_reasoning_effort": effort}, cwd=str(folder),
        approval_mode=ApprovalMode.deny_all, ephemeral=True,
        developer_instructions="Work only in the current directory; follow AGENTS.md and instructions.md.",
    )
    return thread


def turn(thread, folder, prompt, label):
    print(f"  {label}", flush=True)
    (folder / f"{label}-prompt.md").write_text(prompt)
    result = thread.run(prompt)
    (folder / f"{label}-response.md").write_text(result.final_response or "")
    write(folder / f"{label}-usage.json", {"thread_id": thread.id,
                                          "usage": getattr(result, "usage", None)})



def verify(folder, report, seeds, samples):
    report.unlink(missing_ok=True)
    command = [sys.executable, str(HERE / "verify.py"), "--generator", str(folder / "generator.py"),
        "--parents", str(folder / "parents"), "--report", str(report),
        "--samples", str(samples), "--seeds", *map(str, seeds)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        if not report.exists():
            write(report, {"valid": False, "errors": [result.stderr[-4000:] or result.stdout[-4000:] or
                                                       f"Verifier exited {result.returncode}"]})
    except subprocess.TimeoutExpired:
        write(report, {"valid": False, "errors": ["Verification exceeded 300 seconds"]})
    return read(report)


def setup_stage(job_root, stage, job):
    package = job_root / "llm-inputs" / stage
    folder = package / "workspace"
    if folder.exists() and (package / "protected.json").exists():
        return folder, read(package / "protected.json")
    if folder.exists():
        if list(folder.glob("turn-*-prompt.md")):
            raise ValueError(f"Missing protected-input manifest for started stage: {stage}")
        shutil.rmtree(folder)
    if (package / "inputs").exists():
        shutil.rmtree(package / "inputs")
    workspace(folder)
    copy_tree(job_root / "parents", folder / "parents")
    if job.get("proposal_sha256"):
        if sha(job_root / "proposal.json") != job["proposal_sha256"]:
            raise ValueError("Pinned composition proposal changed")
        # Copy exact bytes; the protected-input manifest pins this specification.
        shutil.copyfile(job_root / "proposal.json", folder / "proposal.json")
    for filename in ("v2_storage.py", "verify.py"):
        copy_file(HERE / filename, folder / filename)
    copy_file(HERE / "inspect_puzzles.py", folder / "tools/inspect_puzzles.py")
    editable = set()
    if stage == "review-improve":
        for name in ("generator_v1.py", "composition.md", "validation_v1.json"):
            copy_file(job_root / name, folder / name)
        copy_file(job_root / "task_v1.json", folder / "task.json")
        (folder / "task.txt").write_text(puzzle_text(read(folder / "task.json"), "initial composed seed 0"))
        source = job_root / "llm-inputs/author/checked-samples"
        seeds = job["check_seeds"]["author"]
        count = job["review_sample_count"]
        # Evenly spaced across the recorded list, including canonical seed zero.
        positions = [i * (len(seeds) - 1) // max(1, count - 1) for i in range(count)]
        for position in positions:
            name = f"seed-{seeds[position]:010d}.json"
            copy_file(source / name, folder / "samples" / name)
            copy_file((source / name).with_suffix(".txt"), (folder / "samples" / name).with_suffix(".txt"))
        copy_file(job_root / "task_spec_v1.md", folder / "source_spec.md")
        copy_tree(job_root / "llm-inputs/evidence/examples", folder / "examples")
        copy_file(HERE / "finish_review.py", folder / "finish_review.py")
        # Improved code/spec do not exist until finish_review.py checkpoints the review.
    instructions = (HERE / "prompts" / f"{stage}.md").read_text()
    if job.get("proposal_sha256"):
        instructions += ("\nThis job has a REQUIRED saved composition in proposal.json. "
            "Its idea and parent A/B order define the rule you must implement. "
            "This requirement overrides the generic invitation to invent a different rule. "
            "Preserve the specified causal dependency and parent contributions. "
            "Resolve missing generation details consistently; do not replace the idea. "
            "The audit is historical context, including rejections, not permission to change the rule. "
            "Record any unsupported or contradictory requirement honestly in composition.md.\n")
    instructions += f"\nUse this Python executable: {sys.executable}\n"
    instructions += "Host verification seeds: " + ", ".join(map(str, job["check_seeds"].get(stage, []))) + "\n"
    (folder / "instructions.md").write_text(instructions)
    write(folder / "environment.json", {"python_executable": sys.executable})
    baseline = hashes(folder)
    protected = {name: digest for name, digest in baseline.items() if name not in editable}
    # Save the exact starting package, including editable v1 copies, before the first call.
    copy_tree(folder, package / "inputs")
    write(package / "protected.json", protected)
    return folder, protected


def integrity(folder, protected):
    return [f"Protected evidence changed: {name}" for name, digest in protected.items()
            if (folder / name).is_symlink() or not (folder / name).is_file() or sha(folder / name) != digest]


def checkpoint(job_root, stage, names):
    path = job_root / "checkpoints.json"
    current = read(path) if path.exists() else {}
    current[stage] = {name: sha(job_root / name) for name in names}
    write(path, current)


def finished_stage(job_root, stage):
    path = job_root / "checkpoints.json"
    stages = read(path) if path.exists() else {}
    if stage not in stages:
        return False
    for name, digest in stages[stage].items():
        if not (job_root / name).is_file() or sha(job_root / name) != digest:
            raise ValueError(f"Checkpoint artifact changed: {stage}/{name}")
    return True


def review_integrity(folder, package, *, pin=True):
    """Check the phase barrier and freeze it host-side before verification/repairs."""
    record_path = folder / "review-complete.json"
    review = folder / "review.md"
    checkpoint_path = package / "review-checkpoint.json"
    if not record_path.is_file() or not review.is_file():
        return ["Finish review.md and run finish_review.py before creating the improved generator"]
    try:
        record = read(record_path)
        if record["review_sha256"] != sha(review):
            return ["Completed review changed after improvement began"]
        if record["source_generator_sha256"] != sha(folder / "generator_v1.py"):
            return ["Review checkpoint source differs from protected v1"]
        if not isinstance(record["completed_at_ns"], int):
            return ["Invalid review completion time"]
        for name in ("generator.py", "task_spec.md"):
            if (folder / name).exists() and (folder / name).stat().st_mtime_ns < record["completed_at_ns"]:
                return ["Improved deliverables must be created after the review checkpoint"]
        current = {"review.md": sha(review), "review-complete.json": sha(record_path)}
        if checkpoint_path.exists():
            if read(checkpoint_path) != current:
                return ["Host-pinned review/checkpoint changed"]
        elif pin:
            write(checkpoint_path, current)
        return []
    except (ValueError, KeyError, OSError, TypeError):
        return ["Invalid review completion checkpoint"]


def run_stage(client, job_root, stage, job, plan):
    folder, protected = setup_stage(job_root, stage, job)
    required = ("generator.py", "task_spec.md")
    required += ("composition.md",) if stage == "author" else ("review.md", "review_coverage.md")
    errors = integrity(folder, protected)
    if stage == "review-improve" and ((folder / "review-complete.json").exists()
            or (job_root / "llm-inputs" / stage / "review-checkpoint.json").exists()):
        errors += review_integrity(folder, job_root / "llm-inputs" / stage)
    if errors:
        raise ValueError("; ".join(errors))
    thread = invoke(client, folder, plan["model"], plan["effort"])
    prompt = (folder / "instructions.md").read_text()
    previous = max((int(p.stem.split("-")[-2]) for p in folder.glob("turn-*-prompt.md")), default=0)
    if previous:
        prompt += "\nResuming this stage. Inspect current deliverables and feedback.json, if present, " \
                  "before continuing; prior attempt files are retained.\n"
    attempts = 3
    for number in range(previous + 1, previous + attempts + 1):
        turn(thread, folder, prompt, f"turn-{number}")
        errors = integrity(folder, protected)
        errors += [f"Missing or empty deliverable: {name}" for name in required
                   if not (folder / name).is_file() or (folder / name).is_symlink()
                   or not (folder / name).read_text().strip()]
        if stage == "review-improve":
            errors.extend(review_integrity(folder, job_root / "llm-inputs" / stage))
        feedback = {"valid": not errors, "errors": errors}
        samples = job_root / "llm-inputs" / stage / "checked-samples"
        if feedback["valid"]:
            if samples.exists():
                shutil.rmtree(samples)
            feedback = verify(folder, folder / f"verification-{number}.json", job["check_seeds"][stage], samples)
            errors = integrity(folder, protected)
            if stage == "review-improve":
                errors += review_integrity(folder, job_root / "llm-inputs" / stage)
            if errors:
                feedback["valid"] = False
                feedback["errors"].extend(errors)
        write(folder / "feedback.json", feedback)
        archive = job_root / "saved-outputs" / stage / f"attempt-{number}"
        for name in (*required, "feedback.json"):
            if (folder / name).is_file():
                copy_file(folder / name, archive / name)
        if feedback["valid"]:
            if stage == "author":
                exports = {"generator.py": "generator_v1.py", "task_spec.md": "task_spec_v1.md",
                           "composition.md": "composition.md"}
                for source, target in exports.items():
                    copy_file(folder / source, job_root / target)
                copy_file(samples / "seed-0000000000.json", job_root / "task_v1.json")
                write(job_root / "validation_v1.json", feedback)
                sample_names = tuple(p.relative_to(job_root).as_posix() for p in sorted(samples.glob("*")))
                checkpoint(job_root, stage, (*exports.values(), "task_v1.json", "validation_v1.json", *sample_names))
            else:
                for source, target in (("generator.py", "generator_v2.py"),
                    ("generator.py", "generator.py"), ("task_spec.md", "task_spec.md"),
                    ("review_coverage.md", "review_coverage.md"), ("review.md", "review.md"),
                    ("review-complete.json", "review-complete.json")):
                    copy_file(folder / source, job_root / target)
                write(job_root / "validation.json", feedback)
                # Publish only final v2 samples. V1/history remains under excluded folders.
                pending_samples = job_root / "llm-inputs/review-improve/final-samples"
                if pending_samples.exists():
                    shutil.rmtree(pending_samples)
                pending_samples.mkdir()
                for seed in job["check_seeds"][stage][:30]:
                    name = f"seed-{seed:010d}.json"
                    copy_file(samples / name, pending_samples / name)
                    copy_file((samples / name).with_suffix(".txt"), (pending_samples / name).with_suffix(".txt"))
                final = job_root / "samples"
                if final.exists():
                    shutil.rmtree(final)
                pending_samples.rename(final)
                copy_file(final / "seed-0000000000.json", job_root / "task.json")
                sample_names = tuple(p.relative_to(job_root).as_posix() for p in sorted(final.glob("*")))
                checkpoint(job_root, stage, ("generator_v2.py", "generator.py", "task_spec.md",
                                            "review.md", "review-complete.json", "review_coverage.md", "validation.json", "task.json", *sample_names))
            return True
        # Protected-input changes are not repairable by altering the evidence further.
        unrepairable = integrity(folder, protected)
        if stage == "review-improve" and (folder / "review-complete.json").exists():
            unrepairable += review_integrity(folder, job_root / "llm-inputs" / stage, pin=False)
        if unrepairable:
            raise ValueError("; ".join(unrepairable))
        prompt = ("Read feedback.json and repair only the concrete execution/check failures. "
                  "Preserve the intended composed rule, update task_spec.md as needed, "
                  "and rerun local verification. Keep protected evidence unchanged.")
        if stage == "review-improve":
            prompt += (" If the review checkpoint is missing, finish review.md before improvement; "
                       "remove premature improvement deliverables, then run finish_review.py. "
                       "Otherwise preserve the completed review/checkpoint and implement its requests.")
    write(job_root / "status.json", {"state": "failed", "stage": stage, "feedback": feedback})
    return False


def run_job(client, root, job, plan):
    job_root = root / job["name"]
    status_path = job_root / "status.json"
    if status_path.exists() and read(status_path)["state"] == "completed":
        for stage in STAGES:
            if not finished_stage(job_root, stage):
                raise ValueError(f"Completed job missing checkpoint: {job['name']}/{stage}")
        print(f"Skip completed {job['name']}", flush=True)
        return
    for stage in requested_stages(plan):
        if finished_stage(job_root, stage):
            continue
        write(status_path, {"state": "running", "stage": stage})
        print(f"{job['name']}: {stage}", flush=True)
        if not run_stage(client, job_root, stage, job, plan):
            return
    if requested_stages(plan) == ("author",):
        write(status_path, {"state": "author_completed", "stage": "author", "parents": job["parents"],
                           "dataset": plan["dataset"], "check_count": 42,
                           "note": "Author stage and local checks passed; review/improvement not run."})
        print(f"  author completed: {job_root}", flush=True)
        return
    write(job_root / "provenance.json", {"parents": job["parents"], "parent_samples": job["parent_samples"],
        "dataset": plan["dataset"], "pair_rank": job["pair_rank"],
        "review_examples": job["review_examples"], "model": plan["model"], "effort": plan["effort"],
        "generator_v1_sha256": sha(job_root / "generator_v1.py"),
        "generator_v2_sha256": sha(job_root / "generator_v2.py"),
        "semantic_verification": "generator-owned checks only; no independent solver or blind solve"})
    write(status_path, {"state": "completed", "stage": "review-improve", "parents": job["parents"], "dataset": plan["dataset"],
        "samples": "samples", "sample_count": 30, "check_count": 42, "review_sample_count": job["review_sample_count"],
        "note": "Local checks passed; completion does not assert independent semantic approval."})
    print(f"  completed: {job_root}", flush=True)


def _interrupt_worker(signum, frame):
    raise KeyboardInterrupt


def _mark_interrupted(root, name, error):
    path = root / name / "status.json"
    if path.exists() and read(path)["state"] in ("completed", "author_completed", "failed"):
        return
    write(path, {**(read(path) if path.exists() else {}), "state": "interrupted", "error": str(error) or "Run cancelled"})


def _run_worker(root, job, plan, runner=None):
    # Each worker owns its Codex connections and subprocess group. Terminal
    # Ctrl+C is handled by the coordinator, which stops every active group.
    if os.name == "posix":
        os.setsid()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, _interrupt_worker)
    try:
        if runner is not None:
            runner(root, job, plan)
        else:
            with TaskClients() as client:
                run_job(client, root, job, plan)
    except BaseException as error:
        _mark_interrupted(root, job["name"], error)
        if isinstance(error, KeyboardInterrupt):
            raise SystemExit(130)
        raise


def _signal_worker(process, signum):
    if process.pid is None:
        return
    if os.name == "posix":
        try:
            # Includes Codex and local verification subprocesses, even if the
            # worker itself has just exited. Never target another run's group.
            os.killpg(process.pid, signum)
            return
        except ProcessLookupError:
            pass  # The worker may not have reached setsid() yet.
    if process.is_alive():
        if signum == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()


def _stop_workers(active):
    for process in active.values():
        _signal_worker(process, signal.SIGTERM)
    deadline = time.monotonic() + 5
    for process in active.values():
        if process.pid is not None:
            process.join(timeout=max(0, deadline - time.monotonic()))
            # Also remove any descendants left after an interrupted SDK startup.
            _signal_worker(process, signal.SIGKILL)
            process.join()


def run_jobs(root, plan, workers=1, *, runner=None):
    """Run the reserved least-used batch with bounded, unrestricted concurrency."""
    root = Path(root).resolve()
    with pair_policy.dataset_lock(plan):
        ledger = pair_policy.load_ledger(plan)
        pair_policy.check_plan(ledger, root, plan)
        pair_policy.check_single_queue(ledger, root)
        try:
            return _run_jobs(root, plan, workers, ledger, runner=runner)
        finally:
            save_coverage(root, plan, ledger)


def save_coverage(root, plan, ledger):
    participation = pair_policy.completed_counts(plan, ledger)
    report = {'dataset': plan['dataset'], 'pairs': len(plan['jobs']),
              'generated_compositions': sum(participation.values()) // 2,
              'unfinished_compositions': sum(not pair_policy.author_complete(root, j['name']) for j in plan['jobs']),
              'participation': dict(participation), 'pair_policy': pair_policy.POLICY}
    write(root / 'coverage.json', report)
    return report


def _run_jobs(root, plan, workers, ledger, *, runner=None):
    if workers < 1:
        raise ValueError("workers must be at least 1")
    pending = deque()
    for job in plan["jobs"]:
        if stages_done(root, job, requested_stages(plan)):
            print(f"Skip finished stages {job['name']}", flush=True)
        else:
            pending.append(job)
    limit = min(workers, len(pending))
    print(f"Maximum parallel compositions: {limit} (requested workers: {workers})", flush=True)
    context = multiprocessing.get_context("spawn")
    active = {}
    active_jobs = {}
    try:
        while pending or active:
            # Observe every exit before scheduling replacements, so a runtime
            # failure stops the run without starting another queued composition.
            for name, process in list(active.items()):
                if process.exitcode is None:
                    continue
                process.join()
                if process.exitcode:
                    status = root / name / "status.json"
                    error = read(status).get("error", "Worker exited") if status.exists() else "Worker exited"
                    raise RuntimeError(f"Composition {name} stopped: {error}")
                job = active_jobs.pop(name)
                if not stages_done(root, job, requested_stages(plan)):
                    raise RuntimeError(f'Composition {name} did not finish; resume it before continuing the queue')
                del active[name]
            while pending and len(active) < limit:
                # Allocation already balanced and reserved these pairs. Their
                # execution order and shared parents do not restrict concurrency.
                job = pending.popleft()
                process = context.Process(target=_run_worker, args=(root, job, plan, runner),
                                          name=job["name"])
                active[job["name"]] = process
                active_jobs[job['name']] = job
                process.start()
            if active:
                time.sleep(0.1)
    except BaseException as error:
        _stop_workers(active)
        for name, process in active.items():
            if process.pid is not None:
                _mark_interrupted(root, name, error)
        raise
    return int(any(read(root / job["name"] / "status.json")["state"] == "failed"
                   for job in plan["jobs"]))


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--parents-root", type=Path,
                        help="Root of completed task-variation packages; default results/task-variations")
    parser.add_argument("--dataset", help="Relative dataset folder, e.g. arc2/evaluation or conceptarc/training; default arc2/evaluation")
    parser.add_argument("--output-root", type=Path, default=RESULTS)
    amounts = parser.add_mutually_exclusive_group()
    amounts.add_argument("--count", type=positive_int,
                        help="Total unique families desired; required for new runs, increase to append jobs")
    amounts.add_argument("--batch-size", type=positive_int,
                         help="Append this many pairs once the previous requested stages finish; otherwise resume")
    parser.add_argument("--stage", choices=("all", "author"), default="all",
                        help="Run both stages or only the author stage")
    parser.add_argument("--review-samples", type=positive_int,
                        help="Solved puzzles shown to the review (1..42; default 5)")
    parser.add_argument("--seed", type=int, help="Reproducible pair/example/sample selection")
    parser.add_argument("--model", help="Default: gpt-6.1-sol")
    parser.add_argument("--effort", help="Default: xhigh")
    parser.add_argument("--workers", type=positive_int, default=1,
                        help="Maximum concurrent compositions; each owns two sequential model stages")
    args = parser.parse_args()
    if args.review_samples is not None and args.review_samples > 42:
        parser.error("--review-samples must be 1..42")
    try:
        with run_lock(args.output_root.resolve()):
            plan = prepare(args)
            root = args.output_root.resolve()
            for job in plan["jobs"]:
                for name, digest in job["input_hashes"].items():
                    if sha(root / job["name"] / name) != digest:
                        raise ValueError(f"Prepared evidence changed: {job['name']}/{name}")
            print(f"Plan: {root / 'plan.json'} (seed {plan['seed']}); "
                  f"{len(plan['jobs'])}/{plan['pair_capacity']} unique pairs", flush=True)
            with pair_policy.dataset_lock(plan):
                coverage = save_coverage(root, plan, pair_policy.load_ledger(plan))
                participation = coverage['participation']
            print(f"Task coverage: {min(participation.values())}..{max(participation.values())} "
                  f"successful appearances per task across {len(participation)} tasks", flush=True)
            if args.command == "prepare":
                return 0
            plan["run_stages"] = requested_stages(args)
            return run_jobs(root, plan, args.workers)
    except KeyboardInterrupt:
        print("Stopped composition workers. Rerun to resume completed stages.", flush=True)
        return 130
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Composition v3 failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
