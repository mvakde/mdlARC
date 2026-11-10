#!/usr/bin/env python3
"""Offline artifact bridge for native Claude Code workers; never calls an LLM API."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import FunctionType, SimpleNamespace
import uuid

import run as core
import pair_policy
from v2_storage import read, write, sha

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MODEL = "claude-opus-5-5"
EFFORT = "xhigh"
# Workers default to the plan's xhigh; a coordinator may request high for a run.
WORKER_EFFORTS = ("high", "xhigh")
DEFAULT_OUTPUT = REPO / "results/task-composition-v3-claude-arc2-training"
STATE = "claude-web.json"
DRAFT_FILES = ("composition.md", "generator.py", "task_spec.md", "review.md",
               "review-complete.json", "review_coverage.md", "feedback.json", "native-response.md")


class RepairRequired(Exception):
    """The submitted turn failed checks; a native worker must do the repair."""


class SubmittedArtifacts:
    """Feed one already-finished native turn into v3's existing check/publish loop.

    No inference occurs here. A second requested turn returns control to Claude
    Code instead of launching Codex, a nested CLI, or a separately billed API.
    """
    def __init__(self, folder, receipt):
        self.folder = folder
        self.receipt = receipt
        self.used = False
        self.id = "native-claude-code:" + receipt["agent_id"]

    def thread_start(self, **options):
        if (Path(options["cwd"]) != self.folder or options["model"] != MODEL
                or options["effort"] != EFFORT):
            raise ValueError("Native receipt belongs to a different workspace/model/effort")
        return self

    def run(self, prompt):
        if self.used:
            raise RepairRequired()
        self.used = True
        return SimpleNamespace(final_response=self.receipt["summary"], usage=None)


class NoModelCalls:
    def thread_start(self, **options):
        raise RuntimeError("Unfinished stage requires a native Claude Code worker")


def run_native_stage(client, job_root, stage, job, plan):
    """Use the unmodified v3 stage loop with a private, SDK-free turn starter.

    Copy only the function's globals so the Codex runner and its pinned source
    hash stay untouched. The original invoke() imports a Codex approval enum;
    native artifact acceptance has no reason to import that SDK. No process-wide
    monkeypatch is used, and all verification/publication helpers remain shared.
    """
    def native_start(client, folder, model, effort):
        return client.thread_start(cwd=str(folder), model=model, effort=effort)
    workflow = FunctionType(core.run_stage.__code__,
                            {**core.run_stage.__globals__, "invoke": native_start},
                            core.run_stage.__name__, core.run_stage.__defaults__, core.run_stage.__closure__)
    return workflow(client, job_root, stage, job, plan)


def inside_repo(path):
    path = Path(path).resolve()
    path.relative_to(REPO.resolve())
    return path


def implementation_hashes():
    paths = [HERE / "claude_web.py", HERE / "claude-web",
             REPO / ".claude/skills/arc-compose/SKILL.md",
             REPO / ".claude/agents/arc-composition-worker.md"]
    return {str(p.relative_to(REPO)): sha(p) for p in paths}


def load_state(root, *, takeover=False):
    state = read(root / STATE)
    if (state.get("backend") != "claude-code-web" or state.get("requested_model") != MODEL
            or state.get("requested_effort") != EFFORT):
        raise ValueError("This output root is not an Opus 5.5 xhigh Claude Code web run")
    current = implementation_hashes()
    if state["implementation_hashes"] != current:
        # Validation code stays pinned by plan.json. Coordinator changes may apply
        # between stages, never to a worker assigned under the previous procedure.
        if state["active"] and not takeover:
            raise ValueError("Claude workflow changed while stages were active; finish them with "
                             "the previous workflow or use a new --output-root")
        for task in state["active"]:
            # start hands active stages to new workers; earlier workers' tickets stop working.
            task["ticket"] = uuid.uuid4().hex
        state.setdefault("previous_implementation_hashes", []).append(state["implementation_hashes"])
        state["implementation_hashes"] = current
    return state


def relocate(root, state):
    """Rebase v3's absolute pointers after cloning onto another cloud VM."""
    old_repo = Path(state["repo_root"])
    plan_path = root / "plan.json"
    if not plan_path.exists():
        plan = {"parents_root": str(REPO / state["parents_relative"]), "dataset": state["dataset"]}
    else:
        plan = read(plan_path)
        plan["parents_root"] = str(REPO / state["parents_relative"])
        plan["python"] = sys.executable
        for parent in plan["candidates"]:
            parent["path"] = str(Path(plan["parents_root"]) / plan["dataset"] / parent["id"])
    with pair_policy.dataset_lock(plan):
        marker = pair_policy.ledger_path(plan)
        if marker.exists():
            ledger = read(marker)
            roots = {}
            for saved, records in ledger["roots"].items():
                path = Path(saved)
                if path.is_relative_to(REPO):
                    destination = path
                elif path.is_relative_to(old_repo):
                    destination = REPO / path.relative_to(old_repo)
                else:
                    # A committed ledger can predate this web workflow. Match an
                    # existing repo-relative output by its dataset/fingerprint.
                    parts = path.parts
                    if "results" not in parts:
                        raise ValueError("Pair ledger points outside this checkout: " + saved)
                    destination = REPO.joinpath(*parts[parts.index("results"):])
                    other = destination / "plan.json"
                    if not other.exists() or read(other)["dataset"] != plan["dataset"]:
                        raise ValueError("Missing prior run needed by the pair ledger: " + saved)
                if str(destination) in roots:
                    raise ValueError("Relocation would merge two ledger roots")
                roots[str(destination)] = records
            ledger["roots"] = roots
            write(marker, ledger)
        if plan_path.exists():
            write(plan_path, plan)
    state["repo_root"] = str(REPO)
    write(root / STATE, state)


def restore_evidence(root, job):
    """Canonical examples are Git-ignored copies; restore only exact missing bytes."""
    folder = root / job["name"]
    missing = [name for name in job["input_hashes"] if not (folder / name).exists()]
    if missing:
        if any(not name.startswith("llm-inputs/evidence/examples/") for name in missing):
            raise ValueError("Missing published parent evidence for " + job["name"])
        with tempfile.TemporaryDirectory() as temp:
            examples = Path(temp) / "examples"
            core.stage_examples(job["review_examples"], examples)
            for name in missing:
                source = examples / Path(name).relative_to("llm-inputs/evidence/examples")
                if not source.is_file() or sha(source) != job["input_hashes"][name]:
                    raise ValueError("Rebuilt example differs from pinned evidence: " + name)
                target = folder / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
    for name, digest in job["input_hashes"].items():
        path = folder / name
        if path.is_symlink() or not path.is_file() or sha(path) != digest:
            raise ValueError("Prepared evidence changed: " + job["name"] + "/" + name)


def restore_samples(root, job):
    """Rebuild ignored samples and require their original checkpoint hashes."""
    folder = root / job["name"]
    checkpoints = folder / "checkpoints.json"
    if not checkpoints.exists():
        return
    for stage, files in read(checkpoints).items():
        missing = [name for name in files if not (folder / name).exists()]
        if missing:
            allowed = "llm-inputs/author/checked-samples/" if stage == "author" else "samples/"
            if any(not name.startswith(allowed + "seed-") for name in missing):
                raise ValueError("Missing published stage artifact for " + job["name"])
            with tempfile.TemporaryDirectory() as temp:
                work = Path(temp)
                source = folder / ("generator_v1.py" if stage == "author" else "generator_v2.py")
                if sha(source) != files[source.name]:
                    raise ValueError("Published generator changed: " + str(source))
                shutil.copyfile(source, work / "generator.py")
                core.copy_tree(folder / "parents", work / "parents")
                samples = work / "samples"
                result = core.verify(work, work / "report.json", job["check_seeds"][stage], samples)
                if not result["valid"]:
                    raise ValueError("Published generator failed sample reconstruction: " + str(result["errors"]))
                for name in missing:
                    source = samples / Path(name).name
                    if not source.is_file() or sha(source) != files[name]:
                        raise ValueError("Rebuilt sample differs from its checkpoint: " + name)
                    target = folder / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
        core.finished_stage(folder, stage)


def restore_run(root, plan, deferred=()):
    for job in plan["jobs"]:
        if job["name"] in deferred:
            continue
        restore_evidence(root, job)
        restore_samples(root, job)
        if core.stages_done(root, job, core.requested_stages(plan)):
            finalize_job(root, job, plan)


def finalize_job(root, job, plan):
    core.run_job(NoModelCalls(), root, job, plan)
    provenance = root / job["name"] / "provenance.json"
    if provenance.exists():
        # plan.json pins xhigh; receipts record the effort each stage's workers were requested at.
        efforts = {stage: sorted({read(r).get("requested_effort", EFFORT) for r in receipts})
                   for stage in core.STAGES
                   if (receipts := list((root / job["name"] / "llm-inputs" / stage).glob("native-turn-*.json")))}
        write(provenance, {**read(provenance), "backend": "claude-code-web",
                           "usage_mode": "native-session-no-api-client",
                           "native_requested_effort": efforts})


def stage_workspace(root, job, stage):
    package = root / job["name"] / "llm-inputs" / stage
    existed = (package / "workspace").exists() and (package / "protected.json").exists()
    previous = read(package / "protected.json") if (package / "protected.json").exists() else None
    folder, protected = core.setup_stage(root / job["name"], stage, job)
    if not existed:
        # Repo-relative venv paths keep protected starting inputs portable.
        runtime = (os.path.relpath(sys.executable, folder)
                   if Path(sys.executable).is_relative_to(REPO) else sys.executable)
        instructions = (folder / "instructions.md").read_text().replace(sys.executable, runtime)
        (folder / "instructions.md").write_text(instructions)
        write(folder / "environment.json", {"python_executable": runtime})
        (folder / "CLAUDE.md").write_text(
            "This is one ARC composition stage. Read AGENTS.md and instructions.md.\n"
            "Work only in this stage workspace; preserve protected evidence.\n"
            "Do not read other tasks, use network tools, delegate, or commit.\n"
            "Write native-response.md describing the result and any limitations.\n")
        protected = core.hashes(folder)
        if previous is not None and protected != previous:
            write(package / "protected.json", previous)
            raise ValueError("Reconstructed stage differs from its protected starting inputs")
        core.copy_tree(folder, package / "inputs")
        write(package / "protected.json", protected)
        drafts = package / "saved-output/native"
        if (drafts / "manifest.json").exists():
            for name, record in read(drafts / "manifest.json")["files"].items():
                if name not in DRAFT_FILES or (drafts / name).is_symlink() or sha(drafts / name) != record["sha256"]:
                    raise ValueError("Saved native draft changed: " + name)
                shutil.copyfile(drafts / name, folder / name)
                os.utime(folder / name, ns=(record["mtime_ns"], record["mtime_ns"]))
    errors = core.integrity(folder, protected)
    if errors:
        raise ValueError("; ".join(errors))
    return folder


def snapshot(root, task):
    package = root / task["job"] / "llm-inputs" / task["stage"]
    folder = package / "workspace"
    if not folder.exists():
        return
    errors = core.integrity(folder, read(package / "protected.json"))
    if errors:
        raise ValueError("; ".join(errors))
    target = package / "saved-output/native"
    target.mkdir(parents=True, exist_ok=True)
    records = {}
    for name in DRAFT_FILES:
        source = folder / name
        if source.is_symlink():
            raise ValueError("Native deliverable must not be a symlink: " + name)
        if source.is_file():
            shutil.copyfile(source, target / name)
            # Hash the saved copy: other workers keep writing during rolling checkpoints.
            records[name] = {"sha256": sha(target / name), "mtime_ns": source.stat().st_mtime_ns}
        else:
            (target / name).unlink(missing_ok=True)
    write(target / "manifest.json", {"files": records, "ticket": task["ticket"]})


def fill_slots(root, state, plan, *, resume=False):
    """Refill every free worker slot immediately, like the Codex runner's queue.

    A slot follows one composition: an accepted author frees it for that pair's
    review, and a finished pair frees it for the next pending pair. Started pairs
    precede unstarted ones in plan order, so they are continued first. A pair never
    has two active stages. start (resume=True) returns every active assignment for
    a new coordinator; next returns only new ones, so no worker is dispatched twice.
    """
    if state.get("stopped"):
        return {"action": "stopped", "error": state["stopped"], "tasks": [], "running": []}
    state["active"] = [t for t in state["active"]
                       if not core.finished_stage(root / t["job"], t["stage"])]
    busy = {t["job"] for t in state["active"]}
    added = []
    for job in plan["jobs"]:
        if len(state["active"]) >= state["workers"]:
            break
        if job["name"] in busy or job["name"] in state.get("deferred", ()):
            continue
        stage = next((s for s in core.requested_stages(plan)
                      if not core.finished_stage(root / job["name"], s)), None)
        if stage:
            task = {"job": job["name"], "stage": stage, "ticket": uuid.uuid4().hex, "submissions": 0,
                    "effort": state.get("worker_effort", EFFORT)}
            state["active"].append(task)
            added.append(task)
    write(root / STATE, state)
    tasks = []
    by_name = {j["name"]: j for j in plan["jobs"]}
    for task in state["active"] if resume else added:
        folder = stage_workspace(root, by_name[task["job"]], task["stage"])
        tasks.append({**task, "workspace": str(folder), "python": sys.executable,
                      "instructions": str(folder / "instructions.md"), "model": MODEL,
                      "effort": task.get("effort", EFFORT),
                      "worker": "arc-composition-worker"})
    dispatched = {t["ticket"] for t in tasks}
    running = [t["ticket"] for t in state["active"] if t["ticket"] not in dispatched]
    action = "work" if tasks else "wait" if running else "complete"
    return {"action": action, "tasks": tasks, "running": running, "deferred": state.get("deferred", [])}


def start(root, args):
    root = inside_repo(root)
    parents = inside_repo(args.parents_root or core.PARENTS)
    if Path(args.dataset).is_absolute() or ".." in Path(args.dataset).parts:
        raise ValueError("--dataset must be relative to --parents-root")
    if (root / STATE).exists():
        state = load_state(root, takeover=True)
        if state["dataset"] != args.dataset or state["parents_relative"] != str(parents.relative_to(REPO)):
            raise ValueError("Saved run uses another dataset/parents root; use a new --output-root")
        relocate(root, state)
        for task in state["active"]:
            snapshot(root, task)
        if state.get("stopped"):
            state.pop("stopped")
            state["active"] = []
        if args.stage == "author" and any(t["stage"] == "review-improve" for t in state["active"]):
            raise ValueError("Unfinished review stages exist; resume with --stage all")
        if len(state["active"]) > args.workers:
            raise ValueError("Finish the active stages before reducing --workers")
        names = {j["name"] for j in read(root / "plan.json")["jobs"]} if (root / "plan.json").exists() else set()
        def job_name(value):
            found = [n for n in names if value in (n, n.split("-")[0])]
            if len(found) != 1:
                raise ValueError("Unknown composition: " + value)
            return found[0]
        # A deferred pair keeps its reservation and drafts but is neither scheduled nor
        # rebuilt from its ignored samples until --undefer.
        deferred = set(state.get("deferred", [])) | {job_name(v) for v in args.defer}
        state["deferred"] = sorted(deferred - {job_name(v) for v in args.undefer})
        state["active"] = [t for t in state["active"] if t["job"] not in state["deferred"]]
    else:
        if (root / "plan.json").exists():
            raise ValueError("Existing output root belongs to another backend; use a new --output-root")
        if args.defer or args.undefer:
            raise ValueError("--defer/--undefer apply to compositions of an existing run")
        state = {"schema_version": 1, "backend": "claude-code-web",
                 "usage_mode": "native-session-no-api-client", "requested_model": MODEL,
                 "requested_effort": EFFORT,
                 "repo_root": str(REPO), "parents_relative": str(parents.relative_to(REPO)),
                 "dataset": args.dataset, "implementation_hashes": implementation_hashes(),
                 "workers": args.workers, "active": []}
        write(root / STATE, state)
        relocate(root, state)
    if (root / "plan.json").exists():
        restore_run(root, read(root / "plan.json"), state.get("deferred", ()))
    options = argparse.Namespace(parents_root=parents, output_root=root, dataset=args.dataset,
        seed=args.seed, model=MODEL, effort=EFFORT, review_samples=args.review_samples,
        count=args.count, batch_size=args.batch_size, stage=args.stage)
    if options.count is None and options.batch_size is None and not (root / "plan.json").exists():
        options.batch_size = 100
    plan = core.prepare(options)
    plan["run_stages"] = core.requested_stages(options)
    write(root / "plan.json", plan)
    state["workers"] = args.workers
    state["worker_effort"] = args.effort or state.get("worker_effort", EFFORT)
    for task in state["active"]:
        task["effort"] = state["worker_effort"]
    write(root / STATE, state)
    return fill_slots(root, state, plan, resume=True)


def submit(root, state, plan, args):
    task = next((t for t in state["active"] if t["ticket"] == args.ticket), None)
    if task is None:
        raise ValueError("Unknown or already-submitted ticket")
    if state.get("stopped"):
        raise ValueError("Run is stopped; invoke start to resume")
    job = next(j for j in plan["jobs"] if j["name"] == task["job"])
    folder = stage_workspace(root, job, task["stage"])
    response = folder / "native-response.md"
    if response.is_symlink() or not response.is_file() or not response.read_text().strip():
        raise ValueError("Worker must write nonempty native-response.md before submission")
    receipt = {"backend": "claude-code-web", "requested_model": MODEL,
               "requested_effort": task.get("effort", EFFORT),
               "session_id": args.session_id, "agent_id": args.agent_id,
               "summary": response.read_text()[:20000],
               "billing": "native Claude Code session; no API request by this bridge",
               "model_verification": "requested model/effort; native runtime settings are not independently inspected"}
    task["submissions"] += 1
    write(folder.parent / f"native-turn-{uuid.uuid4().hex}.json", receipt)
    try:
        passed = run_native_stage(SubmittedArtifacts(folder, receipt), root / job["name"],
                                  task["stage"], job, plan)
    except RepairRequired:
        passed = False
    except (ValueError, RuntimeError) as error:
        state["stopped"] = str(error)
        write(root / STATE, state)
        raise
    finally:
        snapshot(root, task)
    if passed:
        state["active"].remove(task)
        if core.stages_done(root, job, core.requested_stages(plan)):
            finalize_job(root, job, plan)
        else:
            write(root / job["name"] / "status.json", {"state": "author_completed", "stage": "author"})
    elif task["submissions"] >= 3:
        state["stopped"] = "Three submissions failed for " + task["job"] + "/" + task["stage"]
    write(root / STATE, state)
    with pair_policy.dataset_lock(plan):
        core.save_coverage(root, plan, pair_policy.load_ledger(plan))
    return {"valid": passed, "action": "accepted" if passed else "stopped" if state.get("stopped") else "repair",
            "feedback": str(folder / "feedback.json"), "ticket": task["ticket"]}


def status(root, state, plan):
    jobs = [j for j in plan["jobs"] if j["name"] not in state.get("deferred", ())]
    return {"backend": state["backend"], "requested_model": MODEL,
            "requested_effort": state.get("worker_effort", EFFORT), "dataset": plan["dataset"],
            "pairs": len(plan["jobs"]), "workers": state["workers"],
            "author_completed": sum(core.finished_stage(root / j["name"], "author") for j in jobs),
            "completed": sum(core.stages_done(root, j, core.requested_stages(plan)) for j in jobs),
            "stopped": state.get("stopped"), "active": state["active"],
            "deferred": state.get("deferred", []),
            "persist_paths": [str(root.relative_to(REPO)),
                str((Path(plan["parents_root"]) / plan["dataset"] / pair_policy.LEDGER).relative_to(REPO)),
                str((Path(plan["parents_root"]) / plan["dataset"] / ".composition-v3-pair-order.json").relative_to(REPO))]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    launch = commands.add_parser("start", help="Reserve/resume a batch, then return native worker assignments")
    launch.add_argument("--dataset", default="arc2/training")
    launch.add_argument("--parents-root", type=Path)
    amounts = launch.add_mutually_exclusive_group()
    amounts.add_argument("--batch-size", type=core.positive_int)
    amounts.add_argument("--count", type=core.positive_int)
    launch.add_argument("--workers", type=core.positive_int, default=4)
    launch.add_argument("--stage", choices=("all", "author"), default="all")
    launch.add_argument("--seed", type=int)
    launch.add_argument("--review-samples", type=core.positive_int)
    launch.add_argument("--effort", choices=WORKER_EFFORTS,
                        help="Worker effort for this and later assignments (default: the run's, initially xhigh)")
    launch.add_argument("--defer", action="append", default=[], metavar="JOB",
                        help="Stop scheduling this composition (name or number); keeps its drafts")
    launch.add_argument("--undefer", action="append", default=[], metavar="JOB",
                        help="Schedule a deferred composition again")
    for name in ("next", "submit", "checkpoint", "status"):
        command = commands.add_parser(name)
        if name == "submit":
            command.add_argument("--ticket", required=True)
            command.add_argument("--session-id", required=True)
            command.add_argument("--agent-id", required=True)
    for command in commands.choices.values():
        command.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if getattr(args, "review_samples", None) and args.review_samples > 42:
        parser.error("--review-samples must be 1..42")
    try:
        root = inside_repo(args.output_root)
        # v3 progress goes to stderr; stdout is always machine-readable JSON.
        with redirect_stdout(sys.stderr), core.run_lock(root):
            if args.command == "start":
                result = start(root, args)
            else:
                state = load_state(root)
                relocate(root, state)
                plan = read(root / "plan.json")
                restore_run(root, plan, state.get("deferred", ()))
                if args.command == "next":
                    result = fill_slots(root, state, plan)
                elif args.command == "submit":
                    result = submit(root, state, plan, args)
                elif args.command == "checkpoint":
                    for task in state["active"]:
                        snapshot(root, task)
                    result = status(root, state, plan)
                else:
                    result = status(root, state, plan)
        print(json.dumps({"ok": True, **result}))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
