#!/usr/bin/env python3
"""Record Colab run facts and render small, deterministic agent handoffs."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path


if __package__:
    from . import colab_accounts
    from .run_layout import branch_name, results_root, runs_root
else:
    import colab_accounts
    from run_layout import branch_name, results_root, runs_root


ROOT = Path(__file__).resolve().parents[1]
STAGE_RE = re.compile(r"Starting (training stage|evaluation) '([^']+)'")
EPOCH_RE = re.compile(r"Epoch (\d+)/(\d+)")
ERROR_RE = re.compile(r"(?:torch\.)?\w*(?:Error|Exception): .+")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}-", delete=False) as output:
        temp_path = Path(output.name)
        output.write(content)
    temp_path.replace(path)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _journal_path(root: Path, session: str) -> Path:
    return results_root(root) / "colab_runs" / "_active" / f"{session}.json"


def _check_session_name(session: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", session) or session in {".", ".."}:
        raise ValueError("Invalid Colab session name")


def prompt_for_notes(run_name: str) -> tuple[str, str]:
    """Ask two short questions in native macOS dialogs before allocating a GPU."""
    if sys.platform != "darwin" or not shutil.which("osascript"):
        raise RuntimeError("A macOS notes dialog is unavailable; pass both --intent and --changed-from-previous")
    script = (
        "on run argv\n"
        "set answerRecord to display dialog (item 1 of argv) default answer \"\" "
        "with title \"mdlARC run notes\" buttons {\"Cancel\", \"Continue\"} "
        "default button \"Continue\" cancel button \"Cancel\"\n"
        "return text returned of answerRecord\n"
        "end run"
    )
    answers = []
    for question in (
        f"What is your intent for the {run_name} run? You can edit this note later.",
        "What changed from the previous run? You can edit this note later.",
    ):
        result = subprocess.run(
            ["osascript", "-e", script, question],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False,
        )
        if result.returncode:
            if "User canceled" in result.stderr or "(-128)" in result.stderr:
                raise RuntimeError("Run cancelled in the notes dialog; no GPU was allocated")
            raise RuntimeError(f"Could not open the notes dialog: {result.stderr.strip() or 'osascript failed'}")
        answers.append(result.stdout.rstrip("\n"))
    return answers[0], answers[1]


def create_note(root: Path, session: str, run_name: str, intent: str, changes: str) -> Path:
    _check_session_name(session)
    path = results_root(root) / "experiments" / "notes" / f"{session}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(
            f"# {session} — {run_name}\n\n"
            f"## Intent\n\n{intent.strip() or '_Not entered._'}\n\n"
            f"## What changed from the previous run\n\n{changes.strip() or '_Not entered._'}\n\n"
            "## Observations\n\n- \n\n"
            "## My interpretation\n\n- \n\n"
            "## Next action\n\n- \n"
        )
    return path


def start_journal(root: Path, session: str, *, config: str, run_name: str, gpu: str,
                  high_mem: bool, min_gpu_memory_gb: float, spec_path: Path) -> dict:
    _check_session_name(session)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False,
    )
    now = utc_now()
    journal = {
        "session": session,
        "branch": branch_name(root),
        "run_name": run_name,
        "config": config,
        "state": "preparing",
        "created_at": now,
        "updated_at": now,
        "requested_gpu": gpu,
        "high_mem": high_mem,
        "min_gpu_memory_gb": min_gpu_memory_gb,
        "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "session_disposition": "not_allocated",
    }
    _write_atomic(_journal_path(root, session), json.dumps(journal, indent=2) + "\n")
    return journal


def update_journal(root: Path, session: str, **fields) -> dict:
    path = _journal_path(root, session)
    journal = _read_json(path)
    if not journal:
        raise FileNotFoundError(f"No session record: {path}")
    journal.update(fields)
    journal["updated_at"] = utc_now()
    _write_atomic(path, json.dumps(journal, indent=2) + "\n")
    return journal


def observe_output(root: Path, session: str, line: str) -> bool:
    """Record only stage and epoch markers; return whether progress changed."""
    match = STAGE_RE.search(line)
    if match:
        update_journal(root, session, stage=match.group(2), epoch=None, epoch_total=None,
                       last_progress_at=utc_now())
        return True
    match = EPOCH_RE.search(line)
    if match:
        update_journal(root, session, epoch=int(match.group(1)), epoch_total=int(match.group(2)),
                       last_progress_at=utc_now())
        return True
    return False


def _link(source: Path, from_dir: Path, label: str) -> str:
    if not source.is_file():
        return label
    return f"[{label}]({os.path.relpath(source, from_dir)})"


def _short(value: object, limit: int = 240) -> str:
    text = str(value).replace("\n", " ").replace("\r", " ").replace("|", "\\|")
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _log_facts(path: Path) -> dict:
    facts: dict = {}
    if not path.is_file():
        return facts
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stage = STAGE_RE.search(line)
        if stage:
            facts.update(stage=stage.group(2), epoch=None, epoch_total=None)
        epoch = EPOCH_RE.search(line)
        if epoch:
            facts.update(epoch=int(epoch.group(1)), epoch_total=int(epoch.group(2)))
        error = ERROR_RE.search(line)
        if error:
            message = _short(error.group(0))
            if "OutOfMemoryError" in message or "OutOfMemoryError" not in facts.get("error", ""):
                facts["error"] = message
    return facts


def _result_dir(root: Path, session: str, run_name: str | None) -> Path | None:
    base = results_root(root) / "colab_runs" / session / "runs"
    if run_name and (base / run_name).is_dir():
        return base / run_name
    if base.is_dir():
        directories = [path for path in base.iterdir() if path.is_dir()]
        if len(directories) == 1:
            return directories[0]
    return None


def _metric_lines(run_dir: Path, summary_dir: Path) -> list[str]:
    lines = []
    validation = run_dir / "validation_scores.json"
    if validation.is_file():
        data = _read_json(validation)
        values = [f"{key}={data[key]['percent']:.4g}%" for key in ("top1", "top2")
                  if isinstance((data.get(key) or {}).get("percent"), (int, float))]
        lines.append(f"- Validation: {', '.join(values)}; {data.get('tasks', '?')} tasks, "
                     f"{data.get('queries', '?')} queries "
                     f"({_link(validation, summary_dir, 'validation_scores.json')})")
    training = run_dir / "training_stats.json"
    if training.is_file():
        data = _read_json(training)
        values = [f"{key}={data[key]:.4g}" for key in
                  ("seconds", "trainable_parameters", "peak_allocated_gib", "peak_reserved_gib")
                  if isinstance(data.get(key), (int, float))]
        lines.append(f"- Adaptation measurements: {', '.join(values)} "
                     f"({_link(training, summary_dir, 'training_stats.json')})")
    for path in sorted((run_dir / "evals").glob("*/score.json")):
        data = _read_json(path)
        values = [f"{key}={data[key]:.4g}" for key in ("score", "max_score", "percentage")
                  if isinstance(data.get(key), (int, float))]
        lines.append(f"- {path.parent.name}: {', '.join(values) or 'score file present'} "
                     f"({_link(path, summary_dir, 'score.json')})")
    for path in sorted((run_dir / "evals").glob("*/accuracy.json")):
        data = _read_json(path)
        test = ((data.get("splits") or {}).get("test") or {}).get("summary") or {}
        values = [f"{key}={test[key]:.4g}" for key in
                  ("exact_output_accuracy", "color_accuracy", "special_token_accuracy")
                  if isinstance(test.get(key), (int, float))]
        lines.append(f"- {path.parent.name}: {', '.join(values) or 'accuracy file present'} "
                     f"({_link(path, summary_dir, 'accuracy.json')})")
    return lines


def render_session(root: Path, session: str) -> tuple[str, str]:
    _check_session_name(session)
    journal = _read_json(_journal_path(root, session))
    result_base = results_root(root) / "colab_runs" / session
    status_path = result_base / "run_status.json"
    status = _read_json(status_path)
    run_dir = _result_dir(root, session, journal.get("run_name"))
    experiment_path = run_dir / "experiment.json" if run_dir else Path("/nonexistent")
    experiment = _read_json(experiment_path)
    spec_provenance = _read_json(result_base / "spec_provenance.json")
    run_name = journal.get("run_name") or (run_dir.name if run_dir else "unknown")
    summary = results_root(root) / "experiments" / "generated" / f"{session}.md"
    summary_dir = summary.parent
    note = results_root(root) / "experiments" / "notes" / f"{session}.md"
    console = run_dir / "console.log" if run_dir else results_root(root) / "colab_runs" / "_active" / f"{session}.log"
    vm_console = results_root(root) / "colab_runs" / "_active" / f"{session}.vm.log"
    if run_dir is None and vm_console.is_file():
        console = vm_console
    facts = _log_facts(console)
    ledger_path = run_dir / "checkpoint_backups.json" if run_dir else Path("/nonexistent")
    ledger = _read_json(ledger_path)
    if status:
        state = "completed" if status.get("ok") and status.get("backups_complete", True) else "failed"
        if status.get("backups_complete") is False:
            state = "needs recovery"
    else:
        state = journal.get("state", "unknown")
    lines = [
        f"# {session}", "", "Generated facts only. Your interpretation belongs in the separate note.", "",
        f"- **Run:** {_short(run_name)}",
        f"- **State:** {_short(state)}" + (" (last recorded; may be stale)" if not status else ""),
        f"- **Last updated:** {_short(journal.get('updated_at') or status.get('finished_at') or 'unknown')}",
        f"- **Manual note:** {_link(note, summary_dir, note.name) if note.is_file() else 'not created'}",
        f"- **Raw status:** {_link(status_path, summary_dir, 'run_status.json')}",
    ]
    if journal.get("session_disposition"):
        lines.append(f"- **Colab session:** {_short(journal['session_disposition'])}")
    if journal.get("colab_account"):
        lines.append(f"- **Colab compute account:** {_short(journal['colab_account'])} ({_short(journal.get('colab_email'))})")
    if journal.get("last_progress_at"):
        lines.append(f"- **Last progress observed:** {_short(journal['last_progress_at'])}")
    if journal.get("last_vm_check_at"):
        lines.append(f"- **Last VM file check:** {_short(journal['last_vm_check_at'])}; "
                     "a successful check alone does not confirm training progress")
    identity_path = results_root(root) / "colab_runs" / "_active" / f"{session}.connection.json"
    connection_path = identity_path.with_suffix(".status.json")
    connection = _read_json(connection_path)
    if identity_path.is_file():
        lines.append(f"- **Saved kernel/execution identity:** {_link(identity_path, summary_dir, 'connection.json')}; "
                     "attachment does not resubmit execution")
    if connection:
        lines.append(f"- **Local connection:** {_short(connection.get('state'))}; "
                     f"{connection.get('connections', 0)} successful connections in this worker, "
                     f"last recorded {_short(connection.get('updated_at'))} "
                     f"({_link(connection_path, summary_dir, 'connection status')}); may be stale")
    check = journal.get("session_check") or {}
    if check:
        lines.append(f"- **CLI check:** {_short(check.get('result'))} at {_short(check.get('checked_at'))}; "
                     "this does not confirm training progress")
    if status.get("drive_folder_url"):
        lines.append(f"- **Drive run folder:** [open]({status['drive_folder_url']})")
    if journal.get("launcher_error"):
        lines.append(f"- **Launcher error:** {_short(journal['launcher_error'])}")
    if status.get("error"):
        lines.append(f"- **Run error:** {_short(status['error'])}")
    lines += ["", "## Provenance", ""]
    for label, value in (
        ("Spec path in uploaded source", journal.get("config")),
        ("Spec SHA-256 before launch", journal.get("spec_sha256")),
        ("Uploaded source SHA-256", experiment.get("source_sha256") or journal.get("source_sha256")),
        ("Git revision at launch", journal.get("git_revision")),
        ("Requested GPU", journal.get("requested_gpu")),
        ("High-memory machine requested", journal.get("high_mem")),
        ("Minimum GPU memory GiB", journal.get("min_gpu_memory_gb")),
        ("GPU", experiment.get("gpu") or journal.get("requested_gpu")),
        ("GPU memory GiB", experiment.get("gpu_memory_gib")),
    ):
        if value is not None:
            lines.append(f"- **{label}:** {_short(value)}")
    saved_spec = result_base / "spec.json"
    if saved_spec.is_file():
        lines.append(f"- **Selected spec in uploaded archive:** {_link(saved_spec, summary_dir, 'spec.json')}")
    archived_spec = result_base / "specs" / Path(journal.get("config") or "").name
    if archived_spec.is_file():
        archived_sha256 = hashlib.sha256(archived_spec.read_bytes()).hexdigest()
        if journal.get("spec_sha256") and archived_sha256 != journal["spec_sha256"]:
            lines.append("- **Archived selected spec:** hash differs from the launch journal")
        else:
            lines.append(f"- **Archived selected spec:** {_link(archived_spec, summary_dir, archived_spec.name)}")
    if spec_provenance.get("trial_id"):
        lines.append(f"- **Trial ID:** {_short(spec_provenance['trial_id'])}")
        lines.append(f"- **Trial record:** {_link(result_base / 'trial.json', summary_dir, 'trial.json')}")
        lines.append(f"- **Template at launch:** {_link(result_base / 'template.json', summary_dir, 'template.json')}")
    if experiment:
        lines.append(f"- **Exact spec:** {_link(experiment_path, summary_dir, 'experiment.json')}")
    if experiment.get("source_sha256") and journal.get("source_sha256"):
        if experiment["source_sha256"] != journal["source_sha256"]:
            lines.append("- **Source verification:** local archive hash differs from the recovered experiment record")
    spec = experiment.get("spec") or {}
    if spec:
        adaptation = spec.get("checkpoint_adaptation")
        if adaptation:
            selection = (f"last {adaptation.get('last', 4)} blocks" if adaptation["method"] == "last_blocks"
                         else f"all-layer LoRA rank {adaptation.get('rank', 16)}, alpha {adaptation.get('alpha', 16)}")
            lines.append(f"- **Checkpoint adaptation:** {selection}; {adaptation.get('epochs', 90)} epochs; "
                         f"microbatch {adaptation.get('batch_size', 1)}, accumulation {adaptation.get('accumulation', 40)}; "
                         f"train/eval augmentations {adaptation.get('train_augments', 80)}/{adaptation.get('eval_augments', 80)}")
        defaults = spec.get("defaults") or {}
        for label, key in (
            ("Pretrain batch size", "batch_size"),
            ("Gradient accumulation steps", "gradient_accumulation_steps"),
            ("Optimizer", "optimizer"),
            ("Model width", "d_model"),
            ("Model layers", "n_layers"),
        ):
            if key in defaults:
                lines.append(f"- **{label}:** {_short(defaults[key])}")
        pretrain = spec.get("pretrain") or {}
        if pretrain:
            lines.append(f"- **Pretrain:** {pretrain.get('epochs', 'unknown')} epochs; "
                         f"checkpoint epochs {pretrain.get('checkpoint_epochs', 'unspecified')}")
        if isinstance(spec.get("finetunes"), dict):
            for name, settings in spec["finetunes"].items():
                lines.append(f"- **Finetune {name}:** {settings.get('epochs', 'unknown')} epochs; "
                             f"checkpoint epochs {settings.get('checkpoint_epochs', 'unspecified')}")
        finetune_defaults = spec.get("finetune_defaults") or {}
        if "batch_size" in finetune_defaults:
            lines.append(f"- **Finetune batch size:** {_short(finetune_defaults['batch_size'])}")
    if len(lines) >= 2 and lines[-2:] == ["## Provenance", ""]:
        lines.append("- No provenance record available")
    lines += ["", "## Progress and results", ""]
    stage = facts.get("stage") or journal.get("stage")
    epoch = facts.get("epoch") if facts.get("stage") else journal.get("epoch")
    total = facts.get("epoch_total") if facts.get("stage") else journal.get("epoch_total")
    lines.append(f"- **Last observed stage:** {_short(stage or 'none recorded')}")
    if epoch is not None:
        lines.append(f"- **Last observed epoch:** {epoch}/{total or '?'}")
    if console.is_file():
        lines.append(f"- **Console evidence:** {_link(console, summary_dir, 'console.log')}")
    if facts.get("error"):
        lines.append(f"- **Last parsed error:** {facts['error']}")
    if ledger:
        files = ledger.get("files") or []
        lines.append(f"- **Checkpoint uploads:** {len(files)} recorded; "
                     f"backup ledger complete={ledger.get('complete', 'unknown')} "
                     f"({_link(ledger_path, summary_dir, 'checkpoint_backups.json')})")
        for item in files:
            if isinstance(item, dict) and item.get("path"):
                lines.append(f"  - {_short(item['path'])}")
    elif status:
        lines.append("- **Checkpoint uploads:** no ledger found")
    if run_dir:
        lines.extend(_metric_lines(run_dir, summary_dir))
    if status and not status.get("ok") and not facts.get("error"):
        failure = run_dir / "failure.txt" if run_dir else Path("/nonexistent")
        if failure.is_file():
            lines.append(f"- **Failure trace:** {_link(failure, summary_dir, 'failure.txt')}")
    lines += ["", "## Agent review", ""]
    if status.get("backups_complete") is False:
        lines.append("- Check backup records and the Drive folder before stopping or retrying this session.")
    elif status and not status.get("ok"):
        lines.append("- Inspect the failure and exact uploaded spec before proposing a change.")
    elif status.get("ok"):
        lines.append("- Review evaluation results and compare with earlier runs.")
    else:
        lines.append("- Verify the session state before acting; live progress can be stale after a launcher interruption.")
    if note.is_file():
        lines.append("- Read the manual note for intent, interpretation, and next action.\n")
    else:
        lines.append("- No manual note was captured for this run.\n")
    _write_atomic(summary, "\n".join(lines))
    return state, run_name


def known_sessions(root: Path) -> list[str]:
    # Connection identities/status and diagnostic markers also live in _active.
    # Only a session journal owns a state and an exact matching session name.
    active = [path for path in (results_root(root) / "colab_runs" / "_active").glob("*.json")
              if (record := _read_json(path)).get("session") == path.stem and "state" in record]
    results = (results_root(root) / "colab_runs").glob("*/run_status.json")
    return sorted({path.stem for path in active} | {path.parent.name for path in results})


def render_index(root: Path) -> None:
    lines = [
        "# Colab experiments", "",
        "Generated from local run records. Read the generated facts alongside each manual note.", "",
        "| Session | Run | State | Notes |", "| --- | --- | --- | --- |",
    ]
    for session in reversed(known_sessions(root)):
        state, run_name = render_session(root, session)
        note = results_root(root) / "experiments" / "notes" / f"{session}.md"
        note_cell = f"[notes](notes/{session}.md)" if note.is_file() else "—"
        lines.append(f"| [{session}](generated/{session}.md) | {_short(run_name)} | "
                     f"{_short(state)} | {note_cell} |")
    _write_atomic(results_root(root) / "experiments" / "INDEX.md", "\n".join(lines) + "\n")


def refresh_session(root: Path, session: str, account=None) -> dict:
    _check_session_name(session)
    owner = colab_accounts.run_account(root, session, account)
    path = _journal_path(root, session)
    if not path.is_file():
        now = utc_now()
        _write_atomic(path, json.dumps({"session": session, "state": "untracked",
                                         "created_at": now, "updated_at": now,
                                         "colab_account": owner, "colab_email": colab_accounts.ACCOUNTS[owner]}, indent=2) + "\n")
    colab = shutil.which("colab")
    if not colab:
        result = "unknown (Colab CLI unavailable)"
    else:
        try:
            command = subprocess.run(colab_accounts.cli_command(colab, "status", "-s", session, account=owner), cwd=root,
                                     stdin=subprocess.DEVNULL, capture_output=True,
                                     text=True, timeout=30, check=False)
            if command.returncode != 0:
                result = "unknown (CLI check failed)"
            elif "not found" in command.stdout.lower() or "no active sessions" in command.stdout.lower():
                # `colab status` exits zero even when the named session is gone.
                result = "session not visible to CLI"
            else:
                result = "session visible to CLI"
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            result = "unknown (CLI check failed)"
    journal = update_journal(root, session, session_check={"checked_at": utc_now(), "result": result})
    render_index(root)
    return journal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", help="Branch owning the records (default: current Git branch)")
    subparsers = parser.add_subparsers(dest="command", required=True)
    render = subparsers.add_parser("render", help="Regenerate reports from local records")
    render.add_argument("--session", help="Render one session (default: all)")
    refresh = subparsers.add_parser("refresh", help="Check a session with the Colab CLI and regenerate reports")
    refresh.add_argument("--session", required=True)
    refresh.add_argument("--account", choices=colab_accounts.ACCOUNTS,
                         help="For untracked sessions only; recorded sessions use their saved account")
    args = parser.parse_args()
    if args.branch:
        os.environ["MDLARC_RUN_BRANCH"] = args.branch
    if args.command == "render":
        if args.session:
            _check_session_name(args.session)
            render_session(ROOT, args.session)
        render_index(ROOT)
    else:
        journal = refresh_session(ROOT, args.session, args.account)
        print(f"{args.session}: {journal['session_check']['result']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
