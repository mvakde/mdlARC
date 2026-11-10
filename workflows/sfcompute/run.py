#!/usr/bin/env python3
"""Launch, observe and recover SF Compute Autoresearch nodes through the shared supervisor."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import time
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workflows.sfcompute import policy as policies
from workflows.vast import run as shared, snapshot
from workflows.vast.common import DEFAULT_TOKEN, atomic_json, hashes, lock, safe_name
from workflows.vast.vast_api import AmbiguousSubmission

ROOT = shared.ROOT
DEFAULT_POLICY = Path.home() / ".config/mdlarc-sfcompute/policy.json"


def launch_policy(args):
    value = json.loads(args.policy.read_text())
    if args.gpu_template:
        value.update(chip=policies.TEMPLATES[args.gpu_template],
                     num_gpus=8 if args.gpu_template == "h100-8" else 1)
    for key in ("image", "workspace", "api_url"):
        if getattr(args, key, None) is not None:
            value[key] = getattr(args, key)
    return policies.validate({**value, **policies.parse_overrides(args.policy_override)})


def preflight_account(client, policy, sizes):
    # Read effective account limits before requesting any paid capacity.
    limits = {row["name"]: row["value"] for row in client.request("get", "/limits", reading=True)["limits"]}
    ceiling = limits.get("max_chunked_write_bytes", limits.get("max_write_file_bytes", 0))
    if any(size > ceiling for size in sizes):
        raise ValueError("An input exceeds this account's SF Compute upload limit; raise the limit or reduce the package before renting")
    if policy["scratch_gib"] > limits.get("node_scratch_gib", 0):
        raise ValueError("Requested scratch exceeds the SF Compute account limit")
    billing = client.request("get", "/billing", reading=True)
    balance = billing.get("balance") or {}
    available = balance.get("available_usd", billing.get("available_to_spend_usd", 0))
    if type(available) not in (int, float) or not math.isfinite(available) or available < policy["max_total_usd"]:
        raise ValueError("SF Compute available credit does not cover the saved total allowance; no credit was purchased")


def image_warnings(spec, policy):
    from src.attention_runtime import selected_versions
    selected = selected_versions(spec)
    catalog = not any(c in policy["image"] for c in "/:@")
    warnings = []
    if catalog and selected.intersection({2, 3}):
        warnings.append("FA2/FA3 requires a custom image with the selected attention backend preinstalled; catalog PyTorch images do not provide it")
    if catalog and 4 in selected and "cuda13" not in policy["image"]:
        warnings.append("The pinned FA4 cu13 stack requires a CUDA 13 image")
    return warnings


def launch(args):
    requested_at = time.time()
    branch = shared.branch_name(args.branch)
    spec_path = args.config.expanduser().resolve()
    spec_relative = spec_path.relative_to(ROOT).as_posix()
    policy = launch_policy(args)
    args.workspace, args.api_url = policy["workspace"], policy["api_url"]
    spec = json.loads(spec_path.read_text())
    warnings = image_warnings(spec, policy)
    if warnings and not args.dry_run:
        raise ValueError("; ".join(warnings) + "; no node was requested")
    spec_sha = hashes(spec_path)["sha256"]
    trial = shared.trial_provenance(args, spec_path, spec)
    resolved = shared.preview(spec_path)
    from launch_run import validate_distributed_gpu_count
    validate_distributed_gpu_count(resolved, policy["num_gpus"])
    run_name = safe_name(spec.get("run_name") or spec_path.stem)
    session = safe_name(args.session or f"sfcompute-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:8]}")
    cache = shared.cache_dir(branch, session, "sfcompute")
    if cache.exists():
        raise ValueError("Session already has records; use status/reconcile/handoff instead of launching again")
    with tempfile.TemporaryDirectory(prefix="mdlarc-sfcompute-source-") as temp:
        archive = Path(temp) / "source.tar.gz"
        source_inputs = snapshot.pinned_family_input(ROOT, spec)
        source = snapshot.create(ROOT, spec_path, archive, external_inputs=source_inputs or None)
        with tarfile.open(archive, "r:gz") as contents:
            packaged_spec = contents.extractfile(spec_relative).read()
        if hashlib.sha256(packaged_spec).hexdigest() != spec_sha:
            raise ValueError("Experiment spec changed while packaging; repeat the preview")
        if args.dry_run:
            print(json.dumps({"backend": "sfcompute", "branch": branch, "session": session,
                "policy": policy, "source": source, "source_inputs": source_inputs,
                "resolved_experiment": resolved,
                "image_warnings": warnings,
                "estimated_ceiling_usd": policy["max_hourly_usd"] * policy["max_runtime_hours"]}, indent=2, default=str))
            return 0
        if args.secrets_file:
            shared.validate_training_secrets(args.secrets_file)
        intent, changes = shared.notes(args, run_name)
        store = shared.store_for(args, branch)
        store.preflight()
        client = shared.client_for(args)
        sizes = [source["size"], *(identity["size"] for identity in source_inputs.values())]
        preflight_account(client, policy, sizes)
        # Preserve the unique provider name and source before the create request.
        cache.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(archive, cache / "source.tar.gz")
        (cache / "spec.json").write_bytes(packaged_spec)
    with lock(cache / ".launch.lock"):
        store = shared.store_for(args, branch, cache)
        folder = store.create_run(session)
        rented_at = time.time()
        plan = {"schema_version": 1, "backend": "sfcompute", "branch": branch, "session": session,
                "run_name": run_name, "spec_relative": spec_relative, "spec_sha256": spec_sha,
                "source": source, "source_inputs": source_inputs, "policy": policy,
                "launch_requested_at_unix": requested_at, "rented_at_unix": rented_at,
                "deadline_unix": rented_at + policy["max_runtime_hours"] * 3600,
                "label": "mdlarc-" + uuid.uuid4().hex, "instance_id": None,
                "gpu_name": "NVIDIA H100 80GB HBM3", "gpu_template": args.gpu_template,
                "policy_overrides": policies.parse_overrides(args.policy_override),
                "drive_config": store.config, "drive_folder_id": folder,
                "remote_workspace": policy["remote_root"] + "/" + session,
                "intent": intent, "changed_from_previous": changes}
        if trial:
            plan["trial"] = {"trial_id": trial.trial_id, "record": json.loads(trial.trial_bytes),
                             "template": json.loads(trial.template_bytes)}
        atomic_json(cache / "launch.json", plan)
        shared.save_lease(cache, plan, args.lease_export_dir)
        (cache / "note.md").write_text(f"# {session}\n\nRun: {run_name}\n\nIntent: {intent}\n\nChanged from previous: {changes}\n")
        shared.save_state(cache, state="rental_submitting")
        try:
            node = client.create(policy, plan["label"])
        except AmbiguousSubmission:
            shared.save_state(cache, state="rental_unconfirmed")
            raise RuntimeError(f"SF Compute creation is unconfirmed. Use reconcile --session {session}; do not launch again") from None
        except Exception:
            shared.save_state(cache, state="rental_rejected")
            raise
        plan["instance_id"] = node["id"]
        atomic_json(cache / "launch.json", plan)
        shared.save_lease(cache, plan, args.lease_export_dir)
        shared.save_state(cache, state="instance_loading", instance_id=node["id"])
        try:
            plan["quote"] = policies.verify_node(node, policy, plan["label"], node["id"])
            atomic_json(cache / "launch.json", plan)
            shared.handoff(args, cache, plan, client)
        except BaseException:
            phase = json.loads((cache / "state.json").read_text())["state"]
            shared.save_state(cache, state="handoff_needs_recovery", phase_before_error=phase)
            if phase in {"instance_loading", "uploading"}:
                try:
                    shared.stop_owned(client, plan)
                    shared.save_state(cache, disposition="stop_requested")
                except Exception:
                    shared.save_state(cache, disposition="stop_unconfirmed")
            raise
    print(f"SF Compute handoff confirmed on {plan['label']}. The detached supervisor owns training and verified Drive backups. "
          f"Recover with: workflows/sfcompute/run.py sync --branch {branch} --session {session}", flush=True)
    return 0 if args.detach else shared.watch(args, cache, plan)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(backend="sfcompute")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("templates", help="List H100 shapes without network access")
    for name in ("launch", "watch", "sync", "status", "reconcile", "handoff", "recover-backups", "start", "stop"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--branch")
        cmd.add_argument("--session", required=name != "launch")
        cmd.add_argument("--autor", type=Path, help="Path to the installed official autor CLI")
        cmd.add_argument("--workspace")
        cmd.add_argument("--api-url")
        cmd.add_argument("--drive-token", type=Path, default=DEFAULT_TOKEN)
        cmd.add_argument("--drive-config", type=Path, default=ROOT / "workflows/vast/drive.example.json")
        cmd.add_argument("--poll-seconds", type=int, default=30)
        cmd.add_argument("--secrets-file", type=Path)
        if name == "launch":
            cmd.add_argument("--config", type=Path, required=True)
            cmd.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
            cmd.add_argument("--policy-override", action="append", default=[])
            cmd.add_argument("--gpu-template", choices=policies.TEMPLATES)
            cmd.add_argument("--image")
            cmd.add_argument("--intent")
            cmd.add_argument("--changed-from-previous")
            cmd.add_argument("--dry-run", action="store_true")
            cmd.add_argument("--detach", action="store_true")
            cmd.add_argument("--lease-export-dir", type=Path)
            provenance = cmd.add_mutually_exclusive_group()
            provenance.add_argument("--trial-record", type=Path)
            provenance.add_argument("--standalone-spec", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "templates":
        print(json.dumps(policies.TEMPLATES, indent=2))
        return 0
    if args.poll_seconds < 5:
        raise ValueError("Poll interval must be at least five seconds")
    if args.command == "launch":
        return launch(args)
    return shared.main(parsed_args=args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Local watcher closed; the existing job was not resubmitted.", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"SF Compute workflow: {str(exc) if isinstance(exc, (ValueError, RuntimeError)) else type(exc).__name__}", file=sys.stderr)
        raise SystemExit(1) from None
