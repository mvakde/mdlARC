#!/usr/bin/env python3
"""Launch and recover Runpod Pods using the shared detached experiment supervisor."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import time
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workflows.runpod import policy as policies
from workflows.vast import run as shared, snapshot
from workflows.vast.common import DEFAULT_TOKEN, atomic_json, hashes, lock, safe_name
from workflows.vast.ssh import public_key
from workflows.vast.vast_api import AmbiguousSubmission

ROOT = shared.ROOT
DEFAULT_POLICY = Path.home() / ".config/mdlarc-runpod/policy.json"


def launch_policy(args):
    policy = policies.load(args.policy)
    if args.gpu_template:
        policy = {**policy, "gpu_type_ids": policies.TEMPLATES[args.gpu_template], "num_gpus": 1}
    if args.image:
        policy = {**policy, "image": args.image}
    return policies.validate({**policy, **policies.parse_overrides(args.policy_override)})


def launch(args):
    requested_at = time.time()
    branch = shared.branch_name(args.branch)
    spec_path = args.config.expanduser().resolve()
    spec_relative = spec_path.relative_to(ROOT).as_posix()
    policy = launch_policy(args)
    spec = json.loads(spec_path.read_text())
    spec_sha = hashes(spec_path)["sha256"]
    trial = shared.trial_provenance(args, spec_path, spec)
    resolved = shared.preview(spec_path)
    from launch_run import validate_distributed_gpu_count
    validate_distributed_gpu_count(resolved, policy["num_gpus"])
    run_name = safe_name(spec.get("run_name") or spec_path.stem)
    session = safe_name(args.session or f"runpod-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:8]}")
    cache = shared.cache_dir(branch, session, "runpod")
    if cache.exists():
        raise ValueError("Session already has records; use status/reconcile/handoff instead of launching again")
    with tempfile.TemporaryDirectory(prefix="mdlarc-runpod-source-") as temp:
        archive_path = Path(temp) / "source.tar.gz"
        source_inputs = snapshot.pinned_family_input(ROOT, spec)
        source = (snapshot.create(ROOT, spec_path, archive_path, external_inputs=source_inputs)
                  if source_inputs else snapshot.create(ROOT, spec_path, archive_path))
        with tarfile.open(archive_path, "r:gz") as archive:
            packaged_spec = archive.extractfile(spec_relative).read()
        if hashlib.sha256(packaged_spec).hexdigest() != spec_sha:
            raise ValueError("Experiment spec changed while packaging; repeat the preview")
        if args.dry_run:
            body = (None if args.pod_id else
                    policies.create_body(policy, "PREVIEW", policy["gpu_type_ids"][0], "PUBLIC_KEY_AT_LAUNCH"))
            print(json.dumps({"backend": "runpod", "branch": branch, "session": session,
                              "policy": policy, "gpu_template": args.gpu_template,
                              "policy_overrides": policies.parse_overrides(args.policy_override),
                              "existing_pod_id": args.pod_id, "pod_request": body, "source": source,
                              "source_inputs": source_inputs,
                              "resolved_experiment": resolved}, indent=2, default=str))
            return 0
        identity = args.ssh_key.expanduser().resolve()
        if not identity.is_file() or identity.is_relative_to(ROOT):
            raise ValueError("Configure a private SSH key outside the repository")
        key = public_key(identity)
        if args.secrets_file:
            shared.validate_training_secrets(args.secrets_file)
        intent, changes = shared.notes(args, run_name)
        # All inputs and credentials are checked before renting. No local
        # checkpoint is accepted by the shared snapshot builder.
        store = shared.store_for(args, branch)
        store.preflight()
        client = shared.client_for(args)
        adopted = None
        previous_runs = []
        if args.pod_id:
            from workflows.runpod.api import normalize, pod_id
            pod_id(args.pod_id)
            for record in (ROOT / "results").glob("*/runpod_runs/*/launch.json"):
                saved = json.loads(record.read_text())
                if saved.get("instance_id") == args.pod_id:
                    old_store = shared.store_for(args, saved["branch"], config=saved["drive_config"])
                    completed = shared.manifest_from_drive(old_store, saved)
                    if not completed or not completed.get("finalized") or not completed.get("backups_complete"):
                        raise ValueError(f"Pod run lacks finalized backups: {saved['branch']}/{saved['session']}; use recovery")
                    archive = completed["archive"]
                    old_store._verify_binary(old_store.metadata(archive["id"]), saved["drive_folder_id"], archive["name"], archive)
                    previous_runs.append({key: saved[key] for key in
                        ("branch", "session", "instance_id", "label", "remote_workspace", "drive_folder_id")})
            raw = client.raw_pod(args.pod_id)
            quote = policies.verify_pod(raw, policy, raw.get("name"), adopted=True)
            adopted = normalize(raw)
            gpu = adopted["gpu_name"]
            # An adopted rental is always preserved, including errors and deadlines.
            policy = {**policy, "success_action": "keep", "failure_action": "keep"}
        else:
            policies.validate_api_cuda(policy, client.supported_cuda_versions())
            quote, gpu = policies.choose_gpu(client.gpu_types(policy), policy)
        policy = {**policy, "gpu_type_ids": [gpu]}
        cache.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(archive_path, cache / "source.tar.gz")
        (cache / "spec.json").write_bytes(packaged_spec)
    with lock(cache / ".launch.lock"):
        store = shared.store_for(args, branch, cache)
        folder = store.create_run(session)
        rented_at = time.time()
        plan = {"schema_version": 1, "backend": "runpod", "branch": branch, "session": session,
                "run_name": run_name, "spec_relative": spec_relative, "spec_sha256": spec_sha,
                "source_inputs": source_inputs, "launch_requested_at_unix": requested_at,
                "source": source, "policy": policy, "quote": quote, "gpu_template": args.gpu_template,
                "policy_overrides": policies.parse_overrides(args.policy_override),
                "drive_config": store.config, "drive_folder_id": folder,
                "label": adopted["label"] if adopted else "mdlarc:runpod:" + uuid.uuid4().hex,
                "instance_id": args.pod_id,
                "rented_at_unix": rented_at, "deadline_unix": rented_at + policy["max_runtime_hours"] * 3600,
                "remote_workspace": policy["remote_root"] + "/" + session,
                "intent": intent, "changed_from_previous": changes, "ssh_key_path": str(identity)}
        if adopted:
            plan.update(adopted_instance=adopted, preserve_instance=True, gpu_name=gpu,
                        allowance_started_at="adoption", previous_runs=previous_runs)
        if trial:
            plan["trial"] = {"trial_id": trial.trial_id, "record": json.loads(trial.trial_bytes),
                             "template": json.loads(trial.template_bytes)}
        atomic_json(cache / "launch.json", plan)
        shared.save_lease(cache, plan, args.lease_export_dir)
        (cache / "note.md").write_text(f"# {session}\n\nRun: {run_name}\n\nIntent: {intent}\n\nChanged from previous: {changes}\n")
        shared.save_state(cache, state="adoption_prepared" if adopted else "rental_submitting")
        try:
            if not adopted:
                plan["instance_id"] = client.create(policies.create_body(policy, plan["label"], gpu, key))
        except AmbiguousSubmission:
            shared.save_state(cache, state="rental_unconfirmed")
            raise RuntimeError(f"Pod creation outcome is unknown. Run reconcile --session {session}; do not launch again") from None
        except Exception:
            shared.save_state(cache, state="rental_rejected")
            raise
        atomic_json(cache / "launch.json", plan)
        shared.save_lease(cache, plan, args.lease_export_dir)
        shared.save_state(cache, state="instance_loading", instance_id=plan["instance_id"])
        try:
            # Catalog prices are estimates. Recheck the allocated pod before
            # sending source, Drive credentials, or training to it.
            actual = policies.verify_pod(client.raw_pod(plan["instance_id"]), policy, plan["label"], adopted=bool(adopted))
            plan["allocated_quote"] = actual
            atomic_json(cache / "launch.json", plan)
            shared.handoff(args, cache, plan, client)
        except BaseException:
            phase = json.loads((cache / "state.json").read_text())["state"]
            shared.save_state(cache, state="handoff_needs_recovery", phase_before_error=phase)
            if not adopted and phase in {"instance_loading", "uploading"}:
                try:
                    shared.stop_owned(client, plan)
                    shared.save_state(cache, disposition="stop_requested")
                except Exception:
                    shared.save_state(cache, disposition="stop_unconfirmed")
            raise
    print(f"Runpod handoff confirmed on pod {plan['instance_id']}. You can close the laptop. "
          f"Recover with: workflows/runpod/run.py sync --branch {branch} --session {session}", flush=True)
    return 0 if args.detach else shared.watch(args, cache, plan)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(backend="runpod")
    commands = parser.add_subparsers(dest="command", required=True)
    templates = commands.add_parser("templates", help="Show GPU templates without network access or renting")
    templates.add_argument("--json", action="store_true")
    for name in ("launch", "retry", "watch", "sync", "status", "reconcile", "handoff", "recover-backups", "start", "stop"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--branch")
        cmd.add_argument("--session", required=name != "launch")
        cmd.add_argument("--drive-token", type=Path, default=DEFAULT_TOKEN)
        cmd.add_argument("--drive-config", type=Path, default=ROOT / "workflows/vast/drive.example.json")
        cmd.add_argument("--api-key-file", type=Path)
        cmd.add_argument("--poll-seconds", type=int, default=30)
        cmd.add_argument("--secrets-file", type=Path)
        if name == 'retry':
            cmd.add_argument('--retry-session', required=True)
            cmd.add_argument('--restart-config', type=Path,
                             help='Explicit fresh restart using this replacement spec; preserves all parent checkpoint backups')
            cmd.add_argument('--source-update', type=Path, action='append', default=[])
            cmd.add_argument('--skip-cuda-validation', action='store_true',
                             help='Reuse passed CUDA/FA4 validation from this same pod and skip optional validation')
            cmd.add_argument('--detach', action='store_true')
            cmd.add_argument('--dry-run', action='store_true')
        if name == "launch":
            cmd.add_argument("--config", type=Path, required=True)
            cmd.add_argument("--pod-id", help="Adopt this existing idle rental; never creates, stops, or destroys it")
            cmd.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
            cmd.add_argument("--policy-override", action="append", default=[], metavar="FIELD=JSON_VALUE")
            cmd.add_argument("--gpu-template", choices=policies.GPU_TEMPLATES)
            cmd.add_argument("--image")
            cmd.add_argument("--ssh-key", type=Path, default=shared.DEFAULT_SSH_KEY)
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
        if args.json:
            print(json.dumps(policies.TEMPLATES, indent=2))
        else:
            for name, ids in policies.TEMPLATES.items():
                print(f"{name}: {policies.GPU_TEMPLATES[name]}; {', '.join(ids)}")
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
        print("Local watcher closed. The existing remote job was not resubmitted.", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:
        message = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else type(exc).__name__
        print(f"Runpod workflow: {message}", file=sys.stderr)
        raise SystemExit(1) from None
