#!/usr/bin/env python3
"""Launch, watch, reconcile, and recover an independent Vast.ai experiment."""

import argparse
import base64
from contextlib import nullcontext
from datetime import datetime
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workflows.vast import policy as policies, snapshot
from workflows.vast.common import DEFAULT_TOKEN, atomic_json, branch_folder, extract_archive, hashes, keep_b300_running, lock, now, relative_file, safe_name
from workflows.vast.drive import Drive, service
from workflows.vast.ssh import SSH, public_key
from workflows.vast.vast_api import AmbiguousSubmission, ApiError, Vast, api_key, public_instance

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = Path.home() / ".config/mdlarc-vast/policy.json"
DEFAULT_SSH_KEY = Path.home() / ".ssh/id_ed25519"


def branch_name(explicit=None):
    value = explicit or os.environ.get("MDLARC_RUN_BRANCH")
    if not value:
        result = subprocess.run(["git", "symbolic-ref", "--quiet", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, check=False)
        value = result.stdout.strip()
    branch_folder(value)
    return value


def cache_dir(branch, session, backend="vast"):
    if backend not in {"vast", "runpod", "sfcompute"}:
        raise ValueError("Unknown remote backend")
    return ROOT / "results" / branch_folder(branch) / f"{backend}_runs" / safe_name(session)


def store_for(args, branch, cache=None, config=None):
    token = args.drive_token.expanduser().resolve()
    if token.is_relative_to(ROOT) or token.is_symlink() or not token.is_file():
        raise ValueError("Dedicated remote-run Drive token must be a regular file outside the repository; run workflows/vast/setup_drive.py")
    config = config or json.loads(args.drive_config.read_text())
    return Drive(service(token), config, branch, control=cache / "private/drive" if cache else None)


def client_for(args):
    if getattr(args, "backend", "vast") == "sfcompute":
        from workflows.sfcompute.api import Autor
        identity = getattr(args, "sfcompute_identity", None)
        return Autor(getattr(args, "autor", None), workspace=args.workspace,
                     api_url=args.api_url, identities=dict([identity]) if identity else None)
    if getattr(args, "backend", "vast") == "runpod":
        from workflows.runpod.api import Runpod, api_key as runpod_key
        return Runpod(runpod_key(args.api_key_file))
    return Vast(api_key(args.api_key_file))


def connection_for(client, instance, plan, cache):
    if plan.get("backend") == "sfcompute":
        from workflows.sfcompute.transport import Transport
        return Transport(client, instance)
    return SSH(instance, plan["ssh_key_path"], cache / "known_hosts")


def save_state(cache, **values):
    path = cache / "state.json"
    state = json.loads(path.read_text()) if path.exists() else {}
    state.update(values, observed_at=now())
    atomic_json(path, state)
    return state


def save_lease(cache, plan, export_dir=None):
    lease = {k: plan[k] for k in ("instance_id", "label", "deadline_unix", "branch", "session")}
    lease.update(gpu_name=plan.get("gpu_name") or (plan.get("adopted_instance") or {}).get("gpu_name"),
                 preserve_instance=keep_b300_running(plan))
    atomic_json(cache / "lease.json", lease)
    if export_dir:
        directory = Path(export_dir).expanduser().resolve() / branch_folder(plan["branch"]) / plan["session"]
        atomic_json(directory / "lease.json", lease)


def preview(spec_path):
    # This validates the same scientific schema without loading GPU modules or installing anything.
    import launch_run
    return launch_run.preview_plan(launch_run.build_plan(launch_run.load_spec(spec_path), spec_path))


def notes(args, run_name):
    if (args.intent is None) != (args.changed_from_previous is None):
        raise ValueError("Provide both --intent and --changed-from-previous, or neither for the existing Mac dialogs")
    if args.intent is None:
        from scripts.colab_run_docs import prompt_for_notes
        return prompt_for_notes(run_name)
    return args.intent, args.changed_from_previous


def trial_provenance(args, spec_path, spec):
    if args.trial_record:
        from scripts.materialize_sweep_spec import materialize
        trial = materialize(ROOT, args.trial_record)
        if spec_path.read_bytes() != trial.spec_bytes:
            raise ValueError("Selected experiment does not match --trial-record; materialize the trial again")
        return trial
    if (spec_path.relative_to(ROOT).parts[:2] == ("specs", "temp specs")
            and not args.standalone_spec and not isinstance(spec.get("checkpoint"), dict)):
        raise ValueError("Sweep-generated temp specs require --trial-record; manually authored temp specs require --standalone-spec")
    return None


def load_plan(args):
    branch = branch_name(args.branch)
    backend = getattr(args, "backend", "vast")
    cache = cache_dir(branch, args.session, backend)
    plan = json.loads((cache / "launch.json").read_text())
    if (plan["branch"] != branch or plan["session"] != args.session
            or plan.get("backend", "vast") != backend):
        raise ValueError("Saved launch identity differs from the selected branch/session")
    if backend == "sfcompute":
        for key in ("workspace", "api_url"):
            saved = plan["policy"][key]
            if getattr(args, key, None) not in (None, saved):
                raise ValueError("SF Compute connection differs from the saved run")
            setattr(args, key, saved)
        args.sfcompute_identity = (plan["instance_id"], plan["label"])
    return cache, plan


def launch_policy(args):
    policy = policies.load(args.policy)
    if args.gpu_template:
        policy = policies.apply_template(policy, args.gpu_template)
    if args.image:
        policy = {**policy, "image": args.image, "image_selection": "fixed"}
    if args.image_selection:
        policy = {**policy, "image_selection": args.image_selection}
    policy = {**policy, **policies.parse_overrides(getattr(args, "policy_override", []))}
    return policies.validate(policy, launching=not args.dry_run)


def ensure_instance_unclaimed(instance_id):
    for record in (ROOT / "results").glob("*/vast_runs/*/launch.json"):
        saved = json.loads(record.read_text())
        if saved.get("instance_id") == instance_id:
            raise ValueError(f"Instance already has a saved run: {saved['branch']}/{saved['session']}; observe or recover it")


def verified_previous_b300_runs(args, instance):
    """Reuse a kept B300 only after every saved run has canonical final backups."""
    previous = []
    for record in sorted((ROOT / "results").glob("*/vast_runs/*/launch.json")):
        saved = json.loads(record.read_text())
        if saved.get("instance_id") == instance["id"]:
            if record.parent != cache_dir(saved["branch"], saved["session"]):
                raise ValueError("Previous run record differs from its saved branch/session")
            previous.append(saved)
    if not previous:
        return []
    if not keep_b300_running({"instance_id": instance["id"]}, instance):
        ensure_instance_unclaimed(instance["id"])
    if instance.get("label") not in {saved["label"] for saved in previous}:
        raise ValueError("B300 label does not match any saved previous run")
    verified = []
    for saved in previous:
        store = store_for(args, saved["branch"], config=saved["drive_config"])
        completion = manifest_from_drive(store, saved)
        if not completion or not completion.get("finalized") or not completion.get("backups_complete"):
            raise ValueError(f"Previous B300 run lacks finalized verified backups: {saved['session']}")
        identity = completion["archive"]
        store._verify_binary(store.metadata(identity["id"]), saved["drive_folder_id"], identity["name"], identity)
        verified.append({**{key: saved[key] for key in
            ("branch", "session", "instance_id", "label", "remote_workspace", "drive_folder_id")},
            "ok": completion["ok"], "finalized": True, "backups_complete": True,
            "archive": identity})
    return verified


def existing_rental_start(instance, policy):
    started = instance.get("start_date")
    if type(started) not in (int, float) or not math.isfinite(started) or started <= 0 or started > time.time() + 60:
        raise ValueError("Selected instance has invalid rental start metadata")
    if started + policy["max_runtime_hours"] * 3600 - policy["backup_grace_seconds"] <= time.time():
        raise ValueError("Selected instance's runtime allowance has already expired")
    return started


def retry_batch_overrides(spec, batch_size=None, gradient_accumulation_steps=None, *, effective_batch_size=None):
    """Apply explicit pretraining batch settings; effective changes need an exact opt-in."""
    values = {'batch_size': batch_size, 'gradient_accumulation_steps': gradient_accumulation_steps}
    if all(value is None for value in values.values()):
        if effective_batch_size is not None:
            raise ValueError('Effective batch override requires both batch size and accumulation')
        return {}
    if any(type(value) is not int or value <= 0 for value in values.values()):
        raise ValueError('Retry batch size and accumulation must both be positive integers')
    previous = {key: (spec.get('pretrain') or {}).get(key, spec.get('defaults', {}).get(key))
                for key in values}
    if any(type(value) is not int or value <= 0 for value in previous.values()):
        raise ValueError('Saved spec must explicitly define its original batch size and accumulation')
    target = previous['batch_size'] * previous['gradient_accumulation_steps']
    if effective_batch_size is not None:
        if type(effective_batch_size) is not int or effective_batch_size <= 0:
            raise ValueError('Effective batch override must be a positive integer')
        target = effective_batch_size
    if batch_size * gradient_accumulation_steps != target:
        raise ValueError('Retry must preserve the original effective batch size or match its explicit override')
    spec.setdefault('pretrain', {}).update(values)
    return {f'pretrain.{key}': {'before': previous[key], 'after': value}
            for key, value in values.items()}


def evaluation_retry_overrides(spec, checkpoint, run_name, batch_size, token_budget):
    if any(type(value) is not int or value <= 0 for value in (batch_size, token_budget)):
        raise ValueError("Evaluation retry batch size and token budget must be positive integers")
    if not isinstance(checkpoint, dict) or checkpoint.get("sha256") == "drive":
        raise ValueError("Evaluation retry requires an exact saved Drive checkpoint identity")
    if spec.get("skip_evals") or not spec.get("evals") or any(
            key in spec for key in ("training_benchmark", "backend_benchmark", "speculative_benchmark", "evaluation_budget_benchmark")):
        raise ValueError("Evaluation retry requires the original regular evaluation stages")
    import launch_run
    launch_run.resolve_input_checkpoint(checkpoint, Path("runs") / safe_name(run_name))
    changes = {}
    for key, value in (("run_name", run_name), ("checkpoint", checkpoint), ("eval_only", True), ("enable_finetune", False)):
        if spec.get(key) != value:
            changes[key] = {"before": spec.get(key), "after": value}
            spec[key] = value
    defaults = spec.setdefault("eval_defaults", {})
    for key, value, fallback in (("batch_size", batch_size, 120), ("prefill_token_budget", token_budget, 524288)):
        changes[f"eval_defaults.{key}"] = {"before": defaults.get(key, fallback), "after": value}
        defaults[key] = value
        for name, stage in spec["evals"].items():
            if key in stage:
                changes[f"evals.{name}.{key}"] = {"before": stage[key], "after": value}
                stage[key] = value
    return changes


def retry_source(cache, plan, destination, source_updates=(), *, batch_size=None, gradient_accumulation_steps=None,
                 effective_batch_size=None, eval_checkpoint=None, eval_run_name=None,
                 eval_batch_size=None, eval_token_budget=None, replacement_spec=None, skip_cuda_validation=False,
                 cached_inputs=False):
    """Preserve the experiment; explicitly selected source fixes retain hash provenance."""
    original = cache / "source.tar.gz"
    if hashes(original)["sha256"] != plan["source"]["sha256"]:
        raise ValueError("Saved source snapshot checksum differs from the original launch")
    with tempfile.TemporaryDirectory(prefix="mdlarc-vast-retry-source-") as temp:
        root = Path(temp).resolve()
        extract_archive(original, root, exclude_checkpoints=True)
        spec_path = root / plan["spec_relative"]
        if hashes(spec_path)["sha256"] != plan["spec_sha256"]:
            raise ValueError("Saved experiment spec checksum differs from the original launch")
        updates = []
        seen = set()
        for selected in source_updates:
            selected = Path(selected)
            if (selected.is_absolute() or '..' in selected.parts or selected.suffix != '.py'
                    or not (selected.parts[0] in {'src', 'dataset_building_scripts'}
                            or selected.as_posix() in {'launch_run.py', 'workflows/vast/worker.py'})):
                raise ValueError('Source updates must name repository-relative training or supervisor Python files')
            name = selected.as_posix()
            if name in seen:
                raise ValueError('Duplicate source update')
            seen.add(name)
            local = relative_file(ROOT, selected)
            target = relative_file(root, selected)
            if not local.is_file() or not target.is_file():
                raise ValueError('Source updates must replace existing snapshot files')
            before, after = hashes(target), hashes(local)
            shutil.copyfile(local, target)
            updates.append({'path': name, 'before_sha256': before['sha256'], 'sha256': after['sha256']})
        spec = json.loads(spec_path.read_text())
        replacement_changes = {}
        if replacement_spec is not None:
            def differences(before, after, prefix=''):
                for key in before.keys() | after.keys():
                    name = prefix + key
                    old, new = before.get(key), after.get(key)
                    if isinstance(old, dict) and isinstance(new, dict):
                        differences(old, new, name + '.')
                    elif old != new:
                        replacement_changes[name] = {'before': old, 'after': new}
            differences(spec, replacement_spec)
            spec = json.loads(json.dumps(replacement_spec))
            spec_path.write_text(json.dumps(spec, indent=2) + '\n')
        spec_changes = retry_batch_overrides(spec, batch_size, gradient_accumulation_steps,
                                            effective_batch_size=effective_batch_size)
        spec_changes = {**replacement_changes, **spec_changes}
        if skip_cuda_validation:
            spec_changes['validate_cuda'] = {'before': spec.get('validate_cuda', False), 'after': False}
            spec['validate_cuda'] = False
        if eval_checkpoint is not None:
            if spec_changes:
                raise ValueError("An evaluation-only retry cannot change training batches")
            spec_changes = evaluation_retry_overrides(spec, eval_checkpoint, eval_run_name,
                                                     eval_batch_size, eval_token_budget)
        elif eval_batch_size is not None or eval_token_budget is not None:
            raise ValueError("Evaluation overrides require --eval-checkpoint")
        if spec_changes:
            spec_path.write_text(json.dumps(spec, indent=2) + '\n')
        dataset = spec.get("dataset") or {}
        added = []
        if ((dataset.get("pretrain_data") or {}).get("kind") == "task_variations"
                or dataset.get("finetune_source") == "task_variations_full"):
            config_path = root / "scripts/colab_drive.json"
            if not config_path.exists():
                local = ROOT / "scripts/colab_drive.json"
                if local.is_symlink() or not local.is_file():
                    raise ValueError("Required local scripts/colab_drive.json is missing or linked")
                config_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(local, config_path)
                added.append("scripts/colab_drive.json")
            config = json.loads(config_path.read_text())
            expected = plan["drive_config"]["task_variations"]
            if (config.get("dataset_size") != expected["size"]
                    or config.get("dataset_sha256") != expected["sha256"]):
                raise ValueError("Dataset metadata differs from the original approved archive")
        resolved = preview(spec_path)
        source_inputs = plan.get('source_inputs')
        if plan.get('backend', 'vast') in {'runpod', 'vast'} and not source_inputs:
            family = spec.get('dataset', {}).get('pretrain_data', {})
            if cached_inputs and spec.get('skip_data_build') and family.get('kind') == 'family_sampled':
                source_inputs = {'family_catalog': family['catalog']}
            else:
                source_inputs = snapshot.pinned_family_input(root, spec)
        source = snapshot.create(root, spec_path, destination, external_inputs=source_inputs,
                                 cached_inputs=cached_inputs)
        return source, spec_path.read_bytes(), added, resolved, updates, spec_changes


def continue_run(args):
    """Continue training or evaluation using verified inputs on the saved VM."""
    parent_cache, parent = load_plan(args)
    policy_overrides = policies.parse_overrides(getattr(args, "policy_override", []))
    allowed_overrides = {"max_total_usd", "max_runtime_hours", "max_upload_gb", "max_download_gb", "upload_reserve_gb", "checkpoint_upload_mode", "checkpoint_upload_every_epochs"}
    if policy_overrides.keys() - allowed_overrides:
        raise ValueError("Continuation overrides are limited to explicit runtime, budget and backup allowances")
    continuation_policy = policies.validate({**parent["policy"], **policy_overrides})
    continuation_quote = parent["quote"]
    if policy_overrides:
        current_instance = client_for(args).instance(parent["instance_id"])
        if current_instance.get("label") != parent["label"]:
            raise ValueError("Continuation allowance update requires the original identified instance")
        continuation_quote = policies.quote_cost(current_instance, continuation_policy)
    deadline = parent["deadline_unix"]
    if getattr(args, "deadline_at", None) is not None:
        selected_deadline = datetime.fromisoformat(args.deadline_at)
        if selected_deadline.tzinfo is None:
            raise ValueError("Continuation deadline requires an explicit timezone")
        deadline = selected_deadline.timestamp()
        latest_runtime = parent["rented_at_unix"] + continuation_policy["max_runtime_hours"] * 3600
        latest_funded = parent["rented_at_unix"] + (
            continuation_policy["max_total_usd"] - continuation_quote["bandwidth_reserved_usd"]
        ) / continuation_quote["hourly_usd"] * 3600
        if not math.isfinite(deadline) or deadline > min(latest_runtime, latest_funded) + 1e-6:
            raise ValueError("Requested continuation deadline exceeds the explicit runtime or spending allowance")
    if getattr(args, "runtime_hours", None) is not None:
        hours = args.runtime_hours
        if not math.isfinite(hours) or not 0 < hours <= continuation_policy["max_runtime_hours"]:
            raise ValueError("Continuation runtime must be positive and within the saved maximum")
        funded_deadline = parent["rented_at_unix"] + (
            continuation_policy["max_total_usd"] - continuation_quote["bandwidth_reserved_usd"]
        ) / continuation_quote["hourly_usd"] * 3600
        deadline = min(time.time() + hours * 3600, funded_deadline)
    if time.time() >= deadline - parent["policy"]["backup_grace_seconds"]:
        raise ValueError("Saved training allowance expired; explicitly extend it before continuation")
    session = safe_name(args.new_session)
    cache = cache_dir(parent["branch"], session)
    if cache.exists():
        raise ValueError("Continuation already has records; inspect it and use handoff")
    selected = args.config.expanduser().resolve()
    selected.relative_to(ROOT)
    spec = json.loads(selected.read_text())
    if (spec.get("run_name") != session
            or not spec.get("skip_data_build") or not isinstance(spec.get("checkpoint"), dict)):
        raise ValueError("Cached continuation requires matching run_name, skip_data_build and a Drive checkpoint identity")
    evaluation_only = bool(spec.get("eval_only", False))
    pretraining = not evaluation_only and not spec.get("finetune_only", False)
    resumes_checkpoint = evaluation_only or pretraining or spec.get("finetune_resume", False)
    if pretraining and not isinstance(spec.get("pretrain"), dict):
        raise ValueError("Pretraining continuation requires an explicit pretrain stage")
    checkpoint_path = Path(args.reuse_checkpoint)
    relative_file(Path("/"), checkpoint_path)
    with lock(parent_cache / ".launch.lock"), lock(ROOT / "results" / f".vast-instance-{parent['instance_id']}.lock"):
        with tarfile.open(parent_cache / "source.tar.gz") as archive:
            original = json.load(archive.extractfile(parent["spec_relative"]))
        if ((not resumes_checkpoint and original.get("checkpoint") != spec["checkpoint"])
                or spec["checkpoint"].get("sha256") in (None, "drive")):
            raise ValueError("Continuation must reuse the exact checkpoint staged by its parent")
        if resumes_checkpoint and spec.get("dataset") != original.get("dataset"):
            raise ValueError("Cached checkpoint continuation must preserve the parent dataset")
        with tempfile.TemporaryDirectory(prefix="mdlarc-continuation-") as temp:
            archive = Path(temp) / "source.tar.gz"
            source, spec_bytes, added, resolved, updates, spec_changes = retry_source(
                parent_cache, parent, archive, args.source_update, replacement_spec=spec, cached_inputs=True)
            if pretraining and resolved["pretrain"] is None:
                raise ValueError("Pretraining continuation must contain a pretrain stage")
            if evaluation_only and (resolved["pretrain"] is not None or resolved["finetunes"]):
                raise ValueError("Evaluation continuation cannot contain training stages")
            if not evaluation_only and not pretraining and (not resolved["finetunes"] or resolved["pretrain"] is not None):
                raise ValueError("Continuation must contain only fine-tuning and evaluation stages")
            if args.dry_run:
                print(json.dumps({"continuation_of": parent["session"], "session": session,
                    "deadline_unix": deadline,
                    "resolved_experiment": resolved, "source_updates": updates, "source": source}, indent=2, default=str))
                return 0
            store = store_for(args, parent["branch"], config=parent["drive_config"])
            completion = manifest_from_drive(store, parent)
            backups_verified = completion and (completion.get("backups_complete") or (
                pretraining and parent["policy"].get("checkpoint_upload_mode") == "manual"
                and completion.get("record_backups_complete")))
            if not completion or not completion.get("finalized") or not backups_verified:
                raise ValueError("Continuation parent requires finalized, verified Drive backups")
            identity = completion["archive"]
            store._verify_binary(store.metadata(identity["id"]), parent["drive_folder_id"], identity["name"], identity)
            if resumes_checkpoint:
                checkpoint = spec["checkpoint"]
                metadata = store.metadata(checkpoint["drive_id"])
                staged_input = original.get("checkpoint") == checkpoint
                if staged_input:
                    if not store.branch_id:
                        store.preflight()
                    store.assert_under(checkpoint["drive_id"], store.branch_id)
                if ((not staged_input and metadata.get("parents") != [parent["drive_folder_id"]])
                        or metadata.get("name") != checkpoint["name"]
                        or int(metadata.get("size", -1)) != checkpoint["size"]
                        or (metadata.get("sha256Checksum") or metadata.get("appProperties", {}).get("sha256")) != checkpoint["sha256"]):
                    raise ValueError("Continuation checkpoint differs from the verified parent output")
            client = client_for(args)
            instance = client.instance(parent["instance_id"])
            if (instance.get("label") != parent["label"] or instance.get("actual_status") != "running"
                    or instance.get("intended_status") != "running"):
                raise ValueError("Continuation requires its original identified, powered-on instance")
            connection = SSH(instance, parent["ssh_key_path"], parent_cache / "known_hosts")
            verify_idle_instance(connection, parent)
            probe = """import pathlib,json,sys
p=pathlib.Path(sys.argv[1]); plan=json.loads((p/'launch.json').read_text()); sys.path.insert(0,str(p/'code'))
from workflows.vast.common import hashes,relative_file
run=p/'code/runs'/plan['run_name']; checkpoint=relative_file(run,sys.argv[2])
if not checkpoint.is_file(): raise ValueError('Saved on-VM checkpoint missing')
data=run/'data'; candidates=data.rglob('*') if sys.argv[3] == 'pretrain' else data.glob('*.json')
files=[{'path':f.relative_to(data).as_posix(),'identity':hashes(f)} for f in sorted(candidates) if f.is_file() and not f.is_symlink()]
tasks=json.loads((data/'priv_challenges.json').read_text())
aliases=set()
index_path=data/'variation_index.json'
if sys.argv[3] == 'pretrain' and index_path.exists():
    index=json.loads(index_path.read_text())
    for variants in index.get('parents',{}).values():
        for variant in variants:
            parts=pathlib.Path(variant['path']).parts
            if len(parts)>=5 and parts[0]=='runs' and parts[2]=='data': aliases.add('/'.join(parts[:3]))
if sys.argv[3] == 'finetune' and any('output' in q for t in tasks.values() for q in t.get('test',[])): raise ValueError('Query answers must be absent from adaptation challenges')
print(json.dumps({'checkpoint_path':sys.argv[2],'checkpoint_size':checkpoint.stat().st_size,'data_files':files,
'data_aliases':sorted(aliases),'training_tasks':len(tasks),'demonstration_pairs':sum(len(t['train']) for t in tasks.values()),'query_pairs':sum(len(t.get('test',[])) for t in tasks.values())}))
"""
            reuse = json.loads(connection.python(probe, parent["remote_workspace"], str(checkpoint_path),
                                                 "pretrain" if pretraining else "finetune", python=parent["policy"]["python"]))
            if reuse["checkpoint_size"] != spec["checkpoint"]["size"]:
                raise ValueError("Saved on-VM checkpoint size differs from its identity")
            cache.mkdir(parents=True, exist_ok=False)
            shutil.copyfile(archive, cache / "source.tar.gz")
            (cache / "spec.json").write_bytes(spec_bytes)
        store = store_for(args, parent["branch"], cache, config=parent["drive_config"])
        ancestor = {key: parent[key] for key in ("session", "branch", "instance_id", "remote_workspace", "drive_folder_id")}
        ancestor["ok"] = completion["ok"]
        import hashlib
        plan = {**parent, "session": session, "run_name": session, "source": source,
                "policy": continuation_policy, "quote": continuation_quote,
                "continuation_policy_overrides": policy_overrides,
                "continuation_deadline_at": getattr(args, "deadline_at", None),
                "deadline_unix": deadline, "previous_deadline_unix": parent["deadline_unix"],
                "continuation_runtime_hours": getattr(args, "runtime_hours", None),
                "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(), "source_updates": updates,
                "drive_folder_id": store.create_run(session), "label": f"mdlarc:{branch_folder(parent['branch'])}:{session}",
                "remote_workspace": parent["policy"]["remote_root"] + "/" + session,
                "continuation_of": ancestor, "retry_of": ancestor,
                "continuation_kind": "evaluation" if evaluation_only else "pretrain" if pretraining else "finetune",
                "skip_native_validation": bool(getattr(args, "skip_native_validation", False)),
                "spec_overrides": spec_changes,
                "retry_ancestors": [*parent.get("retry_ancestors", []), ancestor],
                "reuse_parent_inputs": reuse, "evaluation_only": evaluation_only,
                "adopted_instance": public_instance(instance),
                "intent": ("Evaluate the exact saved output checkpoint using cached data." if evaluation_only else
                           "Continue pretraining from the exact saved output checkpoint and cached data." if pretraining
                           else "Fine-tune from the exact checkpoint already on this VM using its provided demonstrations."),
                "changed_from_previous": f"Explicit continuation config {selected.relative_to(ROOT)}; cached checkpoint/data reused."}
        atomic_json(cache / "launch.json", plan)
        save_lease(cache, plan)
        save_state(cache, state="continuation_prepared")
        try:
            if client.instance(plan["instance_id"]).get("label") != parent["label"]:
                raise ValueError("Original instance identity changed before continuation")
            client.label(plan["instance_id"], plan["label"])
        except AmbiguousSubmission:
            if client.instance(plan["instance_id"]).get("label") != plan["label"]:
                raise
        if client.instance(plan["instance_id"]).get("label") != plan["label"]:
            raise ValueError("Continuation label is unconfirmed; inspect this saved session")
        handoff(args, cache, plan, client)
        print(f"{'Evaluation' if evaluation_only else 'Training'} continuation handoff confirmed on original instance {plan['instance_id']}; no checkpoint download.")
        return 0 if args.detach else watch(args, cache, plan)


def verify_idle_instance(connection, plan):
    """Read-only check before reusing a powered-on rental or uploading training."""
    probe = """import json,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]); current=pathlib.Path(sys.argv[2])
allowed={pathlib.Path(value) for value in json.loads(sys.argv[3])}
if any(p.parent.parent != current and p.parent.parent not in allowed for p in root.glob('*/private/started.json')):
    raise RuntimeError('Selected instance contains another managed run')
result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,timeout=15,check=True)
if result.stdout.strip(): raise RuntimeError('Selected GPU already has a compute process')
for proc in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
    try: args=proc.read_bytes().split(bytes([0]))
    except (FileNotFoundError,PermissionError,ProcessLookupError): continue
    for arg in args[1:]:
        value=pathlib.Path(arg.decode('utf-8','replace'))
        if value.name == 'launch_run.py' or (value.name == 'worker.py' and value.parent.name == 'vast'):
            raise RuntimeError('Selected instance still has a training or supervisor process')
"""
    ancestors = [*plan.get("retry_ancestors", []), *plan.get("previous_runs", [])]
    connection.python(probe, plan["policy"]["remote_root"], plan["remote_workspace"],
                      json.dumps([ancestor["remote_workspace"] for ancestor in ancestors]),
                      python=plan["policy"]["python"])


def ancestor_backups_verified(plan, ancestor, status):
    """An explicit pretraining continuation may reuse a separately verified output."""
    return bool(status.get("backups_complete") or (
        plan.get("continuation_kind") == "pretrain"
        and ancestor in [plan.get("continuation_of"), *plan.get("retry_ancestors", [])]
        and plan.get("policy", {}).get("checkpoint_upload_mode") == "manual"
        and status.get("record_backups_complete")))


def refresh_b300_controllers(cache, plan):
    """Before any upload, replace only stop-capable controllers, retaining provenance."""
    if not keep_b300_running(plan):
        return
    original = cache / "source.tar.gz"
    if hashes(original)["sha256"] != plan["source"]["sha256"]:
        raise ValueError("Saved source snapshot checksum differs from the launch")
    updates = []
    replacements = {}
    with tarfile.open(original, "r:gz") as archive:
        for name in ("workflows/vast/common.py", "workflows/vast/worker.py", "workflows/vast/vast_api.py"):
            before = archive.extractfile(name).read()
            after = relative_file(ROOT, name).read_bytes()
            if before != after:
                import hashlib
                updates.append({"path": name, "before_sha256": hashlib.sha256(before).hexdigest(),
                                "sha256": hashlib.sha256(after).hexdigest()})
                replacements[name] = after
    if replacements:
        previous = plan["source"]["sha256"]
        preserved = cache / f"source-before-b300-protection-{previous}.tar.gz"
        if not preserved.exists():
            os.link(original, preserved)
        atomic_json(cache / f"launch-before-b300-protection-{previous}.json", plan)
        with tempfile.TemporaryDirectory(prefix="mdlarc-b300-controller-") as temp:
            candidate = Path(temp) / "source.tar.gz"
            with tarfile.open(original, "r:gz") as src, tarfile.open(candidate, "w:gz", compresslevel=6) as dst:
                for member in src:
                    if not member.isfile():
                        raise ValueError("Source snapshot contains a non-file member")
                    if member.name in replacements:
                        data = replacements[member.name]
                        member.size = len(data)
                        dst.addfile(member, io.BytesIO(data))
                    else:
                        with src.extractfile(member) as data:
                            dst.addfile(member, data)
            # Every other file, including the pinned scientific spec and data, is unchanged.
            staged = cache / ".source-b300-protection.tar.gz"
            shutil.copyfile(candidate, staged)
            staged.replace(original)
        plan["source"] = {**plan["source"], **hashes(original)}
        plan.setdefault("controller_updates", []).append({"at": now(), "previous_source_sha256": previous,
                                                         "files": updates})
    plan["preserve_instance"] = True
    atomic_json(cache / "launch.json", plan)
    save_lease(cache, plan)


def retry(args):
    """Explicitly retry a finalized failure on its original rental and allowance."""
    parent_cache, parent = load_plan(args)
    requested_at = time.time()
    training_deadline = parent["deadline_unix"] - parent["policy"]["backup_grace_seconds"]
    if time.time() >= training_deadline:
        raise ValueError("Original training allowance expired; retry cannot extend it")
    session = safe_name(args.retry_session)
    replacement_spec = None
    restart_config = getattr(args, 'restart_config', None)
    if restart_config is not None:
        selected = restart_config.expanduser().resolve()
        if not selected.is_file() or restart_config.is_symlink():
            raise ValueError('Restart config must be a regular JSON file')
        replacement_spec = json.loads(selected.read_text())
        if replacement_spec.get('checkpoint') is not None or replacement_spec.get('eval_only') or replacement_spec.get('finetune_only'):
            raise ValueError('Fresh restart config must start training without a checkpoint')
    eval_checkpoint = None
    if getattr(args, "eval_checkpoint", None) is not None:
        selected = args.eval_checkpoint.expanduser()
        if selected.is_symlink() or not selected.is_file():
            raise ValueError("Evaluation checkpoint identity must be a regular JSON file")
        eval_checkpoint = json.loads(selected.read_text())
        if eval_checkpoint.get("source_branch", parent["branch"]) != parent["branch"]:
            raise ValueError("Evaluation checkpoint must belong to the saved run branch")
    backend = parent.get('backend', 'vast')
    if replacement_spec is not None and getattr(args, 'eval_checkpoint', None) is not None:
        raise ValueError('Fresh restart and evaluation retry are mutually exclusive')
    cache = cache_dir(parent["branch"], session, backend)
    if cache.exists():
        raise ValueError("Retry session already has records; inspect it and use handoff, never submit another retry blindly")
    with lock(parent_cache / ".launch.lock"), lock(ROOT / "results" / f".{backend}-instance-{parent['instance_id']}.lock"):
        # Validate the exact preserved source before any remote mutation.
        with tempfile.TemporaryDirectory(prefix="mdlarc-vast-retry-") as temp:
            archive = Path(temp) / "source.tar.gz"
            source, spec_bytes, added, resolved, updates, spec_changes = retry_source(
                parent_cache, parent, archive, getattr(args, 'source_update', ()),
                batch_size=getattr(args, 'batch_size', None),
                gradient_accumulation_steps=getattr(args, 'gradient_accumulation_steps', None),
                effective_batch_size=getattr(args, 'effective_batch_size', None),
                eval_checkpoint=eval_checkpoint, eval_run_name=session,
                eval_batch_size=getattr(args, 'eval_batch_size', None),
                eval_token_budget=getattr(args, 'eval_token_budget', None),
                replacement_spec=replacement_spec,
                skip_cuda_validation=getattr(args, 'skip_cuda_validation', False))
            import hashlib
            spec_sha256 = hashlib.sha256(spec_bytes).hexdigest()
            if args.dry_run:
                print(json.dumps({"retry_of": parent["session"], "session": session,
                    "instance_id": parent["instance_id"], "deadline_unix": parent["deadline_unix"],
                    "policy": parent["policy"], "spec_sha256": spec_sha256,
                    "parent_spec_sha256": parent["spec_sha256"], "spec_overrides": spec_changes,
                    "effective_batch_size_override": getattr(args, 'effective_batch_size', None),
                    "source": source, "added_source_files": added, "source_updates": updates,
                    "resolved_experiment": resolved}, indent=2, default=str))
                return 0
            client = client_for(args)
            instance = client.instance(parent["instance_id"])
            if instance.get("label") != parent["label"]:
                raise ValueError("Original instance label differs from the saved retry parent")
            stopped = (instance.get("actual_status") in {"stopped", "exited"}
                       and instance.get("intended_status") == "stopped")
            running = (instance.get("actual_status") == "running" and instance.get("intended_status") == "running")
            if backend == 'runpod':
                running = instance.get('intended_status') == 'running'
            if not stopped and not (running and keep_b300_running(parent, instance)):
                raise ValueError("Retry requires a stopped instance or a verified idle, powered-on B300")
            store = store_for(args, parent["branch"], config=parent["drive_config"])
            completion = manifest_from_drive(store, parent)
            if (not completion or completion.get("ok") is not False
                    or not completion.get("finalized") or not completion.get("backups_complete")):
                raise ValueError("Retry requires a finalized failure with verified Drive backups")
            # Verify the completed backup, including the checkpoint ledger. A retry
            # starts from scratch; checkpoint-bearing runs need a continuation spec.
            identity = completion["archive"]
            store._verify_binary(store.metadata(identity["id"]), parent["drive_folder_id"], identity["name"], identity)
            backup = Path(temp) / "results.tar.gz"
            store.download(identity["id"], backup, expected=identity, ancestor=parent["drive_folder_id"])
            recovered = Path(temp) / "results"
            extract_archive(backup, recovered, exclude_checkpoints=True)
            ledger_path = recovered / "checkpoint_backups.json"
            ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
            if eval_checkpoint is not None:
                matches = [entry for entry in ledger.get("files", []) if entry.get("id") == eval_checkpoint["drive_id"]]
                if (not ledger.get("complete") or ledger.get("errors") or len(matches) != 1
                        or any(matches[0].get(key) != eval_checkpoint[key] for key in ("name", "size", "sha256"))):
                    raise ValueError("Evaluation checkpoint differs from the verified parent backup ledger")
                entry = matches[0]
                store._verify_binary(store.metadata(entry["id"]), parent["drive_folder_id"], entry["name"], entry)
            elif ledger.get("files"):
                if replacement_spec is None:
                    raise ValueError("Original run has checkpoints; use an explicit checkpoint continuation or fresh restart config")
                if not ledger.get('complete') or ledger.get('errors'):
                    raise ValueError('Fresh restart requires complete parent checkpoint backups')
                for entry in ledger['files']:
                    store._verify_binary(store.metadata(entry['id']), parent['drive_folder_id'], entry['name'], entry)
            if time.time() >= training_deadline:
                raise ValueError("Original training allowance expired during verification")
            if running:
                connection = SSH(instance, parent["ssh_key_path"], parent_cache / "known_hosts")
                verify_idle_instance(connection, parent)
            cache.mkdir(parents=True, exist_ok=False)
            shutil.copyfile(archive, cache / "source.tar.gz")
            (cache / "spec.json").write_bytes(spec_bytes)
        store = store_for(args, parent["branch"], cache, config=parent["drive_config"])
        folder = store.create_run(session)
        ancestor = {key: parent[key] for key in ("session", "branch", "instance_id", "remote_workspace", "drive_folder_id")}
        plan = {**parent, "session": session, "source": source, "drive_folder_id": folder,
            "launch_requested_at_unix": requested_at,
            "run_name": json.loads(spec_bytes).get('run_name', parent['run_name']),
            "fresh_restart": replacement_spec is not None,
            "evaluation_only": eval_checkpoint is not None,
            "evaluation_checkpoint": eval_checkpoint,
            "skip_native_validation": bool(getattr(args, 'skip_native_validation', False)),
            "spec_sha256": spec_sha256, "spec_overrides": spec_changes,
            "effective_batch_size_override": getattr(args, 'effective_batch_size', None),
            "label": parent['label'] if backend == 'runpod' else f"mdlarc:{branch_folder(parent['branch'])}:{session}",
            "remote_workspace": parent["policy"]["remote_root"] + "/" + session,
            "adopted_instance": public_instance(instance),
            "retry_of": ancestor, "retry_ancestors": [*parent.get("retry_ancestors", []), ancestor],
            "added_source_files": added,
            "source_updates": updates,
            "intent": parent["intent"],
            "changed_from_previous": "Explicit retry of finalized failure; original allowance preserved; "
                f"evaluation_only={eval_checkpoint is not None}; spec overrides: {spec_changes}; added source files: {added}; "
                f"explicitly updated source files: {[value['path'] for value in updates]}."}
        atomic_json(cache / "launch.json", plan)
        if backend in {'runpod', 'vast'} and not plan.get('source_inputs'):
            family = json.loads(spec_bytes).get('dataset', {}).get('pretrain_data', {})
            if family.get('kind') == 'family_sampled':
                plan['source_inputs'] = {'family_catalog': family['catalog']}
                atomic_json(cache / 'launch.json', plan)
        if getattr(args, 'skip_cuda_validation', False):
            plan['runtime_validated_from'] = parent.get('runtime_validated_from') or ancestor
            atomic_json(cache / 'launch.json', plan)
        save_lease(cache, plan)
        (cache / "note.md").write_text(f"# {session}\n\nRetry of: {parent['session']}\n\nOriginal rental deadline retained. Added source files: {', '.join(added) or 'none'}.\n\nSource fixes: {', '.join(value['path'] for value in updates) or 'none'}.\n\nEvaluation only: {eval_checkpoint is not None}. Spec overrides: {spec_changes}.\n")
        save_state(cache, state="retry_prepared", instance_id=plan["instance_id"])
        # Relabel only the saved rental. Ambiguous responses are inspected, not resubmitted.
        try:
            if client.instance(parent["instance_id"]).get("label") != parent["label"]:
                raise ValueError("Original rental ownership changed before retry")
            if backend != 'runpod':
                client.label(plan["instance_id"], plan["label"])
        except AmbiguousSubmission:
            if client.instance(plan["instance_id"]).get("label") != plan["label"]:
                raise
        if client.instance(plan["instance_id"]).get("label") != plan["label"]:
            raise ValueError("Retry label could not be verified; inspect the saved retry records")
        if not running:
            save_state(cache, state="retry_start_requested")
            try:
                client.start(plan["instance_id"])
            except ApiError:
                save_state(cache, state="retry_start_unconfirmed")
                raise RuntimeError("Original instance start is unconfirmed; inspect this retry session before using start or handoff") from None
        handoff(args, cache, plan, client)
        print(f"Retry handoff confirmed on instance {plan['instance_id']}; session {session}. Original rental deadline retained.")
        return 0 if args.detach else watch(args, cache, plan)


def list_templates(args):
    base = policies.load(args.policy)
    rows = [{"name": "saved", "label": "Saved policy", "policy": base}]
    rows += [{"name": name, "label": label, "policy": policies.apply_template(base, name)}
             for name, label in policies.GPU_TEMPLATES.items()]
    for row in rows:
        row["image_rules"] = policies.image_rules(row["policy"])
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    for row in rows:
        policy = row["policy"]
        memory = f">= {policy['min_gpu_memory_gb']:g} GB"
        if policy["max_gpu_memory_gb"] is not None:
            memory = f"{policy['min_gpu_memory_gb']:g}-{policy['max_gpu_memory_gb']:g} GB"
        cuda = f">= {policy['min_cuda_version']:g}" if policy["min_cuda_version"] is not None else "unset"
        selection = "CUDA-matched NGC" if row["image_rules"] else "fixed"
        print(f"{row['name']}: {row['label']}; {policy['num_gpus']} GPU; memory {memory}; CUDA {cuda}; {policy['image']} ({selection})")
    print("Templates inherit the selected policy's CPU, disk, price, runtime, transfer, location, and cleanup limits.")
    return 0


def launch(args):
    if args.instance_id is not None and args.instance_id <= 0:
        raise ValueError("Instance ID must be a positive integer")
    branch = branch_name(args.branch)
    spec_path = args.config.expanduser().resolve()
    spec_path.relative_to(ROOT)
    policy = launch_policy(args)
    spec = json.loads(spec_path.read_text())
    expected_spec_sha256 = hashes(spec_path)["sha256"]
    trial = trial_provenance(args, spec_path, spec)
    resolved = preview(spec_path)
    from launch_run import validate_distributed_gpu_count
    validate_distributed_gpu_count(resolved, policy["num_gpus"])
    run_name = safe_name(spec.get("run_name") or spec_path.stem)
    session = safe_name(args.session or f"vast-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:8]}")
    with tempfile.TemporaryDirectory(prefix="mdlarc-vast-preview-") as temp:
        info = snapshot.create(ROOT, spec_path, Path(temp) / "source.tar.gz")
    if args.dry_run:
        print(json.dumps({"backend": "vast", "branch": branch, "session": session, "run_name": run_name,
                          "policy": policy, "gpu_template": args.gpu_template,
                          "policy_overrides": policies.parse_overrides(args.policy_override),
                          "image_rules": policies.image_rules(policy),
                          "query": policies.offer_query(policy), "source": info,
                          "existing_instance_id": args.instance_id,
                          "resolved_experiment": resolved}, indent=2, default=str))
        return 0
    identity = args.ssh_key.expanduser().resolve()
    if not identity.is_file() or identity.is_relative_to(ROOT):
        raise ValueError("Configure a private SSH key outside the repository")
    public_key(identity)  # Reject a missing/invalid paired public key before renting.
    if args.secrets_file:
        validate_training_secrets(args.secrets_file)
    intent, changes = notes(args, run_name)
    cache = cache_dir(branch, session)
    cache.mkdir(parents=True, exist_ok=False)
    instance_lock = (lock(ROOT / "results" / f".vast-instance-{args.instance_id}.lock")
                     if args.instance_id is not None else nullcontext())
    with lock(cache / ".launch.lock"), instance_lock:
        store = store_for(args, branch, cache)
        store.preflight()
        client = client_for(args)
        existing = None
        previous_runs = []
        if args.instance_id is not None:
            existing = client.instance(args.instance_id)
            previous_runs = verified_previous_b300_runs(args, existing)
            policy, quote = policies.existing_instance(existing, policy, verified_previous_runs=bool(previous_runs))
            rented_at = existing_rental_start(existing, policy)
            if previous_runs:
                connection = SSH(existing, identity, cache / "known_hosts")
                verify_idle_instance(connection, {"policy": policy,
                    "remote_workspace": policy["remote_root"] + "/" + session,
                    "previous_runs": previous_runs})
            offer = existing
            print(f"Using existing instance {args.instance_id}; {offer['gpu_name']}; actual container {policy['image']}; allocated disk {policy['disk_gb']:g} GB", flush=True)
        else:
            quote, offer = policies.choose_offer(client.offers(policies.offer_query(policy)), policy)
            policy = policies.resolve_image(policy, offer)
            rented_at = time.time()
            print(f"Selected {offer['gpu_name']} with CUDA support {offer['cuda_max_good']:g}; container {policy['image']}", flush=True)
        source = snapshot.create(ROOT, spec_path, cache / "source.tar.gz")
        source_spec = None
        import tarfile
        with tarfile.open(cache / "source.tar.gz", "r:gz") as archive:
            source_spec = archive.extractfile(spec_path.relative_to(ROOT).as_posix()).read()
        (cache / "spec.json").write_bytes(source_spec)
        if hashes(cache / "spec.json")["sha256"] != expected_spec_sha256:
            raise ValueError("Experiment spec changed between validation and packaging; rerun the preview")
        folder = store.create_run(session)
        if existing is None:
            rented_at = time.time()
        plan = {"schema_version": 1, "backend": "vast", "branch": branch, "session": session, "run_name": run_name,
                "spec_relative": spec_path.relative_to(ROOT).as_posix(), "spec_sha256": hashes(cache / "spec.json")["sha256"],
                "source": source, "policy": policy, "gpu_name": offer["gpu_name"], "gpu_template": args.gpu_template,
                "policy_overrides": policies.parse_overrides(args.policy_override),
                "drive_config": store.config, "drive_folder_id": folder,
                "offer_id": offer.get("bundle_id") if existing else offer["id"], "quote": quote, "label": f"mdlarc:{branch_folder(branch)}:{session}",
                "runtime_compatibility": {"min_cuda_version": policy["min_cuda_version"],
                    "image": policy["image"],
                    "image_selection": "cuda" if policies.image_rules(policy) else "fixed",
                    "offer_cuda_max_good": offer["cuda_max_good"], "offer_driver_version": offer.get("driver_version")},
                "instance_id": args.instance_id, "rented_at_unix": rented_at,
                "deadline_unix": rented_at + policy["max_runtime_hours"] * 3600,
                "remote_workspace": policy["remote_root"] + "/" + session,
                "intent": intent, "changed_from_previous": changes,
                "ssh_key_path": str(identity)}
        if existing:
            plan["adopted_instance"] = public_instance(existing)
        if previous_runs:
            plan["previous_runs"] = previous_runs
        if trial:
            plan["trial"] = {"trial_id": trial.trial_id, "record": json.loads(trial.trial_bytes),
                             "template": json.loads(trial.template_bytes)}
        atomic_json(cache / "launch.json", plan)
        # Export the label/deadline before the possibly ambiguous rental submission.
        save_lease(cache, plan, args.lease_export_dir)
        note = cache / "note.md"
        note.write_text(f"# {session}\n\nRun: {run_name}\n\nIntent: {intent}\n\nChanged from previous: {changes}\n")
        save_state(cache, state="existing_instance_claiming" if existing else "rental_submitting")
        try:
            if existing:
                if client.instance(args.instance_id).get("label") != existing.get("label"):
                    raise RuntimeError("Existing instance ownership changed during preparation; no training was submitted")
                try:
                    client.label(args.instance_id, plan["label"])
                except AmbiguousSubmission:
                    # Inspect this exact instance; never rent or resend training.
                    if client.instance(args.instance_id).get("label") != plan["label"]:
                        raise
                if client.instance(args.instance_id).get("label") != plan["label"]:
                    raise RuntimeError("Existing instance label could not be verified; no training was submitted")
            else:
                plan["instance_id"] = client.create(offer["id"], policy, plan["label"])
        except AmbiguousSubmission:
            save_state(cache, state="adoption_unconfirmed" if existing else "rental_unconfirmed")
            raise RuntimeError(f"Rental outcome is unknown. Run reconcile --session {session}; do not launch again") from None
        except Exception:
            save_state(cache, state="rental_rejected")
            raise
        atomic_json(cache / "launch.json", plan)
        save_lease(cache, plan, args.lease_export_dir)
        save_state(cache, state="instance_loading", instance_id=plan["instance_id"])
        print(f"{'Adopted' if existing else 'Rented'} instance {plan['instance_id']} for {session}. Estimated allowance: ${quote['estimated_max_usd']:.2f}", flush=True)
        try:
            handoff(args, cache, plan, client)
        except BaseException:
            phase = json.loads((cache / "state.json").read_text())["state"]
            save_state(cache, state="handoff_needs_recovery", phase_before_error=phase)
            # A transport failure after dispatch must not interrupt a possibly running job.
            if phase in {"instance_loading", "uploading"}:
                try:
                    result = stop_owned(client, plan)
                    save_state(cache, disposition="keep_b300_running" if result == "keep_b300_running" else "stop_requested")
                except Exception:
                    save_state(cache, disposition="stop_unconfirmed")
            raise
    print(f"Remote handoff confirmed. You can close the laptop. Recover later with: run.py sync --branch {branch} --session {session}", flush=True)
    return 0 if args.detach else watch(args, cache, plan)


def handoff(args, cache, plan, client=None):
    client = client or client_for(args)
    deadline = min(time.time() + plan["policy"]["startup_timeout_seconds"], plan["deadline_unix"])
    connection = None
    key_attached = False
    key_attachment_error = None
    while time.time() < deadline:
        instance = client.instance(plan["instance_id"])
        if instance.get("label") != plan["label"]:
            raise ValueError("Remote instance label differs from saved rental identity")
        if plan.get("backend") == "runpod":
            from workflows.runpod.policy import verify_pod
            try:
                verify_pod(client.raw_pod(plan["instance_id"]), plan["policy"], plan["label"],
                           adopted=bool(plan.get("adopted_instance")))
            except ValueError:
                stop_owned(client, plan)
                save_state(cache, disposition="policy_stop_requested")
                raise
        if plan.get("backend") == "sfcompute":
            from workflows.sfcompute.policy import verify_node
            try:
                verify_node(client.raw_node(plan["instance_id"]), plan["policy"],
                            plan["label"], plan["instance_id"], ready=instance.get("actual_status") == "running")
            except ValueError:
                stop_owned(client, plan)
                save_state(cache, disposition="policy_stop_requested")
                raise
        save_state(cache, instance=public_instance(instance))
        try:
            connection = connection_for(client, instance, plan, cache)
        except (RuntimeError, ValueError):
            time.sleep(5)
            continue
        try:
            # Existing rentals and reconnects can already have this key. Vast
            # may reject a duplicate attachment even though SSH works.
            connection.python("print('ready')", python=plan["policy"]["python"])
            break
        except (RuntimeError, ValueError):
            if not key_attached and plan.get("backend", "vast") == "vast":
                # Team accounts require per-instance registration. Attempt it
                # once, then observe SSH even if the API response is rejected
                # or lost; never repeatedly attach an already installed key.
                try:
                    client.attach_ssh_key(plan["instance_id"], public_key(plan["ssh_key_path"]))
                except ApiError as exc:
                    key_attachment_error = str(exc)
                    save_state(cache, ssh_key_attachment="unconfirmed", ssh_key_attachment_error=key_attachment_error)
                key_attached = True
            time.sleep(5)
    else:
        if key_attachment_error:
            raise RuntimeError("Instance did not become SSH-ready; Vast SSH-key registration could not be confirmed")
        raise RuntimeError("Instance did not become SSH-ready within its allowance")
    python, remote = plan["policy"]["python"], plan["remote_workspace"]
    existing = connection.json_file(remote + "/private/started.json", python=python)
    if existing:
        save_state(cache, state="worker_already_started")
        return
    if keep_b300_running(plan, instance):
        plan["gpu_name"] = instance.get("gpu_name") or plan.get("gpu_name")
        # Uploaded source is immutable. Never silently replace a dispatched worker.
        uploaded = connection.python("import pathlib,sys; print(int((pathlib.Path(sys.argv[1])/'source.tar.gz').exists()))",
                                     remote, python=python).strip()
        if uploaded == b"0":
            refresh_b300_controllers(cache, plan)
        elif not plan.get("preserve_instance"):
            raise ValueError("Existing B300 source lacks no-stop protection; inspect its saved handoff before proceeding")
    ancestors = plan.get("retry_ancestors", [])
    if ancestors:
        for ancestor in ancestors:
            status = connection.json_file(ancestor["remote_workspace"] + "/status.json", python=python)
            if (not status or any(status.get(key) != ancestor[key] for key in ("branch", "session", "instance_id"))
                    or not status.get("finalized") or not ancestor_backups_verified(plan, ancestor, status)
                    or status.get("ok") is not ancestor.get("ok", False)):
                raise ValueError("Retry ancestor is not a finalized, backed-up failure on this instance")
        totals = connection.json_file(ancestors[-1]["remote_workspace"] + "/private/transfers.json", python=python)
        if (not isinstance(totals, dict) or set(totals) != {"upload", "download"}
                or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in totals.values())):
            raise ValueError("Original transfer allowance usage is unavailable; retry cannot reset it")
        plan["prior_transfers"] = totals
        atomic_json(cache / "launch.json", plan)
    if plan.get("adopted_instance"):
        verify_idle_instance(connection, plan)
    if plan.get("backend") == "runpod":
        from workflows.runpod.api import CONTROL_CHECK
        # Preserved adoptions use the API-verified SSH endpoint and never need
        # a remote lifecycle key. Account credentials remain on the launcher.
        if not (plan.get("adopted_instance") and plan.get("preserve_instance") is True):
            connection.python(CONTROL_CHECK, str(plan["instance_id"]), python=python)
    elif plan.get("backend") == "sfcompute":
        # Metadata is node-scoped and needs no account token or SSH credential.
        connection.python("import json,sys,urllib.request; o=urllib.request.build_opener(urllib.request.ProxyHandler({})); r=urllib.request.Request('http://169.254.42.1/v1/node',headers={'Metadata-Flavor':'givemeanode'}); v=json.load(o.open(r,timeout=20)); n=v.get('node',v); assert n.get('id',n.get('node_id')) == sys.argv[1] and n.get('name') == sys.argv[2], 'Node metadata identity differs'", str(plan["instance_id"]), plan["label"], python=python)
    else:
        connection.python("import os,pathlib,sys; d=dict(os.environ); p=pathlib.Path('/root/.mdlarc-vast-instance.env'); d.update(dict(line.split('=',1) for line in p.read_text().splitlines()) if p.is_file() else {}); assert d.get('CONTAINER_API_KEY') and d.get('CONTAINER_ID') == sys.argv[1], 'Instance control credentials unavailable'", str(plan["instance_id"]), python=python)
    connection.python("import os,pathlib,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(parents=True,exist_ok=True,mode=0o700); (p/'private').mkdir(exist_ok=True,mode=0o700)", remote, python=python)
    save_state(cache, state="uploading")
    connection.upload(cache / "source.tar.gz", remote + "/source.tar.gz", python=python)
    connection.upload(cache / "launch.json", remote + "/launch.json", python=python)
    token = args.drive_token.expanduser().resolve()
    if token.is_relative_to(ROOT):
        raise ValueError("Vast token must remain outside the repository")
    connection.upload(token, remote + "/private/drive-token.json", python=python)
    if args.secrets_file:
        validate_training_secrets(args.secrets_file)
        connection.upload(args.secrets_file, remote + "/private/training-secrets.json", python=python)
    unpack = """import hashlib,pathlib,shutil,sys,tarfile
p=pathlib.Path(sys.argv[1]); expected=sys.argv[2]
with (p/'source.tar.gz').open('rb') as src:
    h=hashlib.sha256()
    for b in iter(lambda:src.read(8388608),b''): h.update(b)
if h.hexdigest()!=expected: raise ValueError('Source checksum mismatch')
code=p/'code'
if not code.exists():
    staging=p/'code-staging'; shutil.rmtree(staging,ignore_errors=True); staging.mkdir()
    with tarfile.open(p/'source.tar.gz','r:gz') as archive:
        seen=set()
        for member in archive:
            name=pathlib.Path(member.name)
            if not member.isfile() or name.is_absolute() or '..' in name.parts or '\\\\' in member.name or member.name in seen: raise ValueError('Unsafe source member')
            seen.add(member.name); target=staging/name; target.parent.mkdir(parents=True,exist_ok=True)
            with archive.extractfile(member) as src, target.open('xb') as out: shutil.copyfileobj(src,out)
    staging.rename(code)
"""
    connection.python(unpack, remote, plan["source"]["sha256"], python=python, timeout=300)
    input_bytes = 0
    for name, identity in plan.get('source_inputs', {}).items():
        spec = json.loads((cache / 'spec.json').read_text())
        if name != 'family_catalog' or identity != spec['dataset']['pretrain_data']['catalog']:
            raise ValueError('Saved external input differs from the scientific spec')
        blob = plan['policy']['remote_root'] + '/.inputs/' + identity['sha256'] + '.zip'
        # Upgrade older kept pods without uploading their existing pinned ZIP
        # from the laptop again. Only verified ancestor workspaces are eligible.
        seed = """import json,os,pathlib,sys
blob=pathlib.Path(sys.argv[1]); expected=json.loads(sys.argv[2]); pod=sys.argv[3]
if not blob.exists():
 for previous in json.loads(sys.argv[4]):
    root=pathlib.Path(previous['remote_workspace'])
    if not (root/'launch.json').is_file():continue
    saved=json.loads((root/'launch.json').read_text())
    if str(saved['instance_id'])!=pod:raise ValueError('Input cache owner differs')
    if not (root/'code'/saved['spec_relative']).is_file():continue
    spec=json.loads((root/'code'/saved['spec_relative']).read_text())
    if (spec.get('dataset',{}).get('pretrain_data',{}).get('catalog'))!=expected:continue
    source=root/'code'/expected['archive_path']
    if source.is_file() and not source.is_symlink():
        blob.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        os.link(source,blob); break
"""
        candidates = [*plan.get('previous_runs', []), *plan.get('retry_ancestors', [])]
        if candidates:
            connection.python(seed, blob, json.dumps(identity), str(plan['instance_id']), json.dumps(candidates), python=python)
        if connection.cache_input(ROOT / identity['archive_path'], blob, identity, python=python):
            input_bytes += identity['size']
        materialize = """import os,pathlib,sys
code=pathlib.Path(sys.argv[1]); p=code/sys.argv[2]; blob=pathlib.Path(sys.argv[3])
if not p.resolve().is_relative_to(code.resolve()):raise ValueError('Invalid input target')
p.parent.mkdir(parents=True,exist_ok=True)
if not p.exists():os.link(blob,p)
elif p.is_symlink() or not os.path.samefile(blob,p):raise ValueError('Existing input is not the pinned cache link')
"""
        connection.python(materialize, remote + '/code', identity['archive_path'], blob, python=python)
    transfers = {**plan.get("prior_transfers", {"upload": 0, "download": 0})}
    transfers["download"] += plan["source"]["size"] + input_bytes
    connection.python("import json,pathlib,sys; p=pathlib.Path(sys.argv[1])/'private/transfers.json'; p.write_text(sys.argv[2]) if not p.exists() else None", remote, json.dumps(transfers), python=python)
    save_state(cache, state="worker_submitting")
    # If this response is lost, the next action is observation only.
    try:
        connection.launch(remote, python)
    except RuntimeError:
        save_state(cache, state="worker_submission_unconfirmed")
    observation_deadline = min(time.time() + 120, plan["deadline_unix"])
    while time.time() < observation_deadline:
        status = connection.json_file(remote + "/status.json", python=python)
        if status:
            save_state(cache, state=status["state"], remote_status=status)
            if status["state"] in {"needs_recovery", "complete"}:
                if status["state"] == "needs_recovery":
                    raise RuntimeError("Remote worker failed during setup; inspect status and recover artifacts")
                return
            if status["state"] in {"preflighting_runtime", "preparing_inputs", "running", "preparing_data", "data_ready", "model_setup", "compiling_training", "training"}:
                return
        time.sleep(2)
    raise RuntimeError("Worker handoff is not confirmed; observe this instance before taking further action")


def validate_training_secrets(path):
    path = Path(path).expanduser().resolve()
    if path.is_relative_to(ROOT):
        raise ValueError("Training secrets must be outside the repository")
    value = json.loads(path.read_text())
    allowed = {"KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY", "HF_TOKEN"}
    if not isinstance(value, dict) or set(value) - allowed or not all(isinstance(v, str) for v in value.values()):
        raise ValueError("Training secrets may contain only Kaggle or Hugging Face credentials")
    return value


def reconcile(args):
    cache, plan = load_plan(args)
    client = client_for(args)
    matches = [i for i in client.instances() if i.get("label") == plan["label"]]
    if len(matches) != 1:
        save_state(cache, state="rental_unconfirmed", matching_instances=len(matches))
        raise RuntimeError("Rental cannot be uniquely reconciled; no instance was created or training submitted")
    if plan["instance_id"] is not None and plan["instance_id"] != matches[0]["id"]:
        raise ValueError("Reconciled instance differs from the original identity")
    plan["instance_id"] = matches[0]["id"]
    atomic_json(cache / "launch.json", plan)
    atomic_json(cache / "lease.json", {k: plan[k] for k in ("instance_id", "label", "deadline_unix", "branch", "session")})
    save_state(cache, state="rental_reconciled", instance=public_instance(matches[0]))
    print(f"Reconciled instance {plan['instance_id']}. No training was submitted. Use handoff if setup was incomplete.")
    return 0


def manifest_from_drive(store, plan):
    folder = plan["drive_folder_id"]
    store.preflight()
    store.verify_run(folder, plan["session"])
    matches = store.children(folder, name="completion.json")
    if len(matches) > 1:
        raise ValueError("Duplicate completion manifests; inspect the run folder")
    items = store.children(folder)
    recovered = [item for item in items if item["name"].startswith("completion-recovered-") and item["name"].endswith(".json")]
    final_candidates = matches + recovered
    candidates = final_candidates or [item for item in items if item["name"].startswith("live-") and item["name"].endswith(".json")]
    if not candidates:
        return None
    if all(item.get("createdTime") for item in candidates):
        candidates = [max(candidates, key=lambda item: item["createdTime"])]
    # Use provider creation timestamps, then validate the selected pointer's bytes.
    values = []
    with tempfile.TemporaryDirectory(prefix="mdlarc-vast-manifest-") as temp:
        for index, item in enumerate(candidates):
            path = Path(temp) / f"{index}.json"
            store.download(item["id"], path, ancestor=folder)
            values.append(json.loads(path.read_text()))
    manifest = max(values, key=lambda item: item["finished_at"] if final_candidates else item["at"])
    if final_candidates and (manifest.get("session") != plan["session"] or manifest.get("branch") != plan["branch"] or manifest.get("instance_id") != plan["instance_id"]):
        raise ValueError("Drive completion manifest differs from saved launch identity")
    return manifest


def sync(args, cache=None, plan=None, *, quiet=False):
    if cache is None:
        cache, plan = load_plan(args)
    store = store_for(args, plan["branch"], config=plan["drive_config"])
    manifest = manifest_from_drive(store, plan)
    if manifest is None:
        if not quiet:
            print("No artifact snapshot has been backed up yet.")
        return None
    identity = manifest["archive"]
    store._verify_binary(store.metadata(identity["id"]), plan["drive_folder_id"], identity["name"], identity)
    destination = cache / "artifacts" / identity["sha256"]
    if not destination.exists():
        with tempfile.TemporaryDirectory(prefix=".recover-", dir=cache) as temp:
            path = Path(temp) / "results.tar.gz"
            store.download(identity["id"], path, expected=identity, ancestor=plan["drive_folder_id"])
            staging = Path(temp) / "files"
            extract_archive(path, staging, exclude_checkpoints=True)
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging.rename(destination)
    atomic_json(cache / "latest_artifacts.json", {"directory": str(destination), "manifest": manifest})
    save_state(cache, local_recovery="verified", latest_artifact_sha256=identity["sha256"])
    if not quiet:
        print(f"Recovered non-checkpoint artifacts to {destination}")
    return manifest


def watch(args, cache=None, plan=None):
    if cache is None:
        cache, plan = load_plan(args)
    with lock(cache / ".watch.lock"):
        client = client_for(args)
        missing, offset, last_sync = 0, 0, 0
        log_path = cache / "observed-console.log"
        if log_path.exists():
            offset = log_path.stat().st_size
        while True:
            try:
                manifest = sync(args, cache, plan, quiet=True) if time.monotonic() - last_sync >= args.poll_seconds else None
                last_sync = time.monotonic()
                if manifest and manifest.get("finalized"):
                    save_state(cache, state=manifest["state"], completion=manifest)
                    print(f"Run finalized: {manifest['state']}. Drive backups: {manifest['backups_complete']}")
                    return 0 if manifest["ok"] and manifest["backups_complete"] else 1
            except ValueError:
                raise  # Identity/checksum violations are not transient observations.
            except Exception as exc:
                save_state(cache, drive_observation="unknown", drive_error=type(exc).__name__)
                print("Drive observation unavailable; continuing to check the original instance.", flush=True)
            try:
                listing = client.instances()
                matches = [i for i in listing if i.get("id") == plan["instance_id"]]
                if not matches:
                    missing += 1
                    if missing >= 2:
                        save_state(cache, state="unavailable")
                        print("Original instance is unavailable. Existing Drive backups and local records are preserved.")
                        return 1
                else:
                    missing = 0
                    instance = matches[0]
                    if instance.get("label") != plan["label"]:
                        raise ValueError("Original instance identity changed")
                    if instance.get("actual_status") == "stopped":
                        save_state(cache, state="stopped", instance=public_instance(instance))
                        print("Original instance is stopped with its disk retained. Use sync for available Drive artifacts or start/recover-backups for pending uploads.")
                        return 1
                    connection = connection_for(client, instance, plan, cache)
                    status = connection.json_file(plan["remote_workspace"] + "/status.json", python=plan["policy"]["python"])
                    if status:
                        save_state(cache, state=status["state"], remote_status=status, instance=public_instance(instance))
                    console = connection.logs(plan["remote_workspace"] + "/code/runs/" + plan["run_name"] + "/console.log", offset, python=plan["policy"]["python"])
                    data = base64.b64decode(console["data"])
                    if data:
                        with log_path.open("ab") as log:
                            log.write(data)
                        print(data.decode("utf-8", "replace"), end="", flush=True)
                        offset = console["offset"]
                    if status and status["state"] == "needs_recovery" and instance.get("actual_status") == "stopped":
                        print("Run needs backup recovery; original instance disk was retained.")
                        return 1
                if time.time() >= plan["deadline_unix"]:
                    result = stop_owned(client, plan)
                    save_state(cache, disposition="keep_b300_running" if result == "keep_b300_running" else "deadline_stop_requested")
            except (ApiError, RuntimeError) as exc:
                missing = 0  # Network/auth failure is unknown, never evidence of deletion.
                save_state(cache, observation="unknown", observation_error=type(exc).__name__)
                print("Observation unavailable; retrying without resubmitting training.", flush=True)
            time.sleep(args.poll_seconds)


def stop_owned(client, plan, *, explicit=False):
    instance = client.instance(plan["instance_id"])
    if instance.get("label") != plan["label"]:
        raise ValueError("Refusing to stop an instance outside this saved run")
    if not explicit and keep_b300_running(plan, instance):
        return "keep_b300_running"
    return client.stop(plan["instance_id"])


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    templates = commands.add_parser("templates", help="List saved policy and built-in GPU templates without renting")
    templates.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    templates.add_argument("--json", action="store_true")
    for name in ("launch", "retry", "continue", "watch", "sync", "status", "reconcile", "handoff", "recover-backups", "start", "stop"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--branch")
        cmd.add_argument("--session", required=name != "launch")
        cmd.add_argument("--drive-token", type=Path, default=DEFAULT_TOKEN)
        cmd.add_argument("--drive-config", type=Path, default=Path(__file__).with_name("drive.example.json"))
        cmd.add_argument("--api-key-file", type=Path)
        cmd.add_argument("--poll-seconds", type=int, default=30)
        cmd.add_argument("--secrets-file", type=Path)
        if name == "continue":
            cmd.add_argument("--config", type=Path, required=True)
            cmd.add_argument("--new-session", required=True)
            cmd.add_argument("--reuse-checkpoint", required=True, help="Checkpoint path relative to the parent run directory")
            allowance = cmd.add_mutually_exclusive_group()
            allowance.add_argument("--runtime-hours", type=float,
                             help="Bounded window for this newly requested continuation, capped by the saved total budget including rental time already used")
            allowance.add_argument("--deadline-at", help="Explicit ISO-8601 hard deadline, including timezone; backup grace is reserved before it")
            cmd.add_argument("--policy-override", action="append", default=[], metavar="FIELD=JSON_VALUE",
                             help="Explicitly amend continuation runtime, total budget or backup allowances; preserves the parent launch record")
            cmd.add_argument("--source-update", type=Path, action="append", default=[])
            cmd.add_argument("--dry-run", action="store_true")
            cmd.add_argument("--detach", action="store_true")
            cmd.add_argument("--skip-native-validation", action="store_true")
        if name == "retry":
            cmd.add_argument("--retry-session", required=True, help="Fresh session name for this explicit retry on the original rental")
            cmd.add_argument('--restart-config', type=Path,
                             help='Explicit fresh restart using this replacement spec; preserves parent checkpoint backups')
            cmd.add_argument("--dry-run", action="store_true", help="Validate the preserved source/spec without network or remote changes")
            cmd.add_argument("--detach", action="store_true")
            cmd.add_argument("--skip-native-validation", action="store_true",
                             help="Explicit user waiver of repeated native CUDA/FA4 tests for this retry; records a skipped result, never a validation pass")
            cmd.add_argument("--source-update", type=Path, action="append", default=[],
                             help="Replace this repository-relative training Python file or workflows/vast/worker.py from the current checkout; repeat for explicit fixes. Preserves the exact saved spec, dataset, policy and rental deadline")
            cmd.add_argument("--batch-size", type=int,
                             help="Explicit pretraining physical batch override; requires accumulation and preserves effective batch")
            cmd.add_argument("--gradient-accumulation-steps", type=int,
                             help="Explicit pretraining accumulation override; requires batch size and preserves effective batch")
            cmd.add_argument("--effective-batch-size", type=int,
                             help="Explicitly authorize this effective batch when changing it; must equal batch size times accumulation")
            cmd.add_argument("--eval-checkpoint", type=Path,
                             help="Rerun only evaluation using this exact Drive checkpoint identity JSON; verified against the parent backup ledger")
            cmd.add_argument("--eval-batch-size", type=int,
                             help="Evaluation-only retry batch cap; requires --eval-checkpoint")
            cmd.add_argument("--eval-token-budget", type=int,
                             help="Evaluation-only retry padded prefill token budget; requires --eval-checkpoint")
        if name == "launch":
            cmd.add_argument("--config", type=Path, required=True)
            cmd.add_argument("--instance-id", type=int, help="Use this already rented, unclaimed instance; never create a rental")
            cmd.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
            cmd.add_argument("--policy-override", action="append", default=[], metavar="FIELD=JSON_VALUE",
                             help="Override one infrastructure field for this launch; repeat for multiple fields. Applied after template/image options and recorded without changing the saved policy")
            cmd.add_argument("--gpu-template", choices=policies.GPU_TEMPLATES,
                             help="Apply a GPU template to the saved infrastructure policy")
            cmd.add_argument("--image")
            cmd.add_argument("--image-selection", choices=("cuda", "fixed"),
                             help="CUDA-match supported NGC tags or pin the policy image; --image pins by default")
            cmd.add_argument("--ssh-key", type=Path, default=DEFAULT_SSH_KEY)
            cmd.add_argument("--intent")
            cmd.add_argument("--changed-from-previous")
            cmd.add_argument("--dry-run", action="store_true")
            cmd.add_argument("--detach", action="store_true")
            cmd.add_argument("--lease-export-dir", type=Path, help="Mounted/shared lease directory read by an independent always-on watchdog")
            provenance = cmd.add_mutually_exclusive_group()
            provenance.add_argument("--trial-record", type=Path)
            provenance.add_argument("--standalone-spec", action="store_true")
    return parser


def main(argv=None, *, parsed_args=None):
    args = parsed_args if parsed_args is not None else build_parser().parse_args(argv)
    if args.command == "templates":
        return list_templates(args)
    if args.poll_seconds < 5:
        raise ValueError("Poll interval must be at least five seconds")
    if args.command == "launch":
        return launch(args)
    if args.command == "retry":
        return retry(args)
    if args.command == "continue":
        return continue_run(args)
    if args.command == "reconcile":
        return reconcile(args)
    if args.command == "sync":
        sync(args)
        return 0
    cache, plan = load_plan(args)
    if args.command == "watch":
        return watch(args, cache, plan)
    if args.command == "status":
        print(json.dumps({"launch": plan, "last_local_observation": json.loads((cache / "state.json").read_text())}, indent=2))
        return 0
    if args.command == "handoff":
        with lock(cache / ".launch.lock"):
            handoff(args, cache, plan)
        return 0
    if args.command in {"start", "recover-backups"}:
        if time.time() >= plan["deadline_unix"]:
            raise ValueError("Original rental allowance expired; use a new continuation for training and explicitly extend a recovery lease if needed")
        client = client_for(args)
        instance = client.instance(plan["instance_id"])
        if instance.get("label") != plan["label"]:
            raise ValueError("Refusing an instance outside this saved run")
        if args.command == "start":
            client.start(plan["instance_id"])
            save_state(cache, disposition="start_requested")
        else:
            connection = connection_for(client, instance, plan, cache)
            code = "import pathlib,subprocess,sys; p=pathlib.Path(sys.argv[1]); log=(p/'recovery.log').open('ab'); subprocess.Popen([sys.executable,'-u',str(p/'code/workflows/vast/worker.py'),'--workspace',str(p),'--recover-only'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)"
            interpreter = (plan["remote_workspace"] + "/venv/bin/python" if plan.get("backend") == "sfcompute"
                           else plan["policy"]["python"])
            connection.python(code, plan["remote_workspace"], python=interpreter)
            save_state(cache, backup_recovery="dispatched_without_training")
        return 0
    if args.command == "stop":
        stop_owned(client_for(args), plan, explicit=True)
        save_state(cache, disposition="stop_requested")
        return 0
    raise ValueError("Unsupported command")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Local watcher closed. The existing remote job was not resubmitted.", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:
        # Local configuration errors are useful; provider/transport bodies stay private.
        message = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else type(exc).__name__
        print(f"Vast workflow: {message}", file=sys.stderr)
        raise SystemExit(1) from None
