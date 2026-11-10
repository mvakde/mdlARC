"""Offline regression checks for missing dataset metadata and explicit retries."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace
import zipfile

import pytest

from workflows.vast import common, run, snapshot, vast_api


ROOT = Path(__file__).resolve().parents[3]


def source_tree(root):
    for name in ("launch_run.py", "requirements.txt", "scripts/run_layout.py",
                 "dataset_building_scripts/build_variation_dataset.py", "dataset_building_scripts/task_splits.py"):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    spec = root / "specs/selected.json"
    spec.parent.mkdir()
    spec.write_text(json.dumps({"run_name": "selected", "defaults": {"batch_size": 32, "gradient_accumulation_steps": 2},
                               "enable_finetune": False, "evals": {"final": {"checkpoint": "pretrain.final"}},
                               "dataset": {"pretrain_data": {"kind": "task_variations"}}}))
    (root / 'src').mkdir()
    (root / 'src/example.py').write_text('VALUE = 1\n')
    with zipfile.ZipFile(root / "dataset.zip", "w") as archive:
        archive.writestr("exported-datasets/example.json", "{}")
    identity = common.hashes(root / "dataset.zip")
    (root / "scripts/colab_drive.json").write_text(json.dumps({
        "dataset_size": identity["size"], "dataset_sha256": identity["sha256"]}))
    return spec, identity


def test_packaged_variation_builder_can_verify_its_staged_archive(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    spec, identity = source_tree(source)
    archive = tmp_path / "source.tar.gz"
    snapshot.create(source, spec, archive)
    packaged = tmp_path / "packaged"
    common.extract_archive(archive, packaged)
    assets = packaged / "assets_tmp"
    assets.mkdir()
    shutil.copyfile(source / "dataset.zip", assets / "task-variations-v1.zip")
    result = subprocess.run([sys.executable, "-c",
        "from dataset_building_scripts.build_variation_dataset import _archive_identity; print(_archive_identity()[1])"],
        cwd=packaged, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == identity["sha256"]


def prepare_retry(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    spec, dataset_identity = source_tree(root)
    monkeypatch.setattr(run, "ROOT", root)
    monkeypatch.setattr(run, "preview", lambda _: {})
    cache = run.cache_dir("newreason", "original")
    cache.mkdir(parents=True)
    # Reproduce the historical omission without changing the preserved archive.
    with tarfile.open(cache / "source.tar.gz", "w:gz") as archive:
        for path in snapshot.source_files(root, spec):
            if path.name != "colab_drive.json":
                archive.add(path, arcname=path.relative_to(root).as_posix())
    plan = {"branch": "newreason", "session": "original", "instance_id": 123,
        "run_name": "selected", "spec_relative": "specs/selected.json",
        "spec_sha256": common.hashes(spec)["sha256"], "source": common.hashes(cache / "source.tar.gz"),
        "label": "mdlarc:newreason:original", "drive_folder_id": "original-folder",
        "remote_workspace": "/workspace/mdlarc-vast/original", "intent": "same experiment",
        "rented_at_unix": time.time() - 3600, "deadline_unix": time.time() + 3600,
        "policy": {"remote_root": "/workspace/mdlarc-vast", "backup_grace_seconds": 1800,
                   "max_total_usd": 40, "max_runtime_hours": 15},
        "drive_config": {"task_variations": dataset_identity}}
    common.atomic_json(cache / "launch.json", plan)
    results = tmp_path / "backup.tar.gz"
    ledger = tmp_path / "checkpoint_backups.json"
    ledger.write_text('{"files":[],"complete":true}')
    with tarfile.open(results, "w:gz") as archive:
        archive.add(ledger, arcname=ledger.name)
    manifest = {"ok": False, "finalized": True, "backups_complete": True,
        "archive": {"id": "archive-id", "name": "results.tar.gz", **common.hashes(results)}}
    selected = {"id": 123, "label": plan["label"], "actual_status": "exited", "intended_status": "stopped"}
    calls = []
    def label(identity, value):
        calls.append(("label", identity))
        selected["label"] = value
    def start(identity):
        calls.append(("start", identity))
    def create_run(session):
        calls.append(("drive", session))
        return "retry-folder"
    client = SimpleNamespace(instance=lambda _: dict(selected), label=label, start=start,
        create=lambda *a: pytest.fail("Retry must never rent"), offers=lambda *a: pytest.fail("Retry must never search offers"))
    store = SimpleNamespace(create_run=create_run, metadata=lambda _: {}, _verify_binary=lambda *a: None,
        download=lambda identity, destination, **kwargs: shutil.copyfile(results, destination))
    monkeypatch.setattr(run, "client_for", lambda _: client)
    monkeypatch.setattr(run, "store_for", lambda *a, **kw: store)
    monkeypatch.setattr(run, "manifest_from_drive", lambda *a: manifest)
    monkeypatch.setattr(run, "handoff", lambda *a: calls.append(("handoff", a[2]["instance_id"])))
    args = run.build_parser().parse_args(["retry", "--branch", "newreason", "--session", "original",
                                        "--retry-session", "retry1", "--detach"])
    return args, plan, calls, selected, manifest, cache, results


def test_retry_preserves_exact_spec_source_and_original_allowance(tmp_path, monkeypatch):
    args, parent, calls, selected, _, cache, _ = prepare_retry(tmp_path, monkeypatch)
    before = (cache / "source.tar.gz").read_bytes()
    assert run.retry(args) == 0
    child_cache = run.cache_dir("newreason", "retry1")
    child = json.loads((child_cache / "launch.json").read_text())
    for key in ("instance_id", "rented_at_unix", "deadline_unix", "policy", "spec_sha256", "drive_config"):
        assert child[key] == parent[key]
    assert child["added_source_files"] == ["scripts/colab_drive.json"]
    with tarfile.open(cache / "source.tar.gz") as old, tarfile.open(child_cache / "source.tar.gz") as new:
        assert set(new.getnames()) - set(old.getnames()) == {"scripts/colab_drive.json"}
        for name in old.getnames():
            assert old.extractfile(name).read() == new.extractfile(name).read()
    assert (cache / "source.tar.gz").read_bytes() == before
    assert calls == [("drive", "retry1"), ("label", 123), ("start", 123), ("handoff", 123)]
    with pytest.raises(ValueError, match="already has records"):
        run.retry(args)


@pytest.mark.parametrize("failure", ["identity", "running", "unfinalized", "success", "backups", "expired", "source", "metadata"])
def test_retry_refuses_uncertain_or_ineligible_parent_before_mutation(tmp_path, monkeypatch, failure):
    args, plan, calls, selected, manifest, cache, _ = prepare_retry(tmp_path, monkeypatch)
    if failure == "identity": selected["label"] = "someone-else"
    if failure == "running": selected.update(actual_status="running", intended_status="running")
    if failure == "unfinalized": manifest["finalized"] = False
    if failure == "success": manifest["ok"] = True
    if failure == "backups": manifest["backups_complete"] = False
    if failure == "expired":
        plan["deadline_unix"] = time.time() + 100
        common.atomic_json(cache / "launch.json", plan)
    if failure == "source": (cache / "source.tar.gz").write_bytes(b"corrupt")
    if failure == "metadata": (run.ROOT / "scripts/colab_drive.json").write_text('{}')
    with pytest.raises(ValueError): run.retry(args)
    assert calls == []


def test_retry_dry_run_is_offline(tmp_path, monkeypatch):
    args, _, calls, _, _, _, _ = prepare_retry(tmp_path, monkeypatch)
    args.dry_run = True
    monkeypatch.setattr(run, "client_for", lambda _: pytest.fail("Offline preview must not query Vast"))
    monkeypatch.setattr(run, "store_for", lambda *a, **kw: pytest.fail("Offline preview must not query Drive"))
    assert run.retry(args) == 0
    assert calls == []


@pytest.mark.parametrize('mismatch', [False, True])
def test_evaluation_retry_uses_only_exact_verified_parent_checkpoint(tmp_path, monkeypatch, mismatch):
    args, parent, calls, _, manifest, cache, results = prepare_retry(tmp_path, monkeypatch)
    entry = {"path": "pretrain/tiny.pt", "id": "checkpoint-id", "name": "tiny.pt", "size": 123, "sha256": "a" * 64}
    ledger = tmp_path / 'checkpoint_backups.json'
    ledger.write_text(json.dumps({'files': [entry], 'complete': True, 'errors': []}))
    with tarfile.open(results, 'w:gz') as archive:
        archive.add(ledger, arcname=ledger.name)
    manifest['archive'].update(common.hashes(results))
    checkpoint = {'drive_id': entry['id'], **{k:entry[k] for k in ('name','size','sha256')}}
    if mismatch:
        checkpoint['sha256'] = 'b' * 64
    args.eval_checkpoint = tmp_path / 'checkpoint.json'
    args.eval_checkpoint.write_text(json.dumps(checkpoint))
    args.eval_batch_size, args.eval_token_budget = 150, 262144
    if mismatch:
        with pytest.raises(ValueError, match='ledger'):
            run.retry(args)
        assert calls == []
        return
    assert run.retry(args) == 0
    child_cache = run.cache_dir('newreason', 'retry1')
    child = json.loads((child_cache / 'launch.json').read_text())
    spec = json.loads((child_cache / 'spec.json').read_text())
    assert child['evaluation_only'] and child['run_name'] == 'retry1'
    assert spec['eval_only'] and spec['checkpoint'] == checkpoint
    assert spec['eval_defaults'] == {'batch_size': 150, 'prefill_token_budget': 262144}
    with tarfile.open(cache / 'source.tar.gz') as archive:
        original = json.load(archive.extractfile(parent['spec_relative']))
    for key in ('defaults', 'dataset', 'evals'):
        assert spec[key] == original[key]
    assert child['deadline_unix'] == parent['deadline_unix']
    assert calls == [('drive', 'retry1'), ('label', 123), ('start', 123), ('handoff', 123)]


@pytest.mark.parametrize('busy', [False, True])
def test_b300_retry_reuses_powered_on_capacity_after_idle_verification(tmp_path, monkeypatch, busy):
    args, parent, calls, selected, _, cache, _ = prepare_retry(tmp_path, monkeypatch)
    selected.update(gpu_name='B300', actual_status='running', intended_status='running')
    parent['ssh_key_path'] = '/private/key'
    parent['policy']['python'] = 'python3'
    common.atomic_json(cache / 'launch.json', parent)
    monkeypatch.setattr(run, 'SSH', lambda *a: object())
    def verify(*a):
        if busy:
            raise RuntimeError('GPU or supervisor is still busy')
        calls.append(('verified_idle', 123))
    monkeypatch.setattr(run, 'verify_idle_instance', verify)
    if busy:
        with pytest.raises(RuntimeError, match='busy'):
            run.retry(args)
        assert calls == []
    else:
        assert run.retry(args) == 0
        assert calls == [('verified_idle', 123), ('drive', 'retry1'), ('label', 123), ('handoff', 123)]


def test_retry_batch_override_preserves_recipe_source_and_original_allowance(tmp_path, monkeypatch):
    args, parent, calls, _, _, cache, _ = prepare_retry(tmp_path, monkeypatch)
    args.batch_size, args.gradient_accumulation_steps = 16, 4
    before = (cache / 'source.tar.gz').read_bytes()
    assert run.retry(args) == 0
    child_cache = run.cache_dir('newreason', 'retry1')
    child = json.loads((child_cache / 'launch.json').read_text())
    for key in ('instance_id', 'policy', 'rented_at_unix', 'deadline_unix', 'drive_config'):
        assert child[key] == parent[key]
    assert child['spec_sha256'] != parent['spec_sha256']
    assert common.hashes(child_cache / 'spec.json')['sha256'] == child['spec_sha256']
    assert child['spec_overrides'] == {
        'pretrain.batch_size': {'before': 32, 'after': 16},
        'pretrain.gradient_accumulation_steps': {'before': 2, 'after': 4}}
    with tarfile.open(cache / 'source.tar.gz') as old, tarfile.open(child_cache / 'source.tar.gz') as new:
        original = json.load(old.extractfile(parent['spec_relative']))
        updated = json.load(new.extractfile(child['spec_relative']))
        assert updated.pop('pretrain') == {'batch_size': 16, 'gradient_accumulation_steps': 4}
        assert updated == original
        for name in old.getnames():
            if name != parent['spec_relative']:
                assert old.extractfile(name).read() == new.extractfile(name).read()
    assert (cache / 'source.tar.gz').read_bytes() == before
    assert calls == [('drive', 'retry1'), ('label', 123), ('start', 123), ('handoff', 123)]


@pytest.mark.parametrize('batch,accumulation', [(16, None), (None, 4), (0, 4), (-16, -4), (16, 2), (True, 64)])
def test_retry_invalid_batch_override_is_rejected_before_mutation(tmp_path, monkeypatch, batch, accumulation):
    args, _, calls, _, _, _, _ = prepare_retry(tmp_path, monkeypatch)
    args.batch_size, args.gradient_accumulation_steps = batch, accumulation
    with pytest.raises(ValueError):
        run.retry(args)
    assert calls == []


def test_retry_batch_override_uses_saved_stage_values_before_defaults():
    spec = {'defaults': {'batch_size': 32, 'gradient_accumulation_steps': 2},
            'pretrain': {'batch_size': 8, 'gradient_accumulation_steps': 4, 'epochs': 2}}
    changes = run.retry_batch_overrides(spec, 16, 2)
    assert changes['pretrain.batch_size']['before'] == 8
    assert changes['pretrain.gradient_accumulation_steps']['before'] == 4
    assert spec['pretrain']['epochs'] == 2


def test_retry_batch_override_does_not_guess_unrecorded_defaults():
    with pytest.raises(ValueError, match='explicitly define'):
        run.retry_batch_overrides({}, 16, 4)


def test_retry_explicit_effective_batch_change_preserves_other_settings(tmp_path, monkeypatch):
    args, parent, calls, _, _, cache, _ = prepare_retry(tmp_path, monkeypatch)
    args.batch_size, args.gradient_accumulation_steps, args.effective_batch_size = 12, 5, 60
    before = (cache / 'source.tar.gz').read_bytes()
    assert run.retry(args) == 0
    child_cache = run.cache_dir('newreason', 'retry1')
    child = json.loads((child_cache / 'launch.json').read_text())
    assert child['effective_batch_size_override'] == 60
    for key in ('instance_id', 'policy', 'rented_at_unix', 'deadline_unix', 'drive_config'):
        assert child[key] == parent[key]
    spec = json.loads((child_cache / 'spec.json').read_text())
    assert spec.pop('pretrain') == {'batch_size': 12, 'gradient_accumulation_steps': 5}
    with tarfile.open(cache / 'source.tar.gz') as old:
        assert spec == json.load(old.extractfile(parent['spec_relative']))
    assert (cache / 'source.tar.gz').read_bytes() == before
    assert calls == [('drive', 'retry1'), ('label', 123), ('start', 123), ('handoff', 123)]


@pytest.mark.parametrize('batch,accumulation,effective', [
    (12, 5, None), (12, 5, 64), (12, 5, 0), (12, 5, True), (None, None, 60), (12, None, 60),
])
def test_retry_effective_batch_change_requires_an_exact_opt_in(batch, accumulation, effective):
    spec = {'defaults': {'batch_size': 32, 'gradient_accumulation_steps': 2}}
    with pytest.raises(ValueError):
        run.retry_batch_overrides(spec, batch, accumulation, effective_batch_size=effective)
    assert 'pretrain' not in spec


def test_retry_explicit_source_fixes_preserve_every_other_file_and_allowance(tmp_path, monkeypatch):
    args, parent, calls, _, _, cache, _ = prepare_retry(tmp_path, monkeypatch)
    before = (cache / 'source.tar.gz').read_bytes()
    (run.ROOT / 'src/example.py').write_text('VALUE = 2\n')
    args.source_update = [Path('src/example.py')]
    assert run.retry(args) == 0
    child_cache = run.cache_dir('newreason', 'retry1')
    child = json.loads((child_cache / 'launch.json').read_text())
    for key in ('policy', 'deadline_unix', 'rented_at_unix', 'spec_sha256', 'drive_config'):
        assert child[key] == parent[key]
    assert child['source_updates'][0]['path'] == 'src/example.py'
    assert child['source_updates'][0]['before_sha256'] != child['source_updates'][0]['sha256']
    with tarfile.open(cache / 'source.tar.gz') as old, tarfile.open(child_cache / 'source.tar.gz') as new:
        assert new.extractfile('src/example.py').read() == b'VALUE = 2\n'
        for name in old.getnames():
            if name != 'src/example.py':
                assert old.extractfile(name).read() == new.extractfile(name).read()
    assert (cache / 'source.tar.gz').read_bytes() == before
    assert calls == [('drive', 'retry1'), ('label', 123), ('start', 123), ('handoff', 123)]


@pytest.mark.parametrize('selected', [
    ['specs/selected.json'], ['../outside.py'], ['/tmp/outside.py'],
    ['src/missing.py'], ['src/example.py', 'src/example.py'], ['src/linked.py'],
])
def test_retry_rejects_unsafe_source_fixes_before_mutation(tmp_path, monkeypatch, selected):
    args, _, calls, _, _, _, _ = prepare_retry(tmp_path, monkeypatch)
    (run.ROOT / 'src/linked.py').symlink_to(run.ROOT / 'src/example.py')
    args.source_update = [Path(value) for value in selected]
    with pytest.raises(ValueError):
        run.retry(args)
    assert calls == []


def test_retry_refuses_to_discard_completed_checkpoints(tmp_path, monkeypatch):
    args, _, calls, _, _, _, results = prepare_retry(tmp_path, monkeypatch)
    ledger = tmp_path / "checkpoint_backups.json"
    ledger.write_text('{"files":[{"path":"pretrain.pt"}],"complete":true}')
    with tarfile.open(results, "w:gz") as archive: archive.add(ledger, arcname=ledger.name)
    with pytest.raises(ValueError, match="checkpoint continuation"):
        run.retry(args)
    assert calls == []


def test_ambiguous_retry_start_is_never_resubmitted(tmp_path, monkeypatch):
    args, _, calls, _, _, _, _ = prepare_retry(tmp_path, monkeypatch)
    client = run.client_for(args)
    def start(identity):
        calls.append(("start", identity))
        raise vast_api.AmbiguousSubmission("unconfirmed")
    client.start = start
    with pytest.raises(RuntimeError, match="start is unconfirmed"):
        run.retry(args)
    assert calls == [("drive", "retry1"), ("label", 123), ("start", 123)]


@pytest.mark.parametrize("failure", [None, "unfinished", "transfers", "gpu", "other-run"])
def test_retry_handoff_preserves_usage_and_requires_idle_finalized_ancestors(tmp_path, monkeypatch, failure):
    root = tmp_path / "remote"
    parent = root / "original"
    current = root / "retry1"
    ancestor = {"branch": "newreason", "session": "original", "instance_id": 123,
                "remote_workspace": str(parent)}
    common.atomic_json(parent / "private/started.json", {"pid": 1})
    if failure == "other-run":
        common.atomic_json(root / "unknown/private/started.json", {"pid": 2})
    parent_status = {**ancestor, "ok": False, "finalized": failure != "unfinished", "backups_complete": True}
    usage = None if failure == "transfers" else {"upload": 1000, "download": 2000}
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="456\n" if failure == "gpu" else ""))
    class Connection:
        def json_file(self, path, **kwargs):
            if path == str(parent / "status.json"): return parent_status
            if path == str(parent / "private/transfers.json"): return usage
            if path == str(current / "private/started.json"): return None
            if path == str(current / "status.json"): return {"state": "preparing_inputs"}
            pytest.fail(f"Unexpected remote read: {path}")
        def python(self, code, *args, **kwargs):
            if ("--query-compute-apps=pid" in code or "private/transfers.json" in code
                    or "p.mkdir(parents=True,exist_ok=True,mode=0o700)" in code):
                with monkeypatch.context() as context:
                    context.setattr(sys, "argv", ["-c", *args])
                    exec(code, {})
            return b"ready"
        def upload(self, local, remote, **kwargs): calls.append(("upload", str(remote)))
        def launch(self, *args): calls.append(("launch", str(current)))
    monkeypatch.setattr(run, "SSH", lambda *a: Connection())
    plan = {"instance_id": 123, "label": "retry-label", "branch": "newreason",
        "session": "retry1", "remote_workspace": str(current), "retry_ancestors": [ancestor],
        "adopted_instance": {"id": 123}, "source": {"size": 50, "sha256": "unused"},
        "policy": {"python": sys.executable, "remote_root": str(root), "startup_timeout_seconds": 20},
        "deadline_unix": time.time() + 3600, "ssh_key_path": str(tmp_path / "identity")}
    cache = tmp_path / "cache"
    cache.mkdir()
    args = SimpleNamespace(drive_token=tmp_path / "external-token", secrets_file=None)
    client = SimpleNamespace(instance=lambda _: {"id": 123, "label": "retry-label"})
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            run.handoff(args, cache, plan, client)
        assert calls == []
    else:
        run.handoff(args, cache, plan, client)
        assert plan["prior_transfers"] == usage
        assert json.loads((current / "private/transfers.json").read_text()) == {"upload": 1000, "download": 2050}
        assert calls[-1] == ("launch", str(current))
        assert len([call for call in calls if call[0] == "launch"]) == 1
