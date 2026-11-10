"""Offline checks for launching on an explicitly selected existing rental."""

import json
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

import pytest

from workflows.vast import common, policy, run, vast_api


def test_external_ssh_worker_never_controls_provider_rental(monkeypatch):
    from workflows.vast import worker

    def forbidden(*args, **kwargs):
        pytest.fail("External SSH workers must not construct a provider API client")

    monkeypatch.setattr(worker, "Vast", forbidden)
    external = {"backend": "ssh", "instance_id": "external-vm"}
    assert worker.instance_client(external) is None
    for status in ({}, {"ok": False}, {"ok": True, "backups_complete": True, "finalized": True}):
        assert worker.cleanup_action(status, {"success_action": "destroy", "failure_action": "stop"}, external) == "keep"


def test_external_ssh_deadline_guard_leaves_vm_running(tmp_path, monkeypatch):
    from workflows.vast import worker

    def forbidden(*args, **kwargs):
        pytest.fail("An external SSH deadline guard must not access provider control")

    monkeypatch.setattr(worker, "instance_client", forbidden)
    assert worker.deadline_guard(tmp_path, {"backend": "ssh", "instance_id": "external-vm"}) == 0
    assert json.loads((tmp_path / "deadline-status.json").read_text())["state"] == "keep_external_instance"


def infrastructure():
    return policy.apply_template(policy.validate({
        "image": "nvcr.io/nvidia/pytorch:26.05-py3", "gpu_names": ["H100_PCIE"],
        "min_cuda_version": 13.2, "max_hourly_usd": 3, "max_total_usd": 54,
        "max_runtime_hours": 20, "min_cpu_cores_effective": 16,
        "min_cpu_ram_gb": 32, "min_download_mbps": 2000,
    }), "h100")


def instance(**overrides):
    return {"id": 123, "label": None, "actual_status": "running", "intended_status": "running",
            "gpu_name": "H100 PCIE", "num_gpus": 1, "gpu_ram": 81559, "cpu_ram": 257392,
            "cpu_cores_effective": 16, "inet_up": 3342, "inet_down": 3426,
            "disk_space": 40, "cuda_max_good": 13.3,
            "image_uuid": "nvcr.io/nvidia/pytorch:26.05-py3",
            "dph_total": 2.5946074074074073, "inet_up_cost": 0.01171875,
            "inet_down_cost": 0.01171875, "start_date": time.time() - 60,
            "jupyter_token": "must-not-be-recorded", **overrides}


def test_existing_container_is_recorded_without_replacing_it():
    original = infrastructure()
    selected, quote = policy.existing_instance(instance(), original)
    assert selected["image"] == "nvcr.io/nvidia/pytorch:26.05-py3"
    assert selected["image_selection"] == "fixed"
    assert selected["disk_gb"] == 40
    assert quote["estimated_max_usd"] == pytest.approx(52.712460648148145)
    for key in original.keys() - {"image_selection", "disk_gb"}:
        assert selected[key] == original[key]
    assert original["disk_gb"] == 100


@pytest.mark.parametrize("overrides", [
    {"actual_status": "stopped"}, {"intended_status": "stopped"},
    {"label": "mdlarc:another:run"}, {"label": "mdlarc-vast|legacy"},
    {"num_gpus": 2}, {"gpu_name": "RTX 4090"}, {"gpu_ram": 24000},
    {"cpu_cores_effective": 8}, {"inet_down": 100}, {"disk_space": 0},
    {"disk_space": float("nan")}, {"image_uuid": "unknown/image:latest"},
    {"cuda_max_good": 12.9}, {"dph_total": 3.1}, {"inet_down_cost": None},
])
def test_existing_instance_rejects_incompatible_or_claimed_rentals(overrides):
    with pytest.raises(ValueError):
        policy.existing_instance(instance(**overrides), infrastructure())


def test_existing_rental_still_obeys_total_allowance():
    limits = {**infrastructure(), "max_total_usd": 40}
    with pytest.raises(ValueError, match="total estimated allowance"):
        policy.existing_instance(instance(), limits)


def test_adopted_ngc_cuda13_minor_compatibility_keeps_actual_runtime():
    selected, _ = policy.existing_instance(instance(
        image_uuid="nvcr.io/nvidia/pytorch:26.09-py3",
        cuda_max_good=13.2, driver_version="595.71.05",
    ), infrastructure())
    assert selected["image"] == "nvcr.io/nvidia/pytorch:26.09-py3"
    assert selected["min_cuda_version"] == 13.4


@pytest.mark.parametrize("overrides", [
    {"driver_version": "579.99.01"}, {"driver_version": "unknown"},
    {"driver_version": None}, {"cuda_max_good": 12.9},
])
def test_adopted_minor_compatibility_requires_supported_driver(overrides):
    with pytest.raises(ValueError, match="driver does not support"):
        policy.existing_instance(instance(**{
            "image_uuid": "nvcr.io/nvidia/pytorch:26.09-py3", "cuda_max_good": 13.2,
            "driver_version": "595.71.05", **overrides,
        }), infrastructure())


def test_minor_compatibility_does_not_relax_explicit_host_floor():
    with pytest.raises(ValueError, match="driver does not support"):
        policy.existing_instance(instance(
            image_uuid="nvcr.io/nvidia/pytorch:26.09-py3", cuda_max_good=13.2,
            driver_version="595.71.05",
        ), {**infrastructure(), "min_cuda_version": 13.3})


def prepare_launch(tmp_path, monkeypatch, *, label_error=False, budget=54):
    root = tmp_path / "repo"
    root.mkdir()
    spec = root / "spec.json"
    spec.write_text('{"run_name":"smoke"}')
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps({**infrastructure(), "max_total_usd": budget}))
    key = tmp_path / "identity"
    key.write_text("unused test identity")
    monkeypatch.setattr(run, "ROOT", root)
    monkeypatch.setattr(run, "preview", lambda _: {})
    monkeypatch.setattr(run, "public_key", lambda _: "synthetic public key")
    def package(root, spec_path, destination):
        with tarfile.open(destination, "w:gz") as archive:
            archive.add(spec_path, arcname="spec.json")
        return common.hashes(destination)
    monkeypatch.setattr(run.snapshot, "create", package)
    calls = []
    selected = instance()
    def refuse(*args):
        pytest.fail("Existing-instance launch must never search offers or create a rental")
    def label(identity, value):
        calls.append(("label", identity))
        selected["label"] = value
        if label_error:
            raise vast_api.AmbiguousSubmission("simulated lost label response")
    def create_run(session):
        calls.append(("drive", session))
        return "synthetic-folder"
    client = SimpleNamespace(instance=lambda identity: dict(selected), label=label, offers=refuse, create=refuse)
    monkeypatch.setattr(run, "client_for", lambda _: client)
    monkeypatch.setattr(run, "store_for", lambda *args: SimpleNamespace(
        preflight=lambda: None, create_run=create_run, config={}))
    monkeypatch.setattr(run, "handoff", lambda *args: calls.append(("handoff", args[2]["instance_id"])))
    args = run.build_parser().parse_args(["launch", "--branch", "newreason", "--session", "existing-test",
        "--config", str(spec), "--policy", str(policy_path), "--ssh-key", str(key),
        "--intent", "test", "--changed-from-previous", "existing H100", "--instance-id", "123", "--detach"])
    return args, calls, selected


@pytest.mark.parametrize("label_error", [False, True])
def test_adoption_never_rents_and_preserves_original_rental_deadline(tmp_path, monkeypatch, label_error):
    args, calls, selected = prepare_launch(tmp_path, monkeypatch, label_error=label_error)
    assert run.launch(args) == 0
    _, saved = run.load_plan(args)
    assert saved["instance_id"] == 123
    assert saved["adopted_instance"]["label"] is None
    assert saved["rented_at_unix"] == selected["start_date"]
    assert saved["deadline_unix"] == selected["start_date"] + 20 * 3600
    assert saved["policy"]["image"] == selected["image_uuid"]
    assert saved["runtime_compatibility"]["image_selection"] == "fixed"
    assert "must-not-be-recorded" not in json.dumps(saved)
    assert calls == [("drive", "existing-test"), ("label", 123), ("handoff", 123)]


def test_adoption_rejects_budget_before_claim_or_drive_folder(tmp_path, monkeypatch):
    args, calls, _ = prepare_launch(tmp_path, monkeypatch, budget=40)
    with pytest.raises(ValueError, match="total estimated allowance"):
        run.launch(args)
    assert calls == []


def test_adoption_refuses_an_instance_with_any_saved_run(tmp_path, monkeypatch):
    args, calls, _ = prepare_launch(tmp_path, monkeypatch)
    saved = run.ROOT / "results/another-branch/vast_runs/prior/launch.json"
    common.atomic_json(saved, {"instance_id": 123, "branch": "another-branch", "session": "prior"})
    with pytest.raises(ValueError, match="already has a saved run"):
        run.launch(args)
    assert calls == []


def completed_b300(tmp_path, monkeypatch):
    args, calls, selected = prepare_launch(tmp_path, monkeypatch)
    selected.update(gpu_name="B300", label="mdlarc:newreason:prior")
    args.policy_override = ['gpu_names=["B300"]']
    prior = {"instance_id": 123, "branch": "newreason", "session": "prior",
             "label": selected["label"], "remote_workspace": "/workspace/mdlarc-vast/prior",
             "drive_folder_id": "prior-folder", "drive_config": {}}
    common.atomic_json(run.cache_dir("newreason", "prior") / "launch.json", prior)
    completion = {"ok": True, "finalized": True, "backups_complete": True,
                  "archive": {"id": "prior-archive", "name": "results.tar.gz"}}
    monkeypatch.setattr(run, "manifest_from_drive", lambda store, plan: completion)
    store_for = run.store_for
    def verified_store(*values, **kwargs):
        store = store_for(*values)
        store.metadata = lambda _: {}
        store._verify_binary = lambda *args: None
        return store
    monkeypatch.setattr(run, "store_for", verified_store)
    monkeypatch.setattr(run, "SSH", lambda *args: object())
    monkeypatch.setattr(run, "verify_idle_instance", lambda connection, plan: calls.append(("idle", plan["previous_runs"][0]["session"])))
    return args, calls, selected, completion


def test_fresh_run_reuses_finalized_b300_and_preserves_previous_identity(tmp_path, monkeypatch):
    args, calls, selected, _ = completed_b300(tmp_path, monkeypatch)
    assert run.launch(args) == 0
    _, saved = run.load_plan(args)
    assert saved["previous_runs"][0]["session"] == "prior"
    assert saved["previous_runs"][0]["backups_complete"] is True
    assert saved["adopted_instance"]["label"] == "mdlarc:newreason:prior"
    assert saved["deadline_unix"] == selected["start_date"] + 20 * 3600
    assert (run.cache_dir("newreason", "prior") / "launch.json").is_file()
    assert calls == [("idle", "prior"), ("drive", "existing-test"), ("label", 123), ("handoff", 123)]


@pytest.mark.parametrize("field", ["finalized", "backups_complete"])
def test_fresh_b300_run_refuses_incomplete_previous_backups(tmp_path, monkeypatch, field):
    args, calls, _, completion = completed_b300(tmp_path, monkeypatch)
    completion[field] = False
    with pytest.raises(ValueError, match="finalized verified backups"):
        run.launch(args)
    assert calls == []


def test_fresh_b300_run_refuses_unknown_label(tmp_path, monkeypatch):
    args, calls, selected, _ = completed_b300(tmp_path, monkeypatch)
    selected["label"] = "another-owner"
    with pytest.raises(ValueError, match="label does not match"):
        run.launch(args)
    assert calls == []


def test_fresh_b300_run_checks_idleness_before_claiming_or_creating_drive_folder(tmp_path, monkeypatch):
    args, calls, _, _ = completed_b300(tmp_path, monkeypatch)
    def busy(*args):
        raise RuntimeError("Selected GPU already has a compute process")
    monkeypatch.setattr(run, "verify_idle_instance", busy)
    with pytest.raises(RuntimeError, match="compute process"):
        run.launch(args)
    assert calls == []


def test_verified_previous_runs_cannot_bypass_other_gpu_ownership():
    with pytest.raises(ValueError, match="limited to kept B300"):
        policy.existing_instance(instance(label="mdlarc:branch:prior"), infrastructure(), verified_previous_runs=True)


def test_adoption_dry_run_is_offline(tmp_path, monkeypatch):
    args, calls, _ = prepare_launch(tmp_path, monkeypatch)
    args.dry_run = True
    monkeypatch.setattr(run, "client_for", lambda _: pytest.fail("Dry run must not query the rental"))
    monkeypatch.setattr(run, "store_for", lambda *args: pytest.fail("Dry run must not access Drive"))
    assert run.launch(args) == 0
    assert calls == []


@pytest.mark.parametrize("speed", [100, None])
def test_explicit_download_override_allows_adoption_and_is_recorded(tmp_path, monkeypatch, speed):
    args, calls, selected = prepare_launch(tmp_path, monkeypatch)
    selected["inet_down"] = speed
    args.policy_override = ["min_download_mbps=0"]
    assert run.launch(args) == 0
    _, saved = run.load_plan(args)
    assert saved["policy_overrides"] == {"min_download_mbps": 0}
    assert saved["policy"]["min_download_mbps"] == 0
    assert saved["policy"]["max_total_usd"] == 54
    assert saved["deadline_unix"] == selected["start_date"] + 20 * 3600
    assert json.loads(args.policy.read_text())["min_download_mbps"] == 2000
    assert calls == [("drive", "existing-test"), ("label", 123), ("handoff", 123)]


def test_download_override_does_not_bypass_budget(tmp_path, monkeypatch):
    args, calls, selected = prepare_launch(tmp_path, monkeypatch, budget=40)
    selected["inet_down"] = 100
    args.policy_override = ["min_download_mbps=0"]
    with pytest.raises(ValueError, match="total estimated allowance"):
        run.launch(args)
    assert calls == []


def test_expired_existing_rental_cannot_reset_its_deadline():
    with pytest.raises(ValueError, match="already expired"):
        run.existing_rental_start(instance(start_date=time.time() - 21 * 3600), infrastructure())


def test_label_api_changes_only_label():
    client = vast_api.Vast("synthetic-test-key")
    calls = []
    client.request = lambda *args: calls.append(args)
    client.label(123, "mdlarc:branch:session")
    assert calls == [("PUT", "/instances/123/", {"label": "mdlarc:branch:session"})]


@pytest.mark.parametrize("busy", ["gpu", "managed-run"])
def test_busy_existing_instance_is_refused_before_uploading_credentials(tmp_path, monkeypatch, busy):
    root = tmp_path / "remote"
    remote = root / "new-session"
    if busy == "managed-run":
        marker = root / "old-session/private/started.json"
        common.atomic_json(marker, {"started": True})
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="456\n" if busy == "gpu" else ""))
    class Connection:
        def python(self, code, *args, **kwargs):
            if "--query-compute-apps=pid" in code:
                with monkeypatch.context() as context:
                    context.setattr(sys, "argv", ["-c", *args])
                    exec(code, {})
            return b"ready"
        def json_file(self, *args, **kwargs):
            return None
        def upload(self, *args, **kwargs):
            pytest.fail("Do not upload source or credentials to a busy instance")
        def launch(self, *args, **kwargs):
            pytest.fail("Do not start another worker on a busy instance")
    monkeypatch.setattr(run, "SSH", lambda *args: Connection())
    monkeypatch.setattr(run, "public_key", lambda *args: "synthetic public key")
    selected = instance(label="mdlarc:new:session")
    client = SimpleNamespace(instance=lambda _: selected, attach_ssh_key=lambda *args: None)
    plan = {"policy": {**infrastructure(), "remote_root": str(root)}, "instance_id": 123,
            "label": selected["label"], "deadline_unix": time.time() + 3600,
            "ssh_key_path": str(tmp_path / "identity"), "remote_workspace": str(remote),
            "adopted_instance": {"id": 123}}
    with pytest.raises(RuntimeError, match="compute process|another managed run"):
        run.handoff(SimpleNamespace(), tmp_path / "cache", plan, client)


@pytest.mark.parametrize("rejected", [False, True])
def test_handoff_checks_ssh_before_key_registration_and_observes_rejected_duplicates(tmp_path, monkeypatch, rejected):
    probes = []
    attachments = []
    class Connection:
        def __init__(self, *args):
            pass
        def python(self, *args, **kwargs):
            probes.append(1)
            if len(probes) == 1:
                raise RuntimeError("SSH not ready yet")
            return b"ready"
        def json_file(self, *args, **kwargs):
            return {"started": True}
    def attach(identity, key):
        assert len(probes) == 1
        attachments.append(identity)
        if rejected:
            raise vast_api.ApiError("Vast API rejected the request")
    monkeypatch.setattr(run, "SSH", Connection)
    monkeypatch.setattr(run, "public_key", lambda _: "synthetic public key")
    monkeypatch.setattr(run.time, "sleep", lambda _: None)
    plan = {"instance_id": 123, "label": "mine", "deadline_unix": time.time() + 100,
            "policy": infrastructure(), "ssh_key_path": "/unused/key", "remote_workspace": "/workspace/session"}
    client = SimpleNamespace(instance=lambda _: {"id": 123, "label": "mine"}, attach_ssh_key=attach)
    run.handoff(SimpleNamespace(), tmp_path, plan, client)
    assert len(probes) == 2
    assert attachments == [123]
    assert json.loads((tmp_path / "state.json").read_text())["state"] == "worker_already_started"
