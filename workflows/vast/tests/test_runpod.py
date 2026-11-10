"""Offline Runpod contracts: no rentals, credentials, or Drive requests in tests."""

import io
import json
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest

from scripts import sync_records
from workflows.runpod import api, policy, run
from workflows.vast import choose, common, run as shared, snapshot, ssh, watchdog, worker


def pod(**changes):
    return {"id": "pod123", "name": "mdlarc:runpod:test", "desiredStatus": "RUNNING",
            "image": policy.DEFAULTS["image"], "gpu": {"id": policy.TEMPLATES["a100-80gb"][0], "count": 1},
            "interruptible": False, "costPerHr": "1.25", "vcpuCount": 16, "memoryInGb": 64,
            "volumeInGb": 100, "containerDiskInGb": 30, "volumeMountPath": "/workspace",
            "publicIp": "203.0.113.1", "portMappings": {"22": 12345},
            "machine": {"secureCloud": True, "maxUploadSpeedMbps": 100, "maxDownloadSpeedMbps": 3000},
            "env": {"RUNPOD_API_KEY": "must-not-be-saved"}, **changes}


class Opener:
    def __init__(self, *values):
        self.values = iter(values)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append(request)
        value = next(self.values)
        if isinstance(value, Exception):
            raise value
        return io.BytesIO(value if isinstance(value, bytes) else json.dumps(value).encode())


def test_rest_contract_and_redacted_normalized_pod():
    opener = Opener(pod(), [pod()], b"", b"", b"")
    client = api.Runpod("account-key", opener=opener)
    item = client.instance("pod123")
    assert item["id"] == "pod123" and item["num_gpus"] == 1
    assert item["ssh_port"] == 12345
    assert "env" not in item and "must-not-be-saved" not in json.dumps(item)
    assert client.instances() == [item]
    client.stop("pod123")
    client.start("pod123")
    client.destroy("pod123")
    assert [(r.method, r.full_url) for r in opener.requests] == [
        ("GET", api.API_URL + "/pods/pod123?includeMachine=true"), ("GET", api.API_URL + "/pods"),
        ("POST", api.API_URL + "/pods/pod123/stop"), ("POST", api.API_URL + "/pods/pod123/start"),
        ("DELETE", api.API_URL + "/pods/pod123")]
    assert all(r.get_header("Authorization") == "Bearer account-key" for r in opener.requests)
    assert all(r.get_header("User-agent") == api.USER_AGENT for r in opener.requests)
    assert all("account-key" not in r.full_url for r in opener.requests)


@pytest.mark.parametrize("failure", [URLError("secret-response"), b"not json", {},
    HTTPError(api.API_URL, 503, "secret-response", {}, io.BytesIO(b"secret-response"))])
def test_creation_is_never_automatically_retried(failure):
    opener = Opener(failure)
    client = api.Runpod("secret-account-key", opener=opener)
    with pytest.raises(api.AmbiguousSubmission) as error:
        client.create({"name": "unique"})
    assert len(opener.requests) == 1
    assert "secret-response" not in str(error.value) and "secret-account-key" not in str(error.value)


def test_catalog_query_is_read_only_and_retries():
    rows = [{"id": "GPU", "memoryInGb": 80, "lowestPrice": {"stockStatus": "High", "uninterruptablePrice": 1}}]
    opener = Opener(URLError("unavailable"), {"data": {"gpuTypes": rows}})
    client = api.Runpod("key", opener=opener, sleep=lambda _: None)
    assert client.gpu_types(policy.validate({})) == rows
    query = json.loads(opener.requests[-1].data)["query"]
    assert "secureCloud: true" in query and "gpuCount: 1" in query
    assert "mutation" not in query and len(opener.requests) == 2


def test_supported_cuda_versions_are_read_from_live_create_schema():
    body = {"properties": {"allowedCudaVersions": {"items": {"enum": ["13.0", "12.9"]}}}}
    schema = {"paths": {"/pods": {"post": {"requestBody": {"content": {"application/json": {
        "schema": {"$ref": "#/components/schemas/CreatePod"}}}}}}}, "components": {"schemas": {"CreatePod": body}}}
    opener = Opener(schema)
    assert api.Runpod("key", opener=opener).supported_cuda_versions() == ["13.0", "12.9"]
    assert opener.requests[0].method == "GET" and opener.requests[0].full_url.endswith("/openapi.json")


def test_image_and_allowed_cuda_versions_must_match():
    with pytest.raises(ValueError, match="requires CUDA"):
        policy.validate({"image": "nvcr.io/nvidia/pytorch:26.05-py3", "allowed_cuda_versions": ["13.0"]})


def test_cloudflare_signature_denial_has_safe_diagnostic_and_is_not_retried():
    body = {"error_code": "1010", "cloudflare_error": True, "detail": "secret-provider-body"}
    error = HTTPError(api.API_URL, 403, "Forbidden", {}, io.BytesIO(json.dumps(body).encode()))
    opener = Opener(error)
    with pytest.raises(api.ApiError, match="Cloudflare rejected the client signature") as caught:
        api.Runpod("secret-account-key", opener=opener).create({"name": "unique"})
    assert len(opener.requests) == 1
    assert "secret-provider-body" not in str(caught.value)
    assert "secret-account-key" not in str(caught.value)


@pytest.mark.parametrize("identity", ["../pod", "pod?secret", "", None, 123])
def test_invalid_pod_identity_never_reaches_api(identity):
    client = api.Runpod("key", opener=Opener())
    with pytest.raises(ValueError):
        client.stop(identity)


def test_pod_scoped_client_refuses_another_pod():
    opener = Opener(b"")
    client = api.Runpod("scoped-key", opener=opener, own_pod="pod123")
    with pytest.raises(ValueError):
        client.destroy("another-pod")
    assert not opener.requests
    client.stop("pod123")
    assert len(opener.requests) == 1


@pytest.mark.parametrize("changes", [{"max_total_usd": 0}, {"num_gpus": True}, {"remote_root": "/tmp/run"},
    {"gpu_type_ids": []}, {"allowed_cuda_versions": []}, {"failure_action": "destroy"},
    {"backup_grace_seconds": 30000}, {"cloud_type": "ALL"}, {"min_cpu_ram_gb": float("nan")},
    {"unexpected": 1}])
def test_policy_rejects_invalid_constraints(changes):
    with pytest.raises(ValueError):
        policy.validate(changes)


def test_create_request_uses_persistent_workspace_scoped_controls_and_resource_filters():
    configured = policy.validate({})
    body = policy.create_body(configured, "saved-identity", "allowed-GPU", "ssh-ed25519 public-key")
    assert body["gpuTypeIds"] == ["allowed-GPU"]
    assert body["gpuTypePriority"] == "custom"
    assert body["interruptible"] is False and body["supportPublicIp"] is True
    assert body["ports"] == ["22/tcp"] and body["volumeMountPath"] == "/workspace"
    assert body["minVCPUPerGPU"] == 16 and body["minDownloadMbps"] == 2000
    assert body["env"] == {"SSH_PUBLIC_KEY": "ssh-ed25519 public-key"}
    assert body["allowedCudaVersions"] == configured["allowed_cuda_versions"]
    assert "dockerEntrypoint" not in body
    assert body["dockerStartCmd"][:2] == ["bash", "-lc"]
    assert "HostKey=/workspace/.mdlarc-runpod-ssh/host_ed25519_key" in body["dockerStartCmd"][2]
    subprocess.run(["bash", "-n"], input=body["dockerStartCmd"][2], text=True, check=True)


def test_cheapest_gpu_with_stock_and_compute_plus_storage_budget():
    configured = policy.validate({"gpu_type_ids": ["cheap", "expensive", "absent"]})
    def row(name, price, stock="High"):
        return {"id": name, "lowestPrice": {"stockStatus": stock, "uninterruptablePrice": price}}
    estimate, selected = policy.choose_gpu([row("unapproved", .1), row("expensive", 2),
        row("cheap", 1), row("absent", .1, "None")], configured)
    assert selected == "cheap" and estimate["estimated_max_usd"] == 13
    with pytest.raises(ValueError):
        policy.choose_gpu([row("cheap", 1)], {**configured, "max_total_usd": 12})
    estimate, selected = policy.choose_gpu([row("cheap", 1)], {**configured, "num_gpus": 2})
    assert estimate["hourly_usd"] == 2 and estimate["estimated_max_usd"] == 21
    with pytest.raises(ValueError):
        policy.choose_gpu([row("cheap", 1)], {**configured, "num_gpus": 2, "max_hourly_usd": 1.5})


@pytest.mark.parametrize("changes", [{"costPerHr": "2.01"}, {"costPerHr": "NaN"}, {"name": "someone-else"},
    {"gpu": {"id": "unapproved", "count": 1}}, {"gpu": {"id": policy.TEMPLATES["a100-80gb"][0], "count": 2}},
    {"volumeMountPath": "/ephemeral"}, {"volumeInGb": 1}, {"interruptible": True}, {"vcpuCount": 4},
    {"machine": {"secureCloud": False}}])
def test_allocated_pod_is_rechecked(changes):
    with pytest.raises(ValueError):
        policy.verify_pod(pod(**changes), policy.validate({}), "mdlarc:runpod:test")


@pytest.fixture
def launch_setup(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    spec = root / "spec.json"
    spec.write_text('{"run_name":"smoke"}')
    configured = tmp_path / "policy.json"
    configured.write_text("{}")
    key = tmp_path / "identity"
    key.write_text("unused-private-key")
    token = tmp_path / "drive-token.json"
    token.write_text("{}")
    monkeypatch.setattr(run, "ROOT", root)
    monkeypatch.setattr(shared, "ROOT", root)
    monkeypatch.setattr(shared, "preview", lambda _: {})
    monkeypatch.setattr(run, "public_key", lambda _: "synthetic-public-key")
    def package(root, spec_path, destination):
        with tarfile.open(destination, "w:gz") as archive:
            archive.add(spec_path, arcname="spec.json")
        return common.hashes(destination)
    monkeypatch.setattr(snapshot, "create", package)
    monkeypatch.setattr(shared, "store_for", lambda *args, **kwargs: SimpleNamespace(
        preflight=lambda: None, create_run=lambda _: "folder", config={}))
    class Client:
        def __init__(self):
            self.created, self.stopped = [], []
            self.failure, self.price = None, "1.25"
        def supported_cuda_versions(self):
            return ["13.0"]
        def gpu_types(self, configured):
            return [{"id": configured["gpu_type_ids"][0], "lowestPrice": {"stockStatus": "High", "uninterruptablePrice": 1}}]
        def create(self, body):
            self.created.append(body)
            if self.failure:
                raise self.failure
            return "pod123"
        def raw_pod(self, identity):
            return pod(name=self.created[0]["name"], costPerHr=self.price)
        def instance(self, identity):
            return api.normalize(self.raw_pod(identity))
        def instances(self):
            return [self.instance("pod123")]
        def stop(self, identity):
            self.stopped.append(identity)
    client = Client()
    monkeypatch.setattr(shared, "client_for", lambda _: client)
    args = run.build_parser().parse_args(["launch", "--branch", "test/branch", "--session", "example",
        "--config", str(spec), "--policy", str(configured), "--ssh-key", str(key), "--drive-token", str(token),
        "--intent", "test", "--changed-from-previous", "backend", "--detach"])
    return args, client


def test_launch_saves_immutable_policy_and_no_account_key(launch_setup, monkeypatch):
    args, client = launch_setup
    calls = []
    monkeypatch.setattr(shared, "handoff", lambda *values: calls.append(values))
    assert run.launch(args) == 0
    cache, plan = shared.load_plan(args)
    assert cache.parts[-3:] == ("test%2Fbranch", "runpod_runs", "example")
    assert len(client.created) == len(calls) == 1
    assert plan["instance_id"] == "pod123" and plan["backend"] == "runpod"
    assert plan["policy"]["gpu_type_ids"] == client.created[0]["gpuTypeIds"]
    assert "must-not-be-saved" not in (cache / "launch.json").read_text()
    args.policy.write_text('{"image":"changed"}')
    assert shared.load_plan(args)[1]["policy"]["image"] == policy.DEFAULTS["image"]
    with pytest.raises(ValueError, match="already has records"):
        run.launch(args)
    assert len(client.created) == 1


def test_ambiguous_create_reconciles_same_pod_without_training(launch_setup, monkeypatch):
    args, client = launch_setup
    client.failure = api.AmbiguousSubmission("lost response")
    monkeypatch.setattr(shared, "handoff", lambda *args: pytest.fail("Must not submit training"))
    with pytest.raises(RuntimeError, match="reconcile"):
        run.launch(args)
    cache, plan = shared.load_plan(args)
    assert plan["instance_id"] is None
    assert json.loads((cache / "state.json").read_text())["state"] == "rental_unconfirmed"
    assert shared.reconcile(args) == 0
    assert shared.load_plan(args)[1]["instance_id"] == "pod123"
    assert len(client.created) == 1 and client.stopped == []


def test_allocated_price_violation_stops_before_upload(launch_setup, monkeypatch):
    args, client = launch_setup
    client.price = "99"
    monkeypatch.setattr(shared, "handoff", lambda *args: pytest.fail("Must not upload or train"))
    with pytest.raises(ValueError, match="hourly"):
        run.launch(args)
    assert client.stopped == ["pod123"]
    assert shared.load_plan(args)[1]["instance_id"] == "pod123"


def test_dry_run_needs_no_api_or_drive(launch_setup, monkeypatch, capsys):
    args, client = launch_setup
    args.dry_run = True
    monkeypatch.setattr(shared, "client_for", lambda _: pytest.fail("Offline preview"))
    monkeypatch.setattr(shared, "store_for", lambda *a, **kw: pytest.fail("Offline preview"))
    assert run.launch(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["backend"] == "runpod" and not client.created


def test_unsupported_cuda_stops_before_pod_creation_or_run_folder(launch_setup, monkeypatch):
    args, client = launch_setup
    args.policy.write_text(json.dumps({"image": "nvcr.io/nvidia/pytorch:26.05-py3",
                                      "allowed_cuda_versions": ["13.2", "13.3", "13.4"]}))
    with pytest.raises(ValueError, match="live API does not accept"):
        run.launch(args)
    assert client.created == []
    assert not shared.cache_dir("test/branch", "example", "runpod").exists()


def test_handoff_recovers_same_worker_and_never_transfers_account_key(launch_setup, monkeypatch):
    args, client = launch_setup
    original = shared.handoff
    monkeypatch.setattr(shared, "handoff", lambda *values: None)
    run.launch(args)
    cache, plan = shared.load_plan(args)
    calls, uploaded = [], []
    class Connection:
        started = False
        def __init__(self, *a):
            pass
        def python(self, code, *a, **kw):
            calls.append(code)
        def json_file(self, path, **kw):
            if path.endswith("started.json"):
                return {"pid": 7} if self.started else None
            return {"state": "running"}
        def upload(self, local, remote, **kw):
            uploaded.append((local, remote))
        def launch(self, *a):
            Connection.started = True
    monkeypatch.setattr(shared, "SSH", Connection)
    original(args, cache, plan, client)
    count = len(uploaded)
    assert count == 3  # source, launch identity, dedicated Drive-only token
    assert uploaded[-1][0] == args.drive_token
    assert api.CONTROL_CHECK in calls
    original(args, cache, plan, client)
    assert len(uploaded) == count
    assert json.loads((cache / "state.json").read_text())["state"] == "worker_already_started"


def test_runpod_worker_requires_matching_scoped_credentials_and_training_cannot_inherit_them(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "CONTROL_FILE", str(tmp_path / "absent-control-file"))
    monkeypatch.setenv("RUNPOD_POD_ID", "pod123")
    monkeypatch.setenv("RUNPOD_API_KEY", "pod-scoped-key")
    client = worker.instance_client({"backend": "runpod", "instance_id": "pod123"})
    assert client.own_pod == "pod123" and client.key == "pod-scoped-key"
    with pytest.raises(RuntimeError):
        worker.instance_client({"backend": "runpod", "instance_id": "another-pod"})
    private = tmp_path / "private"
    private.mkdir()
    env = worker.clean_training_env({"branch": "branch", "run_name": "smoke"}, private)
    assert "RUNPOD_API_KEY" not in env


def test_native_and_cli_chooser_use_runpod_without_colab_account_dialog(monkeypatch):
    commands = []
    monkeypatch.setattr(choose.subprocess, "call", lambda args, **kw: commands.append(args) or 0)
    assert choose.main(["--backend", "runpod", "--", "--config", "spec.json", "--dry-run"]) == 0
    assert commands[-1][1:] == [str(choose.ROOT / "workflows/runpod/run.py"), "launch", "--config", "spec.json", "--dry-run"]
    monkeypatch.setattr(choose.sys, "platform", "darwin")
    monkeypatch.setattr(choose, "dialog", lambda script: "Runpod" if "Choose experiment workflow" in script else pytest.fail("Unexpected dialog"))
    assert choose.main(["--", "--config", "spec.json"]) == 0


def test_runpod_research_records_selected_but_live_identity_and_secrets_excluded():
    assert sync_records.selected("results/main/runpod_runs/session/spec.json")
    assert not sync_records.selected("results/main/runpod_runs/session/launch.json")
    assert not sync_records.selected("results/main/runpod_runs/session/private/drive-token.json")
    with pytest.raises(sync_records.SyncError):
        sync_records.validate_text("results/main/runpod_runs/session/spec.json", b'{"RUNPOD_API_KEY":"secret"}')


def test_string_pod_ids_work_with_existing_deadline_enforcement():
    stopped = []
    client = SimpleNamespace(instances=lambda: [api.normalize(pod())], stop=stopped.append)
    assert watchdog.enforce(client, {"instance_id": "pod123", "label": "mdlarc:runpod:test", "deadline_unix": 1}, 2) == "stop_requested"
    assert stopped == ["pod123"]
