"""Offline contracts for SF Compute: identity, budgets, handoff and recovery without rentals."""

import io
import json
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace

import pytest

from scripts import sync_records
from workflows.sfcompute import api, policy, run, transport
from workflows.vast import choose, common, run as shared, snapshot, worker


def node(**changes):
    return {"id": "node-id", "name": "saved-node", "chip": "h100", "image": policy.DEFAULTS["image"],
            "status": "running (idle)", "rate": {"node_per_min_usd_micros": 84000},
            "resources": {"cpu_limit": 28, "memory_limit_bytes": 115 * 2**30}, **changes}


def test_cli_uses_explicit_workspace_and_stdin_for_secrets(monkeypatch):
    calls = []
    def execute(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout='{"ok":true}')
    monkeypatch.setattr(api.subprocess, "run", execute)
    client = api.Autor("/installed/autor", workspace="research")
    client.request("post", "/nodes/saved-node/commands", {"env": {"HF_TOKEN": "secret-value"}})
    command, options = calls[0]
    assert command[:5] == ["/installed/autor", "--api-url", "https://givemeanode.com", "--workspace", "research"]
    assert "secret-value" not in " ".join(command)
    assert json.loads(options["input"])["env"]["HF_TOKEN"] == "secret-value"
    assert command[-2:] == ["--data", "-"]


@pytest.mark.parametrize("failure", [subprocess.TimeoutExpired("autor", 10),
    SimpleNamespace(returncode=1, stdout="", stderr="secret-server-response"),
    SimpleNamespace(returncode=5, stdout="", stderr="secret-response"),
    SimpleNamespace(returncode=0, stdout="not-json", stderr="secret-response")])
def test_mutations_are_not_retried_after_unknown_response(monkeypatch, failure):
    calls = []
    def execute(*args, **kwargs):
        calls.append(args)
        if isinstance(failure, Exception):
            raise failure
        return failure
    monkeypatch.setattr(api.subprocess, "run", execute)
    with pytest.raises(api.AmbiguousSubmission) as error:
        api.Autor().create(policy.validate({}), "unique-node")
    assert len(calls) == 1 and "secret-response" not in str(error.value)


def test_paginated_listing_and_identity_change_are_checked(monkeypatch):
    client = api.Autor()
    pages = iter([{"items": [node()], "next_cursor": "next"}, {"items": [node(id="other", name="other-node")]}])
    calls = []
    monkeypatch.setattr(client, "request", lambda method, path, **kw: calls.append(path) or next(pages))
    assert len(client.instances()) == 2
    assert calls == ["/nodes", "/nodes?cursor=next"]
    monkeypatch.setattr(client, "request", lambda *a, **k: node(id="replacement"))
    with pytest.raises(ValueError, match="identity changed"):
        client.stop("node-id")


def test_create_uses_official_cli_shape_normalization_then_full_node_metadata(monkeypatch):
    client=api.Autor(); calls=[]
    def cli(args, **kwargs):
        calls.append(args)
        if args[:2]==['node','create']:
            return {'id':'node-id','name':'saved-node','status':'queued','chip':'h100'}
        return node()
    monkeypatch.setattr(client,'cli',cli)
    result=client.create(policy.validate({}),'saved-node')
    assert result['image']==policy.DEFAULTS['image']
    assert calls[0][:2]==['node','create'] and '--chip' in calls[0] and '--max-wait' in calls[0]
    assert calls[1]==['api','get','/nodes/saved-node']


def test_transport_uses_official_run_without_raw_stdin_body():
    calls=[]
    client=SimpleNamespace(instance=lambda _:api.normalize(node()),
        cli=lambda args,**kwargs:calls.append(args) or {'exit_code':0,'stdout':'ready','truncated':False})
    conn=transport.Transport(client,api.normalize(node()))
    assert conn.python("print('ready')")==b'ready'
    assert calls==[['run','saved-node','--timeout','60','--','python3','-c',"print('ready')"]]


def test_pagination_refuses_repeated_cursor(monkeypatch):
    client = api.Autor()
    pages = iter([{"items": [node()], "next_cursor": "same"}, {"items": [], "next_cursor": "same"}])
    monkeypatch.setattr(client, "request", lambda *a, **kw: next(pages))
    with pytest.raises(api.ApiError, match="cursor"):
        client.nodes()


@pytest.mark.parametrize("changes", [{"num_gpus": 8}, {"num_gpus": True}, {"remote_root": "/tmp/mdlarc"},
    {"workspace": "../other"}, {"api_url": "https://untrusted.example"}, {"success_action": "destroy"},
    {"failure_action": "destroy"}, {"max_total_usd": 10}, {"max_hourly_usd": float("nan")},
    {"scratch_gib": -1}, {"cache_runtime_validation": 1}, {"unknown": 1}])
def test_policy_preserves_constraints(changes):
    with pytest.raises(ValueError):
        policy.validate(changes)


def test_price_checks_whole_node_and_live_resources():
    configured = policy.validate({})
    assert policy.verify_node(node(), configured, "saved-node", "node-id", ready=True)["hourly_usd"] == 5.04
    for changes in ({"image": "wrong"}, {"chip": "8xh100"}, {"id": "wrong"},
                    {"resources": {"cpu_limit": 112, "memory_limit_bytes": 2**30}},
                    {"rate": {"node_per_min_usd_micros": 1000000}}, {"status": "stopped"}):
        with pytest.raises(ValueError):
            policy.verify_node(node(**changes), configured, "saved-node", "node-id", ready=True)
    normalized = api.normalize(node(env={"TOKEN": "must-not-save"}))
    assert normalized["actual_status"] == "running" and "env" not in normalized
    assert "must-not-save" not in json.dumps(normalized)


@pytest.mark.parametrize("key,value,expected", [("state", "stopped (disk intact)", "stopped"),
    ("status", "stopped (disk intact)", "stopped"), ("status", "stopping", "stopped"),
    ("state", "running (idle)", "running")])
def test_real_listing_and_detail_state_formats(key, value, expected):
    assert api.normalize({"id": "node-id", "name": "saved-node", "chip": "h100", key: value})["actual_status"] == expected


def test_no_observation_exec_on_stopped_node():
    client = SimpleNamespace(instance=lambda _: api.normalize(node(status="stopped")),
                             request=lambda *a, **k: pytest.fail("must not wake the node"))
    connection = transport.Transport(client, api.normalize(node()))
    with pytest.raises(RuntimeError, match="will not wake"):
        connection.json_file("/home/dev/status.json")
    with pytest.raises(RuntimeError, match="never wakes"):
        transport.Transport(client, api.normalize(node(status="stopped")))


def test_upload_requires_verified_exact_content(tmp_path, monkeypatch):
    source = tmp_path / "private-token.json"
    source.write_text('{"private":"never-in-command"}')
    client = SimpleNamespace(cli=lambda *a, **kw: {"written": True, "sha256_verified": True,
                                                  "sha256": "wrong", "bytes_written": source.stat().st_size})
    connection = transport.Transport(client, api.normalize(node()))
    calls = []
    monkeypatch.setattr(connection, "python", lambda *a, **k: calls.append(a) or b"missing\n")
    with pytest.raises(ValueError, match="exact file identity"):
        connection.upload(source, "/home/dev/run/private/drive-token.json")
    assert "never-in-command" not in str(calls)
    assert len(calls) == 1  # No chmod/launch after a failed checksum.


def test_dispatch_is_exclusive_even_after_lost_response(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    (workspace / "private").mkdir(parents=True)
    (workspace / "code/workflows/sfcompute").mkdir(parents=True)
    # A harmless bootstrap fixture lets the real remote dispatch marker execute locally.
    (workspace / "code/workflows/sfcompute/bootstrap.py").write_text("pass\n")
    connection = transport.Transport(SimpleNamespace(), api.normalize(node()))
    def python(code, directory, **kwargs):
        return subprocess.check_output([__import__('sys').executable, "-c", code, directory])
    monkeypatch.setattr(connection, "python", python)
    connection.launch(str(workspace), "python3")
    first = (workspace / "private/dispatched.json").read_bytes()
    connection.launch(str(workspace), "python3")
    assert (workspace / "private/dispatched.json").read_bytes() == first


class Opener:
    def __init__(self, values):
        self.values, self.requests = iter(values), []
    def open(self, request, timeout):
        self.requests.append(request)
        return io.BytesIO(json.dumps(next(self.values)).encode())


def test_remote_control_uses_only_own_node_metadata():
    opener = Opener([node(), node(), {"status": "stopping"}])
    client = api.MetadataClient("node-id", "saved-node", opener=opener)
    with pytest.raises(ValueError, match="another node"):
        client.stop("someone-else")
    client.stop("node-id")
    assert [r.method for r in opener.requests] == ["GET", "GET", "POST"]
    assert all(r.get_header("Metadata-flavor") == "givemeanode" for r in opener.requests)
    assert all(r.get_header("Authorization") is None for r in opener.requests)
    with pytest.raises(ValueError, match="identity"):
        api.MetadataClient("wrong", "saved-node", opener=Opener([node()]))


def test_chooser_and_native_dialog_route_sfcompute(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(choose.subprocess, "call", lambda args, **kw: calls.append(args) or 0)
    assert choose.main(["--backend", "sfcompute", "--", "--config", "specs/example.json", "--dry-run"]) == 0
    assert calls[0][1:] == [str(choose.ROOT / "workflows/sfcompute/run.py"), "launch", "--config", "specs/example.json", "--dry-run"]
    monkeypatch.setattr(choose.sys, "platform", "darwin")
    monkeypatch.setattr(choose.Path, "home", lambda: tmp_path)
    answers = iter(["SF Compute", "/repo/spec.json", "/private/policy.json", "H100 80 GB (1 GPU)"])
    monkeypatch.setattr(choose, "dialog", lambda *a: next(answers))
    assert choose.main([]) == 0
    assert calls[-1][2:] == ["launch", "--config", "/repo/spec.json", "--policy", "/private/policy.json", "--gpu-template", "h100-1"]


def test_resume_freezes_workspace_and_branch(tmp_path, monkeypatch):
    monkeypatch.setattr(shared, "ROOT", tmp_path)
    cache = shared.cache_dir("branch-one", "session", "sfcompute")
    common.atomic_json(cache / "launch.json", {"backend": "sfcompute", "branch": "branch-one", "session": "session",
        "instance_id": "node-id", "label": "saved-node", "policy": policy.validate({"workspace": "lab"})})
    args = run.build_parser().parse_args(["status", "--branch", "branch-one", "--session", "session"])
    shared.load_plan(args)
    assert args.workspace == "lab" and args.sfcompute_identity == ("node-id", "saved-node")
    args.workspace = "other-lab"
    with pytest.raises(ValueError, match="connection differs"):
        shared.load_plan(args)
    assert shared.cache_dir("branch-two", "session", "sfcompute") != cache


def test_upload_and_balance_preflight_happens_without_rental():
    calls = []
    def request(method, path, **kw):
        calls.append((method, path))
        return ({"limits": [{"name": "max_chunked_write_bytes", "value": 100},
                             {"name": "node_scratch_gib", "value": 250}]} if path == "/limits"
                else {"balance": {"available_usd": 525}})
    client = SimpleNamespace(request=request)
    run.preflight_account(client, policy.validate({}), [99])
    assert all(method == "get" for method, _ in calls)
    with pytest.raises(ValueError, match="upload limit"):
        run.preflight_account(client, policy.validate({}), [101])
    client.request = lambda method, path, **kw: ({"limits": [{"name": "max_chunked_write_bytes", "value": 100},
        {"name": "node_scratch_gib", "value": 250}]} if path == "/limits" else {"balance": {"available_usd": float("nan")}})
    with pytest.raises(ValueError, match="credit"):
        run.preflight_account(client, policy.validate({}), [99])


def test_incompatible_catalog_image_is_rejected_before_any_rental(tmp_path, monkeypatch):
    spec = tmp_path / "experiment.json"
    spec.write_text('{"run_name":"tiny", "defaults":{"FA":2}}')
    configured = tmp_path / "policy.json"
    configured.write_text(json.dumps(policy.DEFAULTS))
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(shared, "client_for", lambda *a: pytest.fail("no provider request before image validation"))
    with pytest.raises(ValueError, match="no node was requested"):
        run.main(["launch", "--config", str(spec), "--policy", str(configured), "--branch", "test"])
    assert not run.image_warnings({"defaults": {"FA": 4}}, policy.validate({}))
    assert run.image_warnings({"defaults": {"FA": 4}}, policy.validate({"image": "pytorch-2.13-cuda12.9"}))


def test_paid_launch_records_identity_and_reuses_shared_handoff(tmp_path, monkeypatch):
    spec = tmp_path / "experiment.json"
    spec.write_text('{"run_name":"tiny", "defaults":{"FA":4}, "skip_evals":true}')
    configured = tmp_path / "policy.json"
    configured.write_text(json.dumps(policy.DEFAULTS))
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(shared, "ROOT", tmp_path)
    monkeypatch.setattr(shared, "preview", lambda _: {"pretrain": {}, "finetunes": {}, "evals": {}})
    monkeypatch.setattr(shared, "notes", lambda *a: ("test intent", "new backend"))
    monkeypatch.setattr(shared, "trial_provenance", lambda *a: None)
    monkeypatch.setattr(run, "preflight_account", lambda *a: None)
    def package(root, selected, archive, **kwargs):
        with tarfile.open(archive, "w:gz") as output:
            output.add(spec, arcname="experiment.json")
        return common.hashes(archive)
    monkeypatch.setattr(snapshot, "create", package)
    store = SimpleNamespace(config={"test": True}, preflight=lambda: None, create_run=lambda _: "drive-folder")
    monkeypatch.setattr(shared, "store_for", lambda *a, **kw: store)
    requests = []
    class Client:
        def create(self, policy, name):
            saved = json.loads((shared.cache_dir("test", "session", "sfcompute") / "launch.json").read_text())
            assert saved["label"] == name and saved["instance_id"] is None
            requests.append("create")
            return node(name=name)
    monkeypatch.setattr(shared, "client_for", lambda *a: Client())
    handoffs = []
    monkeypatch.setattr(shared, "handoff", lambda args, cache, plan, client: handoffs.append(plan))
    assert run.main(["launch", "--branch", "test", "--session", "session", "--config", str(spec),
                     "--policy", str(configured), "--detach"]) == 0
    assert requests == ["create"] and handoffs[0]["instance_id"] == "node-id"
    assert "ssh_key_path" not in handoffs[0] and handoffs[0]["drive_folder_id"] == "drive-folder"
    with pytest.raises(ValueError, match="already has records"):
        run.main(["launch", "--branch", "test", "--session", "session", "--config", str(spec), "--policy", str(configured)])
    assert requests == ["create"]


def test_shared_handoff_transfers_private_token_separately_and_uses_no_ssh(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    for filename, text in [("source.tar.gz", "source"), ("launch.json", "{}"), ("spec.json", "{}")]:
        (cache / filename).write_text(text)
    token = tmp_path / "dedicated-drive-token.json"
    token.write_text("secret-drive-token")
    configured = policy.validate({})
    plan = {"backend": "sfcompute", "branch": "test", "session": "session", "instance_id": "node-id", "label": "saved-node",
            "deadline_unix": shared.time.time() + 1000, "policy": configured, "remote_workspace": "/home/dev/run/session",
            "source": {"sha256": "expected-source", "size": 6}, "run_name": "tiny"}
    calls = []
    class Connection:
        def json_file(self, path, **kwargs):
            return {"state": "preparing_data"} if path.endswith("/status.json") else None
        def python(self, *args, **kwargs):
            calls.append(("python", args))
            return b"ready\n"
        def upload(self, path, target, **kwargs):
            calls.append(("upload", str(path), target))
        def launch(self, *args):
            calls.append(("launch", args))
    client = SimpleNamespace(instance=lambda _: api.normalize(node()), raw_node=lambda _: node())
    monkeypatch.setattr(shared, "connection_for", lambda *a: Connection())
    monkeypatch.setattr(shared, "SSH", lambda *a: pytest.fail("SF Compute has no SSH"))
    args = SimpleNamespace(drive_token=token, secrets_file=None)
    shared.handoff(args, cache, plan, client)
    assert ("upload", str(token), "/home/dev/run/session/private/drive-token.json") in calls
    assert len([c for c in calls if c[0] == "launch"]) == 1
    assert "CONTAINER_API_KEY" not in str(calls) and "secret-drive-token" not in str(calls)


def test_worker_backend_retains_backup_cleanup_and_removes_account_tokens(tmp_path, monkeypatch):
    plan = {"backend": "sfcompute", "branch": "test", "run_name": "tiny", "policy": policy.validate({})}
    monkeypatch.setenv("GMN_TOKEN", "local-account-secret")
    env = worker.clean_training_env(plan, tmp_path / "private")
    assert "GMN_TOKEN" not in env and env["TMPDIR"] == "/tmp"
    assert env["MDLARC_SKIP_CUDA_VALIDATION"] == "1"
    assert worker.cleanup_action({"ok": True, "finalized": True, "backups_complete": True}, plan["policy"], plan) == "stop"
    assert worker.cleanup_action({"ok": True, "finalized": True, "backups_complete": False}, plan["policy"], plan) == "keep"


def test_sfcompute_records_sync_but_private_and_live_state_do_not():
    assert sync_records.selected("results/branch/sfcompute_runs/session/spec.json")
    assert not sync_records.selected("results/branch/sfcompute_runs/session/launch.json")
    assert not sync_records.selected("results/branch/sfcompute_runs/session/private/drive-token.json")
    assert not sync_records.selected("results/branch/sfcompute_runs/session/source.tar.gz")
