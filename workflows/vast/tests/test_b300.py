"""B300 rentals stay powered on; scientific retries keep their frozen inputs."""

import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

from workflows.vast import common, run, ssh, vast_api, watchdog, worker


@pytest.mark.parametrize("plan,instance", [
    ({"gpu_name": "NVIDIA B300"}, None),
    ({"adopted_instance": {"gpu_name": "B300"}}, None),
    ({"policy": {"gpu_names": ["B300"]}}, None),
    ({"preserve_instance": True}, None),
    ({"instance_id": 57431449}, None),
    ({}, {"gpu_name": "B300"}),
])
def test_b300_protection_survives_saved_and_live_metadata(plan, instance):
    assert common.keep_b300_running(plan, instance)


@pytest.mark.parametrize("status", [
    {"ok": False, "backups_complete": False, "finalized": False},
    {"ok": False, "backups_complete": True, "finalized": True},
    {"ok": True, "backups_complete": True, "finalized": True},
])
def test_b300_cleanup_keeps_capacity_even_when_saved_policy_says_stop_or_destroy(status):
    policy = {"success_action": "destroy", "failure_action": "stop"}
    assert worker.cleanup_action(status, policy, {"gpu_name": "B300"}) == "keep"


def test_b300_deadline_guard_never_obtains_stop_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "instance_client", lambda _: pytest.fail("B300 deadline cannot stop the instance"))
    assert worker.deadline_guard(tmp_path, {"gpu_name": "B300", "deadline_unix": 0}) == 0
    assert json.loads((tmp_path / "deadline-status.json").read_text())["state"] == "keep_b300_running"


def test_automatic_stop_protects_b300_and_still_checks_ownership():
    stopped = []
    instance = {"id": 7, "label": "mine", "gpu_name": "B300"}
    client = SimpleNamespace(instance=lambda _: instance, stop=stopped.append)
    plan = {"instance_id": 7, "label": "mine"}
    assert run.stop_owned(client, plan) == "keep_b300_running"
    assert stopped == []
    instance["label"] = "someone-else"
    with pytest.raises(ValueError):
        run.stop_owned(client, plan)
    assert stopped == []


def test_explicit_stop_command_can_honor_a_later_user_request():
    stopped = []
    client = SimpleNamespace(instance=lambda _: {"label": "mine", "gpu_name": "B300"}, stop=stopped.append)
    run.stop_owned(client, {"instance_id": 7, "label": "mine"}, explicit=True)
    assert stopped == [7]


def test_watchdog_protects_b300_with_a_legacy_lease():
    client = SimpleNamespace(instances=lambda: [{"id": 7, "label": "mine", "gpu_name": "B300", "actual_status": "running"}],
                             stop=lambda _: pytest.fail("Deadline must retain the B300"))
    assert watchdog.enforce(client, {"instance_id": 7, "label": "mine", "deadline_unix": 1}, 2) == "keep_b300_running"


@pytest.mark.parametrize('direct', [False, True])
def test_explicit_direct_ssh_uses_only_advertised_host_port(tmp_path, monkeypatch, direct):
    monkeypatch.setenv('MDLARC_SSH_DIRECT', '1' if direct else '0')
    instance = {'ssh_host': 'proxy.example', 'ssh_port': 1000, 'public_ipaddr': '192.0.2.1',
                'ports': {'22/tcp': [{'HostPort': '2000'}]}}
    connection = ssh.SSH(instance, tmp_path/'key', tmp_path/'known_hosts')
    assert (connection.host, connection.port) == (('192.0.2.1', 2000) if direct else ('proxy.example', 1000))


@pytest.mark.parametrize('method,endpoint,body', [
    ('PUT', '/instances/57431449/', {'state': 'stopped'}),
    ('DELETE', '/instances/57431449/', None),
])
def test_protected_b300_shutdown_is_blocked_before_any_api_request(method, endpoint, body):
    client = vast_api.Vast('dummy', opener=SimpleNamespace(open=lambda *a, **kw: pytest.fail('Shutdown request escaped protection')))
    with pytest.raises(vast_api.ApiError, match='protected'):
        client.request(method, endpoint, body)


def test_controller_refresh_preserves_spec_data_and_all_training_source(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    frozen = tmp_path / "frozen"
    cache = tmp_path / "cache"
    cache.mkdir()
    originals = {"spec.json": b'{"pretrain":{"batch_size":12,"gradient_accumulation_steps":5}}',
                 "src/model.py": b"FROZEN_MODEL = True\n", "data.zip": b"pinned data",
                 "workflows/vast/common.py": b"old common\n", "workflows/vast/worker.py": b"old worker\n",
                 "workflows/vast/vast_api.py": b"old client\n"}
    for name, data in originals.items():
        path = frozen / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    for name in ("workflows/vast/common.py", "workflows/vast/worker.py", "workflows/vast/vast_api.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"updated no-stop controller\n")
    with tarfile.open(cache / "source.tar.gz", "w:gz") as archive:
        for name in originals:
            archive.add(frozen / name, arcname=name)
    before = (cache / "source.tar.gz").read_bytes()
    plan = {"branch": "scaling-ladder", "session": "retry", "instance_id": 7, "label": "mine",
            "gpu_name": "B300", "deadline_unix": 1234, "spec_sha256": common.hashes(frozen / "spec.json")["sha256"],
            "source": {"files": len(originals), **common.hashes(cache / "source.tar.gz")}}
    monkeypatch.setattr(run, "ROOT", root)
    run.refresh_b300_controllers(cache, plan)
    assert plan["preserve_instance"] is True and plan["deadline_unix"] == 1234
    assert common.hashes(cache / "source.tar.gz")["sha256"] == plan["source"]["sha256"]
    assert next(cache.glob("source-before-*.tar.gz")).read_bytes() == before
    with tarfile.open(cache / "source.tar.gz") as archive:
        assert set(archive.getnames()) == set(originals)
        for name, data in originals.items():
            actual = archive.extractfile(name).read()
            assert actual == (b"updated no-stop controller\n" if name.startswith("workflows/") else data)
    assert len(plan["controller_updates"]) == 1
    run.refresh_b300_controllers(cache, plan)
    assert len(plan["controller_updates"]) == 1
    assert json.loads((cache / "lease.json").read_text())["preserve_instance"] is True
