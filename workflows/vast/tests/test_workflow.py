"""Offline checks of rental safety, Drive verification, and remote lifecycle."""

import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from types import ModuleType, SimpleNamespace
from urllib.error import URLError
from urllib.parse import parse_qs, urlsplit

import httplib2
from googleapiclient.errors import HttpError
import pytest

from workflows.vast import common, drive, policy, run, snapshot, vast_api, watchdog, worker, choose, ssh


def test_ssh_known_hosts_preserves_encoded_branch_path(tmp_path):
    if not shutil.which("ssh"):
        pytest.skip("OpenSSH is unavailable")
    known_hosts = tmp_path / "codex%2Fdepth16-research%25" / "known_hosts"
    connection = ssh.SSH({"ssh_host": "127.0.0.1", "ssh_port": 22},
                         tmp_path / "identity", known_hosts)
    command = connection.command(["true"])
    # Resolve OpenSSH's actual config without making a network connection.
    command.insert(1, "-G")
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert f"userknownhostsfile {known_hosts.resolve()}" in result.stdout.splitlines()


def offer(**overrides):
    return {"id": 10, "verified": True, "rentable": True, "rented": False, "num_gpus": 1,
            "gpu_name": "A100 SXM4", "gpu_ram": 80000, "cpu_ram": 64000, "disk_space": 200,
            "cpu_cores": 64, "cpu_cores_effective": 16,
            "cuda_max_good": 13.4,
            "reliability": 0.999, "inet_up": 100, "inet_down": 100, "duration": 40000,
            "dph_total": 1, "inet_up_cost": 0.01, "inet_down_cost": 0.01, **overrides}


def configured():
    return policy.validate({"image": "example/pinned:1", "gpu_names": ["A100_SXM4"],
        "min_gpu_memory_gb": 75, "min_cuda_version": 13.4}, launching=True)


@pytest.mark.parametrize("field,value", [("schema_version", 2), ("max_total_usd", 0), ("num_gpus", True),
    ("min_reliability", 2), ("max_runtime_hours", float("nan")), ("remote_root", "/"), ("failure_action", "destroy")])
def test_policy_rejects_invalid_constraints(field, value):
    with pytest.raises(ValueError):
        policy.validate({field: value})


def test_policy_placeholder_and_override():
    with pytest.raises(ValueError):
        policy.validate({}, launching=True)
    assert policy.offer_query(configured())["gpu_ram"] == {"gte": 75000}
    assert policy.offer_query(configured())["duration"]["gte"] == 30600
    assert policy.offer_query(configured())["gpu_name"] == {"in": ["A100 SXM4"]}
    assert policy.offer_query(configured())["cuda_max_good"] == {"gte": 13.4}


@pytest.mark.parametrize("value", [True, 0, -1, "13.4", float("nan"), float("inf")])
def test_policy_rejects_invalid_cuda_minimum(value):
    with pytest.raises(ValueError):
        policy.validate({"min_cuda_version": value})


def test_new_rental_requires_explicit_cuda_minimum():
    value = {"image": "example/pinned:1", "gpu_names": ["A100_SXM4"]}
    assert policy.validate(value)["min_cuda_version"] is None
    with pytest.raises(ValueError, match="min_cuda_version"):
        policy.validate(value, launching=True)


@pytest.mark.parametrize("field", ["min_cpu_cores_effective", "min_download_mbps"])
@pytest.mark.parametrize("value", [True, None, -1, "16", float("nan"), float("inf")])
def test_policy_rejects_invalid_capacity_minimum(field, value):
    with pytest.raises(ValueError, match=field):
        policy.validate({field: value})


def capacity_policy():
    return policy.validate({**configured(), "min_cpu_cores_effective": 16, "min_download_mbps": 2000})


def test_capacity_search_filters_use_allocated_cpu_and_mbps():
    query = policy.offer_query(capacity_policy())
    assert query["cpu_cores_effective"] == {"gte": 16}
    assert "cpu_cores" not in query
    assert query["inet_down"] == {"gte": 2000}
    # Existing policies can leave the new CPU constraint disabled.
    assert "cpu_cores_effective" not in policy.offer_query(configured())


@pytest.mark.parametrize("field,value", [
    ("cpu_cores_effective", 8), ("cpu_cores_effective", 15.9),
    ("inet_down", 723.7), ("inet_down", 1999.9),
] + [(field, value) for field in ("cpu_cores_effective", "inet_down")
     for value in (None, True, "2000", float("nan"), float("inf"))])
def test_capacity_is_rechecked_even_when_search_ignores_filters(field, value):
    row = offer(**{"cpu_cores": 128, "inet_down": 2500, field: value})
    with pytest.raises(ValueError, match=field):
        policy.quote_offer(row, capacity_policy())


@pytest.mark.parametrize("field", ["cpu_cores_effective", "inet_down"])
def test_missing_capacity_metadata_is_rejected(field):
    row = offer(inet_down=2500)
    row.pop(field)
    with pytest.raises(ValueError, match=field):
        policy.quote_offer(row, capacity_policy())


def test_capacity_filter_excludes_cheaper_slow_or_small_allocations():
    _, selected = policy.choose_offer([
        offer(id=1, cpu_cores=128, cpu_cores_effective=8, inet_down=2500, dph_total=0.1),
        offer(id=2, inet_down=723.7, dph_total=0.2),
        offer(id=3, cpu_cores=32, cpu_cores_effective=16, inet_down=2000),
    ], capacity_policy())
    assert selected["id"] == 3


@pytest.mark.parametrize("value", [None, True, "13.4", 12.7, 13.3, float("nan"), float("inf")])
def test_offer_cuda_minimum_is_rechecked_even_when_search_ignores_filter(value):
    with pytest.raises(ValueError, match="CUDA support"):
        policy.quote_offer(offer(cuda_max_good=value), configured())


def test_missing_cuda_metadata_is_rejected_before_rental():
    value = offer()
    value.pop("cuda_max_good")
    with pytest.raises(ValueError, match="CUDA support"):
        policy.quote_offer(value, configured())


def test_cuda_filter_excludes_cheaper_incompatible_host():
    quote, selected = policy.choose_offer([
        offer(id=1, cuda_max_good=12.7, dph_total=0.1),
        offer(id=2, cuda_max_good=13.3, dph_total=0.2),
        offer(id=3, cuda_max_good=13.4),
        offer(id=4, cuda_max_good=13.5, dph_total=1.5),
    ], configured())
    assert selected["id"] == 3
    with pytest.raises(RuntimeError, match="nothing was rented"):
        policy.choose_offer([offer(cuda_max_good=12.7)], configured())


@pytest.mark.parametrize("overrides", [
    {"cuda_max_good": 12.7}, {"cpu_cores_effective": 8}, {"inet_down": 723.7},
    {"cpu_cores_effective": None}, {"inet_down": None},
])
def test_launch_does_not_rent_or_create_drive_run_for_incompatible_offers(tmp_path, monkeypatch, overrides):
    root = tmp_path / "repo"
    root.mkdir()
    spec = root / "spec.json"
    spec.write_text('{"run_name":"smoke"}')
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(capacity_policy()))
    key = tmp_path / "identity"
    key.write_text("unused private key")
    monkeypatch.setattr(run, "ROOT", root)
    monkeypatch.setattr(run, "preview", lambda _: {})
    monkeypatch.setattr(run.snapshot, "create", lambda *args: {})
    monkeypatch.setattr(run, "public_key", lambda _: "public key")
    def refuse(*args):
        pytest.fail("Incompatible offers must be rejected before any rental or Drive run creation")
    def offers(query):
        assert query["cuda_max_good"] == {"gte": 13.4}
        assert query["cpu_cores_effective"] == {"gte": 16}
        assert query["inet_down"] == {"gte": 2000}
        return [offer(**{"inet_down": 2500, **overrides})]
    monkeypatch.setattr(run, "client_for", lambda _: SimpleNamespace(offers=offers, create=refuse))
    monkeypatch.setattr(run, "store_for", lambda *args: SimpleNamespace(preflight=lambda: None, create_run=refuse))
    args = run.build_parser().parse_args(["launch", "--branch", "newreason", "--session", "vast-test",
        "--config", str(spec), "--policy", str(policy_path), "--ssh-key", str(key),
        "--intent", "test", "--changed-from-previous", "CUDA guard"])
    with pytest.raises(RuntimeError, match="nothing was rented"):
        run.launch(args)


@pytest.mark.parametrize("overrides", [{"gpu_ram": 24000}, {"dph_total": 3}, {"reliability": 0.9},
    {"duration": 10}, {"inet_up_cost": 1}, {"verified": False}, {"gpu_name": "RTX 4090"},
    {"num_gpus": 2}, {"inet_down_cost": None}])
def test_offer_is_rechecked_before_rental(overrides):
    with pytest.raises(ValueError):
        policy.quote_offer(offer(**overrides), configured())


def test_choose_cheapest_with_bandwidth_reservation():
    quote, selected = policy.choose_offer([offer(id=1, dph_total=1.5), offer(id=2)], configured())
    assert selected["id"] == 2
    assert quote["estimated_max_usd"] == 8.7


@pytest.mark.parametrize("cuda,tag", [(13.2, "26.05"), (13.299, "26.05"),
    (13.3, "26.07"), (13.399, "26.07"), (13.4, "26.09"), (13.5, "26.09")])
def test_ngc_image_matches_offer_cuda_without_changing_other_limits(cuda, tag):
    base = policy.apply_template(configured(), "a100-80gb")
    selected = policy.resolve_image(base, offer(cuda_max_good=cuda))
    assert selected["image"] == f"nvcr.io/nvidia/pytorch:{tag}-py3"
    assert selected["min_cuda_version"] == min(int(cuda * 10) / 10, 13.4)
    assert {k: v for k, v in selected.items() if k not in {"image", "min_cuda_version"}} == {
        k: v for k, v in base.items() if k not in {"image", "min_cuda_version"}}
    assert base["image"] == "nvcr.io/nvidia/pytorch:26.05-py3"


def test_ngc_selection_preserves_fixed_custom_and_newer_base_images():
    base = policy.apply_template(configured(), "a100-80gb")
    fixed = policy.validate({**base, "image_selection": "fixed"})
    assert policy.resolve_image(fixed, offer())["image"] == base["image"]
    custom = configured()
    assert policy.resolve_image(custom, offer()) == custom
    newer = policy.validate({**base, "image": "nvcr.io/nvidia/pytorch:26.07-py3", "min_cuda_version": 13.3})
    assert [rule["min_cuda_version"] for rule in policy.image_rules(newer)] == [13.3, 13.4]
    with pytest.raises(ValueError, match="CUDA support"):
        policy.resolve_image(newer, offer(cuda_max_good=13.2))
    with pytest.raises(ValueError, match="min_cuda_version"):
        policy.validate({**base, "image": "nvcr.io/nvidia/pytorch:26.09-py3"})


@pytest.mark.parametrize("value", [None, True, "auto", [], {}])
def test_policy_rejects_invalid_image_selection(value):
    with pytest.raises(ValueError, match="image_selection"):
        policy.validate({"image_selection": value})


@pytest.mark.parametrize("value", [True, -1, 0, "45", float("nan"), float("inf"), 74])
def test_policy_rejects_invalid_gpu_memory_ceiling(value):
    with pytest.raises(ValueError, match="max_gpu_memory_gb"):
        policy.validate({**configured(), "max_gpu_memory_gb": value})


@pytest.mark.parametrize("name,gpu,memory", [
    ("a100-40gb", "A100 SXM4", 40960), ("a100-80gb", "A100 PCIE", 81920),
    ("h100", "H100 NVL", 95830), ("h200", "H200 NVL", 143771), ("b200", "B200", 183359),
])
def test_templates_restrict_gpu_and_preserve_saved_budget(name, gpu, memory):
    base = policy.validate({**configured(), "max_hourly_usd": 1.5, "max_total_usd": 13,
                           "min_download_mbps": 2000, "failure_action": "keep", "countries": ["US"]})
    selected = policy.apply_template(base, name)
    for key in ("max_hourly_usd", "max_total_usd", "max_runtime_hours", "max_upload_gb",
                "max_download_gb", "disk_gb", "min_cpu_ram_gb", "min_cpu_cores_effective",
                "min_download_mbps", "success_action", "failure_action", "countries"):
        assert selected[key] == base[key]
    policy.quote_offer(offer(gpu_name=gpu, gpu_ram=memory, inet_down=2500, country="US"), selected)
    with pytest.raises(ValueError):
        policy.quote_offer(offer(gpu_name="RTX 4090", gpu_ram=memory, inet_down=2500, country="US"), selected)


def test_a100_templates_cannot_select_each_others_memory_variant():
    small = policy.apply_template(configured(), "a100-40gb")
    large = policy.apply_template(configured(), "a100-80gb")
    assert policy.offer_query(small)["gpu_ram"] == {"gte": 35000, "lte": 45000}
    assert policy.offer_query(large)["gpu_ram"] == {"gte": 75000, "lte": 85000}
    with pytest.raises(ValueError, match="max_gpu_memory_gb"):
        policy.quote_offer(offer(gpu_ram=81920), small)
    with pytest.raises(ValueError, match="gpu_ram"):
        policy.quote_offer(offer(gpu_ram=40960), large)
    assert policy.choose_offer([offer(id=1, gpu_ram=81920, dph_total=0.1),
                               offer(id=2, gpu_ram=40960)], small)[1]["id"] == 2


def test_template_cannot_override_cost_policy(tmp_path, monkeypatch):
    (tmp_path / "h200.json").write_text('{"max_hourly_usd":100}')
    monkeypatch.setattr(policy, "TEMPLATE_DIR", tmp_path)
    with pytest.raises(ValueError, match="only override"):
        policy.apply_template(configured(), "h200")


def test_cuda_selection_keeps_least_expensive_offer():
    base = policy.apply_template(configured(), "a100-80gb")
    _, selected = policy.choose_offer([offer(id=1, cuda_max_good=13.2, dph_total=0.5),
                                     offer(id=2, cuda_max_good=13.4)], base)
    assert selected["id"] == 1
    assert policy.resolve_image(base, selected)["image"] == "nvcr.io/nvidia/pytorch:26.05-py3"


def test_launch_pins_explicit_image_and_allows_fixed_selection(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(configured()))
    args = run.build_parser().parse_args(["launch", "--config", "specs/example.json", "--policy", str(path),
        "--gpu-template", "h200", "--image", "example/pinned:2", "--dry-run"])
    selected = run.launch_policy(args)
    assert selected["image"] == "example/pinned:2"
    assert selected["image_selection"] == "fixed"
    args.image = None
    args.image_selection = "fixed"
    assert run.launch_policy(args)["image_selection"] == "fixed"


@pytest.mark.parametrize("cuda,tag", [(13.2, "26.05"), (13.3, "26.07"), (13.4, "26.09")])
def test_launch_rents_and_saves_same_resolved_image(tmp_path, monkeypatch, cuda, tag):
    root = tmp_path / "repo"
    root.mkdir()
    spec = root / "spec.json"
    spec.write_text('{"run_name":"smoke"}')
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(configured()))
    key = tmp_path / "identity"
    key.write_text("unused private key")
    monkeypatch.setattr(run, "ROOT", root)
    monkeypatch.setattr(run, "preview", lambda _: {})
    def package(root, spec_path, destination):
        with tarfile.open(destination, "w:gz") as archive:
            archive.add(spec_path, arcname="spec.json")
        return common.hashes(destination)
    monkeypatch.setattr(run.snapshot, "create", package)
    monkeypatch.setattr(run, "public_key", lambda _: "public key")
    requests = []
    class Opener:
        def open(self, request, timeout):
            requests.append(json.loads(request.data))
            return io.BytesIO(b'{"new_contract":123}')
    client = vast_api.Vast("synthetic-test-key", opener=Opener())
    client.offers = lambda _: [offer(gpu_name="H200", gpu_ram=143771, cuda_max_good=cuda)]
    monkeypatch.setattr(run, "client_for", lambda _: client)
    monkeypatch.setattr(run, "store_for", lambda *args: SimpleNamespace(
        preflight=lambda: None, create_run=lambda _: "synthetic-folder", config={}))
    monkeypatch.setattr(run, "handoff", lambda *args: None)
    args = run.build_parser().parse_args(["launch", "--branch", "newreason", "--session", "vast-test",
        "--config", str(spec), "--policy", str(policy_path), "--ssh-key", str(key),
        "--intent", "test", "--changed-from-previous", "GPU template", "--gpu-template", "h200", "--detach"])
    assert run.launch(args) == 0
    _, saved = run.load_plan(args)
    image = f"nvcr.io/nvidia/pytorch:{tag}-py3"
    assert len(requests) == 1
    assert requests[0]["image"] == saved["policy"]["image"] == image
    assert saved["runtime_compatibility"]["image"] == image
    assert saved["runtime_compatibility"]["offer_cuda_max_good"] == cuda
    assert saved["policy"]["min_cuda_version"] == cuda
    assert saved["gpu_template"] == "h200"
    policy_path.write_text(json.dumps(policy.apply_template(configured(), "b200")))
    assert run.load_plan(args)[1]["policy"]["image"] == image


def test_templates_listing_is_offline_and_uses_saved_limits(tmp_path, monkeypatch, capsys):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({**configured(), "max_hourly_usd": 0.8}))
    def refuse(*args):
        pytest.fail("Template listing must remain offline")
    monkeypatch.setattr(run, "client_for", refuse)
    monkeypatch.setattr(run, "store_for", refuse)
    assert run.main(["templates", "--policy", str(path), "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [row["name"] for row in rows] == ["saved", *policy.GPU_TEMPLATES]
    assert all(row["policy"]["max_hourly_usd"] == 0.8 for row in rows)


def test_unknown_rental_is_submitted_once():
    class Opener:
        calls = 0
        def open(self, request, timeout):
            self.calls += 1
            assert request.get_header("Authorization") == "Bearer private-key"
            assert b"private-key" not in request.data
            raise URLError("private-key should never be reported")
    opener = Opener()
    client = vast_api.Vast("private-key", opener=opener, sleep=lambda _: None)
    with pytest.raises(vast_api.AmbiguousSubmission) as exc:
        client.create(10, configured(), "mdlarc:branch:session")
    assert opener.calls == 1
    assert "private-key" not in str(exc.value)


def test_retry_read_only_requests():
    class Opener:
        calls = 0
        def open(self, request, timeout):
            self.calls += 1
            if self.calls < 3:
                raise URLError("network")
            return io.BytesIO(b'{"instances": []}')
    opener = Opener()
    assert vast_api.Vast("private", opener=opener, sleep=lambda _: None).instances() == []
    assert opener.calls == 3


def test_instance_reconciliation_reads_all_v1_pages():
    class Opener:
        calls = []
        def open(self, request, timeout):
            url = urlsplit(request.full_url)
            query = parse_qs(url.query)
            assert url.path == "/api/v1/instances/"
            assert request.get_method() == "GET"
            assert request.data is None
            assert query["limit"] == ["25"]
            assert json.loads(query["order_by"][0]) == [{"col": "id", "dir": "asc"}]
            self.calls.append(query)
            if len(self.calls) == 1:
                page = {"instances": [{"id": 1, "label": "unrelated"}], "next_token": "cursor+with/slash="}
            else:
                assert query["after_token"] == ["cursor+with/slash="]
                page = {"instances": [{"id": 2, "label": "saved-rental"}], "next_token": None}
            return io.BytesIO(json.dumps(page).encode())
    opener = Opener()
    rows = vast_api.Vast("private", opener=opener).instances()
    assert rows[-1] == {"id": 2, "label": "saved-rental"}
    assert len(opener.calls) == 2


@pytest.mark.parametrize("pages", [
    [{"success": True}],
    [{"instances": [{"id": "1"}]}],
    [{"instances": [], "next_token": 1}],
    [{"instances": [], "next_token": ""}],
    [{"instances": [], "next_token": "loop"}, {"instances": [], "next_token": "loop"}],
    [{"instances": [{"id": 1}], "next_token": "next"}, {"instances": [{"id": 1}]}],
])
def test_instance_reconciliation_rejects_incomplete_or_inconsistent_pages(pages):
    class Opener:
        def open(self, request, timeout):
            return io.BytesIO(json.dumps(pages.pop(0)).encode())
    with pytest.raises(vast_api.ApiError):
        vast_api.Vast("private", opener=Opener()).instances()


def test_instance_record_does_not_persist_provider_secrets():
    assert vast_api.public_instance({"id": 1, "jupyter_token": "secret", "extra_env": ["secret"]}) == {"id": 1}


def test_only_public_ssh_key_is_read_and_attached(tmp_path):
    import base64
    identity = tmp_path / "identity"
    identity.write_text("PRIVATE KEY MUST NOT BE READ")
    kind = b"ssh-ed25519"
    blob = len(kind).to_bytes(4, "big") + kind + (32).to_bytes(4, "big") + b"a" * 32
    public = "ssh-ed25519 " + base64.b64encode(blob).decode()
    identity.with_suffix(".pub").write_text(public + " personal-comment\n")
    class Opener:
        def open(self, request, timeout):
            assert request.full_url == "https://console.vast.ai/api/v0/instances/7/ssh/"
            assert request.get_method() == "POST"
            assert json.loads(request.data) == {"ssh_key": public}
            assert b"PRIVATE" not in request.data and b"personal-comment" not in request.data
            return io.BytesIO(b'{"success": true}')
    vast_api.Vast("private", opener=Opener()).attach_ssh_key(7, ssh.public_key(identity))


@pytest.mark.parametrize("content", [None, "-----BEGIN OPENSSH PRIVATE KEY-----", "ssh-ed25519 invalid!", "ssh-rsa YWJjZA=="])
def test_missing_or_invalid_public_ssh_key_is_rejected(tmp_path, content):
    identity = tmp_path / "identity"
    if content is not None:
        identity.with_suffix(".pub").write_text(content)
    with pytest.raises(ValueError, match="matching OpenSSH public key"):
        ssh.public_key(identity)


class Request:
    def __init__(self, callback):
        self.callback = callback
        self.resumable_uri = None
        self.resumable_progress = 0
    def execute(self, **kwargs):
        return self.callback()


class FakeDrive:
    def __init__(self):
        self.entries = {
            common.ROOT_FOLDER: {"id": common.ROOT_FOLDER, "name": "mdlARC", "mimeType": drive.FOLDER, "parents": []},
            "runs-id": {"id": "runs-id", "name": "runs", "mimeType": drive.FOLDER, "parents": [common.ROOT_FOLDER], "capabilities": {"canAddChildren": True}},
        }
        self.bytes = {}
        self.serial = 0
        self.email = common.ACCOUNT
        self.create_calls = 0
        self.ambiguous_upload = False
    def about(self):
        return SimpleNamespace(get=lambda **kwargs: Request(lambda: {"user": {"emailAddress": self.email}}))
    def files(self):
        return self
    def get(self, fileId, **kwargs):
        def value():
            if fileId not in self.entries:
                raise HttpError(httplib2.Response({"status": "404"}), b"not found")
            return dict(self.entries[fileId])
        return Request(value)
    def list(self, q, **kwargs):
        parent = re.search(r"'([^']+)' in parents", q)[1]
        name = re.search(r"name = '([^']+)'", q)
        return Request(lambda: {"files": [dict(value) for value in self.entries.values()
            if value.get("parents") == [parent] and (not name or value["name"] == name[1])
            and ("mimeType =" not in q or value["mimeType"] == drive.FOLDER)]})
    def generateIds(self, **kwargs):
        self.serial += 1
        return Request(lambda: {"ids": [f"generated-{self.serial}"]})
    def create(self, body, media_body=None, **kwargs):
        self.create_calls += 1
        def value():
            result = {**body}
            if media_body:
                path = Path(media_body._filename)
                digest = common.hashes(path)
                result.update(size=str(digest["size"]), md5Checksum=digest["md5"], sha256Checksum=digest["sha256"], mimeType="application/octet-stream")
                self.bytes[body["id"]] = path.read_bytes()
            self.entries[body["id"]] = result
            return result
        request = Request(value)
        if media_body:
            def chunk(**kwargs):
                result = value()
                request.resumable_uri = "private-upload-uri"
                request.resumable_progress = int(result["size"])
                if self.ambiguous_upload:
                    self.ambiguous_upload = False
                    raise RuntimeError("lost response after completion")
                return None, result
            request.next_chunk = chunk
        return request


def make_store(tmp_path, api=None, **kwargs):
    api = api or FakeDrive()
    return drive.Drive(api, {"root_folder_id": common.ROOT_FOLDER, "runs_folder_id": "runs-id"}, "newreason", control=tmp_path / "control", **kwargs)


def test_drive_transport_accepts_resumable_progress_without_redirect_location(tmp_path, monkeypatch):
    from googleapiclient.http import HttpRequest, MediaFileUpload
    import googleapiclient.discovery

    class Credentials:
        def refresh(self, request):
            pass
        def before_request(self, request, method, url, headers):
            headers["authorization"] = "Bearer test-token"

    monkeypatch.setattr(drive, "credentials", lambda _: Credentials())
    monkeypatch.setattr(googleapiclient.discovery, "build", lambda *args, **kwargs: kwargs["http"])
    http = drive.service(tmp_path / "unused-token.json")
    calls = []
    responses = iter([
        (httplib2.Response({"status": "308", "range": "bytes=0-262143"}), b""),
        (httplib2.Response({"status": "200"}), b'{"id":"saved-checkpoint"}'),
    ])
    def connection_request(connection, uri, method, body, headers):
        calls.append(headers["content-range"])
        return next(responses)
    monkeypatch.setattr(http.http, "_conn_request", connection_request)
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"x" * 500000)
    request = HttpRequest(http, lambda response, content: json.loads(content),
        "https://www.googleapis.com/upload/drive/v3/files", method="POST",
        resumable=MediaFileUpload(str(path), chunksize=256 * 1024, resumable=True))
    request.resumable_uri = "https://www.googleapis.com/upload/drive/v3/files?upload_id=test"
    progress, result = request.next_chunk(num_retries=0)
    assert result is None and progress.resumable_progress == 262144
    _, result = request.next_chunk(num_retries=0)
    assert result == {"id": "saved-checkpoint"}
    assert calls == ["bytes 0-262143/500000", "bytes 262144-499999/500000"]


def test_drive_rejects_personal_account_before_writes(tmp_path):
    api = FakeDrive()
    api.email = "personal@example.com"
    with pytest.raises(ValueError):
        make_store(tmp_path, api).preflight(create_branch=True)
    assert api.create_calls == 0


def test_drive_branch_and_session_are_verified(tmp_path):
    store = make_store(tmp_path)
    folder = store.create_run("vast-one")
    assert store.verify_run(folder, "vast-one")["name"] == "vast-one"
    store.api.entries[folder]["parents"] = [common.ROOT_FOLDER]
    with pytest.raises(ValueError):
        store.verify_run(folder)


def test_ambiguous_drive_upload_reconciles_same_id(tmp_path):
    store = make_store(tmp_path)
    folder = store.create_run("vast-one")
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"checkpoint")
    store.api.ambiguous_upload = True
    first = store.upload(path, folder, "checkpoint.pt")
    before = store.api.create_calls
    second = store.upload(path, folder, "checkpoint.pt")
    assert first == second
    assert store.api.create_calls == before
    assert "private-upload-uri" not in json.dumps(first)


def test_upload_bad_checksum_prevents_success(tmp_path):
    store = make_store(tmp_path)
    folder = store.create_run("vast-one")
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"checkpoint")
    result = store.upload(path, folder, "checkpoint.pt")
    store.api.entries[result["id"]]["md5Checksum"] = "bad"
    with pytest.raises(ValueError):
        store.upload(path, folder, "checkpoint.pt")


def test_transfer_allowance_shared_by_backup_processes(tmp_path):
    api = FakeDrive()
    first = make_store(tmp_path / "first", api, max_upload_gb=1e-8, transfer_file=tmp_path / "meter.json")
    second = make_store(tmp_path / "second", api, max_upload_gb=1e-8, transfer_file=tmp_path / "meter.json")
    first.charge("upload", 6)
    with pytest.raises(RuntimeError):
        second.charge("upload", 5)
    assert json.loads((tmp_path / "meter.json").read_text())["upload"] == 6


@pytest.mark.parametrize("name", ["../escape", "/absolute", "folder\\escape"])
def test_result_recovery_rejects_unsafe_paths(tmp_path, name):
    path = tmp_path / "bad.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        item = tarfile.TarInfo(name)
        item.size = 1
        archive.addfile(item, io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        common.extract_archive(path, tmp_path / "output")


def test_live_archives_exclude_checkpoints_and_unfinished_files(tmp_path):
    source = tmp_path / "run"
    source.mkdir()
    for name in ("epoch.pt", "final.safetensors", ".epoch.pt.tmp", "unfinished.partial", "console.log", "metrics.json"):
        (source / name).write_text("data")
    path = tmp_path / "result.tar.gz"
    snapshot.results_archive(source, path)
    with tarfile.open(path) as archive:
        assert set(archive.getnames()) == {"console.log", "metrics.json"}


def test_recovery_rejects_checkpoint_even_if_archive_hash_is_valid(tmp_path):
    path = tmp_path / "bad.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        item = tarfile.TarInfo("checkpoint.pt")
        item.size = 1
        archive.addfile(item, io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        common.extract_archive(path, tmp_path / "output", exclude_checkpoints=True)


def test_failure_never_destroys_instance():
    assert worker.cleanup_action({"ok": True, "backups_complete": False, "finalized": True}, configured()) == "stop"
    assert worker.cleanup_action({"ok": True, "backups_complete": True, "finalized": False}, configured()) == "stop"
    assert worker.cleanup_action({"ok": True, "backups_complete": True, "finalized": True}, configured()) == "destroy"


def test_changed_checkpoint_invalidates_final_backup_ledger(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"original")
    ledger = {"files": [{"path": "model.pt", **common.hashes(path)}]}
    assert worker.verify_checkpoint_ledger(tmp_path, ledger)
    path.write_bytes(b"modified")
    assert not worker.verify_checkpoint_ledger(tmp_path, ledger)


def test_rolling_resume_checkpoint_is_not_backed_up(tmp_path):
    for name in ("pretrain/tiny.rolling.pt", "pretrain/tiny.epoch050.pt", "pretrain/tiny.pt"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"checkpoint")
    assert worker.checkpoint_paths(tmp_path) == {"pretrain/tiny.epoch050.pt", "pretrain/tiny.pt"}
    ledger = {"files": [{"path": name, **common.hashes(tmp_path / name)}
                        for name in ("pretrain/tiny.epoch050.pt", "pretrain/tiny.pt")]}
    (tmp_path / "pretrain/tiny.rolling.pt").write_bytes(b"overwritten later")
    assert worker.verify_checkpoint_ledger(tmp_path, ledger)


def test_watchdog_handles_ambiguous_rental_and_checks_identity():
    stopped = []
    client = SimpleNamespace(instances=lambda: [{"id": 7, "label": "mine", "actual_status": "running"}], stop=stopped.append)
    lease = {"instance_id": None, "label": "mine", "deadline_unix": 1}
    assert watchdog.enforce(client, lease, 2) == "stop_requested"
    assert stopped == [7]
    with pytest.raises(ValueError):
        watchdog.enforce(client, {**lease, "instance_id": 7, "label": "different"}, 2)


def test_watchdog_network_failure_is_not_absence():
    client = SimpleNamespace(instances=lambda: (_ for _ in ()).throw(vast_api.ApiError("network")))
    with pytest.raises(vast_api.ApiError):
        watchdog.enforce(client, {"instance_id": 1, "label": "mine", "deadline_unix": 1}, 2)


@pytest.mark.parametrize("backend,entry", [
    ("vast", "workflows/vast/run.py"),
    ("colab", "scripts/colab_oauth/launch.py"),
    ("colab-oauth", "scripts/colab_oauth/launch.py"),
    ("colab-manual", "scripts/run_on_colab.py"),
])
def test_backend_selector_preserves_arguments(monkeypatch, backend, entry):
    captured = []
    monkeypatch.setattr(choose.subprocess, "call", lambda args, **kwargs: captured.append(args) or 0)
    assert choose.main(["--backend", backend, "--", "--config", "specs/example.json", "--dry-run"]) == 0
    assert captured[0][1] == str(choose.ROOT / entry)
    assert captured[0][2:] == (["launch"] if backend == "vast" else []) + ["--config", "specs/example.json", "--dry-run"]


@pytest.mark.parametrize("selected,entry,prefix", [
    ("Vast.ai", "workflows/vast/run.py", ["launch"]),
    ("Colab", "scripts/colab_oauth/launch.py", []),
    ("Colab (manual auth, legacy)", "scripts/run_on_colab.py", []),
])
def test_native_selector_prefers_new_workflows(monkeypatch, selected, entry, prefix):
    monkeypatch.setattr(choose.sys, "platform", "darwin")
    dialogs, commands = [], []
    def select(script):
        dialogs.append(script)
        return selected
    monkeypatch.setattr(choose, "dialog", select)
    monkeypatch.setattr(choose.subprocess, "call", lambda args, **kwargs: commands.append(args) or 0)
    assert choose.main(["--", "--config", "specs/example.json", "--dry-run"]) == 0
    assert 'default items {"Vast.ai"}' in dialogs[0]
    assert dialogs[0].index('"Vast.ai"') < dialogs[0].index('"Colab"') < dialogs[0].index('"Colab (manual auth, legacy)"')
    assert commands[0][1:] == [str(choose.ROOT / entry), *prefix, "--config", "specs/example.json", "--dry-run"]


@pytest.mark.parametrize("template,label", [*policy.GPU_TEMPLATES.items(), (None, "Saved policy")])
def test_native_vast_gpu_chooser_forwards_template(tmp_path, monkeypatch, template, label):
    monkeypatch.setattr(choose.sys, "platform", "darwin")
    monkeypatch.setattr(choose.Path, "home", lambda: tmp_path)
    saved = tmp_path / ".config/mdlarc-vast/policy.json"
    saved.parent.mkdir(parents=True)
    saved.write_text("{}")
    dialogs, commands = [], []
    answers = iter(["Vast.ai", "specs/example.json", label])
    def select(script):
        dialogs.append(script)
        return next(answers)
    monkeypatch.setattr(choose, "dialog", select)
    monkeypatch.setattr(choose.subprocess, "call", lambda args, **kwargs: commands.append(args) or 0)
    assert choose.main([]) == 0
    assert 'default items {"A100 80 GB"}' in dialogs[-1]
    assert all(f'"{label}"' in dialogs[-1] for label in policy.GPU_TEMPLATES.values())
    expected = ["launch", "--config", "specs/example.json"]
    if template:
        expected += ["--gpu-template", template]
    assert commands[0][2:] == expected


def test_credentials_are_redacted_across_log_chunks():
    redactor = worker.LogRedactor({"KAGGLE_API_TOKEN": "secret-token-abcdef"})
    chunks = [b"hello secret-to", b"ken-abc", b"def done\n"]
    output = b"".join(redactor.feed(data) for data in chunks) + redactor.feed(b"", final=True)
    assert output == b"hello [REDACTED] done\n"


def test_drive_failure_does_not_hide_authenticated_instance_absence(tmp_path, monkeypatch):
    checks = []
    def listing():
        checks.append(1)
        return []
    monkeypatch.setattr(run, "client_for", lambda _: SimpleNamespace(instances=listing))
    monkeypatch.setattr(run, "sync", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Drive disconnected")))
    monkeypatch.setattr(run.time, "sleep", lambda _: None)
    plan = {"instance_id": 7, "deadline_unix": time.time() + 100}
    assert run.watch(SimpleNamespace(poll_seconds=5), tmp_path, plan) == 1
    assert len(checks) == 2
    assert json.loads((tmp_path / "state.json").read_text())["state"] == "unavailable"


def test_stopped_instance_does_not_watch_forever(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "client_for", lambda _: SimpleNamespace(instances=lambda: [{"id": 7, "label": "mine", "actual_status": "stopped"}]))
    monkeypatch.setattr(run, "sync", lambda *args, **kwargs: None)
    monkeypatch.setattr(run, "SSH", lambda *args: pytest.fail("Stopped instance must not require SSH"))
    assert run.watch(SimpleNamespace(poll_seconds=5), tmp_path, {"instance_id": 7, "label": "mine"}) == 1


@pytest.mark.parametrize("remote_state", ["preflighting_runtime", "preparing_inputs", "running"])
def test_handoff_observes_lost_dispatch_response_without_second_execution(tmp_path, monkeypatch, remote_state):
    dispatches = []
    class Connection:
        def __init__(self, *args):
            pass
        def python(self, *args, **kwargs):
            return b"ready"
        def upload(self, *args, **kwargs):
            pass
        def json_file(self, path, **kwargs):
            return None if path.endswith("started.json") else {"state": remote_state}
        def launch(self, *args):
            dispatches.append(1)
            raise RuntimeError("lost SSH response")
    monkeypatch.setattr(run, "SSH", Connection)
    monkeypatch.setattr(run, "public_key", lambda _: "public")
    attachments = []
    plan = {"instance_id": 7, "label": "mine", "deadline_unix": time.time() + 1000, "policy": configured(),
            "ssh_key_path": "/private/key", "remote_workspace": "/workspace/session", "source": {"sha256": "hash", "size": 10}}
    args = SimpleNamespace(drive_token=tmp_path / "outside-token", secrets_file=None)
    run.handoff(args, tmp_path, plan, SimpleNamespace(instance=lambda _: {"id": 7, "label": "mine"},
        attach_ssh_key=lambda identity, key: attachments.append((identity, key))))
    assert attachments == []
    assert dispatches == [1]
    assert json.loads((tmp_path / "state.json").read_text())["state"] == remote_state


def test_handoff_never_attaches_key_to_another_rental(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "public_key", lambda _: pytest.fail("Must verify ownership before reading the key"))
    plan = {"instance_id": 7, "label": "mine", "deadline_unix": time.time() + 1000, "policy": configured()}
    client = SimpleNamespace(instance=lambda _: {"id": 7, "label": "someone-else"},
        attach_ssh_key=lambda *args: pytest.fail("Must not modify another rental"))
    with pytest.raises(ValueError, match="label differs"):
        run.handoff(SimpleNamespace(), tmp_path, plan, client)


def test_missing_helpers_install_only_drive_dependencies(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(worker.importlib.util, "find_spec", lambda _: None)
    monkeypatch.setattr(worker, "run_training", lambda command, *args: commands.append(command))
    worker.ensure_drive_dependencies(tmp_path / "code", tmp_path / "run",
        {"policy": configured(), "branch": "newreason", "run_name": "run", "deadline_unix": time.time() + 4000}, tmp_path / "private")
    assert len(commands) == 1
    assert commands[0][-1] == str(tmp_path / "code/workflows/vast/requirements.txt")
    assert "flash-attn" not in commands[0] and "torch" not in commands[0]


@pytest.mark.parametrize("name", worker.PATH_LIST_ENV_VARS)
def test_compiler_paths_remove_cwd_and_keep_other_entries(tmp_path, name):
    cwd = tmp_path / "code"
    entries = ["", ".", "./", "child/..", str(cwd), "/usr/local/cuda/lib64", "../lib", "/usr/lib", ""]
    env = {name: ":".join(entries), "OTHER_SETTING": "unchanged"}
    assert worker.sanitize_path_environment(env, cwd) == {name: 6}
    assert env == {name: "/usr/local/cuda/lib64:../lib:/usr/lib", "OTHER_SETTING": "unchanged"}
    assert worker.sanitize_path_environment(env, cwd) == {}


@pytest.mark.parametrize("value", ["", ":", ".", "./:."])
def test_compiler_path_with_no_safe_entries_is_unset(tmp_path, value):
    env = {"LIBRARY_PATH": value}
    assert worker.sanitize_path_environment(env, tmp_path)["LIBRARY_PATH"] > 0
    assert "LIBRARY_PATH" not in env


def test_training_and_helper_environment_preserves_paths_and_dataset_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("LIBRARY_PATH", ":./:/usr/local/cuda/lib64:")
    monkeypatch.setenv("CPATH", ".")
    monkeypatch.setenv("CONTAINER_API_KEY", "control-secret")
    monkeypatch.setenv("VAST_API_KEY", "account-secret")
    private = tmp_path / "private"
    private.mkdir()
    (private / "training-secrets.json").write_text('{"KAGGLE_API_TOKEN":"dataset-secret"}')
    original = dict(worker.os.environ)
    removed = {}
    env = worker.clean_training_env({"branch": "newreason", "run_name": "smoke"}, private, removed_paths=removed)
    assert env["LIBRARY_PATH"] == "/usr/local/cuda/lib64"
    assert "CPATH" not in env and "CONTAINER_API_KEY" not in env and "VAST_API_KEY" not in env
    assert env["KAGGLE_API_TOKEN"] == "dataset-secret"
    assert env["MDLARC_RUN_BRANCH"] == "newreason" and env["PYTHONUNBUFFERED"] == "1"
    assert removed["LIBRARY_PATH"] == 3 and removed["CPATH"] == 1
    assert dict(worker.os.environ) == original
    commands = []
    monkeypatch.setattr(worker.importlib.util, "find_spec", lambda _: None)
    monkeypatch.setattr(worker, "run_training", lambda command, cwd, env, *args: commands.append(env))
    worker.ensure_drive_dependencies(tmp_path / "code", tmp_path / "run",
        {"policy": configured(), "branch": "newreason", "run_name": "smoke", "deadline_unix": time.time() + 4000}, private)
    assert commands[0]["LIBRARY_PATH"] == "/usr/local/cuda/lib64"
    assert "CPATH" not in commands[0]


@pytest.mark.parametrize("failure", [None, "cuda", "bf16", "cudagraph", "triton", "compile", "wrong_result"])
def test_cuda_preflight_executes_driver_compile_and_checks_result(tmp_path, monkeypatch, failure):
    calls = []
    class Tensor:
        def sin(self):
            return 3
    def compile_model(function, **kwargs):
        calls.append("compile")
        assert kwargs == {"fullgraph": True}
        def execute(tensor):
            calls.append("execute")
            if failure == "compile":
                raise RuntimeError("compiler failed")
            return 5 if failure == "wrong_result" else function(tensor)
        return execute
    def arange(size, **kwargs):
        assert size == 8 and kwargs == {"device": "cuda", "dtype": "float32"}
        return Tensor()
    def driver_target():
        calls.append("triton")
        if failure == "triton":
            raise RuntimeError("gcc failed")
        return "cuda-target"
    def check_result(actual, expected):
        calls.append("verify")
        assert actual == expected
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: failure != "cuda", is_bf16_supported=lambda: failure != "bf16",
        synchronize=lambda: calls.append("synchronize"), get_device_name=lambda _: "test GPU")
    torch.compiler = SimpleNamespace() if failure == "cudagraph" else SimpleNamespace(cudagraph_mark_step_begin=lambda: None)
    torch.compile, torch.arange, torch.float32 = compile_model, arange, "float32"
    torch.testing = SimpleNamespace(assert_close=check_result)
    torch.__version__, torch.version = "test", SimpleNamespace(cuda="test-cuda")
    triton, runtime = ModuleType("triton"), ModuleType("triton.runtime")
    runtime.driver = SimpleNamespace(active=SimpleNamespace(get_current_target=driver_target))
    triton.runtime = runtime
    for name, module in (("torch", torch), ("triton", triton), ("triton.runtime", runtime)):
        monkeypatch.setitem(sys.modules, name, module)
    path = tmp_path / "runtime_environment.json"
    common.atomic_json(path, {"torch_compile_smoke": False, "sanitized_current_directory_path_entries": {"LIBRARY_PATH": 1}})
    monkeypatch.setattr(sys, "argv", ["preflight", str(path)])
    if failure:
        with pytest.raises((RuntimeError, AssertionError)):
            exec(worker.CUDA_PREFLIGHT_CODE, {})
        assert not json.loads(path.read_text())["torch_compile_smoke"]
    else:
        exec(worker.CUDA_PREFLIGHT_CODE, {})
        report = json.loads(path.read_text())
        assert calls == ["triton", "compile", "execute", "synchronize", "verify"]
        assert report["torch_compile_smoke"] and report["triton_target"] == "cuda-target"
        assert report["sanitized_current_directory_path_entries"] == {"LIBRARY_PATH": 1}


def test_runtime_preflight_uses_sanitized_child_env_and_private_path_counts(tmp_path, monkeypatch):
    code = tmp_path / "code"
    code.mkdir()
    run_dir = code / "runs/smoke"
    run_dir.mkdir(parents=True)
    monkeypatch.setenv("LIBRARY_PATH", ":.:/private/compiler/location")
    monkeypatch.setattr(worker, "CUDA_PREFLIGHT_CODE", """import json, os, pathlib, sys
assert os.environ['LIBRARY_PATH'] == '/private/compiler/location'
path = pathlib.Path(sys.argv[1])
report = json.loads(path.read_text())
report['torch_compile_smoke'] = True
path.write_text(json.dumps(report))
print('compile succeeded', flush=True)
""")
    worker.preflight_runtime(code, run_dir,
        {"branch": "newreason", "run_name": "smoke", "policy": configured(), "deadline_unix": time.time() + 4000}, tmp_path / "private")
    report = json.loads((run_dir / "runtime_environment.json").read_text())
    assert report["torch_compile_smoke"]
    assert report["sanitized_current_directory_path_entries"]["LIBRARY_PATH"] == 2
    assert "/private/compiler/location" not in json.dumps(report)
    assert "compile succeeded" in (run_dir / "runtime_preflight.log").read_text()


def test_runtime_preflight_kills_child_when_time_allowance_expires(tmp_path, monkeypatch):
    run_dir = tmp_path / "runs/smoke"
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(worker, "CUDA_PREFLIGHT_CODE", "import time; time.sleep(60)")
    ticks = iter([0, 0, 301])
    monkeypatch.setattr(worker.time, "time", lambda: next(ticks, 301))
    original_popen, processes = subprocess.Popen, []
    def spawn(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(worker.subprocess, "Popen", spawn)
    with pytest.raises(RuntimeError, match="time allowance"):
        worker.preflight_runtime(tmp_path, run_dir,
            {"branch": "newreason", "run_name": "smoke", "policy": configured(), "deadline_unix": 4000}, tmp_path / "private")
    assert len(processes) == 1 and processes[0].poll() is not None


def test_runtime_preflight_reserves_backup_time_before_starting_child(tmp_path, monkeypatch):
    run_dir = tmp_path / "runs/smoke"
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(worker, "run_training", lambda *args: pytest.fail("Expired preflight must not start a child"))
    with pytest.raises(RuntimeError, match="time allowance"):
        worker.preflight_runtime(tmp_path, run_dir,
            {"branch": "newreason", "run_name": "smoke", "policy": configured(), "deadline_unix": time.time() + 100}, tmp_path / "private")


def test_source_snapshot_omits_wheels_secrets_old_runs_and_other_specs(tmp_path):
    files = ["launch_run.py", "requirements.txt", "scripts/run_layout.py", "specs/selected.json", "specs/other.json",
             "src/model.py", "assets/.env", "wheels/flash.whl", "assets/old.pt", "results/other/note.md"]
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"run_name":"selected"}' if name.endswith("selected.json") else "data")
    selected = snapshot.source_files(tmp_path, tmp_path / "specs/selected.json")
    assert {p.relative_to(tmp_path).as_posix() for p in selected} == {
        "launch_run.py", "requirements.txt", "scripts/run_layout.py", "specs/selected.json", "src/model.py"}


def test_drive_download_verifies_hash_before_publishing(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    folder = store.create_run("vast-one")
    path = tmp_path / "input.pt"
    path.write_bytes(b"original checkpoint")
    identity = store.upload(path, folder, "input.pt")
    monkeypatch.setattr(store.api, "get_media", lambda **kwargs: store.api.bytes[kwargs["fileId"]], raising=False)
    class Downloader:
        def __init__(self, out, value, **kwargs):
            self.out, self.value = out, value
        def next_chunk(self, **kwargs):
            self.out.write(self.value)
            return None, True
    monkeypatch.setattr("googleapiclient.http.MediaIoBaseDownload", Downloader)
    destination = tmp_path / "download.pt"
    store.download(identity["id"], destination, expected=identity, ancestor=store.branch_id)
    assert destination.read_bytes() == b"original checkpoint"
    store.api.bytes[identity["id"]] = b"corrupt checkpoint!"
    with pytest.raises(ValueError):
        store.download(identity["id"], tmp_path / "bad.pt", expected=identity, ancestor=store.branch_id)
    assert not (tmp_path / "bad.pt").exists()
    assert not (tmp_path / "bad.pt.partial").exists()


@pytest.mark.parametrize("failure_phase", [None, "training", "preflight"])
@pytest.mark.parametrize("gpu_name", ["A100", "B300"])
@pytest.mark.parametrize("checkpoint_mode", ["automatic", "manual"])
def test_offline_supervisor_runs_once_and_backs_up_before_cleanup(tmp_path, monkeypatch, failure_phase, gpu_name, checkpoint_mode):
    training_fails = failure_phase is not None
    workspace = tmp_path / "workspace"
    code = workspace / "code"
    code.mkdir(parents=True)
    (workspace / "private/queue").mkdir(parents=True)
    spec = code / "spec.json"
    spec.write_text('{"run_name":"tiny"}')
    (workspace / "source.tar.gz").write_bytes(b"exact source bytes")
    (code / "launch_run.py").write_text('''import json,os,pathlib
root=pathlib.Path(os.environ['MDLARC_RUN_DIR']); root.mkdir(parents=True,exist_ok=True)
p=root/'epoch.pt'; p.write_bytes(b'model-checkpoint')
if 'MDLARC_CHECKPOINT_QUEUE' in os.environ:
 q=pathlib.Path(os.environ['MDLARC_CHECKPOINT_QUEUE']); q.mkdir(parents=True,exist_ok=True)
 (q/'one.json').write_text(json.dumps({'path':'epoch.pt','size':p.stat().st_size}))
(root/'metrics.json').write_text('{"accuracy":0.5}')
print('training complete',flush=True)
''')
    if failure_phase == "training":
        with (code / "launch_run.py").open("a") as out:
            out.write("import sys; sys.exit(1)\n")
    selected_policy = configured()
    selected_policy["checkpoint_upload_mode"] = checkpoint_mode
    automatic = checkpoint_mode == "automatic"
    plan = {"run_name": "tiny", "session": "vast-test", "branch": "newreason", "instance_id": 7, "gpu_name": gpu_name,
            "drive_folder_id": "folder", "policy": selected_policy, "source": common.hashes(workspace / "source.tar.gz"),
            "spec_relative": "spec.json", "spec_sha256": common.hashes(spec)["sha256"],
            "rented_at_unix": time.time(), "deadline_unix": time.time() + 4000}
    common.atomic_json(workspace / "launch.json", plan)
    uploads, destroyed, stopped = [], [], []
    store = SimpleNamespace(upload=lambda path, folder, name: uploads.append((name, Path(path).read_bytes())) or
        {"id": name, "name": name, **common.hashes(path)})
    monkeypatch.setattr(worker, "store_for", lambda *args: store)
    phases = []
    def preflight(code, run_dir, plan, private):
        phases.append("preflight")
        common.atomic_json(run_dir / "runtime_environment.json", {"torch_compile_smoke": failure_phase != "preflight"})
        (run_dir / "runtime_preflight.log").write_text("gcc failed" if failure_phase == "preflight" else "compile succeeded")
        if failure_phase == "preflight":
            raise RuntimeError("gcc failed")
    def inputs(*args):
        phases.append("inputs")
        return {}
    monkeypatch.setattr(worker, "preflight_runtime", preflight)
    monkeypatch.setattr(worker, "prepare_inputs", inputs)
    monkeypatch.setattr(worker, "instance_client", lambda _: SimpleNamespace(destroy=destroyed.append, stop=stopped.append))
    original_popen = subprocess.Popen
    class ThreadProcess:
        def __init__(self):
            self.returncode = None
            def task():
                self.returncode = worker.uploader(workspace, plan)
            self.thread = threading.Thread(target=task, daemon=True)
            self.thread.start()
        def poll(self):
            return self.returncode
        def wait(self, timeout):
            self.thread.join(timeout)
            if self.thread.is_alive():
                raise subprocess.TimeoutExpired("uploader", timeout)
            return self.returncode
    def spawn(args, **kwargs):
        if "--deadline-only" in args:
            return SimpleNamespace()
        if "--uploader" in args:
            return ThreadProcess()
        return original_popen(args, **kwargs)
    monkeypatch.setattr(worker.subprocess, "Popen", spawn)
    assert worker.supervise(workspace, plan) == (1 if training_fails else 0)
    assert destroyed == ([] if training_fails or gpu_name == "B300" or not automatic else [7])
    assert stopped == ([7] if training_fails and gpu_name != "B300" and automatic else [])
    manifest = json.loads((workspace / "completion.json").read_text())
    assert manifest["ok"] is (not training_fails)
    assert manifest["backups_complete"] is (automatic or failure_phase == "preflight")
    assert manifest["record_backups_complete"] and manifest["finalized"]
    assert manifest["checkpoint_upload_mode"] == checkpoint_mode
    names = [name for name, _ in uploads]
    if failure_phase == "preflight":
        assert phases == ["preflight"] and "epoch.pt" not in names
        assert not (code / "runs/tiny/metrics.json").exists()
    else:
        assert phases == ["preflight", "inputs"]
        if automatic:
            assert names.index("epoch.pt") < names.index("completion.json")
        else:
            assert "epoch.pt" not in names
            assert (code / "runs/tiny/epoch.pt").is_file()
    archive_bytes = next(data for name, data in uploads if name.startswith("results-without-checkpoints-"))
    with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as archive:
        assert ("metrics.json" in archive.getnames()) is (failure_phase != "preflight")
        assert "runtime_environment.json" in archive.getnames() and "runtime_preflight.log" in archive.getnames()
        assert not any(name.endswith(".pt") for name in archive.getnames())
    before = len(uploads)
    assert worker.supervise(workspace, plan) == 0
    assert len(uploads) == before
    assert destroyed == ([] if training_fails or gpu_name == "B300" or not automatic else [7])
