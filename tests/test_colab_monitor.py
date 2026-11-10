import io
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import colab_monitor as monitor
from scripts import colab_monitor_backend as backend
from scripts.colab_monitor_data import measure_speed, parse_console


def console(epoch=3, step=20):
    return ("Starting training stage 'pretrain' in an isolated process...\n"
            f"Epoch {epoch - 1}/330\n"
            f"epoch={epoch - 1} step=85 loss=0.3 total_loss=0.6\n"
            f"epoch={epoch - 1} loss=0.2 mtp_loss=0.3 total_loss=0.29 color_accuracy=90.12% exact_output_accuracy=2.34%\n"
            f"Epoch {epoch}/330\n"
            f"epoch={epoch} step={step} loss=0.1 mtp_loss=0.2 total_loss=0.16\n")


def test_step_and_completed_epoch_metrics_remain_distinct_and_duplicates_do_not_count():
    value = parse_console(console() + console().splitlines()[-1] + "\n")
    assert (value["epoch"], value["step"], value["steps_per_epoch"]) == (3, 20, 85)
    assert value["step_metrics"]["loss"] == 0.1
    assert value["epoch_metrics"]["epoch"] == 2
    assert value["epoch_metrics"]["loss"] == 0.2
    assert value["epoch_metrics"]["color_accuracy"] == 90.12
    assert "color_accuracy" not in value["step_metrics"]


def test_stage_transition_clears_old_accuracies_and_invalidates_speed():
    first = parse_console(console())
    last = parse_console(console() + "Starting training stage 'finetune:heldout' in an isolated process...\nEpoch 1/90\nepoch=1 step=10 loss=0.4\n")
    assert last["stage"] == "finetune:heldout"
    assert last["epoch_metrics"] == {}
    assert "per_second" not in measure_speed(first, last, 30)


def test_speed_uses_fractional_epochs_and_rejects_changed_epoch_lengths():
    first, last = parse_console(console(3, 20)), parse_console(console(4, 20))
    speed = measure_speed(first, last, 30)
    assert speed["per_second"] == pytest.approx(1 / 30)
    assert speed["seconds_per_unit"] == 30
    last["steps_per_epoch"] = 90
    assert "changed" in measure_speed(first, last, 30)["note"]
    assert "completed epoch" in measure_speed(parse_console("Epoch 1/330\nepoch=1 step=10 loss=2\n"), first, 30)["note"]


def test_evaluation_speed_uses_batches_and_never_training_accuracy():
    first = parse_console(console() + "Starting evaluation 'final'...\n[test] Finished batch 4 / 10\n")
    last = parse_console(console() + "Starting evaluation 'final'...\n[test] Finished batch 7 / 10\n")
    assert first["epoch_metrics"] == {}
    assert measure_speed(first, last, 30)["per_second"] == 0.1
    assert measure_speed(first, last, 30)["unit"] == "batches"
    assert parse_console("[test] Accuracy batch 2 / 4\n")["evaluation"]["batch"] == 2


def test_no_progress_is_zero_observed_progress_not_a_failure():
    first = parse_console(console())
    assert measure_speed(first, first, 30)["per_second"] == 0
    assert measure_speed(first, first, 30)["seconds_per_unit"] is None


def test_nonfinite_training_values_are_not_emitted_as_json_numbers():
    value = parse_console("epoch=2 step=10 loss=1e999 total_loss=0.2\n")
    assert "loss" not in value["step_metrics"]
    json.dumps(value, allow_nan=False)


def test_new_epoch_does_not_relabel_previous_step_loss_as_current():
    value = parse_console(console(3, 85) + "epoch=3 loss=0.1 color_accuracy=92%\nEpoch 4/330\n")
    assert value["epoch"] == 4 and value["step"] == 0
    assert value["step_metrics"] == {}
    assert value["epoch_metrics"]["epoch"] == 3


def test_error_is_flagged_without_exporting_credential_bearing_exception_text():
    value = parse_console(console() + "torch.OutOfMemoryError: token=private\n")
    assert value["error_observed"] is True
    assert "private" not in json.dumps(value)


class FakeWorker:
    def __init__(self, account, failure=False):
        self.account, self.failure, self.calls = account, failure, []

    def call(self, request):
        self.calls.append(request)
        if self.failure:
            raise monitor.MonitorError("Offline; live state is unknown.")
        if request["action"] == "list":
            return [{"endpoint": "vm", "session": "same-name", "gpu": "A100", "shape": "HIGH_RAM"}]
        return {"observed_at": "2026-10-16T15:30:00+00:00", "run_name": "same-name",
                "progress": parse_console(console(3 if request["action"] != "progress" else 4)),
                "gpus": [], "scores": []}

    def close(self):
        pass


def app(tmp_path, failure=False):
    return monitor.Monitor(tmp_path, {a: FakeWorker(a, failure and a == "2") for a in ("1", "2")})


def test_account_failures_are_unknown_and_other_account_still_works(tmp_path):
    value = app(tmp_path, failure=True).listing()
    assert len(value["accounts"][0]["runs"]) == 1
    assert value["accounts"][1]["error"].startswith("Offline")


def test_listing_is_cached_without_background_calls_and_account_keys_are_isolated(tmp_path):
    service = app(tmp_path)
    service.listing()
    service.listing()
    assert all(len(w.calls) == 1 for w in service.workers.values())
    service.check("1", "vm", "metrics")
    assert service.workers["1"].calls[-1]["action"] == "metrics"
    assert len(service.workers["2"].calls) == 1
    with pytest.raises(monitor.MonitorError):
        service.check("1", "unlisted-vm", "metrics")
    with pytest.raises(monitor.MonitorError):
        service.check("1", "vm", "execute")


def test_speed_sampling_never_sleeps_or_polls_in_server_and_only_finishes_on_request(tmp_path):
    service = app(tmp_path)
    service.listing()
    first = service.check("1", "vm", "performance")
    assert len(service.workers["1"].calls) == 2
    with pytest.raises(monitor.MonitorError, match="still collecting"):
        service.finish(first["sample_id"])
    service.samples[first["sample_id"]]["time"] -= 30
    last = service.finish(first["sample_id"])
    assert last["speed"]["per_second"] == pytest.approx(1 / 30, rel=0.01)
    assert service.workers["1"].calls[-1]["action"] == "progress"
    assert first["sample_id"] not in service.samples


def test_exited_workers_are_reaped_without_another_request_and_restart_cleanly(tmp_path, monkeypatch):
    script = tmp_path / "stub_worker.py"
    script.write_text('import sys\nfor line in sys.stdin:\n print(\'{"ok":true,"data":[]}\',flush=True)\n break\n')
    monkeypatch.setattr(monitor, "WORKER", script)
    worker = monitor.Worker(sys.executable, "1")
    try:
        assert worker.call({"action": "list"}) == []
        first = worker.process
        deadline = time.monotonic() + 2
        while first.returncode is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert first.returncode == 0
        assert worker.call({"action": "list"}) == []
        assert worker.process.pid != first.pid
    finally:
        worker.close()


def test_local_journals_do_not_mix_accounts_or_read_identity_as_journal(tmp_path):
    directory = tmp_path / "results/encoded-branch/colab_runs/_active"
    directory.mkdir(parents=True)
    for account, session in (("1", "run.with.dots"), ("2", "other")):
        (directory / f"{session}.json").write_text(json.dumps({"session": session, "run_name": session, "branch": "branch", "colab_account": account}))
        (directory / f"{session}.connection.json").write_text(json.dumps({"session": session, "run_name": session, "endpoint": "vm", "colab_account": account}))
    records = monitor.local_records(tmp_path)
    assert records[("1", "vm")]["session"] == "run.with.dots"
    assert records[("2", "vm")]["session"] == "other"
    assert len(records) == 4


def test_reused_session_alias_cannot_attach_another_vm_s_saved_run(tmp_path):
    directory = tmp_path / "results/branch/colab_runs/_active"
    directory.mkdir(parents=True)
    (directory / "same-name.json").write_text(json.dumps({"session": "same-name", "run_name": "old-run", "branch": "old-branch"}))
    (directory / "same-name.connection.json").write_text(json.dumps({"endpoint": "old-vm"}))
    service = app(tmp_path)
    listing = service.listing()
    assert listing["accounts"][0]["runs"][0]["branch"] is None
    assert service.hints[("1", "vm")] == {}


def test_comparison_selects_matching_epoch_and_keeps_cross_branch_caveats(tmp_path):
    service = app(tmp_path)
    service.listing()
    root = tmp_path / "results/branch"
    directory = root / "status_checks/same-name"
    directory.mkdir(parents=True)
    (directory / "comparison_references.json").write_text(json.dumps({
        "comparison_caveats": ["Historical single-pair context; different learning rates."],
        "references": [{"label": "R1 1k/1k", "session": "historical", "score": {"score": 210, "max_score": 400, "percentage": 52.5, "fully_solved_tasks": ["omit"]},
                        "epoch_metrics": {"2": {"loss": 0.4, "raw_line": "omit"}, "3": {"loss": 0.2}}}]}))
    service.hints[("1", "vm")] = {"cache_dir": str(root / "colab_runs/same-name"), "session": "same-name"}
    result = service.check("1", "vm", "metrics")["comparisons"]
    assert result["epoch"] == 2
    assert result["rows"][0]["metrics"] == {"loss": 0.4}
    assert "fully_solved_tasks" not in result["rows"][0]["score"]
    assert result["caveats"]


class Response:
    status_code = 200

    def __init__(self, chunks):
        self.chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def raise_for_status(self):
        pass

    def iter_content(self, _):
        return iter(self.chunks)


def reader(response):
    account = backend.Account.__new__(backend.Account)
    proxy = SimpleNamespace(url="https://runtime.invalid", token="never-export-this")
    account.assignments = {"vm": SimpleNamespace(runtime_proxy_info=proxy)}
    account.listed_at = time.monotonic()
    account.http = SimpleNamespace(get=lambda *args, **kwargs: response)
    return account


def test_resources_accept_xssi_prefix_and_never_export_runtime_tokens():
    value = reader(Response([b")]}'\n", b'{"gpus":[]}'])).request("vm", "/api/colab/resources")
    assert value == {"gpus": []}
    assert "never-export-this" not in json.dumps(value)


def test_saved_uploaded_spec_fast_path_requires_matching_original_vm_identity():
    account = backend.Account.__new__(backend.Account)
    account.resolved = {}
    def unexpected(*args, **kwargs):
        raise AssertionError("No remote metadata reads for a matching saved VM identity")
    account.contents = unexpected
    hint = {"endpoint": "vm", "run_name": "trial", "branch": "b", "spec": {"pretrain": {"epochs": 330}}}
    assert account.resolve_run("vm", hint)["spec"]["pretrain"]["epochs"] == 330
    with pytest.raises(AssertionError):
        account.resolve_run("different-vm", hint)


def test_active_pretraining_never_probes_missing_score_or_completion_files():
    account = backend.Account.__new__(backend.Account)
    account.resolve_run = lambda *_: {"run_name": "trial", "branch": "b", "run_count": 1,
                                     "spec": {"enable_finetune": False, "evals": {"final": {}}}}
    account.progress = lambda *_: parse_console(console())
    def unexpected(*_):
        raise AssertionError("Pending artifacts should not be requested during pretraining")
    account.file_json = unexpected
    result = account.check({"endpoint": "vm", "action": "metrics"})
    assert result["scores"] == []
    assert result["evaluation_configured"] is True


def test_completed_training_probes_real_scored_results_and_matching_manifest():
    account = backend.Account.__new__(backend.Account)
    account.resolve_run = lambda *_: {"run_name": "trial", "branch": "b", "run_count": 1,
                                     "spec": {"evals": {"final": {}}}}
    account.progress = lambda *_: parse_console(console(330, 85) + "epoch=330 loss=0.01 color_accuracy=99%\n")
    def file_json(_, path):
        if path.endswith("/score.json"):
            return {"score": 200, "max_score": 400, "percentage": 50, "predictions": ["omit"]}
        if path.endswith("results-manifest.json"):
            return {"run_name": "trial", "finalized": True}
        return {"ok": True, "backups_complete": True}
    account.file_json = file_json
    result = account.check({"endpoint": "vm", "action": "metrics"})
    assert result["scores"] == [{"eval": "final", "score": 200, "max_score": 400, "percentage": 50}]
    assert result["completion"] == {"finalized": True, "ok": True, "backups_complete": True}


def test_ignored_range_streams_only_bounded_tail_and_transfer_limit_is_enforced(monkeypatch):
    monkeypatch.setattr(backend, "TAIL", 8)
    monkeypatch.setattr(backend, "MAX_FILE", 20)
    assert reader(Response([b"abcdefgh", b"ijklmnop", b"qrst"])).request("vm", "/files/log", tail=True) == "mnopqrst"
    with pytest.raises(backend.MonitorError, match="4 MiB"):
        reader(Response([b"x" * 21])).request("vm", "/files/log", tail=True)


def test_worker_errors_are_sanitized_before_crossing_ipc(monkeypatch, capsys):
    monkeypatch.setattr(backend.sys, "stdin", io.StringIO('{"action":"list"}\n'))
    checks = iter([([True], [], []), ([], [], [])])
    monkeypatch.setattr(backend.select, "select", lambda *_: next(checks))
    def fail(_):
        raise RuntimeError("https://runtime?token=secret")
    monkeypatch.setattr(backend, "Account", fail)
    backend.main("1")
    output = capsys.readouterr().out
    assert "secret" not in output
    assert "RuntimeError" in json.loads(output)["error"]


@pytest.fixture
def http_server(tmp_path):
    server = monitor.ThreadingHTTPServer(("127.0.0.1", 0), monitor.Handler)
    server.monitor = app(tmp_path)
    server.token = "test-capability"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def test_http_only_serves_assets_and_requires_same_origin_capability(http_server):
    server, url = http_server
    with urlopen(url) as response:
        html = response.read().decode()
        assert "test-capability" in html
        assert "{{TOKEN}}" not in html
        assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    for path in ("/../AGENTS.md", "/api/runs"):
        with pytest.raises(HTTPError) as exc:
            urlopen(url + path)
        assert exc.value.code == 404
    for headers in ({}, {"X-Monitor-Token": server.token, "Origin": "https://evil.invalid"}, {"X-Monitor-Token": server.token, "Host": "evil.invalid"}):
        request = Request(url + "/api/runs", data=b"{}", headers=headers)
        with pytest.raises(HTTPError) as exc:
            urlopen(request)
        assert exc.value.code == 403
    request = Request(url + "/api/runs", data=b"{}", headers={"X-Monitor-Token": server.token})
    with urlopen(request) as response:
        assert len(json.load(response)["accounts"]) == 2
