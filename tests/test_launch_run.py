from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import launch_run  # noqa: E402


@pytest.mark.parametrize("checkpoint", [None, "checkpoints/local.pt", "/tmp/local.pt"])
def test_checkpoint_paths_and_fresh_plans_keep_existing_behavior(checkpoint):
    plan = launch_run.build_plan({"enable_finetune": False, "checkpoint": checkpoint}, Path("spec.json"))
    assert plan["pretrain"]["checkpoint_path"] == (Path(checkpoint) if checkpoint else None)


def test_every_run_keeps_fiftieth_and_rolling_tenth_checkpoints_by_default():
    # Base specs set pretrain.checkpoint_epochs to []; the cadence still applies.
    plan = launch_run.build_plan({"enable_finetune": False, "pretrain": {"epochs": 330, "checkpoint_epochs": []},
                                  "evals": {"mid": {"checkpoint": "pretrain.epoch.300", "mode": "accuracy-only"}}},
                                 Path("spec.json"))
    pretrain = plan["pretrain"]
    assert (pretrain["kept_checkpoint_every"], pretrain["rolling_checkpoint_every"]) == (50, 10)
    assert [epoch for epoch in range(1, 331) if launch_run.checkpoint_is_saved(pretrain, epoch)] == [
        50, 100, 150, 200, 250, 300, 330]
    assert launch_run.resolve_checkpoint_ref("pretrain.epoch.300", plan).name == "tiny.epoch300.pt"
    summary = launch_run.stage_summary(pretrain)
    assert (summary["kept_checkpoint_every"], summary["rolling_checkpoint_every"]) == (50, 10)
    disabled = launch_run.build_plan({"enable_finetune": False, "defaults": {
        "kept_checkpoint_every": 0, "rolling_checkpoint_every": 0}}, Path("spec.json"))["pretrain"]
    assert not launch_run.checkpoint_is_saved(disabled, 50)


@pytest.mark.parametrize("key", ["kept_checkpoint_every", "rolling_checkpoint_every"])
@pytest.mark.parametrize("value", [-10, True, 2.5, "50"])
def test_invalid_checkpoint_cadence_is_rejected(key, value):
    with pytest.raises(ValueError, match=key):
        launch_run.build_plan({"enable_finetune": False, "pretrain": {key: value}}, Path("spec.json"))


def test_execute_plan_isolates_training_and_eval_stages(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    plan = {
        "spec_path": Path("/tmp/experiment.json"),
        "run_name": "experiment",
        "skip_data_build": True,
        "eval_only": False,
        "pretrain": {"name": "pretrain"},
        "finetunes": {"arc": {"name": "finetune:arc"}},
        "evals": {"final": {}},
    }

    monkeypatch.setattr(
        launch_run,
        "load_runtime_modules",
        lambda: (_ for _ in ()).throw(
            AssertionError("orchestrator loaded CUDA runtime")
        ),
    )
    monkeypatch.setattr(
        launch_run,
        "run_training_stage_isolated",
        lambda _plan, name: calls.append(("train", name)),
    )
    monkeypatch.setattr(
        launch_run,
        "run_eval_stage_isolated",
        lambda _plan, name: calls.append(("eval", name)),
    )

    launch_run.execute_plan(plan)

    assert calls == [
        ("train", "pretrain"),
        ("train", "finetune:arc"),
        ("eval", "final"),
    ]


def test_isolated_training_command_targets_hidden_stage(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, *, check, cwd, env=None):
        captured.update(command=command, check=check, cwd=cwd, env=env)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launch_run.subprocess, "run", fake_run)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    spec_path = Path("/tmp/experiment.json")

    launch_run.run_training_stage_isolated(
        {"spec_path": spec_path},
        "finetune:arc",
    )
    assert captured["env"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
    launch_run.run_training_stage_isolated({"spec_path": spec_path}, "finetune:arc")
    assert captured["env"]["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:64"

    assert captured["command"] == [
        sys.executable,
        "-u",
        str(Path(launch_run.__file__).resolve()),
        "--config",
        str(spec_path),
        "--train-stage",
        "finetune:arc",
    ]
    assert captured["check"] is True
    assert captured["cwd"] == launch_run.REPO_ROOT


def test_isolated_eval_command_targets_hidden_stage(monkeypatch) -> None:
    captured = []
    monkeypatch.setattr(
        launch_run.subprocess, "run", lambda command, **kwargs: captured.append((command, kwargs))
    )
    launch_run.run_eval_stage_isolated({"spec_path": Path("/tmp/experiment.json")}, "final")
    assert captured == [([
        sys.executable, "-u", str(Path(launch_run.__file__).resolve()),
        "--config", "/tmp/experiment.json", "--eval-stage", "final",
    ], {"check": True, "cwd": launch_run.REPO_ROOT})]


def test_eval_only_launches_no_training(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(launch_run, "run_training_stage_isolated", lambda *args: pytest.fail("training started"))
    monkeypatch.setattr(launch_run, "run_eval_stage_isolated", lambda plan, name: calls.append(name))
    launch_run.execute_plan({
        "spec_path": Path("/tmp/eval.json"), "run_name": "eval",
        "skip_data_build": True, "eval_only": True, "evals": {"final": {}},
    })
    assert calls == ["final"]


def test_failed_child_stops_later_stages(monkeypatch) -> None:
    calls = []

    def fail(command, **kwargs):
        calls.append(command[-1])
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(launch_run.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        launch_run.execute_plan({
            "spec_path": Path("/tmp/experiment.json"), "run_name": "experiment",
            "skip_data_build": True, "eval_only": False,
            "finetunes": {"arc": {}}, "evals": {"final": {}},
        })
    assert calls == ["pretrain"]


@pytest.mark.parametrize("flag,name,expected", [
    ("--train-stage", "pretrain", "pretrain"),
    ("--train-stage", "finetune:arc", "finetune:arc"),
    ("--eval-stage", "final", "final"),
])
def test_child_dispatches_only_requested_stage(monkeypatch, flag, name, expected) -> None:
    calls = []
    plan = {"eval_only": False, "pretrain": {}, "finetunes": {"arc": {}}, "evals": {"final": {}}}
    monkeypatch.setattr(sys, "argv", ["launch_run.py", "--config", "/tmp/spec.json", flag, name])
    monkeypatch.setattr(launch_run, "update_env_from_file", lambda *args: None)
    monkeypatch.setattr(launch_run, "load_spec", lambda path: {})
    monkeypatch.setattr(launch_run, "build_plan", lambda *args: plan)
    monkeypatch.setattr(launch_run, "run_training_stage", lambda stage, label: calls.append(label))
    monkeypatch.setattr(launch_run, "run_eval_stage", lambda plan, name, cfg: calls.append(name))
    monkeypatch.setattr(launch_run, "execute_plan", lambda plan: pytest.fail("child relaunched the plan"))
    launch_run.main()
    assert calls == [expected]


def test_eval_modes_default_inherit_override_and_preview():
    spec = {"enable_finetune": False}
    assert launch_run.build_plan(spec, Path("spec.json"))["evals"]["final"]["mode"] == "full-decode"
    spec.update(
        eval_defaults={"mode": "accuracy-only"},
        evals={
            "fast": {"checkpoint": "pretrain.final"},
            "decode": {"checkpoint": "pretrain.final", "mode": "full-decode"},
        },
    )
    plan = launch_run.build_plan(spec, Path("spec.json"))
    preview = launch_run.preview_plan(plan)["evals"]
    assert preview["fast"]["mode"] == "accuracy-only"
    assert preview["fast"]["accuracy_path"].name == "accuracy.json"
    assert "submission_path" not in preview["fast"]
    assert preview["decode"]["mode"] == "full-decode"
    assert preview["decode"]["submission_path"].name == "submission.json"


@pytest.mark.parametrize("mode", [None, "accuracy", "decode", True, 1, []])
@pytest.mark.parametrize("in_defaults", [False, True])
def test_invalid_eval_mode_rejected(mode, in_defaults):
    spec = {"enable_finetune": False}
    if in_defaults:
        spec["eval_defaults"] = {"mode": mode}
    else:
        spec["evals"] = {"final": {"checkpoint": "pretrain.final", "mode": mode}}
    with pytest.raises(ValueError, match="mode must be"):
        launch_run.build_plan(spec, Path("spec.json"))


@pytest.mark.parametrize("mode", ["full-decode", "accuracy-only"])
def test_eval_stage_only_scores_and_visualizes_decoded_outputs(monkeypatch, mode):
    plan = launch_run.build_plan({
        "enable_finetune": False, "score_results": True, "visualize": True,
        "eval_defaults": {"mode": mode},
    }, Path("spec.json"))
    calls = []

    def evaluate(cfg, **kwargs):
        assert cfg.mode == mode
        assert cfg.solutions_path == plan["assets"]["solutions_path"]
        return ("final", {}, Path("accuracy.json" if mode == "accuracy-only" else "submission.json"))

    monkeypatch.setattr(launch_run, "load_runtime_modules", lambda: None)
    monkeypatch.setattr(launch_run, "evaluate", SimpleNamespace(run_evaluation=evaluate))
    monkeypatch.setattr(launch_run, "utils", SimpleNamespace(
        score_arc_submission=lambda *args: calls.append("score"),
        visualize_submissions=lambda *args, **kwargs: calls.append("visualize"),
    ))
    monkeypatch.setattr(launch_run, "write_score_summary", lambda *args, **kwargs: calls.append("summary"))
    launch_run.run_eval_stage(plan, "final", plan["evals"]["final"])
    assert calls == (["score", "summary", "visualize"] if mode == "full-decode" else [])


def test_local_checkpoint_alias_resolves_against_the_runs_folder(tmp_path):
    run_dir = tmp_path / "runs/current"
    path = launch_run.resolve_input_checkpoint("runs/earlier/pretrain/tiny.pt", run_dir)
    assert path == tmp_path / "runs/earlier/pretrain/tiny.pt"
    explicit = tmp_path / "elsewhere/earlier/tiny.pt"
    assert launch_run.resolve_input_checkpoint(str(explicit), run_dir) == explicit
    with pytest.raises(TypeError, match="local path"):
        launch_run.resolve_input_checkpoint({"id": "saved"}, run_dir)
