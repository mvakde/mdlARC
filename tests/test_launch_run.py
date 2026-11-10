from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import launch_run  # noqa: E402


def test_two_demo_mode_reaches_training_and_evaluation_stages():
    plan = launch_run.build_plan({
        "defaults": {"attention_pattern": "pair_group", "training_demo_schedule": "two"},
        "eval_defaults": {"inference_demo_policy": "two"},
        "finetunes": {"heldout": {}},
        "evals": {"final": {"checkpoint": "pretrain.final"}},
    }, Path("two-demo.json"))
    assert plan["pretrain"]["training_demo_schedule"] == "two"
    assert plan["finetunes"]["heldout"]["training_demo_schedule"] == "two"
    assert plan["evals"]["final"]["inference_demo_policy"] == "two"


@pytest.mark.parametrize('cap', [0, 7, True, 2.5])
def test_invalid_training_demo_cap_is_rejected(cap):
    with pytest.raises(ValueError, match='max_training_demos'):
        launch_run.build_plan({'defaults': {'max_training_demos': cap}}, Path('cap.json'))


def test_drive_checkpoint_plan_preserves_reference_and_resolves_input_path():
    source = {"drive_id": "saved-checkpoint", "name": "tiny.epoch12.pt", "size": 123, "sha256": "drive"}
    plan = launch_run.build_plan({
        "run_name": "continuation", "enable_finetune": False,
        "checkpoint": source, "pretrain": {"epochs": 41},
    }, Path("spec.json"))
    assert plan["pretrain"]["checkpoint_path"] == plan["run_dir"] / "inputs/pretrain.pt"
    assert plan["pretrain"]["checkpoint_mode"] == "resume"
    assert plan["pretrain"]["epochs"] == 41
    assert plan["drive_checkpoint"] == source
    assert launch_run.preview_plan(plan)["drive_checkpoint"] == source


@pytest.mark.parametrize("checkpoint", [None, "checkpoints/local.pt", "/tmp/local.pt"])
def test_checkpoint_paths_and_fresh_plans_keep_existing_behavior(checkpoint):
    plan = launch_run.build_plan({"enable_finetune": False, "checkpoint": checkpoint}, Path("spec.json"))
    assert plan["drive_checkpoint"] is None
    assert plan["pretrain"]["checkpoint_path"] == (Path(checkpoint) if checkpoint else None)


def test_finetune_resume_preserves_optimizer_schedule_and_epoch_mode():
    spec = {"run_name": "resume-adapt", "finetune_only": True, "finetune_resume": True,
            "checkpoint": "saved.pt", "dataset": {"style": "finetune"},
            "finetunes": {"validation": {"epochs": 330}}}
    stage = launch_run.build_plan(spec, Path("resume.json"))["finetunes"]["validation"]
    assert stage["checkpoint_mode"] == "resume" and stage["epochs"] == 330
    assert stage["allow_missing_tasks"] is True


@pytest.mark.parametrize("value,finetune_only", [(True, False), ("yes", True)])
def test_finetune_resume_requires_explicit_finetune_only(value, finetune_only):
    with pytest.raises(ValueError, match="finetune_resume"):
        launch_run.build_plan({"finetune_only": finetune_only, "finetune_resume": value}, Path("resume.json"))


def test_finetune_only_uses_saved_weights_and_never_runs_pretraining(monkeypatch):
    spec = {"run_name": "adapt", "finetune_only": True, "checkpoint": "saved.pt",
            "skip_data_build": True, "dataset": {"style": "finetune"},
            "defaults": {"batch_size": 12, "gradient_accumulation_steps": 5},
            "finetunes": {"validation": {"epochs": 90, "max_augments": 80}}}
    plan = launch_run.build_plan(spec, Path("adapt.json"))
    stage = plan["finetunes"]["validation"]
    assert stage["checkpoint_path"] == Path("saved.pt") and stage["checkpoint_mode"] == "finetune"
    assert stage["epochs"] == 90 and stage["max_augments"] == 80
    assert stage["batch_size"] == 12 and stage["gradient_accumulation_steps"] == 5
    assert launch_run.preview_plan(plan)["pretrain"] is None
    calls = []
    monkeypatch.setattr(launch_run, "validate_run_split", lambda _: None)
    def train_stage(p, name):
        assert name != "pretrain"
        calls.append(name)
    monkeypatch.setattr(launch_run, "run_training_stage_isolated", train_stage)
    monkeypatch.setattr(launch_run, "run_eval_stage_isolated", lambda *a: None)
    launch_run.execute_plan(plan)
    assert calls == ["finetune:validation"]


@pytest.mark.parametrize('changes', [{'checkpoint': None}, {'enable_finetune': False}, {'eval_only': True}, {'finetune_only': 'yes'}])
def test_finetune_only_rejects_invalid_or_ambiguous_modes(changes):
    spec = {'finetune_only': True, 'checkpoint': 'saved.pt', 'dataset': {'style': 'finetune'}, **changes}
    with pytest.raises(ValueError, match='finetune_only'):
        launch_run.build_plan(spec, Path('adapt.json'))
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


@pytest.mark.parametrize("field,value", [
    ("drive_id", "https://drive.google.com/file/d/foo"), ("drive_id", 123),
    ("name", "../saved.pt"), ("name", "saved.txt"),
    ("size", True), ("size", 0), ("size", -1), ("size", "123"), ("size", 1.5),
    ("sha256", None), ("sha256", "not-a-hash"), ("unknown", "value"),
])
def test_invalid_drive_checkpoint_metadata_is_rejected(field, value):
    source = {"drive_id": "saved", "name": "saved.pt", "size": 123, "sha256": "drive", field: value}
    with pytest.raises(ValueError, match="checkpoint"):
        launch_run.build_plan({"checkpoint": source}, Path("spec.json"))


def test_drive_checkpoint_requires_complete_identity_and_safe_run_name():
    with pytest.raises(ValueError, match="missing required"):
        launch_run.build_plan({"checkpoint": {"drive_id": "saved"}}, Path("spec.json"))
    with pytest.raises(ValueError, match="run_name"):
        launch_run.build_plan({"run_name": "../escape", "checkpoint": {
            "drive_id": "saved", "name": "saved.pt", "size": 123, "sha256": "a" * 64,
        }}, Path("spec.json"))


def test_drive_spec_direct_execution_fails_before_data_build_when_not_staged(monkeypatch, tmp_path):
    import json
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"run_name": "continuation", "checkpoint": {
        "drive_id": "saved", "name": "saved.pt", "size": 123, "sha256": "drive",
    }}))
    monkeypatch.setattr(launch_run, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["launch_run.py", "--config", str(spec)])
    monkeypatch.setattr(launch_run, "execute_plan", lambda plan: pytest.fail("Training or data build started"))
    with pytest.raises(FileNotFoundError, match="run_on_colab.py"):
        launch_run.main()


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

    def fake_run(command, *, check, cwd):
        captured.update(command=command, check=check, cwd=cwd)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launch_run.subprocess, "run", fake_run)
    spec_path = Path("/tmp/experiment.json")

    launch_run.run_training_stage_isolated(
        {"spec_path": spec_path},
        "finetune:arc",
    )

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
    plan = {"eval_only": False, "drive_checkpoint": None, "pretrain": {}, "finetunes": {"arc": {}}, "evals": {"final": {}}}
    monkeypatch.setattr(sys, "argv", ["launch_run.py", "--config", "/tmp/spec.json", flag, name])
    monkeypatch.setattr(launch_run, "update_env_from_file", lambda *args: None)
    monkeypatch.setattr(launch_run, "load_spec", lambda path: {})
    monkeypatch.setattr(launch_run, "build_plan", lambda *args: plan)
    monkeypatch.setattr(launch_run, "run_training_stage", lambda stage, label: calls.append(label))
    monkeypatch.setattr(launch_run, "run_eval_stage", lambda plan, name, cfg: calls.append(name))
    monkeypatch.setattr(launch_run, "execute_plan", lambda plan: pytest.fail("child relaunched the plan"))
    launch_run.main()
    assert calls == [expected]


def test_eval_child_does_not_require_deleted_training_input(monkeypatch, tmp_path):
    calls = []
    plan = {"eval_only": False, "drive_checkpoint": {"drive_id": "old-input"},
            "pretrain": {"checkpoint_path": tmp_path / "removed-resume-input.pt"},
            "evals": {"final": {"checkpoint": "finetune.arc.final"}}}
    monkeypatch.setattr(sys, "argv", ["launch_run.py", "--config", "/tmp/spec.json", "--eval-stage", "final"])
    monkeypatch.setattr(launch_run, "update_env_from_file", lambda *args: None)
    monkeypatch.setattr(launch_run, "load_spec", lambda path: {})
    monkeypatch.setattr(launch_run, "build_plan", lambda *args: plan)
    monkeypatch.setattr(launch_run, "validate_run_split", lambda plan: None)
    monkeypatch.setattr(launch_run, "run_eval_stage", lambda plan, name, cfg: calls.append(name))
    launch_run.main()
    assert calls == ["final"]


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
def test_eval_stage_only_scores_and_visualizes_decoded_outputs(monkeypatch, mode, tmp_path):
    plan = launch_run.build_plan({
        "enable_finetune": False, "score_results": True, "visualize": True,
        "eval_defaults": {"mode": mode},
    }, Path("spec.json"))
    calls = []

    def evaluate(cfg, **kwargs):
        assert cfg.mode == mode
        assert cfg.solutions_path == plan["assets"]["solutions_path"]
        return ("final", {"test": {"results": [{"output_grid": [[1]], "pair_index": 0}]}},
                tmp_path / ("accuracy.json" if mode == "accuracy-only" else "submission.json"))

    monkeypatch.setattr(launch_run, "load_runtime_modules", lambda: None)
    monkeypatch.setattr(launch_run, "evaluate", SimpleNamespace(run_evaluation=evaluate))
    monkeypatch.setattr(launch_run, "utils", SimpleNamespace(
        score_arc_submission=lambda *args: calls.append("score"),
        visualize_submissions=lambda *args, **kwargs: calls.append("visualize"),
    ))
    monkeypatch.setattr(launch_run, "write_score_summary", lambda *args, **kwargs: calls.append("summary"))
    launch_run.run_eval_stage(plan, "final", plan["evals"]["final"])
    assert calls == (["score", "summary", "visualize"] if mode == "full-decode" else [])
    candidates = tmp_path / "generated_candidates.json.gz"
    if mode == "full-decode":
        import gzip
        import json
        with gzip.open(candidates, "rt") as handle:
            assert json.load(handle)["test"]["results"] == [{"output_grid": [[1]], "pair_index": 0}]
    else:
        assert not candidates.exists()


def test_original_tasks_accept_arc2_eval_parent_split(monkeypatch, tmp_path):
    import json
    manifest = {"schema_version": 1, "name": "development", "source": "arc2_eval",
                "source_sha256": "a" * 64, "train": ["train-task"],
                "eval_groups": {"development_validation": ["heldout-task"]}}
    (tmp_path / "split.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(launch_run, "REPO_ROOT", tmp_path)
    dataset = {"dataset": "arc2", "style": "finetune",
               "evaluation": {"source": "arc2_eval", "split_manifest": "split.json"},
               "pretrain_data": {"kind": "original_tasks", "sources": ["arc2_train"]}}
    plan = launch_run.build_plan({"finetune_only": True, "checkpoint": "saved.pt",
                                 "dataset": dataset,
                                 "finetunes": {"development_validation": {"epochs": 90}},
                                 "evals": {"final": {"checkpoint": "finetune.development_validation.final",
                                                    "group": "development_validation"}}}, tmp_path / "spec.json")
    assert plan["dataset"]["evaluation"] == dataset["evaluation"]
    assert plan["dataset"]["pretrain_data"]["kind"] == "original_tasks"
    assert plan["finetunes"]["development_validation"]["data_path"].name == "priv_challenges.json"
    assert plan["evals"]["final"]["data_path"].name == "eval_development_validation_challenges.json"
    manifest["source"] = "arc2_train"
    (tmp_path / "split.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source must match"):
        launch_run.build_dataset_config(dataset)


def test_local_checkpoint_alias_uses_selected_branch_directory(tmp_path):
    run_dir = tmp_path / "results/newreason/runs/current"
    path, drive = launch_run.resolve_input_checkpoint("runs/earlier/pretrain/tiny.pt", run_dir)
    assert path == tmp_path / "results/newreason/runs/earlier/pretrain/tiny.pt"
    assert drive is None
    explicit = tmp_path / "results/colab-restart/runs/earlier/tiny.pt"
    assert launch_run.resolve_input_checkpoint(str(explicit), run_dir)[0] == explicit
