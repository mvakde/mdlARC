from pathlib import Path
from types import SimpleNamespace

import pytest

import launch_run
from dataset_building_scripts import build_datasets


def _solved_task():
    return {
        "train": [
            {
                "input": [[1]],
                "output": [[2]],
            }
        ],
        "test": [
            {
                "input": [[3]],
                "output": [[4]],
            },
            {
                "input": [[5]],
                "output": [[6]],
            },
        ],
    }


def test_conceptarc_build_separates_test_inputs_and_solutions(monkeypatch):
    tasks = {"Concept1": _solved_task()}
    monkeypatch.setattr(build_datasets, "_load_conceptarc", lambda: tasks)

    outputs = build_datasets._build_style_outputs(
        SimpleNamespace(dataset="conceptarc")
    )

    assert outputs["challenges.json"] == {
        "Concept1": {
            "train": tasks["Concept1"]["train"],
            "test": [{"input": [[3]]}, {"input": [[5]]}],
        }
    }
    assert outputs["solutions.json"] == {"Concept1": [[[4]], [[6]]]}
    assert tasks["Concept1"]["test"][0]["output"] == [[4]]


def test_conceptarc_build_requires_every_test_solution():
    task = _solved_task()
    del task["test"][1]["output"]

    with pytest.raises(ValueError, match="Missing ConceptARC test output"):
        build_datasets._build_conceptarc_eval_outputs({"Concept1": task})


def test_conceptarc_plan_evaluates_final_checkpoint_by_default():
    spec = {
        "run_name": "conceptarc-test",
        "dataset": {
            "dataset": "conceptarc",
            "keep_train_test_separate": True,
        },
        "pretrain": {"epochs": 3},
    }

    plan = launch_run.build_plan(spec, Path("conceptarc-test.json").resolve())

    assert plan["skip_evals"] is False
    assert plan["assets"]["evaluation_data_path"].name == "challenges.json"
    assert plan["assets"]["solutions_path"].name == "solutions.json"
    assert plan["evals"]["final"]["checkpoint"] == "pretrain.final"


def test_conceptarc_pretrain_eval_spec_runs_90_epochs_and_scores():
    spec_path = Path("specs/conceptarc_pretrain_eval.json").resolve()
    plan = launch_run.build_plan(launch_run.load_spec(spec_path), spec_path)

    assert plan["dataset"]["dataset"] == "conceptarc"
    assert plan["dataset"]["keep_train_test_separate"] is True
    assert plan["pretrain"]["epochs"] == 90
    assert plan["score_results"] is True
    assert plan["evals"]["final"]["checkpoint"] == "pretrain.final"


def test_conceptarc_smoke_spec_remains_train_only():
    spec_path = Path("specs/conceptarc_smoke.json").resolve()
    plan = launch_run.build_plan(launch_run.load_spec(spec_path), spec_path)

    assert plan["pretrain"]["epochs"] == 2
    assert plan["skip_evals"] is True
    assert plan["evals"] == {}
