"""Held-out package integrity, answer isolation, and post-decode scoring."""

import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle"))
from checkpoint_profiles import PROFILES
from make_holdout_dataset import package_holdout
import make_notebook
import submission_utils as ku


@pytest.fixture
def holdout(tmp_path):
    tasks = {
        "a": {"train": [{"input": [[1]], "output": [[2]]}],
              "test": [{"input": [[3]], "output": [[4]]}, {"input": [[5]], "output": [[6]]}]},
        "b": {"train": [{"input": [[1, 2]], "output": [[2, 1]]}],
              "test": [{"input": [[7]], "output": [[8]]}]},
    }
    source = tmp_path / "source.json"
    source.write_text(json.dumps(tasks))
    split = tmp_path / "split.json"
    split.write_text(json.dumps({
        "name": "arc2_dev80_val40_v1", "train": ["unused_pretraining_task"],
        "eval_groups": {"development_validation": ["a", "b"]},
        "source_manifests": {"arc2/evaluation": {"sha256": hashlib.sha256(source.read_bytes()).hexdigest()}},
        "validation": {"challenge_tasks": 2, "demonstration_pairs": 2, "query_pairs": 3},
    }))
    return source, split


def test_package_separates_query_answers_and_preserves_input_order(holdout, tmp_path):
    source, split = holdout
    output = tmp_path / "package"
    manifest = package_holdout(source, split, output, "owner/holdout")
    tasks = json.loads((output / "arc2_val40_challenges.json").read_text())
    solutions = json.loads((output / "arc2_val40_solutions.json").read_text())
    assert tasks["a"]["train"][0]["output"] == [[2]]
    assert tasks["a"]["test"] == [{"input": [[3]]}, {"input": [[5]]}]
    assert solutions["a"] == [[[4]], [[6]]]
    assert manifest["tasks"] == 2 and manifest["query_pairs"] == 3
    for name, identity in manifest["files"].items():
        assert (output / name).stat().st_size == identity["size"]
        assert ku.sha256_file(output / name) == identity["sha256"]
    assert package_holdout(source, split, output, "owner/holdout") == manifest
    source.write_text("{}")
    with pytest.raises(ValueError, match="pinned"):
        package_holdout(source, split, tmp_path / "invalid", "owner/holdout")


def test_validation_discovery_is_explicit_and_hash_pinned(holdout, tmp_path):
    source, split = holdout
    folder = tmp_path / "dataset"
    manifest = package_holdout(source, split, folder, "owner/holdout")
    name = "arc2_val40_challenges.json"
    identity = {"challenges": {"filename": name, **manifest["files"][name]}}
    assert ku.find_validation_challenges(tmp_path, None, identity) == folder / name
    with pytest.raises(ValueError, match="competition"):
        ku.find_test_challenges(tmp_path, folder / name)
    (folder / name).write_text("{}")
    with pytest.raises(ValueError, match="pinned holdout"):
        ku.find_validation_challenges(tmp_path, None, identity)


def test_validation_scores_task_weighting_and_either_attempt(holdout, tmp_path):
    source, split = holdout
    folder = tmp_path / "dataset"
    package_holdout(source, split, folder, "owner/holdout")
    challenges = json.loads((folder / "arc2_val40_challenges.json").read_text())
    solutions = json.loads((folder / "arc2_val40_solutions.json").read_text())
    predictions = {
        "a": [{"attempt_1": [[4]], "attempt_2": [[0]]},
              {"attempt_1": [[0]], "attempt_2": [[6]]}],
        "b": [{"attempt_1": [[0]], "attempt_2": [[0]]}],
    }
    result = ku.score_validation(predictions, challenges, solutions)
    assert result["top1"]["percent"] == 25.0
    assert result["top2"]["percent"] == 50.0
    assert result["top2"]["correct_queries"] == 2
    assert result["top2"]["query_percent"] == pytest.approx(200 / 3)
    assert result["tasks"] == 2 and result["queries"] == 3
    assert solutions["a"] == [[[4]], [[6]]] and predictions["a"][1]["attempt_1"] == [[0]]
    del solutions["b"]
    with pytest.raises(ValueError, match="validation solutions"):
        ku.score_validation(predictions, challenges, solutions)


def test_validation_notebook_has_full_holdout_defaults_and_reads_answers_last():
    notebook = json.loads((ROOT / "kaggle/validate_arc2_600m_val40.ipynb").read_text())
    settings = "".join(notebook["cells"][1]["source"])
    assert "SMOKE_TEST = False" in settings
    assert "EPOCHS = 90" in settings
    assert PROFILES["600m-val40"]["validation"]["tasks"] == 40
    assert PROFILES["600m-val40"]["validation"]["query_pairs"] == 56
    assert "find_validation_challenges" in "".join(notebook["cells"][7]["source"])
    previous = "".join("".join(cell["source"]) for cell in notebook["cells"][:-1])
    assert "solutions_path.read_text" not in previous
    final = "".join(notebook["cells"][-1]["source"])
    assert final.index("mg.launch_inference") < final.index("solutions_path.read_text")
    assert "validation_predictions.json" in final and "VALIDATION COMPLETE" in final
    assert 'else "submission.json"' not in final and "READY:" not in final
    standard = make_notebook.frozen_runtime(ROOT / "kaggle/submit_arc2_600m.ipynb")
    validation = make_notebook.frozen_runtime(ROOT / "kaggle/validate_arc2_600m_val40.ipynb")
    assert {k: v for k, v in standard.items() if k.startswith("src/")} == {
        k: v for k, v in validation.items() if k.startswith("src/")}
