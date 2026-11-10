"""Decode-context demo cap (inference_max_demos) for pair_group evaluation prompts."""
from collections import Counter
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import launch_run
from common import ARCExampleDataset
from pair_identity import leave_one_out, schedule_seed


def _dataset(tmp_path, demos, **options):
    pairs = [{"input": [[i % 10, 1]], "output": [[(i + 1) % 10]]} for i in range(demos)]
    path = tmp_path / f"tasks{demos}.json"
    path.write_text(json.dumps({"task": {"train": pairs, "test": [{"input": [[7, 8]], "output": [[9]]}]}}))
    return ARCExampleDataset(path, seed=42, splits=("test",), attention_pattern="pair_group", **options)


@pytest.mark.parametrize("demos", range(2, 10))
def test_default_cap_keeps_leave_one_out_unchanged(tmp_path, demos):
    dataset = _dataset(tmp_path, demos, inference_demo_policy="leave_one_out")
    example = next(dataset.iter_examples(split="test"))
    seed = schedule_seed(42, example.task_id, example.split, example.pair_index)
    for visit in range(3 * demos):
        assert dataset.demonstration_indices(example, visit) == leave_one_out(range(demos), visit=visit, seed=seed)


@pytest.mark.parametrize("cap", [1, 3, 4])
@pytest.mark.parametrize("demos", range(2, 10))
def test_cap_bounds_leave_one_out_context_and_balances_use(tmp_path, demos, cap):
    dataset = _dataset(tmp_path, demos, inference_demo_policy="leave_one_out", inference_max_demos=cap)
    example = next(dataset.iter_examples(split="test"))
    uses = Counter()
    for visit in range(5 * demos):
        row = dataset.demonstration_indices(example, visit)
        assert len(row) == len(set(row)) == min(demos - 1, cap)
        assert set(row) <= set(range(demos))
        uses.update(row)
    values = [uses[i] for i in range(demos)]
    assert max(values) - min(values) <= 1


@pytest.mark.parametrize("demos", range(1, 9))
def test_cap_applies_to_all_demo_policy(tmp_path, demos):
    dataset = _dataset(tmp_path, demos, inference_demo_policy="all", inference_max_demos=3)
    example = next(dataset.iter_examples(split="test"))
    assert all(len(dataset.demonstration_indices(example, visit)) == min(demos, 3) for visit in range(8))


@pytest.mark.parametrize("bad", [0, 7, True, 4.0])
def test_invalid_cap_is_rejected(tmp_path, bad):
    with pytest.raises(ValueError, match="inference_max_demos"):
        _dataset(tmp_path, 4, inference_demo_policy="leave_one_out", inference_max_demos=bad)
    with pytest.raises(ValueError, match="inference_max_demos"):
        launch_run.training_overrides({"inference_max_demos": bad}, "eval_defaults",
                                      allowed_keys=launch_run.EVAL_CONFIG_KEYS)


def test_launcher_accepts_cap_in_eval_defaults():
    assert "inference_max_demos" in launch_run.EVAL_CONFIG_KEYS
    assert launch_run.training_overrides({"inference_max_demos": 4}, "eval_defaults",
                                         allowed_keys=launch_run.EVAL_CONFIG_KEYS)["inference_max_demos"] == 4
