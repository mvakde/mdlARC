from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import evaluate
import launch_run
from common import compute_positions_3d, encode_pair
from evaluate import AccuracyAttempt, InferenceAttempt
from src.speculative_benchmark import fixed_prefix_logits, validate_config
from test_speculative_decoding import small_model


@pytest.mark.parametrize("config", [{"task_count": 0}, {"repeats": True}, {"typo": 1}, None])
def test_benchmark_rejects_invalid_config(config):
    with pytest.raises(ValueError):
        validate_config(config)


def test_benchmark_routes_without_training(monkeypatch):
    spec = {"eval_only": True, "checkpoint": "saved.pt", "enable_finetune": False,
            "speculative_benchmark": {}}
    plan = launch_run.build_plan(spec, Path("test.json"))
    plan["skip_data_build"] = True
    calls = []
    monkeypatch.setattr(launch_run, "validate_run_split", lambda _: None)
    monkeypatch.setattr("src.speculative_benchmark.run_benchmark", lambda p: calls.append(p))
    monkeypatch.setattr(launch_run, "run_training_stage_isolated", lambda *_: pytest.fail("Training started"))
    monkeypatch.setattr(launch_run, "run_eval_stage_isolated", lambda *_: pytest.fail("Regular eval started"))
    launch_run.execute_plan(plan)
    assert calls == [plan]
    assert launch_run.preview_plan(plan)["speculative_benchmark"]["repeats"] == 3
    for changed in ({"eval_only": False}, {"enable_finetune": True}, {"checkpoint": None}, {"backend_benchmark": {}}):
        with pytest.raises(ValueError, match="speculative_benchmark"):
            launch_run.build_plan({**spec, **changed}, Path("test.json"))


@torch.inference_mode()
def test_fixed_prefix_probe_compares_the_same_histories(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    model = small_model()
    model.config.max_seq_len = 256
    tokens, sep = encode_pair({"input": [[1, 2]], "output": [[2, 1], [1, 2]]})
    positions = compute_positions_3d(tokens[None])[0]
    item = InferenceAttempt(tokens[:sep + 1].tolist(), positions[:sep + 1], "task", 0,
                            (0, -1), 0, list(range(10)), [1, 2], [[1, 2]])
    attempt = AccuracyAttempt(item, tokens, positions)
    args = (model, attempt, torch.device("cpu"), 8, 128)
    serial = fixed_prefix_logits(*args, block_width=1)
    block = fixed_prefix_logits(*args, block_width=4)
    assert serial.shape == block.shape == (6, 14)
    torch.testing.assert_close(serial, block, rtol=1e-5, atol=1e-6)
    assert not hasattr(model, "_compiled_decode")
    assert not hasattr(model, "_compiled_verify")
