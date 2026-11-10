from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import evaluate
import launch_run
from common import END_PAIR_TOKEN_ID, NEXT_LINE_TOKEN_ID, MAX_PAIR_LEN, MAX_SEQ_LEN, encode_example, encode_context
from src.evaluation_budget_benchmark import compare_outputs, validate_config


def test_maximum_output_budget_already_includes_end_token():
    grid = [[2] * 30 for _ in range(30)]
    prompt = encode_example(grid, include_output=False)
    output = encode_example(grid, grid)[len(prompt):]
    assert len(prompt) == 931
    assert len(output) == evaluate.DEFAULT_MAX_NEW_TOKENS == 930
    assert sum(token == 2 for token in output) == 900
    assert output.count(NEXT_LINE_TOKEN_ID) == 29
    assert output.count(END_PAIR_TOKEN_ID) == 1
    assert output[-1] == END_PAIR_TOKEN_ID
    assert len(prompt) + len(output) == MAX_PAIR_LEN
    context, sep = encode_context([{"input": grid, "output": grid}] * 2, {"input": grid})
    assert len(context) == sep + 1 == MAX_SEQ_LEN - 930
    assert evaluate.batched_greedy_generate.__wrapped__.__defaults__[0] == 930


def test_larger_token_budget_respects_requested_batch_cap_and_output_capacity():
    work = [SimpleNamespace(prompt=[0] * 931) for _ in range(600)]
    def batches(budget, cap):
        return evaluate._build_inference_batches(work, batch_size=cap, token_budget=budget,
            max_seq_len=MAX_SEQ_LEN, max_new_tokens=930)
    assert [len(b.items) for b in batches(65536, 32)] == [len(b.items) for b in batches(262144, 32)]
    assert len(batches(65536, 512)[0].items) == 64
    assert len(batches(262144, 512)[0].items) == 256
    for batch in batches(262144, 512):
        assert batch.padded_prompt_length == 1024
        assert batch.padded_prompt_length + 930 <= MAX_SEQ_LEN
        assert len(batch.items) * batch.padded_prompt_length <= 262144


def test_eval_only_spec_threads_custom_budget_to_generation_without_training(tmp_path, monkeypatch):
    spec = {"run_name": "eval-retry", "eval_only": True, "checkpoint": "saved.pt",
            "enable_finetune": False, "eval_defaults": {"batch_size": 150, "prefill_token_budget": 262144}}
    plan = launch_run.build_plan(spec, tmp_path / "spec.json")
    assert launch_run.preview_plan(plan)["pretrain"] is None
    assert launch_run.preview_plan(plan)["evals"]["final"]["prefill_token_budget"] == 262144
    plan["skip_data_build"] = True
    calls = []
    monkeypatch.setattr(launch_run, "validate_run_split", lambda _: None)
    monkeypatch.setattr(launch_run, "run_training_stage_isolated", lambda *a: pytest.fail("Evaluation retry retrained"))
    monkeypatch.setattr(launch_run, "run_eval_stage_isolated", lambda p, n: calls.append(p["evals"][n]))
    launch_run.execute_plan(plan)
    assert len(calls) == 1 and calls[0]["batch_size"] == 150

    class ReachedGeneration(Exception):
        pass
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(evaluate, "load_checkpoint", lambda _: {})
    monkeypatch.setattr(evaluate, "build_model_and_data", lambda *a, **kw: (object(), object(), None, "cpu", None))
    def generation(**kw):
        assert kw["batch_size"] == 150 and kw["prefill_token_budget"] == 262144
        raise ReachedGeneration
    monkeypatch.setattr(evaluate, "run_split_inference", generation)
    with pytest.raises(ReachedGeneration):
        evaluate.run_evaluation(SimpleNamespace(**calls[0]), run_name="eval-retry", max_augments=80,
                                data_path=Path("data.json"), checkpoint_path=Path("saved.pt"), batch_size=150)


@pytest.mark.parametrize("budget", [0, -1, True, 2.5, "262144", None])
def test_eval_spec_rejects_invalid_token_budget(budget):
    with pytest.raises(ValueError, match="prefill_token_budget"):
        launch_run.build_plan({"eval_defaults": {"prefill_token_budget": budget}}, Path("invalid.json"))


def test_inference_batcher_receives_configured_budget(monkeypatch):
    class ReachedBatcher(Exception):
        pass
    monkeypatch.setattr(evaluate, "_build_inference_attempts", lambda *a: [])
    monkeypatch.setattr(evaluate, "_prepare_model_for_inference", lambda m: m)
    def batcher(*a, **kw):
        assert kw["token_budget"] == 262144 and kw["batch_size"] == 150
        raise ReachedBatcher
    monkeypatch.setattr(evaluate, "_build_inference_batches", batcher)
    model = SimpleNamespace(config=SimpleNamespace(max_seq_len=10000))
    with pytest.raises(ReachedBatcher):
        evaluate.run_split_inference(model, None, "test", "cpu", batch_size=150, prefill_token_budget=262144)


@pytest.mark.parametrize("config", [None, {"typo": 1}, {"repeats": True}, {"headroom_fraction": 1},
                                  {"variants": []}, {"variants": [{"token_budget": 0, "batch_size": 32}]}])
def test_budget_diagnostic_rejects_invalid_config(config):
    with pytest.raises(ValueError):
        validate_config(config)


def test_budget_diagnostic_routes_without_training(monkeypatch):
    spec = {"eval_only": True, "checkpoint": "saved.pt", "enable_finetune": False,
            "evaluation_budget_benchmark": {}}
    plan = launch_run.build_plan(spec, Path("test.json"))
    plan["skip_data_build"] = True
    calls = []
    monkeypatch.setattr(launch_run, "validate_run_split", lambda _: None)
    monkeypatch.setattr("src.evaluation_budget_benchmark.run_benchmark", lambda p: calls.append(p))
    monkeypatch.setattr(launch_run, "run_training_stage_isolated", lambda *_: pytest.fail("Training started"))
    monkeypatch.setattr(launch_run, "run_eval_stage_isolated", lambda *_: pytest.fail("Regular eval started"))
    launch_run.execute_plan(plan)
    assert calls == [plan]
    assert launch_run.preview_plan(plan)["evaluation_budget_benchmark"]["max_new_tokens"] == 930
    for changed in ({"eval_only": False}, {"enable_finetune": True}, {"checkpoint": None},
                    {"speculative_benchmark": {}}, {"backend_benchmark": {}}, {"training_benchmark": {}}):
        with pytest.raises(ValueError, match="evaluation_budget_benchmark"):
            launch_run.build_plan({**spec, **changed}, Path("test.json"))


def test_budget_comparison_reports_greedy_changes_by_attempt():
    left = {"selection_sha256": "same", "outputs": {"ordinary": [[1, 13], [2, 13]]}}
    right = {"selection_sha256": "same", "outputs": {"ordinary": [[1, 13], [3, 13]]}}
    assert compare_outputs(left, right)["ordinary"] == {
        "changed_attempts": 1, "attempt_indices": [1], "total_attempts": 2,
    }
    with pytest.raises(ValueError, match="different inference"):
        compare_outputs(left, {**right, "selection_sha256": "different"})
