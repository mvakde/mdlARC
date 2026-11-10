from pathlib import Path

import pytest
import torch

import launch_run
from src.backend_benchmark import cached_target_logits, compare_logits, compare_runs, validate_config
from evaluate import AccuracyAttempt, InferenceAttempt
from common import compute_positions_3d, encode_pair
from tinytransformer import TinyTransformer, TinyTransformerConfig


@pytest.mark.parametrize("options", [{"task_count": 0}, {"parity_tokens": True},
                                     {"variants": ["current_bf16", "current_bf16"]},
                                     {"variants": ["unknown"]}, {"typo": 1}])
def test_invalid_configuration(options):
    with pytest.raises(ValueError):
        validate_config(options)


def test_diagnostic_cannot_train_or_resume():
    spec = {"backend_benchmark": {}, "eval_only": True, "checkpoint": "saved.pt",
            "enable_finetune": False}
    plan = launch_run.build_plan(spec, Path("test.json"))
    assert plan["backend_benchmark"]["task_count"] == 40
    for updates in ({"eval_only": False}, {"checkpoint": None}, {"enable_finetune": True}):
        with pytest.raises(ValueError, match="backend_benchmark"):
            launch_run.build_plan({**spec, **updates}, Path("test.json"))


def test_comparison_reports_prediction_flips_and_rejects_nonfinite_logits():
    result = compare_logits([[2, 1], [1, 1.01]], [[2.1, 1], [1.02, 1]])
    assert result["tokens"] == 2 and result["argmax_disagreements"] == 1
    assert result["changed_reference_margins"][0] == pytest.approx(.01)
    with pytest.raises(ValueError):
        compare_logits([[float("nan"), 1]], [[2, 1]])


def test_accuracy_comparison_refuses_different_schedules():
    with pytest.raises(ValueError, match="different examples"):
        compare_runs({"selection_sha256": "a"}, {"selection_sha256": "b"})


@torch.inference_mode()
def test_cached_probe_matches_full_forward_with_bucket_padding():
    torch.manual_seed(5)
    model = TinyTransformer(TinyTransformerConfig(
        d_model=24, n_heads=4, d_ff=48, n_layers=2, max_seq_len=256,
        num_examples=1, num_task_embedding_latents=2,
        dropout=0, attention_dropout=0, embedding_dropout=0,
    )).eval()
    tokens, sep = encode_pair({"input": [[1, 2]], "output": [[2], [1]]})
    positions = compute_positions_3d(tokens[None])[0]
    item = InferenceAttempt(tokens[:sep + 1].tolist(), positions[:sep + 1],
                            "task", 0, (0, -1), 3, list(range(10)), [1, 2], [[1, 2]])
    attempt = AccuracyAttempt(item, tokens, positions)
    expected = model(input_ids=tokens[None], example_ids=torch.tensor([[0, -1]]),
                     dihedral_ids=torch.tensor([3]), sep_indices=torch.tensor([sep]),
                     positions_3d=positions[None])["logits"][0, sep:-1]
    actual = cached_target_logits(model, attempt, torch.device("cpu"), 128, 128,
                                  compile_decode=False)
    torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-5)
