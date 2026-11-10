import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import evaluate
from common import ARCExampleDataset, compute_positions_3d, encode_pair
from tinytransformer import TinyTransformer, TinyTransformerConfig, build_output_targets_and_mask


def _dataset(tmp_path, outputs, *, inline=False):
    tasks = {
        str(index): {
            "train": [{"input": [[1, 2]], "output": [[2, 1]]}],
            "test": [{"input": [[1, 2]], **({"output": output} if inline else {})}],
        }
        for index, output in enumerate(outputs)
    }
    path = tmp_path / "challenges.json"
    path.write_text(json.dumps(tasks))
    return ARCExampleDataset(path, splits=("test",), seed=42), {
        str(index): [output] for index, output in enumerate(outputs)
    }


class ScriptedModel(torch.nn.Module):
    """Predict every reference token except deliberately chosen output errors."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(max_seq_len=256)
        self.calls = []

    def forward(self, **batch):
        assert not self.training
        assert not torch.is_grad_enabled()
        self.calls.append(batch)
        inputs = batch["input_ids"]
        boundaries = batch["cu_seqlens"]
        lengths = boundaries[1:] - boundaries[:-1]
        targets, valid = build_output_targets_and_mask(
            inputs, batch["sep_indices"], cu_seqlens=boundaries
        )
        predictions = targets.clamp_min(0).clone()
        predictions[~valid] = 10  # Wrong prompt predictions must be ignored.
        for row, task_ids in enumerate(batch["example_ids"].tolist()):
            error_offset = {1: 0, 2: 1, 3: 3}.get(task_ids[0])
            if error_offset is not None:
                index = boundaries[row] + batch["sep_indices"][row] + error_offset
                predictions[index] = (predictions[index] + 1) % 14
        logits = torch.full((len(inputs), 14), -10.0)
        logits.scatter_(1, predictions[:, None], 10.0)
        sequence_ids = torch.arange(len(lengths)).repeat_interleave(lengths)
        counts = TinyTransformer._compute_training_metric_counts(
            self, logits, targets, valid, sequence_ids, len(lengths)
        )
        return {"training_metric_counts": counts}


@pytest.mark.parametrize("batch_size,num_batches", [(1, 4), (3, 2), (20, 1)])
def test_accuracy_uses_one_forward_per_batch_and_rejects_any_output_error(tmp_path, batch_size, num_batches):
    dataset, solutions = _dataset(tmp_path, [[[1]], [[1], [2]], [[1], [2]], [[1], [2]]])
    model = ScriptedModel()
    summary = evaluate.run_split_accuracy(
        model, dataset, "test", torch.device("cpu"), solutions, batch_size=batch_size,
    )
    assert len(model.calls) == num_batches
    assert model.calls[0]["input_ids"].ndim == 1
    assert summary["exact_output_correct"] == 1
    assert summary["exact_output_total"] == summary["total_cases"] == 4
    assert summary["exact_output_accuracy"] == 0.25
    assert (summary["color_correct"], summary["color_total"]) == (6, 7)
    assert (summary["special_token_correct"], summary["special_token_total"]) == (5, 7)


def test_accuracy_targets_follow_the_existing_augmentation_and_prompt_schedule(tmp_path):
    dataset, solutions = _dataset(tmp_path, [[[1, 2], [2, 1], [1, 1]]])
    dataset.pairs_by_task["0"] *= 4
    prompts = evaluate._build_inference_attempts(dataset, "test", None, None, 10, True)
    attempts = evaluate._build_accuracy_attempts(dataset, "test", prompts, solutions, 256)
    assert len(attempts) >= 10
    assert len({item.inference.dihedral_index for item in attempts}) > 1
    for item in attempts:
        inference = item.inference
        expected, sep = encode_pair(
            {"input": inference.original_input, "output": solutions["0"][0]},
            inference.dihedral_index, torch.tensor(inference.color_mapping),
        )
        # Context is already fixed by the inference schedule; only append the
        # independently transformed target output to it.
        expected = torch.tensor(inference.prompt + expected[sep + 1:].tolist())
        torch.testing.assert_close(item.tokens, expected)
        assert item.tokens[:len(inference.prompt)].tolist() == inference.prompt
        torch.testing.assert_close(item.positions, compute_positions_3d(expected[None]).squeeze(0))
    batches = evaluate._build_accuracy_batches(attempts, batch_size=3, token_budget=100)
    assert sum(map(len, batches)) == len(attempts)
    assert all(len(batch) <= 3 for batch in batches)
    assert all(len(batch) == 1 or sum(item.tokens.numel() for item in batch) <= 100 for batch in batches)


@pytest.mark.parametrize("problem", ["missing", "partial", "conflict", "empty", "long"])
def test_bad_accuracy_targets_fail_before_forward(tmp_path, problem):
    dataset, solutions = _dataset(tmp_path, [[[1]]], inline=problem == "conflict")
    if problem == "missing":
        solutions = {}
    elif problem == "partial":
        solutions["0"] = []
    elif problem == "conflict":
        solutions["0"] = [[[2]]]
    elif problem == "empty":
        solutions["0"] = [[]]
    model = ScriptedModel()
    if problem == "long":
        model.config.max_seq_len = 4
    with pytest.raises(ValueError):
        evaluate.run_split_accuracy(model, dataset, "test", torch.device("cpu"), solutions)
    assert model.calls == []


def _tiny_model(n_predict=1):
    torch.manual_seed(12)
    return TinyTransformer(TinyTransformerConfig(
        d_model=24, n_heads=4, d_ff=48, n_layers=2, max_seq_len=256,
        num_examples=1, num_task_embedding_latents=2,
        dropout=0.0, attention_dropout=0.0, embedding_dropout=0.0,
        n_predict=n_predict,
    )).eval()


@torch.inference_mode()
@pytest.mark.parametrize("n_predict", [1, 4])
def test_single_pass_target_logits_match_cached_generation(n_predict):
    model = _tiny_model(n_predict)
    tokens, sep = encode_pair(
        {"input": [[1, 2]], "output": [[2], [1]]},
    )
    positions = compute_positions_3d(tokens[None])
    ids = torch.tensor([[0, -1]], dtype=torch.int32)
    dihedrals = torch.tensor([3], dtype=torch.int32)
    measured = model(
        input_ids=tokens, example_ids=ids, sep_indices=torch.tensor([sep]),
        dihedral_ids=dihedrals,
        positions_3d=positions[0], cu_seqlens=torch.tensor([0, len(tokens)], dtype=torch.int32),
        max_seqlen=len(tokens), compute_training_metrics=True,
    )
    prefix_length = sep + 1
    prefill = model.forward_generate(
        input_ids=tokens[None, :prefix_length], example_ids=ids,
        dihedral_ids=dihedrals,
        positions_3d=positions[:, :prefix_length],
    )
    torch.testing.assert_close(measured["logits"][sep], prefill["logits"][0, -1], atol=1e-5, rtol=1e-5)
    caches = []
    for keys, values in prefill["past_key_values"]:
        buffers = [torch.zeros((*keys.shape[:2], len(tokens), keys.shape[-1])) for _ in range(2)]
        buffers[0][:, :, :prefix_length] = keys
        buffers[1][:, :, :prefix_length] = values
        caches.append(tuple(buffers))
    for index in range(prefix_length, len(tokens) - 1):
        decoded = model.forward_generate(
            input_ids=tokens[None, index:index + 1], example_ids=ids,
            dihedral_ids=dihedrals,
            positions_3d=positions[:, index:index + 1], past_key_values=tuple(caches),
            cache_position=torch.tensor([index]),
            attention_mask=(torch.arange(len(tokens)) <= index)[None],
        )
        torch.testing.assert_close(measured["logits"][index], decoded["logits"][0, -1], atol=1e-5, rtol=1e-5)


def test_accuracy_only_skips_mtp_heads(tmp_path, monkeypatch):
    dataset, solutions = _dataset(tmp_path, [[[1]]])
    model = _tiny_model(4)
    for head in model.lm_head_mtp:
        monkeypatch.setattr(head, "forward", lambda *args: pytest.fail("Accuracy evaluation ran an MTP head"))
    summary = evaluate.run_split_accuracy(model, dataset, "test", torch.device("cpu"), solutions)
    assert summary["total_cases"] == 1


@pytest.mark.parametrize("inline", [False, True])
def test_accuracy_evaluation_writes_report_without_decoding(tmp_path, monkeypatch, inline):
    dataset, solutions = _dataset(tmp_path, [[[0]]], inline=inline)
    if not inline:
        (tmp_path / "solutions.json").write_text(json.dumps(solutions))
    model = _tiny_model()
    with torch.no_grad():
        model.lm_head.weight.zero_()  # Correct color; wrong end token.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(evaluate, "load_checkpoint", lambda path: {})
    monkeypatch.setattr(evaluate, "build_model_and_data", lambda *args, **kwargs: (
        model, dataset, None, torch.device("cpu"), dataset.source_path,
    ))
    monkeypatch.setattr(evaluate, "run_split_inference", lambda **kwargs: pytest.fail("decoded"))
    monkeypatch.setattr(evaluate, "run_aaivr_on_results", lambda *args: pytest.fail("AAIVR ran"))
    cfg = argparse.Namespace(mode="accuracy-only", enable_aug=False, seed=42, max_augments=80)
    original_cfg = vars(cfg).copy()
    _, results, path = evaluate.run_evaluation(
        cfg, run_name="test", max_augments=80, data_path=dataset.source_path,
        checkpoint_path=tmp_path / "model.pt",
    )
    report = json.loads(path.read_text())
    assert path.name == "accuracy.json"
    assert report["mode"] == "accuracy-only"
    assert report["splits"] == results
    summary = results["test"]["summary"]
    assert summary["color_accuracy"] == 1.0
    assert summary["special_token_accuracy"] == 0.0
    assert summary["exact_output_accuracy"] == 0.0
    assert summary["exact_output_total"] == 1
    assert not path.with_name("submission.json").exists()
    assert vars(cfg) == original_cfg


@pytest.mark.parametrize("mode", [None, "full-decode"])
def test_default_and_explicit_full_decode_still_build_aaivr_submission(tmp_path, monkeypatch, mode):
    dataset, _ = _dataset(tmp_path, [[[2]]])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(evaluate, "load_checkpoint", lambda path: {})
    monkeypatch.setattr(evaluate, "build_model_and_data", lambda *args, **kwargs: (
        None, dataset, None, torch.device("cpu"), dataset.source_path,
    ))
    monkeypatch.setattr(evaluate, "run_split_accuracy", lambda **kwargs: pytest.fail("accuracy ran"))
    calls = []

    def inference(**kwargs):
        calls.append(kwargs)
        return [{
            "task_id": "0", "pair_index": 0, "output_grid": [[2]],
            "original_input": [[1]], "color_mapping": list(range(14)),
            "source_colors": [1, 2], "dihedral_index": 0,
        }]

    monkeypatch.setattr(evaluate, "run_split_inference", inference)
    cfg = argparse.Namespace(enable_aug=True, seed=42, inference_temperature=0.7, inference_top_k=3)
    if mode is not None:
        cfg.mode = mode
    _, _, path = evaluate.run_evaluation(
        cfg, run_name="decode", max_augments=10, data_path=dataset.source_path,
        checkpoint_path=tmp_path / "model.pt",
    )
    assert calls[0]["augmentation_attempts"] == 10
    assert calls[0]["temperature"] == 0.7
    assert calls[0]["top_k"] == 3
    assert path.name == "submission.json"
    assert json.loads(path.read_text()) == {"0": [{"attempt_1": [[2]], "attempt_2": [[2]]}]}
    assert not path.with_name("accuracy.json").exists()
