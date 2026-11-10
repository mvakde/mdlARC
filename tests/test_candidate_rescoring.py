"""Candidate re-scoring: teacher-forced likelihood of a decode's candidates under its schedule."""
import argparse
import gzip
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate
from common import compute_positions_3d, encode_pair
from tinytransformer import build_output_targets_and_mask
from test_accuracy_evaluation import _dataset


class ReferenceModel(torch.nn.Module):
    """Confident about the reference output of every prompt; log-prob -0.1 per right token."""

    def __init__(self, references):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(max_seq_len=256)
        self.references = references
        self.calls = 0

    def forward(self, **batch):
        assert not torch.is_grad_enabled()
        self.calls += 1
        inputs, boundaries = batch["input_ids"], batch["cu_seqlens"]
        logits = torch.zeros((len(inputs), 14))
        for row in range(len(boundaries) - 1):
            start, end = int(boundaries[row]), int(boundaries[row + 1])
            sep = start + int(batch["sep_indices"][row])
            prompt = inputs[start:sep + 1].tolist()
            reference = self.references[tuple(prompt)]
            for offset, token in enumerate(reference):
                if sep + offset < end:
                    logits[sep + offset, token] = 8.0
        return {"logits": logits}


def _candidates_file(tmp_path, pairs):
    path = tmp_path / "candidates.json.gz"
    with gzip.open(path, "wt") as handle:
        json.dump({"format": 1, "expected_candidates_per_test_output": 4, "pairs": pairs}, handle)
    return path


def _references(dataset, solutions, attempts):
    references = {}
    for item in attempts:
        tokens, sep = encode_pair({"input": item.original_input, "output": solutions[item.task_id][item.pair_index]},
                                  item.dihedral_index, torch.tensor(item.color_mapping))
        references[tuple(item.prompt)] = tokens[sep + 1:].tolist()
    return references


def test_attempts_append_each_candidate_under_the_decode_schedule(tmp_path):
    dataset, _ = _dataset(tmp_path, [[[1, 2], [2, 1]]])
    dataset.pairs_by_task["0"] *= 4
    work = evaluate._build_inference_attempts(dataset, "test", None, None, 6, True)
    candidates = {("0", 0): [{"grid": [[2, 1], [1, 2]], "count": 3}, {"grid": [[1]], "count": 1}]}
    attempts, too_long = evaluate._build_rescore_attempts(work, candidates, 256)
    assert too_long == 0 and len(attempts) == 2 * len(work)
    assert len({item.inference.dihedral_index for item in attempts}) > 1
    for item in attempts:
        grid = candidates[("0", 0)][item.candidate_index]["grid"]
        expected, sep = encode_pair({"input": item.inference.original_input, "output": grid},
                                    item.inference.dihedral_index, torch.tensor(item.inference.color_mapping))
        torch.testing.assert_close(item.tokens, torch.tensor(item.inference.prompt + expected[sep + 1:].tolist()))
        torch.testing.assert_close(item.positions, compute_positions_3d(item.tokens[None]).squeeze(0))


def test_likelihood_beats_votes_and_sums_over_every_augmentation(tmp_path):
    dataset, solutions = _dataset(tmp_path, [[[2, 1]], [[1, 1], [2, 2]]])
    work = evaluate._build_inference_attempts(dataset, "test", None, None, 5, True)
    model = ReferenceModel(_references(dataset, solutions, work))
    candidates = {
        ("0", 0): [{"grid": [[1, 2]], "count": 4}, {"grid": [[2, 1]], "count": 1}],
        ("1", 0): [{"grid": [[1, 1], [2, 2]], "count": 3}, {"grid": [[1, 1]], "count": 2}],
    }
    scored = evaluate.run_split_rescore(model, dataset, "test", torch.device("cpu"), candidates,
                                        batch_size=3, augmentation_attempts=5, prefix_cache=False)
    per_pair = {key: sum(1 for item in work if (item.task_id, item.pair_index) == key) for key in candidates}
    for key, rows in scored.items():
        assert [row["scored_attempts"] for row in rows] == [per_pair[key]] * 2
    wrong, right = scored[("0", 0)]
    assert right["log_likelihood"] > wrong["log_likelihood"]
    # The right grid costs only the softmax mass of 13 other tokens at each output position;
    # transposing augmentations change the output's token count.
    references = _references(dataset, solutions, work)
    output_tokens = sum(len(references[tuple(item.prompt)]) for item in work if item.task_id == "0")
    per_token = -torch.log1p(13 * torch.exp(torch.tensor(-8.0))).item()
    assert right["log_likelihood"] == pytest.approx(output_tokens * per_token, rel=1e-4)
    assert evaluate._rank_candidates(scored[("0", 0)], "votes")[0]["grid"] == [[1, 2]]
    assert evaluate._rank_candidates(scored[("0", 0)], "likelihood")[0]["grid"] == [[2, 1]]
    tied = [{"count": 2, "log_likelihood": -5.0, "grid": "a"}, {"count": 2, "log_likelihood": -1.0, "grid": "b"}]
    assert evaluate._rank_candidates(tied, "votes_then_likelihood")[0]["grid"] == "b"


def test_report_scores_each_rule_and_records_candidates(tmp_path):
    solutions = {"a": [[[1]], [[2]]], "b": [[[3]]]}
    (tmp_path / "solutions.json").write_text(json.dumps(solutions))
    scored = {
        ("a", 0): [{"grid": [[9]], "count": 5, "log_likelihood": -9.0, "scored_attempts": 2},
                   {"grid": [[8]], "count": 4, "log_likelihood": -8.0, "scored_attempts": 2},
                   {"grid": [[1]], "count": 1, "log_likelihood": -1.0, "scored_attempts": 2}],
        ("a", 1): [{"grid": [[2]], "count": 3, "log_likelihood": -2.0, "scored_attempts": 2}],
        ("b", 0): [{"grid": [[3]], "count": 2, "log_likelihood": float("-inf"), "scored_attempts": 1},
                   {"grid": [[4]], "count": 1, "log_likelihood": -0.5, "scored_attempts": 2}],
    }
    path = evaluate.write_rescore_report(tmp_path, scored, tmp_path / "solutions.json", metadata={"mode": "rescore"})
    report = json.loads(path.read_text())
    assert report["rules"]["votes"]["score"] == pytest.approx(1.5)
    assert report["rules"]["likelihood"]["score"] == pytest.approx(2.0)
    assert report["test_outputs_with_correct_candidate"] == 3
    assert [c["correct"] for c in report["pairs"][0]["candidates"]] == [False, False, True]
    submission = json.loads((tmp_path / "rescore_likelihood_submission.json").read_text())
    assert submission["a"][0]["attempt_1"] == [[1]] and submission["b"][0]["attempt_2"] == [[3]]


def test_load_drops_input_copies_and_keeps_top_k_by_votes(tmp_path):
    path = _candidates_file(tmp_path, [{"task_id": "0", "pair_index": 0, "candidates": [
        {"grid": [[1]], "count": 1, "input_copy": False},
        {"grid": [[2]], "count": 9, "input_copy": True},
        {"grid": [[3]], "count": 4, "input_copy": False},
        {"grid": [[4]], "count": 4, "input_copy": False},
    ]}])
    loaded = evaluate._load_rescore_candidates(path, 2)
    assert [entry["grid"] for entry in loaded[("0", 0)]] == [[[3]], [[4]]]


def test_rescore_mode_writes_report_without_decoding(tmp_path, monkeypatch):
    dataset, solutions = _dataset(tmp_path, [[[2, 1]]])
    (tmp_path / "solutions.json").write_text(json.dumps(solutions))
    work = evaluate._build_inference_attempts(dataset, "test", None, None, 4, True)
    model = ReferenceModel(_references(dataset, solutions, work))
    path = _candidates_file(tmp_path, [{"task_id": "0", "pair_index": 0, "candidates": [
        {"grid": [[1, 2]], "count": 3, "input_copy": False}, {"grid": [[1, 1]], "count": 2, "input_copy": False},
        {"grid": [[2, 1]], "count": 1, "input_copy": False}]}])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(evaluate, "load_checkpoint", lambda path: {})
    monkeypatch.setattr(evaluate, "build_model_and_data", lambda *args, **kwargs: (
        model, dataset, None, torch.device("cpu"), dataset.source_path))
    monkeypatch.setattr(evaluate, "run_split_inference", lambda **kwargs: pytest.fail("decoded"))
    cfg = argparse.Namespace(mode="rescore", enable_aug=True, seed=42, max_augments=4,
                             candidates_path=str(path), rescore_top_k=5, solutions_path=str(tmp_path / "solutions.json"),
                             rescore_prefix_cache=False)  # ReferenceModel scores full sequences only
    _, results, report_path = evaluate.run_evaluation(
        cfg, run_name="test", max_augments=4, data_path=dataset.source_path, checkpoint_path=tmp_path / "m.pt")
    report = json.loads(report_path.read_text())
    assert report_path.name == "rescore.json" and report["top_k"] == 5
    assert report["rules"]["votes"]["percentage"] == 0.0
    assert report["rules"]["likelihood"]["percentage"] == 100.0
    assert results["test"]["summary"] == report["rules"]
    assert not (report_path.parent / "submission.json").exists()


def test_launcher_requires_candidates_for_rescore():
    import launch_run
    with pytest.raises(ValueError, match="candidates_path"):
        launch_run.validate_eval_mode({"mode": "rescore"}, "evals.rescore")
    with pytest.raises(ValueError, match="rescore_top_k"):
        launch_run.validate_eval_mode({"mode": "rescore", "candidates_path": "c.json.gz", "rescore_top_k": 0}, "evals.r")
    launch_run.validate_eval_mode({"mode": "rescore", "candidates_path": "c.json.gz", "rescore_top_k": 20}, "evals.r")


def test_packed_likelihood_matches_single_sequence_forward_for_a_looped_model(tmp_path):
    from tinytransformer import TinyTransformer, TinyTransformerConfig
    torch.manual_seed(0)
    model = TinyTransformer(TinyTransformerConfig(
        d_model=24, n_heads=4, d_ff=48, n_layers=4, max_seq_len=256, num_examples=8,
        num_task_embedding_latents=4, n_predict=2, dropout=0, embedding_dropout=0, attention_dropout=0,
        task_embeddings=False, loop_iterations=3, loop_start_layer=1, loop_end_layer=3)).eval()
    dataset, _ = _dataset(tmp_path, [[[2, 1]], [[1, 1], [2, 2]]])
    candidates = {("0", 0): [{"grid": [[2, 1]], "count": 2}, {"grid": [[1]], "count": 1}],
                  ("1", 0): [{"grid": [[1, 1], [2, 2]], "count": 1}]}
    scored = evaluate.run_split_rescore(model, dataset, "test", torch.device("cpu"), candidates,
                                        batch_size=4, augmentation_attempts=3, prefix_cache=False)
    work = evaluate._build_inference_attempts(dataset, "test", None, None, 3, True)
    attempts, _ = evaluate._build_rescore_attempts(work, candidates, 256)
    manual = {}
    with torch.no_grad():
        for item in attempts:
            sep = len(item.inference.prompt) - 1
            length = item.tokens.numel()
            logits = model(input_ids=item.tokens, example_ids=torch.tensor([item.inference.task_embedding_ids], dtype=torch.int32),
                           sep_indices=torch.tensor([sep], dtype=torch.int32),
                           dihedral_ids=torch.tensor([item.inference.dihedral_index], dtype=torch.int32),
                           positions_3d=item.positions, cu_seqlens=torch.tensor([0, length], dtype=torch.int32),
                           max_seqlen=length, compute_mtp_loss=False)["logits"]
            value = torch.log_softmax(logits[sep:length - 1].float(), -1).gather(
                1, item.tokens[sep + 1:, None]).sum().item()
            key = (item.inference.task_id, item.inference.pair_index, item.candidate_index)
            manual[key] = manual.get(key, 0.0) + value
    for (task_id, pair_index), rows in scored.items():
        for index, row in enumerate(rows):
            assert row["log_likelihood"] == pytest.approx(manual[(task_id, pair_index, index)], rel=1e-4, abs=1e-4)
