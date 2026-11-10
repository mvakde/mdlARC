"""Oracle (any-candidate) and voted top-2 scoring written beside every scored full decode."""
import argparse
import gzip
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import evaluate
import launch_run
from evaluate import AAIVRSelection
from utils import oracle_report, oracle_summary_fields as summary_fields
from test_accuracy_evaluation import _dataset


def _selection(task, index, candidates, top2):
    ranked = [{"grid": grid, "count": count} for grid, count in candidates]
    return AAIVRSelection(task_id=task, original_pair_index=index, selected_outputs=top2, ranked_candidates=ranked,
                          num_generated=sum(c for _, c in candidates), num_valid=sum(c for _, c in candidates),
                          discarded_non_rectangular=0, discarded_input_copies=0)


def test_oracle_counts_any_candidate_and_matches_task_mean_aggregation():
    solutions = {"a": [[[1]], [[2]]], "b": [[[3]]], "c": [[[4]]]}
    raw = [_selection("a", 0, [([[9]], 5), ([[8]], 3), ([[1]], 1)], []),   # correct only at rank 3
           _selection("a", 1, [([[2]], 4)], []),
           _selection("b", 0, [([[3]], 1), ([[7]], 6)], []),             # input copy, filtered later
           _selection("c", 0, [([[0]], 2)], [])]
    voted = [_selection("a", 0, [([[9]], 5), ([[8]], 3), ([[1]], 1)], [[[9]], [[8]]]),
             _selection("a", 1, [([[2]], 4)], [[[2]]]),
             _selection("b", 0, [([[7]], 6)], [[[7]]]),
             _selection("c", 0, [([[0]], 2)], [[[0]]])]
    report = oracle_report(raw, voted, solutions, expected_candidates=8)
    m = report["metrics"]
    assert m["top2"]["score"] == 0.5                      # a: 1 of 2 outputs; b, c: miss
    assert m["oracle"]["score"] == 2.0                    # a: both, b: raw candidate, c: miss
    assert m["oracle_after_filters"]["score"] == 1.0      # b's correct grid was filtered out
    assert m["oracle"]["fully_solved_tasks"] == ["a", "b"]
    pair = next(p for p in report["pairs"] if p["task_id"] == "a" and p["pair_index"] == 0)
    assert pair["correct_candidate_votes"] == 1 and pair["correct_candidate_rank_after_filters"] == 3
    missing = oracle_report(raw[:1], voted[:1], solutions)
    assert missing["metrics"]["oracle"]["score"] == 0.5   # absent outputs count as misses


def test_full_decode_writes_oracle_without_changing_the_submission(tmp_path, monkeypatch):
    def run(with_solutions):
        work = tmp_path / ("with" if with_solutions else "without")
        work.mkdir()
        dataset, solutions = _dataset(work, [[[2]]])
        if with_solutions:
            (work / "solutions.json").write_text(json.dumps(solutions))
        monkeypatch.chdir(work)
        monkeypatch.setattr(evaluate, "load_checkpoint", lambda path: {})
        monkeypatch.setattr(evaluate, "build_model_and_data", lambda *args, **kwargs: (
            None, dataset, None, torch.device("cpu"), dataset.source_path))
        outputs = [[[2]], [[5]], [[2]], [[1, 2]]]          # the last one copies the test input
        monkeypatch.setattr(evaluate, "run_split_inference", lambda **kwargs: [{
            "task_id": "0", "pair_index": 0, "output_grid": grid, "original_input": [[1, 2]],
            "color_mapping": list(range(14)), "source_colors": [1, 2], "dihedral_index": 0,
        } for grid in outputs])
        cfg = argparse.Namespace(enable_aug=True, seed=42, inference_temperature=None, inference_top_k=None)
        _, _, path = evaluate.run_evaluation(cfg, run_name="decode", max_augments=4,
                                             data_path=dataset.source_path, checkpoint_path=work / "model.pt")
        return path.resolve()
    plain, scored = run(False), run(True)
    assert json.loads(plain.read_text()) == json.loads(scored.read_text())
    assert not plain.with_name("oracle_score.json").exists()
    for path in (plain, scored):                           # candidates are dumped with or without solutions
        with gzip.open(path.with_name("candidates.json.gz"), "rt") as handle:
            dump = json.load(handle)
        assert dump["expected_candidates_per_test_output"] == 4
        (pair,) = dump["pairs"]
        assert pair["num_generated"] == 4
        assert {(json.dumps(c["grid"]), c["count"], c["input_copy"]) for c in pair["candidates"]} == {
            ("[[2]]", 2, False), ("[[5]]", 1, False), ("[[1, 2]]", 1, True)}
    report = json.loads(scored.with_name("oracle_score.json").read_text())
    assert report["metrics"]["top2"]["score"] == report["metrics"]["oracle"]["score"] == 1.0
    assert report["expected_candidates_per_test_output"] == 4
    fields = summary_fields(scored)
    assert fields["oracle_percentage"] == fields["top2_percentage"] == 100.0


def test_score_summary_carries_top2_and_oracle(tmp_path):
    submission = tmp_path / "submission.json"
    submission.write_text("{}")
    score = {"score": 1.0, "max_score": 2, "percentage": 50.0, "fully_solved_tasks": ["a"]}
    plain = json.loads(launch_run.write_score_summary(submission, score, checkpoint_ref="x", run_name="r").read_text())
    assert "oracle_percentage" not in plain
    (tmp_path / "oracle_score.json").write_text(json.dumps({"metrics": {
        "top2": {"score": 1.0, "percentage": 50.0, "fully_solved_tasks": ["a"]},
        "oracle": {"score": 1.5, "percentage": 75.0, "fully_solved_tasks": ["a"]},
        "oracle_after_filters": {"score": 1.5, "percentage": 75.0, "fully_solved_tasks": ["a"]}}}))
    summary = json.loads(launch_run.write_score_summary(submission, score, checkpoint_ref="x", run_name="r").read_text())
    assert summary["percentage"] == summary["top2_percentage"] == 50.0
    assert summary["oracle_percentage"] == 75.0 and summary["oracle_score"] == 1.5
