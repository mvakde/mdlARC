"""Default full-decode selection: symbolic guards, prefix-cached likelihood rescoring, launcher defaults."""
import argparse
import gzip
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
import build
import evaluate
import launch_run
import train
from common import (
    apply_color_permutation_to_grid,
    apply_dihedral_transform,
    symbolic_guard_rejection,
    symbolic_guards,
    upscale_grid,
)
from scale_stream_fixture import fixture_tasks, training_args, write_fixture

LOOP = dict(n_layers=4, loop_iterations=3, loop_start_layer=1, loop_end_layer=3, n_predict=2)


def _looped_eval_model(tmp_path, monkeypatch, scale_views=True):
    """A small looped pair_group model (random weights) and the scale-view eval dataset."""
    monkeypatch.setattr(build, "resolve_device", lambda _: torch.device("cpu"))
    path = write_fixture(tmp_path / "tasks.json")
    model, train_ds, _, _, _ = build.build_model_and_data(training_args(path, scale_augmentation=True, **LOOP))
    torch.manual_seed(0)
    for parameter in model.parameters():
        parameter.data.normal_(0, 0.3)
    checkpoint = tmp_path / "model.pt"
    train.maybe_save_model(model, train_ds, path, checkpoint)
    args = training_args(path, inference_demo_policy="leave_one_out", inference_scale_views=scale_views,
                         scale_augmentation=True, **LOOP)
    model, dataset, _, _, _ = build.build_model_and_data(
        args, checkpoint=build.load_checkpoint(checkpoint), is_eval=True, build_dataloader=False)
    return model, dataset, path


def _wrong_size(grid):
    return [list(row) for row in grid] + [list(grid[0])]


def _changed_cell(grid):
    out = [list(row) for row in grid]
    out[0][0] = (out[0][0] + 1) % 10
    return out


# ----------------------------------------------------------------------------- prefix-cached rescoring

def test_prefix_cached_rescore_matches_full_sequences_on_scale_views(tmp_path, monkeypatch):
    model, dataset, _ = _looped_eval_model(tmp_path, monkeypatch)
    truth = {task_id: task["test"][0]["output"] for task_id, task in fixture_tasks().items()}
    candidates = {(task_id, 0): [{"grid": grid, "count": 3}, {"grid": _changed_cell(grid), "count": 2},
                                 {"grid": [[1, 2]], "count": 1}] for task_id, grid in truth.items()}
    attempts = evaluate._build_inference_attempts(dataset, "test", None, None, 8, True)
    assert any(item.scale_factor != (1, 1) for item in attempts)
    cached = evaluate.run_split_rescore(model, dataset, "test", torch.device("cpu"), candidates,
                                        batch_size=5, augmentation_attempts=8, prefix_cache=True)
    packed = evaluate.run_split_rescore(model, dataset, "test", torch.device("cpu"), candidates,
                                        batch_size=5, augmentation_attempts=8, prefix_cache=False)
    assert cached.keys() == packed.keys()
    for key in cached:
        for a, b in zip(cached[key], packed[key]):
            assert a["scored_attempts"] == b["scored_attempts"] == 8
            assert a["log_likelihood"] == pytest.approx(b["log_likelihood"], rel=1e-3)
        # Clearly separated candidates keep their order between the two paths.
        values = [row["log_likelihood"] for row in packed[key]]
        if min(abs(x - y) for i, x in enumerate(values) for y in values[i + 1:]) > 1.0:
            assert (sorted(range(3), key=lambda j: -cached[key][j]["log_likelihood"])
                    == sorted(range(3), key=lambda j: -values[j]))


def test_prefix_cached_scores_match_one_sequence_at_a_time(tmp_path, monkeypatch):
    """Rows with different candidate counts and lengths share one prefilled batch."""
    model, dataset, _ = _looped_eval_model(tmp_path, monkeypatch, scale_views=False)
    model = evaluate._prepare_model_for_inference(model)
    items = evaluate._build_inference_attempts(dataset, "test", None, None, 3, True)[:4]
    grids = [[[1, 2]], [[3, 3, 3], [4, 4, 4]], [[5]]]
    continuations = [[evaluate._candidate_output_tokens(item, g) for g in grids[:1 + row % 3]]
                     for row, item in enumerate(items)]
    batched = evaluate._score_continuations_cached(model, items, continuations, torch.device("cpu"))
    for row, item in enumerate(items):
        alone = evaluate._score_continuations_cached(model, [item], [continuations[row]], torch.device("cpu"))[0]
        assert batched[row] == pytest.approx(alone, rel=1e-3)
        assert len(batched[row]) == len(continuations[row]) and None not in batched[row]


# ----------------------------------------------------------------------------- symbolic guards

def _selection(task_id, grids_counts):
    ranked = [{"grid": grid, "count": count} for grid, count in grids_counts]
    return evaluate.AAIVRSelection(task_id=task_id, original_pair_index=0,
                                   selected_outputs=[entry["grid"] for entry in ranked[:2]],
                                   ranked_candidates=ranked, num_generated=sum(c for _, c in grids_counts),
                                   num_valid=sum(c for _, c in grids_counts), discarded_non_rectangular=0,
                                   discarded_input_copies=0)


class _Tasks:
    def __init__(self, tasks):
        self.pairs_by_task = {t: task["train"] for t, task in tasks.items()}
        self.test_pairs_by_task = {t: task["test"] for t, task in tasks.items()}


def test_guards_drop_rule_breaking_candidates_and_keep_vote_order():
    # Demos: output = input with colour 3 erased to background (same size, two colours kept).
    tasks = {"t": {"train": [{"input": [[1, 3], [2, 0]], "output": [[1, 0], [2, 0]]},
                             {"input": [[3, 4], [5, 0]], "output": [[0, 4], [5, 0]]}],
                   "test": [{"input": [[3, 6], [7, 0]]}]}}
    guards = symbolic_guards(tasks["t"]["train"])
    assert guards["row_ratio"] == 1 and guards["dropped_color"] == 3 and not guards["single_color"]
    right, wrong_size, kept_colour = [[0, 6], [7, 0]], [[0, 6]], [[3, 6], [7, 7]]
    selection = _selection("t", [(wrong_size, 9), (kept_colour, 5), (right, 2), ([[0, 6], [7, 7]], 1)])
    (guarded,) = evaluate.apply_symbolic_guards([selection], _Tasks(tasks))
    assert guarded.ranked_candidates == [{"grid": right, "count": 2}, {"grid": [[0, 6], [7, 7]], "count": 1}]
    assert guarded.selected_outputs == [right, [[0, 6], [7, 7]]]
    assert guarded.guard_rejections == {"row_ratio": 9, "dropped_color": 5}
    assert symbolic_guard_rejection(guards, wrong_size, tasks["t"]["test"][0]["input"]) == "row_ratio"
    # Every candidate rejected: keep the unfiltered ranking rather than submit nothing.
    (fallback,) = evaluate.apply_symbolic_guards([_selection("t", [(wrong_size, 3)])], _Tasks(tasks))
    assert fallback.selected_outputs == [wrong_size]


# ----------------------------------------------------------------------------- default full-decode pipeline

def test_full_decode_submits_likelihood_top2_after_guards_and_reports_votes(tmp_path, monkeypatch):
    model, dataset, path = _looped_eval_model(tmp_path, monkeypatch)
    tasks = fixture_tasks()
    truth = {task_id: task["test"][0]["output"] for task_id, task in tasks.items()}
    (tmp_path / "solutions.json").write_text(json.dumps({t: [g] for t, g in truth.items()}))
    attempts = evaluate._build_inference_attempts(dataset, "test", None, None, 8, True)

    def decoded(**kwargs):
        results = []
        for visit, a in enumerate(attempts):
            # Half the views vote for a wrong-size grid, the rest split between truth and a near miss.
            grid = {0: _wrong_size, 1: _wrong_size, 2: lambda g: g, 3: _changed_cell}[visit % 4](truth[a.task_id])
            answer = apply_dihedral_transform(
                upscale_grid(apply_color_permutation_to_grid(grid, a.color_mapping), a.scale_factor),
                a.dihedral_index)
            results.append(dict(task_id=a.task_id, pair_index=0, output_grid=answer, original_input=a.original_input,
                                color_mapping=a.color_mapping, source_colors=a.source_colors,
                                dihedral_index=a.dihedral_index,
                                **({} if a.scale_factor == (1, 1) else {"scale_factor": list(a.scale_factor)})))
        return results

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(evaluate, "load_checkpoint", lambda _: {})
    monkeypatch.setattr(evaluate, "build_model_and_data", lambda *a, **k: (model, dataset, None, torch.device("cpu"), path))
    monkeypatch.setattr(evaluate, "run_split_inference", decoded)
    cfg = argparse.Namespace(mode="full-decode", enable_aug=True, seed=42, max_augments=8,
                             inference_symbolic_guards=True, inference_rescore=True, rescore_top_k=3,
                             solutions_path=str(tmp_path / "solutions.json"))
    _, _, submission_path = evaluate.run_evaluation(cfg, run_name="sel", max_augments=8, data_path=path,
                                                    checkpoint_path=tmp_path / "m.pt", batch_size=5)
    submission = json.loads(submission_path.read_text())
    dump = json.load(gzip.open(submission_path.with_name("candidates.json.gz")))
    report = json.loads(submission_path.with_name("oracle_score.json").read_text())
    assert {"top2", "top2_votes", "top2_votes_guards", "oracle", "oracle_after_filters"} <= set(report["metrics"])
    rejected_somewhere = False
    for pair in dump["pairs"]:
        task_id = pair["task_id"]
        guards = symbolic_guards(tasks[task_id]["train"])
        test_input = tasks[task_id]["test"][0]["input"]
        kept = [c for c in pair["candidates"] if not c["input_copy"] and "guard" not in c]
        for c in pair["candidates"]:
            assert c.get("guard") == symbolic_guard_rejection(guards, c["grid"], test_input)
        rejected_somewhere |= any("guard" in c for c in pair["candidates"])
        # The rescored pool is the top-3 surviving candidates by votes; the submission is its two most likely.
        pool = sorted(kept, key=lambda c: -c["count"])[:3]
        assert all("log_likelihood" in c for c in pool)
        expected = [c["grid"] for c in sorted(pool, key=lambda c: -c["log_likelihood"])[:2]]
        attempt = submission[task_id][0]
        assert [attempt["attempt_1"], attempt["attempt_2"]][:len(expected)] == expected
    assert rejected_somewhere
    assert report["metrics"]["oracle"]["percentage"] == 100.0


# ----------------------------------------------------------------------------- launcher defaults

def test_launcher_defaults_are_the_plain_fast_recipe(tmp_path):
    plan = launch_run.build_plan({"enable_finetune": False, "defaults": {}}, tmp_path / "spec.json")
    shared = plan["shared"]
    # Scale, guards and rescoring are opt-in; speculative greedy decoding is on.
    assert plan["pretrain"]["scale_augmentation"] is False
    assert shared["inference_scale_views"] is False and shared["inference_symbolic_guards"] is False
    assert shared["inference_rescore"] is False and shared["rescore_prefix_cache"] is True
    assert shared["rescore_top_k"] == 20
    assert plan["dataset"]["add_rearc"] is None
    assert plan["evals"]["final"]["inference_speculative"] is True
    on = launch_run.build_plan({"enable_finetune": False, "dataset": {"add_rearc": 1000}, "defaults": {
        "scale_augmentation": True, "inference_scale_views": True, "inference_symbolic_guards": True,
        "inference_rescore": True}}, tmp_path / "spec.json")
    assert all(on["shared"][key] for key in ("scale_augmentation", "inference_scale_views",
                                             "inference_symbolic_guards", "inference_rescore"))
    assert on["dataset"]["add_rearc"] == 1000
    auto = launch_run.build_plan({"enable_finetune": False, "eval_defaults": {"inference_speculative": "auto"}},
                                 tmp_path / "spec.json")
    assert auto["evals"]["final"]["inference_speculative"] is True
    # A sampled decode turns the speculative default off; an explicit request still has to be greedy.
    sampled = launch_run.build_plan({"enable_finetune": False, "defaults": {"inference_temperature": 0.7}},
                                    tmp_path / "spec.json")
    assert sampled["evals"]["final"]["inference_speculative"] is False
    sampled_eval = launch_run.build_plan(
        {"enable_finetune": False, "evals": {"hot": {"checkpoint": "pretrain.final", "inference_temperature": 0.7}}},
        tmp_path / "spec.json")
    assert sampled_eval["evals"]["hot"]["inference_speculative"] is False
    with pytest.raises(ValueError, match="greedy"):
        launch_run.build_plan({"enable_finetune": False, "evals": {"hot": {
            "checkpoint": "pretrain.final", "inference_temperature": 0.7, "inference_speculative": True}}},
            tmp_path / "spec.json")
    for bad in ({"inference_symbolic_guards": 1}, {"inference_rescore": "yes"}, {"rescore_prefix_cache": None}):
        with pytest.raises(ValueError):
            launch_run.build_plan({"enable_finetune": False, "defaults": bad}, tmp_path / "spec.json")
    off = launch_run.build_plan({"enable_finetune": False, "defaults": {
        "scale_augmentation": False, "inference_scale_views": False, "inference_symbolic_guards": False,
        "inference_rescore": False, "inference_speculative": False}}, tmp_path / "spec.json")
    assert not any(off["shared"][key] for key in ("scale_augmentation", "inference_scale_views",
                                                   "inference_symbolic_guards", "inference_rescore",
                                                   "inference_speculative"))


def test_default_scale_turns_itself_off_without_augmentation(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "resolve_device", lambda _: torch.device("cpu"))
    path = write_fixture(tmp_path / "tasks.json")
    _, dataset, _, _, _ = build.build_model_and_data(training_args(path, scale_augmentation=True, enable_aug=False))
    assert dataset.scale_augmentation is False


def test_resuming_a_pre_scale_checkpoint_keeps_its_recipe_unless_the_spec_sets_scale(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "resolve_device", lambda _: torch.device("cpu"))
    path = write_fixture(tmp_path / "tasks.json")
    model, dataset, _, _, _ = build.build_model_and_data(training_args(path, scale_augmentation=False))
    checkpoint = tmp_path / "old.pt"
    train.maybe_save_model(model, dataset, path, checkpoint)
    assert "scale_augmentation" not in build.load_checkpoint(checkpoint)
    resume = dict(checkpoint_path=checkpoint, checkpoint_mode="resume", scale_augmentation=True)
    _, kept, _, _, _ = build.build_model_and_data(training_args(path, scale_augmentation_explicit=False, **resume))
    _, changed, _, _, _ = build.build_model_and_data(training_args(path, scale_augmentation_explicit=True, **resume))
    assert kept.scale_augmentation is False and changed.scale_augmentation is True
    plan = launch_run.build_plan({"enable_finetune": False}, tmp_path / "spec.json")
    assert plan["pretrain"]["scale_augmentation_explicit"] is False
    plan = launch_run.build_plan({"enable_finetune": False, "pretrain": {"scale_augmentation": True}}, tmp_path / "s.json")
    assert plan["pretrain"]["scale_augmentation_explicit"] is True
