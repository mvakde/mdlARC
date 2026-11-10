"""Block-upscale (scale) augmentation: transforms, spec pools, training stream and decode inverse."""
import json
import random
import sys
from collections import Counter
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
from common import (
    IO_SEPARATOR_TOKEN_ID,
    START_TOKEN_ID,
    apply_color_permutation_to_grid,
    apply_dihedral_transform,
    apply_inverse_dihedral_transform,
    compute_positions_3d,
    encode_context,
    split_grids_from_tokens,
    demo_bounded_scale_specs,
    downscale_grid_exact,
    inference_scale_schedule,
    oriented_scale_factor,
    pair_scale_specs,
    scale_specs_for_dims,
    sequence_scale_specs,
    upscale_grid,
)
from scale_stream_fixture import fixture_tasks, stream_digest, training_args, training_loader, write_fixture

# Training-stream digest of tests/scale_stream_fixture.py recorded with the code before scale
# augmentation existed (se-exp 2d8a70e, and identically with the loopL-d384-exit-mid6-e330 snapshot).
PRE_SCALE_STREAM_DIGEST = "c54383d43b6c6c3f616c1cfa7d5b0ebc03c235f8e62caf3bf89edabe60f6812e"


def random_grid(rng, rows, cols):
    return [[rng.randrange(10) for _ in range(cols)] for _ in range(rows)]


@pytest.fixture
def cpu(monkeypatch):
    monkeypatch.setattr(build, "resolve_device", lambda _: torch.device("cpu"))


# ----------------------------------------------------------------------------- transforms

def test_upscale_downscale_round_trip():
    rng = random.Random(1)
    for _ in range(200):
        rows, cols = rng.randint(1, 10), rng.randint(1, 10)
        grid = random_grid(rng, rows, cols)
        for spec in scale_specs_for_dims(rows, cols, max_specs=None):
            scaled = upscale_grid(grid, spec)
            assert len(scaled) == rows * spec[0] and all(len(r) == cols * spec[1] for r in scaled)
            for r in range(len(scaled)):
                for c in range(len(scaled[0])):
                    assert scaled[r][c] == grid[r // spec[0]][c // spec[1]]
            assert downscale_grid_exact(scaled, spec) == grid
    assert upscale_grid([[1, 2]], (1, 1)) == [[1, 2]]


def test_exact_downscale_rejects_non_upscales():
    scaled = upscale_grid([[1, 2], [3, 4]], (2, 3))
    assert downscale_grid_exact(scaled, (2, 3)) == [[1, 2], [3, 4]]
    assert downscale_grid_exact(scaled, (3, 3)) is None  # 4 rows not divisible by 3
    assert downscale_grid_exact(scaled, (2, 4)) is None  # 6 columns not divisible by 4
    broken = [list(row) for row in scaled]
    broken[1][4] = 9  # one cell of the (0, 1) block differs
    assert downscale_grid_exact(broken, (2, 3)) is None
    assert downscale_grid_exact([[1, 1], [1]], (1, 1)) is None  # ragged
    assert downscale_grid_exact([], (2, 2)) is None
    # A uniform grid is an upscale of a smaller one under any factor that divides it.
    assert downscale_grid_exact([[5] * 4] * 2, (2, 2)) == [[5, 5]]


def test_oriented_factor_commutes_scale_with_every_dihedral():
    rng = random.Random(2)
    grid = random_grid(rng, 3, 5)
    for dihedral in range(8):
        for spec in ((1, 1), (2, 3), (3, 1), (1, 2)):
            direct = apply_dihedral_transform(upscale_grid(grid, spec), dihedral)
            oriented = upscale_grid(apply_dihedral_transform(grid, dihedral), oriented_scale_factor(spec, dihedral))
            assert direct == oriented
            restored = downscale_grid_exact(direct, oriented_scale_factor(spec, dihedral))
            assert apply_inverse_dihedral_transform(restored, dihedral) == grid


# ----------------------------------------------------------------------------- spec pools

def test_pool_order_identity_first_and_capped_at_twelve():
    # transductARC's documented draw order for a grid that admits every factor.
    assert scale_specs_for_dims(1, 1, max_specs=12) == (
        (1, 1), (2, 2), (2, 1), (1, 2), (3, 3), (3, 1), (1, 3), (3, 2), (2, 3), (4, 4), (4, 1), (1, 4))
    assert scale_specs_for_dims(1, 1, anisotropic=False, max_specs=4) == ((1, 1), (2, 2), (3, 3), (4, 4))
    assert scale_specs_for_dims(16, 16, max_specs=12) == ((1, 1),)
    assert scale_specs_for_dims(10, 2, max_specs=12) == (
        (1, 1), (2, 2), (2, 1), (1, 2), (3, 3), (3, 1), (1, 3), (3, 2), (2, 3), (1, 4), (2, 4), (3, 4))


def test_pool_keeps_every_grid_within_30_in_every_orientation():
    rng = random.Random(3)
    for _ in range(500):
        dims = [rng.randint(1, 30) for _ in range(4)]
        for fr, fc in scale_specs_for_dims(*dims, max_specs=None):
            for rows, cols in ((dims[0], dims[1]), (dims[2], dims[3])):
                assert rows * fr <= 30 and cols * fc <= 30  # transposed: (cols*fc) x (rows*fr), same bounds
    # The output bounds the pool as much as the input.
    assert (2, 2) not in pair_scale_specs({"input": [[1]], "output": [[1] * 16]})


def test_sequence_pool_is_the_target_pool_admitted_by_every_demo():
    target = {"input": [[1] * 2] * 2, "output": [[1] * 2] * 2}   # caps 15 x 15
    wide = {"input": [[1] * 10], "output": [[1] * 10]}            # caps 30 x 3
    tall = {"input": [[1]] * 12, "output": [[1]] * 12}            # caps 2 x 30
    specs = sequence_scale_specs(target, [wide, tall])
    assert specs[0] == (1, 1)
    assert set(specs) == {s for s in pair_scale_specs(target) if s[0] <= 2 and s[1] <= 3}
    assert sequence_scale_specs(target, []) == pair_scale_specs(target)


def test_demo_bounded_decode_pool_matches_the_reference_rules():
    double = [{"input": [[1] * 3] * 2, "output": [[1] * 6] * 4}]  # output = 2x input on both axes
    specs = demo_bounded_scale_specs([[1] * 5] * 5, double)
    # Projected output 10 x 10 caps both axes at 3 (the input alone would allow 6).
    assert specs[0] == (1, 1) and set(specs) == {(r, c) for r in (1, 2, 3) for c in (1, 2, 3)}
    # A demo that admits only identity pins the decode to identity.
    narrow = demo_bounded_scale_specs([[1]], [{"input": [[1] * 20], "output": [[1] * 20]}])
    assert narrow[0] == (1, 1) and len(narrow) == 12 and all(fc == 1 for _, fc in narrow)
    # Without integral projections only the test input bounds the specs.
    odd = [{"input": [[1] * 3] * 3, "output": [[1] * 2] * 2}]
    assert demo_bounded_scale_specs([[1] * 4] * 4, odd) == list(scale_specs_for_dims(4, 4, max_specs=12))
    assert demo_bounded_scale_specs([[1]], double, max_specs=1) == [(1, 1)]


def test_inference_scale_schedule_is_balanced_and_deterministic():
    specs = list(scale_specs_for_dims(1, 1, max_specs=12))
    schedule = inference_scale_schedule(specs, 80, seed=7)
    counts = Counter(schedule)
    assert len(schedule) == 80 and set(counts) == set(specs)
    assert max(counts.values()) - min(counts.values()) <= 1
    assert schedule == inference_scale_schedule(specs, 80, seed=7)
    assert inference_scale_schedule([(1, 1)], 5, seed=0) == [(1, 1)] * 5


# ----------------------------------------------------------------------------- training stream

def test_scale_off_reproduces_the_pre_scale_training_stream(tmp_path, cpu):
    path = write_fixture(tmp_path / "tasks.json")
    for overrides in ({}, {"scale_augmentation": False}):
        _, loader = training_loader(build, training_args(path, **overrides))
        assert stream_digest(loader, 6) == PRE_SCALE_STREAM_DIGEST


def _planned_sequences(dataset, loader, epochs):
    sampler = loader.batch_sampler
    for epoch in range(epochs):
        loader.augmentor.set_epoch(epoch)
        sampler.set_epoch(epoch)
        for index in sampler._balanced_indices()[epoch]:
            example = dataset[index]
            mapping, dihedral = loader.augmentor.select_for_example(example)
            yield example, mapping, dihedral


def test_scale_on_applies_one_factor_to_every_grid_and_keeps_dihedral_colour_demos(tmp_path, cpu):
    path = write_fixture(tmp_path / "tasks.json")
    off_ds, off_loader = training_loader(build, training_args(path))
    on_ds, on_loader = training_loader(build, training_args(path, scale_augmentation=True))
    scales = Counter()
    for (ex_off, map_off, d_off), (ex_on, map_on, d_on) in zip(
            _planned_sequences(off_ds, off_loader, 6), _planned_sequences(on_ds, on_loader, 6)):
        assert (ex_off.task_id, ex_off.pair_index, d_off) == (ex_on.task_id, ex_on.pair_index, d_on)
        assert torch.equal(map_off, map_on)  # colour plan unchanged by scale
        visit = ex_on.augment_epoch_override
        assert off_ds.demonstration_indices(ex_off, visit) == on_ds.demonstration_indices(ex_on, visit)
        plain, plain_sep, _ = off_ds.build_training_sequence(ex_off, d_off, map_off)
        scaled, scaled_sep, positions = on_ds.build_training_sequence(ex_on, d_on, map_on)
        pair = on_ds.pairs_by_task[ex_on.task_id][ex_on.pair_index]
        spec = on_ds.training_scale_factor(ex_on, pair, visit)
        scales[spec] += 1
        demos = [on_ds.pairs_by_task[ex_on.task_id][i] for i in on_ds.demonstration_indices(ex_on, visit)]
        assert spec in sequence_scale_specs(pair, demos)
        # Exactly encode_context with the drawn spec, applied before the dihedral.
        expected, expected_sep = encode_context(demos, pair, d_on, map_on, scale=spec)
        assert torch.equal(scaled, expected) and scaled_sep == expected_sep
        oriented = oriented_scale_factor(spec, d_on)
        scaled_grids, plain_grids = split_grids_from_tokens(scaled.tolist()), split_grids_from_tokens(plain.tolist())
        assert len(scaled_grids) == len(plain_grids) == 2 * (len(demos) + 1)
        for big, small in zip(scaled_grids, plain_grids):
            assert len(big) <= 30 and len(big[0]) <= 30
            assert downscale_grid_exact(big, oriented) == small  # same factor on every grid
        if spec == (1, 1):
            assert torch.equal(scaled, plain) and scaled_sep == plain_sep
        # 3D RoPE positions follow the scaled grid: x = column, y = row, z = grid number.
        cursor = 1
        for z, big in enumerate(scaled_grids, start=1):
            for r, row in enumerate(big):
                for c in range(len(row)):
                    assert positions[cursor].tolist() == [c, r, z]
                    cursor += 1
                cursor += 1  # newline / separator / end token
        assert positions[0].tolist() == [0, 0, 0] and scaled[0] == START_TOKEN_ID
        assert scaled[scaled_sep] == IO_SEPARATOR_TOKEN_ID
    assert scales[(1, 1)] < sum(scales.values()) and len(scales) > 6


def test_training_scale_draw_is_uniform_and_order_independent(tmp_path, cpu):
    path = write_fixture(tmp_path / "tasks.json")
    ds, _ = training_loader(build, training_args(path, scale_augmentation=True))
    example = next(e for e in ds.examples if e.task_id == "t_tiny")
    pair = ds.pairs_by_task["t_tiny"][example.pair_index]
    draws = [ds.training_scale_factor(example, pair, visit) for visit in range(2400)]
    assert draws == [ds.training_scale_factor(example, pair, visit) for visit in reversed(range(2400))][::-1]
    pools = {sequence_scale_specs(pair, [ds.pairs_by_task["t_tiny"][i] for i in ds.demonstration_indices(example, v)])
             for v in range(2400)}
    assert len(pools) == 1  # every grid here is tiny: the full 12-spec pool
    counts = Counter(draws)
    assert set(counts) == set(next(iter(pools))) and len(counts) == 12
    assert min(counts.values()) > 140 and max(counts.values()) < 260  # ~200 each
    big = next(e for e in ds.examples if e.task_id == "t_big")
    assert {ds.training_scale_factor(big, ds.pairs_by_task["t_big"][big.pair_index], v) for v in range(50)} == {(1, 1)}


def test_scale_on_collated_batches_stay_within_capacity(tmp_path, cpu):
    path = write_fixture(tmp_path / "tasks.json")
    ds, loader = training_loader(build, training_args(path, scale_augmentation=True))
    longest = 0
    for epoch in range(4):
        loader.augmentor.set_epoch(epoch)
        loader.batch_sampler.set_epoch(epoch)
        for batch in loader:
            longest = max(longest, int(batch["max_seqlen"]))
            # Every packed sequence starts with <start>, which resets the 3D cursor.
            assert torch.equal(batch["positions_3d"], compute_positions_3d(batch["input_ids"].unsqueeze(0)).squeeze(0))
    assert longest <= ds.max_seq_len


def test_scale_needs_augmented_training_and_turns_itself_off_without_it(tmp_path, cpu):
    path = write_fixture(tmp_path / "tasks.json")
    _, dataset, loader, _, _ = build.build_model_and_data(training_args(path, scale_augmentation=True, enable_aug=False))
    assert dataset.scale_augmentation is False
    assert all(dataset.training_scale_factor(example, {"input": [[1]], "output": [[1]]}, 0) is None
               for example in dataset.examples)


def test_checkpoint_records_scale_settings_only_when_on(tmp_path, cpu):
    import train
    path = write_fixture(tmp_path / "tasks.json")
    for enabled in (False, True):
        model, ds, _, _, _ = build.build_model_and_data(training_args(path, scale_augmentation=enabled))
        out = tmp_path / f"model-{enabled}.pt"
        train.maybe_save_model(model, ds, path, out)
        checkpoint = build.load_checkpoint(out)
        if enabled:
            assert checkpoint["scale_augmentation"] == {"max_specs": 12, "anisotropic": True}
        else:
            assert "scale_augmentation" not in checkpoint


# ----------------------------------------------------------------------------- decoding

def _eval_dataset(path, monkeypatch, scale_views):
    monkeypatch.setattr(build, "resolve_device", lambda _: torch.device("cpu"))
    args = training_args(path, scale_augmentation=True)
    model, train_ds, _, _, _ = build.build_model_and_data(args)
    import train
    checkpoint_path = path.with_name("model.pt")
    train.maybe_save_model(model, train_ds, path, checkpoint_path)
    eval_args = training_args(path, inference_demo_policy="leave_one_out", inference_scale_views=scale_views,
                              scale_augmentation=True)
    _, dataset, _, _, _ = build.build_model_and_data(
        eval_args, checkpoint=build.load_checkpoint(checkpoint_path), is_eval=True, build_dataloader=False)
    assert not dataset.scale_augmentation  # training scale never applies to eval datasets
    return dataset


def test_decode_scale_views_are_demo_bounded_and_invert_to_the_truth(tmp_path, monkeypatch):
    path = write_fixture(tmp_path / "tasks.json")
    plain = evaluate._build_inference_attempts(_eval_dataset(path, monkeypatch, False), "test", None, None, 80, True)
    dataset = _eval_dataset(path, monkeypatch, True)
    attempts = evaluate._build_inference_attempts(dataset, "test", None, None, 80, True)
    assert len(attempts) == len(plain) == 80 * len(fixture_tasks())
    assert all(a.scale_factor == (1, 1) for a in plain)
    truth = {task_id: task["test"][0]["output"] for task_id, task in fixture_tasks().items()}
    results, by_task = [], Counter()
    for a, b in zip(attempts, plain):
        assert (a.task_id, a.dihedral_index, a.color_mapping) == (b.task_id, b.dihedral_index, b.color_mapping)
        task = fixture_tasks()[a.task_id]
        specs = demo_bounded_scale_specs(task["test"][0]["input"], task["train"])
        assert a.scale_factor in specs
        by_task[a.task_id, a.scale_factor] += 1
        if a.scale_factor == (1, 1):
            assert a.prompt == b.prompt
        for grid in split_grids_from_tokens(a.prompt):
            assert len(grid) <= 30 and len(grid[0]) <= 30
        # A perfect model's answer in the augmented frame: colour, upscale, dihedral.
        answer = apply_dihedral_transform(
            upscale_grid(apply_color_permutation_to_grid(truth[a.task_id], a.color_mapping), a.scale_factor),
            a.dihedral_index)
        assert len(answer) <= 30 and len(answer[0]) <= 30  # the demo bound held for this fixture
        results.append(dict(task_id=a.task_id, pair_index=0, output_grid=answer, original_input=a.original_input,
                            color_mapping=a.color_mapping, source_colors=a.source_colors,
                            dihedral_index=a.dihedral_index,
                            **({} if a.scale_factor == (1, 1) else {"scale_factor": list(a.scale_factor)})))
    assert len({spec for (task, spec) in by_task if task == "t_tiny"}) == 12
    assert {spec for (task, spec) in by_task if task == "t_big"} == {(1, 1)}
    for selection in evaluate.run_aaivr_on_results(results):
        assert selection.ranked_candidates[0]["grid"] == truth[selection.task_id]
        assert selection.ranked_candidates[0]["count"] == 80 and selection.discarded_scale_mismatch == 0


def test_decode_inverse_rejects_non_uniform_and_misfitting_scaled_outputs():
    truth = [[1, 2, 3], [4, 5, 6]]
    identity = list(range(14))
    good = apply_dihedral_transform(upscale_grid(truth, (2, 3)), 1)  # transposing dihedral
    ragged_block = [list(row) for row in good]
    ragged_block[0][0] = 9
    results = [
        dict(task_id="t", pair_index=0, output_grid=good, original_input=[[0]], color_mapping=identity,
             source_colors=list(range(10)), dihedral_index=1, scale_factor=[2, 3]),
        dict(task_id="t", pair_index=0, output_grid=ragged_block, original_input=[[0]], color_mapping=identity,
             source_colors=list(range(10)), dihedral_index=1, scale_factor=[2, 3]),
        dict(task_id="t", pair_index=0, output_grid=good[:-1], original_input=[[0]], color_mapping=identity,
             source_colors=list(range(10)), dihedral_index=1, scale_factor=[2, 3]),
        # Undoing with the unoriented factor would be wrong: (2, 3) on a rot90 frame is (3, 2).
        dict(task_id="t", pair_index=0, output_grid=upscale_grid(apply_dihedral_transform(truth, 1), (2, 3)),
             original_input=[[0]], color_mapping=identity, source_colors=list(range(10)), dihedral_index=1,
             scale_factor=[2, 3]),
    ]
    (selection,) = evaluate.run_aaivr_on_results(results)
    assert selection.ranked_candidates == [{"grid": truth, "count": 1}]
    assert selection.num_generated == 4 and selection.num_valid == 1
    assert selection.discarded_scale_mismatch == 3


def test_accuracy_targets_follow_scaled_attempts(tmp_path, monkeypatch):
    path = write_fixture(tmp_path / "tasks.json")
    dataset = _eval_dataset(path, monkeypatch, True)
    attempts = evaluate._build_inference_attempts(dataset, "test", ["t_tiny"], None, 24, True)
    solutions = {"t_tiny": [fixture_tasks()["t_tiny"]["test"][0]["output"]]}
    for item in evaluate._build_accuracy_attempts(dataset, "test", attempts, solutions, dataset.max_seq_len):
        grids = split_grids_from_tokens(item.tokens.tolist())
        assert downscale_grid_exact(grids[-1], oriented_scale_factor(item.inference.scale_factor,
                                                                     item.inference.dihedral_index)) is not None


def test_rescore_scores_each_candidate_in_its_scale_view_frame(tmp_path, monkeypatch):
    path = write_fixture(tmp_path / "tasks.json")
    dataset = _eval_dataset(path, monkeypatch, True)
    attempts = evaluate._build_inference_attempts(dataset, "test", None, None, 80, True)
    assert any(a.scale_factor != (1, 1) for a in attempts)
    truth = {task_id: task["test"][0]["output"] for task_id, task in fixture_tasks().items()}
    candidates = {(task_id, 0): [{"grid": grid, "count": 2}, {"grid": [[1, 2]], "count": 1}]
                  for task_id, grid in truth.items()}
    rescored, too_long = evaluate._build_rescore_attempts(attempts, candidates, dataset.max_seq_len)
    assert too_long == 0 and len(rescored) == 2 * len(attempts)
    # The right candidate is scored on exactly the sequence accuracy-only teacher-forces for that view.
    targets = {id(item.inference): item.tokens for item in evaluate._build_accuracy_attempts(
        dataset, "test", attempts, {t: [g] for t, g in truth.items()}, dataset.max_seq_len)}
    results = []
    for item in rescored:
        a = item.inference
        if item.candidate_index == 0:
            torch.testing.assert_close(item.tokens, targets[id(a)])
        torch.testing.assert_close(item.positions, compute_positions_3d(item.tokens[None]).squeeze(0))
        # Every appended output inverts through the decode's own AAIVR back to its canonical candidate.
        results.append(dict(task_id=a.task_id, pair_index=0, output_grid=split_grids_from_tokens(item.tokens.tolist())[-1],
                            original_input=a.original_input, color_mapping=a.color_mapping,
                            source_colors=a.source_colors, dihedral_index=a.dihedral_index,
                            **({} if a.scale_factor == (1, 1) else {"scale_factor": list(a.scale_factor)})))
    for selection in evaluate.run_aaivr_on_results(results, discard_input_copies=False):
        assert sorted(map(str, (c["grid"] for c in selection.ranked_candidates))) == sorted(
            map(str, (truth[selection.task_id], [[1, 2]])))
        assert all(c["count"] == 80 for c in selection.ranked_candidates)
        assert selection.discarded_scale_mismatch == 0


# ----------------------------------------------------------------------------- launcher

def test_spec_flags_default_off_and_validate(tmp_path):
    plan = launch_run.build_plan({"enable_finetune": False, "defaults": {}}, tmp_path / "spec.json")
    assert plan["pretrain"]["scale_augmentation"] is False and plan["pretrain"]["scale_max_specs"] == 12
    # Evaluations run with plan["shared"] merged under their own settings (run_eval_stage).
    assert plan["shared"]["inference_scale_views"] is False and plan["shared"]["scale_augmentation"] is False
    plan = launch_run.build_plan({"enable_finetune": False, "defaults": {
        "scale_augmentation": True, "inference_scale_views": True}}, tmp_path / "spec.json")
    assert plan["pretrain"]["scale_augmentation"] is True and plan["shared"]["inference_scale_views"] is True
    plan = launch_run.build_plan({"enable_finetune": False, "defaults": {"scale_augmentation": False},
                                  "evals": {"plain": {"checkpoint": "pretrain.final", "inference_scale_views": False}}},
                                 tmp_path / "spec.json")
    assert plan["pretrain"]["scale_augmentation"] is False
    assert plan["evals"]["plain"]["inference_scale_views"] is False
    for bad in ({"scale_augmentation": 1}, {"scale_max_specs": 0}, {"inference_scale_views": "yes"}):
        with pytest.raises(ValueError):
            launch_run.build_plan({"enable_finetune": False, "defaults": bad}, tmp_path / "spec.json")
