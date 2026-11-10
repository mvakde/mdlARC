"""ReARC training under context_palette: sibling demos per visit, per-visit palettes, launcher knob."""
import copy
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
import build
import launch_run
from common import ARCExampleDataset, LazyTrainBatchSampler, collate_examples, encode_context
from context_palette import ContextPaletteAugmentor, grid_colors
from scale_stream_fixture import training_args, training_loader, write_fixture


def _pair(color, rows=2):
    return {"input": [[0, color]] * rows, "output": [[color, color]] * rows}


def _write(tmp_path, siblings=8):
    regular = {"reg": {"train": [
        {"input": [[0, 1]], "output": [[3, 1]]}, {"input": [[0, 2]], "output": [[3, 2]]},
        {"input": [[0, 1]], "output": [[7, 1]]}], "test": [{"input": [[0, 1]], "output": [[7, 3]]}]}}
    rng = random.Random(1)
    # Two original pairs plus generated siblings; the last pair's output paints with 8, a
    # colour its input lacks, so 8 is pinned whenever no shown demo exhibits it.
    rearc = {"re": {"train": [_pair(1), _pair(2)] + [_pair(rng.choice([1, 2, 3, 4, 5])) for _ in range(siblings - 1)]
                    + [{"input": [[0, 1], [1, 0]], "output": [[8, 8], [8, 8]]}], "test": []}}
    tasks, rearc_path = tmp_path / "tasks.json", tmp_path / "rearc.json"
    tasks.write_text(json.dumps(regular))
    rearc_path.write_text(json.dumps(rearc))
    return tasks, rearc_path


def _dataset(tmp_path, **kwargs):
    tasks, rearc_path = _write(tmp_path)
    values = dict(seed=42, max_augments=80, augmentation_sampling_mode="context_palette", attention_pattern="pair_group",
                  training_demo_schedule="half_or_more", inference_demo_policy="leave_one_out",
                  rearc_pairs_per_epoch=3, rearc_max_demos=4)
    values.update(kwargs)
    return ARCExampleDataset(tasks, rearc_json_path=rearc_path, **values)


def _sampler(ds, epochs):
    return LazyTrainBatchSampler(
        regular_train_indices_by_example_id=ds.regular_train_indices_by_example_id,
        rearc_pair_counts_by_example_id=ds.rearc_pair_counts_by_example_id, nvarc_pair_counts_by_example_id={},
        task_ids=ds.task_ids, batch_size=2, seed=42, balance_dihedrals=True, num_epochs=epochs,
        rearc_pairs_per_task=ds.rearc_pairs_per_epoch)


def test_rearc_targets_get_sibling_demos_and_context_palettes_from_the_sampler_plan(tmp_path):
    ds = _dataset(tmp_path)
    assert ds.has_rearc_train_sampling and len(ds.rearc_pairs_by_task["re"]) == 10
    s = _sampler(ds, epochs=20)
    before = copy.deepcopy(s._balanced_indices())
    aug = ContextPaletteAugmentor(ds, seed=42, max_augments=80)
    aug.prepare_training_schedule(s)
    assert before == s._balanced_indices()
    replay = ContextPaletteAugmentor(ds, seed=42, max_augments=80)
    replay.prepare_training_schedule(s)
    pairs = ds.pairs_by_task["re"]
    assert set(ds.task_colors["re"]) >= {0, 1, 2, 3, 4, 5, 8}
    seen_rearc, demo_sets, eight_pinned, onto_unseen = 0, {}, 0, 0
    rng, torch_rng = random.getstate(), torch.get_rng_state()
    for epoch in before:
        for item in epoch:
            ex = ds[item]
            mapping, geometry = aug.select_for_example(ex)
            assert geometry == item[-2]
            again, geometry_again = replay.select_for_example(ds[item])
            assert geometry_again == geometry and torch.equal(again, mapping)
            if item[0] != "rearc":
                continue
            seen_rearc += 1
            assert ex.is_rearc and ex.task_id == "re" and ex.split == "train"
            visit = item[3]
            demos = ds.demonstration_indices(ex, visit)
            assert 2 <= len(demos) <= 4 and ex.pair_index not in demos and len(set(demos)) == len(demos)
            assert list(demos) == sorted(demos)
            demo_sets.setdefault(ex.pair_index, set()).add(demos)
            tokens, sep, _ = ds.build_training_sequence(ex, geometry, mapping)
            expected, expected_sep = encode_context([pairs[i] for i in demos], pairs[ex.pair_index], geometry, mapping)
            assert torch.equal(tokens, expected) and sep == expected_sep
            # Only colours that appear solely in the target output stay fixed; every other
            # label, seen in the sequence or not, takes part in one bijection of the ten.
            visible = grid_colors(pairs[ex.pair_index]["input"])
            for i in demos:
                visible |= grid_colors(pairs[i]["input"]) | grid_colors(pairs[i]["output"])
            pinned = grid_colors(pairs[ex.pair_index]["output"]) - visible
            sequence_palette = visible | grid_colors(pairs[ex.pair_index]["output"])
            assert sorted(mapping[:10].tolist()) == list(range(10)) and mapping[10:].tolist() == [10, 11, 12, 13]
            for color in pinned:
                assert int(mapping[color]) == color
            eight_pinned += 8 in pinned
            onto_unseen += any(int(mapping[color]) not in sequence_palette for color in visible)
    assert seen_rearc == 20 * 3 and eight_pinned > 0
    # The pool's other pairs do not shrink the palette: visible colours land on unseen labels.
    assert onto_unseen > seen_rearc // 2
    # Different visits of one pair draw different sibling sets.
    assert any(len(sets) > 1 for sets in demo_sets.values())
    assert random.getstate() == rng and torch.equal(torch_rng, torch.get_rng_state())


def test_collate_accepts_a_mixed_regular_and_rearc_batch(tmp_path):
    ds = _dataset(tmp_path)
    s = _sampler(ds, epochs=2)
    aug = ContextPaletteAugmentor(ds, seed=42, max_augments=80)
    aug.prepare_training_schedule(s)
    plan = s._balanced_indices()[0]
    regular = next(item for item in plan if item[0] == "regular")
    rearc = next(item for item in plan if item[0] == "rearc")
    batch = collate_examples([ds[regular], ds[rearc]], ds, augment_selector=aug.select_for_example)
    assert batch["task_ids"] == ["reg", "re"]
    assert batch["dihedral_ids"].tolist() == [regular[-2], rearc[-2]]
    assert batch["cu_seqlens"].numel() == 3 and batch["input_ids"].numel() == int(batch["cu_seqlens"][-1])


def test_rearc_demo_count_follows_half_or_more_over_rearc_max_demos(tmp_path):
    for max_demos, allowed in ((1, {1}), (2, {2}), (3, {2, 3}), (4, {2, 3, 4}), (6, {3, 4, 5, 6})):
        ds = _dataset(tmp_path, rearc_max_demos=max_demos)
        example = ds[("rearc", ds.task_id_to_example_id["re"], 0, 0)]
        counts = {len(ds.demonstration_indices(example, visit)) for visit in range(40)}
        assert counts <= allowed and (max_demos <= 2 or len(counts) > 1)
    with pytest.raises(ValueError, match="rearc_max_demos"):
        _dataset(tmp_path, rearc_max_demos=0)


def test_regular_targets_keep_their_balanced_palettes_beside_rearc(tmp_path):
    from test_context_palette import assert_balanced
    from context_palette import protection_schedule
    ds = _dataset(tmp_path)
    s = _sampler(ds, epochs=80)
    aug = ContextPaletteAugmentor(ds, seed=42, max_augments=80)
    aug.prepare_training_schedule(s)
    by_target = {}
    for epoch in s._balanced_indices():
        for item in epoch:
            if item[0] == "regular":
                by_target.setdefault(item[1], []).append(aug.select_for_example(ds[item])[0])
    assert by_target
    for index, maps in by_target.items():
        assert_balanced(maps, protection_schedule(ds, ds.examples[index], 80))


def test_nvarc_and_variation_sampling_still_rejected():
    for flags in ({"has_nvarc_train_sampling": True}, {"has_variation_sampling": True}):
        stub = SimpleNamespace(**{"has_rearc_train_sampling": True, "has_nvarc_train_sampling": False, **flags})
        with pytest.raises(ValueError, match="regular ARC, ReARC and family-sampled"):
            ContextPaletteAugmentor(stub, seed=42, max_augments=80)


def test_builder_trains_rearc_with_scale_on_regular_tasks_only(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "resolve_device", lambda _: torch.device("cpu"))
    path = write_fixture(tmp_path / "tasks.json")
    rearc_path = tmp_path / "rearc.json"
    rng = random.Random(3)
    rearc_path.write_text(json.dumps({"re_task": {"train": [_pair(rng.choice([1, 2, 3])) for _ in range(12)], "test": []}}))
    dataset, loader = training_loader(build, training_args(
        path, rearc_data_path=rearc_path, rearc_pairs_per_epoch=4, rearc_max_demos=3, scale_augmentation=True))
    assert dataset.has_rearc_train_sampling and dataset.scale_augmentation is True and dataset.rearc_max_demos == 3
    loader.augmentor.set_epoch(0)
    loader.batch_sampler.set_epoch(0)
    task_ids = [task_id for batch in loader for task_id in batch["task_ids"]]
    assert task_ids.count("re_task") == 4 and len(set(task_ids) - {"re_task"}) >= 3
    example = dataset[("rearc", dataset.task_id_to_example_id["re_task"], 0, 0)]
    assert dataset.training_scale_factor(example, dataset.pairs_by_task["re_task"][0], 0) is None
    assert len(dataset.demonstration_indices(example, 0)) in (2, 3)


def test_launcher_accepts_and_validates_rearc_knobs(tmp_path):
    plan = launch_run.build_plan({"enable_finetune": False, "defaults": {"rearc_max_demos": 3, "rearc_pairs_per_epoch": 5}},
                                 tmp_path / "spec.json")
    assert plan["pretrain"]["rearc_max_demos"] == 3 and plan["pretrain"]["rearc_pairs_per_epoch"] == 5
    assert launch_run.stage_summary(plan["pretrain"])["rearc_max_demos"] == 3
    for bad in ({"rearc_max_demos": 0}, {"rearc_max_demos": True}, {"rearc_pairs_per_epoch": -1}, {"rearc_max_demos": 2.5}):
        with pytest.raises(ValueError, match="rearc"):
            launch_run.build_plan({"enable_finetune": False, "defaults": bad}, tmp_path / "spec.json")
