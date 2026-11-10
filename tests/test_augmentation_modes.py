"""Behavioral coverage for reversible augmentation policies."""

import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import build
import evaluate
import launch_run
import train
import variations
from common import (
    ARCExampleDataset, CombinedAugmentor, build_augmentor,
    Augmentor, build_unique_input_augments, collate_examples,
    _input_geometry_bytes, _input_grid_bytes,
    build_augmentation_schedule, build_balanced_color_schedules,
    extract_task_output_only_colors, apply_color_permutation_to_grid,
    apply_dihedral_transform,
)


def _task():
    return {"train": [
        {"input": [[0, 1]], "output": [[3, 1]]},
        {"input": [[2, 1]], "output": [[3, 2]]},
    ], "test": [{"input": [[0, 2]], "output": [[8]]}]}


def _dataset(tmp_path, mode="single_grid_prefix"):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"task": _task()}))
    return ARCExampleDataset(path, augmentation_sampling_mode=mode, max_augments=80)


def _assert_restricted(mapping):
    assert int(mapping[3]) == 3
    assert sorted(mapping[:10].tolist()) == list(range(10))
    assert mapping[10:].tolist() == [10, 11, 12, 13]
    assert all(int(mapping[color]) != 3 for color in (0, 1, 2))


def test_fallback_matches_pre_change_training_and_inference_schedules():
    # Captured before the mode changes, including the exact RNG/order behavior.
    keys = [("task", "train", 0), ("task", "train", 1)]
    regular = build_balanced_color_schedules([0, 1, 4], keys, 80, 42)
    generic, dihedrals = build_augmentation_schedule([0, 1, 4], 80, 42)
    inference = evaluate._build_inference_case_schedule(
        source_colors=[0, 1, 4], attempt_count=80, use_aug=True,
        seed=42, task_id="task", pair_index=0,
        augmentation_sampling_mode="full_task_prefix", protected_colors=[4],
    )
    record = {
        "regular": [(list(key), value.tolist()) for key, value in regular.items()],
        "generic": (generic.tolist(), dihedrals),
        "inference": [(item.color_mapping.tolist(), item.dihedral_index) for item in inference],
    }
    assert hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest() == (
        "aa08b999d7895cbf43e74ea9daf8e024b2b650ca2c4158a7e80f4fe8f6f7681b"
    )


def test_protection_uses_task_inputs_and_demonstration_outputs_only():
    task = _task()
    assert extract_task_output_only_colors(task) == [3]
    task["test"][0]["output"] = [[9, 1, 3]]
    assert extract_task_output_only_colors(task) == [3]
    task["test"][0]["input"] = [[3]]
    assert extract_task_output_only_colors(task) == []


def test_regular_training_keeps_output_only_fixed_and_deduplicates_task_inputs(tmp_path):
    dataset = _dataset(tmp_path)
    augmentor = build_augmentor(
        dataset.examples, dataset.task_colors, max_augments=80, seed=42,
        augmentation_sampling_mode=dataset.augmentation_sampling_mode,
        task_output_only_colors=dataset.task_output_only_colors,
    )
    seen = {_input_grid_bytes(pair["input"]) for pairs in (_task()["train"], _task()["test"]) for pair in pairs}
    for key, augments in augmentor.augments_by_key.items():
        for mapping in augments.color_maps:
            _assert_restricted(mapping)
        pair = _task()[key[1]][key[2]]
        _assert_unique_additions(augments, pair["input"], seen)
        assert 1 <= len(augments.dihedral_indices) < 80
    assert len(seen) == 144  # These three inputs share the same 144-element orbit.


def _assert_unique_additions(augments, grid, seen):
    assert augments.dihedral_indices[0] == 0
    assert augments.color_maps[augments.color_map_indices[0]].tolist() == list(range(14))
    for color_index, dihedral in zip(augments.color_map_indices[1:], augments.dihedral_indices[1:]):
        transformed = apply_dihedral_transform(
            apply_color_permutation_to_grid(grid, augments.color_maps[color_index]), dihedral,
        )
        key = _input_grid_bytes(transformed)
        assert key not in seen
        seen.add(key)


def test_inference_protects_colors_without_reading_test_answers_and_keeps_geometry(tmp_path):
    restricted = _dataset(tmp_path)
    original = ARCExampleDataset(restricted.source_path, max_augments=80)
    attempts = evaluate._build_inference_attempts(restricted, "test", None, None, 80, True)
    fallback = evaluate._build_inference_attempts(original, "test", None, None, 80, True)
    assert len(attempts) == len(fallback) == 80
    assert Counter(item.dihedral_index for item in attempts) == Counter({index: 10 for index in range(8)})
    assert len({tuple(item.prompt) for item in attempts}) == 80
    for item in attempts:
        _assert_restricted(torch.tensor(item.color_mapping))
    task = _task()
    task["test"][0]["output"] = [[9, 8, 7]]
    restricted.source_path.write_text(json.dumps({"task": task}))
    changed = ARCExampleDataset(restricted.source_path, augmentation_sampling_mode="single_grid_prefix")
    repeated = evaluate._build_inference_attempts(changed, "test", None, None, 80, True)
    assert [(item.prompt, item.color_mapping) for item in repeated] == [
        (item.prompt, item.color_mapping) for item in attempts
    ]
    results = []
    for item in attempts:
        output = apply_dihedral_transform(
            apply_color_permutation_to_grid([[3, 1]], item.color_mapping), item.dihedral_index,
        )
        results.append({
            "task_id": item.task_id, "pair_index": item.pair_index, "output_grid": output,
            "original_input": item.original_input, "color_mapping": item.color_mapping,
            "source_colors": item.source_colors, "dihedral_index": item.dihedral_index,
        })
    votes = evaluate.run_aaivr_on_results(results)
    assert votes[0].selected_outputs == [[[3, 1]]]
    assert votes[0].ranked_candidates[0]["count"] == 80


@pytest.mark.parametrize("protected", [[], [3], list(range(1, 10)), list(range(10))])
def test_inference_handles_tiny_and_empty_movable_palettes(protected):
    schedule = evaluate._build_inference_case_schedule(
        source_colors=list(range(10)), attempt_count=80, use_aug=True,
        seed=42, task_id="edge-case", pair_index=0,
        augmentation_sampling_mode="single_grid_prefix", protected_colors=protected,
        input_grid=[[0]],
    )
    assert len(schedule) == max(1, 10 - len(protected))
    for item in schedule:
        assert sorted(item.color_mapping[:10].tolist()) == list(range(10))
        assert all(int(item.color_mapping[color]) == color for color in protected)


@pytest.mark.parametrize("source", ["rearc", "nvarc"])
def test_synthetic_training_reserves_output_only_colors(tmp_path, source):
    dataset = _dataset(tmp_path)
    example = dataset.examples[0]
    setattr(example, "is_" + source, True)
    augmentor = CombinedAugmentor(
        None, rearc_seed=42, task_colors=dataset.task_colors, max_augments=80,
        augmentation_sampling_mode="single_grid_prefix",
        task_output_only_colors=dataset.task_output_only_colors,
    )
    for epoch in range(30):
        augmentor.set_epoch(epoch)
        mapping, geometry = augmentor.select_for_example(example)
        _assert_restricted(mapping)
        assert 0 <= geometry < 8


def test_variation_augmentation_uses_its_own_known_pairs(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path)
    monkeypatch.setattr(dataset, "_pairs_for_path", lambda path: _task()["train"], raising=False)
    augmentor = variations.VariationAugmentor(dataset, seed=42, max_augments=80)
    example = dataset.examples[0]
    example.variation_path = "variant.json"
    example.variation_colors = (0, 1, 2, 3)
    example.variation_output_only_colors = (3,)
    for epoch in range(30):
        example.augment_epoch_override = epoch
        mapping, _ = augmentor.select_for_example(example)
        _assert_restricted(mapping)
    assert len(augmentor.unique_pools) == 1
    seen = {_input_grid_bytes(pair["input"]) for pair in _task()["train"]}
    for key, augments in augmentor.unique_pools[example.variation_path].augments_by_key.items():
        _assert_unique_additions(augments, _task()["train"][key[2]]["input"], seen)


def test_variation_can_skip_duplicate_checks_while_originals_keep_them(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path)
    dataset.variation_index = {"sampling": {"deduplicate_variation_inputs": False}}
    augmentor = variations.VariationAugmentor(dataset, seed=42, max_augments=80)
    # Normal tasks still use the existing unique-input pools.
    seen = {_input_grid_bytes(pair["input"]) for pairs in (_task()["train"], _task()["test"]) for pair in pairs}
    for key, augments in augmentor.regular.augments_by_key.items():
        _assert_unique_additions(augments, _task()[key[1]][key[2]]["input"], seen)

    def must_not_build_pool(*args, **kwargs):
        pytest.fail("Variations must not build a duplicate-checked augmentation pool")

    monkeypatch.setattr(variations, "build_unique_input_augments", must_not_build_pool)
    monkeypatch.setattr(dataset, "_pairs_for_path", must_not_build_pool, raising=False)
    example = dataset.examples[0]
    example.variation_path = "variant.json"
    example.variation_colors = (0, 1, 2, 3)
    example.variation_output_only_colors = (3,)
    selected = []
    for epoch in range(80):
        example.augment_epoch_override = epoch
        mapping, geometry = augmentor.select_for_example(example)
        _assert_restricted(mapping)
        assert 0 <= geometry < 8
        selected.append((tuple(mapping.tolist()), geometry))
        repeated_mapping, repeated_geometry = augmentor.select_for_example(example)
        assert (tuple(repeated_mapping.tolist()), repeated_geometry) == selected[-1]
    assert len(set(selected)) > 1
    assert not augmentor.unique_pools


def test_checkpoint_inherits_mode_and_explicit_fallback_is_available(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path)
    monkeypatch.setattr(build, "resolve_device", lambda _: torch.device("cpu"))
    args = SimpleNamespace(
        seed=42, data_path=dataset.source_path, checkpoint_path=None,
        enable_aug=True, max_augments=80, augmentation_sampling_mode="single_grid_prefix",
        d_model=24, n_heads=4, d_ff=48, n_layers=1, num_task_embedding_latents=2,
        dropout=0, embedding_dropout=0, attention_dropout=0, n_predict=1,
    )
    model, actual, _, _, _ = build.build_model_and_data(args, build_dataloader=False)
    path = tmp_path / "model.pt"
    train.maybe_save_model(model, actual, dataset.source_path, path)
    checkpoint = build.load_checkpoint(path)
    assert checkpoint["augmentation_sampling_mode"] == "single_grid_prefix"
    args.augmentation_sampling_mode = None
    assert build.build_model_and_data(args, checkpoint=checkpoint, build_dataloader=False)[1].augmentation_sampling_mode == "single_grid_prefix"
    args.augmentation_sampling_mode = "full_task_prefix"
    assert build.build_model_and_data(args, checkpoint=checkpoint, build_dataloader=False)[1].augmentation_sampling_mode == "full_task_prefix"
    checkpoint.pop("augmentation_sampling_mode")
    args.augmentation_sampling_mode = None
    assert build.build_model_and_data(args, checkpoint=checkpoint, build_dataloader=False)[1].augmentation_sampling_mode == "full_task_prefix"


def test_spec_policy_propagates_to_training_and_evaluation():
    plan = launch_run.build_plan({
        "enable_finetune": False,
        "defaults": {"augmentation_sampling_mode": "single_grid_prefix"},
    }, Path("spec.json"))
    assert plan["pretrain"]["augmentation_sampling_mode"] == "single_grid_prefix"
    assert all(stage["augmentation_sampling_mode"] == "single_grid_prefix" for stage in plan["evals"].values())
    with pytest.raises(ValueError, match="augmentation_sampling_mode"):
        launch_run.build_plan({"defaults": {"augmentation_sampling_mode": "typo"}}, Path("spec.json"))


@pytest.mark.parametrize("grid", [[[1]], [[1, 1], [1, 1]], [[1, 2, 1], [2, 1, 2]]])
def test_unique_pool_underfills_and_preserves_duplicate_originals(grid):
    inputs = {("task", "train", index): grid for index in range(2)}
    colors = sorted({color for row in grid for color in row})
    pools = build_unique_input_augments(inputs, colors, max_augments=80, seed=42)
    seen = {_input_grid_bytes(grid)}
    for augments in pools.values():
        _assert_unique_additions(augments, grid, seen)
    assert sum(len(pool.dihedral_indices) - 1 for pool in pools.values()) == len(seen) - 1
    if colors == [1]:
        assert len(seen) == 10  # Nine recolorings, regardless of eight identical rotations.


def test_changes_to_colors_absent_from_input_do_not_create_more_attempts():
    key = ("task", "test", 0)
    pools = build_unique_input_augments({key: [[1]]}, list(range(10)), max_augments=80, seed=42)
    assert len(pools[key].dihedral_indices) == 10


@pytest.mark.parametrize("grid", [[[1, 2, 3]], [[1], [2], [3]], [[1, 2, 3], [4, 5, 6]]])
def test_fast_geometry_keys_match_serialized_grid_transforms(grid):
    assert _input_geometry_bytes(grid) == [
        _input_grid_bytes(apply_dihedral_transform(grid, index)) for index in range(8)
    ]
    assert _input_grid_bytes([[1, 2]]) != _input_grid_bytes([[1], [2]])


def test_known_originals_are_reserved_even_outside_training_split(tmp_path):
    dataset = _dataset(tmp_path)
    augmentor = build_augmentor(
        dataset.iter_examples(split="train"), dataset.task_colors, max_augments=80, seed=42,
        augmentation_sampling_mode="single_grid_prefix",
        task_output_only_colors=dataset.task_output_only_colors,
        task_inputs={"task": [pair["input"] for pair in _task()["train"] + _task()["test"]]},
    )
    seen = {_input_grid_bytes(pair["input"]) for pair in _task()["train"] + _task()["test"]}
    for key, augments in augmentor.augments_by_key.items():
        _assert_unique_additions(augments, _task()["train"][key[2]]["input"], seen)


def test_inference_filters_duplicates_across_multiple_test_pairs(tmp_path):
    task = _task()
    task["test"] = [{"input": [[0]]}, {"input": [[0]]}]
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"task": task}))
    dataset = ARCExampleDataset(path, augmentation_sampling_mode="single_grid_prefix")
    attempts = evaluate._build_inference_attempts(dataset, "test", None, None, 80, True)
    # Protected 3 leaves only nine inputs. Preserve both originals and split
    # the other eight distinct recolorings between the two test pairs.
    assert len(attempts) == 10
    # The target-only augmentation schedule is unchanged by the new demos.
    assert len({tuple(item.prompt[-2:]) for item in attempts}) == 9
    assert sum(item.prompt[-2:] == [0, 12] for item in attempts) == 2
    selected = evaluate._build_inference_attempts(dataset, "test", None, 0, 80, True)
    assert [item.prompt for item in selected] == [item.prompt for item in attempts if item.pair_index == 0]


def test_exhausted_geometry_pool_collates_selected_transform(tmp_path):
    dataset = _dataset(tmp_path)
    example = dataset.examples[0]
    key = (example.task_id, example.split, example.pair_index)
    pools = build_unique_input_augments({key: [[0, 1]]}, [0, 1], max_augments=1, seed=42)
    augmentor = Augmentor(pools)
    example.preselected_dihedral_index = 3
    batch = collate_examples([example], dataset, augment_selector=augmentor.select_for_example)
    tokens, _, _ = dataset.build_training_sequence(example, 0, None)
    assert batch["input_ids"].tolist() == tokens.tolist()
    assert batch["dihedral_ids"].tolist() == [0]


@pytest.mark.parametrize("source", ["rearc", "nvarc"])
def test_lazy_training_pools_are_shared_across_pairs_and_cached(tmp_path, monkeypatch, source):
    dataset = _dataset(tmp_path)
    augmentor = CombinedAugmentor(
        None, dataset=dataset, rearc_seed=42, task_colors=dataset.task_colors, max_augments=80,
        augmentation_sampling_mode="single_grid_prefix", task_output_only_colors=dataset.task_output_only_colors,
    )
    for example in dataset.examples[:2]:
        setattr(example, "is_" + source, True)
        augmentor.select_for_example(example)
    pools = augmentor._unique_augmentors["task"].augments_by_key
    seen = {_input_grid_bytes(pair["input"]) for pair in _task()["train"] + _task()["test"]}
    for key, pool in pools.items():
        _assert_unique_additions(pool, _task()["train"][key[2]]["input"], seen)
    import common
    monkeypatch.setattr(common, "build_unique_input_augments", lambda *args, **kwargs: pytest.fail("Rebuilt cached pool"))
    for epoch in range(200):
        augmentor.set_epoch(epoch)
        augmentor.select_for_example(dataset.examples[0])


def test_unique_pool_epoch_cycles_cover_pool_before_repeating_and_are_reproducible():
    from common import _select_augment_index
    key = ("task", "train", 0)
    first = build_unique_input_augments({key: [[0, 1], [1, 2]]}, [0, 1, 2], max_augments=80, seed=42)[key]
    second = build_unique_input_augments({key: [[0, 1], [1, 2]]}, [0, 1, 2], max_augments=80, seed=42)[key]
    for epoch in range(240):
        assert _select_augment_index(first, first.order_seed, epoch) == _select_augment_index(second, second.order_seed, epoch)
    for start in range(0, 240, 80):
        assert len({_select_augment_index(first, first.order_seed, epoch) for epoch in range(start, start + 80)}) == 80


def test_training_honors_balanced_geometry_requests_when_pool_contains_them(tmp_path):
    from dataclasses import replace
    dataset = _dataset(tmp_path)
    example = dataset.examples[0]
    augmentor = build_augmentor(
        dataset.examples, dataset.task_colors, max_augments=80, seed=42,
        augmentation_sampling_mode="single_grid_prefix", task_output_only_colors=dataset.task_output_only_colors,
    )
    selected = []
    for occurrence in range(16):
        for geometry in range(8):
            requested = replace(example, preselected_dihedral_index=geometry,
                                dihedral_occurrence_override=occurrence,
                                augment_epoch_override=occurrence * 8 + geometry)
            _, actual = augmentor.select_for_example(requested)
            assert actual == geometry
            selected.append(actual)
    assert Counter(selected) == Counter({index: 16 for index in range(8)})
