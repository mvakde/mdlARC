"""Balanced capped demo contexts (max_training_demos / inference_max_demos), next to the uncapped default."""
from collections import Counter
import json
import random
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common import ARCExampleDataset, encode_context
from demo_schedule import MAX_CONTEXT_DEMOS, capped_half_or_more_demonstrations, half_or_more_demonstrations
from pair_identity import balanced_leave_one_out, leave_one_out


@pytest.mark.parametrize('count', [0, 1, 2, 3, 7, 11])
def test_capped_leave_one_out_balances_omissions(count):
    totals = Counter({i: 0 for i in range(count)})
    for visit in range(80):
        visible = balanced_leave_one_out(range(count), visit=visit, seed=71)
        assert len(visible) == max(0, min(6, count - 1))
        assert visible == balanced_leave_one_out(range(count), visit=visit, seed=71)
        totals.update(set(range(count)) - set(visible))
        if totals:
            assert max(totals.values()) - min(totals.values()) <= 1


@pytest.mark.parametrize('count', [0, 1, 2, 3, 4, 6, 7, 9, 11, 12, 21, 100])
def test_capped_schedule_balances_sizes_and_demo_use_at_every_prefix(count):
    sizes = Counter()
    totals = Counter({i: 0 for i in range(count)})
    upper = min(MAX_CONTEXT_DEMOS, count)
    lower = min(upper, max(2, (count + 1) // 2))
    for visit in range(180):
        subset = capped_half_or_more_demonstrations(range(count), visit=visit, seed=41)
        k = len(subset); sizes[k] += 1
        assert lower <= k <= upper
        assert len(set(subset)) == k
        totals.update(subset)
        if totals:
            assert max(totals.values()) - min(totals.values()) <= 1
    assert max(sizes.values()) - min(sizes.values()) <= 1
    assert set(sizes) == set(range(lower, upper + 1))


def test_uncapped_default_is_unchanged(tmp_path):
    pairs = [{'input': [[i % 10]], 'output': [[(i + 1) % 10]]} for i in range(9)]
    path = tmp_path / 'tasks.json'
    path.write_text(json.dumps({'puzzle': {'train': pairs, 'test': [{'input': [[9]], 'output': [[8]]}]}}))
    ds = ARCExampleDataset(path, seed=41, attention_pattern='pair_group', training_demo_schedule='half_or_more',
                           inference_demo_policy='leave_one_out')
    assert not ds.capped_training_demos and not ds.capped_inference_demos
    for ex in ds.examples:
        for visit in range(10):
            visible = ds.demonstration_indices(ex, visit)
            if ex.split == 'train':
                eligible = [i for i in range(9) if i != ex.pair_index]
                assert visible == half_or_more_demonstrations(eligible, visit=visit, seed=ds_seed(ds, ex))
            else:
                assert len(visible) == 8 and visible == leave_one_out(
                    list(range(9)), visit=visit, seed=__import__('pair_identity').schedule_seed(
                        ds.seed, ex.task_id, ex.split, ex.pair_index))


def ds_seed(ds, ex):
    from common import _derive_order_seed
    return _derive_order_seed(ds.seed, (ex.variation_path or ex.task_id, 'half-or-more-demos', ex.pair_index))


@pytest.mark.parametrize('count', [1, 2, 3, 4, 6, 7, 8, 10, 13, 21])
@pytest.mark.parametrize('schedule', ['all', 'half_or_more', 'two'])
def test_training_balances_within_epoch_and_each_target_across_epochs(tmp_path, count, schedule):
    pairs = [{'input': [[i % 10]], 'output': [[(i + 1) % 10]]} for i in range(count)]
    path = tmp_path / 'tasks.json'
    path.write_text(json.dumps({'puzzle': {'train': pairs}}))
    ds = ARCExampleDataset(path, splits=('train',), seed=41, attention_pattern='pair_group',
                           training_demo_schedule=schedule, max_training_demos=6)
    histories = {ex.pair_index: Counter({i: 0 for i in range(count) if i != ex.pair_index})
                 for ex in ds.examples}
    rng_state, torch_state = random.getstate(), torch.get_rng_state()
    saved = []
    for visit in range(80):
        epoch = Counter({i: 0 for i in range(count)})
        for ex in ds.examples:
            visible = ds.demonstration_indices(ex, visit)
            assert ex.pair_index not in visible
            assert len(visible) == len(set(visible)) <= 6
            if schedule == 'two':
                assert len(visible) == min(2, count - 1)
            epoch.update(visible)
            history = histories[ex.pair_index]
            history.update(visible)
            if history:
                assert max(history.values()) - min(history.values()) <= 1
            saved.append((ex, visit, visible))
        assert len(set(epoch.values())) == 1
    for ex, visit, visible in reversed(saved):
        assert ds.demonstration_indices(ex, visit) == visible
    ds.seed += 1
    if count > 7 or (schedule == 'half_or_more' and count > 3) or (schedule == 'two' and count > 3):
        assert any(ds.demonstration_indices(ex, visit) != visible for ex, visit, visible in saved)
    assert random.getstate() == rng_state
    assert torch.equal(torch_state, torch.get_rng_state())


@pytest.mark.parametrize('count', [0, 1, 2, 5, 6, 7, 8, 10, 21])
@pytest.mark.parametrize('policy', ['all', 'leave_one_out', 'two'])
def test_inference_caps_after_loo_and_balances_all_eighty_prompts(tmp_path, count, policy):
    import evaluate
    pairs = [{'input': [[i % 10]], 'output': [[(i + 1) % 10]]} for i in range(count)]
    path = tmp_path / 'tasks.json'
    path.write_text(json.dumps({'puzzle': {'train': pairs, 'test': [{'input': [[9]], 'output': [[8]]}]}}))
    ds = ARCExampleDataset(path, seed=41, attention_pattern='pair_group', inference_max_demos=6,
        inference_demo_policy=policy, augmentation_sampling_mode='context_palette')
    ex = next(ds.iter_examples(split='test'))
    attempts = evaluate._build_inference_attempts(ds, 'test', None, None, 80, True)
    assert len(attempts) == 80
    totals = Counter({i: 0 for i in range(count)})
    for visit, attempt in enumerate(attempts):
        visible = ds.demonstration_indices(ex, visit)
        expected_count = min(2, count) if policy == 'two' else max(0, min(6, count - int(policy == 'leave_one_out')))
        assert len(visible) == expected_count
        totals.update(visible)
        if totals:
            assert max(totals.values()) - min(totals.values()) <= 1
        expected, sep = encode_context([pairs[i] for i in visible], {'input': [[9]]},
            attempt.dihedral_index, torch.tensor(attempt.color_mapping))
        assert attempt.prompt == expected.tolist() and len(attempt.prompt) == sep + 1


def test_training_cap_reaches_built_model_dataset_and_checkpoint_selects_capped_decode(tmp_path, monkeypatch):
    import build
    pairs = [{'input': [[i % 10]], 'output': [[(i + 1) % 10]]} for i in range(8)]
    path = tmp_path / 'tasks.json'
    path.write_text(json.dumps({'puzzle': {'train': pairs}}))
    monkeypatch.setattr(build, 'resolve_device', lambda _: torch.device('cpu'))
    args = SimpleNamespace(seed=42, device='cpu', checkpoint_path=None, data_path=path,
        attention_pattern='pair_group', training_demo_schedule='half_or_more', max_training_demos=3,
        batch_size=2, enable_aug=False, max_augments=1, d_model=24, n_heads=4,
        d_ff=48, n_layers=1, num_task_embedding_latents=2, n_predict=1)
    _, ds, _, _, _ = build.build_model_and_data(args, build_dataloader=False)
    for visit in range(20):
        counts = Counter({i: 0 for i in range(8)})
        for ex in ds.examples:
            shown = ds.demonstration_indices(ex, visit)
            assert len(shown) == 3 and ex.pair_index not in shown
            counts.update(shown)
        assert len(set(counts.values())) == 1
    # Decoding defaults: capped for capped-context checkpoints, uncapped otherwise; explicit wins.
    capped = {"demo_selection": {"training": "balanced_capped", "max_training_demos": 3}}
    family = {"task_variations": {"sampling": {"unit": "family"}}}
    assert build._resolve_inference_max_demos(SimpleNamespace(), capped) == 6
    assert build._resolve_inference_max_demos(SimpleNamespace(), family) == 6
    assert build._resolve_inference_max_demos(SimpleNamespace(), {"task_ids": []}) is None
    assert build._resolve_inference_max_demos(SimpleNamespace(inference_max_demos=4), {}) == 4


@pytest.mark.parametrize('count', [1, 2, 6])
@pytest.mark.parametrize('is_eval', [False, True])
def test_two_demo_mode_builds_training_and_both_evaluation_splits(tmp_path, monkeypatch, count, is_eval):
    import build
    pairs = [{'input': [[i]], 'output': [[i + 1]]} for i in range(count)]
    path = tmp_path / 'tasks.json'
    path.write_text(json.dumps({'puzzle': {'train': pairs, 'test': [{'input': [[9]]}]}}))
    monkeypatch.setattr(build, 'resolve_device', lambda _: torch.device('cpu'))
    args = SimpleNamespace(seed=42, device='cpu', checkpoint_path=None, data_path=path,
        attention_pattern='pair_group', training_demo_schedule='two', inference_demo_policy='two',
        batch_size=2, enable_aug=False, max_augments=1, d_model=24, n_heads=4,
        d_ff=48, n_layers=1, num_task_embedding_latents=2, n_predict=1)
    _, ds, _, _, _ = build.build_model_and_data(args, is_eval=is_eval,
        eval_splits=('train', 'test'), build_dataloader=False)
    for ex in ds.examples:
        for visit in range(8):
            visible = ds.demonstration_indices(ex, visit)
            assert len(visible) == min(2, count - int(ex.split == 'train'))
            if ex.split == 'train':
                assert ex.pair_index not in visible
