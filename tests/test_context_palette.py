"""Safety, balance, baseline schedules, worker replay and inference inversion."""
import copy
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
import evaluate
import build
import launch_run
from common import ARCExampleDataset, LazyTrainBatchSampler, collate_examples, create_dataloader
from common import encode_context, apply_color_permutation_to_grid, apply_dihedral_transform
from context_palette import ContextPaletteAugmentor, color_schedule, protection_schedule


def dataset(tmp_path, **kwargs):
    path = tmp_path / 'tasks.json'
    # Output-only 3 is witnessed repeatedly. Color 7 has only one demo witness;
    # it is protected on that omission. Label 8 is unused and remains legal.
    task = {'train': [
        {'input': [[0, 1]], 'output': [[3, 1]]},
        {'input': [[0, 2]], 'output': [[3, 2]]},
        {'input': [[0, 1]], 'output': [[7, 1]]},
        {'input': [[0, 2]], 'output': [[3, 2]]},
    ], 'test': [{'input': [[0, 1]], 'output': [[7, 3]]}]}
    path.write_text(json.dumps({'task': task}))
    return ARCExampleDataset(path, seed=42, max_augments=80,
        augmentation_sampling_mode='context_palette', attention_pattern='pair_group',
        training_demo_schedule='half_or_more', inference_demo_policy='leave_one_out', **kwargs)


def sampler(ds, epochs=80):
    return LazyTrainBatchSampler(regular_train_indices_by_example_id=ds.regular_train_indices_by_example_id,
        rearc_pair_counts_by_example_id={}, nvarc_pair_counts_by_example_id={},
        task_ids=ds.task_ids, batch_size=2, seed=42, balance_dihedrals=True, num_epochs=epochs)


def assert_balanced(mappings, protected):
    groups = defaultdict(list)
    for mapping, hidden in zip(mappings, protected):
        assert sorted(mapping[:10].tolist()) == list(range(10))
        assert mapping[10:].tolist() == [10, 11, 12, 13]
        assert all(int(mapping[c]) == c for c in hidden)
        groups[hidden].append(mapping)
    for hidden, values in groups.items():
        legal = sorted(set(range(10)) - set(hidden))
        for source in legal:
            counts = Counter(int(mapping[source]) for mapping in values)
            assert max(counts[d] for d in legal) - min(counts[d] for d in legal) <= 1


def test_visible_output_only_colors_move_unused_destinations_allowed_and_hidden_reserved(tmp_path):
    ds = dataset(tmp_path)
    ex = next(ds.iter_examples(split='test'))
    geometry = list(range(8)) * 10
    maps = color_schedule(ds, ex, geometry, seed=42, max_augments=80)
    hidden = protection_schedule(ds, ex, 80)
    assert_balanced(maps, hidden)
    assert any(int(m[3]) != 3 for m in maps)  # old output-only protection released
    assert any(int(m[1]) == 8 for m in maps)  # unused label is a legal destination
    for visit, m in enumerate(maps):
        if 2 not in ds.demonstration_indices(ex, visit):
            assert 7 in hidden[visit] and m[7] == 7
            assert all(m[c] != 7 for c in range(10) if c != 7)


def test_gold_training_target_is_not_visible_evidence(tmp_path):
    ds = dataset(tmp_path)
    ex = ds.examples[2]  # only solved output containing 7, held out as target
    assert all(7 in hidden for hidden in protection_schedule(ds, ex, 80))


@pytest.mark.parametrize('legal_count', range(1, 11))
@pytest.mark.parametrize('count', [1, 79, 80, 330])
def test_every_palette_size_and_partial_latin_block(legal_count, count):
    pair = dict(input=[list(range(legal_count))], output=[[0]])
    ds = SimpleNamespace(task_colors={'task': list(range(10))},
        demonstration_pairs=lambda ex: [pair],
        demonstration_indices=lambda ex, visit: (0,),
        test_pairs_by_task={'task': [dict(input=[[0]])]})
    ex = SimpleNamespace(task_id='task', split='test', pair_index=0)
    geometry = [i % 8 for i in range(count)]
    maps = color_schedule(ds, ex, geometry, seed=42, max_augments=80)
    assert_balanced(maps, [tuple(range(legal_count, 10))] * count)


@pytest.mark.parametrize('epochs', [1, 9, 80, 330])
def test_training_preserves_exact_sampler_demo_geometry_rng_and_color_balance(tmp_path, epochs):
    ds = dataset(tmp_path)
    s = sampler(ds, epochs)
    before = copy.deepcopy(s._balanced_indices())
    aug = ContextPaletteAugmentor(ds, seed=42, max_augments=80)
    aug.prepare_training_schedule(s)
    assert before == s._balanced_indices()
    rng, torch_rng = random.getstate(), torch.get_rng_state()
    by_target = defaultdict(list)
    for epoch in before:
        for index in epoch:
            ex = ds[index]
            m, geometry = aug.select_for_example(ex)
            assert geometry == index[3]
            tokens, sep, _ = ds.build_training_sequence(ex, geometry, m)
            visible = ds.demonstration_indices(ex, index[2])
            expected, expected_sep = encode_context([ds.pairs_by_task[ex.task_id][i] for i in visible],
                ds.pairs_by_task[ex.task_id][ex.pair_index], geometry, m)
            assert torch.equal(tokens, expected) and sep == expected_sep
            by_target[ex.pair_index].append(m)
    for index, maps in by_target.items():
        assert_balanced(maps, protection_schedule(ds, ds.examples[index], epochs))
    assert random.getstate() == rng and torch.equal(torch_rng, torch.get_rng_state())
    # Selection remains identical in reverse order, after reconstruction/resume.
    replay = ContextPaletteAugmentor(ds, seed=42, max_augments=80)
    replay.prepare_training_schedule(s)
    for epoch in reversed(before):
        for index in reversed(epoch):
            a, d = aug.select_for_example(ds[index]); b, e = replay.select_for_example(ds[index])
            assert d == e and torch.equal(a, b)


def test_inference_budget_omissions_d4_no_answers_and_vote_inversion(tmp_path):
    ds = dataset(tmp_path)
    ex = next(ds.iter_examples(split='test'))
    attempts = evaluate._build_inference_attempts(ds, 'test', None, None, 80, True)
    assert len(attempts) == 80
    assert Counter(a.dihedral_index for a in attempts) == {d: 10 for d in range(8)}
    omissions = Counter()
    results = []
    for visit, a in enumerate(attempts):
        visible = ds.demonstration_indices(ex, visit)
        omissions.update(set(range(4)) - set(visible))
        m = torch.tensor(a.color_mapping)
        expected, sep = encode_context([ds.pairs_by_task['task'][i] for i in visible],
            {'input': [[0, 1]]}, a.dihedral_index, m)
        assert a.prompt == expected.tolist() and len(a.prompt) == sep + 1
        output = apply_dihedral_transform(apply_color_permutation_to_grid([[7, 3]], m), a.dihedral_index)
        results.append(dict(task_id='task', pair_index=0, output_grid=output,
            original_input=a.original_input, color_mapping=a.color_mapping,
            source_colors=a.source_colors, dihedral_index=a.dihedral_index))
    assert list(omissions.values()) == [20] * 4
    assert_balanced([torch.tensor(a.color_mapping) for a in attempts], protection_schedule(ds, ex, 80))
    assert evaluate.run_aaivr_on_results(results)[0].ranked_candidates[0]['count'] == 80
    original = json.loads(ds.source_path.read_text())
    original['task']['test'][0]['output'] = [[8, 9]]
    ds.source_path.write_text(json.dumps(original))
    changed = ARCExampleDataset(ds.source_path, seed=42, augmentation_sampling_mode='context_palette',
        attention_pattern='pair_group', inference_demo_policy='leave_one_out')
    repeated = evaluate._build_inference_attempts(changed, 'test', None, None, 80, True)
    assert [(a.prompt, a.color_mapping) for a in attempts] == [(a.prompt, a.color_mapping) for a in repeated]


def test_symmetric_prompts_and_unchanged_target_inputs_do_not_drop_attempts(tmp_path):
    ds = dataset(tmp_path)
    task = {'train': [{'input': [[0]], 'output': [[0]]}] * 3, 'test': [{'input': [[0]]}]}
    ds.source_path.write_text(json.dumps({'task': task}))
    simple = ARCExampleDataset(ds.source_path, seed=42, augmentation_sampling_mode='context_palette',
        attention_pattern='pair_group', inference_demo_policy='leave_one_out')
    attempts = evaluate._build_inference_attempts(simple, 'test', None, None, 80, True)
    assert len(attempts) == 80
    assert len({tuple(a.prompt) for a in attempts}) == 10  # repeats are explicitly permitted


def test_worker_and_shared_epoch_replay(tmp_path):
    ds = dataset(tmp_path, splits=('train',))
    def loader(workers):
        aug = ContextPaletteAugmentor(ds, seed=42, max_augments=80)
        result = create_dataloader(ds, 2, seed=42, augmentor=aug,
            augment_selector=aug.select_for_example, use_length_bucketing=False,
            num_workers=workers, persistent_workers=bool(workers), num_epochs=16)
        aug.prepare_training_schedule(result.batch_sampler)
        return result
    a, b = loader(0), loader(2)
    for epoch in [0, 7, 15, 3]:
        a.batch_sampler.set_epoch(epoch); b.batch_sampler.set_epoch(epoch)
        for x, y in zip(a, b):
            for key in ['input_ids', 'positions_3d', 'dihedral_ids', 'example_ids']:
                assert torch.equal(x[key], y[key])


def test_finalized_drop_last_sparse_visits_and_changed_geometry(tmp_path):
    ds = dataset(tmp_path, splits=('train',))
    aug = ContextPaletteAugmentor(ds, seed=42, max_augments=80)
    s = sampler(ds, 80)
    s.batch_size = 3  # four targets: a different target can be dropped each epoch
    aug.prepare_training_schedule(s)
    before = copy.deepcopy(s._balanced_indices())
    s.drop_last = True  # exactly what _train_single_model does after construction
    finalized = copy.deepcopy(s._balanced_indices())
    assert before != finalized
    aug.prepare_training_schedule(s)
    maps, visits = defaultdict(list), defaultdict(list)
    for epoch in finalized:
        for index in epoch:
            ex = ds[index]
            m, geometry = aug.select_for_example(ex)
            assert geometry == index[3]
            maps[ex.pair_index].append(m)
            visits[ex.pair_index].append(index[2])
    assert sum(map(len, maps.values())) == 80 * 3
    assert any(values != list(range(len(values))) for values in visits.values())
    for index, values in maps.items():
        hidden = protection_schedule(ds, ds.examples[index], len(values), visits=visits[index])
        assert_balanced(values, hidden)


def test_real_training_loop_finalizes_before_worker_start(tmp_path, monkeypatch):
    import train
    from argparse import Namespace
    ds = dataset(tmp_path, splits=('train',))
    aug = ContextPaletteAugmentor(ds, seed=42, max_augments=80)
    loader = create_dataloader(ds, 3, seed=42, augmentor=aug,
        augment_selector=aug.select_for_example, use_length_bucketing=False,
        num_workers=2, persistent_workers=True, num_epochs=3)
    loader.augmentor = aug
    class Scalar(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
        def forward(self, *args, **kwargs):
            return {'loss': self.weight, 'output_loss': self.weight}
    model = Scalar()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    monkeypatch.setattr(train, '_build_optimizer', lambda *args: optimizer)
    monkeypatch.setattr(train, '_save_checkpoint', lambda *args, **kwargs: None)
    args = Namespace(epochs=3, weight_decay=0., grad_clip=0., warmup_pct=0.,
        wsd_decay_start_pct=1., resume_warmup_pct=0., lr_floor=.0001,
        train_log_mode='never', log_location='none')
    train._train_single_model(args, model, loader, ds, torch.device('cpu'),
        ds.source_path, save_path=None, checkpoint_schedule=None)
    assert loader.batch_sampler.drop_last
    assert model.weight.item() == pytest.approx(.7)  # three actual optimizer steps


def test_mode_plan_build_and_checkpoint_roundtrip(tmp_path, monkeypatch):
    import train
    ds = dataset(tmp_path)
    monkeypatch.setattr(build, 'resolve_device', lambda _: torch.device('cpu'))
    args = SimpleNamespace(seed=42, data_path=ds.source_path, checkpoint_path=None,
        enable_aug=True, max_augments=80, augmentation_sampling_mode='context_palette',
        attention_pattern='pair_group', training_demo_schedule='half_or_more',
        d_model=24, n_heads=4, d_ff=48, n_layers=1, num_task_embedding_latents=2,
        dropout=0, embedding_dropout=0, attention_dropout=0, n_predict=1,
        batch_size=2, epochs=10, num_workers=0)
    model, actual, loader, _, _ = build.build_model_and_data(args)
    assert isinstance(actual.augmentor, ContextPaletteAugmentor)
    actual.augmentor.prepare_training_schedule(loader.batch_sampler)
    assert next(iter(loader))['input_ids'].numel() > 0
    path = tmp_path / 'model.pt'
    train.maybe_save_model(model, actual, ds.source_path, path)
    checkpoint = build.load_checkpoint(path)
    assert checkpoint['augmentation_sampling_mode'] == 'context_palette'
    args.augmentation_sampling_mode = None
    assert build.build_model_and_data(args, checkpoint=checkpoint, build_dataloader=False)[1].augmentation_sampling_mode == 'context_palette'
    plan = launch_run.build_plan({'enable_finetune': False,
        'defaults': {'augmentation_sampling_mode': 'context_palette'}}, tmp_path / 'spec.json')
    assert plan['pretrain']['augmentation_sampling_mode'] == 'context_palette'
    assert all(stage['augmentation_sampling_mode'] == 'context_palette' for stage in plan['evals'].values())
