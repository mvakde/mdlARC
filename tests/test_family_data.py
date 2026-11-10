"""Family membership, boundary and immutable inventory contracts using tiny saved corpora."""
import collections
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import launch_run
import variations
from dataset_building_scripts import family_data as fd
from dataset_building_scripts.build_variation_dataset import ORIGINAL_FILES
from dataset_building_scripts.task_splits import content_sha256


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(fd.encoded(value))
    return fd.sha(path.read_bytes())


def task(color):
    return {'train': [{'input': [[color]], 'output': [[(color + 1) % 10]]}],
            'test': [{'input': [[color, color]], 'output': [[color, color]]}]}


@pytest.fixture
def corpus(tmp_path):
    root, synth = tmp_path / 'repo', tmp_path / 'synth'
    sources = {'arc2_train': {'a': task(1)}, 'arc2_eval': {'b': task(3), 'c': task(5)},
               'conceptarc': {'concept': task(7)}, 'arc1_filtered': {}}
    for source, tasks in sources.items():
        save(root / 'assets_tmp' / ORIGINAL_FILES[source], tasks)
        for tid, original in tasks.items():
            job = synth / 'results/task-variations' / fd.GROUPS[source] / tid
            save(job / 'task.json', original)
            (job / 'generator.py').write_text('# saved generator; never executed\n')
            save(job / 'llm-inputs/01/recorded-inputs.json', {'demonstrations': []})
            hashes = {}
            for seed in range(2):
                instance = task(8 + seed)
                if seed == 1:instance['train'].append(instance['train'][0])
                hashes[f'seed-{seed}.json'] = save(job / 'samples' / f'seed-{seed}.json', instance)
            save(job / 'samples/manifest', {'state': 'complete', 'file_sha256': hashes,
                 'task_paths': list(hashes), 'completed_seeds': [0, 1], 'sampling': {'count': 2},
                 'generator': {'path': 'generator.py', 'sha256': fd.sha((job/'generator.py').read_bytes())}})
    manifest = {'schema_version': 1, 'name': 'tiny', 'source': 'arc2_eval',
                'source_sha256': content_sha256(sources['arc2_eval']), 'train': ['b'],
                'eval_groups': {'development_validation': ['c']}}
    save(root / 'split.json', manifest)
    raw = {'dataset': 'arc2', 'style': 'finetune', 'evaluation': {'source': 'arc2_eval', 'split_manifest': 'split.json'},
           'pretrain_data': {'kind': 'family_sampled', 'mode': 'development', 'sources': ['arc2_train', 'arc2_eval'],
                            'sampling': {}, 'discovery': {'composition_pipelines': ['v3'], 'accepted_states': ['author_completed', 'completed']}}}
    return SimpleNamespace(root=root, synth=synth, sources=sources, raw=raw)


def composition(c, name, parents, *, context=(), state='author_completed', dataset='arc2/evaluation'):
    directory = c.synth / 'results/task-composition-v3-fresh'
    plan_path = directory / 'plan.json'
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {'dataset': dataset, 'jobs': []}
    plan['jobs'].append({'name': name, 'parents': parents, 'review_examples': list(context)})
    save(plan_path, plan)
    job = directory / name
    status = {'state': state, 'parents': parents, 'dataset': dataset}
    if state == 'completed':status['samples'] = 'samples'
    save(job / 'status.json', status)
    stages = ['author', 'review-improve'] if state == 'completed' else ['author']
    for stage in stages:
        inputs = job / f'llm-inputs/{stage}/inputs'
        protected = {'environment.json': save(inputs / 'environment.json', {})}
        if stage == 'review-improve':
            protected['examples/README.md'] = save(inputs/'examples/README.md', {'description': 'Worked examples'})
            protected['examples/manifest.json'] = save(inputs/'examples/manifest.json', {'example_ids': list(context)})
        for label, parent in zip(('A', 'B'), parents):
            if parent.startswith(('arc2/training/', 'arc2/evaluation/')):
                _, split, native = parent.split('/', 2)
                source = 'arc2_train' if split == 'training' else 'arc2_eval'
            else:
                source, native = parent.split('/', 1) if '/' in parent else ('arc2_eval', parent)
            value = c.sources.get(source, {}).get(native, task(2))
            protected[f'parents/{label}/task.json'] = save(inputs / f'parents/{label}/task.json', value)
        for tid in context:
            protected[f'examples/{tid}/task.json'] = save(inputs / f'examples/{tid}/task.json', c.sources['arc2_eval'][tid])
        save(job / f'llm-inputs/{stage}/protected.json', protected)
    prefix = 'samples' if state == 'completed' else 'llm-inputs/author/checked-samples'
    hashes = {f'{prefix}/seed-{i}.json': save(job / prefix / f'seed-{i}.json', task(8 + i)) for i in range(2)}
    save(job / 'checkpoints.json', {stages[-1]: hashes})
    source = 'arc2_mixed' if dataset == 'arc2/mixed' else 'arc2_eval'
    return job, f'composition/{directory.name}/{source}/{name}'


def inventory(c):
    return fd.discover(c.root, c.synth, fd.normalize_config(c.raw, c.root))


@pytest.mark.parametrize('workers', [0, 2])
@pytest.mark.parametrize('schedule', ['half_or_more', 'two'])
def test_family_context_palette_build_replay_balance_and_visible_context(corpus, monkeypatch, workers, schedule):
    import random
    import torch
    import build
    from context_palette import protection_schedule
    from common import encode_context, collate_examples
    c = corpus
    catalog, payload = inventory(c)
    identity = fd.package_catalog(catalog, payload, c.root/'assets_tmp/context.zip')
    identity['archive_path'] = 'assets_tmp/context.zip'
    config = fd.normalize_config(c.raw, c.root)
    config['pretrain_data']['catalog'] = identity
    output = c.root/'run/data'
    fd.build_family_dataset(config, output, root=c.root)
    monkeypatch.setattr(variations, 'ROOT', c.root)
    monkeypatch.setattr(build, 'resolve_device', lambda _: torch.device('cpu'))
    args = SimpleNamespace(seed=42, device='cpu', checkpoint_path=None,
        data_path=output/'challenges.json', variation_index_path=output/'variation_index.json',
        augmentation_sampling_mode='context_palette', attention_pattern='pair_group',
        training_demo_schedule=schedule, enable_aug=True, max_augments=80,
        batch_size=4, epochs=50, num_workers=workers, persistent_workers=True,
        d_model=24, n_heads=4, d_ff=48, n_layers=1, num_task_embedding_latents=2,
        dropout=0., embedding_dropout=0., attention_dropout=0., n_predict=1)
    _, ds, loader, _, _ = build.build_model_and_data(args)
    sampler, aug = loader.batch_sampler, loader.augmentor
    before = sampler._indices_for_epoch(0)
    aug.prepare_training_schedule(sampler)
    assert before == sampler._indices_for_epoch(0)
    groups = collections.defaultdict(list)
    actual_visits = collections.Counter()
    saved = []
    hidden_variant_output_seen = False
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        indices = sampler._indices_for_epoch(epoch)
        for index in indices:
            ex = ds[index]
            if ex.pair_index == 0:
                actual_visits[(ex.task_id, ex.variation_path)] += 1
            mapping, geometry = aug.select_for_example(ex)
            visit = ex.demo_visit_override
            protected = protection_schedule(ds, ex, 1, visits=[visit])[0]
            assert all(int(mapping[color]) == color for color in protected)
            assert sorted(mapping[:10].tolist()) == list(range(10))
            assert mapping[10:].tolist() == [10, 11, 12, 13]
            if ex.variation_colors == (8, 9) and ex.pair_index == 0:
                hidden_variant_output_seen = True
                assert 9 in protected and mapping[9] == 9
            pairs = ds.demonstration_pairs(ex)
            demos = [pairs[i] for i in ds.demonstration_indices(ex, visit)]
            if schedule == 'two':
                assert len(demos) == min(2, len(pairs) - 1)
            tokens, sep, _ = ds.build_training_sequence(ex, geometry, mapping)
            expected, expected_sep = encode_context(demos, pairs[ex.pair_index], geometry, mapping)
            assert torch.equal(tokens, expected) and sep == expected_sep
            groups[(ex.task_id, ex.variation_path, ex.pair_index, protected)].append(mapping)
            saved.append((index, mapping.clone(), geometry))
        if epoch in (0, 1):
            actual_batches = list(loader)
            assert sum(len(batch['example_ids']) for batch in actual_batches) == len(indices)
            for actual, indexes in zip(actual_batches, sampler):
                expected = collate_examples([ds[index] for index in indexes], dataset=ds,
                    augment_selector=aug.select_for_example)
                assert torch.equal(actual['input_ids'], expected['input_ids'])
    assert hidden_variant_output_seen
    assert dict(actual_visits) == aug._family_visits
    # Every physical candidate is balanced within its actual protection sets.
    for (_, _, _, protected), mappings in groups.items():
        legal = set(range(10)) - set(protected)
        for source in legal:
            counts = collections.Counter(int(mapping[source]) for mapping in mappings)
            assert max(counts[d] for d in legal) - min(counts[d] for d in legal) <= 1
    # Ignore DataLoader's own documented RNG consumption when checking selection.
    rng, torch_rng = random.getstate(), torch.get_rng_state()
    aug._plans.clear()
    for index, mapping, geometry in reversed(saved):
        replay, d4 = aug.select_for_example(ds[index])
        assert d4 == geometry and torch.equal(replay, mapping)
    assert random.getstate() == rng and torch.equal(torch_rng, torch.get_rng_state())


@pytest.mark.parametrize('size', [1, 2, 3, 42, 43])
def test_family_cycles_reshuffle_without_replacement_and_replay_on_resume(size):
    import random

    sizes = {'arc2_train/a': size, 'arc2_eval/b': size,
             'composition/example': max(1, size - 1)}
    dataset = SimpleNamespace(
        variation_index={'sampling': {'unit': 'family'},
                         'composition_parents': ['composition/example']},
        variation_parents={parent: [{'pair_count': variant % 3 + 1}
                                   for variant in range(count)]
                           for parent, count in sizes.items()},
        regular_train_indices_by_example_id={},
    )

    def make_sampler(seed=42):
        sampler = variations.VariationBatchSampler(dataset, 4, seed=seed,
            variations_per_parent_per_epoch=1, eval_support_repeats_per_epoch=0,
            minibatch_multiple=2)
        sampler.drop_last = True  # Even short batches must preserve every pair.
        return sampler

    rng = random.getstate()
    sampler = make_sampler()
    horizon = size * 8
    orders = collections.defaultdict(list)
    saved = []
    for epoch in range(horizon):
        sampler.set_epoch(epoch)
        indices = [index for batch in sampler for index in batch]
        saved.append(indices)
        assert len(list(sampler)) == len(sampler) == (len(indices) + 3) // 4
        selected = collections.defaultdict(set)
        pairs = collections.defaultdict(list)
        for _, parent, variant, pair, actual_epoch, visit in indices:
            assert actual_epoch == epoch
            assert visit == epoch // sizes[parent]
            selected[parent].add(variant)
            pairs[parent].append(pair)
        assert set(selected) == set(sizes)
        for parent, variants in selected.items():
            assert len(variants) == 1
            variant = next(iter(variants))
            orders[parent].append(variant)
            assert sorted(pairs[parent]) == list(range(variant % 3 + 1))

    for parent, count in sizes.items():
        cycles = [orders[parent][start:start + count]
                  for start in range(0, horizon - count + 1, count)]
        assert all(sorted(order) == list(range(count)) for order in cycles)
        if count > 1:
            assert all(a != b for a, b in zip(cycles, cycles[1:]))
            # Candidate zero (the original in an original-task family) has no
            # reserved first position in a cycle.
            assert any(order[0] != 0 for order in cycles)
    if size >= 3:
        assert orders['arc2_train/a'] != orders['arc2_eval/b']
        changed_seed = make_sampler(seed=123)
        assert any(changed_seed._indices_for_epoch(epoch) != saved[epoch]
                   for epoch in range(horizon))

    # Jump directly into later cycles and backwards, both after normal
    # iteration and on a fresh sampler as used by a resumed run.
    for replay in (sampler, make_sampler()):
        for epoch in reversed(range(horizon)):
            replay.set_epoch(epoch)
            assert [index for batch in replay for index in batch] == saved[epoch]
            assert len(replay) == (len(saved[epoch]) + 3) // 4
    assert random.getstate() == rng


@pytest.mark.parametrize('mode', ['development', 'full'])
@pytest.mark.parametrize('cap', [3, 6])
def test_capped_context_balances_complete_large_family_instances(corpus, monkeypatch, mode, cap):
    from common import collate_examples
    c = corpus
    c.raw['pretrain_data']['mode'] = mode
    if mode == 'full':
        c.raw['evaluation'].pop('split_manifest')
    # Give each generated instance ten solved pairs while keeping the original
    # candidate small. Re-pin the fixture payloads through the normal catalog.
    for source, tasks in c.sources.items():
        for tid in tasks:
            samples = c.synth / 'results/task-variations' / fd.GROUPS[source] / tid / 'samples'
            manifest = json.loads((samples / 'manifest').read_text())
            for name in manifest['task_paths']:
                value = json.loads((samples / name).read_text())
                pairs = value['train'] + value['test']
                manifest['file_sha256'][name] = save(samples / name,
                    {'train': (pairs * 10)[:9], 'test': [pairs[-1]]})
            save(samples / 'manifest', manifest)
    catalog, payload = inventory(c)
    identity = fd.package_catalog(catalog, payload, c.root/'assets_tmp/capped.zip')
    identity['archive_path'] = 'assets_tmp/capped.zip'
    config = fd.normalize_config(c.raw, c.root)
    config['pretrain_data']['catalog'] = identity
    output = c.root/'run/data'
    fd.build_family_dataset(config, output, root=c.root)
    monkeypatch.setattr(variations, 'ROOT', c.root)
    dataset = variations.VariationTrainingDataset(json_path=output/'challenges.json',
        index_path=output/'variation_index.json', seed=42, max_augments=80,
        attention_pattern='pair_group', training_demo_schedule='half_or_more', max_training_demos=cap)
    sampler = variations.VariationBatchSampler(dataset, 4, seed=42,
        variations_per_parent_per_epoch=1, eval_support_repeats_per_epoch=0)
    saw_large = False
    for epoch in range(12):
        contexts = collections.defaultdict(collections.Counter)
        examples = []
        for index in sampler._indices_for_epoch(epoch):
            ex = dataset[index]
            examples.append(ex)
            visible = dataset.demonstration_indices(ex, ex.demo_visit_override)
            count = len(dataset.demonstration_pairs(ex))
            assert ex.pair_index not in visible and len(visible) <= cap
            if count == 10:
                saw_large = True
                assert min(cap, 5) <= len(visible) <= cap
            contexts[ex.variation_path].update(visible)
        assert all(len(set(counts.values())) <= 1 for counts in contexts.values())
        if epoch == 0:
            batch = collate_examples(examples, dataset)
            assert batch['positions_3d'][:, 2].max() <= 14
    assert saw_large
    metadata = dataset.checkpoint_metadata(enable_aug=True)
    assert metadata.get('max_training_demos', 6) == cap


def test_original_is_one_candidate_and_every_family_contributes_one_whole_instance(corpus, monkeypatch):
    c = corpus
    _, comp = composition(c, 'allowed', ['b', 'b'])
    catalog, payload = inventory(c)
    assert catalog['summary']['ready']
    assert set(catalog['families']) == {'arc2_train/a', 'arc2_eval/b', comp}
    assert [i['kind'] for i in catalog['families']['arc2_eval/b']['instances']].count('original') == 1
    assert len(catalog['families'][comp]['instances']) == 2
    identity = fd.package_catalog(catalog, payload, c.root/'assets_tmp/catalog.zip')
    identity['archive_path'] = 'assets_tmp/catalog.zip'
    config = fd.normalize_config(c.raw, c.root); config['pretrain_data']['catalog'] = identity
    output = c.root/'run/data'
    fd.build_family_dataset(config, output, root=c.root)
    monkeypatch.setattr(variations, 'ROOT', c.root)
    dataset = variations.VariationTrainingDataset(json_path=output/'challenges.json', index_path=output/'variation_index.json', seed=42, max_augments=80)
    sampler = variations.VariationBatchSampler(dataset, 4, seed=42, variations_per_parent_per_epoch=1, eval_support_repeats_per_epoch=0, minibatch_multiple=2)
    sampler.drop_last = True  # Training must still visit the final short batch.
    visits = collections.defaultdict(collections.Counter)
    for epoch in range(12):
        sampler.set_epoch(epoch)
        indices = [i for batch in sampler for i in batch]
        assert len(indices) >= 6
        assert len(list(sampler)) == len(sampler) == (len(indices) + 3) // 4
        selected = collections.defaultdict(set)
        for i in indices:
            selected[i[1]].add(i[2]); example = dataset[i]
            assert len(dataset.demonstration_pairs(example)) == dataset.variation_parents[i[1]][i[2]]["pair_count"]
        assert set(selected) == set(catalog['families'])
        assert all(len(v) == 1 for v in selected.values())
        assert len(indices) == sum(dataset.variation_parents[f][next(iter(v))]['pair_count'] for f, v in selected.items())
        for family, indexes in selected.items():visits[family][next(iter(indexes))] += 1
        fresh = variations.VariationBatchSampler(dataset, 4, seed=42, variations_per_parent_per_epoch=1, eval_support_repeats_per_epoch=0)
        assert indices == fresh._indices_for_epoch(epoch)
    assert set(visits['arc2_eval/b'].values()) == {4}  # Original has exactly the same weight as either seed.
    assert set(visits[comp].values()) == {6}
    metadata = dataset.checkpoint_metadata(enable_aug=True)
    assert metadata['family_catalog'] == identity
    # Resume uses its original package even when a new saved composition appears.
    composition(c, 'new', ['b', 'b'])
    assert len(inventory(c)[0]['families']) == 4
    fd.build_family_dataset(config, output, root=c.root)
    assert len(dataset.variation_parents) == 3
    altered = copy.deepcopy(config); altered['pretrain_data']['catalog']['sha256'] = 'f'*64
    with pytest.raises(ValueError, match='identity changed'):
        fd.build_family_dataset(altered, output, root=c.root)
    held = json.loads((output/'priv_challenges.json').read_text())
    assert set(held) == {'arc2_eval/c'}
    assert all('output' not in p for p in held['arc2_eval/c']['test'])
    assert json.loads((output/'solutions.json').read_text())['arc2_eval/c'] == [[[5, 5]]]


@pytest.mark.parametrize('parents', [['b', 'c'], ['c', 'b'], ['c', 'c']])
def test_either_parent_excludes_entire_composition(corpus, parents):
    _, key = composition(corpus, 'tainted', parents)
    catalog, _ = inventory(corpus)
    assert key in catalog['excluded']
    assert 'arc2_eval/c' not in catalog['families']


@pytest.mark.parametrize('state', ['author_completed', 'completed'])
def test_mixed_split_saved_plans_resolve_qualified_parents_and_keep_stage(corpus, state):
    _, key = composition(corpus, 'mixed', ['arc2/evaluation/b', 'arc2/training/a'],
                         dataset='arc2/mixed', state=state)
    catalog, _ = inventory(corpus)
    assert catalog['summary']['ready']
    assert catalog['families'][key]['ancestors'] == ['arc2_eval/b', 'arc2_train/a']
    expected_stage = 'author' if state == 'author_completed' else 'review-improve'
    assert {v['stage'] for v in catalog['families'][key]['instances']} == {expected_stage}
    assert catalog['summary']['eligible_composition_sources'] == {'arc2_mixed': 1}


@pytest.mark.parametrize('parents', [
    ['arc2/evaluation/c', 'arc2/training/a'],
    ['arc2/training/a', 'arc2/evaluation/c'],
    ['b', 'arc2/training/a'],  # Bare IDs have no split in a mixed plan.
    ['arc2/evaluation/absent', 'arc2/training/a'],
])
def test_mixed_split_heldout_or_unknown_parent_fails_closed(corpus, parents):
    _, key = composition(corpus, 'mixed', parents, dataset='arc2/mixed')
    catalog, _ = inventory(corpus)
    assert key in catalog['excluded']
    assert key not in catalog['families']


@pytest.mark.parametrize('parent,enabled', [('arc2/evaluation/b', False), ('arc2/training/a', True)])
def test_mixed_split_respects_enabled_original_sources(corpus, parent, enabled):
    _, key = composition(corpus, 'mixed', [parent, 'arc2/training/a'], dataset='arc2/mixed')
    corpus.raw['pretrain_data']['sources'] = ['arc2_train']
    catalog, _ = inventory(corpus)
    assert (key in catalog['families']) is enabled
    assert catalog['summary']['eligible_original_families'] == {'arc2_train': 1}


def test_mixed_split_inherits_parent_authoring_context_and_excludes_descendants(corpus):
    _, key = composition(corpus, 'mixed', ['arc2/evaluation/b', 'arc2/training/a'], dataset='arc2/mixed')
    _, child = composition(corpus, 'child', [key, 'arc2/training/a'], dataset='arc2/mixed')
    save(corpus.synth/'results/task-variations/arc2/evaluation/b/llm-inputs/01/recorded-inputs.json',
         {'demonstrations': [{'id': 'c', 'source_split': 'evaluation',
                              'task_sha256': content_sha256(corpus.sources['arc2_eval']['c'])}]})
    catalog, _ = inventory(corpus)
    assert {catalog['excluded'][key], catalog['excluded'][child]} == {'heldout_or_alias_ancestor'}


def test_recursive_unknown_and_cyclic_lineage_fail_closed():
    originals = {'train', 'held'}
    nodes = {'ok': ['train'], 'bad': ['held'], 'descendant': ['bad', 'train'],
             'missing': ['absent'], 'unknown': None, 'loop': ['loop']}
    accepted, excluded = fd.lineage_closure(nodes, originals, {'held'})
    assert accepted == {'ok': ['train']}
    assert set(excluded) == {'bad', 'descendant', 'missing', 'unknown', 'loop'}


def test_context_and_transitive_composition_exclusion(corpus):
    c = corpus
    job, key = composition(c, 'context', ['b', 'b'], context=['c'], state='completed')
    _, child = composition(c, 'descendant', [key, 'b'])
    catalog, _ = inventory(c)
    assert {key, child} <= set(catalog['excluded'])
    assert not any(k.startswith('composition/') for k in catalog['families'])
    (job/'llm-inputs/author/protected.json').unlink()
    catalog, _ = inventory(c)
    assert key in catalog['excluded']


def test_missing_payloads_cannot_produce_partial_package(corpus):
    c = corpus
    sample = c.synth/'results/task-variations/arc2/training/a/samples/seed-0.json'
    sample.unlink()
    catalog, payload = inventory(c)
    assert catalog['summary']['missing_payloads'] == {'variation': 1}
    assert len(catalog['families']['arc2_train/a']['instances']) == 3
    with pytest.raises(ValueError, match='not ready'):
        fd.package_catalog(catalog, payload, c.root/'assets_tmp/incomplete.zip')
    assert not (c.root/'assets_tmp/incomplete.zip').exists()


def test_alias_fingerprints_exclude_whole_other_source_family(corpus):
    c = corpus
    # Reuse a heldout pair in an otherwise different filtered original.
    c.sources['arc1_filtered']['alias'] = {'train': [c.sources['arc2_eval']['c']['train'][0]], 'test': []}
    save(c.root/'assets_tmp'/ORIGINAL_FILES['arc1_filtered'], c.sources['arc1_filtered'])
    catalog, _ = inventory(c)
    assert catalog['aliases_excluded'] == ['arc1_filtered/alias']
    _, key = composition(c, 'alias-child', ['arc1_filtered/alias', 'b'])
    catalog, _ = inventory(c)
    assert key in catalog['excluded']


def test_full_data_mode_includes_all_public_answers_and_never_scores_them_as_heldout(corpus):
    c = corpus; c.raw['pretrain_data']['mode'] = 'full'; c.raw['evaluation'].pop('split_manifest')
    catalog, payload = inventory(c)
    assert set(catalog['families']) == {'arc2_train/a', 'arc2_eval/b', 'arc2_eval/c'}
    original = catalog['families']['arc2_eval/c']['instances'][0]
    assert json.loads(payload(original))['test'][0]['output'] == [[5, 5]]
    assert not catalog['heldout']


def test_conceptarc_is_an_explicit_configurable_addition(corpus):
    c = corpus; c.raw['pretrain_data']['sources'].append('conceptarc')
    catalog, _ = inventory(c)
    assert len(catalog['families']['conceptarc/concept']['instances']) == 3
    assert catalog['summary']['instances_per_epoch'] == 3


@pytest.mark.parametrize('mode', ['development', 'full'])
def test_family_configs_preserve_default_scaling_ladder_recipe(mode):
    base = json.loads((ROOT/'specs/pretrain_eval.json').read_text())
    path = ROOT/f'specs/arc2_family_{mode}.json'
    spec = json.loads(path.read_text())
    # Data variants must not reintroduce the legacy FA2/dense training recipe.
    for key in base.keys() - {'run_name', 'dataset', 'evals', 'skip_evals', 'score_results'}:
        assert spec[key] == base[key], key
    assert spec['dataset']['style'] == base['dataset']['style']
    assert spec['dataset']['pretrain_data']['mode'] == mode
    assert spec['dataset']['pretrain_data']['sources'] == ['arc2_train', 'arc2_eval']
    assert 'catalog' not in spec['dataset']['pretrain_data']
    plan = launch_run.build_plan(spec, path)
    assert not plan['finetunes']
    if mode == 'full':
        assert plan['skip_evals'] and not plan['score_results']
        assert plan['evals'] == {}
        assert 'split_manifest' not in spec['dataset']['evaluation']
        spec['skip_evals'] = False
        with pytest.raises(ValueError, match='public answers'):
            launch_run.build_plan(spec, path)
        return
    assert spec['skip_evals'] == base['skip_evals']
    assert spec['score_results'] == base['score_results']
    assert spec['evals']['final'] == {**base['evals']['final'], 'group': 'development_validation'}
    assert plan['evals']['final']['group'] == 'development_validation'
    frozen = ROOT/'results/scaling-ladder/experiments/arc2-development-split-v1.json'
    tracked = ROOT/'dataset_building_scripts/dataset_splits/arc2_dev80_val40_v1.json'
    if frozen.exists():assert frozen.read_bytes() == tracked.read_bytes()
    manifest = json.loads(tracked.read_text())
    assert (len(manifest['train']), len(manifest['eval_groups']['development_validation'])) == (80, 40)


def test_pending_lineage_snapshots_block_packaging_without_shrinking_inventory(corpus):
    c = corpus
    job, key = composition(c, 'pending', ['b', 'b'])
    (job/'llm-inputs/author/inputs/parents/A/task.json').unlink()
    catalog, payload = inventory(c)
    assert key in catalog['families']
    assert catalog['summary']['missing_lineage_payloads'] == 1
    assert not catalog['summary']['ready']
    with pytest.raises(ValueError, match='not ready'):
        fd.package_catalog(catalog, payload, c.root/'assets_tmp/pending.zip')


def test_changed_generator_pin_and_unknown_composition_parent_fail_closed(corpus):
    c = corpus
    _, key = composition(c, 'unknown-parent', ['absent', 'b'])
    catalog, _ = inventory(c)
    assert key in catalog['excluded']
    (c.synth/'results/task-variations/arc2/training/a/generator.py').write_text('# changed')
    catalog, _ = inventory(c)
    assert 'arc2_train/a' in catalog['summary']['blocking_metadata_issues']


def test_generated_pair_leak_excludes_family_and_every_descendant(corpus):
    c = corpus
    job, key = composition(c, 'leak', ['b', 'b'])
    _, child = composition(c, 'child', [key, 'b'])
    path = job/'llm-inputs/author/checked-samples/seed-0.json'
    digest = save(path, c.sources['arc2_eval']['c'])
    checkpoint = json.loads((job/'checkpoints.json').read_text())
    checkpoint['author']['llm-inputs/author/checked-samples/seed-0.json'] = digest
    save(job/'checkpoints.json', checkpoint)
    catalog, _ = inventory(c)
    assert {key, child} <= set(catalog['excluded'])
    assert not ({key, child} & set(catalog['families']))


def test_pinned_packages_are_included_in_local_source_snapshots(corpus, monkeypatch):
    c = corpus
    from scripts import run_on_colab
    from workflows.vast import snapshot
    catalog, payload = inventory(c)
    identity = fd.package_catalog(catalog, payload, c.root/'assets_tmp/catalog.zip')
    identity['archive_path'] = 'assets_tmp/catalog.zip'
    spec = copy.deepcopy(c.raw); spec['pretrain_data']['catalog'] = identity
    # Snapshot entry points consume a complete experiment object, not a dataset object.
    full_spec = {'run_name': 'family-fixture', 'dataset': spec, 'defaults': {'FA': 4}}
    for filename in ['launch_run.py', 'requirements.txt', 'scripts/run_layout.py']:
        target = c.root/filename;target.parent.mkdir(parents=True, exist_ok=True);target.write_text('# fixture')
    spec_path = c.root/'selected.json';save(spec_path, full_spec)
    monkeypatch.setattr(run_on_colab, 'ROOT', c.root)
    monkeypatch.setattr(run_on_colab, 'SOURCE_DIRS', [])
    monkeypatch.setattr(run_on_colab, 'GROUPED_DIRS', [])
    assert c.root/'assets_tmp/catalog.zip' in run_on_colab.source_files(spec_path)
    assert c.root/'assets_tmp/catalog.zip' in snapshot.source_files(c.root, spec_path)
    (c.root/'assets_tmp/catalog.zip').write_bytes(b'changed')
    with pytest.raises(ValueError, match='identity changed'):
        run_on_colab.source_files(spec_path)
    with pytest.raises(ValueError, match='identity changed'):
        snapshot.source_files(c.root, spec_path)


def test_inventory_cannot_be_packaged_after_source_metadata_changes(corpus):
    c = corpus
    catalog, payload = inventory(c)
    save(c.root/'assets_tmp'/ORIGINAL_FILES['arc2_train'], {'a': task(2)})
    with pytest.raises(ValueError, match='Metadata changed'):
        fd.package_catalog(catalog, payload, c.root/'assets_tmp/stale.zip')
    assert not (c.root/'assets_tmp/stale.zip').exists()


def test_saved_status_sample_count_prevents_partial_composition_membership(corpus):
    c = corpus
    job, key = composition(c, 'partial', ['b', 'b'])
    status = json.loads((job/'status.json').read_text());status['check_count'] = 3
    save(job/'status.json', status)
    catalog, _ = inventory(c)
    assert key in catalog['excluded']
    assert catalog['issues'][key] == 'composition sample membership differs from saved status'


def test_historical_generator_hash_contract_uses_rstrip_without_rewriting_source(corpus):
    c = corpus
    job = c.synth/'results/task-variations/arc2/training/a'
    generator = job/'generator.py'
    before = generator.read_bytes()
    manifest = json.loads((job/'samples/manifest').read_text())
    manifest['generator']['sha256'] = fd.sha(generator.read_text().rstrip().encode())
    save(job/'samples/manifest', manifest)
    catalog, _ = inventory(c)
    assert catalog['summary']['ready']
    assert generator.read_bytes() == before


def test_recovery_cache_supplies_missing_bytes_without_rewriting_history(corpus):
    from dataset_building_scripts.family_recovery import RecoveryCache
    c = corpus
    path = c.synth/'results/task-variations/arc2/training/a/samples/seed-0.json'
    raw = path.read_bytes(); path.unlink()
    cache = RecoveryCache(c.root/'results/recovery')
    assert not cache.save(b'wrong', fd.sha(raw))
    assert cache.save(raw, fd.sha(raw))
    config = fd.normalize_config(c.raw, c.root)
    catalog, payload = fd.discover(c.root, c.synth, config, recovery=cache)
    assert catalog['summary']['ready']
    fd.package_catalog(catalog, payload, c.root/'assets_tmp/recovered.zip')
    assert not path.exists()
    # Existing corruption cannot be hidden by a correct recovery object.
    path.write_bytes(b'wrong')
    catalog, _ = fd.discover(c.root, c.synth, config, recovery=cache)
    assert catalog['summary']['corrupt_payloads'] == 1
    cache.path(fd.sha(raw)).write_bytes(b'wrong')
    with pytest.raises(ValueError, match='Corrupt recovery'):
        cache.read(fd.sha(raw))


def test_package_has_reproducible_member_metadata_and_bytes(corpus):
    import zipfile
    c = corpus
    catalog, payload = inventory(c)
    a = fd.package_catalog(catalog, payload, c.root/'assets_tmp/a.zip')
    b = fd.package_catalog(catalog, payload, c.root/'assets_tmp/b.zip')
    assert (a['sha256'], a['size']) == (b['sha256'], b['size'])
    with zipfile.ZipFile(c.root/'assets_tmp/a.zip') as archive:
        assert {info.date_time for info in archive.infolist()} == {(1980, 1, 1, 0, 0, 0)}


def test_protected_text_reconstruction_is_verified_and_separate(corpus):
    from dataset_building_scripts.family_recovery import RecoveryCache, recover_evidence, puzzle_text
    c = corpus
    job, key = composition(c, 'pending-text', ['b', 'b'])
    inputs = job/'llm-inputs/author/inputs'
    path = inputs/'parents/A/task.txt'
    raw = puzzle_text(c.sources['arc2_eval']['b'], 'b')
    protected = json.loads((job/'llm-inputs/author/protected.json').read_bytes())
    protected['parents/A/task.txt'] = fd.sha(raw)
    save(job/'llm-inputs/author/protected.json', protected)
    catalog, _ = inventory(c)
    assert catalog['summary']['missing_lineage_payloads'] == 1
    cache = RecoveryCache(c.root/'results/recovery')
    recover_evidence(catalog['missing_lineage_payloads'], c.synth, cache)
    catalog, _ = fd.discover(c.root, c.synth, fd.normalize_config(c.raw, c.root), recovery=cache)
    assert catalog['summary']['ready']
    assert not path.exists()


def test_recovery_reuses_archives_and_runs_only_missing_pinned_seeds(corpus):
    import zipfile
    from dataset_building_scripts.family_recovery import RecoveryCache, recover_catalog
    c = corpus
    job = c.synth/'results/task-variations/arc2/training/a'
    code = 'def generate_puzzle(seed, **kwargs):\n return ' + repr(task(2)) + '\ndef validate_generated_puzzle(task): pass\n'
    (job/'generator.py').write_text(code)
    manifest = json.loads((job/'samples/manifest').read_bytes())
    manifest['generator']['sha256'] = fd.sha(code.encode())
    manifest['sampling'].update(num_train=None, num_test=None)
    for seed, value in ((0, task(0)), (1, task(2))):
        manifest['file_sha256'][f'seed-{seed}.json'] = save(job/f'samples/seed-{seed}.json', value)
    save(job/'samples/manifest', manifest)
    first = job/'samples/seed-0.json'; raw = first.read_bytes(); first.unlink()
    second = job/'samples/seed-1.json'; second.unlink()
    archive = c.root/'puzzles.zip'
    with zipfile.ZipFile(archive, 'w') as output:
        output.writestr('../../unrelated-name.json', raw)  # Read by hash, never extract paths.
    catalog, _ = inventory(c)
    cache = RecoveryCache(c.root/'results/recovery')
    result = recover_catalog(catalog, c.synth, cache, archives=[archive], regenerate=True, workers=1)
    assert result['remaining_paths'] == 0
    assert result['generator_families_attempted'] == 1
    catalog, payload = fd.discover(c.root, c.synth, fd.normalize_config(c.raw, c.root), recovery=cache)
    assert catalog['summary']['ready']
    assert not first.exists() and not second.exists()
    fd.package_catalog(catalog, payload, c.root/'assets_tmp/recovered.zip')


def test_generator_mismatches_do_not_change_saved_sample_pins(corpus):
    from dataset_building_scripts.family_recovery import RecoveryCache, recover_catalog
    c = corpus
    job = c.synth/'results/task-variations/arc2/training/a'
    code = 'def generate_puzzle(seed, **kwargs):\n return ' + repr(task(1)) + '\ndef validate_generated_puzzle(task): pass\n'
    (job/'generator.py').write_text(code)
    manifest = json.loads((job/'samples/manifest').read_bytes())
    manifest['generator']['sha256'] = fd.sha(code.encode())
    manifest['sampling'].update(num_train=None, num_test=None)
    manifest['file_sha256']['seed-0.json'] = save(job/'samples/seed-0.json', task(2))
    save(job/'samples/manifest', manifest)
    before = (job/'samples/manifest').read_bytes()
    (job/'samples/seed-0.json').unlink()
    catalog, _ = inventory(c)
    cache = RecoveryCache(c.root/'results/recovery')
    result = recover_catalog(catalog, c.synth, cache, regenerate=True, workers=1)
    assert result['remaining_paths'] == 1
    assert (job/'samples/manifest').read_bytes() == before
    catalog, _ = fd.discover(c.root, c.synth, fd.normalize_config(c.raw, c.root), recovery=cache)
    assert not catalog['summary']['ready']


@pytest.mark.parametrize('state,version', [('author_completed', 'v1'), ('completed', 'v2')])
def test_composition_recovery_uses_the_accepted_stage_generator(corpus, state, version):
    from dataset_building_scripts.family_recovery import RecoveryCache, recover_catalog
    c = corpus
    job, key = composition(c, 'recover-composition', ['b', 'b'], state=state)
    stage = 'author' if version == 'v1' else 'review-improve'
    prefix = 'llm-inputs/author/checked-samples' if version == 'v1' else 'samples'
    values = {0: task(2), 1: task(0)}
    code = f'def generate_puzzle(seed, **kwargs):\n return {values!r}[seed]\ndef validate_generated_puzzle(task): pass\n'
    filename = f'generator_{version}.py'
    (job/filename).write_text(code)
    checkpoint = json.loads((job/'checkpoints.json').read_bytes())
    checkpoint[stage][filename] = fd.sha(code.encode())
    for seed, value in values.items():
        path = job/prefix/f'seed-{seed}.json'
        checkpoint[stage][f'{prefix}/seed-{seed}.json'] = save(path, value)
        path.unlink()
    save(job/'checkpoints.json', checkpoint)
    catalog, _ = inventory(c)
    cache = RecoveryCache(c.root/'results/recovery')
    result = recover_catalog(catalog, c.synth, cache, regenerate=True, workers=1)
    assert result['remaining_paths'] == 0
    assert result['generator_families_attempted'] == 1
    catalog, _ = fd.discover(c.root, c.synth, fd.normalize_config(c.raw, c.root), recovery=cache)
    assert catalog['summary']['ready']
    assert key in catalog['families']


def test_changed_generator_and_nondeterminism_cannot_enter_recovery(corpus):
    from dataset_building_scripts.family_recovery import RecoveryCache, generate, generator_request
    c = corpus
    catalog, _ = inventory(c)
    record = catalog['families']['arc2_train/a']['instances'][1:]
    job = c.synth/'results/task-variations/arc2/training/a'
    (job/'generator.py').write_text('# changed')
    with pytest.raises(ValueError, match='generator changed'):
        generator_request('arc2_train/a', record, c.synth, catalog)
    code = 'counter=0\ndef generate_puzzle(seed, **kwargs):\n global counter\n counter+=1\n return {"train":[{"input":[[counter]],"output":[[0]]}],"test":[]}\ndef validate_generated_puzzle(task): pass\n'
    (job/'generator.py').write_text(code)
    request = {'family': 'arc2_train/a', 'generator': str(job/'generator.py'),
               'generator_sha256': fd.sha(code.encode()), 'instances': record, 'kwargs': {}}
    with pytest.raises(ValueError, match='Nondeterministic'):
        generate(request, RecoveryCache(c.root/'results/recovery'))


def test_cross_process_string_set_order_recovers_only_original_hash(corpus):
    import os
    import subprocess
    from dataset_building_scripts.family_recovery import RecoveryCache, recover_catalog
    c = corpus
    job = c.synth/'results/task-variations/arc2/training/a'
    code = ('def generate_puzzle(seed, **kwargs):\n'
            ' order=[len(x) for x in {"red","blue","green","yellow"}]\n'
            ' return {"train":[{"input":[order],"output":[[8]]}],"test":[]}\n'
            'def validate_generated_puzzle(task): pass\n')
    program = code + '\nimport json\nprint(json.dumps(generate_puzzle(0),sort_keys=True,separators=(",",":")))'
    def output(seed):
        return subprocess.check_output([sys.executable, '-c', program],
                                       env={**os.environ, 'PYTHONHASHSEED': str(seed)}).strip()
    baseline = output(0)
    expected = next(raw for seed in range(1, 33) if (raw := output(seed)) != baseline)
    (job/'generator.py').write_text(code)
    manifest = json.loads((job/'samples/manifest').read_bytes())
    manifest['generator']['sha256'] = fd.sha(code.encode())
    manifest['file_sha256']['seed-0.json'] = fd.sha(expected)
    save(job/'samples/manifest', manifest)
    (job/'samples/seed-0.json').unlink()
    before = (job/'samples/manifest').read_bytes()
    catalog, _ = inventory(c)
    cache = RecoveryCache(c.root/'results/recovery')
    result = recover_catalog(catalog, c.synth, cache, regenerate=True, workers=1)
    assert result['remaining_paths'] == 0
    assert cache.read(fd.sha(expected)) == expected
    assert (job/'samples/manifest').read_bytes() == before
    journal = [json.loads(row) for row in (cache.directory/'generation.jsonl').read_text().splitlines()]
    assert journal[0]['hash_seed_trials'] > 0
