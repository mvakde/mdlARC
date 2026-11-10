"""Balanced contexts and identities, independent of augmentation/worker state."""
import copy
import json
import random
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common import ARCExampleDataset, collate_examples, compute_positions_3d, encode_context
from demo_schedule import MAX_CONTEXT_DEMOS, half_or_more_demonstrations
from pair_identity import balanced_pair_vertices, leave_one_out, simplex_table
from tinytransformer import TinyTransformer, TinyTransformerConfig
from test_mtp import _batch, _set_path
from test_group_attention import DEMOS, TARGET, training_compile_settings


@pytest.mark.parametrize('count', range(12))
def test_omissions_balanced_at_every_prefix_and_replay(count):
    totals = Counter({i: 0 for i in range(count)})
    for visit in range(80):
        visible = leave_one_out(range(count), visit=visit, seed=71)
        assert len(visible) == max(0, min(6, count - 1))
        assert visible == leave_one_out(range(count), visit=visit, seed=71)
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
        subset = half_or_more_demonstrations(range(count), visit=visit, seed=41)
        k = len(subset); sizes[k] += 1
        assert lower <= k <= upper
        assert len(set(subset)) == k
        totals.update(subset)
        if totals:
            assert max(totals.values()) - min(totals.values()) <= 1
    assert max(sizes.values()) - min(sizes.values()) <= 1
    assert set(sizes) == set(range(lower, upper + 1))


@pytest.mark.parametrize('count', [1, 2, 3, 4, 6, 7, 8, 10, 13, 21])
@pytest.mark.parametrize('schedule', ['all', 'half_or_more', 'two'])
def test_training_balances_within_epoch_and_each_target_across_epochs(tmp_path, count, schedule):
    pairs = [{'input': [[i % 10]], 'output': [[(i + 1) % 10]]} for i in range(count)]
    path = tmp_path / 'tasks.json'
    path.write_text(json.dumps({'puzzle': {'train': pairs}}))
    ds = ARCExampleDataset(path, splits=('train',), seed=41, attention_pattern='pair_group',
                           training_demo_schedule=schedule)
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
    ds = ARCExampleDataset(path, seed=41, attention_pattern='pair_group',
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


@pytest.mark.parametrize('count', [1, 2, 6])
@pytest.mark.parametrize('is_eval', [False, True])
def test_two_demo_mode_builds_training_and_both_evaluation_splits(tmp_path, monkeypatch, count, is_eval):
    from types import SimpleNamespace
    import build

    pairs = [{'input': [[i]], 'output': [[i + 1]]} for i in range(count)]
    path = tmp_path / 'tasks.json'
    path.write_text(json.dumps({'puzzle': {'train': pairs, 'test': [TARGET]}}))
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


def test_training_cap_reaches_built_model_dataset_without_target_leakage(tmp_path, monkeypatch):
    from types import SimpleNamespace
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


def test_simplex_geometry_and_physical_pair_balance_without_rng_use():
    table = simplex_table(11, 24)
    torch.testing.assert_close(table[1:].sum(0), torch.zeros(24), atol=1e-6, rtol=0)
    gram = table[1:] @ table[1:].T
    torch.testing.assert_close(gram, torch.eye(11)*1.1 - .1, atol=2e-7, rtol=0)
    python_state = random.getstate(); torch_state = torch.get_rng_state()
    for count in range(1, 12):
        physical = tuple(range(10, 10 + count)); counts = {p: Counter() for p in physical}
        for visit in range(33):
            ids = balanced_pair_vertices(physical, visit=visit, seed=42, vertices=11)
            assert len(set(ids.values())) == count
            for p, v in ids.items(): counts[p][v] += 1
        assert all(set(c.values()) == {3} and len(c) == 11 for c in counts.values())
    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)
    with pytest.raises(ValueError): balanced_pair_vertices(range(12), visit=0, seed=42, vertices=11)


def dataset(tmp_path, extra_demos=0, **options):
    path = tmp_path / 'tasks.json'
    path.write_text(json.dumps({'puzzle': {'train': DEMOS + [TARGET] * (1 + extra_demos), 'test': [{'input': [[9]]}]}}))
    return ARCExampleDataset(path, seed=42, attention_pattern='pair_group', **options)


@pytest.mark.parametrize('extra_demos', [0, 8])
def test_pair_tokens_boundaries_target_neutral_and_worker_resume(tmp_path, extra_demos):
    ds = dataset(tmp_path, extra_demos=extra_demos, pair_identity_vertices=11)
    batch = [replace(ds.examples[0], augment_epoch_override=visit) for visit in [0, 1, 10, 11, 79]]
    direct = collate_examples(batch, ds)
    loader = torch.utils.data.DataLoader(batch, batch_size=5, num_workers=2,
        collate_fn=__import__('functools').partial(collate_examples, dataset=ds))
    worker = next(iter(loader))
    for key in direct:
        if torch.is_tensor(direct[key]): assert torch.equal(direct[key], worker[key]), key
    replay = collate_examples(batch[-2:], ds)
    assert torch.equal(replay['pair_identity_ids'], direct['pair_identity_ids'][direct['cu_seqlens'][-3]:])
    for i, ex in enumerate(batch):
        lo, hi = direct['cu_seqlens'][i:i+2]; pos = direct['positions_3d'][lo:hi]
        ids = direct['pair_identity_ids'][lo:hi]; visible = ds.demonstration_indices(ex, ex.augment_epoch_override)
        for slot in range(len(visible)):
            pair_ids = ids[(pos[:, 2] == 2*slot+1) | (pos[:, 2] == 2*slot+2)]
            assert pair_ids.unique().numel() == 1 and pair_ids[0] > 0
        assert not ids[pos[:, 2] > 2*len(visible)].any()
        assert not ids[pos[:, 2] == 0].any()


def model(simplex=True, split=True):
    torch.manual_seed(7)
    return TinyTransformer(TinyTransformerConfig(d_model=24,n_heads=4,d_ff=48,n_layers=3,
        num_examples=2,num_task_embedding_latents=3,task_embeddings=False,
        dropout=0,embedding_dropout=0,attention_dropout=0,n_predict=4,
        attention_pattern='pair_group',split_target_demo_attention=split,
        pair_identity_vertices=11 if simplex else 0)).eval()


def ids_for(batch):
    pos = batch['positions_3d'][..., 2]
    return torch.where((pos > 0) & (pos <= 2*len(DEMOS)), (pos+1)//2, 0).long()


@pytest.mark.parametrize('compiled', [False, True])
def test_simplex_packed_padded_and_gradients(monkeypatch, compiled, training_compile_settings):
    packed, padded = model(), model()
    _set_path(packed, monkeypatch, 'packed'); _set_path(padded, monkeypatch, 'padded')
    sequences = [encode_context(DEMOS, TARGET), encode_context(DEMOS, TARGET)]
    a = _batch(packed=True, sequences=sequences); b = _batch(sequences=sequences)
    a['pair_identity_ids'] = ids_for(a); b['pair_identity_ids'] = ids_for(b)
    actual = torch.compile(packed, backend='aot_eager') if compiled else packed
    x, y = actual(**a), padded(**b)
    torch.testing.assert_close(x['logits'], torch.cat(list(y['logits'])), rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(x['loss'], y['loss'])
    x['loss'].backward(); y['loss'].backward()
    for n,p in packed.named_parameters():
        if p.requires_grad: torch.testing.assert_close(p.grad, dict(padded.named_parameters())[n].grad, rtol=5e-4, atol=3e-6)


def test_shared_weights_and_missing_identity_rejected():
    baseline, actual = model(False), model(True)
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, actual.state_dict()[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match='explicit'):
        actual(**_batch(sequences=[encode_context(DEMOS,TARGET)]))


def test_pair_identity_tensor_check_supports_full_graph_compilation():
    m=model()
    embed=torch.randn(1,4,24);ids=torch.tensor([[0,1,1,0]])
    compiled=torch.compile(m._add_pair_identities,backend='aot_eager',fullgraph=True)
    torch.testing.assert_close(compiled(embed,ids),m._add_pair_identities(embed,ids))
    with pytest.raises(ValueError,match='palette'):
        m._add_pair_identities(embed,ids+12)


def test_old_checkpoint_formats_preserve_attention_and_strict_weights():
    from build import _validate_checkpoint_format, build_checkpoint_config
    from dataclasses import asdict
    for fmt in ['all_demo_pair_group_target_roles_v2','half_or_more_pair_group_target_roles_v1']:
        old = model(False, False); cfg = asdict(old.config)
        cfg.pop('split_target_demo_attention'); cfg.pop('pair_identity_vertices');cfg.pop('pair_identity_strength')
        checkpoint = {'config':cfg,'sequence_format':fmt}
        _validate_checkpoint_format(checkpoint)
        loaded = TinyTransformer(build_checkpoint_config(checkpoint))
        assert not loaded.config.split_target_demo_attention
        loaded.load_state_dict(old.state_dict(), strict=True)
        checkpoint['config']['split_target_demo_attention'] = True
        with pytest.raises(ValueError): _validate_checkpoint_format(checkpoint)


@pytest.mark.parametrize('policy', ['leave_one_out', 'two'])
def test_subset_prompts_unchanged_augments_and_no_test_answers(tmp_path, policy):
    import evaluate
    ds = dataset(tmp_path, inference_demo_policy=policy, augmentation_sampling_mode='single_grid_prefix')
    ex = next(ds.iter_examples(split='test'))
    torch_state, random_state = torch.get_rng_state(), random.getstate()
    attempts = evaluate._build_inference_attempts(ds, 'test', None, None, 80, True)
    ds.inference_demo_policy = 'all'
    full = evaluate._build_inference_attempts(ds, 'test', None, None, 80, True)
    assert len(attempts) == len(full) <= 80  # unique-input augmentation may exhaust
    ds.inference_demo_policy = policy
    for visit,(a,b) in enumerate(zip(attempts,full)):
        assert a.dihedral_index == b.dihedral_index and a.color_mapping == b.color_mapping
        visible = ds.demonstration_indices(ex, visit)
        expected, sep = encode_context([ds.pairs_by_task[ex.task_id][i] for i in visible],
            {'input': [[9]]},a.dihedral_index,torch.tensor(a.color_mapping))
        assert a.prompt == expected.tolist() and len(a.prompt) == sep+1
    assert torch.equal(torch_state,torch.get_rng_state()) and random_state == random.getstate()


@pytest.mark.parametrize('native', [False, True])
@torch.inference_mode()
def test_simplex_cached_decode_and_future_output_isolation(monkeypatch,native):
    import torch.nn.functional as F
    m=model();_set_path(m,monkeypatch,'packed' if native else 'padded')
    tokens,sep=encode_context(DEMOS,TARGET)
    batch=_batch(sequences=[(tokens,sep)]);batch['pair_identity_ids']=ids_for(batch)
    expected=m(**batch)['logits'][0]
    changed=copy.deepcopy(batch);changed['input_ids'][0,sep+2]=(changed['input_ids'][0,sep+2]+1)%10
    torch.testing.assert_close(expected[:sep+2],m(**changed)['logits'][0,:sep+2],rtol=0,atol=0)
    prompt=tokens[:sep+1][None];mask=torch.ones_like(prompt,dtype=torch.bool)
    initial=m.forward_generate(prompt,torch.tensor([0]),attention_mask=mask,
        pair_identity_ids=batch['pair_identity_ids'][:,:sep+1])
    torch.testing.assert_close(initial['logits'][0,0],expected[sep],rtol=3e-5,atol=3e-6)
    cache=tuple(tuple(F.pad(v,(0,0,0,len(tokens)+4-v.shape[-2])) for v in layer) for layer in initial['past_key_values'])
    valid=F.pad(mask,(0,len(tokens)+4-mask.shape[-1]));positions=batch['positions_3d']
    demos=torch.zeros_like(valid)
    demos[:,:sep+1]=(positions[:,:sep+1,2]>0)&(positions[:,:sep+1,2]<positions[:,sep:sep+1,2])
    for t in range(sep+1,len(tokens)):
        valid[:,t]=True
        output=m.forward_generate(tokens[t:t+1][None],torch.tensor([0]),past_key_values=cache,
            positions_3d=positions[:,t:t+1],attention_mask=valid,cache_position=torch.tensor([t]),
            demo_attention_mask=demos)
        torch.testing.assert_close(output['logits'][0,0],expected[t],rtol=3e-5,atol=3e-6)


def test_named_checkpoint_plan_validation_and_paired_breakdown(tmp_path, monkeypatch):
    import launch_run
    from evaluation_comparison import compare_submissions
    identity={'drive_id':'example','name':'pretrain__tiny.pt','size':100,'sha256':'a'*64}
    spec=json.loads((Path(__file__).resolve().parents[1]/'specs/pretrain_eval.json').read_text())
    spec.update(run_name='multi-eval',eval_only=True,enable_finetune=False,checkpoint=identity,
        evaluation_checkpoints={'older':identity},evals={'a':{'checkpoint':'checkpoint'},'b':{'checkpoint':'input.older'}})
    plan=launch_run.build_plan(spec,tmp_path/'spec.json')
    assert launch_run.resolve_checkpoint_ref('input.older',plan)==plan['run_dir']/'inputs/older.pt'
    for name in ['../escape','pretrain','a.b']:
        with pytest.raises(ValueError):launch_run.resolve_evaluation_checkpoints(dict(spec,evaluation_checkpoints={name:identity}),tmp_path)
    with pytest.raises(ValueError):launch_run.resolve_checkpoint_ref('input.absent',plan)
    challenge={'x':{'train':[{'output':[[1]]}]*3},'y':{'train':[{'output':[[1]]}]*5}}
    solutions={'x':[[[1]],[[2]]],'y':[[[3]]]}
    old={'x':[{'attempt_1':[[1]]},{}],'y':[{'attempt_2':[[3]]}]}
    new={'x':[{'attempt_1':[[1]]},{'attempt_2':[[2]]}],'y':[{}]}
    result=compare_submissions(old,new,solutions,challenge)
    assert result['change']==-.5 and result['by_demo_count']['3']['change']==.5
