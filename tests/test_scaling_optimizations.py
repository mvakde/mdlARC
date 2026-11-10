"""Optimized two-lookup training and decoding against independent references."""
import copy
import json
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import evaluate
import tinytransformer
import train
from common import ARCExampleDataset, collate_examples, compute_positions_3d, encode_context
from target_demo_attention import PreparedLayout, build_split_layout, prepare_decode_cache
from test_group_attention import DEMOS, TARGET, training_compile_settings
from test_mtp import _batch, _reference_flash, _set_path
from test_target_demo_attention import BRANCH_OPTIONS, model


@pytest.mark.parametrize('count', [0, 1, 3])
@pytest.mark.parametrize('compiled', [False, True])
@pytest.mark.parametrize('options', BRANCH_OPTIONS)
def test_prepared_training_matches_dense_values_and_all_gradients(monkeypatch, count, compiled, training_compile_settings, options):
    actual, expected = model(**options), model(**options)
    sequences = [encode_context((DEMOS * 3)[:count], TARGET), encode_context([], TARGET)]
    packed = _batch(packed=True, sequences=sequences)
    layout = build_split_layout(packed['input_ids'], packed['cu_seqlens'], packed['sep_indices'] + 1)
    packed['target_demo_layout'] = layout
    _set_path(actual, monkeypatch, 'packed')
    _set_path(expected, monkeypatch, 'padded')
    monkeypatch.setattr(actual, '_build_packed_prefix_lm_layout',
        lambda *a, **kw: pytest.fail('Prepared layout was rebuilt inside the model'))
    calls, gate_rows = [], []
    for block in actual.blocks:
        def attend(*args, **kwargs):
            calls.append(kwargs.get('is_causal', args[-1] if args else None))
            return _reference_flash(*args, **kwargs)
        monkeypatch.setattr(block.attention, '_apply_varlen_flash_attention', attend)
        block.attention.target_demo_gate.register_forward_pre_hook(lambda module, args: gate_rows.append(len(args[0])))
    runner = torch.compile(actual, backend='aot_eager') if compiled else actual
    result = runner(**packed, compute_training_metrics=True)
    reference = expected(**_batch(sequences=sequences), compute_training_metrics=True)
    flat = torch.cat([reference['logits'][row, :len(tokens)] for row, (tokens, _) in enumerate(sequences)])
    torch.testing.assert_close(result['logits'], flat, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(result['loss'], reference['loss'])
    torch.testing.assert_close(result['training_metric_counts'], reference['training_metric_counts'])
    result['loss'].backward(); reference['loss'].backward()
    for name, param in actual.named_parameters():
        if param.requires_grad:
            target = dict(expected.named_parameters())[name].grad
            if param.grad is None:
                # With no demos the packed reference has always omitted the
                # gate entirely; dense masking produces an explicit zero.
                assert count == 0 and '.target_demo_gate.' in name
                assert target is not None and not target.any()
            else:
                torch.testing.assert_close(param.grad, target, rtol=5e-4, atol=4e-6)
    if not compiled:
        assert len(calls) == actual.config.n_layers * (3 if count else 2)
        assert gate_rows == ([len(layout['cross']['q'])] * actual.config.n_layers if count else [])


def test_collation_prepares_final_augmented_tokens_and_device_transfer(tmp_path):
    path = tmp_path / 'tasks.json'
    path.write_text(json.dumps({'a': {'train': DEMOS + [TARGET]}}))
    dataset = ARCExampleDataset(path, attention_pattern='pair_group', training_demo_schedule='half_or_more')
    batch = collate_examples(dataset.examples[:2], dataset)
    assert isinstance(batch['target_demo_layout'], PreparedLayout)
    moved = train._move_batch_to_device(batch, torch.device('cpu'))
    assert isinstance(moved['target_demo_layout'], dict)
    reference = build_split_layout(batch['input_ids'], batch['cu_seqlens'], batch['sep_indices'] + 1)
    for part in reference:
        for key, value in reference[part].items():
            if torch.is_tensor(value):
                assert torch.equal(moved['target_demo_layout'][part][key], value)
            else:
                assert moved['target_demo_layout'][part][key] == value


@pytest.mark.parametrize('count', [0, 1, 3])
@pytest.mark.parametrize('compiled', [False, True])
@pytest.mark.parametrize('options', BRANCH_OPTIONS)
@torch.inference_mode()
def test_prepared_decode_block_matches_full_cache(monkeypatch, count, compiled, options):
    m = model(**options).eval()
    _set_path(m, monkeypatch, 'padded')
    for block in m.blocks:
        torch.nn.init.normal_(block.attention.target_demo_gate.weight, std=.2)
    sequences = [encode_context((DEMOS * 3)[:count], TARGET), encode_context([], TARGET)]
    prompts = [tokens[:sep+1].tolist() for tokens, sep in sequences]
    ids, mask = evaluate._left_pad_sequences(prompts, 13, torch.device('cpu'))
    positions = compute_positions_3d(ids, mask)
    examples = torch.tensor([0, 1])
    initial = m.forward_generate(ids, examples, positions_3d=positions, attention_mask=mask)
    demos = mask & (positions[..., 2] > 0) & (positions[..., 2] < positions[:, -1:, 2])
    budget = 8
    full_cache = tuple(tuple(F.pad(t, (0, 0, 0, budget)) for t in layer) for layer in initial['past_key_values'])
    valid = F.pad(mask, (0, budget)); membership = F.pad(demos, (0, budget))
    local_cache, frozen, local_valid, local_start = prepare_decode_cache(
        initial['past_key_values'], demos, mask, budget, [b.attention for b in m.blocks])
    # There is one concatenated segment per row, never one softmax per demo.
    if count:
        assert frozen[0]['owners'] == (0,)
        assert frozen[0]['lengths'] == (int(demos[0].sum()),)
    tokens = torch.stack([t[sep+1:sep+4] for t, sep in sequences])
    pos = torch.stack([compute_positions_3d(t[None])[0, sep+1:sep+4] for t, sep in sequences])
    full_slots = (torch.arange(3) + ids.size(1))[None].expand(2, -1)
    local_slots = (torch.arange(3) + local_start)[None].expand(2, -1)
    valid.scatter_(1, full_slots, True); local_valid.scatter_(1, local_slots, True)
    expected = m.forward_generate(tokens, examples, past_key_values=full_cache, positions_3d=pos,
        attention_mask=valid, cache_position=full_slots, demo_attention_mask=membership)
    runner = torch.compile(m.forward_generate, backend='aot_eager', fullgraph=True) if compiled else m.forward_generate
    actual = runner(tokens, examples, past_key_values=local_cache, positions_3d=pos,
        attention_mask=local_valid, cache_position=local_slots,
        target_decode_cache=tuple({'demo': cache} for cache in frozen))
    torch.testing.assert_close(actual['logits'], expected['logits'], rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize('budget', [0, 1, 9])
@pytest.mark.parametrize('seed', [7, 19])
@torch.inference_mode()
def test_real_bf16_generation_variants_preserve_tokens(monkeypatch, budget, seed):
    torch.manual_seed(seed)
    m = model().eval().to(torch.bfloat16)
    prompts = [t[:s+1].tolist() for t, s in [encode_context(DEMOS, TARGET), encode_context([], TARGET)]]
    kwargs = dict(model=m, prompts=prompts, example_ids=[0, 1], device=torch.device('cpu'),
        max_new_tokens=budget, compile_decode=False)
    reference = evaluate.batched_greedy_generate(**kwargs, use_prepared_cache=False, compact_finished=False)
    assert evaluate.batched_greedy_generate(**kwargs) == reference
    assert evaluate.batched_greedy_generate(**kwargs, speculative='auto') == reference


def test_split_decode_skips_unused_flex_mask():
    m = model()
    m.build_decode_block_mask_for_step = lambda *a: pytest.fail('Unused Flex mask built')
    assert evaluate._decode_mask(m, torch.ones(1, 5, dtype=torch.bool), torch.tensor([2]), torch.device('cuda')) is None


def test_fa4_adapter_keywords_return_and_no_backend_fallback(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode
    calls = []
    def backend(q, k, v, **kwargs):
        calls.append(kwargs)
        return q, None
    monkeypatch.setattr(tinytransformer, 'flash_attn4_varlen_func', backend)
    attention = tinytransformer.MultiHeadSelfAttention(replace(model().config, FA=4), 0)
    assert attention._has_selected_varlen_flash_attention()
    with FakeTensorMode():
        q = torch.empty(3, 2, 64, dtype=torch.bfloat16, device='cuda')
        cu = torch.empty(2, dtype=torch.int32, device='cuda')
        result = attention._apply_varlen_flash_attention(q, q, q, cu, cu, 3, 3, 0., True)
        assert result is q
        assert calls[0]['cu_seqlens_q'] is cu and calls[0]['causal'] is True
        assert calls[0]['return_lse'] is False and 'dropout_p' not in calls[0]
        with pytest.raises(RuntimeError, match='dropout'):
            attention._apply_varlen_flash_attention(q, q, q, cu, cu, 3, 3, .1, True)
        attention._has_flash_attn4_varlen_func = False
        assert not attention._has_selected_varlen_flash_attention()
        with pytest.raises(RuntimeError, match='requires flash-attn-4'):
            attention._apply_varlen_flash_attention(q, q, q, cu, cu, 3, 3, 0., True)
    with pytest.raises(ValueError, match='attention_dropout'):
        replace(model().config, FA=4, attention_dropout=.1)


def test_default_fa4_recipe_preserves_reference_except_authorized_changes():
    import launch_run
    root = Path(__file__).resolve().parents[1]
    baseline = json.loads((root/'specs/context_palette_schedule_two_lookup.json').read_text())
    for name in ['pretrain_eval', 'example']:
        updated = json.loads((root/f'specs/{name}.json').read_text())
        plan = launch_run.build_plan(updated, root/f'specs/{name}.json')
        assert plan['pretrain']['FA'] == plan['evals']['final']['FA'] == 4
        assert plan['pretrain']['attention_dropout'] == 0
        updated.pop('run_name'); reference = copy.deepcopy(baseline); reference.pop('run_name')
        for key in ['FA', 'attention_dropout', 'inference_speculative']:
            updated['defaults'].pop(key, None); reference['defaults'].pop(key, None)
        for key in ['FA', 'inference_speculative']:
            updated['eval_defaults'].pop(key, None); reference['eval_defaults'].pop(key, None)
        assert updated == reference
    with pytest.raises(ValueError, match='attention_dropout'):
        launch_run.build_plan({'defaults': {'FA': 4}}, Path('invalid.json'))


def test_fa4_runtime_stage_selection_and_colab_snapshot_without_fa2_wheel(tmp_path, monkeypatch):
    from attention_runtime import selected_versions
    from scripts import run_on_colab
    spec = {'defaults': {'FA': 4}, 'enable_finetune': False}
    assert selected_versions(spec) == {4}
    assert selected_versions({**spec, 'eval_defaults': {'FA': 2}}) == {2, 4}
    assert selected_versions({'eval_only': True, 'eval_defaults': {'FA': 4}}) == {4}
    for name, content in {'launch_run.py': '', 'requirements.txt': '', 'scripts/run_layout.py': '',
                          'requirements-fa4.txt': 'flash-attn-4[cu13]==4.0.0b36',
                          'specs/test.json': json.dumps(spec)}.items():
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(content)
    monkeypatch.setattr(run_on_colab, 'ROOT', tmp_path)
    files = run_on_colab.source_files(tmp_path/'specs/test.json')
    assert tmp_path/'requirements-fa4.txt' in files
    assert not any('wheels' in path.parts for path in files)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='FA4 native Blackwell/Hopper validation requires CUDA')
@pytest.mark.parametrize('options', BRANCH_OPTIONS)
def test_native_fa4_compiled_prepared_training_matches_dense(options):
    if tinytransformer.flash_attn4_varlen_func is None:
        pytest.skip('flash-attn-4 is not installed')
    from attention_runtime import check_fa4
    check_fa4()
    torch.manual_seed(42)
    config = tinytransformer.TinyTransformerConfig(d_model=128, n_heads=2, d_ff=256, n_layers=2,
        num_examples=2, num_task_embedding_latents=2, task_embeddings=False, n_predict=4,
        dropout=0, embedding_dropout=0, attention_dropout=0, FA=4,
        attention_pattern='pair_group', split_target_demo_attention=True, **options)
    actual = tinytransformer.TinyTransformer(config).cuda()
    reference = copy.deepcopy(actual)
    sequences = [encode_context(DEMOS, TARGET), encode_context([], TARGET)]
    batch = _batch(packed=True, sequences=sequences)
    batch['target_demo_layout'] = PreparedLayout(build_split_layout(batch['input_ids'], batch['cu_seqlens'], batch['sep_indices']+1))
    batch = train._move_batch_to_device(batch, torch.device('cuda'))
    dense = {k: v.cuda() if torch.is_tensor(v) else v for k, v in _batch(sequences=sequences).items()}
    train._enable_compile_scalar_capture()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        result = torch.compile(actual)(**batch)
        expected = reference(**dense)
    torch.testing.assert_close(result['loss'], expected['loss'], rtol=.03, atol=.02)
    result['loss'].backward(); expected['loss'].backward()
    for name, parameter in actual.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad, rtol=.1, atol=.05)
