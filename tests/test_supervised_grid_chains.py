import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from common import (ARCExampleDataset, END_PAIR_TOKEN_ID, IO_SEPARATOR_TOKEN_ID,
                    SUPERVISED_GRID_CHAIN_SEQUENCE_FORMAT, compute_positions_3d, encode_context)
from grid_chain import balanced_grid_weights, final_grid_mask
from group_attention import role_positions
from tinytransformer import TinyTransformer, TinyTransformerConfig


def model():
    torch.manual_seed(3)
    return TinyTransformer(TinyTransformerConfig(d_model=48, n_heads=4, d_ff=96, n_layers=2,
        num_examples=1, task_embeddings=False, dihedral_embeddings=False,
        dropout=0, embedding_dropout=0, attention_dropout=0, attention_pattern="pair_group",
        split_target_demo_attention=True, ordered_grid_positions=True,
        predict_intermediate_grids=True, n_predict=4, FA=2))


def data(tmp_path):
    demo = {"input": [[1]], "output": [[4]], "intermediate_grid_candidates": [[[2]], [[3]]]}
    other = {"input": [[5]], "output": [[6]], "intermediate_grid_candidates": []}
    target = {"input": [[7]], "output": [[0, 1]], "intermediate_grid_candidates": [[[8]], [[9]]]}
    p = tmp_path / "data.json"
    p.write_text(json.dumps({"t": {"train": [demo, other, target], "test": [{"input": [[7]]}]}}))
    return ARCExampleDataset(p, splits=("train", "test"), seed=42,
        attention_pattern="pair_group", training_demo_schedule="two", inference_demo_policy="two",
        predict_intermediate_grids=True)


def test_equal_grid_and_example_weighting_in_padded_and_packed_targets():
    # One tiny intermediate and one large final grid each receive half of
    # example 0's loss. Example 1 has no intermediate and gets equal sample weight.
    targets = torch.tensor([[2, 12, 3, 3, 3, 3, 3, 13], [4, 13, -100, -100, -100, -100, -100, -100]])
    valid = targets >= 0
    losses = torch.tensor([[2., 2., 6., 6., 6., 6., 6., 6.], [10., 10., 0., 0., 0., 0., 0., 0.]], requires_grad=True)
    weights, units = balanced_grid_weights(targets, valid)
    assert units == 2
    torch.testing.assert_close((weights * losses).sum(), torch.tensor(7.))
    torch.testing.assert_close(weights.sum(1), torch.tensor([.5, .5]))
    packed_weights, packed_units = balanced_grid_weights(targets.flatten(), valid.flatten(), torch.tensor([0, 8, 16]))
    torch.testing.assert_close(weights.flatten(), packed_weights)
    assert packed_units == units
    final = final_grid_mask(targets, valid)
    assert final[0].tolist() == [False, False, True, True, True, True, True, True]
    torch.testing.assert_close(final.flatten(), final_grid_mask(targets.flatten(), valid.flatten(), torch.tensor([0, 8, 16])))
    (weights * losses).sum().backward()
    assert torch.isfinite(losses.grad).all()
    empty, count = balanced_grid_weights(targets, torch.zeros_like(valid))
    assert not empty.any() and count == 0


def test_training_target_has_trace_but_inference_and_palette_do_not_leak_it(tmp_path):
    from context_palette import protection_schedule
    ds = data(tmp_path)
    ex = ds.examples[2]
    tokens, sep, _ = ds.build_training_sequence(ex, 0, None)
    assert tokens[sep + 1:].tolist() == [8, 12, 9, 12, 0, 1, 13]
    prompt, _ = ds.build_context_sequence(ex, {"input": [[7]]}, 0, None, visit=0)
    assert 8 not in prompt and 9 not in prompt
    assert {8, 9}.issubset(protection_schedule(ds, ex, 1)[0])
    assert ds.max_seq_len == 11161
    positions = compute_positions_3d(tokens[None])
    roles = role_positions(positions, positions[:, sep, 2:3], tokens[None], ordered=True, target_chain=True)
    assert roles[0, sep + 1:, 2].tolist() == [6, 6, 7, 7, 8, 8, 8]
    test = next(ds.iter_examples(split="test"))
    query, _ = ds.build_context_sequence(test, {"input": [[7]]}, 0, None, visit=0)
    assert query[-1] == IO_SEPARATOR_TOKEN_ID


@pytest.mark.parametrize("middle_count", [0, 1, 2])
def test_target_chain_is_causal_and_cached_decode_matches_training(tmp_path, middle_count):
    from build import _validate_checkpoint_format
    from evaluate import _derive_initial_state_from_prompt, _speculative_grid_positions
    ds = data(tmp_path)
    pair = {"input": [[7]], "intermediate_grids": [[[8]], [[9]]][:middle_count], "output": [[0, 1]]}
    tokens, sep = encode_context([ds.pairs_by_task["t"][1]], pair, predict_intermediate_grids=True)
    positions = compute_positions_3d(tokens[None])
    net = model().eval()
    _validate_checkpoint_format({"config": asdict(net.config), "sequence_format": SUPERVISED_GRID_CHAIN_SEQUENCE_FORMAT})
    with pytest.raises(ValueError, match="own checkpoint"):
        _validate_checkpoint_format({"config": asdict(net.config), "sequence_format": "intermediate_grid_pair_group_two_lookup_v1"})
    args = dict(input_ids=tokens[None], example_ids=torch.tensor([0]), positions_3d=positions,
                sep_indices=torch.tensor([sep]), targets=tokens[None], compute_training_metrics=True)
    full = net(**args)
    assert torch.isfinite(full["loss"]) and full["grid_loss_units"] == 1
    # Metrics cover only the three final-grid tokens, not intermediate colors.
    assert full["training_metric_counts"][1] == 2
    assert full["training_metric_counts"][3] == 1
    changed = tokens.clone(); changed[-2] = 4
    altered = net(**{**args, "input_ids": changed[None], "targets": changed[None]})
    torch.testing.assert_close(full["logits"][:, :-2], altered["logits"][:, :-2])
    width, capacity = sep + 1, len(tokens)
    pref = net.forward_generate(tokens[:width][None], torch.tensor([0]), positions_3d=positions[:, :width])
    cache = tuple(tuple(torch.cat((t, t.new_zeros((*t.shape[:2], capacity - width, t.size(-1)))), dim=2)
                        for t in layer) for layer in pref["past_key_values"])
    state, _ = _derive_initial_state_from_prompt(tokens[:width][None], positions[:, :width])
    _, output_positions = _speculative_grid_positions(state, tokens[width:][None])
    decoded = net.forward_generate(tokens[width:][None], torch.tensor([0]), past_key_values=cache,
        positions_3d=output_positions, attention_mask=torch.ones((1, capacity), dtype=torch.bool),
        demo_attention_mask=(positions[..., 2] > 0) & (positions[..., 2] < positions[:, sep, 2:3]),
        cache_position=torch.arange(width, capacity), target_input_z=positions[:, sep, 2:3])
    torch.testing.assert_close(full["logits"][:, width:], decoded["logits"], atol=1e-5, rtol=1e-4)
    full["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_generated_evaluation_scores_only_last_grid_and_oracle_refuses_queries(tmp_path, monkeypatch):
    import evaluate
    ds = data(tmp_path); net = model()
    monkeypatch.setattr(evaluate, "_prepare_model_for_inference", lambda m: m)
    monkeypatch.setattr(evaluate, "batched_greedy_generate", lambda **kw: [p + [8, 12, 9, 12, 0, 1, 13] for p in kw['prompts']])
    result = evaluate.run_split_inference(net, ds, "test", torch.device("cpu"), batch_size=1)[0]
    assert result['output_grid'] == [[0, 1]] and result['intermediate_grids'] == [[[8]], [[9]]]
    with pytest.raises(ValueError, match="never test queries"):
        evaluate.run_split_inference(net, ds, "test", torch.device("cpu"), oracle_intermediates=True)


def test_gold_intermediates_are_consumed_through_causal_decode(tmp_path):
    from evaluate import batched_greedy_generate
    net = model().eval()
    # Force the ordinary head to terminate immediately. Oracle intermediates
    # must nevertheless enter the cache, followed by the model's own END token.
    with torch.no_grad():
        net.lm_head = torch.nn.Linear(net.config.d_model, net.config.vocab_size, bias=True)
        net.lm_head.weight.zero_(); net.lm_head.bias.fill_(-10); net.lm_head.bias[13] = 10
    prompt, _ = encode_context([], {"input": [[1]]})
    generated = batched_greedy_generate(net, [prompt.tolist()], [(0, -1)], torch.device("cpu"),
        max_new_tokens=12, compile_decode=False, speculative="auto", forced_prefixes=[[2, 12, 3, 12]])[0]
    assert generated[len(prompt):] == [2, 12, 3, 12, 13]


def test_native_packed_and_padded_chain_losses_and_metrics_match(monkeypatch):
    import torch.nn.functional as F
    net = model().eval()
    sequences, seps = zip(*(encode_context([], p, predict_intermediate_grids=True) for p in [
        {"input": [[1]], "intermediate_grids": [[[2]], [[3, 3, 3]]], "output": [[4]]},
        {"input": [[5]], "output": [[6, 7]]}]))
    lengths = [len(s) for s in sequences]; width = max(lengths)
    inputs = torch.full((2, width), 13, dtype=torch.long); mask = torch.zeros_like(inputs, dtype=torch.bool)
    for i, sequence in enumerate(sequences):
        inputs[i, :len(sequence)] = sequence; mask[i, :len(sequence)] = True
    positions = compute_positions_3d(inputs, mask)
    args = dict(example_ids=torch.zeros((2, 2), dtype=torch.long), sep_indices=torch.tensor(seps), compute_training_metrics=True)
    padded = net(input_ids=inputs, attention_mask=mask, positions_3d=positions, targets=inputs, **args)
    def reference(queries, keys, values, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p, is_causal):
        q,k,v = queries, keys, values
        out = []
        for i in range(len(cu_seqlens_q) - 1):
            a,b = cu_seqlens_q[i:i+2]; c,d = cu_seqlens_k[i:i+2]
            qa,ka,va = q[a:b].transpose(0,1), k[c:d].transpose(0,1), v[c:d].transpose(0,1)
            causal = torch.arange(d-c)[None] <= torch.arange(b-a)[:,None] + (d-c)-(b-a) if is_causal else None
            out.append(F.scaled_dot_product_attention(qa, ka, va, attn_mask=causal).transpose(0,1))
        return torch.cat(out)
    monkeypatch.setattr(net, "_has_varlen_flash_attention", lambda: True)
    for block in net.blocks:
        monkeypatch.setattr(block.attention, "_apply_varlen_flash_attention", reference)
    packed = net(input_ids=inputs[mask], positions_3d=positions[mask],
        cu_seqlens=torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int32), max_seqlen=width, **args)
    for name in ("loss", "output_loss", "mtp_loss", "training_metric_counts", "grid_loss_units"):
        torch.testing.assert_close(padded[name], packed[name], atol=1e-5, rtol=1e-4)


def test_diagnostic_runs_same_withheld_pairs_with_generated_and_gold_intermediates(tmp_path, monkeypatch):
    import evaluate
    from dataset_building_scripts.intermediate_grids import pair_key
    ds = data(tmp_path); net = model(); monkeypatch.chdir(tmp_path)
    (tmp_path/'dataset_manifest.json').write_text(json.dumps({'diagnostic_holdouts':[
        {'task':'t','pair_index':0,'io_sha256':pair_key(ds.pairs_by_task['t'][0])}]}))
    monkeypatch.setattr(evaluate, 'load_checkpoint', lambda _: {})
    monkeypatch.setattr(evaluate, 'build_model_and_data', lambda *a,**k: (net,ds,None,torch.device('cpu'),ds.source_path))
    calls=[]
    def inference(*args, **kwargs):
        calls.append(kwargs)
        return [{'task_id':'t','pair_index':0,'output_grid':[[4]] if kwargs['oracle_intermediates'] else [[0]]}]
    monkeypatch.setattr(evaluate, 'run_split_inference', inference)
    _, result, path=evaluate.run_evaluation(SimpleNamespace(mode='intermediate-diagnostic', enable_aug=False,
        seed=42, diagnostic_cases=1), run_name='diagnostic', max_augments=1, data_path=ds.source_path,
        checkpoint_path=tmp_path/'checkpoint.pt')
    assert [c['oracle_intermediates'] for c in calls] == [False, True]
    assert all(c['task_ids']==['t'] and c['pair_index']==0 for c in calls)
    assert result['gold_intermediate_accuracy_gain']==1.0 and result['held_out_from_pretraining']
    assert path.name=='intermediate_diagnostic.json' and not result['official_arc_score']


def test_preparation_withholds_exact_io_globally_preserving_official_queries(tmp_path):
    import zipfile
    from dataset_building_scripts.intermediate_grids import encoded, sha, pair_key, reserve_diagnostic_pairs
    pairs=[{'input':[[i]],'output':[[i+1]],'intermediate_grid_candidates':[[[i+2]]]} for i in range(4)]
    training={'task':{'train':pairs},'duplicate':{'train':[pairs[0],pairs[1]]}}
    evaluation={'task':{'train':pairs,'test':[{'input':[[9]]}]}}
    raw=encoded(training); original_eval=encoded(evaluation)
    payloads={'challenges.json':raw,'priv_challenges.json':original_eval,'solutions.json':encoded({'task':[[[8]]]}),
        'dataset_manifest.json':encoded({'training_sha256':sha(raw),'summary':{}})}
    source=tmp_path/'assets_tmp/source.zip'; source.parent.mkdir()
    with zipfile.ZipFile(source,'w') as z:
        for k,v in payloads.items():z.writestr(k,v)
    identity={'archive_path':'assets_tmp/source.zip','size':source.stat().st_size,'sha256':sha(source.read_bytes())}
    output=tmp_path/'assets_tmp/reserved.zip'
    new=reserve_diagnostic_pairs(identity, root=tmp_path, package=output, count=1)
    with zipfile.ZipFile(output) as z:
        remaining=json.loads(z.read('challenges.json')); manifest=json.loads(z.read('dataset_manifest.json'))
        assert z.read('priv_challenges.json')==original_eval and z.read('solutions.json')==payloads['solutions.json']
    assert manifest['summary']['heldout_occurrences_removed']==2
    assert all(pair_key(p)!=pair_key(pairs[0]) for t in remaining.values() for p in t['train'])
    assert source.read_bytes() and sha(source.read_bytes())==identity['sha256'] and new!=identity


def test_target_chain_speculation_matches_greedy_and_caps_intermediates():
    from evaluate import batched_greedy_generate, _limit_target_grids
    net=model().eval()
    first,_=encode_context([], {'input':[[1]]})
    second,_=encode_context([{'input':[[2]],'intermediate_grids':[[[3]],[[4]]],'output':[[5]]}], {'input':[[6]]})
    kw=dict(model=net,prompts=[first.tolist(),second.tolist()],example_ids=[(0,-1),(0,-1)],
        device=torch.device('cpu'),max_new_tokens=18,compile_decode=False)
    assert batched_greedy_generate(**kw,speculative=False)==batched_greedy_generate(**kw,speculative=True)
    logits=torch.zeros(2,3,14); z=torch.tensor([[6,7,8],[2,3,4]])
    capped=_limit_target_grids(logits,z,torch.tensor([[5],[1]]))
    assert not torch.isinf(capped[:,:2]).any() and torch.isneginf(capped[:,2,12]).all()
