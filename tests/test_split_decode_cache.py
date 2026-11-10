"""Split demo/target decode cache, sync-free stop checks and CUDA-graph replay."""
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F
from torch._dynamo.testing import CompileCounterWithBackend

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import evaluate
import launch_run
import target_demo_attention as tda
from common import END_PAIR_TOKEN_ID, compute_positions_3d, encode_context
from tinytransformer import TinyTransformer, TinyTransformerConfig
from test_group_attention import DEMOS, TARGET
from test_mtp import _batch, _set_path

CUDA = torch.cuda.is_available()


def model(seed=7, **overrides):
    torch.manual_seed(seed)
    settings = dict(d_model=24, n_heads=4, d_ff=48, n_layers=3, num_examples=2, num_task_embedding_latents=3,
                    task_embeddings=False, dropout=0, embedding_dropout=0, attention_dropout=0, n_predict=4,
                    attention_pattern="pair_group", split_target_demo_attention=True)
    return TinyTransformer(TinyTransformerConfig(**{**settings, **overrides})).eval()


SEQUENCES = [encode_context(DEMOS, TARGET), encode_context([], TARGET), encode_context(DEMOS[:1], TARGET)]
PROMPTS = [tokens[:sep + 1].tolist() for tokens, sep in SEQUENCES]
IDS = [(0, -1), (1, -1), (0, -1)]


@pytest.fixture
def padded_prefill(monkeypatch):
    # CPU prompts take the padded route even where FlashAttention is installed.
    monkeypatch.setattr(TinyTransformer, "_has_varlen_flash_attention", lambda self: False)


@pytest.fixture
def eager(monkeypatch, padded_prefill):
    monkeypatch.setattr(torch, "compile", lambda function, **kwargs: function)
    monkeypatch.setattr(evaluate, "_compiled_grid_update", evaluate._grid_update)


def explicit_attention(q, keys, values, lengths, causal):
    """Per-row, per-query softmax over the readable slots, written out by hand."""
    out = torch.zeros_like(q)
    for row in range(q.size(0)):
        for query in range(q.size(1)):
            readable = int(lengths[row]) - (q.size(1) - 1 - query if causal else 0)
            scores = torch.einsum("hd,lhd->hl", q[row, query], keys[row, :readable]) / q.size(-1) ** 0.5
            out[row, query] = torch.einsum("hl,lhd->hd", scores.softmax(-1), values[row, :readable])
    return out


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("width", [1, 4])
def test_reference_cache_attention_reads_only_filled_slots(causal, width):
    torch.manual_seed(0)
    q = torch.randn(3, width, 2, 8)
    keys, values = torch.randn(3, 16, 2, 8), torch.randn(3, 16, 2, 8)
    lengths = torch.tensor([5, 16, 9], dtype=torch.int32)
    expected = explicit_attention(q, keys, values, lengths, causal)
    torch.testing.assert_close(tda.reference_kv_cache_attention(q, keys, values, lengths, causal), expected)
    # Stale slots past each row's length never matter.
    stale = keys.clone()
    for row, length in enumerate(lengths.tolist()):
        stale[row, length:] = 1e4
    torch.testing.assert_close(tda.reference_kv_cache_attention(q, stale, values, lengths, causal), expected)


def test_split_prompt_cache_compacts_demo_and_target_keys_in_order():
    valid = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    demo = torch.tensor([[0, 0, 0, 1, 1, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    # Each key encodes its source position in every element.
    keys = torch.arange(8.).view(1, 1, 8, 1).expand(2, 2, 8, 3).contiguous()
    cache = [(keys, keys + 100)]
    layers, demo_lengths, target_lengths = tda.split_prompt_cache(cache, valid, demo, decode_slots=5, slot_multiple=4)
    assert cache == [None]  # converted layers are released
    demo_k, demo_v, target_k, target_v = layers[0]
    assert demo_lengths.tolist() == [2, 5] and target_lengths.tolist() == [4, 3]
    assert demo_lengths.dtype == target_lengths.dtype == torch.int32
    assert demo_k.shape == (2, 8, 2, 3) and target_k.shape == (2, 12, 2, 3)
    assert demo_k[0, :2, 0, 0].tolist() == [3, 4] and demo_k[1, :5, 0, 0].tolist() == [1, 2, 3, 4, 5]
    assert target_k[0, :4, 0, 0].tolist() == [2, 5, 6, 7] and target_k[1, :3, 0, 0].tolist() == [0, 6, 7]
    assert target_v[1, :3, 1, 2].tolist() == [100, 106, 107]
    assert demo_k.is_contiguous() and target_k.is_contiguous()


@pytest.mark.parametrize("block", [1, 2, 3])
@pytest.mark.parametrize("native", [False, True])
@torch.inference_mode()
def test_split_cache_steps_and_blocks_match_teacher_forcing(monkeypatch, native, block):
    m = model()
    _set_path(m, monkeypatch, "packed" if native else "padded")
    sequences = SEQUENCES[:2]
    full_batch = _batch(sequences=sequences)
    expected = m(**full_batch)["logits"]
    prompts = [tokens[:sep + 1] for tokens, sep in sequences]
    width = max(map(len, prompts)) + 3
    ids = torch.full((2, width), END_PAIR_TOKEN_ID, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for row, prompt in enumerate(prompts):
        ids[row, -len(prompt):] = prompt
        mask[row, -len(prompt):] = True
    pos = compute_positions_3d(ids, mask)
    prefill = m.forward_generate(ids, torch.arange(2), positions_3d=pos, attention_mask=mask)
    out_len = len(sequences[0][0]) - sequences[0][1] - 1
    demo = mask & (pos[..., 2] > 0) & (pos[..., 2] < pos[:, -1:, 2])
    layers, demo_lengths, target_lengths = tda.split_prompt_cache(
        list(prefill["past_key_values"]), mask, demo, decode_slots=out_len + block, slot_multiple=8)
    assert demo_lengths[0] > 0 and demo_lengths[1] == 0
    cache = (layers, demo_lengths, target_lengths, tda.KV_CACHE_SDPA)
    for start in range(0, out_len, block):
        count = min(block, out_len - start)
        new = torch.stack([tokens[sep + 1 + start:sep + 1 + start + count] for tokens, sep in sequences])
        positions = torch.stack([full_batch["positions_3d"][row, sep + 1 + start:sep + 1 + start + count]
                                 for row, (_, sep) in enumerate(sequences)])
        result = m.forward_generate(new, torch.arange(2), positions_3d=positions, split_cache=cache,
                                    return_hidden_states=True)
        target_lengths += count
        for row, (_, sep) in enumerate(sequences):
            torch.testing.assert_close(result["logits"][row], expected[row, sep + 1 + start:sep + 1 + start + count],
                                       rtol=3e-5, atol=2e-6)


@pytest.mark.parametrize("seed", [1, 7, 19])
@pytest.mark.parametrize("budget", [0, 1, 5, 24])
@torch.inference_mode()
def test_split_generation_matches_full_cache(eager, seed, budget):
    m = model(seed).to(torch.bfloat16)
    kwargs = dict(model=m, prompts=PROMPTS, example_ids=IDS, device=torch.device("cpu"), max_new_tokens=budget)
    full = evaluate.batched_greedy_generate(**kwargs, decode_cache="full")
    assert evaluate.batched_greedy_generate(**kwargs, decode_cache="split") == full
    assert evaluate.batched_greedy_generate(**kwargs, decode_cache="split", speculative=True) == full
    torch.manual_seed(3)
    sampled_full = evaluate.batched_greedy_generate(**kwargs, decode_cache="full", temperature=0.9, top_k=4)
    torch.manual_seed(3)
    assert evaluate.batched_greedy_generate(**kwargs, decode_cache="split", temperature=0.9, top_k=4) == sampled_full


@torch.inference_mode()
def test_split_generation_respects_model_length_limit(eager):
    m = model().to(torch.bfloat16)
    m.config.max_seq_len = max(map(len, PROMPTS)) + 6
    kwargs = dict(model=m, prompts=PROMPTS, example_ids=IDS, device=torch.device("cpu"), max_new_tokens=24)
    full = evaluate.batched_greedy_generate(**kwargs, decode_cache="full")
    assert evaluate.batched_greedy_generate(**kwargs, decode_cache="split") == full
    assert evaluate.batched_greedy_generate(**kwargs, decode_cache="split", speculative=True) == full


@torch.inference_mode()
def test_split_steps_compile_as_full_graphs(monkeypatch, padded_prefill):
    counter = CompileCounterWithBackend("aot_eager")
    real_compile = torch.compile
    monkeypatch.setattr(torch, "compile", lambda function, **kwargs: real_compile(function, backend=counter, fullgraph=True))
    m = model().to(torch.bfloat16)
    kwargs = dict(model=m, prompts=PROMPTS, example_ids=IDS, device=torch.device("cpu"), max_new_tokens=12)
    greedy = evaluate.batched_greedy_generate(**kwargs)
    assert evaluate.batched_greedy_generate(**kwargs, speculative=True) == greedy
    evaluate.batched_greedy_generate(**kwargs, temperature=0.8, top_k=3)
    assert counter.frame_count >= 3
    assert "_compiled_split_decode_step" in m.__dict__
    assert "_compiled_split_speculative_step" in m.__dict__


@torch.inference_mode()
def test_each_step_compiles_once_across_batch_shapes(monkeypatch, padded_prefill):
    # Rows, prompt padding and cache slots change per batch; one trace each must serve them all.
    counter = CompileCounterWithBackend("aot_eager")
    real_compile = torch.compile
    monkeypatch.setattr(torch, "compile", lambda function, **kwargs: real_compile(function, backend=counter, fullgraph=True))
    m = model().to(torch.bfloat16)
    for rows, padding in ((3, None), (2, 200), (3, 600), (2, 1100)):
        for speculative in (False, True):
            evaluate.batched_greedy_generate(m, PROMPTS[:rows], IDS[:rows], torch.device("cpu"), max_new_tokens=6,
                                             padded_prompt_length=padding, speculative=speculative)
    assert counter.frame_count == 2


def test_done_monitor_reads_the_cpu_flag_directly():
    flag = torch.zeros((), dtype=torch.bool)
    monitor = evaluate._DoneMonitor(flag)
    assert not monitor.done(0)
    flag.fill_(True)
    monitor.record(0)
    assert monitor.done(1)


def test_kernel_choice_and_override(monkeypatch):
    assert tda.choose_kv_cache_kernel(torch.device("cpu"), torch.bfloat16, 64, 3) == tda.KV_CACHE_SDPA
    monkeypatch.setenv("MDLARC_KV_CACHE_KERNEL", "bogus")
    with pytest.raises(ValueError, match="MDLARC_KV_CACHE_KERNEL"):
        tda.choose_kv_cache_kernel(torch.device("cpu"), torch.bfloat16, 64, 2)
    monkeypatch.setenv("MDLARC_KV_CACHE_KERNEL", "flash")
    with pytest.raises(RuntimeError, match="no FlashAttention cache kernel"):
        tda.choose_kv_cache_kernel(torch.device("cpu"), torch.bfloat16, 64, 2)


def test_split_cache_rejects_masks_and_unsplit_models():
    m = model(split_target_demo_attention=False)
    layers = tuple((torch.zeros(1, 4, 4, 6),) * 4 for _ in m.layer_schedule)
    cache = (layers, torch.ones(1, dtype=torch.int32), torch.ones(1, dtype=torch.int32), tda.KV_CACHE_SDPA)
    kwargs = dict(input_ids=torch.tensor([[1]]), example_ids=torch.tensor([0]),
                  positions_3d=torch.zeros(1, 1, 3, dtype=torch.int32), split_cache=cache)
    with pytest.raises(ValueError, match="split target/demo attention"):
        m.forward_generate(**kwargs)
    with pytest.raises(ValueError, match="attention_mask"):
        model().forward_generate(**kwargs, attention_mask=torch.ones(1, 4, dtype=torch.bool))


def test_launcher_defaults_greedy_decodes_to_speculative():
    plan = launch_run.build_plan({"evals": {"test": {"from": "pretrain.final"}}}, Path("test.json"))
    assert plan["evals"]["test"]["inference_speculative"] is True
    sampled = launch_run.build_plan({"defaults": {"inference_temperature": 0.7},
                                     "evals": {"test": {"from": "pretrain.final"}}}, Path("test.json"))
    assert sampled["evals"]["test"]["inference_speculative"] is False
    sampled_eval = launch_run.build_plan({"evals": {"test": {"from": "pretrain.final", "inference_temperature": 0.7}}},
                                         Path("test.json"))
    assert sampled_eval["evals"]["test"]["inference_speculative"] is False
    off = launch_run.build_plan({"eval_defaults": {"inference_speculative": False},
                                 "evals": {"test": {"from": "pretrain.final"}}}, Path("test.json"))
    assert off["evals"]["test"]["inference_speculative"] is False


# ---------------------------------------------------------------------------
# CUDA: FlashAttention cache kernels, graph replay, lagged stop checks.
# ---------------------------------------------------------------------------

def _flash_kernels():
    kernels = []
    if CUDA and tda._fa2_with_kvcache is not None:
        kernels.append(2)
    if CUDA and tda._fa3_with_kvcache is not None and torch.cuda.get_device_capability()[0] == 9:
        kernels.append(3)
    return kernels


@pytest.mark.skipif(not _flash_kernels(), reason="requires a FlashAttention cache kernel")
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("width", [1, 4])
def test_flash_cache_kernels_match_reference(causal, width):
    torch.manual_seed(0)
    q = torch.randn(5, width, 4, 64, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(5, 384, 4, 64, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    lengths = torch.tensor([7, 384, 129, 64, 300], device="cuda", dtype=torch.int32)
    expected = tda.reference_kv_cache_attention(q.float(), keys.float(), values.float(), lengths, causal)
    for kernel in _flash_kernels():
        actual = tda.flash_kv_cache_attention(q, keys, values, lengths, causal, kernel)
        torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)
        compiled = torch.compile(lambda *args: tda.flash_kv_cache_attention(*args), fullgraph=True)
        torch.testing.assert_close(compiled(q, keys, values, lengths, causal, kernel), actual)


@pytest.mark.skipif(not CUDA, reason="requires CUDA")
@torch.inference_mode()
def test_done_monitor_lags_on_cuda():
    flag = torch.zeros((), dtype=torch.bool, device="cuda")
    monitor = evaluate._DoneMonitor(flag)
    for index in range(monitor.LAG):
        assert not monitor.done(index)  # nothing is read before LAG steps
        flag.fill_(True)
        monitor.record(index)
    assert monitor.done(monitor.LAG)


@pytest.mark.skipif(not CUDA, reason="requires CUDA")
@pytest.mark.parametrize("seed", [1, 7])
@torch.inference_mode()
def test_cuda_graph_replay_matches_uncaptured_and_full_cache(seed):
    m = model(seed).to(device="cuda", dtype=torch.bfloat16)
    kwargs = dict(model=m, prompts=PROMPTS, example_ids=IDS, device=torch.device("cuda"), max_new_tokens=40)
    full = evaluate.batched_greedy_generate(**kwargs, decode_cache="full", cuda_graphs=False)
    full_graphed = evaluate.batched_greedy_generate(**kwargs, decode_cache="full", cuda_graphs=True)
    full_speculative = evaluate.batched_greedy_generate(**kwargs, decode_cache="full", cuda_graphs=True,
                                                        speculative=True)
    uncaptured = evaluate.batched_greedy_generate(**kwargs, cuda_graphs=False)
    speculative_uncaptured = evaluate.batched_greedy_generate(**kwargs, cuda_graphs=False, speculative=True)
    before = dict(evaluate.DECODE_GRAPH_STATS)
    graphed = evaluate.batched_greedy_generate(**kwargs, cuda_graphs=True)
    speculative = evaluate.batched_greedy_generate(**kwargs, cuda_graphs=True, speculative=True)
    assert evaluate.DECODE_GRAPH_STATS["captured"] == before["captured"] + 2
    assert evaluate.DECODE_GRAPH_STATS["failed"] == before["failed"]
    # Replay runs the captured kernels: bit-identical to the uncaptured step.
    assert graphed == uncaptured
    assert speculative == speculative_uncaptured
    # FlashAttention and the old masked SDPA may round differently; these tiny
    # models decode identically. Speculative vs ordinary is exact on the CPU
    # (above) but may flip GPU near-ties, as recorded on real checkpoints.
    assert graphed == full
    assert full_graphed == full  # the graphed single-cache step keeps the legacy loop's outputs
    assert len(full_speculative) == len(full)
