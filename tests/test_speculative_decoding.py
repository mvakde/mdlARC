"""Greedy verification, cache transactions, and token-dependent grid positions."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import evaluate
import launch_run
from common import END_PAIR_TOKEN_ID, IO_SEPARATOR_TOKEN_ID, NEXT_LINE_TOKEN_ID, START_TOKEN_ID, compute_positions_3d
from tinytransformer import TinyTransformer, TinyTransformerConfig, _cache_query_positions


@pytest.fixture
def eager(monkeypatch):
    # Control-flow tests don't need Inductor compilation. CUDA tests below do.
    monkeypatch.setattr(torch, "compile", lambda function, **kwargs: function)
    monkeypatch.setattr(evaluate, "_compiled_grid_update", evaluate._grid_update)


def small_model(seed=7, **kwargs):
    torch.manual_seed(seed)
    return TinyTransformer(TinyTransformerConfig(
        d_model=24, n_heads=4, d_ff=48, n_layers=2, max_seq_len=144,
        num_examples=3, num_task_embedding_latents=2, n_predict=4,
        dropout=0, embedding_dropout=0, attention_dropout=0, **kwargs,
    )).eval()


PROMPTS = [[START_TOKEN_ID, 1, IO_SEPARATOR_TOKEN_ID],
           [START_TOKEN_ID, 2, 3, NEXT_LINE_TOKEN_ID, 4, IO_SEPARATOR_TOKEN_ID]]


@pytest.mark.parametrize("accepted", range(4))
def test_commit_stops_at_first_mismatch_and_restores_state(accepted):
    draft = torch.tensor([[1, NEXT_LINE_TOKEN_ID, 3]])
    targets = torch.tensor([[1, NEXT_LINE_TOKEN_ID, 3, 4]])
    if accepted < 3:
        targets[0, accepted] = 8
    states, _ = evaluate._speculative_grid_positions(torch.tensor([[0, 0, 2]], dtype=torch.int32),
                                                     torch.cat([torch.tensor([[7]]), draft], dim=1))
    hidden = torch.arange(4).view(1, 4, 1).float()
    mask = torch.ones(1, 16, dtype=torch.bool)
    generated = torch.full((1, 14), END_PAIR_TOKEN_ID)
    anchor, state, lengths, emitted, finished, pending = evaluate._commit_speculative_block(
        draft, targets, hidden, states, torch.tensor([5]), mask, torch.tensor([1]),
        torch.tensor([False]), generated, 10)
    assert lengths.tolist() == [6 + accepted]
    assert emitted.tolist() == [2 + accepted]
    assert pending.item() == targets[0, accepted]
    assert anchor.item() == accepted
    assert torch.equal(state, states[:, accepted + 1])
    assert mask[0].tolist() == [True] * (6 + accepted) + [False] * (10 - accepted)
    assert generated[0, 1:2 + accepted].tolist() == draft[0, :accepted].tolist() + [pending.item()]
    assert not finished.item()


def test_commit_eos_budget_and_inactive_rows():
    draft = torch.tensor([[1, END_PAIR_TOKEN_ID, 3], [2, 3, 4], [4, 5, 6]])
    targets = torch.cat([draft, torch.tensor([[7], [8], [9]])], dim=1)
    generated = torch.full((3, 9), END_PAIR_TOKEN_ID)
    generated[2, :5] = torch.tensor([1, 2, 3, 4, 5])
    mask = torch.ones(3, 20, dtype=torch.bool)
    _, _, _, emitted, finished, _ = evaluate._commit_speculative_block(
        draft, targets, torch.randn(3, 4, 2), torch.zeros(3, 5, 3, dtype=torch.int32),
        torch.tensor([5, 5, 5]), mask, torch.tensor([1, 4, 5]),
        torch.tensor([False, False, True]), generated, 5)
    assert emitted.tolist() == [3, 5, 5]
    assert finished.tolist() == [True, True, True]
    assert generated[0, 1:4].tolist() == [1, END_PAIR_TOKEN_ID, END_PAIR_TOKEN_ID]
    assert generated[1, 4].item() == 2
    assert generated[2, :5].tolist() == [1, 2, 3, 4, 5]


@torch.inference_mode()
@pytest.mark.parametrize("per_row", [False, True])
def test_block_forward_matches_sequential_and_masks_future_keys(per_row):
    model = small_model()
    ids = torch.tensor([0, 1])
    prompt, mask = evaluate._left_pad_sequences(PROMPTS, END_PAIR_TOKEN_ID, torch.device("cpu"))
    prefill = model.forward_generate(prompt, ids, dihedral_ids=torch.tensor([0, 1]), attention_mask=mask,
                                     positions_3d=compute_positions_3d(prompt, mask))
    length, capacity = prompt.size(1), 20
    cache = tuple(tuple(torch.cat([v, v.new_zeros((*v.shape[:2], capacity - length, v.shape[-1]))], dim=2)
                        for v in pair) for pair in prefill["past_key_values"])
    serial_cache = tuple(tuple(v.clone() for v in pair) for pair in cache)
    full_mask = torch.zeros(2, capacity, dtype=torch.bool)
    full_mask[:, :length] = mask
    serial_mask = full_mask.clone()
    tokens = torch.tensor([[1, NEXT_LINE_TOKEN_ID, 4], [3, 5, 7]])
    starts = torch.tensor([length, length + 2 if per_row else length])
    # The second row may have holes, which remain invalid in both routes.
    slots = starts[:, None] + torch.arange(3)[None]
    state, _ = evaluate._derive_initial_state_from_prompt(prompt, compute_positions_3d(prompt, mask))
    _, positions = evaluate._speculative_grid_positions(state, tokens)
    full_mask.scatter_(1, slots, True)
    output = model.forward_generate(tokens, ids, dihedral_ids=torch.tensor([0, 1]), past_key_values=cache, cache_position=slots if per_row else slots[0],
                                     positions_3d=positions, attention_mask=full_mask, return_hidden_states=True)
    expected = []
    for index in range(3):
        current = slots[:, index:index + 1]
        serial_mask.scatter_(1, current, True)
        step = model.forward_generate(tokens[:, index:index + 1], ids, dihedral_ids=torch.tensor([0, 1]), past_key_values=serial_cache,
                                      cache_position=current if per_row else current[0],
                                      positions_3d=positions[:, index:index + 1], attention_mask=serial_mask)
        expected.append(step["logits"])
    torch.testing.assert_close(output["logits"], torch.cat(expected, dim=1), rtol=1e-5, atol=1e-6)
    for actual_pair, expected_pair in zip(cache, serial_cache):
        for actual, expected_tensor in zip(actual_pair, expected_pair):
            torch.testing.assert_close(actual, expected_tensor, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("seed", [1, 7, 19])
@pytest.mark.parametrize("budget", [0, 1, 2, 9, 16])
@torch.inference_mode()
def test_real_bf16_generation_matches_ordinary(eager, seed, budget):
    model = small_model(seed).to(torch.bfloat16)
    kwargs = dict(model=model, prompts=PROMPTS, example_ids=[0, 1], device=torch.device("cpu"), max_new_tokens=budget)
    assert evaluate.batched_greedy_generate(**kwargs, speculative=True) == evaluate.batched_greedy_generate(**kwargs)


class ScriptedModel(TinyTransformer):
    """Known target strings with different draft error positions per row."""
    def __init__(self, errors):
        super().__init__(small_model().config)
        self.to(torch.bfloat16)
        self.errors = errors
        self.register_buffer("answers", torch.tensor([
            [1, NEXT_LINE_TOKEN_ID, 2, 3, 4, 5, END_PAIR_TOKEN_ID, END_PAIR_TOKEN_ID],
            [4, 5, 6, NEXT_LINE_TOKEN_ID, 7, END_PAIR_TOKEN_ID, END_PAIR_TOKEN_ID, END_PAIR_TOKEN_ID],
        ]))
        self.calls = []

    def predict_mtp(self, hidden_states):
        offset = hidden_states[:, 0].long()[:, None] + torch.arange(1, 4)[None]
        draft = self.answers.gather(1, offset.clamp(max=7)).clone()
        for row, error in enumerate(self.errors):
            if error is not None:
                draft[row, error] = IO_SEPARATOR_TOKEN_ID  # wrong boundary would change z
        return draft

    def forward_generate(self, input_ids, example_ids, past_key_values=None, positions_3d=None,
                         attention_mask=None, cache_position=None, return_hidden_states=False, **kwargs):
        batch, width = input_ids.shape
        if past_key_values is None:
            self.prompt_len = width
            kv = torch.zeros(batch, 1, width, 1, dtype=torch.bfloat16)
            result = {"past_key_values": ((kv, kv.clone()),)}
            offsets = torch.zeros(batch, 1, dtype=torch.long)
        else:
            slots = _cache_query_positions(cache_position, batch, width)
            offsets = slots - self.prompt_len + 1
            for tensor in past_key_values[0]:
                tensor.scatter_(2, slots[:, None, :, None], input_ids[:, None, :, None].to(tensor.dtype))
            self.calls.append((slots.clone(), positions_3d.clone(), input_ids.clone()))
            result = {}
        next_tokens = self.answers.gather(1, offsets.clamp(max=7))
        logits = torch.full((batch, offsets.size(1), 14), -10.)
        result["logits"] = logits.scatter(2, next_tokens[:, :, None], 10.)
        if return_hidden_states:
            result["hidden_states"] = offsets[:, :, None].to(torch.bfloat16)
        return result


@pytest.mark.parametrize("errors", [(None, None), (0, 2), (1, 0)])
def test_heterogeneous_acceptance_matches_ordinary_and_grid_state(eager, errors):
    model = ScriptedModel(errors)
    kwargs = dict(model=model, prompts=PROMPTS, example_ids=[0, 1], device=torch.device("cpu"), max_new_tokens=12)
    actual = evaluate.batched_greedy_generate(**kwargs, speculative=True)
    speculative_calls = list(model.calls)
    assert actual == evaluate.batched_greedy_generate(**kwargs)
    if errors == (None, None):
        assert len(speculative_calls) == 2  # 7 and 6 output tokens including EOS
    for slots, positions, _ in speculative_calls:
        for row in range(2):
            offset = int(slots[row, 0]) - model.prompt_len
            if offset >= len(actual[row]) - len(PROMPTS[row]):
                continue
            true_positions = compute_positions_3d(torch.tensor(actual[row])[None])[0]
            assert torch.equal(positions[row, 0], true_positions[len(PROMPTS[row]) + offset])


def test_greedy_only_and_legacy_fallback(eager, monkeypatch):
    kwargs = dict(prompts=PROMPTS, example_ids=[0, 1], device=torch.device("cpu"), max_new_tokens=2)
    with pytest.raises(ValueError, match="greedy"):
        evaluate.batched_greedy_generate(model=small_model(), speculative=True, temperature=.7, **kwargs)
    model = TinyTransformer(replace(small_model().config, n_predict=1)).to(torch.bfloat16)
    monkeypatch.setattr(model, "predict_mtp", lambda *args: pytest.fail("Legacy model drafted"))
    assert evaluate.batched_greedy_generate(model=model, speculative=True, **kwargs) == evaluate.batched_greedy_generate(model=model, **kwargs)


def test_config_inheritance_and_validation():
    spec = {"defaults": {"inference_speculative": True}, "evals": {"test": {"from": "pretrain.final"}}}
    plan = launch_run.build_plan(spec, Path("test.json"))
    assert plan["evals"]["test"]["inference_speculative"] is True
    for invalid in (1, None, "true"):
        with pytest.raises(ValueError, match="inference_speculative"):
            launch_run.build_plan({"defaults": {"inference_speculative": invalid}}, Path("test.json"))
    with pytest.raises(ValueError, match="greedy"):
        launch_run.build_plan({"defaults": {"inference_speculative": True, "inference_temperature": .7}}, Path("test.json"))


def test_finished_rows_and_cache_capacity(eager):
    model = ScriptedModel((None, 0))
    model.config.max_seq_len = 9
    kwargs = dict(model=model, prompts=PROMPTS, example_ids=[0, 1], device=torch.device("cpu"), max_new_tokens=12)
    assert evaluate.batched_greedy_generate(**kwargs, speculative=True) == evaluate.batched_greedy_generate(**kwargs)


def test_block_mask_keeps_padding_and_rejected_slots_masked():
    model = small_model()
    mask = torch.zeros(2, 256, dtype=torch.bool)
    mask[0, 5:132] = True
    mask[1, 9:128] = True
    slots = torch.tensor([[128, 129, 130, 131], [124, 125, 126, 127]])
    block = model.build_decode_block_mask_for_step(mask, slots, 256, torch.device("cpu"))
    assert block.full_kv_num_blocks is None
    assert block.kv_num_blocks.tolist() == [[[2]]]
    assert not block.mask_mod(torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(129))
    assert block.mask_mod(torch.tensor(0), torch.tensor(0), torch.tensor(3), torch.tensor(131))
    assert not block.mask_mod(torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(4))
    assert not block.mask_mod(torch.tensor(1), torch.tensor(0), torch.tensor(3), torch.tensor(128))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_cuda_compiled_speculation_matches_ordinary_across_cache_boundary():
    model = small_model().to(device="cuda", dtype=torch.bfloat16)
    kwargs = dict(model=model, prompts=PROMPTS, example_ids=[0, 1], device=torch.device("cuda"),
                  padded_prompt_length=126, max_new_tokens=16)
    assert evaluate.batched_greedy_generate(**kwargs, speculative=True) == evaluate.batched_greedy_generate(**kwargs)


def test_verification_and_commit_compile_fullgraph():
    model = small_model()
    compiled = torch.compile(model.forward_generate, backend="aot_eager", fullgraph=True)
    cache = tuple((torch.zeros(2, 4, 12, 6), torch.zeros(2, 4, 12, 6)) for _ in model.blocks)
    kwargs = dict(input_ids=torch.tensor([[1, 2], [3, 4]]), example_ids=torch.tensor([0, 1]),
                  dihedral_ids=torch.tensor([0, 1]),
                  past_key_values=cache, positions_3d=torch.zeros(2, 2, 3, dtype=torch.int32),
                  attention_mask=torch.ones(2, 12, dtype=torch.bool),
                  cache_position=torch.tensor([[3, 4], [5, 6]]), return_hidden_states=True)
    result = compiled(**kwargs)
    assert torch.isfinite(result["logits"]).all()
    args = (torch.tensor([[1], [2]]), torch.tensor([[1, 3], [2, 4]]), result["hidden_states"],
            torch.zeros(2, 3, 3, dtype=torch.int32), torch.tensor([3, 5]),
            kwargs["attention_mask"], torch.ones(2, dtype=torch.long), torch.zeros(2, dtype=torch.bool),
            torch.full((2, 8), END_PAIR_TOKEN_ID), 4)
    torch.compile(evaluate._commit_speculative_block, backend="aot_eager", fullgraph=True)(*args)


def test_verify_has_its_own_compile_cache():
    # A block and a single token must coexist even with one entry per route.
    torch._dynamo.reset()
    model = small_model()
    cache = tuple((torch.zeros(2, 4, 12, 6), torch.zeros(2, 4, 12, 6)) for _ in model.blocks)
    mask = torch.zeros(2, 12, dtype=torch.bool)
    mask[:, :4] = True
    shared = dict(example_ids=torch.tensor([0, 1]), dihedral_ids=torch.tensor([0, 1]),
                  past_key_values=cache, attention_mask=mask)
    try:
        with torch.inference_mode(), torch._dynamo.config.patch(recompile_limit=1):
            decode = torch.compile(model.forward_generate, backend="aot_eager", fullgraph=True)
            verify = torch.compile(model.forward_verify, backend="aot_eager", fullgraph=True)
            serial = decode(input_ids=torch.tensor([[1], [2]]), cache_position=torch.tensor([3]),
                            positions_3d=torch.zeros(2, 1, 3, dtype=torch.int32), **shared)
            mask[:, 4] = True
            block = verify(input_ids=torch.tensor([[1, 3], [2, 4]]), cache_position=torch.tensor([[3, 4], [3, 4]]),
                           positions_3d=torch.zeros(2, 2, 3, dtype=torch.int32), **shared)
            torch.testing.assert_close(serial["logits"], block["logits"][:, :1], rtol=1e-5, atol=1e-6)
    finally:
        torch._dynamo.reset()
