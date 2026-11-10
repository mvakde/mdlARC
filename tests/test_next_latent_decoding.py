"""NextLat drafting alignment and exact greedy verification with cache rollback."""
from dataclasses import replace

import pytest
import torch

from test_speculative_decoding import PROMPTS, ScriptedModel, eager, small_model
import evaluate
from common import END_PAIR_TOKEN_ID, compute_positions_3d
from tinytransformer import TinyTransformer


def latent_model(horizon=4, **options):
    return TinyTransformer(replace(small_model().config, n_predict=1,
                                  next_latent_prediction=True,
                                  next_latent_horizon=horizon, **options)).eval()


@pytest.mark.parametrize("horizon", [1, 4, 8])
@torch.inference_mode()
def test_recursive_proposals_consume_pending_then_predicted_tokens(horizon):
    model = latent_model(horizon).to(torch.bfloat16)
    anchor = torch.randn(2, 24, dtype=torch.bfloat16)
    pending = torch.tensor([3, 7])
    state, token, expected = anchor.clone(), pending.clone(), []
    for _ in range(horizon):
        state = model.next_latent_dynamics(state, model.token_embedding(token))
        token = model.lm_head(state).argmax(-1)
        expected.append(token)
    actual = model.predict_next_latent(anchor, pending)
    assert torch.equal(actual, torch.stack(expected, -1))
    compiled = torch.compile(model.predict_next_latent, backend="aot_eager", fullgraph=True)
    assert torch.equal(actual, compiled(anchor, pending))


@pytest.mark.parametrize("horizon", [1, 4, 8])
@pytest.mark.parametrize("budget", [0, 1, 2, 9, 16])
@pytest.mark.parametrize("split", [False, True])
def test_latent_bf16_generation_matches_greedy(eager, horizon, budget, split, monkeypatch):
    model = latent_model(horizon, split_target_demo_attention=split,
                         attention_pattern="pair_group" if split else "dense").to(torch.bfloat16)
    monkeypatch.setattr(model, "predict_mtp", lambda *a: pytest.fail("NextLat used MTP"))
    drafts = []
    propose = model.predict_next_latent
    monkeypatch.setattr(model, "predict_next_latent", lambda *a: drafts.append(1) or propose(*a))
    kwargs = dict(model=model, prompts=PROMPTS, example_ids=[0, 1],
                  device=torch.device("cpu"), max_new_tokens=budget)
    assert evaluate.batched_greedy_generate(**kwargs, speculative=True) == evaluate.batched_greedy_generate(**kwargs)
    if budget > 1:
        assert drafts  # both the split-cache and the full-cache routes draft with NextLat


class ScriptedLatentModel(ScriptedModel):
    def __init__(self, errors):
        super().__init__(errors)
        self.config = replace(self.config, n_predict=1, next_latent_prediction=True,
                              next_latent_horizon=3)

    def predict_next_latent(self, hidden_states, pending):
        offsets = hidden_states[:, 0].long().clamp(max=7)
        # Finished rows may contain EOS; every active row's pending token must
        # align with the verified transformer state rather than a draft state.
        expected = self.answers.gather(1, offsets[:, None]).squeeze(1)
        assert torch.all((pending == expected) | (pending == END_PAIR_TOKEN_ID))
        return super().predict_mtp(hidden_states)

    def predict_mtp(self, *args):
        pytest.fail("Latent speculation dispatched to MTP")


@pytest.mark.parametrize("errors", [(None, None), (0, 2), (1, 0)])
def test_latent_heterogeneous_acceptance_eos_and_positions(eager, errors):
    model = ScriptedLatentModel(errors)
    kwargs = dict(model=model, prompts=PROMPTS, example_ids=[0, 1],
                  device=torch.device("cpu"), max_new_tokens=12)
    actual = evaluate.batched_greedy_generate(**kwargs, speculative=True)
    calls = list(model.calls)
    assert actual == evaluate.batched_greedy_generate(**kwargs)
    if errors == (None, None):
        assert len(calls) == 2
    for slots, positions, _ in calls:
        for row in range(2):
            offset = int(slots[row, 0]) - model.prompt_len
            if offset >= len(actual[row]) - len(PROMPTS[row]):
                continue
            expected = compute_positions_3d(torch.tensor(actual[row])[None])[0]
            assert torch.equal(positions[row, 0], expected[len(PROMPTS[row]) + offset])


def test_latent_cache_capacity_tail_and_greedy_only(eager):
    model = latent_model(8, max_seq_len=12).to(torch.bfloat16)
    kwargs = dict(model=model, prompts=PROMPTS, example_ids=[0, 1],
                  device=torch.device("cpu"), max_new_tokens=20)
    assert evaluate.batched_greedy_generate(**kwargs, speculative=True) == evaluate.batched_greedy_generate(**kwargs)
    with pytest.raises(ValueError, match="greedy"):
        evaluate.batched_greedy_generate(**kwargs, speculative=True, temperature=.7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("horizon", [4, 8])
def test_cuda_compiled_latent_speculation_matches_greedy(horizon):
    # CUDA FlexAttention requires head dimensions >= 16; CPU fixtures use 6.
    # Exercise the split target/demo attention used by the NextLat recipes.
    model = latent_model(horizon, d_model=64, d_ff=128, split_target_demo_attention=True,
                         attention_pattern="pair_group").to(device="cuda", dtype=torch.bfloat16)
    kwargs = dict(model=model, prompts=PROMPTS, example_ids=[0, 1],
                  device=torch.device("cuda"), max_new_tokens=20)
    assert evaluate.batched_greedy_generate(**kwargs, speculative=True) == evaluate.batched_greedy_generate(**kwargs)
