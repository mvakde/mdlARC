"""NextLat on a looped model: training target and lossless speculative decoding."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import evaluate
from common import END_PAIR_TOKEN_ID
from test_mtp import _batch, _sequences, _set_path
from test_next_latent import latent_model, reference_losses
from test_speculative_decoding import PROMPTS, eager, small_model
from tinytransformer import TinyTransformer

# The launcher builds pair_group models with split target/demo attention.
LOOP = dict(n_layers=4, loop_iterations=3, loop_start_layer=1, loop_end_layer=3,
            loop_input_injection=True, loop_exit_weight=0.3,
            attention_pattern="pair_group", split_target_demo_attention=True)


@pytest.mark.parametrize("path", ["padded", "fallback", "packed"])
@pytest.mark.parametrize("horizon", [1, 4])
def test_looped_latent_loss_uses_final_state_and_keeps_exit_loss(path, horizon, monkeypatch):
    model = latent_model(next_latent_horizon=horizon, next_latent_regression_weight=2.0, **LOOP).train()
    _set_path(model, monkeypatch, path)
    captured = []
    handle = model.norm.register_forward_hook(lambda module, args, result: captured.append(result))
    result = model(**_batch(path != "padded"))
    handle.remove()
    # Deep supervision normalizes one early exit per non-final iteration first;
    # the last call is the final state after every loop iteration and the coda.
    assert len(captured) == LOOP["loop_iterations"]
    hidden = captured[-1]
    logits = model.lm_head(hidden)
    if logits.dim() != result["logits"].dim():  # padded fallback returns packed logits
        logits = logits[_batch()["attention_mask"]]
    torch.testing.assert_close(result["logits"], logits)
    states = hidden.unbind(0) if hidden.dim() == 3 else hidden.split([len(t) for t, _ in _sequences()])
    padded = _batch()
    regression, kl, count = reference_losses(model, states, padded["input_ids"].tolist())
    torch.testing.assert_close(result["next_latent_regression_loss"], regression)
    torch.testing.assert_close(result["next_latent_kl_loss"], kl, atol=2e-7, rtol=2e-5)
    assert result["num_next_latent_tokens"] == count
    assert result["loop_exit_loss"] is not None and result["loop_exit_loss"] > 0
    assert result["mtp_loss"] is None and len(model.lm_head_mtp) == 0
    torch.testing.assert_close(result["loss"], result["output_loss"] + 0.3 * result["loop_exit_loss"]
                               + 2.0 * regression + kl)
    result["loss"].backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is None or torch.isfinite(parameter.grad).all(), name
    # The latent objective reaches the dynamics, every block of the schedule
    # (prelude, shared loop span, coda) and the loop's input injection.
    model.zero_grad(set_to_none=True)
    model(**_batch(path != "padded"))["next_latent_loss"].backward()
    assert model.next_latent_dynamics.mlp[-1].weight.grad.norm() > 0
    for block in model.blocks:
        assert block.attention.qkv_proj.weight.grad.norm() > 0
    assert model.loop_input_scale.grad is not None


@pytest.mark.parametrize("path", ["padded", "fallback", "packed"])
def test_looped_latent_adds_to_unchanged_mtp(path, monkeypatch):
    """MTP baseline plus a lighter NextLat: both objectives enter the loss once."""
    model = latent_model(next_latent_horizon=4, next_latent_regression_weight=0.5,
                         next_latent_kl_weight=0.25, n_predict=4, mtp_weight=0.3, **LOOP).train()
    _set_path(model, monkeypatch, path)
    result = model(**_batch(path != "padded"))
    assert len(model.lm_head_mtp) == 3 and result["mtp_loss"] > 0
    assert result["next_latent_loss"] > 0 and result["loop_exit_loss"] > 0
    torch.testing.assert_close(result["next_latent_loss"], 0.5 * result["next_latent_regression_loss"]
                               + 0.25 * result["next_latent_kl_loss"])
    torch.testing.assert_close(result["loss"], result["output_loss"] + 0.3 * result["mtp_loss"]
                               + 0.3 * result["loop_exit_loss"] + result["next_latent_loss"])


def looped_latent_model(horizon, seed=7, **options):
    config = replace(small_model(seed).config, next_latent_prediction=True,
                     next_latent_horizon=horizon, max_seq_len=256, **{"n_predict": 1, **LOOP, **options})
    torch.manual_seed(seed)
    return TinyTransformer(config).eval().to(torch.bfloat16)


def _oracle_drafts(reference, prompts, horizon, correct):
    """Draft the true greedy continuation, wrong from draft index ``correct`` on.

    Every active row then advances by exactly ``min(correct, horizon) + 1``
    tokens per verification, so the pass count fixes each row's position.
    """
    continuations = [row[len(prompt):] for row, prompt in zip(reference, prompts)]
    step = min(correct, horizon) + 1
    calls = []

    def draft(hidden_states, pending):
        start = len(calls) * step
        calls.append(pending.clone())
        rows = []
        for continuation in continuations:
            tokens = continuation[start + 1:start + 1 + horizon]
            tokens = tokens + [END_PAIR_TOKEN_ID] * (horizon - len(tokens))
            rows.append([token if index < correct else (token + 1) % END_PAIR_TOKEN_ID
                         for index, token in enumerate(tokens)])
        return torch.tensor(rows, dtype=torch.long, device=pending.device)

    return draft, calls


@pytest.mark.parametrize("horizon", [1, 4])
@pytest.mark.parametrize("budget", [1, 9, 24])
@pytest.mark.parametrize("seed", [7, 31])
@pytest.mark.parametrize("n_predict", [1, 4])  # with MTP heads, NextLat still drafts
def test_looped_latent_speculation_matches_greedy(eager, horizon, budget, seed, n_predict, monkeypatch):
    model = looped_latent_model(horizon, seed, n_predict=n_predict)
    monkeypatch.setattr(model, "predict_mtp", lambda *a: pytest.fail("NextLat used MTP"))
    kwargs = dict(model=model, prompts=PROMPTS, example_ids=[0, 1],
                  device=torch.device("cpu"), max_new_tokens=budget)
    reference = evaluate.batched_greedy_generate(**kwargs)
    assert evaluate.batched_greedy_generate(**kwargs, speculative=True) == reference
    stats = getattr(model, "_speculative_acceptance", None)
    if budget > 1 and any(len(row) - len(prompt) > 1 for row, prompt in zip(reference, PROMPTS)):
        verifications, accepted, proposed = stats.tolist()
        assert verifications > 0 and 0 <= accepted <= proposed == verifications * horizon


@pytest.mark.parametrize("horizon", [2, 4])
@pytest.mark.parametrize("correct", [0, 1, 4])
@pytest.mark.parametrize("seed", [7, 31])  # rows end at 4 and 7 tokens / both run to the budget
def test_looped_cache_commits_accepted_blocks_exactly(eager, horizon, correct, seed, monkeypatch):
    """Force acceptances on a real looped model: every iteration's cache must commit them."""
    model = looped_latent_model(horizon, seed)
    kwargs = dict(model=model, prompts=PROMPTS, example_ids=[0, 1],
                  device=torch.device("cpu"), max_new_tokens=24)
    reference = evaluate.batched_greedy_generate(**kwargs)
    lengths = [len(row) - len(prompt) for row, prompt in zip(reference, PROMPTS)]
    assert max(lengths) > horizon + 1, "fixture must generate past one block"
    draft, calls = _oracle_drafts(reference, PROMPTS, horizon, correct)
    monkeypatch.setattr(model, "predict_next_latent", draft)
    assert model.cache_layer_count == 1 + 2 * 3 + 1  # prelude, 3 x two-block loop, coda
    assert evaluate.batched_greedy_generate(**kwargs, speculative=True) == reference
    step = min(correct, horizon) + 1
    assert len(calls) == -(-(max(lengths) - 1) // step)
    _, accepted, proposed = model._speculative_acceptance.tolist()
    assert (accepted > 0) == (correct > 0) and proposed > 0
    assert evaluate.speculative_summary(model).startswith("Speculative acceptance: ")
