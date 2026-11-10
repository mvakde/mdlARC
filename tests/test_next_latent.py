"""Independent NextLat loss oracle, mask boundaries, gradients and training."""
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import END_PAIR_TOKEN_ID, IGNORE_INDEX, compute_positions_3d, encode_context
from next_latent import NextLatentDynamics, next_latent_losses
from tinytransformer import TinyTransformer, TinyTransformerConfig
import train
import next_latent
from test_mtp import _batch, _sequences, _set_path
from test_group_attention import DEMOS, TARGET


def latent_model(**options):
    torch.manual_seed(7)
    config = dict(d_model=24, n_heads=4, d_ff=48, n_layers=2, max_seq_len=128,
                  num_examples=2, num_task_embedding_latents=2, n_predict=1,
                  dropout=0, embedding_dropout=0, attention_dropout=0,
                  next_latent_prediction=True)
    config.update(options)
    return TinyTransformer(TinyTransformerConfig(**config))


def reference_losses(model, states, labels):
    horizon = model.config.next_latent_horizon
    predictions, targets, kl_terms = [[] for _ in range(horizon)], [[] for _ in range(horizon)], [[] for _ in range(horizon)]
    for row, (tokens, sep) in enumerate(_sequences()):
        for position in range(sep, len(tokens) - 1):
            prediction = states[row][position]
            for step in range(horizon):
                following = position + step + 1
                if (following >= len(tokens) or labels[row][following] == IGNORE_INDEX
                        or tokens[following] == END_PAIR_TOKEN_ID):
                    break
                prediction = model.next_latent_dynamics(prediction, model.token_embedding(tokens[following]))
                target = states[row][following].detach()
                predictions[step].append(prediction)
                targets[step].append(target)
                if following + 1 < len(tokens) and labels[row][following + 1] != IGNORE_INDEX:
                    teacher = F.softmax(F.linear(target, model.lm_head.weight.detach()).float(), -1)
                    student_log = F.log_softmax(F.linear(prediction, model.lm_head.weight.detach()).float(), -1)
                    kl_terms[step].append((teacher * (teacher.log() - student_log)).sum())
    regression = sum(F.smooth_l1_loss(torch.stack(p).float(), torch.stack(t).float())
                     for p, t in zip(predictions, targets) if p) / horizon
    kl = sum(torch.stack(terms).mean() for terms in kl_terms if terms) / horizon
    return regression, kl, sum(map(len, predictions))


@pytest.mark.parametrize("path", ["padded", "fallback", "packed"])
@pytest.mark.parametrize("custom_targets", [False, True])
@pytest.mark.parametrize("horizon", [1, 4, 8])
def test_next_latent_matches_independent_per_example_oracle(path, custom_targets, horizon, monkeypatch):
    model = latent_model(next_latent_horizon=horizon)
    _set_path(model, monkeypatch, path)
    captured = []
    handle = model.norm.register_forward_hook(lambda module, args, result: captured.append(result))
    result = model(**_batch(path != "padded", custom_targets=custom_targets))
    handle.remove()
    hidden = captured[0]
    states = hidden.unbind(0) if hidden.dim() == 3 else hidden.split([len(tokens) for tokens, _ in _sequences()])
    padded = _batch(custom_targets=custom_targets)
    labels = padded.get("targets", padded["input_ids"]).tolist()
    regression, kl, count = reference_losses(model, states, labels)
    torch.testing.assert_close(result["next_latent_regression_loss"], regression)
    torch.testing.assert_close(result["next_latent_kl_loss"], kl, atol=2e-7, rtol=2e-5)
    torch.testing.assert_close(result["loss"], result["output_loss"] + regression + kl)
    assert result["num_next_latent_tokens"] == count
    assert result["mtp_loss"] is None and len(model.lm_head_mtp) == 0
    result["loss"].backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert model.next_latent_dynamics.mlp[-1].weight.grad.norm() > 0


@pytest.mark.parametrize("pattern", ["dense", "pair_group"])
@pytest.mark.parametrize("horizon", [1, 4, 8])
def test_packed_padded_and_fallback_latent_gradients_agree(pattern, horizon, monkeypatch):
    model = latent_model(attention_pattern=pattern, split_target_demo_attention=pattern == "pair_group",
                         next_latent_horizon=horizon, next_latent_regression_weight=2.0)
    sequences = [encode_context(DEMOS, TARGET), encode_context([], TARGET)]
    reference = model(**_batch(custom_targets=True, sequences=sequences))
    reference["loss"].backward()
    gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
    for path in ("fallback", "packed"):
        model.zero_grad(set_to_none=True)
        _set_path(model, monkeypatch, path)
        actual = model(**_batch(True, custom_targets=True, sequences=sequences))
        for key in ("loss", "output_loss", "next_latent_loss", "next_latent_regression_loss", "next_latent_kl_loss"):
            torch.testing.assert_close(actual[key], reference[key])
        actual["loss"].backward()
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad, gradients[name], atol=3e-6, rtol=3e-5)


def test_latent_teacher_and_readout_are_stopped_but_current_state_and_embedding_learn():
    dynamics = NextLatentDynamics(6)
    embedding = torch.nn.Embedding(14, 6)
    head = torch.nn.Linear(6, 14, bias=False)
    states = torch.randn(1, 5, 6, requires_grad=True)
    tokens = torch.tensor([[10, 1, 12, 2, END_PAIR_TOKEN_ID]])
    regression, kl, count = next_latent_losses(dynamics, states, tokens,
        torch.tensor([[False, False, True, True]]), embedding, head)
    (regression + kl).backward()
    assert count == 1
    assert states.grad[0, 2].norm() > 0
    assert states.grad[0, [0, 1, 3, 4]].count_nonzero() == 0
    assert head.weight.grad is None
    assert embedding.weight.grad[2].norm() > 0
    assert dynamics.mlp[-1].weight.grad.norm() > 0


def test_native_packed_reuses_attention_indices_without_another_selection(monkeypatch):
    model = latent_model()
    _set_path(model, monkeypatch, "packed")
    monkeypatch.setattr(next_latent, "_transition_indices", lambda *args: pytest.fail("Repeated index selection"))
    result = model(**_batch(True, custom_targets=True))
    assert result["num_next_latent_tokens"] > 0
    result["loss"].backward()
    assert model.next_latent_dynamics.mlp[-1].weight.grad.norm() > 0


@pytest.mark.parametrize("path", ["padded", "packed", "fallback"])
@pytest.mark.parametrize("horizon", [1, 4, 8])
def test_no_valid_transitions_are_finite_and_never_cross_examples(path, horizon, monkeypatch):
    model = latent_model(next_latent_horizon=horizon)
    _set_path(model, monkeypatch, path)
    # One output consists solely of END, the other is an empty prefix-only row.
    sequences = [(torch.tensor([10, 1, 12, END_PAIR_TOKEN_ID]), 2), (torch.tensor([10, 2, 12]), 2)]
    result = model(**_batch(path != "padded", sequences=sequences))
    assert result["num_next_latent_tokens"] == 0
    assert result["next_latent_loss"].item() == 0
    result["loss"].backward()
    for parameter in model.next_latent_dynamics.parameters():
        assert parameter.grad is not None and parameter.grad.count_nonzero() == 0


def test_initial_trunk_logits_rng_and_generation_do_not_change(monkeypatch):
    baseline = latent_model(next_latent_prediction=False).eval()
    rng_before = torch.random.get_rng_state()
    model = latent_model().eval()
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before)
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    monkeypatch.setattr(model.next_latent_dynamics, "forward", lambda *args: pytest.fail("Inference ran dynamics"))
    actual = model(**_batch(), compute_next_latent_loss=False)
    torch.testing.assert_close(actual["logits"], baseline(**_batch())["logits"], rtol=0, atol=0)
    tokens, sep = _sequences()[0]
    kwargs = dict(input_ids=tokens[None, :sep + 1], example_ids=torch.tensor([0]),
                  positions_3d=compute_positions_3d(tokens[None, :sep + 1]))
    torch.testing.assert_close(model.forward_generate(**kwargs)["logits"], baseline.forward_generate(**kwargs)["logits"], rtol=0, atol=0)


@pytest.mark.parametrize("with_mtp", [False, True])
@pytest.mark.parametrize("logging", [False, True])
def test_training_updates_total_latent_objective_with_accumulation_and_mtp_decay(with_mtp, logging, capsys):
    expected = latent_model(n_predict=4 if with_mtp else 1)
    actual = latent_model(n_predict=4 if with_mtp else 1)
    batch = _batch()
    result = expected(**batch)
    objective = result["output_loss"] + result["next_latent_loss"]
    if with_mtp:
        objective = objective + .3 * .1 * result["mtp_loss"]
    objective.backward()
    torch.optim.SGD(expected.parameters(), lr=.01).step()
    train.train_one_epoch(actual, [batch, batch], torch.optim.SGD(actual.parameters(), lr=.01),
        torch.device("cpu"), grad_clip=0, gradient_accumulation_steps=2,
        epoch=9, total_epochs=10, mtp_decay=[.8, .9, .1],
        train_log_mode="step" if logging else "never", log_location="terminal" if logging else "none")
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], value, atol=2e-7, rtol=2e-6)
    log = capsys.readouterr().out
    if logging:
        assert "next_latent_regression_loss=" in log and "next_latent_kl_loss=" in log
        assert f"total_loss={objective.item():.4f}" in log
    else:
        assert log == ""
