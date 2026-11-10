from pathlib import Path
import sys

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import END_PAIR_TOKEN_ID, IGNORE_INDEX, compute_positions_3d  # noqa: E402
from tinytransformer import TinyTransformer, TinyTransformerConfig  # noqa: E402
from train import train_one_epoch  # noqa: E402


def _model():
    torch.manual_seed(1)
    return TinyTransformer(TinyTransformerConfig(
        d_model=24, n_heads=4, d_ff=48, n_layers=2, max_seq_len=64,
        num_examples=2, num_task_embedding_latents=2,
        dropout=0.0, attention_dropout=0.0, embedding_dropout=0.0,
    ))


def _batch(packed):
    sequences = [torch.tensor([10, 1, 12, 2, 13]), torch.tensor([10, 3, 11, 4, 12, 4, 11, 3, 13])]
    inputs = torch.full((2, 9), END_PAIR_TOKEN_ID, dtype=torch.long)
    mask = torch.zeros_like(inputs, dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        inputs[row, :sequence.numel()] = sequence
        mask[row, :sequence.numel()] = True
    positions = compute_positions_3d(inputs, mask)
    batch = {
        "input_ids": inputs, "example_ids": torch.tensor([0, 1], dtype=torch.int32),
        "dihedral_ids": torch.tensor([0, 3], dtype=torch.int32),
        "sep_indices": torch.tensor([2, 4], dtype=torch.int16),
        "positions_3d": positions, "attention_mask": mask, "has_padding": True,
    }
    if packed:
        batch.update(input_ids=inputs[mask], positions_3d=positions[mask],
                     cu_seqlens=torch.tensor([0, 5, 14], dtype=torch.int32), max_seqlen=9,
                     has_padding=False)
        del batch["attention_mask"]
    return batch


def _forward(model, batch, metrics=True):
    return model(**{key: value for key, value in batch.items() if key != "has_padding"},
                 compute_training_metrics=metrics)


@pytest.mark.parametrize("packed", [False, True])
def test_metric_counts_exclude_prompt_padding_and_empty_outputs(packed):
    model = _model()
    targets = torch.tensor([[9, 1, 2, 11, 13, IGNORE_INDEX],
                            [8, 3, 4, 11, 13, IGNORE_INDEX],
                            [7, 0, 0, 0, 0, IGNORE_INDEX]])
    predictions = torch.tensor([[0, 1, 2, 11, 13, 9], [0, 3, 5, 10, 13, 9], [0, 0, 0, 0, 0, 9]])
    valid = torch.tensor([[False, True, True, True, True, False],
                          [False, True, True, True, True, False],
                          [False, False, False, False, False, False]])
    logits = torch.full((*targets.shape, 14), -10.0)
    logits.scatter_(-1, predictions.unsqueeze(-1), 10.0)
    args = {}
    if packed:
        logits, targets, valid = logits.flatten(0, 1), targets.flatten(), valid.flatten()
        args = {"sequence_ids": torch.arange(3).repeat_interleave(6), "num_sequences": 3}
    counts = model._compute_training_metric_counts(logits, targets, valid, **args)
    torch.testing.assert_close(counts, torch.tensor([3, 4, 3, 4, 1, 2]))


def test_packed_fallback_and_padded_metrics_agree():
    model = _model()
    # Exercise the CPU fallback independently of whether flash-attn is installed.
    model._has_varlen_flash_attention = lambda: False
    padded = _forward(model, _batch(False))
    packed = _forward(model, _batch(True))
    torch.testing.assert_close(padded["training_metric_counts"], packed["training_metric_counts"])
    assert packed["training_metric_counts"][[1, 3, 5]].tolist() == [3, 3, 2]
    torch.testing.assert_close(padded["loss"], packed["loss"])


def test_packed_training_path_reports_the_same_metrics(monkeypatch):
    model = _model()
    padded = _forward(model, _batch(False))

    def reference_varlen(queries, keys, values, cu_seqlens_q, cu_seqlens_k,
                         max_seqlen_q, max_seqlen_k, dropout_p, is_causal):
        # Stand in for CUDA FlashAttention while exercising the real packed
        # forward and supervision paths, including suffix-query alignment.
        outputs = []
        for row in range(cu_seqlens_q.numel() - 1):
            q = queries[cu_seqlens_q[row]:cu_seqlens_q[row + 1]].transpose(0, 1)
            k = keys[cu_seqlens_k[row]:cu_seqlens_k[row + 1]].transpose(0, 1)
            v = values[cu_seqlens_k[row]:cu_seqlens_k[row + 1]].transpose(0, 1)
            mask = None
            if is_causal:
                q_positions = torch.arange(q.size(1)) + k.size(1) - q.size(1)
                mask = torch.arange(k.size(1))[None, :] <= q_positions[:, None]
            outputs.append(torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=dropout_p,
            ).transpose(0, 1))
        return torch.cat(outputs)

    monkeypatch.setattr(model, "_has_varlen_flash_attention", lambda: True)
    for block in model.blocks:
        monkeypatch.setattr(block.attention, "_apply_varlen_flash_attention", reference_varlen)
    packed = _forward(model, _batch(True))
    torch.testing.assert_close(packed["logits"], padded["logits"][_batch(False)["attention_mask"]])
    torch.testing.assert_close(packed["loss"], padded["loss"])
    torch.testing.assert_close(packed["training_metric_counts"], padded["training_metric_counts"])
    packed["loss"].backward()
    assert torch.isfinite(model.lm_head.weight.grad).all()


@pytest.mark.parametrize("packed", [False, True])
def test_metrics_do_not_change_loss_logits_or_gradients(packed):
    model = _model()
    model._has_varlen_flash_attention = lambda: False
    baseline = _forward(model, _batch(packed), metrics=False)
    assert baseline["training_metric_counts"] is None
    baseline["loss"].backward()
    baseline_grads = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    measured = _forward(model, _batch(packed))
    assert not measured["training_metric_counts"].requires_grad
    torch.testing.assert_close(baseline["logits"], measured["logits"])
    torch.testing.assert_close(baseline["loss"], measured["loss"])
    measured["loss"].backward()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(baseline_grads[name], parameter.grad)


@pytest.mark.parametrize("log_mode", ["step", "10_steps", "epoch"])
def test_training_reports_epoch_metrics(log_mode, capsys):
    model = _model()
    model._has_varlen_flash_attention = lambda: False
    batch = _batch(True)
    counts = _forward(model, batch)["training_metric_counts"].tolist()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    train_one_epoch(model, [batch, batch], optimizer, torch.device("cpu"),
                    grad_clip=0.0, epoch=0, train_log_mode=log_mode)
    output = capsys.readouterr().out
    for name, correct, total in zip(
        ["color_accuracy", "special_token_accuracy", "exact_output_accuracy"],
        counts[::2], counts[1::2],
    ):
        assert f"{name}={100 * correct / total:.2f}%" in output


@pytest.mark.parametrize("log_mode,location", [("never", "both"), ("epoch", "none")])
def test_disabled_logging_skips_metric_computation(log_mode, location, monkeypatch, capsys):
    model = _model()
    monkeypatch.setattr(model, "_compute_training_metric_counts", lambda *args: pytest.fail("metrics computed"))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    train_one_epoch(model, [_batch(False)], optimizer, torch.device("cpu"),
                    grad_clip=0.0, epoch=0, train_log_mode=log_mode, log_location=location)
    assert capsys.readouterr().out == ""
