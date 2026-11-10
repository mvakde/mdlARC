"""MTP target alignment, gradients, inference, and checkpoint compatibility."""

from argparse import Namespace
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import build
import launch_run
import train
from common import END_PAIR_TOKEN_ID, IGNORE_INDEX, compute_positions_3d, encode_pair
from tinytransformer import TinyTransformer, TinyTransformerConfig


def _model(n_predict=4, mtp_weight=0.3):
    torch.manual_seed(7)
    return TinyTransformer(TinyTransformerConfig(
        d_model=24, n_heads=4, d_ff=48, n_layers=2, max_seq_len=64,
        num_examples=2, num_task_embedding_latents=2,
        dropout=0.0, embedding_dropout=0.0, attention_dropout=0.0,
        n_predict=n_predict, mtp_weight=mtp_weight,
    ))


def _sequences():
    return [encode_pair(pair) for pair in (
        {"input": [[1]], "output": [[2]]},
        {"input": [[3, 4]], "output": [[5, 6], [7, 8]]},
    )]


def _batch(packed=False, *, custom_targets=False, sequences=None):
    sequences = _sequences() if sequences is None else sequences
    lengths = [len(tokens) for tokens, _ in sequences]
    inputs = torch.full((len(lengths), max(lengths)), END_PAIR_TOKEN_ID, dtype=torch.long)
    mask = torch.zeros_like(inputs, dtype=torch.bool)
    for row, (tokens, _) in enumerate(sequences):
        inputs[row, :len(tokens)] = tokens
        mask[row, :len(tokens)] = True
    batch = dict(
        input_ids=inputs, attention_mask=mask,
        positions_3d=compute_positions_3d(inputs, mask),
        example_ids=torch.arange(len(lengths), dtype=torch.int32),
        dihedral_ids=torch.arange(len(lengths), dtype=torch.int32) % 8,
        sep_indices=torch.tensor([sep for _, sep in sequences], dtype=torch.int16),
    )
    if custom_targets:
        # Nontrivial labels and ignore masks must be respected by all heads.
        targets = inputs.clone()
        for row, (_, sep) in enumerate(sequences):
            targets[row, sep + 1] = IGNORE_INDEX
            targets[row, sep + 2] = 9
        batch["targets"] = targets.masked_fill(~mask, IGNORE_INDEX)
    if packed:
        if custom_targets:
            labels = batch["targets"]
            next_labels = torch.full_like(labels, IGNORE_INDEX)
            next_labels[:, :-1] = labels[:, 1:]
            batch["targets"] = next_labels[mask]
        batch.update(
            input_ids=inputs[mask], positions_3d=batch["positions_3d"][mask],
            cu_seqlens=torch.tensor([0] + list(torch.tensor(lengths).cumsum(0)), dtype=torch.int32),
            max_seqlen=max(lengths),
        )
        del batch["attention_mask"]
    return batch


def _reference_flash(queries, keys, values, cu_seqlens_q, cu_seqlens_k,
                     max_seqlen_q, max_seqlen_k, dropout_p, is_causal):
    """CPU attention oracle, exercising the native packed model/loss path."""
    outputs = []
    for row in range(len(cu_seqlens_q) - 1):
        q = queries[cu_seqlens_q[row]:cu_seqlens_q[row + 1]].transpose(0, 1)
        k = keys[cu_seqlens_k[row]:cu_seqlens_k[row + 1]].transpose(0, 1)
        v = values[cu_seqlens_k[row]:cu_seqlens_k[row + 1]].transpose(0, 1)
        mask = None
        if is_causal:
            q_positions = torch.arange(q.size(1)) + k.size(1) - q.size(1)
            mask = torch.arange(k.size(1))[None, :] <= q_positions[:, None]
        outputs.append(F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=dropout_p,
        ).transpose(0, 1))
    return torch.cat(outputs)


def _set_path(model, monkeypatch, path):
    monkeypatch.setattr(model, "_has_varlen_flash_attention", lambda: path == "packed")
    if path == "packed":
        for block in model.blocks:
            monkeypatch.setattr(block.attention, "_apply_varlen_flash_attention", _reference_flash)


def _reference_losses(model, states, labels, sequences):
    # Enumerate each example independently, rather than reusing production masks
    # or packed tensor rolls. This catches off-by-one and cross-example errors.
    losses = []
    for offset, head in enumerate(model.lm_head_mtp, start=2):
        predictions, expected = [], []
        for row, (tokens, sep) in enumerate(sequences):
            for position in range(sep, len(tokens) - offset):
                if labels[row][position + 1] != IGNORE_INDEX and labels[row][position + offset] != IGNORE_INDEX:
                    predictions.append(head(states[row][position]))
                    expected.append(labels[row][position + offset])
        losses.append(
            F.cross_entropy(torch.stack(predictions), torch.tensor(expected))
            if predictions else head.weight.sum() * 0
        )
    return torch.stack(losses)


@pytest.mark.parametrize("path", ["padded", "fallback", "packed"])
@pytest.mark.parametrize("custom_targets", [False, True])
def test_mtp_matches_independent_per_example_loss(path, custom_targets, monkeypatch):
    model = _model()
    _set_path(model, monkeypatch, path)
    captured = []
    hook = model.norm.register_forward_hook(lambda module, args, output: captured.append(output))
    batch = _batch(path != "padded", custom_targets=custom_targets)
    result = model(**batch, compute_training_metrics=True)
    hook.remove()
    sequences = _sequences()
    hidden = captured[0]
    states = hidden.unbind(0) if hidden.dim() == 3 else hidden.split([len(t) for t, _ in sequences])
    padded = _batch(custom_targets=custom_targets)
    labels = padded.get("targets", padded["input_ids"]).tolist()
    reference = _reference_losses(model, states, labels, sequences)
    torch.testing.assert_close(result["mtp_losses"], reference)
    torch.testing.assert_close(result["loss"], result["output_loss"] + 0.3 * reference.mean())
    result["loss"].backward()
    for name, param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
    assert all(head.weight.grad.abs().sum() > 0 for head in model.lm_head_mtp)


@pytest.mark.parametrize("custom_targets", [False, True])
def test_packed_padded_and_fallback_losses_and_gradients_agree(custom_targets, monkeypatch):
    model = _model()
    _set_path(model, monkeypatch, "padded")
    padded_batch = _batch(custom_targets=custom_targets)
    reference = model(**padded_batch, compute_training_metrics=True)
    reference["loss"].backward()
    gradients = {name: param.grad.clone() for name, param in model.named_parameters()}
    for path in ("fallback", "packed"):
        model.zero_grad(set_to_none=True)
        _set_path(model, monkeypatch, path)
        actual = model(**_batch(True, custom_targets=custom_targets), compute_training_metrics=True)
        for key in ("loss", "output_loss", "mtp_loss", "mtp_losses", "training_metric_counts"):
            torch.testing.assert_close(actual[key], reference[key])
        torch.testing.assert_close(actual["logits"], reference["logits"][padded_batch["attention_mask"]])
        actual["loss"].backward()
        for name, param in model.named_parameters():
            torch.testing.assert_close(param.grad, gradients[name], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("path", ["padded", "fallback", "packed"])
def test_empty_output_and_horizons_longer_than_sequence_are_finite(path, monkeypatch):
    model = _model(n_predict=8)
    _set_path(model, monkeypatch, path)
    sequences = [(torch.tensor([10, 1, 12]), 2), (torch.tensor([10, 2, 12, 13]), 2)]
    result = model(**_batch(path != "padded", sequences=sequences))
    assert result["mtp_loss"].item() == 0
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    for head in model.lm_head_mtp:
        assert torch.isfinite(head.weight.grad).all()
        assert head.weight.grad.count_nonzero() == 0


def test_mtp_keeps_next_head_initialization_logits_and_generation_unchanged(monkeypatch):
    baseline, mtp = _model(1), _model(4)
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, mtp.state_dict()[name], rtol=0, atol=0)
    reference = baseline(**_batch(), compute_training_metrics=True)
    actual = mtp(**_batch(), compute_training_metrics=True)
    for key in ("logits", "output_loss", "training_metric_counts"):
        torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0)
    for head in mtp.lm_head_mtp:
        monkeypatch.setattr(head, "forward", lambda *args: pytest.fail("Inference ran an MTP head"))
    skipped = mtp(**_batch(), compute_mtp_loss=False)
    assert skipped["mtp_loss"] is None
    torch.testing.assert_close(skipped["loss"], reference["loss"])
    tokens, sep = _sequences()[0]
    prompt = tokens[None, :sep + 1]
    prefill_args = dict(input_ids=prompt, example_ids=torch.tensor([0]),
                        dihedral_ids=torch.tensor([0]),
                        positions_3d=compute_positions_3d(prompt))
    old, new = baseline.forward_generate(**prefill_args), mtp.forward_generate(**prefill_args)
    torch.testing.assert_close(old["logits"], new["logits"], rtol=0, atol=0)
    for old_kv, new_kv in zip(old["past_key_values"], new["past_key_values"]):
        for old_value, new_value in zip(old_kv, new_kv):
            torch.testing.assert_close(old_value, new_value, rtol=0, atol=0)


def test_zero_weight_skips_auxiliary_work(monkeypatch):
    model = _model(mtp_weight=0)
    for head in model.lm_head_mtp:
        monkeypatch.setattr(head, "forward", lambda *args: pytest.fail("Zero-weight head ran"))
    result = model(**_batch())
    assert result["mtp_loss"] is None
    torch.testing.assert_close(result["loss"], result["output_loss"], rtol=0, atol=0)


def test_training_backpropagates_total_loss_through_accumulation_and_logs_separately(capsys):
    expected, actual = _model(), _model()
    batch = _batch()
    reference = expected(**batch)
    reference["loss"].backward()
    torch.optim.SGD(expected.parameters(), lr=0.01).step()
    optimizer = torch.optim.SGD(actual.parameters(), lr=0.01)
    train.train_one_epoch(actual, [batch, batch], optimizer, torch.device("cpu"),
                          grad_clip=0, gradient_accumulation_steps=2,
                          train_log_mode="step", log_location="terminal")
    for name, param in expected.named_parameters():
        torch.testing.assert_close(actual.state_dict()[name], param, atol=1e-7, rtol=1e-6)
    output = capsys.readouterr().out
    for key, name in (("output_loss", "loss"), ("mtp_loss", "mtp_loss"), ("loss", "total_loss")):
        assert f"{name}={reference[key].item():.4f}" in output


@pytest.mark.parametrize("progress,expected", [
    (0.0, 1.0), (0.79, 1.0), (0.8, 1.0), (0.825, 0.775),
    (0.85, 0.55), (0.875, 0.325), (0.9, 0.1), (1.0, 0.1),
])
def test_mtp_decay_boundaries_and_linear_interpolation(progress, expected):
    assert train._mtp_loss_multiplier(progress, [0.8, 0.9, 0.1]) == pytest.approx(expected)


@pytest.mark.parametrize("final_multiplier", [0.0, 1.0])
def test_mtp_decay_supports_zero_and_constant_final_multipliers(final_multiplier):
    decay = [0.8, 0.9, final_multiplier]
    assert train._mtp_loss_multiplier(0.85, decay) == pytest.approx((1 + final_multiplier) / 2)
    assert train._mtp_loss_multiplier(0.9, decay) == final_multiplier
    assert train._mtp_loss_multiplier(1.0, decay) == final_multiplier


@pytest.mark.parametrize("path", ["padded", "fallback", "packed"])
@pytest.mark.parametrize("epoch,decay,multiplier", [
    (2, [0.8, 0.9, 0.1], 1.0),
    (8, [0.8, 0.9, 0.1], 0.775),  # Minibatches start at 80% and 85%.
    (9, [0.8, 0.9, 0.1], 0.1),
    (9, [0.8, 0.9, 0.0], 0.0),
    (3, [0.2, 0.4, 0.2], 0.5),
])
def test_training_mtp_decay_scales_gradients_and_logged_total_loss(
    path, epoch, decay, multiplier, monkeypatch, capsys,
):
    expected, actual = _model(), _model()
    _set_path(expected, monkeypatch, path)
    _set_path(actual, monkeypatch, path)
    batch = _batch(path != "padded")
    reference = expected(**batch)
    loss = reference["output_loss"] + expected.config.mtp_weight * multiplier * reference["mtp_loss"]
    loss.backward()
    torch.optim.SGD(expected.parameters(), lr=0.01).step()
    optimizer = torch.optim.SGD(actual.parameters(), lr=0.01)
    step = train.train_one_epoch(
        actual, [batch, batch], optimizer, torch.device("cpu"),
        grad_clip=0, gradient_accumulation_steps=2, start_step=1234,
        epoch=epoch, total_epochs=10, mtp_decay=decay,
        train_log_mode="step", log_location="terminal",
    )
    assert step == 1236
    for name, param in expected.named_parameters():
        torch.testing.assert_close(actual.state_dict()[name], param, atol=1e-7, rtol=1e-6)
    output = capsys.readouterr().out
    assert f"loss={reference['output_loss'].item():.4f}" in output
    assert f"mtp_loss={reference['mtp_loss'].item():.4f}" in output
    assert f"total_loss={loss.item():.4f}" in output
    assert actual.config.mtp_weight == 0.3


def test_training_mtp_decay_applies_with_logging_disabled():
    expected, actual = _model(), _model()
    batch = _batch()
    reference = expected(**batch)
    loss = reference["output_loss"] + expected.config.mtp_weight * reference["mtp_loss"]
    loss.backward()
    torch.optim.SGD(expected.parameters(), lr=0.01).step()
    train.train_one_epoch(
        actual, [batch], torch.optim.SGD(actual.parameters(), lr=0.01), torch.device("cpu"),
        grad_clip=0, epoch=9, total_epochs=10, train_log_mode="never", log_location="none",
    )
    for name, param in expected.named_parameters():
        torch.testing.assert_close(actual.state_dict()[name], param, atol=1e-7, rtol=1e-6)


def test_mtp_decay_defaults_and_stage_overrides():
    plan = launch_run.build_plan({}, Path("spec.json"))
    assert plan["pretrain"]["mtp_decay"] == [0.8, 0.9, 1.0]
    assert launch_run.stage_summary(plan["pretrain"])["mtp_decay"] == [0.8, 0.9, 1.0]
    plan = launch_run.build_plan({
        "defaults": {"mtp_decay": [0.6, 0.95, 0.3]},
        "pretrain": {"mtp_decay": [0.5, 0.8, 0.0]},
        "finetunes": {"custom": {"mtp_decay": [0.2, 0.4, 1.0]}},
    }, Path("spec.json"))
    assert plan["pretrain"]["mtp_decay"] == [0.5, 0.8, 0.0]
    assert plan["finetunes"]["custom"]["mtp_decay"] == [0.2, 0.4, 1.0]


@pytest.mark.parametrize("decay", [
    None, "0.8,0.9,0.1", 0.8, [], [0.8], [0.8, 0.9], [0.7, 0.8, 0.9, 0.1],
    [-0.1, 0.9, 0.1], [0.8, 1.1, 0.1], [0.9, 0.8, 0.1], [0.8, 0.8, 0.1],
    [True, 0.9, 0.1], [0.8, False, 0.1], [0.8, 0.9, True], ["0.8", 0.9, 0.1],
    [float("nan"), 0.9, 0.1], [0.8, float("inf"), 0.1],
    [0.8, 0.9, -0.1], [0.8, 0.9, 1.1], [0.8, 0.9, "0.1"],
    [0.8, 0.9, float("nan")], [0.8, 0.9, float("inf")],
])
def test_invalid_mtp_decay_is_rejected_before_training(decay):
    with pytest.raises(ValueError, match="mtp_decay"):
        launch_run.build_plan({"defaults": {"mtp_decay": decay}}, Path("spec.json"))


def test_all_prediction_heads_use_adamw_under_normuon():
    model = _model()
    groups = train._collect_param_groups(model, include_muon=True)
    adam_ids = {id(param) for param in groups["decay"]}
    muon_ids = {id(param) for key, params in groups.items() if key.startswith("muon_") for param in params}
    for head in [model.lm_head, *model.lm_head_mtp]:
        assert id(head.weight) in adam_ids
        assert id(head.weight) not in muon_ids
    grouped = [id(param) for params in groups.values() for param in params]
    assert len(grouped) == len(set(grouped)) == len(list(model.parameters()))


def _checkpoint(model, *, legacy=False):
    config = asdict(model.config)
    if legacy:
        config.pop("n_predict")
        config.pop("mtp_weight")
    return dict(config=config, model_state=model.state_dict(), embedding_task_ids=["a", "b"])


def test_legacy_checkpoint_inheritance_and_explicit_finetune_head_additions():
    original = _model(1)
    checkpoint = _checkpoint(original, legacy=True)
    config = build.build_checkpoint_config(checkpoint)
    assert config.n_predict == 1
    resume = Namespace(checkpoint_mode="resume", n_predict=None, mtp_weight=None)
    assert build._configure_checkpoint_mtp(config, resume, is_eval=False).n_predict == 1
    changed = Namespace(checkpoint_mode="resume", n_predict=4, mtp_weight=0.3)
    with pytest.raises(ValueError, match="finetune"):
        build._configure_checkpoint_mtp(config, changed, is_eval=False)
    changed.checkpoint_mode = "finetune"
    updated = build._configure_checkpoint_mtp(config, changed, is_eval=False)
    assert updated.n_predict == 4
    model = TinyTransformer(updated)
    with pytest.raises(RuntimeError, match="Missing key"):
        build._load_model_state(model, checkpoint, ["a", "b"])
    build._load_model_state(model, checkpoint, ["a", "b"], allow_mtp_resize=True)
    for name, value in original.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    assert build._configure_checkpoint_mtp(config, changed, is_eval=True).n_predict == 1


@pytest.mark.parametrize("missing", ["lm_head_mtp.1.weight", "lm_head.weight", "blocks.0.ff.fc_in.weight"])
def test_finetune_does_not_hide_checkpoint_corruption(missing):
    model = _model()
    checkpoint = _checkpoint(model)
    del checkpoint["model_state"][missing]
    with pytest.raises((ValueError, RuntimeError)):
        build._load_model_state(_model(5), checkpoint, ["a", "b"], allow_mtp_resize=True)


@pytest.mark.parametrize("key,value", [
    ("n_predict", 0), ("n_predict", True), ("n_predict", 2.5), ("n_predict", "4"),
    ("mtp_weight", -0.1), ("mtp_weight", float("nan")), ("mtp_weight", float("inf")),
    ("mtp_weight", True), ("mtp_weight", "0.3"),
])
def test_invalid_mtp_settings_fail_before_training(key, value):
    with pytest.raises(ValueError, match=key):
        TinyTransformerConfig(**{key: value})
    with pytest.raises(ValueError, match=key):
        launch_run.build_plan({"defaults": {key: value}}, Path("spec.json"))


@pytest.mark.parametrize("task_latents", [2, 7, None])
def test_fresh_builder_enables_mtp_and_saved_checkpoint_resumes_exactly(tmp_path, monkeypatch, task_latents):
    monkeypatch.setattr(build, "resolve_device", lambda _: torch.device("cpu"))
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({name: {"train": [{"input": [[1]], "output": [[2]]}], "test": []}
                                for name in ("a", "b")}))
    args = Namespace(**{**launch_run.BASE_ARGS, "device": "cpu", "d_model": 24,
                       "n_heads": 4, "d_ff": 48, "n_layers": 2,
                       "num_task_embedding_latents": task_latents, "data_path": path,
                       "checkpoint_path": None, "checkpoint_mode": "resume"})
    model, dataset, _, _, _ = build.build_model_and_data(args, build_dataloader=False)
    assert (model.config.n_predict, model.config.mtp_weight) == (4, 0.3)
    expected_latents = 2 if task_latents is None else task_latents
    assert model.task_factor_embedding.weight.shape == (2, expected_latents)
    save_path = tmp_path / "mtp.pt"
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    model(**_batch())["loss"].backward()
    if expected_latents > 2:
        gradient = model.task_factor_embedding.weight.grad[:, 2:]
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0
    optimizer.step()
    train.maybe_save_model(model, dataset, path, save_path, optimizer=optimizer,
                          epoch=1, global_step=1)
    checkpoint = build.load_checkpoint(save_path)
    assert checkpoint["config"]["n_predict"] == 4
    assert checkpoint["config"]["num_task_embedding_latents"] == expected_latents
    args.checkpoint_path = save_path
    args.num_task_embedding_latents = 1
    resumed, _, _, _, _ = build.build_model_and_data(args, build_dataloader=False)
    restored_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.001)
    train._load_optimizer_state(restored_optimizer, checkpoint["optimizer_state"], torch.device("cpu"))
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, resumed.state_dict()[name], rtol=0, atol=0)
    # An identical next update must keep weights and optimizer moments aligned.
    for candidate, opt in ((model, optimizer), (resumed, restored_optimizer)):
        candidate.eval()
        opt.zero_grad(set_to_none=True)
        candidate(**_batch())["loss"].backward()
        opt.step()
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, resumed.state_dict()[name], rtol=0, atol=0)


def test_standalone_checkpoint_can_explicitly_reset_for_mtp_finetuning():
    plan = launch_run.build_plan({
        "checkpoint": "old.pt", "enable_finetune": False,
        "pretrain": {"reset_training_state": True, "n_predict": 4, "mtp_weight": 0.3},
    }, Path("spec.json"))
    assert plan["pretrain"]["checkpoint_mode"] == "finetune"
    resumed = launch_run.build_plan({"checkpoint": "old.pt", "enable_finetune": False}, Path("spec.json"))
    assert resumed["pretrain"]["checkpoint_mode"] == "resume"
    assert resumed["pretrain"]["n_predict"] is None


@pytest.mark.parametrize("packed", [False, True])
def test_mtp_loss_and_backward_compile_without_graph_breaks(packed):
    from tinytransformer import build_output_targets_and_mask

    model = _model()
    batch = _batch(packed)
    inputs = batch["input_ids"]
    next_targets, valid = build_output_targets_and_mask(
        inputs, batch["sep_indices"], attention_mask=batch.get("attention_mask"),
        cu_seqlens=batch.get("cu_seqlens"),
    )
    hidden = torch.randn((*inputs.shape, model.config.d_model), requires_grad=True)
    kwargs = {"cu_seqlens": batch.get("cu_seqlens")}
    eager = model._compute_mtp_loss(hidden, next_targets, valid, **kwargs)
    eager[0].backward()
    gradients = [hidden.grad.clone()] + [head.weight.grad.clone() for head in model.lm_head_mtp]
    model.zero_grad(set_to_none=True)
    hidden.grad = None
    compiled = torch.compile(model._compute_mtp_loss, backend="aot_eager", fullgraph=True, dynamic=True)
    actual = compiled(hidden, next_targets, valid, **kwargs)
    for result, reference in zip(actual, eager):
        torch.testing.assert_close(result, reference)
    actual[0].backward()
    for result, reference in zip([hidden.grad] + [head.weight.grad for head in model.lm_head_mtp], gradients):
        torch.testing.assert_close(result, reference)


@pytest.mark.parametrize("packed", [False, True])
def test_mtp_bfloat16_autocast_has_float32_finite_losses_and_gradients(packed, monkeypatch):
    model = _model()
    _set_path(model, monkeypatch, "packed" if packed else "padded")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = model(**_batch(packed))
    for key in ("loss", "output_loss", "mtp_loss", "mtp_losses"):
        assert result[key].dtype == torch.float32
        assert torch.isfinite(result[key]).all()
    result["loss"].backward()
    assert all(torch.isfinite(param.grad).all() for param in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_flash_attention_mtp_training_is_finite_and_matches_padded():
    model = TinyTransformer(TinyTransformerConfig(
        d_model=64, n_heads=4, d_ff=128, n_layers=2, max_seq_len=64,
        num_examples=2, num_task_embedding_latents=2,
        dropout=0.0, embedding_dropout=0.0, attention_dropout=0.0, n_predict=4,
    )).cuda()
    if not model._has_varlen_flash_attention():
        pytest.skip("FlashAttention unavailable")
    padded, packed = [
        {key: value.cuda() if torch.is_tensor(value) else value for key, value in _batch(path).items()}
        for path in (False, True)
    ]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        reference = model(**padded)
        actual = model(**packed)
    for key in ("loss", "output_loss", "mtp_loss", "mtp_losses"):
        assert actual[key].dtype == torch.float32
        torch.testing.assert_close(actual[key], reference[key], atol=0.02, rtol=0.02)
    actual["loss"].backward()
    assert all(param.grad is not None and torch.isfinite(param.grad).all() for param in model.parameters())


@pytest.mark.parametrize("packed", [False, True])
def test_stable_cross_entropy_keeps_saturated_target_gradients(packed, monkeypatch):
    class ConfidentHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.tensor([20.] + [0.] * 13))

        def forward(self, hidden):
            return self.logits.expand(*hidden.shape[:-1], 14)

    model = _model(n_predict=2)
    model.stable_cross_entropy = True
    _set_path(model, monkeypatch, "packed" if packed else "padded")
    model.lm_head = ConfidentHead()
    model.lm_head_mtp = torch.nn.ModuleList([ConfidentHead()])
    batch = _batch(packed, custom_targets=True)
    batch["targets"] = batch["targets"].masked_fill(batch["targets"] != IGNORE_INDEX, 0)
    result = model(**batch)
    result["loss"].backward()
    for key in ("output_loss", "mtp_loss"):
        torch.testing.assert_close(result[key], torch.tensor(2.6794997e-8), rtol=2e-6, atol=0)
    for head in (model.lm_head, *model.lm_head_mtp):
        assert head.logits.grad[0] < 0
        assert (head.logits.grad[1:] > 0).all()
        assert head.logits.grad.sum().abs() < 1e-14
