"""Transform labels disambiguate symmetric inputs across all model paths."""
from argparse import Namespace
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import build
import evaluate
import launch_run
import train
from common import ARCExampleDataset, END_PAIR_TOKEN_ID, apply_dihedral_transform, collate_examples, encode_pair
from tinytransformer import TinyTransformer, TinyTransformerConfig
from test_mtp import _batch, _set_path


def _model(**options):
    torch.manual_seed(7)
    options.setdefault("dihedral_embeddings", True)
    return TinyTransformer(TinyTransformerConfig(
        d_model=24, n_heads=4, d_ff=48, n_layers=2, max_seq_len=64,
        num_examples=2, num_task_embedding_latents=2,
        dropout=0, embedding_dropout=0, attention_dropout=0, **options,
    ))


def _dataset(tmp_path):
    pair = {"input": [[1, 1], [1, 1]], "output": [[1, 2], [3, 4]]}
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"task": {"train": [pair], "test": [pair]}}))
    return ARCExampleDataset(path, splits=("train", "test"), max_augments=8), pair


def test_restoration_preserves_shared_initial_weights_for_controlled_ablations():
    baseline = _model(dihedral_embeddings=False)
    restored = _model()
    for name, weight in baseline.state_dict().items():
        torch.testing.assert_close(weight, restored.state_dict()[name], rtol=0, atol=0)
    assert restored.dihedral_embedding.weight.shape == (8, 24)
    assert restored.dihedral_gate_logit.item() == -2


@pytest.mark.parametrize("index", [-1, 8, 2**32])
def test_out_of_range_transform_ids_cannot_alias_a_valid_embedding(index):
    with pytest.raises(IndexError):
        _model()(**{**_batch(), "dihedral_ids": torch.tensor([0, index])})


def test_collation_labels_the_actual_transform_and_preserves_ambiguous_pairs(tmp_path):
    dataset, pair = _dataset(tmp_path)
    example = dataset.examples[0]
    examples = [replace(example, preselected_dihedral_index=index) for index in range(8)]
    batch = collate_examples(examples, dataset)
    assert batch["dihedral_ids"].tolist() == list(range(8))
    assert batch["dihedral_ids"].dtype == torch.int32
    prompts, answers = [], []
    for row, index in enumerate(batch["dihedral_ids"].tolist()):
        start, end = batch["cu_seqlens"][row:row + 2].tolist()
        tokens = batch["input_ids"][start:end]
        expected, sep = encode_pair({key: apply_dihedral_transform(grid, index) for key, grid in pair.items()})
        torch.testing.assert_close(tokens, expected)
        prompts.append(tuple(tokens[:sep + 1].tolist()))
        answers.append(tuple(tokens[sep + 1:].tolist()))
    assert len(set(prompts)) == 1 and len(set(answers)) == 8
    assert collate_examples([example], dataset)["dihedral_ids"].tolist() == [0]


@pytest.mark.parametrize("path", ["padded", "fallback", "packed"])
def test_all_eight_embeddings_receive_gradients_and_disambiguate_inputs(tmp_path, monkeypatch, path):
    dataset, _ = _dataset(tmp_path)
    examples = [replace(dataset.examples[0], preselected_dihedral_index=index) for index in range(8)]
    packed = collate_examples(examples, dataset)
    packed = {key: value for key, value in packed.items() if key not in ("has_padding", "task_ids", "splits", "has_output")}
    model = _model(n_predict=4)
    with torch.no_grad():
        for block in model.blocks:
            block.example_input_skip_scale.fill_(.2)
            block.dihedral_input_skip_scale.fill_(.4)
    padded = dict(packed)
    padded.pop("cu_seqlens")
    padded.pop("max_seqlen")
    padded["input_ids"] = packed["input_ids"].reshape(8, -1)
    padded["positions_3d"] = packed["positions_3d"].reshape(8, -1, 3)
    reference = model(**padded)
    reference["loss"].backward()
    gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    _set_path(model, monkeypatch, path)
    result = model(**(padded if path == "padded" else packed))
    torch.testing.assert_close(result["loss"], reference["loss"])
    torch.testing.assert_close(result["logits"].reshape(8, -1, 14), reference["logits"])
    result["loss"].backward()
    for name, param in model.named_parameters():
        assert torch.isfinite(param.grad).all()
        torch.testing.assert_close(param.grad, gradients[name], rtol=3e-4, atol=2e-6)
    assert (model.dihedral_embedding.weight.grad.norm(dim=1) > 0).all()
    assert model.dihedral_gate_logit.grad.abs() > 0
    # First output prediction has identical input tokens and task conditioning.
    logits = result["logits"].reshape(8, -1, 14)[:, packed["sep_indices"][0]]
    assert not torch.allclose(logits[0], logits[1])


@pytest.mark.parametrize("compiled", [False, True])
@torch.inference_mode()
def test_cached_conditioning_prefill_and_decode_match_training_with_separate_shortcuts(compiled):
    model = _model().eval()
    batch = _batch()
    batch["dihedral_ids"] = torch.tensor([2, 7], dtype=torch.int32)
    with torch.no_grad():
        for block in model.blocks:
            block.example_input_skip_scale.fill_(.3)
            block.dihedral_input_skip_scale.fill_(-.4)
    full = model(**batch)["logits"]
    conditioning = model.compute_conditioning_embeds(batch["example_ids"], batch["dihedral_ids"])
    values = model.compute_value_embeds(batch["example_ids"])
    decode = torch.compile(model.forward_generate, backend="aot_eager", fullgraph=True) if compiled else model.forward_generate
    # Generate each unequal-length example independently, using the same IDs
    # and cached conditioning for prefill and every incremental output token.
    for row in range(2):
        length = int(batch["attention_mask"][row].sum())
        prefix = int(batch["sep_indices"][row]) + 1
        common = dict(example_ids=batch["example_ids"][row:row + 1],
                      dihedral_ids=batch["dihedral_ids"][row:row + 1])
        cached = dict(conditioning_embeds=conditioning[row:row + 1],
                      value_embeds=tuple(None if value is None else value[row:row + 1] for value in values))
        args = dict(input_ids=batch["input_ids"][row:row + 1, :prefix],
                    positions_3d=batch["positions_3d"][row:row + 1, :prefix], **common)
        prefill = model.forward_generate(**args, **cached)
        torch.testing.assert_close(prefill["logits"], model.forward_generate(**args)["logits"])
        torch.testing.assert_close(prefill["logits"][0, -1], full[row, prefix - 1])
        buffers = tuple(tuple(torch.cat((value, value.new_zeros(1, value.size(1), length - prefix, value.size(3))), dim=2)
                              for value in pair) for pair in prefill["past_key_values"])
        for position in range(prefix, length):
            decoded = decode(
                input_ids=batch["input_ids"][row:row + 1, position:position + 1],
                positions_3d=batch["positions_3d"][row:row + 1, position:position + 1],
                past_key_values=buffers, cache_position=torch.tensor([position]),
                attention_mask=(torch.arange(length) <= position)[None], **common, **cached,
            )
            torch.testing.assert_close(decoded["logits"][0, -1], full[row, position], rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("enabled", [False, True])
def test_checkpoint_roundtrip_and_legacy_architecture_are_preserved(enabled):
    source = _model(dihedral_embeddings=enabled)
    config = asdict(source.config)
    if not enabled:
        del config["dihedral_embeddings"]  # Actual pre-restoration config.
    checkpoint = dict(config=config, model_state=source.state_dict(), embedding_task_ids=["a", "b"])
    restored = TinyTransformer(build.build_checkpoint_config(checkpoint))
    build._load_model_state(restored, checkpoint, ["a", "b"])
    assert restored.config.dihedral_embeddings == enabled
    torch.testing.assert_close(source(**_batch())["logits"], restored(**_batch())["logits"], rtol=0, atol=0)
    if not enabled:
        assert not any("dihedral" in name for name in restored.state_dict())
        batch = _batch()
        batch.pop("dihedral_ids")
        torch.testing.assert_close(restored(**batch)["logits"], source(**_batch())["logits"], rtol=0, atol=0)
    for mode in ("resume", "finetune"):
        with pytest.raises(ValueError, match="fresh models only"):
            build._configure_checkpoint_mtp(restored.config, Namespace(checkpoint_mode=mode, dihedral_embeddings=not enabled), is_eval=False)


def test_dihedral_parameters_use_their_adamw_settings(monkeypatch):
    model = _model()
    monkeypatch.setattr(train, "_normuon_supported", lambda device: (True, ""))
    args = Namespace(optimizer="normuon", normuon_lr=.002, adamw_lr=.003, weight_decay=.1,
                     dihedral_embedding_weight_decay=.07, adamw_dihedral_embedding_lr_multiplier=.4,
                     adamw_dihedral_embedding_lr_floor=.2)
    optimizer = train._build_optimizer(args, model, torch.device("cpu"), .01, .01, .01)
    group = next(group for group in optimizer.param_groups if group["group_name"] == "adamw_dihedral_embed")
    assert group["params"] == [model.dihedral_embedding.weight]
    assert not group["use_muon"] and group["weight_decay"] == .07
    assert group["lr"] == pytest.approx(.0012)
    assert train._lr_floor_for_group(args, group) == .2
    parameters = [id(param) for group in optimizer.param_groups for param in group["params"]]
    assert len(parameters) == len(set(parameters)) == len(list(model.parameters()))
    fixed = _model(input_shortcuts="fixed")
    assert all(not block.dihedral_input_skip_scale.requires_grad for block in fixed.blocks)


def test_both_evaluation_modes_forward_each_attempts_transform(tmp_path, monkeypatch):
    dataset, _ = _dataset(tmp_path)
    model = _model()
    monkeypatch.setattr(evaluate, "_prepare_model_for_inference", lambda model: model)
    seen = []

    def generate(**kwargs):
        seen.extend(kwargs["dihedral_ids"])
        return [prompt + [END_PAIR_TOKEN_ID] for prompt in kwargs["prompts"]]

    monkeypatch.setattr(evaluate, "batched_greedy_generate", generate)
    results = evaluate.run_split_inference(model, dataset, "test", torch.device("cpu"), augmentation_attempts=8)
    assert seen == [result["dihedral_index"] for result in results]
    assert set(seen) == set(range(8))
    seen.clear()
    actual_forward = model.forward

    def forward(**kwargs):
        seen.extend(kwargs["dihedral_ids"].tolist())
        return actual_forward(**kwargs)

    monkeypatch.setattr(model, "forward", forward)
    evaluate.run_split_accuracy(model, dataset, "test", torch.device("cpu"), {}, augmentation_attempts=8)
    assert set(seen) == set(range(8)) and len(seen) == 8


@pytest.mark.parametrize("ids", [None, torch.tensor([[0], [1]]), torch.tensor([0]), torch.tensor([0., 1.])])
def test_enabled_models_reject_missing_or_malformed_transform_labels(ids):
    with pytest.raises(ValueError, match="dihedral_ids"):
        _model()(**{**_batch(), "dihedral_ids": ids})


@pytest.mark.parametrize("value", ["true", 1, {}])
def test_config_requires_a_boolean(value):
    with pytest.raises(ValueError, match="dihedral_embeddings"):
        TinyTransformerConfig(dihedral_embeddings=value)
    with pytest.raises(ValueError, match="dihedral_embeddings"):
        launch_run.training_overrides({"dihedral_embeddings": value}, "defaults")
