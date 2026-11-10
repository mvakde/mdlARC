"""Balanced demos, unchanged targets, joint augmentation and task ablation."""
from argparse import Namespace
from collections import Counter
from dataclasses import replace
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
import evaluate
import launch_run
import train
import variations
from common import (
    ARCExampleDataset, SEQUENCE_FORMAT, START_TOKEN_ID,
    collate_examples, compute_positions_3d, create_dataloader, encode_context,
    encode_pair, select_balanced_demonstrations,
)
from tinytransformer import TinyTransformer, TinyTransformerConfig
from test_mtp import _batch, _set_path


@pytest.mark.parametrize("count", range(2, 16))
def test_every_schedule_prefix_balances_total_use_and_both_positions(count):
    eligible = list(range(count))
    totals, slots = Counter(), [Counter(), Counter()]
    schedule = []
    for visit in range(count * 7 + 1):
        row = select_balanced_demonstrations(eligible, visit=visit, seed=42)
        assert len(row) == len(set(row)) == 2
        totals.update(row)
        for position, index in enumerate(row):
            slots[position][index] += 1
        for counts in (totals, *slots):
            values = [counts[i] for i in eligible]
            assert max(values) - min(values) <= 1
        assert all(abs(slots[0][i] - slots[1][i]) <= 1 for i in eligible)
        schedule.append(row)
    # Stateless, resumable and independent of call order; seed changes the order.
    assert list(reversed(schedule)) == [select_balanced_demonstrations(eligible, visit=v, seed=42)
                                       for v in reversed(range(len(schedule)))]
    assert schedule != [select_balanced_demonstrations(eligible, visit=v, seed=43)
                        for v in range(len(schedule))]


def _dataset(tmp_path, count=5, **options):
    pairs = [{"input": [[i, i + 1], [i + 1, i]], "output": [[i + 2, i]]}
             for i in range(count)]
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"task": {"train": pairs, "test": [
        {"input": [[7, 8]], "output": [[9]]},
        {"input": [[8, 7]], "output": [[9, 9]]},
    ]}}))
    return ARCExampleDataset(path, seed=42, **options)


@pytest.mark.parametrize("split", ["train", "test"])
def test_context_excludes_target_and_test_answers_with_shared_augmentation(tmp_path, split):
    dataset = _dataset(tmp_path)
    example = next(dataset.iter_examples(split=split))
    pairs = dataset.pairs_by_task["task"]
    target = (dataset.test_pairs_by_task["task"] if split == "test" else pairs)[example.pair_index]
    mapping = torch.arange(14)
    mapping[:10] = torch.arange(9, -1, -1)
    for visit in range(13):
        indices = dataset.demonstration_indices(example, visit)
        assert len(set(indices)) == 2
        if split == "train":
            assert example.pair_index not in indices
        item = replace(example, augment_epoch_override=visit)
        tokens, sep, positions = dataset.build_training_sequence(item, 1, mapping)
        expected, expected_sep = encode_context([pairs[i] for i in indices], target, 1, mapping)
        torch.testing.assert_close(tokens, expected)
        assert sep == expected_sep
        assert tokens.tolist().count(START_TOKEN_ID) == 1
        single, single_sep = encode_pair(target, 1, mapping)
        torch.testing.assert_close(tokens[sep - single_sep + 1:], single[1:])
        assert positions[-1, 2].item() == 6
    attempts = evaluate._build_inference_attempts(dataset, split, ["task"], 0, 13, True)
    assert len(attempts) == 13
    for visit, item in enumerate(attempts):
        expected, sep = encode_context(
            [pairs[i] for i in dataset.demonstration_indices(example, visit)],
            {"input": target["input"]}, item.dihedral_index, torch.tensor(item.color_mapping),
        )
        assert item.prompt == expected.tolist()
        assert len(item.prompt) == sep + 1
    # Accuracy evaluation must attach only the target output to the same prompt.
    accuracy = evaluate._build_accuracy_attempts(dataset, split, attempts, {}, dataset.max_seq_len)
    assert all(item.tokens[:len(item.inference.prompt)].tolist() == item.inference.prompt for item in accuracy)


@pytest.mark.parametrize("count", [1, 2])
def test_small_tasks_use_available_other_pairs_without_dropping_targets(tmp_path, count):
    dataset = _dataset(tmp_path, count=count, splits=("train",))
    assert len(dataset) == count
    for example in dataset.examples:
        assert len(dataset.demonstration_indices(example, 0)) == count - 1
        tokens, sep, _ = dataset.build_training_sequence(example, 0, None)
        torch.testing.assert_close(tokens[sep + 1:], example.tokens[example.sep_index + 1:])


def _model(enabled=True):
    torch.manual_seed(7)
    return TinyTransformer(TinyTransformerConfig(
        d_model=24, n_heads=4, d_ff=48, n_layers=2, num_examples=2,
        num_task_embedding_latents=3, n_predict=4, dropout=0,
        attention_dropout=0, task_embeddings=enabled,
    ))


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("path", ["padded", "fallback", "packed"])
def test_context_loss_and_gradients_match_target_only_reference(enabled, path, monkeypatch):
    model = _model(enabled)
    sequences = [encode_context([{"input": [[5]], "output": [[6]]},
                                 {"input": [[7, 8]], "output": [[9]]}], target)
                 for target in ({"input": [[1]], "output": [[2, 3]]},
                                {"input": [[3, 4]], "output": [[5], [6]]})]
    batch = _batch(sequences=sequences)
    reference = model(**batch)
    logits = torch.cat([reference["logits"][i, sep:len(tokens) - 1]
                        for i, (tokens, sep) in enumerate(sequences)])
    targets = torch.cat([tokens[sep + 1:] for tokens, sep in sequences])
    torch.testing.assert_close(reference["output_loss"], F.cross_entropy(logits, targets))
    _set_path(model, monkeypatch, path)
    result = model(**_batch(packed=path != "padded", sequences=sequences))
    torch.testing.assert_close(result["loss"], reference["loss"])
    result["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)


@torch.inference_mode()
@pytest.mark.parametrize("compiled", [False, True])
def test_disabled_task_conditioning_is_inert_in_training_prefill_and_cached_decode(compiled):
    enabled, model = _model(True), _model(False).eval()
    for name, weight in enabled.state_dict().items():
        torch.testing.assert_close(weight, model.state_dict()[name], rtol=0, atol=0)
    tokens, sep = encode_context([{"input": [[5]], "output": [[6]]}] * 2,
                                 {"input": [[1, 2]], "output": [[3, 4]]})
    positions = compute_positions_3d(tokens[None])
    ids = torch.tensor([[0, -1]])
    with torch.no_grad():
        for block in model.blocks:
            block.example_input_skip_scale.fill_(1)
    common = dict(example_ids=ids)
    full = model(input_ids=tokens[None], positions_3d=positions,
                 sep_indices=torch.tensor([sep]), **common)["logits"]
    other = model(input_ids=tokens[None], positions_3d=positions,
                  sep_indices=torch.tensor([sep]), example_ids=torch.tensor([[1, 0]]))["logits"]
    torch.testing.assert_close(full, other, rtol=0, atol=0)
    assert not model.compute_conditioning_embeds(ids).any()
    assert all(value is None for value in model.compute_value_embeds(ids))
    cached = dict(conditioning_embeds=model.compute_conditioning_embeds(ids),
                  value_embeds=model.compute_value_embeds(ids))
    prefill = model.forward_generate(input_ids=tokens[None, :sep + 1],
                                      positions_3d=positions[:, :sep + 1], **common, **cached)
    torch.testing.assert_close(prefill["logits"][0, -1], full[0, sep])
    state, _ = evaluate._derive_initial_state_from_prompt(tokens[None, :sep + 1], positions[:, :sep + 1])
    assert state.tolist() == [[0, 0, 6]]
    _, streamed = evaluate._speculative_grid_positions(state, tokens[None, sep + 1:])
    torch.testing.assert_close(streamed, positions[:, sep + 1:])
    capacity = len(tokens)
    cache = tuple(tuple(torch.cat((t, t.new_zeros(1, t.size(1), capacity - sep - 1, t.size(3))), dim=2)
                        for t in pair) for pair in prefill["past_key_values"])
    decode = torch.compile(model.forward_generate, backend="aot_eager", fullgraph=True) if compiled else model.forward_generate
    for index in range(sep + 1, capacity):
        step = decode(
            input_ids=tokens[None, index:index + 1], positions_3d=positions[:, index:index + 1],
            past_key_values=cache, cache_position=torch.tensor([index]),
            attention_mask=(torch.arange(capacity) <= index)[None], **common, **cached,
        )
        torch.testing.assert_close(step["logits"][0, -1], full[0, index])


@pytest.mark.parametrize("enabled", [False, True])
def test_task_switch_checkpoint_roundtrip_and_resume_rules(tmp_path, monkeypatch, enabled):
    dataset = _dataset(tmp_path)
    monkeypatch.setattr(build, "resolve_device", lambda _: torch.device("cpu"))
    args = Namespace(
        seed=42, device="cpu", checkpoint_path=None, data_path=dataset.source_path,
        batch_size=2, num_workers=0, enable_aug=False, max_augments=80,
        task_embeddings=enabled, d_model=24, n_heads=4, d_ff=48, n_layers=2,
        n_predict=4, num_task_embedding_latents=3, dropout=0, attention_dropout=0,
    )
    model, dataset, loader, _, _ = build.build_model_and_data(args)
    batch = next(iter(loader))
    result = model(**{key: value for key, value in batch.items()
                      if key not in ("has_padding", "task_ids", "splits", "has_output")})
    result["loss"].backward()
    muon, adamw = train._build_param_groups(model, .1, .01, .01, .01)
    optimized = {id(p) for group in [*muon, *adamw] for p in group["params"]}
    assert optimized == {id(p) for p in model.parameters() if p.requires_grad}
    path = tmp_path / "model.pt"
    train.maybe_save_model(model, dataset, dataset.source_path, path)
    checkpoint = build.load_checkpoint(path)
    assert checkpoint["sequence_format"] == SEQUENCE_FORMAT
    config = build.build_checkpoint_config(checkpoint)
    assert config.task_embeddings is enabled
    args.task_embeddings = None
    restored, eval_dataset, _, _, _ = build.build_model_and_data(
        args, checkpoint=checkpoint, is_eval=True, build_dataloader=False,
    )
    for name, weight in model.state_dict().items():
        torch.testing.assert_close(weight, restored.state_dict()[name], rtol=0, atol=0)
    attempts = evaluate._build_inference_attempts(eval_dataset, "test", None, None, 1, False)
    assert all(item.prompt.count(START_TOKEN_ID) == 1 for item in attempts)
    assert restored.config.dihedral_embeddings is False
    for mode in ("resume", "finetune"):
        with pytest.raises(ValueError, match="fresh models only"):
            build._configure_checkpoint_mtp(config, Namespace(checkpoint_mode=mode, task_embeddings=not enabled), is_eval=False)
        assert build._configure_checkpoint_mtp(config, Namespace(task_embeddings=not enabled), is_eval=True) is config


def test_persistent_worker_and_resume_do_not_change_context_or_target_order(tmp_path):
    dataset = _dataset(tmp_path, splits=("train",))
    direct = create_dataloader(dataset, 2, seed=42, num_workers=0, use_length_bucketing=False)
    workers = create_dataloader(dataset, 2, seed=42, num_workers=1,
                                persistent_workers=True, pin_memory=False, use_length_bucketing=False)
    for epoch in (0, 1, 7, 0):
        direct.batch_sampler.set_epoch(epoch)
        workers.batch_sampler.set_epoch(epoch)
        for left, right in zip(direct, workers, strict=True):
            for key in ("input_ids", "positions_3d", "sep_indices", "example_ids", "cu_seqlens"):
                torch.testing.assert_close(left[key], right[key])


@pytest.mark.parametrize("source", ["rearc", "nvarc"])
def test_lazy_synthetic_targets_take_demos_from_their_own_pair_pool(tmp_path, source):
    base = _dataset(tmp_path, splits=("train",))
    pairs = list(base.pairs_by_task["task"])
    if source == "rearc":
        path = tmp_path / "rearc.json"
        path.write_text(json.dumps({"synthetic": {"train": pairs}}))
        options = dict(rearc_json_path=path)
        task_id = "synthetic"
    else:
        path = tmp_path / "nvarc"
        path.mkdir()
        (path / "first_second.json").write_text(json.dumps(pairs))
        options = dict(nvarc_training_dir=path)
        task_id = "first_second"
    dataset = ARCExampleDataset(base.source_path, seed=42, splits=("train",), **options)
    task_index = dataset.task_ids.index(task_id)
    slots = [Counter(), Counter()]
    for visit in range(12):
        example = dataset[(source, task_index, 0, visit)]
        assert getattr(example, "is_" + source)
        indices = dataset.demonstration_indices(example, visit)
        assert 0 not in indices
        for slot, index in zip(slots, indices):
            slot[index] += 1
        tokens, sep, _ = dataset.build_training_sequence(example, 0, None)
        expected, expected_sep = encode_context([pairs[i] for i in indices], pairs[0])
        torch.testing.assert_close(tokens, expected)
        assert sep == expected_sep
    assert all(len(set(slot.values())) == 1 for slot in slots)


def test_variations_and_repeated_supports_advance_demos_per_visit(tmp_path, monkeypatch):
    base = _dataset(tmp_path, splits=("train",))
    monkeypatch.setattr(variations, "ROOT", tmp_path)
    pairs = [{"input": [[i]], "output": [[i + 1]]} for i in range(5)]
    for seed in range(2):
        (tmp_path / f"seed{seed}.json").write_text(json.dumps({"train": pairs, "test": []}))
    index = tmp_path / "index.json"
    index.write_text(json.dumps({
        "snapshot": "task-variations-v1", "sampling": {"variations_per_parent_per_epoch": 1},
        "parents": {"task": [{"seed": i, "path": f"seed{i}.json", "pair_count": 5} for i in range(2)]},
        "eval_support_task_ids": ["task"],
    }))
    dataset = variations.VariationTrainingDataset(json_path=base.source_path, index_path=index,
                                                seed=42, max_augments=80)
    sampler = variations.VariationBatchSampler(dataset, 4, seed=42,
        variations_per_parent_per_epoch=1, eval_support_repeats_per_epoch=3)
    visits = {}
    for epoch in range(8):
        for batch in (sampler._indices_for_epoch(epoch),):
            for item in batch:
                example = dataset[item]
                key = (example.variation_path, example.pair_index)
                visits.setdefault(key, []).append(example.demo_visit_override)
                if example.variation_path is not None:
                    actual, sep, _ = dataset.build_training_sequence(example, 0, None)
                    selected = dataset.demonstration_indices(example, example.demo_visit_override)
                    expected, expected_sep = encode_context([pairs[i] for i in selected], pairs[example.pair_index])
                    torch.testing.assert_close(actual, expected)
                    assert sep == expected_sep
    for (_, _), values in visits.items():
        assert sorted(values) == list(range(len(values)))


@pytest.mark.parametrize("value", [1, "false", {}])
def test_task_switch_requires_boolean(value):
    with pytest.raises(ValueError, match="task_embeddings"):
        TinyTransformerConfig(task_embeddings=value)
    with pytest.raises(ValueError, match="task_embeddings"):
        launch_run.training_overrides({"task_embeddings": value}, "defaults")
