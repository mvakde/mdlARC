import json
import random
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from common import (ARCExampleDataset, END_PAIR_TOKEN_ID, IO_SEPARATOR_TOKEN_ID,
                    compute_positions_3d, encode_context)
from group_attention import role_positions, padded_prefix_structure, packed_prefix_layout
from dataset_building_scripts.intermediate_grids import candidate_grids, recover_saved_originals, encoded, sha, pair_key


def frame(grid, label="paint(...)", **state):
    return {"label": label, "state": {"grid": grid, "background": 0, **state}}


def test_selector_preserves_crops_and_discards_resets_and_duplicates():
    initial, final = [[0, 1]], [[4, 4]]
    crop, middle = [[1]], [[2, 0]]
    trace = [frame(initial), frame(initial), frame([[0, 0]], "canvas(...)"),
             frame(crop, "crop(...)"), frame(crop), frame(middle), frame(final), frame(final)]
    assert candidate_grids(trace, initial, final) == [crop, middle]
    many = [frame(initial)] + [frame([[i, 0]]) for i in range(1, 7)] + [frame(final)]
    assert candidate_grids(many, initial, final) == [[[i, 0]] for i in range(1, 7)]


def test_filter_understands_clear_operations_but_keeps_useful_erasures_and_crops():
    initial, final = [[1, 2, 3]], [[4, 4, 4]]
    erased, cut, crop, background_crop = [[1, 0, 3]], [[1, 0, 0]], [[5]], [[0]]
    blank = frame([[9, 9, 9]], "each 1/2 · patch = cut(...)", background=9)
    trace = [frame(initial), frame(erased, "erase(...)"), frame(cut, "cut(...)"),
             blank, {**blank, "label": "select(...)", "state": {**blank["state"], "selection": [0]}},
             frame(crop, "crop(...)"), frame(background_crop, "crop(...)"), frame(final)]
    counts = Counter()
    assert candidate_grids(trace, initial, final, counts=counts) == [erased, cut, crop, background_crop]
    assert counts["blank_reset_or_clear_frames"] == 1
    # Native loop prefixes must not hide a canvas reset; later metadata cannot
    # reintroduce it under the label of a non-clearing operation.
    reset = frame([[0, 0]], "repeat 1/2 · canvas(...)")
    assert candidate_grids([frame(initial), reset, frame([[0, 0]], "point(...)", pointer=[1, 1]),
                            frame(final)], initial, final) == []


def test_filter_removes_only_fully_reverted_work_not_work_saved_in_hidden_state():
    initial, final, detour, useful = [[1, 2]], [[4, 4]], [[7, 2]], [[1]]
    trace = [frame(initial), frame(detour), frame(initial), frame(useful, "crop(...)"), frame(final)]
    assert candidate_grids(trace, initial, final) == [useful]
    # The visible grid returns, but the extracted result is saved for later.
    trace[2] = frame(initial, "paste(...)", clipboard={"grid": detour})
    assert candidate_grids(trace, initial, final) == [detour, useful]


def test_random_candidates_replay_per_visit_and_match_visible_palette_and_inference(tmp_path):
    from context_palette import protection_schedule
    from evaluate import _build_inference_attempts

    # Colors 2..7 occur only in the candidate pools; 9 only in the gold target.
    pool = [[[i]] for i in range(2, 8)]
    demo = {"input": [[0]], "output": [[1]], "intermediate_grid_candidates": pool}
    target = {"input": [[0]], "output": [[9]], "intermediate_grid_candidates": [[[9]]]}
    path = tmp_path / "pool.json"
    path.write_text(json.dumps({"t": {"train": [demo, demo, target], "test": [{"input": [[0]]}]}}))
    dataset = ARCExampleDataset(path, seed=42, splits=("train", "test"), attention_pattern="pair_group",
        training_demo_schedule="two", inference_demo_policy="two", augmentation_sampling_mode="context_palette")
    example = dataset.examples[2]
    before_rng, before_torch = random.getstate(), torch.get_rng_state()
    signatures = set()
    protected = protection_schedule(dataset, example, 24)
    for visit in range(24):
        choices = dataset.intermediate_grid_indices(example, 0, visit)
        assert choices == dataset.intermediate_grid_indices(example, 1, visit)
        assert 0 <= choices[0] < 3 <= choices[1] < 6
        signatures.add(choices)
        context, sep = dataset.build_context_sequence(example, target, 0, None, visit=visit)
        replay, _ = dataset.build_context_sequence(example, target, 0, None, visit=visit)
        assert torch.equal(context, replay)
        observed = set(context[:sep + 1].tolist()) & set(range(10))
        assert observed == {0, 1, *(i + 2 for i in choices)}
        assert 9 not in observed and 9 in protected[visit]
        assert set(protected[visit]) == set(dataset.task_colors[example.task_id]) - observed
        # The training path uses the same sampler visit, not the augmentation
        # index (which may wrap at 80) or a mutable worker-local RNG counter.
        train_example = replace(example, demo_visit_override=visit, augment_epoch_override=visit % 8)
        training, train_sep, _ = dataset.build_training_sequence(train_example, 0, None)
        assert torch.equal(context, training) and sep == train_sep
    assert len(signatures) > 1
    assert random.getstate() == before_rng and torch.equal(torch.get_rng_state(), before_torch)
    # Inference passes its attempt visit through the same context builder.
    attempts = _build_inference_attempts(dataset, split="test", use_aug=True, attempt_count=8,
                                        task_ids=None, pair_index=None)
    assert len(attempts) == 8
    for visit, attempt in enumerate(attempts):
        ex = next(dataset.iter_examples(split="test"))
        expected, _ = dataset.build_context_sequence(ex, {"input": [[0]]}, attempt.dihedral_index,
                                                     torch.tensor(attempt.color_mapping), visit=visit)
        assert attempt.prompt == expected.tolist()


def test_catalog_omission_recovers_only_requested_originals_with_pinned_artifacts(tmp_path):
    from types import SimpleNamespace
    owner = tmp_path
    catalog_path = owner / "data/programs/catalog.json"
    run = owner / "results/saved/arc2/evaluation/task/runs/original"
    run.mkdir(parents=True)
    query = {"task_id": "task-variations/arc2/evaluation/task", "pair_id": "query-000",
             "input": [[1]], "output": [[2]], "program": "background(0)\nselect(all())\npaint(2)"}
    other = {**query, "pair_id": "query-001", "input": [[3]]}
    raw = encoded(query) + b"\n" + encoded(other) + b"\n"
    label_path = run / "primary-labels.unassigned.jsonl"
    label_path.write_bytes(raw)
    manifest = {"status": "verified", "puzzle_source": "original",
                "artifact_sha256": {label_path.name: sha(raw)}}
    manifest_path = run / "manifest.json"
    manifest_path.write_bytes(encoded(manifest))
    provenance = {"run": {"path": str(manifest_path.relative_to(owner)),
                          "manifest_sha256": sha(encoded(manifest))}}
    reader = SimpleNamespace(verified_archive=lambda *_: ({"provenance.json": encoded(provenance)}, {}))
    catalog = {"archive": {"sha256": "archive"}, "banks": {"originals": {}}}
    labels, pins = {}, {}
    restored = recover_saved_originals(reader, catalog_path, catalog, ["originals"],
                                      {"task": {pair_key(query)}}, labels, pins)
    assert labels == {pair_key(query): query["program"]}
    assert len(restored) == 1 and str(label_path) in pins
    label_path.write_bytes(raw + b"\n")
    with pytest.raises(ValueError, match="label hash changed"):
        recover_saved_originals(reader, catalog_path, catalog, ["originals"],
                                {"task": {pair_key(query)}}, {}, {})


def test_grid_chains_share_groups_and_never_expose_target_trace(tmp_path):
    demo = {"input": [[1]], "intermediate_grids": [[[2]], [[3]]], "output": [[4]]}
    other = {"input": [[5]], "intermediate_grids": [[[6]]], "output": [[7]]}
    target = {"input": [[8]], "intermediate_grids": [[[9]]], "output": [[0]]}
    tokens, separator = encode_context([demo, other], target)
    assert 9 not in tokens.tolist()
    assert tokens.tolist().count(END_PAIR_TOKEN_ID) == 3
    assert tokens.tolist().count(IO_SEPARATOR_TOKEN_ID) == 6
    positions = compute_positions_3d(tokens[None])
    target_z = positions[:, separator, 2:3]
    roles = role_positions(positions, target_z, tokens[None], ordered=True)[0, :, 2]
    assert [int(roles[(tokens == color).nonzero()[0]]) for color in (1, 2, 3, 4, 5, 6, 7, 8, 0)] == [1, 2, 3, 4, 1, 3, 4, 5, 6]
    groups, hub = padded_prefix_structure(tokens[None], torch.ones_like(tokens[None], dtype=torch.bool), torch.tensor([separator + 1]))
    assert len(set(groups[0, (tokens >= 1) & (tokens <= 4)].tolist())) == 1
    assert not hub[0, tokens == 2].any()
    # Packed sequence resets must not let one chain borrow the next chain's roles.
    packed = torch.cat((tokens, tokens))
    raw_positions = torch.cat((positions[0], positions[0]))
    packed_roles = role_positions(raw_positions, target_z.expand(1, len(packed)).flatten(), packed, ordered=True)
    assert torch.equal(packed_roles[:len(tokens)], packed_roles[len(tokens):])
    layout = packed_prefix_layout(packed, torch.tensor([0, len(tokens), len(packed)], dtype=torch.int32), torch.tensor([separator + 1, separator + 1]))
    assert len(layout["prefix_cu_seqlens"]) - 1 == 6  # Two demos + target per sequence.
    path = tmp_path / "challenges.json"
    path.write_text(json.dumps({"t": {"train": [demo, other, target]}}))
    dataset = ARCExampleDataset(json_path=path, splits=("train",), attention_pattern="pair_group", training_demo_schedule="two")
    example = dataset.examples[2]
    sequence, sep, _ = dataset.build_training_sequence(example, 0, None)
    assert len(sequence) <= dataset.max_seq_len
    assert sequence[sep + 1:].tolist() == [0, END_PAIR_TOKEN_ID]
    mapping = torch.arange(14)
    mapping[:10] = torch.arange(10).flip(0)
    transformed, _ = encode_context([demo], {"input": [[8]]}, 1, mapping)
    assert [v for v in transformed.tolist() if v < 10] == [8, 7, 6, 5, 1]


@pytest.mark.parametrize("middle_count", [0, 1, 2])
def test_chain_forward_backward_and_cached_decode_agree(middle_count):
    from dataclasses import asdict
    from build import build_checkpoint_config
    from evaluate import _derive_initial_state_from_prompt, _speculative_grid_positions
    from tinytransformer import TinyTransformer, TinyTransformerConfig
    torch.manual_seed(2)
    model = TinyTransformer(TinyTransformerConfig(
        d_model=48, n_heads=4, d_ff=96, n_layers=2,
        num_examples=1, task_embeddings=False, dihedral_embeddings=False,
        attention_pattern="pair_group", dropout=0, embedding_dropout=0, attention_dropout=0,
        ordered_grid_positions=True, split_target_demo_attention=True,
        n_predict=1, FA=2,
    ))
    assert build_checkpoint_config({"config": asdict(model.config)}).ordered_grid_positions
    demo = {"input": [[1]], "intermediate_grids": [[[2]], [[3]]][:middle_count], "output": [[4]]}
    sequence, sep = encode_context([demo], {"input": [[5]], "output": [[6, 7]]})
    positions = compute_positions_3d(sequence[None])
    mapped = role_positions(positions, positions[:, sep, 2:3], sequence[None], ordered=True)
    colors = sequence < 10
    expected_demo = {0: [1, 4], 1: [1, 3, 4], 2: [1, 2, 3, 4]}[middle_count]
    assert mapped[0, colors, 2].tolist() == expected_demo + [5, 6, 6]
    torch.testing.assert_close(mapped[..., :2], positions[..., :2])
    prompt = sequence[:sep + 1]
    model.eval()
    args = dict(input_ids=sequence[None], positions_3d=positions,
                example_ids=torch.tensor([0]))
    full = model(**args, sep_indices=torch.tensor([sep]))
    cached = model.forward_generate(prompt[None], torch.tensor([0]), positions_3d=positions[:, :sep + 1])
    torch.testing.assert_close(full["logits"][:, sep], cached["logits"][:, -1], atol=1e-5, rtol=1e-4)
    width = len(prompt)
    output = sequence[sep + 1:][None]
    capacity = len(sequence)
    cache = tuple(tuple(torch.cat((t, t.new_zeros((*t.shape[:2], capacity - width, t.size(-1)))), dim=2)
                        for t in layer) for layer in cached["past_key_values"])
    state, _ = _derive_initial_state_from_prompt(prompt[None], positions[:, :width])
    _, output_positions = _speculative_grid_positions(state, output)
    decoded = model.forward_generate(output, torch.tensor([0]), past_key_values=cache,
        positions_3d=output_positions, attention_mask=torch.ones((1, capacity), dtype=torch.bool),
        demo_attention_mask=(positions[..., 2] > 0) & (positions[..., 2] < positions[:, sep, 2:3]),
        cache_position=torch.arange(width, capacity))
    torch.testing.assert_close(full["logits"][:, width:], decoded["logits"], atol=1e-5, rtol=1e-4)
    full["logits"].sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
