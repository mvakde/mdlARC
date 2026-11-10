import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import build
import evaluate
from common import ARCExampleDataset, IO_SEPARATOR_TOKEN_ID, MAX_SEQ_LEN, MAX_PAIR_LEN, SEQUENCE_FORMAT, encode_pair, encode_context
from tinytransformer import TinyTransformer, TinyTransformerConfig, build_output_targets_and_mask


@pytest.mark.parametrize("split", ["train", "test"])
def test_one_target_is_preserved_with_two_other_training_pairs(tmp_path, split):
    target = {"input": [[1, 2]], "output": [[2, 1]]}
    other = {"input": [[8] * 30] * 30, "output": [[9] * 30] * 30}
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"task": {"train": [target] + [other] * 12, "test": [target]}}))
    dataset = ARCExampleDataset(path, splits=(split,), max_augments=8)
    example = next(dataset.iter_examples(split=split))
    demos = [other, other] if split == "train" else [dataset.pairs_by_task["task"][i]
             for i in dataset.demonstration_indices(example, 0)]
    expected, sep = encode_context(demos, target)
    tokens, actual_sep, positions = dataset.build_training_sequence(example, 0, None)
    torch.testing.assert_close(tokens, expected)
    assert actual_sep == sep
    assert tokens.tolist().count(IO_SEPARATOR_TOKEN_ID) == 3
    assert set(positions[:, 2].tolist()) == set(range(7))
    attempts = evaluate._build_inference_attempts(dataset, split, None, 0, 1, False)
    assert attempts[0].prompt == expected[:sep + 1].tolist()
    assert attempts[0].task_embedding_ids == example.task_embedding_ids


def test_largest_pair_fits_and_old_checkpoints_require_fresh_training():
    tokens, _ = encode_pair({"input": [[1] * 30] * 30, "output": [[2] * 30] * 30})
    assert len(tokens) == MAX_PAIR_LEN
    largest, _ = encode_context([{"input": [[1] * 30] * 30, "output": [[2] * 30] * 30}] * 2,
                                {"input": [[1] * 30] * 30, "output": [[2] * 30] * 30})
    assert len(largest) == MAX_SEQ_LEN == 5581
    for old_format in (None, "multi_pair_context_v1", "single_pair_bidirectional_v1"):
        with pytest.raises(ValueError, match="fresh training"):
            build._validate_checkpoint_format({"sequence_format": old_format})
    build._validate_checkpoint_format({"sequence_format": SEQUENCE_FORMAT})


def test_input_is_bidirectional_output_is_causal_and_only_output_is_supervised():
    model = TinyTransformer(TinyTransformerConfig(
        d_model=24, n_heads=4, d_ff=48, n_layers=1, num_examples=1,
        num_task_embedding_latents=2,
    ))
    tokens, sep = encode_context([{"input": [[5]], "output": [[6]]},
                                  {"input": [[7]], "output": [[8]]}],
                                 {"input": [[1, 2]], "output": [[3, 4]]})
    allowed = torch.isfinite(model._build_prefix_lm_sdpa_mask(
        torch.ones(1, len(tokens), dtype=torch.bool), torch.tensor([sep + 1]), torch.float32,
    ))[0, 0]
    assert allowed[:sep + 1, :sep + 1].all()
    assert not allowed[:sep + 1, sep + 1:].any()
    for query in range(sep + 1, len(tokens)):
        assert allowed[query, :query + 1].all()
        assert not allowed[query, query + 1:].any()
    targets, valid = build_output_targets_and_mask(tokens[None], torch.tensor([sep]))
    assert not valid[0, :sep].any()
    assert valid[0, sep:].all()
    assert targets[0, valid[0]].tolist() == tokens[sep + 1:].tolist()
