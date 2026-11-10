"""Regression checks against the previous indexing and row-copy semantics."""

from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import build
import common
import evaluate
from tinytransformer import TinyTransformer, TinyTransformerConfig


def legacy_index(task_ids, *, regular_task_ids=None, nvarc_task_ids=None, embedding_task_whitelist=None):
    """The old union-per-task algorithm, retained only as a parity oracle."""
    regular = set(regular_task_ids or ())
    nvarc = set(nvarc_task_ids or ())
    rows, indices, tasks = [], {}, {}

    def ensure(task_id):
        if task_id not in indices:
            indices[task_id] = len(rows)
            rows.append(task_id)
        return indices[task_id]

    for task_id in embedding_task_whitelist or ():
        ensure(str(task_id))
    for task_id in task_ids:
        composed = task_id in nvarc
        parts = None
        if not composed and "_" in task_id:
            try:
                parts = common.split_nvarc_task_id(task_id)
            except ValueError:
                pass
            if parts is not None:
                known = regular | set(indices)
                composed = parts[0] in known and parts[1] in known
        if composed:
            first, second = parts or common.split_nvarc_task_id(task_id)
            tasks[task_id] = (ensure(first), ensure(second))
        else:
            tasks[task_id] = (ensure(task_id), -1)
    return rows, indices, tasks


def legacy_load(model, checkpoint, current_ids):
    state = dict(checkpoint["model_state"])
    rows = model.state_dict()["task_factor_embedding.weight"].clone()
    saved = {str(task_id): i for i, task_id in enumerate(checkpoint["embedding_task_ids"])}
    for i, task_id in enumerate(current_ids):
        if str(task_id) in saved:
            rows[i] = state["task_factor_embedding.weight"][saved[str(task_id)]]
    state["task_factor_embedding.weight"] = rows
    model.load_state_dict(state)


def model_config(rows):
    return TinyTransformerConfig(
        num_examples=rows, num_task_embedding_latents=5,
        d_model=24, n_heads=4, d_ff=48, n_layers=1, n_predict=2,
        dropout=0.0, attention_dropout=0.0, embedding_dropout=0.0,
    )


@pytest.mark.parametrize("whitelist", [None, ["b", "a", "arc2_eval/task", "b"]])
def test_index_matches_old_algorithm_as_base_ids_grow(whitelist):
    task_ids = ["a", "b", "a_b", "arc2_eval/task", "a_b_c", "c", "a_c", "x_y", "x_y"]
    kwargs = dict(regular_task_ids={"a", "b", "arc2_eval/task"},
                  nvarc_task_ids={"x_y"}, embedding_task_whitelist=whitelist)
    assert common.build_task_embedding_index(task_ids, **kwargs) == legacy_index(task_ids, **kwargs)


@pytest.mark.parametrize("saved_ids,current_ids", [
    (["a", "b", "c"], ["a", "b", "c"]),
    (["a", "b", "c"], ["c", "a"]),
    (["a", "b", "c"], ["new", "c", "b", "other"]),
    ([], ["new", "other"]),
    (["a", "b", "c"], ["new", "other"]),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_bulk_factor_loading_is_exactly_equal_to_row_copy(saved_ids, current_ids, dtype):
    source = TinyTransformer(model_config(max(1, len(saved_ids))))
    state = dict(source.state_dict())
    # Also exercise the old assignment's dtype conversion and empty Kaggle table.
    state["task_factor_embedding.weight"] = torch.randn(len(saved_ids), 5, dtype=torch.float64)
    checkpoint = dict(config=asdict(source.config), model_state=state, embedding_task_ids=saved_ids)
    baseline = TinyTransformer(model_config(len(current_ids))).to(dtype=dtype)
    actual = TinyTransformer(model_config(len(current_ids))).to(dtype=dtype)
    actual.load_state_dict(baseline.state_dict())
    initial_rows = actual.task_factor_embedding.weight.detach().clone()
    original_saved_rows = state["task_factor_embedding.weight"].clone()
    legacy_load(baseline, checkpoint, current_ids)
    build._load_model_state(actual, checkpoint, current_ids)
    for key, value in baseline.state_dict().items():
        assert torch.equal(actual.state_dict()[key], value), key
    for i, task_id in enumerate(current_ids):
        if task_id not in saved_ids:
            assert torch.equal(actual.task_factor_embedding.weight[i], initial_rows[i])
    assert torch.equal(state["task_factor_embedding.weight"], original_saved_rows)


def test_matching_ids_load_the_saved_table_directly(monkeypatch):
    source = TinyTransformer(model_config(3))
    checkpoint = dict(config=asdict(source.config), model_state=source.state_dict(),
                      embedding_task_ids=["a", "b", "c"])
    target = TinyTransformer(model_config(3))
    original_load = target.load_state_dict

    def capture(state):
        assert state["task_factor_embedding.weight"] is checkpoint["model_state"]["task_factor_embedding.weight"]
        return original_load(state)

    monkeypatch.setattr(target, "load_state_dict", capture)
    build._load_model_state(target, checkpoint, ["a", "b", "c"])


@pytest.mark.parametrize("mode", ["single_grid_prefix", "full_task_prefix"])
def test_selected_eval_preserves_rows_augments_logits_and_predictions(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(build, "resolve_device", lambda _: torch.device("cpu"))
    monkeypatch.setattr(torch, "compile", lambda function, **kwargs: function)
    monkeypatch.setattr(evaluate, "_compiled_grid_update", evaluate._grid_update)
    tasks = {task_id: {"train": [{"input": [[1, 2]], "output": [[2, 1]]}],
                      "test": [{"input": [[3, 1]], "output": [[1, 3]]}]}
             for task_id in ["second", "first", "third"]}
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(tasks))
    row_ids = ["arc2_train/absent", "a", "b", "first", "second", "third"]
    task_ids = [*row_ids[:3], "a_b", *row_ids[3:]]
    source = TinyTransformer(model_config(len(row_ids)))
    checkpoint = dict(sequence_format=common.SEQUENCE_FORMAT, augmentation_sampling_mode=mode,
                      config=asdict(source.config), model_state=source.state_dict(),
                      task_ids=task_ids, embedding_task_ids=row_ids)
    selected = ["second", "first"]
    args = SimpleNamespace(seed=42, device="cpu", checkpoint_path=None, data_path=path,
                           batch_size=1, enable_aug=True, max_augments=8,
                           extend_checkpoint_task_ids=False)
    # Previous evaluation indexed all saved tasks, including absent compositions.
    with monkeypatch.context() as patch:
        patch.setattr(common, "build_task_embedding_index", legacy_index)
        old_dataset = common.ARCExampleDataset(
            path, splits=["test"], task_whitelist=task_ids, embedding_task_whitelist=row_ids,
            allow_missing_tasks=True, augmentation_sampling_mode=mode, seed=42,
        )
    common.set_seed(42)
    old_model = TinyTransformer(model_config(len(row_ids)))
    legacy_load(old_model, checkpoint, row_ids)
    indexed_tasks = []
    original_index = common.build_task_embedding_index

    def capture(task_ids, **kwargs):
        indexed_tasks.extend(task_ids)
        return original_index(task_ids, **kwargs)

    monkeypatch.setattr(common, "build_task_embedding_index", capture)
    new_model, new_dataset, _, _, _ = build.build_model_and_data(
        args, checkpoint=checkpoint, is_eval=True, eval_task_ids=selected, build_dataloader=False,
    )
    assert indexed_tasks == new_dataset.task_ids == ["first", "second"]
    assert set(new_dataset.task_id_to_embedding_ids) == set(selected)
    assert new_dataset.embedding_task_ids == old_dataset.embedding_task_ids
    assert checkpoint["task_ids"] == task_ids  # Input checkpoint stays intact.
    for key, value in old_model.state_dict().items():
        assert torch.equal(new_model.state_dict()[key], value), key
    old_attempts = evaluate._build_inference_attempts(old_dataset, "test", selected, None, 8, True)
    new_attempts = evaluate._build_inference_attempts(new_dataset, "test", selected, None, 8, True)
    assert len(old_attempts) == len(new_attempts) > 0
    old_model.eval()
    new_model.eval()
    with torch.inference_mode():
        for old, new in zip(old_attempts, new_attempts):
            assert torch.equal(old.positions, new.positions)
            assert {k: v for k, v in vars(old).items() if k != "positions"} == {
                k: v for k, v in vars(new).items() if k != "positions"}
            inputs = dict(input_ids=torch.tensor([new.prompt]),
                          example_ids=torch.tensor([new.task_embedding_ids]),
                          dihedral_ids=torch.tensor([new.dihedral_index]),
                          sep_indices=torch.tensor([len(new.prompt) - 1]),
                          attention_mask=torch.ones(1, len(new.prompt), dtype=torch.bool),
                          positions_3d=new.positions.unsqueeze(0))
            before = old_model(**inputs)["logits"]
            after = new_model(**inputs)["logits"]
            assert torch.equal(before, after)
            assert torch.equal(before.argmax(-1), after.argmax(-1))
    batch = [new_attempts[0], new_attempts[-1]]
    generation = dict(prompts=[item.prompt for item in batch],
                      example_ids=[item.task_embedding_ids for item in batch],
                      dihedral_ids=[item.dihedral_index for item in batch],
                      cached_positions=[item.positions for item in batch],
                      device=torch.device("cpu"), max_new_tokens=8)
    evaluate._prepare_model_for_inference(old_model)
    evaluate._prepare_model_for_inference(new_model)
    assert evaluate.batched_greedy_generate(old_model, **generation) == evaluate.batched_greedy_generate(new_model, **generation)
