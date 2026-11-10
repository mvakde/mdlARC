import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import build
import launch_run
import train


def test_eval_prefix_preserves_trained_embeddings_and_native_submission_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "resolve_device", lambda _: torch.device("cpu"))
    pair = {"input": [[1]], "output": [[2]]}
    original_path = tmp_path / "original.json"
    original_path.write_text(json.dumps({"arc2_eval/abc": {"train": [pair]}}))
    args = SimpleNamespace(
        seed=42, device="cpu", checkpoint_path=None, data_path=original_path,
        batch_size=1, num_workers=0, enable_aug=False, max_augments=1,
        d_model=32, n_heads=4, d_ff=64, n_layers=1, num_task_embedding_latents=4,
        n_predict=1, dropout=0.0, embedding_dropout=0.0, attention_dropout=0.0, FA=2,
    )
    original, dataset, _, _, _ = build.build_model_and_data(args, build_dataloader=False)
    checkpoint_path = tmp_path / "model.pt"
    train.maybe_save_model(original, dataset, original_path, checkpoint_path)
    checkpoint = build.load_checkpoint(checkpoint_path)
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(json.dumps({"abc": {"train": [pair], "test": [{"input": [[3]]}]}}))
    args.data_path = eval_path
    args.checkpoint_task_id_prefix = "arc2_eval/"
    args.extend_checkpoint_task_ids = False
    restored, eval_dataset, _, _, _ = build.build_model_and_data(
        args, checkpoint=checkpoint, is_eval=True, build_dataloader=False,
    )
    assert checkpoint["task_ids"] == ["arc2_eval/abc"]
    assert eval_dataset.task_ids == ["abc"]
    assert eval_dataset.embedding_task_ids == ["abc"]
    assert eval_dataset.get_task_embedding_ids("abc") == (0, -1)
    assert eval_dataset.num_examples == original.config.num_examples
    assert len(eval_dataset.indices_by_split["test"]) == 1
    for name, tensor in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
    with pytest.raises(ValueError, match="requires checkpoint evaluation"):
        build.build_model_and_data(args, checkpoint=checkpoint, build_dataloader=False)


@pytest.mark.parametrize("ids, message", [
    (["arc2_eval/other"], "no trained"),
    (["arc2_eval/abc", "abc"], "duplicate"),
])
def test_eval_prefix_rejects_missing_rows_and_collisions(tmp_path, ids, message):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps({"abc": {"test": [{"input": [[1]]}]}}))
    with pytest.raises(ValueError, match=message):
        build._alias_eval_checkpoint_tasks({"task_ids": ids, "embedding_task_ids": ids}, path, "arc2_eval/")


def test_eval_prefix_is_accepted_only_as_an_evaluation_setting():
    spec = {
        "eval_only": True, "enable_finetune": False, "checkpoint": "model.pt",
        "eval_defaults": {"checkpoint_task_id_prefix": "arc2_eval/"},
    }
    plan = launch_run.build_plan(spec, Path("eval.json"))
    assert plan["evals"]["final"]["checkpoint_task_id_prefix"] == "arc2_eval/"
    spec["eval_defaults"]["checkpoint_task_id_prefix"] = "arc2_eval"
    with pytest.raises(ValueError, match="namespace"):
        launch_run.build_plan(spec, Path("eval.json"))
