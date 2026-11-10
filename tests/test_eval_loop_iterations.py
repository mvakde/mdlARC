"""Decode a looped checkpoint at another iteration count (eval_loop_iterations, paper row A7)."""
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import build
import launch_run
from tinytransformer import TinyTransformer, TinyTransformerConfig


def config(**overrides):
    settings = dict(d_model=16, n_heads=2, d_ff=32, n_layers=4, num_examples=2, num_task_embedding_latents=2,
                    loop_iterations=6, loop_start_layer=1, loop_end_layer=3, loop_exit_weight=0.3,
                    dropout=0, embedding_dropout=0, attention_dropout=0)
    return TinyTransformerConfig(**{**settings, **overrides})


def test_unset_or_trained_count_keeps_the_checkpoint_config():
    base = config()
    for value in (None, 6):
        assert build._configure_checkpoint_mtp(base, Namespace(eval_loop_iterations=value), is_eval=True) is base


@pytest.mark.parametrize("iterations", [2, 3, 8])
def test_other_counts_change_only_the_schedule(iterations):
    base = config(loop_exit_iterations=(2, 4))
    updated = build._configure_checkpoint_mtp(base, Namespace(eval_loop_iterations=iterations), is_eval=True)
    assert updated.loop_iterations == iterations and base.loop_iterations == 6
    assert replace(updated, loop_iterations=6, loop_exit_iterations=(2, 4)) == base
    torch.manual_seed(0)
    trained = TinyTransformer(base)
    model = TinyTransformer(updated)
    # The same parameters load; only the logical schedule (and KV cache count) differs.
    model.load_state_dict(trained.state_dict())
    assert model.cache_layer_count == 1 + 2 * iterations + 1
    # Training never applies it.
    assert build._configure_checkpoint_mtp(base, Namespace(eval_loop_iterations=iterations), is_eval=False) == base


def test_rejects_k_dependent_parameters_and_bad_values():
    with pytest.raises(ValueError, match="depend on the iteration count"):
        build._configure_checkpoint_mtp(config(loop_iteration_embedding=True),
                                        Namespace(eval_loop_iterations=3), is_eval=True)
    for value in (1, True, 2.5):
        with pytest.raises(ValueError, match="eval_loop_iterations"):
            build._configure_checkpoint_mtp(config(), Namespace(eval_loop_iterations=value), is_eval=True)
    with pytest.raises(ValueError, match="eval_loop_iterations"):
        build._configure_checkpoint_mtp(config(loop_iterations=1, loop_start_layer=None, loop_end_layer=None,
                                               loop_exit_weight=0.0),
                                        Namespace(eval_loop_iterations=3), is_eval=True)


def test_launcher_accepts_it_on_evals_only(tmp_path):
    plan = launch_run.build_plan({"enable_finetune": False, "evals": {
        "k3": {"checkpoint": "pretrain.final", "eval_loop_iterations": 3}}}, tmp_path / "spec.json")
    assert plan["evals"]["k3"]["eval_loop_iterations"] == 3
    with pytest.raises(ValueError, match="eval_loop_iterations"):
        launch_run.build_plan({"enable_finetune": False, "evals": {
            "k1": {"checkpoint": "pretrain.final", "eval_loop_iterations": 1}}}, tmp_path / "spec.json")
    with pytest.raises(ValueError, match="eval_loop_iterations"):
        launch_run.build_plan({"enable_finetune": False, "defaults": {"eval_loop_iterations": 3}},
                              tmp_path / "spec.json")


def test_evaluation_builds_the_checkpoint_at_the_requested_count(tmp_path, monkeypatch):
    import json
    import train
    pairs = [{"input": [[i]], "output": [[i + 1]]} for i in range(4)]
    (tmp_path / "tasks.json").write_text(json.dumps({"task": {"train": pairs, "test": [{"input": [[5]]}]}}))
    monkeypatch.setattr(build, "resolve_device", lambda _: torch.device("cpu"))
    args = Namespace(seed=42, device="cpu", checkpoint_path=None, data_path=tmp_path / "tasks.json",
                     enable_aug=False, max_augments=1, batch_size=2, d_model=16, n_heads=2, d_ff=32,
                     n_layers=4, num_task_embedding_latents=2, n_predict=1, loop_iterations=6,
                     loop_start_layer=1, loop_end_layer=3, loop_exit_weight=0.3)
    model, dataset, _, _, data_path = build.build_model_and_data(args, build_dataloader=False)
    train.maybe_save_model(model, dataset, data_path, tmp_path / "tiny.pt")
    for iterations, expected in ((None, 6), (3, 3), (8, 8)):
        evaluated, *_ = build.build_model_and_data(
            Namespace(seed=42, device="cpu", checkpoint_path=tmp_path / "tiny.pt", data_path=tmp_path / "tasks.json",
                      eval_loop_iterations=iterations),
            is_eval=True, eval_splits=["test"], build_dataloader=False)
        assert evaluated.config.loop_iterations == expected
        assert evaluated.cache_layer_count == 2 + 2 * expected
        for key, value in model.state_dict().items():
            torch.testing.assert_close(evaluated.state_dict()[key], value)
