"""Block-wise torch.compile for training: same loss and gradients, unchanged module tree."""
from argparse import Namespace
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
import launch_run
import train
from test_looping import loop_model
from test_mtp import _batch


def test_block_compiled_training_matches_eager_and_keeps_state_dict():
    eager = loop_model(loop_iterations=3)
    eager.loop_input_scale.data.fill_(0.3)
    compiled = loop_model(loop_iterations=3)
    compiled.loop_input_scale.data.fill_(0.3)
    keys_before = list(compiled.state_dict())
    returned, options = train.compile_training_model(compiled, Namespace(compile_scope="blocks"), backend="aot_eager")
    assert options["compile_scope"] == "blocks"
    assert returned is compiled and train._unwrap_state_model(returned) is compiled
    assert all(getattr(block, "_compiled_forward", False) for block in compiled.blocks)
    assert list(compiled.state_dict()) == keys_before
    expected = eager(**_batch(True))
    actual = compiled(**_batch(True))
    torch.testing.assert_close(actual["loss"], expected["loss"])
    actual["loss"].backward()
    expected["loss"].backward()
    for (name, parameter), (_, reference) in zip(compiled.named_parameters(), eager.named_parameters()):
        if reference.grad is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, reference.grad, rtol=1e-4, atol=1e-5)
    # A second call does not wrap the blocks again.
    train.compile_training_model(compiled, Namespace(compile_scope="blocks"), backend="aot_eager")
    assert list(compiled.state_dict()) == keys_before


def test_compile_scope_values_and_launcher_default(tmp_path):
    model = loop_model(loop_iterations=2)
    assert train.compile_training_model(model, Namespace(compile_scope="none"))[0] is model
    with pytest.raises(ValueError, match="compile_scope"):
        train.compile_training_model(model, Namespace(compile_scope="layers"))
    plan = launch_run.build_plan({"enable_finetune": False}, tmp_path / "spec.json")
    assert plan["pretrain"]["compile_scope"] == "blocks"
    assert launch_run.stage_summary(plan["pretrain"])["compile_scope"] == "blocks"
    whole = launch_run.build_plan({"enable_finetune": False, "defaults": {"compile_scope": "model"}}, tmp_path / "s.json")
    assert whole["pretrain"]["compile_scope"] == "model"
    with pytest.raises(ValueError, match="compile_scope"):
        launch_run.build_plan({"enable_finetune": False, "defaults": {"compile_scope": "fast"}}, tmp_path / "s.json")
