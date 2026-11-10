"""Resuming a DDP checkpoint under another world size (MDLARC_RESUME_RESEED_RANKS)."""
from argparse import Namespace
import json
from pathlib import Path
import random
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import build
import train
from common import capture_rng_state


def _checkpoint(tmp_path, monkeypatch):
    pairs = [{"input": [[i]], "output": [[i + 1]]} for i in range(4)]
    (tmp_path / "tasks.json").write_text(json.dumps({"task": {"train": pairs}}))
    monkeypatch.setattr(build, "resolve_device", lambda _: torch.device("cpu"))
    args = Namespace(seed=42, device="cpu", checkpoint_path=None, data_path=tmp_path / "tasks.json",
                     enable_aug=False, max_augments=1, batch_size=2, d_model=16, n_heads=2, d_ff=32,
                     n_layers=1, num_task_embedding_latents=2, n_predict=1)
    model, dataset, _, _, data_path = build.build_model_and_data(args, build_dataloader=False)
    state = capture_rng_state(torch.device("cpu"))
    train.maybe_save_model(model, dataset, data_path, tmp_path / "tiny.pt", epoch=3, global_step=6,
                           rng_state=state, rng_state_by_rank=[state, state])
    return Namespace(**{**vars(args), "checkpoint_path": tmp_path / "tiny.pt", "checkpoint_mode": "resume"})


def test_world_size_change_requires_the_opt_in(tmp_path, monkeypatch):
    args = _checkpoint(tmp_path, monkeypatch)
    monkeypatch.delenv("MDLARC_RESUME_RESEED_RANKS", raising=False)
    with pytest.raises(ValueError, match="world size"):
        build.build_model_and_data(args, build_dataloader=False)


def test_opt_in_reseeds_rank_streams_from_the_epoch(tmp_path, monkeypatch):
    args = _checkpoint(tmp_path, monkeypatch)
    monkeypatch.setenv("MDLARC_RESUME_RESEED_RANKS", "1")
    build.build_model_and_data(args, build_dataloader=False)
    seed = 1_000_003 * (3 + 1) + 0  # epoch 3, rank 0
    observed = (random.random(), torch.rand(1).item())
    random.seed(seed); torch.manual_seed(seed)
    assert observed == (random.random(), torch.rand(1).item())
