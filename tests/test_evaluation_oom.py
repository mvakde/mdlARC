"""OOM recovery preserves inference attempts and rejects unrelated failures."""
from pathlib import Path
import random
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import _generate_with_oom_splitting


def test_oom_splitting_preserves_order_and_rng_after_failed_attempts():
    torch.manual_seed(42)
    random.seed(42)
    expected = [(i, float(torch.rand(())), random.random()) for i in range(5)]
    torch.manual_seed(42)
    random.seed(42)

    def generate(items):
        if len(items) > 2:
            torch.rand(7)
            random.random()
            raise torch.cuda.OutOfMemoryError("simulated capacity limit")
        return [(i, float(torch.rand(())), random.random()) for i in items]

    assert _generate_with_oom_splitting(list(range(5)), generate) == expected


def test_oom_for_one_attempt_is_reported():
    def generate(items):
        raise torch.cuda.OutOfMemoryError("single attempt cannot fit")
    with pytest.raises(torch.cuda.OutOfMemoryError, match="single attempt"):
        _generate_with_oom_splitting([1], generate)


def test_nonmemory_failure_is_not_retried():
    calls = []
    def generate(items):
        calls.append(items)
        raise RuntimeError("invalid model input")
    with pytest.raises(RuntimeError, match="invalid model"):
        _generate_with_oom_splitting([1, 2], generate)
    assert calls == [[1, 2]]
