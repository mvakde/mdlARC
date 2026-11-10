"""Offline eval batch-limit guesses and the spec-level prefill token budget."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "src")]
import decode_batch_limits as limits
import launch_run

D256X6 = limits.Shape(256, 1024, 8, 6, 2, 6, 4)

# Prompt prefill + split-cache build, random weights, H200, 2026-11-08 (GiB peak allocated).
MEASURED_BATCHES = [
    (D256X6, 6656, 20, 9.04), (D256X6, 6656, 60, 26.97), (D256X6, 512, 150, 6.94), (D256X6, 512, 450, 20.69),
    (limits.Shape(384, 1536, 8, 6, 2, 6, 4), 6656, 60, 40.42), (limits.Shape(384, 1536, 8, 6, 2, 6, 4), 512, 450, 31.00),
    (limits.Shape(768, 3072, 8), 6656, 30, 15.66), (limits.Shape(768, 3072, 8), 512, 300, 12.83),
    (limits.Shape(768, 3072, 8, 4, 2, 6, 4), 6656, 30, 30.80), (limits.Shape(768, 3072, 8, 4, 2, 6, 4), 512, 300, 29.97),
]


@pytest.mark.parametrize("shape,length,rows,measured", MEASURED_BATCHES)
def test_batch_formula_tracks_measured_prefill_peaks(shape, length, rows, measured):
    predicted = limits.batch_peak_gib(shape, rows, length)
    # The fixed term makes small batches err high; never more than 2 GiB low or 25% high.
    assert measured - 1.0 <= predicted <= measured * 1.25 + 0.5


@pytest.mark.parametrize("multiple,measured", [(1, 36.4), (2, 71.5), (3, 106.6)])
def test_whole_eval_peaks_match_full_decodes(multiple, measured):
    predicted = limits.eval_peak_gib(D256X6, 300 * multiple, 524_288 * multiple)
    assert measured <= predicted <= measured * 1.05


def test_cache_bytes_follow_looped_depth_not_parameters():
    assert D256X6.layer_passes == 28 and D256X6.cache_bytes_per_token == 28 * 1024
    assert limits.Shape(768, 3072, 8).cache_bytes_per_token < D256X6.cache_bytes_per_token
    assert limits.Shape(768, 3072, 8, 4, 2, 6).layer_passes == 20


def test_recommendation_is_capped_and_shrinks_with_memory_and_cache():
    h200 = limits.recommend(D256X6, limits.GPU_MEMORY_GIB["H200"])
    assert h200["default_multiple"] == limits.MAX_DEFAULT_MULTIPLE
    assert (h200["batch_size"], h200["prefill_token_budget"]) == (600, 1_048_576)
    h100 = limits.recommend(D256X6, limits.GPU_MEMORY_GIB["H100"])
    l4 = limits.recommend(D256X6, limits.GPU_MEMORY_GIB["L4"])
    assert l4["prefill_token_budget"] < h100["prefill_token_budget"] <= h200["prefill_token_budget"]
    deep = limits.recommend(limits.Shape(768, 3072, 8, 4, 2, 6, 4), limits.GPU_MEMORY_GIB["H100"])
    assert deep["prefill_token_budget"] < h100["prefill_token_budget"]
    for result in (h200, h100, l4, deep):
        assert result["predicted_peak_gib"] <= result["usable_gib"]


def test_spec_shape_and_cli(tmp_path, capsys):
    spec = {"defaults": {"d_model": 384, "n_heads": 6, "d_ff": 1536, "n_layers": 8, "loop_iterations": 6,
                         "loop_start_layer": 2, "loop_end_layer": 6},
            "eval_defaults": {"batch_size": 900, "prefill_token_budget": 2_097_152},
            "evals": {"final": {"from": "pretrain.final"}}}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    shape, current = limits.shape_from_spec(path)
    assert (shape.d_model, shape.layer_passes) == (384, 28)
    assert current == {"batch_size": 900, "prefill_token_budget": 2_097_152}
    assert limits.main(["--spec", str(path), "--gpu", "H100"]) == 0
    out = capsys.readouterr().out
    assert "LIKELY OUT OF MEMORY" in out and '"prefill_token_budget"' in out
    assert limits.main(["--d-model", "256", "--n-layers", "8", "--memory-gib", "80"]) == 0


def test_launcher_accepts_and_validates_the_token_budget():
    plan = launch_run.build_plan({"eval_defaults": {"prefill_token_budget": 1_048_576},
                                  "evals": {"final": {"from": "pretrain.final"}}}, Path("test.json"))
    assert plan["evals"]["final"]["prefill_token_budget"] == 1_048_576
    for invalid in (0, -1, 1.5, True, "1024"):
        with pytest.raises(ValueError, match="prefill_token_budget"):
            launch_run.build_plan({"eval_defaults": {"prefill_token_budget": invalid},
                                   "evals": {"final": {"from": "pretrain.final"}}}, Path("test.json"))
