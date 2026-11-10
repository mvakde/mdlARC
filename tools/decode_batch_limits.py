#!/usr/bin/env python3
"""Offline guess of eval decode batch limits from model shape and GPU memory.

An eval batch holds prompts of one padded length L. Two limits size it:
``batch_size`` caps its rows and ``prefill_token_budget`` caps rows * L. The
decode's peak memory comes from one of two moments:

* the prompt prefill, which holds every layer pass's prompt keys/values while
  it builds the split cache: about PROMPT_KV_FACTOR x the prompt's cache bytes
  plus ACTIVATION_BYTES_PER_WIDTH x (d_model + d_ff) per prompt token; and
* the resident split cache: each row keeps its prompt's keys plus about
  max_new_tokens + ANSWER_SLOT_OVERHEAD answer slots.

Cache bytes per token are layer passes x 2 (keys, values) x d_model x 2 bytes:
what matters is the looped depth times the width, not the parameter count.

Calibrated on 2026-11-08 on an H200 (torch 2.14, NGC 26.09) with d256 x6,
d384 x6, d768 (no loop) and d768 x4 shapes: the formula matches the measured
prompt-pass peaks within ~1% and the full LOO80 decode peaks of d256 x6
(36.4 / 71.5 / 106.6 GiB at 1x / 2x / 3x the reference limits of 300 rows and
524,288 tokens) within ~3%, erring high.
Speed stops improving well before memory runs out (each batch decodes until
its longest answer): on d256 x6 / H200, 2x the reference limits was 14%
faster than 1x and 3x was slower than 2x. The recommendation therefore never
exceeds MAX_DEFAULT_MULTIPLE x the reference limits.

Usage:
  tools/decode_batch_limits.py --spec specs/foo.json --gpu H100
  tools/decode_batch_limits.py --d-model 768 --n-layers 8 --loop-iterations 4 --gpu H200
Prints the recommended eval_defaults entries and their predicted peak. It runs
in well under a second and imports neither torch nor the model.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BATCH_SIZE = 300          # reference limits: the batch cap used by current full-decode specs
DEFAULT_TOKEN_BUDGET = 524_288    # and src/evaluate.py PREFILL_TOKEN_BUDGET
MAX_NEW_TOKENS = 930              # 30 x 30 colours + 29 newlines + END
PROMPT_BUCKET = 512               # src/evaluate.py PROMPT_LENGTH_BUCKET_SIZE
LONGEST_PROMPT = 8192             # covers ARC-1 (6,656) and ARC-2 leave-one-out prompts

PROMPT_KV_FACTOR = 2.2            # prefill peak / prompt cache bytes
ACTIVATION_BYTES_PER_WIDTH = 7.6  # prefill activations per prompt token per unit of d_model + d_ff
ANSWER_SLOT_OVERHEAD = 270        # answer slots beyond max_new_tokens: draft width and 128-slot rounding
FIXED_GIB = 2.0                   # workspace, graph pools, allocator slack (weights added separately)
CONTEXT_GIB = 1.5                 # CUDA context and non-PyTorch memory
SAFETY = 0.9                      # fraction of the remaining memory a batch may use
MAX_DEFAULT_MULTIPLE = 2.0        # no measured speed gain beyond 2x the reference limits

GPU_MEMORY_GIB = {
    "H100": 79.6, "H100-NVL": 93.6, "H200": 139.8, "H200-NVL": 139.8, "B200": 178.0,
    "A100-80": 79.2, "A100-40": 39.4, "L40S": 44.4, "L4": 22.0, "A10G": 22.0,
    "RTX-4090": 23.6, "RTX-6000-ADA": 47.5, "T4": 14.7,
}


@dataclass(frozen=True)
class Shape:
    d_model: int
    d_ff: int
    n_layers: int
    loop_iterations: int = 1
    loop_start_layer: int | None = None
    loop_end_layer: int | None = None
    n_predict: int = 1

    @property
    def layer_passes(self) -> int:
        if self.loop_iterations <= 1 or self.loop_start_layer is None or self.loop_end_layer is None:
            return self.n_layers
        return self.n_layers + (self.loop_iterations - 1) * (self.loop_end_layer - self.loop_start_layer)

    @property
    def cache_bytes_per_token(self) -> int:
        return self.layer_passes * 2 * self.d_model * 2

    @property
    def parameter_bytes(self) -> int:
        # bf16 weights; attention + feed-forward per unique block (embeddings are small).
        return 2 * self.n_layers * (4 * self.d_model ** 2 + 2 * self.d_model * self.d_ff)


def batch_peak_gib(shape: Shape, rows: int, prompt_length: int) -> float:
    """Predicted peak allocated memory (GiB) of one decode batch."""
    tokens = rows * prompt_length
    cache = shape.cache_bytes_per_token
    prefill = tokens * (PROMPT_KV_FACTOR * cache + ACTIVATION_BYTES_PER_WIDTH * (shape.d_model + shape.d_ff))
    resident = rows * cache * (prompt_length + MAX_NEW_TOKENS + ANSWER_SLOT_OVERHEAD)
    return FIXED_GIB + shape.parameter_bytes / 2 ** 30 + max(prefill, resident) / 2 ** 30


def eval_peak_gib(shape: Shape, batch_size: int, token_budget: int, longest_prompt: int = LONGEST_PROMPT) -> float:
    """Worst batch over every padded prompt length the eval can form."""
    worst = 0.0
    for length in range(PROMPT_BUCKET, longest_prompt + 1, PROMPT_BUCKET):
        rows = max(1, min(batch_size, token_budget // length))
        worst = max(worst, batch_peak_gib(shape, rows, length))
    return worst


def recommend(shape: Shape, gpu_gib: float, longest_prompt: int = LONGEST_PROMPT) -> dict:
    """Largest multiple of the reference limits (at most MAX_DEFAULT_MULTIPLE) that fits."""
    usable = (gpu_gib - CONTEXT_GIB) * SAFETY
    low, high = 0.01, MAX_DEFAULT_MULTIPLE
    if eval_peak_gib(shape, *_limits(high), longest_prompt) <= usable:
        low = high
    else:
        for _ in range(40):
            middle = (low + high) / 2
            if eval_peak_gib(shape, *_limits(middle), longest_prompt) <= usable:
                low = middle
            else:
                high = middle
    batch_size, token_budget = _limits(low)
    return {
        "batch_size": batch_size,
        "prefill_token_budget": token_budget,
        "default_multiple": round(low, 2),
        "predicted_peak_gib": round(eval_peak_gib(shape, batch_size, token_budget, longest_prompt), 1),
        "usable_gib": round(usable, 1),
        "fits_at_one_row": eval_peak_gib(shape, 1, PROMPT_BUCKET, longest_prompt) <= usable,
    }


def _limits(multiple: float) -> tuple[int, int]:
    batch_size = max(1, int(DEFAULT_BATCH_SIZE * multiple) // 10 * 10 or 1)
    token_budget = max(PROMPT_BUCKET, int(DEFAULT_TOKEN_BUDGET * multiple) // 32_768 * 32_768)
    return batch_size, token_budget


def shape_from_spec(path: Path) -> tuple[Shape, dict]:
    """The model shape the launcher would train or evaluate, plus the spec's eval limits."""
    sys.path.insert(0, str(ROOT))
    import launch_run  # light: resolves the plan without importing torch

    spec = json.loads(path.read_text())
    plan = launch_run.build_plan(spec, path)
    config = dict(plan["shared"])
    checkpoint = spec.get("checkpoint")
    if spec.get("eval_only") and isinstance(checkpoint, str) and Path(checkpoint).is_file():
        import torch  # only for an eval-only spec whose shape lives in its checkpoint

        config.update(torch.load(checkpoint, map_location="cpu", weights_only=False)["config"])
    evals = next(iter(plan["evals"].values()), {})
    return _shape(config), {"batch_size": evals.get("batch_size"), "prefill_token_budget": evals.get("prefill_token_budget")}


def _shape(config: dict) -> Shape:
    return Shape(d_model=int(config["d_model"]), d_ff=int(config.get("d_ff") or 4 * int(config["d_model"])),
                 n_layers=int(config["n_layers"]), loop_iterations=int(config.get("loop_iterations") or 1),
                 loop_start_layer=config.get("loop_start_layer"), loop_end_layer=config.get("loop_end_layer"),
                 n_predict=int(config.get("n_predict") or 1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--spec", type=Path, help="experiment spec to read the model shape and eval limits from")
    parser.add_argument("--gpu", choices=sorted(GPU_MEMORY_GIB), help="GPU model (sets its memory)")
    parser.add_argument("--memory-gib", type=float, help="GPU memory in GiB instead of --gpu")
    parser.add_argument("--longest-prompt", type=int, default=LONGEST_PROMPT)
    for name in ("d-model", "d-ff", "n-layers", "loop-iterations", "loop-start-layer", "loop-end-layer", "n-predict"):
        parser.add_argument(f"--{name}", type=int, help="override the spec's value")
    args = parser.parse_args(argv)
    if (args.gpu is None) == (args.memory_gib is None):
        parser.error("give exactly one of --gpu or --memory-gib")
    config, current = {}, {}
    if args.spec:
        shape, current = shape_from_spec(args.spec)
        config = shape.__dict__.copy()
    for name in ("d_model", "d_ff", "n_layers", "loop_iterations", "loop_start_layer", "loop_end_layer", "n_predict"):
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    if "d_model" not in config or "n_layers" not in config:
        parser.error("give --spec or at least --d-model and --n-layers")
    shape = _shape(config)
    gpu_gib = args.memory_gib if args.memory_gib is not None else GPU_MEMORY_GIB[args.gpu]
    result = recommend(shape, gpu_gib, args.longest_prompt)
    print(f"model: d_model {shape.d_model}, d_ff {shape.d_ff}, {shape.layer_passes} layer passes, "
          f"{shape.cache_bytes_per_token / 1024:.0f} KiB cache per token")
    print(f"GPU: {gpu_gib:.0f} GiB, {result['usable_gib']} GiB usable for a batch")
    default_peak = eval_peak_gib(shape, DEFAULT_BATCH_SIZE, DEFAULT_TOKEN_BUDGET, args.longest_prompt)
    print(f"reference limits (batch_size {DEFAULT_BATCH_SIZE}, prefill_token_budget {DEFAULT_TOKEN_BUDGET}): "
          f"predicted peak {default_peak:.1f} GiB")
    if current.get("batch_size"):
        budget = current.get("prefill_token_budget") or DEFAULT_TOKEN_BUDGET
        peak = eval_peak_gib(shape, int(current["batch_size"]), int(budget), args.longest_prompt)
        verdict = "fits" if peak <= result["usable_gib"] else "LIKELY OUT OF MEMORY"
        print(f"spec now (batch_size {current['batch_size']}, prefill_token_budget {budget}): "
              f"predicted peak {peak:.1f} GiB, {verdict}")
    if not result["fits_at_one_row"]:
        print("WARNING: even a single-row batch is predicted not to fit on this GPU.")
    print(f"recommended ({result['default_multiple']}x the reference limits, predicted peak "
          f"{result['predicted_peak_gib']} GiB):")
    print(json.dumps({"eval_defaults": {"batch_size": result["batch_size"],
                                        "prefill_token_budget": result["prefill_token_budget"]}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
