"""Compare compiled CE forward/backward, including an optional four-head LM path.

Run on the training GPU to establish native throughput; CPU results establish
only local timing. Redirect JSON under results/<branch>/ for ignored evidence.
Compilation/warmup are excluded, and variants alternate within every repeat.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stable_losses import hard_target_cross_entropy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--rows", type=int, nargs="+", default=[4096, 32768])
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--classes", type=int, default=14)
    parser.add_argument("--with-projection", action="store_true")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    if min(*args.rows, args.steps, args.repeats, args.heads, args.width, args.classes) < 1:
        parser.error("Rows, steps, repeats, heads, width, and classes must be positive.")
    torch.manual_seed(20261024)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    synchronize = torch.cuda.synchronize if device.type == "cuda" else lambda: None
    head_count = args.heads if args.with_projection else 1
    report = {"torch": torch.__version__, "device": str(device), "cpu_threads": torch.get_num_threads(),
              "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
              "dtype": args.dtype, "compiled": not args.eager,
              "with_projection": args.with_projection, "heads": head_count, "cases": []}
    for rows in args.rows:
        targets = torch.randint(args.classes, (rows,), device=device)
        targets[::11] = -100
        valid = torch.rand(rows, device=device) > .5
        inputs = torch.randn(rows, args.width if args.with_projection else args.classes,
                             device=device, dtype=torch.float32 if args.with_projection else dtype,
                             requires_grad=True)
        weights = [torch.randn(args.classes, args.width, device=device, requires_grad=True)
                   for _ in range(head_count)] if args.with_projection else []

        def make_objective(loss_fn):
            def objective(x, labels, mask, *heads):
                with torch.autocast(device.type, dtype=dtype, enabled=dtype == torch.bfloat16):
                    result = x.new_zeros((), dtype=torch.float32)
                    for i in range(head_count):
                        logits = F.linear(x, heads[i]) if args.with_projection else x
                        losses = loss_fn(logits.float(), labels, reduction="none")
                        result = result + (losses * mask).sum() / mask.sum().clamp(min=1)
                    return result / head_count
            return objective

        functions = {"standard": make_objective(F.cross_entropy),
                     "stable": make_objective(hard_target_cross_entropy)}
        if not args.eager:
            functions = {key: torch.compile(fn, fullgraph=True) for key, fn in functions.items()}

        def step(fn):
            inputs.grad = None
            for weight in weights:
                weight.grad = None
            value = fn(inputs, targets, valid, *weights)
            value.backward()

        for fn in functions.values():
            for _ in range(5):
                step(fn)
        synchronize()
        samples = {key: [] for key in functions}
        for repeat in range(args.repeats):
            order = list(functions) if repeat % 2 == 0 else list(reversed(functions))
            for key in order:
                synchronize()
                start = time.perf_counter()
                for _ in range(args.steps):
                    step(functions[key])
                synchronize()
                samples[key].append((time.perf_counter() - start) * 1000 / args.steps)
        standard, stable = [statistics.median(samples[key]) for key in ("standard", "stable")]
        report["cases"].append({"rows": rows, "standard_ms": standard, "stable_ms": stable,
                                "stable_over_standard": stable / standard, "samples_ms": samples})
        print(f"rows={rows}: standard={standard:.4f} ms, stable={stable:.4f} ms, "
              f"ratio={stable / standard:.3f}", file=sys.stderr, flush=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
