#!/usr/bin/env python3
"""Select an unchanged parent generator's broader mode, or a random valid sample."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from v2_storage import save_puzzle, sha, write
from verify import encode, load


POLICIES = {
    "71e489b6": "broad shape families and highest configured richness",
    "7b5033c1": "mixed length, geometry and coloring",
    "4c7dc4dd": "highest configured variation level",
    "409aa875": "all eight arrow directions in both splits",
    "faa9f03d": "mixed operations, varied directions and broad canvas range",
    "7666fa5d": "multicolor boundaries, mixed slopes and free canvas sizes",
    "a6f40cea": "documented maximal regime (seed residue 1 or 2 modulo 14)",
}


def profile(task_id, local):
    if task_id == "71e489b6":
        p = {k: local[k] for k in ("regime", "richness")}
        return p, p == {"regime": 1, "richness": 2}
    if task_id == "7b5033c1":
        p = {k: local["parameters"][k] for k in ("mode", "focused")}
        return p, p["mode"] == "mixed"
    if task_id == "4c7dc4dd":
        p = {"variation": local["style"]["variation"]}
        return p, p["variation"] == 1.0
    if task_id == "409aa875":
        p = {"train_directions": len(local["train_dirs"]), "test_directions": len(local["test_dirs"])}
        return p, p == {"train_directions": 8, "test_directions": 8}
    if task_id == "faa9f03d":
        p = {k: local[k] for k in ("regime", "homogeneous", "size_mode")}
        return p, p["regime"] != 0 and not p["homogeneous"] and p["size_mode"] == 0
    if task_id == "7666fa5d":
        p = {k: local["settings"][k] for k in ("multi", "slope_mode", "size_mode")}
        return p, p == {"multi": True, "slope_mode": "mixed", "size_mode": "free"}
    if task_id == "a6f40cea":
        p = {"mode": local["parameters"]["mode"]}
        return p, p["mode"] == "maximal"
    return {}, False


def generate_observed(module, seed):
    """Observe actual sampled locals; never override RNG, parameters or source code."""
    observed = {}
    code = module.generate_puzzle.__code__

    def trace(frame, event, arg):
        if frame.f_code is code:
            if event == "return":
                observed.update(frame.f_locals)
            return trace
        return None

    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        puzzle = module.generate_puzzle(seed, num_train=4, num_test=1)
    finally:
        sys.settrace(previous)
    return puzzle, observed


def select(generator, task_id, fallback_seed, *, search_limit=64):
    module = load(generator, "parent_generator")
    errors, fallback_reason = [], "No reliable task-specific broader-mode policy"
    if task_id in POLICIES:
        for seed in range(search_limit):
            try:
                raw, local = generate_observed(module, seed)
                details, eligible = profile(task_id, local)
                if not eligible:
                    continue
                puzzle = encode(raw)
                if callable(getattr(module, "validate_generated_puzzle", None)):
                    module.validate_generated_puzzle(raw)
                return puzzle, metadata(generator, seed, "broader_mode", details, POLICIES[task_id], errors)
            except (KeyError, TypeError) as error:
                fallback_reason = f"Policy incompatible with this generator: {error}"
                break
            except Exception as error:
                errors.append({"seed": seed, "error": str(error)})
        else:
            fallback_reason = f"No valid broader-mode sample in seeds 0..{search_limit - 1}"
    rng = random.Random(fallback_seed)
    for seed in rng.sample(range(1000, 10**9), 16):
        try:
            raw = module.generate_puzzle(seed, num_train=4, num_test=1)
            puzzle = encode(raw)
            if callable(getattr(module, "validate_generated_puzzle", None)):
                module.validate_generated_puzzle(raw)
            return puzzle, metadata(generator, seed, "random_fallback", {}, fallback_reason, errors)
        except Exception as error:
            errors.append({"seed": seed, "error": str(error)})
    raise RuntimeError(f"No valid parent sample for {task_id}: {errors}")


def metadata(generator, seed, method, observed, reason, errors):
    return {"generator_sha256": sha(generator), "seed": seed, "num_train": 4, "num_test": 1,
            "method": method, "observed_mode": observed, "reason": reason, "sampling_errors": errors,
            "note": "Known broader mode or random fallback; not a global maximum-diversity guarantee."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--fallback-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    puzzle, selection = select(args.generator, args.task_id, args.fallback_seed)
    selection["input_shapes"] = [[len(p["input"]), len(p["input"][0])]
                                  for split in ("train", "test") for p in puzzle[split]]
    save_puzzle(args.output, puzzle, f"parent {args.task_id}, seed {selection['seed']}")
    write(args.selection, selection)


if __name__ == "__main__":
    main()
