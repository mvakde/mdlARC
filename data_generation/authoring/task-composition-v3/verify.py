#!/usr/bin/env python3
"""Local structure, determinism and generator-owned solver checks; no model judging."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import random

import numpy as np
from v2_storage import read, save_puzzle, write


def load(path, name="composed_generator"):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def grid(value):
    array = np.asarray(value)
    assert array.ndim == 2 and all(1 <= n <= 30 for n in array.shape), "Grid dimensions must be 1..30"
    assert np.issubdtype(array.dtype, np.integer), "Grid cells must be integers"
    assert np.all((0 <= array) & (array <= 9)), "Grid colors must be 0..9"
    return array.astype(int).tolist()


def encode(task, num_train=4, num_test=1):
    assert isinstance(task, dict) and set(task) == {"train", "test"}, "Puzzle must have exactly train/test keys"
    result, seen = {}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        assert isinstance(task[split], list) and len(task[split]) == count, f"Wrong {split} pair count"
        result[split] = []
        for pair in task[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}, "Pair must have exactly input/output keys"
            pair = {k: grid(v) for k, v in pair.items()}
            key = json.dumps(pair["input"], separators=(",", ":"))
            assert key not in seen, "Duplicate input within a puzzle"
            seen.add(key)
            result[split].append(pair)
    return result


def structure(value):
    roles = {}
    return tuple(tuple(roles.setdefault(c, len(roles)) for c in row) for row in value)


def demo_schedule(seeds, generator_bytes):
    """Explicit 40/40/15/5% counts; stable per generator, independent of defaults."""
    weights = {3: 40, 4: 40, 5: 15, 6: 5}
    quota = {n: len(seeds) * percent // 100 for n, percent in weights.items()}
    order = sorted(weights, key=lambda n: (-(len(seeds) * weights[n] % 100), n))
    for n in order[:len(seeds) - sum(quota.values())]:
        quota[n] += 1
    result = [n for n, count in quota.items() for _ in range(count)]
    random.Random(int.from_bytes(hashlib.sha256(generator_bytes).digest(), 'big')).shuffle(result)
    return result


def check(generator_path, seeds, *, parents=None, samples=None):
    generator = load(generator_path)
    for name in ("generate_puzzle", "solve", "validate_generated_puzzle", "self_test"):
        assert callable(getattr(generator, name, None)), f"Missing callable {name}"
    forbidden = set()
    if parents:
        for path in Path(parents).glob("*/*.json"):
            if path.name not in ("task.json", "variation.json"):
                continue
            task = read(path)
            forbidden.update(json.dumps(p, sort_keys=True) for split in ("train", "test") for p in task[split])
    signatures, shapes, structures = set(), set(), set()
    pairs = changes = 0

    def inspect(seed, a, b, *, record=False):
        task = generator.generate_puzzle(seed, num_train=a, num_test=b)
        puzzle = encode(task, a, b)
        generator.validate_generated_puzzle(task)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert json.dumps(pair, sort_keys=True) not in forbidden, f"Exact parent pair copied at seed {seed}"
                predicted = grid(generator.solve(np.array(pair["input"], dtype=int)))
                assert predicted == pair["output"], f"Generator-owned solve disagrees at seed {seed}/{split}"
                if record:
                    for role in ("input", "output"):
                        shapes.add((role, len(pair[role]), len(pair[role][0])))
                        structures.add((role, structure(pair[role])))
        return puzzle

    demo_counts = demo_schedule(seeds, Path(generator_path).read_bytes())
    for seed, demos in zip(seeds, demo_counts, strict=True):
        puzzle = inspect(seed, demos, 1, record=True)
        assert puzzle == inspect(seed, demos, 1), f"Nondeterministic seed {seed}"
        signatures.add(json.dumps(puzzle, sort_keys=True))
        for split in ("train", "test"):
            pairs += len(puzzle[split])
            changes += sum(p["input"] != p["output"] for p in puzzle[split])
        if samples:
            save_puzzle(Path(samples) / f"seed-{seed:010d}.json", puzzle, f"seed {seed}")
    assert len(signatures) == len(seeds), "Different seeds produced identical puzzles"
    assert changes, "Family is entirely identity transformations"
    # Exercise every legal explicit count combination, not just generator defaults.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            inspect(seeds[0], a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4)):
        try:
            generator.generate_puzzle(seeds[0], num_train=a, num_test=b)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Illegal counts accepted: {a}/{b}")
    generator.self_test()
    return {"valid": True, "seeds": list(seeds), "pairs_checked": pairs, "errors": [],
            "sample_counts": [[seed, demos, 1] for seed, demos in zip(seeds, demo_counts, strict=True)],
            "demo_distribution": dict(sorted(Counter(demo_counts).items())),
            "independent_solver": False, "blind_solve": False,
            "semantic_verification": "generator-owned checks only",
            "diversity": {"distinct_puzzles": len(signatures), "grid_shapes": sorted(shapes),
                          "color_normalized_structures": len(structures),
                          "note": "Descriptive measurements; not proof of conceptual diversity or correctness."}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", type=Path, default=Path("generator.py"))
    parser.add_argument("--parents", type=Path, default=Path("parents"))
    parser.add_argument("--report", type=Path, default=Path("scratch/verification.json"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42)))
    parser.add_argument("--samples", type=Path)
    args = parser.parse_args()
    try:
        report = check(args.generator, args.seeds, parents=args.parents, samples=args.samples)
    except Exception as error:
        report = {"valid": False, "seeds": args.seeds, "errors": [f"{type(error).__name__}: {error}"]}
    write(args.report, report)
    print(json.dumps(report, separators=(",", ":")))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
