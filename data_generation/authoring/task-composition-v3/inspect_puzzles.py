"""Convenience tools; agents may also write their own task-specific analyses."""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
from pathlib import Path


def jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def sample(generator, seed, num_train=None, num_test=None):
    spec = importlib.util.spec_from_file_location("arc_review_sample", generator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return jsonable(module.generate_puzzle(seed, num_train=num_train, num_test=num_test))


def format_grid(grid):
    """One ARC color digit per cell, preserving rows and leading zeroes."""
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("Grid must be a nonempty rectangle")
    if any(type(cell) is not int or not 0 <= cell <= 9 for row in grid for cell in row):
        raise ValueError("Grid colors must be integers from 0 to 9")
    return "\n".join("".join(str(cell) for cell in row) for row in grid)


def format_puzzle(puzzle, *, label=None, split=None, pair=None):
    """Readable grids for the reviewer; JSON remains the programmatic format."""
    if split not in (None, "train", "test"):
        raise ValueError("Split must be train or test")
    if pair is not None and (split is None or pair < 1):
        raise ValueError("A one-based --pair requires --split")
    puzzle = jsonable(puzzle)
    sections = [f"Puzzle: {label}"] if label else []
    for part in ((split,) if split else ("train", "test")):
        pairs = puzzle[part]
        if pair is not None and pair > len(pairs):
            raise ValueError(f"No {part} pair {pair}")
        for index, grids in enumerate(pairs, 1):
            if pair is not None and index != pair:
                continue
            lines = [f"{part.title()} pair {index}"]
            for role in ("input", "output"):
                if role in grids:
                    lines.extend([f"{role.title()}:", format_grid(grids[role])])
            sections.append("\n".join(lines))
    return "\n\n".join(sections) + "\n"


def describe(paths):
    shapes, palettes, counts = collections.Counter(), collections.Counter(), collections.Counter()
    structures, records = set(), []
    for path in paths:
        puzzle = json.loads(path.read_text())
        counts[(len(puzzle["train"]), len(puzzle["test"]))] += 1
        pairs = []
        for split in ("train", "test"):
            for index, pair in enumerate(puzzle[split], 1):
                info = {"split": split, "pair": index}
                for role, grid in pair.items():
                    shape = (len(grid), len(grid[0]))
                    palette = tuple(sorted({c for row in grid for c in row}))
                    shapes[(role, *shape)] += 1
                    palettes[(role, *palette)] += 1
                    mapping = {}
                    norm = [[mapping.setdefault(c, len(mapping)) for c in row] for row in grid]
                    structures.add(hashlib.sha256(json.dumps(norm).encode()).hexdigest())
                    info[role] = {"shape": shape, "palette": palette}
                pairs.append(info)
        records.append({"file": str(path), "pairs": pairs})
    def rows(counter):
        return [{"value": key, "count": value} for key, value in sorted(counter.items())]
    return {"sample_count": len(records), "pair_counts": rows(counts), "shapes": rows(shapes),
            "palettes": rows(palettes), "recolor_normalized_grid_count": len(structures),
            "samples": records, "note": "Descriptive statistics only; neither correctness nor conceptual diversity is established."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("sample")
    p.add_argument("generator", type=Path)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--num-train", type=int)
    p.add_argument("--num-test", type=int)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("show", help="Show selected puzzles as compact digit grids")
    p.add_argument("input", type=Path, nargs="+")
    p.add_argument("--split", choices=("train", "test"))
    p.add_argument("--pair", type=int, help="One-based pair number; requires --split")
    p.add_argument("--output", type=Path, help="Write text to a file instead of stdout")
    p = commands.add_parser("stats")
    p.add_argument("directory", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "show":
        try:
            result = "\n".join(format_puzzle(json.loads(path.read_text()), label=str(path),
                                           split=args.split, pair=args.pair) for path in args.input)
        except ValueError as exc:
            parser.error(str(exc))
        if args.output is None:
            print(result, end="")
            return
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
        print(args.output.resolve())
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "sample":
        task = sample(args.generator.resolve(), args.seed, args.num_train, args.num_test)
        args.output.write_text(json.dumps(task, separators=(",", ":")) + "\n")
    elif args.command == "stats":
        args.output.write_text(json.dumps(describe(sorted(args.directory.glob("seed-*.json"))), separators=(",", ":")) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
