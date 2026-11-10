"""Coordinate-hole completion followed by a clipped wave inside each tile.

Only NumPy and the standard library are used. No files, parent modules, seeds,
or demonstrations are read by solve(). See task_spec.md for the domain.
"""
from __future__ import annotations

from hashlib import sha256
from itertools import combinations, islice
import json

import numpy as np


HOST_SEEDS = (
    0, 972818356, 233066242, 321925704, 626380171, 801403991,
    273179941, 586290491, 215742180, 299495820, 228698704, 421233744,
    398243120, 674919155, 287715836, 508711738, 794406801, 710875788,
    141793193, 624418640, 139083476, 964177095, 755414622, 819477028,
    812084862, 790032819, 905569386, 481626589, 408027104, 766528600,
    869891737, 766314214, 90517698, 76450686, 865561854, 527425235,
    802520979, 544260319, 17922231, 610388446, 723729164, 436111778,
)


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _integer(value, name, low, high=None):
    _require(isinstance(value, (int, np.integer))
             and not isinstance(value, (bool, np.bool_)),
             f"{name} must be an integer")
    value = int(value)
    _require(value >= low and (high is None or value <= high),
             f"{name} outside its allowed range")
    return value


def _offset(distance):
    return (0, -1, 0, 1)[distance % 4]


def _side(n):
    return n * n + n - 1


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= x <= 30 for x in grid.shape),
             "grid must be 2D with sides in 1..30")
    _require(np.issubdtype(grid.dtype, np.integer), "grid must contain integers")
    _require(np.all((grid >= 0) & (grid <= 9)), "color outside 0..9")
    return grid


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _mask_key(mask):
    return sum(int(value) << index for index, value in enumerate(mask.flat))


def _hole_survives(n, i, j):
    return 0 <= j + _offset(n - 1 - i) < n


def _parse_input(input_grid):
    """Read all conventions from dimensions, separators, and surviving clues."""
    grid = _grid(input_grid)
    orders = [n for n in range(2, 6) if grid.shape == (_side(n), _side(n))]
    _require(len(orders) == 1, "expected an unpadded square lattice of order 2..5")
    n = orders[0]
    # A corner can be foreground or a hole; the first separator is reliable.
    bg = int(grid[n, 0])
    _require(np.all(grid[n::n + 1, :] == bg)
             and np.all(grid[:, n::n + 1] == bg), "damaged separators")
    colors = [int(c) for c in np.unique(grid) if c != bg]
    _require(len(colors) == 1, "need one visible foreground color")
    fg = colors[0]
    visible = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            tile = grid[i * (n + 1):i * (n + 1) + n,
                        j * (n + 1):j * (n + 1) + n]
            if np.all(tile == bg):
                continue
            _require(tile[i, j] == bg and np.count_nonzero(tile == fg) == n * n - 1,
                     "clue must have exactly its coordinate-indexed hole")
            visible[i, j] = True
    _require(0 < int(visible.sum()) < n * n,
             "need a visible clue and a completely erased tile")
    return grid, n, bg, fg, visible


def _complete(n, bg, fg, hole_mode="own"):
    """B's intermediate mask. Modes other than 'own' are test ablations."""
    grid = np.full((_side(n), _side(n)), bg, dtype=np.int8)
    for i in range(n):
        for j in range(n):
            r0, c0 = i * (n + 1), j * (n + 1)
            grid[r0:r0 + n, c0:c0 + n] = fg
            if hole_mode == "own":
                hole = i, j
            elif hole_mode == "shared":
                hole = 0, 0
            elif hole_mode == "transpose":
                hole = j, i
            elif hole_mode == "solid":
                continue
            else:
                raise ValueError("unknown hole ablation")
            grid[r0 + hole[0], c0 + hole[1]] = bg
    return grid


def _wave_tiles(grid, n, bg, anchor="bottom", phase=0, wrap=False):
    """A's row translation, scoped to tiles. Nondefaults are test ablations."""
    output = np.full_like(grid, bg)
    for i in range(n):
        for j in range(n):
            r0, c0 = i * (n + 1), j * (n + 1)
            for r in range(n):
                if anchor == "bottom":
                    distance = n - 1 - r
                elif anchor == "top":
                    distance = r
                elif anchor == "canvas":
                    distance = grid.shape[0] - 1 - (r0 + r)
                else:
                    raise ValueError("unknown wave anchor")
                shift = _offset(distance + phase)
                row = grid[r0 + r, c0:c0 + n]
                if wrap:
                    output[r0 + r, c0:c0 + n] = np.roll(row, shift)
                elif shift == -1:
                    output[r0 + r, c0:c0 + n - 1] = row[1:]
                elif shift == 1:
                    output[r0 + r, c0 + 1:c0 + n] = row[:-1]
                else:
                    output[r0 + r, c0:c0 + n] = row
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only inverse rasterization of the completed and deformed tiles."""
    grid, n, bg, fg, _ = _parse_input(input_grid)
    output = np.full(grid.shape, bg, dtype=np.int8)
    for i in range(n):
        for j in range(n):
            r0, c0 = i * (n + 1), j * (n + 1)
            for r in range(n):
                d = n - 1 - r
                shift = 0 if d % 2 == 0 else (-1 if d % 4 == 1 else 1)
                for c in range(n):
                    source_c = c - shift
                    if 0 <= source_c < n and not (r == i and source_c == j):
                        output[r0 + r, c0 + c] = fg
    return output


def _anchor_coordinates(rng, n):
    rows = [int(x) for x in rng.choice(n, 3, replace=False)]
    cols = [int(x) for x in rng.choice(n, 3, replace=False)]
    if all(i == j for i, j in zip(rows, cols)):
        cols = cols[1:] + cols[:1]
    return set(zip(rows, cols))


def _mask_is_useful(mask, required):
    n = len(mask)
    return (all(mask[i, j] for i, j in required)
            and 0 < int(mask.sum()) < n * n
            and any(not mask[i, j] and _hole_survives(n, i, j)
                    for i in range(n) for j in range(n)))


def _choose_mask(rng, n, used, anchor=False, density="mixed"):
    required = _anchor_coordinates(rng, n) if anchor else set()
    area = n * n
    low, high = max(1, len(required)), area - 1
    if density == "dense":
        low = max(low, (area + 1) // 2)
    elif density == "sparse":
        high = area // 3
    for _ in range(40):
        if density == "mixed":
            band = int(rng.integers(4))
            a, b = ((low, min(high, max(low, n))),
                    (max(low, n), max(low, min(high, area * 2 // 3))),
                    (max(low, area - 2), high),
                    (max(low, min(high, 7)), max(low, min(high, 7))))[band]
            count = int(rng.integers(a, b + 1))
        else:
            count = int(rng.integers(low, high + 1))
        style = int(rng.integers(6))
        coordinates = [(i, j) for i in range(n) for j in range(n)]
        noise = rng.random((n, n))
        if style == 1:
            direction = int(rng.integers(4))
            priority = lambda i, j: (i, n - 1 - i, j, n - 1 - j)[direction] + noise[i, j] / 3
        elif style in (2, 3):
            center = (int(rng.integers(n)), int(rng.integers(n)))
            if style == 2:
                center = (int(rng.choice((0, n - 1))), int(rng.choice((0, n - 1))))
            priority = lambda i, j: abs(i - center[0]) + abs(j - center[1]) + noise[i, j] / 3
        else:
            priority = lambda i, j: noise[i, j]
        if style in (4, 5):
            axis = style - 4
            candidates = [k for k in range(n) if all(p[axis] != k for p in required)]
            if candidates and n * (n - 1) >= low:
                gap = int(rng.choice(candidates))
                coordinates = [p for p in coordinates if p[axis] != gap]
                count = min(count, len(coordinates))
        coordinates.sort(key=lambda p: priority(*p))
        chosen = required | set([p for p in coordinates if p not in required][:count - len(required)])
        mask = np.zeros((n, n), dtype=bool)
        for i, j in chosen:
            mask[i, j] = True
        if _mask_is_useful(mask, required) and _mask_key(mask) not in used:
            used.add(_mask_key(mask))
            return mask
    # At most 13 masks are already used. Every candidate includes the anchors.
    # Only at most two nearly-full masks can hide no surviving hole. Therefore
    # a bounded prefix of 32 combinations per count suffices; no 2**25 scan.
    free = [p for p in ((i, j) for i in range(n) for j in range(n)) if p not in required]
    for count in range(low, high + 1):
        for extra in islice(combinations(free, count - len(required)), 32):
            mask = np.zeros((n, n), dtype=bool)
            for i, j in required | set(extra):
                mask[i, j] = True
            if _mask_is_useful(mask, required) and _mask_key(mask) not in used:
                used.add(_mask_key(mask))
                return mask
    raise RuntimeError("no unused clue layout within the bounded search")


def _shape_plan(rng, num_train, num_test):
    mode = int(rng.integers(3))
    if mode == 0:
        train, pool = [3, 4], (2, 3, 4)
        test = [5] * num_test
    elif mode == 1:
        train, pool = [4, 5], (4, 5)
        test = [int(rng.choice((2, 3))) for _ in range(num_test)]
    else:
        odd = int(rng.choice((3, 5)))
        train, pool = [4, odd], (4, odd)
        test = [int(rng.choice(pool)) for _ in range(num_test)]
    train.extend(int(rng.choice(pool)) for _ in range(num_train - 2))
    return mode, train, test


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    _require(num_train > num_test, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    mode, train_orders, test_orders = _shape_plan(rng, num_train, num_test)
    shared_bg = (int(rng.choice(10, p=(0.46,) + (0.06,) * 9))
                 if rng.random() < 0.5 else None)
    used = {n: set() for n in range(2, 6)}
    previous_fg = None
    puzzle = {"train": [], "test": []}
    for split, orders in (("train", train_orders), ("test", test_orders)):
        for index, n in enumerate(orders):
            bg = shared_bg
            if bg is None:
                bg = int(rng.choice(10, p=(0.46,) + (0.06,) * 9))
            choices = [c for c in range(10) if c != bg]
            if split == "train" and index == 1:
                fg = int(rng.choice([c for c in choices if c != previous_fg]))
            elif previous_fg in choices and rng.random() < 0.35:
                fg = previous_fg
            else:
                fg = int(rng.choice(choices))
            previous_fg = fg
            density = ("dense" if split == "train" else "sparse") if mode == 2 else "mixed"
            mask = _choose_mask(rng, n, used[n], split == "train" and index < 2, density)
            completed = _complete(n, bg, fg)
            inp = np.full_like(completed, bg)
            for i, j in np.argwhere(mask):
                r0, c0 = int(i) * (n + 1), int(j) * (n + 1)
                inp[r0:r0 + n, c0:c0 + n] = completed[r0:r0 + n, c0:c0 + n]
            out = _wave_tiles(completed, n, bg)
            puzzle[split].append({"input": inp, "output": out})
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    return puzzle


def _inspect_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
             "pair needs exactly input/output")
    inp, n, bg, fg, visible = _parse_input(pair["input"])
    out = _grid(pair["output"])
    _require(inp.shape == out.shape, "input/output sizes differ")
    _require(set(int(x) for x in np.unique(out)) == {bg, fg}, "output colors changed")
    _require(np.all(out[n::n + 1, :] == bg) and np.all(out[:, n::n + 1] == bg),
             "output overwrote a separator")
    for i in range(n):
        for j in range(n):
            for r in range(n):
                shift = _offset(n - 1 - r)
                row = out[i * (n + 1) + r, j * (n + 1):j * (n + 1) + n]
                expected = np.full(n, fg, dtype=np.int8)
                if shift == -1:
                    expected[-1] = bg
                elif shift == 1:
                    expected[0] = bg
                if r == i and 0 <= j + shift < n:
                    expected[j + shift] = bg
                _require(np.array_equal(row, expected), "wrong tile row, hole, or clipping")
                expected_count = n - abs(shift) - int(r == i and 0 <= j + shift < n)
                _require(np.count_nonzero(row == fg) == expected_count, "wrong row population")
    _require(_mask_is_useful(visible, set()), "no erased tile with a surviving coordinate hole")
    completed = _complete(n, bg, fg)
    _require(not np.array_equal(out, completed), "omitting A does not change the answer")
    _require(not np.array_equal(out, _wave_tiles(inp, n, bg)),
             "omitting B's reconstruction does not change the answer")
    # A more specific B ablation keeps every clue, but completes blank tiles solid.
    solid_blanks = _complete(n, bg, fg, "solid")
    for i, j in np.argwhere(visible):
        solid_blanks[int(i) * (n + 1) + int(i), int(j) * (n + 1) + int(j)] = bg
    _require(not np.array_equal(out, _wave_tiles(solid_blanks, n, bg)),
             "coordinate holes in newly filled tiles are not exercised")
    _require(np.array_equal(solve(inp), out), "solve disagrees")
    return {"n": n, "bg": bg, "fg": fg, "visible": visible,
            "count": int(visible.sum()), "density": float(visible.mean())}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle needs exactly train/test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "invalid pair counts")
    seen, structures = set(), set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _inspect_pair(pair)
            key = _key(np.asarray(pair["input"]))
            structure = info["n"], _mask_key(info["visible"])
            _require(key not in seen and structure not in structures,
                     "duplicate input or color-normalized clue layout")
            seen.add(key)
            structures.add(structure)
            infos[split].append(info)
    train = infos["train"]
    orders = {info["n"] for info in train}
    _require(4 in orders and bool(orders & {3, 5}), "training must show full cycle and contrasting tile parity")
    for acceptable in ({4}, {3, 5}):
        _require(any(info["n"] in acceptable
                     and np.count_nonzero(info["visible"].any(axis=0)) >= 3
                     and np.count_nonzero(info["visible"].any(axis=1)) >= 3
                     and np.any(info["visible"] & ~np.eye(info["n"], dtype=bool))
                     for info in train), "training lacks coordinate witnesses")
    _require(len({info["fg"] for info in train}) >= 2, "training lacks foreground role variation")
    # Reject all four fixed phases of a top-anchored alternative, not just phase 0.
    for phase in range(4):
        _require(any(_offset(n - 1 - r) != _offset(r + phase)
                     for n in orders for r in range(n)), "top anchor fits the demonstrations")
        _require(any(_offset(n - 1 - r) != _offset(_side(n) - 1 - (i * (n + 1) + r) + phase)
                     for n in orders for i in range(n) for r in range(n)),
                 "canvas anchor fits the demonstrations")
    for info in infos["test"]:
        n = info["n"]
        _require(n > max(orders) or n < min(orders)
                 or (n in orders and info["density"] < min(x["density"] for x in train)),
                 "test has no structural scale or erasure shift")


def _puzzle_digest(puzzle):
    digest = sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeds, finite edge cases and all legal counts; no disk writes."""
    signatures, layouts, palettes, shapes = set(), set(), set(), set()
    counts = {"train": set(), "test": set()}
    events = {name: 0 for name in ("larger_test", "smaller_test", "sparser_test",
              "train_axis_gap", "test_axis_gap", "train_sparse", "test_sparse",
              "train_near_full", "test_near_full", "zero_foreground", "nonzero_background")}
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = _puzzle_digest(puzzle)
        _require(signature == _puzzle_digest(generate_puzzle(seed)), "nondeterminism")
        signatures.add(signature)
        train_orders = {_parse_input(p["input"])[1] for p in puzzle["train"]}
        train_density = min(_parse_input(p["input"])[4].mean() for p in puzzle["train"])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, n, bg, fg, mask = _parse_input(pair["input"])
                layouts.add((n, _mask_key(mask)))
                palettes.add((bg, fg))
                shapes.add(inp.shape)
                count = int(mask.sum())
                counts[split].add(count)
                gap = not mask.any(axis=0).all() or not mask.any(axis=1).all()
                events[split + "_axis_gap"] += int(gap)
                events[split + "_sparse"] += int(count * 3 <= n * n)
                events[split + "_near_full"] += int(n * n - count <= 2)
                events["zero_foreground"] += int(fg == 0)
                events["nonzero_background"] += int(bg != 0)
                if split == "test":
                    events["larger_test"] += int(n > max(train_orders))
                    events["smaller_test"] += int(n < min(train_orders))
                    events["sparser_test"] += int(n in train_orders and mask.mean() < train_density)
    _require(len(signatures) == 42 and len(layouts) >= 100, "insufficient structural diversity")
    _require(shapes == {(_side(n), _side(n)) for n in range(2, 6)}, "missing lattice orders")
    _require(len(palettes) >= 25 and all(events.values()), "missing variation axis")
    _require(all(len(x) >= 8 for x in counts.values()), "insufficient clue-count variation")
    legal_calls = 0
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(31, a, b)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (a, b), "counts ignored")
            validate_generated_puzzle(puzzle)
            legal_calls += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # All single-clue and single-erasure layouts, including a hidden hole that
    # clips away. solve supports these even when the curriculum does not emit them.
    edge_cases = 0
    for n in range(2, 6):
        completed = _complete(n, 7, 0)
        expected = _wave_tiles(completed, n, 7)
        for i in range(n):
            for j in range(n):
                r0, c0 = i * (n + 1), j * (n + 1)
                for just_one in (True, False):
                    inp = np.full_like(completed, 7) if just_one else completed.copy()
                    inp[r0:r0 + n, c0:c0 + n] = completed[r0:r0 + n, c0:c0 + n] if just_one else 7
                    before = inp.copy()
                    _require(np.array_equal(solve(inp), expected), "edge layout failed")
                    _require(np.array_equal(inp, before), "solve mutated its input")
                    edge_cases += 1
        for ablation in ("shared", "transpose", "solid"):
            _require(not np.array_equal(expected, _wave_tiles(_complete(n, 7, 0, ablation), n, 7)),
                     "coordinate ablation did not matter")
        _require(not np.array_equal(expected, _wave_tiles(completed, n, 7, wrap=True)),
                 "wraparound indistinguishable")
    return {"valid": True, "seeds": 42, "pairs": 210, "distinct_puzzles": len(signatures),
            "distinct_clue_layouts": len(layouts), "palettes": len(palettes),
            "shapes": sorted(shapes), "visible_counts": {k: sorted(v) for k, v in counts.items()},
            "events": events, "legal_count_calls": legal_calls, "edge_layout_checks": edge_cases,
            "semantic_verification": "generator-owned invariants and ablations only"}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
