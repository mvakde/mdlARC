"""Shared-gap rod packing: input-only Boolean geometry and greedy allocation."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.tobytes()


def _counts(num_train, num_test):
    for value, name, lo, hi in ((num_train, "num_train", 2, 10),
                                (num_test, "num_test", 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _parse(value):
    """All roles come from this input, never from a puzzle or seed."""
    a = _grid(value)
    h, total = a.shape
    if total % 2 or not 5 <= total // 2 <= 15:
        raise ValueError("the two screens must have width 5..15")
    w = total // 2
    bg = int(a[-1, 0])
    if not np.all(a[-1] == bg):
        raise ValueError("last row must be blank")
    uniform = np.all(a == a[:, :1], axis=1) & (a[:, 0] != bg)
    rail_rows = np.flatnonzero(uniform)
    if len(rail_rows) != 1:
        raise ValueError("expected one uniform non-background rail")
    r = int(rail_rows[0])
    rail = int(a[r, 0])
    if not 3 <= r <= 20 or r >= h - 2:
        raise ValueError("invalid board/rack height")
    left, right = a[:r, :w], a[:r, w:]
    lc = set(int(c) for c in np.unique(left)) - {bg}
    rc = set(int(c) for c in np.unique(right)) - {bg}
    if len(lc) != 1 or len(rc) != 1 or lc & rc or rail in lc | rc:
        raise ValueError("each screen must have its own obstacle color")
    structural = {bg, rail} | lc | rc
    rack = a[r + 1:]
    colors = sorted(set(int(c) for c in np.unique(rack)) - {bg})
    if not 3 <= len(colors) <= 6 or structural & set(colors):
        raise ValueError("rack must contain 3..6 distinct rod colors")
    rods = []  # (length, color, original row, original start)
    for color in colors:
        yy, xx = np.nonzero(rack == color)
        length = len(xx)
        if (not np.all(yy == yy[0]) or not 1 <= length < w
                or not np.array_equal(xx, np.arange(xx[0], xx[0] + length))):
            raise ValueError("each color must be one rigid horizontal rod")
        rods.append((length, color, int(yy[0]) + r + 1, int(xx[0])))
    if len({rod[0] for rod in rods}) != len(rods):
        raise ValueError("rod lengths must be distinct")
    for row in {rod[2] for rod in rods}:
        group = sorted((x, length) for length, _, y, x in rods if y == row)
        if len(group) > 2 or (len(group) == 2
                and group[0][0] + group[0][1] >= group[1][0]):
            raise ValueError("rack rods require a blank separating cell")
    rods.sort(key=lambda rod: (rod[2], rod[3]))
    free = (left == bg) & (right == bg)
    states = (left != bg).astype(np.int8) + 2 * (right != bg).astype(np.int8)
    return dict(a=a, r=r, w=w, bg=bg, rail=rail, rods=rods,
                left=left, right=right, free=free, states=states)


def _pack(free, rods, order="decreasing"):
    """Return (length, color, row, start) placements, or fail if no fit exists."""
    available = free.copy()
    if order == "decreasing":
        rods = sorted(rods, key=lambda rod: -rod[0])
    elif order == "increasing":
        rods = sorted(rods, key=lambda rod: rod[0])
    elif order != "source":
        raise ValueError("unknown priority")
    placed = []
    h, w = available.shape
    for length, color, _, _ in rods:
        destination = None
        for row in range(h - 1, -1, -1):
            for start in range(w - length, -1, -1):
                if np.all(available[row, start:start + length]):
                    destination = row, start
                    break
            if destination is not None:
                break
        if destination is None:
            raise ValueError("a rod cannot be allocated under the packing rule")
        row, start = destination
        available[row, start:start + length] = False
        placed.append((length, color, row, start))
    return placed


def _positions(placements):
    return {color: (row, start) for _, color, row, start in placements}


def _render(free, placements, bg, rail):
    r, w = free.shape
    out = np.full((r + 1, w), bg, dtype=np.int8)
    out[:r][~free] = rail
    out[-1] = rail
    for length, color, row, start in placements:
        out[row, start:start + length] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _parse(input_grid)
    placements = _pack(info["free"], info["rods"])
    return _render(info["free"], placements, info["bg"], info["rail"])


def _free_field(rng, r, w, style, lengths):
    """Mix row cavities and 2D fields; add a feasible set of supported spans."""
    free = np.zeros((r, w), dtype=bool)
    if style == 0:  # isolated shelves, variable endpoints
        for y in range(r):
            if rng.random() < .83:
                length = int(rng.integers(1, w + 1))
                x = int(rng.integers(w - length + 1))
                free[y, x:x + length] = True
    elif style == 1:  # wandering corridor
        x = int(rng.integers(0, max(1, w // 3)))
        end = int(rng.integers(max(x + 1, w // 2), w + 1))
        for y in range(r):
            x = int(np.clip(x + rng.integers(-2, 3), 0, w - 2))
            end = int(np.clip(end + rng.integers(-2, 3), x + 1, w))
            if rng.random() < .9:
                free[y, x:end] = True
    elif style == 2:  # separated galleries, competing endpoints
        for y in range(r):
            cut = int(rng.integers(2, w - 1))
            x = int(rng.integers(0, cut))
            end = int(rng.integers(cut + 1, w + 1))
            free[y, x:cut] = True
            free[y, cut + 1:end] = True
    elif style == 3:  # overlapping chambers
        for _ in range(int(rng.integers(3, 8))):
            y0 = int(rng.integers(r))
            y1 = int(rng.integers(y0 + 1, r + 1))
            x0 = int(rng.integers(w))
            x1 = int(rng.integers(x0 + 1, w + 1))
            free[y0:y1, x0:x1] = True
    else:  # irregular small vacancies plus horizontal corridors
        free = rng.random((r, w)) < float(rng.uniform(.16, .57))
        for y in rng.choice(r, size=max(2, r // 2), replace=False):
            x = int(rng.integers(w - 1))
            end = int(rng.integers(x + 1, w + 1))
            free[y, x:end] = True
    # One span per distinct row gives a feasible matching. Greedy placements
    # can also exploit other spans or place several rods on the same row.
    for y, length in zip(rng.choice(r, size=len(lengths), replace=False), lengths):
        x = int(rng.integers(w - length + 1))
        free[y, x:x + length] = True
    return free


def _screens(rng, free):
    r, w = free.shape
    style = int(rng.integers(4))
    yy, xx = np.indices((r, w))
    if style == 0:
        states = rng.choice([1, 2, 3], size=(r, w), p=[.36, .36, .28])
    elif style == 1:
        states = (xx + yy // 3 + int(rng.integers(3))) % 3 + 1
    elif style == 2:
        coarse = rng.integers(1, 4, size=(3, 4))
        states = coarse[yy * 3 // r, xx * 4 // w]
    else:
        states = (yy + xx // 3 + int(rng.integers(3))) % 3 + 1
    states[free] = 0
    blocked = np.flatnonzero(~free)
    if len(blocked) < 6:
        raise ValueError("too few obstructions for diagnostic masks")
    # Both exclusive states and the overlap are visibly present.
    for index, state in zip(rng.choice(blocked, size=3, replace=False), (1, 2, 3)):
        states.flat[index] = state
    return states.astype(np.int8)


def _rack(rng, w, lengths, colors, shared):
    order = [int(i) for i in rng.permutation(len(lengths))]
    groups = [order[:2]] + [[i] for i in order[2:]] if shared else [[i] for i in order]
    height = len(groups) + int(rng.integers(1, 5))
    rows = sorted(int(y) for y in rng.choice(height - 1, size=len(groups), replace=False))
    records = []
    for row, group in zip(rows, groups):
        if len(group) == 1:
            i = group[0]
            records.append((lengths[i], colors[i], row,
                            int(rng.integers(2 * w - lengths[i] + 1))))
        else:
            a, b = group
            maximum_gap = 2 * w - lengths[a] - lengths[b]
            gap = int(rng.integers(1, min(4, maximum_gap) + 1))
            start = int(rng.integers(2 * w - lengths[a] - lengths[b] - gap + 1))
            records.extend([(lengths[a], colors[a], row, start),
                            (lengths[b], colors[b], row, start + lengths[a] + gap)])
    return height, records


def _diagnostics(info, placed):
    positions = _positions(placed)
    rods = info["rods"]
    ignored_left = _positions(_pack(info["right"] == info["bg"], rods))
    ignored_right = _positions(_pack(info["left"] == info["bg"], rods))
    increasing = _positions(_pack(info["free"], rods, "increasing"))
    source = _positions(_pack(info["free"], rods, "source"))
    row_counts = Counter(row for _, _, row, _ in placed)
    available = info["free"].copy()
    skips = 0
    for length, _, row, start in placed:
        skips += bool(available[row + 1:].any())
        available[row, start:start + length] = False
    runs = np.sum(info["free"] & ~np.pad(info["free"][:, :-1], ((0, 0), (1, 0))), axis=1)
    return dict(both_screens=positions != ignored_left and positions != ignored_right,
                decreasing=positions != increasing, source_order=positions != source,
                copacked=max(row_counts.values()) >= 2,
                internal=any(start + length < info["w"] for length, _, _, start in placed),
                skips=bool(skips), multiple_gaps=bool(np.any(runs >= 2)))


def _make_pair(rng, count, bg, rail, stage, index):
    for _ in range(512):
        w = int(rng.integers(9, 16) if stage == "test" else rng.integers(7, 14))
        r = int(rng.integers(7, 15) if stage == "test" else rng.integers(5, 12))
        if stage == "train" and index == 0:
            lengths = sorted([1, *[int(x) for x in rng.choice(np.arange(2, w), 2, replace=False)]])
            style = 0
        else:
            lengths = sorted(int(x) for x in rng.choice(np.arange(1, w), count, replace=False))
            style = 2 if stage == "train" and index == 1 else int(rng.integers(5))
        try:
            free = _free_field(rng, r, w, style, lengths)
            states = _screens(rng, free)
            palette = [int(c) for c in rng.permutation([c for c in range(10) if c not in (bg, rail)])]
            lc, rc = palette[:2]
            colors = palette[2:2 + count]
            rack_h, records = _rack(rng, w, lengths, colors, bool(rng.integers(2)))
            a = np.full((r + 1 + rack_h, 2 * w), bg, dtype=np.int8)
            a[:r, :w] = np.where((states & 1) != 0, lc, bg)
            a[:r, w:] = np.where((states & 2) != 0, rc, bg)
            a[r] = rail
            for length, color, row, start in records:
                a[r + 1 + row, start:start + length] = color
            info = _parse(a)
            placed = _pack(free, info["rods"])
            diagnostic = _diagnostics(info, placed)
        except ValueError:
            continue
        if not diagnostic["both_screens"] or not diagnostic["decreasing"]:
            continue
        if stage == "train" and index < 2:
            if not diagnostic["source_order"]:
                continue
            if index == 0 and not diagnostic["copacked"]:
                continue
            if index == 1 and not all(diagnostic[k] for k in ("internal", "skips", "multiple_gaps")):
                continue
        return {"input": a, "output": _render(free, placed, bg, rail)}
    raise RuntimeError("bounded sampler could not produce a diagnostic pair")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    bg, rail = (int(c) for c in rng.choice(10, size=2, replace=False))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            rod_count = (3 if index == 0 else 4) if split == "train" and index < 2 else int(
                rng.integers(3, 5) if split == "train" else rng.integers(5, 7))
            for _ in range(32):
                pair = _make_pair(rng, rod_count, bg, rail, split, index)
                signature = _key(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not produce distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_placement_semantics(info, out, placed):
    """Replay and check all preferable candidates, exclusion, and conservation."""
    available = info["free"].copy()
    r, w = available.shape
    assert out.shape == (r + 1, w)
    assert np.all(out[-1] == info["rail"])
    assert np.all(out[:r][~available] == info["rail"])
    for length, color, row, start in placed:
        assert available[row, start:start + length].all()
        # An explicit candidate audit checks both row and horizontal priority.
        for y in range(row, r):
            for x in range(w - length + 1):
                if (y, x) > (row, start):
                    assert not available[y, x:x + length].all()
        assert np.all(out[row, start:start + length] == color)
        assert np.count_nonzero(out == color) == length
        available[row, start:start + length] = False
    assert np.all(out[:r][available] == info["bg"])
    assert np.count_nonzero(out == info["rail"]) == np.count_nonzero(~info["free"]) + w
    assert sum(length for length, _, _, _ in info["rods"]) == np.count_nonzero(
        ~np.isin(out, [info["bg"], info["rail"]]))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, roles = set(), set()
    diagnostics = []
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            info = _parse(pair["input"])
            out = _grid(pair["output"])
            signature = _key(info["a"])
            assert signature not in seen
            seen.add(signature)
            roles.add((info["bg"], info["rail"]))
            assert set(int(x) for x in np.unique(info["states"])) == {0, 1, 2, 3}
            placed = _pack(info["free"], info["rods"])
            assert np.array_equal(out, solve(info["a"]))
            _check_placement_semantics(info, out, placed)
            diagnostic = _diagnostics(info, placed)
            assert diagnostic["both_screens"] and diagnostic["decreasing"]
            k = len(info["rods"])
            if split == "train":
                if index < 2:
                    assert k == (3 if index == 0 else 4)
                else:
                    assert k in (3, 4)
                diagnostics.append(diagnostic)
            else:
                assert k in (5, 6)
    assert len(roles) == 1
    assert all(d["source_order"] for d in diagnostics[:2])
    assert diagnostics[0]["copacked"]
    assert all(diagnostics[1][k] for k in ("internal", "skips", "multiple_gaps"))


def _puzzle_key(puzzle):
    return tuple((_key(np.asarray(p["input"])), _key(np.asarray(p["output"])))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded 42-seed audit plus count and input-only metamorphic checks."""
    shapes, output_shapes, palettes, structures, lengths, signatures = (set() for _ in range(6))
    features = Counter()
    widths, heights, rod_counts = set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        signatures.add(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _parse(pair["input"])
                a, out = info["a"], pair["output"]
                widths.add(info["w"])
                heights.add(info["r"])
                rod_counts.add(len(info["rods"]))
                shapes.add(a.shape)
                output_shapes.add(out.shape)
                palettes.add((info["bg"], info["rail"]))
                lengths.add(tuple(sorted(rod[0] for rod in info["rods"])))
                mapping = {}
                normalized = tuple(mapping.setdefault(int(c), len(mapping)) for c in a.flat)
                structures.add((a.shape, normalized))
                placed = _pack(info["free"], info["rods"])
                features.update(k for k, v in _diagnostics(info, placed).items() if v)
                features["shared_rack_row"] += len({rod[2] for rod in info["rods"]}) < len(info["rods"])
                features["crossing_rack_midpoint"] += any(
                    start < info["w"] < start + length for length, _, _, start in info["rods"])
                if seed < 6:
                    swapped = a.copy()
                    swapped[:info["r"], :info["w"]] = info["right"]
                    swapped[:info["r"], info["w"]:] = info["left"]
                    assert np.array_equal(solve(swapped), out)
                    permutation = np.random.default_rng(seed + 100).permutation(10)
                    assert np.array_equal(solve(permutation[a]), permutation[out])
                    # Discard all rack geometry and rebuild in decreasing length order.
                    rods = sorted(info["rods"], reverse=True)
                    reracked = np.full((info["r"] + 2 + len(rods), 2 * info["w"]),
                                       info["bg"], dtype=np.int8)
                    reracked[:info["r"] + 1] = a[:info["r"] + 1]
                    for i, (length, color, _, _) in enumerate(rods):
                        reracked[info["r"] + 1 + i, :length] = color
                    assert np.array_equal(solve(reracked), out)
    assert len(signatures) == 42
    assert len(shapes) >= 60 and len(output_shapes) >= 45
    assert len(palettes) >= 25 and len(structures) >= 200 and len(lengths) >= 100
    assert len(widths) >= 8 and len(heights) >= 9 and rod_counts == {3, 4, 5, 6}
    assert features["copacked"] >= 70 and features["multiple_gaps"] >= 100
    assert features["internal"] >= 100 and features["skips"] >= 100
    assert features["shared_rack_row"] >= 45 and features["crossing_rack_midpoint"] >= 50
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(17000 + train * 10 + test, train, test)
            assert len(p["train"]) == train and len(p["test"]) == test
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts were accepted")
    # Hand-constructed domain boundaries, outside the sampler's size ranges.
    # Expected placements are explicit rather than obtained from _pack.
    for r, w, rod_lengths, expected_positions in (
            (3, 5, [1, 2, 4], [(2, 0), (1, 3), (2, 1)]),
            (20, 15, [1, 2, 3, 4, 5, 14],
             [(19, 0), (18, 1), (18, 3), (18, 6), (18, 10), (19, 1)])):
        bg, rail, lc, rc = 0, 1, 2, 3
        a = np.zeros((r + len(rod_lengths) + 2, 2 * w), dtype=np.int8)
        a[0, 0], a[0, w + 1] = lc, rc
        a[r] = rail
        expected = np.zeros((r + 1, w), dtype=np.int8)
        expected[0, :2] = rail
        expected[-1] = rail
        for i, (length, (row, start)) in enumerate(zip(rod_lengths, expected_positions)):
            color = 4 + i
            a[r + 1 + i, :length] = color
            expected[row, start:start + length] = color
        assert np.array_equal(solve(a), expected)
    return {"seeds": 42, "pairs": 210, "input_shapes": len(shapes),
            "output_shapes": len(output_shapes), "normalized_structures": len(structures),
            "length_sets": len(lengths), "background_rail_assignments": len(palettes),
            "features": dict(features)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
