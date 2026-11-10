"""Marker-completed horizontal seeds drive alternating gap-center cascades.

All geometry and colors are inferred from the input. No parent imports, file
reads, cached examples, random state, or learned palette are used by solve.
"""
from __future__ import annotations

import hashlib
from itertools import product

import numpy as np


HOST_SEEDS = (0, 262822509, 600875783, 253927486, 723635497, 614217856,
              532138323, 91736159, 534699997, 510973651, 972754063, 687955736,
              82914810, 427916698, 514405327, 163289978, 489508691, 421333229,
              272845570, 372736062, 762516267, 902692668, 384775801, 583061078,
              674261935, 855862091, 500786768, 381728126, 549126781, 91935215,
              718544841, 586882574, 665937428, 47309677, 317038540, 925579214,
              147212755, 238307149, 930305936, 647325958, 663670649, 340758155)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grid must have two dimensions of size 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and
             bool(np.all((0 <= a) & (a <= 9))), "grid needs integer colors 0..9")
    return a.astype(np.int8, copy=False)


def _next(mask):
    following = np.zeros(len(mask), dtype=bool)
    following[1:-1] = mask[:-2] & ~mask[1:-1] & mask[2:]
    return following


def _rows(mask):
    """Nonempty derived masks only. Each step strictly shrinks the active hull."""
    result = []
    current = mask.copy()
    for _ in range(len(mask)):
        current = _next(current)
        if not current.any():
            return result
        result.append(current)
    raise ValueError("noncontracting recurrence")


def _read(input_grid):
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    _require(len(colors) == 3 and counts.max() > a.size // 2,
             "input needs background, gappy seed ink, and marker ink")
    bg = int(colors[counts.argmax()])
    candidates = []
    for color in colors:
        color = int(color)
        if color == bg:
            continue
        cells = np.argwhere(a == color)
        if len(np.unique(cells[:, 0])) == 1:
            columns = cells[:, 1]
            if len(columns) >= 2 and columns[-1] - columns[0] + 1 > len(columns):
                candidates.append((color, int(cells[0, 0]), columns))
    _require(len(candidates) == 1, "source must be the unique gappy monochrome row")
    seed, source_row, columns = candidates[0]
    marker = next(int(c) for c in colors if c not in (bg, seed))
    cells = np.argwhere(a == marker)
    mr, mc = map(int, cells[0])
    if len(cells) == 1:
        kind = "P"
    elif np.all(cells[:, 0] == mr) and np.array_equal(
            cells[:, 1], np.arange(cells[0, 1], cells[-1, 1] + 1)):
        kind = "H"
    elif np.all(cells[:, 1] == mc) and np.array_equal(
            cells[:, 0], np.arange(cells[0, 0], cells[-1, 0] + 1)):
        kind = "V"
    else:
        raise ValueError("marker must be a straight segment or singleton")
    if kind == "V":
        _require(cells[-1, 0] == source_row - 1,
                 "vertical axis must end immediately above source row")
        _require(bool(np.all(columns < mc) or np.all(columns > mc)),
                 "vertical-mirror source must be on one side of the axis")
        target_row = source_row
        target_columns = 2 * mc - columns
    else:
        _require(mr > source_row, "horizontal axis/center must be below source")
        target_row = 2 * mr - source_row
        target_columns = columns.copy() if kind == "H" else 2 * mc - columns
    _require(0 <= target_row < a.shape[0] and
             bool(np.all((0 <= target_columns) & (target_columns < a.shape[1]))),
             "symmetric copy must fit")
    _require(bool(np.all(a[target_row, target_columns] == bg)),
             "symmetric destinations must be clear")
    _require(bool(np.all(a[target_row + 1:] == bg)),
             "canvas below the completed seed must be empty")
    mask = np.zeros(a.shape[1], dtype=bool)
    if kind == "V":
        mask[columns] = True
    mask[target_columns] = True
    generations = _rows(mask)
    _require(bool(generations), "completed seed must generate at least one row")
    _require(target_row + 2 * len(generations) < a.shape[0],
             "canvas must fit the entire cascade")
    return {"array": a, "background": bg, "seed": seed, "marker": marker,
            "kind": kind, "source_row": source_row, "columns": columns,
            "marker_row": mr, "marker_column": mc, "marker_cells": cells,
            "target_row": target_row, "target_columns": target_columns,
            "mask": mask, "generations": generations}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Complete marker symmetry once, then recurse from its lowest seed row."""
    d = _read(input_grid)
    output = d["array"].copy()
    output[d["target_row"], d["target_columns"]] = d["seed"]
    for generation, mask in enumerate(d["generations"], 1):
        output[d["target_row"] + 2 * generation, mask] = (
            d["marker"] if generation % 2 else d["seed"])
    return output


def _alternating(depth):
    result = np.zeros(2 * depth + 1, dtype=bool)
    result[::2] = True
    return result


def _pack(rng, pieces, max_width=30):
    """Two or more blanks separate independent fragments; no hidden ownership."""
    pieces = list(pieces)
    rng.shuffle(pieces)
    gaps = [int(rng.integers(2, 5)) for _ in range(len(pieces) - 1)]
    if sum(map(len, pieces)) + sum(gaps) > max_width:
        gaps = [2] * len(gaps)
    size = sum(map(len, pieces)) + sum(gaps)
    _require(size <= max_width, "fragments too wide")
    result = np.zeros(size, dtype=bool)
    start = 0
    for i, piece in enumerate(pieces):
        result[start:start + len(piece)] = piece
        start += len(piece) + (gaps[i] if i < len(gaps) else 0)
    return result


def _horizontal_pattern(rng, depth, witness, max_width=30):
    # All constructions have a depth-setting chain, but it need not be the
    # largest or only source fragment. Dense one-hole seeds are a special case.
    if depth == 1 and rng.random() < .45:
        left, right = map(int, rng.integers(1, 6, size=2))
        return np.array([1] * left + [0] + [1] * right, dtype=bool)
    pieces = [_alternating(depth)]
    room = max_width - len(pieces[0]) - 2
    solid = int(rng.integers(3 if witness else 1, min(6, room) + 1))
    pieces.append(np.ones(solid, dtype=bool))
    room = max_width - sum(map(len, pieces)) - 4
    if room >= 3 and rng.random() < .65:
        other_depth = int(rng.integers(1, min(depth, (room - 1) // 2) + 1))
        pieces.append(_alternating(other_depth))
    return _pack(rng, pieces, max_width)


def _vertical_pattern(rng, depth, witness):
    # Distances 1,3,...,depth on both sides create a chain ACROSS the axis.
    # Its depth is odd. Extra solid cells do not generate centers of XXX.
    _require(depth % 2 == 1, "seam cascade needs an odd depth")
    if depth == 1:
        pattern = np.array([1, 1, 1, 0, 0, 1, 0, 1], dtype=bool)
    else:
        solid_extra = int(rng.integers(2 if witness else 1, min(3, 14 - depth) + 1))
        pattern = np.zeros(depth + solid_extra, dtype=bool)
        pattern[:depth:2] = True
        pattern[depth:] = True
        remaining = 14 - len(pattern)
        if remaining >= 3 and rng.random() < .7:
            gap = int(rng.integers(2, min(4, remaining - 1) + 1))
            count = int(rng.integers(1, remaining - gap + 1))
            extra = rng.random(count) < .5
            extra[-1] = True
            pattern = np.concatenate((pattern, np.zeros(gap, bool), extra))
        if witness and np.array_equal(pattern, pattern[::-1]):
            # Optional far fragments must not erase the orientation witness.
            pattern = pattern[:depth + solid_extra]
    mask = np.zeros(2 * len(pattern) + 1, dtype=bool)
    center = len(pattern)
    distances = np.flatnonzero(pattern) + 1
    mask[center - distances] = mask[center + distances] = True
    _require(len(_rows(mask)) == depth, "vertical construction depth changed")
    return pattern


def _asymmetric(columns):
    relative = columns - columns.min()
    return set(map(int, relative)) != set(map(int, relative.max() - relative))


def _make_pair(rng, kind, depth, palette, witness=False):
    bg, seed, marker = palette
    if kind == "V":
        pattern = _vertical_pattern(rng, depth, witness)
        minimum_width = 2 * len(pattern) + 1
        width = int(rng.integers(minimum_width, 31))
        axis = int(rng.integers(len(pattern), width - len(pattern)))
        side = int(rng.choice((-1, 1)))
        columns = np.sort(axis + side * (np.flatnonzero(pattern) + 1))
        length = int(rng.integers(2, 5))
        source_row = int(rng.integers(length, min(9, 29 - 2 * depth) + 1))
        launch_row = source_row
        marker_cells = [(r, axis) for r in range(source_row - length, source_row)]
    else:
        # Reversal witnesses distinguish half-turns from unchanged translation.
        for _ in range(64):
            pattern = _horizontal_pattern(rng, depth, witness, 29 if kind == "P" else 30)
            columns0 = np.flatnonzero(pattern)
            if kind != "P" or _asymmetric(columns0):
                break
        else:
            # A chain followed after two blanks by a solid run is asymmetric.
            pattern = np.concatenate((_alternating(depth), np.zeros(2, bool),
                                      np.ones(3, bool)))
        minimum_width = len(pattern) + (kind == "P" and len(pattern) % 2 == 0)
        width = int(rng.integers(minimum_width, 31))
        left = int(rng.integers(0, width - len(pattern) + 1))
        columns = left + np.flatnonzero(pattern)
        distance = int(rng.integers(1, min(3, (29 - 2 * depth) // 2) + 1))
        source_row = int(rng.integers(0, min(8, 29 - 2 * depth - 2 * distance) + 1))
        axis_row = source_row + distance
        launch_row = source_row + 2 * distance
        if kind == "H":
            length = int(rng.integers(2, min(7, width) + 1))
            start = int(rng.integers(0, width - length + 1))
            marker_cells = [(axis_row, c) for c in range(start, start + length)]
        else:
            low = (int(columns.max()) + 1) // 2
            high = (width - 1 + int(columns.min())) // 2
            axis = int(rng.integers(low, high + 1))
            marker_cells = [(axis_row, axis)]
    tail = int(rng.integers(0, min(8, 29 - launch_row - 2 * depth) + 1))
    height = launch_row + 2 * depth + 1 + tail
    inp = np.full((height, width), bg, dtype=np.int8)
    inp[source_row, columns] = seed
    for cell in marker_cells:
        inp[cell] = marker
    out = solve(inp)
    _require(len(_read(inp)["generations"]) == depth, "unexpected cascade depth")
    return {"input": inp, "output": out}


def _count(value, name, low, high):
    _require(isinstance(value, (int, np.integer)) and
             not isinstance(value, (bool, np.bool_)) and low <= value <= high,
             f"{name} must be an integer in {low}..{high}")
    return int(value)


def _signature(a):
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Deterministic, bounded construction; at most 14 pairs and 64 retries/pair."""
    nt = _count(num_train, "num_train", 2, 10)
    ns = _count(num_test, "num_test", 1, 4)
    _require(nt > ns, "num_train must exceed num_test")
    _require(isinstance(seed, (int, np.integer)) and
             not isinstance(seed, (bool, np.bool_)) and seed >= 0,
             "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    base_palette = tuple(map(int, rng.choice(10, 3, replace=False)))
    vary_palette = bool(rng.random() < .35)
    subsets = [("H",), ("V",), ("P",), ("H", "V"), ("H", "P"), ("V", "P")]
    if nt >= 3:
        subsets.extend([("H", "V", "P")] * 3)
    kinds = subsets[int(rng.integers(len(subsets)))]
    cap = int(rng.integers(3, 6))
    train_kinds = list(kinds) + [str(rng.choice(kinds)) for _ in range(nt - len(kinds))]
    rng.shuffle(train_kinds)
    witness_indices = {kind: train_kinds.index(kind) for kind in kinds}
    train_plan = []
    for i, kind in enumerate(train_kinds):
        witness = witness_indices[kind] == i
        choices = [3, 5] if kind == "V" else list(range(2 if witness else 1, cap + 1))
        choices = [d for d in choices if d <= cap]
        depth = int(rng.choice(choices))
        train_plan.append((kind, depth, witness))
    max_train = max(d for _, d, _ in train_plan)
    test_plan = []
    shift_index = int(rng.integers(ns))
    for i in range(ns):
        kind = str(rng.choice(kinds))
        low = max_train + 1 if i == shift_index else 1
        choices = ([d for d in (1, 3, 5, 7, 9) if d >= low] if kind == "V"
                   else list(range(low, 10)))
        test_plan.append((kind, int(rng.choice(choices)), False))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, plan in (("train", train_plan), ("test", test_plan)):
        for kind, depth, witness in plan:
            for _ in range(64):
                palette = base_palette
                if vary_palette:
                    # Marker ink remains stable; source/background roles may vary.
                    other = [c for c in range(10) if c != base_palette[2]]
                    bg, ink = map(int, rng.choice(other, 2, replace=False))
                    palette = (bg, ink, base_palette[2])
                pair = _make_pair(rng, kind, depth, palette, witness)
                key = _signature(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("unique input retry budget exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _no_symmetry_output(d):
    """Counterfactual B acting directly on the original source row."""
    out = d["array"].copy()
    source_mask = d["array"][d["source_row"]] == d["seed"]
    for generation, mask in enumerate(_rows(source_mask), 1):
        row = d["source_row"] + 2 * generation
        if row >= len(out):
            break
        vacant = out[row] == d["background"]
        out[row, mask & vacant] = d["marker"] if generation % 2 else d["seed"]
    return out


def validate_generated_puzzle(puzzle) -> None:
    """Check labels, preservation, exact spacers, causal witnesses, and coverage."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle keys must be train/test")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _count(nt, "num_train", 2, 10)
    _count(ns, "num_test", 1, 4)
    _require(nt > ns, "num_train must exceed num_test")
    seen, known, oriented = set(), set(), set()
    train_depths, test_depths = [], []
    trained_solid = False
    marker_colors = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair keys must be input/output")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "canvas changed")
            key = _signature(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            d = _read(inp)
            marker_colors.add(d["marker"])
            bg, seed = d["background"], d["seed"]
            _require(np.array_equal(inp[inp != bg], out[inp != bg]), "input ink changed")
            _require(np.array_equal(out, solve(inp)), "incorrect symmetry/cascade")
            _require(np.count_nonzero(inp != out) == len(d["columns"]) +
                     sum(int(mask.sum()) for mask in d["generations"]), "wrong additions")
            _require(np.all(out[d["target_row"], d["target_columns"]] == seed),
                     "missing symmetric seed")
            current = set(map(int, np.flatnonzero(d["mask"])))
            for generation in range(1, len(d["generations"]) + 1):
                # Scalar stencil checks the actual rows, including no active centers.
                following = {c for c in range(1, inp.shape[1] - 1)
                             if c not in current and c - 1 in current and c + 1 in current}
                row = d["target_row"] + 2 * generation
                expected_ink = d["marker"] if generation % 2 else seed
                _require(np.all(out[row - 1] == bg), "spacer must be exactly one blank row")
                _require(set(map(int, np.flatnonzero(out[row] != bg))) == following,
                         "wrong triplet centers")
                _require(np.all(out[row, sorted(following)] == expected_ink), "wrong phase ink")
                current = following
            _require(not any(c not in current and c - 1 in current and c + 1 in current
                             for c in range(1, inp.shape[1] - 1)), "cascade stopped early")
            last = d["target_row"] + 2 * len(d["generations"])
            _require(np.all(out[last + 1:] == bg), "tail must remain blank")
            completed = inp.copy()
            completed[d["target_row"], d["target_columns"]] = seed
            _require(not np.array_equal(completed, out), "B has no effect")
            no_a = _no_symmetry_output(d)
            _require(not np.array_equal(no_a, out), "A has no effect")
            # Causal rather than just additive: lower generation cells differ too.
            _require(not np.array_equal(no_a[d["target_row"] + 1:],
                                       out[d["target_row"] + 1:]),
                     "symmetry must change the later cascade")
            if split == "train":
                known.add(d["kind"])
                if d["kind"] == "H" or _asymmetric(d["columns"]):
                    oriented.add(d["kind"])
                train_depths.append(len(d["generations"]))
                mask = d["mask"]
                trained_solid |= bool(np.any(mask[:-2] & mask[1:-1] & mask[2:]))
            else:
                _require(d["kind"] in known, "untrained marker kind")
                test_depths.append(len(d["generations"]))
    _require(len(marker_colors) == 1, "marker ink convention changed within puzzle")
    _require(known <= oriented, "training conceals reversal as translation")
    _require(max(train_depths) >= 2 and trained_solid,
             "training needs parity and active-center rejection witnesses")
    _require(max(test_depths) > max(train_depths), "missing recurrence-depth test shift")


def _puzzle_digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = _grid(pair[role])
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _fixtures():
    # Hand specified additions, independently pinned to coordinates.
    cases = []
    v = np.zeros((12, 11), dtype=np.int8)
    v[0:2, 5] = 8
    v[2, [1, 2, 4]] = 3
    expected = v.copy()
    expected[2, [6, 8, 9]] = 3
    expected[4, [3, 5, 7]] = 8
    expected[6, [4, 6]] = 3
    expected[8, 5] = 8
    cases.append((v, expected))
    # A valid even-depth vertical member outside the sampler's odd seam depths.
    even = np.zeros((7, 15), dtype=np.int8)
    even[0:2, 7] = 8
    even[2, [1, 3, 5, 6]] = 3
    expected = even.copy()
    expected[2, [8, 9, 11, 13]] = 3
    expected[4, [2, 4, 7, 10, 12]] = 8
    expected[6, [3, 11]] = 3
    cases.append((even, expected))
    for kind in ("H", "P"):
        a = np.zeros((15, 12), dtype=np.int8)
        a[1, [1, 3, 5, 6, 7]] = 3
        if kind == "H":
            a[3, 3:6] = 8
            copied, first, second = [1, 3, 5, 6, 7], [2, 4], [3]
        else:
            a[3, 5] = 8
            copied, first, second = [3, 4, 5, 7, 9], [6, 8], [7]
        expected = a.copy()
        expected[5, copied] = 3
        expected[7, first] = 8
        expected[9, second] = 3
        cases.append((a, expected))
    return cases


def self_test() -> dict:
    """Streaming 42-seed audit; no files or saved execution traces."""
    for triple in product((False, True), repeat=3):
        expected = triple == (True, False, True)
        assert bool(_next(np.array(triple))[1]) == expected
    for inp, expected in _fixtures():
        before = inp.copy()
        assert np.array_equal(solve(inp), expected)
        assert np.array_equal(inp, before)
    shapes, layouts, palettes, modes, depths, signatures = set(), set(), set(), set(), set(), set()
    source_sides, marker_lengths, offsets = set(), set(), set()
    stats = {"seeds": 42, "pairs": 0, "one_generation": 0, "multi_cascade": 0,
             "edge_contact": 0, "long_tail": 0, "causal_pairs": 0}
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        assert _puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed))
        signatures.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _read(pair["input"])
                depth = len(d["generations"])
                shapes.add(d["array"].shape)
                layouts.add((d["kind"], d["array"].shape[1],
                             tuple(map(int, np.flatnonzero(d["mask"])))))
                palettes.add((d["background"], d["seed"], d["marker"]))
                modes.add(d["kind"])
                depths.add(depth)
                marker_lengths.add(len(d["marker_cells"]))
                offsets.add(d["target_row"] - d["source_row"])
                if d["kind"] == "V":
                    source_sides.add(int(np.sign(d["columns"][0] - d["marker_column"])))
                stats["pairs"] += 1
                stats["one_generation"] += depth == 1
                stats["multi_cascade"] += bool(np.any(np.diff(np.flatnonzero(d["generations"][0])) > 2))
                stats["edge_contact"] += bool(d["mask"][0] or d["mask"][-1])
                stats["long_tail"] += d["array"].shape[0] - 1 - d["target_row"] - 2 * depth >= 5
                stats["causal_pairs"] += 1  # validator checks the lower-row ablation
    assert len(signatures) == 42 and modes == {"H", "V", "P"}
    assert len(shapes) >= 50 and len(layouts) >= 100 and len(palettes) >= 30
    assert {1, 2, 3}.issubset(depths) and max(depths) >= 8
    assert source_sides == {-1, 1} and {1, 2, 3, 4}.issubset(marker_lengths)
    assert {0, 2, 4, 6}.issubset(offsets)
    assert all(stats[key] > 0 for key in ("one_generation", "multi_cascade", "edge_contact", "long_tail"))
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            generate_puzzle(10000 + 10 * nt + ns, nt, ns)
    for nt, ns in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (2.0, 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Label corruptions: missing symmetry, wrong phase, off-by-one spacer.
    p = generate_puzzle(0)
    d = _read(p["train"][0]["input"])
    row = d["target_row"] + 2
    column = int(np.flatnonzero(d["generations"][0])[0])
    for mutation in ("symmetry", "phase", "spacer"):
        bad = {s: [{k: np.array(v, copy=True) for k, v in pair.items()} for pair in p[s]] for s in p}
        out = bad["train"][0]["output"]
        if mutation == "symmetry":
            out[d["target_row"], int(d["target_columns"][0])] = d["background"]
        elif mutation == "phase":
            out[row, column] = d["seed"]
        else:
            out[row - 1, column] = d["marker"]
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("corrupt label accepted")
    stats.update({"distinct_puzzles": len(signatures), "shapes": len(shapes),
                  "seed_layouts": len(layouts), "palettes": len(palettes),
                  "modes": sorted(modes), "depths": sorted(depths),
                  "marker_lengths": sorted(marker_lengths), "seed_row_offsets": sorted(offsets)})
    return stats


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
