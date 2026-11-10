"""Two registered stencils interpreted by an embedded Boolean color table.

Standard library and NumPy only. solve() reads pixels, never generation state.
"""
from itertools import permutations

import numpy as np


_DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not np.issubdtype(grid.dtype, np.integer)
            or not all(1 <= n <= 30 for n in grid.shape)
            or not np.all((0 <= grid) & (grid <= 9))):
        raise ValueError("grids must be integer 2D arrays, dimensions 1..30, colors 0..9")
    return grid


def _components(cells):
    remaining = set(cells)
    parts = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIR4:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        parts.append(part)
    return parts


def _shift(cells, delta):
    dr, dc = delta
    return {(r + dr, c + dc) for r, c in cells}


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _class_grid(a, b):
    top, bottom, left, right = _bbox(a | b)
    states = np.zeros((bottom - top + 1, right - left + 1), dtype=np.int8)
    for r, c in a:
        states[r - top, c - left] += 2
    for r, c in b:
        states[r - top, c - left] += 1
    return states


def _render(a, b, key):
    return np.asarray(key, dtype=np.int8).reshape(-1)[_class_grid(a, b)].copy()


def _parse(input_grid):
    grid = _grid(input_grid)
    border = np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))
    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) != 7:
        raise ValueError("input needs exactly seven semantic colors")
    bg = int(np.bincount(border.astype(int), minlength=10).argmax())
    if np.count_nonzero(border == bg) <= len(border) // 2:
        raise ValueError("background must occupy a strict perimeter majority")
    singleton = {int(c) for c, n in zip(colors, counts) if n == 1}
    if len(singleton) != 4 or bg in singleton:
        raise ValueError("four unique singleton key colors required")
    key_cells = {tuple(map(int, p)) for p in np.argwhere(np.isin(grid, list(singleton)))}
    kr, _, kc, _ = _bbox(key_cells)
    if key_cells != {(kr, kc), (kr + 1, kc), (kr, kc + 1), (kr + 1, kc + 1)}:
        raise ValueError("key cells must form one 2 by 2 block")
    key = grid[kr:kr + 2, kc:kc + 2].copy()
    nonbg = {tuple(map(int, p)) for p in np.argwhere(grid != bg)}
    parts = _components(nonbg)
    if len(parts) != 2 or not any(key_cells <= p for p in parts):
        raise ValueError("exactly two components, with key wholly in one")
    # A whole blank pixel separates components, including diagonal neighbors.
    halo = {(r + dr, c + dc) for r, c in parts[0]
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
    if not halo.isdisjoint(parts[1]):
        raise ValueError("stencils must have a background-cell gap")
    parts.sort(key=lambda p: not key_cells <= p)
    ordinary_roles = [int(c) for c in colors if c != bg and c not in singleton]
    choices = []
    for marker, ink in (ordinary_roles, ordinary_roles[::-1]):
        bars = _components({tuple(map(int, p)) for p in np.argwhere(grid == marker)})
        if len(bars) != 2 or len(bars[0]) != len(bars[1]) or len(bars[0]) < 2:
            continue
        ordered_bars = []
        for part in parts:
            own = [bar for bar in bars if bar <= part]
            if len(own) != 1:
                break
            ordered_bars.append(own[0])
        if len(ordered_bars) != 2:
            continue
        axes, starts = [], []
        for bar in ordered_bars:
            r0, r1, c0, c1 = _bbox(bar)
            if r0 == r1 and c1 - c0 + 1 == len(bar):
                axes.append(0)  # horizontal guide, row normal
            elif c0 == c1 and r1 - r0 + 1 == len(bar):
                axes.append(1)
            else:
                break
            starts.append((r0, c0))
        if len(axes) != 2 or axes[0] != axes[1]:
            continue
        masks = [part - bar for part, bar in zip(parts, ordered_bars)]
        if any(not mask or not all(int(grid[q]) in (ink, *singleton) for q in mask)
               for mask in masks):
            continue
        normal = (-1, 0) if axes[0] == 0 else (0, -1)
        sides = []
        for mask, bar in zip(masks, ordered_bars):
            backed = [bool(mask & _shift(bar, normal)),
                      bool(mask & _shift(bar, (-normal[0], -normal[1])))]
            if sum(backed) != 1:
                break
            sides.append(normal if backed[0] else (-normal[0], -normal[1]))
        if len(sides) != 2 or sides[0] != (-sides[1][0], -sides[1][1]):
            continue
        delta = (starts[0][0] + sides[0][0] - starts[1][0],
                 starts[0][1] + sides[0][1] - starts[1][1])
        a, b = masks[0], _shift(masks[1], delta)
        if max(_class_grid(a, b).shape) > 30:
            continue
        choices.append({"a": a, "b": b, "native_b": masks[1], "key": key,
                        "key_cells": key_cells, "bars": ordered_bars,
                        "aligned_bars": [ordered_bars[0], _shift(ordered_bars[1], delta)],
                        "delta": delta, "normal": sides[0], "axis": axes[0],
                        "background": bg, "ink": ink, "marker": marker,
                        "parts": parts, "length": len(bars[0])})
    if len(choices) != 1:
        raise ValueError("guide roles and backing must have exactly one interpretation")
    return choices[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    parsed = _parse(input_grid)
    return _render(parsed["a"], parsed["b"], parsed["key"])


def _line(cells, start, end):
    r0, c0 = start
    r1, c1 = end
    if r0 != r1 and c0 != c1:
        raise ValueError("axis-aligned stroke expected")
    cells.update((r, c) for r in range(min(r0, r1), max(r0, r1) + 1)
                 for c in range(min(c0, c1), max(c0, c1) + 1))


def _mask(rng, length, height, side, depth, style, partial):
    """Negative-backed stencil with optional side arms crossing its guide.

    Shape creation only: no source masks, examples, or solver templates.
    """
    left, right = int(rng.integers(1, 4)), int(rng.integers(1, 4))
    lo, hi = -left, length - 1 + right
    cells = set()
    if style == 0:  # outline plus a 2x2 thickened corner
        cells = {(r, c) for r in range(-height, 0) for c in range(lo, hi + 1)
                 if r in (-height, -1) or c in (lo, hi)}
    elif style == 1:  # solid with independently carved upper corners
        cells = {(r, c) for r in range(-height, 0) for c in range(lo, hi + 1)}
        for right_corner in (False, True):
            if rng.random() < .6:
                dh = int(rng.integers(1, height - 2))
                dw = int(rng.integers(1, max(2, (hi - lo) // 2)))
                cells -= {(r, c) for r in range(-height, -height + dh)
                          for c in (range(hi - dw + 1, hi + 1) if right_corner
                                    else range(lo, lo + dw))}
    elif style == 2:  # wandering stepped edge, full opposite spine
        cut = lo
        for r in range(-height, 0):
            cut = int(np.clip(cut + rng.integers(-1, 2), lo, hi - 2))
            cells.update((r, c) for c in range(cut, hi + 1))
    elif style == 3:  # branching spine
        spine = int(rng.integers(lo, hi + 1))
        _line(cells, (-height, spine), (-1, spine))
        for r in range(-height, 0):
            if rng.random() < .7:
                _line(cells, (r, int(rng.integers(lo, spine + 1))),
                      (r, int(rng.integers(spine, hi + 1))))
    else:  # bent arms around an open recess
        spine = lo if rng.integers(2) else hi
        _line(cells, (-height, spine), (-1, spine))
        _line(cells, (-height, lo), (-height, hi))
        for r in range(-height + 2, -1, 2):
            if rng.random() < .5:
                end = int(rng.integers(lo, hi + 1))
                _line(cells, (r, spine), (r, end))
    # Fully connected backing rail; a pad permits a key at different locations.
    _line(cells, (-1, lo), (-1, hi))
    pad_col = lo if rng.integers(2) else hi - 1
    pad_row = -height if rng.integers(2) else -3
    cells.update((r, c) for r in range(pad_row, pad_row + 2)
                 for c in range(pad_col, pad_col + 2))
    # Connect the pad to a guaranteed boundary rail without filling its cavity.
    _line(cells, (pad_row, pad_col), (-1, pad_col))
    if side is not None:
        x = -int(rng.integers(1, left + 1)) if side == 0 else length - 1 + int(rng.integers(1, right + 1))
        _line(cells, (-1, x), (depth, x))
        arm = int(rng.integers(3))
        if arm:
            end = int(rng.integers(lo, hi + 1))
            _line(cells, (depth, x), (depth, end))
            if arm == 2 and (end < 0 or end >= length):
                _line(cells, (-1, end), (depth, end))
    if partial:
        # For outlined masks the bar itself keeps interrupted rail parts joined.
        n = int(rng.integers(1, length))
        for c in rng.choice(length, n, replace=False):
            cells.discard((-1, int(c)))
    return cells


def _orient(cells, direction):
    # Transform construction coordinates only; assembly never rotates a piece.
    if direction == 0:
        return set(cells)
    if direction == 1:
        return {(-r, c) for r, c in cells}
    if direction == 2:
        return {(c, r) for r, c in cells}
    return {(c, -r) for r, c in cells}


def _key_locations(cells):
    return [(r, c) for r, c in sorted(cells)
            if {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)} <= cells]


def _stage(rng, a, b, bar, key_cells, key, palette, mode):
    groups = []
    for mask in (a, b):
        t, bot, l, right = _bbox(mask | bar)
        groups.append((_shift(mask, (-t, -l)), _shift(bar, (-t, -l)),
                       bot - t + 1, right - l + 1, (-t, -l)))
    order = list(map(int, rng.permutation(2)))
    first, second = groups[order[0]], groups[order[1]]
    _, _, h1, w1, _ = first
    _, _, h2, w2, _ = second
    gap = int(rng.integers(1, 5))
    delta = None
    if mode == "interleaved":
        halo = {(r + dy, c + dx) for r, c in first[0] | first[1]
                for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
        candidates = [(dr, dc) for dr in range(-h2 + 1, h1)
                      for dc in range(-w2 + 1, w1)]
        for index in rng.permutation(len(candidates)):
            candidate = candidates[int(index)]
            if _shift(second[0] | second[1], candidate).isdisjoint(halo):
                delta = candidate
                break
    if delta is None:
        if mode in ("horizontal", "interleaved"):
            delta = (int(rng.integers(-h2 + 1, h1)), w1 + gap)
        elif mode == "vertical":
            delta = (h1 + gap, int(rng.integers(-w2 + 1, w1)))
        else:
            delta = h1 + gap, w1 + gap
    positions = {order[0]: (0, 0), order[1]: delta}
    all_cells = set()
    for i, group in enumerate(groups):
        all_cells |= _shift(group[0] | group[1], positions[i])
    t, bot, l, right = _bbox(all_cells)
    h, w = bot - t + 1, right - l + 1
    if h > 28 or w > 28:
        return None
    # Independent zero-to-three margins; enough border background is checked.
    pads = []
    for size in (h, w):
        before = int(rng.integers(0, min(3, 30 - size) + 1))
        after = int(rng.integers(0, min(3, 30 - size - before) + 1))
        pads.append((before, after))
    bg, ink, marker = palette
    grid = np.full((h + sum(pads[0]), w + sum(pads[1])), bg, dtype=np.int8)
    offsets = {}
    for i, group in enumerate(groups):
        dr, dc = positions[i]
        offset = dr - t + pads[0][0], dc - l + pads[1][0]
        for cells, color in ((group[0], ink), (group[1], marker)):
            for q in _shift(cells, offset):
                grid[q] = color
        offsets[i] = (offset[0] + group[4][0], offset[1] + group[4][1])
    for (r, c), color in zip(key_cells, np.asarray(key).reshape(-1)):
        grid[r + offsets[0][0], c + offsets[0][1]] = color
    return grid


def _alternative(parsed, name):
    a, b, key = parsed["a"], parsed["b"], parsed["key"]
    n = parsed["normal"]
    if name == "coincident":
        return _render(a, _shift(b, (-n[0], -n[1])), key)
    if name == "reverse":
        return _render(a, _shift(b, (-2 * n[0], -2 * n[1])), key)
    if name == "native":
        return _render(a, parsed["native_b"], key)
    if name == "guides_as_ink":
        return _render(a | parsed["aligned_bars"][0], b | parsed["aligned_bars"][1], key)
    if name == "erase_guides":
        guides = set.union(*parsed["aligned_bars"])
        states = _class_grid(a, b)
        top, _, left, _ = _bbox(a | b)
        for r, c in guides:
            if 0 <= r - top < states.shape[0] and 0 <= c - left < states.shape[1]:
                states[r - top, c - left] = 0
        return key.reshape(-1)[states]
    if name == "drop_key":
        return _render(a - parsed["key_cells"], b, key)
    states = _class_grid(a, b)
    if name == "anchor_priority":
        states[states == 3] = 2
    elif name == "mover_priority":
        states[states == 3] = 1
    elif name == "xor":
        states[states == 3] = 0
    elif name == "union":
        return np.where(states > 0, parsed["ink"], parsed["background"]).astype(np.int8)
    elif name == "quadrants":
        rows, cols = np.indices(states.shape)
        return key[(rows >= states.shape[0] // 2).astype(int),
                   (cols >= states.shape[1] // 2).astype(int)]
    else:
        raise ValueError(name)
    return key.reshape(-1)[states]


_ALTERNATIVES = ("coincident", "reverse", "native", "guides_as_ink",
                 "erase_guides", "drop_key", "anchor_priority", "mover_priority",
                 "xor", "union", "quadrants")


def _make_pair(rng, length, height_range, direction, mode, palette, key, teaching, used):
    for attempt in range(200):
        h1, h2 = map(int, rng.integers(*height_range, size=2))
        # Shared crossing side in teaching examples gives a real intersection.
        side = int(rng.integers(2))
        if teaching:
            sides = side, side
        elif rng.random() < .2:
            # Abutment without crossing arms is a legal three-state result.
            # All four table entries have already been taught in training.
            sides = None, None
        else:
            sides = tuple(None if rng.random() < .12 else int(rng.integers(2))
                          for _ in range(2))
        depths = tuple(map(int, rng.integers(2, 5, size=2)))
        styles = (0, int(rng.integers(1, 5))) if teaching else tuple(map(int, rng.integers(5, size=2)))
        partial = teaching == 2 or (not teaching and rng.random() < .45)
        a = _mask(rng, length, h1, sides[0], depths[0], styles[0], partial)
        b = {(-r, c) for r, c in _mask(rng, length, h2, sides[1], depths[1], styles[1], partial)}
        bar = {(0, c) for c in range(length)}
        a, b, bar = (_orient(cells, direction) for cells in (a, b, bar))
        # Apply the original canonical -1 shift in the oriented normal axis.
        normal = _orient({(-1, 0)}, direction).pop()
        assembled_b = _shift(b, normal)
        classes = _class_grid(a, assembled_b)
        if teaching and set(map(int, np.unique(classes))) != {0, 1, 2, 3}:
            continue
        locations = _key_locations(a)
        if not locations:
            continue
        r, c = locations[int(rng.integers(len(locations)))]
        key_cells = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
        grid = _stage(rng, a, b, bar, key_cells, key, palette,
                      mode if attempt < 150 else "horizontal")
        if grid is None or _signature(grid) in used:
            continue
        expected = _render(a, assembled_b, key)
        try:
            p = _parse(grid)
        except ValueError:
            continue
        if not np.array_equal(_render(p["a"], p["b"], p["key"]), expected):
            continue
        if teaching and any(np.array_equal(expected, _alternative(p, name))
                            for name in ("coincident", "reverse", "native", "drop_key")):
            continue
        guides = set.union(*p["aligned_bars"])
        if teaching == 2:
            if not all(bar_cells - (p["a"] | p["b"]) for bar_cells in p["aligned_bars"]):
                continue
            if not guides & (p["a"] | p["b"]):
                continue
        return {"input": grid, "output": expected}
    raise AssertionError("bounded stencil sampling exhausted")


def _argument(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    ntrain = _argument("num_train", num_train, 2, 10)
    ntest = _argument("num_test", num_test, 1, 4)
    if ntrain <= ntest:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    mixed_axes = bool(rng.integers(2))
    direction = int(rng.integers(4))
    base_length = int(rng.integers(2, 5))
    stages = ["horizontal", "vertical", "diagonal"]
    train_stage, test_stage = [stages[int(x)] for x in rng.choice(3, 2, replace=False)]
    roles = tuple(map(int, rng.choice(10, 3, replace=False)))
    changing_palette = bool(rng.integers(2))
    first_key = rng.choice([c for c in range(10) if c not in roles], 4, replace=False)
    used, puzzle = set(), {"train": [], "test": []}
    for split, count in (("train", ntrain), ("test", ntest)):
        for i in range(count):
            teaching = i + 1 if split == "train" and i < 2 else 0
            if regime == 0:
                length = (base_length if i % 2 == 0 else base_length + 2) if split == "train" else base_length + 1
            else:
                length = base_length + i % 2  # familiar lengths in tests
            heights = (4, 7) if regime == 1 and split == "train" else (7, 10) if regime == 1 else (4, 10)
            axis_dir = direction
            if mixed_axes:
                axis_dir = (direction + 2 * (i % 2)) % 4
            if split == "test" and mixed_axes:
                axis_dir = (direction + 2 * int(rng.integers(2))) % 4
            if i >= 2 or split == "test":
                axis_dir ^= int(rng.integers(2))  # either backing sign
            stage = (train_stage if split == "train" else test_stage) if regime == 2 else str(rng.choice(stages + ["interleaved"]))
            palette = roles
            if changing_palette and not teaching:
                palette = tuple(map(int, rng.choice(10, 3, replace=False)))
            if teaching == 1:
                key = first_key.copy()
            elif teaching == 2:
                key = np.roll(first_key, int(rng.integers(1, 4)))
            else:
                key = rng.choice([c for c in range(10) if c not in palette], 4, replace=False)
            pair = _make_pair(rng, length, heights, axis_dir, stage, palette,
                              key.reshape(2, 2), teaching, used)
            used.add(_signature(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def measure_pair(pair):
    p = _parse(pair["input"])
    states = _class_grid(p["a"], p["b"])
    guides = set.union(*p["aligned_bars"])
    boxes = [_bbox(part) for part in p["parts"]]
    at, ab, al, ar = boxes[0]
    bt, bb, bl, br = boxes[1]
    x_sep, y_sep = ar < bl or br < al, ab < bt or bb < at
    stage = "diagonal" if x_sep and y_sep else "horizontal" if x_sep else "vertical" if y_sep else "interleaved"
    bg = p["background"]
    return {"input_shape": tuple(np.asarray(pair["input"]).shape), "output_shape": tuple(states.shape),
            "areas": (len(p["a"]), len(p["b"])), "overlap": len(p["a"] & p["b"]),
            "classes": tuple(map(int, np.bincount(states.ravel(), minlength=4))),
            "axis": p["axis"], "normal": p["normal"], "length": p["length"],
            "stage": stage, "anchor_first": min(p["parts"][0]) < min(p["parts"][1]),
            "uncovered_guides": len(guides - (p["a"] | p["b"])),
            "covered_guides": len(guides & (p["a"] | p["b"])),
            "palette": (bg, p["ink"], p["marker"]), "key": tuple(map(int, p["key"].flat)),
            "key_location": (min(r for r, c in p["key_cells"]) - at,
                             min(c for r, c in p["key_cells"]) - al),
            "edge_contact": any(np.any(edge != bg) for edge in
                                (np.asarray(pair["input"])[0], np.asarray(pair["input"])[-1],
                                 np.asarray(pair["input"])[:, 0], np.asarray(pair["input"])[:, -1])),
            "normalized_masks": tuple(tuple(sorted(_shift(mask, (-_bbox(mask)[0], -_bbox(mask)[2]))))
                                      for mask in (p["a"], p["b"]))}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "illegal counts")
    used, parsed_train = set(), []
    axes = {"train": set(), "test": set()}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            signature = _signature(inp)
            _require(signature not in used, "duplicate input")
            used.add(signature)
            p = _parse(inp)
            _require(np.array_equal(out, solve(inp.copy())), "incorrect membership table output")
            states = _class_grid(p["a"], p["b"])
            union = p["a"] | p["b"]
            _require(np.count_nonzero(states) == len(union), "union area")
            _require(len(union) == len(p["a"]) + len(p["b"]) - len(p["a"] & p["b"]), "area conservation")
            class_counts = (out.size - len(union), len(p["b"] - p["a"]),
                            len(p["a"] - p["b"]), len(p["a"] & p["b"]))
            _require(tuple(int(np.count_nonzero(out == color)) for color in p["key"].flat)
                     == class_counts, "lookup colors do not conserve the four set classes")
            _require(np.all(np.isin(out, p["key"])), "temporary colors survived")
            _require(all(np.any(edge > 0) for edge in (states[0], states[-1], states[:, 0], states[:, -1])), "crop is not tight")
            axes[split].add(p["axis"])
            if split == "train":
                parsed_train.append((p, out))
    _require(axes["test"] <= axes["train"], "test axis not demonstrated")
    for p, out in parsed_train[:2]:
        _require(set(map(int, np.unique(_class_grid(p["a"], p["b"])))) == {0, 1, 2, 3}, "core demonstration lacks a membership state")
    _require(not np.array_equal(parsed_train[0][0]["key"], parsed_train[1][0]["key"]), "key must change in training")
    _require(not (set.union(*parsed_train[0][0]["aligned_bars"]) -
                  (parsed_train[0][0]["a"] | parsed_train[0][0]["b"])), "first guide case should be fully backed")
    second = parsed_train[1][0]
    _require(all(bar - (second["a"] | second["b"]) for bar in second["aligned_bars"]), "need uncovered cells on both guides")
    _require(bool(set.union(*second["aligned_bars"]) & (second["a"] | second["b"])), "need covered guides too")
    for name in _ALTERNATIVES:
        _require(any(not np.array_equal(out, _alternative(p, name)) for p, out in parsed_train),
                 "training failed to distinguish " + name)
    # Every wrong key index permutation must be falsified by demonstrations.
    for mapping in permutations(range(4)):
        if mapping == (0, 1, 2, 3):
            continue
        _require(any(not np.array_equal(out, p["key"].reshape(-1)[list(mapping)][_class_grid(p["a"], p["b"])])
                     for p, out in parsed_train), "ambiguous key interpretation")


def self_test():
    records, signatures, shapes, masks = [], set(), set(), set()
    for seed in range(42):
        ntest = 1 + seed % 4
        ntrain = ntest + 1 + (seed // 4) % (10 - ntest)
        puzzle = generate_puzzle(seed, ntrain, ntest)
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                signatures.add(_signature(pair["input"]))
                record = measure_pair(pair)
                records.append(record)
                shapes.add(record["input_shape"])
                masks.add(record["normalized_masks"])
        if seed in (0, 17, 41):
            replay = generate_puzzle(seed, ntrain, ntest)
            _require(all(np.array_equal(a[k], b[k]) for s in puzzle
                         for a, b in zip(puzzle[s], replay[s]) for k in ("input", "output")), "determinism")
    _require(len(signatures) == len(records), "repeated input across 42 seeds")
    _require(len(shapes) >= 70 and len(masks) >= 150, "structural variation collapsed")
    _require({r["axis"] for r in records} == {0, 1}, "both guide axes")
    _require({r["normal"] for r in records} == set(_DIR4), "four backing directions")
    _require({r["length"] for r in records} >= {2, 3, 4, 5, 6}, "guide length spectrum")
    _require({r["stage"] for r in records} >= {"horizontal", "vertical", "diagonal"}, "scatter spectrum")
    _require({r["anchor_first"] for r in records} == {False, True}, "scatter order shortcut")
    _require(any(r["edge_contact"] for r in records) and any(not r["edge_contact"] for r in records), "margins")
    _require(max(r["overlap"] for r in records) >= 8 and len({r["overlap"] for r in records}) >= 12, "collision spectrum")
    _require(any(r["overlap"] == 0 for r in records), "collision-free abutments missing")
    _require(len({r["classes"][0] for r in records}) >= 25, "neither-region spectrum")
    _require(len({r["areas"] for r in records}) >= 100, "area spectrum")
    _require(len({r["key_location"] for r in records}) >= 20, "key location spectrum")
    # All legal counts; generation validation checks demonstrations at minimum.
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain - 1) + 1):
            p = generate_puzzle(500 + 10 * ntrain + ntest, ntrain, ntest)
            _require(len(p["train"]) == ntrain and len(p["test"]) == ntest, "explicit counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(123)
    # A uniform palette permutation must commute with the transformation.
    # This catches literal-color/background assumptions using input pixels only.
    palette_rng = np.random.default_rng(20270)
    for _ in range(8):
        mapping = palette_rng.permutation(10)
        pair = p["test"][0]
        _require(np.array_equal(solve(mapping[pair["input"]]), mapping[pair["output"]]),
                 "solver is not palette-equivariant")
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")


if __name__ == "__main__":
    self_test()
    print("42-seed structural and semantic self-test passed")
