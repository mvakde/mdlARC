"""Projected rectangular flip windows on a reversible two-color square.

Only the standard library and NumPy are used.  There is no puzzle-level state in
solve: marker equality, square geometry, and the two exchanged colors are visible.
"""

from collections import deque
import json

import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "invalid grid color")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _counts(n, rectangles, outlines=False, strict=False):
    result = np.zeros((n, n), dtype=np.int8)
    for r0, r1, c0, c1 in rectangles:
        mask = np.zeros((n, n), dtype=bool)
        if strict:
            mask[r0 + 1:r1, c0 + 1:c1] = True
        else:
            mask[r0:r1 + 1, c0:c1 + 1] = True
            if outlines:
                mask[r0 + 1:r1, c0 + 1:c1] = False
        result += mask
    return result


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    components = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue, cells = deque([start]), [start]
        while queue:
            r, c = queue.popleft()
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    cells.append(neighbor)
                    queue.append(neighbor)
        components.append(cells)
    return sorted(components, key=len, reverse=True)


def _decode(input_grid):
    """Parse the visible grammar, independently of any sampling recipe."""
    a = _grid(input_grid)
    bg = int(a[0, 0])
    _require(all(np.all(edge == bg) for edge in (a[0], a[-1], a[:, 0], a[:, -1])),
             "the outer canvas border must be background")
    components = _components(a != bg)
    _require(len(components) >= 2 and len(components[0]) > len(components[1]),
             "missing unique largest square component")
    coords = np.array(components[0])
    top, left = coords.min(axis=0).tolist()
    bottom, right = coords.max(axis=0).tolist()
    n = bottom - top + 1
    _require(n == right - left + 1 and 5 <= n <= 18, "unsupported square")
    tile = a[top:bottom + 1, left:right + 1].copy()
    _require(len(coords) == n * n and np.all(tile != bg), "tile must be solid")
    colors = list(map(int, np.unique(tile)))
    _require(len(colors) == 2, "tile must have two colors")
    outer = int(tile[0, 0])
    core = next(c for c in colors if c != outer)
    core_coords = np.argwhere(tile == core)
    r0, c0 = core_coords.min(axis=0).tolist()
    r1, c1 = core_coords.max(axis=0).tolist()
    core_side = r1 - r0 + 1
    _require(core_side == c1 - c0 + 1 and core_side in (1, 2, 3, 4), "invalid core")
    _require(n - core_side >= 2 and (n - core_side) % 2 == 0,
             "core must fit centrally inside a ring")
    _require(r0 == c0 == (n - core_side) // 2, "core must be centered")
    expected_tile = np.full((n, n), outer, dtype=np.int8)
    expected_tile[r0:r1 + 1, c0:c1 + 1] = core
    _require(np.array_equal(tile, expected_tile), "invalid two-color motif")
    _require(np.count_nonzero(tile == outer) > np.count_nonzero(tile == core),
             "outer fill must dominate")

    exterior = a.copy()
    exterior[top:bottom + 1, left:right + 1] = bg
    _require(not np.any(np.isin(exterior, colors)), "tile colors cannot label markers")
    marker_colors = [int(c) for c in np.unique(exterior) if c != bg]
    _require(2 <= len(marker_colors) <= 4, "need two to four window colors")
    rectangles, sides, axis_spans = [], set(), []
    for color in marker_colors:
        points = np.argwhere(exterior == color)
        _require(len(points) == 4, "each window has four endpoint markers")
        row_points, col_points = [], []
        for r, c in points.tolist():
            if top <= r <= bottom and (c <= left - 2 or c >= right + 2):
                row_points.append((r, c))
            elif left <= c <= right and (r <= top - 2 or r >= bottom + 2):
                col_points.append((r, c))
            else:
                raise AssertionError("marker is not on an exterior projection rail")
        _require(len(row_points) == len(col_points) == 2, "need two markers per axis")
        _require(row_points[0][1] == row_points[1][1], "row endpoints need one rail")
        _require(col_points[0][0] == col_points[1][0], "column endpoints need one rail")
        rows = sorted(r - top for r, _ in row_points)
        cols = sorted(c - left for _, c in col_points)
        _require(rows[0] < rows[1] and cols[0] < cols[1], "window has zero extent")
        rectangles.append((rows[0], rows[1], cols[0], cols[1]))
        hc, vr = row_points[0][1], col_points[0][0]
        hd = left - hc if hc < left else hc - right
        vd = top - vr if vr < top else vr - bottom
        axis_spans.append((hd, tuple(rows), vd, tuple(cols)))
        sides.add("L" if row_points[0][1] < left else "R")
        sides.add("T" if col_points[0][0] < top else "B")
    _require(np.count_nonzero(a == bg) > a.size // 2, "background must be a majority")
    return {"tile": tile, "outer": outer, "core": core, "core_side": core_side,
            "n": n, "rectangles": rectangles, "sides": sides,
            "axis_spans": axis_spans,
            "box": (top, bottom, left, right), "bg": bg,
            "palette": (bg, outer, core, *marker_colors), "shape": a.shape}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Crop the square; exchange its colors once for each projected window."""
    d = _decode(input_grid)
    tile = d["tile"]
    flip = np.zeros(tile.shape, dtype=bool)
    for r0, r1, c0, c1 in d["rectangles"]:
        flip[r0:r1 + 1, c0:c1 + 1] ^= True
    result = tile.copy()
    result[flip & (tile == d["outer"])] = d["core"]
    result[flip & (tile == d["core"])] = d["outer"]
    return result


def _distance_shortcut(d):
    rows = sorted((hd, row_span) for hd, row_span, _, _ in d["axis_spans"])
    cols = sorted((vd, col_span) for _, _, vd, col_span in d["axis_spans"])
    wrong_rectangles = [(r[0], r[1], c[0], c[1]) for (_, r), (_, c) in zip(rows, cols)]
    return _counts(d["n"], wrong_rectangles) % 2 == 1


def _random_rectangle(rng, n, min_side=2):
    h, w = map(int, rng.integers(min_side, n + 1, size=2))
    r, c = int(rng.integers(n - h + 1)), int(rng.integers(n - w + 1))
    return r, r + h - 1, c, c + w - 1


def _inside(rng, rect, minimum=2):
    r0, r1, c0, c1 = rect
    h, w = r1 - r0 - 1, c1 - c0 - 1
    rh, rw = int(rng.integers(minimum, h + 1)), int(rng.integers(minimum, w + 1))
    r, c = int(rng.integers(r0 + 1, r1 - rh + 1)), int(rng.integers(c0 + 1, c1 - rw + 1))
    return r, r + rh - 1, c, c + rw - 1


def _contains(a, b):
    return a[0] <= b[0] and b[1] <= a[1] and a[2] <= b[2] and b[3] <= a[3]


def _strict_contains(a, b):
    return a[0] < b[0] and b[1] < a[1] and a[2] < b[2] and b[3] < a[3]


def _relations(rectangles):
    pairs = [(a, b) for i, a in enumerate(rectangles) for b in rectangles[i + 1:]]
    nested = any(_strict_contains(a, b) or _strict_contains(b, a) for a, b in pairs)
    incomparable = any(not _contains(a, b) and not _contains(b, a) for a, b in pairs)
    return nested, incomparable


def _sample_rectangles(rng, n, k, mode, core_side, teaching=False, max_depth=4):
    center = (n - core_side) // 2
    core_mask = np.zeros((n, n), dtype=bool)
    core_mask[center:center + core_side, center:center + core_side] = True
    for _ in range(6000):
        if mode == "nested":
            outer = _random_rectangle(rng, n, 6)
            middle = _inside(rng, outer, 4)
            rectangles = [outer, middle, _inside(rng, middle)]
        elif mode == "mixed":
            outer = _random_rectangle(rng, n, 4)
            rectangles = [outer, _inside(rng, outer), _random_rectangle(rng, n)]
            # The third window crosses the outer one, rather than nesting in it.
            if _contains(rectangles[0], rectangles[2]) or _contains(rectangles[2], rectangles[0]):
                continue
            if k == 4:
                rectangles.append(_random_rectangle(rng, n))
        elif mode == "duplicate":
            first = _random_rectangle(rng, n)
            rectangles = [first, first] + [_random_rectangle(rng, n) for _ in range(k - 2)]
        elif mode == "bars":
            rectangles = []
            for _ in range(k):
                r0, r1, c0, c1 = _random_rectangle(rng, n)
                if rng.random() < 0.5:
                    r1 = r0 + 1
                else:
                    c1 = c0 + 1
                rectangles.append((r0, r1, c0, c1))
        else:
            rectangles = [_random_rectangle(rng, n) for _ in range(k)]
        if mode == "crossed" and (_contains(*rectangles) or _contains(*rectangles[::-1])):
            continue
        if mode == "crossed" and (rectangles[0][:2] == rectangles[1][:2]
                                  or rectangles[0][2:] == rectangles[1][2:]):
            continue
        # Test combines two relations that training teaches separately. Equal
        # windows can still cancel in training without constituting strict nesting.
        if mode != "mixed" and all(_relations(rectangles)):
            continue
        depth = _counts(n, rectangles)
        levels = set(map(int, np.unique(depth)))
        if not {0, 1, 2} <= levels or int(depth.max()) > max_depth:
            continue
        if mode in ("nested", "mixed") and 3 not in levels:
            continue
        odd = depth % 2 == 1
        if not np.any(odd & ~core_mask) or not np.any(~odd & ~core_mask):
            continue
        if teaching:
            if not np.any(odd & core_mask):
                continue
            if core_side > 1 and not np.any(~odd & core_mask):
                continue
            # Teach filled, inclusive intervals rather than outlines or open intervals.
            if np.array_equal(odd, _counts(n, rectangles, outlines=True) % 2 == 1):
                continue
            if np.array_equal(odd, _counts(n, rectangles, strict=True) % 2 == 1):
                continue
        order = rng.permutation(len(rectangles))
        return [rectangles[int(i)] for i in order]
    raise RuntimeError(f"failed to sample {mode} windows on side {n}")


def _make_pair(rng, n, core_side, k, mode, palette, teaching=False, max_depth=4):
    rectangles = _sample_rectangles(rng, n, k, mode, core_side, teaching, max_depth)
    bg, outer, core = map(int, palette[:3])
    tile = np.full((n, n), outer, dtype=np.int8)
    start = (n - core_side) // 2
    tile[start:start + core_side, start:start + core_side] = core
    # Each label gets a separate rail distance; endpoint markers never collide.
    base = k + 2
    margins = []
    for _ in range(2):
        room = 30 - n - 2 * base
        extra = int(rng.integers(min(6, room) + 1))
        before = int(rng.integers(extra + 1))
        margins.extend((base + before, base + extra - before))
    top, bottom, left, right = margins
    a = np.full((n + top + bottom, n + left + right), bg, dtype=np.int8)
    a[top:top + n, left:left + n] = tile
    horizontal_sides = rng.integers(0, 2, size=k)
    vertical_sides = rng.integers(0, 2, size=k)
    if mode == "crossed" and teaching:
        horizontal_sides[1] = 1 - horizontal_sides[0]
        vertical_sides[1] = 1 - vertical_sides[0]
    horizontal_distances = rng.permutation(np.arange(2, k + 2))
    vertical_distances = rng.permutation(np.arange(2, k + 2))
    if mode == "crossed" and teaching:
        vertical_distances = horizontal_distances[::-1]
    for index, (r0, r1, c0, c1) in enumerate(rectangles):
        color = int(palette[3 + index])
        hd, vd = int(horizontal_distances[index]), int(vertical_distances[index])
        rail_col = left - hd if horizontal_sides[index] == 0 else left + n - 1 + hd
        rail_row = top - vd if vertical_sides[index] == 0 else top + n - 1 + vd
        a[top + r0, rail_col] = a[top + r1, rail_col] = color
        a[rail_row, left + c0] = a[rail_row, left + c1] = color
    depth = _counts(n, rectangles)
    output = np.where(depth % 2 == 0, tile,
                      np.where(tile == outer, core, outer)).astype(np.int8)
    return {"input": a, "output": output}


def _legal_counts(num_train, num_test):
    for value, lo, hi, name in ((num_train, 2, 10, "num_train"), (num_test, 1, 4, "num_test")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not lo <= value <= hi:
            raise ValueError(f"{name} outside {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _legal_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    stable_palette = bool(rng.random() < 0.35)
    base_palette = rng.choice(10, size=7, replace=False)
    # These two anchors suffice even for the smallest legal training set.
    plans = [(int(rng.choice(np.arange(5, 18, 2))), 1, 2, "crossed", True),
             (int(rng.choice(np.arange(8, 19, 2))), 2, 3, "nested", True)]
    for index in range(2, num_train):
        n = int(rng.integers(5, 19))
        core_side = 1 if n % 2 else 2
        if n >= (7 if n % 2 else 8) and rng.random() < 0.5:
            core_side += 2
        k = int(rng.integers(2, 5))
        modes = ["free", "bars"] + (["duplicate"] if k >= 3 else [])
        plans.append((n, core_side, k, str(rng.choice(modes)), False))
    rng.shuffle(plans)
    taught_cores = sorted({plan[1] for plan in plans})
    max_windows = max(plan[2] for plan in plans)
    puzzle, used = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if split == "train":
                n, core_side, k, mode, teaching = plans[index]
                allowed_depth = 4
            else:
                core_side = int(rng.choice(taught_cores))
                possible = [n for n in range(5, 19) if (n - core_side) % 2 == 0 and n >= core_side + 2]
                n = int(rng.choice(possible))
                k = int(rng.integers(3, max_windows + 1))
                mode, teaching = "mixed", False
                allowed_depth = max(int(_counts(d["n"], d["rectangles"]).max())
                                    for d in map(lambda p: _decode(p["input"]), puzzle["train"]))
            for _ in range(64):
                palette = base_palette if stable_palette else rng.choice(10, size=7, replace=False)
                pair = _make_pair(rng, n, core_side, k, mode, palette, teaching, allowed_depth)
                if _key(pair["input"]) not in used:
                    break
            else:
                raise RuntimeError("could not sample distinct inputs")
            used.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
    a, out = _grid(pair["input"]), _grid(pair["output"])
    d = _decode(a)
    tile = d["tile"]
    depth = _counts(d["n"], d["rectangles"])
    odd = depth % 2 == 1
    # Arithmetic superposition provides a separate check from solve's in-place XOR.
    expected = tile.copy()
    expected[odd] = d["outer"] + d["core"] - tile[odd]
    _require(out.shape == tile.shape and np.array_equal(out, expected), "incorrect flip-window output")
    _require({0, 1, 2} <= set(map(int, np.unique(depth))), "missing survival/cancellation witnesses")
    _require(np.any(odd & (tile == d["outer"])) and np.any(~odd & (tile == d["outer"])),
             "need both exchanged and preserved outer cells")
    _require(set(map(int, np.unique(out))) == {d["outer"], d["core"]}, "output loses a tile color")
    _require(np.array_equal(solve(a), out), "solve disagrees")
    d.update(depth=depth, odd=odd)
    return d


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _legal_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, details = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            d = _inspect_pair(pair)
            key = _key(pair["input"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            details[split].append(d)
    train = details["train"]
    _require(all(not all(_relations(d["rectangles"])) for d in train),
             "training must teach strict nesting and crossing separately")
    crossed = [d for d in train if len(d["rectangles"]) == 2
               and not _contains(*d["rectangles"]) and not _contains(*d["rectangles"][::-1])]
    nested = [d for d in train if len(d["rectangles"]) == 3
              and all(_contains(a, b) or _contains(b, a)
                      for i, a in enumerate(d["rectangles"]) for b in d["rectangles"][i + 1:])]
    _require(crossed and nested, "training must demonstrate crossed and nested windows")
    _require({d["n"] % 2 for d in train} == {0, 1}, "training must show both square parities")
    _require(set.union(*(d["sides"] for d in crossed)) == {"L", "R", "T", "B"},
             "teach all marker-bank directions")
    witnesses = [d for d in train if np.any(d["odd"] & (d["tile"] == d["core"]))]
    _require(witnesses, "training must exchange core as well as outer color")
    _require(any(np.any(~d["odd"] & (d["tile"] == d["core"])) for d in witnesses),
             "training must show partial exchange of a larger core")
    for alternative in ("outlines", "strict"):
        _require(any(not np.array_equal(d["odd"], _counts(d["n"], d["rectangles"],
                     **{alternative: True}) % 2 == 1) for d in train), "ambiguous interval convention")
    _require(any(not np.array_equal(d["odd"], _distance_shortcut(d)) for d in train),
             "training must distinguish color matching from rail-distance matching")
    max_depth = max(int(d["depth"].max()) for d in train)
    max_windows = max(len(d["rectangles"]) for d in train)
    for d in details["test"]:
        _require(d["core_side"] in {x["core_side"] for x in train}, "untaught core size")
        _require(int(d["depth"].max()) <= max_depth, "untaught overlap multiplicity")
        _require(len(d["rectangles"]) <= max_windows, "untaught window count")
        _require(all(_relations(d["rectangles"])) and d["depth"].max() >= 3,
                 "test must recombine containment and crossing")


def self_test():
    """Bounded structural/semantic coverage over 42 seeds and every legal count."""
    shapes, sides, cores, palettes, patterns, signatures, margins = set(), set(), set(), set(), set(), set(), set()
    window_counts, max_depths = set(), set()
    duplicate_pairs = expanded = variable_palette = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = tuple(_key(p[k]) for s in ("train", "test") for p in puzzle[s] for k in ("input", "output"))
        signatures.add(signature)
        local_palettes = set()
        for pair in puzzle["train"] + puzzle["test"]:
            d = _inspect_pair(pair)
            shapes.add(d["shape"])
            sides.add(d["n"])
            cores.add(d["core_side"])
            palettes.add(d["palette"])
            local_palettes.add(d["palette"][:3])
            top, bottom, left, right = d["box"]
            margins.add((top, d["shape"][0] - bottom - 1, left, d["shape"][1] - right - 1))
            patterns.add(_key(d["odd"]))
            window_counts.add(len(d["rectangles"]))
            max_depths.add(int(d["depth"].max()))
            duplicate_pairs += len(set(d["rectangles"])) < len(d["rectangles"])
            expanded += d["core_side"] > 2
            # Parent ablations: OR flips double coverage; copying ignores the swap.
            _require(np.any((d["depth"] > 0) != d["odd"]), "XOR ablation has no effect")
            _require(not np.array_equal(d["tile"], pair["output"]), "exchange ablation has no effect")
        variable_palette += len(local_palettes) > 1
        if seed < 3:
            again = generate_puzzle(seed)
            _require(all(np.array_equal(a[k], b[k]) for s in ("train", "test")
                         for a, b in zip(puzzle[s], again[s]) for k in ("input", "output")),
                     "seed is not deterministic")
    _require(len(signatures) == 42 and len(patterns) >= 150, "insufficient structural variation")
    _require(len(shapes) >= 50 and len(margins) >= 100, "insufficient canvas variation")
    _require(len(sides) >= 12 and cores == {1, 2, 3, 4}, "missing size/core variation")
    _require(window_counts == {2, 3, 4} and {2, 3, 4} <= max_depths, "missing overlap variation")
    _require(len(palettes) >= 80 and 10 <= variable_palette <= 38, "missing palette regimes")
    _require(duplicate_pairs >= 5 and expanded >= 10, "missing cancellation/expanded-core cases")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            generate_puzzle(10000 + 100 * a + b, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted illegal counts")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "input_shapes": len(shapes),
            "parity_patterns": len(patterns), "square_sides": sorted(sides), "core_sides": sorted(cores),
            "window_counts": sorted(window_counts), "overlap_depths": sorted(max_depths),
            "palettes": len(palettes), "margin_tuples": len(margins),
            "variable_palette_puzzles": variable_palette, "duplicate_window_pairs": duplicate_pairs,
            "expanded_core_pairs": expanded}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
