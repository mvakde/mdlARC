"""Partition a sparse run code, reconstruct rectangles, and dock region masks.

The rule has no puzzle-specific hidden rendering parameters.  A lattice cell is
one pixel, its pitch is two, and all axes lie in separator lanes.
"""
from __future__ import annotations

import hashlib
import numpy as np


SIDES = ("top", "right", "bottom", "left")
ORIENTATIONS = tuple((s,) for s in SIDES) + tuple(
    (SIDES[i], SIDES[(i + 1) % 4]) for i in range(4))


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=True)


def _decode(value):
    grid = _grid(value)
    h, w = grid.shape
    if h < 3 or w < 3 or not (h % 2 and w % 2):
        raise ValueError("the lattice has odd dimensions at least three")
    bg = int(grid[0, 0])
    border = np.zeros(grid.shape, dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    locations = [tuple(map(int, p)) for p in np.argwhere(border & (grid != bg))]
    if len(locations) not in (1, 2):
        raise ValueError("one or two border markers are required")
    marker = int(grid[locations[0]])
    markers, ar, ac = {}, None, None
    clean = grid.copy()
    for r, c in locations:
        if int(grid[r, c]) != marker:
            raise ValueError("markers must have one common color")
        if r in (0, h - 1) and 0 < c < w - 1 and c % 2 == 0:
            side = "top" if r == 0 else "bottom"
            ac = c // 2
        elif c in (0, w - 1) and 0 < r < h - 1 and r % 2 == 0:
            side = "left" if c == 0 else "right"
            ar = r // 2
        else:
            raise ValueError("markers must point along interior separator lanes")
        markers[side] = (r, c)
        clean[r, c] = bg
    if len(markers) != len(locations) or (len(markers) == 2 and (ar is None or ac is None)):
        raise ValueError("two markers must be on adjacent sides")
    if not (np.all(clean[::2, :] == bg) and np.all(clean[:, ::2] == bg)):
        raise ValueError("only lattice intersections may contain payload")
    coarse = clean[1::2, 1::2].copy()
    if np.any(coarse == marker) or not np.any(coarse != bg):
        raise ValueError("the marker color is reserved and payload must be nonempty")
    return grid, coarse, bg, marker, markers, ar, ac


def _regions(coarse, markers, ar, ac):
    """Return (row start/end, column start/end, corner, retained)."""
    rows, cols = coarse.shape
    if ar is None:
        side = next(iter(markers))
        return [(0, rows, 0, ac, (side, "left"), True),
                (0, rows, ac, cols, (side, "right"), True)]
    if ac is None:
        side = next(iter(markers))
        return [(0, ar, 0, cols, ("top", side), True),
                (ar, rows, 0, cols, ("bottom", side), True)]
    return [(r0, r1, c0, c1, (v, s), v in markers or s in markers)
            for r0, r1, v in ((0, ar, "top"), (ar, rows, "bottom"))
            for c0, c1, s in ((0, ac, "left"), (ac, cols, "right"))]


def _runs(row, bg):
    result, left = [], 0
    while left < len(row):
        color = int(row[left])
        if color == bg:
            left += 1
            continue
        right = left + 1
        while right < len(row) and int(row[right]) == color:
            right += 1
        result.append((color, left, right))
        left = right
    return result


def _rectangles(coarse, bg, vertical=True):
    runs = [_runs(row, bg) for row in coarse]
    result = []
    for r, row in enumerate(runs):
        for color, left, right in row:
            key = (color, left, right)
            if vertical and r and key in runs[r - 1]:
                continue
            bottom = r + 1
            while vertical and bottom < len(runs) and key in runs[bottom]:
                bottom += 1
            result.append((color, r, bottom, left, right))
    return result


def _render(coarse, bg, mode="exact"):
    out = np.full((2 * coarse.shape[0] - 1, 2 * coarse.shape[1] - 1), bg, dtype=np.int8)
    for color, top, bottom, left, right in _rectangles(coarse, bg, mode != "no_vertical"):
        out[2 * top:2 * bottom - 1, 2 * left:2 * right - 1] = color
    if mode == "overlap":
        # Deliberately wrong diagnostic: connect equal cells in adjacent rows.
        for r in range(coarse.shape[0] - 1):
            for a, l, u in _runs(coarse[r], bg):
                for b, x, y in _runs(coarse[r + 1], bg):
                    if a == b and max(l, x) < min(u, y):
                        out[2 * r + 1, 2 * max(l, x):2 * min(u, y) - 1] = a
    return out


def _transform(input_grid, mode="exact"):
    grid, coarse, bg, marker, markers, ar, ac = _decode(input_grid)
    out = np.full_like(grid, bg)
    for position in markers.values():
        out[position] = marker
    global_mask = _render(coarse, bg) if mode == "global" else None
    for r0, r1, c0, c1, corner, keep in _regions(coarse, markers, ar, ac):
        if not keep and mode != "retain_all":
            continue
        mask = (global_mask[2 * r0:2 * r1 - 1, 2 * c0:2 * c1 - 1]
                if global_mask is not None else _render(coarse[r0:r1, c0:c1], bg, mode))
        rr, cc = np.nonzero(mask != bg)
        if not len(rr):
            continue
        if mode == "source":
            y, x, cut = 2 * r0 + 1, 2 * c0 + 1, mask
        else:
            cut = mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
            y = 0 if corner[0] == "top" else grid.shape[0] - cut.shape[0]
            x = 0 if corner[1] == "left" else grid.shape[1] - cut.shape[1]
        target = out[y:y + cut.shape[0], x:x + cut.shape[1]]
        occupied = cut != bg
        if target.shape != cut.shape or np.any(target[occupied] != bg):
            raise ValueError("corner translation must fit without collisions")
        target[occupied] = cut[occupied]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from visible markers, lattice values, and the fixed family rule."""
    return _transform(input_grid)


def _paint(a, row, left, right, color, bg):
    a[row, left:right] = color
    if left and a[row, left - 1] == color:
        a[row, left - 1] = bg
    if right < a.shape[1] and a[row, right] == color:
        a[row, right] = bg


def _unequal_spans(width, rng):
    left = int(rng.integers(width - 1))
    right = int(rng.integers(left + 2, width + 1))
    others = [(l, r) for l in range(width) for r in range(l + 1, width + 1)
              if (l, r) != (left, right) and max(l, left) < min(r, right)]
    return (left, right), others[int(rng.integers(len(others)))]


def _basic_witness(coarse, regions, colors, bg, rng, deep=False):
    choices = [s for s in regions if s[-1] and s[1] - s[0] >= 3 and s[3] - s[2] >= 2]
    r0, r1, c0, c1, _, _ = choices[int(rng.integers(len(choices)))]
    a = coarse[r0:r1, c0:c1]
    depth = 3 if deep and a.shape[0] >= 4 else 2
    top = int(rng.integers(a.shape[0] - depth))
    first, second = _unequal_spans(a.shape[1], rng)
    color = int(rng.choice(colors))
    for r in range(top, top + depth):
        _paint(a, r, *first, color, bg)
    _paint(a, top + depth, *second, color, bg)


def _cross_witness(coarse, markers, ar, ac, colors, bg, rng):
    # Both corners along the marked vertical side are retained.  Cut first:
    # the left halves match, although the complete crossing runs do not.
    top_side = "top" in markers
    r0, r1 = (0, ar) if top_side else (ar, coarse.shape[0])
    top = int(rng.integers(r0, r1 - 1))
    left = int(rng.integers(ac))
    ends = rng.choice(np.arange(ac + 1, coarse.shape[1] + 1), 2, replace=False)
    color = int(rng.choice(colors))
    for r, end in zip((top, top + 1), ends):
        _paint(coarse, r, left, int(end), color, bg)


def _make_pair(rng, shape, orientation, roles, role):
    rows, cols = shape
    bg, marker, colors = roles
    ar = int(rng.integers(2, rows - 1)) if any(s in ("left", "right") for s in orientation) else None
    ac = int(rng.integers(2, cols - 1)) if any(s in ("top", "bottom") for s in orientation) else None
    h, w = 2 * rows + 1, 2 * cols + 1
    markers = {s: ((0, 2 * ac) if s == "top" else (h - 1, 2 * ac) if s == "bottom"
                   else (2 * ar, 0) if s == "left" else (2 * ar, w - 1)) for s in orientation}
    density = float(rng.choice((.18, .32, .50, .72, .92)))
    coarse = np.full(shape, bg, dtype=np.int8)
    occupied = rng.random(shape) < density
    values = rng.choice(colors, shape)
    coarse[occupied] = values[occupied]
    regions = _regions(coarse, markers, ar, ac)
    # Random run operations occur before mandatory evidence, so cannot erase it.
    for _ in range(int(rng.integers(4))):
        r0, r1, c0, c1, _, _ = regions[int(rng.integers(len(regions)))]
        a = coarse[r0:r1, c0:c1]
        left = int(rng.integers(a.shape[1]))
        right = int(rng.integers(left + 1, a.shape[1] + 1))
        top = int(rng.integers(a.shape[0]))
        depth = int(rng.integers(1, a.shape[0] - top + 1))
        color = int(rng.choice(colors))
        for row in range(top, top + depth):
            _paint(a, row, left, right, color, bg)
    if role == "cross":
        _cross_witness(coarse, markers, ar, ac, colors, bg, rng)
    elif role in ("basic", "test"):
        _basic_witness(coarse, regions, colors, bg, rng, role == "test")
    for r0, r1, c0, c1, _, _ in regions:
        a = coarse[r0:r1, c0:c1]
        if not np.any(a != bg):
            a[int(rng.integers(a.shape[0])), int(rng.integers(a.shape[1]))] = int(rng.choice(colors))
    grid = np.full((h, w), bg, dtype=np.int8)
    grid[1::2, 1::2] = coarse
    for p in markers.values():
        grid[p] = marker
    return {"input": grid, "output": solve(grid)}


def _roles(rng):
    palette = [int(c) for c in rng.permutation(10)]
    count = int(rng.integers(1, 6))
    return palette[0], palette[1], palette[2:2 + count]


def _key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    schedule = [(int(rng.integers(4)), "basic"), (int(rng.integers(4, 8)), "cross")]
    schedule += [(int(rng.integers(8)), "free") for _ in range(num_train - 2)]
    roles, consistent = _roles(rng), bool(rng.random() < .3)
    puzzle, seen, training_shapes = {"train": [], "test": []}, set(), set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            orientation, role = schedule[index] if split == "train" else (int(rng.integers(8)), "test")
            if not consistent:
                roles = _roles(rng)
            pool = [(r, c) for r in range(6, 12 if split == "train" else 15)
                    for c in range(6, 12 if split == "train" else 15)
                    if split == "train" or (r, c) not in training_shapes]
            shape = pool[int(rng.integers(len(pool)))]
            for _ in range(32):
                pair = _make_pair(rng, shape, ORIENTATIONS[orientation], roles, role)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == "train":
                        training_shapes.add(shape)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    order = rng.permutation(num_train)
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]
    return puzzle


def _features(input_grid):
    grid, coarse, bg, marker, markers, ar, ac = _decode(input_grid)
    f = {"horizontal": False, "exact": False, "unequal": False, "deleted": 0,
         "rectangles": 0, "max_depth": 0, "holes": bool(np.any(coarse == bg)),
         "orientation": tuple(s for s in SIDES if s in markers), "background": bg,
         "palette_size": len(set(map(int, coarse.flat)) - {bg}), "shape": coarse.shape}
    for r0, r1, c0, c1, _, keep in _regions(coarse, markers, ar, ac):
        a = coarse[r0:r1, c0:c1]
        if not keep:
            f["deleted"] += int(np.count_nonzero(a != bg))
            continue
        runs = [_runs(row, bg) for row in a]
        rects = _rectangles(a, bg)
        f["rectangles"] += len(rects)
        f["max_depth"] = max(f["max_depth"], max((b - t for _, t, b, _, _ in rects), default=0))
        f["horizontal"] |= any(r - l > 1 for row in runs for _, l, r in row)
        f["exact"] |= any(bool(set(a) & set(b)) for a, b in zip(runs, runs[1:]))
        f["unequal"] |= any(a == b and (l, r) != (x, y) and max(l, x) < min(r, y)
                            for upper, lower in zip(runs, runs[1:])
                            for a, l, r in upper for b, x, y in lower)
    return f


def validate_generated_puzzle(puzzle):
    assert type(puzzle) is dict and set(puzzle) == {"train", "test"}
    assert all(type(puzzle[s]) is list for s in ("train", "test"))
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, features = set(), {"train": [], "test": []}
    witnesses = dict.fromkeys(("no_vertical", "overlap", "global", "source", "retain_all"), False)
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert type(pair) is dict and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            assert np.array_equal(solve(inp), out)
            assert not np.array_equal(inp, out)
            grid, coarse, bg, marker, markers, ar, ac = _decode(inp)
            total = len(markers)
            for r0, r1, c0, c1, corner, keep in _regions(coarse, markers, ar, ac):
                local = coarse[r0:r1, c0:c1]
                assert np.any(local != bg), "generated source regions are nonempty"
                if not keep:
                    continue
                rects = _rectangles(local, bg)
                total += sum((2 * (b - t) - 1) * (2 * (r - l) - 1)
                             for _, t, b, l, r in rects)
                mask = _render(local, bg)
                rr, cc = np.nonzero(mask != bg)
                height, width = int(rr.max() - rr.min() + 1), int(cc.max() - cc.min() + 1)
                y = 0 if corner[0] == "top" else out.shape[0] - height
                x = 0 if corner[1] == "left" else out.shape[1] - width
                target = out[y:y + height, x:x + width]
                assert np.any(target[0] != bg) and np.any(target[-1] != bg)
                assert np.any(target[:, 0] != bg) and np.any(target[:, -1] != bg)
            assert np.count_nonzero(out != bg) == total, "area conservation after decoding/deletion"
            assert np.count_nonzero(out == marker) == len(markers)
            assert all(int(out[p]) == marker for p in markers.values())
            assert set(map(int, out.flat)) <= set(map(int, inp.flat))
            features[split].append(_features(inp))
            if split == "train":
                for mode in witnesses:
                    witnesses[mode] |= not np.array_equal(_transform(inp, mode), out)
    assert all(witnesses.values()), f"missing training distinction: {witnesses}"
    train = features["train"]
    assert all(any(f[k] for f in train) for k in ("horizontal", "exact", "unequal", "deleted"))
    assert {len(f["orientation"]) for f in train} == {1, 2}
    assert all(f["shape"] not in {t["shape"] for t in train} for f in features["test"])


def self_test():
    """Bounded, in-memory checks; diagnostics are aggregate values, never traces."""
    # Hand-specified regressions isolate span equality, interruptions, and cuts.
    assert np.array_equal(_render(np.array([[2, 2, 0], [2, 2, 0], [0, 2, 2]]), 0),
                          np.array([[2, 2, 2, 0, 0], [2, 2, 2, 0, 0],
                                    [2, 2, 2, 0, 0], [0, 0, 0, 0, 0], [0, 0, 2, 2, 2]]))
    assert np.array_equal(_render(np.array([[2, 2], [0, 0], [2, 2]]), 0),
                          np.array([[2, 2, 2], [0, 0, 0], [0, 0, 0], [0, 0, 0], [2, 2, 2]]))
    assert np.array_equal(_render(np.array([[2], [3]]), 0), np.array([[2], [0], [3]]))
    crossing = np.array([[int(c) for c in row] for row in (
        "000070000", "020202000", "000000000", "020202020", "000000000")])
    expected = np.array([[int(c) for c in row] for row in (
        "222070200", "222000000", "222000222", "000000000", "000000000")])
    assert np.array_equal(solve(crossing), expected)
    assert not np.array_equal(_transform(crossing, "global"), expected)
    shapes, layouts, backgrounds, palettes, signatures = set(), set(), set(), set(), set()
    orientations = {"train": set(), "test": set()}
    rectangle_counts, depths, holes = set(), set(), 0
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                f = _features(inp)
                pairs += 1
                digest.update(bytes(inp.shape) + inp.tobytes() + out.tobytes())
                shapes.add(inp.shape)
                backgrounds.add(f["background"])
                palettes.add(f["palette_size"])
                orientations[split].add(f["orientation"])
                rectangle_counts.add(f["rectangles"])
                depths.add(f["max_depth"])
                holes += f["holes"]
                _, coarse, bg, marker, _, _, _ = _decode(inp)
                mapping = {bg: 0, marker: 1}
                normalized = []
                for c in coarse.flat:
                    c = int(c)
                    if c not in mapping:
                        mapping[c] = len(mapping)
                    normalized.append(mapping[c])
                layouts.add((coarse.shape, bytes(normalized)))
        signatures.add(digest.digest())
    assert len(signatures) == 42 and len(layouts) >= 180
    assert len(shapes) >= 40 and len(backgrounds) == 10 and len(palettes) >= 4
    assert all(len(v) == 8 for v in orientations.values())
    assert len(rectangle_counts) >= 20 and max(depths) >= 4 and holes >= 100
    a, b = generate_puzzle(7123, 7, 3), generate_puzzle(7123, 7, 3)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s])
               for k in ("input", "output"))
    permutation = np.array([8, 3, 0, 7, 2, 9, 1, 6, 4, 5], dtype=np.int8)
    for pair in a["train"] + a["test"]:
        assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(9000 + 10 * nt + ns, nt, ns)
            validate_generated_puzzle(puzzle)
            assert (len(puzzle["train"]), len(puzzle["test"])) == (nt, ns)
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Small legal family members outside the sampler's demonstration sizes.
    for rows, cols, side in ((1, 2, "top"), (2, 1, "left"), (2, 2, "bottom")):
        tiny = np.zeros((2 * rows + 1, 2 * cols + 1), dtype=np.int8)
        tiny[1::2, 1::2] = 3
        position = (0, 2) if side == "top" else (2, 0) if side == "left" else (2 * rows, 2)
        tiny[position] = 7
        assert solve(tiny).shape == tiny.shape
    return {"seeds": 42, "pairs": pairs, "unique_puzzles": len(signatures),
            "input_shapes": len(shapes), "normalized_coarse_layouts": len(layouts),
            "backgrounds": len(backgrounds), "payload_palette_sizes": sorted(palettes),
            "orientations_per_split": {s: len(v) for s, v in orientations.items()},
            "rectangle_count_range": [min(rectangle_counts), max(rectangle_counts)],
            "maximum_stack_depth": max(depths), "pairs_with_holes": holes}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
