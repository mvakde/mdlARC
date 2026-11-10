"""Exact-span rectangle decoding followed by one global bottom-anchored shear.

Only NumPy and the standard library are used. Nothing is read or saved at runtime.
The sampler constructs coarse codes; solve reads only a supplied sparse grid.
"""
from __future__ import annotations

import numpy as np


AXES = (1, 2, 3, 4, 6, 8)
CANVAS = (26, 26)
GAP = 2


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}: {value}")
    return value


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= d <= 30 for d in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer grid with colors 0..9")
    return grid.astype(np.int8, copy=True)


def _decode(value):
    grid = _grid(value)
    h, w = grid.shape
    if h < 3 or w < 3 or h % 2 != 1 or w % 2 != 1:
        raise ValueError("the sparse code needs odd dimensions of at least three")
    bg = int(grid[0, 0])
    if not (np.all(grid[::2, :] == bg) and np.all(grid[:, ::2] == bg)):
        raise ValueError("the alternating scaffold must have one background color")
    coarse = grid[1::2, 1::2].copy()
    if any(d not in AXES for d in coarse.shape) or not np.any(coarse != bg):
        raise ValueError("unsupported lattice size or empty code")
    return coarse, bg


def _encode(coarse, bg):
    r, c = coarse.shape
    grid = np.full((2 * r + 1, 2 * c + 1), bg, dtype=np.int8)
    grid[1::2, 1::2] = coarse
    return grid


def _runs(row, bg, horizontal=True):
    """Triples (color, left, exclusive right), in increasing column order."""
    runs = []
    c = 0
    while c < len(row):
        color = int(row[c])
        if color == bg:
            c += 1
            continue
        end = c + 1
        if horizontal:
            while end < len(row) and int(row[end]) == color:
                end += 1
        runs.append((color, c, end))
        c = end
    return runs


def _rectangles(coarse, bg, vertical=True, horizontal=True):
    """Rectangles (top, exclusive bottom, left, exclusive right, color)."""
    rows, cols = coarse.shape
    py, px = 24 // rows, 24 // cols
    all_runs = [set(_runs(row, bg, horizontal)) for row in coarse]
    rectangles = []
    for r in range(rows):
        for color, left, right in sorted(all_runs[r], key=lambda run: run[1]):
            run = color, left, right
            if vertical and r and run in all_runs[r - 1]:
                continue
            end = r + 1
            if vertical:
                while end < rows and run in all_runs[end]:
                    end += 1
            rectangles.append((2 + r * py, end * py,
                               2 + left * px, right * px, color))
    return rectangles


def _render(coarse, bg, vertical=True, horizontal=True):
    canvas = np.full(CANVAS, bg, dtype=np.int8)
    for top, bottom, left, right, color in _rectangles(coarse, bg, vertical, horizontal):
        canvas[top:bottom, left:right] = color
    return canvas


def _shear(canvas, bg, anchor=None, rate=1, direction=-1, wrap=False):
    """Rebuild from background: no trails, no recomputation after clipping."""
    out = np.full(canvas.shape, bg, dtype=np.int8)
    ys, xs = np.nonzero(canvas != bg)
    if not len(ys):
        return out
    if anchor is None:
        anchor = int(ys.max())
    targets = xs + direction * rate * (anchor - ys)
    if wrap:
        targets %= canvas.shape[1]
    keep = (targets >= 0) & (targets < canvas.shape[1])
    out[ys[keep], targets[keep]] = canvas[ys[keep], xs[keep]]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode A's exact-span rectangles, then shear B's shared rendered field."""
    coarse, bg = _decode(input_grid)
    return _shear(_render(coarse, bg), bg)


def _bridge_render(coarse, bg, overlap=False):
    """Local cell/bridge formulation for validation, without rectangle stacks.

    overlap=True is deliberately WRONG: it joins unequal overlapping runs too.
    """
    rows, cols = coarse.shape
    py, px = 24 // rows, 24 // cols
    canvas = np.full(CANVAS, bg, dtype=np.int8)
    for r in range(rows):
        for c in range(cols):
            color = int(coarse[r, c])
            if color == bg:
                continue
            y, x = 2 + r * py, 2 + c * px
            canvas[y:y + py - 2, x:x + px - 2] = color
            if c + 1 < cols and int(coarse[r, c + 1]) == color:
                canvas[y:y + py - 2, x + px - 2:x + px] = color
    for r in range(rows - 1):
        upper, lower = _runs(coarse[r], bg), _runs(coarse[r + 1], bg)
        for ca, la, ra in upper:
            for cb, lb, rb in lower:
                same = (ca, la, ra) == (cb, lb, rb)
                if same or (overlap and ca == cb and max(la, lb) < min(ra, rb)):
                    left, right = max(la, lb), min(ra, rb)
                    y = (r + 1) * py
                    canvas[y:y + 2, 2 + left * px:right * px] = ca
    return canvas


def _separate_anchor(coarse, bg):
    """Plausible wrong model: each rectangle gets its own bottom anchor."""
    out = np.full(CANVAS, bg, dtype=np.int8)
    for top, bottom, left, right, color in _rectangles(coarse, bg):
        for y in range(top, bottom):
            lo, hi = left - (bottom - 1 - y), right - (bottom - 1 - y)
            if hi > 0:
                out[y, max(0, lo):hi] = color
    return out


def _differences(coarse, bg, out=None):
    canvas = _render(coarse, bg)
    if out is None:
        out = _shear(canvas, bg)
    alternatives = {
        "no_horizontal_merge": _shear(_render(coarse, bg, horizontal=False), bg),
        "no_exact_merge": _shear(_render(coarse, bg, vertical=False), bg),
        "overlap_merge": _shear(_bridge_render(coarse, bg, overlap=True), bg),
        "no_shear": canvas,
        "canvas_bottom": _shear(canvas, bg, anchor=25),
        "per_rectangle_bottom": _separate_anchor(coarse, bg),
        "double_rate": _shear(canvas, bg, rate=2),
        "right_shear": _shear(canvas, bg, direction=1),
        "wrap": _shear(canvas, bg, wrap=True),
    }
    return {name: int(np.count_nonzero(out != other)) for name, other in alternatives.items()}


def _features(coarse, bg):
    rows = [_runs(row, bg) for row in coarse]
    horizontal = any(right - left >= 2 for row in rows for _, left, right in row)
    exact = any(set(a) & set(b) for a, b in zip(rows, rows[1:]))
    unequal = any(ca == cb and (la, ra) != (lb, rb) and max(la, lb) < min(ra, rb)
                  for a, b in zip(rows, rows[1:])
                  for ca, la, ra in a for cb, lb, rb in b)
    return bool(horizontal), bool(exact), bool(unequal)


def _paint(coarse, r, left, right, color, bg):
    coarse[r, left:right] = color
    if left and coarse[r, left - 1] == color:
        coarse[r, left - 1] = bg
    if right < coarse.shape[1] and coarse[r, right] == color:
        coarse[r, right] = bg


def _inject(coarse, bg, colors, kind, rng):
    rows, cols = coarse.shape
    color = int(rng.choice(colors))
    if kind == "exact" and rows >= 2:
        depth = int(rng.integers(2, rows + 1))
        top = int(rng.integers(rows - depth + 1))
        length = int(rng.integers(min(2, cols), cols + 1))
        left = int(rng.integers(cols - length + 1))
        for r in range(top, top + depth):
            _paint(coarse, r, left, left + length, color, bg)
    elif kind == "unequal" and rows >= 2 and cols >= 2:
        r = int(rng.integers(rows - 1))
        spans = [(a, b) for a in range(cols) for b in range(a + 1, cols + 1)]
        first = spans[int(rng.integers(len(spans)))]
        choices = [span for span in spans if span != first
                   and max(span[0], first[0]) < min(span[1], first[1])]
        second = choices[int(rng.integers(len(choices)))]
        _paint(coarse, r, *first, color, bg)
        _paint(coarse, r + 1, *second, color, bg)
    elif kind == "horizontal" and cols >= 2:
        r = int(rng.integers(rows))
        left = int(rng.integers(cols - 1))
        right = int(rng.integers(left + 2, cols + 1))
        _paint(coarse, r, left, right, color, bg)
    elif kind == "return" and rows >= 3:
        top = int(rng.integers(rows - 2))
        end = int(rng.integers(top + 2, rows))
        left = int(rng.integers(cols))
        right = int(rng.integers(left + 1, cols + 1))
        for r in (top, end):
            _paint(coarse, r, left, right, color, bg)
        coarse[top + 1, left:right] = bg
    elif kind == "blank_row" and rows >= 2:
        coarse[int(rng.integers(rows)), :] = bg
    elif kind == "blank_column" and cols >= 2:
        coarse[:, int(rng.integers(cols))] = bg


def _sample_coarse(rng, shape, bg, role, trailing_blank):
    rows, cols = shape
    active_rows = rows - trailing_blank
    k = int(rng.integers(1, min(6, active_rows * cols) + 1))
    colors = rng.choice([c for c in range(10) if c != bg], size=k, replace=False)
    density = float(rng.choice((0.2, 0.35, 0.55, 0.8, 1.0)))
    coarse = np.full(shape, bg, dtype=np.int8)
    active = coarse[:active_rows]
    occupied = rng.random(active.shape) < density
    fill = rng.choice(colors, size=active.shape)
    active[occupied] = fill[occupied]
    kinds = ("exact", "unequal", "horizontal", "return", "blank_row", "blank_column")
    for _ in range(int(rng.integers(4))):
        _inject(active, bg, colors, str(rng.choice(kinds)), rng)
    if role != "free":
        _inject(active, bg, colors, role, rng)
    # Establish an occupied row above the requested blank suffix. This does not
    # crop or renormalize the lattice, and witness checks follow this operation.
    if not np.any(active[-1] != bg):
        active[-1, int(rng.integers(cols))] = int(rng.choice(colors))
    return coarse


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def _canonical(coarse, bg):
    labels = {bg: 0}
    values = []
    for color in coarse.flat:
        color = int(color)
        if color not in labels:
            labels[color] = len(labels)
        values.append(labels[color])
    return coarse.shape, tuple(values)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    ns = _integer(num_test, "num_test", 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    base_bg = int(rng.integers(10))
    variable_bg = bool(rng.integers(2))
    roles = ["exact", "unequal"] + ["free"] * (nt - 2)
    # One essential demonstration has trailing empty coarse rows. The other
    # reaches the last coarse row: occupied-row anchoring is therefore visible.
    blank_role = str(rng.choice(("exact", "unequal")))
    shapes = []
    for role in roles:
        pool = ([(r, c) for r in AXES if r >= 3 for c in AXES if c >= 2]
                if role != "free" else [(r, c) for r in AXES for c in AXES])
        if len(shapes) == 1:
            pool = [s for s in pool if s[0] != shapes[0][0] and s[1] != shapes[0][1]]
        shapes.append(pool[int(rng.integers(len(pool)))])
    train_specs = list(zip(shapes, roles))
    rng.shuffle(train_specs)
    held_out = [(r, c) for r in AXES for c in AXES
                if (r, c) not in shapes and (r, c) != (1, 1)]
    test_indices = rng.choice(len(held_out), size=ns, replace=False)
    test_specs = []
    for index in test_indices:
        shape = held_out[int(index)]
        role = (str(rng.choice(("exact", "unequal", "free")))
                if min(shape) >= 2 else "free")
        test_specs.append((shape, role))
    puzzle, seen = {"train": [], "test": []}, set()
    specs = [("train", shape, role) for shape, role in train_specs]
    specs += [("test", shape, role) for shape, role in test_specs]
    for split, shape, role in specs:
        bg = int(rng.integers(10)) if variable_bg else base_bg
        if split == "train" and role in ("exact", "unequal"):
            trailing = int(rng.integers(1, shape[0] - 1)) if role == blank_role else 0
        else:
            trailing = int(rng.integers(shape[0])) if rng.random() < 0.3 else 0
            if role != "free":
                trailing = min(trailing, shape[0] - 2)
        for _ in range(192):
            coarse = _sample_coarse(rng, shape, bg, role, trailing)
            inp = _encode(coarse, bg)
            if _key(inp) in seen:
                continue
            canvas = _render(coarse, bg)
            out = _shear(canvas, bg)
            if role != "free" and np.array_equal(out, canvas):
                continue
            if split == "train" and role != "free":
                diff = _differences(coarse, bg, out)
                if role == "exact" and not (diff["no_horizontal_merge"] and diff["no_exact_merge"]):
                    continue
                if role == "unequal" and not (diff["overlap_merge"] and diff["per_rectangle_bottom"]
                                               and diff["wrap"]):
                    continue
            seen.add(_key(inp))
            puzzle[split].append({"input": inp, "output": out})
            break
        else:
            raise RuntimeError(f"bounded sampling exhausted for {split}/{role}/{shape}")
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check schema plus rendered geometry, clipping, anchors and visible evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, train_shapes, anchors, evidence = set(), set(), set(), {}
    test_shapes = []
    train_features = [False, False, False]
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            coarse, bg = _decode(inp)
            assert out.shape == CANVAS
            canvas = _render(coarse, bg)
            assert np.array_equal(canvas, _bridge_render(coarse, bg))
            assert np.array_equal(out, solve(inp))
            ys, xs = np.nonzero(canvas != bg)
            anchor = int(ys.max())
            # Cell-by-cell check of the row affine map, including complete loss
            # of some upper rows and clearing of all vacated positions.
            expected = np.full(CANVAS, bg, dtype=np.int8)
            retained = 0
            for y, x in zip(ys.tolist(), xs.tolist()):
                target = x - (anchor - y)
                if target >= 0:
                    expected[y, target] = canvas[y, x]
                    retained += 1
            assert np.array_equal(out, expected)
            assert np.count_nonzero(out != bg) == retained > 0
            assert np.array_equal(out[anchor], canvas[anchor])
            assert np.all(out[anchor + 1:] == bg)
            assert np.all(out[:2] == bg) and np.all(out[:, 24:] == bg)
            assert set(np.unique(out)) <= set(np.unique(inp))
            if split == "train":
                train_shapes.add(coarse.shape)
                anchors.add(anchor)
                train_features = [a or b for a, b in zip(train_features, _features(coarse, bg))]
                for name, difference in _differences(coarse, bg, out).items():
                    evidence[name] = evidence.get(name, False) or difference > 0
            else:
                test_shapes.append(coarse.shape)
    assert len(train_shapes) >= 2 and all(s not in train_shapes for s in test_shapes)
    assert len(set(test_shapes)) == ns
    assert 23 in anchors and any(a < 23 for a in anchors)
    assert all(train_features) and all(evidence.values())


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output"))
               for s in ("train", "test"))


def self_test():
    """Bounded, streaming 42-seed audit; returns only compact aggregate facts."""
    shapes, structures, backgrounds, palettes, anchors, depths, lengths = (set() for _ in range(7))
    rectangle_counts, input_hashes = set(), set()
    stats = {"seeds": 42, "pairs": 0, "clipped_pairs": 0, "retained_pairs": 0,
             "zero_payload_pairs": 0, "empty_column_pairs": 0,
             "exact_pairs": 0, "unequal_pairs": 0, "minimum_train_puzzles": 42}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _same(puzzle, generate_puzzle(seed))
        validate_generated_puzzle(puzzle)
        small = generate_puzzle(seed, 2, 1)
        validate_generated_puzzle(small)
        input_hashes.add(tuple(_key(p["input"]) for p in puzzle["train"] + puzzle["test"]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                coarse, bg = _decode(pair["input"])
                canvas = _render(coarse, bg)
                shapes.add(coarse.shape)
                structures.add(_canonical(coarse, bg))
                backgrounds.add(bg)
                palettes.add(len(set(coarse.flat) - {bg}))
                anchors.add(int(np.nonzero(canvas != bg)[0].max()))
                rects = _rectangles(coarse, bg)
                rectangle_counts.add(len(rects))
                py, px = 24 // coarse.shape[0], 24 // coarse.shape[1]
                for top, bottom, left, right, _ in rects:
                    depths.add((bottom - top + 2) // py)
                    lengths.add((right - left + 2) // px)
                clipped = np.count_nonzero(canvas != bg) > np.count_nonzero(pair["output"] != bg)
                stats["pairs"] += 1
                stats["clipped_pairs" if clipped else "retained_pairs"] += 1
                stats["zero_payload_pairs"] += int(bg != 0 and np.any(coarse == 0))
                stats["empty_column_pairs"] += int(np.any(np.all(coarse == bg, axis=0)))
                f = _features(coarse, bg)
                stats["exact_pairs"] += int(f[1])
                stats["unequal_pairs"] += int(f[2])
    assert len(input_hashes) == 42 and len(shapes) >= 20 and len(structures) >= 150
    assert backgrounds == set(range(10)) and {1, 2, 3, 4, 5}.issubset(palettes)
    assert len(anchors) >= 6 and max(depths) >= 3 and max(lengths) >= 3
    assert min(rectangle_counts) == 1 and max(rectangle_counts) >= 15
    assert min(stats["clipped_pairs"], stats["retained_pairs"]) > 0
    assert stats["zero_payload_pairs"] >= 15 and stats["empty_column_pairs"] >= 10
    # Exhaust all legal explicit counts; keep only one puzzle live at a time.
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(4321 + 10 * nt + ns, nt, ns)
            validate_generated_puzzle(puzzle)
            assert (len(puzzle["train"]), len(puzzle["test"])) == (nt, ns)
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {nt}/{ns}")
    # Input-only coverage beyond the sampler's demonstration requirements:
    # all 36 axis combinations, entirely blank internal rows, and one label.
    for r in AXES:
        for c in AXES:
            coarse = np.full((r, c), 7, dtype=np.int8)
            coarse[-1, -1] = 0
            inp = _encode(coarse, 7)
            before = inp.copy()
            assert np.array_equal(solve(inp), _shear(_bridge_render(coarse, 7), 7))
            assert np.array_equal(inp, before)
    stats.update({"coarse_shapes": len(shapes), "normalized_layouts": len(structures),
                  "background_colors": len(backgrounds), "payload_palette_sizes": sorted(palettes),
                  "anchor_rows": sorted(anchors), "rectangle_count_range": [min(rectangle_counts), max(rectangle_counts)],
                  "max_stack_depth": max(depths), "max_run_length": max(lengths)})
    return stats
