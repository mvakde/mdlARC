"""Exact-run contour reconstruction followed by a visible, directed cavity key.

The top two rows are an unchanged source/fill key. Row 2 is blank. The
remaining odd-sized panel is an every-other-cell lattice. Its output panel
uses pitch 4, gap 1, and hollow one-cell rectangle boundaries.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import numpy as np


HOST_SEEDS = (0, 562352796, 367715133, 552375974, 954306458, 675318675,
              265576648, 901884624, 196765925, 493964656, 660083460,
              523122666, 530626741, 46662226, 466497116, 953136233,
              423058654, 754422694, 844844630, 753479991, 994668098,
              25400865, 533555332, 212049950, 595392379, 921689505,
              793981256, 356528073, 237639500, 64413582, 529590698,
              745550662, 723591399, 728279859, 947031276, 565490925,
              879889890, 787539978, 23546777, 51005771, 263147992,
              687896524)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid of size 1..30 and colors 0..9")
    return a.astype(np.int8, copy=True)


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


def _rectangles(coarse, bg, merge=True):
    """Return (color, top, bottom_exclusive, left, right_exclusive)."""
    rows = [set(_runs(row, bg)) for row in coarse]
    rectangles = []
    for r, runs in enumerate(rows):
        for color, left, right in sorted(runs):
            run = (color, left, right)
            if merge and r and run in rows[r - 1]:
                continue
            end = r + 1
            while merge and end < len(rows) and run in rows[end]:
                end += 1
            rectangles.append((color, r, end, left, right))
    return rectangles


def _decode(grid):
    h, w = grid.shape
    if h < 6 or h % 2 or w < 3 or not w % 2:
        raise ValueError("input must have a two-row key, separator, and odd lattice panel")
    rows, cols = (h - 4) // 2, (w - 1) // 2
    if not (1 <= rows <= 6 and 1 <= cols <= 7):
        raise ValueError("coarse dimensions must be 1..6 by 1..7")
    bg = int(grid[3, 0])
    if (np.any(grid[2] != bg) or np.any(grid[3::2] != bg)
            or np.any(grid[3:, ::2] != bg)):
        raise ValueError("broken background scaffold")
    positions = np.flatnonzero(grid[0] != bg)
    if (not 2 <= len(positions) <= 7
            or np.any(np.diff(positions) != 1)
            or not np.array_equal(grid[0] != bg, grid[1] != bg)):
        raise ValueError("key must be one dense two-lane top-edge strip")
    sources = list(map(int, grid[0, positions]))
    fills = list(map(int, grid[1, positions]))
    if len(set(sources)) != len(sources):
        raise ValueError("key sources must be unique")
    coarse = grid[4::2, 1::2].copy()
    if not np.any(coarse != bg):
        raise ValueError("empty lattice")
    return coarse, bg, dict(zip(sources, fills)), positions


def _encode(coarse, bg, sources, fills, start):
    rows, cols = coarse.shape
    grid = np.full((2 * rows + 4, 2 * cols + 1), bg, dtype=np.int8)
    grid[4::2, 1::2] = coarse
    grid[0, start:start + len(sources)] = sources
    grid[1, start:start + len(sources)] = fills
    return grid


def _bounds(rectangle):
    color, top, bottom, left, right = rectangle
    return color, 4 + 4 * top, 3 + 4 * bottom, 1 + 4 * left, 4 * right


def _render(grid, coarse, bg, rectangles):
    rows, cols = coarse.shape
    out = np.full((4 * rows + 4, 4 * cols + 1), bg, dtype=np.int8)
    out[:2, :grid.shape[1]] = grid[:2]
    for rectangle in rectangles:
        color, y0, y1, x0, x1 = _bounds(rectangle)
        out[y0, x0:x1] = out[y1 - 1, x0:x1] = color
        out[y0:y1, x0] = out[y0:y1, x1 - 1] = color
    return out


def _cavities(grid, bg):
    """Actual four-connected background components, with their boundary colors."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    enclosed = []
    for y in range(h):
        for x in range(w):
            if seen[y, x] or int(grid[y, x]) != bg:
                continue
            seen[y, x] = True
            pending, cells, boundary, edge = [(y, x)], [], set(), False
            while pending:
                r, c = pending.pop()
                cells.append((r, c))
                edge |= r in (0, h - 1) or c in (0, w - 1)
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if not (0 <= rr < h and 0 <= cc < w):
                        continue
                    color = int(grid[rr, cc])
                    if color != bg:
                        boundary.add(color)
                    elif not seen[rr, cc]:
                        seen[rr, cc] = True
                        pending.append((rr, cc))
            if not edge:
                if len(boundary) != 1:
                    raise ValueError("cavity does not have one contour owner")
                enclosed.append((next(iter(boundary)), cells))
    return enclosed


def _fill(contours, bg, mapping):
    out = contours.copy()
    # Read every owner from the unfilled rendering: exactly one lookup.
    for color, cells in _cavities(contours, bg):
        if color in mapping:
            rr, cc = zip(*cells)
            out[rr, cc] = mapping[color]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    coarse, bg, mapping, _ = _decode(grid)
    contours = _render(grid, coarse, bg, _rectangles(coarse, bg))
    return _fill(contours, bg, mapping)


def _box_answer(grid, coarse, bg, mapping, rectangles=None):
    """Analytic check/counterfactual renderer; valid boxes have disjoint interiors."""
    if rectangles is None:
        rectangles = _rectangles(coarse, bg)
    out = _render(grid, coarse, bg, rectangles)
    for rectangle in rectangles:
        color, y0, y1, x0, x1 = _bounds(rectangle)
        if color in mapping:
            out[y0 + 1:y1 - 1, x0 + 1:x1 - 1] = mapping[color]
    return out


def _loose_rectangles(coarse, bg):
    """Deliberately wrong ablation: bounding boxes of touching same-color cells."""
    rows, cols = coarse.shape
    seen, rectangles = set(), []
    for r in range(rows):
        for c in range(cols):
            color = int(coarse[r, c])
            if color == bg or (r, c) in seen:
                continue
            seen.add((r, c))
            pending, cells = [(r, c)], []
            while pending:
                y, x = pending.pop()
                cells.append((y, x))
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (0 <= yy < rows and 0 <= xx < cols and (yy, xx) not in seen
                            and int(coarse[yy, xx]) == color):
                        seen.add((yy, xx))
                        pending.append((yy, xx))
            yy, xx = zip(*cells)
            rectangles.append((color, min(yy), max(yy) + 1, min(xx), max(xx) + 1))
    return rectangles


def _features(grid):
    coarse, bg, mapping, positions = _decode(grid)
    rectangles = _rectangles(coarse, bg)
    runs = [_runs(row, bg) for row in coarse]
    active = [q for q in rectangles if q[0] in mapping]
    return {
        "shape": coarse.shape, "background": bg, "key_length": len(mapping),
        "key_start": int(positions[0]), "rectangles": len(rectangles),
        "max_depth": max(q[2] - q[1] for q in rectangles),
        "max_width": max(q[4] - q[3] for q in rectangles),
        "active_depth": max((q[2] - q[1] for q in active), default=0),
        "active_width": max((q[4] - q[3] for q in active), default=0),
        "horizontal": any(q[4] - q[3] > 1 for q in active),
        "exact": any(q[2] - q[1] > 1 for q in active),
        "unequal": any(a == b and a in mapping and (l, r) != (s, t)
                       and max(l, s) < min(r, t)
                       for up, down in zip(runs, runs[1:])
                       for a, l, r in up for b, s, t in down),
        "chain": any(mapping[q[0]] in mapping
                     and mapping[mapping[q[0]]] != mapping[q[0]] for q in active),
        "self_map": any(mapping[q[0]] == q[0] for q in active),
        "direction": any(mapping[q[0]] != q[0] and q[0] not in mapping.values()
                         for q in active),
        "fill_only_inert": any(q[0] in set(mapping.values()) - set(mapping) for q in rectangles),
        "absent_inert": any(q[0] not in set(mapping) | set(mapping.values()) for q in rectangles),
        "active_count": len(active), "inert_count": len(rectangles) - len(active),
        "empty_row": any(np.all(row == bg) for row in coarse),
        "empty_column": any(np.all(col == bg) for col in coarse.T),
        "zero_payload": bg != 0 and np.any(coarse == 0),
    }


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _paint(coarse, row, left, right, color, bg):
    coarse[row, left:right] = color
    if left and int(coarse[row, left - 1]) == color:
        coarse[row, left - 1] = bg
    if right < coarse.shape[1] and int(coarse[row, right]) == color:
        coarse[row, right] = bg


def _mapping(rng, bg, role, max_sources):
    colors = list(map(int, rng.permutation([c for c in range(10) if c != bg])))
    n = int(rng.integers(3 if role == "exact" else 2, max_sources + 1))
    sources, outside = colors[:n], colors[n:]
    # Keep one color absent from both lanes. Source 0 never appears as a fill,
    # supplying an unambiguous reversal counterexample.
    allowed = [c for c in colors if c not in (sources[0], outside[1])]
    fills = list(map(int, rng.choice(allowed, n)))
    if role == "exact":
        fills[0], fills[1], fills[2] = sources[1], outside[0], sources[2]
    else:
        fills[0] = outside[0]
    if role == "random" and rng.random() < .4:
        fills = [outside[0]] * n
    return sources, fills, outside


def _make_pair(rng, shift, role):
    is_test = role == "test"
    min_rows = 3 if is_test and shift == "depth" else 2 if role in ("exact", "unequal") else 1
    min_cols = 3 if role in ("exact", "unequal") or (is_test and shift == "width") else 1
    rows = int(rng.integers(min_rows, 7))
    cols = int(rng.integers(min_cols, 8))
    bg = int(rng.integers(10))
    sources, fills, outside = _mapping(rng, bg, "random" if is_test else role, min(7, 2 * cols + 1))
    primary = sources[0]
    width_limit = 2 if not is_test and shift == "width" else cols
    depth_limit = 2 if not is_test and shift == "depth" else rows
    values = [bg] + sources + outside[:2]
    for attempt in range(32):
        # A sampled run grammar supplies occupancy, blanks, repeated colors,
        # deep stacks and sparse/dense scenes, rather than a fixed whole-grid template.
        coarse = np.full((rows, cols), bg, dtype=np.int8)
        if attempt < 24:
            density = float(rng.uniform(.2, .8))
            for r in range(rows):
                if r and rng.random() < .25:
                    coarse[r] = coarse[r - 1]
                    continue
                c = 0
                while c < cols:
                    length = int(rng.integers(1, min(width_limit, cols - c) + 1))
                    color = int(rng.choice(values[1:])) if rng.random() < density else bg
                    coarse[r, c:c + length] = color
                    c += length
        protected = np.zeros(coarse.shape, dtype=bool)
        if role == "unequal":
            spans = [(l, r) for l in range(cols) for r in range(l + 1, cols + 1)
                     if r - l <= width_limit]
            choices = [(a, b) for a in spans for b in spans
                       if a != b and max(a[0], b[0]) < min(a[1], b[1])]
            first, second = choices[int(rng.integers(len(choices)))]
            top = int(rng.integers(rows - 1))
            for row, (left, right) in zip((top, top + 1), (first, second)):
                _paint(coarse, row, left, right, primary, bg)
                protected[row, left:right] = True
            extras = [outside[1]]
        elif role == "exact" or is_test:
            if is_test and shift == "depth":
                depth = int(rng.integers(3, rows + 1))
                width = int(rng.integers(1, cols + 1))
            elif is_test:
                depth = int(rng.integers(1, rows + 1))
                width = int(rng.integers(3, cols + 1))
            else:
                depth = int(rng.integers(2, depth_limit + 1))
                width = int(rng.integers(2, min(width_limit, cols - 1) + 1))
            top = int(rng.integers(rows - depth + 1))
            left = int(rng.integers(cols - width + 1))
            for row in range(top, top + depth):
                _paint(coarse, row, left, left + width, primary, bg)
                protected[row, left:left + width] = True
            extras = [sources[2], outside[0]] if role == "exact" else []
        else:
            extras = [primary]
        # Put witnesses outside the protected run spans. Different colors cannot
        # extend those spans, even when the available slots are immediately adjacent.
        free = np.argwhere(~protected)
        if len(free) < len(extras):
            continue
        for index, color in zip(rng.permutation(len(free)), extras):
            r, c = free[int(index)]
            coarse[r, c] = color
        # Random-role pairs must also contain at least one nontrivial mapped cavity.
        if role == "random" and not np.any(coarse == primary):
            coarse[int(rng.integers(rows)), int(rng.integers(cols))] = primary
        order = rng.permutation(len(sources))
        source_lane = [sources[int(i)] for i in order]
        fill_lane = [fills[int(i)] for i in order]
        start = int(rng.integers(2 * cols + 2 - len(sources)))
        grid = _encode(coarse, bg, source_lane, fill_lane, start)
        f = _features(grid)
        if not is_test and (f["max_depth"] > depth_limit or f["max_width"] > width_limit):
            continue
        if role == "exact" and not (f["exact"] and f["horizontal"] and f["chain"]
                                    and f["self_map"] and f["fill_only_inert"]):
            continue
        if role == "unequal" and not (f["unequal"] and f["absent_inert"]):
            continue
        if is_test and f["active_depth" if shift == "depth" else "active_width"] < 3:
            continue
        if not f["direction"]:
            continue
        return {"input": grid, "output": solve(grid)}
    raise RuntimeError("bounded scene construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    shift = "depth" if int(rng.integers(2)) else "width"
    roles = ["exact", "unequal"] + ["random"] * (num_train - 2)
    roles = [roles[int(i)] for i in rng.permutation(num_train)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, plans in (("train", roles), ("test", ["test"] * num_test)):
        for role in plans:
            for _ in range(12):
                pair = _make_pair(rng, shift, role)
                signature = (pair["input"].shape, pair["input"].tobytes())
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded uniqueness attempts exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, records = set(), {"train": [], "test": []}
    ablations = Counter()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, out = _grid(pair["input"]), _grid(pair["output"])
            key = (grid.shape, grid.tobytes())
            assert key not in seen, "duplicate input"
            seen.add(key)
            coarse, bg, mapping, positions = _decode(grid)
            rectangles = _rectangles(coarse, bg)
            contours = _render(grid, coarse, bg, rectangles)
            assert out.shape == contours.shape
            assert np.array_equal(out, solve(grid)), "solve disagrees"
            assert np.array_equal(out, _box_answer(grid, coarse, bg, mapping)), "box/cavity check disagrees"
            assert np.array_equal(out[:2, :grid.shape[1]], grid[:2]), "key changed"
            assert np.all(out[:2, grid.shape[1]:] == bg)
            assert np.all(out[2:4] == bg), "separator changed"
            assert np.all(out[contours != bg] == contours[contours != bg]), "contour/key ink changed"
            assert np.any(out != contours), "no mapped fill"
            # Each reconstructed rectangle is separated, and owns exactly one cavity.
            occupied = np.zeros(out.shape, dtype=bool)
            for rectangle in rectangles:
                color, y0, y1, x0, x1 = _bounds(rectangle)
                assert not np.any(occupied[y0:y1, x0:x1]), "overlapping boxes"
                occupied[y0:y1, x0:x1] = True
                assert np.all(out[y0 + 1:y1 - 1, x0 + 1:x1 - 1] == mapping.get(color, bg))
            cavities = _cavities(contours, bg)
            assert len(cavities) == len(rectangles), "unexpected cavity ownership"
            assert sum(len(cells) for _, cells in cavities) == sum(
                (4 * (q[2] - q[1]) - 3) * (4 * (q[4] - q[3]) - 3) for q in rectangles)
            f = _features(grid)
            records[split].append(f)
            if split == "train":
                atomic = [(int(coarse[r, c]), r, r + 1, c, c + 1)
                          for r in range(coarse.shape[0]) for c in range(coarse.shape[1])
                          if int(coarse[r, c]) != bg]
                alternatives = {
                    "no_run_grouping": _box_answer(grid, coarse, bg, mapping, atomic),
                    "no_exact_merge": _box_answer(grid, coarse, bg, mapping, _rectangles(coarse, bg, False)),
                    "loose_merge": _box_answer(grid, coarse, bg, mapping, _loose_rectangles(coarse, bg)),
                    "no_key_fill": contours,
                    "reverse_key": _box_answer(grid, coarse, bg, {v: k for k, v in mapping.items()}),
                    "own_color_fill": _box_answer(grid, coarse, bg, {q[0]: q[0] for q in rectangles}),
                }
                for name, wrong in alternatives.items():
                    ablations[name] += not np.array_equal(out, wrong)
                    if name in ("no_exact_merge", "loose_merge"):
                        wrong_rects = (_rectangles(coarse, bg, False) if name == "no_exact_merge"
                                       else _loose_rectangles(coarse, bg))
                        wrong_contours = _render(grid, coarse, bg, wrong_rects)
                        fill_either = (out != contours) | (wrong != wrong_contours)
                        ablations[name + "_fill"] += bool(np.any(fill_either & (out != wrong)))
    train, test = records["train"], records["test"]
    for feature in ("horizontal", "exact", "unequal", "chain", "self_map", "direction",
                    "fill_only_inert", "absent_inert"):
        assert any(f[feature] for f in train), "missing training witness: " + feature
    assert all(ablations[name] for name in (
        "no_run_grouping", "no_exact_merge", "loose_merge", "no_key_fill", "reverse_key", "own_color_fill",
        "no_exact_merge_fill", "loose_merge_fill"))
    assert ((max(f["max_depth"] for f in train) <= 2 and min(f["active_depth"] for f in test) >= 3)
            or (max(f["max_width"] for f in train) <= 2 and min(f["active_width"] for f in test) >= 3)), "missing structural shift"


def _puzzle_hash(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(grid.shape))
                h.update(grid.tobytes())
    return h.hexdigest()


def _unit_checks():
    # Hand-computed exact stack beside an unequal overlap. A->B->D is one lookup.
    coarse = np.array([[1, 1, 0, 4], [1, 1, 1, 4], [0, 2, 2, 4], [0, 2, 2, 0]], dtype=np.int8)
    grid = _encode(coarse, 0, [1, 2, 3], [2, 5, 3], 1)
    answer = solve(grid)
    assert answer.shape == (20, 17)
    # The unequal color-1 rows remain separate, with their intervening gap blank.
    assert answer[5, 2] == 2 and answer[9, 2] == 2 and answer[7, 2] == 0
    # Color 2 merges across two rows: the old inter-row gap is now cavity fill.
    assert answer[15, 6] == 5 and answer[15, 5] == 2
    # The unlisted color 4 has a tall boundary, but an untouched interior.
    assert answer[5, 14] == 0 and answer[7, 13] == 4
    assert np.array_equal(grid[:2], answer[:2, :grid.shape[1]])
    # Smallest and largest legal lattice axes, including blank margins, are soluble.
    for shape in ((1, 1), (1, 7), (6, 1), (6, 7)):
        a = np.full(shape, 7, dtype=np.int8)
        a[-1, -1] = 0  # Black payload on a nonblack background.
        x = _encode(a, 7, [0, 1], [1, 0], 0)
        assert np.array_equal(solve(x), _box_answer(x, a, 7, {0: 1, 1: 0}))
    # Flood semantics itself: an opening to the exterior must leak.
    open_ring = np.zeros((7, 9), dtype=np.int8)
    open_ring[1:6, 1] = open_ring[1:6, 7] = 1
    open_ring[1, 1:8] = open_ring[5, 1:8] = 1
    assert np.all(_fill(open_ring, 0, {1: 2})[2:5, 2:7] == 2)
    open_ring[1, 4] = 0
    assert np.array_equal(_fill(open_ring, 0, {1: 2}), open_ring)


def self_test():
    """42 specified seeds, streamed; no puzzle payloads or logs are saved."""
    _unit_checks()
    puzzle_hashes, structures, shapes, backgrounds, key_lengths, counts = set(), set(), set(), set(), set(), set()
    measurements = Counter()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = _puzzle_hash(puzzle)
        assert digest == _puzzle_hash(generate_puzzle(seed)), "nondeterminism"
        puzzle_hashes.add(digest)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, bg, _, _ = _decode(pair["input"])
                f = _features(pair["input"])
                roles = {bg: 0}
                normalized = tuple(roles.setdefault(int(v), len(roles)) for v in a.flat)
                structures.add((a.shape, normalized))
                shapes.add(a.shape)
                backgrounds.add(bg)
                key_lengths.add(f["key_length"])
                counts.add(f["rectangles"])
                for name in ("exact", "unequal", "horizontal", "empty_row", "empty_column", "zero_payload"):
                    measurements[name] += bool(f[name])
                measurements["pairs"] += 1
                measurements["deep_test"] += split == "test" and f["active_depth"] >= 3
                measurements["wide_test"] += split == "test" and f["active_width"] >= 3
                measurements["within_palette_chain"] += bool(f["chain"])
    assert len(puzzle_hashes) == 42 and len(structures) >= 140
    assert len(shapes) >= 20 and len(backgrounds) == 10 and len(key_lengths) >= 5
    assert len(counts) >= 12
    assert all(measurements[name] >= 5 for name in (
        "deep_test", "wide_test", "empty_row", "empty_column", "zero_payload"))
    # Exhaust the small API count domain in memory, using legal explicit values.
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain - 1) + 1):
            puzzle = generate_puzzle(1234567 + ntrain * 10 + ntest, ntrain, ntest)
            assert len(puzzle["train"]) == ntrain and len(puzzle["test"]) == ntest
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    broken = generate_puzzle(0)
    bad = broken["train"][0]["output"]
    bad[0, 0] = (int(bad[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("mutated key accepted")
    return {"seeds": len(HOST_SEEDS), "pairs": measurements["pairs"],
            "normalized_coarse_structures": len(structures), "coarse_shapes": len(shapes),
            "backgrounds": sorted(backgrounds), "key_lengths": sorted(key_lengths),
            "rectangle_counts": sorted(counts), "features": dict(measurements),
            "independent_semantic_verification": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
