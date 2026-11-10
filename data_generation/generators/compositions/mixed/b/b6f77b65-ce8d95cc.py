"""Swatch removal, rigid gravity, then lower-area equality-run contraction.

No parent imports, data files, random state, or demonstrations are used by solve.
All diagnostics in self_test are bounded and kept in memory.
"""
from itertools import combinations

import numpy as np


# Rods are (orientation, top row, left column, length, color).
def _cells(rod):
    o, r, c, n, _ = rod
    return frozenset((r + (k if o == "V" else 0),
                      c + (k if o == "H" else 0)) for k in range(n))


def _shift(rod, d):
    return rod[0], rod[1] + d, rod[2], rod[3], rod[4]


def _joints(rods):
    footprints = [_cells(p) for p in rods]
    return {(j, i) for i in range(len(rods)) for j in range(i)
            if footprints[i] & footprints[j]}


def _settle(rods, height, joints=None):
    """Greatest simultaneously movable set; pre-existing crossings slide."""
    rods = list(rods)
    joints = _joints(rods) if joints is None else joints
    for _ in range(height * (len(rods) + 1)):
        cells = [_cells(p) for p in rods]
        below = [{(r + 1, c) for r, c in s} for s in cells]
        moving = {i for i, s in enumerate(below) if all(r < height for r, _ in s)}
        while moving:
            blocked = {i for i in moving for j in range(len(rods))
                       if j not in moving and (min(i, j), max(i, j)) not in joints
                       and below[i] & cells[j]}
            if not blocked:
                break
            moving -= blocked
        if not moving:
            return rods
        rods = [_shift(p, 1) if i in moving else p for i, p in enumerate(rods)]
    raise AssertionError("gravity did not converge")


def _settle_events(rods, height):
    """Bounded event-distance cross-check, not an independent semantic solver."""
    rods = list(rods)
    joints = _joints(rods)
    for _ in range(height * (len(rods) + 1)):
        changed = False
        for i in sorted(range(len(rods)),
                        key=lambda k: max(r for r, _ in _cells(rods[k])), reverse=True):
            distance = 0
            for d in range(1, height + 1):
                target = _cells(_shift(rods[i], d))
                if any(r >= height for r, _ in target) or any(
                    target & _cells(q) for j, q in enumerate(rods)
                    if j != i and (min(i, j), max(i, j)) not in joints
                ):
                    break
                distance = d
            if distance:
                rods[i] = _shift(rods[i], distance)
                changed = True
        if not changed:
            return rods
    raise AssertionError("event settling did not converge")


def _render(shape, bg, band, rods):
    grid = np.full(shape, bg, dtype=np.int8)
    grid[0] = band
    # Horizontal rods cover rails at crossings; contraction never repaints them.
    for p in sorted(rods, key=lambda p: p[0] == "H"):
        for cell in _cells(p):
            grid[cell] = p[4]
    return grid


def _checked_grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return grid.astype(np.int8, copy=False)


def _parse(value):
    grid = _checked_grid(value)
    h, w = grid.shape
    if h < 6 or w < 3 or not np.all(grid[1] == grid[1, 0]):
        raise ValueError("expected a top swatch row and uniform separator")
    bg = int(grid[1, 0])
    commands = tuple(int(c) for c in grid[0] if c != bg)
    if not 1 <= len(commands) <= 3 or len(commands) != len(set(commands)):
        raise ValueError("expected one to three distinct swatches")
    horizontal = np.zeros((h, w), dtype=bool)
    rods = []
    for r in range(2, h):
        c = 0
        while c < w:
            stop = c + 1
            while stop < w and grid[r, stop] == grid[r, c]:
                stop += 1
            if grid[r, c] != bg and stop - c >= 2:
                rods.append(("H", r, c, stop - c, int(grid[r, c])))
                horizontal[r, c:stop] = True
            c = stop
    # Only bracketed one-cell occlusions are joined. Hidden endpoints are never
    # guessed, and a vertical piece needs at least two exposed vertical cells.
    for c in range(w):
        for color in sorted(int(x) for x in np.unique(grid[2:, c]) if x != bg):
            groups = []
            for r in np.flatnonzero(grid[2:, c] == color) + 2:
                r = int(r)
                if groups and (r == groups[-1][-1] + 1 or
                               (r == groups[-1][-1] + 2 and horizontal[r - 1, c])):
                    groups[-1].append(r)
                else:
                    groups.append([r])
            for rows in groups:
                if sum(not horizontal[r, c] for r in rows) >= 2:
                    rods.append(("V", rows[0], c, rows[-1] - rows[0] + 1, color))
    if not rods or not np.array_equal(_render(grid.shape, bg, grid[0], rods), grid):
        raise ValueError("unresolved straight-stick decomposition")
    for i, j in _joints(rods):
        if rods[i][0] == rods[j][0] or rods[i][4] == rods[j][4]:
            raise ValueError("ambiguous crossing")
    return grid, bg, commands, rods


def _runs(grid, axis):
    vectors = grid if axis == 0 else grid.T
    starts = [0]
    for i in range(1, len(vectors)):
        if not np.array_equal(vectors[i - 1], vectors[i]):
            starts.append(i)
    lengths = np.diff(starts + [len(vectors)]).tolist()
    return starts, lengths


def _contract(grid):
    # Compare both axes on the SAME settled lower matrix. Keeping one complete
    # copy of every row cannot change equality of any pair of columns.
    rows, _ = _runs(grid, 0)
    cols, _ = _runs(grid, 1)
    return grid[np.ix_(rows, cols)].copy()


def _pack(original, lower, bg):
    """Keep the full control band; left-align the shorter lower rectangle."""
    compact = _contract(lower)
    out = np.full((2 + compact.shape[0], original.shape[1]), bg, dtype=np.int8)
    out[:2] = original[:2]
    out[2:, :compact.shape[1]] = compact
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid, bg, commands, rods = _parse(input_grid)
    survivors = [p for p in rods if p[4] not in commands]
    settled = _settle(survivors, grid.shape[0])
    lower = _render(grid.shape, bg, grid[0], settled)[2:]
    return _pack(grid, lower, bg)


def _band(rng, width, bg, commands, edge=False):
    band = np.full(width, bg, dtype=np.int8)
    positions = rng.choice(width, len(commands), replace=False).tolist()
    if edge and width - 1 not in positions:
        positions[0] = width - 1
    values = list(commands)
    rng.shuffle(values)
    for c, color in zip(positions, values):
        band[c] = color
    return band


def _pair(shape, bg, band, rods, commands):
    inp = _render(shape, bg, band, rods)
    kept = [p for p in rods if p[4] not in commands]
    lower = _render(shape, bg, band, _settle(kept, shape[0]))[2:]
    return {"input": inp, "output": _pack(inp, lower, bg)}


def _anchors(rng, rails):
    h, w = int(rng.integers(15, 27)), int(rng.integers(11, 28))
    bg, beam, tall, short, foot, rail = [int(x) for x in rng.choice(10, 6, replace=False)]
    left = int(rng.integers(1, 3))
    right = int(rng.integers(w - 4, w - 1))
    # Unobstructed beam columns on BOTH sides make separated equal columns
    # observable even in the minimum two-demonstration puzzle.
    xs = rng.choice(np.arange(left + 1, right), 4, replace=False).tolist()
    base = int(rng.integers(2, 5))
    lower = h - base - 1
    tall_n = int(rng.integers(3, min(8, lower - 5) + 1))
    short_n = int(rng.integers(2, tall_n))
    upper = lower - tall_n - 1
    rods = [("H", upper, left, right - left + 1, beam),
            ("H", lower, left, right - left + 1, beam),
            ("V", upper + 1, xs[0], tall_n, tall),
            ("V", lower - short_n, xs[1], short_n, short),
            ("V", lower + 1, xs[2], base, foot)]
    if rails:
        rods.append(("V", upper - 2, xs[3], h - upper + 2, rail))
    pairs = []
    for commands in ((tall, short), (tall,)):
        band = _band(rng, w, bg, commands, edge=True)
        pairs.append(_pair((h, w), bg, band, rods, commands))
    return pairs


def _tier_geometry(rng, shelves, rails):
    """Compose an irregular support network rather than equal-width modules."""
    w = int(rng.integers(10, 31))
    spine = int(rng.integers(3, w - 3))
    max_gap = min(6, 24 // shelves - 1)
    gaps = [int(rng.integers(2, max_gap + 1)) for _ in range(shelves)]
    minimum_h = sum(n + 1 for n in gaps) + 4
    h = int(rng.integers(minimum_h, 31))
    spans = [(int(rng.integers(0, spine - 1)),
              int(rng.integers(spine + 2, w))) for _ in range(shelves)]
    rods = []
    below = h
    for i, gap in enumerate(gaps):
        left, right = spans[i]
        if i:
            left = max(left, spans[i - 1][0])
            right = min(right, spans[i - 1][1])
        choices = [c for c in range(left, right + 1) if c != spine]
        count = int(rng.integers(1, min(3, len(choices)) + 1))
        xs = rng.choice(choices, count, replace=False).tolist()
        lengths = [gap] + [int(rng.integers(2, gap + 1)) for _ in xs[1:]]
        for c, n in zip(xs, lengths):
            rods.append(("V", below - n, int(c), n, 0))
        row = below - gap - 1
        left, right = spans[i]
        rods.append(("H", row, left, right - left + 1, 0))
        below = row
    if rails:
        top = below - 2
        rods.append(("V", top, spine, h - top, 0))
    return (h, w), rods


def _color(rng, geometry, bg):
    cells = [_cells(p) for p in geometry]
    edges = [set() for _ in geometry]
    for i, s in enumerate(cells):
        neighbors = s | {(r + dr, c + dc) for r, c in s
                         for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))}
        for j in range(i):
            ambiguous = False
            a, b = geometry[i], geometry[j]
            if a[0] == b[0] == "V" and a[2] == b[2]:
                # Two posts separated only by one shelf must not look like one
                # bracketed continuous rail of a shared identity.
                ambiguous = min(abs(a[1] - (b[1] + b[3] - 1)),
                                abs(b[1] - (a[1] + a[3] - 1))) <= 2
            if neighbors & cells[j] or ambiguous:
                edges[i].add(j)
                edges[j].add(i)
    palette = rng.permutation([c for c in range(10) if c != bg]).tolist()
    palette = palette[:int(rng.integers(4, 10))]
    assigned = {}
    for i in sorted(range(len(geometry)), key=lambda i: -len(edges[i])):
        options = [c for c in palette if c not in {assigned[j] for j in edges[i] if j in assigned}]
        if not options:
            return None
        assigned[i] = int(rng.choice(options))
    return [(*p[:4], assigned[i]) for i, p in enumerate(geometry)]


def _generic_pair(rng, shelves, rails, collapse):
    for _ in range(64):
        shape, geometry = _tier_geometry(rng, shelves, rails)
        bg = int(rng.integers(10))
        rods = _color(rng, geometry, bg)
        if rods is None:
            continue
        probe = _render(shape, bg, _band(rng, shape[1], bg, [rods[0][4]]), rods)
        try:
            _, _, _, parsed = _parse(probe)
        except ValueError:
            continue
        if set(parsed) != set(rods) or _settle(rods, shape[0]) != rods:
            continue
        present = sorted({p[4] for p in rods})
        absent = [c for c in range(10) if c != bg and c not in present]
        options = [(c,) for c in present + absent[:1]]
        for n in (2, 3):
            combos = list(combinations(present, n))
            if len(combos) > 16:
                combos = [combos[int(i)] for i in rng.choice(len(combos), 16, replace=False)]
            options.extend(combos)
        rng.shuffle(options)
        # Test cases must visibly depend on gravity even after contraction.
        for commands in options:
            survivors = [p for p in rods if p[4] not in commands]
            settled = _settle(survivors, shape[0])
            if collapse and (not survivors or settled == survivors):
                continue
            band = _band(rng, shape[1], bg, commands)
            pair = _pair(shape, bg, band, rods, commands)
            if collapse:
                deleted = _render(shape, bg, band, survivors)[2:]
                if np.array_equal(_pack(pair["input"], deleted, bg), pair["output"]):
                    continue
            return pair
    raise AssertionError("bounded construction could not find a readable support network")


def _integer(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f"invalid {name}")
    return int(value)


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    num_train = _integer(num_train, 2, 10, "num_train")
    num_test = _integer(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    rails = bool(rng.integers(0, 2))
    train = _anchors(rng, rails)
    seen = {_key(p["input"]) for p in train}
    for _ in range(num_train - 2):
        for _ in range(16):
            pair = _generic_pair(rng, int(rng.integers(2, 4)),
                                 rails and bool(rng.integers(0, 2)), bool(rng.integers(0, 2)))
            if _key(pair["input"]) not in seen:
                seen.add(_key(pair["input"]))
                train.append(pair)
                break
        else:
            raise AssertionError("distinct-input budget exhausted")
    maximum = max(sum(p[0] == "H" for p in _parse(pair["input"])[3]) for pair in train)
    tests = []
    for i in range(num_test):
        for _ in range(16):
            pair = _generic_pair(rng, maximum + 1 + i, rails, True)
            if _key(pair["input"]) not in seen:
                seen.add(_key(pair["input"]))
                tests.append(pair)
                break
        else:
            raise AssertionError("distinct-input budget exhausted")
    rng.shuffle(train)
    rng.shuffle(tests)
    puzzle = {"train": train, "test": tests}
    validate_generated_puzzle(puzzle)
    return puzzle


def _separated_nonempty(vectors, bg):
    return any(np.array_equal(vectors[i], vectors[j]) and np.any(vectors[i] != bg)
               and any(not np.array_equal(vectors[k], vectors[i]) for k in range(i + 1, j))
               for i in range(len(vectors)) for j in range(i + 2, len(vectors)))


def _analyze_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    grid, bg, commands, rods = _parse(pair["input"])
    out = _checked_grid(pair["output"])
    assert _settle(rods, grid.shape[0]) == rods, "input scaffold must already be stable"
    survivors = [p for p in rods if p[4] not in commands]
    settled = _settle(survivors, grid.shape[0])
    joints = _joints(survivors)
    assert _settle(settled, grid.shape[0], joints) == settled
    assert all(q[1] >= p[1] and q[0] == p[0] and q[2:] == p[2:]
               for p, q in zip(survivors, settled))
    for i, q in enumerate(settled):
        for j in range(i):
            assert not (_cells(q) & _cells(settled[j])) or (j, i) in joints
    full = _render(grid.shape, bg, grid[0], settled)
    lower = full[2:]
    compact = _contract(lower)
    rows, row_lengths = _runs(lower, 0)
    cols, col_lengths = _runs(lower, 1)
    assert np.array_equal(np.repeat(np.repeat(compact, row_lengths, axis=0),
                                    col_lengths, axis=1), lower)
    assert np.array_equal(_contract(compact), compact)
    assert np.array_equal(_contract(lower.T).T, compact), "axes must commute"
    assert np.array_equal(out, _pack(grid, lower, bg))
    assert np.array_equal(out[:2], grid[:2]), "control band must stay byte-for-byte intact"
    assert out.shape[1] == grid.shape[1]
    deleted = _render(grid.shape, bg, grid[0], survivors)[2:]
    pixels = np.full_like(deleted, bg)
    for c in range(deleted.shape[1]):
        colors = deleted[:, c][deleted[:, c] != bg]
        if len(colors):
            pixels[-len(colors):, c] = colors
    shelf_rows = {p[1] - 2 for p in settled if p[0] == "H"}
    return {"bg": bg, "shelves": sum(p[0] == "H" for p in rods), "rods": len(rods),
            "posts": sum(p[0] == "V" for p in rods), "joints": len(_joints(rods)),
            "commands": len(commands), "palette": len({p[4] for p in rods}),
            "falls": sorted({q[1] - p[1] for p, q in zip(survivors, settled)} - {0}),
            "row_shrink": lower.shape[0] - compact.shape[0],
            "col_shrink": lower.shape[1] - compact.shape[1],
            "colored_row_run": any(n > 1 and np.any(lower[r] != bg)
                                   for r, n in zip(rows, row_lengths)),
            "shelf_row_run": any(n > 1 and r in shelf_rows
                                 for r, n in zip(rows, row_lengths)),
            "colored_col_run": any(n > 1 and np.any(lower[:, c] != bg)
                                   for c, n in zip(cols, col_lengths)),
            "separated_rows": _separated_nonempty(compact, bg),
            "separated_cols": _separated_nonempty(compact.T, bg),
            "deletion_matters": not np.array_equal(_pack(grid, grid[2:], bg), out),
            "gravity_matters": not np.array_equal(_pack(grid, deleted, bg), out),
            "compression_matters": not np.array_equal(full, out),
            "pixel_gravity_differs": not np.array_equal(_pack(grid, pixels, bg), out),
            "stale_masks_differ": not np.array_equal(
                lower[np.ix_(_runs(grid[2:], 0)[0], _runs(grid[2:], 1)[0])], compact)}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            key = _key(_checked_grid(pair["input"]))
            assert key not in seen, "duplicate input"
            seen.add(key)
            records[split].append(_analyze_pair(pair))
    train = records["train"]
    assert any(f["shelf_row_run"] and f["row_shrink"] and f["col_shrink"] for f in train)
    assert any(f["separated_rows"] and f["separated_cols"] for f in train)
    for concept in ("deletion_matters", "gravity_matters", "compression_matters",
                    "stale_masks_differ", "pixel_gravity_differs"):
        assert any(f[concept] for f in train), concept
    max_train = max(f["shelves"] for f in train)
    assert all(f["shelves"] > max_train and f["gravity_matters"] for f in records["test"])
    assert len({f["shelves"] for f in records["test"]}) == len(records["test"])
    if any(f["joints"] for f in records["test"]):
        assert any(f["joints"] and f["falls"] for f in train)


def _fingerprint(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    """42 streamed seeds, every legal count, and causal/cell-preservation checks."""
    # Literal contractions retain separated repeats, including crossing values.
    matrix = np.array([[0, 0, 2, 0, 0], [0, 0, 2, 0, 0],
                       [3, 3, 2, 3, 3], [0, 0, 2, 0, 0]], dtype=np.int8)
    assert _contract(matrix).tolist() == [[0, 2, 0], [3, 2, 3], [0, 2, 0]]
    shapes, out_shapes, layouts, fingerprints, backgrounds = set(), set(), set(), set(), set()
    metrics = {k: 0 for k in ("joints", "colored_row_run", "colored_col_run",
                              "separated_rows", "separated_cols", "stale_masks_differ")}
    counts, fall_sets, palettes = set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        for pair in puzzle["train"] + puzzle["test"]:
            f = _analyze_pair(pair)
            grid, bg, commands, rods = _parse(pair["input"])
            survivors = [p for p in rods if p[4] not in commands]
            assert _settle_events(survivors, grid.shape[0]) == _settle(survivors, grid.shape[0])
            assert np.array_equal(solve(grid.copy()), pair["output"])
            shapes.add(grid.shape)
            out_shapes.add(pair["output"].shape)
            layouts.add((grid.shape, (grid[2:] != bg).tobytes()))
            backgrounds.add(bg)
            counts.add((f["shelves"], f["posts"]))
            fall_sets.update(f["falls"])
            palettes.add(f["palette"])
            for k in metrics:
                metrics[k] += bool(f[k])
    assert len(fingerprints) == 42 and len(backgrounds) == 10
    assert len(shapes) >= 35 and len(out_shapes) >= 25 and len(layouts) >= 120
    assert len(counts) >= 12 and len(fall_sets) >= 5 and len(palettes) >= 4
    assert all(v >= 15 for v in metrics.values()), metrics
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(719, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid example counts accepted")
    for seed in (-1, True, 1.5):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid seed accepted")
    return {"seeds": 42, "pairs": 210, "input_shapes": len(shapes),
            "output_shapes": len(out_shapes), "layouts": len(layouts),
            "inventories": len(counts), "fall_distances": sorted(fall_sets),
            "backgrounds": len(backgrounds), "palette_sizes": sorted(palettes),
            "features": metrics, "independent_solver": False}
