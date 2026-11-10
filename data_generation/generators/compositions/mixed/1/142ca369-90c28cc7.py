"""L rays in a nonuniformly enlarged, alternating-substrate mosaic.

Only the standard library and NumPy are used. No files, examples, seed state,
or generation metadata are consulted by solve(). See task_spec.md.
"""

from collections import defaultdict
import hashlib

import numpy as np


DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError("num_test must be in 1..4 and smaller than num_train")
    return int(num_train), int(num_test)


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((0 <= grid) & (grid <= 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return grid.astype(np.int8, copy=True)


def _bands(grid):
    rows = np.r_[0, np.flatnonzero(np.any(grid[1:] != grid[:-1], axis=1)) + 1]
    cols = np.r_[0, np.flatnonzero(np.any(grid[:, 1:] != grid[:, :-1], axis=0)) + 1]
    return rows, cols


def _compress(grid):
    rows, cols = _bands(grid)
    return grid[np.ix_(rows, cols)].copy()


def _visible(grid):
    return (not np.any(np.all(grid[1:] == grid[:-1], axis=1))
            and not np.any(np.all(grid[:, 1:] == grid[:, :-1], axis=0)))


def _decode(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    if not grid[0, -1] == grid[-1, 0] == grid[-1, -1] == bg:
        raise ValueError("four corners must show the exterior color")
    rr, cc = np.nonzero(grid != bg)
    if not len(rr):
        raise ValueError("missing mosaic")
    top, bottom, left, right = int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1
    crop = grid[top:bottom, left:right].copy()
    if np.any(crop == bg) or len(rr) != crop.size:
        raise ValueError("mosaic must be one filled rectangle")
    rows, cols = _bands(crop)
    heights = np.diff(np.r_[rows, crop.shape[0]])
    widths = np.diff(np.r_[cols, crop.shape[1]])
    logical = crop[np.ix_(rows, cols)].copy()
    if not np.array_equal(np.repeat(np.repeat(logical, heights, 0), widths, 1), crop):
        raise ValueError("invalid rectangular bands")
    # A substrate color appears on exactly one checkerboard parity. Every L
    # spans both parities, so its color cannot impersonate a substrate color.
    parity = np.indices(logical.shape).sum(axis=0) % 2
    present = [set(map(int, np.unique(logical[parity == p]))) for p in (0, 1)]
    exclusive = [present[0] - present[1], present[1] - present[0]]
    if any(len(values) != 1 for values in exclusive):
        raise ValueError("two alternating substrate colors must be identifiable")
    base = tuple(next(iter(values)) for values in exclusive)
    return logical, base, crop, heights, widths, (top, grid.shape[0] - bottom, left, grid.shape[1] - right), bg


def _sources(logical, base):
    remaining = set(map(tuple, np.argwhere(~np.isin(logical, base))))
    result = []
    while remaining:
        start = min(remaining)
        color = int(logical[start])
        remaining.remove(start)
        todo, cells = [start], set()
        while todo:
            r, c = todo.pop()
            cells.add((r, c))
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining and logical[q] == color:
                    remaining.remove(q)
                    todo.append(q)
        r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
        square = {(r0 + r, c0 + c) for r in (0, 1) for c in (0, 1)}
        if len(cells) != 3 or not cells < square:
            raise ValueError("each source component must be a three-cell L")
        missing = next(iter(square - cells))
        elbow = (2 * r0 + 1 - missing[0], 2 * c0 + 1 - missing[1])
        direction = (elbow[0] - missing[0], elbow[1] - missing[1])
        result.append((elbow, direction, color, cells))
    if not result:
        raise ValueError("missing L source")
    return result


def _path(elbow, direction, shape):
    r, c = elbow
    dr, dc = direction
    distance = min(shape[0] - 1 - r if dr > 0 else r,
                   shape[1] - 1 - c if dc > 0 else c)
    return [(r + step * dr, c + step * dc) for step in range(1, distance + 1)]


def _trace(logical, base):
    sources = _sources(logical, base)
    occupied = {cell for _, _, _, cells in sources for cell in cells}
    painted = {}
    answer = logical.copy()
    for elbow, direction, color, _ in sources:
        path = _path(elbow, direction, logical.shape)
        if len(path) < 2:
            raise ValueError("each ray needs at least two visible new cells")
        for cell in path:
            if cell in occupied:
                raise ValueError("ray strikes a source")
            if cell in painted and painted[cell] != color:
                raise ValueError("different-colored rays demand the same cell")
            painted[cell] = color
            answer[cell] = color
    if not _visible(answer):
        raise ValueError("completed mosaic must retain all logical boundaries")
    return answer, sources, painted


def solve(input_grid: np.ndarray) -> np.ndarray:
    logical, base, crop, heights, widths, _, _ = _decode(input_grid)
    _, _, painted = _trace(logical, base)
    # A acts on whole expanded rectangles. B acts on the completed pixel
    # mosaic, so separated equal colors are never globally deduplicated.
    row_edges = np.r_[0, np.cumsum(heights)]
    col_edges = np.r_[0, np.cumsum(widths)]
    for (r, c), color in painted.items():
        crop[row_edges[r]:row_edges[r + 1], col_edges[c]:col_edges[c + 1]] = color
    return _compress(crop)


def _source_cells(elbow, direction):
    r, c = elbow
    dr, dc = direction
    return {(r, c), (r - dr, c), (r, c - dc)}


def _repeat_witness(painted, answer):
    groups = defaultdict(list)
    for (r, c), shade in painted.items():
        groups[(r, shade)].append(c)
    for (r, shade), columns in groups.items():
        if len(columns) >= 2:
            lo, hi = min(columns), max(columns)
            if hi - lo >= 2 and np.any(answer[r, lo:hi + 1] != shade):
                return True
    return False


def _scene(rng, shape, count, colors, base):
    parity = np.indices(shape).sum(axis=0) % 2
    logical = np.asarray(base, dtype=np.int8)[parity]
    options = []
    for r in range(shape[0]):
        for c in range(shape[1]):
            for direction in DIRECTIONS:
                cells = _source_cells((r, c), direction)
                if not all(0 <= a < shape[0] and 0 <= b < shape[1] for a, b in cells):
                    continue
                path = _path((r, c), direction, shape)
                if len(path) >= 2:
                    options.append(((r, c), direction, cells, path))
    rng.shuffle(options)
    occupied, painted, selected = set(), {}, []
    for color in colors:
        for index, (elbow, direction, cells, path) in enumerate(options):
            if (cells & occupied or cells & painted.keys() or any(p in occupied for p in path)
                    or any(p in painted and painted[p] != color for p in path)):
                continue
            if any((a + dr, b + dc) in occupied and logical[a + dr, b + dc] == color
                   for a, b in cells for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))):
                continue
            for p in cells:
                logical[p] = color
            occupied.update(cells)
            painted.update((p, color) for p in path)
            selected.append((elbow, direction))
            options.pop(index)
            break
        else:
            return None
    if len(selected) != count or not _visible(logical):
        return None
    try:
        answer, sources, painted = _trace(logical, base)
    except ValueError:
        return None
    return logical, answer, sources, painted


def _axis(rng, size, strong=False, uneven=False):
    if not strong and not uneven and rng.random() < .22:
        scales = np.full(size, int(rng.integers(2, max(3, min(4, 28 // size)) + 1)), dtype=int)
    else:
        scales = rng.integers(1, 4 if strong else 5, size=size)
    peak = int(rng.integers(size))
    if strong:
        scales[peak] = int(rng.integers(5, 10))
        scales[(peak + int(rng.integers(1, size))) % size] = 1
    while scales.sum() > 28:
        choices = [i for i in range(size) if scales[i] > 1 and (not strong or i != peak)]
        scales[int(rng.choice(choices))] -= 1
    if uneven and len(set(map(int, scales))) == 1:
        scales[peak] = 1
        scales[(peak + 1) % size] = 2
    if scales.max() == 1:
        scales[peak] = 2
    return scales


def _embed(rng, logical, heights, widths, background):
    expanded = np.repeat(np.repeat(logical, heights, 0), widths, 1)
    extras = [int(rng.integers(2, min(6, 30 - n) + 1)) for n in expanded.shape]
    top, left = [int(rng.integers(1, extra)) for extra in extras]
    bottom, right = extras[0] - top, extras[1] - left
    if rng.random() < .24:
        side = int(rng.integers(4))
        if side == 0:
            top, bottom = 0, extras[0]
        elif side == 1:
            top, bottom = extras[0], 0
        elif side == 2:
            left, right = 0, extras[1]
        else:
            left, right = extras[1], 0
    grid = np.full((expanded.shape[0] + top + bottom, expanded.shape[1] + left + right),
                   background, dtype=np.int8)
    grid[top:top + expanded.shape[0], left:left + expanded.shape[1]] = expanded
    return grid


def _key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _make_pair(rng, kind, seen, avoid_directions=()):
    # At most 128 small scene attempts per pair; no files or retained attempts.
    for attempt in range(128):
        low, high = 5, 12
        if attempt >= 64:
            low, high = 9, 12
        shape = tuple(map(int, rng.integers(low, high, size=2)))
        count = 1 if kind == "single" else int(rng.integers(2, 5 if kind == "repeat" else 7))
        if kind == "ordinary":
            count = int(rng.integers(1, 6))
        palette = list(map(int, rng.permutation(10)))
        background, base = palette[0], tuple(palette[1:3])
        capacity = 1 if kind in ("single", "repeat") else int(rng.integers(1, min(5, count) + 1))
        source_palette = palette[3:3 + capacity]
        colors = source_palette + list(map(int, rng.choice(source_palette, count - capacity)))
        rng.shuffle(colors)
        scene = _scene(rng, shape, count, colors, base)
        if scene is None:
            continue
        logical, answer, sources, painted = scene
        directions = {s[1] for s in sources}
        if kind == "repeat" and (len(directions) < 2 or not _repeat_witness(painted, answer)
                                  or not (directions - set(avoid_directions))):
            continue
        if kind == "test" and len(directions) < 2:
            continue
        heights = _axis(rng, shape[0], strong=kind == "test", uneven=kind in ("single", "repeat"))
        widths = _axis(rng, shape[1], strong=kind == "test", uneven=kind in ("single", "repeat"))
        grid = _embed(rng, logical, heights, widths, background)
        if _key(grid) in seen:
            continue
        return {"input": grid, "output": answer.copy()}
    raise RuntimeError("bounded construction exhausted; no alternative rule was substituted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if not _integer(seed):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    seen = set()
    first = _make_pair(rng, "single", seen)
    seen.add(_key(first["input"]))
    logical, base, *_ = _decode(first["input"])
    directions = {s[1] for s in _sources(logical, base)}
    second = _make_pair(rng, "repeat", seen, directions)
    seen.add(_key(second["input"]))
    puzzle = {"train": [first, second], "test": []}
    for split, count, kind in (("train", num_train, "ordinary"), ("test", num_test, "test")):
        while len(puzzle[split]) < count:
            pair = _make_pair(rng, kind, seen)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _analytic_answer(logical, base):
    """Cell-membership check, distinct from solve's rectangle painter.

    This is a generator-owned check, not an independent reference solver.
    """
    rr, cc = np.indices(logical.shape)
    expected = logical.copy()
    demanded = np.full(logical.shape, -1, dtype=int)
    source_mask = ~np.isin(logical, base)
    for (r, c), (dr, dc), shade, _ in _sources(logical, base):
        step = (rr - r) * dr
        mask = (step >= 1) & ((cc - c) * dc == step)
        assert np.count_nonzero(mask) >= 2
        assert not np.any(mask & source_mask), "source impact"
        assert not np.any(mask & (demanded >= 0) & (demanded != shade)), "ray conflict"
        demanded[mask] = shade
        expected[mask] = shade
    return expected


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_features = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            logical, base, crop, heights, widths, margins, bg = _decode(inp)
            expected, sources, painted = _trace(logical, base)
            assert np.array_equal(out, expected)
            assert np.array_equal(out, _analytic_answer(logical, base))
            assert np.array_equal(out, solve(inp))
            assert _visible(logical) and _visible(out)
            assert all(n >= 1 for n in (*heights, *widths)) and max(heights) > 1 and max(widths) > 1
            assert sum(m == 0 for m in margins) <= 1
            assert bg not in out and set(map(int, np.unique(out))) <= set(map(int, np.unique(inp)))
            source_mask = ~np.isin(logical, base)
            assert np.array_equal(out[source_mask], logical[source_mask])
            assert np.count_nonzero(out != logical) == len(painted) >= 2
            # Necessity of A: compression alone omits the actual ray cells.
            assert not np.array_equal(_compress(crop), out)
            # Necessity of B: the completed expanded mosaic is still larger.
            completed = np.repeat(np.repeat(out, heights, 0), widths, 1)
            assert completed.shape[0] > out.shape[0] and completed.shape[1] > out.shape[1]
            assert np.array_equal(_compress(completed), out)
            feature = {"directions": {s[1] for s in sources}, "count": len(sources),
                       "repeat": _repeat_witness(painted, out),
                       "uneven": len(set(heights)) > 1 and len(set(widths)) > 1}
            if split == "train":
                train_features.append(feature)
                assert max(heights) <= 4 and max(widths) <= 4
            else:
                assert max(heights) >= 5 and max(widths) >= 5
                assert 1 in heights and 1 in widths and len(feature["directions"]) >= 2
    assert any(f["count"] == 1 and f["uneven"] for f in train_features)
    assert any(f["count"] >= 2 and f["repeat"] and f["uneven"] for f in train_features)
    assert len(set().union(*(f["directions"] for f in train_features))) >= 2


def _same(first, second):
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test():
    # Streaming measurements: retain hashes and small feature sets, not grids.
    signatures, shapes, logical_shapes, structures, profiles = set(), set(), set(), set(), set()
    directions, counts, backgrounds, substrates, placements = set(), set(), set(), set(), set()
    touching = overlap = zero_margin = repeated = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                logical, base, _, rows, cols, margins, bg = _decode(pair["input"])
                out, sources, painted = _trace(logical, base)
                assert np.array_equal(out, pair["output"])
                digest.update(bytes(pair["input"].shape) + pair["input"].tobytes())
                shapes.add(pair["input"].shape)
                logical_shapes.add(logical.shape)
                profiles.add((tuple(map(int, rows)), tuple(map(int, cols))))
                backgrounds.add(bg)
                substrates.update(base)
                counts.add(len(sources))
                directions.update(s[1] for s in sources)
                zero_margin += 0 in margins
                repeated += _repeat_witness(painted, out)
                mapping = {}
                normalized = bytes(mapping.setdefault(int(x), len(mapping)) for x in logical.flat)
                structures.add(hashlib.sha256(bytes(logical.shape) + normalized).digest())
                union = set()
                for elbow, direction, _, cells in sources:
                    path = set(_path(elbow, direction, logical.shape))
                    overlap += bool(path & union)
                    union.update(path)
                    sides = int(any(r in (0, logical.shape[0] - 1) for r, _ in cells))
                    sides += int(any(c in (0, logical.shape[1] - 1) for _, c in cells))
                    placements.add(sides)
                for i, a in enumerate(sources):
                    for b in sources[i + 1:]:
                        touching += any(abs(r - rr) + abs(c - cc) == 1
                                        for r, c in a[3] for rr, cc in b[3])
        signatures.add(digest.digest())
    assert len(signatures) == 42 and len(shapes) >= 50 and len(logical_shapes) >= 20
    assert len(structures) >= 150 and len(profiles) >= 150
    assert directions == set(DIRECTIONS) and counts >= {1, 2, 3, 4, 5, 6}
    assert backgrounds == substrates == set(range(10)) and placements == {0, 1, 2}
    assert touching > 0 and overlap > 0 and zero_margin > 0 and repeated >= 42
    assert _same(generate_puzzle(0), generate_puzzle(0))
    assert not _same(generate_puzzle(0), generate_puzzle(1))
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(1000 + 10 * train + test, train, test)
            assert len(p["train"]) == train and len(p["test"]) == test
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    _regressions()


def _regressions():
    # Manually enumerated logical prediction, then warped with uneven bands.
    logical = np.array([[8, 1, 8, 9, 8], [1, 1, 9, 8, 9], [8, 9, 8, 9, 8],
                        [9, 8, 9, 8, 9], [8, 9, 8, 9, 8]], dtype=np.int8)
    expected = logical.copy()
    expected[2, 2] = expected[3, 3] = expected[4, 4] = 1
    rows, cols = [1, 4, 2, 1, 3], [3, 1, 2, 4, 1]
    expanded = np.repeat(np.repeat(logical, rows, 0), cols, 1)
    inp = np.pad(expanded, ((2, 1), (1, 3)), constant_values=0)
    for turns in range(4):
        assert np.array_equal(solve(np.rot90(inp, turns)), np.rot90(expected, turns))
    assert np.array_equal(inp, np.pad(expanded, ((2, 1), (1, 3)), constant_values=0))
    # Solve extends beyond sampled dimensions and band lengths. This 4x4
    # quotient has two ray cells and a band of length twelve.
    small = np.asarray([8, 9], dtype=np.int8)[np.indices((4, 4)).sum(0) % 2]
    small[0, 1] = small[1, 0] = small[1, 1] = 1
    expected_small = small.copy()
    expected_small[2, 2] = expected_small[3, 3] = 1
    warped = np.repeat(np.repeat(small, [1, 12, 2, 1], 0), [2, 1, 10, 2], 1)
    assert np.array_equal(solve(np.pad(warped, 1)), expected_small)
    # Seven distinct source colors on a 14x16 logical canvas: outside the
    # sampler's source count, palette size and logical dimension ranges.
    wide = np.asarray([8, 9], dtype=np.int8)[np.indices((14, 16)).sum(0) % 2]
    expected_wide = wide.copy()
    for color in range(1, 8):
        c = 2 * color - 1
        wide[0, c] = wide[1, c - 1] = wide[1, c] = color
        expected_wide[0, c] = expected_wide[1, c - 1] = expected_wide[1, c] = color
        for step in range(1, min(12, 15 - c) + 1):
            expected_wide[1 + step, c + step] = color
    warped = np.repeat(np.repeat(wide, [2] + [1] * 13, 0), [1] * 15 + [2], 1)
    assert np.array_equal(solve(np.pad(warped, 1)), expected_wide)
    # Same-colored shared cells have a unique answer; different-colored
    # shared cells, source impacts and merged source components are invalid.
    crossing = np.asarray([8, 9], dtype=np.int8)[np.indices((7, 7)).sum(0) % 2]
    crossing[0, 1] = crossing[1, 0] = crossing[1, 1] = 1
    crossing[0, 5] = crossing[1, 5] = crossing[1, 6] = 1
    expected_crossing = crossing.copy()
    for r in range(2, 7):
        expected_crossing[r, r] = expected_crossing[r, 6 - r] = 1
    assert np.array_equal(solve(np.pad(crossing, 1)), expected_crossing)
    conflicting = crossing.copy()
    conflicting[0, 5] = conflicting[1, 5] = conflicting[1, 6] = 2
    impact = np.asarray([8, 9], dtype=np.int8)[np.indices((7, 7)).sum(0) % 2]
    impact[0, 1] = impact[1, 0] = impact[1, 1] = 1
    impact[3, 4] = impact[4, 3] = impact[4, 4] = 2
    merged = np.asarray([8, 9], dtype=np.int8)[np.indices((7, 7)).sum(0) % 2]
    merged[0, 1] = merged[1, 0] = merged[1, 1] = 1
    merged[0, 3] = merged[1, 2] = merged[1, 3] = 1
    for invalid in (conflicting, impact, merged):
        try:
            solve(np.pad(invalid, 1))
        except ValueError:
            pass
        else:
            raise AssertionError("ambiguous or obstructed scene accepted")
    # Compression is of full vectors, preserves separated repetitions, and
    # keeps a same-color seam if another row witnesses the boundary.
    repeated = np.array([[1, 1, 2, 1], [3, 4, 3, 4], [1, 1, 2, 1]], dtype=np.int8)
    assert np.array_equal(_compress(np.repeat(np.repeat(repeated, [2, 3, 1], 0), [1, 4, 2, 3], 1)), repeated)
    # A wrong/missing ray label must be rejected, not merely accepted by shape.
    puzzle = generate_puzzle(17, 2, 1)
    bad = {"train": puzzle["train"], "test": [dict(puzzle["test"][0])]}
    bad["test"][0]["output"] = bad["test"][0]["output"].copy()
    logical, base, *_ = _decode(bad["test"][0]["input"])
    _, _, painted = _trace(logical, base)
    cell = next(iter(painted))
    bad["test"][0]["output"][cell] = logical[cell]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing ray accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, all legal explicit counts, semantic regressions)")
