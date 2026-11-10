"""Diamond-window shear: reconstruct walls, extract voids, project each void.

Only standard-library modules and NumPy are used. No parent code or files are
loaded. All information used by solve is present in its single input grid.
"""
from collections import deque
import json

import numpy as np


_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_KINDS = ("notched", "stair", "unequal", "partition", "mesh", "crossing", "nested")


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid of size 1..30, colors 0..9")
    return a


def _roles(grid):
    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) != 2 or int(counts.max()) * 2 <= grid.size:
        raise ValueError("input must have two colors and a strict-majority background")
    i = int(np.argmax(counts))
    return int(colors[i]), int(colors[1 - i])


def _tips(point):
    r, c = point
    return {(r + dr, c + dc) for dr, dc in _DIRECTIONS}


def _centers(grid, background, ink):
    h, w = grid.shape
    return [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)
            if grid[r, c] == background
            and all(grid[r + dr, c + dc] == ink for dr, dc in _DIRECTIONS)]


def _edges(centers):
    rows, cols = {}, {}
    for r, c in centers:
        rows.setdefault(r, []).append(c)
        cols.setdefault(c, []).append(r)
    result = []
    for r in sorted(rows):
        values = sorted(rows[r])
        result.extend(((r, a), (r, b)) for a, b in zip(values, values[1:]))
    for c in sorted(cols):
        values = sorted(cols[c])
        result.extend(((a, c), (b, c)) for a, b in zip(values, values[1:]))
    return result


def _walls(shape, centers, connect=True):
    wall = np.zeros(shape, dtype=bool)
    for center in centers:
        for r, c in _tips(center) | {center}:
            wall[r, c] = True
    if connect:
        for (r1, c1), (r2, c2) in _edges(centers):
            if r1 == r2:
                wall[r1, c1 + 2:c2 - 1] = True
            else:
                wall[r1 + 2:r2 - 1, c1] = True
    return wall


def _windows(wall):
    """Return all bounded four-neighbor components of the non-wall cells."""
    h, w = wall.shape
    seen = wall.copy()
    result = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            cells, exterior = [], False
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                exterior |= rr in (0, h - 1) or cc in (0, w - 1)
                for dr, dc in _DIRECTIONS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            if not exterior:
                result.append(np.asarray(cells, dtype=int))
    return result


def _projection(shape, windows, background, ink, anchor="local", boxes=False):
    out = np.full(shape, background, dtype=np.int8)
    if not windows:
        return out
    global_bottom = max(int(cells[:, 0].max()) for cells in windows)
    for cells in windows:
        if boxes:
            lo, hi = cells.min(axis=0), cells.max(axis=0)
            rr, cc = np.mgrid[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1]
            cells = np.column_stack((rr.ravel(), cc.ravel()))
        bottom = int(cells[:, 0].max())
        if anchor == "global":
            bottom = global_bottom
        elif anchor == "canvas":
            bottom = shape[0] - 1
        elif anchor == "none":
            bottom = None
        elif anchor != "local":
            raise ValueError("unknown anchor")
        rows, cols = cells.T
        destinations = cols if bottom is None else cols - (bottom - rows)
        keep = (destinations >= 0) & (destinations < shape[1])
        out[rows[keep], destinations[keep]] = ink
    return out


def _analyze(grid):
    background, ink = _roles(grid)
    centers = _centers(grid, background, ink)
    if not centers:
        raise ValueError("no hollow diamonds")
    wall = _walls(grid.shape, centers)
    return background, ink, centers, wall, _windows(wall)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from the input alone; never consult generation or training data."""
    a = _grid(input_grid)
    background, ink, _, _, windows = _analyze(a)
    return _projection(a.shape, windows, background, ink)


def _draw(shape, centers, background=0, ink=1):
    """Return None for ambiguous, overlapping, or obstructed marker layouts."""
    if not all(1 <= n <= 30 for n in shape) or len(set(centers)) != len(centers):
        return None
    h, w = shape
    occupied = set()
    for center in centers:
        r, c = center
        if not (1 <= r < h - 1 and 1 <= c < w - 1):
            return None
        footprint = _tips(center) | {center}
        if occupied & footprint:
            return None
        occupied.update(footprint)
    grid = np.full(shape, background, dtype=np.int8)
    for center in centers:
        for r, c in _tips(center):
            grid[r, c] = ink
    if set(_centers(grid, background, ink)) != set(centers):
        return None
    for (r1, c1), (r2, c2) in _edges(centers):
        if abs(r2 - r1) + abs(c2 - c1) < 4:
            return None
        corridor = grid[r1, c1 + 2:c2 - 1] if r1 == r2 else grid[r1 + 2:r2 - 1, c1]
        if np.any(corridor != background):
            return None
    if int(np.count_nonzero(grid == background)) * 2 <= grid.size:
        return None
    return grid


def _gaps(rng, n, low=4, high=7):
    return [int(v) for v in rng.integers(low, high + 1, size=n)]


def _cumulative(values):
    return [0] + list(np.cumsum(values).astype(int))


def _stair(rng, levels):
    rows = _cumulative(_gaps(rng, levels))
    cols = _cumulative(_gaps(rng, levels))
    points = [(0, 0), (0, cols[-1])]
    for i in range(levels):
        points.append((rows[i + 1], cols[levels - i]))
        points.append((rows[i + 1], cols[levels - i - 1]))
    return points, rows


def _core(rng, kind):
    if kind == "notched":
        h, w = int(rng.integers(5, 15)), int(rng.integers(9, 20))
        points = [(0, 0), (0, w), (h, 0), (h, w)]
        # One extra vertex on a side makes a one-cell inward tip, not a chord.
        points.append((int(rng.choice((0, h))), int(rng.integers(4, w - 3))))
    elif kind == "stair":
        points, _ = _stair(rng, int(rng.integers(2, 4)))
    elif kind == "unequal":
        a, b = _gaps(rng, 2, 5, 10)
        h1 = int(rng.integers(5, 10))
        h2 = h1 + int(rng.integers(4, 9))
        points = [(0, 0), (0, a), (0, a + b), (h1, 0),
                  (h1, a), (h2, a), (h2, a + b)]
    elif kind == "partition":
        points, rows = _stair(rng, 3)
        points.append((int(rng.choice(rows[1:-1])), 0))
    elif kind in ("mesh", "crossing"):
        nr = int(rng.integers(1, 4)) if kind == "mesh" else 2
        nc = int(rng.integers(1, 4)) if kind == "mesh" else int(rng.integers(2, 4))
        if nr * nc == 1:
            nr = 2
        rows = _cumulative(_gaps(rng, nr))
        cols = _cumulative(_gaps(rng, nc))
        points = [(r, c) for r in rows for c in cols
                  if kind == "mesh" or r in (rows[0], rows[-1])
                  or c in (cols[0], cols[-1])]
    elif kind == "nested":
        # A retained annular projection needs room for both height and width.
        h, w = int(rng.integers(14, 17)), int(rng.integers(14, 17))
        w = min(w, 31 - h)
        r1, c1 = int(rng.integers(4, 6)), int(rng.integers(4, 6))
        r2, c2 = h - int(rng.integers(4, 6)), w - int(rng.integers(4, 6))
        points = [(0, 0), (0, w), (h, 0), (h, w),
                  (r1, c1), (r1, c2), (r2, c1), (r2, c2)]
    else:
        raise ValueError("unknown construction")
    if rng.random() < 0.5:
        points = [(c, r) for r, c in points]
    if rng.random() < 0.5:
        points = [(-r, c) for r, c in points]
    if rng.random() < 0.5:
        points = [(r, -c) for r, c in points]
    return _normalize(points)


def _normalize(points):
    r0, c0 = min(r for r, _ in points), min(c for _, c in points)
    return sorted((int(r - r0), int(c - c0)) for r, c in points)


def _base_grid(points):
    points = _normalize(points)
    centers = [(r + 1, c + 1) for r, c in points]
    shape = (max(r for r, _ in points) + 3, max(c for _, c in points) + 3)
    return _draw(shape, centers), centers


def _decorate(rng, points):
    """Subdivide an edge or grow a spur; all resulting geometry is rechecked."""
    for _ in range(int(rng.integers(0, 3))):
        proposed = None
        long_edges = [(a, b) for a, b in _edges(points)
                      if abs(a[0] - b[0]) + abs(a[1] - b[1]) >= 8]
        if long_edges and rng.random() < 0.75:
            a, b = long_edges[int(rng.integers(len(long_edges)))]
            if a[0] == b[0]:
                proposed = (a[0], int(rng.integers(a[1] + 4, b[1] - 3)))
            else:
                proposed = (int(rng.integers(a[0] + 4, b[0] - 3)), a[1])
        elif rng.random() < 0.5:
            a = points[int(rng.integers(len(points)))]
            dr, dc = _DIRECTIONS[int(rng.integers(4))]
            gap = int(rng.integers(4, 7))
            proposed = (a[0] + gap * dr, a[1] + gap * dc)
        if proposed is not None and proposed not in points:
            candidate = _normalize(points + [proposed])
            grid, _ = _base_grid(candidate)
            if grid is not None:
                points = candidate
    return points


def _add_exterior_isolates(rng, grid, centers, desired):
    centers = list(centers)
    wall = _walls(grid.shape, centers)
    windows = _windows(wall)
    inside = np.zeros(grid.shape, dtype=bool)
    for cells in windows:
        inside[cells[:, 0], cells[:, 1]] = True
    for _ in range(desired):
        candidates = [(r, c) for r in range(1, grid.shape[0] - 1)
                      for c in range(1, grid.shape[1] - 1)
                      if r not in {rr for rr, _ in centers}
                      and c not in {cc for _, cc in centers}
                      and all(not (wall[rr, cc] or inside[rr, cc])
                              for rr, cc in _tips((r, c)) | {(r, c)})]
        if not candidates:
            break
        for index in rng.permutation(len(candidates))[:48]:
            candidate = candidates[int(index)]
            proposed = centers + [candidate]
            drawn = _draw(grid.shape, proposed)
            if drawn is None:
                continue
            new_wall = _walls(grid.shape, proposed)
            new_inside = np.zeros(grid.shape, dtype=bool)
            for cells in _windows(new_wall):
                new_inside[cells[:, 0], cells[:, 1]] = True
            if np.array_equal(new_inside, inside):
                centers, grid, wall = proposed, drawn, new_wall
                break
        else:
            break
    return grid, centers


def _shape_constraints(windows):
    return (1 <= len(windows) <= 12
            and all(len(a) >= 9 and np.all(a.max(axis=0) - a.min(axis=0) >= 2)
                    for a in windows))


def _geometry_facts(grid):
    background, ink, centers, wall, windows = _analyze(grid)
    out = _projection(grid.shape, windows, background, ink)
    raw = _projection(grid.shape, windows, background, ink, anchor="none")
    clipped = any(np.any(a[:, 1] - (int(a[:, 0].max()) - a[:, 0]) < 0) for a in windows)
    single_evidence = (len(windows) == 1
                       and not np.array_equal(out, _projection(grid.shape, windows, background, ink, boxes=True))
                       and not np.array_equal(out, _projection(grid.shape, windows, background, ink, anchor="canvas")))
    multi_evidence = (len(windows) >= 2 and len({int(a[:, 0].max()) for a in windows}) >= 2
                      and not np.array_equal(out, _projection(grid.shape, windows, background, ink, anchor="global")))
    return {"background": background, "ink": ink, "centers": centers, "wall": wall,
            "windows": windows, "output": out, "raw": raw, "clipped": clipped,
            "single_evidence": single_evidence, "multi_evidence": multi_evidence}


def _make_pair(rng, kind, clipped, evidence, background, ink):
    for _ in range(240):
        points = _decorate(rng, _core(rng, kind))
        base, centers = _base_grid(points)
        if base is None:
            continue
        windows = _windows(_walls(base.shape, centers))
        if not _shape_constraints(windows):
            continue
        minimum = min(int(np.min(a[:, 1] - (int(a[:, 0].max()) - a[:, 0]))) for a in windows)
        if clipped:
            if minimum >= 0:
                continue
            shift_c = int(rng.integers(0, -minimum))
        else:
            necessary = max(0, -minimum)
            spare = 30 - base.shape[1] - necessary
            if spare < 0:
                continue
            shift_c = necessary + int(rng.integers(0, min(3, spare) + 1))
        shift_r = int(rng.integers(0, min(4, 30 - base.shape[0]) + 1))
        height = base.shape[0] + shift_r
        height += int(rng.integers(0, min(4, 30 - height) + 1))
        width = base.shape[1] + shift_c
        if width > 30:
            continue
        width += int(rng.integers(0, min(4, 30 - width) + 1))
        if max(height, width) > 30:
            continue
        centers = [(r + shift_r, c + shift_c) for r, c in centers]
        drawn = _draw((height, width), centers)
        if drawn is None:
            continue
        drawn, centers = _add_exterior_isolates(rng, drawn, centers, int(rng.integers(0, 3)))
        facts = _geometry_facts(drawn)
        if (not _shape_constraints(facts["windows"])
                or facts["clipped"] != clipped
                or np.array_equal(facts["output"], facts["raw"])
                or np.array_equal(facts["output"], drawn)
                or (evidence and not facts[evidence + "_evidence"])
                or _windows(_walls(drawn.shape, centers, connect=False))):
            continue
        inp = np.where(drawn == 1, ink, background).astype(np.int8)
        out = np.where(facts["output"] == 1, ink, background).astype(np.int8)
        return {"input": inp, "output": out}
    raise RuntimeError(f"bounded construction failed: {kind}, clipped={clipped}, evidence={evidence}")


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= value <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _key(grid):
    a = np.asarray(grid)
    return a.shape, tuple(int(v) for v in a.flat)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    heldout = str(rng.choice(("partition", "mesh", "crossing")))
    single = str(rng.choice(("notched", "stair")))
    multi = str(rng.choice([k for k in ("unequal", "partition", "mesh", "crossing", "nested") if k != heldout]))
    clip_first = bool(rng.integers(2))
    train_specs = [(single, clip_first, "single"), (multi, not clip_first, "multi")]
    train_pool = [k for k in _KINDS if k != heldout]
    train_specs += [(str(rng.choice(train_pool)), bool(rng.integers(2)), None)
                    for _ in range(num_train - 2)]
    rng.shuffle(train_specs)
    nested_seen = any(kind == "nested" for kind, _, _ in train_specs)
    test_specs = []
    for _ in range(num_test):
        kind = "nested" if nested_seen and rng.random() < 0.25 else heldout
        test_specs.append((kind, bool(rng.integers(2)), None))
    fixed_palette = bool(rng.random() < 0.35)
    base_palette = tuple(map(int, rng.choice(10, size=2, replace=False)))
    puzzle, used = {"train": [], "test": []}, set()
    for split, specs in (("train", train_specs), ("test", test_specs)):
        for kind, clipped, evidence in specs:
            for _ in range(12):
                palette = base_palette if fixed_palette else tuple(map(int, rng.choice(10, size=2, replace=False)))
                pair = _make_pair(rng, kind, clipped, evidence, *palette)
                signature = _key(pair["input"])
                if signature not in used:
                    used.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not create a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            signature = _key(inp)
            assert signature not in seen, "duplicate input"
            seen.add(signature)
            facts = _geometry_facts(inp)
            background, ink, centers = facts["background"], facts["ink"], facts["centers"]
            markers = set().union(*(_tips(p) for p in centers))
            assert len(markers) == 4 * len(centers)
            assert markers == {tuple(map(int, p)) for p in np.argwhere(inp == ink)}
            assert _draw(inp.shape, centers, background, ink) is not None
            assert _shape_constraints(facts["windows"])
            assert not _windows(_walls(inp.shape, centers, connect=False)), "B ablation enclosed a region"
            assert np.array_equal(out, facts["output"])
            assert not np.array_equal(out, inp)
            assert not np.array_equal(out, facts["raw"]), "A ablation made no difference"
            assert set(map(int, np.unique(out))) <= {background, ink}
            assert np.any(out == ink)
            for cells in facts["windows"]:
                b = int(cells[:, 0].max())
                bottom_cols = cells[cells[:, 0] == b, 1]
                assert np.all(out[b, bottom_cols] == ink), "bottom slice did not stay fixed"
            if split == "train":
                training.append(facts)
    assert any(f["single_evidence"] for f in training), "missing irregular single-window evidence"
    assert any(f["multi_evidence"] for f in training), "missing local-anchor evidence"
    assert {f["clipped"] for f in training} == {False, True}, "missing clipping/retention contrast"


def _puzzle_key(puzzle):
    return tuple((split, tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[split]))
                 for split in ("train", "test"))


def _manual_test():
    """A two-chamber expected answer, without wall/flood/projection helpers."""
    centers = [(2, 3), (2, 8), (2, 14), (7, 3), (7, 8), (11, 8), (11, 14)]
    inp = np.zeros((14, 19), dtype=int)
    for r, c in centers:
        inp[r - 1, c] = inp[r + 1, c] = inp[r, c - 1] = inp[r, c + 1] = 6
    expected = np.zeros_like(inp)
    for r in range(3, 7):
        for c in range(4, 8):
            expected[r, c - (6 - r)] = 6
    for r in range(3, 11):
        for c in range(9, 14):
            # The shared-wall diamond at (7,8) indents the right window.
            if (r, c) != (7, 9):
                expected[r, c - (10 - r)] = 6
    before = inp.copy()
    assert np.array_equal(solve(inp), expected)
    assert np.array_equal(inp, before), "solve mutated its input"
    # A color permutation preserves geometry and needs no learned answer color.
    permuted = np.where(inp == 6, 0, 8)
    assert np.array_equal(solve(permuted), np.where(expected == 6, 0, 8))


def self_test():
    _manual_test()
    shapes, layouts, palettes, signatures = set(), set(), set(), set()
    region_counts, center_counts, clips = set(), set(), {"train": set(), "test": set()}
    overlaps = unequal_anchors = irregular = varying_palettes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        signatures.add(_puzzle_key(puzzle))
        current_palettes = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                before = inp.copy()
                assert np.array_equal(solve(inp.copy()), pair["output"])
                assert np.array_equal(inp, before)
                f = _geometry_facts(inp)
                shapes.add(inp.shape)
                centers = _normalize(f["centers"])
                layouts.add(tuple(centers))
                palettes.add((f["background"], f["ink"]))
                current_palettes.add((f["background"], f["ink"]))
                region_counts.add(len(f["windows"]))
                center_counts.add(len(centers))
                clips[split].add(f["clipped"])
                unequal_anchors += len({int(a[:, 0].max()) for a in f["windows"]}) > 1
                irregular += not np.array_equal(f["output"], _projection(
                    inp.shape, f["windows"], f["background"], f["ink"], boxes=True))
                survivors = sum(int(np.count_nonzero(a[:, 1] - (int(a[:, 0].max()) - a[:, 0]) >= 0))
                                for a in f["windows"])
                overlaps += survivors > int(np.count_nonzero(pair["output"] == f["ink"]))
        varying_palettes += len(current_palettes) > 1
    assert len(signatures) == 42
    assert len(shapes) >= 40 and len(layouts) >= 100 and len(palettes) >= 25
    assert {1, 2, 3, 4}.issubset(region_counts) and len(center_counts) >= 8
    assert clips["train"] == clips["test"] == {False, True}
    assert unequal_anchors >= 42 and irregular >= 42 and varying_palettes >= 10
    assert overlaps > 0
    count_calls = 0
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(10000 + 10 * a + b, a, b)
            assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
            validate_generated_puzzle(puzzle)
            count_calls += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, "1")):
        try:
            generate_puzzle(42, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count accepted: {a!r}, {b!r}")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shapes": len(shapes),
            "normalized_layouts": len(layouts), "palettes": len(palettes),
            "window_counts": sorted(region_counts), "center_counts": sorted(center_counts),
            "unequal_anchor_pairs": unequal_anchors, "irregular_projection_pairs": irregular,
            "overlap_pairs": overlaps, "changing_palette_puzzles": varying_palettes,
            "legal_count_calls": count_calls}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
