"""Arrow-projected sources, Manhattan ownership, even Chebyshev contours.

Only NumPy and the standard library are used. Nothing is read or written by the
public functions. The fixed five-step projection and all color roles are visible
or family-wide, so solve needs just one input grid.
"""
from __future__ import annotations

import hashlib
import numbers

import numpy as np


_DIRECTIONS = ((-1, 0), (-1, 1), (0, 1), (1, 1),
               (1, 0), (1, -1), (0, -1), (-1, -1))
_STEP = 5
_ALTERNATIVES = ("tip_sources", "chebyshev_owners", "manhattan_rings",
                 "filled_territories", "paint_ties", "odd_rings",
                 "rings_before_ownership", "keep_arrows", "endpoints_only")


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((grid >= 0) & (grid <= 9))):
        raise ValueError("grids must be integer 2D arrays, sides 1..30, colors 0..9")
    return grid.astype(np.int8, copy=False)


def _cells(row, col, dr, dc):
    if dr == 0:
        offsets = ((0, 0), (-1, -dc), (1, -dc))
    elif dc == 0:
        offsets = ((0, 0), (-dr, -1), (-dr, 1))
    else:
        offsets = ((0, 0), (-dr, 0), (0, -dc))
    return tuple((row + a, col + b) for a, b in offsets)


def _in_bounds(point, shape):
    return 0 <= point[0] < shape[0] and 0 <= point[1] < shape[1]


def _decode(input_grid):
    """Read separated monochrome V/L arrowheads, without generation metadata.

    An arrow is (tip_row, tip_col, direction_row, direction_col, ink_color).
    Eight-connectivity keeps the diagonal wings of cardinal arrows together.
    """
    grid = _grid(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    background = int(counts.argmax())
    if int(counts[background]) * 2 <= grid.size:
        raise ValueError("input background must occupy more than half the grid")
    remaining = {tuple(map(int, p)) for p in np.argwhere(grid != background)}
    arrows = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, todo = {start}, [start]
        while todo:
            row, col = todo.pop()
            neighbors = {(row + dr, col + dc) for dr in (-1, 0, 1)
                         for dc in (-1, 0, 1)} & remaining
            remaining.difference_update(neighbors)
            component.update(neighbors)
            todo.extend(neighbors)
        if len(component) != 3 or len({int(grid[p]) for p in component}) != 1:
            raise ValueError("foreground components must be isolated monochrome arrowheads")
        matches = [(r, c, dr, dc, int(grid[r, c]))
                   for r, c in sorted(component) for dr, dc in _DIRECTIONS
                   if set(_cells(r, c, dr, dc)) == component]
        if len(matches) != 1:
            raise ValueError("arrowhead must have exactly one tip and direction")
        arrows.append(matches[0])
    if not arrows:
        raise ValueError("at least one arrow is required")
    sources = _sources(arrows)
    if (len({(r, c) for r, c, _ in sources}) != len(sources)
            or not all(_in_bounds((r, c), grid.shape) for r, c, _ in sources)):
        raise ValueError("five-step endpoints must be distinct and inside the canvas")
    if any(int(grid[r, c]) != background for r, c, _ in sources):
        raise ValueError("endpoints must be background cells in the input")
    return grid, background, arrows


def _sources(arrows, step=_STEP):
    return [(r + step * dr, c + step * dc, color)
            for r, c, dr, dc, color in arrows]


def _geometry(shape, sources):
    rows, cols = np.indices(shape)
    row_offset = np.abs(rows[None, :, :] - np.array([s[0] for s in sources])[:, None, None])
    col_offset = np.abs(cols[None, :, :] - np.array([s[1] for s in sources])[:, None, None])
    return row_offset + col_offset, np.maximum(row_offset, col_offset)


def _field(shape, background, sources, alternative=None):
    """The default is the rule; named deviations are bounded teaching probes."""
    manhattan, chebyshev = _geometry(shape, sources)
    distances = chebyshev if alternative == "chebyshev_owners" else manhattan
    radii = manhattan if alternative == "manhattan_rings" else chebyshev
    even = radii % 2 == (1 if alternative == "odd_rings" else 0)
    if alternative == "rings_before_ownership":
        distances = np.where(even, distances, 10000)
    minimum = distances.min(axis=0)
    unique = (distances == minimum).sum(axis=0) == 1
    winner = distances.argmin(axis=0)
    if alternative == "paint_ties":
        unique = np.ones(shape, dtype=bool)
    answer = np.full(shape, background, dtype=np.int8)
    for index, (_, _, color) in enumerate(sources):
        paint = unique & (winner == index)
        if alternative != "filled_territories":
            paint &= even[index]
        answer[paint] = color
    return answer


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the entire answer from a single input; do not preserve the glyphs."""
    grid, background, arrows = _decode(input_grid)
    return _field(grid.shape, background, _sources(arrows))


def _probes(grid, background, arrows):
    sources = _sources(arrows)
    expected = _field(grid.shape, background, sources)
    rejected = set()
    for alternative in _ALTERNATIVES:
        if alternative == "tip_sources":
            other = _field(grid.shape, background, _sources(arrows, 0))
        elif alternative == "endpoints_only":
            other = np.full(grid.shape, background, dtype=np.int8)
            for row, col, color in sources:
                other[row, col] = color
        elif alternative == "keep_arrows":
            other = expected.copy()
            other[grid != background] = grid[grid != background]
        else:
            other = _field(grid.shape, background, sources, alternative)
        if not np.array_equal(expected, other):
            rejected.add(alternative)
    return rejected


def _tie_witness(shape, sources):
    manhattan, chebyshev = _geometry(shape, sources)
    nearest = manhattan == manhattan.min(axis=0)
    return bool(np.any((nearest.sum(axis=0) > 1)
                       & np.any(nearest & (chebyshev % 2 == 0), axis=0)))


def _shape(rng, low, high):
    shape = tuple(map(int, rng.integers(low, high + 1, size=2)))
    return (shape[0], shape[0]) if rng.random() < 0.27 else shape


def _sample_points(rng, shape, count, style, boundary):
    """Sample endpoint geometry, then reconstruct compatible arrows backward."""
    low = 0 if boundary else 1
    high = (shape[0] - low, shape[1] - low)
    if style in ("axis", "line"):
        axis = int(rng.integers(2))
        # Even pitch gives cell-wide bisectors, including a paintable tie.
        pitches = [p for p in range(4, 11, 2)
                   if (count - 1) * p < high[axis] - low]
        if not pitches:
            return None
        pitch = pitches[int(rng.integers(len(pitches)))]
        start = int(rng.integers(low, high[axis] - (count - 1) * pitch))
        fixed = int(rng.integers(low, high[1 - axis]))
        points = [(start + i * pitch, fixed) if axis == 0
                  else (fixed, start + i * pitch) for i in range(count)]
    elif style == "rectangle":
        rr = sorted(map(int, rng.choice(np.arange(low, high[0]), size=2, replace=False)))
        cc = sorted(map(int, rng.choice(np.arange(low, high[1]), size=2, replace=False)))
        if rr[1] - rr[0] < 4 or cc[1] - cc[0] < 4:
            return None
        points = [(rr[0], cc[0]), (rr[0], cc[1]), (rr[1], cc[1]), (rr[1], cc[0])]
    else:
        points = []
        for _ in range(120):
            point = (int(rng.integers(low, high[0])), int(rng.integers(low, high[1])))
            if all(max(abs(point[0] - r), abs(point[1] - c)) >= 3 for r, c in points):
                points.append(point)
            if len(points) == count:
                break
        if len(points) != count:
            return None
    if boundary:
        # Translation preserves distances and bisectors but puts a source at an
        # edge. Arrows are realized afterward, so they still fit the canvas.
        axis = int(rng.integers(2))
        edge = int(rng.integers(2))
        shift = (-min(p[axis] for p in points) if edge == 0
                 else shape[axis] - 1 - max(p[axis] for p in points))
        points = [tuple(p[a] + (shift if a == axis else 0) for a in (0, 1))
                  for p in points]
    return points


def _realize(rng, shape, points):
    """Finite search over at most seven arrows and eight orientations each.

    A one-cell king-neighbor blank gap separates motifs. Endpoints stay blank
    in the input. There is no ray obstruction rule or path drawing.
    """
    source_cells = set(points)
    options = []
    for sr, sc in points:
        candidates = []
        for dr, dc in _DIRECTIONS:
            r, c = sr - _STEP * dr, sc - _STEP * dc
            cells = _cells(r, c, dr, dc)
            if all(_in_bounds(p, shape) for p in cells) and not (source_cells & set(cells)):
                candidates.append((r, c, dr, dc))
        rng.shuffle(candidates)
        if not candidates:
            return None
        options.append(candidates)
    order = sorted(range(len(points)), key=lambda i: len(options[i]))
    selected = [None] * len(points)
    budget = [600]

    def search(depth, occupied):
        if depth == len(order):
            return True
        index = order[depth]
        for arrow in options[index]:
            budget[0] -= 1
            if budget[0] < 0:
                return False
            cells = _cells(*arrow)
            if any((r + a, c + b) in occupied for r, c in cells
                   for a in (-1, 0, 1) for b in (-1, 0, 1)):
                continue
            selected[index] = arrow
            if search(depth + 1, occupied | set(cells)):
                return True
        return False

    return selected if search(0, set()) else None


def _ink(rng, background, count, teaching):
    palette = [c for c in range(10) if c != background]
    if teaching:
        return list(map(int, rng.choice(palette, size=count, replace=False)))
    mode = int(rng.integers(5))
    if mode == 0:
        return [int(rng.choice(palette))] * count
    if mode == 1:
        size = min(count, 2)
        subset = rng.choice(palette, size=size, replace=False)
        return list(map(int, rng.choice(subset, size=count, replace=True)))
    return list(map(int, rng.choice(palette, size=count, replace=False)))


def _neutral_key(grid, background, arrows):
    # Color-independent exact geometry, including arrow orientation.
    return grid.shape, tuple(sorted(a[:4] for a in arrows))


def _make_pair(rng, background, count, style, low, high, boundary, teaching, seen):
    # All retries are in memory; no candidate grids or traces are saved.
    for _ in range(180):
        shape = _shape(rng, low, high)
        points = _sample_points(rng, shape, count, style, boundary)
        if points is None:
            continue
        bare = _realize(rng, shape, points)
        if bare is None:
            continue
        colors = _ink(rng, background, count, teaching == "asymmetric")
        if teaching == "ties" and rng.random() < 0.35:
            colors = [colors[0]] * count
        grid = np.full(shape, background, dtype=np.int8)
        arrows = [tuple(arrow) + (color,) for arrow, color in zip(bare, colors)]
        for arrow in arrows:
            for point in _cells(*arrow[:4]):
                grid[point] = arrow[4]
        key = _neutral_key(grid, background, arrows)
        if key in seen:
            continue
        if teaching == "ties" and not _tie_witness(shape, _sources(arrows)):
            continue
        if teaching == "asymmetric":
            # Metric differences must be visible in the actual output, not just
            # in a theoretical nearest-source map at an unpainted cell.
            needed = {"chebyshev_owners", "manhattan_rings", "rings_before_ownership"}
            if not needed <= _probes(grid, background, arrows):
                continue
        seen.add(key)
        return {"input": grid, "output": _field(shape, background, _sources(arrows))}
    raise RuntimeError("bounded construction exhausted; no ambiguous fallback is used")


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral)
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    shift = ("edge", "count", "scale")[int(rng.integers(3))]
    background_mode = int(rng.integers(3))
    base = 0 if background_mode == 0 else int(rng.integers(1, 10))
    other = int(rng.choice([c for c in range(10) if c != base]))
    seen = set()
    train = []
    # Two logical anchors, later shuffled: a visible paintable tie and an
    # asymmetric competition case separating the two distance metrics.
    for index in range(num_train):
        background = other if background_mode == 2 and index % 2 else base
        if index == 0:
            count, style, teaching = 2, "axis", "ties"
        elif index == 1:
            count, style, teaching = 3, "scatter", "asymmetric"
        else:
            count = int(rng.integers(1, 4 if shift == "count" else 7))
            styles = ["scatter"]
            if 2 <= count <= 4:
                styles.append("line")
            if count == 4:
                styles.append("rectangle")
            style, teaching = styles[int(rng.integers(len(styles)))], None
        train.append(_make_pair(rng, background, count, style, 12, 22,
                                False, teaching, seen))
    rng.shuffle(train)
    backgrounds = [int(np.bincount(p["input"].ravel(), minlength=10).argmax()) for p in train]
    single_taught = any(len(_decode(p["input"])[2]) == 1 for p in train)
    test = []
    for _ in range(num_test):
        count = int(rng.integers(4, 8)) if shift == "count" else int(rng.integers(1 if single_taught else 2, 8))
        styles = ["scatter"]
        if 2 <= count <= 4:
            styles.append("line")
        if count == 4:
            styles.append("rectangle")
        style = styles[int(rng.integers(len(styles)))]
        background = backgrounds[int(rng.integers(len(backgrounds)))]
        low, high = (24, 30) if shift == "scale" else (12, 28)
        test.append(_make_pair(rng, background, count, style, low, high,
                               shift == "edge", None, seen))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_pair(pair, seen_inputs, seen_geometry):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
    grid, background, arrows = _decode(pair["input"])
    answer = _grid(pair["output"])
    _require(grid.shape == answer.shape, "input/output dimensions must agree")
    key = (grid.shape, grid.tobytes())
    _require(key not in seen_inputs, "duplicate input")
    seen_inputs.add(key)
    key = _neutral_key(grid, background, arrows)
    _require(key not in seen_geometry, "geometry repeated with only recoloring")
    seen_geometry.add(key)
    sources = _sources(arrows)
    manhattan, chebyshev = _geometry(grid.shape, sources)
    minimum = manhattan.min(axis=0)
    nearest = manhattan == minimum
    unique = nearest.sum(axis=0) == 1
    paint = unique & np.any(nearest & (chebyshev % 2 == 0), axis=0)
    _require(np.all(answer[~paint] == background), "ties and odd radii must be background")
    for index, (row, col, color) in enumerate(sources):
        mask = unique & nearest[index] & (chebyshev[index] % 2 == 0)
        _require(np.all(answer[mask] == color), "even contour has incorrect owner color")
        _require(int(answer[row, col]) == color, "radius-zero endpoint source missing")
    _require(set(map(int, np.unique(answer))) == set(map(int, np.unique(grid))), "palette not preserved")
    _require(not np.array_equal(grid, answer), "identity pair")
    _require(np.array_equal(solve(grid.copy()), answer), "input-only solve disagrees")
    return grid, background, arrows


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen_inputs, seen_geometry = set(), set()
    records = {split: [_validate_pair(p, seen_inputs, seen_geometry) for p in puzzle[split]]
               for split in ("train", "test")}
    rejected = set()
    for grid, background, arrows in records["train"]:
        rejected.update(_probes(grid, background, arrows))
    _require(set(_ALTERNATIVES) <= rejected, "demonstrations fail to separate simpler alternatives")
    _require(any(_tie_witness(g.shape, _sources(a)) for g, _, a in records["train"]),
             "training must expose a tie that would otherwise be painted")
    _require(any(len(a) >= 3 and len({x[4] for x in a}) >= 3 for _, _, a in records["train"]),
             "training must show competing colors and multiple boundaries")
    # Recover the fixed distance from training labels within a stated finite
    # signed-distance hypothesis class. No query output participates in fitting.
    for step in range(-29, 30):
        if step == _STEP:
            continue
        fits = True
        for grid, background, arrows in records["train"]:
            sources = _sources(arrows, step)
            if (not all(_in_bounds(s[:2], grid.shape) for s in sources)
                    or len({s[:2] for s in sources}) != len(sources)
                    or not np.array_equal(_field(grid.shape, background, sources),
                                          _field(grid.shape, background, _sources(arrows)))):
                fits = False
                break
        _require(not fits, f"projection length {step} remains indistinguishable")
    trained_backgrounds = {b for _, b, _ in records["train"]}
    _require(all(b in trained_backgrounds for _, b, _ in records["test"]), "query background not demonstrated")
    single_taught = any(len(a) == 1 for _, _, a in records["train"])
    _require(all(len(a) != 1 or single_taught for _, _, a in records["test"]),
             "single-source query requires a single-source demonstration")
    # Check the observable shift, without seed or hidden shift metadata.
    all_train_interior = all(all(0 < r < g.shape[0] - 1 and 0 < c < g.shape[1] - 1
                                for r, c, _ in _sources(a)) for g, _, a in records["train"])
    _require(all_train_interior, "generated training endpoints must be interior")
    max_count = max(len(a) for _, _, a in records["train"])
    max_side = max(max(g.shape) for g, _, _ in records["train"])
    for grid, _, arrows in records["test"]:
        edge = any(r in (0, grid.shape[0] - 1) or c in (0, grid.shape[1] - 1)
                   for r, c, _ in _sources(arrows))
        _require(edge or len(arrows) > max_count or min(grid.shape) > max_side,
                 "query needs a boundary, source-count, or scale shift")


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def _regressions():
    # Two same-colored sources: source identity, not color identity, decides ties.
    inp = np.zeros((17, 20), dtype=np.int8)
    arrows = [(11, 4, -1, 0, 2), (11, 12, -1, 0, 2)]
    for arrow in arrows:
        for p in _cells(*arrow[:4]):
            inp[p] = arrow[4]
    out = solve(inp)
    _require(out[6, 4] == out[6, 12] == 2, "five-step endpoints lost")
    _require(out[6, 8] == 0, "same-color Manhattan tie must be blank")
    _require(out[8, 4] == 2 and out[7, 4] == 0, "even/odd Chebyshev radii confused")
    _require(out[2, 4] == 2 and out[0, 4] == 2, "contours do not clip at edge")
    _require(out[11, 4] == 0, "tip incorrectly preserved")
    # Exact asymmetric witness: L1 picks (2,8), L-infinity picks (4,4).
    sources = [(2, 8, 2), (4, 4, 3)]
    _require(_field((15, 15), 0, sources)[8, 8] == 2, "Manhattan ownership lost")
    _require(_field((15, 15), 0, sources, "chebyshev_owners")[8, 8] == 3,
             "bad metric witness")
    # All eight geometries have one unique interpretation, with source radius0.
    for dr, dc in _DIRECTIONS:
        inp = np.full((19, 21), 7, dtype=np.int8)
        for p in _cells(9, 10, dr, dc):
            inp[p] = 0
        decoded = _decode(inp)[2]
        _require(decoded == [(9, 10, dr, dc, 0)], "arrow direction decoded incorrectly")
        _require(solve(inp)[9 + 5 * dr, 10 + 5 * dc] == 0, "wrong projected source")
    return out


def self_test() -> dict:
    """42 seeds, streamed in memory; no files and no retained puzzle corpus."""
    _regressions()
    shapes, counts, directions, backgrounds, signatures, neutral = set(), set(), set(), set(), set(), set()
    repeated_colors = edge_sources = single_sources = 0
    shifts = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_digest(puzzle))
        max_train_count = max(len(_decode(p["input"])[2]) for p in puzzle["train"])
        max_train_side = max(max(p["input"].shape) for p in puzzle["train"])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, background, arrows = _decode(pair["input"])
                shapes.add(grid.shape)
                counts.add(len(arrows))
                directions.update(a[2:4] for a in arrows)
                backgrounds.add(background)
                neutral.add(_neutral_key(grid, background, arrows))
                repeated_colors += len({a[4] for a in arrows}) < len(arrows)
                single_sources += len(arrows) == 1
                edge = any(r in (0, grid.shape[0] - 1) or c in (0, grid.shape[1] - 1)
                           for r, c, _ in _sources(arrows))
                edge_sources += edge
                if split == "test":
                    if edge:
                        shifts.add("edge")
                    if len(arrows) > max_train_count:
                        shifts.add("count")
                    if min(grid.shape) > max_train_side:
                        shifts.add("scale")
                # Spatial and palette covariance exercise the input-only rule.
                for transformed_input, transformed_output in (
                        (np.rot90(grid), np.rot90(pair["output"])),
                        (np.fliplr(grid), np.fliplr(pair["output"])),
                        ((grid + 3) % 10, (pair["output"] + 3) % 10)):
                    _require(np.array_equal(solve(transformed_input), transformed_output),
                             "rotation/reflection/color covariance failed")
    _require(len(signatures) == 42, "seeds did not produce 42 distinct puzzles")
    _require(len(shapes) >= 65 and len(neutral) >= 180, "geometry variation too narrow")
    _require(set(range(1, 8)) <= counts, "source counts 1..7 not covered")
    _require(directions == set(_DIRECTIONS) and len(backgrounds) >= 8, "role/direction variation missing")
    _require(repeated_colors >= 25 and edge_sources >= 10 and single_sources >= 5,
             "repeated inks, boundary sources, or simple cases missing")
    _require(shifts == {"edge", "count", "scale"}, "shift modes missing")
    first = generate_puzzle(314159, 7, 3)
    _require(_puzzle_digest(first) == _puzzle_digest(generate_puzzle(314159, 7, 3)), "seed is not deterministic")
    # Mutated source and tie pixels must be rejected by the semantic validator.
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    grid, bg, arrows = _decode(broken["test"][0]["input"])
    row, col, color = _sources(arrows)[0]
    broken["test"][0]["output"][row, col] = bg
    try:
        validate_generated_puzzle(broken)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("incorrect source pixel accepted")
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    for pair in broken["train"]:
        grid, bg, arrows = _decode(pair["input"])
        manhattan, chebyshev = _geometry(grid.shape, _sources(arrows))
        nearest = manhattan == manhattan.min(axis=0)
        witnesses = np.argwhere((nearest.sum(axis=0) > 1)
                                & np.any(nearest & (chebyshev % 2 == 0), axis=0))
        if witnesses.size:
            row, col = map(int, witnesses[0])
            pair["output"][row, col] = arrows[0][4]
            break
    else:
        raise AssertionError("missing mutation fixture for a paintable tie")
    try:
        validate_generated_puzzle(broken)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("painted tie pixel accepted")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(2026, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "grid_shapes": len(shapes),
            "arrow_geometries": len(neutral), "source_counts": sorted(counts),
            "directions": len(directions), "backgrounds": len(backgrounds),
            "repeated_color_pairs": repeated_colors, "edge_source_pairs": edge_sources,
            "single_source_pairs": single_sources, "shift_modes": sorted(shifts)}
