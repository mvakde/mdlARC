"""Complete odd endpoint spans, then shear the complete scaffold left.

Color 5 is the family-wide midpoint separator. All other roles are local.
Generation uses explicit completed cells; solve uses completed-row slicing.
Nothing is loaded from disk, and generation/self_test save no artifacts.
"""

from __future__ import annotations

import numpy as np


SEPARATOR = 5
_ALTERNATIVES = (
    "no_A", "no_B", "no_separator", "rank_distance", "canvas_bottom",
    "rightward", "double_rate", "separate_runs", "shear_sparse_first",
    "recenter_after_clipping", "rigid_translation",
)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or any(not 1 <= n <= 30 for n in array.shape)
            or not np.issubdtype(array.dtype, np.integer)
            or np.any((array < 0) | (array > 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer ARC grid")
    return array


def _parse(value):
    """The strict-majority background is visible even on nonblack canvases."""
    grid = _grid(value)
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    if int(counts.max()) * 2 <= grid.size or background == SEPARATOR:
        raise ValueError("Background must be a strict majority and differ from 5")
    if np.any(grid == SEPARATOR):
        raise ValueError("Separator 5 must be absent from endpoint inputs")
    spans = []
    for row in range(grid.shape[0]):
        columns = np.flatnonzero(grid[row] != background)
        if len(columns) == 0:
            continue
        if len(columns) != 2:
            raise ValueError("Every occupied row needs exactly two endpoints")
        left, right = map(int, columns)
        a, b = int(grid[row, left]), int(grid[row, right])
        if right - left < 2 or (right - left) % 2 or a == b:
            raise ValueError("Distinct endpoints must define an odd inclusive span")
        spans.append((row, left, right, a, b))
    if not spans:
        raise ValueError("At least one endpoint pair is required")
    return grid, background, spans


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the background, endpoints and lowest occupied row from input only."""
    grid, background, spans = _parse(input_grid)
    output = np.full(grid.shape, background, dtype=np.int8)
    bottom = spans[-1][0]
    width = grid.shape[1]
    for row, left, right, a, b in spans:
        midpoint = (left + right) // 2
        completed = np.full(width, background, dtype=np.int8)
        completed[left:midpoint] = a
        completed[midpoint] = SEPARATOR
        completed[midpoint + 1:right + 1] = b
        distance = bottom - row
        if distance < width:
            output[row, :width - distance] = completed[distance:]
    return output


def _render(shape, background, spans):
    """Construction oracle: paint each original cell before shifting/clipping."""
    output = np.full(shape, background, dtype=np.int8)
    bottom = spans[-1][0]
    for row, left, right, a, b in spans:
        middle = (left + right) // 2
        for column in range(left, right + 1):
            destination = column - (bottom - row)
            if destination >= 0:
                output[row, destination] = (
                    a if column < middle else SEPARATOR if column == middle else b)
    return output


def _checked_integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"Illegal {name}: {value}")
    return value


def _positions(rng, count, extent):
    if count == extent + 1:
        return list(range(count))
    middle = sorted(map(int, rng.choice(np.arange(1, extent), count - 2, replace=False)))
    return [0, *middle, extent]


def _sample_geometry(rng, rows, retained=False, teaching=False):
    """Aligned bars, variable radii, staggered bars, free spans, or full spans."""
    extent = rows[-1] - rows[0]
    minimum = extent + (7 if teaching else 3) if retained else 5
    width = int(rng.integers(minimum, 31))
    kind = int(rng.integers(0, 4 if retained else 5))
    half_min = 2 if teaching else 1
    padding = 1 if teaching else 0
    maximum_left = extent + padding if retained else 0
    max_half = (width - 1 - padding - maximum_left) // 2
    common_half = int(rng.integers(half_min, max_half + 1))
    base = int(rng.integers(maximum_left, width - padding - 2 * common_half))
    center = int(rng.integers(maximum_left + half_min, width - padding - half_min))
    intervals = []
    slope = int(rng.choice((-1, 1)))
    for index, row in enumerate(rows):
        lower = rows[-1] - row + padding if retained else 0
        if kind == 0:
            left, half = base, common_half
        elif kind == 1:
            half = int(rng.integers(half_min, min(center - lower, width - 1 - padding - center) + 1))
            left = center - half
        elif kind == 2:
            half = common_half
            left = int(np.clip(base + slope * index, lower, width - 1 - padding - 2 * half))
        elif kind == 3:
            half = int(rng.integers(half_min, (width - 1 - padding - lower) // 2 + 1))
            left = int(rng.integers(lower, width - padding - 2 * half))
        else:
            half = (width - 1) // 2
            left = int(rng.integers(width - 2 * half))
        intervals.append((left, left + 2 * half))
    return width, intervals


def _pair(rng, background, palette, relative_rows, width, intervals, bottom_padding=None):
    top_padding = int(rng.integers(0, min(6, 30 - relative_rows[-1])))
    if bottom_padding is None:
        bottom_padding = int(rng.integers(0, min(7, 30 - top_padding - relative_rows[-1])))
    height = top_padding + relative_rows[-1] + 1 + bottom_padding
    # Canvas padding is sampled independently of occupied-row spacing.
    height = min(30, height)
    grid = np.full((height, width), background, dtype=np.int8)
    legal = [c for c in palette if c != background]
    prior = None
    spans = []
    for relative, (left, right) in zip(relative_rows, intervals):
        if prior is not None and rng.random() < 0.48:
            a, b = prior if rng.random() < 0.6 else prior[::-1]
        else:
            a, b = map(int, rng.choice(legal, 2, replace=False))
        prior = (a, b)
        row = top_padding + relative
        grid[row, left], grid[row, right] = a, b
        spans.append((row, left, right, a, b))
    return {"input": grid, "output": _render(grid.shape, background, spans)}


def _retained_demo(rng, background, palette):
    first, second = int(rng.integers(1, 4)), int(rng.integers(2, 4))
    rows = [0, first, first + second]
    width, intervals = _sample_geometry(rng, rows, retained=True, teaching=True)
    return _pair(rng, background, palette, rows, width, intervals, int(rng.integers(1, 6)))


def _clipped_demo(rng, background, palette):
    middle_distance = int(rng.integers(3, 6))
    top_distance = middle_distance + int(rng.integers(1, 9 - middle_distance))
    rows = [0, top_distance - middle_distance, top_distance]
    # Top: lose the left endpoint but retain part of its completed territory.
    half = int(rng.integers(3, 7))
    loss = int(rng.integers(1, min(half - 1, top_distance) + 1))
    left = top_distance - loss
    top_span = (left, left + 2 * half)
    # Middle: the seam is off-canvas; a right-colored fragment still survives.
    middle_half = int(rng.integers((middle_distance + 1) // 2, middle_distance))
    middle_span = (0, 2 * middle_half)
    bottom_half = int(rng.integers(2, 7))
    bottom_left = int(rng.integers(0, 8))
    bottom_span = (bottom_left, bottom_left + 2 * bottom_half)
    minimum_width = max(b for _, b in (top_span, middle_span, bottom_span)) + 1
    width = int(rng.integers(minimum_width, 31))
    return _pair(rng, background, palette, rows, width,
                 [top_span, middle_span, bottom_span], int(rng.integers(1, 6)))


def _key(grid):
    array = np.asarray(grid, dtype=np.int8)
    return array.shape, array.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _checked_integer(seed, "seed", 0)
    num_train = _checked_integer(num_train, "num_train", 2, 10)
    num_test = _checked_integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    colors = [c for c in range(10) if c != SEPARATOR]
    fixed_background = int(rng.choice(colors)) if rng.random() < 0.5 else None
    palette = list(map(int, rng.choice(colors, int(rng.integers(3, 10)), replace=False)))
    shift_mode = str(rng.choice(("more_rows", "greater_distance")))
    seen = set()

    def unique(make):
        for _ in range(32):
            bg = fixed_background if fixed_background is not None else int(rng.choice(colors))
            candidate = make(bg)
            signature = _key(candidate["input"])
            if signature not in seen:
                seen.add(signature)
                return candidate
        raise RuntimeError("Distinct-input construction exhausted 32 attempts")

    train = [unique(lambda bg: _retained_demo(rng, bg, palette)),
             unique(lambda bg: _clipped_demo(rng, bg, palette))]
    for _ in range(num_train - 2):
        count = int(rng.integers(2, 5))
        extent = int(rng.integers(count - 1, 9))
        rows = _positions(rng, count, extent)
        retained = rng.random() < 0.35
        width, intervals = _sample_geometry(rng, rows, retained)
        train.append(unique(lambda bg: _pair(rng, bg, palette, rows, width, intervals)))
    rng.shuffle(train)
    max_train_extent = max(_parse(p["input"])[2][-1][0] - _parse(p["input"])[2][0][0] for p in train)
    test = []
    for _ in range(num_test):
        if shift_mode == "more_rows":
            count = int(rng.integers(5, 9))
            extent = int(rng.integers(count - 1, min(23, count + 8)))
        else:
            extent = max_train_extent + int(rng.integers(1, 8))
            count = int(rng.integers(2, min(7, extent + 1) + 1))
        rows = _positions(rng, count, extent)
        retained = rng.random() < 0.3
        width, intervals = _sample_geometry(rng, rows, retained)
        test.append(unique(lambda bg: _pair(rng, bg, palette, rows, width, intervals)))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative(value, name):
    """Finite, explicit shortcut falsification; this is not a reference solver."""
    grid, bg, spans = _parse(value)
    height, width = grid.shape
    output = np.full(grid.shape, bg, dtype=np.int8)
    bottom = spans[-1][0]
    for index, (row, left, right, a, b) in enumerate(spans):
        distance = bottom - row
        if name == "no_B":
            distance = 0
        elif name == "rank_distance":
            distance = len(spans) - 1 - index
        elif name == "canvas_bottom":
            distance = height - 1 - row
        elif name == "rightward":
            distance = -distance
        elif name == "double_rate":
            distance *= 2
        elif name == "rigid_translation":
            distance = bottom - spans[0][0]
        elif name == "separate_runs":
            end = index
            while end + 1 < len(spans) and spans[end + 1][0] == spans[end][0] + 1:
                end += 1
            distance = spans[end][0] - row
        middle = (left + right) // 2
        if name == "shear_sparse_first":
            survivors = [(column - distance, color) for column, color in ((left, a), (right, b))
                         if 0 <= column - distance < width]
            if len(survivors) == 2:
                lo, hi = survivors[0][0], survivors[1][0]
                mid = (lo + hi) // 2
                output[row, lo:mid] = a
                output[row, mid] = SEPARATOR
                output[row, mid + 1:hi + 1] = b
            else:
                for column, color in survivors:
                    output[row, column] = color
            continue
        if name == "recenter_after_clipping":
            lo, hi = max(0, left - distance), right - distance
            if hi >= lo:
                mid = (lo + hi) // 2
                output[row, lo:mid] = a
                output[row, mid] = SEPARATOR
                output[row, mid + 1:hi + 1] = b
            continue
        for column in range(left, right + 1):
            if name == "no_A" and column not in (left, right):
                continue
            destination = column - distance
            if 0 <= destination < width:
                color = a if column < middle else SEPARATOR if column == middle else b
                if name == "no_separator" and column == middle:
                    color = a
                output[row, destination] = color
    return output


def _inspect(pair):
    grid, bg, spans = _parse(pair["input"])
    output = _grid(pair["output"])
    _require(grid.shape == output.shape, "Canvas dimensions changed")
    _require(np.array_equal(output, _render(grid.shape, bg, spans)), "Wrong completed-cell shear")
    _require(np.array_equal(output, solve(grid)), "Input-only solver disagrees")
    _require(not np.array_equal(grid, output), "Transformation is trivial")
    bottom = spans[-1][0]
    fates = []
    losses = 0
    for row, left, right, a, b in spans:
        middle = (left + right) // 2
        distance = bottom - row
        nleft = max(0, middle - max(left, distance))
        nright = max(0, right - max(middle + 1, distance) + 1)
        nseam = int(middle >= distance)
        _require(int(np.count_nonzero(output[row] == a)) == nleft, "Left territory changed")
        _require(int(np.count_nonzero(output[row] == b)) == nright, "Right territory changed")
        _require(int(np.count_nonzero(output[row] == SEPARATOR)) == nseam, "Wrong seam survival")
        _require(int(np.count_nonzero(output[row] != bg)) == nleft + nright + nseam,
                 "Unexpected foreground in a row")
        losses += right - left + 1 - nleft - nright - nseam
        fates.append("vanished" if distance > right else "seam_lost" if distance > middle
                     else "left_clipped" if distance > left else "retained")
    for name in ("no_A", "no_B", "no_separator"):
        _require(not np.array_equal(output, _alternative(grid, name)), f"Parent ablation {name} has no effect")
    return {"background": bg, "spans": spans, "count": len(spans),
            "extent": bottom - spans[0][0], "losses": losses, "fates": fates}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "Splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "Illegal pair counts")
    seen = set()
    facts = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Wrong pair keys")
            signature = _key(_grid(pair["input"]))
            _require(signature not in seen, "Duplicate input")
            seen.add(signature)
            facts[split].append(_inspect(pair))
    train = facts["train"]
    _require(all(info["count"] >= 2 for info in train + facts["test"]), "Generated scaffold needs multiple rows")
    _require(any(info["losses"] == 0 for info in train), "No retained training demonstration")
    _require(any("left_clipped" in info["fates"] and "seam_lost" in info["fates"] for info in train),
             "Training must expose completion before clipping and seam loss")
    _require(all(any(not np.array_equal(p["output"], _alternative(p["input"], name))
                     for p in puzzle["train"]) for name in _ALTERNATIVES),
             "Training fails to distinguish a documented shortcut")
    more_rows = all(info["count"] > max(t["count"] for t in train) for info in facts["test"])
    more_distance = all(info["extent"] > max(t["extent"] for t in train) for info in facts["test"])
    _require(more_rows or more_distance, "Missing structural train/test shift")


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(p[k], q[k]) for p, q in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test():
    """Bounded streaming checks: 42 seeds, 30 legal count pairs, no saved grids."""
    shapes, layouts, backgrounds, endpoints, counts, fates, shifts, aspects = (set() for _ in range(8))
    signatures = set()
    fixed_bg = variable_bg = gap_pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        _require(_same(puzzle, generate_puzzle(seed)), "Seed is nondeterministic")
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        local_backgrounds = set()
        train = [_inspect(p) for p in puzzle["train"]]
        tests = [_inspect(p) for p in puzzle["test"]]
        if all(t["count"] > max(x["count"] for x in train) for t in tests):
            shifts.add("more_rows")
        if all(t["extent"] > max(x["extent"] for x in train) for t in tests):
            shifts.add("greater_distance")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _inspect(pair)
                grid = pair["input"]
                shapes.add(grid.shape)
                layouts.add((grid.shape, tuple(s[:3] for s in info["spans"])))
                backgrounds.add(info["background"])
                local_backgrounds.add(info["background"])
                endpoints.update(c for s in info["spans"] for c in s[3:])
                counts.add(info["count"])
                fates.update(info["fates"])
                rows = [s[0] for s in info["spans"]]
                gap_pairs += any(y - x > 1 for x, y in zip(rows, rows[1:]))
                h, w = grid.shape
                aspects.add("square" if h == w else "tall" if h > w else "wide")
        fixed_bg += len(local_backgrounds) == 1
        variable_bg += len(local_backgrounds) > 1
    _require(len(signatures) == 42 and len(shapes) >= 60 and len(layouts) >= 150,
             "Insufficient geometric variation across 42 seeds")
    _require(backgrounds == endpoints == set(range(10)) - {SEPARATOR}, "Missing palette role")
    _require(len(counts) >= 6 and gap_pairs >= 100, "Missing row-count or gap variation")
    _require(fates == {"retained", "left_clipped", "seam_lost", "vanished"}, "Missing clipping regimes")
    _require(shifts == {"more_rows", "greater_distance"}, "Missing train/test regime")
    _require(aspects == {"square", "tall", "wide"} and fixed_bg > 0 and variable_bg > 0,
             "Missing canvas/palette regimes")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(20261023 + 10 * a + b, a, b)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (a, b), "Counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(1, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted illegal counts {a}/{b}")
    # Solver domain includes degenerate one-row cases and arbitrary distance,
    # independent of the teaching sampler's more restricted layouts.
    for background in (0, 7):
        grid = np.full((30, 7), background, dtype=np.int8)
        grid[0, 1], grid[0, 5] = 2, 9
        grid[29, 2], grid[29, 4] = 8, 3
        out = solve(grid)
        _require(np.all(out[0] == background), "Far-away row did not vanish")
        _require(out[29, 3] == SEPARATOR, "Anchor midpoint is wrong")
        single = grid[29:30].copy()
        _require(np.array_equal(solve(single), out[29:30]), "Single-row anchor failed")
    return {"seeds": 42, "shapes": len(shapes), "layouts": len(layouts),
            "background_colors": len(backgrounds), "endpoint_colors": len(endpoints),
            "row_counts": sorted(counts), "clipping_regimes": sorted(fates),
            "fixed_background_puzzles": fixed_bg, "variable_background_puzzles": variable_bg,
            "legal_count_combinations": 30, "independent_solver": False}
