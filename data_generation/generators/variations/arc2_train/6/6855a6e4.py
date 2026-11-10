"""Synthetic local mirror puzzles with two facing brackets."""

import hashlib
import numpy as np


_COUNT_OPTIONS = ((2, 1), (3, 1), (4, 2), (5, 3), (6, 1),
                  (7, 2), (8, 3), (9, 4), (10, 4))
_SOURCE_INPUT_DIGESTS = {
    "9a7fca05782873979d454e83f31518fa5dc6560e8d6d91f507074089c6a89e8e",
    "5b9646ccb43cd9d87a87ba88f75264fd55c3cc12a7714dd3cb204b10f39a662e",
    "be6b28f45cc1cb64f14f2e1aeedefba6fe14dd6882df7d39dc2666e7b9c03a52",
    "7cfc4d2f0549f8c7659eaf11ce5e59435d657989055f66e8adb91b1c6508d7b7",
}


def _count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not (2 <= value <= 10 if name == "num_train" else 1 <= value <= 4):
        raise ValueError(f"{name} out of range")
    return value


def _choose_counts(rng, seed, num_train, num_test):
    n = _count(num_train, "num_train")
    m = _count(num_test, "num_test")
    if n is None and m is None:
        return _COUNT_OPTIONS[seed % len(_COUNT_OPTIONS)]
    if n is None:
        n = int(rng.integers(max(2, m + 1), 11))
    if m is None:
        m = int(rng.integers(1, min(4, n - 1) + 1))
    if n <= m:
        raise ValueError("num_train must exceed num_test")
    return n, m


def _connected(cells):
    cells = set(cells)
    if not cells:
        return False
    seen = {next(iter(cells))}
    todo = list(seen)
    while todo:
        d, p = todo.pop()
        for q in ((d - 1, p), (d + 1, p), (d, p - 1), (d, p + 1)):
            if q in cells and q not in seen:
                seen.add(q)
                todo.append(q)
    return len(seen) == len(cells)


def _has_hole(cells):
    cells = set(cells)
    for d, p in cells:
        center = d + 1, p + 1
        if (center not in cells
                and (d, p + 1) in cells
                and (d + 2, p + 1) in cells
                and (d + 1, p) in cells
                and (d + 1, p + 2) in cells):
            return True
    return False


def _motif(rng, length, allowed):
    """Choose one side's depths, size, and topology independently."""
    layers = int(rng.integers(1, 4 if allowed["three_layer"] else 3))
    first = int(rng.integers(1 if allowed["depth1"] else 2, 7 - layers))
    depths = range(first, first + layers)
    span = length - 2
    width = int(rng.integers(2 if not allowed["depth1"] else 1, span + 1))
    left = int(rng.integers(1, span - width + 2))
    region = {(d, p) for d in depths for p in range(left, left + width)}
    if allowed["hole"] and layers == 3 and width >= 3 and rng.random() < 0.2:
        ring_left = int(rng.integers(left, left + width - 2))
        return tuple(sorted((d, p) for d in depths
                            for p in range(ring_left, ring_left + 3)
                            if (d, p) != (first + 1, ring_left + 1)))
    scattered = allowed["disconnected"] and rng.random() < 0.32
    if scattered:
        count = int(rng.integers(layers, min(len(region), 16) + 1))
        cells = {(d, int(rng.integers(left, left + width))) for d in depths}
        remaining = sorted(region - cells)
        if count > len(cells):
            chosen = rng.choice(len(remaining), size=count - len(cells), replace=False)
            cells.update(remaining[int(i)] for i in chosen)
    else:
        cells = {(first, int(rng.integers(left, left + width)))}
        while {d for d, _ in cells} != set(depths):
            next_d = max(d for d, _ in cells) + 1
            previous = [p for d, p in cells if d == next_d - 1]
            cells.add((next_d, int(rng.choice(previous))))
        minimum = max(len(cells), 2 if not allowed["depth1"] else 1)
        target = int(rng.integers(minimum, min(len(region), 16) + 1))
        while len(cells) < target:
            frontier = sorted(q for q in region - cells if any(
                (q[0] + dd, q[1] + dp) in cells
                for dd, dp in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ))
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(len(frontier)))])
    return tuple(sorted(cells))


def _make_pair(rng, orientation, palette, variable_palette, allowed,
               regime, seen_lengths=()):
    if variable_palette:
        palette = tuple(int(x) for x in rng.choice(10, size=4, replace=False))
    bg, structural, primary, secondary = palette
    if regime == "target_test" and seen_lengths and rng.random() < 0.48:
        length1 = int(rng.choice(seen_lengths))
    else:
        length1 = int(rng.integers(5, 12 if regime == "source_train" else 14))
    delta = int(rng.integers(-2, 3)) if allowed["unequal"] else 0
    length2 = min(13, max(5, length1 + delta))
    offset = int(rng.integers(-2, 3)) if allowed["offset"] else 0
    motif1 = _motif(rng, length1, allowed)
    motif2 = _motif(rng, length2, allowed)
    depth1 = max(d for d, _ in motif1)
    depth2 = max(d for d, _ in motif2)
    separation = depth1 + depth2 + 1 + int(rng.integers(0, 7))
    start1 = -min(0, offset)
    start2 = start1 + offset
    parallel_size = max(start1 + length1, start2 + length2)
    perpendicular_size = depth1 + separation + depth2 + 1
    assert parallel_size <= 30 and perpendicular_size <= 30
    if parallel_size <= 15 and perpendicular_size <= 15 and rng.random() < 0.13:
        width, height = 15, 15
        pad_left = int(rng.integers(0, 16 - parallel_size))
        pad_top = int(rng.integers(0, 16 - perpendicular_size))
    else:
        mode = int(rng.integers(0, 3))
        if mode == 0 and perpendicular_size < 30:
            # Back orientation does not determine the canvas aspect ratio.
            height = int(rng.integers(perpendicular_size,
                                      min(29, perpendicular_size + 6) + 1))
            least_width = max(parallel_size, height + 1)
            width = int(rng.integers(least_width, min(30, least_width + 6) + 1))
        elif mode == 1 and parallel_size < 30:
            width = int(rng.integers(parallel_size,
                                     min(29, parallel_size + 6) + 1))
            least_height = max(perpendicular_size, width + 1)
            height = int(rng.integers(least_height, min(30, least_height + 6) + 1))
        else:
            side = int(rng.integers(max(parallel_size, perpendicular_size),
                                    min(30, max(parallel_size, perpendicular_size) + 6) + 1))
            width = height = side
        pad_left = int(rng.integers(0, width - parallel_size + 1))
        pad_top = int(rng.integers(0, height - perpendicular_size + 1))
    start1 += pad_left
    start2 += pad_left
    axis1 = pad_top + depth1
    axis2 = axis1 + separation
    grid = np.full((height, width), bg, dtype=np.int8)
    for axis, start, length, inward in ((axis1, start1, length1, 1),
                                        (axis2, start2, length2, -1)):
        grid[axis, start:start + length] = structural
        grid[axis + inward, start] = structural
        grid[axis + inward, start + length - 1] = structural
    sources = []
    for axis, start, sign, motif in ((axis1, start1, -1, motif1),
                                     (axis2, start2, 1, motif2)):
        for depth, parallel in motif:
            sources.append(((axis + sign * depth, start + parallel),
                            (axis - sign * depth, start + parallel)))
    colored = allowed["multicolor"] and len(sources) >= 2 and rng.random() < 0.4
    values = [primary] * len(sources)
    if colored:
        values = [primary, secondary] + [
            primary if rng.random() < 0.5 else secondary
            for _ in range(len(sources) - 2)
        ]
        rng.shuffle(values)
    output = grid.copy()
    for (source, target), color in zip(sources, values):
        assert grid[source] == bg and grid[target] == bg
        grid[source] = color
        output[target] = color
    if orientation == "vertical":
        grid = grid.T.copy()
        output = output.T.copy()
    return {"input": grid, "output": output}


def _structure(grid, bg):
    """Infer complete facing brackets from one input."""
    for color in (int(x) for x in np.unique(grid) if x != bg):
        for orientation in ("horizontal", "vertical"):
            a = grid if orientation == "horizontal" else grid.T
            mask = a == color
            rows = np.flatnonzero(mask.sum(axis=1) >= 5)
            if len(rows) != 2:
                continue
            upper, lower = map(int, rows)
            if lower - upper < 3:
                continue
            bounds = []
            expected = np.zeros_like(mask)
            for row, hook in ((upper, 1), (lower, -1)):
                cols = np.flatnonzero(mask[row])
                if len(cols) < 5 or cols[-1] - cols[0] + 1 != len(cols):
                    break
                start, end = int(cols[0]), int(cols[-1])
                expected[row, start:end + 1] = True
                expected[row + hook, start] = True
                expected[row + hook, end] = True
                bounds.append((start, end))
            if len(bounds) == 2 and np.array_equal(mask, expected):
                return {"orientation": orientation, "color": color,
                        "axes": (upper, lower), "bounds": tuple(bounds)}
    raise AssertionError("no complete facing brackets")


def _infer(grid):
    counts = np.bincount(grid.ravel().astype(int), minlength=10)
    bg = int(np.argmax(counts))
    assert np.count_nonzero(counts == counts[bg]) == 1
    geometry = _structure(grid, bg)
    a = grid if geometry["orientation"] == "horizontal" else grid.T
    upper, lower = geometry["axes"]
    source_cells = [[], []]
    targets = []
    for row, col in np.argwhere((a != bg) & (a != geometry["color"])):
        row, col = int(row), int(col)
        if row < upper:
            side, depth, target_row = 0, upper - row, 2 * upper - row
        elif row > lower:
            side, depth, target_row = 1, row - lower, 2 * lower - row
        else:
            raise AssertionError("payload lies between brackets")
        lo, hi = geometry["bounds"][side]
        assert lo < col < hi and upper < target_row < lower
        assert a[target_row, col] == bg
        source_cells[side].append((depth, col - lo))
        targets.append(((row, col), (target_row, col)))
    assert all(source_cells) and len({t for _, t in targets}) == len(targets)
    predicted = a.copy()
    for source, target in targets:
        predicted[target] = a[source]
        predicted[source] = bg
    if geometry["orientation"] == "vertical":
        predicted = predicted.T.copy()
    assert not np.array_equal(predicted, grid)
    lengths = tuple(hi - lo + 1 for lo, hi in geometry["bounds"])
    starts = tuple(lo for lo, _ in geometry["bounds"])
    features = {
        "depth1": any(d == 1 for side in source_cells for d, _ in side),
        "three_layer": any(len({d for d, _ in side}) >= 3 for side in source_cells),
        "multicolor": len({int(a[s]) for s, _ in targets}) >= 2,
        "unequal": lengths[0] != lengths[1],
        "offset": starts[0] != starts[1],
        "disconnected": any(not _connected(side) for side in source_cells),
        "hole": any(_has_hole(side) for side in source_cells),
    }
    depths = [max(d for d, _ in side) for side in source_cells]
    gap = lower - upper - sum(depths) - 1
    assert gap >= 0
    return predicted, {"orientation": geometry["orientation"],
                       "lengths": lengths, "starts": starts,
                       "motifs": tuple(tuple(sorted(x)) for x in source_cells),
                       "features": features, "gap": gap,
                       "colors": (bg, geometry["color"]), "shape": grid.shape}


def _key(grid):
    return grid.shape, tuple(int(x) for x in grid.ravel())


def _source_copy(grid):
    if grid.shape != (15, 15):
        return False
    digits = "".join(str(int(x)) for x in grid.ravel())
    return hashlib.sha256(digits.encode()).hexdigest() in _SOURCE_INPUT_DIGESTS


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    info = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            x, y = pair["input"], pair["output"]
            for grid in (x, y):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((grid >= 0) & (grid <= 9))
            assert x.shape == y.shape
            assert not _source_copy(x)
            ix, iy = _key(x), _key(y)
            assert ix not in seen_inputs and (ix, iy) not in seen_pairs
            seen_inputs.add(ix)
            seen_pairs.add((ix, iy))
            prediction, facts = _infer(x)
            assert np.array_equal(prediction, y)
            assert all(5 <= length <= 13 for length in facts["lengths"])
            assert all(1 <= len(m) <= 16 for m in facts["motifs"])
            info[section].append(facts)
    trained_orientations = {f["orientation"] for f in info["train"]}
    trained_features = {k for f in info["train"] for k, v in f["features"].items() if v}
    for facts in info["test"]:
        assert facts["orientation"] in trained_orientations
        assert {k for k, v in facts["features"].items() if v} <= trained_features


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    n, m = _choose_counts(rng, seed, num_train, num_test)
    palette = tuple(int(x) for x in rng.choice(10, size=4, replace=False))
    variable_palette = bool(rng.integers(0, 2))
    first_orientation = "horizontal" if rng.random() < 0.5 else "vertical"
    if rng.random() < 0.52:
        orientations = [first_orientation] * n
    else:
        other = "vertical" if first_orientation == "horizontal" else "horizontal"
        orientations = [first_orientation, other] + [
            first_orientation if rng.random() < 0.5 else other for _ in range(n - 2)
        ]
        rng.shuffle(orientations)
    all_allowed = dict.fromkeys(("depth1", "three_layer", "multicolor",
                                 "unequal", "offset", "disconnected", "hole"), True)
    puzzle = {"train": [], "test": []}
    seen = set()

    def append(section, orientation, allowed, lengths=()):
        for _ in range(60):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            pair = _make_pair(pair_rng, orientation, palette, variable_palette,
                              allowed, "source_train" if section == "train" else "target_test",
                              lengths)
            ix = _key(pair["input"])
            actual_features = _infer(pair["input"])[1]["features"]
            if (ix not in seen and not _source_copy(pair["input"])
                    and all(not value or allowed[key]
                            for key, value in actual_features.items())):
                seen.add(ix)
                puzzle[section].append(pair)
                return
        raise RuntimeError("distinct pair construction failed")

    plain = dict.fromkeys(all_allowed, False)
    for i, orientation in enumerate(orientations):
        append("train", orientation, plain if i == 0 else all_allowed)
    train_facts = [_infer(p["input"])[1] for p in puzzle["train"]]
    available = {key: any(f["features"][key] for f in train_facts)
                 for key in all_allowed}
    trained_orientations = sorted({f["orientation"] for f in train_facts})
    lengths = tuple(length for f in train_facts for length in f["lengths"])
    for _ in range(m):
        append("test", str(rng.choice(trained_orientations)), available, lengths)
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for section in ("train", "test"):
            for pair in puzzle[section]:
                assert np.array_equal(_infer(pair["input"])[0], pair["output"])
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    facts = [_infer(pair["input"])[1] for puzzle in puzzles
             for section in ("train", "test") for pair in puzzle[section]]
    assert {f["orientation"] for f in facts} == {"horizontal", "vertical"}
    assert all(any(f["features"][key] for f in facts)
               for key in ("depth1", "three_layer", "multicolor",
                           "unequal", "offset", "disconnected", "hole"))
    assert any(f["gap"] == 0 for f in facts)
    assert len({f["shape"] for f in facts}) >= 20
    assert len({f["colors"] for f in facts}) >= 20
    assert any(f["shape"] == (15, 15) for f in facts)
    assert any(f["shape"][0] > f["shape"][1] for f in facts)
    assert any(f["shape"][0] < f["shape"][1] for f in facts)
    assert all(len({f["shape"] for f in (
        _infer(pair["input"])[1] for section in ("train", "test")
        for pair in puzzle[section])}) > 1 for puzzle in puzzles)
    assert all(len({length for f in (
        _infer(pair["input"])[1] for section in ("train", "test")
        for pair in puzzle[section]) for length in f["lengths"]}) > 1
        for puzzle in puzzles)
    assert all(len({len(side) for f in (
        _infer(pair["input"])[1] for section in ("train", "test")
        for pair in puzzle[section]) for side in f["motifs"]}) > 1
        for puzzle in puzzles)
    same_orientation = sum(len({_infer(pair["input"])[1]["orientation"]
        for pair in puzzle["train"]}) == 1 for puzzle in puzzles)
    assert 5 <= same_orientation <= 37
    assert sum(_infer(pair["input"])[1]["orientation"] == "vertical"
        for puzzle in puzzles for pair in puzzle["test"]) >= 5
    assert sum(f["gap"] == 0 for f in facts) >= 10
    assert sum(f["shape"] == (15, 15) for f in facts) >= 1
    assert sum(f["features"]["unequal"] and f["features"]["offset"]
        and f["features"]["multicolor"] for f in facts) >= 10
    assert sum(len({d for d, _ in side}) == 1 for f in facts
        for side in f["motifs"]) >= 40
    assert sum(len({d for d, _ in side}) == 3 for f in facts
        for side in f["motifs"]) >= 20
    assert sum(len(f["motifs"][0]) != len(f["motifs"][1])
        for f in facts) >= 20
    assert sum({d for d, _ in f["motifs"][0]} !=
        {d for d, _ in f["motifs"][1]} for f in facts) >= 20
    for n, m in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(100 + n + m, n, m)
        assert (len(p["train"]), len(p["test"])) == (n, m)
    for n, m in ((None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(300 + (n or 0) + (m or 0), n, m))
    for n, m in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4), (10, 10)):
        try:
            generate_puzzle(500, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a = generate_puzzle(99173, 7, 3)
    b = generate_puzzle(99173, 7, 3)
    assert all(np.array_equal(x[k], y[k]) for section in ("train", "test")
               for x, y in zip(a[section], b[section]) for k in ("input", "output"))
    assert _key(generate_puzzle(101)["train"][0]["input"]) != _key(
        generate_puzzle(102)["train"][0]["input"])
