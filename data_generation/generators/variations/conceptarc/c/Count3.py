import numpy as np


# Exact source inputs are excluded, not their dimensions or color sets.
_SOURCE_INPUTS = (
    ("000000000000", "000000000080", "011111000080", "000000000080",
     "000000000080", "000000020000", "005550020000", "000000020000",
     "000000020000", "000000020000", "000000020000", "000000020000"),
    ("00000000000000", "00044444400000", "00000000000000", "00666666666000",
     "80000000000000", "80000000000000", "80000555555030", "80000000000030",
     "80000000000030", "80000000000000"),
    ("0000", "0400", "0400", "0400", "0400", "0000"),
    ("000000009000", "000600009000", "000600009000", "000600000000",
     "000600000000", "000600055555", "000000000000", "000000000000",
     "000000000000", "044444400000", "000000000000", "000000000000",
     "033333333330", "000000000000"),
    ("400000", "400000", "400000", "044440", "000000", "000000"),
    ("666666",),
)
_SOURCE_KEYS = {(len(rows), len(rows[0]), rows) for rows in _SOURCE_INPUTS}


def _key(grid):
    return grid.shape, grid.tobytes()


def _source_key(grid):
    return (grid.shape[0], grid.shape[1],
            tuple("".join(str(int(v)) for v in row) for row in grid))


def _background(grid):
    if min(grid.shape) == 1:
        return 0
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _components(grid):
    occupied = grid != _background(grid)
    seen = np.zeros(grid.shape, dtype=bool)
    result = []
    for y, x in np.argwhere(occupied):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        stack = [(y, x)]
        seen[y, x] = True
        cells = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]
                        and occupied[rr, cc] and not seen[rr, cc]):
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        result.append(cells)
    return result


def _can_place(occupied, cells):
    h, w = occupied.shape
    for y, x in cells:
        if occupied[y, x]:
            return False
        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= yy < h and 0 <= xx < w and occupied[yy, xx]:
                return False
    return True


def _diagonal_touch(first, second):
    return any(abs(y - yy) == 1 and abs(x - xx) == 1
               for y, x in first for yy, xx in second)


def _shape(rng, count, diagonal):
    density = int(rng.choice(3, p=(0.43, 0.39, 0.18)))
    area_per_line = (11, 22, 43)[density]
    area = count * area_per_line + int(rng.integers(0, 13))
    ratio = float(np.exp(rng.uniform(-1.05, 1.05)))
    h = int(round(np.sqrt(area * ratio)))
    w = int(round(np.sqrt(area / ratio)))
    h = max(3, min(30, h + int(rng.integers(-2, 3))))
    w = max(3, min(30, w + int(rng.integers(-2, 3))))
    if diagonal:
        h, w = max(h, 4), max(w, 5)
    return h, w


def _initial_diagonal(rng, h, w):
    # The two lines may be parallel or perpendicular; reflections and rotation
    # distribute the contact over the canvas and both orientations.
    parallel = bool(rng.integers(0, 2))
    if parallel:
        a = int(rng.integers(2, min(7, w - 1)))
        b = int(rng.integers(2, min(7, w - a + 1)))
        x = int(rng.integers(0, w - a - b + 1))
        y = int(rng.integers(0, h - 1))
        first = [(y, x + i) for i in range(a)]
        second = [(y + 1, x + a + i) for i in range(b)]
    else:
        a = int(rng.integers(2, min(8, w)))
        b = int(rng.integers(2, min(8, h)))
        x = int(rng.integers(0, w - a))
        y = int(rng.integers(0, h - b))
        first = [(y, x + i) for i in range(a)]
        second = [(y + 1 + i, x + a) for i in range(b)]
    if bool(rng.integers(0, 2)):
        first = [(yy, w - 1 - xx) for yy, xx in first]
        second = [(yy, w - 1 - xx) for yy, xx in second]
    if bool(rng.integers(0, 2)):
        first = [(h - 1 - yy, xx) for yy, xx in first]
        second = [(h - 1 - yy, xx) for yy, xx in second]
    if bool(rng.integers(0, 2)):
        first = [(xx, yy) for yy, xx in first]
        second = [(xx, yy) for yy, xx in second]
        h, w = w, h
    return (h, w), first, second


def _length(rng, span):
    choice = int(rng.choice(4, p=(0.28, 0.35, 0.25, 0.12)))
    if choice == 0:
        return int(rng.integers(2, min(5, span + 1)))
    if choice == 1:
        return int(rng.integers(2, min(10, span + 1)))
    if choice == 2:
        return int(rng.integers(max(2, span // 2), span + 1))
    return span


def _line(rng, h, w):
    horizontal = bool(rng.integers(0, 2))
    span = w if horizontal else h
    length = _length(rng, span)
    if horizontal:
        y = int(rng.integers(0, h))
        x = int(rng.integers(0, w - length + 1))
        return [(y, x + i) for i in range(length)]
    y = int(rng.integers(0, h - length + 1))
    x = int(rng.integers(0, w))
    return [(y + i, x) for i in range(length)]


def _two_dimensional(rng, count, diagonal):
    for _ in range(90):
        h, w = _shape(rng, count, diagonal)
        if diagonal:
            (h, w), first, second = _initial_diagonal(rng, h, w)
            lines = [first, second]
        else:
            lines = []
        occupied = np.zeros((h, w), dtype=bool)
        for line in lines:
            for y, x in line:
                occupied[y, x] = True
        for _line_index in range(len(lines), count):
            placed = False
            for _ in range(180):
                cells = _line(rng, h, w)
                if _can_place(occupied, cells):
                    lines.append(cells)
                    for y, x in cells:
                        occupied[y, x] = True
                    placed = True
                    break
            if not placed:
                break
        if len(lines) == count and occupied.sum() * 2 < h * w:
            return (h, w), lines
    raise RuntimeError("Could not arrange isolated 2D lines")


def _strip(rng, count):
    for _ in range(100):
        lengths = [_length(rng, int(rng.integers(4, 27))) for _ in range(count)]
        gaps = [int(rng.integers(1, 6)) for _ in range(count - 1)]
        if count == 1 and rng.random() < 0.65:
            left = right = 0
        else:
            left = int(rng.integers(0, 5))
            right = int(rng.integers(0, 5))
        size = left + right + sum(lengths) + sum(gaps)
        if size > 30 or size < 2:
            continue
        lines = []
        pos = left
        for i, length in enumerate(lengths):
            lines.append([(0, pos + j) for j in range(length)])
            pos += length
            if i < count - 1:
                pos += gaps[i]
        if bool(rng.integers(0, 2)):
            return (size, 1), [[(x, 0) for _, x in line] for line in lines]
        return (1, size), lines
    raise RuntimeError("Could not arrange strip lines")


def _colors(rng, count, pattern, background, diagonal):
    choices = [v for v in range(1, 10) if v != background]
    if pattern == "unique" or count == 1:
        return [int(v) for v in rng.choice(choices, size=count, replace=False)]
    if pattern == "all_same":
        return [int(rng.choice(choices))] * count
    if pattern == "one_pair":
        colors = [int(v) for v in rng.choice(choices, size=count - 1, replace=False)]
        if diagonal:
            colors.insert(1, colors[0])
        else:
            colors.insert(int(rng.integers(0, count)), int(rng.choice(colors)))
        return colors
    colors = [int(v) for v in rng.choice(choices, size=count, replace=True)]
    if len(set(colors)) == count:
        colors[1] = colors[0]
    if diagonal:
        colors[1] = colors[0]
    return colors


def _spec(rng, regime, forced=None):
    strip = rng.random() < (0.27 if regime == "train" else 0.36)
    if strip:
        probabilities = ((.25, .26, .22, .17, .10) if regime == "train"
                         else (.28, .25, .22, .16, .09))
        count = int(rng.choice(np.arange(1, 6), p=probabilities))
    else:
        count = int(rng.choice(np.arange(1, 8),
                               p=(.16, .24, .20, .16, .12, .08, .04)))
    if forced is not None:
        strip = forced.get("strip", strip)
        count = forced.get("count", count)
    patterns = ("unique", "one_pair", "all_same", "mixed")
    probabilities = (.47, .27, .16, .10) if regime == "train" else (.31, .32, .24, .13)
    pattern = str(rng.choice(patterns, p=probabilities))
    if forced is not None:
        pattern = forced.get("pattern", pattern)
    if count == 1:
        pattern = "unique"
    diagonal = (not strip and count >= 2
                and rng.random() < (.32 if regime == "train" else .43))
    if forced is not None:
        diagonal = forced.get("diagonal", diagonal)
    return {"strip": strip, "count": count, "pattern": pattern,
            "diagonal": diagonal, "background": 0}


def _make_pair(rng, spec):
    if spec["strip"]:
        shape, lines = _strip(rng, spec["count"])
    else:
        shape, lines = _two_dimensional(rng, spec["count"], spec["diagonal"])
    background = spec["background"]
    grid = np.full(shape, background, dtype=np.int8)
    colors = _colors(rng, len(lines), spec["pattern"], background,
                     spec["diagonal"] and spec["pattern"] != "unique")
    for line, color in zip(lines, colors):
        for y, x in line:
            grid[y, x] = color
    return {"input": grid, "output": np.full((1, len(lines)), 3, dtype=np.int8)}


def _requested_counts(rng, num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"),
                                   (num_test, 1, 4, "num_test")):
        if (value is not None and (not isinstance(value, (int, np.integer))
                                   or isinstance(value, (bool, np.bool_))
                                   or not low <= value <= high)):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is None:
        low = 2 if num_test is None else num_test + 1
        num_train = int(rng.integers(low, 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _requested_counts(rng, num_train, num_test)
    # Color 3 in the output and four-neighbor counting are the puzzle conventions.
    # Input colors, background shade, and all geometry are pair-level choices.
    repeat_demonstration = num_train == 2 or rng.random() < .78
    mixed_backgrounds = rng.random() < .23
    for _ in range(28):
        train_specs = []
        core = {"strip": False, "count": 2,
                "pattern": "one_pair" if repeat_demonstration else "unique",
                "diagonal": bool(rng.integers(0, 2))}
        train_specs.append(_spec(rng, "train", core))
        other_count = int(rng.choice((1, 3, 4, 5)))
        train_specs.append(_spec(rng, "train", {"count": other_count,
                                                "pattern": "unique"}))
        for _extra in range(num_train - 2):
            train_specs.append(_spec(rng, "train",
                                     None if repeat_demonstration else
                                     {"pattern": "unique"}))
        rng.shuffle(train_specs)
        test_specs = [_spec(rng, "test") for _test in range(num_test)]
        if not repeat_demonstration:
            test_specs[int(rng.integers(0, num_test))] = _spec(
                rng, "test", {"count": int(rng.integers(2, 6)),
                              "pattern": "one_pair"})
        if mixed_backgrounds:
            nonzero_train = int(rng.integers(0, num_train))
            for i, spec in enumerate(train_specs):
                if not spec["strip"] and (i == nonzero_train or rng.random() < .18):
                    spec["background"] = int(rng.choice((1, 2, 4, 5, 6, 7, 8, 9)))
            if not any(spec["background"] for spec in train_specs):
                for spec in train_specs:
                    if not spec["strip"]:
                        spec["background"] = int(rng.choice((1, 2, 4, 5, 6, 7, 8, 9)))
                        break
            for spec in test_specs:
                if not spec["strip"] and rng.random() < .35:
                    spec["background"] = int(rng.choice((1, 2, 4, 5, 6, 7, 8, 9)))
        pairs = {"train": [], "test": []}
        seen = set()
        try:
            for regime, specs in (("train", train_specs), ("test", test_specs)):
                for spec in specs:
                    for _attempt in range(20):
                        pair = _make_pair(rng, spec)
                        grid = pair["input"]
                        key = _key(grid)
                        if key not in seen and _source_key(grid) not in _SOURCE_KEYS:
                            seen.add(key)
                            pairs[regime].append(pair)
                            break
                    else:
                        raise RuntimeError("Could not produce distinct pair")
            validate_generated_puzzle(pairs)
            return pairs
        except (AssertionError, RuntimeError, ValueError):
            continue
    raise RuntimeError("Could not construct a coherent puzzle")


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4
    assert len(train) > len(test)
    seen = set()
    train_counts = set()
    train_orientations = set()
    train_backgrounds = set()
    for regime, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = pair["input"], pair["output"]
            for array in (grid, output):
                assert isinstance(array, np.ndarray) and array.ndim == 2
                assert 1 <= array.shape[0] <= 30 and 1 <= array.shape[1] <= 30
                assert np.issubdtype(array.dtype, np.integer)
                assert np.all((array >= 0) & (array <= 9))
            assert _key(grid) not in seen
            seen.add(_key(grid))
            assert _source_key(grid) not in _SOURCE_KEYS
            assert not np.array_equal(grid, output)
            background = _background(grid)
            if min(grid.shape) == 1:
                assert background == 0
            else:
                assert np.count_nonzero(grid == background) > grid.size // 2
            components = _components(grid)
            assert 1 <= len(components) <= 8
            for cells in components:
                assert len(cells) >= 2
                assert len({int(grid[y, x]) for y, x in cells}) == 1
                ys = {y for y, _ in cells}
                xs = {x for _, x in cells}
                assert len(ys) == 1 or len(xs) == 1
                assert len(cells) == max(max(ys) - min(ys) + 1,
                                         max(xs) - min(xs) + 1)
                if regime == "train":
                    train_orientations.add("H" if len(ys) == 1 else "V")
            assert output.shape == (1, len(components)) and np.all(output == 3)
            if regime == "train":
                train_counts.add(len(components))
                train_backgrounds.add(background)
            elif background != 0:
                assert any(v != 0 for v in train_backgrounds)
    assert len(train_counts) >= 2
    assert train_orientations == {"H", "V"}
    if len(train) == 2:
        assert any(len({int(p["input"][c[0]]) for c in _components(p["input"])})
                   < len(_components(p["input"])) for p in train)


def _equal_puzzles(a, b):
    return all(len(a[group]) == len(b[group]) and all(
        np.array_equal(p["input"], q["input"])
        and np.array_equal(p["output"], q["output"])
        for p, q in zip(a[group], b[group])) for group in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, shapes, layouts, fingerprints = set(), set(), set(), set(), set()
    repeated_train = separated_repeats = diagonal_repeats = 0
    strips_train = strips_test = full_strips = nonzero_backgrounds = long_2d = 0
    new_test_colors = ordinary_tests = source_like = combinations = 0
    horizontal_strips = vertical_strips = compact_dense = source_dimensions = 0
    short_full = long_full = nonzero_tests = all_same_train = all_same_test = 0
    two_line_gap = two_line_diagonal = 0
    source_shapes = {(h, w) for h, w, _ in _SOURCE_KEYS}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _equal_puzzles(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_colors = {int(v) for pair in puzzle["train"]
                        for v in np.unique(pair["input"])}
        train_has_repeat = any(len({int(pair["input"][c[0]])
                                    for c in _components(pair["input"])})
                               < len(_components(pair["input"]))
                               for pair in puzzle["train"])
        test_has_repeat = any(len({int(pair["input"][c[0]])
                                   for c in _components(pair["input"])})
                              < len(_components(pair["input"]))
                              for pair in puzzle["test"])
        test_new_color = any(any(int(v) not in train_colors
                                 for v in np.unique(pair["input"])
                                 if v != _background(pair["input"]))
                             for pair in puzzle["test"])
        test_has_strip = any(min(pair["input"].shape) == 1
                             for pair in puzzle["test"])
        new_test_colors += test_new_color
        source_like += not train_has_repeat and test_has_repeat
        combinations += train_has_repeat and test_new_color and test_has_strip
        fingerprint = []
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                grid = pair["input"]
                comps = _components(grid)
                colors = [int(grid[cells[0]]) for cells in comps]
                assert pair["output"].shape == (1, len(comps))
                shapes.add(grid.shape)
                source_dimensions += grid.shape in source_shapes
                layouts.add((grid.shape, tuple(sorted((len(cells),
                    min(y for y, _ in cells), min(x for _, x in cells)) for cells in comps))))
                fingerprint.append(_key(grid))
                if len(colors) > len(set(colors)):
                    if regime == "train":
                        repeated_train += 1
                    same_diag = any(colors[i] == colors[j] and _diagonal_touch(a, b)
                                    for i, a in enumerate(comps)
                                    for j, b in enumerate(comps) if i < j)
                    diagonal_repeats += same_diag
                    separated_repeats += not same_diag
                    two_line_diagonal += len(comps) == 2 and same_diag
                    two_line_gap += len(comps) == 2 and not same_diag
                if regime == "test" and len(colors) == len(set(colors)):
                    ordinary_tests += 1
                if len(colors) > 1 and len(set(colors)) == 1:
                    if regime == "train":
                        all_same_train += 1
                    else:
                        all_same_test += 1
                if min(grid.shape) == 1:
                    strips_train += regime == "train"
                    strips_test += regime == "test"
                    horizontal_strips += grid.shape[0] == 1
                    vertical_strips += grid.shape[1] == 1
                    full_strips += len(comps) == 1 and np.all(grid != 0)
                    if len(comps) == 1 and np.all(grid != 0):
                        short_full += max(grid.shape) <= 7
                        long_full += max(grid.shape) >= 10
                elif _background(grid) != 0:
                    nonzero_backgrounds += 1
                    nonzero_tests += regime == "test"
                if min(grid.shape) > 1 and max(grid.shape) <= 10:
                    density = sum(len(c) for c in comps) / grid.size
                    compact_dense += .13 <= density <= .35
                if min(grid.shape) > 1 and max(len(c) for c in comps) >= 10:
                    long_2d += 1
        fingerprints.add(tuple(fingerprint))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 45 and len(layouts) >= 150 and len(fingerprints) == 42
    assert repeated_train >= 10 and separated_repeats >= 10 and diagonal_repeats >= 5
    assert strips_train >= 5 and strips_test >= 5 and full_strips >= 3
    assert nonzero_backgrounds >= 3 and long_2d >= 10
    assert new_test_colors >= 5 and ordinary_tests >= 10 and source_like >= 3
    assert combinations >= 3 and horizontal_strips >= 8 and vertical_strips >= 8
    assert compact_dense >= 20 and source_dimensions >= 1
    assert short_full >= 1 and long_full >= 1 and nonzero_tests >= 1
    assert all_same_train >= 10 and all_same_test >= 5
    assert two_line_gap >= 5 and two_line_diagonal >= 5
    for train_count, test_count in ((2, 1), (10, 1), (10, 4),
                                    (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(931, train_count, test_count)
        validate_generated_puzzle(puzzle)
        assert _equal_puzzles(puzzle, generate_puzzle(931, train_count, test_count))
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    for train_count, test_count in ((1, None), (11, None), (None, 0), (None, 5),
                                    (2, 2), (3, 4), (10, 10)):
        try:
            generate_puzzle(931, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal pair counts were accepted")
