import numpy as np


_SOURCE_INPUT_ROWS = frozenset(
    {
        (
            "00000181000110",
            "00000111000110",
            "11110111000110",
            "18110000000000",
            "11110000000000",
            "00000000011810",
            "00000000018110",
            "00011110011180",
            "00011810000000",
            "00018110000000",
            "00011110001811",
            "00000000001118",
            "00000000001881",
            "00000000001111",
        ),
        (
            "00000000000000",
            "01810000000000",
            "01180111001800",
            "00000181001100",
            "00000118000000",
            "00000811000111",
            "00000000000181",
            "01180000000111",
            "08110000000000",
            "01110001110000",
            "00000001110000",
            "00000008110000",
            "00000001110000",
            "00000000000000",
        ),
        (
            "000001111008100",
            "111108118001800",
            "181101111000000",
            "111101181001111",
            "000000000001811",
            "000000000001111",
            "001181100000000",
            "001118100000000",
            "001811100000000",
            "000000000118000",
            "001111000181000",
            "001111000111000",
            "001111000818000",
        ),
        (
            "000000001811",
            "111000001111",
            "811000001818",
            "111081100000",
            "000011100000",
            "000011800000",
            "000011101111",
            "000000001181",
            "011111101111",
            "011118101111",
            "018111100000",
            "011111101110",
            "000000001110",
        ),
    }
)


def _validate_explicit_count(
    name: str,
    value: int | None,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return value


def _resolve_pair_counts(
    rng: np.random.Generator,
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _validate_explicit_count(
        "num_train", num_train, 2, 10
    )
    num_test = _validate_explicit_count(
        "num_test", num_test, 1, 4
    )

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test

    if num_train is None and num_test is None:
        # This mapping guarantees broad count diversity for consecutive seeds.
        seed_key = int(seed)
        num_test = 1 + seed_key % 4
        minimum_train = max(2, num_test + 1)
        span = 11 - minimum_train
        num_train = minimum_train + (seed_key // 4) % span
        return num_train, num_test

    if num_train is None:
        assert num_test is not None
        minimum_train = max(2, num_test + 1)
        num_train = int(rng.integers(minimum_train, 11))
        return num_train, num_test

    maximum_test = min(4, num_train - 1)
    if maximum_test < 1:
        raise ValueError("no legal test count exists")
    num_test = int(rng.integers(1, maximum_test + 1))
    return num_train, num_test


def _sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict:
    colors = rng.choice(10, size=3, replace=False)
    return {
        "background": int(colors[0]),
        "carrier": int(colors[1]),
        "indicator": int(colors[2]),
    }


# Shapes include the source's 3x5 object, the test's 4x6 object,
# their rotations, and smaller objects.  Every shape can have either fate.
_SHAPES = tuple(
    (h, w) for h in range(2, 8) for w in range(2, 8)
    if h * w <= 28 and min(h, w) <= 5
)
_CONTRAST_SHAPES = ((3, 5), (4, 6), (5, 3), (6, 4),
                    (2, 5), (5, 2), (3, 4), (4, 3), (4, 4))


def _sample_shape(rng, canvas_area, large_bias=False):
    candidates = [(h, w) for h, w in _SHAPES if h * w <= canvas_area * 0.16]
    weights = np.array([
        (h * w) ** (-0.7 if large_bias else -1.1)
        for h, w in candidates
    ], dtype=float)
    weights /= weights.sum()
    return candidates[int(rng.choice(len(candidates), p=weights))]


def _sample_canvas(rng, regime):
    # The two regimes overlap.  Test examples more often use the small
    # source-like canvases or extreme aspects, but either split can do so.
    if rng.random() < (0.34 if regime == "target_test" else 0.22):
        height = int(rng.integers(11, 16))
        width = int(rng.integers(11, 16))
    else:
        height = int(rng.integers(12, 24))
        width = int(rng.integers(12, 24))
    return height, width


def _sample_marker_count(rng, mode, area):
    if mode == "keep":
        return int(rng.integers(0, 2))
    if mode == "erase":
        return int(rng.integers(2, min(6, area - 1) + 1))
    return _sample_marker_count(rng, "keep" if rng.random() < 0.5 else "erase", area)


def _make_specs(rng, regime, pair_index, height, width, contrast_shape):
    area = height * width
    minimum = 3 if regime == "source_train" and pair_index == 0 else 2
    maximum = min(9, max(minimum, area // 25))
    count = int(rng.integers(minimum, maximum + 1))
    if regime == "source_train" and pair_index == 0:
        mode = "mixed"
        counts = [0, 1, 2]
    else:
        mode = str(rng.choice(("keep", "erase", "mixed"), p=(0.23, 0.24, 0.53)))
        counts = []
    specs = []
    for marker_count in counts:
        h, w = _sample_shape(rng, area)
        specs.append((h, w, marker_count))
    if regime == "source_train" and pair_index in (0, 1):
        # A retained and erased object of identical dimensions appear in
        # the training set, often with a wide source-scale footprint.
        h, w = contrast_shape
        forced_count = 1 if pair_index == 0 else 2
        if pair_index == 0:
            specs[1] = (h, w, forced_count)
        else:
            specs.append((h, w, forced_count))
    while len(specs) < count:
        h, w = _sample_shape(rng, area, large_bias=(regime == "target_test" and rng.random() < 0.4))
        specs.append((h, w, _sample_marker_count(rng, mode, h * w)))
    rng.shuffle(specs)
    return specs


def _place_rectangles(rng, height, width, specs):
    order = sorted(range(len(specs)), key=lambda i: specs[i][0] * specs[i][1], reverse=True)
    want_edge = rng.random() < 0.72
    want_corner = rng.random() < 0.13
    for _ in range(12):
        occupied = np.zeros((height, width), dtype=bool)
        positions = [None] * len(specs)
        success = True
        for depth, index in enumerate(order):
            h, w, _ = specs[index]
            candidates = [(r, c) for r in range(height - h + 1)
                          for c in range(width - w + 1)]
            rng.shuffle(candidates)
            if depth == 0 and want_edge:
                if want_corner:
                    candidates.sort(key=lambda p: not (
                        p[0] in (0, height - h) and p[1] in (0, width - w)))
                else:
                    candidates.sort(key=lambda p: not (
                        p[0] in (0, height - h) or p[1] in (0, width - w)))
            chosen = None
            for r, c in candidates:
                halo = occupied[max(0, r - 1):min(height, r + h + 1),
                                max(0, c - 1):min(width, c + w + 1)]
                if not np.any(halo):
                    chosen = (r, c)
                    break
            if chosen is None:
                success = False
                break
            r, c = chosen
            occupied[r:r + h, c:c + w] = True
            positions[index] = chosen
        if success:
            return positions
    return None


def _marker_indices(rng, height, width, count):
    if count == 0:
        return []
    cells = [(r, c) for r in range(height) for c in range(width)]
    if count == 1:
        category = str(rng.choice(("corner", "edge", "interior", "any")))
        if category == "corner":
            choices = [(r, c) for r, c in cells if r in (0, height-1) and c in (0, width-1)]
        elif category == "edge":
            choices = [(r, c) for r, c in cells if r in (0, height-1) or c in (0, width-1)]
        elif category == "interior":
            choices = [(r, c) for r, c in cells if 0 < r < height-1 and 0 < c < width-1]
        else:
            choices = cells
        if not choices:
            choices = cells
        return [choices[int(rng.integers(len(choices)))]]
    pattern = str(rng.choice(("close", "far", "scatter")))
    first = cells[int(rng.integers(len(cells)))]
    rest = [p for p in cells if p != first]
    if pattern == "close":
        rest.sort(key=lambda p: abs(p[0]-first[0]) + abs(p[1]-first[1]))
        pool = rest[:max(count - 1, min(len(rest), count + 2))]
    elif pattern == "far":
        rest.sort(key=lambda p: abs(p[0]-first[0]) + abs(p[1]-first[1]), reverse=True)
        pool = rest[:max(count - 1, min(len(rest), count + 2))]
    else:
        pool = rest
    chosen = rng.choice(len(pool), size=count-1, replace=False)
    return [first] + [pool[int(i)] for i in chosen]


def _render_pair(rng, height, width, specs, positions, roles):
    background, carrier, indicator = roles
    inp = np.full((height, width), background, dtype=np.int8)
    out = inp.copy()
    for (h, w, marker_count), (r, c) in zip(specs, positions):
        inp[r:r+h, c:c+w] = carrier
        for mr, mc in _marker_indices(rng, h, w, marker_count):
            inp[r+mr, c+mc] = indicator
        if marker_count <= 1:
            out[r:r+h, c:c+w] = inp[r:r+h, c:c+w]
    return inp, out


def _generate_pair(rng, regime, pair_index, roles, contrast_shape):
    for _ in range(160):
        height, width = _sample_canvas(rng, regime)
        specs = _make_specs(rng, regime, pair_index, height, width, contrast_shape)
        if sum(h*w for h, w, _ in specs) > 0.43 * height * width:
            continue
        positions = _place_rectangles(rng, height, width, specs)
        if positions is not None:
            return _render_pair(rng, height, width, specs, positions, roles)
    raise RuntimeError("could not pack separated rectangles")


def _grid_signature(grid):
    return (grid.shape, grid.tobytes())


def _rows_signature(grid):
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _find_components(grid, background):
    height, width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    components = []
    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(grid[r, c]) == background:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for ny, nx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                    if 0 <= ny < height and 0 <= nx < width and not seen[ny, nx] and int(grid[ny, nx]) != background:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            components.append(cells)
    return components


def _analyze_pair(inp, out, background, carrier, indicator):
    assert inp.shape == out.shape
    assert set(int(v) for v in np.unique(inp)) <= {background, carrier, indicator}
    assert set(int(v) for v in np.unique(out)) <= {background, carrier, indicator}
    assert np.count_nonzero(inp == background) > inp.size // 2
    components = _find_components(inp, background)
    assert 2 <= len(components) <= 9
    objects = []
    expected = inp.copy()
    for cells in components:
        rows, cols = zip(*cells)
        r0, r1, c0, c1 = min(rows), max(rows)+1, min(cols), max(cols)+1
        h, w = r1-r0, c1-c0
        region = inp[r0:r1, c0:c1]
        assert 2 <= h <= 7 and 2 <= w <= 7 and h*w <= 28
        assert len(cells) == h*w and np.all(region != background)
        markers = int(np.count_nonzero(region == indicator))
        carriers = int(np.count_nonzero(region == carrier))
        assert carriers >= 1 and markers <= min(6, h*w-1)
        if markers >= 2:
            expected[r0:r1, c0:c1] = background
        objects.append((r0, c0, h, w, markers))
    objects.sort()
    for i, (r, c, h, w, _) in enumerate(objects):
        for rr, cc, hh, ww, _ in objects[i+1:]:
            assert r+h < rr or rr+hh < r or c+w < cc or cc+ww < c
    assert np.array_equal(out, expected)
    return objects


def _infer_roles_from_train(train):
    # Background is the majority color in every generated input.  The
    # retained zero-marker and erased two-marker demonstrations determine
    # which of the other two colors is the carrier.
    backgrounds = []
    colors = set()
    for pair in train:
        values, counts = np.unique(pair["input"], return_counts=True)
        backgrounds.append(int(values[int(np.argmax(counts))]))
        colors.update(int(v) for v in values)
    assert len(set(backgrounds)) == 1
    background = backgrounds[0]
    foreground = list(colors - {background})
    assert len(foreground) == 2
    assignments = []
    for carrier, indicator in (foreground, foreground[::-1]):
        try:
            objects = [_analyze_pair(p["input"], p["output"], background, carrier, indicator) for p in train]
        except AssertionError:
            continue
        counts = [o[4] for pair_objects in objects for o in pair_objects]
        if {0, 1, 2} <= set(counts):
            assignments.append((background, carrier, indicator))
    assert len(assignments) == 1
    return assignments[0]


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            assert pair["input"].shape == pair["output"].shape
            assert all(11 <= n <= 23 for n in pair["input"].shape)
            signature = _grid_signature(pair["input"])
            assert signature not in seen
            seen.add(signature)
            assert _rows_signature(pair["input"]) not in _SOURCE_INPUT_ROWS
    roles = _infer_roles_from_train(puzzle["train"])
    train_objects = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            objects = _analyze_pair(pair["input"], pair["output"], *roles)
            if split == "train":
                train_objects.append(objects)
            # The same learned colors and rule apply to tests, including
            # outputs without either foreground color.
    counts = {o[4] for objects in train_objects for o in objects}
    assert {0, 1, 2} <= counts
    shapes_by_fate = {}
    for objects in train_objects:
        for _, _, h, w, n in objects:
            shapes_by_fate.setdefault((h, w), set()).add(n <= 1)
    assert any(fates == {False, True} for fates in shapes_by_fate.values())


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    sampled = _sample_puzzle_parameters(rng)
    roles = (sampled["background"], sampled["carrier"], sampled["indicator"])
    train_count, test_count = _resolve_pair_counts(rng, seed, num_train, num_test)
    contrast_shape = _CONTRAST_SHAPES[int(rng.integers(len(_CONTRAST_SHAPES)))]
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max, size=train_count+test_count, dtype=np.int64)
    puzzle = {"train": [], "test": []}
    seen = set()
    index = 0
    for split, regime, count in (("train", "source_train", train_count), ("test", "target_test", test_count)):
        for pair_index in range(count):
            pair_rng = np.random.default_rng(int(pair_seeds[index]))
            index += 1
            for _ in range(20):
                inp, out = _generate_pair(pair_rng, regime, pair_index, roles, contrast_shape)
                signature = _grid_signature(inp)
                if signature not in seen:
                    break
            else:
                raise RuntimeError("could not produce a unique input")
            seen.add(signature)
            puzzle[split].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first, second):
    return all(np.array_equal(a[role], b[role])
               for split in ("train", "test")
               for a, b in zip(first[split], second[split])
               for role in ("input", "output")) and all(
                   len(first[split]) == len(second[split]) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_infer_roles_from_train(p["train"]) for p in puzzles}) >= 20
    features = {"train": [], "test": []}
    seen_layouts = set()
    edges = 0
    near = far = 0
    contrast_wide = 0
    for puzzle in puzzles:
        train_shapes = {}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                roles = _infer_roles_from_train(puzzle["train"])
                objects = _analyze_pair(pair["input"], pair["output"], *roles)
                features[split].append((pair["input"].shape, len(objects), tuple(o[4] for o in objects)))
                seen_layouts.add((pair["input"].shape, tuple(objects)))
                H, W = pair["input"].shape
                edges += any(r == 0 or c == 0 or r+h == H or c+w == W for r, c, h, w, _ in objects)
                for r, c, h, w, n in objects:
                    if split == "train":
                        train_shapes.setdefault((h, w), set()).add(n <= 1)
                    if n == 2:
                        ys, xs = np.where(pair["input"][r:r+h, c:c+w] == roles[2])
                        distance = abs(int(ys[0])-int(ys[1])) + abs(int(xs[0])-int(xs[1]))
                        near += distance <= 2
                        far += distance >= max(h, w) - 1
        contrast_wide += any(w >= 5 and fates == {False, True} for (h, w), fates in train_shapes.items())
    total = len(features["train"]) + len(features["test"])
    assert len(seen_layouts) > total * 0.95
    assert edges > total * 0.45
    assert near > 10 and far > 10
    assert contrast_wide >= 8

    # Measure review properties from returned grids, rather than from
    # requested sampling modes or generation metadata.
    measured = {split: {} for split in ("train", "test")}
    def add(split, name, value=True):
        measured[split][name] = measured[split].get(name, 0) + int(value)

    shared_dimensions = smaller_tests = 0
    for puzzle in puzzles:
        roles = _infer_roles_from_train(puzzle["train"])
        train_shapes = {pair["input"].shape for pair in puzzle["train"]}
        largest_train = max(pair["input"].size for pair in puzzle["train"])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                H, W = grid.shape
                objects = _analyze_pair(grid, pair["output"], *roles)
                counts = [obj[4] for obj in objects]
                add(split, "small", grid.size <= 156)
                add(split, "large", grid.size >= 350)
                add(split, "square", H == W)
                add(split, "landscape", H < W)
                add(split, "portrait", H > W)
                add(split, "all_keep", max(counts) <= 1)
                add(split, "all_erase", min(counts) >= 2)
                add(split, "missing_count_class", not {0, 1, 2, 3} <= {min(n, 3) for n in counts})
                add(split, "edge_and_wide", any(
                    (r == 0 or c == 0 or r+h == H or c+w == W)
                    and (h >= 5 or w >= 5)
                    for r, c, h, w, _ in objects))
                if split == "test":
                    shared_dimensions += grid.shape in train_shapes
                    smaller_tests += grid.size < largest_train
                for r, c, h, w, n in objects:
                    add(split, "wide_keep", w >= 5 and n <= 1)
                    add(split, "wide_erase", w >= 5 and n >= 2)
                    add(split, "tall_keep", h >= 5 and n <= 1)
                    add(split, "tall_erase", h >= 5 and n >= 2)
                    add(split, "top", r == 0)
                    add(split, "bottom", r+h == H)
                    add(split, "left", c == 0)
                    add(split, "right", c+w == W)
                    add(split, "corner", (r == 0 or r+h == H) and (c == 0 or c+w == W))
                    if n == 1:
                        ys, xs = np.where(grid[r:r+h, c:c+w] == roles[2])
                        y, x = int(ys[0]), int(xs[0])
                        add(split, "single_corner", y in (0, h-1) and x in (0, w-1))
                        add(split, "single_interior", 0 < y < h-1 and 0 < x < w-1)
    for split in ("train", "test"):
        assert all(measured[split].get(name, 0) > 0 for name in (
            "small", "large", "square", "landscape", "portrait",
            "all_keep", "all_erase", "missing_count_class",
            "edge_and_wide", "wide_keep", "wide_erase",
            "tall_keep", "tall_erase", "top", "bottom", "left",
            "right", "corner", "single_corner", "single_interior"))
    assert shared_dimensions > 0 and smaller_tests > 0

    for offset, (nt, ne) in enumerate(((2, 1), (10, 1), (10, 4))):
        p = generate_puzzle(1000+offset, nt, ne)
        assert (len(p["train"]), len(p["test"])) == (nt, ne)
    for offset, (nt, ne) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(2000+offset, nt, ne)
        assert nt is None or len(p["train"]) == nt
        assert ne is None or len(p["test"]) == ne
    assert _puzzles_equal(generate_puzzle(314159, 7, 3), generate_puzzle(314159, 7, 3))
    assert _puzzles_equal(generate_puzzle(271828), generate_puzzle(271828))
    assert not _puzzles_equal(generate_puzzle(5001, 6, 2), generate_puzzle(5002, 6, 2))
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4},
                   {"num_train": True}, {"num_test": 1.5}):
        try:
            generate_puzzle(9999, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")
