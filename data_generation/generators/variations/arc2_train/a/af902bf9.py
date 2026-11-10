import numpy as np


def _grid_from_rows(rows: list[str]) -> np.ndarray:
    return np.array(
        [[int(cell) for cell in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_PAIRS = [
    (
        _grid_from_rows([
            "0000000000",
            "0400004000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0400004000",
            "0000000000",
            "0000000000",
            "0000000000",
        ]),
        _grid_from_rows([
            "0000000000",
            "0400004000",
            "0022220000",
            "0022220000",
            "0022220000",
            "0022220000",
            "0400004000",
            "0000000000",
            "0000000000",
            "0000000000",
        ]),
    ),
    (
        _grid_from_rows([
            "0000000000",
            "0000000000",
            "0000000000",
            "0004040000",
            "0000000000",
            "0004040000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
        ]),
        _grid_from_rows([
            "0000000000",
            "0000000000",
            "0000000000",
            "0004040000",
            "0000200000",
            "0004040000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
        ]),
    ),
    (
        _grid_from_rows([
            "0000000000",
            "0404000000",
            "0000000000",
            "0404000000",
            "0000000000",
            "0000000000",
            "0000400004",
            "0000000000",
            "0000000000",
            "0000400004",
        ]),
        _grid_from_rows([
            "0000000000",
            "0404000000",
            "0020000000",
            "0404000000",
            "0000000000",
            "0000000000",
            "0000400004",
            "0000022220",
            "0000022220",
            "0000400004",
        ]),
    ),
    (
        _grid_from_rows([
            "0000040004",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000040004",
            "4004000000",
            "0000000000",
            "0000000000",
            "4004000000",
            "0000000000",
        ]),
        _grid_from_rows([
            "0000040004",
            "0000002220",
            "0000002220",
            "0000002220",
            "0000040004",
            "4004000000",
            "0220000000",
            "0220000000",
            "4004000000",
            "0000000000",
        ]),
    ),
]


def _coerce_count(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    low, high = (2, 10) if name == "num_train" else (1, 4)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, seed, num_train, num_test):
    train = _coerce_count(num_train, "num_train")
    test = _coerce_count(num_test, "num_test")
    if train is None and test is None:
        test = 1 + seed % 4
        train = max(2, test + 1) + (seed // 4) % (11 - max(2, test + 1))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must be greater than num_test")
    return train, test


def _grid_key(grid):
    normalized = np.ascontiguousarray(grid.astype(np.int8, copy=False))
    return normalized.shape, normalized.tobytes()


def _pair_key(pair):
    return _grid_key(pair["input"]), _grid_key(pair["output"])


def _is_source_pair(input_grid, output_grid):
    return any(np.array_equal(input_grid, source_input) and
               np.array_equal(output_grid, source_output)
               for source_input, source_output in _SOURCE_PAIRS)


def _partition(rng, length, count):
    """Disjoint inclusive intervals; spare cells vary widths and gaps."""
    assert length >= 3 * count
    extra = length - 3 * count
    allocations = np.zeros(2 * count - 1, dtype=int)
    if extra:
        draws = rng.integers(0, len(allocations), size=extra)
        for index in draws:
            allocations[index] += 1
    intervals = []
    cursor = 0
    for index in range(count):
        end = cursor + 2 + int(allocations[2 * index])
        intervals.append((cursor, end))
        cursor = end + 1
        if index < count - 1:
            cursor += int(allocations[2 * index + 1])
    assert intervals[-1][1] == length - 1
    return intervals


def _zones(rng, height, width, count, layout):
    if count == 1:
        return [(0, height - 1, 0, width - 1)]
    if layout == "side":
        return [(0, height - 1, c0, c1)
                for c0, c1 in _partition(rng, width, count)]
    if layout == "stack":
        return [(r0, r1, 0, width - 1)
                for r0, r1 in _partition(rng, height, count)]
    rows = _partition(rng, height, 2)
    columns = _partition(rng, width, 2)
    cells = [(r0, r1, c0, c1) for r0, r1 in rows for c0, c1 in columns]
    rng.shuffle(cells)
    return cells[:count]


def _extent(rng, maximum):
    maximum = min(14, maximum)
    draw = float(rng.random())
    if draw < 0.17:
        return 1
    if draw < 0.32:
        return maximum
    return int(rng.integers(1, maximum + 1))


def _side_set(rect, height, width):
    r0, r1, c0, c1 = rect[:4]
    sides = set()
    if r0 == 0:
        sides.add("top")
    if r1 == height - 1:
        sides.add("bottom")
    if c0 == 0:
        sides.add("left")
    if c1 == width - 1:
        sides.add("right")
    return frozenset(sides)


def _position_in_zone(rng, zone, height, width, shape, required):
    zr0, zr1, zc0, zc1 = zone
    inner_h, inner_w = shape
    span_h, span_w = inner_h + 2, inner_w + 2
    if span_h > zr1 - zr0 + 1 or span_w > zc1 - zc0 + 1:
        return None
    r_starts = list(range(zr0, zr1 - span_h + 2))
    c_starts = list(range(zc0, zc1 - span_w + 2))
    if required == "inset":
        r_starts = [r for r in r_starts if r > 0 and r + span_h - 1 < height - 1]
        c_starts = [c for c in c_starts if c > 0 and c + span_w - 1 < width - 1]
    else:
        if "top" in required:
            r_starts = [r for r in r_starts if r == 0]
        if "bottom" in required:
            r_starts = [r for r in r_starts if r + span_h - 1 == height - 1]
        if "left" in required:
            c_starts = [c for c in c_starts if c == 0]
        if "right" in required:
            c_starts = [c for c in c_starts if c + span_w - 1 == width - 1]
    if not r_starts or not c_starts:
        return None
    r0 = int(rng.choice(r_starts))
    c0 = int(rng.choice(c_starts))
    return (r0, r0 + span_h - 1, c0, c0 + span_w - 1)


def _candidate_rectangles(grid, background):
    """Find every four-corner object whose remaining box is background."""
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        if color == background:
            continue
        coords = [tuple(map(int, point)) for point in np.argwhere(grid == color)]
        if len(coords) % 4:
            return []
        rows = sorted({r for r, _ in coords})
        columns = sorted({c for _, c in coords})
        points = set(coords)
        for i, r0 in enumerate(rows):
            for r1 in rows[i + 1:]:
                if r1 - r0 < 2:
                    continue
                for j, c0 in enumerate(columns):
                    for c1 in columns[j + 1:]:
                        if c1 - c0 < 2:
                            continue
                        corners = frozenset(((r0, c0), (r0, c1),
                                             (r1, c0), (r1, c1)))
                        if not corners <= points:
                            continue
                        box = grid[r0:r1 + 1, c0:c1 + 1].copy()
                        box[0, 0] = background
                        box[0, -1] = background
                        box[-1, 0] = background
                        box[-1, -1] = background
                        if np.all(box == background):
                            candidates.append(((r0, r1, c0, c1, color), corners))
    return candidates


def _parse_input(grid, background=None):
    """Return the sole exact cover of marker cells, or reject ambiguity."""
    values, counts = np.unique(grid, return_counts=True)
    if background is None:
        background = int(values[np.argmax(counts)])
    markers = frozenset((int(r), int(c)) for r, c in np.argwhere(grid != background))
    if not markers or len(markers) % 4:
        raise AssertionError("markers must partition into four-corner groups")
    candidates = _candidate_rectangles(grid, background)
    by_point = {point: [] for point in markers}
    for rect, corners in candidates:
        for point in corners:
            by_point[point].append((rect, corners))
    solutions = []

    def search(remaining, chosen):
        if len(solutions) >= 2:
            return
        if not remaining:
            solutions.append(tuple(sorted(chosen)))
            return
        point = min(remaining, key=lambda p: len(by_point[p]))
        for rect, corners in by_point[point]:
            if not corners <= remaining:
                continue
            r0, r1, c0, c1 = rect[:4]
            if any(not (r1 < old[0] or old[1] < r0 or
                        c1 < old[2] or old[3] < c0) for old in chosen):
                continue
            search(remaining - corners, chosen + [rect])

    search(markers, [])
    if len(solutions) != 1:
        raise AssertionError("corner grouping is absent or ambiguous")
    return solutions[0]


def _construct_output(input_grid, rectangles, fill):
    output = input_grid.copy()
    for r0, r1, c0, c1, _ in rectangles:
        output[r0 + 1:r1, c0 + 1:c1] = fill
    return output


def _shape(rect):
    return rect[1] - rect[0] - 1, rect[3] - rect[2] - 1


def _make_pair(rng, fill, count, background, mixed_markers,
               boundary, withheld=False, forced_shape=None):
    sides = ("top", "bottom", "left", "right",
             "top+left", "top+right", "bottom+left", "bottom+right")
    assert boundary in ("inset", "any") + sides
    required = (frozenset() if boundary in ("any", "inset")
                else frozenset(boundary.split("+")))
    for _ in range(140):
        layout = "single" if count == 1 else str(rng.choice(
            ["side", "stack", "grid"] if count >= 3 else ["side", "stack"]))
        min_height = 3 * count if layout == "stack" else (6 if layout == "grid" else 5)
        min_width = 3 * count if layout == "side" else (6 if layout == "grid" else 5)
        height = int(rng.integers(min_height, 31))
        width = int(rng.integers(min_width, 31))
        zones = _zones(rng, height, width, count, layout)
        eligible = []
        for index, zone in enumerate(zones):
            zr0, zr1, zc0, zc1 = zone
            possible = set()
            if zr0 == 0:
                possible.add("top")
            if zr1 == height - 1:
                possible.add("bottom")
            if zc0 == 0:
                possible.add("left")
            if zc1 == width - 1:
                possible.add("right")
            if required <= possible:
                eligible.append(index)
        if not eligible:
            continue
        target = int(rng.choice(eligible))
        rectangles = []
        successful = True
        for index, zone in enumerate(zones):
            max_h = zone[1] - zone[0] - 1
            max_w = zone[3] - zone[2] - 1
            shape = forced_shape if index == target and forced_shape else (
                _extent(rng, max_h), _extent(rng, max_w))
            if withheld and shape in ((2, 2), (3, 3)):
                successful = False
                break
            request = boundary if boundary == "inset" or index == target else "any"
            rect = _position_in_zone(rng, zone, height, width, shape, request)
            if rect is None:
                successful = False
                break
            rectangles.append(rect)
        if not successful:
            continue
        palette = [color for color in range(10) if color not in (background, fill)]
        if mixed_markers and count > 1:
            colors = [int(x) for x in rng.choice(palette, size=count, replace=False)]
        else:
            colors = [int(rng.choice(palette))] * count
        input_grid = np.full((height, width), background, dtype=np.int8)
        colored = []
        for (r0, r1, c0, c1), color in zip(rectangles, colors):
            for r, c in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
                input_grid[r, c] = color
            colored.append((r0, r1, c0, c1, color))
        try:
            parsed = _parse_input(input_grid, background)
        except AssertionError:
            continue
        if set(parsed) != set(colored):
            continue
        output_grid = _construct_output(input_grid, parsed, fill)
        if _is_source_pair(input_grid, output_grid):
            continue
        return {"input": input_grid, "output": output_grid}
    raise ValueError("unable to place a uniquely groupable rectangle layout")


def _analyze_pair(pair):
    input_grid, output_grid = pair["input"], pair["output"]
    assert input_grid.shape == output_grid.shape
    changed = input_grid != output_grid
    assert np.any(changed)
    origins = np.unique(input_grid[changed])
    fills = np.unique(output_grid[changed])
    assert len(origins) == len(fills) == 1
    background, fill = int(origins[0]), int(fills[0])
    assert background != fill
    assert fill not in input_grid
    values, counts = np.unique(input_grid, return_counts=True)
    assert background == int(values[np.argmax(counts)])
    rectangles = _parse_input(input_grid, background)
    assert np.array_equal(output_grid, _construct_output(input_grid, rectangles, fill))
    assert set(np.unique(output_grid)) == set(np.unique(input_grid)) | {fill}
    return {"background": background, "fill": fill,
            "rectangles": rectangles, "shapes": tuple(_shape(rect) for rect in rectangles),
            "colors": frozenset(rect[4] for rect in rectangles),
            "sides": tuple(_side_set(rect, *input_grid.shape) for rect in rectangles)}


def _predict_from_training(train, test_input):
    fill_colors = []
    for pair in train:
        introduced = set(np.unique(pair["output"])) - set(np.unique(pair["input"]))
        assert len(introduced) == 1
        fill_colors.append(int(next(iter(introduced))))
    assert len(set(fill_colors)) == 1
    values, counts = np.unique(test_input, return_counts=True)
    background = int(values[np.argmax(counts)])
    assert fill_colors[0] not in test_input
    return _construct_output(test_input, _parse_input(test_input, background), fill_colors[0])


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed_value = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed_value)
    train_count, test_count = _resolve_counts(rng, seed_value, num_train, num_test)
    fill = int(rng.integers(0, 10))
    # The fill is the only puzzle-wide color convention. Background and marker
    # hues are visible local roles, so they can vary from one pair to another.
    backgrounds = [int(x) for x in rng.permutation([c for c in range(10) if c != fill])]
    regime = int(rng.integers(0, 4))  # withheld size, boundary, mixed, familiar
    train = []
    test = []
    seen = set()
    train_counts = [1, int(rng.integers(2, 4))] + [int(rng.integers(1, 5))
                                                       for _ in range(train_count - 2)]
    rng.shuffle(train_counts)
    boundaries = ("inset", "top", "bottom", "left", "right",
                  "top+left", "top+right", "bottom+left", "bottom+right")
    train_mixed = False
    for index, count in enumerate(train_counts):
        # The first multi-object example demonstrates that a corner hue is a
        # per-object cue. Later pairs may use one hue for several objects.
        mixed = count > 1 and (not train_mixed or bool(rng.integers(0, 2)))
        background = backgrounds[index % len(backgrounds)]
        for _ in range(60):
            boundary = str(rng.choice(boundaries, p=(.36, .08, .08, .08, .08,
                                                      .08, .08, .08, .08)))
            try:
                pair = _make_pair(rng, fill, count, background, mixed,
                                  boundary, withheld=(regime == 0))
            except ValueError:
                continue
            if regime == 1 and any(
                {"top", "right"} <= side for side in _analyze_pair(pair)["sides"]
            ):
                continue
            key = _grid_key(pair["input"])
            if key not in seen:
                break
        else:
            raise ValueError("unable to make unique training pair")
        seen.add(key)
        train.append(pair)
        train_mixed |= mixed
    familiar = None
    if regime == 3:
        options = []
        for pair in train:
            analysis = _analyze_pair(pair)
            for rect, side in zip(analysis["rectangles"], analysis["sides"]):
                if len(side) <= 2 and side not in (
                    frozenset(("top", "bottom")),
                    frozenset(("left", "right")),
                ):
                    options.append((_shape(rect), side))
        if options:
            familiar = options[int(rng.integers(0, len(options)))]
    for index in range(test_count):
        count = int(rng.integers(1, 5))
        if regime == 0 and index == 0:
            count = 2
        if regime == 3 and index == 0 and familiar is not None:
            count = 1
        background = backgrounds[(train_count + index) % len(backgrounds)]
        mixed = count > 1 and train_mixed and bool(rng.integers(0, 2))
        for _ in range(60):
            if index == 0 and regime in (0, 1):
                boundary = "top+right"
            elif index == 0 and regime == 3 and familiar is not None:
                boundary = ("inset" if not familiar[1] else "+".join(
                    side for side in ("top", "bottom", "left", "right")
                    if side in familiar[1]))
            else:
                boundary = str(rng.choice(
                    boundaries, p=(.16, .105, .105, .105, .105,
                                   .105, .105, .105, .105)))
            if index == 0 and regime == 0:
                forced = (2, 2) if bool(rng.integers(0, 2)) else (3, 3)
            elif index == 0 and regime == 3 and familiar is not None:
                forced = familiar[0]
            else:
                forced = None
            try:
                pair = _make_pair(rng, fill, count, background, mixed,
                                  boundary, forced_shape=forced)
            except ValueError:
                continue
            key = _grid_key(pair["input"])
            if key not in seen:
                break
        else:
            raise ValueError("unable to make unique test pair")
        seen.add(key)
        test.append(pair)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            input_key, pair_key = _grid_key(pair["input"]), _pair_key(pair)
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert not _is_source_pair(pair["input"], pair["output"])
            analyses[split].append(_analyze_pair(pair))
    fills = {analysis["fill"] for split in analyses for analysis in analyses[split]}
    assert len(fills) == 1
    assert any(len(item["rectangles"]) == 1 for item in analyses["train"])
    assert any(len(item["rectangles"]) >= 2 for item in analyses["train"])
    assert len({item["background"] for item in analyses["train"]}) >= 2
    if any(len(item["colors"]) > 1 for item in analyses["test"]):
        assert any(len(item["colors"]) > 1 for item in analyses["train"])
    # The output-only fill color is demonstrated by every training pair.
    for pair in puzzle["test"]:
        assert np.array_equal(_predict_from_training(puzzle["train"], pair["input"]),
                              pair["output"])


def _puzzles_equal(first, second):
    return all(len(first[split]) == len(second[split]) and
               all(np.array_equal(a["input"], b["input"]) and
                   np.array_equal(a["output"], b["output"])
                   for a, b in zip(first[split], second[split]))
               for split in ("train", "test"))


def _require_value_error(**kwargs):
    try:
        generate_puzzle(12345, **kwargs)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def self_test() -> None:
    assert _puzzles_equal(generate_puzzle(7319), generate_puzzle(7319))
    assert _puzzles_equal(generate_puzzle(918273, 8, 3), generate_puzzle(918273, 8, 3))
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(20000 + index, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for index, (train, test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(30000 + index, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        _require_value_error(**kwargs)

    train_counts, test_counts, fingerprints = set(), set(), set()
    grid_shapes, layouts, shapes, fills, backgrounds = set(), set(), set(), set(), set()
    counts = {"train": set(), "test": set()}
    sides = {"train": set(), "test": set()}
    stats = {name: 0 for name in (
        "train_inset", "test_inset", "train_corner", "test_corner",
        "train_fully_inset", "test_fully_inset", "test_large", "test_small",
        "test_wide", "test_tall", "test_square", "test_nonsquare",
        "test_thin_h", "test_thin_v", "side", "stack", "staggered",
        "same_marker_multi", "mixed_marker_multi", "first_single",
        "first_multi", "larger_first", "smaller_first", "withheld",
        "boundary_shift", "familiar", "combined", "background_variation", "count_variation",
        "shape_variation", "size_variation")}
    gap_values = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        fingerprints.add(tuple((split, _pair_key(pair))
                               for split in ("train", "test") for pair in puzzle[split]))
        analyses = {split: [_analyze_pair(pair) for pair in puzzle[split]]
                    for split in ("train", "test")}
        all_analyses = analyses["train"] + analyses["test"]
        all_pairs = puzzle["train"] + puzzle["test"]
        fills.add(analyses["train"][0]["fill"])
        backgrounds.update(item["background"] for item in all_analyses)
        stats["background_variation"] += len({x["background"] for x in analyses["train"]}) > 1
        stats["count_variation"] += len({len(x["rectangles"]) for x in all_analyses}) > 1
        stats["shape_variation"] += len({shape for x in all_analyses for shape in x["shapes"]}) > 1
        stats["size_variation"] += len({x["input"].shape for x in all_pairs}) > 1
        stats["first_single"] += len(analyses["train"][0]["rectangles"]) == 1
        stats["first_multi"] += len(analyses["train"][0]["rectangles"]) > 1
        train_shapes = {shape for x in analyses["train"] for shape in x["shapes"]}
        test_shapes = {shape for x in analyses["test"] for shape in x["shapes"]}
        stats["withheld"] += not bool(train_shapes & {(2, 2), (3, 3)}) and bool(
            test_shapes & {(2, 2), (3, 3)})
        stats["boundary_shift"] += (
            not any({"top", "right"} <= side for x in analyses["train"] for side in x["sides"])
            and any({"top", "right"} <= side for x in analyses["test"] for side in x["sides"]))
        stats["familiar"] += bool(train_shapes & test_shapes)
        for split in ("train", "test"):
            for pair, analysis in zip(puzzle[split], analyses[split]):
                height, width = pair["input"].shape
                rectangles = analysis["rectangles"]
                counts[split].add(len(rectangles))
                grid_shapes.add((height, width))
                layouts.add((split, (height, width), rectangles))
                shapes.update(analysis["shapes"])
                sides[split].update(analysis["sides"])
                stats[f"{split}_inset"] += sum(not side for side in analysis["sides"])
                stats[f"{split}_corner"] += sum(len(side) == 2 for side in analysis["sides"])
                stats[f"{split}_fully_inset"] += all(not side for side in analysis["sides"])
                if split == "test":
                    stats["test_large"] += height * width >= 500
                    stats["test_small"] += height * width <= 100
                    stats["test_wide"] += width >= 2 * height
                    stats["test_tall"] += height >= 2 * width
                    stats["test_square"] += any(h == w for h, w in analysis["shapes"])
                    stats["test_nonsquare"] += any(h != w for h, w in analysis["shapes"])
                    stats["test_thin_h"] += any(h == 1 and w > 1 for h, w in analysis["shapes"])
                    stats["test_thin_v"] += any(w == 1 and h > 1 for h, w in analysis["shapes"])
                    stats["combined"] += (len(rectangles) >= 3 and height * width >= 500
                                          and any(len(side) == 2 for side in analysis["sides"])
                                          and any(h != w for h, w in analysis["shapes"]))
                if len(rectangles) > 1:
                    stats["same_marker_multi"] += len(analysis["colors"]) == 1
                    stats["mixed_marker_multi"] += len(analysis["colors"]) > 1
                    first_area = _shape(rectangles[0])[0] * _shape(rectangles[0])[1]
                    second_area = _shape(rectangles[1])[0] * _shape(rectangles[1])[1]
                    stats["larger_first"] += first_area > second_area
                    stats["smaller_first"] += first_area < second_area
                    for i, a in enumerate(rectangles):
                        for b in rectangles[i + 1:]:
                            row_overlap = max(a[0], b[0]) <= min(a[1], b[1])
                            col_overlap = max(a[2], b[2]) <= min(a[3], b[3])
                            if row_overlap and not col_overlap:
                                stats["side"] += 1
                                gap_values.add(max(a[2], b[2]) - min(a[3], b[3]) - 1)
                                stats["staggered"] += (a[0], a[1]) != (b[0], b[1])
                            if col_overlap and not row_overlap:
                                stats["stack"] += 1
                                gap_values.add(max(a[0], b[0]) - min(a[1], b[1]) - 1)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(fingerprints) == 42 and len(grid_shapes) >= 80
    assert len(layouts) >= 200 and len(shapes) >= 60
    assert len(fills) >= 8 and len(backgrounds) == 10
    assert counts["test"] == {1, 2, 3, 4}
    assert counts["train"] == {1, 2, 3, 4}
    for split in ("train", "test"):
        assert all(any(side_name in side for side in sides[split])
                   for side_name in ("top", "bottom", "left", "right"))
        assert frozenset() in sides[split]
        assert any(len(side) == 2 for side in sides[split])
    for name in ("train_inset", "test_inset", "train_corner", "test_corner",
                 "train_fully_inset", "test_fully_inset", "test_large", "test_small",
                 "test_wide", "test_tall", "test_square", "test_nonsquare",
                 "test_thin_h", "test_thin_v", "side", "stack", "staggered",
                 "same_marker_multi", "mixed_marker_multi", "first_single",
                 "first_multi", "larger_first", "smaller_first", "withheld",
                 "boundary_shift", "familiar", "combined"):
        assert stats[name] >= 1, name
    assert len(gap_values) >= 8 and 0 in gap_values and max(gap_values) >= 5
    for name in ("background_variation", "count_variation", "shape_variation",
                 "size_variation"):
        assert stats[name] == 42, name
