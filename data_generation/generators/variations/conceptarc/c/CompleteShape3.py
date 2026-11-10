"""Generate incomplete, single-color rectangular outlines for ARC."""

import numpy as np


_SOURCE_INPUTS = (
    ("0000000000", "0000000000", "0400440000", "0400000000",
     "0000000400", "0400000400", "0000000400", "0440044400",
     "0000000000", "0000000000"),
    ("0000000000", "0000000000", "0666666060", "0000000060",
     "0600000060", "0000000000", "0600000060", "0660666060",
     "0000000000", "0000000000"),
    ("0000000000", "0000000000", "0777077070", "0700000000",
     "0000000070", "0700000070", "0000000070", "0700000070",
     "0700000000", "0000707770"),
    ("00000000000000", "00000000000000", "00000000000000",
     "00800800080800", "00000000000000", "00000000000000",
     "00000000000800", "00000000000000", "00800000000000",
     "00000000000800", "00800000000000", "00000000000000",
     "00800008000800", "00000000000000", "00000000000000"),
    ("0000000000000000", "0000000000000000", "0000000000000000",
     "0009000900009000", "0000000000000000", "0009000000009000",
     "0000000000000000", "0000000000000000", "0009000000009000",
     "0000000000000000", "0000000000000000"),
)
_SOURCE_KEYS = {(len(x), len(x[0]), "".join(x)) for x in _SOURCE_INPUTS}


def _valid_count(value, low, high, name):
    if value is not None and (isinstance(value, bool)
                              or not isinstance(value, (int, np.integer))
                              or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


def _key(grid):
    return grid.shape, tuple(int(x) for x in grid.flat)


def _source_key(grid):
    return grid.shape[0], grid.shape[1], "".join(str(int(x)) for x in grid.flat)


def _sides(top, bottom, left, right):
    return (
        [(top, c) for c in range(left, right + 1)],
        [(bottom, c) for c in range(left, right + 1)],
        [(r, left) for r in range(top, bottom + 1)],
        [(r, right) for r in range(top, bottom + 1)],
    )


def _perimeter(top, bottom, left, right):
    return ([(top, c) for c in range(left, right + 1)]
            + [(r, right) for r in range(top + 1, bottom + 1)]
            + [(bottom, c) for c in range(right - 1, left - 1, -1)]
            + [(r, left) for r in range(bottom - 1, top, -1)])


def _dimension(rng):
    band = int(rng.choice(3, p=(0.34, 0.38, 0.28)))
    low, high = ((8, 12), (11, 20), (19, 30))[band]
    return int(rng.integers(low, high + 1))


def _canvas(rng):
    if rng.random() < 0.12:
        return 10, 10
    height = _dimension(rng)
    if rng.random() < 0.18:
        width = int(np.clip(height + rng.integers(-2, 3), 8, 30))
    else:
        width = _dimension(rng)
    return height, width


def _frame_dimension(rng, canvas_dimension, mode):
    if mode == "small":
        return int(rng.integers(5, min(9, canvas_dimension) + 1))
    if mode == "large":
        return canvas_dimension - int(rng.integers(0, min(4, canvas_dimension - 5) + 1))
    if mode == "medium":
        low = max(5, int(round(canvas_dimension * 0.42)))
        high = max(low, int(round(canvas_dimension * 0.82)))
        return int(rng.integers(low, high + 1))
    return int(rng.integers(5, canvas_dimension + 1))


def _position(rng, slack):
    if slack == 0:
        return 0
    roll = rng.random()
    if roll < 0.23:
        return 0
    if roll < 0.46:
        return slack
    return int(rng.integers(0, slack + 1))


def _fragment(rng, box, density_band):
    top, bottom, left, right = box
    perimeter = _perimeter(*box)
    sides = _sides(*box)
    size = len(perimeter)
    bounds = {point: i for i, point in enumerate(perimeter)}
    ranges = {"dense": (0.62, 0.84), "moderate": (0.46, 0.66),
              "sparse": (0.24, 0.42)}
    low, high = ranges[density_band]
    count = max(7, int(np.ceil(low * size)),
                min(int(np.floor(high * size)),
                    int(round(float(rng.uniform(low, high)) * size))))
    count = min(size - 2, count)
    style = str(rng.choice(("scattered", "runs", "mixed", "intact", "corners"),
                           p=(0.18, 0.29, 0.28, 0.13, 0.12)))
    allowed = set(perimeter)
    chosen = set()

    if style == "intact":
        eligible = [side for side in sides if len(side) <= count - 1]
        if eligible:
            chosen.update(eligible[int(rng.integers(len(eligible)))])
        else:
            style = "mixed"
    if style == "corners":
        side = sides[int(rng.integers(4))]
        chosen.update((side[0], side[-1]))
        allowed.difference_update(side[1:-1])
        count = min(count, len(allowed) - 1)

    # Each visible extreme must coincide with the intended extreme.
    for side in sides:
        if not chosen.intersection(side):
            candidates = [point for point in side if point in allowed]
            chosen.add(candidates[int(rng.integers(len(candidates)))])

    scores = rng.random(size)
    if style in ("runs", "mixed", "intact"):
        number_of_runs = int(rng.integers(2, 6)) if style == "runs" else int(rng.integers(1, 4))
        for _ in range(number_of_runs):
            start = int(rng.integers(size))
            length = int(rng.integers(2, max(4, size // 4) + 1))
            boost = float(rng.uniform(1.2, 2.4))
            for offset in range(length):
                scores[(start + offset) % size] += boost
    candidates = sorted((point for point in allowed if point not in chosen),
                        key=lambda point: scores[bounds[point]], reverse=True)
    chosen.update(candidates[:count - len(chosen)])
    assert len(chosen) == count
    assert count < size
    return chosen


def _make_pair(rng, background, outline, geometry_mode, density_band):
    height, width = _canvas(rng)
    frame_height = _frame_dimension(rng, height, geometry_mode)
    frame_width = _frame_dimension(rng, width, geometry_mode)
    if rng.random() < 0.23:
        frame_height = min(frame_height, height - 2)
        frame_width = min(frame_width, width - 2)
        top = int(rng.integers(1, height - frame_height))
        left = int(rng.integers(1, width - frame_width))
    else:
        top = _position(rng, height - frame_height)
        left = _position(rng, width - frame_width)
    bottom = top + frame_height - 1
    right = left + frame_width - 1
    box = (top, bottom, left, right)
    output = np.full((height, width), background, dtype=np.int8)
    output[top, left:right + 1] = outline
    output[bottom, left:right + 1] = outline
    output[top:bottom + 1, left] = outline
    output[top:bottom + 1, right] = outline
    input_grid = np.full((height, width), background, dtype=np.int8)
    for row, col in _fragment(rng, box, density_band):
        input_grid[row, col] = outline
    return {"input": input_grid, "output": output}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    _valid_count(num_train, 2, 10, "num_train")
    _valid_count(num_test, 1, 4, "num_test")
    if num_train is None and num_test is None:
        test_count = int(rng.integers(1, 5))
        train_count = int(rng.integers(max(2, test_count + 1), 11))
    elif num_train is None:
        test_count = int(num_test)
        train_count = int(rng.integers(max(2, test_count + 1), 11))
    elif num_test is None:
        train_count = int(num_train)
        test_count = int(rng.integers(1, min(4, train_count - 1) + 1))
    else:
        train_count, test_count = int(num_train), int(num_test)
    if train_count <= test_count:
        raise ValueError("num_train must exceed num_test")

    # The transformation is the only learned convention. Colors are local roles.
    palette = [int(x) for x in rng.permutation(10)]
    nonzero = [x for x in palette if x != 0]
    first_background = 0 if rng.random() < 0.55 else nonzero[0]
    train_backgrounds = [first_background,
                         nonzero[1] if first_background == 0 else 0]
    for _ in range(2, train_count):
        train_backgrounds.append(0 if rng.random() < 0.55
                                 else nonzero[int(rng.integers(len(nonzero)))])
    train_outlines = []
    for background in train_backgrounds:
        if background != 0 and rng.random() < 0.38:
            train_outlines.append(0)
        else:
            train_outlines.append(nonzero[int(rng.integers(len(nonzero)))])
            if train_outlines[-1] == background:
                train_outlines[-1] = next(x for x in nonzero if x != background)
    if train_outlines[1] == train_outlines[0]:
        train_outlines[1] = next(x for x in palette
                                 if x != train_backgrounds[1]
                                 and x != train_outlines[0])

    geometry_modes = ["small", "medium", "large", "free"]
    initial_modes = [str(x) for x in rng.choice(geometry_modes, size=2, replace=False)]
    train_modes = initial_modes + [str(rng.choice(geometry_modes))
                                   for _ in range(2, train_count)]
    train_bands = ["dense", str(rng.choice(("moderate", "sparse")))]
    train_bands += [str(rng.choice(("dense", "moderate", "sparse")))
                    for _ in range(2, train_count)]
    # Shuffle pair plans so pair number does not dictate geometry or damage.
    order = [int(x) for x in rng.permutation(train_count)]
    train_plan = [(train_backgrounds[i], train_outlines[i],
                   train_modes[i], train_bands[i]) for i in order]

    puzzle = {"train": [], "test": []}
    seen = set()
    train_geometries = set()

    def append_pair(section, background, outline, mode, band):
        for _ in range(80):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            pair = _make_pair(pair_rng, background, outline, mode, band)
            input_grid = pair["input"]
            key = _key(input_grid)
            frame_rows, frame_cols = np.where(pair["output"] == outline)
            geometry = (input_grid.shape,
                        int(frame_rows.max() - frame_rows.min() + 1),
                        int(frame_cols.max() - frame_cols.min() + 1))
            if (key in seen or _source_key(input_grid) in _SOURCE_KEYS
                    or (section == "train" and geometry in train_geometries)):
                continue
            seen.add(key)
            if section == "train":
                train_geometries.add(geometry)
            puzzle[section].append(pair)
            return
        raise RuntimeError("Could not construct a distinct valid pair")

    for background, outline, mode, band in train_plan:
        append_pair("train", background, outline, mode, band)

    for _ in range(test_count):
        background = 0 if rng.random() < 0.52 else nonzero[int(rng.integers(len(nonzero)))]
        if rng.random() < 0.48:
            eligible = [x for x in train_outlines if x != background]
            outline = int(rng.choice(eligible)) if eligible else next(x for x in palette if x != background)
        else:
            eligible = [x for x in palette if x != background and x not in train_outlines]
            if not any(x == 0 for x in train_outlines):
                eligible = [x for x in eligible if x != 0]
            outline = int(rng.choice(eligible)) if eligible else next(x for x in palette if x != background)
        mode = str(rng.choice(geometry_modes))
        band = str(rng.choice(("dense", "moderate", "sparse"), p=(0.28, 0.34, 0.38)))
        if band == "sparse" and "sparse" not in train_bands:
            band = "moderate"
        append_pair("test", background, outline, mode, band)

    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    grid = pair["input"]
    values, counts = np.unique(grid, return_counts=True)
    assert len(values) == 2
    background = int(values[int(np.argmax(counts))])
    outline = int(values[int(np.argmin(counts))])
    rows, cols = np.where(grid == outline)
    top, bottom = int(rows.min()), int(rows.max())
    left, right = int(cols.min()), int(cols.max())
    return background, outline, (top, bottom, left, right)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    train_backgrounds, train_outlines, train_geometry = set(), set(), set()
    train_densities, test_densities = [], []
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = pair["input"], pair["output"]
            for item in (grid, output):
                assert isinstance(item, np.ndarray) and item.ndim == 2
                assert all(8 <= x <= 30 for x in item.shape)
                assert np.issubdtype(item.dtype, np.integer)
                assert np.all((item >= 0) & (item <= 9))
            assert grid.shape == output.shape
            key = _key(grid)
            pair_key = (key, _key(output))
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            assert _source_key(grid) not in _SOURCE_KEYS
            background, outline, box = _features(pair)
            assert background != outline
            top, bottom, left, right = box
            assert bottom - top >= 4 and right - left >= 4
            sides = _sides(*box)
            perimeter = {point for side in sides for point in side}
            rows, cols = np.where(grid == outline)
            visible = set(zip(rows.tolist(), cols.tolist()))
            assert visible.issubset(perimeter)
            assert all(visible.intersection(side) for side in sides)
            assert 7 <= len(visible) <= len(perimeter) - 2
            assert np.count_nonzero(grid == background) > np.count_nonzero(grid == outline)
            density = len(visible) / len(perimeter)
            expected = np.full(grid.shape, background, dtype=output.dtype)
            expected[top, left:right + 1] = outline
            expected[bottom, left:right + 1] = outline
            expected[top:bottom + 1, left] = outline
            expected[top:bottom + 1, right] = outline
            assert np.array_equal(output, expected)
            assert np.count_nonzero(output == background) > np.count_nonzero(output == outline)
            assert not np.array_equal(grid, output)
            if section == "train":
                train_backgrounds.add(background)
                train_outlines.add(outline)
                train_geometry.add((grid.shape, bottom - top + 1, right - left + 1))
                train_densities.append(density)
            else:
                test_densities.append(density)
                if outline == 0:
                    assert 0 in train_outlines
    assert len(train_backgrounds) >= 2
    assert len(train_outlines) >= 2
    assert len(train_geometry) == len(puzzle["train"])
    if any(density <= 0.45 for density in test_densities):
        assert any(density <= 0.45 for density in train_densities)


def _puzzle_key(puzzle):
    return tuple(tuple((_key(pair["input"]), _key(pair["output"]))
                       for pair in puzzle[section]) for section in ("train", "test"))


def self_test() -> None:
    signatures, train_counts, test_counts = set(), set(), set()
    train_shapes, test_shapes = set(), set()
    frame_sizes, densities, backgrounds, outlines = set(), set(), set(), set()
    orientations, contacts, band_sections = set(), set(), {"train": set(), "test": set()}
    review_counts = {"train": {name: 0 for name in ("source_size", "small", "near",
                                                   "interior", "intact", "corner_only",
                                                   "long_run", "isolated", "nonzero_bg")},
                     "test": {name: 0 for name in ("source_size", "small", "near",
                                                  "interior", "intact", "corner_only",
                                                  "long_run", "isolated", "nonzero_bg",
                                                  "reused", "new")}}
    shared_sizes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        signatures.add(_puzzle_key(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_shapes_here = {pair["input"].shape for pair in puzzle["train"]}
        test_shapes_here = {pair["input"].shape for pair in puzzle["test"]}
        shared_sizes += bool(train_shapes_here & test_shapes_here)
        assert len(train_shapes_here) >= 2
        assert len({tuple(_features(pair)[2][i + 1] - _features(pair)[2][i] + 1
                          for i in (0, 2)) for pair in puzzle["train"]}) >= 2
        train_outline_here = {_features(pair)[1] for pair in puzzle["train"]}
        for section in ("train", "test"):
            for pair in puzzle[section]:
                grid, output = pair["input"], pair["output"]
                background, outline, (top, bottom, left, right) = _features(pair)
                # Independent mask construction from the visible input bounds.
                prediction = np.full(grid.shape, background, dtype=grid.dtype)
                for row in range(top, bottom + 1):
                    for col in range(left, right + 1):
                        if row in (top, bottom) or col in (left, right):
                            prediction[row, col] = outline
                assert np.array_equal(prediction, output)
                (train_shapes if section == "train" else test_shapes).add(grid.shape)
                frame_sizes.add((bottom - top + 1, right - left + 1))
                density = int(np.count_nonzero(grid == outline)) / int(np.count_nonzero(output == outline))
                measured = review_counts[section]
                measured["source_size"] += grid.shape == (10, 10)
                measured["small"] += bottom - top + 1 <= 7 and right - left + 1 <= 7
                measured["near"] += (grid.shape[0] - (bottom - top + 1) <= 2
                                      and grid.shape[1] - (right - left + 1) <= 2)
                measured["interior"] += (top > 0 and bottom < grid.shape[0] - 1
                                         and left > 0 and right < grid.shape[1] - 1)
                measured["nonzero_bg"] += background != 0
                sides = _sides(top, bottom, left, right)
                visible = set(zip(*np.where(grid == outline)))
                measured["intact"] += any(set(side) <= visible for side in sides)
                measured["corner_only"] += any(side[0] in visible and side[-1] in visible
                                               and not set(side[1:-1]) & visible for side in sides)
                order = _perimeter(top, bottom, left, right)
                run_lengths = []
                run = 0
                for point in order + order[:1]:
                    if point in visible:
                        run += 1
                    elif run:
                        run_lengths.append(run)
                        run = 0
                measured["long_run"] += max(run_lengths, default=0) >= 4
                measured["isolated"] += 1 in run_lengths
                if section == "test":
                    measured["reused" if outline in train_outline_here else "new"] += 1
                densities.add(round(density, 2))
                band_sections[section].add("dense" if density >= 0.62 else
                                           "sparse" if density <= 0.43 else "moderate")
                backgrounds.add(background)
                outlines.add(outline)
                orientations.add("wide" if right - left > bottom - top else
                                 "tall" if right - left < bottom - top else "square")
                contacts.add((top == 0, bottom == grid.shape[0] - 1,
                              left == 0, right == grid.shape[1] - 1))
    assert len(signatures) == 42
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(train_shapes) >= 30 and len(test_shapes) >= 20
    assert len(frame_sizes) >= 45 and len(densities) >= 35
    assert len(backgrounds) >= 7 and len(outlines) >= 7
    assert orientations == {"wide", "tall", "square"}
    assert len(contacts) >= 8
    assert shared_sizes >= 1
    assert all(all(count > 0 for count in measured.values())
               for measured in review_counts.values())
    assert all(band_sections[section] == {"dense", "moderate", "sparse"}
               for section in ("train", "test"))
    for counts in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                   (2, None), (10, None)):
        puzzle = generate_puzzle(511, *counts)
        validate_generated_puzzle(puzzle)
        if counts[0] is not None:
            assert len(puzzle["train"]) == counts[0]
        if counts[1] is not None:
            assert len(puzzle["test"]) == counts[1]
    for counts in ((1, 1), (11, 1), (2, 0), (2, 5), (2, 2), (2, 3), (3, 4)):
        try:
            generate_puzzle(511, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid counts accepted: {counts}")
