import numpy as np


_SOURCE_PAIRS = (
    (
        (
            "8888888888",
            "8888888882",
            "8888888882",
            "8888888882",
            "8888888882",
            "8888888882",
            "8888888888",
        ),
        (
            "2222288888",
            "8888888882",
            "8888888882",
            "8888888882",
            "8888888882",
            "8888888882",
            "2222288888",
        ),
    ),
    (
        (
            "9998888888",
            "9888888888",
            "8888888888",
            "8888888888",
        ),
        (
            "9998888999",
            "9888888889",
            "8888888888",
            "8889999888",
        ),
    ),
    (
        (
            "888888888",
            "888888888",
            "488888888",
            "488888888",
            "888888888",
            "888888888",
            "888888888",
        ),
        (
            "844884488",
            "888888884",
            "488888884",
            "488888888",
            "888888888",
            "888888884",
            "448844884",
        ),
    ),
    (
        (
            "88888888888888",
            "88888888888888",
            "88888888555888",
        ),
        (
            "55588855588855",
            "88888888888885",
            "88555888555888",
        ),
    ),
)


def _perimeter_coords(
    height: int,
    width: int,
) -> list[tuple[int, int]]:
    """Return a clockwise perimeter beginning at the top-left cell."""
    coords = [(0, column) for column in range(width)]
    coords += [
        (row, width - 1)
        for row in range(1, height)
    ]
    coords += [
        (height - 1, column)
        for column in range(width - 2, -1, -1)
    ]
    coords += [
        (row, 0)
        for row in range(height - 2, 0, -1)
    ]
    return coords


def _seed_start(mask: np.ndarray) -> int:
    """Find the first cell of the one cyclic True run."""
    starts = np.flatnonzero(mask & ~np.roll(mask, 1))
    if len(starts) != 1:
        raise ValueError(
            "accent cells do not form one cyclic perimeter run"
        )
    return int(starts[0])


def _complete_pattern(
    input_grid: np.ndarray,
    background: int,
    accent: int,
) -> np.ndarray:
    """Complete the alternating k-accent/k-background border."""
    height, width = input_grid.shape
    perimeter = _perimeter_coords(height, width)

    mask = np.array(
        [
            input_grid[row, column] == accent
            for row, column in perimeter
        ],
        dtype=bool,
    )
    k = int(mask.sum())

    if k <= 0 or len(perimeter) % (2 * k) != 0:
        raise ValueError(
            "seed length does not evenly tile the perimeter"
        )

    start = _seed_start(mask)
    output_grid = input_grid.copy()

    for row, column in perimeter:
        output_grid[row, column] = background

    for offset in range(len(perimeter)):
        if offset % (2 * k) < k:
            row, column = perimeter[
                (start + offset) % len(perimeter)
            ]
            output_grid[row, column] = accent

    return output_grid


def _corner_indices(height: int, width: int) -> tuple[int, ...]:
    return (0, width - 1, width + height - 2, 2 * width + height - 3)


def _seed_geometry(height: int, width: int, k: int, start: int) -> tuple[str, tuple[str, ...]]:
    """Classify the occupied cyclic path, counting each corner once."""
    perimeter = _perimeter_coords(height, width)
    length = len(perimeter)
    occupied = {perimeter[(start + i) % length] for i in range(k)}
    corners = _corner_indices(height, width)
    crossings = tuple(
        name for name, index in zip(
            ("top_left", "top_right", "bottom_right", "bottom_left"), corners
        )
        if any((start + i) % length == index for i in range(1, k - 1))
    )
    if len(crossings) >= 2:
        return "multi_corner", crossings
    if crossings:
        return "bend", crossings
    for side, predicate in (
        ("top", lambda r, c: r == 0),
        ("right", lambda r, c: c == width - 1),
        ("bottom", lambda r, c: r == height - 1),
        ("left", lambda r, c: c == 0),
    ):
        if all(predicate(r, c) for r, c in occupied):
            return side, ()
    return "corner_endpoint", ()


def _make_pair(
    shape: tuple[int, int], k: int, start: int, background: int, accent: int
) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    grid = np.full(shape, background, dtype=np.int8)
    perimeter = _perimeter_coords(height, width)
    for offset in range(k):
        grid[perimeter[(start + offset) % len(perimeter)]] = accent
    return grid, _complete_pattern(grid, background, accent)


def _rows_to_grid(rows: tuple[str, ...]) -> np.ndarray:
    return np.array(
        [
            [int(character) for character in row]
            for row in rows
        ],
        dtype=np.int8,
    )


def _is_source_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    for source_input_rows, source_output_rows in _SOURCE_PAIRS:
        source_input = _rows_to_grid(source_input_rows)

        if (
            input_grid.shape != source_input.shape
            or not np.array_equal(input_grid, source_input)
        ):
            continue

        source_output = _rows_to_grid(source_output_rows)
        if np.array_equal(output_grid, source_output):
            return True

    return False


def _grid_signature(grid: np.ndarray) -> tuple:
    return (
        tuple(grid.shape),
        tuple(int(value) for value in grid.flat),
    )


def _pair_signature(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> tuple:
    return (
        _grid_signature(input_grid),
        _grid_signature(output_grid),
    )


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    if num_train is not None:
        if (
            isinstance(num_train, (bool, np.bool_))
            or not isinstance(num_train, (int, np.integer))
        ):
            raise ValueError(
                "num_train must be an integer or None"
            )

        num_train = int(num_train)
        if not 2 <= num_train <= 10:
            raise ValueError(
                "num_train must be between 2 and 10"
            )

    if num_test is not None:
        if (
            isinstance(num_test, (bool, np.bool_))
            or not isinstance(num_test, (int, np.integer))
        ):
            raise ValueError(
                "num_test must be an integer or None"
            )

        num_test = int(num_test)
        if not 1 <= num_test <= 4:
            raise ValueError(
                "num_test must be between 1 and 4"
            )

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError(
                "num_train must be greater than num_test"
            )
        return num_train, num_test

    if num_train is None and num_test is None:
        chosen_test = int(rng.integers(1, 5))
        chosen_train = int(
            rng.integers(chosen_test + 1, 11)
        )
        return chosen_train, chosen_test

    if num_train is None:
        chosen_train = int(
            rng.integers(max(2, num_test + 1), 11)
        )
        return chosen_train, num_test

    max_test = min(4, num_train - 1)
    chosen_test = int(rng.integers(1, max_test + 1))
    return num_train, chosen_test


# A shape is compatible with k precisely when h+w-2 is a multiple of k.
# Keeping the full catalog makes dimensions, k, and repetition count composable.
_CANDIDATES = tuple(
    (height, width, k)
    for height in range(3, 31)
    for width in range(3, 31)
    for k in range(1, (height + width - 2) // 2 + 1)
    if (height + width - 2) % k == 0
)


def _shape_group(spec: tuple[int, int, int], group: int) -> bool:
    height, width, _ = spec
    return (
        (group == 0 and height == width)
        or (group == 1 and height <= 6 and width >= 12)
        or (group == 2 and width <= 6 and height >= 12)
        or (group == 3 and width > height + 2)
        or (group == 4 and height > width + 2)
        or (group == 5 and height >= 20 and width >= 20)
        or (group == 6 and height <= 10 and width <= 10)
        or group == 7
    )


def _length_group(k: int, group: int) -> bool:
    bounds = ((1, 1), (2, 3), (4, 5), (6, 9), (10, 15), (16, 29))
    low, high = bounds[group]
    return low <= k <= high


def _sample_spec(rng: np.random.Generator, predicate=None) -> tuple[int, int, int]:
    eligible = [spec for spec in _CANDIDATES if predicate is None or predicate(spec)]
    if not eligible:
        raise ValueError("no compatible canvas and run length")
    length_group = int(rng.integers(6))
    shape_group = int(rng.integers(8))
    preferred = [
        spec for spec in eligible
        if _length_group(spec[2], length_group) and _shape_group(spec, shape_group)
    ]
    if not preferred:
        preferred = [spec for spec in eligible if _length_group(spec[2], length_group)]
    if not preferred:
        preferred = [spec for spec in eligible if _shape_group(spec, shape_group)]
    if not preferred:
        preferred = eligible
    return preferred[int(rng.integers(len(preferred)))]


def _starts_for(spec: tuple[int, int, int], mode: str) -> list[int]:
    height, width, k = spec
    length = 2 * (height + width - 2)
    starts = []
    for start in range(length):
        kind, corners = _seed_geometry(height, width, k, start)
        if (
            mode == "any"
            or (mode == "bend" and kind == "bend")
            or (mode == "straight" and kind in ("top", "right", "bottom", "left"))
            or (mode == "nonbottom" and kind != "bottom")
            or (mode == "upper_bend" and kind == "bend" and corners[0] in ("top_left", "top_right"))
            or (mode == "upper_straight" and kind in ("top", "right", "left"))
            or (mode == "bottom" and kind == "bottom")
        ):
            starts.append(start)
    return starts


def _choose_pair(
    rng: np.random.Generator,
    spec: tuple[int, int, int],
    start_mode: str,
    background: int,
    accent: int,
    used_geometry: set[tuple[int, int, int, int]],
) -> dict | None:
    starts = _starts_for(spec, start_mode)
    if not starts:
        return None
    for index in rng.permutation(len(starts)):
        start = starts[int(index)]
        geometry = (*spec, start)
        if geometry in used_geometry:
            continue
        input_grid, output_grid = _make_pair(spec[:2], spec[2], start, background, accent)
        if _is_source_pair(input_grid, output_grid):
            continue
        used_geometry.add(geometry)
        return {"input": input_grid, "output": output_grid}
    return None


def _different_color(rng: np.random.Generator, excluded: set[int]) -> int:
    choices = [color for color in range(10) if color not in excluded]
    return choices[int(rng.integers(len(choices)))]


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)

    # The only shared convention is the perimeter completion rule. Colors,
    # canvas, run length, and cyclic phase are visible pair-level choices.
    regime = ("source_like", "reverse", "shared", "mixed")[int(rng.integers(4))]
    train_wide = bool(rng.integers(2))
    def train_orientation(spec):
        return regime != "reverse" or ((spec[1] > spec[0]) == train_wide)
    def test_orientation(spec):
        return regime != "reverse" or ((spec[1] > spec[0]) != train_wide)

    anchor = _sample_spec(
        rng,
        lambda spec: 3 <= spec[2] <= 15
        and spec[2] <= max(spec[:2])
        and train_orientation(spec),
    )
    train_specs = [anchor]
    # A second demonstration has a different k and a straight seed.
    same_canvas = bool(rng.integers(2))
    second = _sample_spec(
        rng,
        lambda spec: spec[2] != anchor[2]
        and spec[2] <= max(spec[:2])
        and train_orientation(spec)
        and (not same_canvas or spec[:2] == anchor[:2]),
    )
    train_specs.append(second)
    if train_count >= 3:
        third = _sample_spec(
            rng,
            lambda spec: spec[2] == anchor[2]
            and spec[0] + spec[1] != anchor[0] + anchor[1]
            and train_orientation(spec),
        )
        train_specs.append(third)
    while len(train_specs) < train_count:
        train_specs.append(_sample_spec(rng, train_orientation))

    bg0 = int(rng.integers(10))
    accent0 = _different_color(rng, {bg0})
    train_colors = [(bg0, accent0)]
    if train_count == 2:
        accent1 = _different_color(rng, {accent0})
        bg1 = accent0 if accent1 == bg0 and bool(rng.integers(2)) else _different_color(rng, {accent1})
    else:
        accent1 = accent0
        bg1 = _different_color(rng, {accent1})
    train_colors.append((bg1, accent1))
    if train_count >= 3:
        if rng.random() < 0.4:
            bg2, accent2 = accent0, bg0
        else:
            accent2 = _different_color(rng, {accent0})
            bg2 = _different_color(rng, {accent2, bg0})
        train_colors.append((bg2, accent2))
    while len(train_colors) < train_count:
        accent = _different_color(rng, set())
        background = _different_color(rng, {accent})
        train_colors.append((background, accent))

    puzzle = {"train": [], "test": []}
    used_geometry: set[tuple[int, int, int, int]] = set()
    for index, (spec, (background, accent)) in enumerate(zip(train_specs, train_colors)):
        mode = (
            "upper_bend" if regime == "source_like" else "bend"
        ) if index == 0 else (
            "upper_straight" if regime == "source_like" else "straight"
        ) if index == 1 else "nonbottom" if regime == "source_like" else "any"
        pair = _choose_pair(rng, spec, mode, background, accent, used_geometry)
        if pair is None:
            raise ValueError("could not place a unique training seed")
        puzzle["train"].append(pair)

    train_lengths = {spec[2] for spec in train_specs}
    train_accents = {accent for _, accent in train_colors}
    for _ in range(test_count):
        if regime == "source_like":
            predicate = lambda spec: spec[2] not in train_lengths and spec[2] <= spec[1]
            mode = "bottom"
        elif regime == "shared" and rng.random() < 0.7:
            predicate = lambda spec: spec[2] in train_lengths
            mode = "any"
        else:
            predicate = test_orientation
            mode = "any"
        spec = _sample_spec(rng, predicate)
        if regime == "source_like" or rng.random() < 0.5:
            accent = _different_color(rng, train_accents)
        else:
            familiar = sorted(train_accents)
            accent = familiar[int(rng.integers(len(familiar)))]
        if rng.random() < 0.3 and bg0 != accent:
            background = bg0
        else:
            background = _different_color(rng, {accent})
        pair = _choose_pair(rng, spec, mode, background, accent, used_geometry)
        if pair is None:
            # A fixed specification can lose its last free phase in a large
            # requested puzzle; resample a compatible one deterministically.
            for _ in range(100):
                spec = _sample_spec(rng, predicate)
                pair = _choose_pair(rng, spec, mode, background, accent, used_geometry)
                if pair is not None:
                    break
        if pair is None:
            raise ValueError("could not place a unique test seed")
        puzzle["test"].append(pair)

    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair: dict) -> dict:
    """Reconstruct the answer from the visible input, independently of sampling."""
    input_grid, output_grid = pair["input"], pair["output"]
    height, width = input_grid.shape
    assert height >= 3 and width >= 3
    assert output_grid.shape == input_grid.shape
    colors, counts = np.unique(input_grid, return_counts=True)
    assert len(colors) == 2
    background = int(colors[int(np.argmax(counts))])
    accent = int(colors[int(np.argmin(counts))])
    assert int(counts.max()) > int(counts.min())
    assert set(int(value) for value in np.unique(output_grid)) == {background, accent}
    assert np.all(input_grid[1:-1, 1:-1] == background)
    assert np.array_equal(output_grid[1:-1, 1:-1], input_grid[1:-1, 1:-1])
    perimeter = _perimeter_coords(height, width)
    mask = np.array([input_grid[r, c] == accent for r, c in perimeter], dtype=bool)
    k = int(mask.sum())
    assert k == int(np.count_nonzero(input_grid == accent))
    assert 1 <= k and len(perimeter) % (2 * k) == 0
    repetitions = len(perimeter) // (2 * k)
    assert repetitions >= 2
    starts = np.flatnonzero(mask & ~np.roll(mask, 1))
    assert len(starts) == 1
    start = int(starts[0])
    assert all(mask[(start + offset) % len(perimeter)] for offset in range(k))
    expected = np.full(input_grid.shape, background, dtype=input_grid.dtype)
    for offset in range(len(perimeter)):
        if offset % (2 * k) < k:
            expected[perimeter[(start + offset) % len(perimeter)]] = accent
    assert np.array_equal(output_grid, expected)
    assert not np.array_equal(input_grid, output_grid)
    kind, corners = _seed_geometry(height, width, k, start)
    return {
        "shape": (height, width), "k": k, "repetitions": repetitions,
        "start": start, "kind": kind, "corners": corners,
        "background": background, "accent": accent,
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs, geometries = set(), set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            input_key = _grid_signature(pair["input"])
            pair_key = _pair_signature(pair["input"], pair["output"])
            assert input_key not in inputs and pair_key not in pairs
            assert not _is_source_pair(pair["input"], pair["output"])
            inputs.add(input_key)
            pairs.add(pair_key)
            info = _analyze_pair(pair)
            geometry = (info["shape"], info["k"], info["start"])
            assert geometry not in geometries
            geometries.add(geometry)
            analyses[split].append(info)
    train = analyses["train"]
    assert train[0]["kind"] == "bend"
    assert train[1]["kind"] in {"top", "right", "bottom", "left"}
    assert train[0]["k"] != train[1]["k"]
    if len(train) == 2:
        assert train[0]["accent"] != train[1]["accent"]
    if len(train) >= 3:
        assert train[0]["accent"] == train[1]["accent"]
        assert train[0]["k"] == train[2]["k"]
        assert train[0]["accent"] != train[2]["accent"]
        assert train[0]["shape"] != train[2]["shape"]
        assert train[0]["shape"][0] + train[0]["shape"][1] != train[2]["shape"][0] + train[2]["shape"][1]
        assert train[0]["background"] != train[2]["background"]
    # No palette, orientation, or size is a hidden puzzle convention. The
    # demonstrations establish one cyclic border rule for both splits.


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple(
        tuple(_pair_signature(pair["input"], pair["output"]) for pair in puzzle[split])
        for split in ("train", "test")
    )


def self_test() -> None:
    # The same square supports three very different block/repetition pairs.
    for k, repetitions in ((1, 14), (2, 7), (7, 2)):
        input_grid, output_grid = _make_pair((8, 8), k, 3, 8, 4)
        info = _analyze_pair({"input": input_grid, "output": output_grid})
        assert info["k"] == k and info["repetitions"] == repetitions
    input_grid, output_grid = _make_pair((3, 13), 7, 10, 8, 4)
    info = _analyze_pair({"input": input_grid, "output": output_grid})
    assert info["kind"] == "multi_corner"
    assert info["corners"] == ("top_right", "bottom_right")

    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(puzzle["train"]) for puzzle in puzzles}) >= 4
    assert len({len(puzzle["test"]) for puzzle in puzzles}) >= 3
    assert len({_puzzle_signature(puzzle) for puzzle in puzzles}) >= 40
    all_info = []
    train_info = []
    test_info = []
    coverage = {
        "same_canvas_diff_k": 0, "same_k_diff_perimeter": 0,
        "role_swap": 0, "unseen_test_k": 0, "shared_test_k": 0,
        "novel_test_accent": 0, "familiar_test_accent": 0,
        "source_like_shift": 0, "wide_to_tall": 0, "tall_to_wide": 0,
    }
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        info = {split: [_analyze_pair(pair) for pair in puzzle[split]] for split in ("train", "test")}
        all_info.extend(info["train"] + info["test"])
        train_info.extend(info["train"])
        test_info.extend(info["test"])
        train, test = info["train"], info["test"]
        train_lengths = {item["k"] for item in train}
        train_accents = {item["accent"] for item in train}
        coverage["same_canvas_diff_k"] += any(
            a["shape"] == b["shape"] and a["k"] != b["k"] for a in train for b in train
        )
        coverage["same_k_diff_perimeter"] += any(
            a["k"] == b["k"] and sum(a["shape"]) != sum(b["shape"])
            for a in train for b in train
        )
        coverage["role_swap"] += any(
            a["background"] == b["accent"] and a["accent"] == b["background"]
            for a in train for b in train
        )
        coverage["unseen_test_k"] += any(item["k"] not in train_lengths for item in test)
        coverage["shared_test_k"] += any(item["k"] in train_lengths for item in test)
        coverage["novel_test_accent"] += any(item["accent"] not in train_accents for item in test)
        coverage["familiar_test_accent"] += any(item["accent"] in train_accents for item in test)
        coverage["source_like_shift"] += (
            all(item["kind"] != "bottom" for item in train)
            and all(item["kind"] == "bottom" and item["k"] not in train_lengths for item in test)
        )
        coverage["wide_to_tall"] += (
            any(a["shape"][1] > a["shape"][0] for a in train)
            and any(b["shape"][0] > b["shape"][1] for b in test)
        )
        coverage["tall_to_wide"] += (
            any(a["shape"][0] > a["shape"][1] for a in train)
            and any(b["shape"][1] > b["shape"][0] for b in test)
        )
    assert coverage["same_canvas_diff_k"] >= 5
    assert coverage["same_k_diff_perimeter"] >= 30
    assert coverage["role_swap"] >= 5
    assert coverage["source_like_shift"] >= 5
    assert all(coverage[key] >= 10 for key in (
        "unseen_test_k", "shared_test_k", "novel_test_accent",
        "familiar_test_accent", "wide_to_tall", "tall_to_wide",
    ))
    assert len({x["shape"] for x in all_info}) >= 80
    assert len({(x["shape"], x["k"], x["start"]) for x in all_info}) >= 200
    assert {x["k"] for x in train_info} >= {1, 2, 3, 4, 5}
    assert max(x["k"] for x in test_info) > 10
    assert {x["kind"] for x in train_info} >= {"top", "right", "bottom", "left", "bend", "multi_corner"}
    assert {corner for x in all_info for corner in x["corners"]} == {"top_left", "top_right", "bottom_right", "bottom_left"}
    assert len({x["repetitions"] for x in all_info}) >= 10
    assert len({x["background"] for x in all_info}) == 10
    assert len({x["accent"] for x in all_info}) == 10
    assert any(x["shape"][0] <= 5 and x["shape"][1] >= 12 for x in all_info)
    assert any(x["shape"][1] <= 5 and x["shape"][0] >= 12 for x in all_info)
    assert any(x["shape"][0] == x["shape"][1] for x in all_info)
    for seed, counts in ((200, (2, 1)), (201, (10, 1)), (202, (10, 4)),
                         (300, (None, 1)), (301, (None, 4)),
                         (302, (2, None)), (303, (10, None))):
        puzzle = generate_puzzle(seed, *counts)
        validate_generated_puzzle(puzzle)
        for actual, requested in zip((len(puzzle["train"]), len(puzzle["test"])), counts):
            assert requested is None or actual == requested
    first = generate_puzzle(123456, 7, 3)
    second = generate_puzzle(123456, 7, 3)
    assert _puzzle_signature(first) == _puzzle_signature(second)
    assert _puzzle_signature(generate_puzzle(1001, 6, 2)) != _puzzle_signature(generate_puzzle(1002, 6, 2))
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": 4, "num_test": 4}):
        try:
            generate_puzzle(400, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count request accepted: {kwargs}")
