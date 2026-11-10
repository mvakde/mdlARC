import hashlib

import numpy as np


# (cell pitch, square token side). A background cell separates a token
# from every full-length divider, even when the token uses divider color.
_GEOMETRIES = ((5, 2), (6, 2), (6, 3), (7, 2), (7, 3))
_SOURCE_PAIRS = frozenset({
    "1c4fb968ef265edd8f757fed4289121c6bb6aa95d2621439a8bf38e6ddec19c4",
    "5edb9c03ba9f713c32077d543e6c9687ecb3c402c729d18d96a77a31491728c7",
    "348ea2e5d9589e915068325a1d0f6c2aa52360def5b0363cc471d6839e9dfb36",
    "0dde38b978b70bde3c29ac42de019a284dd4cbb4539cac3648e539d837c02015",
})


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, seed, num_train, num_test):
    train = _checked_count("num_train", num_train, 2, 10)
    test = _checked_count("num_test", num_test, 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if train is None and test is None:
        test = 1 + seed % 4
        minimum = max(2, test + 1)
        return minimum + (seed // 4) % (11 - minimum), test
    if train is None:
        return int(rng.integers(max(2, test + 1), 11)), test
    return train, int(rng.integers(1, min(4, train - 1) + 1))


def _max_cells(pitch, side):
    return 1 + (30 - 1 - side) // pitch


def _count_options(rows, cols, relation="any", shortest_above_one=False,
                   gap=False, min_categories=2):
    capacity = rows * cols
    start = 2 if shortest_above_one else 1
    options = []
    for categories in range(min_categories, min(6, capacity - 1, 8) + 1):
        baseline = sum(range(start, start + categories - 1))
        low = start + categories - 1
        if gap:
            low += 1
        if relation == "gt":
            low = max(low, cols + 1)
        high = min(30, capacity - 1 - baseline)
        if relation == "lt":
            high = min(high, cols - 1)
        if low <= high:
            options.append((categories, low, high))
    return options


def _sample_frequencies(rng, rows, cols, relation="any",
                        shortest_above_one=False, gap=False, min_categories=2):
    options = _count_options(
        rows, cols, relation, shortest_above_one, gap, min_categories
    )
    if not options:
        raise ValueError("infeasible count regime")
    categories, low, high = options[int(rng.integers(len(options)))]
    maximum = int(rng.integers(low, high + 1))
    start = 2 if shortest_above_one else 1
    baseline = list(range(start, start + categories - 1))
    chosen = baseline
    # Random feasible lower bars introduce independent frequency and null
    # variation. The baseline is a guaranteed, deterministic fallback.
    for _ in range(40):
        candidate = sorted(int(x) for x in rng.choice(
            np.arange(start, maximum), size=categories - 1, replace=False
        ))
        if sum(candidate) + maximum > rows * cols - 1:
            continue
        if gap and maximum - candidate[0] + 1 == categories:
            continue
        chosen = candidate
        break
    return chosen + [maximum]


def _shapes_for(pitch, side, relation="any", shortest_above_one=False,
                gap=False, min_categories=2):
    limit = _max_cells(pitch, side)
    return [
        (rows, cols)
        for rows in range(2, limit + 1)
        for cols in range(2, limit + 1)
        if _count_options(
            rows, cols, relation, shortest_above_one, gap, min_categories
        )
    ]


def _render_input(rng, rows, cols, pitch, side, background, divider, counts):
    minimum_height = (rows - 1) * pitch + 1 + side
    minimum_width = (cols - 1) * pitch + 1 + side
    height = minimum_height + int(rng.integers(
        0, min(pitch - 2, 30 - minimum_height) + 1
    ))
    width = minimum_width + int(rng.integers(
        0, min(pitch - 2, 30 - minimum_width) + 1
    ))
    grid = np.full((height, width), background, dtype=np.int8)
    grid[pitch - 1::pitch, :] = divider
    grid[:, pitch - 1::pitch] = divider
    tokens = [color for color, count in counts.items() for _ in range(count)]
    tokens.extend([divider] * (rows * cols - len(tokens)))
    rng.shuffle(tokens)
    for index, color in enumerate(tokens):
        row, col = divmod(index, cols)
        r0, c0 = row * pitch + 1, col * pitch + 1
        grid[r0:r0 + side, c0:c0 + side] = color
    return grid


def _histogram(background, counts):
    ordered = sorted(counts, key=counts.get)
    output = np.full((len(ordered), max(counts.values())), background, dtype=np.int8)
    for row, color in enumerate(ordered):
        output[row, :counts[color]] = color
    return output


def _grid_key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _pair_fingerprint(input_grid, output_grid):
    digest = hashlib.sha256()
    for grid in (input_grid, output_grid):
        digest.update(bytes(grid.shape))
        digest.update(grid.astype(np.int8, copy=False).tobytes())
    return digest.hexdigest()


def _make_pair(rng, shape, geometry, background, divider, forced_payload,
               relation="any", shortest_above_one=False, gap=False,
               min_categories=2):
    rows, cols = shape
    pitch, side = geometry
    frequencies = _sample_frequencies(
        rng, rows, cols, relation, shortest_above_one, gap, min_categories
    )
    choices = [c for c in range(10) if c not in (background, divider)]
    if forced_payload is not None:
        assert forced_payload in choices
        choices.remove(forced_payload)
        colors = [forced_payload]
        remaining = rng.choice(choices, size=len(frequencies) - 1, replace=False)
        colors.extend(int(c) for c in remaining)
    else:
        colors = [int(c) for c in rng.choice(
            choices, size=len(frequencies), replace=False
        )]
    rng.shuffle(colors)
    rng.shuffle(frequencies)
    counts = dict(zip(colors, frequencies))
    return {
        "input": _render_input(
            rng, rows, cols, pitch, side, background, divider, counts
        ),
        "output": _histogram(background, counts),
    }


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, int(seed), num_train, num_test)

    # Literal colors are pair roles. The two required demonstrations exchange
    # divider and payload roles and use different backgrounds.
    backgrounds = [int(c) for c in rng.choice(10, size=2, replace=False)]
    remaining = [c for c in range(10) if c not in backgrounds]
    dividers = [int(c) for c in rng.choice(remaining, size=2, replace=False)]
    small_token_geometry = _GEOMETRIES[int(rng.choice([0, 1, 3]))]
    large_token_geometry = _GEOMETRIES[int(rng.choice([2, 4]))]
    anchors = [small_token_geometry, large_token_geometry]
    rng.shuffle(anchors)

    puzzle = {"train": [], "test": []}
    used_inputs = set()

    def append_unique(destination, shape, geometry, background, divider,
                      forced_payload=None, relation="any",
                      shortest_above_one=False, gap=False, min_categories=2):
        def add_if_new(pair):
            key = _grid_key(pair["input"])
            if key in used_inputs or _pair_fingerprint(
                pair["input"], pair["output"]
            ) in _SOURCE_PAIRS:
                return False
            used_inputs.add(key)
            destination.append(pair)
            return True

        for _ in range(32):
            pair = _make_pair(
                rng, shape, geometry, background, divider, forced_payload,
                relation, shortest_above_one, gap, min_categories
            )
            if add_if_new(pair):
                return
        # If token shuffles collide, change the coarse shape in a fixed order
        # while keeping the same visible color roles and frequency regime.
        for fallback_shape in _shapes_for(
            *geometry, relation, shortest_above_one, gap, min_categories
        ):
            if fallback_shape == shape:
                continue
            pair = _make_pair(
                rng, fallback_shape, geometry, background, divider,
                forced_payload, relation, shortest_above_one, gap,
                min_categories
            )
            if add_if_new(pair):
                return
        raise RuntimeError("could not construct a unique pair")

    for index in range(2):
        geometry = anchors[index]
        relation = "lt" if index == 0 else "gt"
        short = index == 0
        gap = index == 1
        shapes = _shapes_for(*geometry, relation, short, gap)
        shape = shapes[int(rng.integers(len(shapes)))]
        append_unique(
            puzzle["train"], shape, geometry, backgrounds[index],
            dividers[index], dividers[1 - index], relation, short, gap
        )

    for _ in range(2, train_count):
        geometry = _GEOMETRIES[int(rng.integers(len(_GEOMETRIES)))]
        shapes = _shapes_for(*geometry)
        shape = shapes[int(rng.integers(len(shapes)))]
        background = int(rng.integers(10))
        divider = int(rng.choice([c for c in range(10) if c != background]))
        append_unique(puzzle["train"], shape, geometry, background, divider)

    trained_geometries = sorted({
        _decode_pair(p["input"])["geometry"] for p in puzzle["train"]
    })
    train_sizes = [_decode_pair(p["input"])["shape"] for p in puzzle["train"]]
    max_rows = max(shape[0] for shape in train_sizes)
    max_cols = max(shape[1] for shape in train_sizes)
    for _ in range(test_count):
        geometry = trained_geometries[int(rng.integers(len(trained_geometries)))]
        category_floor = int(rng.choice([2, 3, 4, 5], p=[0.3, 0.3, 0.3, 0.1]))
        shapes = _shapes_for(*geometry, min_categories=category_floor)
        larger = [
            shape for shape in shapes
            if shape[0] > max_rows or shape[1] > max_cols
        ]
        if larger and rng.integers(2):
            shapes = larger
        shape = shapes[int(rng.integers(len(shapes)))]
        background = int(rng.integers(10))
        if rng.integers(2):
            available = [d for d in dividers if d != background]
            divider = int(rng.choice(available))
        else:
            divider = int(rng.choice([
                c for c in range(10) if c != background and c not in dividers
            ]))
        append_unique(
            puzzle["test"], shape, geometry, background, divider,
            min_categories=category_floor
        )

    validate_generated_puzzle(puzzle)
    return puzzle


def _decode_pair(grid):
    height, width = grid.shape
    background = int(grid[0, 0])
    full_rows = [
        r for r in range(1, height)
        if int(grid[r, 0]) != background and np.all(grid[r, :] == grid[r, 0])
    ]
    full_cols = [
        c for c in range(1, width)
        if int(grid[0, c]) != background and np.all(grid[:, c] == grid[0, c])
    ]
    assert full_rows and full_cols
    pitch = full_rows[0] + 1
    assert full_cols[0] + 1 == pitch
    assert pitch in (5, 6, 7)
    divider = int(grid[full_rows[0], 0])
    assert divider != background
    assert full_rows == list(range(pitch - 1, height, pitch))
    assert full_cols == list(range(pitch - 1, width, pitch))
    assert all(int(grid[r, 0]) == divider for r in full_rows)
    assert all(int(grid[0, c]) == divider for c in full_cols)

    first = int(grid[1, 1])
    assert first != background
    side = 1
    while 1 + side < pitch - 1 and int(grid[1, 1 + side]) == first:
        side += 1
    assert side in (2, 3)
    assert (pitch, side) in _GEOMETRIES
    rows = 1 + (height - 1 - side) // pitch
    cols = 1 + (width - 1 - side) // pitch
    assert rows >= 2 and cols >= 2
    minimum_height = (rows - 1) * pitch + 1 + side
    minimum_width = (cols - 1) * pitch + 1 + side
    assert 0 <= height - minimum_height <= pitch - 2
    assert 0 <= width - minimum_width <= pitch - 2

    reconstructed = np.full(grid.shape, background, dtype=np.int8)
    reconstructed[pitch - 1::pitch, :] = divider
    reconstructed[:, pitch - 1::pitch] = divider
    tokens = np.empty((rows, cols), dtype=np.int8)
    for row in range(rows):
        for col in range(cols):
            r0, c0 = row * pitch + 1, col * pitch + 1
            block = grid[r0:r0 + side, c0:c0 + side]
            assert block.shape == (side, side)
            color = int(block[0, 0])
            assert color != background and np.all(block == color)
            tokens[row, col] = color
            reconstructed[r0:r0 + side, c0:c0 + side] = color
    assert np.array_equal(grid, reconstructed)
    counts = {}
    for value in tokens.flat:
        color = int(value)
        if color != divider:
            counts[color] = counts.get(color, 0) + 1
    assert 2 <= len(counts) <= 6
    assert len(set(counts.values())) == len(counts)
    null_count = int(np.count_nonzero(tokens == divider))
    assert null_count >= 1
    return {
        "background": background, "divider": divider,
        "geometry": (pitch, side), "shape": (rows, cols),
        "tokens": tokens, "counts": counts, "nulls": null_count,
        "tail": (height - minimum_height, width - minimum_width),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    decoded = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            input_key = _grid_key(pair["input"])
            pair_key = (input_key, _grid_key(pair["output"]))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert _pair_fingerprint(
                pair["input"], pair["output"]
            ) not in _SOURCE_PAIRS
            info = _decode_pair(pair["input"])
            assert np.array_equal(
                pair["output"], _histogram(info["background"], info["counts"])
            )
            decoded[split].append(info)

    training = decoded["train"]
    tests = decoded["test"]
    # These are visible demonstrations, not hidden generator metadata.
    assert training[0]["background"] != training[1]["background"]
    assert training[0]["divider"] != training[1]["divider"]
    assert training[0]["divider"] in training[1]["counts"]
    assert training[1]["divider"] in training[0]["counts"]
    assert max(training[0]["counts"].values()) < training[0]["shape"][1]
    assert min(training[0]["counts"].values()) > 1
    assert max(training[1]["counts"].values()) > training[1]["shape"][1]
    second_values = sorted(training[1]["counts"].values())
    assert second_values != list(range(second_values[0], second_values[-1] + 1))
    assert training[0]["geometry"][1] != training[1]["geometry"][1]
    trained_geometries = {item["geometry"] for item in training}
    assert all(item["geometry"] in trained_geometries for item in tests)


def _puzzles_equal(first, second):
    return all(
        np.array_equal(left[field], right[field])
        for split in ("train", "test")
        for left, right in zip(first[split], second[split])
        for field in ("input", "output")
    ) and all(len(first[k]) == len(second[k]) for k in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    fingerprint_set, shapes, physical_shapes, geometries, normalized, backgrounds = (
        set(), set(), set(), set(), set(), set()
    )
    widths_less = widths_more = skipped = raised_min = 0
    test_categories, test_nulls, test_frequencies, train_frequencies = (
        set(), set(), set(), set()
    )
    trained_dividers_used_as_payload = 0
    reused_test_dividers = novel_test_dividers = 0
    new_test_payloads = extended_tests = familiar_size_tests = 0
    near_limit = six_cell_axes = zero_background = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        fingerprint_set.add(tuple(
            _pair_fingerprint(pair["input"], pair["output"])
            for split in ("train", "test") for pair in puzzle[split]
        ))
        train = [_decode_pair(p["input"]) for p in puzzle["train"]]
        train_dividers = {item["divider"] for item in train}
        train_payloads = set().union(*(set(item["counts"]) for item in train))
        max_train_rows = max(item["shape"][0] for item in train)
        max_train_cols = max(item["shape"][1] for item in train)
        if any(
            item["divider"] in other["counts"]
            for item in train for other in train if item is not other
        ):
            trained_dividers_used_as_payload += 1
        for split in ("train", "test"):
            for pair in puzzle[split]:
                item = _decode_pair(pair["input"])
                assert np.array_equal(
                    pair["output"], _histogram(item["background"], item["counts"])
                )
                assert _pair_fingerprint(
                    pair["input"], pair["output"]
                ) not in _SOURCE_PAIRS
                shapes.add(item["shape"])
                physical_shapes.add(pair["input"].shape)
                geometries.add(item["geometry"])
                backgrounds.add(item["background"])
                near_limit += max(pair["input"].shape) >= 28
                six_cell_axes += max(item["shape"]) == 6
                zero_background += item["background"] == 0
                ordered = sorted(item["counts"], key=item["counts"].get)
                ranks = {color: rank + 1 for rank, color in enumerate(ordered)}
                normalized.add((
                    item["shape"], bytes(
                        ranks.get(int(value), 0) for value in item["tokens"].flat
                    )
                ))
                values = sorted(item["counts"].values())
                widths_less += values[-1] < item["shape"][1]
                widths_more += values[-1] > item["shape"][1]
                skipped += values != list(range(values[0], values[-1] + 1))
                raised_min += values[0] > 1
                if split == "test":
                    test_categories.add(len(values))
                    test_nulls.add(item["nulls"])
                    test_frequencies.add(tuple(values))
                    reused_test_dividers += item["divider"] in train_dividers
                    novel_test_dividers += item["divider"] not in train_dividers
                    new_test_payloads += bool(set(item["counts"]) - train_payloads)
                    extends = (
                        item["shape"][0] > max_train_rows
                        or item["shape"][1] > max_train_cols
                    )
                    extended_tests += extends
                    familiar_size_tests += not extends
                else:
                    train_frequencies.add(tuple(values))
    assert len(fingerprint_set) == 42
    assert len(shapes) >= 15 and len(geometries) == len(_GEOMETRIES)
    assert len(physical_shapes) >= 100
    assert len(normalized) >= 100 and len(backgrounds) == 10
    assert widths_less >= 42 and widths_more >= 42
    assert skipped >= 42 and raised_min >= 42
    assert trained_dividers_used_as_payload == 42
    assert len(test_categories) >= 4 and len(test_nulls) >= 8
    assert len(test_frequencies) >= 20 and len(train_frequencies) >= 30
    assert reused_test_dividers > 0 and novel_test_dividers > 0
    assert new_test_payloads > 0
    assert extended_tests > 0 and familiar_size_tests > 0
    assert near_limit > 0 and six_cell_axes > 0 and zero_background > 0
    for seed, train, test in ((100, 2, 1), (101, 10, 1), (102, 10, 4), (103, 6, 3)):
        result = generate_puzzle(seed, train, test)
        assert (len(result["train"]), len(result["test"])) == (train, test)
    for seed, train, test in ((200, None, 1), (201, None, 4),
                              (202, 2, None), (203, 10, None)):
        result = generate_puzzle(seed, train, test)
        assert (train is None or len(result["train"]) == train)
        assert (test is None or len(result["test"]) == test)
    assert _puzzles_equal(generate_puzzle(12345, 7, 3),
                          generate_puzzle(12345, 7, 3))
    assert not _puzzles_equal(generate_puzzle(12345, 7, 3),
                              generate_puzzle(12346, 7, 3))
    for arguments in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0},
        {"num_test": 5}, {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
    ):
        try:
            generate_puzzle(999, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {arguments}")
