"""Solved ARC puzzles built by substituting the input for its foreground cells."""

import numpy as np


_SOURCE_INPUT_DATA = (
    ((3, 3, 0, 3), (3, 0, 0, 3), (0, 0, 0, 3), (3, 3, 0, 3)),
    ((0, 0, 0, 1), (0, 1, 1, 0), (0, 0, 0, 0), (0, 0, 0, 1)),
    ((0, 0, 0, 0), (4, 4, 4, 4), (0, 4, 4, 0), (4, 4, 0, 0)),
    ((1, 0, 1, 0), (1, 1, 0, 0), (1, 1, 1, 1), (1, 0, 0, 1)),
    ((2, 0, 0, 2), (2, 2, 2, 2), (2, 0, 0, 2), (0, 2, 0, 2)),
    ((0, 2, 0, 2), (2, 2, 0, 2), (2, 2, 0, 0), (0, 0, 0, 2)),
)
_SOURCE_INPUTS = tuple(np.array(rows, dtype=np.int8) for rows in _SOURCE_INPUT_DATA)
_SHAPES = tuple(
    (height, width)
    for height in range(2, 6)
    for width in range(2, 6)
    if height >= 3 or width >= 3
)


def _checked_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _checked_count(num_train, "num_train", 2, 10)
    num_test = _checked_count(num_test, "num_test", 1, 4)
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return num_train, num_test
    if num_train is None and num_test is None:
        legal = [(a, b) for a in range(2, 11) for b in range(1, min(4, a - 1) + 1)]
        return legal[int(rng.integers(len(legal)))]
    if num_train is None:
        return int(rng.integers(max(2, num_test + 1), 11)), num_test
    return num_train, int(rng.integers(1, min(4, num_train - 1) + 1))


def _key(grid):
    return grid.shape, np.ascontiguousarray(grid).tobytes()


def _is_source(grid):
    return any(grid.shape == source.shape and np.array_equal(grid, source) for source in _SOURCE_INPUTS)


def _component_count(mask):
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for row in range(height):
        for column in range(width):
            if not mask[row, column] or seen[row, column]:
                continue
            count += 1
            seen[row, column] = True
            stack = [(row, column)]
            while stack:
                r, c = stack.pop()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _orbit(mask):
    if mask.shape[0] == mask.shape[1]:
        images = [image for turns in range(4) for image in (np.rot90(mask, turns), np.fliplr(np.rot90(mask, turns)))]
    else:
        images = [mask, np.flipud(mask), np.fliplr(mask), np.rot90(mask, 2)]
    return len({_key(image) for image in images})


def _grown_mask(rng, height, width, wanted):
    mask = np.zeros((height, width), dtype=bool)
    first = int(rng.integers(height * width))
    mask[first // width, first % width] = True
    while int(mask.sum()) < wanted:
        neighbors = set()
        for row, column in np.argwhere(mask):
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = int(row) + dr, int(column) + dc
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    neighbors.add(nr * width + nc)
        chosen = sorted(neighbors)[int(rng.integers(len(neighbors)))]
        mask[chosen // width, chosen % width] = True
    return mask


def _random_mask(rng, height, width, wanted):
    mask = np.zeros(height * width, dtype=bool)
    mask[rng.choice(height * width, size=wanted, replace=False)] = True
    return mask.reshape(height, width)


def _motif(rng, height, width, kind, wanted=None):
    area = height * width
    mask = np.zeros((height, width), dtype=bool)
    if kind == "random":
        if wanted is None:
            wanted = int(rng.integers(2, area - 1))
        return _random_mask(rng, height, width, wanted)
    if kind == "connected":
        return _grown_mask(rng, height, width, int(rng.integers(2, area - 1)))
    if kind == "sparse":
        mask.flat[int(rng.integers(area))] = True
    elif kind == "dense":
        mask[:] = True
        mask.flat[int(rng.integers(area))] = False
    elif kind == "row":
        mask[int(rng.integers(height)), :] = True
        if rng.random() < 0.35:
            mask.flat[int(rng.integers(area))] = True
    elif kind == "column":
        mask[:, int(rng.integers(width))] = True
        if rng.random() < 0.35:
            mask.flat[int(rng.integers(area))] = True
    elif kind == "diagonal":
        side = min(height, width)
        row_start = int(rng.integers(height - side + 1))
        col_start = int(rng.integers(width - side + 1))
        reverse = bool(rng.integers(2))
        for index in range(side):
            mask[row_start + index, col_start + (side - 1 - index if reverse else index)] = True
        if rng.random() < 0.25:
            mask.flat[int(rng.integers(area))] = True
    elif kind == "checker":
        offset = int(rng.integers(2))
        mask = np.fromfunction(lambda r, c: (r + c + offset) % 2 == 0, (height, width), dtype=int)
    elif kind == "frame":
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
    elif kind == "center":
        inner_height = int(rng.integers(1, height - 1))
        inner_width = int(rng.integers(1, width - 1))
        top = (height - inner_height) // 2
        left = (width - inner_width) // 2
        mask[top:top + inner_height, left:left + inner_width] = True
    elif kind == "islands":
        possibilities = [(a, b) for a in range(area) for b in range(a + 1, area)
                         if abs(a // width - b // width) + abs(a % width - b % width) > 1]
        first, second = possibilities[int(rng.integers(len(possibilities)))]
        mask.flat[first] = mask.flat[second] = True
        if rng.random() < 0.5:
            candidates = [index for index in range(area) if not mask.flat[index]]
            mask.flat[candidates[int(rng.integers(len(candidates)))]] = True
    else:
        raise ValueError("unknown motif kind")
    return mask


def _possible_kinds(height, width, first_training):
    kinds = ["random", "connected", "row", "column", "diagonal", "checker", "islands"]
    if height >= 3 and width >= 3:
        kinds += ["frame", "center"]
    if not first_training:
        kinds += ["sparse", "dense"]
    return kinds


def _make_input(rng, shape, color, used, training_index=None, preferred=None, familiar_count=None):
    height, width = shape
    area = height * width
    kinds = _possible_kinds(height, width, training_index is not None and training_index < 2)
    if preferred in kinds:
        kinds = [preferred] + [kind for kind in kinds if kind != preferred]
    for attempt in range(240):
        kind = kinds[0] if attempt < 24 and preferred in kinds else kinds[int(rng.integers(len(kinds)))]
        wanted = familiar_count if kind == "random" and attempt < 40 else None
        mask = _motif(rng, height, width, kind, wanted)
        count = int(mask.sum())
        if training_index is not None and training_index < 2 and not 2 <= count <= area - 2:
            continue
        grid = mask.astype(np.int8) * np.int8(color)
        key = _key(grid)
        if key not in used and not _is_source(grid):
            used.add(key)
            return grid
    # Finite fallback, even for the smallest shapes and requested pair counts.
    for count in range(2 if training_index is not None and training_index < 2 else 1, area):
        for shift in range(area):
            mask = np.zeros(area, dtype=bool)
            mask[(np.arange(count) * max(1, area // count) + shift) % area] = True
            grid = mask.reshape(shape).astype(np.int8) * np.int8(color)
            key = _key(grid)
            if key not in used and not _is_source(grid):
                used.add(key)
                return grid
    raise ValueError("could not construct a distinct motif")


def _expand(grid):
    return np.kron((grid != 0).astype(np.int8), grid).astype(np.int8)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    total = num_train + num_test

    # Zero is the shared background role. Color and geometry belong to each pair.
    fixed_shape = rng.random() < 0.36
    home_shape = _SHAPES[int(rng.integers(len(_SHAPES)))]
    shapes = [home_shape] if fixed_shape else [_SHAPES[int(rng.integers(len(_SHAPES)))]]
    if not fixed_shape:
        alternatives = [shape for shape in _SHAPES if shape != shapes[0]]
        shapes.append(alternatives[int(rng.integers(len(alternatives)))])
    while len(shapes) < total:
        shapes.append(home_shape if fixed_shape else _SHAPES[int(rng.integers(len(_SHAPES)))])

    fixed_color = rng.random() < 0.30
    palette = [int(color) for color in rng.choice(np.arange(1, 10), size=1 if fixed_color else int(rng.integers(2, 6)), replace=False)]
    colors = [palette[0]]
    if fixed_color:
        colors.extend([palette[0]] * (total - 1))
    else:
        colors.append(palette[1])
        while len(colors) < num_train:
            colors.append(palette[int(rng.integers(len(palette)))])
        for _ in range(num_test):
            unseen = [color for color in range(1, 10) if color not in colors[:num_train]]
            if unseen and rng.random() < 0.38:
                colors.append(unseen[int(rng.integers(len(unseen)))])
            else:
                colors.append(colors[int(rng.integers(num_train))])

    used = set()
    train = []
    test = []
    special_kinds = ["row", "column", "diagonal", "checker", "frame", "center", "islands", "connected", "random"]
    rng.shuffle(special_kinds)
    for index in range(total):
        shape = shapes[index]
        pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
        if index < num_train:
            preferred = special_kinds[index % len(special_kinds)]
            grid = _make_input(pair_rng, shape, colors[index], used, training_index=index, preferred=preferred)
        else:
            test_index = index - num_train
            roll = rng.random()
            if roll < 0.14:
                preferred = "sparse"
            elif roll < 0.28:
                preferred = "dense"
            elif roll < 0.42:
                preferred = "center"
            elif roll < 0.52:
                preferred = "frame"
            else:
                preferred = special_kinds[(index + test_index) % len(special_kinds)]
            familiar = None
            if rng.random() < 0.30:
                counts = [int(np.count_nonzero(pair["input"])) for pair in train]
                possible = [count for count in counts if 1 <= count < shape[0] * shape[1]]
                if possible:
                    familiar = possible[int(rng.integers(len(possible)))]
                    preferred = "random"
            grid = _make_input(pair_rng, shape, colors[index], used, preferred=preferred, familiar_count=familiar)
        pair = {"input": grid, "output": _expand(grid)}
        (train if index < num_train else test).append(pair)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    inputs = set()
    pairs = set()
    for split, collection in (("train", train), ("test", test)):
        for index, pair in enumerate(collection):
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            source, result = pair["input"], pair["output"]
            for grid in (source, result):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= side <= 30 for side in grid.shape)
                assert np.all((grid >= 0) & (grid <= 9))
            height, width = source.shape
            assert 2 <= height <= 5 and 2 <= width <= 5 and (height >= 3 or width >= 3)
            values = np.unique(source)
            assert len(values) == 2 and int(values[0]) == 0
            assert result.shape == (height * height, width * width)
            assert set(int(value) for value in np.unique(result)) == {0, int(values[1])}
            assert not np.array_equal(source, result)
            assert not _is_source(source)
            if split == "train" and index < 2:
                assert 2 <= np.count_nonzero(source) <= height * width - 2
            for row in range(height):
                for column in range(width):
                    block = result[row * height:(row + 1) * height, column * width:(column + 1) * width]
                    expected = source if source[row, column] != 0 else np.zeros_like(source)
                    assert np.array_equal(block, expected)
            input_key = _key(source)
            pair_key = input_key, _key(result)
            assert input_key not in inputs and pair_key not in pairs
            inputs.add(input_key)
            pairs.add(pair_key)


def _predict_from_training(train, test_input):
    # The demonstrations establish zero as blank and literal input substitution.
    for pair in train:
        source, result = pair["input"], pair["output"]
        height, width = source.shape
        if result.shape != (height * height, width * width):
            raise AssertionError("training dimensions do not reveal squaring")
        for row in range(height):
            for column in range(width):
                block = result[row * height:(row + 1) * height, column * width:(column + 1) * width]
                if not np.array_equal(block, source if source[row, column] else np.zeros_like(source)):
                    raise AssertionError("training blocks disagree")
    height, width = test_input.shape
    predicted = np.zeros((height * height, width * width), dtype=np.int8)
    for row in range(height):
        for column in range(width):
            if test_input[row, column] != 0:
                predicted[row * height:(row + 1) * height, column * width:(column + 1) * width] = test_input
    return predicted


def _puzzles_equal(first, second):
    return all(len(first[split]) == len(second[split]) and all(
        np.array_equal(a["input"], b["input"]) and np.array_equal(a["output"], b["output"])
        for a, b in zip(first[split], second[split])) for split in ("train", "test"))


def _review_features(grid):
    mask = grid != 0
    height, width = mask.shape
    area = height * width
    count = int(mask.sum())
    features = set()
    if count == 1:
        features.add("one_foreground")
    if count == area - 1:
        features.add("one_background")
    if _component_count(mask) == 1:
        features.add("connected")
    else:
        features.add("disconnected")
    if _orbit(mask) < (8 if height == width else 4):
        features.add("symmetric")
    if not (mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()):
        features.add("interior")
    if (~mask.any(axis=1)).any() or (~mask.any(axis=0)).any():
        features.add("blank_band")
    if mask.all(axis=1).any():
        features.add("full_row")
    if mask.all(axis=0).any():
        features.add("full_column")
    side = min(height, width)
    if any(all(mask[top + i, left + (side - 1 - i if reverse else i)] for i in range(side))
           for top in range(height - side + 1)
           for left in range(width - side + 1)
           for reverse in (False, True)):
        features.add("diagonal")
    if all(mask[r, c] == (mask[0, 0] if (r + c) % 2 == 0 else not mask[0, 0])
           for r in range(height) for c in range(width)):
        features.add("checker")
    if (height >= 3 and width >= 3 and mask[0].all() and mask[-1].all()
            and mask[:, 0].all() and mask[:, -1].all() and not mask[1:-1, 1:-1].any()):
        features.add("frame")
    if height == 2 or width == 2:
        features.add("two_side")
    return features


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for pair in puzzle["test"]:
            assert np.array_equal(_predict_from_training(puzzle["train"], pair["input"].copy()), pair["output"])
    assert len({len(puzzle["train"]) for puzzle in puzzles}) >= 4
    assert len({len(puzzle["test"]) for puzzle in puzzles}) >= 3
    assert len({tuple(_key(pair["input"]) for pair in puzzle["train"] + puzzle["test"]) for puzzle in puzzles}) == 42
    assert _puzzles_equal(generate_puzzle(90210, 7, 3), generate_puzzle(90210, 7, 3))
    assert _puzzles_equal(generate_puzzle(17), generate_puzzle(17))
    assert not _puzzles_equal(generate_puzzle(90210, 7, 3), generate_puzzle(90211, 7, 3))
    for index, (n_train, n_test) in enumerate(((2, 1), (10, 1), (10, 4), (6, 2), (7, 3))):
        puzzle = generate_puzzle(1000 + index, n_train, n_test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (n_train, n_test)
        validate_generated_puzzle(puzzle)
    for index, (n_train, n_test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(2000 + index, n_train, n_test)
        assert n_train is None or len(puzzle["train"]) == n_train
        assert n_test is None or len(puzzle["test"]) == n_test
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                      {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4},
                      {"num_train": True, "num_test": 1}):
        try:
            generate_puzzle(0, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {arguments}")

    all_pairs = [pair for puzzle in puzzles for split in ("train", "test") for pair in puzzle[split]]
    shapes = {pair["input"].shape for pair in all_pairs}
    masks = {_key(pair["input"] != 0) for pair in all_pairs}
    colors = {int(np.max(pair["input"])) for pair in all_pairs}
    counts = {int(np.count_nonzero(pair["input"])) for pair in all_pairs}
    components = {_component_count(pair["input"] != 0) for pair in all_pairs}
    assert len(shapes) >= 12 and any(2 in shape for shape in shapes)
    assert len(masks) >= len(all_pairs) // 2 and len(colors) == 9 and len(counts) >= 20
    assert 1 in components and any(count >= 2 for count in components)
    assert any(len({pair["input"].shape for pair in puzzle["train"] + puzzle["test"]}) > 1 for puzzle in puzzles)
    assert any(len({int(np.max(pair["input"])) for pair in puzzle["train"] + puzzle["test"]}) > 1 for puzzle in puzzles)
    assert any(any(int(np.max(pair["input"])) not in {int(np.max(p["input"])) for p in puzzle["train"]}
                   for pair in puzzle["test"]) for puzzle in puzzles)
    assert any(np.count_nonzero(pair["input"]) == 1 for pair in all_pairs)
    assert any(np.count_nonzero(pair["input"]) == pair["input"].size - 1 for pair in all_pairs)
    assert any(_orbit(pair["input"] != 0) < (8 if pair["input"].shape[0] == pair["input"].shape[1] else 4)
               for puzzle in puzzles for pair in puzzle["test"])
    assert any(not (mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
               for puzzle in puzzles for pair in puzzle["test"] for mask in [pair["input"] != 0])

    # Review properties are measured from returned grids, not sampled mode names.
    train_flags = [_review_features(pair["input"]) for puzzle in puzzles for pair in puzzle["train"]]
    test_flags = [_review_features(pair["input"]) for puzzle in puzzles for pair in puzzle["test"]]
    assert all(any(name in flags for flags in train_flags) for name in
               ("connected", "disconnected", "full_row", "full_column", "diagonal",
                "checker", "frame", "interior", "blank_band", "one_foreground", "one_background"))
    assert all(any(name in flags for flags in test_flags) for name in
               ("connected", "disconnected", "symmetric", "interior", "full_row",
                "full_column", "diagonal", "checker", "frame", "one_foreground", "one_background"))
    assert any({"interior", "symmetric"} <= flags for flags in test_flags)
    assert any({"two_side", "disconnected"} <= flags for flags in train_flags)
    assert any({"checker", "disconnected"} <= flags for flags in test_flags)
    assert any(len({pair["input"].shape for pair in puzzle["train"] + puzzle["test"]}) == 1
               for puzzle in puzzles)
    assert any(len({pair["input"].shape for pair in puzzle["train"] + puzzle["test"]}) > 1
               for puzzle in puzzles)
    assert any(len({int(np.max(pair["input"])) for pair in puzzle["train"] + puzzle["test"]}) == 1
               for puzzle in puzzles)
    assert any(len({int(np.max(pair["input"])) for pair in puzzle["train"] + puzzle["test"]}) > 1
               for puzzle in puzzles)
    assert any(any(int(np.count_nonzero(pair["input"])) not in
                   {int(np.count_nonzero(train_pair["input"])) for train_pair in puzzle["train"]}
                   for pair in puzzle["test"]) for puzzle in puzzles)
    assert any(any(int(np.count_nonzero(pair["input"])) in
                   {int(np.count_nonzero(train_pair["input"])) for train_pair in puzzle["train"]}
                   for pair in puzzle["test"]) for puzzle in puzzles)
    assert any(_component_count(puzzle["train"][0]["input"] != 0) > 1 for puzzle in puzzles)
