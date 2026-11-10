import numpy as np


_SOURCE_ROW_PAIRS = (
    (
        (
            "00000000000",
            "00000000000",
            "00001110000",
            "00001310000",
            "00001110000",
            "00000000000",
            "00000000000",
            "00000000000",
            "00000000000",
            "00000000000",
            "00000000000",
            "00000000000",
        ),
        (
            "333",
            "313",
            "333",
        ),
    ),
    (
        (
            "0000000000000",
            "0002222000000",
            "0002442000000",
            "0002442000000",
            "0002222000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
        ),
        (
            "4444",
            "4224",
            "4224",
            "4444",
        ),
    ),
    (
        (
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0044444000000",
            "0044444000000",
            "0044644000000",
            "0044444000000",
            "0044444000000",
            "0000000000000",
        ),
        (
            "66666",
            "66666",
            "66466",
            "66666",
            "66666",
        ),
    ),
    (
        (
            "0000000000000",
            "0033333300000",
            "0033333300000",
            "0033883300000",
            "0033883300000",
            "0033333300000",
            "0033333300000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
        ),
        (
            "888888",
            "888888",
            "883388",
            "883388",
            "888888",
            "888888",
        ),
    ),
)


def _grid_from_strings(rows: tuple[str, ...]) -> np.ndarray:
    return np.array(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


def _array_key(grid: np.ndarray) -> tuple:
    normalized = np.asarray(grid, dtype=np.int8)
    return tuple(normalized.shape), normalized.tobytes(order="C")


def _pair_key(input_grid: np.ndarray, output_grid: np.ndarray) -> tuple:
    return _array_key(input_grid), _array_key(output_grid)


def _source_pair_keys() -> set:
    keys = set()
    for input_rows, output_rows in _SOURCE_ROW_PAIRS:
        input_grid = _grid_from_strings(input_rows)
        output_grid = _grid_from_strings(output_rows)
        keys.add(_pair_key(input_grid, output_grid))
    return keys


def _checked_count(
    value: int | None,
    name: str,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}, inclusive"
        )
    return result


def _resolve_counts(
    seed_value: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _checked_count(num_train, "num_train", 2, 10)
    test_count = _checked_count(num_test, "num_test", 1, 4)

    if train_count is None and test_count is None:
        # This schedule guarantees broad count diversity over consecutive seeds.
        test_count = 1 + seed_value % 4
        base_train_count = 2 + (seed_value // 4) % 9
        train_count = max(base_train_count, test_count + 1)
    elif train_count is None:
        lower = max(2, test_count + 1)
        train_count = int(rng.integers(lower, 11))
    elif test_count is None:
        upper = min(4, train_count - 1)
        if upper < 1:
            raise ValueError("no legal num_test exists for this num_train")
        test_count = int(rng.integers(1, upper + 1))

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    """Choose only distribution regimes; the transformation has no hidden map."""
    source_like = bool(rng.random() < 0.20)
    return {
        "source_like": source_like,
        "stable_background": source_like or bool(rng.random() < 0.32),
        "edge_regime": 0 if source_like else int(rng.choice(3, p=[0.30, 0.40, 0.30])),
        "expanded_core": not source_like and bool(rng.random() < 0.62),
    }


def _make_object(side: int, core_side: int, outer: int, core: int) -> np.ndarray:
    result = np.full((side, side), outer, dtype=np.int8)
    start = (side - core_side) // 2
    result[start:start + core_side, start:start + core_side] = core
    return result


def _sample_train_shapes(rng: np.random.Generator, count: int, params: dict) -> list[tuple[int, int]]:
    if params["source_like"]:
        shapes = [(3, 1), (4, 2)]
        if count >= 3:
            shapes.append((5, 1))
        while len(shapes) < count:
            side = int(rng.integers(3, 7))
            shapes.append((side, 1 if side % 2 else 2))
        if rng.random() < 0.65:
            shapes.sort()
        else:
            rng.shuffle(shapes)
        return shapes

    odd = int(rng.choice([3, 5, 7, 9, 11]))
    even = int(rng.choice([4, 6, 8, 10]))
    shapes = [(odd, 1), (even, 2)]
    if count >= 3 and params["expanded_core"]:
        parity = int(rng.integers(0, 2))
        side = int(rng.choice([7, 9, 11] if parity else [8, 10]))
        shapes.append((side, 3 if parity else 4))
        if count >= 4 and rng.random() < 0.35:
            other = 1 - parity
            side = int(rng.choice([7, 9, 11] if other else [8, 10]))
            shapes.append((side, 3 if other else 4))
    while len(shapes) < count:
        side = int(rng.integers(3, 12))
        expanded = side >= (7 if side % 2 else 8) and rng.random() < 0.20
        core = (3 if side % 2 else 4) if expanded else (1 if side % 2 else 2)
        shapes.append((side, core))
    rng.shuffle(shapes)
    return shapes


def _sample_test_shapes(rng: np.random.Generator, count: int, train_shapes: list[tuple[int, int]], params: dict) -> list[tuple[int, int]]:
    shapes = []
    expanded_train = [(side, core) for side, core in train_shapes if core >= 3]
    for index in range(count):
        if index == 0 and expanded_train and rng.random() < 0.75:
            _, core = expanded_train[int(rng.integers(0, len(expanded_train)))]
            side = int(rng.choice([7, 9, 11] if core == 3 else [8, 10]))
            shapes.append((side, core))
            continue
        if params["source_like"] and rng.random() < 0.70:
            side = 6
        elif params["source_like"]:
            side = int(rng.choice([3, 4, 5, 6, 7, 8]))
        else:
            side = int(rng.integers(3, 12))
        minimal = 1 if side % 2 else 2
        taught = {core for _, core in train_shapes if core % 2 == side % 2 and core <= side - 2}
        choices = [minimal]
        if side >= (7 if side % 2 else 8) and minimal + 2 in taught:
            choices.append(minimal + 2)
        core = choices[-1] if len(choices) == 2 and rng.random() < 0.55 else minimal
        shapes.append((side, core))
    return shapes


def _sample_palettes(rng: np.random.Generator, train_count: int, test_count: int, params: dict) -> list[tuple[int, int, int]]:
    total = train_count + test_count
    stable = params["stable_background"]
    fixed_background = int(rng.integers(0, 10))
    palettes = []
    for index in range(total):
        reuse = index > 1 and rng.random() < 0.22
        if index == train_count:
            reuse = rng.random() < 0.60
        if reuse:
            source_index = int(rng.integers(0, train_count if index >= train_count else index))
            _, old_outer, old_core = palettes[source_index]
            outer, core = old_core, old_outer
        else:
            outer = core = -1
        if stable:
            background = fixed_background
        elif index == 1:
            background = palettes[0][1]
        elif reuse:
            background = int(rng.choice([c for c in range(10) if c not in (outer, core)]))
        else:
            background = int(rng.integers(0, 10))
        if reuse and background in (outer, core):
            background = int(rng.choice([c for c in range(10) if c not in (outer, core)]))
        if not reuse:
            available = [c for c in range(10) if c != background]
            chosen = rng.choice(available, size=2, replace=False)
            outer, core = int(chosen[0]), int(chosen[1])
            if index == 1 and (outer, core) == palettes[0][1:]:
                outer, core = core, outer
        palettes.append((background, outer, core))
    return palettes


def _sample_contact_categories(rng: np.random.Generator, train_count: int, test_count: int, params: dict) -> list[int]:
    regime = params["edge_regime"]
    train = [0] * train_count
    test = [0] * test_count
    if regime:
        train[1] = regime
        for index in range(2, train_count):
            train[index] = regime if rng.random() < 0.30 else 0
        test[0] = regime
        for index in range(1, test_count):
            test[index] = regime if rng.random() < 0.58 else 0
    return train + test


def _sample_canvas(rng: np.random.Generator, side: int, contact: int) -> tuple[int, int, int, int]:
    # The background exceeds half the cells even for the largest object.
    shortest = max(side + 2, int(np.floor(side * np.sqrt(2))) + 2)
    shortest = int(rng.integers(shortest, min(30, side + 16) + 1))
    longest = int(rng.integers(shortest, min(30, shortest + 9) + 1))
    aspect = int(rng.choice(3, p=[0.27, 0.365, 0.365]))
    if aspect == 0:
        height = width = shortest
    elif aspect == 1:
        height, width = longest, shortest
        if height == width and height < 30:
            height += 1
    else:
        height, width = shortest, longest
        if height == width and width < 30:
            width += 1
    vertical_room = height - side
    horizontal_room = width - side
    if contact == 0:
        top = int(rng.integers(1, vertical_room))
        left = int(rng.integers(1, horizontal_room))
    elif contact == 1:
        edge = int(rng.integers(0, 4))
        top = int(rng.integers(1, vertical_room))
        left = int(rng.integers(1, horizontal_room))
        if edge == 0:
            top = 0
        elif edge == 1:
            top = vertical_room
        elif edge == 2:
            left = 0
        else:
            left = horizontal_room
    else:
        top = 0 if rng.random() < 0.5 else vertical_room
        left = 0 if rng.random() < 0.5 else horizontal_room
    return height, width, top, left


def _assert_valid_grid(grid: np.ndarray, label: str) -> None:
    assert isinstance(grid, np.ndarray), f"{label} must be a NumPy array"
    assert grid.ndim == 2, f"{label} must be two-dimensional"
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30, f"{label} dimensions are out of range"
    assert np.issubdtype(grid.dtype, np.integer), f"{label} must have an integer dtype"
    assert int(grid.min()) >= 0 and int(grid.max()) <= 9, f"{label} has an illegal color"


def _inspect_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> dict:
    height, width = input_grid.shape
    counts = np.bincount(input_grid.ravel(), minlength=10)
    background = int(np.argmax(counts))
    assert counts[background] > input_grid.size // 2, "background must be uniquely dominant"
    assert len(np.unique(input_grid)) == 3, "input must contain background and two foreground colors"
    coords = np.argwhere(input_grid != background)
    top, left = coords.min(axis=0).tolist()
    bottom, right = coords.max(axis=0).tolist()
    side = bottom - top + 1
    assert side == right - left + 1 and 3 <= side <= 11, "foreground must be a supported square"
    assert side < height and side < width, "background must surround some part of the object"
    crop = input_grid[top:bottom + 1, left:right + 1]
    assert np.all(crop != background), "square must be solidly occupied"
    outer = int(crop[0, 0])
    core_coords = np.argwhere(crop != outer)
    assert len(core_coords) > 0, "center must have a distinct color"
    core = int(crop[tuple(core_coords[0])])
    assert core != background and len(np.unique(crop)) == 2, "object must have exactly two colors"
    core_top, core_left = core_coords.min(axis=0).tolist()
    core_bottom, core_right = core_coords.max(axis=0).tolist()
    core_side = core_bottom - core_top + 1
    assert core_side == core_right - core_left + 1, "core must be square"
    assert core_side in ((1, 3) if side % 2 else (2, 4)), "core parity or size is invalid"
    assert side - core_side >= 2 and core_top == core_left == (side - core_side) // 2, "core must be centered inside a ring"
    expected_object = _make_object(side, core_side, outer, core)
    assert np.array_equal(crop, expected_object), "input geometry is invalid"
    assert side * side - core_side * core_side > core_side * core_side, "outer role must dominate"
    expected_output = np.where(crop == outer, core, outer)
    assert output_grid.shape == (side, side) and np.array_equal(output_grid, expected_output), "output must be the cropped, globally swapped square"
    margins = (top, height - bottom - 1, left, width - right - 1)
    contact = sum(margin == 0 for margin in margins)
    assert contact <= 2 and (margins[0] or margins[1]) and (margins[2] or margins[3]), "background placement is ambiguous"
    return {
        "background": background, "outer": outer, "core": core,
        "foreground_set": tuple(sorted((outer, core))), "side": side,
        "core_side": core_side, "shape": (height, width),
        "margins": margins, "contact": contact,
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle must have train and test"
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists"
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count, "invalid pair counts"
    seen_inputs, seen_pairs = set(), set()
    source_keys = _source_pair_keys()
    details = {"train": [], "test": []}
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema"
            inp, out = pair["input"], pair["output"]
            _assert_valid_grid(inp, f"{split}[{index}].input")
            _assert_valid_grid(out, f"{split}[{index}].output")
            input_key, pair_key = _array_key(inp), _pair_key(inp, out)
            assert input_key not in seen_inputs and pair_key not in seen_pairs, "duplicate generated pair"
            assert pair_key not in source_keys, "exact source pair copied"
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            details[split].append(_inspect_pair(inp, out))
    train = details["train"]
    test = details["test"]
    assert {d["side"] % 2 for d in train} == {0, 1}, "training must show both core parities"
    assert len({(d["background"], d["outer"], d["core"]) for d in train}) >= 2, "training needs color-role variation"
    train_cores = {(d["side"] % 2, d["core_side"]) for d in train}
    train_contacts = {d["contact"] for d in train}
    for detail in test:
        assert (detail["side"] % 2, detail["core_side"]) in train_cores, "test core regime must be taught"
        assert detail["contact"] in train_contacts, "test edge regime must be demonstrated"


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _resolve_counts(int(seed), rng, num_train, num_test)
    params = sample_puzzle_parameters(rng)
    train_shapes = _sample_train_shapes(rng, train_count, params)
    test_shapes = _sample_test_shapes(rng, test_count, train_shapes, params)
    palettes = _sample_palettes(rng, train_count, test_count, params)
    contacts = _sample_contact_categories(rng, train_count, test_count, params)
    child_seeds = rng.integers(0, 2**63, size=train_count + test_count, dtype=np.int64)
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    source_keys = _source_pair_keys()
    for index, (side, core_side) in enumerate(train_shapes + test_shapes):
        pair_rng = np.random.default_rng(int(child_seeds[index]))
        background, outer, core = palettes[index]
        object_grid = _make_object(side, core_side, outer, core)
        output = np.where(object_grid == outer, core, outer).astype(np.int8)
        for _ in range(30):
            height, width, top, left = _sample_canvas(pair_rng, side, contacts[index])
            inp = np.full((height, width), background, dtype=np.int8)
            inp[top:top + side, left:left + side] = object_grid
            if _array_key(inp) not in seen_inputs and _pair_key(inp, output) not in source_keys:
                break
        else:
            raise RuntimeError("could not construct a unique non-source pair")
        seen_inputs.add(_array_key(inp))
        split = "train" if index < train_count else "test"
        puzzle[split].append({"input": inp, "output": output.copy()})
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(a: dict, b: dict) -> bool:
    return all(
        len(a[split]) == len(b[split]) and all(
            np.array_equal(x["input"], y["input"]) and np.array_equal(x["output"], y["output"])
            for x, y in zip(a[split], b[split])
        )
        for split in ("train", "test")
    )


def _expect_value_error(call) -> None:
    try:
        call()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def self_test() -> None:
    assert _puzzles_equal(generate_puzzle(12345), generate_puzzle(12345))
    assert _puzzles_equal(generate_puzzle(54321, 7, 3), generate_puzzle(54321, 7, 3))
    train_counts, test_counts, signatures = set(), set(), set()
    profiles, positions, roles, backgrounds = set(), set(), set(), set()
    mixed_backgrounds = reversed_roles = expanded_tests = 0
    smaller_tests = odd_tests = square_tests = landscape_tests = corner_tests = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(tuple(_pair_key(p["input"], p["output"]) for p in puzzle["train"] + puzzle["test"]))
        train_details = [_inspect_pair(p["input"], p["output"]) for p in puzzle["train"]]
        test_details = [_inspect_pair(p["input"], p["output"]) for p in puzzle["test"]]
        all_details = train_details + test_details
        profiles.add(tuple((d["side"], d["core_side"]) for d in all_details))
        backgrounds.update(d["background"] for d in all_details)
        positions.update((d["shape"], d["margins"]) for d in all_details)
        roles.update((d["background"], d["outer"], d["core"]) for d in all_details)
        mixed_backgrounds += len({d["background"] for d in all_details}) > 1
        reversed_roles += any((t["outer"], t["core"]) == (d["core"], d["outer"]) for t in test_details for d in train_details)
        expanded_tests += any(d["core_side"] >= 3 for d in test_details)
        smaller_tests += any(t["side"] < max(d["side"] for d in train_details) for t in test_details)
        odd_tests += any(d["side"] % 2 for d in test_details)
        square_tests += any(d["shape"][0] == d["shape"][1] for d in test_details)
        landscape_tests += any(d["shape"][0] < d["shape"][1] for d in test_details)
        corner_tests += any(d["contact"] == 2 for d in test_details)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) == 42 and len(profiles) >= 30
    assert len(backgrounds) == 10 and len(positions) >= 150 and len(roles) >= 100
    assert mixed_backgrounds >= 15 and reversed_roles >= 10 and expanded_tests >= 5
    assert smaller_tests >= 12 and odd_tests >= 12
    assert square_tests >= 5 and landscape_tests >= 8 and corner_tests >= 5
    for seed, ntrain, ntest in ((6001, 2, 1), (6002, 10, 1), (6003, 10, 4), (6004, None, 1), (6005, None, 4), (6006, 2, None), (6007, 10, None)):
        puzzle = generate_puzzle(seed, ntrain, ntest)
        validate_generated_puzzle(puzzle)
        assert ntrain is None or len(puzzle["train"]) == ntrain
        assert ntest is None or len(puzzle["test"]) == ntest
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}, {"num_train": True}, {"num_test": 2.0}):
        _expect_value_error(lambda kwargs=kwargs: generate_puzzle(8001, **kwargs))
