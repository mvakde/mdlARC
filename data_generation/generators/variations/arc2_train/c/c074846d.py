import numpy as np


# Directions are ordered clockwise: north, east, south, west.
_DIRECTION_VECTORS = (
    (-1, 0),
    (0, 1),
    (1, 0),
    (0, -1),
)


# Exact source-task pairs, used to prevent accidental copying.
_SOURCE_PAIR_KEYS = {
    (
        (
            "0000000",
            "0000000",
            "0000000",
            "0222500",
            "0000000",
            "0000000",
        ),
        (
            "0000200",
            "0000200",
            "0000200",
            "0333500",
            "0000000",
            "0000000",
        ),
    ),
    (
        (
            "000",
            "052",
            "000",
        ),
        (
            "000",
            "053",
            "020",
        ),
    ),
    (
        (
            "0000000",
            "0000000",
            "0000000",
            "0000000",
            "0000500",
            "0000200",
            "0000200",
        ),
        (
            "0000000",
            "0000000",
            "0000000",
            "0000000",
            "0022500",
            "0000300",
            "0000300",
        ),
    ),
    (
        (
            "0000000",
            "0000000",
            "0002000",
            "0005000",
            "0000000",
            "0000000",
            "0000000",
        ),
        (
            "0000000",
            "0000000",
            "0003000",
            "0005200",
            "0000000",
            "0000000",
            "0000000",
        ),
    ),
    (
        (
            "0000000",
            "0000000",
            "0225000",
            "0000000",
            "0000000",
        ),
        (
            "0002000",
            "0002000",
            "0335000",
            "0000000",
            "0000000",
        ),
    ),
    (
        (
            "00000200000",
            "00000200000",
            "00000200000",
            "00000200000",
            "00000500000",
            "00000000000",
            "00000000000",
            "00000000000",
            "00000000000",
        ),
        (
            "00000300000",
            "00000300000",
            "00000300000",
            "00000300000",
            "00000522220",
            "00000000000",
            "00000000000",
            "00000000000",
            "00000000000",
        ),
    ),
    (
        (
            "000000000",
            "000000000",
            "000000000",
            "000000000",
            "000000000",
            "000500000",
            "000200000",
            "000200000",
            "000200000",
        ),
        (
            "000000000",
            "000000000",
            "000000000",
            "000000000",
            "000000000",
            "222500000",
            "000300000",
            "000300000",
            "000300000",
        ),
    ),
}


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _normalize_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be a nonnegative integer")
    seed_value = int(seed)
    if seed_value < 0:
        raise ValueError("seed must be a nonnegative integer")
    return seed_value


def _validate_requested_count(
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
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return result


def _array_key(grid: np.ndarray) -> tuple:
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.ravel()),
    )


def _pair_key(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> tuple:
    return (_array_key(input_grid), _array_key(output_grid))


def _grid_rows(grid: np.ndarray) -> tuple[str, ...]:
    return tuple(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _pair_rows_key(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (_grid_rows(input_grid), _grid_rows(output_grid))


def _ray_cells(
    pivot: tuple[int, int],
    direction: int,
    length: int,
) -> list[tuple[int, int]]:
    dr, dc = _DIRECTION_VECTORS[direction]
    pr, pc = pivot
    return [
        (pr + step * dr, pc + step * dc)
        for step in range(1, length + 1)
    ]


def _cells_fit(
    cells: list[tuple[int, int]],
    height: int,
    width: int,
) -> bool:
    return all(
        0 <= row < height and 0 <= col < width
        for row, col in cells
    )


def _valid_pivots(
    height: int,
    width: int,
    direction: int,
    length: int,
) -> list[tuple[int, int]]:
    clockwise_direction = (direction + 1) % 4
    candidates: list[tuple[int, int]] = []

    for row in range(height):
        for col in range(width):
            pivot = (row, col)
            source = _ray_cells(pivot, direction, length)
            destination = _ray_cells(
                pivot, clockwise_direction, length
            )
            if (
                _cells_fit(source, height, width)
                and _cells_fit(destination, height, width)
            ):
                candidates.append(pivot)

    return candidates


def _endpoint_on_boundary(
    pivot: tuple[int, int],
    direction: int,
    length: int,
    height: int,
    width: int,
) -> bool:
    dr, dc = _DIRECTION_VECTORS[direction]
    row = pivot[0] + length * dr
    col = pivot[1] + length * dc
    return (
        row == 0
        or row == height - 1
        or col == 0
        or col == width - 1
    )


def _choose_counts(rng, num_train, num_test):
    train = _validate_requested_count("num_train", num_train, 2, 10)
    test = _validate_requested_count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(test + 1, 11))
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _palette(rng, trail, base=None):
    if base is None:
        available = [x for x in range(10) if x != trail]
        return tuple(int(x) for x in rng.choice(available, size=3, replace=False)) + (trail,)
    changed = int(rng.integers(1, 8))
    roles = list(base[:3])
    for index in range(3):
        if changed & (1 << index):
            choices = [x for x in range(10) if x != trail and x != base[index] and x not in roles[:index]]
            roles[index] = int(choices[int(rng.integers(len(choices)))])
    if len(set(roles + [trail])) != 4:
        available = [x for x in range(10) if x != trail]
        for index in range(3):
            if roles[index] in roles[:index] or roles[index] == trail:
                roles[index] = next(x for x in available if x not in roles[:index] and x not in roles[index + 1:])
    return tuple(roles) + (trail,)


def _border_kind(pivot, direction, length, height, width):
    source = _endpoint_on_boundary(pivot, direction, length, height, width)
    target = _endpoint_on_boundary(pivot, (direction + 1) % 4, length, height, width)
    if pivot[0] in (0, height - 1) and pivot[1] in (0, width - 1):
        return "corner"
    if source and target:
        return "both"
    if source:
        return "source"
    if target:
        return "destination"
    return "neither"


def _geometry(rng, direction, length):
    desired = ("neither", "source", "destination", "both", "corner")[
        int(rng.integers(5))
    ]
    last = None
    for _ in range(40):
        minimum = max(3, length + 1)
        # Independent side margins create compact squares and both rectangle orientations.
        slacks = []
        for _axis in range(2):
            band = int(rng.integers(20))
            if band < 15:
                slacks.append(int(rng.integers(0, 4)))
            elif band < 19:
                slacks.append(int(rng.integers(4, 9)))
            else:
                slacks.append(int(rng.integers(9, 14)))
        height, width = minimum + slacks[0], minimum + slacks[1]
        pivots = _valid_pivots(height, width, direction, length)
        matches = [p for p in pivots if _border_kind(p, direction, length, height, width) == desired]
        last = (height, width, pivots)
        if matches:
            return height, width, matches[int(rng.integers(len(matches)))]
    height, width, pivots = last
    return height, width, pivots[int(rng.integers(len(pivots)))]


def _make_pair(rng, direction, length, roles):
    height, width, pivot = _geometry(rng, direction, length)
    background, pivot_color, arm, trail = roles
    source = _ray_cells(pivot, direction, length)
    destination = _ray_cells(pivot, (direction + 1) % 4, length)
    inp = np.full((height, width), background, dtype=np.int8)
    inp[pivot] = pivot_color
    for cell in source:
        inp[cell] = arm
    out = inp.copy()
    for cell in source:
        out[cell] = trail
    for cell in destination:
        out[cell] = arm
    return {"input": inp, "output": out}


def _analyze_pair(inp, out):
    _require(inp.shape == out.shape, "dimensions changed")
    colors, counts = np.unique(inp, return_counts=True)
    _require(len(colors) == 3, "input must have three roles")
    background = int(colors[int(np.argmax(counts))])
    _require(int(np.max(counts)) > int(np.sum(counts) - np.max(counts)), "background is not dominant")
    new = set(int(x) for x in np.unique(out)) - set(int(x) for x in colors)
    _require(len(new) == 1 and len(np.unique(out)) == 4, "one distinct trail must be introduced")
    trail = next(iter(new))
    source = {tuple(map(int, p)) for p in np.argwhere(out == trail)}
    length = len(source)
    _require(length > 0, "empty source")
    arm_colors = {int(inp[p]) for p in source}
    _require(len(arm_colors) == 1, "source is not one color")
    arm = next(iter(arm_colors))
    _require(arm != background and int(np.count_nonzero(inp == arm)) == length, "invalid arm")
    pivot_colors = set(int(x) for x in colors) - {background, arm}
    _require(len(pivot_colors) == 1, "pivot color missing")
    pivot_color = next(iter(pivot_colors))
    pivots = np.argwhere(inp == pivot_color)
    _require(len(pivots) == 1, "pivot is not a singleton")
    pivot = tuple(map(int, pivots[0]))
    directions = [d for d in range(4) if set(_ray_cells(pivot, d, length)) == source]
    _require(len(directions) == 1, "arm is not one straight attached ray")
    direction = directions[0]
    destination = _ray_cells(pivot, (direction + 1) % 4, length)
    height, width = inp.shape
    _require(_cells_fit(destination, height, width), "destination is clipped")
    _require(all(int(inp[p]) == background for p in destination), "destination is occupied")
    expected = inp.copy()
    for p in source:
        expected[p] = trail
    for p in destination:
        expected[p] = arm
    _require(np.array_equal(expected, out), "incorrect clockwise rotation or trail")
    return {"height": height, "width": width, "length": length,
            "direction": direction, "pivot": pivot,
            "roles": (background, pivot_color, arm, trail),
            "border": _border_kind(pivot, direction, length, height, width)}


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed_value = _normalize_seed(seed)
    rng = np.random.default_rng(seed_value)
    train_count, test_count = _choose_counts(rng, num_train, num_test)
    trail = int(rng.integers(10))
    base = _palette(rng, trail)
    palette_count = 1 if bool(rng.integers(3) == 0) else int(rng.integers(2, min(4, train_count) + 1))
    palettes = [base]
    for _ in range(1, palette_count):
        for _attempt in range(30):
            candidate = _palette(rng, trail, base)
            if candidate not in palettes and all(
                set(candidate[:3]) != set(previous[:3]) for previous in palettes
            ):
                palettes.append(candidate)
                break
        else:
            raise AssertionError("palette selection failed")

    mode = int(rng.integers(3))
    first_direction = int(rng.integers(4))
    if mode == 0:  # focused demonstrations and familiar test direction
        train_directions = [first_direction] * train_count
        test_directions = [first_direction] * test_count
    elif mode == 1:  # a nonparallel pair, then an unseen test direction
        second = (first_direction + (1 if bool(rng.integers(2)) else 3)) % 4
        train_directions = [first_direction, second]
        train_directions += [int(rng.choice((first_direction, second))) for _ in range(train_count - 2)]
        unseen = [d for d in range(4) if d not in (first_direction, second)]
        test_directions = [int(rng.choice(unseen))]
        test_directions += [int(rng.integers(4)) for _ in range(test_count - 1)]
    else:
        train_directions = [int(rng.integers(4)) for _ in range(train_count)]
        test_directions = [int(rng.integers(4)) for _ in range(test_count)]

    train_palette_indices = list(range(palette_count))
    train_palette_indices += [int(rng.integers(palette_count)) for _ in range(train_count - palette_count)]
    train_lengths = [int(rng.integers(2, 7))]
    for index in range(1, train_count):
        low = 2 if train_palette_indices[index] != 0 else 1
        train_lengths.append(int(rng.integers(low, 7)))
    test_lengths = []
    for _ in range(test_count):
        if bool(rng.integers(4) == 0) and max(train_lengths) < 8:
            test_lengths.append(int(rng.integers(max(train_lengths) + 1, 9)))
        else:
            test_lengths.append(int(rng.integers(1, 9)))
    test_palette_indices = [0 if length == 1 else int(rng.integers(palette_count)) for length in test_lengths]

    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count, directions, lengths, indices in (
        ("train", train_count, train_directions, train_lengths, train_palette_indices),
        ("test", test_count, test_directions, test_lengths, test_palette_indices),
    ):
        for index in range(count):
            for _attempt in range(80):
                pair = _make_pair(rng, directions[index], lengths[index], palettes[indices[index]])
                key = _array_key(pair["input"])
                if key in seen or _pair_rows_key(pair["input"], pair["output"]) in _SOURCE_PAIR_KEYS:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise AssertionError("could not construct a distinct non-source pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "invalid pair counts")
    analyses = {"train": [], "test": []}
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be a 2D array")
                _require(all(1 <= x <= 30 for x in grid.shape), "invalid grid dimensions")
                _require(np.issubdtype(grid.dtype, np.integer), "grid must be integer")
                _require(bool(np.all((grid >= 0) & (grid <= 9))), "invalid ARC color")
            inp, out = pair["input"], pair["output"]
            ikey, pkey = _array_key(inp), _pair_key(inp, out)
            _require(ikey not in seen_inputs and pkey not in seen_pairs, "duplicate pair")
            seen_inputs.add(ikey)
            seen_pairs.add(pkey)
            _require(_pair_rows_key(inp, out) not in _SOURCE_PAIR_KEYS, "source pair copied")
            info = _analyze_pair(inp, out)
            _require(info["height"] >= 3 and info["width"] >= 3 and 1 <= info["length"] <= 8, "outside supported geometry")
            analyses[split].append(info)
    train = analyses["train"]
    test = analyses["test"]
    _require(all(info["length"] <= 6 for info in train), "training arm exceeds source regime")
    _require(train[0]["length"] >= 2, "first training arm cannot establish color roles")
    trail = train[0]["roles"][3]
    _require(all(x["roles"][3] == trail for x in train + test), "trail convention varies")
    train_palettes = {x["roles"] for x in train}
    _require(
        len({frozenset(roles[:3]) for roles in train_palettes}) == len(train_palettes),
        "input colors imply conflicting trained role assignments",
    )
    _require(all(x["roles"] in train_palettes for x in test), "test palette not taught in training")
    _require(any(x["length"] >= 2 for x in train), "no unambiguous arm demonstration")
    _require(all(x["length"] >= 2 for x in train if x["roles"] != train[0]["roles"]), "new training palette has ambiguous length one")
    base = train[0]["roles"]
    _require(any(x["roles"] == base and x["length"] >= 2 for x in train), "base roles unclear")
    _require(all(x["roles"] == base for x in test if x["length"] == 1), "ambiguous length-one test palette")


def _puzzle_fingerprint(puzzle):
    return tuple(tuple(_pair_key(p["input"], p["output"]) for p in puzzle[s]) for s in ("train", "test"))


def _predict_from_train(train, test_input):
    trained = [_analyze_pair(pair["input"], pair["output"]) for pair in train]
    trail = trained[0]["roles"][3]
    colors, counts = np.unique(test_input, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    foreground = [int(color) for color in colors if color != background]
    repetitions = [color for color in foreground if int(np.count_nonzero(test_input == color)) > 1]
    if repetitions:
        arm = repetitions[0]
        pivot_color = next(color for color in foreground if color != arm)
    else:
        known = {frozenset(info["roles"][:3]): info["roles"] for info in trained}
        roles = known[frozenset(int(color) for color in colors)]
        pivot_color, arm = roles[1], roles[2]
    pivot = tuple(int(x) for x in np.argwhere(test_input == pivot_color)[0])
    source = {tuple(int(x) for x in p) for p in np.argwhere(test_input == arm)}
    length = len(source)
    direction = next(d for d in range(4) if set(_ray_cells(pivot, d, length)) == source)
    output = test_input.copy()
    for p in source:
        output[p] = trail
    for p in _ray_cells(pivot, (direction + 1) % 4, length):
        output[p] = arm
    return output


def self_test() -> None:
    train_counts, test_counts, shapes, layouts, palettes = set(), set(), set(), set(), set()
    direction_sets, test_directions, test_lengths, borders = set(), set(), set(), set()
    varying_palettes = focused = unseen = train_large_test_small = 0
    shared_palettes = changed_background = changed_pivot = changed_arm = 0
    familiar_test = longer_test = combined = 0
    test_borders, train_shapes, test_shapes = set(), set(), set()
    fingerprints = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        fingerprints.add(_puzzle_fingerprint(puzzle))
        train_info = [_analyze_pair(p["input"], p["output"]) for p in puzzle["train"]]
        test_info = [_analyze_pair(p["input"], p["output"]) for p in puzzle["test"]]
        train_dirs = {x["direction"] for x in train_info}
        direction_sets.add(tuple(sorted(train_dirs)))
        focused += len(train_dirs) == 1
        unseen += any(x["direction"] not in train_dirs for x in test_info)
        varying_palettes += len({x["roles"] for x in train_info}) > 1
        shared_palettes += len({x["roles"] for x in train_info}) == 1
        changed_background += len({x["roles"][0] for x in train_info}) > 1
        changed_pivot += len({x["roles"][1] for x in train_info}) > 1
        changed_arm += len({x["roles"][2] for x in train_info}) > 1
        familiar_test += any(x["length"] <= max(y["length"] for y in train_info) for x in test_info)
        longer_test += any(x["length"] > max(y["length"] for y in train_info) for x in test_info)
        train_large_test_small += max(x["height"] * x["width"] for x in train_info) > min(x["height"] * x["width"] for x in test_info)
        for pair in puzzle["test"]:
            _require(np.array_equal(_predict_from_train(puzzle["train"], pair["input"].copy()), pair["output"]), "training examples do not determine a test output")
        for split, infos in (("train", train_info), ("test", test_info)):
            for x in infos:
                shapes.add((split, x["height"], x["width"]))
                layouts.add((split, x["height"], x["width"], x["pivot"], x["direction"], x["length"]))
                palettes.add(x["roles"])
                borders.add(x["border"])
                if split == "test":
                    test_directions.add(x["direction"])
                    test_lengths.add(x["length"])
                    test_borders.add(x["border"])
                    test_shapes.add((x["height"], x["width"]))
                    if len({y["roles"] for y in train_info}) > 1 and x["direction"] in (1, 3) and x["border"] != "neither":
                        combined += 1
                else:
                    train_shapes.add((x["height"], x["width"]))
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic count diversity failed")
    _require(len(fingerprints) >= 40 and len(shapes) >= 35 and len(layouts) >= 100, "structural diversity failed")
    _require(len(palettes) >= 25 and varying_palettes >= 10, "palette diversity failed")
    _require(shared_palettes >= 5 and min(changed_background, changed_pivot, changed_arm) >= 5, "role variation failed")
    _require(focused >= 4 and unseen >= 4 and len(direction_sets) >= 6, "demonstration diversity failed")
    _require(train_large_test_small >= 4, "scale direction diversity failed")
    _require(familiar_test >= 5 and longer_test >= 5, "test length regimes failed")
    _require(len(train_shapes) >= 35 and len(test_shapes) >= 25 and combined >= 2, "shape or combined variation failed")
    _require(test_directions == {0, 1, 2, 3} and {1, 2, 3, 4, 5, 6}.issubset(test_lengths), "test direction or length diversity failed")
    _require(borders == {"neither", "source", "destination", "both", "corner"}, "border coverage failed")
    _require(test_borders == borders, "test border coverage failed")
    _require(_puzzle_fingerprint(generate_puzzle(9)) == _puzzle_fingerprint(generate_puzzle(9)), "nondeterminism")
    for ntrain, ntest in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(101, ntrain, ntest)
        validate_generated_puzzle(p)
        if ntrain is not None:
            _require(len(p["train"]) == ntrain, "train count ignored")
        if ntest is not None:
            _require(len(p["test"]) == ntest, "test count ignored")
    for ntrain, ntest in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (2, 3)):
        try:
            generate_puzzle(101, ntrain, ntest)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
