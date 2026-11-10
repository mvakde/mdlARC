import numpy as np


_SOURCE_PAIRS = {
    (
        "1333330000/1113333300/1113333000/1133330000/1133330000/"
        "1333333300/1333333300/1111333333/1113333300/1133333000",
        "1333330000/1333331100/1133331000/1133330000/1133330000/"
        "3333333100/3333333100/3333331111/1333331100/1133331000",
    ),
    (
        "3333377777/3337777779/3377777799/7777777999/7777777999/"
        "3777779999/3377777999/3337777999/7777777999/3377779999",
        "3777773333/7777773339/7777773399/7777777999/7777777999/"
        "3777779999/3777773999/3377773999/7777777999/3377779999",
    ),
    (
        "55777777766666666/55577777776666666/55577777766666666/"
        "55555577766666666/55557777777666666/55577777777666666/"
        "55577777777666666/55577777776666666/55557777766666666/"
        "55577777776666666/55777777777666666/57777777777766666/"
        "55777777777776666/55577777777777666/55557777777776666/"
        "55557777777766666/55577777777776666",
        "55777777766666666/55777777756666666/55577777766666666/"
        "55555577766666666/55777777755666666/57777777755666666/"
        "57777777755666666/55777777756666666/55557777766666666/"
        "55777777756666666/77777777755666666/77777777777566666/"
        "77777777777556666/77777777777555666/77777777755556666/"
        "57777777755566666/77777777775556666",
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _integer_count(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(
    seed: int, rng: np.random.Generator,
    num_train: int | None, num_test: int | None,
) -> tuple[int, int]:
    if num_train is not None:
        if not _integer_count(num_train) or not 2 <= int(num_train) <= 10:
            raise ValueError("num_train must be an integer from 2 through 10")
        num_train = int(num_train)
    if num_test is not None:
        if not _integer_count(num_test) or not 1 <= int(num_test) <= 4:
            raise ValueError("num_test must be an integer from 1 through 4")
        num_test = int(num_test)
    if num_train is None and num_test is None:
        num_test = int(seed) % 4 + 1
        num_train = num_test + 1 + (int(seed) // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _dimensions(
    rng: np.random.Generator, num_train: int, num_test: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    # A source-like larger odd test is one regime among several. Heights are
    # sampled separately from widths even when training widths coincide.
    regime = int(rng.integers(0, 4))
    if regime == 0:
        width = int(rng.choice(np.asarray([8, 10, 12, 14])))
        train = [(int(rng.integers(8, 22)), width) for _ in range(num_train)]
        if train[1][0] == train[0][0]:
            train[1] = (8 + (train[1][0] - 7) % 14, width)
        test_width = width + int(rng.choice(np.asarray([3, 5, 7, 9])))
        test = [(int(rng.integers(11, 29)), test_width) for _ in range(num_test)]
    else:
        low = 10 if regime == 2 else 8
        train = [
            (int(rng.integers(7, 27)), int(rng.integers(low, 24)))
            for _ in range(num_train)
        ]
        if train[1][1] == train[0][1]:
            train[1] = (train[1][0], 8 + (train[1][1] - 7) % 16)
        test = []
        for _ in range(num_test):
            chosen_width = train[int(rng.integers(0, num_train))][1]
            if regime == 1:
                width = chosen_width
            elif regime == 2:
                width = int(rng.integers(7, chosen_width))
            else:
                width = int(rng.integers(7, 26))
            test.append((int(rng.integers(7, 29)), width))
    return train, test


def _palettes(
    rng: np.random.Generator, num_train: int, num_test: int,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    colors = np.arange(10)
    first = tuple(int(x) for x in rng.choice(colors, size=3, replace=False))
    if rng.random() < 0.38:
        # Retain the band's literal color while an outer color crosses sides.
        crossed_role = int(rng.choice(np.asarray([0, 2])))
        new_role = 2 - crossed_role
        second = [-1, first[1], -1]
        second[new_role] = first[crossed_role]
        available = [x for x in range(10) if x not in (first[1], first[crossed_role])]
        second[crossed_role] = int(rng.choice(np.asarray(available)))
    else:
        crossed_role = int(rng.integers(0, 3))
        new_role = (crossed_role + int(rng.integers(1, 3))) % 3
        second = [-1, -1, -1]
        second[new_role] = first[crossed_role]
        other = rng.choice(colors[colors != first[crossed_role]], size=2, replace=False)
        for slot, color in zip((i for i in range(3) if i != new_role), other):
            second[slot] = int(color)
    train = [first, tuple(second)]
    train += [
        tuple(int(x) for x in rng.choice(colors, size=3, replace=False))
        for _ in range(num_train - 2)
    ]
    familiar = {color for roles in train for color in roles}
    fresh = [color for color in range(10) if color not in familiar]
    test = []
    for _ in range(num_test):
        if len(fresh) >= 2 and rng.random() < 0.38:
            outer = rng.choice(np.asarray(fresh), size=2, replace=False)
        elif familiar and rng.random() < 0.58:
            first_outer = int(rng.choice(np.asarray(sorted(familiar))))
            other_pool = [color for color in range(10) if color != first_outer]
            outer = (first_outer, int(rng.choice(np.asarray(other_pool))))
        else:
            outer = rng.choice(colors, size=2, replace=False)
        left, right = int(outer[0]), int(outer[1])
        band = int(rng.choice(np.asarray([x for x in range(10) if x not in (left, right)])))
        test.append((left, band, right))
    return train, test


def _decompose(row: np.ndarray, roles: tuple[int, int, int]) -> tuple[int, int, int] | None:
    first, band, opposite = roles
    width = len(row)
    index = 0
    while index < width and int(row[index]) == first:
        index += 1
    a = index
    while index < width and int(row[index]) == band:
        index += 1
    b = index - a
    if b == 0:
        return None
    while index < width and int(row[index]) == opposite:
        index += 1
    if index != width:
        return None
    return a, b, width - a - b


def _roles(grid: np.ndarray) -> tuple[int, int, int] | None:
    palette = [int(x) for x in np.unique(grid)]
    if len(palette) != 3:
        return None
    candidates = [
        (a, b, c) for a in palette for b in palette for c in palette
        if len({a, b, c}) == 3
        and all(_decompose(row, (a, b, c)) is not None for row in grid)
    ]
    return candidates[0] if len(candidates) == 1 else None


def _transform(grid: np.ndarray, roles: tuple[int, int, int]) -> np.ndarray:
    first, band, opposite = roles
    pivot = grid.shape[1] // 2 + 1
    rows = []
    for row in grid:
        profile = _decompose(row, roles)
        _require(profile is not None, "malformed input row")
        a, b, c = profile
        shift = min(a, max(0, a + b - pivot))
        rows.append(
            [first] * (a - shift) + [band] * b
            + [first] * shift + [opposite] * c
        )
    return np.asarray(rows, dtype=np.int8)


def _event_profile(
    rng: np.random.Generator, width: int, event: str,
) -> tuple[int, int]:
    pivot = width // 2 + 1
    if event == "before":
        endpoint = int(rng.integers(3, pivot))
        first = int(rng.integers(1, endpoint))
    elif event == "boundary":
        endpoint = pivot
        first = int(rng.integers(1, pivot))
    elif event == "partial":
        excess = int(rng.integers(1, min(3, width - pivot - 1) + 1))
        endpoint = pivot + excess
        first = int(rng.integers(excess + 1, endpoint))
    elif event == "residual":
        first = int(rng.integers(1, max(2, width - pivot)))
        band = int(rng.integers(pivot + 1, width - first + 1))
        endpoint = first + band
    elif event == "no_first":
        first = 0
        endpoint = int(rng.integers(2, width))
    elif event == "no_opposite":
        first = int(rng.integers(1, width - 1))
        endpoint = width
    else:
        raise AssertionError("unknown row event")
    return first, endpoint


def _profiles(
    rng: np.random.Generator, height: int, width: int,
    style: int, density: int,
) -> list[tuple[int, int]]:
    pivot = width // 2 + 1
    low_first = 1
    high_endpoint = width - 1
    first = int(rng.integers(low_first, min(width - 1, pivot + 1)))
    endpoint_low = max(first + 1, pivot - (2 if density == 0 else 1))
    endpoint = int(rng.integers(endpoint_low, high_endpoint + 1))
    if density == 2 and rng.random() < 0.5:
        first = int(rng.integers(1, max(2, width // 4 + 1)))
        endpoint = int(rng.integers(max(first + 1, pivot + 1), width))
    result = [(first, endpoint)]
    slope_a = int(rng.choice(np.asarray([-1, 1])))
    slope_e = int(rng.choice(np.asarray([-1, 1])))
    for _ in range(1, height):
        if style == 0:  # short correlated steps
            da = int(rng.choice(np.asarray([-1, 0, 0, 0, 1])))
            de = int(rng.choice(np.asarray([-1, 0, 0, 0, 1])))
        elif style == 1:  # long plateaus
            da = int(rng.choice(np.asarray([-2, -1, 0, 0, 0, 0, 0, 1, 2])))
            de = int(rng.choice(np.asarray([-2, -1, 0, 0, 0, 0, 0, 1, 2])))
        elif style == 2:  # coherent slope
            da = slope_a * int(rng.random() < 0.65)
            de = slope_e * int(rng.random() < 0.65)
        elif style == 3:  # widening or narrowing
            direction = slope_a
            da = -direction * int(rng.random() < 0.7)
            de = direction * int(rng.random() < 0.7)
        else:  # jagged rows remain in the distribution
            first = int(rng.integers(low_first, width - 1))
            endpoint = int(rng.integers(first + 1, high_endpoint + 1))
            result.append((first, endpoint))
            continue
        first = min(max(low_first, first + da), width - 2)
        endpoint = min(max(first + 1, endpoint + de), high_endpoint)
        result.append((first, endpoint))
    if density == 2:
        omitted_rows = int(rng.integers(1, max(2, height // 4 + 1)))
        for index in rng.choice(np.arange(height), size=omitted_rows, replace=False):
            event = "no_first" if rng.random() < 0.5 else "no_opposite"
            result[int(index)] = _event_profile(rng, width, event)
    return result


def _row_events(profiles: list[tuple[int, int]], width: int) -> set[str]:
    pivot = width // 2 + 1
    events = set()
    for first, endpoint in profiles:
        band = endpoint - first
        shift = min(first, max(0, endpoint - pivot))
        if endpoint < pivot:
            events.add("before")
        if endpoint == pivot:
            events.add("boundary")
        if 0 < shift < first:
            events.add("partial")
        if first > 0 and shift == first:
            events.add("full")
        if first > 0 and shift == first and band > pivot:
            events.add("residual")
        if first == 0:
            events.add("no_first")
        if endpoint == width:
            events.add("no_opposite")
        if shift:
            events.add("changed")
        if first > 0 and endpoint < width:
            events.add("anchor")
    return events


def _pair(
    rng: np.random.Generator, dimensions: tuple[int, int],
    roles: tuple[int, int, int], required: tuple[str, ...],
) -> dict:
    height, width = dimensions
    style = int(rng.integers(0, 5))
    density = int(rng.choice(np.asarray([0, 0, 1, 1, 2])))
    profiles = _profiles(rng, height, width, style, density)
    reserved = set()
    for event in required:
        available = [i for i in range(1, height - 1) if i not in reserved]
        position = int(rng.choice(np.asarray(available)))
        profiles[position] = _event_profile(rng, width, event)
        reserved.add(position)
    if "anchor" not in _row_events(profiles, width):
        profiles[0] = _event_profile(rng, width, "boundary")
    if style in (0, 1, 2, 3) and rng.random() < 0.56:
        positions = [i for i in range(1, height) if i not in reserved and i - 1 not in reserved]
        if positions:
            position = int(rng.choice(np.asarray(positions)))
            profiles[position] = profiles[position - 1]
    pivot = width // 2 + 1
    changed = sum(first > 0 and endpoint > pivot for first, endpoint in profiles)
    for raw_position in rng.permutation(height):
        if changed >= 3:
            break
        position = int(raw_position)
        first, endpoint = profiles[position]
        if position not in reserved and (first == 0 or endpoint <= pivot):
            profiles[position] = _event_profile(rng, width, "partial")
            changed += 1
    first_color, band_color, opposite_color = roles
    rows = [
        [first_color] * first + [band_color] * (endpoint - first)
        + [opposite_color] * (width - endpoint)
        for first, endpoint in profiles
    ]
    input_grid = np.asarray(rows, dtype=np.int8)
    return {"input": input_grid, "output": _transform(input_grid, roles)}


def _grid_key(grid: np.ndarray) -> tuple:
    return grid.shape, grid.tobytes()


def _source_copy(pair: dict) -> bool:
    def text_grid(grid: np.ndarray) -> str:
        return "/".join("".join(str(int(x)) for x in row) for row in grid)
    return (text_grid(pair["input"]), text_grid(pair["output"])) in _SOURCE_PAIRS


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)
    train_dimensions, test_dimensions = _dimensions(rng, train_count, test_count)
    train_roles, test_roles = _palettes(rng, train_count, test_count)
    puzzle = {"train": [], "test": []}
    seen = set()
    for section, dimensions_list, palettes in (
        ("train", train_dimensions, train_roles),
        ("test", test_dimensions, test_roles),
    ):
        for index, (dimensions, roles) in enumerate(zip(dimensions_list, palettes)):
            required = ()
            if section == "train" and index == 0:
                required = ("before", "partial")
            if section == "train" and index == 1:
                required = ("boundary", "residual")
            for _ in range(20):
                pair = _pair(rng, dimensions, roles, required)
                key = _grid_key(pair["input"])
                if key not in seen and not _source_copy(pair):
                    seen.add(key)
                    puzzle[section].append(pair)
                    break
            else:
                raise AssertionError("could not construct distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle sections")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "section lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count, "pair counts")
    seen_inputs, seen_pairs = set(), set()
    train_events = set()
    train_roles = []
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair fields")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid array")
                _require(np.issubdtype(grid.dtype, np.integer), "grid integer dtype")
                _require(all(1 <= size <= 30 for size in grid.shape), "grid dimensions")
                _require(bool(np.all((grid >= 0) & (grid <= 9))), "grid colors")
            inp, out = pair["input"], pair["output"]
            _require(inp.shape == out.shape and not np.array_equal(inp, out), "nontrivial same-size output")
            _require(not _source_copy(pair), "source pair copied")
            input_key = _grid_key(inp)
            pair_key = (input_key, _grid_key(out))
            _require(input_key not in seen_inputs and pair_key not in seen_pairs, "duplicate pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            roles = _roles(inp)
            _require(roles is not None, "input roles not unique")
            _require(np.array_equal(_transform(inp, roles), out), "incorrect row transformation")
            _require(set(np.unique(inp)) == set(np.unique(out)), "palette changed")
            profiles = [_decompose(row, roles) for row in inp]
            events = _row_events([(a, a + b) for a, b, _ in profiles], inp.shape[1])
            changed_rows = sum(
                not np.array_equal(left, right) for left, right in zip(inp, out)
            )
            _require("anchor" in events and changed_rows >= 3, "insufficient pair evidence")
            if section == "train":
                train_events.update(events)
                train_roles.append(roles)
    _require({"before", "boundary", "partial", "full", "residual"} <= train_events,
             "training misses a pivot or bounded-shift case")
    # Color is a visible row role, never a hidden puzzle-wide mapping.
    _require(any(
        color in roles and roles.index(color) != train_roles[0].index(color)
        for roles in train_roles[1:] for color in train_roles[0] if color in roles
    ), "training does not demonstrate role reassignment")


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(
        tuple((_grid_key(pair["input"]), _grid_key(pair["output"])) for pair in puzzle[section])
        for section in ("train", "test")
    )


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts = {len(p["train"]) for p in puzzles}
    test_counts = {len(p["test"]) for p in puzzles}
    shapes, layouts, role_triplets = set(), set(), set()
    test_relations, repeated, smooth, jagged, cross_roles = set(), 0, 0, 0, 0
    omitted_first = omitted_opposite = ordinary = 0
    rectangles = squares = combined = 0
    familiar_test_outer = fresh_test_outer = 0
    recurring_first_band = changing_first_band = source_analog = 0
    train_parities = set()
    varied_train_widths = 0
    varied_train_heights_fixed_width = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_widths = [pair["input"].shape[1] for pair in puzzle["train"]]
        train_heights = [pair["input"].shape[0] for pair in puzzle["train"]]
        train_parities.update(width % 2 for width in train_widths)
        varied_train_widths += len(set(train_widths)) > 1
        varied_train_heights_fixed_width += len(set(train_widths)) == 1 and len(set(train_heights)) > 1
        roles_list = []
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inp, out = pair["input"], pair["output"]
                roles = _roles(inp)
                _require(roles is not None, "self-test role inference")
                roles_list.append(roles)
                role_triplets.add(roles)
                shapes.add(inp.shape)
                rectangles += inp.shape[0] != inp.shape[1]
                squares += inp.shape[0] == inp.shape[1]
                profiles = [_decompose(row, roles) for row in inp]
                layouts.add((inp.shape, tuple(profiles)))
                first, band, opposite = roles
                pivot = inp.shape[1] // 2 + 1
                for row, result, (a, b, c) in zip(inp, out, profiles):
                    shift = min(a, max(0, a + b - pivot))
                    _require(
                        list(result) == [first] * (a - shift) + [band] * b
                        + [first] * shift + [opposite] * c,
                        "independent row reconstruction",
                    )
                has_repeat = any(profiles[i] == profiles[i - 1] for i in range(1, len(profiles)))
                repeated += has_repeat
                local_steps = sum(
                    abs(profiles[i][0] - profiles[i - 1][0]) <= 1
                    and abs((profiles[i][0] + profiles[i][1]) -
                            (profiles[i - 1][0] + profiles[i - 1][1])) <= 1
                    for i in range(1, len(profiles))
                )
                smooth += local_steps >= (len(profiles) - 1) // 2
                jagged += local_steps < (len(profiles) - 1) // 3
                no_first = any(a == 0 for a, b, c in profiles)
                no_opposite = any(c == 0 for a, b, c in profiles)
                omitted_first += no_first
                omitted_opposite += no_opposite
                ordinary += not no_first and not no_opposite
                combined += inp.shape[0] != inp.shape[1] and has_repeat and (no_first or no_opposite)
        cross_roles += any(
            color in later and earlier.index(color) != later.index(color)
            for earlier in roles_list[:len(puzzle["train"])]
            for later in roles_list[:len(puzzle["train"])]
            for color in earlier if color in later
        )
        recurring_first_band += roles_list[0][1] == roles_list[1][1]
        changing_first_band += roles_list[0][1] != roles_list[1][1]
        familiar_colors = {color for roles in roles_list[:len(puzzle["train"])] for color in roles}
        source_analog += (
            len(set(train_widths)) == 1
            and train_widths[0] % 2 == 0
            and any(pair["input"].shape[1] % 2 == 1
                    and pair["input"].shape[1] > train_widths[0]
                    for pair in puzzle["test"])
        )
        for pair in puzzle["test"]:
            width = pair["input"].shape[1]
            roles = _roles(pair["input"])
            has_familiar_outer = roles[0] in familiar_colors or roles[2] in familiar_colors
            familiar_test_outer += has_familiar_outer
            fresh_test_outer += not has_familiar_outer
            test_relations.add("equal" if width in train_widths else
                               "smaller" if width < min(train_widths) else
                               "larger" if width > max(train_widths) else "between")
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic count diversity")
    _require(len(shapes) >= 55 and len(layouts) >= 150, "shape/layout diversity")
    _require(len(role_triplets) >= 70 and cross_roles == 42, "palette role diversity")
    _require({"equal", "smaller", "larger"} <= test_relations and source_analog >= 3,
             "test scale regimes")
    _require(train_parities == {0, 1} and rectangles >= 80 and squares >= 5,
             "dimension and parity diversity")
    _require(varied_train_widths >= 15 and varied_train_heights_fixed_width >= 3, "within-puzzle dimension diversity")
    _require(repeated >= 35 and smooth >= 35 and jagged >= 20, "boundary profile diversity")
    _require(omitted_first >= 5 and omitted_opposite >= 5 and ordinary >= 80 and combined >= 5,
             "edge-case distribution and combinations")
    _require(recurring_first_band >= 5 and changing_first_band >= 5
             and familiar_test_outer >= 10 and fresh_test_outer >= 3,
             "palette recurrence and test reassignment")
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "cross-seed puzzle uniqueness")
    a = generate_puzzle(123456, 7, 3)
    b = generate_puzzle(123456, 7, 3)
    _require(_puzzle_key(a) == _puzzle_key(b), "determinism")
    for count_a, count_b in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(20000 + count_a * 10 + count_b, count_a, count_b)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (count_a, count_b), "explicit counts")
    for offset, args in enumerate((
        {"num_train": None, "num_test": 1},
        {"num_train": None, "num_test": 4},
        {"num_train": 2, "num_test": None},
        {"num_train": 10, "num_test": None},
    )):
        puzzle = generate_puzzle(30000 + offset, **args)
        for key, count in (("num_train", len(puzzle["train"])), ("num_test", len(puzzle["test"]))):
            _require(args[key] is None or args[key] == count, "partial counts")
    for args in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0},
        {"num_test": 5}, {"num_train": 2, "num_test": 2},
        {"num_train": 4, "num_test": 4}, {"num_train": True, "num_test": 1},
    ):
        try:
            generate_puzzle(40000, **args)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
