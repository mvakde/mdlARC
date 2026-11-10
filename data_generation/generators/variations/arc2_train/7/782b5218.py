import numpy as np


_NEUTRAL = 0
_SEPARATOR = 2
_FILL_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)

_SOURCE_PAIRS = (
    (
        (
            "1011111110",
            "0010110101",
            "1100111110",
            "2222222222",
            "0011000000",
            "1101000010",
            "0111011010",
            "1100010010",
            "0010001100",
            "0111011001",
        ),
        (
            "0000000000",
            "0000000000",
            "0000000000",
            "2222222222",
            "1111111111",
            "1111111111",
            "1111111111",
            "1111111111",
            "1111111111",
            "1111111111",
        ),
    ),
    (
        (
            "5550500055",
            "5005505055",
            "0550550500",
            "2055205025",
            "5202020202",
            "0025552050",
            "5500055555",
            "0505505055",
            "0550505055",
            "5500555555",
        ),
        (
            "0000000000",
            "0000000000",
            "0000000000",
            "2000200020",
            "5202520252",
            "5525552555",
            "5555555555",
            "5555555555",
            "5555555555",
            "5555555555",
        ),
    ),
    (
        (
            "0880808800",
            "2088080008",
            "2280000000",
            "0220800000",
            "8822000880",
            "8882280000",
            "0000228880",
            "8080022888",
            "8800002280",
            "0808088228",
        ),
        (
            "0000000000",
            "2000000000",
            "2200000000",
            "8220000000",
            "8822000000",
            "8882200000",
            "8888220000",
            "8888822000",
            "8888882200",
            "8888888220",
        ),
    ),
    (
        (
            "9990000000",
            "9990900900",
            "9000909000",
            "0099909000",
            "0222022292",
            "2202229222",
            "9009990990",
            "0000909009",
            "0990000990",
            "9090090900",
        ),
        (
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0222022202",
            "2292229222",
            "9999999999",
            "9999999999",
            "9999999999",
            "9999999999",
        ),
    ),
)


def _checked_optional_count(
    value: int | None,
    name: str,
    low: int,
    high: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train = _checked_optional_count(num_train, "num_train", 2, 10)
    test = _checked_optional_count(num_test, "num_test", 1, 4)

    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must be greater than num_test")
        return train, test

    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test + 1), 11))
        return train, test

    if train is None:
        assert test is not None
        train = int(rng.integers(max(2, test + 1), 11))
        return train, test

    upper_test = min(4, train - 1)
    if upper_test < 1:
        raise ValueError("no legal num_test exists for num_train")
    test = int(rng.integers(1, upper_test + 1))
    return train, test


def _components(
    mask: np.ndarray,
    diagonal: bool = False,
) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    height, width = mask.shape
    labels = np.full((height, width), -1, dtype=np.int16)
    components: list[list[tuple[int, int]]] = []

    if diagonal:
        directions = (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
    else:
        directions = ((-1, 0), (1, 0), (0, -1), (0, 1))

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or labels[row, col] >= 0:
                continue

            label = len(components)
            labels[row, col] = label
            stack = [(row, col)]
            cells: list[tuple[int, int]] = []

            while stack:
                current_row, current_col = stack.pop()
                cells.append((current_row, current_col))

                for delta_row, delta_col in directions:
                    next_row = current_row + delta_row
                    next_col = current_col + delta_col
                    if (
                        0 <= next_row < height
                        and 0 <= next_col < width
                        and mask[next_row, next_col]
                        and labels[next_row, next_col] < 0
                    ):
                        labels[next_row, next_col] = label
                        stack.append((next_row, next_col))

            components.append(cells)

    return labels, components


def _wall_topology(wall: np.ndarray) -> tuple[np.ndarray, int, int] | None:
    height, width = wall.shape
    if wall[0, width - 1] or wall[height - 1, 0]:
        return None
    _, wall_parts = _components(wall, diagonal=True)
    if len(wall_parts) != 1:
        return None
    labels, regions = _components(~wall)
    if len(regions) != 2 or min(map(len, regions)) < max(4, int(0.045 * height * width)):
        return None
    northeast = int(labels[0, width - 1])
    southwest = int(labels[height - 1, 0])
    if northeast < 0 or southwest < 0 or northeast == southwest:
        return None
    edges = sum(bool(np.any(side)) for side in (wall[0], wall[-1], wall[:, 0], wall[:, -1]))
    if edges < 2:
        return None
    return labels, northeast, southwest


def _choose_dimension(rng: np.random.Generator) -> int:
    band = int(rng.choice(3, p=[0.23, 0.52, 0.25]))
    low, high = ((7, 10), (11, 22), (23, 30))[band]
    return int(rng.integers(low, high + 1))


def _horizontal_profile(rng: np.random.Generator, height: int, width: int, family: str) -> list[int]:
    low, high = 2, height - 3
    if family == "straight":
        return [int(rng.integers(low, high + 1))] * width
    if family == "periodic":
        base = int(rng.integers(low, max(low, high - 2) + 1))
        amplitude = min(2, high - base)
        pattern = [base, base + 1, base + amplitude, base + 1]
        phase = int(rng.integers(4))
        return [pattern[(index + phase) % 4] for index in range(width)]
    if family == "fast":
        base = int(rng.integers(low, high))
        profile: list[int] = []
        state = int(rng.integers(2))
        while len(profile) < width:
            profile.extend([base + state] * int(rng.integers(1, 3)))
            state = 1 - state
        return profile[:width]
    if family == "slope":
        start = int(rng.integers(low, high + 1))
        choices = [value for value in range(low, high + 1) if abs(value - start) >= 2]
        if not choices:
            return _horizontal_profile(rng, height, width, "fast")
        end = int(rng.choice(choices))
        return [int(round(start + (end - start) * index / (width - 1))) for index in range(width)]
    if family == "bend":
        base = int(rng.integers(low, high + 1))
        choices = [value for value in range(low, high + 1) if abs(value - base) >= 2]
        if not choices:
            return _horizontal_profile(rng, height, width, "fast")
        peak = int(rng.choice(choices))
        pivot = int(rng.integers(max(2, width // 3), min(width - 1, 2 * width // 3 + 1)))
        return [int(round(base + (peak - base) * (index / pivot if index <= pivot else (width - 1 - index) / (width - 1 - pivot)))) for index in range(width)]
    if family not in ("slow", "irregular"):
        raise ValueError("unknown wall family")
    row = int(rng.integers(low, high + 1))
    profile = []
    while len(profile) < width:
        span = int(rng.integers(3, 8) if family == "slow" else rng.integers(1, 7))
        profile.extend([row] * span)
        choices = [value for value in (row - 1, row + 1) if low <= value <= high]
        row = int(rng.choice(choices))
    return profile[:width]


def _left_bottom_profile(rng: np.random.Generator, height: int, width: int, family: str) -> list[int]:
    start_low = max(1, height - width + 2)
    start_high = height - 3
    start = int(rng.integers(start_low, start_high + 1))
    rises = height - 1 - start
    exit_col = int(rng.integers(rises, width - 1))
    rise_count = rises - 1
    span = exit_col - 1
    if family in ("straight", "slope"):
        increments = [((index + 1) * rise_count // span) - (index * rise_count // span) for index in range(span)]
    elif family in ("fast", "periodic"):
        phase = int(rng.integers(2))
        order = sorted(range(span), key=lambda index: ((index + phase) % 2, index))
        increments = [0] * span
        for index in order[:rise_count]:
            increments[index] = 1
    elif family == "bend":
        order = sorted(range(span), key=lambda index: min(index, span - 1 - index))
        increments = [0] * span
        for index in order[:rise_count]:
            increments[index] = 1
    elif family in ("slow", "irregular"):
        increments = [1] * rise_count + [0] * (span - rise_count)
        rng.shuffle(increments)
        if family == "slow":
            increments.sort(reverse=bool(rng.integers(2)))
    else:
        raise ValueError("unknown wall family")
    profile = [start]
    for step in increments + [1]:
        profile.append(profile[-1] + step)
    return profile


def _wall_from_profile(rng: np.random.Generator, height: int, profile: list[int], thickness: str) -> np.ndarray:
    wall = np.zeros((height, len(profile)), dtype=bool)
    for col, row in enumerate(profile):
        wall[row, col] = True
        if col and row != profile[col - 1]:
            chance = {"thin": 0.0, "sparse": 0.35, "dense": 0.85, "bulge": 0.55}[thickness]
            if rng.random() < chance:
                wall[min(row, profile[col - 1]): max(row, profile[col - 1]) + 1, col] = True
    if thickness == "bulge":
        start = int(rng.integers(1, max(2, len(profile) - 1)))
        length = int(rng.integers(1, min(5, len(profile) - start + 1)))
        sign = int(rng.choice([-1, 1]))
        for col in range(start, min(len(profile), start + length)):
            extra = profile[col] + sign
            if 1 <= extra < height - 1:
                wall[extra, col] = True
    return wall


def _construct_wall(rng: np.random.Generator, height: int, width: int, endpoint: str, family: str, thickness: str) -> np.ndarray:
    for _ in range(48):
        if endpoint in ("left_right", "top_bottom"):
            base_height, base_width = (height, width) if endpoint == "left_right" else (width, height)
            profile = _horizontal_profile(rng, base_height, base_width, family)
            wall = _wall_from_profile(rng, base_height, profile, thickness)
            if endpoint == "top_bottom":
                wall = wall.T.copy()
        elif endpoint in ("left_bottom", "top_right"):
            profile = _left_bottom_profile(rng, height, width, family)
            wall = _wall_from_profile(rng, height, profile, thickness)
            if wall.shape[1] < width:
                wall = np.pad(wall, ((0, 0), (0, width - wall.shape[1])))
            if endpoint == "top_right":
                wall = wall[::-1, ::-1].copy()
        else:
            raise ValueError("unknown wall endpoints")
        if _wall_topology(wall) is not None:
            return wall
    raise RuntimeError("could not construct a valid separator")


def _wall_candidates(inp: np.ndarray) -> list[tuple[int, np.ndarray, int, int]]:
    result = []
    for color in np.unique(inp):
        color = int(color)
        wall = inp == color
        topology = _wall_topology(wall)
        if topology is not None:
            result.append((color, *topology))
    return result


def _noisy_pair(rng: np.random.Generator, wall: np.ndarray, neutral: int, fill: int, separator: int, noise_style: str) -> dict:
    topology = _wall_topology(wall)
    assert topology is not None
    labels, northeast, southwest = topology
    height, width = wall.shape
    if noise_style == "ordinary":
        probabilities = rng.uniform(0.25, 0.75, 2)
    elif noise_style == "opposed":
        probabilities = np.array([rng.uniform(0.68, 0.9), rng.uniform(0.1, 0.32)])
    elif noise_style == "sparse":
        probabilities = rng.uniform(0.03, 0.16, 2)
    elif noise_style == "heavy":
        probabilities = rng.uniform(0.84, 0.97, 2)
    elif noise_style == "asymmetric":
        probabilities = np.array([rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)])
        if abs(float(probabilities[0] - probabilities[1])) < 0.4:
            probabilities[1] = 0.9 if probabilities[0] < 0.5 else 0.1
    elif noise_style == "clustered":
        probabilities = rng.uniform(0.15, 0.85, 2)
    else:
        raise ValueError("unknown noise style")
    if noise_style == "clustered":
        block_h = int(rng.integers(2, 6))
        block_w = int(rng.integers(2, 6))
        coarse_shape = ((height + block_h - 1) // block_h, (width + block_w - 1) // block_w)
        upper = np.repeat(np.repeat(rng.random(coarse_shape) < probabilities[0], block_h, axis=0), block_w, axis=1)[:height, :width]
        lower = np.repeat(np.repeat(rng.random(coarse_shape) < probabilities[1], block_h, axis=0), block_w, axis=1)[:height, :width]
        flips = rng.random((height, width)) < float(rng.uniform(0.04, 0.16))
        fill_mask = np.where(labels == northeast, upper, lower) ^ flips
    else:
        random_field = rng.random((height, width))
        fill_mask = np.where(labels == northeast, random_field < probabilities[0], random_field < probabilities[1])
    inp = np.where(fill_mask, fill, neutral).astype(np.int8)
    inp[wall] = separator
    for label in (northeast, southwest):
        cells = np.argwhere(labels == label)
        chosen = rng.choice(len(cells), size=2, replace=False)
        inp[tuple(cells[int(chosen[0])])] = neutral
        inp[tuple(cells[int(chosen[1])])] = fill
    out = np.full((height, width), neutral, dtype=np.int8)
    out[labels == southwest] = fill
    out[wall] = separator
    return {"input": inp, "output": out}


def _grid_key(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    normalized = np.asarray(grid, dtype=np.int8)
    return tuple(normalized.shape), normalized.tobytes()


def _rows_key(grid: np.ndarray) -> tuple[str, ...]:
    return tuple("".join(str(int(value)) for value in row) for row in grid)


def _is_original_pair(inp: np.ndarray, out: np.ndarray) -> bool:
    return inp.shape == (10, 10) and (_rows_key(inp), _rows_key(out)) in _SOURCE_PAIRS


def _inspect_pair(pair: dict) -> dict:
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = pair["input"], pair["output"]
    for grid in (inp, out):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
        assert np.issubdtype(grid.dtype, np.integer)
        assert np.all((0 <= grid) & (grid <= 9))
    assert inp.shape == out.shape and not np.array_equal(inp, out)
    assert len(np.unique(inp)) == 3 and len(np.unique(out)) == 3
    candidates = _wall_candidates(inp)
    assert len(candidates) == 1
    separator, labels, northeast, southwest = candidates[0]
    wall = inp == separator
    assert np.array_equal(wall, out == separator)
    neutral = int(out[0, -1])
    fill = int(out[-1, 0])
    assert len({neutral, fill, separator}) == 3
    assert set(int(value) for value in np.unique(inp)) == {neutral, fill, separator}
    assert np.all(out[labels == northeast] == neutral)
    assert np.all(out[labels == southwest] == fill)
    assert np.array_equal(inp[wall], out[wall])
    for label in (northeast, southwest):
        assert set(int(value) for value in np.unique(inp[labels == label])) == {neutral, fill}
    return {"neutral": neutral, "fill": fill, "separator": separator, "wall": wall, "labels": labels, "northeast": northeast, "southwest": southwest}


def _predict_from_examples(train_pairs: list[dict], test_input: np.ndarray) -> np.ndarray:
    neutral_values = []
    separator_values = []
    fill_values = []
    for pair in train_pairs:
        info = _inspect_pair(pair)
        neutral_values.append(info["neutral"])
        separator_values.append(info["separator"])
        fill_values.append(info["fill"])
    assert len(set(neutral_values)) == 1 and len(set(fill_values)) >= 2
    candidates = _wall_candidates(test_input)
    assert len(candidates) == 1
    separator, labels, northeast, southwest = candidates[0]
    neutral = neutral_values[0]
    assert separator != neutral
    if len(set(separator_values)) == 1:
        assert separator == separator_values[0]
    fill_candidates = set(int(value) for value in np.unique(test_input)) - {neutral, separator}
    assert len(fill_candidates) == 1
    fill = fill_candidates.pop()
    output = np.full(test_input.shape, neutral, dtype=np.int8)
    output[labels == southwest] = fill
    output[test_input == separator] = separator
    return output


def _sample_palette(rng: np.random.Generator, train_count: int, test_count: int) -> tuple[int, list[int], list[int], list[int], list[int]]:
    neutral = int(rng.integers(10))
    other = [value for value in range(10) if value != neutral]
    rng.shuffle(other)
    variable_separator = bool(rng.integers(2))
    fixed_separator = other[0]
    fill_pool = [value for value in other if variable_separator or value != fixed_separator]
    rng.shuffle(fill_pool)
    train_pool_size = int(rng.integers(2, min(5, len(fill_pool), train_count) + 1))
    train_pool = fill_pool[:train_pool_size]
    train_fills = list(train_pool)
    while len(train_fills) < train_count:
        train_fills.append(int(rng.choice(train_pool)))
    rng.shuffle(train_fills)
    if variable_separator:
        separator_pool = [value for value in other if value not in train_pool]
        rng.shuffle(separator_pool)
        train_separators = [separator_pool[index % min(4, len(separator_pool))] for index in range(train_count)]
        rng.shuffle(train_separators)
    else:
        train_separators = [fixed_separator] * train_count
    test_fills = []
    test_separators = []
    for index in range(test_count):
        reuse = bool(rng.integers(2))
        fresh = [value for value in fill_pool if value not in train_pool]
        fill = int(rng.choice(train_pool if reuse or not fresh else fresh))
        test_fills.append(fill)
        if variable_separator:
            choices = [value for value in other if value != fill]
            test_separators.append(int(rng.choice(choices)))
        else:
            test_separators.append(fixed_separator)
    return neutral, train_fills, train_separators, test_fills, test_separators


def _pair_plan(rng: np.random.Generator, train_count: int, test_count: int) -> tuple[list[tuple[str, str, str, str]], list[tuple[str, str, str, str]]]:
    endpoints = ["left_right", "top_bottom", "left_bottom", "top_right"]
    families = ["straight", "periodic", "fast", "slow", "irregular", "slope", "bend"]
    thicknesses = ["thin", "sparse", "dense", "bulge"]
    noises = ["ordinary", "opposed", "sparse", "heavy", "asymmetric", "clustered"]
    train = []
    simple_family = str(rng.choice(["straight", "periodic", "fast"]))
    train.append(("left_right", simple_family, "thin", "ordinary"))
    second_family = str(rng.choice([family for family in families if family != simple_family]))
    second_endpoint = str(rng.choice(endpoints[1:]))
    train.append((second_endpoint, second_family, str(rng.choice(thicknesses)), "opposed"))
    for _ in range(train_count - 2):
        train.append((str(rng.choice(endpoints, p=[0.48, 0.17, 0.19, 0.16])), str(rng.choice(families)), str(rng.choice(thicknesses)), str(rng.choice(noises))))
    rng.shuffle(train)
    test = []
    for _ in range(test_count):
        endpoint = str(rng.choice(endpoints, p=[0.4, 0.2, 0.2, 0.2]))
        family = str(rng.choice(families, p=[0.09, 0.12, 0.2, 0.12, 0.2, 0.13, 0.14]))
        test.append((endpoint, family, str(rng.choice(thicknesses, p=[0.13, 0.22, 0.35, 0.3])), str(rng.choice(noises))))
    return train, test


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    neutral, train_fills, train_separators, test_fills, test_separators = _sample_palette(rng, train_count, test_count)
    train_plan, test_plan = _pair_plan(rng, train_count, test_count)
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    for split, plan, fills, separators in (("train", train_plan, train_fills, train_separators), ("test", test_plan, test_fills, test_separators)):
        for (endpoint, family, thickness, noise), fill, separator in zip(plan, fills, separators):
            for _ in range(64):
                pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                height = _choose_dimension(pair_rng)
                width = _choose_dimension(pair_rng)
                try:
                    wall = _construct_wall(pair_rng, height, width, endpoint, family, thickness)
                except RuntimeError:
                    continue
                pair = _noisy_pair(pair_rng, wall, neutral, fill, separator, noise)
                key = _grid_key(pair["input"])
                if key in seen_inputs or _is_original_pair(pair["input"], pair["output"]):
                    continue
                try:
                    _inspect_pair(pair)
                except AssertionError:
                    continue
                break
            else:
                raise RuntimeError("could not construct a unique valid pair")
            seen_inputs.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _inspect_pair(pair)
            key = _grid_key(pair["input"])
            assert key not in seen
            seen.add(key)
            assert not _is_original_pair(pair["input"], pair["output"])
            infos[split].append(info)
    neutral_values = {info["neutral"] for split in infos for info in infos[split]}
    assert len(neutral_values) == 1
    assert len({info["fill"] for info in infos["train"]}) >= 2
    train_separators = {info["separator"] for info in infos["train"]}
    if len(train_separators) == 1:
        assert all(info["separator"] in train_separators for info in infos["test"])
    # A simple left-to-right demonstration anchors the corner convention.
    assert any(np.all(np.any(info["wall"], axis=0)) and not np.any(info["wall"][0]) and not np.any(info["wall"][-1]) and np.count_nonzero(np.any(info["wall"], axis=1)) <= 3 for info in infos["train"])
    assert len({_grid_key(info["wall"].astype(np.int8)) for info in infos["train"]}) >= 2
    for pair in puzzle["test"]:
        prediction = _predict_from_examples(puzzle["train"], pair["input"].copy())
        assert np.array_equal(prediction, pair["output"])


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[split]) == len(second[split]) and all(np.array_equal(a[role], b[role]) for a, b in zip(first[split], second[split]) for role in ("input", "output")) for split in ("train", "test"))


def _endpoint(wall: np.ndarray) -> str:
    top, bottom, left, right = (bool(np.any(edge)) for edge in (wall[0], wall[-1], wall[:, 0], wall[:, -1]))
    if left and right and not top and not bottom:
        return "left_right"
    if top and bottom and not left and not right:
        return "top_bottom"
    if left and bottom and not top and not right:
        return "left_bottom"
    if top and right and not left and not bottom:
        return "top_right"
    return "other"


def self_test() -> None:
    train_counts, test_counts = set(), set()
    dimensions, neutrals, separators, endpoints = set(), set(), set(), set()
    thickness_count = 0
    thin_count = 0
    sparse_double_count = 0
    frequent_double_count = 0
    straight_count = 0
    fast_count = 0
    slow_count = 0
    periodic_count = 0
    reversal_count = 0
    short_run_count = 0
    long_run_count = 0
    narrow_count = 0
    large_count = 0
    tall_count = 0
    wide_count = 0
    sparse_count = 0
    heavy_count = 0
    asymmetric_count = 0
    opposed_count = 0
    clustered_count = 0
    reused_count = 0
    novel_count = 0
    varying_separator_puzzles = 0
    fixed_separator_puzzles = 0
    unique = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        unique.add(tuple((_grid_key(pair["input"]), _grid_key(pair["output"])) for split in ("train", "test") for pair in puzzle[split]))
        train_info = [_inspect_pair(pair) for pair in puzzle["train"]]
        assert len({_endpoint(info["wall"]) for info in train_info}) >= 2
        assert len({pair["input"].shape for pair in puzzle["train"] + puzzle["test"]}) >= 2
        train_fills = {info["fill"] for info in train_info}
        if len({info["separator"] for info in train_info}) > 1:
            varying_separator_puzzles += 1
        else:
            fixed_separator_puzzles += 1
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _inspect_pair(pair)
                wall = info["wall"]
                height, width = wall.shape
                dimensions.add((height, width))
                neutrals.add(info["neutral"])
                separators.add(info["separator"])
                endpoints.add(_endpoint(wall))
                if min(height, width) <= 10:
                    narrow_count += 1
                if max(height, width) >= 27:
                    large_count += 1
                if height >= 2 * width:
                    tall_count += 1
                if width >= 2 * height:
                    wide_count += 1
                endpoint = _endpoint(wall)
                axis = wall.T if endpoint == "top_bottom" else (wall[::-1, ::-1] if endpoint == "top_right" else wall)
                axis = axis[:, np.any(axis, axis=0)]
                double_fraction = float(np.mean(np.sum(axis, axis=0) > 1))
                if double_fraction > 0:
                    thickness_count += 1
                else:
                    thin_count += 1
                if 0 < double_fraction < 0.2:
                    sparse_double_count += 1
                if double_fraction >= 0.35:
                    frequent_double_count += 1
                if _endpoint(wall) == "left_right":
                    rows = np.flatnonzero(np.any(wall, axis=1))
                    if len(rows) == 1:
                        straight_count += 1
                    profile = [int(np.flatnonzero(wall[:, col])[0]) for col in range(width)]
                    changes = sum(a != b for a, b in zip(profile, profile[1:]))
                    if changes >= 0.35 * (width - 1):
                        fast_count += 1
                    if changes <= 0.2 * (width - 1):
                        slow_count += 1
                    if width >= 9 and all(profile[index] == profile[index + 4] for index in range(width - 4)) and len(set(profile)) >= 2:
                        periodic_count += 1
                    run_lengths = []
                    run_start = 0
                    for index in range(1, len(profile) + 1):
                        if index == len(profile) or profile[index] != profile[run_start]:
                            run_lengths.append(index - run_start)
                            run_start = index
                    short_run_count += int(1 in run_lengths)
                    long_run_count += int(any(length >= 3 for length in run_lengths))
                    steps = [b - a for a, b in zip(profile, profile[1:]) if a != b]
                    reversal_count += int(any(a * b < 0 for a, b in zip(steps, steps[1:])))
                labels = info["labels"]
                upper = pair["input"][labels == info["northeast"]]
                lower = pair["input"][labels == info["southwest"]]
                p_upper = float(np.mean(upper == info["fill"]))
                p_lower = float(np.mean(lower == info["fill"]))
                if p_upper < 0.2 and p_lower < 0.2:
                    sparse_count += 1
                if p_upper > 0.8 and p_lower > 0.8:
                    heavy_count += 1
                if abs(p_upper - p_lower) > 0.4:
                    asymmetric_count += 1
                if p_upper > 0.55 and p_lower < 0.45:
                    opposed_count += 1
                adjacent_total = 0
                adjacent_excess = 0.0
                for label, probability in ((info["northeast"], p_upper), (info["southwest"], p_lower)):
                    region = labels == label
                    for first, second, both in ((pair["input"][:, :-1], pair["input"][:, 1:], region[:, :-1] & region[:, 1:]), (pair["input"][:-1], pair["input"][1:], region[:-1] & region[1:])):
                        count = int(np.sum(both))
                        adjacent_total += count
                        adjacent_excess += float(np.sum((first == second) & both)) - count * (probability ** 2 + (1 - probability) ** 2)
                if adjacent_excess / max(1, adjacent_total) > 0.15:
                    clustered_count += 1
                if split == "test":
                    if info["fill"] in train_fills:
                        reused_count += 1
                    else:
                        novel_count += 1
                    assert np.array_equal(_predict_from_examples(puzzle["train"], pair["input"].copy()), pair["output"])
    assert len(unique) >= 40 and len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(dimensions) >= 90 and len(neutrals) >= 7 and len(separators) >= 7
    assert endpoints == {"left_right", "top_bottom", "left_bottom", "top_right"}
    assert min(thickness_count, thin_count, sparse_double_count, frequent_double_count, straight_count, fast_count, slow_count, periodic_count, reversal_count, short_run_count, long_run_count) >= 5
    assert min(narrow_count, large_count, tall_count, wide_count) >= 5
    assert min(sparse_count, heavy_count, asymmetric_count, opposed_count, clustered_count) >= 3
    assert min(reused_count, novel_count, varying_separator_puzzles, fixed_separator_puzzles) >= 5
    for seed, train_count, test_count in ((101, 3, 1), (102, 6, 2), (103, 9, 4), (7003, 2, 1), (7011, 10, 1), (7014, 10, 4)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        validate_generated_puzzle(puzzle)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
    for seed, train_count, test_count in ((8101, None, 1), (8102, None, 4), (8103, 2, None), (8104, 10, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        validate_generated_puzzle(puzzle)
        assert train_count is None or len(puzzle["train"]) == train_count
        assert test_count is None or len(puzzle["test"]) == test_count
    for train_count, test_count in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(999, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count combination accepted")
    assert _puzzles_equal(generate_puzzle(918273, 7, 3), generate_puzzle(918273, 7, 3))
    assert not _puzzles_equal(generate_puzzle(918273, 7, 3), generate_puzzle(918274, 7, 3))
    assert _puzzles_equal(generate_puzzle(192837), generate_puzzle(192837))
