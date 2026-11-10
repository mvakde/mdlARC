import numpy as np


_TRAIN_LOW_HEIGHTS = (9, 10, 12, 13)
_TRAIN_HIGH_HEIGHTS = (24, 25, 27, 28)
_TARGET_HEIGHTS = (14, 17, 20, 23)

_TRAIN_LOW_WIDTHS = (8, 9, 10, 11, 12, 13)
_TRAIN_HIGH_WIDTHS = (21, 22, 23, 24, 25, 26, 27, 28)
_TARGET_WIDTHS = (14, 15, 16, 17, 18, 19, 20)


_SOURCE_EXAMPLES = (
    (
        (
            "000000000000",
            "000000000000",
            "000000000000",
            "202020202020",
            "222222222222",
            "202020202020",
            "000000000000",
            "000000000000",
            "000000000000",
            "000000000000",
        ),
        (
            "202020202020",
            "222222222222",
            "202020202020",
            "202020202020",
            "222222222222",
            "202020202020",
            "202020202020",
            "222222222222",
            "202020202020",
            "202020202020",
        ),
    ),
    (
        (
            "0000000000000000000",
            "0000000000000000000",
            "0000000000000000000",
            "0000000000000000000",
            "0080000080000080000",
            "8888888888888888888",
            "0000080000080000080",
            "0000000000000000000",
            "0000000000000000000",
            "0000000000000000000",
            "0000000000000000000",
            "0000000000000000000",
            "0000000000000000000",
            "0000000000000000000",
            "0000000000000000000",
        ),
        (
            "0000080000080000080",
            "0080000080000080000",
            "8888888888888888888",
            "0000080000080000080",
            "0080000080000080000",
            "8888888888888888888",
            "0000080000080000080",
            "0080000080000080000",
            "8888888888888888888",
            "0000080000080000080",
            "0080000080000080000",
            "8888888888888888888",
            "0000080000080000080",
            "0080000080000080000",
            "8888888888888888888",
        ),
    ),
    (
        (
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0100100100100",
            "1011011011011",
            "1011011011011",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
            "0000000000000",
        ),
        (
            "0100100100100",
            "1011011011011",
            "1011011011011",
            "0100100100100",
            "1011011011011",
            "1011011011011",
            "0100100100100",
            "1011011011011",
            "1011011011011",
            "0100100100100",
            "1011011011011",
            "1011011011011",
            "0100100100100",
            "1011011011011",
        ),
    ),
)


def _grid_from_rows(rows: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


def _array_key(array: np.ndarray) -> tuple:
    normalized = np.asarray(array, dtype=np.int8)
    return tuple(normalized.shape), normalized.tobytes()


def _pair_key(input_grid: np.ndarray, output_grid: np.ndarray) -> tuple:
    return _array_key(input_grid), _array_key(output_grid)


def _is_source_duplicate(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    for source_input_rows, source_output_rows in _SOURCE_EXAMPLES:
        source_input = _grid_from_rows(source_input_rows)
        source_output = _grid_from_rows(source_output_rows)
        if (
            input_grid.shape == source_input.shape
            and output_grid.shape == source_output.shape
            and np.array_equal(input_grid, source_input)
            and np.array_equal(output_grid, source_output)
        ):
            return True
    return False


def _validate_optional_count(
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


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _validate_optional_count(
        "num_train", num_train, 2, 10
    )
    test_count = _validate_optional_count(
        "num_test", num_test, 1, 4
    )

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count

    if train_count is None and test_count is None:
        test_count = int(rng.choice((1, 2, 3, 4), p=(0.62, 0.24, 0.10, 0.04)))
        choices = np.arange(test_count + 1, 11)
        weights = np.array([0.53 ** (int(value) - 2) for value in choices])
        weights /= weights.sum()
        train_count = int(rng.choice(choices, p=weights))
        return train_count, test_count

    if train_count is None:
        assert test_count is not None
        train_count = int(rng.integers(test_count + 1, 11))
        return train_count, test_count

    maximum_test = min(4, train_count - 1)
    test_count = int(rng.integers(1, maximum_test + 1))
    return train_count, test_count


_RELATIONS = ("ABC", "ABA", "ABB", "AAB", "AAA")
_RELATION_ROWS = {
    "ABC": (0, 1, 2),
    "ABA": (0, 1, 0),
    "ABB": (0, 1, 1),
    "AAB": (0, 0, 1),
    "AAA": (0, 0, 0),
}


def _relation(motif: np.ndarray) -> str:
    equal_01 = np.array_equal(motif[0], motif[1])
    equal_02 = np.array_equal(motif[0], motif[2])
    equal_12 = np.array_equal(motif[1], motif[2])
    if equal_01 and equal_02:
        return "AAA"
    if equal_01:
        return "AAB"
    if equal_02:
        return "ABA"
    if equal_12:
        return "ABB"
    return "ABC"


def _row_bits(rng: np.random.Generator, width: int, period: int | None,
              style: str, used: set[bytes]) -> np.ndarray:
    for _ in range(64):
        if period is None:
            density = float(rng.uniform(0.22, 0.79))
            bits = (rng.random(width) < density).astype(np.uint8)
            if style == "runs":
                bits[:] = 0
                for _ in range(int(rng.integers(2, 6))):
                    left = int(rng.integers(0, width))
                    length = int(rng.integers(1, max(3, width // 3 + 1)))
                    bits[left:min(width, left + length)] = 1
        else:
            if style == "bar":
                tile = np.ones(period, dtype=np.uint8)
            elif style == "sparse":
                tile = np.zeros(period, dtype=np.uint8)
                tile[int(rng.integers(0, period))] = 1
            elif style == "dense":
                tile = np.ones(period, dtype=np.uint8)
                tile[int(rng.integers(0, period))] = 0
            elif style == "alternating":
                tile = (np.arange(period) % 2 == int(rng.integers(0, 2))).astype(np.uint8)
            else:
                tile = (rng.random(period) < float(rng.uniform(0.28, 0.74))).astype(np.uint8)
            bits = np.tile(tile, (width + period - 1) // period)[:width].copy()
        if np.any(bits) and bits.tobytes() not in used:
            return bits
    candidates = []
    unit = width if period is None else period
    for mask in range(1, 1 << unit):
        if period is None and mask > (1 << min(width, 10)):
            break
        tile = np.array([(mask >> column) & 1 for column in range(unit)], dtype=np.uint8)
        bits = np.tile(tile, (width + unit - 1) // unit)[:width].copy()
        if bits.tobytes() not in used:
            candidates.append(bits)
        if len(candidates) >= 16:
            break
    if not candidates:
        raise RuntimeError("Cannot make a distinct row")
    return candidates[int(rng.integers(0, len(candidates)))]


def _make_motif(rng: np.random.Generator, width: int, background: int,
                colors: tuple[int, ...], relation: str, rhythmic: bool) -> np.ndarray:
    period = int(rng.integers(2, min(6, width // 2) + 1)) if rhythmic else None
    style_order = tuple(rng.permutation(("sparse", "bar", "dense", "alternating", "random")))
    if rhythmic and int(rng.integers(0, 3)) == 0:
        style_order = ("sparse", "bar", "dense", "alternating", "random")
    used: set[bytes] = set()
    unique_count = max(_RELATION_ROWS[relation]) + 1
    base_rows = []
    for index in range(unique_count):
        style = style_order[index] if rhythmic else ("runs" if index == 0 and int(rng.integers(0, 2)) else "random")
        bits = _row_bits(rng, width, period, style, used)
        used.add(bits.tobytes())
        base_rows.append(bits)

    rows = []
    for index, bits in enumerate(base_rows):
        row = np.full(width, background, dtype=np.int8)
        active = np.flatnonzero(bits)
        if len(colors) == 1 or unique_count > 1:
            row[active] = colors[index % len(colors)]
        else:
            if active.size < 2:
                active = np.array([0, width - 1])
            for offset, column in enumerate(active):
                row[column] = colors[offset % len(colors)]
        rows.append(row)
    if not any(np.any(row == background) for row in rows):
        rows[0][0] = background
    return np.stack([rows[index] for index in _RELATION_ROWS[relation]])


def _choose_background(rng: np.random.Generator, excluded: set[int]) -> int:
    choices = [color for color in range(10) if color not in excluded]
    return int(rng.choice(choices))


def _make_pair(rng: np.random.Generator, relation: str, rhythmic: bool,
               multicolor: bool, background: int, placement: str) -> tuple[np.ndarray, np.ndarray]:
    height = int(rng.integers(8, 27))
    width = int(rng.integers(6, 28))
    if placement == "top":
        start = 0
    elif placement == "bottom":
        start = height - 3
    else:
        start = int(rng.integers(1, height - 3))
    available = [color for color in range(10) if color != background]
    rng.shuffle(available)
    number_colors = (int(rng.integers(2, 4)) if relation == "ABC" else 2) if multicolor else 1
    colors = tuple(available[:number_colors])
    motif = _make_motif(rng, width, background, colors, relation, rhythmic)
    input_grid = np.full((height, width), background, dtype=np.int8)
    input_grid[start:start + 3] = motif
    output_grid = motif[(np.arange(height) - start) % 3].copy()
    return input_grid, output_grid


def _metadata(input_grid: np.ndarray, output_grid: np.ndarray) -> dict:
    height, width = input_grid.shape
    candidates = []
    for background in np.unique(input_grid):
        nonblank = np.flatnonzero(np.any(input_grid != background, axis=1))
        if nonblank.size == 3 and np.array_equal(nonblank, np.arange(nonblank[0], nonblank[0] + 3)):
            candidates.append((int(background), int(nonblank[0])))
    assert len(candidates) == 1
    background, start = candidates[0]
    motif = input_grid[start:start + 3]
    assert np.all(np.any(motif != background, axis=1))
    assert height >= 8 and 6 <= width <= 27
    assert np.array_equal(output_grid, motif[(np.arange(height) - start) % 3])
    assert not np.array_equal(input_grid, output_grid)
    assert set(np.unique(output_grid)) == set(np.unique(input_grid))
    relation = _relation(motif)
    assert relation in _RELATIONS
    return {
        "height": height, "width": width, "background": background,
        "start": start, "relation": relation, "motif": motif,
        "colors": len(set(np.unique(motif)) - {background}),
        "rhythmic": _common_period(motif, background) is not None,
    }


def _common_period(motif: np.ndarray, background: int) -> int | None:
    width = motif.shape[1]
    masks = motif != background
    for period in range(1, min(6, width // 2) + 1):
        if all(np.array_equal(row[:-period], row[period:]) for row in masks):
            return period
    return None


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    # The sole learned convention is the fixed three-row vertical period.
    # A multicolor case is taught before it can appear in testing.
    multicolor_family = bool(rng.random() < 0.42)
    train_relations = ["ABC", str(rng.choice(("ABA", "ABB", "AAB", "AAA"), p=(.38, .30, .25, .07)))]
    train_relations.extend(str(rng.choice(_RELATIONS, p=(.32, .24, .22, .17, .05))) for _ in range(train_count - 2))
    train_specs = []
    for index, relation in enumerate(train_relations):
        if index == 0:
            placement = "interior"
            rhythmic = True
        else:
            placement = str(rng.choice(("interior", "top", "bottom"), p=(.56, .22, .22)))
            rhythmic = bool(rng.random() < .76)
        train_specs.append((relation, rhythmic, placement,
                            multicolor_family and (index == 0 or rng.random() < .40)))
    rng.shuffle(train_specs)
    test_specs = [
        (str(rng.choice(_RELATIONS, p=(.27, .23, .22, .21, .07))),
         bool(rng.random() < .72),
         str(rng.choice(("interior", "top", "bottom"), p=(.53, .235, .235))),
         multicolor_family and bool(rng.random() < .56))
        for _ in range(test_count)
    ]
    puzzle = {"train": [], "test": []}
    seen: set[tuple] = set()
    previous_backgrounds: set[int] = set()
    for section, specs in (("train", train_specs), ("test", test_specs)):
        for relation, rhythmic, placement, multicolor in specs:
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            for attempt in range(64):
                excluded = previous_backgrounds if section == "train" and len(previous_backgrounds) < 2 else set()
                background = _choose_background(pair_rng, excluded)
                pair = _make_pair(pair_rng, relation, rhythmic, multicolor, background, placement)
                key = _pair_key(*pair)
                if key not in seen and not _is_source_duplicate(*pair):
                    seen.add(key)
                    puzzle[section].append({"input": pair[0], "output": pair[1]})
                    if section == "train":
                        previous_backgrounds.add(background)
                    break
            else:
                raise RuntimeError("Unable to construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    seen_inputs: set[tuple] = set()
    seen_pairs: set[tuple] = set()
    metadata = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            input_grid, output_grid = pair["input"], pair["output"]
            for grid in (input_grid, output_grid):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((grid >= 0) & (grid <= 9))
            assert input_grid.shape == output_grid.shape
            input_key, pair_key = _array_key(input_grid), _pair_key(input_grid, output_grid)
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert not _is_source_duplicate(input_grid, output_grid)
            metadata[section].append(_metadata(input_grid, output_grid))
    assert any(item["relation"] == "ABC" for item in metadata["train"])
    assert any(0 < item["start"] < item["height"] - 3 for item in metadata["train"])
    if any(item["colors"] >= 2 for item in metadata["test"]):
        assert any(item["colors"] >= 2 for item in metadata["train"])


def _signature(puzzle: dict) -> tuple:
    return tuple((section, tuple(_pair_key(pair["input"], pair["output"])
                                 for pair in puzzle[section])) for section in ("train", "test"))


def _predict_from_input(input_grid: np.ndarray) -> np.ndarray:
    candidates = []
    for background in np.unique(input_grid):
        rows = np.flatnonzero(np.any(input_grid != background, axis=1))
        if rows.size == 3 and np.array_equal(rows, np.arange(rows[0], rows[0] + 3)):
            candidates.append(int(rows[0]))
    assert len(candidates) == 1
    start = candidates[0]
    return np.stack([input_grid[start + (row - start) % 3]
                     for row in range(input_grid.shape[0])])


def self_test() -> None:
    train_counts, test_counts = set(), set()
    signatures, shapes, layouts, patterns = set(), set(), set(), set()
    train_relations, test_relations = set(), set()
    backgrounds, foregrounds = set(), set()
    rhythmic, irregular, edge, interior, multicolor = 0, 0, 0, 0, 0
    compact, varying_background = 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(_signature(puzzle))
        compact += len(puzzle["train"]) <= 4 and len(puzzle["test"]) == 1
        train_backgrounds = set()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                input_grid, output_grid = pair["input"], pair["output"]
                assert np.array_equal(_predict_from_input(input_grid), output_grid)
                data = _metadata(input_grid, output_grid)
                shapes.add((data["height"], data["width"]))
                layouts.add((data["height"], data["width"], data["start"], data["relation"]))
                patterns.add((data["width"], (data["motif"] != data["background"]).tobytes()))
                backgrounds.add(data["background"])
                foregrounds.update(int(color) for color in np.unique(data["motif"]) if color != data["background"])
                rhythmic += data["rhythmic"]
                irregular += not data["rhythmic"]
                edge += data["start"] in (0, data["height"] - 3)
                interior += 0 < data["start"] < data["height"] - 3
                multicolor += data["colors"] >= 2
                if section == "train":
                    train_relations.add(data["relation"])
                    train_backgrounds.add(data["background"])
                else:
                    test_relations.add(data["relation"])
        varying_background += len(train_backgrounds) >= 2
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) == 42 and len(shapes) >= 90 and len(layouts) >= 150
    assert len(patterns) >= 150
    assert train_relations == set(_RELATIONS) and test_relations == set(_RELATIONS)
    assert backgrounds == set(range(10)) and foregrounds == set(range(10))
    assert rhythmic >= 100 and irregular >= 20 and edge >= 20 and interior >= 80
    assert multicolor >= 15 and compact >= 10 and varying_background == 42
    for seed, train_count, test_count in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        assert len(puzzle["train"]) == train_count and len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)
    for seed, train_count, test_count in ((2001, None, 1), (2002, None, 4),
                                          (2003, 2, None), (2004, 10, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)
    assert _signature(generate_puzzle(314159)) == _signature(generate_puzzle(314159))
    assert _signature(generate_puzzle(271828, 7, 3)) == _signature(generate_puzzle(271828, 7, 3))
    assert _signature(generate_puzzle(98765, 6, 2)) != _signature(generate_puzzle(98766, 6, 2))
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}, {"num_train": 4, "num_test": 4},
                      {"num_train": True, "num_test": 1},
                      {"num_train": 3.0, "num_test": 1}):
        try:
            generate_puzzle(999, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {arguments}")
