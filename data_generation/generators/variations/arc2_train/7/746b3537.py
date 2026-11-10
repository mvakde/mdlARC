import numpy as np


_SOURCE = (
    ("22/66/88/88", "2/6/8"),
    ("346/346/346", "346"),
    ("23381/23381/23381", "2381"),
    ("4444/4444/2222/2222/8888/3333", "4/2/8/3"),
    ("111/222/111", "1/2/1"),
    ("112333884/112333884/112333884/112333884", "12384"),
)


def _key(grid: np.ndarray) -> tuple:
    return tuple(grid.shape), tuple(int(v) for v in grid.flat)


def _source_keys() -> set:
    def grid(text: str) -> np.ndarray:
        return np.asarray([[int(v) for v in row] for row in text.split("/")], dtype=np.int8)
    return {(_key(grid(inp)), _key(grid(out))) for inp, out in _SOURCE}


def _counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= value <= high):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _runs(values: np.ndarray) -> tuple[tuple[int, ...], tuple[int, ...]]:
    colors = [int(values[0])]
    widths = [1]
    for raw in values[1:]:
        color = int(raw)
        if color == colors[-1]:
            widths[-1] += 1
        else:
            colors.append(color)
            widths.append(1)
    return tuple(colors), tuple(widths)


def _derive_output(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    rows_repeat = bool(np.all(grid == grid[:1, :]))
    columns_repeat = bool(np.all(grid == grid[:, :1]))
    assert rows_repeat != columns_repeat, "input must have one changing axis"
    if rows_repeat:
        colors, thicknesses = _runs(grid[0, :])
        output = np.asarray(colors, dtype=np.int8).reshape(1, -1)
        orientation = "vertical"
        perpendicular = grid.shape[0]
    else:
        colors, thicknesses = _runs(grid[:, 0])
        output = np.asarray(colors, dtype=np.int8).reshape(-1, 1)
        orientation = "horizontal"
        perpendicular = grid.shape[1]
    assert len(colors) >= 2
    return output, {"orientation": orientation, "colors": colors,
                    "thicknesses": thicknesses, "perpendicular": perpendicular}


def _make_input(colors: tuple[int, ...], thicknesses: tuple[int, ...],
                orientation: str, perpendicular: int) -> np.ndarray:
    expanded = np.repeat(np.asarray(colors, dtype=np.int8), thicknesses)
    if orientation == "vertical":
        return np.tile(expanded.reshape(1, -1), (perpendicular, 1))
    return np.tile(expanded.reshape(-1, 1), (1, perpendicular))


def _run_count(rng: np.random.Generator, regime: str) -> int:
    if regime == "source_train":
        interval = ((2, 4), (5, 8))[int(rng.choice(2, p=(0.58, 0.42)))]
    else:
        interval = ((2, 4), (5, 8), (9, 12))[int(rng.choice(3, p=(0.32, 0.40, 0.28)))]
    return int(rng.integers(interval[0], interval[1] + 1))


def _distinct_count(rng: np.random.Generator, count: int, recurrence: bool) -> int:
    maximum = min(count - int(recurrence), 10)
    if maximum == 2:
        return 2
    mode = int(rng.integers(0, 4))
    if mode == 0:
        return 2
    if mode == 1:
        return maximum
    return int(rng.integers(2, maximum + 1))


def _palette(rng: np.random.Generator, count: int, train_colors: set[int] | None,
             relation: str) -> tuple[int, ...]:
    all_colors = list(range(10))
    if relation == "subset" and train_colors is not None and len(train_colors) >= count:
        pool = sorted(train_colors)
        return tuple(int(v) for v in rng.choice(pool, size=count, replace=False))
    if relation == "new" and train_colors is not None and len(train_colors) < 10:
        unseen = sorted(set(all_colors) - train_colors)
        new_color = int(rng.choice(unseen))
        others = [v for v in all_colors if v != new_color]
        values = [new_color] + [int(v) for v in rng.choice(others, size=count - 1, replace=False)]
        return tuple(values)
    return tuple(int(v) for v in rng.choice(all_colors, size=count, replace=False))


def _sequence(rng: np.random.Generator, palette: tuple[int, ...], count: int) -> tuple[int, ...]:
    sequence = [int(v) for v in rng.permutation(palette)]
    while len(sequence) < count:
        choices = [v for v in palette if v != sequence[-1]]
        sequence.append(int(rng.choice(choices)))
    return tuple(sequence)


def _thicknesses(rng: np.random.Generator, count: int, force_thick: bool) -> tuple[int, ...]:
    mode = int(rng.integers(0, 4))
    if mode == 0:
        values = np.ones(count, dtype=np.int64)
    elif mode == 1:
        width = int(rng.integers(2, min(7, 30 // count) + 1))
        values = np.full(count, width, dtype=np.int64)
    elif mode == 2:
        base = int(rng.integers(1, min(2, 30 // count) + 1))
        values = np.full(count, base, dtype=np.int64)
        index = int(rng.integers(0, count))
        limit = min(10, 30 - base * (count - 1))
        values[index] = int(rng.integers(base + 1, limit + 1)) if limit > base else base
    else:
        values = rng.integers(1, 9, size=count, dtype=np.int64)
        while int(values.sum()) > 30:
            eligible = np.flatnonzero(values > 1)
            index = int(rng.choice(eligible))
            values[index] -= 1
    if force_thick and bool(np.all(values == values[0])):
        index = int(rng.integers(0, count))
        values[index] = 2 if int(values[0]) == 1 else 1
    return tuple(int(v) for v in values)


def _perpendicular(rng: np.random.Generator, changing_length: int,
                   regime: str) -> int:
    mode = int(rng.integers(0, 10))
    if mode == 0:
        return changing_length  # exact squares are rare under independent dimensions
    if mode <= 2:
        return int(rng.integers(1, 4))
    if mode <= 5:
        return int(rng.integers(4, 10))
    if mode <= 8:
        return int(rng.integers(10, 21))
    return int(rng.integers(21, 31))


def _candidate(rng: np.random.Generator, orientation: str, regime: str,
               force_recurrence: bool, force_thick: bool, force_repeat: bool,
               train_colors: set[int] | None, relation: str) -> dict:
    if regime == "target_test" and rng.random() < 0.15:
        # Compose recurrence, thin stripes and a long cross-section in either axis.
        count = int(rng.integers(6, 12))
        distinct = int(rng.integers(2, min(4, count - 1) + 1))
        palette = _palette(rng, distinct, train_colors, relation)
        colors = _sequence(rng, palette, count)
        thicknesses = tuple(int(v) for v in rng.integers(1, 3, size=count))
        perpendicular = int(rng.integers(sum(thicknesses) + 1, 31))
    else:
        count = _run_count(rng, regime)
        if force_recurrence:
            count = max(count, 3)
        distinct = _distinct_count(rng, count, force_recurrence)
        palette = _palette(rng, distinct, train_colors, relation)
        colors = _sequence(rng, palette, count)
        thicknesses = _thicknesses(rng, count, force_thick)
        perpendicular = _perpendicular(rng, sum(thicknesses), regime)
    if force_repeat and perpendicular == 1:
        perpendicular = 2
    if perpendicular == 1 and all(v == 1 for v in thicknesses):
        values = list(thicknesses)
        values[int(rng.integers(0, count))] = 2
        thicknesses = tuple(values)
    inp = _make_input(colors, thicknesses, orientation, perpendicular)
    out, _ = _derive_output(inp)
    return {"input": inp, "output": out}


def _unique_pair(rng: np.random.Generator, seen: set, orientation: str,
                 regime: str, force_recurrence: bool, force_thick: bool,
                 force_repeat: bool,
                 train_colors: set[int] | None, relation: str) -> dict:
    sources = _source_keys()
    for _ in range(80):
        pair = _candidate(rng, orientation, regime, force_recurrence,
                          force_thick, force_repeat, train_colors, relation)
        key = (_key(pair["input"]), _key(pair["output"]))
        if key[0] not in seen and key not in sources:
            seen.add(key[0])
            return pair
    # A bounded deterministic fallback varies both color labels and cross-section.
    for perpendicular in range(2, 31):
        for shift in range(10):
            colors = ((shift, (shift + 1) % 10, shift) if force_recurrence
                      else (shift, (shift + 1) % 10, (shift + 2) % 10))
            inp = _make_input(colors, (2, 1, 3), orientation, perpendicular)
            out, _ = _derive_output(inp)
            key = (_key(inp), _key(out))
            if key[0] not in seen and key not in sources:
                seen.add(key[0])
                return {"input": inp, "output": out}
    raise RuntimeError("unique stripe pair could not be constructed")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(seed, num_train, num_test)
    # No color or geometry convention carries across pairs in this source family.
    # The only puzzle-level choice is which stripe orientations to demonstrate.
    mode = int(rng.choice(3, p=(0.72, 0.14, 0.14)))
    if mode == 0:
        orientations = ["vertical", "horizontal"]
        orientations += ["vertical" if rng.integers(0, 2) else "horizontal"
                         for _ in range(num_train - 2)]
        rng.shuffle(orientations)
    else:
        orientations = ["vertical" if mode == 1 else "horizontal"] * num_train
    recurrence_demo = int(rng.integers(0, num_train))
    thick_demo = int(rng.integers(0, num_train))
    repeated_demo = int(rng.integers(0, num_train))
    seen: set = set()
    train = []
    train_colors: set[int] = set()
    for index, orientation in enumerate(orientations):
        pair = _unique_pair(rng, seen, orientation, "source_train",
                            index == recurrence_demo, index == thick_demo,
                            index == repeated_demo,
                            None, "free")
        train.append(pair)
        train_colors.update(int(v) for v in pair["input"].flat)
    test = []
    for _ in range(num_test):
        orientation = orientations[int(rng.integers(0, num_train))]
        relation = ("free", "subset", "new")[int(rng.integers(0, 3))]
        test.append(_unique_pair(rng, seen, orientation, "target_test", False,
                                 False, False, train_colors, relation))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    seen_pairs = set()
    sources = _source_keys()
    info = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dimension <= 30 for dimension in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((grid >= 0) & (grid <= 9)))
            inp, out = pair["input"], pair["output"]
            pair_key = (_key(inp), _key(out))
            assert pair_key[0] not in seen_inputs and pair_key not in seen_pairs
            assert pair_key not in sources
            seen_inputs.add(pair_key[0])
            seen_pairs.add(pair_key)
            expected, features = _derive_output(inp)
            assert np.array_equal(out, expected)
            assert not np.array_equal(inp, out)
            assert 2 <= len(features["colors"]) <= (8 if split == "train" else 12)
            assert sum(features["thicknesses"]) <= 30
            assert len(features["colors"]) == len(features["thicknesses"])
            info[split].append(features)
    train_axes = {item["orientation"] for item in info["train"]}
    assert {item["orientation"] for item in info["test"]} <= train_axes
    assert any(len(set(item["colors"])) < len(item["colors"]) for item in info["train"])
    assert any(len(set(item["thicknesses"])) > 1 for item in info["train"])
    assert any(item["perpendicular"] > 1 for item in info["train"])


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(tuple((_key(pair["input"]), _key(pair["output"]))
                       for pair in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    assert _puzzle_key(generate_puzzle(12345, 7, 3)) == _puzzle_key(generate_puzzle(12345, 7, 3))
    assert _puzzle_key(generate_puzzle(12345)) == _puzzle_key(generate_puzzle(12345))
    for seed, train_count, test_count in ((8000, 2, 1), (8001, 10, 1),
                                          (8002, 10, 4), (8003, 6, 2),
                                          (8100, None, 1), (8101, None, 4),
                                          (8102, 2, None), (8103, 10, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        validate_generated_puzzle(puzzle)
        assert train_count is None or len(puzzle["train"]) == train_count
        assert test_count is None or len(puzzle["test"]) == test_count
    for train_count, test_count in ((1, None), (11, None), (None, 0),
                                    (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(99, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count was accepted")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) >= 40
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    all_info = {split: [_derive_output(pair["input"])[1]
                        for puzzle in puzzles for pair in puzzle[split]]
                for split in ("train", "test")}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({_key(pair["input"])[0] for puzzle in puzzles
                for split in ("train", "test") for pair in puzzle[split]}) >= 35
    assert len({info["thicknesses"] for split in all_info for info in all_info[split]}) >= 50
    assert {info["orientation"] for info in all_info["test"]} == {"vertical", "horizontal"}
    assert any(len({item["orientation"] for item in
                    [_derive_output(p["input"])[1] for p in puzzle["test"]]}) == 2
               for puzzle in puzzles)
    assert any(all(_derive_output(p["input"])[1]["orientation"] ==
                   _derive_output(puzzle["train"][0]["input"])[1]["orientation"]
                   for split in ("train", "test") for p in puzzle[split])
               for puzzle in puzzles)
    assert any(len(set(info["colors"])) == len(info["colors"]) for info in all_info["test"])
    assert any(len(set(info["colors"])) < len(info["colors"]) for info in all_info["test"])
    assert any(len(info["colors"]) >= 6 and len(set(info["colors"])) == 2
               for info in all_info["test"])
    assert any(all(value == 1 for value in info["thicknesses"]) for info in all_info["test"])
    assert any(len(set(info["thicknesses"])) == 1 and info["thicknesses"][0] > 1
               for info in all_info["test"])
    assert any(max(info["thicknesses"]) >= 5 for info in all_info["test"])
    assert any(info["perpendicular"] == 1 for info in all_info["test"])
    assert any(info["perpendicular"] >= 10 for info in all_info["test"])
    assert any((_derive_output(pair["input"])[1]["orientation"] == "vertical"
                and pair["input"].shape[0] > pair["input"].shape[1])
               for puzzle in puzzles for pair in puzzle["test"])
    assert any((_derive_output(pair["input"])[1]["orientation"] == "horizontal"
                and pair["input"].shape[1] > pair["input"].shape[0])
               for puzzle in puzzles for pair in puzzle["test"])
    assert any(pair["input"].shape[0] == pair["input"].shape[1]
               for puzzle in puzzles for split in ("train", "test") for pair in puzzle[split])
    assert any(len(_derive_output(pair["input"])[1]["colors"]) <
               max(len(_derive_output(p["input"])[1]["colors"]) for p in puzzle["train"])
               for puzzle in puzzles for pair in puzzle["test"])
    assert any(len(_derive_output(pair["input"])[1]["colors"]) >
               max(len(_derive_output(p["input"])[1]["colors"]) for p in puzzle["train"])
               for puzzle in puzzles for pair in puzzle["test"])
    palette_relations = set()
    color_difficulty = set()
    perpendicular_difficulty = set()
    later_recurrence = False
    pair_palettes_vary = True
    for puzzle in puzzles:
        train_info = [_derive_output(pair["input"])[1] for pair in puzzle["train"]]
        test_info = [_derive_output(pair["input"])[1] for pair in puzzle["test"]]
        train_palette = set().union(*(set(item["colors"]) for item in train_info))
        all_palettes = [frozenset(item["colors"]) for item in train_info + test_info]
        pair_palettes_vary &= len(set(all_palettes)) > 1
        later_recurrence |= (len(set(train_info[0]["colors"])) == len(train_info[0]["colors"])
                             and any(len(set(item["colors"])) < len(item["colors"])
                                     for item in train_info[1:]))
        for item in test_info:
            colors = set(item["colors"])
            if colors < train_palette:
                palette_relations.add("subset")
            if colors - train_palette:
                palette_relations.add("unseen")
            color_max = max(len(set(train["colors"])) for train in train_info)
            perpendicular_max = max(train["perpendicular"] for train in train_info)
            color_difficulty.add("higher" if len(colors) > color_max else
                                 "lower" if len(colors) < color_max else "equal")
            perpendicular_difficulty.add("higher" if item["perpendicular"] > perpendicular_max
                                         else "lower" if item["perpendicular"] < perpendicular_max
                                         else "equal")
    assert pair_palettes_vary and later_recurrence
    assert palette_relations == {"subset", "unseen"}
    assert {"higher", "lower"} <= color_difficulty
    assert {"higher", "lower"} <= perpendicular_difficulty
