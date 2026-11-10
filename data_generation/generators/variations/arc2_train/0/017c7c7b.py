"""Synthetic whole-row continuation puzzles in the 017c7c7b family."""

import numpy as np


_SOURCE_INPUTS = (
    ("010", "101", "010", "101", "010", "101"),
    ("010", "110", "010", "011", "010", "110"),
    ("010", "110", "010", "010", "110", "010"),
    ("111", "010", "010", "111", "010", "010"),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _checked_count(value: int | None, name: str, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng: np.random.Generator, num_train: int | None,
                    num_test: int | None) -> tuple[int, int]:
    train = _checked_count(num_train, "num_train", 2, 10)
    test = _checked_count(num_test, "num_test", 1, 4)
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


def _signature(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _shortest_period(rows: np.ndarray) -> int:
    for period in range(1, rows.shape[0] + 1):
        if np.array_equal(rows[period:], rows[:-period]):
            return period
    raise AssertionError("finite sequences always have a period")


def _row(rng: np.random.Generator, width: int) -> np.ndarray:
    kind = int(rng.choice(8, p=[.08, .09, .18, .16, .16, .10, .10, .13]))
    row = np.zeros(width, dtype=np.int8)
    if kind == 0:
        return row
    if kind == 1:
        row[:] = 1
        return row
    if kind == 7:
        half = rng.integers(0, 2, size=(width + 1) // 2, dtype=np.int8)
        return np.concatenate((half, half[-2::-1] if width % 2 else half[::-1]))
    if kind == 2:
        count = 1
    elif kind == 3:
        count = max(1, width // 3)
    elif kind == 4:
        count = max(1, width - 1)
    elif kind == 5:
        count = int(rng.integers(1, width))
    else:
        count = int(rng.integers(0, width + 1))
    columns = rng.choice(width, size=count, replace=False)
    row[columns] = 1
    return row


def _cycle(rng: np.random.Generator, period: int, width: int,
           regime: str) -> np.ndarray:
    # The source test's motif remains possible in either split.
    motif_probability = .35 if regime == "target_test" else .18
    if period == 3 and rng.random() < motif_probability:
        singleton = np.zeros(width, dtype=np.int8)
        position = width // 2 if rng.random() < .5 else int(rng.integers(width))
        singleton[position] = 1
        return np.stack((np.ones(width, dtype=np.int8), singleton, singleton))

    rows = np.stack([_row(rng, width) for _ in range(period)])
    if period >= 3 and rng.random() < (.38 if regime == "target_test" else .32):
        first = int(rng.integers(period))
        second = (first + int(rng.integers(2, period))) % period
        rows[second] = rows[first]
    return rows


def _eligible_widths(period: int, crossing: bool) -> list[int]:
    widths = []
    for width in range(2, 11):
        overlap = 2 * width - period
        if overlap < (2 if period <= 4 else 3):
            continue
        if crossing and width % period == 0:
            continue
        widths.append(width)
    return widths


def _make_pair(rng: np.random.Generator, foreground: int, replacement: int,
               background: int, period: int, regime: str,
               crossing: bool = False) -> dict:
    widths = _eligible_widths(period, crossing)
    for _ in range(256):
        width = int(rng.choice(widths))
        cycle = _cycle(rng, period, width, regime)
        phase = int(rng.integers(period))
        binary = cycle[(np.arange(2 * width) + phase) % period]
        if not np.any(binary) or np.all(binary):
            continue
        if _shortest_period(binary) != period:
            continue
        if width == 3 and foreground == 1 and replacement == 2 and background == 0:
            lines = tuple("".join(str(int(x)) for x in row) for row in binary)
            if lines in _SOURCE_INPUTS:
                continue
        extended = np.empty((3 * width, width), dtype=np.int8)
        extended[:2 * width] = binary
        for index in range(2 * width, 3 * width):
            extended[index] = extended[index - period]
        return {
            "input": np.where(binary, foreground, background).astype(np.int8),
            "output": np.where(extended, replacement, background).astype(np.int8),
        }
    raise RuntimeError("could not construct a valid periodic pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    foreground, replacement = (int(x) for x in rng.choice(10, size=2, replace=False))
    backgrounds = [x for x in range(10) if x not in (foreground, replacement)]
    rng.shuffle(backgrounds)

    # First two demonstrations establish variable period and variable background.
    first = int(rng.integers(2, 7))
    second_choices = [x for x in range(2, 7) if x != first]
    second = int(rng.choice(second_choices))
    train_periods = [first, second]
    train_periods.extend(int(rng.integers(1, 7)) for _ in range(train_count - 2))
    rng.shuffle(train_periods)
    crossing_index = int(rng.choice([i for i, p in enumerate(train_periods) if p >= 2]))
    # One demonstration crosses a width boundary; the others are unrestricted.
    train_backgrounds = backgrounds[:min(train_count, len(backgrounds))]
    while len(train_backgrounds) < train_count:
        train_backgrounds.append(int(rng.choice(backgrounds)))
    rng.shuffle(train_backgrounds)

    unused = [x for x in backgrounds if x not in train_backgrounds]
    test_backgrounds = []
    for _ in range(test_count):
        if unused and rng.random() < .55:
            chosen = int(rng.choice(unused))
            unused.remove(chosen)
        else:
            chosen = int(rng.choice(backgrounds))
        test_backgrounds.append(chosen)

    seen = set()
    puzzle = {"train": [], "test": []}
    for regime, periods, selected_backgrounds in (
        ("source_train", train_periods, train_backgrounds),
        ("target_test", [int(rng.integers(1, 7)) for _ in range(test_count)],
         test_backgrounds),
    ):
        split = "train" if regime == "source_train" else "test"
        for index, (period, background) in enumerate(zip(periods, selected_backgrounds)):
            for _ in range(256):
                pair = _make_pair(rng, foreground, replacement, background,
                                  period, regime,
                                  crossing=(split == "train" and index == crossing_index))
                key = _signature(pair["input"])
                previous = puzzle[split]
                if (index == len(periods) - 1 and len(periods) > 1 and previous
                        and len({item["input"].shape[1] for item in previous}) == 1
                        and pair["input"].shape[1] == previous[0]["input"].shape[1]):
                    continue
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct unique inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid: np.ndarray, label: str) -> None:
    _require(isinstance(grid, np.ndarray), f"{label} is not an ndarray")
    _require(grid.ndim == 2, f"{label} must be two-dimensional")
    _require(all(1 <= x <= 30 for x in grid.shape), f"{label} has illegal dimensions")
    _require(np.issubdtype(grid.dtype, np.integer), f"{label} must have integer dtype")
    _require(bool(np.all((grid >= 0) & (grid <= 9))), f"{label} has illegal colors")


def _roles_and_period(pair: dict) -> tuple[int, int, int, int]:
    inp, out = pair["input"], pair["output"]
    height, width = inp.shape
    _require(height == 2 * width, "input must be 2w by w")
    _require(out.shape == (3 * width, width), "output must be 3w by w")
    colors = np.unique(inp)
    _require(len(colors) == 2, "input must use both semantic colors")
    mapping = {}
    for color in colors:
        targets = np.unique(out[:height][inp == color])
        _require(len(targets) == 1, "prefix recoloring must be uniform")
        mapping[int(color)] = int(targets[0])
    fixed = [x for x in mapping if mapping[x] == x]
    changed = [x for x in mapping if mapping[x] != x]
    _require(len(fixed) == 1 and len(changed) == 1, "exactly one color must change")
    background, foreground = fixed[0], changed[0]
    replacement = mapping[foreground]
    _require(replacement not in (background, foreground), "color roles must differ")
    _require(set(int(x) for x in np.unique(out)) == {background, replacement},
             "output palette is wrong")
    period = _shortest_period(inp)
    minimum_overlap = 2 if period <= 4 else 3
    _require(height - period >= minimum_overlap, "period lacks repeated visible rows")
    extended = np.empty((3 * width, width), dtype=np.int8)
    extended[:height] = inp
    for index in range(height, 3 * width):
        extended[index] = extended[index - period]
    expected = np.where(extended == foreground, replacement, background)
    _require(np.array_equal(out, expected), "incorrect periodic continuation")
    return background, foreground, replacement, period


def _exact_source_copy(pair: dict) -> bool:
    inp, out = pair["input"], pair["output"]
    if inp.shape != (6, 3) or out.shape != (9, 3):
        return False
    lines = tuple("".join(str(int(x)) for x in row) for row in inp)
    if lines not in _SOURCE_INPUTS:
        return False
    source_period = _shortest_period(inp)
    source_out = np.empty((9, 3), dtype=np.int8)
    source_out[:6] = inp
    for index in range(6, 9):
        source_out[index] = source_out[index - source_period]
    source_out = np.where(source_out == 1, 2, 0)
    return np.array_equal(out, source_out)


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    train_periods, train_backgrounds, train_widths = set(), set(), set()
    test_widths = set()
    crossing_seen = False
    learned_roles = None
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            label = f"{split}[{index}]"
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     f"{label} has incorrect schema")
            _validate_grid(pair["input"], label + ".input")
            _validate_grid(pair["output"], label + ".output")
            key = _signature(pair["input"])
            pair_key = (key, _signature(pair["output"]))
            _require(key not in seen_inputs and pair_key not in seen_pairs,
                     "duplicate input or pair")
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            _require(not _exact_source_copy(pair), "exact source pair was copied")
            background, foreground, replacement, period = _roles_and_period(pair)
            _require(1 <= period <= 6, "period outside supported regimes")
            if learned_roles is None:
                learned_roles = foreground, replacement
            _require((foreground, replacement) == learned_roles,
                     "foreground convention changes within puzzle")
            if split == "train":
                train_periods.add(period)
                train_backgrounds.add(background)
                train_widths.add(pair["input"].shape[1])
                crossing_seen |= pair["input"].shape[1] % period != 0
            else:
                test_widths.add(pair["input"].shape[1])
    _require(len(train_periods) >= 2, "training must show variable periods")
    _require(len(train_backgrounds) >= 2, "training must show variable backgrounds")
    _require(len(train_widths) >= 2, "training must show variable widths")
    if test_count > 1:
        _require(len(test_widths) >= 2, "multiple tests must vary widths")
    _require(crossing_seen, "training must rule out width-block copying")


def _same_puzzle(a: dict, b: dict) -> bool:
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[role], y[role])
        for x, y in zip(a[s], b[s]) for role in ("input", "output"))
        for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    _require(_same_puzzle(generate_puzzle(811), generate_puzzle(811)),
             "automatic calls are nondeterministic")
    _require(_same_puzzle(generate_puzzle(812, 8, 3), generate_puzzle(812, 8, 3)),
             "explicit calls are nondeterministic")
    _require(not _same_puzzle(generate_puzzle(811), generate_puzzle(812)),
             "different seeds made identical puzzles")
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4),
                                          (None, 1), (None, 4), (2, None),
                                          (10, None), (7, 3))):
        puzzle = generate_puzzle(9000 + index, train, test)
        _require(train is None or len(puzzle["train"]) == train,
                 "requested train count ignored")
        _require(test is None or len(puzzle["test"]) == test,
                 "requested test count ignored")
    for args in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4), (True, 1), (3.0, 1)):
        try:
            generate_puzzle(9999, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {args}")

    train_counts = {len(p["train"]) for p in puzzles}
    test_counts = {len(p["test"]) for p in puzzles}
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "automatic pair counts lack diversity")
    by_split = {s: {"periods": set(), "widths": set(), "layouts": set(),
                    "blank": 0, "full": 0, "repeated": 0, "short": 0,
                    "aligned": 0, "crossing": 0} for s in ("train", "test")}
    palettes, backgrounds, unseen_test_backgrounds = set(), set(), 0
    familiar_test_periods, novel_test_periods = 0, 0
    for puzzle in puzzles:
        train_bgs = set()
        puzzle_train_periods = {_shortest_period(pair["input"])
                                for pair in puzzle["train"]}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                bg, fg, rep, period = _roles_and_period(pair)
                width = pair["input"].shape[1]
                binary = pair["input"] == fg
                stats = by_split[split]
                stats["periods"].add(period)
                stats["widths"].add(width)
                stats["layouts"].add(_signature(binary))
                stats["blank"] += int(bool(np.any(np.all(~binary, axis=1))))
                stats["full"] += int(bool(np.any(np.all(binary, axis=1))))
                stats["repeated"] += int(len({_signature(row) for row in binary[:period]}) < period)
                stats["short"] += int(period > width)
                stats["aligned"] += int(width % period == 0)
                stats["crossing"] += int(width % period != 0)
                stats.setdefault("symmetric_mixed", 0)
                stats["symmetric_mixed"] += int(any(
                    0 < int(row.sum()) < width and np.array_equal(row, row[::-1])
                    for row in binary[:period]))
                cycle_counts = sorted(int(row.sum()) for row in binary[:period])
                singletons = [tuple(row) for row in binary[:period] if row.sum() == 1]
                stats.setdefault("source_motif", 0)
                stats["source_motif"] += int(period == 3 and
                    cycle_counts == [1, 1, width] and len(set(singletons)) == 1)
                if split == "train":
                    train_bgs.add(bg)
                    palettes.add((fg, rep))
                else:
                    unseen_test_backgrounds += int(bg not in train_bgs)
                    familiar_test_periods += int(period in puzzle_train_periods)
                    novel_test_periods += int(period not in puzzle_train_periods)
                backgrounds.add(bg)
    for split in ("train", "test"):
        stats = by_split[split]
        _require(stats["periods"] == set(range(1, 7)), f"{split} period coverage")
        _require(stats["widths"] == set(range(2, 11)), f"{split} width coverage")
        _require(len(stats["layouts"]) >= 90, f"{split} layout diversity")
        for feature in ("blank", "full", "repeated", "short", "aligned",
                        "crossing", "symmetric_mixed"):
            _require(stats[feature] > 0, f"{split} lacks {feature} cases")
        _require(stats["source_motif"] > 0, f"{split} lacks source-like motif")
    _require(len(palettes) >= 20 and len(backgrounds) == 10,
             "color role diversity is too small")
    _require(unseen_test_backgrounds > 0, "no test used a new background")
    _require(familiar_test_periods > 0 and novel_test_periods > 0,
             "test periods do not mix familiar and new values")
