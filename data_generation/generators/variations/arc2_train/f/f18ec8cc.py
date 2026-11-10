import numpy as np


_AUTOMATIC_COUNT_SCHEDULE = (
    (2, 1),
    (3, 1),
    (4, 2),
    (5, 1),
    (5, 2),
    (6, 3),
    (7, 1),
    (7, 2),
    (8, 3),
    (9, 4),
    (10, 1),
    (10, 4),
)


def _text_grid_key(rows: tuple[str, ...]) -> tuple:
    return (
        len(rows),
        len(rows[0]),
        tuple(int(ch) for row in rows for ch in row),
    )


_SOURCE_ROW_PAIRS = (
    (
        (
            "55555777777999",
            "55555777777999",
            "55555777777999",
            "55555777777999",
            "55555777777999",
            "55955779777999",
            "55755777777979",
            "55555777757959",
            "55555777777999",
        ),
        (
            "99977777755555",
            "99977777755555",
            "99977777755555",
            "99977777755555",
            "99977777755555",
            "99977977755955",
            "97977777755755",
            "95977775755555",
            "99977777755555",
        ),
    ),
    (
        (
            "898999191393",
            "888999111333",
            "888989181383",
            "888999111333",
            "838939131333",
            "888999111333",
            "818919111313",
        ),
        (
            "999191393898",
            "999111333888",
            "989181383888",
            "999111333888",
            "939131333838",
            "999111333888",
            "919111313818",
        ),
    ),
    (
        (
            "8888822221111",
            "8881822221111",
            "8882822221111",
            "8888822221111",
            "8888822821811",
            "8888821221121",
            "8888822221111",
            "8888822221111",
        ),
        (
            "2222111188888",
            "2222111188818",
            "2222111188828",
            "2222111188888",
            "2282181188888",
            "2122112188888",
            "2222111188888",
            "2222111188888",
        ),
    ),
    (
        (
            "5557771111",
            "5157771111",
            "5557771111",
            "5557571711",
            "5757171111",
            "5557771111",
            "5557771151",
            "5557771111",
        ),
        (
            "7771111555",
            "7771111515",
            "7771111555",
            "7571711555",
            "7171111575",
            "7771111555",
            "7771151555",
            "7771111555",
        ),
    ),
    (
        (
            "9994441111111",
            "9994441111111",
            "9994941119111",
            "9194141111111",
            "9494441114111",
            "9994441111111",
            "9994441111111",
            "9994441111111",
        ),
        (
            "4441111111999",
            "4441111111999",
            "4941119111999",
            "4141111111919",
            "4441114111949",
            "4441111111999",
            "4441111111999",
            "4441111111999",
        ),
    ),
    (
        (
            "111177788888333",
            "111177788888333",
            "131173788838333",
            "171177788878373",
            "181178788888383",
            "111171788818313",
            "111177788888333",
        ),
        (
            "777888883331111",
            "777888883331111",
            "737888383331311",
            "777888783731711",
            "787888883831811",
            "717888183131111",
            "777888883331111",
        ),
    ),
)


_SOURCE_PAIR_KEYS = frozenset(
    (_text_grid_key(input_rows), _text_grid_key(output_rows))
    for input_rows, output_rows in _SOURCE_ROW_PAIRS
)


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        int(grid.shape[0]),
        int(grid.shape[1]),
        tuple(int(value) for value in grid.ravel()),
    )


def _pair_key(pair: dict) -> tuple:
    return (_grid_key(pair["input"]), _grid_key(pair["output"]))


def _normalize_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be an integer")
    seed_value = int(seed)
    if seed_value < 0:
        raise ValueError("seed must be nonnegative")
    return seed_value


def _normalize_optional_count(
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
    normalized = int(value)
    if not minimum <= normalized <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return normalized


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _normalize_optional_count(
        num_train, "num_train", 2, 10
    )
    test_count = _normalize_optional_count(
        num_test, "num_test", 1, 4
    )

    if train_count is None and test_count is None:
        train_count, test_count = _AUTOMATIC_COUNT_SCHEDULE[
            seed % len(_AUTOMATIC_COUNT_SCHEDULE)
        ]
    elif train_count is None:
        assert test_count is not None
        minimum_train = max(2, test_count + 1)
        train_count = int(rng.integers(minimum_train, 11))
    elif test_count is None:
        maximum_test = min(4, train_count - 1)
        if maximum_test < 1:
            raise ValueError("no legal test count for num_train")
        test_count = int(rng.integers(1, maximum_test + 1))

    assert train_count is not None
    assert test_count is not None

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    # The reading direction is learned from the demonstrations, then shared by
    # every pair in this puzzle. The leftmost slab always finishes last.
    return {"direction": ("top_down", "bottom_up")[int(rng.integers(2))]}


def _sample_widths(rng: np.random.Generator, n: int, profile: str) -> tuple[int, ...]:
    if profile == "equal":
        base = int(rng.integers(3, 7 if n == 3 else 6))
        return (base,) * n
    if profile == "mild":
        base = int(rng.integers(3, 5 if n == 5 else 6))
        offsets = rng.integers(0, 3, size=n)
        if np.all(offsets == offsets[0]):
            offsets[int(rng.integers(n))] = (int(offsets[0]) + 1) % 3
        return tuple(int(base + offset) for offset in offsets)
    if profile == "strong":
        widths = rng.integers(3, 6, size=n)
        wide = int(rng.integers(n))
        widths[wide] = int(rng.integers(7, 11 if n <= 4 else 9))
        return tuple(int(value) for value in widths)
    raise ValueError("unknown width profile")


def _sample_order(rng: np.random.Generator, n: int, noncycle: bool) -> tuple[int, ...]:
    candidates = [int(i) for i in range(1, n)]
    if not noncycle and int(rng.integers(3)) == 0:
        return tuple(candidates + [0])
    for _ in range(16):
        shuffled = tuple(int(i) for i in rng.permutation(candidates))
        if not noncycle or shuffled != tuple(candidates):
            return shuffled + (0,)
    return tuple(reversed(candidates)) + (0,)


def generate_pair(
    rng: np.random.Generator,
    regime: str,
    puzzle_parameters: dict,
    band_count: int,
    profile: str,
    noncycle: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if regime not in {"source_train", "target_test"}:
        raise ValueError("unknown generation regime")
    if band_count not in (3, 4, 5):
        raise ValueError("invalid band count")

    n = band_count
    widths = _sample_widths(rng, n, profile)
    # Training and testing overlap in both axes. Tests lean toward larger or
    # less balanced cases, without making dimensions an answer label.
    if regime == "target_test" and int(rng.integers(3)) == 0:
        height = int(rng.integers(12, 19))
    else:
        height = int(rng.integers(7, 16))
    colors = [int(c) for c in rng.choice(10, size=n, replace=False)]
    order = _sample_order(rng, n, noncycle)
    rank = {index: position for position, index in enumerate(order)}
    direction = puzzle_parameters["direction"]
    aligned_rows = (int(rng.integers(4)) == 0)
    common_rows = (sorted(int(r) for r in rng.choice(height, size=n, replace=False))
                   if aligned_rows else None)

    bands = []
    for owner in range(n):
        band = np.full((height, widths[owner]), colors[owner], dtype=np.int8)
        others = sorted((target for target in range(n) if target != owner), key=rank.get)
        if direction == "bottom_up":
            others.reverse()
        # Each band samples its own rows. The rank relation is stable, but a
        # color need not occupy the same row in different bands.
        if common_rows is None:
            rows = sorted(int(r) for r in rng.choice(height, size=n - 1, replace=False))
        else:
            rows = [common_rows[rank[target] if direction == "top_down"
                                else n - 1 - rank[target]] for target in others]
        shared_column = (int(rng.integers(4)) == 0)
        fixed_column = int(rng.integers(1, widths[owner] - 1))
        for row, target in zip(rows, others):
            column = (fixed_column if shared_column else
                      int(rng.integers(1, widths[owner] - 1)))
            band[row, column] = colors[target]
        bands.append(band)

    input_grid = np.concatenate(bands, axis=1)
    output_grid = np.concatenate([bands[index] for index in order], axis=1)
    return input_grid, output_grid


def _make_specs(
    rng: np.random.Generator, train_count: int, test_count: int
) -> tuple[list[tuple[int, str, bool]], list[tuple[int, str, bool]]]:
    # Every training set covers both original band counts, includes a
    # noncyclic order, and shows two width profiles. Positions are shuffled.
    train_n = [3, 4] + [int(rng.choice((3, 4, 5), p=(0.44, 0.44, 0.12)))
                        for _ in range(train_count - 2)]
    rng.shuffle(train_n)
    profiles = ["equal", "strong" if int(rng.integers(2)) else "mild"]
    profiles += [str(rng.choice(("equal", "mild", "strong")))
                 for _ in range(train_count - 2)]
    rng.shuffle(profiles)
    special = int(rng.integers(train_count))
    train_specs = [(train_n[i], profiles[i], i == special)
                   for i in range(train_count)]

    test_n = [int(rng.choice((3, 4, 5), p=(0.4, 0.46, 0.14)))
              for _ in range(test_count)]
    # Five-band tests get a matching demonstration of the same composition.
    if 5 in test_n and 5 not in train_n:
        test_n = [4 if n == 5 else n for n in test_n]
    test_profiles = [str(rng.choice(("equal", "mild", "strong"),
                                    p=(0.2, 0.3, 0.5)))
                     for _ in range(test_count)]
    special = int(rng.integers(test_count))
    test_specs = [(test_n[i], test_profiles[i], i == special)
                  for i in range(test_count)]
    return train_specs, test_specs


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed_value = _normalize_seed(seed)
    rng = np.random.default_rng(seed_value)
    train_count, test_count = _resolve_counts(seed_value, rng, num_train, num_test)
    puzzle_parameters = sample_puzzle_parameters(rng)
    train_specs, test_specs = _make_specs(rng, train_count, test_count)

    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    for section, regime, specs in (
        ("train", "source_train", train_specs),
        ("test", "target_test", test_specs),
    ):
        for n, profile, noncycle in specs:
            for _ in range(100):
                child_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                child_rng = np.random.default_rng(child_seed)
                inp, out = generate_pair(child_rng, regime, puzzle_parameters,
                                         n, profile, noncycle)
                pair = {"input": inp, "output": out}
                key = _grid_key(inp)
                if key in seen_inputs or _pair_key(pair) in _SOURCE_PAIR_KEYS:
                    continue
                seen_inputs.add(key)
                puzzle[section].append(pair)
                break
            else:
                raise RuntimeError("could not generate a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert all(1 <= size <= 30 for size in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((0 <= grid) & (grid <= 9))


def _analyze_input(grid: np.ndarray) -> dict:
    height, width = grid.shape
    column_colors = []
    for column in range(width):
        counts = np.bincount(grid[:, column].astype(np.int64), minlength=10)
        winners = np.flatnonzero(counts == counts.max())
        assert len(winners) == 1
        column_colors.append(int(winners[0]))
    starts = [0] + [j for j in range(1, width)
                    if column_colors[j] != column_colors[j - 1]] + [width]
    bands = [grid[:, starts[i]:starts[i + 1]] for i in range(len(starts) - 1)]
    colors = tuple(column_colors[start] for start in starts[:-1])
    widths = tuple(int(band.shape[1]) for band in bands)
    n = len(bands)
    assert n in (3, 4, 5) and len(set(colors)) == n
    assert all(w >= 3 for w in widths)
    positions = []
    for owner, band in enumerate(bands):
        counts = {int(c): int(k) for c, k in zip(*np.unique(band, return_counts=True))}
        assert set(counts) == set(colors)
        assert counts[colors[owner]] == height * widths[owner] - (n - 1)
        local = {}
        for target in range(n):
            if target == owner:
                continue
            assert counts[colors[target]] == 1
            row, col = np.argwhere(band == colors[target])[0]
            assert 1 <= col < widths[owner] - 1
            local[target] = (int(row), int(col))
        positions.append(local)
    return {"height": height, "width": width, "n": n,
            "bands": bands, "colors": colors, "widths": widths,
            "positions": positions}


def _decoded_order(analysis: dict, direction: str) -> tuple[int, ...]:
    assert direction in ("top_down", "bottom_up")
    first_markers = analysis["positions"][0]
    others = sorted(first_markers, key=lambda index: first_markers[index][0],
                    reverse=(direction == "bottom_up"))
    order = tuple(others) + (0,)
    rank = {band: position for position, band in enumerate(order)}
    for local in analysis["positions"]:
        by_row = sorted(local, key=lambda index: local[index][0],
                        reverse=(direction == "bottom_up"))
        assert by_row == sorted(local, key=rank.get)
    return order


def _output_for(analysis: dict, direction: str) -> np.ndarray:
    return np.concatenate([analysis["bands"][i]
                           for i in _decoded_order(analysis, direction)], axis=1)


def _infer_direction(train: list, analyses: list) -> str:
    candidates = []
    for direction in ("top_down", "bottom_up"):
        try:
            if all(np.array_equal(_output_for(analysis, direction), pair["output"])
                   for pair, analysis in zip(train, analyses)):
                candidates.append(direction)
        except AssertionError:
            pass
    assert len(candidates) == 1
    return candidates[0]


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4
    assert train_count > test_count
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            _validate_grid(inp)
            _validate_grid(out)
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            input_key, pair_key = _grid_key(inp), _pair_key(pair)
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert pair_key not in _SOURCE_PAIR_KEYS
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            analysis = _analyze_input(inp)
            analyses[section].append(analysis)
            assert 7 <= analysis["height"] <= 18
            assert 9 <= analysis["width"] <= 30
            assert np.array_equal(np.bincount(inp.ravel(), minlength=10),
                                  np.bincount(out.ravel(), minlength=10))

    assert {a["n"] for a in analyses["train"]}.issuperset({3, 4})
    assert len({len(set(a["widths"])) == 1 for a in analyses["train"]}) == 2
    direction = _infer_direction(puzzle["train"], analyses["train"])
    assert {a["n"] for a in analyses["test"]}.issubset(
        {a["n"] for a in analyses["train"]})
    for section in ("train", "test"):
        orders = []
        for pair, analysis in zip(puzzle[section], analyses[section]):
            order = _decoded_order(analysis, direction)
            assert order[-1] == 0
            assert np.array_equal(_output_for(analysis, direction), pair["output"])
            orders.append(order)
        assert any(order != tuple(range(1, len(order))) + (0,) for order in orders)


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[s]) == len(second[s]) and
               all(np.array_equal(a[role], b[role])
                   for a, b in zip(first[s], second[s])
                   for role in ("input", "output"))
               for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert _puzzles_equal(generate_puzzle(100, 7, 3), generate_puzzle(100, 7, 3))
    assert len({_puzzle_key(p) for p in puzzles}) == 42

    heights, widths, n_values, colors, orders = set(), set(), set(), set(), set()
    profiles, placements, directions = set(), set(), set()
    staggered = mixed_columns = noncycle_tests = small_tests = 0
    total_tests = 0
    for puzzle in puzzles:
        analyses = {s: [_analyze_input(pair["input"]) for pair in puzzle[s]]
                    for s in ("train", "test")}
        direction = _infer_direction(puzzle["train"], analyses["train"])
        directions.add(direction)
        max_train_width = max(a["width"] for a in analyses["train"])
        for section in ("train", "test"):
            for a in analyses[section]:
                heights.add(a["height"])
                widths.add(a["width"])
                n_values.add(a["n"])
                colors.update(a["colors"])
                profiles.add(a["widths"])
                order = _decoded_order(a, direction)
                orders.add((a["n"], order))
                placements.add(tuple(tuple(sorted(local.items()))
                                     for local in a["positions"]))
                if any(len({local[t][0] for local in a["positions"] if t in local}) > 1
                       for t in range(a["n"])):
                    staggered += 1
                if any(len({col for _, col in local.values()}) > 1
                       for local in a["positions"]):
                    mixed_columns += 1
                if section == "test":
                    total_tests += 1
                    noncycle_tests += order != tuple(range(1, a["n"])) + (0,)
                    small_tests += a["width"] <= max_train_width
    assert heights.issuperset({7, 8, 9, 12, 15, 17})
    assert len(widths) >= 14 and n_values == {3, 4, 5}
    assert colors == set(range(10)) and directions == {"top_down", "bottom_up"}
    assert len(profiles) >= 65 and len(placements) >= 200
    assert staggered >= 150 and mixed_columns >= 80
    assert noncycle_tests >= total_tests // 2 and small_tests >= 8
    assert len(orders) >= 9

    for seed, train, test in ((1000, 2, 1), (1001, 10, 1),
                              (1002, 10, 4), (1003, None, 1),
                              (1004, None, 4), (1005, 2, None),
                              (1006, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for kwargs in ({"num_train": 1}, {"num_train": 11},
                   {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4},
                   {"num_train": True}, {"num_test": 1.5}):
        try:
            generate_puzzle(3000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(tuple(_pair_key(pair) for pair in puzzle[section])
                 for section in ("train", "test"))
