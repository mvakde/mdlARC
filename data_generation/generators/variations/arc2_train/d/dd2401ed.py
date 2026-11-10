"""Synthetic instances of the delimiter/count ARC task."""

import numpy as np


_SOURCE_INPUT_ROWS = (
    ("050000000000000", "150020000002000", "050020000200000",
     "050000000000000", "050000000000000", "050000200000200",
     "050000000000000"),
    ("000105200020000", "010005020000000", "000005000200020",
     "000105000000200", "000005002000000", "100015000000002",
     "000005200200000"),
    ("000500000000002", "100502000000020", "000500000002000",
     "000500000000000", "001500000200200", "000500200000020",
     "000500000000000"),
    ("000050200000000", "000050000002000", "010050000000000",
     "000050020000000", "001050000000000", "100050000000020",
     "000050002000000"),
    ("005200020020020", "105000200000000", "005000002000002",
     "005020000002000", "015000002000200", "105200000000000",
     "005000200020002"),
)


def _grid_key(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


_SOURCE_INPUT_KEYS = frozenset(
    ((len(rows), len(rows[0])), tuple(int(x) for row in rows for x in row))
    for rows in _SOURCE_INPUT_ROWS
)


def _count_arg(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return value


def _pair_counts(rng, num_train, num_test):
    num_train = _count_arg(num_train, "num_train", 2, 10)
    num_test = _count_arg(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _delimiter(rng):
    # The long window is uncommon, but is sampled in both splits.
    band = int(rng.choice(3, p=[0.48, 0.38, 0.14]))
    lo, hi = ((1, 5), (6, 10), (11, 13))[band]
    return int(rng.integers(lo, hi + 1))


def _spec(rng, regime, delimiter=None, outcome=None):
    d = _delimiter(rng) if delimiter is None else delimiter
    h = int(rng.integers(5, 25))
    if outcome in ("equal", "above"):
        h = max(7, h)
    maximum_tail = min(18, 30 - (2 * d + 2))
    bucket = int(rng.integers(3))
    if bucket == 0:
        lo, hi = 0, min(2, maximum_tail)
    elif bucket == 1:
        lo, hi = min(3, maximum_tail), min(8, maximum_tail)
    else:
        lo, hi = min(9, maximum_tail), maximum_tail
    tail = int(rng.integers(lo, hi + 1))
    if outcome == "below" and tail == 0:
        tail = 1
    return {"d": d, "h": h, "tail": tail, "outcome": outcome,
            "regime": regime}


def _maximum_count(spec):
    # Keep one background cell per column so the delimiter is the sole
    # non-background full-span line. Other columns can be nearly full.
    return spec["d"] * (spec["h"] - 1)


def _ordinary_count_limit(spec):
    # Most examples retain an easily recognized background. Rare
    # near-capacity examples put very few markers in the other block.
    area = spec["h"] * spec["d"]
    return min(_maximum_count(spec), max(2, int(0.58 * area)))


def _level(rng, low, high, regime):
    assert low <= high
    if low == high:
        return low
    bucket = int(rng.choice(3, p=[0.27, 0.38, 0.35] if regime == "source_train"
                            else [0.16, 0.35, 0.49]))
    if bucket == 0:
        upper = min(high, max(low, 5))
        return int(rng.integers(low, upper + 1))
    if bucket == 1:
        lower = max(low, high // 5)
        upper = max(lower, (3 * high) // 5)
        return int(rng.integers(lower, upper + 1))
    return int(rng.integers(max(low, (3 * high) // 5), high + 1))


def _choose_block_counts(rng, spec):
    limit = _ordinary_count_limit(spec)
    outcome = spec["outcome"]
    if outcome == "equal":
        reference = _level(rng, 3, limit, spec["regime"])
        candidate = reference
    elif outcome == "low_equal":
        reference, candidate = 2, 2
    elif outcome == "below":
        if int(rng.integers(8)) == 0:
            capacity = _maximum_count(spec)
            reference = int(rng.integers(max(2, int(0.8 * capacity)), capacity + 1))
            candidate = int(rng.integers(0, max(1, min(reference, capacity // 10 + 1))))
        else:
            reference = _level(rng, 2, limit, spec["regime"])
            if int(rng.integers(5)) == 0:
                candidate = 0
            else:
                gap = _level(rng, 1, reference, spec["regime"])
                candidate = reference - gap
    elif outcome == "above":
        if int(rng.integers(8)) == 0:
            capacity = _maximum_count(spec)
            candidate = int(rng.integers(max(2, int(0.8 * capacity)), capacity + 1))
            reference = int(rng.integers(1, max(2, min(candidate, capacity // 10 + 2))))
        else:
            candidate = _level(rng, 3, limit, spec["regime"])
            reference = candidate - _level(rng, 1, candidate - 1, spec["regime"])
    else:
        raise ValueError("unknown comparison outcome")
    return reference, candidate


def _palette(rng):
    return tuple(int(x) for x in rng.choice(10, size=4, replace=False))


def _place(grid, columns, color, count, rng, mode):
    if count == 0:
        return
    h = grid.shape[0]
    columns = list(columns)
    # Reserve one random background cell in each column. All choices use
    # the same full rectangular block; modes only change arrangements.
    reserved = {col: int(rng.integers(h)) for col in columns}
    cells = [(row, col) for col in columns for row in range(h)
             if row != reserved[col]]
    assert count <= len(cells)
    weights = np.ones(len(cells), dtype=float)
    if mode == "cluster":
        center_row = int(rng.integers(h))
        center_col = int(rng.choice(columns))
        for i, (row, col) in enumerate(cells):
            weights[i] = 0.08 + 8.0 / (1 + abs(row - center_row) + abs(col - center_col)) ** 2
    elif mode == "edges" and len(columns) > 1:
        left, right = columns[0], columns[-1]
        for i, (_, col) in enumerate(cells):
            weights[i] = 4.0 if col in (left, right) else 1.0
    elif mode == "spread":
        # Sequential weighted sampling penalizes rows and nearby cells
        # already used. It also permits multiple markers in one row.
        penalties = np.zeros(len(cells), dtype=float)
        available = list(range(len(cells)))
        selected = []
        while len(selected) < count:
            scores = 1.0 / (1.0 + penalties[available])
            scores /= scores.sum()
            pick = int(rng.choice(len(available), p=scores))
            chosen = available.pop(pick)
            selected.append(chosen)
            row, col = cells[chosen]
            for i in available:
                other_row, other_col = cells[i]
                if other_row == row:
                    penalties[i] += 3
                if abs(other_row - row) <= 1 and abs(other_col - col) <= 1:
                    penalties[i] += 2
        for i in selected:
            grid[cells[i]] = color
        return
    weights /= weights.sum()
    for i in rng.choice(len(cells), size=count, replace=False, p=weights):
        grid[cells[int(i)]] = color


def _make_pair(rng, spec, colors):
    background, line, reference_color, candidate_color = colors
    d, h, tail = spec["d"], spec["h"], spec["tail"]
    destination = 2 * d + 1
    width = destination + 1 + tail
    reference, candidate = spec["counts"]
    grid = np.full((h, width), background, dtype=np.int8)
    grid[:, d] = line
    modes = ("uniform", "cluster", "spread", "edges")
    _place(grid, range(d), reference_color, reference, rng,
           modes[int(rng.integers(4))])
    _place(grid, range(d + 1, destination), candidate_color, candidate, rng,
           modes[int(rng.integers(4))])
    if tail:
        # Both an empty/short tail and a conspicuous misleading tail occur.
        minimum = 1 if candidate == 0 or spec.get("misleading", False) else 0
        tail_cap = min(h * tail - tail, max(1, int(0.42 * h * tail)))
        if spec.get("misleading", False):
            minimum = max(minimum, reference - candidate)
        minimum = min(minimum, tail_cap)
        tail_count = _level(rng, minimum, tail_cap, spec["regime"])
        _place(grid, range(destination + 1, width), candidate_color,
               tail_count, rng, modes[int(rng.integers(4))])
    answer = grid.copy()
    answer[:, d] = background
    if candidate >= reference and candidate >= 3:
        block = answer[:, d + 1:destination]
        block[block == candidate_color] = reference_color
    answer[:, destination] = line
    return {"input": grid, "output": answer}


def _inspect(pair):
    grid, answer = pair["input"], pair["output"]
    assert grid.shape == answer.shape
    h, w = grid.shape
    values, amounts = np.unique(grid, return_counts=True)
    assert len(values) == 4
    background = int(values[np.argmax(amounts)])
    assert int(np.max(amounts)) > int(np.partition(amounts, -2)[-2])
    lines = [(col, int(grid[0, col])) for col in range(w)
             if grid[0, col] != background and np.all(grid[:, col] == grid[0, col])]
    assert len(lines) == 1
    d, line = lines[0]
    destination = 2 * d + 1
    assert 1 <= d <= 13 and destination < w
    assert np.all(grid[:, destination] == background)
    near_colors = set(int(x) for x in np.unique(grid[:, :d])) - {background}
    far_colors = set(int(x) for x in np.unique(grid[:, d + 1:])) - {background}
    assert len(near_colors) == 1 and len(far_colors) == 1
    reference_color = near_colors.pop()
    candidate_color = far_colors.pop()
    assert len({background, line, reference_color, candidate_color}) == 4
    assert set(int(x) for x in np.unique(grid[:, d + 1:destination])) <= {background, candidate_color}
    assert set(int(x) for x in np.unique(grid[:, destination + 1:])) <= {background, candidate_color}
    reference = int(np.count_nonzero(grid[:, :d] == reference_color))
    candidate = int(np.count_nonzero(grid[:, d + 1:destination] == candidate_color))
    later = int(np.count_nonzero(grid[:, destination + 1:] == candidate_color))
    assert 1 <= reference <= _maximum_count({"h": h, "d": d})
    assert 0 <= candidate <= _maximum_count({"h": h, "d": d})
    assert candidate or later
    expected = grid.copy()
    expected[:, d] = background
    if candidate >= reference and candidate >= 3:
        block = expected[:, d + 1:destination]
        block[block == candidate_color] = reference_color
    expected[:, destination] = line
    assert np.array_equal(answer, expected)
    assert not np.array_equal(grid, answer)
    return {"roles": (background, line, reference_color, candidate_color),
            "d": d, "h": h, "w": w, "reference": reference,
            "candidate": candidate, "later": later,
            "outcome": "below" if candidate < reference else
                       "low_equal" if candidate == reference and candidate < 3 else
                       "equal" if candidate == reference else "above"}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    n_train, n_test = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n_train <= 10 and 1 <= n_test <= 4 and n_train > n_test
    seen = set()
    facts = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            key = _grid_key(pair["input"])
            assert key not in seen and key not in _SOURCE_INPUT_KEYS
            seen.add(key)
            facts[split].append(_inspect(pair))
    train = facts["train"]
    assert len({x["d"] for x in train}) >= 2
    equal = [x for x in train if x["outcome"] == "equal"]
    below = [x for x in train if x["outcome"] == "below"]
    above = [x for x in train if x["outcome"] == "above"]
    low_equal = [x for x in train if x["outcome"] == "low_equal"]
    assert equal and below
    assert any(a["candidate"] == b["candidate"] and
               b["reference"] == a["reference"] + 1
               for a in equal for b in below)
    if n_train >= 3:
        assert low_equal
    if n_train >= 4:
        assert above
    if any(x["outcome"] == "above" for x in facts["test"]):
        assert above
    if any(x["outcome"] == "low_equal" for x in facts["test"]):
        assert low_equal
    # Literal palettes may change between pairs. In either case every
    # input reveals its four roles through the line and the two sides.
    assert all(len(set(x["roles"])) == 4 for x in train + facts["test"])


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    n_train, n_test = _pair_counts(rng, num_train, num_test)
    stable_palette = bool(rng.integers(2)) or n_train == 2
    shared_colors = _palette(rng)

    required_outcomes = (["equal", "below"] +
                         (["low_equal"] if n_train >= 3 else []) +
                         (["above"] if n_train >= 4 else []))
    train = [_spec(rng, "source_train", outcome=outcome)
             for outcome in (required_outcomes +
                             [str(rng.choice(("equal", "below", "above", "low_equal")))
                              for _ in range(n_train - len(required_outcomes))])]
    # Two widths are required even for the shortest puzzle. A later
    # pair may reuse a width with a different outcome.
    if train[0]["d"] == train[1]["d"]:
        train[1]["d"] = 1 + train[1]["d"] % 13
    if n_train >= 4:
        train[3]["d"] = train[0]["d"]
    for spec in train:
        spec["tail"] = min(spec["tail"], 30 - (2 * spec["d"] + 2))
        if spec["outcome"] == "below":
            spec["tail"] = max(1, spec["tail"])
    train[1]["h"] = max(7, train[1]["h"])
    # Matched accepted/rejected examples share the candidate count but
    # have different thresholds. The pair order is randomized below.
    common_limit = min(_ordinary_count_limit(train[0]),
                       _ordinary_count_limit(train[1]) - 1)
    common = _level(rng, 3, common_limit, "source_train")
    train[0]["counts"] = (common, common)
    train[1]["counts"] = (common + 1, common)
    train[1]["misleading"] = True
    for spec in train[2:]:
        spec["counts"] = _choose_block_counts(rng, spec)
        if spec["outcome"] == "below" and bool(rng.integers(2)):
            spec["misleading"] = True
    rng.shuffle(train)

    train_widths = {x["d"] for x in train}
    allowed_test_outcomes = (["equal", "below"] +
                             (["low_equal"] if n_train >= 3 else []) +
                             (["above"] if n_train >= 4 else []))
    test_outcomes = [str(rng.choice(allowed_test_outcomes))
                     for _ in range(n_test)]
    if n_test >= 2:
        test_outcomes[0] = "below"
        test_outcomes[1] = "equal" if n_train <= 3 else str(rng.choice(("equal", "above")))
    rng.shuffle(test_outcomes)
    test = []
    for outcome in test_outcomes:
        if bool(rng.integers(2)):
            opposite = [x["d"] for x in train if x["outcome"] != outcome]
            d = int(rng.choice(opposite)) if opposite else int(rng.choice(tuple(train_widths)))
        else:
            held_out = [d for d in range(1, 14) if d not in train_widths]
            d = int(rng.choice(held_out)) if held_out else _delimiter(rng)
        spec = _spec(rng, "target_test", delimiter=d, outcome=outcome)
        spec["counts"] = _choose_block_counts(rng, spec)
        if outcome == "below" and (n_test >= 2 or bool(rng.integers(2))):
            spec["misleading"] = True
        test.append(spec)

    puzzle = {"train": [], "test": []}
    used = set()
    for split, specs in (("train", train), ("test", test)):
        for index, spec in enumerate(specs):
            colors = shared_colors if stable_palette else _palette(rng)
            # Pair RNGs are distinct deterministic streams. Extremely
            # unlikely collisions use a bounded redraw with new streams.
            for attempt in range(12):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                pair = _make_pair(pair_rng, spec, colors)
                key = _grid_key(pair["input"])
                if key not in used and key not in _SOURCE_INPUT_KEYS:
                    break
            else:
                raise RuntimeError("unable to make a distinct pair")
            used.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple((split, tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                               for p in puzzle[split])) for split in ("train", "test"))


def self_test() -> None:
    for seed, train, test in ((1000, 2, 1), (1001, 10, 1), (1002, 10, 4),
                              (1100, None, 1), (1101, None, 4),
                              (1102, 2, None), (1103, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 2, "num_test": 3}):
        try:
            generate_puzzle(1200, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid pair counts accepted")
    assert _puzzle_key(generate_puzzle(123, 6, 2)) == _puzzle_key(generate_puzzle(123, 6, 2))
    assert _puzzle_key(generate_puzzle(123)) != _puzzle_key(generate_puzzle(124))

    train_counts, test_counts, shapes, palettes, patterns = set(), set(), set(), set(), set()
    widths, outcomes = set(), set()
    large_counts = zero_blocks = misleading_train = misleading_test = 0
    test_seen = test_held = mixed_palettes = wide_pairs = 0
    low_train = low_test = near_capacity = sparse = empty_tail = 0
    same_width_opposite_decisions = 0
    per_seed = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        per_seed.add(_puzzle_key(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        infos = {split: [_inspect(p) for p in puzzle[split]] for split in ("train", "test")}
        train_widths = {x["d"] for x in infos["train"]}
        mixed_palettes += len({x["roles"] for x in infos["train"]}) > 1
        same_width_opposite_decisions += any(
            a["d"] == b["d"] and
            (a["outcome"] in ("below", "low_equal")) !=
            (b["outcome"] in ("below", "low_equal"))
            for i, a in enumerate(infos["train"])
            for b in infos["train"][i + 1:])
        for split in ("train", "test"):
            for x in infos[split]:
                shapes.add((x["h"], x["w"]))
                palettes.add(x["roles"])
                patterns.add((x["d"], x["reference"], x["candidate"], x["later"]))
                widths.add(x["d"])
                outcomes.add((split, x["outcome"]))
                large_counts += max(x["reference"], x["candidate"]) >= x["h"]
                near_capacity += max(x["reference"], x["candidate"]) >= 0.8 * x["h"] * x["d"]
                sparse += max(x["reference"], x["candidate"]) <= 5
                empty_tail += x["later"] == 0
                zero_blocks += x["candidate"] == 0 and x["later"] > 0
                wide_pairs += x["d"] > 5
                if x["outcome"] == "below" and x["candidate"] + x["later"] >= x["reference"]:
                    if split == "train":
                        misleading_train += 1
                    else:
                        misleading_test += 1
                if split == "test":
                    test_seen += x["d"] in train_widths
                    test_held += x["d"] not in train_widths
                    low_test += x["outcome"] == "low_equal"
                else:
                    low_train += x["outcome"] == "low_equal"
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(per_seed) == 42 and len(shapes) >= 70 and len(patterns) >= 150
    assert len(palettes) >= 30 and len(widths) >= 10
    assert {("test", o) for o in ("equal", "below", "above")} <= outcomes
    assert large_counts >= 10 and zero_blocks >= 3 and wide_pairs >= 25
    assert misleading_train >= 20 and misleading_test >= 5
    assert test_seen >= 10 and test_held >= 10 and mixed_palettes >= 5
    assert low_train >= 20 and low_test >= 2
    assert near_capacity >= 3 and sparse >= 50 and empty_tail >= 5
    assert same_width_opposite_decisions >= 8
