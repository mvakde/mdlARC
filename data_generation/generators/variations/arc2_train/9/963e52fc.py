"""Generate ARC puzzles that continue short horizontal row motifs to double width."""

import numpy as np


_SOURCE_INPUTS = (
    ("0000000", "0000000", "2332332", "0000000", "0000000"),
    ("000000", "282828", "282828", "000000", "000000"),
    ("00000000", "00000000", "12212212", "21221221", "00000000"),
    ("000000000", "311311311", "311311311", "000000000", "000000000"),
)
_SOURCE_KEYS = frozenset(
    (len(rows), len(rows[0]), tuple(int(c) for row in rows for c in row))
    for rows in _SOURCE_INPUTS
)


def _key(grid):
    return (grid.shape[0], grid.shape[1], tuple(int(x) for x in grid.flat))


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _counts(seed, rng, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _period(row, limit=None):
    width = len(row)
    if limit is None:
        limit = width // 2
    for period in range(1, min(limit, width - 1) + 1):
        if np.array_equal(row[period:], row[:-period]):
            return period
    return None


def _motif(rng, period, colors, background, gap):
    symbols = list(colors)
    if gap:
        symbols.append(background)
    assert len(symbols) <= period and len(symbols) >= 2
    for _ in range(32):
        values = symbols + [int(rng.choice(symbols)) for _ in range(period - len(symbols))]
        rng.shuffle(values)
        motif = np.asarray(values, dtype=np.int8)
        if _period(motif, period - 1) is None:
            return motif
    motif = np.full(period, symbols[0], dtype=np.int8)
    for index, symbol in enumerate(symbols[1:], 1):
        motif[-index] = symbol
    assert _period(motif, period - 1) is None
    return motif


def _choose_height(rng, minimum):
    if int(rng.integers(0, 4)) == 0:
        return int(rng.integers(max(minimum, 15), 31))
    return int(rng.integers(max(minimum, 4), 15)) if minimum < 15 else minimum


def _choose_colors(rng, background, count, familiar=(), fresh=(), mix=False):
    available = [c for c in range(10) if c != background]
    selected = []
    if mix and familiar and fresh and count >= 2:
        selected = [int(rng.choice(familiar)), int(rng.choice(fresh))]
    remaining = [c for c in available if c not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[:count - len(selected)])
    return tuple(selected)


def _build_pair(rng, mode, seen_train_colors):
    if mode == "basic":
        period = int(rng.integers(2, 5))
        foreground_count, gap = 2, False
        occupied_count = int(rng.integers(1, 3))
    elif mode == "phase":
        period = int(rng.integers(4, 7))
        foreground_count, gap = 3, True
        occupied_count = int(rng.integers(2, 5))
    elif mode == "challenge":
        period = int(rng.integers(3, 7))
        foreground_count, gap = 2, True
        occupied_count = int(rng.integers(2, 5))
    else:
        period = int(rng.integers(2, 7))
        gap = bool(rng.integers(0, 2))
        minimum_colors = 1 if gap else 2
        foreground_count = int(rng.integers(minimum_colors, min(4, period - int(gap)) + 1))
        occupied_count = int(rng.integers(1, 7))

    widths = [w for w in range(2 * period, 16) if mode not in ("basic", "challenge") or w % period]
    width = int(rng.choice(widths))
    minimum_height = 2 * occupied_count + 1 if mode == "challenge" else occupied_count + 1
    height = _choose_height(rng, minimum_height)
    occupied_count = min(occupied_count, height - 1)
    if mode == "challenge":
        positions = np.sort(rng.choice(height - occupied_count + 1, occupied_count, replace=False))
        positions = positions + np.arange(occupied_count)
    else:
        positions = np.sort(rng.choice(height, occupied_count, replace=False))

    background = 0 if int(rng.integers(0, 2)) == 0 else int(rng.integers(1, 10))
    familiar = [c for c in seen_train_colors if c != background]
    fresh = [c for c in range(10) if c not in seen_train_colors and c != background]
    colors = _choose_colors(rng, background, foreground_count, familiar, fresh,
                            mix=mode == "challenge")
    first = _motif(rng, period, colors, background, gap)
    motifs = [first]
    for index in range(1, occupied_count):
        if index == 1 and mode in ("phase", "challenge"):
            motifs.append(np.roll(first, int(rng.integers(1, period))).copy())
            continue
        relation = int(rng.integers(0, 4))
        if relation == 0:
            motifs.append(first.copy())
        elif relation == 1:
            motifs.append(np.roll(first, int(rng.integers(1, period))).copy())
        else:
            other_period = period if relation == 2 else int(rng.integers(2, min(6, width // 2) + 1))
            other_gap = bool(rng.integers(0, 2))
            color_limit = min(len(colors), other_period - int(other_gap))
            if color_limit == 1 and not other_gap:
                other_gap = True
            chosen = tuple(int(c) for c in rng.choice(colors, size=color_limit, replace=False))
            motifs.append(_motif(rng, other_period, chosen, background, other_gap))

    inp = np.full((height, width), background, dtype=np.int8)
    out = np.full((height, 2 * width), background, dtype=np.int8)
    for row_index, motif in zip(positions, motifs):
        inp[int(row_index)] = motif[np.arange(width) % len(motif)]
        out[int(row_index)] = motif[np.arange(2 * width) % len(motif)]
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _count(seed, "seed", 0, 2**63 - 1)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)
    # Direction, width multiplier, and rowwise period continuation are shared.
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    train_colors = set()
    for section, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            mode = ("basic" if index == 0 else "phase" if index == 1 else "free") if section == "train" else ("challenge" if index == 0 else "free")
            for _ in range(128):
                pair = _build_pair(rng, mode, train_colors)
                key = _key(pair["input"])
                if key in seen_inputs or key in _SOURCE_KEYS:
                    continue
                if section == "test" and index == test_count - 1:
                    all_pairs = puzzle["train"] + puzzle["test"]
                    if len({p["input"].shape[0] for p in all_pairs + [pair]}) < 2:
                        continue
                    if len({p["input"].shape[1] for p in all_pairs + [pair]}) < 2:
                        continue
                break
            else:
                raise RuntimeError("could not construct a distinct pair")
            seen_inputs.add(key)
            puzzle[section].append(pair)
            if section == "train":
                train_colors.update(int(c) for c in np.unique(pair["input"]))
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_ok(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert all(1 <= side <= 30 for side in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((0 <= grid) & (grid <= 9))


def _profile(pair):
    inp, out = pair["input"], pair["output"]
    height, width = inp.shape
    assert out.shape == (height, 2 * width)
    assert 4 <= width <= 15
    assert np.array_equal(out[:, :width], inp)
    assert set(int(c) for c in np.unique(out)) == set(int(c) for c in np.unique(inp))
    constant = [r for r in range(height) if np.all(inp[r] == inp[r, 0])]
    assert constant
    background = int(inp[constant[0], 0])
    assert all(int(inp[r, 0]) == background for r in constant)
    patterned = [r for r in range(height) if r not in constant]
    assert patterned and len(patterned) <= 8
    assert len(set(int(c) for c in np.unique(inp)) - {background}) <= 4
    periods = {}
    gap = False
    for r in range(height):
        period = _period(inp[r])
        assert period is not None and period <= 6
        assert np.array_equal(out[r], inp[r, np.arange(2 * width) % period])
        if r in patterned:
            assert period >= 2
            periods[r] = period
            gap |= bool(np.any(inp[r] == background))
        else:
            assert np.all(out[r] == background)
    return {"background": background, "rows": patterned, "periods": periods,
            "gap": gap, "foreground": len(set(int(c) for c in np.unique(inp)) - {background}),
            "seam": not np.array_equal(out[:, width:], inp),
            "unequal": len({tuple(int(c) for c in inp[r]) for r in patterned}) > 1}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    profiles = {"train": [], "test": []}
    heights, widths = set(), set()
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            _grid_ok(pair["input"])
            _grid_ok(pair["output"])
            inp_key, out_key = _key(pair["input"]), _key(pair["output"])
            assert inp_key not in seen_inputs and (inp_key, out_key) not in seen_pairs
            assert inp_key not in _SOURCE_KEYS
            seen_inputs.add(inp_key)
            seen_pairs.add((inp_key, out_key))
            heights.add(pair["input"].shape[0])
            widths.add(pair["input"].shape[1])
            profiles[section].append(_profile(pair))
    assert len(heights) >= 2 and len(widths) >= 2
    assert any(p["seam"] for p in profiles["train"])
    assert any(p["foreground"] >= 3 and p["gap"] for p in profiles["train"])
    phase_evidence = False
    for pair, profile in zip(puzzle["train"], profiles["train"]):
        rows = profile["rows"]
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = pair["input"][rows[i]], pair["input"][rows[j]]
                period = profile["periods"][rows[i]]
                if period == profile["periods"][rows[j]] and any(
                    np.array_equal(np.roll(a[:period], shift), b[:period])
                    for shift in range(1, period)
                ):
                    phase_evidence = True
    assert phase_evidence
    assert any(p["seam"] for p in profiles["test"])
    assert any(p["gap"] and p["unequal"] and len(p["rows"]) >= 2 and
               any(b - a > 1 for a, b in zip(p["rows"], p["rows"][1:]))
               for p in profiles["test"])


def _equal_puzzles(first, second):
    return all(
        len(first[s]) == len(second[s]) and all(
            np.array_equal(a[role], b[role]) and a[role].dtype == b[role].dtype
            for a, b in zip(first[s], second[s]) for role in ("input", "output")
        ) for s in ("train", "test")
    )


def _independent_check(puzzle):
    for section in ("train", "test"):
        for pair in puzzle[section]:
            inp, out = pair["input"], pair["output"]
            prediction = np.empty_like(out)
            for r, row in enumerate(inp):
                candidates = [p for p in range(1, len(row) // 2 + 1)
                              if all(row[c] == row[c - p] for c in range(p, len(row)))]
                assert candidates
                prediction[r] = row[np.arange(out.shape[1]) % candidates[0]]
            assert np.array_equal(prediction, out)


def self_test() -> None:
    train_counts, test_counts = set(), set()
    signatures, structures, dimensions = set(), set(), set()
    periods, backgrounds, foreground_counts = set(), set(), set()
    seam_count = gap_count = separated_count = unequal_test_count = 0
    short_test = equal_test = long_test = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _independent_check(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(tuple(tuple((_key(p["input"]), _key(p["output"]))
                                   for p in puzzle[s]) for s in ("train", "test")))
        train_widths = [p["input"].shape[1] for p in puzzle["train"]]
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inp = pair["input"]
                profile = _profile(pair)
                dimensions.add(inp.shape)
                backgrounds.add(profile["background"])
                foreground_counts.add(profile["foreground"])
                seam_count += profile["seam"]
                gap_count += profile["gap"]
                separated_count += any(b - a > 1 for a, b in zip(profile["rows"], profile["rows"][1:]))
                for row_index, period in profile["periods"].items():
                    periods.add(period)
                    structures.add((inp.shape, tuple(profile["rows"]),
                                    tuple(int(c) for c in inp[row_index])))
                if section == "test":
                    unequal_test_count += len({tuple(inp[r]) for r in profile["rows"]}) > 1
                    width = inp.shape[1]
                    short_test += width < min(train_widths)
                    equal_test += width in train_widths
                    long_test += width > max(train_widths)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) >= 40 and len(structures) >= 100
    assert len(dimensions) >= 40 and len(periods) >= 5
    assert len(backgrounds) >= 6 and len(foreground_counts) >= 3
    assert min(seam_count, gap_count, separated_count, unequal_test_count) > 0
    assert min(short_test, equal_test, long_test) > 0
    for train, test in ((2, 1), (10, 1), (10, 4), (6, 3),
                        (None, 1), (None, 4), (2, None), (10, None)):
        first = generate_puzzle(987654 + (train or 0) + (test or 0), train, test)
        second = generate_puzzle(987654 + (train or 0) + (test or 0), train, test)
        assert _equal_puzzles(first, second)
        validate_generated_puzzle(first)
        _independent_check(first)
        assert train is None or len(first["train"]) == train
        assert test is None or len(first["test"]) == test
    assert not _equal_puzzles(generate_puzzle(13579, 5, 2), generate_puzzle(13580, 5, 2))
    corrupted = generate_puzzle(0, 2, 1)
    seam_pair = next(pair for pair in corrupted["train"]
                     if not np.array_equal(pair["output"][:, pair["input"].shape[1]:], pair["input"]))
    width = seam_pair["input"].shape[1]
    seam_pair["output"][:, width:] = seam_pair["input"]
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("whole-width restart was accepted")
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (4, 4), (3, 4)):
        try:
            generate_puzzle(40000, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts were accepted")
