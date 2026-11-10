"""Synthetic row-local ARC puzzles with learned unordered compatible color pairs."""

import numpy as np


_SOURCE_STRING_PAIRS = (
    (("0080000900", "0060000700", "0700000004", "0002040000", "0000100008", "0030009000", "6000000400", "0040070000", "0000008010", "0000000000"),
     ("0080000900", "0060000700", "0740000000", "0002040000", "0000180000", "0030009000", "6000000400", "0047000000", "0000008100", "0000000000")),
    (("40700", "09002", "00104"), ("47000", "09002", "00104")),
    (("0804000", "4000007", "0010080", "0900400"), ("0804000", "4700000", "0018000", "0900400")),
    (("00000000", "01080000", "00600007", "00040700", "30004000", "02000900", "00000000"),
     ("00000000", "01800000", "00600007", "00047000", "30004000", "02000900", "00000000")),
    (("0704", "6080", "8010", "0403"), ("0740", "6080", "8100", "0403")),
)


def _grid_key(grid):
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


def _source_key(rows):
    return (tuple((len(rows), len(rows[0]))), tuple(int(x) for row in rows for x in row))


_SOURCE_PAIR_KEYS = frozenset((_source_key(a), _source_key(b)) for a, b in _SOURCE_STRING_PAIRS)


def _key(a, b):
    a, b = int(a), int(b)
    return (a, b) if a < b else (b, a)


def _counts(seed, num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = max(2, num_test + 1) + (seed // 4) % (11 - max(2, num_test + 1))
    elif num_train is None:
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _parameters(rng):
    colors = [int(x) for x in rng.permutation(10)]
    nclasses = 3 if rng.random() < 0.35 else 2
    mates = tuple(_key(colors[2*i], colors[2*i+1]) for i in range(nclasses))
    other = colors[2*nclasses:]
    nbackgrounds = 2 if rng.random() < 0.55 else 1
    backgrounds = tuple(other[-nbackgrounds:])
    distractors = tuple(other[:-nbackgrounds])
    cross = [_key(a, b) for i, first in enumerate(mates) for second in mates[i+1:] for a in first for b in second]
    rng.shuffle(cross)
    cross = tuple(cross[:(4 if nclasses == 3 else 3)])
    mixed = tuple(_key(mate[int(rng.integers(0, 2))], distractors[i % len(distractors)]) for i, mate in enumerate(mates))
    nonmate = _key(distractors[0], distractors[1])
    negatives = tuple(dict.fromkeys(cross + mixed + (nonmate,)))
    return mates, backgrounds, negatives, cross, mixed, nonmate


def _dimension(rng, width=False, test=False):
    p = float(rng.random())
    if width:
        edges = (0.32 if test else 0.25, 0.66, 0.86)
        bounds = ((4, 6), (7, 11), (12, 20), (21, 30))
    else:
        edges = (0.30 if test else 0.20, 0.65, 0.84)
        bounds = ((2, 4), (5, 9), (10, 17), (18, 30))
    bucket = sum(p >= edge for edge in edges)
    lo, hi = bounds[bucket]
    return int(rng.integers(lo, hi + 1))


def _background(grid):
    values, counts = np.unique(grid, return_counts=True)
    top = np.flatnonzero(counts == counts.max())
    assert len(top) == 1, "background must be visually identifiable"
    return int(values[int(top[0])])


def _apply_rule(grid, background, mates):
    output = grid.copy()
    for r, row in enumerate(grid):
        cols = np.flatnonzero(row != background)
        if len(cols) != 2:
            continue
        left, right = (int(c) for c in cols)
        if _key(row[left], row[right]) in mates and right > left + 1:
            output[r, right] = background
            output[r, left + 1] = row[right]
    return output


def _distance(rng, width, must_separate=False, hint=0):
    if hint == 2:
        return 2
    if hint == 4:
        return int(rng.integers(4, width))
    if not must_separate and rng.random() < 0.15:
        return 1
    return int(rng.integers(2, width))


def _build_pair(rng, background, mates, negatives, mandatory, test=False, unchanged=False):
    width = _dimension(rng, width=True, test=test)
    if any(spec[2] == 4 for spec in mandatory):
        width = max(width, 8)
    height = max(_dimension(rng, test=test), len(mandatory))
    if len(mandatory) >= height and height < 30:
        height += int(rng.integers(0, min(4, 30-height) + 1))
    if test:
        occupied = int(rng.integers(1, height + 1))
    else:
        occupied = min(height, len(mandatory) + int(rng.integers(0, min(6, height-len(mandatory)) + 1)))
        occupied = max(1, occupied)
    specs = list(mandatory)
    while len(specs) < occupied:
        if unchanged or rng.random() < 0.48:
            key = negatives[int(rng.integers(0, len(negatives)))]
        else:
            key = mates[int(rng.integers(0, len(mates)))]
        oriented = key if rng.random() < 0.5 else key[::-1]
        specs.append((oriented[0], oriented[1], 0, False))
    rng.shuffle(specs)
    grid = np.full((height, width), background, dtype=np.int8)
    rows = rng.permutation(height)[:len(specs)]
    for r, (left_color, right_color, hint, must_separate) in zip(rows, specs):
        distance = _distance(rng, width, must_separate, hint)
        left = int(rng.integers(0, width-distance))
        grid[int(r), left] = left_color
        grid[int(r), left+distance] = right_color
    return {"input": grid, "output": _apply_rule(grid, background, mates)}


def _witnesses(mates, cross, mixed, nonmate, rng):
    specs = []
    for i, mate in enumerate(mates):
        specs.append((mate[0], mate[1], 2 if i == 0 else 0, True))
        specs.append((mate[1], mate[0], 4 if i == 0 else 0, True))
    for i, key in enumerate(cross):
        specs.append((key[0], key[1], 2 if i == 0 else 0, True))
        if i < 2:
            specs.append((key[1], key[0], 4 if i == 0 else 0, True))
    for key in mixed + (nonmate,):
        oriented = key if rng.random() < 0.5 else key[::-1]
        specs.append((oriented[0], oriented[1], 0, True))
    rng.shuffle(specs)
    groups = [[], []]
    for spec in specs:
        groups[int(rng.integers(0, 2))].append(spec)
    if not groups[0]:
        groups[0].append(groups[1].pop())
    if not groups[1]:
        groups[1].append(groups[0].pop())
    return groups


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    ntrain, ntest = _counts(int(seed), num_train, num_test)
    rng = np.random.default_rng(int(seed))
    mates, backgrounds, negatives, cross, mixed, nonmate = _parameters(rng)
    groups = _witnesses(mates, cross, mixed, nonmate, rng)
    puzzle = {"train": [], "test": []}
    used = set()
    train_backgrounds = [backgrounds[i % len(backgrounds)] for i in range(ntrain)]
    test_backgrounds = [backgrounds[int(rng.integers(0, len(backgrounds)))] for _ in range(ntest)]
    for split, count, bgs in (("train", ntrain, train_backgrounds), ("test", ntest, test_backgrounds)):
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63-1)))
            mandatory = groups[index] if split == "train" and index < 2 else []
            unchanged = (split == "test" or index >= 2) and pair_rng.random() < 0.18
            if split == "test" and not unchanged:
                mate = mates[int(pair_rng.integers(0, len(mates)))]
                oriented = mate if pair_rng.random() < 0.5 else mate[::-1]
                mandatory = [(oriented[0], oriented[1], 2, True)]
            for _ in range(32):
                pair = _build_pair(pair_rng, bgs[index], mates, negatives, mandatory, test=split == "test", unchanged=unchanged)
                key = _grid_key(pair["input"])
                both = (key, _grid_key(pair["output"]))
                if key not in used and both not in _SOURCE_PAIR_KEYS:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise ValueError("could not construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _training_evidence(train):
    positive = set()
    negative = set()
    positive_order = {}
    negative_order = {}
    for pair in train:
        bg = _background(pair["input"])
        for row, outrow in zip(pair["input"], pair["output"]):
            cols = np.flatnonzero(row != bg)
            if len(cols) != 2:
                continue
            left, right = map(int, cols)
            key = _key(row[left], row[right])
            if right == left + 1:
                continue
            if np.array_equal(row, outrow):
                negative.add(key)
                negative_order.setdefault(key, set()).add((int(row[left]), int(row[right])))
            else:
                positive.add(key)
                positive_order.setdefault(key, set()).add((int(row[left]), int(row[right])))
    return positive, negative, positive_order, negative_order


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    backgrounds = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= x <= 30 for x in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            assert inp.shape == out.shape
            bg = _background(inp)
            backgrounds.add(bg)
            assert _background(out) == bg
            key = _grid_key(inp)
            assert key not in seen
            seen.add(key)
            assert (key, _grid_key(out)) not in _SOURCE_PAIR_KEYS
            for row in inp:
                colors = row[row != bg]
                assert len(colors) in (0, 2)
                if len(colors) == 2:
                    assert int(colors[0]) != int(colors[1])
    positive, negative, positive_order, negative_order = _training_evidence(puzzle["train"])
    assert 2 <= len(positive) <= 3 and not (positive & negative)
    mate_colors = [color for key in positive for color in key]
    assert len(mate_colors) == len(set(mate_colors))
    assert all(len(positive_order[key]) == 2 for key in positive)
    mate_set = set(mate_colors)
    cross = {key for key in negative if set(key) <= mate_set}
    mixed = {key for key in negative if len(set(key) & mate_set) == 1}
    nonmate = {key for key in negative if not (set(key) & mate_set)}
    assert len(cross) >= 2 and len(mixed) >= len(positive) and nonmate
    assert sum(len(negative_order[key]) == 2 for key in cross) >= 2
    assert not (backgrounds & mate_set)
    for split in ("train", "test"):
        for pair in puzzle[split]:
            inp, out = pair["input"], pair["output"]
            bg = _background(inp)
            assert np.array_equal(_apply_rule(inp, bg, positive), out)
            for row in inp:
                cols = np.flatnonzero(row != bg)
                if len(cols) == 2:
                    left, right = map(int, cols)
                    key = _key(row[left], row[right])
                    if right > left + 1 and key not in positive:
                        assert key in negative
    # Prediction uses only visible training examples and each test input.
    for pair in puzzle["test"]:
        inp = pair["input"].copy()
        inferred = _apply_rule(inp, _background(inp), positive)
        assert np.array_equal(inferred, pair["output"])


def _fingerprint(puzzle):
    return tuple((split, tuple((_grid_key(p["input"]), _grid_key(p["output"])) for p in puzzle[split])) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_fingerprint(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    for p in puzzles:
        validate_generated_puzzle(p)
    for seed, ntrain, ntest in ((1100, 2, 1), (1101, 10, 1), (1102, 10, 4), (1200, None, 1), (1201, None, 4), (1202, 2, None), (1203, 10, None)):
        p = generate_puzzle(seed, ntrain, ntest)
        validate_generated_puzzle(p)
        assert ntrain is None or len(p["train"]) == ntrain
        assert ntest is None or len(p["test"]) == ntest
    assert _fingerprint(generate_puzzle(2027, 7, 3)) == _fingerprint(generate_puzzle(2027, 7, 3))
    assert _fingerprint(generate_puzzle(2028)) == _fingerprint(generate_puzzle(2028))
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4), (2, 4), (True, None), (None, False)):
        try:
            generate_puzzle(19, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid counts: {args}")
    shapes, layouts, backgrounds, class_counts, moving, test_heights, test_widths = set(), set(), set(), set(), set(), set(), set()
    adjacent_positive = adjacent_negative = empty_rows = cross_controls = large_test = 0
    for p in puzzles:
        positive, negative, _, _ = _training_evidence(p["train"])
        class_counts.add(len(positive))
        colors = {x for pair in positive for x in pair}
        for split in ("train", "test"):
            for pair in p[split]:
                inp, out = pair["input"], pair["output"]
                bg = _background(inp)
                backgrounds.add(bg)
                shapes.add(inp.shape)
                layouts.add((inp.shape, tuple(tuple(int(c) for c in np.flatnonzero(row != bg)) for row in inp)))
                count = 0
                for row, outrow in zip(inp, out):
                    cols = np.flatnonzero(row != bg)
                    if len(cols) == 0:
                        empty_rows += 1
                        continue
                    left, right = map(int, cols)
                    key = _key(row[left], row[right])
                    if not np.array_equal(row, outrow):
                        count += 1
                    if right == left+1:
                        if key in positive:
                            adjacent_positive += 1
                        else:
                            adjacent_negative += 1
                    if right > left+1 and key in negative and set(key) <= colors:
                        cross_controls += 1
                        assert np.array_equal(row, outrow)
                moving.add(count)
                if split == "test":
                    test_heights.add(inp.shape[0]); test_widths.add(inp.shape[1])
                    large_test += inp.shape[0] >= 12 or inp.shape[1] >= 12
    assert len(shapes) >= 35 and len(layouts) >= 150 and len(backgrounds) >= 8
    assert class_counts == {2, 3} and {0, 1, 2, 3} <= moving
    assert len(test_heights) >= 12 and len(test_widths) >= 12 and large_test >= 10
    assert adjacent_positive >= 5 and adjacent_negative >= 5 and empty_rows >= 20
    assert cross_controls >= 84  # At least two witnessed hard negatives per puzzle.

    # These counts come from the returned grids, rather than sampled parameters.
    switching_backgrounds = compact_tests = dense_tests = sparse_tests = 0
    small_tests = large_tests = reduced_tests = rich_tests = 0
    repeated_train = absent_train = unchanged_train = unchanged_test = 0
    long_train = long_test = cross_train = cross_test = 0
    for puzzle in puzzles:
        mates, negatives, _, _ = _training_evidence(puzzle["train"])
        mate_colors = {c for key in mates for c in key}
        local_backgrounds = set()
        shortcut_fails = False
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                bg = _background(inp)
                local_backgrounds.add(bg)
                h, w = inp.shape
                if split == "test":
                    compact_tests += (h, w) == (4, 4)
                    small_tests += h <= 4 and w <= 6
                    large_tests += h >= 12 or w >= 12
                    reduced_tests += len(np.unique(inp)) < 7
                    rich_tests += len(np.unique(inp)) > 7
                empty = sum(np.all(row == bg) for row in inp)
                if split == "test":
                    dense_tests += empty == 0
                    sparse_tests += empty >= h // 2
                counts = {key: 0 for key in mates}
                changes = 0
                for row, outrow in zip(inp, out):
                    cols = np.flatnonzero(row != bg)
                    if len(cols) != 2:
                        continue
                    left, right = map(int, cols)
                    key = _key(row[left], row[right])
                    if key in counts:
                        counts[key] += 1
                    changed = not np.array_equal(row, outrow)
                    changes += changed
                    if changed and right-left >= 6:
                        if split == "train":
                            long_train += 1
                        else:
                            long_test += 1
                    if right > left+1 and key in negatives and set(key) <= mate_colors:
                        if split == "train":
                            cross_train += 1
                            # The palette-membership shortcut moves this row incorrectly.
                            shortcut_fails = True
                        else:
                            cross_test += 1
                        assert not changed
                if split == "train":
                    repeated_train += any(n >= 2 for n in counts.values())
                    absent_train += any(n == 0 for n in counts.values())
                    unchanged_train += changes == 0
                else:
                    unchanged_test += changes == 0
        switching_backgrounds += len(local_backgrounds) > 1
        assert shortcut_fails
    assert switching_backgrounds >= 10 and compact_tests >= 1
    assert dense_tests >= 5 and sparse_tests >= 20
    assert small_tests >= 5 and large_tests >= 20
    assert reduced_tests >= 30 and rich_tests >= 5
    assert repeated_train >= 40 and absent_train >= 80
    assert unchanged_train >= 20 and unchanged_test >= 10
    assert long_train >= 50 and long_test >= 20
    assert cross_train >= 84 and cross_test >= 20
