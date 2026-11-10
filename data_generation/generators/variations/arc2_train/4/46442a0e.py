import numpy as np


_TURNS = (0, -1, 1, 2)
_LEGAL_COUNTS = tuple(
    (train, test)
    for test in range(1, 5)
    for train in range(test + 1, 11)
    if train >= 2
)
_SOURCE_INPUTS = (
    ((7, 7, 8), (7, 7, 8), (8, 8, 8)),
    ((8, 6), (6, 8)),
    ((6, 9, 9), (6, 4, 4), (6, 4, 4)),
    ((1, 4, 1), (4, 9, 4), (9, 1, 9)),
)
_SOURCE_KEYS = frozenset((len(rows), tuple(v for row in rows for v in row))
                         for rows in _SOURCE_INPUTS)


def _key(grid):
    return grid.shape, tuple(int(v) for v in grid.flat)


def _pinwheel(grid):
    top = np.concatenate((grid, np.rot90(grid, -1)), axis=1)
    bottom = np.concatenate((np.rot90(grid, 1), np.rot90(grid, 2)), axis=1)
    return np.concatenate((top, bottom), axis=0)


def _views(grid):
    rotations = [np.rot90(grid, turn) for turn in range(4)]
    reflection = np.fliplr(grid)
    return rotations + [np.rot90(reflection, turn) for turn in range(4)]


def _view_count(grid):
    return len({_key(view) for view in _views(grid)})


def _counts(seed, rng, num_train, num_test):
    for name, value, low, high in (
        ("num_train", num_train, 2, 10),
        ("num_test", num_test, 1, 4),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high
        ):
            raise ValueError(name + " is outside its legal range")
    if num_train is None and num_test is None:
        return _LEGAL_COUNTS[seed % len(_LEGAL_COUNTS)]
    if num_train is None:
        return int(rng.integers(max(2, int(num_test) + 1), 11)), int(num_test)
    if num_test is None:
        return int(num_train), int(rng.integers(1, min(4, int(num_train) - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _side(rng, low, high):
    if low == high:
        return low
    roll = rng.random()
    if roll < 0.13:
        return low
    if roll < 0.26:
        return high
    return int(rng.integers(low, high + 1))


def _base_labels(rng, side, colors, style):
    shape = (side, side)
    if style == 0:
        return rng.integers(0, colors, size=shape, dtype=np.int8)
    if style == 1:
        result = np.zeros(shape, dtype=np.int8)
        if colors > 1:
            chance = float(rng.uniform(0.015, 0.4))
            mask = rng.random(shape) < chance
            result[mask] = rng.integers(1, colors, size=int(mask.sum()), dtype=np.int8)
        return result
    if style == 2:
        result = np.empty(shape, dtype=np.int8)
        bands = int(rng.integers(1, min(side, 7) + 1))
        cuts = sorted(int(v) for v in rng.choice(np.arange(1, side), size=bands - 1, replace=False)) if bands > 1 else []
        starts = [0] + cuts
        ends = cuts + [side]
        for start, end in zip(starts, ends):
            result[start:end, :] = int(rng.integers(colors))
        if rng.random() < 0.5:
            result = result.T.copy()
        return result
    if style == 3:
        height = int(rng.integers(1, max(2, side // 3 + 1)))
        width = int(rng.integers(1, max(2, side // 3 + 1)))
        row, column = np.indices(shape)
        return np.asarray((row // height + column // width) % colors, dtype=np.int8)
    if style == 4:
        period = int(rng.integers(1, min(side, 5) + 1))
        strips = rng.integers(0, colors, size=(period, side), dtype=np.int8)
        result = strips[np.arange(side) % period, :]
        if rng.random() < 0.5:
            result = result.T.copy()
        return result
    result = np.zeros(shape, dtype=np.int8)
    if style == 5:
        for _ in range(int(rng.integers(1, max(3, side // 2 + 2)))):
            r0 = int(rng.integers(side))
            c0 = int(rng.integers(side))
            r1 = int(rng.integers(r0 + 1, side + 1))
            c1 = int(rng.integers(c0 + 1, side + 1))
            result[r0:r1, c0:c1] = int(rng.integers(colors))
    elif style == 6:
        for inset in range(int(rng.integers(1, max(2, (side + 1) // 2 + 1)))):
            end = side - inset
            if inset >= end:
                break
            color = int(rng.integers(colors))
            result[inset, inset:end] = color
            result[end - 1, inset:end] = color
            result[inset:end, inset] = color
            result[inset:end, end - 1] = color
        if side > 3 and rng.random() < 0.5:
            result = np.roll(result, (int(rng.integers(-2, 3)), int(rng.integers(-2, 3))), axis=(0, 1))
    elif style == 7:
        for _ in range(int(rng.integers(1, 4))):
            row = int(rng.integers(side))
            col = int(rng.integers(side))
            row_end = int(rng.integers(side))
            col_end = int(rng.integers(side))
            color = int(rng.integers(colors))
            result[min(row, row_end):max(row, row_end) + 1, col] = color
            result[row_end, min(col, col_end):max(col, col_end) + 1] = color
    else:
        for _ in range(int(rng.integers(2, max(4, side + 2)))):
            row = int(rng.integers(side))
            col = int(rng.integers(side))
            height = int(rng.integers(1, min(4, side - row) + 1))
            width = int(rng.integers(1, min(4, side - col) + 1))
            result[row:row + height, col:col + width] = int(rng.integers(colors))
    return result


def _orbits(side, symmetry):
    unseen = {(r, c) for r in range(side) for c in range(side)}
    groups = []
    while unseen:
        row, col = min(unseen)
        if symmetry == 0:
            group = {(row, col)}
        elif symmetry == 1:
            group = {(row, col), (row, side - 1 - col)}
        elif symmetry == 2:
            group = {(row, col), (side - 1 - row, col)}
        elif symmetry == 3:
            group = {(row, col), (side - 1 - row, side - 1 - col)}
        else:
            group = set()
            rr, cc = row, col
            for _ in range(4):
                group.add((rr, cc))
                if symmetry == 5:
                    group.add((rr, side - 1 - cc))
                rr, cc = cc, side - 1 - rr
        unseen.difference_update(group)
        groups.append(sorted(group))
    return groups


def _labels(rng, side, requested_colors, style, symmetry):
    groups = _orbits(side, symmetry)
    colors = min(requested_colors, len(groups))
    base = _base_labels(rng, side, colors, style)
    if rng.random() < 0.32 and side >= 3:
        overlay = _base_labels(rng, side, colors, int(rng.choice((5, 6, 7, 8))))
        mask = overlay != 0
        base[mask] = overlay[mask]
    if rng.random() < 0.5:
        base = np.rot90(base, int(rng.integers(4))).copy()
    values = [int(base[group[0]]) for group in groups]
    for missing in range(colors):
        if missing in values:
            continue
        donors = [i for i, value in enumerate(values) if values.count(value) > 1]
        index = int(rng.choice(donors))
        values[index] = missing
    result = np.empty((side, side), dtype=np.int8)
    for group, value in zip(groups, values):
        for row, col in group:
            result[row, col] = value
    return result


def _make_input(rng, side, witness=False):
    for attempt in range(64 if witness else 1):
        if witness:
            colors = int(rng.integers(2, min(6, side * side) + 1))
            symmetry = 0
        else:
            colors = int(rng.choice(
                np.arange(1, 11),
                p=(0.08, 0.16, 0.15, 0.12, 0.10, 0.09, 0.08, 0.08, 0.07, 0.07),
            ))
            symmetry = int(rng.choice(6, p=(0.46, 0.12, 0.12, 0.10, 0.10, 0.10)))
        style = int(rng.integers(9))
        labels = _labels(rng, side, colors, style, symmetry)
        if witness and _view_count(labels) != 8:
            continue
        palette = rng.choice(10, size=int(np.unique(labels).size), replace=False)
        grid = np.asarray(palette[labels], dtype=np.int8)
        if _key(grid) not in _SOURCE_KEYS:
            return grid
    if witness:
        for _ in range(64):
            labels = rng.integers(0, 4, size=(side, side), dtype=np.int8)
            if _view_count(labels) == 8:
                palette = rng.choice(10, size=4, replace=False)
                grid = np.asarray(palette[labels], dtype=np.int8)
                if _key(grid) not in _SOURCE_KEYS:
                    return grid
    raise RuntimeError("unable to construct a distinct motif")


def _manual_view(grid, turn, reflected):
    side = grid.shape[0]
    result = np.empty_like(grid)
    for row in range(side):
        for col in range(side):
            rr, cc = row, col
            if reflected:
                cc = side - 1 - cc
            for _ in range(turn):
                rr, cc = cc, side - 1 - rr
            result[rr, cc] = grid[row, col]
    return result


def _prediction_from_training(train, test_input):
    candidates = [set(range(8)) for _ in range(4)]
    for pair in train:
        inp, out = pair["input"], pair["output"]
        side = inp.shape[0]
        views = [_manual_view(inp, turn, reflected)
                 for reflected in (False, True) for turn in range(4)]
        for quadrant, (row, col) in enumerate(((0, 0), (0, side), (side, 0), (side, side))):
            block = out[row:row + side, col:col + side]
            candidates[quadrant].intersection_update(
                index for index, view in enumerate(views) if np.array_equal(block, view)
            )
    assert all(len(options) == 1 for options in candidates), "training does not identify all quadrant operations"
    views = [_manual_view(test_input, turn, reflected)
             for reflected in (False, True) for turn in range(4)]
    selected = [views[next(iter(options))] for options in candidates]
    return np.block([[selected[0], selected[1]], [selected[2], selected[3]]])


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)
    profile = int(rng.integers(5))
    train_range, test_range = (
        ((1, 15), (1, 15)),
        ((1, 7), (1, 7)),
        ((5, 15), (5, 15)),
        ((2, 8), (9, 15)),
        ((5, 15), (1, 4)),
    )[profile]
    witness_index = int(rng.integers(train_count))
    train_sides = [_side(rng, *train_range) for _ in range(train_count)]
    train_sides[witness_index] = _side(rng, max(3, train_range[0]), train_range[1])
    if train_count > 1 and len(set(train_sides)) == 1:
        other = (witness_index + 1) % train_count
        options = [v for v in range(train_range[0], train_range[1] + 1) if v != train_sides[other]]
        train_sides[other] = int(rng.choice(options))
    test_sides = [_side(rng, *test_range) for _ in range(test_count)]
    puzzle = {"train": [], "test": []}
    seen = set()
    for section, sides in (("train", train_sides), ("test", test_sides)):
        for index, side in enumerate(sides):
            for _ in range(64):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                inp = _make_input(pair_rng, side, section == "train" and index == witness_index)
                key = _key(inp)
                if key not in seen:
                    seen.add(key)
                    puzzle[section].append({"input": inp, "output": _pinwheel(inp)})
                    break
            else:
                raise RuntimeError("unable to construct unique puzzle inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[part], list) for part in ("train", "test"))
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    for section in (train, test):
        for pair in section:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= dimension <= 30 for dimension in grid.shape)
                assert bool(np.all((grid >= 0) & (grid <= 9)))
            assert inp.shape[0] == inp.shape[1] and inp.shape[0] <= 15
            side = inp.shape[0]
            assert out.shape == (2 * side, 2 * side)
            key = _key(inp)
            assert key not in seen and key not in _SOURCE_KEYS
            seen.add(key)
            assert np.array_equal(out, _pinwheel(inp)), "wrong quadrant rotations"
            assert set(np.unique(out).tolist()) == set(np.unique(inp).tolist())
            for quadrant, (row, col) in enumerate(((0, 0), (0, side), (side, 0), (side, side))):
                expected = _manual_view(inp, (0, 1, 3, 2)[quadrant], False)
                assert np.array_equal(out[row:row + side, col:col + side], expected)
    assert any(_view_count(pair["input"]) == 8 for pair in train), "missing asymmetric training witness"
    for pair in test:
        predicted = _prediction_from_training(train, pair["input"])
        assert np.array_equal(predicted, pair["output"]), "test is not recoverable from demonstrations"


def _fingerprint(puzzle):
    return tuple(tuple((_key(pair["input"]), _key(pair["output"]))
                       for pair in puzzle[part]) for part in ("train", "test"))


def _canonical_key(grid):
    labels = {}
    normalized = []
    for value in grid.flat:
        color = int(value)
        if color not in labels:
            labels[color] = len(labels)
        normalized.append(labels[color])
    return grid.shape, tuple(normalized)


def self_test() -> None:
    sample = generate_puzzle(12345)
    assert _fingerprint(sample) == _fingerprint(generate_puzzle(12345))
    for train_count, test_count in ((2, 1), (10, 1), (10, 4), (7, 3)):
        puzzle = generate_puzzle(7000 + train_count + test_count, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
        validate_generated_puzzle(puzzle)
    for index, (train_count, test_count) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(8000 + index, train_count, test_count)
        assert train_count is None or len(puzzle["train"]) == train_count
        assert test_count is None or len(puzzle["test"]) == test_count
    for train_count, test_count in ((1, None), (11, None), (None, 0), (None, 5),
                                    (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(9999, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    recolored_source_test = np.array(((2, 5, 2), (5, 7, 5), (7, 2, 7)), dtype=np.int8)
    source_symmetry_probe = generate_puzzle(0, 2, 1)
    source_symmetry_probe["test"][0] = {
        "input": recolored_source_test,
        "output": _pinwheel(recolored_source_test),
    }
    validate_generated_puzzle(source_symmetry_probe)
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_fingerprint(puzzle) for puzzle in puzzles}) >= 40
    assert len({len(puzzle["train"]) for puzzle in puzzles}) >= 4
    assert len({len(puzzle["test"]) for puzzle in puzzles}) >= 3
    grids = [pair["input"] for puzzle in puzzles for part in ("train", "test")
             for pair in puzzle[part]]
    sides = {grid.shape[0] for grid in grids}
    colors = {int(np.unique(grid).size) for grid in grids}
    orbits = {_view_count(grid) for grid in grids}
    assert {1, 2, 3, 4, 5, 10, 15}.issubset(sides)
    assert {1, 2, 3, 4}.issubset(colors) and max(colors) >= 9
    assert {1, 2, 4, 8}.issubset(orbits)
    assert len({_key(grid) for grid in grids}) > len(grids) * 0.9
    assert len({_canonical_key(grid) for grid in grids}) >= 250
    for part in ("train", "test"):
        section = [pair["input"] for puzzle in puzzles for pair in puzzle[part]]
        assert set(range(1, 16)).issubset({grid.shape[0] for grid in section})
        assert set(range(1, 11)).issubset({np.unique(grid).size for grid in section})
        assert {1, 2, 4, 8}.issubset({_view_count(grid) for grid in section})
        assert any(grid.shape == (2, 2) and np.unique(grid).size == 4 for grid in section)
        assert any(np.unique(grid).size > 1 and
                   min(np.unique(grid, return_counts=True)[1]) == 1 for grid in section)
    assert sum(len({pair["input"].shape[0] for pair in puzzle["train"]}) > 1
               for puzzle in puzzles) >= 35
    assert sum(min(pair["input"].shape[0] for pair in puzzle["test"]) >
               max(pair["input"].shape[0] for pair in puzzle["train"])
               for puzzle in puzzles) >= 5
    assert sum(max(pair["input"].shape[0] for pair in puzzle["test"]) <
               min(pair["input"].shape[0] for pair in puzzle["train"])
               for puzzle in puzzles) >= 5
    assert sum(bool({np.unique(pair["input"]).size for pair in puzzle["test"]} -
                    {np.unique(pair["input"]).size for pair in puzzle["train"]})
               for puzzle in puzzles) >= 15
    assert sum(bool({_view_count(pair["input"]) for pair in puzzle["test"]} -
                    {_view_count(pair["input"]) for pair in puzzle["train"]})
               for puzzle in puzzles) >= 8
    transition_rates = []
    dominant_shares = []
    for grid in grids:
        side = grid.shape[0]
        transitions = (np.count_nonzero(grid[1:, :] != grid[:-1, :]) +
                       np.count_nonzero(grid[:, 1:] != grid[:, :-1]))
        transition_rates.append(float(transitions) / (2 * side * (side - 1)) if side > 1 else 0.0)
        dominant_shares.append(float(max(np.unique(grid, return_counts=True)[1])) / grid.size)
    assert sum(rate < 0.2 for rate in transition_rates) >= 50
    assert sum(rate > 0.6 for rate in transition_rates) >= 50
    assert sum(share >= 0.8 for share in dominant_shares) >= 50
    assert sum(share < 0.4 for share in dominant_shares) >= 40
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for pair in puzzle["test"]:
            assert np.array_equal(_prediction_from_training(puzzle["train"], pair["input"]), pair["output"])
