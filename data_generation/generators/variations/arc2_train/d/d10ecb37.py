"""Generate mosaics of rotated 2-by-2 motifs."""

import numpy as np


def _digit_grid(*rows: str) -> np.ndarray:
    return np.array([[int(c) for c in row] for row in rows], dtype=np.int8)


_SOURCE_PAIRS = (
    (_digit_grid("24225245", "25544222", "45522224", "22425425",
                 "24225245", "25544222", "45522224", "22425425"),
     _digit_grid("24", "25")),
    (_digit_grid("436406", "600334", "644330", "036046", "063043", "344660"),
     _digit_grid("43", "60")),
    (_digit_grid("321341", "144223", "133224", "421431", "412432", "233114",
                 "244113", "312342", "321341", "144223", "133224", "421431"),
     _digit_grid("32", "14")),
    (_digit_grid("96299269", "29966992", "69922996", "92699629"),
     _digit_grid("96", "29")),
)


def _check_count(name: str, value: int | None, low: int, high: int) -> None:
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")


def _resolve_counts(seed: int, rng: np.random.Generator,
                    num_train: int | None, num_test: int | None) -> tuple[int, int]:
    _check_count("num_train", num_train, 2, 10)
    _check_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        test_count = 1 + seed % 4
        train_count = test_count + 1 + (seed // 4) % (10 - test_count)
    elif num_train is None:
        test_count = int(num_test)
        train_count = int(rng.integers(test_count + 1, 11))
    elif num_test is None:
        train_count = int(num_train)
        test_count = int(rng.integers(1, min(4, train_count - 1) + 1))
    else:
        train_count, test_count = int(num_train), int(num_test)
    if train_count <= test_count:
        raise ValueError("num_train must exceed num_test")
    return train_count, test_count


def _sample_motif(rng: np.random.Generator, color_count: int,
                  repeat_kind: str | None = None) -> np.ndarray:
    colors = rng.choice(10, size=color_count, replace=False).astype(np.int8)
    rng.shuffle(colors)
    if color_count == 4:
        return colors.reshape(2, 2)
    flat = np.empty(4, dtype=np.int8)
    if color_count == 3:
        if repeat_kind is None:
            repeat_kind = "adjacent" if rng.integers(2) else "diagonal"
        positions = ((0, 1), (0, 2), (1, 3), (2, 3)) if repeat_kind == "adjacent" else ((0, 3), (1, 2))
        duplicate = positions[int(rng.integers(len(positions)))]
        others = [p for p in range(4) if p not in duplicate]
        flat[list(duplicate)] = colors[0]
        flat[others] = colors[1:]
    elif color_count == 2:
        kind = repeat_kind or ("three_one", "stripe", "checker")[int(rng.integers(3))]
        if kind == "three_one":
            flat[:] = colors[0]
            flat[int(rng.integers(4))] = colors[1]
        else:
            positions = ((0, 1), (0, 2)) if kind == "stripe" else ((0, 3), (1, 2))
            first = positions[int(rng.integers(len(positions)))]
            flat[:] = colors[1]
            flat[list(first)] = colors[0]
    else:
        raise ValueError("motif must have two, three, or four colors")
    return flat.reshape(2, 2)


def _sample_shape(rng: np.random.Generator, regime: str) -> tuple[int, int]:
    # Both splits overlap. Original-like wide tests receive a modest preference.
    weights = ([22, 14, 15, 15, 22, 12] if regime == "source_train"
               else [24, 12, 25, 15, 18, 6])
    mode = int(rng.choice(6, p=np.array(weights) / 100))
    if mode == 0:
        choices = ((3, 3), (4, 4), (6, 3), (2, 4), (2, 5), (5, 3))
        return choices[int(rng.integers(len(choices)))]
    if mode == 1:
        side = int(rng.integers(2, 16))
        return side, side
    if mode == 2:
        rows = int(rng.integers(2, 10))
        return rows, int(rng.integers(rows + 1, 16))
    if mode == 3:
        cols = int(rng.integers(2, 10))
        return int(rng.integers(cols + 1, 16)), cols
    if mode == 4:
        return int(rng.integers(2, 16)), int(rng.integers(2, 16))
    return int(rng.integers(2, 6)), int(rng.integers(2, 6))


def _visible_turns(motif: np.ndarray) -> list[int]:
    representatives = []
    for turns in range(4):
        rotated = np.rot90(motif, turns)
        if not any(np.array_equal(rotated, np.rot90(motif, other))
                   for other in representatives):
            representatives.append(turns)
    return representatives


def _rotation_map(rng: np.random.Generator, rows: int, cols: int,
                  motif: np.ndarray, regime: str, force_four: bool = False) -> np.ndarray:
    representatives = _visible_turns(motif)
    if len(representatives) == 2:
        count = 2
    elif force_four:
        count = 4
    else:
        weights = ([18, 24, 58] if regime == "source_train" else [16, 21, 63])
        count = int(rng.choice((2, 3, 4), p=np.array(weights) / 100))
    other_turns = representatives[1:]
    rng.shuffle(other_turns)
    chosen_turns = [0] + other_turns[:count - 1]
    rr, cc = np.indices((rows, cols))
    style = int(rng.integers(8))
    if style == 0:  # short periodic arithmetic
        a, b = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        labels = (a * rr + b * cc) % count
    elif style == 1:  # longer periods and unequal bands
        a, b = int(rng.integers(2, 6)), int(rng.integers(2, 6))
        labels = (rr // a + cc // b + cc % b) % count
    elif style == 2:  # horizontal or vertical bands
        width = int(rng.integers(1, 5))
        labels = ((rr if rng.integers(2) else cc) // width) % count
    elif style == 3:  # independent row and column phase shifts
        row_phase = rng.integers(0, count, size=rows)
        col_phase = rng.integers(0, count, size=cols)
        labels = (row_phase[:, None] + col_phase[None, :]) % count
    elif style == 4:  # locally irregular, with unequal frequencies
        weights = rng.dirichlet(np.ones(count) * 1.5)
        labels = rng.choice(count, size=(rows, cols), p=weights)
    elif style == 5:  # coarse compartments with a changing phase
        height, width = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        labels = (rr // height + 2 * (cc // width) + rr % height) % count
    elif style == 6:  # random walk across each row
        labels = np.empty((rows, cols), dtype=np.int8)
        phase = 0
        for row in range(rows):
            phase = (phase + int(rng.integers(count))) % count
            for col in range(cols):
                phase = (phase + int(rng.integers(count))) % count
                labels[row, col] = phase
    else:  # a compact two by two repeating layout
        labels = (2 * (rr % 2) + cc % 2) % count
    labels = np.asarray(labels, dtype=np.int8)
    labels[0, 0] = 0
    # Repair only missing turns; leave complete periodic layouts untouched.
    free = np.arange(1, rows * cols)
    rng.shuffle(free)
    frequencies = np.bincount(labels.ravel(), minlength=count)
    for wanted in range(1, count):
        if frequencies[wanted] == 0:
            for position in free:
                old = int(labels.flat[int(position)])
                if frequencies[old] > 1:
                    labels.flat[int(position)] = wanted
                    frequencies[old] -= 1
                    frequencies[wanted] += 1
                    break
    return np.asarray(chosen_turns, dtype=np.int8)[labels]


def _tile_grid(motif: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    rows, cols = rotations.shape
    grid = np.empty((2 * rows, 2 * cols), dtype=np.int8)
    for row in range(rows):
        for col in range(cols):
            grid[2 * row:2 * row + 2, 2 * col:2 * col + 2] = np.rot90(
                motif, int(rotations[row, col]))
    return grid


def _grid_key(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(v) for v in grid.flat))


def _is_source_pair(inp: np.ndarray, out: np.ndarray) -> bool:
    return any(np.array_equal(inp, si) and np.array_equal(out, so)
               for si, so in _SOURCE_PAIRS)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)

    # The only shared conventions are the fixed tile size and top-left anchor.
    # The source changes colors and motif geometry independently between pairs.
    order = rng.permutation(train_count)
    forced_colors = {int(order[0]): 4, int(order[1]): 3}
    forced_repeat = {}
    if train_count >= 4:
        forced_colors[int(order[2])] = 3
        forced_repeat[int(order[1])] = "adjacent"
        forced_repeat[int(order[2])] = "diagonal"

    puzzle = {"train": [], "test": []}
    used_inputs = set()
    train_shapes = set()
    for split, regime, count in (("train", "source_train", train_count),
                                 ("test", "target_test", test_count)):
        for index in range(count):
            for attempt in range(100):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                colors = (forced_colors[index] if split == "train" and index in forced_colors
                          else int(pair_rng.choice((2, 3, 4), p=(.12, .45, .43))))
                motif = _sample_motif(pair_rng, colors,
                                      forced_repeat.get(index) if split == "train" else None)
                rows, cols = _sample_shape(pair_rng, regime)
                if (split == "train" and index == train_count - 1
                        and train_count >= 3 and len(train_shapes) == 1
                        and (rows, cols) in train_shapes):
                    continue
                rotations = _rotation_map(pair_rng, rows, cols, motif, regime,
                                          force_four=(split == "train" and index == int(order[0])))
                inp = _tile_grid(motif, rotations)
                out = inp[:2, :2].copy()
                key = _grid_key(inp)
                if key in used_inputs or _is_source_pair(inp, out):
                    continue
                used_inputs.add(key)
                puzzle[split].append({"input": inp, "output": out})
                if split == "train":
                    train_shapes.add((rows, cols))
                break
            else:
                raise RuntimeError("unable to construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(inp: np.ndarray, out: np.ndarray) -> tuple[int, int, tuple]:
    assert out.shape == (2, 2)
    assert inp.shape[0] % 2 == inp.shape[1] % 2 == 0
    assert inp.shape[0] >= 4 and inp.shape[1] >= 4
    assert np.array_equal(inp[:2, :2], out)
    assert set(np.unique(inp)) == set(np.unique(out))
    turns = [np.rot90(out, k) for k in range(4)]
    rows, cols = inp.shape[0] // 2, inp.shape[1] // 2
    visible = []
    for row in range(rows):
        for col in range(cols):
            tile = inp[2 * row:2 * row + 2, 2 * col:2 * col + 2]
            matches = [k for k in range(4) if np.array_equal(tile, turns[k])]
            assert matches  # Symmetric motifs may match more than one turn.
            visible.append(_grid_key(tile))
    assert len(set(visible)) >= 2
    return rows, cols, tuple(visible)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    train_counts, train_visible, train_shapes = set(), set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            key = _grid_key(inp)
            assert key not in seen
            seen.add(key)
            assert not _is_source_pair(inp, out)
            rows, cols, visible = _analyze_pair(inp, out)
            assert 2 <= len(np.unique(out)) <= 4
            if split == "train":
                train_counts.add(len(np.unique(out)))
                train_visible.add(len(set(visible)))
                train_shapes.add((rows, cols))
    # A four-color example with all turns and a repeated-color example
    # jointly demonstrate the fixed crop rule in every training set.
    assert {3, 4} <= train_counts
    assert 4 in train_visible
    if len(puzzle["train"]) >= 3:
        assert len(train_shapes) >= 2


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def _motif_kind(motif: np.ndarray) -> str:
    values, counts = np.unique(motif, return_counts=True)
    if len(values) == 4:
        return "four"
    if len(values) == 3:
        repeated = values[counts == 2][0]
        positions = set(int(i) for i in np.flatnonzero(motif.ravel() == repeated))
        return "three_diagonal" if positions in ({0, 3}, {1, 2}) else "three_adjacent"
    if 3 in counts:
        return "two_three_one"
    return "two_checker" if motif[0, 0] == motif[1, 1] else "two_stripe"


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    kinds = {"train": set(), "test": set()}
    turns = {"train": set(), "test": set()}
    shapes = {"train": set(), "test": set()}
    layouts = set()
    wide_train = tall_test = mixed_repeat_puzzles = 0
    short_periodic = other_large = unequal = combined = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        repeated = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                rows, cols, visible = _analyze_pair(inp, out)
                assert np.array_equal(out, inp[:2, :2])
                kind = _motif_kind(out)
                kinds[split].add(kind)
                if kind.startswith("three_"):
                    repeated.add(kind)
                turns[split].add(len(set(visible)))
                shapes[split].add((rows, cols))
                ids = {}
                labels = np.empty((rows, cols), dtype=np.int8)
                for position, tile in enumerate(visible):
                    if tile not in ids:
                        ids[tile] = len(ids)
                    labels.flat[position] = ids[tile]
                layouts.add((rows, cols, tuple(int(v) for v in labels.flat)))
                frequencies = np.bincount(labels.ravel())
                unequal += int(len(set(int(v) for v in frequencies)) > 1)
                if rows >= 4 and cols >= 4:
                    horizontal = any(np.array_equal(labels[:, period:], labels[:, :-period])
                                     for period in range(1, 4))
                    vertical = any(np.array_equal(labels[period:, :], labels[:-period, :])
                                   for period in range(1, 4))
                    short_periodic += int(horizontal and vertical)
                    other_large += int(not (horizontal and vertical))
                combined += int(split == "test" and cols > rows
                                and kind == "three_adjacent" and len(ids) == 4)
                wide_train += int(split == "train" and cols > rows)
                tall_test += int(split == "test" and rows > cols)
        mixed_repeat_puzzles += int({"three_adjacent", "three_diagonal"} <= repeated)
    assert len(shapes["train"] | shapes["test"]) >= 35
    assert len(layouts) >= 120
    assert short_periodic >= 5 and other_large >= 20
    assert unequal >= 50 and combined >= 1
    assert wide_train >= 10 and tall_test >= 5
    assert {"three_adjacent", "three_diagonal", "four"} <= kinds["train"]
    assert {"three_adjacent", "three_diagonal", "four"} <= kinds["test"]
    assert {"two_three_one", "two_stripe", "two_checker"} <= kinds["train"] | kinds["test"]
    assert {2, 3, 4} <= turns["test"]
    assert mixed_repeat_puzzles >= 10
    for train_count, test_count in ((2, 1), (10, 1), (10, 4), (3, 1), (6, 2)):
        p = generate_puzzle(3000 + train_count + test_count, train_count, test_count)
        assert (len(p["train"]), len(p["test"])) == (train_count, test_count)
    for seed, train_count, test_count in ((4001, None, 1), (4002, None, 4),
                                          (4003, 2, None), (4004, 10, None)):
        p = generate_puzzle(seed, train_count, test_count)
        assert train_count is None or len(p["train"]) == train_count
        assert test_count is None or len(p["test"]) == test_count
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(5000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
    assert _puzzle_key(generate_puzzle(987654, 7, 3)) == _puzzle_key(generate_puzzle(987654, 7, 3))
    assert len({_puzzle_key(generate_puzzle(2000 + i, 6, 2)) for i in range(10)}) >= 9
