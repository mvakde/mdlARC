"""Four-pass, endpoint-overlapped vertical bounce puzzle generator."""

import numpy as np


_AUTO_COUNTS = ((2, 1), (3, 1), (4, 2), (5, 1), (6, 2), (6, 3),
                (7, 1), (8, 3), (9, 4), (10, 1), (10, 2), (10, 4))
_SOURCE_INPUTS = (
    ("00200020002000200", "02020202020202020",
     "02020202020202020", "20002000200020002"),
    ("00800080008000800", "08080808080808080", "80008000800080008"),
    ("00300030003000300", "03030303030303030",
     "03030303030303030", "03030303030303030", "30003000300030003"),
)


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return result


def _seed(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError("seed must be a non-negative integer")
    return result


def _counts(seed, rng, num_train, num_test):
    train = None if num_train is None else _integer(num_train, "num_train", 2, 10)
    test = None if num_test is None else _integer(num_test, "num_test", 1, 4)
    if train is None and test is None:
        return _AUTO_COUNTS[seed % len(_AUTO_COUNTS)]
    if train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    if test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _bounce_output(grid):
    return np.concatenate((grid, grid[-2::-1], grid[1:], grid[-2::-1]), axis=0)


def _row_bank(rng, width, count, coherent, special):
    """Build distinct rows with one short shared period or irregular motifs."""
    if coherent:
        period = int(rng.integers(4 if count > 6 else 3, min(9, width // 2) + 1))
    rows, seen = [], set()
    columns = np.arange(width)
    for _ in range(1500):
        if coherent:
            tile = rng.random(period) < float(rng.uniform(0.2, 0.8))
            row = tile[columns % period]
        else:
            row = rng.random(width) < float(rng.uniform(0.18, 0.82))
            if bool(rng.integers(0, 3) == 0):
                row = np.repeat(row[::2], 2)[:width]
        if not (np.any(row) and np.any(~row)):
            continue
        key = row.tobytes()
        if key not in seen:
            rows.append(row.astype(np.uint8))
            seen.add(key)
            if len(rows) == count:
                break
    if len(rows) != count:
        raise ValueError("could not construct distinct row motifs")
    if special and count >= 4:
        rows[2] = np.full(width, int(rng.integers(0, 2)), dtype=np.uint8)
    return rows


def _binary_band(rng, height, width, mode, coherent, special):
    if mode == "nonadjacent" and height < 5:
        mode = "distinct"
    if mode == "adjacent" and height < 4:
        mode = "distinct"
    bank = _row_bank(rng, width, height, coherent, special)
    middle = list(range(2, height))
    if mode == "adjacent":
        position = int(rng.integers(1, len(middle)))
        middle[position] = middle[position - 1]
        if len(middle) >= 4 and bool(rng.integers(0, 2)):
            position = int(rng.integers(1, len(middle)))
            middle[position] = middle[position - 1]
    elif mode == "nonadjacent":
        middle[-1] = middle[0]
    return np.stack([bank[index] for index in [0] + middle + [1]])


def _heights(rng, seed, train_count, test_count):
    # Every puzzle teaches four distinct rows; shift cycles between longer,
    # comparable, and shorter tests without changing the transformation.
    shift = (seed // 12) % 3
    train = [4, 3 if shift == 0 else (4 if shift == 1 else 5)]
    low, high = ((3, 5), (3, 8), (4, 9))[shift]
    train.extend(int(rng.integers(low, high)) for _ in range(train_count - 2))
    if shift == 0:
        test = [int(rng.choice((5, 5, 5, 6, 7, 8)))]
        test.extend(int(rng.integers(5, 9)) for _ in range(test_count - 1))
    elif shift == 1:
        test = [int(rng.choice(train))]
        test.extend(int(rng.integers(3, 9)) for _ in range(test_count - 1))
    else:
        test = [2 if bool(rng.integers(0, 5) == 0) else 3]
        test.extend(int(rng.integers(3, 8)) for _ in range(test_count - 1))
    return train, test


def _colors(rng, reused_motif=None):
    background = 0 if rng.random() < 0.55 else int(rng.integers(1, 10))
    if reused_motif is not None and reused_motif != background:
        motif = reused_motif
    else:
        motif = int(rng.choice([color for color in range(10) if color != background]))
    return background, motif


def _grid_key(grid):
    return (grid.shape, grid.astype(np.int16, copy=False).tobytes())


def _source_copy(inp, out):
    for rows in _SOURCE_INPUTS:
        source = np.array([[int(char) for char in row] for row in rows], dtype=np.int8)
        if np.array_equal(inp, source) and np.array_equal(out, _bounce_output(source)):
            return True
    return False


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _seed(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)
    train_heights, test_heights = _heights(rng, seed, train_count, test_count)

    # The sole puzzle-level convention is the fixed vertical bounce. Colors,
    # widths, row patterns, and band lengths are visible pair-level choices.
    widths = [int(rng.integers(8, 31)) for _ in range(train_count + test_count)]
    if len(set(widths)) == 1:
        widths[1] = 8 + (widths[0] - 7) % 23
    pair_seeds = rng.integers(0, 2**63, size=len(widths), dtype=np.int64)
    puzzle = {"train": [], "test": []}
    used, train_motifs = set(), []
    coherent_first = bool(rng.integers(0, 2))

    for split, heights in (("train", train_heights), ("test", test_heights)):
        for index, height in enumerate(heights):
            absolute = index if split == "train" else train_count + index
            pair_rng = np.random.default_rng(int(pair_seeds[absolute]))
            if split == "train" and index == 0:
                mode, coherent = "distinct", coherent_first
            elif split == "train" and index == 1:
                mode, coherent = ("adjacent" if height >= 4 else "distinct"), not coherent_first
            else:
                mode = ("distinct", "adjacent", "nonadjacent")[int(pair_rng.integers(0, 3))]
                coherent = bool(pair_rng.random() < 0.62)
            special = height >= 4 and bool(pair_rng.random() < 0.13)
            reuse = None
            if split == "test" and pair_rng.random() < 0.5:
                reuse = int(pair_rng.choice(train_motifs))
            background, motif = _colors(pair_rng, reuse)
            if split == "train":
                train_motifs.append(motif)
            for _ in range(80):
                binary = _binary_band(pair_rng, height, widths[absolute], mode, coherent, special)
                inp = np.where(binary, motif, background).astype(np.int8)
                out = _bounce_output(inp)
                key = _grid_key(inp)
                if key not in used and not _source_copy(inp, out):
                    used.add(key)
                    puzzle[split].append({"input": inp, "output": out})
                    break
            else:
                raise ValueError("could not construct a unique source-free pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _distinct_rows(grid):
    return len({row.tobytes() for row in grid})


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, widths = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= size <= 30 for size in grid.shape)
                assert bool(np.all((grid >= 0) & (grid <= 9)))
            height, width = inp.shape
            assert 2 <= height <= 8 and 8 <= width <= 30
            assert out.shape == (4 * height - 3, width)
            assert np.array_equal(out, _bounce_output(inp))
            assert not np.array_equal(inp, out)
            assert len(np.unique(inp)) == 2
            assert not np.array_equal(inp[0], inp[-1])
            assert _distinct_rows(inp) >= min(3, height)
            assert not _source_copy(inp, out)
            key = _grid_key(inp)
            assert key not in seen
            seen.add(key)
            widths.add(width)
    assert any(pair["input"].shape[0] >= 4 and _distinct_rows(pair["input"]) >= 4
               for pair in puzzle["train"])
    assert len(widths) >= 2


def _has_short_common_period(grid):
    width = grid.shape[1]
    return any(np.array_equal(grid[:, :-period], grid[:, period:])
               for period in range(2, min(8, width // 2) + 1))


def _row_class(grid):
    rows = [row.tobytes() for row in grid[1:-1]]
    if len(rows) >= 3 and rows[0] == rows[-1] and rows[0] != rows[1]:
        return "nonadjacent"
    if any(a == b for a, b in zip(rows, rows[1:])):
        return "adjacent"
    return "distinct"


def self_test() -> None:
    first, second = generate_puzzle(9001), generate_puzzle(9002)
    assert any(not np.array_equal(a["input"], b["input"])
               for a, b in zip(first["train"], second["train"]))
    for seed, train, test in ((101, 2, 1), (102, 10, 1), (103, 10, 4),
                              (201, None, 1), (202, None, 4),
                              (203, 2, None), (204, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
        validate_generated_puzzle(puzzle)
    for request in ((1, None), (11, None), (None, 0), (None, 5),
                    (2, 2), (3, 4), (True, 1), (3, False)):
        try:
            generate_puzzle(300, *request)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count was accepted")

    train_counts, test_counts, heights, widths, structures = set(), set(), set(), set(), set()
    shift_cases, classes = set(), set()
    short_period = zero_absent = source_width = special_rows = motif_reuse = 0
    varied_width_puzzles = 0
    for seed in range(42):
        puzzle, again = generate_puzzle(seed), generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert all(np.array_equal(a["input"], b["input"]) and
                   np.array_equal(a["output"], b["output"])
                   for split in ("train", "test")
                   for a, b in zip(puzzle[split], again[split]))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_heights = [p["input"].shape[0] for p in puzzle["train"]]
        test_heights = [p["input"].shape[0] for p in puzzle["test"]]
        if min(test_heights) > max(train_heights):
            shift_cases.add("longer")
        if any(h in train_heights for h in test_heights):
            shift_cases.add("same")
        if min(test_heights) < min(train_heights):
            shift_cases.add("shorter")
        all_pairs = puzzle["train"] + puzzle["test"]
        # Infer the traversal count from demonstration dimensions alone.
        walks = {(p["output"].shape[0] - 1) // (p["input"].shape[0] - 1)
                 for p in puzzle["train"]}
        assert walks == {4}
        assert all(p["output"].shape[0] == 1 + 4 * (p["input"].shape[0] - 1)
                   for p in puzzle["train"])
        all_widths = {p["input"].shape[1] for p in all_pairs}
        varied_width_puzzles += len(all_widths) > 1
        train_colors = {int(color) for p in puzzle["train"] for color in np.unique(p["input"])}
        test_colors = {int(color) for p in puzzle["test"] for color in np.unique(p["input"])}
        motif_reuse += bool(train_colors & test_colors)
        for pair in all_pairs:
            inp, out = pair["input"], pair["output"]
            h, w = inp.shape
            period = 2 * (h - 1)
            predicted = np.stack([inp[min(r % period, period - r % period)]
                                  for r in range(1 + next(iter(walks)) * (h - 1))])
            assert np.array_equal(predicted, out)
            heights.add(h)
            widths.add(w)
            structures.add((inp.shape, (inp == inp[0, 0]).tobytes()))
            classes.add(_row_class(inp))
            short_period += _has_short_common_period(inp)
            zero_absent += 0 not in np.unique(inp)
            source_width += w == 17
            special_rows += any(len(np.unique(row)) == 1 for row in inp)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(heights) >= 6 and len(widths) >= 18 and 17 in widths
    assert len(structures) >= 150
    assert shift_cases == {"longer", "same", "shorter"}
    assert classes == {"distinct", "adjacent", "nonadjacent"}
    assert short_period >= 35 and zero_absent >= 10 and source_width >= 4
    assert special_rows >= 5 and motif_reuse >= 10
    assert varied_width_puzzles == 42
