"""Generate ARC grids by duplicating both outer rows and outer columns."""

import numpy as np


_IDX = [0, 0, 1, 2, 2]
_SOURCE = {
    (1, 3, 5, 1, 2, 8, 8, 3, 8),
    (6, 5, 5, 5, 1, 7, 4, 5, 2),
    (2, 3, 7, 2, 1, 6, 1, 5, 7),
    (1, 2, 5, 7, 3, 6, 7, 6, 5),
}
_COUNTS = (
    (3, 1), (3, 1), (2, 1), (3, 1),
    (4, 1), (3, 1), (4, 2), (2, 1),
    (5, 2), (6, 3), (8, 3), (10, 4),
)
_MODES = (
    "random", "random", "random", "mono", "solid_row",
    "solid_col", "repeat_rows", "repeat_cols", "mirror",
    "diagonal", "center", "center", "corners", "row_pair",
    "row_pair", "row_pair", "col_pair", "col_pair", "distinct",
    "distinct", "border_pair", "source_like",
)


def _expand(grid: np.ndarray) -> np.ndarray:
    return grid[np.ix_(_IDX, _IDX)].astype(np.int8, copy=True)


def _counts(seed: int, rng: np.random.Generator,
            train: int | None, test: int | None) -> tuple[int, int]:
    for name, value, lo, hi in (("num_train", train, 2, 10),
                                ("num_test", test, 1, 4)):
        if value is not None:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
            if not lo <= int(value) <= hi:
                raise ValueError(f"{name} is outside {lo}..{hi}")
    if train is None and test is None:
        train, test = _COUNTS[seed % len(_COUNTS)]
    elif train is None:
        test = int(test)
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        train = int(train)
        test = int(rng.integers(1, min(4, train - 1) + 1))
    else:
        train, test = int(train), int(test)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _random_pattern(rng: np.random.Generator, colors: int) -> np.ndarray:
    labels = np.concatenate((np.arange(colors), rng.integers(0, colors, 9 - colors)))
    rng.shuffle(labels)
    return labels.reshape(3, 3)


def _reduce(rng: np.random.Generator, labels: np.ndarray, count: int) -> np.ndarray:
    labels = labels.copy()
    while len(np.unique(labels)) > count:
        a, b = rng.choice(np.unique(labels), size=2, replace=False)
        labels[labels == a] = b
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse.reshape(3, 3)


def _pattern(rng: np.random.Generator, mode: str) -> np.ndarray:
    if mode == "mono":
        return np.zeros((3, 3), dtype=np.int8)
    if mode == "distinct":
        return rng.permutation(9).reshape(3, 3)
    if mode == "random":
        k = int(rng.choice([2, 3, 4, 5, 5, 6, 6, 7, 8, 9]))
        return _random_pattern(rng, k)

    a = np.arange(9).reshape(3, 3)
    if mode == "solid_row":
        row = int(rng.integers(3))
        a[row, :] = a[row, 0]
    elif mode == "solid_col":
        col = int(rng.integers(3))
        a[:, col] = a[0, col]
    elif mode == "repeat_rows":
        x, y = rng.choice(3, size=2, replace=False)
        a[y, :] = a[x, :]
    elif mode == "repeat_cols":
        x, y = rng.choice(3, size=2, replace=False)
        a[:, y] = a[:, x]
    elif mode == "mirror":
        a[:, 2] = a[:, 0]
    elif mode == "diagonal":
        a[1, 1] = a[2, 2] = a[0, 0]
    elif mode == "center":
        other = int(rng.choice([0, 1, 2, 3, 5, 6, 7, 8]))
        a[1, 1] = a.flat[other]
    elif mode == "corners":
        a[0, 2] = a[2, 0] = a[2, 2] = a[0, 0]
    elif mode == "row_pair":
        row = int(rng.integers(3))
        x, y = rng.choice(3, size=2, replace=False)
        a[row, y] = a[row, x]
    elif mode == "col_pair":
        col = int(rng.integers(3))
        x, y = rng.choice(3, size=2, replace=False)
        a[y, col] = a[x, col]
    elif mode in ("border_pair", "source_like"):
        border = [0, 1, 2, 3, 5, 6, 7, 8]
        choices = [(x, y) for x in border for y in border
                   if x < y and x // 3 != y // 3 and x % 3 != y % 3]
        x, y = choices[int(rng.integers(len(choices)))]
        a.flat[y] = a.flat[x]
    else:
        raise ValueError("unknown mode")

    maximum = len(np.unique(a))
    if mode in ("center", "row_pair", "col_pair", "border_pair"):
        count = maximum
    elif mode == "source_like":
        count = 6
    else:
        count = int(rng.integers(2, maximum + 1))
    return _reduce(rng, a, count)


def _witness(grid: np.ndarray) -> bool:
    return (len({tuple(row) for row in grid}) == 3
            and len({tuple(grid[:, c]) for c in range(3)}) == 3
            and all(len(np.unique(grid[r, :])) > 1 for r in range(3))
            and all(len(np.unique(grid[:, c])) > 1 for c in range(3))
            and any(len(np.unique(row)) == 3 for row in grid)
            and any(len(np.unique(grid[:, c])) == 3 for c in range(3))
            and max(np.unique(grid, return_counts=True)[1]) <= 2)


def _distinct_cell_signatures(first: np.ndarray, second: np.ndarray) -> bool:
    return len({(int(first.flat[i]), int(second.flat[i]))
                for i in range(9)}) == 9


def _sample_pattern(rng: np.random.Generator, maximum: int = 9,
                    exact: int | None = None, witness: bool = False) -> np.ndarray:
    if witness:
        for _ in range(100):
            k = exact if exact is not None else int(rng.integers(5, min(7, maximum) + 1))
            candidate = _random_pattern(rng, k)
            if _witness(candidate):
                return candidate
        raise RuntimeError("could not make a clear training witness")
    for _ in range(300):
        candidate = _pattern(rng, _MODES[int(rng.integers(len(_MODES)))])
        k = len(np.unique(candidate))
        if k <= maximum and (exact is None or k == exact):
            return candidate
    k = exact if exact is not None else int(rng.integers(1, maximum + 1))
    return _random_pattern(rng, k)


def _colors(rng: np.random.Generator, pool, count: int) -> tuple[int, ...]:
    return tuple(int(x) for x in rng.choice(pool, size=count, replace=False))


def _paint(pattern: np.ndarray, palette: tuple[int, ...]) -> np.ndarray:
    labels = np.unique(pattern)
    mapping = {int(label): palette[i] for i, label in enumerate(labels)}
    return np.array([[mapping[int(v)] for v in row] for row in pattern], dtype=np.int8)


def _second_palette(rng: np.random.Generator, pool: tuple[int, ...],
                    first: tuple[int, ...], count: int, relation: str) -> tuple[int, ...]:
    unused = tuple(c for c in pool if c not in first)
    if relation == "disjoint":
        return _colors(rng, unused, count)
    if relation == "identical":
        return tuple(int(x) for x in rng.permutation(first))
    low = max(1, len(first) + count - len(pool))
    high = min(len(first), count) - 1
    shared = int(rng.integers(low, high + 1))
    return _colors(rng, first, shared) + _colors(rng, unused, count - shared)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)

    # Pool and overlap are generation choices, not semantic color roles.
    # Occasionally reproduce the source's cross-pair palette relationship:
    # first two share three colors; train 3 and test 1 have one six-color set.
    source_palette_case = train_count >= 3 and seed % 11 == 0
    relation = ("source_like" if source_palette_case else
                ("disjoint", "partial", "identical")[int(rng.integers(3))])
    pool_minimum = (8 if source_palette_case else 7 if relation == "disjoint"
                    else 6 if relation == "partial" else 5)
    pool_size = int(rng.integers(pool_minimum, 11))
    pool = tuple(int(x) for x in rng.permutation(10)[:pool_size])
    first_max = (min(7, pool_size - 2) if relation == "disjoint"
                 else min(7, pool_size - 1) if relation == "partial"
                 else min(7, pool_size))
    first_pattern = _sample_pattern(
        rng, maximum=first_max, exact=5 if source_palette_case else None, witness=True)
    first_palette = _colors(rng, pool, len(np.unique(first_pattern)))
    if relation == "disjoint":
        for _ in range(1000):
            second_pattern = _sample_pattern(rng, maximum=pool_size - len(first_palette))
            if _distinct_cell_signatures(first_pattern, second_pattern):
                break
        else:
            raise RuntimeError("could not distinguish training cells")
    elif relation in ("identical", "source_like"):
        for _ in range(1000):
            second_pattern = _sample_pattern(
                rng, exact=6 if source_palette_case else len(first_palette))
            if _distinct_cell_signatures(first_pattern, second_pattern):
                break
        else:
            raise RuntimeError("could not distinguish training cells")
    else:
        for _ in range(1000):
            second_pattern = _sample_pattern(rng, maximum=pool_size)
            k = len(np.unique(second_pattern))
            if (max(1, len(first_palette) + k - pool_size) <= min(len(first_palette), k) - 1
                    and _distinct_cell_signatures(first_pattern, second_pattern)):
                break
        else:
            raise RuntimeError("could not distinguish training cells")
    if source_palette_case:
        shared = _colors(rng, first_palette, 3)
        first_only = tuple(c for c in first_palette if c not in shared)
        outside = tuple(c for c in pool if c not in first_palette)
        second_palette = shared + _colors(rng, outside, 3)
        source_palette = (shared + _colors(rng, first_only, 1)
                          + _colors(rng, second_palette[3:], 2))
    else:
        second_palette = _second_palette(
            rng, pool, first_palette, len(np.unique(second_pattern)), relation)

    puzzle: dict[str, list[dict[str, np.ndarray]]] = {"train": [], "test": []}
    used: set[tuple[int, ...]] = set()

    def add(split: str, pattern: np.ndarray, palette: tuple[int, ...]) -> bool:
        grid = _paint(pattern, palette)
        key = tuple(int(x) for x in grid.flat)
        if key in used or key in _SOURCE:
            return False
        used.add(key)
        puzzle[split].append({"input": grid, "output": _expand(grid)})
        return True

    for pattern, palette in ((first_pattern, first_palette), (second_pattern, second_palette)):
        if not add("train", pattern, palette):
            for _ in range(300):
                replacement = _sample_pattern(
                    rng, exact=len(palette), witness=len(puzzle["train"]) == 0)
                other = puzzle["train"][0]["input"] if puzzle["train"] else None
                if (other is None or _distinct_cell_signatures(other, replacement)) and add("train", replacement, palette):
                    break
            else:
                raise RuntimeError("could not make distinct initial training pairs")
    while len(puzzle["train"]) < train_count:
        for _ in range(300):
            pattern = (_sample_pattern(rng, exact=6) if source_palette_case
                       and len(puzzle["train"]) == 2 else
                       _sample_pattern(rng, maximum=min(pool_size, 9)))
            palette = (source_palette if source_palette_case
                       and len(puzzle["train"]) == 2 else
                       _colors(rng, pool, len(np.unique(pattern))))
            if add("train", pattern, palette):
                break
        else:
            raise RuntimeError("could not make a distinct training pair")

    train_palettes = [tuple(int(x) for x in np.unique(p["input"]))
                      for p in puzzle["train"]]
    train_union = set().union(*(set(p) for p in train_palettes))
    unseen = tuple(c for c in range(10) if c not in train_union)
    while len(puzzle["test"]) < test_count:
        for _ in range(300):
            if source_palette_case and not puzzle["test"]:
                pattern = _sample_pattern(rng, exact=6)
                palette = source_palette
                if add("test", pattern, palette):
                    break
                continue
            pattern = _sample_pattern(rng)
            k = len(np.unique(pattern))
            if unseen and rng.random() < 0.35:
                new = int(rng.choice(unseen))
                palette = (new,) + _colors(rng, [c for c in range(10) if c != new], k - 1)
            elif rng.random() < 0.18:
                matching = [p for p in train_palettes if len(p) == k]
                palette = (tuple(int(x) for x in rng.permutation(
                    matching[int(rng.integers(len(matching)))]))
                    if matching else _colors(rng, range(10), k))
            else:
                available = pool if k <= pool_size and rng.random() < 0.55 else range(10)
                palette = _colors(rng, available, k)
            if add("test", pattern, palette):
                break
        else:
            raise RuntimeError("could not make a distinct test pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    n, m = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n <= 10 and 1 <= m <= 4 and n > m
    seen: set[tuple[int, ...]] = set()
    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        source, target = pair["input"], pair["output"]
        for grid in (source, target):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
            assert np.issubdtype(grid.dtype, np.integer)
            assert np.all((grid >= 0) & (grid <= 9))
        assert source.shape == (3, 3) and target.shape == (5, 5)
        assert np.array_equal(target, source[np.ix_(_IDX, _IDX)])
        assert set(np.unique(source)) == set(np.unique(target))
        key = tuple(int(x) for x in source.flat)
        assert key not in seen and key not in _SOURCE
        seen.add(key)
    assert _witness(puzzle["train"][0]["input"])
    assert len({tuple(int(pair["input"].flat[i]) for pair in puzzle["train"])
                for i in range(9)}) == 9


def _canonical(grid: np.ndarray) -> tuple[int, ...]:
    seen: dict[int, int] = {}
    result = []
    for v in grid.flat:
        v = int(v)
        if v not in seen:
            seen[v] = len(seen)
        result.append(seen[v])
    return tuple(result)


def _separated_border(grid: np.ndarray) -> bool:
    border = [0, 1, 2, 3, 5, 6, 7, 8]
    return any(grid.flat[a] == grid.flat[b] for a in border for b in border
               if a < b and a // 3 != b // 3 and a % 3 != b % 3)


def _single_repeat(grid: np.ndarray) -> str | None:
    if len(np.unique(grid)) != 8:
        return None
    a, b = [i for i in range(9) if np.count_nonzero(grid == grid.flat[i]) == 2]
    if 4 in (a, b):
        return "center"
    if a // 3 == b // 3:
        return "row"
    if a % 3 == b % 3:
        return "column"
    return "separated"


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert [len(p["train"]) for p in puzzles].count(3) >= 10
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3

    overlaps, unions, patterns, palettes = set(), set(), set(), set()
    sizes, test_geometries = set(), set()
    features = set()
    source_palette_examples = 0
    for puzzle in puzzles:
        train = puzzle["train"]
        a, b = (set(np.unique(p["input"])) for p in train[:2])
        overlaps.add("disjoint" if not a & b else "identical" if a == b else "partial")
        pairs = train + puzzle["test"]
        if (len(train) >= 3 and len(a & b) == 3
                and len(np.unique(train[2]["input"])) == 6
                and set(np.unique(train[2]["input"]))
                == set(np.unique(puzzle["test"][0]["input"]))):
            source_palette_examples += 1
        unions.add(len(set().union(*(set(np.unique(p["input"])) for p in pairs))))
        train_colors = set().union(*(set(np.unique(p["input"])) for p in train))
        # Locate each output cell using training examples only. No test output,
        # seed, or sampled palette is used to construct this prediction.
        addresses = []
        for row in range(5):
            address_row = []
            for column in range(5):
                candidates = [(i, j) for i in range(3) for j in range(3)
                              if all(int(pair["output"][row, column])
                                     == int(pair["input"][i, j]) for pair in train)]
                assert len(candidates) == 1
                address_row.append(candidates[0])
            addresses.append(address_row)
        for pair in puzzle["test"]:
            predicted = np.array(
                [[pair["input"][addresses[row][column]] for column in range(5)]
                 for row in range(5)])
            assert np.array_equal(predicted, pair["output"])
        for split, group in (("train", train), ("test", puzzle["test"])):
            for pair in group:
                grid, output = pair["input"], pair["output"]
                expected = np.array([[grid[r, c] for c in _IDX] for r in _IDX])
                assert np.array_equal(output, expected)
                assert grid.dtype == output.dtype == np.int8
                patterns.add(_canonical(grid))
                palettes.add(tuple(sorted(int(x) for x in np.unique(grid))))
                sizes.add(len(np.unique(grid)))
                if any(len(np.unique(row)) == 1 for row in grid):
                    features.add("solid_row")
                if any(len(np.unique(grid[:, c])) == 1 for c in range(3)):
                    features.add("solid_col")
                if len({tuple(row) for row in grid}) < 3:
                    features.add("repeat_rows")
                if len({tuple(grid[:, c]) for c in range(3)}) < 3:
                    features.add("repeat_cols")
                if np.array_equal(grid, grid[:, ::-1]):
                    features.add("mirror")
                if grid[0, 0] == grid[1, 1] == grid[2, 2]:
                    features.add("diagonal")
                if grid[0, 0] == grid[0, 2] == grid[2, 0] == grid[2, 2]:
                    features.add("corners")
                if split == "test":
                    geometry = _single_repeat(grid)
                    if geometry:
                        test_geometries.add(geometry)
                    if len(np.unique(grid)) == 9:
                        test_geometries.add("none")
                    if len(np.unique(grid)) == 6 and _separated_border(grid):
                        test_geometries.add("source_like")
                    if set(np.unique(grid)) - train_colors:
                        features.add("unseen_test_color")
    assert overlaps == {"disjoint", "partial", "identical"}
    assert source_palette_examples >= 1
    assert len(unions) >= 3 and max(unions) >= 9
    assert len(patterns) >= 100 and len(palettes) >= 50
    assert {1, 7, 8, 9} <= sizes
    assert {"solid_row", "solid_col", "repeat_rows", "repeat_cols", "mirror",
            "diagonal", "corners", "unseen_test_color"} <= features
    assert {"center", "row", "column", "none", "source_like"} <= test_geometries

    for seed, n, m in ((101, 2, 1), (102, 10, 1), (103, 10, 4)):
        p = generate_puzzle(seed, n, m)
        assert (len(p["train"]), len(p["test"])) == (n, m)
    for seed, n, m in ((201, None, 1), (202, None, 4),
                       (203, 2, None), (204, 10, None)):
        p = generate_puzzle(seed, n, m)
        assert n is None or len(p["train"]) == n
        assert m is None or len(p["test"]) == m
    for n, m in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4), (True, 1)):
        try:
            generate_puzzle(300, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    a, b = generate_puzzle(777, 8, 3), generate_puzzle(777, 8, 3)
    for split in ("train", "test"):
        for left, right in zip(a[split], b[split]):
            assert np.array_equal(left["input"], right["input"])
            assert np.array_equal(left["output"], right["output"])
    assert not np.array_equal(a["train"][0]["input"],
                              generate_puzzle(778, 8, 3)["train"][0]["input"])
