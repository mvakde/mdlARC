import numpy as np


_SOURCE_INPUTS = (
    np.array([[3, 8, 2], [3, 2, 2], [8, 5, 2]]),
    np.array([[8, 5, 0], [8, 5, 3], [0, 3, 2]]),
    np.array([[0, 3, 0], [6, 6, 6], [0, 3, 0]]),
    np.array([[2, 5, 0], [2, 5, 1], [3, 1, 1]]),
)


def _make_output(grid: np.ndarray) -> np.ndarray:
    n = grid.shape[0]
    result = np.empty((2 * n, 2 * n), dtype=np.int64)
    result[:n, :n] = grid
    result[:n, n:] = np.rot90(grid, -1)
    result[n:, :n] = np.rot90(grid, 1)
    result[n:, n:] = np.rot90(grid, 2)
    return result


def _key(grid: np.ndarray) -> tuple:
    return grid.shape, tuple(int(x) for x in grid.flat)


def _symmetry(grid: np.ndarray) -> str:
    if np.all(grid == grid.flat[0]):
        return "uniform"
    turn = np.array_equal(grid, np.rot90(grid))
    half = np.array_equal(grid, np.rot90(grid, 2))
    horizontal = np.array_equal(grid, np.flipud(grid))
    vertical = np.array_equal(grid, np.fliplr(grid))
    if turn:
        return "quarter_reflected" if horizontal else "quarter"
    if horizontal and vertical:
        return "both_axes"
    if half:
        if horizontal or vertical:
            return "both_axes"
        diagonal = np.array_equal(grid, grid.T)
        return "half_diagonal" if diagonal else "half"
    if horizontal:
        return "horizontal"
    if vertical:
        return "vertical"
    orientations = {
        _key(np.rot90(grid, k)) for k in range(4)
    } | {
        _key(np.fliplr(np.rot90(grid, k))) for k in range(4)
    }
    return "asymmetric" if len(orientations) == 8 else "diagonal"


def _orbits(n: int, mode: str) -> list[list[tuple[int, int]]]:
    def images(r: int, c: int) -> set[tuple[int, int]]:
        if mode == "half":
            return {(r, c), (n - 1 - r, n - 1 - c)}
        if mode == "horizontal":
            return {(r, c), (n - 1 - r, c)}
        if mode == "vertical":
            return {(r, c), (r, n - 1 - c)}
        if mode == "both_axes":
            return {
                (r, c), (n - 1 - r, c),
                (r, n - 1 - c), (n - 1 - r, n - 1 - c),
            }
        if mode == "quarter":
            return {
                (r, c), (c, n - 1 - r),
                (n - 1 - r, n - 1 - c), (n - 1 - c, r),
            }
        return {(r, c)}

    unseen = {(r, c) for r in range(n) for c in range(n)}
    result = []
    while unseen:
        orbit = images(*min(unseen))
        result.append(sorted(orbit))
        unseen -= orbit
    return result


def _pattern(
    rng: np.random.Generator,
    n: int,
    palette: np.ndarray,
    style: str,
) -> np.ndarray:
    rows, cols = np.indices((n, n))
    k = len(palette)
    if k == 1:
        return np.full((n, n), palette[0], dtype=np.int64)
    if style == "random":
        labels = rng.integers(k, size=(n, n))
    elif style == "skewed":
        weights = np.ones(k)
        weights[0] = float(rng.uniform(4, 12))
        labels = rng.choice(k, size=(n, n), p=weights / weights.sum())
    elif style == "bands":
        width = int(rng.integers(1, max(3, n // 3 + 1)))
        a, b = rng.integers(0, 4, size=2)
        if a == b == 0:
            a = 1
        labels = (a * (rows // width) + b * (cols // width)) % k
    elif style == "rings":
        depth = np.minimum.reduce((rows, cols, n - 1 - rows, n - 1 - cols))
        labels = (depth + int(rng.integers(k))) % k
    elif style == "blocks":
        height = int(rng.integers(1, max(3, n // 2 + 1)))
        width = int(rng.integers(1, max(3, n // 2 + 1)))
        labels = (rows // height + 2 * (cols // width)) % k
    elif style == "sparse":
        labels = np.zeros((n, n), dtype=np.int64)
        for color in range(1, k):
            r = int(rng.integers(n))
            c = int(rng.integers(n))
            labels[r, c] = color
        for _ in range(int(rng.integers(0, 3))):
            r = int(rng.integers(n))
            c = int(rng.integers(n))
            labels[r, c] = int(rng.integers(1, k))
    elif style == "connected":
        labels = np.zeros((n, n), dtype=np.int64)
        r = int(rng.integers(n))
        c = int(rng.integers(n))
        for step in range(int(rng.integers(n, 3 * n + 1))):
            labels[r, c] = 1 + step % (k - 1)
            dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[int(rng.integers(4))]
            r = min(n - 1, max(0, r + dr))
            c = min(n - 1, max(0, c + dc))
    elif style == "hollow":
        labels = np.zeros((n, n), dtype=np.int64)
        top = int(rng.integers(0, max(1, n - 2)))
        left = int(rng.integers(0, max(1, n - 2)))
        bottom = int(rng.integers(top + 1, n))
        right = int(rng.integers(left + 1, n))
        labels[top, left:right + 1] = 1
        labels[bottom, left:right + 1] = 1
        labels[top:bottom + 1, left] = 1
        labels[top:bottom + 1, right] = 1
        if k > 2:
            labels[top, right] = 2
    elif style == "corners":
        labels = rng.integers(k, size=(n, n))
        for index, (r, c) in enumerate(
            ((0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1))
        ):
            labels[r, c] = index % k
        for _ in range(max(1, n // 2)):
            labels[int(rng.integers(n)), int(rng.integers(n))] = int(rng.integers(k))
    else:
        raise AssertionError("unknown pattern")
    return palette[labels]


def _make_grid(
    rng: np.random.Generator,
    n: int,
    colors: tuple[int, ...],
    mode: str,
    style: str,
) -> np.ndarray:
    palette = np.asarray(colors, dtype=np.int64)
    if mode == "uniform":
        return np.full((n, n), palette[0], dtype=np.int64)
    orbits = _orbits(n, mode)
    for attempt in range(80):
        raw = _pattern(rng, n, palette, style)
        grid = np.empty((n, n), dtype=np.int64)
        if mode == "asymmetric":
            grid[:] = raw
            missing = (
                list(rng.permutation(palette))
                if attempt >= 40
                else [color for color in palette if color not in grid]
            )
            positions = rng.choice(n * n, size=len(missing), replace=False)
            for position, color in zip(positions, missing):
                grid.flat[int(position)] = int(color)
        else:
            assignments = (
                rng.choice(palette, size=len(orbits))
                if attempt >= 40
                else np.array([raw[orbit[0]] for orbit in orbits], dtype=np.int64)
            )
            missing = (
                list(rng.permutation(palette))
                if attempt >= 40
                else [color for color in palette if color not in assignments]
            )
            positions = rng.choice(
                len(orbits), size=len(missing), replace=False
            )
            for position, color in zip(positions, missing):
                assignments[int(position)] = int(color)
            for orbit, color in zip(orbits, assignments):
                for r, c in orbit:
                    grid[r, c] = int(color)
        if _symmetry(grid) == mode and len(np.unique(grid)) == len(palette):
            return grid
    if mode == "asymmetric":
        grid = np.full((n, n), palette[0], dtype=np.int64)
        if n == 2:
            for cell, color in zip(((0, 0), (0, 1), (1, 0), (1, 1)), palette):
                grid[cell] = int(color)
        else:
            grid[0, 0] = palette[1]
            grid[1, 0] = palette[1]
            free_cells = (
                (r, c) for r in range(n) for c in range(n)
                if (r, c) not in ((0, 0), (1, 0))
            )
            for color, (r, c) in zip(palette[2:], free_cells):
                grid[r, c] = color
        if _symmetry(grid) == mode and len(np.unique(grid)) == len(palette):
            return grid
    else:
        limit = min(12, len(orbits))
        for first in range(limit):
            for second in range(first, limit):
                marked = {first, second}
                grid = np.full((n, n), palette[0], dtype=np.int64)
                for index in marked:
                    for cell in orbits[index]:
                        grid[cell] = palette[1]
                if _symmetry(grid) != mode:
                    continue
                unmarked = (
                    index for index in range(len(orbits))
                    if index not in marked
                )
                for color, index in zip(palette[2:], unmarked):
                    for cell in orbits[index]:
                        grid[cell] = color
                if _symmetry(grid) == mode and len(np.unique(grid)) == len(palette):
                    return grid
    raise AssertionError("unable to construct requested input symmetry")


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    for name, value, low, high in (
        ("num_train", num_train, 2, 10),
        ("num_test", num_test, 1, 4),
    ):
        if value is not None:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
            if not low <= int(value) <= high:
                raise ValueError(f"{name} outside {low}..{high}")
    train = None if num_train is None else int(num_train)
    test = None if num_test is None else int(num_test)
    if test is None and train is None:
        test = int(rng.integers(1, 5))
    if train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    if test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _sample_size(rng: np.random.Generator) -> int:
    return int(rng.choice(
        np.arange(2, 16),
        p=np.array([8, 10, 10, 9, 8, 8, 8, 7, 7, 6, 5, 5, 5, 4], dtype=float) / 100,
    ))


def _sample_mode(rng: np.random.Generator, n: int) -> str:
    if n == 2:
        modes = ("asymmetric", "horizontal", "vertical", "uniform")
        weights = (0.60, 0.16, 0.16, 0.08)
    elif n == 3:
        modes = ("asymmetric", "half", "horizontal", "vertical", "both_axes", "uniform")
        weights = (0.44, 0.17, 0.12, 0.12, 0.11, 0.04)
    else:
        modes = ("asymmetric", "half", "horizontal", "vertical",
                 "both_axes", "quarter", "uniform")
        weights = (0.43, 0.14, 0.10, 0.10, 0.10, 0.09, 0.04)
    return str(rng.choice(modes, p=weights))


def _sample_style(rng: np.random.Generator, n: int, mode: str) -> str:
    if mode == "uniform":
        return "random"
    styles = ["random", "skewed", "bands", "rings", "blocks",
              "sparse", "connected", "hollow", "corners"]
    if n == 2:
        styles.remove("hollow")
        styles.remove("connected")
    return str(rng.choice(styles))


def _capacity(n: int, mode: str, style: str) -> int:
    if mode == "uniform":
        return 1
    capacity = len(_orbits(n, mode))
    if style in ("sparse", "connected", "hollow"):
        capacity = min(capacity, 3)
    if n == 2 and mode == "asymmetric":
        capacity = 4
    return min(capacity, 10)


def _choose_palette(
    rng: np.random.Generator,
    available: list[int],
    capacity: int,
    mandatory: int | None = None,
    minimum: int = 1,
) -> tuple[int, ...]:
    maximum = min(capacity, len(available))
    minimum = min(minimum, maximum)
    k = int(rng.integers(minimum, maximum + 1))
    if mandatory is None:
        selected = rng.choice(available, size=k, replace=False)
    else:
        others = [x for x in available if x != mandatory]
        selected = [mandatory] + [int(x) for x in rng.choice(
            others, size=k - 1, replace=False
        )]
    return tuple(int(x) for x in rng.permutation(selected))


def _test_palette(
    rng: np.random.Generator,
    training_colors: set[int],
    capacity: int,
    minimum: int,
) -> tuple[int, ...]:
    old = sorted(training_colors)
    novel = [x for x in range(10) if x not in training_colors]
    regime = str(rng.choice(
        ("overlap", "one_novel", "several_novel", "disjoint"),
        p=(0.28, 0.31, 0.27, 0.14),
    ))
    if regime == "overlap" or not novel:
        novel_count = 0
    elif regime == "one_novel" or capacity == 1:
        novel_count = 1
    elif regime == "disjoint":
        novel_count = int(rng.integers(
            min(2, len(novel), capacity),
            min(4, len(novel), capacity) + 1,
        ))
    else:
        novel_count = int(rng.integers(
            2, min(4, len(novel), capacity) + 1,
        )) if min(len(novel), capacity) >= 2 else 1
    old_capacity = min(len(old), capacity - novel_count)
    if regime == "disjoint" and novel_count:
        old_count = 0
    elif novel_count == 0:
        old_count = int(rng.integers(1, old_capacity + 1))
    else:
        old_count = int(rng.integers(0, old_capacity + 1))
    while old_count + novel_count < minimum:
        if old_count < min(len(old), capacity - novel_count):
            old_count += 1
        elif novel_count < min(len(novel), capacity - old_count):
            novel_count += 1
        else:
            raise AssertionError("insufficient colors for test input")
    colors = list(rng.choice(old, size=old_count, replace=False))
    colors += list(rng.choice(novel, size=novel_count, replace=False))
    return tuple(int(x) for x in rng.permutation(colors))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    # The only rule convention is the fixed quadrant order. These puzzle-level
    # choices set distributions, not hidden information needed by a solver.
    same_size = bool(rng.random() < 0.24)
    common_size = _sample_size(rng) if same_size else None
    training_pool = [int(x) for x in rng.choice(
        10, size=int(rng.integers(4, 9)), replace=False
    )]
    early_count = int(rng.integers(4, len(training_pool) + 1))
    early_colors = training_pool[:early_count]
    demo_index = int(rng.integers(train_count))
    pair_seeds = rng.integers(
        0, np.iinfo(np.uint64).max,
        size=train_count + test_count, dtype=np.uint64,
    )
    used = set()
    puzzle = {"train": [], "test": []}

    for index in range(train_count + test_count):
        split = "train" if index < train_count else "test"
        pair_rng = np.random.default_rng(pair_seeds[index])
        n = common_size if same_size else _sample_size(pair_rng)
        mode = "asymmetric" if index == demo_index else _sample_mode(pair_rng, n)
        style = _sample_style(pair_rng, n, mode)
        if index == demo_index and n == 2:
            style = "corners"
        capacity = _capacity(n, mode, style)
        if split == "train":
            available = early_colors if index < min(2, train_count - 1) else training_pool
            mandatory = (
                training_pool[-1]
                if index == train_count - 1 and len(training_pool) > early_count
                else None
            )
            minimum = (
                4 if index == demo_index and n == 2
                else 3 if mode == "asymmetric" and n == 2
                else 2 if mode != "uniform" else 1
            )
            colors = _choose_palette(
                pair_rng, available, capacity, mandatory, minimum
            )
        else:
            training_colors = {
                int(x) for pair in puzzle["train"]
                for x in np.unique(pair["input"])
            }
            minimum = 3 if mode == "asymmetric" and n == 2 else (
                2 if mode != "uniform" else 1
            )
            colors = _test_palette(pair_rng, training_colors, capacity, minimum)

        for attempt in range(35):
            grid = _make_grid(pair_rng, n, colors, mode, style)
            if _key(grid) not in used and not any(
                np.array_equal(grid, original) for original in _SOURCE_INPUTS
            ):
                break
            if split == "train":
                grid_colors = list(available)
                colors = _choose_palette(
                    pair_rng, grid_colors, capacity, mandatory, minimum
                )
            else:
                colors = _test_palette(pair_rng, training_colors, capacity, minimum)
        else:
            raise AssertionError("unable to construct unique input")
        used.add(_key(grid))
        puzzle[split].append({"input": grid, "output": _make_output(grid)})

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    keys = set()
    informative = False
    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        source, target = pair["input"], pair["output"]
        for grid in (source, target):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert np.issubdtype(grid.dtype, np.integer)
            assert all(1 <= dimension <= 30 for dimension in grid.shape)
            assert np.all((grid >= 0) & (grid <= 9))
        n = source.shape[0]
        assert 2 <= n <= 15 and source.shape == (n, n)
        assert target.shape == (2 * n, 2 * n)
        assert np.array_equal(target[:n, :n], source)
        assert np.array_equal(target[:n, n:], np.rot90(source, -1))
        assert np.array_equal(target[n:, :n], np.rot90(source, 1))
        assert np.array_equal(target[n:, n:], np.rot90(source, 2))
        assert set(np.unique(source)) == set(np.unique(target))
        key = _key(source)
        assert key not in keys
        keys.add(key)
        assert not any(np.array_equal(source, old) for old in _SOURCE_INPUTS)
    for pair in puzzle["train"]:
        informative |= _symmetry(pair["input"]) == "asymmetric"
    assert informative


def _same_puzzle(a: dict, b: dict) -> bool:
    return all(
        np.array_equal(pa[name], pb[name])
        for split in ("train", "test")
        for pa, pb in zip(a[split], b[split])
        for name in ("input", "output")
    ) and all(len(a[split]) == len(b[split]) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert _same_puzzle(generate_puzzle(17), generate_puzzle(17))
    assert _same_puzzle(generate_puzzle(17, 8, 3), generate_puzzle(17, 8, 3))
    assert not _same_puzzle(generate_puzzle(17), generate_puzzle(18))
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    sizes = set()
    symmetries = set()
    palette_counts = set()
    novelty = set()
    mixed_sizes = 0
    symmetric_test = 0
    late_color = 0
    all_keys = set()
    total = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_colors = set()
        first_two = set()
        local_sizes = set()
        for index, pair in enumerate(puzzle["train"]):
            colors = {int(x) for x in np.unique(pair["input"])}
            train_colors |= colors
            if index < 2:
                first_two |= colors
        late_color += bool(train_colors - first_two) if len(puzzle["train"]) > 2 else False
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                n = grid.shape[0]
                sizes.add(n)
                local_sizes.add(n)
                symmetries.add(_symmetry(grid))
                palette_counts.add(len(np.unique(grid)))
                all_keys.add(_key(grid))
                total += 1
                assert np.array_equal(pair["output"], np.block([
                    [grid, np.rot90(grid, 3)],
                    [np.rot90(grid, 1), np.rot90(grid, 2)],
                ]))
                if split == "test":
                    symmetric_test += _symmetry(grid) != "asymmetric"
                    novelty.add(min(2, len(set(np.unique(grid)) - train_colors)))
        mixed_sizes += len(local_sizes) > 1
    assert len(sizes) >= 10 and 2 in sizes and 15 in sizes
    assert {"asymmetric", "horizontal", "vertical", "both_axes", "half", "quarter", "uniform"} <= symmetries
    assert {1, 2, 5} <= palette_counts
    assert novelty == {0, 1, 2}
    assert mixed_sizes >= 15 and symmetric_test >= 10 and late_color >= 3
    assert len(all_keys) >= int(0.95 * total)
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4), (6, 3))):
        puzzle = generate_puzzle(100 + index, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for index, kwargs in enumerate((
        {"num_test": 1}, {"num_test": 4},
        {"num_train": 2}, {"num_train": 10},
    )):
        validate_generated_puzzle(generate_puzzle(200 + index, **kwargs))
    for kwargs in (
        {"num_train": 1}, {"num_train": 11},
        {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
    ):
        try:
            generate_puzzle(1, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
