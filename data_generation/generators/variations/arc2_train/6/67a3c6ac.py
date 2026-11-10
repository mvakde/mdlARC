"""Whole-grid left/right reflection puzzles."""

import numpy as np


_SOURCE_INPUTS = (
    ((7, 7, 7, 6, 6, 6, 2), (6, 7, 1, 1, 7, 7, 1),
     (7, 7, 2, 1, 2, 6, 6), (2, 2, 7, 7, 7, 2, 2),
     (7, 2, 7, 1, 2, 7, 2), (6, 6, 6, 2, 2, 1, 1),
     (6, 2, 6, 6, 6, 6, 6)),
    ((6, 6, 6, 2), (6, 1, 6, 2), (7, 2, 7, 2), (1, 7, 2, 2)),
    ((1, 2, 7, 1, 1, 1), (2, 1, 7, 7, 2, 6),
     (2, 1, 2, 6, 2, 1), (1, 2, 1, 7, 6, 2),
     (2, 7, 1, 2, 7, 1), (2, 1, 6, 2, 7, 7)),
    ((7, 6, 1), (6, 7, 6), (6, 2, 2)),
)
_SOURCE_KEYS = {
    (len(rows), len(rows[0]), tuple(value for row in rows for value in row))
    for rows in _SOURCE_INPUTS
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _validate_count(name, value, low, high):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(seed, num_train, num_test):
    _validate_count("num_train", num_train, 2, 10)
    _validate_count("num_test", num_test, 1, 4)
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return int(num_train), int(num_test)
    if num_train is None and num_test is None:
        tests = 1 + seed % 4
        minimum = max(2, tests + 1)
        return minimum + (seed // 4) % (11 - minimum), tests
    if num_train is None:
        tests = int(num_test)
        minimum = max(2, tests + 1)
        return minimum + seed % (11 - minimum), tests
    trains = int(num_train)
    return trains, 1 + seed % min(4, trains - 1)


def _key(grid):
    return (grid.shape[0], grid.shape[1], tuple(int(x) for x in grid.flat))


def _sample_shape(rng, low, high, parity=None):
    # Aspect choice and dimensions are pair-level choices. Rectangles can be
    # tall or wide, while the square branch retains compact source-like grids.
    kind = int(rng.integers(0, 5))
    height = int(rng.integers(low, high + 1))
    width = int(rng.integers(low, high + 1))
    midpoint = (low + high) // 2
    if kind == 1:
        width = height
    elif kind == 2 and low < high:
        height = int(rng.integers(low, midpoint + 1))
        width = int(rng.integers(midpoint + 1, high + 1))
    elif kind == 3 and low < high:
        height = int(rng.integers(midpoint + 1, high + 1))
        width = int(rng.integers(low, midpoint + 1))
    if parity is not None:
        eligible = np.arange(low, high + 1)
        eligible = eligible[eligible % 2 == parity]
        width = int(rng.choice(eligible))
    return height, width


def _sample_palette(rng, pool, area, anchor=False, novel=None):
    choices = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    probabilities = np.array([0.07, 0.23, 0.21, 0.17, 0.13, 0.09, 0.06, 0.04])
    target = int(rng.choice(choices, p=probabilities))
    target = min(max(target, 2 if anchor else 1), len(pool) + int(novel is not None), area)
    if novel is None:
        selected = rng.choice(pool, size=target, replace=False)
    else:
        selected = np.concatenate((
            rng.choice(pool, size=target - 1, replace=False),
            np.array([novel]),
        ))
        rng.shuffle(selected)
    return np.asarray(selected, dtype=np.int8)


def _symmetric_grid(rng, height, width, colors):
    half = rng.choice(colors, size=(height, (width + 1) // 2))
    if width % 2:
        return np.concatenate((half, half[:, :-1][:, ::-1]), axis=1).astype(np.int8)
    return np.concatenate((half, half[:, ::-1]), axis=1).astype(np.int8)


def _make_layout(rng, height, width, colors, mode):
    background = int(colors[0])
    count = len(colors)
    weights = rng.dirichlet(np.full(count, float(rng.uniform(0.45, 1.7))))
    def recolor(mask, grid):
        number = int(np.count_nonzero(mask))
        if number:
            grid[mask] = rng.choice(colors, size=number, p=weights)

    if mode == 0:  # Dense source-like texture
        grid = rng.choice(colors, size=(height, width), p=weights)
    elif mode == 1:  # Horizontal bands
        grid = np.repeat(rng.choice(colors, size=height, p=weights)[:, None], width, axis=1)
        recolor(rng.random((height, width)) < rng.uniform(0.1, 0.4), grid)
    elif mode == 2:  # Vertical bands
        grid = np.repeat(rng.choice(colors, size=width, p=weights)[None, :], height, axis=0)
        recolor(rng.random((height, width)) < rng.uniform(0.1, 0.4), grid)
    elif mode == 3:  # Diagonal or checker-like recurrence
        rr, cc = np.indices((height, width))
        a = int(rng.integers(1, max(2, count)))
        b = int(rng.integers(1, max(2, count)))
        grid = colors[(a * rr + b * cc + int(rng.integers(0, count))) % count].copy()
        recolor(rng.random((height, width)) < rng.uniform(0.04, 0.25), grid)
    elif mode == 4:  # Unequal rectangular regions
        grid = np.full((height, width), background, dtype=np.int8)
        row_cuts = sorted(set([0, height] + [int(rng.integers(1, height)) for _ in range(min(3, height - 1))]))
        col_cuts = sorted(set([0, width] + [int(rng.integers(1, width)) for _ in range(min(3, width - 1))]))
        for r0, r1 in zip(row_cuts, row_cuts[1:]):
            for c0, c1 in zip(col_cuts, col_cuts[1:]):
                grid[r0:r1, c0:c1] = int(rng.choice(colors))
        recolor(rng.random((height, width)) < rng.uniform(0.02, 0.22), grid)
    elif mode == 5:  # Related but modified rows
        grid = np.empty((height, width), dtype=np.int8)
        grid[0] = rng.choice(colors, size=width, p=weights)
        for r in range(1, height):
            grid[r] = np.roll(grid[r - 1], int(rng.integers(-3, 4)))
            recolor(rng.random(width) < rng.uniform(0.15, 0.45), grid[r])
    elif mode == 6:  # Offset connected shapes on a sparse field
        grid = np.full((height, width), background, dtype=np.int8)
        for _ in range(int(rng.integers(1, min(6, 1 + max(height, width))))):
            r0 = int(rng.integers(0, height))
            c0 = int(rng.integers(0, width))
            rh = int(rng.integers(1, max(2, min(5, height - r0) + 1)))
            cw = int(rng.integers(1, max(2, min(5, width - c0) + 1)))
            color = int(rng.choice(colors[1:])) if count > 1 else background
            grid[r0:r0 + rh, c0:c0 + cw] = color
            if rh > 1 and cw > 1 and rng.random() < 0.5:
                grid[r0 + 1:r0 + rh, c0 + 1:c0 + cw] = background
    elif mode == 7:  # Separated small colored blocks
        grid = np.full((height, width), background, dtype=np.int8)
        occupied = np.zeros((height, width), dtype=bool)
        for _ in range(int(rng.integers(2, 7))):
            for _ in range(12):
                rh = int(rng.integers(1, min(5, height) + 1))
                cw = int(rng.integers(1, min(5, width) + 1))
                r0 = int(rng.integers(0, height - rh + 1))
                c0 = int(rng.integers(0, width - cw + 1))
                r1, c1 = r0 + rh, c0 + cw
                if not np.any(occupied[max(0, r0 - 1):min(height, r1 + 1), max(0, c0 - 1):min(width, c1 + 1)]):
                    grid[r0:r1, c0:c1] = int(rng.choice(colors[1:])) if count > 1 else background
                    occupied[r0:r1, c0:c1] = True
                    break
    elif mode == 8:  # Asymmetric horizontal and vertical linework
        grid = np.full((height, width), background, dtype=np.int8)
        for _ in range(int(rng.integers(2, 8))):
            color = int(rng.choice(colors[1:])) if count > 1 else background
            if rng.random() < 0.5:
                r = int(rng.integers(0, height))
                c0 = int(rng.integers(0, width))
                c1 = int(rng.integers(c0 + 1, width + 1))
                grid[r, c0:c1] = color
            else:
                c = int(rng.integers(0, width))
                r0 = int(rng.integers(0, height))
                r1 = int(rng.integers(r0 + 1, height + 1))
                grid[r0:r1, c] = color
    elif mode == 9:  # Border and an independently placed interior mark
        grid = np.full((height, width), background, dtype=np.int8)
        border = int(colors[1]) if count > 1 else background
        grid[0, :] = border
        grid[-1, :] = border
        grid[:, 0] = border
        grid[:, -1] = border
        if height > 2 and width > 2:
            r = int(rng.integers(1, height - 1))
            c = int(rng.integers(1, width - 1))
            grid[r, c] = int(colors[-1]) if count > 2 else background
        else:
            grid[int(rng.integers(0, height)), int(rng.integers(0, width))] = background
    else:  # Near symmetry with a small off-axis change
        grid = _symmetric_grid(rng, height, width, colors)
        for _ in range(int(rng.integers(1, min(4, height + 1)))):
            r = int(rng.integers(0, height))
            c = int(rng.integers(0, width // 2)) if width > 1 else 0
            alternatives = colors[colors != grid[r, c]]
            if len(alternatives):
                grid[r, c] = int(rng.choice(alternatives))
    return np.asarray(grid, dtype=np.int8)


def _clear_anchor(grid):
    reflected = grid[:, ::-1]
    if grid.shape[0] < 2 or grid.shape[1] < 2:
        return False
    if np.count_nonzero(np.any(grid != reflected, axis=1)) < 2:
        return False
    if np.array_equal(grid[0], grid[-1]):
        return False
    alternatives = (grid, grid[::-1], grid[::-1, ::-1],
                    np.rot90(grid), np.rot90(grid, 3), grid.T)
    return all(other.shape != grid.shape or not np.array_equal(reflected, other)
               for other in alternatives)


def _special_test(rng, shape, colors, kind):
    height, width = shape
    if kind == "uniform":
        return np.full(shape, int(colors[0]), dtype=np.int8)
    grid = _symmetric_grid(rng, height, width, colors)
    if kind == "one_asymmetric_row" and width > 1:
        r = int(rng.integers(0, height))
        c = int(rng.integers(0, width // 2))
        alternatives = colors[colors != grid[r, c]]
        if len(alternatives):
            grid[r, c] = int(rng.choice(alternatives))
    return grid


def _puzzles_equal(first, second):
    return all(
        len(first[split]) == len(second[split])
        and all(np.array_equal(a[role], b[role])
                for a, b in zip(first[split], second[split])
                for role in ("input", "output"))
        for split in ("train", "test")
    )


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)

    # The only learned convention is reflection across the vertical axis.
    # A preferred color pool is visual continuity, not a semantic color map.
    pool = rng.choice(np.arange(10, dtype=np.int8), size=int(rng.integers(5, 9)), replace=False)
    shift = seed % 3  # 0 smaller tests, 1 comparable tests, 2 larger tests
    modes = list(map(int, rng.permutation(11)))
    while len(modes) < train_count + test_count:
        modes.extend(map(int, rng.permutation(11)))
    puzzle = {"train": [], "test": []}
    seen = set()
    train_shapes = []
    train_palettes = []

    for split, number in (("train", train_count), ("test", test_count)):
        for index in range(number):
            anchor = split == "train" and index < 2
            special = None
            if split == "test" and index == 0 and shift == 0:
                special = {0: "symmetric", 3: "one_row", 6: "one_column",
                           9: "one_asymmetric_row", 5: "uniform"}.get(seed % 11)
            for attempt in range(128):
                if split == "train":
                    low, high = ((5, 27), (3, 22), (2, 7))[shift]
                    shape = _sample_shape(rng, low, high, index % 2 if anchor else None)
                    if anchor and shape[0] < 3:
                        shape = (3, shape[1])
                elif special == "one_row":
                    shape = (1, int(rng.integers(2, 5)))
                elif special == "one_column":
                    shape = (int(rng.integers(2, 5)), 1)
                elif shift == 0:
                    shape = _sample_shape(rng, 1, 4)
                elif shift == 2:
                    shape = _sample_shape(rng, 8, 30)
                else:
                    base = train_shapes[int(rng.integers(0, len(train_shapes)))]
                    if rng.random() < 0.25:
                        base = base[::-1]
                    shape = tuple(max(1, min(30, x + int(rng.integers(-2, 3)))) for x in base)
                if special in ("symmetric", "one_asymmetric_row", "uniform") and min(shape) < 2:
                    shape = (3, 4)
                new_color = None
                if split == "test" and special is None and seed % 4 == 0 and index == 0:
                    unseen = np.setdiff1d(np.arange(10, dtype=np.int8), pool)
                    new_color = int(rng.choice(unseen))
                colors = _sample_palette(rng, pool, shape[0] * shape[1], anchor, new_color)
                if special in ("symmetric", "one_asymmetric_row", "uniform"):
                    grid = _special_test(rng, shape, colors, special)
                elif special in ("one_row", "one_column"):
                    grid = _make_layout(rng, *shape, colors, modes[train_count + index])
                else:
                    mode = modes[index if split == "train" else train_count + index]
                    if anchor and (mode == 9 or attempt >= 48):
                        mode = 0
                    grid = _make_layout(rng, *shape, colors, mode)
                if new_color is not None:
                    grid[int(rng.integers(0, shape[0])), int(rng.integers(0, shape[1]))] = new_color
                if anchor and not _clear_anchor(grid):
                    continue
                if split == "train" and index == 1 and tuple(np.unique(grid)) == train_palettes[0]:
                    continue
                key = _key(grid)
                if key in seen or key in _SOURCE_KEYS:
                    continue
                seen.add(key)
                output = grid[:, ::-1].copy()
                puzzle[split].append({"input": grid, "output": output})
                if split == "train":
                    train_shapes.append(shape)
                    train_palettes.append(tuple(np.unique(grid)))
                break
            else:
                raise RuntimeError(f"could not construct unique {split}[{index}]")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    _require(all(isinstance(puzzle[split], list) for split in ("train", "test")),
             "train and test must be lists")
    trains, tests = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= trains <= 10 and 1 <= tests <= 4 and trains > tests,
             "illegal pair counts")
    seen = set()
    anchors = []
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            label = f"{split}[{index}]"
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     f"{label}: pair schema")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         f"{label}: grid type or rank")
                _require(all(1 <= side <= 30 for side in grid.shape),
                         f"{label}: grid dimensions")
                _require(np.issubdtype(grid.dtype, np.integer),
                         f"{label}: grid dtype")
                _require(bool(np.all((grid >= 0) & (grid <= 9))),
                         f"{label}: grid colors")
            _require(inp.shape == out.shape and inp.dtype == out.dtype,
                     f"{label}: dimensions or dtype changed")
            _require(np.array_equal(out, inp[:, ::-1]),
                     f"{label}: incorrect left/right reflection")
            key = _key(inp)
            _require(key not in seen, f"{label}: duplicate input")
            _require(key not in _SOURCE_KEYS, f"{label}: exact source pair")
            seen.add(key)
            if split == "train" and _clear_anchor(inp):
                anchors.append(inp.shape[1] % 2)
    _require(len(anchors) >= 2 and {0, 1}.issubset(anchors),
             "training does not distinguish reflection on odd and even widths")
    # Reflection has no learned color map or background: each literal input
    # value determines its own output, including values absent from training.


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train counts lack variety")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test counts lack variety")
    for seed, train, test in ((2000, 2, 1), (2001, 10, 1), (2002, 10, 4),
                              (2003, 5, 2), (2004, 7, 3)):
        puzzle = generate_puzzle(seed, train, test)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train, test),
                 "explicit counts were ignored")
    for seed, train, test in ((3000, None, 1), (3001, None, 4),
                              (3002, 2, None), (3003, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        _require(train is None or len(puzzle["train"]) == train,
                 "partial train count was ignored")
        _require(test is None or len(puzzle["test"]) == test,
                 "partial test count was ignored")
    for request in ((1, None), (11, None), (None, 0), (None, 5),
                    (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(4000, *request)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {request}")
    _require(_puzzles_equal(generate_puzzle(12345, 8, 3),
                            generate_puzzle(12345, 8, 3)), "nondeterministic puzzle")
    _require(not _puzzles_equal(generate_puzzle(12345, 8, 3),
                                generate_puzzle(12346, 8, 3)), "seeds have no effect")

    all_pairs = [(split, pair) for puzzle in puzzles
                 for split in ("train", "test") for pair in puzzle[split]]
    shapes = {pair["input"].shape for _, pair in all_pairs}
    palettes = {tuple(np.unique(pair["input"])) for _, pair in all_pairs}
    color_counts = {len(np.unique(pair["input"])) for _, pair in all_pairs}
    _require(len(shapes) >= 35 and any(h != w for h, w in shapes)
             and any(h >= 20 or w >= 20 for h, w in shapes),
             "geometry diversity insufficient")
    _require(any(h == 1 for h, w in shapes) and any(w == 1 for h, w in shapes),
             "missing one-row or one-column cases")
    _require({1, 2, 3}.issubset(color_counts) and max(color_counts) >= 6
             and len(palettes) >= 40, "palette diversity insufficient")
    _require(sum(len({p["input"].shape for split in ("train", "test")
                      for p in puzzle[split]}) > 1 for puzzle in puzzles) >= 35,
             "within-puzzle shape diversity insufficient")
    _require(sum(len({tuple(np.unique(p["input"])) for split in ("train", "test")
                      for p in puzzle[split]}) > 1 for puzzle in puzzles) >= 35,
             "within-puzzle palette diversity insufficient")
    _require(any(not np.array_equal(pair["input"], pair["output"])
                 for _, pair in all_pairs), "no changed outputs")
    test_pairs = [pair for split, pair in all_pairs if split == "test"]
    _require(any(np.array_equal(p["input"], p["output"]) for p in test_pairs),
             "no valid symmetric test")
    _require(any(np.count_nonzero(np.any(p["input"] != p["output"], axis=1)) == 1
                 for p in test_pairs), "no one-asymmetric-row test")
    areas = [(min(p["input"].size for p in puzzle["train"]),
              max(p["input"].size for p in puzzle["train"]),
              [p["input"].size for p in puzzle["test"]]) for puzzle in puzzles]
    _require(any(all(x < low for x in tests) for low, high, tests in areas),
             "no smaller-test regime")
    _require(any(all(x > high for x in tests) for low, high, tests in areas),
             "no larger-test regime")
    _require(len({_key(pair["input"]) for _, pair in all_pairs}) >= 250,
             "grid diversity insufficient")

    smaller = sum(all(x < low for x in tests) for low, high, tests in areas)
    larger = sum(all(x > high for x in tests) for low, high, tests in areas)
    _require(smaller >= 5 and larger >= 5, "train/test size shifts lack variety")
    novel_tests = 0
    sparse = 0
    near_symmetric = 0
    bordered = 0
    dominant_colors = set()
    rich_rectangles = 0
    for puzzle in puzzles:
        learned_colors = {int(c) for pair in puzzle["train"]
                          for c in np.unique(pair["input"])}
        for pair in puzzle["test"]:
            novel_tests += bool(set(int(c) for c in np.unique(pair["input"]))
                                - learned_colors)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                height, width = grid.shape
                values, counts = np.unique(grid, return_counts=True)
                dominant_colors.add(int(values[np.argmax(counts)]))
                rich_rectangles += height != width and len(values) >= 5
                sparse += len(values) > 1 and counts.max() / grid.size >= 0.65
                differences = int(np.count_nonzero(grid != grid[:, ::-1]))
                near_symmetric += (height >= 3 and width >= 3
                                   and 0 < differences <= max(4, int(grid.size * 0.2)))
                if height >= 3 and width >= 3:
                    edge = np.concatenate((grid[0], grid[-1],
                                           grid[1:-1, 0], grid[1:-1, -1]))
                    bordered += (len(np.unique(edge)) == 1
                                 and np.any(grid[1:-1, 1:-1] != edge[0]))
    _require(novel_tests >= 5, "no meaningful test-side novel-color coverage")
    _require(sparse >= 30 and near_symmetric >= 15 and bordered >= 5,
             "sparse, near-symmetric, or bordered arrangements are scarce")
    _require(rich_rectangles >= 10 and len(dominant_colors) >= 8,
             "palette and geometry combinations are too narrow")
