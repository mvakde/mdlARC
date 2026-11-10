import numpy as np


# Exact source inputs are kept only to reject copied examples.
_SOURCE_ROWS = (
    ("73312", "18241", "27872", "77418", "81771"),
    ("81214", "44248", "37248", "27787", "87748"),
    ("2743483", "2371233", "8743224", "1121447",
     "2431141", "4874482", "7384328"),
    ("2813241", "4411434", "1111473", "1123813",
     "4111784", "3284184", "1471234"),
)
_SOURCE_KEYS = frozenset(
    ((len(rows), len(rows[0])),
     tuple(int(c) for row in rows for c in row))
    for rows in _SOURCE_ROWS
)


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def _seed_key(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    return int(seed) % (1 << 64)


def _optional_count(value: int | None, name: str, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return value


def _counts(seed_key: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _optional_count(num_train, "num_train", 2, 10)
    test = _optional_count(num_test, "num_test", 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    quotient = seed_key // 4
    if train is None and test is None:
        test = 1 + seed_key % 4
    if train is None:
        lower = max(2, test + 1)
        train = lower + quotient % (11 - lower)
    if test is None:
        test = 1 + quotient % min(4, train - 1)
    return train, test


def _key(grid: np.ndarray) -> tuple:
    return tuple(grid.shape), tuple(int(v) for v in grid.ravel())


def _sample_shape(rng: np.random.Generator, witness: bool = False) -> tuple[int, int]:
    if witness:
        kind = int(rng.choice(4, p=[0.45, 0.30, 0.13, 0.12]))
    else:
        kind = int(rng.choice(6, p=[0.31, 0.27, 0.13, 0.13, 0.09, 0.07]))
    if kind == 0:
        side = int(rng.choice([4, 5, 5, 5, 6, 7, 7, 7, 8, 9]))
        return side, side
    if kind == 1:
        return int(rng.integers(4, 15)), int(rng.integers(3, 15))
    if kind == 2:
        return int(rng.integers(9, 24)), int(rng.integers(3, 8))
    if kind == 3:
        return int(rng.integers(4, 9)), int(rng.integers(10, 25))
    if kind == 4:
        return int(rng.integers(3, 20)), int(rng.choice([1, 1, 2]))
    return int(rng.integers(13, 31)), int(rng.integers(11, 31))


def _test_shape(rng: np.random.Generator, train_shapes: list[tuple[int, int]]) -> tuple[int, int]:
    base = train_shapes[int(rng.integers(len(train_shapes)))]
    relation = int(rng.choice(5, p=[0.22, 0.38, 0.14, 0.14, 0.12]))
    if relation == 0:
        return base
    if relation == 1:
        return _sample_shape(rng)
    if relation == 2:
        return max(3, base[0] - int(rng.integers(1, 7))), max(1, base[1] - int(rng.integers(1, 7)))
    if relation == 3:
        return min(30, base[0] + int(rng.integers(1, 8))), min(30, base[1] + int(rng.integers(1, 8)))
    # Change aspect ratio even when the reference training grid is square.
    if rng.random() < 0.5:
        return int(rng.integers(4, 10)), int(rng.integers(12, 27))
    return int(rng.integers(12, 27)), int(rng.integers(2, 8))


def _palette(rng: np.random.Generator, mode: int, available: list[int],
             required: int | None = None) -> np.ndarray:
    maximum = 9 if mode <= 5 else (5 if mode in (6, 7, 10, 12) else 6)
    size = int(rng.integers(2, min(maximum, len(available)) + 1))
    if required is None:
        colors = rng.choice(available, size=size, replace=False)
    else:
        other = [c for c in available if c != required]
        colors = np.concatenate((
            np.array([required], dtype=np.int8),
            rng.choice(other, size=size - 1, replace=False),
        ))
    return np.asarray(colors, dtype=np.int8)


def _make_input(rng: np.random.Generator, shape: tuple[int, int],
                palette: np.ndarray, mode: int) -> np.ndarray:
    height, width = shape
    k = len(palette)
    if mode == 0:
        grid = rng.choice(palette, size=shape)
    elif mode == 1:
        weights = rng.dirichlet(np.full(k, 0.45))
        grid = rng.choice(palette, size=shape, p=weights)
    elif mode == 2:
        rows = rng.choice(palette, size=height)
        grid = np.repeat(rows[:, None], width, axis=1)
        mask = rng.random(shape) < float(rng.uniform(0.12, 0.45))
        grid[mask] = rng.choice(palette, size=int(mask.sum()))
    elif mode == 3:
        columns = rng.choice(palette, size=width)
        grid = np.repeat(columns[None, :], height, axis=0)
        mask = rng.random(shape) < float(rng.uniform(0.12, 0.45))
        grid[mask] = rng.choice(palette, size=int(mask.sum()))
    elif mode == 4:
        grid = rng.choice(palette, size=shape)
        for _ in range(int(rng.integers(2, 8))):
            r0 = int(rng.integers(height))
            c0 = int(rng.integers(width))
            r1 = int(rng.integers(r0 + 1, height + 1))
            c1 = int(rng.integers(c0 + 1, width + 1))
            grid[r0:r1, c0:c1] = int(rng.choice(palette))
    elif mode == 5:
        rows = np.arange(height)[:, None]
        columns = np.arange(width)[None, :]
        a = int(rng.integers(1, k + 1))
        b = int(rng.integers(1, k + 1))
        grid = palette[(a * rows + b * columns + int(rng.integers(k))) % k]
        mask = rng.random(shape) < float(rng.uniform(0.04, 0.26))
        grid[mask] = rng.choice(palette, size=int(mask.sum()))
    elif mode == 6:
        runs = int(rng.integers(2, min(height, 7) + 1))
        cuts = sorted(int(v) for v in rng.choice(np.arange(1, height), size=runs - 1, replace=False))
        grid = np.empty(shape, dtype=np.int8)
        bounds = [0] + cuts + [height]
        for i in range(runs):
            color = int(palette[0]) if i == 0 else int(rng.choice(palette))
            grid[bounds[i]:bounds[i + 1], :] = color
    elif mode == 7:
        grid = np.full(shape, int(palette[0]), dtype=np.int8)
        maximum = max(2, height * width // 7)
        marks = int(rng.integers(1, min(maximum, height * width) + 1))
        positions = rng.choice(height * width, size=marks, replace=False)
        grid.ravel()[positions] = rng.choice(palette[1:], size=marks)
    elif mode == 8:
        grid = np.full(shape, int(palette[0]), dtype=np.int8)
        for _ in range(int(rng.integers(1, 6))):
            rh = int(rng.integers(1, max(2, height // 2 + 2)))
            cw = int(rng.integers(1, max(2, width // 2 + 2)))
            r0 = int(rng.integers(0, height - rh + 1))
            c0 = int(rng.integers(0, width - cw + 1))
            grid[r0:r0 + rh, c0:c0 + cw] = int(rng.choice(palette[1:]))
    elif mode == 9:
        half = rng.choice(palette, size=(height, (width + 1) // 2))
        grid = np.concatenate((half, half[:, :width // 2][:, ::-1]), axis=1)
    elif mode == 10:
        templates = rng.choice(palette, size=(min(height - 1, int(rng.integers(2, 5))), width))
        indices = rng.integers(0, len(templates), size=height)
        indices[1] = indices[0]
        grid = templates[indices]
    elif mode == 11:
        half = rng.choice(palette, size=((height + 1) // 2, width))
        grid = np.concatenate((half, half[:height // 2][::-1]), axis=0)
    elif mode == 12:
        grid = rng.choice(palette[:-1], size=shape)
        grid[height // 2, :] = int(palette[-1])
    else:
        raise ValueError("unknown layout mode")
    return np.asarray(grid, dtype=np.int8).copy()


def _is_witness(grid: np.ndarray) -> bool:
    output = np.flipud(grid)
    if np.array_equal(grid, output):
        return False
    alternatives = (
        grid, np.fliplr(grid), np.rot90(grid, 2), grid.T,
        np.rot90(grid, 1), np.rot90(grid, 3),
    )
    return all(a.shape != output.shape or not np.array_equal(a, output)
               for a in alternatives)


def _new_pair(rng: np.random.Generator, shape: tuple[int, int], mode: int,
              available: list[int], used: set, required: int | None = None,
              witness: bool = False, require_changed: bool = False) -> dict:
    for _ in range(48):
        palette = _palette(rng, mode, available, required)
        grid = _make_input(rng, shape, palette, mode)
        key = _key(grid)
        if key in used or key in _SOURCE_KEYS:
            continue
        if len(np.unique(grid)) < 2:
            continue
        if require_changed and np.array_equal(grid, np.flipud(grid)):
            continue
        if required is not None and required not in grid:
            continue
        if witness and not _is_witness(grid):
            continue
        used.add(key)
        return {"input": grid, "output": np.flipud(grid).copy()}
    # Every shape is still usable if an unusually small structured space was exhausted.
    for _ in range(96):
        palette = _palette(rng, 0, available, required)
        grid = _make_input(rng, shape, palette, 0)
        key = _key(grid)
        if key not in used and key not in _SOURCE_KEYS and len(np.unique(grid)) >= 2 and (required is None or required in grid) and (not witness or _is_witness(grid)) and (not require_changed or not np.array_equal(grid, np.flipud(grid))):
            used.add(key)
            return {"input": grid, "output": np.flipud(grid).copy()}
    raise RuntimeError("could not construct a unique grid")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed_key = _seed_key(seed)
    train_count, test_count = _counts(seed_key, num_train, num_test)
    rng = np.random.default_rng(seed_key)

    # The operation is the sole puzzle-wide convention. These independent
    # sampling choices alter the examples but never the rule.
    one_shape = bool(rng.random() < 0.18)
    dense_preference = bool(rng.random() < 0.27)
    reserved_color = int(rng.integers(10)) if rng.random() < 0.34 else None
    train_colors = [c for c in range(10) if c != reserved_color]
    used: set = set()
    witness_shape = _sample_shape(rng, witness=True)
    train_shapes = [witness_shape]
    for _ in range(train_count - 1):
        if one_shape:
            train_shapes.append(witness_shape)
        elif rng.random() < 0.19:
            train_shapes.append(train_shapes[int(rng.integers(len(train_shapes)))])
        else:
            train_shapes.append(_sample_shape(rng))

    train_modes = [0]
    for _ in range(train_count - 1):
        if dense_preference and rng.random() < 0.72:
            train_modes.append(int(rng.choice([0, 1, 2, 3, 4, 5])))
        else:
            train_modes.append(int(rng.integers(0, 11)))
    puzzle = {"train": [], "test": []}
    for i in range(train_count):
        mode = train_modes[i]
        shape = train_shapes[i]
        if mode == 10 and shape[0] < 3:
            mode = 6
        puzzle["train"].append(_new_pair(
            rng, shape, mode, train_colors, used, witness=(i == 0),
            require_changed=True,
        ))

    for i in range(test_count):
        shape = witness_shape if one_shape else _test_shape(rng, train_shapes)
        mode = int(rng.choice(
            13, p=[0.10, 0.08, 0.06, 0.05, 0.08, 0.06,
                   0.14, 0.12, 0.12, 0.06, 0.05, 0.05, 0.03]
        ))
        if mode == 12 and shape[0] % 2 == 0:
            mode = 6
        if mode == 10 and shape[0] < 3:
            mode = 6
        required = reserved_color if i == 0 else None
        puzzle["test"].append(_new_pair(
            rng, shape, mode, list(range(10)), used, required=required
        ))
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid: np.ndarray, label: str) -> None:
    _require(isinstance(grid, np.ndarray), f"{label} must be a NumPy array")
    _require(grid.ndim == 2, f"{label} must be two-dimensional")
    _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30,
             f"{label} size must be in [1, 30]")
    _require(np.issubdtype(grid.dtype, np.integer), f"{label} dtype must be integer")
    _require(bool(np.all((0 <= grid) & (grid <= 9))), f"{label} colors must be in [0, 9]")


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "invalid pair counts")
    used: set = set()
    for index, pair in enumerate(puzzle["train"] + puzzle["test"]):
        _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                 f"pair {index} has invalid keys")
        grid, output = pair["input"], pair["output"]
        _validate_grid(grid, f"pair {index} input")
        _validate_grid(output, f"pair {index} output")
        _require(grid.shape == output.shape, "shape changed")
        _require(np.array_equal(output, grid[::-1, :]), "not a whole-grid row reversal")
        _require(_key(grid) not in used, "duplicate input")
        _require(_key(grid) not in _SOURCE_KEYS, "source pair copied")
        used.add(_key(grid))
    _require(any(_is_witness(pair["input"]) for pair in puzzle["train"]),
             "training does not distinguish the vertical flip from common alternatives")
    _require(all(not np.array_equal(pair["input"], pair["output"])
                 for pair in puzzle["train"]),
             "training examples must visibly demonstrate the transformation")


def _puzzles_equal(a: dict, b: dict) -> bool:
    return all(
        len(a[split]) == len(b[split]) and all(
            np.array_equal(x["input"], y["input"]) and
            np.array_equal(x["output"], y["output"]) and
            x["input"].dtype == y["input"].dtype and
            x["output"].dtype == y["output"].dtype
            for x, y in zip(a[split], b[split])
        ) for split in ("train", "test")
    )


def self_test() -> None:
    _require(_puzzles_equal(generate_puzzle(314159), generate_puzzle(314159)),
             "automatic generation is nondeterministic")
    _require(_puzzles_equal(generate_puzzle(271828, 7, 3),
                            generate_puzzle(271828, 7, 3)),
             "explicit generation is nondeterministic")
    for i, (train, test) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(5000 + i, train, test)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train, test),
                 "explicit count not honored")
    for i, (train, test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(6000 + i, train, test)
        _require((train is None or len(puzzle["train"]) == train)
                 and (test is None or len(puzzle["test"]) == test),
                 "partial count not honored")
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (2, 3), (4, 4), (3.5, 1), (3, 1.5)):
        try:
            generate_puzzle(7000, train, test)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")

    train_counts, test_counts, puzzle_keys = set(), set(), set()
    shapes, palettes, layouts = set(), set(), set()
    one_size = multiple_size = source_squares = thin = large = 0
    repeated = unchanged_test = pair_palette_changes = test_only = 0
    odd_only = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_keys.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        all_pairs = puzzle["train"] + puzzle["test"]
        current_shapes = {p["input"].shape for p in all_pairs}
        one_size += len(current_shapes) == 1
        multiple_size += len(current_shapes) >= 3
        odd_only += all(p["input"].shape[0] % 2 for p in puzzle["train"])
        current_palettes = {tuple(int(c) for c in np.unique(p["input"])) for p in all_pairs}
        pair_palette_changes += len(current_palettes) > 1
        train_colors = {int(c) for p in puzzle["train"] for c in np.unique(p["input"])}
        test_colors = {int(c) for p in puzzle["test"] for c in np.unique(p["input"])}
        test_only += bool(test_colors - train_colors)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, output = pair["input"], pair["output"]
                h, w = grid.shape
                shapes.add((h, w))
                palettes.add(tuple(int(c) for c in np.unique(grid)))
                layouts.add((h, w, len(np.unique(grid)),
                             int(np.count_nonzero(grid[:, 1:] == grid[:, :-1])),
                             int(np.count_nonzero(grid[1:] == grid[:-1]))))
                source_squares += (h, w) in ((5, 5), (7, 7))
                thin += w <= 2
                large += h >= 16 or w >= 16
                repeated += len({tuple(row) for row in grid}) < h
                _require(grid.shape == output.shape and
                         np.array_equal(output, np.flipud(grid)),
                         "self-test found rule drift")
                _require(np.array_equal(np.sort(grid.ravel()), np.sort(output.ravel())),
                         "self-test found a color change")
                _require(_key(grid) not in _SOURCE_KEYS, "source copy")
        unchanged_test += sum(np.array_equal(p["input"], p["output"]) for p in puzzle["test"])
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "count diversity too low")
    _require(len(puzzle_keys) == 42 and len(shapes) >= 45 and len(layouts) >= 120,
             "structural diversity too low")
    _require(len(palettes) >= 40, "pair palette diversity too low")
    _require(one_size >= 2 and multiple_size >= 15, "shape regime diversity too low")
    _require(source_squares >= 20 and thin >= 5 and large >= 10,
             "dimension coverage too low")
    _require(repeated >= 12 and unchanged_test >= 1, "row structure coverage too low")
    _require(pair_palette_changes >= 35 and test_only >= 6,
             "color variation coverage too low")
    _require(odd_only >= 1, "parity plan remains fixed")
