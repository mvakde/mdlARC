import numpy as np


# Exact inputs from the five source pairs. The rule, rather than their size or
# palette, is what new puzzles inherit.
_SOURCE_INPUTS = (
    ((8, 8, 8), (5, 5, 7), (5, 7, 8)),
    ((7, 7, 8), (5, 8, 8), (5, 8, 8)),
    ((8, 8, 7), (7, 5, 5), (5, 7, 8)),
    ((7, 5, 7), (5, 5, 7), (7, 7, 5)),
    ((8, 5, 7), (5, 7, 5), (8, 8, 5)),
)
_SOURCE_KEYS = frozenset((3, 3, tuple(sum(rows, ()))) for rows in _SOURCE_INPUTS)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _key(grid: np.ndarray) -> tuple:
    return (grid.shape[0], grid.shape[1], tuple(int(x) for x in grid.flat))


def _output(grid: np.ndarray) -> np.ndarray:
    mirror = grid[:, ::-1]
    return np.concatenate((mirror, grid, mirror, grid), axis=1)


def _count(value: int | None, name: str, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _counts(rng: np.random.Generator, seed: int, train: int | None,
            test: int | None) -> tuple[int, int]:
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
        return train, test
    if train is None:
        return int(rng.integers(test + 1, 11)), test
    return train, int(rng.integers(1, min(4, train - 1) + 1))


def _size(rng: np.random.Generator, source_like: bool,
          min_width: int = 1) -> tuple[int, int]:
    if source_like:
        return 3, 3
    heights = np.asarray((2, 3, 3, 3, 4, 5, 6, 7, 8, 9, 11, 14))
    widths = np.asarray((1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 7))
    height = int(rng.choice(heights))
    allowed = widths[widths >= min_width]
    return height, int(rng.choice(allowed))


def _color_count(rng: np.random.Generator, height: int, width: int,
                 available: int, source_like: bool, regime: str) -> int:
    maximum = min(height * width, available, 9)
    if regime == "target_test" and rng.random() < 0.04:
        return 1
    if source_like:
        return min(maximum, int(rng.choice((2, 2, 3, 3, 4))))
    if maximum >= 5 and rng.random() < 0.28:
        return int(rng.integers(5, maximum + 1))
    return min(maximum, int(rng.choice((2, 2, 3, 3, 4))))


def _palette(rng: np.random.Generator, count: int, allowed: np.ndarray,
             forced: int | None = None) -> np.ndarray:
    if forced is None:
        return np.asarray(rng.choice(allowed, size=count, replace=False), dtype=np.int8)
    other = allowed[allowed != forced]
    result = [forced]
    if count > 1:
        result.extend(int(c) for c in rng.choice(other, size=count - 1, replace=False))
    return np.asarray(result, dtype=np.int8)


def _draw_row(rng: np.random.Generator, width: int,
              colors: np.ndarray, kind: int) -> np.ndarray:
    if kind == 0:  # solid
        return np.full(width, rng.choice(colors), dtype=np.int8)
    if kind == 1:  # palindrome, possibly nonuniform
        half = np.asarray(rng.choice(colors, size=(width + 1) // 2), dtype=np.int8)
        return np.concatenate((half, half[:width // 2][::-1]))
    if kind == 2:  # alternating or short-period stripe
        if width == 1:
            return np.asarray((rng.choice(colors),), dtype=np.int8)
        period = min(width, int(rng.integers(2, min(4, width) + 1)))
        return np.asarray(np.resize(rng.choice(colors, size=period), width), dtype=np.int8)
    if kind == 3:  # a block on a field
        row = np.full(width, rng.choice(colors), dtype=np.int8)
        start = int(rng.integers(0, width))
        end = int(rng.integers(start + 1, width + 1))
        row[start:end] = rng.choice(colors)
        return row
    return np.asarray(rng.choice(colors, size=width), dtype=np.int8)


def _motif(rng: np.random.Generator, height: int, width: int,
           colors: np.ndarray, style: int, symmetry: int,
           need_witness: bool) -> np.ndarray:
    if style == 0:  # irregular pixels
        grid = np.asarray(rng.choice(colors, size=(height, width)), dtype=np.int8)
    elif style == 1:  # sparse colored accents
        grid = np.full((height, width), rng.choice(colors), dtype=np.int8)
        for _ in range(max(1, height * width // 5)):
            grid[int(rng.integers(height)), int(rng.integers(width))] = rng.choice(colors)
    elif style == 2:  # vertical columns, with occasional local changes
        pattern = np.asarray(rng.choice(colors, size=width), dtype=np.int8)
        grid = np.tile(pattern, (height, 1))
        for _ in range(int(rng.integers(0, max(2, height // 2 + 1)))):
            grid[int(rng.integers(height)), int(rng.integers(width))] = rng.choice(colors)
    elif style == 3:  # filled rectangular regions
        grid = np.full((height, width), rng.choice(colors), dtype=np.int8)
        for _ in range(int(rng.integers(1, 4))):
            top = int(rng.integers(height))
            left = int(rng.integers(width))
            bottom = int(rng.integers(top + 1, height + 1))
            right = int(rng.integers(left + 1, width + 1))
            grid[top:bottom, left:right] = rng.choice(colors)
    elif style == 4:  # repeated rows with variation
        row = _draw_row(rng, width, colors, int(rng.integers(2, 5)))
        grid = np.tile(row, (height, 1))
        for _ in range(int(rng.integers(1, max(2, height // 2 + 1)))):
            grid[int(rng.integers(height))] = _draw_row(rng, width, colors, int(rng.integers(5)))
    else:  # independently composed rows
        grid = np.stack([_draw_row(rng, width, colors, int(rng.integers(5)))
                         for _ in range(height)])

    # Row operations can coexist with the underlying columns, blocks, and pixels.
    if style not in (4, 5) and rng.random() < 0.35:
        row = int(rng.integers(height))
        grid[row] = _draw_row(rng, width, colors, int(rng.integers(5)))
    if height > 1 and style != 4 and rng.random() < 0.20:
        source, target = rng.choice(height, size=2, replace=False)
        grid[target] = grid[source]

    if symmetry == 1 and width > 1:  # a symmetric middle row in either split
        row = height // 2
        grid[row, width // 2:] = grid[row, :width // 2][::-1] if width % 2 == 0 else grid[row, :width // 2 + 1][::-1]
    elif symmetry == 2 and width > 1:  # every row asymmetric
        for row in range(height):
            if np.array_equal(grid[row], grid[row, ::-1]):
                choices = colors[colors != grid[row, 0]]
                if len(choices):
                    grid[row, -1] = rng.choice(choices)
    elif symmetry == 3:  # an entirely symmetric motif
        for row in range(height):
            for col in range(width // 2):
                grid[row, width - 1 - col] = grid[row, col]

    # Realize the chosen palette. Preserve a solid row if there is another row
    # available for missing colors.
    missing = [int(c) for c in colors if c not in grid]
    positions = [(r, c) for r in range(height) for c in range(width)]
    if symmetry == 3:
        positions = [(r, c) for r in range(height) for c in range((width + 1) // 2)]
    if len(missing) <= len(positions):
        for color, position in zip(missing, rng.permutation(len(positions))):
            row, col = positions[int(position)]
            grid[row, col] = color
            if symmetry == 3:
                grid[row, width - 1 - col] = color

    if need_witness and width > 1 and np.array_equal(grid, grid[:, ::-1]):
        row = int(rng.integers(height))
        choices = colors[colors != grid[row, 0]]
        grid[row, -1] = rng.choice(choices)
    return grid


def _construct_pair(rng: np.random.Generator, regime: str, size: tuple[int, int],
                    allowed: np.ndarray, source_like: bool, used: set,
                    forced_color: int | None, need_witness: bool) -> dict:
    height, width = size
    count = _color_count(rng, height, width, len(allowed), source_like, regime)
    if need_witness:
        count = max(2, count)
    if forced_color is not None:
        count = max(2, count)
    colors = _palette(rng, count, allowed, forced_color)
    for _ in range(64):
        style = int(rng.choice((0, 0, 0, 1, 2, 3, 4, 5, 5)))
        symmetry = int(rng.choice((0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3)))
        if need_witness and symmetry == 3:
            symmetry = 0
        grid = _motif(rng, height, width, colors, style, symmetry, need_witness)
        if forced_color is not None and forced_color not in grid:
            grid[0, 0] = forced_color
            if symmetry == 3:
                grid[0, -1] = forced_color
        key = _key(grid)
        if key not in used and key not in _SOURCE_KEYS:
            used.add(key)
            return {"input": grid, "output": _output(grid)}

    # A bounded systematic fallback covers the tiny motifs where collisions are
    # most plausible. Its digits remain drawn from this pair's palette.
    for number in range(10000):
        grid = np.full((height, width), colors[0], dtype=np.int8)
        code = number
        for position in range(min(height * width, 5)):
            row, col = divmod(position, width)
            grid[row, col] = colors[code % len(colors)]
            code //= len(colors)
        if need_witness and np.array_equal(grid, grid[:, ::-1]):
            continue
        key = _key(grid)
        if key not in used and key not in _SOURCE_KEYS:
            used.add(key)
            return {"input": grid, "output": _output(grid)}
    raise RuntimeError("could not construct a unique motif")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    train_arg = _count(num_train, "num_train", 2, 10)
    test_arg = _count(num_test, "num_test", 1, 4)
    if train_arg is not None and test_arg is not None and train_arg <= test_arg:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _counts(rng, int(seed), train_arg, test_arg)

    # The order of four panels is the sole puzzle-level convention. A reserved
    # color is only a construction device for occasional unseen test colors;
    # colors themselves have no role in the rule.
    source_like = bool(rng.random() < 0.14)
    reserved = int(rng.integers(10)) if rng.random() < 0.42 else None
    train_colors = np.asarray([c for c in range(10) if c != reserved], dtype=np.int8)
    all_colors = np.arange(10, dtype=np.int8)

    train_sizes = [_size(rng, source_like, min_width=2 if i == 0 else 1)
                   for i in range(train_count)]
    if not source_like and train_count >= 2:
        first_h, first_w = train_sizes[0]
        heights = [h for h in (2, 3, 4, 5, 6, 7, 8, 9, 11, 14) if h != first_h]
        widths = [w for w in (2, 3, 4, 5, 6, 7) if w != first_w]
        train_sizes[1] = (int(rng.choice(heights)), int(rng.choice(widths)))

    used: set = set()
    train: list[dict] = []
    for i, size in enumerate(train_sizes):
        train.append(_construct_pair(rng, "source_train", size, train_colors,
                                     source_like, used, None, i == 0))

    test: list[dict] = []
    for i in range(test_count):
        size = _size(rng, source_like)
        if not source_like and rng.random() < 0.58:
            seen_sizes = set(train_sizes)
            for _ in range(16):
                candidate = _size(rng, False)
                if candidate not in seen_sizes:
                    size = candidate
                    break
        forced = reserved if i == 0 else None
        test.append(_construct_pair(rng, "target_test", size, all_colors,
                                    source_like, used, forced, False))

    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_grid(grid: object, label: str) -> None:
    _require(isinstance(grid, np.ndarray), f"{label} is not a NumPy array")
    _require(grid.ndim == 2, f"{label} is not two-dimensional")
    _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30,
             f"{label} has invalid dimensions")
    _require(np.issubdtype(grid.dtype, np.integer), f"{label} has noninteger cells")
    _require(0 <= int(grid.min()) and int(grid.max()) <= 9,
             f"{label} has invalid colors")


def _learn_panels(train: list[dict]) -> tuple[str, ...]:
    # Infer a usable rule from returned demonstrations alone. Symmetric rows
    # are neutral; at least one training motif must distinguish both phases.
    learned: tuple[str, ...] | None = None
    for pair in train:
        inp, out = pair["input"], pair["output"]
        width = inp.shape[1]
        if np.array_equal(inp, inp[:, ::-1]):
            continue
        panels = tuple(out[:, k * width:(k + 1) * width] for k in range(4))
        kinds = tuple("M" if np.array_equal(panel, inp[:, ::-1]) else
                      "I" if np.array_equal(panel, inp) else "?" for panel in panels)
        _require("?" not in kinds, "training contains a foreign panel")
        if learned is None:
            learned = kinds
        else:
            _require(kinds == learned, "training panels disagree")
    _require(learned is not None, "training lacks an asymmetric phase witness")
    return learned


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list),
             "train and test must be lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "invalid pair counts")
    seen: set = set()
    for split_name, pairs in (("train", train), ("test", test)):
        for index, pair in enumerate(pairs):
            label = f"{split_name}[{index}]"
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     f"{label} has invalid keys")
            inp, out = pair["input"], pair["output"]
            _check_grid(inp, label + ".input")
            _check_grid(out, label + ".output")
            height, width = inp.shape
            _require(width <= 7 and out.shape == (height, 4 * width),
                     f"{label} has invalid four-panel dimensions")
            _require(inp.dtype == out.dtype, f"{label} changes dtype")
            _require(np.array_equal(out, _output(inp)), f"{label} violates the rule")
            _require(set(int(x) for x in np.unique(inp)) ==
                     set(int(x) for x in np.unique(out)), f"{label} changes colors")
            key = _key(inp)
            _require(key not in seen, f"{label} duplicates an input")
            _require(key not in _SOURCE_KEYS, f"{label} copies a source pair")
            seen.add(key)

    learned = _learn_panels(train)
    _require(learned == ("M", "I", "M", "I"), "training does not show mirror first")
    for index, pair in enumerate(test):
        inp = pair["input"]
        prediction = np.concatenate(tuple(inp[:, ::-1] if kind == "M" else inp
                                          for kind in learned), axis=1)
        _require(np.array_equal(prediction, pair["output"]),
                 f"test[{index}] is not recoverable from training")


def _same_puzzle(left: dict, right: dict) -> bool:
    return all(len(left[split]) == len(right[split]) and
               all(np.array_equal(a[grid], b[grid]) for a, b in
                   zip(left[split], right[split]) for grid in ("input", "output"))
               for split in ("train", "test"))


def self_test() -> None:
    for seed, train_arg, test_arg in ((17, None, None), (314159, 7, 3),
                                      (5000, 2, 1), (5001, 10, 1), (5002, 10, 4),
                                      (6000, None, 1), (6001, None, 4),
                                      (6002, 2, None), (6003, 10, None)):
        puzzle = generate_puzzle(seed, train_arg, test_arg)
        validate_generated_puzzle(puzzle)
        _require(_same_puzzle(puzzle, generate_puzzle(seed, train_arg, test_arg)),
                 "generation is not deterministic")
        if train_arg is not None:
            _require(len(puzzle["train"]) == train_arg, "train count ignored")
        if test_arg is not None:
            _require(len(puzzle["test"]) == test_arg, "test count ignored")
    _require(not _same_puzzle(generate_puzzle(100), generate_puzzle(101)),
             "different seeds produced the same puzzle")
    for train_arg, test_arg in ((1, None), (11, None), (None, 0), (None, 5),
                                (2, 2), (2, 3), (4, 4)):
        try:
            generate_puzzle(7, train_arg, test_arg)
        except ValueError:
            continue
        raise AssertionError("illegal count request was accepted")
    try:
        validate_generated_puzzle({"train": [], "test": []})
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted an empty puzzle")

    train_counts: set = set()
    test_counts: set = set()
    sizes: set = set()
    heights: set = set()
    widths: set = set()
    color_counts: set = set()
    palettes: set = set()
    layouts: set = set()
    middle = {"train": set(), "test": set()}
    varied_train_sizes = uniform_train = uniform_test = 0
    repeated_train = repeated_test = whole_symmetric_test = 0
    no_symmetric_train = no_symmetric_test = unseen_test = high_palette = 0
    varied_heights = varied_widths = source_like_puzzles = 0
    new_test_size = new_test_aspect = new_size_unseen = 0
    repeated_columns = sparse_motifs = irregular_motifs = block_motifs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_sizes = {p["input"].shape for p in puzzle["train"]}
        varied_train_sizes += len(train_sizes) > 1
        varied_heights += len({size[0] for size in train_sizes}) > 1
        varied_widths += len({size[1] for size in train_sizes}) > 1
        source_like_puzzles += all(p["input"].shape == (3, 3)
                                   for split in ("train", "test") for p in puzzle[split])
        train_aspects = {h / w for h, w in train_sizes}
        train_colors = set(int(x) for p in puzzle["train"] for x in p["input"].flat)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                height, width = grid.shape
                output = pair["output"]
                for index, expected in enumerate((grid[:, ::-1], grid,
                                                  grid[:, ::-1], grid)):
                    _require(np.array_equal(output[:, index * width:(index + 1) * width],
                                            expected), "self-test found a wrong panel")
                sizes.add(grid.shape)
                heights.add(height)
                widths.add(width)
                values, frequencies = np.unique(grid, return_counts=True)
                count = len(values)
                color_counts.add(count)
                palettes.add(tuple(int(x) for x in values))
                layouts.add(_key(grid))
                high_palette += count >= 5
                sym = [np.array_equal(row, row[::-1]) for row in grid]
                middle[split].add(sym[height // 2])
                rows = [tuple(int(x) for x in row) for row in grid]
                columns = [tuple(int(x) for x in grid[:, column])
                           for column in range(width)]
                solid = any(len(set(row)) == 1 for row in rows)
                repeated = len(set(rows)) < len(rows)
                same_columns = len(set(columns)) < len(columns)
                repeated_columns += same_columns
                sparse_motifs += count >= 2 and max(frequencies) / grid.size >= 0.7
                irregular_motifs += (count >= 3 and not solid and not repeated
                                     and not same_columns and max(frequencies) / grid.size < 0.6)
                block_motifs += count >= 2 and any(
                    np.all(grid[row:row + 2, col:col + 2] == grid[row, col])
                    for row in range(height - 1) for col in range(width - 1))
                if split == "train":
                    uniform_train += solid
                    repeated_train += repeated
                    no_symmetric_train += not any(sym)
                else:
                    uniform_test += solid
                    repeated_test += repeated
                    no_symmetric_test += not any(sym)
                    whole_symmetric_test += all(sym)
                    unseen = bool(set(int(x) for x in grid.flat) - train_colors)
                    new_size = grid.shape not in train_sizes
                    unseen_test += unseen
                    new_test_size += new_size
                    new_test_aspect += height / width not in train_aspects
                    new_size_unseen += new_size and unseen
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "automatic pair counts lack diversity")
    _require(len(sizes) >= 25 and len(heights) >= 8 and len(widths) >= 6,
             "motif dimensions lack diversity")
    _require(varied_train_sizes >= 25, "too few puzzles vary training sizes")
    _require(min(varied_heights, varied_widths) >= 25 and source_like_puzzles >= 1,
             "within-puzzle size mixing or source-like puzzles are missing")
    _require((3, 3) in sizes and {True, False} == middle["train"] == middle["test"],
             "source-like size or middle-row symmetry coverage missing")
    _require(min(uniform_train, uniform_test, repeated_train, repeated_test) >= 3,
             "uniform or repeated rows are too rare")
    _require(min(no_symmetric_train, no_symmetric_test, whole_symmetric_test) >= 3,
             "row-symmetry regimes are too rare")
    _require(unseen_test >= 5 and high_palette >= 10 and max(color_counts) >= 6,
             "palette variation is insufficient")
    _require(min(new_test_size, new_test_aspect) >= 20 and new_size_unseen >= 5,
             "test novelty and combined size/color shifts are too rare")
    _require(min(repeated_columns, sparse_motifs, irregular_motifs, block_motifs) >= 8,
             "motif structures are insufficiently diverse")
    _require(len(palettes) >= 30 and len(layouts) >= 150,
             "motif arrangements or palettes are insufficiently diverse")
