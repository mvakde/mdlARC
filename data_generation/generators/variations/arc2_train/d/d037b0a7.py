"""Synthetic instances of the columnwise downward-bar ARC rule."""

import numpy as np


_SOURCE_PAIRS = frozenset(
    {
        ((3, 3), (0, 2, 0, 7, 0, 8, 0, 0, 0), (0, 2, 0, 7, 2, 8, 7, 2, 8)),
        ((3, 3), (0, 0, 6, 0, 4, 0, 3, 0, 0), (0, 0, 6, 0, 4, 6, 3, 4, 6)),
        ((3, 3), (4, 0, 0, 0, 2, 0, 0, 0, 0), (4, 0, 0, 4, 2, 0, 4, 2, 0)),
        ((3, 3), (4, 0, 8, 0, 0, 0, 0, 7, 0), (4, 0, 8, 4, 0, 8, 4, 7, 8)),
    }
)
_SOURCE_TEST_LAYOUT = ((0, 0), (0, 2), (2, 1))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _background(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    index = int(np.argmax(counts))
    _require(
        int(counts[index]) > grid.size // 2,
        "input has no unambiguous majority background",
    )
    return int(colors[index])


def _project_down(grid: np.ndarray) -> np.ndarray:
    background = _background(grid)
    output = grid.copy()
    for column in range(grid.shape[1]):
        rows = np.flatnonzero(grid[:, column] != background)
        _require(rows.size <= 1, "more than one seed in a column")
        if rows.size:
            row = int(rows[0])
            output[row:, column] = grid[row, column]
    return output


def _input_key(grid: np.ndarray) -> tuple:
    return (tuple(int(x) for x in grid.shape), tuple(int(x) for x in grid.flat))


def _pair_key(pair: dict) -> tuple:
    return (_input_key(pair["input"]), _input_key(pair["output"]))


def _source_key(pair: dict) -> tuple:
    return (
        tuple(int(x) for x in pair["input"].shape),
        tuple(int(x) for x in pair["input"].flat),
        tuple(int(x) for x in pair["output"].flat),
    )


def _layout_key(grid: np.ndarray) -> tuple:
    background = _background(grid)
    positions = tuple(
        (int(row), int(column))
        for row, column in np.argwhere(grid != background)
    )
    return (tuple(int(x) for x in grid.shape), positions)


def _checked_count(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train_count = _checked_count(num_train, "num_train")
    test_count = _checked_count(num_test, "num_test")
    if train_count is not None and not 2 <= train_count <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if test_count is not None and not 1 <= test_count <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if train_count is not None and test_count is not None and train_count <= test_count:
        raise ValueError("num_train must exceed num_test")
    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        train_count = test_count + 1 + (seed // 4) % (10 - test_count)
    elif train_count is None:
        train_count = test_count + 1 + seed % (10 - test_count)
    elif test_count is None:
        test_count = 1 + seed % min(4, train_count - 1)
    return train_count, test_count


def _sample_background(rng: np.random.Generator) -> int:
    if rng.random() < 0.55:
        return 0
    return int(rng.integers(1, 10))


def _sample_shape(rng: np.random.Generator) -> tuple[int, int]:
    if rng.random() < 0.30:
        return (3, 3)
    return (int(rng.integers(3, 9)), int(rng.integers(3, 10)))


def _sample_columns(rng: np.random.Generator, width: int, count: int) -> tuple[int, ...]:
    if count == 2 and width >= 5 and rng.random() < 0.35:
        return (0, width - 1)
    return tuple(sorted(int(x) for x in rng.choice(width, size=count, replace=False)))


def _sample_rows(
    rng: np.random.Generator,
    height: int,
    count: int,
    regime: str,
) -> list[int]:
    if regime == "target_test" and count >= 2:
        mode = int(rng.choice(4, p=(0.32, 0.25, 0.27, 0.16)))
        if mode == 0:
            rows = [0, height - 1]
            rows.extend(int(x) for x in rng.integers(0, height, size=count - 2))
            rng.shuffle(rows)
        elif mode == 1:
            shared = int(rng.integers(0, height))
            rows = [shared, shared]
            rows.extend(int(x) for x in rng.integers(0, height, size=count - 2))
            rng.shuffle(rows)
        elif mode == 2:
            if count <= height:
                rows = [int(x) for x in rng.choice(height, size=count, replace=False)]
            else:
                rows = [int(x) for x in rng.integers(0, height, size=count)]
        else:
            rows = [int(x) for x in rng.integers(0, height, size=count)]
    else:
        rows = [int(x) for x in rng.integers(0, height, size=count)]
    if all(row == height - 1 for row in rows):
        rows[int(rng.integers(count))] = int(rng.integers(0, height - 1))
    return rows


def _sample_colors(
    rng: np.random.Generator,
    background: int,
    count: int,
    familiar: set[int] | None,
) -> tuple[int, ...]:
    available = [color for color in range(10) if color != background]
    known = sorted(set(available) & familiar) if familiar is not None else []
    colors = []
    for _ in range(count):
        if known and rng.random() < 0.70:
            colors.append(int(rng.choice(known)))
        else:
            colors.append(int(rng.choice(available)))
    if count >= 2 and rng.random() < 0.28:
        first, second = (int(x) for x in rng.choice(count, size=2, replace=False))
        colors[second] = colors[first]
    return tuple(colors)


def _make_pair(
    rng: np.random.Generator,
    shape: tuple[int, int],
    seed_count: int,
    background: int,
    regime: str,
    seen_inputs: set[tuple],
    seen_layouts: set[tuple],
    required_rows: tuple[str, ...] = (),
    familiar: set[int] | None = None,
    source_layout: bool = False,
    force_zero_seed: bool = False,
) -> dict:
    height, width = shape
    for _ in range(200):
        columns = _sample_columns(rng, width, seed_count)
        rows = _sample_rows(rng, height, seed_count, regime)
        if source_layout:
            columns = (0, 1, 2)
            rows = [0, 2, 0]
        elif required_rows:
            slots = rng.choice(seed_count, size=len(required_rows), replace=False)
            for slot, requirement in zip(slots, required_rows):
                rows[int(slot)] = (
                    0 if requirement == "top"
                    else height - 1 if requirement == "bottom"
                    else int(rng.integers(1, height - 1))
                )
        if all(row == height - 1 for row in rows):
            if "bottom" in required_rows:
                continue
            rows[int(rng.integers(seed_count))] = 0
        colors = list(_sample_colors(rng, background, seed_count, familiar))
        if force_zero_seed:
            extending_slots = [index for index, row in enumerate(rows) if row < height - 1]
            colors[int(rng.choice(extending_slots))] = 0
        grid = np.full(shape, background, dtype=np.int8)
        for column, row, color in zip(columns, rows, colors):
            grid[row, column] = color
        pair = {"input": grid, "output": _project_down(grid)}
        input_key = _input_key(grid)
        layout_key = _layout_key(grid)
        if (
            input_key not in seen_inputs
            and layout_key not in seen_layouts
            and _source_key(pair) not in _SOURCE_PAIRS
        ):
            seen_inputs.add(input_key)
            seen_layouts.add(layout_key)
            return pair
    raise ValueError("unable to construct a distinct pair")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)

    # Downward projection and majority-background identification are the
    # only puzzle-wide conventions. Background and literal bar colors can
    # change from pair to pair; both background regimes appear in training.
    train_shapes = [(3, 3), (int(rng.integers(4, 9)), int(rng.integers(4, 10)))]
    train_shapes.extend(_sample_shape(rng) for _ in range(train_count - 2))
    rng.shuffle(train_shapes)
    mandatory_counts = [2, 3]
    rng.shuffle(mandatory_counts)
    train_seed_counts = mandatory_counts + [
        int(rng.choice([1, 2, 3, 4, 5], p=[0.08, 0.35, 0.36, 0.15, 0.06]))
        for _ in range(train_count - 2)
    ]
    train_seed_counts = [
        min(count, shape[1]) for count, shape in zip(train_seed_counts, train_shapes)
    ]
    # Distribute the three height demonstrations among the five slots in
    # the two mandatory examples instead of assigning one fixed pair.
    slots = [(pair_index, slot) for pair_index in range(2)
             for slot in range(train_seed_counts[pair_index])]
    selected = rng.choice(len(slots), size=3, replace=False)
    row_requirements = [[], []]
    for chosen, requirement in zip(selected, ("top", "middle", "bottom")):
        row_requirements[slots[int(chosen)][0]].append(requirement)
    # A zero and a nonzero background occur in randomly ordered training pairs.
    first_background = int(rng.integers(1, 10))
    train_backgrounds = [0, first_background]
    rng.shuffle(train_backgrounds)
    train_backgrounds.extend(_sample_background(rng) for _ in range(train_count - 2))

    pair_seeds = rng.integers(0, 2**63, size=train_count + test_count, dtype=np.uint64)
    pair_rngs = [np.random.default_rng(int(x)) for x in pair_seeds]
    seen_inputs: set[tuple] = set()
    seen_layouts: set[tuple] = set()
    train = []
    for index, shape in enumerate(train_shapes):
        train.append(
            _make_pair(
                pair_rngs[index], shape, train_seed_counts[index],
                train_backgrounds[index], "source_train", seen_inputs,
                seen_layouts, tuple(row_requirements[index]) if index < 2 else (),
                force_zero_seed=index < 2 and train_backgrounds[index] != 0,
            )
        )
    rng.shuffle(train)
    familiar = {
        int(pair["input"][row, column])
        for pair in train
        for row, column in np.argwhere(pair["input"] != _background(pair["input"]))
    }
    max_height = max(pair["input"].shape[0] for pair in train)
    max_width = max(pair["input"].shape[1] for pair in train)
    test = []
    for offset in range(test_count):
        local_rng = pair_rngs[train_count + offset]
        source_layout = (
            seed % 11 == 0 and offset == 0
            and ((3, 3), _SOURCE_TEST_LAYOUT) not in seen_layouts
        )
        if source_layout:
            shape = (3, 3)
            count = 3
        elif local_rng.random() < 0.25:
            shape = (3, 3)
            count = int(local_rng.choice([1, 2, 3], p=[0.08, 0.46, 0.46]))
        else:
            shape = (
                int(local_rng.integers(3, max_height + 1)),
                int(local_rng.integers(3, max_width + 1)),
            )
            count = int(local_rng.choice([1, 2, 3, 4, 5], p=[0.07, 0.39, 0.38, 0.12, 0.04]))
            count = min(count, shape[1])
        background = _sample_background(local_rng)
        test.append(
            _make_pair(
                local_rng, shape, count, background, "target_test",
                seen_inputs, seen_layouts, familiar=familiar,
                source_layout=source_layout,
            )
        )
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train and test")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list), "splits must be lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "illegal pair counts")
    seen_inputs: set[tuple] = set()
    seen_pairs: set[tuple] = set()
    seen_layouts: set[tuple] = set()
    train_counts: set[int] = set()
    train_tiers: set[str] = set()
    train_backgrounds: set[int] = set()
    train_zero_extended = False
    train_shapes: set[tuple[int, int]] = set()
    test_shapes: list[tuple[int, int]] = []
    for split_name, pairs in (("train", train), ("test", test)):
        for index, pair in enumerate(pairs):
            label = f"{split_name}[{index}]"
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     f"{label} must contain input and output")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         f"{label} grid must be a 2-D NumPy array")
                _require(np.issubdtype(grid.dtype, np.integer), f"{label} must be integer")
                _require(all(1 <= extent <= 30 for extent in grid.shape),
                         f"{label} dimensions out of range")
                _require(bool(np.all((0 <= grid) & (grid <= 9))), f"{label} color out of range")
            _require(inp.shape == out.shape, f"{label} changes dimensions")
            _require(inp.shape[0] >= 3 and inp.shape[1] >= 3,
                     f"{label} lacks a clear sparse-seed canvas")
            background = _background(inp)
            positions = [(int(row), int(column)) for row, column in np.argwhere(inp != background)]
            _require(1 <= len(positions) <= min(5, inp.shape[1]),
                     f"{label} has an unsupported seed count")
            _require(len({column for _, column in positions}) == len(positions),
                     f"{label} has two seeds in one column")
            _require(any(row < inp.shape[0] - 1 for row, _ in positions),
                     f"{label} has no visible completion")
            _require(np.array_equal(out, _project_down(inp)),
                     f"{label} violates downward projection")
            _require(not np.array_equal(inp, out), f"{label} is unchanged")
            input_key, pair_key, layout_key = _input_key(inp), _pair_key(pair), _layout_key(inp)
            _require(input_key not in seen_inputs and pair_key not in seen_pairs,
                     f"{label} is duplicated")
            _require(layout_key not in seen_layouts, f"{label} repeats a layout")
            _require(_source_key(pair) not in _SOURCE_PAIRS, f"{label} copies a source pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            seen_layouts.add(layout_key)
            if split_name == "train":
                train_counts.add(len(positions))
                train_backgrounds.add(background)
                train_zero_extended |= any(
                    int(inp[row, column]) == 0 and row < inp.shape[0] - 1
                    for row, column in positions
                )
                train_shapes.add(tuple(int(x) for x in inp.shape))
                for row, _ in positions:
                    train_tiers.add(
                        "top" if row == 0 else "bottom" if row == inp.shape[0] - 1 else "middle"
                    )
            else:
                test_shapes.append(tuple(int(x) for x in inp.shape))
    _require({2, 3} <= train_counts, "training must show two and three seeds")
    _require(train_tiers == {"top", "middle", "bottom"},
             "training must show top, interior, and bottom starts")
    _require(0 in train_backgrounds and any(color != 0 for color in train_backgrounds),
             "training must show zero and nonzero backgrounds")
    _require(train_zero_extended,
             "training must show a zero-colored seed visibly extending")
    _require((3, 3) in train_shapes and any(h > 3 or w > 3 for h, w in train_shapes),
             "training must show source-sized and expanded canvases")
    max_height = max(shape[0] for shape in train_shapes)
    max_width = max(shape[1] for shape in train_shapes)
    _require(all(h <= max_height and w <= max_width for h, w in test_shapes),
             "test introduces an undemonstrated size range")


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(
        len(first[split]) == len(second[split])
        and all(
            np.array_equal(a["input"], b["input"])
            and np.array_equal(a["output"], b["output"])
            for a, b in zip(first[split], second[split])
        )
        for split in ("train", "test")
    )


def _independent_projection(grid: np.ndarray) -> np.ndarray:
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    result = np.full(grid.shape, background, dtype=grid.dtype)
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            if grid[row, column] != background:
                for below in range(row, grid.shape[0]):
                    result[below, column] = grid[row, column]
    return result


def self_test() -> None:
    a = generate_puzzle(314159, 7, 3)
    b = generate_puzzle(314159, 7, 3)
    _require(_puzzles_equal(a, b), "same seed and counts changed output")
    _require(not _puzzles_equal(a, generate_puzzle(314160, 7, 3)),
             "different seeds produced the same puzzle")
    requests = [(2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)]
    for offset, (num_train, num_test) in enumerate(requests):
        puzzle = generate_puzzle(2000 + offset, num_train, num_test)
        validate_generated_puzzle(puzzle)
        if num_train is not None:
            _require(len(puzzle["train"]) == num_train, "explicit train count ignored")
        if num_test is not None:
            _require(len(puzzle["test"]) == num_test, "explicit test count ignored")
    for kwargs in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0},
        {"num_test": 5}, {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
    ):
        try:
            generate_puzzle(99, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal request accepted: {kwargs}")
    train_counts, test_counts = set(), set()
    shapes, layouts, backgrounds = set(), set(), set()
    seed_counts, bar_lengths, palettes = set(), set(), set()
    repeated_colors = bottom_seeds = zero_seeds = 0
    source_layouts = within_background_changes = 0
    puzzle_keys = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_keys.add(tuple(
            (split, tuple(_pair_key(pair) for pair in puzzle[split]))
            for split in ("train", "test")
        ))
        pair_backgrounds = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                _require(np.array_equal(_independent_projection(inp), out),
                         "independent downward reconstruction failed")
                background = _background(inp)
                pair_backgrounds.add(background)
                backgrounds.add(background)
                shapes.add(tuple(int(x) for x in inp.shape))
                layouts.add(_layout_key(inp))
                positions = [(int(r), int(c)) for r, c in np.argwhere(inp != background)]
                colors = tuple(int(inp[r, c]) for r, c in positions)
                palettes.add((background, tuple(sorted(set(colors)))))
                seed_counts.add(len(positions))
                repeated_colors += len(set(colors)) < len(colors)
                bottom_seeds += any(r == inp.shape[0] - 1 for r, _ in positions)
                zero_seeds += 0 in colors
                source_layouts += inp.shape == (3, 3) and tuple(positions) == _SOURCE_TEST_LAYOUT
                bar_lengths.update(inp.shape[0] - r for r, _ in positions)
        within_background_changes += len(pair_backgrounds) > 1
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "automatic pair counts lack diversity")
    _require(len(puzzle_keys) == 42 and len(shapes) >= 15 and len(layouts) >= 100,
             "insufficient structural diversity")
    _require({1, 2, 3, 4} <= seed_counts and len(bar_lengths) >= 6,
             "seed counts or bar lengths lack diversity")
    _require(len(backgrounds) >= 8 and len(palettes) >= 80,
             "background or color choices lack diversity")
    _require(repeated_colors >= 10 and bottom_seeds >= 20 and zero_seeds >= 10,
             "color repetition or bottom/zero seeds are too rare")
    _require(source_layouts >= 2 and within_background_changes >= 30,
             "source layout or within-puzzle background variation missing")
