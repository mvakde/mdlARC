"""Generator for self-indexing, one-hole compartment lattices."""

from itertools import combinations

import numpy as np


_SOURCE_SIGNATURES = {
    (2, 0, 8, (0, 0, 1, 1)),
    (3, 0, 2, (0, 1, 1, 0, 1, 0, 0, 1, 1)),
    (4, 0, 1, (0, 0, 1, 0, 0, 1, 1, 1,
               0, 1, 0, 0, 1, 1, 0, 0)),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _validate_count(name: str, value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    low, high = (2, 10) if name == "num_train" else (1, 4)
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(rng: np.random.Generator,
                    num_train: int | None,
                    num_test: int | None) -> tuple[int, int]:
    _validate_count("num_train", num_train)
    _validate_count("num_test", num_test)
    train = None if num_train is None else int(num_train)
    test = None if num_test is None else int(num_test)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if train is None and test is None:
        test = int(rng.integers(1, 5))
    if train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    if test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _side(order: int) -> int:
    return order * order + order - 1


def _order_from_side(side: int) -> int:
    matches = [n for n in range(2, 6) if _side(n) == side]
    _require(len(matches) == 1, "invalid compartment dimension")
    return matches[0]


def _complete_grid(rows: int, cols: int,
                   background: int, foreground: int) -> np.ndarray:
    grid = np.full((_side(rows), _side(cols)), background, dtype=np.int8)
    for tile_row in range(rows):
        row0 = tile_row * (rows + 1)
        for tile_col in range(cols):
            col0 = tile_col * (cols + 1)
            grid[row0:row0 + rows, col0:col0 + cols] = foreground
            grid[row0 + tile_row, col0 + tile_col] = background
    return grid


def _source_signature(rows: int, cols: int, background: int,
                      foreground: int, mask: np.ndarray) -> tuple | None:
    if rows != cols:
        return None
    return (rows, background, foreground,
            tuple(int(value) for value in mask.flat))


def _shape_plan(rng: np.random.Generator, train_count: int,
                test_count: int) -> tuple[list[tuple[int, int]],
                                          list[tuple[int, int]]]:
    """Choose a training anchor and an independently sampled test regime."""
    mode = int(rng.integers(0, 5))
    if mode == 0:  # Source 2/3-to-4 extrapolation.
        train = [(3, 3), (2, 2)]
        pool = [(2, 2), (3, 3)]
        test = [(4, 4)] * test_count
    elif mode == 1:  # Tests at a demonstrated order.
        anchor = int(rng.choice((3, 4)))
        train = [(anchor, anchor), (int(rng.choice((2, 3, 4))),) * 2]
        pool = [(n, n) for n in (2, 3, 4)]
        test = []
    elif mode == 2:  # Smaller tests.
        anchor = int(rng.choice((4, 5)))
        train = [(anchor, anchor), (int(rng.choice((3, 4))),) * 2]
        pool = [(n, n) for n in (3, 4, 5)]
        test = [(int(rng.choice((2, 3))),) * 2 for _ in range(test_count)]
    elif mode == 3:  # Unseen order five.
        anchor = int(rng.choice((3, 4)))
        train = [(anchor, anchor), (int(rng.choice((2, 3, 4))),) * 2]
        pool = [(n, n) for n in (2, 3, 4)]
        test = [(5, 5)]
        test.extend((int(rng.choice((4, 5))),) * 2
                    for _ in range(test_count - 1))
    else:  # Rectangular test with a rectangular training demonstration.
        rectangles = [(r, c) for r in (3, 4, 5)
                      for c in (3, 4, 5) if r != c]
        anchor = rectangles[int(rng.integers(len(rectangles)))]
        train = [anchor, (int(rng.choice((2, 3, 4))),) * 2]
        pool = rectangles + [(2, 2), (3, 3), (4, 4)]
        test = [rectangles[int(rng.integers(len(rectangles)))]]
        test.extend(pool[int(rng.integers(len(pool)))]
                    for _ in range(test_count - 1))
    while len(train) < train_count:
        train.append(pool[int(rng.integers(len(pool)))])
    if mode == 1:
        demonstrated = sorted(set(train))
        test = [demonstrated[int(rng.integers(len(demonstrated)))]
                for _ in range(test_count)]
    return train, test


def _visible_count(rng: np.random.Generator, area: int) -> int:
    if area == 4:
        return int(rng.integers(1, 4))
    difficulty = int(rng.choice(4, p=(0.30, 0.32, 0.28, 0.10)))
    if difficulty == 0:
        return int(rng.integers(1, max(3, area // 3 + 1)))
    if difficulty == 1:
        low = max(2, area // 3)
        high = max(low, min(area - 2, (2 * area) // 3))
        return int(rng.integers(low, high + 1))
    if difficulty == 2:
        return area - int(rng.integers(1, min(4, area - 1) + 1))
    return min(7, area - 1)


def _anchor_mask(rng: np.random.Generator, rows: int,
                 cols: int) -> np.ndarray:
    """Expose three distinct coordinate values on each axis in training."""
    mask = np.zeros((rows, cols), dtype=bool)
    row_choices = rng.choice(rows, size=3, replace=False)
    col_choices = rng.choice(cols, size=3, replace=False)
    for row, col in zip(row_choices, col_choices):
        mask[int(row), int(col)] = True
    extras = int(rng.integers(1, max(2, rows * cols // 3 + 1)))
    remaining = np.argwhere(~mask)
    chosen = rng.choice(len(remaining), size=min(extras, len(remaining) - 1),
                        replace=False)
    for index in np.atleast_1d(chosen):
        row, col = remaining[int(index)]
        mask[int(row), int(col)] = True
    return mask


def _sample_mask(rng: np.random.Generator, rows: int,
                 cols: int, anchor: bool = False) -> np.ndarray:
    if anchor:
        return _anchor_mask(rng, rows, cols)
    visible = _visible_count(rng, rows * cols)
    pattern = int(rng.integers(0, 6))
    coordinates = [(r, c) for r in range(rows) for c in range(cols)]
    noise = rng.random((rows, cols))
    if pattern == 0:  # Scattered.
        priority = lambda r, c: noise[r, c]
    elif pattern == 1:  # Edge strip, horizontal or vertical.
        direction = int(rng.integers(0, 4))
        priority = lambda r, c: (r, rows - 1 - r, c, cols - 1 - c)[direction] + noise[r, c] / 2
    elif pattern == 2:  # Corner cluster.
        corner_r = int(rng.choice((0, rows - 1)))
        corner_c = int(rng.choice((0, cols - 1)))
        priority = lambda r, c: abs(r - corner_r) + abs(c - corner_c) + noise[r, c] / 2
    elif pattern == 3:  # Contiguous patch.
        center_r = int(rng.integers(rows))
        center_c = int(rng.integers(cols))
        priority = lambda r, c: max(abs(r - center_r), abs(c - center_c)) + noise[r, c] / 2
    elif pattern == 4:  # Whole erased lattice row.
        gap = int(rng.integers(rows))
        coordinates = [(r, c) for r, c in coordinates if r != gap]
        visible = min(visible, len(coordinates))
        priority = lambda r, c: noise[r, c]
    else:  # Whole erased lattice column.
        gap = int(rng.integers(cols))
        coordinates = [(r, c) for r, c in coordinates if c != gap]
        visible = min(visible, len(coordinates))
        priority = lambda r, c: noise[r, c]
    coordinates.sort(key=lambda point: priority(*point))
    mask = np.zeros((rows, cols), dtype=bool)
    for row, col in coordinates[:visible]:
        mask[row, col] = True
    return mask


def _mask_key(rows: int, cols: int, background: int,
              foreground: int, mask: np.ndarray) -> tuple:
    return (rows, cols, background, foreground,
            tuple(int(value) for value in mask.flat))


def _choose_mask(rng: np.random.Generator, rows: int, cols: int,
                 background: int, foreground: int,
                 used: set[tuple], anchor: bool) -> np.ndarray:
    for _ in range(80):
        mask = _sample_mask(rng, rows, cols, anchor)
        key = _mask_key(rows, cols, background, foreground, mask)
        if key not in used and _source_signature(rows, cols, background,
                                                  foreground, mask) not in _SOURCE_SIGNATURES:
            used.add(key)
            return mask
    # Bounded random search above handles ordinary cases. This enumeration
    # guarantees a distinct mask even for a crowded small-grid request.
    area = rows * cols
    for visible in range(1, area):
        for indices in combinations(range(area), visible):
            mask = np.zeros((rows, cols), dtype=bool)
            mask.flat[list(indices)] = True
            if anchor and (len(set(i // cols for i in indices)) < 3 or
                           len(set(i % cols for i in indices)) < 3):
                continue
            key = _mask_key(rows, cols, background, foreground, mask)
            if key not in used and _source_signature(rows, cols, background,
                                                      foreground, mask) not in _SOURCE_SIGNATURES:
                used.add(key)
                return mask
    raise RuntimeError("no unused clue layout remains")


def _pair_colors(rng: np.random.Generator, shared_background: int | None,
                 previous: tuple[int, int] | None) -> tuple[int, int]:
    if shared_background is not None:
        background = shared_background
    elif previous is None:
        background = int(rng.integers(10))
    else:
        candidates = [color for color in range(10) if color != previous[0]]
        background = int(rng.choice(candidates))
    previous_foreground = None if previous is None else previous[1]
    if (previous_foreground is not None and previous_foreground != background
            and rng.random() < 0.35):
        foreground = previous_foreground
    else:
        candidates = [color for color in range(10)
                      if color != background and color != previous_foreground]
        foreground = int(rng.choice(candidates))
    return background, foreground


def _make_pair(rng: np.random.Generator, rows: int, cols: int,
               background: int, foreground: int, used: set[tuple],
               anchor: bool) -> dict:
    mask = _choose_mask(rng, rows, cols, background, foreground, used, anchor)
    output = _complete_grid(rows, cols, background, foreground)
    input_grid = np.full_like(output, background)
    for tile_row in range(rows):
        row0 = tile_row * (rows + 1)
        for tile_col in range(cols):
            if mask[tile_row, tile_col]:
                col0 = tile_col * (cols + 1)
                input_grid[row0:row0 + rows, col0:col0 + cols] = \
                    output[row0:row0 + rows, col0:col0 + cols]
    return {"input": input_grid, "output": output}


def _inspect_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> dict:
    _require(input_grid.shape == output_grid.shape,
             "input/output dimensions differ")
    rows = _order_from_side(input_grid.shape[0])
    cols = _order_from_side(input_grid.shape[1])
    background = int(output_grid[rows, 0])
    output_colors = {int(value) for value in np.unique(output_grid)}
    _require(len(output_colors) == 2 and background in output_colors,
             "output color roles are invalid")
    foreground = next(color for color in output_colors if color != background)
    _require({int(value) for value in np.unique(input_grid)} ==
             {background, foreground}, "input does not reveal both colors")
    expected = _complete_grid(rows, cols, background, foreground)
    _require(np.array_equal(output_grid, expected),
             "output violates the self-indexing tile rule")
    for row in range(rows, input_grid.shape[0], rows + 1):
        _require(np.all(input_grid[row, :] == background),
                 "input separator row is damaged")
    for col in range(cols, input_grid.shape[1], cols + 1):
        _require(np.all(input_grid[:, col] == background),
                 "input separator column is damaged")
    mask = np.zeros((rows, cols), dtype=bool)
    for tile_row in range(rows):
        row0 = tile_row * (rows + 1)
        for tile_col in range(cols):
            col0 = tile_col * (cols + 1)
            actual = input_grid[row0:row0 + rows, col0:col0 + cols]
            if np.all(actual == background):
                continue
            _require(np.array_equal(actual,
                                    expected[row0:row0 + rows, col0:col0 + cols]),
                     "visible tile is partial or incorrect")
            mask[tile_row, tile_col] = True
    visible = int(mask.sum())
    _require(1 <= visible < rows * cols,
             "every pair needs a visible clue and an erased tile")
    _require(_source_signature(rows, cols, background, foreground,
                              mask) not in _SOURCE_SIGNATURES,
             "original source pair copied")
    return {"rows": rows, "cols": cols, "background": background,
            "foreground": foreground, "mask": mask, "visible": visible,
            "erased": rows * cols - visible}


def _grid_key(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(value) for value in grid.flat))


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle needs exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and
             train_count > test_count, "invalid pair counts")
    seen_inputs = set()
    seen_pairs = set()
    inspected = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair needs exactly input and output")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grids must be 2D NumPy arrays")
                _require(all(1 <= side <= 30 for side in grid.shape),
                         "grid side out of range")
                _require(np.issubdtype(grid.dtype, np.integer),
                         "grid dtype must be integer")
                _require(int(grid.min()) >= 0 and int(grid.max()) <= 9,
                         "grid color out of range")
            input_key = _grid_key(pair["input"])
            pair_key = (input_key, _grid_key(pair["output"]))
            _require(input_key not in seen_inputs and pair_key not in seen_pairs,
                     "duplicate input or pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            inspected[split].append(_inspect_pair(pair["input"], pair["output"]))
    train_rows = {row for info in inspected["train"]
                  for row in range(info["rows"]) if info["mask"][row].any()}
    train_cols = {col for info in inspected["train"]
                  for col in range(info["cols"]) if info["mask"][:, col].any()}
    _require(len(train_rows) >= 3 and len(train_cols) >= 3,
             "training does not show three coordinates on both axes")
    if any(info["rows"] != info["cols"] for info in inspected["test"]):
        _require(any(info["rows"] != info["cols"] for info in inspected["train"]),
                 "rectangular test was not demonstrated in training")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    train_shapes, test_shapes = _shape_plan(rng, train_count, test_count)
    shared_background = (int(rng.integers(10)) if rng.random() < 0.5 else None)
    used: set[tuple] = set()
    previous = None
    train = []
    for index, (rows, cols) in enumerate(train_shapes):
        background, foreground = _pair_colors(rng, shared_background, previous)
        previous = (background, foreground)
        train.append(_make_pair(rng, rows, cols, background, foreground,
                                used, index == 0))
    test = []
    for rows, cols in test_shapes:
        background, foreground = _pair_colors(rng, shared_background, previous)
        previous = (background, foreground)
        test.append(_make_pair(rng, rows, cols, background, foreground,
                               used, False))
    rng.shuffle(train)
    rng.shuffle(test)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple((split, _grid_key(pair["input"]), _grid_key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test() -> None:
    train_counts, test_counts, shapes, layouts, puzzle_keys = \
        set(), set(), set(), set(), set()
    events = {name: 0 for name in (
        "train_gap", "test_gap", "train_sparse", "test_sparse",
        "train_near_full", "test_near_full", "seven",
        "rect_train", "rect_test", "order_five", "source_shift",
        "seen_test", "smaller_test", "train_color_change",
        "repeat_foreground", "background_change", "rect_gap",
        "large_sparse")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzle_keys.add(_puzzle_key(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        infos = {split: [_inspect_pair(pair["input"], pair["output"])
                         for pair in puzzle[split]] for split in ("train", "test")}
        train_shapes = {(i["rows"], i["cols"]) for i in infos["train"]}
        test_shapes = {(i["rows"], i["cols"]) for i in infos["test"]}
        all_infos = infos["train"] + infos["test"]
        events["source_shift"] += (train_shapes == {(2, 2), (3, 3)} and
                                   test_shapes == {(4, 4)})
        events["seen_test"] += bool(test_shapes & train_shapes)
        events["smaller_test"] += any(i["rows"] * i["cols"] <
                                      max(j["rows"] * j["cols"]
                                          for j in infos["train"])
                                      for i in infos["test"])
        events["train_color_change"] += len({i["foreground"]
                                              for i in infos["train"]}) > 1
        events["repeat_foreground"] += len({i["foreground"] for i in all_infos}) < len(all_infos)
        events["background_change"] += len({i["background"] for i in all_infos}) > 1
        for split in ("train", "test"):
            for info in infos[split]:
                rows, cols = info["rows"], info["cols"]
                mask = info["mask"]
                gap = not mask.any(axis=0).all() or not mask.any(axis=1).all()
                events[f"{split}_gap"] += gap
                events[f"{split}_sparse"] += info["visible"] * 3 <= rows * cols
                events[f"{split}_near_full"] += info["erased"] <= 2
                events["seven"] += info["visible"] == 7
                events["rect_gap"] += rows != cols and gap
                events["large_sparse"] += (rows == cols == 5 and
                                           info["visible"] * 3 <= rows * cols)
                if rows != cols:
                    events[f"rect_{split}"] += 1
                events["order_five"] += rows == cols == 5
                shapes.add((_side(rows), _side(cols)))
                layouts.add((rows, cols, tuple(int(v) for v in mask.flat)))
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "automatic pair counts lack diversity")
    _require(len(puzzle_keys) >= 40 and len(shapes) >= 8 and
             len(layouts) >= 120, "cross-seed structures lack diversity")
    for name in events:
        _require(events[name] > 0, f"review feature not exercised: {name}")
    for seed, train, test in ((7100, 2, 1), (7101, 10, 1), (7102, 10, 4)):
        puzzle = generate_puzzle(seed, train, test)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train, test),
                 "explicit pair counts were not honored")
        validate_generated_puzzle(puzzle)
        _require(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed, train, test)),
                 "explicit generation is not deterministic")
    for seed, train, test in ((7200, None, 1), (7201, None, 4),
                              (7202, 2, None), (7203, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            _require(len(puzzle["train"]) == train, "partial train count failed")
        if test is not None:
            _require(len(puzzle["test"]) == test, "partial test count failed")
    _require(_puzzle_key(generate_puzzle(17)) == _puzzle_key(generate_puzzle(17)),
             "automatic generation is not deterministic")
    _require(_puzzle_key(generate_puzzle(18)) != _puzzle_key(generate_puzzle(19)),
             "different seeds produced the same puzzle")
    for arguments in ({"num_train": 1}, {"num_train": 11},
                      {"num_test": 0}, {"num_test": 5},
                      {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4},
                      {"num_train": 2.0, "num_test": 1}):
        try:
            generate_puzzle(7300, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {arguments}")
