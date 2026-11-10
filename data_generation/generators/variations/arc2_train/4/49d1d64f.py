"""One-cell edge-copy frames for arbitrary ARC color grids."""

import numpy as np


_SOURCE_PAIRS = frozenset({
    (
        ((1, 8, 4), (8, 3, 8)),
        ((0, 1, 8, 4, 0), (1, 1, 8, 4, 4),
         (8, 8, 3, 8, 8), (0, 8, 3, 8, 0)),
    ),
    (
        ((1, 2), (3, 8)),
        ((0, 1, 2, 0), (1, 1, 2, 2),
         (3, 3, 8, 8), (0, 3, 8, 0)),
    ),
    (
        ((2, 1, 4), (8, 0, 2), (3, 2, 8)),
        ((0, 2, 1, 4, 0), (2, 2, 1, 4, 4),
         (8, 8, 0, 2, 2), (3, 3, 2, 8, 8),
         (0, 3, 2, 8, 0)),
    ),
    (
        ((2, 8), (1, 4), (3, 4)),
        ((0, 2, 8, 0), (2, 2, 8, 8),
         (1, 1, 4, 4), (3, 3, 4, 4),
         (0, 3, 4, 0)),
    ),
})


def _normalize_count(name, value, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _normalize_count("num_train", num_train, 2, 10)
    num_test = _normalize_count("num_test", num_test, 1, 4)
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return num_train, num_test
    if num_train is None and num_test is None:
        legal = [(a, b) for a in range(2, 11) for b in range(1, 5) if a > b]
        return legal[int(rng.integers(len(legal)))]
    if num_train is None:
        return int(rng.integers(max(2, num_test + 1), 11)), num_test
    return num_train, int(rng.integers(1, min(4, num_train - 1) + 1))


def _grid_tuple(grid):
    return tuple(tuple(int(x) for x in row) for row in grid)


def _key(grid):
    return grid.shape, tuple(int(x) for x in grid.flat)


def _edge_frame(grid, corner):
    height, width = grid.shape
    output = np.full((height + 2, width + 2), corner, dtype=np.int8)
    output[1:-1, 1:-1] = grid
    output[0, 1:-1] = grid[0, :]
    output[-1, 1:-1] = grid[-1, :]
    output[1:-1, 0] = grid[:, 0]
    output[1:-1, -1] = grid[:, -1]
    return output


def _edge_contains(grid, color):
    return bool(np.any(grid[0] == color) or np.any(grid[-1] == color)
                or np.any(grid[:, 0] == color) or np.any(grid[:, -1] == color))


def _informative(grid, corner):
    """One example that distinguishes copying from reflection and wrapping."""
    if min(grid.shape) < 2 or np.any(grid == corner):
        return False
    a, b, c, d = (int(grid[0, 0]), int(grid[0, -1]),
                  int(grid[-1, 0]), int(grid[-1, -1]))
    return len({a, b, c, d}) == 4


def _size(rng):
    band = int(rng.choice(3, p=[0.46, 0.34, 0.20]))
    bounds = ((1, 5), (5, 13), (13, 29))[band]
    return int(rng.integers(*bounds))


def _shape(rng, orientation="any", similar_to=None):
    if similar_to is not None:
        height = int(np.clip(similar_to[0] + rng.integers(-3, 4), 1, 28))
        width = int(np.clip(similar_to[1] + rng.integers(-3, 4), 1, 28))
    else:
        mode = int(rng.integers(100))
        if mode < 5:
            height, width = 1, 1
        elif mode < 12:
            height, width = 1, _size(rng)
        elif mode < 19:
            height, width = _size(rng), 1
        elif mode < 31:
            height = width = _size(rng)
        elif mode < 49:
            height, width = int(rng.integers(1, 5)), int(rng.integers(12, 29))
            if bool(rng.integers(2)):
                height, width = width, height
        else:
            height, width = _size(rng), _size(rng)
    if orientation in ("wide", "wide_or_square") and height > width:
        height, width = width, height
    if orientation in ("tall", "tall_or_square") and height < width:
        height, width = width, height
    if orientation == "wide" and height == width:
        if width < 28:
            width += 1
        else:
            height -= 1
    if orientation == "tall" and height == width:
        if height < 28:
            height += 1
        else:
            width -= 1
    return height, width


def _palette(rng, cell_count, corner, cap=None):
    maximum = min(10, cell_count, cap if cap is not None else 10)
    roll = int(rng.integers(100))
    if roll < 13:
        count = 1
    elif roll < 37:
        count = int(rng.integers(2, min(3, maximum) + 1)) if maximum >= 2 else 1
    elif roll < 69:
        count = int(rng.integers(4, min(6, maximum) + 1)) if maximum >= 4 else maximum
    else:
        count = int(rng.integers(7, maximum + 1)) if maximum >= 7 else maximum
    if count == 1:
        color = corner if bool(rng.integers(2)) else int(rng.choice(
            [x for x in range(10) if x != corner]))
        return np.array([color], dtype=np.int8)
    available = np.arange(10, dtype=np.int8)
    if count < 10 and bool(rng.integers(2)):
        others = available[available != corner]
        palette = np.concatenate(([corner], rng.choice(others, count - 1, replace=False)))
    else:
        palette = rng.choice(available, count, replace=False)
    rng.shuffle(palette)
    return palette.astype(np.int8)


def _make_input(rng, shape, corner):
    height, width = shape
    mode = int(rng.integers(100))
    if mode < 23:  # freely arranged, with every selected color observable
        palette = _palette(rng, height * width, corner)
        labels = rng.integers(len(palette), size=shape)
        for label, position in enumerate(rng.permutation(height * width)[:len(palette)]):
            labels.flat[int(position)] = label
        return palette[labels].astype(np.int8)
    if mode < 37:  # a dominant field with a few isolated exceptions
        palette = _palette(rng, height * width, corner, cap=min(5, height * width))
        labels = np.zeros(shape, dtype=np.int64)
        if len(palette) > 1:
            positions = rng.permutation(height * width)
            for label, position in enumerate(positions[:len(palette) - 1], start=1):
                labels.flat[int(position)] = label
            extra = int(rng.integers(0, max(1, height * width // 12) + 1))
            for position in positions[len(palette) - 1:len(palette) - 1 + extra]:
                labels.flat[int(position)] = int(rng.integers(1, len(palette)))
        return palette[labels].astype(np.int8)
    if mode < 52:  # full rows are assigned together
        palette = _palette(rng, height * width, corner, cap=height)
        labels = rng.integers(len(palette), size=height)
        for label, row in enumerate(rng.permutation(height)[:len(palette)]):
            labels[int(row)] = label
        return np.broadcast_to(palette[labels, None], shape).copy()
    if mode < 67:  # full columns are assigned together
        palette = _palette(rng, height * width, corner, cap=width)
        labels = rng.integers(len(palette), size=width)
        for label, col in enumerate(rng.permutation(width)[:len(palette)]):
            labels[int(col)] = label
        return np.broadcast_to(palette[labels][None, :], shape).copy()
    if mode < 77:  # checkerboards and diagonal repeats
        palette = _palette(rng, height * width, corner, cap=min(3, height * width))
        rows, cols = np.indices(shape)
        labels = (rows + cols + int(rng.integers(len(palette)))) % len(palette)
        return palette[labels].astype(np.int8)
    if mode < 87 and min(shape) >= 4:  # repeated rectangular patches
        block_h = int(rng.integers(2, min(6, height) + 1))
        block_w = int(rng.integers(2, min(6, width) + 1))
        blocks_h = (height + block_h - 1) // block_h
        blocks_w = (width + block_w - 1) // block_w
        palette = _palette(rng, blocks_h * blocks_w, corner)
        labels = rng.integers(len(palette), size=(blocks_h, blocks_w))
        for label, position in enumerate(rng.permutation(blocks_h * blocks_w)[:len(palette)]):
            labels.flat[int(position)] = label
        return np.repeat(np.repeat(palette[labels], block_h, axis=0),
                         block_w, axis=1)[:height, :width].copy()
    if mode < 96 and min(shape) >= 3:  # uniform rim, contrasting interior
        rim = int(rng.integers(10))
        inner_palette = np.arange(10, dtype=np.int8)
        inner_palette = inner_palette[inner_palette != rim]
        count = int(rng.integers(1, min(9, (height - 2) * (width - 2)) + 1))
        colors = rng.choice(inner_palette, count, replace=False)
        grid = np.full(shape, rim, dtype=np.int8)
        labels = rng.integers(count, size=(height - 2, width - 2))
        grid[1:-1, 1:-1] = colors[labels]
        return grid
    # The edge and interior use separate draws; the latter never changes the frame.
    palette = _palette(rng, height * width, corner)
    grid = rng.choice(palette, size=shape).astype(np.int8)
    if min(shape) >= 3:
        interior_palette = _palette(rng, (height - 2) * (width - 2), corner)
        grid[1:-1, 1:-1] = rng.choice(
            interior_palette, size=(height - 2, width - 2))
    return grid


def _teaching_input(rng, shape, corner, role):
    height, width = max(2, shape[0]), max(2, shape[1])
    shape = (height, width)
    available = np.array([c for c in range(10) if c != corner], dtype=np.int8)
    if role == "informative":
        palette = rng.choice(available, size=int(rng.integers(4, 9)), replace=False)
        grid = rng.choice(palette, size=shape).astype(np.int8)
        corners = rng.choice(palette, size=4, replace=False)
        grid[0, 0], grid[0, -1], grid[-1, 0], grid[-1, -1] = corners
        return grid
    palette = np.concatenate(([corner], rng.choice(
        available, size=int(rng.integers(1, 6)), replace=False)))
    grid = rng.choice(palette, size=shape).astype(np.int8)
    edge = [(0, c) for c in range(width)] + [(height - 1, c) for c in range(width)]
    edge += [(r, 0) for r in range(1, height - 1)]
    edge += [(r, width - 1) for r in range(1, height - 1)]
    r, c = edge[int(rng.integers(len(edge)))]
    grid[r, c] = corner
    r, c = edge[int(rng.integers(len(edge)))]
    if not np.any(grid != corner):
        grid[r, c] = int(rng.choice(available))
    return grid


def _neutral_center(rng, corner):
    other = int(rng.choice([c for c in range(10) if c != corner]))
    grid = np.full((3, 3), other, dtype=np.int8)
    grid[1, 1] = corner
    return grid


def _twin(rng, grid, same_edges):
    twin = grid.copy()
    if same_edges:
        r = int(rng.integers(1, grid.shape[0] - 1))
        c = int(rng.integers(1, grid.shape[1] - 1))
    else:
        edge = [(0, c) for c in range(grid.shape[1])]
        edge += [(grid.shape[0] - 1, c) for c in range(grid.shape[1])]
        edge += [(r, 0) for r in range(1, grid.shape[0] - 1)]
        edge += [(r, grid.shape[1] - 1) for r in range(1, grid.shape[0] - 1)]
        r, c = edge[int(rng.integers(len(edge)))]
    choices = [color for color in range(10) if color != int(twin[r, c])]
    twin[r, c] = int(rng.choice(choices))
    return twin


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    # Stratification covers every legal corner color in any ten consecutive
    # seeds, while the color remains visible only through the training pairs.
    corner = int(seed % 10)
    regime = int(rng.integers(4))  # source shift, reverse, mixed, similar
    base = _shape(rng) if regime == 3 else None
    if regime == 0:
        train_orientation, test_orientation = "wide_or_square", "tall"
    elif regime == 1:
        train_orientation, test_orientation = "tall_or_square", "wide"
    else:
        train_orientation = test_orientation = "any"

    roles = [None] * num_train
    role_indices = rng.choice(num_train, 2, replace=False)
    roles[int(role_indices[0])] = "informative"
    roles[int(role_indices[1])] = "neutral_edge"
    center_index = None
    eligible = [i for i in range(num_train) if roles[i] is None]
    if eligible and rng.random() < 0.22:
        center_index = int(rng.choice(eligible))

    puzzle = {"train": [], "test": []}
    used = set()
    for split, count, orientation in (
        ("train", num_train, train_orientation),
        ("test", num_test, test_orientation),
    ):
        for index in range(count):
            role = roles[index] if split == "train" else None
            for attempt in range(128):
                shape = _shape(rng, orientation, base)
                if split == "train" and index == center_index:
                    shape = (3, 3)
                elif role is not None:
                    shape = (max(2, shape[0]), max(2, shape[1]))
                if split == "train" and index == 1:
                    previous = puzzle["train"][0]["input"].shape
                    if shape == previous:
                        continue
                if index == center_index and split == "train":
                    grid = _neutral_center(rng, corner)
                elif role is not None:
                    grid = _teaching_input(rng, shape, corner, role)
                else:
                    grid = _make_input(rng, shape, corner)
                key = _key(grid)
                if key in used or _grid_tuple(grid) in {p[0] for p in _SOURCE_PAIRS}:
                    continue
                used.add(key)
                puzzle[split].append({"input": grid, "output": _edge_frame(grid, corner)})
                break
            else:
                raise RuntimeError("could not construct a distinct input")

    # Occasionally show that the interior and the copied edge are independent.
    candidates = [i for i, role in enumerate(roles) if role is None
                  and min(puzzle["train"][i]["input"].shape) >= 3]
    if len(candidates) >= 2 and rng.random() < 0.65:
        first, second = rng.choice(candidates, size=2, replace=False)
        first, second = int(first), int(second)
        source = puzzle["train"][first]["input"]
        twin = _twin(rng, source, same_edges=bool(rng.integers(2)))
        old = puzzle["train"][second]["input"]
        remaining = {_key(p["input"]) for s in ("train", "test") for p in puzzle[s]}
        remaining.remove(_key(old))
        if (_key(twin) not in remaining
                and _grid_tuple(twin) not in {p[0] for p in _SOURCE_PAIRS}
                and len({p["input"].shape for i, p in enumerate(puzzle["train"])
                         if i != second} | {twin.shape}) >= 2):
            puzzle["train"][second] = {
                "input": twin, "output": _edge_frame(twin, corner)}

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    corner = None
    seen_inputs, seen_pairs = set(), set()
    train_shapes = set()
    informative = neutral_edge = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = pair["input"], pair["output"]
            for value in (grid, output):
                assert isinstance(value, np.ndarray) and value.ndim == 2
                assert all(1 <= n <= 30 for n in value.shape)
                assert np.issubdtype(value.dtype, np.integer)
                assert bool(np.all((0 <= value) & (value <= 9)))
            assert all(n <= 28 for n in grid.shape)
            assert output.shape == (grid.shape[0] + 2, grid.shape[1] + 2)
            corners = (int(output[0, 0]), int(output[0, -1]),
                       int(output[-1, 0]), int(output[-1, -1]))
            assert len(set(corners)) == 1
            if corner is None:
                corner = corners[0]
            assert corners[0] == corner
            assert np.array_equal(output[1:-1, 1:-1], grid)
            assert np.array_equal(output[0, 1:-1], grid[0])
            assert np.array_equal(output[-1, 1:-1], grid[-1])
            assert np.array_equal(output[1:-1, 0], grid[:, 0])
            assert np.array_equal(output[1:-1, -1], grid[:, -1])
            pair_key = (_grid_tuple(grid), _grid_tuple(output))
            assert pair_key not in _SOURCE_PAIRS
            assert _key(grid) not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(_key(grid))
            seen_pairs.add(pair_key)
            if split == "train":
                train_shapes.add(grid.shape)
                informative |= _informative(grid, corner)
                neutral_edge |= _edge_contains(grid, corner)
    assert len(train_shapes) >= 2
    assert informative and neutral_edge


def _signature(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts = {len(p["train"]) for p in puzzles}
    test_counts = {len(p["test"]) for p in puzzles}
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len({_signature(p) for p in puzzles}) >= 40
    corner_colors, shapes, palettes, modes = set(), set(), set(), set()
    small = large = single = interior_test = neutral_mono = other_mono = 0
    rim_count = center_count = rich_test = rows_count = cols_count = checker_count = 0
    patch_count = same_edge_twins = same_interior_twins = 0
    larger_test = smaller_test = similar_test = 0
    informative_positions = set()
    aspect_categories = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        corner = int(puzzle["train"][0]["output"][0, 0])
        corner_colors.add(corner)
        train = puzzle["train"]
        tests = puzzle["test"]
        train_shapes = {p["input"].shape for p in train}
        assert len(train_shapes) >= 2
        assert len({len(np.unique(p["input"])) for p in train + tests}) >= 2
        assert any(_informative(p["input"], corner) for p in train)
        assert any(_edge_contains(p["input"], corner) for p in train)
        informative_positions.update(
            i for i, p in enumerate(train) if _informative(p["input"], corner))
        train_areas = [p["input"].size for p in train]
        test_areas = [p["input"].size for p in tests]
        larger_test += max(test_areas) > max(train_areas)
        smaller_test += min(test_areas) < min(train_areas)
        similar_test += any(.5 <= a / b <= 2 for a in test_areas for b in train_areas)
        if all(g["input"].shape[0] <= g["input"].shape[1] for g in train) and all(
                g["input"].shape[0] > g["input"].shape[1] for g in tests):
            modes.add("source")
        if all(g["input"].shape[0] >= g["input"].shape[1] for g in train) and all(
                g["input"].shape[0] < g["input"].shape[1] for g in tests):
            modes.add("reverse")
        aspect_categories.add(tuple(sorted({
            "square" if p["input"].shape[0] == p["input"].shape[1]
            else "tall" if p["input"].shape[0] > p["input"].shape[1]
            else "wide" for p in train + tests})))
        for i, first in enumerate(train):
            a = first["input"]
            if min(a.shape) < 3:
                continue
            for second in train[i + 1:]:
                b = second["input"]
                if a.shape != b.shape:
                    continue
                same_edges = (np.array_equal(a[0], b[0])
                              and np.array_equal(a[-1], b[-1])
                              and np.array_equal(a[:, 0], b[:, 0])
                              and np.array_equal(a[:, -1], b[:, -1]))
                same_interior = np.array_equal(a[1:-1, 1:-1], b[1:-1, 1:-1])
                same_edge_twins += same_edges and not same_interior
                same_interior_twins += same_interior and not same_edges
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, output = pair["input"], pair["output"]
                h, w = grid.shape
                shapes.add(grid.shape)
                palette = len(np.unique(grid))
                palettes.add(palette)
                small += max(h, w) <= 4
                large += max(h, w) >= 15
                single += min(h, w) == 1
                neutral_mono += palette == 1 and int(grid[0, 0]) == corner
                other_mono += palette == 1 and int(grid[0, 0]) != corner
                if split == "test":
                    interior_test += min(h, w) >= 3
                    rich_test += min(h, w) >= 6 and palette >= 7
                rows_count += h >= 2 and palette >= 2 and all(
                    len(np.unique(row)) == 1 for row in grid)
                cols_count += w >= 2 and palette >= 2 and all(
                    len(np.unique(col)) == 1 for col in grid.T)
                if min(h, w) >= 2 and palette == 2:
                    rows, cols = np.indices(grid.shape)
                    parity = (rows + cols) % 2
                    checker_count += (len(np.unique(grid[parity == 0])) == 1
                                      and len(np.unique(grid[parity == 1])) == 1)
                if min(h, w) >= 3:
                    rim = np.concatenate((grid[0], grid[-1],
                                          grid[1:-1, 0], grid[1:-1, -1]))
                    rim_count += (len(np.unique(rim)) == 1
                                  and bool(np.any(grid[1:-1, 1:-1] != rim[0])))
                    center_count += (grid.shape == (3, 3)
                                     and int(grid[1, 1]) == corner
                                     and np.count_nonzero(grid == corner) == 1)
                if min(h, w) >= 6:
                    blocks = [int(grid[r, c]) for r in range(1, h - 2)
                              for c in range(1, w - 2)
                              if len(np.unique(grid[r:r + 2, c:c + 2])) == 1]
                    patch_count += len(blocks) >= 3 and len(set(blocks)) >= 2
                # Coordinate reconstruction uses only the observed train corner
                # and the input; it does not read the test output to predict it.
                predicted = np.full((h + 2, w + 2), corner, dtype=np.int8)
                for r in range(h + 2):
                    for c in range(w + 2):
                        if (r in (0, h + 1)) and (c in (0, w + 1)):
                            continue
                        predicted[r, c] = grid[min(max(r - 1, 0), h - 1),
                                               min(max(c - 1, 0), w - 1)]
                assert np.array_equal(predicted, output)
    assert len(corner_colors) == 10
    assert len(shapes) >= 35 and len(palettes) >= 8
    assert small >= 15 and large >= 15 and single >= 5
    assert interior_test >= 15
    assert neutral_mono >= 2 and other_mono >= 2
    assert {"source", "reverse"} <= modes
    assert len(aspect_categories) >= 4
    assert len(informative_positions) >= 4
    assert rim_count >= 5 and center_count >= 2
    assert rich_test >= 2 and rows_count >= 5 and cols_count >= 5
    assert checker_count >= 1 and patch_count >= 2
    assert same_edge_twins >= 1 and same_interior_twins >= 1
    assert larger_test >= 2 and smaller_test >= 2 and similar_test >= 2
    for a, b in ((2, 1), (10, 1), (10, 4), (7, 3)):
        p = generate_puzzle(1000 + a * 10 + b, a, b)
        assert (len(p["train"]), len(p["test"])) == (a, b)
        assert _signature(p) == _signature(generate_puzzle(1000 + a * 10 + b, a, b))
    for a, b in ((None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(2000 + (a or 0) * 10 + (b or 0), a, b)
        assert (a is None or len(p["train"]) == a)
        assert (b is None or len(p["test"]) == b)
    for a, b in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(50, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts were accepted")
