import numpy as np


_MOTIFS = (
    (1, 1, 1, 1),  # Filled 2x2 square
    (0, 1, 1, 1),  # Three-cell L shapes
    (1, 0, 1, 1),
    (1, 1, 0, 1),
    (1, 1, 1, 0),
)

_SOURCE_INPUT_KEYS = {
    (
        14,
        16,
        "".join(
            (
                "0000000000000000",
                "0200002000020000",
                "0011000000000110",
                "0011000000000100",
                "0000000100000000",
                "0000000110000000",
                "0200002000020000",
                "0000000000000000",
                "0000000011000100",
                "0000100010000110",
                "0001100000000000",
                "0200002000020000",
                "0000000000000000",
                "0000000000000000",
            )
        ),
    ),
    (
        13,
        19,
        "".join(
            (
                "0000000000000000000",
                "0000000000000000000",
                "0020000200002000020",
                "0001000000000000100",
                "0001100001100001100",
                "0000000000100000000",
                "0000000000000000000",
                "0020000200002000020",
                "0000000001100000000",
                "0000000001100110000",
                "0001000000000100000",
                "0001100000000000000",
                "0020000200002000020",
            )
        ),
    ),
    (
        17,
        10,
        "".join(
            (
                "0000000000",
                "2000020000",
                "0000000100",
                "0110000110",
                "0110000000",
                "0000000000",
                "2000020000",
                "0000000000",
                "0000000100",
                "0110000110",
                "0100000000",
                "2000020000",
                "0000100000",
                "0001100110",
                "0000000100",
                "0000000000",
                "2000020000",
            )
        ),
    ),
    (
        17,
        13,
        "".join(
            (
                "0000000000000",
                "0020000200002",
                "0000000000000",
                "0000000000000",
                "0001100001100",
                "0001100001000",
                "0020000200002",
                "0000000000010",
                "0000000000110",
                "0000110000000",
                "0000110000000",
                "0020000200002",
                "0000000001100",
                "0000110001000",
                "0000010000000",
                "0000000000000",
                "0020000200002",
            )
        ),
    ),
}


def _array_key(array: np.ndarray) -> tuple:
    compact = array.astype(np.int8, copy=False)
    return tuple(int(x) for x in array.shape), compact.tobytes()


def _source_input_key(array: np.ndarray) -> tuple:
    text = "".join(str(int(x)) for x in array.ravel())
    return int(array.shape[0]), int(array.shape[1]), text


def _connected(mask: np.ndarray) -> bool:
    points = {
        tuple(int(x) for x in point)
        for point in np.argwhere(mask)
    }
    if not points:
        return False

    seen = set()
    stack = [next(iter(points))]

    while stack:
        point = stack.pop()
        if point in seen:
            continue

        seen.add(point)
        row, col = point

        for neighbor in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if neighbor in points and neighbor not in seen:
                stack.append(neighbor)

    return seen == points


_CELL_SIZES = (4, 6, 8)


def _analyze_input(grid: np.ndarray) -> dict:
    height, width = grid.shape
    colors, counts = np.unique(grid, return_counts=True)
    order = np.argsort(counts)
    assert len(colors) >= 3
    assert counts[order[-1]] > counts[order[-2]]
    background = int(colors[order[-1]])

    guide_candidates = []
    for value in colors:
        color = int(value)
        if color == background:
            continue
        points = np.argwhere(grid == color)
        rows = tuple(int(x) for x in np.unique(points[:, 0]))
        cols = tuple(int(x) for x in np.unique(points[:, 1]))
        if len(rows) < 2 or len(cols) < 2:
            continue
        row_steps = {b - a for a, b in zip(rows, rows[1:])}
        col_steps = {b - a for a, b in zip(cols, cols[1:])}
        if len(row_steps) != 1 or len(col_steps) != 1:
            continue
        cell_height = next(iter(row_steps)) - 1
        cell_width = next(iter(col_steps)) - 1
        if cell_height not in _CELL_SIZES or cell_width not in _CELL_SIZES:
            continue
        if len(points) != len(rows) * len(cols):
            continue
        if not all(grid[r, c] == color for r in rows for c in cols):
            continue
        guide_candidates.append((color, rows, cols, cell_height, cell_width))

    assert len(guide_candidates) == 1
    guide, rows, cols, cell_height, cell_width = guide_candidates[0]
    active_rows = tuple(r for r in rows if r + cell_height < height)
    active_cols = tuple(c for c in cols if c + cell_width < width)
    assert active_rows and active_cols
    assert active_rows == rows[:len(active_rows)]
    assert active_cols == cols[:len(active_cols)]
    assert len(rows) - len(active_rows) in (0, 1)
    assert len(cols) - len(active_cols) in (0, 1)
    assert len(rows) >= 2 and len(cols) >= 2

    covered = np.zeros(grid.shape, dtype=bool)
    predicted = grid.copy()
    occupied = empty = stationary = moving = 0
    shapes = set()
    offsets = set()
    payload_colors = set()
    for row in active_rows:
        for col in active_cols:
            top, left = row + 1, col + 1
            block = grid[top:top + cell_height, left:left + cell_width]
            assert block.shape == (cell_height, cell_width)
            assert not np.any(block == guide)
            covered[top:top + cell_height, left:left + cell_width] = True
            values = {int(x) for x in np.unique(block)} - {background}
            assert len(values) <= 1
            if not values:
                empty += 1
                continue
            color = next(iter(values))
            mask = block == color
            assert _connected(mask)
            points = np.argwhere(mask)
            low = points.min(axis=0)
            high = points.max(axis=0)
            local_row, local_col = (int(x) for x in low)
            shape_height, shape_width = (int(x) for x in high - low + 1)
            assert shape_height >= 2 and shape_width >= 2
            assert shape_height <= cell_height - 2
            assert shape_width <= cell_width - 2
            assert shape_height % 2 == 0 and shape_width % 2 == 0
            center_row = (cell_height - shape_height) // 2
            center_col = (cell_width - shape_width) // 2
            bitmap = mask[local_row:local_row + shape_height,
                          local_col:local_col + shape_width]
            predicted[top:top + cell_height, left:left + cell_width][mask] = background
            target = predicted[top + center_row:top + center_row + shape_height,
                               left + center_col:left + center_col + shape_width]
            target[bitmap] = color
            occupied += 1
            payload_colors.add(color)
            shapes.add((shape_height, shape_width, tuple(int(x) for x in bitmap.ravel())))
            offsets.add((local_row, local_col))
            if (local_row, local_col) == (center_row, center_col):
                stationary += 1
            else:
                moving += 1

    assert occupied >= 1
    assert not np.any((grid != background) & (grid != guide) & ~covered)
    return {
        "background": background, "guide": guide,
        "cell_size": (cell_height, cell_width),
        "layout": (len(active_rows), len(active_cols)),
        "guide_rows": rows, "guide_cols": cols,
        "terminal_row": len(rows) > len(active_rows),
        "terminal_col": len(cols) > len(active_cols),
        "occupied": occupied, "empty": empty,
        "stationary": stationary, "moving": moving,
        "payload_colors": frozenset(payload_colors),
        "shapes": frozenset(shapes), "offsets": frozenset(offsets),
        "shape": grid.shape, "predicted": predicted,
    }


def _axis(rng, cell_size: int, count: int, terminal: bool, origin: int):
    step = cell_size + 1
    coordinates = tuple(origin + index * step
                        for index in range(count + int(terminal)))
    minimum = (coordinates[-1] + 1 if terminal
               else origin + count * step)
    assert minimum <= 30
    margin = int(rng.integers(0, min(3, 30 - minimum) + 1))
    return coordinates, minimum + margin


def _motif(rng, shape_height: int, shape_width: int) -> np.ndarray:
    if shape_height == shape_width == 2:
        return np.array(_MOTIFS[int(rng.integers(0, 5))], dtype=bool).reshape(2, 2)
    bitmap = np.zeros((shape_height, shape_width), dtype=bool)
    if rng.random() < 0.35:
        bitmap[:] = True
        return bitmap
    row = col = 0
    bitmap[row, col] = True
    steps = [0] * (shape_height - 1) + [1] * (shape_width - 1)
    for step in rng.permutation(steps):
        if step == 0:
            row += 1
        else:
            col += 1
        bitmap[row, col] = True
    for _ in range(int(rng.integers(0, shape_height * shape_width // 2 + 1))):
        occupied = np.argwhere(bitmap)
        r, c = (int(x) for x in occupied[int(rng.integers(0, len(occupied)))])
        neighbors = [(r + dr, c + dc) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                     if 0 <= r + dr < shape_height and 0 <= c + dc < shape_width]
        rr, cc = neighbors[int(rng.integers(0, len(neighbors)))]
        bitmap[rr, cc] = True
    if rng.integers(0, 2):
        bitmap = bitmap[::-1, :]
    if rng.integers(0, 2):
        bitmap = bitmap[:, ::-1]
    return bitmap


def _feasible_layouts(cell_height: int, cell_width: int, minimum: int = 1):
    return [(r, c) for r in range(1, 6) for c in range(1, 6)
            if r * c >= minimum and r * c <= 16
            and r * (cell_height + 1) + 1 <= 30
            and c * (cell_width + 1) + 1 <= 30]


def _make_pair(rng, cell_size, layout, previous_colors=(),
               mode="random", force_multicolor=False):
    cell_height, cell_width = cell_size
    row_count, col_count = layout
    row_terminal = bool(rng.integers(0, 2)) or row_count == 1
    col_terminal = bool(rng.integers(0, 2)) or col_count == 1
    max_row_origin = 30 - ((row_count + int(row_terminal) - 1)
                           * (cell_height + 1) + (1 if row_terminal else cell_height + 1))
    max_col_origin = 30 - ((col_count + int(col_terminal) - 1)
                           * (cell_width + 1) + (1 if col_terminal else cell_width + 1))
    row_origin = int(rng.integers(0, min(4, max_row_origin) + 1))
    col_origin = int(rng.integers(0, min(4, max_col_origin) + 1))
    guide_rows, height = _axis(rng, cell_height, row_count, row_terminal, row_origin)
    guide_cols, width = _axis(rng, cell_width, col_count, col_terminal, col_origin)

    available = [c for c in range(10) if c not in previous_colors]
    if len(available) < 4:
        available = list(range(10))
    selected = [int(x) for x in rng.permutation(available)]
    background, guide = selected[:2]
    palette = selected[2:5]
    grid = np.full((height, width), background, dtype=np.int8)
    for row in guide_rows:
        for col in guide_cols:
            grid[row, col] = guide
    cells = [(r, c) for r in guide_rows[:row_count]
             for c in guide_cols[:col_count]]
    order = [cells[int(i)] for i in rng.permutation(len(cells))]
    occupied = len(cells)
    if mode in ("simple", "scaled"):
        occupied -= 1
    else:
        occupied -= int(rng.integers(0, min(len(cells) - 1, max(1, len(cells) // 3)) + 1))
    occupied = max(1, occupied)
    stationary_count = min(2 if len(cells) >= 5 and rng.random() < 0.4 else 1,
                           occupied - 1) if mode in ("simple", "scaled") else 0
    if mode == "random" and rng.random() < 0.55:
        stationary_count = int(rng.integers(1, occupied + 1))
    if mode == "all_centered":
        stationary_count = occupied
    multicolor = force_multicolor or (mode == "random" and occupied >= 2 and rng.random() < 0.55)
    output = grid.copy()
    for index, (row, col) in enumerate(order[:occupied]):
        if mode == "simple":
            shape_height = shape_width = 2
        elif mode == "scaled" and index == 0:
            shape_height = 4 if cell_height >= 6 else 2
            shape_width = 4 if cell_width >= 6 else 2
        else:
            shape_height = int(rng.choice(np.arange(2, cell_height, 2)))
            shape_width = int(rng.choice(np.arange(2, cell_width, 2)))
        bitmap = _motif(rng, shape_height, shape_width)
        color = palette[index % min(3, occupied)] if multicolor else palette[0]
        if index < stationary_count:
            local_row = (cell_height - shape_height) // 2
            local_col = (cell_width - shape_width) // 2
        else:
            offsets = [(r, c) for r in range(cell_height - shape_height + 1)
                       for c in range(cell_width - shape_width + 1)
                       if (r, c) != ((cell_height - shape_height) // 2,
                                     (cell_width - shape_width) // 2)]
            local_row, local_col = offsets[int(rng.integers(0, len(offsets)))]
        target_row = (cell_height - shape_height) // 2
        target_col = (cell_width - shape_width) // 2
        input_region = grid[row + 1 + local_row:row + 1 + local_row + shape_height,
                            col + 1 + local_col:col + 1 + local_col + shape_width]
        target_region = output[row + 1 + target_row:row + 1 + target_row + shape_height,
                               col + 1 + target_col:col + 1 + target_col + shape_width]
        input_region[bitmap] = color
        target_region[bitmap] = color
    return grid, output


def _validate_optional_count(name: str, value: int | None, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return result


def _resolve_counts(seed, rng, num_train, num_test):
    train_count = _validate_optional_count("num_train", num_train, 2, 10)
    test_count = _validate_optional_count("num_test", num_test, 1, 4)
    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        low = test_count + 1
        train_count = low + (seed // 4) % (11 - low)
    elif train_count is None:
        train_count = int(rng.integers(test_count + 1, 11))
    elif test_count is None:
        test_count = int(rng.integers(1, min(4, train_count - 1) + 1))
    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")
    return train_count, test_count


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    puzzle = {"train": [], "test": []}
    seen = set()
    first_colors = ()
    first_layout = None
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            if split == "train" and index == 0:
                cell_size = (4, 4)
                layouts = [(2, 2), (2, 3), (3, 2), (1, 3), (3, 1), (1, 4), (4, 1)]
                layout = layouts[int(rng.integers(0, len(layouts)))]
                first_layout = layout
                mode = "simple"
            elif split == "train" and index == 1:
                cell_size = [(6, 6), (6, 8), (8, 6), (8, 8)][int(rng.integers(0, 4))]
                layouts = [(2, 2), (1, 3), (3, 1)]
                layouts = [x for x in layouts if x[0] * x[1] != first_layout[0] * first_layout[1]]
                layout = layouts[int(rng.integers(0, len(layouts)))]
                mode = "scaled"
            else:
                cell_size = (_CELL_SIZES[int(rng.integers(0, 3))],
                             _CELL_SIZES[int(rng.integers(0, 3))])
                layouts = _feasible_layouts(*cell_size)
                layout = layouts[int(rng.integers(0, len(layouts)))]
                mode = "all_centered" if split == "train" and index >= 2 and rng.random() < 0.12 else "random"
            for attempt in range(30):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                pair = _make_pair(pair_rng, cell_size, layout,
                                  previous_colors=first_colors if split == "train" and index == 1 else (),
                                  mode=mode, force_multicolor=split == "train" and index == 1)
                key = _array_key(pair[0])
                if key in seen or _source_input_key(pair[0]) in _SOURCE_INPUT_KEYS:
                    continue
                try:
                    _analyze_input(pair[0])
                except AssertionError:
                    continue
                seen.add(key)
                puzzle[split].append({"input": pair[0], "output": pair[1]})
                if split == "train" and index == 0:
                    analysis = _analyze_input(pair[0])
                    first_colors = (analysis["background"], analysis["guide"])
                break
            else:
                raise ValueError("could not construct a unique valid pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
        assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
        assert 2 <= len(puzzle["train"]) <= 10
        assert 1 <= len(puzzle["test"]) <= 4
        assert len(puzzle["train"]) > len(puzzle["test"])
        seen_inputs = set()
        seen_pairs = set()
        analyses = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert isinstance(pair, dict) and set(pair) == {"input", "output"}
                for grid in (pair["input"], pair["output"]):
                    assert isinstance(grid, np.ndarray) and grid.ndim == 2
                    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                    assert np.issubdtype(grid.dtype, np.integer)
                    assert int(grid.min()) >= 0 and int(grid.max()) <= 9
                inp, out = pair["input"], pair["output"]
                input_key = _array_key(inp)
                pair_key = (input_key, _array_key(out))
                assert input_key not in seen_inputs and pair_key not in seen_pairs
                assert _source_input_key(inp) not in _SOURCE_INPUT_KEYS
                seen_inputs.add(input_key)
                seen_pairs.add(pair_key)
                analysis = _analyze_input(inp)
                assert np.array_equal(analysis["predicted"], out)
                analyses[split].append(analysis)
        train = analyses["train"]
        assert sum(a["moving"] for a in train) >= 2
        assert sum(a["stationary"] for a in train) >= 1
        assert sum(a["empty"] for a in train) >= 1
        assert len({a["background"] for a in train}) >= 2
        assert len({a["guide"] for a in train}) >= 2
        assert any(len(a["payload_colors"]) >= 2 for a in train)
        assert (4, 4) in {a["cell_size"] for a in train}
        assert any(max(a["cell_size"]) >= 6 for a in train)
        assert len({a["layout"][0] * a["layout"][1] for a in train}) >= 2
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError("malformed or invalid generated puzzle") from exc


def _puzzles_equal(first, second):
    return all(len(first[s]) == len(second[s]) and
               all(np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s])
                   for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    cell_sizes, layouts, dimensions = set(), set(), set()
    terminal_pairs, backgrounds, guides, shapes = set(), set(), set(), set()
    test_terminals, test_cell_sizes, test_origins = set(), set(), set()
    bbox_sizes = set()
    seen = set()
    empty = stationary = moving = multicolor = centered_pairs = 0
    mixed_stationary_moving = 0
    single = strip = square = large = 0
    palette_varying = count_varying = 0
    same_layout = count_extrapolation = new_shape = 0
    new_row_origin = new_col_origin = payload_varying = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _puzzles_equal(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        analyses = [_analyze_input(p["input"]) for s in ("train", "test") for p in puzzle[s]]
        train_analyses = analyses[:len(puzzle["train"])]
        test_analyses = analyses[len(puzzle["train"]):]
        palette_varying += len({a["background"] for a in train_analyses}) > 1 and len({a["guide"] for a in train_analyses}) > 1
        count_varying += len({a["layout"][0] * a["layout"][1] for a in analyses}) > 1
        payload_varying += len(set().union(*(a["payload_colors"] for a in train_analyses))) > 1
        same_layout += any(a["layout"] == b["layout"] for a in train_analyses for b in test_analyses)
        count_extrapolation += any(b["occupied"] > max(a["occupied"] for a in train_analyses)
                                   for b in test_analyses)
        new_row_origin += any(b["guide_rows"][0] not in {a["guide_rows"][0] for a in train_analyses}
                              for b in test_analyses)
        new_col_origin += any(b["guide_cols"][0] not in {a["guide_cols"][0] for a in train_analyses}
                              for b in test_analyses)
        train_shapes = set().union(*(a["shapes"] for a in train_analyses))
        new_shape += any(set(b["shapes"]) - train_shapes for b in test_analyses)
        test_terminals.update((b["terminal_row"], b["terminal_col"]) for b in test_analyses)
        test_cell_sizes.update(b["cell_size"] for b in test_analyses)
        test_origins.update((b["guide_rows"][0], b["guide_cols"][0]) for b in test_analyses)
        for analysis in analyses:
            cell_sizes.add(analysis["cell_size"])
            layouts.add(analysis["layout"])
            dimensions.add(analysis["shape"])
            terminal_pairs.add((analysis["terminal_row"], analysis["terminal_col"]))
            backgrounds.add(analysis["background"])
            guides.add(analysis["guide"])
            shapes.update(analysis["shapes"])
            bbox_sizes.update((h, w) for h, w, _ in analysis["shapes"])
            empty += analysis["empty"]
            stationary += analysis["stationary"]
            moving += analysis["moving"]
            multicolor += len(analysis["payload_colors"]) >= 2
            centered_pairs += analysis["moving"] == 0
            mixed_stationary_moving += analysis["stationary"] >= 2 and analysis["moving"] > 0
            r, c = analysis["layout"]
            single += (r, c) == (1, 1)
            strip += (r == 1) != (c == 1)
            square += r == c and r > 1
            large += r * c >= 8
            seen.add(_array_key(analysis["predicted"]))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(cell_sizes) == 9 and len(layouts) >= 12 and len(dimensions) >= 30
    assert len(terminal_pairs) == 4 and len(backgrounds) == 10 and len(guides) == 10
    assert len(test_terminals) == 4 and len(test_cell_sizes) == 9 and len(test_origins) >= 10
    assert len(bbox_sizes) == 9 and len(shapes) >= 20 and len(seen) >= 150
    assert empty > 30 and stationary > 30 and moving > 100 and multicolor > 20
    assert mixed_stationary_moving > 0 and centered_pairs > 0
    assert single > 0 and strip > 0 and square > 0 and large > 0
    assert palette_varying == 42 and count_varying == 42 and payload_varying == 42
    assert same_layout >= 5 and count_extrapolation >= 5 and new_shape >= 5
    assert new_row_origin >= 5 and new_col_origin >= 5
    for i, (ntrain, ntest) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(800 + i, ntrain, ntest)
        assert len(puzzle["train"]) == ntrain and len(puzzle["test"]) == ntest
        validate_generated_puzzle(puzzle)
    for i, (ntrain, ntest) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(900 + i, ntrain, ntest)
        assert ntrain is None or len(puzzle["train"]) == ntrain
        assert ntest is None or len(puzzle["test"]) == ntest
        validate_generated_puzzle(puzzle)
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(1000, **args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    assert not _puzzles_equal(generate_puzzle(702, 5, 2), generate_puzzle(703, 5, 2))
