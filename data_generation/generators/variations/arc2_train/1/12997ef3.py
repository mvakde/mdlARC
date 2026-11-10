import numpy as np


_SOURCE_PAIR_KEYS = frozenset(
    {
        (
            (
                "00000000000",
                "00000006000",
                "00010000000",
                "00110002000",
                "00001000000",
                "00000000000",
                "00000000000",
                "00000000000",
                "00000000000",
                "00000000000",
                "00000000000",
            ),
            ("060", "660", "006", "020", "220", "002"),
        ),
        (
            (
                "000000000",
                "000003000",
                "000000000",
                "000006000",
                "011000000",
                "111008000",
                "010000000",
                "000000000",
                "000000000",
                "000000000",
                "000000000",
            ),
            ("033", "333", "030", "066", "666", "060", "088", "888", "080"),
        ),
        (
            (
                "0011000000",
                "0110000000",
                "0011000000",
                "0000000000",
                "0000000000",
                "0000000000",
                "0302000000",
                "0000000000",
                "0000000000",
                "0000000000",
                "0000000000",
            ),
            ("033022", "330220", "033022"),
        ),
        (
            (
                "00000000000000",
                "01010000000000",
                "00100000000000",
                "01110000000000",
                "00000000000000",
                "00000000000000",
                "00000000000000",
                "00002030400000",
                "00000000000000",
                "00000000000000",
                "00000000000000",
            ),
            ("202303404", "020030040", "222333444"),
        ),
        (
            (
                "00001000000",
                "00011100000",
                "07001000000",
                "00000000000",
                "06000000000",
                "00000000000",
                "03000000000",
                "00000000000",
                "02000000000",
            ),
            (
                "070",
                "777",
                "070",
                "060",
                "666",
                "060",
                "030",
                "333",
                "030",
                "020",
                "222",
                "020",
            ),
        ),
        (
            (
                "00000000000",
                "01100000000",
                "11000000000",
                "11000000000",
                "00000000000",
                "00000000000",
                "00208030600",
                "00000000000",
                "00000000000",
                "00000000000",
                "00000000000",
            ),
            ("022088033066", "220880330660", "220880330660"),
        ),
    }
)


def _rows_key(grid: np.ndarray) -> tuple[str, ...]:
    return tuple(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(grid.shape),
        tuple(int(value) for value in grid.ravel()),
    )


def _pair_key(pair: dict) -> tuple:
    return (
        _grid_key(pair["input"]),
        _grid_key(pair["output"]),
    )


def _is_source_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    return (
        _rows_key(input_grid),
        _rows_key(output_grid),
    ) in _SOURCE_PAIR_KEYS


def _checked_optional_count(
    value: int | None,
    name: str,
    low: int,
    high: int,
) -> int | None:
    if value is None:
        return None

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    if not low <= value <= high:
        raise ValueError(
            f"{name} must be between {low} and {high}"
        )
    return value


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _checked_optional_count(
        num_train, "num_train", 2, 10
    )
    num_test = _checked_optional_count(
        num_test, "num_test", 1, 4
    )

    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        lower = max(2, num_test + 1)
        num_train = lower + (seed // 4) % (11 - lower)

    elif num_train is None:
        lower = max(2, num_test + 1)
        num_train = int(rng.integers(lower, 11))

    elif num_test is None:
        upper = min(4, num_train - 1)
        if upper < 1:
            raise ValueError(
                "no legal num_test exists for num_train"
            )
        num_test = int(rng.integers(1, upper + 1))

    if num_train <= num_test:
        raise ValueError(
            "num_train must be greater than num_test"
        )

    return num_train, num_test


def _sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    # The transformation has no hidden per-puzzle rule. A visual style controls
    # frequency only; all color roles and geometric choices remain input-local.
    return {
        "style": str(rng.choice(["aligned", "mixed", "varied"], p=[0.24, 0.43, 0.33])),
        "repetition": bool(rng.random() < 0.32),
    }


def _connected(mask: np.ndarray) -> bool:
    cells = {tuple(map(int, point)) for point in np.argwhere(mask)}
    if not cells:
        return False
    reached = {next(iter(cells))}
    pending = list(reached)
    while pending:
        row, col = pending.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if abs(dr) + abs(dc) != 1:
                    continue
                neighbor = (row + dr, col + dc)
                if neighbor in cells and neighbor not in reached:
                    reached.add(neighbor)
                    pending.append(neighbor)
    return reached == cells


def _sample_mask_dimensions(
    rng: np.random.Generator, count: int, orientation: str, regime: str,
    style: str,
) -> tuple[int, int, str]:
    def choose_size(low: int, high: int) -> int:
        sizes = np.arange(low, high + 1)
        weights = np.array(
            [5 if n <= 3 else 4 if n == 4 else 3 if n == 5 else 1 for n in sizes],
            dtype=float,
        )
        return int(rng.choice(sizes, p=weights / weights.sum()))

    axis_limit = min(10, 30 // count)
    other_limit = 8 if regime == "source_train" else 9
    if style == "aligned":
        axis_limit = min(axis_limit, 4)
        other_limit = 4
    mode = str(rng.choice(
        ["bar", "solid", "outline", "cross", "irregular", "gapped"],
        p=[0.24, 0.12, 0.12, 0.12, 0.32, 0.08],
    ))
    if mode == "bar":
        lengths = np.arange(2, min(9, max(axis_limit, other_limit)) + 1)
        weights = np.array([10 if n == 2 else 5 if n == 3 else 3 if n == 4 else 1 for n in lengths], dtype=float)
        length = int(rng.choice(lengths, p=weights / weights.sum()))
        if rng.random() < 0.5 and length <= axis_limit:
            axial, transverse = length, 1
        else:
            axial, transverse = 1, min(length, other_limit)
    else:
        low = 3 if mode in ("outline", "cross", "gapped") else 2
        if mode == "irregular" and rng.random() < 0.12:
            low = 1
        axial = choose_size(low, axis_limit)
        transverse = choose_size(low, other_limit)
        if style != "aligned" and count in (3, 5, 6) and rng.random() < 0.12:
            axial = 30 // count
    if orientation == "vertical":
        return axial, transverse, mode
    return transverse, axial, mode


def _sample_mask(
    rng: np.random.Generator, height: int, width: int,
    mode: str, require_connected: bool,
) -> np.ndarray:
    for _ in range(100):
        if mode == "bar" or mode == "solid":
            mask = np.ones((height, width), dtype=bool)
        elif mode == "outline":
            mask = np.zeros((height, width), dtype=bool)
            mask[0, :] = mask[-1, :] = True
            mask[:, 0] = mask[:, -1] = True
            if rng.random() < 0.35:
                mask[int(rng.integers(1, height - 1)), int(rng.integers(1, width - 1))] = True
        elif mode == "cross":
            mask = np.zeros((height, width), dtype=bool)
            row = int(rng.integers(1, height - 1))
            col = int(rng.integers(1, width - 1))
            mask[row, :] = True
            mask[:, col] = True
            if rng.random() < 0.35:
                mask[int(rng.integers(height)), int(rng.integers(width))] = True
        elif mode == "gapped":
            mask = np.zeros((height, width), dtype=bool)
            if rng.random() < 0.5:
                gap = int(rng.integers(1, height - 1))
                mask[0, int(rng.integers(width))] = True
                mask[-1, int(rng.integers(width))] = True
                mask[:, 0] |= rng.random(height) < 0.45
                mask[gap, :] = False
            else:
                gap = int(rng.integers(1, width - 1))
                mask[int(rng.integers(height)), 0] = True
                mask[int(rng.integers(height)), -1] = True
                mask[0, :] |= rng.random(width) < 0.45
                mask[:, gap] = False
        else:
            density = float(rng.uniform(0.32, 0.82))
            mask = rng.random((height, width)) < density
        if int(mask.sum()) < 2:
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if require_connected and not _connected(mask):
            continue
        return mask
    # A connected fallback is always available for any permitted rectangle.
    return np.ones((height, width), dtype=bool)


def _relative_layout(
    rng: np.random.Generator, height: int, width: int,
    count: int, orientation: str, style: str,
) -> list[tuple[int, int]]:
    aligned = style == "aligned" or (style == "mixed" and rng.random() < 0.43) or (style == "varied" and rng.random() < 0.16)
    if aligned:
        steps = [2] * (count - 1)
        offsets = [0] * count
        separation = int(rng.integers(1, 3 if style == "aligned" else 4))
    else:
        steps = [int(value) for value in rng.integers(2, 6, size=count - 1)]
        offsets = [int(value) for value in rng.integers(-1, 2, size=count)]
        separation = int(rng.integers(2, 6))
    if len(set(steps)) == 1 and not aligned and count >= 3 and rng.random() < 0.7:
        steps[-1] = 2 if steps[-1] != 2 else 4
    span = sum(steps)
    cross_span = max(offsets) - min(offsets)
    if span < cross_span + 2:
        offsets = [0] * count
    axis_size = height if orientation == "vertical" else width
    locus_probabilities = [0.66, 0.07, 0.07, 0.20] if style == "aligned" else [0.38, 0.18, 0.18, 0.26]
    locus = str(rng.choice(["overlap", "before", "after", "crossing"], p=locus_probabilities))
    if locus == "before":
        first = -span - 2 - int(rng.integers(0, 4))
    elif locus == "after":
        first = axis_size + 1 + int(rng.integers(0, 4))
    elif locus == "crossing":
        first = -int(rng.integers(0, 5))
    else:
        first = int(rng.integers(-1, axis_size))
    mains = [first]
    for step in steps:
        mains.append(mains[-1] + step)
    side = -1 if rng.random() < 0.5 else 1
    if orientation == "vertical":
        base = -separation - 1 if side < 0 else width + separation
        return [(main, base + offset) for main, offset in zip(mains, offsets)]
    base = -separation - 1 if side < 0 else height + separation
    return [(base + offset, main) for main, offset in zip(mains, offsets)]


def _sample_colors(
    rng: np.random.Generator, count: int, seen_train: set[int],
    regime: str, repeat: bool, previous_roles: tuple[int, int] | None,
) -> tuple[int, int, tuple[int, ...]]:
    for _ in range(100):
        roles = tuple(int(x) for x in rng.choice(10, size=2, replace=False))
        if regime == "source_train" and previous_roles is not None and roles == previous_roles:
            continue
        background, template_color = roles
        available = [color for color in range(10) if color not in roles]
        unique_count = count - int(repeat)
        familiar = [color for color in available if color in seen_train]
        novel = [color for color in available if color not in seen_train]
        if regime == "target_test":
            desired = int(rng.choice([0, 1, 2, 3], p=[0.34, 0.31, 0.23, 0.12]))
            min_novel = max(0, unique_count - len(familiar))
            max_novel = min(unique_count, len(novel))
            n_novel = min(max(desired, min_novel), max_novel)
            selected = [int(x) for x in rng.choice(novel, size=n_novel, replace=False)] if n_novel else []
            selected += [int(x) for x in rng.choice(familiar, size=unique_count - n_novel, replace=False)] if unique_count > n_novel else []
        else:
            selected = [int(x) for x in rng.choice(available, size=unique_count, replace=False)]
        rng.shuffle(selected)
        if repeat:
            repeated = int(rng.choice(selected))
            if rng.random() < 0.55:
                index = int(rng.integers(0, len(selected)))
                selected.insert(index, repeated)
            else:
                selected.append(repeated)
                rng.shuffle(selected)
        return background, template_color, tuple(selected)
    raise RuntimeError("could not sample local colors")


def _generate_pair(
    rng: np.random.Generator, regime: str, style: str,
    orientation: str, count: int, seen_train: set[int],
    repeat: bool, previous_roles: tuple[int, int] | None,
) -> dict:
    for _ in range(160):
        background, template_color, marker_colors = _sample_colors(
            rng, count, seen_train, regime, repeat, previous_roles
        )
        height, width, mode = _sample_mask_dimensions(rng, count, orientation, regime, style)
        if repeat and mode == "gapped":
            mode = "irregular"
        mask = _sample_mask(rng, height, width, mode, repeat)
        if repeat and not _connected(mask):
            continue
        positions = _relative_layout(rng, height, width, count, orientation, style)
        min_row = min(0, min(row for row, _ in positions))
        max_row = max(height - 1, max(row for row, _ in positions))
        min_col = min(0, min(col for _, col in positions))
        max_col = max(width - 1, max(col for _, col in positions))
        extent_height = max_row - min_row + 1
        extent_width = max_col - min_col + 1
        if extent_height > 30 or extent_width > 30:
            continue
        margin_cap = 4 if style == "aligned" else 10
        slack_h = min(margin_cap, 30 - extent_height)
        slack_w = min(margin_cap, 30 - extent_width)
        extra_h = int(rng.integers(0, slack_h + 1))
        extra_w = int(rng.integers(0, slack_w + 1))
        top = int(rng.integers(0, extra_h + 1))
        left = int(rng.integers(0, extra_w + 1))
        row_shift, col_shift = top - min_row, left - min_col
        input_grid = np.full((extent_height + extra_h, extent_width + extra_w), background, dtype=np.int8)
        rr, cc = np.nonzero(mask)
        input_grid[rr + row_shift, cc + col_shift] = template_color
        for (row, col), color in zip(positions, marker_colors):
            input_grid[row + row_shift, col + col_shift] = color
        if int(np.count_nonzero(input_grid == background)) <= input_grid.size // 2:
            continue
        tiles = [np.where(mask, color, background).astype(np.int8) for color in marker_colors]
        output_grid = np.concatenate(tiles, axis=0 if orientation == "vertical" else 1)
        pair = {"input": input_grid, "output": output_grid}
        try:
            _decode_pair(input_grid, output_grid)
        except AssertionError:
            continue
        return pair
    # Compact fallback keeps legal calls total even for unfavorable random layouts.
    background, template_color, marker_colors = _sample_colors(
        rng, count, seen_train, regime, repeat, previous_roles
    )
    mask = np.ones((3, 3), dtype=bool)
    h, w = mask.shape
    positions = [(2 * i, w + 2) if orientation == "vertical" else (h + 2, 2 * i) for i in range(count)]
    input_grid = np.full((max(h, 2 * count - 1) + 2 if orientation == "vertical" else h + 3,
                          w + 3 if orientation == "vertical" else max(w, 2 * count - 1) + 2), background, dtype=np.int8)
    input_grid[:h, :w] = template_color
    for (row, col), color in zip(positions, marker_colors):
        input_grid[row, col] = color
    tiles = [np.where(mask, color, background).astype(np.int8) for color in marker_colors]
    output_grid = np.concatenate(tiles, axis=0 if orientation == "vertical" else 1)
    return {"input": input_grid, "output": output_grid}


def _components(points: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    remaining = set(points)
    groups = []
    while remaining:
        group = {remaining.pop()}
        frontier = list(group)
        while frontier:
            row, col = frontier.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    candidate = (row + dr, col + dc)
                    if candidate in remaining:
                        remaining.remove(candidate)
                        group.add(candidate)
                        frontier.append(candidate)
        groups.append(group)
    return groups


def _decode_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> dict:
    values, counts = np.unique(input_grid, return_counts=True)
    assert len(values) >= 3
    assert int(counts.max()) > input_grid.size // 2
    background = int(values[int(np.argmax(counts))])
    foreground = {(int(row), int(col)) for row, col in np.argwhere(input_grid != background)}
    multi = [component for component in _components(foreground) if len(component) >= 2]
    if len(multi) == 1:
        component_colors = {int(input_grid[row, col]) for row, col in multi[0]}
        assert len(component_colors) == 1
        template_color = next(iter(component_colors))
    else:
        repeated = [int(value) for value, n in zip(values, counts) if int(value) != background and int(n) >= 2]
        assert len(repeated) == 1
        template_color = repeated[0]
    template_cells = np.argwhere(input_grid == template_color)
    assert len(template_cells) >= 2
    row0, col0 = [int(x) for x in template_cells.min(axis=0)]
    row1, col1 = [int(x) for x in template_cells.max(axis=0)]
    rectangle = input_grid[row0:row1 + 1, col0:col1 + 1]
    assert set(int(x) for x in np.unique(rectangle)) <= {background, template_color}
    mask = rectangle == template_color
    assert mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()
    markers = [(int(row), int(col), int(input_grid[row, col])) for row, col in foreground if int(input_grid[row, col]) != template_color]
    assert 2 <= len(markers) <= 8
    for row, col, color in markers:
        assert color != background
        assert not (row0 <= row <= row1 and col0 <= col <= col1)
        assert all(max(abs(row - r), abs(col - c)) >= 2 for r, c in foreground if (r, c) != (row, col))
    row_span = max(row for row, _, _ in markers) - min(row for row, _, _ in markers)
    col_span = max(col for _, col, _ in markers) - min(col for _, col, _ in markers)
    if row_span >= col_span + 2:
        orientation = "vertical"
        assert len({row for row, _, _ in markers}) == len(markers)
        ordered = sorted(markers)
        assert all(ordered[i + 1][0] - ordered[i][0] >= 2 for i in range(len(ordered) - 1))
        assert all(col <= col0 - 2 for _, col, _ in markers) or all(col >= col1 + 2 for _, col, _ in markers)
        side = "left" if all(col <= col0 - 2 for _, col, _ in markers) else "right"
        axis = 0
    else:
        assert col_span >= row_span + 2
        orientation = "horizontal"
        assert len({col for _, col, _ in markers}) == len(markers)
        ordered = sorted(markers, key=lambda item: item[1])
        assert all(ordered[i + 1][1] - ordered[i][1] >= 2 for i in range(len(ordered) - 1))
        assert all(row <= row0 - 2 for row, _, _ in markers) or all(row >= row1 + 2 for row, _, _ in markers)
        side = "above" if all(row <= row0 - 2 for row, _, _ in markers) else "below"
        axis = 1
    expected = np.concatenate([np.where(mask, color, background) for _, _, color in ordered], axis=axis)
    assert np.array_equal(output_grid, expected)
    assert not np.array_equal(input_grid, output_grid)
    colors = [color for _, _, color in ordered]
    if len(set(colors)) < len(colors):
        assert _connected(mask)
    mains = [item[0 if axis == 0 else 1] for item in ordered]
    crosses = [item[1 if axis == 0 else 0] for item in ordered]
    template_min, template_max = (row0, row1) if axis == 0 else (col0, col1)
    if max(mains) < template_min:
        locus = "before"
    elif min(mains) > template_max:
        locus = "after"
    else:
        locus = "alongside"
    return {
        "background": background, "template_color": template_color,
        "bbox": (row0, col0, row1, col1), "mask": mask,
        "markers": ordered, "orientation": orientation, "side": side,
        "locus": locus, "steps": tuple(b - a for a, b in zip(mains, mains[1:])),
        "aligned": len(set(crosses)) == 1, "colors": tuple(colors),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    decoded = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            input_key, pair_key = _grid_key(pair["input"]), _pair_key(pair)
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert not _is_source_pair(pair["input"], pair["output"])
            info = _decode_pair(pair["input"], pair["output"])
            n = len(info["markers"])
            assert 2 <= n <= (5 if split == "train" else 8)
            decoded[split].append(info)
    train = decoded["train"]
    assert {info["orientation"] for info in train} == {"vertical", "horizontal"}
    assert len({(info["background"], info["template_color"]) for info in train}) >= 2
    if any(len(set(info["colors"])) < len(info["colors"]) for info in decoded["test"]):
        assert any(len(set(info["colors"])) < len(info["colors"]) for info in train)
    if len(train) >= 4:
        pairs = {(info["orientation"], len(info["markers"])) for info in train}
        assert {("vertical", 2), ("horizontal", 2), ("vertical", 3), ("horizontal", 3)} <= pairs


def _train_schedule(rng: np.random.Generator, count: int) -> list[tuple[str, int]]:
    orientations = ["vertical", "horizontal"]
    if count >= 4:
        schedule = [(orientation, n) for n in (2, 3) for orientation in orientations]
    elif count == 3:
        schedule = [("vertical", 2), ("horizontal", 2), (str(rng.choice(orientations)), 3)]
    else:
        first = str(rng.choice(orientations))
        second = orientations[1 - orientations.index(first)]
        schedule = [(first, 2), (second, 2 if rng.random() < 0.35 else 3)]
    while len(schedule) < count:
        schedule.append((str(rng.choice(orientations)), int(rng.choice([2, 3, 4, 5], p=[0.27, 0.34, 0.27, 0.12]))))
    rng.shuffle(schedule)
    return schedule


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, rng, num_train, num_test)
    params = _sample_puzzle_parameters(rng)
    puzzle = {"train": [], "test": []}
    used = set()
    seen_train_colors: set[int] = set()
    first_roles = None
    repeated_train = False
    schedule = _train_schedule(rng, num_train)
    repeat_slot = int(rng.integers(num_train)) if params["repetition"] else -1
    for index, (orientation, count) in enumerate(schedule):
        pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
        for _ in range(40):
            repeat = index == repeat_slot or (params["repetition"] and pair_rng.random() < 0.16)
            pair = _generate_pair(pair_rng, "source_train", params["style"], orientation,
                                  count, seen_train_colors, repeat, first_roles)
            key = _grid_key(pair["input"])
            if key not in used and not _is_source_pair(pair["input"], pair["output"]):
                break
        else:
            raise RuntimeError("could not create unique training pair")
        puzzle["train"].append(pair)
        used.add(key)
        info = _decode_pair(pair["input"], pair["output"])
        seen_train_colors.update(info["colors"])
        repeated_train |= len(set(info["colors"])) < len(info["colors"])
        if first_roles is None:
            first_roles = info["background"], info["template_color"]
    for _ in range(num_test):
        pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
        orientation = str(pair_rng.choice(["vertical", "horizontal"]))
        count = int(pair_rng.choice([2, 3, 4, 5, 6, 7, 8], p=[0.09, 0.12, 0.20, 0.20, 0.17, 0.13, 0.09]))
        for _ in range(40):
            repeat = repeated_train and pair_rng.random() < 0.42
            pair = _generate_pair(pair_rng, "target_test", params["style"], orientation,
                                  count, seen_train_colors, repeat, None)
            key = _grid_key(pair["input"])
            if key not in used and not _is_source_pair(pair["input"], pair["output"]):
                break
        else:
            raise RuntimeError("could not create unique test pair")
        puzzle["test"].append(pair)
        used.add(key)
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple((split, tuple(_pair_key(pair) for pair in puzzle[split])) for split in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, signatures = set(), set(), set()
    shapes, masks, roles, layouts = set(), set(), set(), set()
    sides = {"train": set(), "test": set()}
    test_lengths, novelty_counts = set(), set()
    aligned_puzzles = 0
    crossed = 0
    staggered = irregular_steps = compact = full = two_cell = one_wide = large = blank_line = max_output = repeats = consecutive_repeats = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(_puzzle_key(puzzle))
        infos = {split: [_decode_pair(pair["input"], pair["output"]) for pair in puzzle[split]] for split in ("train", "test")}
        assert {info["orientation"] for info in infos["train"]} == {"vertical", "horizontal"}
        assert len({(info["background"], info["template_color"]) for info in infos["train"]}) >= 2
        if all(info["aligned"] and set(info["steps"]) == {2} for split in infos for info in infos[split]):
            aligned_puzzles += 1
        if {("vertical", 2), ("horizontal", 2), ("vertical", 3), ("horizontal", 3)} <= {(info["orientation"], len(info["markers"])) for info in infos["train"]}:
            crossed += 1
        seen_colors = {color for info in infos["train"] for color in info["colors"]}
        for split in ("train", "test"):
            for pair, info in zip(puzzle[split], infos[split]):
                mask = info["mask"]
                n = len(info["markers"])
                sides[split].add(info["side"])
                shapes.add(tuple(pair["input"].shape))
                masks.add((tuple(mask.shape), tuple(bool(x) for x in mask.ravel())))
                roles.add((info["background"], info["template_color"]))
                layouts.add((info["side"], info["locus"], n, tuple(mask.shape), tuple(pair["input"].shape)))
                staggered += not info["aligned"]
                irregular_steps += len(set(info["steps"])) > 1
                compact += info["aligned"] and set(info["steps"]) == {2}
                full += bool(mask.all())
                two_cell += int(mask.sum()) == 2
                one_wide += min(mask.shape) == 1
                large += max(mask.shape) >= 6
                blank_line += any(not row.any() for row in mask[1:-1]) or any(not col.any() for col in mask.T[1:-1])
                max_output += 30 in pair["output"].shape
                repeats += len(set(info["colors"])) < n
                consecutive_repeats += any(a == b for a, b in zip(info["colors"], info["colors"][1:]))
                if split == "test":
                    test_lengths.add(n)
                    novelty_counts.add(len(set(info["colors"]) - seen_colors))
    assert len(train_counts) >= 4 and len(test_counts) >= 3 and len(signatures) == 42
    assert crossed >= 15 and aligned_puzzles >= 4
    assert sides["train"] == sides["test"] == {"left", "right", "above", "below"}
    assert test_lengths == set(range(2, 9)) and {0, 1, 2} <= novelty_counts
    assert len(shapes) >= 45 and len(masks) >= 80 and len(roles) >= 35 and len(layouts) >= 100
    assert staggered >= 25 and irregular_steps >= 20 and compact >= 35
    assert full >= 12 and two_cell >= 4 and one_wide >= 12 and large >= 35 and blank_line >= 5
    assert max_output >= 2 and repeats >= 12 and consecutive_repeats >= 4
    a = generate_puzzle(12345, 6, 3)
    b = generate_puzzle(12345, 6, 3)
    assert _puzzle_key(a) == _puzzle_key(b)
    assert _puzzle_key(a) != _puzzle_key(generate_puzzle(12346, 6, 3))
    for seed, tr, te in [(100, 2, 1), (101, 10, 1), (102, 10, 4), (103, None, 1), (104, None, 4), (105, 2, None), (106, 10, None)]:
        puzzle = generate_puzzle(seed, tr, te)
        validate_generated_puzzle(puzzle)
        if tr is not None:
            assert len(puzzle["train"]) == tr
        if te is not None:
            assert len(puzzle["test"]) == te
    for options in [{"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}]:
        try:
            generate_puzzle(999, **options)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {options}")
