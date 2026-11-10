import numpy as np


_SOURCE_EXACT_PAIRS = (
    (
        (
            "0002000200000000",
            "0002000200000000",
            "0002222200000000",
            "0002000200000000",
            "0002000200000000",
            "2222222200000000",
            "0000000200000000",
            "0000000222222222",
            "0000000200200000",
            "0000000200200000",
            "0000000200200000",
        ),
        (
            "0002888211111111",
            "0002888211111111",
            "0002222211111111",
            "0002888211111111",
            "0002888211111111",
            "2222222211111111",
            "0000000211111111",
            "0000000222222222",
            "0000000288200000",
            "0000000288200000",
            "0000000288200000",
        ),
    ),
    (
        (
            "0000200000000",
            "0000222222222",
            "0000200000000",
            "0000200000000",
            "0000200000000",
            "2222200000000",
            "0000200000000",
            "0000200000000",
            "0000200000000",
            "0000200000000",
            "0000200000000",
        ),
        (
            "0000288888888",
            "0000222222222",
            "0000211111111",
            "0000211111111",
            "0000211111111",
            "2222211111111",
            "0000211111111",
            "0000211111111",
            "0000211111111",
            "0000211111111",
            "0000211111111",
        ),
    ),
    (
        (
            "0002000000002000",
            "0002000000002000",
            "0002000000002222",
            "0002000000002000",
            "2222222222222000",
            "0020000000002000",
            "0020000000002000",
            "0020000000002000",
            "0020000000002000",
            "2220000000002000",
            "0020000000002222",
            "0020000000002000",
            "0020000000002000",
            "0020000000002000",
            "0020000000002000",
        ),
        (
            "0002000000002888",
            "0002000000002888",
            "0002000000002222",
            "0002000000002000",
            "2222222222222000",
            "0021111111112000",
            "0021111111112000",
            "0021111111112000",
            "0021111111112000",
            "2221111111112000",
            "0021111111112222",
            "0021111111112000",
            "0021111111112000",
            "0021111111112000",
            "0021111111112000",
        ),
    ),
    (
        (
            "0200002000000",
            "2200002000000",
            "0200002000000",
            "0200002222222",
            "0200002002000",
            "0222222002000",
            "0200002002000",
            "0200002222222",
            "0200002000020",
            "2222222222222",
            "0020000000000",
            "0020000000000",
            "0020000000000",
            "0020000000000",
            "0020000000000",
            "0020000000000",
            "0020000000000",
            "0020000000000",
        ),
        (
            "8200002000000",
            "2200002000000",
            "0200002000000",
            "0200002222222",
            "0200002002000",
            "0222222002000",
            "0200002002000",
            "0200002222222",
            "0200002000028",
            "2222222222222",
            "0021111111111",
            "0021111111111",
            "0021111111111",
            "0021111111111",
            "0021111111111",
            "0021111111111",
            "0021111111111",
            "0021111111111",
        ),
    ),
    (
        (
            "0000000000200000",
            "0000000000200000",
            "2222222222200000",
            "0000000000200000",
            "0000000000222222",
            "0000000000200000",
            "0000000000200000",
            "0000000000200000",
            "0000000000200000",
            "2222222222222222",
            "0002000000202000",
            "2222000000202000",
            "0002000000202000",
        ),
        (
            "0000000000200000",
            "0000000000200000",
            "2222222222200000",
            "1111111111200000",
            "1111111111222222",
            "1111111111200000",
            "1111111111200000",
            "1111111111200000",
            "1111111111200000",
            "2222222222222222",
            "8882000000282000",
            "2222000000282000",
            "8882000000282000",
        ),
    ),
)

_SOURCE_WALL_KEYS = frozenset(
    (
        len(input_rows),
        len(input_rows[0]),
        tuple(cell == "2" for row in input_rows for cell in row),
    )
    for input_rows, _ in _SOURCE_EXACT_PAIRS
)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    components = []

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue

            seen[y, x] = True
            stack = [(y, x)]
            cells = []

            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))

                for ny, nx in (
                    (cy - 1, cx),
                    (cy + 1, cx),
                    (cy, cx - 1),
                    (cy, cx + 1),
                ):
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and not seen[ny, nx]
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))

            components.append(np.asarray(cells, dtype=np.int16))

    return components


def _component_bounds(component: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(component[:, 0].min()),
        int(component[:, 0].max()),
        int(component[:, 1].min()),
        int(component[:, 1].max()),
    )


def _is_rectangular_component(component: np.ndarray) -> bool:
    y0, y1, x0, x1 = _component_bounds(component)
    return len(component) == (y1 - y0 + 1) * (x1 - x0 + 1)


def _intervals_from_gaps(
    gaps: list[int],
) -> tuple[list[tuple[int, int]], list[int], int]:
    intervals = []
    cuts = []
    position = 0

    for index, gap in enumerate(gaps):
        gap = int(gap)
        intervals.append((position, position + gap - 1))
        position += gap

        if index + 1 < len(gaps):
            cuts.append(position)
            position += 1

    return intervals, cuts, position


def _wall_mask_key(mask: np.ndarray) -> tuple:
    return (
        int(mask.shape[0]),
        int(mask.shape[1]),
        tuple(bool(value) for value in mask.ravel()),
    )


def _layout_info(wall_mask: np.ndarray) -> dict | None:
    if (
        not isinstance(wall_mask, np.ndarray)
        or wall_mask.ndim != 2
        or wall_mask.dtype != np.bool_
    ):
        return None

    height, width = wall_mask.shape

    if (
        height < 1
        or width < 1
        or not wall_mask.any()
        or wall_mask.all()
    ):
        return None

    # The wall network itself must be orthogonally connected.
    if len(_components_mask(wall_mask)) != 1:
        return None

    # Disallow two-cell-thick wall blocks.
    if (
        height > 1
        and width > 1
        and np.any(
            wall_mask[:-1, :-1]
            & wall_mask[1:, :-1]
            & wall_mask[:-1, 1:]
            & wall_mask[1:, 1:]
        )
    ):
        return None

    # Any interior endpoint must meet another wall segment.
    for y, x in np.argwhere(wall_mask):
        degree = 0
        for ny, nx in (
            (y - 1, x),
            (y + 1, x),
            (y, x - 1),
            (y, x + 1),
        ):
            if (
                0 <= ny < height
                and 0 <= nx < width
                and wall_mask[ny, nx]
            ):
                degree += 1

        if 0 < y < height - 1 and 0 < x < width - 1 and degree < 2:
            return None

    components = _components_mask(~wall_mask)

    if len(components) < 3:
        return None

    if any(not _is_rectangular_component(c) for c in components):
        return None

    areas = [len(component) for component in components]
    minimum = min(areas)
    maximum = max(areas)

    minimum_indices = [
        index for index, area in enumerate(areas) if area == minimum
    ]
    maximum_indices = [
        index for index, area in enumerate(areas) if area == maximum
    ]

    if len(maximum_indices) != 1 or minimum == maximum:
        return None

    # At least one intermediate compartment must remain unchanged.
    if len(components) <= len(minimum_indices) + 1:
        return None

    return {
        "components": components,
        "areas": areas,
        "bounds": [
            _component_bounds(component) for component in components
        ],
        "min_ids": minimum_indices,
        "max_idx": maximum_indices[0],
        "wall_mask": wall_mask,
    }


def _paint_layout(
    wall_mask: np.ndarray,
    roles: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    background, wall, major, minor = roles
    info = _layout_info(wall_mask)
    assert info is not None

    input_grid = np.full(
        wall_mask.shape,
        background,
        dtype=np.int8,
    )
    input_grid[wall_mask] = wall
    output_grid = input_grid.copy()

    minimum = min(info["areas"])
    maximum = max(info["areas"])

    for component, area in zip(info["components"], info["areas"]):
        if area == maximum:
            output_grid[component[:, 0], component[:, 1]] = major
        elif area == minimum:
            output_grid[component[:, 0], component[:, 1]] = minor

    return input_grid, output_grid


def _array_key(array: np.ndarray) -> tuple:
    return (
        tuple(int(v) for v in array.shape),
        tuple(int(v) for v in array.ravel()),
    )


def _row_key(array: np.ndarray) -> tuple[str, ...]:
    return tuple(
        "".join(str(int(value)) for value in row)
        for row in array
    )


def _is_source_exact(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    return (
        _row_key(input_grid),
        _row_key(output_grid),
    ) in _SOURCE_EXACT_PAIRS


def _analyze_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> dict:
    input_colors = [int(value) for value in np.unique(input_grid)]
    assert len(input_colors) == 2

    candidates = []

    for wall in input_colors:
        background = (
            input_colors[0]
            if wall == input_colors[1]
            else input_colors[1]
        )
        info = _layout_info(input_grid == wall)

        if info is not None:
            candidates.append((background, wall, info))

    # Only the true wall color can form one connected partition network.
    assert len(candidates) == 1
    background, wall, info = candidates[0]

    assert output_grid.shape == input_grid.shape
    assert np.all(output_grid[info["wall_mask"]] == wall)

    component_colors = []

    for component in info["components"]:
        values = output_grid[component[:, 0], component[:, 1]]
        assert np.all(values == values[0])
        component_colors.append(int(values[0]))

    major = component_colors[info["max_idx"]]
    minor_colors = {
        component_colors[index] for index in info["min_ids"]
    }

    assert len(minor_colors) == 1
    minor = next(iter(minor_colors))
    assert len({background, wall, major, minor}) == 4

    for index, color in enumerate(component_colors):
        if index == info["max_idx"]:
            expected = major
        elif index in info["min_ids"]:
            expected = minor
        else:
            expected = background

        assert color == expected

    assert set(int(value) for value in np.unique(output_grid)) == {
        background,
        wall,
        major,
        minor,
    }
    assert not np.array_equal(input_grid, output_grid)

    result = dict(info)
    result["roles"] = (background, wall, major, minor)
    return result


def _split_gaps(total, forced_count, forced_height, minimum, interior, rng):
    """Compose one band into background intervals separated by wall rows."""
    options = list(range(max(1, forced_count), 5))
    rng.shuffle(options)
    for count in options:
        positions = list(range(count))
        if interior and count < 3:
            continue
        fixed = {}
        if forced_count:
            if interior:
                targets = [1] + [j for j in range(count) if j != 1][:forced_count - 1]
            else:
                rng.shuffle(positions)
                targets = positions[:forced_count]
            for pos in targets:
                fixed[pos] = forced_height
        base = [fixed.get(j, minimum) for j in range(count)]
        spare = total - (count - 1) - sum(base)
        free = [j for j in range(count) if j not in fixed]
        if spare < 0 or (spare and not free):
            continue
        if free:
            additions = rng.multinomial(spare, [1 / len(free)] * len(free))
            for j, extra in zip(free, additions):
                base[j] += int(extra)
        return base
    return None


def _make_wall_mask(rng, tie_count, shape_kind, family, complexity):
    # A set of full-height trunks connects horizontal cuts made independently
    # in each band. The branch family adds a perpendicular cut inside a room.
    for _ in range(450):
        if shape_kind == "thin_trap":
            min_height, min_width = 2, 3
        elif shape_kind == "mixed":
            min_height, min_width = 1, 3
        else:
            min_height, min_width = (
                (1, 1), (1, 2), (1, 3), (1, 5),
                (1, 8), (2, 2), (2, 3), (3, 2), (2, 4),
            )[int(rng.integers(0, 9))]
        area = min_height * min_width
        band_count = int(rng.integers(3, 5 if min_width >= 8 else 6))
        widths = [min_width]
        if shape_kind == "thin_trap":
            widths.append(1)
        elif shape_kind == "mixed":
            widths.append(1)
        while len(widths) < band_count:
            widths.append(int(rng.integers(2, 10)))
        rng.shuffle(widths)
        canvas_width = sum(widths) + len(widths) - 1
        if canvas_width > 29:
            continue
        height = int(rng.integers(max(14, tie_count * min_height + 7), 29))
        chosen = [j for j, width in enumerate(widths) if width == min_width]
        target_band = int(rng.choice(chosen))
        if shape_kind == "interior":
            internal = [j for j in chosen if 0 < j < band_count - 1]
            if not internal:
                continue
            target_band = int(rng.choice(internal))
        forced_counts = [0] * band_count
        forced_counts[target_band] = tie_count
        if shape_kind == "mixed" and tie_count >= 2:
            thin = [j for j, width in enumerate(widths) if width == 1 and j != target_band]
            if not thin:
                continue
            forced_counts[target_band] -= 1
            forced_counts[thin[0]] = 1
        gap_sets = []
        valid = True
        for band, width in enumerate(widths):
            forced_count = forced_counts[band]
            if forced_count:
                if area % width:
                    valid = False
                    break
                min_gap = area // width
                interior = shape_kind == "interior" and band == target_band
                gaps = _split_gaps(height, forced_count, min_gap, area // width + 1, interior, rng)
            else:
                gaps = _split_gaps(height, 0, 0, area // width + 1, False, rng)
            if gaps is None:
                valid = False
                break
            gap_sets.append(gaps)
        if not valid:
            continue
        wall = np.zeros((height, canvas_width), dtype=bool)
        start = 0
        for band, (width, gaps) in enumerate(zip(widths, gap_sets)):
            end = start + width - 1
            row = 0
            for gap in gaps[:-1]:
                row += gap
                wall[row, start:end + 1] = True
                row += 1
            if band < band_count - 1:
                wall[:, end + 1] = True
            start = end + 2
        info = _layout_info(wall)
        if info is None or len(info["min_ids"]) != tie_count or min(info["areas"]) != area:
            continue
        if shape_kind == "interior":
            if not any(
                y0 > 0 and y1 < height - 1 and x0 > 0 and x1 < canvas_width - 1
                for i in info["min_ids"]
                for y0, y1, x0, x1 in [info["bounds"][i]]
            ):
                continue
        if shape_kind == "thin_trap":
            if not any(
                (x1 == x0 or y1 == y0) and area < a < max(info["areas"])
                for a, (y0, y1, x0, x1) in zip(info["areas"], info["bounds"])
            ):
                continue
        if family in ("branch", "recursive"):
            options = []
            for i, (y0, y1, x0, x1) in enumerate(info["bounds"]):
                if i == info["max_idx"] or i in info["min_ids"]:
                    continue
                for x in range(x0 + 2, x1 - 1):
                    if (x - x0) * (y1 - y0 + 1) > area and (x1 - x) * (y1 - y0 + 1) > area:
                        options.append((y0, y1, x))
            if not options:
                continue
            y0, y1, x = options[int(rng.integers(len(options)))]
            wall[y0:y1 + 1, x] = True
            info = _layout_info(wall)
            if info is None or len(info["min_ids"]) != tie_count or min(info["areas"]) != area:
                continue
        if family == "recursive":
            for _ in range(int(rng.integers(1, 3))):
                options = []
                for i, (y0, y1, x0, x1) in enumerate(info["bounds"]):
                    if i == info["max_idx"] or i in info["min_ids"]:
                        continue
                    for x in range(x0 + 2, x1 - 1):
                        if min(x - x0, x1 - x) * (y1 - y0 + 1) > area:
                            options.append(("vertical", y0, y1, x))
                    for y in range(y0 + 2, y1 - 1):
                        if min(y - y0, y1 - y) * (x1 - x0 + 1) > area:
                            options.append(("horizontal", x0, x1, y))
                if not options:
                    break
                axis, low, high, position = options[int(rng.integers(len(options)))]
                if axis == "vertical":
                    wall[low:high + 1, position] = True
                else:
                    wall[position, low:high + 1] = True
                info = _layout_info(wall)
                if info is None:
                    break
            if info is None or len(info["min_ids"]) != tie_count or min(info["areas"]) != area:
                continue
        if complexity == "dense" and len(info["components"]) < 6:
            continue
        if int(rng.integers(2)):
            wall = wall.T
        if int(rng.integers(2)):
            wall = np.flipud(wall)
        if int(rng.integers(2)):
            wall = np.fliplr(wall)
        wall = np.ascontiguousarray(wall)
        if _wall_mask_key(wall) in _SOURCE_WALL_KEYS:
            continue
        return wall
    raise RuntimeError("could not construct a partition")


def _requested_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    train = _requested_count(num_train, "num_train", 2, 10)
    test = _requested_count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must be greater than num_test")
    return train, test


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    major, minor = (int(x) for x in rng.choice(10, size=2, replace=False))
    available = [c for c in range(10) if c not in (major, minor)]
    fixed_palette = bool(rng.integers(2))
    background, wall = (int(x) for x in rng.choice(available, size=2, replace=False))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            if split == "train":
                tie_count = (1, 3, 2)[index % 3]
            else:
                tie_count = 1 + (seed + index) % 3
            ordinal = index if split == "train" else train_count + index
            shape_kind = ("general", "interior", "thin_trap", "mixed")[
                (seed + ordinal) % 4
            ]
            if split == "train" and index == seed % 2:
                shape_kind = "thin_trap"
            family = ("band", "branch", "recursive")[(seed + 2 * ordinal) % 3]
            complexity = "dense" if split == "test" and ordinal % 2 == 0 else "any"
            for _ in range(40):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                mask = _make_wall_mask(pair_rng, tie_count, shape_kind, family, complexity)
                if not fixed_palette:
                    background, wall = (int(x) for x in rng.choice(available, size=2, replace=False))
                pair = dict(zip(("input", "output"), _paint_layout(mask, (background, wall, major, minor))))
                key = _array_key(pair["input"])
                if key in seen or _is_source_exact(pair["input"], pair["output"]):
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("could not make a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    highlights = None
    training_ties = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert pair["input"].shape == pair["output"].shape
            key = _array_key(pair["input"])
            assert key not in seen
            seen.add(key)
            assert not _is_source_exact(pair["input"], pair["output"])
            info = _analyze_pair(pair["input"], pair["output"])
            assert _wall_mask_key(info["wall_mask"]) not in _SOURCE_WALL_KEYS
            assert 3 <= len(info["components"])
            assert 1 <= len(info["min_ids"]) <= 3
            pair_highlights = info["roles"][2:]
            if highlights is None:
                highlights = pair_highlights
            assert highlights == pair_highlights
            if split == "train":
                training_ties.append(len(info["min_ids"]))
    assert 1 in training_ties and 3 in training_ties
    if len(training_ties) >= 3:
        assert 2 in training_ties


def _puzzles_equal(a, b):
    return all(
        len(a[split]) == len(b[split]) and all(
            np.array_equal(pa[side], pb[side])
            for pa, pb in zip(a[split], b[split]) for side in ("input", "output")
        ) for split in ("train", "test")
    )


def self_test() -> None:
    a = generate_puzzle(31415)
    assert _puzzles_equal(a, generate_puzzle(31415))
    assert not _puzzles_equal(a, generate_puzzle(31416))
    assert _puzzles_equal(generate_puzzle(2718, 6, 2), generate_puzzle(2718, 6, 2))
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(9000 + (train or 0) * 10 + (test or 0), train, test)
        assert train is None or len(p["train"]) == train
        assert test is None or len(p["test"]) == test
        validate_generated_puzzle(p)
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                 {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(7, **args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    counts = set()
    test_counts = set()
    layouts = set()
    colors = set()
    shapes = set()
    min_areas = set()
    tie_counts = set()
    interior_min = False
    interior_max = False
    two_d_min = False
    thin_intermediate = False
    combined_area_trap = False
    mixed_min_shapes = False
    rank_not_widest = False
    rank_not_tallest = False
    rank_not_longest_perimeter = False
    fixed_palettes = 0
    varied_palettes = 0
    minimum_shapes = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        validate_generated_puzzle(puzzle)
        palettes = set()
        training_area_trap = False
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze_pair(pair["input"], pair["output"])
                palettes.add(info["roles"][:2])
                layouts.add(_wall_mask_key(info["wall_mask"]))
                colors.add(info["roles"])
                shapes.add(pair["input"].shape)
                area = min(info["areas"])
                min_areas.add(area)
                tie_counts.add(len(info["min_ids"]))
                h, w = pair["input"].shape
                y0, y1, x0, x1 = info["bounds"][info["max_idx"]]
                interior_max |= 0 < y0 and y1 < h - 1 and 0 < x0 and x1 < w - 1
                room_widths = [bx1 - bx0 + 1 for _, _, bx0, bx1 in info["bounds"]]
                room_heights = [by1 - by0 + 1 for by0, by1, _, _ in info["bounds"]]
                major_index = info["max_idx"]
                rank_not_widest |= room_widths[major_index] < max(room_widths)
                rank_not_tallest |= room_heights[major_index] < max(room_heights)
                perimeters = [2 * (rh + rw) for rh, rw in zip(room_heights, room_widths)]
                rank_not_longest_perimeter |= perimeters[major_index] < max(perimeters)
                pair_two_d = False
                for i in info["min_ids"]:
                    y0, y1, x0, x1 = info["bounds"][i]
                    minimum_shapes.add((y1 - y0 + 1, x1 - x0 + 1))
                    interior_min |= 0 < y0 and y1 < h - 1 and 0 < x0 and x1 < w - 1
                    pair_two_d |= y1 > y0 and x1 > x0
                two_d_min |= pair_two_d
                mixed_min_shapes |= len({
                    (room_heights[i], room_widths[i]) for i in info["min_ids"]
                }) > 1
                pair_thin = any(
                    (y0 == y1 or x0 == x1) and area < a < max(info["areas"])
                    for i, (a, (y0, y1, x0, x1)) in enumerate(zip(info["areas"], info["bounds"]))
                )
                thin_intermediate |= pair_thin
                combined_area_trap |= pair_two_d and pair_thin
                if split == "train":
                    training_area_trap |= pair_two_d and pair_thin
        assert training_area_trap
        if len(palettes) == 1:
            fixed_palettes += 1
        else:
            varied_palettes += 1
    assert len(counts) >= 4 and len(test_counts) >= 3
    assert len(layouts) >= 120 and len(colors) >= 30 and len(shapes) >= 20
    assert len(min_areas) >= 5 and tie_counts == {1, 2, 3}
    assert interior_min and two_d_min and thin_intermediate
    assert interior_max and combined_area_trap and mixed_min_shapes
    assert rank_not_widest and rank_not_tallest and rank_not_longest_perimeter
    assert (1, 1) in minimum_shapes and (1, 8) in minimum_shapes
    assert (2, 3) in minimum_shapes and (3, 2) in minimum_shapes
    assert fixed_palettes >= 10 and varied_palettes >= 10
