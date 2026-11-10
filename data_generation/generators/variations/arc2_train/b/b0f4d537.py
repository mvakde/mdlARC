import numpy as np


_SOURCE_PAIRS = (
    (
        (
            "0101050040040",
            "2121250000000",
            "0101050000000",
            "3131354040044",
            "0101050000000",
            "5555550000000",
            "5555550000000",
            "5555554040044",
            "5555550000000",
            "5555550000000",
            "5555550040040",
        ),
        (
            "0010010",
            "0010010",
            "0010010",
            "2212212",
            "0010010",
            "0010010",
            "0010010",
            "3313313",
            "0010010",
            "0010010",
            "0010010",
        ),
    ),
    (
        (
            "0201050400040",
            "3333354400044",
            "0201050000000",
            "3333350000000",
            "0201050000000",
            "5555550000000",
            "5555554400044",
            "5555550000000",
            "5555550400040",
        ),
        (
            "0200010",
            "3333333",
            "0200010",
            "0200010",
            "0200010",
            "0200010",
            "3333333",
            "0200010",
            "0200010",
        ),
    ),
    (
        (
            "0040400502030",
            "0000000512131",
            "0000000502030",
            "0000000555555",
            "4040404555555",
            "0000000555555",
            "0040400555555",
        ),
        (
            "0020300",
            "0020300",
            "0020300",
            "0020300",
            "1121311",
            "0020300",
            "0020300",
        ),
    ),
    (
        (
            "01050000400",
            "33350000000",
            "01054000404",
            "21250000000",
            "01050000000",
            "55554000404",
            "55550000000",
            "55550000000",
            "55550000400",
        ),
        (
            "0000100",
            "0000100",
            "3333333",
            "0000100",
            "0000100",
            "2222122",
            "0000100",
            "0000100",
            "0000100",
        ),
    ),
    (
        (
            "040404050202020",
            "000000053233323",
            "000000050202020",
            "000000051112111",
            "440404450202020",
            "000000055555555",
            "000000055555555",
            "000000055555555",
            "000000055555555",
            "440404455555555",
            "000000055555555",
            "040404055555555",
        ),
        (
            "0202020",
            "0202020",
            "0202020",
            "0202020",
            "3233323",
            "0202020",
            "0202020",
            "0202020",
            "0202020",
            "1112111",
            "0202020",
            "0202020",
        ),
    ),
)


def _grid_rows(grid: np.ndarray) -> tuple[str, ...]:
    return tuple(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _matches_source(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    input_rows = _grid_rows(input_grid)
    output_rows = _grid_rows(output_grid)
    return any(
        input_rows == source_input and output_rows == source_output
        for source_input, source_output in _SOURCE_PAIRS
    )


def _grid_signature(grid: np.ndarray) -> tuple:
    return (
        tuple(grid.shape),
        tuple(int(value) for value in grid.ravel()),
    )


def _positive_composition(total, parts, rng, style=None):
    if parts < 1 or total < parts:
        raise ValueError("no positive composition exists")
    extra = total - parts
    if style is None:
        style = rng.choice(("random", "random", "uniform", "edge"))
    if style == "uniform" and extra % parts == 0:
        return (total // parts,) * parts
    if style == "edge" and extra:
        bands = [1] * parts
        bands[int(rng.integers(parts))] += extra
        return tuple(bands)
    if parts == 1:
        return (total,)
    cuts = np.sort(rng.choice(np.arange(1, total), size=parts - 1, replace=False))
    return tuple(int(x) for x in np.diff(np.concatenate(([0], cuts, [total]))))


def _dividers_from_bands(bands):
    cursor = bands[0]
    result = []
    for band in bands[1:]:
        result.append(cursor)
        cursor += band + 1
    return tuple(result)


def _axis_map(size, dividers, template_size):
    if template_size < 3 or template_size % 2 != 1 or len(dividers) != (template_size - 1) // 2:
        raise ValueError("divider count does not match template")
    result = []
    cursor = 0
    for index, divider in enumerate(dividers):
        divider = int(divider)
        if divider <= cursor or divider >= size - 1:
            raise ValueError("divider leaves empty band")
        result.extend([2 * index] * (divider - cursor))
        result.append(2 * index + 1)
        cursor = divider + 1
    if cursor >= size:
        raise ValueError("empty final band")
    result.extend([template_size - 1] * (size - cursor))
    return np.asarray(result, dtype=np.int64)


def _band_lengths(size, dividers):
    points = (-1,) + tuple(dividers) + (size,)
    return tuple(points[i + 1] - points[i] - 1 for i in range(len(points) - 1))


def _make_lattice(height, width, divider_rows, divider_cols, background, guide):
    grid = np.full((height, width), background, dtype=np.int8)
    rows = np.asarray(divider_rows, dtype=np.int64)
    cols = np.asarray(divider_cols, dtype=np.int64)
    grid[0, cols] = guide
    grid[-1, cols] = guide
    grid[rows, 0] = guide
    grid[rows, -1] = guide
    grid[np.ix_(rows, cols)] = guide
    return grid


def _make_template(rng, shape, background, content, mode=None):
    height, width = shape
    if mode is None:
        mode = rng.choice(("structured", "structured", "structured", "mixed", "random"))
    palette = np.asarray((background,) + tuple(content), dtype=np.int8)
    weights = np.asarray([0.52] + [0.48 / len(content)] * len(content))
    template = rng.choice(palette, size=shape, p=weights).astype(np.int8)
    if mode in ("structured", "mixed"):
        base = np.full(width, background, dtype=np.int8)
        for col in range(1, width, 2):
            base[col] = int(rng.choice(content))
        if width >= 5 and rng.random() < 0.25:
            base[int(rng.integers(0, (width + 1) // 2)) * 2] = int(rng.choice(content))
        for row in range(0, height, 2):
            template[row] = base
            if mode == "mixed" and row > 0:
                edits = int(rng.integers(1, max(2, width // 2 + 1)))
                for col in rng.choice(width, size=edits, replace=False):
                    template[row, col] = int(rng.choice(palette))
        for row in range(1, height, 2):
            motif = rng.choice(("solid", "solid", "alternating", "patchy"))
            if motif == "solid":
                template[row] = int(rng.choice(content))
            elif motif == "alternating":
                a, b = rng.choice(palette, size=2, replace=True)
                template[row] = np.asarray([a if col % 2 == 0 else b for col in range(width)])
            else:
                template[row] = rng.choice(palette, size=width, p=weights)
    else:
        # An irregular minority of templates shows that repeated rows are a motif,
        # while every cell still follows the same exact-copy rule.
        for row in range(1, height, 2):
            if rng.random() < 0.3:
                template[row] = int(rng.choice(content))

    intersections = [(r, c) for r in range(1, height, 2) for c in range(1, width, 2)]
    intersection_mode = rng.choice(("natural", "natural", "uniform", "background", "varied"))
    if intersection_mode in ("uniform", "background"):
        color = background if intersection_mode == "background" else int(rng.choice(content))
        for row, col in intersections:
            template[row, col] = color
    elif intersection_mode == "varied" and len(intersections) >= 2:
        a, b = rng.choice(palette, size=2, replace=False)
        for row, col in intersections:
            template[row, col] = int(rng.choice(palette))
        template[intersections[0]] = a
        template[intersections[1]] = b

    if not np.any(template == background):
        template[0, 0] = background
    if not np.any(template != background):
        template[0, 1] = int(rng.choice(content))
    return template


def _candidate_decode(input_grid, separator_col, lattice_left):
    height, width = input_grid.shape
    if separator_col <= 0 or separator_col >= width - 1:
        return None
    padding = int(input_grid[0, separator_col])
    if not np.all(input_grid[:, separator_col] == padding):
        return None
    left = input_grid[:, :separator_col]
    right = input_grid[:, separator_col + 1:]
    lattice, canvas = (left, right) if lattice_left else (right, left)
    if lattice.shape[1] < 3 or canvas.shape[1] < 3 or np.any(lattice == padding):
        return None
    occupied = np.flatnonzero(np.any(canvas != padding, axis=1))
    if len(occupied) < 3 or len(occupied) >= height:
        return None
    start, stop = int(occupied[0]), int(occupied[-1]) + 1
    if len(occupied) != stop - start or np.any(canvas[start:stop] == padding):
        return None
    template = canvas[start:stop].copy()
    if template.shape[0] % 2 != 1 or template.shape[1] % 2 != 1:
        return None
    template_colors = set(int(x) for x in np.unique(template))
    lattice_colors = set(int(x) for x in np.unique(lattice))
    if len(template_colors) < 2 or len(lattice_colors) != 2:
        return None
    common = template_colors & lattice_colors
    if len(common) != 1:
        return None
    background = next(iter(common))
    guide = next(iter(lattice_colors - common))
    divider_cols = tuple(int(x) for x in np.flatnonzero(lattice[0] == guide))
    divider_rows = tuple(int(x) for x in np.flatnonzero(lattice[:, 0] == guide))
    if divider_cols != tuple(int(x) for x in np.flatnonzero(lattice[-1] == guide)):
        return None
    if divider_rows != tuple(int(x) for x in np.flatnonzero(lattice[:, -1] == guide)):
        return None
    try:
        row_map = _axis_map(height, divider_rows, template.shape[0])
        col_map = _axis_map(lattice.shape[1], divider_cols, template.shape[1])
    except ValueError:
        return None
    expected = _make_lattice(height, lattice.shape[1], divider_rows, divider_cols, background, guide)
    if not np.array_equal(lattice, expected):
        return None
    return {"template": template, "lattice": lattice.copy(), "orientation": "left" if lattice_left else "right",
            "padding": padding, "guide": guide, "background": background,
            "content": frozenset(template_colors - {background}), "offset": start,
            "divider_rows": divider_rows, "divider_cols": divider_cols,
            "row_map": row_map, "col_map": col_map}


def _decode_input(input_grid):
    candidates = []
    for col in range(1, input_grid.shape[1] - 1):
        if np.all(input_grid[:, col] == input_grid[0, col]):
            for lattice_left in (True, False):
                candidate = _candidate_decode(input_grid, col, lattice_left)
                if candidate is not None:
                    candidates.append(candidate)
    if len(candidates) != 1:
        raise AssertionError(f"input has {len(candidates)} valid decodings rather than one")
    return candidates[0]


def _solve_input(input_grid):
    metadata = _decode_input(input_grid)
    output = metadata["template"][np.ix_(metadata["row_map"], metadata["col_map"])].astype(np.int8, copy=True)
    return output, metadata


def _valid_count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def _resolve_counts(seed_value, rng, num_train, num_test):
    if num_train is not None:
        num_train = _valid_count(num_train, "num_train", 2, 10)
    if num_test is not None:
        num_test = _valid_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed_value % 4
        num_train = num_test + 1 + (seed_value // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return num_train, num_test


def _shape(rng, regime, source_like=False):
    if source_like:
        return (5, 7)
    train_pool = ((3, 3), (3, 5), (5, 3), (5, 5), (3, 7), (7, 3), (5, 7), (7, 5), (7, 7), (3, 9), (9, 3))
    test_pool = train_pool + ((7, 9), (9, 5), (5, 9), (9, 9))
    pool = train_pool if regime == "source_train" else test_pool
    return pool[int(rng.integers(len(pool)))]


def _sample_axis(rng, template_size, max_output, minimum_extra, source_total=None):
    parts = (template_size + 1) // 2
    if source_total is not None:
        output_size = source_total
    else:
        maximum_extra = min(11, max_output - template_size)
        if maximum_extra < minimum_extra:
            raise ValueError("axis cannot fit")
        if rng.random() < 0.25:
            extra = minimum_extra
        elif rng.random() < 0.2:
            extra = maximum_extra
        else:
            extra = int(rng.integers(minimum_extra, maximum_extra + 1))
        output_size = template_size + extra
    bands = _positive_composition(output_size - (parts - 1), parts, rng)
    return output_size, bands


def _generate_pair(rng, regime, params, shape, orientation, offset_case=False, source_like=False, force_both=False):
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown regime")
    template_height, template_width = shape
    row_min = 2 if offset_case or force_both else 1
    col_min = 2 if force_both else 0
    output_height, row_bands = _sample_axis(rng, template_height, 20, row_min, 12 if source_like else None)
    max_width = min(19, 29 - template_width)
    output_width, col_bands = _sample_axis(rng, template_width, max_width, col_min, 7 if source_like else None)
    if source_like:
        orientation = "left"
        offset_case = False
    offset = int(rng.integers(1, output_height - template_height + 1)) if offset_case else 0
    if params["fixed_palette"]:
        content = params["fixed_content"]
    else:
        count = int(rng.integers(1, min(4, len(params["content_pool"])) + 1))
        content = tuple(int(x) for x in rng.choice(params["content_pool"], size=count, replace=False))
    template = _make_template(rng, shape, params["background"], content)
    divider_rows = _dividers_from_bands(row_bands)
    divider_cols = _dividers_from_bands(col_bands)
    lattice = _make_lattice(output_height, output_width, divider_rows, divider_cols, params["background"], params["guide"])
    row_map = _axis_map(output_height, divider_rows, template_height)
    col_map = _axis_map(output_width, divider_cols, template_width)
    output = template[np.ix_(row_map, col_map)].astype(np.int8, copy=True)
    canvas = np.full((output_height, template_width), params["padding"], dtype=np.int8)
    canvas[offset:offset + template_height] = template
    separator = np.full((output_height, 1), params["padding"], dtype=np.int8)
    if orientation == "left":
        input_grid = np.concatenate((lattice, separator, canvas), axis=1)
    else:
        input_grid = np.concatenate((canvas, separator, lattice), axis=1)
    return input_grid, output


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed_value = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed_value)
    num_train, num_test = _resolve_counts(seed_value, rng, num_train, num_test)
    colors = [int(x) for x in rng.permutation(10)]
    fixed_palette = bool(rng.integers(2))
    params = {"background": colors[0], "guide": colors[1], "padding": colors[2],
              "content_pool": tuple(colors[3:]), "fixed_palette": fixed_palette,
              "fixed_content": tuple(colors[3:3 + int(rng.integers(1, 5))])}
    offset_enabled = bool(rng.random() < 0.45)
    puzzle = {"train": [], "test": []}
    seen = set()
    orientation_order = ("left", "right") if int(rng.integers(2)) else ("right", "left")
    for index in range(num_train):
        for _ in range(80):
            pair_rng = np.random.default_rng(int(rng.bit_generator.random_raw()))
            shape = _shape(pair_rng, "source_train")
            orientation = orientation_order[index] if index < 2 else ("left" if int(pair_rng.integers(2)) else "right")
            offset_case = offset_enabled and (index == 1 or (index > 1 and pair_rng.random() < 0.5))
            pair = _generate_pair(pair_rng, "source_train", params, shape, orientation,
                                  offset_case=offset_case, force_both=index == 0)
            signature = _grid_signature(pair[0])
            if signature not in seen and not _matches_source(*pair):
                break
        else:
            raise ValueError("could not make distinct training pair")
        puzzle["train"].append({"input": pair[0], "output": pair[1]})
        seen.add(signature)

    for index in range(num_test):
        for _ in range(80):
            pair_rng = np.random.default_rng(int(rng.bit_generator.random_raw()))
            source_like = index == 0 and pair_rng.random() < 0.22
            shape = _shape(pair_rng, "target_test", source_like)
            orientation = "left" if int(pair_rng.integers(2)) else "right"
            offset_case = offset_enabled and not source_like and pair_rng.random() < 0.45
            pair = _generate_pair(pair_rng, "target_test", params, shape, orientation,
                                  offset_case=offset_case, source_like=source_like)
            signature = _grid_signature(pair[0])
            if signature not in seen and not _matches_source(*pair):
                break
        else:
            raise ValueError("could not make distinct test pair")
        puzzle["test"].append({"input": pair[0], "output": pair[1]})
        seen.add(signature)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    metadata = {"train": [], "test": []}
    seen = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            signature = _grid_signature(pair["input"])
            assert signature not in seen
            seen.add(signature)
            assert not _matches_source(pair["input"], pair["output"])
            solved, decoded = _solve_input(pair["input"])
            assert np.array_equal(solved, pair["output"])
            assert pair["input"].shape[0] == pair["output"].shape[0]
            assert pair["output"].shape == decoded["lattice"].shape
            assert set(int(x) for x in np.unique(pair["output"])) == set(int(x) for x in np.unique(decoded["template"]))
            assert decoded["guide"] not in pair["output"] and decoded["padding"] not in pair["output"]
            metadata[split].append(decoded)
    all_meta = metadata["train"] + metadata["test"]
    assert len({(m["background"], m["guide"], m["padding"]) for m in all_meta}) == 1
    assert {m["orientation"] for m in metadata["train"]} == {"left", "right"}
    assert any(m["lattice"].shape[0] > m["template"].shape[0] and
               m["lattice"].shape[1] > m["template"].shape[1] for m in metadata["train"])
    if any(m["offset"] > 0 for m in metadata["test"]):
        assert any(m["offset"] > 0 for m in metadata["train"])
    for m in all_meta:
        assert m["template"].shape[0] < m["lattice"].shape[0]
        assert m["template"].shape[1] <= m["lattice"].shape[1]
        assert all(x >= 1 for x in _band_lengths(m["lattice"].shape[0], m["divider_rows"]))
        assert all(x >= 1 for x in _band_lengths(m["lattice"].shape[1], m["divider_cols"]))


def _puzzles_equal(first, second):
    return all(np.array_equal(a[key], b[key]) for split in ("train", "test")
               for a, b in zip(first[split], second[split]) for key in ("input", "output")) and \
           all(len(first[split]) == len(second[split]) for split in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    shapes, layouts, palettes = set(), set(), set()
    horizontal_tests = vertical_tests = both_tests = offset_tests = right_tests = 0
    structured = uniform_intersections = background_intersections = 0
    varying_palette_puzzles = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _puzzles_equal(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        metas = {split: [_decode_input(p["input"]) for p in puzzle[split]] for split in ("train", "test")}
        if len({tuple(sorted(m["content"])) for split in metas for m in metas[split]}) > 1:
            varying_palette_puzzles += 1
        for split in ("train", "test"):
            for pair, m in zip(puzzle[split], metas[split]):
                template = m["template"]
                expected = template[np.ix_(m["row_map"], m["col_map"])]
                assert np.array_equal(expected, pair["output"])
                shapes.add((split, template.shape, m["lattice"].shape))
                layouts.add((split, m["orientation"], template.shape,
                             _band_lengths(m["lattice"].shape[0], m["divider_rows"]),
                             _band_lengths(m["lattice"].shape[1], m["divider_cols"])))
                palettes.add((m["background"], m["guide"], m["padding"], tuple(sorted(m["content"]))))
                if all(np.array_equal(template[0], template[r]) for r in range(2, template.shape[0], 2)):
                    structured += 1
                intersections = [int(template[r, c]) for r in range(1, template.shape[0], 2)
                                 for c in range(1, template.shape[1], 2)]
                if len(set(intersections)) == 1:
                    uniform_intersections += 1
                if all(x == m["background"] for x in intersections):
                    background_intersections += 1
                if split == "test":
                    row_stretch = m["lattice"].shape[0] > template.shape[0] + 1
                    col_stretch = m["lattice"].shape[1] > template.shape[1]
                    vertical_tests += row_stretch
                    horizontal_tests += col_stretch
                    both_tests += row_stretch and col_stretch
                    offset_tests += m["offset"] > 0
                    right_tests += m["orientation"] == "right"
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 30 and len(layouts) >= 100 and len(palettes) >= 35
    assert horizontal_tests >= 15 and vertical_tests >= 15 and both_tests >= 10
    assert offset_tests >= 5 and right_tests >= 15
    assert structured >= 20 and uniform_intersections >= 20 and background_intersections >= 5
    assert varying_palette_puzzles >= 10
    assert not _puzzles_equal(generate_puzzle(2001), generate_puzzle(2002))
    for train, test in ((2, 1), (10, 1), (10, 4), (7, 3)):
        p = generate_puzzle(3000 + train * 10 + test, train, test)
        assert len(p["train"]) == train and len(p["test"]) == test
        validate_generated_puzzle(p)
    for train, test in ((None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(4000 + (train or 0) * 10 + (test or 0), train, test)
        if train is not None:
            assert len(p["train"]) == train
        if test is not None:
            assert len(p["test"]) == test
        validate_generated_puzzle(p)
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(5000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
