"""Sparse-marker ARC puzzles: tile, then mark copied diagonal neighbors."""

import numpy as np


_SOURCE_INPUTS = (
    ((0, 0, 0), (0, 4, 0), (0, 0, 0), (0, 0, 0), (4, 0, 0)),
    ((0, 0, 6, 0), (0, 0, 0, 0), (0, 6, 0, 0)),
    ((0, 0, 0, 0), (0, 2, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
    ((0, 0, 0, 0), (0, 5, 0, 0)),
    ((0, 3, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0),
     (0, 0, 0, 3, 0), (0, 0, 0, 0, 0), (0, 3, 0, 0, 0)),
)


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    train = _checked_count("num_train", num_train, 2, 10)
    test = _checked_count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test + 1), 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    index = int(np.argmax(counts))
    assert int(counts[index]) > grid.size // 2, "background must be dominant"
    return int(colors[index])


def _construct_output(grid, background, added):
    tiled = np.tile(grid, (2, 2))
    markers = tiled != background
    diagonal = np.zeros(tiled.shape, dtype=bool)
    diagonal[1:, 1:] |= markers[:-1, :-1]
    diagonal[1:, :-1] |= markers[:-1, 1:]
    diagonal[:-1, 1:] |= markers[1:, :-1]
    diagonal[:-1, :-1] |= markers[1:, 1:]
    result = tiled.copy()
    result[diagonal & ~markers] = added
    return result


def _construct_local_tile_output(grid, background, added):
    """Plausible but incorrect: annotate each input tile before copying."""
    markers = grid != background
    diagonal = np.zeros(grid.shape, dtype=bool)
    diagonal[1:, 1:] |= markers[:-1, :-1]
    diagonal[1:, :-1] |= markers[:-1, 1:]
    diagonal[:-1, 1:] |= markers[1:, :-1]
    diagonal[:-1, :-1] |= markers[1:, 1:]
    local = grid.copy()
    local[diagonal & ~markers] = added
    return np.tile(local, (2, 2))


def _construct_wrapped_output(grid, background, added):
    """Plausible but incorrect: connect the expanded canvas's outer edges."""
    tiled = np.tile(grid, (2, 2))
    markers = tiled != background
    diagonal = np.zeros(tiled.shape, dtype=bool)
    for dr in (-1, 1):
        for dc in (-1, 1):
            diagonal |= np.roll(markers, (dr, dc), axis=(0, 1))
    wrapped = tiled.copy()
    wrapped[diagonal & ~markers] = added
    return wrapped


def _sample_dimensions(rng, section):
    # Independent axis bands permit square, tall, and wide grids in either split.
    weights = (0.43, 0.39, 0.18) if section == "train" else (0.25, 0.43, 0.32)
    bands = ((2, 5), (5, 10), (10, 15))
    dimensions = []
    for _ in range(2):
        band = bands[int(rng.choice(3, p=weights))]
        dimensions.append(int(rng.integers(band[0], band[1] + 1)))
    return tuple(dimensions)


def _random_count(rng, area):
    cap = min(18, max(1, area // 5))
    bands = ((1, 1), (2, 3), (4, 6), (7, 11), (12, 18))
    weights = (0.19, 0.30, 0.27, 0.16, 0.08)
    choice = int(rng.choice(len(bands), p=weights))
    low, high = bands[choice]
    if low > cap:
        low, high = 1, cap
    return int(rng.integers(low, min(high, cap) + 1))


def _boundary_point(rng, height, width):
    side = int(rng.integers(0, 4))
    if side == 0:
        return (0, int(rng.integers(0, width)))
    if side == 1:
        return (height - 1, int(rng.integers(0, width)))
    if side == 2:
        return (int(rng.integers(0, height)), 0)
    return (int(rng.integers(0, height)), width - 1)


def _precedence_points(rng, height, width):
    kind = int(rng.integers(0, 4))
    if kind == 0:  # diagonal adjacency already visible in the input
        row = int(rng.integers(0, height - 1))
        col = int(rng.integers(0, width - 1))
        points = [(row, col), (row + 1, col + 1)]
    elif kind == 1:  # adjacency appears only across the horizontal seam
        col = int(rng.integers(0, width - 1))
        points = [(height - 1, col), (0, col + 1)]
    elif kind == 2:  # adjacency appears only across the vertical seam
        row = int(rng.integers(0, height - 1))
        points = [(row, width - 1), (row + 1, 0)]
    else:  # both seams meet
        points = [(height - 1, width - 1), (0, 0)]
    if bool(rng.integers(0, 2)):
        points = [(height - 1 - row, col) for row, col in points]
    if bool(rng.integers(0, 2)):
        points = [(row, width - 1 - col) for row, col in points]
    return points


def _motif_points(rng, height, width):
    kind = int(rng.integers(0, 9))
    if kind == 0:  # diagonal pair
        row = int(rng.integers(0, height - 1))
        col = int(rng.integers(0, width - 1))
        points = [(row, col), (row + 1, col + 1)]
    elif kind == 1:  # orthogonal horizontal run
        row = int(rng.integers(0, height))
        col = int(rng.integers(0, width - 1))
        points = [(row, col), (row, col + 1)]
    elif kind == 2:  # orthogonal vertical run
        row = int(rng.integers(0, height - 1))
        col = int(rng.integers(0, width))
        points = [(row, col), (row + 1, col)]
    elif kind == 3 and height >= 3:  # a shared diagonal target
        row = int(rng.integers(0, height - 2))
        col = int(rng.integers(0, width))
        points = [(row, col), (row + 2, col)]
    elif kind == 4 and width >= 3:
        row = int(rng.integers(0, height))
        col = int(rng.integers(0, width - 2))
        points = [(row, col), (row, col + 2)]
    elif kind == 5 and height >= 3 and width >= 3:
        row = int(rng.integers(0, height - 2))
        col = int(rng.integers(0, width - 2))
        points = [(row + k, col + k) for k in range(3)]
    elif kind == 6:  # short block
        row = int(rng.integers(0, height - 1))
        col = int(rng.integers(0, width - 1))
        points = [(row + dr, col + dc) for dr in (0, 1) for dc in (0, 1)]
    elif kind == 7:  # corner plus another edge
        row = (0, height - 1)[int(rng.integers(0, 2))]
        col = (0, width - 1)[int(rng.integers(0, 2))]
        points = [(row, col), _boundary_point(rng, height, width)]
    else:
        points = []
    return points


def _sample_positions(rng, height, width, count, mode):
    positions = set()
    if mode == "boundary_witness":
        positions.add(_boundary_point(rng, height, width))
    elif mode == "precedence_witness":
        positions.update(_precedence_points(rng, height, width))
    elif count >= 2 and bool(rng.integers(0, 4)):
        positions.update(_motif_points(rng, height, width)[:count])

    region = int(rng.integers(0, 7)) if mode in ("generic", "progression_test") else 0
    if region == 1 and height > 2 and width > 2:
        pool = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    elif region in (2, 3, 4, 5):
        if region == 2:
            pool = [(0, c) for c in range(width)]
        elif region == 3:
            pool = [(height - 1, c) for c in range(width)]
        elif region == 4:
            pool = [(r, 0) for r in range(height)]
        else:
            pool = [(r, width - 1) for r in range(height)]
    else:
        pool = [(r, c) for r in range(height) for c in range(width)]
    if len(set(pool) | positions) < count:
        pool = [(r, c) for r in range(height) for c in range(width)]
    available = [point for point in pool if point not in positions]
    need = count - len(positions)
    if need > 0:
        chosen = rng.choice(len(available), size=need, replace=False)
        positions.update(available[int(index)] for index in np.atleast_1d(chosen))
    return sorted(positions)


def _grid_key(grid):
    return (grid.shape, tuple(int(value) for value in grid.flat))


def _geometry_key(grid):
    background = _background(grid)
    return (grid.shape, tuple(bool(value != background) for value in grid.flat))


def _is_source_input(grid):
    return any(_grid_key(grid) == _grid_key(np.asarray(rows, dtype=np.int8))
               for rows in _SOURCE_INPUTS)


def _has_boundary_marker(grid):
    background = _background(grid)
    mask = grid != background
    return bool(np.any(mask[0]) or np.any(mask[-1]) or
                np.any(mask[:, 0]) or np.any(mask[:, -1]))


def _has_precedence(grid):
    tiled = np.tile(grid != _background(grid), (2, 2))
    return bool(np.any(tiled[1:, 1:] & tiled[:-1, :-1]) or
                np.any(tiled[1:, :-1] & tiled[:-1, 1:]))


def _has_seam_mark(grid):
    height, width = grid.shape
    tiled = np.tile(grid != _background(grid), (2, 2))
    for row in range(2 * height):
        for col in range(2 * width):
            if tiled[row, col]:
                continue
            for dr in (-1, 1):
                for dc in (-1, 1):
                    rr, cc = row + dr, col + dc
                    if (0 <= rr < 2 * height and 0 <= cc < 2 * width
                            and tiled[rr, cc] and
                            (row // height != rr // height or col // width != cc // width)):
                        return True
    return False


def _geometry_features(grid):
    cells = {tuple(int(value) for value in point)
             for point in np.argwhere(grid != _background(grid))}
    height, width = grid.shape
    edges = (sum(row == 0 for row, col in cells),
             sum(row == height - 1 for row, col in cells),
             sum(col == 0 for row, col in cells),
             sum(col == width - 1 for row, col in cells))
    return {
        "edges": tuple(bool(value) for value in edges),
        "interior_only": not any(edges),
        "multiple_on_edge": max(edges) >= 2,
        "top_corner": (0, 0) in cells or (0, width - 1) in cells,
        "input_diagonal": any((row + 1, col + dc) in cells
                              for row, col in cells for dc in (-1, 1)),
        "orthogonal": any((row + dr, col + dc) in cells
                          for row, col in cells for dr, dc in ((1, 0), (0, 1))),
        "shared_target": any((row + 2, col) in cells or (row, col + 2) in cells
                             for row, col in cells),
        "block": any(all((row + dr, col + dc) in cells
                         for dr in (0, 1) for dc in (0, 1)) for row, col in cells),
    }


def _pair_from_spec(rng, section, mode, background, added, familiar, multicolor,
                    compact_train):
    for _ in range(160):
        if compact_train and section == "train":
            height, width = int(rng.integers(3, 6)), int(rng.integers(3, 6))
        elif mode == "progression_test":
            height, width = int(rng.integers(6, 16)), int(rng.integers(6, 16))
        else:
            height, width = _sample_dimensions(rng, section)
        if mode == "precedence_witness" and (height < 3 or width < 3):
            continue
        if mode == "progression_test":
            count = int(rng.integers(4, min(12, height * width // 5 + 1) + 1))
        elif compact_train and section == "train":
            upper = min(3, height * width // 5 + 1)
            lower = 2 if mode == "precedence_witness" else 1
            count = int(rng.integers(lower, upper + 1))
        elif mode == "boundary_witness":
            count = int(rng.integers(1, min(4, max(1, height * width // 5)) + 1))
        elif mode == "precedence_witness":
            count = int(rng.integers(2, min(5, max(2, height * width // 5)) + 1))
        else:
            count = _random_count(rng, height * width)
        positions = _sample_positions(rng, height, width, count, mode)
        if len(positions) != count or count * 2 >= height * width:
            continue
        eligible = [color for color in range(10) if color not in (background, added)]
        if familiar and bool(rng.integers(0, 2)):
            known = [color for color in familiar if color in eligible]
            primary = int(rng.choice(known if known else eligible))
        else:
            unknown = [color for color in eligible if color not in familiar]
            primary = int(rng.choice(unknown if unknown else eligible))
        colors = [primary]
        use_multi = multicolor and count >= 2 and (mode == "precedence_witness" or
                                                   bool(rng.integers(0, 2)))
        if use_multi:
            colors.append(int(rng.choice([color for color in eligible if color != primary])))
            if count >= 5 and bool(rng.integers(0, 3) == 0):
                colors.append(int(rng.choice([color for color in eligible if color not in colors])))
        grid = np.full((height, width), background, dtype=np.int8)
        assignments = [colors[index % len(colors)] for index in range(count)]
        rng.shuffle(assignments)
        for (row, col), color in zip(positions, assignments):
            grid[row, col] = color
        if _is_source_input(grid):
            continue
        if mode == "boundary_witness" and not (_has_boundary_marker(grid) and _has_seam_mark(grid)):
            continue
        if mode == "precedence_witness" and not _has_precedence(grid):
            continue
        output = _construct_output(grid, background, added)
        if added not in output or background not in output:
            continue
        if mode == "boundary_witness" and (
            np.array_equal(output, _construct_local_tile_output(grid, background, added))
            or np.array_equal(output, _construct_wrapped_output(grid, background, added))
        ):
            continue
        return {"input": grid, "output": output}
    raise RuntimeError("could not construct a distinct valid pair")


def _assert_cellwise_rule(input_grid, output_grid, background, added):
    height, width = input_grid.shape
    assert output_grid.shape == (2 * height, 2 * width)
    for row in range(2 * height):
        for col in range(2 * width):
            original = int(input_grid[row % height, col % width])
            expected = original
            if original == background:
                for dr in (-1, 1):
                    for dc in (-1, 1):
                        rr, cc = row + dr, col + dc
                        if (0 <= rr < 2 * height and 0 <= cc < 2 * width and
                                int(input_grid[rr % height, cc % width]) != background):
                            expected = added
            assert int(output_grid[row, col]) == expected, "incorrect tiled diagonal rule"


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    seen_inputs, seen_pairs, seen_geometry = set(), set(), set()
    added_color = None
    train_backgrounds = set()
    train_marker_colors = set()
    train_multicolor = False
    test_multicolor = False
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= value <= 30 for value in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            height, width = inp.shape
            assert 2 <= height <= 15 and 2 <= width <= 15
            background = _background(inp)
            markers = inp != background
            assert 1 <= int(np.count_nonzero(markers)) <= min(18, inp.size // 5 + 1)
            assert len(np.unique(inp[markers])) <= 3
            input_colors = set(int(c) for c in np.unique(inp))
            output_colors = set(int(c) for c in np.unique(out))
            added = output_colors - input_colors
            assert len(added) == 1, "each demonstration needs one added color"
            added = next(iter(added))
            if added_color is None:
                added_color = added
            assert added == added_color, "added color must be puzzle-wide"
            assert added_color not in input_colors
            assert output_colors == input_colors | {added_color}
            _assert_cellwise_rule(inp, out, background, added_color)
            assert not _is_source_input(inp)
            input_key = _grid_key(inp)
            pair_key = (input_key, _grid_key(out))
            geometry_key = _geometry_key(inp)
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert geometry_key not in seen_geometry, "repeated marker geometry"
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            seen_geometry.add(geometry_key)
            if section == "train":
                train_backgrounds.add(background)
                train_marker_colors.update(input_colors - {background})
                train_multicolor |= len(input_colors) > 2
            else:
                test_multicolor |= len(input_colors) > 2
    assert all(added_color not in set(int(c) for c in np.unique(pair["input"]))
               for section in ("train", "test") for pair in puzzle[section])
    assert any(_has_boundary_marker(pair["input"]) and _has_seam_mark(pair["input"])
               for pair in puzzle["train"]), "training needs a clipped seam witness"
    assert any(_has_precedence(pair["input"]) for pair in puzzle["train"]), \
        "training needs marker precedence evidence"
    assert len(train_backgrounds) >= 2, "training should identify local backgrounds"
    assert len(train_marker_colors) >= 2, "training should demonstrate color independence"
    assert not test_multicolor or train_multicolor, \
        "multiple marker colors in tests need a training demonstration"
    assert any(not np.array_equal(pair["output"], _construct_local_tile_output(
                   pair["input"], _background(pair["input"]), added_color))
               for pair in puzzle["train"]), "training must distinguish tile-local marking"
    assert any(not np.array_equal(pair["output"], _construct_wrapped_output(
                   pair["input"], _background(pair["input"]), added_color))
               for pair in puzzle["train"]), "training must distinguish outer wrapping"


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    added = int(rng.integers(0, 10))
    multicolor = bool(rng.integers(0, 3) == 0)
    progression = bool(rng.integers(0, 5) == 0)
    backgrounds = [color for color in range(10) if color != added]
    puzzle = {"train": [], "test": []}
    seen_inputs, seen_geometry = set(), set()
    familiar = set()

    # The two core demonstrations vary in order and geometry. Further slots
    # are composed freely from the same family.
    train_modes = ["boundary_witness", "precedence_witness"]
    rng.shuffle(train_modes)
    train_modes.extend("generic" for _ in range(train_count - 2))
    previous_background = None
    test_modes = ["progression_test" if progression else "generic"]
    test_modes.extend("generic" for _ in range(test_count - 1))
    for section, modes in (("train", train_modes), ("test", test_modes)):
        for mode in modes:
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            for _ in range(160):
                choices = [color for color in backgrounds if color != previous_background]
                background = int(pair_rng.choice(choices))
                pair = _pair_from_spec(pair_rng, section, mode, background, added,
                                       familiar, multicolor, progression)
                inp = pair["input"]
                input_key, geometry_key = _grid_key(inp), _geometry_key(inp)
                if input_key in seen_inputs or geometry_key in seen_geometry:
                    continue
                if section == "train" and len(puzzle["train"]) == 1 and len(familiar) < 2:
                    marker_colors = set(int(color) for color in np.unique(inp)) - {background}
                    if not marker_colors - familiar:
                        continue
                seen_inputs.add(input_key)
                seen_geometry.add(geometry_key)
                puzzle[section].append(pair)
                previous_background = background
                if section == "train":
                    familiar.update(int(color) for color in np.unique(inp) if color != background)
                break
            else:
                raise RuntimeError("could not avoid repeated input geometry")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first, second):
    return all(len(first[section]) == len(second[section]) and
               all(np.array_equal(a[role], b[role])
                   for a, b in zip(first[section], second[section])
                   for role in ("input", "output"))
               for section in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({tuple(_grid_key(pair["input"]) for section in ("train", "test")
                      for pair in p[section]) for p in puzzles}) == 42
    assert _puzzles_equal(generate_puzzle(987654, 7, 3), generate_puzzle(987654, 7, 3))
    assert not _puzzles_equal(generate_puzzle(1001, 5, 2), generate_puzzle(1002, 5, 2))

    shapes, geometries, added_colors, backgrounds = set(), set(), set(), set()
    counts, cases, diagonal_inputs = set(), set(), 0
    train_multi, test_multi, test_interior, test_edge, test_many = 0, 0, 0, 0, 0
    test_smaller, familiar_test, unseen_test = 0, 0, 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_marker_colors = set()
        for pair in puzzle["train"]:
            inp = pair["input"]
            bg = _background(inp)
            train_marker_colors.update(int(c) for c in np.unique(inp) if c != bg)
            train_multi += len(np.unique(inp)) > 2
        max_train_height = max(pair["input"].shape[0] for pair in puzzle["train"])
        max_train_width = max(pair["input"].shape[1] for pair in puzzle["train"])
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inp, out = pair["input"], pair["output"]
                bg = _background(inp)
                added = (set(int(c) for c in np.unique(out)) -
                         set(int(c) for c in np.unique(inp))).pop()
                _assert_cellwise_rule(inp, out, bg, added)
                shapes.add(inp.shape)
                geometries.add(_geometry_key(inp))
                backgrounds.add(bg)
                added_colors.add(added)
                count = int(np.count_nonzero(inp != bg))
                counts.add(count)
                cases.add((section, inp.shape[0] // 5, inp.shape[1] // 5,
                           min(count, 7), _has_boundary_marker(inp), _has_precedence(inp)))
                diagonal_inputs += _has_precedence(inp)
                if section == "test":
                    test_multi += len(np.unique(inp)) > 2
                    test_interior += not _has_boundary_marker(inp)
                    test_edge += _has_boundary_marker(inp)
                    test_many += count > 6
                    test_smaller += inp.shape[0] < max_train_height or inp.shape[1] < max_train_width
                    marker_colors = set(int(c) for c in np.unique(inp) if c != bg)
                    familiar_test += bool(marker_colors & train_marker_colors)
                    unseen_test += bool(marker_colors - train_marker_colors)
    assert len(shapes) >= 55 and len(geometries) >= 180 and len(cases) >= 85
    assert len(backgrounds) >= 8 and len(added_colors) >= 8
    assert max(counts) > 6 and 1 in counts and 2 in counts
    assert diagonal_inputs >= 50
    assert train_multi >= 10 and test_multi >= 3
    assert test_interior >= 5 and test_edge >= 15 and test_many >= 5
    assert test_smaller >= 8 and familiar_test >= 10 and unseen_test >= 5

    review = {name: 0 for name in (
        "first_precedence", "first_without_precedence", "input_diagonal",
        "tiling_only_precedence", "orthogonal", "shared_target", "block",
        "test_seam", "test_seam_free", "test_top_corner", "test_multiple_edge",
        "test_source_progression", "test_count_overlap", "test_repeated_color",
        "test_multicolor_edge_seam", "test_simple", "test_moderate",
    )}
    edge_signatures = set()
    for puzzle in puzzles:
        review["first_precedence" if _has_precedence(puzzle["train"][0]["input"])
               else "first_without_precedence"] += 1
        train_features = [_geometry_features(pair["input"]) for pair in puzzle["train"]]
        train_counts = [int(np.count_nonzero(pair["input"] != _background(pair["input"])))
                        for pair in puzzle["train"]]
        train_colors = {int(color) for pair in puzzle["train"]
                        for color in np.unique(pair["input"])
                        if color != _background(pair["input"])}
        assert len(train_colors) >= 2
        assert len({pair["input"].shape for pair in puzzle["train"] + puzzle["test"]}) >= 2
        assert len({int(np.count_nonzero(pair["input"] != _background(pair["input"])))
                    for pair in puzzle["train"] + puzzle["test"]}) >= 2
        test_color_sets = []
        for section in ("train", "test"):
            for pair in puzzle[section]:
                grid = pair["input"]
                f = _geometry_features(grid)
                review["input_diagonal"] += f["input_diagonal"]
                review["tiling_only_precedence"] += (_has_precedence(grid) and
                                                     not f["input_diagonal"])
                review["orthogonal"] += f["orthogonal"]
                review["shared_target"] += f["shared_target"]
                review["block"] += f["block"]
                edge_signatures.add(f["edges"])
                if section == "test":
                    marker_count = int(np.count_nonzero(grid != _background(grid)))
                    review["test_seam" if _has_seam_mark(grid) else "test_seam_free"] += 1
                    review["test_top_corner"] += f["top_corner"]
                    review["test_multiple_edge"] += f["multiple_on_edge"]
                    review["test_simple"] += marker_count == 1
                    review["test_moderate"] += 2 <= marker_count <= 6
                    review["test_source_progression"] += (
                        grid.shape[0] > max(x["input"].shape[0] for x in puzzle["train"])
                        and grid.shape[1] > max(x["input"].shape[1] for x in puzzle["train"])
                        and marker_count > max(train_counts))
                    review["test_count_overlap"] += marker_count <= max(train_counts)
                    colors = {int(color) for color in np.unique(grid)
                              if color != _background(grid)}
                    test_color_sets.append(colors)
                    review["test_multicolor_edge_seam"] += (
                        len(colors) > 1 and any(f["edges"]) and _has_seam_mark(grid))
        review["test_repeated_color"] += any(
            test_color_sets[left] & test_color_sets[right]
            for left in range(len(test_color_sets))
            for right in range(left + 1, len(test_color_sets)))
    assert review["first_precedence"] >= 5 and review["first_without_precedence"] >= 5
    assert review["input_diagonal"] >= 20 and review["tiling_only_precedence"] >= 20
    assert review["orthogonal"] >= 20 and review["shared_target"] >= 10
    assert review["block"] >= 2
    assert review["test_seam"] >= 20 and review["test_seam_free"] >= 5
    assert review["test_top_corner"] >= 5 and review["test_multiple_edge"] >= 10
    assert review["test_source_progression"] >= 4 and review["test_count_overlap"] >= 20
    assert review["test_repeated_color"] >= 3
    assert review["test_multicolor_edge_seam"] >= 3
    assert review["test_simple"] >= 5 and review["test_moderate"] >= 20
    assert len(edge_signatures) >= 12

    for train, test in ((2, 1), (10, 1), (10, 4), (4, 2), (7, 3)):
        puzzle = generate_puzzle(20000 + train * 10 + test, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
        validate_generated_puzzle(puzzle)
    for index, (train, test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(30000 + index, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
        validate_generated_puzzle(puzzle)
    for index, (train, test) in enumerate(((1, None), (11, None), (None, 0),
                                          (None, 5), (2, 2), (3, 4), (4, 4),
                                          (2.5, 1), (3, True))):
        try:
            generate_puzzle(40000 + index, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts must raise ValueError")
