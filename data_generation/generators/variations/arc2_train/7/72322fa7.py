import numpy as np


def _compact_literal(text: str) -> str:
    return "".join(text.split())


# Rejecting any exact source input is stronger than merely rejecting an exact
# source input/output pair.
_SOURCE_INPUT_KEYS = {
    (
        13,
        12,
        _compact_literal(
            """
000000000000
000000000000
000000000000
048400000000
000000000080
000000000000
000000000000
000000000000
000000000000
404000000000
000000080000
000000000000
000000000000
"""
        ),
    ),
    (
        13,
        22,
        _compact_literal(
            """
0000000000000000000000
0000000000000300000000
0000000000000000000000
0000008000000000000000
0000086800000000000000
0000008000000000000600
0000000000000000000000
0000000000000000000000
0000000000000000000000
0101000000000000800000
0030000006000008080000
0101000000000000800000
0000000000000000000000
"""
        ),
    ),
    (
        14,
        19,
        _compact_literal(
            """
0000000000000000000
0000000000000008280
0000000000000000000
0000000080800000000
0001000000000000000
0003000000000000000
0001000000000000000
0000000000000000000
0000000000000000000
0000000000000000000
0000000000000000000
0000020000000000000
0000000000003000000
0000000000000000000
"""
        ),
    ),
    (
        19,
        19,
        _compact_literal(
            """
0000000000000000000
0000000000003730000
0000000080000000000
0004000000000000000
0000000000000000000
0000000000000000000
0000030300000000000
0000000000000000200
0000000000000002420
0000000000000000200
0800000007000000000
0000000000000000000
0000000000000000000
0000000000000000000
0000000000020000000
0000000000202000010
0000040000020000080
0000000000000000010
0000000000000000000
"""
        ),
    ),
}


def _check_count(name: str, value: int, low: int, high: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


_AUTO_COUNT_OPTIONS = ((2, 1), (3, 1), (4, 2), (5, 3), (6, 2),
                       (7, 4), (8, 1), (9, 3), (10, 4))

# Every stencil has a unique-colored center and at least two equal-colored
# satellites. Its intact form is connected under eight-neighbor adjacency.
_STENCILS = (
    ((0, -1), (0, 1)),
    ((-1, 0), (1, 0)),
    ((-1, 0), (0, -1), (0, 1), (1, 0)),
    ((-1, -1), (-1, 1), (1, -1), (1, 1)),
    ((0, -2), (0, -1), (0, 1)),
    ((-2, 0), (-1, 0), (1, 0)),
    ((0, -1), (0, 1), (0, 2)),
    ((-1, 0), (1, 0), (2, 0)),
    ((-1, 0), (0, 1), (1, 1)),
    ((-1, -1), (0, -1), (0, 1)),
    ((-2, 0), (-1, 0), (0, 1)),
    ((0, -2), (0, -1), (1, 0)),
)


def _resolve_counts(seed, rng, num_train, num_test):
    if num_train is not None:
        num_train = _check_count("num_train", num_train, 2, 10)
    if num_test is not None:
        num_test = _check_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        return _AUTO_COUNT_OPTIONS[seed % len(_AUTO_COUNT_OPTIONS)]
    if num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _source_input_key(grid):
    return (grid.shape[0], grid.shape[1],
            "".join(str(int(v)) for v in grid.ravel()))


def _background(grid):
    counts = np.bincount(grid.ravel().astype(np.int64), minlength=10)
    return int(np.argmax(counts))


def _components(grid, background):
    height, width = grid.shape
    seen = set()
    result = []
    for row in range(height):
        for col in range(width):
            if int(grid[row, col]) == background or (row, col) in seen:
                continue
            queue = [(row, col)]
            seen.add((row, col))
            component = []
            for rr, cc in queue:
                component.append((rr, cc, int(grid[rr, cc])))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = rr + dr, cc + dc
                        if (0 <= nr < height and 0 <= nc < width and
                            (nr, nc) not in seen and int(grid[nr, nc]) != background):
                            seen.add((nr, nc))
                            queue.append((nr, nc))
            result.append(component)
    return result


def _infer_families(input_grid, background):
    """Read only the input's intact, two-color connected components."""
    families = {}
    for component in _components(input_grid, background):
        colors = {value for _, _, value in component}
        if len(colors) != 2:
            continue
        by_color = {color: [(r, c) for r, c, v in component if v == color]
                    for color in colors}
        singles = [color for color, cells in by_color.items() if len(cells) == 1]
        if len(singles) != 1:
            raise AssertionError("two-color object lacks a unique center")
        center_color = singles[0]
        satellite_color = next(iter(colors - {center_color}))
        if len(by_color[satellite_color]) < 2:
            raise AssertionError("too few satellites")
        row, col = by_color[center_color][0]
        offsets = tuple(sorted((r - row, c - col)
                               for r, c in by_color[satellite_color]))
        if any(max(abs(dr), abs(dc)) > 2 for dr, dc in offsets):
            raise AssertionError("stencil extends too far")
        family = (center_color, satellite_color, offsets)
        if center_color in families and families[center_color] != family:
            raise AssertionError("conflicting local exemplars")
        families[center_color] = family
    if not families:
        raise AssertionError("missing intact exemplar")
    used_colors = [color for center, satellite, _ in families.values()
                   for color in (center, satellite)]
    if len(used_colors) != len(set(used_colors)):
        raise AssertionError("overlapping family palettes")
    if set(used_colors) != set(int(v) for v in np.unique(input_grid)) - {background}:
        raise AssertionError("a fragment has no local exemplar")
    return tuple(sorted(families.values()))


def _predict_from_input(input_grid):
    """Reconstruct output without a test label, seed, or generator metadata."""
    bg = _background(input_grid)
    families = _infer_families(input_grid, bg)
    result = input_grid.copy()
    height, width = result.shape
    for center_color, satellite_color, offsets in families:
        center_cells = np.argwhere(input_grid == center_color)
        for row_value, col_value in center_cells:
            row, col = int(row_value), int(col_value)
            positions = [(row + dr, col + dc) for dr, dc in offsets]
            if not all(0 <= rr < height and 0 <= cc < width for rr, cc in positions):
                raise AssertionError("clipped center fragment")
            values = [int(input_grid[rr, cc]) for rr, cc in positions]
            if all(value == bg for value in values):
                for rr, cc in positions:
                    result[rr, cc] = satellite_color
            elif not all(value == satellite_color for value in values):
                raise AssertionError("partial satellite stencil")

        # A satellite-only copy has the complete exemplar stencil and a
        # background center. Reject accidental matches via the final partition.
        for row in range(height):
            for col in range(width):
                if int(input_grid[row, col]) != bg:
                    continue
                positions = [(row + dr, col + dc) for dr, dc in offsets]
                if (all(0 <= rr < height and 0 <= cc < width for rr, cc in positions)
                    and all(int(input_grid[rr, cc]) == satellite_color
                            for rr, cc in positions)):
                    result[row, col] = center_color
    return result, bg, families


def _analyze_pair(input_grid, output_grid):
    predicted, bg, families = _predict_from_input(input_grid)
    assert np.array_equal(predicted, output_grid)
    assert np.any(input_grid != output_grid)
    assert np.all((input_grid == bg) | (input_grid == output_grid))
    occupied = set()
    instances = []
    height, width = input_grid.shape
    for center_color, satellite_color, offsets in families:
        for row_value, col_value in np.argwhere(output_grid == center_color):
            row, col = int(row_value), int(col_value)
            positions = [(row, col)] + [(row + dr, col + dc) for dr, dc in offsets]
            assert all(0 <= rr < height and 0 <= cc < width for rr, cc in positions)
            assert int(output_grid[row, col]) == center_color
            assert all(int(output_grid[rr, cc]) == satellite_color
                       for rr, cc in positions[1:])
            assert all(position not in occupied for position in positions)
            occupied.update(positions)
            present_center = int(input_grid[row, col]) == center_color
            present_satellites = [int(input_grid[rr, cc]) == satellite_color
                                  for rr, cc in positions[1:]]
            assert all(present_satellites) or not any(present_satellites)
            assert present_center or all(present_satellites)
            status = ("complete" if present_center and all(present_satellites)
                      else "lone" if present_center else "satellites")
            instances.append((family_key(center_color, satellite_color, offsets),
                              row, col, status))
    all_cells = {(int(r), int(c)) for r, c in np.argwhere(output_grid != bg)}
    assert occupied == all_cells
    for family in families:
        assert any(item[0] == family_key(*family) and item[3] == "complete"
                   for item in instances)
    return {"background": bg, "families": families, "instances": instances}


def family_key(center_color, satellite_color, offsets):
    return (center_color, satellite_color, offsets)


def _choose_background(rng, preference):
    if rng.random() < preference:
        return 0
    return int(rng.integers(1, 10))


def _choose_families(rng, count, background, base_colors, palette_reuse,
                     shared_family, anchor_chance):
    families = []
    if (background not in shared_family[:2] and rng.random() < anchor_chance):
        families.append(shared_family)
    occupied = {color for center, satellite, _ in families
                for color in (center, satellite)}
    available = [color for color in range(10)
                 if color != background and color not in occupied]
    needed = 2 * (count - len(families))
    if rng.random() < palette_reuse:
        favored = [color for color in base_colors
                   if color != background and color not in occupied]
        pool = favored + [color for color in available if color not in favored]
        colors = [int(v) for v in rng.permutation(favored)]
        if len(colors) < needed:
            colors += [int(v) for v in rng.permutation(
                [color for color in pool if color not in colors])]
        colors = colors[:needed]
    else:
        colors = [int(v) for v in rng.choice(available, size=needed,
                                              replace=False)]
    # Shapes are sampled independently: two colors may share one shape.
    families.extend((colors[2 * i], colors[2 * i + 1],
                     tuple(sorted(_STENCILS[int(rng.integers(len(_STENCILS)))])))
                    for i in range(count - len(families)))
    return tuple(families)


def _dimensions(rng, regime, count, attempt):
    if attempt >= 36:
        return 30, 30
    low, high = (11, 24) if regime == "source_train" else (13, 29)
    aspect = int(rng.integers(4))
    if aspect == 0:
        height = int(rng.integers(low, high + 1))
        width = int(rng.integers(low, high + 1))
    elif aspect == 1:
        height = int(rng.integers(low, min(high, 17) + 1))
        width = int(rng.integers(max(low, 18), high + 1))
    elif aspect == 2:
        width = int(rng.integers(low, min(high, 17) + 1))
        height = int(rng.integers(max(low, 18), high + 1))
    else:
        height = int(rng.integers(low, high + 1))
        width = max(low, min(high, height + int(rng.integers(-3, 4))))
    if height * width < count * 24:
        return None
    return height, width


def _try_place(rng, height, width, specifications, background):
    input_grid = np.full((height, width), background, dtype=np.int8)
    output_grid = input_grid.copy()
    reserved = set()
    centers = []
    order = rng.permutation(len(specifications))
    for index_value in order:
        center_color, satellite_color, offsets, status = specifications[int(index_value)]
        legal = []
        for row in range(height):
            for col in range(width):
                footprint = [(row, col)] + [(row + dr, col + dc)
                                            for dr, dc in offsets]
                if not all(0 <= rr < height and 0 <= cc < width
                           for rr, cc in footprint):
                    continue
                if any(position in reserved for position in footprint):
                    continue
                legal.append((row, col, footprint))
        if not legal:
            return None
        row, col, footprint = legal[int(rng.integers(len(legal)))]
        centers.append((row, col))
        for rr, cc in footprint:
            output_grid[rr, cc] = center_color if (rr, cc) == (row, col) else satellite_color
            if ((rr, cc) == (row, col) and status != "satellites"):
                input_grid[rr, cc] = center_color
            elif (rr, cc) != (row, col) and status != "lone":
                input_grid[rr, cc] = satellite_color
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    reserved.add((rr + dr, cc + dc))
    return input_grid, output_grid


def _make_pair(rng, regime, family_count, required_status,
               base_colors, palette_reuse, zero_preference,
               shared_family, anchor_chance):
    for attempt in range(52):
        background = _choose_background(rng, zero_preference)
        families = _choose_families(rng, family_count, background,
                                    base_colors, palette_reuse,
                                    shared_family, anchor_chance)
        specifications = []
        for family_index, (center, satellite, offsets) in enumerate(families):
            ceiling = 5 if regime == "target_test" else 4
            count = int(rng.integers(2, ceiling + 1))
            statuses = ["complete"]
            if family_index == 0 and required_status is not None:
                statuses.append(required_status)
            else:
                statuses.append(("lone", "satellites")[int(rng.integers(2))])
            if count >= 4 and rng.random() < 0.23:
                statuses.append("complete")
            while len(statuses) < count:
                statuses.append(("lone", "satellites")[int(rng.integers(2))])
            specifications.extend((center, satellite, offsets, status)
                                  for status in statuses)
        size = _dimensions(rng, regime, len(specifications), attempt)
        if size is None:
            continue
        grids = _try_place(rng, *size, specifications, background)
        if grids is None:
            continue
        input_grid, output_grid = grids
        if _source_input_key(input_grid) in _SOURCE_INPUT_KEYS:
            continue
        try:
            _analyze_pair(input_grid, output_grid)
        except AssertionError:
            continue
        return {"input": input_grid, "output": output_grid}
    raise RuntimeError("could not place unambiguous motifs")


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    train_statuses = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            input_grid, output_grid = pair["input"], pair["output"]
            assert input_grid.shape == output_grid.shape
            input_key = _grid_key(input_grid)
            pair_key = (input_key, _grid_key(output_grid))
            assert input_key not in inputs and pair_key not in pairs
            inputs.add(input_key)
            pairs.add(pair_key)
            assert _source_input_key(input_grid) not in _SOURCE_INPUT_KEYS
            info = _analyze_pair(input_grid, output_grid)
            assert 1 <= len(info["families"]) <= 3
            counts = [sum(item[0] == family for item in info["instances"])
                      for family in info["families"]]
            assert all(2 <= count <= (4 if split == "train" else 5)
                       for count in counts)
            if split == "train":
                train_statuses.update(item[3] for item in info["instances"])
    # The train examples teach both complementary completion operations.
    assert {"complete", "lone", "satellites"} <= train_statuses


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
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, rng, num_train, num_test)
    # Puzzle-level preferences produce both palette-stable and recolored
    # regimes. The only semantic convention is exemplar-based completion,
    # demonstrated by both kinds of fragments in training.
    base_colors = tuple(int(v) for v in rng.choice(np.arange(1, 10),
                                                   size=6, replace=False))
    palette_reuse = float(rng.uniform(0.1, 0.75))
    stable_motif = bool(rng.random() < 0.22)
    zero_preference = 1.0 if stable_motif else float(rng.uniform(0.55, 0.95))
    shared_family = (base_colors[0], base_colors[1],
                     tuple(sorted(_STENCILS[int(rng.integers(len(_STENCILS)))])))
    anchor_chance = 1.0 if stable_motif else 0.14
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    for index in range(num_train + num_test):
        child_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        regime = "source_train" if index < num_train else "target_test"
        if regime == "source_train":
            probabilities = (0.46, 0.39, 0.15)
        else:
            probabilities = (0.20, 0.35, 0.45)
        family_count = int(child_rng.choice((1, 2, 3), p=probabilities))
        required_status = ("lone" if index == 0 else
                           "satellites" if index == 1 else None)
        for _ in range(8):
            pair = _make_pair(child_rng, regime, family_count,
                              required_status, base_colors,
                              palette_reuse, zero_preference,
                              shared_family, anchor_chance)
            key = _grid_key(pair["input"])
            if key not in used_inputs:
                used_inputs.add(key)
                puzzle["train" if regime == "source_train" else "test"].append(pair)
                break
        else:
            raise RuntimeError("could not make a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first, second):
    return all(np.array_equal(a[role], b[role])
               for split in ("train", "test")
               for a, b in zip(first[split], second[split])
               for role in ("input", "output")) and all(
                   len(first[split]) == len(second[split])
                   for split in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    signatures, dimensions, masks = set(), set(), set()
    backgrounds, family_counts, shapes, aspect_classes = set(), set(), set(), set()
    status_sets, local_count_sets = set(), set()
    two_occurrence = repeated_shapes = irregular = near_edge = 0
    same_color_different_shape = 0
    within_background_changes = test_new_shapes = reused_test_motifs = 0
    role_swaps = changed_pair_shapes = smaller_tests = combined_variation = 0
    familiar = {tuple(sorted(offsets)) for offsets in _STENCILS[:4]}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(tuple(_grid_key(p["input"]) for split in ("train", "test")
                             for p in puzzle[split]))
        family_shapes_by_colors = {}
        pair_infos = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                info = _analyze_pair(inp, out)
                pair_infos[split].append(info)
                dimensions.add(inp.shape)
                masks.add((inp.shape, (inp != info["background"]).tobytes()))
                backgrounds.add(info["background"])
                family_counts.add(len(info["families"]))
                status_sets.add(tuple(sorted({v[3] for v in info["instances"]})))
                aspect_classes.add("wide" if inp.shape[1] >= inp.shape[0] + 4
                                   else "tall" if inp.shape[0] >= inp.shape[1] + 4
                                   else "near_square")
                centers = [(v[1], v[2]) for v in info["instances"]]
                if len({r % 4 for r, _ in centers}) > 1 or len({c % 4 for _, c in centers}) > 1:
                    irregular += 1
                if any(r <= 2 or c <= 2 or r >= inp.shape[0] - 3 or c >= inp.shape[1] - 3
                       for r, c in centers):
                    near_edge += 1
                shape_list = [family[2] for family in info["families"]]
                repeated_shapes += len(shape_list) > len(set(shape_list))
                combined_variation += (len(info["families"]) >= 2 and
                                       info["background"] != 0 and
                                       any(shape not in familiar for shape in shape_list))
                for center, satellite, offsets in info["families"]:
                    shapes.add(offsets)
                    colors = frozenset((center, satellite))
                    if colors in family_shapes_by_colors and offsets not in family_shapes_by_colors[colors]:
                        same_color_different_shape += 1
                    family_shapes_by_colors.setdefault(colors, set()).add(offsets)
                    count = sum(v[0] == (center, satellite, offsets)
                                for v in info["instances"])
                    local_count_sets.add(count)
                    two_occurrence += count == 2
        all_infos = pair_infos["train"] + pair_infos["test"]
        within_background_changes += len({i["background"] for i in all_infos}) > 1
        changed_pair_shapes += len({f[2] for i in all_infos for f in i["families"]}) > 1
        trained_shapes = {f[2] for i in pair_infos["train"] for f in i["families"]}
        trained_families = {f for i in pair_infos["train"] for f in i["families"]}
        test_new_shapes += any(f[2] not in trained_shapes
                               for i in pair_infos["test"] for f in i["families"])
        reused_test_motifs += any(f in trained_families
                                  for i in pair_infos["test"] for f in i["families"])
        families_in_puzzle = [f for i in all_infos for f in i["families"]]
        role_swaps += any(a[0] == b[1] and a[1] == b[0]
                          for j, a in enumerate(families_in_puzzle)
                          for b in families_in_puzzle[j + 1:])
        train_height = max(p["input"].shape[0] for p in puzzle["train"])
        train_width = max(p["input"].shape[1] for p in puzzle["train"])
        smaller_tests += any(p["input"].shape[0] <= train_height and
                             p["input"].shape[1] <= train_width
                             for p in puzzle["test"])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) == 42 and len(dimensions) >= 35 and len(masks) >= 100
    assert len(backgrounds) >= 3 and family_counts == {1, 2, 3}
    assert len(shapes) >= 8 and aspect_classes == {"wide", "tall", "near_square"}
    assert two_occurrence >= 10 and repeated_shapes >= 1
    assert irregular >= 100 and near_edge >= 20
    assert same_color_different_shape >= 1
    assert within_background_changes >= 5 and test_new_shapes >= 10
    assert reused_test_motifs >= 5 and role_swaps >= 5
    assert changed_pair_shapes >= 20 and smaller_tests >= 5
    assert combined_variation >= 5
    assert {2, 3, 4, 5} <= local_count_sets
    assert any("lone" in statuses and "satellites" not in statuses
               for statuses in status_sets)
    assert any("satellites" in statuses and "lone" not in statuses
               for statuses in status_sets)
    assert _puzzles_equal(generate_puzzle(271828), generate_puzzle(271828))
    assert _puzzles_equal(generate_puzzle(314159, 7, 3),
                          generate_puzzle(314159, 7, 3))
    assert not _puzzles_equal(generate_puzzle(41, 6, 2), generate_puzzle(42, 6, 2))
    for train, test in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(7000 + train + test, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for index, (train, test) in enumerate(((None, 1), (None, 4),
                                           (2, None), (10, None))):
        puzzle = generate_puzzle(8000 + index, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(9000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count accepted: {kwargs}")
