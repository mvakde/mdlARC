import numpy as np


_EDGES = ("bottom", "top", "left", "right")
_SOURCE_PAIRS = (
    (("77777777", "77777777", "77777777", "77777777", "77777777", "79777777", "99977222", "88888888"),
     ("77777777", "79777777", "99977777", "77777222", "77777777", "77777777", "77777777", "88888888")),
    (("77777777", "77777777", "77777777", "77777777", "77797777", "72797773", "72797173", "88888888"),
     ("77777777", "77797777", "77797777", "72797773", "72777773", "77777177", "77777777", "88888888")),
    (("77777777", "77777777", "77777777", "77777777", "77777774", "11777374", "11173374", "88888888"),
     ("11777777", "11177774", "77777374", "77773374", "77777777", "77777777", "77777777", "88888888")),
    (("77777777", "77777777", "77777777", "77777777", "77777777", "77777777", "55555576", "88888888"),
     ("55555577", "77777777", "77777777", "77777777", "77777777", "77777776", "77777777", "88888888")),
)


def _validate_count(value, name, low, high):
    if value is not None:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
            raise ValueError(f"{name} must be an integer from {low} through {high}")


def _choose_counts(rng, num_train, num_test):
    _validate_count(num_train, "num_train", 2, 10)
    _validate_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_test = int(num_test)
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_train = int(num_train)
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    else:
        num_train, num_test = int(num_train), int(num_test)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _normalize(cells):
    cells = tuple(cells)
    bottom = max(r for r, _ in cells)
    left = min(c for _, c in cells)
    return tuple(sorted((r - bottom, c - left) for r, c in cells))


def _shape(rng, area, kind=None):
    if area == 1:
        return ((0, 0),)
    if kind is None:
        kind = ("horizontal", "vertical", "hook", "step", "grow")[int(rng.integers(0, 5))]
    if kind == "horizontal":
        cells = [(0, c) for c in range(area)]
    elif kind == "vertical":
        cells = [(r, 0) for r in range(area)]
    elif kind == "hook":
        if area == 2:
            cells = [(0, 0), (-1, 0)]
        else:
            length = int(rng.integers(2, area))
            cells = [(0, c) for c in range(length)]
            end = 0 if int(rng.integers(0, 2)) == 0 else length - 1
            cells += [(-r, end) for r in range(1, area - length + 1)]
    elif kind == "step":
        if area == 2:
            cells = [(0, 0), (0, 1)]
        else:
            lower = (area + 1) // 2
            upper = area - lower
            cells = [(0, c) for c in range(lower)]
            offset = int(rng.integers(0, lower - upper + 1))
            cells += [(-1, offset + c) for c in range(upper)]
    else:
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = {
                (r + dr, c + dc)
                for r, c in cells
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if (r + dr, c + dc) not in cells
            }
            choices = sorted(frontier)
            cells.add(choices[int(rng.integers(0, len(choices)))])
    shape = _normalize(cells)
    if int(rng.integers(0, 2)):
        right = max(c for _, c in shape)
        shape = _normalize((r, right - c) for r, c in shape)
    return shape


def _orient(grid, edge):
    if edge == "bottom":
        return grid.copy()
    if edge == "top":
        return np.flipud(grid).copy()
    if edge == "left":
        return np.rot90(grid, 3).copy()
    if edge == "right":
        return np.rot90(grid, 1).copy()
    raise AssertionError("invalid edge")


def _canonical(grid, edge):
    if edge == "bottom":
        return grid.copy()
    if edge == "top":
        return np.flipud(grid).copy()
    if edge == "left":
        return np.rot90(grid, 1).copy()
    if edge == "right":
        return np.rot90(grid, 3).copy()
    raise AssertionError("invalid edge")


def _roles_for_pairs(rng, palette, count_train, count_test):
    total = count_train + count_test
    stable_edge = int(rng.integers(0, 4)) == 0
    stable_palette = int(rng.integers(0, 3)) == 0
    base_edge = _EDGES[int(rng.integers(0, 4))]
    base_roles = (palette[0], palette[1])
    edges = [base_edge] * total
    roles = [base_roles] * total
    if not stable_edge:
        edges[1] = _EDGES[(_EDGES.index(base_edge) + int(rng.integers(1, 4))) % 4]
        for i in range(2, total):
            edges[i] = _EDGES[int(rng.integers(0, 4))]
    if not stable_palette:
        choices = palette[:7]
        for i in range(1, total):
            available = [(a, b) for a in choices for b in choices if a != b and (a, b) != base_roles]
            roles[i] = available[int(rng.integers(0, len(available)))]
        if roles[1] == roles[0]:
            raise AssertionError("role variation failed")
    # Stable training roles imply stable test roles. Mixed training explicitly
    # demonstrates that the line and dominant background can change colors.
    return list(zip(edges, roles))


def _choose_colors(rng, count, pool, background, support, anchor=None):
    available = [c for c in pool if c not in (background, support)]
    assert available
    colors = []
    if anchor is not None:
        assert anchor in available
        colors.append(anchor)
    while len(colors) < count:
        unused = [c for c in available if c not in colors]
        if unused and (len(colors) < 2 or int(rng.integers(0, 5)) != 0):
            colors.append(unused[int(rng.integers(0, len(unused)))])
        else:
            colors.append(available[int(rng.integers(0, len(available)))])
    return colors


def _records_for_train(rng, index, equal_index, same_area, other_areas, anchor, pool, roles):
    background, support = roles
    if index == equal_index:
        area = same_area
        kinds = (("horizontal", "vertical"), ("horizontal", "hook"), ("step", "vertical"))
        first_kind, second_kind = kinds[int(rng.integers(0, len(kinds)))]
        shapes = [_shape(rng, area, first_kind), _shape(rng, area, second_kind)]
        if shapes[0] == shapes[1]:
            shapes[1] = _shape(rng, area, "vertical" if first_kind != "vertical" else "horizontal")
        if int(rng.integers(0, 3)) == 0:
            third = _shape(rng, area, "grow")
            if third not in shapes:
                shapes.append(third)
        colors = _choose_colors(rng, len(shapes), pool, background, support, anchor)
        records = list(zip([area] * len(shapes), shapes, colors))
    elif index < 2:
        areas = list(other_areas)
        if int(rng.integers(0, 2)):
            areas.append(int(rng.integers(1, 6)))
        shapes = [_shape(rng, area) for area in areas]
        colors = _choose_colors(rng, len(areas), pool, background, support, anchor)
        records = list(zip(areas, shapes, colors))
    else:
        count = int(rng.integers(1, 5))
        areas = [int(x) for x in rng.integers(1, 6, size=count)]
        if count >= 2 and int(rng.integers(0, 4)) == 0:
            areas[1] = areas[0]
        shapes = [_shape(rng, area) for area in areas]
        colors = _choose_colors(rng, count, pool, background, support)
        records = list(zip(areas, shapes, colors))
    return [records[int(i)] for i in rng.permutation(len(records))]


def _records_for_test(rng, index, source_like, train_pool, novel_pool, palette_mode, roles):
    background, support = roles
    if source_like:
        areas = [1, 6]
        shapes = [_shape(rng, 1), _shape(rng, 6, "horizontal")]
    else:
        count = int(rng.integers(1, 5))
        areas = [int(x) for x in rng.integers(1, 7, size=count)]
        if index == 0:
            areas[int(rng.integers(0, count))] = 6
        shapes = [_shape(rng, area) for area in areas]
    if palette_mode == "novel":
        pool = novel_pool
    elif palette_mode == "shared":
        pool = train_pool
    else:
        pool = train_pool + novel_pool
    colors = _choose_colors(rng, len(areas), pool, background, support)
    records = list(zip(areas, shapes, colors))
    return [records[int(i)] for i in rng.permutation(len(records))]


def _place(rng, records, source_like):
    widths = [max(c for _, c in shape) + 1 for _, shape, _ in records]
    if source_like:
        blanks = [0, 1, 0]
        width, height = 8, 8
    else:
        min_width = max(8, sum(widths) + len(records) - 1)
        assert min_width <= 30
        extra = int(rng.integers(0, min(10, 30 - min_width) + 1))
        width = min_width + extra
        blanks = [0] + [1] * (len(records) - 1) + [0]
        slack = width - (sum(widths) + len(records) - 1)
        if slack:
            allocation = rng.multinomial(slack, [1 / len(blanks)] * len(blanks))
            blanks = [b + int(a) for b, a in zip(blanks, allocation)]
        minimum_height = max(8, max(area + (1 - min(r for r, _ in shape)) + 1 for area, shape, _ in records))
        height = minimum_height + int(rng.integers(0, min(9, 30 - minimum_height) + 1))
    starts = []
    cursor = blanks[0]
    for i, object_width in enumerate(widths):
        starts.append(cursor)
        cursor += object_width
        if i + 1 < len(records):
            cursor += blanks[i + 1]
    assert cursor + blanks[-1] == width
    return starts, height, width


def _make_pair(rng, records, edge, roles, source_like=False):
    background, support = roles
    starts, height, width = _place(rng, records, source_like)
    input_grid = np.full((height, width), background, dtype=np.int8)
    input_grid[-1, :] = support
    objects = []
    for (area, shape, color), start in zip(records, starts):
        cells = [(height - 2 + r, start + c) for r, c in shape]
        assert len(cells) == area
        for r, c in cells:
            assert 0 <= r < height - 1 and input_grid[r, c] == background
            input_grid[r, c] = color
        objects.append((cells, area, color))
    output_grid = input_grid.copy()
    for cells, _, _ in objects:
        for r, c in cells:
            output_grid[r, c] = background
    for cells, area, color in objects:
        for r, c in cells:
            assert r - area >= 0 and input_grid[r - area, c] == background
            assert output_grid[r - area, c] == background
            output_grid[r - area, c] = color
    return {"input": _orient(input_grid, edge), "output": _orient(output_grid, edge)}


def _signature(grid):
    return (tuple(grid.shape), grid.tobytes())


def _is_source(pair):
    x, y = pair["input"], pair["output"]
    if x.shape != (8, 8) or y.shape != (8, 8):
        return False
    rows = lambda grid: tuple("".join(str(int(v)) for v in row) for row in grid)
    return (rows(x), rows(y)) in _SOURCE_PAIRS


def _detect_roles(grid):
    candidates = []
    for edge in _EDGES:
        if edge == "bottom":
            line = grid[-1, :]
        elif edge == "top":
            line = grid[0, :]
        elif edge == "left":
            line = grid[:, 0]
        else:
            line = grid[:, -1]
        support = int(line[0])
        if np.all(line == support) and np.count_nonzero(grid == support) == line.size:
            candidates.append((edge, support))
    assert len(candidates) == 1
    edge, support = candidates[0]
    values, counts = np.unique(grid[grid != support], return_counts=True)
    assert len(values) >= 2
    background = int(values[int(np.argmax(counts))])
    return edge, background, support


def _components(grid, background, support):
    height, width = grid.shape
    seen = set()
    objects = []
    for row in range(height):
        for column in range(width):
            if (row, column) in seen or int(grid[row, column]) in (background, support):
                continue
            color = int(grid[row, column])
            cells = set()
            stack = [(row, column)]
            seen.add((row, column))
            while stack:
                r, c = stack.pop()
                cells.add((r, c))
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in seen and int(grid[rr, cc]) == color:
                        seen.add((rr, cc))
                        stack.append((rr, cc))
            objects.append((color, tuple(sorted(cells))))
    return objects


def _predict(input_grid, factor=1):
    edge, background, support = _detect_roles(input_grid)
    canonical = _canonical(input_grid, edge)
    height, width = canonical.shape
    assert width >= 8 and height >= 8
    assert np.all(canonical[-1, :] == support)
    assert np.count_nonzero(canonical == support) == width
    objects = _components(canonical, background, support)
    assert 1 <= len(objects) <= 4
    occupied = set()
    for _, cells in objects:
        assert max(r for r, _ in cells) == height - 2
        occupied.update(cells)
    expected = canonical.copy()
    for _, cells in objects:
        for r, c in cells:
            expected[r, c] = background
    destination = set()
    for color, cells in objects:
        distance = factor * len(cells)
        for r, c in cells:
            rr = r - distance
            assert 0 <= rr < height - 1
            assert (rr, c) not in occupied and (rr, c) not in destination
            destination.add((rr, c))
            expected[rr, c] = color
    return _orient(expected, edge), objects, (edge, background, support)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            x, y = pair["input"], pair["output"]
            for grid in (x, y):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert x.shape == y.shape
            assert _signature(x) not in seen
            seen.add(_signature(x))
            assert not _is_source(pair)
            predicted, objects, roles = _predict(x)
            assert np.array_equal(predicted, y) and not np.array_equal(x, y)
            assert np.array_equal(_canonical(y, roles[0])[-1], np.full(_canonical(y, roles[0]).shape[1], roles[2]))
            assert all(1 <= len(cells) <= (5 if split == "train" else 6) for _, cells in objects)
            infos[split].append((objects, roles, pair))
    train_edges = {roles[0] for _, roles, _ in infos["train"]}
    train_role_colors = {roles[1:] for _, roles, _ in infos["train"]}
    if len(train_edges) == 1:
        assert all(roles[0] in train_edges for _, roles, _ in infos["test"])
    if len(train_role_colors) == 1:
        assert all(roles[1:] in train_role_colors for _, roles, _ in infos["test"])
    train_areas = {len(cells) for objects, _, _ in infos["train"] for _, cells in objects}
    assert len(train_areas) >= 3
    assert any(len(objects) >= 2 and len({len(cells) for _, cells in objects}) > 1 for objects, _, _ in infos["train"])
    assert any(len(objects) >= 2 and len({len(cells) for _, cells in objects}) == 1 and len({tuple((r - min(rr for rr, _ in cells), c - min(cc for _, cc in cells)) for r, c in cells) for _, cells in objects}) > 1 for objects, _, _ in infos["train"])
    color_areas = {}
    for objects, _, _ in infos["train"]:
        for color, cells in objects:
            color_areas.setdefault(color, set()).add(len(cells))
    assert any(len(areas) > 1 for areas in color_areas.values())
    assert any(len(cells) == 6 for objects, _, _ in infos["test"] for _, cells in objects)
    # Infer the only area multiplier fitting the demonstrations, then solve
    # tests using training pairs and test inputs alone.
    fits = []
    for factor in range(1, 7):
        try:
            if all(np.array_equal(_predict(pair["input"], factor)[0], pair["output"]) for _, _, pair in infos["train"]):
                fits.append(factor)
        except AssertionError:
            pass
    assert fits == [1]
    for _, _, pair in infos["test"]:
        assert np.array_equal(_predict(pair["input"], fits[0])[0], pair["output"])


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    num_train, num_test = _choose_counts(rng, num_train, num_test)
    palette = [int(v) for v in rng.permutation(10)]
    conventions = _roles_for_pairs(rng, palette, num_train, num_test)
    train_pool, novel_pool = palette[2:7], palette[7:10]
    available_anchor = [c for c in train_pool if c not in conventions[0][1] and c not in conventions[1][1]]
    anchor = available_anchor[int(rng.integers(0, len(available_anchor)))]
    same_area = int(rng.integers(2, 6))
    equal_index = int(rng.integers(0, 2))
    other_areas = [a for a in range(1, 6) if a != same_area]
    selected = rng.choice(other_areas, size=2, replace=False)
    other_areas = [int(v) for v in selected]
    palette_mode = ("shared", "novel", "mixed")[int(rng.integers(0, 3))]
    source_like = int(rng.integers(0, 5)) == 0
    puzzle = {"train": [], "test": []}
    seen = set()
    for global_index in range(num_train + num_test):
        split = "train" if global_index < num_train else "test"
        index = global_index if split == "train" else global_index - num_train
        edge, roles = conventions[global_index]
        for attempt in range(80):
            if split == "train":
                records = _records_for_train(rng, index, equal_index, same_area, other_areas, anchor, train_pool, roles)
                special = False
            else:
                special = source_like and index == 0
                records = _records_for_test(rng, index, special, train_pool, novel_pool, palette_mode, roles)
            pair = _make_pair(rng, records, edge, roles, special)
            if _signature(pair["input"]) not in seen and not _is_source(pair):
                seen.add(_signature(pair["input"]))
                puzzle[split].append(pair)
                break
        else:
            raise ValueError("could not construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(a, b):
    return all(len(a[split]) == len(b[split]) and all(np.array_equal(p["input"], q["input"]) and np.array_equal(p["output"], q["output"]) for p, q in zip(a[split], b[split])) for split in ("train", "test"))


def self_test() -> None:
    assert _puzzles_equal(generate_puzzle(731, 7, 3), generate_puzzle(731, 7, 3))
    assert not _puzzles_equal(generate_puzzle(731, 7, 3), generate_puzzle(732, 7, 3))
    for train, test in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(1000 + train * 10 + test, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for train, test in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(1200 + (train or 0) * 10 + (test or 0), train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, None), (None, 2.0)):
        try:
            generate_puzzle(1, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts = {len(p["train"]) for p in puzzles}
    test_counts = {len(p["test"]) for p in puzzles}
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    signatures = set()
    dimensions = set()
    edges = set()
    object_counts = set()
    six_shapes = set()
    six_kinds = set()
    gaps = set()
    train_area_sets = set()
    first_train_patterns = set()
    checks = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_roles = [_detect_roles(pair["input"]) for pair in puzzle["train"]]
        test_roles = [_detect_roles(pair["input"]) for pair in puzzle["test"]]
        if len({r[0] for r in train_roles}) > 1:
            checks.add("mixed_train_edges")
        if len({r[1:] for r in train_roles}) > 1:
            checks.add("mixed_train_roles")
        if len({r[0] for r in train_roles}) == 1:
            checks.add("stable_train_edges")
        if len({r[1:] for r in train_roles}) == 1:
            checks.add("stable_train_roles")
        if any(r[0] != train_roles[0][0] for r in test_roles):
            checks.add("changed_side_test")
        if any(r[0] == train_roles[0][0] for r in test_roles):
            checks.add("same_side_test")
        train_colors = {color for pair in puzzle["train"] for color, _ in _predict(pair["input"])[1]}
        areas_in_training = set()
        for split in ("train", "test"):
            for pair_index, pair in enumerate(puzzle[split]):
                signatures.add(_signature(pair["input"]))
                dimensions.add(pair["input"].shape)
                predicted, objects, roles = _predict(pair["input"])
                assert np.array_equal(predicted, pair["output"])
                edges.add(roles[0])
                object_counts.add(len(objects))
                canonical = _canonical(pair["input"], roles[0])
                areas = [len(cells) for _, cells in objects]
                if split == "train":
                    areas_in_training.update(areas)
                    if pair_index == 0:
                        first_train_patterns.add("equal" if len(set(areas)) == 1 else "unequal")
                if split == "test":
                    if any(color in train_colors for color, _ in objects):
                        checks.add("reused_test_color")
                    if any(color not in train_colors for color, _ in objects):
                        checks.add("novel_test_color")
                    if pair["input"].shape == (8, 8):
                        checks.add("eight_by_eight_test")
                    if len(objects) >= 3 and 6 in areas and any(area < 6 for area in areas):
                        checks.add("combined_geometry")
                bounds = sorted((min(c for _, c in cells), max(c for _, c in cells)) for _, cells in objects)
                gaps.update(right[0] - left[1] - 1 for left, right in zip(bounds, bounds[1:]))
                if any(left == 0 or right == canonical.shape[1] - 1 for left, right in bounds):
                    checks.add("lateral_edge")
                if any(left == 1 or right == canonical.shape[1] - 2 for left, right in bounds):
                    checks.add("one_cell_margin")
                if len({color for color, _ in objects}) < len(objects):
                    checks.add("same_color_objects")
                if split == "test":
                    for color, cells in objects:
                        if len(cells) == 6:
                            shape = _normalize(cells)
                            six_shapes.add(shape)
                            rows = {r for r, _ in cells}
                            columns = {c for _, c in cells}
                            six_kinds.add("parallel" if len(rows) == 1 else "perpendicular" if len(columns) == 1 else "bent")
                            if min(rows) - 6 > 0:
                                checks.add("interior_six")
                            if min(rows) - 6 == 0:
                                checks.add("edge_six")
                    if sorted(areas) == [1, 6] and any(len(cells) == 6 and len({r for r, _ in cells}) == 1 for _, cells in objects):
                        checks.add("source_like_test")
        train_area_sets.add(tuple(sorted(areas_in_training)))
    assert len(signatures) >= 150 and len(dimensions) >= 20
    assert edges == set(_EDGES) and object_counts == {1, 2, 3, 4}
    assert len(six_shapes) >= 6
    assert six_kinds == {"parallel", "perpendicular", "bent"}
    assert {1, 2, 3}.issubset(gaps) and len(train_area_sets) >= 4
    assert first_train_patterns == {"equal", "unequal"}
    assert checks == {"mixed_train_edges", "mixed_train_roles", "stable_train_edges", "stable_train_roles", "changed_side_test", "same_side_test", "reused_test_color", "novel_test_color", "eight_by_eight_test", "combined_geometry", "same_color_objects", "lateral_edge", "one_cell_margin", "interior_six", "edge_six", "source_like_test"}
