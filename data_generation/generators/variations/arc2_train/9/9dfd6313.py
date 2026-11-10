import numpy as np


_SOURCE_PAIRS = {
    (("5000", "0500", "6050", "6045"),
     ("5066", "0500", "0054", "0005")),
    (("500", "350", "005"), ("530", "050", "005")),
    (("50000", "05000", "88500", "02050", "02015"),
     ("50800", "05822", "00500", "00051", "00005")),
    (("500000", "050000", "335000", "000500", "208850", "206005"),
     ("503022", "053000", "005086", "000580", "000050", "000005")),
}
_STYLES = (("main", -1), ("main", 1), ("anti", -1), ("anti", 1))


def _signature(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _rows(grid):
    return tuple("".join(str(int(x)) for x in row) for row in grid)


def _source_copy(inp, out):
    return (_rows(inp), _rows(out)) in _SOURCE_PAIRS


def _checked_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(seed, num_train, num_test):
    train = _checked_count(num_train, "num_train", 2, 10)
    test = _checked_count(num_test, "num_test", 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
    elif train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    else:
        test = 1 + seed % min(4, train - 1)
    return train, test


def _key(row, col, size, axis):
    return row - col if axis == "main" else row + col - size + 1


def _axis_cells(size, axis):
    if axis == "main":
        return [(i, i) for i in range(size)]
    return [(i, size - 1 - i) for i in range(size)]


def _triangle(size, axis, side):
    return [(r, c) for r in range(size) for c in range(size)
            if side * _key(r, c, size, axis) > 0]


def _reflect(grid, axis):
    return grid.T.copy() if axis == "main" else grid[::-1, ::-1].T.copy()


def _input_convention(grid):
    """Recover every role from the visible grid, without using its answer."""
    size = grid.shape[0]
    candidates = []
    for axis in ("main", "anti"):
        line = _axis_cells(size, axis)
        marker = int(grid[line[0]])
        if any(int(grid[p]) != marker for p in line):
            continue
        if int(np.count_nonzero(grid == marker)) != size:
            continue
        for side in (-1, 1):
            destination = _triangle(size, axis, -side)
            source = _triangle(size, axis, side)
            background = int(grid[destination[0]])
            if marker == background:
                continue
            if any(int(grid[p]) != background for p in destination):
                continue
            payload = [p for p in source if int(grid[p]) != background]
            # A fully colored wedge makes background and source side ambiguous.
            if not payload or len(payload) == len(source):
                continue
            if any(int(grid[p]) == marker for p in source):
                continue
            candidates.append((background, marker, axis, side))
    assert len(candidates) == 1
    return candidates[0]


def _payload_features(grid, convention):
    background, _, axis, side = convention
    size = grid.shape[0]
    positions = [p for p in _triangle(size, axis, side)
                 if int(grid[p]) != background]
    colors = {int(grid[p]) for p in positions}
    distances = {abs(_key(p[0], p[1], size, axis)) for p in positions}
    return positions, colors, distances


def _shape_candidates(kind, allowed, size, rng):
    candidates = []
    if kind == "bar":
        for r, c in allowed:
            for dr, dc in ((1, 0), (0, 1)):
                for length in range(2, min(5, size) + 1):
                    shape = tuple((r + i * dr, c + i * dc) for i in range(length))
                    if all(p in allowed for p in shape):
                        candidates.append(shape)
    elif kind == "bend":
        for r, c in allowed:
            for dr in (-1, 1):
                for dc in (-1, 1):
                    for a in (2, 3):
                        for b in (2, 3):
                            shape = tuple(dict.fromkeys(
                                [(r + i * dr, c) for i in range(a)] +
                                [(r, c + i * dc) for i in range(b)]))
                            if all(p in allowed for p in shape):
                                candidates.append(shape)
    else:
        for start in allowed:
            shape = {start}
            target = int(rng.integers(2, min(7, len(allowed)) + 1))
            while len(shape) < target:
                frontier = sorted({(r + dr, c + dc) for r, c in shape
                                   for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                                   if (r + dr, c + dc) in allowed and
                                   (r + dr, c + dc) not in shape})
                if not frontier:
                    break
                shape.add(frontier[int(rng.integers(len(frontier)))])
            if len(shape) >= 2:
                candidates.append(tuple(sorted(shape)))
    return candidates


def _choose_shape(kind, available, size, rng, occupied=()):
    allowed = set(available)
    candidates = _shape_candidates(kind, allowed, size, rng)
    if not candidates:
        return []
    occupied = set(occupied)
    separated = [shape for shape in candidates if all(
        abs(r - rr) + abs(c - cc) > 1
        for r, c in shape for rr, cc in occupied)]
    pool = separated if separated and rng.random() < 0.8 else candidates
    return list(pool[int(rng.integers(len(pool)))])


def _payload(rng, size, axis, side, background, marker, regime, anchor):
    cells = _triangle(size, axis, side)
    capacity = len(cells)
    palette = [c for c in range(10) if c not in (background, marker)]
    rng.shuffle(palette)
    if anchor:
        mode = "groups"
    elif regime == "target_test":
        mode = str(rng.choice(("groups", "bar", "bend", "patch", "scatter", "dense"),
                              p=(.30, .12, .11, .14, .19, .14)))
    else:
        mode = str(rng.choice(("groups", "bar", "bend", "patch", "scatter", "dense"),
                              p=(.25, .16, .14, .16, .19, .10)))
    result = {}
    if mode == "scatter":
        count = int(rng.integers(1, min(capacity - 1, max(3, size * 2)) + 1))
        positions = [cells[int(i)] for i in rng.choice(len(cells), count, replace=False)]
        color_count = int(rng.integers(1, min(5, count) + 1))
        colors = palette[:color_count]
        for color, p in zip(colors, positions):
            result[p] = color
        for p in positions[color_count:]:
            result[p] = colors[int(rng.integers(color_count))]
    elif mode == "dense":
        low = max(1, int(np.ceil(.65 * capacity)))
        count = int(rng.integers(low, capacity))
        positions = [cells[int(i)] for i in rng.choice(len(cells), count, replace=False)]
        color_count = int(rng.integers(1, min(4, count) + 1))
        centers = [positions[int(i)] for i in rng.choice(count, color_count, replace=False)]
        for p in positions:
            nearest = min(range(color_count), key=lambda i:
                          (abs(p[0] - centers[i][0]) + abs(p[1] - centers[i][1]), i))
            result[p] = palette[nearest]
    else:
        group_count = int(rng.integers(2, 5)) if mode == "groups" else 1
        if anchor:
            group_count = 2
        for group in range(group_count):
            available = [p for p in cells if p not in result]
            if len(available) <= 1:
                break
            kind = str(rng.choice(("bar", "bend", "patch"))) if mode == "groups" else mode
            shape = _choose_shape(kind, available, size, rng, result)
            if not shape:
                for fallback in ("bar", "patch"):
                    shape = _choose_shape(fallback, available, size, rng, result)
                    if shape:
                        break
            if not shape:
                break
            color = palette[group]
            for p in shape:
                result[p] = color
        if not result:
            result[cells[int(rng.integers(capacity))]] = palette[0]
        if mode == "groups" and len(set(result.values())) < 2:
            free = [p for p in cells if p not in result]
            if free:
                result[free[int(rng.integers(len(free)))]] = palette[1]
        if rng.random() < .3:
            free = [p for p in cells if p not in result]
            if free:
                p = free[int(rng.integers(len(free)))]
                result[p] = palette[min(len(set(result.values())), 4)]
    # Leave an empty cell in the source triangle so its side is identifiable.
    if len(result) >= capacity:
        del result[next(iter(result))]
    if anchor:
        distances = {abs(_key(*p, size, axis)) for p in result}
        if len(distances) < 2:
            free = [p for p in cells if p not in result and
                    abs(_key(*p, size, axis)) not in distances]
            if free:
                result[free[int(rng.integers(len(free)))]] = palette[2]
        if len(set(result.values())) < 2:
            p = sorted(result)[-1]
            result[p] = palette[1]
    return result, mode


def _make_pair(rng, size, style, roles, regime, anchor):
    axis, side = style
    background, marker = roles
    grid = np.full((size, size), background, dtype=np.int64)
    for p in _axis_cells(size, axis):
        grid[p] = marker
    payload, _ = _payload(rng, size, axis, side, background, marker, regime, anchor)
    for p, color in payload.items():
        grid[p] = color
    return {"input": grid, "output": _reflect(grid, axis)}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    # These are puzzle-level style preferences, not hidden transformation rules.
    base_style = _STYLES[int(rng.integers(4))]
    base_roles = tuple(int(x) for x in rng.choice(10, 2, replace=False))
    mixed_style = bool(rng.random() < .72)
    mixed_roles = bool(rng.random() < .76)
    source_progression = bool(rng.random() < .22)
    styles = [base_style]
    for i in range(1, train_count):
        if not mixed_style or (i > 1 and rng.random() < .25):
            styles.append(base_style)
        else:
            choices = [s for s in _STYLES if s != base_style]
            styles.append(choices[int(rng.integers(len(choices)))])
    train_sizes = []
    for i in range(train_count):
        if source_progression:
            size = 5 if i == 0 else int(rng.integers(3, 6))
        else:
            size = int(rng.integers(5, 11)) if i == 0 else int(rng.integers(3, 12))
        train_sizes.append(size)
    test_sizes = [int(rng.integers(6, 9)) if source_progression else
                  int(rng.integers(3, 13)) for _ in range(test_count)]
    puzzle = {"train": [], "test": []}
    used = set()
    plan = [("train", i, train_sizes[i], styles[i]) for i in range(train_count)]
    plan += [("test", i, test_sizes[i], styles[int(rng.integers(train_count))])
             for i in range(test_count)]
    for split, index, size, style in plan:
        for attempt in range(80):
            if split == "train" and index == 0:
                roles = base_roles
            elif not mixed_roles and rng.random() < .75:
                roles = base_roles
            else:
                pair_colors = tuple(int(x) for x in rng.choice(10, 2, replace=False))
                roles = pair_colors if pair_colors != base_roles or attempt else (
                    base_roles[1], base_roles[0])
            child = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            pair = _make_pair(child, size, style, roles,
                              "source_train" if split == "train" else "target_test",
                              split == "train" and index == 0)
            key = _signature(pair["input"])
            if key in used or _source_copy(pair["input"], pair["output"]):
                continue
            used.add(key)
            puzzle[split].append(pair)
            break
        else:
            raise RuntimeError("Could not construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    used_inputs = set()
    used_pairs = set()
    train_styles = set()
    informative = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 3 <= grid.shape[0] <= 30 and grid.shape[0] == grid.shape[1]
                assert np.issubdtype(grid.dtype, np.integer)
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            inp_key, pair_key = _signature(inp), (_signature(inp), _signature(out))
            assert inp_key not in used_inputs and pair_key not in used_pairs
            used_inputs.add(inp_key)
            used_pairs.add(pair_key)
            assert not _source_copy(inp, out)
            convention = _input_convention(inp)
            background, marker, axis, side = convention
            assert background != marker
            assert np.array_equal(out, _reflect(inp, axis))
            assert all(int(out[p]) == marker for p in _axis_cells(inp.shape[0], axis))
            assert all(int(out[p]) == background for p in _triangle(inp.shape[0], axis, side))
            positions, colors, distances = _payload_features(inp, convention)
            assert colors.isdisjoint({background, marker})
            if split == "train":
                train_styles.add((axis, side))
                informative |= len(colors) >= 2 and len(distances) >= 2
            else:
                assert (axis, side) in train_styles
    assert informative


def _equal(first, second):
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s])
        for k in ("input", "output")) for s in ("train", "test"))


def _must_reject(**kwargs):
    try:
        generate_puzzle(17, **kwargs)
    except ValueError:
        return
    raise AssertionError(f"Expected ValueError for {kwargs}")


def _observed_shapes(grid, colors):
    """Classify connected same-color payload components from grid contents."""
    kinds = set()
    for color in colors:
        remaining = set(zip(*np.nonzero(grid == color)))
        while remaining:
            component = {remaining.pop()}
            frontier = list(component)
            while frontier:
                row, col = frontier.pop()
                for neighbor in ((row - 1, col), (row + 1, col),
                                 (row, col - 1), (row, col + 1)):
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.add(neighbor)
                        frontier.append(neighbor)
            if len(component) < 2:
                continue
            rows = {row for row, _ in component}
            columns = {col for _, col in component}
            if len(rows) == 1 or len(columns) == 1:
                kinds.add("bar")
            elif any(all(row == r or col == c for row, col in component)
                     for r, c in component):
                kinds.add("bend")
            else:
                kinds.add("patch")
    return kinds


def _payload_components(positions):
    remaining = set(positions)
    groups = []
    while remaining:
        component = {remaining.pop()}
        frontier = list(component)
        while frontier:
            row, col = frontier.pop()
            for neighbor in ((row - 1, col), (row + 1, col),
                             (row, col - 1), (row, col + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    frontier.append(neighbor)
        groups.append(component)
    return groups


def self_test() -> None:
    assert _equal(generate_puzzle(7), generate_puzzle(7))
    assert _equal(generate_puzzle(8, 10, 4), generate_puzzle(8, 10, 4))
    for train, test in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(9000 + train * 10 + test, train, test)
        assert len(puzzle["train"]) == train and len(puzzle["test"]) == test
    for kwargs in ({"num_test": 1}, {"num_test": 4},
                   {"num_train": 2}, {"num_train": 10}):
        puzzle = generate_puzzle(555, **kwargs)
        for name, value in kwargs.items():
            assert len(puzzle["train" if name == "num_train" else "test"]) == value
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        _must_reject(**kwargs)
    train_counts, test_counts, puzzle_keys, layouts = set(), set(), set(), set()
    style_pairs, role_pairs, sizes, palette_sizes = set(), set(), set(), set()
    mixed_style = mixed_background = mixed_marker = overlap = large_train = 0
    sparse_test = mono_test = dense_train = colored_groups = 0
    source_progression = small_test = two_distances = border_payload = 0
    changed_axis = changed_side = combined = 0
    observed_shapes = set()
    separated_payload = separated_colored_shapes = 0
    for seed in range(1000, 1042):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_keys.add(tuple(_signature(p["input"]) for split in ("train", "test")
                              for p in puzzle[split]))
        train_sizes = {p["input"].shape[0] for p in puzzle["train"]}
        test_sizes = {p["input"].shape[0] for p in puzzle["test"]}
        overlap += bool(train_sizes & test_sizes)
        large_train += max(train_sizes) >= 7
        source_progression += max(train_sizes) <= 5 and min(test_sizes) >= 6
        local_styles, local_backgrounds, local_markers = set(), set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                convention = _input_convention(inp)
                background, marker, axis, side = convention
                positions, colors, distances = _payload_features(inp, convention)
                assert np.array_equal(out, _reflect(inp, axis))
                assert positions and len(positions) < len(_triangle(inp.shape[0], axis, side))
                assert not _source_copy(inp, out)
                style_pairs.add((axis, side))
                role_pairs.add((background, marker))
                sizes.add(inp.shape[0])
                palette_sizes.add(len(colors))
                observed_shapes.update(_observed_shapes(inp, colors))
                local_styles.add((axis, side))
                local_backgrounds.add(background)
                local_markers.add(marker)
                layouts.add((inp.shape[0], axis, side, tuple(positions)))
                components = _payload_components(positions)
                separated_payload += len(components) >= 2
                separated_colored_shapes += sum(
                    len(component) >= 2 and
                    len({int(inp[p]) for p in component}) == 1
                    for component in components) >= 2
                two_distances += len(distances) >= 2
                border_payload += any(r in (0, inp.shape[0] - 1) or
                                      c in (0, inp.shape[0] - 1) for r, c in positions)
                if split == "test":
                    sparse_test += len(positions) <= 3
                    mono_test += len(colors) == 1
                    small_test += inp.shape[0] <= 5
                else:
                    dense_train += len(positions) >= .60 * len(_triangle(inp.shape[0], axis, side))
                    colored_groups += any(sum(int(inp[p] == color) for p in positions) >= 2
                                          for color in colors)
        mixed_style += len(local_styles) > 1
        mixed_background += len(local_backgrounds) > 1
        mixed_marker += len(local_markers) > 1
        changed_axis += len({style[0] for style in local_styles}) > 1
        changed_side += len({style[1] for style in local_styles}) > 1
        combined += (len(local_styles) > 1 and len(local_backgrounds) > 1
                     and len(local_markers) > 1 and bool(train_sizes & test_sizes))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzle_keys) == 42 and len(layouts) >= 100
    assert style_pairs == set(_STYLES) and len(role_pairs) >= 25
    assert set(range(3, 12)).issubset(sizes) and max(sizes) >= 12
    assert 1 in palette_sizes and max(palette_sizes) >= 4
    assert mixed_style >= 15 and mixed_background >= 25 and mixed_marker >= 25
    assert changed_axis >= 10 and changed_side >= 10 and combined >= 5
    assert overlap >= 10 and large_train >= 25
    assert source_progression >= 3 and small_test >= 5
    assert sparse_test >= 5 and mono_test >= 5 and dense_train >= 5
    assert two_distances >= 100 and border_payload >= 100
    assert observed_shapes == {"bar", "bend", "patch"}
    assert separated_payload >= 30 and separated_colored_shapes >= 10
    assert colored_groups >= 50
