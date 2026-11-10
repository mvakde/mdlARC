import numpy as np


# Source inputs as square origins in the left and right panels. The rule fixes
# their outputs, so excluding these layouts excludes exact source pairs.
_SOURCE_LAYOUTS = {
    (((0, 1), (2, 5), (4, 7)), ((2, 2), (2, 5))),
    (((0, 0), (4, 0)), ((0, 7), (1, 3), (4, 0))),
    (((0, 2), (0, 6), (3, 2), (3, 6)), ((1, 1), (4, 5))),
    (((0, 3), (3, 3)), ((4, 8),)),
    (((1, 2), (1, 6), (4, 0)), ((0, 1), (0, 5), (0, 8), (3, 1), (3, 5))),
    (((1, 1), (1, 6), (4, 3), (4, 6)), ((0, 1), (1, 7), (2, 3))),
}


def _valid_count(value, low, high):
    return (isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_))
            and low <= int(value) <= high)


def _capacity(height, width, side):
    slots = ((height + 1) // (side + 1)) * ((width + 1) // (side + 1))
    area_limit = (45 * height * width) // (100 * side * side)
    return min(12, slots, area_limit)


_GEOMETRIES = tuple(
    (h, w, side, _capacity(h, w, side))
    for side in (2, 3, 4) for h in range(4, 19) for w in range(5, 15)
    if _capacity(h, w, side) >= 2
)


def _geometry(rng, challenging=False, exclude=None, fixed_side=None,
              avoid_sides=(), avoid_shapes=(), min_capacity=2):
    if fixed_side is None:
        sides = [s for s in (2, 3, 4) if s not in avoid_sides]
        weights = np.array([{2: 0.48, 3: 0.33, 4: 0.19}[s] for s in sides])
        side = int(rng.choice(sides, p=weights / weights.sum()))
    else:
        side = fixed_side
    pool = [g for g in _GEOMETRIES if g[2] == side and g[3] >= min_capacity
            and (g[0], g[1]) not in avoid_shapes]
    if challenging and rng.random() < 0.65:
        pool = [g for g in pool if g[0] <= 6 or g[0] >= 15
                or g[1] <= 7 or g[1] >= 12]
    if exclude is not None:
        filtered = [g for g in pool if g[0] != exclude[0]
                    and g[1] != exclude[1]]
        pool = filtered or pool
    return pool[int(rng.integers(len(pool)))]


def _counts(rng, capacity, swap, excluded=()):
    kind = int(rng.integers(3))
    if kind == 0:
        high = int(rng.integers(2, min(capacity, 4) + 1))
    elif kind == 1:
        high = int(rng.integers(2, capacity + 1))
    else:
        high = int(rng.integers(max(2, capacity - 3), capacity + 1))
    low = int(rng.integers(1, high))
    pair = (high, low) if swap else (low, high)
    if pair in excluded:
        choices = [(a, b) if swap else (b, a)
                   for a in range(2, capacity + 1) for b in range(1, a)
                   if ((a, b) if swap else (b, a)) not in excluded]
        pair = choices[int(rng.integers(len(choices)))]
    return pair


def _bands(rng, length, side, count, zero_margin=None):
    slack = length - (count * side + count - 1)
    spaces = [0] * (count + 1)
    choices = [i for i in range(count + 1) if i != zero_margin]
    for _ in range(slack):
        spaces[choices[int(rng.integers(len(choices)))]] += 1
    positions = []
    cursor = spaces[0]
    for index in range(count):
        positions.append(cursor)
        cursor += side + 1 + spaces[index + 1]
    return positions


def _positions(rng, height, width, side, count, edge=None):
    nr = (height + 1) // (side + 1)
    nc = (width + 1) // (side + 1)
    rows = _bands(rng, height, side, nr)
    edge_row = int(rng.integers(nr)) if edge else -1
    slots, required = [], None
    for index, row in enumerate(rows):
        forced = None
        if index == edge_row:
            forced = 0 if edge == "left" else nc
        cols = _bands(rng, width, side, nc, forced)
        slots.extend((row, col) for col in cols)
        if index == edge_row:
            required = (row, cols[0] if edge == "left" else cols[-1])
    selected = [required] if required is not None else []
    if len(selected) == count:
        return selected
    for index in rng.permutation(len(slots)):
        point = slots[int(index)]
        if point not in selected:
            selected.append(point)
        if len(selected) == count:
            break
    return selected


def _make_pair(rng, geometry, palette, swap, excluded_counts=()):
    height, width, side, capacity = geometry
    left_count, right_count = _counts(rng, capacity, swap, excluded_counts)
    edge_panel = int(rng.integers(2)) if rng.random() < 0.42 else -1
    edge = ("left", "right")[int(rng.integers(2))]
    left = _positions(rng, height, width, side, left_count,
                      edge if edge_panel == 0 else None)
    right = _positions(rng, height, width, side, right_count,
                       edge if edge_panel == 1 else None)
    background, divider, object_color = palette
    grid = np.full((height, 2 * width + 1), background, dtype=np.int8)
    grid[:, width] = divider
    for row, col in left:
        grid[row:row + side, col:col + side] = object_color
    for row, col in right:
        col += width + 1
        grid[row:row + side, col:col + side] = object_color
    output = grid.copy()
    if swap:
        output[:, :width] = grid[:, width + 1:]
        output[:, width + 1:] = grid[:, :width]
    return {"input": grid, "output": output}


def _objects(panel, color):
    remaining = set((int(r), int(c)) for r, c in np.argwhere(panel == color))
    objects = []
    while remaining:
        start = min(remaining)
        frontier, component = [start], {start}
        remaining.remove(start)
        while frontier:
            row, col = frontier.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    point = (row + dr, col + dc)
                    if point in remaining:
                        remaining.remove(point)
                        component.add(point)
                        frontier.append(point)
        top, bottom = min(r for r, _ in component), max(r for r, _ in component)
        left, right = min(c for _, c in component), max(c for _, c in component)
        side = bottom - top + 1
        assert side >= 2 and side == right - left + 1
        assert len(component) == side * side
        objects.append((top, left, side))
    return sorted(objects)


def _signature(grid):
    return (grid.shape, grid.astype(np.uint8, copy=False).tobytes())


def _inspect_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    source, result = pair["input"], pair["output"]
    for grid in (source, result):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert grid.dtype.kind in "iu"
        assert all(1 <= v <= 30 for v in grid.shape)
        assert np.all((grid >= 0) & (grid <= 9))
    assert source.shape == result.shape
    height, full_width = source.shape
    assert full_width % 2 == 1
    width = (full_width - 1) // 2
    assert 4 <= height <= 18 and 5 <= width <= 14
    colors, frequencies = np.unique(source, return_counts=True)
    assert len(colors) == 3
    background = int(colors[int(np.argmax(frequencies))])
    divider = int(source[0, width])
    assert divider != background
    assert np.all(source[:, width] == divider)
    assert np.all(result[:, width] == divider)
    object_color = next(int(c) for c in colors if c != background and c != divider)
    for grid in (source, result):
        assert np.all((grid[:, :width] == background) |
                      (grid[:, :width] == object_color))
        assert np.all((grid[:, width + 1:] == background) |
                      (grid[:, width + 1:] == object_color))
    left = _objects(source[:, :width], object_color)
    right = _objects(source[:, width + 1:], object_color)
    assert 1 <= len(left) <= 12 and 1 <= len(right) <= 12
    assert len(left) != len(right)
    sizes = {s for _, _, s in left + right}
    assert len(sizes) == 1
    side = next(iter(sizes))
    assert side in (2, 3, 4)
    assert len(left) <= _capacity(height, width, side)
    assert len(right) <= _capacity(height, width, side)
    expected = source.copy()
    if len(left) > len(right):
        expected[:, :width] = source[:, width + 1:]
        expected[:, width + 1:] = source[:, :width]
    assert np.array_equal(result, expected)
    if (height, width, side, background, divider, object_color) == (6, 10, 2, 0, 5, 4):
        layout = (tuple((r, c) for r, c, _ in left),
                  tuple((r, c) for r, c, _ in right))
        assert layout not in _SOURCE_LAYOUTS
    return height, width, side, (background, divider, object_color), (len(left), len(right)), left, right


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    seen_inputs, seen_pairs = set(), set()
    details = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            feature = _inspect_pair(pair)
            input_key = _signature(pair["input"])
            pair_key = (input_key, _signature(pair["output"]))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key); seen_pairs.add(pair_key)
            details[split].append(feature)
    train, test = details["train"], details["test"]
    assert any(d[4][0] > d[4][1] for d in train)
    assert any(d[4][0] < d[4][1] for d in train)
    assert len({d[0] for d in train}) >= 2 and len({d[1] for d in train}) >= 2
    train_palettes = {d[3] for d in train}
    if len(train_palettes) == 1:
        assert all(d[3] in train_palettes for d in test)
    train_configs = {(d[0], d[1], d[2], d[4]) for d in train}
    assert all((d[0], d[1], d[2], d[4]) not in train_configs for d in test)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if not _valid_count(seed, 0, 2**63 - 1):
        raise ValueError("seed must be a nonnegative integer")
    if num_train is not None and not _valid_count(num_train, 2, 10):
        raise ValueError("num_train must be between 2 and 10")
    if num_test is not None and not _valid_count(num_test, 1, 4):
        raise ValueError("num_test must be between 1 and 4")
    rng = np.random.default_rng(seed)
    if num_train is None and num_test is None:
        train_count = 2 + int(seed) % 9
        test_count = 1 + (int(seed) // 3) % min(4, train_count - 1)
    elif num_train is None:
        test_count = int(num_test)
        train_count = int(rng.integers(max(2, test_count + 1), 11))
    elif num_test is None:
        train_count = int(num_train)
        test_count = int(rng.integers(1, min(4, train_count - 1) + 1))
    else:
        train_count, test_count = int(num_train), int(num_test)
    if train_count <= test_count:
        raise ValueError("num_train must exceed num_test")
    stable_palette = int(seed) % 4 == 0
    fixed_side = (2, 3, 4)[int(rng.integers(3))] if int(seed) % 8 in (0, 1) else None
    base_palette = (0, 5, 4) if int(seed) % 7 == 0 else tuple(
        int(v) for v in rng.choice(10, size=3, replace=False))
    palettes = [base_palette]
    if stable_palette:
        palettes *= train_count + test_count
    else:
        for _ in range(train_count + test_count - 1):
            palette = tuple(int(v) for v in rng.choice(10, size=3, replace=False))
            if palette == palettes[-1]:
                palette = (palette[1], palette[2], palette[0])
            palettes.append(palette)
    directions = [True, False]
    directions.extend(bool(rng.integers(2)) for _ in range(train_count - 2))
    rng.shuffle(directions)
    puzzle, used_inputs, train_configs = {"train": [], "test": []}, set(), set()
    used_geometries = set()
    first_geometry = None
    for index in range(train_count + test_count):
        training = index < train_count
        swap = directions[index] if training else bool(rng.integers(2))
        first_test = index == train_count
        avoid_sides = ({g[2] for g in train_configs}
                       if first_test and fixed_side is None
                       and len({g[2] for g in train_configs}) < 3 else ())
        avoid_shapes = ({(g[0], g[1]) for g in train_configs}
                        if first_test else ())
        excluded_counts = ({g[3] for g in train_configs}
                           if first_test else ())
        for _ in range(32):
            geometry = _geometry(rng, challenging=not training,
                                 exclude=first_geometry if index == 1 else None,
                                 fixed_side=fixed_side,
                                 avoid_sides=avoid_sides,
                                 avoid_shapes=avoid_shapes,
                                 min_capacity=6 if first_test else 2)
            pair = _make_pair(rng, geometry, palettes[index], swap,
                              excluded_counts)
            detail = _inspect_pair(pair)
            config = (detail[0], detail[1], detail[2], detail[4])
            if (_signature(pair["input"]) not in used_inputs
                    and (training or config not in train_configs)):
                break
        else:
            geometry = next(g for g in _GEOMETRIES
                            if (g[0], g[1], g[2]) not in used_geometries
                            and (fixed_side is None or g[2] == fixed_side)
                            and (not first_test or (g[3] >= 6
                                     and (g[0], g[1]) not in avoid_shapes
                                     and g[2] not in avoid_sides))
                            and (index != 1 or (g[0] != first_geometry[0]
                                                and g[1] != first_geometry[1])))
            pair = _make_pair(rng, geometry, palettes[index], swap,
                              excluded_counts)
            detail = _inspect_pair(pair)
            config = (detail[0], detail[1], detail[2], detail[4])
        used_inputs.add(_signature(pair["input"]))
        used_geometries.add((geometry[0], geometry[1], geometry[2]))
        if training:
            puzzle["train"].append(pair)
            train_configs.add(config)
            if index == 0:
                first_geometry = geometry
        else:
            puzzle["test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _same_puzzles(a, b):
    return all(len(a[split]) == len(b[split]) and
               all(np.array_equal(x[key], y[key])
                   for x, y in zip(a[split], b[split])
                   for key in ("input", "output"))
               for split in ("train", "test"))


def self_test() -> None:
    train_lengths, test_lengths = set(), set()
    heights, widths, sides, count_pairs, totals = set(), set(), set(), set(), set()
    palettes, structures, puzzle_keys = set(), set(), set()
    stable = variable = combined = 0
    side_varied = palette_varied = new_test_count = 0
    sparse = dense = long_blank = lone_right_outer = 0
    edge_outer = edge_divider = one_gap = shared_rows = 0
    modes, test_outcomes = set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_lengths.add(len(puzzle["train"]))
        test_lengths.add(len(puzzle["test"]))
        train = [_inspect_pair(p) for p in puzzle["train"]]
        test = [_inspect_pair(p) for p in puzzle["test"]]
        stable += len({d[3] for d in train}) == 1
        variable += len({d[3] for d in train}) > 1
        all_details = train + test
        side_changed = len({d[2] for d in all_details}) > 1
        palette_changed = len({d[3] for d in all_details}) > 1
        modes.add((palette_changed, side_changed))
        side_varied += side_changed
        palette_varied += palette_changed
        new_test_count += any(d[4] not in {t[4] for t in train} for d in test)
        test_outcomes.add(tuple(sorted({d[4][0] > d[4][1] for d in test})))
        train_configs = {(d[0], d[1], d[2], d[4]) for d in train}
        for d in test:
            assert (d[0], d[1], d[2], d[4]) not in train_configs
            if (d[4] not in {t[4] for t in train}
                    and d[2] not in {t[2] for t in train}
                    and (d[0], d[1]) not in {(t[0], t[1]) for t in train}):
                combined += 1
        key = []
        for pair in puzzle["train"] + puzzle["test"]:
            h, w, side, palette, counts, left, right = _inspect_pair(pair)
            heights.add(h); widths.add(w); sides.add(side)
            palettes.add(palette); count_pairs.add(counts); totals.add(sum(counts))
            density = sum(counts) * side * side / (2 * h * w)
            sparse += density < 0.12
            dense += density >= 0.30
            longest = 0
            for panel in (pair["input"][:, :w], pair["input"][:, w + 1:]):
                for blank in (np.all(panel == palette[0], axis=0),
                              np.all(panel == palette[0], axis=1)):
                    run = 0
                    for value in blank:
                        run = run + 1 if value else 0
                        longest = max(longest, run)
            long_blank += longest >= 5
            structures.add((h, w, side, tuple((r, c) for r, c, _ in left),
                            tuple((r, c) for r, c, _ in right)))
            key.append((_signature(pair["input"]), _signature(pair["output"])))
            for panel_index, objects in enumerate((left, right)):
                if any(c == (0 if panel_index == 0 else w - side)
                       for _, c, _ in objects):
                    edge_outer += 1
                if any(c == (w - side if panel_index == 0 else 0)
                       for _, c, _ in objects):
                    edge_divider += 1
                if panel_index == 1 and len(objects) == 1 and objects[0][1] == w - side:
                    lone_right_outer += 1
                if len({r for r, _, _ in objects}) < len(objects):
                    shared_rows += 1
                for i, (r1, c1, _) in enumerate(objects):
                    for r2, c2, _ in objects[i + 1:]:
                        if ((abs(c1 - c2) == side + 1 and abs(r1 - r2) < side)
                                or (abs(r1 - r2) == side + 1 and abs(c1 - c2) < side)):
                            one_gap += 1
            expected = pair["input"].copy()
            if counts[0] > counts[1]:
                expected[:, :w] = pair["input"][:, w + 1:]
                expected[:, w + 1:] = pair["input"][:, :w]
            assert np.array_equal(expected, pair["output"])
        puzzle_keys.add(tuple(key))
    assert len(train_lengths) >= 4 and len(test_lengths) >= 3
    assert min(heights) <= 5 and max(heights) >= 15
    assert min(widths) <= 7 and max(widths) >= 12
    assert sides == {2, 3, 4}
    assert len(count_pairs) >= 20 and len(totals) >= 10
    original_counts = {(3, 2), (2, 3), (4, 2), (2, 1),
                       (3, 5), (4, 3), (5, 4)}
    assert len(count_pairs - original_counts) >= 15
    assert len(palettes) >= 40 and len(structures) >= 100
    assert len(puzzle_keys) >= 40
    assert stable >= 5 and variable >= 20 and combined >= 3
    assert side_varied >= 20 and palette_varied >= 20
    assert new_test_count >= 20
    assert modes == {(False, False), (False, True),
                     (True, False), (True, True)}
    assert test_outcomes == {(False,), (True,), (False, True)}
    assert sparse >= 15 and dense >= 15 and long_blank >= 40
    assert edge_outer >= 10 and edge_divider >= 10
    assert lone_right_outer >= 5 and one_gap >= 10 and shared_rows >= 10
    for a, b in ((2, 1), (10, 1), (10, 4)):
        first = generate_puzzle(101, a, b)
        validate_generated_puzzle(first)
        assert len(first["train"]) == a and len(first["test"]) == b
        assert _same_puzzles(first, generate_puzzle(101, a, b))
    for a, b in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(107, a, b)
        validate_generated_puzzle(puzzle)
        assert a is None or len(puzzle["train"]) == a
        assert b is None or len(puzzle["test"]) == b
    for a, b in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
