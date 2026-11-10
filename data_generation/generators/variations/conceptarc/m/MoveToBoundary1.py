"""Whole-pattern translation toward a full-length barrier."""

import numpy as np


_SOURCE_INPUTS = {
    ("0000020", "0000020", "0110020", "0110020", "0000020", "0000020", "0000020"),
    ("000020", "010020", "011020", "010020", "000020", "000020"),
    ("00000020", "00000020", "00011020", "00110020", "00011020", "00000020", "00000020", "00000020"),
    ("000000", "011111", "010001", "000000", "222222", "000000"),
    ("000020", "000020", "111020", "000020", "000020", "000020"),
    ("000111020", "000110020", "000100020", "000110020", "000111020", "000110020", "000100020", "000110020", "000111020"),
    ("000000000", "000000000", "000000000", "000000000", "010001000", "101010101", "000100010", "000000000", "222222222"),
}
_DIRECTIONS = ("right", "left", "down", "up")


def _resolve_counts(rng, num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high
        ):
            raise ValueError("Invalid pair count")
    legal = [
        (a, b) for a in range(2, 11) for b in range(1, 5)
        if a > b and (num_train is None or a == num_train)
        and (num_test is None or b == num_test)
    ]
    if not legal:
        raise ValueError("Training count must exceed test count")
    return legal[int(rng.integers(len(legal)))]


def _palette(rng):
    if rng.random() < 0.16:
        return (0, 1, 2)
    return tuple(int(x) for x in rng.choice(10, size=3, replace=False))


def _grow(rng, depth, breadth, target):
    """A connected polyomino anchored at depth zero."""
    cells = {(0, int(rng.integers(breadth)))}
    while len(cells) < target:
        frontier = set()
        for d, o in cells:
            for q in ((d - 1, o), (d + 1, o), (d, o - 1), (d, o + 1)):
                if 0 <= q[0] < depth and 0 <= q[1] < breadth and q not in cells:
                    frontier.add(q)
        if not frontier:
            break
        choices = sorted(frontier)
        cells.add(choices[int(rng.integers(len(choices)))])
    return cells


def _disconnected(rng, depth_limit, orth_limit, aligned):
    count = int(rng.integers(2, min(4, (orth_limit + 1) // 2) + 1))
    total_width = int(rng.integers(count, min(orth_limit - count + 1, 3 * count + 2) + 1))
    widths = [1] * count
    for _ in range(total_width - count):
        widths[int(rng.integers(count))] += 1
    surplus = orth_limit - total_width - (count - 1)
    margins = [0] * (count + 1)
    for _ in range(surplus):
        margins[int(rng.integers(count + 1))] += 1
    start = margins[0]
    cells = set()
    first_size = None
    for index, width in enumerate(widths):
        if index == 0 or aligned:
            depth_start = 0
        elif index == 1:
            depth_start = int(rng.integers(1, depth_limit - 1))
        else:
            depth_start = int(rng.integers(0, depth_limit - 1))
        capacity = min(6, (depth_limit - depth_start) * width)
        if index == 0:
            size = int(rng.integers(1, min(capacity, 4) + 1))
            first_size = size
        elif index == 1:
            options = [n for n in range(1, capacity + 1) if n != first_size]
            size = options[int(rng.integers(len(options)))] if options else 1
        else:
            size = int(rng.integers(1, capacity + 1))
        component = _grow(rng, depth_limit - depth_start, width, size)
        for d, o in component:
            cells.add((d + depth_start, o + start + 1))
        start += width + 1 + margins[index + 1]
    return cells


def _scattered(rng, depth_limit, orth_limit):
    parity = int(rng.integers(2))
    candidates = [
        (d, o + 1) for d in range(depth_limit) for o in range(orth_limit)
        if (d + o) % 2 == parity
    ]
    rng.shuffle(candidates)
    count = int(rng.integers(5, min(9, len(candidates)) + 1))
    cells = set(candidates[:count])
    if not any(d == 0 for d, _ in cells):
        cells.pop()
        cells.add(next(point for point in candidates if point[0] == 0))
    if not any(d > 0 for d, _ in cells):
        cells.pop()
        cells.add(next(point for point in candidates if point[0] > 0))
    return cells


def _shape(rng, kind, depth_limit, orth_limit):
    if kind == "scattered":
        return _scattered(rng, depth_limit, orth_limit)
    if kind in ("decisive", "aligned"):
        return _disconnected(rng, depth_limit, orth_limit, kind == "aligned")
    if kind == "hollow":
        depth = int(rng.integers(3, min(8, depth_limit) + 1))
        breadth = int(rng.integers(3, min(10, orth_limit) + 1))
        local = {
            (d, o) for d in range(depth) for o in range(breadth)
            if d in (0, depth - 1) or o in (0, breadth - 1)
        }
    elif kind == "elongated":
        long_depth = depth_limit >= 3 and (orth_limit < 3 or rng.random() < 0.5)
        if long_depth:
            length = int(rng.integers(3, min(depth_limit, 15) + 1))
            local = {(d, 0) for d in range(length)}
            if orth_limit >= 2 and rng.random() < 0.45:
                for d in range(length):
                    if rng.random() < 0.28:
                        local.add((d, 1))
        else:
            length = int(rng.integers(3, min(orth_limit, 15) + 1))
            local = {(0, o) for o in range(length)}
            if depth_limit >= 2 and rng.random() < 0.45:
                for o in range(length):
                    if rng.random() < 0.28:
                        local.add((1, o))
    else:
        for _ in range(16):
            depth = int(rng.integers(1, min(depth_limit, 7) + 1))
            breadth = int(rng.integers(1, min(orth_limit, 9) + 1))
            if depth * breadth >= 3:
                break
        else:
            depth, breadth = 1, 3
        upper = min(depth * breadth, 25 if kind == "irregular" else 12)
        lower = min(upper, 5 if kind == "irregular" else 3)
        local = _grow(rng, depth, breadth, int(rng.integers(lower, upper + 1)))
    first_orth = min(o for _, o in local)
    local = {(d, o - first_orth) for d, o in local}
    breadth = max(o for _, o in local) + 1
    shift = int(rng.integers(1, orth_limit - breadth + 2))
    return {(d, o + shift) for d, o in local}


def _make_pair(rng, regime, roles, direction, kind, fixed_gap=None):
    background, motif, barrier = roles
    if regime == "target_test":
        size_band = rng.random()
        if size_band < 0.23:
            height = int(rng.integers(6, 10))
            width = int(rng.integers(6, 10))
        elif size_band < 0.78:
            height = int(rng.integers(12, 23))
            width = int(rng.integers(10, 23))
        else:
            height = int(rng.integers(6, 17))
            width = int(rng.integers(6, 17))
    elif regime == "source_train" and rng.random() < 0.22:
        height = int(rng.integers(12, 22))
        width = int(rng.integers(10, 22))
    else:
        height = int(rng.integers(6, 17))
        width = int(rng.integers(6, 17))
    vertical = direction in ("right", "left")
    axis_length = width if vertical else height
    orth_length = height if vertical else width
    if kind in ("decisive", "scattered", "aligned", "hollow") or fixed_gap == 2:
        axis_length = max(axis_length, 8)
    if vertical:
        width = axis_length
    else:
        height = axis_length

    required_depth = 3 if kind in ("decisive", "scattered", "aligned", "hollow") else 1
    minimum_side = required_depth + (fixed_gap if fixed_gap is not None else 1) + 1
    side = int(rng.integers(minimum_side, axis_length - 1))
    if fixed_gap is None:
        gap = int(rng.integers(1, min(6, side - required_depth - 1) + 1))
    else:
        gap = fixed_gap
    depth_limit = side - gap - 1
    orth_limit = orth_length - 2
    line = side if direction in ("right", "down") else axis_length - side - 1

    for _ in range(40):
        if kind == "elongated" and depth_limit >= 2 and rng.random() < 0.28:
            cells = {(0, o) for o in range(orth_length)}
            for o in range(orth_length):
                if rng.random() < 0.35:
                    cells.add((1, o))
            if len(cells) == orth_length:
                cells.add((1, int(rng.integers(orth_length))))
        else:
            cells = _shape(rng, kind, depth_limit, orth_limit)
        if len(cells) < (height * width - orth_length) // 2:
            break
    else:
        raise RuntimeError("Could not fit motif")

    before = np.full((height, width), background, dtype=np.int8)
    after = before.copy()
    if vertical:
        before[:, line] = barrier
        after[:, line] = barrier
    else:
        before[line, :] = barrier
        after[line, :] = barrier
    for depth, orth in cells:
        distance = gap + 1 + depth
        new_distance = 1 + depth
        if direction == "right":
            before[orth, line - distance] = motif
            after[orth, line - new_distance] = motif
        elif direction == "left":
            before[orth, line + distance] = motif
            after[orth, line + new_distance] = motif
        elif direction == "down":
            before[line - distance, orth] = motif
            after[line - new_distance, orth] = motif
        else:
            before[line + distance, orth] = motif
            after[line + new_distance, orth] = motif
    return {"input": before, "output": after}


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        component = {remaining.pop()}
        frontier = list(component)
        while frontier:
            row, col = frontier.pop()
            for point in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if point in remaining:
                    remaining.remove(point)
                    component.add(point)
                    frontier.append(point)
        result.append(component)
    return result


def _analyze_pair(pair):
    before, after = pair["input"], pair["output"]
    assert before.shape == after.shape
    height, width = before.shape
    colors, counts = np.unique(before, return_counts=True)
    assert len(colors) == 3
    order = np.argsort(counts)
    background = int(colors[order[-1]])
    assert int(counts[order[-1]]) > int(counts[order[-2]])
    candidates = []
    for value in colors:
        color = int(value)
        if color == background:
            continue
        amount = int(np.count_nonzero(before == color))
        for column in range(width):
            if amount == height and np.all(before[:, column] == color):
                candidates.append(("vertical", column, color))
        for row in range(height):
            if amount == width and np.all(before[row, :] == color):
                candidates.append(("horizontal", row, color))
    assert len(candidates) == 1
    orientation, line, barrier = candidates[0]
    motif = next(int(value) for value in colors if int(value) not in (background, barrier))
    positions = np.argwhere(before == motif)
    assert len(positions) >= 1
    axis = 1 if orientation == "vertical" else 0
    coords = positions[:, axis]
    if np.all(coords < line):
        direction = "right" if axis == 1 else "down"
        gap = line - int(coords.max()) - 1
        delta = gap
    else:
        assert np.all(coords > line)
        direction = "left" if axis == 1 else "up"
        gap = int(coords.min()) - line - 1
        delta = -gap
    assert gap >= 1
    expected = before.copy()
    expected[before == motif] = background
    shifted = positions.copy()
    shifted[:, axis] += delta
    assert np.all((shifted[:, axis] >= 0) & (shifted[:, axis] < before.shape[axis]))
    expected[shifted[:, 0], shifted[:, 1]] = motif
    assert np.array_equal(after, expected)
    assert not np.array_equal(before, after)
    components = _components(before == motif)
    hollow = False
    for component in components:
        component_rows = [point[0] for point in component]
        component_cols = [point[1] for point in component]
        top, bottom = min(component_rows), max(component_rows)
        left, right = min(component_cols), max(component_cols)
        if bottom - top >= 2 and right - left >= 2:
            border = {
                (r, c) for r in range(top, bottom + 1) for c in range(left, right + 1)
                if r in (top, bottom) or c in (left, right)
            }
            if border <= component and len(component) < (bottom - top + 1) * (right - left + 1):
                hollow = True
    component_near = []
    for component in components:
        positions_on_axis = [point[axis] for point in component]
        component_near.append(
            line - max(positions_on_axis) if delta > 0 else min(positions_on_axis) - line
        )
    decisive = len(set(component_near)) > 1
    rows, cols = np.nonzero(before == motif)
    extent = (int(rows.max() - rows.min() + 1), int(cols.max() - cols.min() + 1))
    return {
        "roles": (background, motif, barrier), "direction": direction,
        "orientation": orientation, "line": line, "gap": gap,
        "components": len(components), "component_sizes": tuple(sorted(map(len, components))),
        "decisive": decisive, "motif_cells": len(positions), "extent": extent,
        "hollow": hollow,
    }


def _grid_key(grid):
    return (grid.shape, tuple(int(value) for value in grid.flat))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    features = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            input_key = _grid_key(pair["input"])
            pair_key = (input_key, _grid_key(pair["output"]))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            source_format = tuple("".join(str(int(v)) for v in row) for row in pair["input"])
            assert source_format not in _SOURCE_INPUTS
            features[split].append(_analyze_pair(pair))
    assert any(item["components"] == 1 for item in features["train"])
    assert any(item["decisive"] for item in features["train"])
    assert len({item["gap"] for item in features["train"]}) >= 2
    train_palettes = {item["roles"] for item in features["train"]}
    if len(train_palettes) == 1:
        assert all(item["roles"] in train_palettes for item in features["test"])


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    stable_palette = rng.random() < 0.38
    shared_roles = _palette(rng)
    focused = rng.random() < 0.25
    if focused:
        direction = _DIRECTIONS[int(rng.integers(4))]
        train_directions = [direction] * train_count
        test_directions = [direction] * test_count
    else:
        vertical_direction = _DIRECTIONS[int(rng.integers(2))]
        horizontal_direction = _DIRECTIONS[int(rng.integers(2, 4))]
        train_directions = [vertical_direction, horizontal_direction]
        train_directions += [_DIRECTIONS[int(rng.integers(4))] for _ in range(train_count - 2)]
        test_directions = [_DIRECTIONS[int(rng.integers(4))] for _ in range(test_count)]
    train_kinds = ["connected", "decisive"]
    extra_kinds = ("connected", "irregular", "elongated", "hollow", "decisive", "aligned", "scattered")
    train_kinds += [extra_kinds[int(rng.integers(len(extra_kinds)))] for _ in range(train_count - 2)]
    test_choices = ("connected", "irregular", "elongated", "hollow", "decisive", "decisive", "aligned", "scattered", "scattered")
    test_kinds = [test_choices[int(rng.integers(len(test_choices)))] for _ in range(test_count)]
    roles = []
    for index in range(train_count + test_count):
        if stable_palette:
            roles.append(shared_roles)
        elif index == 0:
            roles.append(shared_roles)
        else:
            candidate = _palette(rng)
            if index == 1:
                while candidate == shared_roles:
                    candidate = _palette(rng)
            roles.append(candidate)
    puzzle = {"train": [], "test": []}
    seen = set()
    specifications = (
        [("train", train_directions[i], train_kinds[i], 1 if i == 0 else 2 if i == 1 else None, roles[i]) for i in range(train_count)]
        + [("test", test_directions[i], test_kinds[i], None, roles[train_count + i]) for i in range(test_count)]
    )
    for split, direction, kind, fixed_gap, palette in specifications:
        for _ in range(100):
            pair = _make_pair(rng, "source_train" if split == "train" else "target_test", palette, direction, kind, fixed_gap)
            key = _grid_key(pair["input"])
            source_format = tuple("".join(str(int(v)) for v in row) for row in pair["input"])
            if key not in seen and source_format not in _SOURCE_INPUTS:
                seen.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("Could not make a new example")
    validate_generated_puzzle(puzzle)
    return puzzle


def _fingerprint(puzzle):
    return tuple(tuple((_grid_key(pair["input"]), _grid_key(pair["output"])) for pair in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts = {len(p["train"]) for p in puzzles}
    test_counts = {len(p["test"]) for p in puzzles}
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len({_fingerprint(p) for p in puzzles}) >= 40
    directions, gaps, dimensions, layouts = set(), set(), set(), set()
    train_sizes, test_sizes = set(), set()
    palette_varying = palette_stable = 0
    train_direction_varying = 0
    test_connected = test_decisive = test_aligned = 0
    small_test = large_train = interior = 0
    many_components = source_palette = hollow_train = hollow_test = 0
    far_gaps = full_orthogonal = uneven_components = near_edge = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train = [_analyze_pair(pair) for pair in puzzle["train"]]
        test = [_analyze_pair(pair) for pair in puzzle["test"]]
        assert any(x["decisive"] for x in train)
        assert any(x["components"] == 1 for x in train)
        assert len({x["gap"] for x in train}) >= 2
        palette_varying += len({x["roles"] for x in train}) > 1
        palette_stable += len({x["roles"] for x in train}) == 1
        train_direction_varying += len({x["direction"] for x in train}) > 1
        for split, information in (("train", train), ("test", test)):
            for pair, feature in zip(puzzle[split], information):
                directions.add(feature["direction"])
                gaps.add(feature["gap"])
                dimensions.add(pair["input"].shape)
                layout = pair["input"] == feature["roles"][1]
                points = np.argwhere(layout)
                origin = points.min(axis=0)
                layouts.add(tuple(sorted(tuple(map(int, point - origin)) for point in points)))
                line_length = pair["input"].shape[1 if feature["orientation"] == "vertical" else 0]
                interior += 3 <= feature["line"] <= line_length - 4
                near_edge += feature["line"] <= 2 or feature["line"] >= line_length - 3
                far_gaps += feature["gap"] >= 4
                many_components += feature["components"] >= 5
                uneven_components += len(set(feature["component_sizes"])) > 1
                source_palette += feature["roles"] == (0, 1, 2)
                orthogonal_axis = 0 if feature["orientation"] == "vertical" else 1
                full_orthogonal += feature["extent"][orthogonal_axis] == pair["input"].shape[orthogonal_axis]
                if split == "train":
                    train_sizes.add(pair["input"].shape)
                    large_train += max(pair["input"].shape) >= 15
                    hollow_train += feature["hollow"]
                else:
                    test_sizes.add(pair["input"].shape)
                    small_test += max(pair["input"].shape) <= 9
                    hollow_test += feature["hollow"]
                    test_connected += feature["components"] == 1
                    test_decisive += feature["decisive"]
                    test_aligned += feature["components"] > 1 and not feature["decisive"]
    assert directions == set(_DIRECTIONS)
    assert {1, 2, 3, 4}.issubset(gaps)
    assert len(dimensions) >= 50 and len(layouts) >= 60
    assert train_sizes & test_sizes
    assert palette_varying >= 12 and palette_stable >= 5
    assert train_direction_varying >= 20
    assert test_connected >= 10 and test_decisive >= 10 and test_aligned >= 3
    assert small_test >= 3 and large_train >= 10 and interior >= 50
    assert many_components >= 10 and uneven_components >= 20
    assert source_palette >= 5 and far_gaps >= 10
    assert hollow_train >= 5 and hollow_test >= 3
    assert full_orthogonal >= 1 and near_edge >= 20
    assert _fingerprint(puzzles[7]) == _fingerprint(generate_puzzle(7))
    for a, b in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        candidate = generate_puzzle(510, a, b)
        validate_generated_puzzle(candidate)
        assert a is None or len(candidate["train"]) == a
        assert b is None or len(candidate["test"]) == b
        assert _fingerprint(candidate) == _fingerprint(generate_puzzle(510, a, b))
    for a, b in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(510, a, b)
        except ValueError:
            continue
        raise AssertionError("Invalid counts accepted")
