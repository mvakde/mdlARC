"""Solved puzzles in which a center marker fills its complete lattice tile."""

import numpy as np


_SOURCE_LAYOUTS = (
    frozenset({(0, 1), (1, 1), (2, 1), (2, 2)}),
    frozenset({(0, 0), (1, 1), (2, 2)}),
    frozenset({(0, 2), (1, 0), (1, 2), (2, 0)}),
)


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(seed, num_train, num_test):
    train = _checked_count("num_train", num_train, 2, 10)
    test = _checked_count("num_test", num_test, 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if test is None and train is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    else:
        test = 1 + seed % min(4, train - 1)
    return int(train), int(test)


def _grid_key(grid):
    return grid.shape, np.ascontiguousarray(grid, dtype=np.int64).tobytes()


def _signature(puzzle):
    return tuple(
        tuple((_grid_key(p["input"]), _grid_key(p["output"])) for p in puzzle[split])
        for split in ("train", "test")
    )


def _render(layout, shape, tile, background, marker, ink):
    height, width = shape
    inp = np.full((height * tile, width * tile), background, dtype=np.int64)
    out = inp.copy()
    center = tile // 2
    for row, col in layout:
        inp[row * tile + center, col * tile + center] = marker
        out[row * tile:(row + 1) * tile, col * tile:(col + 1) * tile] = ink
    return {"input": inp, "output": out}


def _source_keys():
    return {
        (_grid_key(pair["input"]), _grid_key(pair["output"]))
        for pair in (_render(layout, (3, 3), 3, 0, 5, 1) for layout in _SOURCE_LAYOUTS)
    }


def _shapes(tile):
    limit = 5 if tile == 3 else 4
    return [(3, 3)] * 4 + [
        (h, w) for h in range(2, limit + 1) for w in range(2, limit + 1)
        if (h, w) != (3, 3)
    ]


def _sample_layout(rng, shape, kind, center_state=None):
    height, width = shape
    cells = [(r, c) for r in range(height) for c in range(width)]
    area = len(cells)
    center = (1, 1) if shape == (3, 3) else None

    if kind == "full":
        chosen = set(cells)
    elif kind == "single":
        chosen = {cells[int(rng.integers(area))]}
    elif kind == "pair":
        chosen = {cells[int(i)] for i in rng.choice(area, size=2, replace=False)}
    elif kind == "line":
        horizontal = bool(rng.integers(2))
        if horizontal:
            length = int(rng.integers(2, width + 1))
            row = int(rng.integers(height))
            start = int(rng.integers(width - length + 1))
            chosen = {(row, col) for col in range(start, start + length)}
        else:
            length = int(rng.integers(2, height + 1))
            col = int(rng.integers(width))
            start = int(rng.integers(height - length + 1))
            chosen = {(row, col) for row in range(start, start + length)}
    elif kind == "diagonal":
        length = int(rng.integers(2, min(height, width) + 1))
        row = int(rng.integers(height - length + 1))
        col = int(rng.integers(width - length + 1))
        if bool(rng.integers(2)):
            chosen = {(row + i, col + i) for i in range(length)}
        else:
            chosen = {(row + i, col + length - 1 - i) for i in range(length)}
    elif kind == "cluster":
        row = int(rng.integers(height - 1))
        col = int(rng.integers(width - 1))
        chosen = {(r, c) for r in (row, row + 1) for c in (col, col + 1)}
    elif kind == "corners":
        corners = sorted({(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)})
        count = int(rng.integers(2, len(corners) + 1))
        chosen = {corners[int(i)] for i in rng.choice(len(corners), size=count, replace=False)}
    elif kind == "ring":
        chosen = {(r, c) for r, c in cells if r in (0, height - 1) or c in (0, width - 1)}
    elif kind == "separated":
        chosen = set()
        for index in rng.permutation(area):
            candidate = cells[int(index)]
            if all(abs(candidate[0] - r) + abs(candidate[1] - c) > 1 for r, c in chosen):
                chosen.add(candidate)
            if len(chosen) >= min(5, (area + 1) // 2):
                break
        if len(chosen) < 2:
            chosen = {(0, 0), (height - 1, width - 1)}
    else:
        if kind == "dense" and area >= 7:
            low, high = 5, min(area - 1, max(8, area * 3 // 4))
        else:
            low, high = 1, area - 1
        count = int(rng.integers(low, high + 1))
        chosen = {cells[int(i)] for i in rng.choice(area, size=count, replace=False)}

    if center is not None and center_state is not None and kind != "full":
        if center_state and center not in chosen:
            if len(chosen) == area - 1:
                chosen.add(center)
            else:
                chosen.remove(sorted(chosen)[int(rng.integers(len(chosen)))])
                chosen.add(center)
        elif not center_state and center in chosen:
            chosen.remove(center)
            available = sorted(set(cells) - chosen - {center})
            chosen.add(available[int(rng.integers(len(available)))])
    return frozenset(chosen)


def _colors(palette, ink, mode, pair_index):
    if mode == 0:
        background, marker = palette[0], palette[1]
    elif mode == 1:
        background, marker = palette[0], palette[1 + pair_index % 8]
    elif mode == 2:
        background, marker = palette[1 + pair_index % 8], palette[0]
    else:
        background, marker = palette[pair_index % 9], palette[(pair_index + 3) % 9]
    assert len({background, marker, ink}) == 3
    return background, marker


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed_key = int(seed) % (1 << 64)
    train_count, test_count = _counts(seed_key, num_train, num_test)
    rng = np.random.default_rng(seed_key)

    # The tile footprint and output ink are learned once per puzzle.
    tile = 5 if seed_key % 5 == 0 else 3
    shift_mode = (seed_key // 4) % 4
    color_mode = (seed_key // 7) % 4
    colors = [int(c) for c in rng.permutation(10)]
    ink, palette = colors[0], colors[1:]
    shapes = _shapes(tile)
    rectangles = [shape for shape in shapes if shape[0] != shape[1]]
    kinds = ("single", "pair", "line", "diagonal", "cluster", "corners",
             "ring", "separated", "dense", "random", "random")
    puzzle = {"train": [], "test": []}
    used_structures = set()
    used_inputs = set()
    source_keys = _source_keys()

    for split, count in (("train", train_count), ("test", test_count)):
        for split_index in range(count):
            pair_index = len(puzzle["train"]) + len(puzzle["test"])
            background, marker = _colors(palette, ink, color_mode, pair_index)
            if shift_mode == 0:
                forced = split == "train"
            elif shift_mode == 1:
                forced = split == "test"
            elif shift_mode == 2:
                forced = True
            else:
                forced = None

            for attempt in range(96):
                if split == "train" and split_index == 0:
                    shape, kind = (3, 3), "single"
                elif split == "train" and split_index == 1:
                    shape = rectangles[int(rng.integers(len(rectangles)))]
                    kind = "cluster"
                elif split == "test" and split_index == 0 and shift_mode != 3:
                    shape = (3, 3)
                    kind = "full" if shift_mode == 2 and seed_key % 2 == 0 else kinds[int(rng.integers(len(kinds)))]
                else:
                    shape = shapes[int(rng.integers(len(shapes)))]
                    kind = kinds[int(rng.integers(len(kinds)))]
                if split == "test" and kind != "full" and rng.random() < 0.045 and forced is not False:
                    kind = "full"
                if kind == "full" and (split == "train" or forced is False):
                    kind = "dense"
                layout = _sample_layout(rng, shape, kind, forced)
                if split == "train" and len(layout) == shape[0] * shape[1]:
                    continue
                structure = (shape, layout)
                if structure in used_structures:
                    continue
                pair = _render(layout, shape, tile, background, marker, ink)
                input_key = _grid_key(pair["input"])
                pair_key = (input_key, _grid_key(pair["output"]))
                if input_key in used_inputs or pair_key in source_keys:
                    continue
                break
            else:
                # Exhaustive 3-by-3 layouts give a deterministic bounded fallback.
                shape = (3, 3)
                cells = [(r, c) for r in range(3) for c in range(3)]
                for mask in range(1, 1 << 9):
                    layout = frozenset(cells[i] for i in range(9) if mask & (1 << i))
                    if split == "train" and len(layout) == 9:
                        continue
                    if forced is not None and ((1, 1) in layout) != forced:
                        continue
                    structure = (shape, layout)
                    if structure in used_structures:
                        continue
                    pair = _render(layout, shape, tile, background, marker, ink)
                    input_key = _grid_key(pair["input"])
                    if input_key not in used_inputs and (input_key, _grid_key(pair["output"])) not in source_keys:
                        break
                else:
                    raise AssertionError("could not construct a unique pair")
            used_structures.add(structure)
            used_inputs.add(input_key)
            puzzle[split].append(pair)

    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_facts(pair, tile, ink):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    assert all(n % tile == 0 for n in inp.shape)
    values, counts = np.unique(inp, return_counts=True)
    assert len(values) == 2
    background = int(values[int(np.argmax(counts))])
    marker = int(values[int(np.argmin(counts))])
    assert background != marker and ink not in (background, marker)
    assert int(np.max(counts)) > int(np.min(counts))
    locations = np.argwhere(inp == marker)
    center = tile // 2
    assert len(locations) >= 1
    assert all(int(r) % tile == center and int(c) % tile == center for r, c in locations)
    layout = frozenset((int(r) // tile, int(c) // tile) for r, c in locations)
    shape = (inp.shape[0] // tile, inp.shape[1] // tile)
    assert shape[0] >= 2 and shape[1] >= 2
    expected = _render(layout, shape, tile, background, marker, ink)
    assert np.array_equal(inp, expected["input"])
    assert np.array_equal(out, expected["output"]), "incorrect tile expansion"
    assert not np.array_equal(inp, out)
    return shape, layout, background, marker


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    pairs = puzzle["train"] + puzzle["test"]
    for pair in pairs:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        for grid in pair.values():
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert all(1 <= n <= 30 for n in grid.shape)
            assert np.issubdtype(grid.dtype, np.integer)
            assert bool(np.all((grid >= 0) & (grid <= 9)))
    input_keys = [_grid_key(p["input"]) for p in pairs]
    pair_keys = [(_grid_key(p["input"]), _grid_key(p["output"])) for p in pairs]
    assert len(input_keys) == len(set(input_keys))
    assert len(pair_keys) == len(set(pair_keys))
    assert not set(pair_keys) & _source_keys()

    first = puzzle["train"][0]
    values, counts = np.unique(first["input"], return_counts=True)
    assert len(values) == 2
    background = int(values[int(np.argmax(counts))])
    output_values = set(int(c) for c in np.unique(first["output"]))
    assert background in output_values and len(output_values) == 2
    ink = next(iter(output_values - {background}))
    area = int(np.count_nonzero(first["output"] == ink))
    tile = int(np.sqrt(area))
    assert tile in (3, 5) and tile * tile == area
    assert first["input"].shape == (3 * tile, 3 * tile)

    facts = [_pair_facts(p, tile, ink) for p in pairs]
    train_facts = facts[:len(puzzle["train"])]
    assert len(train_facts[0][1]) == 1, "first demonstration must isolate the tile footprint"
    second_shape, second_layout, _, _ = train_facts[1]
    assert second_shape[0] != second_shape[1]
    assert any(abs(r - rr) + abs(c - cc) == 1 for r, c in second_layout for rr, cc in second_layout)
    for shape, layout, _, _ in train_facts:
        assert len(layout) < shape[0] * shape[1], "training must expose background and ink"
    structures = [(shape, layout) for shape, layout, _, _ in facts]
    assert len(structures) == len(set(structures))
    for pair in puzzle["train"]:
        assert ink in pair["output"]


def _predict_from_training(train, test_input):
    first = train[0]
    values, counts = np.unique(first["input"], return_counts=True)
    background = int(values[int(np.argmax(counts))])
    ink = next(int(c) for c in np.unique(first["output"]) if int(c) != background)
    footprint = int(np.count_nonzero(first["output"] == ink))
    tile = int(np.sqrt(footprint))
    values, counts = np.unique(test_input, return_counts=True)
    test_background = int(values[int(np.argmax(counts))])
    marker = int(values[int(np.argmin(counts))])
    result = np.full(test_input.shape, test_background, dtype=np.int64)
    for row, col in np.argwhere(test_input == marker):
        r, c = int(row) // tile, int(col) // tile
        result[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = ink
    return result


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_signature(p) for p in puzzles}) == 42

    tiles, shapes, layouts, counts, colors = set(), set(), set(), set(), set()
    train_centers, test_centers = set(), set()
    variable_background = variable_marker = full_tests = 0
    varied_shapes = varied_counts = 0
    center_reversals = center_no_shift = 0
    observed_patterns = set()
    cluster_origins = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        first = puzzle["train"][0]
        vals, amounts = np.unique(first["input"], return_counts=True)
        background = int(vals[int(np.argmax(amounts))])
        ink = next(int(c) for c in np.unique(first["output"]) if int(c) != background)
        tile = int(np.sqrt(np.count_nonzero(first["output"] == ink)))
        tiles.add(tile)
        backgrounds, markers = set(), set()
        local_shapes, local_counts = set(), set()
        local_center = {"train": set(), "test": set()}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                shape, layout, bg, marker = _pair_facts(pair, tile, ink)
                local_shapes.add(pair["input"].shape)
                local_counts.add(len(layout))
                shapes.add(pair["input"].shape)
                layouts.add((shape, layout))
                counts.add(len(layout))
                colors.add((bg, marker, ink))
                backgrounds.add(bg)
                markers.add(marker)
                center = tile // 2
                for row in range(shape[0]):
                    for col in range(shape[1]):
                        marked = pair["input"][row * tile + center, col * tile + center] == marker
                        block = pair["output"][row * tile:(row + 1) * tile, col * tile:(col + 1) * tile]
                        assert bool(np.all(block == (ink if marked else bg)))
                rows = {r for r, _ in layout}
                cols = {c for _, c in layout}
                if len(layout) >= 2 and len(rows) == 1:
                    observed_patterns.add("row")
                if len(layout) >= 2 and len(cols) == 1:
                    observed_patterns.add("column")
                if len(layout) >= 2 and len({r + c for r, c in layout}) == 1:
                    observed_patterns.add("other_diagonal")
                if len(layout) >= 2 and all(abs(r - rr) + abs(c - cc) > 1
                                            for r, c in layout for rr, cc in layout if (r, c) != (rr, cc)):
                    observed_patterns.add("separated")
                border = {(r, c) for r in range(shape[0]) for c in range(shape[1])
                          if r in (0, shape[0] - 1) or c in (0, shape[1] - 1)}
                if layout == border and len(border) < shape[0] * shape[1]:
                    observed_patterns.add("ring")
                for r in range(shape[0] - 1):
                    for c in range(shape[1] - 1):
                        if {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)} <= layout:
                            observed_patterns.add("cluster")
                            cluster_origins.add((shape, r, c))
                if shape == (3, 3):
                    (train_centers if split == "train" else test_centers).add((1, 1) in layout)
                    local_center[split].add((1, 1) in layout)
                if split == "test":
                    assert np.array_equal(_predict_from_training(puzzle["train"], pair["input"]), pair["output"])
                    full_tests += len(layout) == shape[0] * shape[1]
        variable_background += len(backgrounds) > 1
        variable_marker += len(markers) > 1
        varied_shapes += len(local_shapes) > 1
        varied_counts += len(local_counts) > 1
        center_reversals += (local_center["train"] == {True} and local_center["test"] == {False}) or (
            local_center["train"] == {False} and local_center["test"] == {True})
        center_no_shift += local_center["train"] == local_center["test"] == {True}
    assert tiles == {3, 5}
    assert len(shapes) >= 12 and len(layouts) >= 100
    assert {1, 2, 3, 4, 5, 6, 7, 8}.issubset(counts)
    assert len(colors) >= 60
    assert train_centers == {False, True} and test_centers == {False, True}
    assert variable_background >= 8 and variable_marker >= 8
    assert full_tests >= 1
    assert varied_shapes == 42 and varied_counts == 42
    assert center_reversals >= 8 and center_no_shift >= 2
    assert observed_patterns == {"row", "column", "other_diagonal", "separated", "ring", "cluster"}
    assert len(cluster_origins) >= 12

    for seed, train, test in ((1000, 2, 1), (1001, 10, 1), (1002, 10, 4),
                              (2000, None, 1), (2001, None, 4),
                              (2002, 2, None), (2003, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    assert _signature(generate_puzzle(314159)) == _signature(generate_puzzle(314159))
    assert _signature(generate_puzzle(271828, 7, 3)) == _signature(generate_puzzle(271828, 7, 3))
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": 2.0},
                   {"num_test": True}):
        try:
            generate_puzzle(9999, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"illegal count accepted: {kwargs}")
