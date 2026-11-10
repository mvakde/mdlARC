"""Generate three-marker ARC puzzles with clipped concentric square outlines."""

import numpy as np


# The four source inputs are sufficient signatures: the rule fixes their outputs.
_SOURCE_INPUT_SIGNATURES = frozenset({
    ((23, 23), 0, 2, ((11, 13), (13, 11), (15, 9))),
    ((23, 23), 0, 8, ((2, 11), (5, 14), (8, 17))),
    ((23, 23), 0, 3, ((4, 12), (8, 8), (12, 4))),
    ((28, 28), 0, 4, ((12, 7), (18, 13), (24, 19))),
})


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
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


def _integer(rng, low, high):
    return int(rng.integers(low, high + 1))


def _dimensions(rng, minimum):
    """Independent aspect choices, with additional square and long canvases."""
    shape_kind = int(rng.integers(0, 5))
    if shape_kind == 0:
        side = _integer(rng, minimum, 30)
        return side, side
    if shape_kind == 1:
        return _integer(rng, minimum, min(30, minimum + 5)), _integer(rng, max(minimum, 24), 30)
    if shape_kind == 2:
        return _integer(rng, max(minimum, 24), 30), _integer(rng, minimum, min(30, minimum + 5))
    return _integer(rng, minimum, 30), _integer(rng, minimum, 30)


def _test_dimensions(rng, d, train_shapes):
    minimum = max(9, 2 * d + 1)
    mode = int(rng.integers(0, 5))
    if mode == 0:
        possible = [shape for shape in train_shapes if min(shape) >= minimum]
        if possible:
            return possible[int(rng.integers(len(possible)))]
    if mode == 1:
        possible = [shape for shape in train_shapes if min(shape) >= minimum and (shape[0] > minimum or shape[1] > minimum)]
        if possible:
            h, w = possible[int(rng.integers(len(possible)))]
            for _ in range(8):
                candidate = (_integer(rng, minimum, h), _integer(rng, minimum, w))
                if candidate != (h, w):
                    return candidate
            return (minimum, w) if h > minimum else (h, minimum)
    if mode == 2:
        possible = [shape for shape in train_shapes if shape[0] < 30 and shape[1] < 30]
        if possible:
            h, w = possible[int(rng.integers(len(possible)))]
            return _integer(rng, max(minimum, h + 1), 30), _integer(rng, max(minimum, w + 1), 30)
    return _dimensions(rng, minimum)


def _center_pools(height, width, d):
    pools = {name: [] for name in ("border", "near_one", "near_both", "deep_two", "deep_three")}
    for row in range(d, height - d):
        for col in range(d, width - d):
            margins = (row, height - 1 - row, col, width - 1 - col)
            minimum = min(margins)
            if minimum == d:
                kind = "border"
            elif minimum < 2 * d:
                row_clip = min(margins[:2]) < 2 * d
                col_clip = min(margins[2:]) < 2 * d
                kind = "near_both" if row_clip and col_clip else "near_one"
            elif minimum < 3 * d:
                kind = "deep_two"
            else:
                kind = "deep_three"
            pools[kind].append((row, col))
    return pools


def _center(rng, height, width, d, requested):
    pools = _center_pools(height, width, d)
    if requested == "clipped":
        kinds = [k for k in ("border", "near_one", "near_both") if pools[k]]
    elif requested == "deep_three":
        kinds = ["deep_three"]
    else:
        kinds = [k for k in pools if pools[k]]
    assert kinds
    kind = kinds[int(rng.integers(len(kinds)))]
    pool = pools[kind]
    return pool[int(rng.integers(len(pool)))]


def _draw_pair(shape, center, d, orientation, background, foreground):
    height, width = shape
    row, col = center
    inp = np.full(shape, background, dtype=np.int8)
    for marker_row, marker_col in ((row - d, col - orientation * d),
                                   (row, col),
                                   (row + d, col + orientation * d)):
        inp[marker_row, marker_col] = foreground
    rows, cols = np.indices(shape)
    distance = np.maximum(np.abs(rows - row), np.abs(cols - col))
    out = np.full(shape, background, dtype=np.int8)
    out[distance % d == 0] = foreground
    return {"input": inp, "output": out}


def _grid_key(grid):
    return grid.shape, grid.astype(np.uint8, copy=False).tobytes()


def _analyze(pair):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    colors, counts = np.unique(inp, return_counts=True)
    assert len(colors) == 2
    foreground = int(colors[int(np.argmin(counts))])
    background = int(colors[int(np.argmax(counts))])
    assert int(np.min(counts)) == 3
    coordinates = np.argwhere(inp == foreground)
    step_a = coordinates[1] - coordinates[0]
    step_b = coordinates[2] - coordinates[1]
    assert np.array_equal(step_a, step_b)
    d = int(step_a[0])
    assert d >= 2 and abs(int(step_a[1])) == d
    center = tuple(int(x) for x in coordinates[1])
    orientation = 1 if step_a[1] > 0 else -1
    rows, cols = np.indices(inp.shape)
    distance = np.maximum(np.abs(rows - center[0]), np.abs(cols - center[1]))
    expected = np.where(distance % d == 0, foreground, background)
    assert np.array_equal(out, expected)
    assert not np.array_equal(inp, out)
    height, width = inp.shape
    margins = (center[0], height - 1 - center[0], center[1], width - 1 - center[1])
    max_distance = max(margins)
    clipped_sides_two = sum(margin < 2 * d for margin in margins)
    return {
        "shape": inp.shape, "center": center, "d": d, "orientation": orientation,
        "background": background, "foreground": foreground,
        "complete_rings": min(margins) // d,
        "visible_rings": max_distance // d,
        "marker_border": min(margins) == d,
        "clipped_sides_two": clipped_sides_two,
        "source_signature": (inp.shape, background, foreground,
                             tuple(tuple(int(x) for x in point) for point in coordinates)),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= side <= 30 for side in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            input_key = _grid_key(pair["input"])
            pair_key = input_key, _grid_key(pair["output"])
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            info = _analyze(pair)
            assert info["source_signature"] not in _SOURCE_INPUT_SIGNATURES
            infos[split].append(info)
    train = infos["train"]
    test = infos["test"]
    # Training establishes spacing as a local geometric cue, including repeated rings
    # and clipping. Slopes and colors are local cues, so no shared palette is required.
    assert len({item["d"] for item in train}) >= 2
    assert all(item["visible_rings"] >= 2 for item in train)
    assert any(item["complete_rings"] >= 3 for item in train)
    assert any(item["clipped_sides_two"] > 0 for item in train)
    assert len({item["shape"] for item in train}) >= 2
    assert all(2 <= item["d"] <= 6 for item in train)
    assert all(2 <= item["d"] <= 8 for item in test)
    train_configs = {(item["shape"], item["center"], item["d"]) for item in train}
    assert all((item["shape"], item["center"], item["d"]) not in train_configs for item in test)


def _colors(rng, train_count, test_count):
    count = train_count + test_count
    background_mode = int(rng.integers(0, 2))
    if background_mode == 0:
        backgrounds = [int(rng.integers(10))] * count
    else:
        backgrounds = [int(rng.integers(10)) for _ in range(count)]
        if len(set(backgrounds)) == 1:
            backgrounds[-1] = (backgrounds[-1] + 1) % 10
    palette_mode = int(rng.integers(0, 3))
    if palette_mode == 0:
        order = [int(x) for x in rng.permutation(10)]
        groups = [order[:5]] * train_count + [order[5:]] * test_count
        foregrounds = [int(rng.choice([c for c in group if c != bg])) for group, bg in zip(groups, backgrounds)]
    elif palette_mode == 1:
        common = (int(rng.choice([c for c in range(10) if c != backgrounds[0]]))
                  if background_mode == 0 else int(rng.integers(10)))
        palette = [common] + [int(x) for x in rng.permutation([c for c in range(10) if c != common])[:3]]
        for index in (0, train_count):
            if backgrounds[index] == common:
                backgrounds[index] = (common + 1 + int(rng.integers(9))) % 10
        if background_mode == 1 and len(set(backgrounds)) == 1:
            backgrounds[1] = int(rng.choice([c for c in range(10)
                                              if c not in (common, backgrounds[1])]))
        foregrounds = [int(rng.choice([c for c in palette if c != bg])) for bg in backgrounds]
        foregrounds[0] = common
        foregrounds[train_count] = common
        if train_count >= 3 and backgrounds[1] != common:
            foregrounds[1] = common
    else:
        foregrounds = [int(rng.choice([c for c in range(10) if c != bg])) for bg in backgrounds]
    return list(zip(backgrounds, foregrounds))


def _orientations(rng, train_count, test_count):
    mode = int(rng.integers(0, 3))
    sign = 1 if int(rng.integers(2)) else -1
    if mode == 0:
        return [sign] * (train_count + test_count)
    if mode == 2:
        return [sign] * train_count + [-sign] + [int(rng.choice((-1, 1))) for _ in range(test_count - 1)]
    return [int(rng.choice((-1, 1))) for _ in range(train_count + test_count)]


def _sample_specs(rng, train_count, test_count):
    anchor_index = int(rng.integers(train_count))
    clipped_index = int(rng.choice([i for i in range(train_count) if i != anchor_index]))
    anchor_d = int(rng.choice((2, 3)))
    alternate_d = int(rng.choice([d for d in range(2, 6) if d != anchor_d]))
    train_d = [int(rng.integers(2, 7)) for _ in range(train_count)]
    train_d[anchor_index] = anchor_d
    train_d[clipped_index] = alternate_d
    colors = _colors(rng, train_count, test_count)
    orientations = _orientations(rng, train_count, test_count)
    specs = {"train": [], "test": []}
    train_shapes = []
    for index, d in enumerate(train_d):
        minimum = max(12, 4 * d + 1)
        if index == anchor_index:
            minimum = max(21, 6 * d + 1)
        for _ in range(32):
            shape = _dimensions(rng, minimum)
            if shape not in train_shapes:
                break
        else:
            # A deterministic distinct shape is always available at these ranges.
            candidates = [(h, w) for h in range(minimum, 31) for w in range(minimum, 31)
                          if (h, w) not in train_shapes]
            shape = candidates[int(rng.integers(len(candidates)))]
        train_shapes.append(shape)
        requested = "deep_three" if index == anchor_index else "clipped" if index == clipped_index else "any"
        center = _center(rng, *shape, d, requested)
        specs["train"].append((shape, center, d, orientations[index], *colors[index]))
    max_train_d = max(train_d)
    for index in range(test_count):
        spacing_mode = int(rng.integers(0, 3))
        if spacing_mode == 0:
            d = int(rng.choice(train_d))
        elif spacing_mode == 1 and max_train_d < 8:
            d = _integer(rng, max_train_d + 1, min(8, max_train_d + 3))
        else:
            d = _integer(rng, 2, 8)
        shape = _test_dimensions(rng, d, train_shapes)
        center = _center(rng, *shape, d, "any")
        color = colors[train_count + index]
        orientation = orientations[train_count + index]
        specs["test"].append((shape, center, d, orientation, *color))
    return specs


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    for _ in range(16):
        specs = _sample_specs(rng, train_count, test_count)
        puzzle = {split: [_draw_pair(*spec) for spec in specs[split]] for split in ("train", "test")}
        try:
            validate_generated_puzzle(puzzle)
        except AssertionError:
            continue
        return puzzle
    raise ValueError("could not construct a valid puzzle")


def _puzzle_key(puzzle):
    return tuple((_grid_key(pair["input"]), _grid_key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                # A separate pixel calculation checks the construction, without
                # relying on the validator's expected-grid calculation.
                inp, out = pair["input"], pair["output"]
                values, frequencies = np.unique(inp, return_counts=True)
                fg = int(values[int(np.argmin(frequencies))])
                bg = int(values[int(np.argmax(frequencies))])
                points = np.argwhere(inp == fg)
                center = points[1]
                step = int(points[1, 0] - points[0, 0])
                for row in range(inp.shape[0]):
                    for col in range(inp.shape[1]):
                        radius = max(abs(row - int(center[0])), abs(col - int(center[1])))
                        assert int(out[row, col]) == (fg if radius % step == 0 else bg)
                info = _analyze(pair)
                assert info["source_signature"] not in _SOURCE_INPUT_SIGNATURES
    assert len({len(puzzle["train"]) for puzzle in puzzles}) >= 4
    assert len({len(puzzle["test"]) for puzzle in puzzles}) >= 3
    assert len({_puzzle_key(puzzle) for puzzle in puzzles}) == 42
    assert _puzzle_key(generate_puzzle(314159, 7, 3)) == _puzzle_key(generate_puzzle(314159, 7, 3))
    assert _puzzle_key(generate_puzzle(314159, 7, 3)) != _puzzle_key(generate_puzzle(314160, 7, 3))
    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(1000 + train_count + test_count, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
    for train_count, test_count in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(2000 + (train_count or 0) + (test_count or 0), train_count, test_count)
        assert train_count is None or len(puzzle["train"]) == train_count
        assert test_count is None or len(puzzle["test"]) == test_count
    for train_count, test_count in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(99, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    infos = [{split: [_analyze(pair) for pair in puzzle[split]] for split in ("train", "test")}
             for puzzle in puzzles]
    all_infos = [item for group in infos for split in ("train", "test") for item in group[split]]
    assert len({item["shape"] for item in all_infos}) >= 50
    assert len({(item["shape"], item["center"], item["d"], item["orientation"]) for item in all_infos}) >= 150
    assert {item["d"] for item in all_infos} == set(range(2, 9))
    assert {item["orientation"] for item in all_infos} == {-1, 1}
    assert len({item["background"] for item in all_infos}) == 10
    assert any(len({item["background"] for split in ("train", "test") for item in group[split]}) > 1 for group in infos)
    assert any({item["foreground"] for item in group["train"]} & {item["foreground"] for item in group["test"]} for group in infos)
    assert any({item["foreground"] for item in group["train"]}.isdisjoint({item["foreground"] for item in group["test"]}) for group in infos)
    assert any(len({item["orientation"] for item in group["train"]}) == 1 and
               any(item["orientation"] != group["train"][0]["orientation"] for item in group["test"]) for group in infos)
    assert any(all(item["orientation"] == group["train"][0]["orientation"] for split in ("train", "test") for item in group[split]) for group in infos)
    assert any(item["marker_border"] for item in all_infos)
    assert any(item["marker_border"] for group in infos for item in group["train"])
    assert any(item["marker_border"] for group in infos for item in group["test"])
    assert any(item["clipped_sides_two"] == 1 and not item["marker_border"] for item in all_infos)
    assert any(item["clipped_sides_two"] >= 2 and not item["marker_border"] for item in all_infos)
    assert any(item["complete_rings"] >= 4 for item in all_infos)
    assert any(item["complete_rings"] >= 3 for group in infos for item in group["test"])
    assert any(item["visible_rings"] == 1 for group in infos for item in group["test"])
    assert any(item["shape"][0] == item["shape"][1] for item in all_infos)
    assert any(item["shape"][0] > item["shape"][1] + 7 for item in all_infos)
    assert any(item["shape"][1] > item["shape"][0] + 7 for item in all_infos)
    assert any(item["d"] in {train["d"] for train in group["train"]} for group in infos for item in group["test"])
    assert any(item["d"] > max(train["d"] for train in group["train"]) for group in infos for item in group["test"])
    assert any(item["shape"] in {train["shape"] for train in group["train"]} for group in infos for item in group["test"])
    assert any(all(item["shape"][axis] < max(train["shape"][axis] for train in group["train"]) for axis in (0, 1))
               for group in infos for item in group["test"])
    assert len({index for group in infos for index, item in enumerate(group["train"])
                if item["clipped_sides_two"] > 0}) >= 4
    assert all(len({item["shape"] for item in group["train"]}) >= 2 and
               len({item["d"] for item in group["train"]}) >= 2 and
               len({item["center"] for item in group["train"]}) >= 2 for group in infos)
