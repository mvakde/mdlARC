import numpy as np


def _counts(rng, num_train, num_test):
    for value, low, high, name in (
        (num_train, 2, 10, "num_train"),
        (num_test, 1, 4, "num_test"),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high
        ):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _palette(rng):
    return tuple(int(x) for x in rng.choice(10, size=3, replace=False))


def _next_palette(rng, first, history, force_change=False):
    if force_change:
        if bool(rng.integers(2)):
            return (first[1], first[2], first[0])
        remaining = [x for x in range(10) if x not in first]
        return tuple(int(x) for x in rng.choice(remaining, size=3, replace=False))
    mode = int(rng.integers(4))
    if mode == 0:
        return history[int(rng.integers(len(history)))]
    if mode == 1:
        # A literal digit may take a different semantic role in another pair.
        return (first[1], first[2], first[0])
    if mode == 2:
        remaining = [x for x in range(10) if x != first[0]]
        chosen = rng.choice(remaining, size=2, replace=False)
        candidate = (first[0], int(chosen[0]), int(chosen[1]))
    else:
        candidate = _palette(rng)
    return candidate


def _dimension(rng):
    band = int(rng.choice(4, p=(0.24, 0.29, 0.29, 0.18)))
    bounds = ((5, 9), (10, 14), (15, 22), (23, 30))[band]
    return int(rng.integers(bounds[0], bounds[1] + 1))


def _shape(rng, square=None):
    if square is None:
        square = bool(rng.integers(2))
    height = _dimension(rng)
    width = height if square else _dimension(rng)
    if not square and height == width:
        width = 5 + (width - 4) % 26
    return height, width


def _make_pair(height, width, palette, orientation, line_side, marker_side, offset):
    background, line, mover = palette
    inp = np.full((height, width), background, dtype=np.int8)
    out = np.full((height, width), background, dtype=np.int8)
    if orientation == "vertical":
        line_col = 0 if line_side == 0 else width - 1
        source_row = 0 if marker_side == 0 else height - 1
        destination_row = height - 1 - source_row
        inp[:, line_col] = line
        inp[source_row, offset] = mover
        out[destination_row, offset] = mover
    else:
        line_row = 0 if line_side == 0 else height - 1
        source_col = 0 if marker_side == 0 else width - 1
        destination_col = width - 1 - source_col
        inp[line_row, :] = line
        inp[offset, source_col] = mover
        out[offset, destination_col] = mover
    return {"input": inp, "output": out}


def _signature(grid):
    return grid.shape, tuple(int(x) for x in grid.flat)


def _source_signatures():
    result = set()
    for offset in (1, 2, 4, 5):
        pair = _make_pair(6, 6, (0, 4, 1), "vertical", 0, 1, offset)
        result.add((_signature(pair["input"]), _signature(pair["output"])))
    for offset in (2, 4):
        pair = _make_pair(6, 6, (0, 4, 1), "horizontal", 0, 0, offset)
        result.add((_signature(pair["input"]), _signature(pair["output"])))
    pair = _make_pair(6, 6, (0, 6, 9), "horizontal", 0, 0, 2)
    result.add((_signature(pair["input"]), _signature(pair["output"])))
    return result


def _analyze_input(inp):
    height, width = inp.shape
    assert height >= 5 and width >= 5
    colors, counts = np.unique(inp, return_counts=True)
    assert len(colors) == 3
    background = int(colors[int(np.argmax(counts))])
    boundaries = (
        ("vertical", 0, inp[:, 0]),
        ("vertical", 1, inp[:, -1]),
        ("horizontal", 0, inp[0, :]),
        ("horizontal", 1, inp[-1, :]),
    )
    candidates = [
        (orientation, side, int(edge[0]))
        for orientation, side, edge in boundaries
        if edge[0] != background and bool(np.all(edge == edge[0]))
    ]
    assert len(candidates) == 1
    orientation, line_side, line = candidates[0]
    marker_colors = [int(x) for x in colors if x not in (background, line)]
    assert len(marker_colors) == 1
    mover = marker_colors[0]
    locations = np.argwhere(inp == mover)
    assert len(locations) == 1
    row, col = (int(x) for x in locations[0])
    expected_input = np.full(inp.shape, background, dtype=np.int8)
    expected_output = np.full(inp.shape, background, dtype=np.int8)
    if orientation == "vertical":
        assert row in (0, height - 1)
        assert col != (0 if line_side == 0 else width - 1)
        marker_side, offset = (0 if row == 0 else 1), col
        expected_input[:, 0 if line_side == 0 else width - 1] = line
        expected_input[row, col] = mover
        expected_output[height - 1 - row, col] = mover
    else:
        assert col in (0, width - 1)
        assert row != (0 if line_side == 0 else height - 1)
        marker_side, offset = (0 if col == 0 else 1), row
        expected_input[0 if line_side == 0 else height - 1, :] = line
        expected_input[row, col] = mover
        expected_output[row, width - 1 - col] = mover
    assert np.array_equal(inp, expected_input)
    return {
        "palette": (background, line, mover),
        "orientation": orientation,
        "line_side": line_side,
        "marker_side": marker_side,
        "offset": offset,
        "shape": inp.shape,
        "expected_output": expected_output,
    }


def _inspect_pair(pair):
    info = _analyze_input(pair["input"])
    assert pair["output"].shape == pair["input"].shape
    assert np.array_equal(pair["output"], info["expected_output"])
    return info


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[split], list) for split in ("train", "test"))
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10
    assert 1 <= len(test) <= 4 and len(train) > len(test)
    seen_inputs = set()
    seen_pairs = set()
    originals = _source_signatures()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            input_key = _signature(pair["input"])
            pair_key = (input_key, _signature(pair["output"]))
            assert input_key not in seen_inputs
            assert pair_key not in seen_pairs and pair_key not in originals
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            infos[split].append(_inspect_pair(pair))
    training = infos["train"]
    assert {info["orientation"] for info in training} == {"vertical", "horizontal"}
    assert len({info["palette"] for info in training}) >= 2
    assert all(len({info["palette"][role] for info in training}) >= 2 for role in range(3))
    assert len({info["shape"] for info in training}) >= 2
    assert any(h == w for h, w in (info["shape"] for info in training))
    assert any(h != w for h, w in (info["shape"] for info in training))
    seen_directions = {(x["orientation"], x["marker_side"]) for x in training}
    seen_line_sides = {(x["orientation"], x["line_side"]) for x in training}
    for info in infos["test"]:
        assert (info["orientation"], info["marker_side"]) in seen_directions
        assert (info["orientation"], info["line_side"]) in seen_line_sides


def _offset(rng, shape, orientation, line_side, familiar=None, far=False):
    height, width = shape
    length = width if orientation == "vertical" else height
    blocked = 0 if line_side == 0 else length - 1
    allowed = [x for x in range(length) if x != blocked]
    if far:
        return length - 1 - blocked
    if familiar and rng.random() < 0.38:
        available = [x for x in familiar if x in allowed]
        if available:
            return int(available[int(rng.integers(len(available)))])
    return int(allowed[int(rng.integers(len(allowed)))])


def _unique_pair(shape, palette, geometry, offset, seen, originals):
    orientation, line_side, marker_side = geometry
    height, width = shape
    square = height == width
    for delta in range(26):
        if square:
            trial_height = trial_width = 5 + (height - 5 + delta) % 26
        else:
            trial_height = height
            trial_width = 5 + (width - 5 + delta) % 26
            if trial_width == trial_height:
                continue
        length = trial_width if orientation == "vertical" else trial_height
        blocked = 0 if line_side == 0 else length - 1
        trial_offset = offset
        if trial_offset >= length or trial_offset == blocked:
            trial_offset = length - 1 - blocked
        pair = _make_pair(
            trial_height, trial_width, palette, orientation,
            line_side, marker_side, trial_offset,
        )
        input_key = _signature(pair["input"])
        pair_key = (input_key, _signature(pair["output"]))
        if input_key not in seen and pair_key not in originals:
            seen.add(input_key)
            return pair
    raise AssertionError("no unique grid found")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    # The line-to-axis rule is fixed for the family. There is no hidden
    # direction or color convention to sample at puzzle level.
    matched_corners = num_train >= 4 and bool(rng.random() < 0.28)
    first_palette = _palette(rng)
    train_geometries = [
        ("vertical", int(rng.integers(2)), int(rng.integers(2))),
        ("horizontal", int(rng.integers(2)), int(rng.integers(2))),
    ]
    if matched_corners:
        train_geometries[:2] = [
            ("vertical", 0, 1),
            ("horizontal", 0, 1),
        ]
    for _ in range(num_train - 2):
        train_geometries.append(
            (
                ("vertical", "horizontal")[int(rng.integers(2))],
                int(rng.integers(2)),
                int(rng.integers(2)),
            )
        )

    if matched_corners:
        shared_shape = _shape(rng, square=True)
        train_shapes = [shared_shape, shared_shape, _shape(rng, square=False)]
        train_shapes += [_shape(rng) for _ in range(num_train - 3)]
    else:
        train_shapes = [_shape(rng, square=True), _shape(rng, square=False)]
        if bool(rng.integers(2)):
            train_shapes.reverse()
        train_shapes += [_shape(rng) for _ in range(num_train - 2)]

    palettes = [first_palette]
    if matched_corners:
        palettes.append(first_palette)
        palettes.append(_next_palette(rng, first_palette, palettes, True))
    else:
        palettes.append(_next_palette(rng, first_palette, palettes, True))
    while len(palettes) < num_train:
        palettes.append(_next_palette(rng, first_palette, palettes))

    originals = _source_signatures()
    seen = set()
    train_pairs = []
    for index, geometry in enumerate(train_geometries):
        orientation, line_side, _ = geometry
        shape = train_shapes[index]
        offset = _offset(
            rng, shape, orientation, line_side,
            far=matched_corners and index < 2,
        )
        train_pairs.append(
            _unique_pair(shape, palettes[index], geometry, offset, seen, originals)
        )
    rng.shuffle(train_pairs)
    train_info = [_inspect_pair(pair) for pair in train_pairs]
    test_geometries = []
    for orientation in ("vertical", "horizontal"):
        line_sides = {x["line_side"] for x in train_info if x["orientation"] == orientation}
        marker_sides = {x["marker_side"] for x in train_info if x["orientation"] == orientation}
        test_geometries.extend(
            (orientation, line_side, marker_side)
            for line_side in line_sides for marker_side in marker_sides
        )
    demonstrated = set(train_geometries)
    test_pairs = []
    for _ in range(num_test):
        held_out = [g for g in test_geometries if g not in demonstrated]
        choices = held_out if held_out and rng.random() < 0.38 else test_geometries
        geometry = choices[int(rng.integers(len(choices)))]
        orientation, line_side, _ = geometry
        shape = _shape(rng)
        familiar_offsets = [
            x["offset"] for x in train_info if x["orientation"] == orientation
        ]
        offset = _offset(rng, shape, orientation, line_side, familiar_offsets)
        palette_mode = int(rng.integers(4))
        if palette_mode == 0:
            palette = train_info[int(rng.integers(len(train_info)))]["palette"]
        elif palette_mode == 1:
            palette = _next_palette(rng, first_palette, palettes, True)
        else:
            palette = _palette(rng)
        test_pairs.append(
            _unique_pair(shape, palette, geometry, offset, seen, originals)
        )
    puzzle = {"train": train_pairs, "test": test_pairs}
    validate_generated_puzzle(puzzle)
    return puzzle


def _same_puzzles(first, second):
    return all(
        len(first[split]) == len(second[split])
        and all(
            np.array_equal(a[role], b[role])
            for a, b in zip(first[split], second[split])
            for role in ("input", "output")
        )
        for split in ("train", "test")
    )


def _predict_from_demonstrations(train_pairs, test_input):
    # Only supplied demonstrations and the unlabelled test input are read.
    demonstrations = [_inspect_pair(pair) for pair in train_pairs]
    geometry = _analyze_input(test_input)
    assert (geometry["orientation"], geometry["marker_side"]) in {
        (x["orientation"], x["marker_side"]) for x in demonstrations
    }
    assert (geometry["orientation"], geometry["line_side"]) in {
        (x["orientation"], x["line_side"]) for x in demonstrations
    }
    return geometry["expected_output"]


def self_test() -> None:
    train_counts, test_counts = set(), set()
    shapes, palettes, layouts, inputs = set(), set(), set(), set()
    directions, line_sides, train_balance = set(), set(), set()
    role_swaps = familiar_test_palettes = novel_test_palettes = 0
    familiar_test_offsets = held_out_geometries = long_crossings = 0
    matched_corners = 0
    puzzles = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzles(puzzle, generate_puzzle(seed))
        puzzles.append(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_info = [_inspect_pair(pair) for pair in puzzle["train"]]
        test_info = [_inspect_pair(pair) for pair in puzzle["test"]]
        vertical = sum(x["orientation"] == "vertical" for x in train_info)
        train_balance.add(
            "vertical" if vertical > len(train_info) - vertical
            else "horizontal" if vertical < len(train_info) - vertical
            else "balanced"
        )
        assert len({x["palette"] for x in train_info}) >= 2
        assert all(len({x["palette"][role] for x in train_info}) >= 2 for role in range(3))
        assert len({x["shape"] for x in train_info}) >= 2
        assert any(x["shape"][0] == x["shape"][1] for x in train_info)
        assert any(x["shape"][0] != x["shape"][1] for x in train_info)
        if any(
            a["orientation"] == "vertical" and a["line_side"] == 0
            and a["marker_side"] == 1 and a["offset"] == a["shape"][1] - 1
            and b["orientation"] == "horizontal" and b["line_side"] == 0
            and b["marker_side"] == 1 and b["offset"] == b["shape"][0] - 1
            and a["shape"] == b["shape"] and a["palette"] == b["palette"]
            for a in train_info for b in train_info
        ):
            matched_corners += 1
        train_geometries = {
            (x["orientation"], x["line_side"], x["marker_side"]) for x in train_info
        }
        train_offsets = {(x["orientation"], x["offset"]) for x in train_info}
        train_palettes = {x["palette"] for x in train_info}
        for a in train_info:
            for b in train_info:
                if a["palette"] != b["palette"] and set(a["palette"]) == set(b["palette"]):
                    role_swaps += 1
                    break
        for test_index, info in enumerate(test_info):
            familiar_test_palettes += info["palette"] in train_palettes
            novel_test_palettes += info["palette"] not in train_palettes
            familiar_test_offsets += (info["orientation"], info["offset"]) in train_offsets
            held_out_geometries += (
                info["orientation"], info["line_side"], info["marker_side"]
            ) not in train_geometries
            predicted = _predict_from_demonstrations(
                [
                    {"input": pair["input"].copy(), "output": pair["output"].copy()}
                    for pair in puzzle["train"]
                ],
                puzzle["test"][test_index]["input"].copy(),
            )
            assert np.array_equal(predicted, info["expected_output"])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _inspect_pair(pair)
                assert np.array_equal(pair["output"], info["expected_output"])
                shapes.add(info["shape"])
                palettes.add(info["palette"])
                layouts.add(
                    (info["shape"], info["orientation"], info["line_side"],
                     info["marker_side"], info["offset"])
                )
                inputs.add(_signature(pair["input"]))
                directions.add((info["orientation"], info["marker_side"]))
                line_sides.add((info["orientation"], info["line_side"]))
                length = info["shape"][0] if info["orientation"] == "vertical" else info["shape"][1]
                long_crossings += length > 12
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 45 and len(palettes) >= 60
    assert len(layouts) >= 120 and len(inputs) >= 120
    assert len(directions) == 4 and len(line_sides) == 4
    assert train_balance == {"vertical", "horizontal", "balanced"}
    assert role_swaps >= 5 and matched_corners >= 2
    assert familiar_test_palettes >= 4 and novel_test_palettes >= 10
    assert familiar_test_offsets >= 3 and held_out_geometries >= 1
    assert long_crossings >= 30
    assert all(
        not _same_puzzles(puzzles[i], puzzles[j])
        for i in range(42) for j in range(i + 1, 42)
    )
    cases = (
        (100, 2, 1), (101, 10, 1), (102, 10, 4),
        (103, None, 1), (104, None, 4),
        (105, 2, None), (106, 10, None), (107, 7, 3),
    )
    for seed, train_count, test_count in cases:
        puzzle = generate_puzzle(seed, train_count, test_count)
        validate_generated_puzzle(puzzle)
        assert _same_puzzles(puzzle, generate_puzzle(seed, train_count, test_count))
        assert train_count is None or len(puzzle["train"]) == train_count
        assert test_count is None or len(puzzle["test"]) == test_count
    for train_count, test_count in (
        (1, None), (11, None), (None, 0), (None, 5),
        (2, 2), (3, 4), (2, 4),
    ):
        try:
            generate_puzzle(999, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
