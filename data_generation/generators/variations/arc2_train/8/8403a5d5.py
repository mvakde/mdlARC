"""Generate solved boundary-seeded serpentine ARC puzzles."""

import numpy as np

_SIDES = ("bottom", "top", "left", "right")
_INWARD = {"bottom": (-1, 0), "top": (1, 0), "left": (0, 1), "right": (0, -1)}
_COLORS = tuple(range(10))


def _counts(rng, num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= value <= high):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _sweep(side, hand):
    dr, dc = _INWARD[side]
    return hand * dc, -hand * dr


def _axes(height, width, side):
    return (height, width) if side in ("bottom", "top") else (width, height)


def _shape(stroke_length, lateral_size, side):
    return ((stroke_length, lateral_size) if side in ("bottom", "top")
            else (lateral_size, stroke_length))


def _seed(height, width, side, remaining, hand):
    sr, sc = _sweep(side, hand)
    if side == "bottom":
        return height - 1, remaining if sc < 0 else width - 1 - remaining
    if side == "top":
        return 0, remaining if sc < 0 else width - 1 - remaining
    if side == "left":
        return remaining if sr < 0 else height - 1 - remaining, 0
    return remaining if sr < 0 else height - 1 - remaining, width - 1


def _draw(height, width, side, row, col, background, path, marker, hand):
    """Walk each full stroke and place a single-cell turn at its endpoint."""
    out = np.full((height, width), background, dtype=np.int8)
    dr, dc = _INWARD[side]
    sr, sc = _sweep(side, hand)
    while True:
        while 0 <= row < height and 0 <= col < width:
            out[row, col] = path
            last_row, last_col = row, col
            row += dr
            col += dc
        marker_row, marker_col = last_row + sr, last_col + sc
        if not (0 <= marker_row < height and 0 <= marker_col < width):
            break
        out[marker_row, marker_col] = marker
        row, col = marker_row + sr, marker_col + sc
        if not (0 <= row < height and 0 <= col < width):
            break
        dr, dc = -dr, -dc
    return out


def _pair(stroke_length, lateral_size, side, remaining, background,
          path, marker, hand):
    height, width = _shape(stroke_length, lateral_size, side)
    row, col = _seed(height, width, side, remaining, hand)
    inp = np.full((height, width), background, dtype=np.int8)
    inp[row, col] = path
    return {"input": inp, "output": _draw(
        height, width, side, row, col, background, path, marker, hand)}


def _signature(grid):
    return grid.shape, grid.tobytes()


def _pair_signature(pair):
    return _signature(pair["input"]), _signature(pair["output"])


def _source_signatures():
    return {
        _pair_signature(_pair(10, 10, "bottom", 9 - col, 0, color, 5, 1))
        for col, color in ((5, 3), (1, 2), (4, 4), (2, 1))
    }


_SOURCE_SIGNATURES = _source_signatures()


def _decode_input(inp):
    colors, counts = np.unique(inp, return_counts=True)
    assert len(colors) == 2
    background = int(colors[np.argmax(counts)])
    assert int(np.max(counts)) == inp.size - 1
    point = np.argwhere(inp != background)
    assert len(point) == 1
    row, col = (int(v) for v in point[0])
    height, width = inp.shape
    sides = [side for side, yes in (
        ("bottom", row == height - 1), ("top", row == 0),
        ("left", col == 0), ("right", col == width - 1)) if yes]
    assert len(sides) == 1, "seed must be on one noncorner boundary"
    return background, int(inp[row, col]), row, col, sides[0]


def _formula_output(inp, marker, hand):
    """Rebuild by lane coordinates, independently of the walking renderer."""
    background, path, row, col, side = _decode_input(inp)
    height, width = inp.shape
    stroke_length, _ = _axes(height, width, side)
    dr, dc = _INWARD[side]
    sr, sc = _sweep(side, hand)
    out = np.full(inp.shape, background, dtype=np.int8)
    lane = 0
    while True:
        base_row, base_col = row + lane * sr, col + lane * sc
        if not (0 <= base_row < height and 0 <= base_col < width):
            break
        if lane % 2 == 0:
            for distance in range(stroke_length):
                out[base_row + distance * dr, base_col + distance * dc] = path
        else:
            distance = stroke_length - 1 if (lane // 2) % 2 == 0 else 0
            out[base_row + distance * dr, base_col + distance * dc] = marker
        lane += 1
    return out


def _inspect_pair(pair):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    background, path, row, col, side = _decode_input(inp)
    extra = set(int(v) for v in np.unique(out)) - {background, path}
    assert len(extra) == 1, "one distinct turn color is required"
    marker = extra.pop()
    height, width = inp.shape
    stroke_length, lateral_size = _axes(height, width, side)
    dr, dc = _INWARD[side]
    far_row = row + dr * (stroke_length - 1)
    far_col = col + dc * (stroke_length - 1)
    hands = [hand for hand in (-1, 1)
             if 0 <= far_row + _sweep(side, hand)[0] < height
             and 0 <= far_col + _sweep(side, hand)[1] < width
             and out[far_row + _sweep(side, hand)[0],
                     far_col + _sweep(side, hand)[1]] == marker]
    assert len(hands) == 1, "first turn must reveal sweep direction"
    hand = hands[0]
    sr, sc = _sweep(side, hand)
    remaining = ((height - 1 - row if sr > 0 else row) if sr else
                 (width - 1 - col if sc > 0 else col))
    assert 1 <= remaining <= lateral_size - 2
    assert np.array_equal(out, _formula_output(inp, marker, hand)), (
        "output differs from the full-stroke, one-cell-turn rule")
    return {
        "height": height, "width": width, "side": side,
        "stroke_length": stroke_length, "lateral_size": lateral_size,
        "remaining": remaining, "stroke_count": remaining // 2 + 1,
        "terminal_connector": bool(remaining % 2),
        "background": background, "path": path, "marker": marker,
        "hand": hand, "seed_row": row, "seed_col": col,
    }


def _predict(train_pairs, test_input):
    """Infer marker and handedness from demonstrations, then solve the input."""
    examples = [_inspect_pair(pair) for pair in train_pairs]
    assert len({item["marker"] for item in examples}) == 1
    assert len({item["hand"] for item in examples}) == 1
    marker, hand = examples[0]["marker"], examples[0]["hand"]
    background, path, _, _, _ = _decode_input(test_input)
    assert len({background, path, marker}) == 3
    return _formula_output(test_input, marker, hand)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dimension <= 30 for dimension in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            signature = _pair_signature(pair)
            input_signature = _signature(pair["input"])
            assert input_signature not in inputs and signature not in pairs
            assert signature not in _SOURCE_SIGNATURES
            inputs.add(input_signature)
            pairs.add(signature)
            infos[split].append(_inspect_pair(pair))
    training = infos["train"]
    all_infos = training + infos["test"]
    assert len({item["marker"] for item in all_infos}) == 1
    assert len({item["hand"] for item in all_infos}) == 1
    assert {item["terminal_connector"] for item in training} == {False, True}
    assert any(item["remaining"] >= 4 for item in training)
    assert len({item["path"] for item in training}) >= 2
    assert len({item["background"] for item in training}) >= 2
    for pair in puzzle["test"]:
        assert np.array_equal(_predict(puzzle["train"], pair["input"]), pair["output"])


def _choose_color(rng, excluded):
    options = [color for color in _COLORS if color not in excluded]
    return int(options[int(rng.integers(len(options)))])


def _choose_remaining(rng, lateral_size, parity=None, minimum=1):
    options = [value for value in range(minimum, lateral_size - 1)
               if parity is None or value % 2 == parity]
    assert options
    return int(options[int(rng.integers(len(options)))])


def _random_train_spec(rng, marker, base, side, phase=None,
                       new_background=False, new_path=False):
    stroke_length = int(rng.integers(5, 31))
    lateral_size = int(rng.integers(5, 31))
    if rng.random() < 0.25:
        lateral_size = stroke_length
    background = _choose_color(
        rng, {marker, base["background"]} if new_background else {marker})
    path = _choose_color(
        rng, {marker, background, base["path"]} if new_path else
        {marker, background})
    return dict(stroke_length=stroke_length, lateral_size=lateral_size,
                side=side, remaining=_choose_remaining(rng, lateral_size, phase),
                background=background, path=path)


def _random_test_spec(rng, marker, training_specs, side):
    mode = int(rng.integers(6))
    if mode == 0:
        parent = training_specs[int(rng.integers(len(training_specs)))]
        stroke_length, lateral_size, side = (
            parent["stroke_length"], parent["lateral_size"], parent["side"])
    elif mode == 1:
        stroke_length = lateral_size = 10
    elif mode == 2:
        stroke_length = int(rng.integers(5, 13))
        lateral_size = int(rng.integers(5, 13))
    elif mode == 3:
        stroke_length = int(rng.integers(5, 31))
        lateral_size = stroke_length
    elif mode == 4:
        stroke_length = int(rng.integers(5, 31))
        lateral_size = int(rng.integers(5, 31))
    else:
        stroke_length = int(rng.integers(9, 31))
        lateral_size = int(rng.integers(22, 31))
    if mode == 5:
        remaining = _choose_remaining(rng, lateral_size, minimum=15)
    elif rng.random() < 0.35:
        remaining = int(rng.integers(1, min(3, lateral_size - 2) + 1))
    else:
        remaining = _choose_remaining(rng, lateral_size)
    training_paths = {item["path"] for item in training_specs}
    if rng.random() < 0.5:
        path = int(sorted(training_paths)[int(rng.integers(len(training_paths)))])
    else:
        new_paths = set(_COLORS) - training_paths - {marker}
        path = (_choose_color(rng, set(_COLORS) - new_paths) if new_paths else
                int(sorted(training_paths)[int(rng.integers(len(training_paths)))]))
    background = _choose_color(rng, {marker, path})
    return dict(stroke_length=stroke_length, lateral_size=lateral_size,
                side=side, remaining=remaining, background=background, path=path)


def _materialize(spec, marker, hand):
    return _pair(spec["stroke_length"], spec["lateral_size"], spec["side"],
                 spec["remaining"], spec["background"], spec["path"],
                 marker, hand)


def _append_unique(target, spec, marker, hand, used):
    """Resolve a rare collision by cycling seed colors, then stroke length."""
    candidate = dict(spec)
    for attempt in range(300):
        pair = _materialize(candidate, marker, hand)
        key = _signature(pair["input"])
        if key not in used and _pair_signature(pair) not in _SOURCE_SIGNATURES:
            target.append(pair)
            used.add(key)
            return
        options = [c for c in _COLORS if c not in (marker, candidate["background"])]
        candidate["path"] = options[(options.index(candidate["path"]) + 1) % len(options)]
        if attempt % len(options) == len(options) - 1:
            candidate["stroke_length"] = 5 + (candidate["stroke_length"] - 4) % 26
    raise AssertionError("could not construct a unique pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    marker = int(rng.integers(10))
    hand = 1 if rng.random() < 0.5 else -1
    base_side = _SIDES[int(rng.integers(4))]
    mixed_sides = bool(rng.integers(2))
    alternatives = [side for side in _SIDES if side != base_side]
    alternate_side = alternatives[int(rng.integers(3))]
    source_sized = rng.random() < 0.25
    stroke_length = 10 if source_sized else int(rng.integers(6, 31))
    lateral_size = 10 if source_sized else int(rng.integers(10, 31))
    background = _choose_color(rng, {marker})
    path = _choose_color(rng, {marker, background})
    base = dict(stroke_length=stroke_length, lateral_size=lateral_size,
                side=base_side,
                remaining=_choose_remaining(rng, lateral_size, minimum=4),
                background=background, path=path)
    specs = [base]
    if train_count >= 3:
        positional = dict(base)
        positional["remaining"] = _choose_remaining(
            rng, lateral_size, parity=1 - base["remaining"] % 2)
        specs.append(positional)
    if train_count >= 5:
        length_variant = dict(base)
        options = [value for value in range(5, 31) if value != stroke_length]
        length_variant["stroke_length"] = int(
            options[int(rng.integers(len(options)))])
        specs.append(length_variant)
    if train_count >= 4:
        color_variant = dict(base)
        color_variant["path"] = _choose_color(rng, {marker, background, path})
        specs.append(color_variant)
    first_extra_index = len(specs)
    while len(specs) < train_count:
        first_extra = len(specs) == first_extra_index
        side = (alternate_side if mixed_sides and first_extra else
                base_side if not mixed_sides else _SIDES[int(rng.integers(4))])
        specs.append(_random_train_spec(
            rng, marker, base, side,
            phase=1 - base["remaining"] % 2 if train_count == 2 else None,
            new_background=first_extra, new_path=first_extra))
    rng.shuffle(specs)
    train_pairs, test_pairs, used = [], [], set()
    for spec in specs:
        _append_unique(train_pairs, spec, marker, hand, used)
    for _ in range(test_count):
        side = base_side if not mixed_sides else _SIDES[int(rng.integers(4))]
        spec = _random_test_spec(rng, marker, specs, side)
        _append_unique(test_pairs, spec, marker, hand, used)
    puzzle = {"train": train_pairs, "test": test_pairs}
    validate_generated_puzzle(puzzle)
    return puzzle


def _expect_value_error(**kwargs):
    try:
        generate_puzzle(123456, **kwargs)
    except ValueError:
        return
    raise AssertionError(f"expected ValueError for {kwargs}")


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    for index, (train, test) in enumerate((
        (2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
        (2, None), (10, None))):
        puzzle = generate_puzzle(1000 + index, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
        validate_generated_puzzle(puzzle)
    def puzzle_key(puzzle):
        return tuple(_pair_signature(pair) for split in ("train", "test")
                     for pair in puzzle[split])
    assert puzzle_key(generate_puzzle(314159, 7, 3)) == puzzle_key(
        generate_puzzle(314159, 7, 3))
    assert puzzle_key(generate_puzzle(314159, 7, 3)) != puzzle_key(
        generate_puzzle(314160, 7, 3))
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": True},
                   {"num_train": 3.0}):
        _expect_value_error(**kwargs)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({puzzle_key(p) for p in puzzles}) >= 40
    train_infos = [[_inspect_pair(pair) for pair in p["train"]] for p in puzzles]
    test_infos = [[_inspect_pair(pair) for pair in p["test"]] for p in puzzles]
    all_train = [i for group in train_infos for i in group]
    all_test = [i for group in test_infos for i in group]
    all_infos = all_train + all_test
    assert {i["side"] for i in all_infos} == set(_SIDES)
    assert {i["hand"] for i in all_infos} == {-1, 1}
    assert len({i["marker"] for i in all_infos}) >= 7
    assert len({i["background"] for i in all_infos}) >= 7
    assert len({(i["height"], i["width"]) for i in all_infos}) >= 70
    assert len({(i["side"], i["height"], i["width"], i["remaining"])
                for i in all_infos}) >= 130
    for group in (all_train, all_test):
        assert any(i["height"] == i["width"] for i in group)
        assert any(i["height"] != i["width"] for i in group)
        assert any((i["height"], i["width"]) == (10, 10) for i in group)
    assert {i["remaining"] for i in all_test} >= {1, 2, 3}
    assert {i["terminal_connector"] for i in all_test} == {False, True}
    assert any(i["remaining"] >= 19 for i in all_test)
    assert any(len({i["side"] for i in group}) > 1 for group in train_infos)
    assert any(set((i["height"], i["width"]) for i in tr)
               & set((i["height"], i["width"]) for i in te)
               for tr, te in zip(train_infos, test_infos))
    assert any(set(i["path"] for i in tr) & set(i["path"] for i in te)
               for tr, te in zip(train_infos, test_infos))
    assert any(set(i["path"] for i in te) - set(i["path"] for i in tr)
               for tr, te in zip(train_infos, test_infos))
    assert any(any(i["stroke_count"] <= max(t["stroke_count"] for t in tr)
                   for i in te) for tr, te in zip(train_infos, test_infos))
    same_size_position = same_offset_length = color_only = 0
    for group in train_infos:
        for index, a in enumerate(group):
            for b in group[index + 1:]:
                if (a["side"], a["height"], a["width"], a["background"], a["path"]) == (
                    b["side"], b["height"], b["width"], b["background"], b["path"]
                ) and a["remaining"] != b["remaining"]:
                    same_size_position += 1
                if (a["side"], a["lateral_size"], a["remaining"],
                    a["background"], a["path"]) == (
                    b["side"], b["lateral_size"], b["remaining"],
                    b["background"], b["path"]
                ) and a["stroke_length"] != b["stroke_length"]:
                    same_offset_length += 1
                if (a["side"], a["height"], a["width"], a["remaining"],
                    a["background"]) == (
                    b["side"], b["height"], b["width"], b["remaining"],
                    b["background"]
                ) and a["path"] != b["path"]:
                    color_only += 1
    assert same_size_position >= 10
    assert same_offset_length >= 5
    assert color_only >= 5
