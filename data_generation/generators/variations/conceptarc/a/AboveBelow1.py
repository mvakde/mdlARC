"""Generate horizontal-separator ARC puzzles.

The only rule shared by all pairs is geometric: retain the separator and
everything above it, and replace everything below it with background.
Colors, separator phase, objects, and dimensions are visible pair properties.
"""

import numpy as np


# Exact source inputs, rather than a ban on their dimensions or palettes.
_SOURCE_INPUTS = {
    ("000000000000", "000444400000", "000444400000", "000444400000",
     "000000000000", "000000000000", "222222222222", "000000000000",
     "000444400444", "000444400444", "000444400444"),
    ("0006660006666066", "0006660006666066", "3333333333333333",
     "0066600000666600", "0066600000666600", "0000000000666600",
     "0000000000666600", "0066006600666600", "0066006600000000",
     "0066006600000000"),
    ("0011100000001100", "0011100111001100", "0000000111000000",
     "8080808080808080", "1101110001110000", "1101110001110000",
     "1100000001110000", "0000000000000000", "0000000000000000"),
    ("0660066606", "0660066606", "0000066600", "4444444444",
     "0000000000", "0000000000", "0666006660", "0666006660",
     "0666000000", "0000000000", "0000000000", "0666600000",
     "0666600000", "0666600000"),
    ("0033333000000", "0033333000000", "0000000000000",
     "0000000000000", "9090909090909", "0000000000000"),
    ("000000000000", "000000000000", "000000000000",
     "000000000000", "000000000000", "000000000000",
     "505050505050", "000000000000", "005550055500",
     "005550055500"),
}


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
            raise ValueError(f"{name} is outside its legal range")

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


def _source_rows(grid):
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _background(grid):
    corners = (grid[0, 0], grid[0, -1], grid[-1, 0], grid[-1, -1])
    assert all(v == corners[0] for v in corners)
    bg = int(corners[0])
    counts = np.bincount(grid.ravel().astype(np.int64), minlength=10)
    assert counts[bg] > max(counts[c] for c in range(10) if c != bg)
    return bg


def _separator_rows(grid, bg):
    width = grid.shape[1]
    parity = np.arange(width) % 2 == 0
    found = []
    for y, row in enumerate(grid):
        colors = [int(c) for c in np.unique(row) if int(c) != bg]
        if len(colors) != 1:
            continue
        color = colors[0]
        if np.all(row == color):
            found.append((y, "solid", color, -1))
        elif np.array_equal(row, np.where(parity, color, bg)):
            found.append((y, "alternating", color, 0))
        elif np.array_equal(row, np.where(parity, bg, color)):
            found.append((y, "alternating", color, 1))
    return found


def _size(rng, mode):
    if mode == "compact":
        return int(rng.integers(6, 12)), int(rng.integers(8, 18))
    if mode == "large":
        return int(rng.integers(24, 31)), int(rng.integers(16, 31))
    return int(rng.integers(10, 24)), int(rng.integers(10, 27))


def _separator_position(rng, height, placement):
    if placement == "near_top":
        low, high = 1, min(4, height - 2)
    elif placement == "near_bottom":
        low, high = max(1, height - 4), height - 2
    elif placement == "middle":
        low, high = max(1, height // 3), min(height - 2, 2 * height // 3)
    else:
        low, high = 1, height - 2
    return int(rng.integers(low, high + 1))


def _paint_region(grid, start, stop, bg, colors, rng, bar):
    if start == stop:
        return 0
    rows, width = stop - start, grid.shape[1]
    target = int(rng.integers(1, min(8, 2 + rows * width // 28) + 1))
    painted = 0
    for index in range(target):
        for _ in range(36):
            if index == 0 and bar:
                if rng.integers(2):
                    rh = 1
                    rw = int(rng.integers(1, min(10, width - 2) + 1))
                else:
                    rw = 1
                    rh = int(rng.integers(1, min(9, rows) + 1))
            else:
                rh = int(rng.integers(1, min(9, rows) + 1))
                rw = int(rng.integers(1, min(10, width - 2) + 1))
            y = int(rng.integers(start, stop - rh + 1))
            x = int(rng.integers(0, width - rw + 1))
            halo = grid[max(start, y - 1):min(stop, y + rh + 1),
                        max(0, x - 1):min(width, x + rw + 1)]
            if np.any(halo != bg):
                continue
            color = colors[0] if index == 0 else int(rng.choice(colors))
            grid[y:y + rh, x:x + rw] = color
            painted += 1
            break
    assert painted >= 1
    return painted


def _random_spec(rng, regime):
    if regime == "source_train":
        size = str(rng.choice(("compact", "standard", "large"), p=(.22, .60, .18)))
    else:
        size = str(rng.choice(("compact", "standard", "large"), p=(.30, .30, .40)))
    return {
        "size": size,
        "placement": str(rng.choice(("near_top", "middle", "near_bottom", "any"))),
        "state": str(rng.choice(("both", "upper_empty", "lower_empty"),
                                p=(.55, .22, .23))),
        "kind": str(rng.choice(("solid", "alternating"))),
        "same": bool(rng.integers(2)),
        "bar": bool(rng.integers(2)),
    }


def _make_pair(rng, spec, bg, theme):
    for _ in range(160):
        height, width = _size(rng, spec["size"])
        sep = _separator_position(rng, height, spec["placement"])
        grid = np.full((height, width), bg, dtype=np.int8)

        preferred = [int(c) for c in theme if int(c) != bg]
        others = [c for c in range(10) if c != bg and c not in preferred]
        available = preferred + others
        core = int(rng.choice(available[:min(5, len(available))]))
        extra_pool = [c for c in available if c != core]
        ncolors = int(rng.integers(1, 4))
        objects = [core] + [int(c) for c in rng.choice(extra_pool,
                                                        size=ncolors - 1,
                                                        replace=False)]
        if spec["same"]:
            sep_color = core
        else:
            sep_color = int(rng.choice([c for c in available if c not in objects]))

        upper = spec["state"] != "upper_empty"
        lower = spec["state"] != "lower_empty"
        bar_side = int(rng.integers(2))
        if upper:
            _paint_region(grid, 0, sep, bg, objects, rng,
                          spec["bar"] and bar_side == 0)
        if lower:
            _paint_region(grid, sep + 1, height, bg, objects, rng,
                          spec["bar"] and (bar_side == 1 or not upper))
        if spec["kind"] == "solid":
            grid[sep, :] = sep_color
        else:
            phase = int(rng.integers(2))
            grid[sep, phase::2] = sep_color

        try:
            assert _background(grid) == bg
        except AssertionError:
            continue
        if _separator_rows(grid, bg) != [
            (sep, spec["kind"], sep_color,
             -1 if spec["kind"] == "solid" else phase)
        ]:
            continue
        if _source_rows(grid) in _SOURCE_INPUTS:
            continue

        output = grid.copy()
        output[sep + 1:] = bg
        return {"input": grid, "output": output}
    raise RuntimeError("could not place an unambiguous separator scene")


def _pair_facts(pair):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    bg = _background(inp)
    markers = _separator_rows(inp, bg)
    assert len(markers) == 1
    sep, kind, sep_color, phase = markers[0]
    assert 1 <= sep <= inp.shape[0] - 2
    assert np.array_equal(out[:sep + 1], inp[:sep + 1])
    assert np.all(out[sep + 1:] == bg)
    assert _source_rows(inp) not in _SOURCE_INPUTS
    above = set(int(c) for c in np.unique(inp[:sep])) - {bg}
    below = set(int(c) for c in np.unique(inp[sep + 1:])) - {bg}
    assert above or below
    return {
        "bg": bg, "sep": sep, "kind": kind, "phase": phase,
        "same": sep_color in above | below,
        "upper": bool(above), "lower": bool(below),
        "shared": bool(above & below),
        "object_colors": len(above | below),
    }


def _predict_from_training(train_pairs, test_input):
    # Only the visible demonstrations and the test input are used here.
    assert any(np.any(p["input"] != p["output"]) for p in train_pairs)
    bg = _background(test_input)
    markers = _separator_rows(test_input, bg)
    assert len(markers) == 1
    sep = markers[0][0]
    predicted = test_input.copy()
    predicted[sep + 1:] = bg
    return predicted


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4
    assert len(train) > len(test)

    inputs, pairs, facts = set(), set(), []
    for pair in train + test:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        for grid in (pair["input"], pair["output"]):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
            assert np.issubdtype(grid.dtype, np.integer)
            assert np.all((0 <= grid) & (grid <= 9))
        inp_key = (pair["input"].shape, pair["input"].tobytes())
        out_key = (pair["output"].shape, pair["output"].tobytes())
        assert inp_key not in inputs and (inp_key, out_key) not in pairs
        inputs.add(inp_key)
        pairs.add((inp_key, out_key))
        facts.append(_pair_facts(pair))

    training = facts[:len(train)]
    assert {f["kind"] for f in training} == {"solid", "alternating"}
    assert any(f["upper"] and f["lower"] and f["shared"] for f in training)
    assert any(f["same"] for f in training)
    assert any(f["lower"] for f in training)
    assert all(f["kind"] in {t["kind"] for t in training} for f in facts)
    if len(train) >= 4:
        assert any(not f["upper"] for f in training)
        assert any(not f["lower"] for f in training)
    assert any(p["input"].shape[0] >= 24 or
               f["sep"] >= p["input"].shape[0] - 4
               for p, f in zip(test, facts[len(train):]))
    for pair in test:
        assert np.array_equal(
            _predict_from_training(train, pair["input"]), pair["output"]
        )


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)

    # A color theme ties a puzzle together visually. It never encodes the
    # answer: the background and all other color roles vary by pair.
    theme = [int(c) for c in rng.permutation(10)]
    base_bg, alternate_bg = theme[:2]

    train_specs = [
        dict(_random_spec(rng, "source_train"), state="both", kind="solid",
             same=False),
        dict(_random_spec(rng, "source_train"),
             state=str(rng.choice(("both", "upper_empty", "lower_empty"))),
             kind="alternating", same=True),
    ]
    if num_train >= 3:
        train_specs.append(dict(_random_spec(rng, "source_train"),
                                state="upper_empty"))
    if num_train >= 4:
        train_specs.append(dict(_random_spec(rng, "source_train"),
                                state="lower_empty"))
    while len(train_specs) < num_train:
        train_specs.append(_random_spec(rng, "source_train"))

    test_specs = [_random_spec(rng, "target_test") for _ in range(num_test)]
    if rng.integers(2):
        test_specs[0]["size"] = "large"
        test_specs[0]["placement"] = "near_top"
    else:
        test_specs[0]["size"] = "compact"
        test_specs[0]["placement"] = "near_bottom"

    puzzle = {"train": [], "test": []}
    seen = set()
    for section, specs in (("train", train_specs), ("test", test_specs)):
        for index, spec in enumerate(specs):
            bg = (base_bg if index == 0 and section == "train" else
                  alternate_bg if index == 1 and section == "train" else
                  int(rng.choice(theme)))
            for _ in range(60):
                pair = _make_pair(rng, spec, bg, theme)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[section].append(pair)
                    break
            else:
                raise RuntimeError("could not generate distinct inputs")
        rng.shuffle(puzzle[section])

    validate_generated_puzzle(puzzle)
    return puzzle


def _equal_puzzles(a, b):
    return all(
        len(a[section]) == len(b[section]) and all(
            np.array_equal(pa[key], pb[key])
            for pa, pb in zip(a[section], b[section])
            for key in ("input", "output")
        )
        for section in ("train", "test")
    )


def _object_boxes(region, bg):
    """Return bounding dimensions of monochrome four-connected objects."""
    visited = np.zeros(region.shape, dtype=bool)
    boxes = []
    for y in range(region.shape[0]):
        for x in range(region.shape[1]):
            if visited[y, x] or region[y, x] == bg:
                continue
            color = region[y, x]
            stack = [(y, x)]
            visited[y, x] = True
            ys, xs = [], []
            while stack:
                cy, cx = stack.pop()
                ys.append(cy)
                xs.append(cx)
                for ny, nx in ((cy - 1, cx), (cy + 1, cx),
                               (cy, cx - 1), (cy, cx + 1)):
                    if (0 <= ny < region.shape[0] and
                        0 <= nx < region.shape[1] and
                        not visited[ny, nx] and region[ny, nx] == color):
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            boxes.append((max(ys) - min(ys) + 1,
                          max(xs) - min(xs) + 1))
    return boxes


def self_test() -> None:
    train_counts, test_counts, shapes, signatures = set(), set(), set(), set()
    backgrounds, kinds, phases, states = set(), set(), set(), set()
    train_empty_lower = train_empty_upper = test_empty_lower = 0
    same_train = same_test = multi_color = near_bottom = near_30 = 0
    train_bars = test_bars = tall_blocks = source_dimensions = 0
    upper_empty_same = combined_same_multicolor = penultimate = 0
    train_both_phases = varying_backgrounds = 0
    test_nochange_positions, test_same_positions = set(), set()
    object_counts = set()
    pair_total = 0

    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _equal_puzzles(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signature = []
        local_backgrounds = set()
        local_train_phases = set()
        for section in ("train", "test"):
            for index, pair in enumerate(puzzle[section]):
                fact = _pair_facts(pair)
                inp = pair["input"]
                pair_total += 1
                shapes.add(inp.shape)
                backgrounds.add(fact["bg"])
                local_backgrounds.add(fact["bg"])
                kinds.add(fact["kind"])
                phases.add(fact["phase"])
                states.add((fact["upper"], fact["lower"]))
                multi_color += fact["object_colors"] >= 2
                near_bottom += fact["sep"] >= inp.shape[0] - 4
                near_30 += max(inp.shape) >= 28
                penultimate += fact["sep"] == inp.shape[0] - 2
                source_dimensions += inp.shape in {
                    (11, 12), (10, 16), (9, 16),
                    (14, 10), (6, 13), (10, 12),
                }
                boxes = (_object_boxes(inp[:fact["sep"]], fact["bg"]) +
                         _object_boxes(inp[fact["sep"] + 1:], fact["bg"]))
                object_counts.add(len(boxes))
                bar = any(h == 1 or w == 1 for h, w in boxes)
                tall_blocks += any(h >= 5 for h, _ in boxes)
                upper_empty_same += not fact["upper"] and fact["same"]
                combined_same_multicolor += (
                    fact["same"] and fact["object_colors"] >= 2
                )
                signature.append((inp.shape, inp.tobytes(),
                                  pair["output"].tobytes()))
                if section == "train":
                    if fact["kind"] == "alternating":
                        local_train_phases.add(fact["phase"])
                    train_empty_lower += not fact["lower"]
                    train_empty_upper += not fact["upper"]
                    same_train += fact["same"]
                    train_bars += bar
                else:
                    test_empty_lower += not fact["lower"]
                    same_test += fact["same"]
                    test_bars += bar
                    if not fact["lower"]:
                        test_nochange_positions.add(index)
                    if fact["same"]:
                        test_same_positions.add(index)
                    assert np.array_equal(
                        _predict_from_training(puzzle["train"], inp),
                        pair["output"],
                    )
        assert len(local_backgrounds) >= 2
        varying_backgrounds += len(local_backgrounds) >= 2
        train_both_phases += len(local_train_phases) == 2
        signatures.add(tuple(signature))

    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 85 and len(signatures) >= 40
    assert len(backgrounds) == 10
    assert kinds == {"solid", "alternating"} and phases == {-1, 0, 1}
    assert states == {(True, True), (True, False), (False, True)}
    assert train_empty_lower >= 12 and train_empty_upper >= 12
    assert test_empty_lower >= 5 and same_train >= 42 and same_test >= 8
    assert multi_color >= 30 and near_bottom >= 20 and near_30 >= 20
    assert train_bars >= 30 and test_bars >= 10 and tall_blocks >= 20
    assert source_dimensions >= 2 and penultimate >= 10
    assert upper_empty_same >= 10 and combined_same_multicolor >= 15
    assert train_both_phases >= 15 and varying_backgrounds == 42
    assert len(test_nochange_positions) >= 3
    assert len(test_same_positions) >= 3
    assert len(object_counts) >= 8
    assert pair_total >= 150

    for ntrain, ntest in (
        (2, 1), (10, 1), (10, 4),
        (None, 1), (None, 4), (2, None), (10, None),
    ):
        puzzle = generate_puzzle(843, ntrain, ntest)
        validate_generated_puzzle(puzzle)
        assert ntrain is None or len(puzzle["train"]) == ntrain
        assert ntest is None or len(puzzle["test"]) == ntest
        assert _equal_puzzles(puzzle, generate_puzzle(843, ntrain, ntest))

    for kwargs in (
        {"num_train": 1}, {"num_train": 11},
        {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
    ):
        try:
            generate_puzzle(1, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid counts: {kwargs}")

    puzzle = generate_puzzle(23, 4, 2)
    broken = {section: [
        {key: pair[key].copy() for key in ("input", "output")}
        for pair in puzzle[section]
    ] for section in ("train", "test")}
    broken["train"][0]["output"][-1, 0] = (
        int(broken["train"][0]["input"][0, 0]) + 1
    ) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a changed lower output")
