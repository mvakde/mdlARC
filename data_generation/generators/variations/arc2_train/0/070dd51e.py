"""Compositional endpoint-connector puzzles with vertical foreground crossings."""

import numpy as np


# Exact source inputs, represented by shape and colored endpoint coordinates.
# The output is determined by these endpoints, so rejecting these input keys
# also prevents an exact source pair copy without banning its canvas size.
_SOURCE_INPUT_KEYS = {
    ((20, 10), 0, ((3, 4, 2), (3, 4, 7), (4, 2, 3), (4, 10, 3),
                    (7, 8, 2), (7, 8, 5), (8, 14, 1), (8, 14, 6),
                    (9, 12, 5), (9, 18, 5))),
    ((30, 20), 0, ((2, 2, 6), (2, 13, 6), (3, 6, 3), (3, 6, 11),
                    (5, 20, 2), (5, 20, 7), (6, 18, 4), (6, 27, 4),
                    (8, 12, 14), (8, 17, 14))),
    ((20, 20), 0, ((2, 3, 1), (2, 3, 16), (3, 1, 3), (3, 18, 3),
                    (5, 2, 9), (5, 9, 9), (7, 7, 7), (7, 7, 13),
                    (8, 14, 8), (8, 14, 14))),
}


def _checked_count(value, name, lower, upper):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _resolve_counts(rng, num_train, num_test):
    train = None if num_train is None else _checked_count(num_train, "num_train", 2, 10)
    test = None if num_test is None else _checked_count(num_test, "num_test", 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test + 1), 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _dimension(rng, lower):
    # Independent height and width bins include source dimensions, small
    # canvases, and both long and wide canvases.
    if rng.random() < 0.24:
        anchors = [size for size in (10, 20, 30) if size >= lower]
        if anchors:
            return anchors[int(rng.integers(0, len(anchors)))]
    bins = ((5, 10), (11, 20), (21, 30))
    available = [(max(lower, lo), hi) for lo, hi in bins if max(lower, lo) <= hi]
    lo, hi = available[int(rng.integers(0, len(available)))]
    return int(rng.integers(lo, hi + 1))


def _containing_lows(size, point, length):
    """Possible starts for a segment with point strictly in its interior."""
    low = max(0, point - length + 2)
    high = min(point - 1, size - length)
    return range(low, high + 1)


def _demo_core(rng, height, width, short_vertical=None, box=None):
    """A crossing, optionally with a prescribed relative length order."""
    r0, c0, h, w = box if box is not None else (0, 0, height, width)
    row = int(rng.integers(1, h - 1))
    col = int(rng.integers(1, w - 1))
    options = [
        (lv, lh)
        for lv in range(3, h + 1)
        if _containing_lows(h, row, lv)
        for lh in range(3, w + 1)
        if _containing_lows(w, col, lh)
        and (short_vertical is None or (lv < lh) == short_vertical)
        and (short_vertical is None or lv != lh)
    ]
    lv, lh = options[int(rng.integers(0, len(options)))]
    vlows = tuple(_containing_lows(h, row, lv))
    hlows = tuple(_containing_lows(w, col, lh))
    vlo = vlows[int(rng.integers(0, len(vlows)))]
    hlo = hlows[int(rng.integers(0, len(hlows)))]
    return [
        ("v", c0 + col, r0 + vlo, r0 + vlo + lv - 1),
        ("h", r0 + row, c0 + hlo, c0 + hlo + lh - 1),
    ]


def _network_core(rng, height, width, count, fan=False):
    """A complete rectangular crossing network or a one-line fan."""
    if fan:
        many = int(rng.integers(2, min(4, count - 1) + 1))
        nv, nh = (1, many) if bool(rng.integers(0, 2)) else (many, 1)
    else:
        nv = int(rng.integers(2, min(3, count - 2) + 1))
        nh = int(rng.integers(2, min(3, count - nv) + 1))
    cols = sorted(int(x) for x in rng.choice(np.arange(1, width - 1), nv, replace=False))
    rows = sorted(int(x) for x in rng.choice(np.arange(1, height - 1), nh, replace=False))
    segments = []
    for col in cols:
        lo = int(rng.integers(0, rows[0]))
        hi = int(rng.integers(rows[-1] + 1, height))
        segments.append(("v", col, lo, hi))
    for row in rows:
        lo = int(rng.integers(0, cols[0]))
        hi = int(rng.integers(cols[-1] + 1, width))
        segments.append(("h", row, lo, hi))
    return segments


def _compatible(segment, existing):
    orient, fixed, lo, hi = segment
    for other_orient, other_fixed, other_lo, other_hi in existing:
        if orient == other_orient:
            # Same-line intervals must have a visible background gap.
            if fixed == other_fixed and not (hi + 1 < other_lo or other_hi + 1 < lo):
                return False
            continue
        if orient == "v":
            col, vlo, vhi = fixed, lo, hi
            row, hlo, hhi = other_fixed, other_lo, other_hi
        else:
            col, vlo, vhi = other_fixed, other_lo, other_hi
            row, hlo, hhi = fixed, lo, hi
        if hlo <= col <= hhi and vlo <= row <= vhi:
            # Crossings are interior to both segments, leaving endpoints visible.
            if not (hlo < col < hhi and vlo < row < vhi):
                return False
    return True


def _random_segment(rng, orientation, height, width, span_bias):
    fixed_size, extent = (width, height) if orientation == "v" else (height, width)
    if rng.random() < 0.23:
        fixed = 0 if bool(rng.integers(0, 2)) else fixed_size - 1
    else:
        fixed = int(rng.integers(0, fixed_size))
    roll = rng.random()
    if roll < 0.20:
        length = 3
    elif roll < 0.20 + span_bias * 0.65:
        length = int(rng.integers(max(3, (extent + 1) // 2), extent + 1))
    else:
        length = int(rng.integers(3, extent + 1))
    if rng.random() < 0.34:
        lo = 0 if bool(rng.integers(0, 2)) else extent - length
    else:
        lo = int(rng.integers(0, extent - length + 1))
    return orientation, fixed, lo, lo + length - 1


def _sample_plan(rng, regime, index, profile):
    count = int(rng.integers(1, 10))
    if regime == "source_train" and index < 2:
        count = max(2, count)
    elif profile == 0 and regime == "target_test":
        count = max(count, int(rng.integers(4, 10)))
    elif profile == 0:
        count = min(count, int(rng.integers(2, 6)))
    elif profile == 2 and regime == "target_test":
        count = min(count, int(rng.integers(1, 6)))
    elif profile == 2:
        count = max(count, int(rng.integers(5, 10)))

    if regime == "source_train" and index < 2:
        style = "demo"
    else:
        choices = ["plain"]
        if count >= 2:
            choices.append("demo")
        if count >= 3:
            choices.append("fan")
        if count >= 4:
            choices.extend(("network", "two_groups"))
        style = choices[int(rng.integers(0, len(choices)))]
        if profile == 0 and regime == "target_test" and count >= 4 and rng.random() < 0.40:
            style = "network"

    span_bias = float(rng.random())
    if profile == 0 and regime == "target_test":
        span_bias = max(span_bias, 0.55)
    if profile == 2 and regime == "target_test":
        span_bias *= 0.55
    return count, style, span_bias


def _sample_geometry(rng, height, width, count, style, span_bias, short_vertical):
    if style == "demo":
        segments = _demo_core(rng, height, width, short_vertical)
    elif style == "fan":
        segments = _network_core(rng, height, width, count, fan=True)
    elif style == "network":
        segments = _network_core(rng, height, width, count)
    elif style == "two_groups":
        h0, w0 = height // 2, width // 2
        if bool(rng.integers(0, 2)):
            boxes = ((0, 0, h0, w0), (h0, w0, height - h0, width - w0))
        else:
            boxes = ((0, w0, h0, width - w0), (h0, 0, height - h0, w0))
        segments = _demo_core(rng, height, width, box=boxes[0])
        segments += _demo_core(rng, height, width, box=boxes[1])
    else:
        segments = []

    # Orientation count is independent of pair index. An existing crossing
    # core imposes only the necessary minimum of each orientation.
    remaining = count - len(segments)
    target_verticals = int(rng.integers(0, remaining + 1))
    orientations = ["v"] * target_verticals + ["h"] * (remaining - target_verticals)
    rng.shuffle(orientations)
    for orientation in orientations:
        for _ in range(320):
            candidate = _random_segment(rng, orientation, height, width, span_bias)
            if _compatible(candidate, segments):
                segments.append(candidate)
                break
    assert len(segments) >= max(1, 2 if style == "demo" else 0)
    return segments


def _generate_pair(rng, regime, index, profile, background):
    count, style, span_bias = _sample_plan(rng, regime, index, profile)
    lower = max(5, count + 3, 8 if style in ("demo", "network", "fan") else 5)
    if style == "two_groups":
        lower = max(lower, 12)
    height = _dimension(rng, lower)
    width = _dimension(rng, lower)
    short_vertical = None
    if regime == "source_train" and index < 2:
        short_vertical = index == 0
    segments = _sample_geometry(rng, height, width, count, style, span_bias, short_vertical)
    palette = [color for color in range(10) if color != background]
    colors = [int(c) for c in rng.choice(palette, len(segments), replace=False)]
    inp = np.full((height, width), background, dtype=np.int8)
    out = np.full((height, width), background, dtype=np.int8)
    for (orientation, fixed, lo, hi), color in zip(segments, colors):
        if orientation == "v":
            inp[lo, fixed] = inp[hi, fixed] = color
        else:
            inp[fixed, lo] = inp[fixed, hi] = color
    for (orientation, fixed, lo, hi), color in zip(segments, colors):
        if orientation == "h":
            out[fixed, lo:hi + 1] = color
    for (orientation, fixed, lo, hi), color in zip(segments, colors):
        if orientation == "v":
            out[lo:hi + 1, fixed] = color
    return {"input": inp, "output": out}


def _grid_key(grid):
    return grid.shape, tuple(int(v) for v in grid.flat)


def _input_key(inp, background):
    points = tuple(sorted((int(inp[r, c]), int(r), int(c))
                          for r, c in np.argwhere(inp != background)))
    return inp.shape, background, points


def _make_unique_pair(rng, regime, index, profile, background, seen_inputs, exclude_shapes):
    for _ in range(80):
        pair = _generate_pair(rng, regime, index, profile, background)
        key = _grid_key(pair["input"])
        if (key not in seen_inputs and pair["input"].shape not in exclude_shapes
                and _input_key(pair["input"], background) not in _SOURCE_INPUT_KEYS):
            seen_inputs.add(key)
            return pair
    raise ValueError("could not construct a distinct endpoint scene")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    # The only learned crossing convention is the source's fixed vertical
    # foreground. Profile changes complexity, not the transformation rule.
    profile = int(rng.integers(0, 3))
    total = train_count + test_count
    raw_seeds = rng.integers(0, np.iinfo(np.uint64).max, size=total, dtype=np.uint64)
    backgrounds = [int(rng.integers(0, 10))]
    backgrounds.append(int(rng.choice([c for c in range(10) if c != backgrounds[0]])))
    backgrounds += [int(rng.integers(0, 10)) for _ in range(total - 2)]
    seen_inputs = set()
    train, test = [], []
    for index in range(total):
        regime = "source_train" if index < train_count else "target_test"
        local_index = index if regime == "source_train" else index - train_count
        pair_rng = np.random.default_rng([int(raw_seeds[index]), index, int(seed) % (2**32)])
        exclude_shapes = {train[0]["input"].shape} if index == 1 else set()
        pair = _make_unique_pair(pair_rng, regime, local_index, profile,
                                 backgrounds[index], seen_inputs, exclude_shapes)
        (train if regime == "source_train" else test).append(pair)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(inp, out):
    """Reconstruct the answer from visible colored pairs, independent of sampling."""
    assert inp.shape == out.shape
    values, counts = np.unique(inp, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    assert int(np.max(counts)) > 2
    assert 2 <= len(values) <= 10
    verticals, horizontals = [], []
    for value, amount in zip(values, counts):
        color = int(value)
        if color == background:
            continue
        assert int(amount) == 2
        (r0, c0), (r1, c1) = np.argwhere(inp == color)
        r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)
        assert (r0 == r1) != (c0 == c1)
        if c0 == c1:
            lo, hi = sorted((r0, r1))
            verticals.append((color, c0, lo, hi))
        else:
            lo, hi = sorted((c0, c1))
            horizontals.append((color, r0, lo, hi))
        assert hi - lo >= 2
    assert 1 <= len(verticals) + len(horizontals) <= 9

    for lines in (verticals, horizontals):
        for i, (_, fixed, lo, hi) in enumerate(lines):
            for _, other_fixed, other_lo, other_hi in lines[i + 1:]:
                assert fixed != other_fixed or hi + 1 < other_lo or other_hi + 1 < lo

    crossings = []
    for vcolor, col, vlo, vhi in verticals:
        for hcolor, row, hlo, hhi in horizontals:
            if hlo <= col <= hhi and vlo <= row <= vhi:
                assert hlo < col < hhi and vlo < row < vhi
                assert int(inp[row, col]) == background
                crossings.append((vhi - vlo + 1, hhi - hlo + 1, row, col))

    expected = np.full(inp.shape, background, dtype=inp.dtype)
    for color, row, lo, hi in horizontals:
        expected[row, lo:hi + 1] = color
    for color, col, lo, hi in verticals:
        expected[lo:hi + 1, col] = color
    assert np.array_equal(expected, out)
    assert not np.array_equal(inp, out)
    return {
        "background": background,
        "verticals": verticals,
        "horizontals": horizontals,
        "crossings": crossings,
    }


def _topology(analysis):
    verticals = analysis["verticals"]
    horizontals = analysis["horizontals"]
    edges = set()
    for _, _, row, col in analysis["crossings"]:
        vi = next(i for i, (_, fixed, lo, hi) in enumerate(verticals)
                  if fixed == col and lo < row < hi)
        hj = next(j for j, (_, fixed, lo, hi) in enumerate(horizontals)
                  if fixed == row and lo < col < hi)
        edges.add((vi, hj))
    vdegree = [sum(i == vi for i, _ in edges) for vi in range(len(verticals))]
    hdegree = [sum(j == hj for _, j in edges) for hj in range(len(horizontals))]
    remaining = set(edges)
    groups = 0
    while remaining:
        groups += 1
        vi, hj = remaining.pop()
        active_v, active_h = {vi}, {hj}
        changed = True
        while changed:
            attached = {(v, h) for v, h in remaining if v in active_v or h in active_h}
            changed = bool(attached)
            remaining -= attached
            active_v.update(v for v, _ in attached)
            active_h.update(h for _, h in attached)
    return {
        "groups": groups,
        "fan_v": max(vdegree, default=0),
        "fan_h": max(hdegree, default=0),
        "network": sum(n >= 2 for n in vdegree) >= 2
                   and sum(n >= 2 for n in hdegree) >= 2,
        "uncrossed": sum(n == 0 for n in vdegree + hdegree),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    training_orders, training_backgrounds, training_shapes = set(), set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            assert inp.shape == out.shape
            key = _grid_key(inp)
            pair_key = key, _grid_key(out)
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            analysis = _analyze_pair(inp, out)
            assert _input_key(inp, analysis["background"]) not in _SOURCE_INPUT_KEYS
            if split == "train":
                training_backgrounds.add(analysis["background"])
                training_shapes.add(inp.shape)
                for lv, lh, _, _ in analysis["crossings"]:
                    training_orders.add(-1 if lv < lh else 1 if lv > lh else 0)
    # Both length orders distinguish vertical foreground from longer wins.
    assert {-1, 1} <= training_orders
    assert len(training_backgrounds) >= 2
    assert len(training_shapes) >= 2


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k])
        for x, y in zip(a[s], b[s]) for k in ("input", "output")
    ) for s in ("train", "test"))


def _must_raise(**kwargs):
    try:
        generate_puzzle(919191, **kwargs)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def self_test() -> None:
    automatic = generate_puzzle(12345)
    explicit = generate_puzzle(54321, 6, 3)
    validate_generated_puzzle(automatic)
    validate_generated_puzzle(explicit)
    assert _puzzles_equal(explicit, generate_puzzle(54321, 6, 3))
    assert not _puzzles_equal(explicit, generate_puzzle(54322, 6, 3))
    for train, test in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(70000 + train * 10 + test, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
        validate_generated_puzzle(puzzle)
    for offset, (train, test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(80000 + offset, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
        validate_generated_puzzle(puzzle)
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        _must_raise(**kwargs)

    corpus = [generate_puzzle(seed) for seed in range(4200, 4242)]
    assert len({len(p["train"]) for p in corpus}) >= 4
    assert len({len(p["test"]) for p in corpus}) >= 3
    signatures, shapes, layouts, colors, counts = set(), set(), set(), set(), set()
    split_aspects = {"train": set(), "test": set()}
    within = {"pair_count": 0, "orientation": 0, "aspect": 0,
              "crossings": 0, "sparse_to_dense": 0, "rich_to_simple": 0}
    seen = {"single": 0, "all_one_orientation": 0, "uncrossed": 0,
            "network": 0, "same_line": 0, "edge_line": 0,
            "corner": 0, "three_cell": 0, "short_vertical": 0,
            "long_vertical": 0, "zero_connector": 0, "all_vertical": 0,
            "all_horizontal": 0, "multiple_groups": 0,
            "fan_vertical": 0, "fan_horizontal": 0}
    for puzzle in corpus:
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(tuple(_grid_key(pair["input"]) for pair in puzzle[split])
                             for split in ("train", "test")))
        puzzle_features = []
        crossing_means = {}
        for split in ("train", "test"):
            split_crossings = []
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                analysis = _analyze_pair(inp, out)
                topology = _topology(analysis)
                v, h = analysis["verticals"], analysis["horizontals"]
                all_lines = v + h
                n = len(all_lines)
                aspect = 0 if inp.shape[0] == inp.shape[1] else (
                    1 if inp.shape[0] > inp.shape[1] else -1)
                split_aspects[split].add(aspect)
                puzzle_features.append((n, len(v), aspect, len(analysis["crossings"])))
                split_crossings.append(len(analysis["crossings"]))
                counts.add(n)
                shapes.add(inp.shape)
                colors.add(analysis["background"])
                layouts.add((inp.shape, tuple(sorted((int(r), int(c)) for r, c in np.argwhere(
                    inp != analysis["background"])))))
                seen["single"] += n == 1
                seen["all_one_orientation"] += not v or not h
                seen["uncrossed"] += not analysis["crossings"]
                seen["network"] += len(analysis["crossings"]) >= 4
                seen["multiple_groups"] += topology["groups"] >= 2
                seen["fan_vertical"] += topology["fan_v"] >= 3
                seen["fan_horizontal"] += topology["fan_h"] >= 3
                seen["all_vertical"] += not h
                seen["all_horizontal"] += not v
                seen["same_line"] += len({s[1] for s in v}) < len(v) or len({s[1] for s in h}) < len(h)
                seen["edge_line"] += any(s[1] in (0, inp.shape[1] - 1) for s in v) or any(
                    s[1] in (0, inp.shape[0] - 1) for s in h)
                seen["corner"] += any(inp[r, c] != analysis["background"] for r, c in
                                      ((0, 0), (0, inp.shape[1] - 1),
                                       (inp.shape[0] - 1, 0), (inp.shape[0] - 1, inp.shape[1] - 1)))
                seen["three_cell"] += any(s[3] - s[2] == 2 for s in all_lines)
                seen["short_vertical"] += any(lv < lh for lv, lh, _, _ in analysis["crossings"])
                seen["long_vertical"] += any(lv > lh for lv, lh, _, _ in analysis["crossings"])
                seen["zero_connector"] += analysis["background"] != 0 and bool(np.any(inp == 0))
            crossing_means[split] = sum(split_crossings) / len(split_crossings)
        within["pair_count"] += len({feature[0] for feature in puzzle_features}) > 1
        within["orientation"] += len({feature[1] for feature in puzzle_features}) > 1
        within["aspect"] += len({feature[2] for feature in puzzle_features}) > 1
        within["crossings"] += len({feature[3] for feature in puzzle_features}) > 1
        within["sparse_to_dense"] += crossing_means["test"] > crossing_means["train"]
        within["rich_to_simple"] += crossing_means["test"] < crossing_means["train"]
    assert len(signatures) >= 40 and len(shapes) >= 35 and len(layouts) >= 100
    assert len(colors) == 10 and counts == set(range(1, 10))
    assert split_aspects["train"] == {-1, 0, 1}
    assert split_aspects["test"] == {-1, 0, 1}
    assert all(within[name] >= 10 for name in within)
    assert all(shape in shapes for shape in ((20, 10), (30, 20), (20, 20)))
    for name in seen:
        assert seen[name] > 0, name
