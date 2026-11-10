"""Generate straight tracks whose paired perpendicular bars move to the centre."""

import numpy as np


def _render(s):
    h, w = s["shape"]
    g = np.full((h, w), s["bg"], dtype=np.int8)
    f, start, length = s["fixed"], s["track_start"], s["length"]
    a = start + s["bar_start"]
    b = a + s["width"]
    if s["orientation"] == "H":
        g[f, start:start + length] = s["track"]
        for d in range(1, s["depth"] + 1):
            g[f - d, a:b] = s["bar"]
            g[f + d, a:b] = s["bar"]
    else:
        g[start:start + length, f] = s["track"]
        for d in range(1, s["depth"] + 1):
            g[a:b, f - d] = s["bar"]
            g[a:b, f + d] = s["bar"]
    return g


def _decode(g):
    """Recover every property from the visible grid, without generation state."""
    values, counts = np.unique(g, return_counts=True)
    assert len(values) == 3
    bg = int(values[np.argmax(counts)])
    assert counts.max() > g.size // 2
    colors = [int(v) for v in values if v != bg]
    matches = []
    for track, bar in (colors, colors[::-1]):
        rows, cols = np.where(g == track)
        bar_rows, bar_cols = np.where(g == bar)
        for orientation in ("H", "V"):
            if orientation == "H":
                if not np.all(rows == rows[0]):
                    continue
                fixed = int(rows[0])
                axis = np.sort(cols)
                bar_axis = np.unique(bar_cols)
                distances = np.abs(bar_rows - fixed)
                perp = g.shape[0]
            else:
                if not np.all(cols == cols[0]):
                    continue
                fixed = int(cols[0])
                axis = np.sort(rows)
                bar_axis = np.unique(bar_rows)
                distances = np.abs(bar_cols - fixed)
                perp = g.shape[1]
            length, width = len(axis), len(bar_axis)
            if length < 4 or not 1 <= width <= length - 2:
                continue
            if not np.array_equal(axis, np.arange(axis[0], axis[0] + length)):
                continue
            if not np.array_equal(bar_axis, np.arange(bar_axis[0], bar_axis[0] + width)):
                continue
            start = int(axis[0])
            bar_start = int(bar_axis[0]) - start
            depth = int(distances.max())
            if not 0 <= bar_start <= length - width:
                continue
            if depth < 1 or fixed - depth < 0 or fixed + depth >= perp:
                continue
            s = dict(shape=g.shape, bg=bg, track=track, bar=bar,
                     orientation=orientation, fixed=fixed, track_start=start,
                     length=length, bar_start=bar_start, width=width, depth=depth)
            if np.array_equal(_render(s), g):
                matches.append(s)
    assert len(matches) == 1
    return matches[0]


def _signature(s):
    return tuple(s[k] for k in ("shape", "bg", "track", "bar", "orientation",
                                "fixed", "track_start", "length", "bar_start",
                                "width", "depth"))


_ORIGINAL_INPUTS = {
    ((10, 10), 0, 1, 3, "H", 3, 0, 7, 2, 1, 1),
    ((10, 10), 0, 2, 4, "V", 1, 3, 7, 0, 1, 1),
    ((10, 10), 0, 6, 8, "H", 5, 1, 7, 6, 1, 1),
    ((7, 7), 0, 8, 2, "H", 3, 0, 7, 1, 1, 1),
    ((10, 10), 0, 3, 7, "V", 7, 1, 5, 3, 1, 1),
    ((10, 10), 0, 4, 1, "H", 2, 0, 10, 1, 2, 2),
}


def _lengths(axis, area, width, depth):
    return [n for n in range(max(4, width + 2), axis + 1)
            if (n - width) % 2 == 0 and n + 2 * width * depth < area // 2]


def _sample_pair(rng, regime, orientation, side, bar_width, depth, bg,
                 required_length=None):
    bands = ((7, 10), (9, 15), (14, 20))
    band = int(rng.choice(3, p=(0.30, 0.40, 0.30)))
    rectangle = rng.random() < 0.65

    def eligible(low, high, want_rectangle):
        result = []
        for h in range(low, high + 1):
            for w in range(low, high + 1):
                if want_rectangle and abs(h - w) < 2:
                    continue
                axis = w if orientation == "H" else h
                perp = h if orientation == "H" else w
                if perp < 2 * depth + 1:
                    continue
                lengths = _lengths(axis, h * w, bar_width, depth)
                if required_length is not None:
                    lengths = [n for n in lengths if n == required_length]
                if lengths:
                    result.append((h, w))
        return result

    low, high = bands[band]
    choices = eligible(low, high, rectangle)
    if not choices:
        choices = eligible(low, high, False)
    if not choices:
        choices = eligible(7, 20, rectangle)
    if not choices:
        choices = eligible(7, 20, False)
    assert choices
    h, w = choices[int(rng.integers(len(choices)))]
    axis = w if orientation == "H" else h
    perp = h if orientation == "H" else w
    lengths = _lengths(axis, h * w, bar_width, depth)
    if required_length is not None:
        length = required_length
    elif regime == "target_test" and rng.random() < 0.55:
        length = int(rng.choice(lengths[len(lengths) // 2:]))
    else:
        length = int(rng.choice(lengths))
    center = (length - bar_width) // 2
    assert center >= 1
    roll = rng.random()
    if roll < 0.30:
        steps = center
    elif roll < 0.52:
        steps = 1
    else:
        steps = int(rng.integers(1, center + 1))
    bar_start = center + side * steps
    track_limit = axis - length
    if track_limit and rng.random() < 0.32:
        track_start = int(rng.choice((0, track_limit)))
    else:
        track_start = int(rng.integers(0, track_limit + 1))
    if rng.random() < 0.32:
        fixed = int(rng.choice((depth, perp - depth - 1)))
    else:
        fixed = int(rng.integers(depth, perp - depth))
    colors = [c for c in range(10) if c != bg]
    track, bar = (int(v) for v in rng.choice(colors, 2, replace=False))
    s = dict(shape=(h, w), bg=bg, track=track, bar=bar,
             orientation=orientation, fixed=fixed, track_start=track_start,
             length=length, bar_start=bar_start, width=bar_width, depth=depth)
    return {"input": _render(s), "output": _render(dict(s, bar_start=center))}


def _grid_key(g):
    return g.shape, tuple(int(v) for v in g.flat)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    if num_train is not None and (type(num_train) is not int or not 2 <= num_train <= 10):
        raise ValueError("num_train must be an integer from 2 through 10")
    if num_test is not None and (type(num_test) is not int or not 1 <= num_test <= 4):
        raise ValueError("num_test must be an integer from 1 through 4")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")

    # No hidden color rule is needed: each grid visibly identifies its roles.
    shared_bg = int(rng.integers(10)) if rng.random() < 0.42 else None
    backgrounds = [shared_bg if shared_bg is not None else int(rng.integers(10))
                   for _ in range(num_train + num_test)]
    if shared_bg is None and len(set(backgrounds[:num_train])) == 1:
        backgrounds[1] = (backgrounds[0] + int(rng.integers(1, 10))) % 10
    anchor_orientation = str(rng.choice(("H", "V")))
    side = int(rng.choice((-1, 1)))
    varied_width = int(rng.integers(1, 5))
    varied_depth = int(rng.integers(1, 4))
    if (varied_width, varied_depth) == (1, 1):
        if rng.random() < 0.5:
            varied_width = int(rng.integers(2, 5))
        else:
            varied_depth = int(rng.integers(2, 4))
    plans = [(anchor_orientation, side, 1, 1),
             (anchor_orientation, -side, varied_width, varied_depth)]
    rng.shuffle(plans)
    if num_train >= 3:
        other = "V" if anchor_orientation == "H" else "H"
        plans.append((other, int(rng.choice((-1, 1))),
                      int(rng.integers(1, 5)), int(rng.integers(1, 4))))
    while len(plans) < num_train:
        plans.append((str(rng.choice(("H", "V"))), int(rng.choice((-1, 1))),
                      int(rng.integers(1, 5)), int(rng.integers(1, 4))))

    puzzle = {"train": [], "test": []}
    seen = set()

    def append(regime, orientation, direction, width, depth, bg, length=None):
        dest = "train" if regime == "source_train" else "test"
        for _ in range(80):
            pair = _sample_pair(rng, regime, orientation, direction, width, depth,
                                bg, length)
            key = _grid_key(pair["input"])
            if key not in seen and _signature(_decode(pair["input"])) not in _ORIGINAL_INPUTS:
                puzzle[dest].append(pair)
                seen.add(key)
                return
        raise AssertionError("Could not construct a distinct non-source pair")

    for i, (orientation, direction, width, depth) in enumerate(plans):
        append("source_train", orientation, direction, width, depth, backgrounds[i])
    train_specs = [_decode(pair["input"]) for pair in puzzle["train"]]
    for i in range(num_test):
        choice = rng.random()
        length = None
        if choice < 0.30:
            width, depth = 1, 1
        elif choice < 0.58:
            example = train_specs[int(rng.integers(len(train_specs)))]
            width, depth, length = example["width"], example["depth"], example["length"]
        else:
            width = int(rng.integers(1, 5))
            depth = int(rng.integers(1, 4))
            if (width, depth) == (1, 1):
                width = int(rng.integers(2, 5))
        orientation = (anchor_orientation if num_train == 2
                       else str(rng.choice(("H", "V"))))
        append("target_test", orientation, int(rng.choice((-1, 1))), width, depth,
               backgrounds[num_train + i], length)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs, training = set(), set(), []
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for g in pair.values():
                assert isinstance(g, np.ndarray) and g.ndim == 2
                assert all(1 <= n <= 30 for n in g.shape)
                assert np.issubdtype(g.dtype, np.integer)
                assert np.all((0 <= g) & (g <= 9))
            before, after = pair["input"], pair["output"]
            assert before.shape == after.shape
            a, b = _grid_key(before), _grid_key(after)
            assert a not in inputs and (a, b) not in pairs
            inputs.add(a)
            pairs.add((a, b))
            spec = _decode(before)
            assert _signature(spec) not in _ORIGINAL_INPUTS
            assert (spec["length"] - spec["width"]) % 2 == 0
            center = (spec["length"] - spec["width"]) // 2
            assert spec["bar_start"] != center
            assert np.array_equal(after, _render(dict(spec, bar_start=center)))
            assert _decode(after) == dict(spec, bar_start=center)
            if regime == "train":
                training.append(spec)
    assert len({(s["width"], s["depth"]) for s in training}) >= 2
    assert any((s["width"], s["depth"]) == (1, 1) for s in training)
    if len(training) >= 3:
        assert {s["orientation"] for s in training} == {"H", "V"}
    assert any(
        a["orientation"] == b["orientation"]
        and (a["bar_start"] - (a["length"] - a["width"]) // 2)
        * (b["bar_start"] - (b["length"] - b["width"]) // 2) < 0
        for i, a in enumerate(training) for b in training[i + 1:])


def _puzzle_key(p):
    return tuple(tuple((_grid_key(q["input"]), _grid_key(q["output"]))
                       for q in p[r]) for r in ("train", "test"))


def self_test() -> None:
    puzzles, shapes, layouts, palettes, combos = set(), set(), set(), set(), set()
    train_counts, test_counts = set(), set()
    mixed, train_rectangles, wide_vertical = 0, 0, 0
    short_moves, long_moves, track_edges = 0, 0, 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        key = _puzzle_key(p)
        assert key == _puzzle_key(generate_puzzle(seed))
        puzzles.add(key)
        train_counts.add(len(p["train"]))
        test_counts.add(len(p["test"]))
        train_specs = [_decode(q["input"]) for q in p["train"]]
        mixed += len({s["bg"] for s in train_specs}) > 1
        train_rectangles += any(s["shape"][0] != s["shape"][1] for s in train_specs)
        for regime in ("train", "test"):
            for q in p[regime]:
                s = _decode(q["input"])
                shapes.add(s["shape"])
                palettes.add((s["bg"], s["track"], s["bar"]))
                combos.add((s["orientation"], s["width"], s["depth"]))
                layouts.add((s["shape"], s["orientation"], s["fixed"],
                             s["track_start"], s["length"], s["bar_start"],
                             s["width"], s["depth"]))
                wide_vertical += s["orientation"] == "V" and s["width"] >= 2
                steps = abs(s["bar_start"] - (s["length"] - s["width"]) // 2)
                short_moves += steps == 1
                long_moves += steps >= 2
                axis = s["shape"][1 if s["orientation"] == "H" else 0]
                track_edges += s["track_start"] == 0 or s["track_start"] + s["length"] == axis
    assert len(puzzles) == 42
    assert len(train_counts) >= 5 and len(test_counts) == 4
    assert len(shapes) >= 45 and len(layouts) >= 150 and len(palettes) >= 80
    assert ("H", 1, 2) in combos and ("V", 1, 2) in combos
    assert ("H", 3, 1) in combos and ("V", 3, 1) in combos
    assert wide_vertical >= 10 and train_rectangles >= 20
    assert mixed >= 15 and short_moves >= 30 and long_moves >= 30
    assert track_edges >= 50
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain - 1) + 1):
            p = generate_puzzle(12345, ntrain, ntest)
            validate_generated_puzzle(p)
            assert (len(p["train"]), len(p["test"])) == (ntrain, ntest)
    for a, b in ((None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(37, a, b))
    for a, b in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(3, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts were accepted")
