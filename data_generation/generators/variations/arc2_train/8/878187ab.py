import numpy as np


_SOURCE_INPUT_TEXTS = (
    """7877777777777777
7777777777778777
7777778777777777
7677777777776777
7776777778777777
7777777777777777
7777777777777777
7777777777777777
7777778777777777
7777777777777777
7777777777777787
7778776777777777
7777777776777787
7777777787777777
7777777777777777
7777777777777787""",
    """777777777777777
747777777777777
777777777777774
777777774577777
777777777777777
777777777777777
777777777777777
777475777777774
777777775777777
777777777777777
777777777775777
777777775777777
747777477777777
777777777777777
777777777777777""",
    """7777777777757777
7777777777077707
7775777777077770
7777757777777777
7777777770707777
7777777577777777
7777777777777777
7577777707070577
7777707777577770
7777777777777777
7577777777075777
7757777777757777
7707777777777777
7770777777777775
7777777777777777
7777777777777777""",
)
_SOURCE_INPUTS = tuple(
    np.array([[int(c) for c in row] for row in text.splitlines()], dtype=np.int8)
    for text in _SOURCE_INPUT_TEXTS
)


def _fingerprint(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _source_copy(grid):
    return any(np.array_equal(grid, source) for source in _SOURCE_INPUTS)


def _count_argument(value, name, low, high):
    if value is not None and (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not low <= int(value) <= high
    ):
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


def _pair_counts(seed_key, num_train, num_test):
    _count_argument(num_train, "num_train", 2, 10)
    _count_argument(num_test, "num_test", 1, 4)
    train = None if num_train is None else int(num_train)
    test = None if num_test is None else int(num_test)
    if train is None and test is None:
        test = 1 + seed_key % 4
        low = max(2, test + 1)
        train = low + (seed_key // 4) % (11 - low)
    elif train is None:
        low = max(2, test + 1)
        train = low + seed_key % (11 - low)
    elif test is None:
        test = 1 + seed_key % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _output(shape, height, width, background, body, accent):
    rows, columns = shape
    out = np.full(shape, background, dtype=np.int8)
    out[rows - height:, :width] = body
    for d in range(height):
        out[rows - 1 - d, d] = accent
        out[rows - 1 - d, width - 1 - d] = accent
    return out


def _input_facts(grid):
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 3
    bg_index = int(np.argmax(counts))
    bg = int(colors[bg_index])
    signals = [(int(colors[i]), int(counts[i])) for i in range(3) if i != bg_index]
    assert int(counts[bg_index]) > sum(count for _, count in signals)
    assert signals[0][1] != signals[1][1]
    height, width = sorted(count for _, count in signals)
    return bg, height, width, signals


def _canvas(rng):
    if int(rng.integers(3)) == 0:
        return (16, 16)
    rows = int(rng.integers(16, 25))
    columns = int(rng.integers(16, 25))
    if rows == columns:
        columns = 16 if columns == 24 else columns + 1
    return rows, columns


def _train_shapes(rng, canvas, count):
    rows, columns = canvas
    choices = [
        (h, w)
        for h in range(3, rows - 2)
        for w in range(h + 1, columns - 2)
    ]
    # The two controls have the same total, but different frequencies and
    # noticeably different diagonal behavior.
    controls = [
        (a, b)
        for a in choices for b in choices
        if a[0] + a[1] == b[0] + b[1]
        and 12 <= a[0] + a[1] <= 21
        and b[0] - a[0] >= 2
        and a[0] <= a[1] // 2
        and b[0] > (b[1] + 1) // 2
    ]
    first, second = controls[int(rng.integers(len(controls)))]
    shapes = [first, second]
    if int(rng.integers(2)):
        shapes.reverse()
    if count >= 3:
        shapes.append(shapes[0])  # same counts, new colors and layout
    for index in range(3, count):
        desired = index % 3
        pool = [
            shape for shape in choices if shape not in shapes
            and (0 if shape[0] * shape[1] <= 45 else
                 1 if shape[0] * shape[1] <= 90 else 2) == desired
        ]
        if not pool:
            pool = [shape for shape in choices if shape not in shapes]
        shapes.append(pool[int(rng.integers(len(pool)))])
    return shapes


def _test_shapes(rng, seed_key, canvas, train_shapes, count):
    rows, columns = canvas
    possible = [(h, w) for h in range(3, rows) for w in range(h + 1, columns + 1)]
    low_h = min(h for h, _ in train_shapes)
    high_h = max(h for h, _ in train_shapes)
    low_w = min(w for _, w in train_shapes)
    high_w = max(w for _, w in train_shapes)
    result = []
    for index in range(count):
        category = (seed_key + index) % 3
        if category == 0:  # interpolation, sometimes an exact count match
            pool = [s for s in possible if low_h <= s[0] <= high_h
                    and low_w <= s[1] <= high_w]
            if index == 0 and int(rng.integers(2)):
                pool = list(train_shapes)
        elif category == 1:  # one dimension beyond the demonstrations
            pool = [s for s in possible if
                    (s[0] > high_h and low_w <= s[1] <= high_w)
                    or (s[1] > high_w and low_h <= s[0] <= high_h)]
        else:  # both dimensions beyond the demonstrations
            pool = [s for s in possible if s[0] > high_h and s[1] > high_w]
        if not pool:
            pool = [s for s in possible if s not in train_shapes]
        fresh = [s for s in pool if s not in result]
        if fresh:
            pool = fresh
        result.append(pool[int(rng.integers(len(pool)))])
    return result


def _input_shape(rng, count):
    options = [(h, w) for h in range(12, 28) for w in range(12, 28)
               if h * w >= 4 * count]
    return options[int(rng.integers(len(options)))]


def _positions(rng, shape, count, mode):
    rows, columns = shape
    total = rows * columns
    all_positions = np.arange(total)
    rr = all_positions // columns
    cc = all_positions % columns
    if mode == 0:  # plain scatter
        selected = rng.choice(total, size=count, replace=False)
    elif mode == 1:  # local concentration, with a scattered tail
        ar = int(rng.integers(rows))
        ac = int(rng.integers(columns))
        distance = np.abs(rr - ar) + np.abs(cc - ac)
        weights = 0.16 + 1.0 / (1.0 + distance) ** 2
        selected = rng.choice(total, size=count, replace=False, p=weights / weights.sum())
    elif mode == 2:  # short horizontal or vertical runs
        chosen = set()
        for _ in range(3):
            r = int(rng.integers(rows))
            c = int(rng.integers(columns))
            length = int(rng.integers(2, 6))
            vertical = int(rng.integers(2))
            for d in range(length):
                chosen.add(((r + d * vertical) % rows) * columns
                           + (c + d * (1 - vertical)) % columns)
        rest = np.array([p for p in all_positions if int(p) not in chosen])
        if len(chosen) < count:
            extra = rng.choice(rest, size=count - len(chosen), replace=False)
            chosen.update(int(p) for p in extra)
        selected = np.array(list(chosen), dtype=int)
        if len(selected) > count:
            selected = rng.choice(selected, size=count, replace=False)
    elif mode == 3:  # corners and edges
        edge = np.minimum.reduce((rr, rows - 1 - rr, cc, columns - 1 - cc))
        corner = np.minimum.reduce((rr + cc, rr + columns - 1 - cc,
                                    rows - 1 - rr + cc,
                                    rows + columns - 2 - rr - cc))
        weights = 0.2 + 2.0 / (1.0 + edge) + 3.0 / (1.0 + corner)
        selected = rng.choice(total, size=count, replace=False, p=weights / weights.sum())
    else:  # offset checkerboard bias
        parity = int(rng.integers(2))
        weights = np.where((rr + cc) % 2 == parity, 4.0, 1.0)
        selected = rng.choice(total, size=count, replace=False, p=weights / weights.sum())
    rng.shuffle(selected)
    return selected


def _palette(rng, available, overlap, different_from=None):
    pairs = [(a, b) for i, a in enumerate(available) for b in available[i + 1:]]
    if overlap:
        special = [p for p in pairs if p[0] in overlap or p[1] in overlap]
        if special:
            pairs = special
    if different_from is not None:
        distinct = [p for p in pairs if frozenset(p) != frozenset(different_from)]
        if distinct:
            pairs = distinct
    return pairs[int(rng.integers(len(pairs)))]


def _pair(rng, canvas, shape, background, body, accent, palette, mode,
          small_color=None):
    height, width = shape
    grid_shape = _input_shape(rng, height + width)
    if small_color is None:
        small_color = palette[int(rng.integers(2))]
    large_color = next(color for color in palette if color != small_color)
    for _ in range(8):
        grid = np.full(grid_shape, background, dtype=np.int8)
        positions = _positions(rng, grid_shape, height + width, mode)
        flat = grid.ravel()
        flat[positions[:height]] = small_color
        flat[positions[height:]] = large_color
        if not _source_copy(grid):
            break
    else:
        raise AssertionError("unexpected source input collision")
    return {"input": grid, "output": _output(canvas, height, width,
                                                background, body, accent)}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed_key = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed_key)
    train_count, test_count = _pair_counts(seed_key, num_train, num_test)
    canvas = _canvas(rng)
    body, accent = [int(x) for x in rng.choice(10, size=2, replace=False)]
    bg_options = [c for c in range(10) if c not in (body, accent)]
    variable_bg = bool(rng.integers(2))
    bg_count = min(train_count, int(rng.integers(2, 5))) if variable_bg else 1
    backgrounds = [int(x) for x in rng.choice(bg_options, size=bg_count, replace=False)]
    control_backgrounds = backgrounds[:min(bg_count, 2)]
    control_colors = [c for c in range(10) if c not in control_backgrounds]
    base_palette = _palette(rng, control_colors,
                            (body, accent) if rng.random() < 0.7 else ())
    base_small = base_palette[int(rng.integers(2))]
    train_shapes = _train_shapes(rng, canvas, train_count)
    test_shapes = _test_shapes(rng, seed_key, canvas, train_shapes, test_count)
    if train_count == 2:
        test_shapes[0] = train_shapes[0]
    pair_seeds = rng.integers(0, np.iinfo(np.uint64).max,
                              size=train_count + test_count, dtype=np.uint64)
    train = []
    test = []
    for index, shape in enumerate(train_shapes + test_shapes):
        pair_rng = np.random.default_rng(pair_seeds[index])
        if variable_bg and index < train_count:
            background = backgrounds[index % bg_count]
        else:
            background = backgrounds[int(pair_rng.integers(bg_count))]
        if index == 0:
            palette = base_palette
            small_color = base_small
            mode = 0
        elif index == 1:
            palette = base_palette
            small_color = next(c for c in base_palette if c != base_small)
            mode = 1 + int(pair_rng.integers(3))
        else:
            local_colors = [c for c in range(10) if c != background]
            palette = _palette(pair_rng, local_colors, (),
                               base_palette if index == 2 else None)
            small_color = None
            mode = (index + int(pair_rng.integers(5))) % 5
        pair = _pair(pair_rng, canvas, shape, background, body, accent,
                     palette, mode, small_color)
        (train if index < train_count else test).append(pair)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    first = train[0]
    assert isinstance(first, dict) and set(first) == {"input", "output"}
    canvas = first["output"].shape
    assert len(canvas) == 2 and 1 <= canvas[0] <= 30 and 1 <= canvas[1] <= 30
    body = int(first["output"][-1, 1])
    accent = int(first["output"][-1, 0])
    assert body != accent
    input_keys = set()
    pair_keys = set()
    palettes = set()
    train_shapes = []
    for section, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            inp_key = _fingerprint(inp)
            pair_key = (inp_key, _fingerprint(out))
            assert inp_key not in input_keys and pair_key not in pair_keys
            input_keys.add(inp_key)
            pair_keys.add(pair_key)
            assert not _source_copy(inp)
            assert not (inp.shape == out.shape and np.array_equal(inp, out))
            bg, height, width, signals = _input_facts(inp)
            assert len({bg, body, accent}) == 3
            assert 3 <= height < width <= canvas[1] and height < canvas[0]
            assert out.shape == canvas
            assert np.array_equal(out, _output(canvas, height, width, bg, body, accent))
            assert set(int(c) for c in np.unique(out)) == {bg, body, accent}
            assert inp.size >= 4 * (height + width)
            palettes.add(frozenset(c for c, _ in signals))
            if section == "train":
                assert height <= canvas[0] - 3 and width <= canvas[1] - 3
                train_shapes.append((height, width))
    assert len(set(train_shapes)) >= 2
    assert len(palettes) >= 2
    shape_palettes = {}
    for pair in train + test:
        _, height, width, signals = _input_facts(pair["input"])
        shape_palettes.setdefault((height, width), set()).add(
            frozenset(color for color, _ in signals)
        )
    assert any(len(value) >= 2 for value in shape_palettes.values())
    a = _input_facts(train[0]["input"])
    b = _input_facts(train[1]["input"])
    assert a[1] + a[2] == b[1] + b[2] and (a[1], a[2]) != (b[1], b[2])
    assert {c for c, _ in a[3]} == {c for c, _ in b[3]}
    assert next(c for c, n in a[3] if n == a[1]) != next(c for c, n in b[3] if n == b[1])
    if len(train) >= 3:
        c = _input_facts(train[2]["input"])
        assert (c[1], c[2]) == train_shapes[0]
        assert {color for color, _ in c[3]} != {color for color, _ in a[3]}


def _puzzles_equal(a, b):
    return all(
        len(a[section]) == len(b[section]) and all(
            np.array_equal(x["input"], y["input"]) and
            np.array_equal(x["output"], y["output"])
            for x, y in zip(a[section], b[section])
        ) for section in ("train", "test")
    )


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for pair in puzzle["train"] + puzzle["test"]:
            inp, out = pair["input"], pair["output"]
            bg, height, width, _ = _input_facts(inp)
            body, accent = int(out[-1, 1]), int(out[-1, 0])
            for r in range(out.shape[0]):
                for c in range(out.shape[1]):
                    d = out.shape[0] - 1 - r
                    expected = (bg if d >= height or c >= width else
                                accent if c == d or c == width - 1 - d else body)
                    assert int(out[r, c]) == expected
    for train_count, test_count in ((2, 1), (10, 1), (10, 4), (6, 2), (7, 3)):
        p = generate_puzzle(700 + train_count + test_count, train_count, test_count)
        assert (len(p["train"]), len(p["test"])) == (train_count, test_count)
    for args in ({"num_test": 1}, {"num_test": 4},
                 {"num_train": 2}, {"num_train": 10}):
        validate_generated_puzzle(generate_puzzle(900, **args))
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(1000, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {args}")
    assert _puzzles_equal(generate_puzzle(123456, 7, 3),
                          generate_puzzle(123456, 7, 3))
    assert not _puzzles_equal(puzzles[0], puzzles[1])
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    canvases = {p["train"][0]["output"].shape for p in puzzles}
    train_shapes = set()
    test_shapes = set()
    input_shapes = set()
    input_fingerprints = set()
    layout_fingerprints = set()
    palettes = set()
    overlaps = 0
    variable_backgrounds = 0
    geometry = set()
    regimes = set()
    adjacent_layouts = 0
    dense_layouts = 0
    isolated_layouts = 0
    corner_layouts = 0
    run_layouts = 0
    background_as_signal = 0
    for puzzle in puzzles:
        train_backgrounds = {_input_facts(x["input"])[0] for x in puzzle["train"]}
        if len(train_backgrounds) > 1:
            variable_backgrounds += 1
        facts = [_input_facts(x["input"]) for x in puzzle["train"]]
        hmin, hmax = min(x[1] for x in facts), max(x[1] for x in facts)
        wmin, wmax = min(x[2] for x in facts), max(x[2] for x in facts)
        body, accent = int(puzzle["train"][0]["output"][-1, 1]), int(puzzle["train"][0]["output"][-1, 0])
        for section in ("train", "test"):
            for pair in puzzle[section]:
                bg, h, w, signals = _input_facts(pair["input"])
                (train_shapes if section == "train" else test_shapes).add((h, w))
                input_shapes.add(pair["input"].shape)
                input_fingerprints.add(_fingerprint(pair["input"]))
                palettes.add(frozenset(c for c, _ in signals))
                overlaps += any(c in (body, accent) for c, _ in signals)
                background_as_signal += bool(set(c for c, _ in signals) & train_backgrounds)
                mask = pair["input"] != bg
                layout_fingerprints.add(_fingerprint(mask.astype(np.int8)))
                adjacency = (np.count_nonzero(mask[:, :-1] & mask[:, 1:])
                             + np.count_nonzero(mask[:-1, :] & mask[1:, :]))
                adjacent_layouts += adjacency > 0
                dense_layouts += adjacency >= 5
                isolated_layouts += adjacency == 0
                corner_layouts += bool(mask[0, 0] or mask[0, -1]
                                       or mask[-1, 0] or mask[-1, -1])
                run_layouts += (np.any(mask[:, :-2] & mask[:, 1:-1] & mask[:, 2:])
                                or np.any(mask[:-2, :] & mask[1:-1, :] & mask[2:, :]))
                geometry.add((w % 2, min(3, max(0, h - (w + 1) // 2))))
                if section == "test":
                    regimes.add((h > hmax) + (w > wmax))
    assert len(canvases) >= 12 and any(h != w for h, w in canvases)
    assert len(train_shapes) >= 35 and len(test_shapes) >= 25
    assert len(input_shapes) >= 45 and len(palettes) >= 20
    assert len(input_fingerprints) >= 300 and len(layout_fingerprints) >= 290
    assert overlaps >= 25 and variable_backgrounds >= 8
    assert background_as_signal >= 5
    assert adjacent_layouts >= 180 and dense_layouts >= 50
    assert isolated_layouts >= 30 and corner_layouts >= 40
    assert run_layouts >= 50
    assert len(geometry) >= 5 and regimes == {0, 1, 2}
