"""Perpendicular line completion ARC generator."""

import numpy as np


_SOURCE = (
    (("080000", "080000", "000000", "000022", "000000", "000000"),
     ("080000", "080000", "080000", "242222", "080000", "080000")),
    (("000080", "000080", "220000", "000000", "000000", "000000"),
     ("000080", "000080", "222242", "000080", "000080", "000080")),
    (("000800", "000800", "000000", "000000", "220000", "000000"),
     ("000800", "000800", "000800", "000800", "222422", "000800")),
)


def _counts(seed, train, test):
    for name, value, low, high in (("num_train", train, 2, 10),
                                   ("num_test", test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= value <= high):
            raise ValueError(f"{name} is outside its legal range")
    mix = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + mix % (10 - test)
    elif test is None:
        test = 1 + mix % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return int(train), int(test)


def _shape(rng):
    families = ((6, 9, 6, 9), (9, 17, 9, 17),
                (17, 31, 6, 13), (6, 13, 17, 31),
                (18, 31, 18, 31), (6, 31, 6, 31))
    h0, h1, w0, w1 = families[int(rng.integers(len(families)))]
    return int(rng.integers(h0, h1)), int(rng.integers(w0, w1))


def _spec(rng, height, width, vside=None, hside=None,
          vedge=None, hedge=None, position=None):
    rows = range(height) if position is None else (position[0],)
    cols = range(width) if position is None else (position[1],)
    vr = [(r, s) for r in rows
          for s in ((vside,) if vside else ("above", "below"))
          if (r if s == "above" else height - r - 1)
          >= (3 if vedge is False else 2)]
    hc = [(c, s) for c in cols
          for s in ((hside,) if hside else ("left", "right"))
          if (c if s == "left" else width - c - 1)
          >= (3 if hedge is False else 2)]
    if not vr or not hc:
        return None
    row, vs = vr[int(rng.integers(len(vr)))]
    col, hs = hc[int(rng.integers(len(hc)))]
    vd = row if vs == "above" else height - row - 1
    hd = col if hs == "left" else width - col - 1
    ve = bool(rng.integers(2)) if vedge is None else vedge
    he = bool(rng.integers(2)) if hedge is None else hedge
    return row, col, vs, hs, ve or vd < 3, he or hd < 3


def _interval(rng, limit, crossing, side, edge):
    if side in ("above", "left"):
        if edge:
            return 0, int(rng.integers(1, crossing))
        start = int(rng.integers(1, crossing - 1))
        return start, int(rng.integers(start + 1, crossing))
    if edge:
        return int(rng.integers(crossing + 1, limit - 1)), limit - 1
    end = int(rng.integers(crossing + 2, limit - 1))
    return int(rng.integers(crossing + 1, end)), end


def _pair(rng, shape, roles, spec):
    height, width = shape
    bg, horizontal, vertical, marker = roles
    row, col, vs, hs, ve, he = spec
    v0, v1 = _interval(rng, height, row, vs, ve)
    h0, h1 = _interval(rng, width, col, hs, he)
    inp = np.full(shape, bg, dtype=np.int8)
    inp[v0:v1 + 1, col] = vertical
    inp[row, h0:h1 + 1] = horizontal
    out = inp.copy()
    out[row, :] = horizontal
    out[:, col] = vertical
    out[row, col] = marker
    return {"input": inp, "output": out}


def _key(grid):
    return grid.shape, grid.tobytes()


def _source_copy(pair):
    inp, out = pair["input"], pair["output"]
    if inp.shape != (6, 6):
        return False
    def rows(grid):
        return tuple("".join(str(int(x)) for x in row) for row in grid)
    return (rows(inp), rows(out)) in _SOURCE


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train, num_test = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    marker = int(rng.integers(10))
    colors = [x for x in range(10) if x != marker]
    base = tuple(int(x) for x in rng.choice(colors, size=3, replace=False))
    variable_palette = bool(rng.integers(2))
    variable_shape = bool(rng.integers(2))
    shape_a = _shape(rng)
    shape_b = _shape(rng) if variable_shape else shape_a
    if variable_shape and shape_b == shape_a:
        shape_b = (shape_a[0], 6 if shape_a[1] != 6 else 7)
    new_background = int(rng.choice([x for x in colors if x not in base]))
    roles_b = (new_background, base[2], base[1]) if variable_palette else base
    side_a = "left" if bool(rng.integers(2)) else "right"
    side_b = "right" if side_a == "left" else "left"
    puzzle = {"train": [], "test": []}
    seen = set()
    training_shapes, training_positions = [], []

    for regime, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if regime == "train" and index == 0:
                shape, roles = shape_a, (*base, marker)
                requested = ("above", side_a, True, True)
            elif regime == "train" and index == 1:
                shape, roles = shape_b, (*roles_b, marker)
                requested = ("below", side_b, False, False)
            else:
                if variable_shape:
                    shape = (training_shapes[int(rng.integers(len(training_shapes)))]
                             if bool(rng.integers(2)) else _shape(rng))
                else:
                    shape = shape_a
                if variable_palette:
                    roles = tuple(int(x) for x in rng.choice(colors, size=3, replace=False)) + (marker,)
                else:
                    roles = (*base, marker)
                requested = (None, None, None, None)
            familiar = None
            if regime == "test" and int(rng.integers(3)) == 0:
                matching = [p for s, p in zip(training_shapes, training_positions) if s == shape]
                if matching:
                    familiar = matching[int(rng.integers(len(matching)))]

            pair = spec = None
            for attempt in range(64):
                position = familiar if attempt < 32 else None
                spec = _spec(rng, *shape, *requested, position=position)
                if spec is None:
                    continue
                candidate = _pair(rng, shape, roles, spec)
                if _key(candidate["input"]) not in seen and not _source_copy(candidate):
                    pair = candidate
                    break
            if pair is None:
                for row in range(shape[0]):
                    if pair is not None:
                        break
                    for col in range(shape[1]):
                        spec = _spec(rng, *shape, position=(row, col))
                        if spec is None:
                            continue
                        candidate = _pair(rng, shape, roles, spec)
                        if _key(candidate["input"]) not in seen and not _source_copy(candidate):
                            pair = candidate
                            break
            if pair is None:
                raise AssertionError("no unique pair could be constructed")
            seen.add(_key(pair["input"]))
            puzzle[regime].append(pair)
            if regime == "train":
                training_shapes.append(shape)
                training_positions.append(spec[:2])
    validate_generated_puzzle(puzzle)
    return puzzle


def _describe(pair):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape and not np.array_equal(inp, out)
    values, counts = np.unique(inp, return_counts=True)
    assert len(values) == 3
    bg = int(values[int(np.argmax(counts))])
    assert int(np.max(counts)) > int(np.sum(counts) - np.max(counts))
    fragments = {}
    for value in values:
        color = int(value)
        if color == bg:
            continue
        coords = np.argwhere(inp == color)
        assert len(coords) >= 2
        rs, cs = coords[:, 0], coords[:, 1]
        if np.all(rs == rs[0]):
            axis, coordinate, positions = "h", int(rs[0]), sorted(int(x) for x in cs)
        else:
            assert np.all(cs == cs[0])
            axis, coordinate, positions = "v", int(cs[0]), sorted(int(x) for x in rs)
        assert positions == list(range(positions[0], positions[-1] + 1))
        assert axis not in fragments
        fragments[axis] = color, coordinate, positions
    assert set(fragments) == {"h", "v"}
    hc, row, hpos = fragments["h"]
    vc, col, vpos = fragments["v"]
    assert int(inp[row, col]) == bg
    assert hpos[-1] < col or hpos[0] > col
    assert vpos[-1] < row or vpos[0] > row
    hs = "left" if hpos[-1] < col else "right"
    vs = "above" if vpos[-1] < row else "below"
    he = hpos[0] == 0 if hs == "left" else hpos[-1] == inp.shape[1] - 1
    ve = vpos[0] == 0 if vs == "above" else vpos[-1] == inp.shape[0] - 1
    marker = int(out[row, col])
    assert marker not in values
    expected = inp.copy()
    expected[row, :] = hc
    expected[:, col] = vc
    expected[row, col] = marker
    assert np.array_equal(out, expected)
    return {"shape": inp.shape, "roles": (bg, hc, vc), "marker": marker,
            "row": row, "col": col, "hside": hs, "vside": vs,
            "hedge": he, "vedge": ve, "hlength": len(hpos), "vlength": len(vpos)}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    descriptions = {"train": [], "test": []}
    inputs, pairs = set(), set()
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert 0 <= int(np.min(grid)) and int(np.max(grid)) <= 9
            key = _key(pair["input"])
            pair_key = (key, _key(pair["output"]))
            assert key not in inputs and pair_key not in pairs
            inputs.add(key)
            pairs.add(pair_key)
            assert not _source_copy(pair)
            descriptions[regime].append(_describe(pair))
    train, test = descriptions["train"], descriptions["test"]
    markers = {d["marker"] for d in train + test}
    assert len(markers) == 1
    marker = next(iter(markers))
    assert all(marker not in d["roles"] for d in train + test)
    for feature in ("hside", "vside", "hedge", "vedge"):
        assert {d[feature] for d in test} <= {d[feature] for d in train}
    assert {d["hside"] for d in train} == {"left", "right"}
    assert {d["vside"] for d in train} == {"above", "below"}
    assert {d["hedge"] for d in train} == {True, False}
    assert {d["vedge"] for d in train} == {True, False}
    if len({d["shape"] for d in train}) == 1:
        assert all(d["shape"] == train[0]["shape"] for d in test)
    if len({d["roles"] for d in train}) == 1:
        assert all(d["roles"] == train[0]["roles"] for d in test)


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts = {len(p["train"]) for p in puzzles}
    test_counts = {len(p["test"]) for p in puzzles}
    shapes, layouts, palettes, lengths = set(), set(), set(), set()
    fixed_shapes = mixed_shapes = fixed_palettes = mixed_palettes = 0
    right_tests = below_tests = internal_tests = bottom_trains = 0
    interior_trains = source_scales = border_crossings = combined = 0
    familiar_tests = new_tests = palette_swaps = 0
    forward_shifts = reverse_shifts = little_shifts = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train = [_describe(pair) for pair in puzzle["train"]]
        test = [_describe(pair) for pair in puzzle["test"]]
        fixed_shapes += len({d["shape"] for d in train}) == 1
        mixed_shapes += len({d["shape"] for d in train}) > 1
        fixed_palettes += len({d["roles"] for d in train}) == 1
        mixed_palettes += len({d["roles"] for d in train}) > 1
        bottom_trains += any(d["row"] == d["shape"][0] - 1 for d in train)
        right_tests += any(d["hside"] == "right" for d in test)
        below_tests += any(d["vside"] == "below" for d in test)
        internal_tests += any(not d["hedge"] or not d["vedge"] for d in test)
        interior_trains += any(2 < d["col"] < d["shape"][1] - 3 for d in train)
        source_scales += any(d["shape"] == (6, 6) for d in train + test)
        border_crossings += any(
            d["row"] == 0 or d["col"] in (0, d["shape"][1] - 1)
            for d in train + test)
        combined += any(not d["hedge"] and not d["vedge"]
                        and d["vside"] == "below"
                        and d["shape"][0] != d["shape"][1]
                        for d in train + test)
        familiar_tests += any(
            d["shape"] == t["shape"] and (d["row"], d["col"]) == (t["row"], t["col"])
            for d in test for t in train)
        new_tests += any(
            not any(d["shape"] == t["shape"]
                    and (d["row"], d["col"]) == (t["row"], t["col"])
                    for t in train) for d in test)
        palette_swaps += any(
            a["roles"][0] != b["roles"][0]
            and a["roles"][1] == b["roles"][2]
            and a["roles"][2] == b["roles"][1]
            for a in train for b in train)
        train_row = sum(d["row"] / (d["shape"][0] - 1) for d in train) / len(train)
        test_row = sum(d["row"] / (d["shape"][0] - 1) for d in test) / len(test)
        train_outer = sum(abs(2 * d["col"] / (d["shape"][1] - 1) - 1)
                          for d in train) / len(train)
        test_outer = sum(abs(2 * d["col"] / (d["shape"][1] - 1) - 1)
                         for d in test) / len(test)
        forward_shifts += test_row > train_row + .15 and test_outer < train_outer - .15
        reverse_shifts += test_row < train_row - .15 and test_outer > train_outer + .15
        little_shifts += abs(test_row - train_row) <= .15 and abs(test_outer - train_outer) <= .15
        for d in train + test:
            shapes.add(d["shape"])
            palettes.add((*d["roles"], d["marker"]))
            lengths.update((d["hlength"], d["vlength"]))
            layouts.add((d["shape"], d["row"], d["col"], d["hside"],
                         d["vside"], d["hedge"], d["vedge"],
                         d["hlength"], d["vlength"]))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 30 and len(layouts) >= 100
    assert len(palettes) >= 30 and len(lengths) >= 10
    assert fixed_shapes >= 5 and mixed_shapes >= 5
    assert fixed_palettes >= 5 and mixed_palettes >= 5
    assert min(min(s) for s in shapes) == 6 and max(max(s) for s in shapes) >= 25
    assert any(h >= 2 * w for h, w in shapes)
    assert any(w >= 2 * h for h, w in shapes)
    assert right_tests >= 4 and below_tests >= 4 and internal_tests >= 4
    assert bottom_trains >= 1
    assert interior_trains >= 10 and source_scales >= 1
    assert border_crossings >= 5 and combined >= 10
    assert familiar_tests >= 5 and new_tests >= 10
    assert palette_swaps >= 5
    assert forward_shifts >= 1 and reverse_shifts >= 1 and little_shifts >= 5
    for seed, nt, ns in ((100, 2, 1), (101, 10, 1), (102, 10, 4),
                         (103, 6, 2), (200, None, 1), (201, None, 4),
                         (202, 2, None), (203, 10, None)):
        puzzle = generate_puzzle(seed, nt, ns)
        validate_generated_puzzle(puzzle)
        if nt is not None:
            assert len(puzzle["train"]) == nt
        if ns is not None:
            assert len(puzzle["test"]) == ns
    a, b = generate_puzzle(987654, 8, 3), generate_puzzle(987654, 8, 3)
    assert all(np.array_equal(x[k], y[k]) for regime in ("train", "test")
               for x, y in zip(a[regime], b[regime]) for k in ("input", "output"))
    assert len({_key(generate_puzzle(s)["train"][0]["input"]) for s in range(42)}) >= 40
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": True},
                   {"num_test": 1.5}):
        try:
            generate_puzzle(500, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
