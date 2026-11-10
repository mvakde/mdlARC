"""Generate solved edge-to-edge dragging-brush ARC puzzles."""

import numpy as np


GEOMETRIES = tuple((width, height) for width in (3, 5, 7) for height in (1, 2, 3))
SOURCE_INPUTS = {
    (17, 17, 4, 5, 2, 5, 6, "right", 1, 5, 6),
    (16, 16, 6, 5, 2, 3, 5, "left", 1, 5, 6),
    (18, 18, 3, 5, 2, 9, 4, "left", 1, 5, 6),
}


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(seed, num_train, num_test, rng):
    train = None if num_train is None else _count("num_train", num_train, 2, 10)
    test = None if num_test is None else _count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _key(grid):
    return grid.shape, np.asarray(grid, dtype=np.uint8).tobytes()


def _signature(info):
    return tuple(info[name] for name in (
        "height", "width", "top", "head_width", "head_height", "fringe",
        "floor_rows", "side", "background", "object", "floor"
    ))


def _input_from(info):
    h, w = info["height"], info["width"]
    top, m, head_h = info["top"], info["head_width"], info["head_height"]
    floor_at = h - info["floor_rows"]
    anchor = 0 if info["side"] == "left" else w - m
    grid = np.full((h, w), info["background"], dtype=np.int8)
    grid[floor_at:, :] = info["floor"]
    grid[top:top + head_h, anchor:anchor + m] = info["object"]
    for r in range(top + head_h, floor_at):
        grid[r, anchor:anchor + m:2] = info["object"]
    return grid


def _output_from(info, trace):
    h, w = info["height"], info["width"]
    top, m, head_h = info["top"], info["head_width"], info["head_height"]
    n, side = info["fringe"], info["side"]
    floor_at = h - info["floor_rows"]
    destination = w - m if side == "left" else 0
    grid = np.full((h, w), info["background"], dtype=np.int8)
    grid[floor_at:, :] = info["floor"]
    grid[top:top + head_h, destination:destination + m] = info["object"]
    for i in range(n):
        start = destination - i if side == "left" else destination + i
        grid[top + head_h + i, start:start + m:2] = info["object"]
    tip = destination - (n - 1) + m - 1 if side == "left" else n - 1
    if side == "left":
        grid[floor_at, :tip] = trace
    else:
        grid[floor_at, tip + 1:] = trace
    return grid


def _draw_dimensions(rng, geometry, regime, min_fringe=2, match_width=None):
    m, head_h = geometry
    if regime == "target_test" and rng.random() < 0.30:
        n = int(rng.integers(max(min_fringe, 9), 17))
    else:
        n = int(rng.integers(min_fringe, 12 if regime == "source_train" else 14))
    top = int(rng.integers(0, min(9, 29 - head_h - n) + 1))
    floor = int(rng.integers(1, min(9, 30 - top - head_h - n) + 1))
    min_width = m + n - 1
    if match_width is not None and match_width >= min_width and rng.random() < 0.42:
        width = match_width
    elif rng.random() < 0.23:
        width = min_width
    else:
        width = int(rng.integers(min_width, 31))
    return {"height": top + head_h + n + floor, "width": width,
            "top": top, "head_width": m, "head_height": head_h,
            "fringe": n, "floor_rows": floor}


def _single_axis_variant(rng, base):
    """Alter exactly one of the four independent size controls when feasible."""
    result = dict(base)
    axis = ("width", "fringe", "top", "floor_rows")[int(rng.integers(0, 4))]
    m, head_h = result["head_width"], result["head_height"]
    n, top, floor = result["fringe"], result["top"], result["floor_rows"]
    if axis == "width":
        choices = [v for v in range(m + n - 1, 31) if v != result["width"]]
    elif axis == "fringe":
        choices = [v for v in range(2, min(16, result["width"] - m + 1,
                                         30 - top - head_h - floor) + 1) if v != n]
    elif axis == "top":
        choices = [v for v in range(0, min(9, 30 - head_h - n - floor) + 1) if v != top]
    else:
        choices = [v for v in range(1, min(9, 30 - top - head_h - n) + 1) if v != floor]
    if not choices:
        return None
    result[axis] = choices[int(rng.integers(len(choices)))]
    result["height"] = result["top"] + head_h + result["fringe"] + result["floor_rows"]
    return result


def _matched_shape_variant(rng, base):
    """Change the drag length while retaining the demonstrated canvas size."""
    variants = []
    m, head_h = base["head_width"], base["head_height"]
    for n in range(2, min(16, base["width"] - m + 1) + 1):
        if n == base["fringe"]:
            continue
        for floor in range(1, 10):
            top = base["height"] - head_h - n - floor
            if 0 <= top <= 9:
                candidate = dict(base)
                candidate.update(top=top, fringe=n, floor_rows=floor)
                variants.append(candidate)
    return variants[int(rng.integers(len(variants)))] if variants else None


def _sides(rng, train_count, test_count):
    first = "left" if int(rng.integers(2)) == 0 else "right"
    other = "right" if first == "left" else "left"
    mode = int(rng.integers(3))
    if mode == 0:
        return [first] * train_count, [first] * test_count
    train = [first, other] + [first if int(rng.integers(2)) == 0 else other
                              for _ in range(train_count - 2)]
    if mode == 1 and test_count >= 2:
        test = [first, other] + [first if int(rng.integers(2)) == 0 else other
                                  for _ in range(test_count - 2)]
    elif mode == 2:
        test = [first] * test_count
    else:
        test = [first if int(rng.integers(2)) == 0 else other
                for _ in range(test_count)]
    return train, test


def _colors(rng, trace, previous, index):
    if index == 1 and rng.random() < 0.55:
        colors = (previous[1], previous[2], previous[0])
    else:
        available = [x for x in range(10) if x != trace]
        colors = tuple(int(x) for x in rng.choice(available, size=3, replace=False))
    if colors == previous:
        colors = (colors[1], colors[2], colors[0])
    return colors


def _pair(info, colors, trace):
    complete = dict(info)
    complete.update(zip(("background", "object", "floor"), colors))
    return {"input": _input_from(complete), "output": _output_from(complete, trace)}, complete


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, num_train, num_test, rng)
    trace = int(rng.integers(10))
    train_sides, test_sides = _sides(rng, train_count, test_count)

    first_geometry = (5, 2) if rng.random() < 0.4 else GEOMETRIES[int(rng.integers(len(GEOMETRIES)))]
    alternatives = [g for g in GEOMETRIES if g != first_geometry]
    train_geometries = [first_geometry, alternatives[int(rng.integers(len(alternatives)))]]
    train_geometries += [GEOMETRIES[int(rng.integers(len(GEOMETRIES)))]
                         for _ in range(train_count - 2)]
    test_geometries = [train_geometries[int(rng.integers(train_count))]
                       for _ in range(test_count)]

    puzzle = {"train": [], "test": []}
    used_inputs = set()
    train_specs = []
    previous_colors = None
    all_specs = []
    for i, (side, geom) in enumerate(zip(train_sides, train_geometries)):
        dims = _draw_dimensions(rng, geom, "source_train", min_fringe=4 if i == 0 else 3 if i == 1 else 2)
        dims["side"] = side
        train_specs.append(dims)
        all_specs.append(("train", dims))
    for side, geom in zip(test_sides, test_geometries):
        matching = [x for x in train_specs if (x["head_width"], x["head_height"]) == geom and x["side"] == side]
        shape_matching = [x for x in train_specs if (x["head_width"], x["head_height"]) == geom]
        draw = rng.random()
        if draw < 0.20 and shape_matching:
            dims = _matched_shape_variant(rng, shape_matching[int(rng.integers(len(shape_matching)))])
        elif draw < 0.50 and matching:
            dims = _single_axis_variant(rng, matching[int(rng.integers(len(matching)))])
        else:
            dims = None
        if dims is None:
            width = train_specs[int(rng.integers(train_count))]["width"]
            dims = _draw_dimensions(rng, geom, "target_test", match_width=width)
        dims["side"] = side
        all_specs.append(("test", dims))

    for index, (split, dims) in enumerate(all_specs):
        colors = _colors(rng, trace, previous_colors, index)
        previous_colors = colors
        pair, complete = _pair(dims, colors, trace)
        if _key(pair["input"]) in used_inputs or _signature(complete) in SOURCE_INPUTS:
            available = [x for x in range(10) if x != trace]
            for a in available:
                for b in available:
                    for c in available:
                        if len({a, b, c}) < 3:
                            continue
                        candidate, info = _pair(dims, (a, b, c), trace)
                        if _key(candidate["input"]) not in used_inputs and _signature(info) not in SOURCE_INPUTS:
                            pair, complete = candidate, info
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                raise AssertionError("no distinct palette available")
        used_inputs.add(_key(pair["input"]))
        puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _assert_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert np.issubdtype(grid.dtype, np.integer)
    assert int(grid.min()) >= 0 and int(grid.max()) <= 9


def _analyze_input(grid):
    h, w = grid.shape
    floor = int(grid[-1, 0])
    assert np.all(grid[-1] == floor)
    floor_at = h
    while floor_at and np.all(grid[floor_at - 1] == floor):
        floor_at -= 1
    assert 0 < floor_at < h
    values = {int(x) for x in np.unique(grid)}
    assert len(values) == 3 and floor in values
    candidates = []
    for obj in values - {floor}:
        background = next(iter(values - {floor, obj}))
        rows = np.flatnonzero(np.any(grid == obj, axis=1))
        if not len(rows):
            continue
        top = int(rows[0])
        for m, head_h in GEOMETRIES:
            n = floor_at - top - head_h
            if n < 2 or w < m + n - 1:
                continue
            for side in ("left", "right"):
                info = {"height": h, "width": w, "top": top,
                        "head_width": m, "head_height": head_h,
                        "fringe": n, "floor_rows": h - floor_at, "side": side,
                        "background": background, "object": obj, "floor": floor}
                if np.array_equal(grid, _input_from(info)):
                    candidates.append(info)
    assert len(candidates) == 1, "input must reveal one edge-flush brush and full floor"
    return candidates[0]


def _analyze_pair(pair):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape and not np.array_equal(inp, out)
    info = _analyze_input(inp)
    extra = {int(x) for x in np.unique(out)} - {info["background"], info["object"], info["floor"]}
    assert len(extra) == 1 and len(np.unique(out)) == 4
    trace = next(iter(extra))
    assert np.array_equal(out, _output_from(info, trace)), "incorrect drag or floor endpoint"
    info["trace"] = trace
    return info


def _cell_rule_check(pair, info):
    """Check changed cell sets without constructing an expected output grid."""
    out = pair["output"]
    m, head_h, n = info["head_width"], info["head_height"], info["fringe"]
    top, w, side = info["top"], info["width"], info["side"]
    floor_at = info["height"] - info["floor_rows"]
    dest = w - m if side == "left" else 0
    expected_object = {(r, c) for r in range(top, top + head_h)
                       for c in range(dest, dest + m)}
    for i in range(n):
        start = dest - i if side == "left" else dest + i
        expected_object.update((top + head_h + i, start + k) for k in range(0, m, 2))
    actual_object = {tuple(map(int, x)) for x in np.argwhere(out == info["object"])}
    assert actual_object == expected_object
    tip = dest - n + m if side == "left" else n - 1
    trace_cols = range(tip) if side == "left" else range(tip + 1, w)
    actual_trace = {tuple(map(int, x)) for x in np.argwhere(out == info["trace"])}
    assert actual_trace == {(floor_at, c) for c in trace_cols}
    assert out[floor_at, tip] == info["floor"]
    assert np.all(out[floor_at + 1:] == info["floor"])


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
        assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
        nt, ns = len(puzzle["train"]), len(puzzle["test"])
        assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
        seen_inputs, seen_pairs = set(), set()
        infos = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert isinstance(pair, dict) and set(pair) == {"input", "output"}
                _assert_grid(pair["input"])
                _assert_grid(pair["output"])
                ik, ok = _key(pair["input"]), _key(pair["output"])
                assert ik not in seen_inputs and (ik, ok) not in seen_pairs
                seen_inputs.add(ik)
                seen_pairs.add((ik, ok))
                info = _analyze_pair(pair)
                _cell_rule_check(pair, info)
                assert _signature(info) not in SOURCE_INPUTS
                assert 2 <= info["fringe"] <= (11 if split == "train" else 16)
                assert 0 <= info["top"] <= 9 and 1 <= info["floor_rows"] <= 9
                assert info["width"] >= info["head_width"] + info["fringe"] - 1
                infos[split].append(info)
        train, test = infos["train"], infos["test"]
        assert len({x["trace"] for x in train + test}) == 1
        assert train[0]["fringe"] >= 4 and train[1]["fringe"] >= 3
        train_geometries = {(x["head_width"], x["head_height"]) for x in train}
        assert len(train_geometries) >= 2
        assert {(x["head_width"], x["head_height"]) for x in test} <= train_geometries
        train_sides = {x["side"] for x in train}
        assert {x["side"] for x in test} <= train_sides
        assert len({(x["background"], x["object"], x["floor"]) for x in train}) >= 2
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(f"malformed puzzle: {exc}") from exc


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    shapes, layouts, palettes, geometries, fringes, floors, tops = (set() for _ in range(7))
    modes, swaps, traces = set(), 0, set()
    tight = one_row_floor = top_edge = short = long = single_axis = matched_width = 0
    matched_shape = literal_swap = source_geometry = long_extrapolation = combined = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        parsed = {s: [_analyze_pair(p) for p in puzzle[s]] for s in ("train", "test")}
        for split in ("train", "test"):
            for pair, x in zip(puzzle[split], parsed[split]):
                _cell_rule_check(pair, x)
                shapes.add(pair["input"].shape)
                layouts.add((x["height"], x["width"], x["top"], x["head_width"],
                             x["head_height"], x["fringe"], x["floor_rows"], x["side"]))
                palettes.add((x["background"], x["object"], x["floor"], x["trace"]))
                geometries.add((x["head_width"], x["head_height"]))
                fringes.add(x["fringe"])
                floors.add(x["floor_rows"])
                tops.add(x["top"])
                tight += x["width"] == x["head_width"] + x["fringe"] - 1
                one_row_floor += x["floor_rows"] == 1
                top_edge += x["top"] == 0
                short += x["fringe"] <= 3
                long += x["fringe"] >= 11
        tr, te = parsed["train"], parsed["test"]
        traces.add(tr[0]["trace"])
        modes.add((len({x["side"] for x in tr}), len({x["side"] for x in te})))
        swaps += any((a["background"], a["object"], a["floor"]) !=
                     (b["background"], b["object"], b["floor"]) for a in tr for b in te)
        matched_width += any(a["width"] == b["width"] for a in tr for b in te)
        matched_shape += any((a["height"], a["width"]) == (b["height"], b["width"])
                             for a in tr for b in te)
        literal_swap += any(a["background"] == b["object"] or a["object"] == b["floor"]
                            or a["floor"] == b["background"] for a in tr for b in te)
        source_geometry += sum((x["head_width"], x["head_height"]) == (5, 2) for x in tr + te)
        long_extrapolation += max(x["fringe"] for x in te) > max(x["fringe"] for x in tr)
        combined += any(x["floor_rows"] == 1 and x["top"] == 0 and
                        x["width"] == x["head_width"] + x["fringe"] - 1 for x in tr + te)
        for b in te:
            single_axis += any(sum(a[k] != b[k] for k in ("width", "fringe", "top", "floor_rows")) == 1
                               and a["head_width"] == b["head_width"] and a["head_height"] == b["head_height"]
                               and a["side"] == b["side"] for a in tr)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 50 and len(layouts) >= 160 and len(palettes) >= 80
    assert len(geometries) == 9 and len(fringes) >= 12 and len(floors) >= 8 and len(tops) >= 8
    assert (1, 1) in modes and (2, 1) in modes and (2, 2) in modes
    assert swaps >= 35 and tight >= 20 and one_row_floor >= 10 and top_edge >= 10
    assert short >= 10 and long >= 10 and single_axis >= 4 and matched_width >= 5
    assert matched_shape >= 5 and literal_swap >= 20 and source_geometry >= 15
    assert long_extrapolation >= 5 and combined >= 1 and len(traces) >= 8
    assert _same_puzzle(generate_puzzle(12345), generate_puzzle(12345))
    assert _same_puzzle(generate_puzzle(54321, 7, 3), generate_puzzle(54321, 7, 3))
    assert not _same_puzzle(generate_puzzle(31415, 6, 2), generate_puzzle(31416, 6, 2))
    for seed, train, test in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4), (1004, 6, 2),
                              (2001, None, 1), (2002, None, 4), (2003, 2, None), (2004, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
        validate_generated_puzzle(puzzle)
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                 {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(9999, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {args}")
