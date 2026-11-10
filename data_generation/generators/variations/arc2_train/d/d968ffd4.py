import numpy as np


# Exact source inputs are excluded, while their dimensions and palettes remain legal.
_SOURCE_INPUTS = (
    ("4444444444444444", "4114444444444664", "4114444444444664",
     "4114444444444664", "4444444444444444"),
    ("22222222222222222", "21122222222222332", "21122222222222332",
     "22222222222222222"),
    ("888", "828", "828", "888", "888", "888", "888", "888", "888",
     "888", "888", "888", "888", "888", "888", "888", "818", "818", "888"),
    ("222", "282", "282", "222", "222", "222", "222", "222", "222",
     "222", "222", "222", "222", "222", "222", "222", "222", "222",
     "222", "212", "212", "222"),
    ("333333", "388883", "388883", "333333", "333333", "333333",
     "333333", "333333", "333333", "333333", "333333", "333333",
     "333333", "333333", "322223", "322223", "333333"),
)


def _key(grid):
    return (grid.shape, tuple(int(v) for v in grid.flat))


def _source_keys():
    return {
        ((len(rows), len(rows[0])), tuple(int(c) for row in rows for c in row))
        for rows in _SOURCE_INPUTS
    }


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(seed, rng, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
    elif train is None and test is None:
        test = 1 + seed % 4
        train = max(test + 1, 2 + (seed // 4) % 9)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _child_rng(rng):
    return np.random.default_rng(int(rng.integers(0, 2**63 - 1)))


def _parity_value(rng, low, high, parity):
    values = [v for v in range(low, high + 1) if v % 2 == parity]
    assert values
    return int(values[int(rng.integers(len(values)))])


def _box(grid, color):
    points = np.argwhere(grid == color)
    assert len(points)
    r0, c0 = (int(v) for v in points.min(axis=0))
    r1, c1 = (int(v) for v in points.max(axis=0))
    assert len(points) == (r1 - r0 + 1) * (c1 - c0 + 1)
    assert np.all(grid[r0:r1 + 1, c0:c1 + 1] == color)
    return r0, r1, c0, c1


def _analyze(grid):
    h, w = grid.shape
    background = int(grid[0, 0])
    assert all(int(grid[r, c]) == background for r, c in
               ((0, w - 1), (h - 1, 0), (h - 1, w - 1)))
    assert np.all(grid[0] == background) and np.all(grid[-1] == background)
    assert np.all(grid[:, 0] == background) and np.all(grid[:, -1] == background)
    colors = [int(c) for c in np.unique(grid) if int(c) != background]
    assert len(colors) == 2
    boxes = [_box(grid, color) for color in colors]
    a, b = boxes
    horizontal = a[1] >= b[0] and b[1] >= a[0]
    vertical = a[3] >= b[2] and b[3] >= a[2]
    if horizontal and (a[3] + 2 <= b[2] or b[3] + 2 <= a[2]):
        axis = "horizontal"
        order = sorted(range(2), key=lambda i: boxes[i][2])
        transverse = [(box[0], box[1]) for box in boxes]
        dimension = w
    elif vertical and (a[1] + 2 <= b[0] or b[1] + 2 <= a[0]):
        axis = "vertical"
        order = sorted(range(2), key=lambda i: boxes[i][0])
        transverse = [(box[2], box[3]) for box in boxes]
        dimension = h
    else:
        raise AssertionError("pads do not face each other on a unique axis")
    near, far = (boxes[i] for i in order)
    near_color, far_color = (colors[i] for i in order)
    overlap = min(transverse[0][1], transverse[1][1]) - max(
        transverse[0][0], transverse[1][0]) + 1
    shorter = min(x[1] - x[0] + 1 for x in transverse)
    assert overlap >= max(1, (shorter + 1) // 2)
    assert abs(sum(transverse[0]) - sum(transverse[1])) <= 4
    if axis == "horizontal":
        start, stop = near[3] + 1, far[2]
        end_margins = (near[2], w - 1 - far[3])
        depths = (near[3] - near[2] + 1, far[3] - far[2] + 1)
        side_margins = (near[0], h - 1 - near[1], far[0], h - 1 - far[1])
        assert np.all(grid[:, start:stop] == background)
    else:
        start, stop = near[1] + 1, far[0]
        end_margins = (near[0], h - 1 - far[1])
        depths = (near[1] - near[0] + 1, far[1] - far[0] + 1)
        side_margins = (near[2], w - 1 - near[3], far[2], w - 1 - far[3])
        assert np.all(grid[start:stop, :] == background)
    assert stop - start >= 2
    assert all(m >= 1 for m in end_margins + side_margins)
    assert 1 <= min(depths) and max(depths) <= 3
    assert 1 <= start and stop < dimension
    return {
        "axis": axis, "background": background, "colors": (near_color, far_color),
        "boxes": (near, far), "start": start, "stop": stop, "gap": stop - start,
        "depths": depths, "ends": end_margins, "sides": side_margins,
        "spans": tuple(x[1] - x[0] + 1 for x in
                       (transverse[i] for i in order)),
        "offset": transverse[order[1]][0] - transverse[order[0]][0],
        "shape": grid.shape,
    }


def _fill(grid, info):
    out = grid.copy()
    half = info["gap"] // 2
    a, b = info["colors"]
    start, stop = info["start"], info["stop"]
    if info["axis"] == "horizontal":
        out[:, start:start + half] = a
        out[:, stop - half:stop] = b
    else:
        out[start:start + half, :] = a
        out[stop - half:stop, :] = b
    return out


def _palette_plan(rng, count):
    base = [int(v) for v in rng.choice(10, 3, replace=False)]
    stable = bool(rng.integers(0, 4) == 0)
    first = (base[0], base[1], base[2])
    other = [base[0], base[2]]
    if not stable:
        other[1] = int(rng.choice([c for c in range(10) if c not in base]))
    if bool(rng.integers(2)):
        other.reverse()
    result = [first, (base[1], other[0], other[1])]
    for _ in range(count - 2):
        if stable:
            triple = [int(v) for v in rng.permutation(base)]
        else:
            # Each input is locally interpretable; a new literal needs no map.
            pool = list(dict.fromkeys(base + [v for triple in result for v in triple]))
            if rng.random() < 0.55:
                pool += [c for c in range(10) if c not in pool]
            triple = [int(v) for v in rng.choice(pool, 3, replace=False)]
        result.append(tuple(triple))
    return result


def _geometry(rng, axis, parity, split, anchor=False):
    classic = rng.random() < (0.25 if split == "train" else 0.12)
    if classic:
        depths = (2, 2)
        ends = (1, 1)
        side_start = right_margin = 1
        span_a = span_b = int(rng.integers(1, 10))
        offset = 0
    else:
        depth_a = int(rng.integers(1, 4))
        depth_b = depth_a if rng.random() < 0.72 else int(rng.integers(1, 4))
        depths = (depth_a, depth_b)
        ends = (int(rng.integers(1, 4)), int(rng.integers(1, 4)))
        side_start = int(rng.integers(1, 4))
        right_margin = int(rng.integers(1, 4))
        span_a = int(rng.integers(1, 15))
        if span_a >= 3 and rng.random() < 0.35:
            span_b = max(1, min(14, span_a + int(rng.integers(-2, 3))))
            offset = int(rng.integers(-1, 2))
            offset = max(1 - side_start, offset)
            overlap = min(side_start + span_a, side_start + offset + span_b) - max(
                side_start, side_start + offset)
            if overlap < max(1, (min(span_a, span_b) + 1) // 2) or abs(
                    2 * offset + span_b - span_a) > 4:
                span_b, offset = span_a, 0
        else:
            span_b, offset = span_a, 0
    second_start = side_start + offset
    assert second_start >= 1
    transverse = max(side_start + span_a, second_start + span_b) + right_margin
    maximum_gap = min(24 if split == "test" else 20,
                      30 - sum(depths) - sum(ends))
    if anchor:
        low, high = 8, min(14, maximum_gap)
    elif split == "train":
        mode = int(rng.choice(3, p=[0.30, 0.58, 0.12]))
        low, high = ((2, 5), (6, 13), (14, 20))[mode]
    else:
        mode = int(rng.choice(4, p=[0.25, 0.32, 0.27, 0.16]))
        low, high = ((2, 5), (6, 13), (14, 24), (4, 19))[mode]
        if mode == 3 and transverse < 10:
            span_a = max(span_a, 8)
            span_b = span_a
            offset = 0
            second_start = side_start
            transverse = side_start + span_a + right_margin
    high = min(high, maximum_gap)
    if low > high or not any(v % 2 == parity for v in range(low, high + 1)):
        low, high = 2, maximum_gap
    gap = _parity_value(rng, low, high, parity)
    return depths, ends, (side_start, second_start, span_a, span_b,
                          transverse), gap


def _make_pair(rng, axis, parity, colors, split, anchor=False):
    depths, ends, transverse, gap = _geometry(rng, axis, parity, split, anchor)
    d0, d1 = depths
    e0, e1 = ends
    s0, s1, span0, span1, cross = transverse
    length = e0 + d0 + gap + d1 + e1
    background, near_color, far_color = colors
    if axis == "horizontal":
        grid = np.full((cross, length), background, dtype=np.int8)
        grid[s0:s0 + span0, e0:e0 + d0] = near_color
        far_start = e0 + d0 + gap
        grid[s1:s1 + span1, far_start:far_start + d1] = far_color
    else:
        grid = np.full((length, cross), background, dtype=np.int8)
        grid[e0:e0 + d0, s0:s0 + span0] = near_color
        far_start = e0 + d0 + gap
        grid[far_start:far_start + d1, s1:s1 + span1] = far_color
    info = _analyze(grid)
    assert info["axis"] == axis and info["gap"] == gap
    return {"input": grid, "output": _fill(grid, info)}


def _geometry_signature(info):
    return (info["axis"], info["shape"], info["gap"], info["depths"],
            info["ends"], info["sides"], info["spans"], info["offset"])


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    used, source = set(), _source_keys()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in used and key not in source
            used.add(key)
            info = _analyze(inp)
            half = info["gap"] // 2
            for index in range(inp.shape[1] if info["axis"] == "horizontal"
                               else inp.shape[0]):
                if info["start"] <= index < info["start"] + half:
                    expected = info["colors"][0]
                elif info["stop"] - half <= index < info["stop"]:
                    expected = info["colors"][1]
                else:
                    expected = None
                old = inp[:, index] if info["axis"] == "horizontal" else inp[index, :]
                new = out[:, index] if info["axis"] == "horizontal" else out[index, :]
                if expected is None:
                    assert np.array_equal(new, old)
                else:
                    assert np.all(old == info["background"])
                    assert np.all(new == expected)
            assert not np.array_equal(inp, out)
            analyses[split].append(info)
    train = analyses["train"]
    assert {x["axis"] for x in train} == {"horizontal", "vertical"}
    assert {x["gap"] % 2 for x in train} == {0, 1}
    train_cases = {(x["axis"], x["gap"] % 2) for x in train}
    assert len(train_cases) >= min(nt, 4)
    assert max(x["gap"] for x in train) >= 8
    assert train[0]["background"] != train[1]["background"]
    assert train[1]["background"] in train[0]["colors"]
    train_geometry = {_geometry_signature(x) for x in train}
    for info in analyses["test"]:
        assert info["axis"] in {x["axis"] for x in train}
        if nt == 2:
            assert (info["axis"], info["gap"] % 2) in train_cases
        assert _geometry_signature(info) not in train_geometry


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    nt, ns = _counts(seed, rng, num_train, num_test)
    # Only the invariant split rule is a learned puzzle convention. The
    # palette policy is sampled once, while each pad color remains visible.
    colors = _palette_plan(rng, nt + ns)
    first_axis = "horizontal" if bool(rng.integers(2)) else "vertical"
    first_parity = int(rng.integers(2))
    train_plan = [(first_axis, first_parity),
                  ("vertical" if first_axis == "horizontal" else "horizontal",
                   1 - first_parity)]
    missing_cases = [(first_axis, 1 - first_parity),
                     (train_plan[1][0], first_parity)]
    if bool(rng.integers(2)):
        missing_cases.reverse()
    train_plan += missing_cases[:max(0, min(nt, 4) - 2)]
    train_plan += [("horizontal" if bool(rng.integers(2)) else "vertical",
                    int(rng.integers(2))) for _ in range(max(0, nt - 4))]
    if nt == 2:
        test_plan = [train_plan[int(rng.integers(2))] for _ in range(ns)]
    else:
        test_plan = [("horizontal" if bool(rng.integers(2)) else "vertical",
                      int(rng.integers(2))) for _ in range(ns)]
    anchor_index = int(rng.integers(nt))
    used, train_geometry = set(), set()
    puzzle = {"train": [], "test": []}
    for split, plan in (("train", train_plan), ("test", test_plan)):
        for index, (axis, parity) in enumerate(plan):
            for attempt in range(128):
                pair = _make_pair(_child_rng(rng), axis, parity,
                                  colors[len(puzzle["train"]) + len(puzzle["test"])],
                                  split, split == "train" and index == anchor_index)
                key = _key(pair["input"])
                info = _analyze(pair["input"])
                signature = _geometry_signature(info)
                if (key in used or key in _source_keys() or
                        (split == "test" and signature in train_geometry)):
                    continue
                used.add(key)
                if split == "train":
                    train_geometry.add(signature)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("could not construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple(tuple((_key(pair["input"]), _key(pair["output"]))
                       for pair in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_puzzle_key(p) for p in puzzles}) >= 40
    assert _puzzle_key(generate_puzzle(7)) == _puzzle_key(generate_puzzle(7))
    assert _puzzle_key(generate_puzzle(7)) != _puzzle_key(generate_puzzle(8))
    for args in ((2, 1), (10, 1), (10, 4), (None, 1),
                 (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(901, *args)
        validate_generated_puzzle(puzzle)
        if args[0] is not None:
            assert len(puzzle["train"]) == args[0]
        if args[1] is not None:
            assert len(puzzle["test"]) == args[1]
    for args in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4)):
        try:
            generate_puzzle(901, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair count accepted")
    shapes, structures, depths, end_margins, side_margins = (
        set(), set(), set(), set(), set())
    palettes, test_axes, test_parities, relations = set(), set(), set(), set()
    first_test_cases, width_relations, combined = set(), set(), set()
    within_depth = within_ends = within_sides = within_spans = within_palette = 0
    stable, mixed, asymmetric, offset, short, long = (0,) * 6
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train = [_analyze(p["input"]) for p in puzzle["train"]]
        test = [_analyze(p["input"]) for p in puzzle["test"]]
        local_palettes = {tuple(sorted((x["background"],) + x["colors"]))
                          for x in train + test}
        stable += len(local_palettes) == 1
        mixed += len(local_palettes) > 1
        within_depth += len({x["depths"] for x in train + test}) > 1
        within_ends += len({x["ends"] for x in train + test}) > 1
        within_sides += len({x["sides"] for x in train + test}) > 1
        within_spans += len({x["spans"] for x in train + test}) > 1
        within_palette += len(local_palettes) > 1
        assert train[1]["background"] in train[0]["colors"]
        assert {x["axis"] for x in train} == {"horizontal", "vertical"}
        assert {x["gap"] % 2 for x in train} == {0, 1}
        assert max(x["gap"] for x in train) >= 8
        first_test_cases.add((test[0]["axis"], test[0]["gap"] % 2))
        for x in test:
            test_axes.add(x["axis"])
            test_parities.add(x["gap"] % 2)
            peers = [y for y in train if y["axis"] == x["axis"]]
            same_axis = [y["gap"] for y in peers]
            relations.add("longer" if x["gap"] > max(same_axis) else
                          "shorter" if x["gap"] < min(same_axis) else "within")
            transverse_index = 1 if x["axis"] == "vertical" else 0
            width = x["shape"][transverse_index]
            peer_widths = [y["shape"][transverse_index] for y in peers]
            width_relations.add("broader" if width > max(peer_widths) else
                                "narrower" if width < min(peer_widths)
                                else "within")
        for x in train + test:
            shapes.add(x["shape"])
            structures.add(_geometry_signature(x))
            depths.update(x["depths"])
            end_margins.update(x["ends"])
            side_margins.update(x["sides"])
            palettes.add(tuple(sorted((x["background"],) + x["colors"])))
            asymmetric += x["spans"][0] != x["spans"][1]
            offset += x["offset"] != 0
            short += x["gap"] in (2, 3)
            long += x["gap"] >= 16
            if x["gap"] >= 16 and (x["spans"][0] != x["spans"][1] or x["offset"]):
                combined.add("long_asymmetric")
            if x["gap"] in (2, 3) and x["axis"] == "horizontal" and x["gap"] % 2:
                combined.add("short_horizontal_odd")
    assert stable >= 3 and mixed >= 15
    assert len(shapes) >= 35 and len(structures) >= 100
    assert depths == {1, 2, 3}
    assert end_margins == {1, 2, 3}
    assert side_margins >= {1, 2, 3}
    assert len(palettes) >= 35
    assert test_axes == {"horizontal", "vertical"}
    assert test_parities == {0, 1}
    assert len(first_test_cases) == 4
    assert relations == {"longer", "shorter", "within"}
    assert width_relations == {"broader", "narrower", "within"}
    assert within_depth >= 30 and within_ends >= 30
    assert within_sides >= 30 and within_spans >= 30
    assert within_palette >= 15
    assert asymmetric >= 10 and offset >= 8
    assert short >= 8 and long >= 8
    assert combined == {"long_asymmetric", "short_horizontal_odd"}
