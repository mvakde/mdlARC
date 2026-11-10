"""Nested-contour ARC puzzles: retain a marker exactly inside the inner ring."""

import hashlib
import numpy as np


_SOURCE_PAIR_HASHES = frozenset({
    "b9e98fb202d11dd207bb565d6ab9e5c9b8f16155b44805c12b9717369e2c7752",
    "feec8c89891fddbc0f7e069bc1f0a9a339064963daf77cc7132fc26af850543f",
    "f4d238ebe25c769f869f7c176989d50e4ea05124038a6807c1a9e10d31c7cad8",
    "92743906c00be27485efd97afed7e67819207a67a3e7029bfb8344c597838aa2",
    "3be7bc9258f9e66846358dca98463b0426c027e02b58fc08202e88446cfab8f7",
    "1bfed3295cea510e33116cde609c69ccd6ecd08bed52f685f6d0dc21ad330491",
    "f63de35ba3875a7eecae0d27afc8eec05b202fae10678fa4c44b42c379a6fabd",
})


def _counts(rng, num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
            raise ValueError(name + " is out of range")
    legal = [(a, b) for a in range(2, 11) for b in range(1, 5)
             if a > b and (num_train is None or a == num_train)
             and (num_test is None or b == num_test)]
    if not legal:
        raise ValueError("incompatible pair counts")
    return legal[int(rng.integers(len(legal)))]


def _connected(mask, diagonals=False):
    points = np.argwhere(mask)
    if not len(points):
        return False
    start = tuple(int(v) for v in points[0])
    seen = {start}
    stack = [start]
    h, w = mask.shape
    steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if diagonals:
        steps += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    while stack:
        r, c = stack.pop()
        for dr, dc in steps:
            nr, nc = r + dr, c + dc
            if (0 <= nr < h and 0 <= nc < w and mask[nr, nc]
                    and (nr, nc) not in seen):
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _enclosed(wall):
    h, w = wall.shape
    outside = np.zeros((h, w), dtype=bool)
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if not wall[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not wall[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if (0 <= nr < h and 0 <= nc < w and not wall[nr, nc]
                    and not outside[nr, nc]):
                outside[nr, nc] = True
                stack.append((nr, nc))
    return ~outside & ~wall


def _rim(fill):
    core = fill.copy()
    core[0, :] = False
    core[-1, :] = False
    core[:, 0] = False
    core[:, -1] = False
    core[1:, :] &= fill[:-1, :]
    core[:-1, :] &= fill[1:, :]
    core[:, 1:] &= fill[:, :-1]
    core[:, :-1] &= fill[:, 1:]
    return fill & ~core


def _shape(rng, shape, box, stepped):
    top, left, height, width = box
    local = np.ones((height, width), dtype=bool)
    if stepped:
        sides = rng.choice(4, size=int(rng.integers(1, 4)), replace=False)
        for side in sides:
            length = width if side < 2 else height
            depth_limit = (height if side < 2 else width) - 4
            if length < 6 or depth_limit < 1:
                continue
            depth = int(rng.integers(1, min(6, depth_limit) + 1))
            span = int(rng.integers(2, min(length - 2, max(3, length // 2 + 2)) + 1))
            begin = int(rng.integers(1, length - span))
            if side == 0:
                local[:depth, begin:begin + span] = False
            elif side == 1:
                local[height - depth:, begin:begin + span] = False
            elif side == 2:
                local[begin:begin + span, :depth] = False
            else:
                local[begin:begin + span, width - depth:] = False
    fill = np.zeros(shape, dtype=bool)
    fill[top:top + height, left:left + width] = local
    return fill


def _wall(rng, fill, thick):
    wall = _rim(fill)
    if thick:
        core = fill & ~wall
        adjacent = np.zeros(fill.shape, dtype=bool)
        adjacent[1:, :] |= wall[:-1, :]
        adjacent[:-1, :] |= wall[1:, :]
        adjacent[:, 1:] |= wall[:, :-1]
        adjacent[:, :-1] |= wall[:, 1:]
        candidates = np.argwhere(core & adjacent)
        if len(candidates):
            r, c = candidates[int(rng.integers(len(candidates)))]
            if rng.random() < 0.5:
                selected = candidates[(candidates[:, 0] == r)
                                      & (abs(candidates[:, 1] - c) <= int(rng.integers(1, 5)))]
            else:
                selected = candidates[(candidates[:, 1] == c)
                                      & (abs(candidates[:, 0] - r) <= int(rng.integers(1, 5)))]
            wall[selected[:, 0], selected[:, 1]] = True
    return wall


def _dimension(rng, regime):
    if rng.random() < (0.26 if regime == "target_test" else 0.16):
        return 14, 14
    band = int(rng.choice(3, p=(0.22, 0.53, 0.25) if regime == "source_train"
                          else (0.20, 0.40, 0.40)))
    ranges = ((9, 14), (13, 22), (21, 30))
    lo, hi = ranges[band]
    h = int(rng.integers(lo, hi + 1))
    if rng.random() < 0.35:
        lo, hi = ranges[int(rng.integers(3))]
    w = int(rng.integers(lo, hi + 1))
    return h, w


def _geometry(rng, regime, keep, paired=False):
    for _ in range(320):
        adjacent = keep and not paired and rng.random() < (0.19 if regime == "target_test" else 0.10)
        if adjacent:
            if rng.random() < 0.3:
                h, w = 14, 14
            else:
                h = int(rng.integers(9, 19))
                w = int(rng.integers(9, 19))
            oh = int(rng.integers(6, min(10, h) + 1))
            ow = int(rng.integers(6, min(10, w) + 1))
            ih, iw = oh - 2, ow - 2
            shift_r = shift_c = 1
            outer_step = inner_step = outer_thick = inner_thick = False
        else:
            h, w = _dimension(rng, regime)
            p = 0.48 if regime == "source_train" else 0.68
            outer_step = h >= 11 and w >= 11 and rng.random() < p
            inner_step = h >= 11 and w >= 11 and rng.random() < p
            outer_min = 12 if outer_step and inner_step else (10 if outer_step or inner_step else 6)
            if h < outer_min or w < outer_min:
                outer_step = inner_step = False
                outer_min = 6
            oh = int(rng.integers(outer_min, h + 1))
            ow = int(rng.integers(outer_min, w + 1))
            if oh == h and ow == w:
                oh -= 1
            minimum_inner = 7 if inner_step else 4
            minimum_gap = 2 if outer_step else 1
            if oh < minimum_inner + 2 * minimum_gap or ow < minimum_inner + 2 * minimum_gap:
                continue
            ih = int(rng.integers(minimum_inner, oh - 2 * minimum_gap + 1))
            iw = int(rng.integers(minimum_inner, ow - 2 * minimum_gap + 1))
            shift_r = int(rng.integers(minimum_gap, oh - ih - minimum_gap + 1))
            shift_c = int(rng.integers(minimum_gap, ow - iw - minimum_gap + 1))
            outer_thick = rng.random() < 0.27
            inner_thick = rng.random() < 0.27
        top = int(rng.integers(h - oh + 1))
        left = int(rng.integers(w - ow + 1))
        if h - oh >= 2 and w - ow >= 2 and rng.random() < 0.45:
            top = int(rng.integers(1, h - oh))
            left = int(rng.integers(1, w - ow))
        elif rng.random() < 0.30:
            side = int(rng.integers(4))
            if side == 0:
                top = 0
            elif side == 1:
                top = h - oh
            elif side == 2:
                left = 0
            else:
                left = w - ow
        outer_fill = _shape(rng, (h, w), (top, left, oh, ow), outer_step)
        inner_fill = _shape(rng, (h, w),
                            (top + shift_r, left + shift_c, ih, iw), inner_step)
        outer_wall = _wall(rng, outer_fill, outer_thick)
        inner_wall = _wall(rng, inner_fill, inner_thick)
        if np.any(outer_wall & inner_wall):
            continue
        outer_area = _enclosed(outer_wall)
        inner_area = _enclosed(inner_wall)
        if (not np.any(inner_area) or not _connected(inner_area)
                or not _connected(outer_area) or not _connected(outer_wall, True)
                or not _connected(inner_wall, True)
                or not np.all(outer_area[inner_fill])):
            continue
        between = outer_area & ~inner_fill
        if not np.any(between) and (not keep or paired or not adjacent):
            continue
        if adjacent and np.any(between):
            continue
        return outer_wall, inner_wall, inner_area, between
    raise RuntimeError("could not construct nested contours")


def _pair(rng, geometry, background, marker, colors, keep):
    outer_wall, inner_wall, inner_area, between = geometry
    eligible = inner_area if keep else between
    points = np.argwhere(eligible)
    r, c = points[int(rng.integers(len(points)))]
    grid = np.full(outer_wall.shape, background, dtype=np.int8)
    grid[outer_wall] = colors[0]
    grid[inner_wall] = colors[1]
    grid[r, c] = marker
    output = grid.copy()
    if not keep:
        output[r, c] = background
    return {"input": grid, "output": output}


def _pair_hash(pair):
    a, b = pair["input"], pair["output"]
    payload = bytes(a.shape) + a.astype(np.int8).tobytes() + b.astype(np.int8).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _rule_info(pair, background, marker):
    grid, output = pair["input"], pair["output"]
    assert grid.shape == output.shape
    assert int(np.count_nonzero(grid == marker)) == 1
    colors = [int(x) for x in np.unique(grid) if x not in (background, marker)]
    assert len(colors) == 2
    walls = [grid == color for color in colors]
    areas = [_enclosed(wall) for wall in walls]
    assert all(_connected(wall, True) and _connected(area)
               for wall, area in zip(walls, areas))
    outer_index = int(areas[0].sum() < areas[1].sum())
    inner_index = 1 - outer_index
    outer, inner = walls[outer_index], walls[inner_index]
    outer_area, inner_area = areas[outer_index], areas[inner_index]
    assert int(outer_area.sum()) > int(inner_area.sum())
    assert np.all(outer_area[inner | inner_area])
    assert not np.any(outer & inner)
    point = tuple(int(v) for v in np.argwhere(grid == marker)[0])
    assert outer_area[point] and not (outer | inner)[point]
    keep = bool(inner_area[point])
    expected = grid.copy()
    if not keep:
        expected[point] = background
    assert np.array_equal(output, expected)
    outer_fill = outer | outer_area
    inner_fill = inner | inner_area
    assert _connected(outer_area) and _connected(inner_area)
    def bounds(fill):
        rows, cols = np.where(fill)
        return (int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max()))
    def deficit(fill):
        r0, r1, c0, c1 = bounds(fill)
        return (r1 - r0 + 1) * (c1 - c0 + 1) - int(fill.sum())
    return {
        "keep": keep, "shape": grid.shape, "colors": (colors[outer_index], colors[inner_index]),
        "marker": point, "outer_box": bounds(outer_fill), "inner_box": bounds(inner_fill),
        "outer_step": deficit(outer_fill), "inner_step": deficit(inner_fill),
        "outer_thick": int((outer & ~_rim(outer_fill)).sum()),
        "inner_thick": int((inner & ~_rim(inner_fill)).sum()),
        "annulus": int((outer_area & ~inner_fill).sum()),
        "edge_sides": (bool(np.any(outer[0, :])), bool(np.any(outer[-1, :])),
                       bool(np.any(outer[:, 0])), bool(np.any(outer[:, -1]))),
        "edge_count": int(any(outer[0, :])) + int(any(outer[-1, :]))
                      + int(any(outer[:, 0])) + int(any(outer[:, -1])),
        "inner_size": (bounds(inner_fill)[1] - bounds(inner_fill)[0] + 1,
                       bounds(inner_fill)[3] - bounds(inner_fill)[2] + 1),
    }


def _learn_convention(train):
    changes = []
    for pair in train:
        a, b = pair["input"], pair["output"]
        locations = np.argwhere(a != b)
        if len(locations):
            assert len(locations) == 1
            r, c = locations[0]
            changes.append((int(b[r, c]), int(a[r, c])))
    assert changes and len(set(changes)) == 1
    return changes[0]


def validate_generated_puzzle(puzzle: dict) -> None:
    assert type(puzzle) is dict and set(puzzle) == {"train", "test"}
    assert type(puzzle["train"]) is list and type(puzzle["test"]) is list
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen = set()
    for pair in puzzle["train"] + puzzle["test"]:
        assert type(pair) is dict and set(pair) == {"input", "output"}
        for grid in pair.values():
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
            assert np.issubdtype(grid.dtype, np.integer)
            assert np.all((grid >= 0) & (grid <= 9))
        assert pair["input"].shape == pair["output"].shape
        key = (pair["input"].shape, pair["input"].tobytes())
        assert key not in seen
        seen.add(key)
        assert _pair_hash(pair) not in _SOURCE_PAIR_HASHES
    background, marker = _learn_convention(puzzle["train"])
    assert background != marker
    train_info = [_rule_info(p, background, marker) for p in puzzle["train"]]
    test_info = [_rule_info(p, background, marker) for p in puzzle["test"]]
    assert {x["keep"] for x in train_info} == {False, True}
    training_colors = set().union(*(x["colors"] for x in train_info))
    assert any(set(x["colors"]) - training_colors for x in test_info)
    assert all(int(np.count_nonzero(p["input"] == marker)) == 1
               for p in puzzle["test"])
    for pair in puzzle["test"]:
        assert np.array_equal(_predict_from_training(puzzle["train"], pair["input"]),
                              pair["output"])


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    background = int(rng.integers(10))
    marker = int(rng.choice([x for x in range(10) if x != background]))
    other_colors = [x for x in range(10) if x not in (background, marker)]
    shuffled = [int(x) for x in rng.permutation(other_colors)]
    train_palette = shuffled[:int(rng.integers(2, 6))]
    unseen = shuffled[len(train_palette):]
    train = []
    test = []
    used = set()
    outcomes = [True, False] + [bool(rng.integers(2)) for _ in range(num_train - 2)]
    rng.shuffle(outcomes)
    pair_twins = num_train >= 3 and rng.random() < 0.42
    twin_geometry = None
    twin_colors = None
    twin_indices = []
    if pair_twins:
        twin_indices = [outcomes.index(True), outcomes.index(False)]
    for regime, count in (("source_train", num_train), ("target_test", num_test)):
        for j in range(count):
            keep = outcomes[j] if regime == "source_train" else bool(rng.integers(2))
            if regime == "source_train" and j in twin_indices and twin_geometry is not None:
                geometry = twin_geometry
                colors = twin_colors
                pair_rng = np.random.default_rng(int(rng.integers(2**63 - 1)))
                pair = _pair(pair_rng, geometry, background, marker, colors, keep)
                train.append(pair)
                used.add((pair["input"].shape, pair["input"].tobytes()))
                continue
            for _ in range(80):
                pair_rng = np.random.default_rng(int(rng.integers(2**63 - 1)))
                geometry = _geometry(pair_rng, regime, keep,
                                     paired=(regime == "source_train" and j in twin_indices))
                if regime == "source_train":
                    colors = tuple(int(v) for v in pair_rng.choice(train_palette, 2, replace=False))
                elif j == 0:
                    colors = (int(pair_rng.choice(unseen)),
                              int(pair_rng.choice(train_palette)))
                    if pair_rng.random() < 0.5:
                        colors = colors[::-1]
                else:
                    colors = tuple(int(v) for v in pair_rng.choice(other_colors, 2, replace=False))
                pair = _pair(pair_rng, geometry, background, marker, colors, keep)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in used and _pair_hash(pair) not in _SOURCE_PAIR_HASHES:
                    used.add(key)
                    (train if regime == "source_train" else test).append(pair)
                    if regime == "source_train" and j in twin_indices:
                        twin_geometry, twin_colors = geometry, colors
                    break
            else:
                raise RuntimeError("could not avoid duplicate or source pair")
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _equal_puzzles(a, b):
    return all(len(a[k]) == len(b[k]) and all(
        np.array_equal(x["input"], y["input"])
        and np.array_equal(x["output"], y["output"])
        for x, y in zip(a[k], b[k])) for k in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert all(_equal_puzzles(p, generate_puzzle(i)) for i, p in enumerate(puzzles))
    assert len({p["train"][0]["input"].tobytes() for p in puzzles}) >= 40
    infos = []
    within = []
    for p in puzzles:
        validate_generated_puzzle(p)
        bg, marker = _learn_convention(p["train"])
        entries = [_rule_info(q, bg, marker) for q in p["train"] + p["test"]]
        infos.extend(entries)
        within.append(entries)
        assert all(_predict_from_training(p["train"], q["input"])
                   is not None and np.array_equal(
                       _predict_from_training(p["train"], q["input"]), q["output"])
                   for q in p["test"])
    assert len({x["shape"] for x in infos}) >= 35
    assert len({x["colors"] for x in infos}) >= 35
    assert len({(x["shape"], x["outer_box"], x["inner_box"], x["marker"])
                for x in infos}) >= 180
    assert {x["keep"] for x in infos} == {True, False}
    assert any(x["outer_step"] and not x["inner_step"] for x in infos)
    assert any(x["inner_step"] and not x["outer_step"] for x in infos)
    assert any(x["outer_step"] and x["inner_step"] for x in infos)
    assert any(x["outer_step"] >= 20 for x in infos)
    assert sum(bool(x["outer_thick"] or x["inner_thick"]) for x in infos) >= 35
    assert any(x["annulus"] == 0 for x in infos)
    assert any(1 <= x["annulus"] <= 10 for x in infos)
    assert any(x["annulus"] >= 30 for x in infos)
    assert any(x["edge_count"] >= 2 for x in infos)
    assert any(x["edge_count"] == 0 for x in infos)
    assert all(any(x["edge_sides"][side] for x in infos) for side in range(4))
    assert any(x["shape"] == (14, 14) for x in infos)
    assert any(x["shape"][0] != x["shape"][1] for x in infos)
    assert min(x["shape"][0] for x in infos) <= 10
    assert max(x["shape"][0] for x in infos) >= 28
    assert len({_learn_convention(p["train"])[0] for p in puzzles}) >= 8
    assert len({_learn_convention(p["train"])[1] for p in puzzles}) >= 8
    assert {x["keep"] for x in infos if x["outer_step"] or x["inner_step"]} == {True, False}
    assert {x["keep"] for x in infos if not x["outer_step"] and not x["inner_step"]} == {True, False}
    assert {x["keep"] for x in infos if max(x["inner_size"]) <= 5} == {True, False}
    assert any(len({x["shape"] for x in group}) > 1 for group in within)
    assert any(len({x["colors"] for x in group}) > 1 for group in within)
    assert any(len({x["annulus"] for x in group}) > 1 for group in within)
    assert sum(any(x["edge_count"] for x in group) and
               any(not x["edge_count"] for x in group) for group in within) >= 20
    for p in puzzles:
        background, marker = _learn_convention(p["train"])
        training_colors = set().union(*(
            _rule_info(t, background, marker)["colors"] for t in p["train"]
        ))
        assert any(set(_rule_info(q, background, marker)["colors"])
                   - training_colors for q in p["test"])
    for counts in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                   (2, None), (10, None)):
        p = generate_puzzle(901, *counts)
        validate_generated_puzzle(p)
        assert (counts[0] is None or len(p["train"]) == counts[0])
        assert (counts[1] is None or len(p["test"]) == counts[1])
        assert _equal_puzzles(p, generate_puzzle(901, *counts))
    for counts in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (True, 1), (2.5, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(123, 2, 1)
    bad["test"][0]["output"] = bad["test"][0]["output"].copy()
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong output accepted")


def _predict_from_training(train, test_input):
    background, marker = _learn_convention(train)
    colors = [int(x) for x in np.unique(test_input) if x not in (background, marker)]
    if len(colors) != 2:
        return None
    areas = [_enclosed(test_input == color) for color in colors]
    inner = areas[int(areas[0].sum() > areas[1].sum())]
    point = tuple(int(v) for v in np.argwhere(test_input == marker)[0])
    prediction = test_input.copy()
    if not inner[point]:
        prediction[point] = background
    return prediction
