import numpy as np


_SOURCE_INPUTS = {
    tuple(s.splitlines())
    for s in (
        "0000000000\n0003330000\n0003330000\n0002220000\n0002220000\n0004440000\n0004440000\n0001110000\n0001110000\n0000000000",
        "0000770000\n0000770000\n0002222000\n0002222000\n0066666600\n0066666600\n0066666600\n0066666600\n0333333330\n0333333330\n0333333330\n0333333330",
        "001100\n001100\n002200\n002200\n006600\n006600\n007700\n007700\n000000",
        "0000000000\n0000220000\n0000220000\n0003333000\n0003333000\n0003333000\n0003333000\n0044444400\n0044444400\n0044444400",
        "0000000\n0000000\n0000000\n0000000\n0011000\n0011000\n0044400\n0044400",
    )
}


def _rows(grid):
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _overlap(a, b):
    return max(a[2], b[2]) < min(a[2] + a[3], b[2] + b[3])


def _decode_with_background(grid, background):
    active = np.flatnonzero(np.any(grid != background, axis=1))
    if not len(active) or not np.array_equal(active, np.arange(active[0], active[-1] + 1)):
        return None
    tiers = []
    for y in active:
        xs = np.flatnonzero(grid[y] != background)
        if not len(xs) or not np.array_equal(xs, np.arange(xs[0], xs[-1] + 1)):
            return None
        color = int(grid[y, xs[0]])
        if not np.all(grid[y, xs] == color):
            return None
        tier = (color, 1, int(xs[0]), len(xs))
        if tiers and (tiers[-1][0], tiers[-1][2], tiers[-1][3]) == (tier[0], tier[2], tier[3]):
            old = tiers[-1]
            tiers[-1] = (old[0], old[1] + 1, old[2], old[3])
        else:
            if tiers and tiers[-1][0] == color:
                return None
            tiers.append(tier)
    if len(tiers) < 2 or not all(_overlap(a, b) for a, b in zip(tiers, tiers[1:])):
        return None
    return int(active[0]), tiers


def _parse_tiers(grid):
    possibilities = []
    for background in np.unique(grid):
        parsed = _decode_with_background(grid, int(background))
        if parsed is not None:
            possibilities.append((int(background), *parsed))
    assert len(possibilities) == 1, "background or tier segmentation is ambiguous"
    return possibilities[0]


def _render(shape, top, tiers, background):
    grid = np.full(shape, background, dtype=np.int8)
    y = top
    for color, height, left, width in tiers:
        grid[y:y + height, left:left + width] = color
        y += height
    assert y <= shape[0]
    return grid


def _sample_heights(rng, count, narrow=False):
    while True:
        if narrow:
            heights = [int(rng.choice((1, 2, 2, 3, 3, 4))) for _ in range(count)]
        else:
            heights = [int(rng.choice((1, 2, 2, 3, 3, 4, 4, 5, 6))) for _ in range(count)]
        if sum(heights) <= 27:
            return heights


def _sample_width(rng, narrow):
    if narrow:
        return int(rng.integers(1, 4))
    p = rng.random()
    if p < .12:
        return int(rng.integers(1, 3))
    if p < .82:
        return int(rng.integers(2, 10))
    return int(rng.integers(10, 19))


def _common_positions(rng, widths, relation):
    lefts = [int(rng.integers(1 - width, 1)) for width in widths]
    if relation in ("same", "height", "width"):
        common = int(rng.integers(max(1 - widths[0], 1 - widths[-1]), 1))
        lefts[0] = lefts[-1] = common
    else:
        possible = [x for x in range(1 - widths[-1], 1) if x != lefts[0]]
        if not possible:
            return None
        lefts[-1] = int(rng.choice(possible))
    return lefts


def _staggered_geometry(rng, count):
    span = int(rng.integers(6, 13))
    left_width = int(rng.integers(2, min(5, span - 2)))
    right_width = int(rng.integers(2, min(5, span - left_width + 1)))
    widths = [left_width] + [span] * (count - 2) + [right_width]
    lefts = [0] + [0] * (count - 2) + [span - right_width]
    return widths, lefts


def _geometry(rng, count, relation, layout, simple=False):
    if simple:
        height = int(rng.integers(1, 5))
        width = int(rng.integers(2, 7))
        heights, widths, lefts = [height] * count, [width] * count, [0] * count
    elif layout == "staggered":
        heights = _sample_heights(rng, count)
        widths, lefts = _staggered_geometry(rng, count)
    else:
        narrow = rng.random() < .18
        heights = _sample_heights(rng, count, narrow)
        widths = [_sample_width(rng, narrow) for _ in range(count)]
        if relation in ("same", "width", "position"):
            heights[-1] = heights[0]
        else:
            choices = [h for h in range(1, 7) if h != heights[0]]
            heights[-1] = int(rng.choice(choices))
        if relation in ("same", "height", "position"):
            widths[-1] = widths[0]
        else:
            choices = [w for w in range(1 if narrow else 2, 4 if narrow else 19) if w != widths[0]]
            widths[-1] = int(rng.choice(choices))
        if relation == "position" and widths[0] < 2:
            widths[0] = widths[-1] = 2
        lefts = _common_positions(rng, widths, relation)
        if lefts is None:
            return None
    if sum(heights) > 29 or any(h == 1 and w == 1 for h, w in zip(heights, widths)):
        return None
    left_shift = min(lefts)
    lefts = [x - left_shift for x in lefts]
    span = max(x + w for x, w in zip(lefts, widths))
    if span > 28:
        return None
    return heights, widths, lefts, span


def _choose_colors(rng, count, colors, background, unique=False, repeat=False, force_zero=False, unseen=None):
    pool = [int(c) for c in colors if c != background]
    if unique or count <= len(pool):
        chosen = [int(c) for c in rng.choice(pool, size=count, replace=False)]
    else:
        chosen = [int(c) for c in rng.choice(pool, size=count, replace=True)]
    if force_zero and 0 in pool and 0 not in chosen:
        chosen[int(rng.integers(count))] = 0
    if unseen is not None and unseen not in chosen:
        chosen[int(rng.integers(count))] = unseen
    if repeat and count >= 5:
        safe = [(i, j) for i in range(count) for j in range(i + 2, count)
                if (i, j) != (0, count - 1)
                and abs(([count - 1, *range(1, count - 1), 0].index(i)) -
                         ([count - 1, *range(1, count - 1), 0].index(j))) > 1]
        if safe:
            i, j = safe[int(rng.integers(len(safe)))]
            chosen[j] = chosen[i]
    order = [count - 1, *range(1, count - 1), 0]
    if any(chosen[i] == chosen[j] for i, j in zip(range(count - 1), range(1, count))):
        return None
    if any(chosen[i] == chosen[j] for i, j in zip(order, order[1:])):
        return None
    if chosen[0] == chosen[-1]:
        return None
    return chosen


def _make_pair(rng, background, color_pool, count, relation, layout, simple=False,
               unseen=None, force_zero=False, larger_than=None, unique=False):
    geometry = _geometry(rng, count, relation, layout, simple)
    if geometry is None:
        return None
    heights, widths, lefts, span = geometry
    if layout == "staggered" and max(lefts[0], lefts[-1]) < min(lefts[0] + widths[0], lefts[-1] + widths[-1]):
        return None
    repeat = rng.random() < .28 and count >= 5 and unseen is None and not unique
    colors = _choose_colors(rng, count, color_pool, background,
                            unique=unique or simple, repeat=repeat, force_zero=force_zero, unseen=unseen)
    if colors is None:
        return None
    if simple:
        margin_left = int(rng.integers(0, 5))
        margin_right = int(rng.integers(0, 5))
    else:
        margin_left = int(rng.choice((0, 0, 1, 2, 3, 5, 8)))
        margin_right = int(rng.choice((0, 0, 1, 2, 3, 5, 8)))
    if span + margin_left + margin_right > 30:
        return None
    lefts = [x + margin_left for x in lefts]
    top = int(rng.choice((0, 0, 1, 2, 3, 5)))
    bottom = int(rng.choice((0, 0, 1, 2, 3, 5)))
    if top + sum(heights) + bottom > 30:
        return None
    height = top + sum(heights) + bottom
    width = span + margin_left + margin_right
    if larger_than is not None and rng.random() < .65:
        old_height, old_width = larger_than
        if rng.random() < .5 and height <= old_height and old_height < 30 and old_height + 1 - height <= 6:
            bottom += old_height + 1 - height
            height = old_height + 1
        elif width <= old_width and old_width < 30 and old_width + 1 - width <= 6:
            margin_right += old_width + 1 - width
            width = old_width + 1
    tiers = [(colors[i], heights[i], lefts[i], widths[i]) for i in range(count)]
    reordered = [tiers[-1], *tiers[1:-1], tiers[0]]
    if not all(_overlap(a, b) for a, b in zip(reordered, reordered[1:])):
        return None
    a = _render((height, width), top, tiers, background)
    b = _render((height, width), top, reordered, background)
    try:
        bg_a, parsed_top, parsed = _parse_tiers(a)
        bg_b, parsed_out_top, parsed_out = _parse_tiers(b)
    except AssertionError:
        return None
    if bg_a != background or bg_b != background or parsed_top != top or parsed_out_top != top:
        return None
    if parsed != tiers or parsed_out != reordered or np.array_equal(a, b):
        return None
    return {"input": a, "output": b}


def _check_count(value, low, high, name):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    _check_count(num_train, 2, 10, "num_train")
    _check_count(num_test, 1, 4, "num_test")
    if num_train is None:
        num_train = int(rng.integers(max(2, (num_test or 0) + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")

    background = 0 if rng.random() < .58 else int(rng.integers(1, 10))
    all_tier_colors = [c for c in range(10) if c != background]
    rng.shuffle(all_tier_colors)
    train_colors = all_tier_colors[:6]
    if background != 0 and 0 not in train_colors:
        train_colors[-1] = 0
    reserved_colors = [c for c in all_tier_colors if c not in train_colors]
    relations = ("same", "height", "width", "position", "all")
    train_plan = []
    anchor = int(rng.integers(num_train))
    for i in range(num_train):
        count = int(rng.integers(2, 7))
        if i == anchor:
            count = int(rng.integers(4, 6))
        relation = str(rng.choice(relations))
        simple = i == anchor and rng.random() < .38
        if simple:
            relation = "same"
        layout = "staggered" if not simple and count >= 3 and relation in ("position", "all") and rng.random() < .25 else "common"
        train_plan.append((count, relation, layout, simple))
    if not any(relation == "all" and layout == "common" for _, relation, layout, _ in train_plan):
        demonstrator = (anchor + 1) % num_train
        count = train_plan[demonstrator][0]
        train_plan[demonstrator] = (count, "all", "common", False)
    if num_train >= 3 and len({p[0] for p in train_plan}) == 1:
        j = (anchor + 1) % num_train
        train_plan[j] = (2 if train_plan[j][0] != 2 else 6, *train_plan[j][1:])
    test_counts = [int(rng.choice((2, 2, 3, 3, 4, 4, 5, 6))) for _ in range(num_test)]
    if num_test >= 3 and not any(n >= 5 for n in test_counts):
        test_counts[int(rng.integers(num_test))] = int(rng.choice((5, 6)))

    puzzle = {"train": [], "test": []}
    seen = set()
    for i, (count, relation, layout, simple) in enumerate(train_plan):
        for _ in range(400):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            pair = _make_pair(pair_rng, background, train_colors, count, relation, layout,
                              simple=simple, force_zero=(i == anchor and background != 0),
                              unique=(i == anchor))
            if pair is not None:
                sig = _rows(pair["input"])
                if sig not in seen and sig not in _SOURCE_INPUTS:
                    break
        else:
            raise RuntimeError("Could not construct a unique training pair")
        puzzle["train"].append(pair)
        seen.add(sig)
    largest_train = (max(p["input"].shape[0] for p in puzzle["train"]),
                     max(p["input"].shape[1] for p in puzzle["train"]))
    for count in test_counts:
        for _ in range(400):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            relation = str(pair_rng.choice(relations))
            layout = "staggered" if count >= 3 and pair_rng.random() < .35 else "common"
            unseen = int(pair_rng.choice(reserved_colors)) if pair_rng.random() < .52 else None
            pool = train_colors + reserved_colors
            pair = _make_pair(pair_rng, background, pool, count, relation, layout,
                              unseen=unseen, larger_than=largest_train)
            if pair is not None:
                sig = _rows(pair["input"])
                if sig not in seen and sig not in _SOURCE_INPUTS:
                    break
        else:
            raise RuntimeError("Could not construct a unique test pair")
        puzzle["test"].append(pair)
        seen.add(sig)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen = set()
    backgrounds = set()
    train_tiers = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = pair["input"], pair["output"]
            for grid in (a, b):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert a.shape == b.shape and not np.array_equal(a, b)
            sig = _rows(a)
            assert sig not in seen and sig not in _SOURCE_INPUTS
            seen.add(sig)
            bg, top, tiers = _parse_tiers(a)
            out_bg, out_top, out_tiers = _parse_tiers(b)
            assert bg == out_bg and top == out_top
            assert out_tiers == [tiers[-1], *tiers[1:-1], tiers[0]]
            assert tiers[0][0] != tiers[-1][0]
            assert all(_overlap(x, y) for x, y in zip(out_tiers, out_tiers[1:]))
            assert np.array_equal(a, _render(a.shape, top, tiers, bg))
            assert np.array_equal(b, _render(b.shape, top, out_tiers, bg))
            backgrounds.add(bg)
            if split == "train":
                train_tiers.append(tiers)
    assert len(backgrounds) == 1
    assert any(len(tiers) >= 4 and len({t[0] for t in tiers}) == len(tiers)
               for tiers in train_tiers)
    assert any(tiers[0][1] != tiers[-1][1] and tiers[0][2] != tiers[-1][2]
               and tiers[0][3] != tiers[-1][3] for tiers in train_tiers)


def _signature(puzzle):
    return tuple((split, tuple((p["input"].shape, p["input"].tobytes(),
                                p["output"].tobytes()) for p in puzzle[split]))
                 for split in ("train", "test"))


def self_test() -> None:
    signatures, layouts, shapes, palettes = set(), set(), set(), set()
    train_counts, test_counts, tier_counts = set(), set(), set()
    relations, backgrounds = set(), set()
    staggered = zero_tiers = repeated = unseen = larger = narrow = edge = long_tier = 0
    mixed_train_counts = mixed_train_palettes = one_test_four = reduced_tests = 0
    train_relations, test_tier_counts, simple_positions = set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _signature(puzzle) == _signature(generate_puzzle(seed))
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_colors = set().union(*(set(np.unique(p["input"])) for p in puzzle["train"]))
        train_n, train_palettes = [], []
        max_train = (max(p["input"].shape[0] for p in puzzle["train"]),
                     max(p["input"].shape[1] for p in puzzle["train"]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, b = pair["input"], pair["output"]
                bg, top, tiers = _parse_tiers(a)
                backgrounds.add(bg)
                palettes.add(tuple(sorted(set(np.unique(a)))))
                shapes.add(a.shape)
                tier_counts.add(len(tiers))
                layouts.add((a.shape, top, tuple((h, x, w) for _, h, x, w in tiers)))
                expected = _render(a.shape, top, [tiers[-1], *tiers[1:-1], tiers[0]], bg)
                assert np.array_equal(b, expected)
                assert _rows(a) not in _SOURCE_INPUTS
                relation = (tiers[0][1] != tiers[-1][1],
                            tiers[0][3] != tiers[-1][3],
                            tiers[0][2] != tiers[-1][2])
                relations.add(relation)
                staggered += int(max(t[2] for t in tiers) >= min(t[2] + t[3] for t in tiers))
                zero_tiers += int(bg != 0 and any(t[0] == 0 for t in tiers))
                repeated += int(len(set(t[0] for t in tiers)) < len(tiers))
                narrow += int(a.shape[1] <= 6)
                edge += int(any(t[2] == 0 or t[2] + t[3] == a.shape[1] for t in tiers))
                long_tier += int(any(h >= 5 or w >= 10 or h == 1 or w == 1 for _, h, _, w in tiers))
                if split == "train":
                    train_relations.add(relation)
                    train_palettes.append(tuple(sorted(np.unique(a))))
                    if len(tiers) >= 4 and len({(t[1], t[2], t[3]) for t in tiers}) == 1:
                        simple_positions.add(len(train_n))
                    train_n.append(len(tiers))
                else:
                    test_tier_counts.add(len(tiers))
                    one_test_four += int(len(puzzle["test"]) == 1 and len(tiers) == 4)
                    reduced_tests += int(len(tiers) < max(train_n))
                    unseen += int(bool(set(np.unique(a)) - train_colors))
                    larger += int(a.shape[0] > max_train[0] or a.shape[1] > max_train[1])
        mixed_train_counts += int(len(set(train_n)) > 1)
        mixed_train_palettes += int(len(set(train_palettes)) > 1)
    assert len(signatures) == 42 and len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(layouts) >= 160 and len(shapes) >= 45 and len(palettes) >= 20
    assert tier_counts == {2, 3, 4, 5, 6}
    assert len(relations) >= 5 and len(backgrounds) >= 4
    assert mixed_train_counts >= 20
    assert mixed_train_palettes >= 20 and len(simple_positions) >= 2
    assert {(True, False, False), (False, True, False),
            (False, False, True), (True, True, True)} <= train_relations
    assert test_tier_counts == {2, 3, 4, 5, 6}
    assert one_test_four > 0 and reduced_tests > 0
    assert min(staggered, zero_tiers, repeated, unseen, larger, narrow, edge, long_tier) > 0
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        seed = 1000 + (train or 0) * 10 + (test or 0)
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        assert (train is None or len(puzzle["train"]) == train)
        assert (test is None or len(puzzle["test"]) == test)
        assert _signature(puzzle) == _signature(generate_puzzle(seed, train, test))
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(9, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {kwargs}")
