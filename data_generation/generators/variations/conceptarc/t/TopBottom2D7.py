import numpy as np


_SOURCE_INPUTS = {
    ("11111100", "00000000", "00000000", "00000000", "00033333", "00000000", "00000000", "00000000"),
    ("00000000", "00000000", "03330000", "00011111", "00000000", "00000000", "00000000", "00000000"),
    ("00000000", "00000000", "00000000", "01111000", "00000000", "00333330", "00000000", "00000000"),
    ("00033300", "00033300", "00000000", "00000000", "01100000", "01100000", "00000000", "00000000"),
    ("00000000", "00000000", "01000000", "00000000", "00000000", "00000300", "00000000", "00000000"),
    ("00000000", "00000000", "01100000", "00000000", "00003333", "00000000", "00000000", "00000000"),
    ("0000000000", "0000000000", "0444400000", "0000000000", "0000000000", "0000000000", "6666000000", "0000000000", "0000000000", "0000000000"),
    ("0000000000", "0000000000", "0000777770", "0000777770", "0000777770", "0000000000", "0444444400", "0444444400", "0000000000", "0000000000"),
}


def _rows(grid):
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _checked_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} through {high}")
    return int(value)


def _choose_counts(rng, num_train, num_test):
    num_train = _checked_count(num_train, "num_train", 2, 10)
    num_test = _checked_count(num_test, "num_test", 1, 4)
    if num_train is None:
        num_train = int(rng.integers(max(2, (num_test or 0) + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _kind(h, w):
    if h == w == 1:
        return "pixel"
    if h == 1:
        return "row"
    if w == 1:
        return "column"
    if h == w:
        return "square"
    if h >= 3 and w >= 4:
        return "large_block"
    return "block"


def _shape_options(height, width, relation, overlap, dominance):
    shapes = [(h, w) for h in range(1, min(6, height - 1) + 1)
              for w in range(1, min(9, width - 1) + 1)]
    groups = {}
    for uh, uw in shapes:
        for lh, lw in shapes:
            sign = (uh * uw > lh * lw) - (uh * uw < lh * lw)
            if relation is not None and sign != relation:
                continue
            if dominance and (_sign(uh, lh) != relation or _sign(uw, lw) != relation):
                continue
            if uh * uw + lh * lw >= height * width * 0.46:
                continue
            if overlap:
                if uh < 2 or uw + lw + 1 > width or max(uh, lh + 1) > height:
                    continue
            elif uh + lh > height:
                continue
            groups.setdefault((_kind(uh, uw), _kind(lh, lw)), []).append((uh, uw, lh, lw))
    return groups


def _choose_dimensions(rng, height, width, relation, overlap, preferred, dominance):
    groups = _shape_options(height, width, relation, overlap, dominance)
    assert groups
    keys = [key for key in groups if key[0] == preferred]
    if not keys:
        keys = list(groups)
    key = keys[int(rng.integers(len(keys)))]
    options = groups[key]
    return options[int(rng.integers(len(options)))]


def _choose_positions(rng, height, width, dims, overlap, x_sign):
    uh, uw, lh, lw = dims
    if overlap:
        upper_y = int(rng.integers(0, min(height - uh, height - lh - 1) + 1))
        delta = int(rng.integers(1, min(uh - 1, height - lh - upper_y) + 1))
        lower_y = upper_y + delta
    else:
        gap = int(rng.integers(0, min(3, height - uh - lh) + 1))
        upper_y = int(rng.integers(0, height - uh - lh - gap + 1))
        lower_y = upper_y + uh + gap
    x_options = []
    for upper_x in range(width - uw + 1):
        for lower_x in range(width - lw + 1):
            sign = (upper_x > lower_x) - (upper_x < lower_x)
            if x_sign is not None and sign != x_sign:
                continue
            if overlap and not (upper_x + uw < lower_x or lower_x + lw < upper_x):
                continue
            x_options.append((upper_x, lower_x))
    if not x_options:
        return None
    upper_x, lower_x = x_options[int(rng.integers(len(x_options)))]
    return upper_y, upper_x, lower_y, lower_x


def _background(rng, colors, nonzero=False):
    options = [c for c in range(10) if c not in colors and (c != 0 or not nonzero)]
    if not nonzero and 0 in options and rng.random() < 0.62:
        return 0
    return int(options[int(rng.integers(len(options)))])


def _pair_colors(rng, base, stable, mode, allow_zero):
    if stable or mode == "base":
        return base
    available = [c for c in range(10) if c not in base and (allow_zero or c != 0)]
    if mode == "partial":
        return (base[int(rng.integers(2))], int(available[int(rng.integers(len(available)))]))
    fresh = rng.choice(available, size=2, replace=False)
    return int(fresh[0]), int(fresh[1])


def _make_pair(rng, canvas, colors, background, relation, x_sign, overlap, preferred, selected_color, seen, dominance=False):
    height, width = canvas
    for attempt in range(80):
        dims = _choose_dimensions(rng, height, width, relation, overlap, preferred, dominance)
        positions = _choose_positions(rng, height, width, dims, overlap, x_sign)
        if positions is None:
            continue
        uh, uw, lh, lw = dims
        uy, ux, ly, lx = positions
        upper_color = selected_color if selected_color is not None else colors[int(rng.integers(2))]
        lower_color = colors[0] if upper_color == colors[1] else colors[1]
        grid = np.full(canvas, background, dtype=np.int8)
        grid[uy:uy + uh, ux:ux + uw] = upper_color
        grid[ly:ly + lh, lx:lx + lw] = lower_color
        key = (grid.shape, grid.tobytes())
        if key in seen or _rows(grid) in _SOURCE_INPUTS:
            continue
        seen.add(key)
        return {"input": grid, "output": np.full((uh, uw), upper_color, dtype=np.int8)}
    raise RuntimeError("Could not construct a unique rectangle pair")


def _canvas(rng, regime, index):
    if regime == "train":
        choices = ((8, 8), (8, 8), (8, 8), (7, 9), (9, 7), (9, 10), (10, 9), (10, 10), (10, 12), (12, 10))
    else:
        choices = ((8, 8), (10, 10), (10, 10), (10, 12), (12, 10), (9, 13), (13, 9), (11, 11), (12, 13), (13, 12))
    if regime == "test" and index == 0:
        choices = tuple(c for c in choices if max(c) >= 10)
    return choices[int(rng.integers(len(choices)))]


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _choose_counts(rng, num_train, num_test)
    # A palette tendency and an overlap curriculum are the only shared sampling choices.
    # Neither changes the uppermost-rectangle rule.
    stable = bool(rng.random() < 0.22)
    overlap_curriculum = bool(rng.random() < 0.42)
    base = tuple(int(c) for c in rng.choice(np.arange(10), size=2, replace=False))
    seen = set()
    puzzle = {"train": [], "test": []}
    area_order = [-1, 1]
    side_order = [-1, 1]
    rng.shuffle(area_order)
    rng.shuffle(side_order)
    first_selected = int(rng.integers(2))
    for index in range(train_count):
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
        canvas = _canvas(pair_rng, "train", index)
        if index < 2:
            relation = area_order[index]
            x_sign = side_order[index]
            overlap = overlap_curriculum and index == 1
            colors = base
            selected_color = base[(first_selected + index) % 2]
            preferred = ("row", "column", "square", "block", "large_block")[int(pair_rng.integers(5))]
        else:
            relation = int(pair_rng.integers(-1, 2))
            x_sign = None
            overlap = overlap_curriculum and bool(pair_rng.random() < 0.38)
            mode = ("partial", "fresh", "base")[int(pair_rng.integers(3))]
            if not stable and index == 2:
                mode = "partial"
            colors = _pair_colors(pair_rng, base, stable, mode, allow_zero=0 in base)
            selected_color = None
            preferred = ("pixel", "row", "column", "square", "block", "large_block")[int(pair_rng.integers(6))]
        background = _background(pair_rng, colors, nonzero=index == 0)
        pair = _make_pair(pair_rng, canvas, colors, background, relation, x_sign, overlap, preferred, selected_color, seen, dominance=index < 2)
        puzzle["train"].append(pair)
    rng.shuffle(puzzle["train"])
    for index in range(test_count):
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
        canvas = _canvas(pair_rng, "test", index)
        relation = int(pair_rng.integers(-1, 2))
        overlap = overlap_curriculum and bool(pair_rng.random() < 0.55)
        mode = ("base", "partial", "fresh")[int(pair_rng.integers(3))]
        colors = _pair_colors(pair_rng, base, stable, mode, allow_zero=0 in base)
        background = _background(pair_rng, colors)
        preferred = ("pixel", "row", "column", "square", "block", "large_block")[int(pair_rng.integers(6))]
        puzzle["test"].append(_make_pair(pair_rng, canvas, colors, background, relation, None, overlap, preferred, None, seen))
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert all(1 <= n <= 30 for n in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((0 <= grid) & (grid <= 9))


def _objects(grid):
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 3
    background = int(colors[int(np.argmax(counts))])
    assert int(np.max(counts)) > grid.size // 2
    objects = []
    for color in colors:
        color = int(color)
        if color == background:
            continue
        ys, xs = np.where(grid == color)
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
        assert len(ys) == (y1 - y0 + 1) * (x1 - x0 + 1)
        assert np.all(grid[y0:y1 + 1, x0:x1 + 1] == color)
        objects.append((y0, y1, x0, x1, color))
    objects.sort()
    assert objects[0][0] < objects[1][0]
    upper, lower = objects
    if upper[1] >= lower[0]:
        assert upper[3] + 1 < lower[2] or lower[3] + 1 < upper[2]
    return background, upper, lower


def _sign(a, b):
    return (a > b) - (a < b)


def _area(obj):
    return (obj[1] - obj[0] + 1) * (obj[3] - obj[2] + 1)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    train_details = []
    test_details = []
    for name, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            _check_grid(inp)
            _check_grid(out)
            key = (inp.shape, inp.tobytes())
            assert key not in seen and _rows(inp) not in _SOURCE_INPUTS
            seen.add(key)
            background, upper, lower = _objects(inp)
            y0, y1, x0, x1, color = upper
            assert np.array_equal(out, inp[y0:y1 + 1, x0:x1 + 1])
            assert np.all(out == color) and inp.shape != out.shape
            details = (frozenset((upper[4], lower[4])), color,
                       _sign(_area(upper), _area(lower)),
                       _sign(upper[2], lower[2]), upper[1] >= lower[0],
                       background, inp.shape,
                       _sign(upper[1] - upper[0], lower[1] - lower[0]),
                       _sign(upper[3] - upper[2], lower[3] - lower[2]))
            (train_details if name == "train" else test_details).append(details)
    # Two training examples reuse colors but reverse selection, area, and side.
    anchors = [(a, b) for a in train_details for b in train_details
               if a is not b and a[0] == b[0] and a[1] != b[1]
               and {a[2], b[2]} == {-1, 1} and {a[3], b[3]} == {-1, 1}
               and {a[7], b[7]} == {-1, 1} and {a[8], b[8]} == {-1, 1}]
    assert anchors
    assert any(d[5] != 0 for d in train_details)
    if any(d[4] for d in test_details):
        assert any(d[4] for d in train_details)
    if any(0 in d[0] for d in test_details):
        assert any(0 in d[0] for d in train_details)
    assert max(test_details[0][6]) >= 10


def _equal_puzzles(first, second):
    return all(len(first[k]) == len(second[k]) and all(
        np.array_equal(a[g], b[g]) for a, b in zip(first[k], second[k]) for g in ("input", "output"))
        for k in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, layouts, out_shapes, canvases = set(), set(), set(), set(), set()
    palettes, backgrounds, relations, shapes, overlap_count = set(), set(), set(), set(), 0
    signatures = set()
    review = {name: 0 for name in (
        "stable", "varied", "partial", "zero_object_train", "zero_object_test",
        "small_large_train", "small_large_test", "large_familiar_test",
        "overlap_train", "overlap_test", "overlap_colored_bg",
        "source_progression", "role_swap", "thin_large_test")}
    first_area, first_side = set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _equal_puzzles(puzzle, generate_puzzle(seed))
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signature = []
        details = {"train": [], "test": []}
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                inp, out = pair["input"], pair["output"]
                background, upper, lower = _objects(inp)
                assert np.array_equal(out, np.full((upper[1] - upper[0] + 1, upper[3] - upper[2] + 1), upper[4]))
                layouts.add((inp.shape, upper[:4], lower[:4]))
                out_shapes.add(out.shape)
                canvases.add(inp.shape)
                palettes.add(tuple(sorted((upper[4], lower[4]))))
                backgrounds.add(background)
                relations.add(_sign(_area(upper), _area(lower)))
                shapes.add(_kind(*out.shape))
                overlap_count += upper[1] >= lower[0]
                details[regime].append((inp.shape, out.shape,
                                        frozenset((upper[4], lower[4])), upper[4],
                                        background, upper[1] >= lower[0],
                                        _sign(_area(upper), _area(lower)),
                                        _sign(upper[2], lower[2])))
                signature.append((inp.shape, inp.tobytes(), out.shape, out.tobytes()))
        signatures.add(tuple(signature))
        train, test = details["train"], details["test"]
        all_pairs = train + test
        trained_colors = set().union(*(item[2] for item in train))
        first_area.add(train[0][6])
        first_side.add(train[0][7])
        review["stable"] += len({item[2] for item in all_pairs}) == 1
        review["varied"] += len({item[2] for item in all_pairs}) > 1
        review["partial"] += any(len(a[2] & b[2]) == 1 for a in all_pairs for b in all_pairs)
        review["role_swap"] += any(a[3] in b[2] and a[3] != b[3] for a in train for b in train)
        review["source_progression"] += any(a[0] == (8, 8) for a in train) and any(b[0] == (10, 10) for b in test)
        for regime, pairs in (("train", train), ("test", test)):
            review[f"zero_object_{regime}"] += sum(0 in item[2] for item in pairs)
            review[f"small_large_{regime}"] += sum(max(item[0]) >= 10 and item[1][0] * item[1][1] <= 4 for item in pairs)
            review[f"overlap_{regime}"] += sum(item[5] for item in pairs)
        review["overlap_colored_bg"] += sum(item[5] and item[4] != 0 for item in all_pairs)
        review["large_familiar_test"] += sum(item[1][0] >= 3 and item[1][1] >= 4 and item[2].issubset(trained_colors) for item in test)
        review["thin_large_test"] += sum(max(item[0]) >= 10 and item[1][0] == 1 and item[1][1] <= 4 for item in test)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(layouts) >= 150 and len(out_shapes) >= 20 and len(canvases) >= 8
    assert len(palettes) >= 12 and len(backgrounds) >= 4
    assert relations == {-1, 0, 1} and len(shapes) >= 5 and overlap_count >= 10
    assert len(signatures) == 42
    assert first_area.issuperset({-1, 1}) and first_side.issuperset({-1, 1})
    assert all(review[name] >= minimum for name, minimum in (
        ("stable", 3), ("varied", 15), ("partial", 15), ("role_swap", 42),
        ("zero_object_train", 5), ("zero_object_test", 3),
        ("small_large_train", 5), ("small_large_test", 5),
        ("large_familiar_test", 3), ("overlap_train", 10),
        ("overlap_test", 5), ("overlap_colored_bg", 3),
        ("source_progression", 3), ("thin_large_test", 3)))
    for seed, counts in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)), 100):
        first, second = generate_puzzle(seed, *counts), generate_puzzle(seed, *counts)
        assert _equal_puzzles(first, second)
        validate_generated_puzzle(first)
        if counts[0] is not None:
            assert len(first["train"]) == counts[0]
        if counts[1] is not None:
            assert len(first["test"]) == counts[1]
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(300, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Illegal counts accepted: {counts}")
