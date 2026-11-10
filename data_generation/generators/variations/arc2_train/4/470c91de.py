import numpy as np


_DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_SOURCE_INPUT_TEXTS = {
    "9977777777\n9877781177\n7777711177\n7777711177\n7777711177\n7777777777\n7555587777\n7555557777\n7555557777\n7555557777",
    "7777777777\n7222287777\n7222227777\n7222227777\n7222227777\n7222227777\n7777777777\n7777786667\n7777766667\n7777777777",
    "77777777777\n77777777777\n77777779997\n22227779997\n22227779997\n22227779987\n22287777333\n77777777333\n77777777833\n77777777777\n77777777777",
    "99977777333\n99977777333\n99977777833\n99974877777\n99874477777\n77774477777\n77774422777\n77777728777\n77777777777\n77777777777\n77777777777",
}


def _grid_key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _is_source_input(grid):
    return "\n".join("".join(str(int(x)) for x in row) for row in grid) in _SOURCE_INPUT_TEXTS


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    nt = _checked_count("num_train", num_train, 2, 10)
    nq = _checked_count("num_test", num_test, 1, 4)
    if nt is not None and nq is not None:
        if nt <= nq:
            raise ValueError("num_train must exceed num_test")
    elif nt is None and nq is None:
        nq = int(rng.integers(1, 5))
        nt = int(rng.integers(nq + 1, 11))
    elif nt is None:
        nt = int(rng.integers(nq + 1, 11))
    else:
        nq = int(rng.integers(1, min(4, nt - 1) + 1))
    return nt, nq


def _overlap(a, b, shifted=False):
    ar, ac, ah, aw, adr, adc = a
    br, bc, bh, bw, bdr, bdc = b
    if shifted:
        ar, ac, br, bc = ar + adr, ac + adc, br + bdr, bc + bdc
    return ar < br + bh and br < ar + ah and ac < bc + bw and bc < ac + aw


def _touch(a, b):
    ar, ac, ah, aw = a[:4]
    br, bc, bh, bw = b[:4]
    return ((ar + ah == br or br + bh == ar) and ac < bc + bw and bc < ac + aw) or ((ac + aw == bc or bc + bw == ac) and ar < br + bh and br < ar + ah)


def _same_color_too_near(a, b):
    ar, ac, ah, aw = a[:4]
    br, bc, bh, bw = b[:4]
    return ar <= br + bh and br <= ar + ah and ac <= bc + bw and bc <= ac + aw


def _boundary(a, shape):
    r, c, h, w = a[:4]
    return r == 0 or c == 0 or r + h == shape[0] or c + w == shape[1]


def _marker_cell(a):
    r, c, h, w, dr, dc = a
    return r if dr < 0 else r + h - 1, c if dc < 0 else c + w - 1


def _candidate_positions(shape, size, direction):
    gh, gw = shape
    h, w = size
    dr, dc = direction
    r0 = 1 if dr < 0 else 0
    r1 = gh - h if dr < 0 else gh - h - 1
    c0 = 1 if dc < 0 else 0
    c1 = gw - w if dc < 0 else gw - w - 1
    return [(r, c, h, w, dr, dc) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _sample_shape(rng, count, regime):
    weights = [0.32, 0.30, 0.19, 0.19] if regime == "source_train" else [0.24, 0.33, 0.22, 0.21]
    mode = int(rng.choice(4, p=weights))
    if mode == 0:
        h, w = int(rng.integers(6, 13)), int(rng.integers(6, 13))
    elif mode == 1:
        h, w = int(rng.integers(11, 21)), int(rng.integers(11, 21))
    elif mode == 2:
        h, w = int(rng.integers(20, 31)), int(rng.integers(20, 31))
    elif rng.random() < 0.5:
        h, w = int(rng.integers(7, 16)), int(rng.integers(18, 31))
    else:
        h, w = int(rng.integers(18, 31)), int(rng.integers(7, 16))
    if h * w < count * 9:
        return None
    return h, w


def _sample_size(rng, shape):
    dims = []
    for limit in shape:
        top = min(12, limit - 1)
        if top < 2:
            return None
        if top >= 7 and rng.random() < 0.20:
            dims.append(int(rng.integers(7, top + 1)))
        else:
            dims.append(int(rng.integers(2, min(6, top) + 1)))
    return tuple(dims)


def _place_layout(rng, shape, sizes, directions, colors, boundary_preference, touch_preference):
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
    chosen = []
    for index in order:
        positions = _candidate_positions(shape, sizes[index], directions[index])
        if not positions:
            return None
        rng.shuffle(positions)
        options = []
        for candidate in positions:
            if any(_overlap(candidate, old) or _overlap(candidate, old, True) or (colors[index] == colors[old_index] and _same_color_too_near(candidate, old)) for old_index, old in chosen):
                continue
            touch = any(_touch(candidate, old) and colors[index] != colors[old_index] for old_index, old in chosen)
            boundary = _boundary(candidate, shape)
            score = (2 if touch and touch_preference else 0) + (1 if not touch and not touch_preference and chosen else 0)
            score += (1 if boundary and boundary_preference > 0 else 0)
            score += (1 if not boundary and boundary_preference < 0 else 0)
            options.append((score, candidate))
        if not options:
            return None
        best = max(score for score, _ in options)
        shortlist = [candidate for score, candidate in options if score >= best - 1]
        chosen.append((index, shortlist[int(rng.integers(len(shortlist)))]))
    result = [None] * len(sizes)
    for index, candidate in chosen:
        result[index] = candidate
    return result


def _paint(shape, background, marker, colors, layout):
    input_grid = np.full(shape, background, dtype=np.int8)
    output_grid = np.full(shape, background, dtype=np.int8)
    for color, a in zip(colors, layout):
        r, c, h, w, dr, dc = a
        input_grid[r:r + h, c:c + w] = color
        mr, mc = _marker_cell(a)
        input_grid[mr, mc] = marker
        output_grid[r + dr:r + dr + h, c + dc:c + dc + w] = color
    return {"input": input_grid, "output": output_grid}


def _generate_pair(rng, regime, marker, background, directions, seen):
    count = len(directions)
    for attempt in range(120):
        shape = _sample_shape(rng, count, regime)
        if shape is None:
            continue
        sizes = [_sample_size(rng, shape) for _ in directions]
        if any(size is None for size in sizes):
            continue
        area = sum(h * w for h, w in sizes)
        if area > 0.72 * shape[0] * shape[1]:
            continue
        pool = [color for color in range(10) if color not in {background, marker}]
        color_mode = int(rng.choice(3, p=[0.17, 0.34, 0.49]))
        if color_mode == 0:
            used = 1
        elif color_mode == 1:
            used = int(rng.integers(1, min(count, 4) + 1))
        else:
            used = min(count, len(pool))
        palette = [int(x) for x in rng.choice(pool, size=used, replace=False)]
        colors = [palette[int(rng.integers(used))] for _ in directions] if color_mode != 2 else [palette[i % used] for i in range(count)]
        boundary_preference = int(rng.choice([-1, 0, 1], p=[0.34, 0.33, 0.33]))
        touch_preference = bool(rng.random() < 0.42)
        layout = _place_layout(rng, shape, sizes, directions, colors, boundary_preference, touch_preference)
        if layout is None:
            continue
        pair = _paint(shape, background, marker, colors, layout)
        key = _grid_key(pair["input"])
        if key in seen or _is_source_input(pair["input"]):
            continue
        try:
            solved, _ = _parse(pair["input"], background, marker)
            if not np.array_equal(solved, pair["output"]):
                continue
            _infer_background(pair["input"], marker, None)
        except AssertionError:
            continue
        seen.add(key)
        return pair
    raise RuntimeError("could not construct a unique marked-rectangle scene")


def _components(grid, color):
    remaining = set(map(tuple, np.argwhere(grid == color)))
    while remaining:
        start = remaining.pop()
        stack = [start]
        cells = [start]
        while stack:
            r, c = stack.pop()
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    cells.append(neighbor)
        yield cells


def _parse(grid, background, marker):
    assert background != marker and np.any(grid == background) and np.any(grid == marker)
    objects = []
    markers = set()
    occupied = np.zeros(grid.shape, dtype=bool)
    for color in (int(x) for x in np.unique(grid) if x not in {background, marker}):
        for cells in _components(grid, color):
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            r, c, re, ce = min(rows), min(cols), max(rows) + 1, max(cols) + 1
            h, w = re - r, ce - c
            assert h >= 2 and w >= 2 and len(cells) == h * w - 1
            region = grid[r:re, c:ce]
            missing = np.argwhere(region != color)
            assert missing.shape == (1, 2)
            mr, mc = r + int(missing[0, 0]), c + int(missing[0, 1])
            assert grid[mr, mc] == marker and mr in {r, re - 1} and mc in {c, ce - 1}
            assert (mr, mc) not in markers and not np.any(occupied[r:re, c:ce])
            markers.add((mr, mc))
            occupied[r:re, c:ce] = True
            dr, dc = (-1 if mr == r else 1), (-1 if mc == c else 1)
            assert 0 <= r + dr and r + dr + h <= grid.shape[0]
            assert 0 <= c + dc and c + dc + w <= grid.shape[1]
            objects.append((r, c, h, w, dr, dc, color))
    assert objects and len(markers) == int(np.count_nonzero(grid == marker))
    rebuilt = np.full(grid.shape, background, dtype=np.int8)
    output = np.full(grid.shape, background, dtype=np.int8)
    final_occupied = np.zeros(grid.shape, dtype=bool)
    for r, c, h, w, dr, dc, color in objects:
        rebuilt[r:r + h, c:c + w] = color
        mr, mc = _marker_cell((r, c, h, w, dr, dc))
        rebuilt[mr, mc] = marker
        dest = np.s_[r + dr:r + dr + h, c + dc:c + dc + w]
        assert not np.any(final_occupied[dest])
        final_occupied[dest] = True
        output[dest] = color
    assert np.array_equal(rebuilt, grid)
    return output, objects


def _infer_background(input_grid, marker, required_output):
    solutions = []
    for background in (int(x) for x in np.unique(input_grid) if x != marker):
        try:
            solved, objects = _parse(input_grid, background, marker)
        except AssertionError:
            continue
        if required_output is None or np.array_equal(solved, required_output):
            solutions.append((background, solved, objects))
    assert len(solutions) == 1
    return solutions[0]


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    marker = int(rng.integers(10))
    fixed_background = bool(rng.random() < 0.35)
    background_pool = [x for x in range(10) if x != marker]
    if fixed_background:
        backgrounds = [int(rng.choice(background_pool))] * (train_count + test_count)
    else:
        backgrounds = [int(rng.choice(background_pool)) for _ in range(train_count + test_count)]
        if len(set(backgrounds[:train_count])) == 1:
            backgrounds[1] = int(rng.choice([x for x in background_pool if x != backgrounds[0]]))
    train_counts = [int(rng.integers(1, 7)) for _ in range(train_count)]
    while sum(train_counts) < 4:
        train_counts[int(rng.integers(train_count))] += 1
    if train_count >= 3 and rng.random() < 0.55 and 1 not in train_counts:
        train_counts[int(rng.integers(train_count))] = 1
        while sum(train_counts) < 4:
            train_counts[int(rng.integers(train_count))] += 1
    mean_train = int(round(float(np.mean(train_counts))))
    source_like_shift = bool(rng.random() < 0.45)
    test_counts = []
    for _ in range(test_count):
        if source_like_shift:
            count = max(train_counts) + int(rng.integers(1, 3))
        else:
            count = mean_train + int(rng.integers(-2, 2))
        test_counts.append(max(1, min(7, count)))
    if len(set(train_counts + test_counts)) == 1:
        test_counts[0] = min(7, test_counts[0] + 1) if test_counts[0] < 7 else 6
    train_dirs = [[int(x) for x in rng.integers(0, 4, size=count)] for count in train_counts]
    slots = [(i, j) for i, dirs in enumerate(train_dirs) for j in range(len(dirs))]
    all_same_index = None
    eligible = [i for i in range(train_count) if train_counts[i] >= 2 and len(slots) - train_counts[i] >= 4]
    if eligible and rng.random() < 0.22:
        all_same_index = int(rng.choice(eligible))
        train_dirs[all_same_index] = [int(rng.integers(4))] * train_counts[all_same_index]
    coverage_slots = [slot for slot in slots if slot[0] != all_same_index]
    selected = rng.choice(len(coverage_slots), size=4, replace=False)
    for direction, position in enumerate(selected):
        i, j = coverage_slots[int(position)]
        train_dirs[i][j] = direction
    test_dirs = []
    for count in test_counts:
        if count >= 2 and rng.random() < 0.16:
            dirs = [int(rng.integers(4))] * count
        else:
            dirs = [int(x) for x in rng.integers(0, 4, size=count)]
        test_dirs.append(dirs)
    puzzle = {"train": [], "test": []}
    seen = set()
    for i, dirs in enumerate(train_dirs):
        puzzle["train"].append(_generate_pair(rng, "source_train", marker, backgrounds[i], [_DIRECTIONS[d] for d in dirs], seen))
    for i, dirs in enumerate(test_dirs):
        puzzle["test"].append(_generate_pair(rng, "target_test", marker, backgrounds[train_count + i], [_DIRECTIONS[d] for d in dirs], seen))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    markers, train_directions = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert 0 <= int(grid.min()) and int(grid.max()) <= 9
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert not _is_source_input(inp)
            inp_key, out_key = _grid_key(inp), _grid_key(out)
            assert inp_key not in seen_inputs and (inp_key, out_key) not in seen_pairs
            seen_inputs.add(inp_key)
            seen_pairs.add((inp_key, out_key))
            if split == "train":
                disappearing = set(map(int, np.unique(inp))) - set(map(int, np.unique(out)))
                assert len(disappearing) == 1
                marker = disappearing.pop()
                markers.add(marker)
                _, solved, objects = _infer_background(inp, marker, out)
                assert np.array_equal(solved, out)
                train_directions.update((obj[4], obj[5]) for obj in objects)
    assert len(markers) == 1 and train_directions == set(_DIRECTIONS)
    marker = next(iter(markers))
    for pair in puzzle["test"]:
        _, solved, objects = _infer_background(pair["input"], marker, None)
        assert np.array_equal(solved, pair["output"])
        assert all((obj[4], obj[5]) in train_directions for obj in objects)


def _puzzles_equal(a, b):
    return all(len(a[split]) == len(b[split]) and all(np.array_equal(p[k], q[k]) for p, q in zip(a[split], b[split]) for k in ("input", "output")) for split in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, shapes, sizes, layouts = set(), set(), set(), set(), set()
    single = many = repeat = dense = long_h = long_w = 0
    varied_background = fixed_background = repeated_direction = all_same_direction = 0
    source_like_shift = comparable_shift = 0
    test_interior = test_touch = test_separate = test_vacated = test_repeat = 0
    test_same_color = test_all_same = test_dense = 0
    five_square = five_by_three = small_canvas = large_canvas = wide_canvas = tall_canvas = 0
    same_color_different_size = same_color_different_direction = 0
    varied_count_puzzles = varied_shape_puzzles = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _puzzles_equal(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        infos = []
        first_input_colors = set(map(int, np.unique(puzzle["train"][0]["input"])))
        first_output_colors = set(map(int, np.unique(puzzle["train"][0]["output"])))
        marker = next(iter(first_input_colors - first_output_colors))
        split_counts = {"train": [], "test": []}
        puzzle_shapes = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                bg, solved, objects = _infer_background(pair["input"], marker, None)
                assert np.array_equal(solved, pair["output"])
                shapes.add(pair["input"].shape)
                puzzle_shapes.add(pair["input"].shape)
                sizes.update((o[2], o[3]) for o in objects)
                layouts.add((pair["input"].shape, tuple(sorted(o[:6] for o in objects))))
                single += len(objects) == 1
                many += len(objects) > 4
                repeat += len({o[6] for o in objects}) < len(objects)
                dense += sum(o[2] * o[3] for o in objects) * 2 > pair["input"].size
                long_h += any(o[2] >= 7 and o[2] > o[3] for o in objects)
                long_w += any(o[3] >= 7 and o[3] > o[2] for o in objects)
                five_square += any(o[2:4] == (5, 5) for o in objects)
                five_by_three += any(o[2:4] == (5, 3) for o in objects)
                h, w = pair["input"].shape
                small_canvas += max(h, w) <= 11
                large_canvas += max(h, w) >= 21
                wide_canvas += w >= 2 * h
                tall_canvas += h >= 2 * w
                directions = [(o[4], o[5]) for o in objects]
                same_color_pairs = [(a, b) for i, a in enumerate(objects) for b in objects[i + 1:] if a[6] == b[6]]
                same_color_different_size += any(a[2:4] != b[2:4] for a, b in same_color_pairs)
                same_color_different_direction += any(a[4:6] != b[4:6] for a, b in same_color_pairs)
                repeated_direction += len(set(directions)) < len(directions)
                all_same_direction += len(objects) >= 2 and len(set(directions)) == 1
                split_counts[split].append(len(objects))
                if split == "test":
                    boundary_count = sum(_boundary(o, pair["input"].shape) for o in objects)
                    touching = any(_touch(a, b) for i, a in enumerate(objects) for b in objects[i + 1:])
                    original = [{(r, c) for r in range(o[0], o[0] + o[2]) for c in range(o[1], o[1] + o[3])} for o in objects]
                    moved = [{(r, c) for r in range(o[0] + o[4], o[0] + o[4] + o[2]) for c in range(o[1] + o[5], o[1] + o[5] + o[3])} for o in objects]
                    test_interior += boundary_count == 0
                    test_touch += touching
                    test_separate += not touching
                    test_vacated += any(moved[i] & original[j] for i in range(len(objects)) for j in range(len(objects)) if i != j)
                    test_repeat += len(set(directions)) < len(directions)
                    test_same_color += len({o[6] for o in objects}) < len(objects)
                    test_all_same += len(objects) >= 2 and len(set(directions)) == 1
                    test_dense += 2 * sum(o[2] * o[3] for o in objects) > pair["input"].size
                infos.append(bg)
        varied_background += len(set(infos[:len(puzzle["train"])])) > 1
        fixed_background += len(set(infos)) == 1
        source_like_shift += max(split_counts["test"]) > max(split_counts["train"])
        comparable_shift += any(n in split_counts["train"] for n in split_counts["test"])
        varied_count_puzzles += len(set(split_counts["train"] + split_counts["test"])) > 1
        varied_shape_puzzles += len(puzzle_shapes) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 35 and len(sizes) >= 25 and len(layouts) >= 150
    assert single > 0 and many > 0 and repeat > 0 and dense > 0
    assert long_h > 0 and long_w > 0 and varied_background > 0 and fixed_background > 0
    assert repeated_direction > 0 and all_same_direction > 0
    assert source_like_shift >= 8 and comparable_shift >= 8
    assert test_interior > 0 and test_touch > 0 and test_separate > 0 and test_vacated > 0
    assert test_repeat > 0 and test_same_color > 0 and test_all_same > 0 and test_dense > 0
    assert five_square > 0 and five_by_three > 0 and small_canvas > 0 and large_canvas > 0
    assert wide_canvas > 0 and tall_canvas > 0
    assert same_color_different_size > 0 and same_color_different_direction > 0
    assert varied_count_puzzles == 42 and varied_shape_puzzles == 42
    assert not _puzzles_equal(generate_puzzle(4242, 6, 3), generate_puzzle(4243, 6, 3))
    for seed, nt, nq in ((7001, 2, 1), (7002, 10, 1), (7003, 10, 4)):
        p = generate_puzzle(seed, nt, nq)
        assert len(p["train"]) == nt and len(p["test"]) == nq
    for args in ({"num_train": None, "num_test": 1}, {"num_train": None, "num_test": 4}, {"num_train": 2, "num_test": None}, {"num_train": 10, "num_test": None}):
        validate_generated_puzzle(generate_puzzle(8000, **args))
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}, {"num_train": 2.5}, {"num_test": True}):
        try:
            generate_puzzle(9000, **args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
