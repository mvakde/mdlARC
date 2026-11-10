import numpy as np


_SOURCE_PAIR_KEYS = {
    (
        (
            "000000000",
            "000000000",
            "000000000",
            "000000000",
            "000000000",
            "044444400",
            "040000400",
            "040220400",
            "040220400",
        ),
        (
            "000000000",
            "000000000",
            "000000000",
            "000000002",
            "200000020",
            "044444400",
            "040000400",
            "040220400",
            "040220400",
        ),
    ),
    (
        (
            "00806080",
            "00800080",
            "00888880",
            "00000000",
            "00000000",
            "00000000",
            "00000000",
            "00000000",
        ),
        (
            "00806080",
            "00800080",
            "00888880",
            "06000006",
            "60000000",
            "00000000",
            "00000000",
            "00000000",
        ),
    ),
    (
        (
            "000000000000",
            "000000000000",
            "000000000000",
            "000000000000",
            "000000000000",
            "555555000000",
            "000005000000",
            "444405000000",
            "444405000000",
            "444405000000",
            "444405000000",
            "444405000000",
        ),
        (
            "000000000040",
            "000000000400",
            "000000004000",
            "000000040000",
            "000000400000",
            "555555000000",
            "000005000000",
            "444405000000",
            "444405000000",
            "444405000000",
            "444405000000",
            "444405000000",
        ),
    ),
    (
        (
            "330900",
            "330900",
            "000900",
            "999900",
            "000000",
            "000000",
        ),
        (
            "330900",
            "330900",
            "000900",
            "999900",
            "000030",
            "000003",
        ),
    ),
    (
        (
            "000030440300",
            "000030440300",
            "000030440300",
            "000030000300",
            "000033333300",
            "000000000000",
            "000000000000",
            "000000000000",
            "000000000000",
            "000000000000",
            "000000000000",
            "000000000000",
        ),
        (
            "000030440300",
            "000030440300",
            "000030440300",
            "000030000300",
            "000033333300",
            "000400000040",
            "004000000004",
            "040000000000",
            "400000000000",
            "000000000000",
            "000000000000",
            "000000000000",
        ),
    ),
}



def _checked_optional_count(value, name, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    num_train = _checked_optional_count(num_train, "num_train", 2, 10)
    num_test = _checked_optional_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        low = max(2, num_test + 1)
        num_train = low + (seed // 4) % (11 - low)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _rectangle(mask):
    points = np.argwhere(mask)
    if len(points) == 0:
        return None
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    if int((r1 - r0 + 1) * (c1 - c0 + 1)) != len(points):
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _connected(mask):
    points = np.argwhere(mask)
    if len(points) == 0:
        return False
    start = tuple(map(int, points[0]))
    stack, seen = [start], {start}
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(points)


def _runs(mask):
    for axis in ("horizontal", "vertical"):
        view = mask if axis == "horizontal" else mask.T
        for row in range(view.shape[0]):
            col = 0
            while col < view.shape[1]:
                if not view[row, col]:
                    col += 1
                    continue
                start = col
                while col + 1 < view.shape[1] and view[row, col + 1]:
                    col += 1
                if col - start + 1 >= 4:
                    yield axis, row, start, col
                col += 1


def _emit_horizontal(grid, row, left, right, inward, payload_color, background):
    output = grid.copy()
    lengths = []
    for end, dc in ((left, -1), (right, 1)):
        r, c, length = row - inward, end + dc, 0
        while 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
            if int(grid[r, c]) != background:
                return None
            output[r, c] = payload_color
            length += 1
            r -= inward
            c += dc
        lengths.append(length)
    if not sum(lengths):
        return None
    return output, tuple(lengths)


def _candidates(input_grid):
    values = [int(x) for x in np.unique(input_grid)]
    if len(values) != 3:
        return []
    candidates = []
    for background in values:
        foreground = [v for v in values if v != background]
        rectangles = [_rectangle(input_grid == v) for v in foreground]
        if sum(box is not None for box in rectangles) != 1:
            continue
        payload_index = 0 if rectangles[0] is not None else 1
        payload = foreground[payload_index]
        frame = foreground[1 - payload_index]
        if not _connected(input_grid == frame):
            continue
        for axis, row, left, right in _runs(input_grid == frame):
            view = input_grid if axis == "horizontal" else input_grid.T
            frame_mask = view == frame
            payload_box = _rectangle(view == payload)
            p0, q0, p1, q1 = payload_box
            if p1 <= row - 2:
                inward = -1
            elif p0 >= row + 2:
                inward = 1
            else:
                continue
            back = np.zeros(view.shape, dtype=bool)
            back[row, left:right + 1] = True
            auxiliary = np.argwhere(frame_mask & ~back)
            arms = []
            valid = True
            for end in (left, right):
                depths = sorted(int(inward * (r - row)) for r, c in auxiliary if c == end)
                if depths != list(range(1, len(depths) + 1)):
                    valid = False
                arms.append(len(depths))
            if not valid or len(auxiliary) != sum(arms):
                continue
            depth_end = max(abs(p0 - row), abs(p1 - row))
            if any(length and length < depth_end for length in arms):
                continue
            if arms[0] and arms[1]:
                if not (q0 >= left + 2 and q1 <= right - 2):
                    continue
                mode = "complete"
            elif arms[0] == 0 and arms[1]:
                if not (left == 0 and q0 == 0 and q1 <= right - 2):
                    continue
                mode = "L"
            elif arms[1] == 0 and arms[0]:
                if not (right == view.shape[1] - 1 and q1 == right and q0 >= left + 2):
                    continue
                mode = "L"
            else:
                continue
            if any(inward * (int(r) - row) <= 0 for r, c in auxiliary):
                continue
            emission = _emit_horizontal(view, row, left, right, inward, payload, background)
            if emission is None:
                continue
            expected_view, ray_lengths = emission
            expected = expected_view if axis == "horizontal" else expected_view.T.copy()
            candidates.append({
                "background": background, "frame": frame, "payload": payload,
                "axis": axis, "inward": inward, "mode": mode,
                "back_length": right - left + 1,
                "arm_lengths": tuple(arms), "ray_lengths": ray_lengths,
                "payload_height": p1 - p0 + 1 if axis == "horizontal" else q1 - q0 + 1,
                "payload_width": q1 - q0 + 1 if axis == "horizontal" else p1 - p0 + 1,
                "payload_bbox": (p0, q0, p1, q1),
                "shape": tuple(input_grid.shape), "expected": expected,
            })
    return candidates


def _analyze_pair(input_grid, output_grid):
    matches = [c for c in _candidates(input_grid) if np.array_equal(c["expected"], output_grid)]
    assert matches, "pair has no valid frame interpretation"
    assert len({_array_signature(c["expected"]) for c in matches}) == 1
    return matches[0]


def _sample_margins(rng, scale, mode, pattern):
    limit = (3, 6, 9)[scale]
    if mode == "L":
        return 0, int(rng.integers(1, limit + 1))
    if pattern == "full":
        return int(rng.integers(1, limit + 1)), int(rng.integers(1, limit + 1))
    if pattern == "equal":
        margin = int(rng.integers(1, limit + 1))
        return margin, margin
    if pattern == "unequal":
        left = int(rng.integers(1, limit + 1))
        right = 1 + (left + int(rng.integers(0, limit - 1))) % limit
        return left, right
    if pattern == "clipped":
        margins = [0, int(rng.integers(1, limit + 1))]
        if int(rng.integers(2)):
            margins.reverse()
        return tuple(margins)
    left = int(rng.integers(0, limit + 1))
    right = int(rng.integers(0, limit + 1))
    if left == right == 0:
        right = 1
    return left, right


def _construct_horizontal(rng, regime, mode, inward, pattern, background, frame, payload):
    for _ in range(120):
        probabilities = (0.30, 0.43, 0.27) if regime == "source_train" else (0.22, 0.40, 0.38)
        scale = int(rng.choice(3, p=probabilities))
        if scale == 0:
            ph, pw = int(rng.integers(1, 3)), int(rng.integers(1, 4))
            start = 2
            arm_extra, bar_extra = 1, 2
        elif scale == 1:
            ph, pw = int(rng.integers(1, 6)), int(rng.integers(1, 6))
            start = int(rng.integers(2, 4))
            arm_extra, bar_extra = 3, 4
        else:
            ph, pw = int(rng.integers(1, 7)), int(rng.integers(1, 8))
            start = int(rng.integers(2, 5))
            arm_extra, bar_extra = 5, 5
        if scale and rng.random() < 0.22:
            ph = int(rng.integers(4, 6))
            pw = ph
        last_depth = start + ph - 1
        bar_length = pw + (4 if mode == "complete" else 2) + int(rng.integers(0, bar_extra + 1))
        right_arm = last_depth + int(rng.integers(0, arm_extra + 1))
        left_arm = 0 if mode == "L" else last_depth + int(rng.integers(0, arm_extra + 1))
        left_margin, right_margin = _sample_margins(rng, scale, mode, pattern)
        if mode == "L" and rng.random() < 0.5:
            # Mirroring is applied after construction; the missing arm stays at an edge.
            mirror = True
        else:
            mirror = False
        normal_limit = (3, 7, 11)[scale]
        normal_min = max(left_margin, right_margin) if pattern == "unequal" else 1
        normal_free = int(rng.integers(normal_min, max(normal_min + 1, normal_limit + 1)))
        inside_extra = int(rng.integers(0, (2, 4, 6)[scale]))
        arm_extent = max(left_arm, right_arm)
        height = arm_extent + inside_extra + 1 + normal_free
        width = left_margin + bar_length + right_margin
        if height < 6:
            normal_free += 6 - height
            height = 6
        if width < 6:
            right_margin += 6 - width
            width = 6
        if height > 30 or width > 30:
            continue
        back_row = arm_extent + inside_extra if inward == -1 else normal_free
        left, right = left_margin, left_margin + bar_length - 1
        grid = np.full((height, width), background, dtype=np.int8)
        grid[back_row, left:right + 1] = frame
        for end, length in ((left, left_arm), (right, right_arm)):
            for depth in range(1, length + 1):
                grid[back_row + inward * depth, end] = frame
        if mode == "L":
            payload_left = 0
        else:
            payload_left = int(rng.integers(left + 2, right - pw))
        row0 = back_row + inward * start
        for depth in range(start, start + ph):
            grid[back_row + inward * depth, payload_left:payload_left + pw] = payload
        counts = [int(np.count_nonzero(grid == color)) for color in (background, frame, payload)]
        if counts[0] <= max(counts[1:]):
            continue
        emission = _emit_horizontal(grid, back_row, left, right, inward, payload, background)
        if emission is None:
            continue
        output = emission[0]
        if mirror:
            grid = grid[:, ::-1].copy()
            output = output[:, ::-1].copy()
        return grid, output
    raise RuntimeError("could not construct a valid frame")


def _make_pair(rng, regime, spec, colors):
    mode, axis, inward, pattern = spec
    background, frame, payload = colors
    for _ in range(30):
        inp, out = _construct_horizontal(rng, regime, mode, inward, pattern, background, frame, payload)
        if axis == "vertical":
            inp, out = inp.T.copy(), out.T.copy()
        try:
            chosen = _analyze_pair(inp, out)
        except AssertionError:
            continue
        if chosen["axis"] != axis or chosen["mode"] != mode:
            continue
        return inp, out
    raise RuntimeError("could not generate an unambiguous pair")


def _array_signature(array):
    normalized = np.ascontiguousarray(array, dtype=np.int64)
    return tuple(array.shape), normalized.tobytes()


def _rows_key(grid):
    return tuple("".join(str(int(value)) for value in row) for row in grid)


def _source_pair(input_grid, output_grid):
    return (_rows_key(input_grid), _rows_key(output_grid)) in _SOURCE_PAIR_KEYS


def _training_specs(rng, num_train, l_axis):
    first_sign = -1 if int(rng.integers(2)) == 0 else 1
    specs = [
        ("complete", l_axis, first_sign, "full"),
        ("L", l_axis, -first_sign, "edge"),
    ]
    if num_train >= 3:
        other = "vertical" if l_axis == "horizontal" else "horizontal"
        specs.append(("complete", other, int(rng.choice((-1, 1))), "free"))
    while len(specs) < num_train:
        mode = "L" if rng.random() < 0.24 else "complete"
        axis = l_axis if mode == "L" else str(rng.choice(("horizontal", "vertical")))
        pattern = "edge" if mode == "L" else str(rng.choice(("equal", "unequal", "clipped", "free")))
        specs.append((mode, axis, int(rng.choice((-1, 1))), pattern))
    rng.shuffle(specs)
    return specs


def _test_specs(rng, num_test, train_specs, l_axis):
    complete_axes = tuple(sorted({spec[1] for spec in train_specs if spec[0] == "complete"}))
    specs = []
    for _ in range(num_test):
        mode = "L" if rng.random() < 0.28 else "complete"
        axis = l_axis if mode == "L" else str(rng.choice(complete_axes))
        pattern = "edge" if mode == "L" else str(rng.choice(("equal", "unequal", "clipped", "free"), p=(0.24, 0.40, 0.24, 0.12)))
        specs.append((mode, axis, int(rng.choice((-1, 1))), pattern))
    return specs


def _puzzle_colors(rng, num_train, num_test):
    background = int(rng.integers(10))
    available = [color for color in range(10) if color != background]
    rng.shuffle(available)
    count = num_train + num_test
    palette_size = int(rng.integers(2, min(7, 2 * count) + 1))
    palette = available[:palette_size]
    pairs = []
    for index in range(count):
        frame = palette[(2 * index) % palette_size]
        payload = palette[(2 * index + 1) % palette_size]
        if rng.random() < 0.5:
            frame, payload = payload, frame
        pairs.append((background, frame, payload))
    if rng.random() < 0.38:
        train_colors = {color for _, frame, payload in pairs[:num_train] for color in (frame, payload)}
        index = num_train + int(rng.integers(num_test))
        old_background, old_frame, old_payload = pairs[index]
        fresh = [color for color in available
                 if color not in train_colors and color not in (old_frame, old_payload)]
        if fresh:
            new_color = int(rng.choice(fresh))
            if int(rng.integers(2)):
                pairs[index] = (old_background, new_color, old_payload)
            else:
                pairs[index] = (old_background, old_frame, new_color)
    return pairs


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, rng, num_train, num_test)
    l_axis = str(rng.choice(("horizontal", "vertical")))
    train_specs = _training_specs(rng, num_train, l_axis)
    test_specs = _test_specs(rng, num_test, train_specs, l_axis)
    colors = _puzzle_colors(rng, num_train, num_test)
    puzzle = {"train": [], "test": []}
    seen = set()
    for index, (split, spec) in enumerate(
        [("train", spec) for spec in train_specs] + [("test", spec) for spec in test_specs]
    ):
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
        for _ in range(60):
            inp, out = _make_pair(pair_rng, "source_train" if split == "train" else "target_test", spec, colors[index])
            key = _array_signature(inp)
            if key not in seen and not _source_pair(inp, out):
                break
        else:
            raise RuntimeError("could not generate a distinct pair")
        seen.add(key)
        puzzle[split].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def _predict_from_training(train, test_input):
    learned = [_analyze_pair(pair["input"], pair["output"]) for pair in train]
    l_axes = {item["axis"] for item in learned if item["mode"] == "L"}
    complete_axes = {item["axis"] for item in learned if item["mode"] == "complete"}
    assert len(l_axes) == 1
    candidates = [c for c in _candidates(test_input) if
                  (c["mode"] == "L" and c["axis"] in l_axes) or
                  (c["mode"] == "complete" and c["axis"] in complete_axes)]
    answers = {_array_signature(c["expected"]): c["expected"] for c in candidates}
    assert len(answers) == 1, "training does not identify a unique test output"
    return next(iter(answers.values()))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    features = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= side <= 30 for side in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            inp_key = _array_signature(inp)
            pair_key = (inp_key, _array_signature(out))
            assert inp_key not in seen_inputs and pair_key not in seen_pairs
            assert not _source_pair(inp, out)
            seen_inputs.add(inp_key)
            seen_pairs.add(pair_key)
            item = _analyze_pair(inp, out)
            assert set(np.unique(inp)) == set(np.unique(out))
            assert int(np.count_nonzero(inp != out)) == sum(item["ray_lengths"])
            features[split].append(item)
    all_features = features["train"] + features["test"]
    assert len({item["background"] for item in all_features}) == 1
    assert all(item["frame"] != item["payload"] for item in all_features)
    train = features["train"]
    assert {item["mode"] for item in train} == {"L", "complete"}
    assert {item["inward"] for item in train} == {-1, 1}
    assert any(item["mode"] == "complete" and all(item["ray_lengths"]) for item in train)
    assert len({item["axis"] for item in train if item["mode"] == "L"}) == 1
    for pair in puzzle["test"]:
        predicted = _predict_from_training(puzzle["train"], pair["input"].copy())
        assert np.array_equal(predicted, pair["output"])


def _puzzles_equal(first, second):
    return all(len(first[split]) == len(second[split]) and all(
        np.array_equal(a["input"], b["input"]) and np.array_equal(a["output"], b["output"])
        for a, b in zip(first[split], second[split])
    ) for split in ("train", "test"))


def self_test() -> None:
    assert _puzzles_equal(generate_puzzle(12345, 7, 3), generate_puzzle(12345, 7, 3))
    assert not _puzzles_equal(generate_puzzle(12345, 7, 3), generate_puzzle(12346, 7, 3))
    for index, counts in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(20000 + index, *counts)
        if counts[0] is not None:
            assert len(puzzle["train"]) == counts[0]
        if counts[1] is not None:
            assert len(puzzle["test"]) == counts[1]
        validate_generated_puzzle(puzzle)
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(22000, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for input_rows, output_rows in _SOURCE_PAIR_KEYS:
        inp = np.array([[int(v) for v in row] for row in input_rows], dtype=np.int8)
        out = np.array([[int(v) for v in row] for row in output_rows], dtype=np.int8)
        _analyze_pair(inp, out)
    train_counts, test_counts = set(), set()
    shapes, layouts, palettes = set(), set(), set()
    train_large_payload = test_small_payload = short_bar = long_arm = 0
    l_train = l_test = upward_test = clipped_test = equal_test = unequal_test = 0
    fresh_test_puzzles = role_swap_puzzles = 0
    within_varied = 0
    observed = {"train": set(), "test": set()}
    within_features = {name: 0 for name in
                       ("shape", "axis_inward", "back", "arms", "rays", "payload", "roles")}
    palette_sizes = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        items = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                item = _analyze_pair(pair["input"], pair["output"])
                items.append(item)
                shapes.add(item["shape"])
                layouts.add((item["shape"], item["axis"], item["inward"], item["mode"],
                             item["back_length"], item["arm_lengths"], item["ray_lengths"],
                             item["payload_height"], item["payload_width"], item["payload_bbox"]))
                flags = observed[split]
                flags.add(("mode", item["mode"]))
                flags.add(("axis_inward", item["axis"], item["inward"]))
                if max(item["shape"]) <= 10:
                    flags.add("small_canvas")
                if min(item["shape"]) >= 17:
                    flags.add("large_canvas")
                if item["back_length"] <= 5:
                    flags.add("short_bar")
                if max(item["arm_lengths"]) >= item["back_length"]:
                    flags.add("long_arm")
                area = item["payload_height"] * item["payload_width"]
                if area <= 8:
                    flags.add("small_payload")
                if area >= 20:
                    flags.add("large_payload")
                if item["payload_width"] > item["payload_height"]:
                    flags.add("wide_payload")
                if item["payload_height"] == item["payload_width"] == 5:
                    flags.add("square_5_payload")
                rays = item["ray_lengths"]
                if rays[0] == 0:
                    flags.add("zero_left_ray")
                if rays[1] == 0:
                    flags.add("zero_right_ray")
                if rays[0] == rays[1]:
                    flags.add("equal_rays")
                if min(rays) > 0 and rays[0] != rays[1]:
                    flags.add("unequal_two_rays")
                if area <= 8 and max(rays) >= 5:
                    flags.add("small_payload_long_ray")
                if area >= 20 and max(rays) <= 3:
                    flags.add("large_payload_short_ray")
                payload_points = np.argwhere(pair["input"] == item["payload"])
                low, high = payload_points.min(axis=0), payload_points.max(axis=0)
                if any(low == 0) or any(high == np.array(item["shape"]) - 1):
                    flags.add("boundary_payload")
                short_bar += item["back_length"] <= 5
                long_arm += max(item["arm_lengths"]) >= item["back_length"]
                if split == "train":
                    train_large_payload += item["payload_height"] >= 5 and item["payload_width"] >= 5
                    l_train += item["mode"] == "L"
                else:
                    test_small_payload += item["payload_height"] * item["payload_width"] <= 8
                    l_test += item["mode"] == "L"
                    upward_test += item["axis"] == "horizontal" and item["inward"] == 1
                    clipped_test += 0 in item["ray_lengths"]
                    equal_test += item["ray_lengths"][0] == item["ray_lengths"][1]
                    unequal_test += all(item["ray_lengths"]) and item["ray_lengths"][0] != item["ray_lengths"][1]
        foreground = {c for item in items for c in (item["frame"], item["payload"])}
        palettes.add(tuple(sorted(foreground)))
        palette_sizes.add(len(foreground))
        training_items = [_analyze_pair(pair["input"], pair["output"]) for pair in puzzle["train"]]
        testing_items = [_analyze_pair(pair["input"], pair["output"]) for pair in puzzle["test"]]
        train_colors = {color for item in training_items for color in (item["frame"], item["payload"])}
        fresh_test_puzzles += any(color not in train_colors for item in testing_items for color in (item["frame"], item["payload"]))
        role_swap_puzzles += any(a["frame"] == b["payload"] and a["payload"] == b["frame"]
                                 for a in training_items for b in testing_items)
        within_varied += len({(item["shape"], item["axis"], item["mode"], item["back_length"]) for item in items}) >= 2
        measurements = {
            "shape": [item["shape"] for item in items],
            "axis_inward": [(item["axis"], item["inward"]) for item in items],
            "back": [item["back_length"] for item in items],
            "arms": [item["arm_lengths"] for item in items],
            "rays": [item["ray_lengths"] for item in items],
            "payload": [(item["payload_height"], item["payload_width"]) for item in items],
            "roles": [(item["frame"], item["payload"]) for item in items],
        }
        for name, values in measurements.items():
            within_features[name] += len(set(values)) >= 2
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 25 and len(layouts) >= 120 and len(palettes) >= 20
    assert min(train_large_payload, test_small_payload, short_bar, long_arm, l_train,
               l_test, upward_test, clipped_test, equal_test, unequal_test) > 0
    assert fresh_test_puzzles > 0 and role_swap_puzzles > 0
    assert within_varied >= 40
    assert all(count >= 40 for count in within_features.values())
    assert len(palette_sizes) >= 3
    for split in ("train", "test"):
        assert {"small_canvas", "large_canvas", "short_bar", "long_arm",
                "small_payload", "large_payload", "wide_payload", "square_5_payload",
                "zero_left_ray", "zero_right_ray", "equal_rays", "unequal_two_rays",
                "small_payload_long_ray", "large_payload_short_ray", "boundary_payload"} <= observed[split]
        assert {("mode", "L"), ("mode", "complete")} <= observed[split]
        assert {("axis_inward", axis, inward) for axis in ("horizontal", "vertical")
                for inward in (-1, 1)} <= observed[split]
