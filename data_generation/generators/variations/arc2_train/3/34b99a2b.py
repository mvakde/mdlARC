"""Generate aligned two-panel ARC puzzles with a cellwise XOR answer."""

import numpy as np


_AUTO_COUNTS = tuple(
    (train, test)
    for train in range(2, 11)
    for test in range(1, min(4, train - 1) + 1)
)

_SOURCE_ROWS = (
    (("080040550", "880845005", "880045005", "080840050", "008040505"),
     ("0020", "0200", "0202", "0222", "0222")),
    (("080045050", "080845055", "080840005", "080840505", "000840050"),
     ("2220", "2220", "0200", "0000", "0022")),
    (("000840555", "088840500", "800040505", "808845550", "088045005"),
     ("0220", "0022", "2202", "0202", "2222")),
    (("880040550", "880840055", "000040050", "880040555", "800840005"),
     ("2020", "2220", "0020", "2022", "2000")),
    (("080045000", "080845505", "880840055", "888845055", "008845050"),
     ("2200", "2000", "2220", "0200", "2002")),
)

_SOURCE_PAIRS = tuple(
    (np.array([[int(c) for c in row] for row in x], dtype=np.int8),
     np.array([[int(c) for c in row] for row in y], dtype=np.int8))
    for x, y in _SOURCE_ROWS
)


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} must be at least {low}" if high is None else
                         f"{name} must be between {low} and {high}")
    return value


def _counts(seed, rng, num_train, num_test):
    if num_train is not None:
        num_train = _integer(num_train, "num_train", 2, 10)
    if num_test is not None:
        num_test = _integer(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_train, num_test = _AUTO_COUNTS[seed % len(_AUTO_COUNTS)]
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _plan(seed, rng):
    colors = tuple(int(x) for x in rng.permutation(10)[:5])
    simple = seed % 7 == 0
    return {
        "result": colors[4],
        "base_colors": colors[:4],
        "palette_mode": "stable" if simple else ("swap" if seed % 5 == 0 else "varied"),
        "layout_mode": "stable" if simple or seed % 6 == 0 else "mixed",
        "size_mode": "stable" if simple else "varied",
        "base_orientation": "vertical" if int(rng.integers(2)) == 0 else "horizontal",
        "base_size": None,
        "density_theme": ("balanced", "train_dense", "test_dense", "contrast")[seed % 4],
    }


def _orientation(plan, rng, index, training):
    if plan["layout_mode"] == "stable":
        return plan["base_orientation"]
    if training and index < 2:
        return plan["base_orientation"] if index == 0 else (
            "horizontal" if plan["base_orientation"] == "vertical" else "vertical"
        )
    return "vertical" if int(rng.integers(2)) == 0 else "horizontal"


def _size(rng, orientation, index, plan):
    if plan["size_mode"] == "stable" and plan["base_size"] is not None:
        return plan["base_size"]
    limit_h, limit_w = ((30, 14) if orientation == "vertical" else (14, 30))
    bucket = int(rng.choice(5, p=[0.15, 0.14, 0.39, 0.20, 0.12]))
    if bucket == 0:
        height, width = 5, 4
    elif bucket == 1:
        height = int(rng.integers(2, min(5, limit_h + 1)))
        width = int(rng.integers(2, min(5, limit_w + 1)))
    elif bucket == 2:
        height = int(rng.integers(4, min(11, limit_h + 1)))
        width = int(rng.integers(4, min(11, limit_w + 1)))
    elif bucket == 3:
        if int(rng.integers(2)):
            height = int(rng.integers(2, min(4, limit_h + 1)))
            width = int(rng.integers(8, limit_w + 1))
        else:
            height = int(rng.integers(8, limit_h + 1))
            width = int(rng.integers(2, min(4, limit_w + 1)))
    else:
        if limit_h == 30:
            height = int(rng.integers(20, 31))
            width = int(rng.integers(6, 15))
        else:
            height = int(rng.integers(6, 15))
            width = int(rng.integers(20, 31))
    if plan["size_mode"] == "stable":
        plan["base_size"] = (height, width)
    return height, width


def _palette(plan, rng, index):
    base = plan["base_colors"]
    mode = plan["palette_mode"]
    if mode == "stable" or index == 0:
        return base
    if mode == "swap":
        return (base[0], base[2], base[1], base[3]) if index % 2 else base
    available = [c for c in range(10) if c != plan["result"]]
    colors = tuple(int(c) for c in rng.choice(available, size=4, replace=False))
    if colors == base:
        colors = (colors[0], colors[2], colors[1], colors[3])
    return colors


def _target_density(rng, theme, training, mask_index):
    if theme == "train_dense":
        band = "dense" if training else "sparse"
    elif theme == "test_dense":
        band = "sparse" if training else "dense"
    elif theme == "contrast":
        band = "sparse" if mask_index == 0 else "dense"
    else:
        band = ("sparse", "middle", "dense")[int(rng.integers(3))]
    bounds = {"sparse": (0.12, 0.38), "middle": (0.35, 0.65), "dense": (0.62, 0.88)}[band]
    return float(rng.uniform(*bounds))


def _shape_score(rng, height, width, style):
    rows, cols = np.indices((height, width), dtype=float)
    row_scale = max(1.0, height - 1)
    col_scale = max(1.0, width - 1)
    y, x = rows / row_scale, cols / col_scale
    jitter = rng.random((height, width))
    if style == "scatter":
        return jitter
    if style == "band":
        angle = float(rng.uniform(0, 2 * np.pi))
        return np.cos(angle) * y + np.sin(angle) * x + 0.12 * jitter
    if style == "radial":
        cy, cx = rng.uniform(-0.2, 1.2, size=2)
        return (y - cy) ** 2 + (x - cx) ** 2 + 0.08 * jitter
    if style in ("islands", "holes"):
        count = int(rng.integers(2, 7))
        field = np.zeros((height, width), dtype=float)
        for _ in range(count):
            cy, cx = rng.uniform(0, 1, size=2)
            sy, sx = rng.uniform(0.09, 0.34, size=2)
            bump = np.exp(-0.5 * (((y - cy) / sy) ** 2 + ((x - cx) / sx) ** 2))
            field += float(rng.uniform(0.5, 1.4)) * bump
        if style == "holes":
            cy, cx = rng.uniform(0.25, 0.75, size=2)
            field = -((y - cy) ** 2 + (x - cx) ** 2) - 1.15 * field
        return -field + 0.06 * jitter
    if style == "strokes":
        distances = np.full((height, width), 10.0)
        for _ in range(int(rng.integers(1, 6))):
            ay, ax, by, bx = rng.uniform(-0.1, 1.1, size=4)
            dy, dx = by - ay, bx - ax
            t = np.clip(((y - ay) * dy + (x - ax) * dx) / (dy * dy + dx * dx + 1e-9), 0, 1)
            distance = (y - ay - t * dy) ** 2 + (x - ax - t * dx) ** 2
            distances = np.minimum(distances, distance)
        return distances + 0.03 * jitter
    if style == "repeat":
        period_y = int(rng.integers(2, 6))
        period_x = int(rng.integers(2, 6))
        phase_y, phase_x = rng.uniform(0, 2 * np.pi, size=2)
        return np.sin(rows * 2 * np.pi / period_y + phase_y) + np.sin(cols * 2 * np.pi / period_x + phase_x) + 0.16 * jitter
    # Union of separately positioned rectangles and disks.
    score = np.full((height, width), 10.0)
    for _ in range(int(rng.integers(2, 7))):
        cy, cx = rng.uniform(0, 1, size=2)
        sy, sx = rng.uniform(0.07, 0.38, size=2)
        if int(rng.integers(2)):
            part = np.maximum(np.abs(y - cy) / sy, np.abs(x - cx) / sx)
        else:
            part = ((y - cy) / sy) ** 2 + ((x - cx) / sx) ** 2
        score = np.minimum(score, part)
    return score + 0.05 * jitter


_STYLES = ("scatter", "band", "radial", "islands", "holes", "strokes", "repeat", "shapes")


def _mask(rng, height, width, density, style=None):
    if style is None:
        style = _STYLES[int(rng.integers(len(_STYLES)))]
    score = _shape_score(rng, height, width, style)
    total = height * width
    count = int(np.clip(round(density * total), 1, total - 1))
    mask = np.zeros(total, dtype=bool)
    mask[np.argsort(score.ravel(), kind="stable")[:count]] = True
    return mask.reshape(height, width)


def _proper_mask(mask, rng):
    flat = mask.ravel().copy()
    if not np.any(flat):
        flat[int(rng.integers(flat.size))] = True
    if np.all(flat):
        flat[int(rng.integers(flat.size))] = False
    return flat.reshape(mask.shape)


def _relation_kind(rng, training, index):
    if training and index == 0:
        return "ordinary"
    choices = ("ordinary", "ordinary", "ordinary", "identical", "complementary", "disjoint", "contained")
    if not training:
        choices = ("ordinary", "ordinary", "identical", "complementary", "disjoint", "contained")
    return choices[int(rng.integers(len(choices)))]


def _masks(rng, height, width, plan, training, index):
    kind = _relation_kind(rng, training, index)
    first_density = _target_density(rng, plan["density_theme"], training, 0)
    second_density = _target_density(rng, plan["density_theme"], training, 1)
    first = _mask(rng, height, width, first_density)
    if kind == "identical":
        second = first.copy()
    elif kind == "complementary":
        second = ~first
    elif kind == "disjoint":
        second = _mask(rng, height, width, second_density) & ~first
        if not np.any(second):
            positions = np.flatnonzero((~first).ravel())
            second = np.zeros_like(first)
            second.ravel()[positions[:max(1, len(positions) // 2)]] = True
    elif kind == "contained":
        # The second mask is a proper, nonempty subset of the first.
        positions = np.flatnonzero(first.ravel())
        if len(positions) < 2:
            first.ravel()[int(np.flatnonzero((~first).ravel())[0])] = True
            positions = np.flatnonzero(first.ravel())
        score = _shape_score(rng, height, width, _STYLES[int(rng.integers(len(_STYLES)))])
        order = positions[np.argsort(score.ravel()[positions])]
        take = int(rng.integers(1, len(positions)))
        second = np.zeros_like(first)
        second.ravel()[order[:take]] = True
    else:
        second = _mask(rng, height, width, second_density)
        if training and index == 0:
            # One ordinary demonstration directly exposes the four cases.
            picks = rng.choice(height * width, size=4, replace=False)
            for position, (a, b) in zip(picks, ((0, 0), (1, 0), (0, 1), (1, 1))):
                first.ravel()[position] = bool(a)
                second.ravel()[position] = bool(b)
    first = _proper_mask(first, rng)
    second = _proper_mask(second, rng)
    return first, second


def _make_pair(rng, plan, index, training):
    orientation = _orientation(plan, rng, index, training)
    height, width = _size(rng, orientation, index, plan)
    # The first training example needs room for all four cases.
    if training and index == 0 and height * width < 12:
        height, width = (5, 4)
        if plan["size_mode"] == "stable":
            plan["base_size"] = (height, width)
    background, first_color, second_color, divider_color = _palette(plan, rng, index)
    first, second = _masks(rng, height, width, plan, training, index)
    panel_one = np.where(first, first_color, background).astype(np.int8)
    panel_two = np.where(second, second_color, background).astype(np.int8)
    if orientation == "vertical":
        divider = np.full((height, 1), divider_color, dtype=np.int8)
        input_grid = np.concatenate((panel_one, divider, panel_two), axis=1)
    else:
        divider = np.full((1, width), divider_color, dtype=np.int8)
        input_grid = np.concatenate((panel_one, divider, panel_two), axis=0)
    output_grid = np.where(first ^ second, plan["result"], background).astype(np.int8)
    return {"input": input_grid, "output": output_grid}


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def _pair_key(pair):
    return _grid_key(pair["input"]), _grid_key(pair["output"])


def _source_copy(pair):
    return any(np.array_equal(pair["input"], x) and np.array_equal(pair["output"], y)
               for x, y in _SOURCE_PAIRS)


def _parse_input(input_grid):
    height, width = input_grid.shape
    candidates = []
    if width >= 5 and width % 2 == 1:
        middle = width // 2
        candidates.append(("vertical", input_grid[:, :middle], input_grid[:, middle + 1:], input_grid[:, middle]))
    if height >= 5 and height % 2 == 1:
        middle = height // 2
        candidates.append(("horizontal", input_grid[:middle, :], input_grid[middle + 1:, :], input_grid[middle, :]))
    valid = []
    for orientation, first_panel, second_panel, divider in candidates:
        if not np.all(divider == divider.flat[0]):
            continue
        divider_color = int(divider.flat[0])
        first_colors = set(int(c) for c in np.unique(first_panel))
        second_colors = set(int(c) for c in np.unique(second_panel))
        if len(first_colors) != 2 or len(second_colors) != 2:
            continue
        common = first_colors & second_colors
        if len(common) != 1 or divider_color in first_colors | second_colors:
            continue
        background = next(iter(common))
        first_color = next(iter(first_colors - common))
        second_color = next(iter(second_colors - common))
        if first_color == second_color:
            continue
        first_mask = first_panel == first_color
        second_mask = second_panel == second_color
        categories = first_mask.astype(np.int8) + 2 * second_mask.astype(np.int8)
        counts = tuple(int(np.count_nonzero(categories == case)) for case in range(4))
        valid.append({
            "orientation": orientation, "size": first_panel.shape,
            "background": background, "first_color": first_color,
            "second_color": second_color, "divider": divider_color,
            "first_mask": first_mask, "second_mask": second_mask,
            "category_counts": counts,
        })
    assert len(valid) == 1, "input must have one uniquely identifiable divider and two binary panels"
    return valid[0]


def _learn_result(train_pairs):
    results = set()
    demonstrated = set()
    for pair in train_pairs:
        info = _parse_input(pair["input"])
        output = pair["output"]
        assert output.shape == info["size"]
        other_colors = set(int(c) for c in np.unique(output)) - {info["background"]}
        assert len(other_colors) <= 1
        results.update(other_colors)
        demonstrated.update(case for case, count in enumerate(info["category_counts"]) if count)
    assert len(results) == 1, "training must establish one result color"
    assert demonstrated == {0, 1, 2, 3}, "training must demonstrate the full XOR truth table"
    return next(iter(results))


def _predict(train_pairs, test_input):
    result = _learn_result(train_pairs)
    info = _parse_input(test_input)
    assert result not in {info["background"], info["first_color"], info["second_color"], info["divider"]}
    return np.where(info["first_mask"] ^ info["second_mask"], result, info["background"]).astype(np.int8)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    seen_inputs, seen_pairs = set(), set()
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= dimension <= 30 for dimension in grid.shape)
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            input_key, pair_key = _grid_key(pair["input"]), _pair_key(pair)
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert not _source_copy(pair)
            info = _parse_input(pair["input"])
            assert pair["output"].shape == info["size"]
    result = _learn_result(puzzle["train"])
    for section in ("train", "test"):
        for pair in puzzle[section]:
            info = _parse_input(pair["input"])
            assert result not in {info["background"], info["first_color"], info["second_color"], info["divider"]}
            expected = np.where(info["first_mask"] ^ info["second_mask"], result, info["background"])
            assert np.array_equal(pair["output"], expected), "output must be exact aligned XOR"
    # Predict using only training pairs and each test input.
    for pair in puzzle["test"]:
        assert np.array_equal(_predict(puzzle["train"], pair["input"]), pair["output"])


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _integer(seed, "seed", 0, None)
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(seed, rng, num_train, num_test)
    plan = _plan(seed, rng)
    puzzle = {"train": [], "test": []}
    seen = set()
    for overall_index in range(num_train + num_test):
        training = overall_index < num_train
        section = "train" if training else "test"
        index = overall_index if training else overall_index - num_train
        pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
        for _ in range(128):
            pair = _make_pair(pair_rng, plan, overall_index if not training else index, training)
            key = _grid_key(pair["input"])
            if key not in seen and not _source_copy(pair):
                seen.add(key)
                puzzle[section].append(pair)
                break
        else:
            raise AssertionError("could not construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _fingerprint(puzzle):
    return tuple(_pair_key(pair) for section in ("train", "test") for pair in puzzle[section])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_fingerprint(p) for p in puzzles}) == 42
    orientations, sizes, colors = set(), set(), set()
    mixed_orientations = mixed_sizes = mixed_palettes = 0
    fixed_conventions = combined_variations = swapped_foregrounds = 0
    small = long_thin = large = source_sized = dense_sparse = 0
    vertical_to_horizontal = horizontal_to_vertical = 0
    special = {"identical": 0, "complementary": 0, "disjoint": 0, "contained": 0}
    train_denser = test_denser = overlapping = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        infos = [_parse_input(pair["input"]) for section in ("train", "test") for pair in puzzle[section]]
        train_infos = infos[:len(puzzle["train"])]
        test_infos = infos[len(puzzle["train"]):]
        signatures = []
        result = _learn_result(puzzle["train"])
        for info in infos:
            orientations.add(info["orientation"])
            sizes.add(info["size"])
            height, width = info["size"]
            small += height <= 4 and width <= 4
            long_thin += min(height, width) <= 3 and max(height, width) >= 8
            large += max(height, width) >= 20
            source_sized += (height, width) == (5, 4)
            signature = (info["background"], info["first_color"], info["second_color"], info["divider"])
            signatures.append(signature)
            colors.add(signature)
            neither, first_only, second_only, both = info["category_counts"]
            first_density = (first_only + both) / (height * width)
            second_density = (second_only + both) / (height * width)
            dense_sparse += min(first_density, second_density) < .4 and max(first_density, second_density) > .6
            if first_only == second_only == 0:
                special["identical"] += 1
            if neither == both == 0:
                special["complementary"] += 1
            if both == 0 and first_only and second_only:
                special["disjoint"] += 1
            if both and (first_only == 0) != (second_only == 0):
                special["contained"] += 1
        mixed_orientations += len({i["orientation"] for i in infos}) > 1
        mixed_sizes += len({i["size"] for i in infos}) > 1
        mixed_palettes += len(set(signatures)) > 1
        fixed_conventions += (len({i["orientation"] for i in infos}) == 1 and
                              len({i["size"] for i in infos}) == 1 and len(set(signatures)) == 1)
        combined_variations += (len({i["orientation"] for i in infos}) > 1 and
                                len({i["size"] for i in infos}) > 1 and
                                len(set(signatures)) > 1 and
                                any(not all(i["category_counts"]) for i in infos))
        swapped_foregrounds += any(
            a[0] == b[0] and a[3] == b[3] and a[1] == b[2] and a[2] == b[1]
            for a in signatures for b in signatures if a != b
        )
        vertical_to_horizontal += (any(i["orientation"] == "vertical" for i in train_infos) and
                                   any(i["orientation"] == "horizontal" for i in test_infos))
        horizontal_to_vertical += (any(i["orientation"] == "horizontal" for i in train_infos) and
                                   any(i["orientation"] == "vertical" for i in test_infos))
        for section in ("train", "test"):
            for pair in puzzle[section]:
                info = _parse_input(pair["input"])
                independent_expected = np.where(
                    info["first_mask"] != info["second_mask"], result, info["background"]
                )
                assert np.array_equal(pair["output"], independent_expected)
        train_density = np.mean([(i["category_counts"][1] + i["category_counts"][3]) / np.prod(i["size"]) for i in train_infos])
        test_density = np.mean([(i["category_counts"][1] + i["category_counts"][3]) / np.prod(i["size"]) for i in test_infos])
        train_denser += train_density > test_density
        test_denser += test_density > train_density
        overlapping += abs(train_density - test_density) < 0.15
    assert orientations == {"vertical", "horizontal"}
    assert len(sizes) >= 35 and len(colors) >= 60
    assert mixed_orientations >= 20 and mixed_sizes >= 20 and mixed_palettes >= 20
    assert fixed_conventions >= 3 and combined_variations >= 20 and swapped_foregrounds >= 3
    assert small >= 10 and long_thin >= 20 and large >= 20 and source_sized >= 10
    assert dense_sparse >= 20 and vertical_to_horizontal >= 5 and horizontal_to_vertical >= 5
    assert all(value >= 2 for value in special.values())
    assert train_denser >= 5 and test_denser >= 5 and overlapping >= 5
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(50000 + index, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for index, (train, test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(51000 + index, train, test)
        assert (train is None or len(puzzle["train"]) == train)
        assert (test is None or len(puzzle["test"]) == test)
    for index, (train, test) in enumerate(((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4), (True, 1), (3, False))):
        try:
            generate_puzzle(52000 + index, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request was accepted")
    assert _fingerprint(generate_puzzle(12345, 7, 3)) == _fingerprint(generate_puzzle(12345, 7, 3))
    assert _fingerprint(generate_puzzle(24680)) == _fingerprint(generate_puzzle(24680))
