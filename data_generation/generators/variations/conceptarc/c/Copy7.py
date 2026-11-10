import numpy as np


_ORIGINAL_INPUTS = {
    ("6366660777777", "6030060700007", "6003060700007", "6000360700007", "6000030700007", "6666660777777"),
    ("0000000000", "4444100000", "4441400000", "4414400222", "4144400222", "1444400222", "0000000000", "0000000000", "0000000000", "0000000000"),
    ("11111", "11111", "12221", "11111", "11111", "00000", "00000", "77777", "77777", "77777", "77777", "77777"),
    ("000000000000", "300000000400", "390000004400", "393000044400", "393300444400", "393000044400", "390000004400", "300000000400", "000000000000", "000000000000", "000000000000"),
    ("88888022222", "87778020002", "87778020002", "87778020002", "87778020002", "88888022222"),
}
_KINDS = ("outline", "solid", "stripe", "double", "frame_stripe", "fill", "taper")


def _key(grid):
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _border(height, width):
    mask = np.ones((height, width), dtype=bool)
    mask[1:-1, 1:-1] = False
    return mask


def _outline_mark(size, reverse):
    mask = np.zeros((size, size), dtype=bool)
    for row in range(size - 1):
        mask[row, size - 2 - row if reverse else row + 1] = True
    return mask


def _diagonal(size, reverse):
    mask = np.eye(size, dtype=bool)
    return np.fliplr(mask) if reverse else mask


def _stripe(height, width, axis, index):
    mask = np.zeros((height, width), dtype=bool)
    if axis == "horizontal":
        mask[index, 1:-1] = True
    else:
        mask[1:-1, index] = True
    return mask


def _double(height, width, axis):
    if axis == "horizontal":
        return _stripe(height, width, axis, 1) | _stripe(height, width, axis, height - 2)
    return _stripe(height, width, axis, 1) | _stripe(height, width, axis, width - 2)


def _taper(width):
    height = 2 * width - 1
    occupied = np.zeros((height, width), dtype=bool)
    mark = np.zeros_like(occupied)
    for row in range(height):
        extent = width - abs(row - (width - 1))
        occupied[row, :extent] = True
        if extent >= 2:
            mark[row, 1] = True
    return occupied, mark


def _patches(rng, kind, colors):
    background, source_color, target_color, accent = colors
    if kind == "outline":
        size = int(rng.integers(5, 11))
        carrier = _border(size, size)
        mark = _outline_mark(size, bool(rng.integers(0, 2)))
        source_occupied = carrier | mark
        target_occupied = carrier
        target_mark = mark
    elif kind == "solid":
        source_size = int(rng.integers(4, 10))
        choices = [size for size in range(3, 10) if size != source_size]
        target_size = int(rng.choice(choices))
        reverse = bool(rng.integers(0, 2))
        source_occupied = np.ones((source_size, source_size), dtype=bool)
        target_occupied = np.ones((target_size, target_size), dtype=bool)
        mark = _diagonal(source_size, reverse)
        target_mark = _diagonal(target_size, reverse)
    elif kind in ("stripe", "double", "frame_stripe"):
        height = int(rng.integers(5 if kind == "double" else 4, 10))
        width = int(rng.integers(5 if kind == "double" else 4, 10))
        axis = "horizontal" if bool(rng.integers(0, 2)) else "vertical"
        if kind == "double":
            mark = _double(height, width, axis)
        else:
            limit = height if axis == "horizontal" else width
            index = int(rng.integers(1, limit - 1))
            mark = _stripe(height, width, axis, index)
        source_occupied = np.ones((height, width), dtype=bool) if kind != "frame_stripe" else _border(height, width) | mark
        target_occupied = np.ones((height, width), dtype=bool) if kind != "frame_stripe" else _border(height, width)
        target_mark = mark
    elif kind == "fill":
        height = int(rng.integers(4, 11))
        width = int(rng.integers(4, 11))
        carrier = _border(height, width)
        mark = ~carrier
        source_occupied = carrier | mark
        target_occupied = carrier
        target_mark = mark
    elif kind == "taper":
        width = int(rng.integers(4, 8))
        source_occupied, mark = _taper(width)
        target_occupied = np.fliplr(source_occupied)
        target_mark = np.fliplr(mark)
        turns = int(rng.integers(0, 4))
        source_occupied = np.rot90(source_occupied, turns)
        target_occupied = np.rot90(target_occupied, turns)
        mark = np.rot90(mark, turns)
        target_mark = np.rot90(target_mark, turns)
    else:
        raise AssertionError("unknown motif")

    source = np.where(mark, accent, np.where(source_occupied, source_color, background)).astype(np.int8)
    target = np.where(target_occupied, target_color, background).astype(np.int8)
    completed = np.where(target_mark, accent, target).astype(np.int8)
    return source, target, completed


def _place(rng, source, target, completed, background, layout):
    first_is_source = bool(rng.integers(0, 2))
    first, second = (source, target) if first_is_source else (target, source)
    along = (first.shape[1] + second.shape[1]) if layout == "horizontal" else (first.shape[0] + second.shape[0])
    maximum_gap = min(7, 30 - along - 2)
    assert maximum_gap >= 1
    gap = int(rng.integers(1, maximum_gap + 1))
    cross = (0 if layout == "horizontal" else 1)
    offset_limit = min(4, first.shape[cross] - 1, second.shape[cross] - 1)
    first_offset = int(rng.integers(0, offset_limit + 1))
    second_offset = int(rng.integers(0, offset_limit + 1))

    if layout == "horizontal":
        content_height = max(first_offset + first.shape[0], second_offset + second.shape[0])
        content_width = along + gap
    else:
        content_height = along + gap
        content_width = max(first_offset + first.shape[1], second_offset + second.shape[1])

    top = int(rng.integers(1, min(4, 29 - content_height) + 1))
    bottom = int(rng.integers(1, min(4, 30 - content_height - top) + 1))
    left = int(rng.integers(1, min(4, 29 - content_width) + 1))
    right = int(rng.integers(1, min(4, 30 - content_width - left) + 1))
    height = top + content_height + bottom
    width = left + content_width + right
    grid = np.full((height, width), background, dtype=np.int8)
    if layout == "horizontal":
        positions = ((top + first_offset, left), (top + second_offset, left + first.shape[1] + gap))
    else:
        positions = ((top, left + first_offset), (top + first.shape[0] + gap, left + second_offset))
    for patch, (y, x) in zip((first, second), positions):
        grid[y:y + patch.shape[0], x:x + patch.shape[1]] = patch
    output = grid.copy()
    target_y, target_x = positions[1 if first_is_source else 0]
    output[target_y:target_y + target.shape[0], target_x:target_x + target.shape[1]] = completed
    return {"input": grid, "output": output}


def _components(grid, background):
    occupied = grid != background
    seen = np.zeros(grid.shape, dtype=bool)
    result = []
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            if not occupied[row, col] or seen[row, col]:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for yy in range(max(0, y - 1), min(grid.shape[0], y + 2)):
                    for xx in range(max(0, x - 1), min(grid.shape[1], x + 2)):
                        if occupied[yy, xx] and not seen[yy, xx]:
                            seen[yy, xx] = True
                            stack.append((yy, xx))
            ys, xs = zip(*cells)
            result.append((cells, (min(ys), max(ys) + 1, min(xs), max(xs) + 1)))
    return result


def _recognize(source, target, background, source_color, target_color, accent):
    sh, sw = source.shape
    th, tw = target.shape
    source_mark = source == accent
    target_occ = target != background
    matches = []

    def add(kind, carrier, mark, target_carrier, target_mark, detail):
        expected_source = np.where(mark, accent, np.where(carrier, source_color, background))
        expected_target = np.where(target_carrier, target_color, background)
        if np.array_equal(source, expected_source) and np.array_equal(target, expected_target):
            matches.append((kind, target_mark, detail))

    if np.all(target_occ):
        if sh == sw and th == tw and 4 <= sh <= 9 and 3 <= th <= 9:
            for reverse in (False, True):
                add("solid", np.ones(source.shape, dtype=bool), _diagonal(sh, reverse), target_occ, _diagonal(th, reverse), ("reverse" if reverse else "forward",))
        if source.shape == target.shape and 4 <= sh <= 9 and 4 <= sw <= 9:
            for axis in ("horizontal", "vertical"):
                limit = sh if axis == "horizontal" else sw
                for index in range(1, limit - 1):
                    add("stripe", np.ones(source.shape, dtype=bool), _stripe(sh, sw, axis, index), target_occ, _stripe(th, tw, axis, index), (axis, index))
                if sh >= 5 and sw >= 5:
                    double = _double(sh, sw, axis)
                    add("double", np.ones(source.shape, dtype=bool), double, target_occ, double, (axis,))
    if source.shape == target.shape and np.array_equal(target_occ, _border(th, tw)):
        border = _border(sh, sw)
        if sh == sw and 5 <= sh <= 10:
            for reverse in (False, True):
                mark = _outline_mark(sh, reverse)
                add("outline", border, mark, target_occ, mark, ("reverse" if reverse else "forward",))
        if 4 <= sh <= 10 and 4 <= sw <= 10:
            interior = ~border
            add("fill", border, interior, target_occ, interior, ())
            for axis in ("horizontal", "vertical"):
                limit = sh if axis == "horizontal" else sw
                for index in range(1, limit - 1):
                    mark = _stripe(sh, sw, axis, index)
                    add("frame_stripe", border, mark, target_occ, mark, (axis, index))
    for width in range(4, 8):
        base_occ, base_mark = _taper(width)
        for turns in range(4):
            carrier = np.rot90(base_occ, turns)
            mark = np.rot90(base_mark, turns)
            target_carrier = np.rot90(np.fliplr(base_occ), turns)
            target_mark = np.rot90(np.fliplr(base_mark), turns)
            if source.shape == carrier.shape and target.shape == target_carrier.shape:
                add("taper", carrier, mark, target_carrier, target_mark, (width, turns))
    assert len(matches) == 1, f"ambiguous or invalid motif: {len(matches)} matches"
    kind, mark, detail = matches[0]
    assert np.array_equal(source_mark, source == accent)
    return kind, mark, detail


def _infer(input_grid):
    background = int(input_grid[0, 0])
    assert all(int(input_grid[y, x]) == background for y, x in ((0, -1), (-1, 0), (-1, -1)))
    components = _components(input_grid, background)
    assert len(components) == 2
    source_options = [item for item in components if len({int(input_grid[y, x]) for y, x in item[0]}) == 2]
    assert len(source_options) == 1
    source_component = source_options[0]
    target_component = next(item for item in components if item is not source_component)
    source_box = source_component[1]
    target_box = target_component[1]
    sy0, sy1, sx0, sx1 = source_box
    ty0, ty1, tx0, tx1 = target_box
    source = input_grid[sy0:sy1, sx0:sx1]
    target = input_grid[ty0:ty1, tx0:tx1]
    target_colors = {int(input_grid[y, x]) for y, x in target_component[0]}
    assert len(target_colors) == 1
    target_color = target_colors.pop()
    source_colors = {int(input_grid[y, x]) for y, x in source_component[0]}
    assert target_color not in source_colors and background not in source_colors
    candidates = []
    for accent in source_colors:
        source_color = next(v for v in source_colors if v != accent)
        try:
            kind, target_mark, detail = _recognize(source, target, background, source_color, target_color, accent)
        except AssertionError:
            continue
        candidates.append((kind, target_mark, detail, source_color, accent))
    assert len(candidates) == 1, "source accent is not uniquely identifiable"
    kind, target_mark, detail, source_color, accent = candidates[0]
    expected = input_grid.copy()
    destination = expected[ty0:ty1, tx0:tx1]
    destination[target_mark] = accent
    assert not np.array_equal(expected, input_grid)
    horizontal = sx1 < tx0 or tx1 < sx0
    vertical = sy1 < ty0 or ty1 < sy0
    assert horizontal != vertical
    layout = "horizontal" if horizontal else "vertical"
    source_first = sx0 < tx0 if horizontal else sy0 < ty0
    gap = (max(sx0, tx0) - min(sx1, tx1)) if horizontal else (max(sy0, ty0) - min(sy1, ty1))
    assert 1 <= gap <= 7
    return expected, {
        "kind": kind, "layout": layout, "source_first": source_first,
        "background": background, "palette": (source_color, target_color, accent),
        "source_shape": source.shape, "target_shape": target.shape,
        "detail": detail, "gap": gap,
        "source_origin": (sy0, sx0), "target_origin": (ty0, tx0),
        "signature": (kind, layout, source_first, source.shape, target.shape, detail),
    }


def _check_count(value, minimum, maximum, name):
    if value is None:
        return
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or not minimum <= int(value) <= maximum:
        raise ValueError(f"{name} is outside its legal range")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    _check_count(num_train, 2, 10, "num_train")
    _check_count(num_test, 1, 4, "num_test")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")

    order = list(rng.permutation(_KINDS))
    train_kinds = (order + [str(rng.choice(_KINDS)) for _ in range(max(0, num_train - len(order)))])[:num_train]
    test_kinds = [str(rng.choice(train_kinds)) for _ in range(num_test)]
    first_layout = "horizontal" if bool(rng.integers(0, 2)) else "vertical"
    layouts = [first_layout, "vertical" if first_layout == "horizontal" else "horizontal"]
    layouts.extend("horizontal" if bool(rng.integers(0, 2)) else "vertical" for _ in range(num_train + num_test - 2))
    pair_seeds = rng.integers(0, 2**63 - 1, size=num_train + num_test, dtype=np.int64)
    pairs = []
    features = []
    for index, (kind, layout, pair_seed) in enumerate(zip(train_kinds + test_kinds, layouts, pair_seeds)):
        pair_rng = np.random.default_rng(int(pair_seed))
        for attempt in range(6):
            if index == 0:
                background = 0 if bool(pair_rng.integers(0, 2)) else int(pair_rng.integers(1, 10))
            elif index == 1:
                background = int(pair_rng.integers(1, 10)) if features[0]["background"] == 0 else 0
            else:
                background = 0 if pair_rng.random() < 0.5 else int(pair_rng.integers(1, 10))
            palette = tuple(int(v) for v in pair_rng.choice([v for v in range(10) if v != background], size=3, replace=False))
            source, target, completed = _patches(pair_rng, kind, (background,) + palette)
            pair = _place(pair_rng, source, target, completed, background, layout)
            _, feature = _infer(pair["input"])
            if _key(pair["input"]) in _ORIGINAL_INPUTS:
                continue
            if any(np.array_equal(pair["input"], old["input"]) for old in pairs):
                continue
            if index >= num_train and any(feature["signature"] == old["signature"] for old in features[:num_train]):
                continue
            break
        else:
            raise AssertionError("could not construct a new pair")
        pairs.append(pair)
        features.append(feature)
    puzzle = {"train": pairs[:num_train], "test": pairs[num_train:]}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    seen = set()
    features = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for role in ("input", "output"):
                grid = pair[role]
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dimension <= 30 for dimension in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((grid >= 0) & (grid <= 9)))
            inp = pair["input"]
            assert _key(inp) not in seen and _key(inp) not in _ORIGINAL_INPUTS
            seen.add(_key(inp))
            prediction, feature = _infer(inp)
            assert np.array_equal(prediction, pair["output"]), "incorrect accent transfer or unrelated change"
            features[split].append(feature)
    train = features["train"]
    test = features["test"]
    assert len({item["kind"] for item in train}) >= 2
    assert len({item["layout"] for item in train}) >= 2
    assert len({item["background"] for item in train}) >= 2
    assert len({item["palette"] for item in train}) >= 2
    train_kinds = {item["kind"] for item in train}
    train_signatures = {item["signature"] for item in train}
    assert all(item["kind"] in train_kinds and item["signature"] not in train_signatures for item in test)


def _same_puzzle(first, second):
    return all(len(first[split]) == len(second[split]) and all(np.array_equal(a[role], b[role]) for a, b in zip(first[split], second[split]) for role in ("input", "output")) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(puzzle["train"]) for puzzle in puzzles}) >= 4
    assert len({len(puzzle["test"]) for puzzle in puzzles}) >= 3
    assert all(_same_puzzle(puzzle, generate_puzzle(seed)) for seed, puzzle in enumerate(puzzles))
    assert all(not _same_puzzle(puzzles[index], puzzles[index + 1]) for index in range(41))
    kinds = {"train": set(), "test": set()}
    layouts = {"train": set(), "test": set()}
    backgrounds = set()
    palettes = set()
    shapes = set()
    details = set()
    taper_widths = set()
    taper_turns = set()
    stripe_axes = set()
    outline_sizes = set()
    outline_directions = set()
    source_orders = set()
    gaps = set()
    role_colors = [set(), set(), set()]
    motif_layout_order = set()
    solid_resize_directions = set()
    rectangular_frames = 0
    within_puzzle_order_diversity = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        within_puzzle_order_diversity += len({
            _infer(pair["input"])[1]["source_first"]
            for split in ("train", "test") for pair in puzzle[split]
        }) == 2
        for split in ("train", "test"):
            for pair in puzzle[split]:
                prediction, feature = _infer(pair["input"])
                assert np.array_equal(prediction, pair["output"])
                kinds[split].add(feature["kind"])
                layouts[split].add(feature["layout"])
                backgrounds.add(feature["background"])
                palettes.add(feature["palette"])
                shapes.add(pair["input"].shape)
                details.add((feature["kind"], feature["detail"]))
                source_orders.add(feature["source_first"])
                gaps.add(feature["gap"])
                motif_layout_order.add((feature["kind"], feature["layout"], feature["source_first"]))
                for index, color in enumerate(feature["palette"]):
                    role_colors[index].add(color)
                if feature["kind"] == "taper":
                    taper_widths.add(feature["detail"][0])
                    taper_turns.add(feature["detail"][1])
                if feature["kind"] in ("stripe", "double", "frame_stripe"):
                    stripe_axes.add((feature["kind"], feature["detail"][0]))
                if feature["kind"] == "outline":
                    outline_sizes.add(feature["source_shape"][0])
                    outline_directions.add(feature["detail"][0])
                if feature["kind"] == "solid":
                    solid_resize_directions.add(feature["target_shape"][0] > feature["source_shape"][0])
                if feature["kind"] in ("fill", "frame_stripe"):
                    rectangular_frames += feature["source_shape"][0] != feature["source_shape"][1]
    assert kinds["train"] == set(_KINDS) and kinds["test"] == set(_KINDS)
    assert layouts["train"] == layouts["test"] == {"horizontal", "vertical"}
    assert len(backgrounds) >= 5 and 0 in backgrounds
    assert len(palettes) >= 40 and len(shapes) >= 35 and len(details) >= 30
    assert taper_widths == {4, 5, 6, 7} and taper_turns == {0, 1, 2, 3}
    assert stripe_axes == {(kind, axis) for kind in ("stripe", "double", "frame_stripe") for axis in ("horizontal", "vertical")}
    assert outline_sizes == set(range(5, 11)) and outline_directions == {"forward", "reverse"}
    assert source_orders == {False, True} and within_puzzle_order_diversity >= 30
    assert gaps == set(range(1, 8)) and len(motif_layout_order) >= 25
    assert solid_resize_directions == {False, True} and rectangular_frames >= 20
    assert all(len(colors) >= 8 for colors in role_colors)
    assert np.array_equal(_outline_mark(6, False).nonzero()[1], np.arange(1, 6))
    assert np.array_equal(_outline_mark(6, True).nonzero()[1], np.arange(4, -1, -1))
    for train_count, test_count in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(937, train_count, test_count)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(937, train_count, test_count))
        assert train_count is None or len(puzzle["train"]) == train_count
        assert test_count is None or len(puzzle["test"]) == test_count
    for train_count, test_count in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(99, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts accepted")
