import numpy as np


_SHAPES = ((3, 3), (3, 5), (5, 3), (5, 5), (3, 7), (7, 3))
_SOURCE_INPUTS = {
    (
        "2735000000000", "2335000010000", "3775000000000",
        "0005000000000", "0005010000010", "0005000000000",
        "0005000000000", "0005010010000", "0005000000000",
    ),
    (
        "4225000000000", "2625010000000", "6445000000000",
        "0005000000000", "0005000010000", "0005000000000",
        "0005000000000", "0005000010000", "0005000000000",
    ),
    (
        "3865000000000", "9825010000010", "9995000000000",
        "0005000000000", "0005000010000", "0005000000000",
        "0005000000000", "0005000010010", "0005000000000",
    ),
    (
        "3395000000000", "8445000010010", "8985000000000",
        "0005000000000", "0005010000010", "0005000000000",
        "0005000000000", "0005010010000", "0005000000000",
    ),
}


def _counts(rng, num_train, num_test):
    for name, value, low, high in (
        ("num_train", num_train, 2, 10),
        ("num_test", num_test, 1, 4),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high
        ):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_test = int(num_test)
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_train = int(num_train)
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    else:
        num_train, num_test = int(num_train), int(num_test)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _geometries(orientation, shapes):
    options = []
    for ph, pw in shapes:
        for rows in range(2, 6):
            for cols in range(2, 6):
                height = ph * rows + (ph + 1 if orientation == "horizontal" else 0)
                width = pw * cols + (pw + 1 if orientation == "vertical" else 0)
                if height <= 30 and width <= 30:
                    options.append((ph, pw, rows, cols))
    return options


def _template(rng, shape, colors, background, style):
    ph, pw = shape
    if style == "solid":
        count = 1
    elif style in ("two", "horizontal", "vertical", "checker"):
        count = 2
    else:
        count = int(rng.integers(3, min(5, len(colors)) + 1))
    palette = rng.choice(colors, size=count, replace=False)
    rr, cc = np.indices(shape)
    if style == "solid":
        patch = np.full(shape, palette[0], dtype=np.int8)
    elif style == "horizontal":
        patch = palette[rr % count].astype(np.int8)
    elif style == "vertical":
        patch = palette[cc % count].astype(np.int8)
    elif style == "checker":
        patch = palette[(rr + cc) % count].astype(np.int8)
    elif style == "mirror":
        half = rng.choice(palette, size=(ph, (pw + 1) // 2))
        patch = np.concatenate((half, half[:, :pw // 2][:, ::-1]), axis=1).astype(np.int8)
    elif style == "repeated_rows":
        row = rng.choice(palette, size=pw)
        patch = np.tile(row, (ph, 1)).astype(np.int8)
    else:
        patch = rng.choice(palette, size=shape).astype(np.int8)
        flat = patch.ravel()
        flat[:count] = palette
        patch = rng.permutation(flat).reshape(shape)
    if style == "hollow":
        patch[ph // 2, pw // 2] = background
        for row in range(1, ph - 1):
            for col in range(1, pw - 1):
                if (row, col) != (ph // 2, pw // 2) and rng.random() < 0.2:
                    patch[row, col] = background
    return patch


def _marker_count(rng, total, first):
    choice = int(rng.integers(0, 100))
    if choice < 5:
        count = 0
    elif choice < 21:
        count = 1
    elif choice < 36:
        count = 2
    elif choice < 74:
        count = int(rng.integers(3, total + 1))
    elif choice < 88:
        count = int(rng.integers(max(2, total - 2), total + 1))
    else:
        count = total
    return max(2, count) if first else count


def _selected_slots(rng, rows, cols, count, force_touch):
    total = rows * cols
    if count == 0:
        return ()
    if force_touch:
        adjacent = []
        for row in range(rows):
            for col in range(cols):
                first = row * cols + col
                if col + 1 < cols:
                    adjacent.append((first, first + 1))
                if row + 1 < rows:
                    adjacent.append((first, first + cols))
        selected = list(adjacent[int(rng.integers(len(adjacent)))])
        remaining = [slot for slot in range(total) if slot not in selected]
        if count > 2:
            selected.extend(map(int, rng.choice(remaining, size=count - 2, replace=False)))
        return tuple(sorted(selected))
    return tuple(sorted(map(int, rng.choice(total, size=count, replace=False))))


def _make_pair(rng, convention, geometry, first=False, hollow=False,
               avoid_offset=None, avoid_background=None):
    orientation, source_side, multicolor_markers = convention
    ph, pw, rows, cols = geometry
    if avoid_background is None:
        role_colors = list(map(int, rng.choice(10, size=4, replace=False)))
    else:
        alternatives = [color for color in range(10) if color != avoid_background]
        background = int(rng.choice(alternatives))
        remainder = [color for color in range(10) if color != background]
        role_colors = [background] + list(map(int, rng.choice(remainder, size=3,
                                                               replace=False)))
    background, divider, marker_a, marker_b = role_colors
    total = rows * cols
    count = max(1, _marker_count(rng, total, first)) if hollow else _marker_count(rng, total, first)
    use_two_markers = (multicolor_markers and count >= 2 and
                       (first or rng.random() < 0.45))
    markers = (marker_a, marker_b) if use_two_markers else (marker_a,)
    payload = [color for color in range(10) if color not in role_colors]
    styles = ("solid", "two", "irregular", "irregular", "horizontal",
              "vertical", "checker", "mirror", "repeated_rows", "hollow")
    style = "hollow" if hollow else (
        "irregular" if first else styles[int(rng.integers(len(styles)))])
    patch = _template(rng, (ph, pw), payload, background, style)
    selected = _selected_slots(rng, rows, cols, count, first)

    field_h, field_w = ph * rows, pw * cols
    if orientation == "vertical":
        height, width = field_h, field_w + pw + 1
        source_start = 0 if source_side == "left" else field_w + 1
        field_start = pw + 1 if source_side == "left" else 0
        divider_at = pw if source_side == "left" else field_w
        offsets = [value for value in range(field_h - ph + 1)
                   if value != avoid_offset]
        offset = int(rng.choice(offsets))
        grid = np.full((height, width), background, dtype=np.int8)
        grid[:, divider_at] = divider
        grid[offset:offset + ph, source_start:source_start + pw] = patch
        def slot_origin(slot):
            return ph * (slot // cols), field_start + pw * (slot % cols)
    else:
        height, width = field_h + ph + 1, field_w
        source_start = 0 if source_side == "top" else field_h + 1
        field_start = ph + 1 if source_side == "top" else 0
        divider_at = ph if source_side == "top" else field_h
        offsets = [value for value in range(field_w - pw + 1)
                   if value != avoid_offset]
        offset = int(rng.choice(offsets))
        grid = np.full((height, width), background, dtype=np.int8)
        grid[divider_at, :] = divider
        grid[source_start:source_start + ph, offset:offset + pw] = patch
        def slot_origin(slot):
            return field_start + ph * (slot // cols), pw * (slot % cols)

    result = grid.copy()
    for index, slot in enumerate(selected):
        row, col = slot_origin(slot)
        grid[row + ph // 2, col + pw // 2] = markers[index % len(markers)]
        result[row:row + ph, col:col + pw] = patch
    return {"input": grid, "output": result}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    orientation = ("vertical", "horizontal")[int(rng.integers(2))]
    source_side = (("left", "right") if orientation == "vertical"
                   else ("top", "bottom"))[int(rng.integers(2))]
    convention = (orientation, source_side, bool(rng.integers(2)))
    options = _geometries(orientation, _SHAPES)
    puzzle = {"train": [], "test": []}
    seen = set()
    train_layouts = set()

    for regime, number in (("train", num_train), ("test", num_test)):
        for index in range(number):
            if regime == "train" and index == 0:
                choices = [item for item in options if item[:2] == (3, 3)
                           and item[2] >= 3 and item[3] >= 3]
            elif regime == "train" and index == 1:
                choices = [item for item in options if item[:2] in ((3, 5), (5, 3))
                           and item[2:] != first_geometry[2:]]
            elif regime == "test" and rng.random() < 0.6:
                choices = [item for item in options if item[:2] in
                           ((5, 5), (3, 7), (7, 3))]
            else:
                choices = options
            for attempt in range(24):
                geometry = choices[int(rng.integers(len(choices)))]
                if regime == "train" and index == 0:
                    first_geometry = geometry
                first_feature = (_inspect_pair(puzzle["train"][0])
                                 if regime == "train" and index == 1 else None)
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                pair = _make_pair(pair_rng, convention, geometry,
                                  first=regime == "train" and index == 0,
                                  hollow=regime == "train" and index == 1,
                                  avoid_offset=(first_feature["source_offset"]
                                                if first_feature else None),
                                  avoid_background=(first_feature["roles"][0]
                                                    if first_feature else None))
                key = _grid_key(pair["input"])
                feature = _inspect_pair(pair)
                layout = (feature["shape"], feature["field_shape"],
                          feature["source_offset"], feature["selected"])
                if key not in seen and (regime != "test" or layout not in train_layouts):
                    break
            else:
                raise RuntimeError("Could not construct a distinct pair")
            seen.add(key)
            puzzle[regime].append(pair)
            if regime == "train":
                train_layouts.add(layout)
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_key(grid):
    return (grid.shape, tuple(map(int, grid.flat)))


def _inspect_pair(pair):
    assert type(pair) is dict and set(pair) == {"input", "output"}
    grid, output = pair["input"], pair["output"]
    for item in (grid, output):
        assert isinstance(item, np.ndarray) and item.ndim == 2
        assert np.issubdtype(item.dtype, np.integer)
        assert 1 <= item.shape[0] <= 30 and 1 <= item.shape[1] <= 30
        assert np.all((0 <= item) & (item <= 9))
    assert grid.shape == output.shape
    assert tuple("".join(map(str, row)) for row in grid) not in _SOURCE_INPUTS

    background = int(np.bincount(grid.ravel(), minlength=10).argmax())
    lines = []
    for col in range(grid.shape[1]):
        if np.all(grid[:, col] == grid[0, col]) and grid[0, col] != background:
            lines.append(("vertical", col, int(grid[0, col])))
    for row in range(grid.shape[0]):
        if np.all(grid[row] == grid[row, 0]) and grid[row, 0] != background:
            lines.append(("horizontal", row, int(grid[row, 0])))
    assert len(lines) == 1
    orientation, divider_at, divider = lines[0]
    if orientation == "vertical":
        before, after = grid[:, :divider_at], grid[:, divider_at + 1:]
        assert before.shape[1] and after.shape[1]
        source_side = "left" if before.shape[1] < after.shape[1] else "right"
        source = before if source_side == "left" else after
        field = after if source_side == "left" else before
    else:
        before, after = grid[:divider_at], grid[divider_at + 1:]
        assert before.shape[0] and after.shape[0]
        source_side = "top" if before.shape[0] < after.shape[0] else "bottom"
        source = before if source_side == "top" else after
        field = after if source_side == "top" else before

    occupied = np.argwhere(source != background)
    assert occupied.size
    r0, c0 = occupied.min(axis=0)
    r1, c1 = occupied.max(axis=0) + 1
    ph, pw = int(r1 - r0), int(c1 - c0)
    assert ph >= 3 and pw >= 3 and ph % 2 == pw % 2 == 1
    assert (ph, pw) in _SHAPES
    if orientation == "vertical":
        assert c0 == 0 and c1 == source.shape[1]
    else:
        assert r0 == 0 and r1 == source.shape[0]
    patch = source[r0:r1, c0:c1]
    assert np.all(patch[0] != background) and np.all(patch[-1] != background)
    assert np.all(patch[:, 0] != background) and np.all(patch[:, -1] != background)
    source_copy = source.copy()
    source_copy[r0:r1, c0:c1] = background
    assert np.all(source_copy == background)
    assert field.shape[0] % ph == 0 and field.shape[1] % pw == 0
    rows, cols = field.shape[0] // ph, field.shape[1] // pw
    assert 2 <= rows <= 5 and 2 <= cols <= 5

    prediction = grid.copy()
    expected_field = prediction[:, divider_at + 1:] if orientation == "vertical" and source_side == "left" else None
    if orientation == "vertical" and source_side == "right":
        expected_field = prediction[:, :divider_at]
    if orientation == "horizontal" and source_side == "top":
        expected_field = prediction[divider_at + 1:]
    if orientation == "horizontal" and source_side == "bottom":
        expected_field = prediction[:divider_at]
    marker_colors = set()
    selected = []
    for row in range(rows):
        for col in range(cols):
            cell = field[row * ph:(row + 1) * ph, col * pw:(col + 1) * pw]
            nonbackground = np.argwhere(cell != background)
            if nonbackground.size:
                assert nonbackground.shape == (1, 2)
                assert tuple(nonbackground[0]) == (ph // 2, pw // 2)
                marker_colors.add(int(cell[ph // 2, pw // 2]))
                selected.append(row * cols + col)
                expected_field[row * ph:(row + 1) * ph,
                               col * pw:(col + 1) * pw] = patch
    assert len(marker_colors) <= 2
    assert divider not in marker_colors and background not in marker_colors
    assert not np.isin(patch, list(marker_colors) + [divider]).any()
    assert np.array_equal(output, prediction)
    return {
        "orientation": orientation, "source_side": source_side,
        "shape": (ph, pw), "field_shape": (rows, cols),
        "source_offset": int(r0 if orientation == "vertical" else c0),
        "roles": (background, divider, tuple(sorted(marker_colors))),
        "palette_size": int(np.unique(patch[patch != background]).size),
        "blank_center": bool(patch[ph // 2, pw // 2] == background),
        "blank_interior": bool(np.any(patch == background)),
        "template": patch.copy(), "selected": tuple(selected),
    }


def _touches(selected, cols):
    chosen = set(selected)
    return any(slot + 1 in chosen and slot // cols == (slot + 1) // cols or
               slot + cols in chosen for slot in chosen)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert type(puzzle) is dict and set(puzzle) == {"train", "test"}
    assert type(puzzle["train"]) is list and type(puzzle["test"]) is list
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4
    assert len(train) > len(test)
    features = {"train": [], "test": []}
    seen_inputs, seen_pairs = set(), set()
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            feature = _inspect_pair(pair)
            features[regime].append(feature)
            input_key = _grid_key(pair["input"])
            pair_key = (input_key, _grid_key(pair["output"]))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
    all_features = features["train"] + features["test"]
    assert len({(f["orientation"], f["source_side"]) for f in all_features}) == 1
    assert len({f["roles"] for f in features["train"]}) >= 2
    assert len({f["shape"] for f in features["train"]}) >= 2
    assert len({f["field_shape"] for f in features["train"]}) >= 2
    assert len({f["source_offset"] for f in features["train"]}) >= 2
    assert (3, 3) in {f["shape"] for f in features["train"]}
    assert any(f["shape"][0] != f["shape"][1] for f in features["train"])
    assert any(f["blank_center"] and f["selected"] for f in features["train"])
    assert any(_touches(f["selected"], f["field_shape"][1])
               for f in features["train"])
    assert sum(bool(f["selected"]) for f in features["train"]) >= 2
    if any(len(f["roles"][2]) == 2 for f in features["test"]):
        assert any(len(f["roles"][2]) == 2 for f in features["train"])
    train_layouts = {(f["shape"], f["field_shape"], f["source_offset"],
                      f["selected"]) for f in features["train"]}
    assert all((f["shape"], f["field_shape"], f["source_offset"],
                f["selected"]) not in train_layouts for f in features["test"])


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                       for p in puzzle[regime]) for regime in ("train", "test"))


def self_test() -> None:
    puzzles = []
    train_counts, test_counts = set(), set()
    orientations, shapes, dimensions, roles = set(), set(), set(), set()
    marker_counts = {"train": set(), "test": set()}
    palettes, offsets = set(), set()
    empty, full, blank, two_markers = 0, 0, 0, 0
    horizontal, vertical, checker, repeated, mirrored = 0, 0, 0, 0, 0
    combinations = set()
    test_less, test_more = 0, 0
    within_roles, within_shapes, within_fields = 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        key = _puzzle_key(puzzle)
        assert key == _puzzle_key(generate_puzzle(seed))
        puzzles.append(key)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        features = {part: [_inspect_pair(p) for p in puzzle[part]]
                    for part in ("train", "test")}
        tf = features["train"]
        within_roles += len({f["roles"] for f in tf}) > 1
        within_shapes += len({f["shape"] for f in tf}) > 1
        within_fields += len({f["field_shape"] for f in tf}) > 1
        train_max = max(len(f["selected"]) for f in tf)
        test_less += any(len(f["selected"]) <= train_max for f in features["test"])
        test_more += any(len(f["selected"]) > train_max for f in features["test"])
        for part in ("train", "test"):
            for pair, f in zip(puzzle[part], features[part]):
                orientations.add((f["orientation"], f["source_side"]))
                shapes.add(f["shape"])
                dimensions.add(pair["input"].shape)
                roles.add(f["roles"][:2])
                marker_counts[part].add(len(f["selected"]))
                palettes.add(f["palette_size"])
                offsets.add(f["source_offset"])
                empty += len(f["selected"]) == 0
                full += len(f["selected"]) == np.prod(f["field_shape"])
                blank += f["blank_interior"]
                two_markers += len(f["roles"][2]) == 2
                patch = f["template"]
                ph, pw = patch.shape
                horizontal += (np.array_equal(
                    patch, np.tile(patch[:2], (ph // 2 + 1, 1))[:ph])
                    and not np.array_equal(patch[0], patch[1]))
                vertical += (np.array_equal(
                    patch, np.tile(patch[:, :2], (1, pw // 2 + 1))[:, :pw])
                    and not np.array_equal(patch[:, 0], patch[:, 1]))
                checker += (len(np.unique(patch)) == 2 and
                    np.array_equal(patch, np.where(
                        (np.indices(patch.shape).sum(axis=0) % 2) == 0,
                        patch[0, 0], patch[0, 1])))
                repeated += np.all(patch == patch[0])
                mirrored += np.array_equal(patch, patch[:, ::-1])
                combinations.add((f["orientation"], f["source_side"],
                                  f["shape"], f["blank_interior"]))
    assert len(set(puzzles)) >= 40
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(orientations) == 4 and len(shapes) == len(_SHAPES)
    assert len(dimensions) >= 25 and len(roles) >= 20
    assert {0, 1, 2}.issubset(marker_counts["train"] | marker_counts["test"])
    assert len(marker_counts["train"]) >= 8 and len(marker_counts["test"]) >= 8
    assert {1, 2, 3}.issubset(palettes)
    assert len(offsets) >= 8
    assert empty >= 3 and full >= 3 and blank >= 42 and two_markers >= 10
    assert horizontal >= 10 and vertical >= 10 and checker >= 5
    assert repeated >= 10 and mirrored >= 20 and len(combinations) >= 25
    assert test_less >= 10 and test_more >= 3
    assert within_roles == within_shapes == within_fields == 42

    for seed, requested_train, requested_test in (
        (101, 2, 1), (102, 10, 1), (103, 10, 4),
        (104, None, 1), (105, None, 4), (106, 2, None), (107, 10, None),
    ):
        puzzle = generate_puzzle(seed, requested_train, requested_test)
        validate_generated_puzzle(puzzle)
        if requested_train is not None:
            assert len(puzzle["train"]) == requested_train
        if requested_test is not None:
            assert len(puzzle["test"]) == requested_test
        assert _puzzle_key(puzzle) == _puzzle_key(
            generate_puzzle(seed, requested_train, requested_test))
    for arguments in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0},
        {"num_test": 5}, {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4}, {"num_train": True},
        {"num_test": 1.5},
    ):
        try:
            generate_puzzle(400, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {arguments}")
