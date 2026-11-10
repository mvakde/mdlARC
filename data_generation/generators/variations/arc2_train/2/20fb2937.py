import numpy as np


_SWATCH_STARTS = (0, 4, 8)
_CODE_COLUMNS = (1, 5, 9)

# Rejecting an original input is stronger than merely rejecting an exact
# original input/output pair, because an exact pair must contain that input.
_SOURCE_INPUT_SIGNATURES = {
    (
        "99972227333",
        "99972227333",
        "99972227333",
        "77777777777",
        "75777077787",
        "77777777777",
        "66666666666",
        "77777777777",
        "70777777777",
        "77777577877",
        "77777777777",
        "77777777777",
        "77777777777",
        "78777077757",
        "77777777777",
        "77777777777",
        "77777777777",
        "77777777787",
        "77787777777",
        "77777777777",
    ),
    (
        "55574447666",
        "55574447666",
        "55574447666",
        "77777777777",
        "72777877717",
        "77777777777",
        "66666666666",
        "77777777777",
        "71777777777",
        "77778777787",
        "77777777777",
        "77777777777",
        "72777777727",
        "77777177777",
        "77777777777",
        "77777777777",
        "78777778777",
        "77777777777",
        "77771777777",
        "77777777777",
    ),
    (
        "22271117555",
        "22271117555",
        "22271117555",
        "77777777777",
        "78777677797",
        "77777777777",
        "66666666666",
        "77777777777",
        "77777677797",
        "77977777777",
        "77777777777",
        "77777777777",
        "77877778777",
        "77777777777",
        "77777777777",
        "77777777777",
        "77977877777",
        "77777777797",
        "77777777777",
        "77777777777",
    ),
}


def _validate_optional_count(
    name: str,
    value: int | None,
    low: int,
    high: int,
) -> None:
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    _validate_optional_count("num_train", num_train, 2, 10)
    _validate_optional_count("num_test", num_test, 1, 4)

    train = None if num_train is None else int(num_train)
    test = None if num_test is None else int(num_test)
    key = int(seed)

    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must be greater than num_test")
        return train, test

    if train is None and test is None:
        test = 1 + key % 4
        minimum_train = max(2, test + 1)
        span = 11 - minimum_train
        train = minimum_train + (key // 4) % span
    elif train is None:
        minimum_train = max(2, test + 1)
        span = 11 - minimum_train
        train = minimum_train + key % span
    else:
        maximum_test = min(4, train - 1)
        test = 1 + key % maximum_test

    if not (2 <= train <= 10 and 1 <= test <= 4 and train > test):
        raise ValueError("no legal train/test count combination")

    return train, test


def _array_signature(grid: np.ndarray) -> tuple:
    return grid.shape, tuple(int(x) for x in grid.ravel())


def _rows_signature(grid: np.ndarray) -> tuple[str, ...]:
    return tuple("".join(str(int(x)) for x in row) for row in grid)



_SHAPES = {
    "square3": np.ones((3, 3), dtype=bool),
    "square5": np.ones((5, 5), dtype=bool),
    "wide": np.ones((3, 5), dtype=bool),
    "tall": np.ones((5, 3), dtype=bool),
    "cross": np.array([[0, 0, 1, 0, 0], [0, 0, 1, 0, 0],
                       [1, 1, 1, 1, 1], [0, 0, 1, 0, 0],
                       [0, 0, 1, 0, 0]], dtype=bool),
    "diamond": np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0],
                         [1, 1, 1, 1, 1], [0, 1, 1, 1, 0],
                         [0, 0, 1, 0, 0]], dtype=bool),
}


def _shape_choices(rng, style, first_train, regime, challenge):
    if first_train and len(style) > 1:
        return list(style)
    if first_train:
        return ["square3"] * 3
    if challenge == "entries" and regime == "target_test":
        count = 5
    else:
        maximum = 5 if challenge == "entries" else 6
        count = int(rng.integers(1, maximum))
    names = list(style)
    shapes = [names[int(rng.integers(len(names)))] for _ in range(count)]
    # Five entries must still leave room for visible gaps and a full field.
    while sum(_SHAPES[name].shape[1] for name in shapes) + count + 3 > 29:
        wide = next((i for i, name in enumerate(shapes)
                     if _SHAPES[name].shape[1] == 5), None)
        if wide is None:
            break
        shapes[wide] = "square3"
    return shapes


def _layout(rng, shapes, first_train, regime, challenge):
    widths = [_SHAPES[name].shape[1] for name in shapes]
    heights = [_SHAPES[name].shape[0] for name in shapes]
    gap = int(rng.integers(1, 4))
    left = int(rng.integers(1, 4))
    while sum(widths) + gap * (len(widths) - 1) + left + 1 > 30:
        if gap > 1:
            gap -= 1
        else:
            left -= 1
    right = int(rng.integers(1, 4))
    needed_width = sum(widths) + gap * (len(widths) - 1) + left + right
    width = min(30, max(needed_width, int(rng.integers(8, 29))))
    top = int(rng.integers(0, 3))
    swatch_gap = int(rng.integers(1, 4))
    code_gap = int(rng.integers(1, 3))
    code_row = top + max(heights) + swatch_gap
    divider_row = code_row + code_gap + 1
    min_field = 10 if first_train else 7
    max_field = min(19, 30 - divider_row - 1)
    if max_field < min_field:
        min_field = max_field
    if regime == "target_test" and (
            challenge == "field" or int(rng.integers(3)) == 0):
        field_height = max_field
    else:
        field_height = int(rng.integers(min_field, max_field + 1))
    starts = []
    cursor = left
    for w in widths:
        starts.append(cursor)
        cursor += w + gap
    return width, field_height, code_row, divider_row, starts


def _can_place(candidate, kind, placed, kinds, shapes, gap=0):
    r, c = candidate
    h, w = _SHAPES[shapes[kind]].shape
    for (pr, pc), pk in zip(placed, kinds):
        ph, pw = _SHAPES[shapes[pk]].shape
        if (abs(r - pr) < (h + ph) // 2 + gap and
                abs(c - pc) < (w + pw) // 2 + gap):
            return False
    return True


def _positions(rng, marker_kinds, shapes, height, width, spacing):
    count = len(marker_kinds)
    radii = [(_SHAPES[shapes[k]].shape[0] // 2,
              _SHAPES[shapes[k]].shape[1] // 2) for k in marker_kinds]
    candidates = [
        [(r, c) for r in range(rh, height - rh)
         for c in range(rw, width - rw)] for rh, rw in radii
    ]
    for attempt in range(36):
        placed = []
        for index, kind in enumerate(marker_kinds):
            choices = candidates[index]
            if index == 1 and spacing == "touch" and attempt < 24:
                r, c = placed[0]
                rh, rw = radii[0]
                kh, kw = radii[1]
                contact = []
                for dr in (-rh - kh - 1, rh + kh + 1):
                    for dc in range(-min(rw, kw), min(rw, kw) + 1):
                        contact.append((r + dr, c + dc))
                for dc in (-rw - kw - 1, rw + kw + 1):
                    for dr in range(-min(rh, kh), min(rh, kh) + 1):
                        contact.append((r + dr, c + dc))
                choices = [p for p in contact if p in candidates[index]]
                if not choices:
                    break
            order = rng.permutation(len(choices))
            separation = 1 if spacing == "separate" and attempt < 12 else 0
            found = None
            for j in order:
                p = choices[int(j)]
                if _can_place(p, kind, placed, marker_kinds[:index],
                              shapes, separation):
                    found = p
                    break
            if found is None:
                break
            placed.append(found)
        if len(placed) == count:
            return placed
    # A regular packing is always available because the requested count is
    # bounded by the maximum prototype's rectangular packing capacity.
    mh = max(_SHAPES[name].shape[0] for name in shapes)
    mw = max(_SHAPES[name].shape[1] for name in shapes)
    slots = [(r + mh // 2, c + mw // 2)
             for r in range(0, height - mh + 1, mh)
             for c in range(0, width - mw + 1, mw)]
    chosen = rng.choice(len(slots), size=count, replace=False)
    return [slots[int(j)] for j in chosen]


def _generate_pair(rng, regime, style, role_bias, challenge,
                   pair_index, previous_background):
    first_train = regime == "source_train" and pair_index == 0
    shapes = _shape_choices(rng, style, first_train, regime, challenge)
    width, field_height, code_row, divider_row, starts = _layout(
        rng, shapes, first_train, regime, challenge)
    k = len(shapes)
    bg_choices = [v for v in range(10) if v != previous_background]
    background = int(bg_choices[int(rng.integers(len(bg_choices)))])
    palette = [v for v in range(10) if v != background]
    codes = [int(v) for v in rng.choice(palette, size=k, replace=False)]
    swatches = [int(v) for v in rng.choice(palette, size=k, replace=True)]
    if k > 1 and rng.random() < .28:
        swatches[1] = swatches[0]
    if rng.random() < .30:
        j = int(rng.integers(k))
        swatches[j] = codes[j]
    elif k > 1 and rng.random() < .24:
        j = int(rng.integers(k))
        swatches[j] = codes[(j + 1) % k]
    if role_bias == "none":
        divider_choices = [value for value in palette if value not in codes]
        divider = int(divider_choices[int(rng.integers(len(divider_choices)))])
    else:
        code_chance = (.42 if regime == "target_test" else .29)
        if role_bias == "frequent":
            code_chance = .68
        mode = rng.choice(3, p=(code_chance, (1-code_chance)*.55,
                                 (1-code_chance)*.45))
        if mode == 0:
            divider = codes[int(rng.integers(k))]
        elif mode == 1:
            divider = swatches[int(rng.integers(k))]
        else:
            others = [v for v in palette if v not in codes and v not in swatches]
            divider = int(others[int(rng.integers(len(others)))]) if others else int(rng.choice(palette))

    max_h = max(_SHAPES[name].shape[0] for name in shapes)
    max_w = max(_SHAPES[name].shape[1] for name in shapes)
    capacity = max(1, (field_height // max_h) * (width // max_w))
    minimum = 5 if first_train and len(style) > 1 else (4 if first_train else 1)
    count = int(rng.integers(minimum, min(17, max(capacity, minimum) + 1)))
    if first_train:
        kinds = list(range(k)) + [int(rng.integers(k)) for _ in range(count - k)]
    else:
        kinds = [int(rng.integers(k)) for _ in range(count)]
    rng.shuffle(kinds)
    spacing = "touch" if rng.random() < .46 else "separate"
    positions = _positions(rng, kinds, shapes, field_height, width, spacing)

    canonical = np.full((divider_row + 1 + field_height, width),
                        background, dtype=np.int64)
    for name, start, code, swatch in zip(shapes, starts, codes, swatches):
        mask = _SHAPES[name]
        h, w = mask.shape
        bottom = code_row - int(rng.integers(2, 4))
        top = bottom - h + 1
        # The shared legend envelope leaves room for every prototype.
        if top < 0:
            top = 0
        block = canonical[top:top+h, start:start+w]
        block[mask] = swatch
        canonical[code_row, start + w // 2] = code
    canonical[divider_row, :] = divider
    field = canonical[divider_row+1:, :]
    for (r, c), kind in zip(positions, kinds):
        field[r, c] = codes[kind]
    result = field.copy()
    for (r, c), kind in zip(positions, kinds):
        mask = _SHAPES[shapes[kind]]
        h, w = mask.shape
        block = result[r-h//2:r+h//2+1, c-w//2:c+w//2+1]
        block[mask] = swatches[kind]
    if first_train:
        rotation = 0
    elif regime == "source_train" and pair_index == 1:
        rotation = int(rng.integers(1, 4))
    elif regime == "target_test" and challenge == "orientation":
        rotation = int(rng.integers(1, 4))
    else:
        rotation = int(rng.integers(4))
    return {"input": np.rot90(canonical, rotation).copy(),
            "output": np.rot90(result, rotation).copy()}


def _components(grid, background):
    cells = set(map(tuple, np.argwhere(grid != background)))
    groups = []
    while cells:
        start = min(cells)
        cells.remove(start)
        group = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for p in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)):
                if p in cells:
                    cells.remove(p)
                    group.add(p)
                    stack.append(p)
        groups.append(group)
    return groups


def _parse_top(input_grid):
    background = int(np.bincount(input_grid.ravel(), minlength=10).argmax())
    full = [r for r in range(input_grid.shape[0])
            if input_grid[r, 0] != background and
            np.all(input_grid[r] == input_grid[r, 0])]
    assert len(full) == 1
    divider_row = full[0]
    assert 5 <= divider_row <= input_grid.shape[0] - 6
    legend = input_grid[:divider_row]
    field = input_grid[divider_row+1:]
    occupied_rows = np.flatnonzero(np.any(legend != background, axis=1))
    assert len(occupied_rows)
    code_row = int(occupied_rows[-1])
    assert code_row >= 4 and np.all(legend[code_row-1] == background)
    assert np.all(legend[code_row+1:] == background)
    code_cols = np.flatnonzero(legend[code_row] != background)
    assert 1 <= len(code_cols) <= 5
    codes = [int(legend[code_row, c]) for c in code_cols]
    assert len(set(codes)) == len(codes)
    assert all(code != background for code in codes)
    swatch_region = legend[:code_row-1]
    groups = _components(swatch_region, background)
    assert len(groups) == len(codes)
    prototypes = {}
    colors = []
    names = []
    boxes = []
    for c, code in zip(code_cols, codes):
        matching = [group for group in groups
                    if (min(p[1] for p in group) + max(p[1] for p in group)) // 2 == c]
        assert len(matching) == 1
        group = matching[0]
        rows = [p[0] for p in group]
        cols = [p[1] for p in group]
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        tile = swatch_region[r0:r1+1, c0:c1+1]
        mask = tile != background
        name = next((n for n, known in _SHAPES.items()
                     if known.shape == mask.shape and np.array_equal(known, mask)), None)
        assert name is not None
        color_values = set(int(v) for v in tile[mask])
        assert len(color_values) == 1
        color = color_values.pop()
        assert len(group) == int(mask.sum())
        prototypes[code] = (mask, color)
        colors.append(color)
        names.append(name)
        boxes.append((r0, r1, c0, c1))
    marker_array = np.argwhere(field != background)
    assert len(marker_array) >= 1
    expected = field.copy()
    positions = []
    marker_values = []
    rectangles = []
    for row, col in marker_array:
        r, c = int(row), int(col)
        code = int(field[r, c])
        assert code in prototypes
        mask, color = prototypes[code]
        h, w = mask.shape
        r0, r1 = r-h//2, r+h//2+1
        c0, c1 = c-w//2, c+w//2+1
        assert 0 <= r0 and r1 <= field.shape[0]
        assert 0 <= c0 and c1 <= field.shape[1]
        for a0, a1, b0, b1 in rectangles:
            assert r1 <= a0 or a1 <= r0 or c1 <= b0 or b1 <= c0
        rectangles.append((r0, r1, c0, c1))
        expected[r0:r1, c0:c1][mask] = color
        positions.append((r, c))
        marker_values.append(code)
    contacts = 0
    for i, (r0, r1, c0, c1) in enumerate(rectangles):
        for a0, a1, b0, b1 in rectangles[i+1:]:
            if ((r1 == a0 or a1 == r0) and max(c0,b0) < min(c1,b1)) or (
                (c1 == b0 or b1 == c0) and max(r0,a0) < min(r1,a1)):
                contacts += 1
    return expected, {
        "background": background, "divider": int(input_grid[divider_row, 0]),
        "codes": tuple(codes), "swatches": tuple(colors),
        "shape_names": tuple(names), "field_shape": field.shape,
        "input_shape": input_grid.shape, "divider_row": divider_row,
        "code_row": code_row, "positions": tuple(positions),
        "marker_values": tuple(marker_values), "marker_count": len(positions),
        "contacts": contacts, "entries": len(codes),
        "divider_gap": divider_row - code_row - 1,
        "swatch_code_gap": code_row - max(box[1] for box in boxes) - 1,
        "horizontal_gaps": tuple(boxes[i+1][2] - boxes[i][3] - 1
                                 for i in range(len(boxes)-1)),
        "outer_margins": (min(box[0] for box in boxes),
                          min(box[2] for box in boxes),
                          input_grid.shape[1] - max(box[3] for box in boxes) - 1),
    }


def _read_pair(pair):
    found = []
    for turn in range(4):
        try:
            expected, info = _parse_top(np.rot90(pair["input"], turn))
            found.append((turn, expected, info))
        except (AssertionError, ValueError, IndexError):
            pass
    assert len(found) == 1
    turn, expected, info = found[0]
    assert np.array_equal(expected, np.rot90(pair["output"], turn))
    info["orientation"] = (-turn) % 4
    return info


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs = set()
    pairs = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            insig = _array_signature(pair["input"])
            signature = (insig, _array_signature(pair["output"]))
            assert insig not in inputs and signature not in pairs
            inputs.add(insig)
            pairs.add(signature)
            assert _rows_signature(pair["input"]) not in _SOURCE_INPUT_SIGNATURES
            infos[split].append(_read_pair(pair))
    train_shapes = {name for info in infos["train"] for name in info["shape_names"]}
    test_shapes = {name for info in infos["test"] for name in info["shape_names"]}
    assert test_shapes <= train_shapes
    if test_shapes != {"square3"}:
        assert any(info["entries"] > 1 and info["marker_count"] > 1
                   for info in infos["train"])


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    if int(rng.integers(4)) == 0:
        style = ("square3",)
    else:
        style = ("square3", "square5",
                 ("wide", "tall")[int(rng.integers(2))],
                 ("cross", "diamond")[int(rng.integers(2))])
    role_bias = ("none", "mixed", "mixed", "mixed", "frequent")[
        int(rng.integers(5))]
    challenge = ("entries", "field", "orientation", "familiar")[
        int(rng.integers(4))]
    puzzle = {"train": [], "test": []}
    previous_background = None
    seen = set()
    for split, regime, count in (("train", "source_train", train_count),
                                 ("test", "target_test", test_count)):
        for index in range(count):
            for _ in range(30):
                pair = _generate_pair(rng, regime, style, role_bias,
                                      challenge, index, previous_background)
                sig = _array_signature(pair["input"])
                if sig not in seen and _rows_signature(pair["input"]) not in _SOURCE_INPUT_SIGNATURES:
                    seen.add(sig)
                    puzzle[split].append(pair)
                    previous_background = _read_pair(pair)["background"]
                    break
            else:
                raise RuntimeError("could not construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_signature(puzzle):
    return tuple((_array_signature(pair["input"]), _array_signature(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test() -> None:
    counts = (set(), set())
    signatures = set()
    feature_sets = {name: set() for name in ("background", "entries", "marker_count",
                                             "field_shape", "input_shape", "orientation",
                                             "contacts", "divider_role", "shape_names",
                                             "divider_gap", "swatch_code_gap",
                                             "horizontal_gap", "outer_margin",
                                             "legend_height", "entry_marker_combo")}
    within = {"background": 0, "entries": 0, "marker_count": 0,
              "field_shape": 0, "orientation": 0,
              "divider_gap": 0}
    unused = repeated = own_reuse = duplicate_swatch = isolated = touching = 0
    other_reuse = partial = border_flush = same_color_merge = 0
    train_code_reuse = test_code_reuse = both_splits_reuse = neither_split_reuse = 0
    test_more_entries = test_larger_field = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _puzzle_signature(puzzle) == _puzzle_signature(generate_puzzle(seed))
        signatures.add(_puzzle_signature(puzzle))
        counts[0].add(len(puzzle["train"]))
        counts[1].add(len(puzzle["test"]))
        infos = [_read_pair(pair) for split in ("train", "test") for pair in puzzle[split]]
        ntrain = len(puzzle["train"])
        train_reuses = any(info["divider"] in info["codes"] for info in infos[:ntrain])
        test_reuses = any(info["divider"] in info["codes"] for info in infos[ntrain:])
        test_more_entries += (max(info["entries"] for info in infos[ntrain:]) >
                              max(info["entries"] for info in infos[:ntrain]))
        test_larger_field += (
            max(np.prod(info["field_shape"]) for info in infos[ntrain:]) >
            max(np.prod(info["field_shape"]) for info in infos[:ntrain]))
        train_code_reuse += train_reuses
        test_code_reuse += test_reuses
        both_splits_reuse += train_reuses and test_reuses
        neither_split_reuse += not train_reuses and not test_reuses
        for key in within:
            within[key] += len({info[key] for info in infos}) > 1
        for info in infos:
            role = (info["divider"] in info["codes"],
                    info["divider"] in info["swatches"])
            for key, value in (("background", info["background"]),
                               ("entries", info["entries"]),
                               ("marker_count", info["marker_count"]),
                               ("field_shape", info["field_shape"]),
                               ("input_shape", info["input_shape"]),
                               ("orientation", info["orientation"]),
                               ("contacts", info["contacts"]),
                               ("divider_role", role),
                               ("shape_names", info["shape_names"]),
                               ("divider_gap", info["divider_gap"]),
                               ("swatch_code_gap", info["swatch_code_gap"]),
                               ("outer_margin", info["outer_margins"]),
                               ("legend_height", info["divider_row"]),
                               ("entry_marker_combo",
                                (info["entries"], info["marker_count"]))):
                feature_sets[key].add(value)
            feature_sets["horizontal_gap"].update(info["horizontal_gaps"])
            unused += len(set(info["codes"]) - set(info["marker_values"])) > 0
            repeated += len(info["marker_values"]) > len(set(info["marker_values"]))
            own_reuse += any(a == b for a, b in zip(info["codes"], info["swatches"]))
            other_reuse += any(a == info["swatches"][j]
                               for i, a in enumerate(info["codes"])
                               for j in range(info["entries"]) if i != j)
            duplicate_swatch += len(set(info["swatches"])) < info["entries"]
            isolated += info["contacts"] == 0
            touching += info["contacts"] > 0
            prototypes = {code: (_SHAPES[name].shape, color, name)
                          for code, name, color in zip(info["codes"],
                                                       info["shape_names"],
                                                       info["swatches"])}
            stamped = []
            for (r, c), code in zip(info["positions"], info["marker_values"]):
                (h, w), color, name = prototypes[code]
                box = (r-h//2, r+h//2+1, c-w//2, c+w//2+1, color, name)
                stamped.append(box)
                border_flush += (box[0] == 0 or box[2] == 0 or
                                 box[1] == info["field_shape"][0] or
                                 box[3] == info["field_shape"][1])
            for i, (r0, r1, c0, c1, color, name) in enumerate(stamped):
                for a0, a1, b0, b1, other_color, other_name in stamped[i+1:]:
                    overlap = side = 0
                    if r1 == a0 or a1 == r0:
                        overlap = max(0, min(c1, b1) - max(c0, b0))
                        side = min(c1-c0, b1-b0)
                    if c1 == b0 or b1 == c0:
                        overlap = max(0, min(r1, a1) - max(r0, a0))
                        side = min(r1-r0, a1-a0)
                    if overlap:
                        partial += overlap < side
                        same_color_merge += (color == other_color and
                                             name.startswith("square") and
                                             other_name.startswith("square"))
    assert len(signatures) == 42
    assert len(counts[0]) >= 4 and len(counts[1]) >= 3
    assert len(feature_sets["field_shape"]) >= 20
    assert len(feature_sets["input_shape"]) >= 20
    assert len(feature_sets["orientation"]) == 4
    assert feature_sets["entries"] == {1, 2, 3, 4, 5}
    assert min(feature_sets["marker_count"]) == 1 and max(feature_sets["marker_count"]) >= 10
    assert len(feature_sets["background"]) == 10
    assert set(name for group in feature_sets["shape_names"] for name in group) == set(_SHAPES)
    assert len(feature_sets["legend_height"]) >= 5
    assert len(feature_sets["entry_marker_combo"]) >= 25
    assert len(feature_sets["divider_gap"]) >= 2
    assert len(feature_sets["swatch_code_gap"]) >= 2
    assert len(feature_sets["horizontal_gap"]) >= 3
    assert len(feature_sets["outer_margin"]) >= 20
    assert all(value >= 25 for value in within.values())
    assert min(unused, repeated, own_reuse, other_reuse, duplicate_swatch,
               isolated, touching, partial, border_flush,
               same_color_merge) >= 10
    assert min(train_code_reuse, test_code_reuse, both_splits_reuse,
               neither_split_reuse) >= 3
    assert test_more_entries >= 5 and test_larger_field >= 10
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1),
                        (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(12345, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for train, test in ((1, 1), (11, 1), (2, 0), (6, 5), (2, 2)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count request was accepted")
