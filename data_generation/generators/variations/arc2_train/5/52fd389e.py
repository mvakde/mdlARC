import hashlib

import numpy as np


# Exact source input fingerprints, including the grid dimensions. The output
# is uniquely fixed by the transformation, so an input match is sufficient.
_SOURCE_INPUTS = {
    (25, 25, "e85dc0a7103ea0e2cee09cc965e8a982641f54b0eb8d287184f394dbc5e95540"),
    (25, 25, "e4b71b874cccada5688389a634fd21f59cdd857b36484660b028810abef779e4"),
    (25, 25, "0309b93f12cba1c71286dd3523fe6b79248cb5eda852c7b3df48084484ae23e9"),
    (29, 29, "9561a17119fbdc08a5d72751cd02e7901683d21025d522974ecf287c06f91bd0"),
}


def _count_value(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _pair_counts(rng, num_train, num_test):
    train = _count_value("num_train", num_train, 2, 10)
    test = _count_value("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test + 1), 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _child_rng(rng):
    return np.random.default_rng(int(rng.integers(0, 2**63, dtype=np.int64)))


def _sample_shape(rng, count, object_total):
    limit = 9 if object_total <= 2 else (7 if object_total == 3 else 5)
    if float(rng.random()) < 0.26:
        shared_shapes = ((1, 1), (4, 5), (4, 5), (6, 8), (2, 7), (7, 2))
        possible = [(h, w) for h, w in shared_shapes if h * w >= count]
        inner_h, inner_w = possible[int(rng.integers(len(possible)))]
        return inner_h + 2, inner_w + 2
    for _ in range(24):
        mode = int(rng.integers(0, 4))
        if mode == 0:
            inner_h, inner_w = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        elif mode == 1:
            inner_h, inner_w = int(rng.integers(1, limit + 1)), int(rng.integers(1, limit + 1))
        elif mode == 2:
            inner_h, inner_w = int(rng.integers(1, 4)), int(rng.integers(5, max(7, limit) + 1))
            if int(rng.integers(0, 2)):
                inner_h, inner_w = inner_w, inner_h
        else:
            inner_h, inner_w = int(rng.integers(4, limit + 1)), int(rng.integers(4, limit + 1))
        if inner_h * inner_w >= count:
            return inner_h + 2, inner_w + 2
    return _compact_shape(count)


def _compact_shape(count):
    return {1: (3, 3), 2: (3, 4), 3: (4, 4), 4: (4, 4), 5: (4, 5)}[count]


def _layout(rng, number, fallback=False):
    if fallback:
        return {1: (1, 1), 2: (1, 2), 3: (2, 2), 4: (2, 2)}[number]
    choices = {
        1: ((1, 1),),
        2: ((1, 2), (2, 1), (2, 2)),
        3: ((1, 3), (3, 1), (2, 2), (2, 3)),
        4: ((2, 2), (2, 3), (3, 2), (1, 4), (4, 1)),
    }[number]
    return choices[int(rng.integers(len(choices)))]


def _slots(rng, number, rows, columns, fallback=False):
    if fallback:
        return list(range(number))
    if number == 2 and rows == columns == 2:
        return [0, 3] if int(rng.integers(2)) else [1, 2]
    for _ in range(16):
        result = [int(x) for x in rng.choice(rows * columns, size=number, replace=False)]
        if len({x // columns for x in result}) == rows and len({x % columns for x in result}) == columns:
            return result
    return list(range(number))


def _axis_origins(rng, sizes, total, fallback=False):
    gaps = [0 if fallback else int(rng.integers(0, 3)) for _ in range(len(sizes) - 1)]
    required = sum(sizes) + sum(gaps)
    if required > total:
        return None
    extra = total - required
    if fallback:
        spaces = [0] * (len(sizes) + 1)
        spaces[-1] = extra
    else:
        spaces = [int(x) for x in rng.multinomial(extra, [1 / (len(sizes) + 1)] * (len(sizes) + 1))]
        if extra and float(rng.random()) < 0.36:
            edge = int(rng.integers(2))
            spaces = [0] * (len(sizes) + 1)
            spaces[0 if edge else -1] = extra
    origins = []
    cursor = spaces[0]
    for index, size in enumerate(sizes):
        origins.append(cursor)
        cursor += size
        if index < len(gaps):
            cursor += gaps[index] + spaces[index + 1]
    return origins


def _within_slot(rng, slack, fallback=False):
    if fallback or slack == 0:
        return 0
    draw = float(rng.random())
    if draw < 0.25:
        return 0
    if draw < 0.50:
        return slack
    return int(rng.integers(slack + 1))


def _place(rng, specs, fallback=False):
    number = len(specs)
    rows, columns = _layout(rng, number, fallback)
    slots = _slots(rng, number, rows, columns, fallback)
    ordered = [dict(spec) for spec in specs]
    if not fallback:
        rng.shuffle(ordered)
    row_sizes, column_sizes = [0] * rows, [0] * columns
    for spec, slot in zip(ordered, slots):
        row, column = divmod(slot, columns)
        spec["eh"], spec["ew"] = spec["height"] + 2 * spec["count"], spec["width"] + 2 * spec["count"]
        row_sizes[row] = max(row_sizes[row], spec["eh"])
        column_sizes[column] = max(column_sizes[column], spec["ew"])
        spec["slot"] = (row, column)
    minimum_h = sum(row_sizes) + (rows - 1 if not fallback else 0)
    minimum_w = sum(column_sizes) + (columns - 1 if not fallback else 0)
    if minimum_h > 30 or minimum_w > 30:
        return None
    if fallback:
        height, width = max(9, minimum_h), max(9, minimum_w)
    elif minimum_h <= 25 and minimum_w <= 25 and float(rng.random()) < 0.13:
        height, width = 25, 25
    else:
        height = int(rng.integers(max(9, minimum_h), 31))
        width = int(rng.integers(max(9, minimum_w), 31))
    row_origins = _axis_origins(rng, row_sizes, height, fallback)
    column_origins = _axis_origins(rng, column_sizes, width, fallback)
    if row_origins is None or column_origins is None:
        return None
    for spec in ordered:
        row, column = spec["slot"]
        er0 = row_origins[row] + _within_slot(rng, row_sizes[row] - spec["eh"], fallback)
        ec0 = column_origins[column] + _within_slot(rng, column_sizes[column] - spec["ew"], fallback)
        spec["er0"], spec["ec0"] = er0, ec0
    if not fallback:
        flip_rows, flip_columns = int(rng.integers(2)), int(rng.integers(2))
        for spec in ordered:
            if flip_rows:
                spec["er0"] = height - spec["er0"] - spec["eh"]
            if flip_columns:
                spec["ec0"] = width - spec["ec0"] - spec["ew"]
    for spec in ordered:
        spec["r0"] = spec["er0"] + spec["count"]
        spec["c0"] = spec["ec0"] + spec["count"]
    for index, first in enumerate(ordered):
        for second in ordered[index + 1:]:
            if not _boxes_disjoint(first, second):
                return None
    return height, width, ordered


def _boxes_disjoint(a, b):
    return (
        a["er0"] + a["eh"] <= b["er0"]
        or b["er0"] + b["eh"] <= a["er0"]
        or a["ec0"] + a["ew"] <= b["ec0"]
        or b["ec0"] + b["ew"] <= a["ec0"]
    )


def _boxes_touch(a, b):
    rows_overlap = max(a["er0"], b["er0"]) < min(a["er0"] + a["eh"], b["er0"] + b["eh"])
    cols_overlap = max(a["ec0"], b["ec0"]) < min(a["ec0"] + a["ew"], b["ec0"] + b["ew"])
    return (
        rows_overlap and (a["ec0"] + a["ew"] == b["ec0"] or b["ec0"] + b["ew"] == a["ec0"])
        or cols_overlap and (a["er0"] + a["eh"] == b["er0"] or b["er0"] + b["eh"] == a["er0"])
    )


def _assign_accents(rng, specs, palette, seen_train, test, forced_accent=None,
                    require_distinct=False):
    prior = set(seen_train)
    for index, spec in enumerate(specs):
        candidates = [forced_accent] if index == 0 and forced_accent is not None else list(palette)
        if test and index == 0 and forced_accent is None and prior and float(rng.random()) < 0.45:
            novel = [color for color in candidates if color not in prior]
            if novel:
                candidates = novel
        if index == 1 and specs[0]["count"] != spec["count"] and float(rng.random()) < 0.38:
            candidates = [specs[0]["accent"]]
        if index == 1 and require_distinct:
            candidates = [color for color in palette if color != specs[0]["accent"]]
        rng.shuffle(candidates)
        usable = [
            color for color in candidates
            if all(
                color != earlier["accent"]
                or (spec["count"] != earlier["count"] and not _boxes_touch(spec, earlier))
                for earlier in specs[:index]
            )
        ]
        if not usable:
            usable = [
                color for color in palette
                if all(color != earlier["accent"] or not _boxes_touch(spec, earlier) for earlier in specs[:index])
            ]
        spec["accent"] = int(usable[int(rng.integers(len(usable)))])


def _render_pair(rng, height, width, specs, background, carrier):
    inp = np.full((height, width), background, dtype=np.int8)
    for spec in specs:
        r0, c0, h, w = spec["r0"], spec["c0"], spec["height"], spec["width"]
        inp[r0:r0 + h, c0:c0 + w] = carrier
        choices = rng.choice((h - 2) * (w - 2), size=spec["count"], replace=False)
        for choice in choices:
            row, column = divmod(int(choice), w - 2)
            inp[r0 + 1 + row, c0 + 1 + column] = spec["accent"]
    out = np.full_like(inp, background)
    for spec in specs:
        r0, c0 = spec["er0"], spec["ec0"]
        out[r0:r0 + spec["eh"], c0:c0 + spec["ew"]] = spec["accent"]
    for spec in specs:
        r0, c0, h, w = spec["r0"], spec["c0"], spec["height"], spec["width"]
        out[r0:r0 + h, c0:c0 + w] = inp[r0:r0 + h, c0:c0 + w]
    return {"input": inp, "output": out}


def _make_pair(rng, counts, background, carrier, palette, seen_train=(), test=False,
               forced_shape=None, forced_accent=None, require_distinct=False):
    number = len(counts)
    for attempt in range(65):
        fallback = attempt == 64
        specs = []
        for count in counts:
            shape = _compact_shape(count) if fallback else _sample_shape(rng, count, number)
            specs.append({"count": int(count), "height": shape[0], "width": shape[1]})
        if forced_shape is not None:
            specs[0]["height"], specs[0]["width"] = forced_shape
        if forced_shape is None and not fallback and number >= 2 and counts[0] != counts[1] and float(rng.random()) < 0.22:
            common = _sample_shape(rng, max(counts[:2]), number)
            for spec in specs[:2]:
                spec["height"], spec["width"] = common
        placed = _place(rng, specs, fallback)
        if placed is None:
            continue
        height, width, specs = placed
        _assign_accents(rng, specs, palette, seen_train, test, forced_accent, require_distinct)
        return _render_pair(rng, height, width, specs, background, carrier)
    raise AssertionError("compact placement failed")


def _extra_counts(rng, test, regime):
    number = int(rng.choice([1, 2, 3, 4], p=[0.18, 0.32, 0.32, 0.18] if test else [0.18, 0.39, 0.31, 0.12]))
    maximum = 5 if (test and regime != "familiar") or (not test and regime == "trained_five") else 4
    counts = [int(rng.integers(1, min(3, maximum) + 1)) for _ in range(number)]
    if number <= 3 and float(rng.random()) < (0.48 if test else 0.38):
        counts[int(rng.integers(number))] = int(rng.integers(4, maximum + 1))
    if number == 4 and float(rng.random()) < 0.30:
        counts[int(rng.integers(number))] = maximum
    if number >= 2 and float(rng.random()) < 0.30:
        counts[1] = counts[0]
    return counts


def _fingerprint(grid):
    return grid.shape, grid.astype(np.uint8, copy=False).tobytes()


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _pair_counts(rng, num_train, num_test)
    background = int(rng.integers(10))
    carrier = int(rng.choice([color for color in range(10) if color != background]))
    palette = tuple(color for color in range(10) if color not in (background, carrier))
    regime = ("novel_five", "trained_five", "familiar")[int(rng.choice(3, p=[0.40, 0.35, 0.25]))]
    large = 5 if regime == "trained_five" else 4
    if float(rng.random()) < 0.28:
        mandatory = [[1], [2, 3, large]]
    else:
        mandatory = [[1, 3], [2, large]]
    if int(rng.integers(2)):
        mandatory.reverse()
    shared_shape = ((5, 6), (6, 7), (6, 8), (7, 5))[int(rng.integers(4))]
    puzzle = {"train": [], "test": []}
    seen = set()
    training_accents = set()
    for split, total in (("train", train_count), ("test", test_count)):
        for index in range(total):
            if split == "train" and index < 2:
                counts = mandatory[index]
            else:
                counts = _extra_counts(rng, split == "test", regime)
            if split == "test" and index == 0 and regime == "novel_five":
                counts[int(rng.integers(len(counts)))] = 5
            for _ in range(8):
                pair = _make_pair(
                    _child_rng(rng), counts, background, carrier, palette,
                    training_accents, split == "test",
                    shared_shape if split == "train" and index < 2 else None,
                    min(training_accents) if split == "train" and index == 1 else None,
                    split == "train" and (
                        index == 0 and len(mandatory[1]) == 1
                        or index == 1 and len(training_accents) < 2
                    ),
                )
                key = _fingerprint(pair["input"])
                if key not in seen:
                    break
            else:
                raise AssertionError("could not find a distinct pair")
            seen.add(key)
            puzzle[split].append(pair)
            if split == "train":
                training_accents.update(_describe_pair(pair)["accents"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _components(mask):
    unseen = set(zip(*np.nonzero(mask)))
    components = []
    while unseen:
        start = unseen.pop()
        stack = [start]
        cells = [start]
        while stack:
            row, column = stack.pop()
            for neighbor in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
                    cells.append(neighbor)
        components.append(cells)
    return components


def _inspect_input(inp):
    boundary = np.concatenate((inp[0], inp[-1], inp[1:-1, 0], inp[1:-1, -1]))
    assert len(np.unique(boundary)) == 1
    background = int(boundary[0])
    objects = []
    for cells in _components(inp != background):
        rows, columns = zip(*cells)
        r0, r1, c0, c1 = min(rows), max(rows), min(columns), max(columns)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        assert h >= 3 and w >= 3 and len(cells) == h * w
        block = inp[r0:r1 + 1, c0:c1 + 1]
        rim = np.concatenate((block[0], block[-1], block[1:-1, 0], block[1:-1, -1]))
        assert len(np.unique(rim)) == 1
        carrier = int(rim[0])
        colors = set(int(x) for x in np.unique(block))
        assert len(colors) == 2 and carrier in colors and background not in colors
        accent = (colors - {carrier}).pop()
        count = int(np.count_nonzero(block == accent))
        assert 1 <= count <= 5 and count <= (h - 2) * (w - 2)
        er0, ec0 = r0 - count, c0 - count
        eh, ew = h + 2 * count, w + 2 * count
        assert er0 >= 0 and ec0 >= 0 and er0 + eh <= inp.shape[0] and ec0 + ew <= inp.shape[1]
        objects.append({"r0": r0, "c0": c0, "height": h, "width": w,
                        "er0": er0, "ec0": ec0, "eh": eh, "ew": ew,
                        "carrier": carrier, "accent": accent, "count": count})
    assert objects
    for index, first in enumerate(objects):
        for second in objects[index + 1:]:
            assert _boxes_disjoint(first, second)
            assert first["accent"] != second["accent"] or not _boxes_touch(first, second)
    return background, objects


def _predicted_output(inp, background, objects):
    expected = np.full_like(inp, background)
    for obj in objects:
        expected[obj["er0"]:obj["er0"] + obj["eh"], obj["ec0"]:obj["ec0"] + obj["ew"]] = obj["accent"]
    for obj in objects:
        expected[obj["r0"]:obj["r0"] + obj["height"], obj["c0"]:obj["c0"] + obj["width"]] = inp[
            obj["r0"]:obj["r0"] + obj["height"], obj["c0"]:obj["c0"] + obj["width"]]
    return expected


def _describe_pair(pair):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    background, objects = _inspect_input(inp)
    assert np.array_equal(out, _predicted_output(inp, background, objects))
    assert not np.array_equal(inp, out)
    return {"background": background, "carrier": objects[0]["carrier"],
            "objects": objects, "accents": {obj["accent"] for obj in objects}}


def _assert_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert all(1 <= dimension <= 30 for dimension in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((0 <= grid) & (grid <= 9))


def _predict_from_demonstrations(train_pairs, test_input):
    conventions = []
    for pair in train_pairs:
        description = _describe_pair(pair)
        conventions.append((description["background"], description["carrier"]))
    assert len(set(conventions)) == 1
    background, carrier = conventions[0]
    found_background, objects = _inspect_input(test_input)
    assert found_background == background and all(obj["carrier"] == carrier for obj in objects)
    return _predicted_output(test_input, background, objects)


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
        assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
        assert 2 <= len(puzzle["train"]) <= 10
        assert 1 <= len(puzzle["test"]) <= 4
        assert len(puzzle["train"]) > len(puzzle["test"])
        seen_inputs = set()
        conventions = set()
        train_counts, train_accents = set(), set()
        shape_counts, accent_counts = {}, {}
        descriptions = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert isinstance(pair, dict) and set(pair) == {"input", "output"}
                _assert_grid(pair["input"])
                _assert_grid(pair["output"])
                key = _fingerprint(pair["input"])
                assert key not in seen_inputs
                seen_inputs.add(key)
                source_key = (*pair["input"].shape, hashlib.sha256(key[1]).hexdigest())
                assert source_key not in _SOURCE_INPUTS
                description = _describe_pair(pair)
                descriptions[split].append(description)
                conventions.add((description["background"], description["carrier"]))
                assert all(obj["carrier"] == description["carrier"] for obj in description["objects"])
                assert 1 <= len(description["objects"]) <= 4
                if split == "train":
                    train_counts.update(obj["count"] for obj in description["objects"])
                    train_accents.update(description["accents"])
                    for obj in description["objects"]:
                        shape_counts.setdefault((obj["height"], obj["width"]), set()).add(obj["count"])
                        accent_counts.setdefault(obj["accent"], set()).add(obj["count"])
        assert len(conventions) == 1
        assert {1, 2, 3} <= train_counts and (4 in train_counts or 5 in train_counts)
        assert len(train_accents) >= 2
        assert any(len(counts) > 1 for counts in shape_counts.values())
        assert any(len(counts) > 1 for counts in accent_counts.values())
        for pair in puzzle["test"]:
            assert np.array_equal(_predict_from_demonstrations(puzzle["train"], pair["input"]), pair["output"])
    except AssertionError:
        raise
    except Exception as error:
        raise AssertionError(f"invalid puzzle: {error}") from error


def _puzzles_equal(first, second):
    return all(
        np.array_equal(a[role], b[role])
        for split in ("train", "test")
        for a, b in zip(first[split], second[split])
        for role in ("input", "output")
    ) and all(len(first[split]) == len(second[split]) for split in ("train", "test"))


def self_test() -> None:
    train_lengths, test_lengths, backgrounds, carriers = set(), set(), set(), set()
    train_objects, test_objects, shapes, canvas_shapes = set(), set(), set(), set()
    same_color_different_count = equal_count_different_color = 0
    train_five = novel_five = familiar_test = no_more_test_objects = 0
    sparse_four = full_one = abutting = 0
    rectangular = source_like = staggered = mixed = 0
    touching_side = touching_corner = novel_accent_test = 0
    source_five_shape = source_four_shape = 0
    five_quadrants = set()
    all_inputs = set()
    count_to_shapes = {}
    shape_to_counts = {}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_lengths.add(len(puzzle["train"]))
        test_lengths.add(len(puzzle["test"]))
        train_descriptions = [_describe_pair(pair) for pair in puzzle["train"]]
        test_descriptions = [_describe_pair(pair) for pair in puzzle["test"]]
        backgrounds.add(train_descriptions[0]["background"])
        carriers.add(train_descriptions[0]["carrier"])
        training_counts = {obj["count"] for desc in train_descriptions for obj in desc["objects"]}
        training_accents = {obj["accent"] for desc in train_descriptions for obj in desc["objects"]}
        train_five += 5 in training_counts
        novel_five += 5 not in training_counts and any(
            obj["count"] == 5 for desc in test_descriptions for obj in desc["objects"])
        familiar_test += all(obj["count"] in training_counts for desc in test_descriptions for obj in desc["objects"])
        no_more_test_objects += max(len(d["objects"]) for d in test_descriptions) <= max(
            len(d["objects"]) for d in train_descriptions)
        for split, descriptions in (("train", train_descriptions), ("test", test_descriptions)):
            for pair, desc in zip(puzzle[split], descriptions):
                objects = desc["objects"]
                (train_objects if split == "train" else test_objects).add(len(objects))
                canvas_shapes.add(pair["input"].shape)
                all_inputs.add(_fingerprint(pair["input"]))
                rectangular += pair["input"].shape[0] != pair["input"].shape[1]
                source_like += pair["input"].shape == (25, 25) and len(objects) == 2
                staggered += any(
                    a["er0"] != b["er0"] and a["ec0"] != b["ec0"]
                    for i, a in enumerate(objects) for b in objects[i + 1:]
                )
                mixed += len(objects) >= 3 and (
                    len({obj["er0"] for obj in objects}) > 1
                    and len({obj["ec0"] for obj in objects}) > 1
                )
                novel_accent_test += split == "test" and any(
                    obj["accent"] not in training_accents for obj in objects
                )
                for index, first in enumerate(objects):
                    shapes.add((first["height"], first["width"]))
                    count_to_shapes.setdefault(first["count"], set()).add((first["height"], first["width"]))
                    shape_to_counts.setdefault((first["height"], first["width"]), set()).add(first["count"])
                    area = (first["height"] - 2) * (first["width"] - 2)
                    sparse_four += first["count"] == 4 and area >= 24
                    full_one += first["count"] == 1 and area == 1
                    source_five_shape += first["count"] == 5 and (first["height"], first["width"]) == (6, 7)
                    source_four_shape += first["count"] == 4 and (first["height"], first["width"]) == (8, 10)
                    row_edge = first["er0"] == 0 or first["er0"] + first["eh"] == pair["input"].shape[0]
                    column_edge = first["ec0"] == 0 or first["ec0"] + first["ew"] == pair["input"].shape[1]
                    touching_side += row_edge or column_edge
                    touching_corner += row_edge and column_edge
                    if split == "test" and first["count"] == 5:
                        five_quadrants.add((
                            int(2 * first["er0"] + first["eh"] >= pair["input"].shape[0]),
                            int(2 * first["ec0"] + first["ew"] >= pair["input"].shape[1]),
                        ))
                    for second in objects[index + 1:]:
                        same_color_different_count += first["accent"] == second["accent"] and first["count"] != second["count"]
                        equal_count_different_color += first["count"] == second["count"] and first["accent"] != second["accent"]
                        abutting += _boxes_touch(first, second)
                if split == "test":
                    assert np.array_equal(_predict_from_demonstrations(puzzle["train"], pair["input"]), pair["output"])
    assert len(train_lengths) >= 4 and len(test_lengths) >= 3
    assert train_objects == {1, 2, 3, 4} and test_objects == {1, 2, 3, 4}
    assert 0 in backgrounds and len(backgrounds) >= 8 and len(carriers) >= 8
    assert len(shapes) >= 45 and len(canvas_shapes) >= 90 and len(all_inputs) >= 150
    assert train_five >= 4 and novel_five >= 4 and familiar_test >= 4 and no_more_test_objects >= 4
    assert same_color_different_count >= 4 and equal_count_different_color >= 4
    assert sparse_four >= 2 and full_one >= 2 and abutting >= 1
    assert rectangular >= 50 and source_like >= 2 and staggered >= 30 and mixed >= 10
    assert touching_side >= 30 and touching_corner >= 4 and novel_accent_test >= 4
    assert source_five_shape >= 1 and source_four_shape >= 1 and len(five_quadrants) == 4
    assert any(len(counts) >= 2 for counts in shape_to_counts.values())
    assert all(len(count_to_shapes[count]) >= 5 for count in (1, 2, 3, 4, 5))
    explicit = generate_puzzle(20260827, 7, 3)
    assert _puzzles_equal(explicit, generate_puzzle(20260827, 7, 3))
    assert not _puzzles_equal(explicit, generate_puzzle(20260828, 7, 3))
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1),
                                           (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(3000 + index, train, test)
        validate_generated_puzzle(puzzle)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (2, 3), (3, 4)):
        try:
            generate_puzzle(5000, train, test)
        except ValueError:
            continue
        raise AssertionError("invalid count request was accepted")
