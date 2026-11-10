"""Compositional generators for the asymmetric open-frame ARC rule."""

import numpy as np


# Exact source inputs; retaining its palette, size, and layout category is fine.
_SOURCE_INPUTS = {
    ("0000000000", "0000000000", "0000000000", "0005055000",
     "0005005000", "0005005000", "0005005000", "0005555000",
     "0000000000", "0000000000"),
    ("0000000000", "0000000000", "0000000000", "0055050000",
     "0050050000", "0050050000", "0055550000", "0000000000",
     "0000000000", "0000000000"),
    ("0000000000", "0000000000", "0505500000", "0500500000",
     "0500500000", "0555500000", "0000000000", "0000055050",
     "0000050050", "0000055550"),
    ("0000000000", "0550500000", "0500500000", "0500500000",
     "0555500000", "0000000000", "0000000000", "0000055050",
     "0000050050", "0000055550"),
}
_SOURCE_OUTPUTS = {
    ("0000000000", "0000000000", "0000222222", "0005255000",
     "0005225000", "0005225000", "0005225000", "0005555000",
     "0000000000", "0000000000"),
    ("0000000000", "0000000000", "2222200000", "0055250000",
     "0052250000", "0052250000", "0055550000", "0000000000",
     "0000000000", "0000000000"),
    ("0000000000", "0022222222", "0525500000", "0522500000",
     "0522500000", "0555500000", "2222222200", "0000055250",
     "0000052250", "0000055550"),
    ("2222000000", "0552500000", "0522500000", "0522500000",
     "0555500000", "0000000000", "2222222200", "0000055250",
     "0000052250", "0000055550"),
}
_SIDES = ("top", "right", "bottom", "left")


def _choice(rng, values):
    return values[int(rng.integers(len(values)))]


def _counts(rng, seed, num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is None and num_test is None:
        num_test = 1 + int(seed) % 4
        num_train = num_test + 1 + (int(seed) // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(int(num_test) + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train) - 1) + 1))
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _dimensions(rng, count, forced_sides):
    """Sample each axis separately, with sufficient room for difficult sets."""
    if count == 1 and int(rng.integers(12)) == 0:
        return 4, 4
    if len(forced_sides) >= 3:
        low = 17
    elif count >= 4:
        low = 17
    elif count == 3:
        low = 14
    elif count == 2:
        low = 8
    else:
        low = 4
    scale = int(rng.choice(4, p=[.22, .38, .25, .15]))
    ranges = ((low, max(low, 9)), (max(low, 9), max(low, 20)),
              (max(low, 18), 29), (30, 30))
    a, b = ranges[scale]
    if a > b:
        a = b
    return int(rng.integers(a, b + 1)), int(rng.integers(a, b + 1))


def _object(top, left, depth, side, sign, color):
    height, width = (depth, 4) if side in ("top", "bottom") else (4, depth)
    return {"top": top, "left": left, "height": height, "width": width,
            "side": side, "sign": sign, "color": color}


def _geometry(obj, shape):
    """Return structural cells, gap, cavity, and the full exterior ray."""
    top, left = obj["top"], obj["left"]
    bottom = top + obj["height"] - 1
    right = left + obj["width"] - 1
    side, sign = obj["side"], obj["sign"]
    assert sign in (-1, 1) and side in _SIDES
    assert 0 <= top <= bottom < shape[0] and 0 <= left <= right < shape[1]
    assert min(obj["height"], obj["width"]) >= 3
    cells = ({(top, col) for col in range(left, right + 1)}
             | {(bottom, col) for col in range(left, right + 1)}
             | {(row, left) for row in range(top, bottom + 1)}
             | {(row, right) for row in range(top, bottom + 1)})
    offset = 1 if sign == 1 else 2
    if side == "top":
        gap = (top, left + offset)
        outside = (top - 1, gap[1])
        ray = {(outside[0], col) for col in
               (range(outside[1], shape[1]) if sign == 1
                else range(0, outside[1] + 1))}
    elif side == "bottom":
        gap = (bottom, left + offset)
        outside = (bottom + 1, gap[1])
        ray = {(outside[0], col) for col in
               (range(outside[1], shape[1]) if sign == 1
                else range(0, outside[1] + 1))}
    elif side == "left":
        gap = (top + offset, left)
        outside = (gap[0], left - 1)
        ray = {(row, outside[1]) for row in
               (range(outside[0], shape[0]) if sign == 1
                else range(0, outside[0] + 1))}
    else:
        gap = (top + offset, right)
        outside = (gap[0], right + 1)
        ray = {(row, outside[1]) for row in
               (range(outside[0], shape[0]) if sign == 1
                else range(0, outside[0] + 1))}
    assert 0 <= outside[0] < shape[0] and 0 <= outside[1] < shape[1]
    cells.remove(gap)
    cavity = {(row, col) for row in range(top + 1, bottom)
              for col in range(left + 1, right)}
    assert cavity and gap not in cavity and ray
    return cells, gap, cavity, ray


def _fits(candidate, placed, shape):
    try:
        cells, gap, cavity, ray = _geometry(candidate, shape)
    except AssertionError:
        return False
    body = cells | cavity | {gap}
    for other in placed:
        ocells, ogap, ocavity, oray = _geometry(other, shape)
        obody = ocells | ocavity | {ogap}
        if body & obody or body & oray or obody & ray or ray & oray:
            return False
        # A blank row/column keeps neighboring frames visually separate.
        if not (candidate["top"] > other["top"] + other["height"]
                or other["top"] > candidate["top"] + candidate["height"]
                or candidate["left"] > other["left"] + other["width"]
                or other["left"] > candidate["left"] + candidate["width"]):
            return False
    return True


def _place(rng, shape, specs, colors):
    placed = []
    for side, sign in specs:
        axis = shape[0] if side in ("top", "bottom") else shape[1]
        high = min(11, axis - 1)
        if high < 3:
            return None
        depth = int(rng.integers(3, high + 1))
        color = _choice(rng, colors)
        box_h, box_w = (depth, 4) if side in ("top", "bottom") else (4, depth)
        if box_h > shape[0] or box_w > shape[1]:
            return None
        found = None
        # Enumerating candidate locations gives a bounded fallback and permits
        # vertical bands, horizontal groups, and staggered placements alike.
        positions = [(r, c) for r in range(shape[0] - box_h + 1)
                     for c in range(shape[1] - box_w + 1)]
        for index in rng.permutation(len(positions)):
            top, left = positions[int(index)]
            candidate = _object(top, left, depth, side, sign, color)
            if _fits(candidate, placed, shape):
                found = candidate
                break
        if found is None:
            return None
        placed.append(found)
    return placed


def _render(shape, background, accent, objects):
    inp = np.full(shape, background, dtype=np.int8)
    out = inp.copy()
    for obj in objects:
        cells, gap, cavity, ray = _geometry(obj, shape)
        for row, col in cells:
            inp[row, col] = obj["color"]
            out[row, col] = obj["color"]
        for row, col in cavity | {gap} | ray:
            out[row, col] = accent
    return inp, out


def _grid_key(grid):
    return (grid.shape, tuple(int(v) for v in grid.flat))


def _source_copy(inp, out):
    input_rows = tuple("".join(str(int(v)) for v in row) for row in inp)
    output_rows = tuple("".join(str(int(v)) for v in row) for row in out)
    return input_rows in _SOURCE_INPUTS and output_rows in _SOURCE_OUTPUTS


def _pair(rng, accent, specs, used, fixed_palette=None, fixed_shape=None):
    for attempt in range(18):
        if fixed_palette is None:
            background = _choice(rng, [c for c in range(10) if c != accent])
            available = [c for c in range(10) if c not in (accent, background)]
            # With several frames, sometimes reuse the color and sometimes vary it.
            if len(specs) > 1 and int(rng.integers(3)) == 0:
                available = [_choice(rng, available)]
        else:
            background, structural = fixed_palette
            available = [structural]
        shape = (fixed_shape if fixed_shape is not None else
                 _dimensions(rng, len(specs), [s for s, _ in specs]))
        if attempt >= 10 and fixed_shape is None:
            shape = (30, 30)
        objects = _place(rng, shape, specs, available)
        if objects is None:
            continue
        inp, out = _render(shape, background, accent, objects)
        key = _grid_key(inp)
        if key not in used and not _source_copy(inp, out):
            used.add(key)
            return {"input": inp, "output": out}
    raise ValueError("Could not place an unambiguous unique pair")


def _random_specs(rng, sides, is_test):
    counts = ([1, 2, 3, 4, 5] if is_test else [1, 2, 3, 4])
    probs = ([.17, .27, .28, .20, .08] if is_test
             else [.35, .31, .23, .11])
    count = int(rng.choice(counts, p=probs))
    specs = [(_choice(rng, sides), _choice(rng, (-1, 1)))
             for _ in range(count)]
    return specs


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, seed, num_train, num_test)
    accent = int(rng.integers(10))
    fixed_palette = None
    if int(rng.integers(4)) == 0:
        colors = [c for c in range(10) if c != accent]
        background = _choice(rng, colors)
        fixed_palette = (background, _choice(rng, [c for c in colors
                                                   if c != background]))
    fixed_shape = (30, 30) if int(rng.integers(10)) == 0 else None
    simple_style = int(rng.integers(8)) == 0
    # Orientation is visible locally, but every one used at test is first
    # demonstrated in both gap directions during training.
    max_sides = 2 if train_count == 2 else 4
    side_count = int(rng.choice(range(1, max_sides + 1),
                                p=([.48, .52] if max_sides == 2
                                   else [.35, .30, .23, .12])))
    sides = (("top",) if simple_style else
             tuple(_SIDES[int(i)] for i in rng.choice(4, side_count,
                                                     replace=False)))
    first_signs = {side: _choice(rng, (-1, 1)) for side in sides}
    train_specs = [[(side, first_signs[side]) for side in sides],
                   [(side, -first_signs[side]) for side in sides]]
    for _ in range(train_count - 2):
        train_specs.append([("top", _choice(rng, (-1, 1)))] if simple_style
                           else _random_specs(rng, sides, False))
    rng.shuffle(train_specs)
    test_specs = [([("top", _choice(rng, (-1, 1)))] if simple_style
                   else _random_specs(rng, sides, True)) for _ in range(test_count)]
    used = set()
    puzzle = {"train": [_pair(rng, accent, specs, used, fixed_palette, fixed_shape)
                        for specs in train_specs],
              "test": [_pair(rng, accent, specs, used, fixed_palette, fixed_shape)
                       for specs in test_specs]}
    validate_generated_puzzle(puzzle)
    return puzzle


def _components(grid, background):
    occupied = grid != background
    seen = np.zeros(grid.shape, dtype=bool)
    result = []
    for row0, col0 in np.argwhere(occupied):
        start = (int(row0), int(col0))
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        cells = set()
        while stack:
            row, col = stack.pop()
            cells.add((row, col))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = row + dr, col + dc
                if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] \
                        and occupied[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        result.append(cells)
    return result


def _parse(grid, background):
    components = _components(grid, background)
    assert components
    objects = []
    for cells in components:
        rows, cols = zip(*cells)
        top, left = min(rows), min(cols)
        height, width = max(rows) - top + 1, max(cols) - left + 1
        color = int(grid[top, left])
        assert color != background and all(int(grid[p]) == color for p in cells)
        candidates = []
        for side in _SIDES:
            if ((side in ("top", "bottom") and width != 4)
                    or (side in ("left", "right") and height != 4)):
                continue
            for sign in (-1, 1):
                obj = {"top": top, "left": left, "height": height,
                       "width": width, "side": side, "sign": sign,
                       "color": color}
                try:
                    expected, gap, cavity, ray = _geometry(obj, grid.shape)
                except AssertionError:
                    continue
                if expected == cells and all(int(grid[p]) == background
                                             for p in cavity | {gap}):
                    candidates.append(obj)
        assert len(candidates) == 1
        objects.append(candidates[0])
    all_rays = set()
    for obj in objects:
        _, _, _, ray = _geometry(obj, grid.shape)
        assert not all_rays & ray
        assert all(int(grid[p]) == background for p in ray)
        all_rays |= ray
    for index, obj in enumerate(objects):
        assert _fits(obj, objects[:index] + objects[index + 1:], grid.shape)
    return objects


def _infer_pair(inp, out, known_accent=None):
    assert inp.shape == out.shape
    input_colors = {int(c) for c in np.unique(inp)}
    output_colors = {int(c) for c in np.unique(out)}
    assert len(input_colors) >= 2 and input_colors <= output_colors
    added = output_colors - input_colors
    assert len(added) == 1
    accent = added.pop()
    assert known_accent is None or accent == known_accent
    candidates = []
    for background in input_colors:
        try:
            objects = _parse(inp, background)
            expected = inp.copy()
            for obj in objects:
                _, gap, cavity, ray = _geometry(obj, inp.shape)
                for cell in cavity | {gap} | ray:
                    expected[cell] = accent
            assert np.array_equal(expected, out)
        except AssertionError:
            continue
        candidates.append((background, objects))
    assert len(candidates) == 1
    return accent, candidates[0][0], candidates[0][1]


def _predict(train_pairs, test_input):
    """Use demonstrations and the visible test grid, without test metadata."""
    accents = []
    for pair in train_pairs:
        accents.append(_infer_pair(pair["input"], pair["output"])[0])
    assert len(set(accents)) == 1
    accent = accents[0]
    predictions = []
    for background in (int(c) for c in np.unique(test_input)):
        try:
            objects = _parse(test_input, background)
        except AssertionError:
            continue
        result = test_input.copy()
        for obj in objects:
            _, gap, cavity, ray = _geometry(obj, test_input.shape)
            for cell in cavity | {gap} | ray:
                result[cell] = accent
        predictions.append((result, objects))
    assert len(predictions) == 1
    return predictions[0]


def _check_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert np.issubdtype(grid.dtype, np.integer)
    assert all(1 <= int(v) <= 30 for v in grid.shape)
    assert 0 <= int(grid.min()) and int(grid.max()) <= 9


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    training_forms = set()
    test_forms = set()
    accent = None
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            _check_grid(inp)
            _check_grid(out)
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = _grid_key(inp)
            assert key not in seen and not _source_copy(inp, out)
            seen.add(key)
            pair_accent, _, objects = _infer_pair(inp, out, accent)
            if accent is None:
                accent = pair_accent
            forms = {(obj["side"], obj["sign"]) for obj in objects}
            if split == "train":
                training_forms |= forms
            else:
                test_forms |= forms
                predicted, parsed = _predict(puzzle["train"], inp.copy())
                assert np.array_equal(predicted, out)
                assert len(parsed) == len(objects)
    assert test_forms <= training_forms


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s])
        for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    # The source's lower frame is three cells high, with a one-row cavity.
    source_input = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 5, 0, 5, 5, 0, 0, 0, 0, 0],
                             [0, 5, 0, 0, 5, 0, 0, 0, 0, 0],
                             [0, 5, 0, 0, 5, 0, 0, 0, 0, 0],
                             [0, 5, 5, 5, 5, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 5, 5, 0, 5, 0],
                             [0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
                             [0, 0, 0, 0, 0, 5, 5, 5, 5, 0]], dtype=np.int8)
    source_objects = _parse(source_input, 0)
    assert sorted(obj["height"] for obj in source_objects) == [3, 4]
    reconstructed = source_input.copy()
    for obj in source_objects:
        _, gap, cavity, ray = _geometry(obj, source_input.shape)
        for cell in cavity | {gap} | ray:
            reconstructed[cell] = 2
    expected = ("0000000000", "0022222222", "0525500000", "0522500000",
                "0522500000", "0555500000", "2222222200", "0000055250",
                "0000052250", "0000055550")
    assert tuple("".join(str(int(v)) for v in row) for row in reconstructed) == expected

    for seed, tr, te in ((200, 2, 1), (201, 10, 1), (202, 10, 4),
                         (300, None, 1), (301, None, 4),
                         (302, 2, None), (303, 10, None),
                         (12345, 7, 3)):
        p = generate_puzzle(seed, tr, te)
        validate_generated_puzzle(p)
        assert _equal(p, generate_puzzle(seed, tr, te))
        if tr is not None:
            assert len(p["train"]) == tr
        if te is not None:
            assert len(p["test"]) == te
    for tr, te in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (True, 1)):
        try:
            generate_puzzle(400, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")

    train_counts, test_counts, shapes, sides, signs = set(), set(), set(), set(), set()
    depths, object_counts, backgrounds, frame_colors = set(), set(), set(), set()
    top_rays, large, small, multi_color, horizontal, overlaps = 0, 0, 0, 0, 0, 0
    fingerprints = set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        train_counts.add(len(p["train"]))
        test_counts.add(len(p["test"]))
        fingerprints.add(tuple(_grid_key(pair["input"]) for split in p.values()
                               for pair in split))
        puzzle_backgrounds = set()
        for split in ("train", "test"):
            for pair in p[split]:
                inp, out = pair["input"], pair["output"]
                _, bg, objects = _infer_pair(inp, out)
                puzzle_backgrounds.add(bg)
                backgrounds.add(bg)
                shapes.add(inp.shape)
                object_counts.add(len(objects))
                small += max(inp.shape) <= 8
                large += inp.shape == (30, 30)
                multi_color += len({o["color"] for o in objects}) > 1
                horizontal += any(a["top"] == b["top"] for a in objects
                                  for b in objects if a is not b)
                overlaps += any(a["top"] < b["top"] + b["height"]
                                and b["top"] < a["top"] + a["height"]
                                for a in objects for b in objects if a is not b)
                for obj in objects:
                    sides.add(obj["side"])
                    signs.add(obj["sign"])
                    depths.add(obj["height"] if obj["side"] in ("top", "bottom")
                               else obj["width"])
                    frame_colors.add(obj["color"])
                    top_rays += obj["side"] == "top" and obj["top"] == 1
        assert len(puzzle_backgrounds) >= 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(fingerprints) == 42 and len(shapes) >= 40
    assert sides == set(_SIDES) and signs == {-1, 1}
    assert 3 in depths and max(depths) >= 8
    assert {1, 2, 3} <= object_counts and max(object_counts) >= 4
    assert len(backgrounds) == 10 and len(frame_colors) == 10
    assert top_rays and large and small and multi_color and horizontal and overlaps
    assert not _equal(generate_puzzle(900, 6, 2), generate_puzzle(901, 6, 2))
