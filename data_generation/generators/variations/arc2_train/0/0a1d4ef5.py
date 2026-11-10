"""Geometric compression of solid rectangles in a speckled ARC grid."""

import hashlib

import numpy as np


_SOURCE_INPUT_HASHES = {
    "20a3519d2d49b0d1550a37bcff9214d3d49085f7f3ed623c8bd798bc40813696",
    "785f206c8c151cfeac3cda33a645bfe2ba8aa71ed5cd5307880c7736de15f859",
    "5e43e0714fe2a6fb25a2254755add97480b4111de1cc1f920605fdc0a0f338e8",
    "728b35fa2c721cac24042c56d652557b27008c2cca9ba9472aa3293813dc31a4",
}


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    train = _checked_count("num_train", num_train, 2, 10)
    test = _checked_count("num_test", num_test, 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
    elif train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(test + 1, 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _components(grid):
    """Find complete monochromatic components; no palette role is assumed."""
    height, width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    rectangles = []
    for y in range(height):
        for x in range(width):
            if seen[y, x]:
                continue
            color = int(grid[y, x])
            seen[y, x] = True
            stack = [(y, x)]
            count = 0
            y0 = y1 = y
            x0 = x1 = x
            while stack:
                cy, cx = stack.pop()
                count += 1
                y0 = min(y0, cy)
                y1 = max(y1, cy)
                x0 = min(x0, cx)
                x1 = max(x1, cx)
                for ny, nx in ((cy - 1, cx), (cy + 1, cx),
                               (cy, cx - 1), (cy, cx + 1)):
                    if (0 <= ny < height and 0 <= nx < width
                            and not seen[ny, nx] and grid[ny, nx] == color):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if (y1 - y0 + 1 >= 3 and x1 - x0 + 1 >= 3
                    and count == (y1 - y0 + 1) * (x1 - x0 + 1)):
                rectangles.append((y0, y1 + 1, x0, x1 + 1, color))
    return rectangles


def _groups(rectangles, axis):
    """Cluster spans that share a common scan line, in spatial order."""
    starts = sorted(range(len(rectangles)), key=lambda i: rectangles[i][axis])
    groups = []
    common_end = -1
    for index in starts:
        start = rectangles[index][axis]
        end = rectangles[index][axis + 1]
        if not groups or start >= common_end:
            groups.append([index])
            common_end = end
        else:
            groups[-1].append(index)
            common_end = min(common_end, end)
    return groups


def _recover_tokens(grid):
    """Infer the complete token array using only the input pixels."""
    rectangles = _components(grid)
    if not rectangles:
        raise AssertionError("no solid rectangles")
    row_groups = _groups(rectangles, 0)
    col_groups = _groups(rectangles, 2)
    rows, cols = len(row_groups), len(col_groups)
    if rows > 4 or cols > 4 or rows * cols != len(rectangles):
        raise AssertionError("no complete rectangle array")
    row_of = {index: row for row, group in enumerate(row_groups) for index in group}
    col_of = {index: col for col, group in enumerate(col_groups) for index in group}
    output = np.full((rows, cols), -1, dtype=np.int8)
    for index, rectangle in enumerate(rectangles):
        row, col = row_of[index], col_of[index]
        if output[row, col] != -1:
            raise AssertionError("ambiguous rectangle slot")
        output[row, col] = rectangle[4]
    if np.any(output < 0):
        raise AssertionError("missing rectangle slot")
    return output, rectangles


def _axis(rng, count):
    """Sample independent track sizes, gaps, margins, and canvas extent."""
    for _ in range(40):
        boxes = rng.integers(3, 10, size=count).tolist()
        gaps = rng.integers(1, 5, size=count - 1).tolist()
        span = sum(boxes) + sum(gaps)
        if span <= 30:
            break
    else:
        boxes = [3] * count
        gaps = [1] * (count - 1)
        span = sum(boxes) + sum(gaps)
    if rng.random() < 0.16:
        extent = 30
    else:
        extent = int(rng.integers(max(10, span), 31))
    leading = int(rng.integers(0, extent - span + 1))
    starts = []
    cursor = leading
    for index, box in enumerate(boxes):
        starts.append(cursor)
        cursor += box
        if index < len(gaps):
            cursor += gaps[index]
    return extent, boxes, starts, gaps


def _shape(rng, regime, index, anchor_shape, expansion):
    if regime == "source_train" and index == 0:
        return anchor_shape
    if regime == "target_test" and expansion and rng.random() < 0.65:
        shape = (int(rng.integers(3, 5)), int(rng.integers(2, 5)))
    else:
        values = np.array([1, 2, 3, 4])
        weights = ([0.12, 0.33, 0.38, 0.17] if regime == "source_train"
                   else [0.10, 0.23, 0.31, 0.36])
        shape = (int(rng.choice(values, p=weights)),
                 int(rng.choice(values, p=weights)))
    if regime == "target_test" and shape == anchor_shape:
        shape = (shape[1], shape[0]) if shape[0] != shape[1] else (4, 3)
    return shape


def _palette_and_tokens(rng, shape, anchor, classic):
    rows, cols = shape
    background = int(0 if classic or rng.random() < 0.45 else rng.integers(1, 10))
    available = np.asarray([color for color in range(10) if color != background])
    rng.shuffle(available)
    noise_count = (2 if classic else
                   int(rng.choice([1, 2, 3], p=[0.22, 0.52, 0.26])))
    noise = available[:noise_count]
    payload_pool = available[noise_count:]
    maximum = min(rows * cols, len(payload_pool))
    if anchor:
        color_count = min(maximum, int(rng.integers(3, min(maximum, 6) + 1)))
    else:
        if maximum == 1 or rng.random() < 0.10:
            color_count = 1
        elif rng.random() < 0.38:
            color_count = int(rng.integers(2, min(maximum, 3) + 1))
        else:
            color_count = int(rng.integers(2, maximum + 1))
    payload = payload_pool[:color_count]
    values = np.empty(rows * cols, dtype=np.int8)
    values[:color_count] = payload
    if color_count < values.size:
        values[color_count:] = rng.choice(payload, size=values.size - color_count)
    rng.shuffle(values)
    tokens = values.reshape(shape)
    if anchor:
        tokens[0, 0] = payload[0]
        tokens[0, -1] = payload[1]
        tokens[-1, 0] = payload[2]
    return background, noise, tokens


def _make_rectangles(rng, shape, boxes_y, starts_y, boxes_x, starts_x, tokens):
    uniform = rng.random() < 0.09
    common_height = int(rng.integers(3, min(boxes_y) + 1))
    common_width = int(rng.integers(3, min(boxes_x) + 1))
    rectangles = []
    for row in range(shape[0]):
        for col in range(shape[1]):
            height = common_height if uniform else int(rng.integers(3, boxes_y[row] + 1))
            width = common_width if uniform else int(rng.integers(3, boxes_x[col] + 1))
            # All rectangles in a track contain its center line. Tracks remain
            # separated by at least one cell, even for repeated colors.
            y_anchor = boxes_y[row] // 2
            x_anchor = boxes_x[col] // 2
            y_low = max(0, y_anchor - height + 1)
            y_high = min(y_anchor, boxes_y[row] - height)
            x_low = max(0, x_anchor - width + 1)
            x_high = min(x_anchor, boxes_x[col] - width)
            y0 = starts_y[row] + int(rng.integers(y_low, y_high + 1))
            x0 = starts_x[col] + int(rng.integers(x_low, x_high + 1))
            rectangles.append((y0, y0 + height, x0, x0 + width, int(tokens[row, col])))
    return rectangles


def _noisy_grid(rng, height, width, background, noise, rectangles, regime,
                classic):
    mask = np.zeros((height, width), dtype=bool)
    for y0, y1, x0, x1, _ in rectangles:
        mask[y0:y1, x0:x1] = True
    if classic:
        density = float(rng.uniform(0.42, 0.51))
    elif regime == "source_train":
        density = float(rng.uniform(0.08, 0.53))
    else:
        density = float(rng.uniform(0.16, 0.55))
    grid = np.full((height, width), background, dtype=np.int8)
    noise_mask = (rng.random((height, width)) < density) & ~mask
    weights = rng.dirichlet(np.full(len(noise), float(rng.uniform(0.5, 3.0))))
    grid[noise_mask] = rng.choice(noise, size=int(noise_mask.sum()), p=weights)
    if rng.random() < 0.55:
        locations = np.argwhere(noise_mask)
        if len(locations):
            chosen = rng.choice(len(locations), size=max(1, len(locations) // 12),
                                replace=False)
            for index in chosen:
                y, x = locations[index]
                dy, dx = ((0, 1) if rng.random() < 0.5 else (1, 0))
                ny, nx = int(y + dy), int(x + dx)
                if ny < height and nx < width and not mask[ny, nx]:
                    grid[ny, nx] = grid[y, x]
    for y0, y1, x0, x1, color in rectangles:
        grid[y0:y1, x0:x1] = color
    return grid


def _pair(rng, regime, shape, anchor):
    for _ in range(24):
        classic = (shape[1] == 3 and shape[0] in (2, 3, 4)
                   and rng.random() < 0.28)
        height, boxes_y, starts_y, gaps_y = _axis(rng, shape[0])
        width, boxes_x, starts_x, gaps_x = _axis(rng, shape[1])
        if classic:
            # Keep the source's canvas/noise setting in a subset, while
            # independently sampling its track sizes, gaps, and placement.
            for boxes, starts, gaps in ((boxes_y, starts_y, gaps_y),
                                        (boxes_x, starts_x, gaps_x)):
                span = sum(boxes) + sum(gaps)
                leading = int(rng.integers(0, 31 - span))
                shift = leading - starts[0]
                starts[:] = [value + shift for value in starts]
            height = width = 30
        background, noise, tokens = _palette_and_tokens(rng, shape, anchor,
                                                          classic)
        rectangles = _make_rectangles(rng, shape, boxes_y, starts_y,
                                      boxes_x, starts_x, tokens)
        grid = _noisy_grid(rng, height, width, background, noise,
                           rectangles, regime, classic)
        try:
            prediction, found = _recover_tokens(grid)
        except AssertionError:
            continue
        if len(found) == tokens.size and np.array_equal(prediction, tokens):
            return {"input": grid, "output": tokens.copy()}
    raise RuntimeError("could not construct an unambiguous pair")


def _signature(array):
    contiguous = np.ascontiguousarray(array, dtype=np.int8)
    return (tuple(contiguous.shape), contiguous.tobytes())


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    seen_pairs = set()
    anchor_shape = None
    orientation_shown = False
    for section in ("train", "test"):
        for index, pair in enumerate(puzzle[section]):
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= extent <= 30 for extent in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            assert inp.shape != out.shape
            assert out.shape[0] <= 4 and out.shape[1] <= 4
            key = _signature(inp)
            pair_key = (key, _signature(out))
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            if inp.shape == (30, 30):
                assert hashlib.sha256(key[1]).hexdigest() not in _SOURCE_INPUT_HASHES
            recovered, rectangles = _recover_tokens(inp)
            assert np.array_equal(recovered, out)
            assert len(rectangles) == out.size
            object_mask = np.zeros(inp.shape, dtype=bool)
            for y0, y1, x0, x1, color in rectangles:
                assert y1 - y0 >= 3 and x1 - x0 >= 3
                assert np.all(inp[y0:y1, x0:x1] == color)
                assert not np.any(object_mask[y0:y1, x0:x1])
                object_mask[y0:y1, x0:x1] = True
            assert not np.isin(inp[~object_mask], np.unique(out)).any()
            if section == "train":
                if index == 0:
                    anchor_shape = out.shape
                if (out.shape[0] >= 2 and out.shape[1] >= 2
                        and np.any(out[:, 0] != out[:, -1])
                        and np.any(out[0, :] != out[-1, :])):
                    orientation_shown = True
            else:
                assert out.shape != anchor_shape
    assert orientation_shown


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    # Spatial order and color preservation are invariant for the whole puzzle.
    # Only the distributional preference for a larger test array is sampled here.
    expansion = bool(rng.random() < 0.48)
    anchor_shape = ((2, 3), (3, 2), (3, 3), (2, 2))[int(rng.integers(0, 4))]
    puzzle = {"train": [], "test": []}
    seen = set()
    for section, regime, count in (("train", "source_train", train_count),
                                   ("test", "target_test", test_count)):
        for index in range(count):
            shape = _shape(rng, regime, index, anchor_shape, expansion)
            for _ in range(12):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                pair = _pair(pair_rng, regime, shape,
                             section == "train" and index == 0)
                signature = _signature(pair["input"])
                if signature not in seen:
                    break
            else:
                raise RuntimeError("duplicate pair construction")
            seen.add(signature)
            puzzle[section].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(1000, 1042)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    all_pairs = [pair for puzzle in puzzles
                 for section in ("train", "test") for pair in puzzle[section]]
    assert len({_signature(pair["input"]) for pair in all_pairs}) >= 300
    assert len({pair["input"].shape for pair in all_pairs}) >= 30
    assert len({pair["output"].shape for pair in all_pairs}) >= 10
    assert any(pair["input"].shape == (30, 30) for pair in all_pairs)
    assert any(pair["output"].shape[0] == 1 for pair in all_pairs)
    assert any(pair["output"].shape[1] == 1 for pair in all_pairs)
    assert any(len(np.unique(pair["output"])) == 1 for pair in all_pairs)
    assert any(0 in pair["output"] for pair in all_pairs)
    rectangle_sizes = set()
    outside_palette_sizes = set()
    margins = []
    unequal_gaps = 0
    mixed_test_shapes = 0
    expanding_tests = 0
    role_exchange_puzzles = 0
    for puzzle in puzzles:
        pairs = puzzle["train"] + puzzle["test"]
        assert len({pair["input"].shape for pair in pairs}) >= 2
        assert len({pair["output"].shape for pair in pairs}) >= 2
        mixed_test_shapes += len({pair["output"].shape for pair in puzzle["test"]}) > 1
        expanding_tests += any(pair["output"].size >
                               max(p["output"].size for p in puzzle["train"])
                               for pair in puzzle["test"])
        roles = []
        for pair in pairs:
            inp, out = pair["input"], pair["output"]
            recovered, rectangles = _recover_tokens(inp)
            assert np.array_equal(recovered, out)
            mask = np.zeros(inp.shape, dtype=bool)
            for y0, y1, x0, x1, _ in rectangles:
                mask[y0:y1, x0:x1] = True
                rectangle_sizes.add((y1-y0, x1-x0))
            outside = inp[~mask]
            outside_palette = set(int(v) for v in np.unique(outside))
            payload_palette = set(int(v) for v in np.unique(out))
            assert outside_palette.isdisjoint(payload_palette)
            outside_palette_sizes.add(len(outside_palette))
            roles.append((payload_palette, outside_palette))
            margins.append((min(b[0] for b in rectangles),
                            inp.shape[0] - max(b[1] for b in rectangles),
                            min(b[2] for b in rectangles),
                            inp.shape[1] - max(b[3] for b in rectangles)))
            # Object centers, rather than equally divided canvas bands, define
            # the row and column order. Their gaps may be asymmetric.
            for axis, count in ((0, out.shape[0]), (2, out.shape[1])):
                if count < 3:
                    continue
                ordered = sorted(rectangles, key=lambda b: b[axis] + b[axis+1])
                groups = [ordered[i*(len(rectangles)//count):
                                  (i+1)*(len(rectangles)//count)]
                          for i in range(count)]
                limits = [(min(b[axis] for b in group),
                           max(b[axis+1] for b in group)) for group in groups]
                gaps = [limits[i+1][0]-limits[i][1] for i in range(count-1)]
                unequal_gaps += len(set(gaps)) > 1
        role_exchange_puzzles += any(
            (a[0] & b[1]) or (a[1] & b[0])
            for i, a in enumerate(roles) for b in roles[i+1:])
        validate_generated_puzzle(puzzle)
    assert (3, 3) in rectangle_sizes
    assert any(max(h, w) >= 8 and min(h, w) <= 4
               for h, w in rectangle_sizes)
    assert {2, 3, 4} <= outside_palette_sizes
    assert any(max(margin) >= 7 for margin in margins)
    assert any(max(margin) <= 2 for margin in margins)
    assert unequal_gaps >= 20
    assert mixed_test_shapes >= 10
    assert expanding_tests >= 8
    assert role_exchange_puzzles >= 30
    for train, test in ((2, 1), (10, 1), (10, 4),
                        (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(81000 + (train or 0) * 10 + (test or 0),
                                 train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
        validate_generated_puzzle(puzzle)
    for kwargs in ({"num_train": 1}, {"num_train": 11},
                   {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(82000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    first = generate_puzzle(73129, 7, 3)
    second = generate_puzzle(73129, 7, 3)
    third = generate_puzzle(73130, 7, 3)
    def digest(puzzle):
        return tuple((_signature(pair["input"]), _signature(pair["output"]))
                     for section in ("train", "test") for pair in puzzle[section])
    assert digest(first) == digest(second)
    assert digest(first) != digest(third)
