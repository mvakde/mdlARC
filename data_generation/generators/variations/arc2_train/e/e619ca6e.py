"""Solid rectangles emit two dimension-strided, downward diagonal copy rays."""

import numpy as np


# Exact original inputs, represented by their canvas and filled components.
# Only an identical 0/3 input can duplicate an original solved pair.
_SOURCE_INPUTS = {
    ((25, 22), ((9, 6, 2, 5),)),
    ((29, 29), ((3, 4, 2, 3), (6, 21, 3, 3), (13, 10, 4, 2))),
    ((29, 27), ((5, 8, 2, 3), (13, 16, 4, 2))),
    ((29, 28), ((8, 6, 4, 2), (19, 15, 2, 4))),
}


def _counts(seed, rng, num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                    ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _box(mask, rect, value=True):
    row, col, height, width = rect
    grid_height, grid_width = mask.shape
    top, bottom = max(0, row), min(grid_height, row + height)
    left, right = max(0, col), min(grid_width, col + width)
    if top < bottom and left < right:
        mask[top:bottom, left:right] = value


def _rays(shape, rect):
    row, col, height, width = rect
    left = np.zeros(shape, dtype=bool)
    right = np.zeros(shape, dtype=bool)
    for step in range(1, (shape[0] - 1 - row) // height + 1):
        _box(left, (row + step * height, col - step * width, height, width))
        _box(right, (row + step * height, col + step * width, height, width))
    return left, right


def _render(input_grid, rectangles, foreground):
    output = input_grid.copy()
    for rect in rectangles:
        left, right = _rays(output.shape, rect)
        output[left | right] = foreground
    return output


def _separated(first, second):
    r1, c1, h1, w1 = first
    r2, c2, h2, w2 = second
    return (r1 + h1 + 1 <= r2 or r2 + h2 + 1 <= r1
            or c1 + w1 + 1 <= c2 or c2 + w2 + 1 <= c1)


def _aspect(rect):
    height, width = rect[2:]
    return "H" if width > height else "V" if height > width else "S"


def _dimensions(rng, aspect, height, width):
    cap_h, cap_w = min(8, max(1, height // 3)), min(8, max(1, width // 3))
    if aspect == "S":
        size = int(rng.integers(1, min(cap_h, cap_w, 6) + 1))
        return size, size
    if aspect == "H":
        broad = int(rng.integers(2, cap_w + 1))
        narrow = int(rng.integers(1, min(broad - 1, cap_h) + 1))
        return narrow, broad
    tall = int(rng.integers(2, cap_h + 1))
    narrow = int(rng.integers(1, min(tall - 1, cap_w) + 1))
    return tall, narrow


def _source_mask(shape, rect):
    mask = np.zeros(shape, dtype=bool)
    _box(mask, rect)
    return mask


def _layout_features(shape, rectangles):
    sources = [_source_mask(shape, rect) for rect in rectangles]
    arms = [_rays(shape, rect) for rect in rectangles]
    whole = [sources[i] | arms[i][0] | arms[i][1]
             for i in range(len(rectangles))]
    unique = []
    for i, (left, right) in enumerate(arms):
        other = np.zeros(shape, dtype=bool)
        for j, mask in enumerate(whole):
            if i != j:
                other |= mask
        unique.append((int(np.count_nonzero(left & ~other & ~sources[i])),
                       int(np.count_nonzero(right & ~other & ~sources[i]))))
    intersections = sum(int(np.count_nonzero(whole[i] & whole[j]))
                        for i in range(len(whole)) for j in range(i + 1, len(whole)))
    source_crossing = any(np.any((arms[i][0] | arms[i][1]) & sources[j])
                          for i in range(len(whole)) for j in range(len(whole)) if i != j)
    boundaries = set()
    partial = 0
    for row, col, h, w in rectangles:
        for step in range(1, (shape[0] - 1 - row) // h + 1):
            rr = row + step * h
            for direction in (-1, 1):
                cc = col + direction * step * w
                if cc >= shape[1] or cc + w <= 0:
                    continue
                clipped = set()
                if cc < 0:
                    clipped.add("left")
                if cc + w > shape[1]:
                    clipped.add("right")
                if rr + h > shape[0]:
                    clipped.add("bottom")
                if clipped:
                    partial += 1
                    boundaries |= clipped
    return {"unique": unique, "intersection": intersections,
            "source_crossing": source_crossing, "boundaries": boundaries,
            "partial": partial}


def _geometry_ok(shape, rectangles, independent=False):
    height, width = shape
    if not 1 <= height <= 30 or not 1 <= width <= 30 or not rectangles:
        return False
    if sum(r[2] * r[3] for r in rectangles) * 3 >= height * width:
        return False
    for i, (row, col, h, w) in enumerate(rectangles):
        if min(row, col) < 0 or min(h, w) < 1 or row + h > height or col + w > width:
            return False
        if any(not _separated(rectangles[i], other) for other in rectangles[i + 1:]):
            return False
    features = _layout_features(shape, rectangles)
    if sum(sum(x) for x in features["unique"]) == 0:
        return False
    if independent and any(sum(x) == 0 for x in features["unique"]):
        return False
    return True


def _witness(rng, aspect):
    if aspect == "S":
        side = int(rng.integers(2, 4))
        h, w = side, side
    elif aspect == "H":
        h, w = [(1, 2), (1, 3), (2, 3), (2, 4)][int(rng.integers(4))]
    else:
        h, w = [(2, 1), (3, 1), (3, 2), (4, 2)][int(rng.integers(4))]
    height = int(rng.integers(max(15, 4 * h + 2), 31))
    width = int(rng.integers(max(15, 7 * w), 31))
    row = int(rng.integers(0, min(4, height - 4 * h + 1)))
    col = int(rng.integers(3 * w, width - 4 * w + 1))
    return (height, width), [(row, col, h, w)]


def _candidate(rng, count, aspects, split, ordered):
    minimum = 9 if count <= 2 else 13
    height = int(rng.integers(minimum, 31))
    width = int(rng.integers(minimum, 31))
    rectangles = []
    for aspect in aspects:
        h, w = _dimensions(rng, aspect, height, width)
        found = False
        for _ in range(45):
            row = int(rng.integers(0, height - h + 1))
            col = int(rng.integers(0, width - w + 1))
            candidate = (row, col, h, w)
            if all(_separated(candidate, other) for other in rectangles):
                rectangles.append(candidate)
                found = True
                break
        if not found:
            return None
    rectangles.sort()
    if ordered and [_aspect(rect) for rect in rectangles] != list(aspects):
        return None
    shape = (height, width)
    if not _geometry_ok(shape, rectangles, split == "train"):
        return None
    return shape, rectangles


def _relation_score(features, desired):
    overlap = features["intersection"]
    if desired == "separate":
        return -overlap
    if desired == "small":
        return -abs(overlap - 2) if overlap else -40
    if desired == "dense":
        return min(overlap, 35)
    if desired == "source":
        return 20 + min(overlap, 20) if features["source_crossing"] else -20
    return 0


def _sample_layout(rng, count, aspects, split, desired="any", ordered=False):
    best = None
    best_score = -100000
    attempts = 120 if ordered else 80 if count > 1 and desired != "any" else 35
    for _ in range(attempts):
        candidate = _candidate(rng, count, aspects, split, ordered)
        if candidate is None:
            continue
        shape, rectangles = candidate
        features = _layout_features(shape, rectangles)
        score = _relation_score(features, desired)
        if split == "test":
            score += min(features["partial"], 3) * 0.2
        if score > best_score:
            best, best_score = candidate, score
            if desired == "separate" and score >= 0:
                break
            if desired == "small" and features["intersection"] in (1, 2, 3):
                break
            if desired == "source" and features["source_crossing"]:
                break
            if desired == "dense" and features["intersection"] >= 20:
                break
            if desired == "any":
                break
    if best is None:
        # The broad random sampler has an explicit valid fallback for crowded draws.
        shape = (26, 28)
        if ordered and count == 2:
            dims = {"H": (2, 4), "V": (4, 2), "S": (3, 3)}
            h1, w1 = dims[aspects[0]]
            h2, w2 = dims[aspects[1]]
            rectangles = [(2, 5, h1, w1), (11, 19, h2, w2)]
        else:
            rectangles = [(2, 5, 2, 3), (11, 19, 3, 2), (18, 10, 2, 2)][:count]
        assert _geometry_ok(shape, rectangles, split == "train")
        return shape, rectangles
    return best


def _components(grid, foreground):
    height, width = grid.shape
    remaining = grid == foreground
    rectangles = []
    while np.any(remaining):
        row, col = map(int, np.argwhere(remaining)[0])
        stack = [(row, col)]
        remaining[row, col] = False
        cells = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < height and 0 <= cc < width and remaining[rr, cc]:
                    remaining[rr, cc] = False
                    stack.append((rr, cc))
        rows, cols = zip(*cells)
        top, left = min(rows), min(cols)
        h, w = max(rows) + 1 - top, max(cols) + 1 - left
        assert len(cells) == h * w
        rectangles.append((top, left, h, w))
    return sorted(rectangles)


def _make_pair(shape, rectangles, colors):
    background, foreground = colors
    input_grid = np.full(shape, background, dtype=np.int8)
    for rect in rectangles:
        _box(input_grid, rect, foreground)
    return {"input": input_grid,
            "output": _render(input_grid, rectangles, foreground)}


def _palette(rng, base, variable, index, split):
    if not variable:
        return base
    if split == "train" and index == 0:
        return base
    if split == "train" and index == 1:
        return base[::-1] if rng.random() < 0.5 else tuple(map(int, rng.choice(10, 2, replace=False)))
    mode = int(rng.integers(0, 5))
    if mode == 0:
        return base
    if mode == 1:
        return base[::-1]
    return tuple(map(int, rng.choice(10, 2, replace=False)))


def _signature(grid):
    return (grid.shape, grid.tobytes())


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)
    base = tuple(map(int, rng.choice(10, 2, replace=False)))
    variable_palette = bool(rng.random() < 0.72)
    witness_aspect = "HVS"[int(rng.integers(3))]
    complementary = [a for a in "HVS" if a != witness_aspect]
    mixed_order = list(rng.permutation(complementary))

    geometries = [(_witness(rng, witness_aspect), "witness")]
    geometries.append((_sample_layout(rng, 2, mixed_order, "train",
                                      "small" if rng.random() < 0.5 else "dense",
                                      ordered=True), "mixed"))
    for index in range(2, train_count):
        count = int(rng.choice((1, 2, 3), p=(0.22, 0.43, 0.35)))
        if index == 2 and train_count >= 3 and rng.random() < 0.65:
            count = 2
            aspects = mixed_order[::-1]
        else:
            aspects = ["HVS"[int(rng.integers(3))] for _ in range(count)]
        desired = ("separate", "small", "dense", "source", "any")[int(rng.integers(5))]
        geometries.append((_sample_layout(rng, count, aspects, "train", desired,
                                          ordered=index == 2 and count == 2 and aspects == mixed_order[::-1]), "other"))
    rng.shuffle(geometries)

    puzzle = {"train": [], "test": []}
    used = set()
    for split in ("train", "test"):
        count = train_count if split == "train" else test_count
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            if split == "train":
                (shape, rectangles), _ = geometries[index]
            else:
                for _ in range(12):
                    source_count = (2 if test_count >= 2 and index == 0
                                    else int(pair_rng.choice((1, 2, 3), p=(0.27, 0.43, 0.30))))
                    mixed_test = source_count == 2 and (index == 0 and test_count >= 2
                                                         or pair_rng.random() < 0.75)
                    if mixed_test:
                        aspects = mixed_order if pair_rng.random() < 0.5 else mixed_order[::-1]
                    else:
                        aspects = ["HVS"[int(pair_rng.integers(3))] for _ in range(source_count)]
                    desired = ("separate", "small", "dense", "source", "any")[int(pair_rng.integers(5))]
                    shape, rectangles = _sample_layout(pair_rng, source_count, aspects, "test", desired,
                                                      ordered=mixed_test)
                    if (shape, tuple(rectangles)) not in _SOURCE_INPUTS:
                        break
            colors = _palette(pair_rng, base, variable_palette, index, split)
            pair = _make_pair(shape, rectangles, colors)
            signature = _signature(pair["input"])
            if signature in used or (colors == (0, 3) and (shape, tuple(rectangles)) in _SOURCE_INPUTS):
                if split == "train":
                    # Keep the deliberately clear training role, move its canvas by a row.
                    shifted = [(r + 1, c, h, w) for r, c, h, w in rectangles]
                    if _geometry_ok(shape, shifted, True):
                        pair = _make_pair(shape, shifted, colors)
                    else:
                        shape, rectangles = _sample_layout(pair_rng, len(rectangles),
                                                          [_aspect(r) for r in rectangles], "train")
                        pair = _make_pair(shape, rectangles, colors)
                else:
                    shape, rectangles = _sample_layout(pair_rng, source_count, aspects, "test")
                    pair = _make_pair(shape, rectangles, colors)
                signature = _signature(pair["input"])
            assert signature not in used
            used.add(signature)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert type(puzzle) is dict and set(puzzle) == {"train", "test"}
    assert type(puzzle["train"]) is list and type(puzzle["test"]) is list
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    used = set()
    training_aspects = set()
    clear_witness = False
    mixed = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert type(pair) is dict and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            signature = _signature(inp)
            assert signature not in used
            used.add(signature)
            values, counts = np.unique(inp, return_counts=True)
            assert len(values) == 2
            background = int(values[np.argmax(counts)])
            foreground = int(values[np.argmin(counts)])
            assert counts.max() > counts.min()
            rectangles = _components(inp, foreground)
            assert 1 <= len(rectangles) <= 3
            assert _geometry_ok(inp.shape, rectangles, split == "train")
            assert (inp.shape, tuple(rectangles)) not in _SOURCE_INPUTS or (background, foreground) != (0, 3)
            assert np.array_equal(out, _render(inp, rectangles, foreground))
            assert not np.array_equal(inp, out)
            if split == "train":
                training_aspects.update(_aspect(r) for r in rectangles)
                if len(rectangles) >= 2 and len({_aspect(r) for r in rectangles}) >= 2:
                    mixed = True
                if len(rectangles) == 1:
                    rect = rectangles[0]
                    left, right = _rays(inp.shape, rect)
                    row, col, h, w = rect
                    center = out[row + 2 * h:row + 3 * h, col:col + w]
                    if (row + 3 * h <= inp.shape[0] and np.any(left) and np.any(right)
                            and np.all(center == background)
                            and np.any(_source_mask(inp.shape, (row + 2 * h, col - 2 * w, h, w)) & left)
                            and np.any(_source_mask(inp.shape, (row + 2 * h, col + 2 * w, h, w)) & right)):
                        clear_witness = True
    assert training_aspects == {"H", "V", "S"}
    assert clear_witness and mixed


def _equal_puzzles(first, second):
    return all(np.array_equal(a[key], b[key])
               for split in ("train", "test")
               for a, b in zip(first[split], second[split])
               for key in ("input", "output"))


def _independent_prediction(train, input_grid):
    # Use only visible training grids and this input; no generated test output.
    examples = [(p["input"], p["output"]) for p in train]
    for inp, out in examples:
        colors, counts = np.unique(inp, return_counts=True)
        bg = int(colors[np.argmax(counts)])
        fg = int(colors[np.argmin(counts)])
        trial = inp.copy()
        for r, c, h, w in _components(inp, fg):
            step = 1
            while r + step * h < inp.shape[0]:
                for direction in (-1, 1):
                    _box(trial, (r + step * h, c + direction * step * w, h, w), fg)
                step += 1
        assert np.array_equal(trial, out)
    colors, counts = np.unique(input_grid, return_counts=True)
    fg = int(colors[np.argmin(counts)])
    answer = input_grid.copy()
    for r, c, h, w in _components(input_grid, fg):
        for step in range(1, (input_grid.shape[0] - 1 - r) // h + 1):
            for direction in (-1, 1):
                _box(answer, (r + step * h, c + direction * step * w, h, w), fg)
    return answer


def self_test() -> None:
    for args in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                 (2, None), (10, None), (6, 3)):
        puzzle = generate_puzzle(1000 + len(str(args)), *args)
        validate_generated_puzzle(puzzle)
        if args[0] is not None:
            assert len(puzzle["train"]) == args[0]
        if args[1] is not None:
            assert len(puzzle["test"]) == args[1]
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(17, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    first = generate_puzzle(2201, 5, 2)
    assert _equal_puzzles(first, generate_puzzle(2201, 5, 2))
    assert not _equal_puzzles(first, generate_puzzle(2202, 5, 2))
    train_counts, test_counts = set(), set()
    shapes, dimensions, structures, palettes = set(), set(), set(), set()
    aspect_orders, test_counts_of_sources, boundary_sets = set(), set(), set()
    overlap_kinds, edge_starts = set(), set()
    palette_styles, witness_indices = set(), set()
    source_shape_hits, small_canvases, large_canvases = set(), set(), set()
    unit_sources, row_overlaps, test_clips = set(), set(), set()
    train_order_both = test_reuses_order = test_new_order = 0
    one_arm_tests = zero_arm_tests = source_cross_tests = 0
    separate_multi_tests = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        seed_palettes = set()
        split_orders = {"train": set(), "test": set()}
        for split in ("train", "test"):
            for index, pair in enumerate(puzzle[split]):
                inp = pair["input"]
                colors, counts = np.unique(inp, return_counts=True)
                bg = int(colors[np.argmax(counts)])
                fg = int(colors[np.argmin(counts)])
                palettes.add((bg, fg))
                seed_palettes.add((bg, fg))
                rects = _components(inp, fg)
                assert np.array_equal(_independent_prediction(puzzle["train"], inp), pair["output"])
                shapes.add(inp.shape)
                if inp.shape in {item[0] for item in _SOURCE_INPUTS}:
                    source_shape_hits.add(inp.shape)
                if min(inp.shape) <= 13:
                    small_canvases.add(split)
                if min(inp.shape) >= 24:
                    large_canvases.add(split)
                dimensions.update(r[2:] for r in rects)
                if any(r[2:] == (1, 1) for r in rects):
                    unit_sources.add(split)
                if any(max(a[0], b[0]) < min(a[0] + a[2], b[0] + b[2])
                       for i, a in enumerate(rects) for b in rects[i + 1:]):
                    row_overlaps.add(split)
                structures.add((inp.shape, tuple(rects)))
                edge_starts.add(any(r[0] == 0 or r[1] == 0 or r[1] + r[3] == inp.shape[1]
                                    for r in rects))
                features = _layout_features(inp.shape, rects)
                boundary_sets.add(tuple(sorted(features["boundaries"])))
                if len(rects) > 1:
                    n = features["intersection"]
                    overlap_kinds.add("none" if n == 0 else "small" if n <= 4 else "dense")
                    if features["source_crossing"]:
                        overlap_kinds.add("source")
                if split == "test":
                    test_counts_of_sources.add(len(rects))
                    test_clips.add(tuple(sorted(features["boundaries"])))
                    one_arm_tests += any(sum(x > 0 for x in arm) == 1 for arm in features["unique"])
                    zero_arm_tests += any(sum(x > 0 for x in arm) == 0 for arm in features["unique"])
                    source_cross_tests += features["source_crossing"]
                    separate_multi_tests += len(rects) > 1 and features["intersection"] == 0
                elif len(rects) == 1:
                    r, c, h, w = rects[0]
                    if (r + 3 * h <= inp.shape[0] and c - 2 * w >= 0
                            and c + 3 * w <= inp.shape[1]):
                        witness_indices.add(index)
                if len(rects) == 2 and {_aspect(r) for r in rects} == {"H", "V"}:
                    order = tuple(_aspect(r) for r in rects)
                    aspect_orders.add(order)
                    split_orders[split].add(order)
        palette_styles.add("constant" if len(seed_palettes) == 1 else "varied")
        if any((fg, bg) in seed_palettes for bg, fg in seed_palettes):
            palette_styles.add("swapped")
        train_order_both += len(split_orders["train"]) == 2
        test_reuses_order += bool(split_orders["train"] & split_orders["test"])
        test_new_order += bool(split_orders["test"] - split_orders["train"])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 30 and len(dimensions) >= 20 and len(structures) >= 100
    assert len(palettes) >= 20 and test_counts_of_sources == {1, 2, 3}
    assert {("H", "V"), ("V", "H")} <= aspect_orders
    assert len(boundary_sets) >= 4 and edge_starts == {False, True}
    assert {"none", "small", "dense", "source"} <= overlap_kinds
    assert {"constant", "varied", "swapped"} <= palette_styles
    assert train_order_both >= 5 and test_reuses_order >= 5 and test_new_order >= 3
    assert len(witness_indices) >= 4 and len(source_shape_hits) >= 2
    assert small_canvases == large_canvases == unit_sources == row_overlaps == {"train", "test"}
    assert len(test_clips) >= 6 and one_arm_tests > 0 and zero_arm_tests > 0
    assert source_cross_tests > 0 and separate_multi_tests > 0
    assert any(max(h, w) >= 7 for h, w in dimensions)
