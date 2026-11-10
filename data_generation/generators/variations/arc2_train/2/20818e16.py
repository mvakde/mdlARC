import numpy as np


def _integer(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _counts(seed, rng, num_train, num_test):
    train = _integer(num_train, "num_train", 2, 10)
    test = _integer(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _box(row, col, height, width):
    return (row, row + height - 1, col, col + width - 1)


def _intersect(a, b):
    return max(a[0], b[0]) <= min(a[1], b[1]) and max(a[2], b[2]) <= min(a[3], b[3])


def _inside(a, b):
    return b[0] <= a[0] and a[1] <= b[1] and b[2] <= a[2] and a[3] <= b[3]


def _overlap_edges(boxes):
    return frozenset(
        (i, j)
        for i in range(len(boxes))
        for j in range(i + 1, len(boxes))
        if _intersect(boxes[i], boxes[j])
    )


def _paint(canvas, background, boxes, colors, order):
    grid = np.full(canvas, background, dtype=np.int8)
    for rank in order:
        r0, r1, c0, c1 = boxes[rank]
        grid[r0:r1 + 1, c0:c1 + 1] = colors[rank]
    return grid


def _recover(grid):
    # The four background corners identify the canvas even when rectangles
    # meet its edges. Each color still needs visible witnesses at all extrema.
    background = int(grid[0, 0])
    assert all(int(grid[r, c]) == background for r, c in
               ((0, 0), (0, -1), (-1, 0), (-1, -1)))
    colors = [int(v) for v in np.unique(grid) if int(v) != background]
    assert 2 <= len(colors) <= 8
    union = np.zeros(grid.shape, dtype=bool)
    rectangles = []
    for color in colors:
        cells = np.argwhere(grid == color)
        r0, c0 = cells.min(axis=0)
        r1, c1 = cells.max(axis=0)
        r0, r1, c0, c1 = map(int, (r0, r1, c0, c1))
        assert not np.any(grid[r0:r1 + 1, c0:c1 + 1] == background)
        union[r0:r1 + 1, c0:c1 + 1] = True
        rectangles.append((r1 - r0 + 1, c1 - c0 + 1, color,
                           (r0, r1, c0, c1), len(cells)))
    assert np.array_equal(union, grid != background)
    rectangles.sort(key=lambda item: item[0] * item[1], reverse=True)
    for larger, smaller in zip(rectangles, rectangles[1:]):
        assert larger[0] >= smaller[0] and larger[1] >= smaller[1]
        assert larger[:2] != smaller[:2]
    return background, rectangles


def _output(rectangles):
    height, width, color = rectangles[0][:3]
    result = np.full((height, width), color, dtype=np.int8)
    for height, width, color, _, _ in rectangles[1:]:
        result[:height, :width] = color
    return result


def _largest_size(rng, shift, regime, train_sizes, anchor, count, strong):
    if shift == "width_gain" and regime == "source_train":
        h, w = int(rng.integers(3, 20)), int(rng.integers(3, 13))
    elif shift == "height_gain" and regime == "source_train":
        h, w = int(rng.integers(3, 13)), int(rng.integers(3, 20))
    elif shift == "width_gain" and regime == "target_test":
        h = int(rng.integers(3, 21))
        w = int(rng.integers(max(4, max(w for _, w in train_sizes) + 1), 23))
    elif shift == "height_gain" and regime == "target_test":
        h = int(rng.integers(max(4, max(h for h, _ in train_sizes) + 1), 23))
        w = int(rng.integers(3, 21))
    elif shift == "matched":
        h = int(np.clip(anchor[0] + rng.integers(-3, 4), 2, 22))
        w = int(np.clip(anchor[1] + rng.integers(-3, 4), 2, 22))
    else:
        shape = int(rng.integers(0, 5))
        if shape == 0:  # small
            h, w = int(rng.integers(2, 7)), int(rng.integers(2, 8))
        elif shape == 1:  # large
            h, w = int(rng.integers(12, 23)), int(rng.integers(12, 23))
        elif shape == 2:  # tall, often thin
            h, w = int(rng.integers(9, 23)), int(rng.integers(2, 9))
        elif shape == 3:  # wide, often thin
            h, w = int(rng.integers(2, 9)), int(rng.integers(9, 23))
        else:
            h, w = int(rng.integers(5, 17)), int(rng.integers(5, 17))
    if strong:
        h, w = max(6, h), max(6, w)
    # A chain needs at least count-1 strict moves in its dimension lattice.
    while h + w < count + 1:
        if h <= w:
            h += 1
        else:
            w += 1
    return h, w


def _dimensions(rng, largest, count, strong):
    dimensions = [largest]
    profile = int(rng.integers(0, 5))  # horizontal, vertical, close, leaps, mixed
    for index in range(1, count):
        previous_h, previous_w = dimensions[-1]
        remaining = count - index - 1
        if index == 1 and strong:
            next_size = (previous_h - 1, previous_w - 1)
            if sum(next_size) >= remaining + 2:
                dimensions.append(next_size)
                continue
        candidates, weights = [], []
        for h in range(1, previous_h + 1):
            for w in range(1, previous_w + 1):
                if (h, w) == (previous_h, previous_w) or h + w < remaining + 2:
                    continue
                dh, dw = previous_h - h, previous_w - w
                delta = dh + dw
                weight = 1.0
                if profile == 0:
                    weight *= 6 if dh == 0 else 1
                elif profile == 1:
                    weight *= 6 if dw == 0 else 1
                elif profile == 2:
                    weight *= 8 if delta <= 2 else 1 / delta
                elif profile == 3:
                    weight *= 1 + delta
                if dh == 0 or dw == 0:
                    weight *= 1.4
                candidates.append((h, w))
                weights.append(weight)
        probabilities = np.asarray(weights, dtype=float)
        probabilities /= probabilities.sum()
        dimensions.append(candidates[int(rng.choice(len(candidates), p=probabilities))])
    return dimensions


def _canvas(rng, largest):
    size = []
    for extent in largest:
        room = 30 - extent
        kind = int(rng.integers(0, 3))
        if kind == 0:
            margin = int(rng.integers(0, min(room, 3) + 1))
        elif kind == 1:
            low, high = min(room, 4), min(room, 11)
            margin = int(rng.integers(low, high + 1))
        else:
            low = min(room, 9)
            margin = int(rng.integers(low, room + 1))
        size.append(extent + margin)
    return tuple(size)


def _protrusion(box, reference):
    vertical = box[0] < reference[0] or box[1] > reference[1]
    horizontal = box[2] < reference[2] or box[3] > reference[3]
    return vertical, horizontal


def _positions(rng, canvas, size, boxes, mode, rank, side):
    height, width = size
    candidates = []
    for row in range(canvas[0] - height + 1):
        for col in range(canvas[1] - width + 1):
            box = _box(row, col, height, width)
            hits = [_intersect(box, old) for old in boxes]
            if mode == "separated":
                okay = not any(hits)
            elif mode in ("one_overlap", "strong"):
                okay = (hits[0] and not any(hits[1:])) if rank == 1 else not any(hits)
                if okay and rank == 1:
                    if mode == "strong":
                        okay = _inside(box, boxes[0])
                    elif side != "any":
                        v, h = _protrusion(box, boxes[0])
                        okay = (side == "contain" and not v and not h or
                                side == "vertical" and v and not h or
                                side == "horizontal" and h and not v or
                                side == "corner" and v and h)
            elif mode == "chain":
                okay = hits[-1]
            elif mode == "two_groups":
                if rank == 1:
                    okay = hits[0]
                elif rank == 2:
                    okay = not any(hits)
                elif rank == 3:
                    okay = hits[2] and not hits[0] and not hits[1]
                else:
                    okay = True
            elif mode == "cluster":
                okay = any(hits)
            else:
                okay = True
            if okay:
                candidates.append(box)
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def _layout(rng, dimensions, colors, background, mode, edge_desired):
    n = len(dimensions)
    for _ in range(75):
        canvas = _canvas(rng, dimensions[0])
        h, w = dimensions[0]
        max_row, max_col = canvas[0] - h, canvas[1] - w
        row = int(rng.integers(max_row + 1))
        col = int(rng.integers(max_col + 1))
        if edge_desired:
            edge = int(rng.integers(4))
            if edge == 0:
                row = 0
            elif edge == 1:
                row = max_row
            elif edge == 2:
                col = 0
            else:
                col = max_col
        boxes = [_box(row, col, h, w)]
        side = ("contain", "vertical", "horizontal", "corner")[int(rng.integers(4))]
        for rank in range(1, n):
            box = _positions(rng, canvas, dimensions[rank], boxes, mode, rank, side)
            if box is None and mode == "one_overlap" and rank == 1:
                box = _positions(rng, canvas, dimensions[rank], boxes, mode, rank, "any")
            if box is None:
                break
            boxes.append(box)
        if len(boxes) != n:
            continue
        if mode == "strong":
            order = [0] + [int(i) for i in rng.permutation(np.arange(2, n))] + [1]
        else:
            order = [int(i) for i in rng.permutation(n)]
        grid = _paint(canvas, background, boxes, colors, order)
        if edge_desired and not (
            np.any(grid[0] != background) or np.any(grid[-1] != background)
            or np.any(grid[:, 0] != background) or np.any(grid[:, -1] != background)
        ):
            continue
        try:
            recovered_background, recovered = _recover(grid)
        except AssertionError:
            continue
        if recovered_background != background:
            continue
        if {item[2]: item[:2] for item in recovered} != dict(zip(colors, dimensions)):
            continue
        if mode == "strong" and not (recovered[0][4] < recovered[1][4]):
            continue
        return grid
    return None


def _choose_count(rng, regime, shift, index, previous):
    if regime == "source_train":
        if index == 1:
            options = [n for n in range(2, 6) if n != previous[0]]
        elif shift == "count_gain":
            options = list(range(2, 6))
        else:
            options = list(range(2, 8))
    elif shift == "count_gain":
        options = [6, 7]
    else:
        options = list(range(2, 8))
    return int(rng.choice(options))


def _generate_pair(rng, regime, shift, index, train_sizes, previous_counts, palette, anchor):
    for attempt in range(160):
        count = _choose_count(rng, regime, shift, index, previous_counts)
        if regime == "source_train" and index == 0:
            mode = ("one_overlap", "chain", "strong")[int(rng.integers(3))]
        elif regime == "source_train" and index == 1:
            mode = "separated"
        elif regime == "target_test" and shift == "overlap_gain":
            mode = ("cluster", "chain", "two_groups")[int(rng.integers(3))]
        else:
            choices = ["separated", "one_overlap", "chain", "cluster", "free", "strong"]
            if count >= 4:
                choices.append("two_groups")
            mode = choices[int(rng.integers(len(choices)))]
        if mode == "two_groups" and count < 4:
            mode = "cluster"
        strong = mode == "strong"
        largest = _largest_size(rng, shift, regime, train_sizes, anchor, count, strong)
        dimensions = _dimensions(rng, largest, count, strong)
        background = int(palette[(3 * (index + (0 if regime == "source_train" else 5))) % 10])
        available = [int(c) for c in palette if int(c) != background]
        colors = [int(c) for c in rng.choice(available, size=count, replace=False)]
        edge_desired = bool(rng.integers(0, 3) == 0)
        grid = _layout(rng, dimensions, colors, background, mode, edge_desired)
        if grid is None:
            continue
        _, rectangles = _recover(grid)
        return {"input": grid, "output": _output(rectangles)}
    raise ValueError(f"failed to construct {regime} pair {index}")


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


_SOURCE = {
    ((16, 15),
     "888882222222228" * 5 + "883332222222228" +
     "883333338888888" * 5 + "888888888855888" * 2 +
     "888888888888888" * 3,
     (6, 9), "553333222" * 2 + "333333222" * 4),
    ((14, 15),
     "444444444444444" + "442222222244444" * 3 +
     "442222111114444" * 3 + "444444111114444" * 2 +
     "433334444444444" * 2 + "444444444444444" * 3,
     (6, 8), "33331222" * 2 + "11111222" * 3 + "22222222"),
    ((15, 15),
     "000000000000000" + "044444000008800" + "044444000000000" +
     "044466666666000" * 3 + "000066666666000" * 3 +
     "003333666666000" * 2 + "003333000000000" +
     "000000000000000" * 3,
     (8, 8), "88334666" + "33334666" * 2 + "44444666" * 2 +
     "66666666" * 3),
    ((16, 16),
     "1111111111111111" + "1122222222221111" * 4 +
     "1122223333221111" * 2 + "1111113333111111" * 2 +
     "1111111111111111" + "1111111188811111" * 2 +
     "1111111111111111" + "1116611111111111" +
     "1111111111111111" * 2,
     (6, 10), "6683222222" + "8883222222" +
     "3333222222" * 2 + "2222222222" * 2),
}


def _source_signature(input_grid, output_grid):
    return (input_grid.shape, "".join(map(str, input_grid.flat)),
            output_grid.shape, "".join(map(str, output_grid.flat)))


def _check_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert np.issubdtype(grid.dtype, np.integer)
    assert 0 <= int(grid.min()) <= int(grid.max()) <= 9


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    train_edges = []
    train_counts = []
    train_backgrounds = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            _check_grid(inp)
            _check_grid(out)
            key = _grid_key(inp)
            pair_key = (key, _grid_key(out))
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            assert _source_signature(inp, out) not in _SOURCE
            background, rectangles = _recover(inp)
            assert np.array_equal(out, _output(rectangles))
            assert out.shape == rectangles[0][:2]
            assert set(int(c) for c in np.unique(out)) == {r[2] for r in rectangles}
            assert background not in out
            assert inp.shape != out.shape or not np.array_equal(inp, out)
            if split == "train":
                train_edges.append(_overlap_edges([r[3] for r in rectangles]))
                train_counts.append(len(rectangles))
                train_backgrounds.append(background)
    # The first two demonstrations expose both occlusion and separation,
    # and distinct backgrounds make their semantic role evident.
    assert train_edges[0] and not train_edges[1]
    assert len(set(train_counts)) >= 2
    assert len(set(train_backgrounds)) >= 2


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(int(seed), rng, num_train, num_test)
    # The only learned convention is the same upper-left stacking in every
    # pair. The palette is a puzzle-level supply, with pair-level role changes.
    palette = [int(c) for c in rng.permutation(10)]
    anchor_color = int(seed) % 10
    palette.remove(anchor_color)
    palette.insert(0, anchor_color)
    shift = ("width_gain", "height_gain", "matched", "count_gain",
             "overlap_gain", "mixed")[int(rng.integers(6))]
    anchor = (int(rng.integers(4, 21)), int(rng.integers(4, 21)))
    puzzle = {"train": [], "test": []}
    seen = set()
    train_sizes, train_counts = [], []
    for split, count, regime in (("train", train_count, "source_train"),
                                 ("test", test_count, "target_test")):
        for index in range(count):
            for retry in range(20):
                pair_seed = int(rng.integers(0, 2**63 - 1))
                pair_rng = np.random.default_rng(pair_seed)
                pair = _generate_pair(pair_rng, regime, shift, index,
                                      train_sizes, train_counts, palette, anchor)
                key = _grid_key(pair["input"])
                if key not in seen:
                    break
            else:
                raise ValueError("failed to make a unique pair")
            seen.add(key)
            puzzle[split].append(pair)
            if split == "train":
                train_sizes.append(pair["output"].shape)
                train_counts.append(len(np.unique(pair["input"])) - 1)
    validate_generated_puzzle(puzzle)
    return puzzle


def _independent_answer(input_grid):
    background = int(input_grid[0, 0])
    footprints = []
    for color in np.unique(input_grid):
        if int(color) == background:
            continue
        rows, cols = np.where(input_grid == color)
        footprints.append((int(rows.max() - rows.min() + 1),
                           int(cols.max() - cols.min() + 1), int(color)))
    footprints.sort(key=lambda x: x[0] * x[1], reverse=True)
    result = np.full(footprints[0][:2], footprints[0][2], dtype=np.int8)
    for h, w, color in footprints[1:]:
        result[:h, :w] = color
    return result


def self_test() -> None:
    a = generate_puzzle(9137, 6, 3)
    b = generate_puzzle(9137, 6, 3)
    assert all(np.array_equal(x[k], y[k])
               for split in ("train", "test")
               for x, y in zip(a[split], b[split])
               for k in ("input", "output"))
    other = generate_puzzle(9138, 6, 3)
    assert _grid_key(a["train"][0]["input"]) != _grid_key(other["train"][0]["input"])
    for seed, train, test in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                              (2001, None, 1), (2002, None, 4),
                              (2003, 2, None), (2004, 10, None)):
        p = generate_puzzle(seed, train, test)
        validate_generated_puzzle(p)
        if train is not None:
            assert len(p["train"]) == train
        if test is not None:
            assert len(p["test"]) == test
    for bad in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                {"num_test": 5}, {"num_train": 2, "num_test": 2},
                {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(3000, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair count accepted")
    train_counts, test_counts = set(), set()
    object_counts, output_shapes, input_shapes, layouts = set(), set(), set(), set()
    edge_pairs = separate_tests = overlap_tests = occluded_tests = 0
    small = large = tall = wide = shared = 0
    successive_shared = close_steps = large_steps = 0
    two_groups = multiple_occluders = multi_edge_tests = 0
    contained_tests = protruding_tests = 0
    front_largest = front_second = 0
    protrusion_sides = set()
    backgrounds = {"train": set(), "test": set()}
    count_varied_puzzles = width_gains = height_gains = heldout_counts = 0
    large_count_small_canvas = small_count_large_canvas = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        local_counts = set()
        train_sizes = [pair["output"].shape for pair in puzzle["train"]]
        test_sizes = [pair["output"].shape for pair in puzzle["test"]]
        width_gains += all(w > max(t[1] for t in train_sizes) for _, w in test_sizes)
        height_gains += all(h > max(t[0] for t in train_sizes) for h, _ in test_sizes)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                assert np.array_equal(out, _independent_answer(inp))
                background, rectangles = _recover(inp)
                backgrounds[split].add(background)
                local_counts.add(len(rectangles))
                object_counts.add(len(rectangles))
                input_shapes.add(inp.shape)
                output_shapes.add(out.shape)
                layouts.add((inp.shape, tuple(r[3] for r in rectangles)))
                if len(rectangles) >= 6 and inp.size <= 400:
                    large_count_small_canvas += 1
                if len(rectangles) == 2 and inp.size >= 400:
                    small_count_large_canvas += 1
                if (np.any(inp[0] != inp[0, 0]) or np.any(inp[-1] != inp[0, 0])
                    or np.any(inp[:, 0] != inp[0, 0]) or np.any(inp[:, -1] != inp[0, 0])):
                    edge_pairs += 1
                if out.size <= 30:
                    small += 1
                if out.size >= 140:
                    large += 1
                if out.shape[0] >= 2 * out.shape[1]:
                    tall += 1
                if out.shape[1] >= 2 * out.shape[0]:
                    wide += 1
                if any(a[0] == b[0] or a[1] == b[1]
                       for a, b in zip(rectangles, rectangles[1:])):
                    shared += 1
                if any(a[0] == b[0] == c[0] or a[1] == b[1] == c[1]
                       for a, b, c in zip(rectangles, rectangles[1:], rectangles[2:])):
                    successive_shared += 1
                close_steps += any(a[0] - b[0] + a[1] - b[1] <= 2
                                   for a, b in zip(rectangles, rectangles[1:]))
                large_steps += any(a[0] - b[0] + a[1] - b[1] >= 6
                                   for a, b in zip(rectangles, rectangles[1:]))
                if split == "test":
                    boxes = [r[3] for r in rectangles]
                    edges = _overlap_edges(boxes)
                    separate_tests += not edges
                    overlap_tests += bool(edges)
                    occluded_tests += any(a[4] < b[4] for a, b in zip(rectangles, rectangles[1:]))
                    multi_edge_tests += len(edges) >= 2
                    two_groups += (len(rectangles) >= 4 and (0, 1) in edges
                                   and (2, 3) in edges and not any(
                                       (i, j) in edges for i in (0, 1) for j in (2, 3)))
                    for rectangle in rectangles:
                        r0, r1, c0, c1 = rectangle[3]
                        others = set(int(c) for c in np.unique(inp[r0:r1 + 1, c0:c1 + 1]))
                        if len(others - {background, rectangle[2]}) >= 2:
                            multiple_occluders += 1
                            break
                    if (0, 1) in edges:
                        first, second = boxes[:2]
                        contained_tests += _inside(second, first)
                        protruding_tests += not _inside(second, first)
                        if second[0] < first[0]:
                            protrusion_sides.add("top")
                        if second[1] > first[1]:
                            protrusion_sides.add("bottom")
                        if second[2] < first[2]:
                            protrusion_sides.add("left")
                        if second[3] > first[3]:
                            protrusion_sides.add("right")
                        region = inp[max(first[0], second[0]):min(first[1], second[1]) + 1,
                                     max(first[2], second[2]):min(first[3], second[3]) + 1]
                        first_visible = np.any(region == rectangles[0][2])
                        second_visible = np.any(region == rectangles[1][2])
                        front_largest += first_visible and not second_visible
                        front_second += second_visible and not first_visible
        count_varied_puzzles += len(local_counts) >= 2
        heldout_counts += max(len(np.unique(p["input"])) - 1 for p in puzzle["test"]) > max(
            len(np.unique(p["input"])) - 1 for p in puzzle["train"])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert object_counts == set(range(2, 8))
    assert len(input_shapes) >= 35 and len(output_shapes) >= 35
    assert len(layouts) >= 100
    assert edge_pairs >= 10 and separate_tests >= 3 and overlap_tests >= 15
    assert occluded_tests >= 3
    assert min(small, large, tall, wide, shared) >= 3
    assert min(successive_shared, close_steps, large_steps) >= 3
    assert min(two_groups, multiple_occluders, multi_edge_tests) >= 3
    assert min(contained_tests, protruding_tests, front_largest, front_second) >= 3
    assert protrusion_sides == {"top", "bottom", "left", "right"}
    assert backgrounds["train"] == backgrounds["test"] == set(range(10))
    assert count_varied_puzzles == 42
    assert min(width_gains, height_gains, heldout_counts) >= 3
    assert min(large_count_small_canvas, small_count_large_canvas) >= 3
