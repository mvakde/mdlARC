import hashlib

import numpy as np


_SOURCE_HASHES = {
    "64ea81e2e1ea913b45050c66f1a4f299f7ccf66191801e6af1dc97ff99792abc",
    "74b1490a1e6b5694ba8a8af3dabfd54f408615cd89e7bfe719cf85196d74d90c",
    "f53f613de1d9c6403ed996da01ec2561872650eb4595c9f9a6c401c2260b83df",
    "13600812e59352d4e24b50a5cbfd87dda7b8b46c186d4b5127cb891fd431e27f",
    "85fc141ea6dfa0cd2d30bb16398ea5c11b21c1bb639e856185cbcec2da073371",
}


def _check_count(value, name, low, high):
    if value is not None and (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not low <= value <= high
    ):
        raise ValueError(f"{name} must be an integer from {low} through {high}")


def _connected(mask):
    points = np.argwhere(mask)
    if len(points) == 0:
        return False
    start = tuple(map(int, points[0]))
    seen, stack = {start}, [start]
    height, width = mask.shape
    while stack:
        y, x = stack.pop()
        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= yy < height and 0 <= xx < width and mask[yy, xx] and (yy, xx) not in seen:
                seen.add((yy, xx))
                stack.append((yy, xx))
    return len(seen) == int(mask.sum())


def _box(mask):
    ys, xs = np.where(mask)
    assert len(ys)
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def _tab(mask, rng):
    top, bottom, left, right = _box(mask)
    side = int(rng.integers(4))
    if side < 2:
        y = top if side == 0 else bottom
        x = int(rng.choice(np.where(mask[y, left:right + 1])[0] + left))
        half, depth = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        lo, hi = max(left, x - half), min(right, x + half)
        if side == 0:
            mask[y - depth:y, lo:hi + 1] = True
        else:
            mask[y + 1:y + depth + 1, lo:hi + 1] = True
    else:
        x = left if side == 2 else right
        y = int(rng.choice(np.where(mask[top:bottom + 1, x])[0] + top))
        half, depth = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        lo, hi = max(top, y - half), min(bottom, y + half)
        if side == 2:
            mask[lo:hi + 1, x - depth:x] = True
        else:
            mask[lo:hi + 1, x + 1:x + depth + 1] = True
    return mask


def _core(rng, height, width, style):
    mask = np.zeros((60, 60), dtype=bool)
    top, left = 30 - height // 2, 30 - width // 2
    mask[top:top + height, left:left + width] = True
    if style == "irregular":
        for corner in rng.permutation(4)[:int(rng.integers(1, 4))]:
            cut_h = int(rng.integers(1, min(4, height)))
            cut_w = int(rng.integers(1, min(4, width)))
            rows = slice(top, top + cut_h) if corner < 2 else slice(top + height - cut_h, top + height)
            cols = slice(left, left + cut_w) if corner % 2 == 0 else slice(left + width - cut_w, left + width)
            mask[rows, cols] = False
        if height >= 5 and width >= 5 and rng.random() < 0.6:
            start = top + int(rng.integers(1, height - 2))
            span = int(rng.integers(2, min(5, top + height - start + 1)))
            depth = int(rng.integers(1, min(3, width - 1)))
            if rng.integers(2):
                mask[start:start + span, left:left + depth] = False
            else:
                mask[start:start + span, left + width - depth:left + width] = False
        if rng.random() < 0.65:
            _tab(mask, rng)
    return mask


def _dilate(mask, up, down, left, right):
    result = np.zeros_like(mask)
    for dy in range(-up, down + 1):
        for dx in range(-left, right + 1):
            result |= np.roll(mask, (dy, dx), axis=(0, 1))
    return result


def _geometry(rng, depth, style, thin):
    for attempt in range(41):
        if attempt == 40:
            height, width = (4, min(12, 27 - 2 * (depth - 1))) if style == "elongated" else ((3, 3) if thin else (4, 5))
            if style == "elongated" and rng.integers(2):
                height, width = width, height
        elif style == "elongated":
            short, long = int(rng.integers(2, 5)), int(rng.integers(8, 16))
            height, width = (long, short) if rng.integers(2) else (short, long)
        elif thin:
            height, width = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        else:
            limit = max(5, 17 - depth)
            height, width = int(rng.integers(2, limit + 1)), int(rng.integers(2, limit + 1))
        if attempt == 40 and style == "irregular":
            core = _core(rng, height, width, "regular")
            y0, _, x0, _ = _box(core)
            core[y0, x0] = False
        else:
            core = _core(rng, height, width, style)
        if style == "irregular":
            cy0, cy1, cx0, cx1 = _box(core)
            if int(core.sum()) == (cy1 - cy0 + 1) * (cx1 - cx0 + 1):
                continue
        masks = [core]
        for _ in range(depth - 1):
            choices = (1, 1, 1, 1, 1, 1, 1, 2) if thin else (1, 1, 1, 1, 2, 2, 3)
            steps = (1, 1, 1, 1) if attempt == 40 else tuple(
                int(v) for v in rng.choice(choices, size=4))
            outer = _dilate(masks[-1], *steps)
            if style == "irregular" and rng.random() < 0.58 and attempt < 40:
                _tab(outer, rng)
            masks.append(outer)
        top, bottom, left, right = _box(masks[-1])
        limit = (16 if depth == 7 else 14) if thin else 27
        if bottom - top + 1 <= limit and right - left + 1 <= limit and all(_connected(mask) for mask in masks):
            return masks
    raise RuntimeError("Geometry construction failed")


def _canvas(rng, outer, regime, thin):
    top, bottom, left, right = _box(outer)
    canvas_limit = 18 if thin else 30
    available = [canvas_limit - (bottom - top + 1), canvas_limit - (right - left + 1)]
    margins = []
    for room in available:
        total = int(rng.integers(2, min(4 if thin else 8, room) + 1))
        before = int(rng.integers(1, total))
        margins.extend((before, total - before))
    if rng.random() < (0.13 if regime == "source_train" else 0.36):
        side = int(rng.integers(4))
        axis, opposite = side // 2, side ^ 1
        margins[side] = 0
        margins[opposite] = int(rng.integers(2, min(6, available[axis]) + 1))
        other, room = (2 if axis == 0 else 0), available[1 - axis]
        margins[other] = int(rng.integers(1, room))
        margins[other + 1] = int(rng.integers(1, room - margins[other] + 1))
    return tuple(margins)


def _colors(rng, depth, previous, relation):
    background = 0 if rng.random() < 0.55 else int(rng.integers(1, 10))
    available = [color for color in range(10) if color != background]
    if previous and relation == "disjoint":
        fresh = [color for color in available if color not in previous]
        if len(fresh) >= depth:
            available = fresh
    elif previous and relation == "reuse":
        shared = [color for color in available if color in previous]
        if len(shared) >= depth:
            available = shared
    return background, tuple(int(v) for v in rng.permutation(available)[:depth])


def _make_pair(rng, regime, depth, style, previous, relation, force_thin):
    thin = force_thin or (depth >= 5 and style != "elongated" and rng.random() < 0.35)
    for _ in range(25):
        masks = _geometry(rng, depth, style, thin)
        top, bottom, left, right = _canvas(rng, masks[-1], regime, thin)
        y0, y1, x0, x1 = _box(masks[-1])
        height, width = y1 - y0 + 1, x1 - x0 + 1
        background, colors = _colors(rng, depth, previous, relation)
        grid = np.full((top + height + bottom, left + width + right), background, dtype=np.int8)
        patch = grid[top:top + height, left:left + width]
        for mask, color in zip(reversed(masks), colors):
            patch[mask[y0:y1 + 1, x0:x1 + 1]] = color
        output = np.asarray([colors], dtype=np.int8)
        try:
            _inspect(grid, output)
        except AssertionError:
            continue
        return {"input": grid, "output": output}
    raise RuntimeError("Could not construct a nested object")


def _source_hash(inp, out):
    parts = [str(inp.shape[0]), str(inp.shape[1])]
    parts.extend("".join(str(int(v)) for v in row) for row in inp)
    parts.extend("".join(str(int(v)) for v in row) for row in out)
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _inspect(inp, out):
    height, width = inp.shape
    background = int(inp[0, 0])
    assert int(inp[0, -1]) == int(inp[-1, 0]) == int(inp[-1, -1]) == background
    colors = {int(v) for v in np.unique(inp)}
    assert 2 <= len(colors) - 1 <= 7 and background in colors
    for color in colors:
        assert _connected(inp == color)
    adjacency = {color: set() for color in colors}
    for a, b in zip(inp[:, :-1].flat, inp[:, 1:].flat):
        a, b = int(a), int(b)
        if a != b:
            adjacency[a].add(b)
            adjacency[b].add(a)
    for a, b in zip(inp[:-1, :].flat, inp[1:, :].flat):
        a, b = int(a), int(b)
        if a != b:
            adjacency[a].add(b)
            adjacency[b].add(a)
    assert len(adjacency[background]) == 1
    order, visited = [], {background}
    previous, current = None, background
    while True:
        choices = adjacency[current] - ({previous} if previous is not None else set())
        assert len(choices) <= 1
        if not choices:
            break
        next_color = next(iter(choices))
        assert next_color not in visited
        visited.add(next_color)
        order.append(next_color)
        previous, current = current, next_color
    assert visited == colors
    assert out.shape == (1, len(order)) and np.array_equal(out[0], order)
    masks = [np.isin(inp, order[index:]) for index in range(len(order))]
    boxes = [_box(mask) for mask in masks]
    for outer, inner in zip(boxes, boxes[1:]):
        assert inner[0] > outer[0] and inner[1] < outer[1]
        assert inner[2] > outer[2] and inner[3] < outer[3]
    border_sides = [np.any(inp[0] != background), np.any(inp[-1] != background),
                    np.any(inp[:, 0] != background), np.any(inp[:, -1] != background)]
    assert sum(border_sides) <= 1
    rectangles = [int(mask.sum()) == (box[1] - box[0] + 1) * (box[3] - box[2] + 1)
                  for mask, box in zip(masks, boxes)]
    outer = boxes[0]
    ratio = max(outer[1] - outer[0] + 1, outer[3] - outer[2] + 1) / min(
        outer[1] - outer[0] + 1, outer[3] - outer[2] + 1)
    return {"order": order, "background": background, "rectangles": rectangles,
            "edge": sum(border_sides) == 1, "ratio": ratio, "shape": (height, width),
            "boxes": boxes, "areas": [int((inp == color).sum()) for color in order]}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    _check_count(num_train, "num_train", 2, 10)
    _check_count(num_test, "num_test", 1, 4)
    legal = [(a, b) for a in range(2, 11) for b in range(1, 5)
             if a > b and (num_train is None or a == num_train)
             and (num_test is None or b == num_test)]
    if not legal:
        raise ValueError("num_train must exceed num_test")
    if num_train is None and num_test is None:
        train_count = int(rng.integers(2, 11))
        test_count = int(rng.integers(1, min(4, train_count - 1) + 1))
    else:
        train_count, test_count = legal[int(rng.integers(len(legal)))]
    mixed = bool(rng.integers(2))
    train_styles = ["regular", "irregular" if rng.integers(2) else "elongated"] if mixed else ["regular", "regular"]
    rng.shuffle(train_styles)
    while len(train_styles) < train_count:
        train_styles.append(str(rng.choice(("regular", "regular", "irregular", "elongated"))))
    train_depths = [int(rng.integers(3, 6)), int(rng.integers(3, 6))]
    if train_depths[0] == train_depths[1]:
        train_depths[1] = 3 + (train_depths[1] - 2) % 3
    train_depths.extend(int(rng.integers(2, 7)) for _ in range(train_count - 2))
    test_styles = [str(rng.choice(("regular", "irregular", "irregular", "elongated")))
                   for _ in range(test_count)]
    test_depths = [int(rng.integers(2, 8)) for _ in range(test_count)]
    test_thin = [False] * test_count
    if rng.random() < 0.35:
        compact_index = int(rng.integers(test_count))
        test_styles[compact_index] = "regular"
        test_depths[compact_index] = 6
        test_thin[compact_index] = True
    if not any(style == "irregular" or depth >= 5 for style, depth in zip(test_styles, test_depths)):
        test_styles[int(rng.integers(test_count))] = "irregular"
    puzzle = {"train": [], "test": []}
    seen, first_colors, train_colors = set(), (), set()
    for split, styles, depths, thin_flags in (("train", train_styles, train_depths, [False] * train_count),
                                              ("test", test_styles, test_depths, test_thin)):
        for index, (style, depth) in enumerate(zip(styles, depths)):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            relation = str(pair_rng.choice(("independent", "disjoint", "reuse")))
            previous = first_colors if split == "train" and index == 1 else train_colors
            for _ in range(20):
                pair = _make_pair(pair_rng, "source_train" if split == "train" else "target_test",
                                  depth, style, previous, relation, thin_flags[index])
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen and _source_hash(pair["input"], pair["output"]) not in _SOURCE_HASHES:
                    break
            else:
                raise RuntimeError("Could not make a distinct pair")
            seen.add(key)
            puzzle[split].append(pair)
            if split == "train":
                train_colors.update(int(v) for v in pair["output"][0])
                if index == 0:
                    first_colors = tuple(int(v) for v in pair["output"][0])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[split], list) for split in ("train", "test"))
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= side <= 30 for side in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            input_key = (inp.shape, inp.tobytes())
            pair_key = (input_key, out.shape, out.tobytes())
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert _source_hash(inp, out) not in _SOURCE_HASHES
            infos[split].append(_inspect(inp, out))
    assert any(all(info["rectangles"]) and len(info["order"]) >= 3 for info in infos["train"])
    assert len({len(info["order"]) for info in infos["train"]}) >= 2
    assert any(len(info["order"]) >= 5 or not all(info["rectangles"]) or
               info["ratio"] >= 1.5 or info["edge"] for info in infos["test"])


def _same_puzzle(first, second):
    return all(len(first[split]) == len(second[split]) and all(
        np.array_equal(a["input"], b["input"]) and np.array_equal(a["output"], b["output"])
        for a, b in zip(first[split], second[split])) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(1000, 1042)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({tuple((a["input"].shape, a["input"].tobytes())
                      for split in ("train", "test") for a in p[split]) for p in puzzles}) == 42
    infos = [(split, _inspect(pair["input"], pair["output"])) for p in puzzles
             for split in ("train", "test") for pair in p[split]]
    assert len({info["shape"] for _, info in infos}) >= 35
    assert {len(info["order"]) for _, info in infos} >= {2, 3, 4, 5, 6}
    assert len({info["background"] for _, info in infos}) >= 6
    assert sum(info["edge"] for _, info in infos) >= 10
    assert sum(not all(info["rectangles"]) for _, info in infos) >= 20
    assert sum(info["ratio"] >= 1.5 for _, info in infos) >= 10
    assert any(info["shape"][0] % 2 == info["shape"][1] % 2 == 0 for _, info in infos)
    assert any(set(pair["output"][0]) - set(int(v) for train in p["train"]
                                                  for v in train["output"][0])
               for p in puzzles for pair in p["test"])
    disjoint, overlapping = 0, 0
    compact_deep_tests = irregular_cores = stepped_edges = area_reversals = 0
    thin_bands = thick_bands = asymmetric_bands = 0
    regular_to_irregular = irregular_demos = simple_tests = 0
    horizontal = vertical = 0
    for puzzle in puzzles:
        first, second = (set(int(v) for v in pair["output"][0]) for pair in puzzle["train"][:2])
        disjoint += not bool(first & second)
        overlapping += bool(first & second)
        train_infos = [_inspect(pair["input"], pair["output"]) for pair in puzzle["train"]]
        test_infos = [_inspect(pair["input"], pair["output"]) for pair in puzzle["test"]]
        irregular_demos += any(not all(info["rectangles"]) for info in train_infos)
        regular_to_irregular += (all(all(info["rectangles"]) for info in train_infos)
                                 and any(not all(info["rectangles"]) for info in test_infos))
        simple_tests += any(len(info["order"]) <= 3 and all(info["rectangles"])
                            for info in test_infos)
        for pair, info in zip(puzzle["test"], test_infos):
            compact_deep_tests += len(info["order"]) >= 6 and max(pair["input"].shape) <= 18
            irregular_cores += not info["rectangles"][-1]
            stepped_edges += info["edge"] and not all(info["rectangles"])
            area_reversals += any(a < b for a, b in zip(info["areas"], info["areas"][1:]))
        for info in train_infos + test_infos:
            outer = info["boxes"][0]
            height, width = outer[1] - outer[0] + 1, outer[3] - outer[2] + 1
            horizontal += width / height >= 1.5
            vertical += height / width >= 1.5
            for outer, inner in zip(info["boxes"], info["boxes"][1:]):
                steps = (inner[0] - outer[0], outer[1] - inner[1],
                         inner[2] - outer[2], outer[3] - inner[3])
                thin_bands += 1 in steps
                thick_bands += any(step >= 2 for step in steps)
                asymmetric_bands += len(set(steps)) > 1
    assert disjoint >= 5 and overlapping >= 5
    assert compact_deep_tests >= 3 and irregular_cores >= 10 and stepped_edges >= 3
    assert area_reversals >= 10 and thin_bands >= 100 and thick_bands >= 100
    assert asymmetric_bands >= 50 and horizontal >= 10 and vertical >= 10
    assert regular_to_irregular >= 3 and irregular_demos >= 10 and simple_tests >= 3
    for counts in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        first = generate_puzzle(7, *counts)
        assert _same_puzzle(first, generate_puzzle(7, *counts))
        validate_generated_puzzle(first)
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (2, 4), (4, 4)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {counts}")
    for seed, puzzle in enumerate(puzzles, 1000):
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        validate_generated_puzzle(puzzle)
