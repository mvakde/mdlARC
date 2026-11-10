"""Rectangular tile completion puzzles in the source ARC family."""

import hashlib
import numpy as np


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _resolve_counts(rng, seed, num_train, num_test):
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        low = max(2, num_test + 1)
        num_train = low + (seed // 4) % (11 - low)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    height, width = mask.shape
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            seen[row, col] = True
            stack = [(row, col)]
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            result.append(cells)
    return result


def _rectangle(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), max(rows) + 1, min(cols), max(cols) + 1


def _holes(mask):
    records = []
    for cells in _components(~mask):
        r0, r1, c0, c1 = _rectangle(cells)
        if len(cells) != (r1 - r0) * (c1 - c0):
            return None
        records.append((r0, r1, c0, c1))
    return records


def _key(shapes):
    return tuple(sorted(tuple(map(int, shape)) for shape in shapes))


def _hole_shape(rect):
    return rect[1] - rect[0], rect[3] - rect[2]


def _valid_mask(mask, shapes):
    if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
        return False
    if len(_components(mask)) != 1:
        return False
    holes = _holes(mask)
    return (
        holes is not None
        and _key(map(_hole_shape, holes)) == _key(shapes)
        and int(mask.sum()) > max(h * w for h, w in shapes)
    )


def _pick_position(rng, height, width, shape):
    h, w = shape
    if h > height or w > width:
        return None
    category = int(rng.integers(7))
    row = int(rng.integers(height - h + 1))
    col = int(rng.integers(width - w + 1))
    if category == 0:
        row = 0
    elif category == 1:
        row = height - h
    elif category == 2:
        col = 0
    elif category == 3:
        col = width - w
    elif category == 4:
        row = (0, height - h)[int(rng.integers(2))]
        col = (0, width - w)[int(rng.integers(2))]
    elif category == 5 and height - h >= 2 and width - w >= 2:
        row = int(rng.integers(1, height - h))
        col = int(rng.integers(1, width - w))
    return row, row + h, col, col + w


def _cutout_layout(rng, height, width, shapes):
    for _ in range(180):
        rects = []
        mask = np.ones((height, width), dtype=bool)
        for shape in shapes:
            rect = _pick_position(rng, height, width, shape)
            if rect is None:
                break
            r0, r1, c0, c1 = rect
            mask[r0:r1, c0:c1] = False
            rects.append(rect)
        if len(rects) == len(shapes) and _valid_mask(mask, shapes):
            return mask, rects
    return None


_TILE_SHAPES = tuple(
    (h, w) for h in range(1, 5) for w in range(1, 5)
    if h * w <= 9 and (h, w) != (1, 1)
)


def _tile_shapes(rng, count):
    if count == 2:
        mode = int(rng.integers(8))
        if mode == 0:
            s = ((2, 1), (3, 1), (3, 2), (4, 1))[int(rng.integers(4))]
            return (s, s)
        if mode == 1:
            s = ((1, 2), (1, 3), (1, 3), (2, 3), (1, 4))[int(rng.integers(5))]
            return (s, s)
        if mode == 2:
            s = ((2, 2), (3, 3))[int(rng.integers(2))]
            return (s, s)
        if mode == 7:
            elongated = ((3, 2), (3, 2), (2, 3), (4, 2), (2, 4))[int(rng.integers(5))]
            return ((2, 2), elongated)
        if mode == 3:
            pool = [s for s in _TILE_SHAPES if s[0] < s[1]]
        elif mode == 4:
            pool = [s for s in _TILE_SHAPES if s[0] > s[1]]
        elif mode == 5:
            a = ((1, 2), (1, 3), (2, 3))[int(rng.integers(3))]
            b = ((2, 1), (3, 1), (3, 2))[int(rng.integers(3))]
            return (a, b)
        else:
            pool = _TILE_SHAPES
        choices = rng.choice(len(pool), size=2, replace=False)
        return tuple(pool[int(i)] for i in choices)
    shapes = [tuple(_TILE_SHAPES[int(rng.integers(len(_TILE_SHAPES)))]) for _ in range(count)]
    if count >= 3 and int(rng.integers(3)) == 0:
        shapes[1] = shapes[0]
    return tuple(shapes)


def _selected(rng, shapes, regime):
    low = 6 if regime == "target_test" and int(rng.integers(5)) < 3 else 3
    if _key(shapes) == ((1, 3), (1, 3)) and int(rng.integers(2)) == 0:
        mask = np.ones((3, 5), dtype=bool)
        rects = []
        for row in (0, 2):
            col = int(rng.integers(3))
            rects.append((row, row + 1, col, col + 3))
            mask[row, col:col + 3] = False
        if _valid_mask(mask, shapes):
            return mask, rects
    if _key(shapes) == _key(((2, 2), (3, 2))) and int(rng.integers(2)) == 0:
        # The original tall test has these oriented tiles at opposite ends.
        mask = np.ones((7, 4), dtype=bool)
        upper_col = int(rng.integers(3))
        lower_col = int(rng.integers(3))
        square_rect = (0, 2, upper_col, upper_col + 2)
        vertical_rect = (4, 7, lower_col, lower_col + 2)
        rects = [square_rect if shape == (2, 2) else vertical_rect
                 for shape in shapes]
        for r0, r1, c0, c1 in rects:
            mask[r0:r1, c0:c1] = False
        if _valid_mask(mask, shapes):
            return mask, rects
    for _ in range(45):
        height = int(rng.integers(low, 12))
        width = int(rng.integers(low, 12))
        if height * width < sum(h * w for h, w in shapes) + max(h * w for h, w in shapes) + 2:
            continue
        layout = _cutout_layout(rng, height, width, shapes)
        if layout is not None:
            return layout
    raise RuntimeError("could not construct receiver")


def _decoy_shapes(rng, target):
    mode = int(rng.integers(5))
    count = len(target)
    if mode == 0:
        count = int(rng.choice([n for n in (1, 2, 3, 4) if n != count]))
    shapes = list(target[:count])
    while len(shapes) < count:
        shapes.append(_TILE_SHAPES[int(rng.integers(len(_TILE_SHAPES)))])
    if mode == 1 and len(shapes) == len(target):
        i = int(rng.integers(count))
        options = [s for s in _TILE_SHAPES if s != shapes[i]]
        shapes[i] = options[int(rng.integers(len(options)))]
    elif mode == 2:
        choices = [i for i, (h, w) in enumerate(shapes) if h != w]
        if choices:
            i = choices[int(rng.integers(len(choices)))]
            shapes[i] = shapes[i][::-1]
    elif mode == 3 and count == 2:
        area = sum(h * w for h, w in target)
        choices = [
            (a, b) for a in _TILE_SHAPES for b in _TILE_SHAPES
            if a[0] * a[1] + b[0] * b[1] == area and _key((a, b)) != _key(target)
        ]
        if choices:
            shapes = list(choices[int(rng.integers(len(choices)))])
    else:
        shapes = [_TILE_SHAPES[int(rng.integers(len(_TILE_SHAPES)))] for _ in range(count)]
    if _key(shapes) == _key(target):
        shapes[0] = next(s for s in _TILE_SHAPES if s != shapes[0])
    return tuple(shapes)


def _distractor(rng, target_shapes, selected_shape, index):
    for _ in range(85):
        shapes = _decoy_shapes(rng, target_shapes)
        if index == 0 and int(rng.integers(2)) == 0:
            height, width = selected_shape
        else:
            height = int(rng.integers(4, 12))
            width = int(rng.integers(4, 12))
        if height * width < sum(h * w for h, w in shapes) + max(h * w for h, w in target_shapes) + 2:
            continue
        layout = _cutout_layout(rng, height, width, shapes)
        if layout is not None:
            return layout[0]
    raise RuntimeError("could not construct distractor")


def _colors(rng, palette, candidate_count, shapes):
    order = [int(c) for c in rng.permutation(palette)]
    background = order.pop()
    receiver = order.pop()
    candidate_colors = [receiver]
    for _ in range(candidate_count - 1):
        if int(rng.integers(5)) == 0:
            candidate_colors.append(candidate_colors[int(rng.integers(len(candidate_colors)))])
        else:
            candidate_colors.append(order.pop())
    filler_colors = {}
    for shape in shapes:
        if shape not in filler_colors:
            if filler_colors and int(rng.integers(3)) == 0:
                filler_colors[shape] = list(filler_colors.values())[int(rng.integers(len(filler_colors)))]
            else:
                filler_colors[shape] = order.pop()
    return background, candidate_colors, tuple(filler_colors[s] for s in shapes)


def _pack(rng, objects, background):
    indexed = list(enumerate(objects))
    indexed.sort(key=lambda x: x[1]["mask"].size, reverse=True)
    for attempt in range(28):
        if len(objects) <= 4 and attempt < 6:
            height = width = 16
        elif attempt < 22:
            height = int(rng.integers(16, 31))
            width = int(rng.integers(16, 31))
        else:
            height = width = 30
        positions = {}
        boxes = []
        for index, obj in indexed:
            h, w = obj["mask"].shape
            choices = []
            if h > height or w > width:
                break
            for r in range(height - h + 1):
                for c in range(width - w + 1):
                    if all(r + h < rr or rr + hh < r or c + w < cc or cc + ww < c
                           for rr, cc, hh, ww in boxes):
                        choices.append((r, c))
            if not choices:
                break
            r, c = choices[int(rng.integers(len(choices)))]
            positions[index] = (r, c)
            boxes.append((r, c, h, w))
        if len(positions) != len(objects):
            continue
        grid = np.full((height, width), background, dtype=np.int8)
        for index, obj in enumerate(objects):
            r, c = positions[index]
            mask = obj["mask"]
            h, w = mask.shape
            view = grid[r:r + h, c:c + w]
            view[mask] = obj["color"]
        if int(np.count_nonzero(grid == background)) > height * width // 2:
            return grid
    raise RuntimeError("could not pack objects")


def _make_pair(rng, regime, palette, tile_count, candidate_count):
    for _ in range(35):
        try:
            shapes = _tile_shapes(rng, tile_count)
            receiver, rects = _selected(rng, shapes, regime)
            decoys = [
                _distractor(rng, shapes, receiver.shape, index)
                for index in range(candidate_count - 1)
            ]
            background, candidate_colors, filler_colors = _colors(
                rng, palette, candidate_count, shapes
            )
            output = np.full(receiver.shape, candidate_colors[0], dtype=np.int8)
            for (r0, r1, c0, c1), color in zip(rects, filler_colors):
                output[r0:r1, c0:c1] = color
            objects = [
                {"mask": mask, "color": color}
                for mask, color in zip([receiver] + decoys, candidate_colors)
            ]
            objects.extend(
                {"mask": np.ones(shape, dtype=bool), "color": color}
                for shape, color in zip(shapes, filler_colors)
            )
            input_grid = _pack(rng, objects, background)
            pair = {"input": input_grid, "output": output}
            _inspect(pair)
            return pair
        except (RuntimeError, AssertionError, IndexError):
            continue
    raise RuntimeError("could not generate a valid pair")


def _inspect(pair):
    grid, output = pair["input"], pair["output"]
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    infos = []
    for cells in _components(grid != background):
        r0, r1, c0, c1 = _rectangle(cells)
        crop = grid[r0:r1, c0:c1]
        color = int(grid[cells[0]])
        assert all(int(grid[r, c]) == color for r, c in cells)
        assert np.all((crop == background) | (crop == color))
        solid = len(cells) == crop.size
        info = {"shape": crop.shape, "color": color, "crop": crop,
                "size": len(cells), "solid": solid}
        if not solid:
            holes = _holes(crop != background)
            assert holes and len(_components(crop != background)) == 1
            info["holes"] = holes
        infos.append(info)
    fillers = [x for x in infos if x["solid"]]
    candidates = [x for x in infos if not x["solid"]]
    assert 1 <= len(fillers) <= 4
    assert 2 <= len(candidates) <= 5
    assert all(x["size"] > max(y["size"] for y in fillers) for x in candidates)
    shapes = [x["shape"] for x in fillers]
    for shape in set(shapes):
        assert len({x["color"] for x in fillers if x["shape"] == shape}) == 1
    matching = [x for x in candidates if _key(map(_hole_shape, x["holes"])) == _key(shapes)]
    assert len(matching) == 1
    chosen = matching[0]
    expected = chosen["crop"].copy()
    for rect in chosen["holes"]:
        shape = _hole_shape(rect)
        color = next(x["color"] for x in fillers if x["shape"] == shape)
        r0, r1, c0, c1 = rect
        expected[r0:r1, c0:c1] = color
    assert np.array_equal(output, expected)
    assert not np.any(output == background)
    return {
        "background": background,
        "tile_count": len(fillers),
        "candidate_count": len(candidates),
        "shapes": tuple(shapes),
        "output_shape": output.shape,
        "candidate_shapes": tuple(x["shape"] for x in candidates),
        "hole_counts": tuple(len(x["holes"]) for x in candidates),
        "receiver_color": chosen["color"],
        "candidate_colors": tuple(x["color"] for x in candidates),
        "filler_colors": tuple(x["color"] for x in fillers),
    }


_SOURCE_INPUT_HASHES = frozenset((
    "28eb0102506dbccab9db2ccbc49f86001511b7c888b91841e4ec4162bdbdb24f",
    "f4dc2ad884e691741c18ec09e52c137e8ee8ed648e9d0931be59015299797754",
    "c415fbec6775c5a02f5b395b9a5aa22a82205ad96e7a4e8704f7a1a3bf329424",
    "541b9e084327f34d89c58749aeecce2bbe7712edc276b15148f7e56b05ada4bc",
))


def _signature(grid):
    return grid.shape, np.ascontiguousarray(grid, dtype=np.int8).tobytes()


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    seen_pairs = set()
    train_counts = set()
    train_candidate_counts = set()
    test_counts = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            input_sig = _signature(pair["input"])
            pair_sig = (input_sig, _signature(pair["output"]))
            assert input_sig not in seen_inputs and pair_sig not in seen_pairs
            seen_inputs.add(input_sig)
            seen_pairs.add(pair_sig)
            assert hashlib.sha256(np.asarray(pair["input"], dtype=np.uint8).tobytes()).hexdigest() not in _SOURCE_INPUT_HASHES
            info = _inspect(pair)
            if split == "train":
                train_counts.add(info["tile_count"])
                train_candidate_counts.add(info["candidate_count"])
            else:
                test_counts.add(info["tile_count"])
    assert test_counts.issubset(train_counts)
    assert 2 in train_counts
    assert len(train_counts) >= 2 and len(train_candidate_counts) >= 2


def _same(a, b):
    return all(
        len(a[split]) == len(b[split])
        and all(np.array_equal(x[k], y[k]) for x, y in zip(a[split], b[split]) for k in ("input", "output"))
        for split in ("train", "test")
    )


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, seed, num_train, num_test)
    # Color labels are available throughout the puzzle, but have no fixed role.
    palette = tuple(int(x) for x in rng.permutation(10))
    pair_seeds = rng.integers(0, np.iinfo(np.uint64).max, size=num_train + num_test, dtype=np.uint64)
    train_counts = [2, int(rng.choice((1, 3, 4), p=(0.35, 0.48, 0.17)))]
    for _ in range(num_train - 2):
        train_counts.append(int(rng.choice((1, 2, 3, 4), p=(0.17, 0.52, 0.23, 0.08))))
    rng.shuffle(train_counts)
    test_counts = [int(train_counts[int(rng.integers(num_train))]) for _ in range(num_test)]
    train_candidate_counts = [int(rng.choice((2, 3, 4, 5), p=(0.30, 0.40, 0.25, 0.05)))]
    train_candidate_counts.append(int(rng.choice(
        [count for count in (2, 3, 4, 5) if count != train_candidate_counts[0]])))
    for _ in range(num_train - 2):
        train_candidate_counts.append(int(rng.choice((2, 3, 4, 5), p=(0.30, 0.40, 0.25, 0.05))))
    rng.shuffle(train_candidate_counts)
    test_candidate_counts = [int(rng.choice((2, 3, 4, 5), p=(0.16, 0.35, 0.37, 0.12)))
                             for _ in range(num_test)]
    train = [
        _make_pair(np.random.default_rng(pair_seeds[i]), "source_train", palette,
                   train_counts[i], train_candidate_counts[i])
        for i in range(num_train)
    ]
    test = [
        _make_pair(np.random.default_rng(pair_seeds[num_train + i]), "target_test", palette,
                   test_counts[i], test_candidate_counts[i])
        for i in range(num_test)
    ]
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    for seed, train, test in ((201, 2, 1), (202, 10, 1), (203, 10, 4), (204, 7, 3),
                              (301, None, 1), (302, None, 4), (303, 2, None), (304, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    assert _same(generate_puzzle(777, 7, 3), generate_puzzle(777, 7, 3))
    assert not _same(generate_puzzle(777, 7, 3), generate_puzzle(778, 7, 3))
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": 4.0}):
        try:
            generate_puzzle(900, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair count accepted")
    train_counts = set()
    test_counts = set()
    tile_counts = set()
    candidate_counts = set()
    shapes = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        for pair in puzzle["train"] + puzzle["test"]:
            info = _inspect(pair)
            tile_counts.add(info["tile_count"])
            candidate_counts.add(info["candidate_count"])
            shapes.add(info["output_shape"])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert tile_counts == {1, 2, 3, 4}
    assert {2, 3, 4}.issubset(candidate_counts)
    assert len(shapes) >= 20
