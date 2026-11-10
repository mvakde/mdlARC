"""Solid, separated rectangles become frames in the same positions and colors."""

import hashlib
import numpy as np


# SHA-256 of shape bytes, int8 input bytes, and int8 output bytes for the
# four original solved pairs. Equal canvas dimensions remain available.
_SOURCE_PAIRS = {
    "53c79e8301b997c22e5ab70ee510c3f36ee36a08e586aa06f7b2ec15046ac5c8",
    "e0e34f62fcc35128433719e6ee26207c36a0c50567c4c4c226cea2b55d567bb2",
    "ebffbe0cffd842ea8844f7c083eb167254af5ac6871ce80354ac2a0b040b70d3",
    "b2f43351c6d3bec6cac79de6467e115a8115d7392cd6d0a79d13d4a0a7020237",
}


def _count_value(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    if num_train is not None:
        num_train = _count_value("num_train", num_train, 2, 10)
    if num_test is not None:
        num_test = _count_value("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _digest_pair(pair):
    a = np.asarray(pair["input"], dtype=np.int8)
    b = np.asarray(pair["output"], dtype=np.int8)
    return hashlib.sha256(bytes(a.shape) + a.tobytes() + b.tobytes()).hexdigest()


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def _components(grid, background):
    """Find four-connected monochrome objects using only the visible grid."""
    height, width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    objects = []
    for row in range(height):
        for col in range(width):
            if seen[row, col] or int(grid[row, col]) == background:
                continue
            color = int(grid[row, col])
            stack = [(row, col)]
            seen[row, col] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (0 <= rr < height and 0 <= cc < width
                            and not seen[rr, cc] and int(grid[rr, cc]) == color):
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            top = min(r for r, _ in cells)
            bottom = max(r for r, _ in cells)
            left = min(c for _, c in cells)
            right = max(c for _, c in cells)
            assert len(cells) == (bottom - top + 1) * (right - left + 1)
            assert np.all(grid[top:bottom + 1, left:right + 1] == color)
            objects.append((color, top, bottom, left, right))
    return objects


def _background(grid):
    corners = [int(grid[r, c]) for r, c in
               ((0, 0), (0, -1), (-1, 0), (-1, -1))]
    background = max(set(corners), key=corners.count)
    assert corners.count(background) >= 3
    return background


def _expected_output(grid, background=None):
    if background is None:
        background = _background(grid)
    out = grid.copy()
    for _, top, bottom, left, right in _components(grid, background):
        if bottom - top >= 2 and right - left >= 2:
            out[top + 1:bottom, left + 1:right] = background
    return out


def _clearance(a, b):
    _, at, ab, al, ar = a
    _, bt, bb, bl, br = b
    vertical = bt - ab - 1 if ab < bt else at - bb - 1 if bb < at else -1
    horizontal = bl - ar - 1 if ar < bl else al - br - 1 if br < al else -1
    return max(vertical, horizontal)


def _size(rng, cap_h, cap_w, thin=False, large=False):
    if thin:
        side = int(rng.integers(1, 3))
        if rng.random() < 0.5:
            return side, int(rng.integers(3, cap_w + 1))
        return int(rng.integers(3, cap_h + 1)), side
    if large and (cap_h >= 9 or cap_w >= 9):
        if cap_h >= 9 and (cap_w < 9 or rng.random() < 0.5):
            return int(rng.integers(9, cap_h + 1)), int(rng.integers(3, cap_w + 1))
        return int(rng.integers(3, cap_h + 1)), int(rng.integers(9, cap_w + 1))
    return int(rng.integers(3, cap_h + 1)), int(rng.integers(3, cap_w + 1))


def _free_boxes(rng, height, width, count, thin_index):
    for _ in range(14):
        boxes = []
        max_h = min(height - 1, 20 if count <= 3 else 12 if count <= 5 else 9)
        max_w = min(width - 1, 20 if count <= 3 else 12 if count <= 5 else 9)
        homogeneous = count > 1 and rng.random() < 0.10
        common = None
        for index in range(count):
            thin = index == thin_index
            large = count <= 4 and index == 0 and rng.random() < 0.33
            if homogeneous and common is not None and not thin:
                rh, rw = common
            else:
                rh, rw = _size(rng, max_h, max_w, thin, large)
                if not thin and common is None:
                    common = (rh, rw)
            placed = False
            for trial in range(130):
                top = int(rng.integers(0, height - rh + 1))
                left = int(rng.integers(0, width - rw + 1))
                if trial < 40 and rng.random() < 0.22:
                    edge = int(rng.integers(0, 4))
                    if edge == 0:
                        top = 0
                    elif edge == 1:
                        top = height - rh
                    elif edge == 2:
                        left = 0
                    else:
                        left = width - rw
                box = (0, top, top + rh - 1, left, left + rw - 1)
                if all(_clearance(box, old) >= 1 for old in boxes):
                    boxes.append(box)
                    placed = True
                    break
            if not placed:
                break
        if len(boxes) == count and _corner_count(boxes, height, width) >= 3:
            return boxes
    return None


def _corner_count(boxes, height, width):
    corners = ((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1))
    return sum(not any(t <= r <= b and l <= c <= rr for _, t, b, l, rr in boxes)
               for r, c in corners)


def _partition(rng, length, parts):
    # Every slot has at least three cells; every intervening gap has at least one.
    spare = length - 3 * parts - (parts - 1)
    if spare < 0:
        return None
    allocation = rng.multinomial(spare, np.ones(2 * parts - 1) / (2 * parts - 1))
    sizes = [3 + int(allocation[2 * i]) for i in range(parts)]
    gaps = [1 + int(allocation[2 * i + 1]) for i in range(parts - 1)]
    starts = [0]
    for i in range(1, parts):
        starts.append(starts[-1] + sizes[i - 1] + gaps[i - 1])
    return list(zip(starts, sizes))


def _slot_boxes(rng, height, width, count, thin_index):
    choices = [(rows, cols) for rows in range(1, count + 1)
               for cols in range(1, count + 1)
               if rows * cols >= count and rows * cols <= count + 2
               and 4 * rows - 1 <= height - 2 and 4 * cols - 1 <= width - 2]
    if not choices:
        raise RuntimeError("canvas cannot fit the requested rectangles")
    rows, cols = choices[int(rng.integers(len(choices)))]
    row_slots = _partition(rng, height - 2, rows)
    col_slots = _partition(rng, width - 2, cols)
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(cells)
    boxes = []
    for index, (r, c) in enumerate(cells[:count]):
        row_start, row_size = row_slots[r]
        col_start, col_size = col_slots[c]
        rh, rw = _size(rng, row_size, col_size,
                       thin=index == thin_index,
                       large=count <= 4 and index == 0 and rng.random() < 0.38)
        top = 1 + row_start + int(rng.integers(row_size - rh + 1))
        left = 1 + col_start + int(rng.integers(col_size - rw + 1))
        boxes.append((0, top, top + rh - 1, left, left + rw - 1))
    return boxes


def _colors(rng, background, count):
    available = [c for c in range(10) if c != background]
    mode = rng.random()
    if mode < 0.22:
        color = int(rng.choice(available))
        colors = [color] * count
    elif mode < 0.58:
        colors = [int(x) for x in rng.choice(available, size=count, replace=False)]
    else:
        colors = [int(x) for x in rng.choice(available, size=count, replace=True)]
    if background != 0 and rng.random() < 0.33:
        colors[int(rng.integers(count))] = 0
    return colors


def _object_count(rng, regime, pair_index):
    if regime == "source_train" and pair_index == 0:
        return int(rng.integers(1, 4))
    if regime == "source_train" and pair_index == 1:
        return int(rng.integers(2, 7))
    if regime == "target_test" and rng.random() < 0.48:
        return int(rng.integers(4, 9))
    return int(rng.integers(1, 7))


def _canvas(rng, count):
    minimum = 7 if count == 1 else 9 if count == 2 else 11 if count <= 4 else 16 if count <= 6 else 19
    # Independent height/width sampling preserves compact and spacious scenes
    # at every object count that can physically fit them.
    return (int(rng.integers(minimum, 31)),
            int(rng.integers(minimum, 31)))


def _generate_pair(rng, regime, pair_index, background, thin_requested):
    count = _object_count(rng, regime, pair_index)
    if thin_requested:
        count = max(2, count)
    height, width = _canvas(rng, count)
    thin_index = int(rng.integers(1, count)) if thin_requested else -1
    boxes = _free_boxes(rng, height, width, count, thin_index) if rng.random() < 0.65 else None
    if boxes is None:
        boxes = _slot_boxes(rng, height, width, count, thin_index)
    colors = _colors(rng, background, count)
    grid = np.full((height, width), background, dtype=np.int8)
    for (unused, top, bottom, left, right), color in zip(boxes, colors):
        grid[top:bottom + 1, left:right + 1] = color
    return {"input": grid, "output": _expected_output(grid, background)}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, rng, num_train, num_test)

    # The rule has no hidden color map or direction. Thin examples are paired
    # with a training demonstration when they occur in a puzzle.
    thin_family = bool(rng.random() < 0.58)
    thin_train = int(rng.integers(num_train)) if thin_family else -1
    thin_test = int(rng.integers(num_test)) if thin_family and rng.random() < 0.72 else -1
    backgrounds = [int(x) for x in rng.integers(0, 10, size=num_train + num_test)]
    if backgrounds[1] == backgrounds[0]:
        backgrounds[1] = (backgrounds[0] + int(rng.integers(1, 10))) % 10

    puzzle = {"train": [], "test": []}
    used = set()
    for split, count, offset in (("train", num_train, 0), ("test", num_test, num_train)):
        for index in range(count):
            regime = "source_train" if split == "train" else "target_test"
            thin = index == (thin_train if split == "train" else thin_test)
            for _ in range(32):
                child = np.random.default_rng(int(rng.integers(0, 2**63, dtype=np.int64)))
                pair = _generate_pair(child, regime, index, backgrounds[offset + index], thin)
                key = _grid_key(pair["input"])
                if key not in used and _digest_pair(pair) not in _SOURCE_PAIRS:
                    puzzle[split].append(pair)
                    used.add(key)
                    break
            else:
                raise RuntimeError("could not construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    keys = set()
    pair_keys = set()
    training_thin = False
    test_thin = False
    training_nontrivial = False
    training_multi = False
    backgrounds = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = pair["input"], pair["output"]
            for grid in (a, b):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert a.shape == b.shape
            key = _grid_key(a)
            pair_key = (key, _grid_key(b))
            assert key not in keys and pair_key not in pair_keys
            keys.add(key)
            pair_keys.add(pair_key)
            assert _digest_pair(pair) not in _SOURCE_PAIRS
            background = _background(a)
            backgrounds.append(background)
            objects = _components(a, background)
            assert 1 <= len(objects) <= 8
            assert all(_clearance(objects[i], objects[j]) >= 1
                       for i in range(len(objects)) for j in range(i + 1, len(objects)))
            assert np.array_equal(b, _expected_output(a, background))
            assert not np.array_equal(a, b)
            thin = any(bt - tp < 2 or rt - lf < 2
                       for _, tp, bt, lf, rt in objects)
            if split == "train":
                training_thin |= thin
                training_nontrivial |= not np.array_equal(a, b)
                training_multi |= len(objects) >= 2
            else:
                test_thin |= thin
    assert training_nontrivial and training_multi
    assert len(set(backgrounds[:len(puzzle["train"])])) >= 2
    assert not test_thin or training_thin


def _same_puzzle(first, second):
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s])
        for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert _same_puzzle(generate_puzzle(314159, 7, 3), generate_puzzle(314159, 7, 3))
    assert not _same_puzzle(generate_puzzle(1001, 6, 2), generate_puzzle(1002, 6, 2))
    for ntrain, ntest in ((2, 1), (10, 1), (10, 4), (6, 2), (7, 3)):
        p = generate_puzzle(9000 + ntrain * 10 + ntest, ntrain, ntest)
        assert (len(p["train"]), len(p["test"])) == (ntrain, ntest)
    for i, args in enumerate(({"num_test": 1}, {"num_test": 4},
                              {"num_train": 2}, {"num_train": 10})):
        p = generate_puzzle(12000 + i, **args)
        assert all(len(p[split]) == args[key] for split, key in
                   (("train", "num_train"), ("test", "num_test")) if key in args)
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(77, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count request accepted: {args}")

    all_inputs = set()
    backgrounds = set()
    layouts = set()
    shapes = set()
    counts = set()
    sizes = set()
    thin_pairs = 0
    edge_pairs = 0
    same_color_pairs = 0
    black_objects = 0
    large_objects = 0
    nonblack_background_pairs = 0
    all_distinct_pairs = 0
    homogeneous_pairs = 0
    staggered_pairs = 0
    compact_pairs = 0
    large_canvas_pairs = 0
    novel_test_pairs = 0
    test_fewer_puzzles = 0
    clearance_values = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_colors = {
            obj[0]
            for pair in puzzle["train"]
            for obj in _components(pair["input"], _background(pair["input"]))
        }
        train_counts = [len(_components(pair["input"], _background(pair["input"])))
                        for pair in puzzle["train"]]
        test_counts = [len(_components(pair["input"], _background(pair["input"])))
                       for pair in puzzle["test"]]
        test_fewer_puzzles += min(test_counts) < max(train_counts)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, b = pair["input"], pair["output"]
                bg = _background(a)
                objects = _components(a, bg)
                # Rebuild a frame cell by cell, separately from the slicing
                # used by the generator's output construction.
                rebuilt = a.copy()
                for color, top, bottom, left, right in objects:
                    for row in range(top, bottom + 1):
                        for col in range(left, right + 1):
                            if top < row < bottom and left < col < right:
                                rebuilt[row, col] = bg
                assert np.array_equal(rebuilt, b)
                all_inputs.add(_grid_key(a))
                backgrounds.add(bg)
                shapes.add(a.shape)
                counts.add(len(objects))
                nonblack_background_pairs += bg != 0
                all_distinct_pairs += (len(objects) >= 2 and
                                       len({obj[0] for obj in objects}) == len(objects))
                homogeneous_pairs += (len(objects) >= 2 and
                                      len({(b - t + 1, r - l + 1)
                                           for _, t, b, l, r in objects}) == 1)
                staggered_pairs += (len(objects) >= 2 and
                                    len({t for _, t, b, l, r in objects}) > 1 and
                                    len({l for _, t, b, l, r in objects}) > 1)
                compact_pairs += a.shape[0] <= 17 and a.shape[1] <= 17
                large_canvas_pairs += a.shape[0] >= 22 and a.shape[1] >= 22
                if split == "test":
                    novel_test_pairs += any(obj[0] not in train_colors for obj in objects)
                for first in range(len(objects)):
                    for second in range(first + 1, len(objects)):
                        clearance_values.add(_clearance(objects[first], objects[second]))
                layouts.add((a.shape, tuple((t, bb, l, r) for _, t, bb, l, r in objects)))
                for color, top, bottom, left, right in objects:
                    h, w = bottom - top + 1, right - left + 1
                    sizes.add((h, w))
                    large_objects += h > 8 or w > 8
                    black_objects += color == 0 and bg != 0
                thin_pairs += any(bt - tp < 2 or rt - lf < 2
                                  for _, tp, bt, lf, rt in objects)
                edge_pairs += any(tp == 0 or lf == 0 or bt == a.shape[0] - 1
                                  or rt == a.shape[1] - 1 for _, tp, bt, lf, rt in objects)
                same_color_pairs += len({o[0] for o in objects}) < len(objects)
    assert len(all_inputs) >= 100 and len(layouts) >= 100
    assert len(shapes) >= 40 and len(sizes) >= 40 and len(counts) >= 6
    assert len(backgrounds) == 10
    assert thin_pairs >= 5 and edge_pairs >= 5 and same_color_pairs >= 5
    assert black_objects >= 5 and large_objects >= 5
    assert nonblack_background_pairs >= 30 and all_distinct_pairs >= 10
    assert homogeneous_pairs >= 3 and staggered_pairs >= 20
    assert compact_pairs >= 5 and large_canvas_pairs >= 5
    assert novel_test_pairs >= 3 and test_fewer_puzzles >= 5
    assert len(clearance_values) >= 8 and min(clearance_values) == 1
