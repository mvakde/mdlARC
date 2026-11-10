"""Filled rectangles acquire a checkerboard of background-colored holes."""

import numpy as np


def _signature(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _solid(shape, background, rectangles):
    grid = np.full(shape, background, dtype=np.int8)
    for top, bottom, left, right, color in rectangles:
        grid[top:bottom, left:right] = color
    return grid


# Exclusive lower/right bounds, transcribed from task.json. An exact pair copy
# must also have an exact input copy because the transformation is deterministic.
_SOURCE_SPECS = (
    ((13, 15), ((0, 5, 0, 5, 1), (6, 12, 4, 14, 8))),
    ((16, 15), ((4, 12, 4, 14, 2),)),
    ((17, 16), ((1, 8, 2, 10, 3), (9, 16, 0, 8, 4),
                (9, 14, 10, 16, 8))),
    ((19, 19), ((1, 10, 1, 9, 8), (2, 9, 10, 18, 7),
                (11, 19, 1, 12, 6), (11, 17, 13, 18, 4))),
)
_SOURCE_INPUTS = frozenset(
    _signature(_solid(shape, 0, rectangles))
    for shape, rectangles in _SOURCE_SPECS
)


def _objects(grid):
    """Find the visible background and 4-connected solid color components."""
    values, counts = np.unique(grid, return_counts=True)
    assert len(values) >= 2
    winners = np.flatnonzero(counts == counts.max())
    assert len(winners) == 1
    background = int(values[int(winners[0])])
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    rectangles = []
    for row in range(h):
        for col in range(w):
            if visited[row, col] or int(grid[row, col]) == background:
                continue
            color = int(grid[row, col])
            stack = [(row, col)]
            visited[row, col] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                    if (0 <= nr < h and 0 <= nc < w and
                            not visited[nr, nc] and int(grid[nr, nc]) == color):
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            top = min(r for r, _ in cells)
            bottom = max(r for r, _ in cells) + 1
            left = min(c for _, c in cells)
            right = max(c for _, c in cells) + 1
            assert bottom - top >= 3 and right - left >= 3
            assert len(cells) == (bottom-top) * (right-left)
            rectangles.append((top, bottom, left, right, color))
    assert rectangles
    return background, sorted(rectangles)


def _fill(grid, background, rectangles):
    result = grid.copy()
    for top, bottom, left, right, _ in rectangles:
        for row in range(top+1, bottom-1):
            for col in range(left+1, right-1):
                if (row-top) % 2 == (col-left) % 2:
                    result[row, col] = background
    return result


def _counts(rng, num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or
                not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test+1, 11))
    elif num_train is None:
        num_train = int(rng.integers(int(num_test)+1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train)-1)+1))
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _partition(rng, box, count):
    """Recursively allocate distinct regions, with a visible one-cell gap."""
    if count == 1:
        return [box]
    top, bottom, left, right = box
    options = []
    for first in range(1, count):
        for axis in (0, 1):
            length = (bottom-top) if axis == 0 else (right-left)
            for size in range(4, length-4):
                options.append((first, axis, size))
    if not options:
        return None
    order = rng.permutation(len(options))
    for choice in order[:min(len(order), 100)]:
        first, axis, size = options[int(choice)]
        if axis == 0:
            a = (top, top+size, left, right)
            b = (top+size+1, bottom, left, right)
        else:
            a = (top, bottom, left, left+size)
            b = (top, bottom, left+size+1, right)
        lhs = _partition(rng, a, first)
        if lhs is None:
            continue
        rhs = _partition(rng, b, count-first)
        if rhs is not None:
            return lhs + rhs
    return None


def _sample_size(rng, region, profile, minimum):
    top, bottom, left, right = region
    rh, rw = bottom-top, right-left
    if rh < minimum or rw < minimum:
        return None
    if profile == "packed":
        height = max(minimum, rh-int(rng.integers(0, min(4, rh-minimum+1))))
        width = max(minimum, rw-int(rng.integers(0, min(4, rw-minimum+1))))
    elif profile == "sparse":
        height = int(rng.integers(minimum, max(minimum+1, (rh+minimum)//2+1)))
        width = int(rng.integers(minimum, max(minimum+1, (rw+minimum)//2+1)))
    elif profile == "long" and max(rh, rw) >= 11:
        if (rw >= 11 and (rh < 11 or rng.random() < 0.5)):
            width = int(rng.integers(max(11, rw-3), rw+1))
            height = int(rng.integers(minimum, min(rh, 6)+1))
        else:
            height = int(rng.integers(max(11, rh-3), rh+1))
            width = int(rng.integers(minimum, min(rw, 6)+1))
    else:
        height = int(rng.integers(minimum, rh+1))
        width = int(rng.integers(minimum, rw+1))
    return height, width


def _place(rng, region, shape, size, parity, placement):
    top, bottom, left, right = region
    h, w = shape
    rh, rw = size
    candidates = []
    for row in range(top, bottom-rh+1):
        for col in range(left, right-rw+1):
            if parity is not None and (row+col) % 2 != parity:
                continue
            if placement == "interior" and not (
                    row > 0 and col > 0 and row+rh < h and col+rw < w):
                continue
            if placement == "edge" and not (
                    row == 0 or col == 0 or row+rh == h or col+rw == w):
                continue
            candidates.append((row, row+rh, col, col+rw))
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def _generate_pair(rng, count, background, colors, split, mixed=False,
                   shape_override=None):
    minimum = 4 if split == "train" else 3
    low = {1: 9, 2: 11, 3: 14, 4: 16}[count]
    # The axes vary independently. A test can be smaller than a training pair.
    upper = 28 if split == "train" else 31
    profiles = ("sparse", "balanced", "packed", "long")
    profile = profiles[int(rng.integers(len(profiles)))]
    placement = ("free", "interior", "edge")[
        int(rng.integers(3))] if split == "test" else (
        "interior" if rng.random() < 0.22 else "free")
    for attempt in range(180):
        if shape_override is None:
            h = int(rng.integers(low, upper))
            w = int(rng.integers(low, upper))
        else:
            h, w = shape_override
        regions = _partition(rng, (0, h, 0, w), count)
        if regions is None:
            continue
        rectangles = []
        for index, (region, color) in enumerate(zip(regions, colors)):
            size = _sample_size(rng, region, profile, minimum)
            if size is None:
                break
            # Same pair, opposite absolute origin parity: both local first
            # interior cells are removed in the training output.
            parity = index if mixed and index < 2 else None
            box = _place(rng, region, (h, w), size, parity, placement)
            if box is None:
                break
            rectangles.append((*box, color))
        if len(rectangles) != count:
            continue
        if split == "train" and not any(
                b-t >= 5 and r-l >= 5 for t, b, l, r, _ in rectangles):
            continue
        grid = _solid((h, w), background, rectangles)
        values, amounts = np.unique(grid, return_counts=True)
        if int(values[int(np.argmax(amounts))]) != background or (
                np.count_nonzero(amounts == amounts.max()) != 1):
            continue
        return grid, _fill(grid, background, rectangles)

    # A bounded, predictable relaxation for cramped legal requests.
    if profile != "balanced" or placement != "free":
        return _generate_pair_fallback(rng, count, background, colors, split,
                                       mixed, shape_override)
    raise ValueError("could not construct a pair")


def _generate_pair_fallback(rng, count, background, colors, split, mixed,
                            shape_override=None):
    minimum = 4 if split == "train" else 3
    for _ in range(180):
        if shape_override is None:
            h = int(rng.integers(20, 31))
            w = int(rng.integers(20, 31))
        else:
            h, w = shape_override
        regions = _partition(rng, (0, h, 0, w), count)
        if regions is None:
            continue
        rectangles = []
        for i, (region, color) in enumerate(zip(regions, colors)):
            size = _sample_size(rng, region, "balanced", minimum)
            if size is None:
                break
            box = _place(rng, region, (h, w), size,
                         i if mixed and i < 2 else None, "free")
            if box is None:
                break
            rectangles.append((*box, color))
        if len(rectangles) != count or (split == "train" and not any(
                b-t >= 5 and r-l >= 5 for t, b, l, r, _ in rectangles)):
            continue
        grid = _solid((h, w), background, rectangles)
        values, amounts = np.unique(grid, return_counts=True)
        if int(values[int(np.argmax(amounts))]) == background and (
                np.count_nonzero(amounts == amounts.max()) == 1):
            return grid, _fill(grid, background, rectangles)
    raise ValueError("could not construct a valid pair")


def _colors(rng, count, background, allowed, repeat):
    choices = [c for c in allowed if c != background]
    assert choices
    colors = [int(choices[int(rng.integers(len(choices)))])
              for _ in range(count)]
    if repeat and count >= 2:
        colors[1] = colors[0]
    return colors


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    train_n, test_n = _counts(rng, num_train, num_test)
    colors = [int(c) for c in rng.permutation(10)]
    variable_background = rng.random() < 0.38
    unseen_test_color = rng.random() < 0.52
    stable_palette = (not variable_background and not unseen_test_color
                      and rng.random() < 0.6)
    backgrounds = colors[:2] if variable_background else colors[:1]
    reserved = colors[-2:] if unseen_test_color else []
    train_pool = (colors[1:4] if stable_palette else
                  [c for c in colors if c not in reserved])

    # Counts are pair properties; the training order is deliberately unsorted.
    train_counts = ([2, 3] if train_n == 2 else
                    [1, 2, 3] + [int(rng.integers(1, 4))
                                  for _ in range(train_n-3)])
    rng.shuffle(train_counts)
    mixed_index = next(i for i, c in enumerate(train_counts) if c >= 2)
    test_counts = [4 if rng.random() < 0.48 else
                   int(rng.integers(1, 4)) for _ in range(test_n)]
    if test_n > 1 and rng.random() < 0.6:
        test_counts[0] = 4
        test_counts[1] = int(rng.integers(1, 4))

    train_backgrounds = [backgrounds[0]] * train_n
    if variable_background:
        train_backgrounds[1] = backgrounds[1]
        for i in range(2, train_n):
            train_backgrounds[i] = backgrounds[int(rng.integers(2))]
    test_backgrounds = [backgrounds[int(rng.integers(len(backgrounds)))]
                        for _ in range(test_n)]

    puzzle = {"train": [], "test": []}
    seen = set()
    stable_color_cursor = 0
    for split, counts, pair_backgrounds in (
            ("train", train_counts, train_backgrounds),
            ("test", test_counts, test_backgrounds)):
        for index, (count, background) in enumerate(zip(counts, pair_backgrounds)):
            pair_colors = _colors(rng, count, background,
                                  train_pool if (split == "train" or stable_palette)
                                  else colors,
                                  rng.random() < 0.34)
            if split == "train" and stable_palette:
                for j in range(count):
                    if stable_color_cursor < len(train_pool):
                        pair_colors[j] = train_pool[stable_color_cursor]
                        stable_color_cursor += 1
            if split == "train" and variable_background and index < 2:
                pair_colors[0] = backgrounds[1-index]
            if split == "test" and unseen_test_color and index == 0:
                pair_colors[0] = reserved[0]
                if count >= 2 and rng.random() < 0.45:
                    pair_colors[1] = reserved[0]
            matching_shapes = [pair["input"].shape for pair in puzzle["train"]
                               if len(_objects(pair["input"])[1]) >= count]
            shape_override = (matching_shapes[int(rng.integers(len(matching_shapes)))]
                              if split == "test" and matching_shapes and
                              rng.random() < 0.24 else None)
            for _ in range(24):
                input_grid, output_grid = _generate_pair(
                    rng, count, background, pair_colors, split,
                    mixed=(split == "train" and index == mixed_index),
                    shape_override=shape_override)
                signature = _signature(input_grid)
                if signature not in seen and signature not in _SOURCE_INPUTS:
                    seen.add(signature)
                    puzzle[split].append({"input": input_grid,
                                          "output": output_grid})
                    break
            else:
                raise ValueError("could not make a unique non-source pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    train_parities = []
    train_backgrounds = []
    train_colors = []
    test_backgrounds = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            src, dst = pair["input"], pair["output"]
            for grid in (src, dst):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert src.shape == dst.shape and not np.array_equal(src, dst)
            sig = _signature(src)
            assert sig not in seen and sig not in _SOURCE_INPUTS
            seen.add(sig)
            background, rectangles = _objects(src)
            assert len(rectangles) <= 4
            assert np.array_equal(dst, _fill(src, background, rectangles))
            for top, bottom, left, right, color in rectangles:
                assert int(dst[top+1, left+1]) == background
                assert np.all(dst[top, left:right] == color)
                assert np.all(dst[bottom-1, left:right] == color)
                assert np.all(dst[top:bottom, left] == color)
                assert np.all(dst[top:bottom, right-1] == color)
            if split == "train":
                assert len(rectangles) <= 3
                assert all(b-t >= 4 and r-l >= 4 for t, b, l, r, _ in rectangles)
                assert any(b-t >= 5 and r-l >= 5 for t, b, l, r, _ in rectangles)
                train_parities.append({(t+l) % 2 for t, _, l, _, _ in rectangles})
                train_backgrounds.append(background)
                train_colors.append({color for *_, color in rectangles})
            else:
                test_backgrounds.append(background)
    assert any(parities == {0, 1} for parities in train_parities)
    assert set(test_backgrounds).issubset(set(train_backgrounds))
    if len(set(train_backgrounds)) > 1:
        assert any(background in set.union(*train_colors)
                   for background in train_backgrounds)


def _fingerprint(puzzle):
    return tuple((split, _signature(pair["input"]), _signature(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def _has_quadrants(rectangles, height, width):
    if len(rectangles) != 4:
        return False
    for row in range(1, height-1):
        row_sides = [0 if b <= row else (1 if t > row else 2)
                     for t, b, _, _, _ in rectangles]
        if row_sides.count(0) != 2 or row_sides.count(1) != 2:
            continue
        for col in range(1, width-1):
            positions = {(rs, 0 if r <= col else (1 if l > col else 2))
                         for rs, (_, _, l, r, _) in zip(row_sides, rectangles)}
            if positions == {(0, 0), (0, 1), (1, 0), (1, 1)}:
                return True
    return False


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_fingerprint(p) for p in puzzles}) == 42
    assert _fingerprint(generate_puzzle(12345, 7, 3)) == _fingerprint(
        generate_puzzle(12345, 7, 3))
    shapes, sizes, densities = set(), set(), []
    train_orders, test_counts, palettes, background_switches = set(), set(), set(), 0
    mixed, repeated_test, unseen_test, small_test, interior_tests = 0, 0, 0, 0, 0
    long_h, long_v, thin, quadrant = 0, 0, 0, 0
    familiar_test, equal_shape, several_edges, dense_test = 0, 0, 0, 0
    role_swaps, stable_palettes, nonmonotone = 0, 0, 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train = [_objects(pair["input"]) for pair in puzzle["train"]]
        test = [_objects(pair["input"]) for pair in puzzle["test"]]
        order = tuple(len(rects) for _, rects in train)
        train_orders.add(order)
        nonmonotone += order != tuple(sorted(order))
        test_counts.update(len(rects) for _, rects in test)
        train_bg = {bg for bg, _ in train}
        background_switches += len(train_bg) > 1
        palette = set.union(*(set(c for *_, c in rects) for _, rects in train))
        role_swaps += any(bg in palette for bg in train_bg)
        stable_palettes += (len(train_bg) == 1 and
                            all(c in palette for _, rects in test
                                for *_, c in rects))
        palettes.add(tuple(sorted(palette)))
        mixed += any({(t+l) % 2 for t, _, l, _, _ in rects} == {0, 1}
                     for _, rects in train)
        max_train_area = max(pair["input"].size for pair in puzzle["train"])
        train_shapes = {pair["input"].shape for pair in puzzle["train"]}
        for pair, (bg, rects) in zip(puzzle["test"], test):
            colors = [c for *_, c in rects]
            repeated_test += len(set(colors)) < len(colors)
            unseen_test += any(c not in palette for c in colors)
            familiar_test += len(rects) in set(order)
            small_test += pair["input"].size < max_train_area
            equal_shape += pair["input"].shape in train_shapes
            h, w = pair["input"].shape
            interior_tests += all(t > 0 and l > 0 and b < h and r < w
                                  for t, b, l, r, _ in rects)
            several_edges += sum(t == 0 or l == 0 or b == h or r == w
                                 for t, b, l, r, _ in rects) >= 2
            quadrant += _has_quadrants(rects, h, w)
            dense_test += sum((b-t)*(r-l) for t, b, l, r, _ in rects) / (h*w) >= .5
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                bg, rects = _objects(grid)
                rebuilt = grid.copy()
                for t, b, l, r, _ in rects:
                    rr, cc = np.indices((b-t-2, r-l-2))
                    mask = (rr+cc) % 2 == 0
                    region = rebuilt[t+1:b-1, l+1:r-1]
                    region[mask] = bg
                assert np.array_equal(rebuilt, pair["output"])
                shapes.add(grid.shape)
                densities.append(sum((b-t)*(r-l) for t, b, l, r, _ in rects)/grid.size)
                for t, b, l, r, _ in rects:
                    sizes.add((b-t, r-l))
                    long_h += r-l >= 11
                    long_v += b-t >= 11
                    thin += b-t == 3 or r-l == 3
    assert len(train_orders) >= 12 and mixed == 42
    assert test_counts == {1, 2, 3, 4}
    assert len(shapes) >= 75 and len(sizes) >= 55 and len(palettes) >= 25
    assert background_switches >= 7
    assert role_swaps >= 7 and stable_palettes >= 8
    assert nonmonotone >= 25
    assert repeated_test >= 6 and unseen_test >= 8
    assert small_test >= 10 and interior_tests >= 5
    assert familiar_test >= 8 and equal_shape >= 5
    assert several_edges >= 5 and quadrant >= 4 and dense_test >= 5
    assert long_h >= 8 and long_v >= 8 and thin >= 8
    assert max(densities) >= 0.55 and min(densities) <= 0.15
    for i, (nt, ns) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(2000+i, nt, ns)
        assert len(puzzle["train"]) == nt and len(puzzle["test"]) == ns
    for i, args in enumerate(({"num_test": 1}, {"num_test": 4},
                              {"num_train": 2}, {"num_train": 10})):
        puzzle = generate_puzzle(3000+i, **args)
        validate_generated_puzzle(puzzle)
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(4000, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid counts: {args}")
