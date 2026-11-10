"""Generate overlapping six-by-six ARC rings and their canonical stacks."""

import numpy as np


_RING_MASK = np.array(
    [
        [0, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0],
    ],
    dtype=bool,
)
_RING_CELLS = tuple((int(r), int(c)) for r, c in np.argwhere(_RING_MASK))
_RING_SET = frozenset(_RING_CELLS)
_AUTO_COUNTS = ((2, 1), (3, 1), (4, 2), (5, 3), (6, 4),
                (7, 1), (8, 2), (9, 3), (10, 4))


def _overlap_offsets():
    result = []
    for dr in range(-5, 6):
        for dc in range(-5, 6):
            if dr == dc == 0:
                continue
            shared = len(_RING_SET.intersection(
                (r + dr, c + dc) for r, c in _RING_CELLS
            ))
            if 2 <= shared <= 6:
                result.append((dr, dc, shared))
    return tuple(result)


_OFFSETS = _overlap_offsets()


def _validate_optional_count(name, value, low, high):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(seed, rng, num_train, num_test):
    _validate_optional_count("num_train", num_train, 2, 10)
    _validate_optional_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        return _AUTO_COUNTS[seed % len(_AUTO_COUNTS)]
    if num_train is None:
        num_train = int(rng.integers(max(2, int(num_test) + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train) - 1) + 1))
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _canonical_output(background, colors):
    result = np.full((6 * len(colors), 6), background, dtype=np.int8)
    rows, cols = np.nonzero(_RING_MASK)
    for index, color in enumerate(colors):
        result[index * 6 + rows, cols] = color
    return result


def _parse_canonical_output(output):
    assert output.shape[1] == 6 and output.shape[0] % 6 == 0
    colors = []
    backgrounds = set()
    for start in range(0, output.shape[0], 6):
        tile = output[start:start + 6]
        ring_values = np.unique(tile[_RING_MASK])
        assert len(ring_values) == 1
        colors.append(int(ring_values[0]))
        backgrounds.update(int(v) for v in tile[~_RING_MASK])
    assert len(backgrounds) == 1
    background = next(iter(backgrounds))
    assert background not in colors and len(colors) == len(set(colors))
    assert 2 <= len(colors) <= 5
    return background, tuple(colors)


def _candidate_translations(grid, color, background):
    """The complete observed fragment must fit one translated stencil."""
    height, width = grid.shape
    visible = {(int(r), int(c)) for r, c in np.argwhere(grid == color)}
    if len(visible) < 4:
        return []
    first_r, first_c = min(visible)
    results = []
    for mr, mc in _RING_CELLS:
        top, left = first_r - mr, first_c - mc
        cells = {(top + r, left + c) for r, c in _RING_CELLS
                 if 0 <= top + r < height and 0 <= left + c < width}
        if len(cells) < 8 or not visible.issubset(cells):
            continue
        if any(int(grid[r, c]) == background for r, c in cells):
            continue
        if (top, left) not in results:
            results.append((top, left))
    return results


def _unique_topological_order(edges):
    indegrees = [0] * len(edges)
    for targets in edges:
        for target in targets:
            indegrees[target] += 1
    remaining = set(range(len(edges)))
    order = []
    while remaining:
        choices = [index for index in remaining if indegrees[index] == 0]
        assert len(choices) == 1
        index = choices[0]
        order.append(index)
        remaining.remove(index)
        for target in edges[index]:
            indegrees[target] -= 1
    return order


def _solve_input(grid):
    """Recover a scene from visible colors, never from generation parameters."""
    values, counts = np.unique(grid, return_counts=True)
    modes = values[counts == counts.max()]
    assert len(modes) == 1 and int(counts.max()) > grid.size // 2
    background = int(modes[0])
    placements = {}
    distractors = []
    for raw_color in values:
        color = int(raw_color)
        if color == background:
            continue
        candidates = _candidate_translations(grid, color, background)
        if len(candidates) == 1:
            placements[color] = candidates[0]
        elif len(candidates) == 0:
            distractors.append(color)
        else:
            raise AssertionError("ambiguous stencil translation")
    assert 2 <= len(placements) <= 5

    colors = sorted(placements)
    color_index = {color: index for index, color in enumerate(colors)}
    height, width = grid.shape
    occupants = {}
    for index, color in enumerate(colors):
        top, left = placements[color]
        for dr, dc in _RING_CELLS:
            r, c = top + dr, left + dc
            if 0 <= r < height and 0 <= c < width:
                occupants.setdefault((r, c), []).append(index)
    edges = [set() for _ in colors]
    for (r, c), owners in occupants.items():
        visible = int(grid[r, c])
        assert visible in color_index
        front = color_index[visible]
        assert front in owners
        for other in owners:
            if other != front:
                edges[front].add(other)
    for color in colors:
        for r, c in np.argwhere(grid == color):
            assert (int(r), int(c)) in occupants
    order = _unique_topological_order(edges)
    ordered_colors = tuple(colors[index] for index in order)
    ordered_placements = tuple(placements[color] for color in ordered_colors)
    visible_counts = tuple(int(np.count_nonzero(grid == color)) for color in ordered_colors)
    clipped = any(top < 0 or left < 0 or top + 5 >= height or left + 5 >= width
                  for top, left in ordered_placements)
    edge_touch = any(top == 0 or left == 0 or top + 5 == height - 1
                     or left + 5 == width - 1 for top, left in ordered_placements)
    return _canonical_output(background, ordered_colors), {
        "background": background,
        "colors": ordered_colors,
        "distractors": tuple(sorted(distractors)),
        "placements": ordered_placements,
        "visible_counts": visible_counts,
        "clipped": clipped,
        "edge_touch": edge_touch,
    }


def _sample_geometry(rng, count, compact, clipping, square):
    offsets = [entry for entry in _OFFSETS
               if (entry[2] >= 3 if compact else
                   max(abs(entry[0]), abs(entry[1])) >= 3)]
    for _ in range(200):
        anchors = [(0, 0)]
        for index in range(1, count):
            order = rng.permutation(len(offsets))
            for choice in order:
                dr, dc, _ = offsets[int(choice)]
                anchor = (anchors[-1][0] + dr, anchors[-1][1] + dc)
                if anchor not in anchors:
                    anchors.append(anchor)
                    break
        min_r = min(r for r, _ in anchors)
        min_c = min(c for _, c in anchors)
        span_h = max(r + 6 for r, _ in anchors) - min_r
        span_w = max(c + 6 for _, c in anchors) - min_c
        margins = [int(rng.integers(0, 7)) for _ in range(4)]
        if clipping:
            margins[int(rng.integers(0, 4))] = -1
        top, bottom, left, right = margins
        height, width = span_h + top + bottom, span_w + left + right
        if square:
            side = max(height, width, 8)
            height = width = side
        if not (8 <= height <= 30 and 8 <= width <= 30):
            continue
        placed = [(r - min_r + top, c - min_c + left) for r, c in anchors]
        actually_clipped = any(r < 0 or c < 0 or r + 5 >= height or c + 5 >= width
                               for r, c in placed)
        if actually_clipped != clipping:
            continue
        return height, width, placed
    raise ValueError("could not sample geometry")


def _ring_count_plan(rng, train_count, test_count):
    mode = int(rng.integers(0, 4))
    if mode == 0:  # source-like shift
        train = [2 if i % 2 == 0 else 3 for i in range(train_count)]
        test = [4] * test_count
    elif mode == 1:  # equal complexity in both sections
        common = int(rng.integers(2, 5))
        train = [common] * train_count
        test = [common] * test_count
    elif mode == 2:  # some tests are easier than a demonstration
        train = [int(rng.choice((3, 4, 5), p=(0.3, 0.55, 0.15)))
                 for _ in range(train_count)]
        train[0] = 4
        test = [int(rng.integers(2, 4)) for _ in range(test_count)]
    else:
        train = [int(rng.choice((2, 3, 4, 5), p=(0.3, 0.4, 0.25, 0.05)))
                 for _ in range(train_count)]
        test = [int(rng.choice((2, 3, 4, 5), p=(0.2, 0.3, 0.4, 0.1)))
                for _ in range(test_count)]
    return train, test


def _pair_color_roles(rng, parameters, regime, index, count):
    palette = parameters["palette"]
    mode = parameters["color_mode"]
    if mode in ("stable", "source_like") and regime == "source_train":
        background = int(rng.choice(palette[:2]))
        pool = [int(v) for v in palette[2:7]]
        rings = [pool[int(i)] for i in rng.choice(len(pool), count, replace=False)]
        distractor_pool = [int(v) for v in palette[7:]]
    elif mode == "stable":
        background = int(rng.choice(palette[:2]))
        pool = [int(v) for v in palette[2:7]]
        rings = [pool[int(i)] for i in rng.choice(len(pool), count, replace=False)]
        distractor_pool = [int(v) for v in palette[7:]]
    elif mode == "source_like":
        background = int(rng.choice(palette[2:7]))
        special = [int(palette[0]), int(palette[7])]
        remainder = [int(v) for v in palette if int(v) not in special + [background]]
        rng.shuffle(remainder)
        rings = (special + remainder)[:count]
        distractor_pool = [int(v) for v in palette if int(v) not in rings + [background]]
    else:
        if regime == "source_train" and index == 1:
            background = parameters["first_ring"]
        else:
            background = int(rng.choice(palette))
        pool = [int(v) for v in palette if int(v) != background]
        rng.shuffle(pool)
        rings = pool[:count]
        distractor_pool = pool[count:]
        if regime == "source_train" and index == 0:
            parameters["first_ring"] = rings[0]
    return background, rings, distractor_pool


def _add_distractors(rng, grid, union, colors, count, anchors):
    if count == 0:
        return
    height, width = grid.shape
    available = {(int(r), int(c)) for r, c in np.argwhere(~union)}
    if len(available) < 16:
        raise ValueError("not enough space for distractors")
    for color in colors[:count]:
        amount = int(rng.integers(1, 9))
        preferred = set()
        for top, left in anchors:
            for dr in range(1, 5):
                for dc in range(1, 5):
                    preferred.add((top + dr, left + dc))
        preferred = list(preferred.intersection(available))
        nearby_count = min(len(preferred), amount, int(rng.integers(0, 4)))
        chosen = []
        if nearby_count:
            indices = rng.choice(len(preferred), nearby_count, replace=False)
            chosen.extend(preferred[int(i)] for i in indices)
        remaining = list(available.difference(chosen))
        indices = rng.choice(len(remaining), amount - len(chosen), replace=False)
        chosen.extend(remaining[int(i)] for i in indices)
        for r, c in chosen:
            grid[r, c] = color
        available.difference_update(chosen)


def generate_pair(rng, regime, parameters, index, count, noise):
    background, ring_colors, distractor_pool = _pair_color_roles(
        rng, parameters, regime, index, count
    )
    for attempt in range(500):
        compact = bool(rng.random() < 0.45)
        clipping = bool(rng.random() < 0.12)
        square = bool(rng.random() < 0.3)
        try:
            height, width, anchors = _sample_geometry(
                rng, count, compact, clipping, square
            )
        except ValueError:
            continue
        grid = np.full((height, width), background, dtype=np.int8)
        union = np.zeros((height, width), dtype=bool)
        for top, left in anchors:
            for dr, dc in _RING_CELLS:
                r, c = top + dr, left + dc
                if 0 <= r < height and 0 <= c < width:
                    union[r, c] = True
        for color, (top, left) in reversed(list(zip(ring_colors, anchors))):
            for dr, dc in _RING_CELLS:
                r, c = top + dr, left + dc
                if 0 <= r < height and 0 <= c < width:
                    grid[r, c] = color
        distractor_colors = list(distractor_pool)
        rng.shuffle(distractor_colors)
        try:
            _add_distractors(rng, grid, union, distractor_colors, noise, anchors)
        except ValueError:
            continue
        expected = _canonical_output(background, ring_colors)
        try:
            recovered, info = _solve_input(grid)
            assert np.array_equal(recovered, expected)
            assert info["clipped"] == clipping
            assert len(info["distractors"]) == noise
        except AssertionError:
            continue
        return grid, expected
    raise ValueError("could not generate an unambiguous pair")


def _grid_signature(grid):
    return grid.shape, grid.astype(np.int64, copy=False).tobytes()


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    signatures = set()
    train_noisy = False
    test_noisy = False
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            input_grid, output_grid = pair["input"], pair["output"]
            for grid in (input_grid, output_grid):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            signature = _grid_signature(input_grid)
            assert signature not in signatures
            signatures.add(signature)
            background, colors = _parse_canonical_output(output_grid)
            solved, info = _solve_input(input_grid)
            assert np.array_equal(solved, output_grid)
            assert background == info["background"] and colors == info["colors"]
            assert set(info["distractors"]).isdisjoint(colors)
            assert info["distractors"] == tuple(
                sorted(set(int(v) for v in np.unique(input_grid)) -
                       set(colors) - {background})
            )
            if section == "train":
                train_noisy |= bool(info["distractors"])
            else:
                test_noisy |= bool(info["distractors"])
    assert train_noisy or not test_noisy


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    parameters = {
        "palette": tuple(int(v) for v in rng.permutation(10)),
        "color_mode": ("stable", "source_like", "mixed")[int(rng.integers(0, 3))],
    }
    train_ring_counts, test_ring_counts = _ring_count_plan(rng, train_count, test_count)
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    for regime, section, counts in (
        ("source_train", "train", train_ring_counts),
        ("target_test", "test", test_ring_counts),
    ):
        for index, count in enumerate(counts):
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            if section == "train" and index == 0:
                noise = int(pair_rng.integers(1, 3))
            elif section == "train" and index == 1:
                noise = 0
            else:
                noise = int(pair_rng.choice((0, 1, 2), p=(0.35, 0.45, 0.2)))
            for _ in range(30):
                input_grid, output_grid = generate_pair(
                    pair_rng, regime, parameters, index, count, noise
                )
                signature = _grid_signature(input_grid)
                if signature not in seen_inputs:
                    seen_inputs.add(signature)
                    puzzle[section].append({"input": input_grid, "output": output_grid})
                    break
            else:
                raise ValueError("could not avoid a duplicate input")
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test():
    first = generate_puzzle(123, 5, 2)
    second = generate_puzzle(123, 5, 2)
    for section in ("train", "test"):
        for a, b in zip(first[section], second[section]):
            assert np.array_equal(a["input"], b["input"])
            assert np.array_equal(a["output"], b["output"])
    for arguments in ((1, None), (11, None), (None, 0), (2, 2), (3, 4)):
        try:
            generate_puzzle(12, *arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    modes = set()
    squares = 0
    noisy = 0
    five = 0
    for seed in range(16):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts = [len(pair["output"]) // 6 for pair in puzzle["train"]]
        test_counts = [len(pair["output"]) // 6 for pair in puzzle["test"]]
        if all(n <= 3 for n in train_counts) and all(n == 4 for n in test_counts):
            modes.add("source_like")
        if any(n in test_counts for n in train_counts):
            modes.add("equal")
        if min(test_counts) < max(train_counts):
            modes.add("simpler")
        for section in ("train", "test"):
            for pair in puzzle[section]:
                grid = pair["input"]
                squares += grid.shape[0] == grid.shape[1]
                _, info = _solve_input(grid)
                noisy += bool(info["distractors"])
                five += len(info["colors"]) == 5
    assert {"source_like", "equal", "simpler"} <= modes
    assert squares > 0 and noisy > 0 and five > 0
