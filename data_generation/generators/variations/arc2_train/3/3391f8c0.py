"""Reciprocal exchange of two visible, same-footprint ARC prototypes."""

import hashlib
import numpy as np


_FOOTPRINTS = ((2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 4), (3, 5))
_WEIGHTS = np.array([10, 15, 12, 30, 10, 10, 13], dtype=float) / 100
_SOURCE_DIGESTS = {
    "30f225d058fde4e903321e1260a830209d979a32aac4eed660e1733d86dba546",
    "c759fbbf195c23d2a73c21e3ebcf35d03bdacdf917435a78d5895a463e20cceb",
    "92e683cb0e45cc04c19d36a5603ad042c92114c0180a9478453c998e0390fefd",
    "ef56ce18cf6343d98929e4024bc7929177673186817d12a7bc7dcc871e2124b8",
    "bceb0981900d30757dc2dcfc2887eaa2444b656f49b3d7eacb67d654a5ccf4d8",
}


def _require_optional_count(name, value, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    train = _require_optional_count("num_train", num_train, 2, 10)
    test = _require_optional_count("num_test", num_test, 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if train is None and test is None:
        test = 1 + seed % 4
        train = max(test + 1, 2 + (seed * 5 + seed // 4) % 9)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _neighbors(row, col, height, width):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                rr, cc = row + dr, col + dc
                if 0 <= rr < height and 0 <= cc < width:
                    yield rr, cc


def _components(grid, color):
    height, width = grid.shape
    visited = np.zeros(grid.shape, dtype=bool)
    result = []
    for point in np.argwhere(grid == color):
        start = tuple(int(v) for v in point)
        if visited[start]:
            continue
        visited[start] = True
        todo = [start]
        cells = []
        while todo:
            row, col = todo.pop()
            cells.append((row, col))
            for rr, cc in _neighbors(row, col, height, width):
                if not visited[rr, cc] and grid[rr, cc] == color:
                    visited[rr, cc] = True
                    todo.append((rr, cc))
        row0 = min(r for r, _ in cells)
        col0 = min(c for _, c in cells)
        mask = np.zeros((max(r for r, _ in cells) - row0 + 1,
                         max(c for _, c in cells) - col0 + 1), dtype=bool)
        for r, c in cells:
            mask[r - row0, c - col0] = True
        result.append((row0, col0, mask))
    return result


def _connected(mask):
    return len(_components(mask.astype(np.int8), 1)) == 1


def _sample_mask(rng, shape):
    height, width = shape
    area = height * width
    minimum = 2 if area <= 4 else max(3, int(np.ceil(area * 0.28)))
    maximum = area if area <= 9 else area - 1
    for _ in range(2000):
        cells = int(rng.integers(minimum, maximum + 1))
        mask = np.zeros(area, dtype=bool)
        mask[rng.choice(area, size=cells, replace=False)] = True
        mask = mask.reshape(shape)
        if (mask[0].any() and mask[-1].any() and mask[:, 0].any()
                and mask[:, -1].any() and _connected(mask)):
            return mask
    raise RuntimeError("could not sample connected full-span mask")


def _sample_masks(rng, shape, regime):
    for _ in range(100):
        first = _sample_mask(rng, shape)
        second = _sample_mask(rng, shape)
        if not np.array_equal(first, second):
            return first, second
    raise RuntimeError("could not sample distinct prototypes")


def _sample_roles(rng, count):
    if count == 2:
        return [0, 1]
    roll = rng.random()
    if roll < 0.25:
        first = count // 2
    elif roll < 0.55:
        first = 1 if rng.random() < 0.5 else count - 1
    else:
        first = int(rng.integers(1, count))
    roles = [0] * first + [1] * (count - first)
    rng.shuffle(roles)
    return roles


def _overlap(a, b, height, width):
    return not (a[0] + height <= b[0] or b[0] + height <= a[0]
                or a[1] + width <= b[1] or b[1] + width <= a[1])


def _touching_same_role(a, b, mask):
    ar, ac = a
    br, bc = b
    # Bounding rectangles farther than one cell cannot have touching pixels.
    height, width = mask.shape
    if (ar + height < br or br + height < ar
            or ac + width < bc or bc + width < ac):
        return False
    points = np.argwhere(mask)
    for row1, col1 in points:
        for row2, col2 in points:
            if abs(ar + row1 - br - row2) <= 1 and abs(ac + col1 - bc - col2) <= 1:
                return True
    return False


def _fits(anchor, role, placed, masks):
    height, width = masks[0].shape
    for other_anchor, other_role in placed:
        if _overlap(anchor, other_anchor, height, width):
            return False
        if role == other_role and any(
            _touching_same_role(anchor, other_anchor, mask) for mask in masks
        ):
            return False
    return True


def _grid_anchors(rng, count, shape, style):
    height, width = shape
    if style == "row":
        rows, cols = 1, count
    elif style == "column":
        rows, cols = count, 1
    else:
        rows = int(rng.integers(2, min(count, 5) + 1))
        cols = int(np.ceil(count / rows))
        if rng.random() < 0.4:
            cols += 1
    row_gaps = [int(rng.integers(0, 4)) for _ in range(rows - 1)]
    col_gaps = [int(rng.integers(0, 4)) for _ in range(cols - 1)]
    row_positions = [0]
    col_positions = [0]
    for gap in row_gaps:
        row_positions.append(row_positions[-1] + height + gap)
    for gap in col_gaps:
        col_positions.append(col_positions[-1] + width + gap)
    if style in ("staggered", "triangle"):
        offsets = [int(rng.integers(0, width + 3)) for _ in range(rows)]
        if style == "triangle":
            offsets = [abs(i - (rows - 1) / 2) * (width + 1) for i in range(rows)]
            offsets = [int(x) for x in offsets]
    else:
        offsets = [0] * rows
    slots = [(row_positions[r], col_positions[c] + offsets[r])
             for r in range(rows) for c in range(cols)]
    if len(slots) < count:
        return None
    chosen = rng.choice(len(slots), size=count, replace=False)
    return [slots[int(i)] for i in chosen]


def _scatter_anchors(rng, count, shape, roles, masks):
    height, width = shape
    # Select independent board aspect ratios. Packing failures are retried.
    board_h = int(rng.integers(max(height + 2, 8), 31))
    board_w = int(rng.integers(max(width + 2, 8), 31))
    if board_h * board_w < count * height * width * 1.3:
        return None
    placed = []
    for role in roles:
        found = False
        for _ in range(350):
            anchor = (int(rng.integers(0, board_h - height + 1)),
                      int(rng.integers(0, board_w - width + 1)))
            if _fits(anchor, role, placed, masks):
                placed.append((anchor, role))
                found = True
                break
        if not found:
            return None
    return [anchor for anchor, _ in placed]


def _place(rng, roles, masks, style):
    count = len(roles)
    shape = masks[0].shape
    for _ in range(150):
        if style == "scatter":
            anchors = _scatter_anchors(rng, count, shape, roles, masks)
        else:
            anchors = _grid_anchors(rng, count, shape, style)
        if anchors is None:
            continue
        if any(not _fits(anchor, roles[i], list(zip(anchors[:i], roles[:i])), masks)
               for i, anchor in enumerate(anchors)):
            continue
        min_row = min(a[0] for a in anchors)
        min_col = min(a[1] for a in anchors)
        anchors = [(r - min_row, c - min_col) for r, c in anchors]
        occupied_h = max(r + shape[0] for r, _ in anchors)
        occupied_w = max(c + shape[1] for _, c in anchors)
        if occupied_h > 28 or occupied_w > 28:
            continue
        top = int(rng.integers(1, min(4, 29 - occupied_h) + 1))
        left = int(rng.integers(1, min(4, 29 - occupied_w) + 1))
        bottom = int(rng.integers(1, min(5, 30 - occupied_h - top) + 1))
        right = int(rng.integers(1, min(5, 30 - occupied_w - left) + 1))
        return [(r + top, c + left) for r, c in anchors], (top + occupied_h + bottom,
                                                             left + occupied_w + right)
    raise RuntimeError("could not place separate prototypes")


def _analyze(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    background = int(grid[0, 0])
    colors = tuple(int(x) for x in np.unique(grid) if x != background)
    assert len(colors) == 2, "two foreground roles required"
    masks = {}
    objects = []
    for color in colors:
        components = _components(grid, color)
        assert components
        reference = components[0][2]
        assert _connected(reference)
        for row, col, mask in components:
            assert mask.shape == reference.shape and np.array_equal(mask, reference)
            objects.append((color, row, col))
        masks[color] = reference
    assert masks[colors[0]].shape == masks[colors[1]].shape
    assert not np.array_equal(masks[colors[0]], masks[colors[1]])
    height, width = masks[colors[0]].shape
    for i, (_, row, col) in enumerate(objects):
        assert row > 0 and col > 0 and row + height < grid.shape[0] and col + width < grid.shape[1]
        for _, other_row, other_col in objects[:i]:
            assert not _overlap((row, col), (other_row, other_col), height, width)
    return background, colors, masks, objects


def _exchange(grid, analysis):
    background, colors, masks, objects = analysis
    result = np.full(grid.shape, background, dtype=np.int8)
    height, width = masks[colors[0]].shape
    for color, row, col in objects:
        other = colors[1] if color == colors[0] else colors[0]
        block = result[row:row + height, col:col + width]
        block[masks[other]] = other
    return result


def _pair_digest(pair):
    a, b = pair["input"], pair["output"]
    return hashlib.sha256(bytes(a.shape) + a.tobytes() + b.tobytes()).hexdigest()


def _pair_key(pair):
    a, b = pair["input"], pair["output"]
    return (a.shape, a.tobytes(), b.tobytes())


def _sample_pair(rng, regime, parameters, forced_shape=None):
    for _ in range(300):
        if forced_shape is None:
            weights = _WEIGHTS.copy()
            if regime == "target_test":
                weights *= np.array([0.6, 0.9, 0.9, 1.35, 1.1, 1.3, 1.4])
                weights /= weights.sum()
            shape = _FOOTPRINTS[int(rng.choice(len(_FOOTPRINTS), p=weights))]
        else:
            shape = forced_shape
        masks = _sample_masks(rng, shape, regime)
        maximum = 14 if shape[0] * shape[1] <= 9 else 9
        count = int(rng.integers(2, maximum + 1))
        if rng.random() < 0.18:
            count = 2
        roles = _sample_roles(rng, count)
        styles = ("regular", "staggered", "triangle", "scatter", "row", "column")
        style = styles[int(rng.choice(len(styles), p=[.23, .20, .12, .29, .08, .08]))]
        if ((style == "row" and count * shape[1] > 28) or
                (style == "column" and count * shape[0] > 28)):
            style = "staggered"
        try:
            anchors, board = _place(rng, roles, masks, style)
        except RuntimeError:
            continue
        background = parameters["background"]
        if parameters["stable_colors"]:
            colors = parameters["colors"]
        else:
            colors = tuple(int(x) for x in rng.choice(
                [x for x in range(10) if x != background], size=2, replace=False))
        inp = np.full(board, background, dtype=np.int8)
        for anchor, role in zip(anchors, roles):
            row, col = anchor
            height, width = shape
            block = inp[row:row + height, col:col + width]
            block[masks[role]] = colors[role]
        try:
            analysis = _analyze(inp)
            out = _exchange(inp, analysis)
            output_analysis = _analyze(out)
            assert np.array_equal(_exchange(out, output_analysis), inp)
            assert len(analysis[3]) == count
            assert {(r, c) for _, r, c in analysis[3]} == set(anchors)
            assert {(r, c) for _, r, c in output_analysis[3]} == set(anchors)
            pair = {"input": inp, "output": out}
            assert _pair_digest(pair) not in _SOURCE_DIGESTS
            return pair
        except AssertionError:
            continue
    raise RuntimeError("could not construct a valid pair")


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    seen_pairs = set()
    analyses = {"train": [], "test": []}
    puzzle_background = None
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert 0 <= int(grid.min()) and int(grid.max()) <= 9
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = (inp.shape, inp.tobytes())
            pkey = _pair_key(pair)
            assert key not in seen_inputs and pkey not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pkey)
            assert _pair_digest(pair) not in _SOURCE_DIGESTS
            analysis = _analyze(inp)
            if puzzle_background is None:
                puzzle_background = analysis[0]
            assert analysis[0] == puzzle_background
            assert np.array_equal(_exchange(inp, analysis), out)
            back = _analyze(out)
            assert np.array_equal(_exchange(out, back), inp)
            assert {(r, c) for _, r, c in back[3]} == {(r, c) for _, r, c in analysis[3]}
            assert analysis[0] == back[0] and analysis[1] == back[1]
            analyses[split].append(analysis)
    # Every training example exhibits both directions of the same exchange.
    # Each test input supplies both complete counterpart masks in its own colors.
    for split in ("train", "test"):
        for analysis in analyses[split]:
            assert len(analysis[3]) >= 2
            assert {c for c, _, _ in analysis[3]} == set(analysis[1])
    # The two regimes share the rule; tests draw more often from larger footprints.
    # There is no persistent color or shape dictionary to learn.


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    # Puzzle-level style is optional; the only invariant convention is the swap.
    background = 0 if rng.random() < .85 else int(rng.integers(1, 10))
    stable_colors = rng.random() < .18
    palette = tuple(int(x) for x in rng.choice(
        [x for x in range(10) if x != background], size=2, replace=False))
    parameters = {"background": background, "stable_colors": stable_colors, "colors": palette}
    puzzle = {"train": [], "test": []}
    seen = set()
    # One familiar footprint anchors the demonstrations; the remaining sizes
    # and all test sizes are sampled independently.
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            for _ in range(200):
                forced = (3, 3) if split == "train" and index == 0 else None
                pair = _sample_pair(rng, "source_train" if split == "train" else "target_test",
                                    parameters, forced_shape=forced)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not generate a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    def key(p):
        return tuple(_pair_key(pair) for split in ("train", "test") for pair in p[split])

    source_inputs = (
        '0000000000/0303030300/0030003000/0030003000/0000000000/0080000000/0888000000/0080000000/0000000000',
        '000000000000/022022022000/020020020000/000000000000/000022003000/000020033000/000000000000/000000000000/000000000000',
        '0000000000/0100100070/0010010700/0000000000/0100000000/0010000000/0000000000/0100000000/0010000000/0000000000/0000000000',
        '00000000000000000/00000000800000000/00000008880000000/00000800000800000/00008880008880000/00800001110000800/08880000100008880/00000000000000000/00000000000000000',
        '00000000000000/00505005050000/00050000500000/00555005550000/00000000000000/00666005050000/00606000500000/00006005550000/00000000000000/00000000000000/00000000000000',
    )
    source_digests = set()
    for source in source_inputs:
        inp = np.array([[int(cell) for cell in row] for row in source.split('/')], dtype=np.int8)
        out = _exchange(inp, _analyze(inp))
        source_digests.add(_pair_digest({"input": inp, "output": out}))
    assert source_digests == _SOURCE_DIGESTS

    assert key(generate_puzzle(12345)) == key(generate_puzzle(12345))
    assert key(generate_puzzle(54321, 7, 3)) == key(generate_puzzle(54321, 7, 3))
    for seed, train, test in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                              (1004, None, 1), (1005, None, 4),
                              (1006, 2, None), (1007, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(2000, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    counts = set()
    footprints = set()
    areas = set()
    arrangements = set()
    colors = set()
    balanced = unequal_area = diagonal = 0
    mixed_puzzles = 0
    size_puzzles = 0
    symmetric = nonzero_background = zero_foreground = 0
    tight = single_row = single_column = mixed_row = 0
    large_canvas = compact_canvas = large_footprint = 0
    count_range = set()
    previous = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert previous != key(puzzle)
        previous = key(puzzle)
        counts.add((len(puzzle["train"]), len(puzzle["test"])))
        palettes = set()
        puzzle_sizes = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                background, cs, masks, objects = _analyze(inp)
                assert np.array_equal(_exchange(inp, (background, cs, masks, objects)), pair["output"])
                palettes.add(cs)
                footprints.add(masks[cs[0]].shape)
                puzzle_sizes.add(masks[cs[0]].shape)
                areas.add((int(masks[cs[0]].sum()), int(masks[cs[1]].sum())))
                colors.add((background, cs))
                signature = (masks[cs[0]].shape, tuple(sorted((r, c) for _, r, c in objects)))
                arrangements.add(signature)
                role_counts = [sum(c == role for c, _, _ in objects) for role in cs]
                count_range.add(len(objects))
                balanced += role_counts[0] == role_counts[1]
                unequal_area += masks[cs[0]].sum() != masks[cs[1]].sum()
                diagonal += any(not _four_connected(mask) for mask in masks.values())
                symmetric += any(np.array_equal(mask, np.fliplr(mask)) or
                                 np.array_equal(mask, np.flipud(mask)) for mask in masks.values())
                nonzero_background += background != 0
                zero_foreground += 0 in cs
                large_canvas += 30 in inp.shape
                compact_canvas += inp.shape[0] <= 12 and inp.shape[1] <= 12
                large_footprint += masks[cs[0]].size > 9
                row_colors = {}
                for color, row, col in objects:
                    row_colors.setdefault(row, set()).add(color)
                single_row += len({row for _, row, _ in objects}) == 1
                single_column += len({col for _, _, col in objects}) == 1
                mixed_row += any(len(values) == 2 for values in row_colors.values())
                height, width = masks[cs[0]].shape
                tight += any(not (r1 + height < r2 or r2 + height < r1 or
                                  c1 + width < c2 or c2 + width < c1)
                             for i, (_, r1, c1) in enumerate(objects)
                             for _, r2, c2 in objects[i + 1:])
        mixed_puzzles += len(palettes) >= 2
        size_puzzles += len(puzzle_sizes) >= 2
    assert len(counts) >= 8 and len(footprints) == len(_FOOTPRINTS)
    assert len(areas) >= 15 and len(arrangements) >= 100 and len(colors) >= 20
    assert balanced >= 15 and unequal_area >= 50 and diagonal >= 30
    assert mixed_puzzles >= 25
    assert size_puzzles >= 30 and set(range(2, 15)).issubset(count_range)
    assert symmetric >= 30 and nonzero_background >= 10 and zero_foreground >= 2
    assert tight >= 20 and single_row >= 5 and single_column >= 5 and mixed_row >= 20
    assert large_canvas >= 10 and compact_canvas >= 5 and large_footprint >= 15


def _four_connected(mask):
    points = {tuple(int(x) for x in point) for point in np.argwhere(mask)}
    todo = [next(iter(points))]
    seen = {todo[0]}
    while todo:
        row, col = todo.pop()
        for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if neighbor in points and neighbor not in seen:
                seen.add(neighbor)
                todo.append(neighbor)
    return seen == points
