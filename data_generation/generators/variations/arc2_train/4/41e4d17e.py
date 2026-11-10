import numpy as np


_SOURCE_INPUT_ROWS = (
    (
        "888888888888888", "888888888888888", "888888888888888",
        "888111118888888", "888188818888888", "888188818888888",
        "888188818888888", "888111118888888", "888888888888888",
        "888888881111188", "888888881888188", "888888881888188",
        "888888881888188", "888888881111188", "888888888888888",
    ),
    (
        "888888888888888", "888111118888888", "888188818888888",
        "888188818888888", "888188818888888", "888111118888888",
        "888888888888888", "888888888888888", "888888888888888",
        "888888888888888", "888888888888888", "888888888888888",
        "888888888888888", "888888888888888", "888888888888888",
    ),
    (
        "888888888888888", "888888111118888", "888888188818888",
        "888888188818888", "888888188818888", "888888111118888",
        "888888888888888", "888888888888888", "888888888888888",
        "888111118888888", "888188818888888", "888188818888888",
        "888188818888888", "888111118888888", "888888888888888",
    ),
)

_INTERACTIONS = (
    "shared_row", "shared_col", "horizontal_interior", "horizontal_border",
    "vertical_interior", "vertical_border",
)
_MULTI_MODES = _INTERACTIONS + ("free_multi", "free_multi", "free_multi")


def _validate_optional_count(value, name, minimum, maximum):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _validate_optional_count(num_train, "num_train", 2, 10)
    num_test = _validate_optional_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_train = int(rng.integers(2, 11))
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _draw_frame(grid, top, left, side, color):
    bottom, right = top + side - 1, left + side - 1
    grid[top, left:right + 1] = color
    grid[bottom, left:right + 1] = color
    grid[top:bottom + 1, left] = color
    grid[top:bottom + 1, right] = color


def _construct_output(input_grid, placements, background, axis):
    result = input_grid.copy()
    for top, left, side in placements:
        row, col = top + side // 2, left + side // 2
        result[row, input_grid[row] == background] = axis
        result[input_grid[:, col] == background, col] = axis
    return result


def _odd_sides(limit):
    return tuple(range(3, min(limit, 29) + 1, 2))


def _single_layout(rng, mode):
    if mode == "anchor":
        side = int(rng.choice((5, 7, 9, 11)))
        height = int(rng.integers(side + 2, 31))
        width = int(rng.integers(side + 2, 31))
        if rng.random() < 0.08 and side <= 13:
            height = width = 15
        elif rng.random() < 0.06:
            height = width = 30
        top = int(rng.integers(1, height - side))
        left = int(rng.integers(1, width - side))
    else:
        if mode == "single_compact":
            height = int(rng.integers(3, 11))
            width = int(rng.integers(3, 11))
        elif mode == "single_large":
            side = int(rng.choice((11, 13, 15, 17, 19, 21, 23, 25, 27, 29)))
            height = int(rng.integers(side, 31))
            width = int(rng.integers(side, 31))
        else:
            height = int(rng.integers(3, 31))
            width = int(rng.integers(3, 31))
        if mode != "single_large":
            side = int(rng.choice(_odd_sides(min(height, width))))
        if rng.random() < 0.07 and side <= 15 and mode == "single":
            height = width = 15
        elif rng.random() < 0.08 and mode != "single_compact":
            height = width = 30
        top = int(rng.integers(0, height - side + 1))
        left = int(rng.integers(0, width - side + 1))
    return height, width, [(top, left, side)]


def _interaction_layout(rng, mode):
    horizontal = mode in ("shared_row", "horizontal_interior", "horizontal_border")
    sides = (3, 5, 7, 9, 11)
    first = int(rng.choice(sides))
    second_options = (5, 7, 9, 11) if mode.endswith("interior") else sides
    second = first if first in second_options and rng.random() < 0.28 else int(rng.choice(second_options))
    gap = int(rng.integers(1, 6))
    if mode.startswith("shared"):
        target_index = second // 2
    elif mode.endswith("border"):
        target_index = int(rng.choice((0, second - 1)))
    else:
        indices = [i for i in range(1, second - 1) if i != second // 2]
        target_index = int(rng.choice(indices))

    # The first frame's center axis meets the selected row of the second.
    a = (-(first // 2), 0, first)
    b = (-target_index, first + gap, second)
    if not horizontal:
        a = (a[1], a[0], a[2])
        b = (b[1], b[0], b[2])
    raw = (a, b)
    min_top = min(p[0] for p in raw)
    min_left = min(p[1] for p in raw)
    normalized = [(t - min_top, l - min_left, s) for t, l, s in raw]
    bound_h = max(t + s for t, _, s in normalized)
    bound_w = max(l + s for _, l, s in normalized)
    assert bound_h <= 30 and bound_w <= 30

    if rng.random() < 0.08 and bound_h <= 15 and bound_w <= 15:
        height = width = 15
    else:
        height = bound_h + int(rng.integers(0, min(12, 30 - bound_h) + 1))
        width = bound_w + int(rng.integers(0, min(12, 30 - bound_w) + 1))
    top_margin = int(rng.integers(0, height - bound_h + 1))
    left_margin = int(rng.integers(0, width - bound_w + 1))
    placements = [(t + top_margin, l + left_margin, s) for t, l, s in normalized]
    if rng.random() < 0.5:
        placements.reverse()
    return height, width, placements


def _separate(a, b):
    at, al, az = a
    bt, bl, bz = b
    return (at + az < bt or bt + bz < at or al + az < bl or bl + bz < al)


def _free_multi_layout(rng):
    count = int(rng.integers(2, 5))
    for _ in range(36):
        height = int(rng.integers(9, 31))
        width = int(rng.integers(9, 31))
        if rng.random() < 0.06:
            height = width = 30
        common_side = int(rng.choice((3, 5, 7))) if rng.random() < 0.25 else None
        placements = []
        for _ in range(count):
            placed = False
            for _ in range(100):
                side = common_side or int(rng.choice(_odd_sides(min(height, width, 11))))
                top = int(rng.integers(0, height - side + 1))
                left = int(rng.integers(0, width - side + 1))
                candidate = (top, left, side)
                if all(_separate(candidate, old) for old in placements):
                    placements.append(candidate)
                    placed = True
                    break
            if not placed:
                break
        if len(placements) == count:
            return height, width, placements
    # A bounded construction for unusually crowded random proposals.
    return 7, 4 * count - 1, [(2, 4 * i, 3) for i in range(count)]


def _generate_pair(rng, axis, mode):
    if mode in ("anchor", "single", "single_compact", "single_large"):
        height, width, placements = _single_layout(rng, mode)
    elif mode == "free_multi":
        height, width, placements = _free_multi_layout(rng)
    elif mode in _INTERACTIONS:
        height, width, placements = _interaction_layout(rng, mode)
    else:
        raise ValueError(f"unknown layout mode: {mode}")

    available = [color for color in range(10) if color != axis]
    background = int(rng.choice(available))
    structural = [color for color in available if color != background]
    if len(placements) == 1 or rng.random() < 0.48:
        colors = [int(rng.choice(structural))] * len(placements)
    else:
        palette_size = int(rng.integers(2, min(3, len(placements)) + 1))
        palette = [int(x) for x in rng.choice(structural, size=palette_size, replace=False)]
        colors = palette + [int(rng.choice(palette)) for _ in range(len(placements) - palette_size)]
        rng.shuffle(colors)

    input_grid = np.full((height, width), background, dtype=np.int8)
    for (top, left, side), color in zip(placements, colors):
        _draw_frame(input_grid, top, left, side, color)
    output_grid = _construct_output(input_grid, placements, background, axis)
    return input_grid, output_grid


def _grid_key(grid):
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


def _source_input_arrays():
    return [np.asarray([[int(x) for x in row] for row in rows], dtype=np.int8)
            for rows in _SOURCE_INPUT_ROWS]


def _components(points):
    remaining = set(points)
    components = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = {start}
        while stack:
            row, col = stack.pop()
            for neighbor in ((row - 1, col), (row + 1, col),
                             (row, col - 1), (row, col + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _frames_for_background(grid, background):
    foreground = {(int(r), int(c)) for r, c in np.argwhere(grid != background)}
    if not foreground:
        return None
    frames = []
    for component in _components(foreground):
        top = min(r for r, _ in component)
        bottom = max(r for r, _ in component)
        left = min(c for _, c in component)
        right = max(c for _, c in component)
        side = bottom - top + 1
        if side < 3 or side % 2 != 1 or right - left + 1 != side:
            return None
        expected = {(r, c) for r in range(top, bottom + 1)
                    for c in range(left, right + 1)
                    if r in (top, bottom) or c in (left, right)}
        if component != expected:
            return None
        frame_colors = {int(grid[r, c]) for r, c in component}
        if len(frame_colors) != 1:
            return None
        if np.any(grid[top + 1:bottom, left + 1:right] != background):
            return None
        frames.append((top, left, side, top + side // 2,
                       left + side // 2, frame_colors.pop()))
    frames.sort(key=lambda frame: (frame[0], frame[1]))
    if any(not _separate(a[:3], b[:3])
           for i, a in enumerate(frames) for b in frames[i + 1:]):
        return None
    return frames


def _analyze_input(grid):
    candidates = []
    for color in np.unique(grid):
        background = int(color)
        frames = _frames_for_background(grid, background)
        if frames is not None:
            candidates.append((background, frames))
    assert len(candidates) == 1, "input must reveal one background and separate hollow squares"
    return candidates[0]


def _expected_output(input_grid, background, axis, frames):
    result = input_grid.copy()
    for _, _, _, row, col, _ in frames:
        result[row, input_grid[row] == background] = axis
        result[input_grid[:, col] == background, col] = axis
    return result


def _assert_pair(input_grid, output_grid, axis):
    background, frames = _analyze_input(input_grid)
    assert axis not in set(int(c) for c in np.unique(input_grid))
    assert output_grid.shape == input_grid.shape
    assert np.array_equal(output_grid, _expected_output(input_grid, background, axis, frames)), (
        "output must project each center axis through background only"
    )
    assert np.any(input_grid != output_grid)
    for top, left, side, row, col, color in frames:
        assert input_grid[row, col] == background
        assert output_grid[row, col] == axis
        for r in range(top, top + side):
            for c in range(left, left + side):
                if input_grid[r, c] == color:
                    assert output_grid[r, c] == color
    return background, frames


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    source_inputs = _source_input_arrays()
    seen_inputs = set()
    seen_pairs = set()
    puzzle_axis = None
    train_counts = set()
    anchor_seen = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            input_key, output_key = _grid_key(inp), _grid_key(out)
            assert input_key not in seen_inputs
            assert (input_key, output_key) not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add((input_key, output_key))
            assert all(not np.array_equal(inp, source) for source in source_inputs)
            new_colors = set(int(x) for x in np.unique(out)) - set(int(x) for x in np.unique(inp))
            assert len(new_colors) == 1, "each pair must reveal the output-only axis color"
            axis = new_colors.pop()
            if puzzle_axis is None:
                puzzle_axis = axis
            assert axis == puzzle_axis
            background, frames = _assert_pair(inp, out, axis)
            if split == "train":
                train_counts.add(len(frames))
                if len(frames) == 1:
                    top, left, side, _, _, _ = frames[0]
                    if (side >= 5 and top > 0 and left > 0 and
                            top + side < inp.shape[0] and left + side < inp.shape[1]):
                        anchor_seen = True
            assert background != axis
    assert 1 in train_counts and any(n >= 2 for n in train_counts)
    assert anchor_seen, "training must show a full cross beyond a frame"


def _puzzles_equal(first, second):
    return all(len(first[split]) == len(second[split]) and
               all(np.array_equal(a[role], b[role])
                   for a, b in zip(first[split], second[split])
                   for role in ("input", "output"))
               for split in ("train", "test"))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    axis = int(rng.integers(0, 10))
    # Training always contains an uncomplicated global projection and a
    # multi-frame superposition. Other arrangements are independent draws.
    train_modes = ["anchor", str(rng.choice(_MULTI_MODES))]
    optional_modes = _MULTI_MODES + ("single", "single_compact", "single_large")
    train_modes.extend(str(rng.choice(optional_modes)) for _ in range(train_count - 2))
    rng.shuffle(train_modes)
    test_modes = [str(rng.choice(_MULTI_MODES + ("single_compact", "single_large", "single_large")))
                  for _ in range(test_count)]
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    for split, modes in (("train", train_modes), ("test", test_modes)):
        for mode in modes:
            for _ in range(40):
                pair_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                pair_rng = np.random.default_rng(pair_seed)
                inp, out = _generate_pair(pair_rng, axis, mode)
                key = _grid_key(inp)
                if key not in seen_inputs and all(not np.array_equal(inp, source)
                                                  for source in _source_input_arrays()):
                    seen_inputs.add(key)
                    puzzle[split].append({"input": inp, "output": out})
                    break
            else:
                raise ValueError("could not construct distinct pairs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_metrics(pair):
    inp = pair["input"]
    background, frames = _analyze_input(inp)
    rows = [f[3] for f in frames]
    cols = [f[4] for f in frames]
    horizontal_interior = horizontal_border = False
    vertical_interior = vertical_border = False
    for i, first in enumerate(frames):
        for j, second in enumerate(frames):
            if i == j:
                continue
            t, l, s = second[:3]
            if l > first[1] + first[2] - 1 or first[1] > l + s - 1:
                horizontal_interior |= t < first[3] < t + s - 1
                horizontal_border |= first[3] in (t, t + s - 1)
            if t > first[0] + first[2] - 1 or first[0] > t + s - 1:
                vertical_interior |= l < first[4] < l + s - 1
                vertical_border |= first[4] in (l, l + s - 1)
    layout = "single" if len(frames) == 1 else "many"
    diagonal = None
    if len(frames) == 2:
        first, second = sorted(frames, key=lambda frame: frame[3])
        if first[0] + first[2] <= second[0]:
            layout = "vertical"
            diagonal = ("down_right" if first[4] < second[4] else
                        "down_left" if first[4] > second[4] else "aligned")
        else:
            layout = "horizontal_overlap"
    return {
        "shape": tuple(inp.shape), "count": len(frames),
        "sides": tuple(sorted(f[2] for f in frames)),
        "background": background,
        "frame_colors": len({f[5] for f in frames}),
        "edge": any(t == 0 or l == 0 or t + s == inp.shape[0] or l + s == inp.shape[1]
                    for t, l, s, _, _, _ in frames),
        "shared_row": len(rows) != len(set(rows)),
        "shared_col": len(cols) != len(set(cols)),
        "horizontal_interior": horizontal_interior,
        "horizontal_border": horizontal_border,
        "vertical_interior": vertical_interior,
        "vertical_border": vertical_border,
        "layout": layout, "diagonal": diagonal,
    }


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    metrics = [_pair_metrics(pair) for puzzle in puzzles
               for split in ("train", "test") for pair in puzzle[split]]
    assert len({m["shape"] for m in metrics}) >= 80
    assert len({m["background"] for m in metrics}) == 9 or len({m["background"] for m in metrics}) == 10
    assert {1, 2, 3, 4}.issubset({m["count"] for m in metrics})
    assert len({s for m in metrics for s in m["sides"]}) >= 8
    assert any(len(set(m["sides"])) > 1 for m in metrics)
    assert any(m["frame_colors"] > 1 for m in metrics)
    assert any(m["edge"] for m in metrics)
    assert any(max(m["shape"]) <= 10 for m in metrics)
    assert any(m["shape"] == (15, 15) for m in metrics)
    assert any(m["shape"] == (30, 30) for m in metrics)
    assert any(m["count"] > 1 and len(set(m["sides"])) == 1 for m in metrics)
    assert any(m["count"] > 1 and len(set(m["sides"])) > 1 and
               m["frame_colors"] > 1 and m["edge"] for m in metrics)
    for name in ("shared_row", "shared_col", "horizontal_interior", "horizontal_border",
                 "vertical_interior", "vertical_border"):
        assert any(m[name] for m in metrics), f"missing {name}"
    source_shift = reverse_shift = same_diagonal = False
    test_single = test_many = False
    for puzzle in puzzles:
        local = [_pair_metrics(pair) for split in ("train", "test")
                 for pair in puzzle[split]]
        assert len({m["background"] for m in local}) >= 2
        assert len({side for m in local for side in m["sides"]}) >= 2
        assert len({m["count"] for m in local}) >= 2
        assert len({m["shape"] for m in local}) >= 2
        train_diagonals = {_pair_metrics(pair)["diagonal"] for pair in puzzle["train"]}
        test_metrics = [_pair_metrics(pair) for pair in puzzle["test"]]
        test_diagonals = {m["diagonal"] for m in test_metrics}
        source_shift |= "down_right" in train_diagonals and "down_left" in test_diagonals
        reverse_shift |= "down_left" in train_diagonals and "down_right" in test_diagonals
        same_diagonal |= bool((train_diagonals & test_diagonals) - {None})
        test_single |= any(m["count"] == 1 for m in test_metrics)
        test_many |= any(m["count"] >= 3 for m in test_metrics)
    assert source_shift and reverse_shift and same_diagonal
    assert test_single and test_many
    assert len({_grid_key(pair["input"]) for puzzle in puzzles
                for split in ("train", "test") for pair in puzzle[split]}) > 150

    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4),
                                           (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(731 + index, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    first = generate_puzzle(991, 6, 2)
    assert _puzzles_equal(first, generate_puzzle(991, 6, 2))
    assert not _puzzles_equal(first, generate_puzzle(992, 6, 2))
    for args in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4), (2, 4)):
        try:
            generate_puzzle(1300, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {args}")
