"""Endpoint equality puzzles with varied colors, geometry, and density."""

import numpy as np


_SOURCE_PAIRS_TEXT = {
    (
        "0000000000/8000000008/0000000000/4000000002/0000000000/"
        "3000000004/0000000000/1000000001/0000000000/2000000003",
        "0000000000/8888888888/0000000000/4000000002/0000000000/"
        "3000000004/0000000000/1111111111/0000000000/2000000003",
    ),
    (
        "0000000000/9000000006/0000000000/8000000009/0000000000/"
        "4000000004/0000000000/6000000008/0000000000/0000000000",
        "0000000000/9000000006/0000000000/8000000009/0000000000/"
        "4444444444/0000000000/6000000008/0000000000/0000000000",
    ),
    (
        "0000000000/2000000008/0000000000/3000000004/0000000000/"
        "5000000003/0000000000/8000000002/0000000000/0000000000",
        "0000000000/2000000008/0000000000/3000000004/0000000000/"
        "5000000003/0000000000/8000000002/0000000000/0000000000",
    ),
    (
        "0000000000/4000000002/0000000000/3000000003/0000000000/"
        "2000000009/0000000000/6000000006/0000000000/9000000004",
        "0000000000/4000000002/0000000000/3333333333/0000000000/"
        "2000000009/0000000000/6666666666/0000000000/9000000004",
    ),
}


def _key(grid):
    return (grid.shape, tuple(int(x) for x in grid.ravel()))


def _text(grid):
    return "/".join("".join(str(int(x)) for x in row) for row in grid)


def _count_arg(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    n = _count_arg("num_train", num_train, 2, 10)
    m = _count_arg("num_test", num_test, 1, 4)
    if n is None and m is None:
        m = int(rng.integers(1, 5))
        n = int(rng.integers(max(2, m + 1), 11))
    elif n is None:
        n = int(rng.integers(max(2, m + 1), 11))
    elif m is None:
        m = int(rng.integers(1, min(4, n - 1) + 1))
    if n <= m:
        raise ValueError("num_train must exceed num_test")
    return n, m


def _read_segments(grid, background, orientation):
    view = grid if orientation == "h" else grid.T
    segments = []
    for line_index, line in enumerate(view):
        positions = np.flatnonzero(line != background)
        if not len(positions):
            continue
        if len(positions) != 2 or positions[1] - positions[0] < 2:
            return None
        first, last = (int(positions[0]), int(positions[1]))
        if not np.all(line[first + 1:last] == background):
            return None
        segments.append((line_index, first, last, int(line[first]), int(line[last])))
    return segments or None


def _analyze(grid):
    counts = np.bincount(grid.ravel(), minlength=10)
    background = int(np.argmax(counts))
    candidates = []
    for orientation in ("h", "v"):
        segments = _read_segments(grid, background, orientation)
        if segments is not None:
            candidates.append((orientation, segments))
    if len(candidates) != 1:
        return None
    return background, candidates[0][0], candidates[0][1]


def _complete(grid, background, orientation, segments):
    output = grid.copy()
    view = output if orientation == "h" else output.T
    for line, first, last, left, right in segments:
        if left == right:
            view[line, first:last + 1] = left
    return output


def _axis_size(rng, minimum=1):
    bands = ((1, 5), (6, 12), (13, 22), (23, 30))
    low, high = bands[int(rng.integers(0, len(bands)))]
    low = max(low, minimum)
    if low > high:
        low, high = minimum, max(minimum, 12)
    return int(rng.integers(low, high + 1))


def _cross_size(rng):
    bands = ((5, 7), (8, 13), (14, 21), (22, 30))
    low, high = bands[int(rng.integers(0, len(bands)))]
    return int(rng.integers(low, high + 1))


def _occupied_count(rng, axis):
    kind = int(rng.integers(0, 6))
    if kind == 0:
        return 1
    if kind == 1:
        return min(2, axis)
    if kind == 2:
        return axis
    if kind == 3:
        return max(1, int(np.ceil(axis * rng.uniform(0.65, 0.9))))
    if kind == 4:
        return max(1, int(np.ceil(axis * rng.uniform(0.2, 0.45))))
    return int(rng.integers(1, axis + 1))


def _positions(rng, axis, count):
    if count == axis:
        return np.arange(axis)
    style = int(rng.integers(0, 4))
    if style == 0:
        return np.sort(rng.choice(axis, size=count, replace=False))
    if style == 1:
        start = int(rng.integers(0, axis - count + 1))
        return np.arange(start, start + count)
    if style == 2 and axis >= 2 * count - 1:
        base = np.sort(rng.choice(axis - count + 1, size=count, replace=False))
        return base + np.arange(count)
    if style == 3 and count > 1:
        start = int(rng.integers(0, axis - count + 1))
        end = int(rng.integers(start + count - 1, axis))
        proposed = np.round(np.linspace(start, end, count)).astype(int)
        if len(set(proposed)) == count:
            return proposed
    return np.sort(rng.choice(axis, size=count, replace=False))


def _match_count(rng, count):
    kind = int(rng.integers(0, 5))
    if kind == 0:
        return 0
    if kind == 1:
        return count
    if kind == 2:
        return min(1, count)
    return int(rng.integers(0, count + 1))


def _palette(rng, available, required=()):
    available = tuple(int(x) for x in available)
    required = tuple(dict.fromkeys(int(x) for x in required))
    minimum = max(2, len(required))
    size = int(rng.integers(minimum, len(available) + 1))
    remaining = [x for x in available if x not in required]
    extras = rng.choice(remaining, size=size - len(required), replace=False)
    return required + tuple(int(x) for x in extras)


def _random_specs(rng, count, palette, match_count, forced_matches=(),
                  forced_unequal=None, forbidden_match=None):
    matched = [int(x) for x in forced_matches]
    match_palette = [x for x in palette if x != forbidden_match]
    while len(matched) < match_count:
        if matched and rng.random() < 0.38:
            matched.append(int(rng.choice(matched)))
        else:
            matched.append(int(rng.choice(match_palette)))
    specs = [(color, color) for color in matched]
    unequal_count = count - match_count
    if unequal_count and forced_unequal is not None:
        other = int(rng.choice([x for x in palette if x != forced_unequal]))
        specs.append((int(forced_unequal), other))
        unequal_count -= 1
    for _ in range(unequal_count):
        colors = rng.choice(palette, size=2, replace=False)
        specs.append((int(colors[0]), int(colors[1])))
    return [specs[int(i)] for i in rng.permutation(len(specs))]


def _make_pair(rng, orientation, background, specs, axis_size=None,
               require_inset=False):
    count = len(specs)
    axis = _axis_size(rng, count) if axis_size is None else axis_size
    cross = _cross_size(rng)
    if count <= 5 and rng.random() < 0.12:
        axis, cross = 10, 10
    positions = _positions(rng, axis, count)
    grid = np.full((axis, cross), background, dtype=np.int8)
    for index, (line, (left, right)) in enumerate(zip(positions, specs)):
        if index == 0 and require_inset:
            first = int(rng.integers(1, cross - 2))
            last = int(rng.integers(first + 2, cross))
        elif rng.random() < 0.42:
            first, last = 0, cross - 1
        else:
            first = int(rng.integers(0, cross - 2))
            last = int(rng.integers(first + 2, cross))
        grid[int(line), first] = left
        grid[int(line), last] = right
    if orientation == "v":
        grid = grid.T.copy()
    analysis = _analyze(grid)
    if analysis is None or analysis[1] != orientation:
        return None
    return grid, _complete(grid, *analysis)


def _sample_pair(rng, orientation, background, specs, axis_size=None,
                 require_inset=False):
    for _ in range(60):
        pair = _make_pair(rng, orientation, background, specs,
                          axis_size, require_inset)
        if pair is not None:
            return pair
    # This placement has a unique axis even for two occupied lines.
    axis = axis_size if axis_size is not None else max(3, len(specs))
    axis = max(axis, len(specs))
    grid = np.full((axis, 5), background, dtype=np.int8)
    for index, (left, right) in enumerate(specs):
        first, last = (1, 3) if index == 0 else (0, 4)
        grid[index, first] = left
        grid[index, last] = right
    if orientation == "v":
        grid = grid.T.copy()
    analysis = _analyze(grid)
    assert analysis is not None and analysis[1] == orientation
    return grid, _complete(grid, *analysis)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)

    # The shared rule is color equality. This pool bounds the colors available
    # within a puzzle; each pair chooses its own subset and background.
    pool_size = int(rng.integers(2, 9))
    pool = tuple(int(x) for x in rng.choice(10, size=pool_size, replace=False))
    if pool_size >= 4 and rng.random() < 0.7:
        train_pool_size = int(rng.integers(2, pool_size - 1))
    else:
        train_pool_size = pool_size
    train_pool = pool[:train_pool_size]
    new_pool = pool[train_pool_size:]
    backgrounds = tuple(x for x in range(10) if x not in pool)
    background_plan = [int(rng.choice(backgrounds))]
    background_plan.append(int(rng.choice([x for x in backgrounds if x != background_plan[0]])))
    background_plan.extend(int(rng.choice(backgrounds)) for _ in range(train_count + test_count - 2))

    style = int(rng.integers(0, 5))
    if style == 0:
        orientations = ["h"] * (train_count + test_count)
    elif style == 1:
        orientations = ["v"] * (train_count + test_count)
    elif style == 2:
        first = "h" if rng.random() < 0.5 else "v"
        orientations = [first, "v" if first == "h" else "h"]
        orientations.extend(("h" if rng.random() < 0.5 else "v")
                            for _ in range(train_count + test_count - 2))
    else:
        train_axis, test_axis = ("h", "v") if style == 3 else ("v", "h")
        orientations = [train_axis] * train_count + [test_axis] * test_count

    anchor_index = int(rng.integers(0, 2))
    a, b = train_pool[:2]
    shift = train_pool[2] if len(train_pool) >= 3 else None
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    train_matched = set()
    train_unequal = set()

    for index in range(train_count + test_count):
        regime = "train" if index < train_count else "test"
        background = background_plan[index]
        orientation = orientations[index]
        axis_size = None
        for _ in range(100):
            if regime == "train" and index == anchor_index:
                # Two successful colors and a failure sharing those colors
                # identify equality without a color dictionary.
                specs = [(a, a), (b, b), (a, b)]
                if shift is not None:
                    specs.append((shift, a))
                if rng.random() < 0.45:
                    specs.append((a, a))
                specs = [specs[int(i)] for i in rng.permutation(len(specs))]
            else:
                if regime == "train":
                    available = train_pool
                    forced_unequal = None
                    if index < 2 and shift is not None and rng.random() < 0.5:
                        forced_unequal = shift
                    required = (forced_unequal,) if forced_unequal is not None else ()
                    palette = _palette(rng, available, required)
                else:
                    mode = int(rng.integers(0, 6))
                    if mode == 0:
                        available = tuple(x for x in train_pool if x in train_matched)
                        if len(available) < 2:
                            available = train_pool
                    elif mode == 1 and shift is not None and shift in train_unequal:
                        available = train_pool
                    elif mode == 2 and new_pool:
                        available = tuple(train_pool[:1]) + new_pool
                    elif mode == 3 and len(new_pool) >= 2:
                        available = new_pool
                    else:
                        available = pool
                    if len(available) < 2:
                        available = pool
                    forced_unequal = None
                    required = ()
                    if mode == 0:
                        required = (a,)
                    elif mode == 1 and shift in available and shift in train_unequal:
                        required = (shift,)
                    elif mode == 2 and new_pool:
                        required = (a, new_pool[0])
                    palette = _palette(rng, available, required)

                axis = _axis_size(rng)
                count = _occupied_count(rng, axis)
                axis_size = axis
                matches = _match_count(rng, count)
                forced_matches = ()
                if forced_unequal is not None and matches == count:
                    matches -= 1
                if regime == "test":
                    if mode == 4:
                        matches = 0
                    elif mode == 2 and new_pool:
                        axis_size = max(axis_size, 2)
                        count = max(count, 2)
                        matches = max(matches, 2)
                        forced_matches = (a, new_pool[0])
                    elif mode == 1 and shift in palette and shift in train_unequal:
                        matches = max(matches, 1)
                        forced_matches = (shift,)
                    elif mode == 0:
                        matches = max(matches, 1)
                        forced_matches = (a,)
                    elif mode == 3 and len(new_pool) >= 2:
                        matches = max(matches, 1)
                        forced_matches = (int(rng.choice(new_pool)),)
                specs = _random_specs(rng, count, palette, matches,
                                      forced_matches, forced_unequal,
                                      shift if regime == "train" else None)
            input_grid, output_grid = _sample_pair(
                rng, orientation, background, specs, axis_size,
                regime == "train" and index == anchor_index
            )
            key = _key(input_grid)
            if key not in used_inputs and (_text(input_grid), _text(output_grid)) not in _SOURCE_PAIRS_TEXT:
                break
        else:
            raise RuntimeError("could not construct a unique pair")
        used_inputs.add(key)
        puzzle[regime].append({"input": input_grid, "output": output_grid})
        if regime == "train":
            analysis = _analyze(input_grid)
            assert analysis is not None
            for _, _, _, left, right in analysis[2]:
                if left == right:
                    train_matched.add(left)
                else:
                    train_unequal.update((left, right))

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    keys = set()
    matched, unequal, train_orientations = set(), set(), set()
    test_orientations = set()
    all_backgrounds = set()
    train_has_inset = False
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in keys
            keys.add(key)
            assert (_text(inp), _text(out)) not in _SOURCE_PAIRS_TEXT
            analysis = _analyze(inp)
            assert analysis is not None
            background, orientation, segments = analysis
            assert np.array_equal(out, _complete(inp, *analysis))
            all_backgrounds.add(background)
            if regime == "train":
                train_orientations.add(orientation)
                cross = inp.shape[1] if orientation == "h" else inp.shape[0]
                train_has_inset |= any(last - first + 1 < cross
                                       for _, first, last, _, _ in segments)
                for _, _, _, left, right in segments:
                    if left == right:
                        matched.add(left)
                    else:
                        unequal.update((left, right))
            else:
                test_orientations.add(orientation)
    assert len(matched) >= 2
    assert unequal
    assert matched & unequal
    assert train_has_inset
    # Axis is recoverable from each grid's unique pairing. Rotated test
    # grids may therefore transfer the same demonstrated equality rule.
    assert len(all_backgrounds) >= 2


def _same_puzzle(a, b):
    return all(
        len(a[regime]) == len(b[regime])
        and all(
            np.array_equal(x["input"], y["input"])
            and np.array_equal(x["output"], y["output"])
            for x, y in zip(a[regime], b[regime])
        )
        for regime in ("train", "test")
    )


def self_test() -> None:
    for n, m in ((2, 1), (10, 1), (10, 4), (None, 1),
                 (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(1000 + (n or 0) * 10 + (m or 0), n, m)
        validate_generated_puzzle(puzzle)
        if n is not None:
            assert len(puzzle["train"]) == n
        if m is not None:
            assert len(puzzle["test"]) == m
    for n, m in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4), (True, 1)):
        try:
            generate_puzzle(2, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count request accepted")

    summaries = []
    puzzle_keys = set()
    train_counts, test_counts = set(), set()
    puzzle_features = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signature = tuple(
            _key(pair["input"])
            for regime in ("train", "test")
            for pair in puzzle[regime]
        )
        puzzle_keys.add(signature)
        train_colors = set()
        train_matched = set()
        train_unequal = set()
        for pair in puzzle["train"]:
            _, _, segments = _analyze(pair["input"])
            for _, _, _, left, right in segments:
                train_colors.update((left, right))
                if left == right:
                    train_matched.add(left)
                else:
                    train_unequal.update((left, right))
        backgrounds, puzzle_markers, pair_palettes = set(), set(), set()
        train_axes, test_axes = set(), set()
        test_roles = set()
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                inp = pair["input"]
                background, orientation, segments = _analyze(inp)
                backgrounds.add(background)
                (train_axes if regime == "train" else test_axes).add(orientation)
                # Reconstruct from visible input cells alone.
                predicted = inp.copy()
                view = predicted if orientation == "h" else predicted.T
                for line, first, last, left, right in segments:
                    if left == right:
                        for position in range(first + 1, last):
                            view[line, position] = left
                assert np.array_equal(predicted, pair["output"])
                colors = {x for _, _, _, a, b in segments for x in (a, b)}
                matching = [a for _, _, _, a, b in segments if a == b]
                puzzle_markers.update(colors)
                pair_palettes.add(frozenset(colors))
                if regime == "test":
                    matching_set = set(matching)
                    if colors.isdisjoint(train_colors):
                        test_roles.add("all_new")
                    if matching_set <= train_matched and matching_set:
                        test_roles.add("familiar")
                    if matching_set & (train_unequal - train_matched):
                        test_roles.add("shift")
                    if matching_set & train_colors and matching_set - train_colors:
                        test_roles.add("mixed")
                index_positions = [line for line, _, _, _, _ in segments]
                lengths = [last - first + 1 for _, first, last, _, _ in segments]
                summaries.append((
                    seed, regime, background, orientation, inp.shape,
                    len(segments), len(matching), len(set(matching)) < len(matching),
                    len(segments) > 1 and np.any(np.diff(index_positions) == 1),
                    len(segments) == (inp.shape[0] if orientation == "h" else inp.shape[1]),
                    any(length == (inp.shape[1] if orientation == "h" else inp.shape[0])
                        for length in lengths),
                    any(length < (inp.shape[1] if orientation == "h" else inp.shape[0])
                        for length in lengths),
                    len(colors), bool(colors - train_colors) if regime == "test" else False,
                    bool(set(matching) & train_matched) if regime == "test" else False,
                ))
        puzzle_features.append((len(puzzle_markers), len(backgrounds),
                                len(pair_palettes), train_axes, test_axes,
                                test_roles))
    assert len(puzzle_keys) >= 40
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len({x[2] for x in summaries}) >= 5
    assert {x[3] for x in summaries} == {"h", "v"}
    assert len({x[4] for x in summaries}) >= 60
    assert any(x[5] == 1 for x in summaries)
    assert any(x[5] >= 7 for x in summaries)
    assert any(x[6] == 0 and x[1] == "test" for x in summaries)
    assert any(x[6] == x[5] and x[1] == "train" for x in summaries)
    assert any(x[7] for x in summaries)
    assert any(x[8] for x in summaries)
    assert any(x[9] for x in summaries)
    assert any(x[10] for x in summaries)
    assert any(x[11] for x in summaries)
    assert any(x[13] for x in summaries)
    assert any(x[14] for x in summaries)
    assert any(size <= 3 for size, *_ in puzzle_features)
    assert any(size >= 6 for size, *_ in puzzle_features)
    assert all(backgrounds >= 2 for _, backgrounds, *_ in puzzle_features)
    assert sum(palettes >= 2 for _, _, palettes, *_ in puzzle_features) >= 35
    assert any(train_axes == {"h", "v"} for _, _, _, train_axes, _, _ in puzzle_features)
    assert any(test_axes - train_axes for _, _, _, train_axes, test_axes, _ in puzzle_features)
    assert set().union(*(roles for *_, roles in puzzle_features)) >= {
        "all_new", "familiar", "shift", "mixed"
    }
