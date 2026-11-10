import numpy as np


_SOURCE_INPUT_KEYS = {
    "/".join((
        "000000000000000",
        "000000000000000",
        "000000000000000",
        "333033300000000",
        "000000000000000",
        "000000000000000",
        "000000666066666",
        "000000000000000",
        "444044477077777",
        "000000000500000",
        "000000000500000",
        "000100000500000",
        "000100000500000",
        "000100000500000",
        "000100000500000",
    )),
    "/".join((
        "000000000000000",
        "000000000000000",
        "000000000000000",
        "000000000000000",
        "000000000000000",
        "000000000000000",
        "000000000000000",
        "000000000000000",
        "000000000000000",
        "222022223033333",
        "000000000000000",
        "000300000000000",
        "000300000400000",
        "000300000400000",
        "000300000400000",
    )),
    "/".join((
        "000000000000000",
        "000000000000000",
        "000000777077777",
        "808888000000000",
        "000000000000000",
        "000006666066666",
        "000000000000000",
        "000000000000000",
        "303333000000000",
        "000000000000000",
        "000000000000000",
        "909999555055555",
        "000000000000000",
        "000000000000000",
        "010000000100000",
    )),
    "/".join((
        "000000000000000",
        "000000000000000",
        "222022000000000",
        "000000550555555",
        "000000000000000",
        "000000000000000",
        "333033300000000",
        "000000000000000",
        "000000990999999",
        "000000000000000",
        "000000000000000",
        "444044440444444",
        "000000000000000",
        "000600000000000",
        "000600008000000",
    )),
}


def _rows_key(grid: np.ndarray) -> str:
    return "/".join("".join(str(int(v)) for v in row) for row in grid)


def _array_key(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(v) for v in grid.flat))


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} is outside {low}..{high}")
    return value


def _counts(seed, rng, num_train, num_test):
    nt = _count(num_train, "num_train", 2, 10)
    ne = _count(num_test, "num_test", 1, 4)
    if nt is None and ne is None:
        ne = 1 + seed % 4
        low = max(2, ne + 1)
        nt = low + (seed // 4) % (11 - low)
    elif nt is None:
        nt = int(rng.integers(max(2, ne + 1), 11))
    elif ne is None:
        ne = int(rng.integers(1, min(4, nt - 1) + 1))
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    return nt, ne


def _choose(rng, choices):
    return int(choices[int(rng.integers(len(choices)))])


def _arm(rng, maximum):
    assert maximum >= 1
    x = rng.random()
    if x < 0.19:
        return 1
    if x < 0.40:
        return maximum
    return int(rng.integers(1, maximum + 1))


def _tracks(rng, width):
    options = [(a, b) for a in range(1, width - 5)
               for b in range(a + 4, width - 1)]
    return options[int(rng.integers(len(options)))]


def _row_plan(rng, height, lengths, counts, profile):
    limit = height - max(lengths) - 1
    if profile in ("shared", "independent"):
        common_count = int(rng.integers(1, min(counts) + 1))
    else:
        common_count = int(rng.integers(0, min(counts) + 1))
    union_count = sum(counts) - common_count
    assert union_count <= limit + 1
    chosen = [int(x) for x in rng.choice(limit + 1, size=union_count, replace=False)]
    rng.shuffle(chosen)
    common = chosen[:common_count]
    first_only = chosen[common_count:common_count + counts[0] - common_count]
    second_only = chosen[common_count + counts[0] - common_count:]
    first = sorted(common + first_only)
    second = sorted(common + second_only)
    modes = {}
    for row in common:
        if profile == "shared":
            modes[row] = "shared"
        elif profile == "independent":
            modes[row] = "independent"
        else:
            modes[row] = "shared" if rng.random() < 0.55 else "independent"
    return first, second, modes


def _crossing_colors(rng, background, stub_colors, rows, modes, allow_same, force_same):
    palette = [c for c in range(10) if c != background]
    result = ({}, {})
    current = list(stub_colors)
    history = ([stub_colors[0]], [stub_colors[1]])
    forced = None
    if force_same:
        choices = [(i, row) for i in range(2) for row in rows[i]]
        forced = choices[int(rng.integers(len(choices)))]

    for row in sorted(set(rows[0]) | set(rows[1]), reverse=True):
        active = [i for i in range(2) if row in rows[i]]
        if len(active) == 2 and modes[row] == "shared":
            if forced is not None and forced[1] == row:
                color = current[forced[0]]
            else:
                banned = set() if allow_same else set(current)
                color = _choose(rng, [c for c in palette if c not in banned])
            for i in active:
                result[i][row] = color
                current[i] = color
                history[i].append(color)
            continue

        for i in active:
            if forced == (i, row):
                color = current[i]
            else:
                banned = set()
                if not allow_same or rng.random() >= 0.13:
                    banned.add(current[i])
                if len(active) == 2 and i == 0 and forced == (1, row):
                    banned.add(current[1])
                if len(active) == 2 and i == 1:
                    banned.add(result[0][row])
                reusable = [c for c in history[i][:-1] if c not in banned]
                if reusable and rng.random() < 0.24:
                    color = _choose(rng, reusable)
                else:
                    color = _choose(rng, [c for c in palette if c not in banned])
            if len(active) == 2 and i == 1 and color == result[0][row]:
                color = _choose(rng, [c for c in palette if c != result[0][row]])
            result[i][row] = color
            current[i] = color
            history[i].append(color)
    return result


def _paint_bars(grid, rng, tracks, crossings, modes):
    background = int(grid[0, tracks[0]])
    width = grid.shape[1]
    a, b = tracks
    for row in sorted(set(crossings[0]) | set(crossings[1])):
        first = row in crossings[0]
        second = row in crossings[1]
        if first and second and modes[row] == "shared":
            if rng.random() < 0.14:
                left, right = a, width - 1 - b
            else:
                left = _arm(rng, a)
                right = _arm(rng, width - 1 - b)
            grid[row, a - left:b + right + 1] = crossings[0][row]
        elif first and second:
            inside = b - a - 1
            inner_first = int(rng.integers(1, inside))
            inner_second = _arm(rng, inside - inner_first)
            left = _arm(rng, a)
            right = _arm(rng, width - 1 - b)
            grid[row, a - left:a + inner_first + 1] = crossings[0][row]
            grid[row, b - inner_second:b + right + 1] = crossings[1][row]
        elif first:
            left = _arm(rng, a)
            right = _arm(rng, b - a - 1)
            grid[row, a - left:a + right + 1] = crossings[0][row]
        else:
            left = _arm(rng, b - a - 1)
            right = _arm(rng, width - 1 - b)
            grid[row, b - left:b + right + 1] = crossings[1][row]
        if first:
            grid[row, a] = background
        if second:
            grid[row, b] = background


def _solve(input_grid, tracks, crossings):
    output = input_grid.copy()
    for i, column in enumerate(tracks):
        color = int(input_grid[-1, column])
        for row in range(input_grid.shape[0] - 1, -1, -1):
            if row in crossings[i]:
                color = crossings[i][row]
            output[row, column] = color
    return output


def _make_pair(rng, dimensions, background, profile, counts, same_stubs,
               allow_same, force_same):
    height, width = dimensions
    tracks = _tracks(rng, width)
    palette = [c for c in range(10) if c != background]
    first_color = _choose(rng, palette)
    second_color = first_color if same_stubs else _choose(
        rng, [c for c in palette if c != first_color])
    stubs = (first_color, second_color)
    lengths = (int(rng.integers(1, min(7, height - 6))),
               int(rng.integers(1, min(7, height - 6))))
    rows1, rows2, modes = _row_plan(rng, height, lengths, counts, profile)
    crossings = _crossing_colors(rng, background, stubs,
                                 (rows1, rows2), modes, allow_same, force_same)
    grid = np.full((height, width), background, dtype=np.int8)
    for i, column in enumerate(tracks):
        grid[height - lengths[i]:, column] = stubs[i]
    _paint_bars(grid, rng, tracks, crossings, modes)
    return {"input": grid, "output": _solve(grid, tracks, crossings)}


def _validate_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert all(1 <= n <= 30 for n in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert bool(np.all((0 <= grid) & (grid <= 9)))


def _analyze(pair):
    grid, output = pair["input"], pair["output"]
    _validate_grid(grid)
    _validate_grid(output)
    assert grid.shape == output.shape
    height, width = grid.shape
    counts = np.bincount(grid.flat, minlength=10)
    background = int(np.argmax(counts))
    assert counts[background] > np.partition(counts, -2)[-2]
    tracks = tuple(int(x) for x in np.flatnonzero(grid[-1] != background))
    assert len(tracks) == 2
    assert 1 <= tracks[0] and tracks[1] <= width - 2
    assert tracks[1] - tracks[0] >= 4
    lengths = []
    crossings = []
    for column in tracks:
        stub = int(grid[-1, column])
        length = 0
        while length < height and int(grid[height - length - 1, column]) == stub:
            length += 1
        assert 1 <= length <= 6
        assert bool(np.all(grid[:height - length, column] == background))
        lengths.append(length)
        found = {}
        for row in range(height - length):
            left, right = int(grid[row, column - 1]), int(grid[row, column + 1])
            if left == right and left != background:
                found[row] = left
        assert 1 <= len(found) <= 3
        crossings.append(found)
    bar_rows = set(crossings[0]) | set(crossings[1])
    for row in range(height):
        colored = [col for col in range(width)
                   if col not in tracks and int(grid[row, col]) != background]
        if not colored:
            continue
        assert row in bar_rows
        # Every monochrome run must touch a gap that it explains.
        start = 0
        while start < len(colored):
            end = start + 1
            while (end < len(colored) and colored[end] == colored[end - 1] + 1
                   and grid[row, colored[end]] == grid[row, colored[start]]):
                end += 1
            run = set(colored[start:end])
            color = int(grid[row, colored[start]])
            assert any(row in crossings[i] and crossings[i][row] == color
                       and (column - 1 in run or column + 1 in run)
                       for i, column in enumerate(tracks))
            start = end
    expected = _solve(grid, tracks, crossings)
    assert np.array_equal(output, expected)
    assert not np.array_equal(grid, output)
    return {"background": background, "shape": grid.shape,
            "tracks": tracks, "lengths": tuple(lengths),
            "crossings": tuple(crossings), "input": grid}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    analyses = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            analysis = _analyze(pair)
            key = _array_key(pair["input"])
            assert key not in seen
            assert _rows_key(pair["input"]) not in _SOURCE_INPUT_KEYS
            seen.add(key)
            analyses[section].append(analysis)
    # A two-gap bar and a pair of independent same-row bars are both taught.
    assert any(_row_types(a)[0] for a in analyses["train"])
    assert any(_row_types(a)[1] for a in analyses["train"])
    train_depth = max(len(c) for a in analyses["train"] for c in a["crossings"])
    assert all(len(c) <= train_depth for a in analyses["test"] for c in a["crossings"])
    if any(_has_same_color(a) for a in analyses["test"]):
        assert any(_has_same_color(a) for a in analyses["train"])


def _row_types(analysis):
    first, second = analysis["crossings"]
    shared, independent = False, False
    grid = analysis["input"]
    a, b = analysis["tracks"]
    for row in set(first) & set(second):
        color1, color2 = first[row], second[row]
        if color1 == color2 and bool(np.all(grid[row, a + 1:b] == color1)):
            shared = True
        elif color1 != color2:
            independent = True
    return shared, independent


def _has_same_color(analysis):
    for i, crossing in enumerate(analysis["crossings"]):
        current = int(analysis["input"][-1, analysis["tracks"][i]])
        for row in sorted(crossing, reverse=True):
            if crossing[row] == current:
                return True
            current = crossing[row]
    return False


def _dimensions(rng):
    if rng.random() < 0.08:
        return (15, 15)
    return (int(rng.integers(15, 31)), int(rng.integers(12, 31)))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)
    # Only the bottom-up rule is learned. Palette and geometry have local roles.
    uniform_shape = rng.random() < 0.20
    uniform_background = rng.random() < 0.25
    shared_shape = _dimensions(rng)
    shared_background = int(rng.integers(10))
    allow_same = bool(rng.random() < 0.36)
    shared_index = int(rng.integers(2))
    same_stubs_index = int(rng.integers(2))
    depth_slot = int(rng.integers(2 * train_count))
    result = {"train": [], "test": []}
    seen = set()
    first_shape = None
    first_background = None
    for section, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            if uniform_shape:
                dimensions = shared_shape
            else:
                dimensions = _dimensions(rng)
                if section == "train" and index == 1 and dimensions == first_shape:
                    dimensions = (15 if first_shape[0] != 15 else 30, dimensions[1])
            if uniform_background:
                background = shared_background
            else:
                background = int(rng.integers(10))
                if section == "train" and index == 1 and background == first_background:
                    background = (background + 1) % 10
            if section == "train" and index == 0:
                first_shape, first_background = dimensions, background
            if section == "train":
                profile = "shared" if index == shared_index else (
                    "independent" if index == 1 - shared_index else "mixed")
                same_stubs = (index == same_stubs_index) if index < 2 else rng.random() < 0.35
                counts = [int(rng.integers(1, 4)), int(rng.integers(1, 4))]
                if index == depth_slot // 2:
                    counts[depth_slot % 2] = 3
                force_same = allow_same and index == 0
            else:
                profile = "mixed"
                same_stubs = bool(rng.random() < 0.35)
                counts = [int(rng.choice([1, 2, 2, 3, 3])),
                          int(rng.choice([1, 2, 2, 3, 3]))]
                force_same = False
            for attempt in range(20):
                child = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                pair = _make_pair(child, dimensions, background, profile, tuple(counts),
                                  same_stubs, allow_same, force_same)
                key = _array_key(pair["input"])
                if key not in seen and _rows_key(pair["input"]) not in _SOURCE_INPUT_KEYS:
                    break
            else:
                raise ValueError("unable to construct a unique pair")
            seen.add(key)
            result[section].append(pair)
    validate_generated_puzzle(result)
    return result


def _puzzle_key(puzzle):
    return tuple((_array_key(p["input"]), _array_key(p["output"]))
                 for section in ("train", "test") for p in puzzle[section])


def self_test() -> None:
    train_counts, test_counts, shapes, layouts = set(), set(), set(), set()
    shared_heights, backgrounds = set(), set()
    same_color_count = 0
    close_count = 0
    review_hits = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_shapes, puzzle_backgrounds = set(), set()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                analysis = _analyze(pair)
                shapes.add(analysis["shape"])
                backgrounds.add(analysis["background"])
                puzzle_shapes.add(analysis["shape"])
                puzzle_backgrounds.add(analysis["background"])
                layouts.add((analysis["shape"], analysis["tracks"],
                             tuple(tuple(sorted(c)) for c in analysis["crossings"])))
                same_color_count += _has_same_color(analysis)
                close_count += analysis["tracks"][1] - analysis["tracks"][0] <= 6
                grid = analysis["input"]
                h, w = analysis["shape"]
                a, b = analysis["tracks"]
                crossings = analysis["crossings"]
                rows = sorted(set(crossings[0]) | set(crossings[1]))
                if (h, w) == (15, 15): review_hits.add("source_scale")
                if h >= 27 or w >= 27: review_hits.add("large")
                if h - w >= 5: review_hits.add("tall")
                if w - h >= 5: review_hits.add("wide")
                if a == 1 or b == w - 2: review_hits.add("edge_track")
                if b - a <= 6: review_hits.add("close_track")
                if max(analysis["lengths"]) >= 4: review_hits.add("long_root")
                if any(max(crossings[i]) == h - analysis["lengths"][i] - 1
                       for i in range(2)): review_hits.add("immediate_root")
                if rows[0] <= 2: review_hits.add("near_top")
                if any(y - x == 1 for x, y in zip(rows, rows[1:])):
                    review_hits.add("adjacent_rows")
                if len(crossings[0]) != len(crossings[1]):
                    review_hits.add(section + "_unequal_depth")
                if min(len(c) for c in crossings) == 1:
                    review_hits.add(section + "_simple")
                if len(crossings[0]) == len(crossings[1]) == 3:
                    review_hits.add(section + "_three_each")
                if _has_same_color(analysis): review_hits.add(section + "_same_color")
                if any(
                    colors[j] == colors[k]
                    for i in range(2)
                    for colors in [[int(grid[-1, analysis["tracks"][i]])] +
                                   [crossings[i][r] for r in sorted(crossings[i], reverse=True)]]
                    for j in range(len(colors)) for k in range(j + 2, len(colors))
                ): review_hits.add("separated_color_reuse")
                shared_rows = []
                for row in rows:
                    active = [i for i in range(2) if row in crossings[i]]
                    if len(active) == 2 and crossings[0][row] == crossings[1][row]:
                        shared_rows.append(row)
                        colored = np.flatnonzero(grid[row] != analysis["background"])
                        if int(colored[0]) == 0 and int(colored[-1]) == w - 1:
                            review_hits.add(section + "_full_shared")
                        else:
                            review_hits.add(section + "_partial_shared")
                        if row != rows[-1]: review_hits.add(section + "_higher_shared")
                    elif len(active) == 2:
                        review_hits.add(section + "_independent_same_row")
                    arms = []
                    for i in active:
                        col, color = analysis["tracks"][i], crossings[i][row]
                        left = 0
                        while col-left-1 >= 0 and int(grid[row, col-left-1]) == color:
                            left += 1
                        right = 0
                        while col+right+1 < w and int(grid[row, col+right+1]) == color:
                            right += 1
                        arms.extend((left, right))
                    if min(arms) == 1: review_hits.add("one_cell_arm")
                    if max(arms) >= 6: review_hits.add("long_arm")
                    if len(set(arms)) > 1: review_hits.add("unequal_arms")
                    if int(grid[row, 0]) != analysis["background"] or int(grid[row, -1]) != analysis["background"]:
                        review_hits.add("edge_touch_bar")
                    if len(active) == 2 and crossings[0][row] != crossings[1][row]:
                        if arms[1] + arms[2] == b - a - 1:
                            review_hits.add("different_bars_touch")
                if len(shared_rows) >= 2: review_hits.add(section + "_multiple_shared")
                if _row_types(analysis)[0]:
                    shared_heights.update(set(analysis["crossings"][0]) &
                                          set(analysis["crossings"][1]))
        review_hits.add("uniform_shape" if len(puzzle_shapes) == 1 else "varied_shape")
        review_hits.add("uniform_background" if len(puzzle_backgrounds) == 1 else "varied_background")
    assert len(train_counts) >= 4 and test_counts == {1, 2, 3, 4}
    assert len(shapes) >= 35 and len(layouts) >= 100
    assert len(backgrounds) == 10 and same_color_count >= 5 and close_count >= 5
    assert len(shared_heights) >= 6
    assert {
        "source_scale", "large", "tall", "wide", "edge_track", "close_track",
        "long_root", "immediate_root", "near_top", "adjacent_rows",
        "train_unequal_depth", "test_unequal_depth", "train_simple", "test_simple",
        "train_three_each", "test_three_each", "train_same_color", "test_same_color",
        "separated_color_reuse", "train_full_shared", "train_partial_shared",
        "test_full_shared", "test_partial_shared", "train_higher_shared",
        "test_higher_shared", "train_independent_same_row",
        "test_independent_same_row", "one_cell_arm", "long_arm", "unequal_arms",
        "edge_touch_bar", "different_bars_touch", "train_multiple_shared",
        "test_multiple_shared", "uniform_shape", "varied_shape",
        "uniform_background", "varied_background",
    } <= review_hits
    assert _puzzle_key(generate_puzzle(719)) == _puzzle_key(generate_puzzle(719))
    assert _puzzle_key(generate_puzzle(991, 7, 3)) == _puzzle_key(generate_puzzle(991, 7, 3))
    assert _puzzle_key(generate_puzzle(1201, 6, 2)) != _puzzle_key(generate_puzzle(1202, 6, 2))
    for seed, nt, ne in ((2001, 2, 1), (2002, 10, 1), (2003, 10, 4),
                         (3001, None, 1), (3002, None, 4),
                         (3003, 2, None), (3004, 10, None)):
        puzzle = generate_puzzle(seed, nt, ne)
        assert nt is None or len(puzzle["train"]) == nt
        assert ne is None or len(puzzle["test"]) == ne
        validate_generated_puzzle(puzzle)
    for index, args in enumerate(((1, None), (11, None), (None, 0),
                                  (None, 5), (2, 2), (3, 4))):
        try:
            generate_puzzle(4000 + index, *args)
        except ValueError:
            continue
        raise AssertionError("illegal pair count accepted")
