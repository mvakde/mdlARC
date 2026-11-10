"""Solved examples of the perpendicular-guide, nearest-stripe ARC rule."""

import numpy as np


_SOURCE_INPUT_TEXTS = (
    """7777877777777777
7777877777777777
9999199999999999
7777877777777777
7777877777777777
7777877777777777
7777877777777777
7777877777777777
7777877777777777
7777877777777777
6666166666666666
7777877777777777
7777877777777777
7777877777777777
3333133333333333
7777877777777777""",
    """7778777777777777
7778777777777777
7778777777777777
7778777777777777
5551555555555555
7778777777777777
7778777777777777
7778777777777777
7778777777777777
7778777777777777
7778777777777777
7778777777777777
7778777777777777
2221222222222222
7778777777777777
7778777777777777""",
    """7777777787777777
7777777787777777
4444444414444444
7777777787777777
7777777787777777
7777777787777777
4444444414444444
7777777787777777
7777777787777777
7777777787777777
7777777787777777
7777777787777777
7777777787777777
0000000010000000
7777777787777777
7777777787777777""",
    """7777877777777777
7777877777777777
9999199999999999
7777877777777777
7777877777777777
6666166666666666
7777877777777777
7777877777777777
7777877777777777
6666166666666666
7777877777777777
7777877777777777
7777877777777777
5555155555555555
7777877777777777
7777877777777777""",
)


def _normalize_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    train_count = _normalize_count(num_train, "num_train", 2, 10)
    test_count = _normalize_count(num_test, "num_test", 1, 4)
    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        lower = max(2, test_count + 1)
        train_count = lower + (seed // 4) % (11 - lower)
    elif train_count is None:
        lower = max(2, test_count + 1)
        train_count = int(rng.integers(lower, 11))
    elif test_count is None:
        test_count = int(rng.integers(1, min(4, train_count - 1) + 1))
    if train_count <= test_count:
        raise ValueError("num_train must exceed num_test")
    return train_count, test_count


def _canonical_pattern(labels):
    names = {}
    return tuple(names.setdefault(x, len(names)) for x in labels)


def _anchor_spec(rng, parity, max_sources):
    # Together the two anchors teach equal/different labels at both
    # even and odd separations. Their order and optional extra sources vary.
    labels = [0, 0, 1]
    gaps = [parity, parity]
    if max_sources >= 4 and rng.random() < 0.7:
        labels.append(int(rng.choice((0, 1, 2))))
        gaps.append(int(rng.integers(0, 2)))
    if max_sources >= 5 and rng.random() < 0.35:
        labels.append(int(rng.choice((0, 1, 2, 3))))
        gaps.append(int(rng.integers(0, 2)))
    if int(rng.integers(0, 2)):
        labels.reverse()
        gaps.reverse()
    return tuple(labels), tuple(gaps)


def _sample_pattern(rng, count):
    if count == 2:
        return (0, 1)
    if count == 4 and rng.random() < 0.85:
        # Four stripes are the source task's test length. Visit its
        # distinct equality relationships without fixing colors/gaps.
        patterns = (
            (0, 0, 1, 2), (0, 1, 0, 2),
            (0, 0, 1, 1), (0, 1, 2, 3),
        )
        pattern = patterns[int(rng.integers(0, len(patterns)))]
        return pattern
    modes = ["free", "all_distinct", "left_repeat", "right_repeat",
             "nonconsecutive"]
    if count >= 4:
        modes += ["double_runs", "long_run"]
    mode = modes[int(rng.integers(0, len(modes)))]
    if mode == "all_distinct":
        pattern = list(range(count))
    elif mode == "free":
        colors = int(rng.integers(2, min(count, 7) + 1))
        pattern = list(range(colors))
        pattern += [
            int(rng.integers(0, colors))
            for _ in range(count - colors)
        ]
        rng.shuffle(pattern)
    else:
        bases = {
            "left_repeat": [0, 0, 1],
            "right_repeat": [1, 0, 0],
            "nonconsecutive": [0, 1, 0],
            "double_runs": [0, 0, 1, 1],
            "long_run": [0, 0, 0, 1],
        }
        pattern = bases[mode].copy()
        while len(pattern) < count:
            upper = min(7, max(pattern) + 2)
            pattern.append(int(rng.integers(0, upper)))
        if int(rng.integers(0, 2)):
            pattern.reverse()
    return _canonical_pattern(pattern)


def _sample_test_spec(rng, count):
    labels = _sample_pattern(rng, count)
    parities = [int(x) for x in rng.integers(0, 2, size=count - 1)]
    different = [
        i for i in range(count - 1) if labels[i] != labels[i + 1]
    ]
    # Deliberately visit no seam, one seam, and multiple seam cases.
    seam_choices = [0, 1]
    if len(different) >= 2:
        seam_choices.append(int(rng.integers(2, len(different) + 1)))
    seams = seam_choices[int(rng.integers(0, len(seam_choices)))]
    rng.shuffle(different)
    for rank, i in enumerate(different):
        parities[i] = 0 if rank < seams else 1
    return labels, tuple(parities)


def _sample_positions(rng, length, parities):
    minimum = [2 if parity == 0 else 1 for parity in parities]
    assert sum(minimum) <= length - 1
    gaps = minimum.copy()
    spare = length - 1 - sum(gaps)
    order = [int(x) for x in rng.permutation(len(gaps))]
    # A long interval is useful with sparse sources, including a true
    # midpoint far from both source lines.
    if spare and rng.random() < 0.32:
        eligible = [
            i for i in order
            if gaps[i] + 2 * (spare // 2) >= 10
        ]
        if eligible:
            i = eligible[int(rng.integers(0, len(eligible)))]
            candidates = [
                x for x in range(10, min(21, gaps[i] + spare + 1))
                if x % 2 == gaps[i] % 2
            ]
            chosen = candidates[int(rng.integers(0, len(candidates)))]
            spare -= chosen - gaps[i]
            gaps[i] = chosen
            order.remove(i)
    for i in order:
        cap = min((spare // 2), (20 - gaps[i]) // 2)
        if cap <= 0:
            continue
        if rng.random() < 0.65:
            steps = int(min(cap, rng.geometric(0.53) - 1))
        else:
            steps = int(rng.integers(0, cap + 1))
        gaps[i] += 2 * steps
        spare -= 2 * steps
    span = sum(gaps)
    last_start = length - 1 - span
    edge_choice = rng.random()
    if edge_choice < 0.18:
        start = 0
    elif edge_choice < 0.36:
        start = last_start
    else:
        start = int(rng.integers(0, last_start + 1))
    positions = [start]
    for gap in gaps:
        positions.append(positions[-1] + gap)
    return tuple(positions)


def _make_input(rng, labels, parities, orientation):
    count = len(labels)
    minimum_length = max(8, count + 2,
                         1 + sum(2 if p == 0 else 1 for p in parities))
    axial_length = int(rng.integers(minimum_length, 31))
    cross_length = int(rng.integers(6, 31))
    positions = _sample_positions(rng, axial_length, parities)
    guide_index = int(rng.integers(0, cross_length))
    if rng.random() < 0.24:
        guide_index = int(rng.choice((0, cross_length - 1)))

    palette = [int(x) for x in rng.permutation(10)]
    background, guide, junction = palette[:3]
    symbols = list(dict.fromkeys(labels))
    actual = {
        symbol: palette[3 + i]
        for i, symbol in enumerate(symbols)
    }
    oriented = np.full(
        (axial_length, cross_length), background, dtype=np.int8
    )
    oriented[:, guide_index] = guide
    for position, symbol in zip(positions, labels):
        oriented[position, :] = actual[symbol]
        oriented[position, guide_index] = junction
    return oriented.copy() if orientation == 0 else oriented.T.copy()


def _analyze_input(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    candidates = []
    for orientation in (0, 1):
        oriented = grid if orientation == 0 else grid.T
        axial_length, cross_length = oriented.shape
        if axial_length < 4 or cross_length < 3:
            continue
        for guide_index in range(cross_length):
            outside = np.delete(oriented, guide_index, axis=1)
            row_colors = outside[:, 0]
            if not np.all(outside == row_colors[:, None]):
                continue
            crossings = oriented[:, guide_index]
            crossing_colors = np.unique(crossings)
            if len(crossing_colors) != 2:
                continue
            groups = [
                set(int(x) for x in row_colors[crossings == color])
                for color in crossing_colors
            ]
            backgrounds = [i for i, group in enumerate(groups)
                           if len(group) == 1]
            sources = [i for i, group in enumerate(groups)
                       if len(group) >= 2]
            if len(backgrounds) != 1 or len(sources) != 1:
                continue
            bg_group, source_group = backgrounds[0], sources[0]
            background = next(iter(groups[bg_group]))
            guide = int(crossing_colors[bg_group])
            junction = int(crossing_colors[source_group])
            source_positions = tuple(
                int(i) for i in np.flatnonzero(crossings == junction)
            )
            if not 2 <= len(source_positions) <= 7:
                continue
            if axial_length - len(source_positions) < 2:
                continue
            source_labels = tuple(
                int(row_colors[i]) for i in source_positions
            )
            if background in source_labels:
                continue
            if {guide, junction} & ({background} | set(source_labels)):
                continue
            rebuilt = np.full(oriented.shape, background, dtype=grid.dtype)
            rebuilt[:, guide_index] = guide
            for pos, label in zip(source_positions, source_labels):
                rebuilt[pos, :] = label
                rebuilt[pos, guide_index] = junction
            if not np.array_equal(rebuilt, oriented):
                continue
            candidates.append({
                "orientation": orientation,
                "oriented_shape": oriented.shape,
                "shape": grid.shape,
                "guide_index": guide_index,
                "background": background,
                "guide": guide,
                "junction": junction,
                "source_positions": source_positions,
                "source_labels": source_labels,
            })
    assert len(candidates) == 1, "ambiguous source/guide geometry"
    return candidates[0]


def _solve_input(grid):
    data = _analyze_input(grid)
    axial_length, cross_length = data["oriented_shape"]
    positions = data["source_positions"]
    labels = data["source_labels"]
    guide_index = data["guide_index"]
    guide, junction = data["guide"], data["junction"]
    result = np.empty((axial_length, cross_length), dtype=grid.dtype)
    for row in range(axial_length):
        distances = [abs(row - pos) for pos in positions]
        nearest = [labels[i] for i, distance in enumerate(distances)
                   if distance == min(distances)]
        if row in positions:
            result[row, :] = junction
            result[row, guide_index] = guide
        elif len(set(nearest)) > 1:
            result[row, :] = junction
        else:
            result[row, :] = nearest[0]
            result[row, guide_index] = junction
    return result.copy() if data["orientation"] == 0 else result.T.copy()


def _expected_by_intervals(grid, data):
    """Independent interval construction for validation."""
    length, width = data["oriented_shape"]
    positions = data["source_positions"]
    labels = data["source_labels"]
    line = [None] * length
    for row in range(positions[0]):
        line[row] = labels[0]
    for i, pos in enumerate(positions):
        line[pos] = "source"
        if i + 1 == len(positions):
            break
        right = positions[i + 1]
        for row in range(pos + 1, right):
            left_dist, right_dist = row - pos, right - row
            if left_dist == right_dist and labels[i] != labels[i + 1]:
                line[row] = "seam"
            else:
                line[row] = labels[i] if left_dist <= right_dist else labels[i + 1]
    for row in range(positions[-1] + 1, length):
        line[row] = labels[-1]
    result = np.empty((length, width), dtype=grid.dtype)
    for row, value in enumerate(line):
        if value == "source":
            result[row, :] = data["junction"]
            result[row, data["guide_index"]] = data["guide"]
        elif value == "seam":
            result[row, :] = data["junction"]
        else:
            result[row, :] = value
            result[row, data["guide_index"]] = data["junction"]
    return result if data["orientation"] == 0 else result.T


def _grid_key(grid):
    return (tuple(grid.shape),
            np.ascontiguousarray(grid, dtype=np.int64).tobytes())


def _is_original_input(grid):
    if grid.shape != (16, 16):
        return False
    text = "\n".join("".join(str(int(x)) for x in row) for row in grid)
    return text in _SOURCE_INPUT_TEXTS


def _features(data):
    positions, labels = data["source_positions"], data["source_labels"]
    return {
        (labels[i] == labels[i + 1],
         (positions[i + 1] - positions[i]) % 2)
        for i in range(len(positions) - 1)
    }


def _orientation_plan(rng, train_count, test_count):
    mode = int(rng.integers(0, 5))
    if mode in (0, 1):
        return [mode] * train_count, [mode] * test_count
    if mode == 2:
        train = [0, 1] + [
            int(rng.integers(0, 2)) for _ in range(train_count - 2)
        ]
        rng.shuffle(train)
        test = [int(rng.integers(0, 2)) for _ in range(test_count)]
        return train, test
    train_orientation = mode - 3
    return ([train_orientation] * train_count,
            [1 - train_orientation] * test_count)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(
        seed, rng, num_train, num_test
    )
    # The transformation is the sole puzzle-level convention. Literal
    # colors, geometry, and orientation are visible pair-level choices.
    regime = int(rng.integers(0, 3))
    max_anchor = 4 if regime == 0 else 5
    train_specs = [
        _anchor_spec(rng, 0, max_anchor),
        _anchor_spec(rng, 1, max_anchor),
    ]
    for _ in range(train_count - 2):
        if regime == 0:
            count = int(rng.integers(2, 5))
        elif regime == 1:
            count = int(rng.integers(2, 8))
        else:
            count = int(rng.integers(4, 8))
        labels = _sample_pattern(rng, count)
        parities = tuple(
            int(x) for x in rng.integers(0, 2, size=count - 1)
        )
        train_specs.append((labels, parities))
    rng.shuffle(train_specs)
    largest_train = max(len(labels) for labels, _ in train_specs)
    test_specs = []
    for i in range(test_count):
        if regime == 0:
            count = int(rng.integers(largest_train + 1, 8))
        elif regime == 1 and i == 0:
            count = len(train_specs[int(rng.integers(0, len(train_specs)))][0])
        elif regime == 1:
            count = (4 if rng.random() < 0.45
                     else int(rng.integers(2, 8)))
        else:
            upper = min(4, largest_train - 1)
            count = (4 if upper == 4 and rng.random() < 0.55
                     else int(rng.integers(2, upper + 1)))
        test_specs.append(_sample_test_spec(rng, count))
    train_orientations, test_orientations = _orientation_plan(
        rng, train_count, test_count
    )

    puzzle = {"train": [], "test": []}
    seen = set()
    for split, specs, orientations in (
        ("train", train_specs, train_orientations),
        ("test", test_specs, test_orientations),
    ):
        for (labels, parities), orientation in zip(specs, orientations):
            pair_seed = int(rng.integers(0, 2**63, dtype=np.uint64))
            pair_rng = np.random.default_rng(pair_seed)
            for _ in range(80):
                input_grid = _make_input(
                    pair_rng, labels, parities, orientation
                )
                key = _grid_key(input_grid)
                if key in seen or _is_original_input(input_grid):
                    continue
                try:
                    output_grid = _solve_input(input_grid)
                except AssertionError:
                    continue
                seen.add(key)
                puzzle[split].append({
                    "input": input_grid, "output": output_grid
                })
                break
            else:
                raise RuntimeError("could not create an unambiguous pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    train_features = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict)
            assert set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30
                assert 1 <= grid.shape[1] <= 30
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape
            assert not np.array_equal(inp, out)
            input_key = _grid_key(inp)
            pair_key = (input_key, _grid_key(out))
            assert input_key not in seen_inputs
            assert pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert not _is_original_input(inp)
            data = _analyze_input(inp)
            assert np.array_equal(out, _expected_by_intervals(inp, data))
            assert data["background"] not in out
            assert set(int(x) for x in np.unique(out)) <= (
                {data["guide"], data["junction"]} |
                set(data["source_labels"])
            )
            if split == "train":
                train_features |= _features(data)
    # These examples teach source inversion, both midpoint outcomes,
    # and the absence of a midpoint for odd separations.
    assert {(False, 0), (False, 1), (True, 0), (True, 1)} <= train_features


def _puzzle_key(puzzle):
    return tuple(
        (split, tuple(
            (_grid_key(pair["input"]), _grid_key(pair["output"]))
            for pair in puzzle[split]
        ))
        for split in ("train", "test")
    )


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    shapes, patterns, source_counts = set(), set(), set()
    train_source_counts, test_source_counts = set(), set()
    test_patterns = set()
    seam_counts, orientations = set(), set()
    long_gap = adjacent = edge_source = edge_guide = False
    small_canvas = full_extent = many_labels = False
    all_distinct = repeat_start = repeat_end = False
    nonconsecutive = long_run = multiple_equal_runs = False
    mixed_orientation = varied_shape = varied_colors = False
    fixed_orientations = set()
    backgrounds, guides, junctions = set(), set(), set()
    wide_rectangle = novel_test_label = False
    increase = smaller = same_count = False
    shift_0_to_1 = shift_1_to_0 = False
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_data = [_analyze_input(p["input"]) for p in puzzle["train"]]
        test_data = [_analyze_input(p["input"]) for p in puzzle["test"]]
        all_data = train_data + test_data
        train_counts = {len(d["source_positions"]) for d in train_data}
        test_counts = {len(d["source_positions"]) for d in test_data}
        train_source_counts |= train_counts
        test_source_counts |= test_counts
        increase |= max(train_counts) < min(test_counts)
        smaller |= max(test_counts) < max(train_counts)
        same_count |= bool(train_counts & test_counts)
        train_orientations = {d["orientation"] for d in train_data}
        test_orientations = {d["orientation"] for d in test_data}
        shift_0_to_1 |= train_orientations == {0} and test_orientations == {1}
        shift_1_to_0 |= train_orientations == {1} and test_orientations == {0}
        mixed_orientation |= len({d["orientation"] for d in all_data}) > 1
        if len({d["orientation"] for d in all_data}) == 1:
            fixed_orientations.add(all_data[0]["orientation"])
        varied_shape |= len({d["shape"] for d in all_data}) > 1
        varied_colors |= len({
            (d["background"], d["guide"], d["junction"])
            for d in all_data
        }) > 1
        for d in all_data:
            shapes.add(d["shape"])
            orientations.add(d["orientation"])
            backgrounds.add(d["background"])
            guides.add(d["guide"])
            junctions.add(d["junction"])
            source_counts.add(len(d["source_positions"]))
            patterns.add(_canonical_pattern(d["source_labels"]))
            small_canvas |= min(d["shape"]) <= 9
            full_extent |= max(d["shape"]) == 30
            wide_rectangle |= max(d["shape"]) >= 2 * min(d["shape"])
            many_labels |= len(set(d["source_labels"])) >= 4
            positions = d["source_positions"]
            gaps = [b - a for a, b in zip(positions, positions[1:])]
            long_gap |= any(g > 9 for g in gaps)
            adjacent |= 1 in gaps
            edge_source |= positions[0] == 0 or (
                positions[-1] == d["oriented_shape"][0] - 1
            )
            edge_guide |= d["guide_index"] in (
                0, d["oriented_shape"][1] - 1
            )
        for d in test_data:
            labels = d["source_labels"]
            count = len(labels)
            train_labels = {
                label for other in train_data
                for label in other["source_labels"]
            }
            novel_test_label |= bool(set(labels) - train_labels)
            test_patterns.add(_canonical_pattern(labels))
            all_distinct |= len(set(labels)) == count
            repeat_start |= labels[0] == labels[1]
            repeat_end |= labels[-2] == labels[-1]
            nonconsecutive |= any(
                labels[i] == labels[j] and
                len(set(labels[i + 1:j])) > 0
                for i in range(count) for j in range(i + 2, count)
            )
            long_run |= any(
                labels[i] == labels[i + 1] == labels[i + 2]
                for i in range(count - 2)
            )
            multiple_equal_runs |= sum(
                labels[i] == labels[i + 1]
                for i in range(count - 1)
            ) >= 2
            seam_counts.add(sum(
                a != b and (y - x) % 2 == 0
                for a, b, x, y in zip(
                    d["source_labels"], d["source_labels"][1:],
                    d["source_positions"], d["source_positions"][1:]
                )
            ))
    assert len(shapes) >= 45
    assert len(patterns) >= 20
    assert source_counts == set(range(2, 8))
    assert train_source_counts == set(range(2, 8))
    assert test_source_counts == set(range(2, 8))
    assert {
        (0, 0, 1, 2), (0, 1, 0, 2),
        (0, 0, 1, 1), (0, 1, 2, 3),
    } <= test_patterns
    assert {0, 1, 2} <= seam_counts
    assert orientations == {0, 1}
    assert fixed_orientations == {0, 1}
    assert backgrounds == guides == junctions == set(range(10))
    assert all((long_gap, adjacent, edge_source, edge_guide))
    assert all((small_canvas, full_extent, wide_rectangle,
                many_labels, novel_test_label))
    assert all((all_distinct, repeat_start, repeat_end,
                nonconsecutive, long_run, multiple_equal_runs))
    assert all((mixed_orientation, varied_shape, varied_colors))
    assert all((increase, smaller, same_count))
    assert shift_0_to_1 and shift_1_to_0
    assert _puzzle_key(generate_puzzle(123456, 7, 3)) == _puzzle_key(
        generate_puzzle(123456, 7, 3)
    )
    for seed, train_count, test_count in (
        (500, 2, 1), (501, 10, 1), (502, 10, 4),
        (600, None, 1), (601, None, 4),
        (602, 2, None), (603, 10, None),
    ):
        puzzle = generate_puzzle(seed, train_count, test_count)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)
    for kwargs in (
        {"num_train": 1}, {"num_train": 11},
        {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
        {"num_train": 4, "num_test": 4},
        {"num_train": True, "num_test": 1},
    ):
        try:
            generate_puzzle(700, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
