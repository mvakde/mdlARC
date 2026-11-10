import hashlib

import numpy as np


# Exact sparse source inputs, including shape and literal background. The rule
# determines their outputs, so rejecting these inputs excludes exact source pairs
# without excluding any size, palette, or geometric category.
_SOURCE_INPUTS = (
    ((15, 12), 5, ((2, 9, 6), (3, 1, 1), (5, 3, 6), (8, 6, 6), (9, 7, 1), (13, 4, 6))),
    ((8, 11), 8, ((1, 6, 1), (3, 4, 6), (6, 1, 1))),
    ((9, 12), 4, ((1, 7, 1), (6, 7, 6), (7, 1, 1))),
    ((15, 16), 8, ((1, 8, 6), (2, 2, 1), (3, 12, 1), (9, 9, 6), (10, 1, 6), (11, 11, 1), (14, 3, 1), (14, 13, 6))),
    ((9, 9), 4, ((2, 1, 1), (7, 6, 1))),
    ((25, 21), 3, ((1, 9, 1), (1, 17, 1), (2, 2, 1), (4, 4, 6), (4, 14, 1), (5, 2, 1), (6, 6, 6), (7, 14, 6), (10, 10, 1), (12, 19, 6), (14, 11, 6), (15, 2, 6), (17, 9, 1), (18, 8, 6), (20, 12, 6), (21, 18, 1), (23, 3, 1), (23, 11, 1))),
)


def _source_input(grid: np.ndarray) -> bool:
    background = _background_color(grid)
    for shape, color, dots in _SOURCE_INPUTS:
        if grid.shape == shape and background == color:
            visible = tuple((int(r), int(c), int(grid[r, c]))
                            for r, c in np.argwhere(grid != background))
            if visible == dots:
                return True
    return False


def _grid_key(grid: np.ndarray) -> tuple:
    canonical = grid.astype(np.int8, copy=False)
    return tuple(grid.shape), canonical.tobytes()


def _pair_key(pair: dict) -> tuple:
    return _grid_key(pair["input"]), _grid_key(pair["output"])


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    assert values.size >= 1
    maximum = int(counts.max())
    assert int(np.sum(counts == maximum)) == 1
    return int(values[int(np.argmax(counts))])


def _canonical_edge(a: tuple[int, int], b: tuple[int, int]) -> tuple:
    return tuple(sorted((a, b)))


def _segment_cells(
    a: tuple[int, int],
    b: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    assert dr != 0
    assert abs(dr) == abs(dc)

    row_step = 1 if dr > 0 else -1
    col_step = 1 if dc > 0 else -1
    length = abs(dr)

    return tuple(
        (a[0] + i * row_step, a[1] + i * col_step)
        for i in range(length + 1)
    )


def _segment_slope(
    a: tuple[int, int],
    b: tuple[int, int],
) -> int:
    return 1 if (b[0] - a[0]) * (b[1] - a[1]) > 0 else -1


def _full_diagonal(
    point: tuple[int, int],
    slope: int,
    height: int,
    width: int,
) -> tuple[tuple[int, int], ...]:
    assert slope in (-1, 1)

    row, col = point
    while 0 <= row - 1 < height and 0 <= col - slope < width:
        row -= 1
        col -= slope

    cells = []
    while 0 <= row < height and 0 <= col < width:
        cells.append((row, col))
        row += 1
        col += slope

    return tuple(cells)


def _solve_visible_grid(
    grid: np.ndarray,
    primary_color: int,
    secondary_color: int,
) -> tuple[np.ndarray, dict]:
    colors = {int(x) for x in np.unique(grid)}
    assert primary_color != secondary_color
    assert primary_color in colors
    assert len(colors) in (2, 3)

    background_candidates = colors - {primary_color, secondary_color}
    assert len(background_candidates) == 1
    background = next(iter(background_candidates))
    assert _background_color(grid) == background

    primary_cells = {
        tuple(int(x) for x in location)
        for location in np.argwhere(grid == primary_color)
    }
    secondary_cells = {
        tuple(int(x) for x in location)
        for location in np.argwhere(grid == secondary_color)
    }

    assert len(primary_cells) >= 2

    primary_list = sorted(primary_cells)
    pairs = []
    degrees = {point: 0 for point in primary_list}

    for i in range(len(primary_list)):
        for j in range(i + 1, len(primary_list)):
            a = primary_list[i]
            b = primary_list[j]
            if abs(a[0] - b[0]) == abs(a[1] - b[1]):
                pairs.append((a, b))
                degrees[a] += 1
                degrees[b] += 1

    assert len(pairs) >= 1
    assert all(degree <= 1 for degree in degrees.values())

    segments = []
    segment_slopes = []
    segment_union: set[tuple[int, int]] = set()
    endpoint_cells: set[tuple[int, int]] = set()

    for a, b in pairs:
        cells = _segment_cells(a, b)
        cells_set = set(cells)

        # Same-color segments may cross at an unmarked cell.

        segments.append(cells)
        segment_slopes.append(_segment_slope(a, b))
        segment_union.update(cells_set)
        endpoint_cells.add(a)
        endpoint_cells.add(b)

    active_secondary = secondary_cells & segment_union
    inactive_secondary = secondary_cells - active_secondary
    unpaired_primary = primary_cells - endpoint_cells

    line_by_active: dict[
        tuple[int, int], tuple[tuple[int, int], ...]
    ] = {}
    active_segment_index: dict[tuple[int, int], int] = {}
    line_signatures = set()

    for point in sorted(active_secondary):
        containing_segments = [
            index
            for index, cells in enumerate(segments)
            if point in cells
        ]
        assert len(containing_segments) == 1

        segment_index = containing_segments[0]
        active_segment_index[point] = segment_index
        line_slope = -segment_slopes[segment_index]
        line = _full_diagonal(
            point,
            line_slope,
            grid.shape[0],
            grid.shape[1],
        )
        line_set = set(line)

        # A secondary construction may touch a primary segment only at
        # its own visible trigger.
        assert line_set & segment_union == {point}

        # Original primary markers remain intact; same-color dots can be crossed.
        assert not (line_set & primary_cells)

        signature = tuple(sorted(line_set))
        assert signature not in line_signatures
        line_signatures.add(signature)
        line_by_active[point] = line

    output = grid.copy()

    for segment in segments:
        for row, col in segment:
            output[row, col] = primary_color

    for line in line_by_active.values():
        for row, col in line:
            output[row, col] = secondary_color

    segment_active_counts = []
    active_slopes = set()

    for index, cells in enumerate(segments):
        count = sum(point in active_secondary for point in cells)
        segment_active_counts.append(count)
        if count > 0:
            active_slopes.add(segment_slopes[index])

    analysis = {
        "background": background,
        "pairs": tuple(pairs),
        "segments": tuple(segments),
        "segment_slopes": tuple(segment_slopes),
        "segment_active_counts": tuple(segment_active_counts),
        "active_secondary": frozenset(active_secondary),
        "inactive_secondary": frozenset(inactive_secondary),
        "unpaired_primary": frozenset(unpaired_primary),
        "active_slopes": frozenset(active_slopes),
        "full_lines": tuple(line_by_active.values()),
    }
    return output, analysis


def _validate_count_value(
    name: str,
    value: int | None,
    lower: int,
    upper: int,
) -> None:
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not bool")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not lower <= int(value) <= upper:
        raise ValueError(
            f"{name} must be between {lower} and {upper}, inclusive"
        )


def _resolve_counts(
    rng: np.random.Generator,
    seed_value: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    _validate_count_value("num_train", num_train, 2, 10)
    _validate_count_value("num_test", num_test, 1, 4)

    if num_train is None and num_test is None:
        test_count = 1 + seed_value % 4
        number_of_train_choices = 10 - test_count
        train_count = (
            test_count
            + 1
            + (seed_value // 4) % number_of_train_choices
        )
    elif num_train is None:
        test_count = int(num_test)
        train_count = int(
            rng.integers(test_count + 1, 11)
        )
    elif num_test is None:
        train_count = int(num_train)
        maximum_test = min(4, train_count - 1)
        test_count = int(rng.integers(1, maximum_test + 1))
    else:
        train_count = int(num_train)
        test_count = int(num_test)

    if not (
        2 <= train_count <= 10
        and 1 <= test_count <= 4
        and train_count > test_count
    ):
        raise ValueError(
            "Counts must satisfy 2 <= num_train <= 10, "
            "1 <= num_test <= 4, and num_train > num_test"
        )

    return train_count, test_count


def _sample_puzzle_parameters(rng, num_train, num_test):
    colors = [int(x) for x in rng.permutation(10)]
    backgrounds = colors[2:]
    mode = int(rng.integers(4))
    if mode == 0:  # One background throughout.
        train_bg = [backgrounds[0]] * num_train
        test_bg = [backgrounds[0]] * num_test
    elif mode == 1:  # Variable, with guaranteed cross-split reuse.
        train_bg = [backgrounds[i % min(num_train, 3)] for i in range(num_train)]
        test_bg = [int(rng.choice(train_bg)) for _ in range(num_test)]
    elif mode == 2:  # A background-only novelty is allowed independently of scale.
        train_bg = [int(rng.choice(backgrounds[:3])) for _ in range(num_train)]
        test_bg = [int(rng.choice(backgrounds[3:])) for _ in range(num_test)]
    else:
        train_bg = [int(rng.choice(backgrounds)) for _ in range(num_train)]
        test_bg = [int(rng.choice(backgrounds)) for _ in range(num_test)]
    rng.shuffle(train_bg)
    # Bands overlap; size and object-count shifts are independent choices.
    size_profiles = ((0, 2), (1, 1), (2, 0), (0, 0), (2, 2), (3, 3), (1, 2))
    count_profiles = ((0, 0), (1, 1), (0, 2), (2, 0), (2, 2), (1, 2))
    return {
        'primary': colors[0], 'secondary': colors[1],
        'backgrounds': (train_bg, test_bg),
        'size_bands': size_profiles[int(rng.integers(len(size_profiles)))],
        'count_bands': count_profiles[int(rng.integers(len(count_profiles)))],
    }


def _sample_shape(rng, band):
    low, high = ((5, 15), (10, 24), (17, 31), (5, 31))[band]
    chance = rng.random()
    if chance < .025:
        return 25, 21
    if chance < .05:
        return 30, 30
    if chance < .25:
        shape = (int(rng.integers(4, 11)), int(rng.integers(17, 31)))
        return shape if rng.random() < .5 else shape[::-1]
    return int(rng.integers(low, high)), int(rng.integers(low, high))


def _sample_length(rng, maximum, minimum):
    minimum = min(minimum, maximum)
    choice = rng.random()
    if choice < .35:
        return int(rng.integers(minimum, min(maximum, max(minimum, 5)) + 1))
    if choice < .65:
        return int(rng.integers(max(minimum, maximum - 3), maximum + 1))
    return int(rng.integers(minimum, maximum + 1))


def _place_segments(rng, shape, specifications):
    """Compose free segments, checking endpoint partners and color conflicts."""
    h, w = shape
    segments, triggers = [], []
    primary_union, secondary_union, endpoints = set(), set(), []
    for slope, trigger_floor, untriggered in specifications:
        for _ in range(120):
            length = _sample_length(rng, min(h, w), max(3, trigger_floor + 3))
            steps = length - 1
            r = int(rng.integers(h - steps))
            c = int(rng.integers(w - steps)) + (steps if slope == -1 else 0)
            segment = _segment_cells((r, c), (r + steps, c + slope * steps))
            a, b = segment[0], segment[-1]
            # No endpoint has any diagonal partner besides its intended mate.
            if any(abs(p[0] - q[0]) == abs(p[1] - q[1])
                   for p in (a, b) for q in endpoints):
                continue
            cells = set(segment)
            if cells & secondary_union:
                continue
            combined = primary_union | cells
            eligible = []
            for point in segment[1:-1]:
                if point in primary_union:
                    continue  # A two-slope trigger has no unique launch direction.
                line = set(_full_diagonal(point, -slope, h, w))
                if line & combined == {point}:
                    eligible.append((point, line))
            if len(eligible) < trigger_floor:
                continue
            if untriggered:
                number = 0
            elif trigger_floor:
                number = int(rng.integers(trigger_floor, min(6, len(eligible), length - 3) + 1))
            else:
                cap = min(6, len(eligible), length - 3)
                roll = rng.random()
                number = 0 if roll < .42 or cap == 0 else (
                    1 if roll < .72 or cap == 1 else int(rng.integers(2, cap + 1)))
            chosen = [eligible[int(i)] for i in rng.permutation(len(eligible))[:number]]
            segments.append(segment)
            endpoints.extend((a, b))
            primary_union.update(cells)
            for point, line in chosen:
                triggers.append(point)
                secondary_union.update(line)
            break
        else:
            return None
    return segments, triggers, primary_union, secondary_union


def _add_distractors(rng, shape, geometry, primary_count, secondary_count, need_inert):
    segments, triggers, primary_union, secondary_union = geometry
    h, w = shape
    endpoints = [p for s in segments for p in (s[0], s[-1])]
    primary_dots = endpoints.copy()
    # Zero is a real choice for each role, independently.
    for index in rng.permutation(h * w):
        if len(primary_dots) - len(endpoints) >= primary_count:
            break
        p = (int(index) // w, int(index) % w)
        if p in primary_union or p in secondary_union:
            continue
        if any(abs(p[0] - q[0]) == abs(p[1] - q[1]) for q in primary_dots):
            continue
        primary_dots.append(p)
    if len(primary_dots) - len(endpoints) < primary_count:
        return None
    free = [(r, c) for r in range(h) for c in range(w)
            if (r, c) not in primary_union and (r, c) not in primary_dots]
    if len(free) < secondary_count:
        return None
    inactive = []
    # Some examples deliberately put inert dots on a secondary continuation or
    # beyond primary endpoints. Neither case initiates a recursive construction.
    if secondary_count:
        pools = [free]
        if rng.random() < .35:
            pools.insert(0, [p for p in free if p in secondary_union])
        elif rng.random() < .5:
            pools.insert(0, [p for p in free if any(
                abs(p[0] - s[0][0]) == abs(p[1] - s[0][1]) and
                abs(p[0] - s[-1][0]) == abs(p[1] - s[-1][1]) for s in segments)])
        for pool in pools:
            if pool:
                inactive.append(pool[int(rng.integers(len(pool)))])
                break
        if need_inert and all(p in secondary_union for p in inactive):
            outside = [p for p in free if p not in secondary_union]
            if not outside:
                return None
            inactive[0] = outside[int(rng.integers(len(outside)))]
        for index in rng.permutation(len(free)):
            if len(inactive) == secondary_count:
                break
            p = free[int(index)]
            if p not in inactive:
                inactive.append(p)
    return primary_dots, triggers + inactive


def _fallback_pair(rng, primary, secondary, background, used):
    """Finite, deterministic emergency construction with every teaching witness.

    Normal free placement almost always succeeds. These translated/reflected
    configurations ensure that a legal request cannot fail a rejection budget.
    """
    for shift in rng.permutation(12):
        grid = np.full((30, 30), background, dtype=np.int8)
        # The three segments have disjoint endpoint diagonal coordinates.
        for p in ((1, 1), (7, 7), (16, 25), (23, 18), (24, 3), (28, 7), (1, 20)):
            grid[p] = primary
        for p in ((3, 3), (5, 5), (18, 23), (2, 20)):
            grid[p] = secondary
        # An inert dot supplies twelve distinct fallback inputs before symmetry.
        grid[0, 15 + int(shift)] = secondary
        for rotation in range(4):
            candidate = np.rot90(grid, rotation).copy()
            if _grid_key(candidate) not in used:
                output, _ = _solve_visible_grid(candidate, primary, secondary)
                return {'input': candidate, 'output': output}
    raise AssertionError('Fallback capacity exceeds the maximum puzzle size')


def _generate_pair(rng, parameters, split_index, background, witness, used):
    # A witness requests observable evidence, never a fixed pair index or layout.
    slopes_needed = witness.get('slopes', ())
    specifications = [(s, 2 if witness.get('multiple') and i == 0 else 1, False)
                      for i, s in enumerate(slopes_needed)]
    if witness.get('zero'):
        specifications.append((int(rng.choice((-1, 1))), 0, True))
    count_band = parameters['count_bands'][split_index]
    count = int(rng.choice(((1, 1, 1, 2), (1, 2, 2, 3, 4), (1, 2, 3, 4, 5, 6, 7))[count_band]))
    count = max(count, len(specifications))
    parallel = int(rng.choice((-1, 1))) if rng.random() < .3 else None
    while len(specifications) < count:
        specifications.append((parallel or int(rng.choice((-1, 1))), 0, False))
    # An all-zero mode produces complete multi-segment examples without launches.
    if not slopes_needed and rng.random() < .22:
        specifications = [(s, 0, True) for s, _, _ in specifications]
    primary_count = int(rng.choice((0, 0, 0, 1, 1, 2, 3, 4)))
    secondary_count = int(rng.choice((0, 0, 0, 1, 1, 2, 3, 5, 8)))
    if witness.get('primary_dot'):
        primary_count = max(primary_count, 1)
    if witness.get('inert'):
        secondary_count = max(secondary_count, 1)
    primary, secondary = parameters['primary'], parameters['secondary']
    for _ in range(48):
        shape = _sample_shape(rng, parameters['size_bands'][split_index])
        if min(shape) < max((floor + 3 for _, floor, _ in specifications), default=3):
            continue
        order = rng.permutation(len(specifications))
        geometry = _place_segments(rng, shape, [specifications[int(i)] for i in order])
        if geometry is None:
            continue
        dots = _add_distractors(rng, shape, geometry, primary_count,
                               secondary_count, witness.get('inert', False))
        if dots is None:
            continue
        grid = np.full(shape, background, dtype=np.int8)
        for p in dots[0]:
            grid[p] = primary
        for p in dots[1]:
            grid[p] = secondary
        # Keep the uniform canvas visibly dominant even in a small, crowded
        # placement. Retry with another canvas before any role inference.
        if 2 * (len(dots[0]) + len(dots[1])) >= grid.size:
            continue
        if _grid_key(grid) in used or _source_input(grid):
            continue
        output, _ = _solve_visible_grid(grid, primary, secondary)
        if np.array_equal(output, grid):
            continue
        return {'input': grid, 'output': output}
    return _fallback_pair(rng, primary, secondary, background, used)


def _infer_puzzle_roles(puzzle):
    """Learn roles from training only, including demonstrations with no secondary."""
    markers = set()
    for pair in puzzle['train']:
        grid = pair['input']
        markers.update(set(map(int, np.unique(grid))) - {_background_color(grid)})
    assert len(markers) == 2
    candidates = []
    for primary in sorted(markers):
        secondary = next(iter(markers - {primary}))
        try:
            works = all(np.array_equal(_solve_visible_grid(pair['input'], primary, secondary)[0],
                                       pair['output']) for pair in puzzle['train'])
        except AssertionError:
            works = False
        if works:
            candidates.append((primary, secondary))
    assert len(candidates) == 1, 'Training must identify the two marker roles uniquely'
    return candidates[0]


def _analyze_puzzle(puzzle, primary, secondary):
    analyses = {'train': [], 'test': []}
    for split in analyses:
        for pair in puzzle[split]:
            output, analysis = _solve_visible_grid(pair['input'], primary, secondary)
            assert np.array_equal(output, pair['output'])
            analyses[split].append(analysis)
    return analyses


def _independent_prediction(train, query):
    """A second, cellwise formulation; receives no test label or generation state.

    This internal consistency check is not an externally supplied reference solver.
    """
    def predict(grid, primary, secondary):
        points = [tuple(map(int, p)) for p in np.argwhere(grid == primary)]
        segments = [(a, b) for i, a in enumerate(points) for b in points[i + 1:]
                    if abs(a[0] - b[0]) == abs(a[1] - b[1])]
        output = grid.copy()
        rows, cols = np.indices(grid.shape)
        launches = []
        for a, b in segments:
            slope = 1 if b[1] > a[1] else -1
            mask = (cols - a[1] == slope * (rows - a[0])) & (rows >= a[0]) & (rows <= b[0])
            launches.extend((int(r), int(c), -slope) for r, c in np.argwhere(mask & (grid == secondary)))
            output[mask] = primary
        for r, c, slope in launches:
            output[cols - c == slope * (rows - r)] = secondary
        return output
    markers = set()
    for pair in train:
        values, counts = np.unique(pair['input'], return_counts=True)
        markers.update(map(int, values[counts != counts.max()]))
    candidates = [(p, s) for p in sorted(markers) for s in sorted(markers) if p != s
                  and all(np.array_equal(predict(pair['input'], p, s), pair['output']) for pair in train)]
    assert len(candidates) == 1
    return predict(query, *candidates[0])


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    inputs, pairs = set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            assert pair['input'].shape == pair['output'].shape
            assert not np.array_equal(pair['input'], pair['output'])
            assert _grid_key(pair['input']) not in inputs and _pair_key(pair) not in pairs
            assert not _source_input(pair['input'])
            inputs.add(_grid_key(pair['input']))
            pairs.add(_pair_key(pair))
    primary, secondary = _infer_puzzle_roles(puzzle)
    analyses = _analyze_puzzle(puzzle, primary, secondary)
    training = analyses['train']
    assert set().union(*(a['active_slopes'] for a in training)) == {-1, 1}
    assert any(max(a['segment_active_counts']) >= 2 for a in training)
    assert any(0 in a['segment_active_counts'] for a in training)
    assert any(a['unpaired_primary'] for a in training)
    assert any(a['inactive_secondary'] - set().union(*(set(line) for line in a['full_lines']))
               for a in training)
    # Both regimes obey one geometric rule. Size/count/background profiles are
    # distribution choices, not new semantic restrictions on an individual test.
    for split in analyses:
        for a in analyses[split]:
            assert a['background'] not in (primary, secondary)
            assert all(len(s) >= 3 for s in a['segments'])
    for pair in puzzle['test']:
        predicted = _independent_prediction(puzzle['train'], pair['input'].copy())
        assert np.array_equal(predicted, pair['output'])


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, int(seed), num_train, num_test)
    parameters = _sample_puzzle_parameters(rng, train_count, test_count)
    witnesses = [{} for _ in range(train_count)]
    # Randomly distribute the minimal teaching evidence, then shuffle all pairs.
    active_slots = rng.choice(train_count, size=2, replace=False)
    for i, slope in enumerate(rng.permutation((-1, 1))):
        slot = int(active_slots[i])
        witnesses[slot]['slopes'] = (int(slope),)
        witnesses[slot]['multiple'] = i == 0
    for name in ('zero', 'inert', 'primary_dot'):
        witnesses[int(rng.integers(train_count))][name] = True
    puzzle = {'train': [], 'test': []}
    used = set()
    for split_index, (split, count) in enumerate((('train', train_count), ('test', test_count))):
        for i in range(count):
            pair = _generate_pair(rng, parameters, split_index,
                                  parameters['backgrounds'][split_index][i],
                                  witnesses[i] if split == 'train' else {}, used)
            puzzle[split].append(pair)
            used.add(_grid_key(pair['input']))
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _whole_puzzle_key(puzzle):
    return tuple(tuple(_pair_key(pair) for pair in puzzle[s]) for s in ('train', 'test'))


def _measure_puzzle(puzzle):
    """Observable task features used by self_test and scratch coverage reports."""
    primary, secondary = _infer_puzzle_roles(puzzle)
    analyses = _analyze_puzzle(puzzle, primary, secondary)
    records = []
    for split in ('train', 'test'):
        for index, (pair, a) in enumerate(zip(puzzle[split], analyses[split])):
            grid = pair['input']
            lengths = [len(s) for s in a['segments']]
            counts = list(a['segment_active_counts'])
            lines = [set(line) for line in a['full_lines']]
            line_union = set().union(*lines)
            records.append({
                'split': split, 'index': index, 'shape': tuple(map(int, grid.shape)),
                'roles': (primary, secondary), 'background': a['background'],
                'segments': len(lengths), 'lengths': lengths, 'triggers': counts,
                'slopes': list(a['segment_slopes']),
                'primary_distractors': len(a['unpaired_primary']),
                'secondary_distractors': len(a['inactive_secondary']),
                'absent_secondary': not np.any(grid == secondary),
                'mixed_triggers': 0 in counts and max(counts) > 0,
                'all_untriggered_multi': len(counts) > 1 and max(counts) == 0,
                'unequal_triggers': len(set(counts)) > 1,
                'secondary_majority': int(np.sum(grid == secondary)) > int(np.sum(grid == primary)),
                'parallel_multi': len(lengths) > 1 and len(set(a['segment_slopes'])) == 1,
                'mixed_slopes': len(set(a['segment_slopes'])) == 2,
                'length_spread': max(lengths) - min(lengths),
                'relative_length': max(lengths) / min(grid.shape),
                'boundary_endpoint': any(r in (0, grid.shape[0]-1) or c in (0, grid.shape[1]-1)
                                         for edge in a['pairs'] for r, c in edge),
                'near_endpoint_trigger': any(s[1] in a['active_secondary'] or s[-2] in a['active_secondary']
                                             for s in a['segments']),
                'secondary_intersection': any(x & y for i, x in enumerate(lines) for y in lines[i+1:]),
                'primary_crossing': any(set(x) & set(y) for i, x in enumerate(a['segments'])
                                        for y in a['segments'][i+1:]),
                'secondary_through_inert': bool(a['inactive_secondary'] & line_union),
                'beyond_endpoint': any(abs(p[0]-s[0][0]) == abs(p[1]-s[0][1]) and
                                       abs(p[0]-s[-1][0]) == abs(p[1]-s[-1][1])
                                       for p in a['inactive_secondary'] for s in a['segments']),
                'simple': len(lengths) == 1 and sum(counts) <= 2 and
                          len(a['unpaired_primary']) + len(a['inactive_secondary']) <= 2,
            })
    return records


def self_test() -> None:
    # Frozen source-output digests make these regressions independent of the
    # replacement drawing implementation, including the two-color example.
    source_digests = (
        '58ac68dc905f51b2f36730f0440761a4821dfe366991ea9709c548de541adc4c',
        'e762a0aed6ccd5e1b404759723315c28c167db7287b0a01c7d0a300bb157bd5d',
        'cea94ccc2ab8548cebe10cd78ec8ad64b6304033de2c0d86db94d92e19739953',
        '469b4e88c6f123d2a3aedcf8cce90a9e2abd5cb7473b5942e72456aaf633358a',
        'e740d7be8c785e1a32485da0ebdf76b58a43bba54b89c9ce3c85906bd6f814f8',
        '6e58eb73e033655dc77f73463b99a3879ab3961884cc90d5ab1bb0909b724c9f',
    )
    for (shape, background, dots), digest in zip(_SOURCE_INPUTS, source_digests):
        grid = np.full(shape, background, dtype=np.int8)
        for r, c, color in dots:
            grid[r, c] = color
        assert _source_input(grid)
        predicted, _ = _solve_visible_grid(grid, 1, 6)
        assert hashlib.sha256(predicted.tobytes()).hexdigest() == digest

    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records = [r for p in puzzles for r in _measure_puzzle(p)]
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    assert len({_whole_puzzle_key(p) for p in puzzles}) == 42
    assert len({r['shape'] for r in records}) >= 80
    assert len({r['roles'] for r in records}) >= 20
    for split in ('train', 'test'):
        rows = [r for r in records if r['split'] == split]
        assert {1, 2, 3, 4, 5} <= {r['segments'] for r in rows}
        assert {0, 1, 2, 3} <= {n for r in rows for n in r['triggers']}
        assert {0, 1, 2, 3} <= {r['primary_distractors'] for r in rows}
        assert {0, 1, 2, 3, 5, 8} <= {r['secondary_distractors'] for r in rows}
        assert min(n for r in rows for n in r['lengths']) == 3
        assert max(n for r in rows for n in r['lengths']) >= 25
        for name in ('absent_secondary', 'mixed_triggers', 'all_untriggered_multi',
                     'unequal_triggers', 'secondary_majority', 'parallel_multi', 'mixed_slopes',
                     'boundary_endpoint', 'near_endpoint_trigger', 'secondary_intersection',
                     'primary_crossing', 'secondary_through_inert', 'beyond_endpoint', 'simple'):
            assert any(r[name] for r in rows), (split, name)
        assert any(r['length_spread'] >= 10 for r in rows)
        assert any(17 <= d <= 19 for r in rows for d in r['shape'])
        assert any(max(r['shape']) / min(r['shape']) >= 3 for r in rows)
    assert (25, 21) in {r['shape'] for r in records}
    assert any(set(r['roles']) == {1, 6} for r in records)
    bg_relationships = set()
    area_shifts, count_shifts = set(), set()
    for puzzle in puzzles:
        rows = _measure_puzzle(puzzle)
        tr = [r for r in rows if r['split'] == 'train']
        te = [r for r in rows if r['split'] == 'test']
        b1, b2 = {r['background'] for r in tr}, {r['background'] for r in te}
        bg_relationships.add('constant' if len(b1 | b2) == 1 else 'new' if b1.isdisjoint(b2) else 'shared')
        area_ratio = np.mean([np.prod(r['shape']) for r in te]) / np.mean([np.prod(r['shape']) for r in tr])
        count_ratio = np.mean([r['segments'] for r in te]) / np.mean([r['segments'] for r in tr])
        area_shifts.add('up' if area_ratio > 1.4 else 'down' if area_ratio < .7 else 'similar')
        count_shifts.add('up' if count_ratio > 1.3 else 'down' if count_ratio < .75 else 'similar')
        assert len({r['shape'] for r in rows}) >= 2
    assert bg_relationships == {'constant', 'new', 'shared'}
    assert area_shifts == count_shifts == {'up', 'down', 'similar'}
    for name in ('segments', 'triggers', 'lengths', 'primary_distractors', 'secondary_distractors'):
        assert sum(len({str(r[name]) for r in _measure_puzzle(p)}) > 1 for p in puzzles) >= 30
    assert len({r['segments'] for r in records if r['split'] == 'train' and r['index'] == 0}) >= 4
    assert sum(r['mixed_triggers'] and r['length_spread'] >= 8 for r in records) >= 10
    assert any(r['absent_secondary'] and r['parallel_multi'] for r in records)
    assert any(r['secondary_majority'] and r['segments'] >= 2 for r in records)

    # Exercise the finite fallback budget, including the largest legal puzzle.
    used = set()
    for index in range(14):
        pair = _fallback_pair(np.random.default_rng(index), 1, 6, 0, used)
        used.add(_grid_key(pair['input']))
        _, a = _solve_visible_grid(pair['input'], 1, 6)
        assert a['active_slopes'] == {-1, 1} and 0 in a['segment_active_counts']
    assert len(used) == 14

    # An unmarked same-color crossing is valid; a dot at the intersection has
    # two possible launch slopes and must be rejected. Endpoints stay unique.
    crossing = np.zeros((12, 12), dtype=np.int8)
    for point in ((1, 1), (8, 8), (2, 10), (9, 3)):
        crossing[point] = 1
    _solve_visible_grid(crossing, 1, 6)
    crossing[6, 6] = 6
    try:
        _solve_visible_grid(crossing, 1, 6)
    except AssertionError:
        pass
    else:
        raise AssertionError('An ambiguous two-slope trigger was accepted')
    # The same crossing with a unique trigger can still create a color conflict
    # elsewhere along its continuation, which must also be rejected.
    conflict = np.zeros((12, 12), dtype=np.int8)
    for point in ((1, 1), (8, 8), (1, 5), (7, 11)):
        conflict[point] = 1
    conflict[4, 4] = 6
    try:
        _solve_visible_grid(conflict, 1, 6)
    except AssertionError:
        pass
    else:
        raise AssertionError('A secondary/primary color conflict was accepted')
    for seed, train, test in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                              (1101, None, 1), (1102, None, 4), (1103, 2, None), (1104, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        assert train is None or len(puzzle['train']) == train
        assert test is None or len(puzzle['test']) == test
        assert _whole_puzzle_key(puzzle) == _whole_puzzle_key(generate_puzzle(seed, train, test))
    assert _whole_puzzle_key(generate_puzzle(1202)) == _whole_puzzle_key(generate_puzzle(1202))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4), (True, 1), (3, 1.5)):
        try:
            generate_puzzle(1401, train, test)
        except ValueError:
            continue
        raise AssertionError('Invalid count request was accepted')
    # Mutations test actual rejection, including an absent-color role conflict.
    bad = {s: [{'input': p['input'].copy(), 'output': p['output'].copy()} for p in puzzles[0][s]]
           for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('A corrupted target was accepted')
