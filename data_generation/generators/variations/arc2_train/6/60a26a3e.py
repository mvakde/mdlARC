"""Synthetic hollow-diamond alignment puzzles."""
import numpy as np

# The four original inputs, identified by shape and hollow centers. Their
# colors are background 0 and marker 2. Those shapes remain legal.
_SOURCE_INPUTS = {
    ((14, 16), ((3, 3), (3, 8), (3, 13), (9, 8), (10, 2))),
    ((10, 15), ((1, 3), (1, 10), (5, 10))),
    ((12, 15), ((1, 8), (4, 4), (4, 11), (9, 4), (9, 11))),
    ((9, 15), ((2, 2), (4, 7), (6, 2), (6, 10))),
}


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _choose_counts(seed, rng, num_train, num_test):
    num_train = _checked_count('num_train', num_train, 2, 10)
    num_test = _checked_count('num_test', num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        minimum = max(2, num_test + 1)
        num_train = minimum + (seed // 4) % (11 - minimum)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return num_train, num_test


def _tips(center):
    r, c = center
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _footprint(center):
    return {center, *_tips(center)}


def _detect_centers(grid, background, marker):
    h, w = grid.shape
    return [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)
            if grid[r, c] == background and grid[r - 1, c] == marker
            and grid[r + 1, c] == marker and grid[r, c - 1] == marker
            and grid[r, c + 1] == marker]


def _edge_data(centers):
    rows, columns = {}, {}
    for i, (r, c) in enumerate(centers):
        rows.setdefault(r, []).append(i)
        columns.setdefault(c, []).append(i)
    edges = []
    for indices in rows.values():
        indices.sort(key=lambda i: centers[i][1])
        edges.extend((a, b, 'h') for a, b in zip(indices, indices[1:]))
    for indices in columns.values():
        indices.sort(key=lambda i: centers[i][0])
        edges.extend((a, b, 'v') for a, b in zip(indices, indices[1:]))
    return edges


def _corridor_cells(centers):
    cells = set()
    for a, b, axis in _edge_data(centers):
        r1, c1 = centers[a]
        r2, c2 = centers[b]
        if axis == 'h':
            cells.update((r1, c) for c in range(c1 + 2, c2 - 1))
        else:
            cells.update((r, c1) for r in range(r1 + 2, r2 - 1))
    return cells


def _layout_grid(height, width, centers, background, marker):
    centers = [tuple(map(int, center)) for center in centers]
    if len(set(centers)) != len(centers):
        return None
    occupied = set()
    for center in centers:
        r, c = center
        if not (1 <= r < height - 1 and 1 <= c < width - 1):
            return None
        footprint = _footprint(center)
        if occupied & footprint:
            return None
        occupied.update(footprint)
    for a, b, axis in _edge_data(centers):
        gap = abs(centers[a][1] - centers[b][1]) if axis == 'h' else abs(centers[a][0] - centers[b][0])
        if gap < 4:
            return None
    grid = np.full((height, width), background, dtype=np.int8)
    for center in centers:
        for r, c in _tips(center):
            grid[r, c] = marker
    if set(_detect_centers(grid, background, marker)) != set(centers):
        return None
    corridor = _corridor_cells(centers)
    if not corridor or any(grid[r, c] != background for r, c in corridor):
        return None
    return grid


def _apply_rule(grid, background, marker, connector):
    result = grid.copy()
    for r, c in _corridor_cells(_detect_centers(grid, background, marker)):
        result[r, c] = connector
    return result


def _gaps(rng, count, budget=25):
    gaps = []
    for i in range(count):
        remaining = count - i - 1
        highest = min(14, budget - 4 * remaining)
        gap = int(rng.integers(4, highest + 1))
        gaps.append(gap)
        budget -= gap
    return gaps


def _accumulate(gaps):
    result = [0]
    for gap in gaps:
        result.append(result[-1] + gap)
    return result


def _core(rng, kind):
    if kind in ('line', 'long_line'):
        if kind == 'long_line':
            segment_count = int(rng.choice((3, 4), p=(0.65, 0.35)))
        else:
            segment_count = int(rng.choice((1, 2, 3, 4), p=(0.48, 0.22, 0.20, 0.10)))
        positions = _accumulate(_gaps(rng, segment_count))
        points = [(0, c) for c in positions]
    elif kind == 'elbow':
        horizontal = _accumulate(_gaps(rng, int(rng.integers(1, 4))))
        vertical = _accumulate(_gaps(rng, int(rng.integers(1, 4))))
        hs, vs = int(rng.choice((-1, 1))), int(rng.choice((-1, 1)))
        points = [(0, hs * c) for c in horizontal] + [(vs * r, 0) for r in vertical[1:]]
    elif kind == 't':
        columns = _accumulate(_gaps(rng, int(rng.integers(2, 5))))
        pivot = int(rng.integers(1, len(columns) - 1))
        branch = _accumulate(_gaps(rng, int(rng.integers(1, 3)), 18))[1:]
        sign = int(rng.choice((-1, 1)))
        points = [(0, c) for c in columns] + [(sign * r, columns[pivot]) for r in branch]
    elif kind == 'fourway':
        arms = [_accumulate(_gaps(rng, int(rng.integers(1, 3)), 12))[1:] for _ in range(4)]
        points = [(0, 0)] + [(0, -c) for c in arms[0]] + [(0, c) for c in arms[1]]
        points += [(-r, 0) for r in arms[2]] + [(r, 0) for r in arms[3]]
    elif kind == 'loop':
        rows = _accumulate(_gaps(rng, int(rng.integers(1, 3)), 23))
        columns = _accumulate(_gaps(rng, int(rng.integers(1, 3)), 23))
        full = bool(rng.integers(2))
        points = [(r, c) for i, r in enumerate(rows) for j, c in enumerate(columns)
                  if full or i in (0, len(rows) - 1) or j in (0, len(columns) - 1)]
    elif kind == 'crossing':
        left, right, top, bottom = [int(rng.integers(4, 12)) for _ in range(4)]
        points = [(0, -left), (0, right), (-top, 0), (bottom, 0)]
    elif kind == 'multi':
        gap1 = int(rng.integers(5, 13))
        gap2 = int(rng.integers(5, 13))
        offset = int(rng.choice((2, 3, 4)))
        if offset + gap2 == gap1:
            gap2 += 1
        separation = int(rng.integers(5, 13))
        points = [(0, 0), (0, gap1), (separation, offset), (separation, offset + gap2)]
    else:
        raise ValueError('unknown topology')
    if bool(rng.integers(2)):
        points = [(c, r) for r, c in points]
    return points


def _choose_size_and_place(rng, relative):
    rows = [r for r, _ in relative]
    columns = [c for _, c in relative]
    minimum_height = max(7, max(rows) - min(rows) + 3)
    minimum_width = max(7, max(columns) - min(columns) + 3)
    if max(minimum_height, minimum_width) > 30:
        return None
    familiar_sizes = [(9, 15), (10, 15), (12, 15), (14, 16)]
    fitting_sizes = [(h, w) for h, w in familiar_sizes
                     if h >= minimum_height and w >= minimum_width]
    if fitting_sizes and rng.random() < 0.50:
        weights = np.array([0.45, 0.25, 0.18, 0.12], dtype=float)
        weights = np.array([weights[familiar_sizes.index(size)] for size in fitting_sizes])
        weights /= weights.sum()
        height, width = fitting_sizes[int(rng.choice(len(fitting_sizes), p=weights))]
    else:
        mode = int(rng.choice(3, p=[0.48, 0.37, 0.15]))
        slack_limit = (4, 11, 30)[mode]
        height = int(rng.integers(minimum_height, min(30, minimum_height + slack_limit) + 1))
        width = int(rng.integers(minimum_width, min(30, minimum_width + slack_limit) + 1))
    row_shift = int(rng.integers(1 - min(rows), height - 1 - max(rows)))
    column_shift = int(rng.integers(1 - min(columns), width - 1 - max(columns)))
    return height, width, [(r + row_shift, c + column_shift) for r, c in relative]


def _add_isolates(rng, height, width, centers, background, marker, desired):
    centers = list(centers)
    corridor = _corridor_cells(centers)
    for _ in range(desired):
        used_rows = {r for r, _ in centers}
        used_columns = {c for _, c in centers}
        occupied = set().union(*(_footprint(center) for center in centers))
        candidates = []
        for r in range(1, height - 1):
            if r in used_rows:
                continue
            for c in range(1, width - 1):
                if c in used_columns:
                    continue
                footprint = _footprint((r, c))
                if footprint & (occupied | corridor):
                    continue
                distance = min(abs(fr - cr) + abs(fc - cc)
                               for fr, fc in footprint for cr, cc in corridor)
                off_axis = any(abs(r - ar) == 1 for ar, _ in centers) or any(
                    abs(c - ac) == 1 for _, ac in centers)
                candidates.append(((r, c), distance, off_axis))
        if not candidates:
            break
        preference = ('near', 'far', 'off', 'any')[int(rng.integers(4))]
        if preference == 'near':
            preferred = [item for item in candidates if item[1] <= 2]
        elif preference == 'far':
            preferred = [item for item in candidates if item[1] >= 6]
        elif preference == 'off':
            preferred = [item for item in candidates if item[2]]
        else:
            preferred = candidates
        pools = [preferred, candidates] if preferred else [candidates]
        found = False
        for pool in pools:
            for item_index in rng.permutation(len(pool))[:60]:
                candidate = pool[int(item_index)][0]
                proposed = centers + [candidate]
                if _layout_grid(height, width, proposed, background, marker) is not None:
                    centers = proposed
                    found = True
                    break
            if found:
                break
        if not found:
            break
    return centers


def _grid_key(grid):
    return (tuple(int(x) for x in grid.shape), tuple(int(x) for x in grid.flat))


def _source_copy(grid, centers, background, marker):
    return background == 0 and marker == 2 and (
        tuple(int(x) for x in grid.shape), tuple(sorted(centers))) in _SOURCE_INPUTS


def _make_pair(rng, kind, background, marker, connector, test):
    for _ in range(120):
        relative = _core(rng, kind)
        placed = _choose_size_and_place(rng, relative)
        if placed is None:
            continue
        height, width, centers = placed
        if _layout_grid(height, width, centers, background, marker) is None:
            continue
        if kind == 'line':
            probabilities = [0.68, 0.22, 0.08, 0.02]
        else:
            probabilities = [0.16, 0.37, 0.31, 0.16] if test else [0.26, 0.39, 0.25, 0.10]
        desired = int(rng.choice(4, p=probabilities))
        centers = _add_isolates(rng, height, width, centers, background, marker, desired)
        grid = _layout_grid(height, width, centers, background, marker)
        if grid is None or _source_copy(grid, centers, background, marker):
            continue
        return {'input': grid, 'output': _apply_rule(grid, background, marker, connector)}
    raise RuntimeError('could not construct pair')


def _infer_roles(input_grid, output_grid):
    values, counts = np.unique(input_grid, return_counts=True)
    assert len(values) == 2
    background_index = int(np.argmax(counts))
    assert counts[background_index] > counts[1 - background_index]
    background = int(values[background_index])
    marker = int(values[1 - background_index])
    additions = set(int(v) for v in np.unique(output_grid)) - {background, marker}
    assert len(additions) == 1
    return background, marker, additions.pop()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _choose_counts(seed, rng, num_train, num_test)
    connector = int(rng.integers(10))
    other_colors = [c for c in range(10) if c != connector]
    fixed_colors = bool(rng.random() < 0.24)
    base_colors = rng.choice(other_colors, size=2, replace=False)
    topology_names = ('line', 'long_line', 'elbow', 't', 'fourway', 'loop', 'crossing', 'multi')
    first_train = ('elbow', 't', 'fourway', 'loop', 'crossing')[int(rng.integers(5))]
    puzzle = {'train': [], 'test': []}
    used = set()
    for split, count in (('train', train_count), ('test', test_count)):
        for index in range(count):
            pair = None
            for _ in range(35):
                if split == 'train' and index == 0:
                    kind = first_train
                elif split == 'test' and rng.random() < 0.7:
                    kind = ('fourway', 'loop', 'crossing', 'multi', 'elbow', 'long_line')[int(rng.integers(6))]
                else:
                    kind = topology_names[int(rng.integers(len(topology_names)))]
                if fixed_colors:
                    background, marker = map(int, base_colors)
                else:
                    background, marker = map(int, rng.choice(other_colors, size=2, replace=False))
                child_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                try:
                    candidate = _make_pair(child_rng, kind, background, marker, connector, split == 'test')
                except RuntimeError:
                    continue
                key = _grid_key(candidate['input'])
                if key not in used:
                    used.add(key)
                    pair = candidate
                    break
            if pair is None:
                raise RuntimeError('could not construct unique puzzle')
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen_inputs, seen_pairs = set(), set()
    connector = None
    training_axes = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = pair['input'], pair['output']
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            assert inp.shape == out.shape
            input_key, output_key = _grid_key(inp), _grid_key(out)
            assert input_key not in seen_inputs
            assert (input_key, output_key) not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add((input_key, output_key))
            background, marker, current_connector = _infer_roles(inp, out)
            if connector is None:
                connector = current_connector
            assert current_connector == connector
            centers = _detect_centers(inp, background, marker)
            assert 2 <= len(centers) <= 12
            assert not _source_copy(inp, centers, background, marker)
            expected_markers = set().union(*(_tips(center) for center in centers))
            actual_markers = {tuple(map(int, loc)) for loc in np.argwhere(inp == marker)}
            assert len(expected_markers) == 4 * len(centers)
            assert expected_markers == actual_markers
            edges = _edge_data(centers)
            assert edges
            assert all((abs(centers[a][1] - centers[b][1]) if axis == 'h'
                        else abs(centers[a][0] - centers[b][0])) >= 4
                       for a, b, axis in edges)
            corridor = _corridor_cells(centers)
            assert corridor and all(inp[r, c] == background for r, c in corridor)
            expected = _apply_rule(inp, background, marker, connector)
            assert np.array_equal(out, expected)
            assert {tuple(map(int, loc)) for loc in np.argwhere(out == connector)} == corridor
            assert all(out[r, c] == background for r, c in centers)
            if split == 'train':
                training_axes.update(axis for _, _, axis in edges)
    assert training_axes == {'h', 'v'}


def _puzzle_key(puzzle):
    return tuple((split, tuple((_grid_key(p['input']), _grid_key(p['output']))
                               for p in puzzle[split])) for split in ('train', 'test'))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(generate_puzzle(123456, 7, 3)) == _puzzle_key(generate_puzzle(123456, 7, 3))
    shapes, palettes, layouts, center_counts, gaps = set(), set(), set(), set(), set()
    singleton_counts, axes_sets, within_color_changes = set(), set(), 0
    review_counts = {'close': 0, 'far': 0, 'off_axis': 0, 'line_of_four': 0,
                     'hub': 0, 'multiple_groups': 0, 'large_loop': 0,
                     'crossing': 0, 'large_test': 0, 'fixed_palette': 0,
                     'dense': 0}
    familiar_shapes = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        roles = []
        largest_training_height = max(pair['input'].shape[0] for pair in puzzle['train'])
        review_counts['large_test'] += any(
            pair['input'].shape[0] > largest_training_height for pair in puzzle['test'])
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                background, marker, connector = _infer_roles(inp, out)
                centers = _detect_centers(inp, background, marker)
                edges = _edge_data(centers)
                roles.append((background, marker, connector))
                shapes.add(tuple(inp.shape))
                familiar_shapes.add(tuple(inp.shape))
                palettes.add((background, marker, connector))
                center_counts.add(len(centers))
                axes_sets.add(frozenset(axis for _, _, axis in edges))
                minimum_r = min(r for r, _ in centers)
                minimum_c = min(c for _, c in centers)
                layouts.add(tuple(sorted((r - minimum_r, c - minimum_c) for r, c in centers)))
                degrees = [0] * len(centers)
                adjacency = [set() for _ in centers]
                for a, b, axis in edges:
                    degrees[a] += 1
                    degrees[b] += 1
                    adjacency[a].add(b)
                    adjacency[b].add(a)
                    gaps.add(abs(centers[a][1] - centers[b][1]) - 3 if axis == 'h'
                             else abs(centers[a][0] - centers[b][0]) - 3)
                singleton_counts.add(degrees.count(0))
                corridor = _corridor_cells(centers)
                for i, degree in enumerate(degrees):
                    if degree:
                        continue
                    r, c = centers[i]
                    distance = min(abs(fr - cr) + abs(fc - cc)
                                   for fr, fc in _footprint((r, c)) for cr, cc in corridor)
                    review_counts['close'] += distance <= 2
                    review_counts['far'] += distance >= 6
                    review_counts['off_axis'] += any(
                        abs(r - rr) == 1 or abs(c - cc) == 1
                        for j, (rr, cc) in enumerate(centers) if j != i)
                review_counts['line_of_four'] += any(
                    sum(rr == r for rr, _ in centers) >= 4 for r, _ in centers) or any(
                    sum(cc == c for _, cc in centers) >= 4 for _, c in centers)
                review_counts['hub'] += max(degrees) == 4
                review_counts['dense'] += 4 * len(centers) / inp.size >= 0.12
                unseen = set(range(len(centers)))
                components = []
                while unseen:
                    start = unseen.pop()
                    component = {start}
                    stack = [start]
                    while stack:
                        for neighbor in adjacency[stack.pop()]:
                            if neighbor in unseen:
                                unseen.remove(neighbor)
                                component.add(neighbor)
                                stack.append(neighbor)
                    components.append(component)
                review_counts['multiple_groups'] += sum(len(c) > 1 for c in components) >= 2
                review_counts['large_loop'] += any(
                    len(c) >= 6 and sum(degrees[i] for i in c) // 2 >= len(c)
                    for c in components)
                review_counts['crossing'] += any(
                    (r, c) not in centers and (r, c - 1) in corridor
                    and (r, c + 1) in corridor and (r - 1, c) in corridor
                    and (r + 1, c) in corridor for r, c in corridor)
                predicted = inp.copy()
                for r, c in _corridor_cells(centers):
                    predicted[r, c] = connector
                assert np.array_equal(predicted, out)
        within_color_changes += len(set(roles)) > 1
        review_counts['fixed_palette'] += len(set(roles)) == 1
    assert len(shapes) >= 45 and len(palettes) >= 25 and len(layouts) >= 80
    assert {2, 3, 4, 5}.issubset(center_counts)
    assert {0, 1, 2}.issubset(singleton_counts)
    assert len(gaps) >= 9
    assert {frozenset({'h'}), frozenset({'v'}), frozenset({'h', 'v'})}.issubset(axes_sets)
    assert within_color_changes >= 20
    assert review_counts['fixed_palette'] >= 2
    assert {(9, 15), (10, 15), (12, 15), (14, 16)}.issubset(familiar_shapes)
    assert all(count > 0 for count in review_counts.values())
    for index, (train_count, test_count) in enumerate(((2, 1), (10, 1), (10, 4),
                                                       (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(2000 + index, train_count, test_count)
        if train_count is not None:
            assert len(puzzle['train']) == train_count
        if test_count is not None:
            assert len(puzzle['test']) == test_count
        validate_generated_puzzle(puzzle)
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(4000, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f'expected ValueError for {args}')
