"""Generate projection-to-rectangle ARC puzzles."""

import numpy as np


_DIRECTIONS = ("up", "down", "left", "right")
# Exact source pairs, encoded as slash-separated rows. This permits source
# palettes and dimensions while excluding copies of the actual examples.
_SOURCE_PAIRS = (
    (
        '111121111111/111111111111/111111111111/111111111111/111444411111/111444411111/111444411121/111444411111/111111111111/111111111111/111111111111/121111111111/111111111111/111111111111',
        '111121111111/111121111111/111121111111/111121111111/111444411111/111444411111/111444422221/111444411111/111111111111/111111111111/111111111111/121111111111/111111111111/111111111111',
    ),
    (
        '222222222222/222222222222/222111222222/222111222222/222111222222/222222222222/222222222222/222222222222/222822222222/222222222222',
        '222222222222/222222222222/222111222222/222111222222/222111222222/222822222222/222822222222/222822222222/222222222222',
    ),
    (
        '111111111111111111/111111114111111111/111111111111111111/111111111114111111/111111111111111411/114111155555111111/111111155555111111/111111155555111111/111111155555111111/111111111111111111/111111111111111111/111111111111141111/111111111111111111/111111111111111111',
        '111111111111111111/111111114111111111/111111114111111111/111111114114111111/111111114114111411/114444455555111111/111111155555111111/111111155555111111/111111155555111111/111111111111111111/111111111111111111/111111111111141111/111111111111111111/111111111111111111',
    ),
    (
        '888888888888/883338888888/883338888888/883338888488/883338888888/888888888888/888888888888/888888848888/888888888888',
        '888888888888/883338888888/883338888888/883334444488/883338888888/888888888888/888888888888/888888848888/888888888888',
    ),
    (
        '222222222222222222222/222222222122222222222/221222222222222221222/222222222222212222222/222222222222222222222/222222288888888222222/222222288888888222222/222222288888888222212/222222288888888222222/221222288888888222222/222222288888888222222/222222222222222222222/222222222222222222222/222222222222222122222/222222222222222222222/222222222222222222222/222222222222222222222/222122222221222222222/222222222222222222222',
        '222222222222222222222/222222222122222222222/221222222122222221222/222222222122212222222/222222222122212222222/222222288888888222222/222222288888888222222/222222288888888111112/222222288888888222222/221111188888888222222/222222288888888222222/222222222221222222222/222222222221222222222/222222222221222122222/222222222221222222222/222222222221222222222/222222222221222222222/222122222221222222222/222222222222222222222',
    ),
)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _resolve_counts(rng, num_train, num_test):
    if num_train is not None:
        if not _integer(num_train) or not 2 <= int(num_train) <= 10:
            raise ValueError("num_train must be an integer from 2 through 10")
        num_train = int(num_train)
    if num_test is not None:
        if not _integer(num_test) or not 1 <= int(num_test) <= 4:
            raise ValueError("num_test must be an integer from 1 through 4")
        num_test = int(num_test)
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
    elif num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    else:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    return num_train, num_test


def _encoded_grid(grid):
    return '/'.join(''.join(str(int(value)) for value in row) for row in grid)


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _isolated(points):
    if len(set(points)) != len(points):
        return False
    return all(abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1
               for i, a in enumerate(points) for b in points[i + 1:])


def _point(direction, lane, distance, box):
    r0, r1, c0, c1 = box
    if direction == 'up':
        return r0 - distance, lane
    if direction == 'down':
        return r1 + distance, lane
    if direction == 'left':
        return lane, c0 - distance
    return lane, c1 + distance


def _path(point, box):
    row, col = point
    r0, r1, c0, c1 = box
    if c0 <= col <= c1:
        if row < r0:
            return 'up', [(r, col) for r in range(row + 1, r0)]
        if row > r1:
            return 'down', [(r, col) for r in range(r1 + 1, row)]
    if r0 <= row <= r1:
        if col < c0:
            return 'left', [(row, c) for c in range(col + 1, c0)]
        if col > c1:
            return 'right', [(row, c) for c in range(c1 + 1, col)]
    return 'distractor', []


def _can_add_active(point, points, box, allow_collinear):
    if not _isolated(points + [point]):
        return False
    if allow_collinear:
        return True
    direction, _ = _path(point, box)
    lane = point[1] if direction in ('up', 'down') else point[0]
    for other in points:
        other_direction, _ = _path(other, box)
        other_lane = other[1] if other_direction in ('up', 'down') else other[0]
        if (direction, lane) == (other_direction, other_lane):
            return False
    return True


def _size(rng, regime):
    weights = (0.40, 0.38, 0.22) if regime == 'source_train' else (0.28, 0.40, 0.32)
    band = int(rng.choice(3, p=weights))
    intervals = ((5, 12), (10, 22), (20, 30))
    low, high = intervals[band]
    return int(rng.integers(low, high + 1)), int(rng.integers(low, high + 1))


def _extent(rng, canvas):
    mode = int(rng.integers(0, 4))
    if mode == 0:
        low, high = 2, min(4, canvas - 1)
    elif mode == 1:
        low, high = 3, min(9, canvas - 1)
    elif mode == 2:
        low, high = max(2, canvas // 3), min(canvas - 1, max(2, 2 * canvas // 3))
    else:
        low, high = max(2, canvas // 2), min(canvas - 1, max(2, 4 * canvas // 5))
    if low > high:
        low = high
    return int(rng.integers(low, high + 1))


def _pick_roles(rng, fixed_palette):
    palette = (np.array(fixed_palette) if fixed_palette is not None
               else rng.choice(10, size=3, replace=False))
    order = rng.permutation(3)
    return tuple(int(palette[i]) for i in order)


def _generate_pair(rng, regime, roles, mandatory=(), force_distractor=False,
                   force_multi=False, force_collinear=False, simple=False,
                   allow_multi=True, allow_collinear=True):
    background, rectangle_color, main_marker = roles
    for _ in range(1000):
        height, width = _size(rng, regime)
        rh, rw = _extent(rng, height), _extent(rng, width)
        if simple:
            directions = [str(rng.choice(_DIRECTIONS))] if not mandatory else list(mandatory)
            active_target = max(1, len(directions))
            inactive_target = 0
        else:
            directions = list(mandatory)
            others = [d for d in _DIRECTIONS if d not in directions]
            extra = int(rng.integers(0, len(others) + 1)) if not mandatory else int(rng.integers(0, min(2, len(others)) + 1))
            directions.extend(str(d) for d in rng.choice(others, size=extra, replace=False)) if extra else None
            if not directions:
                directions = [str(rng.choice(_DIRECTIONS))]
            band = int(rng.choice(3, p=(0.45, 0.38, 0.17)))
            low, high = ((1, 4), (5, 8), (9, 13))[band]
            active_target = max(len(directions), int(rng.integers(low, high + 1)))
            inactive_target = int(rng.choice((0, 1, 2, 3, 4, 5, 6), p=(.25, .20, .17, .15, .11, .08, .04)))
            if force_distractor:
                inactive_target = max(1, inactive_target)
        multi = force_multi or (allow_multi and not simple and rng.random() < .28)
        collinear = force_collinear or (allow_collinear and not simple and rng.random() < .25)
        if collinear:
            active_target = max(active_target, 2)
        if multi:
            active_target = max(active_target, 3 if collinear else 2)
        min_margin = {d: (1 if d in directions else 0) for d in _DIRECTIONS}
        if collinear:
            eligible = [d for d in directions if (height - rh if d in ('up', 'down') else width - rw) >= 3]
            if not eligible:
                continue
            collinear_side = str(rng.choice(eligible))
            min_margin[collinear_side] = 3
        if height - rh < min_margin['up'] + min_margin['down'] or width - rw < min_margin['left'] + min_margin['right']:
            continue
        r0 = int(rng.integers(min_margin['up'], height - rh - min_margin['down'] + 1))
        c0 = int(rng.integers(min_margin['left'], width - rw - min_margin['right'] + 1))
        r1, c1 = r0 + rh - 1, c0 + rw - 1
        box = (r0, r1, c0, c1)
        margins = {'up': r0, 'down': height - 1 - r1, 'left': c0, 'right': width - 1 - c1}
        if collinear and margins[collinear_side] < 3:
            continue
        points = []
        if collinear:
            lanes = range(c0, c1 + 1) if collinear_side in ('up', 'down') else range(r0, r1 + 1)
            lane = int(rng.choice(list(lanes)))
            distances = rng.choice(np.arange(1, margins[collinear_side] + 1), size=2, replace=False)
            if abs(int(distances[0]) - int(distances[1])) == 1:
                distances = (1, 3)
            points.extend(_point(collinear_side, lane, int(distance), box) for distance in distances)
        # Guarantee every requested direction has a marker.
        for direction in directions:
            if any(_path(point, box)[0] == direction for point in points):
                continue
            lanes = range(c0, c1 + 1) if direction in ('up', 'down') else range(r0, r1 + 1)
            candidates = [_point(direction, lane, distance, box)
                          for lane in lanes for distance in range(1, margins[direction] + 1)]
            rng.shuffle(candidates)
            chosen = next((p for p in candidates if _can_add_active(p, points, box, allow_collinear)), None)
            if chosen is None:
                break
            points.append(chosen)
        else:
            while len(points) < active_target:
                direction = str(rng.choice(directions))
                lanes = range(c0, c1 + 1) if direction in ('up', 'down') else range(r0, r1 + 1)
                candidates = [_point(direction, lane, distance, box)
                              for lane in lanes for distance in range(1, margins[direction] + 1)]
                rng.shuffle(candidates)
                chosen = next((p for p in candidates if _can_add_active(p, points, box, allow_collinear)), None)
                if chosen is None:
                    # Try all occupied sides before rejecting this geometry.
                    all_candidates = [_point(d, lane, dist, box)
                                      for d in directions
                                      for lane in (range(c0, c1 + 1) if d in ('up', 'down') else range(r0, r1 + 1))
                                      for dist in range(1, margins[d] + 1)]
                    rng.shuffle(all_candidates)
                    chosen = next((p for p in all_candidates if _can_add_active(p, points, box, allow_collinear)), None)
                if chosen is None:
                    break
                points.append(chosen)
            if len(points) != active_target:
                continue
            if not any(_path(point, box)[1] for point in points):
                continue
            distractor_candidates = [(r, c) for r in range(height) for c in range(width)
                                     if not r0 <= r <= r1 and not c0 <= c <= c1]
            rng.shuffle(distractor_candidates)
            distractors = []
            # Direct zero-count regression: no placement loop runs in this case.
            while len(distractors) < inactive_target:
                candidate = next((p for p in distractor_candidates
                                  if _isolated(points + distractors + [p])), None)
                if candidate is None:
                    break
                distractors.append(candidate)
                distractor_candidates.remove(candidate)
            if len(distractors) != inactive_target:
                continue
            groups = {}
            for point in points:
                direction, _ = _path(point, box)
                lane = point[1] if direction in ('up', 'down') else point[0]
                groups.setdefault((direction, lane), []).append(point)
            if multi and len(groups) < 2:
                continue
            other_colors = [c for c in range(10) if c not in (background, rectangle_color, main_marker)]
            second_marker = int(rng.choice(other_colors)) if multi else main_marker
            group_colors = {}
            for i, group in enumerate(groups):
                if multi:
                    group_colors[group] = main_marker if i == 0 else (second_marker if i == 1 else int(rng.choice((main_marker, second_marker))))
                else:
                    group_colors[group] = main_marker
            colored_points = []
            for group, members in groups.items():
                colored_points.extend((p, group_colors[group]) for p in members)
            for point in distractors:
                colored_points.append((point, int(rng.choice((main_marker, second_marker)))))
            input_grid = np.full((height, width), background, dtype=np.int8)
            input_grid[r0:r1 + 1, c0:c1 + 1] = rectangle_color
            for (r, c), color in colored_points:
                input_grid[r, c] = color
            output_grid = input_grid.copy()
            for point, color in colored_points:
                _, path = _path(point, box)
                for r, c in path:
                    if input_grid[r, c] not in (background, color):
                        break
                    output_grid[r, c] = color
                else:
                    continue
                break
            else:
                try:
                    _inspect_pair(input_grid)
                except AssertionError:
                    continue
                return input_grid, output_grid
    raise ValueError('could not construct a valid pair')


def _inspect_pair(grid):
    values = [int(v) for v in np.unique(grid)]
    rectangle_candidates = []
    for color in values:
        cells = np.argwhere(grid == color)
        r0, c0 = cells.min(axis=0)
        r1, c1 = cells.max(axis=0)
        rh, rw = int(r1 - r0 + 1), int(c1 - c0 + 1)
        if rh >= 2 and rw >= 2 and len(cells) == rh * rw and np.all(grid[r0:r1 + 1, c0:c1 + 1] == color):
            rectangle_candidates.append((color, (int(r0), int(r1), int(c0), int(c1))))
    assert len(rectangle_candidates) == 1, 'one geometric rectangle required'
    rectangle_color, box = rectangle_candidates[0]
    others = [v for v in values if v != rectangle_color]
    assert len(others) >= 2
    possible_backgrounds = []
    for candidate in others:
        candidate_markers = [(int(r), int(c)) for r, c in
                             np.argwhere((grid != candidate) & (grid != rectangle_color))]
        if candidate_markers and _isolated(candidate_markers):
            possible_backgrounds.append(candidate)
    assert len(possible_backgrounds) == 1, 'background must be geometrically identifiable'
    background = possible_backgrounds[0]
    marker_colors = set(others) - {background}
    markers = [(int(r), int(c), int(grid[r, c])) for r, c in np.argwhere(np.isin(grid, list(marker_colors)))]
    assert markers and _isolated([(r, c) for r, c, _ in markers])
    expected = grid.copy()
    directions = set()
    active_marker_colors = set()
    active = inactive = adjacent = 0
    groups = {}
    for r, c, color in markers:
        assert not (box[0] <= r <= box[1] and box[2] <= c <= box[3])
        direction, path = _path((r, c), box)
        if direction == 'distractor':
            inactive += 1
            continue
        directions.add(direction)
        active_marker_colors.add(color)
        active += 1
        adjacent += not path
        lane = c if direction in ('up', 'down') else r
        groups.setdefault((direction, lane), set()).add(color)
        for pr, pc in path:
            assert int(grid[pr, pc]) in (background, color), 'conflicting path'
            expected[pr, pc] = color
    assert all(len(colors) == 1 for colors in groups.values())
    assert active >= 1 and np.any(expected != grid)
    r0, r1, c0, c1 = box
    return {'expected': expected, 'roles': (background, rectangle_color),
            'palette': frozenset(values), 'rectangle': box,
            'rectangle_shape': (r1 - r0 + 1, c1 - c0 + 1),
            'margins': (r0, grid.shape[0] - 1 - r1, c0, grid.shape[1] - 1 - c1),
            'markers': tuple(markers), 'active_count': active,
            'distractor_count': inactive, 'adjacent_count': adjacent,
            'directions': frozenset(directions), 'marker_colors': frozenset(marker_colors),
            'active_marker_colors': frozenset(active_marker_colors),
            'collinear_count': sum(len([m for m in markers if _path((m[0], m[1]), box)[0] == d and
                                    (m[1] if d in ('up', 'down') else m[0]) == lane]) - 1
                                   for d, lane in groups)}


def _check_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert all(1 <= n <= 30 for n in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert bool(np.all((0 <= grid) & (grid <= 9)))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen_inputs, seen_pairs = set(), set()
    info = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = pair['input'], pair['output']
            _check_grid(inp)
            _check_grid(out)
            assert inp.shape == out.shape
            key = (_grid_key(inp), _grid_key(out))
            assert key[0] not in seen_inputs and key not in seen_pairs
            seen_inputs.add(key[0])
            seen_pairs.add(key)
            assert (_encoded_grid(inp), _encoded_grid(out)) not in _SOURCE_PAIRS
            inspected = _inspect_pair(inp)
            assert np.array_equal(out, inspected['expected'])
            assert frozenset(int(v) for v in np.unique(out)) == inspected['palette']
            info[split].append(inspected)
    assert set().union(*(x['directions'] for x in info['train'])) == set(_DIRECTIONS)
    assert any(x['distractor_count'] for x in info['train'])
    if any(len(x['marker_colors']) > 1 for x in info['test']):
        assert any(len(x['active_marker_colors']) > 1 for x in info['train'])
    if any(x['collinear_count'] for x in info['test']):
        assert any(x['collinear_count'] for x in info['train'])


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if not _integer(seed):
        raise ValueError('seed must be an integer')
    try:
        rng = np.random.default_rng(int(seed))
    except ValueError as exc:
        raise ValueError('invalid seed') from exc
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    stable_palette = bool(rng.random() < .23)
    fixed_palette = tuple(int(v) for v in rng.choice(10, size=3, replace=False)) if stable_palette else None
    enable_multi = not stable_palette and bool(rng.random() < .65)
    enable_collinear = bool(rng.random() < .55)
    direction_plan = [set() for _ in range(num_train)]
    simple_index = int(rng.integers(num_train)) if rng.random() < .48 else -1
    if simple_index >= 0:
        single = str(rng.choice(_DIRECTIONS))
        direction_plan[simple_index].add(single)
        others = [i for i in range(num_train) if i != simple_index]
        for direction in _DIRECTIONS:
            direction_plan[int(rng.choice(others))].add(direction)
    else:
        for direction in _DIRECTIONS:
            direction_plan[int(rng.integers(num_train))].add(direction)
    coverage_index = int(rng.choice([i for i in range(num_train) if i != simple_index]))
    puzzle = {'train': [], 'test': []}
    used = set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            for _ in range(100):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                roles = _pick_roles(pair_rng, fixed_palette)
                is_train = split == 'train'
                inp, out = _generate_pair(
                    pair_rng, 'source_train' if is_train else 'target_test', roles,
                    mandatory=tuple(d for d in _DIRECTIONS if d in direction_plan[index]) if is_train else (),
                    force_distractor=is_train and index == coverage_index,
                    force_multi=is_train and index == coverage_index and enable_multi,
                    force_collinear=is_train and index == coverage_index and enable_collinear,
                    simple=is_train and index == simple_index,
                    allow_multi=enable_multi, allow_collinear=enable_collinear,
                )
                key = _grid_key(inp)
                if key not in used and (_encoded_grid(inp), _encoded_grid(out)) not in _SOURCE_PAIRS:
                    used.add(key)
                    puzzle[split].append({'input': inp, 'output': out})
                    break
            else:
                raise ValueError('could not construct a unique pair')
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first, second):
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a['input'], b['input']) and np.array_equal(a['output'], b['output'])
        for a, b in zip(first[s], second[s])) for s in ('train', 'test'))


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    infos = [(split, _inspect_pair(pair['input']), pair['input'].shape)
             for puzzle in puzzles for split in ('train', 'test') for pair in puzzle[split]]
    assert any(split == 'train' and info['distractor_count'] == 0 for split, info, _ in infos)
    assert any(split == 'train' and info['active_count'] == 1 and info['distractor_count'] == 0 for split, info, _ in infos)
    assert any(split == 'train' and len(info['directions']) == 4 for split, info, _ in infos)
    assert any(split == 'test' and len(info['directions']) == 1 for split, info, _ in infos)
    assert any(info['collinear_count'] for _, info, _ in infos)
    assert any(len(info['marker_colors']) > 1 for _, info, _ in infos)
    assert any(0 not in info['palette'] for _, info, _ in infos)
    assert any(0 in info['margins'] for _, info, _ in infos)
    assert any(1 in info['margins'] for _, info, _ in infos)
    assert any(info['rectangle_shape'] == (2, 2) for _, info, _ in infos)
    assert any(max(shape) == 30 for _, _, shape in infos)
    assert len({shape for _, _, shape in infos}) >= 70
    assert len({info['rectangle_shape'] for _, info, _ in infos}) >= 50
    assert len({info['palette'] for _, info, _ in infos}) >= 30
    assert any(info['adjacent_count'] for _, info, _ in infos)
    for counts in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(1001, *counts))
    a = generate_puzzle(1001, 6, 3)
    assert _puzzles_equal(a, generate_puzzle(1001, 6, 3))
    assert not _puzzles_equal(a, generate_puzzle(1002, 6, 3))
    for arguments in ((0, None, None), (1, 1, 1), (1, 2, 2), (1, 10, 5),
                      (1, True, None), (1, None, 0), (-1, None, None)):
        if arguments == (0, None, None):
            continue
        try:
            generate_puzzle(*arguments)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid arguments accepted')
    # Directly exercise the formerly broken zero-distractor branch.
    rng = np.random.default_rng(937)
    pair = _generate_pair(rng, 'source_train', (1, 4, 2), simple=True,
                          allow_multi=False, allow_collinear=False)
    assert _inspect_pair(pair[0])['distractor_count'] == 0
