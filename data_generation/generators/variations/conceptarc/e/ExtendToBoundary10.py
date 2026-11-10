import numpy as np


# Exact source inputs are excluded, while their dimensions and colors remain legal.
_SOURCE_INPUTS = (
    '0000000000/0000000000/0044444440/0040000040/0040000040/7000000040/0040000040/0040000040/0044444440/0000000000',
    '00000000000000/00444444444440/00400000000040/00400000000040/00400000000040/00400000000040/00400000000040/00400000000040/00404444404440/00000000000000/00070000000000/00000000070000',
    '000000000000/000000000000/000007000000/000000000000/000000000000/004440444400/004000000400/004000000400/004000000400/004444444400/000000000000/000000000000',
    '00000000000/00000000000/00004444400/00004000400/00004000400/70000000400/00004000400/00004000400/00004000007/00004444400',
    '00000000000/00007000000/04440444400/04040400400/04040400400/04000000400/04000000400/04444444400/00000000000/00000000000/00000000000',
)
_SIDES = ('L', 'R', 'T', 'B')
_VECTORS = {'L': (0, 1), 'R': (0, -1), 'T': (1, 0), 'B': (-1, 0)}


def _choose_counts(rng, num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                  not isinstance(value, (int, np.integer)) or
                                  not low <= value <= high):
            raise ValueError('illegal pair count')
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _sample_plan(rng, test):
    n = int(rng.choice((1, 2, 3), p=(.35, .49, .16) if test else (.48, .42, .10)))
    if n == 1:
        sides = [str(rng.choice(_SIDES))]
    else:
        arrangement = int(rng.integers(4))
        if arrangement == 0:  # parallel, on any side
            side = str(rng.choice(_SIDES))
            sides = [side] * n
        elif arrangement == 1:  # opposing
            sides = list(('L', 'R') if bool(rng.integers(2)) else ('T', 'B'))
        elif arrangement == 2:  # orthogonal
            sides = [str(rng.choice(('L', 'R'))), str(rng.choice(('T', 'B')))]
        else:
            sides = [str(rng.choice(_SIDES)) for _ in range(2)]
        while len(sides) < n:
            sides.append(str(rng.choice(_SIDES)))
    return {
        'sides': tuple(sides),
        'internal': bool(rng.random() < (.48 if test else .29)),
        'inactive': bool(rng.random() < (.32 if test else .22)),
        'detail': int(rng.integers(4)),
    }


def _dimension(rng, minimum):
    band = int(rng.choice((0, 1, 2), p=(.29, .48, .23)))
    lo, hi = ((2, 5), (5, 11), (11, 19))[band]
    return max(minimum, int(rng.integers(lo, hi)))


def _border_cells(top, left, bottom, right):
    for c in range(left + 1, right):
        yield (top, c)
        yield (bottom, c)
    for r in range(top + 1, bottom):
        yield (r, left)
        yield (r, right)


def _candidate_source(rng, side, bounds, margins):
    top, left, bottom, right = bounds
    tm, lm, bm, rm = margins
    dr, dc = _VECTORS[side]
    if side == 'L':
        row = int(rng.integers(top + 1, bottom))
        distance = int(rng.integers(1, lm + 1))
        marker = (row, left - distance)
        entry = (row, left)
        interior = right - left - 1
    elif side == 'R':
        row = int(rng.integers(top + 1, bottom))
        distance = int(rng.integers(1, rm + 1))
        marker = (row, right + distance)
        entry = (row, right)
        interior = right - left - 1
    elif side == 'T':
        col = int(rng.integers(left + 1, right))
        distance = int(rng.integers(1, tm + 1))
        marker = (top - distance, col)
        entry = (top, col)
        interior = bottom - top - 1
    else:
        col = int(rng.integers(left + 1, right))
        distance = int(rng.integers(1, bm + 1))
        marker = (bottom + distance, col)
        entry = (bottom, col)
        interior = bottom - top - 1
    return marker, entry, (dr, dc), distance, interior


def _plan_sources(rng, sides, bounds, margins, require_internal):
    paths, stops, entries, markers, sources = set(), set(), set(), set(), []
    internal_used = False
    for index, side in enumerate(sides):
        for _ in range(90):
            marker, entry, (dr, dc), distance, inner = _candidate_source(
                rng, side, bounds, margins)
            if entry in entries or entry in stops or marker in markers:
                continue
            possible_internal = inner >= 4
            must_internal = require_internal and not internal_used and index == len(sides) - 1
            use_internal = (possible_internal and require_internal and
                            (must_internal or rng.random() < .39))
            if must_internal and not possible_internal:
                continue
            if use_internal:
                stop_index = distance + int(rng.integers(2, inner))
            else:
                stop_index = distance + inner
            cells = [(marker[0] + (i + 1) * dr, marker[1] + (i + 1) * dc)
                     for i in range(stop_index)]
            stop = (marker[0] + (stop_index + 1) * dr,
                    marker[1] + (stop_index + 1) * dc)
            ray = set(cells)
            if (ray & (paths | stops | markers) or
                stop in (paths | entries | markers | stops) or
                marker in (paths | stops) or
                entry in paths):
                continue
            paths.update(ray)
            stops.add(stop)
            entries.add(entry)
            markers.add(marker)
            sources.append((side, marker, entry, stop, cells, use_internal))
            internal_used |= use_internal
            break
        else:
            return None
    if require_internal and not internal_used:
        return None
    return sources, paths, stops, entries, markers


def _make_pair(rng, plan, background, barrier, line):
    requested = plan['sides']
    for count in range(len(requested), 0, -1):
        for _ in range(20):
            min_dim = 5 if plan['internal'] else 2
            inner_h = _dimension(rng, min_dim)
            inner_w = _dimension(rng, min_dim)
            tm, lm, bm, rm = (int(rng.integers(6)) for _ in range(4))
            for side in requested[:count]:
                if side == 'T': tm = max(1, tm)
                if side == 'L': lm = max(1, lm)
                if side == 'B': bm = max(1, bm)
                if side == 'R': rm = max(1, rm)
            top, left = tm, lm
            bottom, right = top + inner_h + 1, left + inner_w + 1
            h, w = bottom + bm + 1, right + rm + 1
            bounds = (top, left, bottom, right)
            planned = _plan_sources(rng, requested[:count], bounds,
                                    (tm, lm, bm, rm), plan['internal'])
            if planned is None:
                continue
            sources, paths, stops, entries, markers = planned
            inp = np.zeros((h, w), dtype=np.int8)
            inp[top, left:right + 1] = 1
            inp[bottom, left:right + 1] = 1
            inp[top:bottom + 1, left] = 1
            inp[top:bottom + 1, right] = 1
            for entry in entries:
                inp[entry] = 0
            for side, marker, entry, stop, cells, internal in sources:
                inp[marker] = 2
                inp[stop] = 1
            # Extend interior stopping cells orthogonally into real wall segments.
            for side, marker, entry, stop, cells, internal in sources:
                if not internal:
                    continue
                candidates = ((-1, 0), (1, 0)) if side in ('L', 'R') else ((0, -1), (0, 1))
                order = rng.permutation(2)
                for which in order:
                    dr, dc = candidates[int(which)]
                    extension = []
                    r, c = stop
                    while top < r + dr < bottom and left < c + dc < right:
                        r += dr
                        c += dc
                        if (r, c) in paths or (r, c) in entries:
                            break
                        extension.append((r, c))
                    if extension:
                        length = int(rng.integers(1, len(extension) + 1))
                        for cell in extension[:length]:
                            inp[cell] = 1
                        break
            # Rails and inward stubs vary corridor widths and compartment outlines.
            for side, marker, entry, stop, cells, internal in sources:
                if not plan['simple'] and rng.random() < .44:
                    offsets = ((-1, 0), (1, 0)) if side in ('T', 'B') else ((0, -1), (0, 1))
                    dr, dc = offsets[int(rng.integers(2))]
                    interior_cells = [p for p in cells if top < p[0] < bottom and left < p[1] < right]
                    length = int(rng.integers(1, len(interior_cells) + 1))
                    for r, c in interior_cells[:length]:
                        cell = (r + dr, c + dc)
                        if top < cell[0] < bottom and left < cell[1] < right and cell not in paths:
                            inp[cell] = 1
            for _ in range(plan['detail']):
                side = str(rng.choice(_SIDES))
                if side in ('T', 'B'):
                    c = int(rng.integers(left + 1, right))
                    start = (top, c) if side == 'T' else (bottom, c)
                else:
                    r = int(rng.integers(top + 1, bottom))
                    start = (r, left) if side == 'L' else (r, right)
                if start in entries or inp[start] != 1:
                    continue
                dr, dc = _VECTORS[side]
                limit = int(rng.integers(1, 6))
                for step in range(1, limit + 1):
                    cell = (start[0] + step * dr, start[1] + step * dc)
                    if not (top < cell[0] < bottom and left < cell[1] < right):
                        break
                    if cell in paths:
                        break
                    inp[cell] = 1
            if plan['inactive']:
                unused = [cell for cell in _border_cells(*bounds)
                          if inp[cell] == 1 and cell not in stops and
                          all(not ((m[0] == cell[0] and s in ('L', 'R')) or
                                   (m[1] == cell[1] and s in ('T', 'B')))
                                  for s, m, _, _, _, _ in sources)]
                if not unused:
                    continue
                inp[unused[int(rng.integers(len(unused)))]] = 0
            out = inp.copy()
            for cell in paths:
                out[cell] = 2
            colors = np.array((background, barrier, line), dtype=np.int8)
            return {'input': colors[inp], 'output': colors[out]}
    raise RuntimeError('unable to construct pair')


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _choose_counts(rng, num_train, num_test)
    line = int(rng.integers(10))
    remaining = [c for c in range(10) if c != line]
    first = rng.choice(remaining, size=2, replace=False)
    base = (int(first[0]), int(first[1]))
    local_colors = bool(rng.random() < .28)
    plans = {'train': [_sample_plan(rng, False) for _ in range(num_train)],
             'test': [_sample_plan(rng, True) for _ in range(num_test)]}
    simple_puzzle = bool(rng.random() < .20)
    if simple_puzzle:
        for split in ('train', 'test'):
            for plan in plans[split]:
                plan['sides'] = plan['sides'][:1]
                plan['internal'] = False
                plan['inactive'] = False
                plan['detail'] = 0
                plan['simple'] = True
    else:
        for split in ('train', 'test'):
            for plan in plans[split]:
                plan['simple'] = False
    for feature in ('internal', 'inactive'):
        if any(p[feature] for p in plans['test']) and not any(p[feature] for p in plans['train']):
            plans['train'][0][feature] = True
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split in ('train', 'test'):
        for index, plan in enumerate(plans[split]):
            if local_colors and split == 'train' and index == 1:
                palette = base[::-1]
            elif local_colors and not (split == 'train' and index == 0):
                chosen = rng.choice(remaining, size=2, replace=False)
                palette = (int(chosen[0]), int(chosen[1]))
            else:
                palette = base
            for _ in range(12):
                pair = _make_pair(rng, plan, *palette, line)
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen and _source_string(pair['input']) not in _SOURCE_INPUTS:
                    break
            else:
                raise RuntimeError('unable to sample unique non-source pair')
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _source_string(grid):
    return '/'.join(''.join(str(int(x)) for x in row) for row in grid)


def _interpret(inp, line, background, barrier):
    walls = np.argwhere(inp == barrier)
    marks = np.argwhere(inp == line)
    if len(walls) < 8 or not 1 <= len(marks) <= 3:
        return None
    top, left = map(int, walls.min(axis=0))
    bottom, right = map(int, walls.max(axis=0))
    if bottom - top < 3 or right - left < 3:
        return None
    border = list(_border_cells(top, left, bottom, right))
    if any(int(inp[p]) not in (background, barrier) for p in border):
        return None
    if any(int(inp[p]) != barrier for p in ((top, left), (top, right), (bottom, left), (bottom, right))):
        return None
    holes = {p for p in border if int(inp[p]) == background}
    if not len(marks) <= len(holes) <= len(marks) + 1:
        return None
    expected = inp.copy()
    entries, paths, internal = set(), set(), False
    for row, col in marks:
        r, c = int(row), int(col)
        options = []
        if top < r < bottom and c < left and (r, left) in holes:
            options.append(((0, 1), (r, left)))
        if top < r < bottom and c > right and (r, right) in holes:
            options.append(((0, -1), (r, right)))
        if left < c < right and r < top and (top, c) in holes:
            options.append(((1, 0), (top, c)))
        if left < c < right and r > bottom and (bottom, c) in holes:
            options.append(((-1, 0), (bottom, c)))
        if len(options) != 1:
            return None
        (dr, dc), entry = options[0]
        if entry in entries:
            return None
        entries.add(entry)
        r += dr
        c += dc
        count = 0
        reached = False
        while 0 <= r < inp.shape[0] and 0 <= c < inp.shape[1] and int(inp[r, c]) != barrier:
            if int(inp[r, c]) != background or (r, c) in paths:
                return None
            expected[r, c] = line
            paths.add((r, c))
            reached |= (r, c) == entry
            count += 1
            r += dr
            c += dc
        if not (0 <= r < inp.shape[0] and 0 <= c < inp.shape[1] and
                reached and count >= 2):
            return None
        internal |= top < r < bottom and left < c < right
    if entries != holes and len(holes - entries) != 1:
        return None
    # A second role assignment must not yield a different valid transformation.
    return expected, {'internal': internal, 'inactive': bool(holes - entries),
                      'sources': len(marks), 'frame': (bottom - top + 1, right - left + 1),
                      'edge': top == 0 or left == 0 or bottom == inp.shape[0] - 1 or right == inp.shape[1] - 1,
                      'inner_walls': int(np.count_nonzero(inp[top + 1:bottom, left + 1:right] == barrier))}


def _predict(inp, line):
    other = [int(c) for c in np.unique(inp) if int(c) != line]
    if len(other) != 2:
        raise AssertionError('three semantic colors required')
    choices = []
    for background, barrier in (other, other[::-1]):
        result = _interpret(inp, line, background, barrier)
        if result is not None:
            choices.append((result, background, barrier))
    if len(choices) != 1:
        raise AssertionError('ambiguous or invalid input roles')
    (expected, info), background, barrier = choices[0]
    return expected, info, background, barrier


def validate_generated_puzzle(puzzle: dict) -> None:
    assert type(puzzle) is dict and set(puzzle) == {'train', 'test'}
    assert type(puzzle['train']) is list and type(puzzle['test']) is list
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    line_color = None
    seen = set()
    features = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert type(pair) is dict and set(pair) == {'input', 'output'}
            inp, out = pair['input'], pair['output']
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            changed = inp != out
            assert np.any(changed) and len(np.unique(inp[changed])) == 1
            assert len(np.unique(out[changed])) == 1
            line = int(np.unique(out[changed])[0])
            assert line != int(np.unique(inp[changed])[0])
            if line_color is None:
                line_color = line
            assert line == line_color
            predicted, info, background, barrier = _predict(inp, line)
            assert np.all(inp[changed] == background)
            assert np.array_equal(out, predicted)
            key = (inp.shape, inp.tobytes())
            assert key not in seen
            seen.add(key)
            assert _source_string(inp) not in _SOURCE_INPUTS
            features[split].append(info)
    for feature in ('internal', 'inactive'):
        if any(info[feature] for info in features['test']):
            assert any(info[feature] for info in features['train'])


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ('input', 'output'))
        for s in ('train', 'test'))


def self_test() -> None:
    train_counts, test_counts, palettes, shapes, structures = set(), set(), set(), set(), set()
    directions, source_counts, frame_sizes = set(), set(), set()
    internal_count = inactive_count = edge_count = small_count = large_count = 0
    multi_direction = orthogonal = parallel_right = parallel_down = 0
    simple_puzzles = varied_sources = varied_palettes = nonzero_background = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        demonstration = puzzle['train'][0]
        train_line = int(np.unique(demonstration['output'][
            demonstration['input'] != demonstration['output']])[0])
        puzzle_infos, puzzle_sources, puzzle_palettes = [], set(), set()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                inferred, info, bg, wall = _predict(inp.copy(), train_line)
                assert np.array_equal(inferred, out)
                palettes.add((bg, wall, train_line))
                puzzle_palettes.add((bg, wall))
                puzzle_sources.add(info['sources'])
                puzzle_infos.append(info)
                nonzero_background += bg != 0
                shapes.add(inp.shape)
                frame_sizes.add(info['frame'])
                source_counts.add(info['sources'])
                internal_count += info['internal']
                inactive_count += info['inactive']
                edge_count += info['edge']
                small_count += inp.shape[0] <= 12
                large_count += inp.shape[0] >= 18 or inp.shape[1] >= 18
                normalized = np.where(inp == bg, 0, np.where(inp == wall, 1, 2)).astype(np.int8)
                structures.add((inp.shape, normalized.tobytes()))
                walls = np.argwhere(inp == wall)
                top, left = walls.min(axis=0)
                bottom, right = walls.max(axis=0)
                ds = []
                for r, c in np.argwhere(inp == train_line):
                    if c < left: ds.append('R')
                    elif c > right: ds.append('L')
                    elif r < top: ds.append('D')
                    else: ds.append('U')
                directions.update(ds)
                multi_direction += len(set(ds)) >= 2
                orthogonal += (any(x in ds for x in ('R', 'L')) and
                               any(x in ds for x in ('D', 'U')))
                parallel_right += ds.count('R') >= 2
                parallel_down += ds.count('D') >= 2
        simple_puzzles += all(not x['internal'] and not x['inactive'] and
                              x['inner_walls'] == 0 for x in puzzle_infos)
        varied_sources += len(puzzle_sources) >= 2
        varied_palettes += len(puzzle_palettes) >= 2
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(palettes) >= 18 and len(shapes) >= 35 and len(structures) >= 150
    assert source_counts == {1, 2, 3} and directions == {'R', 'L', 'D', 'U'}
    assert len(frame_sizes) >= 30
    assert min(internal_count, inactive_count, edge_count, small_count, large_count) >= 5
    assert multi_direction >= 5 and orthogonal >= 5
    assert parallel_right >= 2 and parallel_down >= 2
    assert simple_puzzles >= 3 and varied_sources >= 20 and varied_palettes >= 5
    assert nonzero_background >= 100
    for seed, train, test in ((500, 2, 1), (501, 10, 1), (502, 10, 4),
                              (503, None, 1), (504, None, 4),
                              (505, 2, None), (506, 10, None), (507, 7, 3)):
        first = generate_puzzle(seed, train, test)
        second = generate_puzzle(seed, train, test)
        assert _same_puzzle(first, second)
        if train is not None: assert len(first['train']) == train
        if test is not None: assert len(first['test']) == test
    assert not _same_puzzle(generate_puzzle(700, 2, 1), generate_puzzle(701, 2, 1))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(900, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
