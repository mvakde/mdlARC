"""Count eight-connected monochrome objects and extract a unary tally panel."""
import numpy as np

_AUTO_COUNT_PAIRS = ((2, 1), (3, 1), (4, 1), (4, 2), (5, 2), (6, 2),
                     (6, 3), (7, 3), (8, 3), (8, 4), (9, 4), (10, 4))
_DIAMOND = ((0, 1), (1, 0), (1, 2), (2, 1))
_SQUARE = ((0, 0), (0, 1), (1, 0), (1, 1))
_X = ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2))
_CROSS = ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))
_GLYPHS = (_DIAMOND, _SQUARE, _X, _CROSS,
           ((0, 0), (0, 1), (1, 0), (1, 2), (2, 1)),
           ((0, 0), (0, 1), (0, 2), (1, 1), (2, 0), (2, 1), (2, 2)))
_DIRECTIONS = ((0, 1), (-1, 0), (0, -1), (1, 0))


def _grid_key(grid):
    return tuple(grid.shape), np.ascontiguousarray(grid, dtype=np.int8).tobytes()


def _pair_key(pair):
    return _grid_key(pair['input']), _grid_key(pair['output'])


def _puzzle_key(puzzle):
    return tuple(tuple(_pair_key(p) for p in puzzle[s]) for s in ('train', 'test'))


def _normalize_cells(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _components(mask, diagonal=True):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r)+abs(c) == 1)]
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, component = [start], []
        while stack:
            r, c = stack.pop()
            component.append((int(r), int(c)))
            for dr, dc in steps:
                point = (r+dr, c+dc)
                if point in remaining:
                    remaining.remove(point)
                    stack.append(point)
        result.append(component)
    return result


def _find_panel(grid):
    """Identify field bounds, blank borders, and aligned singleton keys."""
    candidates = []
    for field in np.unique(grid):
        rr, cc = np.where(grid == field)
        r0, r1, c0, c1 = int(rr.min()), int(rr.max())+1, int(cc.min()), int(cc.max())+1
        panel = grid[r0:r1, c0:c1]
        if min(panel.shape) < 3 or not all(np.all(edge == field) for edge in
                                          (panel[0], panel[-1], panel[:, 0], panel[:, -1])):
            continue
        positions = np.argwhere(panel != field)
        if not 1 <= len(positions) <= 8:
            continue
        colors = tuple(int(panel[r, c]) for r, c in positions)
        if len(set(colors)) != len(colors):
            continue
        if len(positions) > 1:
            if len(set(positions[:, 1])) == 1 and np.all(np.diff(positions[:, 0]) >= 2):
                inset = int(positions[0, 1])
                if 2*inset == panel.shape[1]-1:
                    continue
                direction = 0 if 2*inset < panel.shape[1]-1 else 2
            elif len(set(positions[:, 0])) == 1 and np.all(np.diff(positions[:, 1]) >= 2):
                inset = int(positions[0, 0])
                if 2*inset == panel.shape[0]-1:
                    continue
                direction = 3 if 2*inset < panel.shape[0]-1 else 1
            else:
                continue
        else:
            # Lone keys occur only with a fixed, demonstrated orientation.
            # A unique nearest panel side supplies an additional visible cue.
            r, c = positions[0]
            distances = (c, panel.shape[0]-1-r, panel.shape[1]-1-c, r)
            if distances.count(min(distances)) != 1:
                continue
            direction = int(np.argmin(distances))
        outside = np.ones(grid.shape, bool)
        outside[r0:r1, c0:c1] = False
        if not outside.any():
            continue
        bg = int(np.argmax(np.bincount(grid[outside], minlength=10)))
        if bg == field or bg in colors or np.mean(grid[outside] == bg) <= .55:
            continue
        ring = np.zeros(grid.shape, bool)
        ring[max(0, r0-1):r1+1, max(0, c0-1):c1+1] = True
        ring[r0:r1, c0:c1] = False
        if not np.all(grid[ring] == bg):
            continue
        candidates.append(dict(field=int(field), background=bg, bounds=(r0, r1, c0, c1),
                               panel=panel.copy(), positions=positions, keys=colors,
                               direction=direction, outside=outside))
    assert len(candidates) == 1, 'Panel must be visibly unique'
    return candidates[0]


def _inspect_input(grid):
    meta = _find_panel(grid)
    outside, bg = meta['outside'], meta['background']
    objects, serial = {}, 0
    owner = np.full(grid.shape, -1, int)
    for color in sorted(set(map(int, grid[outside])) - {bg}):
        components = _components((grid == color) & outside)
        objects[color] = tuple(_normalize_cells(comp) for comp in components)
        for comp in components:
            for r, c in comp:
                owner[r, c] = serial
            serial += 1
    # Separate objects, including different colors, cannot meet at corners.
    for r, c in np.argwhere(owner >= 0):
        neighbors = owner[max(0, r-1):r+2, max(0, c-1):c+2]
        assert np.all((neighbors < 0) | (neighbors == owner[r, c]))
    counts = {color: len(shapes) for color, shapes in objects.items()}
    assert set(meta['keys']) <= set(counts)
    result = np.full(meta['panel'].shape, meta['field'], np.int8)
    dr, dc = _DIRECTIONS[meta['direction']]
    for (r, c), color in zip(meta['positions'], meta['keys']):
        for i in range(counts[color]):
            rr, cc = int(r)+2*i*dr, int(c)+2*i*dc
            assert 0 < rr < result.shape[0]-1 and 0 < cc < result.shape[1]-1, 'Tally does not fit'
            result[rr, cc] = color
    meta.update(objects=objects, counts=counts, expected=result,
                distractors=tuple(sorted(set(counts)-set(meta['keys']))))
    return meta


def solve(train_pairs, test_input):
    """Prediction receives no test label, seed, or construction metadata."""
    meta = _inspect_input(test_input)
    if len(meta['keys']) == 1:
        directions = {_find_panel(p['input'])['direction'] for p in train_pairs
                      if len(_find_panel(p['input'])['keys']) > 1}
        assert directions == {meta['direction']}, 'Lone keys need demonstrated fixed orientation'
    return meta['expected'].copy()


def _shape(rng, level):
    mode = int(rng.integers(0, 7 if level else 3))
    if mode < 2:
        cells = _GLYPHS[int(rng.integers(len(_GLYPHS)))]
    elif mode == 2:
        cells = tuple((0, i) for i in range(int(rng.integers(1, (5, 6, 8)[level]))))
    elif mode == 3:
        radius = int(rng.integers(1, 3))
        cells = tuple((r, c) for r in range(2*radius+1) for c in range(2*radius+1)
                      if abs(r-radius)+abs(c-radius) == radius)
    else:
        # Compositional growth produces many irregular connected glyphs.
        side = int(rng.integers(3, 5 if level == 1 else 6))
        target = int(rng.integers(2, min(10 if level == 1 else 15, side*side)+1))
        occupied = {(side//2, side//2)}
        while len(occupied) < target:
            frontier = sorted({(r+dr, c+dc) for r, c in occupied
                               for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                               if 0 <= r+dr < side and 0 <= c+dc < side} - occupied)
            occupied.add(frontier[int(rng.integers(len(frontier)))])
        cells = tuple(occupied)
    if rng.random() < .5:
        cells = tuple((r, -c) for r, c in cells)
    for _ in range(int(rng.integers(4))):
        cells = tuple((-c, r) for r, c in cells)
    return _normalize_cells(cells)


def _panel_geometry(rng, counts, level, compact=False):
    k = len(counts)
    inset = 1 if compact or not level else int(rng.integers(1, 4))
    top = 1 if compact or not level else int(rng.integers(1, 4))
    bottom = 1 if compact or not level else int(rng.integers(1, 5))
    if k == 1:
        top, bottom = max(top, inset+1), max(bottom, inset+1)
    gaps = [2 if compact or not level else int(rng.integers(2, 5)) for _ in range(k-1)]
    rows = np.cumsum([top] + gaps).tolist()
    trailing = 1 if compact else int(rng.integers(1, 5))
    width = max(2*inset+3, inset+2*(max(counts)-1)+trailing+1)
    return rows[-1]+bottom+1, width, rows, inset


def _panel_position(rng, h, w, ph, pw):
    mr, mc = h-ph, w-pw
    if rng.random() < .55 and min(mr, mc) >= 2:
        return int(rng.integers(1, mr)), int(rng.integers(1, mc))
    side = int(rng.integers(4))
    if side < 2:
        return (0 if side == 0 else mr), int(rng.integers(mc+1))
    return int(rng.integers(mr+1)), (0 if side == 2 else mc)


def _place_object(grid, blocked, color, shape, rng):
    sh, sw = 1+max(r for r, c in shape), 1+max(c for r, c in shape)
    nh, nw = grid.shape[0]-sh+1, grid.shape[1]-sw+1
    if min(nh, nw) <= 0:
        return False
    allowed = np.ones((nh, nw), bool)
    for r, c in shape:
        allowed &= ~blocked[r:r+nh, c:c+nw]
    candidates = np.argwhere(allowed)
    if not len(candidates):
        return False
    r0, c0 = candidates[int(rng.integers(len(candidates)))]
    for dr, dc in shape:
        r, c = r0+dr, c0+dc
        grid[r, c] = color
        blocked[max(0, r-1):r+2, max(0, c-1):c+2] = True
    return True


def _make_pair(rng, spec):
    keys, counts, level = spec['keys'], spec['counts'], spec['level']
    objects = []
    for i, (color, count) in enumerate(zip(keys+spec['distractors'], counts+spec['dcounts'])):
        prototype = _shape(rng, level)
        for j in range(count):
            shape = _shape(rng, level) if level and rng.random() < .7 else prototype
            if spec['witness'] and i == 0 and j < 3:
                shape = (_DIAMOND, _SQUARE, ((0, 0),))[j]
            if i >= len(keys) and rng.random() < .35:
                shape = objects[int(rng.integers(len(objects)))][1]
            objects.append((color, shape))
    rng.shuffle(objects)
    objects.sort(key=lambda item: -(1+max(r for r, c in item[1]))*(1+max(c for r, c in item[1])))
    geometry = _panel_geometry(rng, counts, spec['layout_level'])
    for attempt in range(9):
        fallback = attempt == 8
        ph, pw, rows, inset = _panel_geometry(rng, counts, level, True) if fallback else geometry
        if fallback:
            h, w, r0, c0 = 30, 30, 0, 0
        else:
            area = ph*pw + sum((2+max(r for r, c in s))*(2+max(c for r, c in s)) for _, s in objects)
            lower = max(9, int(np.sqrt(area*.95)))
            h = min(30, max(ph+2, int(rng.integers(lower, max(lower+1, 31)))+attempt))
            lower_w = max(9, area//h)
            w = min(30, max(pw+2, int(rng.integers(lower_w, max(31, lower_w+1)))+attempt))
            if attempt >= 5:
                h = w = 30
            r0, c0 = _panel_position(rng, h, w, ph, pw)
        grid = np.full((h, w), spec['background'], np.int8)
        grid[r0:r0+ph, c0:c0+pw] = spec['field']
        for row, color in zip(rows, keys):
            grid[r0+row, c0+inset] = color
        blocked = np.zeros(grid.shape, bool)
        blocked[max(0, r0-1):r0+ph+1, max(0, c0-1):c0+pw+1] = True
        if fallback:
            # Guaranteed 3x3 slots preserve counts and diagonal witnesses.
            # Only oversized motifs are simplified in this last resort.
            slots = [(r, c) for r in range(0, 28, 4) for c in range(0, 28, 4)
                     if not blocked[r:r+3, c:c+3].any()]
            assert len(slots) >= len(objects)
            rng.shuffle(slots)
            for (color, shape), (r, c) in zip(objects, slots):
                if max(max(p) for p in shape) > 2:
                    shape = _CROSS
                for dr, dc in shape:
                    grid[r+dr, c+dc] = color
        elif not all(_place_object(grid, blocked, color, shape, rng) for color, shape in objects):
            continue
        # Labels use construction counts. Segmentation separately checks
        # the visible finished scene, so placement errors cannot relabel it.
        output = np.full((ph, pw), spec['field'], np.int8)
        for row, color, count in zip(rows, keys, counts):
            output[row, inset:inset+2*count:2] = color
        if spec['reflect']:
            grid, output = np.flipud(grid), np.flipud(output)
        grid = np.rot90(grid, spec['direction']).copy()
        output = np.rot90(output, spec['direction']).copy()
        try:
            assert np.array_equal(_inspect_input(grid)['expected'], output)
        except AssertionError:
            if fallback:
                raise
            continue
        return {'input': grid, 'output': output}
    raise AssertionError('Unreachable bounded fallback')


def _resolve_counts(seed, num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                 not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError(name+' is outside the legal integer range')
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError('num_train must exceed num_test')
        return int(num_train), int(num_test)
    if num_train is None and num_test is None:
        return _AUTO_COUNT_PAIRS[seed % len(_AUTO_COUNT_PAIRS)]
    if num_train is None:
        low = max(2, num_test+1)
        return low+(3*seed+1) % (11-low), int(num_test)
    return int(num_train), 1+(5*seed+2) % min(4, num_train-1)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    rng = np.random.default_rng(seed)
    nt, nq = _resolve_counts(seed, num_train, num_test)
    # Sampling regimes are puzzle-level; there is no hidden color dictionary.
    regime = int(rng.integers(5))  # same range, counts, categories, source, recombinations
    orientation = int(rng.choice(4, p=[.45, .15, .15, .25]))
    fixed = int(rng.integers(4))
    simple_palette = rng.random() < .18
    palette = list(map(int, rng.permutation(10)))
    novel = palette[-1] if rng.random() < .30 and not simple_palette else None
    first_key = palette[2]
    puzzle, seen, first_k = {'train': [], 'test': []}, set(), None
    for split, n in (('train', nt), ('test', nq)):
        for index in range(n):
            train = split == 'train'
            witness, role_witness = train and index == 0, train and index == 1
            if orientation == 0:
                direction = fixed
            elif orientation == 1:
                direction = int(rng.choice((0, 2) if train else (1, 3)))
            elif orientation == 2:
                direction = int(rng.choice((1, 3) if train else (0, 2)))
            else:
                direction = (fixed+index) % 4 if train else int(rng.integers(4))
            low, high, cap = (1 if orientation == 0 else 2), 6, 7
            if regime == 1:
                high, cap = 4, 3 if train else 8
            elif regime == 2:
                low, high, cap = (2, 3, 5) if train else (4, 6, 5)
            elif regime == 3:
                low, high, cap = (2, 3, 4) if train else (4, 4, 5)
            elif regime == 4 and not train:
                high, cap = 5, 5
            if witness:
                low = max(2, low)
            k = int(rng.integers(low, high+1))
            if role_witness and k == first_k and low < high:
                k = low+(k-low+1) % (high-low+1)
            if witness:
                first_k = k
            available = [c for c in palette if not (train and c == novel)]
            if simple_palette:
                bg, field = palette[:2]
            else:
                structural = [c for c in available if c != novel and
                              (c != first_key or not (witness or role_witness))]
                bg, field = map(int, rng.choice(structural, 2, replace=False))
            pool = [c for c in available if c not in (bg, field)]
            rng.shuffle(pool)
            if witness:
                pool.remove(first_key)
                pool.insert(0, first_key)
            elif role_witness:
                pool.remove(first_key)
            elif not train and novel is not None:
                pool.remove(novel)
                pool.insert(0, novel)
            keys, rest = pool[:k], pool[k:]
            if role_witness:
                rest.insert(0, first_key)
            nd = int(rng.integers(0, min(3, len(rest))+1))
            if role_witness:
                nd = max(1, nd)
            distractors = rest[:nd]
            counts = [int(rng.integers(1, cap+1)) for _ in keys]
            if rng.random() < .10 and not witness and not (not train and regime in (1, 3)):
                counts = [1]*k
            if witness:
                counts[0] = max(3, counts[0])
            if not train and regime in (1, 3):
                counts[int(rng.integers(k))] = int(rng.integers(6, 9)) if regime == 1 else 5
            protected = int(np.argmax(counts))
            while sum(counts) > 17:
                eligible = [i for i, c in enumerate(counts) if c > (3 if witness and i == 0 else 1)
                            and i != protected]
                counts[eligible[int(rng.integers(len(eligible)))]] -= 1
            dcounts = [1]*nd
            if nd:
                budget = min(9, 25-sum(counts))
                for _ in range(int(rng.integers(nd, budget+1))-nd):
                    eligible = [i for i, c in enumerate(dcounts) if c < 8]
                    if eligible:
                        dcounts[eligible[int(rng.integers(len(eligible)))]] += 1
            level = int(rng.choice(3, p=[.35, .45, .20] if regime == 3 else [.18, .42, .40]))
            spec = dict(keys=keys, counts=counts, distractors=distractors, dcounts=dcounts,
                        background=bg, field=field, witness=witness, level=level,
                        layout_level=int(rng.choice(2, p=[.3, .7])),
                        direction=direction, reflect=bool(rng.integers(2)))
            for retry in range(8):
                pair = _make_pair(rng, spec)
                if _grid_key(pair['input']) not in seen and _pair_key(pair) not in _SOURCE_PAIRS:
                    break
            else:
                raise AssertionError('Could not construct a distinct scene')
            seen.add(_grid_key(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert all(1 <= n <= 30 for n in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((0 <= grid) & (grid <= 9))


def _orth_count(shape):
    mask = np.zeros((1+max(r for r, c in shape), 1+max(c for r, c in shape)), bool)
    for r, c in shape:
        mask[r, c] = True
    return len(_components(mask, False))


def _features(grid):
    m = _inspect_input(grid)
    objects = [s for shapes in m['objects'].values() for s in shapes]
    orth = {s: _orth_count(s) for s in set(objects)}
    diagonal = any(orth[s] > 1 for c in m['keys'] for s in m['objects'][c])
    mixed = any(len({len(s) for s in m['objects'][c]}) > 1 for c in m['keys'])
    equal_area = any(len(a) == len(b) and orth[a] != orth[b]
                     for c in m['keys'] for a in m['objects'][c] for b in m['objects'][c])
    r0, r1, c0, c1 = m['bounds']
    panel = np.rot90(m['panel'], -m['direction'])
    positions = np.argwhere(panel != m['field'])
    m.update(diagonal=diagonal, mixed_area=mixed, equal_area_connectivity=equal_area,
             areas=tuple(sorted({len(s) for s in objects})), shape_vocabulary=tuple(sorted(set(objects))),
             edge=r0 == 0 or c0 == 0 or r1 == grid.shape[0] or c1 == grid.shape[1],
             inset=int(positions[0, 1]), top=int(positions[0, 0]),
             bottom=int(panel.shape[0]-1-positions[-1, 0]),
             gaps=tuple(map(int, np.diff(positions[:, 0]))),
             trailing=int(panel.shape[1]-1-positions[0, 1]-2*(max(m['counts'][c] for c in m['keys'])-1)))
    return m


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
        assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
        nt, nq = len(puzzle['train']), len(puzzle['test'])
        assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
        seen, training, allmeta = set(), [], []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
                for grid in pair.values():
                    _validate_grid(grid)
                key = _grid_key(pair['input'])
                assert key not in seen and _pair_key(pair) not in _SOURCE_PAIRS
                seen.add(key)
                meta = _features(pair['input'])
                assert np.array_equal(pair['output'], meta['expected'])
                assert not np.array_equal(pair['input'], pair['output'])
                allmeta.append(meta)
                if split == 'train':
                    training.append(meta)
        assert any(m['diagonal'] and m['mixed_area'] and m['equal_area_connectivity'] for m in training)
        assert any(max(m['counts'][c] for c in m['keys']) >= 3 for m in training)
        assert any(set(a['keys']) & set(b['distractors']) for a in training for b in training)
        assert len({len(m['keys']) for m in training}) >= 2
        assert any(len(m['keys']) > 1 for m in training)
        if any(len(m['keys']) == 1 for m in allmeta):
            assert len({m['direction'] for m in allmeta}) == 1
        for pair in puzzle['test']:
            assert np.array_equal(solve(puzzle['train'], pair['input'].copy()), pair['output'])
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError('Malformed puzzle: '+str(exc)) from exc


def _orthogonal_prediction(grid):
    m = _find_panel(grid)
    result = np.full(m['panel'].shape, m['field'], np.int8)
    dr, dc = _DIRECTIONS[m['direction']]
    for (r, c), color in zip(m['positions'], m['keys']):
        n = len(_components((grid == color) & m['outside'], False))
        for i in range(n):
            rr, cc = r+2*i*dr, c+2*i*dc
            if not (0 <= rr < result.shape[0] and 0 <= cc < result.shape[1]):
                return np.empty((0, 0), np.int8)
            result[rr, cc] = color
    return result


def self_test() -> None:
    for inp, out in _SOURCE_GRIDS:
        assert np.array_equal(_inspect_input(inp)['expected'], out)
    assert sum(np.array_equal(_orthogonal_prediction(inp), out) for inp, out in _SOURCE_GRIDS) == 1
    for glyph, expected4 in ((_DIAMOND, 4), (_X, 5), (_SQUARE, 1)):
        mask = np.zeros((3, 3), bool)
        for r, c in glyph:
            mask[r, c] = True
        assert len(_components(mask)) == 1 and len(_components(mask, False)) == expected4
    examples = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in examples:
        validate_generated_puzzle(puzzle)
    assert _puzzle_key(generate_puzzle(12)) == _puzzle_key(examples[12])
    assert len({_puzzle_key(p) for p in examples}) == 42
    assert len({len(p['train']) for p in examples}) >= 4
    assert len({len(p['test']) for p in examples}) >= 3
    groups = [{split: [_features(p['input']) for p in q[split]] for split in ('train', 'test')}
              for q in examples]
    stats = [m for group in groups for split in ('train', 'test') for m in group[split]]
    assert len({tuple(m['panel'].shape) for m in stats}) > 60
    assert len({s for m in stats for s in m['shape_vocabulary']}) > 120
    assert {m['field'] for m in stats} == set(range(10))
    assert {m['background'] for m in stats} == set(range(10))
    assert {m['direction'] for m in stats} == set(range(4))
    assert {len(m['keys']) for m in stats} >= set(range(1, 7))
    assert {n for m in stats for n in m['counts'].values()} >= set(range(1, 9))
    assert {len(m['distractors']) for m in stats} >= set(range(4))
    assert {m['edge'] for m in stats} == {True, False}
    assert {m['inset'] for m in stats} >= {1, 2, 3}
    assert {g for m in stats for g in m['gaps']} >= {2, 3, 4}
    assert all(len({m[key] for m in stats}) >= 3 for key in ('top', 'bottom', 'trailing'))
    assert any(max(m['counts'].values()) == 1 for m in stats)
    assert any(m['distractors'] and max(m['counts'][c] for c in m['distractors']) >
               max(m['counts'][c] for c in m['keys']) for m in stats)
    assert any(1 in m['areas'] and max(m['areas']) >= 8 for m in stats)
    assert set(_GLYPHS) <= {s for m in stats for s in m['shape_vocabulary']}
    assert any(0 in m['objects'] for m in stats)
    assert any(m['distractors'] and any(set(m['objects'][c]) & set(m['objects'][k])
               for c in m['distractors'] for k in m['keys']) for m in stats)
    assert any(len(shapes) > 1 and len(set(shapes)) == 1
               for m in stats for shapes in m['objects'].values())
    assert sum(m['diagonal'] and m['mixed_area'] and m['field'] != 0 and m['direction'] != 0
               and np.rot90(m['panel'], -m['direction']).shape[0] != 2*len(m['keys'])+1
               for m in stats) >= 30
    orientations, shifts, novel_keys, three_roles = set(), set(), 0, 0
    for puzzle, group in zip(examples, groups):
        train, test = group['train'], group['test']
        td, qd = {m['direction'] for m in train}, {m['direction'] for m in test}
        orientations.add('fixed' if len(td | qd) == 1 else
                         'h_to_v' if td <= {0, 2} and qd <= {1, 3} else
                         'v_to_h' if td <= {1, 3} and qd <= {0, 2} else 'mixed')
        maxima = [max(m['counts'][c] for m in split for c in m['keys']) for split in (train, test)]
        categories = [max(len(m['keys']) for m in split) for split in (train, test)]
        shifts.add((int(np.sign(maxima[1]-maxima[0])), int(np.sign(categories[1]-categories[0]))))
        colors = {int(v) for p in puzzle['train'] for v in p['input'].flat}
        novel_keys += any(set(m['keys'])-colors for m in test)
        both = train+test
        three_roles += any(any(c in m['keys'] for m in both) and
                           any(c in m['distractors'] for m in both) and
                           any(c == m['background'] for m in both) for c in range(10))
        assert len({m['shape_vocabulary'] for m in both}) >= 2
        assert len({m['bounds'] for m in both}) >= 2
    assert orientations == {'fixed', 'h_to_v', 'v_to_h', 'mixed'}
    assert {s[0] for s in shifts} == {-1, 0, 1}
    assert {s[1] for s in shifts} == {-1, 0, 1}
    assert any(a == 1 and b <= 0 for a, b in shifts)
    assert any(b == 1 and a <= 0 for a, b in shifts)
    assert novel_keys >= 3 and three_roles >= 10
    for i, (nt, nq) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(800+i, nt, nq)
        assert nt is None or len(p['train']) == nt
        assert nq is None or len(p['test']) == nq
        assert _puzzle_key(p) == _puzzle_key(generate_puzzle(800+i, nt, nq))
    for nt, nq in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (3.5, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError('Illegal pair counts accepted')
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in examples[0][s]] for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        return
    raise AssertionError('Wrong tally accepted')


# Exact source contents: regression tests and copy checks, never generation templates.
_SOURCE_ROWS = (('3333333333333333333333\n3333333333333333388333\n3338833333333333383833\n3338383333333333338333\n3333833333333333333333\n3333333333333333333333\n3333333333338833333333\n3322333333338383333333\n3322333333333833333333\n3333333333333333333333\n3333333333333333333333\n3333333333333333333333\n3333333333343333333333\n0000000333434333333333\n0800000333343333333333\n0000000333333333333333\n0400000333333333223333\n0000000333333333223333\n0200000333333333333333\n0000000333333333333333', '0000000\n0808080\n0000000\n0400000\n0000000\n0202000\n0000000'), ('11111111111000000000000\n11111111111020000000000\n11222111111000000000000\n11121111111080000000000\n11222111311000000000000\n11111113331030000000000\n13111111311000000000000\n33311111111111111111111\n13111111111111111111111\n11111111111111111111111\n11118181111111181811111\n11111811111111118111111\n11118181111111181811111\n11111111111111111111111\n11111111111111111111111\n11111111111111111113111\n11111111111111111133311\n11111411111131111113111\n11114441111333111111111\n11111111111131111111111\n11111111111111111222111\n11111111111111111121111\n11111111111111111222111', '000000000000\n020200000000\n000000000000\n080800000000\n000000000000\n030303030000\n000000000000'), ('88888888888888888888\n88888888888888888888\n88888888800000888888\n88888888801000888888\n88888888800000888888\n88888888804000888888\n88888888800000888888\n88888888888888888888\n88818888888888888888\n88111888888888888888\n88888888888888888888\n88888888888888888888\n88888888881888888888\n88888888811188888888\n88888888888888888888\n88822888888884888888\n88822888888844488888\n88888888888848488888\n88888888888888888888\n88888888888888888888\n88888888888888888888', '00000\n01010\n00000\n04000\n00000'), ('4444444444444444444444444\n4444884444434444444444444\n4444884444333444444448844\n4644444444434444344448844\n4664444444444443334444444\n4644444244444444344444444\n4444442424444444444441114\n4444444244444444444441414\n4434444444111444488444444\n4333444444141444488444444\n4434444444444444444444444\n4444444444444444444442444\n4444444444444444444424244\n4000000000004444444442444\n4010000000004434444444444\n4000000000004333444444444\n4020000000004434444444444\n4000000000004444444488444\n4080000000004444444488444\n4000000000004443444444444\n4030000000004433344444444\n4000000000004443444444444', '00000000000\n01010000000\n00000000000\n02020000000\n00000000000\n08080808000\n00000000000\n03030303030\n00000000000'))
_SOURCE_GRIDS = tuple(tuple(np.array([[int(c) for c in row] for row in text.splitlines()], dtype=np.int8) for text in pair) for pair in _SOURCE_ROWS)
_SOURCE_PAIRS = {(_grid_key(inp), _grid_key(out)) for inp, out in _SOURCE_GRIDS}
