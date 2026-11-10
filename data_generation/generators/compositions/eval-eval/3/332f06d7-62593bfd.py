"""Linked shape drift: the first wall contact halts a rotational class.

Standard library and NumPy only. Generation and all diagnostics stay in memory.
"""
from hashlib import sha256

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if (r, c) != (0, 0))
SEEDS = (0, 93507096, 70637061, 358919036, 632965177, 576859471,
         142121589, 219002937, 18473628, 19319508, 925540385, 438635989,
         861004869, 286121199, 130610836, 780381949, 94615531, 569189341,
         42662583, 634095467, 346146277, 855041639, 151595227, 429351966,
         102414150, 839079581, 58830606, 900069576, 211008301, 112294439,
         659012319, 621124422, 491432179, 883871790, 820636904, 330600018,
         7286843, 458217427, 78632513, 799089978, 441704711, 11060245)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _counts(num_train, num_test):
    for value, lo, hi, name in ((num_train, 2, 10, 'num_train'),
                                (num_test, 1, 4, 'num_test')):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(name + ' must be an integer in its legal range')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), 'grid dimensions')
    _require(np.issubdtype(a.dtype, np.integer), 'integer colors required')
    _require(bool(np.all((0 <= a) & (a <= 9))), 'color outside 0..9')
    return a.astype(np.int8, copy=False)


def _trim(mask):
    cells = np.argwhere(mask)
    _require(len(cells) > 0, 'empty mask')
    lo, hi = cells.min(axis=0), cells.max(axis=0) + 1
    return np.array(mask[lo[0]:hi[0], lo[1]:hi[1]], dtype=bool, copy=True)


def _key(mask):
    m = _trim(mask)
    return int(m.shape[0]), int(m.shape[1]), m.tobytes()


def _canonical(mask):
    return min(_key(np.rot90(mask, k)) for k in range(4))


def _mirror(mask):
    return _canonical(np.fliplr(mask))


def _segment(grid, directions=DIR8):
    wall = int(grid[0, 0])
    visited = np.zeros(grid.shape, dtype=bool)
    objects = []
    for r, c in np.argwhere((grid != 0) & (grid != wall)):
        r, c = int(r), int(c)
        if visited[r, c]:
            continue
        color = int(grid[r, c])
        cells, todo = [], [(r, c)]
        visited[r, c] = True
        for y, x in todo:
            cells.append((y, x))
            for dy, dx in directions:
                a, b = y + dy, x + dx
                if (0 <= a < grid.shape[0] and 0 <= b < grid.shape[1]
                        and not visited[a, b] and int(grid[a, b]) == color):
                    visited[a, b] = True
                    todo.append((a, b))
        top = min(y for y, x in cells)
        left = min(x for y, x in cells)
        mask = np.zeros((max(y for y, x in cells) - top + 1,
                         max(x for y, x in cells) - left + 1), dtype=bool)
        for y, x in cells:
            mask[y-top, x-left] = True
        objects.append({'mask': mask, 'top': top, 'left': left, 'color': color})
    return objects


def _clearance(grid, obj, direction, *, box=False, interior=True):
    """Check every intermediate placement, never jump across a wall."""
    h, w = grid.shape
    mask = np.ones(obj['mask'].shape, dtype=bool) if box else obj['mask']
    ys, xs = np.nonzero(mask)
    ys, xs = ys + obj['top'], xs + obj['left']
    wall = int(grid[0, 0])
    for step in range(1, h + 1):
        rows = ys + direction * step
        if np.any(rows < 0) or np.any(rows >= h):
            return step - 1
        contact = grid[rows, xs] == wall if interior else ((rows == 0) | (rows == h-1))
        if bool(np.any(contact)):
            return step - 1
    raise AssertionError('frame failed to stop a piece')


def _classification_key(obj, matching):
    m = obj['mask']
    if matching == 'rotation':
        return _canonical(m)
    if matching == 'reflection':
        return min(_canonical(m), _mirror(m))
    if matching == 'identity':
        return _key(m)
    if matching == 'area':
        return int(m.sum())
    if matching == 'color':
        return obj['color']
    raise ValueError('unknown matching convention')


def _predict(grid, *, matching='rotation', sync=True, box=False,
             interior=True, exactly_two=False, connectivity=8, all_single=False):
    objects = _segment(grid, DIR8 if connectivity == 8 else DIR4)
    groups = {}
    for i, obj in enumerate(objects):
        key = i if all_single else _classification_key(obj, matching)
        groups.setdefault(key, []).append(i)
    distances, directions, individual = {}, {}, {}
    for members in groups.values():
        repeated = len(members) == 2 if exactly_two else len(members) > 1
        direction = -1 if repeated else 1
        clear = [_clearance(grid, objects[i], direction, box=box, interior=interior)
                 for i in members]
        for i, clearance in zip(members, clear):
            directions[i] = direction
            individual[i] = clearance
            distances[i] = min(clear) if sync else clearance
    out = grid.copy()
    wall = int(grid[0, 0])
    for obj in objects:
        y, x = np.nonzero(obj['mask'])
        out[y + obj['top'], x + obj['left']] = 0
    for i, obj in enumerate(objects):
        y, x = np.nonzero(obj['mask'])
        y = y + obj['top'] + directions[i] * distances[i]
        x = x + obj['left']
        # Some alternative hypotheses can collide; that is also a contradiction.
        if np.any(out[y, x] != 0):
            return None, None
        out[y, x] = obj['color']
    _require(np.array_equal(out == wall, grid == wall), 'wall altered')
    return out, {'objects': objects, 'groups': list(groups.values()),
                 'directions': directions, 'distances': distances, 'individual': individual}


def _input_constraints(grid):
    wall = int(grid[0, 0])
    _require(wall != 0, 'frame cannot be floor')
    _require(all(bool(np.all(edge == wall)) for edge in
                 (grid[0], grid[-1], grid[:, 0], grid[:, -1])), 'nonuniform wall frame')
    objects = _segment(grid)
    _require(bool(objects), 'no pieces')
    intervals = sorted((o['left'], o['left'] + o['mask'].shape[1]) for o in objects)
    _require(all(a[1] < b[0] for a, b in zip(intervals, intervals[1:])),
             'piece column intervals need a separating column')
    return objects


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only rule; no generator state, seed, examples, or file access."""
    grid = _grid(input_grid)
    _input_constraints(grid)
    out, _ = _predict(grid)
    _require(out is not None, 'invalid overlapping destinations')
    return out


def _connected(mask, directions):
    cells = set(map(tuple, np.argwhere(mask)))
    todo, seen = [min(cells)], {min(cells)}
    for r, c in todo:
        for dr, dc in directions:
            p = r + dr, c + dc
            if p in cells and p not in seen:
                seen.add(p)
                todo.append(p)
    return len(seen) == len(cells)


def _random_mask(rng, limit, area=None):
    if area is None:
        kind = int(rng.integers(7))
        h, w = map(int, rng.integers(2, limit + 1, size=2))
        if kind == 0:
            return np.ones((1, int(rng.integers(1, limit + 1))), dtype=bool)
        if kind == 1:
            m = np.ones((h, w), dtype=bool)
            if h > 2 and w > 2 and rng.random() < .7:
                m[1:-1, 1:-1] = False
            return m
        if kind == 2:
            m = np.zeros((h, w), dtype=bool)
            m[:, 0] = m[-1, :] = True
            return m
        if kind == 3:
            m = np.zeros((h, w), dtype=bool)
            m[int(rng.integers(h)), :] = True
            m[:, int(rng.integers(w))] = True
            return m
        directions = DIR4 if kind == 4 else DIR8
        area = int(rng.integers(2, min(15, limit * limit) + 1))
    else:
        directions = DIR4 if rng.random() < .75 else DIR8
    cells = {(0, 0)}
    for _ in range(area - 1):
        choices = set()
        for r, c in sorted(cells):
            for dr, dc in directions:
                p = r + dr, c + dc
                if p in cells:
                    continue
                trial = cells | {p}
                rows, cols = zip(*trial)
                if max(rows) - min(rows) < limit and max(cols) - min(cols) < limit:
                    choices.add(p)
        _require(bool(choices), 'mask growth exhausted')
        ordered = sorted(choices)
        cells.add(ordered[int(rng.integers(len(ordered)))])
    lo_r, lo_c = min(r for r, c in cells), min(c for r, c in cells)
    m = np.zeros((max(r for r, c in cells) - lo_r + 1,
                  max(c for r, c in cells) - lo_c + 1), dtype=bool)
    for r, c in cells:
        m[r-lo_r, c-lo_c] = True
    return m


def _sample_mask(rng, forbidden, limit, *, chiral=False, area=None, diagonal=False,
                 budget=192):
    def permitted(mask):
        return (_canonical(mask) not in forbidden and _mirror(mask) not in forbidden
                and (area is None or int(mask.sum()) == area)
                and (not chiral or _canonical(mask) != _mirror(mask))
                and (not diagonal or not _connected(mask, DIR4)))

    for _ in range(budget):
        mask = _random_mask(rng, limit, area)
        if permitted(mask):
            return mask
    # Small exact fallback; at most 511 masks, never a saved retry corpus.
    side = min(limit, 3)
    for bits in range(1, 1 << (side * side)):
        mask = _trim(np.array([(bits >> i) & 1 for i in range(side * side)],
                              dtype=bool).reshape(side, side))
        if _connected(mask, DIR8) and permitted(mask):
            return mask
    raise AssertionError('finite connected mask vocabulary exhausted')


def _notches(mask):
    return [c for c in range(mask.shape[1]) if int(np.flatnonzero(mask[:, c])[0]) > 0]


def _objects(rng, profile, limit):
    objects, forbidden = [], set()

    def add(base, count, group, turns=None):
        if turns is None:
            turns = [int(rng.integers(4)) for _ in range(count)]
        for turn in turns:
            objects.append({'mask': _trim(np.rot90(base, turn)), 'group': group})
        forbidden.add(_canonical(base))
        forbidden.add(_mirror(base))

    if profile == 'contrast':
        # Equal-area nonmatches, chirality, quarter turns, and a leading notch.
        base = _sample_mask(rng, forbidden, min(4, limit), chiral=True, area=4)
        orientations = [k for k in range(4) if _notches(np.rot90(base, k))]
        k = int(rng.choice(orientations))
        add(base, 2, 0, [k, (k + 1) % 4])
        objects[0]['notched_culprit'] = True
        add(np.fliplr(base), 1, 1)
        add(_sample_mask(rng, forbidden, limit, area=4), 1, 2)
    elif profile == 'threshold':
        # A diagonal singleton becomes repeated single cells under four-
        # connectivity, reversing its direction under that shortcut.
        singleton = np.eye(2, dtype=bool)
        forbidden.add(_canonical(singleton))
        forbidden.add(_mirror(singleton))
        base = _sample_mask(rng, forbidden, 3)
        k = int(rng.integers(4))
        add(base, 3, 0, [k, k, (k + 1) % 4])
        add(_sample_mask(rng, forbidden, 3, diagonal=True), 2, 1)
        add(singleton, 1, 2)
    else:
        if profile == 'test':
            sizes = [int(rng.choice([2, 3])), int(rng.choice([2, 3]))]
            singles = int(rng.integers(min(2, 7 - sum(sizes)) + 1))
        else:
            count = int(rng.choice([0, 1, 2], p=[.10, .60, .30]))
            sizes = [int(rng.choice([2, 3])) for _ in range(count)]
            singles = int(rng.integers(0, min(3, 7 - sum(sizes)) + 1))
            if sum(sizes) + singles < 3:
                singles = max(singles, 3 - sum(sizes))
        for i, size in enumerate(sizes + [1] * singles):
            add(_sample_mask(rng, forbidden, limit), size, i)
    return objects


def _paint_piece(grid, obj, top):
    y, x = np.nonzero(obj['mask'])
    y, x = y + top, x + obj['left']
    _require(bool(np.all(grid[y, x] == 0)), 'paint collision')
    grid[y, x] = obj['color']


def _make_pair(rng, profile, wall):
    """Finite packing attempts; no sampled path metadata is used by solve."""
    for attempt in range(64):
        limit = (int(rng.choice([3, 4, 5])) if profile == 'test' else
                 4 if profile == 'contrast' else int(rng.choice([3, 4])))
        if attempt >= 32 and profile not in ('contrast', 'threshold'):
            limit = 3
        objects = _objects(rng, profile, limit)
        minimum_width = sum(o['mask'].shape[1] for o in objects) + len(objects) + 1
        if minimum_width > 30:
            continue
        height = int(rng.integers(24, 31) if profile == 'test' else rng.integers(17, 28))
        width = int(rng.integers(minimum_width, min(30, minimum_width + 6) + 1))
        extras = rng.multinomial(width - minimum_width, np.full(len(objects) + 1,
                                                                1 / (len(objects) + 1)))
        cursor = 1 + int(extras[0])
        for j, index in enumerate(rng.permutation(len(objects))):
            obj = objects[int(index)]
            obj['left'] = cursor
            cursor += obj['mask'].shape[1] + 1 + int(extras[j + 1])
        palette = [int(c) for c in rng.permutation(np.arange(1, 10)) if c != wall]
        ncolors = int(rng.integers(1, min(8, len(objects)) + 1))
        for obj in objects:
            obj['color'] = int(rng.choice(palette[:ncolors]))
        if profile == 'contrast':
            for obj, color in zip(objects, (palette[0], palette[1], palette[0], palette[1])):
                obj['color'] = color

        groups = {}
        for i, obj in enumerate(objects):
            groups.setdefault(obj['group'], []).append(i)
        for members in groups.values():
            direction = -1 if len(members) > 1 else 1
            force = profile in ('contrast', 'threshold') and objects[members[0]]['group'] == 0
            jam = force or rng.random() < .78
            culprit = next((i for i in members if objects[i].get('notched_culprit')),
                           int(rng.choice(members)))
            # A fixed clear singleton demonstrates arrival at the bottom frame.
            if profile == 'contrast' and objects[members[0]]['group'] == 1:
                jam = False
            q = int(rng.integers(3, 6))
            for i in members:
                obj = objects[i]
                h, w = obj['mask'].shape
                obj['direction'] = direction
                if jam:
                    planned = q if i == culprit else int(rng.integers(q + 2, q + 6))
                    planned = min(planned, height - h - 4)
                    blocked = i == culprit or rng.random() < .5
                    if direction == -1:
                        obj['top'] = int(rng.integers(planned + 2, height - h))
                        edge = obj['top'] - 1
                    else:
                        obj['top'] = int(rng.integers(1, height - h - planned - 1))
                        edge = height - h - 1 - obj['top']
                    obj['clearance'] = planned if blocked else edge
                    obj['blocked'] = blocked
                else:
                    obj['top'] = int(rng.integers(2, height - h - 1))
                    obj['clearance'] = (obj['top'] - 1 if direction == -1 else
                                        height - h - 1 - obj['top'])
                    obj['blocked'] = False

        grid = np.zeros((height, width), dtype=np.int8)
        grid[0] = grid[-1] = wall
        grid[:, 0] = grid[:, -1] = wall
        for obj in objects:
            if not obj['blocked']:
                continue
            m, top, left = obj['mask'], obj['top'], obj['left']
            direction, travel = obj['direction'], obj['clearance']
            ys, xs = np.nonzero(m)
            swept = np.zeros((height, m.shape[1]), dtype=bool)
            for step in range(travel + 1):
                swept[ys + top + direction * step, xs] = True
            options = _notches(m) if obj.get('notched_culprit') else list(range(m.shape[1]))
            col = int(rng.choice(options))
            leading = int(np.flatnonzero(m[:, col])[0 if direction == -1 else -1])
            row = top + leading + direction * (travel + 1)
            _require(0 < row < height - 1 and not swept[row, col], 'invalid planned contact')
            grid[row, left + col] = wall
            # Secondary teeth never obstruct the guaranteed prefix. They can
            # lie beyond the first contact, making jumping to a later fit wrong.
            allowed = ~swept
            allowed[0] = allowed[-1] = False
            if direction == -1:
                allowed[top:] = False
            else:
                allowed[:top + m.shape[0]] = False
            candidates = np.argwhere(allowed)
            teeth = int(rng.integers(2, 7) if profile == 'test' else rng.integers(0, 4))
            for _ in range(teeth):
                if len(candidates) == 0:
                    break
                y, x = map(int, candidates[int(rng.integers(len(candidates)))])
                for dx in range(int(rng.integers(1, m.shape[1] + 1))):
                    c = (x + dx) % m.shape[1]
                    if allowed[y, c]:
                        grid[y, left + c] = wall
        out = grid.copy()
        for obj in objects:
            _paint_piece(grid, obj, obj['top'])
        for members in groups.values():
            travel = min(objects[i]['clearance'] for i in members)
            for i in members:
                obj = objects[i]
                _paint_piece(out, obj, obj['top'] + obj['direction'] * travel)
        pair = {'input': grid, 'output': out}
        _require(np.array_equal(solve(grid), out), 'planned clearance disagrees with input')
        if not np.array_equal(grid, out):
            return pair
    raise AssertionError('finite arena packing budget exhausted')


def _fingerprint(grid):
    return tuple(grid.shape), np.asarray(grid, dtype=np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    stable_wall = int(rng.integers(1, 10)) if rng.random() < .25 else None
    profiles = ['contrast', 'threshold'] + ['free'] * (num_train - 2)
    profiles = [profiles[int(i)] for i in rng.permutation(num_train)]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, modes in (('train', profiles), ('test', ['test'] * num_test)):
        for mode in modes:
            for _ in range(12):
                wall = stable_wall if stable_wall is not None else int(rng.integers(1, 10))
                pair = _make_pair(rng, mode, wall)
                key = _fingerprint(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError('distinct arena budget exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _different(grid, out, **kwargs):
    candidate, _ = _predict(grid, **kwargs)
    return candidate is None or not np.array_equal(candidate, out)


def _pair_features(inp, out):
    objects = _input_constraints(inp)
    expected, info = _predict(inp)
    _require(expected is not None and np.array_equal(out, expected), 'incorrect linked drift')
    wall = int(inp[0, 0])
    _require(np.array_equal(inp == wall, out == wall), 'static walls changed')
    actual = _segment(out)
    desired = sorted((o['color'], o['left'], _key(o['mask']),
                      o['top'] + info['directions'][i] * info['distances'][i])
                     for i, o in enumerate(objects))
    observed = sorted((o['color'], o['left'], _key(o['mask']), o['top']) for o in actual)
    _require(desired == observed, 'shape, orientation, columns or colors changed')
    _require(not np.array_equal(inp, out), 'all pieces stationary')
    # Direct first-contact and lockstep invariants, reconstructed from pixels.
    for group in info['groups']:
        _require(len({info['distances'][i] for i in group}) == 1, 'class is not synchronous')
        for i in group:
            o = objects[i]
            ys, xs = np.nonzero(o['mask'])
            end = o['top'] + info['directions'][i] * info['individual'][i]
            _require(bool(np.all(inp[ys + end, xs + o['left']] != wall)), 'last valid placement')
            _require(bool(np.any(inp[ys + end + info['directions'][i], xs + o['left']] == wall)),
                     'first invalid placement does not contact wall')
    return {
        'shape': tuple(inp.shape), 'wall': wall, 'pieces': len(objects),
        'classes': tuple(sorted(len(g) for g in info['groups'])),
        'areas': tuple(int(o['mask'].sum()) for o in objects),
        'extents': tuple(o['mask'].shape for o in objects),
        'diagonal': any(not _connected(o['mask'], DIR4) for o in objects),
        'linked_unequal': any(len(g) > 1 and len({info['individual'][i] for i in g}) > 1
                              and info['distances'][g[0]] > 0 for g in info['groups']),
        'moves': tuple(info['distances'].values()),
        'masks': tuple(_canonical(o['mask']) for o in objects),
        'interior_wall_cells': int((inp[1:-1, 1:-1] == wall).sum()),
    }


ALTERNATIVES = {
    'independent': {'sync': False},
    'no_walls': {'interior': False},
    'bounding_box': {'box': True},
    'no_companionship': {'all_single': True},
    'color': {'matching': 'color'},
    'orientation': {'matching': 'identity'},
    'area': {'matching': 'area'},
    'reflection': {'matching': 'reflection'},
    'exactly_two': {'exactly_two': True},
    'four_connectivity': {'connectivity': 4},
}


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    try:
        _counts(len(puzzle['train']), len(puzzle['test']))
    except ValueError as error:
        raise AssertionError(str(error)) from error
    seen, train, features = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'canvas changed')
            key = _fingerprint(inp)
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            features.append(_pair_features(inp, out))
            if split == 'train':
                train.append((inp, out))
    _require(any(f['linked_unequal'] for f in features[:len(train)]), 'missing remote-jam teaching')
    for name, kwargs in ALTERNATIVES.items():
        _require(any(_different(inp, out, **kwargs) for inp, out in train),
                 'training does not distinguish ' + name)


def _synthetic_checks():
    rng = np.random.default_rng(9)
    mask = _sample_mask(rng, set(), 3, chiral=True, area=4, budget=0)
    _require(int(mask.sum()) == 4 and _canonical(mask) != _mirror(mask), 'chiral fallback')
    diagonal = _sample_mask(rng, {_canonical(mask), _mirror(mask)}, 3,
                            diagonal=True, budget=0)
    _require(_connected(diagonal, DIR8) and not _connected(diagonal, DIR4),
             'diagonal fallback')
    # Two identical pieces jam together at the same step; singleton is already
    # against the bottom wall. The two interior contacts are given explicitly.
    inp = np.zeros((10, 12), dtype=np.int8)
    inp[0] = inp[-1] = inp[:, 0] = inp[:, -1] = 9
    inp[5:7, 2:4] = 2
    inp[5:7, 6:8] = 3
    inp[2, 2] = inp[2, 6] = 9
    inp[8, 10] = 4
    expected = inp.copy()
    expected[5:7, 2:4] = expected[5:7, 6:8] = 0
    expected[3:5, 2:4] = 2
    expected[3:5, 6:8] = 3
    _require(np.array_equal(solve(inp), expected), 'tied stops or immediate singleton')
    # One immediate contact prevents even the unobstructed companion moving.
    inp[4, 2] = 9
    _require(np.array_equal(solve(inp), inp), 'immediate group jam')
    # Multiplicity four is algorithmically supported although sampled up to 3.
    inp = np.zeros((12, 18), dtype=np.int8)
    inp[0] = inp[-1] = inp[:, 0] = inp[:, -1] = 9
    for color, col in enumerate((2, 6, 10, 14), 1):
        inp[6:8, col:col+2] = color
    inp[3, 2] = 9
    out = inp.copy()
    for color, col in enumerate((2, 6, 10, 14), 1):
        out[6:8, col:col+2] = 0
        out[4:6, col:col+2] = color
    _require(np.array_equal(solve(inp), out), 'four-member class')


def self_test():
    """Stream 42 seeds and retain only small aggregate feature sets."""
    _synthetic_checks()
    shapes, masks, signatures, profiles, walls, piece_counts, extents = (set() for _ in range(7))
    pairs, linked, diagonal, minmove, maxmove = 0, 0, 0, 30, 0
    structural = set()
    for seed in SEEDS:
        puzzle = generate_puzzle(seed)
        digest = sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                f = _pair_features(inp, out)
                shapes.add(f['shape'])
                masks.update(f['masks'])
                walls.add(f['wall'])
                profiles.add(f['classes'])
                piece_counts.add(f['pieces'])
                extents.update(f['extents'])
                linked += int(f['linked_unequal'])
                diagonal += int(f['diagonal'])
                minmove = min(minmove, min(f['moves']))
                maxmove = max(maxmove, max(f['moves']))
                pairs += 1
                roles = {0: 0, int(inp[0, 0]): 1}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in inp.ravel())
                structural.add(sha256(bytes(inp.shape) + normalized).digest())
                for g in (inp, out):
                    digest.update(bytes(g.shape))
                    digest.update(g.tobytes())
        signatures.add(digest.digest())
    _require(len(signatures) == 42, 'cross-seed repetition')
    _require(len(shapes) >= 40 and len(structural) >= 170, 'geometry variation collapsed')
    _require(len(masks) >= 100 and len(profiles) >= 7, 'mask/class variation collapsed')
    _require(walls == set(range(1, 10)), 'wall palette coverage')
    _require({3, 4, 5, 6, 7} <= piece_counts, 'piece-count variation')
    _require(max(max(e) for e in extents) == 5, 'larger test masks absent')
    _require(linked >= 100 and diagonal >= 42, 'interaction/connectivity coverage')
    _require(maxmove >= 15, 'long travel absent')
    for n in range(2, 11):
        for m in range(1, min(4, n-1) + 1):
            p = generate_puzzle(814 + n * 10 + m, n, m)
            _require(len(p['train']) == n and len(p['test']) == m, 'explicit counts')
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    a, b = generate_puzzle(0), generate_puzzle(0)
    _require(all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s])
                 for k in ('input', 'output')), 'seed nondeterminism')
    a['test'][0]['output'][0, 0] = 0
    try:
        validate_generated_puzzle(a)
    except AssertionError:
        pass
    else:
        raise AssertionError('corruption accepted')
    return {'seeds': 42, 'pairs': pairs, 'canvas_shapes': len(shapes),
            'normalized_inputs': len(structural), 'rotational_masks': len(masks),
            'class_profiles': len(profiles), 'piece_counts': sorted(piece_counts),
            'linked_unequal_scenes': linked, 'diagonal_scenes': diagonal,
            'travel_range': [minmove, maxmove]}
