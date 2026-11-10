"""Relay thread: conditional color transport through closed chambers.

No parent code is imported. Generation retains one puzzle at a time and writes
nothing; self_test streams 42 seeds and retains only bounded hashes/features.
"""
from __future__ import annotations
import hashlib
import json
import numpy as np

D4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _components(cells, diagonal=False):
    todo = set(cells)
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in D8 if diagonal else D4:
                q = (r + dr, c + dc)
                if q in todo:
                    todo.remove(q)
                    group.add(q)
                    stack.append(q)
        result.append(group)
    return result


def _coords(mask):
    return set(map(tuple, np.argwhere(mask).tolist()))


def _chain(cells):
    cells = set(cells)
    neighbors = {q: [p for dr, dc in D4 if (p := (q[0]+dr, q[1]+dc)) in cells] for q in cells}
    ends = sorted(q for q in cells if len(neighbors[q]) == 1)
    assert len(ends) == 2 and all(len(v) in (1, 2) for v in neighbors.values()), 'not a simple thread'
    assert ends[0][0] != ends[1][0], 'endpoint row tie'
    path, previous, current = [], None, ends[0]
    while True:
        path.append(current)
        following = [q for q in neighbors[current] if q != previous]
        if not following:
            break
        previous, current = current, following[0]
        assert len(path) <= len(cells)
    assert len(path) == len(cells), 'disconnected thread'
    return path


def _holes(wall):
    """Four-connected complement flood in a padded object box."""
    rows, cols = zip(*wall)
    r0, r1, c0, c1 = min(rows)-1, max(rows)+1, min(cols)-1, max(cols)+1
    empty = {(r, c) for r in range(r0, r1+1) for c in range(c0, c1+1)} - wall
    return [g for g in _components(empty) if not any(r in (r0, r1) or c in (c0, c1) for r, c in g)]


def _read(input_grid):
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1
    bg = int(colors[counts.argmax()])
    assert np.count_nonzero(a == bg) > a.size // 2, 'background must be a strict majority'
    h, w = a.shape
    # A key is an isolated, solid, multicolor edge component, exactly 2 by 3..6.
    keys = []
    for cells in _components(_coords(a != bg), True):
        rr, cc = zip(*cells)
        r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
        sh, sw = r1-r0+1, c1-c0+1
        if len(cells) != sh*sw:
            continue
        if sh == 2 and 3 <= sw <= 6 and (r0 == 0 or r1 == h-1):
            side = 'top' if r0 == 0 else 'bottom'
            outer, inner = (r0, r1) if side == 'top' else (r1, r0)
            src, dst = list(map(int, a[outer, c0:c1+1])), list(map(int, a[inner, c0:c1+1]))
        elif sw == 2 and 3 <= sh <= 6 and (c0 == 0 or c1 == w-1):
            side = 'left' if c0 == 0 else 'right'
            outer, inner = (c0, c1) if side == 'left' else (c1, c0)
            src, dst = list(map(int, a[r0:r1+1, outer])), list(map(int, a[r0:r1+1, inner]))
        else:
            continue
        if len(set(src)) == len(src) and len(set(src+dst)) >= 3:
            keys.append((cells, dict(zip(src, dst)), side))
    assert len(keys) == 1, 'need one local key'
    key, mapping, side = keys[0]
    key_palette = set(mapping) | set(mapping.values())
    extra = set(map(int, colors)) - key_palette - {bg}
    assert len(extra) == 1, 'thread color is the only non-key ink color'
    wire = extra.pop()
    wire_cells = _coords(a == wire)
    # The colored seed is the unique non-key ink cell next to a wire endpoint
    # which completes a simple chain starting at that colored, upper endpoint.
    degree = {q: sum((q[0]+dr, q[1]+dc) in wire_cells for dr, dc in D4) for q in wire_cells}
    possible = set()
    for q, n in degree.items():
        if n == 1:
            for dr, dc in D4:
                p = q[0]+dr, q[1]+dc
                if 0 <= p[0] < h and 0 <= p[1] < w and p not in key and int(a[p]) in mapping:
                    possible.add(p)
    candidates = []
    for seed in sorted(possible):
        try:
            path = _chain(wire_cells | {seed})
        except AssertionError:
            continue
        if path[0] == seed:
            candidates.append((seed, path))
    assert len(candidates) == 1, 'seed must mark uniquely upper endpoint'
    seed, path = candidates[0]
    path_set, index = set(path), {q: i for i, q in enumerate(path)}
    rest = _coords(a != bg) - key - path_set
    objects = []
    for color in sorted(mapping):
        for wall in _components({q for q in rest if int(a[q]) == color}, True):
            contacts = {p for r, c in wall for dr, dc in D4 if (p := (r+dr, c+dc)) in path_set}
            assert len(contacts) == 1, 'each chamber has one thread contact'
            position = index[next(iter(contacts))]
            assert 1 < position < len(path)-2, 'contacts stay away from endpoints'
            cavities = _holes(wall)
            assert all(int(a[q]) == bg for hole in cavities for q in hole), 'occupied cavity'
            objects.append({'wall': wall, 'color': color, 'index': position, 'holes': cavities})
    assert set().union(*(o['wall'] for o in objects)) == rest, 'unlisted wall color'
    assert len({o['index'] for o in objects}) == len(objects), 'contact tie'
    # Different objects may share a color, but remain spatially distinct.
    for i, o in enumerate(objects):
        for p in objects[i+1:]:
            assert not any((r+dr, c+dc) in p['wall'] for r, c in o['wall'] for dr, dc in D8)
    objects.sort(key=lambda o: o['index'])
    return {'grid': a, 'background': bg, 'wire': wire, 'key': key, 'mapping': mapping,
            'side': side, 'seed': seed, 'path': path, 'objects': objects}


def _execute(info, policy='relay'):
    a, mapping = info['grid'], info['mapping']
    out = a.copy()
    state = int(a[info['seed']])
    events = []
    objects = info['objects']
    if policy == 'raster':
        objects = sorted(objects, key=lambda o: min(o['wall']))
    for o in objects:
        before = state
        closed = bool(o['holes'])
        match = o['color'] == state
        active = closed and match
        if policy == 'all_mapped':
            active = closed
        if policy == 'identity':
            destination = state
        else:
            destination = mapping.get(state, state)
        if active:
            fill = mapping[o['color']] if policy == 'all_mapped' else destination
            for hole in o['holes']:
                for q in hole:
                    out[q] = fill
            if policy != 'static':
                state = fill
        elif policy == 'open_advances' and match:
            state = destination
        events.append((before, o['color'], closed, active, state))
    return out, events


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover all conventions, topology, and state transitions from input only."""
    return _execute(_read(input_grid))[0]


def _grow(rng, h, w, length):
    path = [(int(rng.integers(4, h)), int(rng.integers(w)))]
    occupied, previous = set(path), None
    persistence = float(rng.uniform(3, 15))
    for _ in range(length-1):
        choices, weights = [], []
        for dr, dc in D4:
            q = path[-1][0]+dr, path[-1][1]+dc
            if not (3 <= q[0] < h and 0 <= q[1] < w) or q in occupied:
                continue
            if sum((q[0]+a, q[1]+b) in occupied for a, b in D4) != 1:
                continue
            choices.append(q)
            weights.append(persistence if (dr, dc) == previous else 1)
        if not choices:
            return None
        q = choices[int(rng.choice(len(choices), p=np.array(weights)/sum(weights)))]
        previous = q[0]-path[-1][0], q[1]-path[-1][1]
        occupied.add(q)
        path.append(q)
    return path


def _local_shape(rng, kind, small=False):
    if kind == 'diamond':
        radius = 1 if small else int(rng.choice([1, 1, 2]))
        rr, cc = np.indices((2*radius+1, 2*radius+1))
        wall = abs(rr-radius)+abs(cc-radius) == radius
    elif kind == 'profile':
        cavity = {(1, 1), (2, 1), (2, 2)}
        wall = np.zeros((4, 4), dtype=bool)
        shell = {(r+dr, c+dc) for r, c in cavity for dr, dc in D4} - cavity
        for q in shell:
            wall[q] = True
    elif kind == 'multi':
        wall = np.ones((3, 5), dtype=bool)
        wall[1, 1] = wall[1, 3] = False
    else:
        ih = 1 if small or kind == 'open' else int(rng.integers(1, 3))
        iw = 1 if small or kind == 'open' else int(rng.integers(1, 4))
        wall = np.ones((ih+2, iw+2), dtype=bool)
        wall[1:-1, 1:-1] = False
        if kind == 'open':
            wall[0, 1] = False
    # A short stalk creates exactly one contact even along a straight thread.
    stalk = 1 if small else int(rng.choice([1, 1, 2]))
    mid = wall.shape[0]//2
    assert wall[mid, 0]
    wall = np.pad(wall, ((0, 0), (stalk, 0)))
    wall[mid, :stalk] = True
    coords = np.argwhere(wall) - np.array([mid, -1])
    return coords


def _place(rng, path, h, w, kinds):
    path_set, busy, objects = set(path), set(), []
    used_contacts = set()
    # Large objects first; only geometry is placed here, never event labels.
    locals_ = [_local_shape(rng, kind) for kind in kinds]
    order = sorted(range(len(kinds)), key=lambda i: -len(locals_[i]))
    for j in order:
        transforms = []
        for turn in range(4):
            co = locals_[j].copy()
            for _ in range(turn):
                co = np.stack((co[:, 1], -co[:, 0]), axis=1)
            transforms.append([(int(r), int(c)) for r, c in co])
        halo = {(r+dr, c+dc) for r, c in busy for dr, dc in ((0, 0),)+D8}
        # A uniform random ordering followed by first valid is uniform over the
        # valid placements, without retaining or evaluating every valid payload.
        candidates = [(t, i) for t in range(4) for i in range(4, len(path)-4) if i not in used_contacts]
        chosen = None
        for k in rng.permutation(len(candidates)):
            turn, i = candidates[int(k)]
            cells = {(r+path[i][0], c+path[i][1]) for r, c in transforms[turn]}
            rr, cc = zip(*cells)
            r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
            if r0 < 3 or r1 >= h or c0 < 0 or c1 >= w or cells & (path_set | halo):
                continue
            contacts = {(r+a, c+b) for r, c in cells for a, b in D4 if (r+a, c+b) in path_set}
            if contacts != {path[i]}:
                continue
            if any(r0 <= r <= r1 and c0 <= c <= c1 for r, c in path):
                continue
            chosen = i, cells
            break
        if chosen is None:
            return None
        i, cells = chosen
        objects.append({'wall': cells, 'kind': kinds[j], 'contact': path[i]})
        used_contacts.add(i)
        busy |= cells
    return objects


def _fallback(rng, kinds):
    """Guaranteed 30x30 geometry, used only after bounded random attempts."""
    path = [(r, 9) for r in range(29)] + [(28, c) for c in range(10, 22)] + [(r, 21) for r in range(27, 2, -1)]
    objects = []
    for j, kind in enumerate(kinds):
        contact = (5+5*(j % 5), 9 if j < 5 else 21)
        co = _local_shape(rng, kind, small=True)
        # Left of the first lane, right of the second lane.
        if j < 5:
            co[:, 1] *= -1
        cells = {(int(r+contact[0]), int(c+contact[1])) for r, c in co}
        objects.append({'wall': cells, 'kind': kind, 'contact': contact})
    return 30, 30, path, objects


def _geometry(rng, kinds):
    for attempt in range(96):
        low = 19 if attempt < 24 else 24
        h, w = map(int, rng.integers(low, 31, size=2))
        path = _grow(rng, h, w, int(rng.integers(45, 100)))
        if path is None or path[0][0] == path[-1][0] or path[0][1] == path[-1][1]:
            continue
        objects = _place(rng, path, h, w, kinds)
        if objects is not None and 2*(len(path)+sum(len(o['wall']) for o in objects)+12) < h*w:
            return h, w, path, objects
    return _fallback(rng, kinds)


def _rotate_point(q, shape, turns):
    r, c = q
    h, w = shape
    for _ in range(turns):
        r, c, h, w = w-1-c, r, w, h
    return r, c


def _make_pair(rng, lesson, turns):
    if lesson == 0:
        kinds = ['open', 'diamond', 'profile', 'multi', 'rect', 'rect', 'rect']
        cycle_length = 3
    elif lesson == 1:
        kinds = ['open', 'diamond', 'profile', 'multi', 'rect', 'rect']
        cycle_length = int(rng.integers(2, 5))
    else:
        count = int(rng.integers(8, 10)) if lesson == 'test' else int(rng.integers(4, 8))
        kinds = ['open'] + [str(rng.choice(['rect', 'rect', 'diamond', 'profile', 'multi'])) for _ in range(count-1)]
        cycle_length = int(rng.integers(2, 5))
    terminal_chain = lesson not in (0, 1) and bool(rng.random() < .3)
    if terminal_chain:
        cycle_length = max(3, cycle_length)
    h, w, path, objects = _geometry(rng, kinds)
    shape = h, w
    path = [_rotate_point(q, shape, turns) for q in path]
    for o in objects:
        o['wall'] = {_rotate_point(q, shape, turns) for q in o['wall']}
        o['contact'] = _rotate_point(o['contact'], shape, turns)
    if path[0][0] > path[-1][0]:
        path.reverse()
    indices = {q: i for i, q in enumerate(path)}
    objects.sort(key=lambda o: indices[o['contact']])
    for o in objects:
        o['holes'] = _holes(o['wall'])
    n = int(rng.integers(max(3, cycle_length+1), 7))
    palette = list(map(int, rng.permutation(10)))
    bg, wire = palette[:2]
    sources = palette[2:2+n]
    cycle = sources[:cycle_length]
    mapping = {cycle[i]: cycle[(i+1) % cycle_length] for i in range(cycle_length)}
    if terminal_chain:
        mapping[cycle[-1]] = palette[2+n]
    for c in sources[cycle_length:]:
        mapping[c] = int(rng.choice(sources + palette[2+n:3+n]))
    self_color = sources[cycle_length]
    mapping[self_color] = self_color
    state = self_color if lesson == 1 else cycle[0]
    initial = state
    closed_index = 0
    active_count = 0
    skipped_rect = False
    expected_fills = []
    closed_total = sum(bool(o['holes']) for o in objects)
    for o in objects:
        if not o['holes']:
            color = state if state in mapping else cycle[0]
            active = False
        else:
            if lesson == 0:
                # Early mismatch, activation, repeated old color skipped, then chain.
                active = closed_index not in (0, 2)
            elif lesson == 1:
                active = o['kind'] != 'rect' or skipped_rect
                if o['kind'] == 'rect':
                    skipped_rect = True
            else:
                active = closed_index != 0 and (closed_index >= closed_total-2 or rng.random() < .75)
            active = active and state in mapping
            color = state if active else cycle[(cycle.index(state)+1) % len(cycle)] if state in cycle else cycle[0]
            if lesson == 0 and closed_index == 2:
                color = cycle[0]
            closed_index += 1
        o['color'] = int(color)
        if active:
            state = mapping[color]
            active_count += 1
            expected_fills.append((o['holes'], state))
    assert active_count >= 2
    # Ground truth comes from planted cavities + the program above, not solve().
    base = np.full(shape, bg, dtype=np.int8)
    order = list(map(int, rng.permutation(n)))
    occupied = set(path) | set().union(*(o['wall'] for o in objects))
    key_starts = []
    for start in range(w-n+1):
        cells = {_rotate_point((r, c), shape, turns) for r in (0, 1) for c in range(start, start+n)}
        if not any((r+dr, c+dc) in occupied for r, c in cells for dr, dc in ((0, 0),)+D8):
            key_starts.append(start)
    assert key_starts
    key_start = int(rng.choice(key_starts))
    for j, i in enumerate(order):
        base[0, key_start+j], base[1, key_start+j] = sources[i], mapping[sources[i]]
    grid = np.rot90(base, turns).copy()
    for q in path:
        grid[q] = wire
    grid[path[0]] = initial
    for o in objects:
        for q in o['wall']:
            grid[q] = o['color']
    output = grid.copy()
    for holes, color in expected_fills:
        for hole in holes:
            for q in hole:
                output[q] = color
    return {'input': grid, 'output': output}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def _digest(a):
    a = np.asarray(a, dtype=np.int8)
    return hashlib.sha256(bytes(a.shape) + a.tobytes()).hexdigest()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    rotations = list(map(int, rng.permutation(4)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            lesson = i if split == 'train' and i < 2 else 'test' if split == 'test' else 'ordinary'
            turn = rotations[i % 2] if split == 'train' else int(rng.choice(rotations))
            for _ in range(24):
                pair = _make_pair(rng, lesson, turn)
                key = _digest(pair['input'])
                info = _read(pair['input'])
                expected, _events = _execute(info)
                assert np.array_equal(expected, pair['output']), 'construction/read disagreement'
                order_witness = not np.array_equal(expected, _execute(info, 'raster')[0])
                closure_witness = not np.array_equal(expected, _execute(info, 'open_advances')[0])
                multi_witness = any(len(o['holes']) > 1 and e[3] and e[0] != e[4]
                                    for o, e in zip(info['objects'], _events))
                if (key not in seen and (lesson not in (0, 'test') or order_witness)
                        and (lesson != 0 or closure_witness and multi_witness)):
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded scene/uniqueness witness construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    info = _read(pair['input'])
    out, events = _execute(info)
    assert np.array_equal(out, _grid(pair['output'])), 'incorrect output'
    assert np.array_equal(info['grid'][info['grid'] != info['background']], out[info['grid'] != info['background']]), 'ink changed'
    assert np.any(out != info['grid'])
    objects, path = info['objects'], info['path']
    steps = [(q[0]-p[0], q[1]-p[1]) for p, q in zip(path, path[1:])]
    active = [o for o, e in zip(objects, events) if e[3]]
    inactive = [o for o, e in zip(objects, events) if o['holes'] and not e[3]]
    repeated_disagreement = any(a['color'] == b['color'] for a in active for b in inactive)
    irregular = any(len(hole) != (max(r for r, c in hole)-min(r for r, c in hole)+1)*(max(c for r, c in hole)-min(c for r, c in hole)+1) for o in active for hole in o['holes'])
    return {'side': info['side'], 'background': info['background'], 'wire': info['wire'],
            'shape': info['grid'].shape, 'path_length': len(path),
            'turns': sum(a != b for a, b in zip(steps, steps[1:])),
            'chambers': len(objects), 'active': len(active), 'inactive': len(inactive),
            'open': sum(not o['holes'] for o in objects), 'key_length': len(info['mapping']),
            'irregular_active': irregular, 'multi_active': any(len(o['holes']) > 1 for o in active),
            'multi_change': any(len(o['holes']) > 1 and e[3] and e[0] != e[4] for o, e in zip(objects, events)),
            'diagonal_active': any(len(_components(o['wall'])) > 1 for o in active),
            'self_active': any(e[3] and e[0] == e[4] for e in events),
            'terminal_active': any(e[3] and e[4] not in info['mapping'] for e in events),
            'color_change': any(e[3] and e[0] != e[4] for e in events),
            'repeated_disagreement': repeated_disagreement,
            'raster_diff': not np.array_equal(out, _execute(info, 'raster')[0]),
            'all_mapped_diff': not np.array_equal(out, _execute(info, 'all_mapped')[0]),
            'identity_diff': not np.array_equal(out, _execute(info, 'identity')[0]),
            'static_diff': not np.array_equal(out, _execute(info, 'static')[0]),
            'open_advance_diff': not np.array_equal(out, _execute(info, 'open_advances')[0])}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == b.shape
            key = _digest(a)
            assert key not in seen
            seen.add(key)
            f = _features(pair)
            assert f['active'] and f['inactive'] and f['open']
            features[split].append(f)
    tr, te = features['train'], features['test']
    for name in ('raster_diff', 'all_mapped_diff', 'identity_diff', 'static_diff',
                 'open_advance_diff', 'repeated_disagreement', 'self_active', 'irregular_active', 'multi_active', 'multi_change', 'diagonal_active'):
        assert any(f[name] for f in tr), 'missing demonstration: '+name
    assert all(f['raster_diff'] and f['all_mapped_diff'] and f['identity_diff'] and f['static_diff'] for f in te)
    assert min(f['chambers'] for f in te) > max(f['chambers'] for f in tr)
    assert len({f['side'] for f in tr}) >= 2


def _hand_regressions():
    """Small explicit semantic fixture; targets do not use _holes or _execute."""
    a = np.zeros((30, 27), dtype=np.int8)
    a[0, :5], a[1, :5] = [1, 2, 3, 4, 6], [2, 3, 1, 4, 7]
    wire = ([(r, 8) for r in range(3, 26)] + [(25, c) for c in range(9, 19)]
            + [(r, 18) for r in range(24, 4, -1)])
    for q in wire:
        a[q] = 5
    a[3, 8] = 1
    patterns = {'rect': ('111', '101', '111'), 'open': ('101', '101', '111'),
                'diamond': ('010', '101', '010'), 'profile': ('0100', '1010', '1001', '0110'),
                'multi': ('11111', '10101', '11111')}
    stations = [(6, 8, -1, 2, 'rect'), (11, 8, -1, 1, 'multi'),
                (16, 8, -1, 2, 'open'), (21, 8, -1, 1, 'rect'),
                (21, 18, 1, 2, 'diamond'), (16, 18, 1, 3, 'profile'),
                (11, 18, 1, 1, 'rect')]
    for r, c, side, color, kind in stations:
        rows = patterns[kind]
        a[r, c+side] = color
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                if value == '1':
                    a[r+i-len(rows)//2, c+side*(j+2)] = color
    expected = a.copy()
    for q, color in (((11, 5), 2), ((11, 3), 2), ((21, 21), 3),
                     ((15, 21), 1), ((16, 21), 1), ((16, 22), 1), ((11, 21), 2)):
        expected[q] = color
    assert np.array_equal(solve(a), expected)
    info = _read(a)
    for policy in ('raster', 'all_mapped', 'identity', 'static', 'open_advances'):
        assert not np.array_equal(expected, _execute(info, policy)[0]), policy
    # Make 3 -> 7 terminal. The profile fills 7; the final wall 1 now stays empty.
    terminal = a.copy()
    terminal[1, 2] = 7
    expected_terminal = terminal.copy()
    for q, color in (((11, 5), 2), ((11, 3), 2), ((21, 21), 3),
                     ((15, 21), 7), ((16, 21), 7), ((16, 22), 7)):
        expected_terminal[q] = color
    assert np.array_equal(solve(terminal), expected_terminal)
    # Branches, cycles, tied endpoints and disconnected fragments are invalid.
    for cells in ({(0, 0), (0, 1), (0, 2)}, {(0, 0), (1, 0), (1, 1), (2, 0)},
                  {(0, 0), (0, 1), (1, 0), (1, 1)}, {(0, 0), (1, 0), (3, 0)}):
        try:
            _chain(cells)
        except AssertionError:
            pass
        else:
            raise AssertionError('malformed thread accepted')


def self_test():
    """42 streaming seeds; semantic witnesses, topology, mutation and diversity."""
    _hand_regressions()
    signatures, shapes, structures, lengths, turns, backgrounds, wires, counts, keys, sides = [set() for _ in range(10)]
    totals = {k: 0 for k in ('pairs', 'raster_diff', 'all_mapped_diff', 'identity_diff', 'static_diff',
                             'open_advance_diff', 'repeated_disagreement', 'irregular_active',
                             'multi_active', 'diagonal_active', 'self_active')}
    totals['terminal_active'] = 0
    totals['multi_change'] = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_digest(p[k]) for s in ('train', 'test') for p in puzzle[s] for k in ('input', 'output')))
        if seed == 0:
            assert signatures.copy() == {tuple(_digest(p[k]) for s in ('train', 'test') for p in generate_puzzle(0)[s] for k in ('input', 'output'))}
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _features(pair)
                for collection, name in ((shapes, 'shape'), (lengths, 'path_length'), (turns, 'turns'),
                                         (backgrounds, 'background'), (wires, 'wire'), (counts, 'chambers'),
                                         (keys, 'key_length'), (sides, 'side')):
                    collection.add(f[name])
                role = {}
                norm = np.array([role.setdefault(int(c), len(role)) for c in pair['input'].flat], dtype=np.int8)
                structures.add(hashlib.sha256(bytes(pair['input'].shape)+norm.tobytes()).hexdigest())
                totals['pairs'] += 1
                for name in totals:
                    if name != 'pairs':
                        totals[name] += int(f[name])
                # Arbitrary palette permutation preserves this color-role rule.
                permutation = np.random.default_rng(seed+100).permutation(10)
                assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
    assert len(signatures) == 42 and len(structures) >= 200 and len(shapes) >= 30
    assert len(lengths) >= 30 and len(turns) >= 8
    assert backgrounds == wires == set(range(10)) and len(sides) == 4 and len(keys) >= 3
    assert {6, 7, 8, 9} <= counts
    assert totals['raster_diff'] >= 84 and totals['open_advance_diff'] >= 12
    assert totals['terminal_active'] >= 5
    for name in ('irregular_active', 'multi_active', 'diagonal_active', 'self_active', 'repeated_disagreement'):
        assert totals[name] >= 42, (name, totals[name])
    # The host exhausts every legal count combination. Here retain boundary
    # requests so self_test remains bounded when called by that same host.
    for a, b in ((2, 1), (10, 1), (10, 4), (5, 4)):
        p = generate_puzzle(711+a*7+b, a, b)
        assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    broken = generate_puzzle(9, 2, 1)
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    return {'seeds': 42, 'distinct_puzzles': len(signatures), 'input_shapes': len(shapes),
            'normalized_inputs': len(structures), 'path_lengths': len(lengths),
            'turn_counts': sorted(turns), 'chamber_counts': sorted(counts), 'key_lengths': sorted(keys),
            'backgrounds': sorted(backgrounds), 'wire_colors': sorted(wires), 'features': totals,
            'independent_semantic_review': False}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
