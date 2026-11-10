"""Beam-activated relay assembly. Only NumPy and the standard library are used.

Public solving is input-only. Generation scaffolds a route, scatters the relays,
then obtains the answer from the completed input. No scaffold enters solve().
"""
from __future__ import annotations

import numpy as np


NEIGHBORS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), 'invalid grid size')
    _require(np.issubdtype(a.dtype, np.integer), 'grid must contain integers')
    _require(bool(np.all((a >= 0) & (a <= 9))), 'invalid color')
    return a.astype(int, copy=True)


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _l(cells):
    cells = set(cells)
    _require(len(cells) == 3, 'an emitter must have three pixels')
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    _require(max(r for r, c in cells) == r0 + 1
             and max(c for r, c in cells) == c0 + 1, 'emitter is not a 2x2 L')
    missing = next(iter({(r0, c0), (r0, c0+1), (r0+1, c0), (r0+1, c0+1)} - cells))
    elbow = (2*r0 + 1 - missing[0], 2*c0 + 1 - missing[1])
    return elbow, (elbow[0] - missing[0], elbow[1] - missing[1])


def _source(elbow, direction, color):
    r, c = elbow
    dr, dc = direction
    return {'cells': {elbow: color, (r-dr, c): color, (r, c-dc): color},
            'elbow': elbow, 'direction': direction, 'color': color}


def _components(grid, bg):
    remaining = set(map(tuple, np.argwhere(grid != bg)))
    components = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        todo, cells = [first], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for dr, dc in NEIGHBORS:
                p = r+dr, c+dc
                if p in remaining:
                    remaining.remove(p)
                    todo.append(p)
        components.append({p: int(grid[p]) for p in cells})
    return components


def _connected(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    if not remaining:
        return False
    todo = [remaining.pop()]
    while todo:
        r, c = todo.pop()
        for dr, dc in NEIGHBORS:
            p = r+dr, c+dc
            if p in remaining:
                remaining.remove(p)
                todo.append(p)
    return not remaining


def _neck_turns(mask):
    turns = []
    for t in range(4):
        m = np.rot90(mask, t)
        h, w = m.shape
        if h < 2 or w < 2 or w % 2:
            continue
        neck = np.zeros(w, dtype=bool)
        neck[w//2-1:w//2+1] = True
        if np.array_equal(m[-1], neck):
            turns.append(t)
    return turns


def _part(array, bg):
    """Infer both colors and the unique neck from a complete composite object."""
    colors, counts = np.unique(array[array != bg], return_counts=True)
    _require(len(colors) == 2, 'a relay must have two foreground colors')
    small = np.flatnonzero(counts == 3)
    _require(len(small) == 1 and counts[1-int(small[0])] >= 4, 'relay colors lack body/emitter roles')
    emitter = int(colors[small[0]])
    body = int(colors[1-int(small[0])])
    _l(map(tuple, np.argwhere(array == emitter)))
    positions = np.argwhere(array == body)
    low, high = positions.min(axis=0), positions.max(axis=0)
    mask = array[low[0]:high[0]+1, low[1]:high[1]+1] == body
    _require(_connected(mask), 'body must be eight-connected')
    turns = _neck_turns(mask)
    _require(len(turns) == 1, 'relay has competing or missing attachment edges')
    turn = turns[0]
    canonical = np.rot90(array, turn).copy()
    bp = np.argwhere(canonical == body)
    lo, hi = bp.min(axis=0), bp.max(axis=0)
    base = canonical[lo[0]:hi[0]+1, lo[1]:hi[1]+1] == body
    ep = set(map(tuple, np.argwhere(canonical == emitter)))
    elbow, direction = _l(ep)
    _require(direction[0] == -1 and max(r for r, c in ep) < lo[0],
             'mounted emitter must face outward beyond the body')
    _require(_connected(canonical != bg), 'mounted emitter must touch the body')
    return {'array': canonical, 'body': body, 'emitter': emitter,
            'base': base, 'base_origin': tuple(map(int, lo)),
            'undo_turn': turn, 'input_array': array.copy()}


def _parse(value):
    grid = _grid(value)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(counts[bg] > grid.size//2, 'background must have a strict majority')
    roots, sockets, mirrors, parts = [], [], [], {}
    for cells in _components(grid, bg):
        colors = set(cells.values())
        if len(colors) == 2:
            r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
            r1, c1 = max(r for r, c in cells), max(c for r, c in cells)
            a = np.full((r1-r0+1, c1-c0+1), bg, dtype=int)
            for (r, c), color in cells.items():
                a[r-r0, c-c0] = color
            part = _part(a, bg)
            _require(part['body'] not in parts, 'duplicate relay body color')
            part['storage_cells'] = cells
            parts[part['body']] = part
            continue
        _require(len(colors) == 1, 'unrecognized composite object')
        color = next(iter(colors))
        rows, cols = {r for r, c in cells}, {c for r, c in cells}
        if len(cells) == 3 and len(rows) == len(cols) == 2:
            elbow, direction = _l(cells)
            roots.append({'cells': cells, 'elbow': elbow, 'direction': direction, 'color': color})
        else:
            _require(len(rows) == 1 or len(cols) == 1, 'fixed object is not straight')
            _require(len(cells) == max(rows)-min(rows)+max(cols)-min(cols)+1,
                     'fixed bar has a gap')
            orientation = 'h' if len(rows) == 1 else 'v'
            if len(cells) == 2:
                sockets.append({'cells': cells, 'color': color, 'orientation': orientation})
            else:
                _require(len(cells) % 2 == 1, 'mirrors must have odd length')
                mirrors.append({'cells': cells, 'color': color,
                                'orientation': 'p' if len(cells) == 1 else orientation,
                                'midpoint': ((min(rows)+max(rows))//2, (min(cols)+max(cols))//2)})
    _require(len(roots) == 1, 'exactly one isolated root emitter is required')
    _require(len({s['color'] for s in sockets}) == len(sockets), 'socket labels must be distinct')
    return {'grid': grid, 'shape': grid.shape, 'bg': bg, 'root': roots[0],
            'sockets': sockets, 'mirrors': mirrors, 'parts': parts}


def _dock(part, socket, direction, bg):
    """Dock on the far side, computing the launch from the transported L."""
    dr, dc = direction
    side = (2 if dr > 0 else 0) if socket['orientation'] == 'h' else (1 if dc > 0 else 3)
    targets = sorted((r + (dr if socket['orientation'] == 'h' else 0),
                      c + (dc if socket['orientation'] == 'v' else 0)) for r, c in socket['cells'])
    a = part['array']
    marker = np.zeros(a.shape, dtype=bool)
    r0, c0 = part['base_origin']
    h, w = part['base'].shape
    marker[r0+h-1, c0+w//2-1:c0+w//2+1] = True
    oriented = np.rot90(a, -side)
    neck = np.argwhere(np.rot90(marker, -side))
    shift = np.min(targets, axis=0) - neck.min(axis=0)
    _require(set(map(tuple, neck+shift)) == set(targets), 'attachment misalignment')
    cells = {(int(r+shift[0]), int(c+shift[1])): int(oriented[r, c])
             for r, c in np.argwhere(oriented != bg)}
    elbow, launch = _l(p for p, color in cells.items() if color == part['emitter'])
    return cells, elbow, launch, side


def _contact(mirror, p, direction):
    r, c = mirror['midpoint']
    dr, dc = direction
    if mirror['orientation'] in ('h', 'p') and p == (r-dr, c):
        return 0
    if mirror['orientation'] in ('v', 'p') and p == (r, c-dc):
        return 1
    return None


def _run(scene, *, assemble=True, reflect=True, resume=True):
    """Trace against fixed objects and the structures installed so far."""
    shape, bg = scene['shape'], scene['bg']
    answer = np.full(shape, bg, dtype=int)
    occupied = {}
    for obj in [scene['root']] + scene['sockets'] + scene['mirrors']:
        for p, color in obj['cells'].items():
            _require(_inside(p, shape) and p not in occupied, 'overlapping or out-of-bounds fixed object')
            occupied[p] = color
            answer[p] = color
    socket_at = {p: i for i, s in enumerate(scene['sockets']) for p in s['cells']}
    root = scene['root']
    position, direction, shade = root['elbow'], root['direction'], root['color']
    used, painted, visited = set(), {}, set()
    activations, contacts, flights, flight = [], [], [], []
    end = 'edge'
    while True:
        p = position[0]+direction[0], position[1]+direction[1]
        if not _inside(p, shape):
            flights.append(flight)
            break
        if p in socket_at:
            flights.append(flight)
            flight = []
            index = socket_at[p]
            socket = scene['sockets'][index]
            part = scene['parts'].get(socket['color'])
            if index in used or part is None or not assemble:
                end = 'socket'
                break
            cells, elbow, launch, side = _dock(part, socket, direction, bg)
            _require(all(_inside(q, shape) for q in cells), 'installation clips boundary')
            _require(not set(cells) & (set(occupied) | set(painted)), 'installation overlaps apparatus or beam')
            used.add(index)
            for q, color in cells.items():
                occupied[q] = color
                answer[q] = color
            activations.append({'index': index, 'body': socket['color'], 'side': side,
                                'turn': (part['undo_turn']-side) % 4, 'cells': cells,
                                'elbow': elbow, 'direction': launch, 'part': part})
            if not resume:
                end = 'socket'
                break
            position, direction, shade = elbow, launch, part['emitter']
            continue
        # Installed sockets change the machine. A shared state across two
        # different installation stages need not be an infinite loop.
        state = p, direction, tuple(sorted(used))
        _require(state not in visited, 'beam loop')
        visited.add(state)
        _require(p not in occupied, 'beam strikes an occupied object')
        hits = [(i, _contact(m, p, direction)) for i, m in enumerate(scene['mirrors'])]
        hits = [(i, axis) for i, axis in hits if axis is not None]
        _require(len(hits) <= 1, 'competing mirror contacts')
        if hits and reflect:
            i, axis = hits[0]
            incoming = shade
            shade = scene['mirrors'][i]['color']
            contacts.append({'index': i, 'axis': axis, 'position': p,
                             'incoming': incoming, 'color': shade})
            direction = (-direction[0], direction[1]) if axis == 0 else (direction[0], -direction[1])
            flights.append(flight)
            flight = []
        _require(p not in painted or painted[p] == shade, 'incompatible beam colors')
        painted[p] = shade
        answer[p] = shade
        flight.append((p, shade))
        position = p
    return answer, {'activations': activations, 'contacts': contacts, 'painted': painted,
                    'flights': flights, 'tail': flight, 'end': end, 'occupied': occupied}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete output from this input alone."""
    return _run(_parse(input_grid))[0]


def _mask(rng):
    for _ in range(60):
        w = int(rng.choice((4, 6, 8), p=(.55, .32, .13)))
        h = int(rng.integers(2, 6))
        stem = int(rng.integers(1, h))
        body = h-stem
        m = np.zeros((h, w), dtype=bool)
        mid = w//2
        m[:, mid-1:mid+1] = True
        kind = int(rng.integers(4))
        if kind == 0 and h >= w//2:
            m[:] = False
            wing = w//2-1
            for r in range(wing):
                m[r, r] = m[r, w-1-r] = True
            m[wing:, mid-1:mid+1] = True
        else:
            m[0] = True
            for r in range(1, body):
                if kind == 1:
                    m[r] = True
                else:
                    left = int(rng.integers(mid))
                    right = int(rng.integers(mid+1, w+1))
                    m[r, left:right] = True
                    if kind == 3:
                        for c in range(w):
                            if c not in (mid-1, mid) and rng.random() < .3:
                                m[r, c] = False
        if _connected(m) and _neck_turns(m) == [0]:
            return m
    return np.array([[1, 1, 1, 1], [0, 1, 1, 0]], dtype=bool)


def _make_part(rng, body, bg):
    m = _mask(rng)
    h, w = m.shape
    emitter = int(rng.choice([x for x in range(10) if x not in (body, bg)]))
    a = np.full((h+2, w), bg, dtype=int)
    a[2:][m] = body
    c, dc = (0, -1) if rng.integers(2) else (w-1, 1)
    for p in ((0, c), (1, c), (0, c-dc)):
        a[p] = emitter
    return _part(a, bg)


def _clear(cells, fixed):
    return all(max(abs(r-a), abs(c-b)) > 1 for r, c in cells for a, b in fixed)


def _fixed(scene):
    return {p for o in [scene['root']] + scene['sockets'] + scene['mirrors'] for p in o['cells']}


def _add_mirror(scene, rng, axis=None):
    _, info = _run(scene)
    tail = [p for p, color in info['tail']]
    options = [(p, a, n) for p in tail[1:-2] for a in ([axis] if axis is not None else (0, 1))
               for n in (1, 3, 5, 7)]
    rng.shuffle(options)
    if len(tail) < 4:
        return None
    # The final two ray pixels recover the final direction after all contacts.
    direction = (tail[-1][0]-tail[-2][0], tail[-1][1]-tail[-2][1])
    for p, a, n in options:
        mid = (p[0]+(direction[0] if a == 0 else 0), p[1]+(direction[1] if a == 1 else 0))
        radius = n//2
        coords = {(mid[0]+(k if a == 1 else 0), mid[1]+(k if a == 0 else 0))
                  for k in range(-radius, radius+1)}
        if not all(_inside(q, scene['shape']) for q in coords) or not _clear(coords, _fixed(scene)):
            continue
        shade = int(rng.choice([x for x in range(10) if x != scene['bg']]))
        mirror = {'cells': {q: shade for q in coords}, 'color': shade, 'midpoint': mid,
                  'orientation': 'p' if n == 1 else ('h' if a == 0 else 'v')}
        trial = dict(scene, mirrors=scene['mirrors']+[mirror])
        try:
            _, new = _run(trial)
        except ValueError:
            continue
        if len(new['activations']) == len(info['activations']) and len(new['contacts']) == len(info['contacts'])+1 and len(new['tail']) >= 4:
            return trial
    return None


def _add_socket(scene, rng, label, orientation, *, part=True, room=True):
    _, info = _run(scene)
    tail = [p for p, color in info['tail']]
    options = [(p, sign) for p in tail[1:] for sign in (-1, 1)]
    rng.shuffle(options)
    for attempt in range(12 if part else 1):
        module = _make_part(rng, label, scene['bg']) if part else None
        for p, sign in options:
            other = (p[0], p[1]+sign) if orientation == 'h' else (p[0]+sign, p[1])
            coords = {p, other}
            if not all(_inside(q, scene['shape']) for q in coords) or not _clear(coords, _fixed(scene)):
                continue
            socket = {'cells': {q: label for q in coords}, 'color': label, 'orientation': orientation}
            modules = dict(scene['parts'])
            if part:
                modules[label] = module
            trial = dict(scene, sockets=scene['sockets']+[socket], parts=modules)
            try:
                _, new = _run(trial)
            except ValueError:
                continue
            if len(new['activations']) != len(info['activations'])+int(part):
                continue
            if part and room and len(new['tail']) < 4:
                continue
            if not part and new['end'] != 'socket':
                continue
            return trial
    return None


def _add_inactive(scene, rng, label):
    _, before = _run(scene)
    module = _make_part(rng, label, scene['bg'])
    h, w = scene['shape']
    for _ in range(200):
        r, c = int(rng.integers(h-1)), int(rng.integers(w-1))
        orientation = 'h' if rng.integers(2) else 'v'
        coords = {(r, c), (r, c+1) if orientation == 'h' else (r+1, c)}
        if not _clear(coords, _fixed(scene)):
            continue
        socket = {'cells': {p: label for p in coords}, 'color': label, 'orientation': orientation}
        trial = dict(scene, sockets=scene['sockets']+[socket], parts=dict(scene['parts']))
        trial['parts'][label] = module
        try:
            _, after = _run(trial)
        except ValueError:
            continue
        if [a['body'] for a in before['activations']] == [a['body'] for a in after['activations']]:
            return trial
    return None


def _scatter(scene, rng, forced_turns):
    grid = np.full(scene['shape'], scene['bg'], dtype=int)
    blocked = np.zeros(scene['shape'], dtype=bool)
    for obj in [scene['root']] + scene['sockets'] + scene['mirrors']:
        for (r, c), color in obj['cells'].items():
            grid[r, c] = color
            blocked[max(0, r-1):r+2, max(0, c-1):c+2] = True
    _, info = _run(scene)
    side_by_color = {a['body']: a['side'] for a in info['activations']}
    parts = list(scene['parts'].values())
    rng.shuffle(parts)
    parts.sort(key=lambda p: -p['array'].size)
    for part in parts:
        label = part['body']
        turns = int(rng.integers(4))
        if label in forced_turns:
            turns = (-side_by_color[label]-forced_turns[label]) % 4
        a = np.rot90(part['array'], turns)
        ph, pw = a.shape
        local = np.argwhere(a != scene['bg'])
        if ph > grid.shape[0] or pw > grid.shape[1]:
            return None
        rr, cc = np.indices((grid.shape[0]-ph+1, grid.shape[1]-pw+1))
        rows, cols = rr.ravel()[:, None]+local[:, 0], cc.ravel()[:, None]+local[:, 1]
        good = np.flatnonzero(~blocked[rows, cols].any(axis=1))
        if not len(good):
            return None
        j = int(rng.choice(good))
        for k, (r, c) in enumerate(zip(rows[j], cols[j])):
            grid[r, c] = a[tuple(local[k])]
            blocked[max(0, r-1):r+2, max(0, c-1):c+2] = True
    if np.count_nonzero(grid == scene['bg']) <= grid.size//2:
        return None
    return grid


def _build(rng, nactive, *, teaching=False, absorb=False, anchor=None, decoy=None,
           forced_deltas=None, target_sides=None, previous_bg=None):
    for attempt in range(250):
        excluded = {x for x in (anchor, decoy, previous_bg) if x is not None}
        bg = int(rng.choice([x for x in range(10) if x not in excluded]))
        labels = list(map(int, rng.permutation([x for x in range(10) if x not in (bg, decoy)])))
        if anchor is not None:
            labels.remove(anchor)
            labels.insert(0, anchor)
        # Shorter extra demonstrations can use compact canvases. Long cascades
        # share the same size range as teaching scenes; successful geometry,
        # not a train/test size template, determines what fits.
        low = 18 if teaching or nactive == 3 else 16
        h, w = int(rng.integers(low, 31)), int(rng.integers(low, 31))
        r, c = int(rng.integers(1, h-1)), int(rng.integers(1, w-1))
        direction = DIRECTIONS[int(rng.integers(4))]
        shade = int(rng.choice([x for x in range(10) if x != bg]))
        scene = {'shape': (h, w), 'bg': bg, 'root': _source((r, c), direction, shade),
                 'sockets': [], 'mirrors': [], 'parts': {}}
        axes = list(map(int, rng.permutation(2)))
        socket_axes = ['h', 'v'] if rng.integers(2) else ['v', 'h']
        valid = True
        for i in range(nactive):
            if teaching or rng.random() < .45:
                trial = _add_mirror(scene, rng, axes[i] if teaching and i < 2 else None)
                if trial is None:
                    valid = False
                    break
                scene = trial
            orientation = socket_axes[i] if teaching else ('h' if rng.integers(2) else 'v')
            trial = _add_socket(scene, rng, labels[i], orientation)
            if trial is None:
                valid = False
                break
            scene = trial
        if not valid:
            continue
        if absorb:
            trial = _add_socket(scene, rng, labels[nactive], 'h' if rng.integers(2) else 'v', part=False)
            if trial is None:
                continue
            scene = trial
        next_label = nactive+int(absorb)
        ninactive = 1 if teaching else int(rng.integers(2))
        for i in range(ninactive):
            trial = _add_inactive(scene, rng, labels[next_label+i])
            if trial is None:
                valid = False
                break
            scene = trial
        if not valid:
            continue
        socket_colors = {s['color'] for s in scene['sockets']}
        absent = [x for x in range(10) if x != bg and x not in socket_colors]
        ndeco = 1 if teaching or decoy is not None else int(rng.integers(3))
        if len(absent) < ndeco:
            continue
        if decoy is not None:
            absent.remove(decoy)
            absent.insert(0, decoy)
        for color in absent[:ndeco]:
            scene['parts'][color] = _make_part(rng, color, bg)
        expected, info = _run(scene)
        if teaching and not any(x['incoming'] != x['color'] for x in info['contacts']):
            continue
        forced = {labels[i]: delta for i, delta in enumerate(forced_deltas or [])}
        for _ in range(6):
            grid = _scatter(scene, rng, forced)
            if grid is not None:
                break
        else:
            continue
        try:
            answer, parsed_info = _run(_parse(grid))
        except ValueError:
            continue
        if not np.array_equal(answer, expected):
            continue
        if target_sides is not None:
            sides = {a['side'] for a in parsed_info['activations']}
            rotations = [t for t in range(4) if {(s-t) % 4 for s in sides} == target_sides]
            if not rotations:
                continue
            t = int(rng.choice(rotations))
            grid, answer = np.rot90(grid, t).copy(), np.rot90(answer, t).copy()
        return {'input': grid, 'output': answer}
    raise RuntimeError('bounded relay construction exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, name, lo, hi in ((num_train, 'num_train', 2, 10), (num_test, 'num_test', 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    anchor = int(rng.integers(10))
    deltas = list(map(int, rng.permutation(4)))
    first = _build(rng, 2, teaching=True, absorb=True, anchor=anchor, forced_deltas=deltas[:2])
    first_scene = _parse(first['input'])
    _, first_info = _run(first_scene)
    complement = set(range(4)) - {a['side'] for a in first_info['activations']}
    second = _build(rng, 2, teaching=True, decoy=anchor, forced_deltas=deltas[2:],
                    target_sides=complement, previous_bg=first_scene['bg'])
    puzzle = {'train': [first, second], 'test': []}
    seen = {_key(p['input']) for p in puzzle['train']}
    for split, count in (('train', num_train), ('test', num_test)):
        while len(puzzle[split]) < count:
            for _ in range(10):
                n = int(rng.integers(1, 3)) if split == 'train' else 3
                pair = _build(rng, n, absorb=rng.random() < .3)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('unique-input construction exhausted')
            seen.add(key)
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(scene, info):
    labels = {s['color'] for s in scene['sockets']}
    active = {a['body'] for a in info['activations']}
    return {'sides': {a['side'] for a in info['activations']},
            'turns': {a['turn'] for a in info['activations']},
            'axes': {c['axis'] for c in info['contacts']},
            'active': len(active), 'inactive': len((set(scene['parts']) & labels)-active),
            'decoys': len(set(scene['parts'])-labels),
            'active_colors': active, 'decoy_colors': set(scene['parts'])-labels,
            'changed': any(c['incoming'] != c['color'] for c in info['contacts']),
            'end': info['end']}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'splits must be lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'invalid counts')
    seen, train_features = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'output shape changed')
            k = _key(inp)
            _require(k not in seen, 'duplicate input')
            seen.add(k)
            scene = _parse(inp)
            expected, info = _run(scene)
            _require(np.array_equal(expected, out), 'incorrect completed relay route')
            _require(set(np.unique(out)) <= set(np.unique(inp)), 'introduced invisible color')
            _require(not np.array_equal(inp, out), 'identity pair')
            for obj in [scene['root']] + scene['sockets'] + scene['mirrors']:
                _require(all(out[p] == color for p, color in obj['cells'].items()), 'fixed apparatus moved')
            f = _features(scene, info)
            if split == 'train':
                train_features.append(f)
                # Remove B's installation/transport operation but keep A's ray.
                no_b, _ = _run(scene, assemble=False)
                _require(not np.array_equal(no_b, out), 'assembly contribution has no effect')
                # Remove A's transported-L relaunch, keeping the first B docking.
                no_a, _ = _run(scene, resume=False)
                _require(not np.array_equal(no_a, out), 'transported L direction has no effect')
            else:
                _require(f['active'] == 3, 'test must extend the cascade to three installations')
    _require(set().union(*(f['sides'] for f in train_features)) == set(range(4)), 'missing docking direction')
    _require(set().union(*(f['turns'] for f in train_features)) == set(range(4)), 'missing turn witness')
    _require(set().union(*(f['axes'] for f in train_features)) == {0, 1}, 'missing mirror axis')
    _require(any(f['changed'] for f in train_features), 'missing contact recoloring')
    _require(sum(f['active'] == 2 for f in train_features) >= 2, 'missing two-relay demonstrations')
    _require(any(f['inactive'] and f['decoys'] for f in train_features), 'missing reachability/filter witnesses')
    _require({f['end'] for f in train_features} == {'edge', 'socket'}, 'missing termination evidence')
    active_colors = set().union(*(f['active_colors'] for f in train_features))
    decoy_colors = set().union(*(f['decoy_colors'] for f in train_features))
    _require(bool(active_colors & decoy_colors), 'missing locally varying color-role witness')


def _direct_regression():
    """A small fixture with explicitly enumerated placements and ray pixels.

    Unlike sampling, this expected answer uses no docking/tracing helper.
    """
    inp = np.zeros((24, 20), dtype=int)
    inp[0, 7] = inp[1, 6] = inp[1, 7] = 1
    inp[3:6, 11] = 2  # contact (4,10), column reflection
    inp[7, 7:9] = 4
    inp[14, 2:4] = 7
    inp[3, 1:3] = 5  # unvisited but matched
    first = np.array([[0, 0, 3, 3], [0, 0, 0, 3], [4, 4, 4, 4], [0, 4, 4, 0]])
    second = np.array([[8, 8, 0, 0], [8, 0, 0, 0], [7, 7, 7, 7], [0, 7, 7, 0]])
    unused = np.array([[0, 0, 6, 6], [0, 0, 0, 6], [5, 5, 5, 5], [0, 5, 5, 0]])
    decoy = np.array([[6, 6, 0, 0], [6, 0, 0, 0], [9, 9, 9, 9], [0, 9, 9, 0]])
    inp[0:4, 14:18] = first
    inp[10:14, 14:18] = second
    inp[18:22, 14:18] = unused
    inp[19:23, 0:4] = decoy
    expected = np.zeros_like(inp)
    expected[0, 7] = expected[1, 6] = expected[1, 7] = 1
    expected[3:6, 11] = 2
    expected[7, 7:9] = 4
    expected[14, 2:4] = 7
    expected[3, 1:3] = 5
    expected[8, 7:9] = 4
    expected[9, 6:10] = 4
    for p in ((10, 6), (11, 6), (11, 7)):
        expected[p] = 3
    expected[15, 2:4] = 7
    expected[16, 1:5] = 7
    for p in ((17, 4), (18, 3), (18, 4)):
        expected[p] = 8
    for p in ((2, 8), (3, 9)):
        expected[p] = 1
    for p in ((4, 10), (5, 9), (6, 8)):
        expected[p] = 2
    for p in ((12, 5), (13, 4)):
        expected[p] = 3
    for p in ((19, 5), (20, 6), (21, 7), (22, 8), (23, 9)):
        expected[p] = 8
    assert np.array_equal(solve(inp), expected)
    for t in range(4):
        assert np.array_equal(solve(np.rot90(inp, t)), np.rot90(expected, t))
    # A new bare socket on the final flight absorbs without relaunching.
    absorbed = inp.copy()
    absorbed[21, 7:9] = 9
    # Remove the decoy (it would otherwise become an eligible relay).
    absorbed[19:23, 0:4] = 0
    absorb_expected = expected.copy()
    absorb_expected[21, 7:9] = 9
    absorb_expected[22, 8] = absorb_expected[23, 9] = 0
    assert np.array_equal(solve(absorbed), absorb_expected)
    # Increasing body depth by one transports the emitter one row farther.
    # It now misses the second socket, despite unchanged labels and mirrors.
    deeper = inp.copy()
    deeper[0:5, 14:18] = np.array([[0, 0, 3, 3], [0, 0, 0, 3],
                                  [4, 4, 4, 4], [0, 4, 4, 0], [0, 4, 4, 0]])
    deeper_expected = expected.copy()
    obsolete = [(8, 7), (8, 8), (10, 6), (11, 6), (11, 7), (12, 5), (13, 4),
                (15, 2), (15, 3), (17, 4), (18, 3), (18, 4),
                (19, 5), (20, 6), (21, 7), (22, 8), (23, 9)]
    obsolete += [(9, c) for c in range(6, 10)] + [(16, c) for c in range(1, 5)]
    for p in obsolete:
        deeper_expected[p] = 0
    deeper_expected[8:10, 7:9] = 4
    deeper_expected[10, 6:10] = 4
    for p in ((11, 6), (12, 6), (12, 7), (13, 5), (14, 4), (15, 3), (16, 2), (17, 1), (18, 0)):
        deeper_expected[p] = 3
    assert np.array_equal(solve(deeper), deeper_expected)
    return inp, expected


def self_test():
    """Bounded geometric regressions and measured structural variation."""
    _direct_regression()
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures, dimensions, bases, counts, backgrounds, mirror_lengths = set(), set(), set(), set(), set(), set()
    directions, widths, depths, corners = set(), set(), set(), set()
    diagonal = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                scene = _parse(pair['input'])
                _, info = _run(scene)
                dimensions.add(scene['shape'])
                backgrounds.add(scene['bg'])
                counts.add((len(info['activations']), len(scene['sockets']), len(scene['parts']), len(scene['mirrors'])))
                directions.add(scene['root']['direction'])
                mirror_lengths.update(len(m['cells']) for m in scene['mirrors'])
                for p in scene['parts'].values():
                    m = p['base']
                    bases.add(_key(m))
                    depths.add(m.shape[0])
                    widths.add(m.shape[1])
                    elbow, direction = _l(map(tuple, np.argwhere(p['array'] == p['emitter'])))
                    corners.add(direction[1])
                    # Four-connectedness is measured separately from grouping.
                    rem = set(map(tuple, np.argwhere(m)))
                    stack = [rem.pop()]
                    while stack:
                        r, c = stack.pop()
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            q = r+dr, c+dc
                            if q in rem:
                                rem.remove(q)
                                stack.append(q)
                    diagonal += bool(rem)
    assert len(signatures) == 42
    assert len(dimensions) >= 40 and len(bases) >= 35 and len(counts) >= 12
    assert backgrounds == set(range(10)) and directions == set(DIRECTIONS)
    assert widths == {4, 6, 8} and depths == {2, 3, 4, 5} and corners == {-1, 1}
    assert mirror_lengths >= {1, 3, 5, 7} and diagonal >= 5
    first = puzzles[0]
    repeat = generate_puzzle(0)
    assert all(np.array_equal(a[k], b[k]) for s in ('train', 'test')
               for a, b in zip(first[s], repeat[s]) for k in ('input', 'output'))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000+nt*10+nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Whole-scene rotations and arbitrary bijective recoloring commute with solve.
    pair = first['test'][0]
    for t in range(4):
        assert np.array_equal(solve(np.rot90(pair['input'], t)), np.rot90(pair['output'], t))
    permutation = np.array([3, 9, 0, 5, 7, 1, 8, 2, 6, 4])
    assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in ('train', 'test')}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError('damaged answer accepted')
    return {'seeds': 42, 'distinct_puzzles': len(signatures), 'dimensions': len(dimensions),
            'body_silhouettes': len(bases), 'count_configurations': len(counts),
            'diagonal_bodies': diagonal}
