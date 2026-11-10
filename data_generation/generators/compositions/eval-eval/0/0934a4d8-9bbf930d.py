"""A hidden final-state window on a mirrored, directionally guided rail board.

Only standard-library modules and NumPy are used. Construction coordinates are
checked against the input reader but are never passed to solve().
"""
from __future__ import annotations

import json
import numpy as np

DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(np.all((0 <= a) & (a <= 9)), "ARC colors required")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _plus(p, d):
    return p[0] + d[0], p[1] + d[1]


def _inside(a, p):
    return 0 <= p[0] < a.shape[0] and 0 <= p[1] < a.shape[1]


def _roles(inp):
    candidates = []
    for k in range(4):
        g = np.rot90(inp, k)
        token = int(g[0, 0])
        if np.all(g[:, 0] == token) and np.count_nonzero(g == token) == g.shape[0]:
            candidates.append((g.copy(), k, token))
    _require(len(candidates) == 1, "one uniquely colored full-edge launch strip")
    g, k, token = candidates[0]
    _require(min(g.shape) >= 6, "mirror board too small")
    bg = int(g[0, 1])
    _require(bg != token and np.all(g[:, 1] == bg), "clear approach column")
    masks = []
    for color in np.unique(g):
        color = int(color)
        if color in (token, bg):
            continue
        rr, cc = np.nonzero(g == color)
        r, s, c, d = int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1
        if len(rr) >= 2 and len(rr) == (s-r)*(d-c):
            masks.append((color, (r, s, c, d)))
    _require(len(masks) == 1, "unique rectangular curtain color")
    mask, box = masks[0]
    _require(box[2] >= 2, "curtain cannot erase launch strip or clear approach")
    return g, k, token, bg, mask, box


def _recover(g, token, bg, mask):
    """Check visible static orbits first, then recover every covered cell."""
    board = g.copy()
    board[:, 0] = bg
    h, w = board.shape
    _require(token not in board, "tokens belong only to the mobile strip")
    # Every orbit has up to four members. No orbit may be erased completely.
    witness_pairs = [0, 0]
    for r in range((h+1) // 2):
        for c in range((w+1) // 2):
            orbit = ((r, c), (h-1-r, c), (r, w-1-c), (h-1-r, w-1-c))
            values = {int(board[p]) for p in orbit if board[p] != mask}
            _require(len(values) == 1, "erased orbit or inconsistent visible reflections")
            value = next(iter(values))
            for p in orbit:
                if board[p] == mask:
                    board[p] = value
    rr, cc = np.indices(g.shape)
    for axis in (0, 1):
        static = g.copy()
        static[:, 0] = bg
        mate = np.flip(static, axis)
        visible = (static != mask) & (mate != mask) & (static != bg)
        coord = rr if axis == 0 else cc
        visible &= coord != g.shape[axis]-1-coord
        witness_pairs[axis] = int(np.count_nonzero(visible))
    _require(min(witness_pairs) >= 8, "both axes need visible colored witnesses")
    _require(np.array_equal(board, board[::-1]) and np.array_equal(board, board[:, ::-1]),
             "completed static board must obey both reflections")
    _require(mask not in board, "curtain must disappear from static completion")
    return board


def _paired(g, p, d, bg, token, forbidden=()):
    n = (-d[1], d[0])
    a, b = _plus(p, n), _plus(p, (-n[0], -n[1]))
    if not (_inside(g, a) and _inside(g, b)):
        return False
    value = int(g[a])
    return value == int(g[b]) and value not in (bg, token) and value not in forbidden


def _trace(board, token, bg, *, ignore_matching=False, forbidden=()):
    """Directional movement, not flood filling; turns require local rail equality."""
    g = board.copy()
    g[:, 0] = token
    h, w = g.shape
    routes, inactive = [], []
    for row in range(1, h-1):
        selected = False
        for col in range(1, w):
            if g[row, col] != bg:
                break
            a, b = int(g[row-1, col]), int(g[row+1, col])
            if a not in (bg, token) + tuple(forbidden) and b not in (bg, token) + tuple(forbidden):
                if a == b or ignore_matching:
                    selected = True
                    entrance, guide = (row, col), a
                else:
                    inactive.append((row, col, a, b))
                break
        if not selected:
            continue
        p, direction = (row, 1), (0, 1)
        path, states, turns = [p], set(), 0
        while True:
            _require((p, direction) not in states, "route cycle")
            states.add((p, direction))
            nxt = _plus(p, direction)
            if not _inside(g, nxt):
                break
            if g[nxt] == bg:
                p = nxt
                path.append(p)
                continue
            choices = []
            for d in ((-direction[1], direction[0]), (direction[1], -direction[0])):
                q = _plus(p, d)
                if _inside(g, q) and g[q] == bg and _paired(g, q, d, bg, token, forbidden):
                    choices.append(d)
            _require(len(choices) <= 1, "ambiguous matching turn")
            if not choices:
                break
            direction = choices[0]
            turns += 1
        _require(len(path) == len(set(path)), "route revisits a cell")
        routes.append({"start": (row, 0), "end": p, "path": path,
                       "entrance": entrance, "guide": guide,
                       "turns": turns, "open": p[0] in (0, h-1) or p[1] == w-1})
    _require(len({r['end'] for r in routes}) == len(routes), "duplicate destinations")
    return routes, inactive


def _animate(board, token, bg, routes):
    out = board.copy()
    out[:, 0] = token
    for route in routes:
        out[route['start']] = bg
        out[route['end']] = token
    return out


def _route_geometry(a, route):
    path = route['path']
    directions = [(q[0]-p[0], q[1]-p[1]) for p, q in zip(path, path[1:])]
    support = [False] + [_paired(a['board'], p, d, a['bg'], a['token'])
                         for p, d in zip(path[1:], directions)]
    supported = [i for i, yes in enumerate(support) if yes]
    gaps = 0
    if supported:
        for i in range(max(2, supported[0]+1), min(len(path)-1, supported[-1])):
            if not support[i] and len(set(directions[i-2:i+1])) == 1:
                gaps += 1
    return {'gap_cells': gaps, 'tail': len(path)-1-max(supported, default=0)}


def _analysis(input_grid):
    inp = _grid(input_grid)
    g, rotation, token, bg, mask, box = _roles(inp)
    board = _recover(g, token, bg, mask)
    routes, inactive = _trace(board, token, bg)
    final = _animate(board, token, bg, routes)
    r, s, c, d = box
    output = np.rot90(final[r:s, c:d], -rotation).copy()
    return {"grid": g, "rotation": rotation, "token": token, "bg": bg,
            "mask": mask, "box": box, "board": board, "routes": routes,
            "inactive": inactive, "final": final, "output": output}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read only the input: no training, random seed or construction state."""
    return _analysis(input_grid)['output']


def _polyline(points):
    path = [points[0]]
    for q in points[1:]:
        p = path[-1]
        d = (int(np.sign(q[0]-p[0])), int(np.sign(q[1]-p[1])))
        if (d[0] == 0) == (d[1] == 0):
            return None
        while p != q:
            p = _plus(p, d)
            path.append(p)
    cells = set(path)
    if len(cells) != len(path):
        return None
    for i, p in enumerate(path):
        degree = sum(_plus(p, d) in cells for d in DIRS)
        if degree != (1 if i in (0, len(path)-1) else 2):
            return None
    return path


def _path(rng, m, n, bends, open_end):
    points = [(int(rng.integers(1, m-1)), 1)]
    d = (0, 1)
    for j in range(bends+1):
        p = points[-1]
        if j == bends and open_end:
            if d != (0, 1):
                return None
            q = (p[0], n-1)
            if q[1] - p[1] < 2:
                return None
        else:
            possible = []
            for length in range(2, max(m, n)):
                q = (p[0]+d[0]*length, p[1]+d[1]*length)
                if 1 <= q[0] <= m-2 and 3 <= q[1] <= n-3 and (j or q[1] >= 5):
                    possible.append(q)
            if not possible:
                return None
            q = possible[int(rng.integers(len(possible)))]
        points.append(q)
        d = (-d[1], d[0]) if rng.integers(2) else (d[1], -d[0])
    return _polyline(points)


def _reflect(base, shared_row=False, shared_col=False):
    upper = np.concatenate((base, base[:, -2::-1] if shared_col else base[:, ::-1]), axis=1)
    return np.concatenate((upper, upper[-2::-1] if shared_row else upper[::-1]), axis=0)


def _walls(path, m, n, color, margin):
    cells, walls = set(path), {}
    for p in path:
        for d in DIRS:
            q = _plus(p, d)
            if 0 <= q[0] < m and margin <= q[1] < n and q not in cells:
                walls[q] = color
    return walls


def _scene(rng, m, n, bg, token, palette, bent, open_route, extras, shared_row=False, shared_col=False):
    base = np.full((m, n), bg, dtype=np.int8)
    occupied, intended, entries = set(), {}, []

    def place(path, active, gap=False):
        nonlocal base, occupied
        if path is None or len(path) < 5 or any(base[p] != bg for p in path) or set(path) & occupied:
            return False
        color = int(rng.choice(palette))
        first_turn = next((a[1] for a, b in zip(path, path[1:]) if a[0] != b[0]), path[-1][1])
        margin = int(rng.integers(2, min(4, first_turn-2)+1))
        walls = _walls(path, m, n, color, margin)
        if not active:
            other = int(rng.choice([v for v in palette if v != color]))
            for q in walls:
                if q[0] > path[0][0]:
                    walls[q] = other
        else:
            ds = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:])]
            eligible = []
            for i in range(3, len(path)-2):
                if len(set(ds[max(0, i-2):i+2])) != 1 or path[i][1] < margin+2:
                    continue
                d = ds[i-1]
                normal = (-d[1], d[0])
                flanks = (_plus(path[i], normal), _plus(path[i], (-normal[0], -normal[1])))
                if all(q in walls for q in flanks):
                    eligible.append(flanks)
            if eligible and (gap or rng.random() < .45):
                flanks = eligible[int(rng.integers(len(eligible)))]
                for q in flanks:
                    walls.pop(q, None)
            # A cap blocks independently of its identity; turn witnesses remain.
            cap = _plus(path[-1], ds[-1])
            if cap in walls:
                walls[cap] = int(rng.choice(palette))
        if any(q in occupied or (base[q] != bg and base[q] != v) for q, v in walls.items()):
            return False
        candidate = base.copy()
        for q, v in walls.items():
            candidate[q] = v
        board = _reflect(candidate, shared_row, shared_col)
        height, width = board.shape
        wanted = dict(intended)
        if active:
            endpoint = path[-1] if path[-1][1] < n-1 else (path[-1][0], width-1)
            wanted[(path[0][0], 0)] = endpoint
            wanted[(height-1-path[0][0], 0)] = (height-1-endpoint[0], endpoint[1])
        try:
            routes, inactive = _trace(board, token, bg)
            if {r['start']: r['end'] for r in routes} != wanted:
                return False
        except AssertionError:
            return False
        base = candidate
        occupied |= set(path)
        intended.clear()
        intended.update(wanted)
        entries.append((path, active))
        return True

    # Allocate actual occupied cells, not fixed row bands or a path-template menu.
    for _ in range(100):
        bends = int(rng.integers(1, 4)) if bent else 0
        if place(_path(rng, m, n, bends, False), True, gap=True):
            break
    else:
        return None
    for _ in range(100):
        if place(_path(rng, m, n, 0, False), False):
            break
    else:
        return None
    if open_route:
        for _ in range(100):
            if place(_path(rng, m, n, 0, True), True, gap=True):
                break
        else:
            return None
    for _ in range(extras):
        for attempt in range(50):
            bends = int(rng.integers(0, 4)) if bent else 0
            if place(_path(rng, m, n, bends, False), bool(rng.random() < .75)):
                break
    return _reflect(base, shared_row, shared_col), entries


def _make_pair(rng, params, kind, seen):
    # kind 0 = readable straight/open demonstration; 1 = bent left window;
    # kind 2 = combinations at test; kind 3 = additional varied demonstrations.
    colors = params['colors'] if not params['mix_colors'] else tuple(map(int, rng.permutation(10)))
    bg, token, mask = colors[:3]
    palette = colors[3:3+int(rng.integers(2, 8))]
    for attempt in range(120):
        m = int(rng.integers(10 if kind == 0 else 12, 16))
        n = int(rng.integers(9 if kind == 0 else 11, 16))
        shared_row, shared_col = bool(rng.integers(2)), bool(rng.integers(2))
        bent = kind in (1, 2) or (kind == 3 and rng.random() < .6)
        open_route = kind in (0, 2) or (kind == 3 and rng.random() < .5)
        extras = int(rng.integers(0, 3)) if kind in (2, 3) else 0
        result = _scene(rng, m, n, bg, token, palette, bent, open_route, extras, shared_row, shared_col)
        if result is None:
            continue
        board, entries = result
        height, width = board.shape
        closed, inactive = entries[0][0], entries[1][0]
        rows = [p[0] for path in (closed, inactive) for p in path]
        r = max(0, min(rows)-int(rng.integers(1, 3)))
        s = min(m-int(shared_row), max(rows)+int(rng.integers(2, 4)))
        max_col = max(p[1] for path in (closed, inactive) for p in path)
        mode = 'top' if kind == 0 else ('left' if kind == 1 else str(rng.choice(['top', 'left', 'small'])))
        if mode == 'top':
            if kind == 0:
                r = 0
                s = m-int(shared_row)
            # Keep a visible rail prefix so the row reflection has actual
            # witnesses even when the curtain reaches the opposite edge.
            c, d = min(5, closed[-1][1]), width if kind == 0 else int(rng.integers(n+1, width+1))
        elif mode == 'left':
            c, d = min(int(rng.integers(2, 6)), closed[-1][1]), min(n-int(shared_col), max_col+int(rng.integers(2, 4)))
            s = int(rng.integers(m+1, height+1))
        else:
            # A small terminal/cap window sometimes has visible entrance rails.
            end = closed[-1]
            r, s = max(0, end[0]-int(rng.integers(0, 3))), min(m-int(shared_row), end[0]+int(rng.integers(1, 4)))
            c, d = max(2, end[1]-int(rng.integers(0, 3))), min(n-int(shared_col), end[1]+int(rng.integers(2, 4)))
        inp = board.copy()
        inp[:, 0] = token
        inp[r:s, c:d] = mask
        rotation = params['rotation']
        if params['orientation'] == 'mixed':
            rotation = int(rng.integers(4))
        elif params['orientation'] == 'held' and kind == 2:
            rotation = (rotation+int(rng.integers(1, 4))) % 4
        inp = np.rot90(inp, rotation).copy()
        if _key(inp) in seen:
            continue
        try:
            a = _analysis(inp)
            # Labels come from the actual unmasked constructed board, then are
            # checked against recovery from returned pixels.
            routes, _ = _trace(board, token, bg)
            final = _animate(board, token, bg, routes)
            answer = np.rot90(final[r:s, c:d], rotation).copy()
            if not np.array_equal(answer, a['output']):
                continue
            if not np.any(final[r:s, c:d] == token):
                continue
            if kind in (0, 1) and not _color_ablation_changes(a):
                continue
            if kind == 0 and not any(_route_geometry(a, x)['gap_cells'] for x in a['routes']):
                continue
            if kind == 0 and len({x['guide'] for x in a['routes']}) < 2:
                continue
            if not _no_recovery_changes(a):
                continue
        except AssertionError:
            continue
        return {'input': inp, 'output': answer}
    raise AssertionError('bounded rail-board construction exhausted')


def _crop(a, final):
    r, s, c, d = a['box']
    return final[r:s, c:d]


def _color_ablation_changes(a):
    try:
        wrong, _ = _trace(a['board'], a['token'], a['bg'], ignore_matching=True)
        wrong_final = _animate(a['board'], a['token'], a['bg'], wrong)
        return not np.array_equal(_crop(a, wrong_final), _crop(a, a['final']))
    except AssertionError:
        # Undefined shortcuts are not counted as differing solved examples.
        return False


def _no_recovery_changes(a):
    try:
        raw = a['grid'].copy()
        raw[:, 0] = a['bg']
        wrong, _ = _trace(raw, a['token'], a['bg'], forbidden=(a['mask'],))
        correct = {(r['start'], r['end']) for r in a['routes']}
        return {(r['start'], r['end']) for r in wrong} != correct
    except AssertionError:
        return True


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
        raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed) if seed >= 0 else int(seed) % (1 << 64))
    params = {'colors': tuple(map(int, rng.permutation(10))), 'mix_colors': bool(rng.integers(2)),
              'rotation': int(rng.integers(4)), 'orientation': str(rng.choice(['fixed', 'mixed', 'held']))}
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            kind = (i if i < 2 else 3) if split == 'train' else 2
            pair = _make_pair(rng, params, kind, seen)
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'legal counts')
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, 'distinct inputs')
            seen.add(_key(inp))
            a = _analysis(inp)
            _require(a['routes'] and a['inactive'], 'both matching and mismatched entrances')
            _require(np.array_equal(out, a['output']), 'incorrect hidden final snapshot')
            _require(a['mask'] not in out, 'curtain omitted from output')
            _require(np.any(out == a['token']), 'window observes an actual terminal')
            _require(_no_recovery_changes(a), 'reflection must affect routing, not just background copying')
            paths = [p for route in a['routes'] for p in route['path']]
            _require(len(paths) == len(set(paths)), 'active paths cannot intersect')
            _require(np.count_nonzero(a['final'] == a['token']) == inp.shape[a['rotation'] % 2], 'token conservation')
            rails = a['board'] != a['bg']
            _require(np.array_equal(a['board'][rails], a['final'][rails]), 'static rails preserved')
            original = a['board'].copy()
            original[:, 0] = a['token']
            _require(np.count_nonzero(original != a['final']) == 2*len(a['routes']), 'only starts and endpoints change')
            _require(not np.array_equal(_crop(a, a['board']), _crop(a, a['final'])), 'A-only restoration must fail')
            if split == 'train':
                training.append(a)
    _require(any(any(r['open'] for r in a['routes']) for a in training), 'demonstrate boundary exit')
    _require(any(any(r['turns'] for r in a['routes']) for a in training), 'demonstrate orthogonal bends')
    _require(any(any(_route_geometry(a, r)['gap_cells'] for r in a['routes']) for a in training),
             'demonstrate directional continuation through a real rail interruption')
    _require(all(any(not r['open'] for r in a['routes']) for a in training), 'demonstrate internal stops')
    _require(sum(_color_ablation_changes(a) for a in training) >= 2, 'two equality-discriminating demonstrations')
    _require(any(len({r['guide'] for r in a['routes']}) >= 2 for a in training),
             'show matching across multiple active guide identities in one example')
    _require(len({len(a['routes']) for a in training}) >= 2, 'training mover counts vary')
    _require(any(np.any((a['final'] == a['token']) != (a['final'][:, ::-1] == a['token'])) for a in training),
             'mobile final state breaks column reflection')
    # Both curtain directions have a demonstration, not a hidden test-only rule.
    _require(any(a['box'][3] > a['grid'].shape[1]//2 for a in training), 'demonstrate row-witness recovery')
    _require(any(a['box'][1] > a['grid'].shape[0]//2 and a['box'][3] <= a['grid'].shape[1]//2 for a in training),
             'demonstrate column-witness recovery')


def measurements(puzzle):
    records = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            a = _analysis(pair['input'])
            r, s, c, d = a['box']
            geometry = [_route_geometry(a, route) for route in a['routes']]
            records.append({'split': split, 'shape': list(a['grid'].shape),
                            'window': [s-r, d-c], 'rotation': a['rotation'],
                            'roles': [a['bg'], a['token'], a['mask']],
                            'guides': len(set(map(int, np.unique(a['board'])))-{a['bg']}),
                            'active': len(a['routes']), 'inactive': len(a['inactive']),
                            'active_guides': sorted({x['guide'] for x in a['routes']}),
                            'turns': [x['turns'] for x in a['routes']],
                            'lengths': [len(x['path']) for x in a['routes']],
                            'open': sum(x['open'] for x in a['routes']),
                            'gap_cells': sum(x['gap_cells'] for x in geometry),
                            'tails': [x['tail'] for x in geometry],
                            'cross_row': r < a['grid'].shape[0]//2 < s,
                            'cross_col': c < a['grid'].shape[1]//2 < d,
                            'color_ablation': _color_ablation_changes(a),
                            'recovery_ablation': _no_recovery_changes(a)})
    return records


def self_test():
    """Bounded 42-seed semantic and diversity checks; no external certification."""
    signatures, records = set(), []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signature = tuple(_key(pair[k]) for sp in ('train', 'test') for pair in p[sp] for k in ('input', 'output'))
        signatures.add(signature)
        records.extend(measurements(p))
        for sp in p:
            for pair in p[sp]:
                before = pair['input'].copy()
                _require(np.array_equal(solve(before), pair['output']), 'input-only solve')
                _require(np.array_equal(before, pair['input']), 'solve must not mutate input')
        if seed in (0, 17, 41):
            q = generate_puzzle(seed)
            _require(all(np.array_equal(a[k], b[k]) for sp in p for a, b in zip(p[sp], q[sp]) for k in a), 'reproducibility')
    _require(len(signatures) == 42, '42 distinct puzzles')
    _require(len({tuple(r['shape']) for r in records}) >= 20, 'board geometry diversity')
    _require(len({tuple(r['window']) for r in records}) >= 30, 'window geometry diversity')
    _require({r['rotation'] for r in records} == set(range(4)), 'all launch orientations')
    _require(len({r['active'] for r in records}) >= 3, 'mover-count diversity')
    _require({0, 1, 2} <= {t for r in records for t in r['turns']}, 'bend geometry diversity')
    _require(any(r['gap_cells'] for r in records), 'actual interrupted-guide diversity')
    _require(len({r['guides'] for r in records}) >= 3, 'actual guide palette diversity')
    _require(any(any(n % 2 for n in r['shape']) for r in records) and
             any(all(n % 2 == 0 for n in r['shape']) for r in records), 'through-cell and between-cell seams')
    _require(all(r['recovery_ablation'] for r in records), 'A affects routing in every pair')
    _require(sum(r['color_ablation'] for r in records) >= 84, 'B equality is observable')
    _require(any(r['cross_row'] for r in records) and any(r['cross_col'] for r in records), 'both seam placements')
    _require(len({tuple(r['roles']) for r in records}) >= 30, 'semantic-role variation')
    # Color bijections and rotations are general family members, not seed modes.
    p = generate_puzzle(19, 2, 1)
    perm = np.array([7, 1, 9, 3, 0, 4, 6, 2, 8, 5], dtype=np.int8)
    transformed = {sp: [{k: perm[np.rot90(v)] for k, v in pair.items()} for pair in p[sp]] for sp in p}
    validate_generated_puzzle(transformed)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(800+nt*10+nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = {sp: [{k: v.copy() for k, v in pair.items()} for pair in p[sp]] for sp in p}
    bad['test'][0]['output'].flat[0] = (int(bad['test'][0]['output'].flat[0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('incorrect label accepted')
    a = _analysis(p['train'][0]['input'])
    damaged = a['grid'].copy()
    choices = np.argwhere((damaged != a['bg']) & (damaged != a['token']) & (damaged != a['mask']))
    point = tuple(map(int, choices[0]))
    alternatives = [c for c in range(10) if c not in (a['bg'], a['token'], a['mask'], int(damaged[point]))]
    damaged[point] = alternatives[0]
    try:
        _recover(damaged, a['token'], a['bg'], a['mask'])
    except AssertionError:
        pass
    else:
        raise AssertionError('conflicting visible orbit accepted')
    # A central 2x2 curtain erases a complete static orbit and must be rejected.
    broken = a['board'].copy()
    broken[:, 0] = a['token']
    h, w = broken.shape
    broken[h//2-1:h//2+1, w//2-1:w//2+1] = a['mask']
    try:
        solve(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('fully erased orbit accepted')
    return {'seeds': 42, 'pairs': len(records), 'board_shapes': len({tuple(r['shape']) for r in records}),
            'window_shapes': len({tuple(r['window']) for r in records}),
            'mover_counts': sorted({r['active'] for r in records}),
            'turn_counts': sorted({t for r in records for t in r['turns']}),
            'color_ablation_failures': sum(r['color_ablation'] for r in records)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
