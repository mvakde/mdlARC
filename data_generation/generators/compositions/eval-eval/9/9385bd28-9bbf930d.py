"""Paint packet races: matched rail routes schedule overlapping corner stamps.

Only NumPy and the standard library are used. No parent code or data is loaded.
The solver reads roles, corner stencils, payloads, and travel times from its input.
"""
from __future__ import annotations

import numpy as np


DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _need(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "grid dimensions")
    _need(np.issubdtype(g.dtype, np.integer), "integer grid required")
    _need(np.all((0 <= g) & (g <= 9)), "grid colors")
    return g.astype(np.int8, copy=False)


def _signature(g):
    return g.shape, g.astype(np.int8).tobytes()


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _inside(g, p):
    return 0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1]


def _corners(box, diagonal):
    r, c, s, t = box
    if diagonal == 0:
        return {(r, c), (r, c + 1), (r + 1, c),
                (s, t), (s, t - 1), (s - 1, t)}
    return {(r, t), (r, t - 1), (r + 1, t),
            (s, c), (s, c + 1), (s - 1, c)}


def _covers(box, p):
    return box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]


def _overlap(a, b):
    return max(a[0], b[0]) <= min(a[2], b[2]) and max(a[1], b[1]) <= min(a[3], b[3])


def _components(g, color):
    todo = {tuple(map(int, p)) for p in np.argwhere(g == color)}
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, cells = [start], {start}
        while stack:
            p = stack.pop()
            for d in DIRS:
                q = _add(p, d)
                if q in todo:
                    todo.remove(q)
                    cells.add(q)
                    stack.append(q)
        result.append(cells)
    return result


def _normalise(inp):
    candidates = []
    for k in range(4):
        g = np.rot90(inp, k)
        spine = int(g[0, 0])
        if np.all(g[:, 0] == spine) and np.count_nonzero(g == spine) == g.shape[0]:
            candidates.append((g.copy(), k, spine))
    _need(len(candidates) == 1, "one unique full-edge launcher spine required")
    g, rotation, spine = candidates[0]
    _need(g.shape[1] >= 8 and g.shape[0] >= 6, "launcher and scene must fit")
    bg = int(g[0, 1])
    _need(bg != spine and g[-1, 1] == bg and np.all(g[:, 2] == bg), "clear approach column")
    return g, rotation, spine, bg


def _read_stencils(g, bg, spine):
    """Isolated L triplets distinguish stencils from long, possibly gapped rails."""
    stencils = {}
    for color in sorted(set(map(int, np.unique(g))) - {bg, spine}):
        ls = []
        for cells in _components(g, color):
            if len(cells) == 3:
                rr, cc = zip(*cells)
                if max(rr) - min(rr) == max(cc) - min(cc) == 1:
                    ls.append(cells)
        if not ls:
            continue
        _need(len(ls) == 2, "a cue must have exactly two isolated L triplets")
        cells = ls[0] | ls[1]
        rr, cc = zip(*cells)
        box = (min(rr), min(cc), max(rr), max(cc))
        h, w = box[2] - box[0] + 1, box[3] - box[1] + 1
        _need(min(h, w) >= 2 and (min(h, w) > 2 or max(h, w) >= 4), "separated corners")
        _need(box[1] >= 4, "stencils cannot cover the launcher margin")
        # Recover the opposite-corner arrangement as local patterns.
        local = {(r - box[0], c - box[1]) for r, c in cells}
        patterns = (
            {(0, 0), (0, 1), (1, 0), (h - 1, w - 1), (h - 1, w - 2), (h - 2, w - 1)},
            {(0, w - 1), (0, w - 2), (1, w - 1), (h - 1, 0), (h - 1, 1), (h - 2, 0)},
        )
        _need(local in patterns, "L fragments must occupy opposite corners")
        stencils[color] = dict(cue=color, box=box, markers=cells, diagonal=patterns.index(local))
    _need(2 <= len(stencils) <= 5, "two through five stencil identities required")
    return stencils


def _trace(g, bg, cues, markers):
    """Directional inertia, first-opening selection, and locally forced turns."""
    def rail(p):
        if not _inside(g, p) or p in markers:
            return None
        v = int(g[p])
        return v if v in cues else None

    def paired(p, d):
        n = (-d[1], d[0])
        a, b = rail(_add(p, n)), rail(_add(p, (-n[0], -n[1])))
        return a is not None and a == b

    events, inactive = [], []
    for row in range(1, g.shape[0] - 1):
        if g[row, 1] == bg:
            continue
        source, entrance, unequal = (row, 1), None, False
        for col in range(2, g.shape[1]):
            if g[row, col] != bg:
                break
            a, b = rail((row - 1, col)), rail((row + 1, col))
            if a is not None and b is not None:
                if a == b:
                    entrance = (col, a)
                else:
                    unequal = True
                break
        if entrance is None:
            inactive.append(dict(source=source, mismatch=unequal))
            continue
        p, d = source, (0, 1)
        path, seen, turns, support = [p], set(), 0, []
        while True:
            _need((p, d) not in seen, "cyclic route")
            seen.add((p, d))
            q = _add(p, d)
            if not _inside(g, q):
                break
            if g[q] == bg:
                p = q
                path.append(p)
                support.append((d, paired(p, d)))
                continue
            choices = []
            for turn in ((-d[1], d[0]), (d[1], -d[0])):
                q = _add(p, turn)
                if _inside(g, q) and g[q] == bg and paired(q, turn):
                    choices.append(turn)
            _need(len(choices) <= 1, "ambiguous turn")
            if not choices:
                break
            d = choices[0]
            turns += 1
        _need(len(path) == len(set(path)) and len(path) > 2, "simple nonempty route")
        _need(g[p] == bg and p[1] >= 3, "free terminal outside launcher")
        supported = [i for i, (_, yes) in enumerate(support) if yes]
        gap = any(not support[i][1] and support[i - 1][0] == support[i][0] == support[i + 1][0]
                  for i in range(min(supported) + 1, max(supported))) if supported else False
        events.append(dict(source=source, cue=entrance[1], paint=int(g[source]),
                           path=path, time=len(path) - 1, destination=p, turns=turns,
                           gap=gap, tail=bool(support and not support[-1][1]),
                           open=p[0] in (0, g.shape[0] - 1) or p[1] == g.shape[1] - 1))
    _need(len({e['destination'] for e in events}) == len(events), "conflicting terminals")
    cells = [p for e in events for p in e['path']]
    _need(len(cells) == len(set(cells)), "generated routes must be disjoint")
    return events, inactive


def _analyse(inp):
    g, rotation, spine, bg = _normalise(_grid(inp))
    stencils = _read_stencils(g, bg, spine)
    cues = set(stencils)
    markers = set().union(*(o['markers'] for o in stencils.values()))
    payloads = set(map(int, g[1:-1, 1])) - {bg}
    _need(payloads and not payloads & (cues | {spine}), "payload roles must be distinct")
    _need(set(map(int, np.unique(g))) <= cues | payloads | {spine, bg}, "unidentified role")
    for color in payloads:
        _need(np.all(np.argwhere(g == color)[:, 1] == 1), "input payloads occur only on launch cells")
    events, inactive = _trace(g, bg, cues, markers)
    return dict(grid=g, rotation=rotation, spine=spine, bg=bg, stencils=stencils,
                events=events, inactive=inactive)


def _order(event):
    # A simultaneous pair retains A's top-first convention, in launcher orientation.
    return event['time'], -event['source'][0]


def _compose(g, bg, stencils, events, order=_order):
    """Generation compositor: stamp complete opaque layers chronologically."""
    out = g.copy()
    for event in events:
        out[event['source']] = bg
    for event in sorted(events, key=order):
        obj = stencils[event['cue']]
        r, c, s, t = obj['box']
        out[r:s + 1, c:t + 1] = event['paint']
        for p in obj['markers']:
            out[p] = obj['cue']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read only this grid. At each cell choose its latest covering delivery."""
    a = _analyse(input_grid)
    out = a['grid'].copy()
    for event in a['events']:
        out[event['source']] = a['bg']
    for r in range(out.shape[0]):
        for c in range(4, out.shape[1]):
            covering = [e for e in a['events'] if _covers(a['stencils'][e['cue']]['box'], (r, c))]
            if covering:
                winner = max(covering, key=_order)
                obj = a['stencils'][winner['cue']]
                out[r, c] = obj['cue'] if (r, c) in obj['markers'] else winner['paint']
    return np.rot90(out, -a['rotation']).copy()


def _features(a):
    g, bg, stencils, events = a['grid'], a['bg'], a['stencils'], a['events']
    out = _compose(g, bg, stencils, events)
    f = set()
    if events:
        f.add('matched')
    if any(i['mismatch'] for i in a['inactive']):
        f.add('mismatch')
    for name in ('gap', 'tail', 'open'):
        if any(e[name] for e in events):
            f.add(name)
    if any(not e['open'] for e in events):
        f.add('closed')
    if any(e['turns'] for e in events):
        f.add('turn')
    if not np.array_equal(out, _compose(g, bg, stencils, events, lambda e: (0, -e['source'][0]))):
        f.add('time_priority')
    # Require a timing witness between different rectangles, beyond repainting one cue.
    for r in range(g.shape[0]):
        for c in range(4, g.shape[1]):
            stack = [e for e in events if _covers(stencils[e['cue']]['box'], (r, c))]
            if not stack:
                continue
            timed = max(stack, key=_order)
            positional = min(stack, key=lambda e: e['source'][0])
            if (timed['cue'] != positional['cue'] and timed['paint'] != positional['paint']
                    and (r, c) not in stencils[timed['cue']]['markers']
                    and (r, c) not in stencils[positional['cue']]['markers']):
                f.add('distance_overlap')
    if not np.array_equal(out, _compose(g, bg, stencils, events, lambda e: (e['time'], e['source'][0]))):
        f.add('tie_priority')
    first = {}
    for e in sorted(events, key=_order):
        first.setdefault(e['cue'], e)
    if not np.array_equal(out, _compose(g, bg, stencils, list(first.values()))):
        f.add('repaint')
    for cue, obj in stencils.items():
        if not any(e['cue'] == cue for e in events):
            r, c, s, t = obj['box']
            if any(g[y, x] == bg and out[y, x] == bg and
                   not any(_covers(stencils[e['cue']]['box'], (y, x)) for e in events)
                   for y in range(r, s + 1) for x in range(c, t + 1)):
                f.add('unserved')
        else:
            if any(out[p] == cue for p in obj['markers']):
                f.add('retention')
            if any(out[p] != cue for p in obj['markers']):
                f.add('occlusion')
    return f


def _polyline(points):
    path = [points[0]]
    for target in points[1:]:
        p = path[-1]
        d = (int(np.sign(target[0] - p[0])), int(np.sign(target[1] - p[1])))
        if (d[0] == 0) == (d[1] == 0):
            return None
        while p != target:
            p = _add(p, d)
            path.append(p)
    cells = set(path)
    if len(cells) != len(path):
        return None
    for i, p in enumerate(path):
        degree = sum(_add(p, d) in cells for d in DIRS)
        if degree != (1 if i in (0, len(path) - 1) else 2):
            return None
    return path


def _random_path(rng, h, w, bends, closed):
    points, direction = [(int(rng.integers(1, h - 1)), 1)], (0, 1)
    for j in range(bends + 1):
        p = points[-1]
        if j == bends and not closed and direction != (0, -1):
            target = ((h - 1 if direction[0] > 0 else 0), p[1]) if direction[0] else (p[0], w - 1)
        else:
            choices = []
            for length in range(3, max(h, w)):
                q = (p[0] + length * direction[0], p[1] + length * direction[1])
                if 1 <= q[0] < h - 1 and 5 <= q[1] < w - 1 and (j > 0 or q[1] >= 7):
                    choices.append(q)
            if not choices:
                return None
            target = choices[int(rng.integers(len(choices)))]
        points.append(target)
        direction = ((-direction[1], direction[0]) if rng.integers(2)
                     else (direction[1], -direction[0]))
    return _polyline(points)


def _walls(rng, path, h, w, cue, cues, inactive=False, gap=False):
    cells, walls = set(path), {}
    for p in path:
        for d in DIRS:
            q = _add(p, d)
            if 0 <= q[0] < h and 3 <= q[1] < w and q not in cells:
                walls[q] = cue
    if inactive:
        other = int(rng.choice([c for c in cues if c != cue]))
        for q in walls:
            if q[0] > path[0][0]:
                walls[q] = other
    elif gap:
        directions = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
        options = [i for i in range(5, len(path) - 4)
                   if len(set(directions[i - 2:i + 2])) == 1 and path[i][1] >= 5]
        if options:
            i = int(rng.choice(options))
            d = directions[i]
            n = (-d[1], d[0])
            for q in (_add(path[i], n), _add(path[i], (-n[0], -n[1]))):
                walls.pop(q, None)
    # Unsupported open tails; internal caps keep their blocker but may lose flanks.
    direction = (path[-1][0] - path[-2][0], path[-1][1] - path[-2][1])
    n = (-direction[1], direction[0])
    for q in (_add(path[-1], n), _add(path[-1], (-n[0], -n[1]))):
        walls.pop(q, None)
    cap = _add(path[-1], direction)
    if cap in walls:
        walls[cap] = int(rng.choice(cues))
    return walls


def _circuit(rng, h, w, bg, spine, cues, paints, kind):
    g = np.full((h, w), bg, dtype=np.int8)
    g[:, 0] = spine
    intended, all_paths = {}, []

    def place(path, cue, paint, inactive=False, gap=False):
        nonlocal g, intended
        if path is None or len(path) < 7:
            return False
        reserved = set().union(*all_paths) if all_paths else set()
        if set(path) & reserved or any(g[p] != bg for p in path):
            return False
        walls = _walls(rng, path, h, w, cue, cues, inactive, gap)
        if any(q in reserved or (g[q] != bg and g[q] != color) for q, color in walls.items()):
            return False
        candidate = g.copy()
        for p, color in walls.items():
            candidate[p] = color
        candidate[path[0]] = paint
        wanted = dict(intended)
        if not inactive:
            wanted[path[0]] = path
        try:
            events, inactive_routes = _trace(candidate, bg, set(cues), set())
        except AssertionError:
            return False
        if {e['source']: e['path'] for e in events} != wanted:
            return False
        if inactive and not any(i['source'] == path[0] and i['mismatch'] for i in inactive_routes):
            return False
        g, intended = candidate, wanted
        all_paths.append(set(path))
        return True

    if kind == 'tie':
        top = int(rng.integers(0, h - 11))
        rows = [top + 1, top + 5, top + 9]
        rng.shuffle(rows)
        for i, row in enumerate(rows):
            path = _polyline([(row, 1), (row, w - 1)])
            if not place(path, cues[min(i, 1)], paints[i % 2], inactive=i == 2, gap=i == 0):
                return None
    else:
        bends = int(rng.integers(1, 4))
        for _ in range(90):
            path = _random_path(rng, h, w, bends, closed=True)
            if place(path, cues[0], paints[0], gap=True):
                break
        else:
            return None
        active_count = 3 if kind == 'time' else int(rng.integers(4, 6))
        assigned = [cues[0], cues[1]]
        if kind != 'time':
            assigned.append(cues[2])
        assigned += [int(rng.choice(cues)) for _ in range(active_count - 1 - len(assigned))]
        for index, cue in enumerate(assigned):
            placed = False
            for row in map(int, rng.permutation(np.arange(1, h - 1))):
                end = w - 1 if index == 0 or rng.random() < .55 else int(rng.integers(7, w - 1))
                path = _polyline([(row, 1), (row, end)])
                if place(path, cue, paints[(index + 1) % 2], gap=rng.random() < .35):
                    placed = True
                    break
            if not placed:
                return None
        mismatch_count = 1 if kind == 'time' else int(rng.integers(1, 3))
        for _ in range(mismatch_count):
            placed = False
            for row in map(int, rng.permutation(np.arange(1, h - 1))):
                end = int(rng.integers(7, w))
                if place(_polyline([(row, 1), (row, end)]), int(rng.choice(cues)),
                         int(rng.choice(paints)), inactive=True):
                    placed = True
                    break
            if not placed:
                return None
    events, _ = _trace(g, bg, set(cues), set())
    # Contrasting winners make rectangle priority readable, independent of cue literals.
    zero = max((e for e in events if e['cue'] == cues[0]), key=_order)
    for e in events:
        if e['cue'] == cues[1]:
            g[e['source']] = paints[1] if zero['paint'] == paints[0] else paints[0]
    events, inactive = _trace(g, bg, set(cues), set())
    return g, events, inactive, set().union(*all_paths)


def _sample_box(rng, h, w):
    style = int(rng.integers(5))
    if style == 0:
        hh, ww = int(rng.integers(2, 5)), int(rng.integers(4, w - 3))
    elif style == 1:
        hh, ww = int(rng.integers(4, h + 1)), int(rng.integers(2, 5))
    elif style == 2:
        hh = ww = int(rng.integers(3, min(h, w - 4, 9) + 1))
    elif style == 3:
        hh, ww = int(rng.integers(3, min(h, 8) + 1)), int(rng.integers(3, min(w - 4, 8) + 1))
    else:
        hh, ww = int(rng.integers(3, h + 1)), int(rng.integers(3, w - 3))
    r, c = int(rng.integers(h - hh + 1)), int(rng.integers(4, w - ww + 1))
    return r, c, r + hh - 1, c + ww - 1


def _candidate(rng, kind, count, rotation, required):
    if kind == 'tie':
        h, w = int(rng.integers(18, 27)), int(rng.integers(14, 25))
    elif kind == 'time':
        h, w = int(rng.integers(20, 30)), int(rng.integers(16, 29))
    else:
        h, w = int(rng.integers(22, 31)), int(rng.integers(20, 31))
    colors = list(map(int, rng.permutation(10)))
    bg, spine, cues, paints = colors[0], colors[1], colors[2:2 + count], colors[2 + count:4 + count]
    circuit = _circuit(rng, h, w, bg, spine, cues, paints, kind)
    if circuit is None:
        return None
    g, events, inactive, paths = circuit
    forbidden = set(paths)
    for p in map(tuple, np.argwhere(g != bg)):
        forbidden.add(p)
        forbidden.update(_add(p, d) for d in DIRS)
    stencils = {}
    for i, cue in enumerate(cues):
        for _ in range(500):
            box, diagonal = _sample_box(rng, h, w), int(rng.integers(2))
            cells = _corners(box, diagonal)
            if cells & forbidden or (i == 1 and not _overlap(box, stencils[cues[0]]['box'])):
                continue
            if kind == 'combined' and i == 2 and not any(_overlap(box, o['box']) for o in stencils.values()):
                continue
            stencils[cue] = dict(cue=cue, box=box, markers=cells, diagonal=diagonal)
            for p in cells:
                g[p] = cue
                forbidden.add(p)
                forbidden.update(_add(p, d) for d in DIRS)
            break
        else:
            return None
    try:
        a = _analyse(g)
        if any(a['stencils'][cue]['markers'] != stencils[cue]['markers'] for cue in cues):
            return None
        if {e['source']: e['path'] for e in a['events']} != {e['source']: e['path'] for e in events}:
            return None
        if not required <= _features(a):
            return None
        output = _compose(g, bg, stencils, events)
        if not np.array_equal(solve(g), output):
            return None
    except AssertionError:
        return None
    return dict(input=np.rot90(g, rotation).copy(), output=np.rot90(output, rotation).copy())


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    rng = np.random.default_rng(seed)
    orientation = int(rng.integers(4))
    mixed_orientation = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            if split == 'test':
                kind, stencil_count = 'combined', int(rng.integers(4, 6))
                required = {'matched', 'mismatch', 'turn'}
            elif index == 0:
                kind, stencil_count = 'tie', 2
                required = {'matched', 'mismatch', 'gap', 'tail', 'open', 'tie_priority', 'occlusion', 'retention'}
            elif index == 1:
                kind, stencil_count = 'time', 3
                required = {'matched', 'mismatch', 'turn', 'closed', 'time_priority', 'distance_overlap',
                            'repaint', 'unserved', 'retention'}
            else:
                kind = 'tie' if rng.random() < .5 else 'time'
                stencil_count = 2 if kind == 'tie' else 3
                required = {'matched', 'mismatch'}
            rotation = (orientation + index) % 4 if mixed_orientation else orientation
            for _ in range(400):
                pair = _candidate(rng, kind, stencil_count, rotation, required)
                if pair is not None and _signature(pair['input']) not in seen:
                    seen.add(_signature(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded spatial packing exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'split lists')
    n, m = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= n <= 10 and 1 <= m <= 4 and n > m, 'pair counts')
    seen, features, train_counts = set(), set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(inp.shape == out.shape, 'shape preservation')
            sig = _signature(inp)
            _need(sig not in seen, 'duplicate input')
            seen.add(sig)
            a = _analyse(inp)
            _need(a['events'] and any(i['mismatch'] for i in a['inactive']), 'both selections required')
            _need(np.array_equal(solve(inp), out), 'input-derived answer differs')
            canonical_out = np.rot90(out, a['rotation'])
            stamps = np.zeros(a['grid'].shape, dtype=bool)
            for e in a['events']:
                r, c, s, t = a['stencils'][e['cue']]['box']
                stamps[r:s + 1, c:t + 1] = True
                _need(canonical_out[e['source']] == a['bg'], 'delivered source must be consumed')
            for inactive in a['inactive']:
                p = inactive['source']
                _need(canonical_out[p] == a['grid'][p], 'inactive packet must remain')
            untouched = ~stamps
            for e in a['events']:
                untouched[e['source']] = False
            _need(np.array_equal(canonical_out[untouched], a['grid'][untouched]), 'outside stamps preservation')
            _need(not np.array_equal(inp, out), 'nonidentity pair')
            # A ablation: routing-only consumption cannot produce the rectangles.
            route_only = a['grid'].copy()
            for e in a['events']:
                route_only[e['source']] = a['bg']
            _need(not np.array_equal(route_only, canonical_out), 'A must change the answer')
            # B ablation: treating unequal flanks as a valid entrance consumes an inert packet.
            geometry_only = canonical_out.copy()
            for inactive in a['inactive']:
                if inactive['mismatch']:
                    geometry_only[inactive['source']] = a['bg']
            _need(not np.array_equal(geometry_only, canonical_out), 'rail equality must change the answer')
            if split == 'train':
                features.update(_features(a))
                train_counts.append(len(a['stencils']))
            else:
                _need(len(a['stencils']) > max(train_counts), 'test must combine more stencil identities')
    _need({'matched', 'mismatch', 'gap', 'tail', 'open', 'closed', 'turn', 'time_priority',
           'tie_priority', 'distance_overlap', 'repaint', 'unserved', 'occlusion', 'retention'} <= features,
          'training must expose selection, motion, and stamp scheduling')


def _timing_fixtures():
    """Hand-built causal checks: change a route, keeping sources and boxes fixed."""
    stencil_boxes = {3: (8, 17, 12, 23), 4: (10, 19, 14, 25)}

    def scene(a_points, extra=False):
        g = np.zeros((20, 26), dtype=np.int8)
        g[:, 0] = 9
        packets = [(a_points, 3, 1), ([(15, 1), (15, 14)], 4, 2)]
        if extra:
            packets.append(([(18, 1), (18, 20)], 3, 2))
        for points, cue, paint in packets:
            path = _polyline(points)
            for p, color in _walls(np.random.default_rng(11), path, 20, 26, cue, [3, 4]).items():
                _need(g[p] in (0, color), 'fixture wall collision')
                g[p] = color
            g[path[0]] = paint
        for cue, box in stencil_boxes.items():
            for p in _corners(box, 0):
                _need(g[p] == 0, 'fixture marker collision')
                g[p] = cue
        return g

    short = scene([(2, 1), (2, 10)])
    detour = scene([(2, 1), (2, 8), (6, 8), (6, 12)])
    _need([e['time'] for e in _analyse(short)['events']] == [9, 13], 'straight step counts')
    _need([e['time'] for e in _analyse(detour)['events']] == [15, 13], 'detour step count')
    _need(solve(short)[10, 21] == 2 and solve(detour)[10, 21] == 1,
          'a detour must reverse cross-stencil priority')
    tied = scene([(2, 1), (2, 14)])
    _need(solve(tied)[10, 21] == 1, 'equal-time top-first priority')
    gapped = detour.copy()
    gapped[1, 5] = gapped[3, 5] = 0
    _need(_analyse(gapped)['events'][0]['time'] == 15 and solve(gapped)[10, 21] == 1,
          'a missing rail step still costs one tick')
    mismatched = short.copy()
    mismatched[16, 3:14] = 3
    result = solve(mismatched)
    _need(result[2, 1] == 0 and result[15, 1] == 2 and result[13, 24] == 0,
          'mismatched rails must leave the packet and its unserved stencil unchanged')
    repainted = scene([(2, 1), (2, 10)], extra=True)
    _need(solve(short)[9, 18] == 1 and solve(repainted)[9, 18] == 2,
          'a second payload must repaint the same stencil')
    # Rotate the complete equal-time scene; source order is carried with the launcher.
    expected = solve(tied)
    for k in range(4):
        _need(np.array_equal(solve(np.rot90(tied, k)), np.rot90(expected, k)), 'rotated tie convention')


def self_test():
    """Bounded checks on 42 seeds, every count contract, and semantic ablations."""
    _timing_fixtures()
    shapes, structures, boxes, times, backgrounds, orientations = set(), set(), set(), set(), set(), set()
    packet_counts, stencil_counts, diagonals, thin, signatures = set(), set(), set(), set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(tuple(_signature(x['input']) for s in p for x in p[s]))
        for split in ('train', 'test'):
            for pair in p[split]:
                a = _analyse(pair['input'])
                shapes.add(pair['input'].shape)
                backgrounds.add(a['bg'])
                orientations.add(a['rotation'])
                packet_counts.add(len(a['events']) + len(a['inactive']))
                stencil_counts.add(len(a['stencils']))
                times.update(e['time'] for e in a['events'])
                roles = {}
                structures.add(tuple(roles.setdefault(int(c), len(roles)) for c in pair['input'].flat))
                for obj in a['stencils'].values():
                    r, c, s, t = obj['box']
                    boxes.add((s - r + 1, t - c + 1))
                    diagonals.add(obj['diagonal'])
                    if min(s - r + 1, t - c + 1) == 2:
                        thin.add('horizontal' if s - r == 1 else 'vertical')
    _need(len(signatures) == 42 and len(structures) >= 150, 'structural variation')
    _need(len(shapes) >= 50 and len(boxes) >= 50 and len(times) >= 20, 'geometry variation')
    _need(backgrounds == set(range(10)) and orientations == set(range(4)), 'visible role variation')
    _need(stencil_counts == {2, 3, 4, 5} and len(packet_counts) >= 4, 'count variation')
    _need(diagonals == {0, 1} and thin == {'horizontal', 'vertical'}, 'corner geometry variation')
    a, b = generate_puzzle(117, 2, 1), generate_puzzle(117, 2, 1)
    _need(all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s])
              for k in ('input', 'output')), 'whole-puzzle determinism')
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(19 + n * 4 + m, n, m)
            _need(len(p['train']) == n and len(p['test']) == m, 'explicit counts')
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.5, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = {'train': [{k: v.copy() for k, v in pair.items()} for pair in a['train']],
           'test': [{k: v.copy() for k, v in pair.items()} for pair in a['test']]}
    bad['test'][0]['output'] = bad['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('incorrect output accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed')
