"""Mirror-locked shuttles: recover gate identities virtually, then move tokens.

Only the standard library and NumPy are used. No parent code is imported.
"""
from collections import deque
import numpy as np

BACKGROUND, TRACK, TOKEN, ERASED = 0, 5, 2, 9
GUIDES = frozenset((1, 3, 4, 6, 7, 8))
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'grid bounds')
    _require(np.issubdtype(a.dtype, np.integer), 'integer grid required')
    _require(bool(np.all((a >= 0) & (a <= 9))), 'ARC colors required')
    return a


def _plus(p, d):
    return p[0] + d[0], p[1] + d[1]


def _inside(shape, p):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _orbit(p, shape):
    r, c = p
    h, w = shape
    return tuple(sorted({(r, c), (h - 1 - r, c),
                         (r, w - 1 - c), (h - 1 - r, w - 1 - c)}))


def _paths(grid):
    """Read simple track components. Canonical order starts at the lesser end."""
    cells = set(map(tuple, np.argwhere((grid == TRACK) | (grid == TOKEN))))
    remaining, result = set(cells), []
    while remaining:
        start = min(remaining)
        pending, component = deque([start]), {start}
        remaining.remove(start)
        while pending:
            p = pending.popleft()
            for d in DIRECTIONS:
                q = _plus(p, d)
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    pending.append(q)
        neighbors = {p: sorted(_plus(p, d) for d in DIRECTIONS
                               if _plus(p, d) in component) for p in component}
        ends = sorted(p for p in component if len(neighbors[p]) == 1)
        _require(len(ends) == 2 and all(1 <= len(n) <= 2 for n in neighbors.values()),
                 'tracks must be separated simple paths')
        path, previous, p = [], None, ends[0]
        while True:
            path.append(p)
            onward = [q for q in neighbors[p] if q != previous]
            if not onward:
                break
            _require(len(onward) == 1, 'ambiguous route')
            previous, p = p, onward[0]
        _require(len(path) == len(component), 'cyclic track')
        tokens = [p for p in path if grid[p] == TOKEN]
        _require(len(tokens) <= 1 and all(p in ends for p in tokens),
                 'at most one endpoint shuttle per track')
        result.append(path)
    return result


def _stations(path, grid):
    stations = []
    for i in range(1, len(path) - 1):
        before, p, after = path[i - 1:i + 2]
        d = (p[0] - before[0], p[1] - before[1])
        if d != (after[0] - p[0], after[1] - p[1]):
            continue
        n = (-d[1], d[0])
        a, b = _plus(p, n), _plus(p, (-n[0], -n[1]))
        if not (_inside(grid.shape, a) and _inside(grid.shape, b)):
            continue
        va, vb = int(grid[a]), int(grid[b])
        ga, gb = va in GUIDES or va == ERASED, vb in GUIDES or vb == ERASED
        _require(ga == gb, 'station has only one side marker')
        if ga:
            stations.append((i, a, b, va == vb))
    return stations


def _decode(input_grid):
    inp = _grid(input_grid)
    static = inp.copy()
    static[static == TOKEN] = TRACK
    for raw in np.argwhere(static == ERASED):
        p = tuple(raw)
        witnesses = {int(static[q]) for q in _orbit(p, inp.shape)
                     if inp[q] != ERASED}
        _require(len(witnesses) == 1 and witnesses <= GUIDES,
                 'missing or contradictory guide witness')
        static[p] = next(iter(witnesses))
    _require(np.array_equal(static, static[::-1])
             and np.array_equal(static, static[:, ::-1]), 'static schematic is not folded')
    paths = _paths(inp)
    _require(bool(paths) and np.any(inp == TOKEN), 'at least one shuttle required')
    owned, routes, station_count = set(), [], 0
    for canonical in paths:
        stations = _stations(canonical, static)
        station_count += len(stations)
        owned.update(p for s in stations for p in s[1:3])
        sources = [p for p in canonical if inp[p] == TOKEN]
        if sources:
            path = canonical if sources[0] == canonical[0] else canonical[::-1]
            routes.append((path, _stations(path, static)))
    guide_cells = set(map(tuple, np.argwhere(np.isin(static, tuple(GUIDES)))))
    _require(owned == guide_cells, 'guide cells must belong to two-sided stations')
    return {'input': inp, 'static': static, 'paths': paths, 'routes': routes,
            'stations': station_count}


def _destinations(scene, mode='normal'):
    result = []
    for path, stations in scene['routes']:
        stop = len(path) - 1
        for i, a, b, matched in stations:
            if mode == 'literal':
                matched = scene['input'][a] == scene['input'][b]
            elif mode == 'all_pass':
                matched = True
            elif mode == 'double_holes_pass' and scene['input'][a] == scene['input'][b] == ERASED:
                matched = True
            if not matched:
                stop = i - 1
                break
        result.append((path[0], path[stop]))
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all roles, witnesses, route order and destinations from input only."""
    scene = _decode(input_grid)
    out = scene['input'].copy()
    for source, destination in _destinations(scene):
        out[source] = TRACK
        out[destination] = TOKEN
    return out


def _polyline(points):
    path = [points[0]]
    for target in points[1:]:
        p = path[-1]
        d = (int(np.sign(target[0] - p[0])), int(np.sign(target[1] - p[1])))
        if (d[0] == 0) == (d[1] == 0):
            return None
        while p != target:
            p = _plus(p, d)
            path.append(p)
    cells = set(path)
    if len(cells) != len(path):
        return None
    if any(sum(_plus(p, d) in cells for d in DIRECTIONS)
           != (1 if i in (0, len(path) - 1) else 2) for i, p in enumerate(path)):
        return None
    return path


def _sample_path(rng, shape, kind):
    h, w = shape
    rh, cw = h // 2 - 2, w // 2 - 2
    if kind == 'bridge':
        r, c = int(rng.integers(1, rh + 1)), int(rng.integers(0, min(3, cw - 3) + 1))
        return _polyline([(r, c), (r, w - 1 - c)])
    if kind == 'arch':
        r = int(rng.integers(1, rh - 2))
        bottom = int(rng.integers(r + 3, rh + 1))
        c = int(rng.integers(1, min(3, cw - 3) + 1))
        return _polyline([(r, c), (bottom, c), (bottom, w - 1 - c), (r, w - 1 - c)])
    p = (int(rng.integers(1, rh + 1)), int(rng.integers(1, cw + 1)))
    points, d = [p], DIRECTIONS[int(rng.integers(4))]
    for _ in range(int(rng.integers(1, 5))):
        choices = [_plus(p, (length * d[0], length * d[1]))
                   for length in range(3, max(rh, cw) + 1)
                   if 1 <= p[0] + length * d[0] <= rh
                   and 1 <= p[1] + length * d[1] <= cw]
        if not choices:
            return None
        p = choices[int(rng.integers(len(choices)))]
        points.append(p)
        d = (-d[1], d[0]) if rng.integers(2) else (d[1], -d[0])
    return _polyline(points)


def _piece(rng, shape, palette, gate_count, behavior, kind):
    path = _sample_path(rng, shape, kind)
    if path is None:
        return None
    tracks = {q for p in path for q in _orbit(p, shape)}
    candidates = []
    for i in range(2, len(path) - 2):
        before, p, after = path[i - 1:i + 2]
        d = (p[0] - before[0], p[1] - before[1])
        if (d != (after[0] - p[0], after[1] - p[1])
                or p[0] >= shape[0] // 2 or p[1] >= shape[1] // 2):
            continue
        n = (-d[1], d[0])
        a, b = _plus(p, n), _plus(p, (-n[0], -n[1]))
        if not (_inside(shape, a) and _inside(shape, b)):
            continue
        oa, ob = _orbit(a, shape), _orbit(b, shape)
        if len(oa) != 4 or len(ob) != 4 or set(oa) & set(ob):
            continue
        flanks = set(oa) | set(ob)
        if flanks & tracks or any(sum(_plus(q, t) in tracks for t in DIRECTIONS) != 1
                                 for q in flanks):
            continue
        candidates.append((i, a, b, flanks))
    selected, used = [], set()
    for index in rng.permutation(len(candidates)):
        candidate = candidates[int(index)]
        if used & candidate[3] or any(abs(candidate[0] - x[0]) < 2 for x in selected):
            continue
        selected.append(candidate)
        used.update(candidate[3])
        if len(selected) == gate_count:
            break
    if len(selected) != gate_count:
        return None
    selected.sort(key=lambda x: x[0])
    paint = {p: TRACK for p in tracks}
    first_color = int(rng.choice(palette))
    for j, (_, a, b, _) in enumerate(selected):
        matched = behavior == 'open' or (j == 0 if behavior == 'blocked' else rng.random() < .65 or j == 0)
        color = first_color if j == 0 else int(rng.choice([x for x in palette if x != first_color]))
        other = color if matched else int(rng.choice([x for x in palette if x != color]))
        for q in _orbit(a, shape):
            paint[q] = color
        for q in _orbit(b, shape):
            paint[q] = other
    board = np.zeros(shape, dtype=np.int8)
    for p, color in paint.items():
        board[p] = color
    try:
        paths = _paths(board)
        _require(set(p for route in paths for p in route) == tracks, 'piece contacts')
        for route in paths:
            _stations(route, board)
    except AssertionError:
        return None
    return paint, path, selected


def _make_pair(rng, shape, palette, behavior, is_test):
    needed = 3 if behavior == 'blocked' else int(rng.integers(2, 4))
    for _ in range(160):
        kind = str(rng.choice(('fragment', 'bridge', 'arch'), p=(.45, .35, .20)))
        piece = _piece(rng, shape, palette, needed, behavior, kind)
        if piece is not None:
            break
    else:
        return None
    paint, primary, selected = piece
    board = np.zeros(shape, dtype=np.int8)
    for p, color in paint.items():
        board[p] = color
    # Additional routes use the same folded scene; they are not panel tasks.
    for _ in range(int(rng.integers(0, 3))):
        for attempt in range(24):
            extra = _piece(rng, shape, palette, int(rng.integers(1, 3)), 'mixed',
                           str(rng.choice(('fragment', 'bridge'))))
            if extra is None:
                continue
            additions = extra[0]
            if any(board[p] != BACKGROUND for p in additions):
                continue
            new_tracks = {p for p, color in additions.items() if color == TRACK}
            old_tracks = set(map(tuple, np.argwhere(board == TRACK)))
            if any(_plus(p, d) in old_tracks for p in new_tracks for d in DIRECTIONS):
                continue
            candidate = board.copy()
            for p, color in additions.items():
                candidate[p] = color
            try:
                for path in _paths(candidate):
                    _stations(path, candidate)
            except AssertionError:
                continue
            board = candidate
            break
    paths = _paths(board)
    source = primary[0]
    for path in paths:
        if source in path:
            board[source] = TOKEN
        elif rng.random() < rng.uniform(.35, .85):
            board[path[0] if rng.integers(2) else path[-1]] = TOKEN
    token_mask = board == TOKEN
    if (np.array_equal(token_mask, token_mask[::-1])
            and np.array_equal(token_mask, token_mask[:, ::-1])):
        # The source is never removed; omit one reflected token if needed.
        removable = [tuple(p) for p in np.argwhere(token_mask) if tuple(p) != source]
        if removable:
            board[removable[0]] = TRACK
    # First passing gate: only the diagonal witness survives on one flank.
    _, critical, counterpart, _ = selected[0]
    critical_orbit = _orbit(critical, shape)
    protected = _orbit(counterpart, shape)
    survivor = (shape[0] - 1 - critical[0], shape[1] - 1 - critical[1])
    guide_orbits = sorted({_orbit(tuple(p), shape) for p in
                           np.argwhere(np.isin(board, tuple(GUIDES)))})
    available = [o for o in guide_orbits if o not in (critical_orbit, protected)]
    rng.shuffle(available)
    # Expose the tempting but wrong "two holes are equal" rule in the blocked
    # demonstration. The next unequal station remains as a later contrast.
    blocker = next(((a, b) for _, a, b, _ in selected if board[a] != board[b]), None)
    forced = [_orbit(p, shape) for p in blocker] if blocker else []
    available = forced + [o for o in available if o not in forced]
    inp = board.copy()
    for p in critical_orbit:
        if p != survivor:
            inp[p] = ERASED
    if is_test:
        if len(available) < 2:
            return None
        triple_count = int(rng.integers(2, min(5, len(available)) + 1))
        for orbit in available[:triple_count]:
            # On a reached blocker, erase both local flanks while retaining
            # their separate diagonal witnesses. Each still has its own ID.
            if orbit in forced:
                local = next(p for p in blocker if _orbit(p, shape) == orbit)
                visible = (shape[0] - 1 - local[0], shape[1] - 1 - local[1])
            else:
                visible = orbit[int(rng.integers(len(orbit)))]
            for p in orbit:
                if p != visible:
                    inp[p] = ERASED
        for orbit in available[triple_count:triple_count + int(rng.integers(0, 3))]:
            for index in rng.choice(len(orbit), size=min(2, len(orbit) - 1), replace=False):
                inp[orbit[int(index)]] = ERASED
    else:
        if blocker:
            for p in blocker:
                inp[p] = ERASED
        optional = [o for o in available if o not in forced]
        for orbit in optional[:int(rng.integers(0, 4 - len(forced)))]:
            inp[orbit[int(rng.integers(len(orbit)))]] = ERASED
    rotation = int(rng.integers(4))
    rotated_source, rotated_shape = source, shape
    for _ in range(rotation):
        rotated_source = (rotated_shape[1] - 1 - rotated_source[1], rotated_source[0])
        rotated_shape = (rotated_shape[1], rotated_shape[0])
    inp = np.rot90(inp, rotation).copy()
    try:
        scene = _decode(inp)
        if _destinations(scene) == _destinations(scene, 'literal'):
            return None
        # Packing must retain the promised demonstrations even if nearby
        # guides happen to create an additional geometrically valid station.
        primary_stations = next(stations for route, stations in scene['routes']
                                if route[0] == rotated_source)
        failed = [s for s in primary_stations if not s[3]]
        if behavior == 'open' and (failed or len({int(scene['static'][s[1]])
                                                  for s in primary_stations}) < 2):
            return None
        if behavior == 'blocked' and (len(failed) < 2 or not primary_stations[0][3]
                                     or inp[failed[0][1]] != ERASED
                                     or inp[failed[0][2]] != ERASED):
            return None
    except AssertionError:
        return None
    return {'input': inp, 'output': solve(inp)}


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(name + ' is outside the legal range')
    return int(value)


def _signature(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ns = _count(num_train, 2, 10, 'num_train'), _count(num_test, 1, 4, 'num_test')
    if nt <= ns:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed) % (1 << 64))
    pool = list(map(int, rng.choice(sorted(GUIDES), int(rng.integers(2, 7)), replace=False)))
    mixed_palette = bool(rng.integers(2))
    result, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ns)):
        for index in range(count):
            behavior = ('blocked' if index == 0 else 'open') if split == 'train' and index < 2 else str(rng.choice(('mixed', 'open', 'blocked')))
            for attempt in range(80):
                low = 20 if behavior == 'blocked' or split == 'test' else 18
                shape = tuple(map(int, rng.integers(low, 31, size=2)))
                if rng.random() < .07:
                    shape = (30, 30)
                palette = pool if not mixed_palette else list(map(int, rng.choice(
                    sorted(GUIDES), int(rng.integers(2, 7)), replace=False)))
                pair = _make_pair(rng, shape, palette, behavior, split == 'test')
                if pair is None or _signature(pair['input']) in seen:
                    continue
                if split == 'train' and index == 1 and pair['input'].shape == result['train'][0]['input'].shape:
                    continue
                seen.add(_signature(pair['input']))
                result[split].append(pair)
                break
            else:
                raise RuntimeError('bounded railway construction exhausted')
    validate_generated_puzzle(result)
    return result


def _features(inp):
    scene = _decode(inp)
    orbits = {_orbit(tuple(p), inp.shape) for p in np.argwhere(inp == ERASED)}
    triple = sum(sum(inp[p] == ERASED for p in o) == 3 for o in orbits)
    diagonal = sum(inp[r, inp.shape[1] - 1 - c] == ERASED
                   and inp[inp.shape[0] - 1 - r, c] == ERASED
                   and inp[inp.shape[0] - 1 - r, inp.shape[1] - 1 - c] != ERASED
                   for r, c in np.argwhere(inp == ERASED))
    turns = [sum((b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1])
                 for a, b, c in zip(path, path[1:], path[2:])) for path in scene['paths']]
    return {'shape': inp.shape, 'paths': len(scene['paths']), 'tokens': len(scene['routes']),
            'stations': scene['stations'], 'lengths': [len(p) for p in scene['paths']],
            'turns': turns, 'holes': int(np.count_nonzero(inp == ERASED)),
            'triples': int(triple), 'diagonal_only': int(diagonal),
            'guides': tuple(sorted(set(map(int, scene['static'].flat)) & GUIDES)),
            'blocked': sum(any(not s[3] for s in stations) for p, stations in scene['routes']),
            'open': sum(all(s[3] for s in stations) for p, stations in scene['routes']),
            'crosses_axis': any(min(r for r, c in p) < (inp.shape[0] - 1) / 2 < max(r for r, c in p)
                                or min(c for r, c in p) < (inp.shape[1] - 1) / 2 < max(c for r, c in p)
                                for p in scene['paths'])}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split lists')
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, 'pair counts')
    seen, demonstrated = set(), {'blocked': False, 'open': False, 'first_failure': False,
                                'new_identity': False, 'double_hole': False, 'A': False, 'B': False}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'shape preservation')
            signature = _signature(inp)
            _require(signature not in seen, 'duplicate input')
            seen.add(signature)
            scene, f = _decode(inp), _features(inp)
            _require(np.array_equal(solve(inp), out), 'incorrect shuttle destination')
            _require(np.count_nonzero(inp == TOKEN) == np.count_nonzero(out == TOKEN), 'token conservation')
            static_cells = (inp != TRACK) & (inp != TOKEN)
            _require(np.array_equal(inp[static_cells], out[static_cells]), 'static/erasure preservation')
            endpoints = _destinations(scene)
            _require(len({d for s, d in endpoints}) == len(endpoints), 'destination collision')
            _require(np.count_nonzero(inp != out) == 2 * sum(s != d for s, d in endpoints), 'only terminals change')
            _require(f['diagonal_only'] >= 1, 'both folds must supply a diagonal witness')
            _require(_destinations(scene) != _destinations(scene, 'literal'), 'A has no effect')
            token_mask = inp == TOKEN
            _require(not (np.array_equal(token_mask, token_mask[::-1])
                          and np.array_equal(token_mask, token_mask[:, ::-1])), 'overlay must expose asymmetry')
            if split == 'train':
                _require(3 <= f['holes'] <= 6 and f['triples'] == 1, 'training mask regime')
                demonstrated['blocked'] |= f['blocked'] > 0
                demonstrated['open'] |= f['open'] > 0
                demonstrated['A'] |= _destinations(scene) != _destinations(scene, 'literal')
                demonstrated['B'] |= _destinations(scene) != _destinations(scene, 'all_pass')
                demonstrated['double_hole'] |= _destinations(scene) != _destinations(scene, 'double_holes_pass')
                for path, stations in scene['routes']:
                    failed = [s for s in stations if not s[3]]
                    demonstrated['first_failure'] |= len(failed) >= 2
                    passed = [int(scene['static'][s[1]]) for s in stations if s[3]]
                    demonstrated['new_identity'] |= not failed and len(set(passed)) >= 2
            else:
                _require(f['holes'] >= 9 and f['triples'] >= 3, 'test mask shift')
    _require(all(demonstrated.values()), 'demonstrations omit a required contrast')
    _require(len({np.asarray(p['input']).shape for p in puzzle['train']}) >= 2, 'training shapes must vary')


def self_test():
    records, puzzles, fingerprints = [], [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        fingerprints.add(tuple(_signature(q['input']) for s in puzzle for q in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                original = inp.copy()
                _require(np.array_equal(solve(inp), out) and np.array_equal(inp, original), 'solve mutates input')
                for k in (1, 2, 3):
                    _require(np.array_equal(solve(np.rot90(inp, k)), np.rot90(out, k)), 'rotation equivariance')
                f = _features(inp)
                f['split'] = split
                records.append(f)
    _require(len(fingerprints) == 42, 'seed diversity')
    _require(len({r['shape'] for r in records}) >= 55, 'shape diversity')
    _require((30, 30) in {r['shape'] for r in records}, 'full canvas absent')
    _require(len({r['paths'] for r in records}) >= 4 and len({r['tokens'] for r in records}) >= 5, 'count diversity')
    _require({(h % 2, w % 2) for h, w in (r['shape'] for r in records)} == {(0, 0), (0, 1), (1, 0), (1, 1)}, 'fold parity')
    _require(len({r['guides'] for r in records}) >= 12, 'palette diversity')
    _require(set().union(*(set(r['guides']) for r in records)) == GUIDES, 'guide literals')
    _require({0, 1, 2} <= {t for r in records for t in r['turns']}, 'route geometry')
    _require(any(r['crosses_axis'] for r in records) and any(not r['crosses_axis'] for r in records), 'fold crossing diversity')
    _require(len({(r['paths'], tuple(r['lengths']), tuple(r['turns']), r['stations']) for r in records}) >= 90, 'structure diversity')
    repeat = generate_puzzle(0)
    _require(all(np.array_equal(a[key], b[key]) for s in repeat for a, b in zip(repeat[s], puzzles[0][s])
                 for key in a), 'whole puzzle determinism')
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(100 + nt * 4 + ns, nt, ns)
            _require(len(p['train']) == nt and len(p['test']) == ns, 'explicit counts')
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # An endpoint-only shortcut and an actual static-cell mutation are rejected.
    for mutation in ('endpoint', 'static'):
        bad = {s: [{k: a.copy() for k, a in q.items()} for q in puzzles[0][s]] for s in puzzles[0]}
        pair = bad['train'][0]
        if mutation == 'endpoint':
            scene = _decode(pair['input'])
            pair['output'] = pair['input'].copy()
            for start, end in _destinations(scene, 'all_pass'):
                pair['output'][start], pair['output'][end] = TRACK, TOKEN
        else:
            p = tuple(np.argwhere(pair['input'] == ERASED)[0])
            pair['output'][p] = 1
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('corrupt output accepted')
    return {'seeds': 42, 'pairs': len(records), 'shapes': len({r['shape'] for r in records}),
            'path_counts': sorted({r['paths'] for r in records}),
            'token_counts': sorted({r['tokens'] for r in records}),
            'guide_palettes': len({r['guides'] for r in records}),
            'turn_counts': sorted({t for r in records for t in r['turns']}),
            'cross_fold_pairs': sum(r['crosses_axis'] for r in records)}
