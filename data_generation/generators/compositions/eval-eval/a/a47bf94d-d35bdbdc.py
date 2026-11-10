"""Crossing-scoped relation composition, rendered as split cable terminals.

No files, seeds, examples or construction metadata are read by solve().
"""
from __future__ import annotations

import hashlib
import json
from itertools import permutations

import numpy as np

D4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIAMOND = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
CHECKER = ~DIAMOND


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _read_token(block, bg, forbidden):
    center = int(block[1, 1])
    body = int(block[0, 1])
    assert body not in forbidden and center not in forbidden
    for kind in ("square", "cross"):
        expected = np.full((3, 3), body if kind == "square" else bg, dtype=np.int8)
        if kind == "cross":
            expected[DIAMOND] = body
        expected[1, 1] = center
        if np.array_equal(block, expected):
            return body, center, kind
    raise AssertionError("malformed relation token")


def _trace(field, wire, crossing, start, end, centers):
    """Directional pixel tracing; a marked degree-four cell has two strands."""
    visible = set(map(tuple, np.argwhere((field == wire) | (field == crossing))))
    graph = {p: {d: (p[0] + dr, p[1] + dc) for d, (dr, dc) in enumerate(D4)
                 if (p[0] + dr, p[1] + dc) in visible} for p in visible}
    endpoints = {(r, c) for r in (start, end) for c in centers}
    junctions = set()
    for p, edges in graph.items():
        if p in endpoints:
            assert len(edges) == 1 and field[p] == wire
        elif crossing is not None and field[p] == crossing:
            assert len(edges) == 4, "a crossing needs four straight arms"
            assert all(field[q] == wire for q in edges.values())
            junctions.add(p)
        else:
            assert len(edges) == 2, "ordinary cable branches or breaks"
    routes, destinations, used = [], [], set()
    for c in centers:
        p, incoming, path, states = (start, c), 0, [], set()
        while True:
            assert (p, incoming) not in states, "closed cable"
            states.add((p, incoming))
            path.append(p)
            used.add(p)
            if p in endpoints and p[0] == end:
                destinations.append(centers.index(p[1]))
                break
            edges = graph[p]
            if p == (start, c):
                outgoing = 2
            else:
                assert incoming in edges
                outgoing = (incoming + 2) % 4 if p in junctions else next(d for d in edges if d != incoming)
            p = edges[outgoing]
            incoming = (outgoing + 2) % 4
            assert p[0] != start
        routes.append(path)
    assert sorted(destinations) == list(range(len(centers)))
    assert used == visible, "unattached wire fragment"
    neighbors = [set() for _ in centers]
    encounters = {}
    for p in sorted(junctions):
        owners = [i for i, path in enumerate(routes) if p in path]
        assert len(owners) == 2
        i, j = owners
        neighbors[i].add(j)
        neighbors[j].add(i)
        encounters[i, j] = encounters.get((i, j), 0) + 1
    return destinations, routes, neighbors, encounters


def _parse(input_grid):
    a = _grid(input_grid)
    corners = [int(a[r, c]) for r, c in ((0, 0), (0, -1), (-1, 0), (-1, -1))]
    assert len(set(corners)) == 1
    bg = corners[0]
    candidates = []
    for rotation in range(4):
        g = np.rot90(a, rotation)
        h, w = g.shape
        for wire in sorted(set(map(int, np.unique(g))) - {bg}):
            positions = np.argwhere(g == wire)
            start, end = int(positions[:, 0].min()), int(positions[:, 0].max())
            if start < 3 or end > h - 4 or end - start < 4:
                continue
            centers = list(map(int, np.flatnonzero(g[start] == wire)))
            if (not 2 <= len(centers) <= 5 or centers != list(np.flatnonzero(g[end] == wire))
                    or centers[0] < 1 or centers[-1] > w - 2
                    or any(b - c < 4 for c, b in zip(centers, centers[1:]))):
                continue
            try:
                source = np.zeros(g.shape, dtype=bool)
                terminals = source.copy()
                for c in centers:
                    source[start-3:start, c-1:c+2] = True
                    terminals[start-3:start, c-1:c+2] = True
                    terminals[end+1:end+4, c-1:c+2] = True
                outside = ~source
                outside[start:end+1] = False
                assert np.all(g[outside] == bg), "extra objects or nonblank receiving bank"
                field = g[start:end+1]
                others = set(map(int, np.unique(field))) - {bg, wire}
                assert len(others) <= 1
                crossing = next(iter(others)) if others else None
                forbidden = {bg, wire, crossing}
                tokens = [_read_token(g[start-3:start, c-1:c+2], bg, forbidden) for c in centers]
                dest, routes, neighbors, encounters = _trace(g, wire, crossing, start, end, centers)
                # Conflicting copies are allowed outside this particular scope.
                for i, (_, key, _) in enumerate(tokens):
                    values = {tokens[j][1] for j in neighbors[i] if tokens[j][0] == key}
                    assert len(values) <= 1, "conflicting crossing donors"
                candidates.append(dict(grid=g, rotation=rotation, bg=bg, wire=wire,
                                       crossing=crossing, centers=centers, start=start, end=end,
                                       tokens=tokens, destinations=dest, routes=routes,
                                       neighbors=neighbors, encounters=encounters, terminals=terminals))
            except (AssertionError, KeyError, StopIteration):
                continue
    assert len(candidates) == 1, f"expected one visible interpretation, got {len(candidates)}"
    return candidates[0]


def _answers(tokens, neighbors):
    values = []
    for i, (_, key, _) in enumerate(tokens):
        donors = {tokens[j][1] for j in neighbors[i] if tokens[j][0] == key}
        assert len(donors) <= 1
        values.append(next(iter(donors)) if donors else None)
    return values


def _render(g, bg, centers, start, end, destinations, tokens, values):
    out = g.copy()
    for c in centers:
        out[start-3:start, c-1:c+2] = bg
        out[end+1:end+4, c-1:c+2] = bg
    for i, value in enumerate(values):
        if value is None:
            continue
        c, d = centers[i], centers[destinations[i]]
        out[start-3:start, c-1:c+2] = np.where(DIAMOND, tokens[i][0], bg)
        out[end+1:end+4, d-1:d+2] = np.where(CHECKER, value, bg)
    return out


def _predict(info, values=None, destinations=None):
    if values is None:
        values = _answers(info['tokens'], info['neighbors'])
    if destinations is None:
        destinations = info['destinations']
    result = _render(info['grid'], info['bg'], info['centers'], info['start'], info['end'],
                     destinations, info['tokens'], values)
    return np.rot90(result, -info['rotation']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Compose once using original values of matching crossing neighbors."""
    return _predict(_parse(input_grid))


def _moves(target, rng):
    # A temporary exterior track permits arbitrary permutations without
    # drawing a separate adjacent-swap module for each inversion.
    n = len(target)
    state, moves = list(range(n)) + [None], []
    for j in map(int, rng.permutation(n)):
        if state[j] == target[j]:
            continue
        saved = state[j]
        moves.append((j, n))
        state[n], state[j] = state[j], None
        vacant = j
        while target[vacant] != saved:
            old = state.index(target[vacant])
            moves.append((old, vacant))
            state[vacant], state[old] = state[old], None
            vacant = old
        moves.append((n, vacant))
        state[vacant], state[n] = state[n], None
    return moves


def _scaffold(rng, n, lesson):
    if lesson == 'scope':
        # An induced four-vertex path supplies two different dictionaries
        # for the same query color, ruling out any global color function.
        moves = [(1, n), (0, 1), (2, 0), (n, 2)]
    elif lesson == 'basic':
        # The first adjacent exchange creates a three-vertex star, including
        # a twice-crossing strand pair; every geometry draw remains valid.
        target = list(range(n))
        target[0], target[1] = target[1], target[0]
        moves = _moves(target, rng)
    else:
        target = list(map(int, rng.permutation(n)))
        moves = _moves(target, rng)
    if not moves or (not lesson and len(moves) <= 6 and rng.random() < .45):
        lane = int(rng.integers(n - 1))
        moves = [(lane, n), (n, lane)] + moves
    # Independently sampled spacings and margins; all size budgets explicit.
    gaps = [int(v) for v in rng.integers(4, 7, size=n-1)]
    while sum(gaps) > 23:
        k = int(np.argmax(gaps))
        gaps[k] -= 1
    spare = int(rng.integers(3, min(6, 27 - sum(gaps)) + 1))
    side_budget = 30 - (3 + sum(gaps) + spare)
    left = int(rng.integers(min(3, side_budget) + 1))
    right = int(rng.integers(min(3, side_budget-left) + 1))
    centers = [left + 1]
    for gap in gaps:
        centers.append(centers[-1] + gap)
    columns = centers + [centers[-1] + spare]
    width = columns[-1] + 1 + right
    distances = [2] * max(0, len(moves)-1)
    lead, tail = 2, 2
    budget = 30 - (7 + lead + tail + sum(distances))
    assert budget >= 0
    top = int(rng.integers(1, min(3, budget) + 1))
    budget -= top
    bottom = int(rng.integers(min(3, budget) + 1))
    budget -= bottom
    pieces = [lead, tail] + distances
    for _ in range(int(rng.integers(budget + 1))):
        pieces[int(rng.integers(len(pieces)))] += 1
    lead, tail, distances = pieces[0], pieces[1], pieces[2:]
    start = top + 3
    row = start + lead
    rows = []
    for i in range(len(moves)):
        rows.append(row)
        if i < len(distances):
            row += distances[i]
    end = rows[-1] + tail
    paths = [[(start, c)] for c in centers]
    state, previous = list(range(n)) + [None], start
    for row, (old, new) in zip(rows, moves):
        for lane, owner in enumerate(state):
            if owner is not None:
                paths[owner].extend((r, columns[lane]) for r in range(previous+1, row+1))
        owner = state[old]
        step = 1 if columns[new] > columns[old] else -1
        paths[owner].extend((row, c) for c in range(columns[old]+step, columns[new]+step, step))
        state[new], state[old] = state[old], None
        previous = row
    for lane, owner in enumerate(state):
        if owner is not None:
            paths[owner].extend((r, columns[lane]) for r in range(previous+1, end+1))
    field = np.zeros((end+4+bottom, width), dtype=np.int8)
    owners = {}
    for i, path in enumerate(paths):
        assert len(set(path)) == len(path)
        for p in path:
            owners.setdefault(p, set()).add(i)
    neighbors = [set() for _ in range(n)]
    encounters = {}
    for p, ids in owners.items():
        field[p] = 1 if len(ids) == 1 else 2
        if len(ids) == 2:
            i, j = sorted(ids)
            neighbors[i].add(j)
            neighbors[j].add(i)
            encounters[i, j] = encounters.get((i, j), 0) + 1
        else:
            assert len(ids) == 1
    destinations = [centers.index(path[-1][1]) for path in paths]
    return field, centers, start, end, destinations, neighbors, encounters


def _relations(rng, n, neighbors, encounters, lesson):
    if lesson == 'basic':
        b = next(i for i in range(n) if len(neighbors[i]) >= 2)
        a, c = sorted(neighbors[b])[:2]
        rel = [None] * n
        rel[a], rel[b], rel[c] = (2, 1), (1, 2), (3, 3)
        return rel
    if lesson == 'scope':
        choices = [(i, j, k, l) for i, j, k, l in permutations(range(n), 4)
                   if j in neighbors[i] and k in neighbors[j] and l in neighbors[k]
                   and k not in neighbors[i] and j not in neighbors[l]
                   and encounters.get(tuple(sorted((k, l))), 0) >= 2]
        assert choices
        i, j, k, l = choices[int(rng.integers(len(choices)))]
        rel = [None] * n
        rel[i], rel[j], rel[k], rel[l] = (0, 1), (1, 1), (1, 3), (2, 1)
        return rel
    edges = [(i, j) for i in range(n) for j in sorted(neighbors[i])]
    for _ in range(200):
        k = int(rng.integers(2, min(7, n+2) + 1))
        bodies = list(map(int, rng.integers(k, size=n)))
        values = list(map(int, rng.integers(k, size=n)))
        i, j = edges[int(rng.integers(len(edges)))]
        values[i] = bodies[j]
        if rng.random() < .3:
            for a in range(n):
                if neighbors[a]:
                    values[a] = bodies[int(rng.choice(sorted(neighbors[a])))]
        tokens = [(b, v, 'square') for b, v in zip(bodies, values)]
        try:
            answers = _answers(tokens, neighbors)
        except AssertionError:
            continue
        if any(v is not None and v != values[a] for a, v in enumerate(answers)):
            return list(zip(bodies, values))
    # A guaranteed nonidentity lookup on a chosen edge; other queries have no body.
    rel = [(a+2, 6) for a in range(n)]
    rel[i], rel[j] = (0, 1), (1, 2)
    return rel


def _make_pair(rng, n, lesson, palette, rotation):
    for _ in range(32):
        field, centers, start, end, destinations, neighbors, encounters = _scaffold(rng, n, lesson)
        try:
            rel = _relations(rng, n, neighbors, encounters, lesson)
            break
        except (AssertionError, StopIteration):
            continue
    else:
        raise AssertionError('bounded scaffold construction failed')
    bg, wire, crossing = palette[:3]
    identities = list(map(int, rng.permutation(palette[3:])))
    g = np.array(palette, dtype=np.int8)[field]
    kinds = [str(v) for v in rng.choice(['square', 'cross'], size=n)]
    if lesson == 'basic':
        kinds = ['square'] * n
    elif lesson == 'scope':
        kinds[0], kinds[-1] = 'square', 'cross'
    tokens = [(identities[b], identities[v], kind) for (b, v), kind in zip(rel, kinds)]
    for c, (body, value, kind) in zip(centers, tokens):
        block = np.full((3, 3), body if kind == 'square' else bg, dtype=np.int8)
        if kind == 'cross':
            block[DIAMOND] = body
        block[1, 1] = value
        g[start-3:start, c-1:c+2] = block
    values = _answers(tokens, neighbors)
    out = _render(g, bg, centers, start, end, destinations, tokens, values)
    if rng.random() < .5:
        g, out = g[:, ::-1], out[:, ::-1]
    return {'input': np.rot90(g, rotation).copy(), 'output': np.rot90(out, rotation).copy()}


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count('num_train', num_train, 2, 10)
    nq = _count('num_test', num_test, 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(['count', 'orientation']))
    stable_roles = bool(rng.random() < .35)
    base_palette = list(map(int, rng.permutation(10)))
    axis = int(rng.integers(2))
    lessons = ['basic', 'scope'] + [None] * (nt-2)
    rng.shuffle(lessons)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            lesson = lessons[index] if split == 'train' else None
            n = 3 if lesson == 'basic' else 4 if lesson == 'scope' else int(rng.integers(2, 6))
            if shift == 'count':
                if split == 'test':
                    n = 5
                elif lesson is None:
                    n = int(rng.integers(2, 5))
                rotation = (0 if lesson == 'basic' else 1) if lesson else int(rng.integers(4))
            else:
                rotation = axis + (1 if split == 'test' else 0) + 2*int(rng.integers(2))
            for _ in range(24):
                palette = base_palette if stable_roles else list(map(int, rng.permutation(10)))
                pair = _make_pair(rng, n, lesson, palette, rotation)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise AssertionError('bounded distinct-input construction failed')
            seen.add(key)
            puzzle[split].append(pair)
    return puzzle


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _features(info):
    tokens, neighbors = info['tokens'], info['neighbors']
    values = _answers(tokens, neighbors)
    success = [i for i, v in enumerate(values) if v is not None]
    scope = []
    erased_donor = retained_donor = simultaneous = unchanged = own_body = 0
    for i in success:
        _, key, _ = tokens[i]
        donors = [j for j in neighbors[i] if tokens[j][0] == key]
        scope += [i for j in range(len(tokens)) if j != i and j not in neighbors[i]
                  and tokens[j][0] == key and tokens[j][1] != values[i]]
        erased_donor += any(values[j] is None for j in donors)
        retained_donor += any(values[j] is not None for j in donors)
        simultaneous += any(values[j] is not None and values[j] != tokens[j][1] for j in donors)
        unchanged += values[i] == key
        own_body += values[i] == tokens[i][0]
    contextual = any(tokens[i][1] == tokens[j][1] and values[i] != values[j]
                     for i in success for j in success)
    return dict(shape=info['grid'].shape, routes=len(tokens),
                orientation=info['rotation'] % 2,
                roles=(info['bg'], info['wire'], info['crossing']),
                permutation=tuple(info['destinations']),
                edges=sum(map(len, neighbors))//2,
                crossings=sum(info['encounters'].values()),
                repeated_crossings=sum(v > 1 for v in info['encounters'].values()),
                degrees=tuple(sorted(map(len, neighbors))),
                survivors=len(success), failures=len(tokens)-len(success),
                kinds=tuple(sorted({t[2] for t in tokens})),
                repeated_bodies=len(tokens)-len({t[0] for t in tokens}),
                monochrome=sum(b == v for b, v, _ in tokens),
                scope_conflicts=len(scope), contextual_values=int(contextual), erased_donor=erased_donor,
                retained_donor=retained_donor, simultaneous=simultaneous,
                unchanged=unchanged, own_body=own_body,
                changed=sum(v is not None and v != tokens[i][1] for i, v in enumerate(values)))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _count('num_train', nt, 2, 10)
    _count('num_test', nq, 1, 4)
    assert nt > nq
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == b.shape
            assert _key(a) not in seen
            seen.add(_key(a))
            info = _parse(a)
            expected = _predict(info)
            assert np.array_equal(expected, b), 'scoped original-value composition disagrees'
            canonical_b = np.rot90(b, info['rotation'])
            assert np.array_equal(info['grid'][~info['terminals']], canonical_b[~info['terminals']])
            assert np.array_equal(a == info['wire'], b == info['wire'])
            assert np.array_equal(a == info['crossing'], b == info['crossing'])
            f = _features(info)
            assert f['survivors'] >= 1 and f['changed'] >= 1
            # Removing B: simply carry the input center color along each cable.
            no_lookup = _predict(info, [t[1] for t in info['tokens']])
            assert not np.array_equal(no_lookup, b)
            if split == 'train':
                training.append((info, f))
    for name in ('scope_conflicts', 'erased_donor', 'retained_donor', 'simultaneous',
                 'unchanged', 'own_body', 'monochrome', 'failures', 'repeated_crossings',
                 'contextual_values'):
        assert any(f[name] for _, f in training), f'missing training witness: {name}'
    assert any(len(set(t[2] for t in info['tokens'])) > 1 for info, _ in training)
    # Removing A's permutation: vertically aligned transport changes an answer.
    assert any(not np.array_equal(_predict(info, destinations=list(range(len(info['tokens'])))),
                                  _predict(info)) for info, _ in training)
    # Removing crossing scope must admit a visibly contradictory remote donor.
    assert any(f['scope_conflicts'] for _, f in training)
    for order in ('first', 'last', 'nearest'):
        disagreements = 0
        for info, _ in training:
            tokens, centers = info['tokens'], info['centers']
            values = []
            for i, (_, key, _) in enumerate(tokens):
                donors = [j for j, t in enumerate(tokens) if j != i and t[0] == key]
                if order == 'last':
                    donors.reverse()
                elif order == 'nearest':
                    donors.sort(key=lambda j: (abs(centers[i]-centers[j]), j))
                values.append(tokens[donors[0]][1] if donors else None)
            disagreements += not np.array_equal(_predict(info, values), _predict(info))
        assert disagreements, f'global {order}-donor shortcut fits training'


def self_test():
    """42 streamed seeds, finite count matrix, equivariance and corruption checks."""
    metrics, puzzle_hashes, structural_hashes = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a, b = pair['input'], pair['output']
                info = _parse(a)
                metrics.append(_features(info))
                digest.update(a.tobytes())
                digest.update(b.tobytes())
                normalized = np.zeros(a.shape, dtype=np.int8)
                normalized[a == info['wire']] = 1
                normalized[a == info['crossing']] = 2
                normalized[(a != info['bg']) & (normalized == 0)] = 3
                structural_hashes.add(hashlib.sha256(normalized.tobytes()+str(a.shape).encode()).hexdigest())
        puzzle_hashes.add(digest.hexdigest())
    assert len(puzzle_hashes) == 42 and len(structural_hashes) > 160
    assert {m['routes'] for m in metrics} == {2, 3, 4, 5}
    assert len({m['shape'] for m in metrics}) >= 40
    assert len({m['roles'] for m in metrics}) >= 30
    assert len({m['permutation'] for m in metrics}) >= 20
    assert len({m['degrees'] for m in metrics}) >= 8
    assert len({m['survivors'] for m in metrics}) >= 4
    assert any(m['survivors'] == m['routes'] for m in metrics)
    assert any(m['monochrome'] and m['unchanged'] for m in metrics)
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(92000+a*10+b, a, b)
            validate_generated_puzzle(p)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid counts accepted')
    p, q = generate_puzzle(0), generate_puzzle(0)
    assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s])
               for k in ('input', 'output'))
    for pair in p['train'] + p['test']:
        a, b = pair['input'], pair['output']
        snapshot = a.copy()
        assert np.array_equal(solve(a), b) and np.array_equal(a, snapshot)
        for k in range(4):
            assert np.array_equal(solve(np.rot90(a, k)), np.rot90(b, k))
        assert np.array_equal(solve(a[:, ::-1]), b[:, ::-1])
    # A valid no-crossing diagram outside the nontrivial sampling policy:
    # even globally composable tokens cannot consult a noncrossing cable.
    straight = np.zeros((15, 11), dtype=np.int8)
    straight[4:11, [2, 7]] = 8
    straight[1:4, 1:4] = 1
    straight[2, 2] = 2
    straight[1:4, 6:9] = 2
    straight[2, 7] = 3
    erased = np.zeros_like(straight)
    erased[4:11, [2, 7]] = 8
    assert np.array_equal(solve(straight), erased)
    # Same endpoint permutation and same tokens, different encounter scopes.
    # This rules out treating the network merely as an endpoint permutation.
    routing_rng, prior, found = np.random.default_rng(71623), {}, False
    for _ in range(100):
        drawing = _scaffold(routing_rng, 4, None)
        field, centers, start, end, dest, neighbors, encounters = drawing
        key = tuple(dest)
        if key in prior:
            other = prior[key]
            edge_changes = [(i, j) for i in range(4) for j in range(i+1, 4)
                            if (j in neighbors[i]) != (j in other[5][i])]
            if edge_changes:
                i, j = edge_changes[0]
                for canvas, cols, first, last, permutation, adjacent, _ in (drawing, other):
                    inp = np.array([0, 8, 5], dtype=np.int8)[canvas]
                    expected = inp.copy()
                    for t, c in enumerate(cols):
                        inp[first-3:first, c-1:c+2] = t+1
                        inp[first-2, c] = j+1 if t == i else 9
                    if j in adjacent[i]:
                        c, d = cols[i], cols[permutation[i]]
                        expected[first-3:first, c-1:c+2] = np.where(DIAMOND, i+1, 0)
                        expected[last+1:last+4, d-1:d+2] = np.where(CHECKER, 9, 0)
                    assert np.array_equal(solve(inp), expected)
                found = True
                break
        prior[key] = drawing
    assert found, 'missing same-permutation/different-scope regression'
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    damaged['train'][0]['output'][0, 0] = (int(damaged['train'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('nonterminal corruption accepted')
    return {'seeds': 42, 'pairs': len(metrics), 'shapes': len({m['shape'] for m in metrics}),
            'structures': len(structural_hashes), 'role_assignments': len({m['roles'] for m in metrics}),
            'permutations': len({m['permutation'] for m in metrics}),
            'neighbor_degree_patterns': len({m['degrees'] for m in metrics}),
            'routes': sorted({m['routes'] for m in metrics}),
            'survivors': sorted({m['survivors'] for m in metrics})}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
