"""Addressed docking: simultaneous color lookup selects a geometric destination.

Only Python's standard library and NumPy are used. solve() parses its input;
construction records are used solely to paint labels during generation.
"""
from __future__ import annotations

import json
import numpy as np

_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid of size 1..30")
    return a.astype(int, copy=False)


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask)))
    parts = []
    while todo:
        p = min(todo)
        todo.remove(p)
        part, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in _D4:
                q = r + dr, c + dc
                if q in todo:
                    todo.remove(q)
                    part.add(q)
                    stack.append(q)
        parts.append(part)
    return parts


def _paint(grid, cells, color):
    for p in cells:
        grid[p] = color


def _parse_fork(cells, grid):
    """Recognize arbitrary positive three-run prongs in all four directions."""
    parses = []
    for out in _D4:
        perp = -out[1], out[0]
        kmin = min(r * out[0] + c * out[1] for r, c in cells)
        qmin = min(r * perp[0] + c * perp[1] for r, c in cells)
        local = {(r * out[0] + c * out[1] - kmin,
                  r * perp[0] + c * perp[1] - qmin) for r, c in cells}
        width = max(q for _, q in local) + 1
        if {q for k, q in local if k == 0} != set(range(width)):
            continue
        lengths = [sum((k, q) in local for k in range(1, 31)) for q in range(width)]
        expected = {(0, q) for q in range(width)}
        expected |= {(k, q) for q, length in enumerate(lengths)
                     for k in range(1, length + 1)}
        if expected != local:
            continue
        runs, q = [], 0
        while q < width:
            if lengths[q] == 0:
                q += 1
                continue
            start, length = q, lengths[q]
            while q < width and lengths[q] == length:
                q += 1
            runs.append((start, q, length))
        if (len(runs) != 3 or runs[0][1] >= runs[1][0]
                or runs[1][1] >= runs[2][0] or runs[1][0] + runs[1][1] != width):
            continue
        q0, q1, center_length = runs[1]

        def cell(k, q):
            kk, qq = k + kmin, q + qmin
            return kk * out[0] + qq * perp[0], kk * out[1] + qq * perp[1]

        prong = {cell(k, q) for k in range(1, center_length + 1) for q in range(q0, q1)}
        neutral = {int(grid[p]) for p in cells - prong}
        address = {int(grid[p]) for p in prong}
        if len(neutral) != 1 or len(address) != 1 or neutral == address:
            continue
        parses.append(dict(cells=cells, prong=prong, out=out, perp=perp,
                           kmin=kmin, qmin=qmin, width=width, runs=tuple(runs),
                           q0=q0, span=q1-q0, stop=center_length+1,
                           structure=neutral.pop(), address=address.pop()))
    if len(parses) != 1:
        raise ValueError("receiver must have a unique base and three positive prongs")
    return parses[0]


def _cell(fork, k, q):
    kk, qq = k + fork['kmin'], q + fork['qmin']
    return (kk * fork['out'][0] + qq * fork['perp'][0],
            kk * fork['out'][1] + qq * fork['perp'][1])


def _geometric_candidate(fork, token, grid, background):
    local = {(r * fork['out'][0] + c * fork['out'][1] - fork['kmin'],
              r * fork['perp'][0] + c * fork['perp'][1] - fork['qmin'])
             for r, c in token['cells']}
    qs = {q for _, q in local}
    if qs != set(range(fork['q0'], fork['q0'] + fork['span'])):
        return None
    k0, k1 = min(k for k, _ in local), max(k for k, _ in local)
    if k0 <= fork['stop']:
        return None
    sweep = {_cell(fork, k, q) for k in range(fork['stop'], k1+1) for q in qs}
    h, w = grid.shape
    if any(not (0 <= r < h and 0 <= c < w)
           or ((r, c) not in token['cells'] and grid[r, c] != background) for r, c in sweep):
        return None
    length = k1-k0+1
    final = {_cell(fork, k, q) for k in range(fork['stop'], fork['stop']+length) for q in qs}
    shift = k0-fork['stop']
    center = (token['center'][0]-shift*fork['out'][0],
              token['center'][1]-shift*fork['out'][1])
    return dict(sweep=sweep, final=final, center=center, distance=shift, length=length)


def _interpret(input_grid):
    grid = _grid(input_grid)
    values, counts = np.unique(grid, return_counts=True)
    if sum(counts == counts.max()) != 1:
        raise ValueError("background must be uniquely most frequent")
    background = int(values[np.argmax(counts)])
    tokens, forks = [], []
    for cells in _components(grid != background):
        rows, cols = zip(*cells)
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        h, w = r1-r0+1, c1-c0+1
        if len(cells) == h*w:
            if h % 2 != 1 or w % 2 != 1 or h*w < 3:
                raise ValueError("tokens must be full odd rectangles with a body")
            center = (r0+r1)//2, (c0+c1)//2
            bodies = {int(grid[p]) for p in cells if p != center}
            if len(bodies) != 1:
                raise ValueError("all noncentral token pixels must share a body color")
            tokens.append(dict(cells=cells, center=center, body=bodies.pop(),
                               value=int(grid[center]), shape=(h, w)))
        else:
            forks.append(_parse_fork(cells, grid))
    if not tokens or not forks or len({f['structure'] for f in forks}) != 1:
        raise ValueError("input needs tokens and receivers with one scaffold color")
    structure = forks[0]['structure']
    if any(structure in (t['body'], t['value']) for t in tokens):
        raise ValueError("scaffold and token identities must be distinct")
    mapping = {}
    for token in tokens:
        body, value = token['body'], token['value']
        if body in mapping and mapping[body] != value:
            raise ValueError("same-body donors have conflicting centers")
        mapping[body] = value
    used, swept, matches = set(), set(), []
    for ti, token in enumerate(tokens):
        donors = [j for j, t in enumerate(tokens) if j != ti and t['body'] == token['value']]
        token['donors'] = donors
        token['resolved'] = tokens[donors[0]]['value'] if donors else None
        geometric = []
        for fi, fork in enumerate(forks):
            candidate = _geometric_candidate(fork, token, grid, background)
            if candidate is not None:
                geometric.append(dict(candidate, fi=fi))
        token['geometric'] = geometric
        eligible = [c for c in geometric if forks[c['fi']]['address'] == token['resolved']]
        if len(eligible) > 1:
            raise ValueError("more than one addressed destination for a token")
        if eligible:
            candidate = eligible[0]
            if candidate['fi'] in used or not swept.isdisjoint(candidate['sweep']):
                raise ValueError("competing assignments or intersecting swept paths")
            used.add(candidate['fi'])
            swept |= candidate['sweep']
            matches.append(dict(candidate, ti=ti, resolved=token['resolved']))
    return dict(grid=grid, background=background, structure=structure,
                forks=forks, tokens=tokens, matches=matches)


def _render(a, matches):
    output = a['grid'].copy()
    for token in a['tokens']:
        _paint(output, token['cells'], a['background'])
    for match in matches:
        token = a['tokens'][match['ti']]
        _paint(output, match['sweep']-match['final'], token['value'])
        _paint(output, match['final'], token['body'])
        output[match['center']] = match['resolved']
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _interpret(input_grid)
    return _render(a, a['matches'])


def _fork_pixels(span, lengths, gap, side, base, sign, address, structure):
    width = span+2*(gap+side)
    qmin = -(width//2)
    pixels = {(q, base): structure for q in range(qmin, qmin+width)}
    for q0, q1, length, color in (
        (qmin, qmin+side, lengths[0], structure),
        (-(span//2), span//2+1, lengths[1], address),
        (qmin+width-side, qmin+width, lengths[2], structure),
    ):
        for k in range(1, length+1):
            for q in range(q0, q1):
                pixels[q, base+sign*k] = color
    return pixels


def _module(rng, relation, address, background, structure, opposed=False,
            active=True, wrong_span=False, compact=False):
    """Local geometry oracle paints the expected motion before parsing exists."""
    body, value = relation
    span = 3 if compact or wrong_span else int(rng.choice([1, 3, 5]))
    token_span = 1 if wrong_span else span
    length = 3 if token_span == 1 or compact else int(rng.choice([1, 3, 5]))
    side = 1 if compact else int(rng.choice([1, 1, 2]))
    gap = 1 if compact else int(rng.choice([1, 1, 2]))
    lengths = (2, 1, 2) if compact else tuple(map(int, rng.integers(1, 5, size=3)))
    if not compact and rng.random() < .35:
        lengths = lengths[0], int(rng.integers(3, 6)), lengths[2]
    travel_gap = 4 if opposed else (0 if compact else int(rng.integers(0, 5)))
    start = max(lengths)+2+travel_gap
    pixels = _fork_pixels(span, lengths, gap, side, 0, 1, address, structure)
    final_pixels = dict(pixels)
    if opposed:
        other_lengths = (1, 2, 1) if compact else (int(rng.integers(1, 4)), 4, int(rng.integers(1, 4)))
        other_gap = 1 if compact else int(rng.integers(1, 3))
        base = start+length+max(other_lengths)+other_gap+1
        other = _fork_pixels(span, other_lengths, gap, side, base, -1, value, structure)
        assert set(pixels).isdisjoint(other)
        pixels.update(other)
        final_pixels.update(other)
    token_cells = {(q, k) for q in range(-(token_span//2), token_span//2+1)
                   for k in range(start, start+length)}
    pixels.update({p: body for p in token_cells})
    pixels[0, start+length//2] = value
    if active:
        stop = lengths[1]+1
        footprint = {(q, k) for q in range(-(token_span//2), token_span//2+1)
                     for k in range(stop, stop+length)}
        sweep = {(q, k) for q in range(-(token_span//2), token_span//2+1)
                 for k in range(stop, start+length)}
        final_pixels.update({p: value for p in sweep-footprint})
        final_pixels.update({p: body for p in footprint})
        final_pixels[0, stop+length//2] = address
    return _normalize_object(pixels, final_pixels, 0 if compact else int(rng.integers(4)))


def _normalize_object(pixels, output_pixels, rotation):
    def turn(p):
        r, c = p
        for _ in range(rotation):
            r, c = -c, r
        return r, c
    points = [turn(p) for p in pixels]
    r0, c0 = min(r for r, c in points), min(c for r, c in points)
    def shifted(source):
        result = {}
        for p, color in source.items():
            r, c = turn(p)
            result[r-r0, c-c0] = color
        return result
    inp, out = shifted(pixels), shifted(output_pixels)
    return dict(input=inp, output=out,
                shape=(max(r for r, c in inp)+1, max(c for r, c in inp)+1))


def _loose_token(rng, relation, compact):
    h, w = (3, 3) if compact else [(1, 3), (3, 1), (3, 3), (3, 5), (5, 3)][int(rng.integers(5))]
    body, value = relation
    pixels = {(r, c): body for r in range(h) for c in range(w)}
    pixels[h//2, w//2] = value
    return dict(input=pixels, output={}, shape=(h, w))


def _free_positions(blocked, height, width):
    h, w = blocked.shape
    if height > h or width > w:
        return np.empty((0, 2), dtype=int)
    summed = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    windows = (summed[height:, width:]-summed[:-height, width:]
               -summed[height:, :-width]+summed[:-height, :-width])
    return np.argwhere(windows == 0)


def _pack(rng, objects, background, compact=False, module_count=0):
    if compact:
        # A bounded fallback with distinct axes, not a visible panel convention.
        positions, r = [], 0
        for obj in objects[:module_count]:
            positions.append((r, 0))
            r += obj['shape'][0]+1
        c = 0
        for obj in objects[module_count:]:
            positions.append((r, c))
            c += obj['shape'][1]+2
        h = max(rr+obj['shape'][0] for (rr, cc), obj in zip(positions, objects))+1
        w = max(cc+obj['shape'][1] for (rr, cc), obj in zip(positions, objects))+1
        h, w = max(12, h), max(12, w)
        if h > 30 or w > 30:
            return None
    else:
        min_h = max(12, max(o['shape'][0] for o in objects))
        min_w = max(12, max(o['shape'][1] for o in objects))
        area = sum(o['shape'][0]*o['shape'][1] for o in objects)
        for _ in range(8):
            h, w = int(rng.integers(min_h, 31)), int(rng.integers(min_w, 31))
            if h*w >= area*1.35:
                break
        else:
            h = w = 30
        blocked = np.zeros((h, w), dtype=bool)
        positions = [None]*len(objects)
        order = sorted(range(len(objects)), key=lambda i: -np.prod(objects[i]['shape']))
        for i in order:
            bh, bw = objects[i]['shape']
            free = _free_positions(blocked, bh, bw)
            if len(free) == 0:
                return None
            r, c = map(int, free[int(rng.integers(len(free)))])
            positions[i] = r, c
            blocked[max(0, r-1):min(h, r+bh+1), max(0, c-1):min(w, c+bw+1)] = True
    inp = np.full((h, w), background, dtype=int)
    out = inp.copy()
    for obj, (r, c) in zip(objects, positions):
        for p, color in obj['input'].items():
            inp[r+p[0], c+p[1]] = color
        for p, color in obj['output'].items():
            out[r+p[0], c+p[1]] = color
    rotation = int(rng.integers(4))
    inp, out = np.rot90(inp, rotation), np.rot90(out, rotation)
    if rng.random() < .5:
        inp, out = np.fliplr(inp), np.fliplr(out)
    return dict(input=inp.copy(), output=out.copy())


def _make_pair(rng, mode, roles, varied_roles, mover_count=2, force_compact=False):
    palette = list(map(int, rng.permutation(10)))
    if not varied_roles:
        palette = list(roles)+list(map(int, rng.permutation([c for c in range(10) if c not in roles])))
    background, structure = palette[:2]
    identities = palette[2:]
    # The witness constraints specify graph properties, not a coordinate table.
    if mode == 'remote':
        rel = [(0, 1), (1, 2), (3, 1)]
        movers, loose, failed = [0], [1], 2
        address = {0: 2, 2: 2}
    elif mode == 'simple':
        n = int(rng.integers(1, 3))
        rel = [(i, 2) for i in range(n)]+[(2, 3), (4, 2)]
        movers, loose, failed = list(range(n)), [n, n+1], None
        address = {i: 3 for i in movers}
    else:
        n = mover_count
        rel = [(i, i+1) for i in range(n+1)]+[(n+2, 1)]
        movers, loose, failed = list(range(n)), [n, n+1], None
        address = {i: i+2 for i in movers}
        if n == 2 and (mode == 'chain' or rng.random() < .35):
            failed = len(rel)
            rel.append((n+3, n+4))  # Center identity has no body donor.
            address[failed] = n+4
    if rng.random() < .55:
        # Consistent repeated donors are evidence, never a nearest-token rule.
        duplicate = loose[0]
        loose.append(len(rel))
        rel.append(rel[duplicate])
    colored = [(identities[b], identities[v]) for b, v in rel]
    for attempt in ([44] if force_compact else range(45)):
        compact = attempt == 44
        objects = []
        for i in movers:
            opposed = i == 0 and mode in ('remote', 'combined')
            objects.append(_module(rng, colored[i], identities[address[i]], background,
                                   structure, opposed=opposed, compact=compact))
        if failed is not None:
            objects.append(_module(rng, colored[failed], identities[address[failed]],
                                   background, structure, active=False,
                                   wrong_span=mode == 'remote', compact=compact))
        module_count = len(objects)
        objects += [_loose_token(rng, colored[i], compact) for i in loose]
        pair = _pack(rng, objects, background, compact, module_count)
        if pair is None:
            continue
        try:
            a = _interpret(pair['input'])
            if (a['background'] != background or len(a['tokens']) != len(rel)
                    or len(a['matches']) != len(movers)
                    or not np.array_equal(solve(pair['input']), pair['output'])):
                continue
            f = _features(a)
            if mode == 'remote' and not (f['opposed_far'] and f['wrong_span'] and f['remote_donor']):
                continue
            if mode in ('chain', 'combined') and not f['moving_donor']:
                continue
            if mode == 'combined' and not f['opposed_far']:
                continue
        except ValueError:
            continue
        return pair
    raise RuntimeError("bounded compact fallback failed to realize the relation graph")


def _count(value, low, high, name):
    if (not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count(num_train, 2, 10, 'num_train'), _count(num_test, 1, 4, 'num_test')
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    roles = tuple(map(int, rng.choice(10, 2, replace=False)))
    varied_roles = bool(rng.random() < .7)
    max_movers = 3 if nt >= 3 and rng.random() < .5 else 2
    modes = [('remote', 1), ('chain', 2)]
    if max_movers == 3:
        modes.append(('chain', 3))
    while len(modes) < nt:
        modes.append((str(rng.choice(['remote', 'chain', 'simple'])), 2))
    rng.shuffle(modes)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ne)):
        for index in range(count):
            mode, n = modes[index] if split == 'train' else ('combined', int(rng.integers(2, max_movers+1)))
            for _ in range(12):
                pair = _make_pair(rng, mode, roles, varied_roles, n)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not make distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(a):
    tokens, forks, matches = a['tokens'], a['forks'], a['matches']
    moving = {m['ti']: m for m in matches}
    opposed_far = 0
    for m in matches:
        t, f = tokens[m['ti']], forks[m['fi']]
        opposed_far += any(forks[g['fi']]['out'] == tuple(-x for x in f['out'])
                           and g['distance'] < m['distance']
                           and forks[g['fi']]['address'] == t['value']
                           and f['address'] != t['value'] for g in t['geometric'])
    wrong_span = 0
    for i, t in enumerate(tokens):
        if not t['donors'] or i in moving:
            continue
        for f in forks:
            if f['address'] != t['resolved']:
                continue
            qs = [r*f['perp'][0]+c*f['perp'][1]-f['qmin'] for r, c in t['cells']]
            ks = [r*f['out'][0]+c*f['out'][1]-f['kmin'] for r, c in t['cells']]
            wrong_span += (min(qs)+max(qs) == 2*f['q0']+f['span']-1
                           and max(qs)-min(qs)+1 != f['span'] and min(ks) > f['stop'])
    return dict(
        shape=a['grid'].shape, roles=(a['background'], a['structure']),
        tokens=len(tokens), forks=len(forks), movers=len(matches),
        opposed_far=int(opposed_far), wrong_span=int(wrong_span),
        failed_lookup=sum(bool(t['geometric']) and not t['donors'] for t in tokens),
        compatible_erased=sum(bool(t['donors']) and i not in moving for i, t in enumerate(tokens)),
        remote_donor=sum(any(j not in moving for j in tokens[m['ti']]['donors']) for m in matches),
        moving_donor=sum(any(j in moving and moving[j]['resolved'] != tokens[j]['value']
                             for j in tokens[m['ti']]['donors']) for m in matches),
        repeated_bodies=len(tokens)-len({t['body'] for t in tokens}),
        directions=tuple(sorted({forks[m['fi']]['out'] for m in matches})),
        spans=tuple(sorted({forks[m['fi']]['span'] for m in matches})),
        lengths=tuple(sorted({m['length'] for m in matches})),
        distances=tuple(m['distance'] for m in matches),
        geometries=tuple((f['width'], tuple(v[2] for v in f['runs'])) for f in forks),
        token_shapes=tuple(t['shape'] for t in tokens),
        overlap_motion=sum(m['distance'] < m['length'] for m in matches),
    )


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, stats = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            a = _interpret(inp)
            assert np.array_equal(solve(inp), out) and not np.array_equal(inp, out)
            structures = set().union(*(f['cells'] for f in a['forks']))
            assert all(out[p] == inp[p] for p in structures), "receiver altered"
            moving = {m['ti']: m for m in a['matches']}
            sweeps = set().union(*(m['sweep'] for m in a['matches']))
            assert sweeps.isdisjoint(structures)
            allowed = structures | sweeps | set().union(*(t['cells'] for t in a['tokens']))
            assert all(out[r, c] == inp[r, c] for r in range(inp.shape[0]) for c in range(inp.shape[1])
                       if (r, c) not in allowed), "unrelated background altered"
            for m in a['matches']:
                token = a['tokens'][m['ti']]
                assert token['donors'] and a['forks'][m['fi']]['address'] == m['resolved']
                assert len(m['final']) == len(token['cells']) and m['distance'] >= 1
                assert out[m['center']] == m['resolved']
                assert all(out[p] == token['body'] for p in m['final']-{m['center']})
                assert all(out[p] == token['value'] for p in m['sweep']-m['final'])
                assert all(a['tokens'][j]['value'] == m['resolved'] for j in token['donors'])
            for i, token in enumerate(a['tokens']):
                if i not in moving:
                    assert all(out[p] == a['background'] for p in token['cells'])
            assert 1 <= len(a['matches']) <= 3 and len(a['tokens']) > len(a['matches'])
            stats[split].append(_features(a))
    train, test = stats['train'], stats['test']
    assert any(f['opposed_far'] and f['remote_donor'] and f['wrong_span'] for f in train)
    assert any(f['moving_donor'] and f['failed_lookup'] for f in train)
    assert all(f['opposed_far'] and f['moving_donor'] for f in test)
    assert max(f['movers'] for f in test) <= max(f['movers'] for f in train)
    assert all(f['compatible_erased'] for f in train+test)


def self_test() -> dict:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    rows = [_features(_interpret(pair['input'])) for p in puzzles
            for split in ('train', 'test') for pair in p[split]]
    signatures = {tuple(_key(pair['input']) for split in ('train', 'test') for pair in p[split]) for p in puzzles}
    shapes = {f['shape'] for f in rows}
    geometries = {g for f in rows for g in f['geometries']}
    assert len(signatures) == 42 and len(shapes) >= 35 and len(geometries) >= 80
    assert {1, 2, 3} <= {f['movers'] for f in rows}
    assert len({f['tokens'] for f in rows}) >= 4 and len({f['forks'] for f in rows}) >= 3
    assert {1, 3, 5} <= {s for f in rows for s in f['spans']}
    assert {1, 3, 5} <= {s for f in rows for s in f['lengths']}
    assert set(_D4) == {d for f in rows for d in f['directions']}
    assert len({f['roles'] for f in rows}) >= 25
    assert any(f['repeated_bodies'] for f in rows) and any(f['overlap_motion'] for f in rows)
    assert any(g[1][1] > max(g[1][0], g[1][2]) for g in geometries)
    assert any(g[1][0] != g[1][2] for g in geometries)
    ablations = {'without_geometry': 0, 'without_lookup': 0}
    for puzzle in puzzles:
        differing_a = differing_b = False
        for pair in puzzle['train']:
            a = _interpret(pair['input'])
            # Ablation A: apply the color relation in place to every token.
            color_only = pair['input'].copy()
            direct_matches = []
            for ti, token in enumerate(a['tokens']):
                if token['donors']:
                    color_only[token['center']] = token['resolved']
                else:
                    _paint(color_only, token['cells'], a['background'])
                # Ablation B: use original center as address, without donors.
                direct = [g for g in token['geometric']
                          if a['forks'][g['fi']]['address'] == token['value']]
                if len(direct) == 1:
                    direct_matches.append(dict(direct[0], ti=ti, resolved=token['value']))
            differing_a |= not np.array_equal(color_only, pair['output'])
            differing_b |= not np.array_equal(_render(a, direct_matches), pair['output'])
        assert differing_a and differing_b
        ablations['without_geometry'] += int(differing_a)
        ablations['without_lookup'] += int(differing_b)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            generate_puzzle(7100+nt*10+ne, nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    repeated = generate_puzzle(0)
    assert all(np.array_equal(a[k], b[k]) for split in ('train', 'test')
               for a, b in zip(puzzles[0][split], repeated[split]) for k in ('input', 'output'))
    # Symmetry and identity renaming must commute with the input-only solve.
    renaming = np.array([7, 2, 9, 0, 6, 1, 8, 4, 3, 5])
    for split in ('train', 'test'):
        for pair in puzzles[0][split]:
            inp, out = pair['input'], pair['output']
            for k in range(4):
                assert np.array_equal(solve(np.rot90(inp, k)), np.rot90(out, k))
            assert np.array_equal(solve(np.fliplr(inp)), np.fliplr(out))
            assert np.array_equal(solve(renaming[inp]), renaming[out])
    # Reject a plausible but incorrect donor update and a scaffold edit.
    for kind in ('center', 'scaffold'):
        bad = {s: [{k: np.asarray(q[k]).copy() for k in ('input', 'output')} for q in puzzles[0][s]]
               for s in ('train', 'test')}
        a = _interpret(bad['train'][0]['input'])
        point = a['matches'][0]['center'] if kind == 'center' else min(a['forks'][0]['cells'])
        bad['train'][0]['output'][point] = a['background']
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("semantic corruption accepted")
    _equality_fixtures()
    fallback_rng = np.random.default_rng(99127)
    for mode, n in (('remote', 1), ('chain', 2), ('chain', 3),
                    ('combined', 2), ('combined', 3), ('simple', 2)):
        pair = _make_pair(fallback_rng, mode, (0, 9), False, n, force_compact=True)
        assert np.array_equal(solve(pair['input']), pair['output'])
    return dict(seeds=42, pairs=len(rows), shapes=len(shapes), geometries=len(geometries),
                roles=len({f['roles'] for f in rows}),
                movers=sorted({f['movers'] for f in rows}),
                tokens=sorted({f['tokens'] for f in rows}),
                receivers=sorted({f['forks'] for f in rows}), ablation_witnesses=ablations)


def _equality_fixtures():
    """Exercise allowed equality patterns and errors beyond random recipes."""
    rng = np.random.default_rng(80801)
    bg, scaffold = 0, 9

    def case(mover_relations, addresses, loose_relations, active=True, opposed=False):
        modules = [_module(rng, relation, address, bg, scaffold, active=active,
                           compact=True, opposed=opposed)
                   for relation, address in zip(mover_relations, addresses)]
        loose = [_loose_token(rng, relation, True) for relation in loose_relations]
        pair = _pack(rng, modules+loose, bg, compact=True, module_count=len(modules))
        assert pair is not None
        return pair

    fixtures = [
        case([(2, 3), (3, 2)], [2, 3], []),  # Mutual lookup produces two solid tokens.
        case([(2, 3)], [3], [(3, 3)]),  # Monochrome donor, unchanged final center.
        case([(2, 3), (5, 3)], [4, 4], [(3, 4), (3, 4)]),  # Shared/repeated donor.
        case([(2, 2)], [2], [], active=False),  # A token cannot supply itself.
    ]
    for pair in fixtures:
        original = pair['input'].copy()
        assert np.array_equal(solve(pair['input']), pair['output'])
        assert np.array_equal(pair['input'], original), "solve modified its input"
    mutual = _interpret(fixtures[0]['input'])
    assert len(mutual['matches']) == 2
    assert all(m['resolved'] == mutual['tokens'][m['ti']]['body'] for m in mutual['matches'])
    assert not _interpret(fixtures[3]['input'])['matches']
    # Tight crop makes objects reach the boundary without clipping their support.
    boundary = fixtures[1]
    rr, cc = np.where(boundary['input'] != bg)
    slices = np.s_[min(rr):max(rr)+1, min(cc):max(cc)+1]
    assert np.array_equal(solve(boundary['input'][slices]), boundary['output'][slices])
    conflicting = case([(2, 3)], [4], [(3, 4), (3, 5)])
    ambiguous = case([(2, 3)], [3], [(3, 3)], opposed=True)
    for pair in (conflicting, ambiguous):
        try:
            solve(pair['input'])
        except ValueError:
            pass
        else:
            raise AssertionError("ambiguous relation/destination accepted")
    malformed = fixtures[1]['input'].copy()
    prong = _interpret(malformed)['forks'][0]['prong']
    _paint(malformed, prong, bg)
    try:
        solve(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("zero-length middle prong accepted")


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
