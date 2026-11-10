"""Census transfer: count source objects, compose routes, pool destination tallies.

All semantic information is recovered from the input. Sampling metadata is used
only to build labels and then checked against the visible-grid interpretation.
"""
import heapq
import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
DIAMOND = ((0, 1), (1, 0), (1, 2), (2, 1))
SQUARE = ((0, 0), (0, 1), (1, 0), (1, 1))


def _components(cells, directions=D8):
    remaining = set(cells)
    parts = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        part, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in directions:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        parts.append(part)
    return parts


def _normalize(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a


def _key(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _panel(a):
    candidates = []
    for field in np.unique(a):
        rr, cc = np.where(a == field)
        r0, r1 = int(rr.min()), int(rr.max())+1
        c0, c1 = int(cc.min()), int(cc.max())+1
        p = a[r0:r1, c0:c1]
        if min(p.shape) < 5 or not all(np.all(e == field) for e in
                                      (p[0], p[-1], p[:, 0], p[:, -1])):
            continue
        pos = np.argwhere(p != field)
        if not 2 <= len(pos) <= 6:
            continue
        colors = tuple(int(p[r, c]) for r, c in pos)
        if len(set(colors)) != len(colors):
            continue
        if len(set(pos[:, 1])) == 1 and np.all(np.diff(pos[:, 0]) >= 2):
            inset = int(pos[0, 1])
            if 2*inset == p.shape[1]-1:
                continue
            step = (0, 1 if 2*inset < p.shape[1]-1 else -1)
        elif len(set(pos[:, 0])) == 1 and np.all(np.diff(pos[:, 1]) >= 2):
            inset = int(pos[0, 0])
            if 2*inset == p.shape[0]-1:
                continue
            step = (1 if 2*inset < p.shape[0]-1 else -1, 0)
        else:
            continue
        outside = np.ones(a.shape, bool)
        outside[r0:r1, c0:c1] = False
        if not outside.any():
            continue
        values, freq = np.unique(a[outside], return_counts=True)
        bg = int(values[np.argmax(freq)])
        if freq.max()*2 <= outside.sum() or bg in colors or bg == field:
            continue
        halo = np.zeros(a.shape, bool)
        halo[max(0, r0-1):r1+1, max(0, c0-1):c1+1] = True
        halo[r0:r1, c0:c1] = False
        if not np.all(a[halo] == bg):
            continue
        candidates.append(dict(field=int(field), bg=bg, bounds=(r0, r1, c0, c1),
                               panel=p.copy(), positions=pos, keys=colors,
                               step=step, outside=outside))
    assert len(candidates) == 1, 'A unique bordered census panel is required'
    return candidates[0]


def _support(center, kind):
    r, c = center
    return {(r+dr, c+dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if kind == 'square' or dr == 0 or dc == 0}


def _line_graph(cells):
    # Diagonals are edges unless an orthogonal pixel provides the corner path.
    return {p: {q for dr, dc in D8
                if (q := (p[0]+dr, p[1]+dc)) in cells
                and not (dr and dc and ((p[0]+dr, p[1]) in cells
                                       or (p[0], p[1]+dc) in cells))}
            for p in cells}


def _parse_exterior(a, meta, line):
    outside, bg = meta['outside'], meta['bg']
    parts = _components(map(tuple, np.argwhere(outside & (a != bg) & (a != line))))
    tokens, objects = [], []
    for cells in parts:
        colors = {int(a[p]) for p in cells}
        if len(colors) == 1:
            objects.append((colors.pop(), cells))
            continue
        assert len(colors) == 2 and len(cells) in (5, 9), 'Malformed relation token'
        found = []
        for center in sorted(cells):
            for kind in ('square', 'cross'):
                if cells != _support(center, kind):
                    continue
                body = {int(a[p]) for p in cells if p != center}
                if len(body) == 1 and int(a[center]) not in body:
                    found.append(dict(body=body.pop(), value=int(a[center]),
                                      center=center, kind=kind, cells=cells))
        assert len(found) == 1, 'Relation body and center must be visible'
        tokens += found
    assert 3 <= len(tokens) <= 12
    line_cells = set(map(tuple, np.argwhere(outside & (a == line))))
    graph = _line_graph(line_cells)
    assert graph and all(len(v) in (1, 2) for v in graph.values()), 'Line branches or isolates'
    ends = {p for p, v in graph.items() if len(v) == 1}
    lines = _components(line_cells)
    assert 1 <= len(lines) <= 3
    assert all(len(p) >= 3 and len(p & ends) == 2 for p in lines)
    mapping = {}
    active = []
    for t in tokens:
        b, v = t['body'], t['value']
        assert b not in mapping or mapping[b] == v, 'Conflicting relations'
        mapping[b] = v
        r, c = t['center']
        contacts = {(r+2*dr, c+2*dc) for dr, dc in D4} & ends
        assert len(contacts) <= 1
        t['ends'] = contacts
        t['active'] = bool(contacts)
        if contacts:
            active.append(b)
    assert active and len(active) == len(set(active)), 'Active source bodies must be distinct'
    assert all(sum(p in t['ends'] for t in tokens) <= 1 for p in ends)
    # Different non-line components have a blank halo by their eight-connectivity.
    counts = {}
    for color, cells in objects:
        counts[color] = counts.get(color, 0)+1
    assert all(counts.get(b, 0) >= 1 for b in active)
    destinations, arrivals = {}, {k: 0 for k in meta['keys']}
    for t in tokens:
        if not t['active']:
            continue
        donors = [u for u in tokens if u is not t and u['body'] == t['value']]
        if not donors:
            destinations[t['body']] = None
            continue
        destination = donors[0]['value']
        destinations[t['body']] = destination
        if destination in arrivals:
            arrivals[destination] += counts[t['body']]
    assert any(arrivals.values()), 'At least one arrival must be represented'
    meta.update(line=line, tokens=tokens, objects=objects, graph=graph, lines=lines,
                mapping=mapping, counts=counts, active=tuple(active),
                destinations=destinations, arrivals=arrivals)
    return meta


def _interpret(input_grid):
    a = _grid(input_grid)
    panel = _panel(a)
    candidates = []
    for line in sorted(set(map(int, a[panel['outside']]))-{panel['bg']}):
        if line in panel['keys']:
            continue
        try:
            candidates.append(_parse_exterior(a, dict(panel), line))
        except AssertionError:
            continue
    assert len(candidates) == 1, 'Line and relation roles must be unique'
    return candidates[0]


def _draw_chart(meta, arrivals):
    out = np.full(meta['panel'].shape, meta['field'], np.int8)
    dr, dc = meta['step']
    for (r, c), color in zip(meta['positions'], meta['keys']):
        for i in range(arrivals.get(color, 0)):
            rr, cc = int(r)+2*i*dr, int(c)+2*i*dc
            assert 0 < rr < out.shape[0]-1 and 0 < cc < out.shape[1]-1, 'Tally exceeds panel'
            out[rr, cc] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read original relations, transfer each active census once, and sum arrivals."""
    meta = _interpret(input_grid)
    return _draw_chart(meta, meta['arrivals'])


def _halo(cells, shape, radius=1):
    h, w = shape
    return {(r+dr, c+dc) for r, c in cells for dr in range(-radius, radius+1)
            for dc in range(-radius, radius+1) if 0 <= r+dr < h and 0 <= c+dc < w}


def _route(rng, shape, blocked, start, end):
    if start == end or start in blocked or end in blocked:
        return None
    costs = rng.uniform(1, 1.7, shape)
    distances, previous = {start: 0.}, {}
    queue = [(0., start)]
    while queue:
        cost, p = heapq.heappop(queue)
        if cost != distances[p]:
            continue
        if p == end:
            path = [p]
            while p != start:
                p = previous[p]
                path.append(p)
            return path[::-1] if len(path) >= 3 else None
        for dr, dc in D8:
            q = p[0]+dr, p[1]+dc
            if not (0 <= q[0] < shape[0] and 0 <= q[1] < shape[1]) or q in blocked:
                continue
            new = cost+float(costs[q])
            if new < distances.get(q, float('inf')):
                distances[q] = new
                previous[q] = p
                heapq.heappush(queue, (new, q))
    return None


def _object_shape(rng):
    prototypes = (DIAMOND, SQUARE, ((0, 0),),
                  ((0, 0), (1, 1), (2, 2)),
                  ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2)),
                  ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)))
    if rng.random() < .65:
        cells = prototypes[int(rng.integers(len(prototypes)))]
    else:
        side = int(rng.integers(2, 5))
        cells = {(side//2, side//2)}
        target = int(rng.integers(2, min(10, side*side)+1))
        while len(cells) < target:
            choices = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in D8
                              if 0 <= r+dr < side and 0 <= c+dc < side}-cells)
            cells.add(choices[int(rng.integers(len(choices)))])
    for _ in range(int(rng.integers(4))):
        cells = {(-c, r) for r, c in cells}
    if rng.random() < .5:
        cells = {(r, -c) for r, c in cells}
    return _normalize(cells)


def _relations(rng, mode, regime, test):
    if mode == 'pool':
        mapping = {0: 2, 1: 3, 2: 4, 3: 4, 4: 5, 6: 2}
        active, keys = [0, 1], [4, 5]
        counts = {0: 3, 1: int(rng.integers(1, 3)), 6: 1, 4: 1}
    elif mode == 'chain':
        mapping = {0: 1, 1: 2, 2: 3, 3: 4, 5: 2}
        active, keys = [0, 1, 2, 3], [2, 3, 4]
        counts = {0: 1, 1: int(rng.integers(2, 4)), 2: 2, 3: 1, 5: 1}
    else:
        # Random directed graphs, including convergences and short cycles.
        bodies = list(map(int, rng.choice(7, int(rng.integers(5, 8)), replace=False)))
        mapping = {b: int(rng.choice([v for v in range(7) if v != b])) for b in bodies}
        active = list(map(int, rng.choice(bodies, 4 if test and regime == 2 else
                                         int(rng.integers(2, 5 if test else 4)), replace=False)))
        if test and regime == 2:
            for b in active:
                mapping[b] = int(rng.choice([v for v in bodies if v != b]))
        valid = [b for b in active if mapping[b] in mapping]
        if not valid:
            mapping[active[0]] = next(b for b in bodies if b != active[0])
            valid = [active[0]]
        destinations = sorted({mapping[mapping[b]] for b in valid})
        k = int(rng.integers(4, 6)) if test and regime == 1 else int(rng.integers(2, 4))
        keys = list(map(int, rng.choice(destinations, min(k, len(destinations)), replace=False)))
        others = [v for v in range(7) if v not in keys]
        rng.shuffle(others)
        keys += others[:k-len(keys)]
        counts = {b: int(rng.integers(1, 4)) for b in active}
        for b in rng.choice([b for b in bodies if b not in active],
                            int(rng.integers(1, min(2, len(bodies)-len(active))+1)), replace=False):
            counts[int(b)] = int(rng.integers(1, 3))
        # Keep the scene packable and give training a bounded count range.
        while sum(counts.values()) > 12:
            b = max(counts, key=counts.get)
            counts[b] -= 1
        if not test:
            for z in keys:
                contributing = [b for b in active if mapping[b] in mapping
                                and mapping[mapping[b]] == z]
                while sum(counts[b] for b in contributing) > 5:
                    b = max(contributing, key=counts.get)
                    counts[b] -= 1
    if test and regime == 0:
        # Same convergence mechanism, larger combined population.
        mapping = {0: 2, 1: 3, 2: 4, 3: 4, 4: 5, 6: 2}
        active, keys = [0, 1], [4, 5, 2]
        counts = {0: int(rng.integers(4, 6)), 1: int(rng.integers(2, 4)), 6: 1, 4: 1}
    rng.shuffle(keys)
    arrivals = {z: sum(counts.get(b, 0) for b in active if mapping[b] in mapping
                       and mapping[mapping[b]] == z) for z in keys}
    return mapping, active, keys, counts, arrivals


def _make_pair(rng, mode, regime, test, structural):
    mapping, active, keys, counts, arrivals = _relations(rng, mode, regime, test)
    identities = list(map(int, rng.permutation(structural[3:])))
    bg, field, line = structural[:3]
    # Only a token's two-color support distinguishes it from plain populations.
    bodies = list(mapping)
    if mode != 'pool' and rng.random() < .35:
        bodies.append(int(rng.choice([b for b in mapping if b not in active])))
    kinds = [str(rng.choice(['square', 'cross'])) for _ in bodies]
    inset, top, bottom = int(rng.integers(1, 3)), int(rng.integers(1, 3)), int(rng.integers(1, 3))
    rows = np.cumsum([top]+list(map(int, rng.integers(2, 4, len(keys)-1)))).tolist()
    ph = rows[-1]+bottom+1
    pw = max(2*inset+3, inset+2*(max(arrivals.values())-1)+int(rng.integers(2, 5)))
    objects = []
    for b, n in counts.items():
        for j in range(n):
            shape = (DIAMOND, SQUARE, ((0, 0),))[j] if mode == 'pool' and b == 0 and j < 3 else _object_shape(rng)
            objects.append((b, shape))
    rng.shuffle(objects)
    objects.sort(key=lambda p: -(max(r for r, c in p[1])+2)*(max(c for r, c in p[1])+2))
    for attempt in range(60):
        lower = max(18, int(np.sqrt(ph*pw+25*len(bodies)+18*len(objects))))
        h = int(rng.integers(max(ph+3, min(29, lower-2)), 31))
        w = int(rng.integers(max(pw+3, min(29, lower-2)), 31))
        if attempt >= 30:
            h = w = 30
        pr = int(rng.integers(h-ph+1))
        pc = int(rng.integers(w-pw+1))
        panel_cells = {(r, c) for r in range(pr, pr+ph) for c in range(pc, pc+pw)}
        blocked = _halo(panel_cells, (h, w))
        centers, supports = [], []
        for kind in kinds:
            choices = [(r, c) for r in range(1, h-1) for c in range(1, w-1)
                       if not (_support((r, c), kind) & blocked)
                       and all(max(abs(r-a), abs(c-b)) >= 4 for a, b in centers)]
            if not choices:
                break
            p = choices[int(rng.integers(len(choices)))]
            centers.append(p)
            support = _support(p, kind)
            supports.append(support)
            blocked.update(_halo(support, (h, w)))
        if len(centers) != len(bodies):
            continue
        indices = [bodies.index(b) for b in active]
        rng.shuffle(indices)
        groups = [indices[i:i+2] for i in range(0, len(indices), 2)]
        lines = set()
        good = True
        token_cells = set().union(*supports)
        for group in groups:
            success = False
            for _ in range(16):
                stubs = []
                for i in group:
                    r, c = centers[i]
                    choices = [(r+2*dr, c+2*dc) for dr, dc in D4]
                    choices = [p for p in choices if 0 <= p[0] < h and 0 <= p[1] < w
                               and p not in token_cells and p not in _halo(panel_cells, (h, w))
                               and p not in _halo(lines, (h, w))]
                    if not choices:
                        break
                    stubs.append(choices[int(rng.integers(len(choices)))])
                if len(stubs) != len(group):
                    continue
                if len(stubs) == 1:
                    free = [(r, c) for r in range(h) for c in range(w) if (r, c) not in blocked
                            and (r, c) not in _halo(lines, (h, w))]
                    if not free:
                        continue
                    stubs.append(free[int(rng.integers(len(free)))])
                obstacles = blocked | _halo(lines, (h, w))
                obstacles = obstacles-set(stubs)
                path = _route(rng, (h, w), obstacles, *stubs)
                if path is None:
                    continue
                ends = {path[0], path[-1]}
                attached = {i for i, (r, c) in enumerate(centers)
                            if any((r+2*dr, c+2*dc) in ends for dr, dc in D4)}
                if attached != set(group):
                    continue
                lines.update(path)
                success = True
                break
            if not success:
                good = False
                break
        if not good:
            continue
        if mode == 'pool':
            # Put compatible source 6 beside the interior, never at an endpoint.
            i = bodies.index(6)
            graph = _line_graph(lines)
            ends = {p for p, adj in graph.items() if len(adj) == 1}
            near = sorted({(r+2*dr, c+2*dc) for r, c in lines-ends for dr, dc in D4})
            choices = [p for p in near if 1 <= p[0] < h-1 and 1 <= p[1] < w-1
                       and not (_support(p, kinds[i]) & (lines | _halo(panel_cells, (h, w))))
                       and all(j == i or not (_support(p, kinds[i]) & _halo(s, (h, w)))
                               for j, s in enumerate(supports))
                       and not any((p[0]+2*dr, p[1]+2*dc) in ends for dr, dc in D4)]
            if not choices:
                continue
            centers[i] = choices[int(rng.integers(len(choices)))]
            supports[i] = _support(centers[i], kinds[i])
        a = np.full((h, w), bg, np.int8)
        a[pr:pr+ph, pc:pc+pw] = field
        for r, b in zip(rows, keys):
            a[pr+r, pc+inset] = identities[b]
        for p in lines:
            a[p] = line
        for b, p, cells in zip(bodies, centers, supports):
            for cell in cells:
                a[cell] = identities[b]
            a[p] = identities[mapping[b]]
        blocked = _halo(panel_cells | lines | set().union(*supports), (h, w))
        for b, shape in objects:
            sh, sw = max(r for r, c in shape)+1, max(c for r, c in shape)+1
            choices = [(r, c) for r in range(h-sh+1) for c in range(w-sw+1)
                       if all((r+dr, c+dc) not in blocked for dr, dc in shape)]
            if not choices:
                good = False
                break
            r, c = choices[int(rng.integers(len(choices)))]
            cells = {(r+dr, c+dc) for dr, dc in shape}
            for p in cells:
                a[p] = identities[b]
            blocked.update(_halo(cells, (h, w)))
        if not good:
            continue
        out = np.full((ph, pw), field, np.int8)
        for r, b in zip(rows, keys):
            for i in range(arrivals[b]):
                out[r, inset+2*i] = identities[b]
        rotation = int(rng.integers(4))
        a, out = np.rot90(a, rotation), np.rot90(out, rotation)
        if rng.random() < .5:
            a, out = np.fliplr(a), np.fliplr(out)
        a, out = a.copy(), out.copy()
        try:
            assert np.array_equal(solve(a), out)
        except AssertionError:
            continue
        return dict(input=a, output=out)
    raise RuntimeError('Bounded placement could not build an unambiguous census')


def _counts(num_train, num_test):
    for name, v, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not low <= v <= high:
            raise ValueError(name+' must be an integer in its legal range')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    stable = bool(rng.random() < .25)
    palette = list(map(int, rng.permutation(10)))
    result, seen = {'train': [], 'test': []}, set()
    modes = ['pool', 'chain']+['random']*(nt-2)
    rng.shuffle(modes)
    for split, n in (('train', nt), ('test', nq)):
        for i in range(n):
            for _ in range(8):
                structural = palette if stable else list(map(int, rng.permutation(10)))
                mode = modes[i] if split == 'train' else 'random'
                pair = _make_pair(rng, mode, regime, split == 'test', structural)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('Could not generate a distinct input')
            seen.add(key)
            result[split].append(pair)
    validate_generated_puzzle(result)
    return result


def _features(a):
    m = _interpret(a)
    selected = [t for t in m['tokens'] if t['active']]
    pooled = any(sum(z == k for z in m['destinations'].values()) >= 2 for k in m['keys'])
    diagonal = any(len(_components(cells, D4)) > 1 for color, cells in m['objects']
                   if color in m['destinations'] and m['destinations'][color] in m['keys'])
    mixed = any(len({len(cells) for color, cells in m['objects'] if color == b}) > 1
                for b in m['active'])
    interior = any(not t['active'] and m['mapping'].get(t['value']) in m['keys']
                   and m['counts'].get(t['body'], 0) > 0
                   and any((t['center'][0]+2*dr, t['center'][1]+2*dc) in m['graph']
                           for dr, dc in D4) for t in m['tokens'])
    simultaneous = any(t['value'] in m['active']
                       and m['mapping'].get(m['mapping'].get(t['value'])) is not None
                       and m['mapping'][t['value']] != m['mapping'][m['mapping'][t['value']]]
                       and m['destinations'][t['body']] in m['keys'] for t in selected)
    m.update(pooled=pooled, diagonal=diagonal, mixed=mixed, interior=interior,
             simultaneous=simultaneous, shape=a.shape,
             glyphs=tuple(sorted({_normalize(c) for _, c in m['objects']})))
    return m


def _ablations(m):
    # Omit B by directly counting the destination-key colors (A's original rule).
    direct = {k: m['counts'].get(k, 0) for k in m['keys']}
    # Omit A by giving each successful active relation a unit weight.
    unit = {k: sum(z == k for z in m['destinations'].values()) for k in m['keys']}
    # Ignore endpoints, but retain the correct weighted two-hop relation rule.
    all_routes = {k: 0 for k in m['keys']}
    for b, v in m['mapping'].items():
        z = m['mapping'].get(v)
        if z in all_routes:
            all_routes[z] += m['counts'].get(b, 0)
    one_hop = {k: sum(m['counts'][b] for b in m['active'] if m['mapping'][b] == k)
               for k in m['keys']}
    return direct, unit, all_routes, one_hop


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, train, test = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(a) not in seen
            seen.add(_key(a))
            m = _features(a)
            assert np.array_equal(out, _draw_chart(m, m['arrivals']))
            assert not np.array_equal(a, out)
            assert all(t['body'] != t['value'] for t in m['tokens'])
            assert m['field'] not in {m['bg'], m['line']}
            assert m['line'] not in m['mapping'] and m['line'] not in m['keys']
            (train if split == 'train' else test).append(m)
    assert any(m['pooled'] and m['diagonal'] and m['mixed'] and m['interior'] for m in train)
    assert any(m['simultaneous'] and None in m['destinations'].values() for m in train)
    assert any(0 in m['arrivals'].values() for m in train)
    assert all(any(alt != m['arrivals'] for m in train for alt in [_ablations(m)[i]]) for i in range(4))
    # Every test shifts at least one demonstrated structural or quantitative axis.
    max_count = max(max(m['arrivals'].values()) for m in train)
    max_keys = max(len(m['keys']) for m in train)
    max_success = max(sum(z is not None for z in m['destinations'].values()) for m in train)
    assert all(max(m['arrivals'].values()) > max_count or len(m['keys']) > max_keys
               or sum(z is not None for z in m['destinations'].values()) > max_success for m in test)


def self_test() -> dict:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    metrics = [_features(p['input']) for q in puzzles for s in ('train', 'test') for p in q[s]]
    assert len({tuple(_key(p['input']) for s in ('train', 'test') for p in q[s]) for q in puzzles}) == 42
    assert all(np.array_equal(a[k], b[k]) for s in ('train', 'test')
               for a, b in zip(puzzles[0][s], generate_puzzle(0)[s]) for k in ('input', 'output'))
    assert len({m['shape'] for m in metrics}) >= 25
    assert len({m['panel'].shape for m in metrics}) >= 25
    assert len({g for m in metrics for g in m['glyphs']}) >= 40
    assert {m['step'] for m in metrics} == set(D4)
    assert {m['field'] for m in metrics} == set(range(10))
    assert {m['bg'] for m in metrics} == set(range(10))
    assert {m['line'] for m in metrics} == set(range(10))
    assert {2, 3, 4, 5} <= {len(m['keys']) for m in metrics}
    assert len({max(m['arrivals'].values()) for m in metrics}) >= 6
    assert {1, 2} <= {len(m['lines']) for m in metrics}
    assert any(len(m['mapping']) < len(m['tokens']) for m in metrics)
    assert any(m['destinations'].get(b) == b for m in metrics for b in m['active'])
    for counts in ((2, 1), (10, 4), (6, 3)):
        q = generate_puzzle(500+sum(counts), *counts)
        assert (len(q['train']), len(q['test'])) == counts
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4., 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal count accepted')
    bad = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in puzzles[0][s]]
           for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Corrupt output accepted')
    return {'seeds': 42, 'pairs': len(metrics),
            'input_shapes': len({m['shape'] for m in metrics}),
            'panel_shapes': len({m['panel'].shape for m in metrics}),
            'object_shapes': len({g for m in metrics for g in m['glyphs']}),
            'key_counts': sorted({len(m['keys']) for m in metrics}),
            'largest_tallies': sorted({max(m['arrivals'].values()) for m in metrics})}
