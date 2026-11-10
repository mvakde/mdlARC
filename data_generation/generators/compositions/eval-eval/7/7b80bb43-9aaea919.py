"""Damaged accumulator circuits. No parent code or saved examples are imported.

Only NumPy and the standard library are required. Generation constructs a circuit
and its arithmetic answer; solve independently reads ports, tokens and wiring.
No files are written by this module, including by self_test().
"""
from itertools import product
import heapq
import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= x <= 30 for x in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _at(a, p):
    return 0 <= p[0] < a.shape[0] and 0 <= p[1] < a.shape[1] and bool(a[p])


def _components(mask):
    labels = np.full(mask.shape, -1, dtype=np.int16)
    groups = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if labels[r, c] >= 0:
            continue
        label, todo, cells = len(groups), [(r, c)], []
        labels[r, c] = label
        while todo:
            p = todo.pop()
            cells.append(p)
            for dr, dc in _DIRS:
                q = p[0] + dr, p[1] + dc
                if _at(mask, q) and labels[q] < 0:
                    labels[q] = label
                    todo.append(q)
        groups.append(cells)
    return labels, groups


def _strong(mask):
    result = np.zeros_like(mask)
    for a, b in ((mask, result), (mask.T, result.T)):
        triples = a[:, :-2] & a[:, 1:-1] & a[:, 2:]
        for k in range(3):
            b[:, k:k + triples.shape[1]] |= triples
    return result


def _gaps(mask):
    gaps = []
    for dr, dc in ((0, 1), (1, 0)):
        a = mask if dc else mask.T
        hits = a[:, :-4] & a[:, 4:] & ~(a[:, 1:-3] | a[:, 2:-2] | a[:, 3:-1])
        for y, x in np.argwhere(hits):
            r, c = (int(y), int(x)) if dc else (int(x), int(y))
            gaps.append((r, c, dr, dc))
    return gaps


def _restore(mask):
    """Simultaneous geometric repair; unsupported gaps never bootstrap repairs."""
    strong = _strong(mask)
    added, protected = np.zeros_like(mask), np.zeros_like(mask)
    repairs = []
    for r, c, dr, dc in _gaps(mask):
        if not (_at(mask, (r - dr, c - dc)) or _at(mask, (r + 5*dr, c + 5*dc))):
            continue
        matches = []
        for end in (0, 4):
            ar, ac, toward = r + end*dr, c + end*dc, 1 if end == 0 else -1
            for bend in (-1, 1):
                sr, sc = toward*dr + bend*dc, toward*dc + bend*dr
                cells = [(ar + k*sr, ac + k*sc) for k in (1, 2)]
                if (all(_at(mask, p) and not strong[p] for p in cells)
                        and not _at(mask, (ar + 3*sr, ac + 3*sc))):
                    matches.append(cells)
        if matches:
            assert len(matches) == 1, 'ambiguous repair cue'
            for k in (1, 2, 3):
                added[r + k*dr, c + k*dc] = True
            protected[r, c] = protected[r + 4*dr, c + 4*dc] = True
            repairs.append(((r, c, dr, dc), matches[0]))
    weak = mask & ~_strong(mask | added) & ~protected
    oblique = np.zeros_like(mask)
    for dc in (-1, 1):
        a, b = (weak[:-1, :-1], weak[1:, 1:]) if dc == 1 else (weak[:-1, 1:], weak[1:, :-1])
        links = a & b
        if dc == 1:
            oblique[:-1, :-1] |= links
            oblique[1:, 1:] |= links
        else:
            oblique[:-1, 1:] |= links
            oblique[1:, :-1] |= links
    return (mask & ~oblique) | added, repairs


def _rings(mask):
    return [(r, c) for r in range(1, mask.shape[0] - 1)
            for c in range(1, mask.shape[1] - 1)
            if not mask[r, c] and int(mask[r-1:r+2, c-1:c+2].sum()) == 8]


def _parse(input_grid):
    grid = _grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    assert np.count_nonzero(counts == counts.max()) == 1
    wires = [int(c) for c in colors if c != bg and _rings(grid == c)]
    assert len(wires) == 1, 'one circuit color, identified by a hollow socket'
    wire = wires[0]
    raw = grid == wire
    clean, repairs = _restore(raw)
    _, payloads = _components((grid != bg) & ~raw)
    candidates = []
    # A terminal points away from its stem. Ring centers are empty; cap centers
    # are occupied. Long graph edges cannot impersonate three-cell end caps.
    for kind, centers in (('collector', _rings(clean)), ('source', np.argwhere(clean).tolist())):
        for r, c in centers:
            r, c = int(r), int(c)
            for nr, nc in _DIRS:
                tr, tc = -nc, nr
                if kind == 'source':
                    ends = [(r + s*tr, c + s*tc) for s in (-1, 1)]
                    if not all(_at(clean, p) and sum(_at(clean, (p[0]+dr, p[1]+dc)) for dr, dc in _DIRS) == 1 for p in ends):
                        continue
                    if _at(clean, (r + nr, c + nc)):
                        continue
                    base, anchor = 2, (r, c)
                    stem = (1, 2, 3)
                else:
                    if any(_at(clean, (r + 2*dr, c + 2*dc)) for dr, dc in _DIRS if (dr, dc) != (-nr, -nc)):
                        continue
                    base, anchor = 3, (r - nr, c - nc)
                    stem = (2, 3)
                if not all(_at(clean, (r - k*nr, c - k*nc)) for k in stem):
                    continue
                tokens = []
                for index, cells in enumerate(payloads):
                    uv = [((rr-r)*nr + (cc-c)*nc, (rr-r)*tr + (cc-c)*tc) for rr, cc in cells]
                    us, vs = [p[0] for p in uv], [p[1] for p in uv]
                    if min(us) < base or min(vs) not in (-1, 0) or max(vs) != -min(vs):
                        continue
                    if max(us)-min(us)+1 > 2 or max(vs)-min(vs)+1 not in (1, 3):
                        continue
                    shape = frozenset((u-min(us), v) for u, v in uv)
                    values = {int(grid[p]) for p in cells}
                    assert len(values) == 1
                    tokens.append((min(us), index, shape, max(us)-min(us)+1, values.pop()))
                tokens.sort()
                if not tokens or tokens[0][0] != base:
                    continue
                _, _, shape, depth, color = tokens[0]
                pitch = depth + 1
                if not all(u == base+k*pitch and sh == shape and dep == depth and co == color
                           for k, (u, _, sh, dep, co) in enumerate(tokens)):
                    continue
                candidates.append(dict(kind=kind, center=(r, c), normal=(nr, nc), tangent=(tr, tc),
                                       anchor=anchor, base=base, depth=depth, pitch=pitch,
                                       shape=shape, color=color, count=len(tokens),
                                       token_ids=[x[1] for x in tokens],
                                       cells=[p for x in tokens for p in payloads[x[1]]]))
    assigned = [i for p in candidates for i in p['token_ids']]
    assert sorted(assigned) == list(range(len(payloads))), 'each payload token has one terminal'
    collectors = [i for i, p in enumerate(candidates) if p['kind'] == 'collector']
    assert len(collectors) == 1 and len(candidates) >= 3
    root = collectors[0]
    active = _active(clean, candidates, root)
    assert active, 'a collector must receive a positive contribution'
    return dict(grid=grid, bg=bg, wire=wire, raw=raw, clean=clean, repairs=repairs,
                ports=candidates, root=root, active=active)


def _active(mask, ports, root):
    labels, _ = _components(mask)
    label = int(labels[ports[root]['anchor']])
    assert label >= 0
    return [i for i, p in enumerate(ports) if i != root and labels[p['anchor']] == label]


def _paint(grid, port, k, color):
    r, c = port['center']
    nr, nc = port['normal']
    tr, tc = port['tangent']
    start = port['base'] + k*port['pitch']
    for u, v in port['shape']:
        rr, cc = r+(start+u)*nr+v*tr, c+(start+u)*nc+v*tc
        assert 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1], 'no clipping'
        grid[rr, cc] = color


def solve(input_grid: np.ndarray) -> np.ndarray:
    p = _parse(input_grid)
    result = p['grid'].copy()
    result[p['raw']] = p['bg']
    result[p['clean']] = p['wire']
    for i in p['active']:
        for k in range(p['ports'][i]['count']):
            _paint(result, p['ports'][i], k, p['wire'])
    dst = p['ports'][p['root']]
    total = sum(p['ports'][i]['count'] for i in p['active'])
    for k in range(dst['count'], dst['count'] + total):
        # All added cells are empty in the input; no object may be overwritten.
        before = result.copy()
        _paint(result, dst, k, dst['color'])
        assert np.all(before[result != before] == p['bg'])
        assert np.all(p['grid'][result != before] == p['bg'])
    return result


def _edge(a, b):
    return tuple(sorted((a, b)))


def _node_groups(n, edges):
    groups, unseen = [], set(range(n))
    while unseen:
        todo, group = [min(unseen)], set()
        while todo:
            a = todo.pop()
            if a not in unseen:
                continue
            unseen.remove(a)
            group.add(a)
            todo.extend(b if a == x else x for x, b in edges if a in (x, b))
        groups.append(group)
    return groups


def _path(edges, root, target):
    todo = [(root, [])]
    seen = {root}
    while todo:
        a, path = todo.pop(0)
        if a == target:
            return path
        for x, y in sorted(edges):
            if a in (x, y):
                b = y if a == x else x
                if b not in seen:
                    seen.add(b)
                    todo.append((b, path + [_edge(a, b)]))
    return []


def _token_shape(rng, allow_depth, depth=None):
    if depth is None:
        depth = 2 if allow_depth and rng.random() < .3 else 1
    width = int(rng.choice([1, 3]))
    shape = {(u, v) for u in range(depth) for v in range(-(width//2), width//2+1)}
    if depth == 2 and width == 3 and rng.random() < .6:
        row = int(rng.integers(2))
        for v in (-1, 1):
            if rng.integers(2):
                shape.discard((row, v))
    return frozenset(shape), depth


def _plan(rng, axis, test, rich, collector_depth=None):
    cols = int(rng.choice([2, 3, 3, 4]))
    all_edges = [_edge(r*cols+c, r*cols+c+1) for r in range(2) for c in range(cols-1)]
    all_edges += [_edge(c, cols+c) for c in range(cols)]
    for _ in range(64):
        tree = []
        for j in rng.permutation(len(all_edges)):
            e = all_edges[int(j)]
            groups = _node_groups(2*cols, tree)
            if not any(set(e) <= g for g in groups):
                tree.append(e)
        cut = tree[int(rng.integers(len(tree)))]
        edges = set(tree) - {cut}
        groups = _node_groups(2*cols, edges)
        eligible = [g for g in groups if len(g) >= 3
                    and any(a//cols == b//cols for a, b in edges if a in g)
                    and any(a//cols != b//cols for a, b in edges if a in g)]
        if not eligible:
            continue
        live = eligible[int(rng.integers(len(eligible)))]
        # Loops are compatible with component counting. First demonstrations
        # stay trees, making each single repair an unambiguous connectivity cue.
        if rich and axis is None:
            for e in all_edges:
                if any(set(e) <= g for g in groups) and rng.random() < .3:
                    edges.add(e)
        possibilities = []
        for root in sorted(live):
            for far in sorted(live - {root}):
                path = _path(edges, root, far)
                axes = {'h' if a//cols == b//cols else 'v' for a, b in path}
                if len(path) >= 2 and (axis is None or axis in axes):
                    possibilities.append((root, far, path))
        if possibilities:
            break
    else:
        # Finite geometry fallback: an L with a separated fourth node.
        cols, all_edges = 2, [(0, 1), (0, 2), (1, 3), (2, 3)]
        edges, cut, live = {(0, 1), (1, 3)}, (0, 2), {0, 1, 3}
        possibilities = [(0, 3, [(0, 1), (1, 3)])]
    root, far, path = possibilities[int(rng.integers(len(possibilities)))]
    others = sorted(live - {root, far})
    rng.shuffle(others)
    sources = [far] + others[:int(rng.integers(1, min(2, len(others)) + 1))]
    dead = sorted(set(range(2*cols)) - live)
    rng.shuffle(dead)
    dormant = dead[:int(rng.integers(1, min(3, len(dead)) + 1))]
    port_nodes = sorted([root] + sources + dormant)
    xs = [int(rng.integers(1, 4))]
    for _ in range(cols-1):
        xs.append(xs[-1] + int(rng.integers(6, 9)))
    width = xs[-1] + 2 + int(rng.integers(3))
    step = int(rng.integers(6, 9)) if axis is None else 6
    ports = {}
    for node in port_nodes:
        shape, depth = _token_shape(rng, rich and axis is None,
                                    collector_depth if node == root else None)
        ports[node] = dict(shape=shape, depth=depth, pitch=depth+1,
                           offset=int(rng.integers(3, 5)), base=3 if node == root else 2,
                           count=1, kind='collector' if node == root else 'source')
    if axis is not None:
        # Demonstrate counting whole tokens rather than colored area, even with
        # only two training examples. Which source gets which width is random.
        unit_sources = list(rng.permutation(sources[:2]))
        ports[int(unit_sources[0])]['shape'] = frozenset({(0, 0)})
        ports[int(unit_sources[1])]['shape'] = frozenset({(0, -1), (0, 0), (0, 1)})
    broken = {cut}
    for e in all_edges:
        if e not in edges and not any(set(e) <= g for g in _node_groups(2*cols, edges)) and rng.random() < .4:
            broken.add(e)
    plan = dict(cols=cols, xs=xs, step=step, width=width, edges=edges, broken=broken,
                live=live, root=root, sources=sources, dormant=dormant, ports=ports,
                path=path, axis=axis, test=test)
    options = []
    inactive = {i: int(rng.integers(1, 3)) if axis is None else 1 for i in dormant}
    for operands in product(range(1, 4), repeat=len(sources)):
        total = sum(operands)
        for dest in range(1, 4):
            counts = dict(inactive)
            counts.update(zip(sources, operands))
            counts[root] = dest
            reaches = [0, 0]
            for node, port in ports.items():
                n = dest+total if node == root else counts[node]
                reach = port['offset'] + port['base'] + (n-1)*port['pitch'] + port['depth']-1
                reaches[node//cols] = max(reaches[node//cols], reach)
            height = sum(reaches) + step + 1
            if height <= 30 and width <= 30:
                options.append(dict(counts=counts, total=total, dest=dest, reaches=reaches, height=height))
    return plan, options


def _draw(rng, plan, choice, palette):
    bg, wire = palette
    cols, root = plan['cols'], plan['root']
    spare = 30-choice['height']
    top = int(rng.integers(min(2, spare)+1))
    bottom = int(rng.integers(min(2, spare-top)+1))
    height = choice['height'] + top + bottom
    ys = [choice['reaches'][0]+top, choice['reaches'][0]+top+plan['step']]
    pos = {i: (ys[i//cols], plan['xs'][i%cols]) for i in range(2*cols)}
    clean = np.zeros((height, plan['width']), bool)
    def edge_cells(e):
        a, b = pos[e[0]], pos[e[1]]
        dr, dc = int(np.sign(b[0]-a[0])), int(np.sign(b[1]-a[1]))
        return [(a[0]+k*dr, a[1]+k*dc) for k in range(abs(b[0]-a[0])+abs(b[1]-a[1])+1)]
    for e in sorted(plan['edges'] | plan['broken']):
        cells = edge_cells(e)
        for p in cells:
            clean[p] = True
        if e in plan['broken']:
            start = int(rng.integers(len(cells)-4))
            for p in cells[start+1:start+4]:
                clean[p] = False
    grid = np.full(clean.shape, bg, dtype=np.int8)
    ports = {}
    pool = [c for c in range(10) if c not in palette]
    for node, descriptor in plan['ports'].items():
        port = dict(descriptor)
        nr, nc = (-1, 0) if node//cols == 0 else (1, 0)
        tr, tc = -nc, nr
        y, x = pos[node]
        r, c = y+port['offset']*nr, x
        port.update(center=(r, c), normal=(nr, nc), tangent=(tr, tc),
                    count=choice['counts'][node], color=int(rng.choice(pool)))
        if node == root:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr or dc:
                        clean[r+dr, c+dc] = True
            for k in range(2, port['offset']+1):
                clean[r-k*nr, c] = True
        else:
            for v in (-1, 0, 1):
                clean[r+v*tr, c+v*tc] = True
            for k in range(1, port['offset']+1):
                clean[r-k*nr, c] = True
        ports[node] = port
    # A spatially excluded stack shares a contributing source's payload color.
    ports[plan['dormant'][0]]['color'] = ports[plan['sources'][0]]['color']
    for port in ports.values():
        for k in range(port['count']):
            _paint(grid, port, k, port['color'])
    assert not np.any(clean & (grid != bg))
    expected = grid.copy()
    expected[clean] = wire
    for node in plan['sources']:
        for k in range(ports[node]['count']):
            _paint(expected, ports[node], k, wire)
    for k in range(choice['dest'], choice['dest']+choice['total']):
        _paint(expected, ports[root], k, ports[root]['color'])
    assert np.array_equal(_restore(clean)[0], clean)
    path = plan['path']
    if plan['test']:
        repair_edges = [path[int(i)] for i in rng.choice(len(path), 2, replace=False)]
    else:
        options = [e for e in path if plan['axis'] is None or
                   ('h' if e[0]//cols == e[1]//cols else 'v') == plan['axis']]
        repair_edges = [options[int(rng.integers(len(options)))]]
    # Try finite local placements, retaining no retry artifacts.
    raw = clean.copy()
    for e in repair_edges:
        cells = edge_cells(e)
        choices = list(product(range(len(cells)-4), (0, 4), (-1, 1)))
        accepted = False
        for index in rng.permutation(len(choices)):
            start, endpoint, side = choices[int(index)]
            ar, ac = cells[start+endpoint]
            dr, dc = cells[1][0]-cells[0][0], cells[1][1]-cells[0][1]
            toward = 1 if endpoint == 0 else -1
            sr, sc = toward*dr+side*dc, toward*dc+side*dr
            spur = [(ar+k*sr, ac+k*sc) for k in (1, 2)]
            if not all(0 <= r < height and 0 <= c < grid.shape[1] and not raw[r, c]
                       and grid[r, c] == bg and expected[r, c] == bg for r, c in spur):
                continue
            trial = raw.copy()
            for p in cells[start+1:start+4]:
                trial[p] = False
            for p in spur:
                trial[p] = True
            try:
                rebuilt, evidence = _restore(trial)
                if not np.array_equal(rebuilt, clean):
                    continue
            except AssertionError:
                continue
            raw, accepted = trial, True
            break
        assert accepted, 'local repair placement unavailable'
    # Optional free diagonal clutter, outside payloads and all circuitry.
    if rng.random() < .65:
        for _ in range(24):
            r, c = int(rng.integers(height)), int(rng.integers(grid.shape[1]))
            dc = int(rng.choice([-1, 1]))
            cells = [(r+k, c+k*dc) for k in range(int(rng.choice([2, 3])))]
            if not all(0 <= rr < height and 0 <= cc < grid.shape[1] and grid[rr, cc] == bg
                       and expected[rr, cc] == bg and not raw[rr, cc] and not clean[rr, cc]
                       for rr, cc in cells):
                continue
            trial = raw.copy()
            for p in cells:
                trial[p] = True
            try:
                if np.array_equal(_restore(trial)[0], clean):
                    raw = trial
                    break
            except AssertionError:
                pass
    grid[raw] = wire
    turns = int(rng.integers(4))
    flip = bool(rng.integers(2))
    grid, expected = np.rot90(grid, turns), np.rot90(expected, turns)
    if flip:
        grid, expected = grid[:, ::-1], expected[:, ::-1]
    pair = dict(input=grid.copy(), output=expected.copy())
    info = _analyze(pair)
    assert info['missing_repair_changes_selection'] and info['all_gaps_changes_selection']
    assert not plan['test'] or info['repair_distance'] >= 2
    return pair


def _arithmetic_ok(a, b):
    s, d, t, e = a['total'], a['dest'], b['total'], b['dest']
    if s == t or d == e or s+d == t+e or s*e == t*d:
        return False
    rivals = (lambda s, d: 2*d, lambda s, d: 2*s, lambda s, d: s*d,
              lambda s, d: max(s, d), lambda s, d: s)
    return all(f(s, d) != s+d or f(t, e) != t+e for f in rivals)


def _key(a):
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError('illegal example count')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 2, replace=False)))
    varied_palette, rich = bool(rng.integers(2)), bool(rng.random() < .75)
    axes = ['h', 'v']
    rng.shuffle(axes)
    collector_depths = [1, 2]
    rng.shuffle(collector_depths)
    puzzle, seen = {'train': [], 'test': []}, set()
    for _ in range(128):
        first, aa = _plan(rng, axes[0], False, rich, collector_depths[0])
        second, bb = _plan(rng, axes[1], False, rich, collector_depths[1])
        choices = [(a, b) for a in aa for b in bb if _arithmetic_ok(a, b)
                   and (a['total'] > len(first['sources']) or b['total'] > len(second['sources']))
                   and (len(set(a['counts'][i] for i in first['sources'])) > 1
                        or len(set(b['counts'][i] for i in second['sources'])) > 1)]
        if not choices:
            continue
        a, b = choices[int(rng.integers(len(choices)))]
        try:
            initial = [_draw(rng, first, a, palette), _draw(rng, second, b, palette)]
            if _analyze(initial[0])['axes'] == _analyze(initial[1])['axes']:
                initial[1] = {k: np.rot90(v).copy() for k, v in initial[1].items()}
            if _key(initial[0]['input']) == _key(initial[1]['input']):
                continue
        except AssertionError:
            continue
        puzzle['train'].extend(initial)
        seen.update(_key(p['input']) for p in initial)
        break
    else:
        raise RuntimeError('bounded informative demonstration construction exhausted')
    for split, count in (('train', num_train-2), ('test', num_test)):
        for _ in range(count):
            for retry in range(128):
                plan, options = _plan(rng, None, split == 'test', rich)
                if not options:
                    continue
                local_palette = tuple(map(int, rng.choice(10, 2, replace=False))) if varied_palette else palette
                try:
                    pair = _draw(rng, plan, options[int(rng.integers(len(options)))], local_palette)
                except AssertionError:
                    continue
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded pair construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _repair_distance(parsed):
    """Minimum number of missing stretches crossed on a route to each source."""
    clean, raw, ports = parsed['clean'], parsed['raw'], parsed['ports']
    # Contract each raw-connected piece, then join pieces across repaired gaps.
    labels, groups = _components(clean & raw)
    adjacency = {i: [] for i in range(len(groups))}
    for (r, c, dr, dc), _ in parsed['repairs']:
        a, b = int(labels[r, c]), int(labels[r+4*dr, c+4*dc])
        assert a >= 0 and b >= 0
        adjacency[a].append(b)
        adjacency[b].append(a)
    root = int(labels[ports[parsed['root']]['anchor']])
    distances, todo = {root: 0}, [(0, root)]
    while todo:
        d, a = heapq.heappop(todo)
        for b in adjacency[a]:
            if d+1 < distances.get(b, 999):
                distances[b] = d+1
                heapq.heappush(todo, (d+1, b))
    return max(distances[int(labels[ports[i]['anchor']])] for i in parsed['active'])


def _analyze(pair):
    assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
    inp, out = _grid(pair['input']), _grid(pair['output'])
    assert inp.shape == out.shape
    p = _parse(inp)
    assert np.array_equal(solve(inp), out), 'incorrect complete output'
    ports, active, root = p['ports'], p['active'], p['root']
    dst = ports[root]
    total = sum(ports[i]['count'] for i in active)
    dormant = [i for i in range(len(ports)) if i != root and i not in active]
    for i in active:
        assert all(out[cell] == p['wire'] for cell in ports[i]['cells'])
    for i in dormant:
        assert all(out[cell] == inp[cell] for cell in ports[i]['cells'])
    assert p['repairs'] and np.count_nonzero(p['clean'] & ~p['raw']) == 3*len(p['repairs'])
    assert all(out[cell] == p['wire'] for cell in zip(*np.nonzero(p['clean'])))
    unclosed = _gaps(p['clean'])
    all_closed = p['clean'].copy()
    for r, c, dr, dc in unclosed:
        for k in (1, 2, 3):
            all_closed[r+k*dr, c+k*dc] = True
    no_repairs = _active(p['clean'] & p['raw'], ports, root)
    all_gaps = _active(all_closed, ports, root)
    _, groups = _components(p['clean'])
    cycles = sum(sum((r+1, c) in set(g) for r, c in g)
                 + sum((r, c+1) in set(g) for r, c in g)-len(g)+1 for g in groups)
    # Subtract the mandatory ring's one cycle: the remainder is network cycles.
    return dict(shape=inp.shape, palette=(p['bg'], p['wire']), sources=len(active),
                counts=tuple(ports[i]['count'] for i in active), total=total,
                dest=dst['count'], final=dst['count']+total, dormant=len(dormant),
                repairs=len(p['repairs']), axes=''.join(sorted({'h' if x[0][2] == 0 else 'v' for x in p['repairs']})),
                repair_distance=_repair_distance(p), cycles=cycles-1,
                masks=tuple(sorted((x['depth'], tuple(sorted(x['shape']))) for x in ports)),
                repeated=any(ports[i]['color'] == ports[j]['color'] for i in active for j in dormant),
                missing_repair_changes_selection=set(no_repairs) != set(active),
                all_gaps_changes_selection=set(all_gaps) != set(active), parsed=p)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    n, t = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= n <= 10 and 1 <= t <= 4 and n > t
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info = _analyze(pair)
            key = _key(_grid(pair['input']))
            assert key not in seen
            seen.add(key)
            assert info['dormant'] >= 1 and info['sources'] >= 2 and info['repeated']
            assert info['missing_repair_changes_selection'] and info['all_gaps_changes_selection']
            if split == 'train':
                training.append(info)
            else:
                assert info['repair_distance'] >= 2
    a, b = training[:2]
    assert _arithmetic_ok(a, b)
    assert any(len(set(x['counts'])) > 1 for x in (a, b))
    assert any(x['total'] != x['sources'] for x in (a, b))
    assert set(a['axes'] + b['axes']) == {'h', 'v'}
    assert {x['parsed']['ports'][x['parsed']['root']]['depth'] for x in (a, b)} == {1, 2}


def self_test():
    """42 seeds, streaming measurements, count checks, symmetries and mutations."""
    fingerprints, structures, shapes, palettes, masks, cycles = set(), set(), set(), set(), set(), set()
    source_counts, repairs, depths, operand_pairs = set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(tuple(_key(x['input']) for s in ('train', 'test') for x in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _analyze(pair)
                shapes.add(f['shape'])
                palettes.add(f['palette'])
                masks.update(f['masks'])
                depths.update(x[0] for x in f['masks'])
                source_counts.add(f['sources'])
                repairs.add(f['repairs'])
                cycles.add(f['cycles'])
                operand_pairs.add((f['total'], f['dest']))
                roles = {}
                normalized = bytes(roles.setdefault(int(v), len(roles)) for v in pair['input'].flat)
                structures.add((pair['input'].shape, normalized))
        if seed == 0:
            assert all(np.array_equal(x['input'], y['input']) and np.array_equal(x['output'], y['output'])
                       for s in ('train', 'test') for x, y in zip(puzzle[s], generate_puzzle(seed)[s]))
            pair = puzzle['test'][0]
            for k in range(4):
                for flip in (False, True):
                    a, b = np.rot90(pair['input'], k), np.rot90(pair['output'], k)
                    if flip:
                        a, b = a[:, ::-1], b[:, ::-1]
                    assert np.array_equal(solve(a), b)
            # Negative labels: partial repair, wrong sum, wrong source and a
            # changed dormant stack must each fail complete-output validation.
            p = _parse(pair['input'])
            for defect in ('repair', 'sum', 'source', 'dormant'):
                bad = pair['output'].copy()
                if defect == 'repair':
                    (r, c, dr, dc), spur = p['repairs'][0]
                    for j in (1, 2, 3):
                        bad[r+j*dr, c+j*dc] = p['bg']
                    for cell in spur:
                        bad[cell] = p['wire']
                elif defect == 'sum':
                    d = p['ports'][p['root']]
                    _paint(bad, d, d['count'], p['bg'])
                else:
                    i = p['active'][0] if defect == 'source' else next(i for i in range(len(p['ports'])) if i != p['root'] and i not in p['active'])
                    _paint(bad, p['ports'][i], 0, p['ports'][i]['color'] if defect == 'source' else p['wire'])
                try:
                    _analyze(dict(input=pair['input'], output=bad))
                except AssertionError:
                    pass
                else:
                    raise AssertionError('accepted incorrect ' + defect)
    assert len(fingerprints) == 42 and len(structures) >= 180
    assert len(shapes) >= 30 and len(palettes) >= 30 and len(masks) >= 6
    assert depths == {1, 2} and source_counts == {2, 3} and repairs == {1, 2}
    assert len(operand_pairs) >= 8 and max(cycles) >= 1
    for n in range(2, 11):
        for t in range(1, min(4, n-1)+1):
            p = generate_puzzle(971+n*10+t, n, t)
            assert len(p['train']) == n and len(p['test']) == t
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError('accepted illegal counts')
    return dict(seeds=42, distinct_puzzles=len(fingerprints), input_structures=len(structures),
                grid_shapes=len(shapes), palettes=len(palettes), token_masks=len(masks),
                source_counts=sorted(source_counts), repair_counts=sorted(repairs),
                network_cycles=sorted(cycles), operand_pairs=len(operand_pairs))
