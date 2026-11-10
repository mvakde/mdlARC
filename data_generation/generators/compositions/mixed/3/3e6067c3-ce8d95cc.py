"""Key-directed station bands, followed by exact two-axis run contraction.

The decoder uses only the input. Sampling policies affect neither its conventions
nor the transformation. No parent generator or saved puzzle is loaded at runtime.
"""
from __future__ import annotations

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or any(n < 1 or n > 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with sides 1..30")
    return a.astype(np.int8, copy=True)


def _signature(a):
    return a.shape, a.tobytes()


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for y, x in zip(*np.nonzero(mask)):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        stack, points = [(y, x)], []
        seen[y, x] = True
        while stack:
            r, c = stack.pop()
            points.append((r, c))
            for yy, xx in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1]
                        and mask[yy, xx] and not seen[yy, xx]):
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(points)
    return result


def _parse(a):
    bg = int(a[0, 0])
    if not all(np.all(e == bg) for e in (a[0], a[-1], a[:, 0], a[:, -1])):
        raise ValueError("a uniform background frame is required")
    stations, tokens, geometry = [], [], None
    for component in _components(a != bg):
        if len(component) == 1:
            tokens.append(component[0])
            continue
        ys, xs = zip(*component)
        y, x = min(ys), min(xs)
        size = max(ys)-y+1
        if size < 3 or max(xs)-x+1 != size or len(component) != size*size:
            raise ValueError("stations must be separated filled squares")
        block = a[y:y+size, x:x+size]
        carrier = int(block[0, 0])
        py, px = np.nonzero(block != carrier)
        if not len(py):
            raise ValueError("missing station label")
        offset, patch = int(py.min()), int(py.max()-py.min()+1)
        if (offset < 1 or 2*offset+patch != size or int(px.min()) != offset
                or int(px.max()) != offset+patch-1 or len(py) != patch*patch):
            raise ValueError("labels must be centered square patches")
        label = int(block[offset, offset])
        if label in (bg, carrier) or not np.all(block[offset:offset+patch, offset:offset+patch] == label):
            raise ValueError("structural and label roles must be distinct")
        current = (size, patch, carrier)
        if geometry is not None and current != geometry:
            raise ValueError("one station geometry and carrier per input")
        geometry = current
        stations.append((y, x, label))
    if geometry is None or not 3 <= len(stations) <= 18 or not 3 <= len(tokens) <= 14:
        raise ValueError("invalid station/key inventory")
    size, patch, carrier = geometry
    tokens.sort(key=lambda p: p[1])
    key_y = tokens[0][0]
    if (any(y != key_y for y, x in tokens)
            or key_y < max(y+size for y, x, label in stations)+1
            or any(b[1]-a[1] < 2 for a, b in zip(tokens, tokens[1:]))):
        raise ValueError("the isolated horizontal key must be below the stations")
    key = tuple(int(a[y, x]) for y, x in tokens)
    if len(set(key)) < 2 or any(c in (bg, carrier) for c in key):
        raise ValueError("invalid key colors")
    return dict(background=bg, carrier=carrier, size=size, patch=patch,
                stations=sorted(stations), key=key, key_y=key_y,
                key_xs=tuple(x for y, x in tokens))


def _band(a, b, size, patch):
    offset = (size-patch)//2
    if a[0] == b[0] and abs(a[1]-b[1]) > size:
        return (slice(a[0]+offset, a[0]+offset+patch),
                slice(min(a[1], b[1])+size, max(a[1], b[1])))
    if a[1] == b[1] and abs(a[0]-b[0]) > size:
        return (slice(min(a[0], b[0])+size, max(a[0], b[0])),
                slice(a[1]+offset, a[1]+offset+patch))
    return None


def _routes(graph, labels, key, limit=2):
    candidates = [[i for i, label in enumerate(labels) if label == c] for c in key]
    solutions, dead = [], set()

    def visit(path, used):
        if len(solutions) >= limit:
            return
        if len(path) == len(key):
            solutions.append(tuple(path))
            return
        state = (path[-1] if path else -1, used)
        if state in dead:
            return
        before = len(solutions)
        for i in candidates[len(path)]:
            if not (used >> i) & 1 and (not path or i in graph[path[-1]]):
                visit(path+[i], used | (1 << i))
                if len(solutions) >= limit:
                    return
        if len(solutions) == before:
            dead.add(state)
    visit([], 0)
    return solutions


def _decode(a):
    info = _parse(a)
    stations = info['stations']
    graph = [[] for _ in stations]
    for i, u in enumerate(stations):
        for j in range(i+1, len(stations)):
            band = _band(u, stations[j], info['size'], info['patch'])
            if band is not None and np.all(a[band] == info['background']):
                graph[i].append(j)
                graph[j].append(i)
    routes = _routes(graph, [s[2] for s in stations], info['key'])
    if len(routes) != 1:
        raise ValueError("the key must determine one distinct-station route")
    info.update(route=routes[0], graph=graph)
    return info


def _paint(a, info, width=None, target=False):
    out = a.copy()
    painted = np.zeros(a.shape, dtype=bool)
    bands = []
    for i, j in zip(info['route'], info['route'][1:]):
        u, v = info['stations'][i], info['stations'][j]
        p = info['patch'] if width is None else width
        band = _band(u, v, info['size'], p)
        color = v[2] if target else u[2]
        if np.any(painted[band] & (out[band] != color)):
            raise ValueError("intersecting bands must agree in color")
        out[band] = color
        painted[band] = True
        bands.append(band)
    return out, painted, bands


def _keeps(a):
    rows = np.r_[True, np.any(a[1:] != a[:-1], axis=1)]
    cols = np.r_[True, np.any(a[:, 1:] != a[:, :-1], axis=0)]
    return np.flatnonzero(rows), np.flatnonzero(cols)


def _collapse(a):
    r, c = _keeps(a)
    return a[np.ix_(r, c)].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the route on the original grid, paint, then contract equal vectors."""
    a = _grid(input_grid)
    painted, _, _ = _paint(a, _decode(a))
    return _collapse(painted)


def _visibility(cells):
    graph = [[] for _ in cells]
    for i, a in enumerate(cells):
        for j, b in enumerate(cells):
            if i == j:
                continue
            axis = 1 if a[0] == b[0] else 0 if a[1] == b[1] else None
            if axis is None:
                continue
            other = 1-axis
            lo, hi = sorted((a[axis], b[axis]))
            if not any(c[other] == a[other] and lo < c[axis] < hi for c in cells):
                graph[i].append(j)
    return graph


def _sample_path(rng, cells, graph, length, kind):
    budget = 1500

    def visit(path, axes):
        nonlocal budget
        budget -= 1
        if budget < 0:
            return None
        if len(path) == length:
            return path if kind != 'both' or len(set(axes)) == 2 else None
        for j in rng.permutation(graph[path[-1]]):
            j = int(j)
            if j in path:
                continue
            axis = int(cells[j][0] == cells[path[-1]][0])
            if kind == 'straight' and axes and axes[0] != axis:
                continue
            if kind == 'elbow' and sum(a != b for a, b in zip(axes+[axis], (axes+[axis])[1:])) > 1:
                continue
            found = visit(path+[j], axes+[axis])
            if found is not None:
                return found
        return None
    for i in rng.permutation(len(cells)):
        found = visit([int(i)], [])
        if found is not None:
            return found
        if budget < 0:
            break
    return None


def _cross(a, b, c, d):
    if a[0] == b[0] and c[1] == d[1]:
        return (min(a[1], b[1]) < c[1] < max(a[1], b[1])
                and min(c[0], d[0]) < a[0] < max(c[0], d[0]))
    if a[1] == b[1] and c[0] == d[0]:
        return _cross(c, d, a, b)
    return False


def _labels(rng, cells, graph, path, palette, mode):
    length = len(path)
    parent = list(range(length))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    def join(i, j):
        parent[root(i)] = root(j)

    # Equal-color crossings are the only permitted overlaps; equality is sampled
    # before colors, then uniqueness is checked against the complete route graph.
    for i in range(length-1):
        for j in range(i+1, length-1):
            if _cross(cells[path[i]], cells[path[i+1]], cells[path[j]], cells[path[j+1]]):
                join(i, j)
    if mode == 'repeat' and len({root(i) for i in range(length)}) == length:
        join(0, int(rng.integers(2, length)))
    groups = sorted({root(i) for i in range(length)})
    if mode == 'unique' and len(groups) != length:
        return None
    for attempt in range(80):
        if mode == 'unique' or (len(groups) <= 8 and (attempt > 40 or rng.random() < .55)):
            if len(groups) > 8:
                return None
            chosen = [int(c) for c in rng.permutation(palette)[:len(groups)]]
        else:
            k = int(rng.integers(min(3, len(groups)), min(8, len(groups))+1))
            pool = rng.permutation(palette)[:k]
            chosen = [int(rng.choice(pool)) for _ in groups]
        mapping = dict(zip(groups, chosen))
        key = [mapping[root(i)] for i in range(length)]
        if len(set(key)) < 2:
            continue
        labels = [-1]*len(cells)
        for i, color in zip(path, key):
            labels[i] = color
        absent = [int(c) for c in palette if c not in key]
        for i in range(len(cells)):
            if labels[i] < 0:
                labels[i] = int(rng.choice(absent if absent and rng.random() < .5 else key))
        solutions = _routes(graph, labels, key)
        if len(solutions) == 1 and solutions[0] == tuple(path):
            return labels, key
    return None


def _axis(rng, count, size, limit, gap_max, stretched=False):
    gaps = np.ones(max(0, count-1), dtype=int)
    spare = limit-count*size-len(gaps)
    if spare < 0:
        raise ValueError('layout too large')
    if stretched and len(gaps) and spare >= 3:
        j = int(rng.integers(len(gaps)))
        gaps[j] += 3
        spare -= 3
    for j in rng.permutation(len(gaps)):
        extra = int(rng.integers(0, min(spare, max(0, gap_max-int(gaps[j])))+1))
        gaps[j] += extra
        spare -= extra
    return [0]+[int(c) for c in np.cumsum(size+gaps)]


def _draw(rng, cells, path, labels, key, size, patch, roles, gap_max,
          anchor=False, stretch=False):
    nr = max(r for r, c in cells)+1
    nc = max(c for r, c in cells)+1
    ys = _axis(rng, nr, size, 25 if anchor else 26, gap_max, stretch)
    xs = _axis(rng, nc, size, 27 if anchor else 28, gap_max, stretch)
    bh, bw = ys[-1]+size, xs[-1]+size
    key_gaps = np.full(len(key)-1, 2, dtype=int)
    spare = 28-(2*len(key)-1)
    for j in rng.permutation(len(key_gaps)):
        extra = int(rng.integers(0, min(spare, 2)+1))
        key_gaps[j] += extra
        spare -= extra
    key_xs = np.r_[0, np.cumsum(key_gaps)]
    kw = int(key_xs[-1])+1
    hmin, wmin = bh+(5 if anchor else 4), max(bw, kw)+(3 if anchor else 2)
    h = int(rng.integers(hmin, 31))
    w = int(rng.integers(wmin, 31))
    top = 2 if anchor else int(rng.integers(1, h-bh-2))
    left = int(rng.integers(2 if anchor else 1, w-bw))
    ky = int(rng.integers(top+bh+1, h-1))
    kx = int(rng.integers(2 if anchor else 1, w-kw))
    bg, carrier = roles
    a = np.full((h, w), bg, dtype=np.int8)
    offset = (size-patch)//2
    positions = [(ys[r]+top, xs[c]+left, label) for (r, c), label in zip(cells, labels)]
    for y, x, label in positions:
        a[y:y+size, x:x+size] = carrier
        a[y+offset:y+offset+patch, x+offset:x+offset+patch] = label
    a[ky, kx+key_xs] = key
    # Ground truth is constructed from the sampled endpoints, not solve().
    info = dict(stations=positions, route=path, size=size, patch=patch)
    painted, _, _ = _paint(a, info)
    return {'input': a, 'output': _collapse(painted)}


def _cross_scene(rng, size, patch, roles, gap_max, anchor=False, stretch=False):
    cells = [(1, 0), (1, 2), (0, 2), (0, 1), (2, 1), (0, 0)]
    for _ in range(int(rng.integers(0, 4))):
        cells = [(c, 2-r) for r, c in cells]
    if rng.random() < .5:
        cells = [(r, 2-c) for r, c in cells]
    colors = [int(c) for c in rng.permutation([c for c in range(10) if c not in roles])[:5]]
    key = [colors[0], colors[1], colors[2], colors[0], colors[3]]
    return _draw(rng, cells, list(range(5)), key+[colors[4]], key, size, patch, roles,
                 gap_max, anchor=anchor, stretch=stretch)


def _fallback(rng, size, patch, roles, gap_max, anchor, long_route, stretch, introductory=False):
    colors = [int(c) for c in rng.permutation([c for c in range(10) if c not in roles])]
    if introductory:
        return _draw(rng, [(0, 0), (0, 1), (1, 1)], [0, 1, 2], colors[:3],
                     colors[:3], size, patch, roles, gap_max, anchor=anchor)
    if long_route:
        cells = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0), (2, 0), (2, 1), (2, 2)]
        key = colors+[colors[0]]
        return _draw(rng, cells, list(range(9)), key, key, 3, 1, roles, gap_max,
                     stretch=stretch)
    # A repeated initial label and an unused duplicate; the wrong starts fail
    # spatial alignment. This is also a bounded fallback for the second demo.
    cells = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)]
    key = [colors[0], colors[1], colors[2], colors[0], colors[3]]
    labels = key+[colors[0]]
    return _draw(rng, cells, list(range(5)), labels, key, size, patch, roles,
                 gap_max, anchor=anchor, stretch=stretch)


def _sample_pair(rng, params, split, index):
    anchor = split == 'train' and index < 2
    roles = params['roles'] if params['stable'] else tuple(int(c) for c in rng.choice(10, 2, replace=False))
    palette = [c for c in range(10) if c not in roles]
    shift = params['shift']
    stretch = split == 'test' and shift == 'spacing'
    long_route = split == 'test' and shift == 'route'
    gap_max = 2 if split == 'train' and shift == 'spacing' else 8
    if anchor:
        size, patch = 5, params['widths'][index]
    else:
        sizes = (4, 6) if split == 'test' and shift == 'scale' else (
            (3, 5, 7) if shift == 'scale' else (3, 4) if long_route else (3, 4, 5, 6))
        size = int(rng.choice(sizes))
        patch = int(rng.choice(list(range(1 if size % 2 else 2, size-1, 2))))
    if anchor and index == 1 and rng.random() < .35:
        return _cross_scene(rng, size, patch, roles, gap_max, anchor=True)
    for attempt in range(56):
        length = (int(rng.integers(3, 6)) if anchor and index == 0 else
                  int(rng.integers(9, 13)) if long_route else int(rng.integers(4, 8)))
        unused = 0 if anchor and index == 0 else int(rng.integers(1 if anchor else 0, 4))
        count = length+unused
        maxh, maxw = (25, 27) if anchor else (26, 28)
        options = [(r, c) for r in range(1, 7) for c in range(1, 8)
                   if count <= r*c <= count+7 and r*size+r-1 <= maxh and c*size+c-1 <= maxw]
        if not options:
            break
        r, c = options[int(rng.integers(len(options)))]
        universe = [(y, x) for y in range(r) for x in range(c)]
        cells = sorted(universe[int(i)] for i in rng.choice(len(universe), count, replace=False))
        # Remove completely empty exterior levels; interior missing levels remain.
        ymin, xmin = min(y for y, x in cells), min(x for y, x in cells)
        cells = [(y-ymin, x-xmin) for y, x in cells]
        if shift == 'spacing':
            # Every coordinate level is visibly occupied in this regime, so
            # small training gaps cannot conceal an absent intermediate tier.
            row_ids = sorted({y for y, x in cells})
            col_ids = sorted({x for y, x in cells})
            cells = [(row_ids.index(y), col_ids.index(x)) for y, x in cells]
        graph = _visibility(cells)
        kind = 'both' if anchor else str(rng.choice(('straight', 'elbow', 'free', 'free', 'free')))
        path = _sample_path(rng, cells, graph, length, kind)
        if path is None:
            continue
        mode = 'unique' if anchor and index == 0 else 'repeat' if anchor else 'free'
        assigned = _labels(rng, cells, graph, path, palette, mode)
        if assigned is None:
            continue
        labels, key = assigned
        pair = _draw(rng, cells, path, labels, key, size, patch, roles,
                     gap_max, anchor=anchor, stretch=stretch)
        if pair['input'].shape == pair['output'].shape:
            continue
        if stretch:
            info = _decode(pair['input'])
            if max(_link_lengths(info)) < 4 or max(_level_gaps(info), default=0) < 4:
                continue
        return pair
    # Fit is guaranteed for these small layouts; fallbacks obey the same rule.
    return _fallback(rng, size if size <= 6 else 5, patch if size <= 6 else 1,
                     roles, gap_max, anchor, long_route, stretch,
                     introductory=anchor and index == 0)


def _link_lengths(info):
    result = []
    for i, j in zip(info['route'], info['route'][1:]):
        a, b = info['stations'][i], info['stations'][j]
        result.append(abs(a[0]-b[0])+abs(a[1]-b[1])-info['size'])
    return result


def _level_gaps(info):
    result = []
    for axis in (0, 1):
        levels = sorted({s[axis] for s in info['stations']})
        result.extend(b-a-info['size'] for a, b in zip(levels, levels[1:]))
    return result


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    params = dict(roles=tuple(int(c) for c in rng.choice(10, 2, replace=False)),
                  stable=bool(rng.integers(0, 2)), shift=str(rng.choice(('scale', 'route', 'spacing'))),
                  widths=tuple(int(w) for w in rng.permutation((1, 3))))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            pair = _sample_pair(rng, params, split, index)
            if _signature(pair['input']) in seen:
                # With at least three visible colors these 40 affine maps
                # produce different inputs; at most 13 earlier inputs exist.
                original = pair
                for multiplier in (1, 3, 7, 9):
                    for offset in range(10):
                        lookup = (multiplier*np.arange(10)+offset) % 10
                        pair = {k: lookup[v].astype(np.int8) for k, v in original.items()}
                        if _signature(pair['input']) not in seen:
                            break
                    else:
                        continue
                    break
                else:
                    raise AssertionError('distinct-input fallback exhausted')
            seen.add(_signature(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
    a, out = _grid(pair['input']), _grid(pair['output'])
    info = _decode(a)
    painted, mask, bands = _paint(a, info)
    expected = _collapse(painted)
    assert np.array_equal(out, expected), 'output must be contraction of the painted input'
    assert out.shape != a.shape, 'redundant vectors must be present'
    assert np.any(mask) and np.all(a[mask] == info['background'])
    assert np.array_equal(painted[~mask], a[~mask]), 'station/key/other pixels changed'
    assert not np.array_equal(_collapse(a), out), 'dropping A must change the answer'
    assert not np.array_equal(painted, out), 'dropping B must change the answer'
    rows, cols = _keeps(painted)
    rr = np.diff(np.r_[rows, painted.shape[0]])
    cc = np.diff(np.r_[cols, painted.shape[1]])
    reconstructed = np.repeat(np.repeat(out, rr, axis=0), cc, axis=1)
    assert np.array_equal(reconstructed, painted), 'retained values must reconstruct every run'
    # Independent order of the two contractions, plus idempotence.
    assert np.array_equal(_collapse(painted.T).T, out)
    assert np.array_equal(_collapse(out), out)
    coverage = np.zeros(a.shape, dtype=np.int8)
    for band in bands:
        coverage[band] += 1
    axes = {int(info['stations'][i][0] == info['stations'][j][0])
            for i, j in zip(info['route'], info['route'][1:])}

    def alternative_diff(**kwargs):
        try:
            alternative = _collapse(_paint(a, info, **kwargs)[0])
        except ValueError:
            # An alternative that makes previously consistent crossings
            # contradictory also fails to explain this demonstration.
            return True
        return not np.array_equal(alternative, out)
    return dict(info, input_shape=a.shape, output_shape=out.shape,
                row_runs=tuple(int(n) for n in rr), column_runs=tuple(int(n) for n in cc),
                crossings=int(np.count_nonzero(coverage > 1)), axes=axes,
                nonlocal_rows=len({row.tobytes() for row in out}) < out.shape[0],
                nonlocal_cols=len({col.tobytes() for col in out.T}) < out.shape[1],
                target_diff=alternative_diff(target=True), unit_diff=alternative_diff(width=1))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    infos, seen = {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            a = _grid(pair['input'])
            signature = _signature(a)
            assert signature not in seen, 'duplicate input'
            seen.add(signature)
            infos[split].append(_analyze_pair(pair))
    train = infos['train']
    assert any(a['target_diff'] for a in train), 'source vs target must be observable'
    assert any(a['patch'] > 1 and a['unit_diff'] for a in train), 'patch width must be observable'
    assert any(len(a['axes']) == 2 for a in train), 'demonstrate both orientations'
    assert any(max(a['row_runs']) > 1 and max(a['column_runs']) > 1 for a in train)
    assert any(a['nonlocal_rows'] for a in train), 'separated identical rows must survive'
    assert any(a['nonlocal_cols'] for a in train), 'separated identical columns must survive'
    assert any(len(set(a['key'])) < len(a['key']) for a in train)
    assert any(len(a['stations']) == len(a['key']) for a in train)
    if any(len(a['stations']) > len(a['key']) for a in infos['test']):
        assert any(len(a['stations']) > len(a['key']) for a in train)
    test = infos['test']
    shifts = (
        all(a['size'] % 2 for a in train) and all(a['size'] % 2 == 0 for a in test),
        max(len(a['key']) for a in train) <= 7 and min(len(a['key']) for a in test) >= 9,
        max(g for a in train for g in _level_gaps(a)) <= 2
        and all(max(_level_gaps(a), default=0) >= 4 for a in test),
    )
    assert any(shifts), 'a visible scale, route-length, or spacing holdout is required'


def _puzzle_signature(p):
    return tuple(tuple((_signature(_grid(pair['input'])), _signature(_grid(pair['output'])))
                       for pair in p[s]) for s in ('train', 'test'))


def self_test():
    """Bounded streaming coverage audit: 42 seeds; no files or saved traces."""
    shapes, outputs, layouts, roles, lengths, geometries = set(), set(), set(), set(), set(), set()
    flags = dict(crossing=0, unused=0, sparse=0, unequal=0, row_only=0, repeated=0)
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_signature(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = _analyze_pair(pair)
                shapes.add(a['input_shape'])
                outputs.add(a['output_shape'])
                roles.add((a['background'], a['carrier']))
                lengths.add(len(a['key']))
                geometries.add((a['size'], a['patch']))
                stations = a['stations']
                ys, xs = {s[0] for s in stations}, {s[1] for s in stations}
                layouts.add((a['size'], tuple((y-min(ys), x-min(xs)) for y, x, c in stations),
                             a['row_runs'], a['column_runs']))
                flags['crossing'] += a['crossings'] > 0
                flags['unused'] += len(stations) > len(a['key'])
                flags['sparse'] += len(stations) < len(ys)*len(xs)
                flags['unequal'] += len(set(_link_lengths(a))) > 1
                flags['row_only'] += a['input_shape'][1] == a['output_shape'][1]
                flags['repeated'] += len(set(a['key'])) < len(a['key'])
                source = _grid(pair['input'])
                original = source.copy()
                assert np.array_equal(solve(source), pair['output'])
                assert np.array_equal(source, original), 'solve must not mutate input'
    assert len(signatures) == 42
    assert len(shapes) >= 40 and len(outputs) >= 35 and len(layouts) >= 150
    assert len(roles) >= 25 and {3, 4, 5, 6} <= {s for s, p in geometries}
    assert {1, 2, 3} <= {p for s, p in geometries}
    assert min(lengths) == 3 and max(lengths) >= 10
    assert all(flags[k] >= 5 for k in ('crossing', 'unused', 'sparse', 'unequal', 'repeated'))
    assert _puzzle_signature(generate_puzzle(0)) == _puzzle_signature(generate_puzzle(0))
    for nt, nq in ((2, 1), (5, 4), (10, 1), (10, 4)):
        p = generate_puzzle(1700+nt*10+nq, nt, nq)
        assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p = generate_puzzle(0, 2, 1)
    p['test'][0]['output'][0, 0] = (int(p['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('corrupt output accepted')
