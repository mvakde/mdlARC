"""The regional signal loom: spatially gate two colors on each traced cable.

No parent module is imported. Construction uses explicit paths; solve recovers
regions, occluded connections, strands, and endpoint colors from input cells.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import permutations

import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_DIAMOND = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
_CHECKER = ~_DIAMOND


def _require(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "bad grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "noninteger grid")
    _require(np.all((g >= 0) & (g <= 9)), "bad color")
    return g.astype(np.int8, copy=False)


def _components(mask, diagonal=False):
    cells = set(map(tuple, np.argwhere(mask)))
    result = []
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c] if diagonal else _DIRS
    while cells:
        root = min(cells)
        cells.remove(root)
        found, stack = {root}, [root]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in cells:
                    cells.remove(q)
                    found.add(q)
                    stack.append(q)
        result.append(found)
    return result


def _cut_info(cut):
    """Digital simple path, with diagonal shortcuts around elbows suppressed."""
    cells = set(map(tuple, np.argwhere(cut)))
    graph = {}
    for r, c in cells:
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                q = (r + dr, c + dc)
                if not (dr or dc) or q not in cells:
                    continue
                if dr and dc and ((r + dr, c) in cells or (r, c + dc) in cells):
                    continue
                neighbors.append(q)
        graph[r, c] = neighbors
    ends = [p for p, ns in graph.items() if len(ns) == 1]
    _require(len(cells) >= 3 and len(ends) == 2 and all(len(ns) in (1, 2) for ns in graph.values()),
             "fence is not a simple path")
    h, w = cut.shape
    boundary = lambda p: p[0] in (0, h - 1) or p[1] in (0, w - 1)
    _require(all(boundary(p) for p in ends), "fence endpoints must reach the boundary")
    _require(all(not boundary(p) for p in cells - set(ends)), "extra fence boundary contacts")
    path, prev, p = [], None, min(ends)
    while p is not None and p not in path:
        path.append(p)
        ns = [q for q in graph[p] if q != prev]
        prev, p = p, ns[0] if ns else None
    _require(len(path) == len(cells), "disconnected fence")
    regions = _components(~cut)
    _require(len(regions) == 2, "fence must leave exactly two four-connected regions")
    masks = []
    for region in regions:
        m = np.zeros(cut.shape, dtype=bool)
        for p in region:
            m[p] = True
        masks.append(m)
    return masks, path


def _decode_cut(g):
    bg = int(g[0, 0])
    _require(all(int(g[r, c]) == bg for r, c in ((0, -1), (-1, 0), (-1, -1))),
             "corners must identify the background")
    edge = np.concatenate((g[0], g[-1], g[:, 0], g[:, -1]))
    colors = set(map(int, edge)) - {bg}
    _require(len(colors) == 1, "only the fence color may reach the exterior boundary")
    control = colors.pop()
    parts = sorted(_components(g == control, True), key=len)
    _require(len(parts) == 2 and len(parts[0]) == 1 and len(parts[1]) >= 3,
             "expected a fence and one isolated marker")
    marker = next(iter(parts[0]))
    cut = np.zeros(g.shape, dtype=bool)
    for p in parts[1]:
        cut[p] = True
    regions, path = _cut_info(cut)
    _require(min(max(abs(marker[0] - r), abs(marker[1] - c)) for r, c in path) >= 2,
             "marker touches the fence")
    side = 0 if regions[0][marker] else 1
    return dict(background=bg, control=control, marker=marker, cut=cut,
                regions=regions, marked=regions[side], side=side, fence_path=path)


def _terminal_label(block, pattern, bg):
    _require(block.shape == (3, 3), "incomplete terminal")
    values = block[pattern]
    color = int(values[0])
    _require(color != bg and np.all(values == color) and np.all(block[~pattern] == bg),
             "terminal is not the required stencil")
    return color


def _trace(g, cut, start, end, centers, carrier, cross):
    """Directional tracing; a fence gap continues only along a straight ray."""
    visible = set(map(tuple, np.argwhere(g == carrier)))
    if cross is not None:
        visible.update(map(tuple, np.argwhere(g == cross)))
    endpoints = {(start, c): 2 for c in centers}
    endpoints.update({(end, c): 0 for c in centers})
    graph = {p: {} for p in visible}
    for r, c in visible:
        for d, (dr, dc) in enumerate(_DIRS):
            q = (r + dr, c + dc)
            if q in visible:
                graph[r, c][d] = (q, (d + 2) % 4)
    links = set()
    h, w = g.shape
    # Only a visible strand end pointing into the fence is a true port.
    for p in sorted(visible):
        if p in endpoints or len(graph[p]) != 1:
            continue
        incoming = next(iter(graph[p]))
        d = (incoming + 2) % 4
        dr, dc = _DIRS[d]
        r, c = p[0] + dr, p[1] + dc
        if not (0 <= r < h and 0 <= c < w and cut[r, c]):
            continue
        while 0 <= r < h and 0 <= c < w and cut[r, c]:
            r, c = r + dr, c + dc
        q = (r, c)
        _require(q in visible and q not in endpoints, "fence hides an unpaired cable end")
        _require(len(graph[q]) == 1 and next(iter(graph[q])) == d,
                 "fence hides a bend or an ambiguous connection")
        links.add(tuple(sorted((p, q))))
    hidden_cells = set()
    for p, q in sorted(links):
        dr, dc = q[0] - p[0], q[1] - p[1]
        d = 2 if dr > 0 else (0 if dr < 0 else (1 if dc > 0 else 3))
        step_r, step_c = _DIRS[d]
        middle, cell = set(), (p[0] + step_r, p[1] + step_c)
        while cell != q:
            middle.add(cell)
            cell = (cell[0] + step_r, cell[1] + step_c)
        _require(not middle & hidden_cells, "fence hides a crossing")
        hidden_cells.update(middle)
        _require(d not in graph[p] and (d + 2) % 4 not in graph[q], "duplicate gap edge")
        graph[p][d] = (q, (d + 2) % 4)
        graph[q][(d + 2) % 4] = (p, d)
    overpasses = {}
    for p, edges in graph.items():
        if p in endpoints:
            _require(len(edges) == 1 and endpoints[p] in edges, "broken bank port")
        elif len(edges) == 4:
            r, c = p
            axes = [axis for axis, ds in enumerate(((0, 2), (1, 3)))
                    if cross is not None and all(g[r + _DIRS[d][0], c + _DIRS[d][1]] == cross for d in ds)]
            _require(len(axes) == 1 and g[p] == carrier, "crossing needs one marked overpass axis")
            overpasses[p] = axes[0]
        else:
            _require(len(edges) == 2, "broken or branching cable")
    if cross is not None:
        marked_cells = {(r + _DIRS[d][0], c + _DIRS[d][1])
                        for (r, c), axis in overpasses.items()
                        for d in ((0, 2), (1, 3))[axis]}
        _require(marked_cells == set(map(tuple, np.argwhere(g == cross))), "stray crossing auxiliary mark")
    routes, pairing, used = [], [], set()
    for c in centers:
        p, incoming, seen, route = (start, c), 0, set(), []
        while True:
            _require((p, incoming) not in seen, "cable cycle")
            seen.add((p, incoming))
            route.append(p)
            used.add(p)
            if p in endpoints and p[0] == end:
                pairing.append(centers.index(p[1]))
                break
            edges = graph[p]
            if p == (start, c):
                outgoing = 2
            elif len(edges) == 4:
                outgoing = (incoming + 2) % 4
            else:
                _require(incoming in edges, "invalid incoming strand")
                outgoing = next(d for d in edges if d != incoming)
            p, incoming = edges[outgoing]
            _require(not (p in endpoints and p[0] == start), "route returns to the diamond bank")
        routes.append(route)
    _require(sorted(pairing) == list(range(len(centers))), "routing must be bijective")
    _require(used == visible, "unattached wire fragments")
    return routes, pairing, overpasses, sorted(links)


def _view(g, orientation):
    x = g.T if orientation >= 2 else g
    return x[::-1] if orientation % 2 else x


def _unview(g, orientation):
    x = g[::-1] if orientation % 2 else g
    return x.T.copy() if orientation >= 2 else x.copy()


def _parse(input_grid):
    original = _grid(input_grid)
    _decode_cut(original)
    candidates = []
    for orientation in range(4):
        g = _view(original, orientation)
        d = _decode_cut(g)
        bg, control = d['background'], d['control']
        for carrier in sorted(set(map(int, np.unique(g))) - {bg, control}):
            points = np.argwhere(g == carrier)
            start, end = int(points[:, 0].min()), int(points[:, 0].max())
            if start < 3 or end > g.shape[0] - 4 or end - start < 4:
                continue
            centers = list(map(int, np.flatnonzero(g[start] == carrier)))
            if not 2 <= len(centers) <= 5 or centers != list(np.flatnonzero(g[end] == carrier)):
                continue
            if centers[0] < 1 or centers[-1] >= g.shape[1] - 1:
                continue
            if len(set(np.diff(centers))) != 1 or centers[1] - centers[0] < 4:
                continue
            try:
                footprint = np.zeros(g.shape, dtype=bool)
                alpha, beta = [], []
                for row, pattern, labels in ((start - 3, _DIAMOND, alpha), (end + 1, _CHECKER, beta)):
                    for c in centers:
                        footprint[row:row + 3, c - 1:c + 2] = True
                        labels.append(_terminal_label(g[row:row + 3, c - 1:c + 2], pattern, bg))
                _require(not np.any(d['cut'] & footprint), "fence covers a terminal")
                field = g.copy()
                field[footprint] = bg
                values = set(map(int, np.unique(field))) - {bg, control, carrier}
                _require(len(values) <= 1, "unrecognized routing colors")
                cross = next(iter(values)) if values else None
                _require(not (set(alpha + beta) & {bg, control, carrier, cross}), "labels overlap structural roles")
                routes, pairing, overpasses, links = _trace(field, d['cut'], start, end, centers, carrier, cross)
                _require(all(alpha[i] != beta[j] for i, j in enumerate(pairing)), "a cable needs two different signals")
                # Nothing unrelated may be hidden in the margin around the banks.
                outside = ~footprint
                outside[start:end + 1] = False
                _require(np.all(np.isin(g[outside], [bg, control])), "unrelated exterior content")
                d.update(grid=g, orientation=orientation, carrier=carrier, cross=cross,
                         start=start, end=end, centers=centers, alpha=alpha, beta=beta,
                         footprint=footprint, routes=routes, pairing=pairing,
                         overpasses=overpasses, links=links)
                candidates.append(d)
            except (AssertionError, StopIteration):
                continue
    _require(len(candidates) == 1, f"expected one circuit interpretation, got {len(candidates)}")
    return candidates[0]


def _owners(info):
    owners = defaultdict(list)
    for i, route in enumerate(info['routes']):
        for k, p in enumerate(route):
            axis = None
            if p in info['overpasses']:
                q = route[k + 1]
                axis = 0 if q[1] == p[1] else 1
            owners[p].append((i, axis))
    return owners


def _answer(info, *, aligned=False, ignore_region=None, reverse_depth=False):
    """Also provides explicit counterfactuals used to audit training evidence."""
    out = info['grid'].copy()
    for p, choices in _owners(info).items():
        if info['grid'][p] != info['carrier']:
            continue  # The crossing glyph arms are immutable auxiliary marks.
        if len(choices) == 2:
            axis = info['overpasses'][p]
            if reverse_depth:
                axis = 1 - axis
            choices = [(i, a) for i, a in choices if a == axis]
        _require(len(choices) == 1, "shared pixel without an overpass owner")
        i = choices[0][0]
        marked = bool(info['marked'][p]) if ignore_region is None else ignore_region
        j = i if aligned else info['pairing'][i]
        out[p] = info['alpha'][i] if marked else info['beta'][j]
    return _unview(out, info['orientation'])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Color visible carrier cells using only input geometry and endpoint clues."""
    return _answer(_parse(input_grid))


def _routing_moves(target, rng):
    # A vacant exterior track lets arbitrary cycles use orthogonal long runs.
    state = list(range(len(target))) + [None]
    moves = []
    for dest in map(int, rng.permutation(len(target))):
        if state[dest] == target[dest]:
            continue
        saved = state[dest]
        moves.append((dest, len(target)))
        state[-1], state[dest] = saved, None
        hole = dest
        while target[hole] != saved:
            old = state.index(target[hole])
            moves.append((old, hole))
            state[hole], state[old] = state[old], None
            hole = old
        moves.append((len(target), hole))
        state[hole], state[-1] = state[-1], None
    return moves


def _geometry(rng, n, target, *, detour=False, long_lead=False, mixed_depth=False):
    moves = _routing_moves(target, rng)
    if detour and len(moves) <= 6:
        moves = [(0, n), (n, 0)] + moves
    pitch_max = min(7, (26 - 3) // (n - 1))
    pitch = int(rng.integers(4, pitch_max + 1))
    left = int(rng.integers(2, 5))
    centers = [left + pitch * i for i in range(n)]
    spare_gap = int(rng.integers(3, min(6, 28 - centers[-1]) + 1))
    columns = centers + [centers[-1] + spare_gap]
    width = min(30, columns[-1] + int(rng.integers(2, 5)))
    top, bottom = 1, 1
    lead, tail = (6 if long_lead else 3), 3
    gaps = [2] * max(0, len(moves) - 1)
    minimum = top + 3 + lead + tail + sum(gaps) + 4 + bottom
    _require(minimum <= 30, "routing budget exhausted")
    budget = 30 - minimum
    pieces = [top, bottom, lead, tail] + gaps
    for _ in range(int(rng.integers(budget + 1))):
        pieces[int(rng.integers(len(pieces)))] += 1
    top, bottom, lead, tail, *gaps = pieces
    start = top + 3
    move_rows, row = [], start + lead
    for k in range(len(moves)):
        move_rows.append(row)
        if k < len(gaps):
            row += gaps[k]
    end = (move_rows[-1] if moves else start + lead) + tail
    height = end + 4 + bottom
    paths = [[(start, c)] for c in centers]
    state, previous = list(range(n)) + [None], start
    for row, (old, new) in zip(move_rows, moves):
        for lane, owner in enumerate(state):
            if owner is not None:
                paths[owner].extend((r, columns[lane]) for r in range(previous + 1, row + 1))
        owner = state[old]
        step = 1 if new > old else -1
        paths[owner].extend((row, c) for c in range(columns[old] + step, columns[new] + step, step))
        state[new], state[old] = owner, None
        previous = row
    for lane, owner in enumerate(state):
        if owner is not None:
            paths[owner].extend((r, columns[lane]) for r in range(previous + 1, end + 1))
    field = np.zeros((height, width), dtype=np.int8)
    visits = defaultdict(list)
    for i, path in enumerate(paths):
        for k, p in enumerate(path):
            field[p] = 1
            if p not in visits or i not in [v[0] for v in visits[p]]:
                axis = 0 if k + 1 == len(path) or path[k + 1][1] == p[1] else 1
                visits[p].append((i, axis))
    crossings = {}
    crossing_points = [p for p, owners in visits.items() if len(owners) > 1]
    # One vertical overpass among horizontal ones cannot create a spurious
    # opposite marker pair through neighboring vertical crossing glyphs.
    mixed = len(crossing_points) >= 2 and (mixed_depth or rng.random() < .35)
    exceptional = crossing_points[int(rng.integers(len(crossing_points)))] if mixed else None
    common_axis = 1 if mixed else int(rng.integers(2))
    for (r, c), owners in visits.items():
        if len(owners) > 1:
            _require(len(owners) == 2 and {a for _, a in owners} == {0, 1}, "invalid crossing construction")
            axis = 0 if (r, c) == exceptional else common_axis
            crossings[r, c] = axis
            for d in ((0, 2), (1, 3))[axis]:
                dr, dc = _DIRS[d]
                field[r + dr, c + dc] = 2
    footprint = np.zeros(field.shape, dtype=bool)
    for r in (start - 3, end + 1):
        for c in centers:
            footprint[r:r + 3, c - 1:c + 2] = True
    return dict(field=field, paths=paths, visits=visits, crossings=crossings, centers=centers,
                start=start, end=end, lead=lead, footprint=footprint)


def _line(p, q):
    r, c = p
    r1, c1 = q
    dr, dc = abs(r1 - r), abs(c1 - c)
    sr, sc = (1 if r < r1 else -1), (1 if c < c1 else -1)
    error, points = dc - dr, []
    while True:
        points.append((r, c))
        if (r, c) == q:
            return points
        twice = 2 * error
        if twice > -dr:
            error -= dr
            c += sc
        if twice < dc:
            error += dc
            r += sr


def _fence(rng, geom, kind):
    field, centers = geom['field'], geom['centers']
    h, w = field.shape
    cut = np.zeros(field.shape, dtype=bool)
    gap_cols = [int(rng.integers(a + 2, b - 1)) for a, b in zip(centers, centers[1:])]
    # The initial lead is a safe area of straight parallel cable segments.
    low, high = geom['start'] + 2, geom['start'] + geom['lead'] - 2
    if kind == 'vertical':
        cut[:, gap_cols[int(rng.integers(len(gap_cols)))]] = True
    elif kind == 'horizontal':
        cut[int(rng.integers(geom['start'] + 2, geom['end'] - 1)), :] = True
    elif kind == 'diagonal':
        a = int(rng.integers(low, max(low + 1, high + 1)))
        b = max(low, min(high, a + int(rng.choice([-2, -1, 1, 2]))))
        if a == b:
            b = a + 1
        for p in _line((a, 0), (b, w - 1)):
            cut[p] = True
    elif kind == 'elbow':
        c = gap_cols[int(rng.integers(len(gap_cols)))]
        r = low
        cut[:r + 1, c] = True
        cut[r, c:] = True
        if rng.random() < .5:
            cut = cut[:, ::-1].copy()
        if rng.random() < .5:
            cut = cut[::-1].copy()
    elif kind == 'stepped':
        columns = sorted(map(int, rng.choice(gap_cols, min(len(gap_cols), int(rng.integers(1, 4))), replace=False)))
        row, last = low, 0
        for c in columns:
            cut[row, last:c + 1] = True
            new = high if row == low else low
            cut[min(row, new):max(row, new) + 1, c] = True
            row, last = new, c
        cut[row, last:] = True
        if rng.random() < .5:
            cut = cut[:, ::-1].copy()
    else:
        raise ValueError("unknown fence kind")
    return cut


def _compatible_fence(rng, geom, kind):
    field = geom['field']
    protected = geom['footprint'].copy()
    protected[geom['start']:geom['start'] + 2] |= field[geom['start']:geom['start'] + 2] != 0
    protected[geom['end'] - 1:geom['end'] + 1] |= field[geom['end'] - 1:geom['end'] + 1] != 0
    for r, c in geom['crossings']:
        protected[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True
    for _ in range(100):
        cut = _fence(rng, geom, kind)
        if np.any(cut & protected) or not np.any(cut & (field == 1)):
            continue
        try:
            regions, path = _cut_info(cut)
            sites = []
            for region in regions:
                candidates = []
                for r, c in np.argwhere(region & (field == 0) & ~geom['footprint']):
                    p = (int(r), int(c))
                    if p in ((0, 0), (0, field.shape[1] - 1), (field.shape[0] - 1, 0),
                             (field.shape[0] - 1, field.shape[1] - 1)):
                        continue
                    if min(max(abs(r - a), abs(c - b)) for a, b in path) >= 2:
                        candidates.append(p)
                _require(candidates, "no marker site")
                sites.append(candidates)
            test = field.copy()
            test[cut] = 3
            traced = _trace(test, cut, geom['start'], geom['end'], geom['centers'], 1,
                            2 if np.any(field == 2) else None)
            expected = [geom['centers'].index(p[-1][1]) for p in geom['paths']]
            _require(traced[1] == expected and traced[3], "fence obscures an unspecified pairing")
            return cut, regions, sites
        except AssertionError:
            continue
    raise RuntimeError("could not fit the requested fence geometry")


def _construct_pair(rng, n, kind, palette, orientation, *, lesson=False):
    target = list(map(int, rng.permutation(n)))
    if lesson:
        target = list(range(n))
        target[0], target[-1] = target[-1], target[0]
    elif rng.random() < .2:
        target = list(range(n))
    geom = _geometry(rng, n, target, detour=lesson or rng.random() < .25,
                     long_lead=kind in ('diagonal', 'stepped'), mixed_depth=lesson)
    cut, regions, sites = _compatible_fence(rng, geom, kind)
    side = int(rng.integers(2))
    marker = sites[side][int(rng.integers(len(sites[side])))]
    bg, wire, crossing, control = palette[:4]
    label_pool = palette[4:]
    if lesson:
        alpha = list(label_pool[:n])
        beta_route = alpha[1:] + alpha[:1]
    else:
        count = int(rng.integers(2, min(6, n + 2) + 1))
        pool = label_pool[:count]
        alpha = list(map(int, rng.choice(pool, n)))
        beta_route = [int(rng.choice([v for v in pool if v != a])) for a in alpha]
    inp = np.asarray(palette, dtype=np.int8)[geom['field']]
    inp[cut] = control
    inp[marker] = control
    for i, path in enumerate(geom['paths']):
        for row, c, pattern, color in ((geom['start'] - 3, geom['centers'][i], _DIAMOND, alpha[i]),
                                      (geom['end'] + 1, path[-1][1], _CHECKER, beta_route[i])):
            inp[row:row + 3, c - 1:c + 2] = np.where(pattern, color, bg)
    # Ground truth uses the explicitly constructed routes and overpass ownership.
    out = inp.copy()
    for p, owners in geom['visits'].items():
        if inp[p] != wire:
            continue
        if len(owners) == 2:
            owners = [v for v in owners if v[1] == geom['crossings'][p]]
        i = owners[0][0]
        out[p] = alpha[i] if regions[side][p] else beta_route[i]
    return {'input': _unview(inp, orientation), 'output': _unview(out, orientation)}


def _opposite_marker(pair, rng):
    d = _parse(pair['input'])
    g = d['grid'].copy()
    g[d['marker']] = d['background']
    other = d['regions'][1 - d['side']]
    path = d['fence_path']
    sites = [(int(r), int(c)) for r, c in np.argwhere(other & (g == d['background']) & ~d['footprint'])
             if (r, c) not in ((0, 0), (0, g.shape[1] - 1), (g.shape[0] - 1, 0),
                               (g.shape[0] - 1, g.shape[1] - 1))
             and min(max(abs(r - a), abs(c - b)) for a, b in path) >= 2]
    _require(sites, "no opposite marker site")
    marker = sites[int(rng.integers(len(sites)))]
    g[marker] = d['control']
    inp = _unview(g, d['orientation'])
    # The twin is the known opposite signal at every visible wire pixel.
    out = _answer(d, ignore_region=False)
    oc = _view(out, d['orientation'])
    od = _view(_answer(d, ignore_region=True), d['orientation'])
    oc[d['regions'][1 - d['side']] & (d['grid'] == d['carrier'])] = od[
        d['regions'][1 - d['side']] & (d['grid'] == d['carrier'])]
    oc[d['marker']] = d['background']
    oc[marker] = d['control']
    return {'input': inp, 'output': _unview(oc, d['orientation'])}


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    vary_palette = bool(rng.integers(2))
    shift = str(rng.choice(['fence', 'bundle', 'orientation']))
    base_orientation = int(rng.integers(4))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            if split == 'train' and index == 1:
                pair = _opposite_marker(puzzle['train'][0], rng)
            else:
                lesson = split == 'train' and index == 0
                n = 3 if lesson else int(rng.integers(2, 5 if split == 'train' else 6))
                if split == 'test' and shift == 'bundle':
                    n = 5
                kind = 'vertical' if lesson else str(rng.choice(['vertical', 'horizontal', 'diagonal', 'elbow', 'stepped']))
                if split == 'train' and shift == 'fence' and kind == 'stepped':
                    kind = str(rng.choice(['horizontal', 'diagonal', 'elbow']))
                if split == 'test' and shift == 'fence':
                    kind = 'stepped'
                orientation = base_orientation
                if split == 'test' and shift == 'orientation':
                    orientation = (base_orientation + 2) % 4
                elif shift != 'orientation' and (index >= 2 or split == 'test'):
                    orientation = int(rng.integers(4))
                colors = list(map(int, rng.permutation(10))) if vary_palette and not lesson else palette
                for _ in range(80):
                    try:
                        pair = _construct_pair(rng, n, kind, colors, orientation, lesson=lesson)
                        _parse(pair['input'])
                        key = (pair['input'].shape, pair['input'].tobytes())
                        if key not in seen:
                            break
                    except (AssertionError, RuntimeError):
                        continue
                else:
                    raise RuntimeError("bounded construction exhausted")
            key = (pair['input'].shape, pair['input'].tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            puzzle[split].append(pair)
    return puzzle


def inspect_pair(pair):
    d = _parse(pair['input'])
    steps = [(q[0] - p[0], q[1] - p[1]) for p, q in zip(d['fence_path'], d['fence_path'][1:])]
    bends = sum(a != b for a, b in zip(steps, steps[1:]))
    region_runs, gap_counts = [], []
    for route in d['routes']:
        sides = [bool(d['marked'][p]) for p in route]
        region_runs.append(1 + sum(a != b for a, b in zip(sides, sides[1:])))
        gap_counts.append(sum(abs(p[0] - q[0]) + abs(p[1] - q[1]) > 1 for p, q in zip(route, route[1:])))
    return dict(shape=tuple(d['grid'].shape), orientation=d['orientation'], routes=len(d['routes']),
                pitch=d['centers'][1] - d['centers'][0], pairing=tuple(d['pairing']),
                fence_bends=bends, side=d['side'], region_runs=tuple(region_runs),
                gap_counts=tuple(gap_counts), hidden_links=len(d['links']), crossings=len(d['overpasses']),
                depth_axes=tuple(sorted(set(d['overpasses'].values()))),
                palette=(d['background'], d['carrier'], d['cross'], d['control']),
                both_regions_visible=all(np.any(m & (d['grid'] == d['carrier'])) for m in d['regions']),
                distinct_diamond_labels=len(set(d['alpha'])), distinct_checker_labels=len(set(d['beta'])))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, evidence, depth_axes = set(), set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "bad pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, "answer changes dimensions")
            key = (inp.shape, inp.tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            d = _parse(inp)
            _require(np.array_equal(_answer(d), out), "incorrect regional cable colors")
            canonical_out = _view(out, d['orientation'])
            unchanged = d['grid'] != d['carrier']
            _require(np.array_equal(canonical_out[unchanged], d['grid'][unchanged]), "changed noncarrier cell")
            _require(np.all(np.isin(canonical_out[~unchanged], d['alpha'] + d['beta'])), "wire color not from an endpoint")
            _require(not np.array_equal(inp, out), "identity pair")
            _require(d['links'], "fence must actually occlude cables")
            if split == 'train':
                depth_axes.update(d['overpasses'].values())
                if not np.array_equal(_answer(d, aligned=True), out):
                    evidence.add('routing')
                if not np.array_equal(_answer(d, ignore_region=True), out):
                    evidence.add('checker_region')
                if not np.array_equal(_answer(d, ignore_region=False), out):
                    evidence.add('diamond_region')
                if not np.array_equal(_answer(d, reverse_depth=True), out):
                    evidence.add('depth')
                if max(inspect_pair(pair)['region_runs']) >= 3:
                    evidence.add('reentry')
    _require(evidence == {'routing', 'checker_region', 'diamond_region', 'depth', 'reentry'},
             "training misses an essential interaction")
    _require(depth_axes == {0, 1}, "training must show both overpass axes")
    # The first two demonstrations isolate the marker's effect.
    a, b = puzzle['train'][:2]
    da, db = _parse(a['input']), _parse(b['input'])
    _require(da['orientation'] == db['orientation'] and np.array_equal(da['cut'], db['cut']), "missing marker twin")
    ga, gb = da['grid'].copy(), db['grid'].copy()
    ga[da['marker']], gb[db['marker']] = da['background'], db['background']
    _require(np.array_equal(ga, gb) and da['side'] != db['side'], "twins change more than the selected side")
    oa, ob = _view(np.asarray(a['output']), da['orientation']), _view(np.asarray(b['output']), db['orientation'])
    _require(np.all(oa[ga == da['carrier']] != ob[ga == da['carrier']]), "marker swap does not switch every visible signal")


def _puzzle_key(p):
    return tuple((s, tuple((np.asarray(q[k]).shape, np.asarray(q[k]).tobytes())
                          for q in p[s] for k in ('input', 'output'))) for s in ('train', 'test'))


def self_test():
    """Bounded semantic, counterfactual, equivariance, and diversity checks."""
    metrics, signatures = [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(_puzzle_key(p))
        for split in ('train', 'test'):
            for pair in p[split]:
                metrics.append(inspect_pair(pair))
                _require(np.array_equal(solve(pair['input']), pair['output']), "solve mismatch")
        pair = p['test'][0]
        for transform in (lambda x: x.T, lambda x: x[::-1], lambda x: x[:, ::-1]):
            _require(np.array_equal(solve(transform(pair['input'])), transform(pair['output'])), "geometric equivariance failed")
        # Arbitrary bijective recoloring must preserve the semantic rule.
        lut = np.random.default_rng(seed + 1000).permutation(10)
        _require(np.array_equal(solve(lut[pair['input']]), lut[pair['output']]), "palette equivariance failed")
    _require(len(signatures) == 42, "seed diversity collapsed")
    _require(len({m['shape'] for m in metrics}) >= 20, "shape diversity collapsed")
    _require({m['routes'] for m in metrics} == {2, 3, 4, 5}, "bundle counts collapsed")
    _require({m['orientation'] for m in metrics} == {0, 1, 2, 3}, "orientation collapsed")
    _require({m['side'] for m in metrics} == {0, 1}, "marker selection collapsed")
    _require(any(m['crossings'] == 0 for m in metrics) and any(m['crossings'] >= 6 for m in metrics), "routing spectrum collapsed")
    _require(any(m['fence_bends'] >= 4 for m in metrics), "multiple-bend fences absent")
    _require(len({m['pairing'] for m in metrics}) >= 20, "pairing diversity collapsed")
    _require(len({m['palette'] for m in metrics}) >= 30, "palette diversity collapsed")
    _require(any(max(m['gap_counts']) >= 2 for m in metrics), "repeated occlusion absent")
    _require(any(m['distinct_diamond_labels'] < m['routes'] for m in metrics), "repeated labels absent")
    # Every five-route permutation, without depending on a fence or RNG state.
    rng = np.random.default_rng(5041)
    for target in permutations(range(5)):
        geom = _geometry(rng, 5, target)
        blank_cut = np.zeros(geom['field'].shape, dtype=bool)
        routes, pairing, _, _ = _trace(geom['field'], blank_cut, geom['start'], geom['end'], geom['centers'],
                                      1, 2 if np.any(geom['field'] == 2) else None)
        _require(pairing == [target.index(i) for i in range(5)], "permutation trace failed")
    _require(_puzzle_key(generate_puzzle(15, 7, 3)) == _puzzle_key(generate_puzzle(15, 7, 3)), "nondeterminism")
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(80 + nt + nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (3, 0), (5, 5), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # A stray auxiliary cell on an ordinary wire is not a valid layer glyph.
    pair = generate_puzzle(0)['train'][0]
    info = _parse(pair['input'])
    glyph_cells = {(r + _DIRS[d][0], c + _DIRS[d][1])
                   for (r, c) in info['overpasses'] for d in range(4)}
    ordinary = next(p for p in _owners(info) if info['grid'][p] == info['carrier']
                    and p not in glyph_cells and p not in info['overpasses']
                    and info['start'] < p[0] < info['end'])
    malformed = info['grid'].copy()
    malformed[ordinary] = info['cross']
    try:
        solve(_unview(malformed, info['orientation']))
    except AssertionError:
        pass
    else:
        raise AssertionError("stray layer mark accepted")
    # Removing the marker must not leave a supposedly solvable unlabeled side.
    malformed = info['grid'].copy()
    malformed[info['marker']] = info['background']
    try:
        solve(_unview(malformed, info['orientation']))
    except AssertionError:
        pass
    else:
        raise AssertionError("absent control marker accepted")
    bad = generate_puzzle(0)
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("noncarrier corruption accepted")
    return {'seeds': 42, 'pairs': len(metrics), 'distinct_puzzles': len(signatures),
            'shapes': len({m['shape'] for m in metrics}), 'pairings': len({m['pairing'] for m in metrics}),
            'routing_counts': sorted({m['routes'] for m in metrics})}


if __name__ == '__main__':
    print(self_test())
