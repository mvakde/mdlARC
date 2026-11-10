"""Defect ports: local odd glyphs constrain and stencil an ordered route.

Only standard-library modules and NumPy are used. No parent generator is imported.
"""
from __future__ import annotations

from collections import Counter
import json

import numpy as np


HOST_SEEDS = (0, 903784223, 361876823, 906045179, 609908526, 302413830,
              976516232, 601056354, 31397035, 215681215, 168248917, 464261826,
              229432699, 637015478, 642181896, 622279469, 526903607, 272876927,
              908611182, 277993620, 252024552, 481815860, 939876403, 967323941,
              658735625, 438662662, 527244318, 426889017, 567120563, 174424508,
              390780860, 497355228, 910573454, 456773388, 598744894, 593435535,
              529673108, 117874089, 602640646, 500998305, 683507037, 340833780)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "integer colors required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _puzzle_signature(puzzle):
    return tuple(tuple((_signature(p['input']), _signature(p['output']))
                       for p in puzzle[s]) for s in ('train', 'test'))


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    groups = []
    for r, c in zip(*np.nonzero(mask)):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        stack, points = [(r, c)], []
        seen[r, c] = True
        while stack:
            y, x = stack.pop()
            points.append((y, x))
            for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1]
                        and mask[yy, xx] and not seen[yy, xx]):
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        groups.append(points)
    return groups


def _station_shape(h, w, copies, orientation):
    return ((h + 4, copies * (w + 1) + 3) if orientation == 'h'
            else (copies * (h + 1) + 3, w + 4))


def _panel_origins(h, w, copies, orientation):
    return [(2, 2 + i * (w + 1)) if orientation == 'h'
            else (2 + i * (h + 1), 2) for i in range(copies)]


def _parse_station(block, y, x):
    H, W = block.shape
    carrier = int(block[0, 0])
    label = int(block[0, W // 2])
    _require(label != carrier, "station label must differ from carrier")
    border = block.copy()
    border[0, W // 2] = carrier
    _require(bool(np.all(border[[0, -1], :] == carrier)
                  and np.all(border[:, [0, -1]] == carrier)), "bad station border")
    body = block[1:-1, 1:-1]
    divider = int(body[0, 0])
    _require(divider not in (carrier, label), "distinct divider required")
    rs = np.flatnonzero(np.all(body == divider, axis=1))
    cs = np.flatnonzero(np.all(body == divider, axis=0))
    _require(len(rs) >= 2 and len(cs) >= 2, "missing glyph lattice")
    _require(rs[0] == cs[0] == 0 and rs[-1] == body.shape[0]-1
             and cs[-1] == body.shape[1]-1, "incomplete glyph lattice")
    _require(bool(np.all(np.diff(rs) == rs[1])
                  and np.all(np.diff(cs) == cs[1])), "irregular glyph lattice")
    nr, nc = len(rs)-1, len(cs)-1
    _require((nr == 1 and nc in (3, 5)) or (nc == 1 and nr in (3, 5)),
             "glyphs must form a three- or five-copy strip")
    h, w = int(rs[1]-1), int(cs[1]-1)
    _require(2 <= h <= 4 and 2 <= w <= 4, "glyph dimensions must be 2..4")
    orientation = 'h' if nr == 1 else 'v'
    copies = max(nr, nc)
    _require((H, W) == _station_shape(h, w, copies, orientation), "bad station geometry")
    origins = _panel_origins(h, w, copies, orientation)
    panels = [block[r:r+h, c:c+w].copy() for r, c in origins]
    _require(all(np.all(p != divider) for p in panels), "divider inside a glyph")
    counts = Counter(p.tobytes() for p in panels)
    modal, frequency = counts.most_common(1)[0]
    _require(frequency == copies-1 and len(counts) == 2,
             "each station needs one visible exception and a strict majority")
    master = next(p for p in panels if p.tobytes() == modal)
    index = next(i for i, p in enumerate(panels) if p.tobytes() != modal)
    variant = panels[index]
    colors = set(map(int, np.unique(master)))
    _require(len(colors) == 2 and carrier in colors, "two-color majority glyph required")
    ink = next(iter(colors - {carrier}))
    _require(ink not in (label, divider), "ink and labels must be distinct")
    _require(set(map(int, np.unique(variant))) <= colors, "invalid exceptional glyph color")
    delta = master != variant
    rows = tuple(bool(v) for v in np.any(delta, axis=1))
    cols = tuple(bool(v) for v in np.any(delta, axis=0))
    return dict(y=y, x=x, label=label, master=master, variant=variant,
                delta=delta, rows=rows, cols=cols, exception=index, origins=origins,
                h=h, w=w, H=H, W=W, copies=copies, orientation=orientation,
                carrier=carrier, divider=divider, ink=ink)


def _parse_input(grid):
    bg = int(grid[0, 0])
    _require(all(np.all(edge == bg) for edge in
                 (grid[0], grid[-1], grid[:, 0], grid[:, -1])), "background frame required")
    stations, tokens = [], []
    for component in _components(grid != bg):
        if len(component) == 1:
            tokens.append(component[0])
            continue
        ys, xs = zip(*component)
        y, x = min(ys), min(xs)
        H, W = max(ys)-y+1, max(xs)-x+1
        _require(len(component) == H*W, "stations must be filled rectangles")
        stations.append(_parse_station(grid[y:y+H, x:x+W], y, x))
    stations.sort(key=lambda s: (s['y'], s['x']))
    _require(3 <= len(stations) <= 8, "need three to eight stations")
    s0 = stations[0]
    structural = {bg, s0['carrier'], s0['divider'], s0['ink']}
    _require(len(structural) == 4, "four distinct structural colors required")
    convention = tuple(s0[k] for k in ('h', 'w', 'H', 'W', 'copies',
                                       'orientation', 'carrier', 'divider', 'ink'))
    for station in stations:
        _require(tuple(station[k] for k in ('h', 'w', 'H', 'W', 'copies',
                                          'orientation', 'carrier', 'divider', 'ink'))
                 == convention, "station geometry and structural roles must agree")
        _require(station['label'] not in structural, "label uses a structural color")
    tokens.sort(key=lambda p: p[1])
    _require(3 <= len(tokens) <= len(stations), "invalid key length")
    ky = tokens[0][0]
    _require(all(y == ky for y, x in tokens), "key must be horizontal")
    _require(all(b[1]-a[1] == 2 for a, b in zip(tokens, tokens[1:])), "key spacing must be two")
    _require(ky >= max(s['y'] + s['H'] for s in stations) + 1,
             "key must be separated below the station field")
    key = [int(grid[y, x]) for y, x in tokens]
    _require(len(set(key)) >= 2 and not structural.intersection(key), "invalid key palette")
    return dict(bg=bg, stations=stations, key=key, key_y=ky)


def _edge_region(a, b):
    """Return the central glyph's rectangular projection across an aligned gap."""
    if a['y'] == b['y'] and abs(a['x']-b['x']) > a['W']:
        return ('h', (slice(a['y']+(a['H']-a['h'])//2,
                            a['y']+(a['H']-a['h'])//2+a['h']),
                      slice(min(a['x'], b['x'])+a['W'], max(a['x'], b['x']))))
    if a['x'] == b['x'] and abs(a['y']-b['y']) > a['H']:
        return ('v', (slice(min(a['y'], b['y'])+a['H'], max(a['y'], b['y'])),
                      slice(a['x']+(a['W']-a['w'])//2,
                            a['x']+(a['W']-a['w'])//2+a['w'])))
    return None


def _lane_pixels(a, region):
    axis, (rs, cs) = region
    if axis == 'h':
        return frozenset((rs.start+i, c) for i, active in enumerate(a['rows']) if active
                         for c in range(cs.start, cs.stop))
    return frozenset((r, cs.start+i) for i, active in enumerate(a['cols']) if active
                     for r in range(rs.start, rs.stop))


def _graphs(grid, info):
    stations = info['stations']
    geometric = [[] for _ in stations]
    compatible = [[] for _ in stations]
    pixels = {}
    for i, a in enumerate(stations):
        for j, b in enumerate(stations):
            if i == j:
                continue
            region = _edge_region(a, b)
            if region is None or not np.all(grid[region[1]] == info['bg']):
                continue
            geometric[i].append(j)
            pixels[i, j] = _lane_pixels(a, region)
            profile = 'rows' if region[0] == 'h' else 'cols'
            if a[profile] == b[profile]:
                compatible[i].append(j)
    return geometric, compatible, pixels


def _routes(graph, labels, key, pixels, limit=2):
    answers = []
    candidates = [[i for i, label in enumerate(labels) if label == k] for k in key]
    def visit(path, painted):
        if len(answers) >= limit:
            return
        if len(path) == len(key):
            answers.append(tuple(path))
            return
        for i in candidates[len(path)]:
            if i in path or (path and i not in graph[path[-1]]):
                continue
            edge = pixels[path[-1], i] if path else frozenset()
            if painted.intersection(edge):
                continue
            visit(path+[i], painted | edge)
    visit([], frozenset())
    return answers


def _decode(grid):
    info = _parse_input(grid)
    geometric, graph, pixels = _graphs(grid, info)
    labels = [s['label'] for s in info['stations']]
    routes = _routes(graph, labels, info['key'], pixels)
    _require(len(routes) == 1, "key and ports must determine exactly one route")
    info.update(route=routes[0], geometric=geometric, graph=graph, pixels=pixels)
    return info


def _paint(grid, stations, path):
    out = grid.copy()
    occupied = set()
    for i, j in zip(path, path[1:]):
        a, b = stations[i], stations[j]
        region = _edge_region(a, b)
        _require(region is not None, "unaligned connection")
        p = _lane_pixels(a, region)
        _require(not occupied.intersection(p), "connectors cannot overlap")
        occupied.update(p)
        for r, c in p:
            out[r, c] = a['label']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read the entire rule from this input; no seed or training state is used."""
    grid = _grid(input_grid)
    info = _decode(grid)
    return _paint(grid, info['stations'], info['route'])


def _visible_graph(cells):
    graph = [[] for _ in cells]
    for i, a in enumerate(cells):
        for j, b in enumerate(cells):
            if i == j:
                continue
            if a[0] == b[0]:
                lo, hi = sorted((a[1], b[1]))
                clear = not any(c[0] == a[0] and lo < c[1] < hi for c in cells)
            elif a[1] == b[1]:
                lo, hi = sorted((a[0], b[0]))
                clear = not any(c[1] == a[1] and lo < c[0] < hi for c in cells)
            else:
                continue
            if clear:
                graph[i].append(j)
    return graph


def _crosses(a, b, c, d):
    if a[0] == b[0] and c[1] == d[1]:
        return (min(a[1], b[1]) < c[1] < max(a[1], b[1])
                and min(c[0], d[0]) < a[0] < max(c[0], d[0]))
    if a[1] == b[1] and c[0] == d[0]:
        return _crosses(c, d, a, b)
    return False


def _sample_path(rng, cells, graph, length, both):
    budget = 1200
    def visit(path, axes):
        nonlocal budget
        budget -= 1
        if budget < 0:
            return None
        if len(path) == length:
            return path if not both or len(set(axes)) == 2 else None
        for j in rng.permutation(graph[path[-1]]):
            j = int(j)
            if j in path:
                continue
            a, b = cells[path[-1]], cells[j]
            if any(_crosses(a, b, cells[u], cells[v]) for u, v in zip(path, path[1:])):
                continue
            axis = 'h' if a[0] == b[0] else 'v'
            result = visit(path+[j], axes+[axis])
            if result is not None:
                return result
        return None
    for start in rng.permutation(len(cells)):
        result = visit([int(start)], [])
        if result is not None:
            return result
        if budget < 0:
            break
    return None


def _profiles(rng, cells, path, h, w, require_separated):
    profiles = {}
    for axis, dimension in (('h', h), ('v', w)):
        parent = list(range(len(cells)))
        def root(i):
            while parent[i] != i:
                i = parent[i]
            return i
        for i, j in zip(path, path[1:]):
            if (cells[i][0] == cells[j][0]) == (axis == 'h'):
                parent[root(j)] = root(i)
        choices = {}
        for i in range(len(cells)):
            group = root(i)
            if group not in choices:
                choices[group] = int(rng.integers(1, 1 << dimension))
        if require_separated and dimension >= 3:
            links = [(i, j) for i, j in zip(path, path[1:])
                     if (cells[i][0] == cells[j][0]) == (axis == 'h')]
            if links:
                i, j = links[int(rng.integers(len(links)))]
                choices[root(i)] = 1 | (1 << (dimension-1))
        profiles[axis] = [tuple(bool(choices[root(i)] & (1 << k))
                                for k in range(dimension)) for i in range(len(cells))]
    return list(zip(profiles['h'], profiles['v']))


def _delta_mask(rng, rows, cols):
    rr = list(rng.permutation(np.flatnonzero(rows)))
    cc = list(rng.permutation(np.flatnonzero(cols)))
    delta = np.zeros((len(rows), len(cols)), dtype=bool)
    for k in range(max(len(rr), len(cc))):
        delta[int(rr[k % len(rr)]), int(cc[k % len(cc)])] = True
    cap = max(max(len(rr), len(cc)), int(delta.size * .65))
    for k in rng.permutation(len(rr)*len(cc)):
        r, c = rr[int(k)//len(cc)], cc[int(k)%len(cc)]
        if delta.sum() < cap and rng.random() < .30:
            delta[int(r), int(c)] = True
    return delta


def _master(rng, delta, style, edit_mode):
    h, w = delta.shape
    rr, cc = np.indices((h, w))
    if style == 0:
        mask = rng.random((h, w)) < float(rng.uniform(.2, .8))
    elif style == 1:
        mask = (rr+cc) % int(rng.integers(2, 4)) == 0
    elif style == 2:
        mask = (rr == int(rng.integers(h))) | (cc == int(rng.integers(w)))
    elif style == 3:
        mask = (rr == 0) | (cc == 0) | (rr == h-1) | (cc == w-1)
    else:
        mask = (rr == cc) | (rr+cc == min(h, w)-1)
    if rng.random() < .3:
        mask = ~mask
    points = np.flatnonzero(delta)
    if edit_mode == 'add':
        mask[delta] = False
    elif edit_mode == 'remove':
        mask[delta] = True
    elif edit_mode == 'balanced' and len(points) >= 2 and len(points) % 2 == 0:
        mask[delta] = False
        mask.flat[rng.choice(points, len(points)//2, replace=False)] = True
    # Reference glyphs always have both colors; delta never fills its rectangle.
    if not mask.any():
        outside = np.flatnonzero(~delta)
        mask.flat[int(rng.choice(outside))] = True
    if mask.all():
        outside = np.flatnonzero(~delta)
        mask.flat[int(rng.choice(outside))] = False
    return mask


def _axis_positions(rng, count, size, limit, expanded):
    gaps = np.ones(count-1, dtype=int)
    slack = limit - count*size - len(gaps)
    for j in rng.permutation(len(gaps)):
        high = min(slack, 6 if expanded else 4)
        extra = int(rng.integers(high+1))
        gaps[int(j)] += extra
        slack -= extra
    return [0] + list(map(int, np.cumsum(size+gaps)))


def _draw(rng, cells, path, labels, profiles, geometry, palette, expanded,
          evidence=False):
    h, w, copies, orientation = geometry
    H, W = _station_shape(*geometry)
    nr = max(r for r, c in cells)+1
    nc = max(c for r, c in cells)+1
    ys = _axis_positions(rng, nr, H, 26, expanded)
    xs = _axis_positions(rng, nc, W, 28, expanded)
    bh, bw = ys[-1]+H, xs[-1]+W
    height = int(rng.integers(bh+4, 31))
    width = int(rng.integers(max(bw+2, 2*len(path)+1), 31))
    top = int(rng.integers(1, height-bh-2))
    left = int(rng.integers(1, width-bw))
    bg, carrier, divider, ink = palette[:4]
    grid = np.full((height, width), bg, dtype=np.int8)
    positions, stations = [], []
    for i, ((r, c), (rows, cols)) in enumerate(zip(cells, profiles)):
        y, x = top+ys[r], left+xs[c]
        positions.append((y, x))
        block = np.full((H, W), carrier, dtype=np.int8)
        block[1:-1, 1:-1] = divider
        block[0, W//2] = labels[i]
        delta = _delta_mask(rng, rows, cols)
        mode = ('add', 'remove', 'balanced')[i % 3] if i < 3 else str(
            rng.choice(('add', 'remove', 'balanced', 'mixed')))
        if evidence and i in path[:3]:
            mode = ('add', 'remove', 'balanced')[path.index(i)]
        master = _master(rng, delta, int(rng.integers(5)), mode)
        variant = master ^ delta
        exception = int(rng.integers(copies))
        if evidence and i == path[0] and exception == copies//2:
            exception = int(rng.choice([j for j in range(copies) if j != copies//2]))
        for j, (py, px) in enumerate(_panel_origins(h, w, copies, orientation)):
            mask = variant if j == exception else master
            block[py:py+h, px:px+w] = np.where(mask, ink, carrier)
        grid[y:y+H, x:x+W] = block
        stations.append(dict(y=y, x=x, label=labels[i], rows=rows, cols=cols,
                             h=h, w=w, H=H, W=W, orientation=orientation))
    key_y = int(rng.integers(top+bh+1, height-1))
    key_x = int(rng.integers(1, width-(2*len(path)-1)))
    grid[key_y, key_x:key_x+2*len(path):2] = [labels[i] for i in path]
    return dict(input=grid, output=_paint(grid, stations, path))


def _profile_graph(cells, profiles):
    geometric = _visible_graph(cells)
    compatible = [[] for _ in cells]
    for i, candidates in enumerate(geometric):
        for j in candidates:
            axis = 0 if cells[i][0] == cells[j][0] else 1
            if profiles[i][axis] == profiles[j][axis]:
                compatible[i].append(j)
    # These abstract links only check station uniqueness. Pixel overlap and
    # exact visibility are checked again on the drawn grid before acceptance.
    empty_pixels = {(i, j): frozenset() for i, candidates in enumerate(geometric)
                    for j in candidates}
    return geometric, compatible, empty_pixels


def _labels(rng, cells, path, profiles, colors, gate):
    geometric, graph, pixels = _profile_graph(cells, profiles)
    for attempt in range(100):
        k = int(rng.integers(2, min(6, len(path))+1))
        pool = list(map(int, rng.permutation(colors)[:k]))
        key = pool + [int(rng.choice(pool)) for _ in range(len(path)-k)]
        key = list(map(int, rng.permutation(key)))
        if len(set(key)) < 2:
            continue
        labels = [int(rng.choice(colors)) for _ in cells]
        for i, color in zip(path, key):
            labels[i] = color
        for i in range(len(cells)):
            if i not in path and rng.random() < .7:
                labels[i] = int(rng.choice(key))
        routes = _routes(graph, labels, key, pixels)
        if routes != [tuple(path)]:
            continue
        if gate and len(_routes(geometric, labels, key, pixels)) < 2:
            continue
        return labels
    return None


def _features(grid, info):
    stations, route = info['stations'], info['route']
    axes, separated, colors_differ, projection_witness = set(), False, False, False
    changes = []
    for s in stations:
        added = int(np.count_nonzero((s['master'] == s['carrier']) & s['delta']))
        removed = int(np.count_nonzero((s['master'] == s['ink']) & s['delta']))
        changes.append((added, removed))
    for i, j in zip(route, route[1:]):
        a, b = stations[i], stations[j]
        axis = _edge_region(a, b)[0]
        axes.add(axis)
        profile = a['rows'] if axis == 'h' else a['cols']
        active = np.flatnonzero(profile)
        separated |= bool(len(active) > 1 and np.any(np.diff(active) > 1))
        colors_differ |= a['label'] != b['label']
        projection_witness |= not np.array_equal(a['delta'], b['delta'])
    labels = [s['label'] for s in stations]
    geometric_routes = _routes(info['geometric'], labels, info['key'], info['pixels'])
    gate_witness = any(not np.array_equal(_paint(grid, stations, r),
                                         _paint(grid, stations, route))
                       for r in geometric_routes if r != route)
    overlap_graph = [[] for _ in stations]
    for i, candidates in enumerate(info['geometric']):
        for j in candidates:
            axis = _edge_region(stations[i], stations[j])[0]
            k = 'rows' if axis == 'h' else 'cols'
            if any(a and b for a, b in zip(stations[i][k], stations[j][k])):
                overlap_graph[i].append(j)
    overlap_routes = _routes(overlap_graph, labels, info['key'], info['pixels'])
    overlap_gate = any(not np.array_equal(_paint(grid, stations, r),
                                          _paint(grid, stations, route))
                       for r in overlap_routes if r != route)
    return dict(axes=axes, separated=separated, colors_differ=colors_differ,
                projection_witness=projection_witness,
                gate=gate_witness, overlap_gate=overlap_gate,
                changes=changes, stations=len(stations),
                used_changes=[changes[i] for i in route[:-1]],
                length=len(route), unused=len(stations)-len(route),
                shape=grid.shape, geometry=(stations[0]['h'], stations[0]['w'],
                                           stations[0]['copies'], stations[0]['orientation']),
                layout=tuple((s['y']-stations[0]['y'], s['x']-min(t['x'] for t in stations))
                             for s in stations),
                exceptions=tuple(s['exception'] for s in stations),
                profiles=tuple((s['rows'], s['cols']) for s in stations))


def _fallback(rng, geometry, palette, expanded):
    """A port-blocked duplicate choice, with two separated first-link lanes."""
    h, w, copies, orientation = geometry
    # When a dimension is 4, use its first/last rows for the same witness.
    if orientation == 'h':
        cells = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
        profiles = [(1 | (1 << (h-1)), 1), (1 | (1 << (h-1)), 2),
                    (2, 3), (3, 2), (3, 1), (3, 2)]
    else:
        cells = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]
        profiles = [(1, 1 | (1 << (w-1))), (2, 1 | (1 << (w-1))),
                    (3, 2), (2, 3), (1, 3), (2, 3)]
    profiles = [(tuple(bool(r & (1 << j)) for j in range(h)),
                 tuple(bool(c & (1 << j)) for j in range(w))) for r, c in profiles]
    a, b, c, d, e = map(int, palette[4:9])
    return _draw(rng, cells, [0, 1, 3, 5], [a, b, b, c, e, d], profiles,
                 geometry, palette, expanded, evidence=True)


def _sample_pair(rng, geometry, palette, gate=False, demonstration=False, test=False):
    h, w, copies, orientation = geometry
    H, W = _station_shape(*geometry)
    options = [(nr, nc) for nr in range(1, 5) for nc in range(1, 5)
               if nr*H+nr-1 <= 26 and nc*W+nc-1 <= 28
               and 3 <= nr*nc <= 8
               and (not demonstration or (nr >= 2 and nc >= 2))]
    _require(bool(options), "infeasible generator geometry")
    for attempt in range(90):
        nr, nc = options[int(rng.integers(len(options)))]
        universe = [(r, c) for r in range(nr) for c in range(nc)]
        minimum = 5 if gate else 4 if demonstration else 3
        n = int(rng.integers(min(minimum, len(universe)), len(universe)+1))
        cells = sorted(universe[int(i)] for i in rng.choice(len(universe), n, replace=False))
        max_length = min(n, 7 if test else 5)
        length = int(rng.integers(4 if demonstration else 3, max_length+1))
        path = _sample_path(rng, cells, _visible_graph(cells), length, demonstration)
        if path is None:
            continue
        profiles = _profiles(rng, cells, path, h, w, demonstration)
        labels = _labels(rng, cells, path, profiles, palette[4:], gate)
        if labels is None:
            continue
        pair = _draw(rng, cells, path, labels, profiles, geometry, palette, test,
                     evidence=demonstration)
        try:
            info = _decode(pair['input'])
            features = _features(pair['input'], info)
        except ValueError:
            continue
        if not np.array_equal(pair['output'], _paint(pair['input'], info['stations'], info['route'])):
            continue
        if gate and (not features['gate'] or not features['overlap_gate']):
            continue
        if demonstration and (features['axes'] != {'h', 'v'} or not features['separated']
                              or not features['projection_witness']):
            continue
        if demonstration and not any(a == r > 0 for a, r in features['used_changes']):
            continue
        return pair
    if demonstration:
        # Actual demonstrations use 3-by-2 or 2-by-3 glyphs, so this six-station
        # witness always fits and its third route source has a balanced edit.
        return _fallback(rng, geometry, palette, test)
    # A guaranteed straight route with distinct labels for any sampled geometry.
    nr, nc = next((r, c) for r, c in options if max(r, c) >= 3)
    cells = [(i, 0) for i in range(3)] if nr >= 3 else [(0, i) for i in range(3)]
    profiles = _profiles(rng, cells, [0, 1, 2], h, w, False)
    return _draw(rng, cells, [0, 1, 2], list(map(int, palette[4:7])), profiles,
                 geometry, palette, test)


def _check_count(value, name, low, high):
    _require(not isinstance(value, (bool, np.bool_))
             and isinstance(value, (int, np.integer)), f"{name} must be an integer")
    _require(low <= int(value) <= high, f"{name} must be {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _check_count(num_train, 'num_train', 2, 10)
    nq = _check_count(num_test, 'num_test', 1, 4)
    _require(nt > nq, "num_train must be greater than num_test")
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and int(seed) >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    shared_palette = bool(rng.random() < .25)
    palette = list(map(int, rng.permutation(10)))
    first_orientation = str(rng.choice(('h', 'v')))
    shift = str(rng.choice(('glyph_extent', 'copy_count')))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            if not shared_palette:
                palette = list(map(int, rng.permutation(10)))
            demo = split == 'train' and index < 2
            if demo:
                orient = first_orientation if index == 0 else ('v' if first_orientation == 'h' else 'h')
                geometry = (3, 2, 3, 'h') if orient == 'h' else (2, 3, 3, 'v')
            elif split == 'test' and shift == 'glyph_extent':
                geometry = (4, 2, 3, 'h') if rng.random() < .5 else (2, 4, 3, 'v')
            elif split == 'test':
                orient = str(rng.choice(('h', 'v')))
                geometry = (int(rng.integers(2, 4)), 2, 5, 'h') if orient == 'h' else (
                    2, int(rng.integers(2, 4)), 5, 'v')
            else:
                # Three-copy training supports familiar and smaller scenes,
                # with 2..3-cell glyph extents in independently chosen axes.
                geometry = ((int(rng.integers(2, 4)), 2, 3, 'h') if rng.random() < .5
                            else (2, int(rng.integers(2, 4)), 3, 'v'))
            pair = _sample_pair(rng, geometry, palette, gate=demo and index == 0,
                                demonstration=demo, test=split == 'test')
            # Affine palette permutations preserve all equalities and roles.
            # Forty permutations give distinct inputs because every grid uses
            # at least six colors; at most thirteen prior inputs are excluded.
            if _signature(pair['input']) in seen:
                original = pair
                for mul in (1, 3, 7, 9):
                    for offset in range(10):
                        remap = (mul*np.arange(10)+offset) % 10
                        candidate = {k: remap[v].astype(np.int8) for k, v in original.items()}
                        if _signature(candidate['input']) not in seen:
                            pair = candidate
                            break
                    else:
                        continue
                    break
                _require(_signature(pair['input']) not in seen, "duplicate palette exhaustion")
            seen.add(_signature(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "splits must be lists")
    nt = _check_count(len(puzzle['train']), 'num_train', 2, 10)
    nq = _check_count(len(puzzle['test']), 'num_test', 1, 4)
    _require(nt > nq, "training must outnumber test")
    measurements, seen = {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            grid, output = _grid(pair['input']), _grid(pair['output'])
            _require(grid.shape == output.shape, "output must preserve dimensions")
            signature = _signature(grid)
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            info = _decode(grid)
            expected = _paint(grid, info['stations'], info['route'])
            _require(np.array_equal(output, expected), "wrong defect-port connection")
            changed = grid != output
            _require(bool(changed.any()) and np.all(grid[changed] == info['bg']),
                     "only background gap cells may change")
            for station in info['stations']:
                y, x, H, W = (station[k] for k in ('y', 'x', 'H', 'W'))
                _require(np.array_equal(grid[y:y+H, x:x+W], output[y:y+H, x:x+W]),
                         "station contents must be preserved")
            _require(np.array_equal(grid[info['key_y']], output[info['key_y']]), "key must be preserved")
            measurements[split].append(_features(grid, info))
    training = measurements['train']
    _require(any(m['gate'] for m in training), "demonstrate a route decision caused by ports")
    _require(any(m['overlap_gate'] for m in training),
             "demonstrate exact port matching rather than nonempty overlap")
    _require(any(m['separated'] for m in training), "demonstrate separated lanes")
    _require(any(m['colors_differ'] for m in training), "demonstrate source coloring")
    _require(any(m['projection_witness'] for m in training),
             "demonstrate compatible projections of different discrepancy masks")
    changes = [change for m in training for change in m['used_changes']]
    _require(any(a > 0 and r == 0 for a, r in changes), "demonstrate additions")
    _require(any(r > 0 and a == 0 for a, r in changes), "demonstrate removals")
    _require(any(a == r > 0 for a, r in changes), "demonstrate a count-preserving exception")
    train_axes = set().union(*(m['axes'] for m in training))
    test_axes = set().union(*(m['axes'] for m in measurements['test']))
    _require(test_axes <= train_axes, "test axis lacks training evidence")
    _require(any(any(i != 1 for i in m['exceptions']) for m in training),
             "demonstrate an exception away from the middle copy")
    return True


def self_test():
    """Bounded semantic checks and actual structural variation across 42 seeds."""
    puzzles, records = [], []
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                records.append((split, _features(pair['input'], info)))
                before = pair['input'].copy()
                assert np.array_equal(solve(before), pair['output'])
                assert np.array_equal(before, pair['input']), 'solve mutated input'
    assert len({_puzzle_signature(p) for p in puzzles}) == 42
    assert _puzzle_signature(puzzles[0]) == _puzzle_signature(generate_puzzle(HOST_SEEDS[0]))
    for split in ('train', 'test'):
        values = [m for s, m in records if s == split]
        assert len({m['shape'] for m in values}) >= 12
        assert len({m['layout'] for m in values}) >= 20
        assert len({m['profiles'] for m in values}) >= 30
        assert any(m['unused'] for m in values)
        assert any(m['unused'] == 0 for m in values)
        assert set().union(*(m['axes'] for m in values)) == {'h', 'v'}
    assert {m['geometry'][2] for _, m in records} == {3, 5}
    assert {m['geometry'][3] for _, m in records} == {'h', 'v'}
    assert max(max(m['geometry'][:2]) for _, m in records) == 4
    assert any(m['length'] == 3 for _, m in records)
    assert any(m['length'] >= 5 for _, m in records)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(7100+10*nt+nq, nt, nq)
            assert (len(p['train']), len(p['test'])) == (nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    # Whole-palette equivariance tests geometry-based role recovery.
    base = puzzles[0]
    remap = np.array([7, 0, 6, 1, 9, 2, 8, 3, 5, 4], dtype=np.int8)
    for pair in base['train'] + base['test']:
        assert np.array_equal(solve(remap[pair['input']]), remap[pair['output']])
    # Moving the exceptional glyph between compartments must not move ports.
    pair = base['train'][0]
    moved = pair['input'].copy()
    info = _decode(moved)
    for s in info['stations']:
        origin0 = s['origins'][s['exception']]
        origin1 = s['origins'][(s['exception']+1) % s['copies']]
        for origin, bitmap in ((origin0, s['master']), (origin1, s['variant'])):
            r, c = origin
            moved[s['y']+r:s['y']+r+s['h'], s['x']+c:s['x']+c+s['w']] = bitmap
    expected = moved.copy()
    changed = pair['input'] != pair['output']
    expected[changed] = pair['output'][changed]
    assert np.array_equal(solve(moved), expected), 'ports depend on exceptional-copy position'
    # Complementing every glyph's ink/carrier cells swaps additions and
    # removals while preserving the exact symmetric difference and gap answer.
    complemented = pair['input'].copy()
    for s in info['stations']:
        for r, c in s['origins']:
            view = complemented[s['y']+r:s['y']+r+s['h'], s['x']+c:s['x']+c+s['w']]
            view[:] = np.where(view == s['carrier'], s['ink'], s['carrier'])
    expected = complemented.copy()
    expected[changed] = pair['output'][changed]
    assert np.array_equal(solve(complemented), expected), 'ports depend on flip direction'
    # The decoder covers the full stated glyph space, including combinations
    # outside the sampler's train/test policies and an eight-entry route.
    check_rng = np.random.default_rng(8801)
    for h in range(2, 5):
        for w in range(2, 5):
            for copies in (3, 5):
                for orientation in ('h', 'v'):
                    geometry = (h, w, copies, orientation)
                    H, W = _station_shape(*geometry)
                    if 3*H+2 <= 26 and W <= 28:
                        cells = [(i, 0) for i in range(3)]
                    elif H <= 26 and 3*W+2 <= 28:
                        cells = [(0, i) for i in range(3)]
                    else:
                        continue  # No three-station member fits the grid bound.
                    path = [0, 1, 2]
                    profiles = _profiles(check_rng, cells, path, h, w, False)
                    p = _draw(check_rng, cells, path, [4, 5, 6], profiles,
                              geometry, list(range(10)), False)
                    assert np.array_equal(solve(p['input']), p['output'])
    cells = [(r, c) for r in range(2) for c in range(4)]
    path = [0, 1, 2, 3, 7, 6, 5, 4]
    key = [4, 5, 6, 7, 8, 9, 4, 5]
    labels = [0]*8
    for i, color in zip(path, key):
        labels[i] = color
    profiles = _profiles(check_rng, cells, path, 2, 2, False)
    p = _draw(check_rng, cells, path, labels, profiles, (2, 2, 3, 'v'),
              list(range(10)), False)
    assert np.array_equal(solve(p['input']), p['output'])
    # Corruptions check output preservation, anomaly evidence and ordered key.
    for defect in ('output', 'exception', 'key'):
        bad = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in base[s]] for s in base}
        p = bad['train'][0]
        a = _decode(p['input'])
        if defect == 'output':
            p['output'][0, 0] = (int(p['output'][0, 0])+1) % 10
        elif defect == 'exception':
            s = a['stations'][0]
            r, c = s['origins'][s['exception']]
            p['input'][s['y']+r:s['y']+r+s['h'], s['x']+c:s['x']+c+s['w']] = s['master']
        else:
            xs = np.flatnonzero(p['input'][a['key_y']] != a['bg'])
            p['input'][a['key_y'], xs] = p['input'][a['key_y'], xs][::-1]
            # Reverse order changes source colors, even if the same undirected
            # route survives; the original output must no longer be accepted.
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            continue
        raise AssertionError('corruption accepted: '+defect)
    return {'seeds_checked': 42, 'pairs_checked': len(records),
            'input_shapes': len({m['shape'] for _, m in records}),
            'layouts': len({m['layout'] for _, m in records}),
            'port_configurations': len({m['profiles'] for _, m in records})}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
