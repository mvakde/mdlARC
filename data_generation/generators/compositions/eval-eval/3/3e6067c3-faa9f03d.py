"""Addressed relays: a color itinerary controls a damaged cable network.

No parent generator is imported. All roles and topology are read from the input.
The generator does no I/O; self_test retains only bounded aggregate measurements.
"""
from collections import deque
from hashlib import sha256
import json

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _add(p, d, k=1):
    return p[0] + k * d[0], p[1] + k * d[1]


def _line(a, b):
    assert a[0] == b[0] or a[1] == b[1]
    d = (int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1])))
    return [_add(a, d, k) for k in range(max(abs(a[0]-b[0]), abs(a[1]-b[1]))+1)]


def _poly(points):
    return set().union(*(set(_line(a, b)) for a, b in zip(points, points[1:])))


def _components(cells):
    remaining, result = set(cells), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, part = [start], {start}
        while stack:
            p = stack.pop()
            for d in DIRS:
                q = _add(p, d)
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def _path(graph, a, b):
    queue, previous = deque([a]), {a: None}
    while queue:
        p = queue.popleft()
        if p == b:
            result = [b]
            while previous[result[-1]] is not None:
                result.append(previous[result[-1]])
            return result[::-1]
        for q in sorted(graph[p]):
            if q not in previous:
                previous[q] = p
                queue.append(q)
    return []


def _routes(graph, labels, key):
    """Enumerate up to two distinct-station assignments, without arrow hints."""
    answers = []

    def visit(route):
        if len(answers) == 2:
            return
        if len(route) == len(key):
            answers.append(tuple(route))
            return
        choices = range(len(labels)) if not route else sorted(graph[route[-1]])
        for i in choices:
            if i not in route and labels[i] == key[len(route)]:
                visit(route + [i])

    visit([])
    return answers


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _ray(p, d, bottom, width):
    result, q = [], _add(p, d, 2)
    while 1 <= q[0] < bottom and 1 <= q[1] < width-1:
        result.append(q)
        q = _add(q, d)
    return set(result)


def _straight(p, cells):
    neighbors = [d for d in DIRS if _add(p, d) in cells]
    return len(neighbors) == 2 and _add(neighbors[0], neighbors[1]) == (0, 0)


def _complete(grid, background, wire, repair, tile_cells, bottom):
    cells = {(int(r), int(c)) for r, c in np.argwhere(grid == wire)}
    marked = set() if repair is None else {
        (int(r), int(c)) for r, c in np.argwhere(grid == repair)}
    cells |= marked
    holes = set()
    for r in range(1, bottom):
        for c in range(1, grid.shape[1]-1):
            p = (r, c)
            if p in tile_cells or int(grid[p]) != background:
                continue
            h = (r, c-1) in cells and (r, c+1) in cells
            v = (r-1, c) in cells and (r+1, c) in cells
            assert not (h and v), "an unmarked omission has ambiguous axes"
            if h or v:
                holes.add(p)
    return cells | holes, marked, holes


def _decode(input_grid):
    a = _grid(input_grid)
    h, w = a.shape
    assert h >= 12 and w >= 11
    bottom = h-4
    bg, frame = int(a[-1, 0]), int(a[0, 0])
    assert bg != frame
    assert np.all(a[0] == frame) and np.all(a[bottom] == frame)
    assert np.all(a[:bottom+1, 0] == frame) and np.all(a[:bottom+1, -1] == frame)
    assert np.all(a[-1] == bg) and np.all(a[bottom+1] == bg)
    token_x = np.flatnonzero(a[-2] != bg).tolist()
    assert 3 <= len(token_x) <= 10 and all(y-x == 2 for x, y in zip(token_x, token_x[1:]))
    key = [int(a[-2, x]) for x in token_x]
    stations, tiles, arrows, arrow_colors = [], set(), {}, set()
    for r in range(2, bottom-1):
        for c in range(2, w-2):
            if int(a[r, c]) in (bg, frame):
                continue
            corners = [(r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1)]
            if not all(int(a[p]) == frame for p in corners):
                continue
            altered = [d for d in DIRS if int(a[_add((r, c), d)]) != frame]
            if len(altered) > 1:
                continue
            block = {(y, x) for y in range(r-1, r+2) for x in range(c-1, c+2)}
            assert not block & tiles
            index = len(stations)
            stations.append((r, c, int(a[r, c])))
            tiles |= block
            if altered:
                d = altered[0]
                arrows[index] = d
                arrow_colors.add(int(a[_add((r, c), d)]))
    assert 3 <= len(stations) <= 10 and arrows and len(arrow_colors) == 1
    direction = next(iter(arrow_colors))
    assert direction not in (bg, frame)
    outside = [(r, c) for r in range(1, bottom) for c in range(1, w-1)
               if (r, c) not in tiles and int(a[r, c]) != bg]
    counts = {}
    for p in outside:
        color = int(a[p])
        counts[color] = counts.get(color, 0) + 1
    assert 1 <= len(counts) <= 2
    ordered = sorted(counts, key=lambda color: (-counts[color], color))
    wire = ordered[0]
    repair = ordered[1] if len(ordered) == 2 else None
    assert len(ordered) == 1 or counts[wire] > counts[repair]
    roles = {bg, frame, direction, wire} | (set() if repair is None else {repair})
    assert len(roles) == 4 + (repair is not None)
    assert all(s[2] not in roles for s in stations) and all(k not in roles for k in key)
    # Rebuild all non-cable content, ruling out stray pixels and malformed tiles.
    rebuilt = np.full(a.shape, bg, dtype=np.int8)
    rebuilt[0] = rebuilt[bottom] = frame
    rebuilt[:bottom+1, 0] = rebuilt[:bottom+1, -1] = frame
    rebuilt[-2, token_x] = key
    for i, (r, c, label) in enumerate(stations):
        rebuilt[r-1:r+2, c-1:c+2] = frame
        rebuilt[r, c] = label
        if i in arrows:
            rebuilt[_add((r, c), arrows[i])] = direction
    for p in outside:
        rebuilt[p] = a[p]
    assert np.array_equal(a, rebuilt)
    cells, marked, holes = _complete(a, bg, wire, repair, tiles, bottom)
    assert not cells & tiles
    components = _components(cells)
    graph = {i: set() for i in range(len(stations))}
    edges, ports, terminal = {}, {}, {}
    for part in components:
        assert all(sum(_add(p, d) in part for d in DIRS) <= 2 for p in part)
        attached = [(i, d) for i, s in enumerate(stations) for d in DIRS
                    if _add(s[:2], d, 2) in part]
        assert len(attached) in (1, 2)
        attachment_cells = [_add(stations[i][:2], d, 2) for i, d in attached]
        for p in part:
            degree = sum(_add(p, d) in part for d in DIRS) + attachment_cells.count(p)
            boundary = p[0] in (1, bottom-1) or p[1] in (1, w-2)
            assert degree == (1 if boundary else 2), 'cables may branch only inside stations'
        for i, d in attached:
            assert (i, d) not in ports
            ports[i, d] = part
        if len(attached) == 2:
            i, j = attached[0][0], attached[1][0]
            assert i != j and j not in graph[i]
            graph[i].add(j)
            graph[j].add(i)
            edges[frozenset((i, j))] = part
            assert not any(r in (1, bottom-1) or c in (1, w-2) for r, c in part)
        else:
            i, d = attached[0]
            boundary = [p for p in part if p[0] in (1, bottom-1) or p[1] in (1, w-2)]
            assert len(boundary) == 1
            assert sum(_add(boundary[0], e) in part for e in DIRS) <= 1
            terminal[i, d] = part
    assert all(graph.values()) and len(edges) == len(stations)-1
    assert len(_path(graph, 0, len(stations)-1)) > 0
    assert all(_path(graph, 0, i) for i in graph)
    assert len(terminal) == len(arrows)
    for i, d in arrows.items():
        opposite = (-d[0], -d[1])
        assert (i, opposite) in terminal and (i, d) not in ports
    routes = _routes(graph, [s[2] for s in stations], key)
    assert len(routes) == 1, "the key needs one globally unique itinerary"
    route = routes[0]
    last = route[-1]
    assert last in arrows
    d = arrows[last]
    ray = _ray(stations[last][:2], d, bottom, w)
    assert ray and not ray & tiles
    # Crossings are transverse. New emission has explicit top priority.
    for p in ray & cells:
        assert _straight(p, cells) and not any(_add(p, e) in cells for e in (d, (-d[0], -d[1])))
    old = terminal[last, (-d[0], -d[1])]
    selected = set().union(*(edges[frozenset((i, j))] for i, j in zip(route, route[1:])))
    return dict(grid=a, background=bg, frame=frame, wire=wire, repair=repair,
                direction=direction, bottom=bottom, stations=stations, tiles=tiles,
                arrows=arrows, cells=cells, marked=marked, holes=holes, graph=graph,
                edges=edges, terminal=terminal, route=route, selected=selected, old=old, ray=ray)


def _render(info, *, ignore_key=False, ignore_repair=False):
    out = info['grid'].copy()
    bg, wire = info['background'], info['wire']
    for p in info['cells']:
        out[p] = wire
    for i, d in info['arrows'].items():
        out[_add(info['stations'][i][:2], d)] = info['frame']
    if not ignore_key:
        for i, j in zip(info['route'], info['route'][1:]):
            for p in info['edges'][frozenset((i, j))]:
                out[p] = info['stations'][i][2]
        for p in info['old']:
            out[p] = bg
        for p in info['ray']:
            out[p] = info['stations'][info['route'][-1]][2]
    if ignore_repair:
        # Deliberately wrong baseline: keep the observed damage at retained sites.
        for p in (info['marked'] | info['holes']) - info['old'] - info['ray']:
            out[p] = info['grid'][p]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the public grid, reconstruct cables, then execute its itinerary."""
    return _render(_decode(input_grid))


def _coordinates(rng, n, limit):
    start = int(rng.integers(3, 5))
    gaps = [4] * (n-1)
    spare = limit - start - 4*(n-1)
    for i in rng.permutation(n-1):
        extra = int(rng.integers(0, min(2, spare)+1))
        gaps[int(i)] += extra
        spare -= extra
    return [start] + [start + sum(gaps[:i]) for i in range(1, n)]


def _coarse_tree(rng, nr, nc, needed, fallback, guaranteed=False):
    if fallback:
        points = [(0,0),(0,1),(1,1),(2,1),(2,2),(2,3),(1,3),
                  (1,4),(2,4),(3,4),(4,4),(4,3),(4,2),(3,2)]
        chosen = [points[i] for i in (0,2,3,5,7,10,11,13)]
        k, reflect = (0,False) if guaranteed else (int(rng.integers(4)), bool(rng.integers(2)))

        def transform(p):
            r, c = p
            if reflect:
                c = 4-c
            for _ in range(k):
                r, c = 4-c, r
            return r, c

        points, chosen = list(map(transform, points)), list(map(transform, chosen))
        graph = {p: set() for p in points}
        for a, b in zip(points, points[1:]):
            graph[a].add(b)
            graph[b].add(a)
        return graph, chosen, (k, reflect)
    target = int(rng.integers(max(needed+2, 7), min(nr*nc-2, 18)+1))
    start = (int(rng.integers(nr)), int(rng.integers(nc)))
    budget = [1200]

    def walk(path):
        budget[0] -= 1
        if len(path) == target:
            return path
        if budget[0] <= 0:
            return None
        for index in rng.permutation(4):
            q = _add(path[-1], DIRS[int(index)])
            if 0 <= q[0] < nr and 0 <= q[1] < nc and q not in path:
                answer = walk(path+[q])
                if answer is not None:
                    return answer
        return None

    points = walk([start])
    if points is None:
        return None
    graph = {p: set() for p in points}
    for a, b in zip(points, points[1:]):
        graph[a].add(b)
        graph[b].add(a)
    for _ in range(int(rng.integers(3))):
        options = [(p, _add(p, d)) for p in sorted(graph) for d in DIRS
                   if len(graph[p]) < 3 and _add(p, d) not in graph
                   and 0 <= p[0]+d[0] < nr and 0 <= p[1]+d[1] < nc]
        if options:
            a, b = options[int(rng.integers(len(options)))]
            graph[a].add(b)
            graph[b] = {a}
    mandatory = sorted(p for p in graph if len(graph[p]) != 2)
    lower, upper = max(len(mandatory), needed+1), min(10, len(graph)-1)
    if lower > upper:
        return None
    count = int(rng.integers(lower, upper+1))
    candidates = sorted(set(graph)-set(mandatory))
    rng.shuffle(candidates)
    return graph, sorted(mandatory+candidates[:count-len(mandatory)]), None


def _tail(rng, start, d, blocked, nr, nc):
    """Find a boundary lead in empty coarse cells, with its first direction fixed."""
    first = _add(start, d)
    if not (0 <= first[0] < nr and 0 <= first[1] < nc):
        return [start], d
    if first in blocked:
        return None
    possibilities = []
    # A randomized depth-first walk yields both short and bent leads.
    budget = [160]

    def visit(path):
        budget[0] -= 1
        if budget[0] < 0 or len(path) > nr+nc+2:
            return
        p = path[-1]
        for e in DIRS:
            q = _add(p, e)
            if not (0 <= q[0] < nr and 0 <= q[1] < nc):
                possibilities.append((path.copy(), e))
        if len(possibilities) >= 12:
            return
        for index in rng.permutation(4):
            q = _add(p, DIRS[int(index)])
            if (0 <= q[0] < nr and 0 <= q[1] < nc and q not in blocked and q not in path):
                visit(path+[q])
                if len(possibilities) >= 12:
                    break

    visit([start, first])
    if not possibilities:
        return None
    return possibilities[int(rng.integers(len(possibilities)))]


def _make_pair(rng, length, repeat, witness, palette, fallback=False, guaranteed=False):
    nr = nc = 5 if fallback else int(rng.integers(3 if length <= 4 else 4, 6))
    if not fallback:
        nc = int(rng.integers(3 if length <= 4 else 4, 6))
    coarse = _coarse_tree(rng, nr, nc, length, fallback, guaranteed)
    if coarse is None:
        return None
    graph, station_nodes, _ = coarse
    ys, xs = ([3,7,11,15,19], [3,7,11,15,19]) if guaranteed else (
        _coordinates(rng, nr, 23), _coordinates(rng, nc, 26))
    bottom = 22 if guaranteed else int(rng.integers(ys[-1]+3, min(26, ys[-1]+6)+1))
    width = 23 if guaranteed else int(rng.integers(xs[-1]+4, min(30, xs[-1]+7)+1))
    height = bottom+4
    positions = {p: (ys[p[0]], xs[p[1]]) for p in graph}
    stations = [positions[p] for p in station_nodes]
    tiles = {(r+dr, c+dc) for r, c in stations for dr in (-1,0,1) for dc in (-1,0,1)}
    wire_cells = set().union(*(set(_line(positions[a], positions[b])) for a in graph for b in graph[a]))-tiles
    station_graph = {i: set() for i in range(len(stations))}
    edge_cells = {}
    for i, a in enumerate(station_nodes):
        for j, b in enumerate(station_nodes[:i]):
            path = _path(graph, a, b)
            if not set(path[1:-1]) & set(station_nodes):
                station_graph[i].add(j)
                station_graph[j].add(i)
                edge_cells[frozenset((i,j))] = _poly([positions[p] for p in path])-tiles
    options = []
    for last, node in enumerate(station_nodes):
        for d in DIRS:
            if _add(node, d) in graph[node] or _add(node, (-d[0],-d[1])) in graph[node]:
                continue
            ray = _ray(stations[last], d, bottom, width)
            if ray & tiles or not ray:
                continue
            if any(not _straight(p, wire_cells) or _add(p,d) in wire_cells
                   or _add(p,(-d[0],-d[1])) in wire_cells for p in ray & wire_cells):
                continue
            for first in station_graph:
                route = _path(station_graph, first, last)
                if len(route) != length:
                    continue
                selected = set().union(*(edge_cells[frozenset((a,b))] for a,b in zip(route,route[1:])))
                if witness and not ray & selected:
                    continue
                if witness and not any(any(not _straight(p,part) and
                        sum(_add(p,e) in part for e in DIRS) == 2 for p in part)
                        for a,b in zip(route,route[1:])
                        for part in (edge_cells[frozenset((a,b))],)):
                    continue
                options.append((route,d,ray,selected))
    rng.shuffle(options)
    if guaranteed:
        options = [o for o in options if o[0] == list(range(8-length,8)) and o[1] == (0,1)]
    for route, d, ray, selected in options[:30]:
        final = route[-1]
        tail = ([(3,2),(3,1),(4,1),(4,0)],(0,-1)) if guaranteed else _tail(
            rng, station_nodes[final], (-d[0],-d[1]), set(graph), nr, nc)
        if tail is None:
            continue
        tail_nodes, exit_direction = tail

        def tail_pixels(nodes, exit_d):
            points = [(ys[p[0]], xs[p[1]]) for p in nodes]
            end = points[-1]
            boundary = (1 if exit_d[0] < 0 else bottom-1, end[1]) if exit_d[0] else (
                end[0], 1 if exit_d[1] < 0 else width-2)
            return _poly(points+[boundary])-tiles

        old = tail_pixels(tail_nodes, exit_direction)
        if witness and not any(not _straight(p,old) and
                sum(_add(p,e) in old for e in DIRS) == 2 for p in old):
            continue
        # Every scene has an inactive switch: selecting all arrows is wrong.
        extras = [(0,(1,0),tail_pixels([(0,0)],(-1,0)))] if guaranteed else []
        for i, node in ([] if guaranteed else enumerate(station_nodes)):
            if i == final:
                continue
            for e in DIRS:
                if _add(node,e) in graph[node] or _add(node,(-e[0],-e[1])) in graph[node]:
                    continue
                if _ray(stations[i], e, bottom, width) & tiles:
                    continue
                extra = _tail(rng, node, (-e[0],-e[1]), set(graph)|set(tail_nodes), nr, nc)
                if extra is not None:
                    nodes, ed = extra
                    pixels = tail_pixels(nodes, ed)
                    if not pixels & (wire_cells|old):
                        extras.append((i,e,pixels))
        if not extras:
            continue
        inactive, extra_direction, extra_cells = extras[int(rng.integers(len(extras)))]
        arrows = {final:d, inactive:extra_direction}
        if not guaranteed and rng.random() < .35:
            thirds = [(i,e,pixels) for i,e,pixels in extras if i != inactive
                      and not pixels & extra_cells]
            if thirds:
                i,e,pixels = thirds[int(rng.integers(len(thirds)))]
                extra_cells |= pixels
                arrows[i] = e
        full_wire = wire_cells | old | extra_cells
        if any(not _straight(p, full_wire) or _add(p,d) in full_wire
               or _add(p,(-d[0],-d[1])) in full_wire for p in ray & full_wire):
            continue
        colors = [int(c) for c in palette[5:]]
        labels = None
        for _ in range(80):
            proposed = [int(rng.choice(colors)) for _ in stations]
            if guaranteed:
                pattern = {3:(0,1,2),4:(0,1,0,2),5:(0,1,2,0,3),
                           6:(0,1,2,0,3,1),7:(0,1,2,0,3,1,2)}[length]
                proposed = [colors[4]] * len(stations)
                for i,color_index in zip(route,pattern):
                    proposed[i] = colors[color_index]
            elif repeat and length == 4:
                x,y,z = [int(c) for c in rng.choice(colors, 3, replace=False)]
                for i, color in zip(route, (x,y,x,z)):
                    proposed[i] = color
            elif not repeat and length == 3:
                for i, color in zip(route, rng.choice(colors, 3, replace=False)):
                    proposed[i] = int(color)
            key = [proposed[i] for i in route]
            if len(set(key)) < 2 or key[0] == key[-1]:
                continue
            if witness and any(proposed[a] == proposed[final] for a,b in zip(route,route[1:])
                               if edge_cells[frozenset((a,b))] & ray):
                continue
            if _routes(station_graph, proposed, key) == [tuple(route)]:
                labels = proposed
                break
        if labels is None:
            continue
        bg, frame, wire, repair, direction = map(int,palette[:5])
        inp = np.full((height,width),bg,dtype=np.int8)
        inp[0] = inp[bottom] = frame
        inp[:bottom+1,0] = inp[:bottom+1,-1] = frame
        for p in full_wire:
            inp[p] = wire
        for i,(r,c) in enumerate(stations):
            inp[r-1:r+2,c-1:c+2] = frame
            inp[r,c] = labels[i]
        for i,e in arrows.items():
            inp[_add(stations[i],e)] = direction
        key = [labels[i] for i in route]
        key_start = int(rng.integers(1,width-(2*len(key)-1)))
        inp[-2,key_start:key_start+2*len(key):2] = key
        # Construction ground truth uses the sampled coarse tree, not solve().
        out = inp.copy()
        for i,e in arrows.items():
            out[_add(stations[i],e)] = frame
        for a,b in zip(route,route[1:]):
            for p in edge_cells[frozenset((a,b))]:
                out[p] = labels[a]
        for p in old:
            out[p] = bg
        for p in ray:
            out[p] = labels[final]
        # Require a real connectivity break on the itinerary, not only cosmetic dots.
        gap_choices = sorted(p for p in selected-ray if _straight(p,full_wire)
                             and not any(_add(p,e) in tiles for e in DIRS))
        if not gap_choices:
            continue
        gap = gap_choices[int(rng.integers(len(gap_choices)))]
        inp[gap] = bg
        # At least one old-lead mark and one retained mark; bends are preferred.
        retained = (full_wire-old-ray)-{gap}
        if not retained or not old-ray:
            continue
        def marker_priority(p):
            degree = sum(_add(p,e) in full_wire for e in DIRS)
            return (0 if degree == 2 and not _straight(p,full_wire)
                    else 1 if degree < 2 else 2, p)

        old_options = sorted(old-ray, key=marker_priority)
        kept_options = sorted(retained, key=marker_priority)
        marks = {old_options[0], kept_options[0]}
        others = sorted(full_wire-{gap}-marks)
        rng.shuffle(others)
        marks.update(others[:int(rng.integers(0,5))])
        for p in marks:
            inp[p] = repair
        try:
            info = _decode(inp)
            assert info['cells'] == full_wire
            assert len(info['route']) == length
            assert info['holes'] and info['marked'] & info['old']
            assert not np.array_equal(_render(info,ignore_repair=True),out)
            assert np.array_equal(_render(info),out)
        except AssertionError:
            continue
        return {'input':inp,'output':out}
    return None


def _integer(value, low, high, name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _integer(num_train,2,10,'num_train')
    nq = _integer(num_test,1,4,'num_test')
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    palette = rng.permutation(10)
    varying_roles = bool(rng.integers(2))
    puzzle, seen = {'train':[],'test':[]}, set()
    for split,count in (('train',nt),('test',nq)):
        for index in range(count):
            length = (3 if index == 0 else 4 if index == 1 else int(rng.integers(3,5))) if split == 'train' else int(rng.integers(5,8))
            repeat = split == 'test' or index != 0
            witness = split == 'train' and index == 1
            for attempt in range(100):
                roles = rng.permutation(10) if varying_roles else palette
                pair = _make_pair(rng,length,repeat,witness,roles,fallback=attempt >= 72)
                if pair is None:
                    continue
                signature = (pair['input'].shape,pair['input'].tobytes())
                if signature in seen:
                    continue
                seen.add(signature)
                puzzle[split].append(pair)
                break
            else:
                pair = _make_pair(rng,length,repeat,witness,palette,fallback=True,guaranteed=True)
                assert pair is not None, 'deterministic relay construction failed'
                found = False
                # All 40 affine bijections give distinct images of a scene
                # with at least eight colors; at most 13 earlier inputs exist.
                for multiplier in (1,3,7,9):
                    for offset in range(10):
                        remap = (multiplier*np.arange(10)+offset)%10
                        candidate = {k:remap[v].astype(np.int8) for k,v in pair.items()}
                        signature = (candidate['input'].shape,candidate['input'].tobytes())
                        if signature not in seen:
                            seen.add(signature)
                            puzzle[split].append(candidate)
                            found = True
                            break
                    if found:
                        break
                assert found
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def measure_pair(pair):
    info = _decode(pair['input'])
    route, stations = info['route'], info['stations']
    key = [stations[i][2] for i in route]
    bent_edges = sum(any(not _straight(p,part) and sum(_add(p,d) in part for d in DIRS) == 2
                         for p in part) for part in info['edges'].values())
    return {'shape':list(info['grid'].shape),'stations':len(stations),'length':len(route),
            'unused_stations':len(stations)-len(route),'bent_edges':bent_edges,
            'junction_stations':sum(len(v)>2 for v in info['graph'].values()),
            'old_bends':sum(not _straight(p,info['old']) and sum(_add(p,d) in info['old'] for d in DIRS)==2 for p in info['old']),
            'old_length':len(info['old']),'ray_length':len(info['ray']),
            'switches':len(info['arrows']),
            'selected_crossings':len(info['ray'] & info['selected']),
            'retained_crossings':len(info['ray'] & (info['cells']-info['old'])),
            'marked_bends':sum(not _straight(p,info['cells']) and sum(_add(p,d) in info['cells'] for d in DIRS)==2 for p in info['marked']),
            'markers':len(info['marked']),'gaps':len(info['holes']),
            'repeated_key':len(set(key))<len(key),'direction':list(info['arrows'][route[-1]]),
            'roles':[info[k] for k in ('background','frame','wire','repair','direction')],
            'layout':[[r,c] for r,c,label in stations]}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training, testing = set(), [], []
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            inp,out = _grid(pair['input']),_grid(pair['output'])
            assert inp.shape == out.shape
            signature = (inp.shape,inp.tobytes())
            assert signature not in seen
            seen.add(signature)
            info = _decode(inp)
            expected = _render(info)
            assert np.array_equal(out,expected), 'relay output mismatch, including preserved cells'
            assert not np.array_equal(inp,out)
            assert len(info['arrows']) >= 2
            assert info['holes'] & info['selected']
            assert info['marked'] & info['old']
            assert not np.array_equal(_render(info,ignore_key=True),out), 'A ablation had no effect'
            assert not np.array_equal(_render(info,ignore_repair=True),out), 'B repair ablation had no effect'
            assert info['old']-info['ray'], 'old-lead deletion must be visible'
            assert all(int(out[p]) == info['background'] for p in info['old']-info['ray'])
            assert len(info['terminal']) == len(info['arrows'])
            for (i,d),part in info['terminal'].items():
                if i != info['route'][-1]:
                    assert all(int(out[p]) == info['wire'] for p in part-info['ray'])
            assert not np.any(out == info['repair']) and not np.any(out == info['direction'])
            (training if split == 'train' else testing).append(measure_pair(pair))
    # These are demonstration/shift guarantees, not hidden solver parameters.
    assert any(m['length'] == 3 for m in training)
    assert any(m['length'] == 4 and m['repeated_key'] and m['selected_crossings']
               and m['bent_edges'] and m['old_bends'] for m in training)
    assert max(m['length'] for m in training) < min(m['length'] for m in testing)


def _signature(puzzle):
    return tuple((pair['input'].shape,pair['input'].tobytes(),pair['output'].tobytes())
                 for split in ('train','test') for pair in puzzle[split])


def self_test():
    """42 streaming seeds, invariance checks, meaningful ablations and negatives."""
    shapes, layouts, roles, directions, signatures = set(),set(),set(),set(),set()
    metrics = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(sha256(repr(_signature(puzzle)).encode()).hexdigest())
        for split in ('train','test'):
            for pair in puzzle[split]:
                m = measure_pair(pair)
                metrics.append(m)
                shapes.add(tuple(m['shape']))
                layouts.add(tuple(map(tuple,m['layout'])))
                roles.add(tuple(m['roles']))
                directions.add(tuple(m['direction']))
                # The key's left-to-right order is intentional; reflect scene
                # horizontally AND reverse the tape spatially back afterward.
                reflected = np.fliplr(pair['input']).copy()
                reflected[-2] = pair['input'][-2]
                target = np.fliplr(pair['output']).copy()
                target[-2] = pair['output'][-2]
                assert np.array_equal(solve(reflected),target)
                perm = np.roll(np.arange(10),seed%10)
                assert np.array_equal(solve(perm[pair['input']]),perm[pair['output']])
        if seed == 0:
            assert _signature(puzzle) == _signature(generate_puzzle(0))
            for site_kind in ('preserved','old','ray','repair'):
                q = puzzle['test'][0]
                info = _decode(q['input'])
                site = {'preserved':(0,0),'old':min(info['old']-info['ray']),
                        'ray':min(info['ray']),'repair':min((info['marked']|info['holes'])-info['old']-info['ray'])}[site_kind]
                bad = {'train':puzzle['train'],'test':[dict(q)]}
                bad['test'][0]['output'] = q['output'].copy()
                bad['test'][0]['output'][site] = (int(q['output'][site])+1)%10
                try:
                    validate_generated_puzzle(bad)
                except AssertionError:
                    pass
                else:
                    raise AssertionError('corrupt '+site_kind+' output accepted')
    assert len(signatures) == 42 and len(shapes) >= 25 and len(layouts) >= 100
    assert len(roles) >= 30 and directions == set(DIRS)
    assert {m['length'] for m in metrics} == {3,4,5,6,7}
    assert any(m['junction_stations'] for m in metrics)
    assert any(m['old_bends'] >= 2 for m in metrics)
    assert any(m['marked_bends'] for m in metrics)
    assert len({m['stations'] for m in metrics}) >= 4
    # Exercise the deterministic safeguard directly, including recoloring.
    fallback_rng = np.random.default_rng(814)
    for length in range(3,8):
        q = _make_pair(fallback_rng,length,length!=3,length==4,np.arange(10),
                       fallback=True,guaranteed=True)
        assert q is not None and np.array_equal(solve(q['input']),q['output'])
    for nt in range(2,11):
        for nq in range(1,min(4,nt-1)+1):
            p = generate_puzzle(1900+nt*10+nq,nt,nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt,nq in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1),(4,None)):
        try:
            generate_puzzle(0,nt,nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    return {'seeds':42,'pairs':len(metrics),'shapes':len(shapes),'layouts':len(layouts),
            'role_assignments':len(roles),'key_lengths':sorted({m['length'] for m in metrics}),
            'bent_edge_pairs':sum(m['bent_edges']>0 for m in metrics),
            'bent_old_lead_pairs':sum(m['old_bends']>0 for m in metrics),
            'branching_pairs':sum(m['junction_stations']>0 for m in metrics),
            'selected_crossing_pairs':sum(m['selected_crossings']>0 for m in metrics),
            'marked_bend_pairs':sum(m['marked_bends']>0 for m in metrics)}


if __name__ == '__main__':
    print(json.dumps(self_test(),separators=(',',':')))
