"""Signal-addressed shuttles: relation composition supplies unary counter states.

All reasoning in solve is input-only. Construction and the grid parser are separate;
no parent generator, file, network, or saved example is used at runtime.
"""
from __future__ import annotations

import heapq
import numbers

import numpy as np


_D4 = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _need(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "invalid grid dimensions")
    _need(np.issubdtype(g.dtype, np.integer), "grid must contain integers")
    _need(bool(np.all((0 <= g) & (g <= 9))), "invalid grid color")
    return g


def _components(cells):
    todo = set(cells)
    parts = []
    while todo:
        p = min(todo)
        todo.remove(p)
        stack, part = [p], {p}
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


def _coords(t, b, l, r, direction):
    # The parent A scan, rotated (never reflected) with the direction of travel.
    if direction == 0:  # up
        return [(y, x) for y in range(t + 1, b) for x in range(l + 1, r)]
    if direction == 1:  # right
        return [(y, x) for x in range(r - 1, l, -1) for y in range(t + 1, b)]
    if direction == 2:  # down
        return [(y, x) for y in range(b - 1, t, -1) for x in range(r - 1, l, -1)]
    return [(y, x) for x in range(l + 1, r) for y in range(b - 1, t, -1)]


def _bar(t, b, l, r, direction, distance=0):
    if direction == 0:
        return [(t - 1 - distance, x) for x in range(l, r + 1)]
    if direction == 1:
        return [(y, r + 1 + distance) for y in range(t, b + 1)]
    if direction == 2:
        return [(b + 1 + distance, x) for x in range(l, r + 1)]
    return [(y, l - 1 - distance) for y in range(t, b + 1)]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _exact_bar(g, cells, color):
    if not all(_inside(p, g.shape) and g[p] == color for p in cells):
        return False
    first, last = cells[0], cells[-1]
    dy = int(last[0] != first[0])
    dx = int(last[1] != first[1])
    before, after = (first[0] - dy, first[1] - dx), (last[0] + dy, last[1] + dx)
    return all(not _inside(p, g.shape) or g[p] != color for p in (before, after))


def _vehicles(g, bg):
    found = []
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        for cells in _components(map(tuple, np.argwhere(g == color))):
            ys, xs = zip(*cells)
            t, b, l, r = min(ys), max(ys), min(xs), max(xs)
            # 3x3 rings are relation tiles, not shuttles.
            if b - t < 2 or r - l < 2 or (b - t == 2 and r - l == 2):
                continue
            ring = {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)
                    if y in (t, b) or x in (l, r)}
            if cells != ring:
                continue
            interior = set(map(int, np.unique(g[t + 1:b, l + 1:r])))
            if len(interior) > 2 or interior & {bg, color}:
                continue
            attached = []
            for d in range(4):
                bar = _bar(t, b, l, r, d)
                if not all(_inside(p, g.shape) for p in bar):
                    continue
                marker = int(g[bar[0]])
                if marker not in (bg, color) and _exact_bar(g, bar, marker):
                    attached.append((d, bar, marker))
            if len(attached) != 1:
                continue
            d, bar, marker = attached[0]
            target = None
            for distance in range(1, 31):
                candidate = _bar(t, b, l, r, d, distance)
                if not all(_inside(p, g.shape) for p in candidate):
                    break
                if _exact_bar(g, candidate, marker):
                    target = candidate
                    break
                if any(g[p] != bg for p in candidate):
                    break
            if target is None:
                continue
            ys = [t, b] + [p[0] for p in target]
            xs = [l, r] + [p[1] for p in target]
            found.append(dict(t=t, b=b, l=l, r=r, direction=d, frame=color,
                              marker=marker, distance=distance, bar=bar, target=target,
                              bounds=(min(ys), max(ys), min(xs), max(xs))))
    _need(1 <= len(found) <= 4, "expected one to four uniquely framed shuttles")
    _need(len({v['frame'] for v in found}) == 1, "frame colors disagree")
    cover, swept = set(), set()
    for v in found:
        t, b, l, r = (v[k] for k in ('t', 'b', 'l', 'r'))
        body = {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)}
        v['cells'] = body | set(v['bar'])
        all_cells = v['cells'] | set(v['target'])
        y0, y1, x0, x1 = v['bounds']
        region = {(y, x) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)}
        _need(not (swept & region), "shuttle corridors overlap")
        _need(all(g[p] == bg for p in region - all_cells), "obstructed corridor")
        cover.update(all_cells)
        swept.update(region)
    return sorted(found, key=lambda v: (v['t'], v['l'])), cover, swept


def _support(center, kind):
    r, c = center
    return {(r + dy, c + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if kind == 'square' or dy == 0 or dx == 0}


def _read_tiles(g, bg, frame, covered, swept, line):
    _need(line not in (bg, frame), "line color conflicts")
    line_cells = {tuple(p) for p in np.argwhere(g == line)}
    _need(len(line_cells) >= 3 and not line_cells & swept, "invalid signal line")
    remaining = {tuple(p) for p in np.argwhere((g != bg) & (g != line))} - covered
    tokens = []
    for cells in _components(remaining):
        candidates = []
        for center in sorted(cells):
            for kind in ('square', 'cross'):
                if _support(center, kind) != cells:
                    continue
                bodies = {int(g[p]) for p in cells if p != center}
                if len(bodies) == 1:
                    body = next(iter(bodies))
                    if body not in (bg, frame, line) and int(g[center]) not in (bg, frame, line):
                        candidates.append(dict(center=center, kind=kind, cells=cells,
                                               body=body, value=int(g[center])))
        _need(len(candidates) == 1, "relation tile has no unique center")
        _need(not cells & swept, "tile in a travel corridor")
        tokens.extend(candidates)
    _need(3 <= len(tokens) <= 8, "expected three to eight relation tiles")
    graph = {p: {(p[0] + dy, p[1] + dx) for dy, dx in _D4} & line_cells for p in line_cells}
    _need(all(len(adj) in (1, 2) for adj in graph.values()), "signal line branches or is broken")
    ends = {p for p in graph if len(graph[p]) == 1}
    parts = _components(line_cells)
    _need(1 <= len(parts) <= 2 and all(len(p & ends) == 2 for p in parts), "invalid open signal paths")
    for token in tokens:
        r, c = token['center']
        token['ends'] = {(r + 2 * dy, c + 2 * dx) for dy, dx in _D4} & ends
        _need(len(token['ends']) <= 1, "tile has two endpoint attachments")
        token['eligible'] = bool(token['ends'])
    _need(all(sum(p in t['ends'] for t in tokens) == 1 for p in ends), "each end must select one tile")
    mapping = {}
    for token in tokens:
        x, y = token['body'], token['value']
        _need(x not in mapping or mapping[x] == y, "conflicting donors")
        mapping[x] = y
    for i, token in enumerate(tokens):
        donors = [t for j, t in enumerate(tokens) if j != i and t['body'] == token['value']]
        token['donors'] = donors
        token['retained'] = token['eligible'] and bool(donors)
        token['composed'] = donors[0]['value'] if donors else None
    return tokens, graph, parts, mapping


def _parse(input_grid):
    g = _grid(input_grid)
    bg = int(g[0, 0])
    _need(all(np.all(edge == bg) for edge in (g[0], g[-1], g[:, 0], g[:, -1])), "background border required")
    vehicles, cover, swept = _vehicles(g, bg)
    frame = vehicles[0]['frame']
    explanations = []
    for line in map(int, np.unique(g)):
        try:
            tokens, graph, parts, mapping = _read_tiles(g, bg, frame, cover, swept, line)
            identities = {x for t in tokens for x in (t['body'], t['value'])}
            _need(all(v['marker'] in identities for v in vehicles), "unknown shuttle address")
            enriched = []
            for v in vehicles:
                _need(v['marker'] in mapping, "shuttle has no address tile")
                coords = _coords(v['t'], v['b'], v['l'], v['r'], v['direction'])
                values = [int(g[p]) for p in coords]
                fuel = mapping[v['marker']]
                donor_values = [t['value'] for t in tokens if t['body'] == fuel]
                spent = donor_values[0] if donor_values else None
                if spent is None:
                    _need(all(x == fuel for x in values), "undefined command needs a fresh counter")
                    advanced = 0
                else:
                    _need(fuel != spent, "counter states must differ")
                    advanced = values.count(spent)
                    _need(values == [spent] * advanced + [fuel] * (len(values) - advanced),
                          "counter is not a direction-relative prefix")
                enabled = any(t['body'] == v['marker'] and t['retained'] for t in tokens)
                _need(set(values) <= identities, "nonidentity counter color")
                enriched.append(dict(v, fuel=fuel, spent=spent, advanced=advanced, capacity=len(values),
                                     enabled=enabled, movement=min(len(values) - advanced, v['distance']) if enabled else 0))
            explanations.append(dict(background=bg, frame=frame, line=line, tokens=tokens,
                                     graph=graph, parts=parts, vehicles=enriched))
        except AssertionError:
            continue
    _need(len(explanations) == 1, "input needs one complete geometric interpretation")
    return g, explanations[0]


def _render_solution(g, scene, mode='correct'):
    out = g.copy()
    bg = scene['background']
    for token in scene['tokens']:
        if token['retained']:
            out[token['center']] = token['composed']
        else:
            for p in token['cells']:
                out[p] = bg
    for v in scene['vehicles']:
        for p in v['cells']:
            out[p] = bg
    for v in scene['vehicles']:
        movement = v['movement']
        if mode == 'no_counter' and v['enabled']:
            movement = v['distance']
        if mode == 'no_selection' and v['spent'] is not None:
            movement = min(v['capacity'] - v['advanced'], v['distance'])
        if mode == 'no_lookup':
            movement = 0
        dy, dx = _D4[v['direction']]
        t, b, l, r = (v['t'] + dy * movement, v['b'] + dy * movement,
                      v['l'] + dx * movement, v['r'] + dx * movement)
        out[t:b + 1, l:r + 1] = v['frame']
        for i, p in enumerate(_coords(t, b, l, r, v['direction'])):
            out[p] = v['spent'] if i < v['advanced'] + movement and v['spent'] is not None else v['fuel']
        for p in _bar(t, b, l, r, v['direction']):
            out[p] = v['marker']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Compose original endpoint-selected relations, then tick their addressed shuttles."""
    g, scene = _parse(input_grid)
    return _render_solution(g, scene)


def _unit_step_solution(g, scene):
    """Counter check using rotated blocks and single ticks, without _coords."""
    out = g.copy()
    for token in scene['tokens']:
        if token['retained']:
            out[token['center']] = token['composed']
        else:
            for p in token['cells']:
                out[p] = scene['background']
    for v in scene['vehicles']:
        for p in v['cells']:
            out[p] = scene['background']
    for v in scene['vehicles']:
        block = g[v['t']:v['b'] + 1, v['l']:v['r'] + 1].copy()
        canonical = np.rot90(block, v['direction']).copy()
        interior = canonical[1:-1, 1:-1].copy()
        movement = 0
        while v['enabled'] and movement < v['distance']:
            fuel = np.flatnonzero(interior.ravel() == v['fuel'])
            if not len(fuel):
                break
            interior.flat[int(fuel[0])] = v['spent']
            movement += 1
        canonical[1:-1, 1:-1] = interior
        block = np.rot90(canonical, -v['direction'])
        dy, dx = _D4[v['direction']]
        t, l = v['t'] + dy * movement, v['l'] + dx * movement
        h, w = block.shape
        out[t:t + h, l:l + w] = block
        for p in _bar(t, t + h - 1, l, l + w - 1, v['direction']):
            out[p] = v['marker']
    return out


def _expand(cells, shape):
    return {(r + dy, c + dx) for r, c in cells for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if _inside((r + dy, c + dx), shape)}


def _place_rectangle(rng, blocked, h, w):
    H, W = blocked.shape
    if h > H or w > W:
        return None
    integral = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]
    positions = np.argwhere(sums == 0)
    if not len(positions):
        return None
    return tuple(map(int, positions[int(rng.integers(len(positions)))]))


def _route(rng, shape, blocked, start, finish):
    if start in blocked or finish in blocked or start == finish:
        return None
    costs = rng.uniform(1.0, 1.2, size=shape)
    queue, distance, previous = [(0.0, start)], {start: 0.0}, {}
    while queue:
        cost, p = heapq.heappop(queue)
        if cost != distance[p]:
            continue
        if p == finish:
            path = [p]
            while p != start:
                p = previous[p]
                path.append(p)
            return path[::-1] if len(path) >= 3 else None
        for dy, dx in _D4:
            q = p[0] + dy, p[1] + dx
            if not _inside(q, shape) or q in blocked:
                continue
            new = cost + float(costs[q])
            if new < distance.get(q, float('inf')):
                distance[q], previous[q] = new, p
                heapq.heappush(queue, (new, q))
    return None


def _plan(rng, lesson, split, shift):
    """Abstract visible network; indices will be replaced by arbitrary colors."""
    if lesson == 'relay':
        relations, selected = [(0, 1), (1, 2), (2, 3), (4, 1)], [0, 1]
        stops = list(rng.permutation(['fuel', 'destination']))
        # Keep the fresh address unique: another counter with its same address
        # must not supply a local spent-color prefix in place of remote lookup.
        queries = [0, 1, stops.index('fuel')]
        stops.append('exact')
        near = None
    elif lesson == 'gates':
        relations, selected, queries = [(0, 1), (1, 2), (3, 4), (5, 1)], [0, 2], [0, 0, 3, 5]
        stops, near = ['exact', 'zero', 'random', 'destination'], 3
    else:
        k = int(rng.integers(4, 8))
        mapping = {i: int(rng.integers(i + 1, k)) for i in range(k - 1)}
        mapping[0], mapping[1] = 1, 2  # Ensure one composable edge, without fixing the other edges.
        relations = list(mapping.items())
        nselect = int(rng.integers(1, min(4, len(relations)) + 1))
        selected = [0] + list(map(int, rng.choice(np.arange(1, len(relations)), nselect - 1, replace=False)))
        if rng.random() < .3 and len(relations) < 8:
            relations.append(relations[int(rng.integers(len(relations)))])
        count = int(rng.integers(1, 5))
        if rng.random() < .35:
            queries = [0] + list(map(int, rng.choice(list(mapping), count - 1)))
        else:
            count = min(count, len(mapping))
            queries = [0] + list(map(int, rng.choice(np.arange(1, len(mapping)), count - 1, replace=False)))
        stops = [str(rng.choice(['fuel', 'destination', 'exact', 'zero'])) for _ in queries]
        near = None
    mapping = dict(relations)
    descriptions = []
    for query, stop in zip(queries, stops):
        y, z = mapping[query], mapping.get(mapping[query])
        enabled = any(relations[i][0] == query and relations[i][1] in mapping for i in selected)
        descriptions.append(dict(query=query, fuel=y, spent=z, enabled=enabled, stop=stop,
                                 fresh=lesson == 'relay' and enabled and stop == 'destination'))
    return relations, selected, descriptions, near


def _vehicle_spec(rng, desc, direction, long, deep, informative):
    a = int(rng.integers(5, 9)) if deep else int(rng.integers(2 if informative else 1, 5 if not long else 7))
    b = int(rng.integers(2 if informative or long else 1, 5))
    if a * b == 1:
        a = 2
    cap = a * b
    low, high = (6, 11) if long else (1, 6)
    stop = desc['stop']
    if stop == 'fuel':
        distance = int(rng.integers(max(2, low), high))
        remaining = int(rng.integers(1, min(cap - (1 if informative else 0), distance - 1) + 1))
    elif stop == 'destination':
        upper = min(high - 1, cap - 2 if informative else cap - 1)
        if upper < low:
            return None
        distance = int(rng.integers(low, upper + 1))
        remaining = int(rng.integers(distance + 1, cap - (1 if informative else 0) + 1))
    elif stop == 'exact':
        upper = min(high - 1, cap - (1 if informative else 0))
        if upper < low:
            return None
        distance = int(rng.integers(low, upper + 1))
        remaining = distance
    else:
        distance = int(rng.integers(low, high))
        remaining = 0 if stop == 'zero' else int(rng.integers(cap + 1))
    if desc['spent'] is None:
        remaining = cap
    elif desc['fresh']:
        remaining = cap
    ih, iw = (a, b) if direction in (0, 2) else (b, a)
    h, w = ih + 2, iw + 2
    fh, fw = (h + distance + 1, w) if direction in (0, 2) else (h, w + distance + 1)
    return dict(desc, direction=direction, ih=ih, iw=iw, distance=distance,
                advanced=cap - remaining, capacity=cap, fh=fh, fw=fw)


def _make_pair(rng, palette, directions, lesson, split, shift, active_offset):
    relations, selected, descriptions, near_index = _plan(rng, lesson, split, shift)
    bg, frame, line = palette[:3]
    ids = palette[3:]
    shape = (int(rng.integers(22, 29)), int(rng.integers(22, 29)))
    specs, active_index = [], active_offset
    for desc in descriptions:
        if desc['enabled'] and lesson:
            direction = directions[active_index % len(directions)]
            active_index += 1
        else:
            direction = int(rng.choice(directions))
        informative = bool(lesson and desc['enabled'])
        spec = _vehicle_spec(rng, desc, direction,
                             split == 'test' and shift == 'distance',
                             split == 'test' and shift == 'depth', informative)
        if spec is None:
            return None
        specs.append(spec)
    blocked = np.zeros(shape, dtype=bool)
    occupied, vehicles = set(), []
    for s in sorted(specs, key=lambda v: -v['fh'] * v['fw']):
        position = _place_rectangle(rng, blocked, s['fh'], s['fw'])
        if position is None:
            return None
        y, x = position
        region = {(r, c) for r in range(y, y + s['fh']) for c in range(x, x + s['fw'])}
        occupied.update(region)
        for p in _expand(region, shape):
            blocked[p] = True
        t = y + s['distance'] + 1 if s['direction'] == 0 else y
        l = x + s['distance'] + 1 if s['direction'] == 3 else x
        vehicles.append(dict(s, t=t, b=t + s['ih'] + 1, l=l, r=l + s['iw'] + 1))
    centers, supports, kinds = [], [], []
    for _ in relations:
        position = _place_rectangle(rng, blocked, 3, 3)
        if position is None:
            return None
        y, x = position
        center = y + 1, x + 1
        kind = str(rng.choice(['square', 'cross']))
        cells = _support(center, kind)
        centers.append(center)
        supports.append(cells)
        kinds.append(kind)
        for p in _expand(_support(center, 'square'), shape):
            blocked[p] = True
    # Pair selected tiles into one or two open paths. Their colors need not be related.
    order = list(map(int, rng.permutation(selected)))
    groups = [order[i:i + 2] for i in range(0, len(order), 2)]
    # An odd number of endpoints needs a second selected copy, visibly consistent.
    if len(groups[-1]) == 1:
        index = groups[-1][0]
        if len(relations) >= 8:
            return None
        position = _place_rectangle(rng, blocked, 3, 3)
        if position is None:
            return None
        center = position[0] + 1, position[1] + 1
        relations.append(relations[index])
        centers.append(center)
        kinds.append(str(rng.choice(['square', 'cross'])))
        supports.append(_support(center, kinds[-1]))
        selected.append(len(relations) - 1)
        groups[-1].append(len(relations) - 1)
    tile_cells = set().union(*supports)
    gray, selected_so_far = set(), set()
    for first, second in groups:
        route = None
        for _ in range(16):
            stubs = []
            for i in (first, second):
                r, c = centers[i]
                candidates = [(r + 2 * dy, c + 2 * dx) for dy, dx in _D4]
                candidates = [p for p in candidates if _inside(p, shape) and p not in occupied | tile_cells
                              and all(j == i or p not in _expand(cells, shape) for j, cells in enumerate(supports))]
                if not candidates:
                    break
                stubs.append(candidates[int(rng.integers(len(candidates)))])
            if len(stubs) != 2:
                continue
            forbidden = occupied | tile_cells | _expand(gray, shape)
            path = _route(rng, shape, forbidden, *stubs)
            if path is None:
                continue
            proposed = gray | set(path)
            graph = {p: {(p[0] + dy, p[1] + dx) for dy, dx in _D4} & proposed for p in proposed}
            if not all(len(adj) in (1, 2) for adj in graph.values()):
                continue
            ends = {p for p in graph if len(graph[p]) == 1}
            actual = {i for i, (r, c) in enumerate(centers)
                      if any((r + 2 * dy, c + 2 * dx) in ends for dy, dx in _D4)}
            # Require exactly the intended selected tiles after each new path.
            if actual != selected_so_far | {first, second}:
                continue
            route = set(path)
            break
        if route is None:
            return None
        gray.update(route)
        selected_so_far.update((first, second))
    if near_index is not None:
        # Place the off-line controller beside an interior line cell. Merely
        # touching a line must not activate its shuttle.
        graph = {p: {(p[0] + dy, p[1] + dx) for dy, dx in _D4} & gray for p in gray}
        ends = {p for p in graph if len(graph[p]) == 1}
        other_tiles = set().union(*(cells for i, cells in enumerate(supports) if i != near_index))
        candidates = sorted({(r + 2 * dy, c + 2 * dx) for r, c in gray - ends for dy, dx in _D4})
        rng.shuffle(candidates)
        found = None
        for center in candidates:
            cells = _support(center, kinds[near_index])
            if not all(_inside(p, shape) for p in _support(center, 'square')):
                continue
            if cells & (gray | _expand(occupied | other_tiles, shape)):
                continue
            if any((center[0] + 2 * dy, center[1] + 2 * dx) in ends for dy, dx in _D4):
                continue
            found = center, cells
            break
        if found is None:
            return None
        centers[near_index], supports[near_index] = found
    g = np.full(shape, bg, dtype=np.int8)
    for p in gray:
        g[p] = line
    for (body, value), center, cells in zip(relations, centers, supports):
        for p in cells:
            g[p] = ids[body]
        g[center] = ids[value]
    for v in vehicles:
        t, b, l, r = (v[k] for k in ('t', 'b', 'l', 'r'))
        g[t:b + 1, l:r + 1] = frame
        for i, p in enumerate(_coords(t, b, l, r, v['direction'])):
            g[p] = ids[v['spent']] if i < v['advanced'] else ids[v['fuel']]
        for p in _bar(t, b, l, r, v['direction']) + _bar(t, b, l, r, v['direction'], v['distance']):
            g[p] = ids[v['query']]
    # Crop empty margins, then retain an explicit background border for role inference.
    ys, xs = np.where(g != bg)
    g = g[int(ys.min()):int(ys.max()) + 1, int(xs.min()):int(xs.max()) + 1]
    g = np.pad(g, 1, constant_values=bg)
    # Construction computes output from its visible plan, not by calling solve.
    out = g.copy()
    offset_y, offset_x = 1 - int(ys.min()), 1 - int(xs.min())
    mapping = dict(relations)
    for i, ((body, value), center, cells) in enumerate(zip(relations, centers, supports)):
        retained = i in selected and any(j != i and b == value for j, (b, _) in enumerate(relations))
        if retained:
            out[center[0] + offset_y, center[1] + offset_x] = ids[mapping[value]]
        else:
            for y, x in cells:
                out[y + offset_y, x + offset_x] = bg
    for v in vehicles:
        for y in range(v['t'], v['b'] + 1):
            for x in range(v['l'], v['r'] + 1):
                out[y + offset_y, x + offset_x] = bg
        for y, x in _bar(v['t'], v['b'], v['l'], v['r'], v['direction']):
            out[y + offset_y, x + offset_x] = bg
    for v in vehicles:
        movement = min(v['capacity'] - v['advanced'], v['distance']) if v['enabled'] else 0
        dy, dx = _D4[v['direction']]
        t, b = v['t'] + dy * movement + offset_y, v['b'] + dy * movement + offset_y
        l, r = v['l'] + dx * movement + offset_x, v['r'] + dx * movement + offset_x
        out[t:b + 1, l:r + 1] = frame
        for i, p in enumerate(_coords(t, b, l, r, v['direction'])):
            out[p] = ids[v['spent']] if i < v['advanced'] + movement else ids[v['fuel']]
        for p in _bar(t, b, l, r, v['direction']):
            out[p] = ids[v['query']]
    return {'input': g.copy(), 'output': out}


def _key(g):
    g = np.asarray(g, dtype=np.int8)
    return g.shape, g.tobytes()


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    varied_palette = bool(rng.random() < .6)
    directions = list(map(int, rng.choice(4, int(rng.integers(2, 4)), replace=False)))
    shift = str(rng.choice(['distance', 'depth']))
    lessons = ['relay', 'gates'] + [None] * (ntrain - 2)
    # Lessons are shuffled, but their active orientations are coordinated.
    rng.shuffle(lessons)
    seen, puzzle = set(), {'train': [], 'test': []}
    for split, count in (('train', ntrain), ('test', ntest)):
        for i in range(count):
            lesson = lessons[i] if split == 'train' else None
            for _ in range(96):
                local_palette = list(map(int, rng.permutation(10))) if varied_palette else palette
                pair = _make_pair(rng, local_palette, directions, lesson, split, shift,
                                  2 if lesson == 'gates' else 0)
                if pair is None or _key(pair['input']) in seen:
                    continue
                try:
                    _need(np.array_equal(solve(pair['input']), pair['output']), 'construction/parser disagreement')
                except AssertionError:
                    continue
                seen.add(_key(pair['input']))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError('bounded geometric construction failed')
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(g):
    _, s = _parse(g)
    vs, ts = s['vehicles'], s['tokens']
    ends = {p for p, adjacent in s['graph'].items() if len(adjacent) == 1}
    near_interior = lambda t: any((t['center'][0] + 2 * dy, t['center'][1] + 2 * dx) in s['graph']
                                  and (t['center'][0] + 2 * dy, t['center'][1] + 2 * dx) not in ends
                                  for dy, dx in _D4)
    return dict(shape=g.shape, vehicles=len(vs), tokens=len(ts), lines=len(s['parts']),
                palette=(s['background'], s['frame'], s['line']),
                kinds=tuple(sorted({t['kind'] for t in ts})),
                active=sum(v['enabled'] for v in vs),
                directions={v['direction'] for v in vs if v['enabled'] and v['movement']},
                regimes={('zero' if v['capacity'] == v['advanced'] else
                          'fuel' if v['capacity'] - v['advanced'] < v['distance'] else
                          'exact' if v['capacity'] - v['advanced'] == v['distance'] else 'destination')
                         for v in vs if v['enabled']},
                interior_gate=any(not t['eligible'] and t['donors'] and near_interior(t)
                                  and any(v['marker'] == t['body'] and not v['enabled']
                                          and v['capacity'] > v['advanced'] for v in vs) for t in ts),
                missing_gate=any(t['eligible'] and not t['donors']
                                 and any(v['marker'] == t['body'] for v in vs) for t in ts),
                simultaneous=any(t['retained'] and any(d['retained'] and d['composed'] != d['value']
                                                     for d in t['donors']) for t in ts),
                shared=len(vs) > len({v['marker'] for v in vs}),
                shared_moving=any(a is not b and a['marker'] == b['marker'] and a['movement'] and b['movement']
                                  for a in vs for b in vs),
                max_distance=max(v['distance'] for v in vs),
                depths={(v['b'] - v['t'] - 1 if v['direction'] in (0, 2) else v['r'] - v['l'] - 1) for v in vs},
                capacities={v['capacity'] for v in vs},
                state_pairs={(v['fuel'], v['spent']) for v in vs if v['spent'] is not None},
                advanced={v['advanced'] for v in vs},
                fresh=any(v['advanced'] == 0 for v in vs),
                exhausted=any(v['advanced'] == v['capacity'] for v in vs),
                active_fresh=any(v['enabled'] and v['advanced'] == 0 and v['movement'] for v in vs),
                fresh_unshared=any(v['enabled'] and v['advanced'] == 0 and v['movement']
                                   and sum(w['marker'] == v['marker'] for w in vs) == 1 for v in vs),
                active_exhausted=any(v['enabled'] and v['advanced'] == v['capacity'] for v in vs),
                mixed_moving=any(v['enabled'] and 0 < v['advanced'] < v['capacity'] and v['movement']
                                 and v['b'] - v['t'] > 2 and v['r'] - v['l'] > 2 for v in vs),
                opposite_state_roles=any(a['enabled'] and b['enabled'] and a['fuel'] == b['spent']
                                         and a['spent'] != b['fuel'] for a in vs for b in vs),
                multirow=any(v['movement'] and v['advanced'] // (v['r'] - v['l'] - 1 if v['direction'] in (0, 2) else v['b'] - v['t'] - 1)
                             != (v['advanced'] + v['movement']) // (v['r'] - v['l'] - 1 if v['direction'] in (0, 2) else v['b'] - v['t'] - 1) for v in vs))


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle schema')
    _need(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training, testing = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(inp.shape == out.shape, 'dimensions changed')
            _need(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            g, scene = _parse(inp)
            expected = _render_solution(g, scene)
            _need(np.array_equal(expected, out), 'incorrect output')
            _need(np.array_equal(_unit_step_solution(g, scene), out), 'rotated unit ticks disagree')
            _need(all(out[p] == scene['line'] for p in scene['graph']), 'signal scaffold changed')
            for v in scene['vehicles']:
                _need(all(out[p] == v['marker'] for p in v['target']), 'destination changed')
                if not v['enabled']:
                    _need(all(out[p] == inp[p] for p in v['cells']), 'inactive shuttle moved')
            (training if split == 'train' else testing).append(_features(inp))
    _need({'fuel', 'destination', 'exact'} <= set().union(*(f['regimes'] for f in training)), 'stopping causes not taught')
    _need(len(set().union(*(f['directions'] for f in training))) >= 2, 'rotated counting not taught')
    _need(any(f['simultaneous'] for f in training), 'original-value lookup not taught')
    _need(any(f['interior_gate'] for f in training), 'endpoint selection not contrasted')
    _need(any(f['missing_gate'] for f in training), 'missing-donor stopping not taught')
    for name in ('fresh_unshared', 'active_exhausted', 'mixed_moving', 'opposite_state_roles', 'shared_moving'):
        _need(any(f[name] for f in training), f'training lacks {name}')
    for wrong in ('no_counter', 'no_selection', 'no_lookup'):
        _need(any(not np.array_equal(_render_solution(*_parse(p['input']), mode=wrong), p['output'])
                  for p in puzzle['train']), f'training fails to refute {wrong}')
    _need(set().union(*(f['directions'] for f in testing)) <= set().union(*(f['directions'] for f in training)),
          'test uses an untaught active direction')
    distance_shift = min(f['max_distance'] for f in testing) > max(f['max_distance'] for f in training)
    depth_shift = min(min(f['depths']) for f in testing) > max(max(f['depths']) for f in training)
    _need(distance_shift or depth_shift, 'missing structural test shift')


def self_test():
    """42 streamed seeds; no samples, traces, logs or other disk writes."""
    features, puzzle_keys = [], set()
    donor_checked, counter_checked = False, False
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzle_keys.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        features.extend(_features(p['input']) for s in ('train', 'test') for p in puzzle[s])
        for pair in puzzle['train']:
            inp, scene = _parse(pair['input'])
            for v in scene['vehicles']:
                if not counter_checked and v['enabled'] and 0 < v['capacity'] - v['advanced'] < v['distance']:
                    changed = inp.copy()
                    p = _coords(v['t'], v['b'], v['l'], v['r'], v['direction'])[v['advanced']]
                    changed[p] = v['spent']
                    _need(not np.array_equal(solve(changed), pair['output']), 'counter does not affect the answer')
                    counter_checked = True
                # Change only a remote donor. Require all counters using its value
                # to be fresh, so the perturbed input remains a valid member.
                if donor_checked or not v['enabled'] or v['advanced'] != 0:
                    continue
                if any(w['marker'] == v['fuel'] or (w['fuel'] == v['fuel'] and w['advanced'])
                       for w in scene['vehicles']):
                    continue
                candidates = sorted({t['body'] for t in scene['tokens']} |
                                    {t['value'] for t in scene['tokens']})
                replacement = next((c for c in candidates if c not in (v['fuel'], v['spent'])), None)
                if replacement is None:
                    continue
                changed = inp.copy()
                for token in scene['tokens']:
                    if token['body'] == v['fuel']:
                        changed[token['center']] = replacement
                _need(not np.array_equal(solve(changed), pair['output']), 'remote donor does not affect the answer')
                donor_checked = True
        if seed == 0:
            repeat = generate_puzzle(seed)
            _need(all(np.array_equal(p[k], q[k]) for s in ('train', 'test')
                      for p, q in zip(puzzle[s], repeat[s]) for k in ('input', 'output')), 'nondeterministic seed')
            damaged = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
            damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
            try:
                validate_generated_puzzle(damaged)
            except AssertionError:
                pass
            else:
                raise AssertionError('validator accepted corrupted output')
    _need(len(puzzle_keys) == 42, 'seed diversity collapsed')
    _need(donor_checked and counter_checked, 'missing parent-contribution sensitivity checks')
    _need(len({f['shape'] for f in features}) >= 35, 'canvas diversity collapsed')
    _need({1, 2, 3, 4} <= {f['vehicles'] for f in features}, 'vehicle count support collapsed')
    _need({1, 2} <= {f['lines'] for f in features}, 'path count support collapsed')
    _need(len(set().union(*(f['capacities'] for f in features))) >= 12, 'counter dimensions collapsed')
    _need(len({f['palette'] for f in features}) >= 40, 'structural palette variation collapsed')
    _need(set().union(*(f['directions'] for f in features)) == set(range(4)), 'missing direction')
    _need(all(any(f[name] for f in features) for name in ('fresh', 'exhausted', 'multirow', 'shared')), 'missing family behavior')
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + 10 * a + b, a, b)
            _need(len(p['train']) == a and len(p['test']) == b, 'count contract failed')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
