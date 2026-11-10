"""Rooted relay grafts: color requests and rigid geometric keys act together.

The hollow hub's interior legend visibly supplies the background-independent
pin and inlet-marker roles. No sampled parameter is needed by solve(). Pieces
rotate by quarter turns, preserve their complete cells, and are retained exactly
when reachable through matching ports from the stationary hub.
"""
from __future__ import annotations

from collections import Counter, deque
import copy
import json

import numpy as np


N4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape):
        raise ValueError("grid dimensions must be 1..30")
    if not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9)):
        raise ValueError("grid must contain integer colors 0..9")
    return a.astype(np.int8, copy=True)


def _bbox(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells), max(c for r, c in cells))


def _normal(cells):
    r, c, _, _ = _bbox(cells)
    return tuple(sorted((a-r, b-c) for a, b in cells))


def _rotate_point(q, turns):
    r, c = q
    for _ in range(turns % 4):
        r, c = -c, r  # Counterclockwise in row/column coordinates.
    return r, c


def _transform(obj, turns=0, delta=(0, 0)):
    def point(q):
        r, c = _rotate_point(q, turns)
        return r+delta[0], c+delta[1]
    t, l, b, r = obj['box']
    corners = [point(q) for q in ((t, l), (t, r), (b, l), (b, r))]
    result = dict(cells={point(q): color for q, color in obj['cells'].items()},
                  box=_bbox(corners), color=obj['color'], root=obj['root'],
                  entry=None if obj['entry'] is None else (obj['entry']-turns) % 4,
                  ports=[dict(side=(p['side']-turns) % 4, label=p['label'],
                              pins={point(q) for q in p['pins']}) for p in obj['ports']])
    if 'labels' in obj:
        result['labels'] = tuple(obj['labels'][(s+turns) % 4] for s in range(4))
    return result


def _side_length(box, side):
    t, l, b, r = box
    return r-l+1 if side % 2 == 0 else b-t+1


def _edge_point(box, side, tangent, distance=0):
    t, l, b, r = box
    return ((t-distance, tangent), (tangent, r+distance),
            (b+distance, tangent), (tangent, l-distance))[side]


def _projection(box, side, q):
    t, l, b, r = box
    a, c = q
    return ((c, t-a), (a, c-r), (c, a-b), (a, l-c))[side]


def _midpoint(box, side):
    t, l, b, r = box
    tangent = l+(r-l+1)//2 if side % 2 == 0 else t+(b-t+1)//2
    return _edge_point(box, side, tangent)


def _border(box):
    t, l, b, r = box
    return ({(t, c) for c in range(l, r+1)} | {(b, c) for c in range(l, r+1)} |
            {(a, l) for a in range(t, b+1)} | {(a, r) for a in range(t, b+1)})


def _components(cells):
    todo = set(cells)
    groups = []
    while todo:
        q = min(todo)
        todo.remove(q)
        found, stack = {q}, [q]
        while stack:
            r, c = stack.pop()
            for dr, dc in N8:
                p = r+dr, c+dc
                if p in todo:
                    todo.remove(p)
                    found.add(p)
                    stack.append(p)
        groups.append(found)
    return groups


def _port_cells(box, side, label, pins, pin_color, root=False):
    """A rail and short fingers reach all pins; pins need not be connected."""
    base = 0 if root else 1
    projected = [_projection(box, side, q) for q in pins]
    assert len(pins) in (2, 3)
    low, high = min(a for a, d in projected), max(a for a, d in projected)
    t, l, b, r = box
    start, end = (l, r) if side % 2 == 0 else (t, b)
    assert start < low < high < end
    assert 2 <= high-low+1 <= 5
    assert all(d-base in (1, 2) for a, d in projected)
    cells = {_edge_point(box, side, a, base): label for a in range(low, high+1)}
    for a, distance in projected:
        for d in range(base+1, distance):
            cells[_edge_point(box, side, a, d)] = label
    cells.update({q: pin_color for q in pins})
    return cells


def _add_port(obj, side, label, pins, pin_color):
    assert not any(p['side'] == side for p in obj['ports'])
    cells = _port_cells(obj['box'], side, label, pins, pin_color, obj['root'])
    for q, hue in cells.items():
        assert q not in obj['cells'] or obj['cells'][q] == hue
    obj['cells'].update(cells)
    obj['ports'].append(dict(side=side, label=label, pins=set(pins)))


def _parse_ports(mapping, box, pin_color, root=False, labels=None):
    t, l, b, r = box
    groups = {s: set() for s in range(4)}
    for q, hue in mapping.items():
        if hue != pin_color or t <= q[0] <= b and l <= q[1] <= r:
            continue
        sides = [s for s in range(4) if _projection(box, s, q)[1] > 0
                 and ((l < q[1] < r) if s % 2 == 0 else (t < q[0] < b))]
        assert len(sides) == 1, 'pins must project onto one side, away from corners'
        groups[sides[0]].add(q)
    ports = []
    for side, pins in groups.items():
        if not pins:
            continue
        low = min(_projection(box, side, q)[0] for q in pins)
        rail = _edge_point(box, side, low, 0 if root else 1)
        assert rail in mapping
        label = int(labels[side]) if root else mapping[rail]
        expected = _port_cells(box, side, label, pins, pin_color, root)
        assert all(mapping.get(q) == hue for q, hue in expected.items()), 'incomplete rail/key'
        ports.append(dict(side=side, label=label, pins=pins))
    return ports


def _find_root(grid, bg):
    roots = []
    h, w = grid.shape
    for size in (7, 9):
        for t in range(h-size+1):
            for l in range(w-size+1):
                block = grid[t:t+size, l:l+size]
                if any(block[q] != bg for q in ((0, 0), (0, size-1), (size-1, 0), (size-1, size-1))):
                    continue
                bars = (block[0, 1:-1], block[1:-1, -1], block[-1, 1:-1], block[1:-1, 0])
                labels = tuple(int(bar[0]) for bar in bars)
                if len(set(labels)) != 4 or bg in labels or any(np.any(bar != hue) for bar, hue in zip(bars, labels)):
                    continue
                m = size//2
                pin, inlet = int(block[m, m]), int(block[m+1, m])
                if len(set(labels + (bg, pin, inlet))) != 7 or block[m, m+1] != pin:
                    continue
                interior = np.full((size-2, size-2), bg, dtype=np.int8)
                interior[m-1, m-1:m+1] = pin
                interior[m, m-1] = inlet
                if not np.array_equal(block[1:-1, 1:-1], interior):
                    continue
                box = t, l, t+size-1, l+size-1
                mapping = {(a, c): int(grid[a, c])
                           for a in range(max(0, t-2), min(h, t+size+2))
                           for c in range(max(0, l-2), min(w, l+size+2)) if grid[a, c] != bg}
                ports = _parse_ports(mapping, box, pin, True, labels)
                core = {(t+a, l+c): int(block[a, c]) for a, c in np.argwhere(block != bg)}
                expected = dict(core)
                for p in ports:
                    expected.update(_port_cells(box, p['side'], p['label'], p['pins'], pin, True))
                if expected != mapping:
                    continue
                roots.append(dict(cells=mapping, box=box, color=None, root=True,
                                  entry=None, ports=ports, labels=labels, pin=pin, inlet=inlet))
    assert len(roots) == 1, 'one hollow hub with a three-cell role legend is required'
    return roots[0]


def _parse_piece(mapping, bg, pin, inlet):
    marked = [q for q, hue in mapping.items() if hue == inlet]
    assert len(marked) == 1, 'each module has exactly one inlet cue'
    mr, mc = marked[0]
    candidates = []
    for h in range(5, 9):
        for w in range(5, 9):
            # Both middle cells on an even side are valid. This makes the
            # grammar closed under quarter turns instead of discarding a
            # rotated rectangle for an arbitrary top/left midpoint convention.
            boxes = []
            for offset in sorted({w//2, (w-1)//2}):
                boxes.extend(((0, (mr, mc-offset, mr+h-1, mc-offset+w-1)),
                              (2, (mr-h+1, mc-offset, mr, mc-offset+w-1))))
            for offset in sorted({h//2, (h-1)//2}):
                boxes.extend(((1, (mr-offset, mc-w+1, mr-offset+h-1, mc)),
                              (3, (mr-offset, mc, mr-offset+h-1, mc+w-1))))
            for side, box in boxes:
                t, l, b, r = box
                hue = mapping.get((t, l), bg)
                if hue in (bg, pin, inlet):
                    continue
                if any(mapping.get(q, bg) != (inlet if q == (mr, mc) else hue) for q in _border(box)):
                    continue
                core = {q: mapping[q] for q in mapping if t <= q[0] <= b and l <= q[1] <= r}
                if any(c not in (hue, inlet) for c in core.values()):
                    continue
                try:
                    ports = _parse_ports(mapping, box, pin)
                    entrance = [p for p in ports if p['side'] == side]
                    assert len(entrance) == 1 and entrance[0]['label'] == hue
                    assert all(p['label'] not in (bg, pin, inlet) for p in ports)
                    expected = dict(core)
                    for p in ports:
                        expected.update(_port_cells(box, p['side'], p['label'], p['pins'], pin))
                    assert expected == mapping
                except AssertionError:
                    continue
                candidates.append(dict(cells=mapping, box=box, color=hue, root=False, entry=side, ports=ports))
    assert len(candidates) == 1, 'module core and inlet must be unambiguous'
    return candidates[0]


def _analyze(input_grid):
    grid = _grid(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(counts.argmax())
    assert counts[bg] > grid.size//2, 'background must occupy more than half the canvas'
    root = _find_root(grid, bg)
    remaining = {tuple(q): int(grid[tuple(q)]) for q in np.argwhere(grid != bg)}
    for q in root['cells']:
        remaining.pop(q)
    components = _components(remaining)
    pieces = [_parse_piece({q: remaining[q] for q in comp}, bg, root['pin'], root['inlet']) for comp in components]
    # The hub's legend is exempt from connectedness; each loose rigid module is
    # one spatial component, with an eight-neighbor buffer from every other one.
    groups = [set(root['cells'])] + [set(p['cells']) for p in pieces]
    for i, group in enumerate(groups):
        halo = {(r+dr, c+dc) for r, c in group for dr, dc in N8}
        assert all(not halo & other for other in groups[i+1:])
    return grid, bg, root, pieces


def _inlet_port(piece):
    return next(p for p in piece['ports'] if p['side'] == piece['entry'])


def _pose(piece, port, ignore_color=False, ignore_key=False):
    if not ignore_color and piece['color'] != port['label']:
        return None
    target_side = (port['side']+2) % 4
    turns = (piece['entry']-target_side) % 4
    rotated = _transform(piece, turns)
    entrance = _inlet_port(rotated)
    if not ignore_key and _normal(entrance['pins']) != _normal(port['pins']):
        return None
    ar, ac, _, _ = _bbox(port['pins'])
    br, bc, _, _ = _bbox(entrance['pins'])
    delta = ar-br, ac-bc
    posed = _transform(rotated, delta=delta)
    if not ignore_key:
        assert _inlet_port(posed)['pins'] == port['pins']
    return posed, turns, delta


def _reconstruct(input_grid):
    grid, bg, root, pieces = _analyze(input_grid)
    placed, edges, waiting = {}, [], deque([(None, root, 0)])
    while waiting:
        parent_id, parent, depth = waiting.popleft()
        for port in parent['ports']:
            if port['side'] == parent['entry']:
                continue
            matches = [(i, pose) for i, p in enumerate(pieces) if (pose := _pose(p, port)) is not None]
            assert len(matches) == 1, 'each reachable request must have exactly one color-and-key match'
            i, (posed, turns, delta) = matches[0]
            assert i not in placed, 'a module cannot have two parents or close a cycle'
            assert all(0 <= q[0] < grid.shape[0] and 0 <= q[1] < grid.shape[1] for q in posed['cells']), 'clipping is forbidden'
            for other_id, other in [(None, root)] + [(j, p['object']) for j, p in placed.items()]:
                overlap = set(other['cells']) & set(posed['cells'])
                assert overlap == (port['pins'] if other_id == parent_id else set()), 'only a complete matched key may overlap'
                assert all(other['cells'][q] == posed['cells'][q] == root['pin'] for q in overlap)
            placed[i] = dict(object=posed, turns=turns, delta=delta, depth=depth+1)
            edges.append((parent_id, i, port))
            waiting.append((i, posed, depth+1))
    out = np.full_like(grid, bg)
    for obj in [root] + [p['object'] for p in placed.values()]:
        for q, hue in obj['cells'].items():
            assert out[q] in (bg, hue)
            out[q] = hue
    return out, dict(background=bg, root=root, pieces=pieces, placed=placed, edges=edges)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the entire answer from the visible hub, inlets, colors and keys."""
    return _reconstruct(input_grid)[0]


def _root(rng, size, labels, pin, inlet):
    obj = dict(cells={}, box=(0, 0, size-1, size-1), color=None,
               root=True, entry=None, ports=[], labels=tuple(labels))
    for side, hue in enumerate(labels):
        for a in range(1, size-1):
            obj['cells'][_edge_point(obj['box'], side, a)] = int(hue)
    m = size//2
    obj['cells'][m, m] = obj['cells'][m, m+1] = pin
    obj['cells'][m+1, m] = inlet
    return obj


def _body(rng, box, color, side, inlet, style=None):
    cells = _border(box)
    t, l, b, r = box
    style = int(rng.integers(4)) if style is None else style
    if style == 0:
        cells |= {(a, c) for a in range(t+1, b) for c in range(l+1, r)}
    elif style >= 2:
        target = int(rng.integers(1, max(2, (b-t-1)*(r-l-1))))
        for _ in range(target):
            frontier = sorted({(a+dr, c+dc) for a, c in cells for dr, dc in N4
                               if t < a+dr < b and l < c+dc < r} - cells)
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(len(frontier)))])
    mapping = {q: color for q in cells}
    mapping[_midpoint(box, side)] = inlet
    return dict(cells=mapping, box=box, color=color, root=False, entry=side, ports=[])


def _try_child(rng, objects, parent_index, side, color, pin, inlet,
               pin_count=None, body_style=None):
    parent = objects[parent_index]
    if side == parent['entry'] or any(p['side'] == side for p in parent['ports']):
        return None
    t, l, b, r = parent['box']
    start = l if side % 2 == 0 else t
    length = _side_length(parent['box'], side)
    span = int(rng.integers(3 if pin_count == 3 else 2, min(5, length-2)+1))
    offset = int(rng.integers(1, length-span))
    positions = [start+offset, start+offset+span-1]
    if pin_count == 3 or (pin_count is None and span >= 3 and rng.random() < .3):
        positions.append(int(rng.integers(positions[0]+1, positions[1])))
    base = 0 if parent['root'] else 1
    depths = rng.integers(1, 3, size=len(positions))
    pins = {_edge_point(parent['box'], side, a, base+int(d)) for a, d in zip(positions, depths)}
    tangential_size = int(rng.integers(max(5, span+2), 9))
    normal_size = int(rng.integers(5, 9))
    child_start = min(positions)-int(rng.integers(1, tangential_size-span))
    edge = _edge_point(parent['box'], side, child_start, base+4)
    if side == 0:
        box = edge[0]-normal_size+1, child_start, edge[0], child_start+tangential_size-1
    elif side == 1:
        box = child_start, edge[1], child_start+tangential_size-1, edge[1]+normal_size-1
    elif side == 2:
        box = edge[0], child_start, edge[0]+normal_size-1, child_start+tangential_size-1
    else:
        box = child_start, edge[1]-normal_size+1, child_start+tangential_size-1, edge[1]
    child = _body(rng, box, color, (side+2) % 4, inlet, body_style)
    _add_port(child, child['entry'], color, pins, pin)
    updated = copy.deepcopy(parent)
    _add_port(updated, side, color, pins, pin)
    for i, other in enumerate(objects):
        overlap = set(other['cells']) & set(child['cells'])
        if overlap != (pins if i == parent_index else set()):
            # The parent does not yet contain its new pins.
            if i != parent_index or overlap:
                return None
        if i != parent_index and (set(updated['cells']) - set(parent['cells'])) & set(other['cells']):
            return None
    allcells = set(child['cells']) | set(updated['cells']) | set().union(*(set(o['cells']) for o in objects))
    a, c, z, x = _bbox(allcells)
    if max(z-a+1, x-c+1) > 28:
        return None
    return updated, child


def _layout(rng, config, labels, palette, pin, inlet):
    for _ in range(100):
        objects = [_root(rng, int(rng.choice((7, 9), p=(.85, .15))), labels, pin, inlet)]
        success = True
        for index, (parent, forced_side) in enumerate(config['links'], 1):
            directions = [forced_side] if forced_side is not None else list(map(int, rng.permutation(4)))
            attached = None
            for side in directions:
                color = labels[side] if parent == 0 else int(rng.choice(palette))
                if parent != 0 and index == 3:
                    color = int(rng.choice([c for c in palette if c not in labels]))
                for _ in range(25):
                    attached = _try_child(rng, objects, parent, side, color, pin, inlet,
                                          config.get('pin_counts', {}).get(index),
                                          config.get('body_styles', {}).get(index))
                    if attached is not None:
                        break
                if attached is not None:
                    break
            if attached is None:
                success = False
                break
            objects[parent], child = attached
            objects.append(child)
        if success and 'key' in config.get('decoys', ()):
            source = objects[1]
            distances = {_projection(source['box'], source['entry'], q)[1]
                         for q in _inlet_port(source)['pins']}
            # A reflected normal-offset pattern has the same full bounding
            # box but different occupied cells. Flat keys cannot witness this.
            if len(distances) < 2:
                continue
        if success:
            return objects
    raise RuntimeError('could not fit a bounded relay tree')


def _decoy(rng, source, palette, pin, inlet, kind):
    t, l, b, r = source['box']
    core = {q: hue for q, hue in source['cells'].items() if t <= q[0] <= b and l <= q[1] <= r}
    color = source['color'] if kind == 'key' else int(rng.choice([c for c in palette if c != source['color']]))
    obj = dict(cells={q: (inlet if hue == inlet else color) for q, hue in core.items()},
               box=source['box'], color=color, root=False, entry=source['entry'], ports=[])
    pins = set(_inlet_port(source)['pins'])
    if kind == 'key':
        projected = [_projection(obj['box'], obj['entry'], q) for q in pins]
        assert len({distance for tangent, distance in projected}) == 2
        pins = {_edge_point(obj['box'], obj['entry'], tangent, 5-distance)
                for tangent, distance in projected}
        assert _normal(pins) != _normal(_inlet_port(source)['pins'])
    _add_port(obj, obj['entry'], color, pins, pin)
    return obj


def _scatter(rng, root, pieces, live_count, bg, config):
    # Choose final anchor coordinates from the actual reachable assembly extent.
    assembly = set(root['cells']) | set().union(*(set(o['cells']) for o in pieces[:live_count]))
    t, l, b, r = _bbox(assembly)
    bh, bw = b-t+1, r-l+1
    min_axis = 25 if len(pieces) >= 5 else 22 if len(pieces) >= 4 else 17
    h = int(rng.integers(max(min_axis, bh), 31))
    w = int(rng.integers(max(min_axis, bw), 31))
    for enlargement in range(2):
        if enlargement:
            h = w = 30
        for restart in range(24):
            if rng.random() < .3:
                shift = (int(rng.choice((0, h-bh)))-t, int(rng.choice((0, w-bw)))-l)
            else:
                shift = int(rng.integers(h-bh+1))-t, int(rng.integers(w-bw+1))-l
            anchored = _transform(root, delta=shift)
            # Reserve the whole hub halo so its sparse keys cannot absorb a
            # neighboring loose object during parsing.
            rt, rl, rb, rr = anchored['box']
            forbidden = {(a, c) for a in range(rt-3, rb+4) for c in range(rl-3, rr+4)}
            inp = np.full((h, w), bg, dtype=np.int8)
            for q, hue in anchored['cells'].items():
                inp[q] = hue
            order = list(map(int, rng.permutation(len(pieces))))
            order.sort(key=lambda i: -len(pieces[i]['cells']))
            ok = True
            for i in order:
                if i in config['turns']:
                    turns = config['turns'][i]
                else:
                    turns = int(rng.integers(4))
                oriented = _transform(pieces[i], turns)
                a, c, z, x = _bbox(oriented['cells'])
                ph, pw = z-a+1, x-c+1
                if ph > h or pw > w:
                    ok = False
                    break
                local = np.array(sorted(oriented['cells']), dtype=int) - (a, c)
                tops, lefts = np.indices((h-ph+1, w-pw+1))
                rows = tops.ravel()[:, None]+local[:, 0]
                cols = lefts.ravel()[:, None]+local[:, 1]
                blocked = np.zeros((h, w), dtype=bool)
                for q in forbidden:
                    if 0 <= q[0] < h and 0 <= q[1] < w:
                        blocked[q] = True
                choices = np.flatnonzero(~np.any(blocked[rows, cols], axis=1))
                if not len(choices):
                    ok = False
                    break
                k = int(rng.choice(choices))
                placed = _transform(oriented, delta=(int(tops.ravel()[k])-a, int(lefts.ravel()[k])-c))
                for q, hue in placed['cells'].items():
                    inp[q] = hue
                forbidden |= {(a+dr, c+dc) for a, c in placed['cells'] for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
            if ok and np.count_nonzero(inp == bg) > inp.size//2:
                expected = np.full_like(inp, bg)
                for obj in [anchored] + [_transform(p, delta=shift) for p in pieces[:live_count]]:
                    for q, hue in obj['cells'].items():
                        assert expected[q] in (bg, hue)
                        expected[q] = hue
                return inp, expected
    return None


def _configurations(rng, ntrain, ntest):
    sides = list(map(int, rng.permutation(4)))
    configs = [dict(links=[(0, sides[0]), (0, sides[1]), (1, None)], decoys=['color'],
                    turns={0: 0, 1: 3}, pin_counts={1: 2, 2: 3}, body_styles={1: 1, 2: 2, 3: 0}),
               dict(links=[(0, sides[2]), (0, sides[3]), (1, None), (1, None)], decoys=['key'],
                    turns={0: 2, 1: 1}, pin_counts={1: 2})]
    for _ in range(2, ntrain):
        count = int(rng.integers(1, 5))
        links = [(0, int(rng.integers(4)))]
        for j in range(2, count+1):
            parent = int(rng.choice([0, 1]))
            links.append((parent, None))
        decoys = list(rng.choice(['color', 'key'], size=int(rng.integers(0, 3))))
        configs.append(dict(links=links, decoys=decoys, turns={}))
    for _ in range(ntest):
        links = [(0, int(rng.integers(4))), (1, None), (1, None), (2, None)]
        if rng.random() < .35:
            links.append((0 if rng.random() < .5 else 3, None))
        configs.append(dict(links=links, decoys=[str(rng.choice(['color', 'key']))], turns={}))
    # The two evidence packets stay intact but their display order varies.
    first = configs[:2]
    if rng.random() < .5:
        first.reverse()
    return first + configs[2:]


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    bg, pin, inlet = palette[:3]
    payload = palette[3:]
    configs = _configurations(rng, ntrain, ntest)
    result, seen = {'train': [], 'test': []}, set()
    for i, config in enumerate(configs):
        for attempt in range(120):
            labels = list(map(int, rng.choice(payload, 4, replace=False)))
            objects = _layout(rng, config, labels, payload, pin, inlet)
            live_count = len(objects)-1
            pieces = objects[1:]
            for kind in config['decoys']:
                pieces.append(_decoy(rng, objects[1], payload, pin, inlet, kind))
            sampled = _scatter(rng, objects[0], pieces, live_count, bg, config)
            if sampled is None:
                continue
            inp, expected = sampled
            key = inp.shape, inp.tobytes()
            if key in seen:
                continue
            try:
                output, _ = _reconstruct(inp)
            except AssertionError:
                continue
            if not np.array_equal(output, expected):
                continue
            seen.add(key)
            result['train' if i < ntrain else 'test'].append(dict(input=inp, output=output))
            break
        else:
            raise RuntimeError('bounded sampling could not construct an unambiguous input')
    validate_generated_puzzle(result)
    return result


def _features(pair):
    out, a = _reconstruct(pair['input'])
    assert np.array_equal(out, np.asarray(pair['output'])), 'incorrect output'
    root, pieces, placed = a['root'], a['pieces'], a['placed']
    pin = root['pin']
    ownership = Counter(q for obj in [root]+[v['object'] for v in placed.values()]
                        for q, hue in obj['cells'].items() if hue == pin)
    legend = {(root['box'][0]+(root['box'][2]-root['box'][0]+1)//2,
               root['box'][1]+(root['box'][3]-root['box'][1]+1)//2)}
    q = next(iter(legend))
    legend.add((q[0], q[1]+1))
    assert all(count == (1 if q in legend else 2) for q, count in ownership.items()), 'every reachable geometric pin must have two owners'
    assert sum(len(obj['cells']) for obj in [root]+[v['object'] for v in placed.values()])-len(ownership.keys()-legend) == np.count_nonzero(out != a['background'])
    ignored = set(range(len(pieces)))-set(placed)
    color_witness = key_witness = key_box_witness = False
    for parent, child, port in a['edges']:
        for i in ignored:
            color_pose = _pose(pieces[i], port, ignore_color=True)
            key_pose = _pose(pieces[i], port, ignore_key=True)
            if color_pose is not None and pieces[i]['color'] != port['label']:
                # Same physical inlet/key, different payload: replacing the
                # correct graft by this false one changes a rendered cell.
                false = color_pose[0]
                correct = placed[child]['object']
                color_witness |= false['cells'] != correct['cells']
            if key_pose is not None and _pose(pieces[i], port) is None:
                false = key_pose[0]
                false_pins = _inlet_port(false)['pins']
                changed = false_pins != port['pins'] and false['cells'] != placed[child]['object']['cells']
                key_witness |= changed
                key_box_witness |= (changed and len(false_pins) == len(port['pins'])
                                    and _bbox(false_pins) == _bbox(port['pins']))
    branch = Counter(parent for parent, child, port in a['edges'])
    bodies = []
    for piece in pieces:
        t, l, b, r = piece['box']
        core = {(a-t, c-l) for a, c in piece['cells'] if t <= a <= b and l <= c <= r}
        bodies.append((b-t+1, r-l+1, tuple(sorted(core))))
    signatures = [_normal(port['pins']) for _, _, port in a['edges']]
    return dict(shape=tuple(np.asarray(pair['input']).shape), background=a['background'], pin=pin,
                inlet=root['inlet'], live=len(placed), ignored=len(ignored),
                depth=max((v['depth'] for v in placed.values()), default=0),
                sides={p['side'] for p in root['ports']}, turns={v['turns'] for v in placed.values()},
                relay_branch=any(parent is not None and count >= 2 for parent, count in branch.items()),
                color_witness=bool(color_witness), key_witness=bool(key_witness),
                key_box_witness=bool(key_box_witness),
                inherited_color=any(v['object']['color'] not in root['labels'] for v in placed.values()),
                keys=signatures, bodies=bodies,
                reuse=len(placed)-len({v['object']['color'] for v in placed.values()}),
                root_size=root['box'][2]-root['box'][0]+1,
                edge_touch=any(q[0] in (0, out.shape[0]-1) or q[1] in (0, out.shape[1]-1) for q in zip(*np.nonzero(out != a['background']))))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            key = inp.shape, inp.tobytes()
            assert key not in seen, 'duplicate input'
            seen.add(key)
            records[split].append(_features(pair))
    train, test = records['train'], records['test']
    assert len({(a['background'], a['pin'], a['inlet']) for a in train+test}) == 1
    assert set.union(*(a['sides'] for a in train)) == set(range(4)), 'all root directions need demonstration'
    assert set.union(*(a['turns'] for a in train)) == set(range(4)), 'all quarter turns need demonstration'
    assert any(a['color_witness'] for a in train), 'color must reject a geometrically valid decoy'
    assert any(a['key_witness'] for a in train), 'geometry must reject a same-color decoy'
    assert any(a['key_box_witness'] for a in train), 'full geometry must distinguish equal-area, equal-box keys'
    assert any(a['relay_branch'] for a in train), 'relay branching needs evidence'
    assert any(a['inherited_color'] for a in train), 'flat filtering by hub colors must fail'
    assert {len(key) for a in train for key in a['keys']} == {2, 3}, 'both pin counts need demonstration'
    cores = [body for a in train for body in a['bodies']]
    assert any(len(cells) == h*w for h, w, cells in cores), 'demonstrate a solid core'
    assert any(len(cells) == 2*(h+w)-4 for h, w, cells in cores), 'demonstrate a hollow core'
    assert any(2*(h+w)-4 < len(cells) < h*w for h, w, cells in cores), 'demonstrate partial interior occupancy'
    assert all(a['depth'] <= 2 for a in train)
    assert all(a['depth'] == 3 for a in test), 'test extends propagation depth, using the same rule'


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ('input', 'output')) for s in ('train', 'test'))


def self_test():
    """Bounded semantic checks and measured structural variation on 42 seeds."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records = [_features(pair) for p in puzzles for split in ('train', 'test') for pair in p[split]]
    assert len({tuple((s, tuple((q['input'].shape, q['input'].tobytes()) for q in p[s]))
                      for s in ('train', 'test')) for p in puzzles}) == 42
    assert len({a['shape'] for a in records}) >= 25
    assert len({a['background'] for a in records}) >= 7
    assert len({a['pin'] for a in records}) >= 7
    assert len({a['inlet'] for a in records}) >= 7
    assert len({body for a in records for body in a['bodies']}) >= 100
    assert len({key for a in records for key in a['keys']}) >= 12
    assert {len(key) for a in records for key in a['keys']} == {2, 3}
    assert {a['live'] for a in records} >= {1, 2, 3, 4, 5}
    assert {a['ignored'] for a in records} >= {0, 1, 2}
    assert {a['root_size'] for a in records} == {7, 9}
    assert any(a['reuse'] > 0 for a in records)
    assert any(a['edge_touch'] for a in records)
    assert _equal(puzzles[0], generate_puzzle(0))
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain-1)+1):
            validate_generated_puzzle(generate_puzzle(180+ntrain*5+ntest, ntrain, ntest))
    for args in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, *args)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    damaged = copy.deepcopy(puzzles[0])
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged answer accepted')
    # Color renaming commutes with the transformation, including all role hues.
    permutation = np.array([4, 0, 7, 2, 8, 5, 9, 1, 6, 3], dtype=np.int8)
    p = puzzles[0]['test'][0]
    assert np.array_equal(solve(permutation[p['input']]), permutation[p['output']])
    # Exercise compatible inputs beyond the sampler's 2-to-3-depth schedule:
    # all four hub sides, a depth-four tree, and an entire unreachable relay.
    rng = np.random.default_rng(7128)
    for links, depth in (([(0, s) for s in range(4)], 1),
                         ([(0, 1), (1, 2), (2, 3), (3, 2)], 4)):
        config = dict(links=links, turns={})
        for _ in range(30):
            objects = _layout(rng, config, (1, 2, 3, 4), list(range(1, 8)), 8, 9)
            fixture = _scatter(rng, objects[0], objects[1:], len(links), 0, config)
            if fixture is None:
                continue
            inp, expected = fixture
            try:
                predicted, analysis = _reconstruct(inp)
            except AssertionError:
                continue
            if np.array_equal(predicted, expected):
                break
        else:
            raise AssertionError('could not construct an extended valid fixture')
        assert max(v['depth'] for v in analysis['placed'].values()) == depth
    config = dict(links=[(0, 1)], turns={})
    objects = _layout(rng, config, (1, 2, 3, 4), list(range(1, 8)), 8, 9)
    dead_parent = _decoy(rng, objects[1], [5], 8, 9, 'color')
    attached = None
    for s in range(4):
        for _ in range(20):
            attached = _try_child(rng, [dead_parent], 0, s, 6, 8, 9)
            if attached is not None:
                break
        if attached is not None:
            break
    assert attached is not None
    dead_parent, dead_child = attached
    fixture = _scatter(rng, objects[0], [objects[1], dead_parent, dead_child], 1, 0, config)
    assert fixture is not None
    inp, expected = fixture
    predicted, analysis = _reconstruct(inp)
    assert np.array_equal(predicted, expected) and len(analysis['placed']) == 1
    # Reject a local color-and-key tie rather than using scatter proximity.
    fixture = _scatter(rng, objects[0], [objects[1], copy.deepcopy(objects[1])], 1, 0, config)
    assert fixture is not None
    try:
        solve(fixture[0])
    except AssertionError:
        pass
    else:
        raise AssertionError('two identical inlet candidates must not be resolved by order')
    return dict(seeds=42, pairs=len(records), shapes=len({a['shape'] for a in records}),
                core_masks=len({body for a in records for body in a['bodies']}),
                joint_keys=len({key for a in records for key in a['keys']}))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
