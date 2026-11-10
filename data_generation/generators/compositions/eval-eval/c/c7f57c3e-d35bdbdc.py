"""Signed relays: simultaneous color lookup composes geometric switch states.

All inference uses the current input. No parent code, files, seed or examples are
consulted by solve(). Diagnostics are in-memory and bounded; nothing is saved.
"""
from __future__ import annotations

import hashlib
import heapq
import json
from collections import Counter

import numpy as np


D4 = ((-1, 0), (0, -1), (0, 1), (1, 0))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
RING = {(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c}
KEY = {(-1, -2), (0, -2), (1, -2), (1, -3)}
HOST_SEEDS = (0, 539304437, 410176220, 391755134, 819018810, 608120139,
              8194395, 125037383, 596701545, 982396361, 822100533,
              416909376, 710046966, 383995899, 943057587, 994168445,
              633478600, 217870418, 402558281, 196179215, 968523826,
              301346393, 937923864, 853616339, 415129252, 329102870,
              869701701, 189989646, 260091070, 902500909, 212965146,
              39520575, 440662400, 220992866, 96861632, 767101831,
              443813313, 326309510, 219349746, 62309813, 516721807, 606697050)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid colors")
    return a.astype(np.int8, copy=True)


def _counts(rng, train, test):
    for name, value, low, high in (("num_train", train, 2, 10), ("num_test", test, 1, 4)):
        if value is not None:
            _require(not isinstance(value, (bool, np.bool_)) and
                     isinstance(value, (int, np.integer)) and low <= value <= high,
                     name + " is outside its legal integer range")
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(test + 1, 11))
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    _require(train > test, "num_train must be greater than num_test")
    return int(train), int(test)


def _components(cells, steps=D4):
    todo, parts = set(cells), []
    while todo:
        first = min(todo)
        todo.remove(first)
        part, stack = {first}, [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in todo:
                    todo.remove(q)
                    part.add(q)
                    stack.append(q)
        parts.append(part)
    return parts


def _graph(cells):
    # A diagonal beside an orthogonal corner is not an additional branch.
    return {(r, c): {(r + dr, c + dc) for dr, dc in D8
                     if (r + dr, c + dc) in cells and not
                     (dr and dc and ((r + dr, c) in cells or (r, c + dc) in cells))}
            for r, c in cells}


def _transform(a, rotation, reflected):
    return np.rot90(np.fliplr(a) if reflected else a, rotation).copy()


def _inverse(a, rotation, reflected):
    b = np.rot90(a, -rotation)
    return (np.fliplr(b) if reflected else b).copy()


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _primitive(crop):
    lengths = []
    for a in (crop, crop.T):
        for row in a:
            breaks = np.r_[0, np.flatnonzero(row[1:] != row[:-1]) + 1, len(row)]
            lengths.extend(map(int, np.diff(breaks)))
    s = int(np.gcd.reduce(lengths))
    small = crop[::s, ::s]
    _require(np.array_equal(np.repeat(np.repeat(small, s, 0), s, 1), crop), "nonuniform scale")
    return s, small


def _read_glyph(cells, clean, bg, line):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    r1, c1 = max(r for r, c in cells), max(c for r, c in cells)
    s, small = _primitive(clean[r0:r1 + 1, c0:c1 + 1])
    _require(small.shape[1] == 5 and 4 <= small.shape[0] <= 5, "not a keyed glyph")
    points = {(int(r), int(c)): int(small[r, c]) for r, c in zip(*np.nonzero(small != bg))}
    candidates = []
    for cy in range(1, small.shape[0] - 1):
        def at(p):
            return points.get((cy + p[0], 3 + p[1]), bg)
        body, value, carrier = at((-1, 0)), at((0, 0)), at((0, -2))
        if bg in (body, value, carrier) or line in (body, value, carrier):
            continue
        if carrier in (body, value) or any(at(p) != body for p in RING) or any(at(p) != carrier for p in KEY):
            continue
        relative = {(r - cy, c - 3): color for (r, c), color in points.items()}
        core = RING | KEY | {(0, 0)}
        if not core <= relative.keys():
            continue
        decoration = {p: color for p, color in relative.items() if p not in core}
        if not decoration or any(c in (body, value, carrier, bg, line) for c in decoration.values()):
            continue
        if all(2 <= r <= 3 and -1 <= c <= 1 for r, c in decoration):
            bit = 0  # Same side as the foot of the key.
        elif all(-3 <= r <= -2 and -1 <= c <= 1 for r, c in decoration):
            bit = 1
        else:
            continue
        if len(_components(set(decoration) | RING)) != 1:
            continue
        anchor = r0 + cy * s, c0 + 3 * s
        candidates.append(dict(anchor=anchor, scale=s, body=body, value=value,
                               carrier=carrier, bit=bit, decoration=decoration, cells=cells,
                               port=(anchor[0] + s // 2, anchor[1] + 2 * s),
                               port_strip={(anchor[0] + dy, anchor[1] + 2 * s) for dy in range(s)}))
    _require(len(candidates) == 1, "glyph orientation or center is ambiguous")
    return candidates[0]


def _read_canonical(a, bg, line):
    clean = np.where(a == line, bg, a)
    groups = _components(zip(*np.nonzero(clean != bg)))
    _require(4 <= len(groups) <= 8, "glyph count outside family")
    nodes = [_read_glyph(cells, clean, bg, line) for cells in groups]
    carriers = {n['carrier'] for n in nodes}
    _require(len(carriers) == 1, "inconsistent keys")
    templates = {}
    relation = {}
    for n in nodes:
        sig = tuple(sorted((r, c, color) for (r, c), color in n['decoration'].items()))
        _require(n['bit'] not in templates or templates[n['bit']] == sig, "inconsistent state template")
        templates[n['bit']] = sig
        entry = n['value'], n['bit']
        _require(n['body'] not in relation or relation[n['body']] == entry, "conflicting signed donors")
        relation[n['body']] = entry
    _require(set(templates) == {0, 1}, "both state templates must be visible")
    accents = {color for t in templates.values() for r, c, color in t}
    identities = {n[k] for n in nodes for k in ('body', 'value')}
    _require(not accents & (identities | carriers | {bg, line}), "overlapping palette roles")
    _require(1 <= len(accents) <= 2, "one or two decoration colors required")
    _require({n['bit'] for n in nodes if n['scale'] == 1} == {0, 1}, "unit references for both states required")
    union = RING | KEY | {(0, 0)} | {(r, c) for t in templates.values() for r, c, color in t}
    ymin, ymax = min(r for r, c in union), max(r for r, c in union)
    reserved = set()
    for n in nodes:
        y, x = n['anchor']
        s = n['scale']
        rect = {(r, c) for r in range(y + ymin * s, y + (ymax + 1) * s)
                for c in range(x - 3 * s, x + 2 * s)}
        _require(all(0 <= r < a.shape[0] and 0 <= c < a.shape[1] for r, c in rect), "replacement clipped")
        _require(not reserved & rect, "overlapping replacement regions")
        _require(all(a[p] != line for p in rect), "line crosses replacement region")
        reserved.update(rect)
    line_cells = set(zip(*np.nonzero(a == line)))
    graph = _graph(line_cells)
    _require(len(graph) >= 3 and all(len(v) in (1, 2) for v in graph.values()), "line must be open and unbranched")
    ends = {p for p, adj in graph.items() if len(adj) == 1}
    parts = _components(line_cells, D8)
    _require(1 <= len(parts) <= 2 and all(len(p & ends) == 2 for p in parts), "invalid line components")
    _require(all(sum(bool(n['port_strip'] & p & ends) for n in nodes) >= 1 for p in parts), "each line must select a glyph")
    for i, n in enumerate(nodes):
        _require(len(n['port_strip'] & ends) <= 1, "two endpoints on one glyph")
        n['eligible'] = bool(n['port_strip'] & ends)
        n['interior'] = bool(n['port_strip'] & graph.keys()) and not n['eligible']
        n['donors'] = [j for j, d in enumerate(nodes) if j != i and d['body'] == n['value']]
        n['retained'] = n['eligible'] and bool(n['donors'])
        if n['donors']:
            d = nodes[n['donors'][0]]
            n['new_value'], n['new_bit'] = d['value'], n['bit'] ^ d['bit']
    return dict(grid=a, bg=bg, line=line, nodes=nodes, templates=templates,
                graph=graph, parts=parts, union=union, carrier=next(iter(carriers)))


def _interpret(value):
    grid = _grid(value)
    colors, counts = np.unique(grid, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    _require(int(counts.max()) * 2 > grid.size, "background must occupy a majority")
    candidates = []
    # Test orientation-independent bounding proportions once per line candidate.
    # A keyed unit glyph is 4x5 or 5x5, possibly rotated and uniformly enlarged.
    possible_lines = []
    for line in colors:
        if int(line) == bg:
            continue
        groups = _components(zip(*np.nonzero((grid != bg) & (grid != line))))
        if not 4 <= len(groups) <= 8:
            continue
        good = True
        for cells in groups:
            h = max(r for r, c in cells) - min(r for r, c in cells) + 1
            w = max(c for r, c in cells) - min(c for r, c in cells) + 1
            if not ((h == w and h % 5 == 0) or (5 * h == 4 * w and w % 5 == 0)
                    or (4 * h == 5 * w and h % 5 == 0)):
                good = False
                break
        if good:
            possible_lines.append(int(line))
    for rotation in range(4):
        for reflected in (False, True):
            a = _transform(grid, rotation, reflected)
            for line in possible_lines:
                try:
                    info = _read_canonical(a, bg, int(line))
                except ValueError:
                    continue
                info['orientation'] = rotation, reflected
                candidates.append(info)
    _require(len(candidates) == 1, "input must have one keyed-relay interpretation")
    return candidates[0]


def _paint(grid, mapping, anchor, scale):
    for (r, c), color in mapping.items():
        y, x = anchor[0] + r * scale, anchor[1] + c * scale
        grid[y:y + scale, x:x + scale] = color


def _predict(info, mode='correct'):
    result = info['grid'].copy()
    for n in info['nodes']:
        retained = n['retained']
        if mode == 'ignore_endpoints':
            retained = bool(n['donors'])
        if not retained:
            for p in n['cells']:
                result[p] = info['bg']
            continue
        value, bit = n['new_value'], n['new_bit']
        if mode == 'ignore_A':
            bit = n['bit']
        elif mode == 'ignore_lookup':
            value, bit = n['value'], n['bit'] ^ 1
        _paint(result, {p: info['bg'] for p in info['union']}, n['anchor'], n['scale'])
        mapping = {p: n['body'] for p in RING}
        mapping.update({p: info['carrier'] for p in KEY})
        mapping[0, 0] = value
        mapping.update({(r, c): color for r, c, color in info['templates'][bit]})
        _paint(result, mapping, n['anchor'], n['scale'])
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read all original signed relations, then render their simultaneous composition."""
    info = _interpret(input_grid)
    return _inverse(_predict(info), *info['orientation'])


def _decorations(rng, accent_count):
    result = []
    for bit in (0, 1):
        sign = 1 if bit == 0 else -1
        root = int(rng.integers(-1, 2))
        points = {(2, root)}
        for _ in range(int(rng.integers(0, 5))):
            frontier = sorted({(r + dr, c + dc) for r, c in points for dr, dc in D4
                               if 2 <= r + dr <= 3 and -1 <= c + dc <= 1} - points)
            if frontier:
                points.add(frontier[int(rng.integers(len(frontier)))])
        order = list(sorted(points))
        rng.shuffle(order)
        result.append({(sign * r, c): j % accent_count for j, (r, c) in enumerate(order)})
    # If two colors were requested, make both visibly available.
    if accent_count == 2 and not any(1 in t.values() for t in result):
        root0 = next(iter(result[0]))[1]
        result[0][(2, root0 + 1 if root0 < 1 else root0 - 1)] = 1
    return result


def _separated(a, b):
    y, x, h, w = a
    yy, xx, hh, ww = b
    return y + h + 1 <= yy or yy + hh + 1 <= y or x + w + 1 <= xx or xx + ww + 1 <= x


def _pack(rng, logical_h, scales, spacious):
    sizes = [(logical_h * s, 5 * s) for s in scales]
    options = []
    for width in range(max(w for h, w in sizes), 30):
        for order in (sorted(range(len(sizes)), key=lambda i: sizes[i], reverse=True),
                      list(map(int, rng.permutation(len(sizes))))):
            y = x = shelf = 0
            rects = [None] * len(sizes)
            for i in order:
                h, w = sizes[i]
                if x and x + w > width:
                    y, x, shelf = y + shelf + 1, 0, 0
                rects[i] = y, x, h, w
                x += w + 1
                shelf = max(shelf, h)
            if max(y + h for y, x, h, w in rects) <= 30:
                options.append(rects)
    if not options:
        return None
    rects = options[int(rng.integers(len(options)))]
    h0 = max(y + h for y, x, h, w in rects)
    w0 = max(x + w for y, x, h, w in rects) + 1
    h = 30 if spacious else int(rng.integers(max(h0, 17), 31))
    w = 30 if spacious else int(rng.integers(max(w0, 17), 31))
    for _ in range(90):
        i = int(rng.integers(len(rects)))
        rh, rw = sizes[i]
        p = int(rng.integers(h - rh + 1)), int(rng.integers(w - rw)), rh, rw
        if all(i == j or _separated(p, q) for j, q in enumerate(rects)):
            rects[i] = p
    return (h, w), rects


def _halo(cells):
    return set(cells) | {(r + dr, c + dc) for r, c in cells for dr, dc in D8}


def _route(rng, shape, blocked, start, end):
    if start in blocked or end in blocked or start == end:
        return None
    cost = rng.uniform(1., 1.6, shape)
    distance, previous, heap = {start: 0.}, {}, [(0., start)]
    while heap:
        d, p = heapq.heappop(heap)
        if d != distance[p]:
            continue
        if p == end:
            path = [p]
            while p != start:
                p = previous[p]
                path.append(p)
            return path[::-1]
        for dr, dc in D8:
            q = p[0] + dr, p[1] + dc
            if 0 <= q[0] < shape[0] and 0 <= q[1] < shape[1] and q not in blocked:
                nd = d + float(cost[q])
                if nd < distance.get(q, float('inf')):
                    distance[q], previous[q] = nd, p
                    heapq.heappush(heap, (nd, q))
    return None


def _lines(rng, shape, rectangles, ports, selected, interior):
    occupied = {(r, c) for y, x, h, w in rectangles for r in range(y, y + h) for c in range(x, x + w)}
    order = list(selected)
    rng.shuffle(order)
    groups = [order[i:i + 2] for i in range(0, len(order), 2)]
    for attempt in range(16):
        gray = set()
        for group_index, group in enumerate(groups):
            start = ports[group[0]]
            if len(group) == 2:
                end = ports[group[1]]
            else:
                forbidden = occupied | _halo(gray)
                free = [(r, c) for r in range(shape[0]) for c in range(shape[1])
                        if (r, c) not in forbidden
                        and all(max(abs(r - y), abs(c - x)) >= 3 for y, x in ports)]
                if not free:
                    break
                end = free[int(rng.integers(len(free)))]
            blocked = occupied | _halo(gray)
            if interior is not None and group_index == 0:
                p = ports[interior]
                top, bottom = (p[0] - 1, p[1]), (p[0] + 1, p[1])
                if not (0 <= top[0] and bottom[0] < shape[0]):
                    break
                first = _route(rng, shape, blocked | (_halo({p, bottom}) - {top}), start, top)
                if first is None:
                    break
                second = _route(rng, shape, blocked | (_halo(set(first) | {p}) - {bottom}), bottom, end)
                path = None if second is None else first + [p] + second
            else:
                path = _route(rng, shape, blocked, start, end)
            if path is None or len(path) < 3:
                break
            gray.update(path)
        else:
            graph = _graph(gray)
            ends = {p for p, adj in graph.items() if len(adj) == 1}
            parts = _components(gray, D8)
            if (all(len(adj) in (1, 2) for adj in graph.values()) and len(parts) == len(groups)
                    and all(len(part & ends) == 2 for part in parts)
                    and {i for i, p in enumerate(ports) if p in ends} == set(selected)
                    and (interior is None or len(graph.get(ports[interior], ())) == 2)):
                return gray
    return None


def _relations(rng, mode):
    if mode == 'relay':
        # A->B and B->C both survive; E is a compatible interior-contact decoy.
        return [(0, 1, 1), (1, 2, 1), (2, 3, 0), (3, 1, 0), (4, 1, 0)], [0, 1], 4
    if mode == 'table':
        # Complete the XOR table; the final self-loop has no OTHER donor.
        return [(0, 1, 0), (1, 0, 1), (2, 3, 0), (3, 1, 0), (4, 4, 1)], [0, 1, 2, 4], None
    for _ in range(256):
        k, count = int(rng.integers(3, 6)), int(rng.integers(4, 9))
        bodies = list(map(int, rng.integers(k, size=count)))
        mapping = {b: (int(rng.integers(k)), int(rng.integers(2))) for b in sorted(set(bodies))}
        rel = [(b, *mapping[b]) for b in bodies]
        selected = list(map(int, rng.choice(count, int(rng.integers(1, min(4, count - 1) + 1)), replace=False)))
        donors = [[j for j, (bb, vv, ss) in enumerate(rel) if j != i and bb == v] for i, (b, v, s) in enumerate(rel)]
        if ({s for b, v, s in rel} == {0, 1}
                and any(donors[i] and rel[donors[i][0]][2] == 1 for i in selected)
                and any(donors[i] and rel[donors[i][0]][1] != rel[i][1] for i in selected)
                and any(donors[i] for i in range(count) if i not in selected)):
            return rel, selected, None
    return _relations(rng, 'relay')


def _make_pair(rng, mode, test, shared):
    for attempt in range(80):
        accent_count = int(rng.integers(1, 3))
        decs = _decorations(rng, accent_count)
        palette = list(map(int, rng.permutation(10))) if shared is None else list(shared)
        bg, line, carrier = palette[:3]
        accent = palette[3:3 + accent_count]
        ids = palette[3 + accent_count:]
        decs = [{p: accent[c] for p, c in t.items()} for t in decs]
        if mode == 'relay' and {(-r, c): color for (r, c), color in decs[0].items()} == decs[1]:
            continue  # Teach complete template exchange rather than reflection.
        rel, selected, interior = _relations(rng, mode)
        n = len(rel)
        if test and n > 6:
            continue
        scales = [1] * n
        # Both states retain a unit reference; choose an enlargement afterwards.
        enlarge = [i for i in selected if sum(s == rel[i][2] for b, v, s in rel) >= 2
                   and any(j != i and bb == rel[i][1] and ss == 1
                           for j, (bb, vv, ss) in enumerate(rel))]
        if not enlarge:
            continue
        i = int(rng.choice(enlarge))
        scales[i] = 3 if test else 2
        if mode == 'random' and n <= 6 and rng.random() < .25:
            j = int(rng.integers(n))
            if j != i and sum(scales[k] == 1 and rel[k][2] == rel[j][2] for k in range(n)) > 1:
                scales[j] = 2
        union = KEY | RING | {(0, 0)} | set(decs[0]) | set(decs[1])
        ymin, ymax = min(y for y, x in union), max(y for y, x in union)
        packed = _pack(rng, ymax - ymin + 1, scales, attempt > 12 or rng.random() < .3)
        if packed is None:
            continue
        shape, rectangles = packed
        anchors = [(y - ymin * s, x + 3 * s) for (y, x, h, w), s in zip(rectangles, scales)]
        ports = [(y + s // 2, x + 2 * s) for (y, x), s in zip(anchors, scales)]
        gray = _lines(rng, shape, rectangles, ports, selected, interior)
        if gray is None:
            continue
        inp, out = np.full(shape, bg, dtype=np.int8), np.full(shape, bg, dtype=np.int8)
        for p in gray:
            inp[p] = out[p] = line
        for i, ((b, v, bit), anchor, s) in enumerate(zip(rel, anchors, scales)):
            core = {p: ids[b] for p in RING}
            core.update({p: carrier for p in KEY})
            core[0, 0] = ids[v]
            _paint(inp, core | decs[bit], anchor, s)
            donors = [j for j, (bb, vv, ss) in enumerate(rel) if j != i and bb == v]
            if i in selected and donors:
                d = rel[donors[0]]
                core[0, 0] = ids[d[1]]
                _paint(out, core | decs[bit ^ d[2]], anchor, s)
        if np.count_nonzero(inp == bg) * 2 <= inp.size:
            continue
        rotation, reflected = int(rng.integers(4)), bool(rng.integers(2))
        inp, out = _transform(inp, rotation, reflected), _transform(out, rotation, reflected)
        # Reject ambiguity using pixels; construction's label never guides parsing.
        try:
            info = _interpret(inp)
            if not np.array_equal(_inverse(_predict(info), *info['orientation']), out):
                continue
        except ValueError:
            continue
        return {'input': inp, 'output': out}
    raise RuntimeError("bounded layout construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)), "seed must be an integer")
    rng = np.random.default_rng(seed)
    nt, ns = _counts(rng, num_train, num_test)
    shared = list(map(int, rng.permutation(10))) if rng.random() < .25 else None
    modes = ['relay', 'table'] + ['random'] * (nt - 2)
    rng.shuffle(modes)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ns)):
        for i in range(count):
            for _ in range(8):
                pair = _make_pair(rng, modes[i] if split == 'train' else 'random', split == 'test', shared)
                k = _key(pair['input'])
                if k not in seen:
                    seen.add(k)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("unique input construction exhausted")
    return puzzle


def _features(info):
    nodes = info['nodes']
    kept = [n for n in nodes if n['retained']]
    return dict(nodes=len(nodes), lines=len(info['parts']), shape=info['grid'].shape,
                scales=tuple(sorted(n['scale'] for n in nodes)),
                templates=tuple(info['templates'][b] for b in (0, 1)),
                geometry=tuple(tuple((r, c) for r, c, color in info['templates'][b]) for b in (0, 1)),
                nonmirror={(-r, c, color) for r, c, color in info['templates'][0]} != set(info['templates'][1]),
                roles=(info['bg'], info['line'], info['carrier']),
                survivors=len(kept), failed=sum(n['eligible'] and not n['donors'] for n in nodes),
                interior=sum(n['interior'] and bool(n['donors']) for n in nodes),
                repeated=len(nodes) - len({n['body'] for n in nodes}),
                solid=sum(n['body'] == n['new_value'] for n in kept),
                unchanged=sum(n['value'] == n['new_value'] and n['bit'] == n['new_bit'] for n in kept),
                retained_donor=sum(any(nodes[j]['retained'] for j in n['donors']) for n in kept),
                nonrecursive=sum(any(nodes[j]['retained'] and
                                     (nodes[j]['new_value'], nodes[j]['new_bit']) != (nodes[j]['value'], nodes[j]['bit'])
                                     for j in n['donors']) for n in kept),
                switched_scales={n['scale'] for n in kept if n['bit'] != n['new_bit']},
                cross_scale=sum(any(nodes[j]['scale'] != n['scale'] for j in n['donors']) for n in kept),
                truth={(n['bit'], nodes[n['donors'][0]]['bit'], n['new_bit']) for n in kept})


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "invalid pair counts")
    seen, infos = set(), {'train': [], 'test': []}
    for split in puzzle:
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape and _key(inp) not in seen, "shape mismatch or duplicate input")
            seen.add(_key(inp))
            info = _interpret(inp)
            expected = _predict(info)
            actual = _transform(out, *info['orientation'])
            _require(np.array_equal(expected, actual), "incorrect signed relay output")
            _require(np.array_equal(info['grid'] == info['line'], actual == info['line']), "changed line")
            _require(not np.array_equal(expected, _predict(info, 'ignore_A')), "state exchange has no effect")
            _require(not np.array_equal(expected, _predict(info, 'ignore_endpoints')), "endpoint selection has no effect")
            _require(not np.array_equal(expected, _predict(info, 'ignore_lookup')), "lookup has no effect")
            infos[split].append(_features(info))
    train = infos['train']
    truth = set().union(*(f['truth'] for f in train))
    _require(truth == {(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)}, "training must teach the complete parity table")
    for feature in ('interior', 'nonrecursive', 'failed', 'solid'):
        _require(any(f[feature] for f in train), "training omits " + feature)
    _require(any(f['nonmirror'] for f in train), "training must distinguish substitution from reflection")
    _require(max(f['lines'] for f in train) == 2, "multiple lines not demonstrated")
    _require(all(2 in f['switched_scales'] for f in train), "scale transfer not demonstrated")
    _require(all(3 in f['switched_scales'] for f in infos['test']), "test must extend the demonstrated scale")


def self_test():
    """42 seeds, one puzzle in memory at a time; no files or execution traces."""
    shapes, schedules, templates, palettes, counts, fingerprints = set(), set(), set(), set(), set(), set()
    geometries = set()
    totals, truth = Counter(), set()
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in p[split]:
                info = _interpret(pair['input'])
                f = _features(info)
                assert np.array_equal(solve(pair['input'].copy()), pair['output'])
                shapes.add(f['shape'])
                schedules.add(f['scales'])
                templates.add(f['templates'])
                geometries.add(f['geometry'])
                palettes.add(f['roles'])
                counts.add(f['nodes'])
                truth |= f['truth']
                for feature in ('interior', 'retained_donor', 'nonrecursive', 'solid', 'failed', 'unchanged', 'repeated', 'cross_scale'):
                    totals[feature] += bool(f[feature])
                totals['scale3'] += 3 in f['scales']
                digest.update(pair['input'].tobytes())
                digest.update(pair['output'].tobytes())
        fingerprints.add(digest.digest())
    assert len(fingerprints) == 42 and len(shapes) >= 30 and len(templates) >= 100
    assert len(geometries) >= 40
    assert len(schedules) >= 10 and len(palettes) >= 30 and counts >= {4, 5, 6, 7, 8}
    assert truth == {(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)}
    assert all(totals[f] >= 42 for f in ('interior', 'retained_donor', 'nonrecursive', 'solid', 'failed'))
    assert totals['repeated'] >= 10 and totals['unchanged'] >= 1 and totals['scale3'] == 42
    # Count boundaries and deterministic regeneration; the host checks all legal counts too.
    for a, b in ((2, 1), (10, 4), (None, 4), (2, None)):
        p = generate_puzzle(123, a, b)
        validate_generated_puzzle(p)
        q = generate_puzzle(123, a, b)
        assert all(_key(x[r]) == _key(y[r]) for s in p for x, y in zip(p[s], q[s]) for r in x)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.5, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Reject mutations of unchanged/changed output cells; also support list grids.
    p = generate_puzzle(0, 2, 1)
    line = _interpret(p['test'][0]['input'])['line']
    for point in (tuple(np.argwhere(p['test'][0]['input'] == line)[0]),
                  tuple(np.argwhere(p['test'][0]['input'] != p['test'][0]['output'])[0])):
        bad = {s: [{r: g.copy() for r, g in pair.items()} for pair in p[s]] for s in p}
        g = bad['test'][0]['output']
        g[point] = (int(g[point]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("damaged output accepted")
    info = _interpret(p['train'][0]['input'])
    enlarged = next(n for n in info['nodes'] if n['scale'] == 2)
    broken = info['grid'].copy()
    y, x = enlarged['anchor']
    broken[y, x - 2 * enlarged['scale']] = info['bg']
    try:
        solve(_inverse(broken, *info['orientation']))
    except ValueError:
        pass
    else:
        raise AssertionError("nonuniform pixel enlargement accepted")
    lists = {s: [{r: g.tolist() for r, g in pair.items()} for pair in p[s]] for s in p}
    validate_generated_puzzle(lists)
    return {'seeds': 42, 'pairs': 210, 'shapes': len(shapes), 'scale_schedules': len(schedules),
            'state_vocabularies': len(templates), 'role_assignments': len(palettes),
            'decoration_geometry_pairs': len(geometries),
            'node_counts': sorted(counts), 'features': dict(totals),
            'semantic_verification': 'generator-owned checks only'}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
