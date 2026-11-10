"""Paint pulses flip scaffold-registered switches before selecting an outlet.

Only the standard library and NumPy are used.  Construction uses an abstract
forest and a renderer; solving reconstructs components and templates from pixels.
No construction metadata or training answers are available to solve().
"""

from collections import deque
from math import gcd
import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _components(cells, steps=D4):
    left = set(cells)
    result = []
    while left:
        p = min(left)
        left.remove(p)
        group, todo = {p}, [p]
        for r, c in todo:
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in left:
                    left.remove(q)
                    group.add(q)
                    todo.append(q)
        result.append(group)
    return result


def _neighbors(cells, steps=D4):
    return {(r + dr, c + dc) for r, c in cells for dr, dc in steps} - set(cells)


def _bounds(cells):
    return min(r for r, c in cells), max(r for r, c in cells), min(c for r, c in cells), max(c for r, c in cells)


def _shift(cells, dr, dc):
    return {(r + dr, c + dc) for r, c in cells}


def _key(a):
    return a.shape, a.tobytes()


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int64, copy=True)


def _primitive(a):
    s = 0
    for array in (a, a.T):
        for row in array:
            breaks = [0] + (np.flatnonzero(row[1:] != row[:-1]) + 1).tolist() + [len(row)]
            for x, y in zip(breaks, breaks[1:]):
                s = gcd(s, y - x)
    b = a[::s, ::s]
    assert np.array_equal(a, np.repeat(np.repeat(b, s, 0), s, 1))
    return s, b


def _signature(mapping):
    return tuple(sorted((r, c, int(v)) for (r, c), v in mapping.items()))


def _expand(mapping, anchor, scale):
    ar, ac = anchor
    return {(ar + r * scale + dr, ac + c * scale + dc): color
            for (r, c), color in mapping.items()
            for dr in range(scale) for dc in range(scale)}


def _read_input(a):
    """Recognize cups, reservoirs, scaled states, and their three wire ports."""
    a = _grid(a)
    h, w = a.shape
    canvas = {(r, c) for r in range(h) for c in range(w)}
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    assert counts[bg] * 2 > a.size, "a majority background is required"
    fg = {p for p in canvas if a[p] != bg}
    # A paint cell sits in one corner of an otherwise monochrome 2x2 cup.
    cups = []
    for p in sorted(fg):
        ns = _neighbors({p}, D8) & fg
        if len(ns) == 3 and len({int(a[q]) for q in ns}) == 1:
            wire = int(a[next(iter(ns))])
            r0, r1, c0, c1 = _bounds(ns | {p})
            if wire != a[p] and r1 - r0 == c1 - c0 == 1:
                cups.append((p, int(a[p]), wire))
    assert cups and len({c[2] for c in cups}) == 1, "unambiguous source cups"
    wire = cups[0][2]
    sources = {p for p, _, _ in cups}
    assert len({int(a[p]) for p in sources} & {bg, wire}) == 0
    wires = _components({p for p in fg if a[p] == wire})
    labels = {p: i for i, comp in enumerate(wires) for p in comp}
    supplies = {}
    for p, paint, _ in cups:
        ids = {labels[q] for q in _neighbors({p}, D8) if q in labels}
        assert len(ids) == 1
        node = ids.pop()
        assert node not in supplies or supplies[node] == paint
        supplies[node] = paint

    # A reservoir is a non-border background component with a monochrome rim.
    reservoirs, rims = [], set()
    background_groups = _components(canvas - fg)
    for cells in background_groups:
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells):
            continue
        edge = _neighbors(cells)
        colors = {int(a[q]) for q in edge}
        assert len(colors) == 1 and wire not in colors, "unintended enclosure"
        color = colors.pop()
        assert color != bg and edge <= fg
        rim = {q for q in _neighbors(cells, D8) if q in canvas and a[q] == color}
        # Only the rim's own same-colored eight-connected component is used.
        same = next(g for g in _components({p for p in fg if a[p] == color}, D8) if g & rim)
        assert same == rim, "reservoir boundary has an ambiguous extension"
        ids = {labels[p] for p in _neighbors(rim) if p in labels}
        assert len(ids) == 1, "every reservoir has one attached wire"
        reservoirs.append(dict(inner=cells, rim=rim, node=ids.pop(), color=color))
        rims |= rim
    assert reservoirs

    # Removing routing and reservoirs leaves the multicolor switch objects.
    objects = []
    for cells in _components(fg - set(labels) - sources - rims, D8):
        r0, r1, c0, c1 = _bounds(cells)
        crop = a[r0:r1 + 1, c0:c1 + 1].copy()
        assert {p for p in cells} == {(r, c) for r in range(r0, r1 + 1)
                                    for c in range(c0, c1 + 1)
                                    if a[r, c] not in (bg, wire)} - sources - rims
        # Wire ports may lie inside an object's bounding rectangle. They are
        # external routing, never part of its normalized colored template.
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if (r, c) not in cells:
                    crop[r - r0, c - c0] = bg
        scale, small = _primitive(crop)
        raw = {(int(r), int(c)): int(small[r, c])
               for r, c in zip(*np.where(small != bg))}
        objects.append(dict(box=(r0, c0), scale=scale, raw=_signature(raw), cells=cells))
    assert 2 <= len(objects) <= 8
    signatures = sorted({o['raw'] for o in objects})
    assert len(signatures) == 2, "exactly two complete local switch states"
    minimum = min(o['scale'] for o in objects)
    assert {o['raw'] for o in objects if o['scale'] == minimum} == set(signatures)
    maps = [{(r, c): color for r, c, color in sig} for sig in signatures]
    # A whole-color invariant scaffold determines registration, not the box.
    shared_colors = set(maps[0].values()) & set(maps[1].values())
    candidates = []
    for color in sorted(shared_colors):
        pa = {p for p, v in maps[0].items() if v == color}
        pb = {p for p, v in maps[1].items() if v == color}
        ay, _, ax, _ = _bounds(pa)
        by, _, bx, _ = _bounds(pb)
        shift = ay - by, ax - bx
        if pa == _shift(pb, *shift):
            moved = {(r + shift[0], c + shift[1]): v for (r, c), v in maps[1].items()}
            common = {p: v for p, v in maps[0].items() if moved.get(p) == v}
            if len(common) >= 4:
                candidates.append((len(common), shift, moved, common))
    assert candidates
    best = max(x[0] for x in candidates)
    candidates = {x[1]: x for x in candidates if x[0] == best}
    assert len(candidates) == 1, "unique scaffold registration"
    _, shift, moved, common = next(iter(candidates.values()))
    assert len(set(common.values())) == 1, "one invariant carrier color"
    oy, _, ox, _ = _bounds(common)
    templates = [{(r - oy, c - ox): v for (r, c), v in m.items()} for m in (maps[0], moved)]
    common = {(r - oy, c - ox): v for (r, c), v in common.items()}
    union = set(templates[0]) | set(templates[1])
    assert set(templates[0]) - set(templates[1]) and set(templates[1]) - set(templates[0])
    gate_infos = []
    claimed = set()
    for obj in objects:
        state, s = signatures.index(obj['raw']), obj['scale']
        dy, dx = shift if state else (0, 0)
        anchor = obj['box'][0] + (oy - dy) * s, obj['box'][1] + (ox - dx) * s
        expanded = [_expand(t, anchor, s) for t in templates]
        region = set(expanded[0]) | set(expanded[1])
        assert region <= canvas and not (region & claimed), "replacement collision/clipping"
        assert all(a[p] == v for p, v in expanded[state].items())
        assert all(a[p] == bg for p in region - set(expanded[state]))
        stable = set(_expand(common, anchor, s))
        inlet = {labels[p] for p in _neighbors(stable) if p in labels}
        assert len(inlet) == 1, "one wire touches the stable inlet"
        outlets = []
        for st in range(2):
            # The moving attachment, not a recoloring of the central accent,
            # selects an outlet. Both potential attachments are visible locally.
            attachment = set(expanded[st]) - set(expanded[1 - st])
            ids = {labels[p] for p in _neighbors(attachment) if p in labels}
            assert len(ids) == 1
            outlets.append(ids.pop())
        inlet = inlet.pop()
        assert len({inlet, *outlets}) == 3
        gate_infos.append(dict(anchor=anchor, scale=s, state=state, inlet=inlet,
                               outlets=outlets, expanded=expanded, region=region,
                               stable=stable))
        claimed |= region
    assert claimed.isdisjoint(rims | sources | set(labels))
    # All directed wire edges form a forest: no merging, cycles, or order issue.
    indegree = [0] * len(wires)
    adjacency = [[] for _ in wires]
    for i, g in enumerate(gate_infos):
        adjacency[g['inlet']].append(i)
        for node in g['outlets']:
            indegree[node] += 1
    assert max(indegree, default=0) <= 1
    assert all(indegree[node] == 0 for node in supplies)
    ready = deque(i for i, degree in enumerate(indegree) if degree == 0)
    degree = indegree[:]
    visited = 0
    while ready:
        node = ready.popleft()
        visited += 1
        for i in adjacency[node]:
            for out in gate_infos[i]['outlets']:
                degree[out] -= 1
                if not degree[out]:
                    ready.append(out)
    assert visited == len(wires), "directed cycles are excluded"
    return dict(grid=a, bg=bg, wire=wire, wires=wires, supplies=supplies,
                sources=sources, reservoirs=reservoirs, gates=gate_infos,
                templates=templates, adjacency=adjacency)


def _propagate(info, mode='flip'):
    """Pulse semantics; alternate modes are explicit shortcut ablations."""
    active = dict(info['supplies'])
    depths = {node: 0 for node in active}
    queue = deque(active)
    reached = set()
    while queue:
        node = queue.popleft()
        for i in info['adjacency'][node]:
            g = info['gates'][i]
            reached.add(i)
            states = [0, 1] if mode == 'both' else [g['state'] if mode == 'old' else 1 - g['state']]
            for st in states:
                out = g['outlets'][st]
                assert out not in active or active[out] == active[node]
                if out not in active:
                    active[out] = active[node]
                    depths[out] = depths[node] + 1
                    queue.append(out)
    return active, reached, depths


def _answer(info, mode='flip', flip_all=False, no_fill=False):
    out = info['grid'].copy()
    active, reached, depths = _propagate(info, mode)
    for i, g in enumerate(info['gates']):
        if i in reached or flip_all:
            for p in g['region']:
                out[p] = info['bg']
            for p, color in g['expanded'][1 - g['state']].items():
                out[p] = color
    if not no_fill:
        for res in info['reservoirs']:
            if res['node'] in active:
                for p in res['inner']:
                    out[p] = active[res['node']]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from the current input's visible ports, cups and templates."""
    return _answer(_read_input(input_grid))


def _ri(rng, low, high):
    return int(rng.integers(low, high + 1))


def _pick(rng, values):
    return values[_ri(rng, 0, len(values) - 1)]


def _design(rng, compact, colors):
    carrier, a0, a1, b0, b1 = colors
    h = 3 if compact else _pick(rng, [3, 3, 4, 5])
    w = 2 if compact else _ri(rng, 2, 4)
    mid = _ri(rng, 1, h - 2)
    core = {(r, w - 1): carrier for r in range(h)}
    core.update({(mid, c): carrier for c in range(w)})
    # Optional short twigs retain a tree-shaped carrier and its inlet/ports.
    if not compact and w >= 3 and rng.random() < .5:
        core[mid - 1, w - 2] = carrier
    templates, endpoints, directions = [], [], []
    for st in range(2):
        direction = _pick(rng, [(-1, 0), (0, 1)]) if st == 0 else _pick(rng, [(1, 0), (0, 1)])
        depth = 1 if compact else _ri(rng, 1, 2)
        root = (0 if st == 0 else h - 1, w - 1)
        arm = {(root[0] + direction[0] * d, root[1] + direction[1] * d)
               for d in range(1, depth + 1)}
        if not compact and direction[0] and rng.random() < .5:
            arm.add((root[0] + direction[0], root[1] - 1))
        color, accent = (a0, a1) if st == 0 else (b0, b1)
        template = dict(core)
        template.update({p: color for p in arm})
        if rng.random() < .5 and len(arm) > 1:
            template[min(arm)] = accent
        # Same logical central position in both states, recolored as part of
        # the complete template. The connector itself always stays fixed.
        template[mid, w] = accent
        templates.append(template)
        endpoints.append((root[0] + direction[0] * depth, root[1] + direction[1] * depth))
        directions.append(direction)
    return templates, (mid, 0), endpoints, directions


def _gate_module(templates, inlet, ends, dirs, scale, nodes, state):
    union = set(templates[0]) | set(templates[1])
    y0, y1, x0, x1 = _bounds(union)
    anchor = (-y0 * scale, -x0 * scale)
    maps = [_expand(t, anchor, scale) for t in templates]
    ports = []
    for endpoint, direction, node in zip([inlet] + ends, [(0, -1)] + dirs, nodes):
        y, x = anchor[0] + endpoint[0] * scale, anchor[1] + endpoint[1] * scale
        dy, dx = direction
        p = (y - 1, x + scale // 2) if dy == -1 else (y + scale, x + scale // 2) if dy == 1 else (
            (y + scale // 2, x - 1) if dx == -1 else (y + scale // 2, x + scale))
        ports.append((p, direction, node))
    return dict(kind='gate', shape=((y1 - y0 + 1) * scale, (x1 - x0 + 1) * scale),
                maps=maps, state=state, ports=ports, scale=scale)


def _rotate_point(p, k):
    r, c = p
    for _ in range(k):
        r, c = -c, r
    return r, c


def _source_module(rng, paint, wire, node):
    mapping = {(0, 0): paint, (0, 1): wire, (1, 0): wire, (1, 1): wire}
    k = _ri(rng, 0, 3)
    mapping = {_rotate_point(p, k): v for p, v in mapping.items()}
    y0, y1, x0, x1 = _bounds(mapping)
    mapping = {(r - y0, c - x0): v for (r, c), v in mapping.items()}
    p = _rotate_point((1, 2), k)
    direction = _rotate_point((0, 1), k)
    return dict(kind='source', shape=(y1 - y0 + 1, x1 - x0 + 1), mapping=mapping,
                ports=[((p[0] - y0, p[1] - x0), direction, node)], node=node, paint=paint)


def _loop_module(rng, rim, node):
    h, w = _ri(rng, 1, 2), _ri(rng, 1, 3)
    inner = {(r, c) for r in range(h) for c in range(w)}
    if h == w == 2 and rng.random() < .5:
        inner.remove(_pick(rng, sorted(inner)))
    if rng.random() < .12:
        inner = {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}
    boundary = _neighbors(inner)
    y0, y1, x0, x1 = _bounds(inner | boundary)
    inner = _shift(inner, -y0, -x0)
    boundary = _shift(boundary, -y0, -x0)
    direction = _pick(rng, D4)
    dy, dx = direction
    score = max(r * dy + c * dx for r, c in boundary)
    candidates = sorted(p for p in boundary if p[0] * dy + p[1] * dx == score)
    r, c = _pick(rng, candidates)
    p = r + dy, c + dx
    return dict(kind='loop', shape=(y1 - y0 + 1, x1 - x0 + 1),
                mapping={p: rim for p in boundary}, inner=inner, node=node,
                ports=[(p, direction, node)])


def _network(rng, split, index, extras='none'):
    # The first two lessons guarantee old/new routing, both toggle directions,
    # an inactive switch, and (lesson two) an active enlarged switch.
    depth = 2 if split == 'train' and index < 2 else _ri(rng, 3, 4) if split == 'test' else _ri(rng, 1, 3)
    secondary = extras != 'none' and depth <= 3 and (
        split == 'train' and index == 1 or not (split == 'train' and index == 0) and rng.random() < .35)
    gates, next_node = [], 1
    root_state = _ri(rng, 0, 1)
    selected, old_nodes = 0, []
    for i in range(depth):
        if split == 'train' and index == 0:
            state = (root_state + i) % 2
        elif split == 'train' and index == 1:
            state = root_state
        else:
            state = _ri(rng, 0, 1)
        outs = [next_node, next_node + 1]
        next_node += 2
        gates.append(dict(nodes=[selected] + outs, state=state, scale=1))
        old_nodes.append(outs[state])
        selected = outs[1 - state]
    # This switch is downstream of a rejected arm, not a detached reference.
    inlet = _pick(rng, old_nodes[:-1] if depth >= 3 else old_nodes)
    gates.append(dict(nodes=[inlet, next_node, next_node + 1], state=_ri(rng, 0, 1), scale=1))
    next_node += 2
    roots = [0]
    if secondary:
        if extras == 'sources':
            # A second cup can supply a separate switching subtree.
            inlet = next_node
            roots.append(inlet)
            next_node += 1
        else:
            # Or a live wire can fork to two switches.
            inlet = _pick(rng, [gates[i]['nodes'][0] for i in range(depth)])
        gates.append(dict(nodes=[inlet, next_node, next_node + 1], state=_ri(rng, 0, 1), scale=1))
        next_node += 2
    scaled = (split == 'train' and index == 1 or
              split == 'train' and index > 1 and rng.random() < .45 or
              split == 'test' and depth == 3 and rng.random() < .35)
    if scaled and depth >= 2:
        which = _ri(rng, 0, depth - 1)
        gates[which]['scale'] = 2
        # Both states remain present at the unit reference scale.
        gates[depth]['state'] = gates[which]['state']
    # The least-count demonstrations always contain both states at scale one.
    if not {g['state'] for g in gates if g['scale'] == 1} == {0, 1}:
        gates[-1]['state'] = 1 - gates[0]['state']
    inlets = {g['nodes'][0] for g in gates}
    terminals = [i for i in range(next_node) if i not in inlets]
    loop_nodes = terminals[:]
    if split == 'train' and index == 0:
        loop_nodes.append(0)
    elif rng.random() < .4:
        loop_nodes.append(_pick(rng, sorted(inlets)))
    # Several cups, reservoirs per wire, and branching are separate axes.
    if len(gates) <= 3 and rng.random() < .25:
        loop_nodes.append(_pick(rng, terminals))
    return gates, roots, loop_nodes, next_node


def _separate(a, b, gap=3):
    y, x, h, w = a
    r, c, hh, ww = b
    return y + h + gap <= r or r + hh + gap <= y or x + w + gap <= c or c + ww + gap <= x


def _place(rng, modules, shape):
    h, w = shape
    order = sorted(range(len(modules)), key=lambda i: np.prod(modules[i]['shape']), reverse=True)
    rects = [None] * len(modules)
    for i in order:
        mh, mw = modules[i]['shape']
        if mh + 4 > h or mw + 4 > w:
            return None
        candidates = []
        for _ in range(260):
            y, x = _ri(rng, 2, h - mh - 2), _ri(rng, 2, w - mw - 2)
            rect = y, x, mh, mw
            if all(other is None or _separate(rect, other) for other in rects):
                candidates.append(rect)
                if len(candidates) >= 8:
                    break
        if not candidates:
            return None
        rects[i] = _pick(rng, candidates)
    return rects


def _tree_layout(rng, modules, gates, roots, nnode):
    """Embed the actual forest, rather than routing arbitrary interleaved pins.

    Unequal subtree spans, multiscale gates, forks, side reservoirs and spare
    margins determine placement. Each branch keeps a clear approach channel.
    """
    items = [[] for _ in range(nnode)]
    for i, gate in enumerate(gates):
        items[gate['nodes'][0]].append(i)
    for i, mod in enumerate(modules):
        if mod['kind'] == 'loop':
            items[mod['node']].append(i)
    gap = 1
    cache = {}
    def measure(node):
        if node in cache:
            return cache[node]
        rows = []
        for i in items[node]:
            mod = modules[i]
            mh, mw = mod['shape']
            if mod['kind'] == 'gate':
                a, b = [measure(n) for n in gates[i]['nodes'][1:]]
                rows.append((max(mh, a[0] + gap + b[0]), mw + 3 + max(a[1], b[1])))
            else:
                rows.append((mh, mw))
        assert rows
        cache[node] = sum(t[0] for t in rows) + gap * (len(rows) - 1), max(t[1] for t in rows)
        return cache[node]
    root_sizes = [measure(node) for node in roots]
    height = sum(s[0] for s in root_sizes) + 2 * (len(roots) - 1)
    width = max(s[1] for s in root_sizes) + 2
    if height > 28 or width > 30:
        return None
    h = _ri(rng, max(22, height + 2), 30)
    w = _ri(rng, max(22, width), 30)
    rects = [None] * len(modules)
    def place_node(node, y, x):
        for i in items[node]:
            mod = modules[i]
            mh, mw = mod['shape']
            if mod['kind'] == 'gate':
                out0, out1 = gates[i]['nodes'][1:]
                a, b = measure(out0), measure(out1)
                span = max(mh, a[0] + gap + b[0])
                yy = y + (span - mh) // 2
                rects[i] = yy, x, mh, mw
                offset = (span - a[0] - gap - b[0]) // 2
                place_node(out0, y + offset, x + mw + 3)
                place_node(out1, y + offset + a[0] + gap, x + mw + 3)
                y += span + gap
            else:
                rects[i] = y, x, mh, mw
                # A reservoir faces its supplying branch. Its cavity shape,
                # extent and source-to-reservoir distance still vary.
                boundary = set(mod['mapping'])
                c = min(c for r, c in boundary)
                ps = sorted(p for p in boundary if p[1] == c)
                r, c = _pick(rng, ps)
                mod['ports'] = [((r, c - 1), (0, -1), node)]
                y += mh + gap
    y = _ri(rng, 1, h - height - 1)
    for node in roots:
        place_node(node, y, 2)
        y += measure(node)[0] + 2
    # Compact cups touch the root's input lead. They are placed by that port,
    # and do not consume an entire extra horizontal layer of the drawing.
    for i, mod in enumerate(modules):
        if mod['kind'] == 'source':
            gate_index = next(j for j, g in enumerate(gates) if g['nodes'][0] == mod['node'])
            gy, gx, _, _ = rects[gate_index]
            (pr, pc), _, _ = modules[gate_index]['ports'][0]
            py, px = gy + pr, gx + pc
            mod['mapping'] = {(0, 0): mod['paint'], (0, 1): mod['wire'],
                              (1, 0): mod['wire'], (1, 1): mod['wire']}
            mod['ports'] = [((1, 0), (1, 0), mod['node'])]
            mod['direct'] = False
            rects[i] = py - 1, px - 1, 2, 2
    return (h, w), rects


def _path(rng, start, goals, allowed, shape):
    todo = deque(sorted(start))
    prev = {p: None for p in start}
    directions = [D4[int(i)] for i in rng.permutation(4)]
    end = None
    while todo:
        r, c = todo.popleft()
        if (r, c) in goals:
            end = r, c
            break
        for dr, dc in directions:
            q = r + dr, c + dc
            if q in allowed and q not in prev:
                prev[q] = (r, c)
                todo.append(q)
    if end is None:
        return None
    path = set()
    while end is not None:
        path.add(end)
        end = prev[end]
    return path


def _render_scene(rng, modules, rects, shape, bg, wire, gates, roots, nnode):
    h, w = shape
    canvas = {(r, c) for r in range(h) for c in range(w)}
    blocked = set()
    terms = [set() for _ in range(nnode)]
    all_ports = {}
    inp = np.full(shape, bg, dtype=np.int64)
    for mod, (y, x, mh, mw) in zip(modules, rects):
        box = {(r, c) for r in range(y, y + mh) for c in range(x, x + mw)}
        blocked |= box | _neighbors(box, D8)
        mapping = mod['maps'][mod['state']] if mod['kind'] == 'gate' else mod['mapping']
        for (r, c), color in mapping.items():
            inp[y + r, x + c] = color
        for (r, c), direction, node in mod['ports']:
            p = y + r, x + c
            # A shorter arm can finish inside the union's bounding box.
            # Reserve its straight external lead as far as the outer halo.
            while y - 1 <= p[0] <= y + mh and x - 1 <= p[1] <= x + mw:
                if p not in canvas or (p in all_ports and all_ports[p] != node):
                    return None
                all_ports[p] = node
                terms[node].add(p)
                if mod.get('direct'):
                    break
                p = p[0] + direction[0], p[1] + direction[1]
    for p, node in all_ports.items():
        if any(q in all_ports and all_ports[q] != node for q in _neighbors({p})):
            return None
    wires = [set() for _ in range(nnode)]
    order = sorted(range(nnode), key=lambda i: -len(terms[i]))
    # Shuffle equal-priority nets so no particular spatial ordering is favored.
    rng.shuffle(order)
    for node in order:
        targets = sorted(terms[node])
        if not targets:
            return None
        other_ports = {p for p, owner in all_ports.items() if owner != node}
        existing = set().union(*wires)
        forbidden = blocked | existing | _neighbors(existing) | other_ports | _neighbors(other_ports)
        allowed = (canvas - forbidden) | terms[node]
        tree = {targets[0]}
        for goal in targets[1:]:
            path = _path(rng, tree, {goal}, allowed | tree, shape)
            if path is None:
                return None
            tree |= path
        wires[node] = tree
        for p in tree:
            if inp[p] not in (bg, wire):
                return None
            inp[p] = wire
    # Compute expected output from the abstract forest, before reading pixels.
    paint_at = {}
    for mod in modules:
        if mod['kind'] == 'source':
            paint_at[mod['node']] = mod['paint']
    changed = set()
    for i, gate in enumerate(gates):
        inlet, out0, out1 = gate['nodes']
        if inlet in paint_at:
            changed.add(i)
            paint_at[[out0, out1][1 - gate['state']]] = paint_at[inlet]
    out = inp.copy()
    gi = 0
    for mod, (y, x, mh, mw) in zip(modules, rects):
        if mod['kind'] == 'gate':
            if gi in changed:
                for r, c in set(mod['maps'][0]) | set(mod['maps'][1]):
                    out[y + r, x + c] = bg
                for (r, c), color in mod['maps'][1 - mod['state']].items():
                    out[y + r, x + c] = color
            gi += 1
        elif mod['kind'] == 'loop' and mod['node'] in paint_at:
            for r, c in mod['inner']:
                out[y + r, x + c] = paint_at[mod['node']]
    # Rotate/reflection is global; local copies retain their common orientation.
    k = _ri(rng, 0, 3)
    inp, out = np.rot90(inp, k), np.rot90(out, k)
    if rng.random() < .5:
        inp, out = np.fliplr(inp), np.fliplr(out)
    try:
        info = _read_input(inp)
        if len(info['gates']) != len(gates) or len(info['reservoirs']) != sum(m['kind'] == 'loop' for m in modules):
            return None
        if not np.array_equal(_answer(info), out):
            return None
    except (AssertionError, StopIteration):
        return None
    return dict(input=inp.copy(), output=out.copy())


def _make_pair(rng, split, index, extras='none'):
    gates, roots, loops, nnode = _network(rng, split, index, extras)
    compact = any(g['scale'] > 1 for g in gates) or len(gates) >= 4
    palette = [int(c) for c in rng.permutation(10)]
    bg, wire, carrier, rim, paint0, paint1, a0, a1, b0, b1 = palette
    # Endpoint and central accent may share a color; palette size varies.
    if rng.random() < .4:
        a1 = a0
    if rng.random() < .4:
        b1 = b0
    templates, inlet, ends, dirs = _design(rng, compact, [carrier, a0, a1, b0, b1])
    modules = [_gate_module(templates, inlet, ends, dirs, g['scale'], g['nodes'], g['state']) for g in gates]
    modules += [_source_module(rng, [paint0, paint1][i], wire, node) for i, node in enumerate(roots)]
    for mod in modules:
        if mod['kind'] == 'source':
            mod['wire'] = wire
    modules += [_loop_module(rng, rim, node) for node in loops]
    for _ in range(6):
        layout = _tree_layout(rng, modules, gates, roots, nnode)
        if layout is None:
            break
        shape, rects = layout
        pair = _render_scene(rng, modules, rects, shape, bg, wire, gates, roots, nnode)
        if pair is not None:
            return pair
    # A few bounded scattered proposals are useful for compact networks.
    for attempt in range(2 if len(gates) <= 3 else 0):
        low = 24 if len(gates) <= 3 and not any(g['scale'] > 1 for g in gates) else 27
        if attempt > 50:
            shape = (30, 30)
        else:
            shape = (_ri(rng, low, 30), _ri(rng, low, 30))
        rects = _place(rng, modules, shape)
        if rects is None:
            continue
        pair = _render_scene(rng, modules, rects, shape, bg, wire, gates, roots, nnode)
        if pair is not None:
            return pair
    return None


def _counts(num_train, num_test):
    for name, value, low, high in [('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    extras = _pick(rng, ['none', 'fork', 'sources'])
    result, seen = {'train': [], 'test': []}, set()
    for split, count in [('train', num_train), ('test', num_test)]:
        for i in range(count):
            for _ in range(40):
                pair = _make_pair(rng, split, i, extras)
                if pair is not None and _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    result[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded network packing exhausted')
    validate_generated_puzzle(result)
    return result


def _features(info):
    active, reached, depths = _propagate(info)
    filled = [res for res in info['reservoirs'] if res['node'] in active]
    return dict(shape=list(info['grid'].shape), background=info['bg'],
                palette=len(np.unique(info['grid'])), gates=len(info['gates']),
                reached=len(reached), inactive=len(info['gates']) - len(reached),
                depth=max(depths.values()), sources=len(info['supplies']),
                scales=sorted(g['scale'] for g in info['gates']),
                reservoirs=len(info['reservoirs']), filled=len(filled),
                empty=len(info['reservoirs']) - len(filled),
                areas=sorted(len(res['inner']) for res in info['reservoirs']),
                forks=sum(len(gs) > 1 for gs in info['adjacency']),
                carrier=sorted((r, c) for (r, c), color in info['templates'][0].items()
                               if info['templates'][1].get((r, c)) == color),
                both_directions=len({info['gates'][i]['state'] for i in reached}) == 2,
                repeated_state_step=any(info['gates'][j]['state'] == gate['state']
                    for i, gate in enumerate(info['gates']) if i in reached
                    for j in info['adjacency'][gate['outlets'][1 - gate['state']]] if j in reached))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, features = set(), {'train': [], 'test': []}
    lessons = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            info = _read_input(a)
            assert np.array_equal(_answer(info), b), 'pixel reconstruction disagrees with output'
            active, reached, _ = _propagate(info)
            f = _features(info)
            assert f['filled'] and f['empty'] and reached
            # All source cups, routing, rims, stable scaffold pixels and all
            # inactive switches are checked independently of answer equality.
            stable = set().union(*info['wires'], info['sources'],
                                 *(r['rim'] for r in info['reservoirs']),
                                 *(g['stable'] for g in info['gates']))
            for i, g in enumerate(info['gates']):
                if i not in reached:
                    stable |= g['region']
            assert all(a[p] == b[p] for p in stable)
            for res in info['reservoirs']:
                values = {int(b[p]) for p in res['inner']}
                assert values == {active.get(res['node'], info['bg'])}
            assert not np.array_equal(_answer(info, mode='old'), b)
            assert not np.array_equal(_answer(info, mode='both'), b)
            assert not np.array_equal(_answer(info, no_fill=True), b)
            features[split].append(f)
            if split == 'train':
                lessons.append(info)
    assert any(f['inactive'] and f['both_directions'] and f['depth'] >= 2 for f in features['train'])
    assert features['train'][0]['both_directions'] and features['train'][1]['repeated_state_step']
    assert any(g['scale'] > 1 for info in lessons for i, g in enumerate(info['gates']) if i in _propagate(info)[1])
    assert any(not np.array_equal(_answer(info, flip_all=True), _answer(info)) for info in lessons)
    assert all(f['depth'] >= 3 for f in features['test'])
    assert max(f['sources'] for f in features['test']) <= max(f['sources'] for f in features['train'])
    assert not any(f['forks'] for f in features['test']) or any(f['forks'] for f in features['train'])


def self_test():
    """Bounded geometry, counterfactual, count, corruption and 42-seed audit."""
    shapes, carriers, scales, reservoirs, palettes, backgrounds, depths = set(), set(), set(), set(), set(), set(), set()
    fingerprints, rows = set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(tuple(_key(p[r]) for s in ('train', 'test') for p in puzzle[s] for r in ('input', 'output')))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _features(_read_input(pair['input']))
                rows.append(f)
                shapes.add(tuple(f['shape']))
                carriers.add(tuple(map(tuple, f['carrier'])))
                scales.add(tuple(f['scales']))
                reservoirs.add(f['reservoirs'])
                palettes.add(f['palette'])
                backgrounds.add(f['background'])
                depths.add(f['depth'])
    assert len(fingerprints) == 42 and len(shapes) >= 12 and len(carriers) >= 6
    assert len(scales) >= 5 and len(reservoirs) >= 3 and len(palettes) >= 3
    assert len(backgrounds) == 10 and {1, 2, 3, 4} <= depths
    assert any(f['sources'] == 2 for f in rows) and any(f['forks'] for f in rows)
    assert any(f['repeated_state_step'] for f in rows)
    assert any(max(f['areas']) > min(f['areas']) for f in rows)
    for a, b in [(2, 1), (10, 4)]:
        validate_generated_puzzle(generate_puzzle(160 + a, a, b))
    for a, b in [(1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (4, None)]:
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p = generate_puzzle(42, 2, 1)
    again = generate_puzzle(42, 2, 1)
    assert all(np.array_equal(x[r], y[r]) for s in p for x, y in zip(p[s], again[s]) for r in x)
    a = p['test'][0]['input']
    frozen = a.copy()
    assert np.array_equal(solve(a), p['test'][0]['output']) and np.array_equal(a, frozen)
    bad = {s: [{r: a.copy() for r, a in pair.items()} for pair in p[s]] for s in p}
    q = bad['test'][0]
    point = tuple(np.argwhere(q['input'] != q['output'])[0])
    q['output'][point] = q['input'][point]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('a partially changed output was accepted')
    return dict(seeds=42, pairs=len(rows), shapes=len(shapes), carrier_shapes=len(carriers),
                scale_schedules=len(scales), reservoir_counts=sorted(reservoirs),
                palette_sizes=sorted(palettes), backgrounds=sorted(backgrounds), depths=sorted(depths))


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
