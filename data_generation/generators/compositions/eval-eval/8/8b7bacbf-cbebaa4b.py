"""Remote geometric docking connects insulated paint circuits.

Only the standard library and NumPy are used. Generation first builds a discrete
port/trace graph and constructs labels from it. The input-only analyzer separately
recovers boards, keys, traces, taps and source ownership from raster geometry.
There are no parent imports, stored examples, learned color dictionaries or seeds
in solve(). See task_spec.md for the exact family and its deliberate limits.
"""
from collections import deque
import json

import numpy as np


N4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
PIN = 8


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(train, test):
    if not (_integer(train) and _integer(test) and 2 <= train <= 10
            and 1 <= test <= 4 and train > test):
        raise ValueError('Counts must be integers: 2..10 train, 1..4 test, train > test')
    return int(train), int(test)


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        found = {root}
        todo = [root]
        for r, c in todo:
            for dr, dc in N4:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    found.add(q)
                    todo.append(q)
        result.append(found)
    return result


def _neighbors(cells):
    return {(r + dr, c + dc) for r, c in cells for dr, dc in N4} - set(cells)


def _box(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells), max(c for r, c in cells))


def _normal(cells):
    t, l, _, _ = _box(cells)
    return tuple(sorted((r - t, c - l) for r, c in cells))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= s <= 30 for s in a.shape)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError('Expected an integer ARC grid with dimensions 1..30 and colors 0..9')
    return a.astype(np.int8, copy=True)


def _reachable(graph, root):
    found = {root}
    todo = [root]
    for i in todo:
        for j in sorted(graph[i] - found):
            found.add(j)
            todo.append(j)
    return found


def _analyze(value, *, virtual=True):
    """Recover the complete rule from a single input, before seeing any label."""
    a = _grid(value)
    boards = _components(map(tuple, np.argwhere(a != 0).tolist()))
    if not 3 <= len(boards) <= 7:
        raise ValueError('Expected 3..7 separated boards')
    palettes = [{int(a[q]) for q in cells} for cells in boards]
    common = set.intersection(*palettes)
    if PIN not in common or len(common) != 3:
        raise ValueError('Every board must share casing, trace and key hues')
    candidates = common - {PIN}
    # The family requires the common casing hue to strictly dominate the trace.
    order = sorted(candidates, key=lambda c: (-int(np.count_nonzero(a == c)), c))
    casing, trace = order
    if np.count_nonzero(a == casing) <= np.count_nonzero(a == trace):
        raise ValueError('Casing/trace color roles are not distinguishable')

    groups, ports, cavities, pumps = [], [], [], []
    board_nodes = []
    for bi, cells in enumerate(boards):
        t, l, b, r = _box(cells)
        conductive = {q for q in cells if a[q] != casing}
        local = _components(conductive)
        nodes = []
        owner = {}
        for comp in local:
            node = len(groups)
            nodes.append(node)
            groups.append(comp)
            owner.update({q: node for q in comp})
            for q in sorted(comp):
                if int(a[q]) not in (trace, PIN):
                    pumps.append((q, int(a[q]), node, bi))
        board_nodes.append(nodes)
        by_side = {}
        for q in sorted(cells):
            if a[q] != PIN:
                continue
            rr, cc = q
            sides = [s for s, flag in enumerate((rr == t, cc == r, rr == b, cc == l)) if flag]
            if len(sides) != 1:
                raise ValueError('Keys lie on exactly one board side, away from corners')
            by_side.setdefault(sides[0], set()).add(q)
        if not by_side:
            raise ValueError('Every board must have a port')
        for side, key in sorted(by_side.items()):
            if len(key) not in (2, 3) or len({owner[q] for q in key}) != 1:
                raise ValueError('A port has 2/3 pins on one local conductive bus')
            ports.append(dict(board=bi, side=side, key=_normal(key), cells=key,
                              node=owner[next(iter(key))]))
        empty = {(rr, cc) for rr in range(t, b + 1) for cc in range(l, r + 1) if a[rr, cc] == 0}
        holes = [comp for comp in _components(empty)
                 if not any(rr in (t, b) or cc in (l, r) for rr, cc in comp)]
        if len(holes) != 1:
            raise ValueError('Every board has exactly one enclosed reservoir')
        for hole in holes:
            wall = _neighbors(hole)
            if not wall <= cells:
                raise ValueError('Reservoir wall must be complete')
            if any(a[q] not in (casing, trace) for q in wall):
                raise ValueError('Only casing or trace tap cells may bound a reservoir')
            taps = {owner[q] for q in wall if a[q] == trace}
            if len(taps) > 1:
                raise ValueError('A reservoir cannot join distinct paint circuits')
            cavities.append(dict(board=bi, cells=hole, wall=wall,
                                 node=next(iter(taps)) if taps else None))

    graph = [set() for _ in groups]
    matches = {}
    for p in ports:
        matches.setdefault(p['key'], []).append(p)
    board_graph = [set() for _ in boards]
    for key, pair in sorted(matches.items()):
        if len(pair) != 2 or pair[0]['board'] == pair[1]['board']:
            raise ValueError('Each entire key has exactly one mate on another board')
        p, q = pair
        board_graph[p['board']].add(q['board'])
        board_graph[q['board']].add(p['board'])
        if virtual:
            graph[p['node']].add(q['node'])
            graph[q['node']].add(p['node'])
    if len(_reachable(board_graph, 0)) != len(boards):
        raise ValueError('The geometric port graph must connect every board')

    paint = {}
    networks = []
    remaining = set(range(len(groups)))
    while remaining:
        network = _reachable(graph, min(remaining))
        remaining -= network
        sources = {color for _, color, node, _ in pumps if node in network}
        if len(sources) > 1:
            raise ValueError('Conflicting pumps in one conductive circuit')
        color = next(iter(sources)) if sources else None
        networks.append((network, color))
        if color is not None:
            paint.update({node: color for node in network})
    out = a.copy()
    for hole in cavities:
        if hole['node'] in paint:
            for q in hole['cells']:
                out[q] = paint[hole['node']]
    return out, dict(casing=casing, trace=trace, boards=boards, groups=groups,
                     board_nodes=board_nodes, ports=ports, matches=matches,
                     cavities=cavities, pumps=pumps, graph=graph,
                     board_graph=board_graph, paint=paint, networks=networks)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Dock equal sparse keys remotely, trace paint, fill supplied reservoirs."""
    return _analyze(input_grid)[0]


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _tree(rng, count, style):
    if style == 'chain':
        return [(i - 1, i) for i in range(1, count)]
    if style == 'star' and count == 4:
        return [(0, 1), (1, 2), (1, 3)]
    edges, degrees = [], [0] * count
    for j in range(1, count):
        choices = [i for i in range(j) if degrees[i] < 4]
        if style == 'branch' and j == 3:
            choices = [i for i in choices if degrees[i] == 2] or choices
        i = _pick(rng, choices)
        edges.append((i, j))
        degrees[i] += 1
        degrees[j] += 1
    return edges


def _abstract(rng, count, lesson, style):
    """Topology, local insulation, reservoir taps and pumps, independent of pixels."""
    if lesson == 0:
        edges = [(0, 1), (1, 2)]
        incident = [[0], [0, 1], [1]]
        partitions = [[[0]], [[0, 1]], [[1]]]
        taps = [0, None, 0]
        sources = {(0, 0): 0}
    elif lesson == 1:
        edges = [(0, 1), (1, 2), (1, 3)]
        incident = [[0], [0, 1, 2], [1], [2]]
        partitions = [[[0]], [[0, 1], [2]], [[1]], [[2]]]
        # The junction's own pump sends the second color out, while its
        # reservoir receives the first color through the other local circuit.
        taps = [0, 0, 0, 0]
        sources = {(0, 0): 0, (1, 1): 1}
    else:
        edges = _tree(rng, count, style)
        incident = [[e for e, uv in enumerate(edges) if i in uv] for i in range(count)]
        # Splitting one junction creates two independent paint networks while
        # the mechanical/key graph remains one connected tree.
        hub = _pick(rng, [i for i, es in enumerate(incident) if len(es) >= 2])
        partitions = [[list(es)] for es in incident]
        es = list(incident[hub])
        rng.shuffle(es)
        cut = int(rng.integers(1, len(es)))
        partitions[hub] = [es[:cut], es[cut:]]
        taps = [0 if rng.random() < .78 else None for _ in range(count)]
        taps[hub] = _pick(rng, (None, 0, 1))
        nodes = [(i, k) for i, ps in enumerate(partitions) for k in range(len(ps))]
        graph = {node: set() for node in nodes}
        for e, (i, j) in enumerate(edges):
            x = (i, next(k for k, group in enumerate(partitions[i]) if e in group))
            y = (j, next(k for k, group in enumerate(partitions[j]) if e in group))
            graph[x].add(y)
            graph[y].add(x)
        components, left = [], set(nodes)
        while left:
            comp = _reachable(graph, min(left))
            components.append(comp)
            left -= comp
        sources = {}
        for color_index, comp in enumerate(sorted(components, key=lambda c: min(c))):
            # At least one remote tap in each network. Prefer a pump on a
            # different board, and sometimes place it at a nonterminal node.
            targets = [node for node in sorted(comp) if taps[node[0]] == node[1]]
            if not targets:
                target = _pick(rng, sorted(comp))
                taps[target[0]] = target[1]
                targets = [target]
            target = _pick(rng, targets)
            choices = [node for node in sorted(comp) if node[0] != target[0]]
            if not choices:
                return None
            pump = _pick(rng, choices)
            sources[pump] = color_index
        # Keep a dry board when the topology permits it, without erasing every
        # useful remote tap. The lesson already guarantees a dry relay.
        if all(t is not None for t in taps):
            candidates = [i for i in range(count) if sum(taps[j] is not None
                          for j in range(count) if j != i) >= 2]
            i = _pick(rng, candidates)
            old = taps[i]
            taps[i] = None
            if any(not any(taps[j] == k for j, k in comp) for comp in components):
                taps[i] = old

    node_graph = {(i, k): set() for i, ps in enumerate(partitions) for k in range(len(ps))}
    for e, (i, j) in enumerate(edges):
        x = (i, next(k for k, group in enumerate(partitions[i]) if e in group))
        y = (j, next(k for k, group in enumerate(partitions[j]) if e in group))
        node_graph[x].add(y)
        node_graph[y].add(x)
    colors = {}
    for node, ci in sources.items():
        for target in _reachable(node_graph, node):
            if target in colors and colors[target] != ci:
                return None
            colors[target] = ci
    return dict(edges=edges, partitions=partitions, taps=taps, sources=sources,
                colors=colors, lesson=lesson)


def _keys_and_sides(rng, spec, maximum):
    count = len(spec['partitions'])
    used = [set() for _ in range(count)]
    keys, sides, taken = [], {}, set()
    for e, (i, j) in enumerate(spec['edges']):
        choices = [(si, sj) for si in range(4) for sj in range(4)
                   if si not in used[i] and sj not in used[j] and si % 2 == sj % 2]
        if spec['lesson'] == 1 and e == 1:
            # The two receiver leaves must have parallel, equal-area but
            # differently spaced keys carrying different paint colors. Area
            # plus orientation must not identify which receiver gets which.
            choices = [(si, sj) for si, sj in choices if si % 2 != sides[1, 0] % 2]
        if spec['lesson'] == 1 and e == 2:
            choices = [(si, sj) for si, sj in choices if si % 2 == sides[1, 1] % 2]
        rng.shuffle(choices)
        found = False
        for si, sj in choices:
            patterns = [(0, gap) for gap in range(2, maximum - 4)]
            if e >= 2 and spec['lesson'] != 1:
                patterns += [(0, mid, end) for end in range(3, maximum - 4)
                             for mid in range(1, end)]
            options = [pattern for pattern in patterns if (si % 2, pattern) not in taken]
            if not options:
                continue
            pattern = _pick(rng, options)
            keys.append(pattern)
            sides[i, e], sides[j, e] = si, sj
            taken.add((si % 2, pattern))
            used[i].add(si)
            used[j].add(sj)
            found = True
            break
        if not found:
            return None
    return keys, sides


def _path(rng, starts, goals, allowed):
    queue = deque(sorted(starts))
    previous = {q: None for q in starts}
    directions = [N4[int(i)] for i in rng.permutation(4)]
    end = None
    while queue:
        p = queue.popleft()
        if p in goals:
            end = p
            break
        for dr, dc in directions:
            q = p[0] + dr, p[1] + dc
            if q in allowed and q not in previous:
                previous[q] = p
                queue.append(q)
    if end is None:
        return None
    result = set()
    while end is not None:
        result.add(end)
        end = previous[end]
    return result


def _reservoir(rng, h, w):
    hh = int(rng.integers(1, min(4, h - 4)))
    ww = int(rng.integers(1, min(4, w - 4)))
    top = (h - hh) // 2
    left = (w - ww) // 2
    hole = {(r, c) for r in range(top, top + hh) for c in range(left, left + ww)}
    # Connected asymmetric and indented interiors, as well as rectangles.
    if hh >= 2 and ww >= 2 and rng.random() < .55:
        corner = _pick(rng, [(top, left), (top, left + ww - 1),
                            (top + hh - 1, left), (top + hh - 1, left + ww - 1)])
        hole.remove(corner)
    return hole


def _board(rng, bi, spec, keys, sides, casing, trace, paints, maximum):
    groups = spec['partitions'][bi]
    min_h = max([7] + [keys[e][-1] + 5 for group in groups for e in group if sides[bi, e] % 2])
    min_w = max([7] + [keys[e][-1] + 5 for group in groups for e in group if not sides[bi, e] % 2])
    h = int(rng.integers(min_h, maximum + 1))
    w = int(rng.integers(min_w, maximum + 1))
    shape_cells = {(r, c) for r in range(h) for c in range(w)}
    # Chamfered corners vary the casing without changing side/port grouping.
    if rng.random() < .45:
        for q in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
            if rng.random() < .65:
                shape_cells.remove(q)
    hole = _reservoir(rng, h, w)
    wall = _neighbors(hole)
    buses, pins = [], []
    for group in groups:
        bus, markers = set(), set()
        for e in group:
            side, pattern = sides[bi, e], keys[e]
            length = h if side % 2 else w
            offset = int(rng.integers(2, length - pattern[-1] - 2))
            coordinates = [offset + x for x in pattern]
            lo, hi = coordinates[0], coordinates[-1]
            if side == 0:
                markers |= {(0, c) for c in coordinates}
                bus |= {(1, c) for c in range(lo, hi + 1)}
            elif side == 1:
                markers |= {(r, w - 1) for r in coordinates}
                bus |= {(r, w - 2) for r in range(lo, hi + 1)}
            elif side == 2:
                markers |= {(h - 1, c) for c in coordinates}
                bus |= {(h - 2, c) for c in range(lo, hi + 1)}
            else:
                markers |= {(r, 0) for r in coordinates}
                bus |= {(r, 1) for r in range(lo, hi + 1)}
        buses.append(bus)
        pins.append(markers)
    initial = [b | p for b, p in zip(buses, pins)]
    if any(x & (y | _neighbors(y)) for i, x in enumerate(initial) for y in initial[:i]):
        return None
    traces = [set(bus) for bus in buses]
    order = [int(k) for k in rng.permutation(len(groups))]
    for gi in order:
        other = set().union(*(traces[k] | pins[k] for k in range(len(groups)) if k != gi))
        blocked = hole | wall | other | _neighbors(other)
        allowed = shape_cells - blocked - set().union(*pins)
        terminals = _components(traces[gi])
        joined = set(terminals[0])
        for terminal in terminals[1:]:
            path = _path(rng, joined, terminal, allowed | joined | terminal)
            if path is None:
                return None
            joined |= terminal | path
        if spec['taps'][bi] == gi:
            gates = sorted(wall - (other | _neighbors(other)))
            rng.shuffle(gates)
            path = None
            for gate in gates:
                path = _path(rng, joined, {gate}, allowed | joined | {gate})
                if path is not None:
                    break
            if path is None:
                return None
            joined |= path
        traces[gi] = joined
    if any(x & (y | _neighbors(y)) for i, x in enumerate(traces) for y in traces[:i]):
        return None
    if any(len(_components(traces[k] | pins[k])) != 1 for k in range(len(groups))):
        return None

    a = np.zeros((h, w), dtype=np.int8)
    for q in shape_cells - hole:
        a[q] = casing
    for cells in traces:
        for q in cells:
            a[q] = trace
    for cells in pins:
        for q in cells:
            a[q] = PIN
    for (board_index, gi), ci in sorted(spec['sources'].items()):
        if board_index != bi:
            continue
        # A colored pump cell remains part of its conductive route. It is not
        # on a reservoir wall or the geometric marker signature.
        choices = sorted(traces[gi] - wall - set().union(*buses))
        if not choices:
            other = set().union(*(traces[k] | pins[k] for k in range(len(groups)) if k != gi))
            choices = sorted((_neighbors(traces[gi]) & shape_cells)
                             - hole - wall - other - _neighbors(other)
                             - set().union(*traces, *pins))
        if not choices:
            return None
        a[_pick(rng, choices)] = paints[ci]
    out = a.copy()
    tap = spec['taps'][bi]
    if tap is not None and (bi, tap) in spec['colors']:
        for q in hole:
            out[q] = paints[spec['colors'][bi, tap]]
    return a, out


def _scatter(rng, boards):
    """Free scatter where feasible, with a shuffled bounded shelf backup."""
    n = len(boards)
    order = [int(i) for i in rng.permutation(n)]
    rows = []
    per_row = 2 if n <= 4 else 3
    if n in (5, 6) and rng.random() < .5:
        per_row = 2
    for start in range(0, n, per_row):
        rows.append(order[start:start + per_row])
    needed_h = sum(max(boards[i][0].shape[0] for i in row) for row in rows) + len(rows) - 1
    needed_w = max(sum(boards[i][0].shape[1] for i in row) + len(row) - 1 for row in rows)
    if max(needed_h, needed_w) > 30:
        return None
    h = min(30, needed_h + int(rng.integers(0, 6)))
    w = min(30, needed_w + int(rng.integers(0, 6)))
    positions = None
    for _ in range(8):
        placed = {}
        boxes = []
        for i in sorted(order, key=lambda i: -boards[i][0].size):
            hh, ww = boards[i][0].shape
            for attempt in range(100):
                rr = int(rng.integers(h - hh + 1))
                cc = int(rng.integers(w - ww + 1))
                if any(not (rr + hh < y or y + yh < rr or cc + ww < x or x + xw < cc)
                       for y, x, yh, xw in boxes):
                    continue
                placed[i] = (rr, cc)
                boxes.append((rr, cc, hh, ww))
                break
            else:
                break
        if len(placed) == n:
            positions = placed
            break
    if positions is None:
        positions = {}
        rr = int(rng.integers(h - needed_h + 1))
        for row in rows:
            row_h = max(boards[i][0].shape[0] for i in row)
            row_w = sum(boards[i][0].shape[1] for i in row) + len(row) - 1
            cc = int(rng.integers(w - row_w + 1))
            for i in row:
                hh, ww = boards[i][0].shape
                positions[i] = (rr + int(rng.integers(row_h - hh + 1)), cc)
                cc += ww + 1
            rr += row_h + 1
    inp = np.zeros((h, w), dtype=np.int8)
    out = inp.copy()
    for i, (rr, cc) in positions.items():
        a, b = boards[i]
        hh, ww = a.shape
        inp[rr:rr + hh, cc:cc + ww] = a
        out[rr:rr + hh, cc:cc + ww] = b
    k = int(rng.integers(4))
    inp, out = np.rot90(inp, k).copy(), np.rot90(out, k).copy()
    if rng.random() < .5:
        inp, out = np.fliplr(inp).copy(), np.fliplr(out).copy()
    return dict(input=inp, output=out)


def _features(a, info):
    paints = info['paint']
    filled = [c for c in info['cavities'] if c['node'] in paints]
    dry = [c for c in info['cavities'] if c['node'] not in paints]
    remote = 0
    nearest_errors = 0
    for cavity in filled:
        node = cavity['node']
        network = _reachable(info['graph'], node)
        source_boards = {bi for _, color, pn, bi in info['pumps'] if pn in network}
        remote += cavity['board'] not in source_boards
        rr = sum(r for r, c in cavity['cells']) / len(cavity['cells'])
        cc = sum(c for r, c in cavity['cells']) / len(cavity['cells'])
        ranked = sorted((abs(r - rr) + abs(c - cc), color) for (r, c), color, _, _ in info['pumps'])
        if len(ranked) > 1 and ranked[0][0] < ranked[1][0] and ranked[0][1] != paints[node]:
            nearest_errors += 1
    relays = 0
    mixed = 0
    for bi, nodes in enumerate(info['board_nodes']):
        local_cavity = next(c for c in info['cavities'] if c['board'] == bi)
        conducting_colors = {paints[n] for n in nodes if n in paints}
        mixed += len(conducting_colors) >= 2
        for node in nodes:
            port_count = sum(p['node'] == node for p in info['ports'])
            if port_count >= 2 and node in paints and local_cavity['node'] != node:
                relays += 1
    return dict(boards=len(info['boards']), shape=tuple(a.shape),
                casing=info['casing'], trace=info['trace'],
                keys=tuple(sorted(info['matches'])),
                cavity_shapes=tuple(_normal(c['cells']) for c in info['cavities']),
                board_shapes=tuple(_normal(cells) for cells in info['boards']),
                dry=len(dry), filled=len(filled), remote=remote, relays=relays,
                mixed=mixed, nearest_errors=nearest_errors,
                max_degree=max(map(len, info['board_graph'])),
                sources=len(info['pumps']), colors=tuple(sorted(set(paints.values()))))


def _pair(rng, count, lesson, style, casing, trace, paints):
    maximum = 12 if count <= 4 else 9
    for attempt in range(240):
        spec = _abstract(rng, count, lesson, style)
        if spec is None:
            continue
        ports = _keys_and_sides(rng, spec, maximum)
        if ports is None:
            continue
        keys, sides = ports
        boards = [_board(rng, i, spec, keys, sides, casing, trace, paints, maximum)
                  for i in range(count)]
        if any(board is None for board in boards):
            continue
        pair = _scatter(rng, boards)
        if pair is None:
            continue
        try:
            predicted, info = _analyze(pair['input'])
        except ValueError:
            continue
        if not np.array_equal(predicted, pair['output']):
            continue
        f = _features(pair['input'], info)
        if not f['remote']:
            continue
        if lesson == 1 and not f['nearest_errors']:
            continue
        return pair
    raise RuntimeError('Bounded circuit construction exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = [int(c) for c in rng.permutation([1, 2, 3, 4, 5, 6, 7, 9])]
    casing, trace = palette[:2]
    paints = palette[2:4]
    style = _pick(rng, ['chain', 'branch'])
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, count in [('train', nt), ('test', ne)]:
        for index in range(count):
            lesson = index if split == 'train' and index < 2 else None
            boards = 3 if lesson == 0 else 4 if lesson == 1 else int(rng.integers(3, 6) if split == 'train' else rng.integers(6, 8))
            for attempt in range(20):
                pair = _pair(rng, boards, lesson, style, casing, trace, paints)
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    break
            else:
                raise RuntimeError('Could not generate distinct inputs')
            seen.add(key)
            puzzle[split].append(pair)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, features = set(), {'train': [], 'test': []}
    role_pairs = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == b.shape
            key = (a.shape, a.tobytes())
            assert key not in seen
            seen.add(key)
            expected, info = _analyze(a)
            assert np.array_equal(expected, b), 'Input-derived circuit fill disagrees'
            assert np.any(a != b)
            # Rigid geometry, pumps, keys, traces, and casing are untouched.
            assert np.all(a[a != b] == 0)
            all_holes = set().union(*(c['cells'] for c in info['cavities']))
            assert set(map(tuple, np.argwhere(a != b).tolist())) <= all_holes
            for cavity in info['cavities']:
                values = {int(b[q]) for q in cavity['cells']}
                assert len(values) == 1, 'A reservoir must be filled wholly or left wholly empty'
            assert len(info['matches']) == len(info['boards']) - 1
            assert len(info['pumps']) in (1, 2)
            assert len({color for _, color, _, _ in info['pumps']}) == len(info['pumps'])
            f = _features(a, info)
            assert f['remote'] >= 1
            # Omitting B's remote edges must change the answer.
            local_only = _analyze(a, virtual=False)[0]
            assert not np.array_equal(local_only, b), 'Docking has no meaningful effect'
            # Equal-area, differently shaped keys defeat cardinality matching.
            ks = list(info['matches'])
            assert any(len(x) == len(y) and x != y for i, x in enumerate(ks) for y in ks[:i])
            role_pairs.add((info['casing'], info['trace']))
            features[split].append(f)
            if split == 'train' and len(features[split]) == 2:
                colored_keys = [(key, info['paint'].get(pair[0]['node']))
                                for key, pair in info['matches'].items()]
                # Same axis, same cardinality, different geometry and color.
                assert any(len(x) == len(y) and (x[-1][0] == 0) == (y[-1][0] == 0)
                           and x != y and cx is not None and cy is not None and cx != cy
                           for i, (x, cx) in enumerate(colored_keys)
                           for y, cy in colored_keys[:i])
                assert any(c['board'] == bi and c['node'] in info['paint']
                           and info['paint'][c['node']] != color
                           for _, color, _, bi in info['pumps'] for c in info['cavities'])
    assert len(role_pairs) == 1
    first, second = features['train'][:2]
    assert first['boards'] == 3 and first['dry'] >= 1 and first['relays'] >= 1
    assert second['boards'] == 4 and second['sources'] == 2 and second['mixed'] >= 1
    assert second['max_degree'] == 3 and second['nearest_errors'] >= 1
    # Omitting A's selective local routing would paint the dry relay, and
    # merging every trace on the junction board would mix two source colors.
    assert len(second['colors']) == 2
    assert all(f['boards'] <= 5 for f in features['train'])
    assert all(f['boards'] >= 6 for f in features['test'])


def _puzzle_key(puzzle):
    return tuple((s, tuple((np.asarray(p['input']).shape, np.asarray(p['input']).tobytes(),
                           np.asarray(p['output']).tobytes()) for p in puzzle[s]))
                 for s in ('train', 'test'))


def self_test():
    """Bounded 42-seed audit, meaningful negative checks and count boundaries."""
    records, signatures, shapes, keys, interiors, casings, traces = [], set(), set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_key(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = np.asarray(pair['input'])
                _, info = _analyze(a)
                f = _features(a, info)
                records.append(f)
                shapes.add(f['shape'])
                keys.update(f['keys'])
                interiors.update(f['cavity_shapes'])
                casings.add(f['casing'])
                traces.add(f['trace'])
    assert len(signatures) == 42
    assert {3, 4, 5, 6, 7} <= {f['boards'] for f in records}
    assert len(shapes) >= 25 and len(keys) >= 12 and len(interiors) >= 12
    assert len(casings) >= 6 and len(traces) >= 6
    assert any(len(key) == 3 for key in keys)
    assert any(f['max_degree'] == 4 for f in records)
    assert all(f['remote'] for f in records)
    assert sum(bool(f['dry']) for f in records) >= 42
    assert sum(bool(f['mixed']) for f in records) >= 42
    p = generate_puzzle(123, 2, 1)
    assert _puzzle_key(p) == _puzzle_key(generate_puzzle(123, 2, 1))
    for nt, ne in ((10, 4), (10, 1), (5, 4)):
        q = generate_puzzle(1000 + nt + ne, nt, ne)
        validate_generated_puzzle(q)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    pair = bad['train'][0]
    q = tuple(np.argwhere(pair['input'] != pair['output'])[0])
    pair['output'][q] = 0
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Missing paint accepted')
    # Two sources on a single circuit are invalid even if they have plausible
    # local marker geometry. Do not invent a priority/tie color.
    a = p['train'][0]['input'].copy()
    _, info = _analyze(a)
    active_nodes = {n for n, color in info['paint'].items()}
    available = [q for n in active_nodes for q in sorted(info['groups'][n])
                 if a[q] == info['trace'] and all(q not in c['wall'] for c in info['cavities'])]
    color = next(c for c in range(1, 10) if c not in (PIN, info['casing'], info['trace'])
                 and c not in {pc for _, pc, _, _ in info['pumps']})
    a[available[0]] = color
    try:
        solve(a)
    except ValueError:
        pass
    else:
        raise AssertionError('Conflicting supplies accepted')
    return dict(seeds=42, pairs=len(records), distinct_puzzles=len(signatures),
                canvas_shapes=len(shapes), key_masks=len(keys), cavity_masks=len(interiors),
                casing_hues=len(casings), trace_hues=len(traces),
                board_counts=sorted({f['boards'] for f in records}),
                dry_cases=sum(bool(f['dry']) for f in records),
                mixed_cases=sum(bool(f['mixed']) for f in records))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
