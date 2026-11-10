"""First-completion reservoirs: route-local set accumulation and paint absorption.

Only standard-library modules and NumPy are used.  The solver reads the selector,
chambers, wires, and cupped sources from the input; generation metadata is never
an input to it.  See task_spec.md for the complete domain and rule.
"""
from collections import deque
import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _components(cells, steps=D4):
    left = set(cells)
    groups = []
    while left:
        first = min(left)
        left.remove(first)
        group = {first}
        todo = [first]
        for r, c in todo:
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in left:
                    left.remove(p)
                    group.add(p)
                    todo.append(p)
        groups.append(group)
    return groups


def _neighbors(cells, shape, steps=D4):
    h, w = shape
    cells = set(cells)
    return {(r + dr, c + dc) for r, c in cells for dr, dc in steps
            if 0 <= r + dr < h and 0 <= c + dc < w} - cells


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _view(a, corner):
    if corner & 2:
        a = a[::-1, :]
    if corner & 1:
        a = a[:, ::-1]
    return a


def _as_grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int64, copy=False)


def _analyze(input_grid):
    """Recover an undirected chamber forest, roots, and path histories."""
    original = _as_grid(input_grid)
    assert min(original.shape) >= 6
    selectors = []
    for corner in range(4):
        a = _view(original, corner)
        edge = int(a[3, 3])
        if not (np.all(a[3, :4] == edge) and np.all(a[:4, 3] == edge)):
            continue
        values, counts = np.unique(a[:3, :3], return_counts=True)
        pads = values[counts == 6]
        if len(pads) == 1:
            bg = int(pads[0])
            query = frozenset(int(v) for v in values if v != bg)
            if len(query) == 3 and edge not in query and edge != bg:
                selectors.append((corner, bg, edge, query))
    assert len(selectors) == 1, "Unique three-color corner selector required"
    corner, bg, edge, query = selectors[0]
    a = _view(original, corner)
    h, w = a.shape
    canvas = {(r, c) for r in range(h) for c in range(w)}
    selector = {(r, c) for r in range(4) for c in range(4)}
    # Blocking just the outline recovers interiors even when dense with labels.
    rooms = []
    for cells in _components(zip(*np.where(a != edge))):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells):
            continue
        shell = _neighbors(cells, a.shape)
        assert shell and all(a[p] == edge for p in shell)
        labels = frozenset(int(a[p]) for p in cells if a[p] != bg)
        assert cells.isdisjoint(selector)
        rooms.append(dict(cells=cells, shell=shell, labels=labels,
                          marks=sum(a[p] != bg for p in cells)))
    assert rooms, "At least one fully enclosed chamber required"
    all_inner = set().union(*(r['cells'] for r in rooms))
    all_shell = set().union(*(r['shell'] for r in rooms))
    assert len(all_inner) == sum(len(r['cells']) for r in rooms)
    assert not all_shell & selector
    assert set(zip(*np.where(a == edge))) == all_shell | {
        (3, c) for c in range(4)} | {(r, 3) for r in range(4)}
    for i, room in enumerate(rooms):
        footprint = room['cells'] | room['shell']
        for other in rooms[:i]:
            assert not footprint & (other['cells'] | other['shell'])
            assert not _neighbors(footprint, a.shape) & (other['cells'] | other['shell'])
    outside = {p for p in canvas - selector - all_inner - all_shell if a[p] != bg}
    sources = []
    # A source is the unlike corner of a 2x2 cup of three wire-colored cells.
    for p in sorted(outside):
        r, c = p
        for dr in (-1, 1):
            for dc in (-1, 1):
                cup = {(r + dr, c), (r, c + dc), (r + dr, c + dc)}
                if not cup <= outside:
                    continue
                colors = {int(a[q]) for q in cup}
                near = _neighbors({p}, a.shape, D8) & outside
                if len(colors) == 1 and int(a[p]) not in colors and near == cup:
                    wire = next(iter(colors))
                    paint = int(a[p])
                    if paint not in query | {bg, edge, wire}:
                        sources.append(dict(point=p, cup=cup, paint=paint, wire=wire))
    assert sources and len({s['point'] for s in sources}) == len(sources)
    wire_colors = {s['wire'] for s in sources}
    assert len(wire_colors) == 1
    wire = next(iter(wire_colors))
    assert wire not in query | {bg, edge}
    paint_colors = {s['paint'] for s in sources}
    assert not any(r['labels'] & (paint_colors | {wire, edge}) for r in rooms)
    source_cells = {s['point'] for s in sources}
    assert all(a[p] == wire for p in outside - source_cells)
    wires = _components(outside - source_cells)
    graph = [set() for _ in rooms]
    roots = {}
    segments = []
    for cells in wires:
        near = _neighbors(cells, a.shape)
        attached = [i for i, r in enumerate(rooms) if near & r['shell']]
        caps = [s for s in sources if s['cup'] <= cells]
        if caps:
            assert len(caps) == 1 and len(attached) == 1
            node = attached[0]
            assert node not in roots, "One source per rooted component"
            roots[node] = caps[0]['paint']
            caps[0]['root'] = node
        else:
            assert len(attached) == 2, "Every connector has two chamber ends"
            u, v = attached
            assert v not in graph[u], "No parallel connectors"
            graph[u].add(v)
            graph[v].add(u)
        segments.append(dict(cells=cells, attached=attached, source=bool(caps)))
    assert len(roots) == len(sources)
    # Every chamber component is a tree; at most one root belongs to it.
    groups = []
    left = set(range(len(rooms)))
    while left:
        start = min(left)
        left.remove(start)
        group = {start}
        todo = [start]
        for u in todo:
            for v in graph[u] & left:
                left.remove(v)
                group.add(v)
                todo.append(v)
        assert sum(len(graph[u]) for u in group) == 2 * (len(group) - 1)
        assert len(group & set(roots)) <= 1
        groups.append(group)
    histories, parents, depths, supplied, stopped, fills = {}, {}, {}, {}, {}, {}
    for root, paint in sorted(roots.items()):
        todo = [(root, -1, frozenset(), False, 0)]
        for node, parent, history, blocked, depth in todo:
            collected = history | rooms[node]['labels']
            complete = query <= collected
            histories[node] = collected
            parents[node] = parent
            depths[node] = depth
            supplied[node] = paint
            stopped[node] = blocked
            if complete and not blocked:
                fills[node] = paint
            for child in sorted(graph[node] - {parent}):
                todo.append((child, node, collected, blocked or complete, depth + 1))
    return dict(a=a, corner=corner, bg=bg, edge=edge, wire=wire, query=query,
                rooms=rooms, sources=sources, segments=segments, graph=graph,
                roots=roots, groups=groups, histories=histories, parents=parents,
                depths=depths, supplied=supplied, stopped=stopped, fills=fills)


def _output(info, fills=None):
    b = info['a'].copy()
    for node, paint in (info['fills'] if fills is None else fills).items():
        for p in info['rooms'][node]['cells']:
            b[p] = paint
    return _view(b, info['corner']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Fill only the first query-completing chambers on source-rooted paths."""
    return _output(_analyze(input_grid))


def _ri(rng, lo, hi):
    return int(rng.integers(lo, hi + 1))


def _pick(rng, seq):
    return seq[_ri(rng, 0, len(seq) - 1)]


def _move(cells, r, c):
    return {(y + r, x + c) for y, x in cells}


def _room_shape(rng, teaching):
    """Connected interval profiles with thin, fully closed outlines."""
    if teaching:
        h, w = _pick(rng, ((3, 3), (3, 3), (2, 5), (4, 3)))
    else:
        h, w = _ri(rng, 1, 4), _ri(rng, 2, 5)
        if h * w < 3:
            w = 3
    kind = _pick(rng, ('box', 'box', 'step', 'notch', 'slant'))
    cells = set()
    for r in range(h):
        left = (r // 2) if kind == 'slant' else int(kind == 'step' and r >= h // 2)
        width = w - int(kind == 'notch' and 0 < r < h - 1 and w > 2)
        cells.update((r, c) for c in range(left, left + width))
    if teaching and len(cells) < 9:
        cells = {(r, c) for r in range(3) for c in range(3)}
    shell = {(r + dr, c + dc) for r, c in cells for dr, dc in D4} - cells
    # The cornerless rectangle/stepped outlines are eight-connected barriers
    # to four-connected interiors, as in Parent A's small contours.
    if rng.random() < .5:
        cells = {(c, r) for r, c in cells}
        shell = {(c, r) for r, c in shell}
    r0 = min(r for r, c in shell)
    c0 = min(c for r, c in shell)
    return _move(cells, -r0, -c0), _move(shell, -r0, -c0)


def _bfs(rng, starts, goals, allowed, shape):
    starts = sorted(starts & allowed)
    if not starts or not goals:
        return None
    rng.shuffle(starts)
    todo = deque(starts)
    previous = {p: None for p in starts}
    dirs = [D4[int(i)] for i in rng.permutation(4)]
    end = None
    while todo:
        p = todo.popleft()
        if p in goals:
            end = p
            break
        for dr, dc in dirs:
            q = p[0] + dr, p[1] + dc
            if q in allowed and q not in previous:
                previous[q] = p
                todo.append(q)
    if end is None:
        return None
    path = set()
    while end is not None:
        path.add(end)
        end = previous[end]
    return path


def _ports(room, footprint, shape):
    return _neighbors(room['shell'], shape) - footprint


def _free_layout(rng, n, teaching):
    low = 25 if teaching or n >= 8 else 18
    h, w = _ri(rng, low, 30), _ri(rng, low, 30)
    shape = h, w
    reserve = {(r, c) for r in range(5) for c in range(5)}
    occupied = set(reserve)
    rooms = []
    for _ in range(n):
        inner, shell = _room_shape(rng, teaching)
        rh = max(r for r, c in shell) + 1
        rw = max(c for r, c in shell) + 1
        placed = False
        # Occupancy is unchanged until this room is placed. Reuse its halo
        # across rejected positions instead of rebuilding it up to 160 times.
        clearance = occupied | _neighbors(occupied, shape, D8)
        for _ in range(160):
            row, col = _ri(rng, 1, h - rh - 1), _ri(rng, 1, w - rw - 1)
            ins, edge = _move(inner, row, col), _move(shell, row, col)
            cells = ins | edge
            if not cells & clearance:
                rooms.append(dict(cells=ins, shell=edge))
                occupied |= cells
                placed = True
                break
        if not placed:
            return None
    canvas = {(r, c) for r in range(h) for c in range(w)}
    footprint = set().union(*(r['cells'] | r['shell'] for r in rooms)) | reserve
    base = canvas - footprint - _neighbors(footprint, shape)
    ports = [_ports(room, footprint, shape) for room in rooms]
    centers = [tuple(sum(p[k] for p in room['cells']) / len(room['cells'])
                     for k in (0, 1)) for room in rooms]
    used, edges, segments = set(), [], []
    joined = {_ri(rng, 0, n - 1)}
    while len(joined) < n:
        options = []
        for u in joined:
            for v in set(range(n)) - joined:
                cu, cv = centers[u], centers[v]
                distance = sum(abs(x - y) for x, y in zip(cu, cv))
                options.append((distance * rng.uniform(.65, 1.7), u, v))
        connected = False
        # No wire is added during failed edge attempts in this iteration.
        blocked = used | _neighbors(used, shape)
        for _, u, v in sorted(options):
            starts = ports[u] - blocked
            goals = ports[v] - blocked
            path = _bfs(rng, starts, goals, (base | starts | goals) - blocked, shape)
            if path is not None:
                # No connector may accidentally attach to a third chamber.
                near = _neighbors(path, shape)
                attachments = {i for i, r in enumerate(rooms) if near & r['shell']}
                if attachments != {u, v}:
                    continue
                edges.append((u, v))
                segments.append(path)
                used |= path
                joined.add(v)
                connected = True
                break
        if not connected:
            return None
    return dict(shape=shape, rooms=rooms, edges=edges, segments=segments)


def _fallback_layout(rng):
    """A sparse nine-chamber meander guarantees geometric capacity.

    Ordinary construction uses unrestricted room placement and routed trees.
    This bounded fallback changes only geometry, never the logical lesson.
    """
    rooms = []
    for row in (5, 13, 21):
        columns = (5, 13, 21) if row != 13 else (21, 13, 5)
        for col in columns:
            inner = {(r, c) for r in range(row + 1, row + 4) for c in range(col + 1, col + 4)}
            shell = {(r + dr, c + dc) for r, c in inner for dr, dc in D4} - inner
            rooms.append(dict(cells=inner, shell=shell))
    edges, segments = [], []
    for i in range(8):
        u, v = rooms[i], rooms[i + 1]
        y, x = min(u['cells'])
        yy, xx = min(v['cells'])
        y += 1; x += 1; yy += 1; xx += 1
        if y == yy:
            path = {(y, c) for c in range(min(x, xx) + 3, max(x, xx) - 2)}
        else:
            path = {(r, x) for r in range(min(y, yy) + 3, max(y, yy) - 2)}
        edges.append((i, i + 1)); segments.append(path)
    return dict(shape=(_ri(rng, 28, 30), _ri(rng, 28, 30)), rooms=rooms,
                edges=edges, segments=segments)


def _graph(n, edges):
    g = [set() for _ in range(n)]
    for u, v in edges:
        g[u].add(v); g[v].add(u)
    return g


def _long_paths(graph, minimum):
    candidates = []
    for root in range(len(graph)):
        if len(graph[root]) < 2:
            continue
        todo = [(root, [root])]
        for u, path in todo:
            if len(path) >= minimum:
                candidates.append(path)
            for v in sorted(graph[u] - set(path)):
                todo.append((v, path + [v]))
    return candidates


def _add_source(rng, layout, root, occupied):
    shape = layout['shape']
    h, w = shape
    canvas = {(r, c) for r in range(h) for c in range(w)}
    reserve = {(r, c) for r in range(5) for c in range(5)}
    footprint = set().union(*(r['cells'] | r['shell'] for r in layout['rooms'])) | reserve
    base = canvas - footprint - _neighbors(footprint, shape)
    goals = _ports(layout['rooms'][root], footprint, shape)
    blocked = occupied | _neighbors(occupied, shape, D8)
    goals -= blocked
    # Chambers and existing connectors are fixed for all cup-position retries.
    cup_clearance = footprint | _neighbors(footprint, shape, D8) | blocked
    for _ in range(240):
        rr, cc = _ri(rng, 0, h - 2), _ri(rng, 0, w - 2)
        square = {(rr, cc), (rr + 1, cc), (rr, cc + 1), (rr + 1, cc + 1)}
        if square & cup_clearance:
            continue
        point = _pick(rng, sorted(square))
        cup = square - {point}
        forbidden_tip = _neighbors({point}, shape, D8) | {point}
        allowed = (base | goals | cup) - blocked - forbidden_tip
        allowed |= cup
        path = _bfs(rng, cup, goals, allowed, shape)
        if path is None:
            continue
        path |= cup
        near = _neighbors(path, shape)
        if {i for i, r in enumerate(layout['rooms']) if near & r['shell']} != {root}:
            continue
        return point, path
    return None


def _prepare_layout(rng, kind, multi, inactive):
    teaching = kind in ('history', 'identity')
    n = _ri(rng, 7, 9) if teaching else _ri(rng, 8, 11) if kind == 'shift' else _ri(rng, 3, 9)
    for attempt in range(70):
        layout = _free_layout(rng, n, teaching) if attempt < 60 else _fallback_layout(rng)
        if layout is None:
            continue
        graph = _graph(len(layout['rooms']), layout['edges'])
        second = None
        if multi and kind != 'history':
            leaves = [i for i, adj in enumerate(graph) if len(adj) == 1]
            second = _pick(rng, leaves)
            index = next(i for i, (u, v) in enumerate(layout['edges']) if second in (u, v))
            layout['edges'].pop(index); layout['segments'].pop(index)
            graph = _graph(len(graph), layout['edges'])
        minimum = 5 if kind == 'shift' else 4 if kind == 'history' else 3 if kind == 'identity' else 2
        paths = _long_paths(graph, minimum)
        def detachable(path):
            return [i for i, adj in enumerate(graph) if len(adj) == 1 and i not in path
                    and i != second and (path[0] not in adj or len(graph[path[0]]) > 2)]
        if inactive:
            paths = [p for p in paths if detachable(p)]
        if not paths:
            continue
        path = _pick(rng, paths)
        root = path[0]
        detached = None
        if inactive:
            detached = _pick(rng, detachable(path))
            index = next(i for i, (u, v) in enumerate(layout['edges']) if detached in (u, v))
            layout['edges'].pop(index); layout['segments'].pop(index)
            graph = _graph(len(graph), layout['edges'])
        occupied = set().union(*layout['segments']) if layout['segments'] else set()
        source = _add_source(rng, layout, root, occupied)
        if source is None:
            continue
        caps = [(root, source)]
        if second is not None:
            occupied |= source[1] | {source[0]}
            source2 = _add_source(rng, layout, second, occupied)
            if source2 is None:
                continue
            caps.append((second, source2))
        layout.update(graph=graph, path=path, caps=caps, inactive=detached)
        return layout
    raise RuntimeError('Bounded routing could not realize a reservoir scene')


def _label_rooms(rng, layout, kind, query, extras):
    n = len(layout['rooms'])
    q = list(query)
    # This order schedules witnesses only. Selector placement is shuffled and
    # the semantic predicate is unordered; no order is used by the solver.
    root, path = layout['path'][0], layout['path']
    if kind == 'free':
        marks = []
        for room in layout['rooms']:
            amount = _ri(rng, 0, min(len(room['cells']), 8))
            marks.append([int(v) for v in rng.choice(q + list(extras), amount)])
        return marks
    # Distinct identities matter; repeated marks and extras are permitted in
    # every outcome, and every label decision is independent of room position.
    if kind == 'identity':
        marks = [[q[1], q[2], extras[0]] for _ in range(n)]
        marks[root] = [q[1], q[2], extras[0], extras[1], q[1], q[2], extras[0]]
        marks[path[1]] = list(q)  # three marks, no extras, wholly repainted
        marks[path[2]] = list(q)  # complete descendant, deliberately unchanged
    else:
        marks = [[q[2], extras[0]] if rng.random() < .6 else [] for _ in range(n)]
        marks[root] = [q[0]]
        if kind == 'shift':
            marks[path[1]] = [q[1], q[1]]
            marks[path[2]] = [extras[0], q[0]]
            marks[path[3]] = [q[2]]
            marks[path[4]] = list(q)
        else:
            marks[path[1]] = [q[1]]
            marks[path[2]] = [q[2]]
            if len(path) >= 4:
                marks[path[3]] = list(q)
        # A sibling can supply the globally missing label without sharing it
        # with the long branch. Its high mark count cannot force completion.
        sibling = _pick(rng, sorted(layout['graph'][root] - {path[1]}))
        marks[sibling] = [q[2], extras[0], extras[1], q[2], extras[0]]
    for node, _ in layout['caps'][1:]:
        marks[node] = list(q)
    if layout['inactive'] is not None:
        marks[layout['inactive']] = list(q)
    for i, room in enumerate(layout['rooms']):
        capacity = len(room['cells'])
        if len(marks[i]) > capacity:
            # Free/small shift chambers retain the identity witness before
            # reducing optional multiplicities. Teaching capacities are >=9.
            unique = list(dict.fromkeys(marks[i]))
            assert len(unique) <= capacity
            marks[i] = unique + marks[i][:capacity - len(unique)]
    return marks


def _make_pair(rng, kind, palette, corner, multi, inactive):
    bg, edge, wire, paint1, paint2, q0, q1, q2, x0, x1 = palette
    query = (q0, q1, q2)
    layout = _prepare_layout(rng, kind, multi, inactive)
    marks = _label_rooms(rng, layout, kind, query, (x0, x1))
    a = np.full(layout['shape'], bg, dtype=np.int64)
    a[3, :4] = edge; a[:4, 3] = edge
    for pos, color in zip(rng.choice(9, 3, replace=False), rng.permutation(query)):
        a[int(pos) // 3, int(pos) % 3] = color
    for room, labels in zip(layout['rooms'], marks):
        for p in room['shell']:
            a[p] = edge
        cells = sorted(room['cells'])
        for index, color in zip(rng.choice(len(cells), len(labels), replace=False), rng.permutation(labels)):
            a[cells[int(index)]] = color
    for cells in layout['segments']:
        for p in cells:
            a[p] = wire
    # Construct expected labels from the intended abstract tree, before parsing.
    intended = {}
    for k, (root, (point, cells)) in enumerate(layout['caps']):
        paint = paint1 if k == 0 else paint2
        for p in cells:
            a[p] = wire
        a[point] = paint
        todo = [(root, -1, frozenset(), False)]
        for node, parent, history, blocked in todo:
            colors = history | frozenset(marks[node])
            complete = set(query) <= colors
            if complete and not blocked:
                intended[node] = paint
            for child in sorted(layout['graph'][node] - {parent}):
                todo.append((child, node, colors, blocked or complete))
    b = a.copy()
    for node, paint in intended.items():
        for p in layout['rooms'][node]['cells']:
            b[p] = paint
    a, b = _view(a, corner).copy(), _view(b, corner).copy()
    info = _analyze(a)
    assert np.array_equal(_output(info), b), 'Constructed tree disagrees with visible routing'
    return dict(input=a, output=b)


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    palette = [int(v) for v in rng.permutation(10)]
    varied_roles = bool(rng.integers(2))
    multi = bool(rng.integers(2))
    allow_inactive = bool(rng.integers(2))
    corners = [int(v) for v in rng.choice(4, 2, replace=False)] if rng.random() < .75 else [_ri(rng, 0, 3)]
    puzzle = dict(train=[], test=[])
    seen = set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            kind = ('history' if index == 0 else 'identity' if index == 1 else 'free') if split == 'train' else 'shift'
            colors = [int(v) for v in rng.permutation(10)] if varied_roles else palette
            corner = corners[index % len(corners)] if split == 'train' else _pick(rng, corners)
            inactive = allow_inactive and (kind == 'history' or rng.random() < .5)
            for _ in range(8):
                pair = _make_pair(rng, kind, colors, corner, multi, inactive)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('Could not produce distinct inputs')
            seen.add(key)
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _training_coverage(infos):
    records = [(i, n, r) for i in infos for n, r in enumerate(i['rooms'])]
    assert any(n in i['fills'] and not i['query'] <= r['labels'] for i, n, r in records), 'Cumulative completion'
    assert any(n not in i['fills'] and i['query'] <= r['labels'] and i['stopped'].get(n)
               for i, n, r in records), 'First completion stops the paint'
    assert any(n in i['fills'] and r['marks'] == 3 and r['labels'] == i['query']
               for i, n, r in records), 'Exact local positive'
    assert any(n not in i['fills'] and not i['stopped'].get(n, False) and r['marks'] >= 5
               and len(i['histories'].get(n, frozenset())) >= 4 for i, n, r in records), 'Color/count hard negative'
    # A distributed hit uses three different chambers, making each queried
    # identity necessary. An unblocked sibling witnesses path-local scope.
    assert any(n in i['fills'] and i['depths'][n] >= 2 and
               len(r['labels'] & i['query']) == 1 for i, n, r in records)
    assert any(n not in i['fills'] and not i['stopped'].get(n, False) and
               len(i['histories'].get(n, frozenset()) & i['query']) == 2
               for i, n, r in records)
    assert any(n not in i['fills'] and not i['stopped'].get(n, False)
               and i['query'] <= frozenset().union(*(r['labels'] for r in i['rooms']))
               and i['query'] - i['histories'].get(n, frozenset()) for i, n, _ in records)


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _count(len(puzzle['train']), 'train', 2, 10)
    _count(len(puzzle['test']), 'test', 1, 4)
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, infos = set(), dict(train=[], test=[])
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _as_grid(pair['input']), _as_grid(pair['output'])
            assert a.shape == b.shape
            key = _key(a)
            assert key not in seen
            seen.add(key)
            info = _analyze(a)
            assert np.array_equal(_output(info), b), 'Incorrect frontier or incomplete whole-interior replacement'
            bv = _view(b, info['corner'])
            for n, room in enumerate(info['rooms']):
                if n in info['fills']:
                    assert all(bv[p] == info['fills'][n] for p in room['cells'])
                else:
                    assert all(bv[p] == info['a'][p] for p in room['cells'])
            changed = set(zip(*np.where(info['a'] != bv)))
            allowed = set().union(*(info['rooms'][n]['cells'] for n in info['fills'])) if info['fills'] else set()
            assert changed == allowed
            infos[split].append(info)
    _training_coverage(infos['train'])
    corners = {i['corner'] for i in infos['train']}
    assert all(i['corner'] in corners for i in infos['test'])
    source_counts = {len(i['sources']) for i in infos['train']}
    assert all(len(i['sources']) <= max(source_counts) for i in infos['test'])
    if any(len(i['sources']) == 2 for i in infos['test']):
        assert any(len(set(i['fills'].values())) == 2 for i in infos['train'])
    if any(len(i['groups']) > len(i['sources']) for i in infos['test']):
        assert any(n not in i['supplied'] and i['query'] <= r['labels']
                   for i in infos['train'] for n, r in enumerate(i['rooms']))
    for info in infos['test']:
        assert any(info['depths'][n] >= 3 and not info['query'] <= info['rooms'][n]['labels'] for n in info['fills']), 'Test has delayed distributed completion'


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s]) for s in ('train', 'test'))


def self_test():
    """Bounded structural, counterfactual, diversity, and mutation checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    dimensions, layouts, cavity_shapes, room_counts, source_counts, corners = set(), set(), set(), set(), set(), set()
    roles, depths, degrees, mark_counts = set(), set(), set(), set()
    inactive_scenes = 0
    for seed, puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        if seed < 2:
            assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        for pair in puzzle['train'] + puzzle['test']:
            a, b = pair['input'], pair['output']
            info = _analyze(a)
            dimensions.add(a.shape); corners.add(info['corner'])
            room_counts.add(len(info['rooms'])); source_counts.add(len(info['sources']))
            inactive_scenes += len(info['groups']) > len(info['sources'])
            roles.add((info['bg'], info['edge'], info['wire']))
            degrees.update(map(len, info['graph'])); mark_counts.update(r['marks'] for r in info['rooms'])
            depths.update(info['depths'][n] for n in info['fills'])
            layouts.add((a.shape, tuple(tuple(sorted(r['cells'])) for r in info['rooms']), tuple(tuple(sorted(g)) for g in info['graph'])))
            for r in info['rooms']:
                y = min(p[0] for p in r['cells']); x = min(p[1] for p in r['cells'])
                cavity_shapes.add(tuple(sorted((rr - y, cc - x) for rr, cc in r['cells'])))
            assert np.array_equal(solve(a.tolist()), b)
            # Reflections and global role permutations must commute with solve.
            perm = np.roll(np.arange(10), 1)
            assert np.array_equal(solve(perm[a]), perm[b])
            assert np.array_equal(solve(np.rot90(a)), np.rot90(b))
            moved = a.copy(); mv = _view(moved, info['corner'])
            mv[:3, :3] = np.roll(mv[:3, :3].copy(), 1, axis=0)
            expected = b.copy(); _view(expected, info['corner'])[:3, :3] = mv[:3, :3]
            assert np.array_equal(solve(moved), expected)
            if len(info['sources']) == 2:
                # Repeated paint values on independent roots are legal even
                # though the ordinary sampler uses two different paint roles.
                one, two = info['sources']
                shared = a.copy(); sv = _view(shared, info['corner'])
                sv[two['point']] = one['paint']
                expected = b.copy(); expected[expected == two['paint']] = one['paint']
                assert np.array_equal(solve(shared), expected)
        # Parent ablations: unconditional A fill and local-only B matching both
        # demonstrably give different answers in this puzzle's training.
        a_only = b_only = False
        for pair in puzzle['train']:
            i = _analyze(pair['input'])
            a_only |= not np.array_equal(_output(i, i['supplied']), pair['output'])
            local = {n: i['sources'][0]['paint'] for n, r in enumerate(i['rooms'])
                     if i['query'] <= r['labels']}
            b_only |= not np.array_equal(_output(i, local), pair['output'])
        assert a_only and b_only
        # Remove each query identity along a distributed completion path. Each
        # removal prevents that same reservoir from absorbing paint.
        witness = next((pair, i, n) for pair in puzzle['train']
                       for i in [_analyze(pair['input'])] for n in i['fills']
                       if i['depths'][n] >= 2 and not i['query'] <= i['rooms'][n]['labels'])
        pair, i, n = witness
        ancestors = [n]
        while i['parents'][ancestors[-1]] != -1:
            ancestors.append(i['parents'][ancestors[-1]])
        for q in i['query']:
            a = pair['input'].copy(); v = _view(a, i['corner'])
            for node in ancestors:
                for p in i['rooms'][node]['cells']:
                    if v[p] == q: v[p] = i['bg']
            assert n not in _analyze(a)['fills']
        # Cut the final input connector: a complete descendant now has no
        # source, and its whole component must stay unchanged.
        parent = i['parents'][n]
        segment = next(s for s in i['segments'] if not s['source'] and set(s['attached']) == {parent, n})
        a = pair['input'].copy(); v = _view(a, i['corner'])
        for p in segment['cells']: v[p] = i['bg']
        cut = _analyze(a)
        assert n not in cut['supplied'] and n not in cut['fills']
        # Swap one selector identity for a visible extra: routes stay fixed,
        # and a first-completion frontier must change in a teaching scene.
        changed_query = False
        for pair in puzzle['train']:
            i = _analyze(pair['input'])
            extras = set().union(*(r['labels'] for r in i['rooms'])) - set(i['query'])
            for extra in sorted(extras):
                for old in sorted(i['query']):
                    a = pair['input'].copy(); v = _view(a, i['corner'])
                    v[:3, :3][v[:3, :3] == old] = extra
                    j = _analyze(a)
                    if j['fills'] != i['fills']:
                        changed_query = True
                        break
                if changed_query: break
            if changed_query: break
        assert changed_query
    assert len(dimensions) >= 20 and len(layouts) >= 150 and len(cavity_shapes) >= 20
    assert len(room_counts) >= 6 and source_counts == {1, 2} and corners == {0, 1, 2, 3}
    assert len(roles) >= 30 and max(degrees) >= 3 and {0, 1, 2, 3} <= mark_counts
    assert {1, 2, 3} <= depths
    assert inactive_scenes >= 20
    for a, b in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(1701, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ('train', 'test')}
    p = bad['test'][0]
    r, c = np.argwhere(p['input'] != p['output'])[0]
    p['output'][r, c] = p['input'][r, c]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Partial replacement accepted')
    return {'seeds': 42, 'distinct_layouts': len(layouts), 'grid_shapes': len(dimensions),
            'cavity_shapes': len(cavity_shapes), 'room_counts': sorted(room_counts),
            'source_counts': sorted(source_counts), 'frontier_depths': sorted(depths),
            'inactive_scenes': inactive_scenes}
