"""Repairable routes: minimum station paths subject to finite rectangle packing.

Only the input supplies geometry, inventory, patterns and colors. The divider is
always 9; every other color role is local. See task_spec.md for the full rule.
"""
from collections import Counter
import numpy as np


_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


class _Budget(RuntimeError):
    """Construction screening only; public solve uses exhaustive search."""


def _key(a):
    a = np.asarray(a)
    return a.shape, tuple(int(x) for x in a.flat)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError('Expected a 1..30 by 1..30 integer color grid')
    return a.astype(np.int8, copy=True)


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in _STEPS:
                p = r + dr, c + dc
                if p in todo:
                    todo.remove(p)
                    stack.append(p)
        result.append(np.array(cells, dtype=int))
    return result


def _box(cells):
    lo, hi = cells.min(0), cells.max(0) + 1
    return int(lo[0]), int(lo[1]), int(hi[0]-lo[0]), int(hi[1]-lo[1])


def _square(cells):
    _, _, h, w = _box(cells)
    return h if h == w and len(cells) == h*w else 0


def _paths(adj, start, target):
    """All simple station paths; pixel bends never enter this graph."""
    found = []
    stack = [(start, (start,), ())]
    while stack:
        node, nodes, edges = stack.pop()
        if node == target:
            found.append((nodes, edges))
            continue
        for other, edge in adj[node]:
            if other not in nodes:
                stack.append((other, nodes + (other,), edges + (edge,)))
    return sorted(found, key=lambda p: (len(p[1]), p[1]))


def _network(board, vacancy):
    colors = set(map(int, np.unique(board)))
    if len(colors) != 4 or vacancy not in colors:
        raise ValueError('Network must have four distinct local color roles')
    by_color = {c: _components(board == c) for c in colors - {vacancy}}
    candidates = []
    for endpoint, ec in by_color.items():
        if len(ec) != 2 or _square(ec[0]) < 3 or _square(ec[0]) != _square(ec[1]):
            continue
        size = _square(ec[0])
        for ordinary, oc in by_color.items():
            if ordinary == endpoint or len(oc) < 3 or any(_square(c) != size for c in oc):
                continue
            background = next(iter(colors - {vacancy, endpoint, ordinary}))
            stations = _components((board == endpoint) | (board == ordinary))
            if len(stations) != len(ec) + len(oc):
                continue
            labels = np.full(board.shape, -1, dtype=int)
            for i, cells in enumerate(stations):
                labels[tuple(cells.T)] = i
            endpoints = [i for i, cells in enumerate(stations) if board[tuple(cells[0])] == endpoint]
            connectors = _components(board == vacancy)
            adj, edges = [[] for _ in stations], []
            good = bool(connectors)
            for i, cells in enumerate(connectors):
                touching = set()
                for r, c in cells:
                    for dr, dc in _STEPS:
                        rr, cc = int(r+dr), int(c+dc)
                        if 0 <= rr < board.shape[0] and 0 <= cc < board.shape[1] and labels[rr, cc] >= 0:
                            touching.add(int(labels[rr, cc]))
                if len(touching) != 2:
                    good = False
                    break
                a, b = sorted(touching)
                adj[a].append((b, i)); adj[b].append((a, i))
                edges.append((a, b))
            if not good or any(not neighbors for neighbors in adj):
                continue
            reached, queue = {0}, [0]
            while queue:
                for other, _ in adj[queue.pop()]:
                    if other not in reached:
                        reached.add(other); queue.append(other)
            if len(reached) != len(stations):
                continue
            paths = _paths(adj, *endpoints)
            if paths:
                candidates.append(dict(board=board, vacancy=vacancy, background=background,
                    ordinary=ordinary, endpoint=endpoint, size=size, stations=stations,
                    endpoints=endpoints, connectors=connectors, edges=edges, adj=adj, paths=paths))
    if len(candidates) != 1:
        raise ValueError('Station and endpoint interpretation must be unique')
    return candidates[0]


def _read_input(grid):
    """Recover both panels and every role before seeing an output."""
    dividers = [(axis, k) for axis in (0, 1)
                for k in range(1, grid.shape[axis]-1)
                if np.all(np.take(grid, k, axis=axis) == 9)]
    if len(dividers) != 1:
        raise ValueError('Expected one interior full-span divider of color 9')
    axis, cut = dividers[0]
    if int(np.count_nonzero(grid == 9)) != grid.shape[1-axis]:
        raise ValueError('Color 9 is reserved for the divider')
    a, b = (grid[:cut], grid[cut+1:]) if axis == 0 else (grid[:, :cut], grid[:, cut+1:])
    answers = []
    for board, tray, side in ((a, b, 'top' if axis == 0 else 'left'),
                               (b, a, 'bottom' if axis == 0 else 'right')):
        shared = set(map(int, np.unique(board))) & set(map(int, np.unique(tray)))
        if len(shared) != 1:
            continue
        vacancy = next(iter(shared))
        try:
            data = _network(board, vacancy)
            tiles = []
            for cells in _components(tray != vacancy):
                r, c, h, w = _box(cells)
                if len(cells) != h*w:
                    raise ValueError('Source object is not a filled rectangle')
                tiles.append(tray[r:r+h, c:c+w].copy())
            if not tiles:
                continue
            data.update(tray=tray, tiles=tiles, side=side)
            answers.append(data)
        except ValueError:
            continue
    if len(answers) != 1:
        raise ValueError('Expected exactly one network panel and one source tray')
    return answers[0]


def _inventory(data):
    mask = data['board'] == data['vacancy']
    coords = np.argwhere(mask)
    index = np.full(mask.shape, -1, dtype=int)
    for i, (r, c) in enumerate(coords):
        index[r, c] = i
    groups, lookup = [], {}
    for tile in data['tiles']:
        key = _key(tile)
        if key in lookup:
            groups[lookup[key]]['count'] += 1
        else:
            lookup[key] = len(groups)
            groups.append(dict(pattern=tile, count=1))
    placements, by_cell = [], [[] for _ in coords]
    for g, group in enumerate(groups):
        tile = group['pattern']; h, w = tile.shape
        for r, c in coords:
            if r+h > mask.shape[0] or c+w > mask.shape[1] or not np.all(mask[r:r+h, c:c+w]):
                continue
            ids = tuple(int(i) for i in index[r:r+h, c:c+w].flat)
            bits = sum(1 << i for i in ids)
            p = len(placements)
            placements.append((g, bits, tuple(zip(ids, map(int, tile.flat)))))
            for i in ids:
                by_cell[i].append(p)
    edge_bits = [sum(1 << int(index[r, c]) for r, c in cells) for cells in data['connectors']]
    mono = len({int(c) for tile in data['tiles'] for c in tile.flat}) == 1
    return dict(coords=coords, groups=groups, placements=placements, by_cell=by_cell,
                edge_bits=edge_bits, mono=mono)


def _cover(engine, target, *, unlimited=False, first_only=False, max_nodes=None):
    """Finite exact cover, enumerating rendered answers rather than tile labels."""
    groups, placements = engine['groups'], engine['placements']
    counts = [target.bit_count() if unlimited else g['count'] for g in groups]
    active = [i for i in range(len(engine['coords'])) if target & (1 << i)]
    choices = {i: [p for p in engine['by_cell'][i] if placements[p][1] & target == placements[p][1]]
               for i in active}
    if any(not choices[i] for i in active):
        return {}
    rendered = [-1] * len(engine['coords'])
    outcomes, dead, path = {}, set(), []
    nodes = 0
    stop_at_one = first_only or engine['mono']

    def search(covered):
        nonlocal nodes
        nodes += 1
        if max_nodes is not None and nodes > max_nodes:
            raise _Budget('Construction exact-cover budget')
        if covered == target:
            outcomes.setdefault(tuple(rendered), tuple(path))
            return True
        state = covered, tuple(counts)
        if state in dead:
            return False
        remaining_area = (target ^ covered).bit_count()
        if sum(n * g['pattern'].size for n, g in zip(counts, groups)) < remaining_area:
            dead.add(state)
            return False
        best = None
        for i in active:
            if covered & (1 << i):
                continue
            options = [p for p in choices[i] if counts[placements[p][0]] and not (placements[p][1] & covered)]
            if not options:
                dead.add(state)
                return False
            if best is None or len(options) < len(best):
                best = options
                if len(best) == 1:
                    break
        any_success = False
        for p in best:
            g, bits, assignments = placements[p]
            counts[g] -= 1; path.append(p)
            for i, color in assignments:
                rendered[i] = color
            any_success |= search(covered | bits)
            for i, _ in assignments:
                rendered[i] = -1
            path.pop(); counts[g] += 1
            if len(outcomes) >= (1 if stop_at_one else 2):
                return True
        if not any_success:
            dead.add(state)
        return any_success

    search(0)
    return outcomes


def _path_bits(engine, path):
    bits = 0
    for edge in path[1]:
        bits |= engine['edge_bits'][edge]
    return bits


def _analyze(grid, *, all_routes=False, max_nodes=None):
    data = _read_input(grid)
    engine = _inventory(data)
    feasible, winning_hops = [], None
    for path in data['paths']:
        hops = len(path[1])
        if not all_routes and winning_hops is not None and hops > winning_hops:
            break
        target = _path_bits(engine, path)
        covers = _cover(engine, target, first_only=True, max_nodes=max_nodes)
        if covers:
            if winning_hops is None:
                winning_hops = hops
            feasible.append((path, next(iter(covers.values()))))
    if not feasible:
        raise ValueError('No route can be covered by the source inventory')
    winners = [x for x in feasible if len(x[0][1]) == winning_hops]
    if len(winners) != 1:
        raise ValueError('Minimum repairable station route is tied')
    path, _ = winners[0]
    target = _path_bits(engine, path)
    covers = _cover(engine, target, max_nodes=max_nodes)
    if len(covers) != 1:
        raise ValueError('The winning route has more than one rendered packing')
    rendering, placement_ids = next(iter(covers.items()))
    out = data['board'].copy()
    for i, color in enumerate(rendering):
        if color >= 0:
            out[tuple(engine['coords'][i])] = color
    data.update(engine=engine, feasible=feasible, path=path, target=target,
                placement_ids=placement_ids, output=out)
    return data


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from the visible input; no generation state or demonstrations."""
    return _analyze(_grid(input_grid))['output'].copy()


def _line_cells(points, thickness):
    cells = set()
    for (r, c), (rr, cc) in zip(points, points[1:]):
        if r != rr and c != cc:
            raise ValueError('Generated connectors are rectilinear')
        for y in range(min(r, rr), max(r, rr)+1):
            for x in range(min(c, cc), max(c, cc)+1):
                cells.update((y+dy, x+dx) for dy in range(thickness) for dx in range(thickness))
    return cells


def _draw(anchors, edges, endpoints, size, thickness, margins):
    """Return semantic board and separate connector footprints."""
    bottom, right = margins
    # A suppressed corner can extend beyond every remaining square station.
    # Retain its band footprint when computing the panel bounds.
    points = [p for a, b, line in edges for p in line]
    h = max(max(r for r, c in anchors.values()) + size,
            max(r for r, c in points) + thickness) + bottom
    w = max(max(c for r, c in anchors.values()) + size,
            max(c for r, c in points) + thickness) + right
    board = np.zeros((h, w), dtype=np.int8)
    station_cells = set()
    for node, (r, c) in anchors.items():
        block = {(r+dr, c+dc) for dr in range(size) for dc in range(size)}
        station_cells.update(block)
        for y, x in block:
            board[y, x] = 3 if node in endpoints else 2
    masks = []
    for a, b, points in edges:
        cells = _line_cells(points, thickness) - station_cells
        mask = np.zeros(board.shape, dtype=bool)
        for r, c in cells:
            if board[r, c] != 0:
                raise ValueError('Overlapping connectors')
            board[r, c] = 1; mask[r, c] = True
        masks.append(mask)
    return board, masks


def _palette(rng):
    colors = list(map(int, rng.permutation(9)))
    roles = colors[:4]
    payload = colors[4:4+int(rng.integers(1, 6))]
    return roles, payload


def _pattern(shape, palette, rng, rich=False):
    tile = np.full(shape, palette[0], dtype=np.int8)
    if len(palette) > 1 and rich:
        tile = rng.choice(palette, size=shape).astype(np.int8)
        tile.flat[0] = palette[1]
        if tile.size > 1:
            tile.flat[-1] = palette[0]
    return tile


def _scatter(tiles, width, max_height, vacancy, rng):
    if not tiles or max(t.shape[0] for t in tiles) > max_height or max(t.shape[1] for t in tiles) > width:
        return None
    order = sorted(range(len(tiles)), key=lambda i: -tiles[i].size)
    for attempt in range(16):
        tray = np.full((max_height, width), vacancy, dtype=np.int8)
        occupied = np.zeros(tray.shape, dtype=bool)
        for i in order:
            h, w = tiles[i].shape
            legal = [(r, c) for r in range(max_height-h+1) for c in range(width-w+1)
                     if not occupied[max(0, r-1):min(max_height, r+h+1),
                                     max(0, c-1):min(width, c+w+1)].any()]
            if not legal:
                break
            r, c = legal[int(rng.integers(len(legal)))]
            tray[r:r+h, c:c+w] = tiles[i]; occupied[r:r+h, c:c+w] = True
        else:
            rows = np.flatnonzero(occupied.any(axis=1))
            low = max(0, int(rows[0])-int(rng.integers(3)))
            high = min(max_height, int(rows[-1])+1+int(rng.integers(3)))
            return tray[low:high].copy()
    return None


def _assemble(semantic, rects, extra_rects, rng, *, witness=False):
    roles, payload = _palette(rng)
    if witness and len(payload) == 1:
        payload.append(next(c for c in range(9) if c not in roles+payload))
    all_rects = rects + extra_rects
    shapes = list(dict.fromkeys((h, w) for r, c, h, w in all_rects))
    rich_mode = witness or rng.random() < .55
    patterns = {}
    for shape in shapes:
        h, w = shape
        # Witnesses use solid horizontal stock and asymmetrically patterned
        # vertical stock. This is a sampling choice, never a solving convention.
        rich = (h > w if witness else rng.random() < .65) and rich_mode
        patterns[shape] = _pattern(shape, payload, rng, rich)
    tiles = [patterns[(h, w)].copy() for r, c, h, w in all_rects]
    rng.shuffle(tiles)
    board = np.asarray(roles, dtype=np.int8)[semantic]
    out = board.copy()
    for r, c, h, w in rects:
        if not np.all(semantic[r:r+h, c:c+w] == 1):
            raise ValueError('Construction piece is outside a connector')
        out[r:r+h, c:c+w] = patterns[(h, w)]
    tray = _scatter(tiles, board.shape[1], 29-board.shape[0], roles[1], rng)
    if tray is None:
        return None
    divider = np.full((1, board.shape[1]), 9, dtype=np.int8)
    inp = np.concatenate((board, divider, tray), axis=0)
    return dict(input=inp, output=out)


def _witness_pair(rng, veto, fixed=False):
    """A procedural two-corridor graph, with a chord in the distance witness."""
    size = 3 if fixed else int(rng.choice([3, 4]))
    t, offset = size-2, 1
    gaps = [2, 2, 2] if fixed else list(map(int, rng.integers(2, 5 if size == 3 else 4, 3)))
    x0 = 0 if fixed else int(rng.integers(2))
    xs = [x0]
    for gap in gaps:
        xs.append(xs[-1]+size+gap)
    yt = 0 if fixed else int(rng.integers(2))
    dy = 6 if fixed else int(rng.integers(size+3, size+6))
    if veto:
        # Make an area-only stock check insufficient: enough total material is
        # present, but its orientation and long-piece multiplicity are wrong.
        dy = min(dy, sum(gaps)+offset)
    yb = yt+dy
    xu = xs[1] if not veto or fixed else int(rng.integers(xs[1], xs[2]+1))
    anchors = {0:(yb,xs[0]), 1:(yb,xs[1]), 2:(yb,xs[2]), 3:(yb,xs[3]), 4:(yt,xu)}
    center = lambda n: (anchors[n][0]+offset, anchors[n][1]+offset)
    edges = [(i, i+1, [center(i), center(i+1)]) for i in range(3)]
    edges += [(0,4,[center(0),(yt+offset,xs[0]+offset),center(4)]),
              (4,3,[center(4),(yt+offset,xs[3]+offset),center(3)])]
    if not veto:
        edges.append((4,1,[center(4),center(1)]))
    # An ordinary leaf varies the topology without creating a candidate route.
    leaf = not fixed and size == 3 and rng.random() < .35
    if leaf:
        anchors[5] = yb+size+2, xs[2]
        edges.append((2,5,[center(2),center(5)]))
    margins = (0,0) if fixed else tuple(map(int,rng.integers(0,2,2)))
    semantic, masks = _draw(anchors, edges, {0,3}, size, t, margins)
    direct = [(yb+offset,xs[i]+size,t,gaps[i]) for i in range(3)]
    vertical = [(yt+offset,xs[i]+offset,dy-offset,t) for i in (0,3)]
    horizontal = [(yt+offset,xs[0]+offset+t,t,xu-xs[0]-offset-t),
                  (yt+offset,xu+size,t,xs[3]+offset-xu-size)]
    outer = vertical + horizontal
    extras = []
    if not veto:
        extras.append((yt+size,xu+offset,dy-size,t))
    if leaf:
        extras.append((yb+size,xs[2]+offset,2,t))
    # Each outer elbow can be covered alone. Together they need two identical
    # long vertical tiles, but veto inventory supplies only one physical copy.
    selected = direct if veto else outer
    stock = vertical[:1]+horizontal+extras if veto else direct+extras
    pair = _assemble(semantic, selected, stock, rng, witness=True)
    return pair


def _partition(mask, rng, subdivide=True):
    transpose = rng.random() < .5
    todo = (mask.T if transpose else mask).copy()
    rects = []
    while todo.any():
        r, c = map(int, np.argwhere(todo)[0])
        w = 1
        while c+w < todo.shape[1] and todo[r,c+w]:
            w += 1
        h = 1
        while r+h < todo.shape[0] and np.all(todo[r+h,c:c+w]):
            h += 1
        todo[r:r+h,c:c+w] = False
        rects.append((c,r,w,h) if transpose else (r,c,h,w))
    result = []
    for r,c,h,w in rects:
        length = max(h,w)
        if subdivide and length >= 4 and rng.random() < .35:
            cut = int(rng.integers(1,length))
            if h >= w:
                result.extend([(r,c,cut,w),(r+cut,c,h-cut,w)])
            else:
                result.extend([(r,c,h,cut),(r,c+cut,h,w-cut)])
        else:
            result.append((r,c,h,w))
    return result


def _random_semantic(rng, test, veto_target=False):
    size = int(rng.choice([3,4], p=[.8,.2]))
    t, off = size-2, 1
    nr = int(rng.integers(2,4 if not test else 5))
    nc = int(rng.integers(3,5))
    rows, cols = [int(rng.integers(2))], [int(rng.integers(2))]
    for _ in range(nr-1): rows.append(rows[-1]+size+int(rng.integers(2,5)))
    for _ in range(nc-1): cols.append(cols[-1]+size+int(rng.integers(2,5)))
    if rows[-1]+size > 21 or cols[-1]+size > 29:
        return None
    vertices = [(r,c) for r in range(nr) for c in range(nc)]
    possible = [(v,(v[0]+dr,v[1]+dc)) for v in vertices for dr,dc in ((1,0),(0,1))
                if (v[0]+dr,v[1]+dc) in vertices]
    active = {vertices[int(rng.integers(len(vertices)))]}
    chosen = []
    target = int(rng.integers(max(6,len(vertices)-3),len(vertices)+1))
    while len(active) < target:
        frontier = [e for e in possible if (e[0] in active) != (e[1] in active)]
        if not frontier:
            return None
        edge = frontier[int(rng.integers(len(frontier)))]
        chosen.append(edge); active.update(edge)
    extras = [e for e in possible if e[0] in active and e[1] in active and e not in chosen]
    rng.shuffle(extras)
    chosen += extras[:int(rng.integers(2 if test else 1,5))]
    degree = {v:sum(v in e for e in chosen) for v in active}
    visible = {v for v in active if degree[v] != 2 or rng.random() > .35}
    if len(visible) < (7 if test else 5):
        return None
    anchors = {v:(rows[v[0]],cols[v[1]]) for v in visible}
    graph_adj = {v:[] for v in active}
    for a,b in chosen:
        graph_adj[a].append(b); graph_adj[b].append(a)
    edges, done = [], set()
    for start in sorted(visible):
        for neighbor in graph_adj[start]:
            previous, current = start, neighbor
            walked = [start,neighbor]
            while current not in visible:
                nxt = next(v for v in graph_adj[current] if v != previous)
                previous,current = current,nxt; walked.append(current)
            marker = tuple(sorted((walked[0],walked[1])))
            if marker in done:
                continue
            for a,b in zip(walked,walked[1:]): done.add(tuple(sorted((a,b))))
            if current == start:
                return None
            points = [(rows[r]+off,cols[c]+off) for r,c in walked]
            edges.append((start,current,points))
    # Temporary terminals; the actual choice is made from pixel-derived paths.
    endpoints = set(sorted(visible)[:2])
    semantic, _ = _draw(anchors,edges,endpoints,size,t,tuple(map(int,rng.integers(2,size=2))))
    data = _network(semantic,1)
    choices = []
    for a in range(len(data['stations'])):
        for b in range(a+1,len(data['stations'])):
            paths = _paths(data['adj'],a,b)
            lengths = {len(p[1]) for p in paths}
            if (len(paths) >= (3 if test else 2)
                    and (not test or len(lengths) >= (3 if veto_target else 2))):
                choices.append((a,b,paths))
    if not choices:
        return None
    a,b,paths = choices[int(rng.integers(len(choices)))]
    semantic[semantic == 3] = 2
    for n in (a,b): semantic[tuple(data['stations'][n].T)] = 3
    if test:
        lengths = sorted({len(p[1]) for p in paths})
        target_length = lengths[1] if veto_target else lengths[0]
        target_paths = [p for p in paths if len(p[1]) == target_length]
        path = target_paths[int(rng.integers(len(target_paths)))]
    else:
        path = paths[int(rng.integers(min(len(paths),12)))]
    mask = np.zeros(semantic.shape,dtype=bool)
    for edge in path[1]: mask[tuple(data['connectors'][edge].T)] = True
    rects = _partition(mask,rng)
    spare = []
    longer = [p for p in paths if len(p[1]) > len(path[1])]
    supply_alternative = bool(longer) and (test or rng.random() < .35)
    if supply_alternative or rng.random() < .75:
        other = longer[int(rng.integers(len(longer)))] if supply_alternative else paths[int(rng.integers(len(paths)))]
        other_rects = []
        for edge in other[1]:
            local = np.zeros(semantic.shape,dtype=bool)
            local[tuple(data['connectors'][edge].T)] = True
            other_rects.extend(_partition(local,rng,False))
        if supply_alternative:
            # The stock must support either complete route, not both at once.
            # Maximum counts by shape (rather than summing both plans) retain
            # finite-resource interactions with all the other candidate routes.
            stock_counts = Counter((h,w) for r,c,h,w in rects)
            other_counts = Counter((h,w) for r,c,h,w in other_rects)
            representative = {(h,w):(r,c,h,w) for r,c,h,w in other_rects}
            spare = [representative[shape] for shape,n in other_counts.items()
                     for _ in range(max(0,n-stock_counts[shape]))]
        else:
            rng.shuffle(other_rects)
            spare = other_rects[:int(rng.integers(1,4))]
    if not 2 <= len(rects) <= 13 or len(rects)+len(spare) > 16:
        return None
    return semantic, rects, spare


def _fallback_pair(rng):
    """Two-loop 3x3 network, with all source bars available, used only on retry exhaustion."""
    size, t, off = 3,1,1
    anchors = {(r,c):(r*5,c*5) for r in range(3) for c in range(3)}
    links = [(v,(v[0]+dr,v[1]+dc)) for v in anchors for dr,dc in ((1,0),(0,1))
             if (v[0]+dr,v[1]+dc) in anchors]
    links.remove(((0,0),(0,1))); links.remove(((1,1),(1,2)))
    center = lambda v:(anchors[v][0]+off,anchors[v][1]+off)
    edges = [(a,b,[center(a),center(b)]) for a,b in links]
    semantic, masks = _draw(anchors,edges,{(0,0),(2,2)},size,t,(0,0))
    data = _network(semantic,1)
    eligible = []
    for a in range(9):
        for b in range(a+1,9):
            paths = _paths(data['adj'],a,b)
            if len(paths) >= 3 and len(paths[0][1]) < len(paths[1][1]):
                eligible.append((a,b,paths[0]))
    a,b,path = eligible[int(rng.integers(len(eligible)))]
    semantic[semantic==3] = 2
    for node in (a,b): semantic[tuple(data['stations'][node].T)] = 3
    selected,spare = [],[]
    for i,cells in enumerate(data['connectors']):
        r,c,h,w = _box(cells)
        (selected if i in path[1] else spare).append((r,c,h,w))
    return _assemble(semantic,selected,spare,rng)


def _transform_pair(pair, rng, rotation):
    inp, out = np.rot90(pair['input'],rotation), np.rot90(pair['output'],rotation)
    if rng.random() < .5:
        inp,out = inp[:,::-1],out[:,::-1]
    return dict(input=inp.copy(),output=out.copy())


def _make_pair(rng, kind):
    seek_veto = kind == 'random_test' and rng.random() < .35
    for _ in range(90 if kind == 'random_test' else 45):
        try:
            if kind in ('veto','distance'):
                pair = _witness_pair(rng,kind == 'veto')
            else:
                construction = _random_semantic(rng,kind == 'random_test',seek_veto)
                pair = None if construction is None else _assemble(*construction,rng)
            if pair is None:
                continue
            data = _analyze(pair['input'],all_routes=True,max_nodes=6000)
            if not np.array_equal(data['output'],pair['output']):
                continue
            features = _features(data)
            if kind == 'veto' and not features['global_scarcity']:
                continue
            if kind == 'distance' and not (features['distance_conflict'] and
                    features['tile_count_conflict'] and features['selected_asymmetric']):
                continue
            if kind == 'random_test' and features['feasible_routes'] < 2:
                continue
            return pair
        except (ValueError,_Budget):
            continue
    for _ in range(20):
        pair = (_witness_pair(rng,kind == 'veto',True) if kind in ('veto','distance')
                else _fallback_pair(rng))
        if pair is not None:
            data = _analyze(pair['input'],all_routes=True,max_nodes=20000)
            if np.array_equal(data['output'],pair['output']):
                return pair
    raise RuntimeError('Bounded construction exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name,value,lo,hi in (('num_train',num_train,2,10),('num_test',num_test,1,4)):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not lo <= int(value) <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    # The first two *kinds* are shuffled: the two witnesses always teach the rule.
    kinds = ['veto','distance'] + ['random_train']*(int(num_train)-2)
    rng.shuffle(kinds)
    train_rotation = int(rng.integers(4))
    held_out_side = rng.random() < .5
    result,seen = {'train':[],'test':[]},set()
    for split,scheduled in (('train',kinds),('test',['random_test']*int(num_test))):
        for kind in scheduled:
            for _ in range(12):
                pair = _make_pair(rng,kind)
                rotation = ((train_rotation+(1 if split == 'test' else 0)) % 4
                            if held_out_side else int(rng.integers(4)))
                pair = _transform_pair(pair,rng,rotation)
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input'])); result[split].append(pair)
                    break
            else:
                raise RuntimeError('Could not construct distinct inputs')
    return result


def _features(data):
    engine, path = data['engine'], data['path']
    chosen_area = data['target'].bit_count()
    distance_conflict = any(len(p[0]) > len(path[0]) and _path_bits(engine,p).bit_count() < chosen_area
                            for p,_ in data['feasible'])
    # A nonrectangular connector needs at least two rectangles. This bound
    # proves the witness also defeats minimizing the number of repair pieces.
    tile_lower_bound = sum(1 if len(data['connectors'][e]) ==
        _box(data['connectors'][e])[2]*_box(data['connectors'][e])[3] else 2 for e in path[1])
    tile_count_conflict = any(len(ids) < tile_lower_bound for p,ids in data['feasible'])
    shortest_hops = len(data['paths'][0][1])
    global_scarcity = False
    if len(path[1]) > shortest_hops:
        for other in data['paths']:
            if len(other[1]) != shortest_hops:
                break
            target = _path_bits(engine,other)
            if (sum(t.size for t in data['tiles']) >= target.bit_count()
                    and _cover(engine,target,unlimited=True,first_only=True,max_nodes=10000)
                    and all(_cover(engine,engine['edge_bits'][e],first_only=True,max_nodes=10000)
                            for e in other[1])):
                global_scarcity = True
                break
    used = [engine['groups'][engine['placements'][p][0]]['pattern'] for p in data['placement_ids']]
    return dict(stations=len(data['stations']),routes=len(data['paths']),
        loops=len(data['edges'])-len(data['stations'])+1,
        junctions=sum(len(a)>=3 for a in data['adj']),
        leaves=sum(len(a)==1 for a in data['adj']),
        selected_stations=len(path[0])-2,feasible_routes=len(data['feasible']),
        vetoed_cheaper=len(path[1])>shortest_hops,
        global_scarcity=global_scarcity,distance_conflict=distance_conflict,
        tile_count_conflict=tile_count_conflict,
        selected_asymmetric=any(len(np.unique(t))>1 and not np.array_equal(t,t[::-1,::-1]) for t in used),
        bent_edges=sum(len(c)!=_box(c)[2]*_box(c)[3] for c in data['connectors']),
        used_tiles=len(data['placement_ids']),supplied_tiles=len(data['tiles']),
        selected_area=chosen_area,side=data['side'],size=data['size'],
        tile_shapes=sorted({tuple(t.shape) for t in data['tiles']}),
        roles=(data['background'],data['vacancy'],data['ordinary'],data['endpoint']))


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle,dict) or set(puzzle) != {'train','test'}:
        raise AssertionError('Puzzle must have exactly train and test')
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    nt,ne = len(puzzle['train']),len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen,training = set(),[]
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair)=={'input','output'}
            try:
                inp,out = _grid(pair['input']),_grid(pair['output'])
                data = _analyze(inp,all_routes=True)
                feature = _features(data)
            except (ValueError,_Budget) as exc:
                raise AssertionError(str(exc)) from exc
            assert _key(inp) not in seen
            seen.add(_key(inp))
            assert np.array_equal(data['output'],out), 'Incorrect route or tile rendering'
            # Stations, background and unchosen connectors survive exactly.
            selected = np.zeros(data['board'].shape,dtype=bool)
            for e in data['path'][1]: selected[tuple(data['connectors'][e].T)] = True
            assert np.array_equal(out[~selected],data['board'][~selected])
            assert np.all(out[selected] != data['vacancy'])
            assert np.all(out[selected] != data['background'])
            # Reconstruct the physical finite inventory consumed by this packing.
            consumed = [0]*len(data['engine']['groups'])
            covered = 0
            for p in data['placement_ids']:
                g,bits,_ = data['engine']['placements'][p]
                assert not (bits & covered)
                covered |= bits; consumed[g] += 1
            assert covered == data['target']
            assert all(n<=g['count'] for n,g in zip(consumed,data['engine']['groups']))
            if split == 'train':
                training.append(feature)
            else:
                assert feature['stations'] >= 7 and feature['routes'] >= 3
                assert feature['feasible_routes'] >= 2
    assert any(f['global_scarcity'] for f in training), 'Missing finite-inventory training witness'
    assert any(f['distance_conflict'] and f['tile_count_conflict'] for f in training), 'Missing station-cost training witness'
    assert any(f['selected_asymmetric'] and f['bent_edges'] for f in training)


def _signature(puzzle):
    return tuple((_key(p['input']),_key(p['output'])) for s in ('train','test') for p in puzzle[s])


def self_test():
    """Bounded 42-seed semantic/diversity checks, plus focused counterexamples."""
    features,test_features,structures,signatures,shapes = [],[],set(),set(),set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        if seed < 2:
            assert _signature(puzzle) == _signature(generate_puzzle(seed))
        for split in ('train','test'):
            for pair in puzzle[split]:
                data = _analyze(pair['input'],all_routes=True)
                f = _features(data); features.append(f)
                if split == 'test': test_features.append(f)
                shapes.add(pair['input'].shape)
                semantic = np.zeros(data['board'].shape,dtype=np.int8)
                for role,color in enumerate(f['roles']): semantic[data['board']==color]=role
                structures.add(_key(semantic))
                assert np.array_equal(solve(pair['input']),pair['output'])
    assert len(signatures)==42 and len(structures)>=130 and len(shapes)>=25
    assert {f['side'] for f in features}=={'left','right','top','bottom'}
    assert {f['size'] for f in features}=={3,4}
    assert max(f['routes'] for f in features)>=5 and max(f['loops'] for f in features)>=3
    assert max(f['selected_stations'] for f in features)>=3
    assert all(f['feasible_routes']>=2 for f in test_features)
    assert sum(f['vetoed_cheaper'] for f in test_features)>=5
    assert any(f['leaves'] for f in features) and any(f['bent_edges']>=2 for f in features)
    assert any(f['supplied_tiles']>f['used_tiles'] for f in features)
    assert any(f['supplied_tiles']==f['used_tiles'] for f in features)
    assert len({f['roles'] for f in features})>=80
    assert len({s for f in features for s in f['tile_shapes']})>=25
    # A failed per-edge feasibility shortcut: each isolated edge fits the stock,
    # but the path's shared inventory cannot be spent twice.
    mask = np.ones((1,4),dtype=bool)
    d = dict(board=np.ones((1,4),dtype=np.int8),vacancy=1,
             tiles=[np.array([[4,5]],dtype=np.int8)],connectors=[np.argwhere(mask)])
    engine = _inventory(d)
    assert not _cover(engine,15)
    assert _cover(engine,15,unlimited=True)
    # Same-size different patterns cause a true rendered ambiguity.
    d['tiles'].append(np.array([[6,7]],dtype=np.int8))
    assert len(_cover(_inventory(d),15))==2
    # All network material is supplied in the fallback, so tied shortest
    # station paths really are tied feasible routes, not inventory vetoes.
    tied_input = _fallback_pair(np.random.default_rng(812))['input']
    data = _read_input(tied_input)
    candidates = [(a,b) for a in range(len(data['stations']))
                  for b in range(a+1,len(data['stations']))
                  if len(paths := _paths(data['adj'],a,b)) > 1
                  and len(paths[0][1]) == len(paths[1][1])]
    assert candidates
    a,b = candidates[0]
    modified = data['board'].copy()
    modified[modified == data['endpoint']] = data['ordinary']
    for node in (a,b): modified[tuple(data['stations'][node].T)] = data['endpoint']
    divider = np.full((1,modified.shape[1]),9,dtype=np.int8)
    tied_input = np.concatenate((modified,divider,data['tray']),axis=0)
    try: solve(tied_input)
    except ValueError: pass
    else: raise AssertionError('Tied repairable routes accepted')
    # All explicit count combinations, including the two-demonstration boundary.
    for nt in range(2,11):
        for ne in range(1,min(4,nt-1)+1):
            p = generate_puzzle(9000+nt*10+ne,nt,ne)
            assert len(p['train'])==nt and len(p['test'])==ne
            validate_generated_puzzle(p)
    for nt,ne in [(1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1)]:
        try: generate_puzzle(0,nt,ne)
        except ValueError: pass
        else: raise AssertionError('Illegal counts accepted')
    p = generate_puzzle(84,2,1)
    p['test'][0]['output'][0,0] = (int(p['test'][0]['output'][0,0])+1)%10
    try: validate_generated_puzzle(p)
    except AssertionError: pass
    else: raise AssertionError('Corrupted output accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds, finite packing, route costs, counts, variation)')
