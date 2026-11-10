"""Repair keyed cable modules, then complete complementary terminal labels.

All inference is from pixels.  Generation's route maps are used only to make
answers; solve reconstructs placements and a directional wire graph afresh.
Fixed structural roles: 0 background, 8 wire, 5 straight-through crossing,
9 opaque cover.  Other colors are local socket/housing or terminal labels.
"""
from itertools import product
import json
import numpy as np

BG, WIRE, CROSS, COVER = 0, 8, 5, 9
ROLES = {BG, WIRE, CROSS, COVER}
COLORS = (1, 2, 3, 4, 6, 7)
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
DIAMOND = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
CHECKER = ~DIAMOND


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= d <= 30 for d in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("invalid ARC grid")
    return g.astype(np.int8, copy=True)


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def _components(cells, neighbors=N8):
    unseen = set(cells)
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, found = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                p = (r + dr, c + dc)
                if p in unseen:
                    unseen.remove(p)
                    found.add(p)
                    stack.append(p)
        result.append(found)
    return result


def _trim(g):
    r, c = np.nonzero(g != BG)
    return g[r.min():r.max() + 1, c.min():c.max() + 1].copy()


def _canonical(g):
    """A centered two-cell extreme edge is the neck; deduplicate symmetry."""
    choices = {}
    for t in range(4):
        a = np.rot90(g, t).copy()
        h, w = a.shape
        if h < 2 or w < 4 or w % 2:
            continue
        neck = np.zeros(w, dtype=bool)
        neck[w // 2 - 1:w // 2 + 1] = True
        if np.array_equal(a[-1] != BG, neck):
            choices.setdefault(_key(a), (a, t))
    if len(choices) != 1:
        raise ValueError("module needs one geometrically determined attachment")
    return next(iter(choices.values()))


def _objects(g):
    """Separate loose complete modules from the incomplete fixed circuit."""
    bars, modules = [], []
    for cells in _components(map(tuple, np.argwhere(g != BG))):
        rr, cc = np.array(sorted(cells)).T
        values = set(map(int, g[rr, cc]))
        payload = values - ROLES
        if len(cells) == 2 and len(payload) == len(values) == 1:
            a, b = sorted(cells)
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
                raise ValueError("socket bar is not an adjacent pair")
            bars.append((next(iter(payload)), tuple(sorted(cells))))
        elif len(payload) == 1:
            color = next(iter(payload))
            block = g[rr.min():rr.max() + 1, cc.min():cc.max() + 1].copy()
            # A complete module has a neck made of its housing color.  Circuit
            # components and 3x3 terminal clues do not have this silhouette.
            try:
                canonical, turn = _canonical(block)
            except ValueError:
                continue
            if np.all(canonical[-1, canonical.shape[1] // 2 - 1:
                                canonical.shape[1] // 2 + 1] == color):
                modules.append(dict(color=color, cells=cells, raster=block,
                                    canonical=canonical, undo_turn=turn))
    if not 1 <= len(bars) <= 2 or len({b[0] for b in bars}) != len(bars):
        raise ValueError("expected one or two differently colored sockets")
    if not modules or len(modules) > 4:
        raise ValueError("expected separated loose modules")
    for color, _ in bars:
        if sum(m['color'] == color for m in modules) != 1:
            raise ValueError("each socket needs exactly one matching module")
    return bars, modules


def _placement_options(base, canonical, bar):
    """Try both normals of a bar; visible wire clearance resolves its side."""
    neck = np.zeros(canonical.shape, dtype=bool)
    neck[-1, canonical.shape[1] // 2 - 1:canonical.shape[1] // 2 + 1] = True
    barset = set(bar)
    options = []
    for t in range(4):
        a = np.rot90(canonical, t).copy()
        nn = np.argwhere(np.rot90(neck, t))
        # The original neck faces down. After a CCW turn its normal changes.
        dr, dc = DIRS[(2 - t) % 4]
        target = np.array(bar) - np.array((dr, dc))
        shift = target.min(axis=0) - nn.min(axis=0)
        if set(map(tuple, nn + shift)) != set(map(tuple, target)):
            continue
        coords = np.argwhere(a != BG) + shift
        r, c = coords.T
        if (np.any(r < 0) or np.any(c < 0) or np.any(r >= base.shape[0])
                or np.any(c >= base.shape[1]) or np.any(base[r, c] != BG)):
            continue
        if set(map(tuple, coords)) & barset:
            continue
        vals = a[a != BG]
        options.append(dict(coords=coords, values=vals, turn=t, raster=a))
    return options


def _cover_rectangles(g):
    result = []
    for cells in _components(map(tuple, np.argwhere(g == COVER)), DIRS):
        a = np.array(sorted(cells))
        r0, c0 = a.min(axis=0)
        r1, c1 = a.max(axis=0)
        if r1 <= r0 or c1 <= c0 or len(cells) != (r1-r0+1)*(c1-c0+1):
            raise ValueError("covers must be solid rectangles at least 2 by 2")
        result.append((int(r0), int(r1), int(c0), int(c1)))
    return result


def _wire_graph(g):
    visible = set(map(tuple, np.argwhere((g == WIRE) | (g == CROSS))))
    graph = {}
    for r, c in sorted(visible):
        graph[r, c] = {d: ((r+dr, c+dc), (d+2) % 4)
                       for d, (dr, dc) in enumerate(DIRS)
                       if (r+dr, c+dc) in visible}
    masks = []
    for rect in _cover_rectangles(g):
        r0, r1, c0, c1 = rect
        boundary = [((r0-1, c), 2) for c in range(c0, c1+1)]
        boundary += [((r1+1, c), 0) for c in range(c0, c1+1)]
        boundary += [((r, c0-1), 1) for r in range(r0, r1+1)]
        boundary += [((r, c1+1), 3) for r in range(r0, r1+1)]
        ports = {(p, d) for p, d in boundary if p in graph
                 and len(graph[p]) in (1, 3) and d not in graph[p]}
        if not ports or len(ports) % 2:
            raise ValueError("cover has broken ports")
        remaining, links = set(ports), []
        while remaining:
            p, d = min(remaining)
            remaining.remove((p, d))
            axis = 1 if d % 2 == 0 else 0
            choices = [(q, e) for q, e in remaining
                       if e == (d+2) % 4 and p[axis] == q[axis]]
            if len(choices) != 1:
                raise ValueError("cover needs opposite aligned ports")
            q, e = choices[0]
            remaining.remove((q, e))
            graph[p][d], graph[q][e] = (q, e), (p, d)
            links.append((p, q))
        masks.append((rect, links))
    endpoints = []
    for p, edges in graph.items():
        degree = len(edges)
        if degree == 1:
            if g[p] != WIRE:
                raise ValueError("crossing cannot be an endpoint")
            endpoints.append(p)
        elif degree == 4:
            if g[p] != CROSS:
                raise ValueError("unmarked wire junction")
        elif degree != 2 or g[p] == CROSS:
            raise ValueError("broken wire or misplaced crossing")
    if len(endpoints) != 6:
        raise ValueError("the circuit has three routes and six terminal stubs")
    return graph, sorted(endpoints), masks


def _terminal(g, graph, p):
    inward = next(iter(graph[p]))
    dr, dc = DIRS[inward]
    r, c = p[0] - 2*dr, p[1] - 2*dc
    if not (1 <= r < g.shape[0]-1 and 1 <= c < g.shape[1]-1):
        raise ValueError("terminal footprint falls outside the canvas")
    block = g[r-1:r+2, c-1:c+2]
    colors = set(map(int, np.unique(block))) - {BG}
    if not colors:
        return dict(center=(r, c), kind="blank", label=None)
    if len(colors) != 1 or not colors.isdisjoint(ROLES):
        raise ValueError("bad terminal label")
    label = next(iter(colors))
    if np.all(block == label):
        kind = 'solid'
    elif np.array_equal(block == label, DIAMOND):
        kind = 'diamond'
    elif np.array_equal(block == label, CHECKER):
        kind = 'checker'
    else:
        raise ValueError("terminal clue is not solid, diamond, checker or blank")
    return dict(center=(r, c), kind=kind, label=label)


def _trace_and_complete(g):
    graph, endpoints, masks = _wire_graph(g)
    terminals = {p: _terminal(g, graph, p) for p in endpoints}
    left = set(endpoints)
    seen, routes = set(), []
    out = g.copy()
    while left:
        start = min(left)
        p, incoming, states, route = start, None, set(), []
        while True:
            state = (p, incoming)
            if state in states:
                raise ValueError("wire cycle")
            states.add(state)
            seen.add(p)
            route.append(p)
            if p != start and p in terminals:
                break
            edges = graph[p]
            if incoming is None:
                d = next(iter(edges))
            elif len(edges) == 4:
                d = (incoming+2) % 4
            else:
                d = next(d for d in edges if d != incoming)
            p, incoming = edges[d]
        if p not in left or p == start:
            raise ValueError("route is not a terminal pairing")
        left.remove(start)
        left.remove(p)
        a, b = terminals[start], terminals[p]
        labels = {x['label'] for x in (a, b) if x['label'] is not None}
        if len(labels) != 1:
            raise ValueError("absent or conflicting route label")
        label = next(iter(labels))
        source_candidates = []
        for i, (x, y) in enumerate(((a, b), (b, a))):
            if x['kind'] in ('blank', 'solid', 'diamond') and y['kind'] in ('blank', 'checker'):
                source_candidates.append(i)
        if len(source_candidates) != 1:
            raise ValueError("clues do not identify complementary terminal roles")
        source_index = source_candidates[0]
        for i, term in enumerate((a, b)):
            pattern = DIAMOND if i == source_index else CHECKER
            r, c = term['center']
            out[r-1:r+2, c-1:c+2] = np.where(pattern, label, BG)
        routes.append(dict(path=route, terminals=(a, b), label=label,
                           source_index=source_index))
    if seen != set(graph):
        raise ValueError("unattached wire fragment")
    return out, dict(routes=routes, masks=masks, graph=graph)


def _analyze(input_grid):
    g = _grid(input_grid)
    bars, modules = _objects(g)
    base = g.copy()
    for m in modules:
        for p in m['cells']:
            base[p] = BG
    options = []
    matches = []
    for color, bar in sorted(bars):
        m = next(m for m in modules if m['color'] == color)
        candidates = _placement_options(base, m['canonical'], bar)
        if not candidates:
            raise ValueError("module has no unclipped socket placement")
        options.append(candidates)
        matches.append(m)
    answers = {}
    for arrangement in product(*options):
        repaired = base.copy()
        valid = True
        for item in arrangement:
            r, c = item['coords'].T
            if np.any(repaired[r, c] != BG):
                valid = False
                break
            repaired[r, c] = item['values']
        if not valid:
            continue
        try:
            answer, circuit = _trace_and_complete(repaired)
        except (ValueError, StopIteration):
            continue
        answers[_key(answer)] = (answer, dict(bars=bars, modules=modules,
            matched=matches, arrangement=arrangement, repaired=repaired,
            base=base, circuit=circuit))
    if len(answers) != 1:
        raise ValueError("input must specify one repaired circuit and answer")
    return next(iter(answers.values()))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the uniquely assembled circuit and completed labels from pixels."""
    return _analyze(input_grid)[0]


def _line(path, a, b):
    if a[0] != b[0] and a[1] != b[1]:
        raise ValueError("line must be axial")
    dr, dc = int(np.sign(b[0]-a[0])), int(np.sign(b[1]-a[1]))
    p = a
    if not path or path[-1] != p:
        path.append(p)
    while p != b:
        p = (p[0]+dr, p[1]+dc)
        path.append(p)


def _module(rng, width, body, stem, color, swap):
    a = np.full((body+stem, width), BG, dtype=np.int8)
    a[:body] = color
    mid = width//2
    a[body:, mid-1:mid+1] = color
    ports, spare = (1, width-2), mid-1
    paths = [[], []]
    if swap:
        # Two paths exchange lanes using an interior spare track. Only their
        # straight transverse/longitudinal intersection is a crossover.
        extra = body - 7
        first = 1 + int(rng.integers(extra+1))
        second, third = first+2, first+4
        corners = [(0, ports[0]), (first, ports[0]), (first, spare),
                   (third, spare), (third, ports[1]), (body-1, ports[1])]
        for x, y in zip(corners, corners[1:]):
            _line(paths[0], x, y)
        corners = [(0, ports[1]), (second, ports[1]), (second, ports[0]),
                   (body-1, ports[0])]
        for x, y in zip(corners, corners[1:]):
            _line(paths[1], x, y)
    else:
        for i, col in enumerate(ports):
            _line(paths[i], (0, col), (body-1, col))
    for path in paths:
        for p in path:
            a[p] = WIRE
    for p in set(paths[0]) & set(paths[1]):
        a[p] = CROSS
    # Edge cutouts and holes alter actual silhouettes, never wire cells.
    proposals = [(r, c) for r in range(body) for c in (0, width-1)]
    proposals += [(r, c) for r in range(1, body-1) for c in range(2, width-2)]
    rng.shuffle(proposals)
    for p in proposals[:int(rng.integers(0, 7))]:
        if a[p] != color:
            continue
        old = a[p]
        a[p] = BG
        try:
            canon, _ = _canonical(a)
            if not np.array_equal(canon, a) or len(_components(map(tuple, np.argwhere(a != BG)))) != 1:
                a[p] = old
        except ValueError:
            a[p] = old
    return a, (1, 0) if swap else (0, 1)


def _external_path(r0, r1, col, outward, detour):
    path = []
    if detour and r1-r0 >= 4:
        for a, b in zip(((r0, col), (r0+1, col), (r0+1, col+outward),
                         (r1-1, col+outward), (r1-1, col), (r1, col)),
                        ((r0+1, col), (r0+1, col+outward), (r1-1, col+outward),
                         (r1-1, col), (r1, col))):
            _line(path, a, b)
    else:
        _line(path, (r0, col), (r1, col))
    return path


def _add_covers(rng, fixed, full, regions, wanted, force_multi=False):
    proposals = []
    for r0, r1, left, width in regions:
        for row in range(r0+1, r1):
            for lo, hi in ((left, left+1), (left+width-2, left+width-1),
                           (left, left+width-1)):
                if force_multi and hi-lo < width-1:
                    continue
                proposals.append((row, row+1, lo, hi))
    rng.shuffle(proposals)
    used = []
    for rect in proposals:
        if len(used) >= wanted:
            break
        r0, r1, c0, c1 = rect
        block = full[r0:r1+1, c0:c1+1]
        if not np.all((block == BG) | (block == WIRE)) or not np.any(block == WIRE):
            continue
        if any(not (r1+1 < a or b+1 < r0 or c1+1 < c or d+1 < c0)
               for a, b, c, d in used):
            continue
        trial = full.copy()
        trial[r0:r1+1, c0:c1+1] = COVER
        try:
            _, info = _trace_and_complete(trial)
        except (ValueError, StopIteration):
            continue
        # The complete construction still has the same terminal pairings.
        old_pairs = {frozenset(t['center'] for t in r['terminals'])
                     for r in _trace_and_complete(full)[1]['routes']}
        new_pairs = {frozenset(t['center'] for t in r['terminals']) for r in info['routes']}
        if new_pairs != old_pairs:
            continue
        fixed[r0:r1+1, c0:c1+1] = COVER
        full = trial
        used.append(rect)
    return fixed, full


def _scatter(rng, fixed, loose, core_left, core_width):
    """Use the open side strips; a separated packing is a bounded fallback."""
    h, w = fixed.shape
    for attempt in range(12):
        g = fixed.copy()
        blocked = g != BG
        blocked[:, core_left-1:core_left+core_width+1] = True
        placed = True
        order = list(range(len(loose)))
        order.sort(key=lambda i: -loose[i].size)
        for i in order:
            a = loose[i]
            hh, ww = a.shape
            local = np.argwhere(a != BG)
            candidates = []
            for r in range(h-hh+1):
                for c in range(w-ww+1):
                    rr, cc = (local + (r, c)).T
                    if not np.any(blocked[rr, cc]):
                        candidates.append((r, c))
            if not candidates:
                placed = False
                break
            r, c = candidates[int(rng.integers(len(candidates)))]
            rr, cc = (local + (r, c)).T
            g[rr, cc] = a[a != BG]
            for x, y in zip(rr, cc):
                blocked[max(0, x-1):x+2, max(0, y-1):y+2] = True
        if placed:
            return g
    # All rasters fit vertically in the left 10/11-column strip; the builder
    # reserves their exact total height, including two blank rows between them.
    g = fixed.copy()
    cursor = 0
    for i, a in enumerate(loose):
        r, c = cursor, 0
        cursor += a.shape[0]+2
        rr, cc = np.nonzero(a != BG)
        if r+a.shape[0] > h or c+a.shape[1] > w:
            raise ValueError("packing budget exhausted")
        g[r+rr, c+cc] = a[rr, cc]
    return g


def _build_pair(rng, n, swaps, active_pairs, forced_turns=None, lesson=None, shift=None, scene_turn=None):
    width = int(rng.choice((8, 10)))
    pitch, core_width = width-3, 2*width-3
    palette = list(map(int, rng.permutation(COLORS)))
    housing, labels = palette[:n], palette[n:n+3]
    distractor_count = (1 if lesson else int(rng.integers(0, 3)))
    if n == 2:
        distractor_count = min(1, distractor_count)
    # Unmatched modules can share a color, but matching
    # housing labels are unique and disjoint from terminal label colors.
    unused = palette[n+3:]
    body = [int(rng.integers(7, 10) if swap else rng.integers(4, 9)) for swap in swaps]
    stems = [int(rng.integers(1, 3)) for _ in swaps]
    for _ in range(32):
        if not any(b+s > 10 for b, s in zip(body, stems)) and sum(body)+sum(stems) <= (17 if n == 2 else 10):
            break
        body = [int(rng.integers(7, 9) if swap else rng.integers(4, 8)) for swap in swaps]
        stems = [1 for _ in swaps]
    else:
        body, stems = [7 if swap else 4 for swap in swaps], [1]*n
    minimum = 6 + 1 + sum(body)+sum(stems) + 3*n
    budget = 30-minimum
    if budget < 0:
        raise ValueError("circuit budget exhausted")
    top, bottom = 0, 0
    lead, tail, gap = 1, 3, 3
    if lesson == 'routing' or (shift == 'occlusion' and n == 1):
        add = min(3, budget)
        lead += add
        budget -= add
    for _ in range(int(rng.integers(budget+1))):
        choice = int(rng.integers(5))
        if choice == 0:
            top += 1
        elif choice == 1:
            bottom += 1
        elif choice == 2:
            lead += 1
        elif choice == 3:
            tail += 1
        elif n == 2:
            gap += 1
        else:
            bottom += 1
    left = int(rng.integers(11, 13))
    positions = []
    start_stub = top+3
    row = start_stub+lead
    for b, s in zip(body, stems):
        positions.append(row)
        row += b+s+gap
    last_bar = positions[-1]+body[-1]+stems[-1]
    receive_top = last_bar+tail
    end_stub = receive_top-1
    height = receive_top+3+bottom
    canvas_width = 30 if distractor_count else int(rng.integers(left+core_width+1, 31))
    fixed = np.zeros((height, canvas_width), dtype=np.int8)
    full = fixed.copy()
    ports = tuple(left+1+j*pitch for j in range(3))
    specs, maps, loose = [], [], []
    for i, (r, b, s, color, swap) in enumerate(zip(positions, body, stems, housing, swaps)):
        raster, mapping = _module(rng, width, b, s, color, swap)
        module_left = left+active_pairs[i]*pitch
        full[r:r+raster.shape[0], module_left:module_left+width] = raster
        barrow, mid = r+b+s, module_left+width//2
        fixed[barrow, mid-1:mid+1] = color
        full[barrow, mid-1:mid+1] = color
        specs.append(raster)
        route_map = list(range(3))
        for local, dest in enumerate(mapping):
            route_map[active_pairs[i]+local] = active_pairs[i]+dest
        maps.append(route_map)
        # The third wire bypasses this module and remains visible in the input.
        bypass = 2 if active_pairs[i] == 0 else 0
        fixed[r:r+b, ports[bypass]] = WIRE
        full[r:r+b, ports[bypass]] = WIRE
        turn = forced_turns[i] if forced_turns is not None else int(rng.integers(4))
        loose.append(np.rot90(raster, turn).copy())
    regions = []
    prev = start_stub
    for i, r in enumerate(positions):
        regions.append((prev, r-1, left, core_width))
        for j, col in enumerate(ports):
            path = _external_path(prev, r-1, col, -1 if j == 0 else 1,
                                  rng.random() < .35 and lesson != 'routing')
            for p in path:
                fixed[p] = full[p] = WIRE
        prev = r+body[i]
    regions.append((prev, end_stub, left, core_width))
    for j, col in enumerate(ports):
        path = _external_path(prev, end_stub, col, -1 if j == 0 else 1,
                              rng.random() < .45)
        for p in path:
            fixed[p] = full[p] = WIRE
    permutation = [0, 1, 2]
    for mapping in maps:
        permutation = [mapping[j] for j in permutation]
    kinds = [('solid', 'blank'), ('diamond', 'blank'), ('blank', 'checker'),
             ('solid', 'checker'), ('diamond', 'checker')]
    clue_modes = [kinds[int(rng.integers(len(kinds)))] for _ in range(3)]
    if lesson == 'routing':
        clue_modes = [('solid', 'blank'), ('diamond', 'blank'), ('solid', 'blank')]
    elif lesson == 'assembly':
        clue_modes = [('blank', 'checker'), ('diamond', 'checker'), ('solid', 'blank')]
    # Label colors may repeat in supplemental pairs. Witnesses stay distinct.
    if lesson is None and rng.random() < .25:
        labels[int(rng.integers(1, 3))] = labels[0]
    if shift in ('composition', 'cancellation', 'occlusion'):
        if all(a != 'blank' and b != 'blank' for a, b in clue_modes):
            clue_modes[int(rng.integers(3))] = ('diamond', 'blank')
        if permutation != [0, 1, 2] and all(labels[i] == labels[permutation[i]] for i in range(3)):
            labels = palette[n:n+3]
    expected = full.copy()
    for i, (src_mode, dst_mode) in enumerate(clue_modes):
        for r, col, mode, pattern in ((top, ports[i], src_mode, DIAMOND),
                 (receive_top, ports[permutation[i]], dst_mode, CHECKER)):
            block = np.zeros((3, 3), dtype=np.int8)
            if mode == 'solid':
                block[:] = labels[i]
            elif mode != 'blank':
                block = np.where(DIAMOND if mode == 'diamond' else CHECKER, labels[i], BG).astype(np.int8)
            fixed[r:r+3, col-1:col+2] = block
            full[r:r+3, col-1:col+2] = block
            expected[r:r+3, col-1:col+2] = np.where(pattern, labels[i], BG)
    wanted = 1 if lesson == 'routing' else int(rng.integers(0, 2 if shift == 'train_occlusion' else 3))
    if shift == 'occlusion':
        wanted = 2
    fixed, full = _add_covers(rng, fixed, full, regions, wanted,
                            force_multi=lesson == 'routing')
    if lesson == 'routing' and not np.any(full == COVER):
        raise ValueError('routing demonstration needs its shared cover')
    if shift == 'occlusion' and len(_cover_rectangles(full)) != 2:
        raise ValueError('occlusion query needs two covers')
    expected[full == COVER] = COVER
    for j in range(distractor_count):
        color = int(rng.choice(unused))
        remaining_minimum = (distractor_count-j-1)*5
        max_height = 30-sum(a.shape[0] for a in loose)-2*(n+distractor_count-1)-remaining_minimum
        if lesson == 'routing':
            # A geometrically and functionally identical false key makes local
            # color correspondence indispensable even in two-example puzzles.
            extra = np.where(specs[0] == housing[0], color, specs[0]).astype(np.int8)
        else:
            extra_swap = max_height >= 8 and rng.random() < .35
            extra_body = 7 if extra_swap else int(rng.integers(4, min(9, max_height-1)+1))
            extra, _ = _module(rng, width, extra_body, 1, color, extra_swap)
        choices = (0, 1, 2, 3) if width <= max_height else (0, 2)
        loose.append(np.rot90(extra, int(rng.choice(choices))).copy())
    packing_height = sum(a.shape[0] for a in loose)+2*(len(loose)-1)
    if packing_height > 30:
        raise ValueError('loose objects exceed the height budget')
    if packing_height > height:
        padding = ((0, packing_height-height), (0, 0))
        fixed, expected = np.pad(fixed, padding), np.pad(expected, padding)
    inp = _scatter(rng, fixed, loose, left, core_width)
    # Physical bank direction is encoded by clue forms, not by its location.
    # Both orientations occur in training; tests recombine them with new keys.
    global_turn = int(rng.integers(4)) if scene_turn is None else scene_turn
    inp, expected = np.rot90(inp, global_turn).copy(), np.rot90(expected, global_turn).copy()
    return dict(input=inp, output=expected)


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(('composition', 'cancellation', 'occlusion')))
    witnesses = list(map(int, rng.choice(nt, 2, replace=False)))
    witness_pair = int(rng.integers(2))
    witness_reverse = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            lesson = ('routing' if i == witnesses[0] else 'assembly' if i == witnesses[1]
                      else None) if split == 'train' else None
            if lesson == 'routing':
                n, swaps, turns = 1, [True], [1]
                active_pairs = [witness_pair]
            elif lesson == 'assembly':
                n, swaps, turns = 2, [True, False], [2, 3]
                active_pairs = [1-witness_pair, int(rng.integers(2))]
            else:
                n = 2 if split == 'test' else int(rng.integers(1, 3))
                swaps = list(map(bool, rng.integers(0, 2, n)))
                active_pairs = list(map(int, rng.integers(0, 2, n)))
                # Training has at most one exchange per circuit. In queries,
                # exchanges on different pairs make an order-dependent cycle;
                # exchanges on the same pair cancel.
                if split == 'train' and n == 2:
                    if all(swaps):
                        swaps[int(rng.integers(2))] = False
                if split == 'test' and shift == 'cancellation':
                    swaps = [True, True]
                    active_pairs = [int(rng.integers(2))]*2
                elif split == 'test' and shift == 'composition':
                    swaps = [True, True]
                    active_pairs = list(map(int, rng.permutation((0, 1))))
                turns = None
            scene_turn = ((2 if witness_reverse else 0) if lesson == 'routing' else
                          (1 if witness_reverse else 3) if lesson == 'assembly' else None)
            for attempt in range(40):
                try:
                    pair = _build_pair(rng, n, swaps, active_pairs, turns, lesson,
                                       shift if split == 'test' else
                                       'train_occlusion' if shift == 'occlusion' else None,
                                       scene_turn)
                    if _key(pair['input']) in seen:
                        continue
                    expected, _ = _analyze(pair['input'])
                    if not np.array_equal(expected, pair['output']):
                        raise AssertionError('constructed route map disagrees with pixel inference')
                    break
                except (ValueError, StopIteration):
                    continue
            else:
                raise RuntimeError('bounded construction attempts exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    expected, a = _analyze(pair['input'])
    assert np.array_equal(expected, pair['output']), 'wrong repair or terminal completion'
    transforms = []
    for m, item in zip(a['matched'], a['arrangement']):
        turns = (item['turn']+m['undo_turn']) % 4
        transforms.append(turns)
        # Full multicolor raster, including holes and crossings, is preserved.
        assert np.array_equal(np.rot90(m['raster'], turns), item['raster'])
    terminal_cells = set()
    modes = set()
    labels = []
    sources, receivers, links = [], [], {}
    for route in a['circuit']['routes']:
        labels.append(route['label'])
        source = route['terminals'][route['source_index']]['center']
        receiver = route['terminals'][1-route['source_index']]['center']
        sources.append(source)
        receivers.append(receiver)
        links[source] = receiver
        for term in route['terminals']:
            modes.add(term['kind'])
            r, c = term['center']
            terminal_cells.update((r+dr, c+dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))
    changed_allowed = terminal_cells | set().union(*(m['cells'] for m in a['modules']))
    changed_allowed |= set().union(*(set(map(tuple, item['coords'])) for item in a['arrangement']))
    assert all(tuple(p) in changed_allowed for p in np.argwhere(pair['input'] != pair['output'])), 'fixed circuit was altered'
    for p in np.argwhere((pair['input'] == COVER)):
        assert pair['output'][tuple(p)] == COVER
    assert set(labels).isdisjoint({m['color'] for m in a['modules']})
    permutation = tuple(sorted(receivers).index(links[p]) for p in sorted(sources))
    orientation = 'vertical' if len({r for r, c in sources}) == 1 else 'horizontal'
    matched_patterns = {_key(np.where(m['canonical'] == m['color'], 1, m['canonical']))
                        for m in a['matched']}
    housing_labels = {b[0] for b in a['bars']}
    filtering_twin = any(_key(np.where(m['canonical'] == m['color'], 1, m['canonical'])) in matched_patterns
                         for m in a['modules'] if m['color'] not in housing_labels)
    return dict(shape=pair['input'].shape, sockets=len(a['bars']),
                distractors=len(a['modules'])-len(a['matched']), turns=tuple(transforms),
                module_shapes=tuple(m['canonical'].shape for m in a['matched']),
                silhouettes=tuple(_key((m['canonical'] != BG).astype(np.int8)) for m in a['matched']),
                crossings=sum(int(np.count_nonzero(m['canonical'] == CROSS)) for m in a['matched']),
                covers=len(a['circuit']['masks']),
                multi_cover=any(len(links) > 1 for _, links in a['circuit']['masks']),
                clues=modes, repeated_labels=len(set(labels)) < 3,
                permutation=permutation, orientation=orientation,
                filtering_twin=filtering_twin, analysis=a)


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, train, tests = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            f = _features(dict(input=inp, output=out))
            if split == 'train':
                train.append(f)
            else:
                tests.append(f)
    assert any(f['sockets'] == 1 and f['crossings'] == 1 and f['permutation'] in ((1, 0, 2), (0, 2, 1))
               and not f['repeated_labels'] for f in train), 'show distinct signals exchanging through a moved key'
    assert any(f['filtering_twin'] for f in train), 'show color filtering of an exact functional twin'
    assert {1, 2, 3} <= set(t for f in train for t in f['turns']), 'show three nontrivial rigid rotations'
    assert {'solid', 'diamond', 'checker', 'blank'} <= set.union(*(f['clues'] for f in train))
    assert any(f['multi_cover'] for f in train), 'show aligned continuation through a shared cover'
    assert {f['orientation'] for f in train} == {'vertical', 'horizontal'}, 'show both opposed-bank axes'
    assert {(1, 0, 2), (0, 2, 1)} <= {f['permutation'] for f in train}, 'show both distinct one-exchange functions'
    assert all(f['crossings'] <= 1 for f in train), 'reserve composed exchanges for queries'
    assert all(f['sockets'] == 2 and 'blank' in f['clues'] for f in tests), 'queries compose modules and require label propagation'


def self_test():
    """Bounded semantic audit and measured structural variation over 42 seeds."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    metrics = [_features(p) for puzzle in puzzles for s in ('train', 'test') for p in puzzle[s]]
    assert len({_key(p['input']) for puzzle in puzzles for s in puzzle for p in puzzle[s]}) == len(metrics)
    assert {m['sockets'] for m in metrics} == {1, 2}
    assert {m['distractors'] for m in metrics} == {0, 1, 2}
    assert {m['crossings'] for m in metrics} >= {0, 1, 2}
    assert {m['covers'] for m in metrics} >= {0, 1, 2}
    assert set(t for m in metrics for t in m['turns']) == set(range(4))
    assert len({m['shape'] for m in metrics}) >= 20
    assert len({s for m in metrics for s in m['silhouettes']}) >= 50
    assert len({s for m in metrics for s in m['module_shapes']}) >= 10
    assert any(m['repeated_labels'] for m in metrics)
    assert any(m['multi_cover'] for m in metrics)
    assert {m['permutation'] for m in metrics} == {(0, 1, 2), (1, 0, 2), (0, 2, 1), (2, 0, 1), (1, 2, 0)}
    # A thin passing part is legal even though the sampler usually draws deeper
    # cabinets. Recognition must use its neck and function, not a size catalog.
    thin = np.zeros((14, 26), dtype=np.int8)
    ports = (4, 9, 14)
    for col, label in zip(ports, (2, 3, 4)):
        thin[0:3, col-1:col+2] = label
        thin[3:11, col] = WIRE
    thin[6, 4] = thin[6, 9] = BG
    thin[8, 6:8] = 1
    small = np.array([[1, 8, 1, 1, 1, 1, 8, 1],
                      [0, 0, 0, 1, 1, 0, 0, 0]], dtype=np.int8)
    thin[5:7, 17:25] = small
    thin_answer = thin.copy()
    thin_answer[5:7, 17:25] = BG
    thin_answer[6:8, 3:11] = np.where(small != BG, small, thin_answer[6:8, 3:11])
    for col, label in zip(ports, (2, 3, 4)):
        thin_answer[0:3, col-1:col+2] = np.where(DIAMOND, label, BG)
        thin_answer[11:14, col-1:col+2] = np.where(CHECKER, label, BG)
    assert np.array_equal(solve(thin), thin_answer)
    p = generate_puzzle(713, 2, 1)
    q = generate_puzzle(713, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x)
    # Transform complete scenes, not their generation parameters. These checks
    # exercise socket normals, marker orientation, and literal color equality.
    for puzzle in puzzles[:6]:
        pair = puzzle['test'][0]
        for turns in range(4):
            assert np.array_equal(solve(np.rot90(pair['input'], turns)),
                                  np.rot90(pair['output'], turns))
        assert np.array_equal(solve(np.fliplr(pair['input'])), np.fliplr(pair['output']))
        lut = np.arange(10, dtype=np.int8)
        for old, new in zip(COLORS, COLORS[1:]+COLORS[:1]):
            lut[old] = new
        assert np.array_equal(solve(lut[pair['input']]), lut[pair['output']])
    # Neither inherited mechanism can be omitted from the training witnesses.
    for puzzle in puzzles:
        fs = [_features(pair) for pair in puzzle['train']]
        witness = next(f for f in fs if f['sockets'] == 1 and f['crossings'] == 1
                       and f['multi_cover'] and not f['repeated_labels'])
        a = witness['analysis']
        try:
            _trace_and_complete(a['base'])  # no module assembly: broken routes
        except (ValueError, StopIteration):
            pass
        else:
            raise AssertionError('repair ablation unexpectedly succeeded')
        # Ignoring the crossing is a junction flood, joining distinct labels.
        assert len({r['label'] for r in a['circuit']['routes']}) == 3
        allwire = set(a['circuit']['graph'])
        # Graph connectivity also includes jumps across opaque rectangles.
        reachable, stack = set(), [tuple(np.argwhere(a['repaired'] == CROSS)[0])]
        while stack:
            v = stack.pop()
            if v in reachable:
                continue
            reachable.add(v)
            stack.extend(q for q, _ in a['circuit']['graph'][v].values())
        flooded_labels = {r['label'] for r in a['circuit']['routes'] if set(r['path']) & reachable}
        assert len(flooded_labels) == 2, 'crossing-flood ablation should merge two distinct signals'
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            sample = generate_puzzle(111, nt, nq)
            assert len(sample['train']) == nt and len(sample['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    damaged['test'][0]['output'][0, 0] = 9
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('damaged output accepted')
    return dict(seeds=42, pairs=len(metrics), dimensions=len({m['shape'] for m in metrics}),
                silhouettes=len({s for m in metrics for s in m['silhouettes']}),
                module_dimensions=len({s for m in metrics for s in m['module_shapes']}),
                permutations=len({m['permutation'] for m in metrics}))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
