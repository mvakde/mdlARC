"""Commission a relay mosaic by joint exact cover and directed continuity.

Only standard-library modules and NumPy are used. Solving has no generation state.
"""
from __future__ import annotations

import json
import numpy as np

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid with axes 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _components(mask):
    remaining = {tuple(map(int, x)) for x in np.argwhere(mask)}
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        comp, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    comp.add(q)
                    stack.append(q)
        result.append(comp)
    return result


def _port(tile, color):
    cells = [tuple(map(int, x)) for x in np.argwhere(tile == color)]
    h, w = tile.shape
    if min(h, w) < 3 or not 1 <= len(cells) <= 3:
        raise ValueError("Module dimensions or port span are invalid")
    sides = [d for d, ok in enumerate((all(r == 0 for r, c in cells),
             all(c == w - 1 for r, c in cells),
             all(r == h - 1 for r, c in cells),
             all(c == 0 for r, c in cells))) if ok]
    if len(sides) != 1:
        raise ValueError("A port must identify exactly one side")
    d = sides[0]
    span = sorted(c if d % 2 == 0 else r for r, c in cells)
    edge_length = w if d % 2 == 0 else h
    if (span != list(range(span[0], span[0] + len(span)))
            or span[0] == 0 or span[-1] == edge_length - 1):
        raise ValueError("Port must be contiguous and exclude corners")
    return d, cells


def _extract(tray, vacancy, port, forbidden):
    if any(np.any(tray == c) for c in forbidden):
        raise ValueError("Tray contains a reserved board color")
    tiles = []
    for comp in _components(tray != vacancy):
        rr, cc = zip(*comp)
        a = tray[min(rr):max(rr) + 1, min(cc):max(cc) + 1].copy()
        if a.size != len(comp) or np.any(a == vacancy):
            raise ValueError("Source objects must be filled rectangles")
        _port(a, port)
        tiles.append(a)
    if not 2 <= len(tiles) <= 8:
        raise ValueError("Expected two through eight supplied modules")
    return tiles


def _board_info(board, tray, port, side):
    h, w = board.shape
    if min(h, w) < 8:
        raise ValueError("Board is too small")
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    bg = int(board[0, 0])
    vacancy = int(tray[0, 0])
    if (any(int(board[q]) != bg for q in corners)
            or any(int(tray[q]) != vacancy for q in
                   ((0, 0), (0, tray.shape[1] - 1),
                    (tray.shape[0] - 1, 0),
                    (tray.shape[0] - 1, tray.shape[1] - 1)))
            or len({bg, vacancy, port}) != 3):
        raise ValueError("Background roles must be visible in panel corners")
    other = set(map(int, np.unique(board))) - {bg, vacancy, port}
    if len(other) != 1:
        raise ValueError("Board must show exactly four local roles")
    signal = other.pop()
    tail = {tuple(map(int, x)) for x in np.argwhere(board == signal)}
    if not 2 <= len(tail) <= 4:
        raise ValueError("Invalid launcher length")
    edge = [q for q in tail if q[0] in (0, h - 1) or q[1] in (0, w - 1)]
    if len(edge) != 1:
        raise ValueError("Launcher must start at one noncorner boundary cell")
    start = edge[0]
    initial = 2 if start[0] == 0 else 0 if start[0] == h - 1 else 1 if start[1] == 0 else 3
    dr, dc = DIRS[initial]
    line = {(start[0] + k * dr, start[1] + k * dc) for k in range(len(tail))}
    cap = start[0] + len(tail) * dr, start[1] + len(tail) * dc
    if line != tail or board[cap] != port:
        raise ValueError("Tail must lead inward to the marked cap")
    goals = {tuple(map(int, x)) for x in np.argwhere(board == port)} - {cap}
    if not 1 <= len(goals) <= 3:
        raise ValueError("Expected one marked exit strip")
    gs = [d for d, ok in enumerate((all(r == 0 for r, c in goals),
          all(c == w - 1 for r, c in goals),
          all(r == h - 1 for r, c in goals),
          all(c == 0 for r, c in goals))) if ok]
    if len(gs) != 1 or goals & set(corners):
        raise ValueError("Exit must lie on one noncorner boundary side")
    span = sorted(c if gs[0] % 2 == 0 else r for r, c in goals)
    if span != list(range(span[0], span[0] + len(span))):
        raise ValueError("Exit must be contiguous")
    mask = board == vacancy
    if not mask.any() or mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        raise ValueError("Vacancies must have an intact outer board margin")
    tiles = _extract(tray, vacancy, port, {bg, signal})
    return dict(board=board, tray=tray, bg=bg, vacancy=vacancy, port=port,
                signal=signal, tail=tail, cap=cap, initial=initial,
                goals=goals, mask=mask, tiles=tiles, side=side)


def _panel_infos(inp):
    # A uniform full-span divider, in the port color, explicitly marks the cut.
    for axis in (0, 1):
        for k in range(1, inp.shape[axis] - 1):
            line = inp[k, :] if axis == 0 else inp[:, k]
            if not np.all(line == line[0]):
                continue
            p = int(line[0])
            a, b = (inp[:k], inp[k + 1:]) if axis == 0 else (inp[:, :k], inp[:, k + 1:])
            for board, tray, side in ((a, b, 'top' if axis == 0 else 'left'),
                                     (b, a, 'bottom' if axis == 0 else 'right')):
                try:
                    yield _board_info(board, tray, p, side)
                except (ValueError, IndexError):
                    continue


def _route(info, modules):
    """Input solver: project rectangles onto lanes and choose nearest contacts."""
    out = np.full_like(info['board'], info['bg'])
    for m in modules:
        y, x = m['y'], m['x']
        h, w = m['tile'].shape
        out[y:y + h, x:x + w] = m['tile']
    for q in info['tail'] | {info['cap']}:
        out[q] = info['signal']
    origins, direction, visited = [info['cap']], info['initial'], []
    widths, transitions, gaps, segments = [], [], [], []
    H, W = out.shape
    for _ in range(len(modules) + 1):
        dr, dc = DIRS[direction]
        stops = []
        for r, c in origins:
            limit = r if direction == 0 else W - 1 - c if direction == 1 else H - 1 - r if direction == 2 else c
            hits = [(limit, -1)]
            for i, m in enumerate(modules):
                y, x = m['y'], m['x']
                h, w = m['tile'].shape
                if direction == 0 and x <= c < x + w and y + h - 1 < r:
                    hits.append((r - y - h + 1, i))
                elif direction == 2 and x <= c < x + w and y > r:
                    hits.append((y - r, i))
                elif direction == 1 and y <= r < y + h and x > c:
                    hits.append((x - c, i))
                elif direction == 3 and y <= r < y + h and x + w - 1 < c:
                    hits.append((c - x - w + 1, i))
            stops.append(min(hits))
        if len(set(stops)) != 1 or stops[0][0] <= 0:
            return None
        distance, i = stops[0]
        segment = {(r + k * dr, c + k * dc) for r, c in origins
                   for k in range(1, distance + (i == -1))}
        for q in segment:
            out[q] = info['signal']
        segments.append(segment)
        if i == -1:
            ends = {(r + distance * dr, c + distance * dc) for r, c in origins}
            if ends != info['goals'] or len(visited) != len(modules):
                return None
            return out, dict(visited=visited, widths=widths, transitions=transitions,
                             gaps=gaps, segments=segments,
                             crossing=any(a & b for j, a in enumerate(segments) for b in segments[j + 1:]))
        if i in visited:
            return None
        visited.append(i)
        m = modules[i]
        d, local = _port(m['tile'], info['port'])
        transitions.append((direction, d))
        widths.append(len(local))
        gaps.append(distance - 1)
        origins = [(m['y'] + r, m['x'] + c) for r, c in local]
        direction = d
    return None


def _assemble(info, max_nodes=None):
    """Enumerate finite-inventory covers; retain only commissioned circuits.

    Exhaustive search certifies rendered uniqueness. A generation budget may
    reject an unresolved candidate; public solve has no search cutoff.
    """
    mask, tiles = info['mask'], info['tiles']
    coords = [tuple(map(int, x)) for x in np.argwhere(mask)]
    index = {q: i for i, q in enumerate(coords)}
    groups, lookup = [], {}
    for tile in tiles:
        key = _key(tile)
        if key not in lookup:
            lookup[key] = len(groups)
            groups.append(dict(tile=tile, count=0))
        groups[lookup[key]]['count'] += 1
    placements, by_cell = [], [[] for _ in coords]
    H, W = mask.shape
    for g, group in enumerate(groups):
        tile = group['tile']
        h, w = tile.shape
        for y in range(H - h + 1):
            for x in range(W - w + 1):
                if not np.all(mask[y:y + h, x:x + w]):
                    continue
                ids = [index[r, c] for r in range(y, y + h) for c in range(x, x + w)]
                bits = sum(1 << i for i in ids)
                p = dict(g=g, y=y, x=x, tile=tile, bits=bits)
                pi = len(placements)
                placements.append(p)
                for i in ids:
                    by_cell[i].append(pi)
    remaining = [g['count'] for g in groups]
    full = (1 << len(coords)) - 1
    path, answers, geometries = [], {}, set()
    nodes = covers = 0
    exhausted = False

    def search(covered):
        nonlocal nodes, covers, exhausted
        nodes += 1
        if max_nodes is not None and nodes > max_nodes:
            exhausted = True
            return
        if covered == full:
            if not 2 <= len(path) <= 6:
                return
            covers += 1
            if len(geometries) < 2:
                filled = np.full_like(mask, -1, dtype=np.int16)
                for m in path:
                    y, x = m['y'], m['x']
                    h, w = m['tile'].shape
                    filled[y:y + h, x:x + w] = m['tile']
                geometries.add(filled[mask].tobytes())
            routed = _route(info, path)
            if routed is not None:
                out, facts = routed
                key = _key(out)
                if key not in answers:
                    answers[key] = (out, facts, [m.copy() for m in path])
            return
        if len(path) >= 6:
            return
        best = None
        for i in range(len(coords)):
            if covered & (1 << i):
                continue
            options = [pi for pi in by_cell[i] if remaining[placements[pi]['g']]
                       and not placements[pi]['bits'] & covered]
            if not options:
                return
            if best is None or len(options) < len(best):
                best = options
                if len(best) == 1:
                    break
        for pi in best:
            m = placements[pi]
            remaining[m['g']] -= 1
            path.append(m)
            search(covered | m['bits'])
            path.pop()
            remaining[m['g']] += 1
            if exhausted or len(answers) > 1:
                return

    search(0)
    if exhausted:
        raise ValueError("Unresolved exact-cover search budget")
    if len(answers) != 1:
        raise ValueError("Circuit must have exactly one rendered answer")
    out, facts, modules = next(iter(answers.values()))
    facts.update(nodes=nodes, geometric_covers=covers,
                 geometry_ambiguous=len(geometries) > 1, modules=modules,
                 installed=len(modules), surplus=len(tiles) - len(modules))
    return out, facts


def _analyze(inp, max_nodes=None):
    results = []
    for info in _panel_infos(inp):
        # The divider and visible role checks must be unique, independent of
        # the eventual output. Ambiguous/unsolved panel interpretations reject.
        results.append(info)
    if len(results) != 1:
        raise ValueError("Input must identify one board and one source tray")
    info = results[0]
    out, facts = _assemble(info, max_nodes)
    facts.update(info=info, shape=info['board'].shape, side=info['side'],
                 initial=info['initial'], roles=(info['bg'], info['vacancy'], info['port'], info['signal']))
    comps = _components(info['mask'])
    labels = {q: i for i, comp in enumerate(comps) for q in comp}
    counts = [0] * len(comps)
    for m in facts['modules']:
        counts[labels[m['y'], m['x']]] += 1
    facts['components'] = len(comps)
    facts['max_merged'] = max(counts)
    facts['nonrectangular'] = any(len(comp) != (max(r for r, c in comp) - min(r for r, c in comp) + 1)
                                * (max(c for r, c in comp) - min(c for r, c in comp) + 1) for comp in comps)
    return out, facts


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using only visible panels, inventory, ports, launcher and exit."""
    return _analyze(_grid(input_grid))[0]


def _local_port(h, w, d, start, width):
    if d == 0:
        return [(0, start + k) for k in range(width)]
    if d == 2:
        return [(h - 1, start + k) for k in range(width)]
    if d == 1:
        return [(start + k, w - 1) for k in range(width)]
    return [(start + k, 0) for k in range(width)]


def _pattern(rng, shape, local, roles, palette):
    h, w = shape
    a = int(rng.choice(palette))
    tile = np.full(shape, a, dtype=np.int8)
    yy, xx = np.indices(shape)
    kind = int(rng.integers(5))
    if len(palette) > 1:
        b = int(rng.choice([c for c in palette if c != a]))
        if kind == 1:
            tile[(yy + xx) % 2 == 1] = b
        elif kind == 2:
            tile[yy % 2 == 1] = b
        elif kind == 3:
            tile[xx % 2 == 1] = b
        elif kind == 4:
            tile[0, 0] = b
    for q in local:
        tile[q] = roles[2]
    return tile


def _scatter(rng, tiles, height, width, vacancy):
    for _ in range(24):
        tray = np.full((height, width), vacancy, dtype=np.int8)
        occupied = np.zeros((height, width), dtype=bool)
        order = sorted(rng.permutation(len(tiles)), key=lambda i: -tiles[int(i)].size)
        for ii in order:
            tile = tiles[int(ii)]
            h, w = tile.shape
            options = []
            for y in range(1, height - h):
                for x in range(1, width - w):
                    if not occupied[y - 1:y + h + 1, x - 1:x + w + 1].any():
                        options.append((y, x))
            if not options:
                break
            y, x = options[int(rng.integers(len(options)))]
            tray[y:y + h, x:x + w] = tile
            occupied[y:y + h, x:x + w] = True
        else:
            return tray
    return None


def _raster_expected(info, modules):
    """Constructor check uses moving fronts, independent of lane projection."""
    out = info['board'].copy()
    owner = np.full(out.shape, -1, dtype=int)
    for i, m in enumerate(modules):
        y, x = m['y'], m['x']
        h, w = m['tile'].shape
        out[y:y + h, x:x + w] = m['tile']
        owner[y:y + h, x:x + w] = i
    for q in info['tail'] | {info['cap']}:
        out[q] = info['signal']
    d, origins, reached = info['initial'], [info['cap']], []
    for _ in range(out.size * (len(modules) + 1)):
        dr, dc = DIRS[d]
        front = [(r + dr, c + dc) for r, c in origins]
        if any(not (0 <= r < out.shape[0] and 0 <= c < out.shape[1]) for r, c in front):
            return None
        if set(front) == info['goals']:
            if len(reached) != len(modules):
                return None
            for q in front:
                out[q] = info['signal']
            return out
        hits = {int(owner[q]) for q in front}
        if hits != {-1}:
            if len(hits) != 1 or -1 in hits:
                return None
            i = hits.pop()
            if i in reached:
                return None
            reached.append(i)
            m = modules[i]
            d, local = _port(m['tile'], info['port'])
            origins = [(m['y'] + r, m['x'] + c) for r, c in local]
        else:
            for q in front:
                out[q] = info['signal']
            origins = front
    return None


def _layout(rng, n, gap_count):
    H, W = int(rng.integers(18, 31)), int(rng.integers(12, 19))
    common = int(rng.integers(3, 6)), int(rng.integers(3, 6))
    dims = [common, common]
    for _ in range(n - 2):
        dims.append(common if rng.random() < .6 else (int(rng.integers(3, 7)), int(rng.integers(3, 7))))
    gap_indices = set(map(int, rng.choice(n - 1, size=gap_count, replace=False)))
    h, w = dims[0]
    rects = [(int(rng.integers(1, H - h)), int(rng.integers(1, W - w)), h, w)]
    ports, protected = [], set()
    occupancy = np.zeros((H, W), dtype=bool)
    y, x, h, w = rects[0]
    occupancy[y:y + h, x:x + w] = True
    incoming = None
    for i in range(n - 1):
        y, x, h, w = rects[-1]
        nh, nw = dims[i + 1]
        possibilities = []
        for _ in range(160):
            d = int(rng.integers(4))
            if incoming is not None and d == (incoming + 2) % 4:
                continue
            length = w if d % 2 == 0 else h
            bw = int(rng.integers(1, min(3, length - 2) + 1))
            start = int(rng.integers(1, length - bw))
            local = _local_port(h, w, d, start, bw)
            port = [(y + r, x + c) for r, c in local]
            gap = int(rng.integers(1, 7)) if i in gap_indices else 0
            if d % 2 == 0:
                yy = y - gap - nh if d == 0 else y + h + gap
                lo, hi = x + start + bw - nw, x + start
                xx = int(rng.integers(lo, hi + 1))
            else:
                xx = x + w + gap if d == 1 else x - gap - nw
                lo, hi = y + start + bw - nh, y + start
                yy = int(rng.integers(lo, hi + 1))
            if yy < 1 or xx < 1 or yy + nh >= H or xx + nw >= W:
                continue
            cells = {(r, c) for r in range(yy, yy + nh) for c in range(xx, xx + nw)}
            if occupancy[yy:yy + nh, xx:xx + nw].any() or cells & protected:
                continue
            dr, dc = DIRS[d]
            approach = {(r + k * dr, c + k * dc) for r, c in port for k in range(1, gap + 1)}
            if any(occupancy[q] for q in approach):
                continue
            possibilities.append(((yy, xx, nh, nw), (d, local), approach))
        if not possibilities:
            return None
        rect, p, approach = possibilities[int(rng.integers(len(possibilities)))]
        rects.append(rect)
        ports.append(p)
        protected |= approach
        yy, xx, nh, nw = rect
        occupancy[yy:yy + nh, xx:xx + nw] = True
        incoming = p[0]
    # Last emission must reach an unobstructed board boundary.
    y, x, h, w = rects[-1]
    endings = []
    for d in range(4):
        if incoming is not None and d == (incoming + 2) % 4:
            continue
        length = w if d % 2 == 0 else h
        for bw in range(1, min(3, length - 2) + 1):
            for start in range(1, length - bw):
                local = _local_port(h, w, d, start, bw)
                global_port = [(y + r, x + c) for r, c in local]
                dr, dc = DIRS[d]
                ray, ends = set(), set()
                clear = True
                for rr, cc in global_port:
                    rr, cc = rr + dr, cc + dc
                    while 0 <= rr < H and 0 <= cc < W:
                        if occupancy[rr, cc]:
                            clear = False
                            break
                        ray.add((rr, cc))
                        if rr in (0, H - 1) or cc in (0, W - 1):
                            ends.add((rr, cc))
                        rr, cc = rr + dr, cc + dc
                if clear:
                    endings.append(((d, local), ends, ray))
    if not endings:
        return None
    p, goals, final_ray = endings[int(rng.integers(len(endings)))]
    ports.append(p)
    # Select the entry independently; it must hit the intended first module.
    y, x, h, w = rects[0]
    entries = []
    for d in range(4):
        if ports[0][0] == (d + 2) % 4:
            continue
        transverse = range(x, x + w) if d % 2 == 0 else range(y, y + h)
        for lane in transverse:
            if d == 0:
                start, distance = (H - 1, lane), H - y - h
            elif d == 2:
                start, distance = (0, lane), y
            elif d == 1:
                start, distance = (lane, 0), x
            else:
                start, distance = (lane, W - 1), W - x - w
            if distance < 4:
                continue
            dr, dc = DIRS[d]
            approach = [(start[0] + k * dr, start[1] + k * dc) for k in range(distance)]
            if any(occupancy[q] for q in approach) or set(approach[:5]) & goals:
                continue
            length = int(rng.integers(2, min(4, distance - 2) + 1))
            tail, cap = set(approach[:length]), approach[length]
            if tail & final_ray or cap in final_ray:
                continue
            entries.append((tail, cap, d))
    if not entries:
        return None
    tail, cap, initial = entries[int(rng.integers(len(entries)))]
    return H, W, rects, ports, tail, cap, initial, goals


def _fallback_layout(rng, n, gap_count):
    # Vertical run with optional gaps. Repeated middle modules are identical,
    # and the exit module can geometrically occupy any slot but routes only last.
    H, W = 30, 15
    indices = set(map(int, rng.choice(n - 1, size=gap_count, replace=False)))
    y, x = 5, int(rng.integers(3, 9))
    rects, ports = [], []
    for i in range(n):
        rects.append((y, x, 3, 4))
        ports.append((1, [(1, 3)]) if i == n - 1 else (2, [(2, 1), (2, 2)]))
        y += 3 + (1 if i in indices else 0)
    last_y = rects[-1][0]
    tail, cap = {(0, x + 1), (1, x + 1)}, (2, x + 1)
    return H, W, rects, ports, tail, cap, 2, {(last_y + 1, W - 1)}


def _make_pair(rng, n, gap_count, surplus, roles, rotation, reverse, witness):
    bg, vacancy, port, signal = roles
    palette = [c for c in map(int, rng.permutation(10)) if c not in roles]
    palette = palette[:int(rng.integers(1, 5))]
    for attempt in range(121):
        fallback = attempt == 120
        layout = _fallback_layout(rng, n, gap_count) if fallback else _layout(rng, n, gap_count)
        if layout is None:
            continue
        H, W, rects, ports, tail, cap, initial, goals = layout
        board = np.full((H, W), bg, dtype=np.int8)
        cache, modules, tiles = {}, [], []
        for (y, x, h, w), (d, local) in zip(rects, ports):
            key = (h, w, d, tuple(local))
            if key not in cache:
                cache[key] = _pattern(rng, (h, w), local, roles, palette)
            tile = cache[key].copy()
            modules.append(dict(y=y, x=x, tile=tile))
            tiles.append(tile)
            board[y:y + h, x:x + w] = vacancy
        for q in tail:
            board[q] = signal
        for q in goals | {cap}:
            board[q] = port
        # Surplus is a whole marked module, often of an installed size.
        for _ in range(surplus):
            if fallback:
                shape, d, local = (3, 4), 0, [(0, 1)]
            else:
                shape = tiles[int(rng.integers(n))].shape if rng.random() < .75 else (3, 3)
                d = int(rng.integers(4))
                length = shape[1] if d % 2 == 0 else shape[0]
                start = int(rng.integers(1, length - 1))
                local = _local_port(*shape, d, start, 1)
            tiles.append(_pattern(rng, shape, local, roles, palette))
        tray_width = 30 - W - 1 if fallback else int(rng.integers(max(8, max(t.shape[1] for t in tiles) + 2), 30 - W))
        tray = _scatter(rng, tiles, H, tray_width, vacancy)
        if tray is None:
            continue
        info = dict(board=board, bg=bg, vacancy=vacancy, port=port, signal=signal,
                    tail=tail, cap=cap, initial=initial, goals=goals)
        expected = _raster_expected(info, modules)
        if expected is None:
            continue
        divider = np.full((H, 1), port, dtype=np.int8)
        inp = np.concatenate((tray, divider, board) if reverse else (board, divider, tray), axis=1)
        inp, expected = np.rot90(inp, rotation).copy(), np.rot90(expected, rotation).copy()
        try:
            predicted, facts = _analyze(inp, max_nodes=60000)
        except ValueError:
            continue
        if not np.array_equal(predicted, expected):
            raise AssertionError("Construction and input reconstruction disagree")
        if facts['installed'] != n or facts['surplus'] != surplus:
            continue
        if witness and (not facts['geometry_ambiguous'] or facts['max_merged'] < 2):
            continue
        if witness and (not any(a != b for a, b in facts['transitions'])
                        or max(facts['widths']) < 2):
            continue
        if witness and gap_count and not any(width >= 2 and gap > 0
                for width, gap in zip(facts['widths'][:-1], facts['gaps'][1:])):
            continue
        return dict(input=inp, output=expected)
    raise RuntimeError("Bounded circuit construction failed")


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer from {lo} through {hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(2))  # longer circuits, or more separated ray links
    base = list(map(int, rng.permutation(10)))
    fixed_palette = rng.random() < .3
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            training = split == 'train'
            if training:
                n = int(rng.integers(2 if i != 1 else 3, 5 if shift == 0 else 6))
                gap_count = 0 if i == 0 else 1
            else:
                n = int(rng.integers(5, 7)) if shift == 0 else int(rng.integers(3, 6))
                gap_count = int(rng.integers(0, min(3, n - 1) + 1)) if shift == 0 else 2
            surplus = 0 if training and i == 0 else 1 if training and i == 1 else int(rng.integers(2))
            roles = base[:4] if fixed_palette else list(map(int, np.roll(base, i if training and i < 2 else int(rng.integers(10)))))[:4]
            rotation, reverse = int(rng.integers(4)), bool(rng.integers(2))
            for _ in range(8):
                pair = _make_pair(rng, n, gap_count, surplus, roles, rotation, reverse,
                                  witness=training and i < 2)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError("Could not produce distinct inputs")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, records = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            predicted, facts = _analyze(inp)
            assert np.array_equal(predicted, out)
            info = facts['info']
            assert out.shape == info['board'].shape
            assert len(facts['visited']) == facts['installed']
            # Every installed matrix is copied intact, despite the signal passing
            # through its footprint conceptually. Only background paths are drawn.
            owner = np.zeros(out.shape, dtype=bool)
            used = {}
            for m in facts['modules']:
                y, x = m['y'], m['x']
                h, w = m['tile'].shape
                assert np.array_equal(out[y:y + h, x:x + w], m['tile'])
                assert not owner[y:y + h, x:x + w].any()
                owner[y:y + h, x:x + w] = True
                key = _key(m['tile'])
                used[key] = used.get(key, 0) + 1
            assert np.array_equal(owner, info['mask'])
            for key, count in used.items():
                assert count <= sum(_key(t) == key for t in info['tiles'])
            assert all(out[q] == info['signal'] for q in info['goals'] | {info['cap']} | info['tail'])
            assert not np.any(out == info['vacancy'])
            records.append((split, facts))
    tr = [f for s, f in records if s == 'train']
    te = [f for s, f in records if s == 'test']
    assert sum(f['geometry_ambiguous'] and f['max_merged'] >= 2 for f in tr) >= 2
    assert any(f['surplus'] > 0 for f in tr)
    assert any(f['surplus'] == 0 for f in tr)
    assert any(any(a != b for a, b in f['transitions']) for f in tr)
    assert any(any(width >= 2 and gap > 0 for width, gap in
                   zip(f['widths'][:-1], f['gaps'][1:])) for f in tr)
    assert any(max(f['gaps'], default=0) >= 1 for f in tr)
    # Each puzzle has either a route-length extrapolation or combines multiple
    # free-space links after demonstrations with at most one such link.
    train_links = [sum(g > 0 for g in f['gaps'][1:]) for f in tr]
    assert (min(f['installed'] for f in te) > max(f['installed'] for f in tr)
            or all(sum(g > 0 for g in f['gaps'][1:]) >= 2 for f in te)
            and max(train_links) <= 1)


def _signature(puzzle):
    return tuple((_key(np.asarray(p['input'], dtype=np.int8)), _key(np.asarray(p['output'], dtype=np.int8)))
                 for s in ('train', 'test') for p in puzzle[s])


def self_test():
    signatures, records, shapes, normalized = set(), [], set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(_signature(p))
        assert _signature(p) == _signature(generate_puzzle(seed))
        for s in ('train', 'test'):
            for pair in p[s]:
                prediction, f = _analyze(pair['input'])
                assert np.array_equal(prediction, pair['output'])
                records.append(f)
                shapes.add(f['shape'])
                roles = {}
                normalized.add((pair['input'].shape, tuple(roles.setdefault(int(c), len(roles)) for c in pair['input'].flat)))
    assert len(signatures) == 42
    assert len(shapes) >= 25 and len(normalized) >= 150
    assert {f['side'] for f in records} == {'left', 'right', 'top', 'bottom'}
    assert {f['initial'] for f in records} == set(range(4))
    assert {2, 3, 4, 5, 6} <= {f['installed'] for f in records}
    assert {1, 2} <= {v for f in records for v in f['widths']}
    assert all(any(f[k] for f in records) for k in ('nonrectangular', 'geometry_ambiguous', 'surplus'))
    assert any(f['max_merged'] >= 3 for f in records)
    assert any(f['components'] >= 3 for f in records)
    assert any(max(f['gaps']) >= 6 for f in records)
    assert {0, 1, 3} <= {(b - a) % 4 for f in records for a, b in f['transitions']}
    assert all({f['roles'][i] for f in records} == set(range(10)) for i in range(4))
    # Every legal explicit count pair and representative invalid scalar types.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(314, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Corruptions must fail rather than being used as a hint by the solver.
    p = generate_puzzle(0)
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted answer accepted")
    # Guaranteed fallback is checked even if normal random placement succeeds.
    rng = np.random.default_rng(174)
    for n in (2, 4, 6):
        for gaps in (0, min(2, n - 1)):
            layout = _fallback_layout(rng, n, gaps)
            assert len(layout[2]) == n
    return dict(seeds=42, distinct_puzzles=len(signatures), board_shapes=len(shapes),
                normalized_inputs=len(normalized), pairs=len(records))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
