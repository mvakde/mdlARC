"""Beam-programmed stencil printers. Only NumPy and the standard library are used."""
from __future__ import annotations

import hashlib
import numbers

import numpy as np


# The seven-cell arrow region is the source; its five-cell complement is the target.
_ARROW = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0]], dtype=bool)
_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _components(grid, background):
    remaining = set(map(tuple, np.argwhere(grid != background)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((int(r), int(c)))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining and grid[p] == grid[start]:
                    remaining.remove(p)
                    todo.append(p)
        result.append((int(grid[start]), tuple(sorted(cells))))
    return result


def _bbox(cells):
    rows, cols = zip(*cells)
    return min(rows), max(rows)+1, min(cols), max(cols)+1


def _frame(grid, box, color):
    r, e, c, d = box
    grid[r, c:d] = grid[e-1, c:d] = color
    grid[r:e, c] = grid[r:e, d-1] = color


def _source(elbow, direction):
    r, c = elbow
    dr, dc = direction
    return {(r, c), (r-dr, c), (r, c-dc)}


def _bar(mid, axis, length):
    r, c = mid
    return {(r + (i if axis == 1 else 0), c + (i if axis == 0 else 0))
            for i in range(-(length//2), length//2+1)}


def _mirror(cells, color):
    r, e, c, d = _bbox(cells)
    if len(cells) % 2 != 1 or not (e-r == 1 or d-c == 1):
        raise ValueError("mirror must be an odd contiguous bar or a point")
    if len(cells) != max(e-r, d-c):
        raise ValueError("gapped mirror")
    return {"cells": set(cells), "mid": ((r+e-1)//2, (c+d-1)//2),
            "axis": None if len(cells) == 1 else (0 if e-r == 1 else 1),
            "color": int(color)}


def _contacts(mirrors, position, direction):
    hits = []
    for i, m in enumerate(mirrors):
        for axis in (0, 1):
            if m["axis"] is not None and m["axis"] != axis:
                continue
            mid = list(position)
            mid[axis] += direction[axis]
            if tuple(mid) == m["mid"]:
                hits.append((i, axis))
    if len(hits) > 1:
        raise ValueError("ambiguous mirror contact")
    return hits[0] if hits else None


def _parse(input_grid):
    g = _grid(input_grid)
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] * 2 <= g.size:
        raise ValueError("background must occupy more than half the input")
    frames = []
    for color in np.unique(g):
        if color == bg:
            continue
        cells = list(map(tuple, np.argwhere(g == color)))
        r, e, c, d = _bbox(cells)
        if e-r < 7 or d-c < 7 or len(cells) != 2*(e-r+d-c)-4:
            continue
        if (np.all(g[r, c:d] == color) and np.all(g[e-1, c:d] == color)
                and np.all(g[r:e, c] == color) and np.all(g[r:e, d-1] == color)):
            frames.append(((r, e, c, d), int(color)))
    if len(frames) != 1:
        raise ValueError("need one complete frame in an exclusive color")
    box, frame_color = frames[0]
    r, e, c, d = box
    board = g[r:e, c:d].copy()
    interior = board[1:-1, 1:-1]
    sources, mirrors, receivers = [], [], []
    receiver_colors, sizes = set(), set()
    for shade, local in _components(interior, bg):
        cells = {(a+1, b+1) for a, b in local}
        a, b, x, y = _bbox(cells)
        if b-a == y-x and b-a >= 3 and (b-a) % 2 == 1 and len(cells) == (b-a)**2:
            receivers.append({"box": (a, b, x, y), "center": ((a+b-1)//2, (x+y-1)//2)})
            receiver_colors.add(shade)
            sizes.add(b-a)
        elif len(cells) == 3 and b-a == y-x == 2:
            missing = next(iter({(rr, cc) for rr in range(a, b) for cc in range(x, y)} - cells))
            elbow = (a+b-1-missing[0], x+y-1-missing[1])
            direction = (elbow[0]-missing[0], elbow[1]-missing[1])
            sources.append({"cells": cells, "elbow": elbow, "direction": direction, "color": shade})
        else:
            mirrors.append(_mirror(cells, shade))
    if not sources or not receivers or len(sizes) != 1 or len(receiver_colors) != 1:
        raise ValueError("need sources and equal odd square receivers of one neutral color")
    n, neutral = next(iter(sizes)), next(iter(receiver_colors))
    if neutral in {m["color"] for m in sources+mirrors}:
        raise ValueError("receiver color is reserved")
    # Each key is a solid 3x4 or 4x3 two-color rectangle, with a fixed source region.
    exterior = g.copy()
    exterior[r:e, c:d] = bg
    remaining = exterior != bg
    mapping, key_boxes = {}, []
    # Components ignoring color are used only for the two-color arrow tiles.
    todo = set(map(tuple, np.argwhere(remaining)))
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, cells = [start], []
        while stack:
            p = stack.pop()
            cells.append(p)
            for q in ((p[0]-1, p[1]), (p[0]+1, p[1]), (p[0], p[1]-1), (p[0], p[1]+1)):
                if q in todo:
                    todo.remove(q)
                    stack.append(q)
        colors = {int(g[p]) for p in cells}
        if len(colors) == 1:
            continue
        a, b, x, y = _bbox(cells)
        tile = g[a:b, x:y]
        if len(colors) != 2 or len(cells) != 12:
            raise ValueError("bad key tile")
        candidates = set()
        for k in range(4):
            mask = np.rot90(_ARROW, k)
            if mask.shape == tile.shape:
                src, dst = int(tile[mask][0]), int(tile[~mask][0])
                if src != dst and np.all(tile[mask] == src) and np.all(tile[~mask] == dst):
                    candidates.add((src, dst))
        if len(candidates) != 1:
            raise ValueError("unknown arrow convention")
        src, dst = candidates.pop()
        if src in mapping:
            raise ValueError("duplicate source key")
        mapping[src] = dst
        key_boxes.append((a, b, x, y))
        remaining[a:b, x:y] = False
    if not mapping or set(map(int, np.unique(g[remaining]))) != set(mapping):
        raise ValueError("one stencil and key are required per source color")
    masks, stencil_boxes = {}, []
    for src in mapping:
        cells = list(map(tuple, np.argwhere(remaining & (g == src))))
        a, b, x, y = _bbox(cells)
        if (b-a, y-x) != (n, n) or not np.all(np.isin(g[a:b, x:y], [bg, src])):
            raise ValueError("stencil bounds must match receiver size")
        masks[src] = (g[a:b, x:y] == src).copy()
        stencil_boxes.append((a, b, x, y))
    reserved = {bg, frame_color, neutral}
    if reserved & (set(mapping) | set(mapping.values())):
        raise ValueError("legend colors conflict with structural roles")
    if not {s["color"] for s in sources+mirrors} <= set(mapping):
        raise ValueError("launch and mirror colors must have legend entries")
    # Blank gaps preserve recoverable clue bounds, including disconnected stencils.
    clues = key_boxes+stencil_boxes
    for i, (a, b, x, y) in enumerate(clues):
        if not (b <= r or a >= e or y <= c or x >= d):
            raise ValueError("clue intersects frame")
        for aa, bb, xx, yy in clues[i+1:]:
            if not (b < aa or bb < a or y < xx or yy < x):
                raise ValueError("clues need a blank gap")
    return {"grid": g, "board": board, "background": bg, "frame": frame_color,
            "frame_box": box, "neutral": neutral, "n": n, "sources": sources,
            "mirrors": mirrors, "receivers": receivers, "mapping": mapping,
            "masks": masks, "key_boxes": key_boxes, "stencil_boxes": stencil_boxes}


def _run(a, *, update=True, reflect=True, reset=True, solid=False):
    """Trace original geometry; printed pixels never become new optical objects."""
    board, bg = a["board"], a["background"]
    h, w = board.shape
    output = board.copy()
    occupied = set().union(*(o["cells"] for o in a["sources"]+a["mirrors"]))
    receiver_at = {}
    for i, obj in enumerate(a["receivers"]):
        r, e, c, d = obj["box"]
        for rr in range(r, e):
            for cc in range(c, d):
                receiver_at[rr, cc] = i
    painted, printed, used_mirrors, records = {}, {}, set(), []
    for source in a["sources"]:
        direction, color = source["direction"], source["color"]
        position = source["elbow"]
        seen, touched, centered, traversals = set(), set(), set(), {}
        path, events = [], []
        while True:
            position = (position[0]+direction[0], position[1]+direction[1])
            r, c = position
            if not (1 <= r < h-1 and 1 <= c < w-1):
                break
            state = position, direction
            if state in seen:
                raise ValueError("looping ray")
            seen.add(state)
            if position in occupied:
                raise ValueError("ray hits an object instead of a midpoint face")
            hit = _contacts(a["mirrors"], position, direction) if reflect else None
            receiver = receiver_at.get(position)
            if receiver is not None:
                if hit:
                    raise ValueError("mirror contact inside a receiver")
                touched.add(receiver)
                traversals.setdefault(receiver, []).append((position, direction))
                if position == a["receivers"][receiver]["center"]:
                    if receiver in printed:
                        raise ValueError("receiver visited more than once")
                    if color not in a["mapping"]:
                        raise ValueError("no legend for current state at receiver")
                    dst = a["mapping"][color]
                    printed[receiver] = (color, dst)
                    centered.add(receiver)
                    rr, ee, cc, dd = a["receivers"][receiver]["box"]
                    mask = np.ones((a["n"], a["n"]), bool) if solid else a["masks"][color]
                    output[rr:ee, cc:dd] = np.where(mask, dst, bg)
                    events.append(("print", receiver, color, dst, position))
                    if update:
                        color = dst
            else:
                if hit:
                    i, axis = hit
                    new_color = a["mirrors"][i]["color"]
                    events.append(("mirror", i, color, new_color, position, axis))
                    used_mirrors.add(i)
                    if reset:
                        color = new_color
                    direction = tuple(-x if k == axis else x for k, x in enumerate(direction))
                if position in painted and painted[position] != color:
                    raise ValueError("conflicting ray colors")
                painted[position] = color
                output[position] = color
            path.append((position, direction, color))
        if touched != centered:
            raise ValueError("receiver must be traversed through its center")
        for i, visits in traversals.items():
            mid, z = a["receivers"][i]["center"], a["n"]//2
            dr, dc = visits[0][1]
            expected = [((mid[0]+j*dr, mid[1]+j*dc), (dr, dc)) for j in range(-z, z+1)]
            if visits != expected:
                raise ValueError("each receiver needs one complete centered diagonal traversal")
        records.append({"path": path, "events": events})
    return output, {"printed": printed, "used_mirrors": used_mirrors, "routes": records,
                    "painted": painted}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solution; no examples, seed, sampled route or global state."""
    a = _parse(input_grid)
    return _run(a)[0]


def _geometry(h, w, sources, mirrors):
    """Construction-time empty-board geometry, before receivers or legends exist."""
    occupied = set().union(*(o["cells"] for o in sources+mirrors))
    routes, used = [], set()
    for source in sources:
        p, direction, seen, route = source["elbow"], source["direction"], set(), []
        while True:
            p = (p[0]+direction[0], p[1]+direction[1])
            if not (1 <= p[0] < h-1 and 1 <= p[1] < w-1):
                break
            if (p, direction) in seen or p in occupied:
                raise ValueError("bad optical route")
            seen.add((p, direction))
            hit = _contacts(mirrors, p, direction)
            route.append((p, direction, hit))
            if hit:
                i, axis = hit
                used.add(i)
                direction = tuple(-x if k == axis else x for k, x in enumerate(direction))
        routes.append(route)
    if used != set(range(len(mirrors))):
        raise ValueError("unused mirror")
    return routes


def _clear(cells, h, w, occupied, gap=True):
    if not all(1 <= r < h-1 and 1 <= c < w-1 for r, c in cells):
        return False
    if cells & occupied:
        return False
    return not gap or not any((r+dr, c+dc) in occupied for r, c in cells
                              for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)))


def _optics(rng, h, w, first_axis, wanted):
    dr, dc = _DIRS[int(rng.integers(4))]
    # Start near a corner, but vary offsets rather than pinning four corner layouts.
    r = int(rng.integers(2, 5))
    c = int(rng.integers(2, 5))
    r = r if dr > 0 else h-1-r
    c = c if dc > 0 else w-1-c
    sources = [{"elbow": (r, c), "direction": (dr, dc),
                "cells": _source((r, c), (dr, dc)), "color": 3}]
    mirrors = []
    for bounce in range(wanted):
        route = _geometry(h, w, sources, mirrors)[0]
        indices = [i for i, q in enumerate(route) if q[2]]
        tail = route[indices[-1]+1:] if indices else route
        options = [(p, direction, axis, length) for p, direction, _ in tail
                   for axis in ([first_axis] if bounce == 0 else [0, 1])
                   for length in (1, 3, 5, 7)]
        rng.shuffle(options)
        occupied = set().union(*(o["cells"] for o in sources+mirrors))
        accepted = False
        for p, direction, axis, length in options[:160]:
            mid = list(p)
            mid[axis] += direction[axis]
            cells = _bar(tuple(mid), axis, length)
            if not _clear(cells, h, w, occupied):
                continue
            trial = mirrors+[_mirror(cells, 3)]
            try:
                candidate = _geometry(h, w, sources, trial)[0]
            except ValueError:
                continue
            new_hits = [(pos, hit) for pos, _, hit in candidate if hit]
            old_hits = [(pos, hit) for pos, _, hit in route if hit]
            if new_hits != old_hits+[(p, (len(mirrors), axis))]:
                continue
            if len(candidate) < 14:
                continue
            mirrors = trial
            accepted = True
            break
        if not accepted:
            return None
    return sources, mirrors, _geometry(h, w, sources, mirrors)


def _square(center, n):
    r, c = center
    z = n//2
    return {(rr, cc) for rr in range(r-z, r+z+1) for cc in range(c-z, c+z+1)}


def _receivers(rng, h, w, sources, mirrors, routes, n, count, teaching):
    occupied = set().union(*(o["cells"] for o in sources+mirrors))
    candidates = []
    z = n//2
    for ray, route in enumerate(routes):
        for i in range(z, len(route)-z):
            p, direction, hit = route[i]
            cells = _square(p, n)
            if hit or not _clear(cells, h, w, occupied):
                continue
            if all(route[i+j][0] == (p[0]+j*direction[0], p[1]+j*direction[1])
                   and route[i+j][2] is None for j in range(-z, z+1)):
                # Other rays cannot graze the box or later hit this receiver.
                if any(q in cells for other, route2 in enumerate(routes) if other != ray for q, _, _ in route2):
                    continue
                candidates.append((ray, i, p, cells))
    for _ in range(40):
        order = rng.permutation(len(candidates))
        chosen, blocked = [], occupied.copy()
        for j in order:
            candidate = candidates[int(j)]
            if _clear(candidate[3], h, w, blocked):
                chosen.append(candidate)
                blocked.update(candidate[3])
                if len(chosen) == count:
                    break
        if len(chosen) != count:
            continue
        chosen.sort(key=lambda x: (x[0], x[1]))
        centers = {p for _, _, p, _ in chosen}
        runs, after_mirror = [], False
        for route in routes:
            run, mirror_seen = 0, False
            for p, _, hit in route:
                if hit:
                    runs.append(run)
                    run = 0
                    mirror_seen = True
                elif p in centers:
                    run += 1
                    after_mirror |= mirror_seen
            runs.append(run)
        if teaching == 0 and max(runs) < 2:
            continue
        if teaching == 1 and not after_mirror:
            continue
        # An unvisited receiver demonstrates selection by the optical route.
        ray_cells = {p for route in routes for p, _, _ in route}
        off = []
        places = [(r, c) for r in range(1+z, h-1-z) for c in range(1+z, w-1-z)]
        rng.shuffle(places)
        off_count = 1 if teaching is not None else int(rng.integers(0, 3))
        for p in (places if off_count else []):
            cells = _square(p, n)
            if not cells & ray_cells and _clear(cells, h, w, blocked):
                off.append(p)
                blocked.update(cells)
                if len(off) == off_count:
                    break
        if len(off) < off_count:
            continue
        return [p for _, _, p, _ in chosen], off, runs
    return None


def _mask(rng, n, informative=False):
    y, x = np.indices((n, n))
    kind = int(rng.integers(6))
    if kind == 0:
        m = x == y
    elif kind == 1:
        m = (x == 0) | (x == n-1) | (y == 0) | (y == n-1)
    elif kind == 2:
        m = ((x+y) % 2 == 0)
    elif kind == 3:
        m = (x == n//2) | (y == n//2)
    else:
        m = rng.random((n, n)) < rng.uniform(.25, .8)
    for edge in (m[0], m[-1], m[:, 0], m[:, -1]):
        if not edge.any():
            edge[int(rng.integers(n))] = True
    if informative:
        # An explicitly asymmetric mask, varied by rotation and optional additions.
        m = np.zeros((n, n), bool)
        m[0, :] = m[:, 0] = True
        m[-1, 1] = True
        m = np.rot90(m, int(rng.integers(4))).copy()
    return m


def _program(rng, k, max_run):
    states = list(range(3, 3+k))
    order = list(map(int, rng.permutation(states)))
    # Closed cycles, a feeder into a 2-cycle, and terminating chains are compatible.
    choices = ["cycle"]
    if k == 3:
        choices.append("feeder")
    if max_run <= k:
        choices.append("chain")
    if max_run <= 1:
        choices.append("disjoint")
    kind = str(rng.choice(choices))
    if kind == "cycle":
        mapping = {s: order[(i+1) % k] for i, s in enumerate(order)}
    elif kind == "feeder":
        mapping = {order[0]: order[1], order[1]: order[0], order[2]: order[1]}
    elif kind == "chain":
        mapping = {s: order[i+1] if i+1 < k else 6 for i, s in enumerate(order)}
    else:
        mapping = {s: 6+i for i, s in enumerate(order)}
    return mapping


def _valid_starts(mapping, count):
    result = []
    for src in mapping:
        state = src
        for _ in range(count):
            if state not in mapping:
                break
            state = mapping[state]
        else:
            result.append(src)
    return result


def _place_legend(rng, board, masks, mapping):
    """Pack clues outside the board; source/key order and spacing are independent."""
    n = next(iter(masks.values())).shape[0]
    h, w = board.shape
    # Two shelf banks, or a stencil shelf with a key bank on a perpendicular side.
    perpendicular = w+5 <= 30 and h+n+1 <= 30 and rng.random() < .45
    height = h+n+1 if perpendicular else h+n+5
    width = w+5 if perpendicular else max(w, len(mapping)*5-1)
    if height > 30 or width > 30:
        raise ValueError("legend packing overflow")
    g = np.zeros((height, width), np.int8)
    top = n+1 if perpendicular else n+5
    left = int(rng.integers(width-w+1)) if not perpendicular else 0
    g[top:top+h, left:left+w] = board
    states = list(mapping)
    def starts(length, sizes):
        slack = length-sum(sizes)-(len(sizes)-1)
        if slack < 0:
            raise ValueError("clue bank too wide")
        gaps = rng.multinomial(slack, np.full(len(sizes)+1, 1/(len(sizes)+1)))
        pos, out = int(gaps[0]), []
        for i, size in enumerate(sizes):
            out.append(pos)
            pos += size+1+int(gaps[i+1])
        return out
    order = list(map(int, rng.permutation(states)))
    for src, c in zip(order, starts(width, [n]*len(order))):
        tile = g[:n, c:c+n]
        tile[masks[src]] = src
    order = list(map(int, rng.permutation(states)))
    for src, pos in zip(order, starts(h if perpendicular else width, [3 if perpendicular else 4]*len(order))):
        key = np.where(_ARROW, src, mapping[src]).astype(np.int8)
        if perpendicular:
            g[top+pos:top+pos+3, w+1:w+5] = key
        else:
            g[n+1:n+4, pos:pos+4] = key
    return g


def _make_pair(rng, n, wanted_receivers, first_axis, teaching=None, avoid_direction=None):
    # Every loop is bounded; only the accepted pair survives in memory.
    for attempt in range(240):
        k = int(rng.integers(2, 4))
        h = int(rng.integers(18, 31-(n+5)))
        w = int(rng.integers(20, 31))
        bounces = int(rng.integers(1, 4))
        optics = _optics(rng, h, w, first_axis, bounces)
        if optics is None:
            continue
        sources, mirrors, routes = optics
        # A second independent launch is sometimes feasible in the same optical field.
        if teaching is None and rng.random() < .25:
            for _ in range(15):
                direction = _DIRS[int(rng.integers(4))]
                elbow = (int(rng.integers(2, h-2)), int(rng.integers(2, w-2)))
                cells = _source(elbow, direction)
                occupied = set().union(*(o["cells"] for o in sources+mirrors))
                if not _clear(cells, h, w, occupied):
                    continue
                source = {"cells": cells, "elbow": elbow, "direction": direction, "color": 3}
                try:
                    new_routes = _geometry(h, w, sources+[source], mirrors)
                except ValueError:
                    continue
                if len(new_routes[-1]) >= 3:
                    sources.append(source)
                    routes = new_routes
                    break
        placed = _receivers(rng, h, w, sources, mirrors, routes, n, wanted_receivers, teaching)
        if placed is None:
            continue
        centers, off, runs = placed
        mapping = _program(rng, k, max(runs))
        # Assign independent source/mirror states, constrained only by visible future nodes.
        assigned = {}
        for ray, route in enumerate(routes):
            events = [("print", p) if p in centers else ("mirror", hit[0])
                      for p, _, hit in route if p in centers or hit]
            segments = [("source", ray, 0)]
            for kind, obj in events:
                if kind == "mirror":
                    segments.append(("mirror", obj, 0))
                else:
                    kind0, obj0, amount = segments[-1]
                    segments[-1] = (kind0, obj0, amount+1)
            for kind, index, amount in segments:
                choices = _valid_starts(mapping, amount)
                if not choices:
                    break
                if (kind, index) in assigned and assigned[kind, index] not in choices:
                    break
                assigned.setdefault((kind, index), int(rng.choice(choices)))
            else:
                continue
            break
        else:
            board = np.zeros((h, w), np.int8)
            _frame(board, (0, h, 0, w), 1)
            for i, source in enumerate(sources):
                source["color"] = assigned["source", i]
                for p in source["cells"]:
                    board[p] = source["color"]
            for i, mirror in enumerate(mirrors):
                mirror["color"] = assigned["mirror", i]
                for p in mirror["cells"]:
                    board[p] = mirror["color"]
            for p in centers+off:
                for q in _square(p, n):
                    board[q] = 2
            masks = {src: _mask(rng, n, informative=(teaching is not None and i == 0))
                     for i, src in enumerate(mapping)}
            try:
                g = _place_legend(rng, board, masks, mapping)
                # Rotate/reflect the whole physical scene; no hidden key convention changes.
                rotation = 2*int(rng.integers(2)) if teaching is not None else int(rng.integers(4))
                g = np.rot90(g, rotation).copy()
                if rng.integers(2):
                    g = np.fliplr(g).copy()
                # The arrow is vertically symmetric; all whole-scene dihedral transforms
                # remain one of its four rotations.
                palette = rng.permutation(10)
                g = palette[g].astype(np.int8)
                a = _parse(g)
                if avoid_direction is not None and a["sources"][0]["direction"] == avoid_direction:
                    continue
                output, info = _run(a)
                f = _features(a, info)
                if len(info["printed"]) != wanted_receivers or len(f["incoming"]) < 2:
                    continue
                if teaching == 0 and f["max_run"] < 2:
                    continue
                if teaching == 1 and not f["reset_before_print"]:
                    continue
                if teaching is not None:
                    shown = [a["masks"][s] for s in f["incoming"]]
                    if len({m.tobytes() for m in shown}) < 2 or not any(_asymmetric(m) for m in shown):
                        continue
                if info["used_mirrors"] != set(range(len(mirrors))):
                    continue
            except ValueError:
                continue
            return {"input": g, "output": output}
    raise RuntimeError(f"bounded construction exhausted for n={n}, receivers={wanted_receivers}, teaching={teaching}")


def _features(a, info):
    axes, incoming, max_run, reset_before_print = set(), set(), 0, False
    changed_resets = 0
    for route in info["routes"]:
        run, changed_mirror = 0, False
        for event in route["events"]:
            if event[0] == "print":
                incoming.add(event[2])
                run += 1
                max_run = max(max_run, run)
                reset_before_print |= changed_mirror
            else:
                axes.add(event[5])
                run = 0
                changed_mirror = event[2] != event[3]
                changed_resets += changed_mirror
    return {"axes": axes, "incoming": incoming, "max_run": max_run,
            "reset_before_print": reset_before_print, "changed_resets": changed_resets,
            "printed": len(info["printed"]), "unvisited": len(a["receivers"])-len(info["printed"])}


def _key(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return a.shape, a.tobytes()


def _asymmetric(mask):
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, name, low, high in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    axis = int(rng.integers(2))
    shift = str(rng.choice(["depth", "scale"]))
    puzzle, seen, first_direction = {"train": [], "test": []}, set(), None
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            teaching = index if split == "train" and index < 2 else None
            n = 3
            if split == "test" and shift == "scale":
                n = 5
            elif split == "train" and teaching is None and shift == "depth" and rng.random() < .2:
                n = 5
            amount = (4 if shift == "depth" else 2 if n == 5 else 3) if split == "test" else (2 if teaching is not None else int(rng.choice([2, 3])))
            if split == "train" and n == 5:
                amount = 2
            for _ in range(8):
                pair = _make_pair(rng, n, amount, axis if teaching != 1 else 1-axis, teaching,
                                  first_direction if teaching == 1 else None)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if teaching == 0:
                        first_direction = _parse(pair["input"])["sources"][0]["direction"]
                    break
            else:
                raise RuntimeError("could not create distinct inputs")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            a = _parse(inp)
            expected, info = _run(a)
            assert np.array_equal(out, expected), "incorrect solved output"
            assert info["used_mirrors"] == set(range(len(a["mirrors"])))
            f = _features(a, info)
            assert f["printed"] >= 2 and len(f["incoming"]) >= 2
            assert out.shape == a["board"].shape
            assert set(map(int, np.unique(out))) <= set(map(int, np.unique(inp)))
            # All optical hardware and all unvisited receivers are preserved.
            assert np.all(out[0] == a["frame"]) and np.all(out[-1] == a["frame"])
            assert np.all(out[:, 0] == a["frame"]) and np.all(out[:, -1] == a["frame"])
            for obj in a["sources"]+a["mirrors"]:
                assert all(out[p] == a["board"][p] for p in obj["cells"])
            for i, obj in enumerate(a["receivers"]):
                r, e, c, d = obj["box"]
                if i not in info["printed"]:
                    assert np.all(out[r:e, c:d] == a["neutral"])
                else:
                    src, dst = info["printed"][i]
                    assert np.array_equal(out[r:e, c:d], np.where(a["masks"][src], dst, a["background"]))
            if split == "train":
                training.append((a, info, f))
    assert set().union(*(f["axes"] for _, _, f in training)) == {0, 1}
    assert len({s["direction"] for a, _, _ in training for s in a["sources"]}) >= 2
    assert any(f["max_run"] >= 2 for _, _, f in training)
    assert any(f["reset_before_print"] for _, _, f in training)
    assert any(f["unvisited"] for _, _, f in training)
    assert any(_asymmetric(a["masks"][src]) for a, info, _ in training for src, _ in info["printed"].values())
    assert any(len({a["masks"][src].tobytes() for src, _ in info["printed"].values()}) >= 2 for a, info, _ in training)
    # Counterfactuals distinguish absence of either inherited mechanism.
    for option in ({"update": False}, {"reflect": False}, {"solid": True}, {"reset": False}):
        distinguished = False
        for a, _, _ in training:
            try:
                alternative = _run(a, **option)[0]
                distinguished |= not np.array_equal(alternative, _run(a)[0])
            except ValueError:
                distinguished = True
        assert distinguished, f"training does not distinguish {option}"


def self_test():
    """42 streamed seeds, deterministic replay, count boundaries, and metamorphic checks."""
    shapes, structures, masks, signatures = set(), set(), set(), set()
    backgrounds, sizes, source_counts, print_counts, category_counts = set(), set(), set(), set(), set()
    disconnected = reset_before_print = multi_source = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprint = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _parse(pair["input"])
                out, info = _run(a)
                assert np.array_equal(out, pair["output"])
                assert np.array_equal(solve(pair["input"]), out)
                assert np.array_equal(solve(pair["input"].tolist()), out)
                shapes.add(pair["input"].shape)
                backgrounds.add(a["background"])
                sizes.add(a["n"])
                source_counts.add(len(a["sources"]))
                print_counts.add(len(info["printed"]))
                category_counts.add(len(a["mapping"]))
                f = _features(a, info)
                reset_before_print += f["reset_before_print"]
                multi_source += len(a["sources"]) > 1
                # Canonical color renaming measures spatial rather than palette diversity.
                roles = {}
                normalized = bytes(roles.setdefault(int(v), len(roles)) for v in pair["input"].flat)
                structures.add((pair["input"].shape, normalized))
                for mask in a["masks"].values():
                    masks.add((mask.shape, mask.tobytes()))
                    disconnected += len(_components(mask.astype(np.int8), 0)) > 1
                fingerprint.update(bytes(pair["input"].shape)+pair["input"].tobytes()+out.tobytes())
        signatures.add(fingerprint.digest())
        if seed < 3:
            replay = generate_puzzle(seed)
            assert all(_key(p[k]) == _key(q[k]) for s in puzzle for p, q in zip(puzzle[s], replay[s]) for k in ("input", "output"))
            for pair in puzzle["test"]:
                # Palette permutation and 90-degree scene rotation must commute with solving.
                perm = np.random.default_rng(seed+910).permutation(10)
                assert np.array_equal(solve(perm[pair["input"]]), perm[pair["output"]])
                assert np.array_equal(solve(np.rot90(pair["input"])), np.rot90(pair["output"]))
    assert len(signatures) == 42 and len(structures) >= 180
    assert len(shapes) >= 25 and len(masks) >= 35
    assert backgrounds == set(range(10)) and sizes == {3, 5}
    assert source_counts == {1, 2} and category_counts == {2, 3}
    assert {2, 3, 4} <= print_counts and disconnected >= 20
    assert reset_before_print >= 42 and multi_source >= 3
    _rule_regressions()
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            p = generate_puzzle(900+train*10+test, train, test)
            assert len(p["train"]) == train and len(p["test"]) == test
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")


def _rule_regressions():
    """Small manually specified labels: these expectations do not call the tracer."""
    masks = {3: np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0]], bool),
             4: np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], bool),
             5: np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], bool)}
    for mirror_color, terminal in ((5, False), (4, False), (5, True)):
        board = np.zeros((20, 25), np.int8)
        _frame(board, (0, 20, 0, 25), 1)
        board[1, 2] = board[2, 1] = board[2, 2] = 3
        board[12:15, 14] = mirror_color
        for r, c in ((5, 5), (9, 9), (16, 10), (5, 19)):
            board[r-1:r+2, c-1:c+2] = 2
        mapping = {3: 4, 4: 5, 5: 6 if terminal else 3}
        expected = board.copy()
        expected[4:7, 4:7] = np.where(masks[3], 4, 0)
        expected[8:11, 8:11] = np.where(masks[4], 5, 0)
        final_color = mapping[mirror_color]
        expected[15:18, 9:12] = np.where(masks[mirror_color], final_color, 0)
        for p, color in (((3, 3), 3), ((7, 7), 4), ((11, 11), 5), ((12, 12), 5),
                         ((13, 13), mirror_color), ((14, 12), mirror_color), ((18, 8), final_color)):
            expected[p] = color
        for layout_seed in (10, 20):
            inp = _place_legend(np.random.default_rng(layout_seed), board, masks, mapping)
            assert np.array_equal(solve(inp), expected), "enumerated printing/reflection regression"
            for k in (1, 2, 3):
                assert np.array_equal(solve(np.rot90(inp, k)), np.rot90(expected, k))
            # Move the first receiver sideways so the beam only clips it: invalid.
            bad = inp.copy()
            a = _parse(inp)
            r0, _, c0, _ = a["frame_box"]
            bad[r0+4:r0+7, c0+4:c0+7] = 0
            bad[r0+4:r0+7, c0+5:c0+8] = 2
            try:
                solve(bad)
            except ValueError:
                pass
            else:
                raise AssertionError("off-center receiver traversal accepted")


if __name__ == "__main__":
    self_test()
