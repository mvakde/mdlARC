"""Signal-docked rooms: local first-hit routing creates geometric jigsaw joins.

The input is an inventory, not a physical routing canvas. A room's exit beam
selects another room by its opposite-side inlet signature; that room is then
translated to the exit side. Only the visited rooms' signal traces are assembled.
See task_spec.md for the exact, fixed rule. No parent generator is imported.
"""

from functools import lru_cache
import itertools
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # north, east, south, west


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape),
             "grids must be two-dimensional with axes 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "grid colors must be integers 0..9")
    return a.astype(np.int8, copy=True)


def _step(p, d, distance=1):
    dr, dc = DIRS[d]
    return p[0] + distance * dr, p[1] + distance * dc


def _components(cells):
    remaining, parts = {tuple(map(int, p)) for p in cells}, []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        part, stack = {first}, [first]
        while stack:
            p = stack.pop()
            for d in range(4):
                q = _step(p, d)
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        parts.append(frozenset(part))
    return parts


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _edge_cell(h, w, side, position, depth=0):
    if side == 0:
        return depth, position
    if side == 1:
        return position, w - 1 - depth
    if side == 2:
        return h - 1 - depth, position
    return position, depth


def _signature(cells, d):
    return tuple(sorted(c if d % 2 == 0 else r for r, c in cells))


def _port_side(cells, port):
    r0, r1 = min(r for r, c in cells), max(r for r, c in cells)
    c0, c1 = min(c for r, c in cells), max(c for r, c in cells)
    possible = [d for d, yes in enumerate((all(r == r0 for r, c in port),
                 all(c == c1 for r, c in port), all(r == r1 for r, c in port),
                 all(c == c0 for r, c in port))) if yes]
    _require(len(possible) == 1, "a relay port must mark exactly one extreme side")
    d = possible[0]
    sig = _signature(port, d)
    _require(sig == tuple(range(sig[0], sig[0] + len(sig))), "ports must be contiguous")
    _require(all(_step(p, (d + 2) % 4) in cells - port for p in port),
             "each port pixel must have an inward carrier neighbor")
    return d


def _partition(inp):
    """Infer full-line dividers, equal room sizes, and corner background."""
    h, w = inp.shape
    candidates = []
    for value in np.unique(inp):
        rr = list(map(int, np.flatnonzero(np.all(inp == value, axis=1))))
        cc = list(map(int, np.flatnonzero(np.all(inp == value, axis=0))))
        if not rr and not cc:
            continue
        rb, cb = [-1] + rr + [h], [-1] + cc + [w]
        hs = [b - a - 1 for a, b in zip(rb, rb[1:])]
        ws = [b - a - 1 for a, b in zip(cb, cb[1:])]
        if min(hs + ws) < 5 or len(set(hs)) != 1 or len(set(ws)) != 1:
            continue
        rooms = [inp[r + 1:rb[i + 1], c + 1:cb[j + 1]].copy()
                 for i, r in enumerate(rb[:-1]) for j, c in enumerate(cb[:-1])]
        if len(rooms) < 2 or any(np.any(p == value) for p in rooms):
            continue
        corners = {int(p[r, c]) for p in rooms for r, c in
                   ((0, 0), (0, -1), (-1, 0), (-1, -1))}
        if len(corners) == 1:
            candidates.append((rooms, int(value), corners.pop(), (len(hs), len(ws))))
    _require(len(candidates) == 1, "input must have one equal-room divider partition")
    return candidates[0]


def _decode(inp):
    panels, divider, bg, layout = _partition(inp)
    h, w = panels[0].shape
    rooms, starts = [], []
    cap_colors, port_colors, carriers = set(), set(), set()
    for index, panel in enumerate(panels):
        parts = _components(map(tuple, np.argwhere(panel != bg)))
        boundary = [p for p in parts if any(r in (0, h - 1) or c in (0, w - 1)
                                           for r, c in p)]
        _require(len(boundary) == 1, "each room has one boundary inlet or launcher")
        marker = boundary[0]
        on_edge = frozenset(p for p in marker if p[0] in (0, h - 1) or p[1] in (0, w - 1))
        sides = [d for d, yes in enumerate((all(r == 0 for r, c in on_edge),
                 all(c == w - 1 for r, c in on_edge), all(r == h - 1 for r, c in on_edge),
                 all(c == 0 for r, c in on_edge))) if yes]
        _require(len(sides) == 1, "boundary marker must belong to exactly one room side")
        side = sides[0]
        sig = _signature(on_edge, side)
        _require(1 <= len(sig) <= 2, "inlet width must be one or two")
        colors = set(int(panel[p]) for p in marker)
        if len(colors) == 1:
            cap = colors.pop()
            expected = {_edge_cell(h, w, side, x, k) for x in sig for k in (0, 1)}
            _require(set(marker) == expected, "receivers are two cells deep")
            origins = tuple(_edge_cell(h, w, side, x, 1) for x in sig)
            start, signal = False, None
        else:
            _require(len(colors) == 2 and len(sig) == 1 and 2 <= len(marker) <= 4,
                     "the launcher has a one-to-three-cell tail and one cap")
            line = tuple(_edge_cell(h, w, side, sig[0], k) for k in range(len(marker)))
            _require(set(line) == set(marker), "launcher must be a straight inward stub")
            signal, cap = int(panel[line[0]]), int(panel[line[-1]])
            _require(all(int(panel[p]) == signal for p in line[:-1]) and signal != cap,
                     "bad launcher colors")
            origins, start = (line[-1],), True
            starts.append(index)
        cap_colors.add(cap)
        objects = []
        for part in parts:
            if part == marker:
                continue
            colors, counts = np.unique([panel[p] for p in sorted(part)], return_counts=True)
            _require(len(colors) == 2 and counts[0] != counts[1], "relays have a minority port")
            pc, carrier = int(colors[np.argmin(counts)]), int(colors[np.argmax(counts)])
            port = frozenset(p for p in part if panel[p] == pc)
            _require(1 <= len(port) <= 2 and len(_components(part - port)) == 1,
                     "a relay has a connected carrier and a width-1/2 port")
            out_d = _port_side(part, port)
            _require(all(0 < r < h - 1 and 0 < c < w - 1 for r, c in part),
                     "relays must stay strictly inside a room")
            objects.append(dict(cells=part, port=port, direction=out_d, carrier=carrier))
            port_colors.add(pc)
            carriers.add(carrier)
        _require(objects, "every room must contain at least one relay")
        rooms.append(dict(panel=panel, marker=marker, side=side, signature=sig,
                          origins=origins, direction=(side + 2) % 4, objects=objects,
                          start=start, signal=signal))
    _require(len(starts) == 1 and len(cap_colors) == 1 and len(port_colors) == 1,
             "one launcher and common cap/port roles are required")
    signal = rooms[starts[0]]["signal"]
    cap, pc = next(iter(cap_colors)), next(iter(port_colors))
    _require(len({bg, divider, signal, cap, pc}) == 5 and
             not carriers & {bg, divider, signal, cap, pc}, "semantic color roles must be disjoint")
    return dict(rooms=rooms, start=starts[0], divider=divider, background=bg,
                signal=signal, cap=cap, port_color=pc, carriers=tuple(sorted(carriers)),
                panel_shape=(h, w), layout=layout)


def _nearest(origins, d, objects, shape):
    """Project original footprints onto every lane; require one full-width hit."""
    h, w = shape
    dr, dc = DIRS[d]
    stops = []
    for r, c in origins:
        edge = r + 1 if d == 0 else w - c if d == 1 else h - r if d == 2 else c + 1
        hits = [(edge, -1)]
        for i, obj in enumerate(objects):
            for rr, cc in obj["cells"]:
                distance = (rr - r) * dr + (cc - c) * dc
                transverse = (rr - r) * dc - (cc - c) * dr
                if distance > 0 and transverse == 0:
                    hits.append((distance, i))
        stops.append(min(hits))
    _require(len(set(stops)) == 1, "partial-width or simultaneous multi-object collision")
    return stops[0]


def _trace_room(room, ignore_relays=False):
    objects = [] if ignore_relays else room["objects"]
    origins, d = room["origins"], room["direction"]
    trace, reached, transitions, gaps = set(room["marker"]), [], [], []
    shape = room["panel"].shape
    for _ in range(len(objects) + 1):
        distance, index = _nearest(origins, d, objects, shape)
        trace.update(_step(p, d, k) for p in origins for k in range(1, distance))
        if index == -1:
            exit_cells = tuple(_step(p, d, distance - 1) for p in origins)
            return dict(trace=frozenset(trace), reached=tuple(reached), exit_direction=d,
                        exit_signature=_signature(exit_cells, d), transitions=tuple(transitions),
                        gaps=tuple(gaps))
        _require(index not in reached, "local relay cycle")
        obj = objects[index]
        _require(obj["direction"] != (d + 2) % 4, "a relay cannot emit into its entry face")
        reached.append(index)
        trace.update(obj["cells"])
        transitions.append((d, obj["direction"], len(origins), len(obj["port"])))
        gaps.append(distance - 1)
        origins, d = tuple(sorted(obj["port"])), obj["direction"]
    raise ValueError("unterminated local route")


def _follow(parsed, ignore_relays=False, dock=True):
    rooms, current = parsed["rooms"], parsed["start"]
    order, positions, traces, joins = [], [], [], []
    position = (0, 0)
    for _ in range(len(rooms)):
        _require(current not in order and position not in positions, "room cycle or overlapping docking")
        order.append(current)
        positions.append(position)
        result = _trace_room(rooms[current], ignore_relays)
        traces.append(result)
        d, sig = result["exit_direction"], result["exit_signature"]
        matches = [i for i, room in enumerate(rooms) if i not in order and
                   not room["start"] and room["side"] == (d + 2) % 4 and room["signature"] == sig]
        _require(len(matches) <= 1, "exit must match at most one unused inlet")
        if not matches or not dock:
            return dict(order=tuple(order), positions=tuple(positions), traces=tuple(traces),
                        joins=tuple(joins))
        joins.append((position, d, sig))
        position, current = _step(position, d), matches[0]
    raise ValueError("unterminated room route")


def _render(parsed, route):
    h, w = parsed["panel_shape"]
    positions = route["positions"]
    r0, c0 = min(r for r, c in positions), min(c for r, c in positions)
    nr = max(r for r, c in positions) - r0 + 1
    nc = max(c for r, c in positions) - c0 + 1
    shape = nr * (h + 1) - 1, nc * (w + 1) - 1
    _require(all(x <= 30 for x in shape), "assembled map exceeds 30")
    out = np.full(shape, parsed["background"], dtype=np.int8)
    out[h::h + 1, :] = parsed["divider"]
    out[:, w::w + 1] = parsed["divider"]
    for (r, c), result in zip(positions, route["traces"]):
        rr, cc = (r - r0) * (h + 1), (c - c0) * (w + 1)
        for a, b in result["trace"]:
            out[rr + a, cc + b] = parsed["signal"]
    for (r, c), d, sig in route["joins"]:
        rr, cc = (r - r0) * (h + 1), (c - c0) * (w + 1)
        for x in sig:
            if d == 0:
                p = rr - 1, cc + x
            elif d == 1:
                p = rr + x, cc + w
            elif d == 2:
                p = rr + h, cc + x
            else:
                p = rr + x, cc - 1
            out[p] = parsed["signal"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode and solve only the supplied input; no training labels or state."""
    parsed = _decode(_grid(input_grid))
    return _render(parsed, _follow(parsed))


def _choose(rng, sequence):
    return sequence[int(rng.integers(len(sequence)))]


def _clear(cells, occupied):
    return all((r + dr, c + dc) not in occupied for r, c in cells
               for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def _rectangle(r, c, oh, ow, style):
    cells = {(r + a, c + b) for a in range(oh) for b in range(ow)}
    corners = ((r, c), (r, c + ow - 1), (r + oh - 1, c), (r + oh - 1, c + ow - 1))
    if style == 1:
        cells.difference_update(corners)
    elif style >= 2:
        cells.remove(corners[style - 2])
    return frozenset(cells)


def _ports(cells, side, width):
    r0, r1 = min(r for r, c in cells), max(r for r, c in cells)
    c0, c1 = min(c for r, c in cells), max(c for r, c in cells)
    span = range(c0 + 1, c1) if side % 2 == 0 else range(r0 + 1, r1)
    result = []
    for x in span:
        sig = tuple(range(x, x + width))
        if not all(y in span for y in sig):
            continue
        port = frozenset((r0 if side == 0 else r1, y) if side % 2 == 0
                         else (y, c1 if side == 1 else c0) for y in sig)
        if port <= cells and all(_step(p, (side + 2) % 4) in cells - port for p in port):
            result.append(port)
    return result


@lru_cache(maxsize=256)
def _active_options(h, w, origins, incoming, desired, final_width):
    """Pure finite geometry catalog, computed locally (never stored examples)."""
    options = []
    for oh, ow in itertools.product((3, 4), repeat=2):
        for r in range(1, h - oh):
            for c in range(1, w - ow):
                for style in range(6):
                    cells = _rectangle(r, c, oh, ow, style)
                    try:
                        distance, index = _nearest(origins, incoming, [{"cells": cells}], (h, w))
                    except ValueError:
                        continue
                    if index != 0:
                        continue
                    approach = frozenset(_step(p, incoming, k) for p in origins for k in range(1, distance))
                    sides = (desired,) if desired >= 0 else ((incoming - 1) % 4, incoming, (incoming + 1) % 4)
                    for side in sides:
                        if side == (incoming + 2) % 4:
                            continue
                        for width in ((final_width,) if final_width else (1, 2)):
                            for port in _ports(cells, side, width):
                                options.append((cells, port, side, approach, distance))
    return tuple(options)


def _construction_trace(shape, marker, origins, direction, objects):
    """Constructor's cell-front simulator; solve instead uses projection."""
    h, w = shape
    owner = {p: i for i, obj in enumerate(objects) for p in obj["cells"]}
    trace, reached = set(marker), []
    front, d = tuple(_step(p, direction) for p in origins), direction
    for _ in range(h * w * (len(objects) + 1)):
        inside = [0 <= r < h and 0 <= c < w for r, c in front]
        if not any(inside):
            edge = tuple(_step(p, (d + 2) % 4) for p in front)
            return frozenset(trace), tuple(reached), d, _signature(edge, d)
        if not all(inside):
            return None
        ids = {owner[p] for p in front if p in owner}
        if ids:
            if len(ids) != 1:
                return None
            i = next(iter(ids))
            if any(owner.get(p) != i for p in front) or i in reached:
                return None
            obj = objects[i]
            if obj["direction"] == (d + 2) % 4:
                return None
            reached.append(i)
            trace.update(obj["cells"])
            d = obj["direction"]
            front = tuple(_step(p, d) for p in sorted(obj["port"]))
        else:
            trace.update(front)
            front = tuple(_step(p, d) for p in front)
    return None


def _add_decoy(rng, h, w, objects, marker, trace, origins, incoming, behind=False):
    choices = []
    used = set(marker) | set().union(*(o["cells"] for o in objects))
    dr, dc = DIRS[incoming]
    lanes = {r * dc - c * dr for r, c in origins}
    far = max((r * dr + c * dc for r, c in objects[0]["cells"]), default=-100) if objects else -100
    for oh, ow in ((2, 3), (3, 2), (3, 3)):
        for r in range(1, h - oh):
            for c in range(1, w - ow):
                cells = _rectangle(r, c, oh, ow, 0)
                if cells & trace or not _clear(cells, used):
                    continue
                if behind and not any(a * dc - b * dr in lanes and a * dr + b * dc > far
                                      for a, b in cells):
                    continue
                for side in range(4):
                    for port in _ports(cells, side, 1):
                        choices.append(dict(cells=cells, port=port, direction=side))
    return _choose(rng, choices) if choices else None


def _make_room(rng, h, w, entry_side, sig, start, desired, hits, width,
               colors, force_decoy=False):
    bg, divider, signal, cap, pc, *carrier_pool = colors
    depth = 2 if not start or min(h, w) <= 8 else int(_choose(rng, (2, 3)))
    marker = frozenset(_edge_cell(h, w, entry_side, x, k)
                       for x in sig for k in range(depth))
    origins = tuple(_edge_cell(h, w, entry_side, x, depth - 1) for x in sig)
    incoming = (entry_side + 2) % 4
    budget = [240]

    def grow(objects, current, d, past):
        if len(objects) == hits:
            simulation = _construction_trace((h, w), marker, origins, incoming, objects)
            if simulation is None or simulation[1] != tuple(range(hits)) or simulation[2] != desired:
                return None
            trace = simulation[0]
            decoy = None
            if force_decoy or not hits or rng.random() < .30:
                decoy = _add_decoy(rng, h, w, objects, marker, trace, origins, incoming, force_decoy)
                if (force_decoy or not hits) and decoy is None:
                    return None
            all_objects = objects + ([decoy] if decoy is not None else [])
            checked = _construction_trace((h, w), marker, origins, incoming, all_objects)
            if checked is None or checked[:2] != simulation[:2]:
                return None
            return all_objects, simulation
        last = len(objects) == hits - 1
        options = _active_options(h, w, tuple(sorted(current)), d,
                                  desired if last else -1, width if last else 0)
        order = rng.permutation(len(options))
        occupied = set(marker) | set().union(*(o["cells"] for o in objects))
        for i in order:
            if budget[0] <= 0:
                break
            cells, port, side, approach, distance = options[int(i)]
            if cells & past or not _clear(cells, occupied) or approach & occupied:
                continue
            budget[0] -= 1
            try:
                nearest = _nearest(current, d, objects + [{"cells": cells}], (h, w))
            except ValueError:
                continue
            if nearest != (distance, len(objects)):
                continue
            obj = dict(cells=cells, port=port, direction=side)
            result = grow(objects + [obj], tuple(sorted(port)), side, past | approach | cells)
            if result is not None:
                return result
        return None

    built = grow([], origins, incoming, set(marker))
    if built is None:
        return None
    objects, simulation = built
    panel = np.full((h, w), bg, dtype=np.int8)
    for obj in objects:
        carrier = int(_choose(rng, carrier_pool))
        for p in obj["cells"]:
            panel[p] = pc if p in obj["port"] else carrier
    for p in marker:
        panel[p] = cap
    if start:
        for k in range(depth - 1):
            panel[_edge_cell(h, w, entry_side, sig[0], k)] = signal
    trace, reached, exit_d, exit_sig = simulation
    return panel, dict(trace=trace, reached=reached, exit_direction=exit_d, exit_signature=exit_sig)


@lru_cache(maxsize=32)
def _path_pool(n, nr, nc):
    """Enumerate bounded self-avoiding, bent room walks."""
    paths = []

    def grow(path):
        if len(path) == n:
            ds = [DIRS.index((b[0] - a[0], b[1] - a[1])) for a, b in zip(path, path[1:])]
            if n < 3 or len(set(ds)) > 1:
                paths.append(tuple(path))
            return
        for d in range(4):
            q = _step(path[-1], d)
            if 0 <= q[0] < nr and 0 <= q[1] < nc and q not in path:
                grow(path + [q])

    for p in itertools.product(range(nr), range(nc)):
        grow([p])
    return tuple(paths)


def _paths(rng, n, h, w):
    return _choose(rng, _path_pool(n, min(3, 31 // (h + 1)), min(3, 31 // (w + 1))))


def _inventory(panels, order, layout, divider):
    h, w = panels[0].shape
    nr, nc = layout
    grid = np.full((nr * (h + 1) - 1, nc * (w + 1) - 1), divider, dtype=np.int8)
    for slot, index in enumerate(order):
        r, c = divmod(slot, nc)
        grid[r * (h + 1):r * (h + 1) + h, c * (w + 1):c * (w + 1) + w] = panels[index]
    return grid


def _pair(rng, plan, colors):
    visited, total = plan["visited"], plan["total"]
    h, w = plan["shape"]
    # Retry a fixed structural request; never silently drop its teaching features.
    for _ in range(100):
        path = _paths(rng, visited, h, w)
        directions = [DIRS.index((b[0] - a[0], b[1] - a[1])) for a, b in zip(path, path[1:])]
        initial = (directions[0] + int(_choose(rng, (-1, 1)))) % 4
        transverse = w if initial % 2 == 0 else h
        entry_sig = (int(rng.integers(2, transverse - 2)),)
        entry_side = (initial + 2) % 4
        occlusion_candidates = [i for i in range(1, visited - 1)
                                if directions[i] != directions[i - 1]
                                and (w if directions[i - 1] % 2 else h) >= 10]
        if plan["occlusion"] and not occlusion_candidates:
            continue
        occlusion_at = int(_choose(rng, occlusion_candidates)) if plan["occlusion"] else -1
        two_at = int(_choose(rng, [i for i in range(visited) if i != occlusion_at])) if plan["two_hits"] else -1
        panels, results, keys, exit_keys = [], [], set(), set()
        for i in range(visited):
            if i:
                key = entry_side, entry_sig
                if key in keys:
                    break
                keys.add(key)
            incoming = (entry_side + 2) % 4
            desired = directions[i] if i < visited - 1 else (incoming + int(_choose(rng, (-1, 0, 1)))) % 4
            hits = 2 if i == two_at else 1
            if i == visited - 1 and i != two_at and rng.random() < .45:
                hits, desired = 0, incoming
            # Zero asks the finite geometry search to sample any fitting width.
            # A narrow room must not repeatedly request a geometrically impossible
            # thick turn; mandatory teaching widths remain explicit below.
            width = 2 if i == 0 and plan["wide"] else 0
            if i == 1 and plan["wide"]:
                width = 1
            room = _make_room(rng, h, w, entry_side, entry_sig, i == 0, desired, hits,
                              width, colors, i == occlusion_at)
            if room is None:
                break
            panel, result = room
            panels.append(panel)
            results.append(result)
            entry_side = (result["exit_direction"] + 2) % 4
            entry_sig = result["exit_signature"]
            exit_keys.add((entry_side, entry_sig))
        else:
            # An unused room cannot compete with a real target or terminal exit.
            forbidden = keys | exit_keys
            for unused_index in range(total - visited):
                room = None
                candidates = [(side, tuple(range(x, x + width))) for side in range(4)
                              for width in (1, 2)
                              for x in range(2, (w if side % 2 == 0 else h) - width - 1)
                              if (side, tuple(range(x, x + width))) not in forbidden]
                if plan["wide"] and unused_index == 0:
                    first_side = (results[0]["exit_direction"] + 2) % 4
                    candidates = [(side, sig) for side, sig in candidates
                                  if side == first_side and len(sig) == 2]
                rng.shuffle(candidates)
                for side, sig in candidates:
                    incoming = (side + 2) % 4
                    desired = (incoming + int(_choose(rng, (-1, 0, 1)))) % 4
                    room = _make_room(rng, h, w, side, sig, False, desired, 1,
                                      0, colors)
                    if room is not None:
                        panels.append(room[0])
                        forbidden.add((side, sig))
                        break
                if room is None:
                    break
            else:
                layouts = [(nr, total // nr) for nr in range(1, total + 1) if total % nr == 0
                           and nr * (h + 1) - 1 <= 30 and (total // nr) * (w + 1) - 1 <= 30]
                layout = _choose(rng, layouts)
                order = list(map(int, rng.permutation(total)))
                if order[:visited] == list(range(visited)):
                    order[0], order[-1] = order[-1], order[0]
                inp = _inventory(panels, order, layout, colors[1])
                joins = tuple((path[i], results[i]["exit_direction"], results[i]["exit_signature"])
                              for i in range(visited - 1))
                route = dict(positions=path, traces=tuple(results), joins=joins)
                parsed = dict(panel_shape=(h, w), background=colors[0], divider=colors[1], signal=colors[2])
                out = _render(parsed, route)
                # Input decoding must recover the constructed walk and exact trace.
                decoded = _decode(inp)
                recovered = _follow(decoded)
                if not np.array_equal(_render(decoded, recovered), out):
                    continue
                if plan["wide"] and not any(len(r["exit_signature"]) == 2 for r in results[:-1]):
                    continue
                if plan["wide"] and not _facts(decoded, recovered)["visible_cutouts"]:
                    continue
                return dict(input=inp, output=out)
    raise RuntimeError(f"fixed room-route request could not be constructed: {plan}")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _count(num_train, "num_train", 2, 10)
    _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) if seed >= 0 else int(seed) % (1 << 64))
    base = list(map(int, rng.permutation(10)))
    vary_palette = bool(rng.random() < .65)
    carrier_count = int(rng.integers(1, 4))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            witness = i if split == "train" and i < 2 else -1
            visited = int(rng.integers(5, 8)) if split == "test" else (3 + witness if witness >= 0 else int(rng.integers(2, 5)))
            total = int(_choose(rng, (6, 9))) if visited >= 5 else int(_choose(rng, (4, 6, 9)))
            if visited > total:
                total = 9
            if witness == 1:
                shape, total = _choose(rng, ((9, 10), (10, 9))), 6
            else:
                shape = tuple(map(int, rng.integers(8 if witness == 0 or split == "test" else 7,
                                                   10, size=2)))
            plan = dict(visited=visited, total=total, shape=shape, wide=witness == 0,
                        occlusion=witness == 1, two_hits=witness == 1 or
                        (shape == (9, 9) and rng.random() < .35))
            palette = list(map(int, rng.permutation(10))) if vary_palette and i != 0 else base
            # The second teaching example visibly changes every role in varying mode.
            if vary_palette and witness == 1:
                palette = list(np.roll(base, 1))
            colors = list(map(int, palette[:5 + carrier_count]))
            for _ in range(15):
                pair = _pair(rng, plan, colors)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    order = rng.permutation(num_train)
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def _facts(parsed, route):
    rooms = parsed["rooms"]
    transitions = [t for result in route["traces"] for t in result["transitions"]]
    hidden, cutouts, visible_cutouts, areas, local_counts = False, 0, 0, [], []
    for index, result in zip(route["order"], route["traces"]):
        room = rooms[index]
        local_counts.append(len(result["reached"]))
        if result["reached"]:
            first = room["objects"][result["reached"][0]]
            d = room["direction"]
            dr, dc = DIRS[d]
            lanes = {r * dc - c * dr for r, c in room["origins"]}
            far = max(r * dr + c * dc for r, c in first["cells"])
            hidden |= any(j not in result["reached"] and any(r * dc - c * dr in lanes and
                          r * dr + c * dc > far for r, c in obj["cells"])
                          for j, obj in enumerate(room["objects"]))
        for j in result["reached"]:
            cells = room["objects"][j]["cells"]
            areas.append(len(cells))
            area = (max(r for r, c in cells) - min(r for r, c in cells) + 1) * (max(c for r, c in cells) - min(c for r, c in cells) + 1)
            cutouts += int(area > len(cells))
            visible_cutouts += int(any((r, c) not in cells and (r, c) not in result["trace"]
                                   for r in range(min(r for r, c in cells), max(r for r, c in cells) + 1)
                                   for c in range(min(c for r, c in cells), max(c for r, c in cells) + 1)))
    competitors = 0
    for step, (p, d, sig) in enumerate(route["joins"]):
        used = set(route["order"][:step + 1])
        competitors += sum(i not in used and not room["start"] and room["side"] == (d + 2) % 4
                          and len(room["signature"]) == len(sig) and room["signature"] != sig
                          for i, room in enumerate(rooms))
    rs, cs = zip(*route["positions"])
    return dict(rooms=len(rooms), visited=len(route["order"]), unused=len(rooms) - len(route["order"]),
                panel_shape=parsed["panel_shape"], input_layout=parsed["layout"],
                extent=(max(rs) - min(rs) + 1, max(cs) - min(cs) + 1),
                joins=tuple((d, sig) for p, d, sig in route["joins"]),
                transitions=tuple(transitions), local_counts=tuple(local_counts),
                behind_hit=bool(hidden), cutouts=cutouts, visible_cutouts=visible_cutouts,
                same_width_competitors=competitors, areas=tuple(areas),
                local_decoys=sum(len(rooms[i]["objects"]) - len(t["reached"])
                                 for i, t in zip(route["order"], route["traces"])),
                palette=(parsed["background"], parsed["divider"], parsed["signal"], parsed["cap"], parsed["port_color"]))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _count(len(puzzle["train"]), "training count", 2, 10)
    _count(len(puzzle["test"]), "test count", 1, 4)
    _require(len(puzzle["train"]) > len(puzzle["test"]), "training count must exceed test count")
    seen, records, a_witness, b_witness = set(), {"train": [], "test": []}, False, False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            parsed = _decode(inp)
            route = _follow(parsed)
            expected = _render(parsed, route)
            _require(np.array_equal(out, expected), "output disagrees with input-only routing and docking")
            _require(set(map(int, np.unique(out))) <= {parsed["background"], parsed["divider"], parsed["signal"]},
                     "output contains unerased relay or inlet colors")
            _require(len(_components(map(tuple, np.argwhere(out == parsed["signal"])))) == 1,
                     "assembled signal must be connected, including seam bridges")
            # Exhaustively check original footprints in each visited room; empty
            # slots and erased rooms are also covered by the full render comparison.
            for i, trace in zip(route["order"], route["traces"]):
                room = parsed["rooms"][i]
                for j, obj in enumerate(room["objects"]):
                    _require((obj["cells"] <= trace["trace"]) if j in trace["reached"]
                             else not obj["cells"] & trace["trace"], "whole-footprint activation or decoy erasure failed")
            facts = _facts(parsed, route)
            records[split].append(facts)
            if split == "train":
                local_only = _render(parsed, _follow(parsed, dock=False))
                b_witness |= not np.array_equal(local_only, out)
                try:
                    straight = _render(parsed, _follow(parsed, ignore_relays=True))
                    a_witness |= not np.array_equal(straight, out)
                except ValueError:
                    pass
    train, test = records["train"], records["test"]
    _require(a_witness and b_witness, "training must contain valid counterexamples to omitting either parent")
    _require(any(f["behind_hit"] for f in train), "show first-hit occlusion in training")
    _require(any(2 in f["local_counts"] for f in train), "show two local activations before test recursion")
    _require(any(len(sig) == 2 for f in train for d, sig in f["joins"]), "show a two-cell geometric join")
    _require(any(f["unused"] for f in train) and any(f["visible_cutouts"] for f in train),
             "show room filtering and actual nonrectangular footprints")
    _require(any(f["same_width_competitors"] for f in train), "show exact positional port matching")
    _require(max(f["visited"] for f in train) < min(f["visited"] for f in test),
             "test must extend the number of docked rooms")
    demonstrated = {width for f in train for a, b, iw, width in f["transitions"]}
    _require(all({width for a, b, iw, width in f["transitions"]} <= demonstrated for f in test),
             "test port widths must be demonstrated")
    return None


def _puzzle_key(puzzle):
    return tuple((_key(p["input"]), _key(p["output"])) for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded checks over 42 seeds, legal counts, mutations and role symmetries."""
    records, shapes, masks, puzzles = [], set(), set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                frozen = inp.copy()
                _require(np.array_equal(solve(inp), out) and np.array_equal(inp, frozen), "solver mutates input or disagrees")
                parsed = _decode(inp)
                route = _follow(parsed)
                f = _facts(parsed, route)
                f["split"] = split
                records.append(f)
                shapes.add(inp.shape)
                for room in parsed["rooms"]:
                    for obj in room["objects"]:
                        r0, c0 = min(r for r, c in obj["cells"]), min(c for r, c in obj["cells"])
                        masks.add(tuple(sorted((r - r0, c - c0) for r, c in obj["cells"])))
    _require(len(set(puzzles)) == 42 and len(shapes) >= 15 and len(masks) >= 20,
             "insufficient structural variation across 42 seeds")
    _require({f["rooms"] for f in records} >= {4, 6, 9}, "inventory counts missing")
    _require({f["visited"] for f in records} >= set(range(2, 8)), "route-length variation missing")
    _require(len({f["panel_shape"] for f in records}) >= 8, "rectangular room variation missing")
    _require({d for f in records for d, sig in f["joins"]} == set(range(4)), "docking sides missing")
    _require({len(sig) for f in records for d, sig in f["joins"]} == {1, 2}, "join-width variation missing")
    _require(any(0 in f["local_counts"] for f in records), "miss rooms missing")
    _require(any(f["unused"] == 0 for f in records) and any(f["unused"] >= 4 for f in records),
             "room reachability variation missing")
    for role in range(5):
        _require({f["palette"][role] for f in records} == set(range(10)), "palette role coverage missing")
    _require(_puzzle_key(generate_puzzle(17)) == puzzles[17], "whole puzzle is nondeterministic")
    # Equivariance is a family invariant, independent of any seed schedule.
    pair = generate_puzzle(51, 2, 1)["test"][0]
    for k in range(4):
        _require(np.array_equal(solve(np.rot90(pair["input"], k)), np.rot90(pair["output"], k)),
                 "rotation must rotate the assembled map")
    palette = np.array([4, 7, 0, 9, 1, 8, 3, 6, 2, 5], dtype=np.int8)
    _require(np.array_equal(solve(palette[pair["input"]]), palette[pair["output"]]), "color-role relabeling failed")
    # Inventory locations are irrelevant; scramble rooms without rotating them.
    parsed = _decode(pair["input"])
    reordered = _inventory([room["panel"] for room in parsed["rooms"]],
                           list(reversed(range(len(parsed["rooms"])))), parsed["layout"], parsed["divider"])
    _require(np.array_equal(solve(reordered), pair["output"]), "inventory-order shortcut")
    # The supplied verifier exhausts all legal combinations. Keep this suite
    # bounded while exercising both extremes and a multiple-test case.
    for a, b in ((2, 1), (10, 1), (10, 4), (5, 3)):
        p = generate_puzzle(710 + 10 * a + b, a, b)
        _require(len(p["train"]) == a and len(p["test"]) == b, "explicit counts changed")
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(7, *counts)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    broken = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in generate_puzzle(0)[s]]
              for s in ("train", "test")}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupt output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds plus boundary count combinations)")
