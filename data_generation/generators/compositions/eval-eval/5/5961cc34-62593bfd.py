"""Paired relay portals: shape addresses determine remote ray emitters.

Only NumPy and the Python standard library are used. No files, parent modules,
generation metadata, or examples are consulted by solve().
"""
from collections import Counter
import hashlib

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
NEIGHBORS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                  if (r, c) != (0, 0))


def _shift(cells, r, c):
    return {(y + r, x + c) for y, x in cells}


def _normal(cells):
    return _shift(cells, -min(r for r, c in cells), -min(c for r, c in cells))


def _rotate(cells, k=1):
    result = set(cells)
    for _ in range(k % 4):
        result = _normal({(-c, r) for r, c in result})
    return result


def _key(cells):
    return tuple(sorted(_normal(cells)))


def _canonical(cells):
    return min(_key(_rotate(cells, k)) for k in range(4))


def _mirror(cells):
    return _normal({(r, -c) for r, c in cells})


def _connected(cells):
    if not cells:
        return False
    seen = {min(cells)}
    stack = list(seen)
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS:
            p = r + dr, c + dc
            if p in cells and p not in seen:
                seen.add(p)
                stack.append(p)
    return seen == cells


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        first = min(remaining)
        seen, stack = {first}, [first]
        remaining.remove(first)
        while stack:
            r, c = stack.pop()
            for dr, dc in NEIGHBORS:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    seen.add(p)
                    stack.append(p)
        result.append(seen)
    return result


def _halo(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _ports(mask, width):
    h = max(r for r, c in mask) + 1
    w = max(c for r, c in mask) + 1
    result = []
    for side, (dr, dc) in enumerate(DIRS):
        length = w if side % 2 == 0 else h
        for start in range(1, length - width):
            if side == 0:
                port = {(0, start + i) for i in range(width)}
            elif side == 1:
                port = {(start + i, w - 1) for i in range(width)}
            elif side == 2:
                port = {(h - 1, start + i) for i in range(width)}
            else:
                port = {(start + i, 0) for i in range(width)}
            if (port <= mask and len(mask - port) > width and
                    all((r - dr, c - dc) in mask - port for r, c in port)
                    and _connected(mask - port)):
                result.append((side, port))
    return result


def _random_mask(rng, limit=6):
    h, w = map(int, rng.integers(3, limit + 1, size=2))
    kind = int(rng.integers(6))
    mask = {(r, c) for r in range(h) for c in range(w)}
    if kind == 0:
        a, b = int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))
        mask -= {(r, c) for r in range(a) for c in range(b)}
    elif kind == 1:
        mask = {(r, c) for r, c in mask if r >= h - 2 or c < 2}
    elif kind == 2 and min(h, w) >= 5:
        mask -= {(r, c) for r in range(2, h - 2) for c in range(2, w - 2)}
    elif kind == 3:
        start = int(rng.integers(1, w - 1))
        mask -= {(r, start) for r in range(int(rng.integers(1, h - 1)))}
    elif kind >= 4:
        # Cell-level deletions keep many asymmetric, notched silhouettes.
        for _ in range(int(rng.integers(1, max(2, h * w // 3)))):
            p = sorted(mask)[int(rng.integers(len(mask)))]
            trial = mask - {p}
            if _connected(trial):
                mask = trial
    return _rotate(_normal(mask), int(rng.integers(4)))


def _sample_mask(rng, forbidden, *, chiral=False, orbit4=False):
    def allowed(mask):
        key = _canonical(mask)
        return (key not in forbidden and _canonical(_mirror(mask)) not in forbidden
                and (not chiral or key != _canonical(_mirror(mask)))
                and (not orbit4 or len({_key(_rotate(mask, k)) for k in range(4)}) == 4)
                and bool(_ports(mask, 1)) and bool(_ports(mask, 2)))

    for _ in range(140):
        mask = _random_mask(rng)
        if allowed(mask):
            return mask
    # Finite shape fallback with many distinct areas and corner cuts.
    for h in range(4, 7):
        for w in range(4, 7):
            for a in range(1, h - 1):
                for b in range(1, w - 1):
                    mask = {(r, c) for r in range(h) for c in range(w)
                            if not (r < a and c < b)}
                    if allowed(mask):
                        return mask
    raise RuntimeError("shape vocabulary exhausted")


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected an integer 1..30 grid with colors 0..9")
    return a


def _decode(value):
    a = _grid(value)
    h, w = a.shape
    values, counts = np.unique(a, return_counts=True)
    bg = int(values[int(np.argmax(counts))])
    if int(counts.max()) <= a.size // 2:
        raise ValueError("background must occupy a majority")
    if any(int(a[p]) != bg for p in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))):
        raise ValueError("corners must be background")
    components = _components({tuple(map(int, p)) for p in np.argwhere(a != bg)})
    edge = lambda p: p[0] in (0, h - 1) or p[1] in (0, w - 1)
    launchers = [obj for obj in components if any(edge(p) for p in obj)]
    if len(launchers) != 1:
        raise ValueError("need one boundary launcher")
    launcher = launchers[0]
    boundary = [p for p in launcher if edge(p)]
    if len(boundary) != 1 or not 3 <= len(launcher) <= 5:
        raise ValueError("invalid launcher length or boundary contact")
    start = boundary[0]
    direction = 2 if start[0] == 0 else 0 if start[0] == h - 1 else 1 if start[1] == 0 else 3
    dr, dc = DIRS[direction]
    line = [(start[0] + k * dr, start[1] + k * dc) for k in range(len(launcher))]
    if set(line) != launcher:
        raise ValueError("launcher must be straight")
    signal, cap_color = int(a[start]), int(a[line[-1]])
    if signal == cap_color or any(int(a[p]) != signal for p in line[:-1]):
        raise ValueError("launcher tail/cap colors are ambiguous")
    objects, port_colors, body_colors = [], set(), set()
    for cells in components:
        if cells == launcher:
            continue
        colors = Counter(int(a[p]) for p in cells)
        if len(colors) != 2:
            raise ValueError("relay requires carrier and minority port")
        carrier, pc = sorted(colors, key=lambda c: (-colors[c], c))
        if colors[carrier] <= colors[pc]:
            raise ValueError("port must be a strict minority")
        r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
        mask, port = _shift(cells, -r0, -c0), {p for p in cells if int(a[p]) == pc}
        local_port = _shift(port, -r0, -c0)
        candidates = [(s, p) for s, p in _ports(mask, len(port)) if p == local_port]
        if len(candidates) != 1 or any(edge(p) for p in cells):
            raise ValueError("invalid or ambiguously sided relay port")
        objects.append({"cells": cells, "port": port, "side": candidates[0][0],
                        "mask": mask, "local_port": local_port, "color": carrier,
                        "key": _canonical(mask)})
        port_colors.add(pc)
        body_colors.add(carrier)
    if not objects or len(port_colors) != 1:
        raise ValueError("need relays with a common port color")
    pc = next(iter(port_colors))
    if len({bg, signal, cap_color, pc}) != 4 or body_colors & {bg, signal, cap_color, pc}:
        raise ValueError("semantic colors must be disjoint")
    groups = {}
    for i, obj in enumerate(objects):
        groups.setdefault(obj["key"], []).append(i)
    if any(len(group) > 2 for group in groups.values()):
        raise ValueError("a relay may have at most one partner")
    for group in groups.values():
        for i in group:
            objects[i]["mate"] = next((j for j in group if j != i), None)
    return a, bg, signal, set(line), line[-1], direction, objects


def _project_solve(value, *, local=False):
    """Input-only projection solver; local=True is the explicit A-only ablation."""
    a, bg, signal, launcher, cap, direction, objects = _decode(value)
    trace, active, events, segments = set(launcher), set(), [], []
    origins = [cap]
    termination = None
    terminal = None
    for _ in range(len(objects) + 2):
        dr, dc = DIRS[direction]
        stops = []
        for r, c in origins:
            edge_distance = (r + 1 if direction == 0 else a.shape[1] - c if direction == 1
                             else a.shape[0] - r if direction == 2 else c + 1)
            hits = [(edge_distance, -1)]
            for i, obj in enumerate(objects):
                for rr, cc in obj["cells"]:
                    distance = (rr - r) * dr + (cc - c) * dc
                    if distance > 0 and (rr - r) * dc - (cc - c) * dr == 0:
                        hits.append((distance, i))
            stops.append(min(hits))
        if len(set(stops)) != 1:
            if local:
                # Define the shortcut even outside the intended full-contact subset.
                distance, index = min(stops)
            else:
                raise ValueError("partial-width or tied relay contact")
        else:
            distance, index = stops[0]
        segment = {(r + k * dr, c + k * dc) for r, c in origins
                   for k in range(1, distance)}
        trace.update(segment)
        segments.append(segment)
        if index == -1:
            termination = "boundary"
            break
        if index in active:
            if local:
                termination = "repeat"
                break
            raise ValueError("route revisits an activated object")
        active.add(index)
        trace.update(objects[index]["cells"])
        mate = objects[index]["mate"]
        if not local and mate is None:
            termination, terminal = "singleton", index
            break
        emitter = index if local else mate
        if not local:
            if emitter in active:
                raise ValueError("partner already activated")
            active.add(emitter)
            trace.update(objects[emitter]["cells"])
        events.append((index, emitter, direction, objects[emitter]["side"],
                       len(objects[emitter]["port"]), distance - 1))
        origins, direction = sorted(objects[emitter]["port"]), objects[emitter]["side"]
    if termination is None:
        raise ValueError("unterminated route")
    out = np.full(a.shape, bg, dtype=np.int8)
    for p in trace:
        out[p] = signal
    return out, {"objects": objects, "active": active, "events": events,
                 "termination": termination, "terminal": terminal,
                 "segments": segments, "trace": trace, "launcher": launcher,
                 "background": bg, "signal": signal, "shape": a.shape,
                 "initial": _decode_initial(cap, launcher), "transfers": len(events)}


def _decode_initial(cap, launcher):
    return next(d for d, (dr, dc) in enumerate(DIRS)
                if (cap[0] - dr, cap[1] - dc) in launcher)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the same-sized output entirely from the visible input."""
    return _project_solve(input_grid)[0]


def _launcher(h, w, direction, rng):
    length = int(rng.integers(2, 5))
    lane = int(rng.integers(2, (w if direction % 2 == 0 else h) - 2))
    start = (h - 1, lane) if direction == 0 else (lane, 0) if direction == 1 else (0, lane) if direction == 2 else (lane, w - 1)
    dr, dc = DIRS[direction]
    tail = {(start[0] + k * dr, start[1] + k * dc) for k in range(length)}
    cap = start[0] + length * dr, start[1] + length * dc
    return tail, cap


def _interior(cells, h, w):
    return all(0 < r < h - 1 and 0 < c < w - 1 for r, c in cells)


def _entry(mask, origins, direction, occupied, trace, h, w, rng):
    """Enumerate flat first contacts, then sample gap bands and alignment."""
    oh, ow = max(r for r, c in mask) + 1, max(c for r, c in mask) + 1
    entry_side = (direction + 2) % 4
    axis_length = ow if direction % 2 == 0 else oh
    strips = []
    for start in range(axis_length - len(origins) + 1):
        if entry_side == 0:
            strip = {(0, start + k) for k in range(len(origins))}
        elif entry_side == 1:
            strip = {(start + k, ow - 1) for k in range(len(origins))}
        elif entry_side == 2:
            strip = {(oh - 1, start + k) for k in range(len(origins))}
        else:
            strip = {(start + k, 0) for k in range(len(origins))}
        if strip <= mask:
            strips.append(sorted(strip))
    options = []
    dr, dc = DIRS[direction]
    blocked = _halo(occupied)
    for strip in strips:
        for distance in range(2, 30):
            r0 = origins[0][0] + distance * dr - strip[0][0]
            c0 = origins[0][1] + distance * dc - strip[0][1]
            cells = _shift(mask, r0, c0)
            if not _interior(cells, h, w) or cells & blocked or cells & trace:
                continue
            approach = {(r + k * dr, c + k * dc) for r, c in origins for k in range(1, distance)}
            if approach & occupied:
                continue
            options.append((distance, r0, c0, cells, approach))
    if not options:
        return None
    bands = [[x for x in options if lo <= x[0] <= hi] for lo, hi in ((2, 4), (5, 9), (10, 29))]
    bands = [band for band in bands if band]
    band = bands[int(rng.integers(len(bands)))]
    return band[int(rng.integers(len(band)))]


def _ray_cells(port, side, h, w):
    dr, dc = DIRS[side]
    cells = set()
    for r, c in port:
        for k in range(1, 31):
            p = r + k * dr, c + k * dc
            if not (0 <= p[0] < h and 0 <= p[1] < w):
                break
            cells.add(p)
    return cells


def _remote(mask, width, occupied, trace, h, w, rng, need_room, forbidden_port):
    ports = [(s, p) for s, p in _ports(mask, width) if p != forbidden_port]
    if not ports:
        return None
    oh, ow = max(r for r, c in mask) + 1, max(c for r, c in mask) + 1
    blocked = _halo(occupied)
    for _ in range(150):
        r0, c0 = int(rng.integers(1, h - oh)), int(rng.integers(1, w - ow))
        cells = _shift(mask, r0, c0)
        if cells & blocked or cells & trace:
            continue
        side, port = ports[int(rng.integers(len(ports)))]
        port = _shift(port, r0, c0)
        ray = _ray_cells(port, side, h, w)
        if ray & occupied or len(ray) < width * (8 if need_room else 1):
            continue
        return {"cells": cells, "port": port, "side": side}
    return None


def _offroute(mask, width, occupied, trace, h, w, rng):
    ports = _ports(mask, width)
    oh, ow = max(r for r, c in mask) + 1, max(c for r, c in mask) + 1
    blocked = _halo(occupied)
    for _ in range(160):
        r0, c0 = int(rng.integers(1, h - oh)), int(rng.integers(1, w - ow))
        cells = _shift(mask, r0, c0)
        if cells & blocked or cells & trace:
            continue
        side, port = ports[int(rng.integers(len(ports)))]
        return {"cells": cells, "port": _shift(port, r0, c0), "side": side, "mate": None}
    return None


def _try_layout(rng, spec):
    n = spec["n"]
    lower = 19 if n <= 1 else 23 if n == 2 else 27
    h, w = map(int, rng.integers(lower, 31, size=2))
    if n <= 1 and rng.random() < .35:
        h, w = int(rng.integers(14, 21)), int(rng.integers(24, 31))
    initial = int(rng.integers(4))
    tail, cap = _launcher(h, w, initial, rng)
    occupied, trace = tail | {cap}, tail | {cap}
    objects, forbidden = [], set()
    origins, direction = [cap], initial
    for step in range(n):
        mask = _sample_mask(rng, forbidden, orbit4=spec["quarter"] and step == 0)
        forbidden.add(_canonical(mask))
        placed = _entry(mask, origins, direction, occupied, trace, h, w, rng)
        if placed is None:
            return None
        _, r0, c0, cells, approach = placed
        incoming_width = 1 if spec["quarter"] and step == 0 else int(rng.integers(1, 3))
        choices = _ports(mask, incoming_width)
        entry_side, entry_port = choices[int(rng.integers(len(choices)))]
        k = int(rng.choice([1, 3])) if spec["quarter"] and step == 0 else int(rng.integers(4))
        mate_mask = _rotate(mask, k)
        # Rotate ports in the same bounding coordinate frame as their footprint.
        rotated_full = {(r, c) for r, c in mask}
        rotated_mark = set(entry_port)
        for _ in range(k):
            raw_full = {(-c, r) for r, c in rotated_full}
            rr, cc = min(r for r, c in raw_full), min(c for r, c in raw_full)
            rotated_full = _shift(raw_full, -rr, -cc)
            rotated_mark = _shift({(-c, r) for r, c in rotated_mark}, -rr, -cc)
        remote = _remote(mate_mask, spec["widths"][step], occupied | cells,
                         trace | approach | cells, h, w, rng,
                         step < n - 1 or spec["terminal"], rotated_mark)
        if remote is None:
            return None
        i = len(objects)
        objects.append({"cells": cells, "port": _shift(entry_port, r0, c0),
                        "side": entry_side, "mate": i + 1})
        remote["mate"] = i
        objects.append(remote)
        occupied |= cells | remote["cells"]
        trace |= approach | cells | remote["cells"]
        origins, direction = sorted(remote["port"]), remote["side"]
    if spec["terminal"]:
        mask = _sample_mask(rng, forbidden, chiral=spec["mirror"])
        forbidden.add(_canonical(mask))
        placed = _entry(mask, origins, direction, occupied, trace, h, w, rng)
        if placed is None:
            return None
        _, r0, c0, cells, approach = placed
        options = _ports(mask, int(rng.integers(1, 3)))
        side, port = options[int(rng.integers(len(options)))]
        objects.append({"cells": cells, "port": _shift(port, r0, c0), "side": side, "mate": None})
        occupied |= cells
        trace |= cells | approach
        if spec["mirror"]:
            mirror_mask = _rotate(_mirror(mask), int(rng.integers(4)))
            reflected = _offroute(mirror_mask, 1, occupied, trace, h, w, rng)
            if reflected is None:
                return None
            objects.append(reflected)
            occupied |= reflected["cells"]
            forbidden.add(_canonical(mirror_mask))
    else:
        final_ray = _ray_cells(origins, direction, h, w)
        if final_ray & occupied:
            return None
        trace |= final_ray
    for _ in range(spec["extras"]):
        mask = _sample_mask(rng, forbidden)
        forbidden.add(_canonical(mask))
        obj = _offroute(mask, int(rng.integers(1, 3)), occupied, trace, h, w, rng)
        if obj is None:
            return None
        objects.append(obj)
        occupied |= obj["cells"]
    if spec["dormant"]:
        mask = _sample_mask(rng, forbidden)
        i = len(objects)
        for k in (0, int(rng.integers(4))):
            obj = _offroute(_rotate(mask, k), 1, occupied, trace, h, w, rng)
            if obj is None:
                return None
            objects.append(obj)
            occupied |= obj["cells"]
        objects[i]["mate"], objects[i + 1]["mate"] = i + 1, i
    if not objects or len(occupied) * 2 >= h * w:
        return None
    return h, w, tail, cap, initial, objects


def _fallback_layout(spec):
    """Finite, explicit remote circuit; optional extras are simplified."""
    n = spec["n"]
    h = w = 30
    tail, cap, initial = {(3, 0), (3, 1)}, (3, 2), 1
    square = {(r, c) for r in range(5) for c in range(5)}
    cuts = [{(4, 4)}, {(4, 4), (4, 3), (3, 4)},
            {(4, 4), (4, 3), (3, 4), (3, 3)}]
    entries = [(2, 4), (11, 22), (20, 4)]
    exits = [(2, 22), (11, 4), (20, 22)]
    objects = []
    for j in range(n):
        mask = square - cuts[j]
        mate = _rotate(mask)
        side = 2 if j < n - 1 or n == 1 else (1 if spec["terminal"] else 3) if n == 2 else (3 if spec["terminal"] else 1)
        options = [(s, p) for s, p in _ports(mate, spec["widths"][j]) if s == side]
        # Centered ports align subsequent receivers and preserve separation.
        _, remote_port = min(options, key=lambda x: (abs(min(r if side % 2 else c for r, c in x[1]) - 1), sorted(x[1])))
        entry_side, entry_port = next((s, p) for s, p in _ports(mask, 1) if s == 0)
        i = len(objects)
        objects.append({"cells": _shift(mask, *entries[j]), "port": _shift(entry_port, *entries[j]), "side": entry_side, "mate": i + 1})
        objects.append({"cells": _shift(mate, *exits[j]), "port": _shift(remote_port, *exits[j]), "side": side, "mate": i})
    if spec["terminal"]:
        mask = square - {(4, 4), (3, 4)}
        loc = (13, 22) if n == 1 else (11, 12) if n == 2 else (20, 12)
        side, port = next((s, p) for s, p in _ports(mask, 1) if s == 2)
        objects.append({"cells": _shift(mask, *loc), "port": _shift(port, *loc), "side": side, "mate": None})
        if spec["mirror"]:
            mirrored = _mirror(mask)
            loc = (13, 4) if n == 1 else (21, 12) if n == 2 else (11, 12)
            side, port = _ports(mirrored, 1)[0]
            objects.append({"cells": _shift(mirrored, *loc), "port": _shift(port, *loc), "side": side, "mate": None})
    # A plain singleton is off the circuit, including the zero-hit fallback.
    mask = {(r, c) for r in range(3) for c in range(5)}
    objects.append({"cells": _shift(mask, 22, 12), "port": {(22, 13)}, "side": 0, "mate": None})
    if spec["terminal"] and n == 2 and spec["mirror"]:
        # The mirror already uses this region; move the optional singleton.
        objects[-1] = {"cells": _shift(mask, 22, 22), "port": {(22, 23)}, "side": 0, "mate": None}
    if spec["terminal"] and n == 3:
        objects[-1] = {"cells": _shift(mask, 2, 12), "port": {(2, 13)}, "side": 0, "mate": None}
    return h, w, tail, cap, initial, objects


def _construct_output(h, w, tail, cap, initial, objects, bg, signal):
    """Constructor oracle: advance a strip cell by cell using assigned mates."""
    owner = {p: i for i, obj in enumerate(objects) for p in obj["cells"]}
    trace, seen = tail | {cap}, set()
    direction, origins = initial, [cap]
    front = [(r + DIRS[direction][0], c + DIRS[direction][1]) for r, c in origins]
    for _ in range(30 * (len(objects) + 2)):
        inside = [0 <= r < h and 0 <= c < w for r, c in front]
        if not any(inside):
            break
        assert all(inside)
        hits = {owner[p] for p in front if p in owner}
        if hits:
            assert len(hits) == 1
            i = next(iter(hits))
            assert all(owner.get(p) == i for p in front) and i not in seen
            seen.add(i)
            trace |= objects[i]["cells"]
            mate = objects[i]["mate"]
            if mate is None:
                break
            assert mate not in seen
            seen.add(mate)
            trace |= objects[mate]["cells"]
            direction, origins = objects[mate]["side"], objects[mate]["port"]
            dr, dc = DIRS[direction]
            front = [(r + dr, c + dc) for r, c in origins]
        else:
            trace.update(front)
            dr, dc = DIRS[direction]
            front = [(r + dr, c + dc) for r, c in front]
    else:
        raise AssertionError("constructor did not terminate")
    out = np.full((h, w), bg, dtype=np.int8)
    for p in trace:
        out[p] = signal
    return out


def _make_pair(rng, spec, palette, *, force_fallback=False):
    layout = None
    for _ in range(0 if force_fallback else 100):
        layout = _try_layout(rng, spec)
        if layout is not None:
            break
    fallback = layout is None
    if fallback:
        layout = _fallback_layout(spec)
    h, w, tail, cap, direction, objects = layout
    bg, signal, cap_color, pc = map(int, palette[:4])
    carriers = list(map(int, palette[4:]))
    pool_size = int(rng.integers(2, len(carriers) + 1))
    body_colors = list(map(int, rng.choice(carriers[:pool_size], size=len(objects))))
    # Two explicit color-independence witnesses, without coloring the route order.
    if spec["n"]:
        body_colors[0], body_colors[1] = carriers[0], carriers[1]
        singleton = next((i for i, obj in enumerate(objects) if obj["mate"] is None), None)
        if singleton is not None:
            body_colors[singleton] = carriers[0]
    inp = np.full((h, w), bg, dtype=np.int8)
    for i, obj in enumerate(objects):
        color = body_colors[i]
        for p in obj["cells"]:
            assert inp[p] == bg
            inp[p] = pc if p in obj["port"] else color
    for p in tail:
        inp[p] = signal
    inp[cap] = cap_color
    out = _construct_output(h, w, tail, cap, direction, objects, bg, signal)
    # Whole-scene rotations change launcher side without hidden conventions.
    k = int(rng.integers(4))
    return {"input": np.rot90(inp, k).copy(), "output": np.rot90(out, k).copy()}, fallback


def _fingerprint(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Return one reproducible puzzle with visible, family-wide conventions."""
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    stable = bool(rng.random() < .25)
    base = rng.permutation(10)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            teaching = split == "train" and index < 2
            n = index + 1 if teaching else (3 if split == "test" and index == 0 else int(rng.integers(1 if split == "test" else 0, 4 if split == "test" else 3)))
            terminal = index == 1 if teaching else bool(n and rng.random() < .6)
            widths = list(map(int, rng.integers(1, 3, size=n)))
            if teaching:
                widths = [2] if n == 1 else list(map(int, rng.permutation([1, 2])))
            spec = {"n": n, "terminal": terminal, "mirror": terminal and (teaching or rng.random() < .4),
                    "quarter": teaching, "widths": widths,
                    "extras": int(rng.integers(1 if teaching or n == 0 else 0, 3)),
                    "dormant": bool(rng.random() < .25)}
            if n == 3 and spec["dormant"]:
                # Reserve room for an unreachable *matched* class in long circuits.
                spec["extras"] = 0
            palette = base if stable else rng.permutation(10)
            for attempt in range(12):
                pair, _ = _make_pair(rng, spec, palette, force_fallback=attempt == 11)
                key = _fingerprint(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct input budget exhausted")
    order = rng.permutation(num_train)
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def _facts(pair):
    a, b = _grid(pair["input"]), _grid(pair["output"])
    expected, f = _project_solve(a)
    assert a.shape == b.shape and np.array_equal(b, expected), "incorrect output"
    assert not np.array_equal(a, b)
    objects, active = f["objects"], f["active"]
    for i, obj in enumerate(objects):
        values = {int(b[p]) for p in obj["cells"]}
        assert values == {f["signal"] if i in active else f["background"]}, "footprint activation/erasure"
    f["decoys"] = len(objects) - len(active)
    f["widths"] = {e[4] for e in f["events"]}
    f["quarter"] = False
    f["port_independent"] = False
    f["different_colors"] = False
    for i, j, *_ in f["events"]:
        src, dst = objects[i], objects[j]
        identity = _key(src["mask"]) == _key(dst["mask"])
        half = _key(_rotate(src["mask"], 2)) == _key(dst["mask"])
        f["quarter"] |= not identity and not half
        f["different_colors"] |= src["color"] != dst["color"]
        alignments = []
        for k in range(4):
            full, mark = set(src["mask"]), set(src["local_port"])
            for _ in range(k):
                raw = {(-c, r) for r, c in full}
                rr, cc = min(r for r, c in raw), min(c for r, c in raw)
                full = _shift(raw, -rr, -cc)
                mark = _shift({(-c, r) for r, c in mark}, -rr, -cc)
            if _key(full) == _key(dst["mask"]):
                alignments.append(mark == dst["local_port"])
        f["port_independent"] |= not any(alignments)
    singles = [o for o in objects if o["mate"] is None]
    f["cross_color"] = any(o["color"] == s["color"] for i, o in enumerate(objects)
                            if i in active and o["mate"] is not None for s in singles)
    f["mirror_stop"] = False
    if f["terminal"] is not None:
        terminal = objects[f["terminal"]]
        mk = _canonical(_mirror(terminal["mask"]))
        f["mirror_stop"] = mk != terminal["key"] and any(o["key"] == mk and o["mate"] is None for o in singles)
    local, _ = _project_solve(a, local=True)
    f["local_differs"] = not np.array_equal(local, b)
    for name, selected in (("paired_only_differs", [o for o in objects if o["mate"] is not None]),
                           ("all_objects_differs", objects)):
        shortcut = np.full(a.shape, f["background"], dtype=np.int8)
        for p in f["launcher"] | set().union(*(o["cells"] for o in selected)):
            shortcut[p] = f["signal"]
        f[name] = not np.array_equal(shortcut, b)
    return f


def validate_generated_puzzle(puzzle):
    """Check visible family invariants and collective teaching evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            key = _fingerprint(_grid(pair["input"]))
            assert key not in seen, "duplicate input"
            seen.add(key)
            f = _facts(pair)
            assert 0 <= f["transfers"] <= 3
            records[split].append(f)
    train, test = records["train"], records["test"]
    assert any(f["transfers"] == 1 and f["termination"] == "boundary" for f in train)
    assert any(f["transfers"] == 2 and f["mirror_stop"] for f in train)
    assert {1, 2} <= set().union(*(f["widths"] for f in train))
    assert any(f["quarter"] for f in train)
    for name in ("port_independent", "different_colors", "cross_color", "local_differs",
                 "paired_only_differs", "all_objects_differs"):
        assert any(f[name] for f in train), name
    assert all(f["decoys"] > 0 for f in train if f["transfers"] in (1, 2) and f["mirror_stop"])
    assert max(f["transfers"] for f in train) == 2
    assert test[0]["transfers"] == 3
    assert all(f["widths"] <= {1, 2} for f in test)


def self_test():
    """Stream 42 seeds and finite boundary cases; save no grids or logs."""
    shapes, structures, masks, palettes = set(), set(), set(), set()
    directions, terminations, counts, widths, fingerprints = set(), set(), set(), set(), set()
    dormant, long_approach, pairs = 0, 0, 0
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                prediction = solve(a.copy())
                assert np.array_equal(prediction, pair["output"])
                digest.update(bytes(a.shape) + a.tobytes() + prediction.tobytes())
                f = _facts(pair)
                pairs += 1
                shapes.add(a.shape)
                directions.add(f["initial"])
                terminations.add(f["termination"])
                counts.add(f["transfers"])
                widths.update(f["widths"])
                masks.update(o["key"] for o in f["objects"])
                palettes.add(tuple(map(int, np.unique(a))))
                dormant += any(i not in f["active"] and o["mate"] is not None for i, o in enumerate(f["objects"]))
                long_approach += any(e[5] >= 8 for e in f["events"])
                roles = {}
                normalized = bytes(roles.setdefault(int(v), len(roles)) for v in a.ravel())
                structures.add((a.shape, normalized))
        fingerprints.add(digest.hexdigest())
        if seed == 0:
            first = puzzle
    assert len(fingerprints) == 42 and len(shapes) >= 25
    assert len(structures) >= 190 and len(masks) >= 70 and len(palettes) >= 10
    assert directions == set(range(4)) and terminations == {"boundary", "singleton"}
    assert counts == {0, 1, 2, 3} and widths == {1, 2}
    assert dormant > 0 and long_approach > 0
    again = generate_puzzle(0)
    assert all(np.array_equal(p[k], q[k]) for s in first for p, q in zip(first[s], again[s]) for k in p)
    # All legal explicit counts, including both extremes, are finite and valid.
    legal_count_cases = 0
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(7300 + 10 * a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
            legal_count_cases += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.5, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    fallback_rng = np.random.default_rng(12029)
    fallback_cases = 0
    for n in range(4):
        for terminal in ((False, True) if n else (False,)):
            for width in (1, 2):
                spec = {"n": n, "terminal": terminal, "mirror": terminal,
                        "quarter": True, "widths": [width] * n, "extras": 1, "dormant": False}
                pair, fallback = _make_pair(fallback_rng, spec, np.arange(10), force_fallback=True)
                f = _facts(pair)
                assert fallback and f["transfers"] == n
                assert f["termination"] == ("singleton" if terminal else "boundary")
                fallback_cases += 1
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(fingerprints),
            "input_shapes": len(shapes), "normalized_structures": len(structures),
            "shape_classes": len(masks), "palettes": len(palettes),
            "initial_directions": sorted(directions), "transfer_counts": sorted(counts),
            "dormant_pair_scenes": dormant, "long_approach_scenes": long_approach,
            "legal_count_cases": legal_count_cases, "fallback_cases": fallback_cases}
