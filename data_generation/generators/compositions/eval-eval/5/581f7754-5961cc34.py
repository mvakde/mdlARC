"""Wake, slide, relay: input-visible guides change a signal's causal route.

Only standard-library modules and NumPy are used. Generation records never enter
solve(), which decodes markers, ports, geometry and every activation from input.
"""
from __future__ import annotations

from collections import Counter
import copy
import json
import numpy as np

# Clockwise normals: top, right, bottom, left.
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'grid dimensions must be 1..30')
    _need(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)), 'integer colors 0..9 required')
    return a.astype(np.int64, copy=False)


def _components(points):
    remaining, result = set(points), []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, component = [p], {p}
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    stack.append(q)
        result.append(component)
    return result


def _shift(points, delta):
    return {(r + delta[0], c + delta[1]) for r, c in points}


def _inside(points, shape):
    return all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in points)


def _separate(a, b):
    # Eight-neighbor source separation is a construction convenience.
    return all((r + dr, c + dc) not in b for r, c in a
               for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def _port_side(cells, port):
    r0, r1 = min(r for r, c in cells), max(r for r, c in cells)
    c0, c1 = min(c for r, c in cells), max(c for r, c in cells)
    sides = [i for i, ok in enumerate((all(r == r0 for r, c in port),
              all(c == c1 for r, c in port), all(r == r1 for r, c in port),
              all(c == c0 for r, c in port))) if ok]
    _need(len(sides) == 1, 'port must be on one uniquely identifiable extreme side')
    side = sides[0]
    span = sorted(c if side % 2 == 0 else r for r, c in port)
    _need(span == list(range(span[0], span[0] + len(span))), 'port must be contiguous')
    dr, dc = DIRS[side]
    _need(all((r - dr, c - dc) in cells - port for r, c in port), 'port must have inward support')
    return side


def _decode(value):
    a = _grid(value)
    h, w = a.shape
    bg = int(a[0, 0])
    _need(all(int(a[p]) == bg for p in ((0, w - 1), (h - 1, 0), (h - 1, w - 1))), 'corners must be background')
    _need(np.count_nonzero(a == bg) > a.size / 2, 'background must occupy more than half the input')
    components = _components(map(tuple, np.argwhere(a != bg)))
    edge = lambda p: p[0] in (0, h - 1) or p[1] in (0, w - 1)
    anchors, rest = {}, []
    for cells in components:
        if len(cells) == 1 and edge(next(iter(cells))):
            p = next(iter(cells))
            anchors.setdefault(int(a[p]), set()).add(p)
        else:
            rest.append(cells)
    _need(1 <= len(anchors) <= 4, 'one to four guide identities required')
    systems = {}
    for color, points in anchors.items():
        rows, cols = {r for r, c in points}, {c for r, c in points}
        horizontal = len(rows) == 1 and all(c in (0, w - 1) for r, c in points)
        vertical = len(cols) == 1 and all(r in (0, h - 1) for r, c in points)
        _need(horizontal != vertical, 'inconsistent guide markers')
        axis = 0 if horizontal else 1
        guide = next(iter(rows if horizontal else cols))
        _need(0 < guide < a.shape[axis] - 1, 'corner anchors are not permitted')
        systems[color] = (axis, guide)
    candidates = []
    for cells in rest:
        boundary = [p for p in cells if edge(p)]
        if len(boundary) != 1 or not 3 <= len(cells) <= 6:
            continue
        start = boundary[0]
        direction = 2 if start[0] == 0 else 0 if start[0] == h - 1 else 1 if start[1] == 0 else 3
        dr, dc = DIRS[direction]
        line = [(start[0] + k * dr, start[1] + k * dc) for k in range(len(cells))]
        signal, pc = int(a[start]), int(a[line[-1]]) if set(line) == cells else (-1)
        if (set(line) == cells and signal != pc and signal not in systems and pc not in systems
                and all(int(a[p]) == signal for p in line[:-1])):
            candidates.append((cells, line, direction, signal, pc))
    _need(len(candidates) == 1, 'unique boundary tail and cap required')
    launcher, line, initial, signal, pc = candidates[0]
    _need(len({bg, signal, pc} | set(systems)) == 3 + len(systems), 'semantic roles must be disjoint')
    objects = []
    for cells in rest:
        if cells == launcher:
            continue
        _need(len(cells) >= 7, 'relay must have at least seven cells')
        counts = Counter(int(a[p]) for p in cells)
        keys = [c for c in systems if counts[c] == 1]
        _need(len(keys) == 1 and sum(counts[c] for c in systems) == 1, 'relay requires exactly one guide key')
        key = keys[0]
        reg = next(p for p in cells if int(a[p]) == key)
        port = {p for p in cells if int(a[p]) == pc}
        _need(1 <= len(port) <= 2 and signal not in counts, 'relay requires a port of width 1..2')
        _need(len(cells - port - {reg}) > len(port), 'body must dominate port')
        _need(len(_components(cells - port)) == 1, 'relay must stay connected without its port')
        side = _port_side(cells, port)
        axis, guide = systems[key]
        delta = [0, 0]
        delta[axis] = guide - reg[axis]
        target = _shift(cells, delta)
        _need(_inside(target, a.shape), 'this family excludes clipping')
        objects.append({'cells': cells, 'port': port, 'side': side, 'reg': reg, 'key': key,
                        'delta': tuple(delta), 'target': target,
                        'values': {p: int(a[p]) for p in cells}})
    _need(objects and all(any(o['key'] == key for o in objects) for key in systems), 'every guide must have a matching relay')
    return {'grid': a, 'bg': bg, 'signal': signal, 'pc': pc, 'anchors': anchors,
            'systems': systems, 'tail': set(line[:-1]), 'cap': line[-1],
            'initial': initial, 'objects': objects}


def _next_hit(shape, origins, direction, objects, remaining):
    """Compute first contact in each lane; ties/partial strip hits are illegal."""
    dr, dc = DIRS[direction]
    stops = []
    for r, c in origins:
        boundary_distance = r + 1 if direction == 0 else shape[1] - c if direction == 1 else shape[0] - r if direction == 2 else c + 1
        hits = [(boundary_distance, -1)]
        for index in sorted(remaining):
            for rr, cc in objects[index]['cells']:
                distance = (rr - r) * dr + (cc - c) * dc
                if distance > 0 and (rr - r) * dc == (cc - c) * dr:
                    hits.append((distance, index))
        stops.append(min(hits))
    _need(len(set(stops)) == 1, 'partial-width or simultaneous multiple-relay contact')
    distance, index = stops[0]
    segment = {(r + k * dr, c + k * dc) for r, c in origins for k in range(1, distance)}
    return index, segment, distance - 1


def _run(decoded, mode='dynamic'):
    """Route from visible inputs. Alternative modes are diagnostics only."""
    a, bg = decoded['grid'], decoded['bg']
    objects = copy.deepcopy(decoded['objects'])
    if mode == 'center_alignment':
        for o in objects:
            axis, guide = decoded['systems'][o['key']]
            center = (min(p[axis] for p in o['cells']) + max(p[axis] for p in o['cells'])) // 2
            delta = [0, 0]
            delta[axis] = guide - center
            o['delta'] = tuple(delta)
    if mode == 'prealign':
        for o in objects:
            o['values'] = {(r + o['delta'][0], c + o['delta'][1]): v for (r, c), v in o['values'].items()}
            o['cells'] = o['target']
            o['port'] = _shift(o['port'], o['delta'])
            o['reg'] = (o['reg'][0] + o['delta'][0], o['reg'][1] + o['delta'][1])
            o['delta'] = (0, 0)
    trace = decoded['tail'] | {decoded['cap']}
    origins, direction = [decoded['cap']], decoded['initial']
    remaining, activated, landed, segments, transitions, gaps = set(range(len(objects))), [], [], [], [], []
    fixed = set().union(*decoded['anchors'].values())
    for _ in range(len(objects) + 1):
        index, segment, gap = _next_hit(a.shape, origins, direction, objects, remaining)
        if mode == 'dynamic':
            _need(not segment & set().union(*(set(v) for v in landed)), 'ray meets a previously moved relay')
        trace |= segment
        segments.append(segment)
        if index == -1:
            break
        o = objects[index]
        remaining.remove(index)
        activated.append(index)
        shift = (0, 0) if mode == 'no_alignment' else o['delta']
        values = {(r + shift[0], c + shift[1]): v for (r, c), v in o['values'].items()}
        _need(_inside(set(values), a.shape), 'alternative alignment falls outside the canvas')
        if mode == 'dynamic':
            occupied = fixed | decoded['tail'] | {decoded['cap']} | set().union(*(set(v) for v in landed))
            others = set().union(*(objects[j]['cells'] for j in remaining))
            _need(not set(values) & (occupied | others), 'moved relay overlaps another live object or reference')
            _need(not set(values) & trace, 'moved relay overlaps the accumulated signal')
        landed.append(values)
        transitions.append((direction, o['side'], len(o['port'])))
        gaps.append(gap)
        origins = sorted(o['port'] if mode == 'route_then_align' else _shift(o['port'], shift))
        if mode == 'thin_ports':
            origins = origins[:1]
        direction = o['side']
    else:
        raise ValueError('route did not terminate')
    out = np.full(a.shape, bg, dtype=np.int64)
    for p in trace:
        out[p] = decoded['signal']
    for values in landed:
        for p, v in values.items():
            out[p] = v
    # References are passive, retain their identity, and are transparent to rays.
    for color, points in decoded['anchors'].items():
        for p in points:
            out[p] = color
    facts = {'active': activated, 'landed': landed, 'trace': trace, 'segments': segments,
             'transitions': transitions, 'gaps': gaps, 'decoys': len(objects) - len(activated),
             'moved': sum(objects[j]['delta'] != (0, 0) for j in activated)}
    return out, facts


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve the whole family without a seed, training examples or generator state."""
    return _run(_decode(input_grid))[0]


def _shape(rng, incoming, outgoing, in_width, out_width):
    h, w = int(rng.integers(3, 7)), int(rng.integers(3, 7))
    if outgoing % 2:
        h = max(h, out_width + 2)
    else:
        w = max(w, out_width + 2)
    if incoming % 2:
        h = max(h, in_width)
    else:
        w = max(w, in_width)
    mask = {(r, c) for r in range(h) for c in range(w)}
    kind = int(rng.integers(5))
    if kind == 1:  # beveled corners
        mask -= {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    elif kind == 2:  # hollow frame
        # A two-cell rim supplies the inward support behind an extreme-side
        # port, while retaining an actual hole at these sizes.
        h, w = max(5, h), max(5, w)
        mask = {(r, c) for r in range(h) for c in range(w)
                if r < 2 or r >= h - 2 or c < 2 or c >= w - 2}
    elif kind == 3:  # asymmetric corner cut
        mask -= {(r, c) for r in range(int(rng.integers(1, h - 1))) for c in range(int(rng.integers(1, w - 1)))}
    elif kind == 4:  # a connected concave notch
        c = int(rng.integers(1, w - 1))
        mask -= {(r, c) for r in range(int(rng.integers(1, h - 1)))}
    # Window options on an extreme side, with unambiguous noncorner markings.
    def windows(side, width, noncorner):
        length = w if side % 2 == 0 else h
        candidates = []
        for start in range(1 if noncorner else 0, length - width + (0 if noncorner else 1)):
            cells = {(0, start + k) if side == 0 else (start + k, w - 1) if side == 1
                     else (h - 1, start + k) if side == 2 else (start + k, 0) for k in range(width)}
            dr, dc = DIRS[side]
            if cells <= mask and all((r - dr, c - dc) in mask - cells for r, c in cells):
                candidates.append(cells)
        return candidates
    ports = windows(outgoing, out_width, True)
    entries = windows((incoming + 2) % 4, in_width, False)
    if not ports or not entries or len(mask) < 7:
        return None
    port = ports[int(rng.integers(len(ports)))]
    if len(_components(mask - port)) != 1:
        return None
    entry = entries[int(rng.integers(len(entries)))]
    registers = sorted(mask - port)
    reg = registers[int(rng.integers(len(registers)))]
    return mask, port, entry, reg, h, w


def _launcher(shape, direction, rng):
    h, w = shape
    length = int(rng.integers(2, 5))
    lane = int(rng.integers(2, (w if direction % 2 == 0 else h) - 2))
    start = (h - 1, lane) if direction == 0 else (lane, 0) if direction == 1 else (0, lane) if direction == 2 else (lane, w - 1)
    dr, dc = DIRS[direction]
    line = [(start[0] + k * dr, start[1] + k * dc) for k in range(length + 1)]
    return set(line[:-1]), line[-1]


def _make_guides(rng, shape, colors, count, occupied, mixed=False):
    axes = [0, 1] + list(map(int, rng.integers(2, size=count - 2))) if mixed else list(map(int, rng.integers(2, size=count)))
    systems, anchors = {}, {}
    for j, axis in enumerate(axes):
        key = colors[4 + j]
        guides = list(range(2, shape[axis] - 2))
        rng.shuffle(guides)
        for guide in guides:
            ends = [int(rng.integers(2))] if rng.random() < .65 else [0, 1]
            points = {(guide, 0 if e == 0 else shape[1] - 1) if axis == 0
                      else (0 if e == 0 else shape[0] - 1, guide) for e in ends}
            if _separate(points, occupied) and all(s != (axis, guide) for s in systems.values()):
                systems[key], anchors[key] = (axis, guide), points
                occupied |= points
                break
        else:
            return None
    return systems, anchors


def _colored_object(cells, port, reg, key, side, delta, rng, colors):
    body = int(rng.choice(colors[7:9]))
    values = {p: colors[2] if p in port else key if p == reg else body for p in cells}
    if rng.random() < .55:
        choices = sorted(cells - port - {reg})
        for k in range(int(rng.integers(1, min(4, len(choices)) + 1))):
            p = choices[int(rng.integers(len(choices)))]
            values[p] = colors[9]
    return {'cells': cells, 'port': port, 'reg': reg, 'key': key, 'side': side,
            'delta': tuple(delta), 'target': _shift(cells, delta), 'values': values}


def _render(shape, colors, anchors, tail, cap, objects):
    a = np.full(shape, colors[0], dtype=np.int64)
    for color, points in anchors.items():
        for p in points:
            a[p] = color
    for o in objects:
        for p, v in o['values'].items():
            a[p] = v
    for p in tail:
        a[p] = colors[1]
    a[cap] = colors[2]
    return a


def _try_scene(rng, plan, colors):
    n, decoys = plan['n'], plan['decoys']
    low = 11 if n <= 1 else 18 if n == 2 else 23 if n == 3 else 27
    shape = tuple(map(int, rng.integers(low, 31, size=2)))
    if n <= 1 and rng.random() < .3:
        shape = (int(rng.integers(10, 16)), int(rng.integers(22, 31)))
        if rng.random() < .5:
            shape = shape[::-1]
    initial = int(rng.integers(4))
    tail, cap = _launcher(shape, initial, rng)
    count = 2 if plan['mixed'] else int(rng.integers(1, min(3, n + decoys) + 1))
    result = _make_guides(rng, shape, colors, count, tail | {cap}, mixed=plan['mixed'])
    if result is None:
        return None
    systems, anchors = result
    fixed = tail | {cap} | set().union(*anchors.values())
    keys = list(systems)
    rng.shuffle(keys)
    budget = [160]

    def grow(objects, sources, targets, trace, origins, incoming):
        i = len(objects)
        if i == n:
            # Ensure the final ray escapes without meeting moved carriers.
            try:
                _, final, _ = _next_hit(shape, origins, incoming, objects, set())
            except ValueError:
                return None
            if final & targets:
                return None
            return objects, sources, targets, trace | final
        choices = [incoming, (incoming + 1) % 4, (incoming - 1) % 4]
        if plan['turn'] and i == n - 1 and not any(o['side'] != (initial if j == 0 else objects[j - 1]['side']) for j, o in enumerate(objects)):
            choices = [d for d in choices if d != incoming]
        if plan['straight'] and i == 0:
            choices = [incoming]
        for _ in range(30):
            if budget[0] <= 0:
                return None
            budget[0] -= 1
            outgoing = int(rng.choice(choices))
            width = plan['widths'][i]
            local = _shape(rng, incoming, outgoing, len(origins), width)
            if local is None:
                continue
            mask, port, entry, reg, h, w = local
            key = keys[i % count] if i < count else int(rng.choice(keys))
            axis, guide = systems[key]
            dr, dc = DIRS[incoming]
            # First contact at one flat edge across the whole incoming strip.
            entry_first = min(entry)
            gap_options = list(range(1, max(shape)))
            rng.shuffle(gap_options)
            for gap in gap_options[:18]:
                hit = (origins[0][0] + (gap + 1) * dr, origins[0][1] + (gap + 1) * dc)
                offset = hit[0] - entry_first[0], hit[1] - entry_first[1]
                cells, placed_port = _shift(mask, offset), _shift(port, offset)
                registration = (reg[0] + offset[0], reg[1] + offset[1])
                delta = [0, 0]
                delta[axis] = guide - registration[axis]
                target = _shift(cells, delta)
                if not _inside(cells, shape) or not _inside(target, shape):
                    continue
                if not _separate(cells, sources | fixed) or cells & (trace | targets):
                    continue
                approach = {(r + k * dr, c + k * dc) for r, c in origins for k in range(1, gap + 1)}
                if approach & (sources | targets) or target & (fixed | sources | targets | trace | approach):
                    continue
                if (i == 0 or plan['all_move']) and delta == [0, 0]:
                    continue
                obj = _colored_object(cells, placed_port, registration, key, outgoing, delta, rng, colors)
                new_origins = sorted(_shift(placed_port, delta))
                attempt = grow(objects + [obj], sources | cells, targets | target,
                               trace | approach, new_origins, outgoing)
                if attempt is not None:
                    return attempt
        return None

    route = grow([], set(), set(), tail | {cap}, [cap], initial)
    if route is None:
        return None
    objects, sources, targets, trace = route
    used = {o['key'] for o in objects}
    unused = [key for key in keys if key not in used]
    for j in range(decoys):
        for _ in range(150):
            side = int(rng.integers(4))
            local = _shape(rng, side, side, 1, int(rng.choice([1, 2])))
            if local is None:
                continue
            mask, port, entry, reg, h, w = local
            if h + 2 >= shape[0] or w + 2 >= shape[1]:
                continue
            offset = int(rng.integers(1, shape[0] - h)), int(rng.integers(1, shape[1] - w))
            cells, placed_port = _shift(mask, offset), _shift(port, offset)
            key = unused.pop(0) if unused else int(rng.choice(keys))
            axis, guide = systems[key]
            registration = reg[0] + offset[0], reg[1] + offset[1]
            delta = [0, 0]
            delta[axis] = guide - registration[axis]
            if not _inside(_shift(cells, delta), shape) or not _separate(cells, sources | fixed) or cells & (trace | targets):
                if key not in used and key not in unused:
                    unused.append(key)
                continue
            objects.append(_colored_object(cells, placed_port, registration, key, side, delta, rng, colors))
            sources |= cells
            used.add(key)
            break
        else:
            return None
    if unused:
        return None
    a = _render(shape, colors, anchors, tail, cap, objects)
    try:
        out, facts = _run(_decode(a))
    except ValueError:
        return None
    if len(facts['active']) != n or facts['decoys'] != decoys:
        return None
    return {'input': a, 'output': out}


def _fallback(rng, plan, colors):
    """Constructive monotone route with alternating vertical/horizontal slides.

    Retains the requested activation/decoy counts and port widths. This layout
    is only a bounded search fallback, never a changed transformation rule.
    """
    n, decoys = plan['n'], plan['decoys']
    shape = (30, 30)
    row = int(rng.integers(4, 8))
    guide = row + 9
    k0, k1 = colors[4:6]
    anchors = {k0: {(guide, 0)}}
    if n + decoys >= 2:
        anchors[k1] = {(29, 15)}
    if rng.random() < .5:
        anchors[k0].add((guide, 29))
    tail, cap = {(row, 0), (row, 1)}, (row, 2)
    objects, origins, incoming = [], [cap], 1
    for i in range(n):
        r0, c0 = origins[0][0] - 1, [5, 11, 19, 24][i]
        cells = {(r0 + r, c0 + c) for r in range(4) for c in range(4)}
        if i == 0 or i != 1 and rng.random() < .5:
            cells.remove((r0 + 3, c0 + 3))
        key = k1 if i == 1 else k0
        reg = (r0 + 3, c0 + 1) if i in (0, 1, 3) else (r0, c0 + 1)
        delta = (0, 15 - reg[1]) if i == 1 else (guide - reg[0], 0)
        width = plan['widths'][i]
        outgoing = 0 if plan['turn'] and i == n - 1 else 1
        port = {(r0, c0 + 1 + j) for j in range(width)} if outgoing == 0 else {(r0 + 1 + j, c0 + 3) for j in range(width)}
        if reg in port:
            reg = (r0, c0)  # same registration row, outside the top port
        obj = _colored_object(cells, port, reg, key, outgoing, delta, rng, colors)
        objects.append(obj)
        origins, incoming = sorted(_shift(port, delta)), outgoing
    # A vertical guide needs a matching carrier even in miss/single-hit scenes.
    missing = [key for key in anchors if not any(o['key'] == key for o in objects)]
    _need(decoys >= len(missing), 'fallback guide assignment exceeds object count')
    for j in range(decoys):
        r0, c0 = 23, [3, 10, 18][j]
        cells = {(r0 + r, c0 + c) for r in range(4) for c in range(4)}
        key = missing.pop(0) if missing else int(rng.choice(list(anchors)))
        reg, port = (r0 + 3, c0 + 1), {(r0, c0 + 1)}
        delta = (0, 15 - reg[1]) if key == k1 else (guide - reg[0], 0)
        objects.append(_colored_object(cells, port, reg, key, 0, delta, rng, colors))
    a = _render(shape, colors, anchors, tail, cap, objects)
    # Vary direction and handedness even when the packing fallback is used.
    a = np.rot90(a, int(rng.integers(4))).copy()
    if rng.random() < .5:
        a = a[:, ::-1].copy()
    return {'input': a, 'output': solve(a)}


def _different_alternative(decoded, out, mode):
    try:
        return not np.array_equal(_run(decoded, mode)[0], out)
    except ValueError:
        # An alternative can also contradict the family's unambiguous geometry.
        return True


def _nonrectangular(obj):
    h = max(r for r, c in obj['cells']) - min(r for r, c in obj['cells']) + 1
    w = max(c for r, c in obj['cells']) - min(c for r, c in obj['cells']) + 1
    return len(obj['cells']) < h * w


def _dormant_changes_output(decoded, out, activated):
    # Adding every dormant object's projection must change a visible pixel.
    return any(any(int(out[r + o['delta'][0], c + o['delta'][1]]) != v
                       for (r, c), v in o['values'].items())
               for j, o in enumerate(decoded['objects']) if j not in activated)


def _demo_ok(pair, plan):
    d = _decode(pair['input'])
    out, f = _run(d)
    if len(f['active']) != plan['n'] or f['decoys'] < plan['decoys']:
        return False
    if plan['demo']:
        if f['moved'] < 2 or not all(_different_alternative(d, out, m) for m in
                                   ('no_alignment', 'route_then_align', 'prealign', 'center_alignment', 'thin_ports')):
            return False
        if not any(_nonrectangular(d['objects'][j]) for j in f['active']):
            return False
        if not _dormant_changes_output(d, out, set(f['active'])):
            return False
        # Require an actual different reachable set under the static-port rule,
        # not merely a changed drawn location or an invalid alternative.
        try:
            static_f = _run(d, 'route_then_align')[1]
        except ValueError:
            return False
        if set(static_f['active']) == set(f['active']):
            return False
        widths = {t[2] for t in f['transitions']}
        if not {1, 2} <= widths:
            return False
        if plan['mixed'] and len({d['systems'][d['objects'][j]['key']][0] for j in f['active']}) < 2:
            return False
    return True


def _counts(num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, 'num_train'), (num_test, 1, 4, 'num_test')):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    base = list(map(int, rng.permutation(10)))
    stable = bool(rng.random() < .30)
    puzzle, seen, plans = {'train': [], 'test': []}, set(), []
    for i in range(num_train):
        n = int(rng.choice([2, 3])) if i < 2 else int(rng.choice([0, 1, 2, 3], p=[.12, .18, .35, .35]))
        widths = list(map(int, rng.choice([1, 2], size=n)))
        if i < 2:
            widths[:2] = list(map(int, rng.permutation([1, 2])))
        plans.append({'n': n, 'decoys': int(rng.integers(1, 4)) if i < 2 or n == 0 else int(rng.integers(4)),
                      'widths': widths, 'mixed': i == 0, 'turn': i == 1,
                      'straight': i == 0, 'all_move': i < 2, 'demo': i < 2})
    maximum = max(p['n'] for p in plans)
    for i in range(num_test):
        n = maximum + 1 if i == 0 else int(rng.integers(5))
        plans.append({'n': n, 'decoys': int(rng.integers(1, 4)) if n <= 1 else int(rng.integers(4)),
                      'widths': list(map(int, rng.choice([1, 2], size=n))), 'mixed': n >= 2 and bool(rng.random() < .5),
                      'turn': bool(n > 0 and rng.random() < .5), 'straight': False,
                      'all_move': False, 'demo': False})
    for i, plan in enumerate(plans):
        colors = base if stable or i == 0 else list(map(int, np.roll(base, 1))) if i == 1 else list(map(int, rng.permutation(10)))
        for attempt in range(24):
            pair = _try_scene(rng, plan, colors) if attempt < 20 else _fallback(rng, plan, colors)
            if pair is None or not _demo_ok(pair, plan):
                continue
            key = _signature(pair['input'])
            if key not in seen:
                break
        else:
            raise RuntimeError('bounded construction could not supply the requested demonstration')
        seen.add(key)
        puzzle['train' if i < num_train else 'test'].append(pair)
    order = rng.permutation(num_train)
    puzzle['train'] = [puzzle['train'][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _facts(pair):
    d = _decode(pair['input'])
    out, f = _run(d)
    _need(np.array_equal(out, pair['output']), 'incorrect output')
    _need(out.shape == np.asarray(pair['output']).shape, 'input/output shape mismatch')
    f.update(shape=d['grid'].shape, guides=len(d['systems']), initial=d['initial'],
             widths=tuple(t[2] for t in f['transitions']), axes=tuple(sorted({v[0] for v in d['systems'].values()})),
             body_sizes=tuple(sorted(len(o['cells']) for o in d['objects'])),
             palette=(d['bg'], d['signal'], d['pc']),
             marker_offsets=tuple((o['reg'][0] - min(r for r, c in o['cells']),
                                   o['reg'][1] - min(c for r, c in o['cells'])) for o in d['objects']))
    # Direct invariants of the final carriers, sources, references and beams.
    for color, points in d['anchors'].items():
        _need(all(int(out[p]) == color for p in points), 'guide changed')
    landed = set().union(*(set(v) for v in f['landed']))
    fixed = set().union(*d['anchors'].values())
    support = landed | f['trace'] | fixed
    _need(all(int(out[p]) == d['bg'] for p in np.ndindex(out.shape) if p not in support), 'unexplained output cell')
    for j, o in enumerate(d['objects']):
        if j not in f['active']:
            _need(all(int(out[p]) == d['bg'] for p in o['cells']), 'dormant source was not erased')
    for j, values in zip(f['active'], f['landed']):
        o = d['objects'][j]
        target_reg = (o['reg'][0] + o['delta'][0], o['reg'][1] + o['delta'][1])
        axis, guide = d['systems'][o['key']]
        _need(target_reg[axis] == guide and target_reg[1 - axis] == o['reg'][1 - axis], 'nonorthogonal registration')
        _need(Counter(values.values()) == Counter(o['values'].values()), 'carrier colors changed')
        _need(all(int(out[p]) == v for p, v in values.items()), 'carrier geometry or decoration changed')
    f['joint_witness'] = False
    if len(f['active']) >= 2 and f['moved'] >= 2 and f['decoys']:
        try:
            static = _run(d, 'route_then_align')[1]
            f['joint_witness'] = (set(static['active']) != set(f['active']) and
                                  _dormant_changes_output(d, out, set(f['active'])) and
                                  any(_nonrectangular(d['objects'][j]) for j in f['active']) and
                                  all(_different_alternative(d, out, m) for m in
                                      ('no_alignment', 'prealign', 'center_alignment', 'thin_ports')))
        except ValueError:
            pass
    return f


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'exact train/test schema required')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, records = set(), {'train': [], 'test': []}
    for split in records:
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'exact pair schema required')
            _grid(pair['input'])
            _grid(pair['output'])
            key = _signature(pair['input'])
            _need(key not in seen, 'duplicate input')
            seen.add(key)
            records[split].append(_facts(pair))
    train = records['train']
    _need(sum(f['joint_witness'] for f in train) >= 2, 'two joint causal demonstrations required')
    _need({1, 2} <= {w for f in train for w in f['widths']}, 'widths 1 and 2 must be demonstrated')
    _need(any(a == b for f in train for a, b, w in f['transitions']), 'straight emission not demonstrated')
    _need(any(a != b for f in train for a, b, w in f['transitions']), 'turn not demonstrated')
    _need(any(len(f['axes']) == 2 and f['joint_witness'] for f in train), 'mixed guides not demonstrated')
    _need(len(records['test'][0]['active']) == max(len(f['active']) for f in train) + 1, 'first test must extend the demonstrated causal chain')


def _normalized(a):
    mapping = {}
    return tuple(tuple(mapping.setdefault(int(c), len(mapping)) for c in row) for row in a)


def self_test():
    """Bounded 42-seed diversity suite plus count, mutation and symmetry checks."""
    fingerprints, shapes, structures, records = set(), set(), set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(tuple((_signature(p['input']), _signature(p['output'])) for s in puzzle for p in puzzle[s]))
        for split in puzzle:
            for pair in puzzle[split]:
                f = _facts(pair)
                f['split'] = split
                records.append(f)
                shapes.add(f['shape'])
                structures.add(_normalized(pair['input']))
                _need(np.array_equal(solve(pair['input'].copy()), pair['output']), 'input-only solve mismatch')
    _need(len(fingerprints) == 42 and len(shapes) >= 35 and len(structures) >= 150, 'insufficient structural variation')
    _need({f['initial'] for f in records} == {0, 1, 2, 3}, 'launcher orientation coverage')
    _need({0, 1, 2, 3, 4} <= {len(f['active']) for f in records}, 'activation-count coverage')
    _need({0, 1, 2, 3} <= {f['decoys'] for f in records}, 'decoy-count coverage')
    _need({1, 2, 3} <= {f['guides'] for f in records}, 'guide-count coverage')
    _need(len({f['body_sizes'] for f in records}) >= 50 and len({f['marker_offsets'] for f in records}) >= 50, 'carrier structure variation')
    _need(max(g for f in records for g in f['gaps']) >= 9, 'long approach coverage')
    # Input-only covariance: rotate, reflect and arbitrarily recolor every scene.
    rng = np.random.default_rng(9182)
    p = generate_puzzle(2071, 2, 1)
    for k in range(4):
        q = {'train': [], 'test': []}
        for split in p:
            for pair in p[split]:
                colors = rng.permutation(10)
                q[split].append({role: colors[np.rot90(grid, k)[:, ::-1]].copy() for role, grid in pair.items()})
        validate_generated_puzzle(q)
    # Every count/width/last-turn combination of the constructive fallback.
    fallback_cases = 0
    for n in range(5):
        for decoys in range(4):
            if n + decoys == 0:
                continue
            for turn in (False, True):
                for widths in ([1] * n, [2] * n, [1 + j % 2 for j in range(n)]):
                    plan = {'n': n, 'decoys': decoys, 'turn': turn, 'widths': widths}
                    f = _facts(_fallback(rng, plan, list(range(10))))
                    _need(len(f['active']) == n and f['decoys'] == decoys, 'fallback changed counts')
                    _need(f['widths'] == tuple(widths), 'fallback changed port widths')
                    fallback_cases += 1
    # Legal scenes beyond the constructor's sampled ceilings: five activations
    # on one guide, and a miss with four independently keyed systems.
    long_scene = np.zeros((30, 30), dtype=np.int64)
    long_scene[13, 0] = 4
    long_scene[4, :3] = [1, 1, 2]
    lane = 4
    for i, col in enumerate((5, 10, 15, 20, 25)):
        top = lane - 1
        long_scene[top:top + 4, col:col + 3] = 7
        registration_row = top + 3 if i % 2 == 0 else top
        long_scene[registration_row, col + 1] = 4
        long_scene[top + 1, col + 2] = 2
        lane = 13 - (registration_row - top) + 1
    _need(len(_run(_decode(long_scene))[1]['active']) == 5, 'long input-only route failed')
    four_guides = np.zeros((30, 30), dtype=np.int64)
    four_guides[28, :3] = [1, 1, 2]
    for key, row in zip((3, 4, 5, 6), (2, 8, 14, 20)):
        four_guides[row, 0] = key
        four_guides[row:row + 3, 6:9] = 7
        four_guides[row, 7] = 2
        four_guides[row + 1, 7] = key
    decoded = _decode(four_guides)
    _need(len(decoded['systems']) == 4 and not _run(decoded)[1]['active'], 'four-guide miss failed')
    for invalid in ('double_key', 'corner_port'):
        bad_input = long_scene.copy()
        if invalid == 'double_key':
            bad_input[3, 5] = 4
        else:
            bad_input[4, 7], bad_input[3, 5] = 7, 2
        try:
            solve(bad_input)
        except ValueError:
            pass
        else:
            raise AssertionError('ambiguous relay convention accepted')
    # All legal explicit counts and representative type/range failures.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            generate_puzzle(17, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4., 1), (4, None)):
        try:
            generate_puzzle(17, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p = generate_puzzle(31415)
    repeat = generate_puzzle(31415)
    _need(all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], repeat[s]) for k in x), 'nondeterministic seed')
    bad = copy.deepcopy(p)
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    return {'seeds': 42, 'distinct_puzzles': len(fingerprints), 'grid_shapes': len(shapes),
            'normalized_structures': len(structures), 'active_counts': sorted({len(f['active']) for f in records}),
            'guide_counts': sorted({f['guides'] for f in records}), 'fallback_cases': fallback_cases}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
