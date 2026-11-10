"""Socket-driven self-assembly of a port-directed signal route.

Only NumPy and the standard library are used. The solver decodes the scene,
enumerates rigid placements, and walks the socket graph from the visible tail.
Construction does not supply the solver with its route or palette.
"""
from collections import Counter
import json

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0, 359757353, 549469657, 685143930, 721415711, 262755259,
              122590798, 359618463, 741462539, 285334167, 403805021,
              986384865, 370167778, 997477101, 960400765, 486516729,
              293972702, 546986622, 671053318, 260876618, 684204744,
              317384741, 724338224, 18904746, 763223140, 538788954,
              94768032, 518779928, 707161693, 456700795, 59898,
              508716660, 905077136, 696499425, 281808050, 176805012,
              843727586, 210970047, 819636329, 830012310, 601697213,
              757886448)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid bounds")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)),
             "integer colors 0..9 required")
    return a.astype(np.int8, copy=False)


def _components(points):
    remaining, result = set(points), []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        todo, component = [p], {p}
        while todo:
            r, c = todo.pop()
            for dr, dc in DIRS:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
                    component.add(q)
        result.append(component)
    return result


def _transform(cells, t):
    raw = []
    for (r, c), v in cells.items():
        if t >= 4:
            c = -c
        for _ in range(t % 4):
            r, c = c, -r
        raw.append((r, c, v))
    r0, c0 = min(r for r, c, v in raw), min(c for r, c, v in raw)
    return {(r - r0, c - c0): v for r, c, v in raw}


def _shift(cells, r, c):
    return {(rr + r, cc + c): v for (rr, cc), v in cells.items()}


def _sig(cells):
    return tuple(sorted(_transform(cells, 0).items()))


def _key(points):
    cells = {p: 1 for p in points}
    return min(_sig(_transform(cells, t)) for t in range(8))


def _halo(points):
    return {(r + dr, c + dc) for r, c in points
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _port(cells, color, core_colors):
    core = {p for p, v in cells.items() if v in core_colors}
    patch = {p for p, v in cells.items() if v == color}
    _require(core and 1 <= len(patch) <= 2, "missing core or port")
    r0, r1 = min(r for r, c in core), max(r for r, c in core)
    c0, c1 = min(c for r, c in core), max(c for r, c in core)
    sides = [d for d, ok in enumerate((
        all(r == r0 - 1 and c0 <= c <= c1 for r, c in patch),
        all(c == c1 + 1 and r0 <= r <= r1 for r, c in patch),
        all(r == r1 + 1 and c0 <= c <= c1 for r, c in patch),
        all(c == c0 - 1 and r0 <= r <= r1 for r, c in patch))) if ok]
    _require(len(sides) == 1, "port must protrude from one core side")
    d = sides[0]
    axis = sorted(c if d % 2 == 0 else r for r, c in patch)
    _require(axis == list(range(axis[0], axis[0] + len(axis))), "broken port")
    dr, dc = DIRS[d]
    _require(all((r - dr, c - dc) in core for r, c in patch), "unsupported port")
    return patch, d


def _decode(inp):
    inp = _grid(inp)
    h, w = inp.shape
    bg = int(inp[0, 0])
    _require(all(int(inp[p]) == bg for p in ((0, w - 1), (h - 1, 0), (h - 1, w - 1))),
             "corners identify background")
    objects = _components(map(tuple, np.argwhere(inp != bg)))
    boundary = [o for o in objects if any(r in (0, h - 1) or c in (0, w - 1) for r, c in o)]
    _require(len(boundary) == 1, "one boundary launcher required")
    launcher = boundary[0]
    ends = [p for p in launcher if p[0] in (0, h - 1) or p[1] in (0, w - 1)]
    _require(len(ends) == 1, "launcher has one boundary end")
    r, c = ends[0]
    direction = 2 if r == 0 else 0 if r == h - 1 else 1 if c == 0 else 3
    dr, dc = DIRS[direction]
    line = [(r + k * dr, c + k * dc) for k in range(len(launcher))]
    _require(set(line) == launcher and 3 <= len(line) <= 6, "straight tail and cap")
    signal, inlet = int(inp[line[0]]), int(inp[line[-1]])
    _require(signal != inlet and all(int(inp[p]) == signal for p in line[:-1]), "tail/cap colors")
    targets, sources = [], []
    for obj in objects:
        if obj == launcher:
            continue
        cells = {p: int(inp[p]) for p in obj}
        if len(set(cells.values())) == 1:
            targets.append(obj)
        else:
            sources.append(cells)
    _require(targets and sources, "need loose modules and socket stencils")
    reg_colors = {int(inp[p]) for target in targets for p in target}
    _require(len(reg_colors) == 1, "one registration color")
    reg = next(iter(reg_colors))
    bodies = set().union(*(set(s.values()) for s in sources)) - {signal, inlet, reg}
    _require(len(bodies) == 1, "one body color")
    body = next(iter(bodies))
    _require(len({bg, signal, inlet, reg, body}) == 5, "five distinct visible roles")
    keys = []
    for source in sources:
        _require(set(source.values()) == {body, reg, signal, inlet}, "module roles")
        ip, idir = _port(source, inlet, {body, reg})
        op, odir = _port(source, signal, {body, reg})
        _require(idir != odir, "inlet and outlet are different sides")
        marks = {p for p, v in source.items() if v == reg}
        core = {p for p, v in source.items() if v in {body, reg}}
        _require(len(_components(core)) == 1, "connected module core")
        _require(len(_components(marks)) == 1 and 1 <= len(marks) <= 5, "connected key")
        _require(sum(v == body for v in source.values()) >= 3, "visible body")
        keys.append(_key(marks))
    _require(len(set(keys)) == len(keys), "one reusable template per key geometry")
    _require(all(_key(t) in keys for t in targets), "unmatched socket")
    # Eight-neighbor separation prevents accidental visual grouping.
    for i, obj in enumerate(objects):
        _require(not any(_halo(obj) & other for other in objects[i + 1:]), "objects too close")
    return dict(inp=inp, bg=bg, signal=signal, inlet=inlet, reg=reg, body=body,
                sources=sources, targets=targets, cap=line[-1], tail=set(line), initial=direction)


def _projection_hit(origins, direction, targets, shape):
    """First original stencil hit in every lane, with no partial contact."""
    dr, dc = DIRS[direction]
    stops = []
    for r, c in origins:
        edge = r + 1 if direction == 0 else shape[1] - c if direction == 1 else shape[0] - r if direction == 2 else c + 1
        hits = [(edge, -1)]
        for i, target in enumerate(targets):
            for rr, cc in target:
                distance = (rr - r) * dr + (cc - c) * dc
                if distance > 0 and (rr - r) * dc == (cc - c) * dr:
                    hits.append((distance, i))
        stops.append(min(hits))
    _require(len(set(stops)) == 1, "partial-width or split socket hit")
    return stops[0]


def _candidates(info, target_index, origins, direction, use_compass=True):
    target = info['targets'][target_index]
    others = set().union(*(t for i, t in enumerate(info['targets']) if i != target_index))
    h, w = info['inp'].shape
    dr, dc = DIRS[direction]
    lane = sorted(r * dc - c * dr for r, c in origins)
    results = {}
    for i, source in enumerate(info['sources']):
        for t in range(8):
            cells = _transform(source, t)
            marks = {p for p, v in cells.items() if v == info['reg']}
            if len(marks) != len(target):
                continue
            a, b = min(marks), min(target)
            cells = _shift(cells, b[0] - a[0], b[1] - a[1])
            if {p for p, v in cells.items() if v == info['reg']} != target:
                continue
            ip, idir = _port(cells, info['inlet'], {info['body'], info['reg']})
            if use_compass and (idir != (direction + 2) % 4 or len(ip) != len(origins)):
                continue
            if use_compass and sorted(r * dc - c * dr for r, c in ip) != lane:
                continue
            distance = (min(ip)[0] - min(origins)[0]) * dr + (min(ip)[1] - min(origins)[1]) * dc
            if (use_compass and distance <= 0) or any(not (0 < r < h - 1 and 0 < c < w - 1) for r, c in cells):
                continue
            if set(cells) & (others | info['tail']):
                continue
            op, odir = _port(cells, info['signal'], {info['body'], info['reg']})
            sig = tuple(sorted(cells.items()))
            if sig not in results:
                results[sig] = dict(cells=cells, ip=ip, op=op, odir=odir,
                                    distance=distance, source=i, transforms=[])
            results[sig]['transforms'].append(t)
    return list(results.values())


def _run(inp):
    info = _decode(inp)
    out = np.full_like(info['inp'], info['bg'])
    for p in info['tail']:
        out[p] = info['signal']
    origins, direction = {info['cap']}, info['initial']
    reached, installed, segments, records = [], set(), [], []
    for _ in range(len(info['targets']) + 1):
        distance, target = _projection_hit(origins, direction, info['targets'], out.shape)
        dr, dc = DIRS[direction]
        if target < 0:
            segment = {(r + k * dr, c + k * dc) for r, c in origins for k in range(1, distance)}
            _require(not segment & installed, "ray crosses an installed module")
            for p in segment:
                out[p] = info['signal']
            segments.append(segment)
            info.update(reached=reached, records=records, segments=segments)
            return out, info
        _require(target not in reached, "cyclic socket route")
        candidates = _candidates(info, target, origins, direction)
        _require(len(candidates) == 1, "socket must determine one complete rigid placement")
        chosen = candidates[0]
        # Ablate B's directional label: does registration alone leave distinct
        # activated footprints or outgoing rays? Color-only differences vanish
        # on activation and are deliberately not counted as evidence.
        unoriented = _candidates(info, target, origins, direction, use_compass=False)
        compass_needed = len({(tuple(sorted(c['cells'])), tuple(sorted(c['op'])), c['odir'])
                              for c in unoriented}) > 1
        segment = {(r + k * dr, c + k * dc) for r, c in origins for k in range(1, chosen['distance'])}
        _require(not (segment | set(chosen['cells'])) & installed, "installed module collision")
        for p in segment | set(chosen['cells']):
            out[p] = info['signal']
        reached.append(target)
        installed.update(chosen['cells'])
        records.append(dict(source=chosen['source'], incoming=direction, outgoing=chosen['odir'],
                            in_width=len(origins), out_width=len(chosen['op']),
                            transforms=chosen['transforms'], cells=chosen['cells'],
                            gap=chosen['distance'] - 1, compass_needed=compass_needed))
        segments.append(segment)
        origins, direction = chosen['op'], chosen['odir']
    raise ValueError("unterminated route")


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the output from this input alone; no generation state is used."""
    return _run(input_grid)[0]


def _polyomino(rng, n):
    points = {(0, 0)}
    for _ in range(200):
        if len(points) >= n:
            break
        r, c = sorted(points)[int(rng.integers(len(points)))]
        dr, dc = DIRS[int(rng.integers(4))]
        trial = points | {(r + dr, c + dc)}
        if max(r for r, c in trial) - min(r for r, c in trial) <= 3 and max(c for r, c in trial) - min(c for r, c in trial) <= 3:
            points = trial
    _require(len(points) == n, "bounded key growth exhausted")
    return set(_transform({p: 1 for p in points}, int(rng.integers(8))))


def _make_module(rng, palette, in_width, out_width, n, banned, size):
    bg, signal, inlet, reg, body = palette
    for _ in range(160):
        pattern = _polyomino(rng, n)
        if _key(pattern) in banned:
            continue
        ph, pw = max(r for r, c in pattern) + 1, max(c for r, c in pattern) + 1
        h = int(rng.integers(max(3, ph + 1), max(size, ph + 1) + 1))
        w = int(rng.integers(max(3, pw + 1), max(size, pw + 1) + 1))
        r0, c0 = int(rng.integers(h - ph + 1)), int(rng.integers(w - pw + 1))
        marks = {(r + r0, c + c0) for r, c in pattern}
        entry = int(rng.integers(4))
        # Match a full incoming strip to a straight extreme edge of the key.
        coordinate = min(r for r, c in marks) if entry == 0 else max(c for r, c in marks) if entry == 1 else max(r for r, c in marks) if entry == 2 else min(c for r, c in marks)
        edge = {c if entry % 2 == 0 else r for r, c in marks
                if (r if entry % 2 == 0 else c) == coordinate}
        starts = [x for x in sorted(edge) if all(x + k in edge for k in range(in_width))]
        if not starts:
            continue
        start = starts[int(rng.integers(len(starts)))]
        ip = ({(-1, start + k) for k in range(in_width)} if entry == 0 else
              {(start + k, w) for k in range(in_width)} if entry == 1 else
              {(h, start + k) for k in range(in_width)} if entry == 2 else
              {(start + k, -1) for k in range(in_width)})
        outgoing = int(rng.choice([d for d in range(4) if d != entry]))
        extent = w if outgoing % 2 == 0 else h
        start2 = int(rng.integers(extent - out_width + 1))
        op = ({(-1, start2 + k) for k in range(out_width)} if outgoing == 0 else
              {(start2 + k, w) for k in range(out_width)} if outgoing == 1 else
              {(h, start2 + k) for k in range(out_width)} if outgoing == 2 else
              {(start2 + k, -1) for k in range(out_width)})
        # A connected random core, retaining the whole registration and supports.
        supports = set(marks)
        for p in ip:
            dr, dc = DIRS[entry]
            supports.add((p[0] - dr, p[1] - dc))
        for p in op:
            dr, dc = DIRS[outgoing]
            supports.add((p[0] - dr, p[1] - dc))
        supports.update({(0, int(rng.integers(w))), (h - 1, int(rng.integers(w))),
                         (int(rng.integers(h)), 0), (int(rng.integers(h)), w - 1)})
        core = {min(marks)}
        for er, ec in sorted(supports):
            r, c = min(core, key=lambda p: (abs(p[0] - er) + abs(p[1] - ec), p))
            while (r, c) != (er, ec):
                if r != er and (c == ec or rng.random() < .5):
                    r += 1 if er > r else -1
                else:
                    c += 1 if ec > c else -1
                core.add((r, c))
        density = float(rng.uniform(.45, .95))
        for p in [(r, c) for r in range(h) for c in range(w)]:
            if rng.random() < density:
                core.add(p)
        # Discard accidental disconnected additions rather than joining by rectangles.
        core = set.union(*[comp for comp in _components(core) if comp & supports])
        if len(_components(core)) != 1 or len(core - marks) < 3:
            continue
        cells = {p: reg if p in marks else body for p in core}
        cells.update({p: inlet for p in ip})
        cells.update({p: signal for p in op})
        cells = _transform(cells, 0)
        # Check local orientation uniqueness using an actual receiver and approach.
        placed = _shift(cells, 8, 8)
        ips, entry2 = _port(placed, inlet, {reg, body})
        approach = (entry2 + 2) % 4
        dr, dc = DIRS[approach]
        origins = {(r - 3 * dr, c - 3 * dc) for r, c in ips}
        target = {p for p, v in placed.items() if v == reg}
        local = dict(inp=np.zeros((30, 30), dtype=np.int8), sources=[cells],
                     targets=[target], reg=reg, body=body, inlet=inlet, signal=signal, tail=set())
        if len(_candidates(local, 0, origins, approach)) == 1:
            return cells
    return None


def _plan_route(rng, tools, palette, sequence, shape, initial):
    h, w = shape
    bg, signal, inlet, reg, body = palette
    width = len(_port(tools[sequence[0]], inlet, {body, reg})[0]) if sequence else 1
    # The initial tail is always one pixel wide; width-two entry follows a relay.
    if width != 1:
        return None
    dr, dc = DIRS[initial]
    line = int(rng.integers(3, (w if initial % 2 == 0 else h) - 3))
    start = (h - 1, line) if initial == 0 else (line, 0) if initial == 1 else (0, line) if initial == 2 else (line, w - 1)
    length = int(rng.integers(2, 5))
    tail = {(start[0] + k * dr, start[1] + k * dc) for k in range(length)}
    cap = (start[0] + length * dr, start[1] + length * dc)
    occupied = tail | {cap}

    def extend(depth, origins, direction, finals, traces, blocked, budget):
        if budget[0] <= 0:
            return None
        budget[0] -= 1
        if depth == len(sequence):
            _, index = _projection_hit(origins, direction,
                                       [{p for p, v in f.items() if v == reg} for f in finals], shape)
            if index >= 0:
                return None
            dist, _ = _projection_hit(origins, direction, [], shape)
            ddr, ddc = DIRS[direction]
            last = {(r + k * ddr, c + k * ddc) for r, c in origins for k in range(1, dist)}
            if last & set().union(*(set(f) for f in finals)):
                return None
            return finals, traces + [last]
        source = tools[sequence[depth]]
        options = []
        for t in rng.permutation(8):
            cells = _transform(source, int(t))
            ip, entry = _port(cells, inlet, {body, reg})
            if entry != (direction + 2) % 4 or len(ip) != len(origins):
                continue
            ddr, ddc = DIRS[direction]
            a, b = min(ip), min(origins)
            for gap in range(1, max(h, w)):
                delta = (b[0] + (gap + 1) * ddr - a[0], b[1] + (gap + 1) * ddc - a[1])
                placed = _shift(cells, *delta)
                if any(not (1 <= r < h - 1 and 1 <= c < w - 1) for r, c in placed):
                    continue
                if set(placed) & blocked or set(placed) & set().union(*traces):
                    continue
                segment = {(r + k * ddr, c + k * ddc) for r, c in origins for k in range(1, gap + 1)}
                if segment & set().union(*(set(f) for f in finals)):
                    continue
                op, odir = _port(placed, signal, {body, reg})
                options.append((placed, segment, op, odir))
        rng.shuffle(options)
        for placed, segment, op, odir in options[:45]:
            result = extend(depth + 1, op, odir, finals + [placed], traces + [segment],
                            blocked | _halo(placed), budget)
            if result is not None:
                return result
        return None

    result = extend(0, {cap}, initial, [], [], _halo(occupied), [100])
    if result is None:
        return None
    finals, traces = result
    return tail, cap, finals, traces


def _construct(rng, palette, kind, n, decoys, initial, large=False):
    bg, signal, inlet, reg, body = palette
    # Same-count, noncongruent keys prevent a dictionary based on pixel count.
    size = 6 if large else int(rng.choice([4, 5]))
    count = 2 if kind in ('reuse', 'width') else int(rng.integers(1, 4))
    key_size = int(rng.choice([3, 4, 5]))
    widths = [(1, 1)] * count if kind == 'reuse' else [(1, 2), (2, 1)] if kind == 'width' else [(1, 1)] * count
    if kind == 'random' and count >= 2 and rng.random() < .5:
        widths = [(1, 2), (2, 1)] + [(1, 1)] * (count - 2)
    tools, banned = [], set()
    for iw, ow in widths:
        tool = _make_module(rng, palette, iw, ow, key_size, banned, size)
        if tool is None:
            return None
        tools.append(tool)
        banned.add(_key(p for p, v in tool.items() if v == reg))
    sequence, width = [], 1
    for i in range(n):
        choices = [j for j, (iw, ow) in enumerate(widths) if iw == width]
        index = 0 if kind == 'reuse' else choices[int(rng.integers(len(choices)))]
        sequence.append(index)
        width = widths[index][1]
    low = 23 if n >= 3 else 18
    h, w = int(rng.integers(low, 31)), int(rng.integers(low, 31))
    if n <= 1 and rng.random() < .35:
        h, w = int(rng.integers(14, 20)), int(rng.integers(24, 31))
    try:
        plan = _plan_route(rng, tools, palette, sequence, (h, w), initial)
    except ValueError:
        return None
    if plan is None:
        return None
    tail, cap, finals, traces = plan
    targets = [{p for p, v in f.items() if v == reg} for f in finals]
    full_final = set().union(*(set(f) for f in finals))
    ray_cells = set().union(*traces) | tail | {cap}
    blocked = _halo(full_final | ray_cells)
    for _ in range(decoys):
        found = False
        for attempt in range(100):
            source = tools[int(rng.integers(count))]
            cells = _transform(source, int(rng.integers(8)))
            mh, mw = max(r for r, c in cells) + 1, max(c for r, c in cells) + 1
            r, c = int(rng.integers(1, h - mh)), int(rng.integers(1, w - mw))
            placed = _shift(cells, r, c)
            target = {p for p, v in placed.items() if v == reg}
            # A decoy is a feasible unused installation, not a clipped socket
            # that could be discarded just by looking at the grid boundary.
            if set(placed) & blocked:
                continue
            targets.append(target)
            blocked.update(_halo(placed))
            found = True
            break
        if not found:
            return None
    inp = np.full((h, w), bg, dtype=np.int8)
    for target in targets:
        for p in target:
            inp[p] = reg
    for p in tail:
        inp[p] = signal
    inp[cap] = inlet
    # Sources share the field with sockets; no fixed inventory panel or slots.
    occupied = set().union(*targets) | tail | {cap}
    for tool in tools:
        found = False
        for attempt in range(150):
            cells = _transform(tool, int(rng.integers(8)))
            th, tw = max(r for r, c in cells) + 1, max(c for r, c in cells) + 1
            r, c = int(rng.integers(1, h - th)), int(rng.integers(1, w - tw))
            placed = _shift(cells, r, c)
            if set(placed) & _halo(occupied) or set(placed) & ray_cells:
                continue
            for p, v in placed.items():
                inp[p] = v
            occupied.update(placed)
            found = True
            break
        if not found:
            return None
    expected = np.full_like(inp, bg)
    for p in ray_cells | full_final:
        expected[p] = signal
    try:
        predicted, info = _run(inp)
    except ValueError:
        return None
    if not np.array_equal(expected, predicted) or len(info['reached']) != n:
        return None
    return {'input': inp, 'output': expected}, info


def _facts(info):
    records = info['records']
    keys = [_key(p for p, v in s.items() if v == info['reg']) for s in info['sources']]
    core_extents = []
    for record in records:
        core = [p for p, v in record['cells'].items() if v in {info['reg'], info['body']}]
        core_extents.append(max(max(p[a] for p in core) - min(p[a] for p in core) + 1 for a in (0, 1)))
    return dict(active=len(records), decoys=len(info['targets']) - len(records),
                initial=info['initial'], max_core=max(core_extents, default=0),
                reused=max(Counter(r['source'] for r in records).values(), default=0) >= 2,
                turned=any(r['incoming'] != r['outgoing'] for r in records),
                width_change=any(r['in_width'] != r['out_width'] for r in records),
                rotation=any(0 not in r['transforms'] and any(t < 4 for t in r['transforms']) for r in records),
                reflection=any(all(t >= 4 for t in r['transforms']) for r in records),
                compass_needed=any(r['compass_needed'] for r in records),
                same_count_keys=any(len(a) == len(b) and a != b for i, a in enumerate(keys) for b in keys[i + 1:]))


def _counts(a, b):
    for value, lo, hi in ((a, 2, 10), (b, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError('num_train must be 2..10; num_test must be 1..4 (integers)')
    if a <= b:
        raise ValueError('num_train must exceed num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)[:5]))
    vary_palette = bool(rng.random() < .65)
    regime = str(rng.choice(['longer', 'distractors', 'orientation', 'scale']))
    initial = int(rng.integers(4))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, number in (('train', num_train), ('test', num_test)):
        for index in range(number):
            kind = ('reuse' if index == 0 else 'width' if index == 1 else 'random') if split == 'train' else ('width' if regime == 'longer' else 'random')
            n = 2 if split == 'train' and index < 2 else int(rng.integers(0, 3))
            decoys = int(rng.integers(1, 3)) if n == 0 or kind == 'reuse' else int(rng.integers(0, 3))
            if split == 'test' and regime == 'longer':
                n = int(rng.integers(3, 5))
            elif split == 'test' and regime == 'distractors':
                n, decoys = int(rng.integers(0, 2)), int(rng.integers(2, 4))
                if index == 0:
                    decoys = 3
            elif split == 'test' and regime == 'scale':
                n = int(rng.integers(1, 3))
            direction = (initial + 1) % 4 if split == 'test' and regime == 'orientation' else initial if regime == 'orientation' else int(rng.integers(4))
            colors = list(map(int, rng.permutation(10)[:5])) if vary_palette else palette
            for attempt in range(260):
                result = _construct(rng, colors, kind, n, decoys, direction,
                                    large=split == 'test' and regime == 'scale')
                if result is None:
                    continue
                pair, info = result
                facts = _facts(info)
                if split == 'train' and index == 0 and not (facts['reused'] and facts['turned'] and facts['rotation'] and facts['compass_needed']):
                    continue
                if split == 'train' and index == 1 and not (facts['width_change'] and facts['reflection']):
                    continue
                if split == 'test' and regime == 'scale' and facts['max_core'] < 6:
                    continue
                key = (pair['input'].shape, pair['input'].tobytes())
                if key in seen:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError('bounded construction exhausted; requested complexity was retained')
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'train/test schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'pair lists required')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training, testing = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'same-sized output')
            key = (inp.shape, inp.tobytes())
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            prediction, info = _run(inp)
            _require(np.array_equal(prediction, out), 'incorrect socket assembly or trace')
            _require(set(map(int, np.unique(out))) == {info['bg'], info['signal']}, 'trace palette')
            for i, target in enumerate(info['targets']):
                _require(all(int(out[p]) == (info['signal'] if i in info['reached'] else info['bg']) for p in target),
                         'socket reachability filter')
                if i not in info['reached']:
                    _require(bool(_candidates(info, i, {info['cap']}, info['initial'], use_compass=False)),
                             'decoy is not a geometrically feasible installation')
            for record in info['records']:
                source = info['sources'][record['source']]
                _require(_key(source) == _key(record['cells']), 'rigid footprint preservation')
                _require(record['in_width'] in (1, 2) and record['out_width'] in (1, 2), 'port width')
                _require(record['outgoing'] != (record['incoming'] + 2) % 4, 'no reversal')
            if split == 'train':
                training.append(_facts(info))
            else:
                testing.append(_facts(info))
    # Coverage facts come from returned inputs, not sampling intentions.
    for field in ('reused', 'turned', 'width_change', 'rotation', 'reflection', 'same_count_keys', 'compass_needed'):
        _require(any(f[field] for f in training), 'training lacks ' + field)
    _require(any(f['active'] >= 2 for f in training), 'training lacks chaining')
    _require(any(f['decoys'] >= 1 for f in training), 'training lacks erasure (A ablation)')
    _require(any(max(f[k] for f in testing) > max(f[k] for f in training)
                 for k in ('active', 'decoys', 'max_core'))
             or bool({f['initial'] for f in testing} - {f['initial'] for f in training}),
             'missing observable train/test shift')


def _puzzle_key(puzzle):
    return tuple((np.asarray(p[k]).shape, np.asarray(p[k], dtype=np.int8).tobytes())
                 for s in ('train', 'test') for p in puzzle[s] for k in ('input', 'output'))


def self_test():
    """Bounded checks of 42 seeds, semantics, count API and useful diversity."""
    # The solver's grammar includes simple symmetric keys even though the
    # procedural sampler favors richer ones. Equivalent D4 identifiers must
    # collapse to one map, with no seed-dependent reflection preference.
    for key_length in (1, 2):
        inp = np.zeros((20, 20), dtype=np.int8)
        core_h = key_length + 2
        cells = {(r, c): 4 for r in range(core_h) for c in range(3)}
        for r in range(1, key_length + 1):
            cells[(r, 1)] = 3
        cells[(-1, 1)], cells[(core_h, 1)] = 2, 1
        source = _shift(_transform(cells, 0), 12, 12)
        final = _shift(cells, 5, 4)
        inp[0:2, 5], inp[2, 5] = 1, 2
        for p, v in source.items():
            inp[p] = v
        for p, v in final.items():
            if v == 3:
                inp[p] = v
        expected = np.zeros_like(inp)
        expected[:, 5] = 1
        for p in final:
            expected[p] = 1
        assert np.array_equal(solve(inp), expected)
    all_facts, shapes, structures, palettes, keys, directions = [], set(), set(), set(), set(), set()
    snapshots = []
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        snapshots.append(_puzzle_key(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                out, info = _run(pair['input'])
                assert np.array_equal(out, pair['output'])
                all_facts.append(_facts(info))
                shapes.add(out.shape)
                palettes.add(tuple(info[c] for c in ('bg', 'signal', 'inlet', 'reg', 'body')))
                directions.add(info['initial'])
                for source in info['sources']:
                    keys.add(_key(p for p, v in source.items() if v == info['reg']))
                roles = {}
                structures.add((out.shape, tuple(roles.setdefault(int(c), len(roles)) for c in pair['input'].flat)))
    assert len(set(snapshots)) == 42
    assert len(shapes) >= 30 and len(structures) >= 180 and len(palettes) >= 25
    assert len(keys) >= 10 and directions == {0, 1, 2, 3}
    assert {f['active'] for f in all_facts} >= {0, 1, 2, 3}
    assert any(f['active'] == 4 for f in all_facts)
    assert {f['decoys'] for f in all_facts} >= {0, 1, 2, 3}
    assert _puzzle_key(generate_puzzle(0)) == snapshots[0]
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            generate_puzzle(137 + 10 * a + b, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    corrupted = generate_puzzle(9, 2, 1)
    corrupted['test'][0]['output'] = corrupted['test'][0]['output'].copy()
    corrupted['test'][0]['output'][0, 0] = (int(corrupted['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except ValueError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    return dict(seeds=42, pairs=len(all_facts), shapes=len(shapes),
                structures=len(structures), palettes=len(palettes), key_geometries=len(keys))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
