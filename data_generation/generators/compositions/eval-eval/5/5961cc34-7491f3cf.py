"""Split-chamber print relays: local region selection controls global routing.

Only standard-library modules and NumPy are used. No files are read or written
by generation, solving, validation, or self_test. See task_spec.md for the rule.
"""

from collections import Counter
import hashlib
import numpy as np


BG, BODY, CUT, PORT, MARK = 0, 5, 8, 3, 4
INKS = (1, 2, 6, 7, 9)
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
KINDS = ("straight", "diagonal", "elbow", "step")
HOST_SEEDS = (0, 808305597, 965454150, 408503772, 956602311, 925788090,
              841286552, 935325463, 788284313, 964520751, 611559177,
              53944229, 375887386, 925710940, 913675492, 715354245,
              992538165, 402498081, 505415478, 889824797, 602549179,
              254764626, 999677328, 955214736, 955636634, 171901740,
              44321652, 305497939, 881569809, 656351687, 283813458,
              710073698, 142479252, 788367282, 824262266, 818787062,
              220077861, 521017522, 354016798, 401356188, 11142565, 321744212)


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "grid must contain integer colors 0..9")
    return a.astype(np.int8, copy=False)


def _counts(nt, nq):
    for value, lo, hi in ((nt, 2, 10), (nq, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    return int(nt), int(nq)


def _components(mask, diagonal=False):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    dirs = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                 if (r or c) and (diagonal or abs(r) + abs(c) == 1))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in dirs:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    cells.add(q)
                    stack.append(q)
        result.append(cells)
    return result


def _mask(shape, cells):
    a = np.zeros(shape, dtype=bool)
    for p in cells:
        a[p] = True
    return a


def _partition(cut):
    components = _components(~cut)
    _require(len(components) == 2, "cut must leave two four-connected regions")
    return [_mask(cut.shape, c) for c in components]


def _marker_sites(cut, region):
    h, w = cut.shape
    allowed = region.copy()
    allowed[[0, -1], :] = False
    allowed[:, [0, -1]] = False
    for r, c in np.argwhere(cut):
        allowed[max(0, r-1):min(h, r+2), max(0, c-1):min(w, c+2)] = False
    return [tuple(map(int, p)) for p in np.argwhere(allowed)]


def _path_valid(cut):
    """An eight-connected digital path, without diagonal elbow shortcuts."""
    cells = {tuple(map(int, p)) for p in np.argwhere(cut)}
    h, w = cut.shape
    graph = {}
    for r, c in cells:
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                q = r + dr, c + dc
                if not (dr or dc) or q not in cells:
                    continue
                if dr and dc and ((r+dr, c) in cells or (r, c+dc) in cells):
                    continue
                neighbors.append(q)
        graph[r, c] = neighbors
    ends = [p for p in cells if len(graph[p]) == 1]
    border = lambda p: p[0] in (0, h-1) or p[1] in (0, w-1)
    _require(len(ends) == 2 and all(len(v) in (1, 2) for v in graph.values()),
             "cut must be an unbranched open path")
    _require(all(border(p) == (p in ends) for p in cells), "cut has invalid boundary contacts")
    _require(len(_components(cut, True)) == 1, "disconnected cut")


def _line(p, q):
    r, c = p
    rr, cc = q
    dr, dc = abs(rr-r), abs(cc-c)
    sr, sc = (1 if rr > r else -1), (1 if cc > c else -1)
    error, points = dc-dr, []
    while True:
        points.append((r, c))
        if (r, c) == q:
            return points
        twice = 2*error
        if twice > -dr:
            error -= dr
            c += sc
        if twice < dc:
            error += dc
            r += sr


def _sample_cut(rng, h, w, kind):
    a = np.zeros((h, w), dtype=bool)
    if kind == "straight":
        a[:, int(rng.integers(2, w-2))] = True
    elif kind == "diagonal":
        p = (0, int(rng.integers(0, w-2)))
        q = (h-1, int(rng.integers(2, w)))
        for cell in _line(p, q):
            a[cell] = True
    elif kind == "elbow":
        r, c = int(rng.integers(2, h-2)), int(rng.integers(2, w-2))
        a[:r+1, c] = True
        a[r, c:] = True
    else:
        r = int(rng.integers(2, h-2))
        cols = rng.choice(np.arange(1, w-1), 2, replace=False)
        c, d = map(int, cols)
        a[:r+1, c] = True
        a[r, min(c, d):max(c, d)+1] = True
        a[r:, d] = True
    # Rotating just the local control broadens cuts without fixing exit sides.
    if rng.random() < .5:
        a = np.rot90(a).copy()
    if rng.random() < .5:
        a = a[:, ::-1].copy()
    if rng.random() < .5:
        a = a[::-1].copy()
    return a


def _port_options(region, direction, width):
    h, w = region.shape
    size = w if direction % 2 == 0 else h
    result = []
    for start in range(1, size-width):
        if direction == 0:
            cells = [(0, start+k) for k in range(width)]
        elif direction == 2:
            cells = [(h-1, start+k) for k in range(width)]
        elif direction == 1:
            cells = [(start+k, w-1) for k in range(width)]
        else:
            cells = [(start+k, 0) for k in range(width)]
        if all(region[p] for p in cells):
            result.append(cells)
    return result


def _pattern(rng, shape, p, q, teaching=False):
    h, w = shape
    rr, cc = np.indices(shape)
    style = 0 if teaching else int(rng.integers(7))
    offset = int(rng.integers(3))
    a = np.zeros(shape, dtype=np.int8)
    if style == 0:
        a[:] = np.where((rr+cc+offset) % 2, p, q)
    elif style == 1:
        axis = rr if rng.random() < .5 else cc
        a[(axis+offset) % 3 == 0] = p
        a[(axis+offset) % 3 == 1] = q
    elif style == 2:
        radius = abs(rr-h//2) + abs(cc-w//2)
        a[radius <= min(h, w)//2] = p
        a[(rr == h//2) | (cc == w//2)] = q
    elif style == 3:
        a[(rr == 1) | (rr == h-2) | (cc == 1) | (cc == w-2)] = p
        a[(rr+2*cc+offset) % 4 == 0] = q
    elif style == 4:
        a[(rr-cc+offset) % 3 == 0] = p
        a[(rr+cc+offset) % 4 == 0] = q
    elif style == 5:
        for _ in range(int(rng.integers(2, 6))):
            r, c = int(rng.integers(1, h-1)), int(rng.integers(1, w-1))
            a[r:min(h-1, r+int(rng.integers(1, 3))),
              c:min(w-1, c+int(rng.integers(1, 3)))] = p if rng.random() < .5 else q
    else:
        values = rng.random(shape)
        a[values < .22] = p
        a[(values >= .22) & (values < .52)] = q
    return a


def _assemble_gate(rng, cut, ports, selected_side, p, q, teaching=False):
    regions = _partition(cut)
    sites = _marker_sites(cut, regions[selected_side])
    marker = sites[int(rng.integers(len(sites)))]
    a = _pattern(rng, cut.shape, p, q, teaching)
    a[[0, -1], :] = BODY
    a[:, [0, -1]] = BODY
    a[cut] = CUT
    for cells, _ in ports:
        for cell in cells:
            a[cell] = PORT
    a[marker] = MARK
    return {"array": a, "ports": ports, "chosen": selected_side,
            "cut": cut, "regions": regions, "marker": marker}


def _new_gate(rng, direction, width, kinds, p, q, teaching=False):
    for _ in range(90):
        h, w = map(int, rng.integers(5, 10, size=2))
        kind = str(rng.choice(kinds))
        cut = _sample_cut(rng, h, w, kind)
        try:
            _path_valid(cut)
            regions = _partition(cut)
        except ValueError:
            continue
        if not all(_marker_sites(cut, region) for region in regions):
            continue
        choices = [(side, cells) for side, region in enumerate(regions)
                   for cells in _port_options(region, direction, width)]
        if not choices:
            continue
        side, selected = choices[int(rng.integers(len(choices)))]
        other_width = int(rng.integers(1, 3))
        alternatives = [(cells, d) for d in range(4)
                        for cells in _port_options(regions[1-side], d, other_width)]
        if not alternatives:
            continue
        other = alternatives[int(rng.integers(len(alternatives)))]
        ports = [None, None]
        ports[side] = selected, direction
        ports[1-side] = other
        return _assemble_gate(rng, cut, ports, side, p, q, teaching)
    return None


def _world_ports(gate, r, c, which=None):
    side = gate["chosen"] if which is None else which
    cells, direction = gate["ports"][side]
    return [(r+x, c+y) for x, y in cells], direction


def _hit(origins, direction, boxes, shape):
    """Project each beam lane onto original rectangles; require simultaneous hit."""
    h, w = shape
    stops = []
    for r, c in origins:
        limit = (r+1, w-c, h-r, c+1)[direction]
        candidates = [(limit, -1)]
        for i, (r0, c0, hh, ww) in enumerate(boxes):
            r1, c1 = r0+hh-1, c0+ww-1
            if direction == 0 and c0 <= c <= c1 and r0 < r:
                candidates.append((r-min(r1, r-1), i))
            elif direction == 2 and c0 <= c <= c1 and r1 > r:
                candidates.append((max(r0, r+1)-r, i))
            elif direction == 1 and r0 <= r <= r1 and c1 > c:
                candidates.append((max(c0, c+1)-c, i))
            elif direction == 3 and r0 <= r <= r1 and c0 < c:
                candidates.append((c-min(c1, c-1), i))
        stops.append(min(candidates))
    _require(len(set(stops)) == 1, "partial-width or unequal first contact")
    return stops[0]


def _trace(shape, tail, cap, initial, boxes, exits):
    trace = set(tail) | {cap}
    reached, transitions, gaps = [], [], []
    origins, direction = [cap], initial
    for _ in range(len(boxes)+1):
        distance, index = _hit(origins, direction, boxes, shape)
        dr, dc = DIRS[direction]
        trace.update((r+k*dr, c+k*dc) for r, c in origins for k in range(1, distance))
        if index == -1:
            return trace, reached, transitions, gaps
        _require(index not in reached, "relay cycle")
        reached.append(index)
        cells, outgoing = exits[index]
        transitions.append((direction, outgoing, len(cells)))
        gaps.append(distance-1)
        origins, direction = cells, outgoing
    raise ValueError("unterminated route")


def _cells(box, margin=0):
    r, c, h, w = box
    return {(rr, cc) for rr in range(r-margin, r+h+margin)
            for cc in range(c-margin, c+w+margin)}


def _launcher(rng, shape):
    h, w = shape
    d, length = int(rng.integers(4)), int(rng.integers(2, 4))
    line = int(rng.integers(3, (w if d % 2 == 0 else h)-3))
    start = ((h-1, line), (line, 0), (0, line), (line, w-1))[d]
    dr, dc = DIRS[d]
    tail = [(start[0]+k*dr, start[1]+k*dc) for k in range(length)]
    return tail, (start[0]+length*dr, start[1]+length*dc), d


def _place_options(shape, origins, incoming, gate, occupied, earlier_rays, gap_range):
    h, w = gate["array"].shape
    gh, gw = shape
    dr, dc = DIRS[incoming]
    transverse = sorted(c if incoming % 2 == 0 else r for r, c in origins)
    available = (w if incoming % 2 == 0 else h)-len(origins)
    options = []
    for gap in range(gap_range[0], gap_range[1]+1):
        r, c = origins[0][0]+(gap+1)*dr, origins[0][1]+(gap+1)*dc
        for offset in range(available+1):
            if incoming == 0:
                r0, c0 = r-h+1, transverse[0]-offset
            elif incoming == 2:
                r0, c0 = r, transverse[0]-offset
            elif incoming == 1:
                r0, c0 = transverse[0]-offset, c
            else:
                r0, c0 = transverse[0]-offset, c-w+1
            if not (1 <= r0 and r0+h < gh and 1 <= c0 and c0+w < gw):
                continue
            box = r0, c0, h, w
            if _cells(box, 1) & occupied or _cells(box) & earlier_rays:
                continue
            approach = {(rr+k*dr, cc+k*dc) for rr, cc in origins for k in range(1, gap+1)}
            if approach & occupied:
                continue
            options.append((box, approach))
    return options


def _render_constructed(shape, tail, cap, direction, gates, p, q):
    inp = np.zeros(shape, dtype=np.int8)
    boxes = [box for box, _ in gates]
    exits = [_world_ports(g, box[0], box[1]) for box, g in gates]
    trace, reached, _, _ = _trace(shape, tail, cap, direction, boxes, exits)
    for box, gate in gates:
        r, c, h, w = box
        inp[r:r+h, c:c+w] = gate["array"]
    for cell in tail:
        inp[cell] = p
    inp[cap] = q
    out = np.zeros_like(inp)
    for cell in trace:
        out[cell] = p
    for index in reached:
        box, gate = gates[index]
        r, c, h, w = box
        a, region = gate["array"], gate["regions"][gate["chosen"]]
        local = np.zeros_like(a)
        local[np.isin(a, (BODY, CUT, PORT, MARK))] = p
        local[(a == p) & region] = p
        local[(a == q) & ~region & ~gate["cut"]] = q
        out[r:r+h, c:c+w] = local
    return {"input": inp, "output": out}, reached


def _attempt_layout(rng, n, decoys, kinds, p, q, gap_range, teaching):
    lo = 16 if n == 1 else 21 if n == 2 else 26
    shape = tuple(map(int, rng.integers(lo, 31, size=2)))
    tail, cap, initial = _launcher(rng, shape)
    budget = [65]

    def grow(gates, occupied, rays, origins, incoming):
        if len(gates) == n:
            boxes = [b for b, g in gates]
            exits = [_world_ports(g, b[0], b[1]) for b, g in gates]
            try:
                result = _trace(shape, tail, cap, initial, boxes, exits)
            except ValueError:
                return None
            if result[1] != list(range(n)):
                return None
            if teaching and not any(a != b for a, b, _ in result[2]):
                return None
            return gates, result[0]
        for _ in range(14):
            if budget[0] <= 0:
                return None
            budget[0] -= 1
            outgoing = (incoming+int(rng.choice((-1, 0, 1)))) % 4
            width = 1+(len(gates) % 2) if teaching else int(rng.integers(1, 3))
            gate = _new_gate(rng, outgoing, width, kinds, p, q, teaching)
            if gate is None:
                continue
            options = _place_options(shape, origins, incoming, gate, occupied, rays, gap_range)
            if not options:
                continue
            box, approach = options[int(rng.integers(len(options)))]
            next_origins, next_dir = _world_ports(gate, box[0], box[1])
            result = grow(gates+[(box, gate)], occupied | _cells(box), rays | approach,
                          next_origins, next_dir)
            if result is not None:
                return result
        return None

    result = grow([], set(tail) | {cap}, set(tail) | {cap}, [cap], initial)
    if result is None:
        return None
    gates, trace = result
    occupied = set().union(*(_cells(box) for box, _ in gates)) | set(tail) | {cap}
    for _ in range(decoys):
        gate = _new_gate(rng, int(rng.integers(4)), int(rng.integers(1, 3)), kinds, p, q)
        if gate is None:
            return None
        h, w = gate["array"].shape
        for _ in range(90):
            r = int(rng.integers(1, shape[0]-h))
            c = int(rng.integers(1, shape[1]-w))
            box = r, c, h, w
            if not (_cells(box, 1) & occupied or _cells(box) & trace):
                gates.append((box, gate))
                occupied |= _cells(box)
                break
        else:
            return None
    pair, reached = _render_constructed(shape, tail, cap, initial, gates, p, q)
    return pair if reached == list(range(n)) else None


def _fallback(rng, n, decoys, kinds, p, q, gap_range):
    """Constructive packing, preserving route depth, decoys and gap regime."""
    shape = (30, 30)
    row = int(rng.integers(2, 9))
    long = gap_range[0] >= 5
    cols = [9, 21] if long else [4, 13, 22]
    gates = []
    for i in range(n):
        bent = "elbow" in kinds and ("straight" not in kinds or rng.random() < .5)
        cut = np.zeros((7, 7), dtype=bool)
        if bent:
            cut[:4, 3] = True
            cut[3, 3:] = True
        else:
            cut[3, :] = True
        regions = _partition(cut)
        selected = [(0, 4), (0, 5)] if i == n-1 else [(1, 6)]
        direction = 0 if i == n-1 else 1
        other = [(6, 1)]
        side = next(s for s, r in enumerate(regions) if r[selected[0]])
        ports = [None, None]
        ports[side] = selected, direction
        ports[1-side] = other, 2
        gate = _assemble_gate(rng, cut, ports, side, p, q, True)
        gates.append(((row, cols[i], 7, 7), gate))
    for j in range(decoys):
        cut = np.zeros((7, 7), dtype=bool)
        if "straight" in kinds:
            cut[:, 3] = True
        else:
            cut[:4, 3] = True
            cut[3, 3:] = True
        regions = _partition(cut)
        ports = [([(1, 0)], 3), ([(1, 6)], 1)]
        gate = _assemble_gate(rng, cut, ports, 0, p, q, True)
        gates.append(((20, 14+j*8, 7, 7), gate))
    pair, reached = _render_constructed(shape, [(row+1, 0), (row+1, 1)],
                                        (row+1, 2), 1, gates, p, q)
    _require(reached == list(range(n)), "fallback depth changed")
    # Whole-scene rotations preserve all locally visible conventions.
    k = int(rng.integers(4))
    return {name: np.rot90(a, k).copy() for name, a in pair.items()}


def _make_pair(rng, n, decoys, kinds, p, q, gap_range=(1, 7), teaching=False):
    for _ in range(12):
        pair = _attempt_layout(rng, n, decoys, kinds, p, q, gap_range, teaching)
        if pair is not None:
            return pair
    return _fallback(rng, n, decoys, kinds, p, q, gap_range)


def _decode(input_grid):
    a = _grid(input_grid)
    h, w = a.shape
    boundary = [(r, c) for r in range(h) for c in range(w)
                if (r in (0, h-1) or c in (0, w-1)) and a[r, c] != BG]
    _require(len(boundary) == 1, "need one boundary launcher cell")
    start = boundary[0]
    _require(start[0] not in (0, h-1) or start[1] not in (0, w-1), "corner launcher")
    d = 2 if start[0] == 0 else 0 if start[0] == h-1 else 1 if start[1] == 0 else 3
    p = int(a[start])
    _require(p in INKS, "invalid launcher pigment")
    dr, dc = DIRS[d]
    tail, current = [], start
    while 0 <= current[0] < h and 0 <= current[1] < w and a[current] == p:
        tail.append(current)
        current = current[0]+dr, current[1]+dc
    _require(2 <= len(tail) <= 4 and 0 <= current[0] < h and 0 <= current[1] < w,
             "launcher tail needs 2..4 cells")
    cap, q = current, int(a[current])
    _require(q in INKS and p != q, "cap must encode the second pigment")
    covered = set(tail) | {cap}
    gates, boxes, exits = [], [], []
    structure = np.isin(a, (BODY, CUT, PORT))
    for component in _components(structure, True):
        r0, c0 = min(r for r, c in component), min(c for r, c in component)
        r1, c1 = max(r for r, c in component), max(c for r, c in component)
        _require(0 < r0 < r1 < h-1 and 0 < c0 < c1 < w-1, "relay must be interior")
        local = a[r0:r1+1, c0:c1+1]
        hh, ww = local.shape
        _require(hh >= 5 and ww >= 5, "relay is too small")
        border = np.zeros(local.shape, dtype=bool)
        border[[0, -1], :] = True
        border[:, [0, -1]] = True
        _require(np.all(np.isin(local[border], (BODY, CUT, PORT))), "broken relay outline")
        _require(not np.any((local == BODY) & ~border), "body color is outline-only")
        _require(set(map(int, np.unique(local))) <= {BG, BODY, CUT, PORT, MARK, p, q},
                 "unknown relay pigment")
        cut = local == CUT
        _path_valid(cut)
        regions = _partition(cut)
        _require(all(_marker_sites(cut, region) for region in regions), "unreadable chamber")
        markers = [tuple(map(int, cell)) for cell in np.argwhere(local == MARK)]
        _require(len(markers) == 1, "each relay needs one marker")
        marker = markers[0]
        side = next((i for i, region in enumerate(regions) if region[marker]), -1)
        _require(side >= 0 and marker in _marker_sites(cut, regions[side]), "marker touches cut/border")
        ports = [None, None]
        port_parts = _components(local == PORT)
        _require(len(port_parts) == 2, "each relay needs two separate ports")
        for cells in port_parts:
            _require(1 <= len(cells) <= 2, "port width must be one or two")
            _require(all(not (r in (0, hh-1) and c in (0, ww-1)) for r, c in cells),
                     "ports must exclude corners")
            directions = [i for i, yes in enumerate((all(r == 0 for r, c in cells),
                          all(c == ww-1 for r, c in cells), all(r == hh-1 for r, c in cells),
                          all(c == 0 for r, c in cells))) if yes]
            _require(len(directions) == 1, "port must occupy one noncorner extreme side")
            port_side = next((i for i, region in enumerate(regions)
                              if all(region[cell] for cell in cells)), -1)
            _require(port_side >= 0 and ports[port_side] is None, "one port per chamber")
            ports[port_side] = sorted(cells), directions[0]
        _require(all(item is not None for item in ports), "missing chamber port")
        gate = {"array": local, "cut": cut, "regions": regions, "marker": marker,
                "chosen": side, "ports": ports}
        box = r0, c0, hh, ww
        _require(not (_cells(box) & covered), "overlapping scene objects")
        covered |= _cells(box)
        gates.append(gate)
        boxes.append(box)
        exits.append(_world_ports(gate, r0, c0))
    _require(gates, "no relays")
    _require(all(a[cell] == BG for cell in np.ndindex(a.shape) if cell not in covered),
             "unassigned foreground")
    return {"input": a, "p": p, "q": q, "tail": tail, "cap": cap,
            "initial": d, "gates": gates, "boxes": boxes, "exits": exits}


def _solve_decoded(d):
    a, p, q = d["input"], d["p"], d["q"]
    trace, reached, transitions, gaps = _trace(a.shape, d["tail"], d["cap"], d["initial"],
                                              d["boxes"], d["exits"])
    out = np.zeros_like(a)
    for cell in trace:
        out[cell] = p
    for i in reached:
        r0, c0, h, w = d["boxes"][i]
        g = d["gates"][i]
        selected = g["regions"][g["chosen"]]
        for r in range(h):
            for c in range(w):
                value = int(g["array"][r, c])
                if value in (BODY, CUT, PORT, MARK):
                    out[r0+r, c0+c] = p
                elif (value == p and selected[r, c]) or (value == q and not selected[r, c]):
                    out[r0+r, c0+c] = value
    return out, {"reached": reached, "trace": trace, "transitions": transitions, "gaps": gaps}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover roles, local partitions, chosen exits and all first hits from input."""
    return _solve_decoded(_decode(input_grid))[0]


def _swap_marker(pair):
    d = _decode(pair["input"])
    _, facts = _solve_decoded(d)
    i = facts["reached"][0]
    g, box = d["gates"][i], d["boxes"][i]
    sites = _marker_sites(g["cut"], g["regions"][1-g["chosen"]])
    # Choose an empty site if possible, so the intervention is especially clear.
    site = next((cell for cell in sites if g["array"][cell] == BG), sites[0])
    inp = pair["input"].copy()
    r, c = box[:2]
    old = g["marker"]
    inp[r+old[0], c+old[1]] = BG
    inp[r+site[0], c+site[1]] = MARK
    out = solve(inp)
    _require(not np.array_equal(out, pair["output"]), "ineffective marker intervention")
    return {"input": inp, "output": out}


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    p, q = map(int, rng.choice(INKS, 2, replace=False))
    regime = str(rng.choice(("depth", "cut", "spacing"), p=(.5, .3, .2)))
    kinds = ("straight", "diagonal") if regime == "cut" else KINDS
    train_gaps = (1, 3) if regime == "spacing" else (1, 7)
    for _ in range(12):
        first = _make_pair(rng, 2, 1, kinds, p, q, train_gaps, teaching=True)
        try:
            second = _swap_marker(first)
        except ValueError:
            continue
        # Require actual downstream reachability to change, not only local print.
        fa = _solve_decoded(_decode(first["input"]))[1]
        fb = _solve_decoded(_decode(second["input"]))[1]
        if set(fa["reached"]) != set(fb["reached"]) and len(fb["reached"]) <= 2:
            break
    else:
        first = _fallback(rng, 2, 1, kinds, p, q, train_gaps)
        second = _swap_marker(first)
    puzzle = {"train": [first, second], "test": []}
    seen = {_key(first["input"]), _key(second["input"])}
    for split, count in (("train", nt-2), ("test", nq)):
        for _ in range(count):
            n = 3 if split == "test" and regime == "depth" else int(rng.integers(1, 3))
            decoys = int(rng.integers(0, 3))
            allowed = ("elbow", "step") if split == "test" and regime == "cut" else kinds
            gaps = (5, 8) if split == "test" and regime == "spacing" else train_gaps
            for attempt in range(16):
                pair = _make_pair(rng, n, decoys, allowed, p, q, gaps)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("unable to make distinct input")
            seen.add(key)
            puzzle[split].append(pair)
    order = rng.permutation(nt)
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def _evidence(d, facts):
    flags = set()
    p, q = d["p"], d["q"]
    for i in facts["reached"]:
        g = d["gates"][i]
        a, region = g["array"], g["regions"][g["chosen"]]
        for name, mask in (("keep_p", (a == p) & region), ("drop_p", (a == p) & ~region),
                           ("keep_q", (a == q) & ~region), ("drop_q", (a == q) & region)):
            if np.any(mask):
                flags.add(name)
    if len(facts["reached"]) < len(d["gates"]):
        flags.add("erase_decoy")
    if len(facts["reached"]) >= 2:
        flags.add("chain")
    if any(a != b for a, b, width in facts["transitions"]):
        flags.add("turn")
    return flags


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, palettes, evidence, widths = set(), set(), set(), set()
    training = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape and not np.array_equal(inp, out), "invalid output shape/identity")
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            d = _decode(inp)
            expected, facts = _solve_decoded(d)
            _require(np.array_equal(out, expected), "incorrect routing or regional print")
            _require(set(map(int, np.unique(out))) <= {BG, d["p"], d["q"]}, "unlit hardware survived")
            for i, box in enumerate(d["boxes"]):
                r, c, h, w = box
                if i not in facts["reached"]:
                    _require(not np.any(out[r:r+h, c:c+w]), "unreached relay survived")
            palettes.add((d["p"], d["q"]))
            if split == "train":
                training.append((inp, out, d, facts))
                evidence |= _evidence(d, facts)
                widths.update(t[2] for t in facts["transitions"])
            else:
                _require({t[2] for t in facts["transitions"]} <= widths, "unshown test port width")
    _require(len(palettes) == 1, "pigment roles must stay fixed within puzzle")
    _require({"keep_p", "drop_p", "keep_q", "drop_q", "erase_decoy", "chain", "turn"} <= evidence,
             "training lacks family evidence")
    _require(widths == {1, 2}, "training must show both port widths")
    twins = False
    for i, (a, ao, da, fa) in enumerate(training):
        for b, bo, db, fb in training[i+1:]:
            if a.shape != b.shape or da["boxes"] != db["boxes"]:
                continue
            changes = a != b
            if (np.count_nonzero(changes) == 2 and
                    np.all((a[changes] == MARK) | (b[changes] == MARK)) and
                    set(fa["reached"]) != set(fb["reached"]) and not np.array_equal(ao, bo)):
                twins = True
    _require(twins, "missing marker intervention with changed reachability")
    return True


def _normalized_key(a):
    roles = {}
    return np.array([roles.setdefault(int(x), len(roles)) for x in a.flat],
                    dtype=np.int8).tobytes()


def self_test():
    """42 puzzles, streamed; metamorphic checks and deliberate corruptions."""
    shapes, structures, colors, controls = set(), set(), set(), set()
    directions, depths, decoys, bends, port_sides = set(), set(), set(), set(), set()
    statistics = Counter()
    first_puzzle = None
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        if first_puzzle is None:
            first_puzzle = puzzle
        colors.add(_decode(puzzle["train"][0]["input"])["p"])
        a_witness = b_witness = False
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                d = _decode(inp)
                predicted, f = _solve_decoded(d)
                assert np.array_equal(predicted, out)
                shapes.add(inp.shape)
                structures.add((inp.shape, hashlib.sha256(_normalized_key(inp)).digest()))
                directions.add(d["initial"])
                depths.add(len(f["reached"]))
                decoys.add(len(d["gates"])-len(f["reached"]))
                statistics[split+"_pairs"] += 1
                statistics[split+"_active_"+str(len(f["reached"]))] += 1
                if split == "train":
                    # Omit A: light every relay's hardware, including a decoy.
                    all_lit = out.copy()
                    for i, (r, c, h, w) in enumerate(d["boxes"]):
                        if i not in f["reached"]:
                            hardware = np.isin(d["gates"][i]["array"], (BODY, CUT, PORT, MARK))
                            all_lit[r:r+h, c:c+w][hardware] = d["p"]
                    a_witness |= not np.array_equal(all_lit, out)
                    # Omit B's mask but keep the correct route and hardware.
                    unfiltered = out.copy()
                    for i in f["reached"]:
                        r, c, h, w = d["boxes"][i]
                        local = d["gates"][i]["array"]
                        ink = (local == d["p"]) | (local == d["q"])
                        unfiltered[r:r+h, c:c+w][ink] = local[ink]
                    b_witness |= not np.array_equal(unfiltered, out)
                for g in d["gates"]:
                    controls.add((g["cut"].shape, g["cut"].tobytes()))
                    cells = np.argwhere(g["cut"])
                    bends.add(len(set(map(int, cells[:, 0]))) > 1 and
                              len(set(map(int, cells[:, 1]))) > 1)
                    port_sides.update(item[1] for item in g["ports"])
                # The entire rule commutes with rotations and reflection.
                if split == "test":
                    for k in (1, 2, 3):
                        assert np.array_equal(solve(np.rot90(inp, k)), np.rot90(out, k))
                    assert np.array_equal(solve(inp[:, ::-1]), out[:, ::-1])
                    # Swapping the visible pigment labels swaps exact output labels.
                    p, q = d["p"], d["q"]
                    recolor = np.arange(10, dtype=np.int8)
                    recolor[p], recolor[q] = q, p
                    assert np.array_equal(solve(recolor[inp]), recolor[out])
        assert a_witness and b_witness, "one parent concept has no answer-changing witness"
        again = generate_puzzle(seed)
        assert all(_key(x[k]) == _key(y[k]) for s in puzzle for x, y in zip(puzzle[s], again[s])
                   for k in ("input", "output"))
    assert len(shapes) >= 20 and len(structures) >= 130 and len(controls) >= 60
    assert directions == set(range(4)) and port_sides == set(range(4))
    assert {1, 2, 3} <= depths and {0, 1, 2} <= decoys and bends == {False, True}
    assert colors == set(INKS)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(713+nt*10+nq, nt, nq))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in first_puzzle[s]]
              for s in ("train", "test")}
    broken["test"][0]["output"][0, 0] = BODY
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    statistics.update(distinct_shapes=len(shapes), distinct_input_structures=len(structures),
                      distinct_controls=len(controls), seeds=42)
    return dict(statistics)
