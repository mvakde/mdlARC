"""Object elections, port-weighted cancellation, and a two-ended unary gauge.

Only standard-library modules and NumPy are used. No parent code is imported.
Construction labels are made from placed-object/network records; solve parses
the finished input independently of those records.
"""
from __future__ import annotations

import heapq
from itertools import product
import numpy as np

_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
_DIAMOND = ((0, 1), (1, 0), (1, 2), (2, 1))
_SQUARE = ((0, 0), (0, 1), (1, 0), (1, 1))
_HOST_SEEDS = (0, 474881429, 355942629, 515499047, 923990507, 808007205,
               410637671, 45158530, 613693187, 715139447, 74878602, 328929390,
               230374173, 72668500, 969137433, 566558959, 665035799, 763683256,
               4952336, 208805600, 685974769, 258588580, 224507724, 435011282,
               572466729, 326869927, 702707199, 824503843, 536670370, 91600876,
               466210360, 231981707, 991282049, 801806873, 799727339, 333691960,
               414942871, 639352472, 874805860, 770923018, 733147887, 790462297)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "grid colors")
    return a.astype(np.int8, copy=False)


def _key(grid):
    return grid.shape, np.ascontiguousarray(grid, dtype=np.int8).tobytes()


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    steps = _D8 if diagonal else _D4
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(cells)
    return result


def _rect(r, c, h, w):
    return {(i, j) for i in range(r, r+h) for j in range(c, c+w)}


def _neighbors(cells, h, w, diagonal=False):
    steps = _D8 if diagonal else _D4
    return {(r+dr, c+dc) for r, c in cells for dr, dc in steps
            if 0 <= r+dr < h and 0 <= c+dc < w}


def _ports(box):
    r, c, h, w = box
    return ([(r-1, j) for j in range(c, c+w)] +
            [(i, c+w) for i in range(r, r+h)] +
            [(r+h, j) for j in range(c+w-1, c-1, -1)] +
            [(i, c-1) for i in range(r+h-1, r-1, -1)])


def _gauge(grid):
    """The field occurs only in one rectangle with two opposing interior keys."""
    candidates = []
    for field in map(int, np.unique(grid)):
        rr, cc = np.where(grid == field)
        r, c = int(rr.min()), int(cc.min())
        h, w = int(rr.max())-r+1, int(cc.max())-c+1
        if min(h, w) < 3:
            continue
        block = grid[r:r+h, c:c+w]
        if not all(np.all(edge == field) for edge in
                   (block[0], block[-1], block[:, 0], block[:, -1])):
            continue
        keys = np.argwhere(block != field)
        if len(keys) != 2 or block[tuple(keys[0])] == block[tuple(keys[1])]:
            continue
        a, b = [tuple(map(int, p)) for p in keys]
        if a[0] == b[0] and a[1] == 1 and b[1] == w-2 and w >= 7:
            step = (0, 1)
        elif a[1] == b[1] and a[0] == 1 and b[0] == h-2 and h >= 7:
            step = (1, 0)
        else:
            continue
        candidates.append(dict(field=field, box=(r, c, h, w), shape=(h, w),
                               anchors=(a, b), signals=tuple(int(block[p]) for p in (a, b)),
                               step=step))
    _require(len(candidates) == 1, "gauge must have one interpretation")
    return candidates[0]


def _interpret(grid, gauge, bg, fence, wire):
    h, w = grid.shape
    r, c, gh, gw = gauge['box']
    _require(r > 0 and c > 0 and r+gh < h and c+gw < w, "gauge needs exterior ports")
    signals = gauge['signals']
    frames = _components(grid == fence)
    _require(2 <= len(frames) <= 4, "need two to four chambers")
    bodies, boxes, ballots, local, cell_local, four_local = [], [], [], [], [], []
    objects, ownership = [], np.full(grid.shape, -1, int)
    for index, frame in enumerate(frames):
        rr, cc = zip(*frame)
        fr, fc, fh, fw = min(rr), min(cc), max(rr)-min(rr)+1, max(cc)-min(cc)+1
        body = _rect(fr, fc, fh, fw)
        rim = body - _rect(fr+1, fc+1, fh-2, fw-2)
        _require(min(fh, fw) >= 7 and set(frame) == rim, "chamber must be a rectangular outline")
        inside = grid[fr+1:fr+fh-1, fc+1:fc+fw-1]
        _require(set(map(int, inside.flat)) <= {bg, *signals}, "chamber interior roles")
        _require(all(np.all(edge == bg) for edge in
                     (inside[0], inside[-1], inside[:, 0], inside[:, -1])), "ballot padding")
        counts, areas, orth, chamber_objects = [], [], [], []
        owner = np.full(inside.shape, -1, int)
        for color in signals:
            comps = _components(inside == color, True)
            counts.append(len(comps))
            areas.append(int(np.sum(inside == color)))
            orth.append(len(_components(inside == color)))
            for cells in comps:
                serial = len(chamber_objects)
                for p in cells:
                    owner[p] = serial
                chamber_objects.append((color, tuple(cells)))
        _require(min(counts) >= 1 and counts[0] != counts[1], "local votes must be positive and strict")
        # Distinct objects cannot touch, even across colors and at corners.
        for ir, ic in np.argwhere(owner >= 0):
            ns = owner[max(0, ir-1):ir+2, max(0, ic-1):ic+2]
            _require(np.all((ns < 0) | (ns == owner[ir, ic])), "separate ballot objects")
        bodies.append(body)
        boxes.append((fr, fc, fh, fw))
        ballots.append(tuple(counts))
        local.append(int(counts[1] > counts[0]))
        cell_local.append(None if areas[0] == areas[1] else int(areas[1] > areas[0]))
        four_local.append(None if orth[0] == orth[1] else int(orth[1] > orth[0]))
        objects.append(chamber_objects)
        for p in body:
            ownership[p] = index
    gauge_body = _rect(r, c, gh, gw)
    _require(not any(body & gauge_body for body in bodies), "chamber/gauge overlap")
    all_bodies = set().union(*bodies)
    for i, body in enumerate(bodies):
        _require(not (_neighbors(body, h, w) & (all_bodies-body)), "chamber separation")
    ports = _ports(gauge['box'])
    collar = _rect(r-1, c-1, gh+2, gw+2)-gauge_body
    _require(all(grid[p] in (bg, wire) for p in collar), "gauge collar roles")
    _require(all(grid[p] == bg for p in collar-set(ports)), "no diagonal gauge contacts")
    routes = _components(grid == wire)
    _require(routes, "missing wires")
    weights, owners, port_labels = [0]*len(frames), [], []
    for cells in routes:
        adjacent = {int(ownership[p]) for p in _neighbors(cells, h, w) if ownership[p] >= 0}
        touched = set(cells) & set(ports)
        _require(len(adjacent) == 1 and touched, "every wire component needs one chamber and gauge ports")
        who = adjacent.pop()
        weights[who] += len(touched)
        owners.append(who)
        port_labels.extend([local[who]]*len(touched))
    _require(all(1 <= weight <= 4 for weight in weights), "chamber port weights")
    _require(3 <= len(port_labels) <= 9, "port count")
    votes = (port_labels.count(0), port_labels.count(1))
    _require(min(votes) >= 1 and votes[0] != votes[1], "global votes must be positive and strict")
    # No colored cells are allowed outside their identified semantic structures.
    occupied = gauge_body | all_bodies | {p for route in routes for p in route}
    _require(all(grid[p] == bg for p in np.ndindex(grid.shape) if p not in occupied), "unassigned cells")
    return dict(**gauge, background=bg, fence=fence, wire=wire, boxes=boxes,
                ballots=ballots, local=local, cell_local=cell_local, four_local=four_local,
                objects=objects, routes=routes, owners=owners, weights=weights,
                votes=votes, winner=int(votes[1] > votes[0]), margin=abs(votes[0]-votes[1]))


def _inspect(input_grid):
    grid = _grid(input_grid)
    gauge = _gauge(grid)
    colors, counts = np.unique(grid, return_counts=True)
    _require(len(colors) == 6, "six distinct roles are required")
    bg = int(colors[np.argmax(counts)])
    _require(bg not in (gauge['field'], *gauge['signals']), "background role")
    other = sorted(set(map(int, colors))-{bg, gauge['field'], *gauge['signals']})
    interpretations = []
    for fence, wire in (other, other[::-1]):
        try:
            interpretations.append(_interpret(grid, gauge, bg, fence, wire))
        except ValueError:
            continue
    _require(len(interpretations) == 1, "chambers/wires must have one interpretation")
    meta = interpretations[0]
    # More than half of all cells are background: role inference is
    # geometric plus this declared visible frequency constraint, never hidden.
    _require(int(np.sum(grid == bg)) > grid.size//2, "background must dominate")
    _require(_fits(meta, meta['margin']), "surplus must fit inside the gauge")
    return meta


def _fits(meta, count):
    origin = meta['anchors'][meta['winner']]
    dr, dc = meta['step']
    if meta['winner']:
        dr, dc = -dr, -dc
    r, c = origin[0]+2*(count-1)*dr, origin[1]+2*(count-1)*dc
    return 0 < r < meta['shape'][0]-1 and 0 < c < meta['shape'][1]-1


def _render(meta, count=None, winner=None):
    winner = meta['winner'] if winner is None else winner
    count = meta['margin'] if count is None else count
    out = np.full(meta['shape'], meta['field'], np.int8)
    r, c = meta['anchors'][winner]
    dr, dc = meta['step']
    if winner:
        dr, dc = -dr, -dc
    for i in range(count):
        rr, cc = r+2*i*dr, c+2*i*dc
        if not (0 < rr < out.shape[0]-1 and 0 < cc < out.shape[1]-1):
            return None
        out[rr, cc] = meta['signals'][winner]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the exact crop and vote surplus from this input alone."""
    return _render(_inspect(input_grid))


def _normalize(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _shape(rng, small=False):
    modes = (((0, 0),), ((0, 0), (0, 1)), ((0, 0), (1, 1)),
             _SQUARE, _DIAMOND, ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2)),
             ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)))
    if small or rng.random() < .75:
        cells = modes[int(rng.integers(3 if small else len(modes)))]
    else:
        target = int(rng.integers(3, 8))
        cells = {(1, 1)}
        while len(cells) < target:
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _D8
                               if 0 <= r+dr < 3 and 0 <= c+dc < 3}-cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    if rng.random() < .5:
        cells = [(r, -c) for r, c in cells]
    for _ in range(int(rng.integers(4))):
        cells = [(-c, r) for r, c in cells]
    return _normalize(cells)


def _chamber(rng, signals, majority, witness=False, compact=False):
    """Pack ballots before placing the chamber, retaining their known identities."""
    if witness:
        shapes = [(signals[majority], ((0, 0),))]*2
        shapes += [(signals[majority], ((0, 0), (0, 1)))]
        shapes += [(signals[1-majority], _DIAMOND), (signals[1-majority], _SQUARE)]
        counts = [0, 0]
        counts[majority], counts[1-majority] = 3, 2
    else:
        nmajor = int(rng.integers(2, 4))
        nminor = int(rng.integers(1, nmajor))
        if compact:
            nmajor, nminor = 2, 1
        counts = [0, 0]
        counts[majority], counts[1-majority] = nmajor, nminor
        colors = [signals[majority]]*nmajor+[signals[1-majority]]*nminor
        large_slots = set(map(int, rng.choice(len(colors), min(2, len(colors)), replace=False)))
        shapes = [(color, _shape(rng, compact or i not in large_slots)) for i, color in enumerate(colors)]
    rng.shuffle(shapes)
    shapes.sort(key=lambda item: -len(item[1]))
    for attempt in range(90):
        if witness:
            ah, aw = int(rng.integers(6, 8)), int(rng.integers(6, 8))
        else:
            lo, hi = (3, 6) if compact else (4, 8)
            ah, aw = int(rng.integers(lo, hi)), int(rng.integers(lo, hi))
        blocked = np.zeros((ah, aw), bool)
        placed = []
        for color, shape in shapes:
            sh, sw = 1+max(p[0] for p in shape), 1+max(p[1] for p in shape)
            nh, nw = ah-sh+1, aw-sw+1
            if min(nh, nw) <= 0:
                break
            allowed = np.ones((nh, nw), bool)
            for dr, dc in shape:
                allowed &= ~blocked[dr:dr+nh, dc:dc+nw]
            positions = np.argwhere(allowed)
            if not len(positions):
                break
            r, c = map(int, positions[int(rng.integers(len(positions)))])
            cells = tuple((r+dr+2, c+dc+2) for dr, dc in shape)
            placed.append((color, cells))
            for dr, dc in shape:
                ir, ic = r+dr, c+dc
                blocked[max(0, ir-1):ir+2, max(0, ic-1):ic+2] = True
        else:
            return dict(h=ah+4, w=aw+4, objects=placed, majority=majority, counts=counts)
    # A spacious packing preserves the sampled objects and counts if random
    # placements fail. These are object slots inside a new chamber, never whole
    # puzzle templates. Noncompact shapes have at most two 3x3 objects; all
    # remaining shapes fit in 2x2 slots.
    if compact:
        ah, aw, slots = 5, 5, [(0, 0), (0, 3), (3, 0)]
    else:
        ah, aw, slots = 6, 8, [(0, 0), (0, 4), (4, 0), (4, 3), (4, 6)]
    placed = [(color, tuple((r+dr+2, c+dc+2) for dr, dc in shape))
              for (color, shape), (r, c) in zip(shapes, slots)]
    return dict(h=ah+4, w=aw+4, objects=placed, majority=majority, counts=counts)


def _topology(rng, count, cap=6, lower=1, witness=False, second=False):
    if witness:
        weights, bits = [4, 1, 1], [1, 0, 0]
        order = rng.permutation(3)
        return [weights[i] for i in order], [bits[i] for i in order]
    if second:
        # Two elected chambers jointly defeat the largest-port chamber:
        # (3+2):4 -> one mark. Together with 4:2 -> two marks, this defeats
        # both a constant tally and the largest-chamber shortcut with two
        # demonstrations. Assignment and geometry remain variable.
        weights, bits = [4, 3, 2], [0, 1, 1]
        order = rng.permutation(3)
        return [weights[i] for i in order], [bits[i] for i in order]
    options = []
    for weights in product(range(1, 5), repeat=count):
        if not 3 <= sum(weights) <= 9:
            continue
        for bits in product((0, 1), repeat=count):
            wins = sum(wt*bit for wt, bit in zip(weights, bits))
            margin = 2*wins-sum(weights)
            if 0 < sum(bits) < count and lower <= margin <= cap:
                options.append((weights, bits))
    _require(bool(options), "requested topology is impossible")
    weights, bits = options[int(rng.integers(len(options)))]
    return list(weights), list(bits)


def _layout(rng, chambers, gauge_h, gauge_w):
    count = len(chambers)
    low = 21 if count == 2 else 26
    h, w = int(rng.integers(low, 31)), int(rng.integers(max(low, gauge_w+8), 31))
    r = int(rng.integers(max(3, h//3-2), min(h-gauge_h-3, 2*h//3)+1))
    c = int(rng.integers(2, w-gauge_w-1))
    gauge = (r, c, gauge_h, gauge_w)
    reserved = _rect(r-2, c-2, gauge_h+4, gauge_w+4)
    placed = []
    # Largest rectangles go first, but no rendered layout or slot bank is used.
    order = sorted(range(count), key=lambda i: -chambers[i]['h']*chambers[i]['w'])
    for index in order:
        ch = chambers[index]
        ph, pw = ch['h'], ch['w']
        for attempt in range(140):
            pr, pc = int(rng.integers(h-ph+1)), int(rng.integers(w-pw+1))
            if rng.random() < .7:
                side = int(rng.integers(4))
                pad = int(rng.integers(0, 3))
                if side == 0:
                    pr = pad
                elif side == 1:
                    pr = h-ph-pad
                elif side == 2:
                    pc = pad
                else:
                    pc = w-pw-pad
            cells = _rect(pr, pc, ph, pw)
            if cells & reserved:
                continue
            placed.append(dict(ch, r=pr, c=pc, body=cells, index=index))
            reserved |= cells | _neighbors(cells, h, w)
            break
        else:
            return None
    cr, cc = r+(gauge_h-1)/2, c+(gauge_w-1)/2
    placed.sort(key=lambda ch: float(np.arctan2(ch['r']+(ch['h']-1)/2-cr,
                                               ch['c']+(ch['w']-1)/2-cc)))
    return h, w, gauge, placed


def _path(start, targets, allowed, costs):
    if start in targets:
        return {start}
    if not allowed[start]:
        return None
    h, w = allowed.shape
    heap, dist, previous = [(0.0, *start)], {start: 0.0}, {}
    while heap:
        d, r, c = heapq.heappop(heap)
        p = r, c
        if d != dist[p]:
            continue
        if p in targets:
            cells = {p}
            while p != start:
                p = previous[p]
                cells.add(p)
            return cells
        for dr, dc in _D4:
            q = r+dr, c+dc
            if 0 <= q[0] < h and 0 <= q[1] < w and allowed[q]:
                nd = d+float(costs[q])
                if nd < dist.get(q, float('inf')):
                    dist[q], previous[q] = nd, p
                    heapq.heappush(heap, (nd, *q))
    return None


def _networks(rng, h, w, gauge, chambers, weights):
    r, c, gh, gw = gauge
    center = r+(gh-1)/2, c+(gw-1)/2
    # Sparse attachment candidates leave room between unrelated networks.
    ports = ([(r-1, j) for j in range(c, c+gw, 2)] +
             [(i, c+gw) for i in range(r, r+gh, 2)] +
             [(r+gh, j) for j in range(c, c+gw, 2)] +
             [(i, c-1) for i in range(r, r+gh, 2)])
    ports.sort(key=lambda p: float(np.arctan2(p[0]-center[0], p[1]-center[1])))
    chosen = sorted(map(int, rng.choice(len(ports), sum(weights), replace=False)))
    angles = [float(np.arctan2(ch['r']+(ch['h']-1)/2-center[0],
                               ch['c']+(ch['w']-1)/2-center[1])) for ch in chambers]
    choices = []
    for shift in range(len(chosen)):
        sequence = chosen[shift:]+chosen[:shift]
        groups, offset, score = [], 0, 0.0
        for angle, weight in zip(angles, weights):
            group = {ports[i] for i in sequence[offset:offset+weight]}
            for pr, pc in group:
                delta = abs(angle-float(np.arctan2(pr-center[0], pc-center[1])))
                score += min(delta, 2*np.pi-delta)**2
            offset += weight
            groups.append(group)
        choices.append((score+float(rng.random())*.7, groups))
    groups = min(choices, key=lambda v: v[0])[1]
    # Occasionally use adjacent ports of the same owner, while preserving weight.
    possible = set(_ports(gauge))
    for i, group in enumerate(groups):
        if len(group) >= 2 and rng.random() < .3:
            old = sorted(group)[int(rng.integers(len(group)))]
            candidates = sorted((_neighbors(group-{old}, h, w) & possible)-set().union(*groups))
            others = set().union(*(g for j, g in enumerate(groups) if j != i))
            candidates = [p for p in candidates if p not in _neighbors(others, h, w)]
            if candidates:
                group.remove(old)
                group.add(candidates[int(rng.integers(len(candidates)))])
    body = _rect(r, c, gh, gw)
    collar = _rect(r-1, c-1, gh+2, gw+2)-body
    all_chambers = set().union(*(ch['body'] for ch in chambers))
    networks = [set(g) for g in groups]
    for i in sorted(range(len(chambers)), key=lambda j: -weights[j]):
        own = chambers[i]['body']
        other_bodies = all_chambers-own
        other_nets = set().union(*(net for j, net in enumerate(networks) if j != i))
        blocked = body | collar | all_chambers | other_nets | _neighbors(other_bodies | other_nets, h, w)
        blocked -= groups[i]
        allowed = np.ones((h, w), bool)
        for p in blocked:
            allowed[p] = False
        targets = {p for p in _neighbors(own, h, w)-own if allowed[p]}
        if not targets:
            return None
        roughness = float(rng.choice((0., .5, 2., 5.)))
        costs = 1+roughness*rng.random((h, w))
        coarse = rng.random(((h+3)//4, (w+3)//4))
        costs += roughness*2*np.repeat(np.repeat(coarse, 4, axis=0), 4, axis=1)[:h, :w]
        net = set()
        for start in sorted(groups[i]):
            path = _path(start, targets | net, allowed, costs)
            if path is None:
                return None
            net |= path
        if rng.random() < .35:
            ch = chambers[i]
            fr, fc, fh, fw = ch['r'], ch['c'], ch['h'], ch['w']
            rim = _rect(max(0, fr-1), max(0, fc-1),
                        min(h, fr+fh+1)-max(0, fr-1), min(w, fc+fw+1)-max(0, fc-1))-own
            rim = {p for p in rim if allowed[p] and p not in collar}
            mask = np.zeros((h, w), bool)
            for p in rim | net:
                mask[p] = True
            for comp in _components(mask):
                if set(comp) & net:
                    net |= set(comp)
        if (net & collar) != groups[i]:
            return None
        networks[i] = net
    return networks


def _make_pair(rng, roles, count, winner, rotation, witness=False, second=False,
               cap=6, lower=1):
    bg, field, fence, wire, s0, s1 = roles
    signals = (s0, s1)
    weights, winning = _topology(rng, count, cap, lower, witness, second)
    chambers = [_chamber(rng, signals, winner if bit else 1-winner,
                         witness=witness and bool(bit), compact=count == 4)
                for bit in winning]
    if witness:
        # Reject a tied/wrong witness before routing it. Object arrangements
        # cannot change these counts, so geometric retries would be wasted.
        area_labels, orth_labels = [], []
        for ch in chambers:
            areas, orth = [0, 0], [0, 0]
            for color, cells in ch['objects']:
                k = signals.index(color)
                areas[k] += len(cells)
                mask = np.zeros((ch['h'], ch['w']), bool)
                for p in cells:
                    mask[p] = True
                orth[k] += len(_components(mask))
            if areas[0] == areas[1] or orth[0] == orth[1]:
                return None
            area_labels.append(int(areas[1] > areas[0]))
            orth_labels.append(int(orth[1] > orth[0]))
        for labels in (area_labels, orth_labels):
            nwin = sum(wt for wt, label in zip(weights, labels) if label == winner)
            if 2*nwin >= sum(weights):
                return None
    margin = sum(wt*(1 if bit else -1) for wt, bit in zip(weights, winning))
    gh = int(rng.integers(3, 6))
    gw = max(7, 2*margin+3)+int(rng.integers(0, 4))
    for attempt in range(160):
        layout = _layout(rng, chambers, gh, gw)
        if layout is None:
            continue
        h, w, gauge, placed = layout
        arranged_weights = [weights[ch['index']] for ch in placed]
        nets = _networks(rng, h, w, gauge, placed, arranged_weights)
        if nets is None:
            continue
        r, c, gh, gw = gauge
        inp = np.full((h, w), bg, np.int8)
        inp[r:r+gh, c:c+gw] = field
        row = int(rng.integers(1, gh-1))
        inp[r+row, c+1], inp[r+row, c+gw-2] = signals
        for ch, net in zip(placed, nets):
            fr, fc, fh, fw = ch['r'], ch['c'], ch['h'], ch['w']
            rim = ch['body']-_rect(fr+1, fc+1, fh-2, fw-2)
            for p in rim:
                inp[p] = fence
            for color, cells in ch['objects']:
                for ir, ic in cells:
                    inp[fr+ir, fc+ic] = color
            for p in net:
                inp[p] = wire
        # Direct construction label, without calling solve or its parser.
        out = np.full((gh, gw), field, np.int8)
        anchor, sign = (1, 1) if winner == 0 else (gw-2, -1)
        for i in range(margin):
            out[row, anchor+2*i*sign] = signals[winner]
        if rng.random() < .5:
            inp, out = np.flipud(inp), np.flipud(out)
        inp, out = np.rot90(inp, rotation).copy(), np.rot90(out, rotation).copy()
        try:
            meta = _inspect(inp)
        except ValueError:
            continue
        if not np.array_equal(_render(meta), out):
            raise AssertionError("construction and input reconstruction disagree")
        if witness and not _witness(meta):
            continue
        return dict(input=inp, output=out)
    return None


def _weighted_alternative(meta, labels, weights=None):
    if any(label is None for label in labels):
        return None
    weights = meta['weights'] if weights is None else weights
    n0 = sum(weight for weight, label in zip(weights, labels) if label == 0)
    n1 = sum(weights)-n0
    if n0 == n1:
        return None
    rendered = _render(meta, abs(n0-n1), int(n1 > n0))
    # Overflow is a failed alternative, not a local-vote tie. It must never be
    # silently truncated to the gauge capacity.
    return np.empty((0, 0), np.int8) if rendered is None else rendered


def _witness(meta):
    correct = _render(meta)
    cell = _weighted_alternative(meta, meta['cell_local'])
    orth = _weighted_alternative(meta, meta['four_local'])
    chamber = _weighted_alternative(meta, meta['local'], [1]*len(meta['local']))
    return all(alt is not None and not np.array_equal(alt, correct) for alt in (cell, orth, chamber))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(name+' must be an integer in its legal range')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    mode = int(rng.integers(3))  # chamber count, surplus, gauge axis transfer
    stable = bool(rng.random() < .30)
    palette = list(map(int, rng.choice(10, 6, replace=False)))
    first_rotation = int(rng.choice((0, 2))) if mode == 2 else int(rng.integers(4))
    puzzle, seen = dict(train=[], test=[]), set()
    for split, n in (('train', num_train), ('test', num_test)):
        for i in range(n):
            witness = split == 'train' and i == 0
            second = split == 'train' and i == 1
            count = 3 if witness or second else int(rng.integers(2, 5))
            cap, lower = 6, 1
            if mode == 0:
                count = count if witness or second else (int(rng.integers(2, 4)) if split == 'train' else 4)
            elif mode == 1:
                cap = 3 if split == 'train' else 6
                if split == 'test':
                    count, lower = int(rng.integers(3, 5)), 4
            rotation = first_rotation if witness or second else int(rng.integers(4))
            if mode == 2:
                rotation = first_rotation if split == 'train' else int(rng.choice((1, 3)))
            winner = 0 if witness else 1 if second else int(rng.integers(2))
            roles = palette if stable else list(map(int, rng.choice(10, 6, replace=False)))
            for retry in range(36):
                pair = _make_pair(rng, roles, count, winner, rotation, witness, second, cap, lower)
                if pair is not None and _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('bounded scene construction exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check schema, geometry, votes, exact encoding and training evidence."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'pair counts')
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            meta = _inspect(inp)
            _require(np.array_equal(out, _render(meta)), 'incorrect surplus tally')
            _require(out.shape == meta['shape'], 'exact gauge crop')
            _require(np.sum(out != meta['field']) == meta['margin'], 'unary surplus cardinality')
            _require(set(map(int, out.flat)) == {meta['field'], meta['signals'][meta['winner']]}, 'winner only')
            if split == 'train':
                training.append(meta)
    _require(any(_witness(m) for m in training), 'missing object/port witness')
    _require(any(m['margin'] >= 2 for m in training), 'missing stride demonstration')
    _require({m['winner'] for m in training} == {0, 1}, 'both gauge ends must win in training')
    _require(any(all(m['local'][i] != m['winner'] for i, wt in enumerate(m['weights'])
                     if wt == max(m['weights'])) for m in training), 'largest-port chamber shortcut')


def self_test():
    """Bounded 42-seed audit; checks mechanisms and actual structural variation."""
    signatures, shapes, crops, palettes, glyphs, profiles, margins, axes = set(), set(), set(), set(), set(), set(), set(), set()
    counts, multi_side, forks = set(), 0, 0
    for seed in _HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                meta = _inspect(inp)
                _require(np.array_equal(solve(inp.copy()), out), 'solver mismatch')
                shapes.add(inp.shape)
                crops.add(out.shape)
                palettes.add((meta['background'], meta['field'], meta['fence'], meta['wire'], *meta['signals']))
                profiles.add(tuple(sorted(meta['weights'])))
                counts.add(len(meta['boxes']))
                margins.add(meta['margin'])
                axes.add(meta['step'])
                for chamber in meta['objects']:
                    glyphs.update(_normalize(cells) for color, cells in chamber)
                r, c, gh, gw = meta['box']
                occupied = set(_ports(meta['box']))
                by_chamber = [set() for _ in meta['boxes']]
                for route, owner in zip(meta['routes'], meta['owners']):
                    ps = set(route) & occupied
                    by_chamber[owner] |= ps
                    rs = set(route)
                    forks += sum(sum((ir+dr, ic+dc) in rs for dr, dc in _D4) >= 3 for ir, ic in rs)
                for ps in by_chamber:
                    sides = {0 if pr == r-1 else 1 if pc == c+gw else 2 if pr == r+gh else 3 for pr, pc in ps}
                    multi_side += len(sides) >= 2
    _require(len(signatures) == 42, 'seed diversity')
    _require(len(shapes) >= 25 and len(crops) >= 15, 'geometry diversity')
    _require(len(glyphs) >= 25 and len(palettes) >= 60, 'shape/palette diversity')
    _require(counts == {2, 3, 4} and axes == {(0, 1), (1, 0)}, 'chamber/axis coverage')
    _require(len(profiles) >= 10 and margins >= {1, 2, 3, 4, 5, 6}, 'weight/surplus variation')
    _require(multi_side >= 15 and forks >= 40, 'branching route coverage')
    a, b = generate_puzzle(0), generate_puzzle(0)
    _require(all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x), 'determinism')
    # Equivariance and relabeling exercise visible conventions independently of
    # the generator's chosen axis or palette policy.
    for pair in a['train']:
        for rotation in range(4):
            _require(np.array_equal(solve(np.rot90(pair['input'], rotation)), np.rot90(pair['output'], rotation)), 'rotation')
        permutation = np.random.default_rng(951).permutation(10)
        _require(np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']]), 'color relabeling')
    # The host also exercises every legal count combination. Here the bounded
    # audit keeps just the lower and upper boundaries to avoid duplicate work.
    for tr, te in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(73, tr, te)
        _require(len(p['train']) == tr and len(p['test']) == te, 'explicit counts')
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in a[s]] for s in a}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('wrong output accepted')
    return dict(seeds=42, input_shapes=len(shapes), output_shapes=len(crops),
                glyphs=len(glyphs), palettes=len(palettes), weight_profiles=len(profiles),
                margins=sorted(margins), multi_side_chambers=multi_side, fork_cells=forks)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
