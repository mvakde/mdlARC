"""Bolt collector: collection changes the scaffold that determines collection.

All conventions are visible in the queried input. No parent module is imported.
"""
from collections import deque
from itertools import combinations
import json

import numpy as np


STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 712662262, 502598366, 783822958, 519381732, 757400041,
              275194669, 78826835, 570545682, 91914754, 426010191, 742161810,
              139416528, 962286235, 899757418, 600013976, 778517877, 391804448,
              118820663, 64161909, 569577744, 488479143, 572533715, 102341646,
              691020838, 985750626, 44073421, 849131718, 665730334, 575590010,
              594664411, 205987315, 591991191, 123538806, 963983025, 606040894,
              346883424, 708527679, 275137194, 512715405, 397879702, 433860631)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'Invalid grid shape')
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             'Grids must contain integer ARC colors')
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(zip(*np.where(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue = [start]
        for r, c in queue:
            for dr, dc in STEPS:
                neighbor = (r + dr, c + dc)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        result.append(frozenset(queue))
    return result


def _decode(a):
    h, w = a.shape
    _require(h >= 7 and w >= 7, 'Scene too small for the apparatus')
    wall = int(a[0, 0])
    bg = int(a[1, 1])
    _require(bg != wall, 'Background and frame must differ')
    _require(np.all(a[-1] == wall) and np.all(a[:, 0] == wall)
             and np.all(a[:, -1] == wall), 'Fixed outer frame required')
    swatches = [int(v) for v in a[0, 1:-1] if v != wall]
    _require(len(swatches) == 1, 'One top-frame color key required')
    payload = swatches[0]
    candidates = []
    for color in np.unique(a):
        color = int(color)
        if color in (bg, wall, payload) or np.count_nonzero(a == color) != 8:
            continue
        rr, cc = np.where(a == color)
        r, c = int(rr.min()), int(cc.min())
        if int(rr.max()) == r + 2 and int(cc.max()) == c + 2:
            patch = a[r:r+3, c:c+3]
            marker = int(patch[1, 1])
            if marker not in (bg, wall, payload, color) and np.count_nonzero(a == marker) == 1:
                candidates.append((r, c, color, marker))
    _require(len(candidates) == 1, 'One unique marked 3x3 collector required')
    r, c, slot, marker = candidates[0]
    roles = (bg, wall, payload, slot, marker)
    _require(len(set(roles)) == 5, 'Semantic roles must use distinct colors')
    counter = frozenset((r+i, c+j) for i in range(3) for j in range(3))
    fixed = frozenset(zip(*np.where(a == wall))) | counter
    bolts = frozenset((rr, cc) for rr, cc in zip(*np.where(a == payload)) if rr > 0)
    rods = []
    for color in sorted(set(int(x) for x in np.unique(a)) - set(roles)):
        for cells in _components(a == color):
            rows = {rr for rr, cc in cells}
            cols = {cc for rr, cc in cells}
            _require(len(cells) >= 2 and (len(rows) == 1 or len(cols) == 1),
                     'Every bar component must be an unambiguous straight stick')
            rods.append((color, cells))
    _require(rods, 'At least one rigid bar required')
    _require(not (bolts & fixed), 'Bolts must be outside the fixed apparatus')
    return roles, (r, c), fixed, bolts, rods


def _settle(rods, fixed, bolts, shape):
    """Greatest simultaneously movable set; all new contacts are solid."""
    h, w = shape
    rods = list(rods)
    obstacles = fixed | bolts
    for _ in range(h):
        current = [cells for _, cells in rods]
        targets = [frozenset((r+1, c) for r, c in cells) for cells in current]
        moving = {i for i, cells in enumerate(targets)
                  if all(r < h for r, c in cells) and not cells & obstacles}
        while True:
            blocked = {i for i in moving if any(targets[i] & current[j]
                       for j in range(len(rods)) if j not in moving)}
            if not blocked:
                break
            moving -= blocked
        if not moving:
            return rods
        rods = [(color, targets[i] if i in moving else cells)
                for i, (color, cells) in enumerate(rods)]
    raise AssertionError('Gravity failed to terminate')


def _distances(shape, fixed, rods, counter):
    blocked = set(fixed) - set(counter)
    for _, cells in rods:
        blocked.update(cells)
    distance = np.full(shape, -1, dtype=np.int16)
    queue = deque(sorted(counter))
    for cell in counter:
        distance[cell] = 0
    h, w = shape
    while queue:
        r, c = queue.popleft()
        for dr, dc in STEPS:
            nr, nc = r + dr, c + dc
            if (0 <= nr < h and 0 <= nc < w and (nr, nc) not in blocked
                    and distance[nr, nc] < 0):
                distance[nr, nc] = distance[r, c] + 1
                queue.append((nr, nc))
    return distance


def _select(bolts, distance, capacity):
    reachable = sorted((int(distance[p]), p) for p in bolts if distance[p] >= 0)
    if len(reachable) > capacity:
        _require(reachable[capacity-1][0] < reachable[capacity][0],
                 'Capacity cannot split an equal-distance shell')
    return frozenset(p for d, p in reachable[:capacity]), len(reachable)


def _run(input_grid):
    a = _grid(input_grid)
    roles, pos, fixed, bolts, rods = _decode(a)
    bg, wall, payload, slot, marker = roles
    initial_rods = list(rods)
    initial_bolts = bolts
    r, c = pos
    counter = frozenset((r+i, c+j) for i in range(3) for j in range(3))
    _require(_settle(rods, fixed, bolts, a.shape) == rods, 'Input scaffold must be stable')
    trace = []
    total = 0
    while total < 9:
        distance = _distances(a.shape, fixed, rods, counter)
        selected, reachable = _select(bolts, distance, 9-total)
        if not selected:
            break
        before = list(rods)
        old_bolts = bolts
        bolts = bolts - selected
        rods = _settle(rods, fixed, bolts, a.shape)
        after_distance = _distances(a.shape, fixed, rods, counter)
        trace.append(dict(selected=selected, reachable=reachable,
                          distance=distance, before=before, after=list(rods),
                          unlocked=frozenset(p for p in bolts if distance[p] < 0 <= after_distance[p]),
                          sealed=frozenset(p for p in bolts if after_distance[p] < 0 <= distance[p]),
                          remaining=bolts, old_bolts=old_bolts))
        total += len(selected)
    out = a.copy()
    for _, cells in initial_rods:
        for cell in cells:
            out[cell] = bg
    for cell in initial_bolts - bolts:
        out[cell] = bg
    occupied = set(fixed) | set(bolts)
    for color, cells in rods:
        _require(not cells & occupied, 'Settled bars cannot overlap')
        occupied.update(cells)
        for cell in cells:
            out[cell] = color
    for i in range(total):
        out[r+i//3, c+i%3] = payload
    return out, dict(roles=roles, pos=pos, fixed=fixed, initial_bolts=initial_bolts,
                     bolts=bolts, initial_rods=initial_rods, rods=rods, trace=trace,
                     total=total, counter=counter)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using only the visible input, including its local color key."""
    return _run(input_grid)[0]


def pair_features(input_grid):
    a = _grid(input_grid)
    _, info = _run(a)
    trace = info['trace']
    initial = _distances(a.shape, info['fixed'], info['initial_rods'], info['counter'])
    rr, cc = info['pos']
    def manhattan(p):
        r, c = p
        return max(rr-r, r-rr-2, 0) + max(cc-c, c-cc-2, 0)
    reversals = 0
    excess = 0
    for event in trace:
        selected = event['selected']
        retained_reachable = {p for p in event['remaining'] if event['distance'][p] >= 0}
        excess += max(0, event['reachable'] - len(selected))
        reversals += bool(selected and retained_reachable and
                          max(manhattan(p) for p in selected) > min(manhattan(p) for p in retained_reachable))
    falls = [int(min(r for r, c in b) - min(r for r, c in a))
             for (_, a), (_, b) in zip(info['initial_rods'], info['rods'])]
    collected = info['initial_bolts'] - info['bolts']
    return dict(shape=a.shape, roles=info['roles'], bars=len(info['rods']),
                orientations=tuple(sorted({'H' if len({r for r, c in cells}) == 1 else 'V'
                                           for color, cells in info['initial_rods']})),
                lengths=tuple(sorted(len(cells) for color, cells in info['rods'])),
                collected=info['total'], waves=len(trace), excess=excess,
                initially_unreachable=int(sum(initial[p] < 0 for p in info['initial_bolts'])),
                newly_collected=int(sum(initial[p] < 0 for p in collected)),
                retained=len(info['bolts']), path_reversal=bool(reversals),
                falls=tuple(sorted(set(falls) - {0})),
                unlocked=sum(len(e['unlocked']) for e in trace),
                sealed=sum(len(e['sealed']) for e in trace),
                max_distance=max((int(e['distance'].max()) for e in trace), default=0),
                capacity_stops_feedback=bool(info['total'] == 9 and trace and trace[-1]['unlocked']))


def _box(mask, r0, c0, r1, c1):
    mask[r0, c0:c1+1] = True
    mask[r1, c0:c1+1] = True
    mask[r0:r1+1, c0] = True
    mask[r0:r1+1, c1] = True


def _base_scene(rng, orientation, capacity_case):
    """Build variable chambers joined by a physically supported sliding gate."""
    h = int(rng.integers(19 if capacity_case else 15, 31))
    w = int(rng.integers(22 if capacity_case else 17, 31))
    walls = np.zeros((h, w), dtype=bool)
    walls[0] = walls[-1] = True
    walls[:, 0] = walls[:, -1] = True
    rods = []  # Color-free footprints, colored after geometry is finished.
    if orientation == 'V':
        s = int(rng.integers(15 if capacity_case else 8, w-5))
        length = int(rng.integers(2, min(8, h-7)))
        drop = int(rng.integers(1, 5))
        g0 = int(rng.integers(3, h-length-drop-2))
        g1 = g0 + length - 1
        walls[1:h-1, s] = True
        walls[g0:g1+drop+1, s] = False
        # The lip closes the side of the bolt socket: bolts themselves are
        # traversable, so a bare bolt in the partition would leak a path.
        walls[g1+1:g1+drop+1, s+1] = True
        rods.append(frozenset((r, s) for r in range(g0, g1+1)))
        anchor = (g1+1, s)
        pos = (int(rng.integers(2, h-5)), int(rng.integers(2, s-4)))
        pocket = (int(rng.integers(2, h-4)), w-4)
        hidden = np.zeros((h, w), dtype=bool)
        hidden[1:h-1, s+1:w-1] = True
        if capacity_case:
            # A partial wall forces travel around its lower end. This supplies
            # a geometrically close but path-distant excess bolt.
            k = int(rng.integers(5, s-6))
            end = int(rng.integers(h-6, h-2))
            walls[1:end+1, k] = True
            pos = (int(rng.integers(2, min(7, h-5))),
                   int(rng.integers(k+1, s-3)))
    else:
        s = int(rng.integers(6, h-7))
        start = int(rng.integers(3, w-9))
        length = int(rng.integers(4, min(9, w-start-2)))
        drop = int(rng.integers(1, min(5, h-s-5)))
        support_x = int(rng.integers(start+1, start+length-1))
        walls[s, 1:w-1] = True
        walls[s, start:start+length] = False
        walls[s+drop+1, support_x] = True
        rods.append(frozenset((s, c) for c in range(start, start+length)))
        # Independently movable uprights form a load on the gate, rather than
        # bonding perpendicular contacts into a single rigid object.
        xs = [start+1, start+length-2]
        rng.shuffle(xs)
        for x in xs[:int(rng.integers(1, 3))]:
            n = int(rng.integers(2, min(5, s-1)))
            rods.append(frozenset((r, x) for r in range(s-n, s)))
        anchor = (s+1, support_x)
        pos = (int(rng.integers(s+2, h-4)), int(rng.integers(2, w-5)))
        pocket = (2, int(rng.integers(2, w-4)))
        hidden = np.zeros((h, w), dtype=bool)
        hidden[1:s, 1:w-1] = True
    # One permanently closed pocket teaches preservation of inaccessible bolts.
    pr, pc = pocket
    _box(walls, pr, pc, pr+2, pc+2)
    distraction = (pr+1, pc+1)
    r, c = pos
    walls[r:r+3, c:c+3] = False
    counter = frozenset((r+i, c+j) for i in range(3) for j in range(3))
    fixed = frozenset(zip(*np.where(walls))) | counter
    _require(not any(cells & fixed for cells in rods), 'Base bar overlaps apparatus')
    _require(anchor not in fixed, 'Anchor overlaps apparatus')
    _require(_settle([(0, cells) for cells in rods], fixed, {anchor}, (h, w))
             == [(0, cells) for cells in rods], 'Unstable base gate')
    hidden &= ~walls
    for cell in counter | set().union(*rods) | {anchor, distraction}:
        hidden[cell] = False
    return walls, pos, rods, anchor, distraction, hidden


def _cascade_scene(rng, orientation):
    """Connected chambers: each opened gate exposes the next support bolt.

    Partitions have unequal spacing; gaps, sockets, drops, and loads are
    independently positioned. All chambers participate in the same collector.
    """
    h = int(rng.integers(23, 31))
    w = int(rng.integers(23, 31))
    axis = w if orientation == 'V' else h
    count = int(rng.integers(2, 4)) if axis >= 27 else 2
    choices = [cuts for cuts in combinations(range(7, axis-6), count)
               if all(b-a >= 5 for a, b in zip(cuts, cuts[1:]))]
    if not choices:
        count = 2
        choices = [cuts for cuts in combinations(range(7, axis-6), count)
                   if all(b-a >= 5 for a, b in zip(cuts, cuts[1:]))]
    cuts = choices[int(rng.integers(len(choices)))]
    walls = np.zeros((h, w), dtype=bool)
    walls[0] = walls[-1] = True
    walls[:, 0] = walls[:, -1] = True
    rods, anchors = [], set()
    for j, s in enumerate(cuts):
        if orientation == 'V':
            n = int(rng.integers(2, 7))
            drop = int(rng.integers(1, 5))
            top = int(rng.integers(3, h-n-drop-2))
            bottom = top+n-1
            walls[1:h-1, s] = True
            walls[top:bottom+drop+1, s] = False
            walls[bottom+1:bottom+drop+1, s+1] = True
            rods.append(frozenset((r, s) for r in range(top, bottom+1)))
            anchors.add((bottom+1, s))
        else:
            start = int(rng.integers(3, w-9))
            n = int(rng.integers(4, 9))
            x = int(rng.integers(start+1, start+n-1))
            room_bottom = cuts[j+1] if j+1 < len(cuts) else h-1
            drop = int(rng.integers(1, min(5, room_bottom-s-2)))
            walls[s, 1:w-1] = True
            walls[s, start:start+n] = False
            walls[s+drop+1, x] = True
            rods.append(frozenset((s, c) for c in range(start, start+n)))
            anchors.add((s+1, x))
            # The upper load is a separate vertical stick and shares the
            # gate's movement only through contact and support.
            if rng.random() < .65:
                load_x = start+int(rng.integers(1, n-1))
                rods.append(frozenset((r, load_x) for r in range(s-2, s)))
    hidden = np.zeros((h, w), dtype=bool)
    if orientation == 'V':
        pos = (int(rng.integers(2, h-5)), int(rng.integers(2, cuts[0]-3)))
        hidden[1:h-1, cuts[-1]+1:w-1] = True
        pocket = (int(rng.integers(2, h-4)), w-4)
    else:
        pos = (int(rng.integers(cuts[-1]+2, h-4)), int(rng.integers(2, w-5)))
        hidden[1:cuts[0], 1:w-1] = True
        pocket = (2, int(rng.integers(2, w-4)))
    pr, pc = pocket
    _box(walls, pr, pc, pr+2, pc+2)
    distraction = (pr+1, pc+1)
    r, c = pos
    walls[r:r+3, c:c+3] = False
    hidden &= ~walls
    hidden[distraction] = False
    return walls, pos, rods, anchors, distraction, hidden


def _add_rods(rng, walls, pos, rods, bolts, amount):
    h, w = walls.shape
    fixed = frozenset(zip(*np.where(walls))) | frozenset(
        (pos[0]+i, pos[1]+j) for i in range(3) for j in range(3))
    result = list(rods)
    for _ in range(amount * 8):
        if len(result) >= len(rods)+amount:
            break
        horizontal = bool(rng.integers(2))
        n = int(rng.integers(2, 7))
        r = int(rng.integers(2, h-n-1))
        c = int(rng.integers(2, w-n-1))
        cells = frozenset((r+(0 if horizontal else k), c+(k if horizontal else 0)) for k in range(n))
        occupied = fixed | bolts | set().union(*result)
        if cells & occupied:
            continue
        lowered = _settle([(0, s) for s in result]+[(0, cells)], fixed, bolts, walls.shape)
        # Adding a bar must not move the already stable scaffold.
        if [s for color, s in lowered[:-1]] == result:
            result.append(lowered[-1][1])
    return result


def _color_and_render(rng, palette, walls, pos, rods, bolts):
    bg, wall, payload, slot, marker = palette[:5]
    colors = palette[5:]
    assigned = []
    for i, cells in enumerate(rods):
        neighbors = set(cells)
        for r, c in cells:
            neighbors.update((r+dr, c+dc) for dr, dc in STEPS)
        excluded = {assigned[j] for j in range(i) if neighbors & rods[j]}
        choices = [color for color in colors if color not in excluded]
        _require(choices, 'Not enough colors for unambiguous touching bars')
        assigned.append(int(rng.choice(choices)))
    a = np.full(walls.shape, bg, dtype=np.int8)
    a[walls] = wall
    a[0, int(rng.integers(1, a.shape[1]-1))] = payload
    r, c = pos
    a[r:r+3, c:c+3] = slot
    a[r+1, c+1] = marker
    for cell in bolts:
        a[cell] = payload
    for color, cells in zip(assigned, rods):
        for cell in cells:
            a[cell] = color
    return a


def _make_cascade_pair(rng, palette, orientation):
    for attempt in range(80):
        try:
            walls, pos, rods, anchors, distraction, hidden = _cascade_scene(rng, orientation)
            available = list(zip(*np.where(hidden)))
            rng.shuffle(available)
            amount = int(rng.integers(1, 9-len(anchors)))
            bolts = frozenset(anchors | set(available[:amount]) | {distraction})
            extras = int(rng.integers(0, 6)) if attempt < 50 else 0
            rods = _add_rods(rng, walls, pos, rods, bolts, extras)
            a = _color_and_render(rng, palette, walls, pos, rods, bolts)
            b, info = _run(a)
            _require(len(info['trace']) >= 3 and len(info['bolts']) == 1,
                     'Each connected gate must expose another collection wave')
            return {'input': a, 'output': b}
        except ValueError:
            continue
    raise RuntimeError('Unable to construct connected gates')


def _make_pair(rng, palette, kind, orientation, harder=False):
    if kind == 'cascade':
        return _make_cascade_pair(rng, palette, orientation)
    capacity_case = kind == 'capacity'
    for attempt in range(100):
        try:
            walls, pos, rods, anchor, distraction, hidden = _base_scene(rng, orientation, capacity_case)
            h, w = walls.shape
            counter = frozenset((pos[0]+i, pos[1]+j) for i in range(3) for j in range(3))
            fixed = frozenset(zip(*np.where(walls))) | counter
            initial_distance = _distances(walls.shape, fixed, [(0, s) for s in rods], counter)
            if capacity_case:
                rr, cc = np.indices(walls.shape)
                r, c = pos
                direct = np.maximum(np.maximum(r-rr, rr-r-2), 0)+np.maximum(np.maximum(c-cc, cc-c-2), 0)
                free = (initial_distance > 0) & ~walls
                for cell in set().union(*rods) | {distraction}:
                    free[cell] = False
                # Pick a strict distance frontier, with the anchor selected in
                # some cases and retained in others. Distances are recomputed
                # by the solver after collection, never stored as latent data.
                choices = np.flatnonzero(free)
                rng.shuffle(choices)
                bolts = None
                for f in choices:
                    threshold = int(initial_distance.flat[f])
                    near = np.flatnonzero(free & (initial_distance <= threshold))
                    far = np.flatnonzero(free & (initial_distance > threshold)
                                         & (direct < direct.flat[f]))
                    if len(near) < 9 or not len(far):
                        continue
                    chosen = [int(f)]
                    if initial_distance[anchor] <= threshold and anchor != tuple(np.unravel_index(f, walls.shape)):
                        chosen.append(int(np.ravel_multi_index(anchor, walls.shape)))
                    pool = [int(x) for x in near if int(x) not in chosen]
                    chosen.extend(int(x) for x in rng.choice(pool, 9-len(chosen), replace=False))
                    far_all = [int(x) for x in np.flatnonzero(free & (initial_distance > threshold))]
                    retained = [int(rng.choice(far))]
                    if initial_distance[anchor] > threshold:
                        retained.append(int(np.ravel_multi_index(anchor, walls.shape)))
                    retained = list(dict.fromkeys(retained))
                    pool = [x for x in far_all if x not in retained]
                    n = min(len(pool), int(rng.integers(1, 9)))
                    retained.extend(int(x) for x in rng.choice(pool, n, replace=False))
                    bolts = {tuple(np.unravel_index(x, walls.shape)) for x in chosen+retained}
                    break
                _require(bolts is not None, 'No diagnostic capacity frontier')
                # There is supply on the other side of the gate as well.
                hidden_cells = list(zip(*np.where(hidden & (initial_distance < 0))))
                _require(hidden_cells, 'Need a closed chamber')
                rng.shuffle(hidden_cells)
                bolts.update(hidden_cells[:int(rng.integers(1, min(7, len(hidden_cells))+1))])
            else:
                bolts = {anchor}
                _require(initial_distance[anchor] > 0, 'Support bolt must be reachable')
                opened_rods = _settle([(0, s) for s in rods], fixed, set(), walls.shape)
                opened_distance = _distances(walls.shape, fixed, opened_rods, counter)
                sweep = set().union(*(cells for color, cells in opened_rods))
                available = [p for p in zip(*np.where(hidden))
                             if initial_distance[p] < 0 <= opened_distance[p] and p not in sweep]
                rng.shuffle(available)
                load = int(rng.choice([2, 4])) if kind == 'feedback' else int(rng.integers(1, 9))
                _require(len(available) >= load-1, 'Not enough newly reachable supply')
                bolts.update(available[:load-1])
            bolts.add(distraction)
            extras = 0 if attempt > 65 else int(rng.integers(0, 7 if harder else 4))
            rods = _add_rods(rng, walls, pos, rods, frozenset(bolts), extras)
            a = _color_and_render(rng, palette, walls, pos, rods, bolts)
            out, info = _run(a)
            f = pair_features(a)
            if kind == 'feedback':
                _require(f['collected'] in (2, 4) and f['newly_collected'] > 0
                         and f['retained'] > 0 and f['falls'], 'Need a clear feedback demonstration')
            elif capacity_case:
                _require(f['collected'] == 9 and f['excess'] > 0 and f['path_reversal']
                         and f['retained'] > 0, 'Need diagnostic path/capacity demonstration')
            else:
                _require(f['falls'] and f['newly_collected'] > 0, 'New geometry must exercise feedback')
            return {'input': a, 'output': out}
        except ValueError:
            continue
    raise RuntimeError('Unable to build a diagnostic scene within bounded attempts')


def _count(value, low, high, name):
    _require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
             and low <= value <= high, name+' is outside its legal integer range')
    return int(value)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_))
             and seed >= 0, 'seed must be a nonnegative integer')
    seed = int(seed)
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    _require(num_train > num_test, 'num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    # Local roles are encoded, so changing palettes never changes the rule.
    fixed_palette = rng.permutation(10).tolist()
    vary_palette = bool(rng.integers(2))
    puzzle = {'train': [], 'test': []}
    specs = [('feedback', 'H'), ('capacity', 'V')]
    specs.extend((str(rng.choice(['feedback', 'general', 'capacity', 'cascade'])),
                  str(rng.choice(['H', 'V']))) for _ in range(num_train-2))
    # Capacity witnesses use the vertical partition with an independently
    # parameterized detour. Other scenes admit either gate orientation.
    specs = [(kind, 'V' if kind == 'capacity' else orientation) for kind, orientation in specs]
    seen = set()
    test_specs = [('cascade', str(rng.choice(['H', 'V'])))]
    test_specs.extend((str(rng.choice(['cascade', 'general', 'capacity'])),
                       str(rng.choice(['H', 'V']))) for _ in range(num_test-1))
    test_specs = [(kind, 'V' if kind == 'capacity' else orientation)
                  for kind, orientation in test_specs]
    for split, requests in (('train', specs), ('test', test_specs)):
        for kind, orientation in requests:
            for _ in range(12):
                palette = rng.permutation(10).tolist() if vary_palette else fixed_palette
                # 3-5 bar colors, rather than a fixed one-color scaffold.
                palette = palette[:int(rng.integers(8, 11))]
                pair = _make_pair(rng, palette, kind, orientation, harder=split == 'test')
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('Distinct-input construction exhausted')
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'Invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'Splits must be lists')
    _count(len(puzzle['train']), 2, 10, 'num_train')
    _count(len(puzzle['test']), 1, 4, 'num_test')
    _require(len(puzzle['train']) > len(puzzle['test']), 'Too few demonstrations')
    seen = set()
    features = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'Invalid pair keys')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape, 'Canvas dimensions must be preserved')
            _require(_key(a) not in seen, 'Repeated input')
            seen.add(_key(a))
            expected, info = _run(a)
            _require(np.array_equal(expected, b), 'Incorrect collection/collapse label')
            _require(np.array_equal(a[0], b[0]), 'Control swatch/frame changed')
            # Physical invariants are checked on reconstructed input pieces,
            # even if equal-colored bars touch in the resulting image.
            for (ca, sa), (cb, sb) in zip(info['initial_rods'], info['rods']):
                d = min(r for r, c in sb) - min(r for r, c in sa)
                _require(ca == cb and d >= 0 and sb == frozenset((r+d, c) for r, c in sa),
                         'Bar was deformed, recolored, or shifted sideways')
            _require(_settle(info['rods'], info['fixed'], info['bolts'], a.shape) == info['rods'],
                     'Output scaffold is not stable')
            _require(all(b[p] == a[p] for p in info['fixed'] - info['counter']), 'Fixed walls changed')
            _require(all(b[p] == info['roles'][2] for p in info['bolts']), 'Retained bolt changed')
            _require(len(info['initial_bolts'])-len(info['bolts']) == info['total'] <= 9,
                     'Collected count does not match counter load')
            features[split].append(pair_features(a))
    train = features['train']
    _require(any(f['collected'] in (2, 4) and f['newly_collected'] and f['falls']
                 and f['retained'] for f in train), 'Training must reveal feedback and partial prefix')
    _require(any(f['collected'] == 9 and f['excess'] and f['path_reversal'] for f in train),
             'Training must distinguish path distance and nine-slot capacity')
    _require({'H', 'V'} <= {o for f in train for o in f['orientations']},
             'Training must expose both bar orientations')


def _fingerprint(puzzle):
    return tuple((_key(p['input']), _key(p['output'])) for split in ('train', 'test') for p in puzzle[split])


def self_test():
    """Bounded suite: 42 seeds, physical invariants, coverage and corruption."""
    records, fingerprints, shapes = [], set(), set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = pair['input']
                untouched = a.copy()
                _require(np.array_equal(solve(a), pair['output']), 'Input-only solve mismatch')
                _require(np.array_equal(a, untouched), 'Solver mutated its input')
                f = pair_features(a)
                f['split'] = split
                records.append(f)
                shapes.add(a.shape)
    _require(len(fingerprints) == 42, 'Seeds collapsed to identical puzzles')
    _require(len(shapes) >= 30, 'Canvas geometry diversity collapsed')
    for field, minimum in (('bars', 5), ('collected', 7), ('max_distance', 12), ('lengths', 20)):
        _require(len({f[field] for f in records}) >= minimum, 'Missing variation: '+field)
    _require(len({f['roles'] for f in records}) >= 30, 'Palette role diversity collapsed')
    _require(len({d for f in records for d in f['falls']}) >= 4, 'Fall distance variation missing')
    _require(sum(f['newly_collected'] > 0 for f in records) >= 80, 'Feedback became superficial')
    _require(sum(f['capacity_stops_feedback'] for f in records) >= 3, 'Quota never truncates new access')
    _require(any(len(f['falls']) > 1 for f in records), 'No unequal falls')
    _require(sum(f['waves'] >= 3 for f in records) >= 42, 'Connected feedback chains absent')
    _require(_fingerprint(generate_puzzle(19)) == _fingerprint(generate_puzzle(19)), 'Nondeterministic seed')
    # Targeted input-only edge cases: a zero collection state, a one-bolt
    # state, and an ambiguous capacity shell. Neither edge case changes the
    # rule or requires the original generation metadata.
    edge = np.zeros((13, 18), dtype=np.int8)
    edge[0] = edge[-1] = 1
    edge[:, 0] = edge[:, -1] = 1
    edge[0, 8] = 2
    edge[4:7, 5:8] = 3
    edge[5, 6] = 4
    edge[11, 12:15] = 5  # A horizontal bar supported by the bottom frame.
    _require(np.array_equal(solve(edge), edge), 'Zero collection must be identity')
    one = edge.copy()
    one[8, 3] = 2
    expected = edge.copy()
    expected[4, 5] = 2
    _require(np.array_equal(solve(one), expected), 'One-bolt prefix regression')
    line = edge.copy()
    line[4, 8:13] = line[6, 8:13] = 1
    line[5, 12] = 1
    line[5, 8:12] = 2
    line_expected = line.copy()
    line_expected[5, 8:12] = 0
    for i in range(4):
        line_expected[4+i//3, 5+i%3] = 2
    _require(np.array_equal(solve(line), line_expected),
             'Adjacent bolts must be traversable and collected as cells')
    tied = edge.copy()
    counter = frozenset((r, c) for r in range(4, 7) for c in range(5, 8))
    for r, c in counter:
        for dr, dc in STEPS:
            p = (r+dr, c+dc)
            if p not in counter:
                tied[p] = 2
    try:
        solve(tied)
    except ValueError:
        pass
    else:
        raise AssertionError('A capacity tie must not use arbitrary queue order')
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(71, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3, 1.5), (None, 1)):
        try:
            generate_puzzle(1, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid counts accepted')
    broken = generate_puzzle(123, 2, 1)
    broken['test'][0]['output'] = broken['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError('Corrupt label accepted')
    return dict(seeds=len(HOST_SEEDS), pairs=len(records), shapes=len(shapes),
                bar_counts=sorted({f['bars'] for f in records}),
                loads=sorted({f['collected'] for f in records}),
                fall_distances=sorted({d for f in records for d in f['falls']}),
                feedback_pairs=sum(f['newly_collected'] > 0 for f in records),
                wave_counts=sorted({f['waves'] for f in records}),
                capacity_stops_feedback=sum(f['capacity_stops_feedback'] for f in records),
                semantic_verification='generator-owned checks; no independent solver')


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
