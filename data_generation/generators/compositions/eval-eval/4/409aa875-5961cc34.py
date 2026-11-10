"""Coincidence-powered relays: input-only decoding and bounded scene generation.

Only standard-library modules and NumPy are used. Control arrows are transparent
to the signal. Their simultaneous five-step endpoints power individual ports;
first-contact propagation then computes a finite directed reachability closure.
"""
from collections import Counter, deque
from itertools import combinations

import numpy as np


CARDINALS = ((-1, 0), (0, 1), (1, 0), (0, -1))
VECTORS = CARDINALS + ((-1, -1), (-1, 1), (1, -1), (1, 1))
DISTANCE = 5


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _halo(cells):
    return {(r + a, c + b) for r, c in cells
            for a in (-1, 0, 1) for b in (-1, 0, 1)}


def _components(cells, diagonal=True):
    remaining = set(cells)
    result = []
    steps = VECTORS if diagonal else CARDINALS
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        component, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    stack.append(q)
        result.append(frozenset(component))
    return result


def _arrow_cells(tip, vector):
    r, c = tip
    dr, dc = vector
    if dr == 0:
        offsets = ((0, 0), (-1, -dc), (1, -dc))
    elif dc == 0:
        offsets = ((0, 0), (-dr, -1), (-dr, 1))
    else:
        offsets = ((0, 0), (-dr, 0), (0, -dc))
    return frozenset((r + a, c + b) for a, b in offsets)


def _endpoint(arrow, distance=DISTANCE):
    r, c = arrow['tip']
    dr, dc = arrow['vector']
    return r + distance * dr, c + distance * dc


def _port_groups(cells, marking):
    r0, r1 = min(r for r, _ in cells), max(r for r, _ in cells)
    c0, c1 = min(c for _, c in cells), max(c for _, c in cells)
    assert r1 - r0 >= 2 and c1 - c0 >= 2
    sides = {}
    for r, c in sorted(marking):
        choices = [side for side, yes in enumerate(
            (r == r0, c == c1, r == r1, c == c0)) if yes]
        assert len(choices) == 1, 'Ports occupy extreme sides, excluding corners'
        sides.setdefault(choices[0], set()).add((r, c))
    assert 1 <= len(sides) <= 4
    ports = []
    for side, points in sorted(sides.items()):
        span = sorted(c if side % 2 == 0 else r for r, c in points)
        assert 1 <= len(span) <= 3
        assert span == list(range(span[0], span[0] + len(span)))
        dr, dc = CARDINALS[side]
        assert all((r - dr, c - dc) in cells - marking for r, c in points)
        ports.append({'side': side, 'cells': frozenset(points)})
    return ports


def _decode(input_grid):
    """Infer all roles and geometry from this input alone."""
    inp = _grid(input_grid)
    h, w = inp.shape
    bg = int(inp[0, 0])
    assert all(int(inp[p]) == bg for p in ((0, w - 1), (h - 1, 0), (h - 1, w - 1)))
    assert np.count_nonzero(inp == bg) > inp.size / 2
    foreground = set(map(tuple, np.argwhere(inp != bg).tolist()))
    comps = _components(foreground)
    arrows, relays, launcher = [], [], None
    port_colors, carriers, arrow_colors = set(), set(), set()
    for component in comps:
        palette = Counter(int(inp[p]) for p in component)
        if len(component) == 3 and len(palette) == 1:
            matches = [(tip, v) for tip in sorted(component) for v in VECTORS
                       if _arrow_cells(tip, v) == component]
            assert len(matches) == 1, 'Arrow geometry must have a unique tip and vector'
            tip, vector = matches[0]
            arrow = {'tip': tip, 'vector': vector, 'cells': component}
            end = _endpoint(arrow)
            assert 0 <= end[0] < h and 0 <= end[1] < w
            arrows.append(arrow)
            arrow_colors.update(palette)
            continue
        boundary = sorted(p for p in component if p[0] in (0, h - 1) or p[1] in (0, w - 1))
        if boundary:
            assert launcher is None and len(boundary) == 1
            start = boundary[0]
            side = 2 if start[0] == 0 else 0 if start[0] == h - 1 else 1 if start[1] == 0 else 3
            dr, dc = CARDINALS[side]
            line = [(start[0] + k * dr, start[1] + k * dc) for k in range(len(component))]
            assert 3 <= len(line) <= 6 and set(line) == set(component)
            signal, cap_color = int(inp[start]), int(inp[line[-1]])
            assert signal != cap_color and all(int(inp[p]) == signal for p in line[:-1])
            launcher = (frozenset(line[:-1]), line[-1], side, signal, cap_color)
            continue
        assert len(palette) == 2
        ordered = sorted(palette, key=lambda c: (palette[c], c))
        pc, carrier = ordered
        assert palette[carrier] > palette[pc]
        marking = frozenset(p for p in component if int(inp[p]) == pc)
        body = component - marking
        assert len(_components(component, False)) == 1
        assert len(_components(body, False)) == 1
        ports = _port_groups(component, marking)
        relays.append({'cells': component, 'ports': ports, 'carrier': carrier})
        port_colors.add(pc)
        carriers.add(carrier)
    assert launcher is not None and arrows and relays and len(port_colors) == 1
    tail, cap, initial, signal, cap_color = launcher
    pc = next(iter(port_colors))
    base_roles = {bg, signal, cap_color, pc}
    assert len(base_roles) == 4
    assert not carriers & base_roles and not arrow_colors & (base_roles | carriers)
    return {'input': inp, 'shape': inp.shape, 'bg': bg, 'signal': signal,
            'cap_color': cap_color, 'port_color': pc, 'tail': tail, 'cap': cap,
            'initial': initial, 'relays': relays, 'arrows': arrows}


def _power(scene, distance=DISTANCE, aggregate=False, first_obstacle=False):
    """Coincidence is counted at coordinates, never pooled across a port."""
    obstacle = set().union(*(o['cells'] for o in scene['relays']))
    counts = Counter()
    for a in scene['arrows']:
        end = _endpoint(a, distance)
        if first_obstacle:
            r, c = a['tip']
            dr, dc = a['vector']
            for k in range(1, distance + 1):
                q = (r + k * dr, c + k * dc)
                if q in obstacle:
                    end = q
                    break
        counts[end] += 1
    opened = set()
    for i, obj in enumerate(scene['relays']):
        for j, port in enumerate(obj['ports']):
            yes = (sum(counts[p] for p in port['cells']) >= 2 if aggregate
                   else any(counts[p] >= 2 for p in port['cells']))
            if yes:
                opened.add((i, j))
    return counts, opened


def _trace(scene, opened, activate_powered=False, single_lane=False):
    """Emit once per reached relay; retain original footprints as obstacles."""
    objects = scene['relays']
    owner = {p: i for i, o in enumerate(objects) for p in o['cells']}
    assert len(owner) == sum(len(o['cells']) for o in objects)
    h, w = scene['shape']
    paint = set(scene['tail']) | {scene['cap']}
    queue = deque([(scene['cap'], scene['initial'], -1, 0, -1)])
    reached, depths, edges, contacts = set(), {}, set(), {}

    def activate(i, depth):
        if i in reached:
            return
        reached.add(i)
        depths[i] = depth
        paint.update(objects[i]['cells'])
        for j, port in enumerate(objects[i]['ports']):
            if (i, j) in opened:
                for p in (sorted(port['cells'])[:1] if single_lane else sorted(port['cells'])):
                    queue.append((p, port['side'], i, depth, j))

    if activate_powered:
        for i in sorted({i for i, _ in opened}):
            activate(i, 0)
    while queue:
        (r, c), side, source, depth, port_index = queue.popleft()
        dr, dc = CARDINALS[side]
        hit, length = -1, 0
        while True:
            r, c = r + dr, c + dc
            length += 1
            if not (0 <= r < h and 0 <= c < w):
                break
            if (r, c) in owner:
                hit = owner[r, c]
                edges.add((source, hit))
                activate(hit, depth + 1)
                break
            paint.add((r, c))
        contacts.setdefault((source, port_index), []).append((hit, length))
    out = np.full((h, w), scene['bg'], dtype=np.int8)
    for p in paint:
        out[p] = scene['signal']
    return out, {'reached': reached, 'depths': depths, 'edges': edges,
                 'contacts': contacts, 'paint': paint, 'opened': opened}


def solve(input_grid: np.ndarray) -> np.ndarray:
    scene = _decode(input_grid)
    _, opened = _power(scene)
    return _trace(scene, opened)[0]


def _make_object(rng, r0, c0, h, w, max_width, profile=None):
    cells = {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}
    count = int(rng.choice((2, 3, 4), p=(.55, .35, .10)))
    sides = list(map(int, rng.choice(4, size=count, replace=False)))
    widths = [int(rng.integers(1, min(max_width, (w if s % 2 == 0 else h) - 2) + 1)) for s in sides]
    if profile == 'select':
        widths[:2] = [2, 2]
        open_indices = {0}
    elif profile == 'branch':
        widths[:2] = [1, max_width]
        open_indices = {0, 1}
    elif profile == 'deep':
        open_indices = {0, 1}
    elif profile == 'absorber':
        open_indices = set()
    else:
        open_indices = {i for i in range(count) if rng.random() < .40}
    ports = []
    for j, (side, width) in enumerate(zip(sides, widths)):
        span = (w if side % 2 == 0 else h) - 2
        if width > span:
            return None
        offset = int(rng.integers(1, span - width + 2))
        points = frozenset((r0 if side == 0 else r0 + h - 1, c0 + offset + k)
                           if side % 2 == 0 else
                           (r0 + offset + k, c0 + w - 1 if side == 1 else c0)
                           for k in range(width))
        ports.append({'side': side, 'cells': points, 'open': j in open_indices,
                      'split': profile == 'select' and j == 1})
    # Actual masks vary independently of power. Protect ports and their inward
    # neighbors while cutting corners, steps, or one connected boundary notch.
    protected = set().union(*(p['cells'] for p in ports))
    for p in ports:
        dr, dc = CARDINALS[p['side']]
        protected.update((r - dr, c - dc) for r, c in p['cells'])
    kind = int(rng.choice(5, p=(.30, .25, .15, .15, .15)))
    cut = set()
    if kind == 1:
        cut = {(r0, c0), (r0, c0 + w - 1), (r0 + h - 1, c0), (r0 + h - 1, c0 + w - 1)}
    elif kind == 2:
        rr = r0 if rng.integers(2) else r0 + h - 1
        cc = c0 if rng.integers(2) else c0 + w - 1
        cut = {(rr, cc), (rr + (1 if rr == r0 else -1), cc)}
    elif kind >= 3:
        side = int(rng.integers(4))
        depth = 1 if kind == 3 else 2
        if side % 2 == 0:
            cc = int(rng.integers(c0 + 1, c0 + w - 1))
            cut = {(r0 + k if side == 0 else r0 + h - 1 - k, cc) for k in range(depth)}
        else:
            rr = int(rng.integers(r0 + 1, r0 + h - 1))
            cut = {(rr, c0 + w - 1 - k if side == 1 else c0 + k) for k in range(depth)}
    candidate = cells - cut
    marking = set().union(*(p['cells'] for p in ports))
    if (not cut & protected and len(candidate - marking) > len(marking)
            and len(_components(candidate, False)) == 1
            and len(_components(candidate - marking, False)) == 1
            and min(r for r, c in candidate) == r0 and max(r for r, c in candidate) == r0 + h - 1
            and min(c for r, c in candidate) == c0 and max(c for r, c in candidate) == c0 + w - 1):
        cells = candidate
    if len(cells - marking) <= len(marking) or len(_components(cells - marking, False)) != 1:
        return None
    return {'cells': frozenset(cells), 'ports': ports}


def _layout(rng, profile, max_width, directions, depth_cap):
    n = int(rng.integers(4 if profile == 'deep' else 3, 7))
    low = 24 if profile == 'deep' else 20 if n <= 4 else 22
    shape = tuple(map(int, rng.integers(low, 31, size=2)))
    if profile != 'deep' and n <= 4 and rng.random() < .18:
        shape = (int(rng.integers(17, 21)), 30)
        if rng.integers(2):
            shape = shape[::-1]
    objects, blocked = [], set()
    for i in range(n):
        for _ in range(80):
            root_min = max_width + 2 if profile == 'branch' else 4 if profile == 'select' else 3
            minimum = root_min if i == 0 else 3
            h, w = map(int, rng.integers(minimum, 8, size=2))
            r0 = int(rng.integers(2, shape[0] - h - 1))
            c0 = int(rng.integers(2, shape[1] - w - 1))
            obj = _make_object(rng, r0, c0, h, w, max_width, profile if i == 0 else None)
            if obj is None or obj['cells'] & blocked:
                continue
            objects.append(obj)
            blocked.update(_halo(obj['cells']))
            break
        else:
            return None
    initial = int(rng.choice(directions))
    first = objects[0]['cells']
    rs, cs = [r for r, _ in first], [c for _, c in first]
    line = int(rng.integers(min(cs) + 1, max(cs))) if initial % 2 == 0 else int(rng.integers(min(rs) + 1, max(rs)))
    length = int(rng.integers(2, 5))
    start = ((shape[0] - 1, line) if initial == 0 else (line, 0) if initial == 1
             else (0, line) if initial == 2 else (line, shape[1] - 1))
    dr, dc = CARDINALS[initial]
    tail = frozenset((start[0] + k * dr, start[1] + k * dc) for k in range(length))
    cap = start[0] + length * dr, start[1] + length * dc
    if (tail | {cap}) & blocked:
        return None
    scene = {'shape': shape, 'relays': objects, 'tail': tail, 'cap': cap,
             'initial': initial, 'bg': 0, 'signal': 1}
    opened = {(i, j) for i, o in enumerate(objects) for j, p in enumerate(o['ports']) if p['open']}
    _, facts = _trace(scene, opened)
    if facts['depths'].get(0) != 1:
        return None
    if profile == 'select':
        # The one outgoing gate reaches an absorber, which teaches stopping.
        for i in facts['reached'] - {0}:
            for p in objects[i]['ports']:
                p['open'] = False
        opened = {(i, j) for i, o in enumerate(objects) for j, p in enumerate(o['ports']) if p['open']}
        _, facts = _trace(scene, opened)
    active = len(facts['reached'])
    depth = max(facts['depths'].values())
    if profile != 'deep' and depth > depth_cap:
        return None
    if profile in ('select', 'branch') and (active < 2 or depth > depth_cap):
        return None
    if profile == 'deep' and depth < 3:
        return None
    if profile == 'absorber' and active != 1:
        return None
    inactive = set(range(n)) - facts['reached']
    if not inactive:
        return None
    if not any(p['open'] for i in inactive for p in objects[i]['ports']):
        objects[min(inactive)]['ports'][0]['open'] = True
    return scene


def _program_arrows(rng, scene):
    """Solve endpoint equations, with bounded backtracking for separate motifs."""
    shape = scene['shape']
    occupied = set(scene['tail']) | {scene['cap']} | set().union(*(o['cells'] for o in scene['relays']))
    blocked = _halo(occupied)
    corners = {(0, 0), (0, shape[1] - 1), (shape[0] - 1, 0), (shape[0] - 1, shape[1] - 1)}

    def candidates(point):
        result = []
        for vector in VECTORS:
            tip = point[0] - DISTANCE * vector[0], point[1] - DISTANCE * vector[1]
            cells = _arrow_cells(tip, vector)
            if (cells & (blocked | corners)
                    or not all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in cells)):
                continue
            result.append({'tip': tip, 'vector': vector, 'cells': cells})
        return result

    groups = []
    for o in scene['relays']:
        for port in o['ports']:
            options = []
            if port['open']:
                for point in sorted(port['cells']):
                    for a, b in combinations(candidates(point), 2):
                        if not a['cells'] & _halo(b['cells']):
                            options.append([a, b])
            elif port['split']:
                for x, y in combinations(sorted(port['cells']), 2):
                    for a in candidates(x):
                        for b in candidates(y):
                            if not a['cells'] & _halo(b['cells']):
                                options.append([a, b])
            else:
                continue
            if not options:
                return None
            groups.append([options[int(k)] for k in rng.permutation(len(options))])
    groups.sort(key=len)
    budget = [2500]

    def search(index, arrows, forbidden):
        if index == len(groups):
            return arrows
        for option in groups[index]:
            budget[0] -= 1
            if budget[0] < 0:
                return None
            cells = set().union(*(a['cells'] for a in option))
            if cells & forbidden:
                continue
            found = search(index + 1, arrows + option, forbidden | _halo(cells))
            if found is not None:
                return found
        return None

    arrows = search(0, [], blocked)
    if arrows is None:
        return None
    blocked.update(_halo(set().union(*(a['cells'] for a in arrows))))
    # Optional single votes, third coincident votes, and non-port distractors.
    port_cells = set().union(*(p['cells'] for o in scene['relays'] for p in o['ports']))
    additions = []
    for o in scene['relays']:
        for port in o['ports']:
            if rng.random() < (.20 if port['open'] else .30):
                if port['open']:
                    hits = Counter(_endpoint(a) for a in arrows)
                    point = next(p for p in sorted(port['cells']) if hits[p] >= 2)
                else:
                    hits = Counter(_endpoint(a) for a in arrows)
                    choices = [p for p in sorted(port['cells']) if hits[p] == 0]
                    if not choices:
                        continue
                    point = choices[int(rng.integers(len(choices)))]
                additions.append(point)
    for _ in range(int(rng.integers(3))):
        for _ in range(30):
            if rng.random() < .5:
                obj = scene['relays'][int(rng.integers(len(scene['relays'])))]['cells']
                choices = sorted(obj - port_cells)
                point = choices[int(rng.integers(len(choices)))]
            else:
                point = int(rng.integers(shape[0])), int(rng.integers(shape[1]))
            if point not in port_cells:
                additions.append(point)
                break
    for point in additions:
        choices = candidates(point)
        for index in rng.permutation(len(choices)):
            a = choices[int(index)]
            if not a['cells'] & blocked:
                arrows.append(a)
                blocked.update(_halo(a['cells']))
                break
    scene['arrows'] = arrows
    intended = {(i, j) for i, o in enumerate(scene['relays']) for j, p in enumerate(o['ports']) if p['open']}
    assert _power(scene)[1] == intended
    return scene


def _render(rng, scene, palette, arrow_count, carrier_count):
    bg, signal, cap_color, port_color = map(int, palette[:4])
    arrow_pool = list(map(int, palette[4:4 + arrow_count]))
    carrier_pool = list(map(int, palette[4 + arrow_count:4 + arrow_count + carrier_count]))
    scene = dict(scene, bg=bg, signal=signal)
    inp = np.full(scene['shape'], bg, dtype=np.int8)
    for i, o in enumerate(scene['relays']):
        marking = set().union(*(p['cells'] for p in o['ports']))
        for p in o['cells']:
            inp[p] = port_color if p in marking else carrier_pool[i % len(carrier_pool)]
    for i, a in enumerate(scene['arrows']):
        for p in a['cells']:
            inp[p] = arrow_pool[i % len(arrow_pool)]
    for p in scene['tail']:
        inp[p] = signal
    inp[scene['cap']] = cap_color
    _, opened = _power(scene)
    out, _ = _trace(scene, opened)
    return {'input': inp, 'output': out}


def _facts(scene):
    counts, opened = _power(scene)
    out, facts = _trace(scene, opened)
    reached = facts['reached']
    powered_objects = {i for i, _ in opened}
    split = [(i, j) for i, o in enumerate(scene['relays']) for j, p in enumerate(o['ports'])
             if (i, j) not in opened and sum(counts[x] for x in p['cells']) >= 2]
    widths = {len(scene['relays'][i]['ports'][j]['cells']) for i, j in opened if i in reached}
    nonrectangular = set()
    for i, o in enumerate(scene['relays']):
        height = max(r for r, c in o['cells']) - min(r for r, c in o['cells']) + 1
        width = max(c for r, c in o['cells']) - min(c for r, c in o['cells']) + 1
        if len(o['cells']) < height * width:
            nonrectangular.add(i)
    facts.update(output=out, counts=counts, powered_objects=powered_objects,
                 powered_unreached=powered_objects - reached, split_votes=split,
                 active_widths=widths,
                 branches={i for i in reached if sum(a == i for a, b in opened) >= 2},
                 absorbers={i for i in reached if i not in powered_objects},
                 active_nonrectangular=nonrectangular & reached,
                 crossed_controls=any(a['cells'] & facts['paint'] for a in scene['arrows']),
                 depth=max(facts['depths'].values(), default=0))
    return facts


def _acceptable(scene, profile, essential, allowed_widths=None):
    f = _facts(scene)
    if allowed_widths is not None and not f['active_widths'] <= allowed_widths:
        return False
    if essential:
        if not f['powered_unreached']:
            return False
        all_ports = {(i, j) for i, o in enumerate(scene['relays']) for j, p in enumerate(o['ports'])}
        if np.array_equal(f['output'], _trace(scene, all_ports)[0]):
            return False
        if profile == 'select':
            if not f['absorbers'] or not any(i in f['reached'] for i, j in f['split_votes']):
                return False
            if np.array_equal(f['output'], _trace(scene, _power(scene, aggregate=True)[1])[0]):
                return False
            if np.array_equal(f['output'], _trace(scene, _power(scene, first_obstacle=True)[1])[0]):
                return False
        if profile == 'branch' and (not f['branches'] or not f['crossed_controls'] or not f['active_nonrectangular']):
            return False
    return True


# These are geometry-only fixtures invented for this composition, not parent
# pairs or solver examples. Used only if bounded randomized packing exhausts.
_FALLBACK_GEOMETRY = {"branch3":[[24,29],[[5,0],[5,1],[5,2]],[5,3],1,[[[2,17,6,5],[[2,17],[2,21],[7,17],[7,21]],[[1,5,21,2],[3,5,17,2]]],[[4,11,6,5],[[4,11],[4,15],[9,11],[9,15]],[[0,4,12,3],[1,6,15,1],[2,9,12,3]]],[[12,19,6,5],[[12,23],[13,23]],[[0,12,21,2],[1,14,23,3],[2,17,20,2],[3,13,19,3]]],[[13,2,5,7],[[17,7]],[[2,17,5,1],[3,14,2,2]]],[[15,13,3,4],[],[[0,15,15,1],[2,17,14,1]]]],[[[1,12],7],[[1,22],6],[[4,7],7],[[9,7],1],[[9,23],2],[[11,10],5],[[11,15],0],[[12,26],3],[[17,10],3],[[17,26],4],[[19,14],5],[[19,19],0],[[19,23],0],[[22,5],0],[[22,10],4],[[22,16],5],[[22,21],0],[[22,26],4]]],"select":[[28,25],[[7,0],[7,1]],[7,2],1,[[[3,5,7,6],[],[[0,3,7,2],[2,9,8,2],[3,6,5,2]]],[[10,17,3,5],[],[[0,10,20,1],[2,12,18,2]]],[[12,8,7,4],[[12,8],[12,11],[18,8],[18,11]],[[1,15,11,2],[3,14,8,1]]],[[17,18,4,4],[[17,18],[17,21],[20,18],[20,21]],[[1,18,21,2],[2,20,19,2]]],[[20,3,4,6],[],[[2,23,5,3],[3,21,3,2]]]],[[[3,12],3],[[4,3],7],[[7,19],2],[[8,13],4],[[14,13],4],[[18,16],1],[[23,21],0]]],"deep2":[[29,29],[[0,22],[1,22]],[2,22],2,[[[2,8,4,3],[[2,8],[2,10],[5,8],[5,10]],[[0,2,9,1],[1,3,10,2]]],[[7,17,5,4],[],[[1,8,20,2],[3,8,17,3]]],[[11,10,3,6],[],[[0,11,13,1],[2,13,12,2],[3,12,10,1]]],[[19,19,4,5],[[21,23],[22,23]],[[2,22,21,1],[3,20,19,1]]],[[20,4,3,4],[],[[0,20,6,1],[2,22,5,1]]],[[20,9,3,7],[],[[1,21,15,1],[3,21,9,1]]]],[[[2,4],1],[[3,15],7],[[7,9],0],[[7,14],4],[[8,25],3],[[13,7],1],[[15,24],6],[[16,20],6],[[17,16],7],[[17,5],5],[[22,26],3],[[25,24],4],[[26,4],5],[[26,9],0],[[26,20],4]]],"branch2":[[27,26],[[17,0],[17,1]],[17,2],1,[[[3,11,7,4],[[3,11],[3,14],[9,11],[9,14]],[[0,3,12,2],[3,8,11,1]]],[[9,18,5,3],[[9,18],[9,20],[13,18],[13,20]],[[0,9,19,1],[1,11,20,1],[2,13,19,1],[3,12,18,1]]],[[14,7,6,4],[[14,7],[14,10],[19,7],[19,10]],[[1,17,10,2],[3,17,7,1]]],[[15,16,5,5],[],[[1,17,20,1],[2,19,18,2]]]],[[[3,8],1],[[3,16],6],[[4,24],6],[[8,6],1],[[9,24],3],[[12,2],7],[[12,7],2],[[14,13],7],[[14,24],4],[[18,5],1],[[22,7],0],[[23,15],4],[[24,13],5]]],"deep3":[[29,27],[[13,24],[13,25],[13,26]],[13,23],3,[[[5,18,5,3],[],[[0,5,19,1],[2,9,19,1]]],[[8,8,4,4],[],[[2,11,10,1],[3,9,8,2]]],[[12,16,5,6],[[12,16],[12,21],[16,16],[16,21]],[[2,16,20,1],[3,13,16,3]]],[[13,7,6,4],[[13,7],[13,10],[18,7],[18,10]],[[0,13,8,2],[1,15,10,3],[2,18,8,2]]],[[20,9,7,5],[],[[0,20,10,3],[2,26,12,1]]],[[21,19,3,6],[[21,19],[21,24],[23,19],[23,24]],[[0,21,22,2],[1,22,24,1]]]],[[[4,24],6],[[5,3],7],[[5,8],2],[[6,15],6],[[10,5],7],[[10,15],6],[[11,25],6],[[13,3],1],[[15,5],7],[[15,13],4],[[16,25],3],[[18,3],5],[[18,16],0],[[18,21],4],[[25,5],5],[[26,17],5],[[27,24],0]]]}


def _map_scene(scene, point, vector, shape):
    def side(s):
        return CARDINALS.index(vector(CARDINALS[s]))
    return dict(scene, shape=shape, tail=frozenset(point(p) for p in scene['tail']),
                cap=point(scene['cap']), initial=side(scene['initial']),
                relays=[dict(o, cells=frozenset(point(p) for p in o['cells']),
                             ports=[dict(port, side=side(port['side']),
                                         cells=frozenset(point(p) for p in port['cells']))
                                    for port in o['ports']]) for o in scene['relays']],
                arrows=[dict(a, tip=point(a['tip']), vector=vector(a['vector']),
                             cells=frozenset(point(p) for p in a['cells'])) for a in scene['arrows']])


def _fallback_scene(rng, profile, max_width, directions):
    key = ('branch' + str(max_width) if profile == 'branch' else
           'deep' + str(max_width) if profile == 'deep' else 'select')
    shape, tail, cap, initial, entries, controls = _FALLBACK_GEOMETRY[key]
    objects = []
    for (r0, c0, h, w), cut, markings in entries:
        cells = frozenset((r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)) - frozenset(map(tuple, cut))
        ports = [{'side': side, 'cells': frozenset((r, c + k) if side % 2 == 0 else (r + k, c)
                                                  for k in range(length))}
                 for side, r, c, length in markings]
        objects.append({'cells': cells, 'ports': ports})
    arrows = [{'tip': tuple(tip), 'vector': VECTORS[index],
               'cells': _arrow_cells(tuple(tip), VECTORS[index])} for tip, index in controls]
    scene = {'shape': tuple(shape), 'tail': frozenset(map(tuple, tail)), 'cap': tuple(cap),
             'initial': initial, 'relays': objects, 'arrows': arrows, 'bg': 0, 'signal': 1}
    if profile == 'absorber':
        root = next(j for i, j in _facts(scene)['edges'] if i == -1)
        markings = set().union(*(p['cells'] for p in objects[root]['ports']))
        scene['arrows'] = [a for a in arrows if _endpoint(a) not in markings]
    if rng.integers(2):
        h, w = scene['shape']
        scene = _map_scene(scene, lambda p: (p[0], w - 1 - p[1]),
                           lambda v: (v[0], -v[1]), (h, w))
    desired = int(rng.choice(directions))
    turns = (scene['initial'] - desired) % 4
    for _ in range(turns):
        h, w = scene['shape']
        scene = _map_scene(scene, lambda p: (w - 1 - p[1], p[0]),
                           lambda v: (-v[1], v[0]), (w, h))
    return scene


def _fallback_pair(rng, profile, essential, max_width, directions, colors,
                   arrow_count, carrier_count, allowed_widths, seen):
    scene = _fallback_scene(rng, profile, max_width, directions)
    assert _acceptable(scene, profile, essential, allowed_widths)
    pair = _render(rng, scene, colors, arrow_count, carrier_count)
    if _input_key(pair['input']) not in seen:
        return pair
    # A non-port control can distinguish repeated fallback scenes without
    # changing their power or trace. Enumerate legal positions, not random luck.
    occupied = set(scene['tail']) | {scene['cap']} | set().union(
        *(o['cells'] for o in scene['relays']), *(a['cells'] for a in scene['arrows']))
    blocked = _halo(occupied)
    ports = set().union(*(p['cells'] for o in scene['relays'] for p in o['ports']))
    h, w = scene['shape']
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            for v in VECTORS:
                a = {'tip': (r, c), 'vector': v, 'cells': _arrow_cells((r, c), v)}
                end = _endpoint(a)
                if (end in ports or not (0 <= end[0] < h and 0 <= end[1] < w)
                        or a['cells'] & blocked
                        or not all(0 < rr < h - 1 and 0 < cc < w - 1 for rr, cc in a['cells'])):
                    continue
                candidate = dict(scene, arrows=scene['arrows'] + [a])
                pair = _render(rng, candidate, colors, arrow_count, carrier_count)
                if _input_key(pair['input']) not in seen:
                    return pair
    raise AssertionError('Fallback fixture has no remaining distinct control placement')


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f'{name} must be an integer from {lo} through {hi}')


def _input_key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _count(num_train, 'num_train', 2, 10)
    _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    rng = np.random.default_rng(seed)
    max_width = int(rng.choice((2, 3)))
    palette = rng.permutation(10)
    vary_palette = bool(rng.random() < .65)
    arrow_count = int(rng.choice((1, 2)))
    carrier_count = int(rng.integers(1, min(4, 6 - arrow_count) + 1))
    depth_shift = bool(rng.random() < .65)
    side = int(rng.integers(4))
    train_directions = tuple(range(4)) if depth_shift else (side, (side + 1) % 4)
    test_directions = tuple(range(4)) if depth_shift else ((side + 2) % 4, (side + 3) % 4)
    train_profiles = ['select', 'branch'] + [str(rng.choice(('absorber', 'select', 'branch', 'scatter'))) for _ in range(num_train - 2)]
    if depth_shift:
        test_profiles = ['deep'] * num_test
    else:
        test_profiles = [str(rng.choice(('absorber', 'select', 'branch', 'deep', 'scatter'))) for _ in range(num_test)]
    puzzle, seen, widths = {'train': [], 'test': []}, set(), set()
    number = 0
    for split, profiles, directions in (('train', train_profiles, train_directions),
                                         ('test', test_profiles, test_directions)):
        for index, profile in enumerate(profiles):
            essential = split == 'train' and index < 2
            # Coverage examples visibly change every semantic color role in
            # varying-palette puzzles; the solver uses geometry, not this draw.
            colors = np.roll(palette, number % 10) if vary_palette else palette
            number += 1
            pair = None
            for attempt in range(3000):
                cap = 2 if split == 'train' and depth_shift else 3
                scene = _layout(rng, profile, max_width, directions, cap)
                if scene is None:
                    continue
                scene = _program_arrows(rng, scene)
                if scene is None or not _acceptable(scene, profile, essential, widths if split == 'test' else None):
                    continue
                candidate = _render(rng, scene, colors, arrow_count, carrier_count)
                if _input_key(candidate['input']) in seen:
                    continue
                pair = candidate
                break
            if pair is None:
                pair = _fallback_pair(rng, profile, essential, max_width, directions,
                                      colors, arrow_count, carrier_count,
                                      widths if split == 'test' else None, seen)
            puzzle[split].append(pair)
            seen.add(_input_key(pair['input']))
            if split == 'train':
                widths.update(_facts(_decode(pair['input']))['active_widths'])
    order = rng.permutation(num_train)
    puzzle['train'] = [puzzle['train'][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = _input_key(inp)
            assert key not in seen
            seen.add(key)
            scene = _decode(inp)
            facts = _facts(scene)
            assert np.array_equal(out, facts['output'])
            assert set(map(int, np.unique(out))) <= {scene['bg'], scene['signal']}
            assert all(out[p] == scene['signal'] for i in facts['reached'] for p in scene['relays'][i]['cells'])
            assert all(out[p] == scene['bg'] for i in set(range(len(scene['relays']))) - facts['reached'] for p in scene['relays'][i]['cells'])
            assert all(out[p] == scene['bg'] for a in scene['arrows'] for p in a['cells'] if p not in facts['paint'])
            records[split].append((scene, facts))
    train = records['train']
    assert any(f['branches'] for _, f in train), 'Teach branching'
    assert any(f['absorbers'] for _, f in train), 'Teach absorption at a closed relay'
    assert any(f['powered_unreached'] for _, f in train), 'Power alone does not imply activation'
    assert any(f['crossed_controls'] for _, f in train), 'Control arrows are transparent'
    assert any(f['active_nonrectangular'] for _, f in train), 'Teach the actual footprint'
    assert any(any(i in f['reached'] for i, j in f['split_votes']) for _, f in train), 'Teach coordinate coincidence'
    assert any(len(f['reached']) >= 2 for _, f in train), 'Teach recursive activation'
    widths = set().union(*(f['active_widths'] for _, f in train))
    assert {1, 2} <= widths
    assert all(f['active_widths'] <= widths for _, f in records['test'])
    depth_shift = (max(f['depth'] for _, f in train) <= 2
                   and min(f['depth'] for _, f in records['test']) >= 3)
    orientation_shift = not ({s['initial'] for s, f in train}
                             & {s['initial'] for s, f in records['test']})
    assert depth_shift or orientation_shift, 'Require an observable train/test shift'
    # Counterfactuals ensure both contributions and the connector are observable.
    differences = {'without_A': False, 'without_B': False, 'aggregate': False, 'first_obstacle': False, 'single_lane': False}
    for scene, f in train:
        all_ports = {(i, j) for i, o in enumerate(scene['relays']) for j, p in enumerate(o['ports'])}
        alternatives = {'without_A': _trace(scene, all_ports)[0],
                        'without_B': _trace(scene, f['opened'], activate_powered=True)[0],
                        'aggregate': _trace(scene, _power(scene, aggregate=True)[1])[0],
                        'first_obstacle': _trace(scene, _power(scene, first_obstacle=True)[1])[0],
                        'single_lane': _trace(scene, f['opened'], single_lane=True)[0]}
        for name, prediction in alternatives.items():
            differences[name] |= not np.array_equal(prediction, f['output'])
    assert all(differences.values()), differences
    # Among the nearby fixed-distance hypotheses, training distinguishes five.
    for d in (1, 2, 3, 4, 6, 7, 8, 9):
        assert any(not np.array_equal(f['output'], _trace(s, _power(s, distance=d)[1])[0]) for s, f in train)


def _direct_tests():
    rng = np.random.default_rng(7619)

    def rect(r, c, h, w, side, origin, width, opened):
        points = frozenset((origin[0], origin[1] + k) if side % 2 == 0
                           else (origin[0] + k, origin[1]) for k in range(width))
        return {'cells': frozenset((rr, cc) for rr in range(r, r + h) for cc in range(c, c + w)),
                'ports': [{'side': side, 'cells': points, 'open': opened, 'split': False}]}

    # A->B->A must terminate and retain the original A as a stopping object.
    cycle = {'shape': (30, 30), 'bg': 0, 'signal': 1,
             'tail': frozenset(((0, 9), (1, 9))), 'cap': (2, 9), 'initial': 2,
             'relays': [rect(10, 7, 5, 5, 1, (12, 11), 2, True),
                        rect(10, 18, 5, 5, 3, (12, 18), 2, True)]}
    assert _program_arrows(rng, cycle) is not None
    pair = _render(rng, cycle, np.arange(10), 1, 2)
    expected = np.zeros((30, 30), dtype=np.int8)
    expected[:10, 9] = 1
    expected[10:15, 7:12] = 1
    expected[10:15, 18:23] = 1
    expected[12:14, 12:18] = 1
    assert np.array_equal(pair['output'], expected)
    assert np.array_equal(solve(pair['input']), expected)
    scene = _decode(pair['input'])
    assert len(_facts(scene)['reached']) == 2
    assert len([(i, j) for i, j in _facts(scene)['edges'] if i >= 0]) == 2
    scene['relays'].reverse()
    scene['arrows'].reverse()
    assert np.array_equal(_trace(scene, _power(scene)[1])[0], expected)

    # Three parallel lanes: the upper hits A, the lower hits B, the middle
    # continues to the boundary. No whole-strip stopping or implicit branching.
    partial = {'shape': (30, 30), 'bg': 0, 'signal': 1,
               'tail': frozenset(((13, 0), (13, 1))), 'cap': (13, 2), 'initial': 1,
               'relays': [rect(11, 6, 5, 5, 1, (12, 10), 3, True),
                          rect(10, 18, 3, 4, 3, (11, 18), 1, False),
                          rect(14, 24, 3, 4, 3, (15, 24), 1, False)]}
    assert _program_arrows(rng, partial) is not None
    pair = _render(rng, partial, np.arange(10), 1, 2)
    expected = np.zeros((30, 30), dtype=np.int8)
    expected[13, :6] = 1
    expected[11:16, 6:11] = 1
    expected[10:13, 18:22] = 1
    expected[14:17, 24:28] = 1
    expected[12, 11:18] = 1
    expected[13, 11:] = 1
    expected[14, 11:24] = 1
    assert np.array_equal(pair['output'], expected)
    assert np.array_equal(solve(pair['input']), expected)

    # Two votes at two distinct pixels of that same width-three port stay shut.
    partial['arrows'] = [{'tip': (7, 10), 'vector': (1, 0), 'cells': _arrow_cells((7, 10), (1, 0))},
                         {'tip': (19, 10), 'vector': (-1, 0), 'cells': _arrow_cells((19, 10), (-1, 0))}]
    pair = _render(rng, partial, np.arange(10), 1, 2)
    expected = np.zeros((30, 30), dtype=np.int8)
    expected[13, :6] = 1
    expected[11:16, 6:11] = 1
    assert np.array_equal(solve(pair['input']), expected)
    assert not _power(_decode(pair['input']))[1]

    # Force every fallback variant, including deduplication, outside the random
    # search. A minimal puzzle still has both teaching witnesses and depth shift.
    for width in (2, 3):
        puzzle = {'train': [], 'test': []}
        seen = set()
        for split, profile in (('train', 'select'), ('train', 'branch'), ('test', 'deep')):
            pair = _fallback_pair(rng, profile, split == 'train', width, (1,),
                                  np.arange(10), 2, 3, {1, 2, 3}, seen)
            seen.add(_input_key(pair['input']))
            puzzle[split].append(pair)
        validate_generated_puzzle(puzzle)
        for profile in ('select', 'branch', 'deep', 'absorber', 'scatter'):
            for _ in range(4):
                pair = _fallback_pair(rng, profile, profile in ('select', 'branch'), width,
                                      (1,), np.arange(10), 2, 3, {1, 2, 3}, seen)
                assert _input_key(pair['input']) not in seen
                seen.add(_input_key(pair['input']))
                assert np.array_equal(solve(pair['input']), pair['output'])


def self_test():
    """Bounded tests and measured structural variation on exactly 42 seeds."""
    _direct_tests()
    puzzles, records = [], []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        for split in ('train', 'test'):
            for p in puzzle[split]:
                scene = _decode(p['input'])
                f = _facts(scene)
                assert np.array_equal(solve(p['input'].tolist()), p['output'])
                assert np.array_equal(solve(p['input']), p['output'])
                assert np.array_equal(solve(np.rot90(p['input'])), np.rot90(p['output']))
                assert np.array_equal(solve(np.fliplr(p['input'])), np.fliplr(p['output']))
                records.append((split, scene, f))
    assert len({tuple(_input_key(p['input']) for s in ('train', 'test') for p in t[s]) for t in puzzles}) == 42
    assert len({s['shape'] for _, s, f in records}) >= 35
    assert {s['initial'] for _, s, f in records} == set(range(4))
    assert {a['vector'] for _, s, f in records for a in s['arrows']} == set(VECTORS)
    assert len({len(s['arrows']) for _, s, f in records}) >= 8
    assert len({len(s['relays']) for _, s, f in records}) >= 3
    assert {1, 2, 3} <= set().union(*(f['active_widths'] for _, s, f in records))
    assert any(max(f['counts'].values()) >= 3 for _, s, f in records)
    assert any(f['depth'] >= 3 for split, s, f in records if split == 'test')
    assert any(len(f['reached']) == 1 for _, s, f in records)
    assert any(len(f['branches']) >= 2 for _, s, f in records)
    assert any(len({o['carrier'] for o in s['relays']}) >= 2 for _, s, f in records)
    assert len({(s['bg'], s['signal'], s['port_color']) for _, s, f in records}) >= 25
    # A label mutation must be rejected, as must an illegal explicit count.
    bad = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('A corrupted output passed validation')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Illegal counts accepted: {a}/{b}')
