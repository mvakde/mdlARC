"""Shared counter-thrusters: align, pool, cancel, and broadcast finite jets.

Only standard-library Python and NumPy are used. The constructor paints labels
from sampled geometry; the solver independently recovers machines from pixels.
"""
from __future__ import annotations

import numpy as np


def _orient(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _point(p, shape, t):
    y, x = p
    h, w = shape
    if t >= 4:
        x = w - 1 - x
    for _ in range(t % 4):
        y, x, h, w = w - 1 - x, y, w, h
    return int(y), int(x)


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _components(points):
    remaining = set(points)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        found, stack = {start}, [start]
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    p = y + dy, x + dx
                    if p in remaining:
                        remaining.remove(p)
                        found.add(p)
                        stack.append(p)
        result.append(found)
    return result


def _bbox(points):
    ys, xs = zip(*points)
    return min(ys), min(xs), max(ys) + 1, max(xs) + 1


def _contains(a, b):
    return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]


def _box_cells(box, margin=0):
    r, c, r1, c1 = box
    return {(y, x) for y in range(r-margin, r1+margin)
            for x in range(c-margin, c1+margin)}


def _ray(shape, tip, direction, quantity):
    h, w = shape
    dy, dx = direction
    result = []
    for step in range(1, quantity + 1):
        p = tip[0] + dy * step, tip[1] + dx * step
        if not (0 <= p[0] < h and 0 <= p[1] < w):
            break
        result.append(p)
    return result


def _check_grid(grid):
    a = np.asarray(grid)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _read_horizontal(a, bg):
    """Recognize geometry, without a stored template or construction parameters."""
    h, w = a.shape
    if h % 2 or not 8 <= h <= 14 or not 3 <= w <= 15:
        return []
    middle = h // 2 - 1
    bar = a[middle:middle+2]
    bar_colors = set(map(int, np.unique(bar)))
    if bg in bar_colors or not 2 <= len(bar_colors) <= 3:
        return []
    outside = np.concatenate((a[:middle].ravel(), a[middle+2:].ravel()))
    colors = set(map(int, np.unique(outside))) - {bg}
    possibilities = []
    for frame_color in sorted(colors - bar_colors):
        payload_colors = colors - {frame_color}
        if len(payload_colors) > 1 or payload_colors & bar_colors:
            continue
        upper = set(map(tuple, np.argwhere(a[:middle] == frame_color)))
        if not upper:
            continue
        mirror = {(h-1-y, x) for y, x in upper}
        if mirror != set(map(tuple, np.argwhere(a[middle+2:] == frame_color) + (middle+2, 0))):
            continue
        channels, expected_frame, allowed = [], set(), set()
        good = True
        for comp in _components(upper):
            y0, x0, y1, x1 = _bbox(comp)
            radius = (x1-x0-1) // 2
            center = (x0+x1-1) // 2
            depth = middle-y0
            if (x1-x0 != 2*radius+1 or radius not in (1, 2)
                    or not radius+2 <= depth <= 6 or y1 != middle):
                good = False
                break
            frame, hollow = set(), set()
            for y in range(y0, middle):
                reach = min(y-y0, radius)
                frame.update(((y, center-reach), (y, center+reach)))
                hollow.update((y, x) for x in range(center-reach+1, center+reach))
            if comp != frame:
                good = False
                break
            bottom_hollow = {(h-1-y, x) for y, x in hollow}
            bottom_frame = {(h-1-y, x) for y, x in frame}
            expected_frame.update(frame | bottom_frame)
            allowed.update(frame | bottom_frame | hollow | bottom_hollow)
            channels.append(dict(center=center, radius=radius, depth=depth,
                hollows=(tuple(sorted(hollow)), tuple(sorted(bottom_hollow))),
                tips=((y0, center), (h-1-y0, center)), span=(x0, x1)))
        channels.sort(key=lambda c: c['center'])
        if not good or not 1 <= len(channels) <= 3:
            continue
        if channels[0]['span'][0] != 0 or channels[-1]['span'][1] != w:
            continue
        if any(not 1 <= b['span'][0]-c['span'][1] <= 3
               for c, b in zip(channels, channels[1:])):
            continue
        if set(map(tuple, np.argwhere(a == frame_color))) != expected_frame:
            continue
        if any(a[y, x] != bg for y in range(h) for x in range(w)
               if y not in (middle, middle+1) and (y, x) not in allowed):
            continue
        skeleton = a.copy()
        for ch in channels:
            for hollow in ch['hollows']:
                if any(int(a[p]) not in ({bg} | payload_colors) for p in hollow):
                    good = False
                for p in hollow:
                    skeleton[p] = bg
        if good:
            possibilities.append(dict(skeleton=skeleton, channels=channels,
                frame_color=frame_color, bar_colors=bar_colors,
                payload_colors=payload_colors))
    return possibilities


def _read_machine(patch, origin, bg):
    candidates = []
    labels = np.arange(patch.size).reshape(patch.shape)
    for t in range(8):
        a = _orient(patch, t)
        for candidate in _read_horizontal(a, bg):
            candidate.update(arr=a, labels=_orient(labels, t), transform=t,
                             origin=origin, original_shape=patch.shape)
            candidates.append(candidate)
    assert candidates, 'unrecognized counter-thruster'
    # Every acceptable interpretation must describe the same physical shell.
    orbits = {min(_key(_orient(c['skeleton'], t)) for t in range(8)) for c in candidates}
    assert len(orbits) == 1, 'ambiguous shell interpretation'
    result = min(candidates, key=lambda c: _key(c['skeleton']))
    assert len({_key(_orient(result['skeleton'], t)) for t in range(8)}) == 8, 'ambiguous fingerprint'
    return result


def _world(machine, p):
    label = int(machine['labels'][p])
    y, x = divmod(label, machine['original_shape'][1])
    return y + machine['origin'][0], x + machine['origin'][1]


def _nozzle(machine, channel, side):
    tip_local = machine['channels'][channel]['tips'][side]
    inward = (tip_local[0] + (1 if side == 0 else -1), tip_local[1])
    tip, inner = _world(machine, tip_local), _world(machine, inward)
    return tip, (tip[0]-inner[0], tip[1]-inner[1])


def _read_scene(input_grid):
    a = _check_grid(input_grid)
    values, counts = np.unique(a, return_counts=True)
    bg = int(values[np.argmax(counts)])
    assert counts.max() > a.size // 2, 'background must occupy over half the input'
    components = _components(map(tuple, np.argwhere(a != bg)))
    boxes = [_bbox(c) for c in components]
    outer = [b for i, b in enumerate(boxes)
             if not any(i != j and other != b and _contains(other, b)
                        for j, other in enumerate(boxes))]
    assert 2 <= len(outer) <= 4, 'a fleet has two to four machines'
    machines, occupied = [], set()
    for box in sorted(outer):
        cells = _box_cells(box)
        assert occupied.isdisjoint(cells), 'overlapping machines'
        occupied.update(cells)
        y, x, y1, x1 = box
        machine = _read_machine(a[y:y1, x:x1], (y, x), bg)
        machine['box'] = cells
        machines.append(machine)
    assert len({_key(m['skeleton']) for m in machines}) == 1, 'shells do not match'
    payload_colors = set().union(*(m['payload_colors'] for m in machines))
    assert len(payload_colors) == 1, 'one visibly supplied fuel color per fleet'
    fuel = payload_colors.pop()
    for m in machines:
        quantities = []
        component_quantities = []
        payload_cells = set()
        for ch in m['channels']:
            quantities.append([sum(m['arr'][p] == fuel for p in hollow)
                               for hollow in ch['hollows']])
            component_quantities.append([len(_components(p for p in hollow if m['arr'][p] == fuel))
                                         for hollow in ch['hollows']])
            payload_cells.update(_world(m, p) for hollow in ch['hollows']
                                 for p in hollow if m['arr'][p] == fuel)
        m['counts'] = np.array(quantities, dtype=int)
        m['component_counts'] = np.array(component_quantities, dtype=int)
        m['payload'] = payload_cells
    totals = sum((m['counts'] for m in machines), np.zeros_like(machines[0]['counts']))
    net = totals[:, 0] - totals[:, 1]
    assert np.any(net), 'at least one channel must have a surplus'
    return dict(input=a, background=bg, fuel=fuel, machines=machines,
                totals=totals, net=net)


def _paint(scene, nets=None, strict=True):
    a = scene['input']
    out = a.copy()
    machines = scene['machines']
    nets = [scene['net']] * len(machines) if nets is None else nets
    painted, jets = set(), []
    for m in machines:
        for p in m['payload']:
            out[p] = scene['background']
    for index, (m, net) in enumerate(zip(machines, nets)):
        for channel, value in enumerate(net):
            if not value:
                continue
            q, side = abs(int(value)), int(value < 0)
            tip, direction = _nozzle(m, channel, side)
            ray = _ray(a.shape, tip, direction, q)
            if strict:
                assert painted.isdisjoint(ray), 'jets intersect'
                assert all(a[p] == scene['background'] for p in ray), 'jet strikes an object or fuel'
                assert all(other['box'].isdisjoint(ray) for j, other in enumerate(machines) if j != index)
            for p in ray:
                out[p] = scene['fuel']
            painted.update(ray)
            full = _ray(a.shape, tip, direction, 31)
            jets.append(dict(machine=index, channel=channel, quantity=q,
                             visible=len(ray), clearance=len(full), tip=tip,
                             direction=direction, cells=tuple(ray)))
    return out, jets


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete answer from the input pixels alone."""
    return _paint(_read_scene(input_grid))[0]


def _shell(radii, depths, gaps, barcode, bg, frame):
    d = max(depths)
    length = sum(2*r+1 for r in radii) + sum(gaps)
    a = np.full((2*d+2, length), bg, dtype=np.int8)
    a[d:d+2] = barcode
    channels, cursor = [], 0
    for index, (radius, depth) in enumerate(zip(radii, depths)):
        center, top = cursor+radius, d-depth
        cavity = set()
        for y in range(top, d):
            reach = min(y-top, radius)
            for x in {center-reach, center+reach}:
                a[y, x] = a[a.shape[0]-1-y, x] = frame
            cavity.update((y, x) for x in range(center-reach+1, center+reach))
        channels.append(dict(hollows=(tuple(sorted(cavity)),
            tuple(sorted((a.shape[0]-1-y, x) for y, x in cavity))),
            tips=((top, center), (a.shape[0]-1-top, center))))
        cursor += 2*radius+1 + (gaps[index] if index < len(gaps) else 0)
    return a, channels


def _barcode(rng, length, colors):
    for _ in range(64):
        a = rng.choice(colors, size=(2, length)).astype(np.int8)
        style = int(rng.integers(4))
        if style == 1:
            a[0] = colors[0]
            a[1] = colors[1]
            a[1, :int(rng.integers(1, length))] = colors[0]
        elif style == 2:
            a[0] = colors[0]
            a[1] = colors[1]
            a[:, 0] = colors[-1]
            a[0, -1] = colors[1]
        elif style == 3:
            a[:] = colors[0]
            a[1, 1:] = colors[1]
        for color in colors:
            if color not in a:
                a[1, int(rng.integers(length))] = color
        if set(map(int, a.flat)) == set(colors) and len({_key(_orient(a, t)) for t in range(8)}) == 8:
            return a
    a = np.full((2, length), colors[0], dtype=np.int8)
    a[1, 1:] = colors[1]
    for i, color in enumerate(colors[2:], 1):
        a[1, i] = color
    return a


def _allocate(rng, total, capacity, indices, n):
    amounts = np.zeros(n, dtype=int)
    # Start by distributing, so pooling need not reduce to one loaded copy.
    order = [int(i) for i in rng.permutation(indices)]
    for i in order[:min(total, len(order))]:
        amounts[i] += 1
    remaining = total - int(amounts.sum())
    for _ in range(remaining):
        choices = [i for i in indices if amounts[i] < capacity]
        amounts[int(rng.choice(choices))] += 1
    return amounts


def _fuel_counts(rng, capacities, n, demo, blank, fallback=False):
    k = len(capacities)
    amounts = np.zeros((n, k, 2), dtype=int)
    if demo == 'pool':
        amounts[:2, 0] = ((2, 0), (1, 1))
        amounts[:2, 1] = ((1, 2), (0, 2))
    elif demo == 'balance':
        amounts[:, 0] = ((2, 1), (1, 2))
        amounts[:, 1] = ((3, 1), (2, 0))
    else:
        active = list(range(n-1 if blank else n))
        for j, capacity in enumerate(capacities):
            limit = capacity * len(active)
            # A broad three-channel bank needs modest bidirectional jets when
            # several copies share a <=30-cell canvas. Narrow banks can fire
            # much farther. This is a packing preference, not a solver rule.
            compact_fleet = k == 3 and n >= 3
            max_jet = 2 if fallback or compact_fleet else 7
            if j == 0:
                q = int(rng.integers(2, min(max_jet, limit)+1))
            else:
                high = 1 if compact_fleet and not fallback else max_jet
                q = 0 if rng.random() < .23 else int(rng.integers(1, min(high, limit)+1))
            common_low = min(2, limit-q) if blank else 0
            common = int(rng.integers(common_low, min(5, limit-q)+1))
            totals = [common+q, common]
            if rng.integers(2):
                totals.reverse()
            for side in (0, 1):
                amounts[:, j, side] = _allocate(rng, totals[side], capacity, active, n)
        if not np.any(amounts.sum(axis=0)[:, 0] - amounts.sum(axis=0)[:, 1]):
            # Remove one token from one opposed reservoir, preserving capacity.
            if amounts[:, 0, 0].sum():
                i = int(np.flatnonzero(amounts[:, 0, 0])[0])
                amounts[i, 0, 0] -= 1
            else:
                amounts[0, 0, 0] = 1
    # Independent side exchanges vary the signs of the channel balances.
    for j in range(k):
        if rng.integers(2):
            amounts[:, j] = amounts[:, j, ::-1]
    assert all(np.all(amounts[:, j] <= cap) for j, cap in enumerate(capacities))
    return amounts


def _pick_fuel(rng, hollow, q, style):
    cells = list(hollow)
    if not q:
        return set()
    if style == 'random':
        return {cells[int(i)] for i in rng.choice(len(cells), q, replace=False)}
    selected = {cells[int(rng.integers(len(cells)))]}
    while len(selected) < q:
        rest = sorted(set(cells)-selected)
        distances = [min(abs(p[0]-b[0])+abs(p[1]-b[1]) for b in selected) for p in rest]
        best = min(distances) if style == 'compact' else max(distances)
        choices = [p for p, distance in zip(rest, distances) if distance == best]
        selected.add(choices[int(rng.integers(len(choices)))])
    return selected


def _unit(rng, shell, channels, amounts, net, fuel, t, style):
    a = shell.copy()
    for j, ch in enumerate(channels):
        for side, hollow in enumerate(ch['hollows']):
            for p in _pick_fuel(rng, hollow, int(amounts[j, side]), style):
                a[p] = fuel
    jets = []
    for j, value in enumerate(net):
        if value:
            side = int(value < 0)
            p = channels[j]['tips'][side]
            tip = _point(p, shell.shape, t)
            neighbor = _point((p[0]+(-1 if side == 0 else 1), p[1]), shell.shape, t)
            jets.append((tip, (neighbor[0]-tip[0], neighbor[1]-tip[1]), abs(int(value))))
    return dict(input=_orient(a, t), output=_orient(shell, t), jets=jets)


def _pack(rng, units, shape, all_free):
    placed, boxes, guards, paths = [], set(), set(), set()
    for index, unit in enumerate(units):
        h, w = unit['input'].shape
        positions = [(y, x) for y in range(shape[0]-h+1) for x in range(shape[1]-w+1)]
        free = all_free or index == 0 or np.array_equal(unit['input'], unit['output']) or rng.random() < .5
        for raw in rng.permutation(len(positions)):
            y, x = positions[int(raw)]
            box = _box_cells((y, x, y+h, x+w))
            if not box.isdisjoint(guards | paths):
                continue
            ray_guards, valid = set(), True
            for tip, direction, q in unit['jets']:
                tip = tip[0]+y, tip[1]+x
                ray = _ray(shape, tip, direction, q+1)
                if free and len(ray) != q+1:
                    valid = False
                    break
                ray_guards.update(ray)
            if not valid or not ray_guards.isdisjoint(boxes | paths):
                continue
            placed.append((unit, y, x))
            boxes.update(box)
            guards.update(_box_cells((y, x, y+h, x+w), 1))
            paths.update(ray_guards)
            break
        else:
            return None
    return placed


def _private_pack(rng, units):
    """Bounded, guaranteed layout for the deliberately small fallback geometry."""
    quadrants = [(0, 0), (0, 15), (15, 0), (15, 15)]
    rng.shuffle(quadrants)
    placed = []
    for unit, (y0, x0) in zip(units, quadrants):
        h, w = unit['input'].shape
        extent = _box_cells((0, 0, h, w))
        for tip, direction, q in unit['jets']:
            for step in range(1, q+2):
                extent.add((tip[0]+step*direction[0], tip[1]+step*direction[1]))
        lo_y, lo_x, hi_y, hi_x = _bbox(extent)
        assert hi_y-lo_y <= 15 and hi_x-lo_x <= 15
        # Centering keeps the input shells separated at region boundaries;
        # only a jet/its empty endpoint guard may reach a region's edge.
        y = y0-lo_y + (15-(hi_y-lo_y))//2
        x = x0-lo_x + (15-(hi_x-lo_x))//2
        placed.append((unit, y, x))
    return placed


def _construct_pair(rng, demo, n, k, blank, previous_bg, fallback=False):
    palette = list(map(int, rng.permutation(10)))
    if palette[0] == previous_bg:
        palette[0], palette[-1] = palette[-1], palette[0]
    bg, frame, fuel = palette[:3]
    bar_colors = palette[3:3+int(rng.integers(2, 4))]
    if fallback:
        radii = [1]*k
        depths = [4 if demo == 'balance' else 3]*k
        gaps = [1]*(k-1)
    else:
        radii = [int(rng.choice([1, 1, 2])) for _ in range(k)]
        gaps = list(map(int, rng.integers(1, 4, size=k-1)))
        if sum(2*r+1 for r in radii)+sum(gaps) > (13 if k == 3 else 15):
            return None
        depths = [int(rng.integers(max(r+2, 4 if demo == 'balance' else 3),
                                   5 if n == 4 or (n >= 3 and k == 3) else 7)) for r in radii]
    length = sum(2*r+1 for r in radii)+sum(gaps)
    barcode = _barcode(rng, length, bar_colors)
    shell, channels = _shell(radii, depths, gaps, barcode, bg, frame)
    amounts = _fuel_counts(rng, [len(ch['hollows'][0]) for ch in channels], n, demo, blank, fallback)
    totals = amounts.sum(axis=0)
    net = totals[:, 0]-totals[:, 1]
    if demo == 'pool':
        transforms = [0, 4, 6]
    elif demo == 'balance':
        transforms = [0, 1]
    else:
        mode = int(rng.integers(4))
        if mode == 0:
            transforms = [int(rng.integers(8))]*n
        elif mode == 1:
            base = int(rng.integers(4))
            transforms = [base if i % 2 == 0 else (base+2) % 4 for i in range(n)]
        else:
            transforms = list(map(int, rng.integers(8, size=n)))
    units = [_unit(rng, shell, channels, amounts[i], net, fuel, transforms[i],
                   'compact' if demo and i == 0 else str(rng.choice(['compact', 'scatter', 'random'])))
             for i in range(n)]
    if fallback:
        shape = (30, 30)
        placed = _private_pack(rng, units)
    else:
        low = 17 if n == 2 else 23 if n == 3 else 27
        shape = tuple(map(int, rng.integers(low, 31, size=2)))
        placed = _pack(rng, units, shape, all_free=bool(demo))
        if placed is None:
            return None
    inp = np.full(shape, bg, dtype=np.int8)
    out = inp.copy()
    for unit, y, x in placed:
        h, w = unit['input'].shape
        inp[y:y+h, x:x+w] = unit['input']
        out[y:y+h, x:x+w] = unit['output']
    for unit, y, x in placed:
        for tip, direction, q in unit['jets']:
            for p in _ray(shape, (tip[0]+y, tip[1]+x), direction, q):
                out[p] = fuel
    if np.count_nonzero(inp == bg) <= inp.size // 2:
        return None
    # The whole-grid transform does not change the rule or depend on any label.
    global_transform = int(rng.integers(8))
    inp, out = _orient(inp, global_transform), _orient(out, global_transform)
    predicted, _ = _paint(_read_scene(inp))
    assert np.array_equal(predicted, out), 'construction disagrees with pixel reconstruction'
    return {'input': inp, 'output': out}


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not lo <= value <= hi:
        raise ValueError(f'{name} must be in {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(['fleet', 'channels', 'redistribute']))
    puzzle, seen, previous_bg = {'train': [], 'test': []}, set(), None
    for split, number in (('train', num_train), ('test', num_test)):
        for index in range(number):
            demo = ('pool', 'balance')[index] if split == 'train' and index < 2 else None
            if demo:
                n, k, blank = (3, 2, True) if demo == 'pool' else (2, 2, False)
            else:
                n = int(rng.integers(2, 4 if split == 'train' else 5))
                k = int(rng.integers(1, 3 if shift == 'channels' and split == 'train' else 4))
                blank = bool(rng.random() < .4)
                if split == 'test' and index == 0:
                    blank = True
                    if shift == 'fleet':
                        n = 4
                    elif shift == 'channels':
                        k = 3
                    else:
                        n, blank = int(rng.integers(3, 5)), True
            pair = None
            for attempt in range(21):
                pair = _construct_pair(rng, demo, n, k, blank, previous_bg, fallback=attempt == 20)
                if pair is not None and _key(pair['input']) not in seen:
                    break
            assert pair is not None and _key(pair['input']) not in seen, 'distinct pair construction failed'
            seen.add(_key(pair['input']))
            previous_bg = int(np.bincount(pair['input'].ravel(), minlength=10).argmax())
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _physical_pool_nets(scene):
    """Counterfactual: pool by world-up/left and spatial rank, ignoring barcode."""
    native_counts, mappings = [], []
    for m in scene['machines']:
        _, direction = _nozzle(m, 0, 0)
        vertical = direction[0] != 0
        order = sorted(range(len(m['channels'])),
                       key=lambda j: _nozzle(m, j, 0)[0][1 if vertical else 0])
        swapped = direction[0 if vertical else 1] > 0
        c = m['counts'][order]
        native_counts.append(c[:, ::-1] if swapped else c)
        mappings.append((order, swapped))
    total = sum(native_counts, np.zeros_like(native_counts[0]))
    native_net = total[:, 0]-total[:, 1]
    results = []
    for order, swapped in mappings:
        net = np.zeros_like(native_net)
        for rank, j in enumerate(order):
            net[j] = native_net[rank] * (-1 if swapped else 1)
        results.append(net)
    return results


def _evidence(scenes):
    full_counts, flags = set(), set()
    for scene in scenes:
        expected, jets = _paint(scene)
        full_counts.update(j['quantity'] for j in jets if j['clearance'] > j['quantity'])
        if any(a == b and a > 0 for a, b in scene['totals']):
            flags.add('positive_tie')
        if any(min(a, b) > 0 and a != b for a, b in scene['totals']):
            flags.add('opposition')
        if any(not m['counts'].sum() for m in scene['machines']):
            flags.add('empty_copy')
        if len({m['transform'] >= 4 for m in scene['machines']}) > 1:
            flags.add('reflection')
        if len({_nozzle(m, 0, 0)[1][0] == 0 for m in scene['machines']}) > 1:
            flags.add('quarter_turn')
        local = [m['counts'][:, 0]-m['counts'][:, 1] for m in scene['machines']]
        component_total = sum((m['component_counts'] for m in scene['machines']),
                              np.zeros_like(scene['totals']))
        component_net = component_total[:, 0]-component_total[:, 1]
        alternatives = dict(local=local, fixed_one=[np.sign(scene['net'])]*len(local),
            components=[component_net]*len(local), physical=_physical_pool_nets(scene))
        for name, nets in alternatives.items():
            if not np.array_equal(expected, _paint(scene, nets, strict=False)[0]):
                flags.add('reject_'+name)
    return full_counts, flags


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    n, m = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= n <= 10 and 1 <= m <= 4 and n > m
    seen, scenes = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _check_grid(pair['input']), _check_grid(pair['output'])
            assert inp.shape == out.shape
            assert _key(inp) not in seen, 'repeated input'
            seen.add(_key(inp))
            scene = _read_scene(inp)
            expected, jets = _paint(scene)
            assert np.array_equal(expected, out), 'incorrect cancellation, ray, or preserved cell'
            assert not np.array_equal(inp, out)
            # Fuel is consumed, while the full colored shell is preserved.
            assert all(out[p] == scene['background'] for machine in scene['machines'] for p in machine['payload'])
            assert any(j['clearance'] > j['quantity'] for j in jets), 'all quantities are hidden by clipping'
            scenes[split].append(scene)
    counts, flags = _evidence(scenes['train'])
    assert len(counts) >= 3, 'training must expose multiple finite quantities'
    required = {'positive_tie', 'opposition', 'empty_copy', 'reflection', 'quarter_turn',
                'reject_local', 'reject_fixed_one', 'reject_components', 'reject_physical'}
    assert required <= flags, f'insufficient training evidence: {required-flags}'
    # A count or correspondence shortcut also fails on the requested test split.
    _, test_flags = _evidence(scenes['test'])
    assert {'reject_local', 'reject_fixed_one'} <= test_flags
    return None


def self_test():
    """Bounded variation audit across 42 seeds, plus metamorphic/regression checks."""
    fingerprints, shapes, profiles, barcodes = set(), set(), set(), set()
    backgrounds, channel_counts, fleet_counts, magnitudes = set(), set(), set(), set()
    directions, frame_widths, frame_depths, palettes = set(), set(), set(), set()
    modes, combined, empty, ties = set(), 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                scene = _read_scene(pair['input'])
                _, jets = _paint(scene)
                machines, net = scene['machines'], scene['net']
                first = machines[0]
                shapes.add(pair['input'].shape)
                backgrounds.add(scene['background'])
                channel_counts.add(len(net))
                fleet_counts.add(len(machines))
                palettes.add(len(np.unique(pair['input'])))
                magnitudes.update(map(abs, map(int, net)))
                profiles.add(tuple((c['radius'], c['depth'], c['center']) for c in first['channels']))
                d = first['skeleton'].shape[0]//2-1
                barcodes.add(_key(first['skeleton'][d:d+2]))
                frame_widths.update(2*c['radius']+1 for c in first['channels'])
                frame_depths.update(c['depth'] for c in first['channels'])
                empty += any(not machine['counts'].sum() for machine in machines)
                ties += any(a == b and a > 0 for a, b in scene['totals'])
                combined += len(machines) >= 3 and len(net) >= 2 and any(net > 0) and any(net < 0)
                for jet in jets:
                    directions.add(jet['direction'])
                    q, room = jet['quantity'], jet['clearance']
                    modes.add('free' if q < room else 'exact' if q == room else 'clipped')
                # A fresh color permutation and grid isometry must commute with solving.
                if split == 'test':
                    permutation = np.roll(np.arange(10, dtype=np.int8), 1+seed % 9)
                    t = seed % 8
                    assert np.array_equal(solve(_orient(permutation[pair['input']], t)),
                                          _orient(permutation[pair['output']], t))
    assert len(fingerprints) == 42
    assert len(shapes) >= 35 and len(profiles) >= 35 and len(barcodes) >= 150
    assert backgrounds == set(range(10)) and channel_counts == {1, 2, 3} and fleet_counts == {2, 3, 4}
    assert frame_widths == {3, 5} and len(frame_depths) >= 4 and palettes == {5, 6}
    assert {0, 1, 2, 3, 4} <= magnitudes and max(magnitudes) >= 6
    assert directions == {(-1, 0), (1, 0), (0, -1), (0, 1)}
    assert {'free', 'clipped'} <= modes and min(empty, ties, combined) >= 20
    original = generate_puzzle(0, 2, 1)
    repeated = generate_puzzle(0, 2, 1)
    assert all(np.array_equal(a[r], b[r]) for s in original for a, b in zip(original[s], repeated[s])
               for r in ('input', 'output'))
    # Move a token to a corresponding remote chamber: only the inventory's
    # distribution changes, so all cleared pixels and all jets stay identical.
    pair = original['train'][0]
    scene = _read_scene(pair['input'])
    machines = scene['machines']
    moved = False
    for j in range(len(scene['net'])):
        for side in (0, 1):
            for source in machines:
                tokens = [p for p in source['channels'][j]['hollows'][side]
                          if source['arr'][p] == scene['fuel']]
                if not tokens:
                    continue
                for recipient in machines:
                    spaces = [p for p in recipient['channels'][j]['hollows'][side]
                              if recipient['arr'][p] == scene['background']]
                    if recipient is source or not spaces:
                        continue
                    changed = pair['input'].copy()
                    changed[_world(source, tokens[0])] = scene['background']
                    changed[_world(recipient, spaces[0])] = scene['fuel']
                    assert np.array_equal(solve(changed), pair['output'])
                    moved = True
                    break
                if moved:
                    break
            if moved:
                break
        if moved:
            break
    assert moved
    # Equal new opposition cancels; one extra winning token lengthens every
    # corresponding jet by exactly one, rather than changing only its owner.
    blank = next(m for m in machines if not m['counts'].sum())
    changed = pair['input'].copy()
    for side in (0, 1):
        changed[_world(blank, blank['channels'][0]['hollows'][side][0])] = scene['fuel']
    assert np.array_equal(solve(changed), pair['output'])
    side = int(scene['net'][0] < 0)
    changed = pair['input'].copy()
    changed[_world(blank, blank['channels'][0]['hollows'][side][0])] = scene['fuel']
    extended = pair['output'].copy()
    for machine in machines:
        tip, direction = _nozzle(machine, 0, side)
        ray = _ray(extended.shape, tip, direction, abs(int(scene['net'][0]))+1)
        assert len(ray) == abs(int(scene['net'][0]))+1
        extended[ray[-1]] = scene['fuel']
    assert np.array_equal(solve(changed), extended)
    for counts in ((10, 1), (10, 4), (5, 4)):
        p = generate_puzzle(732, *counts)
        assert (len(p['train']), len(p['test'])) == counts
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.0, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = {s: [{r: a.copy() for r, a in pair.items()} for pair in original[s]] for s in original}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('output corruption accepted')
    # Exercise every guaranteed fallback kind, including its evidence witnesses.
    rng = np.random.default_rng(221)
    a = _construct_pair(rng, 'pool', 3, 2, True, None, True)
    b = _construct_pair(rng, 'balance', 2, 2, False, None, True)
    c = _construct_pair(rng, None, 4, 3, True, None, True)
    validate_generated_puzzle({'train': [a, b], 'test': [c]})
    return dict(seeds=42, puzzles=len(fingerprints), shapes=len(shapes), profiles=len(profiles),
                fingerprints=len(barcodes), channel_counts=sorted(channel_counts),
                fleet_counts=sorted(fleet_counts), jet_magnitudes=sorted(magnitudes),
                clearance_modes=sorted(modes), empty_copy_pairs=empty, tied_pairs=ties,
                combined_pairs=combined)
