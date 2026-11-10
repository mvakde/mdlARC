"""Count-limited signals from hollow launchers vote at reached collector ports.

Only Python's standard library and NumPy are used. Construction paints labels
from sampled records; solve reconstructs roles, frames, graphs and distances.
"""
from collections import deque
import heapq
import itertools

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _components(cells, diagonal=False):
    remaining = set(cells)
    steps = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                  if (r or c) and (diagonal or abs(r) + abs(c) == 1))
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        component, stack = {p}, [p]
        while stack:
            for d in steps:
                q = _add(stack[-1], d)
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    stack.append(q)
                    break
            else:
                stack.pop()
        result.append(component)
    return result


def _distances(cells, root):
    distance = {root: 1}  # The first cell immediately beyond the tip costs one.
    queue = deque([root])
    while queue:
        p = queue.popleft()
        for d in DIRS:
            q = _add(p, d)
            if q in cells and q not in distance:
                distance[q] = distance[p] + 1
                queue.append(q)
    return distance


def _box(r, c, h, w):
    return {(y, x) for y in range(r, r + h) for x in range(c, c + w)}


def _around(cells, diagonal=False):
    steps = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                  if (r or c) and (diagonal or abs(r) + abs(c) == 1))
    return {_add(p, d) for p in cells for d in steps}


def _ports(hub):
    r, c = hub
    return {(r - 1, c + j) for j in range(3)} | {
        (r + 3, c + j) for j in range(3)} | {
        (r + j, c - 1) for j in range(3)} | {
        (r + j, c + 3) for j in range(3)}


def _frame_geometry(cells, direction):
    """Recognize a thin monotone envelope by cross-sections, not a stencil."""
    dr, dc = direction
    inward, lateral = (-dr, -dc), (-dc, dr)
    rows = {}
    for r, c in cells:
        y = r * inward[0] + c * inward[1]
        x = r * lateral[0] + c * lateral[1]
        rows.setdefault(y, []).append(x)
    y0, y1 = min(rows), max(rows)
    if y1 - y0 < 2 or len(rows[y0]) != 1:
        return None
    tip_x = rows[y0][0]
    old_left = old_right = tip_x
    split, cavity = False, set()
    for y in range(y0 + 1, y1 + 1):
        xs = sorted(rows.get(y, []))
        if not split and xs == [tip_x]:
            continue
        if len(xs) != 2:
            return None
        left, right = xs
        if not (left < tip_x < right and 0 <= old_left - left <= 1
                and 0 <= right - old_right <= 1):
            return None
        split = True
        old_left, old_right = left, right
        for x in range(left + 1, right):
            cavity.add((y * inward[0] + x * lateral[0],
                        y * inward[1] + x * lateral[1]))
    if not split:
        return None
    tip = (y0 * inward[0] + tip_x * lateral[0],
           y0 * inward[1] + tip_x * lateral[1])
    return tip, cavity


def _solid_windows(mask):
    h, w = mask.shape
    if h < 3 or w < 3:
        return []
    solid = np.ones((h - 2, w - 2), dtype=bool)
    for r in range(3):
        for c in range(3):
            solid &= mask[r:r + h - 2, c:c + w - 2]
    return [tuple(map(int, p)) for p in np.argwhere(solid)]


def _interpret(grid, bg, scaffold, frame_color, signals, hub):
    h, w = grid.shape
    hr, hc = hub
    if not (1 <= hr <= h - 4 and 1 <= hc <= w - 4):
        raise ValueError("collector needs an exterior port ring")
    objects, occupied = [], set()
    frame_components = _components(map(tuple, np.argwhere(grid == frame_color)), True)
    if not 2 <= len(frame_components) <= 4:
        raise ValueError("expected two to four launchers")
    for frame in frame_components:
        candidates = []
        for direction in DIRS:
            geometry = _frame_geometry(frame, direction)
            if geometry is None:
                continue
            tip, cavity = geometry
            if any(not (0 <= r < h and 0 <= c < w) for r, c in cavity):
                continue
            if any(int(grid[p]) not in (bg, *signals) for p in cavity):
                continue
            tokens = {p for p in cavity if grid[p] != bg}
            counts = [sum(grid[p] == color for p in tokens) for color in signals]
            if not min(counts) or counts[0] == counts[1] or len(tokens) == len(cavity):
                continue
            root = _add(tip, direction)
            if not (0 <= root[0] < h and 0 <= root[1] < w) or grid[root] != scaffold:
                continue
            candidates.append(dict(frame=frame, cavity=cavity, tip=tip,
                                   direction=direction, root=root, tokens=tokens,
                                   quantity=len(tokens), counts=tuple(counts),
                                   majority=signals[int(counts[1] > counts[0])]))
        if len(candidates) != 1:
            raise ValueError("launcher orientation or payload is ambiguous")
        obj = candidates[0]
        if occupied & (obj['frame'] | obj['cavity']):
            raise ValueError("overlapping launchers")
        occupied |= obj['frame'] | obj['cavity']
        objects.append(obj)
    signal_cells = set(map(tuple, np.argwhere(np.isin(grid, signals))))
    if signal_cells != set().union(*(o['tokens'] for o in objects)):
        raise ValueError("all signal cells must be inside launchers")
    hub_cells = _box(hr, hc, 3, 3)
    exterior = set(map(tuple, np.argwhere(grid == scaffold))) - hub_cells
    networks = _components(exterior)
    if len(networks) != len(objects):
        raise ValueError("one isolated conduit network is needed per launcher")
    all_ports = _ports(hub)
    labels = []
    for network in networks:
        owners = [o for o in objects if o['root'] in network]
        if len(owners) != 1:
            raise ValueError("a conduit must have exactly one launcher")
        obj = owners[0]
        potential = network & all_ports
        if not potential:
            raise ValueError("every network must lead to a collector port")
        distance = _distances(network, obj['root'])
        reached = {p for p in network if distance[p] <= obj['quantity']}
        delivered = potential & reached
        obj.update(network=network, distances=distance, potential=potential,
                   reached=reached, delivered=delivered)
        labels.extend([obj['majority']] * len(delivered))
    votes = tuple(labels.count(s) for s in signals)
    if not min(votes) or votes[0] == votes[1]:
        raise ValueError("both signals must arrive, with a strict port majority")
    return dict(background=bg, scaffold=scaffold, frame_color=frame_color,
                signals=signals, hub=hub, hub_cells=hub_cells, objects=objects,
                votes=votes, winner=signals[int(votes[1] > votes[0])])


def _read(input_grid):
    grid = np.asarray(input_grid)
    if (grid.ndim != 2 or not np.issubdtype(grid.dtype, np.integer)
            or not all(1 <= n <= 30 for n in grid.shape)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected an integer ARC grid")
    bg = int(grid[0, 0])
    if not (np.all(grid[0] == bg) and np.all(grid[-1] == bg)
            and np.all(grid[:, 0] == bg) and np.all(grid[:, -1] == bg)):
        raise ValueError("the exterior border must encode the background")
    colors = tuple(int(c) for c in np.unique(grid) if c != bg)
    if len(colors) != 4:
        raise ValueError("expected background, frame, conduit and two signals")
    interpretations = []
    for scaffold in colors:
        hubs = _solid_windows(grid == scaffold)
        if len(hubs) != 1:
            continue
        for frame_color in colors:
            if frame_color == scaffold:
                continue
            signals = tuple(c for c in colors if c not in (scaffold, frame_color))
            try:
                interpretations.append(_interpret(grid, bg, scaffold, frame_color,
                                                 signals, hubs[0]))
            except ValueError:
                pass
    if len(interpretations) != 1:
        raise ValueError("input needs one geometric role interpretation")
    return grid, interpretations[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid, info = _read(input_grid)
    output = grid.copy()
    for obj in info['objects']:
        for p in obj['tokens']:
            output[p] = info['background']
        for p in obj['reached']:
            output[p] = obj['majority']
    r, c = info['hub']
    output[r + 1, c + 1] = info['winner']
    return output


def _profile(rng, large=False):
    """Independent left/right walks, held slope steps, stem and cavity depth."""
    left = int(rng.integers(2 if large else 1, 4))
    right = int(rng.integers(2 if large else 1, 4))
    if rng.random() < .35:
        right = left
    stem = int(rng.integers(0, 2))
    tail = int(rng.integers(4 if large else 2, 7))
    walls = [(0, 0)] * (stem + 1)
    a = b = 0
    while a < left or b < right:
        da = a < left and (b == right or rng.random() < .85)
        db = b < right and (a == left or rng.random() < .85)
        if not da and not db:
            da = a < left
            db = not da
        # The first split must have a wall on both sides of the tip axis.
        if a == b == 0:
            da = db = True
        a, b = a + int(da), b + int(db)
        walls.append((-a, b))
    walls.extend([(-left, right)] * tail)
    frame, cavity = set(), set()
    for y, (lo, hi) in enumerate(walls):
        frame.update({(y, lo), (y, hi)})
        cavity.update((y, x) for x in range(lo + 1, hi))
    return frame, cavity


def _place_profile(profile, tip, direction):
    dr, dc = direction
    frame, cavity = profile
    def transform(p):
        y, x = p
        return tip[0] - y * dr - x * dc, tip[1] - y * dc + x * dr
    return {transform(p) for p in frame}, {transform(p) for p in cavity}


def _path(start, targets, allowed, rng, roughness):
    h, w = allowed.shape
    if not allowed[start]:
        return None
    costs = 1 + rng.random((h, w)) * roughness
    if roughness > 2:
        field = rng.random(((h + 2) // 3, (w + 2) // 3)) * roughness
        costs += np.repeat(np.repeat(field, 3, axis=0), 3, axis=1)[:h, :w]
    distances, parent = {start: 0.}, {}
    heap = [(0., start)]
    while heap:
        cost, p = heapq.heappop(heap)
        if cost != distances[p]:
            continue
        if p in targets:
            route = {p}
            while p != start:
                p = parent[p]
                route.add(p)
            return route
        for d in DIRS:
            q = _add(p, d)
            if 0 <= q[0] < h and 0 <= q[1] < w and allowed[q]:
                new_cost = cost + float(costs[q])
                if new_cost < distances.get(q, float('inf')):
                    distances[q], parent[q] = new_cost, p
                    heapq.heappush(heap, (new_cost, q))
    return None


def _layout(rng, weights, witness=False, compact=False):
    """Sample geometry, then connect grouped ports to separately placed tips."""
    count = len(weights)
    lo = 19 if compact else 25 if count == 4 else 23
    h, w = int(rng.integers(lo, 31)), int(rng.integers(lo, 31))
    if count == 4:
        hr = int(rng.integers(10, h - 11))
        hc = int(rng.integers(10, w - 11))
    else:
        hr = int(rng.integers(7, h - 9))
        hc = int(rng.integers(7, w - 9))
    hub = hr, hc
    center = hr + 1, hc + 1
    # Clockwise nonadjacent positions. Any one side has at most two ports.
    ring = [(hr - 1, hc), (hr - 1, hc + 2), (hr, hc + 3),
            (hr + 2, hc + 3), (hr + 3, hc + 2), (hr + 3, hc),
            (hr + 2, hc - 1), (hr, hc - 1)]
    selected = sorted(map(int, rng.choice(8, sum(weights), replace=False)))
    shift = int(rng.integers(len(selected)))
    selected = selected[shift:] + selected[:shift]
    groups, index = [], 0
    for weight in weights:
        groups.append({ring[j] for j in selected[index:index + weight]})
        index += weight
    # Usually use edge positions, but middle singletons are valid too.
    for group in groups:
        if len(group) == 1 and rng.random() < .35:
            r, c = next(iter(group))
            p = (r, hc + 1) if r in (hr - 1, hr + 3) else (hr + 1, c)
            if all(p not in other and not (_around({p}) & other)
                   for other in groups if other is not group):
                group.clear()
                group.add(p)
    occupied = _box(hr - 2, hc - 2, 7, 7)
    objects = []
    for i, group in enumerate(groups):
        profile = _profile(rng, large=witness or count <= 3)
        mean = np.mean(np.array(sorted(group)), axis=0) - np.array(center)
        angle = float(np.arctan2(mean[0], mean[1]))
        placements = []
        for r in range(2, h - 2):
            for c in range(2, w - 2):
                dy, dx = r - center[0], c - center[1]
                distance = abs(dy) + abs(dx)
                if not 5 <= distance <= (11 if count == 4 else 13):
                    continue
                delta = abs(float(np.arctan2(dy, dx)) - angle)
                if min(delta, 2 * np.pi - delta) > 1.0:
                    continue
                directions = [DIRS[2 if dy < 0 else 0]] if abs(dy) > abs(dx) else [DIRS[1 if dx < 0 else 3]]
                if not witness and rng.random() < .13:
                    directions = list(DIRS)
                for direction in directions:
                    frame, cavity = _place_profile(profile, (r, c), direction)
                    cells = frame | cavity
                    if (cells & occupied or any(not (1 <= y < h - 1 and 1 <= x < w - 1)
                                                for y, x in cells)):
                        continue
                    root = _add((r, c), direction)
                    if root in occupied or not (1 <= root[0] < h - 1 and 1 <= root[1] < w - 1):
                        continue
                    placements.append((frame, cavity, (r, c), direction, root))
        if not placements:
            return None
        frame, cavity, tip, direction, root = placements[int(rng.integers(len(placements)))]
        objects.append(dict(frame=frame, cavity=cavity, tip=tip, direction=direction,
                            root=root, potential=group))
        occupied |= frame | cavity | _around(frame | cavity, True) | {root}
    hub_cells = _box(hr, hc, 3, 3)
    collar = _box(hr - 1, hc - 1, 5, 5) - hub_cells
    body = set().union(*(o['frame'] | o['cavity'] for o in objects))
    networks = [set() for _ in objects]
    # Heaviest groups are routed first, avoiding the small networks' roots.
    for owner in sorted(range(count), key=lambda i: -weights[i]):
        obj = objects[owner]
        othernets = set().union(*(networks[i] for i in range(count) if i != owner))
        otherroots = {o['root'] for i, o in enumerate(objects) if i != owner}
        blocked = (hub_cells | collar | body | _around(body, True)
                   | othernets | _around(othernets) | otherroots | _around(otherroots))
        blocked -= obj['potential'] | {obj['root']}
        allowed = np.ones((h, w), dtype=bool)
        allowed[0] = allowed[-1] = False
        allowed[:, 0] = allowed[:, -1] = False
        for p in blocked:
            if 0 <= p[0] < h and 0 <= p[1] < w:
                allowed[p] = False
        net = {obj['root']}
        roughness = float(rng.choice([.1, 1., 3., 6.]))
        for port in [sorted(obj['potential'])[int(j)] for j in rng.permutation(len(obj['potential']))]:
            path = _path(port, net, allowed, rng, roughness)
            if path is None:
                return None
            net |= path
        if (net & collar) != obj['potential']:
            return None
        networks[owner] = net
        distance = _distances(net, obj['root'])
        obj.update(network=net, distances=distance)
    structural = hub_cells | set().union(*networks)
    mask = np.zeros((h, w), dtype=bool)
    for p in structural:
        mask[p] = True
    if _solid_windows(mask) != [hub]:
        return None
    return dict(shape=(h, w), hub=hub, hub_cells=hub_cells, objects=objects)


def _token_pattern(rng, cavity, quantity, majority, minority):
    cells = sorted(cavity)
    style = int(rng.integers(4))
    if style == 0:
        order = list(map(int, rng.permutation(len(cells))))
    elif style == 1:
        center = cells[int(rng.integers(len(cells)))]
        order = sorted(range(len(cells)), key=lambda j: (
            abs(cells[j][0] - center[0]) + abs(cells[j][1] - center[1]),
            float(rng.random())))
    elif style == 2:
        order = sorted(range(len(cells)), key=lambda j: ((sum(cells[j]) % 2), cells[j]))
    else:
        order = sorted(range(len(cells)), key=lambda j: (cells[j][1], cells[j][0]))
    chosen = [cells[j] for j in order[:quantity]]
    nminor = int(rng.integers(1, (quantity - 1) // 2 + 1))
    minority_indices = set(map(int, rng.choice(quantity, nminor, replace=False)))
    return {p: minority if j in minority_indices else majority for j, p in enumerate(chosen)}


def _paint(layout, roles, quantities, labels, patterns):
    bg, frame_color, scaffold, *signals = roles
    inp = np.full(layout['shape'], bg, dtype=np.int8)
    out = inp.copy()
    for p in layout['hub_cells']:
        inp[p] = out[p] = scaffold
    votes = {s: 0 for s in signals}
    for obj, quantity, label, pattern in zip(layout['objects'], quantities, labels, patterns):
        for p in obj['frame']:
            inp[p] = out[p] = frame_color
        for p, color in pattern.items():
            inp[p] = color
        for p in obj['network']:
            inp[p] = scaffold
            out[p] = label if obj['distances'][p] <= quantity else scaffold
        votes[label] += sum(obj['distances'][p] <= quantity for p in obj['potential'])
    assert min(votes.values()) > 0 and len(set(votes.values())) == 2
    r, c = layout['hub']
    out[r + 1, c + 1] = max(votes, key=votes.get)
    return {'input': inp, 'output': out}


def _witnesses(rng, roles):
    for _ in range(600):
        layout = _layout(rng, [3, 1, 1], witness=True)
        if layout is None:
            continue
        main, *others = layout['objects']
        distances = sorted(main['distances'][p] for p in main['potential'])
        upper = min(24, len(main['cavity']) - 1)
        low = max(3, distances[0])
        if low >= distances[1] or distances[-1] > upper or distances[-1] - low < 2:
            continue
        high = distances[-1]
        other_counts = []
        for obj in others:
            d = max(obj['distances'][p] for p in obj['potential'])
            cap = min(24, len(obj['cavity']) - 1)
            if max(3, d) > cap:
                break
            other_counts.append(int(rng.integers(max(3, d), cap + 1)))
        if len(other_counts) != 2:
            continue
        if sum(d <= high for d in main['distances'].values()) <= high:
            continue  # A branching demonstration must exceed a serial cell budget.
        label = roles[3 + int(rng.integers(2))]
        other = next(s for s in roles[3:] if s != label)
        labels = [label, other, other]
        fixed_patterns = [_token_pattern(rng, obj['cavity'], q, other, label)
                          for obj, q in zip(others, other_counts)]
        pairs = []
        for q in (low, high):
            pattern = _token_pattern(rng, main['cavity'], q, label, other)
            pairs.append(_paint(layout, roles, [q, *other_counts], labels,
                                [pattern, *fixed_patterns]))
        if all(np.array_equal(solve(p['input']), p['output']) for p in pairs):
            rng.shuffle(pairs)
            return pairs
    raise RuntimeError("bounded witness construction exhausted")


def _ordinary(rng, roles, test=False):
    for _ in range(600):
        if test:
            weights = list(([3, 1, 1, 1], [2, 2, 1, 1], [3, 2, 1, 1],
                            [2, 2, 2, 1], [3, 2, 2, 1])[int(rng.integers(5))])
        else:
            weights = list(([1, 1], [2, 1], [3, 1], [1, 1, 1],
                            [2, 1, 1], [3, 1, 1], [2, 2, 1])[int(rng.integers(7))])
        rng.shuffle(weights)
        layout = _layout(rng, weights, compact=len(weights) == 2)
        if layout is None:
            continue
        quantities = []
        for obj in layout['objects']:
            cap = min(24, len(obj['cavity']) - 1)
            nearest = min(obj['distances'][p] for p in obj['potential'])
            if cap < 3 or nearest > cap:
                break
            # Mixture of complete delivery and range gates; one test gate is required.
            if rng.random() < .5:
                q = int(rng.integers(max(3, nearest), cap + 1))
            else:
                q = int(rng.integers(3, cap + 1))
            quantities.append(q)
        if len(quantities) != len(weights):
            continue
        reached = [sum(obj['distances'][p] <= q for p in obj['potential'])
                   for obj, q in zip(layout['objects'], quantities)]
        partial = any(0 < n < len(obj['potential']) for obj, n in zip(layout['objects'], reached))
        if test and not partial:
            continue
        assignments = []
        for labels in itertools.product(roles[3:], repeat=len(weights)):
            totals = [sum(n for n, s in zip(reached, labels) if s == color) for color in roles[3:]]
            if min(totals) and totals[0] != totals[1]:
                assignments.append(labels)
        if not assignments:
            continue
        labels = assignments[int(rng.integers(len(assignments)))]
        patterns = [_token_pattern(rng, obj['cavity'], q, label,
                                   next(s for s in roles[3:] if s != label))
                    for obj, q, label in zip(layout['objects'], quantities, labels)]
        pair = _paint(layout, roles, quantities, labels, patterns)
        if np.array_equal(solve(pair['input']), pair['output']):
            return pair
    raise RuntimeError("bounded scene construction exhausted")


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10),
                                    ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice(10, 5, replace=False)))
    variable_palette = bool(rng.integers(2)) and ntrain >= 3
    witnesses = _witnesses(rng, roles)
    train = witnesses[:]
    seen = {_key(p['input']) for p in train}
    for _ in range(ntrain - 2):
        for attempt in range(12):
            local_roles = tuple(map(int, rng.choice(10, 5, replace=False))) if variable_palette else roles
            pair = _ordinary(rng, local_roles)
            if _key(pair['input']) not in seen:
                break
        else:
            raise RuntimeError("distinct training input construction exhausted")
        seen.add(_key(pair['input']))
        train.append(pair)
    rng.shuffle(train)
    puzzle = {'train': train, 'test': []}
    for _ in range(ntest):
        for attempt in range(12):
            local_roles = tuple(map(int, rng.choice(10, 5, replace=False))) if variable_palette else roles
            pair = _ordinary(rng, local_roles, test=True)
            if _key(pair['input']) not in seen:
                break
        else:
            raise RuntimeError("distinct input construction exhausted")
        seen.add(_key(pair['input']))
        puzzle['test'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    objects, signals = info['objects'], info['signals']
    majority_vote = tuple(sum(o['majority'] == s for o in objects) for s in signals)
    full_vote = tuple(sum(len(o['potential']) for o in objects if o['majority'] == s) for s in signals)
    raw_vote = tuple(sum(o['counts'][j] for o in objects) for j in range(2))
    def winner(votes):
        return None if votes[0] == votes[1] else signals[int(votes[1] > votes[0])]
    return dict(count=len(objects), quantities=tuple(o['quantity'] for o in objects),
                ports=sum(len(o['potential']) for o in objects), votes=info['votes'],
                gate=any(0 < len(o['delivered']) < len(o['potential']) for o in objects),
                branched=any(sum(q in o['network'] for q in (_add(p, d) for d in DIRS)) >= 3
                            for o in objects for p in o['network']),
                radius_not_area=any(len(o['reached']) > o['quantity'] for o in objects),
                port_counterexample=winner(majority_vote) not in (None, info['winner']),
                flood_counterexample=winner(full_vote) not in (None, info['winner']),
                raw_counterexample=winner(raw_vote) not in (None, info['winner']))


def _training_evidence(infos):
    features = [_features(info) for info in infos]
    assert any(f['port_counterexample'] for f in features), "training admits launcher counting"
    assert any(f['flood_counterexample'] for f in features), "training admits unlimited flooding"
    assert any(f['radius_not_area'] and f['branched'] for f in features), "no branching radius witness"
    # A matched pair must change only one launcher's payload, keep its majority,
    # change its delivered weight, and change the collector winner.
    for a, b in itertools.combinations(infos, 2):
        if (a['hub'] != b['hub'] or a['background'] != b['background']
                or a['signals'] != b['signals'] or a['scaffold'] != b['scaffold']
                or a['frame_color'] != b['frame_color'] or a['winner'] == b['winner']):
            continue
        aa, bb = a['objects'], b['objects']
        if len(aa) != len(bb):
            continue
        if all(x['frame'] == y['frame'] and x['network'] == y['network']
               and x['majority'] == y['majority'] for x, y in zip(aa, bb)):
            changed = [(x, y) for x, y in zip(aa, bb) if x['quantity'] != y['quantity']]
            if len(changed) == 1 and changed[0][0]['delivered'] != changed[0][1]['delivered']:
                return
    raise AssertionError("missing matched count-to-voting-power demonstration")


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = (np.asarray(pair[k]) for k in ('input', 'output'))
            for grid in (inp, out):
                assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            _, info = _read(inp)
            assert np.array_equal(solve(inp), out), "incorrect propagation, erasure or vote"
            # Explicit preservation and range invariants, reconstructed from the input.
            edited = {p for o in info['objects'] for p in o['tokens'] | o['reached']}
            r, c = info['hub']
            edited.add((r + 1, c + 1))
            for y, x in np.ndindex(inp.shape):
                if (y, x) not in edited:
                    assert inp[y, x] == out[y, x]
            for obj in info['objects']:
                assert all(out[p] == info['background'] for p in obj['tokens'])
                assert all(out[p] == obj['majority'] for p in obj['reached'])
                assert all(out[p] == info['scaffold'] for p in obj['network'] - obj['reached'])
            infos[split].append(info)
    _training_evidence(infos['train'])
    assert all(_features(i)['count'] == 4 and _features(i)['ports'] >= 6
               and _features(i)['gate'] for i in infos['test'])
    assert all(_features(i)['count'] <= 3 and _features(i)['ports'] <= 5 for i in infos['train'])


def self_test():
    """Bounded 42-seed diversity, interface, mutation and equivariance checks."""
    fingerprints, shapes, profiles, palettes, counts = set(), set(), set(), set(), set()
    quantities, directions, weights, backgrounds = set(), set(), set(), set()
    gates = branches = raw_disagreements = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, info = _read(pair['input'])
                f = _features(info)
                shapes.add(inp.shape)
                palettes.add((info['background'], info['frame_color'], info['scaffold'], *info['signals']))
                backgrounds.add(info['background'])
                counts.add(f['count'])
                gates += f['gate']
                branches += f['branched']
                raw_disagreements += f['raw_counterexample']
                for obj in info['objects']:
                    quantities.add(obj['quantity'])
                    directions.add(obj['direction'])
                    weights.add(len(obj['potential']))
                    tip = obj['tip']
                    profiles.add(tuple(sorted((r - tip[0], c - tip[1]) for r, c in obj['frame'])))
        # Input-only solver is equivariant under rotations, reflections and palette permutations.
        pair = puzzle['test'][0]
        rotated_in = np.rot90(pair['input'], seed % 4)
        rotated_out = np.rot90(pair['output'], seed % 4)
        if seed % 2:
            rotated_in, rotated_out = np.fliplr(rotated_in), np.fliplr(rotated_out)
        palette = np.roll(np.arange(10), seed % 10)
        assert np.array_equal(solve(palette[rotated_in]), palette[rotated_out])
        rearranged = pair['input'].copy()
        _, info = _read(rearranged)
        for obj in info['objects']:
            cells = sorted(obj['cavity'])
            values = sorted(int(rearranged[p]) for p in cells)
            for p, value in zip(cells, values):
                rearranged[p] = value
        assert np.array_equal(solve(rearranged), pair['output']), "payload arrangement must be irrelevant"
    assert len(fingerprints) == 42 and len(shapes) >= 25 and len(profiles) >= 40
    assert len(palettes) >= 30 and len(backgrounds) >= 8
    assert counts == {2, 3, 4} and directions == set(DIRS) and {1, 2, 3} <= weights
    assert len(quantities) >= 12 and gates >= 42 and branches >= 42 and raw_disagreements > 0
    first = generate_puzzle(0, 2, 1)
    repeated = generate_puzzle(0, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in first
               for a, b in zip(first[s], repeated[s]) for k in ('input', 'output'))
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain - 1) + 1):
            puzzle = generate_puzzle(1000 + 10 * ntrain + ntest, ntrain, ntest)
            assert len(puzzle['train']) == ntrain and len(puzzle['test']) == ntest
    for args in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3., 1), (4, None)):
        try:
            generate_puzzle(0, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # A changed collector label and an incorrectly flooded cell must both fail validation.
    for mode in ('center', 'flood'):
        broken = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
        pair = broken['train'][0]
        _, info = _read(pair['input'])
        if mode == 'center':
            r, c = info['hub']
            pair['output'][r + 1, c + 1] = info['scaffold']
        else:
            pair = next(p for p in broken['train'] if any(
                o['network'] - o['reached'] for o in _read(p['input'])[1]['objects']))
            _, info = _read(pair['input'])
            obj = next(o for o in info['objects'] if o['network'] - o['reached'])
            pair['output'][min(obj['network'] - obj['reached'])] = obj['majority']
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError("incorrect output accepted")


if __name__ == '__main__':
    self_test()
    print('self_test passed')
