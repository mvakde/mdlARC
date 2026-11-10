"""Geodesic harvest onto a folded rack. Standard library + NumPy only.

No files are read or written by this module. The input encodes every role.
See task_spec.md for the rule; construction policies are not solver rules.
"""
from collections import deque
import hashlib
import numpy as np


STEPS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0, 972762461, 976981591, 268911378, 833790873, 656948382,
              438148141, 753583279, 784121781, 484797388, 22866595, 613629750,
              76598033, 405946330, 669634939, 927183357, 88099025, 455128995,
              953526254, 630374743, 902231489, 877218094, 314229033, 260751356,
              182539246, 742023301, 887947043, 561503539, 641432320, 102181943,
              29501122, 76640837, 596651096, 986426526, 94631731, 415791943,
              95708076, 439397644, 425264342, 582673895, 579271342, 212758884)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "Grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)), "Grid colors")
    return a


def _trace(cells):
    """Return the graph order, or None if this is not an eligible rack."""
    cells = set(cells)
    if not 5 <= len(cells) <= 30:
        return None
    neighbors = {p: [(p[0]+dr, p[1]+dc) for dr, dc in STEPS
                     if (p[0]+dr, p[1]+dc) in cells] for p in cells}
    ends = [p for p in cells if len(neighbors[p]) == 1]
    if len(ends) != 2 or ends[0][0] == ends[1][0]:
        return None
    if any(len(v) not in (1, 2) for v in neighbors.values()):
        return None
    current, finish = sorted(ends)
    path, previous = [], None
    while current not in path:
        path.append(current)
        choices = [p for p in neighbors[current] if p != previous]
        if not choices:
            return path if len(path) == len(cells) and current == finish else None
        previous, current = current, choices[0]
    return None


def _roles(a):
    """Frame identifies walls; strict majority identifies background; shape identifies rack."""
    a = _grid(a)
    h, w = a.shape
    _require(h >= 5 and w >= 5, "Frame needs an interior")
    wall_color = int(a[0, 0])
    _require(np.all(a[0] == wall_color) and np.all(a[-1] == wall_color)
             and np.all(a[:, 0] == wall_color) and np.all(a[:, -1] == wall_color), "Closed frame")
    histogram = np.bincount(a.ravel().astype(int), minlength=10)
    background = int(histogram.argmax())
    _require(histogram[background] > a.size//2 and background != wall_color, "Background majority")
    candidates = []
    for color in range(10):
        if color in (background, wall_color):
            continue
        path = _trace([tuple(map(int, p)) for p in np.argwhere(a == color)])
        if path is not None:
            candidates.append((color, path))
    _require(len(candidates) == 1, "Exactly one monochrome rack chain")
    rack_color, path = candidates[0]
    wall = a == wall_color
    rack = a == rack_color
    payload = (a != background) & ~wall & ~rack
    return background, wall_color, rack_color, path, wall, rack, payload


def _distances(wall, path):
    """Distance zero at EVERY rack slot; tokens and background are traversable."""
    distance = np.full(wall.shape, -1, dtype=np.int16)
    queue = deque(path)
    for p in path:
        distance[p] = 0
    h, w = wall.shape
    while queue:
        r, c = queue.popleft()
        for dr, dc in STEPS:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and not wall[nr, nc] and distance[nr, nc] < 0:
                distance[nr, nc] = distance[r, c]+1
                queue.append((nr, nc))
    return distance


def _wave_distances(wall, rack):
    """Array wavefront audit of the queue implementation; no saved traces."""
    distance = np.full(wall.shape, -1, dtype=np.int16)
    seen, frontier = rack.copy(), rack.copy()
    distance[rack] = 0
    for depth in range(1, wall.size+1):
        spread = np.zeros_like(wall)
        spread[1:] |= frontier[:-1]
        spread[:-1] |= frontier[1:]
        spread[:, 1:] |= frontier[:, :-1]
        spread[:, :-1] |= frontier[:, 1:]
        frontier = spread & ~wall & ~seen
        if not frontier.any():
            break
        distance[frontier] = depth
        seen |= frontier
    return distance


def _analyze(a):
    a = _grid(a)
    bg, wc, rc, path, wall, rack, payload = _roles(a)
    distance = _distances(wall, path)
    reachable = [tuple(map(int, p)) for p in np.argwhere(payload & (distance >= 0))]
    reachable.sort(key=lambda p: int(distance[p]))
    depths = [int(distance[p]) for p in reachable]
    _require(len(set(depths)) == len(depths), "Reachable payload distances must be distinct")
    selected = reachable[:len(path)]
    return dict(background=bg, wall_color=wc, rack_color=rc, path=path, wall=wall,
                rack=rack, payload=payload, distance=distance, reachable=reachable,
                selected=selected, sequence=[int(a[p]) for p in selected],
                unreachable=int(np.count_nonzero(payload & (distance < 0))))


def _render(a, path, selected, background):
    result = a.copy()
    colors = [int(a[p]) for p in selected]
    for p in selected:
        result[p] = background
    for p, color in zip(path, colors):
        result[p] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the entire answer from this input, with no demonstrations or state."""
    a = _grid(input_grid)
    info = _analyze(a)
    return _render(a, info['path'], info['selected'], info['background'])


def _spatial_orders(path):
    row = sorted(path)
    column = sorted(path, key=lambda p: (p[1], p[0]))
    return (row, row[::-1], column, column[::-1], path[::-1])


def _manhattan(shape, path):
    rr, cc = np.indices(shape)
    return np.minimum.reduce([np.abs(rr-r)+np.abs(cc-c) for r, c in path])


def _features(a, info=None):
    info = _analyze(a) if info is None else info
    path, selected = info['path'], info['selected']
    sequence = info['sequence']
    runs = [c for i, c in enumerate(sequence) if i == 0 or c != sequence[i-1]]
    steps = [(b[0]-a0[0], b[1]-a0[1]) for a0, b in zip(path, path[1:])]
    retained = info['reachable'][len(path):]
    manhattan = _manhattan(a.shape, path)
    answer = _render(a, path, selected, info['background'])
    origin_shortcut = _render(a, path, sorted(info['reachable'], key=lambda p: (int(manhattan[p]), p))[:len(path)], info['background'])
    no_walls = [tuple(map(int, p)) for p in np.argwhere(info['payload'])]
    no_walls.sort(key=lambda p: (int(manhattan[p]), p))
    endpoint_distance = _distances(info['wall'], [path[0]])
    endpoint_order = sorted(info['reachable'], key=lambda p: (int(endpoint_distance[p]), p))
    arrival_alternatives = (sorted(selected), sorted(selected, key=lambda p: (p[1], p[0])),
                            sorted(selected, key=lambda p: (int(a[p]), p)))
    return dict(shape=tuple(a.shape), length=len(path), load=len(selected), supply=len(info['reachable']),
                unreachable=info['unreachable'], excess=len(retained), background=info['background'],
                wall_color=info['wall_color'], rack_color=info['rack_color'],
                turns=sum(x != y for x, y in zip(steps, steps[1:])),
                reversal=(0, 1) in steps and (0, -1) in steps,
                above=min(r for r, _ in path) < path[0][0],
                max_distance=int(info['distance'].max()),
                payload_colors=len(set(int(x) for x in a[info['payload']])),
                recurrence=len(runs) > len(set(runs)),
                path_witness=all(not np.array_equal(answer, _render(a, order, selected, info['background']))
                                 for order in _spatial_orders(path)),
                direction_witness=not np.array_equal(answer, _render(a, path[::-1], selected, info['background'])),
                endpoint_witness=not np.array_equal(answer, _render(a, path, endpoint_order[:len(path)], info['background'])),
                arrival_witness=all(not np.array_equal(answer, _render(a, path, order, info['background']))
                                    for order in arrival_alternatives),
                distance_witness=not np.array_equal(answer, origin_shortcut),
                wall_witness=not np.array_equal(answer, _render(a, path, no_walls[:len(path)], info['background'])),
                detour_frontier=bool(selected and retained and max(int(manhattan[p]) for p in selected)
                                     > min(int(manhattan[p]) for p in retained)),
                layout=tuple((r-path[0][0], c-path[0][1]) for r, c in path))


def _rectangle(wall, r0, c0, r1, c1):
    wall[r0, c0:c1+1] = True
    wall[r1, c0:c1+1] = True
    wall[r0:r1+1, c0] = True
    wall[r0:r1+1, c1] = True


def _geometry(rng):
    h, w = (int(rng.integers(16, 31)) for _ in range(2))
    wall = np.zeros((h, w), dtype=bool)
    wall[0] = wall[-1] = True
    wall[:, 0] = wall[:, -1] = True
    mode = int(rng.integers(5))
    if mode == 0:  # Recursive rooms, with closed cuts or one/two doorways.
        rooms = [(1, 1, h-2, w-2)]
        for _ in range(int(rng.integers(2, 7))):
            if not rooms:
                break
            r0, c0, r1, c1 = rooms.pop(int(rng.integers(len(rooms))))
            horizontal = (rng.random() < .5 and r1-r0 >= 5) or c1-c0 < 5
            if horizontal and r1-r0 >= 5:
                r = int(rng.integers(r0+2, r1-1))
                wall[r, c0:c1+1] = True
                for _ in range(int(rng.integers(3))):
                    wall[r, int(rng.integers(c0, c1+1))] = False
                rooms.extend([(r0, c0, r-1, c1), (r+1, c0, r1, c1)])
            elif c1-c0 >= 5:
                c = int(rng.integers(c0+2, c1-1))
                wall[r0:r1+1, c] = True
                for _ in range(int(rng.integers(3))):
                    wall[int(rng.integers(r0, r1+1)), c] = False
                rooms.extend([(r0, c0, r1, c-1), (r0, c+1, r1, c1)])
    elif mode == 1:  # Alternating long rails: geometry, not a fixed maze template.
        horizontal = bool(rng.integers(2))
        view = wall if horizontal else wall.T
        hh, ww = view.shape
        for j, r in enumerate(range(4, hh-3, int(rng.integers(3, 6)))):
            gap = int(rng.integers(2, min(6, ww-3)))
            if j % 2:
                view[r, 1:ww-gap] = True
            else:
                view[r, gap:ww-1] = True
    elif mode in (2, 3):  # Enclosures and open U-shaped rooms.
        for _ in range(int(rng.integers(1, 5))):
            r0, c0 = int(rng.integers(2, h-6)), int(rng.integers(2, w-6))
            r1, c1 = int(rng.integers(r0+3, h-2)), int(rng.integers(c0+3, w-2))
            _rectangle(wall, r0, c0, r1, c1)
            for _ in range(int(rng.integers(0 if mode == 2 else 1, 4))):
                side = int(rng.integers(4))
                if side < 2:
                    wall[r0 if side == 0 else r1, int(rng.integers(c0+1, c1))] = False
                else:
                    wall[int(rng.integers(r0+1, r1)), c0 if side == 2 else c1] = False
    # Mode 4 keeps the open-field special case, apart from the identifying frame.
    if rng.random() < .75:
        for _ in range(int(rng.integers(1, 4))):
            r, c = int(rng.integers(2, h-3)), int(rng.integers(2, w-3))
            if rng.random() < .5:
                wall[r, c:min(w-1, c+int(rng.integers(2, 9)))] = True
            else:
                wall[r:min(h-1, r+int(rng.integers(2, 9))), c] = True
    return wall


def _fallback_geometry():
    wall = np.zeros((25, 28), dtype=bool)
    wall[0] = wall[-1] = True
    wall[:, 0] = wall[:, -1] = True
    wall[3, 6:27] = True
    wall[3:21, 6] = wall[3:21, 12] = True
    _rectangle(wall, 1, 1, 5, 5)
    return wall


def _safety_rack(length):
    path = [(5, 8)]
    segments = [((0, 1), 2), ((1, 0), 2)]
    for j in range(7):
        segments.extend([((0, -1 if j % 2 == 0 else 1), 3), ((1, 0), 2)])
    for (dr, dc), count in segments:
        for _ in range(count):
            if len(path) == length:
                return path
            path.append((path[-1][0]+dr, path[-1][1]+dc))
    return path[:length]


def _rack(rng, wall, length, witness=False):
    available = np.argwhere(~wall)
    h, w = wall.shape
    for _ in range(90):
        path = [tuple(map(int, available[int(rng.integers(len(available)))]))]
        used = set(path)
        previous = None
        persistence = float(rng.uniform(.5, 5))
        for _ in range(length-1):
            choices, weights = [], []
            r, c = path[-1]
            for dr, dc in STEPS:
                q = (r+dr, c+dc)
                if not (0 <= q[0] < h and 0 <= q[1] < w) or wall[q] or q in used:
                    continue
                if sum((q[0]+a, q[1]+b) in used for a, b in STEPS) != 1:
                    continue
                choices.append((q, (dr, dc)))
                weights.append(persistence if previous == (dr, dc) else 1.)
            if not choices:
                break
            weights = np.asarray(weights)
            q, previous = choices[int(rng.choice(len(choices), p=weights/weights.sum()))]
            path.append(q)
            used.add(q)
        if len(path) != length:
            continue
        path = _trace(path)
        if path is None or (witness and any(path == order for order in _spatial_orders(path))):
            continue
        return path
    return None


def _pick_levels(rng, distance, mask, count, forced=()):
    if count == 0:
        return []
    levels = np.unique(distance[mask])
    if len(levels) < count:
        return None
    chosen = list(forced)
    forced_levels = {int(distance[p]) for p in chosen}
    _require(len(forced_levels) == len(chosen) <= count, "Distinct forced arrival layers")
    choices = [int(d) for d in levels if d not in forced_levels]
    for d in np.asarray(rng.choice(choices, size=count-len(chosen), replace=False)).ravel():
        cells = np.argwhere(mask & (distance == int(d)))
        chosen.append(tuple(map(int, cells[int(rng.integers(len(cells)))])))
    return sorted(chosen, key=lambda p: int(distance[p]))


def _supply(rng, distance, rack, path, count, detour, footprint=False):
    active = (distance > 0) & ~rack
    if not detour:
        return _pick_levels(rng, distance, active, count)
    capacity = len(path)
    levels = np.unique(distance[active])
    manhattan = _manhattan(distance.shape, path)
    forced = []
    if footprint:
        # Depth-one bead far from the first endpoint precedes a depth-two bead
        # near it. This visibly distinguishes whole-rack versus endpoint intake.
        end_distance = _distances(distance < 0, [path[0]])
        first = np.argwhere(active & (distance == 1))
        second = np.argwhere(active & (distance == 2))
        if not len(first) or not len(second):
            return None
        p = tuple(map(int, max(first, key=lambda p: int(end_distance[tuple(p)]))))
        q = tuple(map(int, min(second, key=lambda p: int(end_distance[tuple(p)]))))
        if end_distance[p] <= end_distance[q]:
            return None
        forced = [p, q]
    thresholds = levels[capacity-1:len(levels)-(count-capacity)]
    for threshold in rng.permutation(thresholds):
        near = active & (distance <= threshold)
        far = active & (distance > threshold)
        if not far.any():
            continue
        best = int(manhattan[far].min())
        choices_mask = near & (manhattan > best)
        for p in forced:
            choices_mask &= distance != distance[p]
        choices = np.argwhere(choices_mask)
        if not len(choices):
            continue
        a = tuple(map(int, choices[int(rng.integers(len(choices)))]))
        bs = np.argwhere(far & (manhattan < manhattan[a]))
        b = tuple(map(int, bs[int(rng.integers(len(bs)))]))
        selected = _pick_levels(rng, distance, near, capacity, forced+[a])
        retained = _pick_levels(rng, distance, far, count-capacity, [b])
        if selected is not None and retained is not None:
            return selected+retained
    return None


def _sequence(rng, count, palette, recurrent):
    if not count:
        return []
    mode = int(rng.integers(3))
    sequence = []
    while len(sequence) < count:
        choices = [c for c in palette if not sequence or c != sequence[-1]]
        color = int(rng.choice(choices))
        run = 1 if mode == 0 else int(rng.integers(1, 4))
        sequence.extend([color]*min(run, count-len(sequence)))
    if recurrent and count >= 3:
        sequence[:3] = [palette[0], palette[1], palette[0]]
    return sequence


def _pair(rng, fixed_roles, length, count, unreachable, witness, detour, allow_above=True):
    """Bounded attempts; last attempt is a guaranteed long U-detour construction."""
    for attempt in range(25):
        fallback = attempt == 24
        wall = _fallback_geometry() if fallback else _geometry(rng)
        path = _safety_rack(length) if fallback else _rack(rng, wall, length, witness)
        if path is None:
            continue
        if not allow_above and min(r for r, _ in path) < path[0][0]:
            continue
        rack = np.zeros_like(wall)
        for p in path:
            rack[p] = True
        _require(not (wall & rack).any(), "Construction overlap")
        distance = _distances(wall, path)
        supply = _supply(rng, distance, rack, path, count, detour, footprint=witness)
        inactive = np.argwhere(~wall & ~rack & (distance < 0))
        if supply is None or len(inactive) < unreachable:
            continue
        distraction = [tuple(map(int, inactive[int(i)]))
                       for i in np.asarray(rng.choice(len(inactive), size=unreachable, replace=False)).ravel()]
        roles = fixed_roles if fixed_roles is not None else tuple(map(int, rng.choice(10, 3, replace=False)))
        bg, wc, rc = roles
        remaining = [c for c in range(10) if c not in roles]
        rng.shuffle(remaining)
        palette = remaining[:int(rng.integers(2, 8))]
        for _ in range(32):
            colors = _sequence(rng, len(supply), palette, witness)
            a = np.full(wall.shape, bg, dtype=np.int8)
            a[wall], a[rack] = wc, rc
            for p, color in zip(supply, colors):
                a[p] = color
            for p in distraction:
                a[p] = int(rng.choice(palette))
            if np.count_nonzero(a == bg) <= a.size//2:
                break
            try:
                info = _analyze(a)
            except AssertionError:
                continue
            selected = supply[:length]
            b = _render(a, path, selected, bg)
            f = _features(a, info)
            if witness and not (f['path_witness'] and f['recurrence'] and f['endpoint_witness'] and f['arrival_witness']):
                continue
            if detour and not f['detour_frontier']:
                continue
            return {'input': a, 'output': b}
    raise AssertionError("Bounded rack construction failed")


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(rng, num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (not _integer(value) or not low <= int(value) <= high):
            raise ValueError("Illegal example count")
    if num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train)-1) + 1)) if num_train is not None else int(rng.integers(1, 5))
    if num_train is None:
        num_train = int(rng.integers(max(2, int(num_test)+1), 11))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    num_train, num_test = _counts(rng, num_train, num_test)
    fixed_roles = tuple(map(int, rng.choice(10, 3, replace=False))) if rng.random() < .4 else None
    full_length = int(rng.integers(6, 9))
    partial_length = int(rng.integers(full_length+2, 11))
    # Obligations are shuffled: indices never predict the output load.
    specs = [(partial_length, int(rng.integers(full_length+1, partial_length)), int(rng.integers(1, 5)), False, False),
             (full_length, full_length+int(rng.integers(1, 6)), int(rng.integers(1, 5)), True, True)]

    def random_spec(target, allow_zero=True):
        length = int(rng.integers(11, 17) if target else rng.integers(5, 11))
        u = float(rng.random())
        count = (0 if u < .04 and allow_zero else int(rng.integers(1, length)) if u < .55
                 else length if u < .73 else length+int(rng.integers(1, 7)))
        unreachable = int(rng.integers(0 if count else 1, 6))
        return length, count, unreachable, False, False

    specs.extend(random_spec(False) for _ in range(num_train-2))
    specs = [specs[int(i)] for i in rng.permutation(num_train)]
    puzzle, seen = {'train': [], 'test': []}, set()
    test_specs = [random_spec(True, allow_zero=index > 0) for index in range(num_test)]
    for split, requests in (('train', specs), ('test', test_specs)):
        for spec in requests:
            training_features = [] if split == 'train' else [_features(p['input']) for p in puzzle['train']]
            allow_above = split == 'train' or any(f['above'] and f['direction_witness'] for f in training_features)
            for _ in range(8):
                pair = _pair(rng, fixed_roles, *spec, allow_above=allow_above)
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("Duplicate inputs after bounded retries")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check actual pixels, conservation, prefix order, selection, and teaching evidence."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "Puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Split lists")
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "Pair counts")
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "Pair keys")
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape, "Preserved canvas")
            key = (a.shape, a.astype(np.int8).tobytes())
            _require(key not in seen, "Duplicate input")
            seen.add(key)
            info = _analyze(a)
            path, selected, seq = info['path'], info['selected'], info['sequence']
            erased = np.zeros(a.shape, dtype=bool)
            for p in selected:
                erased[p] = True
            _require(np.array_equal(a[~(erased | info['rack'])], b[~(erased | info['rack'])]), "Unselected field preserved")
            _require(np.all(b[erased] == info['background']), "Selected supply erased")
            _require([int(b[p]) for p in path] == seq+[info['rack_color']]*(len(path)-len(seq)), "Ordered rack prefix")
            for color in set(int(c) for c in a[info['payload']]):
                _require(np.count_nonzero(a == color) == np.count_nonzero(b == color), "Payload color conservation")
            _require(np.count_nonzero(b == info['background']) - np.count_nonzero(a == info['background']) == len(seq), "One erasure per filled slot")
            _require(np.array_equal(solve(a.copy()), b), "Input-only solver agreement")
            features[split].append(_features(a, info))
    tr = features['train']
    _require(any(0 < f['load'] < f['length'] for f in tr), "Partial prefix demonstration")
    _require(any(f['excess'] and f['detour_frontier'] and f['distance_witness'] for f in tr), "Capacity and geodesic-distance demonstration")
    _require(any(f['unreachable'] for f in tr), "Unreachable distraction demonstration")
    _require(any(f['path_witness'] and f['recurrence'] for f in tr), "Ordered folding and repeated-color demonstration")
    _require(any(f['endpoint_witness'] and f['arrival_witness'] for f in tr), "Footprint and arrival-order demonstration")
    _require(any(f['load'] == f['length'] and f['length'] != 9 for f in tr), "Shape capacity differs from fixed nine")
    _require(any(a['load'] > b['length'] for a in tr for b in tr if b['excess']), "No fixed capacity explains both demonstrations")
    _require(any(f['load'] > 0 for f in features['test']), "At least one active test query")
    if any(f['above'] for f in features['test']):
        _require(any(f['above'] and f['direction_witness'] for f in tr), "Higher-endpoint convention must be taught")


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for name in ('input', 'output'):
                a = np.asarray(pair[name], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeds, streamed metrics, explicit count matrix, and semantic mutations.

    At most two puzzles are retained; no disk use, logs, retries, or trace files.
    """
    signatures, shapes, layouts, lengths, loads, turns, palettes = (set() for _ in range(7))
    roles = [set(), set(), set()]
    flags = {name: 0 for name in ('partial', 'excess', 'zero', 'above', 'reversal', 'path_witness', 'distance_witness')}
    for index, seed in enumerate(HOST_SEEDS):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = pair['input']
                info = _analyze(a)
                f = _features(a, info)
                shapes.add(f['shape'])
                layouts.add(f['layout'])
                lengths.add(f['length'])
                loads.add(f['load'])
                turns.add(f['turns'])
                palettes.add(tuple(sorted(int(c) for c in np.unique(a))))
                for values, field in zip(roles, ('background', 'wall_color', 'rack_color')):
                    values.add(f[field])
                flags['partial'] += 0 < f['load'] < f['length']
                flags['excess'] += f['excess'] > 0
                flags['zero'] += f['load'] == 0
                for name in ('above', 'reversal', 'path_witness', 'distance_witness'):
                    flags[name] += bool(f[name])
                if index % 7 == 0:
                    _require(np.array_equal(info['distance'], _wave_distances(info['wall'], info['rack'])), "BFS/wavefront disagreement")
        _require(max(len(_analyze(p['input'])['path']) for p in puzzle['train'])
                 < min(len(_analyze(p['input'])['path']) for p in puzzle['test']), "Capacity extrapolation missing")
    _require(len(signatures) == 42 and len(shapes) >= 65 and len(layouts) >= 150, "Structural diversity")
    _require(len(lengths) >= 11 and len(loads) >= 13 and len(turns) >= 9 and len(palettes) >= 35, "Feature diversity")
    _require(all(len(x) == 10 for x in roles), "Role palette diversity")
    _require(flags['partial'] >= 42 and flags['excess'] >= 42 and flags['zero'] >= 1, "Load regimes")
    _require(flags['above'] >= 8 and flags['reversal'] >= 25, "Fold vocabulary")
    _require(flags['path_witness'] >= 42 and flags['distance_witness'] >= 42, "Both inherited concepts matter")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            puzzle = generate_puzzle(901+nt*10+ne, nt, ne)
            _require(len(puzzle['train']) == nt and len(puzzle['test']) == ne, "Explicit counts")
    for counts in ((None, None), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(491, *counts))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3., 1), (4, False)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted")
    _require(_digest(generate_puzzle(0)) == _digest(generate_puzzle(0)), "Determinism")
    # Hand-specified labels, independent of the generated selection lists.
    fixture = np.zeros((14, 16), dtype=np.int8)
    fixture[0] = fixture[-1] = 5
    fixture[:, 0] = fixture[:, -1] = 5
    path = [(3, 4), (3, 5), (3, 6), (4, 6), (5, 6), (5, 5)]
    beads = [(3, 3), (7, 5), (8, 5), (9, 5), (10, 5), (11, 5), (12, 5)]
    colors = [2, 1, 2, 3, 1, 3, 2]
    for p in path:
        fixture[p] = 8
    for p, color in zip(beads, colors):
        fixture[p] = color
    pocket = np.zeros(fixture.shape, dtype=bool)
    _rectangle(pocket, 8, 10, 11, 13)
    fixture[pocket] = 5
    fixture[9, 11] = 1
    expected = fixture.copy()
    for p in beads[:6]:
        expected[p] = 0
    for p, color in zip(path, colors[:6]):
        expected[p] = color
    _require(np.array_equal(solve(fixture), expected), "Hand capacity/cluster/recurrence label")
    partial = fixture.copy()
    for p in beads[3:]:
        partial[p] = 0
    partial_expected = partial.copy()
    for p in beads[:3]:
        partial_expected[p] = 0
    for p, color in zip(path[:3], colors[:3]):
        partial_expected[p] = color
    _require(np.array_equal(solve(partial), partial_expected), "Hand partial-prefix label")
    zero = partial.copy()
    for p in beads[:3]:
        zero[p] = 0
    _require(np.array_equal(solve(zero), zero), "Hand unreachable-only identity label")
    for offset in range(10):
        _require(np.array_equal(solve((fixture+offset) % 10), (expected+offset) % 10), "Literal palette independence")
    # Track slot indices under each symmetry. A rotated answer must be rewritten
    # when the higher endpoint changes; simply rotating a filled rack is wrong.
    indices = np.zeros_like(fixture)
    erase = np.zeros_like(fixture, dtype=bool)
    for i, p in enumerate(path, 1):
        indices[p] = i
    for p in beads[:6]:
        erase[p] = True
    for rotation in range(4):
        for reflection in (False, True):
            a, slots, removed = [np.rot90(v, rotation) for v in (fixture, indices, erase)]
            if reflection:
                a, slots, removed = [np.fliplr(v) for v in (a, slots, removed)]
            first_row = np.argwhere(slots == 1)[0, 0]
            last_row = np.argwhere(slots == 6)[0, 0]
            b = a.copy()
            b[removed] = 0
            order = range(1, 7) if first_row < last_row else range(6, 0, -1)
            for index, color in zip(order, colors[:6]):
                b[slots == index] = color
            _require(np.array_equal(solve(a), b), "Symmetry requires endpoint reorientation")
    above = np.zeros((14, 16), dtype=np.int8)
    above[0] = above[-1] = 5
    above[:, 0] = above[:, -1] = 5
    above_path = [(5, 4), (4, 4), (3, 4), (3, 5), (3, 6), (4, 6), (5, 6), (6, 6)]
    for p in above_path:
        above[p] = 8
    for p, color in zip(((5, 3), (7, 7), (8, 7)), (2, 3, 1)):
        above[p] = color
    above_expected = above.copy()
    above_expected[5, 3] = above_expected[7, 7] = above_expected[8, 7] = 0
    above_expected[5, 4], above_expected[4, 4], above_expected[3, 4] = 2, 3, 1
    _require(np.array_equal(solve(above), above_expected), "Higher endpoint, not highest interior cell")
    # Broad solver scope: long racks beyond the generation scale bands.
    wall = _fallback_geometry()
    for length in (5, 6, 10, 16, 24, 30):
        path = _safety_rack(length)
        _require(_trace(path) == path and not any(wall[p] for p in path), "Fallback chain")
        d = _distances(wall, path)
        mask = np.zeros(wall.shape, bool)
        for p in path:
            mask[p] = True
        _require(np.array_equal(d, _wave_distances(wall, mask)), "Long rack distance audit")
        cells = _supply(np.random.default_rng(length), d, mask, path, length+3, False)
        _require(cells is not None, "Long fixture has sufficient arrival layers")
        a = np.zeros(wall.shape, dtype=np.int8)
        a[wall], a[mask] = 5, 8
        for i, p in enumerate(cells):
            a[p] = (1, 2, 3)[i % 3]
        _require(np.array_equal(solve(a), _render(a, path, cells[:length], 0)), "Long-rack transfer")
    # Mutations must be rejected, rather than merely recounted as valid labels.
    puzzle = generate_puzzle(817, 2, 1)
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted output accepted")
    # Input-only tests for distance ties and equal-row endpoints use a simple field.
    tied = np.zeros((14, 16), dtype=np.int8)
    tied[0] = tied[-1] = 5
    tied[:, 0] = tied[:, -1] = 5
    tied[3:9, 6] = 8
    tied[4, 5], tied[6, 7] = 1, 2
    try:
        solve(tied)
    except AssertionError:
        pass
    else:
        raise AssertionError("Arrival tie accepted")
    _require(_trace([(2, 2), (3, 2), (4, 2), (4, 3), (4, 4), (3, 4), (2, 4)]) is None, "Endpoint tie accepted")
    return {'seeds': 42, 'distinct_puzzles': len(signatures), 'shapes': len(shapes),
            'rack_geometries': len(layouts), 'lengths': sorted(lengths), 'loads': sorted(loads),
            'turn_counts': sorted(turns), 'palette_sets': len(palettes), 'features': flags}
