"""Demand mosaic: radial samples define color-specific collection capacities.

No filesystem access or generation state is used by solve. Only standard-library
modules and NumPy are required. Diagnostics stream scenes and retain no traces.
"""
from collections import deque
import hashlib

import numpy as np


DIRS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
        (0, 1), (1, -1), (1, 0), (1, 1))
STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
GOODS = (1, 2, 3, 4, 6, 7, 8)
COL_ORDER = (0, 3, 5, 1, 6, 2, 4, 7)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not (_integer(num_train) and _integer(num_test)):
        raise ValueError("Counts must be integers.")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("Require 2..10 train, 1..4 test, and train > test.")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "Invalid grid dimensions.")
    _require(np.issubdtype(a.dtype, np.integer), "Grid cells must be integers.")
    _require(np.all((0 <= a) & (a <= 9)), "Invalid colors.")
    return a


def _analyze(value):
    a = _grid(value)
    markers = np.argwhere(a == 9)
    _require(len(markers) == 1, "Exactly one marker 9 is required.")
    r, c = map(int, markers[0])
    h, w = a.shape
    _require(1 <= r < h - 1 and 1 <= c < w - 1, "Collector does not fit.")
    hub = a[r-1:r+2, c-1:c+2]
    carrier = int(hub[0, 0])
    _require(carrier in GOODS, "Invalid carrier role.")
    perimeter = hub.copy()
    perimeter[1, 1] = carrier
    _require(np.all(perimeter == carrier), "Malformed collector.")
    candidates = []
    max_radius = min(r-1, c-1, h-r-2, w-c-2)
    for radius in range(4, max_radius + 1):
        tiles = np.zeros(a.shape, dtype=bool)
        samples, views, valid = [], [], True
        for dr, dc in DIRS:
            y, x = r + radius * dr - 1, c + radius * dc - 1
            tile = a[y:y+3, x:x+3]
            sample = int(tile[1-dr, 1-dc])
            if (np.any((tile == 0) | (tile == 5) | (tile == 9))
                    or np.count_nonzero(tile == carrier) < 5
                    or sample == carrier):
                valid = False
                break
            tiles[y:y+3, x:x+3] = True
            samples.append(sample)
            views.append(tile)
        if valid:
            hub_mask = np.zeros(a.shape, dtype=bool)
            hub_mask[r-1:r+2, c-1:c+2] = True
            hardware = tiles | hub_mask
            if np.any((a == carrier) & ~hardware):
                continue
            candidates.append(dict(center=(r, c), radius=radius, carrier=carrier,
                                   tiles=tiles, hub=hub_mask, hardware=hardware,
                                   samples=tuple(samples), views=tuple(views)))
    _require(len(candidates) == 1, "The eight-key device must have a unique radius.")
    info = candidates[0]
    field = ~info['hardware']
    _require(np.all(np.isin(a[field], [0, 5] + [v for v in GOODS if v != carrier])),
             "Only floor, walls and loose goods may occur outside hardware.")
    info['blocked'] = info['tiles'] | (a == 5)
    info['field'] = field
    return a, info


def _distances(blocked, center, wave=False, marker_origin=False):
    """BFS for solving; separate boolean wavefront for validation."""
    r, c = center
    distance = np.full(blocked.shape, -1, dtype=np.int16)
    starts = ([(r, c)] if marker_origin else
              [(r+i, c+j) for i in (-1, 0, 1) for j in (-1, 0, 1)])
    for point in starts:
        distance[point] = 0
    if wave:
        frontier = distance == 0
        for depth in range(1, blocked.size + 1):
            spread = np.zeros(blocked.shape, dtype=bool)
            spread[1:] |= frontier[:-1]
            spread[:-1] |= frontier[1:]
            spread[:, 1:] |= frontier[:, :-1]
            spread[:, :-1] |= frontier[:, 1:]
            frontier = spread & ~blocked & (distance < 0)
            if not np.any(frontier):
                break
            distance[frontier] = depth
        return distance
    queue = deque(starts)
    h, w = blocked.shape
    while queue:
        y, x = queue.popleft()
        depth = int(distance[y, x]) + 1
        for dy, dx in STEPS:
            ny, nx = y+dy, x+dx
            if 0 <= ny < h and 0 <= nx < w and not blocked[ny, nx] and distance[ny, nx] < 0:
                distance[ny, nx] = depth
                queue.append((ny, nx))
    return distance


def _selection(a, info, distance):
    selections = {}
    for color in sorted(set(info['samples'])):
        quota = info['samples'].count(color)
        eligible = info['field'] & (a == color) & (distance >= 0)
        ordered = np.sort(distance[eligible])
        if len(ordered) > quota:
            _require(ordered[quota-1] < ordered[quota], "Tie across a color's capacity frontier.")
        threshold = int(ordered[min(quota, len(ordered))-1]) if len(ordered) else -1
        selections[color] = eligible & (distance <= threshold)
    return selections


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive demands, paths, selections and destinations entirely from this input."""
    a, info = _analyze(input_grid)
    distance = _distances(info['blocked'], info['center'])
    selections = _selection(a, info, distance)
    result = a.copy()
    r, c = info['center']
    for color, selected in selections.items():
        result[selected] = 0
        slots = [i for i, value in enumerate(info['samples']) if value == color]
        for i in slots[:int(selected.sum())]:
            dr, dc = DIRS[i]
            result[r+dr, c+dc] = color
    return result


def _rect(wall, r0, c0, r1, c1):
    wall[r0, c0:c1+1] = True
    wall[r1, c0:c1+1] = True
    wall[r0:r1+1, c0] = True
    wall[r0:r1+1, c1] = True


def _walls(rng, shape, center, radius, hardware, kind):
    h, w = shape
    r, c = center
    wall = np.zeros(shape, dtype=bool)
    if kind == 'detour':
        # A barrier outside the ring has a remote endpoint opening. Its side,
        # opening end, lengths and offsets vary with the actual scene.
        side = int(rng.integers(4))
        gap = int(rng.integers(1, 4))
        reverse = bool(rng.integers(2))
        if side < 2:
            y = r-radius-3 if side == 0 else r+radius+3
            if reverse:
                wall[y, gap:] = True
            else:
                wall[y, :w-gap] = True
        else:
            x = c-radius-3 if side == 2 else c+radius+3
            if reverse:
                wall[gap:, x] = True
            else:
                wall[:h-gap, x] = True
        # The top-left corner is outside hardware (minimum margin four).
        _rect(wall, 0, 0, 2, 2)
    elif kind == 'maze':
        wall[:] = True
        nodes = [(y, x) for y in range(1, h-1, 2) for x in range(1, w-1, 2)]
        start = nodes[int(rng.integers(len(nodes)))]
        stack, seen = [start], {start}
        wall[start] = False
        while stack:
            y, x = stack[-1]
            options = [(y+2*dy, x+2*dx) for dy, dx in STEPS
                       if 1 <= y+2*dy < h-1 and 1 <= x+2*dx < w-1
                       and (y+2*dy, x+2*dx) not in seen]
            if not options:
                stack.pop()
                continue
            ny, nx = options[int(rng.integers(len(options)))]
            wall[(y+ny)//2, (x+nx)//2] = False
            wall[ny, nx] = False
            seen.add((ny, nx))
            stack.append((ny, nx))
        for _ in range(int(rng.integers(1, 5))):
            y, x = int(rng.integers(h-3)), int(rng.integers(w-3))
            wall[y:y+int(rng.integers(2, 5)), x:x+int(rng.integers(2, 5))] = False
    elif kind == 'partitions':
        rooms = [(0, 0, h-1, w-1)]
        for _ in range(int(rng.integers(3, 10))):
            if not rooms:
                break
            j = int(rng.integers(len(rooms)))
            y0, x0, y1, x1 = rooms.pop(j)
            horizontal = (rng.random() < .5 and y1-y0 >= 5) or x1-x0 < 5
            if horizontal and y1-y0 >= 5:
                y = int(rng.integers(y0+2, y1-1))
                wall[y, x0:x1+1] = True
                if rng.random() < .8:
                    x = int(rng.integers(x0, x1+1))
                    wall[y, x:x+int(rng.integers(1, 4))] = False
                rooms.extend([(y0, x0, y-1, x1), (y+1, x0, y1, x1)])
            elif x1-x0 >= 5:
                x = int(rng.integers(x0+2, x1-1))
                wall[y0:y1+1, x] = True
                if rng.random() < .8:
                    y = int(rng.integers(y0, y1+1))
                    wall[y:y+int(rng.integers(1, 4)), x] = False
                rooms.extend([(y0, x0, y1, x-1), (y0, x+1, y1, x1)])
    elif kind in ('segments', 'rooms'):
        for _ in range(int(rng.integers(2, 9))):
            y0, x0 = int(rng.integers(h-3)), int(rng.integers(w-3))
            y1, x1 = int(rng.integers(y0+2, h)), int(rng.integers(x0+2, w))
            if kind == 'rooms':
                _rect(wall, y0, x0, y1, x1)
                if rng.random() < .7:
                    y = int(rng.integers(y0+1, y1))
                    wall[y:y+int(rng.integers(1, 3)), x1] = False
            elif rng.random() < .5:
                wall[y0, x0:x1+1] = True
                if rng.random() < .6:
                    wall[y0:y1+1, x1] = True
            else:
                wall[y0:y1+1, x0] = True
    elif kind != 'open':
        raise ValueError("Unknown wall geometry.")
    wall[hardware] = False
    return wall


def _demand(rng, palette, witness=False):
    if witness:
        colors = palette[:int(rng.integers(2, 5))]
        quota = int(rng.integers(3, min(5, 9-len(colors)) + 1))
        for _ in range(64):
            positions = sorted(map(int, rng.choice(8, quota, replace=False)))
            column = [i for i in COL_ORDER if i in positions]
            prefixes = [k for k in range(1, quota) if set(positions[:k]) != set(column[:k])]
            if prefixes:
                break
        else:
            positions, quota, prefixes = [1, 3, 4], 3, [1]
        samples = [None] * 8
        for i in positions:
            samples[i] = colors[0]
        remaining = [i for i in range(8) if samples[i] is None]
        labels = list(colors[1:]) + [int(rng.choice(colors[1:]))
                                   for _ in range(len(remaining)-len(colors)+1)]
        rng.shuffle(labels)
        for i, color in zip(remaining, labels):
            samples[i] = color
        return tuple(samples), colors[0], int(rng.choice(prefixes))
    count = int(rng.integers(1, 7))
    if count <= 2 and rng.random() < .3:
        samples = tuple(palette[0] if count == 1 or dr*dc else palette[1] for dr, dc in DIRS)
    else:
        labels = list(range(count)) + list(map(int, rng.integers(count, size=8-count)))
        rng.shuffle(labels)
        samples = tuple(palette[j] for j in labels)
    return samples, samples[0], 0


def _pick(rng, mask, count, style, forced=()):
    points = list(dict.fromkeys(map(int, forced)))
    available = np.flatnonzero(mask).tolist()
    _require(len(available) >= count and set(points) <= set(available), "Insufficient supply cells.")
    if count == len(points):
        return points
    available = [i for i in available if i not in points]
    anchor = points[0] if points else int(rng.choice(available))
    w = mask.shape[1]
    y, x = divmod(anchor, w)
    rng.shuffle(available)
    if style == 'compact':
        available.sort(key=lambda i: (i//w-y)**2 + (i%w-x)**2)
    elif style == 'line':
        horizontal = bool(rng.integers(2))
        available.sort(key=lambda i: abs(i//w-y)*w+abs(i%w-x) if horizontal
                       else abs(i%w-x)*mask.shape[0]+abs(i//w-y))
    points.extend(available[:count-len(points)])
    return points


def _place_supply(rng, available, distance, center, quota, count, style, detour=False):
    if count <= quota:
        return _pick(rng, available, count, style) if available.sum() >= count else None
    y, x = np.indices(available.shape)
    r, c = center
    direct = np.maximum(abs(y-r)-1, 0) + np.maximum(abs(x-c)-1, 0)
    extra = count - quota
    for flat in rng.permutation(np.flatnonzero(available)):
        flat = int(flat)
        depth = int(distance.flat[flat])
        near = available & (distance <= depth)
        far = available & (distance > depth)
        choices = far & (direct < direct.flat[flat]) if detour else far
        if near.sum() >= quota and far.sum() >= extra and choices.any():
            b = int(rng.choice(np.flatnonzero(choices)))
            return (_pick(rng, near, quota, style, [flat])
                    + _pick(rng, far, extra, style, [b]))
    return None


def _new_pair(rng, radius, parameters, obligation=None):
    full, partial = obligation == 'full', obligation == 'partial'
    for attempt in range(32):
        carrier = parameters['palette'][0] if parameters['fixed_palette'] else int(rng.choice(GOODS))
        palette = ([v for v in parameters['palette'] if v != carrier] if parameters['fixed_palette']
                   else list(map(int, rng.permutation([v for v in GOODS if v != carrier]))))
        samples, primary, partial_count = _demand(rng, palette, full or partial)
        diameter = 2*radius + 3
        margin = 4 if full else 0
        minimum = max(15, diameter + 2*margin)
        h, w = map(int, rng.integers(minimum, 31, size=2))
        r = int(rng.integers(radius+1+margin, h-radius-1-margin))
        c = int(rng.integers(radius+1+margin, w-radius-1-margin))
        a = np.zeros((h, w), dtype=np.int8)
        hardware = np.zeros(a.shape, dtype=bool)
        a[r-1:r+2, c-1:c+2] = carrier
        a[r, c] = 9
        hardware[r-1:r+2, c-1:c+2] = True
        for (dr, dc), sample in zip(DIRS, samples):
            tile = np.full((3, 3), carrier, dtype=np.int8)
            inward = (1-dr, 1-dc)
            positions = [i for i in range(9) if i != inward[0]*3+inward[1]]
            rng.shuffle(positions)
            extra = int(rng.integers(parameters['richness']+1))
            # Mix isolated marks, short lines and compact patches.
            if rng.random() < .5:
                anchor = positions[0]
                positions.sort(key=lambda i: abs(i//3-anchor//3)+abs(i%3-anchor%3))
            colors = [v for v in palette if not full or v != sample]
            for i in positions[:extra]:
                tile.flat[i] = int(rng.choice(colors))
            tile[inward] = sample
            y, x = r+radius*dr-1, c+radius*dc-1
            a[y:y+3, x:x+3] = tile
            hardware[y:y+3, x:x+3] = True
        kind = 'detour' if full else str(rng.choice(parameters['modes']))
        if attempt >= 24:
            kind = 'detour' if full else 'open'
        wall = _walls(rng, a.shape, (r, c), radius, hardware, kind)
        a[wall] = 5
        _, info = _analyze(a)
        distance = _distances(info['blocked'], (r, c))
        active = (~hardware) & (distance >= 0)
        inactive = (~hardware) & ~wall & (distance < 0)
        if full and not inactive.any():
            continue
        used = np.zeros(a.shape, dtype=bool)
        style = str(rng.choice(['dispersed', 'compact', 'line']))
        ordered_colors = [primary] + sorted(set(samples) - {primary})
        valid = True
        for color in ordered_colors:
            quota = samples.count(color)
            if full:
                count = quota + int(rng.integers(1, 7)) if color == primary else quota
            elif partial:
                count = partial_count if color == primary else quota
            else:
                count = int(rng.integers(quota+5))
            cells = _place_supply(rng, active & ~used, distance, (r, c), quota,
                                  count, style, detour=full and color == primary)
            if cells is None:
                valid = False
                break
            a.flat[cells] = color
            used.flat[cells] = True
        if not valid:
            continue
        if inactive.any():
            amount = int(rng.integers(1 if full else 0, min(10, int(inactive.sum()))+1))
            cells = _pick(rng, inactive, amount, style)
            for i in cells:
                a.flat[i] = primary if full else int(rng.choice(palette))
        unused_colors = sorted(set(palette) - set(samples))
        if unused_colors and (partial or rng.random() < .65):
            amount = int(rng.integers(1, 7))
            free = active & ~used
            if free.sum() < amount:
                continue
            cells = _pick(rng, free, amount, style)
            for i in cells:
                a.flat[i] = int(rng.choice(unused_colors))
        # The rule fixes the footprint origin. Do not make the sampled queries
        # depend on distinguishing it from the plausible center-origin variant.
        selection = _selection(a, info, distance)
        try:
            center_selection = _selection(a, info, _distances(info['blocked'], (r, c),
                                                             marker_origin=True))
        except AssertionError:
            continue
        if any(not np.array_equal(selection[v], center_selection[v]) for v in selection):
            continue
        pair = {'input': a, 'output': solve(a)}
        feature = _audit_pair(pair)
        if full and not feature['full_witness']:
            continue
        if partial and not feature['partial_witness']:
            continue
        return pair
    raise AssertionError("Bounded scene construction failed.")


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n, m = _counts(num_train, num_test)
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("Seed must be a nonnegative integer.")
    rng = np.random.default_rng(seed)
    parameters = dict(palette=list(map(int, rng.permutation(GOODS))),
                      fixed_palette=bool(rng.random() < .25),
                      richness=int(rng.integers(1, 4)),
                      modes=(['open', 'open', 'segments', 'rooms', 'partitions']
                             if rng.random() < .3 else
                             ['open', 'segments', 'rooms', 'partitions', 'maze']))
    radii = list(map(int, rng.permutation([4, 5])))
    requests = [(radii[0], 'full'), (radii[1], 'partial')]
    requests.extend((int(rng.integers(4, 6)), None) for _ in range(n-2))
    rng.shuffle(requests)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, specs in (('train', requests),
                         ('test', [(int(rng.integers(6, 8)), None) for _ in range(m)])):
        for radius, obligation in specs:
            for _ in range(16):
                pair = _new_pair(rng, radius, parameters, obligation)
                key = _key(pair['input'])
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise AssertionError("Bounded uniqueness retries exhausted.")
    validate_generated_puzzle(puzzle)
    return puzzle


def _audit_pair(pair):
    """Audit returned cells, using wave distances and scalar slot checks."""
    _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "Invalid pair schema.")
    a, info = _analyze(pair['input'])
    b = _grid(pair['output'])
    _require(a.shape == b.shape, "Output dimensions changed.")
    distance = _distances(info['blocked'], info['center'], wave=True)
    selected = _selection(a, info, distance)
    alternate = _selection(a, info, _distances(info['blocked'], info['center'],
                                              wave=True, marker_origin=True))
    _require(all(np.array_equal(selected[v], alternate[v]) for v in selected),
             "Generated example introduces an unsupported distance-origin distinction.")
    r, c = info['center']
    changed_field = info['field'] & (a != b)
    union = np.zeros(a.shape, dtype=bool)
    for mask in selected.values():
        union |= mask
    _require(np.array_equal(changed_field, union), "Incorrect field removal/preservation.")
    _require(np.all(b[union] == 0), "Selected cells must be restored to floor.")
    _require(np.array_equal(a[info['tiles']], b[info['tiles']]), "Source keys changed.")
    _require(b[r, c] == 9, "Reserved marker changed.")
    filled, partial_prefix, excess, unreachable, reversal = 0, False, 0, 0, False
    rr, cc = np.indices(a.shape)
    direct = np.maximum(abs(rr-r)-1, 0) + np.maximum(abs(cc-c)-1, 0)
    quotas, loads, selected_counts = [], [], []
    for color in sorted(selected):
        mask = selected[color]
        slots = [i for i, value in enumerate(info['samples']) if value == color]
        k = int(mask.sum())
        filled += k
        reach = info['field'] & (a == color) & (distance >= 0)
        lost = info['field'] & (a == color) & (distance < 0)
        retained = reach & ~mask
        quotas.append(len(slots))
        loads.append(int(reach.sum()))
        selected_counts.append(k)
        excess += int(retained.sum())
        unreachable += int(lost.sum())
        _require(k == min(len(slots), int(reach.sum())), "Color conservation/capacity failed.")
        for j, i in enumerate(slots):
            dr, dc = DIRS[i]
            want = color if j < k else info['carrier']
            _require(int(b[r+dr, c+dc]) == want, "Incorrect color-specific prefix.")
        column_slots = [i for i in COL_ORDER if i in slots]
        partial_prefix |= 0 < k < len(slots) and set(slots[:k]) != set(column_slots[:k])
        if mask.any() and retained.any():
            reversal |= int(direct[mask].max()) > int(direct[retained].min())
    unique_samples = all(np.count_nonzero(tile == color) == 1
                         for tile, color in zip(info['views'], info['samples']))
    samples = info['samples']
    nongrouped = (len({samples[i] for i in (0, 2, 5, 7)}) > 1
                 or len({samples[i] for i in (1, 3, 4, 6)}) > 1)
    unrequested = (info['field'] & (a != 0) & (a != 5)
                   & ~np.isin(a, list(set(samples))) & (distance >= 0))
    realized_slots = [i for i, (dr, dc) in enumerate(DIRS)
                      if b[r+dr, c+dc] != info['carrier']]
    skips_global_prefix = realized_slots != list(range(filled))
    # Require another color to saturate in the partial witness, establishing
    # independent color quotas rather than one global collection prefix.
    some_saturated = any(q == k for q, k in zip(quotas, selected_counts))
    normalized, labels = [], {}
    for v in a.flat:
        v = int(v)
        normalized.append(labels.setdefault(v, len(labels)))
    demand_labels, demand_pattern = {}, []
    for v in samples:
        demand_pattern.append(demand_labels.setdefault(v, len(demand_labels)))
    return dict(shape=a.shape, radius=info['radius'], center=(r, c),
                carrier=info['carrier'], palette=tuple(map(int, np.unique(a))),
                demand_pattern=tuple(demand_pattern), colors=len(set(samples)),
                quotas=tuple(quotas), reachable=tuple(loads), selected=filled,
                excess=excess, unreachable=unreachable,
                unrequested=int(unrequested.sum()), path_reversal=bool(reversal),
                wall_count=int(np.count_nonzero(a == 5)),
                max_path=int(distance.max()),
                decoration_counts=tuple(8-int(np.count_nonzero(t == info['carrier']))
                                        for t in info['views']),
                repeated_samples=sum(np.count_nonzero(t == v) > 1
                                     for t, v in zip(info['views'], samples)),
                full_witness=bool(filled == 8 and unique_samples and nongrouped
                                  and excess and unreachable and reversal),
                partial_witness=bool(partial_prefix and some_saturated and unrequested.any()
                                     and skips_global_prefix),
                structure=hashlib.sha256(bytes(normalized)).hexdigest())


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "Invalid puzzle keys.")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists.")
    n, m = len(puzzle['train']), len(puzzle['test'])
    _counts(n, m)
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            feature = _audit_pair(pair)
            key = _key(pair['input'])
            _require(key not in seen, "Duplicate input within puzzle.")
            seen.add(key)
            features[split].append(feature)
    train = features['train']
    _require(any(f['full_witness'] for f in train), "Missing full/path/sample training witness.")
    _require(any(f['partial_witness'] for f in train), "Missing masked-prefix training witness.")
    _require({f['radius'] for f in train} == {4, 5}, "Training must demonstrate two spacings.")
    _require(all(f['radius'] in (6, 7) for f in features['test']), "Missing test spacing shift.")


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _expect_error(call, error=AssertionError):
    try:
        call()
    except error:
        return
    raise AssertionError("Expected rejection did not occur.")


def self_test():
    """Bounded 42-seed semantic/diversity audit; no files or retained puzzle bank."""
    signatures, structures, patterns, shapes, palettes = set(), set(), set(), set(), set()
    radii, carriers, loads, wall_counts, paths = set(), set(), set(), set(), set()
    excess_cases = unreachable_cases = repeated_cases = open_cases = maze_cases = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(_signature(p))
        for split in ('train', 'test'):
            for pair in p[split]:
                a = pair['input']
                before = a.copy()
                prediction = solve(a)
                _require(np.array_equal(a, before), "solve mutated input.")
                _require(np.array_equal(prediction, pair['output']), "solve/audit mismatch.")
                _, info = _analyze(a)
                _require(np.array_equal(_distances(info['blocked'], info['center']),
                                       _distances(info['blocked'], info['center'], wave=True)),
                         "BFS/wavefront mismatch.")
                f = _audit_pair(pair)
                structures.add(f['structure'])
                patterns.add(f['demand_pattern'])
                shapes.add(f['shape'])
                palettes.add(f['palette'])
                radii.add(f['radius'])
                carriers.add(f['carrier'])
                loads.add(f['selected'])
                wall_counts.add(f['wall_count'])
                paths.add(f['max_path'])
                excess_cases += f['excess'] > 0
                unreachable_cases += f['unreachable'] > 0
                repeated_cases += f['repeated_samples'] > 0
                open_cases += f['wall_count'] == 0
                maze_cases += f['wall_count'] > a.size//3
    _require(len(signatures) == 42 and len(structures) >= 180, "Structural seed diversity collapsed.")
    _require(len(patterns) >= 50 and len(shapes) >= 60, "Demand/layout diversity collapsed.")
    _require(len(palettes) >= 20 and carriers == set(GOODS), "Role diversity collapsed.")
    _require(radii == {4, 5, 6, 7} and len(loads) >= 7, "Spacing/load coverage collapsed.")
    _require(len(wall_counts) >= 60 and len(paths) >= 25, "Topology variation collapsed.")
    _require(min(excess_cases, unreachable_cases, repeated_cases, open_cases) >= 8,
             "Important family cases absent.")
    _require(maze_cases > 0, "Dense traversable geometry absent.")
    _require(_signature(generate_puzzle(701, 6, 3)) == _signature(generate_puzzle(701, 6, 3)),
             "Seed not deterministic.")
    for n, m in ((2, 1), (10, 1), (10, 4), (5, 4)):
        p = generate_puzzle(800+n+m, n, m)
        _require((len(p['train']), len(p['test'])) == (n, m), "Explicit counts ignored.")
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (4.0, 1), (True, 1), (4, False), (None, 1)):
        _expect_error(lambda n=n, m=m: generate_puzzle(0, n, m), ValueError)
    p = generate_puzzle(321, 2, 1)
    pair = p['test'][0]
    # Output corruption must be detected even if only a structural cell changes.
    original = pair['output'].copy()
    _, info = _analyze(pair['input'])
    r, c = info['center']
    pair['output'][r, c] = 0
    _expect_error(lambda: validate_generated_puzzle(p))
    pair['output'] = original
    # Strict frontier ties are a rule constraint, not an implicit queue order.
    for color in set(info['samples']):
        q = info['samples'].count(color)
        d = _distances(info['blocked'], (r, c))
        available = info['field'] & (d >= 0)
        groups = [np.flatnonzero(available & (d == v)) for v in np.unique(d[available])]
        group = next((g for g in groups if len(g) > q), None)
        if group is not None:
            tied = pair['input'].copy()
            tied[info['field'] & (tied == color)] = 0
            tied.flat[group[:q+1]] = color
            _expect_error(lambda: solve(tied))
            break
    else:
        raise AssertionError("No frontier-tie adversary constructed.")
    # The input-only solver supports ungenerated spacings that fit the same rule.
    rng = np.random.default_rng(195)
    parameters = dict(palette=list(GOODS), fixed_palette=False, richness=3, modes=['open'])
    for radius in (8, 10, 12):
        pair = _new_pair(rng, radius, parameters)
        _require(np.array_equal(solve(pair['input']), pair['output']), "Larger legal radius failed.")


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds; bounded diagnostics)')
