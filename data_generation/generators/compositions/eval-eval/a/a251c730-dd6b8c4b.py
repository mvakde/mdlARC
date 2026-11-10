"""Deploy learned collector molds, then gather through the resulting maze.

Only standard-library modules and NumPy are used. The input supplies all roles,
templates, intake sites and capacities; solve has no access to generation state.
"""
from collections import deque

import numpy as np


STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= int(num_train) <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= int(num_test) <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("Expected a 1..30 integer grid with colors 0..9")
    return a


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells = [start]
        for r, c in cells:
            for dr, dc in STEPS:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    cells.append(p)
        result.append([(int(r), int(c)) for r, c in cells])
    return result


def _dictionary(region, background):
    templates, repeats, slot = {}, {}, None
    components = _components(region != background)
    if not components:
        raise ValueError("Empty dictionary")
    for cells in components:
        values, counts = np.unique([region[p] for p in cells], return_counts=True)
        if len(values) != 2 or sorted(counts.tolist())[0] != 1 or len(cells) < 4:
            raise ValueError("A mold needs one key and at least three slot cells")
        key = int(values[np.argmin(counts)])
        decor = int(values[np.argmax(counts)])
        if slot is not None and decor != slot:
            raise ValueError("Dictionary slot color is inconsistent")
        slot = decor
        ar, ac = next(p for p in cells if int(region[p]) == key)
        shape = tuple(sorted((r-ar, c-ac) for r, c in cells))
        if key in templates and templates[key] != shape:
            raise ValueError("Conflicting definitions of one key")
        templates[key] = shape
        repeats[key] = repeats.get(key, 0) + 1
    if slot in templates:
        raise ValueError("A key cannot also be the slot color")
    return templates, int(slot), repeats


def _parse(input_grid):
    grid = _grid(input_grid)
    background = int(grid[0, 0])
    dividers = []
    for axis in (0, 1):
        view = grid if axis == 0 else grid.T
        for i in range(1, view.shape[0]-1):
            if np.all(view[i] == view[i, 0]) and int(view[i, 0]) != background:
                dividers.append((axis, i, int(view[i, 0])))
    if len(dividers) != 1:
        raise ValueError("Input needs exactly one full straight divider")
    axis, index, wall_color = dividers[0]
    if axis == 0:
        regions = (grid[:index], grid[index+1:])
    else:
        regions = (grid[:, :index], grid[:, index+1:])
    choices = []
    for ri in (0, 1):
        reference, field = regions[ri], regions[1-ri]
        try:
            templates, slot, repeats = _dictionary(reference, background)
            known = {background, wall_color, slot, *templates}
            if len(known) != len(templates)+3 or slot in np.unique(field):
                continue
            payloads = set(int(x) for x in np.unique(field)) - known
            if len(payloads) != 1:
                continue
            payload = next(iter(payloads))
            anchors = [(int(r), int(c), int(field[r, c]))
                       for r, c in zip(*np.nonzero(np.isin(field, list(templates))))]
            if not anchors:
                continue
            footprint_mask = np.zeros(field.shape, dtype=bool)
            collectors = []
            for ar, ac, key in anchors:
                footprint = [(ar+dr, ac+dc) for dr, dc in templates[key]]
                slots = sorted(p for p in footprint if p != (ar, ac))
                for r, c in footprint:
                    if not (0 <= r < field.shape[0] and 0 <= c < field.shape[1]):
                        raise ValueError("Clipped deployment")
                    if footprint_mask[r, c]:
                        raise ValueError("Overlapping deployments")
                    if (r, c) != (ar, ac) and int(field[r, c]) not in (background, wall_color):
                        raise ValueError("A slot would overwrite payload or another key")
                    footprint_mask[r, c] = True
                collectors.append(dict(anchor=(ar, ac), key=key, footprint=footprint, slots=slots))
            choices.append(dict(field=field, reference=reference, templates=templates,
                                repetitions=repeats, background=background, wall=wall_color,
                                slot=slot, payload=payload, collectors=collectors,
                                footprint=footprint_mask, axis=axis, reference_first=(ri == 0)))
        except ValueError:
            continue
    if len(choices) != 1:
        raise ValueError("Dictionary and field must be uniquely identifiable")
    return choices[0]


def _distances(wall, starts):
    h, w = wall.shape
    distance = np.full((h, w), -1, dtype=np.int16)
    queue = deque()
    for p in starts:
        if wall[p]:
            raise ValueError("Intake site is blocked")
        distance[p] = 0
        queue.append(p)
    while queue:
        r, c = queue.popleft()
        nd = int(distance[r, c])+1
        for dr, dc in STEPS:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and not wall[nr, nc] and distance[nr, nc] < 0:
                distance[nr, nc] = nd
                queue.append((nr, nc))
    return distance


def _ownership(wall, collectors):
    distances = np.stack([_distances(wall, c['footprint']) for c in collectors])
    comparable = np.where(distances < 0, 32767, distances)
    best = comparable.min(axis=0)
    unique = (comparable == best).sum(axis=0) == 1
    owner = comparable.argmin(axis=0).astype(np.int16)
    owner[best == 32767] = -1
    owner[(best < 32767) & ~unique] = -2
    return distances, owner


def _direct(shape, starts):
    rr, cc = np.indices(shape)
    return np.minimum.reduce([np.abs(rr-r)+np.abs(cc-c) for r, c in starts])


def _transform(info):
    field, collectors = info['field'], info['collectors']
    wall = field == info['wall']
    wall = wall.copy()
    wall[info['footprint']] = False
    distances, owner = _ownership(wall, collectors)
    payload = field == info['payload']
    if np.any(payload & (owner == -2)):
        raise ValueError("A payload cell has tied nearest owners")
    result = field.copy()
    selected = np.zeros(field.shape, dtype=bool)
    loads, owned_counts, gaps = [], [], []
    for i, collector in enumerate(collectors):
        for p in collector['slots']:
            result[p] = info['slot']
        owned = payload & (owner == i)
        cells = list(zip(*np.nonzero(owned)))
        cells.sort(key=lambda p: int(distances[i][p]))
        capacity = len(collector['slots'])
        n = min(capacity, len(cells))
        gap = 0
        if len(cells) > capacity:
            gap = int(distances[i][cells[capacity]]) - int(distances[i][cells[capacity-1]])
            if gap <= 0:
                raise ValueError("Capacity frontier has a distance tie")
        for p in cells[:n]:
            selected[p] = True
        for p in collector['slots'][:n]:
            result[p] = info['payload']
        loads.append(n)
        owned_counts.append(len(cells))
        gaps.append(gap)
    result[selected] = info['background']
    info.update(distances=distances, owner=owner, selected=selected,
                loads=loads, owned_counts=owned_counts, gaps=gaps, final_walls=wall)
    return result


def _anchor_origin_ablation(info):
    """Same deployed walls and capacities, but incorrectly use keys as intakes.

    Report a distinguishing answer only when this alternative is itself free of
    ownership/frontier ties. This avoids pretending that an ambiguous shortcut
    makes a particular prediction.
    """
    collectors = [dict(footprint=[c['anchor']]) for c in info['collectors']]
    distances, owner = _ownership(info['final_walls'], collectors)
    payload = info['field'] == info['payload']
    if np.any(payload & (owner == -2)):
        return False
    selected = np.zeros(payload.shape, dtype=bool)
    loads = []
    for i, collector in enumerate(info['collectors']):
        cells = list(zip(*np.nonzero(payload & (owner == i))))
        cells.sort(key=lambda p: int(distances[i][p]))
        capacity = len(collector['slots'])
        if len(cells) > capacity and distances[i][cells[capacity]] == distances[i][cells[capacity-1]]:
            return False
        loads.append(min(capacity, len(cells)))
        for p in cells[:capacity]:
            selected[p] = True
    return loads != info['loads'] or not np.array_equal(selected, info['selected'])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using only the visible divider, dictionary, field and payload."""
    return _transform(_parse(input_grid)).astype(np.int8)


def _random_shape(rng, gate=False, partial=False):
    for _ in range(150):
        h, w = int(rng.integers(2, 6)), int(rng.integers(2, 6))
        size = int(rng.integers(5 if gate or partial else 4, min(11, h*w)+1)) if h*w >= 5 or not (gate or partial) else 4
        if (gate or partial) and size < 5:
            continue
        cells = {(int(rng.integers(h)), int(rng.integers(w)))}
        style = int(rng.integers(3))
        while len(cells) < size:
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in STEPS
                               if 0 <= r+dr < h and 0 <= c+dc < w} - cells)
            degrees = np.array([sum((r+dr, c+dc) in cells for dr, dc in STEPS) for r, c in frontier])
            weights = np.ones(len(frontier)) if style == 0 else degrees.astype(float)**(2 if style == 1 else -2)
            cells.add(frontier[int(rng.choice(len(frontier), p=weights/weights.sum()))])
        anchor_choices = sorted(cells)
        if gate:
            anchor_choices = [p for p in anchor_choices
                              if max(abs(r-p[0]) for r, c in cells) >= 2
                              or max(abs(c-p[1]) for r, c in cells) >= 2]
        if not anchor_choices:
            continue
        ar, ac = _pick(rng, anchor_choices)
        shape = tuple(sorted((r-ar, c-ac) for r, c in cells))
        slots = [p for p in shape if p != (0, 0)]
        if (gate or partial) and not _partial_counts(slots):
            continue
        if gate:
            rows, cols = zip(*shape)
            if (max(rows)-min(rows)+1)*(max(cols)-min(cols)+1) == size:
                continue
        return shape
    raise AssertionError("Could not construct a readable mold")


def _partial_counts(slots):
    rows = sorted(slots)
    columns = sorted(slots, key=lambda p: (p[1], p[0]))
    return [n for n in range(1, min(4, len(slots)-1)+1) if set(rows[:n]) != set(columns[:n])]


def _bounds(shape):
    rows, cols = zip(*shape)
    return min(rows), min(cols), max(rows)-min(rows)+1, max(cols)-min(cols)+1


def _shelf(templates, ids, width):
    y, x, shelf_h = 1, 1, 0
    places = []
    for key in ids:
        lr, lc, h, w = _bounds(templates[key])
        if w+2 > width:
            return None
        if x+w > width-1:
            y += shelf_h+1
            x, shelf_h = 1, 0
        places.append((y-lr, x-lc, key))
        x += w+1
        shelf_h = max(shelf_h, h)
    return y+shelf_h+1, places


def _layout(rng, templates, ids, axis):
    # Pack the same unrotated shapes in either a horizontal or vertical strip.
    options = []
    if axis == 0:
        for width in range(17, 31):
            packed = _shelf(templates, ids, width)
            if packed and packed[0] <= 15:
                options.append(((packed[0], width), packed[1]))
    else:
        for width in range(5, 16):
            packed = _shelf(templates, ids, width)
            if packed and packed[0] <= 30:
                for height in range(max(17, packed[0]), 31):
                    options.append(((height, width), packed[1]))
    if not options:
        return None
    ref_shape, ref_places = _pick(rng, options)
    extent = ref_shape[axis]
    maximum = 30-extent-1
    if maximum < 13:
        return None
    field_shape = list(ref_shape)
    field_shape[axis] = int(rng.integers(13, maximum+1))
    return ref_shape, tuple(field_shape), ref_places


def _placements(rng, shape, templates, ids):
    h, w = shape
    for _ in range(18):
        occupied = np.zeros(shape, dtype=bool)
        places = []
        for key in ids:
            lr, lc, bh, bw = _bounds(templates[key])
            candidates = [(r-lr, c-lc) for r in range(1, h-bh) for c in range(1, w-bw)]
            rng.shuffle(candidates)
            for ar, ac in candidates:
                footprint = [(ar+dr, ac+dc) for dr, dc in templates[key]]
                if all(not occupied[r, c] and all(not occupied[r+dr, c+dc]
                       for dr, dc in STEPS if 0 <= r+dr < h and 0 <= c+dc < w)
                       for r, c in footprint):
                    for p in footprint:
                        occupied[p] = True
                    places.append((ar, ac, key))
                    break
            else:
                break
        if len(places) == len(ids):
            return places
    return None


def _rectangle(wall, r0, c0, r1, c1):
    wall[r0, c0:c1+1] = True
    wall[r1, c0:c1+1] = True
    wall[r0:r1+1, c0] = True
    wall[r0:r1+1, c1] = True


def _wall_geometry(rng, shape, kind):
    h, w = shape
    wall = np.zeros(shape, dtype=bool)
    if kind == 'open':
        return wall
    if kind == 'maze':
        wall[1:-1, 1:-1] = True
        nodes = [(r, c) for r in range(2, h-2, 2) for c in range(2, w-2, 2)]
        start = _pick(rng, nodes)
        stack, visited = [start], {start}
        wall[start] = False
        while stack:
            r, c = stack[-1]
            choices = [(r+2*dr, c+2*dc) for dr, dc in STEPS
                       if 2 <= r+2*dr < h-2 and 2 <= c+2*dc < w-2 and (r+2*dr, c+2*dc) not in visited]
            if not choices:
                stack.pop()
                continue
            nr, nc = _pick(rng, choices)
            wall[(r+nr)//2, (c+nc)//2] = wall[nr, nc] = False
            visited.add((nr, nc))
            stack.append((nr, nc))
        for _ in range(int(rng.integers(1, 5))):
            r, c = int(rng.integers(2, h-3)), int(rng.integers(2, w-3))
            wall[r:r+int(rng.integers(2, 5)), c:c+int(rng.integers(2, 5))] = False
        # One or more connections from the maze to the exterior background rim.
        for _ in range(int(rng.integers(1, 4))):
            r, c = _pick(rng, nodes)
            if rng.random() < .5:
                wall[1:r+1, c] = False
            else:
                wall[r, c:w-1] = False
    elif kind == 'partitions':
        rooms = [(1, 1, h-2, w-2)]
        for _ in range(int(rng.integers(3, 10))):
            if not rooms:
                break
            r0, c0, r1, c1 = rooms.pop(int(rng.integers(len(rooms))))
            horizontal = rng.random() < .5
            if (horizontal and r1-r0 >= 5) or (c1-c0 < 5 and r1-r0 >= 5):
                r = int(rng.integers(r0+2, r1-1))
                wall[r, c0:c1+1] = True
                if rng.random() < .85:
                    c = int(rng.integers(c0, c1+1))
                    wall[r, c:c+int(rng.integers(1, 3))] = False
                rooms.extend([(r0, c0, r-1, c1), (r+1, c0, r1, c1)])
            elif c1-c0 >= 5:
                c = int(rng.integers(c0+2, c1-1))
                wall[r0:r1+1, c] = True
                if rng.random() < .85:
                    r = int(rng.integers(r0, r1+1))
                    wall[r:r+int(rng.integers(1, 3)), c] = False
                rooms.extend([(r0, c0, r1, c-1), (r0, c+1, r1, c1)])
    for _ in range(int(rng.integers(1, 5)) if kind != 'rails' else int(rng.integers(2, 7))):
        r0, c0 = int(rng.integers(1, h-4)), int(rng.integers(1, w-4))
        r1, c1 = int(rng.integers(r0+2, h-1)), int(rng.integers(c0+2, w-1))
        primitive = int(rng.integers(4))
        if kind == 'rings' or primitive == 0:
            _rectangle(wall, r0, c0, r1, c1)
            for _ in range(int(rng.integers(0, 4))):
                side = int(rng.integers(4))
                if side < 2:
                    c = int(rng.integers(c0+1, c1))
                    wall[r0 if side == 0 else r1, c] = False
                else:
                    r = int(rng.integers(r0+1, r1))
                    wall[r, c0 if side == 2 else c1] = False
        elif primitive == 1:
            wall[r0:r1+1, c0] = True
            wall[r0, c0:c1+1] = True
            wall[r0:r1+1, c1] = True
        elif primitive == 2:
            wall[r0, c0:c1+1] = True
        else:
            wall[r0:r1+1, c0] = True
    wall[0] = wall[-1] = False
    wall[:, 0] = wall[:, -1] = False
    return wall


def _gate_ring(rng, wall, collector):
    """Close a rectangle whose boundary the first mold will perforate."""
    h, w = wall.shape
    ar, ac = collector['anchor']
    footprint = collector['footprint']
    candidates = []
    for dr, dc in STEPS:
        projections = [(r-ar)*dr+(c-ac)*dc for r, c in footprint]
        if max(projections) < 2 or min(projections) < 0:
            continue
        if dr == 1:
            bounds = (ar+1, max(1, min(c for r, c in footprint)-2),
                      max(r for r, c in footprint)+int(rng.integers(2, 5)),
                      min(w-2, max(c for r, c in footprint)+2))
        elif dr == -1:
            bounds = (min(r for r, c in footprint)-int(rng.integers(2, 5)),
                      max(1, min(c for r, c in footprint)-2), ar-1,
                      min(w-2, max(c for r, c in footprint)+2))
        elif dc == 1:
            bounds = (max(1, min(r for r, c in footprint)-2), ac+1,
                      min(h-2, max(r for r, c in footprint)+2),
                      max(c for r, c in footprint)+int(rng.integers(2, 5)))
        else:
            bounds = (max(1, min(r for r, c in footprint)-2),
                      min(c for r, c in footprint)-int(rng.integers(2, 5)),
                      min(h-2, max(r for r, c in footprint)+2), ac-1)
        r0, c0, r1, c1 = bounds
        if 1 <= r0 < r1 < h-1 and 1 <= c0 < c1 < w-1 and r1-r0 >= 2 and c1-c0 >= 2:
            candidates.append(bounds)
    if not candidates:
        return False
    bounds = _pick(rng, candidates)
    r0, c0, r1, c1 = bounds
    wall[r0:r1+1, c0:c1+1] = False
    _rectangle(wall, *bounds)
    return True


def _pocket(rng, wall, footprint):
    candidates = []
    h, w = wall.shape
    for bh in range(3, 6):
        for bw in range(3, 6):
            for r in range(1, h-bh):
                for c in range(1, w-bw):
                    if not footprint[max(0, r-1):r+bh+1, max(0, c-1):c+bw+1].any():
                        candidates.append((r, c, bh, bw))
    if not candidates:
        return False
    r, c, bh, bw = _pick(rng, candidates)
    wall[r:r+bh, c:c+bw] = False
    _rectangle(wall, r, c, r+bh-1, c+bw-1)
    return True


def _pick_cells(rng, mask, amount, style, forced=()):
    available = set(zip(*np.nonzero(mask)))
    chosen = list(dict.fromkeys(forced))
    if not set(chosen) <= available or not len(chosen) <= amount <= len(available):
        return None
    available.difference_update(chosen)
    center = chosen[0] if chosen else _pick(rng, sorted(available)) if available else (0, 0)
    horizontal = bool(rng.integers(2))
    while len(chosen) < amount:
        cells = sorted(available)
        if style == 'dispersed':
            cell = _pick(rng, cells)
        else:
            ar, ac = center
            if style == 'compact':
                score = lambda p: (p[0]-ar)**2+(p[1]-ac)**2
            elif horizontal:
                score = lambda p: abs(p[0]-ar)*mask.shape[1]+abs(p[1]-ac)
            else:
                score = lambda p: abs(p[1]-ac)*mask.shape[0]+abs(p[0]-ar)
            best = min(score(p) for p in cells)
            cell = _pick(rng, [p for p in cells if score(p) == best])
        chosen.append(cell)
        available.remove(cell)
    return chosen


def _excess_supply(rng, mask, distance, capacity, extra, direct, reversal):
    # Strict frontier is a property of placed supply, not of all empty field sites.
    depths = [int(x) for x in np.unique(distance[mask])]
    rng.shuffle(depths)
    style = _pick(rng, ('dispersed', 'compact', 'line'))
    for depth in depths:
        near, far = mask & (distance <= depth), mask & (distance > depth)
        if near.sum() < capacity or far.sum() < extra:
            continue
        forced_near, forced_far = [], []
        if reversal:
            a = int(direct[near].max())
            b = int(direct[far].min())
            if a <= b:
                continue
            forced_near = [_pick(rng, list(zip(*np.nonzero(near & (direct == a)))))]
            forced_far = [_pick(rng, list(zip(*np.nonzero(far & (direct == b)))))]
        selected = _pick_cells(rng, near, capacity, style, forced_near)
        retained = _pick_cells(rng, far, extra, style, forced_far)
        return selected+retained
    return None


def _assemble(rng, reference, field, wall_color, axis):
    first = bool(rng.integers(2))
    parts = (reference, field) if first else (field, reference)
    separator_shape = list(reference.shape)
    separator_shape[axis] = 1
    separator = np.full(separator_shape, wall_color, dtype=np.int8)
    return np.concatenate((parts[0], separator, parts[1]), axis=axis)


def _generate_pair(rng, config, purpose, test=False):
    for attempt in range(240):
        ncollect = 2 if purpose in ('gate', 'competition') else int(rng.integers(3, 5) if test else rng.integers(1, 3))
        ntypes = 1 if purpose == 'gate' else 2 if purpose == 'competition' else int(rng.integers(1, min(3 if test else 2, ncollect)+1))
        palette = rng.permutation(10) if config['vary_palette'] else config['palette']
        background, wall_color, slot, payload = [int(x) for x in palette[:4]]
        keys = [int(x) for x in palette[4:4+ntypes]]
        templates = {key: _random_shape(rng, gate=(purpose == 'gate'),
                       partial=(purpose == 'competition' and i == 1)) for i, key in enumerate(keys)}
        if purpose == 'competition' and len(templates[keys[0]])-1 == 9:
            # Saturation must visibly distinguish the learned capacity from B's
            # original fixed nine-slot counter, even with only two train pairs.
            continue
        # Put an extremal anchor on the gate mold, without changing its orientation.
        if purpose == 'gate':
            shape = templates[keys[0]]
            candidates = []
            for anchor in shape:
                shifted = tuple(sorted((r-anchor[0], c-anchor[1]) for r, c in shape))
                if any(min(r*dr+c*dc for r, c in shifted) == 0 and max(r*dr+c*dc for r, c in shifted) >= 2 for dr, dc in STEPS):
                    if _partial_counts([p for p in shifted if p != (0, 0)]):
                        candidates.append(shifted)
            if not candidates:
                continue
            templates[keys[0]] = _pick(rng, candidates)
        ref_ids = [key for key in keys for _ in range(2 if purpose == 'gate' else int(rng.integers(1, 3)))]
        rng.shuffle(ref_ids)
        axis = int(rng.integers(2)) if config['axis'] == 'mixed' else config['axis']
        geometry = _layout(rng, templates, ref_ids, axis)
        if geometry is None:
            continue
        ref_shape, field_shape, ref_places = geometry
        ids = keys + [_pick(rng, keys) for _ in range(ncollect-ntypes)]
        rng.shuffle(ids)
        if purpose == 'competition':
            ids = keys  # Supply obligations refer to collector identities, not display order.
        places = _placements(rng, field_shape, templates, ids)
        if places is None:
            continue
        collectors, footprint = [], np.zeros(field_shape, dtype=bool)
        for ar, ac, key in places:
            cells = [(ar+dr, ac+dc) for dr, dc in templates[key]]
            slots = sorted(p for p in cells if p != (ar, ac))
            collectors.append(dict(anchor=(ar, ac), key=key, footprint=cells, slots=slots))
            for p in cells:
                footprint[p] = True
        kinds = ('open', 'rails', 'rings', 'partitions', 'maze')
        weights = [.2, .27, .18, .22, .13] if config['simple'] and not test else [.08, .18, .15, .29, .30]
        kind = 'open' if purpose == 'gate' else str(rng.choice(kinds, p=weights))
        wall = _wall_geometry(rng, field_shape, kind)
        if purpose == 'gate' and not _gate_ring(rng, wall, collectors[0]):
            continue
        if purpose == 'gate' or rng.random() < .5:
            if not _pocket(rng, wall, footprint) and purpose == 'gate':
                continue
        for collector in collectors:
            wall[collector['anchor']] = False
        deployed_walls = wall.copy()
        deployed_walls[footprint] = False
        distances, owner = _ownership(deployed_walls, collectors)
        available = ~footprint & ~deployed_walls
        available[0] = available[-1] = False
        available[:, 0] = available[:, -1] = False
        supplies = []
        possible = True
        for i, collector in enumerate(collectors):
            capacity = len(collector['slots'])
            mask = available & (owner == i)
            forced = []
            if purpose == 'gate':
                if i == 0:
                    old_distance = _distances(wall, [collector['anchor']])
                    opened = mask & (old_distance < 0)
                    if not opened.any():
                        possible = False
                        break
                    forced = [_pick(rng, list(zip(*np.nonzero(opened))))]
                counts = _partial_counts(collector['slots'])
                if i > 0:
                    counts = [n for n in counts if n != len(supplies)]
                count = 0 if i > 0 and (not counts or rng.random() < .25) else _pick(rng, counts)
                cells = _pick_cells(rng, mask, count, _pick(rng, ('dispersed', 'compact', 'line')), forced)
            elif purpose == 'competition' and i == 0:
                # Excess remains reachable by the other collector, which has spare capacity.
                mask &= distances[1] >= 0
                extra = int(rng.integers(1, 6))
                cells = _excess_supply(rng, mask, distances[i], capacity, extra,
                                       _direct(field_shape, collector['footprint']), True)
            elif purpose == 'competition':
                mask &= distances[0] >= 0
                count = _pick(rng, _partial_counts(collector['slots']))
                cells = _pick_cells(rng, mask, count, _pick(rng, ('dispersed', 'compact', 'line')))
            else:
                mode = rng.random()
                if mode < .32:
                    extra = int(rng.integers(1, 8))
                    cells = _excess_supply(rng, mask, distances[i], capacity, extra,
                                           _direct(field_shape, collector['footprint']), False)
                else:
                    count = 0 if mode < .44 else int(rng.integers(1, capacity+1))
                    cells = _pick_cells(rng, mask, count, _pick(rng, ('dispersed', 'compact', 'line')))
            if cells is None:
                possible = False
                break
            supplies.extend(cells)
        if not possible:
            continue
        inactive = available & (owner == -1)
        minimum = 1 if purpose == 'gate' else 0
        limit = min(12, int(inactive.sum()))
        if limit < minimum:
            continue
        distractors = _pick_cells(rng, inactive, int(rng.integers(minimum, limit+1)),
                                  _pick(rng, ('dispersed', 'compact', 'line')))
        if not supplies and not distractors:
            continue
        field = np.full(field_shape, background, dtype=np.int8)
        field[wall] = wall_color
        for p in supplies+distractors:
            field[p] = payload
        for ar, ac, key in places:
            field[ar, ac] = key
        reference = np.full(ref_shape, background, dtype=np.int8)
        for ar, ac, key in ref_places:
            for dr, dc in templates[key]:
                reference[ar+dr, ac+dc] = key if (dr, dc) == (0, 0) else slot
        input_grid = _assemble(rng, reference, field, wall_color, axis)
        construction = dict(field=field, collectors=collectors, footprint=footprint,
                            background=background, wall=wall_color, slot=slot, payload=payload)
        output = _transform(construction)
        try:
            parsed = _parse(input_grid)
            predicted = _transform(parsed)
            if not np.array_equal(predicted, output):
                continue
            pair = {'input': input_grid, 'output': output}
            f = pair_features(pair)
            if purpose == 'gate' and not f['gate_evidence']:
                continue
            if purpose == 'competition' and not f['competition_evidence']:
                continue
            return pair
        except ValueError:
            continue
    raise AssertionError(f"Bounded construction failed for {purpose}")


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(int(seed))
    config = dict(palette=rng.permutation(10), vary_palette=bool(rng.random() < .75),
                  axis=_pick(rng, (0, 1, 'mixed', 'mixed')), simple=bool(rng.random() < .35))
    purposes = ['gate', 'competition']+['varied']*(num_train-2)
    rng.shuffle(purposes)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, requests in (('train', purposes), ('test', ['varied']*num_test)):
        for purpose in requests:
            for _ in range(12):
                pair = _generate_pair(rng, config, purpose, split == 'test')
                signature = _signature(pair['input'])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("Duplicate inputs after bounded attempts")
    validate_generated_puzzle(puzzle)
    return puzzle


def pair_features(pair):
    """Recover family-specific facts from the actual input, not sampled flags."""
    info = _parse(pair['input'])
    expected = _transform(info)
    if not np.array_equal(expected, pair['output']):
        raise AssertionError("Wrong deployed collection label")
    field, payload = info['field'], info['field'] == info['payload']
    selected, owner, distances = info['selected'], info['owner'], info['distances']
    original_walls = field == info['wall']
    gate_selected = 0
    geometric_reversal, spare_excess, prefix_difference = False, False, False
    shared_reachable = int((payload & ((distances >= 0).sum(axis=0) > 1)).sum())
    for i, collector in enumerate(info['collectors']):
        old = _distances(original_walls, [collector['anchor']])
        own_selected = selected & (owner == i)
        gate_selected += int((own_selected & (old < 0)).sum())
        retained = payload & (owner == i) & ~selected
        direct = _direct(field.shape, collector['footprint'])
        geometric_reversal |= bool(own_selected.any() and retained.any()
                                   and direct[own_selected].max() > direct[retained].min())
        for j, other in enumerate(info['collectors']):
            if i != j and info['loads'][j] < len(other['slots']):
                spare_excess |= bool((retained & (distances[j] >= 0)).any())
        n = info['loads'][i]
        column_slots = sorted(collector['slots'], key=lambda p: (p[1], p[0]))
        prefix_difference |= set(collector['slots'][:n]) != set(column_slots[:n])
    capacities = tuple(len(c['slots']) for c in info['collectors'])
    partial = any(0 < n < cap for n, cap in zip(info['loads'], capacities))
    unreachable = int((payload & (owner == -1)).sum())
    nonsolid = any(_bounds(s)[2]*_bounds(s)[3] > len(s) for s in info['templates'].values())
    retained = int((payload & (owner >= 0) & ~selected).sum())
    repeated_key = len({c['key'] for c in info['collectors']}) < len(info['collectors'])
    same_key_different_loads = any(a['key'] == b['key'] and info['loads'][i] != info['loads'][j]
                                   for i, a in enumerate(info['collectors'])
                                   for j, b in enumerate(info['collectors']) if i < j)
    gate_evidence = (gate_selected > 0 and unreachable > 0 and partial
                     and prefix_difference and nonsolid and repeated_key and same_key_different_loads)
    footprint_origin_difference = _anchor_origin_ablation(info)
    nonnine_saturation = any(owned > cap and cap != 9
                             for owned, cap in zip(info['owned_counts'], capacities))
    competition_evidence = (len(capacities) >= 2 and partial and retained > 0
                            and shared_reachable > 0 and spare_excess and geometric_reversal
                            and nonnine_saturation and footprint_origin_difference)
    split_cluster = any(any(selected[p] for p in cells)
                        and any(owner[p] >= 0 and not selected[p] for p in cells)
                        for cells in _components(payload))
    positive = distances[distances >= 0]
    return dict(input_shape=tuple(np.asarray(pair['input']).shape), field_shape=field.shape,
                axis=info['axis'], reference_first=info['reference_first'],
                palette=(info['background'], info['wall'], info['slot'], info['payload']),
                types=len(info['templates']), collectors=len(capacities), capacities=capacities,
                loads=tuple(info['loads']), owned=tuple(info['owned_counts']),
                repetitions=tuple(info['repetitions'].values()),
                shapes=tuple(sorted(info['templates'].values())),
                selected=int(selected.sum()), excess=retained, unreachable=unreachable,
                opened_walls=int((original_walls & info['footprint']).sum()),
                wall_count=int(original_walls.sum()), max_path=int(positive.max()) if positive.size else 0,
                capacity_gaps=tuple(info['gaps']), shared_reachable=shared_reachable,
                gate_selected=gate_selected, partial=partial, nonsolid=nonsolid,
                geometric_reversal=geometric_reversal, spare_excess=spare_excess,
                footprint_origin_difference=bool(footprint_origin_difference),
                nonnine_saturation=bool(nonnine_saturation), split_cluster=bool(split_cluster),
                prefix_difference=prefix_difference, gate_evidence=bool(gate_evidence),
                repeated_key=bool(repeated_key),
                same_key_different_loads=bool(same_key_different_loads),
                competition_evidence=bool(competition_evidence))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "Invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, evidence = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "Invalid pair keys")
            a, b = _grid(pair['input']), _grid(pair['output'])
            key = _signature(a)
            _require(key not in seen, "Duplicate input")
            seen.add(key)
            info = _parse(a)
            expected = _transform(info)
            _require(np.array_equal(expected, b), "Output contradicts input-only rule")
            field, footprint, selected = info['field'], info['footprint'], info['selected']
            _require(b.shape == field.shape, "Incorrect field crop")
            _require(np.array_equal(b[~footprint & ~selected], field[~footprint & ~selected]),
                     "Unselected field cells changed")
            _require(all(b[c['anchor']] == c['key'] for c in info['collectors']), "Key overwritten")
            filled = sum(sum(int(b[p]) == info['payload'] for p in c['slots']) for c in info['collectors'])
            _require(filled == int(selected.sum()), "Collected cells and filled slots must balance")
            _require(int((field == info['payload']).sum()) == int((b == info['payload']).sum()),
                     "Payload conservation failed")
            _require(all(gap > 0 for gap, owned, c in zip(info['gaps'], info['owned_counts'], info['collectors'])
                         if owned > len(c['slots'])), "Non-strict capacity frontier")
            # The parser allows broader readable sizes; generation invariants are stricter.
            _require(1 <= len(info['templates']) <= 3 and 1 <= len(info['collectors']) <= 4,
                     "Generated type/collector range")
            _require(all(4 <= len(s) <= 11 and max(_bounds(s)[2:]) <= 5 for s in info['templates'].values()),
                     "Generated mold size range")
            _require(all(n in (1, 2) for n in info['repetitions'].values()), "Reference repetition range")
            indexed_footprints = np.full(field.shape, -1, dtype=np.int8)
            for i, collector in enumerate(info['collectors']):
                for p in collector['footprint']:
                    indexed_footprints[p] = i
            for i, collector in enumerate(info['collectors']):
                for r, c in collector['footprint']:
                    _require(all(indexed_footprints[r+dr, c+dc] in (-1, i)
                                 for dr, dc in STEPS if 0 <= r+dr < field.shape[0]
                                 and 0 <= c+dc < field.shape[1]), "Collector footprints must be separated")
            if split == 'train':
                evidence.append(pair_features(pair))
    _require(any(f['gate_evidence'] for f in evidence), "Training lacks wall-opening partial collection")
    _require(any(f['competition_evidence'] for f in evidence), "Training lacks path/ownership/excess evidence")


def _puzzle_signature(puzzle):
    return tuple(tuple((_signature(p['input']), _signature(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def self_test():
    """Bounded 42-seed mechanism audit, legal counts and negative regressions."""
    # A five-cell supply line is split at a three-slot learned capacity. These
    # expected cells are specified directly, without the labeling routine.
    reference = np.zeros((5, 14), dtype=np.int8)
    reference[1, 1:5] = [1, 2, 2, 2]
    field = np.zeros((7, 14), dtype=np.int8)
    field[3, 2] = 1
    field[3, 6:11] = 3
    separator = np.full((1, 14), 8, dtype=np.int8)
    expected = field.copy()
    expected[3, 3:6] = 3
    expected[3, 6:9] = 0
    line_input = np.concatenate((reference, separator, field))
    _require(np.array_equal(solve(line_input), expected), "Capacity must split a connected supply line")
    # Slots replace walls, but a wall in a bounding-box hole must stay intact.
    reference[:] = 0
    reference[1, 1] = 1
    reference[1:4, 2] = 2
    field[:] = 0
    field[2, 3] = 1
    field[3, 3:5] = 8
    field[2, 5] = 3
    expected = field.copy()
    expected[2, 4] = 3
    expected[3:5, 4] = 2
    expected[2, 5] = 0
    _require(np.array_equal(solve(np.concatenate((reference, separator, field))), expected),
             "Transparent holes, wall deployment, or partial slots failed")
    # Actual solve calls must reject tied ownership and a tied capacity frontier.
    reference[:] = 0
    reference[1:3, 1:3] = [[1, 2], [2, 2]]
    tied_owner = np.zeros_like(field)
    tied_owner[2, 2] = tied_owner[2, 9] = 1
    tied_owner[2, 6] = 3
    tied_capacity = np.zeros_like(field)
    tied_capacity[2, 2] = 1
    for p in ((1, 2), (1, 3), (2, 1), (3, 1), (4, 2), (4, 3), (2, 4), (3, 4)):
        tied_capacity[p] = 3
    for malformed_field in (tied_owner, tied_capacity):
        try:
            solve(np.concatenate((reference, separator, malformed_field)))
        except ValueError:
            continue
        raise AssertionError("Ambiguous payload ownership or capacity accepted")
    records, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_signature(puzzle))
        local = []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                original = np.asarray(pair['input']).copy()
                _require(np.array_equal(solve(original), pair['output']), "Solve mismatch")
                _require(np.array_equal(original, pair['input']), "solve mutated its input")
                f = pair_features(pair)
                f['split'] = split
                records.append(f)
                local.append(f)
        _require(len({f['shapes'] for f in local}) >= 2, "Within-puzzle templates collapsed")
        _require(any(f['collectors'] > 2 for f in local if f['split'] == 'test'), "Missing test shift")
    _require(len(signatures) == 42, "Seed diversity collapsed")
    _require(len({f['input_shape'] for f in records}) >= 35, "Input dimensions collapsed")
    _require(len({f['field_shape'] for f in records}) >= 35, "Field dimensions collapsed")
    _require(len({s for f in records for s in f['shapes']}) >= 100, "Mold vocabulary collapsed")
    _require(set(n for f in records for n in f['capacities']) >= set(range(3, 11)), "Capacity range collapsed")
    _require({f['axis'] for f in records} == {0, 1}, "Divider directions missing")
    _require({f['reference_first'] for f in records} == {False, True}, "Reference ordering missing")
    _require({f['collectors'] for f in records} == {1, 2, 3, 4}, "Collector count range collapsed")
    _require({f['types'] for f in records} == {1, 2, 3}, "Type count range collapsed")
    _require(any(0 in f['loads'] for f in records), "Empty collectors absent")
    _require(any(f['wall_count'] == 0 for f in records), "Wall-free fields absent")
    _require(max(f['max_path'] for f in records) >= 35, "Long paths absent")
    _require(len({f['palette'] for f in records}) >= 50, "Role colors collapsed")
    _require(len({f['excess'] for f in records}) >= 8, "Excess amounts collapsed")
    _require(sum(f['gate_evidence'] for f in records) >= 42, "Gate demonstrations missing")
    _require(sum(f['competition_evidence'] for f in records) >= 42, "Competition demonstrations missing")
    _require(any(f['split_cluster'] for f in records), "Cell-wise cluster splitting absent")
    first = generate_puzzle(771, 2, 1)
    _require(_puzzle_signature(first) == _puzzle_signature(generate_puzzle(771, 2, 1)), "Nondeterministic")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1729+nt*10+ne, nt, ne)
            _require(len(p['train']) == nt and len(p['test']) == ne, "Counts ignored")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (3.0, 1), (3, True), (None, 1)):
        try:
            generate_puzzle(1, nt, ne)
        except ValueError:
            continue
        raise AssertionError("Illegal counts accepted")
    broken = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in first[s]] for s in first}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Corrupt output accepted")
    # Two equal-distance receivers have no arbitrary ownership tie break.
    wall = np.zeros((5, 7), dtype=bool)
    collectors = [dict(footprint=[(2, 1)]), dict(footprint=[(2, 5)])]
    distances, owner = _ownership(wall, collectors)
    _require(owner[2, 3] == -2 and distances[0, 2, 3] == distances[1, 2, 3], "Tie recognition failed")
    # A malformed completed mold must not be silently treated as a new key.
    malformed = np.zeros((6, 6), dtype=np.int8)
    malformed[2, 2:5] = [1, 2, 3]
    try:
        _dictionary(malformed, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("Malformed dictionary accepted")


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds and all legal explicit counts)')
