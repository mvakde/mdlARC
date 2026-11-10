"""Collect the nearest reachable payload cells into a nine-slot counter.

Payload identity is shared by a puzzle. Local colors, geometry, and supply
are visible pair variables. Distance ties across capacity are never labeled.
"""
import numpy as np


_SOURCE_INPUTS = frozenset({
    "77777777777\n"
    "76666666997\n"
    "76997776997\n"
    "76666666777\n"
    "77763336777\n"
    "77763236797\n"
    "79763336777\n"
    "99766766667\n"
    "77767779967\n"
    "77766666667\n"
    "77777777777",

    "77777799977\n"
    "77777799779\n"
    "77777777779\n"
    "97766666666\n"
    "77763336977\n"
    "77963236777\n"
    "77763336797\n"
    "77767776777\n"
    "79767976777\n"
    "77967976777\n"
    "77777777777",

    "77777777777\n"
    "77777777797\n"
    "77977777779\n"
    "77777777777\n"
    "77773337777\n"
    "77773237777\n"
    "77773337777\n"
    "77777777777\n"
    "77777777777\n"
    "77777777977\n"
    "77777777777",

    "66977777966\n"
    "66666666666\n"
    "96977777967\n"
    "76766766769\n"
    "76763336767\n"
    "76773237769\n"
    "76763336767\n"
    "76766766769\n"
    "76977977967\n"
    "76666666766\n"
    "77777777766",

    "77769796777\n"
    "77966666779\n"
    "77777777777\n"
    "66966766669\n"
    "77763336777\n"
    "76663236766\n"
    "77763336777\n"
    "66966676669\n"
    "77767776777\n"
    "76667666766\n"
    "99969796999",
})

_SOURCE_REGRESSIONS = (('77777777777\n76666666997\n76997776997\n76666666777\n77763336777\n77763236797\n79763336777\n99766766667\n77767779967\n77766666667\n77777777777', [(8, 7), (8, 8)], 2, 'train'), ('77777799977\n77777799779\n77777777779\n97766666666\n77763336977\n77963236777\n77763336797\n77767776777\n79767976777\n77967976777\n77777777777', [(1, 6), (3, 0), (4, 8), (5, 2), (6, 9), (8, 1), (8, 5), (9, 2), (9, 5)], 9, 'train'), ('77777777777\n77777777797\n77977777779\n77777777777\n77773337777\n77773237777\n77773337777\n77777777777\n77777777777\n77777777977\n77777777777', [(1, 9), (2, 2), (2, 10), (9, 8)], 4, 'train'), ('66977777966\n66666666666\n96977777967\n76766766769\n76763336767\n76773237769\n76763336767\n76766766769\n76977977967\n76666666766\n77777777766', [(2, 0), (2, 2), (2, 8), (8, 2), (8, 5), (8, 8)], 6, 'test'), ('77769796777\n77966666779\n77777777777\n66966766669\n77763336777\n76663236766\n77763336777\n66966676669\n77767776777\n76667666766\n99969796999', [(1, 2), (1, 10), (3, 2), (3, 10), (7, 2), (7, 10), (10, 0), (10, 4), (10, 6)], 9, 'test'))

def _is_integer_argument(value: object) -> bool:
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
    )


def _normalize_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    if num_train is not None:
        if not _is_integer_argument(num_train):
            raise ValueError("num_train must be an integer or None.")
        num_train = int(num_train)
        if not 2 <= num_train <= 10:
            raise ValueError("num_train must be between 2 and 10.")

    if num_test is not None:
        if not _is_integer_argument(num_test):
            raise ValueError("num_test must be an integer or None.")
        num_test = int(num_test)
        if not 1 <= num_test <= 4:
            raise ValueError("num_test must be between 1 and 4.")

    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        minimum_train = num_test + 1
        span = 11 - minimum_train
        num_train = minimum_train + (seed // 4) % span

    elif num_train is None:
        minimum_train = num_test + 1
        span = 11 - minimum_train
        num_train = minimum_train + seed % span

    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        num_test = 1 + seed % maximum_test

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test.")

    return int(num_train), int(num_test)



_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid_signature(grid):
    return grid.shape, tuple(int(x) for x in grid.ravel())


def _puzzle_signature(puzzle):
    return tuple(tuple((_grid_signature(p['input']), _grid_signature(p['output']))
                       for p in puzzle[s]) for s in ('train', 'test'))


def _counter_mask(shape, pos):
    result = np.zeros(shape, dtype=bool)
    r, c = pos
    result[r:r+3, c:c+3] = True
    return result


def _locate_counter(grid):
    candidates = []
    values, counts = np.unique(grid, return_counts=True)
    for color, count in zip(values, counts):
        if count != 8:
            continue
        rr, cc = np.where(grid == color)
        r, c = int(rr.min()), int(cc.min())
        if int(rr.max()) != r+2 or int(cc.max()) != c+2:
            continue
        patch = grid[r:r+3, c:c+3]
        marker = int(patch[1, 1])
        if marker != color and np.count_nonzero(grid == marker) == 1:
            candidates.append(((r, c), int(color), marker))
    _require(len(candidates) == 1, 'Input needs one unique 3x3 marked collector.')
    return candidates[0]


def _training_conventions(train):
    payloads, backgrounds = set(), set()
    for pair in train:
        a, b = pair['input'], pair['output']
        pos, _, _ = _locate_counter(a)
        counter = _counter_mask(a.shape, pos)
        added = b[counter & (a != b)]
        payloads.update(int(x) for x in added)
        erased = (~counter) & (a != b)
        backgrounds.update(int(x) for x in b[erased])
    _require(len(payloads) == 1, 'Training must identify one shared payload color.')
    _require(backgrounds, 'Training must demonstrate collection.')
    # Stable training palettes retain their literal meanings (as in the source).
    # Changing palettes use a visibly dominant background and thin wall strokes.
    return next(iter(payloads)), next(iter(backgrounds)) if len(backgrounds) == 1 else None


def _roles(grid, payload, fixed_background=None):
    pos, slot, marker = _locate_counter(grid)
    outside = ~_counter_mask(grid.shape, pos)
    field = set(int(x) for x in np.unique(grid[outside])) - {payload}
    _require(1 <= len(field) <= 2, 'Only background, wall and payload outside collector.')
    _require(payload not in (slot, marker), 'Payload and collector colors must differ.')
    _require(not ({slot, marker} & field), 'Collector colors must be local to the collector.')
    if fixed_background is None:
        background = max(field, key=lambda color: int(np.count_nonzero(grid == color)))
        if len(field) == 2:
            counts = [int(np.count_nonzero(grid == color)) for color in field]
            _require(counts[0] != counts[1], 'Changing palettes need a clear background.')
    else:
        background = fixed_background
        _require(background in field, 'Shared background is absent.')
    wall_colors = field - {background}
    wall_color = next(iter(wall_colors)) if wall_colors else None
    wall = grid == wall_color if wall_color is not None else np.zeros(grid.shape, bool)
    return pos, background, wall, slot, marker, wall_color


def _distances(wall, pos, center=False):
    """Multi-source BFS through all non-wall cells, including payload and counter."""
    h, w = wall.shape
    r, c = pos
    starts = [(r+1, c+1)] if center else [(r+i, c+j) for i in range(3) for j in range(3)]
    distance = np.full((h, w), -1, dtype=np.int32)
    for cell in starts:
        distance[cell] = 0
    queue = starts[:]
    for r, c in queue:
        nd = int(distance[r, c]) + 1
        for dr, dc in _STEPS:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and not wall[nr, nc] and distance[nr, nc] < 0:
                distance[nr, nc] = nd
                queue.append((nr, nc))
    return distance


def _wave_distances(wall, pos):
    """Independent array wavefront implementation used to audit BFS labels."""
    seen = _counter_mask(wall.shape, pos)
    result = np.full(wall.shape, -1, dtype=np.int32)
    result[seen] = 0
    frontier = seen.copy()
    for depth in range(1, wall.size + 1):
        spread = np.zeros(wall.shape, dtype=bool)
        spread[1:] |= frontier[:-1]
        spread[:-1] |= frontier[1:]
        spread[:, 1:] |= frontier[:, :-1]
        spread[:, :-1] |= frontier[:, 1:]
        frontier = spread & ~seen & ~wall
        if not np.any(frontier):
            break
        seen |= frontier
        result[frontier] = depth
    return result


def _selection(grid, payload, distance, pos):
    supply = (grid == payload) & ~_counter_mask(grid.shape, pos) & (distance >= 0)
    ordered = np.sort(distance[supply])
    if len(ordered) > 9:
        _require(ordered[8] < ordered[9], 'Capacity frontier must have a strict distance gap.')
    threshold = int(ordered[min(8, len(ordered)-1)]) if len(ordered) else -1
    return supply & (distance <= threshold), int(supply.sum())


def _transform(grid, payload, background, wall, pos, independent=False):
    distance = _wave_distances(wall, pos) if independent else _distances(wall, pos)
    selected, _ = _selection(grid, payload, distance, pos)
    result = grid.copy()
    result[selected] = background
    r, c = pos
    for i in range(int(selected.sum())):
        result[r+i//3, c+i%3] = payload
    return result


def solve(train_pairs, test_input):
    """Prediction uses only the demonstrations and the new input."""
    payload, background = _training_conventions(train_pairs)
    pos, background, wall, _, _, _ = _roles(test_input, payload, background)
    return _transform(test_input, payload, background, wall, pos, independent=True)


def _components(mask):
    remaining = set(int(x) for x in np.flatnonzero(mask))
    h, w = mask.shape
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue = [start]
        for cell in queue:
            r, c = divmod(cell, w)
            for dr, dc in _STEPS:
                nr, nc = r+dr, c+dc
                neighbor = nr*w+nc
                if 0 <= nr < h and 0 <= nc < w and neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        result.append(queue)
    return result


def _rectangle(wall, r0, c0, r1, c1):
    wall[r0, c0:c1+1] = True
    wall[r1, c0:c1+1] = True
    wall[r0:r1+1, c0] = True
    wall[r0:r1+1, c1] = True


def _open_rectangle(rng, wall, bounds, openings):
    r0, c0, r1, c1 = bounds
    for _ in range(openings):
        side = int(rng.integers(4))
        if side < 2:
            c = int(rng.integers(c0+1, c1))
            wall[r0 if side == 0 else r1, c:min(c+int(rng.integers(1, 4)), c1)] = False
        else:
            r = int(rng.integers(r0+1, r1))
            wall[r:min(r+int(rng.integers(1, 4)), r1), c0 if side == 2 else c1] = False


def _wall_geometry(rng, shape, pos, kind, complexity):
    h, w = shape
    r, c = pos
    wall = np.zeros(shape, dtype=bool)
    if kind == 'open':
        return wall
    if kind == 'maze':
        wall[:] = True
        nodes = [(rr, cc) for rr in range(1, h-1, 2) for cc in range(1, w-1, 2)]
        start = nodes[int(rng.integers(len(nodes)))]
        seen, stack = {start}, [start]
        wall[start] = False
        while stack:
            rr, cc = stack[-1]
            choices = [(rr+2*dr, cc+2*dc) for dr, dc in _STEPS
                       if 1 <= rr+2*dr < h-1 and 1 <= cc+2*dc < w-1
                       and (rr+2*dr, cc+2*dc) not in seen]
            if not choices:
                stack.pop()
                continue
            nr, nc = choices[int(rng.integers(len(choices)))]
            wall[(rr+nr)//2, (cc+nc)//2] = False
            wall[nr, nc] = False
            seen.add((nr, nc))
            stack.append((nr, nc))
        # Chambers widen some branches without replacing the whole maze.
        for _ in range(int(rng.integers(2, 6))):
            rr, cc = int(rng.integers(h-2)), int(rng.integers(w-2))
            wall[rr:min(h, rr+int(rng.integers(2, 6))), cc:min(w, cc+int(rng.integers(2, 6)))] = False
    elif kind == 'rings':
        for offset in range(1, min(r, c, h-r-3, w-c-3)+1, 2):
            bounds = (max(0, r-offset), max(0, c-offset), min(h-1, r+2+offset), min(w-1, c+2+offset))
            _rectangle(wall, *bounds)
            _open_rectangle(rng, wall, bounds, int(rng.integers(0, 5)))
        if not wall.any():
            _rectangle(wall, r-1, c-1, r+3, c+3)
            _open_rectangle(rng, wall, (r-1, c-1, r+3, c+3), int(rng.integers(1, 5)))
    elif kind == 'rails':
        # An asymmetric U can be an enclosed pocket or run out to a boundary.
        r0 = int(rng.integers(0, r+1))
        r1 = int(rng.integers(r+3, h))
        c0 = int(rng.integers(0, c+1))
        c1 = int(rng.integers(c+3, w))
        _rectangle(wall, r0, c0, r1, c1)
        _open_rectangle(rng, wall, (r0, c0, r1, c1), int(rng.integers(0, 5)))
        if rng.random() < .65:
            side = int(rng.integers(4))
            if side < 2:
                wall[:r0 if side == 0 else h, c0] = True
            else:
                wall[r1, :c0 if side == 2 else w] = True
    elif kind == 'partitions':
        rooms = [(0, 0, h-1, w-1)]
        for _ in range(complexity * 3 + 2):
            if not rooms:
                break
            idx = int(rng.integers(len(rooms)))
            r0, c0, r1, c1 = rooms.pop(idx)
            horizontal = (rng.random() < .5 and r1-r0 >= 5) or c1-c0 < 5
            if horizontal and r1-r0 >= 5:
                rr = int(rng.integers(r0+2, r1-1))
                wall[rr, c0:c1+1] = True
                for _ in range(int(rng.integers(0, 4))):
                    cc = int(rng.integers(c0, c1+1))
                    wall[rr, cc:min(c1+1, cc+int(rng.integers(1, 4)))] = False
                rooms.extend([(r0, c0, rr-1, c1), (rr+1, c0, r1, c1)])
            elif c1-c0 >= 5:
                cc = int(rng.integers(c0+2, c1-1))
                wall[r0:r1+1, cc] = True
                for _ in range(int(rng.integers(0, 4))):
                    rr = int(rng.integers(r0, r1+1))
                    wall[rr:min(r1+1, rr+int(rng.integers(1, 4))), cc] = False
                rooms.extend([(r0, c0, r1, cc-1), (r0, cc+1, r1, c1)])
    # Segment, turn and enclosure primitives also combine with the other modes.
    extras = int(rng.integers(2, complexity*3+4)) if kind == 'segments' else int(rng.integers(complexity+1))
    for _ in range(extras):
        r0, c0 = int(rng.integers(h-2)), int(rng.integers(w-2))
        r1 = int(rng.integers(r0+2, h))
        c1 = int(rng.integers(c0+2, w))
        primitive = int(rng.integers(4))
        if primitive == 0:
            _rectangle(wall, r0, c0, r1, c1)
            _open_rectangle(rng, wall, (r0, c0, r1, c1), int(rng.integers(0, 4)))
        elif primitive == 1:
            wall[r0, c0:c1+1] = True
            wall[r0:r1+1, c1] = True
        elif primitive == 2:
            wall[r0, c0:c1+1] = True
        else:
            wall[r0:r1+1, c0] = True
    wall[r:r+3, c:c+3] = False
    # Keep new local color roles readable; recompute routes after every change.
    for _ in range(h*w):
        if wall.sum() <= .38*(h*w-9):
            break
        rr, cc = int(rng.integers(h)), int(rng.integers(w))
        wall[rr:rr+3, cc:cc+3] = False
    return wall


def _pick_cells(rng, mask, count, style, forced=()):
    available = set(int(x) for x in np.flatnonzero(mask))
    chosen = list(dict.fromkeys(int(x) for x in forced))
    _require(set(chosen) <= available and len(chosen) <= count <= len(available), 'Invalid supply placement.')
    available.difference_update(chosen)
    w = mask.shape[1]
    anchor = chosen[0] if chosen else int(rng.choice(sorted(available))) if available else 0
    ar, ac = divmod(anchor, w)
    horizontal = bool(rng.integers(2))
    while len(chosen) < count:
        cells = np.array(sorted(available), dtype=int)
        if style == 'dispersed':
            cell = int(rng.choice(cells))
        elif style == 'line':
            rr, cc = cells//w, cells%w
            score = (np.abs(rr-ar)*w + np.abs(cc-ac)) if horizontal else (np.abs(cc-ac)*mask.shape[0]+np.abs(rr-ar))
            best = cells[score == score.min()]
            cell = int(rng.choice(best))
        else:
            # Compact supplies can cross both reachability and capacity frontiers.
            rr, cc = cells//w, cells%w
            score = (rr-ar)**2 + (cc-ac)**2
            best = cells[score == score.min()]
            cell = int(rng.choice(best))
        chosen.append(cell)
        available.remove(cell)
    return chosen


def _supply(rng, distance, center_distance, pos, count, style, detour=False):
    outside = ~_counter_mask(distance.shape, pos)
    active = outside & (distance >= 0)
    if count <= 9:
        if active.sum() < count:
            return None
        return _pick_cells(rng, active, count, style)
    # Candidate frontiers simultaneously separate footprint and center distances.
    h, w = distance.shape
    rr, cc = np.indices(distance.shape)
    r, c = pos
    manhattan = np.maximum(r-rr, 0)+np.maximum(rr-r-2, 0)+np.maximum(c-cc, 0)+np.maximum(cc-c-2, 0)
    candidates = rng.permutation(np.flatnonzero(active))
    want_split = rng.random() < .65
    for a in candidates:
        a = int(a)
        da, ca = int(distance.flat[a]), int(center_distance.flat[a])
        near = active & (distance <= da) & (center_distance <= ca)
        far = active & (distance > da) & (center_distance > ca)
        if near.sum() < 9 or far.sum() < count-9:
            continue
        choices = far.copy()
        if detour:
            choices &= manhattan < manhattan.flat[a]
        elif want_split:
            adjacent = np.zeros(distance.shape, dtype=bool)
            ar, ac = divmod(a, w)
            for dr, dc in _STEPS:
                nr, nc = ar+dr, ac+dc
                if 0 <= nr < h and 0 <= nc < w:
                    adjacent[nr, nc] = True
            choices &= adjacent
        if not choices.any():
            continue
        b = int(rng.choice(np.flatnonzero(choices)))
        selected = _pick_cells(rng, near, 9, style, [a])
        retained = _pick_cells(rng, far, count-9, style, [b])
        return selected + retained
    if want_split and not detour:
        # A geometric split is optional; strict capacity separation is mandatory.
        return _supply(rng, distance, center_distance, pos, count, style, detour=True)
    return None


def _fallback_geometry():
    # Constructive U-shaped detour with a separate closed pocket. This is used
    # only if bounded random geometry cannot satisfy a requested demonstration.
    wall = np.zeros((17, 19), dtype=bool)
    pos = (5, 6)
    wall[4, 5:] = True
    wall[4:15, 5] = True
    wall[4:15, 9] = True
    _rectangle(wall, 0, 0, 3, 4)
    return wall, pos


def _generate_pair(rng, parameters, spec, regime, previous_background=None):
    payload = parameters['payload']
    for attempt in range(65):
        if attempt == 64:
            wall, pos = _fallback_geometry()
            h, w = wall.shape
        else:
            if parameters['wide_sizes']:
                h, w = (int(rng.integers(7, 31)) for _ in range(2))
            else:
                bh, bw = parameters['base_shape']
                h, w = max(7, min(30, bh+int(rng.integers(-1, 2)))), max(7, min(30, bw+int(rng.integers(-1, 2))))
            pos = (int(rng.integers(1, h-3)), int(rng.integers(1, w-3)))
            if rng.random() < .2:
                pos = ((h-3)//2, (w-3)//2)
            kinds = ['open', 'rails', 'rings', 'segments', 'partitions', 'maze']
            shifted = regime == 'target_test' and parameters['shift']
            weights = [.16, .23, .15, .20, .18, .08] if parameters['simple'] else [.12, .12, .15, .22, .23, .16]
            if shifted:
                weights = [.10, .10, .17, .18, .25, .20]
            kind = str(rng.choice(kinds, p=weights))
            complexity = int(rng.integers(1, 3 if parameters['simple'] else 5))
            wall = _wall_geometry(rng, (h, w), pos, kind, complexity)
        distance = _distances(wall, pos)
        center_distance = _distances(wall, pos, center=True)
        outside = ~_counter_mask(wall.shape, pos)
        inactive = outside & ~wall & (distance < 0)
        if spec['selective'] and inactive.sum() < 1:
            continue
        style = str(rng.choice(['dispersed', 'compact', 'line']))
        supply = _supply(rng, distance, center_distance, pos, spec['load'], style, spec['detour'])
        if supply is None:
            continue
        minimum = 1 if spec['selective'] or spec['load'] == 0 else 0
        if inactive.sum() < minimum:
            continue
        # Separate unreachable distraction from excess reachable supply.
        limit = min(24, int(inactive.sum()), max(0, (h*w-9)//5))
        if minimum > limit:
            continue
        amount = int(rng.integers(minimum, limit+1)) if rng.random() < .8 or minimum else 0
        distraction = _pick_cells(rng, inactive, amount, str(rng.choice(['dispersed', 'compact', 'line'])))
        if h*w-9-int(wall.sum())-len(supply)-amount <= int(wall.sum()):
            continue
        if parameters['vary_colors']:
            choices = [x for x in range(10) if x != payload]
            colors = [int(x) for x in rng.choice(choices, size=4, replace=False)]
            if previous_background is not None and colors[0] == previous_background:
                colors[0], colors[1] = colors[1], colors[0]
        else:
            colors = parameters['colors']
        background, wall_color, slot, marker = colors
        grid = np.full((h, w), background, dtype=np.int8)
        grid[wall] = wall_color
        grid.flat[supply+distraction] = payload
        r, c = pos
        grid[r:r+3, c:c+3] = slot
        grid[r+1, c+1] = marker
        if '\n'.join(''.join(str(int(x)) for x in row) for row in grid) in _SOURCE_INPUTS:
            continue
        # Both distance origins must give the same uniquely separated frontier.
        selected, _ = _selection(grid, payload, distance, pos)
        other, _ = _selection(grid, payload, center_distance, pos)
        _require(np.array_equal(selected, other), 'Distance origins disagree.')
        return {'input': grid, 'output': _transform(grid, payload, background, wall, pos)}, background
    raise AssertionError('Constructive geometry fallback failed.')


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if not _is_integer_argument(seed) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer.')
    seed = int(seed)
    num_train, num_test = _normalize_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = [int(x) for x in rng.choice(10, size=5, replace=False)]
    parameters = dict(payload=palette[0], colors=palette[1:], vary_colors=bool(rng.random() < .8),
                      wide_sizes=bool(rng.random() < .8), base_shape=tuple(int(x) for x in rng.integers(10, 22, 2)),
                      simple=bool(rng.random() < .25), shift=bool(rng.random() < .6))
    excess = bool(rng.random() < .7)
    # These are evidence obligations, not pair-index schedules. Shuffle them.
    specs = [dict(load=int(rng.choice([2, 4])), selective=not excess, detour=False),
             dict(load=int(rng.integers(10, 25)) if excess else int(rng.integers(5, 10)),
                  selective=excess, detour=excess)]
    def random_spec():
        mode = rng.random()
        load = 0 if mode < .05 else int(rng.integers(10, 35)) if excess and mode < .35 else int(rng.integers(1, 10))
        return dict(load=load, selective=(load == 0), detour=False)
    specs.extend(random_spec() for _ in range(num_train-2))
    specs = [specs[int(i)] for i in rng.permutation(num_train)]
    puzzle = {'train': [], 'test': []}
    seen = set()
    previous_background = None
    for regime, split, requests in [('source_train', 'train', specs),
                                    ('target_test', 'test', [random_spec() for _ in range(num_test)])]:
        for spec in requests:
            for retry in range(16):
                pair, background = _generate_pair(rng, parameters, spec, regime, previous_background)
                signature = _grid_signature(pair['input'])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    # A zero example cannot teach a background through erasure.
                    # Ensure positive demonstrations reveal palette variation.
                    if spec['load'] > 0:
                        previous_background = background
                    break
            else:
                raise AssertionError('Duplicate construction after bounded retries.')
    validate_generated_puzzle(puzzle)
    return puzzle


def pair_features(train_pairs, grid):
    """Measurements reconstructed from demonstrations and visible input only."""
    payload, fixed = _training_conventions(train_pairs)
    pos, background, wall, slot, marker, wall_color = _roles(grid, payload, fixed)
    outside = ~_counter_mask(grid.shape, pos)
    distance = _distances(wall, pos)
    selected, reachable = _selection(grid, payload, distance, pos)
    active = outside & (distance >= 0)
    inactive = outside & ~wall & ~active
    payload_mask = outside & (grid == payload)
    retained = payload_mask & active & ~selected
    center_distance = _distances(wall, pos, True)
    center_selected, _ = _selection(grid, payload, center_distance, pos)
    gap = int(distance[retained].min() - distance[selected].max()) if retained.any() else 0
    center_gap = int(center_distance[retained].min() - center_distance[selected].max()) if retained.any() else 0
    h, w = grid.shape
    r, c = pos
    rr, cc = np.indices(grid.shape)
    manhattan = np.maximum(r-rr, 0)+np.maximum(rr-r-2, 0)+np.maximum(c-cc, 0)+np.maximum(cc-c-2, 0)
    euclidean = np.maximum(np.maximum(r-rr, rr-r-2), 0)**2 + np.maximum(np.maximum(c-cc, cc-c-2), 0)**2
    gates = active & (manhattan == 1)
    directions = [bool(gates[:r].any()), bool(gates[r+3:].any()), bool(gates[:, :c].any()), bool(gates[:, c+3:].any())]
    components = _components(payload_mask)
    split_cluster = any(any(selected.flat[i] for i in comp) and any(retained.flat[i] for i in comp) for comp in components)
    degree = np.zeros(grid.shape, dtype=int)
    traversable = active | ~outside
    degree[1:] += traversable[:-1]
    degree[:-1] += traversable[1:]
    degree[:, 1:] += traversable[:, :-1]
    degree[:, :-1] += traversable[:, 1:]
    inactive_components = _components(inactive)
    active_components = _components(active)
    longest_line = 0
    for row in list(payload_mask) + list(payload_mask.T):
        run = 0
        for value in row:
            run = run+1 if value else 0
            longest_line = max(longest_line, run)
    def reversal(metric):
        return bool(selected.any() and retained.any() and metric[selected].max() > metric[retained].min())
    return dict(height=h, width=w, collector_row=r, collector_col=c,
                off_center=bool(2*r+3 != h or 2*c+3 != w),
                palette=[background, wall_color, payload, slot, marker],
                wall_count=int(wall.sum()), selected=int(selected.sum()), reachable=reachable,
                excess=max(0, reachable-9), unreachable=int((payload_mask & inactive).sum()),
                capacity_gap=gap, center_capacity_gap=center_gap,
                origins_agree=bool(np.array_equal(selected, center_selected)),
                max_distance=int(distance[active].max()) if active.any() else 0,
                max_detour=int((distance-manhattan)[active].max()) if active.any() else 0,
                entrances=int(gates.sum()), entrance_sides=sum(directions),
                active_components=len(active_components), active_sizes=sorted(len(x) for x in active_components),
                inactive_components=len(inactive_components), inactive_sizes=sorted(len(x) for x in inactive_components),
                active_size=int(active.sum()), reaches_edge=bool(active[0].any() or active[-1].any() or active[:, 0].any() or active[:, -1].any()),
                dead_ends=int((active & (degree == 1)).sum()), branches=int((active & (degree >= 3)).sum()),
                path_beats_manhattan=reversal(manhattan), path_beats_euclidean=reversal(euclidean),
                split_cluster=split_cluster, payload_components=len(components),
                largest_cluster=max((len(x) for x in components), default=0), longest_line=longest_line,
                compact_square=bool(np.any(payload_mask[:-1, :-1] & payload_mask[1:, :-1] & payload_mask[:-1, 1:] & payload_mask[1:, 1:])),
                occupied_active_components=sum(any(payload_mask.flat[i] for i in comp) for comp in active_components))


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'Invalid puzzle keys.')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'Splits must be lists.')
    _require(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
             and len(puzzle['train']) > len(puzzle['test']), 'Invalid pair counts.')
    seen = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'Invalid pair schema.')
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, 'Grids must be 2D arrays.')
                _require(all(1 <= n <= 30 for n in grid.shape), 'Invalid grid dimensions.')
                _require(np.issubdtype(grid.dtype, np.integer) and int(grid.min()) >= 0 and int(grid.max()) <= 9, 'Invalid grid colors/dtype.')
            a, b = pair['input'], pair['output']
            _require(a.shape == b.shape, 'Dimensions must be preserved.')
            sig = _grid_signature(a)
            _require(sig not in seen, 'Duplicate input.')
            seen.add(sig)
            text = '\n'.join(''.join(str(int(x)) for x in row) for row in a)
            _require(text not in _SOURCE_INPUTS, 'Source input copied.')
    payload, fixed = _training_conventions(puzzle['train'])
    stats = {'train': [], 'test': []}
    for split in stats:
        for pair in puzzle[split]:
            a, b = pair['input'], pair['output']
            pos, background, wall, _, _, _ = _roles(a, payload, fixed)
            expected = _transform(a, payload, background, wall, pos, independent=True)
            _require(np.array_equal(b, expected), 'Incorrect path/capacity/prefix output.')
            selected, _ = _selection(a, payload, _distances(wall, pos), pos)
            other, _ = _selection(a, payload, _distances(wall, pos, True), pos)
            _require(np.array_equal(selected, other), 'Ambiguous distance origin.')
            feature = pair_features(puzzle['train'], a)
            stats[split].append(feature)
            if not selected.any():
                _require(feature['unreachable'] > 0 and np.array_equal(a, b), 'Zero case needs visible unreachable payload.')
    train, tests = stats['train'], stats['test']
    _require(any(x['selected'] in (2, 4) for x in train), 'Training must reveal prefix order and preserve marker.')
    _require(any(x['selected'] >= 5 for x in train), 'Training must overwrite the marker.')
    _require(any(x['unreachable'] for x in train), 'Training must show selective reachability.')
    if any(x['excess'] for x in train+tests):
        _require(any(x['excess'] and x['path_beats_manhattan'] for x in train), 'Excess requires a path-distance training demonstration.')
    for pair in puzzle['test']:
        prediction = solve([{k: v.copy() for k, v in p.items()} for p in puzzle['train']], pair['input'].copy())
        _require(np.array_equal(prediction, pair['output']), 'Training/input-only prediction failed.')


def self_test() -> None:
    # Payload is traversable. Capacity may cut a line, while a wall-separated
    # cell and the five reachable excess cells must both remain in place.
    line = np.full((8, 23), 7, dtype=np.int8)
    line[:, 19] = 6
    line[1:4, 1:4] = 3
    line[2, 2] = 2
    line[2, 4:18] = 9
    line[2, 21] = 9
    line_expected = line.copy()
    line_expected[2, 4:13] = 7
    line_expected[1:4, 1:4] = 9
    for independent in (False, True):
        _require(np.array_equal(_transform(line, 9, 7, line == 6, (1, 1), independent), line_expected),
                 'Reachable excess, unreachable payload, or traversable payload regression.')
    # Twelve equal-distance candidates must not silently use queue order.
    tied = np.full((7, 7), 7, dtype=np.int8)
    tied[2:5, 2:5] = 3
    tied[3, 3] = 2
    distances = _distances(np.zeros(tied.shape, dtype=bool), (2, 2))
    tied[distances == 1] = 9
    try:
        _selection(tied, 9, distances, (2, 2))
    except AssertionError:
        pass
    else:
        raise AssertionError('Ambiguous capacity tie accepted.')
    # The last construction attempt must support every excess request.
    fallback, fallback_pos = _fallback_geometry()
    fd = _distances(fallback, fallback_pos)
    fc = _distances(fallback, fallback_pos, True)
    fallback_rng = np.random.default_rng(82173)
    for load in range(10, 35):
        for style in ('line', 'compact', 'dispersed'):
            _require(_supply(fallback_rng, fd, fc, fallback_pos, load, style, True) is not None,
                     'Fallback cannot support an excess load.')
    # Fixed numeric source evidence: no source files are read by this module.
    source_train = []
    for text, removed, count, split in _SOURCE_REGRESSIONS:
        grid = np.array([[int(x) for x in row] for row in text.splitlines()], dtype=np.int8)
        expected = grid.copy()
        for r, c in removed:
            expected[r, c] = 7
        for i in range(count):
            expected[4+i//3, 4+i%3] = 9
        wall = grid == 6
        _require(np.array_equal(_transform(grid, 9, 7, wall, (4, 4)), expected), 'Source output regression failed.')
        selected, reachable = _selection(grid, 9, _distances(wall, (4, 4)), (4, 4))
        other, _ = _selection(grid, 9, _distances(wall, (4, 4), True), (4, 4))
        _require(np.array_equal(selected, other), 'Source origin agreement regression failed.')
        if split == 'train':
            source_train.append({'input': grid, 'output': expected})
        else:
            _require(np.array_equal(solve(source_train, grid), expected), 'Source prediction regression failed.')
    puzzles, features, by_puzzle = [], [], []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        local = []
        for split in ('train', 'test'):
            for index, pair in enumerate(puzzle[split]):
                f = pair_features(puzzle['train'], pair['input'])
                f['split'], f['index'] = split, index
                features.append(f)
                local.append(f)
        by_puzzle.append(local)
    _require(len({len(p['train']) for p in puzzles}) >= 4 and len({len(p['test']) for p in puzzles}) >= 3, 'Automatic count diversity collapsed.')
    _require(len({_puzzle_signature(p) for p in puzzles}) == 42, 'Seed diversity collapsed.')
    for field, minimum in [('height', 15), ('width', 15), ('collector_row', 10), ('collector_col', 10),
                           ('selected', 10), ('excess', 12), ('unreachable', 12), ('entrances', 8),
                           ('max_distance', 20), ('inactive_components', 5), ('active_components', 3)]:
        _require(len({f[field] for f in features}) >= minimum, 'Insufficient variation: '+field)
    for field in ['off_center', 'split_cluster', 'compact_square', 'path_beats_manhattan', 'path_beats_euclidean']:
        _require(sum(f[field] for f in features) >= 8, 'Missing reviewed feature: '+field)
    _require(max(f['max_distance'] for f in features) >= 35, 'Long routes absent.')
    _require(any(not f['reaches_edge'] and f['active_size'] > 0 for f in features), 'Enclosed active regions absent.')
    _require(any(f['active_size'] > 250 for f in features), 'Extensive active fields absent.')
    _require(any(f['occupied_active_components'] >= 2 for f in features), 'Multi-branch supplies absent.')
    _require(max(f['longest_line'] for f in features) >= 5, 'Long payload lines absent.')
    _require(any(f['excess'] and f['unreachable'] and f['entrance_sides'] >= 2 and f['max_detour'] >= 8 for f in features), 'Combined complexity absent.')
    for split in ['train', 'test']:
        fs = [f for f in features if f['split'] == split]
        _require(any(f['wall_count'] == 0 for f in fs) and any(f['max_distance'] > 25 and f['wall_count'] > 30 for f in fs), 'Both regimes need simple and complex layouts.')
    for index in range(4):
        _require(len({f['selected'] for f in features if f['split'] == 'test' and f['index'] == index}) >= 4, 'Test-index load schedule detected.')
    for field in ['height', 'width', 'selected', 'entrances', 'unreachable', 'max_distance']:
        _require(sum(len({f[field] for f in fs}) > 1 for fs in by_puzzle) >= 30, 'Within-puzzle variation missing: '+field)
    _require(sum(len({(f['palette'][0], f['palette'][3], f['palette'][4]) for f in fs}) > 1
                 for fs in by_puzzle) >= 25, 'Within-puzzle colors collapsed.')
    for role in range(5):
        _require(len({f['palette'][role] for f in features if f['palette'][role] is not None}) == 10,
                 'Cross-seed role color range collapsed.')
    open_cases = [f for f in features if f['wall_count'] == 0]
    _require(len({f['selected'] for f in open_cases}) >= 7
             and len({f['index'] for f in open_cases if f['split'] == 'train'}) >= 5,
             'Wall-free loads or positions collapsed.')
    _require(_puzzle_signature(generate_puzzle(123456, 7, 3)) == _puzzle_signature(generate_puzzle(123456, 7, 3)), 'Determinism failed.')
    for seed, nt, ne in [(1001, 2, 1), (1002, 10, 1), (1003, 10, 4), (2001, None, 1), (2002, None, 4), (2003, 2, None), (2004, 10, None)]:
        p = generate_puzzle(seed, nt, ne)
        validate_generated_puzzle(p)
        _require(nt is None or len(p['train']) == nt, 'Explicit training count ignored.')
        _require(ne is None or len(p['test']) == ne, 'Explicit test count ignored.')
    for nt, ne in [(1, None), (11, None), (None, 0), (None, 5), (2, 2), (2, 3), (4, 4), (3.0, 1), (3, True)]:
        try:
            generate_puzzle(3001, nt, ne)
        except ValueError:
            continue
        raise AssertionError('Illegal count accepted.')
    # Corrupt a valid label; validation must inspect the returned cells.
    broken = generate_puzzle(123, 2, 1)
    broken['test'][0]['output'] = broken['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        return
    raise AssertionError('Corrupted output accepted.')
