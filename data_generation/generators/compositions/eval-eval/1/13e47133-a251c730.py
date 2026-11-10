"""Hidden islands and contour tides: visible templates change a cyclic depth field.

Only the standard library and NumPy are used. Generation builds labels from its
sampled geometry; solve reconstructs that geometry exclusively from input cells.
"""

from collections import deque
import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("grids must be 2D integers, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in D4:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _masks(mask):
    result = []
    for cells in _components(mask):
        m = np.zeros(mask.shape, dtype=bool)
        rr, cc = zip(*cells)
        m[rr, cc] = True
        result.append(m)
    return result


def _inside(mask, r, c):
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])


def _depth(mask, diagonal=True):
    """Shortest inward distance from the boundary; holes are boundary too."""
    directions = D8 if diagonal else D4
    depths = np.full(mask.shape, -1, dtype=np.int16)
    queue = deque()
    for r, c in np.argwhere(mask).tolist():
        if any(not _inside(mask, r + dr, c + dc) for dr, dc in directions):
            depths[r, c] = 0
            queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if _inside(mask, nr, nc) and depths[nr, nc] < 0:
                depths[nr, nc] = depths[r, c] + 1
                queue.append((nr, nc))
    return depths


def _erosion(mask, diagonal=True):
    """Separate vectorized erosion implementation for label construction/checks."""
    result = np.full(mask.shape, -1, dtype=np.int16)
    current, level = mask.copy(), 0
    h, w = mask.shape
    while np.any(current):
        padded = np.pad(current, 1, constant_values=False)
        interior = current.copy()
        for dr, dc in (D8 if diagonal else D4):
            interior &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        result[current & ~interior] = level
        current = interior
        level += 1
    return result


def _majority(values):
    counts = np.bincount(np.asarray(values).ravel(), minlength=10)
    color = int(np.argmax(counts))
    assert counts[color] > len(np.asarray(values).ravel()) / 2, "fill must be a strict majority"
    return color


def _bounds(signature):
    rr, cc = zip(*[(r, c) for r, c, _ in signature])
    return min(rr), min(cc), max(rr) - min(rr) + 1, max(cc) - min(cc) + 1


def _geometry(signature):
    return tuple((r, c) for r, c, _ in signature)


def _hollow(signature):
    lr, lc, h, w = _bounds(signature)
    occupied = np.zeros((h, w), dtype=bool)
    for r, c, _ in signature:
        occupied[r - lr, c - lc] = True
    return any(all(r not in (0, h - 1) and c not in (0, w - 1) for r, c in comp)
               for comp in _components(~occupied))


def _asymmetric(signature):
    cells = set(_geometry(signature))
    return cells != {(-c, r) for r, c in cells} and cells != {(r, -c) for r, c in cells}


def _find_arena(grid):
    candidates = set()
    for color in np.unique(grid).tolist():
        for component in _components(grid == color):
            rr, cc = zip(*component)
            t, b, l, r = min(rr), max(rr), min(cc), max(cc)
            if b - t + 1 < 7 or r - l + 1 < 7:
                continue
            panel = grid[t:b + 1, l:r + 1]
            same = panel == color
            # A filled exterior is not a one-cell wall network even when its
            # bounding box happens to surround the complete scene.
            if np.any(same[:-1, :-1] & same[1:, :-1] & same[:-1, 1:] & same[1:, 1:]):
                continue
            if (np.all(panel[0] == color) and np.all(panel[-1] == color)
                    and np.all(panel[:, 0] == color) and np.all(panel[:, -1] == color)
                    and np.any(panel[1:-1, 1:-1] != color)):
                candidates.add((t, l, b - t + 1, r - l + 1, int(color)))
    assert candidates, "no enclosing frame"
    biggest = max(h * w for _, _, h, w, _ in candidates)
    choices = [p for p in candidates if p[2] * p[3] == biggest]
    assert len(choices) == 1, "competing largest frames"
    return choices[0]


def _dictionary(grid, box, wall):
    t, l, h, w = box
    outside = np.ones(grid.shape, dtype=bool)
    outside[t:t + h, l:l + w] = False
    exterior = _majority(grid[outside])
    assert exterior != wall
    templates, repetitions = {}, {}
    for component in _components(outside & (grid != exterior)):
        assert 3 <= len(component) <= 16
        keys = [(r, c) for r, c in component if grid[r, c] != wall]
        assert len(keys) == 1, "reference must have one distinctive anchor"
        ar, ac = keys[0]
        key = int(grid[ar, ac])
        signature = tuple(sorted((r - ar, c - ac, int(grid[r, c])) for r, c in component))
        assert max(_bounds(signature)[2:]) <= 5
        assert key not in templates or templates[key] == signature, "conflicting references"
        templates[key] = signature
        repetitions[key] = repetitions.get(key, 0) + 1
        # External references have an exterior-cell moat from each other/arena.
        own = set(component)
        for r, c in component:
            for dr, dc in D8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and (nr, nc) not in own:
                    assert outside[nr, nc] and grid[nr, nc] == exterior
    assert 1 <= len(templates) <= 3
    return exterior, templates, repetitions


def _paths(room, depths, length):
    paths = []
    for r, c in np.argwhere(room & (depths == 0)).tolist():
        for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            if (_inside(room, r - dr, c) or _inside(room, r, c - dc)
                    or not _inside(room, r + dr, c) or not _inside(room, r, c + dc)):
                continue
            cells = tuple((r + d * dr, c + d * dc) for d in range(length))
            if all(_inside(room, rr, cc) and depths[rr, cc] == d
                   for d, (rr, cc) in enumerate(cells)):
                paths.append((cells, (dr, dc)))
    return paths


def _decode(room, depths, arena, neutral, identities):
    cues = np.argwhere(room & (arena != neutral) & ~np.isin(arena, list(identities))).tolist()
    if not cues:
        return (neutral,), (), ()
    length = max(int(depths[r, c]) for r, c in cues) + 1
    assert 1 <= length <= 3
    readings = []
    for cells, direction in _paths(room, depths, length):
        if all((r, c) == cells[int(depths[r, c])] for r, c in cues):
            readings.append((tuple(int(arena[p]) for p in cells), cells, direction))
    assert readings and len({p for p, _, _ in readings}) == 1, "ambiguous corner key"
    period = readings[0][0]
    assert period[-1] != neutral
    return period, readings[0][1], tuple(sorted({d for _, _, d in readings}))


def _reconstruct(input_grid):
    grid = _grid(input_grid)
    t, l, h, w, wall = _find_arena(grid)
    exterior, templates, repetitions = _dictionary(grid, (t, l, h, w), wall)
    arena = grid[t:t + h, l:l + w].copy()
    walls = arena == wall
    assert len(_components(walls)) == 1, "explicit wall network must attach to frame"
    assert not np.any(walls[:-1, :-1] & walls[1:, :-1] & walls[:-1, 1:] & walls[1:, 1:])
    neutral = _majority(arena[~walls])
    assert neutral not in templates and neutral != wall
    rooms = _masks(~walls)
    assert 1 <= len(rooms) <= 4
    labels = np.full(arena.shape, -1, dtype=int)
    regions = []
    for i, room in enumerate(rooms):
        labels[room] = i
        before = _depth(room)
        period, key_cells, directions = _decode(room, before, arena, neutral, templates)
        assert all(c not in templates and c != wall and (exterior == neutral or c != exterior)
                   for c in period)
        regions.append(dict(mask=room, before=before, period=period,
                            key_cells=key_cells, directions=directions))
    anchors = [(r, c, int(arena[r, c])) for r, c in np.argwhere(np.isin(arena, list(templates))).tolist()]
    assert anchors
    assert all(len(c) == 1 for c in _components(np.isin(arena, list(templates))))
    solid = walls.copy()
    stamp_mask = np.zeros(arena.shape, dtype=bool)
    forbidden = np.zeros(arena.shape, dtype=bool)
    stamped = arena.copy()
    placements = []
    for ar, ac, key in anchors:
        room_id = int(labels[ar, ac])
        assert room_id >= 0
        signature = templates[key]
        footprint = [(ar + dr, ac + dc, color) for dr, dc, color in signature]
        region = regions[room_id]
        for r, c, color in footprint:
            assert _inside(region['mask'], r, c) and region['before'][r, c] >= 1, "clipped or wall-touching stamp"
            assert not forbidden[r, c], "touching or overlapping islands"
            assert arena[r, c] == (key if (r, c) == (ar, ac) else neutral), "stamp overwrites a cue or anchor"
            assert (r, c) not in region['key_cells']
        for r, c, color in footprint:
            solid[r, c] = stamp_mask[r, c] = True
            stamped[r, c] = color
            forbidden[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True
        placements.append((ar, ac, key, room_id))
    output = stamped.copy()
    for region in regions:
        remaining = region['mask'] & ~solid
        depths = _depth(remaining)
        assert np.any(remaining)
        for d, p in enumerate(region['key_cells']):
            assert depths[p] == d, "island changes a key's original depth"
        period = np.asarray(region['period'], dtype=np.int8)
        output[remaining] = period[depths[remaining] % len(period)]
        region.update(open=remaining, after=depths)
    return output, dict(grid=grid, box=(t, l, h, w), arena=arena, wall=wall,
                        neutral=neutral, exterior=exterior, templates=templates,
                        repetitions=repetitions, regions=regions, anchors=anchors,
                        placements=placements, walls=walls, solid=solid,
                        stamps=stamp_mask, stamped=stamped)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the completed, contour-colored arena using only visible input data."""
    return _reconstruct(input_grid)[0]


def _witnesses(output, info):
    witnesses = dict(contours=False, no_islands=False, singletons=False,
                     manhattan=False, rectangle=False, repeats=False, cavity=False)
    neutral = info['neutral']
    witnesses['contours'] = bool(np.any((output != info['stamped']) & ~info['solid']))
    for region in info['regions']:
        p = np.asarray(region['period'])
        room, open_cells = region['mask'], region['open']
        witnesses['repeats'] |= len(p) >= 3 and int(region['after'].max()) >= len(p)
        compare = open_cells & (info['arena'] == neutral)
        for cell in region['key_cells']:
            compare[cell] = False
        single_open = room.copy()
        for ar, ac, _ in info['anchors']:
            single_open[ar, ac] = False
        rr, cc = np.nonzero(room)
        rect = np.full(room.shape, -1, dtype=np.int16)
        rect[rr, cc] = np.minimum.reduce((rr - rr.min(), rr.max() - rr, cc - cc.min(), cc.max() - cc))
        alternatives = {'no_islands': region['before'], 'singletons': _depth(single_open),
                        'manhattan': _depth(open_cells, False), 'rectangle': rect}
        for name, depths in alternatives.items():
            witnesses[name] |= bool(np.any(output[compare] != p[depths[compare] % len(p)]))
        for component in _components(open_cells):
            # Closed cavities are separated from the original room's boundary.
            if all(region['before'][r, c] > 0 for r, c in component):
                witnesses['cavity'] |= p[0] != neutral and any(output[r, c] != neutral for r, c in component)
    return witnesses


def _template(rng, key, wall, extent, ring=False, demonstrate=False):
    for _ in range(100):
        if ring:
            h = 3 if extent == 4 else int(rng.integers(3, 6))
            w = 3 if extent == 4 else int(rng.integers(3, 6))
            occupied = {(r, c) for r in range(h) for c in range(w)
                        if r in (0, h - 1) or c in (0, w - 1)}
        else:
            h, w = map(int, rng.integers(2, extent + 1, size=2))
            n = int(rng.integers(4 if demonstrate else 3, min(12, h * w) + 1))
            occupied = {(int(rng.integers(h)), int(rng.integers(w)))}
            style = int(rng.integers(3))
            while len(occupied) < n:
                frontier = sorted({(r + dr, c + dc) for r, c in occupied for dr, dc in D4
                                   if 0 <= r + dr < h and 0 <= c + dc < w} - occupied)
                neighbors = np.asarray([sum((r + dr, c + dc) in occupied for dr, dc in D4)
                                        for r, c in frontier], dtype=float)
                weights = neighbors ** 2 if style == 0 else 1 / neighbors if style == 1 else np.ones(len(frontier))
                occupied.add(frontier[int(rng.choice(len(frontier), p=weights / weights.sum()))])
        cells = sorted(occupied)
        extremal = [p for p in cells if p[0] in (min(r for r, _ in cells), max(r for r, _ in cells))
                    or p[1] in (min(c for _, c in cells), max(c for _, c in cells))]
        ar, ac = _pick(rng, extremal if rng.random() < .6 else cells)
        sig = tuple(sorted((r - ar, c - ac, key if (r, c) == (ar, ac) else wall) for r, c in cells))
        if not demonstrate or (_asymmetric(sig) and np.prod(_bounds(sig)[2:]) > len(sig)):
            return sig
    # A visibly nonsolid asymmetric connected mask, after bounded shape search.
    return ((0, 0, key), (1, 0, wall), (1, 1, wall), (2, 0, wall))


def _wall_layout(rng, h, w, budget, steps, simple=False):
    mask = np.zeros((h, w), dtype=bool)
    mask[0] = mask[-1] = True
    mask[:, 0] = mask[:, -1] = True
    rooms = [(1, h - 2, 1, w - 2)]
    while len(rooms) < budget:
        options = []
        for i, (t, b, l, r) in enumerate(rooms):
            if b - t + 1 >= 15:
                options.append((i, 0))
            if r - l + 1 >= 15:
                options.append((i, 1))
        if not options:
            break
        if simple:
            i, axis = max(options, key=lambda q: (rooms[q[0]][1] - rooms[q[0]][0] + 1)
                         if q[1] == 0 else (rooms[q[0]][3] - rooms[q[0]][2] + 1))
        else:
            i, axis = _pick(rng, options)
        t, b, l, r = rooms.pop(i)
        if axis == 0:
            cut = (t + b) // 2 if simple else int(rng.integers(t + 7, b - 6))
            mask[cut, l - 1:r + 2] = True
            rooms += [(t, cut - 1, l, r), (cut + 1, b, l, r)]
        else:
            cut = (l + r) // 2 if simple else int(rng.integers(l + 7, r - 6))
            mask[t - 1:b + 2, cut] = True
            rooms += [(t, b, l, cut - 1), (t, b, cut + 1, r)]
    if not simple:
        for _ in range(steps):
            for _ in range(60):
                work = mask.T if rng.integers(2) else mask
                rr, cc = np.nonzero(work[1:-1, 1:-1])
                if not len(rr):
                    break
                j = int(rng.integers(len(rr)))
                top, col = int(rr[j] + 1), int(cc[j] + 1)
                end = top + int(rng.integers(4, 9))
                shift = int(_pick(rng, (-2, -1, 1, 2)))
                lo, hi = sorted((col, col + shift))
                if top < 2 or end + 2 >= work.shape[0] or lo < 2 or hi + 2 >= work.shape[1]:
                    continue
                block = work[top - 1:end + 2, lo - 1:hi + 2]
                expected = np.zeros(block.shape, dtype=bool)
                expected[:, col - lo + 1] = True
                if not np.array_equal(block, expected):
                    continue
                work[top + 1:end, col] = False
                work[top, lo:hi + 1] = work[end, lo:hi + 1] = True
                work[top:end + 1, col + shift] = True
                break
    return mask


def _place_targets(rng, arena, templates, target_ids, regions, ring_room=None):
    occupied = np.zeros(arena.shape, dtype=bool)
    places = []
    for occurrence, key in enumerate(target_ids):
        candidates = []
        for i, region in enumerate(regions):
            if occurrence == 0 and ring_room is not None and i != ring_room:
                continue
            for ar, ac in np.argwhere(region['mask']).tolist():
                footprint = [(ar + dr, ac + dc) for dr, dc, _ in templates[key]]
                if not all(_inside(region['mask'], r, c) and region['before'][r, c] >= 1
                           and not occupied[r, c] for r, c in footprint):
                    continue
                if any(max(abs(r - kr), abs(c - kc)) < d + 1
                       for r, c in footprint for d, (kr, kc) in enumerate(region['key_cells'])):
                    continue
                candidates.append((ar, ac, i, footprint))
        if not candidates:
            raise AssertionError("not enough separated island space")
        ar, ac, i, footprint = _pick(rng, candidates)
        places.append((ar, ac, key, i))
        for r, c in footprint:
            occupied[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True
    return places


def _make_pair(rng, split, index):
    first, second = split == 'train' and index == 0, split == 'train' and index == 1
    for attempt in range(48):
        simple = attempt >= 32
        ntypes = 1 if first else 2 if second else int(rng.integers(1 if split == 'train' else 2, 3 if split == 'train' else 4))
        if split == 'test' and simple:
            ntypes = 3
        colors = [int(c) for c in rng.permutation(10)]
        wall, neutral = colors[:2]
        exterior = colors[2] if first or rng.random() < .7 else neutral
        keys = colors[3:3 + ntypes]
        band_colors = [c for c in colors if c not in (wall, neutral, exterior) and c not in keys]
        extent = 4 if split == 'train' else 5
        templates = {key: _template(rng, key, wall, extent,
                                   ring=(second and j == 0) or (not first and not second and rng.random() < .22),
                                   demonstrate=first)
                     for j, key in enumerate(keys)}
        if second and _geometry(templates[keys[0]]) == _geometry(templates[keys[1]]):
            continue
        ah = 22 if simple else int(rng.integers(18 if first or second else 16, 23))
        aw = 27 if simple else int(rng.integers(20 if second else 18, 29))
        budget = 1 if first else 2 if second else int(rng.integers(1 if split == 'train' else 2, 4 if split == 'train' else 5))
        walls = _wall_layout(rng, ah, aw, budget, 0 if first else int(rng.integers(0, 3 if split == 'train' else 5)), simple)
        rooms = sorted(_masks(~walls), key=lambda m: -int(m.sum()))
        if split == 'test' and not (ntypes == 3 or len(rooms) == 4
                or any(max(_bounds(sig)[2:]) == 5 for sig in templates.values())):
            continue
        arena = np.full((ah, aw), neutral, dtype=np.int8)
        arena[walls] = wall
        regions = []
        try:
            for j, room in enumerate(rooms):
                before = _erosion(room)
                if first:
                    period = (band_colors[0], neutral, band_colors[1])
                elif second:
                    period = (band_colors[0], band_colors[1]) if j == 0 else (neutral,)
                elif j and rng.random() < .2:
                    period = (neutral,)
                else:
                    length = int(rng.integers(2 if j == 0 else 1, 4))
                    period = [int(c) for c in rng.choice(band_colors, size=length, replace=rng.random() < .2)]
                    if length >= 2 and rng.random() < .35:
                        period[int(rng.integers(length - 1))] = neutral
                    period = tuple(period)
                if period == (neutral,):
                    cells, directions = (), ()
                else:
                    cells, directions = _pick(rng, _paths(room, before, len(period)))
                    for p, color in zip(cells, period):
                        arena[p] = color
                regions.append(dict(mask=room, before=before, period=period, key_cells=cells))
            target_ids = keys * 2 if first else keys.copy()
            if not first and attempt < 24:
                target_ids += [int(_pick(rng, keys)) for _ in range(int(rng.integers(3)))]
            if not second:
                rng.shuffle(target_ids)
            places = _place_targets(rng, arena, templates, target_ids, regions, 0 if second else None)
            # Labels use sampled footprints/periods and iterative erosion, not solve.
            output = arena.copy()
            solid = walls.copy()
            for ar, ac, key, _ in places:
                arena[ar, ac] = key
                for dr, dc, color in templates[key]:
                    output[ar + dr, ac + dc] = color
                    solid[ar + dr, ac + dc] = True
            for region in regions:
                open_cells = region['mask'] & ~solid
                depths = _erosion(open_cells)
                period = np.asarray(region['period'], dtype=np.int8)
                output[open_cells] = period[depths[open_cells] % len(period)]
            ref_ids = keys * 2 if first else keys.copy()
            if not first and rng.random() < .5:
                ref_ids.append(int(_pick(rng, keys)))
            rng.shuffle(ref_ids)
            sizes = [_bounds(templates[key]) for key in ref_ids]
            shelf_h = max(h for _, _, h, _ in sizes)
            shelf_w = sum(w for _, _, _, w in sizes) + len(sizes) - 1
            width = max(aw + 2, shelf_w + 2)
            height = shelf_h + ah + 3
            assert width <= 30 and height <= 30
            height += int(rng.integers(min(2, 30 - height) + 1))
            width += int(rng.integers(min(2, 30 - width) + 1))
            grid = np.full((height, width), exterior, dtype=np.int8)
            top = shelf_h + 2
            left = int(rng.integers(1, width - aw))
            grid[top:top + ah, left:left + aw] = arena
            x = int(rng.integers(1, width - shelf_w))
            for key, (lr, lc, h, w) in zip(ref_ids, sizes):
                y = int(rng.integers(1, shelf_h - h + 2))
                for dr, dc, color in templates[key]:
                    grid[y + dr - lr, x + dc - lc] = color
                x += w + 1
            turns, reflect = int(rng.integers(4)), bool(rng.integers(2))
            grid, output = np.rot90(grid, turns).copy(), np.rot90(output, turns).copy()
            if reflect:
                grid, output = grid[:, ::-1].copy(), output[:, ::-1].copy()
            predicted, info = _reconstruct(grid)
            assert np.array_equal(predicted, output)
            evidence = _witnesses(output, info)
            if first:
                assert all(evidence[k] for k in ('contours', 'no_islands', 'singletons', 'manhattan', 'rectangle', 'repeats'))
            if second:
                assert evidence['cavity']
            return {'input': grid, 'output': output}
        except (AssertionError, IndexError):
            if attempt == 47:
                raise
    raise AssertionError("bounded construction exhausted")


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")


def _grid_key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p['input']), _grid_key(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(count):
            for _ in range(8):
                pair = _make_pair(rng, split, index)
                key = _grid_key(pair['input'])
                if key not in seen:
                    break
            assert key not in seen, "duplicate input after bounded resampling"
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, output = _grid(pair['input']), _grid(pair['output'])
            assert _grid_key(inp) not in seen
            seen.add(_grid_key(inp))
            predicted, info = _reconstruct(inp)
            assert np.array_equal(predicted, output), "incorrect reconstructed contour field"
            assert np.array_equal(output[info['walls']], info['arena'][info['walls']]), "wall changed"
            for ar, ac, key in info['anchors']:
                assert output[ar, ac] == key, "anchor changed"
            assert output.shape == info['arena'].shape and output.size < inp.size
            assert set(np.unique(output)) <= set(np.unique(inp))
            for region in info['regions']:
                assert np.array_equal(region['before'], _erosion(region['mask']))
                assert np.array_equal(region['after'], _erosion(region['open']))
                p = np.asarray(region['period'])
                assert np.array_equal(output[region['open']], p[region['after'][region['open']] % len(p)])
            analyses[split].append((output, info, _witnesses(output, info)))
    first_out, first, witness = analyses['train'][0]
    assert all(witness[k] for k in ('contours', 'no_islands', 'singletons', 'manhattan', 'rectangle', 'repeats'))
    assert len(first['regions']) == 1 and len(first['templates']) == 1 and len(first['anchors']) == 2
    assert next(iter(first['repetitions'].values())) == 2
    sig = next(iter(first['templates'].values()))
    assert _asymmetric(sig) and np.prod(_bounds(sig)[2:]) > len(sig)
    assert first['exterior'] != first['neutral']
    assert first['regions'][0]['period'][1] == first['neutral']
    _, second, second_witness = analyses['train'][1]
    assert len(second['templates']) == 2 and len(second['regions']) >= 2
    assert len({_geometry(s) for s in second['templates'].values()}) == 2
    assert any(_hollow(s) for s in second['templates'].values()) and second_witness['cavity']
    assert any(r['period'] == (second['neutral'],) for r in second['regions'])
    assert set(second['templates']) <= {k for _, _, k in second['anchors']}
    assert all(2 <= len(info['templates']) <= 3 and 2 <= len(info['regions']) <= 4
               for _, info, _ in analyses['test'])
    assert all(len(info['templates']) == 3 or len(info['regions']) == 4
               or any(max(_bounds(sig)[2:]) == 5 for sig in info['templates'].values())
               for _, info, _ in analyses['test']), "test must combine a new structural capacity"


def self_test():
    """Bounded 42-seed semantic/diversity audit, plus API and corruption checks."""
    shapes, geometry, room_geometry, profiles, roles, periods, directions = (set() for _ in range(7))
    type_counts, room_counts, sizes, multiplicities, repeats = (set() for _ in range(5))
    puzzle_keys, hollow_count, equal_fills, stepped = set(), 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzle_keys.add(_puzzle_key(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                output, info = _reconstruct(pair['input'])
                assert np.array_equal(solve(pair['input']), pair['output'])
                shapes.add((info['grid'].shape, output.shape))
                roles.add((info['wall'], info['neutral'], info['exterior']))
                type_counts.add(len(info['templates']))
                room_counts.add(len(info['regions']))
                multiplicities.add(len(info['anchors']))
                repeats.update(info['repetitions'].values())
                equal_fills += info['exterior'] == info['neutral']
                local_geom = tuple(sorted(_geometry(sig) for sig in info['templates'].values()))
                geometry.update(local_geom)
                sizes.update(len(sig) for sig in info['templates'].values())
                hollow_count += sum(_hollow(sig) for sig in info['templates'].values())
                for region in info['regions']:
                    periods.add(len(region['period']))
                    directions.update(region['directions'])
                    rr, cc = np.nonzero(region['mask'])
                    concave = bool(int(region['mask'].sum()) < (rr.max() - rr.min() + 1) * (cc.max() - cc.min() + 1))
                    stepped += concave
                    room_geometry.add(_grid_key(region['mask']))
                profiles.add((len(info['regions']), local_geom, len(info['anchors']),
                              tuple(sorted(len(r['period']) for r in info['regions']))))
    assert len(puzzle_keys) == 42
    assert len(shapes) >= 80 and len(geometry) >= 100 and len(room_geometry) >= 150
    assert len(profiles) >= 100 and len(roles) >= 100
    assert type_counts == {1, 2, 3} and room_counts == {1, 2, 3, 4}
    assert periods == {1, 2, 3} and directions == {(-1, -1), (-1, 1), (1, -1), (1, 1)}
    assert {3, 4, 5, 6, 7, 8, 9, 10, 12} <= sizes
    assert {1, 2, 3, 4} <= multiplicities and {1, 2} <= repeats
    assert hollow_count >= 42 and equal_fills >= 15 and stepped >= 15
    assert any(30 in shape for shape, _ in shapes)
    assert _puzzle_key(generate_puzzle(111, 2, 1)) == _puzzle_key(generate_puzzle(111, 2, 1))
    for nt, ne in ((2, 1), (5, 4), (10, 1), (10, 4)):
        p = generate_puzzle(99 + nt + ne, nt, ne)
        assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.5, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    good = generate_puzzle(42, 2, 1)
    broken = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in good[s]] for s in good}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt frame label accepted")
    # A template disagreement is detected using only the two external copies.
    _, info = _reconstruct(good['train'][0]['input'])
    bad_input = good['train'][0]['input'].copy()
    t, l, h, w = info['box']
    outside = np.ones(bad_input.shape, dtype=bool)
    outside[t:t + h, l:l + w] = False
    wall_cells = np.argwhere(outside & (bad_input == info['wall']))
    r, c = wall_cells[0]
    bad_input[r, c] = info['exterior']
    try:
        solve(bad_input)
    except AssertionError:
        pass
    else:
        raise AssertionError("damaged reference accepted")
    return {'seeds': 42, 'distinct_puzzles': len(puzzle_keys), 'shape_pairs': len(shapes),
            'template_geometries': len(geometry), 'room_geometries': len(room_geometry),
            'structural_profiles': len(profiles), 'role_assignments': len(roles),
            'motif_sizes': sorted(sizes), 'room_counts': sorted(room_counts),
            'target_counts': sorted(multiplicities), 'hollow_templates': hollow_count,
            'concave_rooms': stepped, 'equal_fill_pairs': equal_fills}


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
