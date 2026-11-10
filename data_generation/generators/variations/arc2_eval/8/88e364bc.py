"""Color-matched direction legends and enclosed, corner-aware marker motion.

Only marker and terminal-tip colors are puzzle conventions. Everything else is
read locally. Construction uses masks; validation reconstructs keys and enclosed
free-space components from the returned grids, without construction metadata.
"""
import hashlib
import numpy as np

_ALL_DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                   (0, 1), (1, -1), (1, 0), (1, 1))
_CARDINAL = tuple(d for d in _ALL_DIRECTIONS if 0 in d)
_DIAGONAL = tuple(d for d in _ALL_DIRECTIONS if 0 not in d)
_SOURCE_HASHES = {
    'c880246cdaa3be542a52d4d57bf5ab68cd74cdad65562ed072baed9e178ee452',
    '53be09a1ccb92995011f206ba589e6c965563dcadc3c59ea2a2d96ae7eca4326',
    '06b53baf42accfaa6a16e2e112be0721338cd2ee0c4ae2ae5e80e9d90ac10247',
    'c8a1f114a62ffe27e0697d4df03a3c320077556d6eecb732593114dfc8512b9d',
    '0ce1c656eda6ae542e17f1b99107f20bfa953386b0e1257d7010fb8ed0e622dd',
}


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _validated_integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be between {low} and {high}')
    return int(value)


def _resolve_counts(seed, rng, num_train, num_test):
    if num_train is not None:
        num_train = _validated_integer(num_train, 'num_train', 2, 10)
    if num_test is not None:
        num_test = _validated_integer(num_test, 'num_test', 1, 4)
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError('num_train must exceed num_test')
    elif num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    else:
        num_test = int(rng.integers(1, min(num_train - 1, 4) + 1))
    return num_train, num_test


def _grid_fingerprint(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _puzzle_fingerprint(puzzle):
    return tuple(tuple((_grid_fingerprint(p['input']), _grid_fingerprint(p['output']))
                       for p in puzzle[s]) for s in ('train', 'test'))


def _source_copy(pair):
    if pair['input'].shape != (20, 20):
        return False
    data = b''.join(pair[k].astype(np.uint8).tobytes() for k in ('input', 'output'))
    return hashlib.sha256(data).hexdigest() in _SOURCE_HASHES


def _components(mask):
    height, width = mask.shape
    seen = ~mask.copy()
    result = []
    for rr, cc in np.argwhere(mask):
        r, c = int(rr), int(cc)
        if seen[r, c]:
            continue
        stack, cells = [(r, c)], []
        seen[r, c] = True
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in _CARDINAL:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        result.append(cells)
    return result


def _dilate(mask, diagonal=False):
    padded = np.pad(mask, 1)
    result = mask.copy()
    for dr, dc in (_ALL_DIRECTIONS if diagonal else _CARDINAL):
        result |= padded[1 + dr:1 + dr + mask.shape[0],
                         1 + dc:1 + dc + mask.shape[1]]
    return result


def _bbox(cells):
    rows, cols = zip(*cells)
    return min(rows), max(rows), min(cols), max(cols)


def _decode_keys(grid, background, marker):
    keys = []
    for cells in _components((grid != background) & (grid != marker)):
        r0, r1, c0, c1 = _bbox(cells)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if h < 4 or w < 4 or len(cells) != h * w:
            continue
        sub = grid[r0:r1 + 1, c0:c1 + 1]
        carrier = int(sub[0, 0])
        border = np.concatenate((sub[0], sub[-1], sub[:, 0], sub[:, -1]))
        others = set(map(int, np.unique(sub))) - {carrier}
        if len(others) != 2 or np.any(border != carrier):
            continue
        interpretations = []
        for tip in others:
            trail = next(iter(others - {tip}))
            tips = set(map(tuple, np.argwhere(sub == tip)))
            trails = set(map(tuple, np.argwhere(sub == trail)))
            if len(tips) < 2:
                continue
            for dr, dc in _ALL_DIRECTIONS:
                used, lengths = set(), []
                for r, c in tips:
                    stroke = []
                    r, c = r - dr, c - dc
                    while (r, c) in trails:
                        stroke.append((r, c))
                        r, c = r - dr, c - dc
                    if len(stroke) < 2 or used.intersection(stroke):
                        break
                    used.update(stroke)
                    lengths.append(len(stroke))
                if len(lengths) == len(tips) and used == trails:
                    interpretations.append((tip, trail, (dr, dc), tuple(lengths)))
        _require(len(interpretations) == 1, 'key must have a unique trail-to-tip interpretation')
        tip, trail, direction, lengths = interpretations[0]
        keys.append(dict(bbox=(r0, r1, c0, c1), structural=carrier, tip=tip,
                         trail=trail, direction=direction, lengths=lengths,
                         strokes=len(lengths), shape=(h, w)))
    _require(keys, 'no direction keys')
    _require(len({k['structural'] for k in keys}) == len(keys), 'conflicting carrier keys')
    return keys


def _read_input(grid, marker, tip=None, background=None):
    # The canvas background is the most frequent local color; it can change
    # between examples. It is never reused by any local object role.
    if background is None:
        background = int(np.bincount(grid.ravel(), minlength=10).argmax())
    _require(background != marker, 'ambiguous background')
    keys = _decode_keys(grid, background, marker)
    tips = {k['tip'] for k in keys}
    _require(len(tips) == 1 and (tip is None or tips == {tip}), 'terminal convention changed')
    occupied_keys = np.zeros(grid.shape, bool)
    for key in keys:
        r0, r1, c0, c1 = key['bbox']
        occupied_keys[r0:r1 + 1, c0:c1 + 1] = True
    carriers = {k['structural'] for k in keys}
    wall_mask = (grid != background) & (grid != marker) & ~occupied_keys
    _require(set(map(int, np.unique(grid[wall_mask]))) == carriers, 'unexplained walls or unused key')
    regions = []
    for cells in _components((grid == background) | (grid == marker)):
        boundary, edge = set(), False
        for r, c in cells:
            for dr, dc in _CARDINAL:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
                    edge = True
                elif grid[nr, nc] not in (background, marker):
                    boundary.add(int(grid[nr, nc]))
        starts = [p for p in cells if grid[p] == marker]
        enclosed = not edge and len(boundary) == 1 and boundary <= carriers
        if starts:
            _require(enclosed, 'marker lacks a unique enclosing wall color')
        if enclosed:
            regions.append(dict(cells=set(cells), structural=next(iter(boundary)), starts=starts))
    _require(sum(len(r['starts']) for r in regions) == int(np.count_nonzero(grid == marker)),
             'unassociated marker')
    _require(all(any(r['starts'] and r['structural'] == k['structural'] for r in regions)
                 for k in keys), 'every system needs a marker')
    return dict(background=background, marker=marker, tip=next(iter(tips)), keys=keys,
                regions=regions, walls=wall_mask)


def _predict_read(grid, info, corner=True, reverse=False):
    """Reconstruct rays using region membership, independently of wall masks."""
    expected = grid.copy()
    expected[grid == info['marker']] = info['background']
    paths, distances, witnesses, used = [], [], [], set()
    directions = {k['structural']: k['direction'] for k in info['keys']}
    for region in info['regions']:
        dr, dc = directions[region['structural']]
        if reverse:
            dr, dc = -dr, -dc
        free = region['cells']
        for start in region['starts']:
            r, c = start
            path = [(r, c)]
            while (r + dr, c + dc) in free:
                if corner and dr and dc and ((r + dr, c) not in free or (r, c + dc) not in free):
                    break
                r, c = r + dr, c + dc
                path.append((r, c))
            # A witness distinguishes the correct stop from a next-cell-only ray.
            nr, nc = r + dr, c + dc
            witness = bool(dr and dc and 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]
                           and grid[nr, nc] in (info['background'], info['marker']))
            _require(used.isdisjoint(path), 'interacting marker trajectories')
            used.update(path)
            expected[r, c] = info['marker']
            paths.append(path)
            distances.append(len(path) - 1)
            witnesses.append(witness)
    info.update(paths=paths, distances=distances, corner_witnesses=witnesses)
    return expected


def _learn(train_pairs):
    markers, backgrounds, analyses = set(), [], []
    for pair in train_pairs:
        a, b = pair['input'], pair['output']
        changed = a != b
        values = list(map(int, np.unique(a[changed])))
        _require(len(values) == 2, 'training must demonstrate marker motion')
        bg, marker = sorted(values, key=lambda x: -int(np.count_nonzero(a == x)))
        _require(np.count_nonzero(a == bg) > np.count_nonzero(a == marker), 'unclear roles')
        info = _read_input(a, marker, background=bg)
        _require(np.array_equal(_predict_read(a, info), b), 'training rule mismatch')
        markers.add(marker)
        backgrounds.append(bg)
        analyses.append(info)
    _require(len(markers) == 1, 'movable color changed across training')
    tips = {a['tip'] for a in analyses}
    _require(len(tips) == 1, 'tip color changed across training')
    return next(iter(markers)), next(iter(tips)), analyses


def solve(train_pairs, test_input):
    """Predict from demonstrations and an unlabeled input, never from RNG state."""
    marker, tip, _ = _learn(train_pairs)
    info = _read_input(test_input, marker, tip)
    return _predict_read(test_input, info)


def _trace_wall_mask(wall, start, direction, corner=True):
    r, c = start
    dr, dc = direction
    path = [(r, c)]
    for _ in range(max(wall.shape)):
        nr, nc = r + dr, c + dc
        if not (0 <= nr < wall.shape[0] and 0 <= nc < wall.shape[1]):
            return None
        if wall[nr, nc] or (corner and dr and dc and (wall[nr, c] or wall[r, nc])):
            return path
        r, c = nr, nc
        path.append((r, c))
    return None


def _make_key(rng, direction, carrier, trail, tip, compact=False):
    dr, dc = direction
    # Two to four parallel strokes on separated lines, with independent offsets.
    count = 2 if compact else int(rng.choice([2, 2, 2, 3, 4]))
    length = 2 if compact else int(rng.choice([2, 2, 2, 3, 4]))
    points, offset = [], 0
    for i in range(count):
        stagger = int(rng.integers(0, 3)) if not compact else 0
        # Horizontal offset for diagonal strokes keeps the carrier compact.
        base = (0, offset) if dr else (offset, 0)
        for j in range(length + 1):
            points.append((base[0] + (j + stagger) * dr,
                           base[1] + (j + stagger) * dc,
                           tip if j == length else trail))
        offset += 2 if compact else int(rng.choice([2, 2, 3]))
    r0 = min(p[0] for p in points)
    c0 = min(p[1] for p in points)
    r1 = max(p[0] for p in points)
    c1 = max(p[1] for p in points)
    padding = [1] * 4 if compact else [int(rng.choice([1, 1, 1, 2])) for _ in range(4)]
    h, w = r1 - r0 + 1 + sum(padding[:2]), c1 - c0 + 1 + sum(padding[2:])
    key = np.full((h, w), carrier, np.int8)
    for r, c, color in points:
        key[r - r0 + padding[0], c - c0 + padding[2]] = color
    return key


def _footprint(rng, h, w, kind):
    filled = np.ones((h, w), bool)
    if kind == 0:
        return filled
    if kind == 1:
        # Concave corner cutouts and unequal lobes.
        for corner in rng.choice(4, int(rng.integers(1, 4)), replace=False):
            dh, dw = int(rng.integers(1, max(2, h // 2))), int(rng.integers(1, max(2, w // 2)))
            rs = slice(0, dh) if corner < 2 else slice(h - dh, h)
            cs = slice(0, dw) if corner % 2 == 0 else slice(w - dw, w)
            filled[rs, cs] = False
    elif kind == 2:
        # Smooth raster slopes; the resulting wall may have diagonal contacts.
        yy, xx = np.indices((h, w))
        cy, cx = (h - 1) / 2, (w - 1) / 2
        exponent = float(rng.uniform(1.1, 2.2))
        filled = (abs((yy - cy) / (cy + .4)) ** exponent
                  + abs((xx - cx) / (cx + .4)) ** exponent <= 1.2)
    else:
        # Independently wandering left and right shores create necks and bays.
        left, right = int(rng.integers(0, max(1, w // 3))), int(rng.integers(0, max(1, w // 3)))
        for r in range(h):
            if rng.random() < .65:
                left = int(np.clip(left + rng.integers(-2, 3), 0, max(0, w // 2 - 2)))
                right = int(np.clip(right + rng.integers(-2, 3), 0, max(0, w // 2 - 2)))
            filled[r, :left] = False
            if right:
                filled[r, w - right:] = False
    return filled


def _enclosure(rng, h, w, direction, desired, complexity, witness=False, fallback=False):
    for attempt in range(45):
        kind = 0 if fallback or attempt == 44 else int(rng.choice(4, p=[.23, .26, .24, .27]))
        footprint = _footprint(rng, h, w, kind)
        padded = np.pad(footprint, 1)
        interior = footprint.copy()
        for dr, dc in _CARDINAL:
            interior &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        wall = footprint & ~interior
        baffles = int(rng.integers(0, complexity + 1))
        if witness and (fallback or attempt == 44):
            # A top-connected stop with a free diagonal cell beyond its tip.
            wall[:] = False
            wall[0] = wall[-1] = True
            wall[:, 0] = wall[:, -1] = True
            wall[:h // 2 + 1, w // 2] = True
            if direction[0] < 0:
                wall = np.flipud(wall).copy()
            if direction[1] < 0:
                wall = np.fliplr(wall).copy()
            footprint[:] = True
        else:
            for _ in range(baffles):
                horizontal = bool(rng.integers(2))
                if horizontal:
                    row = int(rng.integers(2, h - 2))
                    if rng.random() < .4:
                        wall[row, :] |= footprint[row, :]
                    else:
                        cut = int(rng.integers(2, w - 1))
                        cols = slice(0, cut) if rng.random() < .5 else slice(cut, w)
                        wall[row, cols] |= footprint[row, cols]
                else:
                    col = int(rng.integers(2, w - 2))
                    if rng.random() < .4:
                        wall[:, col] |= footprint[:, col]
                    else:
                        cut = int(rng.integers(2, h - 1))
                        rows = slice(0, cut) if rng.random() < .5 else slice(cut, h)
                        wall[rows, col] |= footprint[rows, col]
        free = footprint & ~wall
        labels = np.full(free.shape, -1, dtype=int)
        for label, cells in enumerate(_components(free)):
            for cell in cells:
                labels[cell] = label
        candidates = []
        for rr, cc in np.argwhere(free):
            start = int(rr), int(cc)
            path = _trace_wall_mask(wall, start, direction)
            if path is None or any(not free[p] for p in path):
                continue
            nr, nc = path[-1][0] + direction[0], path[-1][1] + direction[1]
            corner = bool(direction[0] and direction[1] and 0 <= nr < h and 0 <= nc < w
                          and free[nr, nc] and labels[nr, nc] == labels[start])
            candidates.append((path, corner))
        moving = [i for i, (p, _) in enumerate(candidates) if len(p) > 1]
        witnesses = [i for i in moving if candidates[i][1]]
        if not moving or (witness and not witnesses):
            continue
        selected, used = [], set()
        priority = [int(rng.choice(witnesses if witness else moving))]
        if desired > 1 and rng.random() < .12:
            stationary = [i for i, (p, _) in enumerate(candidates) if len(p) == 1]
            if stationary:
                priority.append(int(rng.choice(stationary)))
        if rng.random() < .35:
            short = [i for i, (p, _) in enumerate(candidates) if len(p) == 2]
            if short:
                priority.append(int(rng.choice(short)))
        # Stationary markers are an occasional explicit extension; do not let
        # the many boundary cells make them dominate the random candidate pool.
        priority += [moving[int(i)] for i in rng.permutation(len(moving))]
        for index in priority:
            path, _ = candidates[index]
            if used.isdisjoint(path):
                selected.append(path)
                used.update(path)
            if len(selected) == desired:
                break
        if selected:
            return wall, footprint, selected
    raise AssertionError('bounded enclosure construction failed')


def _place(rng, canvas_shape, objects):
    occupied = np.zeros(canvas_shape, bool)
    placed = []
    # Place enclosures first, allowing silhouettes to interlock, then place keys
    # in any remaining location with a visible background gap around each key.
    for obj in objects:
        mask = obj['mask']
        h, w = mask.shape
        if h > canvas_shape[0] or w > canvas_shape[1]:
            return None
        obstacles = _dilate(occupied, True) if obj['key'] else occupied
        windows = np.lib.stride_tricks.sliding_window_view(obstacles, (h, w))
        valid = ~np.any(windows & mask, axis=(-1, -2))
        candidates = np.argwhere(valid)
        if not len(candidates):
            return None
        if not obj['key'] and placed and rng.random() < .6:
            overlaps = []
            for rr, cc in candidates:
                overlaps.append(any(rr < r + old['mask'].shape[0] and r < rr + h
                                    and cc < c + old['mask'].shape[1] and c < cc + w
                                    for old, r, c in placed if not old['key']))
            if any(overlaps):
                candidates = candidates[np.array(overlaps)]
        r, c = map(int, candidates[int(rng.integers(len(candidates)))])
        occupied[r:r + h, c:c + w] |= mask
        placed.append((obj, r, c))
    return placed


def _generate_pair(rng, directions, marker, tip, regime, complexity, witness_index=None,
                   excluded_background=None):
    n = len(directions)
    available = [x for x in range(10) if x not in (marker, tip, excluded_background)]
    background = int(rng.choice(available))
    carriers = list(map(int, rng.choice([x for x in range(10) if x not in (marker, tip, background)], n, replace=False)))
    independent = [x for x in range(10) if x not in carriers + [marker, tip, background]]
    shared = int(rng.choice(independent))
    color_mode = int(rng.integers(3))
    trails = []
    for carrier in carriers:
        choices = independent if color_mode == 1 else [x for x in carriers if x != carrier]
        trails.append(shared if color_mode == 0 or not choices else int(rng.choice(choices)))
    desired = [int(rng.choice([1, 1, 2, 2, 3, 4, 5])) for _ in directions]
    size_mode = int(rng.integers(5))
    for attempt in range(36):
        fallback = attempt == 35
        compact = fallback or n == 3 or (size_mode == 0 and attempt > 7)
        if fallback:
            height, width = 30, 30
        elif size_mode == 0:
            height = width = 20
        elif size_mode == 1:
            height, width = 30, int(rng.integers(22, 31))
        else:
            low = 15 if n == 1 else (20 if n == 2 else 26)
            height, width = int(rng.integers(low, 31)), int(rng.integers(low, 31))
        if n == 1:
            max_h, max_w = min(height - 3, 23), min(width - 3, 23)
        elif n == 2:
            max_h, max_w = min(height - 8, 17), min(width - 8, 17)
        else:
            max_h, max_w = 11, 11
        walls, keys = [], []
        for i, direction in enumerate(directions):
            key = _make_key(rng, direction, carriers[i], trails[i], tip, compact)
            h = 8 if fallback else int(rng.integers(6, max(7, max_h + 1)))
            w = 8 if fallback else int(rng.integers(6, max(7, max_w + 1)))
            # Preserve the compact-legend / larger-enclosure distinction while
            # still allowing rectangular keys and independently varied scales.
            while h * w < 1.5 * key.size and (h < max_h or w < max_w):
                if h < max_h and (w == max_w or rng.random() < .5):
                    h += 1
                else:
                    w += 1
            wall, footprint, paths = _enclosure(rng, h, w, direction, desired[i],
                                               complexity, i == witness_index, fallback)
            r0, r1, c0, c1 = _bbox(np.argwhere(wall))
            if (r1 - r0 + 1) * (c1 - c0 + 1) < 1.2 * key.size:
                key = _make_key(rng, direction, carriers[i], trails[i], tip, True)
            if (r1 - r0 + 1) * (c1 - c0 + 1) < 1.2 * key.size:
                wall, footprint, paths = _enclosure(rng, max(8, h), max(8, w), direction,
                                                   desired[i], complexity, i == witness_index, True)
            walls.append(dict(key=False, mask=footprint, wall=wall, paths=paths, color=carriers[i]))
            keys.append(dict(key=True, mask=np.ones(key.shape, bool), grid=key))
        # Independent permutations prevent position or key order from encoding association.
        objects = [walls[int(i)] for i in rng.permutation(n)] + [keys[int(i)] for i in rng.permutation(n)]
        placed = _place(rng, (height, width), objects)
        if placed is None and fallback:
            # A guaranteed arrangement for the bounded retry tail, with shuffled keys.
            placed = [(obj, 1, 1 + 10 * i) for i, obj in enumerate(objects[:n])]
            placed += [(obj, 14, 1 + 10 * i) for i, obj in enumerate(objects[n:])]
        if placed is None:
            continue
        a = np.full((height, width), background, np.int8)
        paths = []
        for obj, r, c in placed:
            h, w = obj['mask'].shape
            if obj['key']:
                a[r:r + h, c:c + w] = obj['grid']
            else:
                a[r:r + h, c:c + w][obj['wall']] = obj['color']
                paths += [[(rr + r, cc + c) for rr, cc in path] for path in obj['paths']]
        for path in paths:
            a[path[0]] = marker
        b = a.copy()
        b[a == marker] = background
        for path in paths:
            b[path[-1]] = marker
        # Randomly crop excess canvas, or retain independently sampled full canvas.
        if size_mode >= 2 and rng.random() < .5:
            cells = np.argwhere(a != background)
            r0, r1, c0, c1 = _bbox(cells)
            r0, c0 = max(0, r0 - int(rng.integers(3))), max(0, c0 - int(rng.integers(3)))
            r1, c1 = min(height - 1, r1 + int(rng.integers(3))), min(width - 1, c1 + int(rng.integers(3)))
            a, b = a[r0:r1 + 1, c0:c1 + 1].copy(), b[r0:r1 + 1, c0:c1 + 1].copy()
        if int(np.bincount(a.ravel(), minlength=10).argmax()) != background:
            continue
        pair = dict(input=a, output=b)
        if not _source_copy(pair):
            return pair
    raise AssertionError('bounded pair construction failed')


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    nt, nq = _resolve_counts(seed, rng, num_train, num_test)
    marker, tip = map(int, rng.choice(10, 2, replace=False))
    richness = int(rng.choice(3, p=[.18, .57, .25]))
    count_weights = ([.85, .15, 0], [.48, .48, .04], [.25, .6, .15])[richness]
    counts = list(map(int, rng.choice([1, 2, 3], nt + nq, p=count_weights)))
    cardinal_only = richness == 0 and rng.random() < .6
    choices = _CARDINAL if cardinal_only else _ALL_DIRECTIONS
    pool_size = min(sum(counts[:nt]), int(rng.integers(2, 5)))
    if cardinal_only:
        train_pool = [choices[int(i)] for i in rng.choice(len(choices), pool_size, replace=False)]
    else:
        train_pool = [_CARDINAL[int(rng.integers(4))], _DIAGONAL[int(rng.integers(4))]]
        remaining = [d for d in choices if d not in train_pool]
        train_pool += [remaining[int(i)] for i in rng.choice(len(remaining), pool_size - 2, replace=False)]
    slots = [train_pool[i % len(train_pool)] for i in range(sum(counts[:nt]))]
    # Fill remaining slots independently, then shuffle the whole demonstration order.
    for i in range(len(train_pool), len(slots)):
        slots[i] = train_pool[int(rng.integers(len(train_pool)))]
    slots = [slots[int(i)] for i in rng.permutation(len(slots))]
    mode = int(rng.integers(3))  # shared orientations, held-out reversals/rotations, or a mixture
    heldout = [d for d in choices if d not in train_pool]
    test_pool = train_pool if mode == 0 or not heldout else (heldout if mode == 1 else list(choices))
    plans, offset = [], 0
    for count in counts[:nt]:
        plans.append(slots[offset:offset + count])
        offset += count
    for count in counts[nt:]:
        plans.append([test_pool[int(rng.integers(len(test_pool)))] for _ in range(count)])
    diagonal_training = [(i, j) for i, ds in enumerate(plans[:nt]) for j, d in enumerate(ds) if 0 not in d]
    witness = diagonal_training[int(rng.integers(len(diagonal_training)))] if diagonal_training else None
    puzzle, seen = {'train': [], 'test': []}, set()
    previous_bg = None
    for index, directions in enumerate(plans):
        split = 'train' if index < nt else 'test'
        regime = 'source_train' if split == 'train' else 'target_test'
        # Test novelty also comes from local outlines and arrangements, with a
        # modest baffle-complexity shift rather than a prescribed direction list.
        complexity = int(rng.integers(0, 2 + richness + (split == 'test')))
        for _ in range(8):
            pair = _generate_pair(rng, directions, marker, tip, regime, complexity,
                                  witness[1] if witness is not None and witness[0] == index else None,
                                  previous_bg)
            fp = _grid_fingerprint(pair['input'])
            if fp not in seen:
                break
        _require(fp not in seen, 'duplicate construction')
        seen.add(fp)
        previous_bg = int(np.bincount(pair['input'].ravel(), minlength=10).argmax())
        puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _require(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
             and len(puzzle['train']) > len(puzzle['test']), 'invalid pair counts')
    seen = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair schema')
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, 'grid must be a 2D array')
                _require(np.issubdtype(grid.dtype, np.integer), 'noninteger grid')
                _require(all(1 <= s <= 30 for s in grid.shape), 'grid outside 1..30')
                _require(np.all((0 <= grid) & (grid <= 9)), 'invalid color')
            _require(pair['input'].shape == pair['output'].shape, 'changed dimensions')
            _require(not np.array_equal(pair['input'], pair['output']), 'each pair must show a move')
            fp = _grid_fingerprint(pair['input'])
            _require(fp not in seen and not _source_copy(pair), 'duplicate or source copy')
            seen.add(fp)
    marker, tip, training = _learn(puzzle['train'])
    train_dirs = {k['direction'] for a in training for k in a['keys']}
    _require(len(train_dirs) >= 2, 'training must distinguish key-controlled direction from constant motion')
    train_corner = any(any(a['corner_witnesses']) for a in training)
    train_diagonal = any(0 not in d for d in train_dirs)
    for pair in puzzle['test']:
        # No test output participates in role inference or prediction.
        info = _read_input(pair['input'], marker, tip)
        prediction = _predict_read(pair['input'], info)
        _require(np.array_equal(prediction, pair['output']), 'query rule mismatch')
        _require(not any(info['corner_witnesses']) or train_corner, 'missing corner-blocking demonstration')
        _require(not any(0 not in k['direction'] for k in info['keys']) or train_diagonal,
                 'missing diagonal demonstration')
    for pair in puzzle['train'] + puzzle['test']:
        a, b = pair['input'], pair['output']
        background = int(np.bincount(a.ravel(), minlength=10).argmax())
        _require(np.count_nonzero(a == marker) == np.count_nonzero(b == marker), 'marker count changed')
        fixed = (a != marker) & (a != background)
        _require(np.array_equal(a[fixed], b[fixed]), 'keys or walls changed')


def measure_puzzle(puzzle):
    """Review measurements reconstructed from actual grids, not sampled settings."""
    marker, tip, _ = _learn(puzzle['train'])
    records = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            a = pair['input']
            info = _read_input(a, marker, tip)
            _predict_read(a, info)
            carriers = {k['structural'] for k in info['keys']}
            boxes, irregular, wall_shapes, sides, wall_components = [], [], [], [], []
            paddings, staggered, internal_stops, narrow_cells = [], [], [], []
            for key in info['keys']:
                color = key['structural']
                mask = info['walls'] & (a == color)
                cells = list(map(tuple, np.argwhere(mask)))
                r0, r1, c0, c1 = _bbox(cells)
                boxes.append((r0, r1, c0, c1))
                wall_shapes.append((r1 - r0 + 1, c1 - c0 + 1))
                sub = mask[r0:r1 + 1, c0:c1 + 1]
                irregular.append(not (sub[0].all() and sub[-1].all() and sub[:, 0].all() and sub[:, -1].all()))
                wall_components.append(len(_components(mask)))
                free = np.zeros(a.shape, bool)
                for region in info['regions']:
                    if region['structural'] == color:
                        for cell in region['cells']:
                            free[cell] = True
                f, m = np.pad(free, 1), np.pad(mask, 1)
                internal_stops.append(int(np.count_nonzero(mask & (
                    (f[:-2, 1:-1] & f[2:, 1:-1]) | (f[1:-1, :-2] & f[1:-1, 2:])))))
                narrow_cells.append(int(np.count_nonzero(free & (
                    (m[:-2, 1:-1] & m[2:, 1:-1]) | (m[1:-1, :-2] & m[1:-1, 2:])))))
                kr0, kr1, kc0, kc1 = key['bbox']
                glyph = a[kr0:kr1 + 1, kc0:kc1 + 1]
                gr0, gr1, gc0, gc1 = _bbox(np.argwhere(glyph != color))
                paddings.append((int(gr0), int(glyph.shape[0] - 1 - gr1),
                                 int(gc0), int(glyph.shape[1] - 1 - gc1)))
                tips = np.argwhere(glyph == tip)
                staggered.append(len(set(map(int, tips[:, 0 if key['direction'][0] else 1]))) > 1)
                dy, dx = kr0 + kr1 - r0 - r1, kc0 + kc1 - c0 - c1
                sides.append(('bottom' if dy > 0 else 'top') if abs(dy) >= abs(dx)
                             else ('right' if dx > 0 else 'left'))
            overlap = any(max(b[0], c[0]) <= min(b[1], c[1]) and max(b[2], c[2]) <= min(b[3], c[3])
                          for i, b in enumerate(boxes) for c in boxes[i + 1:])
            touching = any(np.any(_dilate(info['walls'] & (a == c), True) & info['walls'] & (a == d))
                           for c in carriers for d in carriers if c != d)
            dirs = [k['direction'] for k in info['keys']]
            trails = [k['trail'] for k in info['keys']]
            cross_reuse = any(t in carriers for t in trails)
            trail_mode = 'carrier_reuse' if cross_reuse else ('shared' if len(set(trails)) == 1 else 'separate')
            inverted = any((info['keys'][i]['bbox'][2] - info['keys'][j]['bbox'][2])
                           * (boxes[i][2] - boxes[j][2]) < 0 for i in range(len(boxes)) for j in range(i))
            records.append(dict(split=split, shape=a.shape, background=info['background'], marker=marker, tip=tip,
                                systems=len(dirs), directions=dirs, markers=len(info['paths']),
                                distances=info['distances'], corner_witnesses=sum(info['corner_witnesses']),
                                key_shapes=[k['shape'] for k in info['keys']],
                                stroke_counts=[k['strokes'] for k in info['keys']],
                                trail_lengths=[n for k in info['keys'] for n in k['lengths']],
                                key_sides=sides, irregular=sum(irregular), regions=len(info['regions']),
                                wall_shapes=wall_shapes, wall_components=wall_components,
                                key_paddings=paddings, staggered_keys=sum(staggered),
                                internal_stops=internal_stops, narrow_cells=narrow_cells,
                                overlapping_boxes=overlap, touching=touching, inverted_keys=inverted,
                                repeated_direction=len(set(dirs)) < len(dirs), trail_mode=trail_mode,
                                combined=bool(any(irregular) and cross_reuse and len(set(dirs)) > 1
                                              and any(s in ('bottom', 'right') for s in sides))))
    return records


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_fingerprint(p) for p in puzzles}) == 42, 'seed diversity')
    _require(len({len(p['train']) for p in puzzles}) >= 4 and len({len(p['test']) for p in puzzles}) >= 3,
             'automatic count diversity')
    for i, (nt, nq) in enumerate(((2, 1), (10, 1), (10, 4), (7, 3), (None, 1), (None, 4), (2, None), (10, None))):
        a = generate_puzzle(2000 + i, nt, nq)
        b = generate_puzzle(2000 + i, nt, nq)
        _require(_puzzle_fingerprint(a) == _puzzle_fingerprint(b), 'nondeterminism')
        _require((nt is None or len(a['train']) == nt) and (nq is None or len(a['test']) == nq), 'count contract')
    for nt, nq in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (2.5, 1)):
        try:
            generate_puzzle(900, nt, nq)
        except ValueError:
            continue
        raise AssertionError('invalid counts accepted')
    batches = [measure_puzzle(p) for p in puzzles]
    rows = [r for batch in batches for r in batch]
    _require({tuple(d) for r in rows for d in r['directions']} == set(_ALL_DIRECTIONS), 'direction coverage')
    for split in ('train', 'test'):
        subset = [r for r in rows if r['split'] == split]
        _require({r['systems'] for r in subset} == {1, 2, 3}, 'system-count range')
        _require({r['trail_mode'] for r in subset} == {'shared', 'separate', 'carrier_reuse'}, 'local trail modes')
        _require(any(r['repeated_direction'] for r in subset), 'repeated-direction systems')
    _require(len({r['shape'] for r in rows}) >= 45, 'canvas diversity')
    _require(any(r['shape'] == (20, 20) for r in rows) and any(30 in r['shape'] for r in rows), 'source/full canvas')
    _require(min(min(r['shape']) for r in rows) <= 15, 'compact canvas')
    _require(len({s for r in rows for s in r['key_shapes']}) >= 15, 'key geometry diversity')
    _require({n for r in rows for n in r['stroke_counts']} >= {2, 3, 4}, 'repeated strokes')
    _require({n for r in rows for n in r['trail_lengths']} >= {2, 3, 4}, 'trail lengths')
    _require(len({p for r in rows for p in r['key_paddings']}) >= 10, 'key padding diversity')
    _require(sum(r['staggered_keys'] for r in rows) >= 80, 'stroke-offset diversity')
    _require(sum(any(r['internal_stops']) for r in rows) >= 50, 'internal stopping boundaries')
    _require(sum(any(r['narrow_cells']) for r in rows) >= 50, 'narrow passages')
    _require({s for r in rows for s in r['key_sides']} == {'top', 'bottom', 'left', 'right'}, 'key placement coverage')
    _require({d for r in rows for d in r['distances']} >= {0, 1, 2, 3, 4, 5}, 'movement lengths')
    _require(any(r['overlapping_boxes'] for r in rows) and any(r['touching'] for r in rows), 'nearby/interlocking walls')
    _require(sum(r['inverted_keys'] for r in rows) >= 15, 'independent key order')
    _require(sum(r['combined'] for r in rows) >= 10, 'combined variations')
    _require(sum(r['irregular'] > 0 for r in rows) >= len(rows) // 2, 'outer boundary diversity')
    _require(any(r['irregular'] == 0 for r in rows), 'simple enclosures retained')
    _require(any(max(r['wall_components']) > r['regions'] for r in rows), 'diagonal wall contacts')
    _require(max(r['regions'] for r in rows) >= 5, 'chamber arrangements')
    _require(len({s for r in rows for s in r['wall_shapes']}) >= 60, 'enclosure scale and aspect ratios')
    _require(all(len({r['background'] for r in batch}) >= 2 for batch in batches), 'within-puzzle background variation')
    _require(sum(len({s for r in batch for s in r['key_shapes']}) >= 2 for batch in batches) == 42,
             'within-puzzle key variation')
    _require(sum(len({r['systems'] for r in batch}) >= 2 for batch in batches) >= 25,
             'within-puzzle system-count variation')
    # Corruption tests ensure validation checks content, not just dimensions.
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    bad['test'][0]['output'] = bad['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('identity corruption accepted')
