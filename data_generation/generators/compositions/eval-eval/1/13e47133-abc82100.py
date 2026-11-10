"""Phase-registered contour printing: local stencils sample relative room depth.

Only standard library and NumPy are used. The solver decodes the returned input;
no palette, registration, seed, or placement record is retained between calls.
"""
from collections import deque
import copy
import numpy as np

D4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _array(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError('expected an integer ARC grid, axes 1..30, colors 0..9')
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask, directions=D4):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, todo = {start}, [start]
        for r, c in todo:
            for dr, dc in directions:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    cells.add(q)
                    todo.append(q)
        result.append(cells)
    return result


def _mask(shape, cells):
    a = np.zeros(shape, dtype=bool)
    for p in cells:
        a[p] = True
    return a


def _depth(region, directions=D8):
    """Distance from an eight-neighbor boundary, starting with depth zero."""
    depth = {p: 0 for p in region
             if any((p[0] + dr, p[1] + dc) not in region for dr, dc in directions)}
    queue = deque(sorted(depth))
    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            q = r + dr, c + dc
            if q in region and q not in depth:
                depth[q] = depth[(r, c)] + 1
                queue.append(q)
    return depth


def _erosion(shape, region):
    """Vectorized erosion cross-check; independent of the distance queue."""
    current = _mask(shape, region)
    result = np.full(shape, -1, dtype=np.int8)
    level = 0
    h, w = shape
    while np.any(current):
        padded = np.pad(current, 1)
        inner = current.copy()
        for dr, dc in D8:
            inner &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        result[current & ~inner] = level
        current = inner
        level += 1
    return result


def _box_depth(region):
    rr, cc = zip(*region)
    r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
    return {p: min(p[0] - r0, r1 - p[0], p[1] - c0, c1 - p[1]) for p in region}


def _corner_paths(region, depth, length):
    paths = []
    for r, c in sorted(region):
        if depth[(r, c)] != 0:
            continue
        for dr in (-1, 1):
            for dc in (-1, 1):
                if (r - dr, c) in region or (r, c - dc) in region:
                    continue
                if (r + dr, c) not in region or (r, c + dc) not in region:
                    continue
                path = tuple((r + j * dr, c + j * dc) for j in range(length))
                if all(depth.get(p) == j for j, p in enumerate(path)):
                    paths.append(path)
    return paths


def _read_key(a, region, depth, background, stencil, source, target_label):
    cues = {p for p in region if int(a[p]) not in (background, stencil, source)
            and p != target_label}
    if len(cues) not in (2, 3):
        raise ValueError('each room needs one explicit two- or three-color key')
    paths = [path for path in _corner_paths(region, depth, len(cues)) if set(path) == cues]
    if len(paths) != 1:
        raise ValueError('palette is not a unique inward corner diagonal')
    path = paths[0]
    period = tuple(int(a[p]) for p in path)
    if len(set(period)) != len(period):
        raise ValueError('palette entries must be distinct')
    return period, path


def _interpret_roles(a, background, wall, stencil, source, regions):
    records = []
    for region in regions:
        cells = {p for p in region if int(a[p]) == stencil}
        if len(cells) < 3 or len(_components(_mask(a.shape, cells), D8)) != 1:
            raise ValueError('one connected stencil of at least three pixels per room')
        near = {(r + dr, c + dc) for r, c in cells
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        candidates = []
        for s in sorted(region):
            if int(a[s]) != source:
                continue
            for dr, dc in D4:
                t = s[0] + dr, s[1] + dc
                q = s[0] + 2 * dr, s[1] + 2 * dc
                if t not in region or q not in region or q not in near:
                    continue
                if int(a[t]) in (background, wall, stencil, source):
                    continue
                if int(a[q]) not in (background, stencil):
                    continue
                candidates.append((s, t, q, (dr, dc)))
        if len(candidates) != 1:
            raise ValueError('stencil label ray is missing or ambiguous')
        s, t, q, direction = candidates[0]
        depth = _depth(region)
        period, path = _read_key(a, region, depth, background, stencil, source, t)
        if int(a[t]) not in period:
            raise ValueError('target color must occur in its room palette')
        anchors = tuple(p for p in sorted(region) if int(a[p]) == source and p != s)
        if not anchors:
            raise ValueError('each stencil needs at least one external anchor')
        records.append(dict(region=region, depth=depth, period=period, path=path,
                            mask=cells, source_label=s, target_label=t, q=q,
                            direction=direction, target_index=period.index(int(a[t])),
                            offsets=tuple(sorted((r - q[0], c - q[1]) for r, c in cells)),
                            anchors=anchors))
    return records


def _decode(a):
    border = np.concatenate((a[0], a[-1], a[:, 0], a[:, -1]))
    if len(set(map(int, border))) != 1:
        raise ValueError('the outer frame must be a single wall color')
    wall = int(border[0])
    counts = np.bincount(a[a != wall], minlength=10)
    if counts.max() == 0 or np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError('background must be the unique nonwall mode')
    background = int(counts.argmax())
    walls = a == wall
    if len(_components(walls)) != 1:
        raise ValueError('walls must be attached to the frame')
    if np.any(walls[:-1, :-1] & walls[1:, :-1] & walls[:-1, 1:] & walls[1:, 1:]):
        raise ValueError('walls must be one cell wide')
    regions = _components(~walls)
    colors = sorted(set(map(int, np.unique(a))) - {background, wall})
    # Stencils have >=3 connected cells in EVERY room. This geometric filter
    # rejects the tiny palette cues before trying full instruction readings.
    possible_stencils = [color for color in colors
                         if all(sum(int(a[p]) == color for p in region) >= 3
                                for region in regions)]
    readings = []
    for stencil in possible_stencils:
        for source in colors:
            if source == stencil:
                continue
            try:
                records = _interpret_roles(a, background, wall, stencil, source, regions)
                _paint(a.shape, wall, walls, background, records)
            except ValueError:
                continue
            readings.append((stencil, source, records))
    if len(readings) != 1:
        raise ValueError('input does not uniquely identify stencil/source roles')
    stencil, source, records = readings[0]
    return dict(background=background, wall=wall, stencil=stencil, source=source,
                walls=walls, records=records)


def _footprint(rec, anchor, depth=None, mode='relative'):
    region = rec['region']
    d = rec['depth'] if depth is None else depth
    period, t = rec['period'], rec['target_index']
    result = {}
    for dr, dc in rec['offsets']:
        p = anchor[0] + dr, anchor[1] + dc
        if p not in region:
            continue
        if mode == 'uniform':
            index = t
        elif mode == 'absolute':
            index = d[p]
        elif mode == 'reverse':
            index = t - d[p] + d[anchor]
        elif mode == 'unsigned':
            index = t + abs(d[p] - d[anchor])
        else:
            index = t + d[p] - d[anchor]
        result[p] = period[index % len(period)]
    return result


def _paint(shape, wall, walls, background, records, mode='relative', metric=None):
    out = np.full(shape, background, dtype=np.int8)
    out[walls] = wall
    paint = {}
    for rec in records:
        depth = None
        if metric == 'l1':
            depth = _depth(rec['region'], D4)
        elif metric == 'box':
            depth = _box_depth(rec['region'])
        for anchor in rec['anchors']:
            for p, color in _footprint(rec, anchor, depth, mode).items():
                if p in paint and paint[p] != color:
                    raise ValueError('overlapping copies disagree in color')
                paint[p] = color
    for p, color in paint.items():
        out[p] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _array(input_grid)
    info = _decode(a)
    return _paint(a.shape, info['wall'], info['walls'], info['background'], info['records'])


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _layout(rng, split, index, fallback=False):
    if fallback:
        h, w = 30, 30
        walls = np.zeros((h, w), bool)
        walls[[0, -1], :] = True
        walls[:, [0, -1]] = True
        walls[12, 1:11] = True
        return walls
    first = split == 'train' and index < 2
    lo = 22 if first else 14 if split == 'train' else 18
    h, w = int(rng.integers(lo, 31)), int(rng.integers(lo, 31))
    walls = np.zeros((h, w), bool)
    walls[[0, -1], :] = True
    walls[:, [0, -1]] = True
    rooms = [(1, h - 2, 1, w - 2)]
    target = _choose(rng, (1, 1, 2) if first else (1, 1, 2, 2, 3, 4))
    while len(rooms) < target:
        options = [(i, axis) for i, (a, b, c, d) in enumerate(rooms)
                   for axis, n in ((0, b - a + 1), (1, d - c + 1)) if n >= 21]
        if not options:
            break
        i, axis = _choose(rng, options)
        a, b, c, d = rooms.pop(i)
        if axis == 0:
            cut = int(rng.integers(a + 10, b - 9))
            walls[cut, c - 1:d + 2] = True
            rooms.extend(((a, cut - 1, c, d), (cut + 1, b, c, d)))
        else:
            cut = int(rng.integers(c + 10, d - 9))
            walls[a - 1:b + 2, cut] = True
            rooms.extend(((a, b, c, cut - 1), (a, b, cut + 1, d)))
    # Orthogonal dead-end walls create connected, concave rooms. Their tips
    # make Chebyshev erosion visibly differ from rectangular and L1 bands.
    for i, (a, b, c, d) in enumerate(rooms):
        n = int(rng.integers(1 if first and i == 0 else 0, 3))
        for _ in range(n):
            side = int(rng.integers(4))
            if side in (0, 2):
                x = int(rng.integers(c + 3, d - 2))
                length = int(rng.integers(2, min(7, b - a - 4) + 1))
                cells = {(a + k if side == 0 else b - k, x) for k in range(length)}
                attachment = (a - 1 if side == 0 else b + 1, x)
            else:
                x = int(rng.integers(a + 3, b - 2))
                length = int(rng.integers(2, min(7, d - c - 4) + 1))
                cells = {(x, c + k if side == 3 else d - k) for k in range(length)}
                attachment = (x, c - 1 if side == 3 else d + 1)
            # A spur must not join an earlier spur or make a thick wall.
            old = set(map(tuple, np.argwhere(walls).tolist()))
            if any((r + dr, col + dc) in old - {attachment}
                   for r, col in cells for dr, dc in D4):
                continue
            for p in cells:
                walls[p] = True
    return walls


def _rotate(cells, turns):
    cells = set(cells)
    for _ in range(turns):
        cells = {(c, -r) for r, c in cells}
    return tuple(sorted(cells))


def _stencil(rng, hole, teaching=False, extended=False):
    if hole:
        radius = int(rng.integers(3, 5)) if extended else int(rng.integers(2 if teaching else 1, 3))
        width = int(rng.integers(2, 6))
        cells = {(r, c) for r in (-radius, radius) for c in range(width + 1)}
        cells |= {(r, width) for r in range(-radius, radius + 1)}
        # Registration is inside the C, one cell left of its closed side.
        q = (0, width - 1)
    else:
        h = int(rng.integers(2, 6))
        w = int(rng.integers(7, 10)) if extended else int(rng.integers(3, 7))
        family = _choose(rng, ('elbow', 'comb') if teaching else
                         ('line', 'elbow', 'comb', 'branch', 'block', 'growth', 'zigzag'))
        if family == 'line':
            cells = {(0, c) for c in range(w)}
        elif family == 'elbow':
            cells = {(0, c) for c in range(w)} | {(r, w - 1) for r in range(h)}
        elif family == 'comb':
            cells = {(0, c) for c in range(w)}
            cells |= {(r, c) for c in range(0, w, 2) for r in range(h)}
        elif family == 'branch':
            cells = {(r, w // 2) for r in range(h)} | {(h // 2, c) for c in range(w)}
        elif family == 'block':
            cells = {(r, c) for r in range(h) for c in range(w)}
        elif family == 'zigzag':
            cells = {(c % 2, c) for c in range(w)}
        else:
            cells = {(0, 0)}
            for _ in range(int(rng.integers(4, 18))):
                r, c = _choose(rng, sorted(cells))
                dr, dc = _choose(rng, D8)
                if 0 <= r + dr < h and 0 <= c + dc < w:
                    cells.add((r + dr, c + dc))
            cells |= {(0, 0), (0, 1), (0, 2)}
            if extended:
                cells |= {(0, c) for c in range(w)}
            # Retain the component containing the guaranteed three-cell arm.
            cells = _components(_mask((h, w), cells), D8)[0]
        options = [(r, c) for r, c in sorted(cells)
                   if (r, c - 1) not in cells and (r, c - 2) not in cells]
        q = _choose(rng, options)
    offsets = tuple((r - q[0], c - q[1]) for r, c in cells)
    turns = int(rng.integers(4))
    if rng.integers(2):
        offsets = tuple((-r, c) for r, c in offsets)
    offsets = _rotate(offsets, turns)
    direction = _rotate(((0, 1),), turns)[0]
    return offsets, direction


def _candidate_flags(rec, anchor):
    paint = _footprint(rec, anchor)
    if not paint:
        return set()
    flags = set()
    depth, period = rec['depth'], rec['period']
    ds = [depth[p] - depth[anchor] for p in paint]
    if len(paint) == len(rec['offsets']):
        flags.add('full')
    else:
        flags.add('clip')
    if len(set(paint.values())) > 1:
        flags.add('multicolor')
    if min(ds) < 0:
        flags.add('negative')
    if max(ds) > 0:
        flags.add('positive')
    if max(ds) - min(ds) >= len(period):
        flags.add('repeat')
    for mode in ('absolute', 'reverse', 'unsigned'):
        if paint != _footprint(rec, anchor, mode=mode):
            flags.add(mode)
    for metric in ('l1', 'box'):
        if paint != _footprint(rec, anchor, rec[metric]):
            flags.add(metric)
    return flags


def _add_anchors(rng, a, rec, background, source, teaching):
    forbidden = set(rec['path']) | rec['mask'] | {rec['source_label'], rec['target_label'], rec['q']}
    # Instructions are clearly separated from the copies in generated data.
    key_halo = {(r + dr, c + dc) for r, c in rec['path'] for dr, dc in D8 + ((0, 0),)}
    cells = [p for p in sorted(rec['region']) if p not in forbidden | key_halo]
    rng.shuffle(cells)
    candidates = []
    for p in cells:
        paint = _footprint(rec, p)
        if not paint or set(paint) & forbidden:
            continue
        flags = _candidate_flags(rec, p)
        candidates.append((p, paint, flags))
    required = {'full', 'multicolor', 'absolute'}
    if teaching == 0:
        required |= {'l1', 'box', 'repeat'}
    if teaching == 1:
        required |= {'negative', 'positive', 'reverse', 'unsigned', 'repeat', 'clip'}
    if teaching is None:
        required = {'multicolor'}
    planned, anchors, covered = {}, [], set()
    desired = int(rng.integers(2 if teaching is not None else 1, 6 if teaching is not None else 7))
    for _ in range(max(desired, 5 if teaching is not None else desired)):
        compatible = [(p, paint, flags) for p, paint, flags in candidates
                      if p not in anchors and all(q not in planned or planned[q] == v for q, v in paint.items())]
        if not compatible:
            break
        need_phase = teaching == 0 and len({rec['depth'][p] % len(rec['period']) for p in anchors}) < 2
        def score(item):
            p, paint, flags = item
            unseen_phase = not anchors or all(rec['depth'][p] % len(rec['period']) !=
                                             rec['depth'][v] % len(rec['period']) for v in anchors)
            return len((required - covered) & flags) * 10 + (4 if need_phase and unseen_phase else 0)
        best = max(map(score, compatible))
        choices = [item for item in compatible if score(item) == best]
        p, paint, flags = _choose(rng, choices)
        anchors.append(p)
        planned.update(paint)
        covered |= flags
        if len(anchors) >= desired and required <= covered:
            if not need_phase or len({rec['depth'][v] % len(rec['period']) for v in anchors}) >= 2:
                break
    if not anchors or not required <= covered:
        raise ValueError('cannot expose requested contour/registration evidence')
    if teaching == 0 and len({rec['depth'][p] % len(rec['period']) for p in anchors}) < 2:
        raise ValueError('need anchors at different contour phases')
    rec['anchors'] = tuple(sorted(anchors))
    for p in anchors:
        a[p] = source


def _make_pair(rng, split, index):
    teaching = index if split == 'train' and index < 2 else None
    for attempt in range(100):
        walls = _layout(rng, split, index, fallback=attempt >= 85)
        roles = [int(c) for c in rng.permutation(10)]
        background, wall, stencil, source = roles[:4]
        a = np.full(walls.shape, background, dtype=np.int8)
        a[walls] = wall
        regions = sorted(_components(~walls), key=lambda x: (-len(x), min(x)))
        records = []
        try:
            for i, region in enumerate(regions):
                depth = _depth(region)
                length = (2 if teaching == 0 else 3 if teaching == 1 else int(rng.integers(2, 4)))
                paths = _corner_paths(region, depth, length)
                if not paths:
                    raise ValueError('room too shallow for palette')
                path = _choose(rng, paths)
                period = tuple(int(c) for c in rng.choice(roles[4:], length, replace=False))
                for p, color in zip(path, period):
                    a[p] = color
                hole = teaching == 1 or (teaching is None and rng.random() < .45)
                offsets, direction = _stencil(rng, hole, teaching is not None,
                                             extended=split == 'test' and i == 0)
                dr, dc = direction
                relative = set(offsets) | {(0, 0), (-dr, -dc), (-2 * dr, -2 * dc)}
                halo = {(r + x, c + y) for r, c in path for x in (-1, 0, 1) for y in (-1, 0, 1)}
                available = region - halo
                placements = [q for q in sorted(region)
                              if all((q[0] + x, q[1] + y) in available for x, y in relative)]
                if not placements:
                    raise ValueError('instruction does not fit')
                q = _choose(rng, placements)
                cells = {(q[0] + x, q[1] + y) for x, y in offsets}
                s, t = (q[0] - 2 * dr, q[1] - 2 * dc), (q[0] - dr, q[1] - dc)
                target_index = int(rng.integers(length))
                for p in cells:
                    a[p] = stencil
                a[s], a[t] = source, period[target_index]
                box = _box_depth(region)
                rec = dict(region=region, depth=depth, period=period, path=path,
                           offsets=offsets, direction=direction, q=q, mask=cells,
                           source_label=s, target_label=t, target_index=target_index,
                           l1=_depth(region, D4), box=box)
                _add_anchors(rng, a, rec, background, source, teaching if i == 0 else None)
                records.append(rec)
            output = _paint(a.shape, wall, walls, background, records)
            if teaching == 1 and not any(output[p] == background for p in records[0]['anchors']):
                raise ValueError('hole teaching must leave a visibly empty registration')
            decoded = _decode(a)
            if (decoded['background'], decoded['wall'], decoded['stencil'], decoded['source']) != tuple(roles[:4]):
                raise ValueError('visible roles disagree with construction')
            predicted = _paint(a.shape, wall, walls, background, decoded['records'])
            if not np.array_equal(output, predicted):
                raise ValueError('input decoding disagrees with construction')
            return {'input': a, 'output': output}
        except ValueError:
            continue
    raise RuntimeError('bounded construction exhausted')


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(name + ' outside allowed range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    nt, ne = _count(num_train, 2, 10, 'num_train'), _count(num_test, 1, 4, 'num_test')
    if nt <= ne:
        raise ValueError('num_train must be greater than num_test')
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, n in (('train', nt), ('test', ne)):
        for index in range(n):
            for _ in range(20):
                pair = _make_pair(rng, split, index)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('could not construct distinct inputs')
    validate_generated_puzzle(puzzle)
    return puzzle


def _analysis(pair):
    a, out = _array(pair['input']), _array(pair['output'])
    _require(a.shape == out.shape, 'output dimensions changed')
    info = _decode(a)
    expected = _paint(a.shape, info['wall'], info['walls'], info['background'], info['records'])
    _require(np.array_equal(expected, out), 'incorrect phase-registered contour print')
    _require(np.array_equal(out == info['wall'], info['walls']), 'wall changed or leaked')
    _require(not np.any((out == info['stencil']) | (out == info['source'])), 'instruction color survived')
    flags = set()
    for rec in info['records']:
        erosion = _erosion(a.shape, rec['region'])
        _require(all(int(erosion[p]) == rec['depth'][p] for p in rec['region']), 'erosion mismatch')
        rec['l1'] = _depth(rec['region'], D4)
        rec['box'] = _box_depth(rec['region'])
        for anchor in rec['anchors']:
            flags |= _candidate_flags(rec, anchor)
    # Ablation A: ordinary uniformly colored stencil copies must disagree.
    try:
        uniform = _paint(a.shape, info['wall'], info['walls'], info['background'], info['records'], mode='uniform')
        _require(not np.array_equal(uniform, out), 'contour concept has no effect')
    except ValueError:
        pass  # The ablated rule can also introduce conflicting overlap.
    # Ablation B: filling entire rooms with A's absolute contour bands differs.
    full = np.full(a.shape, info['background'], dtype=np.int8)
    full[info['walls']] = info['wall']
    for rec in info['records']:
        for p in rec['region']:
            full[p] = rec['period'][rec['depth'][p] % len(rec['period'])]
    _require(not np.array_equal(full, out), 'registered stencil concept has no effect')
    info['flags'] = flags
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'splits must be lists')
    _require(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
             and len(puzzle['train']) > len(puzzle['test']), 'invalid counts')
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            key = _key(_array(pair['input']))
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            info = _analysis(pair)
            if split == 'train':
                training.append(info)
                _require(all(_span(rec['offsets']) <= 6 for rec in info['records']),
                         'training stencil exceeds its teaching size range')
            else:
                _require(any(7 <= _span(rec['offsets']) <= 9 for rec in info['records']),
                         'test lacks the stencil-reach shift')
    flags = set().union(*(info['flags'] for info in training))
    _require({'full', 'clip', 'multicolor', 'absolute', 'l1', 'box', 'repeat',
              'negative', 'positive', 'reverse', 'unsigned'} <= flags, 'teaching evidence missing')
    recs = [rec for info in training for rec in info['records']]
    _require({2, 3} <= {len(rec['period']) for rec in recs}, 'training needs both period lengths')
    _require(any((0, 0) not in rec['offsets'] for rec in recs), 'no hole-registration teaching')
    _require(any((0, 0) not in rec['offsets'] and
                 any(np.asarray(pair['output'])[p] == info['background'] for p in rec['anchors'])
                 for pair, info in zip(puzzle['train'], training) for rec in info['records']),
             'no visibly empty printed registration hole')
    _require(any((0, 0) in rec['offsets'] and
                 (sum(r for r, c in rec['offsets']) != 0 or sum(c for r, c in rec['offsets']) != 0)
                 for rec in recs), 'no off-center occupied-registration teaching')
    _require(any(len({rec['depth'][p] % len(rec['period']) for p in rec['anchors']}) > 1
                 for rec in recs), 'training must distinguish anchor-relative and absolute phase')


def _puzzle_key(puzzle):
    return tuple((_key(p['input']), _key(p['output']))
                 for split in ('train', 'test') for p in puzzle[split])


def _span(offsets):
    rr, cc = zip(*offsets)
    return max(max(rr) - min(rr) + 1, max(cc) - min(cc) + 1)


def _manual_regressions():
    # Hand-calculated rectangular-room examples do not use the print helper
    # to construct their answers. Outward steps wrap backwards in a 3-cycle.
    a = np.zeros((13, 16), dtype=np.int8)
    a[[0, -1], :] = 8
    a[:, [0, -1]] = 8
    a[1, 1], a[2, 2], a[3, 3] = 2, 3, 4
    a[8, 1], a[8, 2] = 1, 3
    a[8, 3:7] = 5
    a[4, 11], a[8, 10] = 1, 1
    expected = np.zeros_like(a)
    expected[[0, -1], :] = 8
    expected[:, [0, -1]] = 8
    expected[4, 11:15] = (3, 2, 4, 3)
    expected[8, 10:14] = (3, 3, 2, 4)
    assert np.array_equal(solve(a), expected)
    # Empty registration at (8,4); its copy at (4,11) stays empty while the
    # seven C pixels sample both inward and outward room depths.
    a[8, 1:7] = 0
    a[8, 10] = 0
    a[8, 2], a[8, 3] = 1, 2
    for p in ((7, 3), (7, 4), (7, 5), (8, 5), (9, 3), (9, 4), (9, 5)):
        a[p] = 5
    expected[4, 11:15] = 0
    expected[8, 10:14] = 0
    expected[3, 10:13] = (4, 4, 4)
    expected[4, 12] = 4
    expected[5, 10:13] = (3, 2, 4)
    assert np.array_equal(solve(a), expected)
    assert solve(a)[4, 11] == 0
    permutation = np.array((7, 5, 9, 0, 8, 4, 2, 3, 6, 1), dtype=np.int8)
    assert np.array_equal(solve(permutation[a]), permutation[expected])
    assert np.array_equal(solve(a[:, ::-1]), expected[:, ::-1])


def self_test():
    """Bounded 42-seed audit plus equivariance, corruption, and count checks."""
    _manual_regressions()
    signatures, shapes, walls, masks, roles, room_counts = (set() for _ in range(6))
    anchor_counts, directions, lengths, registrations = (set() for _ in range(4))
    clipping = overlap = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_key(puzzle))
        for pair in puzzle['train'] + puzzle['test']:
            a = pair['input']
            info = _decode(a)
            shapes.add(a.shape)
            walls.add(_key(info['walls']))
            roles.add(tuple(info[k] for k in ('background', 'wall', 'stencil', 'source')))
            room_counts.add(len(info['records']))
            clipping += 'clip' in _analysis(pair)['flags']
            for rec in info['records']:
                masks.add(rec['offsets'])
                lengths.add(len(rec['period']))
                anchor_counts.add(len(rec['anchors']))
                directions.add(rec['direction'])
                registrations.add((0, 0) in rec['offsets'])
                visited = set()
                for anchor in rec['anchors']:
                    cells = set(_footprint(rec, anchor))
                    overlap += bool(cells & visited)
                    visited |= cells
        if seed < 3:
            # Rotating all visible geometry must rotate the answer, including
            # corner-key reading, registration holes, clipping and wall depth.
            for pair in puzzle['train'] + puzzle['test']:
                for turns in (1, 2, 3):
                    assert np.array_equal(solve(np.rot90(pair['input'], turns)),
                                          np.rot90(pair['output'], turns))
    assert len(signatures) == 42
    assert len(shapes) >= 45 and len(walls) >= 100 and len(masks) >= 80 and len(roles) >= 100
    assert {1, 2, 3, 4} <= room_counts
    assert lengths == {2, 3} and registrations == {False, True} and directions == set(D4)
    assert min(anchor_counts) == 1 and max(anchor_counts) >= 5 and clipping >= 42 and overlap >= 10
    assert _puzzle_key(generate_puzzle(127, 2, 1)) == _puzzle_key(generate_puzzle(127, 2, 1))
    for nt, ne in ((2, 1), (10, 1), (10, 4), (5, 4)):
        p = generate_puzzle(2026, nt, ne)
        assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    broken = copy.deepcopy(generate_puzzle(9, 2, 1))
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted a corrupt wall')
    return {'seeds': 42, 'distinct_puzzles': len(signatures), 'shapes': len(shapes),
            'wall_geometries': len(walls), 'registered_masks': len(masks),
            'role_assignments': len(roles), 'room_counts': sorted(room_counts),
            'anchor_counts': sorted(anchor_counts), 'clipped_pairs': clipping,
            'overlapping_copies': overlap}
