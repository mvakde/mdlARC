"""Echo docks: pool a template-local launch manifest and dock it in every chamber.

All solving conventions are visible in one input. Construction labels come from
the sampled room geometry and manifest; the decoder independently reconstructs
them from pixels. Only the standard library and NumPy are used.
"""
from __future__ import annotations

import numpy as np

CARDINAL = ((-1, 0), (0, -1), (0, 1), (1, 0))
COMPASS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def _orient(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _fingerprint(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _components(mask):
    seen = ~mask.copy()
    result = []
    h, w = mask.shape
    for rr, cc in np.argwhere(mask):
        p = int(rr), int(cc)
        if seen[p]:
            continue
        stack, cells = [p], []
        seen[p] = True
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in CARDINAL:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        result.append(cells)
    return result


def _bbox(cells):
    rs, cs = zip(*cells)
    return min(rs), max(rs), min(cs), max(cs)


def _glyph(a):
    """Identify two consistent arrows without a literal tip/trail color map."""
    colors, counts = np.unique(a, return_counts=True)
    carrier = int(colors[np.argmax(counts)])
    others = set(map(int, colors)) - {carrier}
    if len(others) != 3:
        return None
    possibilities = []
    for tip in sorted(others):
        tips = set(map(tuple, np.argwhere(a == tip)))
        if len(tips) != 2:
            continue
        for trail in sorted(others - {tip}):
            trails = set(map(tuple, np.argwhere(a == trail)))
            for dr, dc in COMPASS:
                used, lengths = set(), []
                for r, c in sorted(tips):
                    stroke = []
                    q = (r - dr, c - dc)
                    while q in trails:
                        stroke.append(q)
                        q = q[0] - dr, q[1] - dc
                    if len(stroke) != 2 or used.intersection(stroke):
                        break
                    used.update(stroke)
                    lengths.append(len(stroke))
                if len(lengths) == 2 and used == trails:
                    anchor = next(iter(others - {tip, trail}))
                    if np.count_nonzero(a == anchor) == 1:
                        possibilities.append((carrier, trail, tip, anchor, (dr, dc)))
    return possibilities[0] if len(possibilities) == 1 else None


def _trace(wall, start, direction, corner=True):
    """Trace a packet, treating only wall cells as collision surfaces."""
    h, w = wall.shape
    r, c = start
    dr, dc = direction
    if not (0 <= r < h and 0 <= c < w) or wall[r, c]:
        return ()
    path = [(r, c)]
    for _ in range(30):
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            raise AssertionError('a flight escaped its enclosing wall')
        if wall[nr, nc] or (corner and dr and dc and (wall[nr, c] or wall[r, nc])):
            return tuple(path)
        r, c = nr, nc
        path.append((r, c))
    raise AssertionError('unterminated flight')


def _at(labels, label):
    p = np.argwhere(labels == label)
    _require(p.shape == (1, 2), 'invalid correspondence')
    return tuple(map(int, p[0]))


def _read(input_grid):
    a = np.asarray(input_grid)
    _require(a.ndim == 2 and np.issubdtype(a.dtype, np.integer), 'integer 2D input required')
    _require(all(1 <= n <= 30 for n in a.shape) and np.all((0 <= a) & (a <= 9)), 'invalid grid')
    values, counts = np.unique(a, return_counts=True)
    _require(np.count_nonzero(counts == counts.max()) == 1, 'background must be uniquely most frequent')
    bg = int(values[np.argmax(counts)])
    pads = []
    pad_mask = np.zeros(a.shape, bool)
    for cells in _components(a != bg):
        r0, r1, c0, c1 = _bbox(cells)
        sub = a[r0:r1 + 1, c0:c1 + 1]
        if min(sub.shape) != 5 or not 7 <= max(sub.shape) <= 10 or sub.size != len(cells):
            continue
        glyph = _glyph(sub)
        if glyph is None:
            continue
        pads.append(dict(r=r0, c=c0, array=sub.copy(), glyph=glyph, cells=set(cells)))
        pad_mask[r0:r1 + 1, c0:c1 + 1] = True
    _require(2 <= len(pads) <= 4, 'need two to four launch pads')
    canonical = pads[0]['array'].copy()
    if canonical.shape[0] != 5:
        canonical = np.rot90(canonical)
    if _glyph(canonical)[-1][0] < 0:
        canonical = np.flipud(canonical)
    glyph = _glyph(canonical)
    _require(glyph is not None and glyph[-1][0] == 1, 'arrow must point out a long side')
    lateral = glyph[-1][1]
    length = canonical.shape[1]
    forms = [_orient(canonical, t) for t in range(8)]
    _require(len({_fingerprint(x) for x in forms}) == 8, 'ambiguous pad symmetry')
    structural = set(map(int, np.unique(canonical)))
    labels = np.arange(canonical.size).reshape(canonical.shape)
    for pad in pads:
        matches = [t for t, form in enumerate(forms) if np.array_equal(form, pad['array'])]
        _require(len(matches) == 1, 'pads must be exact uniquely oriented copies')
        pad['transform'] = matches[0]
        pad['labels'] = _orient(labels, matches[0])

    remaining = (a != bg) & ~pad_mask
    _require(np.any(remaining), 'missing walls and packets')
    remaining_colors, remaining_counts = np.unique(a[remaining], return_counts=True)
    _require(np.count_nonzero(remaining_counts == remaining_counts.max()) == 1, 'unclear wall role')
    wall_color = int(remaining_colors[np.argmax(remaining_counts)])
    _require(wall_color not in structural and bg not in structural, 'overlapping roles')
    wall = a == wall_color
    packet_mask = remaining & ~wall
    packet_colors = set(map(int, np.unique(a[packet_mask])))
    _require(packet_colors and packet_colors.isdisjoint(structural), 'unclear packet colors')
    _require(np.count_nonzero(wall) > np.count_nonzero(packet_mask), 'wall role not dominant outside pads')

    region_labels = np.full(a.shape, -1, dtype=int)
    enclosed = set()
    for i, cells in enumerate(_components(~wall)):
        for p in cells:
            region_labels[p] = i
        if all(r not in (0, a.shape[0] - 1) and c not in (0, a.shape[1] - 1) for r, c in cells):
            enclosed.add(i)
    regions = []
    for pad in pads:
        ids = {int(region_labels[p]) for p in pad['cells']}
        _require(len(ids) == 1 and ids <= enclosed, 'pad lacks a closed chamber')
        regions.append(next(iter(ids)))
    _require(len(set(regions)) == len(pads), 'each chamber must contain exactly one pad')

    membership = {}
    for i, pad in enumerate(pads):
        p0, p1 = _at(pad['labels'], 4 * length), _at(pad['labels'], 4 * length + 1)
        pinner = _at(pad['labels'], 3 * length)
        direction = (p0[0] - pinner[0] + lateral * (p1[0] - p0[0]),
                     p0[1] - pinner[1] + lateral * (p1[1] - p0[1]))
        _require(direction == pad['glyph'][-1], 'glyph and transported vector disagree')
        pad['direction'], pad['paths'] = direction, {}
        pad['tangent'] = p1[0] - p0[0], p1[1] - p0[1]
        for slot in range(length):
            edge = _at(pad['labels'], 4 * length + slot)
            start = (pad['r'] + edge[0] + direction[0], pad['c'] + edge[1] + direction[1])
            path = _trace(wall, start, direction)
            _require(all(not pad_mask[p] and region_labels[p] == regions[i] for p in path), 'flight entered a pad or another chamber')
            pad['paths'][slot] = path
            for p in path:
                membership.setdefault(p, []).append((i, slot))

    code, allocations, observations = {}, [set() for _ in pads], []
    for rr, cc in np.argwhere(packet_mask):
        p = int(rr), int(cc)
        hits = membership.get(p, [])
        _require(len(hits) == 1, 'packet must identify exactly one launch lane')
        i, slot = hits[0]
        color = int(a[p])
        _require(slot not in code or code[slot] == color, 'conflicting manifest observations')
        _require(slot not in allocations[i], 'at most one observation per dock/slot')
        path = pads[i]['paths'][slot]
        _require(p != path[-1], 'observations must still be in flight')
        code[slot] = color
        allocations[i].add(slot)
        observations.append((i, slot, p, color))
        for dr, dc in COMPASS:
            nr, nc = p[0] + dr, p[1] + dc
            if 0 <= nr < a.shape[0] and 0 <= nc < a.shape[1]:
                _require(a[nr, nc] == bg, 'in-flight observations must be isolated')
    _require(1 <= len(code) <= 4, 'one to four observed slots required')
    _require(all(len(pad['paths'][s]) >= 2 for pad in pads for s in code), 'active flights must have room to travel')
    return dict(grid=a, bg=bg, wall_color=wall_color, wall=wall, pad_mask=pad_mask,
                canonical=canonical, pads=pads, code=code, allocations=allocations,
                observations=observations, packet_mask=packet_mask, regions=regions,
                length=length, lateral=lateral, region_labels=region_labels)


def _render(info, local_only=False, corner=True):
    out = info['grid'].copy()
    out[info['packet_mask']] = info['bg']
    endpoints = set()
    for i, pad in enumerate(info['pads']):
        slots = info['allocations'][i] if local_only else info['code']
        for slot in slots:
            path = pad['paths'][slot]
            if not corner:
                path = _trace(info['wall'], path[0], pad['direction'], corner=False)
            _require(path[-1] not in endpoints, 'colliding dock endpoints')
            endpoints.add(path[-1])
            out[path[-1]] = info['code'][slot]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the entire manifest, orientations and walls from this input only."""
    return _render(_read(input_grid))


def _features(info):
    corners = 0
    distances = []
    for pad in info['pads']:
        for slot in info['code']:
            path = pad['paths'][slot]
            corners += _trace(info['wall'], path[0], pad['direction'], False)[-1] != path[-1]
            distances.append(len(path))
    phases, moves = [], []
    for i, slot, p, _ in info['observations']:
        path = info['pads'][i]['paths'][slot]
        phases.append(path.index(p))
        moves.append(len(path) - 1 - path.index(p))
    wall_shapes = []
    for region in info['regions']:
        # Derive the chamber's footprint from a wall-only flood fill.
        cells = list(map(tuple, np.argwhere(info['region_labels'] == region)))
        r0, r1, c0, c1 = _bbox(cells)
        wall_shapes.append((r1 - r0 + 3, c1 - c0 + 3))
    return dict(shape=info['grid'].shape, background=info['bg'], length=info['length'],
                pads=len(info['pads']), slots=len(info['code']), code=tuple(sorted(info['code'].items())),
                allocations=tuple(map(len, info['allocations'])),
                pooled=all(len(s) < len(info['code']) for s in info['allocations']),
                unhinted=any(not s for s in info['allocations']),
                directions=tuple(p['direction'] for p in info['pads']),
                transforms=tuple(p['transform'] for p in info['pads']),
                corner_witnesses=int(corners), flight_lengths=tuple(distances),
                phases=tuple(phases), moves=tuple(moves), wall_shapes=tuple(wall_shapes),
                repeated_color=len(set(info['code'].values())) < len(info['code']),
                asymmetric_code=info['code'] != {info['length'] - 1 - s: c for s, c in info['code'].items()},
                longitudinal_reflection=any(p['direction'] == q['direction']
                    and p['tangent'] == tuple(-x for x in q['tangent'])
                    for p in info['pads'] for q in info['pads']))


def _make_pad(rng, length, lateral, colors):
    carrier, trail, tip, anchor = colors
    for _ in range(30):
        a = np.full((5, length), carrier, np.int8)
        if lateral == 0:
            columns = sorted(map(int, rng.choice(range(1, length - 1), 2, replace=False)))
        elif lateral == 1:
            start = int(rng.integers(1, length - 6 + 1))
            columns = [start, start + 3]
        else:
            start = int(rng.integers(5, length - 1))
            columns = [start - 3, start]
        for col in columns:
            for j in range(3):
                a[1 + j, col + lateral * j] = tip if j == 2 else trail
        a[0, int(rng.integers(length))] = anchor
        if _glyph(a) and len({_fingerprint(_orient(a, t)) for t in range(8)}) == 8:
            return a
    raise AssertionError('pad construction failed')


def _room(rng, pad, lateral, wall_color, bg, complexity, corner_slot=None):
    length = pad.shape[1]
    h = int(rng.integers(12, 15))
    w = int(rng.integers(length + 4, 15))
    r = 2 if corner_slot is not None else int(rng.integers(2, 4))
    c = int(rng.integers(2, w - length - 1))
    # A wandering downstream shore makes a stepped, concave enclosure.
    depths = np.full(w, h - 1, int)
    if corner_slot is None and rng.random() < .8:
        for col in range(1, w - 1):
            depths[col] = int(np.clip(depths[col - 1] + rng.integers(-1, 2), h - 3, h - 1))
    footprint = np.zeros((h, w), bool)
    for col, depth in enumerate(depths):
        footprint[:depth + 1, col] = True
    interior = footprint.copy()
    padded = np.pad(footprint, 1)
    for dr, dc in CARDINAL:
        interior &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    wall = footprint & ~interior
    for _ in range(complexity):
        row = int(rng.integers(r + 7, h - 1))
        cut = int(rng.integers(2, w - 2))
        if rng.integers(2):
            wall[row, :cut] |= footprint[row, :cut]
        else:
            wall[row, cut:] |= footprint[row, cut:]
    if corner_slot is not None:
        row, col = r + 7, c + corner_slot + 4 * lateral
        if not (1 <= col < w - 1) or row >= h - 2:
            return None
        # Attached half-baffle: the next diagonal cell is free, its side is wall.
        if lateral > 0:
            wall[row, col:] = True
        else:
            wall[row, :col + 1] = True
        wall[row + 1, col] = False
    a = np.full((h, w), bg, np.int8)
    a[wall] = wall_color
    if np.any(wall[r:r + 5, c:c + length]):
        return None
    a[r:r + 5, c:c + length] = pad
    paths, candidates = {}, {}
    for slot in range(length):
        start = r + 5, c + slot + lateral
        path = _trace(wall, start, (1, lateral))
        paths[slot] = path
        safe = []
        for p in path[1:-1]:
            if all(0 <= p[0] + dr < h and 0 <= p[1] + dc < w
                   and a[p[0] + dr, p[1] + dc] == bg for dr, dc in COMPASS):
                safe.append(p)
        candidates[slot] = safe
    return dict(grid=a, paths=paths, candidates=candidates, wall=wall,
                pad_position=(r, c), corner_slot=corner_slot)


def _pack(rng, arrays, bg):
    """Random free placement, with a guaranteed variable-size packing at the tail."""
    n = len(arrays)
    for _ in range(16):
        occupied = np.zeros((30, 30), bool)
        positions = []
        for a in arrays:
            h, w = a.shape
            blocked = np.pad(occupied, 1)
            dilated = occupied.copy()
            for dr, dc in COMPASS:
                dilated |= blocked[1 + dr:31 + dr, 1 + dc:31 + dc]
            windows = np.lib.stride_tricks.sliding_window_view(dilated, (h, w))
            valid = np.argwhere(~np.any(windows, axis=(-1, -2)))
            if not len(valid):
                break
            # Corner-biased, jittered placements make dense packings feasible.
            if n >= 3:
                distances = np.minimum(valid[:, 0], 30 - h - valid[:, 0]) + np.minimum(valid[:, 1], 30 - w - valid[:, 1])
                valid = valid[distances <= distances.min() + int(rng.integers(1, 4))]
            r, c = map(int, valid[int(rng.integers(len(valid)))])
            occupied[r:r + h, c:c + w] = True
            positions.append((r, c))
        if len(positions) == n:
            break
    else:
        # All rooms have sides <=14, so this always fits with visible gaps.
        order = list(map(int, rng.permutation(4)))[:n]
        positions = [(0 if cell < 2 else 30 - arrays[i].shape[0],
                      0 if cell % 2 == 0 else 30 - arrays[i].shape[1])
                     for i, cell in enumerate(order)]
    rmin, cmin = min(p[0] for p in positions), min(p[1] for p in positions)
    rmax = max(r + a.shape[0] for (r, c), a in zip(positions, arrays))
    cmax = max(c + a.shape[1] for (r, c), a in zip(positions, arrays))
    top, left = int(rng.integers(min(2, rmin) + 1)), int(rng.integers(min(2, cmin) + 1))
    bottom, right = int(rng.integers(min(2, 30 - rmax) + 1)), int(rng.integers(min(2, 30 - cmax) + 1))
    origin = rmin - top, cmin - left
    shape = rmax + bottom - origin[0], cmax + right - origin[1]
    canvas = np.full(shape, bg, np.int8)
    for (r, c), a in zip(positions, arrays):
        canvas[r - origin[0]:r - origin[0] + a.shape[0], c - origin[1]:c - origin[1] + a.shape[1]] = a
    return canvas, positions, origin


def _pair(rng, demonstration, test, previous_bg):
    for attempt in range(120):
        palette = list(map(int, rng.permutation(10)))
        if palette[0] == previous_bg:
            palette[0], palette[-1] = palette[-1], palette[0]
        bg, wall_color = palette[:2]
        lateral = 0 if demonstration == 1 else (int(rng.choice([-1, 1])) if demonstration == 2 else int(rng.integers(-1, 2)))
        length = int(rng.integers(8 if lateral else 7, 11))
        pad = _make_pad(rng, length, lateral, palette[2:6])
        n = 2 if demonstration else int(rng.choice([3, 4] if test else [2, 3]))
        # The diagonal core demonstration contains a guaranteed corner witness.
        witness_slot = None
        if demonstration == 2:
            witness_slot = 0 if lateral > 0 else length - 1
        rooms = []
        for i in range(n):
            complexity = 0 if demonstration else int(rng.integers(0, 3 if test else 2))
            room = _room(rng, pad, lateral, wall_color, bg, complexity, witness_slot if i == 0 else None)
            if room is None:
                break
            rooms.append(room)
        if len(rooms) != n:
            continue
        eligible = [s for s in range(length) if all(len(room['paths'][s]) >= 2 for room in rooms)
                    and any(room['candidates'][s] for room in rooms[:2])]
        desired = int(rng.integers(2 if demonstration else 1, 5))
        if len(eligible) < desired or (witness_slot is not None and witness_slot not in eligible):
            continue
        slots = sorted(map(int, rng.choice(eligible, desired, replace=False)))
        if witness_slot is not None and witness_slot not in slots:
            slots[0] = witness_slot
            slots = sorted(slots)
        colors = palette[6:6 + desired]
        if not demonstration and desired > 1 and rng.random() < .3:
            colors[-1] = colors[0]
        rng.shuffle(colors)
        code = dict(zip(slots, colors))
        # Alternating slots expose incomplete per-pad manifests, including adjacent lanes.
        assignments = [(j % 2, s) for j, s in enumerate(slots)]
        if witness_slot is not None:
            assignments = [(0 if s == witness_slot else 1, s) for _, s in assignments]
        else:
            assignments = [(i if rooms[i]['candidates'][s] else 1 - i, s) for i, s in assignments]
        if not demonstration and n > 2 and rng.random() < .3:
            assignments.append((2, int(rng.choice(slots))))
        starts, used = [], [set() for _ in rooms]
        failed = False
        for i, slot in assignments:
            candidates = list(rooms[i]['candidates'][slot])
            rng.shuffle(candidates)
            candidates = [p for p in candidates if all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) > 1 for q in used[i])]
            if not candidates:
                failed = True
                break
            p = candidates[0]
            starts.append((i, slot, p))
            used[i].add(p)
        if failed:
            continue
        base = int(rng.integers(8))
        transforms = [base, base ^ 4] if demonstration == 1 else ([base, (base // 4) * 4 + (base + 1) % 4] if demonstration == 2 else list(map(int, rng.integers(8, size=n))))
        inputs, outputs = [], []
        for i, room in enumerate(rooms):
            a = room['grid'].copy()
            b = room['grid'].copy()
            for ii, slot, p in starts:
                if ii == i:
                    a[p] = code[slot]
            for slot, color in code.items():
                b[room['paths'][slot][-1]] = color
            inputs.append(_orient(a, transforms[i]))
            outputs.append(_orient(b, transforms[i]))
        a, positions, origin = _pack(rng, inputs, bg)
        b = np.full(a.shape, bg, np.int8)
        for (r, c), out in zip(positions, outputs):
            r, c = r - origin[0], c - origin[1]
            b[r:r + out.shape[0], c:c + out.shape[1]] = out
        try:
            info = _read(a)
            _require(np.array_equal(_render(info), b), 'construction/decoding disagree')
            f = _features(info)
            if demonstration == 2:
                _require(f['corner_witnesses'] > 0, 'missing diagonal evidence')
            if demonstration:
                _require(f['pooled'], 'core demonstration must require pooling')
            return {'input': a, 'output': b}
        except AssertionError:
            continue
    raise AssertionError('bounded pair construction exhausted')


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    puzzle, seen, previous_bg = {'train': [], 'test': []}, set(), None
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            for _ in range(8):
                pair = _pair(rng, index + 1 if split == 'train' and index < 2 else 0, split == 'test', previous_bg)
                fp = _fingerprint(pair['input'])
                if fp not in seen:
                    break
            _require(fp not in seen, 'duplicate input construction')
            seen.add(fp)
            previous_bg = _read(pair['input'])['bg']
            puzzle[split].append(pair)
    # The two essential demonstrations need not occupy particular displayed slots.
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'invalid counts')
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair schema')
            a, b = np.asarray(pair['input']), np.asarray(pair['output'])
            _require(b.ndim == 2 and np.issubdtype(b.dtype, np.integer) and a.shape == b.shape, 'invalid output grid')
            _require(np.all((b >= 0) & (b <= 9)), 'invalid output colors')
            fp = _fingerprint(a)
            _require(fp not in seen, 'duplicate input')
            seen.add(fp)
            info = _read(a)
            expected = _render(info)
            _require(np.array_equal(expected, b), 'incorrect docking output')
            _require(not np.array_equal(a, b), 'identity example')
            fixed = ~info['packet_mask'] & (a != info['bg'])
            _require(np.array_equal(a[fixed], b[fixed]), 'wall or pad changed')
            colors = set(info['code'].values())
            count = sum(np.count_nonzero(b == c) for c in colors)
            _require(count == len(info['pads']) * len(info['code']), 'incomplete manifest or duplicate dock')
            if split == 'train':
                training.append((info, _features(info)))
    _require(any(f['pooled'] and not np.array_equal(_render(i, local_only=True), _render(i)) for i, f in training), 'missing A witness')
    _require(any(f['corner_witnesses'] and not np.array_equal(_render(i, corner=False), _render(i)) for i, f in training), 'missing B corner witness')
    _require(any(f['longitudinal_reflection'] and f['asymmetric_code']
                 and f['slots'] >= 2 and len(set(i['code'].values())) >= 2 for i, f in training), 'missing reflected color transfer')
    _require(any(len({t % 2 for t in f['transforms']}) == 2 for _, f in training), 'missing quarter-turn transfer')
    _require(any(i['lateral'] == 0 for i, _ in training) and any(i['lateral'] != 0 for i, _ in training), 'missing cardinal/diagonal contrast')


def self_test():
    """Bounded authored checks, including 42 seeds and meaningful ablations."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records, normalized = [], set()
    for puzzle in puzzles:
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _read(pair['input'])
                f = _features(info)
                f['split'] = split
                records.append(f)
                a = pair['input']
                roles = {}
                normalized.add((a.shape, tuple(roles.setdefault(int(c), len(roles)) for c in a.flat)))
                _require(np.array_equal(solve(a.copy()), pair['output']), 'solve mismatch')
    _require(len(normalized) >= 180, 'insufficient structural variation')
    _require(len({f['shape'] for f in records}) >= 25, 'insufficient canvas variation')
    _require({f['length'] for f in records} == {7, 8, 9, 10}, 'length coverage')
    _require({f['pads'] for f in records} == {2, 3, 4}, 'pad count coverage')
    _require({f['slots'] for f in records} == {1, 2, 3, 4}, 'manifest count coverage')
    _require({d for f in records for d in f['directions']} == set(COMPASS), 'compass coverage')
    _require(len({f['background'] for f in records}) == 10, 'palette coverage')
    _require(any(f['repeated_color'] for f in records), 'repeated slot colors')
    _require(len({p for f in records for p in f['phases']}) >= 3, 'flight-phase variation')
    _require(any(1 in f['moves'] for f in records), 'short motion missing')
    _require(any(f['slots'] >= 2 and any(b[0] - a[0] == 1 for a, b in zip(f['code'], f['code'][1:])) for f in records), 'adjacent slots missing')
    _require(all(f['unhinted'] for f in records if f['split'] == 'test' and f['pads'] == 4), 'sparse query transfer missing')
    original = generate_puzzle(7001, 6, 3)
    again = generate_puzzle(7001, 6, 3)
    _require(all(np.array_equal(p[k], q[k]) for s in original for p, q in zip(original[s], again[s]) for k in p), 'nondeterminism')
    # Whole-scene isometries and role permutations must commute with solving.
    for t in range(8):
        pair = original['test'][0]
        palette = np.roll(np.arange(10), t + 1)
        a, b = _orient(palette[pair['input']], t), _orient(palette[pair['output']], t)
        _require(np.array_equal(solve(a), b), 'isometry/recoloring failed')
    # Reject outputs implementing either tempting parent omission.
    for local_only, corner in ((True, True), (False, False)):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in original[s]] for s in original}
        for pair in bad['train']:
            info = _read(pair['input'])
            wrong = _render(info, local_only=local_only, corner=corner)
            if not np.array_equal(wrong, pair['output']):
                pair['output'] = wrong
                break
        else:
            raise AssertionError('ablation lacks a distinguishing example')
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('parent ablation was accepted')
    # The host enumerates every legal count; retain boundary coverage here.
    for nt, nq in ((2, 1), (10, 1), (10, 4), (5, 4), (7, 3)):
        p = generate_puzzle(9000 + nt * 10 + nq, nt, nq)
        _require(len(p['train']) == nt and len(p['test']) == nq, 'explicit counts')
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds, explicit counts, geometric invariants and ablations)')
