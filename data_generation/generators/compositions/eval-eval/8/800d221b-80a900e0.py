"""Intercepted ballots: local cap majorities, endpoint rays, arrival votes.

Only NumPy and the standard library are used. Construction uses analytic
line/square intersections; solve walks the rays recovered from the input.
"""
from __future__ import annotations

import collections
import numpy as np

DIRS = ((-1, -1), (-1, 1), (1, 1), (1, -1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid grid colors")
    return a.astype(np.int8, copy=False)


def _base(shape, background):
    return np.asarray(background, dtype=np.int8)[np.indices(shape).sum(axis=0) % 2]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _diamond(center, radius, filled=False):
    r, c = center
    return {(r + dr, c + dc) for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1)
            if (abs(dr) + abs(dc) <= radius if filled else abs(dr) + abs(dc) == radius)}


def _cap(center, radius, direction):
    r, c = center
    vr, vc = DIRS[direction]
    return [(r + vr * (k + 1), c + vc * (radius - k + 1)) for k in range(radius + 1)]


def _square(frame, filled=False):
    r, c, size = frame
    s = size // 2
    return {(r + dr, c + dc) for dr in range(-s, s + 1)
            for dc in range(-s, s + 1)
            if filled or max(abs(dr), abs(dc)) == s or (dr == 0 and dc == 0)}


def _background(grid):
    parity = np.indices(grid.shape).sum(axis=0) % 2
    result = []
    for p in (0, 1):
        counts = np.bincount(grid[parity == p].astype(int), minlength=10)
        color = int(counts.argmax())
        _require(int(counts[color]) * 2 > int(np.sum(parity == p)), "checkerboard is not recoverable")
        result.append(color)
    _require(result[0] != result[1], "background must alternate two colors")
    return tuple(result)


def _matching_centers(grid, base, offsets, extent, interior):
    """Find a monochrome outline plus the specified empty interior."""
    h, w = grid.shape
    n, m = h - 2 * extent, w - 2 * extent
    if n <= 0 or m <= 0:
        return []
    dr, dc = offsets[0]
    colors = grid[extent + dr:extent + dr + n, extent + dc:extent + dc + m]
    background = _background(grid)
    good = (colors != background[0]) & (colors != background[1])
    for dr, dc in offsets[1:]:
        good &= grid[extent + dr:extent + dr + n, extent + dc:extent + dc + m] == colors
    for dr, dc in interior:
        good &= (grid[extent + dr:extent + dr + n, extent + dc:extent + dc + m]
                 == base[extent + dr:extent + dr + n, extent + dc:extent + dc + m])
    return [(int(r) + extent, int(c) + extent, int(colors[r, c])) for r, c in np.argwhere(good)]


def _analyze_input(value):
    grid = _array(value)
    background = _background(grid)
    base = _base(grid.shape, background)
    frames, emitters = [], []
    for size in (3, 5, 7):
        s = size // 2
        offsets = sorted(_square((0, 0, size)))
        interior = sorted(_square((0, 0, size), True) - set(offsets))
        for r, c, color in _matching_centers(grid, base, offsets, s, interior):
            frames.append(dict(frame=(r, c, size), color=color))
    for radius in (2, 3, 4):
        offsets = sorted(_diamond((0, 0), radius))
        interior = sorted(_diamond((0, 0), radius, True) - set(offsets))
        for r, c, color in _matching_centers(grid, base, offsets, radius, interior):
            emitters.append(dict(center=(r, c), radius=radius, color=color, caps={}))
    frames.sort(key=lambda f: f['frame'])
    emitters.sort(key=lambda e: e['center'])
    _require(1 <= len(frames) <= 3 and 1 <= len(emitters) <= 4, "wrong object counts")
    structural = {f['color'] for f in frames} | {e['color'] for e in emitters}
    covered, signals = set(), set()
    for f in frames:
        cells = _square(f['frame'])
        _require(not covered & cells, "overlapping structures")
        covered |= cells
    for e in emitters:
        cells = _diamond(e['center'], e['radius'])
        _require(not covered & cells, "overlapping structures")
        covered |= cells
    for e in emitters:
        for d in range(4):
            cells = _cap(e['center'], e['radius'], d)
            if not all(_inside(p, grid.shape) for p in cells):
                continue
            values = [int(grid[p]) for p in cells]
            colors = set(values)
            if len(colors) != 2 or colors & (structural | set(background)):
                continue
            counts = collections.Counter(values)
            _require(len(set(counts.values())) == 2, "local ballot tie")
            _require(not covered.intersection(cells), "overlapping caps")
            covered.update(cells)
            signals |= colors
            e['caps'][d] = max(counts, key=counts.get)
        _require(bool(e['caps']), "diamond has no complete mixed ballot cap")
    _require(len(signals) == 2, "scene must share two ballot colors")
    _require(covered == set(map(tuple, np.argwhere(grid != base))), "unexplained foreground")
    return dict(grid=grid, background=background, base=base, frames=frames,
                emitters=emitters, signals=tuple(sorted(signals)))


def _walk_output(info):
    grid, frames = info['grid'], info['frames']
    frame_at = {p: i for i, f in enumerate(frames) for p in _square(f['frame'])
                if p != f['frame'][:2]}
    body = set().union(*(_diamond(e['center'], e['radius'], True) for e in info['emitters']))
    output = grid.copy()
    claimed, ports = set(), set()
    votes = [[] for _ in frames]
    for e in info['emitters']:
        for d, color in e['caps'].items():
            cap = _cap(e['center'], e['radius'], d)
            vr, vc = DIRS[d]
            for endpoint in (cap[0], cap[-1]):
                p = (endpoint[0] + vr, endpoint[1] + vc)
                while _inside(p, grid.shape):
                    if p in frame_at:
                        _require(p not in ports, "two rays use one collector port")
                        ports.add(p)
                        votes[frame_at[p]].append(color)
                        break
                    _require(grid[p] == info['base'][p] and p not in body, "ray crosses an emitter")
                    _require(p not in claimed, "rays overlap")
                    claimed.add(p)
                    output[p] = color
                    p = (p[0] + vr, p[1] + vc)
    for f, labels in zip(frames, votes):
        counts = collections.Counter(labels)
        _require(bool(counts), "collector receives no votes")
        a, b = (counts[s] for s in info['signals'])
        _require(a != b, "collector vote tie")
        output[f['frame'][:2]] = info['signals'][int(b > a)]
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover shapes, local ballots and intercepted rays from this input alone."""
    return _walk_output(_analyze_input(input_grid))


def _event(endpoint, direction, shape, frames):
    """Closed-form first square intersection, independent of ray walking.

    Return (number of paintable cells, receiver index or None, entry or None).
    Frames are solid obstacles geometrically, although only their outlines are
    foreground. A ray starting outside first encounters the outline.
    """
    vr, vc = DIRS[direction]
    r, c = endpoint
    limit = min(shape[0] - 1 - r if vr > 0 else r,
                shape[1] - 1 - c if vc > 0 else c)
    best, owner = limit + 1, None
    for i, (fr, fc, size) in enumerate(frames):
        s = size // 2
        low_r, high_r = ((fr - s - r, fr + s - r) if vr > 0
                         else (r - fr - s, r - fr + s))
        low_c, high_c = ((fc - s - c, fc + s - c) if vc > 0
                         else (c - fc - s, c - fc + s))
        enter, leave = max(1, low_r, low_c), min(limit, high_r, high_c)
        if enter <= leave and enter < best:
            best, owner = enter, i
    if owner is None:
        return max(0, limit), None, None
    return best - 1, owner, (r + best * vr, c + best * vc)


def _emitter_zone(e):
    return (_diamond(e['center'], e['radius'], True)
            | {p for d in e['caps'] for p in _cap(e['center'], e['radius'], d)})


def _geometry(shape, frames, emitters):
    """Uncolored paths, unambiguous ports and per-cap arrival multiplicities."""
    zones = [_emitter_zone(e) for e in emitters]
    frame_zones = [_square(f, True) for f in frames]
    used = set().union(*frame_zones)
    for zone in zones:
        if any(not _inside(p, shape) for p in zone) or used & zone:
            return None
        used |= zone
    body = set().union(*zones)
    claimed, port_set, records = set(), set(), []
    for i, e in enumerate(emitters):
        for d in e['caps']:
            cells = _cap(e['center'], e['radius'], d)
            paths, entries = [], []
            vr, vc = DIRS[d]
            for endpoint in (cells[0], cells[-1]):
                steps, owner, entry = _event(endpoint, d, shape, frames)
                path = [(endpoint[0] + k * vr, endpoint[1] + k * vc) for k in range(1, steps + 1)]
                if claimed.intersection(path) or body.intersection(path):
                    return None
                if entry is not None and entry in port_set:
                    return None
                claimed.update(path)
                if entry is not None:
                    port_set.add(entry)
                paths.append(path)
                entries.append((owner, entry))
            records.append(dict(emitter=i, direction=d, cells=cells, paths=paths, entries=entries))
    return records


def _geometric_output(info):
    frames = [f['frame'] for f in info['frames']]
    records = _geometry(info['grid'].shape, frames, info['emitters'])
    _require(records is not None, "invalid path geometry")
    out = info['grid'].copy()
    votes = [[] for _ in frames]
    for rec in records:
        color = info['emitters'][rec['emitter']]['caps'][rec['direction']]
        for path in rec['paths']:
            for p in path:
                out[p] = color
        for owner, entry in rec['entries']:
            if owner is not None:
                votes[owner].append(color)
    for frame, labels in zip(frames, votes):
        counts = [labels.count(s) for s in info['signals']]
        _require(sum(counts) > 0 and counts[0] != counts[1], "invalid receiver ballot")
        out[frame[:2]] = info['signals'][int(counts[1] > counts[0])]
    return out, records


def _candidates(shape, frames, radius, direction, target, count, ghost=False, shadow=False):
    """Sample positions by geometric interaction, not a bank of rendered grids."""
    result = []
    for r in range(radius, shape[0] - radius):
        for c in range(radius, shape[1] - radius):
            cells = _cap((r, c), radius, direction)
            if not all(_inside(p, shape) for p in cells):
                continue
            events = [_event(p, direction, shape, frames) for p in (cells[0], cells[-1])]
            if sum(e[1] == target for e in events) != count:
                continue
            if ghost:
                if min(e[0] for e in events) < 2:
                    continue
                if sum(_event(p, direction, shape, frames)[1] == target for p in cells[1:-1]) < 2:
                    continue
            if shadow:
                if not any(e[1] == target and any(
                        _event(p, direction, shape, (f,))[1] is not None
                        for j, f in enumerate(frames) if j != target)
                           for p, e in zip((cells[0], cells[-1]), events)):
                    continue
            result.append((r, c))
    return result


def _add_emitter(rng, shape, frames, emitters, radius, direction, target, count,
                 required=(), ghost=False, shadow=False, extras=True):
    candidates = _candidates(shape, frames, radius, direction, target, count, ghost, shadow)
    if not candidates:
        return None
    order = rng.permutation(len(candidates))[:100]
    for index in order:
        ds = set((direction, *required))
        if extras:
            ds.update(d for d in range(4) if rng.random() < .30)
        e = dict(center=candidates[int(index)], radius=radius, caps={d: 0 for d in sorted(ds)})
        if _geometry(shape, frames, emitters + [e]) is not None:
            return e
        # Extra sides are optional variation; mandatory teaching caps remain.
        e['caps'] = {d: 0 for d in sorted(set((direction, *required)))}
        if _geometry(shape, frames, emitters + [e]) is not None:
            return e
    return None


def _pattern(rng, radius, majority, minority, endpoint_minor=False):
    n = radius + 1
    if endpoint_minor:
        minority_indices = [0, n - 1] if n >= 5 else [int(rng.choice([0, n - 1]))]
    else:
        m = int(rng.integers(1, (n - 1) // 2 + 1))
        minority_indices = list(map(int, rng.choice(n, m, replace=False)))
    values = [majority] * n
    for k in minority_indices:
        values[k] = minority
    return values


def _render(rng, shape, frames, emitters, roles, labels, override=None):
    bg0, bg1, s0, s1, *structs = roles
    inp = _base(shape, (bg0, bg1))
    for f in frames:
        color = int(rng.choice(structs))
        for p in _square(f):
            inp[p] = color
    records = _geometry(shape, frames, emitters)
    if records is None:
        return None
    for e in emitters:
        color = int(rng.choice(structs))
        for p in _diamond(e['center'], e['radius']):
            inp[p] = color
    for rec, bit in zip(records, labels):
        e = emitters[rec['emitter']]
        majority, minority = ((s0, s1) if bit == 0 else (s1, s0))
        e['caps'][rec['direction']] = majority
        values = _pattern(rng, e['radius'], majority, minority,
                          (rec['emitter'], rec['direction']) == override)
        for p, color in zip(rec['cells'], values):
            inp[p] = color
    out = inp.copy()
    votes = [[] for _ in frames]
    for rec, bit in zip(records, labels):
        color = (s0, s1)[bit]
        for path in rec['paths']:
            for p in path:
                out[p] = color
        for owner, entry in rec['entries']:
            if owner is not None:
                votes[owner].append(color)
    if not all(v and v.count(s0) != v.count(s1) for v in votes):
        return None
    for frame, v in zip(frames, votes):
        out[frame[:2]] = s0 if v.count(s0) > v.count(s1) else s1
    try:
        info = _analyze_input(inp)
        if not np.array_equal(_walk_output(info), out):
            return None
    except ValueError:
        return None
    return {'input': inp, 'output': out}


def _weight_witness(rng, params, roles, winner_bit):
    # One double arrival beats a single arrival; a third cap only points its
    # interior lanes at the collector. Rotations, distances and offsets vary.
    for _ in range(100):
        shape = tuple(map(int, rng.integers(25, 31, size=2)))
        center = tuple(int(rng.integers(8, n - 8)) for n in shape)
        frames = [(*center, 3)]
        d = int(rng.integers(4))
        neg = (d + int(rng.choice([1, 3]))) % 4
        plans = [(2, d, 2, (), False), (2, neg, 1, ((neg + 2) % 4,), False),
                 (params['radius'], (d + 2) % 4, 0, (d,), True)]
        emitters = []
        for radius, direction, count, required, ghost in plans:
            e = _add_emitter(rng, shape, frames, emitters, radius, direction, 0, count,
                             required, ghost=ghost, extras=True)
            if e is None:
                break
            emitters.append(e)
        if len(emitters) != 3:
            continue
        records = _geometry(shape, frames, emitters)
        if any(sum(owner is not None for owner, _ in rec['entries'])
               for rec in records if rec['direction'] != plans[rec['emitter']][1]):
            continue
        labels = [winner_bit if rec['emitter'] == 0 and rec['direction'] == d else 1 - winner_bit
                  for rec in records]
        pair = _render(rng, shape, frames, emitters, roles, labels,
                       override=(2, (d + 2) % 4))
        if pair is not None:
            m = _measure(pair['input'])
            if (m['cap_count_disagreement'] and m['full_cap_disagreement'] and m['raw_disagreement']
                    and m['endpoint_witnesses'] == set(range(4))):
                return pair
    return _weight_fallback(rng, params, roles, winner_bit)


def _weight_fallback(rng, params, roles, winner_bit):
    # Guaranteed procedural teaching construction after the free search. Its
    # offsets depend on cap parity: odd-radius ghost rays must use a different
    # lattice parity from the single-arrival source to avoid crossings.
    p = int(rng.integers(5, 10))
    q = int(rng.integers(6, 10) if params['radius'] == 4 else rng.integers(10, 13))
    total = 29 if params['radius'] == 4 else 34
    frames = [(15, 15, 3)]
    emitters = [dict(center=(p, p), radius=2, caps={2: 0}),
                dict(center=(q, total - q), radius=2, caps={1: 0, 3: 0}),
                dict(center=(22, 22), radius=params['radius'], caps={0: 0, 2: 0})]
    records = _geometry((30, 30), frames, emitters)
    _require(records is not None, 'procedural witness geometry failed')
    labels = [winner_bit if rec['emitter'] == 0 else 1 - winner_bit for rec in records]
    pair = _render(rng, (30, 30), frames, emitters, roles, labels, override=(2, 0))
    _require(pair is not None, 'procedural witness rendering failed')
    turns = int(rng.integers(4))
    return {k: np.rot90(v, turns).copy() for k, v in pair.items()}


def _shadow_witness(rng, params, roles, first_bit):
    # Free geometric sampling of two diagonally aligned receivers. The outer
    # sources address different receivers; the farther one is in the shadow.
    for _ in range(120):
        shape = tuple(map(int, rng.integers(28, 31, size=2)))
        d = int(rng.integers(4))
        vr, vc = DIRS[d]
        size_a = int(rng.choice([3, 5]))
        size_b = params['size']
        distance = int(rng.integers(size_a // 2 + size_b // 2 + 3, 11))
        frames = []
        for attempt in range(40):
            a = (int(rng.integers(8, shape[0] - 8)), int(rng.integers(8, shape[1] - 8)))
            b = (a[0] + vr * distance, a[1] + vc * distance)
            if all(_inside(p, shape) for p in _square((*b, size_b), True)):
                frames = [(*a, size_a), (*b, size_b)]
                break
        if not frames:
            continue
        emitters = []
        e = _add_emitter(rng, shape, frames, emitters, 2, d, 0, 2, shadow=True, extras=True)
        if e is None:
            continue
        emitters.append(e)
        e = _add_emitter(rng, shape, frames, emitters, params['radius'], (d + 2) % 4,
                         1, 2, shadow=True, extras=True)
        if e is None:
            continue
        emitters.append(e)
        records = _geometry(shape, frames, emitters)
        labels = [first_bit if rec['emitter'] == 0 else 1 - first_bit for rec in records]
        pair = _render(rng, shape, frames, emitters, roles, labels, override=(1, (d + 2) % 4))
        if pair is not None and _measure(pair['input'])['shadowed_rays']:
            return pair
    return _shadow_fallback(rng, params, roles, first_bit)


def _shadow_fallback(rng, params, roles, first_bit):
    a = int(rng.choice([10, 11]))
    p = int(rng.choice([3, 4]))
    frames = [(a, a, 3), (18, 18, params['size'])]
    emitters = [dict(center=(p, p), radius=2, caps={2: 0}),
                dict(center=(25, 25), radius=params['radius'], caps={0: 0})]
    records = _geometry((30, 30), frames, emitters)
    _require(records is not None, 'procedural occlusion geometry failed')
    labels = [first_bit if rec['emitter'] == 0 else 1 - first_bit for rec in records]
    pair = _render(rng, (30, 30), frames, emitters, roles, labels, override=(1, 0))
    _require(pair is not None, 'procedural occlusion rendering failed')
    turns = int(rng.integers(4))
    return {k: np.rot90(v, turns).copy() for k, v in pair.items()}


def _random_frames(rng, shape, count, max_size):
    frames, reserved = [], set()
    for _ in range(count):
        for attempt in range(100):
            size = int(rng.choice([s for s in (3, 5, 7) if s <= max_size]))
            s = size // 2
            r, c = (int(rng.integers(s + 1, n - s - 1)) for n in shape)
            cells = _square((r, c, size), True)
            if cells & reserved:
                continue
            frames.append((r, c, size))
            reserved.update((x + dr, y + dc) for x, y in cells
                            for dr in (-1, 0, 1) for dc in (-1, 0, 1))
            break
        else:
            return None
    return frames


def _random_pair(rng, params, roles, receivers, emitter_count):
    # Same planner in both splits. Large-object feasibility controls dimensions;
    # it does not encode winners or directions into positions.
    for restart in range(100):
        low = 17 if receivers == 1 and emitter_count <= 2 else 23
        if receivers == 3 or emitter_count == 4:
            low = 27
        shape = tuple(map(int, rng.integers(low, 31, size=2)))
        if rng.random() < .10:
            shape = (30, 30)
        frames = _random_frames(rng, shape, receivers, params['size'])
        if frames is None:
            continue
        emitters = []
        targets = list(map(int, rng.permutation(receivers)))
        targets += list(map(int, rng.integers(receivers, size=emitter_count - receivers)))
        for target in targets:
            e = None
            for attempt in range(12):
                radius = int(rng.integers(2, params['radius'] + 1))
                direction = int(rng.integers(4))
                count = int(rng.choice([1, 2]))
                e = _add_emitter(rng, shape, frames, emitters, radius, direction, target, count)
                if e is not None:
                    break
            if e is None:
                break
            emitters.append(e)
        if len(emitters) != emitter_count:
            continue
        records = _geometry(shape, frames, emitters)
        for attempt in range(50):
            labels = list(map(int, rng.integers(2, size=len(records))))
            pair = _render(rng, shape, frames, emitters, roles, labels)
            if pair is not None:
                return pair
    raise RuntimeError('bounded free-placement search exhausted')


def _counts(num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, 'num_train'), (num_test, 1, 4, 'num_test')):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(name + ' must be an integer')
        if not low <= int(value) <= high:
            raise ValueError(name + ' is outside its legal range')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    return int(num_train), int(num_test)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    params = dict(radius=int(rng.choice([3, 4])), size=int(rng.choice([5, 7])),
                  shift=int(rng.integers(2)), varying=bool(rng.random() < .7))
    bank_size = int(rng.integers(1, 4))
    palette = list(map(int, rng.choice(10, 4 + bank_size, replace=False)))
    teaching_slots = list(map(int, rng.choice(ntrain, 2, replace=False)))
    winner = int(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', ntrain), ('test', ntest)):
        for index in range(count):
            roles = (list(map(int, rng.choice(10, 4 + bank_size, replace=False)))
                     if params['varying'] else palette)
            for attempt in range(12):
                if split == 'train' and index == teaching_slots[0]:
                    pair = _weight_witness(rng, params, roles, winner)
                elif split == 'train' and index == teaching_slots[1]:
                    pair = _shadow_witness(rng, params, roles, 1 - winner)
                else:
                    receivers = int(rng.integers(1, 3))
                    emitters = int(rng.integers(receivers, 4))
                    if split == 'test' and index == 0:
                        if params['shift'] == 0:
                            receivers, emitters = 3, int(rng.choice([3, 4]))
                        else:
                            emitters = 4
                    elif split == 'test':
                        receivers = int(rng.integers(1, 4))
                        emitters = int(rng.integers(receivers, 5))
                    pair = _random_pair(rng, params, roles, receivers, emitters)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('unique input search exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _measure(value):
    info = _analyze_input(value)
    out, records = _geometric_output(info)
    frames = [f['frame'] for f in info['frames']]
    labels = [info['emitters'][rec['emitter']]['caps'][rec['direction']] for rec in records]
    signals = info['signals']
    arrivals = [[0, 0] for _ in frames]
    full = [[0, 0] for _ in frames]
    shadows, overrides = 0, 0
    directions, lengths, endpoint_witnesses, middle_only = set(), [], set(), 0
    for rec, color in zip(records, labels):
        bit = signals.index(color)
        e = info['emitters'][rec['emitter']]
        d = rec['direction']
        directions.add(d)
        lengths.extend(map(len, rec['paths']))
        if min(map(len, rec['paths'])) >= 2:
            vr, vc = DIRS[d]
            probes = [(p[0] + k * vr, p[1] + k * vc) for p in rec['cells'][1:-1] for k in (1, 2)]
            if all(_inside(p, info['grid'].shape) and out[p] == info['base'][p] for p in probes):
                endpoint_witnesses.add(d)
        for p, (owner, entry), path in zip((rec['cells'][0], rec['cells'][-1]), rec['entries'], rec['paths']):
            if owner is not None:
                arrivals[owner][bit] += 1
                shadows += int(any(_event(p, d, info['grid'].shape, (f,))[1] is not None
                                   for j, f in enumerate(frames) if j != owner))
            overrides += int(info['grid'][p] != color and bool(path))
        ghosts_here = 0
        for p in rec['cells']:
            owner = _event(p, d, info['grid'].shape, frames)[1]
            if owner is not None:
                full[owner][bit] += 1
                if p not in (rec['cells'][0], rec['cells'][-1]):
                    ghosts_here += 1
        middle_only += int(not any(owner is not None for owner, _ in rec['entries']) and ghosts_here > 0)
    winners = [int(b > a) for a, b in arrivals]
    cap_counts = [labels.count(s) for s in signals]
    raw_counts = [int(np.sum(info['grid'] == s)) for s in signals]
    return dict(shape=info['grid'].shape, receivers=len(frames), emitters=len(info['emitters']),
                sizes=[f[2] for f in frames], radii=[e['radius'] for e in info['emitters']],
                masks=[sum(1 << d for d in e['caps']) for e in info['emitters']],
                cap_count=len(records), votes=arrivals, margins=[abs(a - b) for a, b in arrivals],
                winners=winners, cap_count_disagreement=any(
                    cap_counts[0] != cap_counts[1] and int(cap_counts[1] > cap_counts[0]) != w for w in winners),
                raw_disagreement=any(raw_counts[0] != raw_counts[1] and int(raw_counts[1] > raw_counts[0]) != w for w in winners),
                full_votes=full,
                full_cap_disagreement=any(int(b > a) != w or a == b for (a, b), w in zip(full, winners)),
                full_cap_strict_disagreement=any(a != b and int(b > a) != w for (a, b), w in zip(full, winners)),
                shadowed_rays=shadows, endpoint_overrides=overrides, middle_only_caps=middle_only,
                directions=directions, endpoint_witnesses=endpoint_witnesses, ray_lengths=lengths,
                zero_length_rays=lengths.count(0), background=info['background'], signals=signals,
                structure=(info['grid'].shape, tuple(frames), tuple(
                    (e['center'], e['radius'], tuple(e['caps'])) for e in info['emitters'])))


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, measures = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            inp, out = (_array(pair[k]) for k in ('input', 'output'))
            _require(inp.shape == out.shape and not np.array_equal(inp, out), "invalid output shape or identity")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _analyze_input(inp)
            prediction, records = _geometric_output(info)
            _require(np.array_equal(prediction, out), "wrong geometric output")
            _require(np.array_equal(_walk_output(info), out), "walking solve mismatch")
            centers = {f['frame'][:2] for f in info['frames']}
            changed = set(map(tuple, np.argwhere(inp != out)))
            allowed = centers | {p for rec in records for path in rec['paths'] for p in path}
            _require(changed == allowed, "unexpected edits or missing ray cells")
            protected = {p for e in info['emitters'] for p in _emitter_zone(e)}
            protected |= {p for f in info['frames'] for p in _square(f['frame'], True) if p not in centers}
            _require(all(inp[p] == out[p] for p in protected), "ballot, outline or interior changed")
            measures[split].append(_measure(inp))
    train, test = measures['train'], measures['test']
    _require(any(m['cap_count_disagreement'] and m['full_cap_disagreement'] and m['raw_disagreement']
                 and m['middle_only_caps'] for m in train), "missing geometric weighted-vote witness")
    _require(any(m['shadowed_rays'] for m in train), "missing first-interceptor witness")
    _require(any(m['endpoint_overrides'] for m in train), "missing local-majority witness")
    _require(set().union(*(m['endpoint_witnesses'] for m in train)) == set(range(4)),
             "training must show endpoint-only extension in every direction")
    _require(max(r for m in train for r in m['radii']) >= max(r for m in test for r in m['radii']),
             "test radius not demonstrated")
    _require(max(s for m in train for s in m['sizes']) >= max(s for m in test for s in m['sizes']),
             "test collector size not demonstrated")


def self_test():
    """Bounded 42-seed structural coverage, metamorphic and corruption checks."""
    # Exercise both procedural fallbacks explicitly, independent of how often
    # the free planner needs them in a finite seed sample.
    for radius in (3, 4):
        for size in (5, 7):
            params = dict(radius=radius, size=size)
            rng = np.random.default_rng(radius * 100 + size)
            roles = [0, 1, 2, 3, 4, 5]
            w = _weight_fallback(rng, params, roles, 0)
            o = _shadow_fallback(rng, params, roles, 1)
            validate_generated_puzzle({'train': [w, o], 'test': [_shadow_fallback(rng, params, [5, 4, 3, 2, 1, 0], 0)]})
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    metrics = [_measure(pair['input']) for p in puzzles for split in ('train', 'test') for pair in p[split]]
    assert len({_key(p['train'][0]['input']) for p in puzzles}) == 42
    assert len({m['structure'] for m in metrics}) >= 180
    assert len({m['shape'] for m in metrics}) >= 35
    assert {r for m in metrics for r in m['radii']} == {2, 3, 4}
    assert {s for m in metrics for s in m['sizes']} == {3, 5, 7}
    assert {m['receivers'] for m in metrics} == {1, 2, 3}
    assert {m['emitters'] for m in metrics} == {1, 2, 3, 4}
    assert {len([d for d in range(4) if mask & (1 << d)]) for m in metrics for mask in m['masks']} == {1, 2, 3, 4}
    assert max(length for m in metrics for length in m['ray_lengths']) >= 15
    assert sum(m['zero_length_rays'] for m in metrics) > 0
    assert len({margin for m in metrics for margin in m['margins']}) >= 4
    # Whole-puzzle reproducibility and strict counts, including non-integer types.
    again = generate_puzzle(0)
    assert all(_key(a[k]) == _key(b[k]) for s in puzzles[0]
               for a, b in zip(puzzles[0][s], again[s]) for k in ('input', 'output'))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Rotation and palette permutations change geometry/roles, not conventions.
    for pair in puzzles[0]['train'] + puzzles[0]['test']:
        inp, out = pair['input'], pair['output']
        for turns in (1, 2, 3):
            assert np.array_equal(solve(np.rot90(inp, turns)), np.rot90(out, turns))
        permutation = np.array([9, 4, 7, 0, 8, 2, 1, 6, 3, 5], dtype=np.int8)
        assert np.array_equal(solve(permutation[inp]), permutation[out])
        # A cap's arrangement is not its vote: preserve counts and new originals.
        info = _analyze_input(inp)
        changed = inp.copy()
        expected = out.copy()
        for e in info['emitters']:
            for d in e['caps']:
                cells = _cap(e['center'], e['radius'], d)
                values = sorted(int(inp[p]) for p in cells)
                for p, v in zip(cells, values):
                    changed[p] = expected[p] = v
        assert np.array_equal(solve(changed), expected)
    # The validator must reject missing geometry and wrong collector labels.
    for kind in ('ray', 'center', 'minority'):
        bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in pairs]
               for s, pairs in puzzles[0].items()}
        pair = bad['test'][0]
        info = _analyze_input(pair['input'])
        if kind == 'center':
            p = info['frames'][0]['frame'][:2]
        elif kind == 'minority':
            e = info['emitters'][0]
            d = next(iter(e['caps']))
            p = next(p for p in _cap(e['center'], e['radius'], d) if pair['input'][p] != e['caps'][d])
        else:
            _, records = _geometric_output(info)
            p = next(p for rec in records for path in rec['paths'] for p in path)
        pair['output'][p] = (int(pair['output'][p]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('validator accepted a corrupted ' + kind)


if __name__ == '__main__':
    self_test()
