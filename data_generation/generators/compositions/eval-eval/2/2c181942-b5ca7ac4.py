"""Compass-driven crate nesting. Standard library and NumPy only.

The hollow compass visibly maps frame colors to canvas edges. A crate's
one-cell tab is a pivot: turn it toward that edge, preserve the perpendicular
tab coordinate, then slide stably using its actual pixels. Unmatched frames
vanish. The compass is an instruction card, not a collision obstacle.
"""
from __future__ import annotations

import numpy as np


_NEIGHBORS = tuple((r, c) for r in (-1, 0, 1)
                   for c in (-1, 0, 1) if r or c)


def _counts(a, b):
    for x, name, lo, hi in ((a, 'num_train', 2, 10),
                            (b, 'num_test', 1, 4)):
        if (isinstance(x, (bool, np.bool_)) or
                not isinstance(x, (int, np.integer)) or not lo <= x <= hi):
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if a <= b:
        raise ValueError('num_train must exceed num_test')
    return int(a), int(b)


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape) or
            not np.issubdtype(g.dtype, np.integer) or
            np.any((g < 0) | (g > 9))):
        raise ValueError('expected a 1..30 integer ARC grid with colors 0..9')
    return g.astype(int, copy=True)


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def _rotate(points, turns):
    p = np.array(points, dtype=int, copy=True)
    for _ in range(turns % 4):
        p = np.column_stack((-p[:, 1], p[:, 0]))
    return p


def _components(g, bg):
    remaining = set(map(tuple, np.argwhere(g != bg)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in _NEIGHBORS:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(np.array(sorted(cells), dtype=int))
    return result


def _hub_patch(bg, labels):
    hub = np.full((4, 4), bg, dtype=int)
    hub[0, 1:3] = labels[0]
    hub[1:3, 3] = labels[1]
    hub[3, 1:3] = labels[2]
    hub[1:3, 0] = labels[3]
    return hub


def _as_hub(g, coords, bg):
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    if tuple(hi - lo + 1) != (4, 4) or len(coords) != 8:
        return None
    r, c = map(int, lo)
    p = g[r:r+4, c:c+4]
    labels = tuple(map(int, (p[0, 1], p[1, 3], p[3, 1], p[1, 0])))
    if len(set(labels)) == 4 and bg not in labels and np.array_equal(
            p, _hub_patch(bg, labels)):
        return (r, c), labels
    return None


def _crate(g, coords, bg):
    """Infer a rectangular frame and its unique geometric tab, without labels."""
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    r, c = map(int, lo)
    patch = g[r:hi[0]+1, c:hi[1]+1]
    candidates = []
    for k in range(4):
        p = np.rot90(patch, k)
        h, w = p.shape[0] - 1, p.shape[1]
        if h < 3 or w < 3 or np.count_nonzero(p[-1] != bg) != 1:
            continue
        tab = int(np.flatnonzero(p[-1] != bg)[0])
        if not 1 <= tab < w - 1 or np.any(p[:-1] == bg):
            continue
        frame = int(p[0, 0])
        body = p[:-1]
        if (frame == bg or p[-1, tab] != frame or
                not all(np.all(edge == frame) for edge in
                        (body[0], body[-1], body[:, 0], body[:, -1])) or
                np.any(body[1:-1, 1:-1] == frame)):
            continue
        marker = np.zeros_like(p, dtype=bool)
        marker[-1, tab] = True
        original_tab = np.argwhere(np.rot90(marker, -k))[0] + lo
        occupied = np.argwhere(p != bg)
        candidates.append(dict(frame=frame, h=h, w=w, tab=tab,
                               pivot=original_tab, raw_turn=(-k) % 4,
                               relative=occupied - (h, tab),
                               values=p[p != bg].copy(),
                               interior=body[1:-1, 1:-1].copy(),
                               raw=coords.copy()))
    if len(candidates) != 1:
        raise ValueError('crate must have one recognizable tab and a complete frame')
    return candidates[0]


def _decode(grid):
    g = _grid(grid)
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] * 2 <= g.size:
        raise ValueError('background must occupy strictly more than half the grid')
    comps = _components(g, bg)
    hubs = [(i, _as_hub(g, c, bg)) for i, c in enumerate(comps)]
    hubs = [(i, h) for i, h in hubs if h is not None]
    if len(hubs) != 1:
        raise ValueError('expected one separated four-color hollow compass')
    hub_index, (hub, labels) = hubs[0]
    objects = []
    for i, coords in enumerate(comps):
        if i == hub_index:
            continue
        o = _crate(g, coords, bg)
        o['side'] = labels.index(o['frame']) if o['frame'] in labels else None
        if o['side'] is not None:
            side = o['side']
            o['target_turn'] = (2 - side) % 4
            o['points'] = _rotate(o['relative'], o['target_turn']) + o['pivot']
            if not _in_bounds(o['points'], g.shape):
                raise ValueError('turned crate leaves the canvas')
            axis = 0 if side % 2 == 0 else 1
            o['axis'] = axis
            o['transverse'] = int(o['pivot'][1-axis])
            o['span'] = (int(o['points'][:, 1-axis].min()),
                         int(o['points'][:, 1-axis].max()))
            o['distance'] = int(o['pivot'][axis] if side in (0, 3)
                                else g.shape[axis]-1-o['pivot'][axis])
        objects.append(o)
    if not objects:
        raise ValueError('the input must contain at least one crate')
    for i, a in enumerate(objects):
        if a['side'] is None:
            continue
        for b in objects[i+1:]:
            if a['side'] == b['side'] and _intersect(a['points'], b['points']):
                raise ValueError('same-class turned crates overlap before sliding')
    return g, bg, hub, labels, objects


def _in_bounds(points, shape):
    return bool(np.all(points >= 0) and np.all(points < np.array(shape)))


def _intersect(a, b):
    return not set(map(tuple, a)).isdisjoint(map(tuple, b))


def _span_overlap(a, b):
    return max(a['span'][0], b['span'][0]) <= min(a['span'][1], b['span'][1])


def _opposite_cross(objects):
    # Reversed starting order on overlapping tracks requires the classes to pass.
    return any(a['side'] in (0, 3) and b['side'] == (a['side']+2) % 4
               and _span_overlap(a, b) and a['pivot'][a['axis']] > b['pivot'][b['axis']]
               for a in objects for b in objects
               if a['side'] is not None and b['side'] is not None)


def _delta(side):
    return np.array(((-1, 0), (0, 1), (1, 0), (0, -1))[side])


def _pack(objects, shape):
    """Stable boundary recurrence, including the tab's one-cell nesting effect."""
    result = []
    for side in range(4):
        group = sorted((o for o in objects if o['side'] == side),
                       key=lambda o: (o['distance'], o['transverse']))
        settled = []
        for o in group:
            blockers = [b for b in settled if _span_overlap(o, b)]
            d = max((b['final_distance'] + b['h'] +
                     int(b['span'][0] <= o['transverse'] <= b['span'][1])
                     for b in blockers), default=0)
            if d > o['distance']:
                raise ValueError('stable packing would require inward motion')
            p = o['points'] + (o['distance'] - d) * _delta(side)
            if not _in_bounds(p, shape):
                raise ValueError('packed class exceeds the canvas')
            settled.append(dict(o, final_distance=d, final_points=p))
        result.extend(settled)
    return result


def _render(shape, bg, hub, labels, packed):
    g = np.full(shape, bg, dtype=int)
    r, c = hub
    g[r:r+4, c:c+4] = _hub_patch(bg, labels)
    occupied = g != bg
    for o in packed:
        p = o['final_points']
        if not _in_bounds(p, shape) or np.any(occupied[p[:, 0], p[:, 1]]):
            raise ValueError('final classes or compass overlap')
        g[p[:, 0], p[:, 1]] = o['values']
        occupied[p[:, 0], p[:, 1]] = True
    return g


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Predict using only the input. No puzzle mapping, seed or labels are stored."""
    g, bg, hub, labels, objects = _decode(input_grid)
    return _render(g.shape, bg, hub, labels, _pack(objects, g.shape))


def _simulate(objects, shape):
    """Validation by literal one-cell motion, without the boundary recurrence."""
    result = []
    for side in range(4):
        group = [dict(o, final_points=o['points'].copy())
                 for o in objects if o['side'] == side]
        occupied = np.zeros(shape, dtype=bool)
        for o in group:
            p = o['final_points']
            occupied[p[:, 0], p[:, 1]] = True
        for _ in range(30 * max(1, len(group)) + 1):
            moved = False
            for o in group:
                p = o['final_points']
                occupied[p[:, 0], p[:, 1]] = False
                trial = o['final_points'] + _delta(side)
                if _in_bounds(trial, shape) and not np.any(occupied[trial[:, 0], trial[:, 1]]):
                    o['final_points'] = trial
                    moved = True
                p = o['final_points']
                occupied[p[:, 0], p[:, 1]] = True
            if not moved:
                break
        else:
            raise AssertionError('unit-step packing failed to converge')
        for o in group:
            # The projecting tip is the extremal point toward the edge.
            ax = o['axis']
            tip = int(o['final_points'][:, ax].min() if side in (0, 3)
                      else o['final_points'][:, ax].max())
            o['final_distance'] = tip if side in (0, 3) else shape[ax]-1-tip
        result.extend(group)
    return result


def _edge_first(objects, shape):
    """Competing shortcut: teleport each follower into the first vacant slot."""
    distances = []
    for side in range(4):
        group = sorted((o for o in objects if o['side'] == side),
                       key=lambda o: (o['distance'], o['transverse']))
        occupied = np.zeros(shape, dtype=bool)
        for o in group:
            for d in range(shape[o['axis']]):
                p = o['points'] + (o['distance'] - d) * _delta(side)
                if _in_bounds(p, shape) and not np.any(occupied[p[:, 0], p[:, 1]]):
                    occupied[p[:, 0], p[:, 1]] = True
                    distances.append((tuple(o['pivot']), d))
                    break
            else:
                raise AssertionError('no edge-first location')
    return dict(distances)


def _make_crate(rng, bg, frame, side, h=None, w=None, tab=None,
                raw_turn=None, interior=None):
    h = int(rng.integers(3, 7)) if h is None else int(h)
    w = int(rng.integers(3, 8)) if w is None else int(w)
    tab = int(rng.integers(1, w-1)) if tab is None else int(tab)
    if interior is None:
        allowed = [c for c in range(10) if c not in (bg, frame)]
        pal = rng.choice(allowed, int(rng.integers(1, min(4, len(allowed))+1)), replace=False)
        rr, cc = np.indices((h-2, w-2))
        mode = int(rng.integers(4))
        indices = (rr+cc if mode == 0 else rr if mode == 1 else cc
                   if mode == 2 else rng.integers(len(pal), size=rr.shape))
        interior = pal[indices % len(pal)]
    body = np.full((h+1, w), bg, dtype=int)
    body[:-1] = frame
    body[1:h-1, 1:w-1] = interior
    body[-1, tab] = frame
    cells = np.argwhere(body != bg)
    target = (2-side) % 4 if side is not None else int(rng.integers(4))
    q = int(rng.integers(4)) if raw_turn is None else int(raw_turn) % 4
    return dict(frame=int(frame), side=side, h=h, w=w, tab=tab,
                relative=cells-(h, tab), values=body[body != bg].copy(),
                interior=np.array(interior, copy=True), raw_turn=q,
                target_turn=target)


def _position(o, pivot, shape):
    o = dict(o, pivot=np.array(pivot, dtype=int))
    o['raw'] = _rotate(o['relative'], o['raw_turn']) + o['pivot']
    if o['side'] is not None:
        ax = 0 if o['side'] % 2 == 0 else 1
        o['points'] = _rotate(o['relative'], o['target_turn']) + o['pivot']
        o['axis'] = ax
        o['span'] = (int(o['points'][:, 1-ax].min()), int(o['points'][:, 1-ax].max()))
        o['transverse'] = int(o['pivot'][1-ax])
        o['distance'] = int(o['pivot'][ax] if o['side'] in (0, 3)
                            else shape[ax]-1-o['pivot'][ax])
    return o


def _separated(a, b):
    bset = set(map(tuple, b))
    return all((int(r)+dr, int(c)+dc) not in bset for r, c in a
               for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def _admissible(objects, shape):
    for i, a in enumerate(objects):
        if not _in_bounds(a['raw'], shape):
            return False
        if a['side'] is not None and not _in_bounds(a['points'], shape):
            return False
        for b in objects[i+1:]:
            if not _separated(a['raw'], b['raw']):
                return False
            if (a['side'] is not None and a['side'] == b['side'] and
                    _intersect(a['points'], b['points'])):
                return False
    try:
        packed = _pack(objects, shape)
    except ValueError:
        return False
    return all(not _intersect(a['final_points'], b['final_points'])
               for i, a in enumerate(packed) for b in packed[i+1:])


def _pivot(side, distance, transverse, shape):
    ax = 0 if side % 2 == 0 else 1
    p = [0, 0]
    p[ax] = distance if side in (0, 3) else shape[ax]-1-distance
    p[1-ax] = transverse
    return p


def _chain(rng, bg, labels, side, training):
    """A real overlap chain, not three independent color-coded pieces."""
    target = (2-side) % 4
    if training:
        quarters = (3, 1, 2) if side in (0, 1) else (1, 3, 2)
        crates = [_make_crate(rng, bg, labels[side], side, 3, 5, 2,
                              raw_turn=target+q) for q in quarters]
    else:
        crates = [_make_crate(rng, bg, labels[side], side,
                              h=int(rng.integers(3, 5)), w=int(rng.integers(4, 7)),
                              raw_turn=target+int(rng.choice((0, 1, 3))))
                  for _ in range(3)]
        # Recombine the learned chain with an attachment away from the center.
        crates[1] = _make_crate(rng, bg, labels[side], side, crates[1]['h'], 5, 1,
                                raw_turn=crates[1]['raw_turn'])
    # Transverse body intervals overlap by two cells along consecutive links.
    starts = [0]
    for o in crates[:-1]:
        starts.append(starts[-1] + o['w'] - 2)
    pivots = [s + (o['w']-1-o['tab'] if side in (0, 1) else o['tab'])
              for s, o in zip(starts, crates)]
    ax = 0 if side % 2 == 0 else 1
    inward_sign = 1 if side in (0, 3) else -1
    extents, trans_extents = [], []
    for o, p in zip(crates, pivots):
        union = np.concatenate((_rotate(o['relative'], o['raw_turn']),
                                _rotate(o['relative'], o['target_turn'])))
        radial = union[:, ax] * inward_sign
        extents.append((int(radial.min()), int(radial.max())))
        trans_extents.append((int(union[:, 1-ax].min())+p,
                              int(union[:, 1-ax].max())+p))
    distances = [-extents[0][0] + int(rng.integers(0, 3))]
    for i in (1, 2):
        distances.append(distances[-1]+extents[i-1][1]-extents[i][0]+2+
                         int(rng.integers(0, 2)))
    required_radial = distances[-1] + extents[-1][1] + 1
    low = min(x[0] for x in trans_extents)
    high = max(x[1] for x in trans_extents)
    radial_size = max(22, required_radial + int(rng.integers(2, 7)))
    transverse_size = max(20, high-low+1+int(rng.integers(7, 15)))
    if radial_size > 30 or transverse_size > 30:
        return None
    shape = (radial_size, transverse_size) if ax == 0 else (transverse_size, radial_size)
    offset = int(rng.integers(-low, transverse_size-high))
    placed = [_position(o, _pivot(side, d, p+offset, shape), shape)
              for o, d, p in zip(crates, distances, pivots)]
    return shape, placed


def _add_random(rng, objects, crate, shape, align=None, disjoint=None):
    """Bounded, visible geometric placement; no output-label tricks."""
    relative = _rotate(crate['relative'], crate['raw_turn'])
    if crate['side'] is not None:
        relative = np.concatenate((relative, _rotate(crate['relative'], crate['target_turn'])))
    low = -relative.min(axis=0)
    high = np.array(shape)-1-relative.max(axis=0)
    if np.any(low > high):
        return None
    if align is not None:
        ax = 0 if crate['side'] % 2 == 0 else 1
        if not low[1-ax] <= align['transverse'] <= high[1-ax]:
            return None
        low[1-ax] = high[1-ax] = align['transverse']
    blocked = np.zeros(shape, dtype=bool)
    same_class = np.zeros(shape, dtype=bool)
    for o in objects:
        for r, c in o['raw']:
            blocked[max(0, r-1):r+2, max(0, c-1):c+2] = True
        if crate['side'] is not None and o['side'] == crate['side']:
            p = o['points']
            same_class[p[:, 0], p[:, 1]] = True
    for _ in range(240):
        pivot = [int(rng.integers(lo, hi+1)) for lo, hi in zip(low, high)]
        candidate = _position(crate, pivot, shape)
        p = candidate['raw']
        if np.any(blocked[p[:, 0], p[:, 1]]):
            continue
        if crate['side'] is not None:
            p = candidate['points']
            if np.any(same_class[p[:, 0], p[:, 1]]):
                continue
        if disjoint is not None and _span_overlap(candidate, disjoint):
            continue
        try:
            packed = _pack(objects+[candidate], shape)
        except ValueError:
            continue
        occupied = np.zeros(shape, dtype=bool)
        good = True
        for o in packed:
            p = o['final_points']
            if np.any(occupied[p[:, 0], p[:, 1]]):
                good = False
                break
            occupied[p[:, 0], p[:, 1]] = True
        if good:
            return candidate
    return None


def _finish(rng, bg, labels, shape, objects, require_card_cross=False):
    if not _admissible(objects, shape):
        return None
    packed = _pack(objects, shape)
    inp = np.full(shape, bg, dtype=int)
    output_cells = set()
    for o in objects:
        p = o['raw']
        inp[p[:, 0], p[:, 1]] = o['values']
    for o in packed:
        output_cells.update(map(tuple, o['final_points']))
    swept = set()
    if require_card_cross:
        for o in packed:
            for step in range(o['distance']-o['final_distance']+1):
                swept.update(map(tuple, o['points']+step*_delta(o['side'])))
    possible = []
    for r in range(shape[0]-3):
        for c in range(shape[1]-3):
            if np.any(inp[max(0, r-1):r+5, max(0, c-1):c+5] != bg):
                continue
            if any((rr, cc) in output_cells for rr in range(max(0, r-1), min(shape[0], r+5))
                   for cc in range(max(0, c-1), min(shape[1], c+5))):
                continue
            if require_card_cross and not any(tuple(p) in swept for p in
                    np.argwhere(_hub_patch(bg, labels) != bg)+(r, c)):
                continue
            possible.append((r, c))
    if not possible:
        return None
    hub = possible[int(rng.integers(len(possible)))]
    r, c = hub
    inp[r:r+4, c:c+4] = _hub_patch(bg, labels)
    if np.count_nonzero(inp == bg) * 2 <= inp.size:
        return None
    return dict(input=inp, output=solve(inp))


def _make_pair(rng, palette, labels, kind, main_side, bg):
    absent = next(c for c in palette[1:6] if c not in labels)
    for _ in range(80):
        if kind in ('witness', 'test'):
            result = _chain(rng, bg, labels, main_side, kind == 'witness')
            if result is None:
                continue
            shape, objects = result
            sides = [(main_side+2) % 4] if kind == 'witness' else list(
                map(int, rng.integers(0, 4, size=int(rng.integers(0, 4)))))
            requests = [_make_crate(rng, bg, labels[s], s, 4, 3, 1,
                                    raw_turn=(2-s) % 4) if kind == 'witness'
                        else _make_crate(rng, bg, labels[s], s) for s in sides]
            distractors = 1 if kind == 'witness' else int(rng.integers(0, 3))
        else:
            shape = tuple(map(int, rng.integers(17, 31, size=2)))
            objects = []
            a, b = (main_side+1) % 4, (main_side+3) % 4
            if kind == 'contrast':
                neutral = int(rng.choice([c for c in range(10)
                                          if c not in (bg, labels[a], labels[b])]))
                shared = np.full((1, 3), neutral, dtype=int)
                # Two aligned blockers; a disjoint band; equal payloads under
                # different frames; an off-center attachment in training.
                requests = [_make_crate(rng, bg, labels[a], a, 3, 5, 2, interior=shared),
                            _make_crate(rng, bg, labels[a], a, 4, 5, 2),
                            _make_crate(rng, bg, labels[b], b, 3, 5, 2, interior=shared),
                            _make_crate(rng, bg, labels[a], a, 4, 5, 1)]
                distractors = 1
            else:
                requests = [_make_crate(rng, bg, labels[int(s)], int(s))
                            for s in rng.integers(0, 4, size=int(rng.integers(1, 8)))]
                distractors = int(rng.integers(0, 3))
        success = True
        for i, crate in enumerate(requests):
            aligned = objects[0] if kind == 'contrast' and i == 1 else None
            disjoint = objects[0] if kind == 'contrast' and i == 3 else None
            o = _add_random(rng, objects, crate, shape, aligned, disjoint)
            if o is None:
                success = False
                break
            objects.append(o)
        if not success:
            continue
        if kind == 'witness' and not _opposite_cross(objects):
            continue
        for _ in range(distractors):
            o = _add_random(rng, objects, _make_crate(rng, bg, absent, None), shape)
            if o is None:
                success = False
                break
            objects.append(o)
        if success:
            pair = _finish(rng, bg, labels, shape, objects, kind == 'witness')
            if pair is not None:
                return pair
    raise RuntimeError('bounded geometric construction exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.permutation(10)))
    main_side = int(rng.integers(4))
    first = tuple(map(int, rng.permutation(palette[1:5])))
    # Explicit training evidence for local labels and match/distractor role reuse.
    second = list(first)
    second[main_side] = palette[5]
    second = tuple(np.roll(second, 1).tolist())
    puzzle = {'train': [], 'test': []}
    seen = set()
    for i in range(num_train + num_test):
        split = 'train' if i < num_train else 'test'
        kind = 'witness' if i == 0 else 'contrast' if i == 1 else (
            'test' if split == 'test' else 'free')
        labels = first if i == 0 else second if i == 1 else tuple(
            map(int, rng.choice(palette[1:6], 4, replace=False)))
        bg = palette[0] if i == 0 else int(rng.choice((palette[0], palette[6], palette[7])))
        for _ in range(10):
            pair = _make_pair(rng, palette, labels, kind,
                              main_side if split == 'train' else int(rng.integers(4)), bg)
            if _key(pair['input']) not in seen:
                break
        else:
            raise RuntimeError('could not construct distinct inputs')
        seen.add(_key(pair['input']))
        puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    g, bg, hub, labels, objects = _decode(pair['input'])
    packed = _pack(objects, g.shape)
    stable = {tuple(o['pivot']): o['final_distance'] for o in packed}
    raw_spans = {tuple(o['pivot']): (int(o['raw'][:, 1-o['axis']].min()),
                                    int(o['raw'][:, 1-o['axis']].max()))
                 for o in objects if o['side'] is not None}
    created = False
    for i, a in enumerate(objects):
        if a['side'] is None:
            continue
        for b in objects[i+1:]:
            if a['side'] == b['side'] and _span_overlap(a, b):
                x, y = raw_spans[tuple(a['pivot'])], raw_spans[tuple(b['pivot'])]
                created |= max(x[0], y[0]) > min(x[1], y[1])
    hub_cells = set(map(tuple, np.argwhere(_hub_patch(bg, labels) != bg)+hub))
    card_cross = any(not hub_cells.isdisjoint(map(tuple, o['points']+step*_delta(o['side'])))
                     for o in packed for step in range(o['distance']-o['final_distance']+1))
    return dict(bg=bg, hub=hub, labels=labels, objects=objects, packed=packed,
                witness=stable != _edge_first(objects, g.shape), created=created,
                card_cross=card_cross, opposite_cross=_opposite_cross(objects),
                rotations={(o['target_turn']-o['raw_turn']) % 4
                           for o in objects if o['side'] is not None})


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            f = _features(dict(input=inp, output=out))
            objects = f['objects']
            # Separate pixel-motion reconstruction, not the generator's formula.
            simulated = _simulate(objects, inp.shape)
            expected = _render(inp.shape, f['bg'], f['hub'], f['labels'], simulated)
            assert np.array_equal(out, expected), 'rotation, payload or stable-contact mismatch'
            assert np.array_equal(out, solve(inp)), 'input-only prediction mismatch'
            assert np.count_nonzero(out == f['bg']) * 2 > out.size
            assert sum(len(o['final_points']) for o in simulated)+8 == np.count_nonzero(out != f['bg'])
            for o in simulated:
                assert np.all((o['final_points']-o['points'])[:, 1-o['axis']] == 0)
                step = o['final_points'] + _delta(o['side'])
                others = [b for b in simulated if b is not o and b['side'] == o['side']]
                assert not _in_bounds(step, inp.shape) or any(
                    _intersect(step, b['final_points']) for b in others), 'not fully compacted'
            records[split].append(f)
    train = records['train']
    assert any(f['witness'] for f in train), 'stable chain must be demonstrated'
    assert any(f['created'] for f in train), 'turn-created blocking must be demonstrated'
    assert any(f['card_cross'] for f in train), 'compass transparency must be demonstrated'
    assert any(f['opposite_cross'] for f in train), 'opposite-class passing must be demonstrated'
    assert set.union(*(f['rotations'] for f in train)) == {0, 1, 2, 3}
    assert {o['side'] for f in train for o in f['objects'] if o['side'] is not None} == set(range(4))
    assert any(o['side'] is None for f in train for o in f['objects']), 'filtering witness absent'
    assert any(o['tab']*2 != o['w']-1 for f in train for o in f['objects']
               if o['side'] is not None), 'off-center pivot not taught'
    assert any(a['frame'] != b['frame'] and a['h'] == b['h'] and a['w'] == b['w']
               and np.array_equal(a['interior'], b['interior'])
               for f in train for a in f['objects'] for b in f['objects']
               if a['side'] is not None and b['side'] is not None), 'payload/direction confound'
    roles = {}
    for f in train:
        for side, color in enumerate(f['labels']):
            roles.setdefault(color, set()).add(side)
    assert any(len(sides) > 1 for sides in roles.values()), 'local compass mapping not taught'
    matches = {o['frame'] for f in train for o in f['objects'] if o['side'] is not None}
    absent = {o['frame'] for f in train for o in f['objects'] if o['side'] is None}
    assert matches & absent, 'match/distractor role reuse not taught'


def self_test():
    """Bounded semantic regressions and actual structural coverage over 42 seeds."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures, shapes, geometry, directions, tabs, counts, backgrounds = (set() for _ in range(7))
    patterned, nesting, witnesses, created = 0, 0, 0, 0
    for puzzle in puzzles:
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _features(pair)
                shapes.add(pair['input'].shape)
                backgrounds.add(f['bg'])
                counts.add(len(f['objects']))
                witnesses += f['witness']
                created += f['created']
                for o in f['objects']:
                    geometry.add((o['h'], o['w'], o['tab']))
                    tabs.add(o['tab'])
                    patterned += len(np.unique(o['interior'])) > 1
                    if o['side'] is not None:
                        directions.add(o['side'])
                # A follower contacts a body's outer row with its tab alongside.
                for a in f['packed']:
                    for b in f['packed']:
                        nesting += (a is not b and a['side'] == b['side'] and _span_overlap(a, b)
                                    and a['final_distance'] == b['final_distance']+b['h']
                                    and not b['span'][0] <= a['transverse'] <= b['span'][1])
    assert len(signatures) == 42 and len(shapes) >= 45
    assert len(geometry) >= 45 and len(counts) >= 6 and len(backgrounds) >= 7
    assert directions == {0, 1, 2, 3} and len(tabs) >= 4
    assert patterned >= 100 and nesting >= 42 and witnesses >= 42 and created >= 42
    for a, b in ((2, 1), (10, 4), (7, 3)):
        p = generate_puzzle(314159, a, b)
        q = generate_puzzle(314159, a, b)
        assert all(np.array_equal(x[k], y[k]) for s in p
                   for x, y in zip(p[s], q[s]) for k in ('input', 'output'))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # The known chain must settle at 0,3,6, not bounding-box 0,4,8 or jump 0,3,0.
    f = _features(puzzles[0]['train'][0])
    groups = [[o for o in f['packed'] if o['side'] == s] for s in range(4)]
    chain = next(g for g in groups if len(g) == 3)
    assert sorted(o['final_distance'] for o in chain) == [0, 3, 6]
    old = _edge_first(f['objects'], puzzles[0]['train'][0]['input'].shape)
    assert sorted(old[tuple(o['pivot'])] for o in chain) == [0, 0, 3]
    # Whole-scene rotation and recoloring must transform instructions visibly.
    pair = puzzles[0]['train'][0]
    for turns in (1, 2, 3):
        assert np.array_equal(solve(np.rot90(pair['input'], turns)),
                              np.rot90(pair['output'], turns))
    perm = np.array((6, 3, 9, 0, 8, 1, 4, 7, 2, 5))
    assert np.array_equal(solve(perm[pair['input']]), perm[pair['output']])
    # A compatible body outside the sampled size range, and all-unmatched erasure.
    large = np.zeros((30, 25), dtype=int)
    hub = _hub_patch(0, (1, 2, 3, 4))
    body = np.full((12, 8), 3, dtype=int)
    rr, cc = np.indices((10, 6))
    body[1:-1, 1:-1] = 5+(rr+cc) % 2
    large[13:17, 2:6] = hub
    large[3:15, 14:22] = body
    large[15, 15] = 3
    expected = np.zeros_like(large)
    expected[13:17, 2:6] = hub
    expected[17:29, 14:22] = body
    expected[29, 15] = 3
    assert np.array_equal(solve(large), expected)
    unmatched = large.copy()
    patch = unmatched[3:15, 14:22]
    patch[patch == 3] = 7
    unmatched[15, 15] = 7
    bare = np.zeros_like(large)
    bare[13:17, 2:6] = hub
    assert np.array_equal(solve(unmatched), bare)
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
               for s in ('train', 'test')}
    damaged['test'][0]['output'][0, 0] = (damaged['test'][0]['output'][0, 0]+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('damaged output accepted')
