"""Corner siphons: cut a keyed stencil into capacity-limited path collectors.

Only the visible input is used by solve. Randomness is confined to construction.
"""
from collections import deque

import numpy as np


STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
SELF_TEST_SEEDS = (
    0, 885204840, 950032658, 658896162, 843615858, 978923929,
    531702224, 678024570, 680907664, 36705796, 543081012, 801067918,
    355085172, 599699881, 536289870, 791896993, 474298976, 95086053,
    915991198, 510359722, 10931466, 145517156, 418326620, 557807063,
    156050687, 734887960, 821539545, 589398685, 916549413, 275690238,
    647098561, 599478, 306768831, 733913642, 474307017, 200702879,
    689950296, 247414683, 95579540, 301563944, 53925918, 336571665,
)


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "Invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "Grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "Invalid grid colors")
    return a


def _neighbors(p):
    r, c = p
    return [(r + dr, c + dc) for dr, dc in STEPS]


def _components(cells):
    remaining, result = set(cells), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        found, queue = {start}, [start]
        for p in queue:
            for q in _neighbors(p):
                if q in remaining:
                    remaining.remove(q)
                    found.add(q)
                    queue.append(q)
        result.append(found)
    return result


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _analyze(value):
    a = _grid(value)
    h, w = a.shape
    _require(h >= 7 and w >= 7, "A keyed stencil needs at least a 7x7 canvas")
    rim = np.zeros(a.shape, dtype=bool)
    rim[[0, -1], :] = True
    rim[:, [0, -1]] = True
    colors, counts = np.unique(a[rim], return_counts=True)
    bg = int(colors[np.argmax(counts)])
    marks = [tuple(map(int, p)) for p in np.argwhere(rim & (a != bg))]
    _require(1 <= len(marks) <= 2, "Need one or two border markers")
    ink = int(a[marks[0]])
    _require(all(a[p] == ink for p in marks), "Border markers must share a color")
    sides, ar, ac = set(), None, None
    for r, c in marks:
        flags = [s for s, ok in (("top", r == 0), ("bottom", r == h - 1),
                                 ("left", c == 0), ("right", c == w - 1)) if ok]
        _require(len(flags) == 1 and flags[0] not in sides, "Markers cannot be corners or share a side")
        side = flags[0]
        sides.add(side)
        if side in ("top", "bottom"):
            ac = c
        else:
            ar = r
    _require(len(sides) == 1 or (ar is not None and ac is not None), "Two markers must be on adjacent sides")
    rr, cc = np.indices(a.shape)
    axes = np.zeros(a.shape, dtype=bool)
    if ar is not None:
        axes |= rr == ar
    if ac is not None:
        axes |= cc == ac
    centers = np.argwhere(axes & ~rim & (a == ink))
    _require(len(centers) == 1, "Exactly one interior ink cell may lie on the axes")
    center = tuple(map(int, centers[0]))
    cr, cc0 = center
    if len(sides) == 2:
        _require(center == (ar, ac), "The key must lie at the axes' intersection")
    _require(1 <= cr < h - 1 and 1 <= cc0 < w - 1, "Key must have a complete ring")
    ring = [(cr + dr, cc0 + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
    slot = int(a[ring[0]])
    _require(slot not in (bg, ink) and all(a[p] == slot for p in ring), "Need a uniform eight-cell ring around the key")
    extra = set(map(int, np.unique(a))) - {bg, ink, slot}
    _require(len(extra) <= 1, "Only one wall color is allowed")
    wall_color = next(iter(extra)) if extra else None
    wall = (a == wall_color) if wall_color is not None else np.zeros(a.shape, dtype=bool)
    _require(not np.any(wall & (axes | rim)), "Walls cannot occupy axes or the border")
    stencil = [tuple(map(int, p)) for p in np.argwhere(a == slot)]
    _require(not any(rim[p] for p in stencil), "Input stencil must be interior")
    _require(len(_components(stencil)) == 1, "Input stencil must be connected")
    groups = {}
    one_side = next(iter(sides)) if len(sides) == 1 else None
    for r, c in stencil:
        if axes[r, c]:
            continue
        corner = (("top" if r < ar else "bottom") if ar is not None else one_side,
                  ("left" if c < ac else "right") if ac is not None else one_side)
        groups.setdefault(corner, []).append((r, c))
    _require(len(groups) == (4 if len(sides) == 2 else 2), "All cut regions must contain slots")
    collectors = {}
    for corner, cells in sorted(groups.items()):
        if not (set(corner) & sides):
            continue
        rs, cs = zip(*cells)
        dr = -min(rs) if corner[0] == "top" else h - 1 - max(rs)
        dc = -min(cs) if corner[1] == "left" else w - 1 - max(cs)
        targets = sorted((r + dr, c + dc) for r, c in cells)
        region = ~axes
        if ar is not None:
            region = region & ((rr < ar) if corner[0] == "top" else (rr > ar))
        if ac is not None:
            region = region & ((cc < ac) if corner[1] == "left" else (cc > ac))
        _require(all(region[p] for p in targets), "A moved stencil cannot cross an axis")
        _require(all(a[p] in (bg, slot) for p in targets), "A dock cannot overwrite walls or supply")
        collectors[corner] = dict(source=sorted(cells), target=targets, region=region, capacity=len(cells))
    guides = []
    gaps = []
    for r, c in marks:
        if r in (0, h - 1):
            hits = sorted(p[0] for p in stencil if p[1] == c)
            _require(hits, "Each guide must intersect the stencil")
            end = max(hits) if r == 0 else min(hits)
            path = [(i, c) for i in range(min(r, end), max(r, end) + 1)]
        else:
            hits = sorted(p[1] for p in stencil if p[0] == r)
            _require(hits, "Each guide must intersect the stencil")
            end = max(hits) if c == 0 else min(hits)
            path = [(r, i) for i in range(min(c, end), max(c, end) + 1)]
        guides.append(path)
        gaps.extend(p for p in path if a[p] == bg and
                    (min(hits) < (p[0] if r in (0, h - 1) else p[1]) < max(hits)))
    supply = (a == ink) & ~axes & ~rim
    return dict(grid=a, bg=bg, ink=ink, slot=slot, wall=wall, wall_color=wall_color,
                axes=axes, ar=ar, ac=ac, center=center, marks=marks, sides=sides,
                stencil=stencil, groups=groups, collectors=collectors, guides=guides,
                internal_gaps=gaps, supply=supply, rim=rim)


def _distances(region, wall, starts):
    h, w = region.shape
    d = np.full(region.shape, -1, dtype=np.int16)
    queue = deque()
    for p in starts:
        _require(region[p] and not wall[p], "Collector roots must be traversable")
        d[p] = 0
        queue.append(p)
    while queue:
        r, c = queue.popleft()
        for dr, dc in STEPS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and region[nr, nc] and not wall[nr, nc] and d[nr, nc] < 0:
                d[nr, nc] = d[r, c] + 1
                queue.append((nr, nc))
    return d


def _select(supply, distance, capacity, strict=True):
    cells = [tuple(map(int, p)) for p in np.argwhere(supply & (distance >= 0))]
    cells.sort(key=lambda p: (int(distance[p]), p))
    if strict and len(cells) > capacity:
        _require(distance[cells[capacity - 1]] < distance[cells[capacity]], "A capacity frontier cannot split a distance tie")
    return cells[:capacity], cells[capacity:]


def _collections(info, original_roots=False, ignore_walls=False, unlimited=False):
    result = {}
    wall = np.zeros_like(info['wall']) if ignore_walls else info['wall']
    for corner, collector in info['collectors'].items():
        starts = collector['source'] if original_roots else collector['target']
        d = _distances(collector['region'], wall, starts)
        capacity = info['grid'].size if unlimited else collector['capacity']
        selected, excess = _select(info['supply'] & collector['region'], d, capacity, strict=not original_roots and not ignore_walls)
        result[corner] = dict(selected=selected, excess=excess, distance=d)
    return result


def _render(info, collections, column_order=False):
    out = info['grid'].copy()
    for cells in info['groups'].values():
        for p in cells:
            out[p] = info['bg']
    for path in info['guides']:
        for p in path:
            if out[p] == info['bg']:
                out[p] = info['ink']
    for corner, collector in info['collectors'].items():
        selected = collections[corner]['selected']
        for p in selected:
            out[p] = info['bg']
        slots = sorted(collector['target'], key=(lambda p: (p[1], p[0])) if column_order else None)
        for i, p in enumerate(slots):
            out[p] = info['ink'] if i < len(selected) else info['slot']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Complete guides, dock stencil pieces, and siphon by path distance and capacity."""
    info = _analyze(input_grid)
    return _render(info, _collections(info))


def _base_stencil():
    # Ring, asymmetric branches, off-center axial gaps, and a separated upper-right cut.
    cells = {(3 + dr, 3 + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc}
    cells |= {(0, 2), (0, 3), (0, 4), (1, 2), (2, 1), (2, 0),
              (5, 2), (6, 2), (6, 3), (6, 4), (4, 5), (4, 6), (3, 6)}
    return cells, (3, 3), (7, 7)


def _stencil(rng, witness=False, fallback=False):
    if witness or fallback:
        base, center, shape = _base_stencil()
    else:
        shape = tuple(map(int, rng.integers(5, 12, size=2)))
        center = (int(rng.integers(2, shape[0] - 2)), int(rng.integers(2, shape[1] - 2)))
        r, c = center
        base = {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc}
    if fallback:
        return base, center, shape
    cr, cc = center
    for _ in range(12):
        cells = set(base)
        amount = int(rng.integers(len(base), min(52, shape[0] * shape[1] - 1) + 1))
        thin = bool(rng.integers(2))
        while len(cells) < amount:
            frontier = sorted({q for p in cells for q in _neighbors(p)
                               if 0 <= q[0] < shape[0] and 0 <= q[1] < shape[1] and q not in cells and q != center})
            if witness:
                # Keep both the off-center guide gap and the isolated upper-right fragment.
                frontier = [p for p in frontier if p not in {(1, 3), (1, 4), (0, 5), (1, 5)}]
            if not frontier:
                break
            if thin:
                degree = [sum(q in cells for q in _neighbors(p)) for p in frontier]
                frontier = [p for p, n in zip(frontier, degree) if n == min(degree)]
            cells.add(frontier[int(rng.integers(len(frontier)))])
        if not witness or len(_components(p for p in cells if p[0] < cr and p[1] > cc)) > 1:
            return cells, center, shape
    return base, center, shape


def _blank(rng, dimensions, double, roles, witness=False, fallback=False):
    h, w = dimensions
    cells, center, shape = _stencil(rng, witness, fallback)
    ph, pw = shape
    if fallback:
        r0, c0 = h // 2 - center[0], w // 2 - center[1]
    else:
        r0 = int(rng.integers(2, h - ph - 1))
        c0 = int(rng.integers(2, w - pw - 1))
    ar, ac = center[0] + r0, center[1] + c0
    bg, ink, slot, _ = roles
    a = np.full((h, w), bg, dtype=np.int8)
    for r, c in cells:
        a[r + r0, c + c0] = slot
    a[ar, ac] = ink
    a[0, ac] = ink
    if double:
        a[ar, -1] = ink
    return a


def _reserved(info):
    reserved = info['axes'] | info['rim'] | (info['grid'] == info['slot'])
    for collector in info['collectors'].values():
        for p in collector['target']:
            reserved[p] = True
    return reserved


def _pocket(rng, wall, allowed):
    h, w = wall.shape
    candidates = []
    for r in range(2, h - 2):
        for c in range(2, w - 2):
            if np.all(allowed[r - 1:r + 2, c - 1:c + 2]):
                candidates.append((r, c))
    if not candidates:
        return False
    r, c = candidates[int(rng.integers(len(candidates)))]
    wall[r - 1:r + 2, c - 1:c + 2] = True
    wall[r, c] = False
    return True


def _walls(rng, info, kind, require_pocket=False):
    h, w = info['grid'].shape
    wall = np.zeros((h, w), dtype=bool)
    reserved = _reserved(info)
    if kind == 'fallback':
        # A corner-fed U detour: a nearby supply behind it needs a long route.
        ar, ac = info['center']
        dock = info['collectors'][('top', 'left')]['target']
        barrier_c = max(c for r, c in dock) + 2
        wall[1, barrier_c:ac] = True
        wall[1:ar - 2, barrier_c] = True
        wall[reserved] = False
    elif kind != 'open':
        amount = int(rng.integers(3, 8)) if kind == 'rails' else int(rng.integers(6, 17))
        for _ in range(amount):
            r0, r1 = sorted(map(int, rng.integers(1, h - 1, size=2)))
            c0, c1 = sorted(map(int, rng.integers(1, w - 1, size=2)))
            primitive = int(rng.integers(4))
            if primitive in (0, 2, 3):
                wall[r0, c0:c1 + 1] = True
            if primitive in (1, 2, 3):
                wall[r0:r1 + 1, c1] = True
            if primitive == 3 and kind == 'rooms':
                wall[r1, c0:c1 + 1] = True
                wall[r0:r1 + 1, c0] = True
                if rng.random() < .7 and r1 - r0 > 2:
                    wall[int(rng.integers(r0 + 1, r1)), c0] = False
        wall[reserved] = False
    if require_pocket or (kind != 'open' and rng.random() < .65):
        corners = list(info['collectors'])
        if kind == 'fallback' and len(corners) > 1:
            corners = [c for c in corners if c != ('top', 'left')]
        corner = corners[int(rng.integers(len(corners)))]
        ok = _pocket(rng, wall, info['collectors'][corner]['region'] & ~reserved)
        if require_pocket and not ok:
            return None
    return wall


def _rotpoint(p, shape, turns):
    r, c = p
    h, w = shape
    for _ in range(turns):
        r, c, h, w = c, h - 1 - r, w, h
    return r, c


def _pick(rng, mask, count, style='dispersed', forced=()):
    chosen = list(dict.fromkeys(forced))
    pool = [tuple(map(int, p)) for p in np.argwhere(mask) if tuple(map(int, p)) not in chosen]
    if len(pool) + len(chosen) < count:
        return None
    if count == len(chosen):
        return chosen
    anchor = chosen[0] if chosen else pool[int(rng.integers(len(pool)))]
    if style != 'dispersed':
        rng.shuffle(pool)
        r, c = anchor
        if style == 'compact':
            pool.sort(key=lambda p: (p[0] - r) ** 2 + (p[1] - c) ** 2)
        elif rng.integers(2):
            pool.sort(key=lambda p: abs(p[0] - r) * 40 + abs(p[1] - c))
        else:
            pool.sort(key=lambda p: abs(p[1] - c) * 40 + abs(p[0] - r))
    else:
        rng.shuffle(pool)
    return chosen + pool[:count - len(chosen)]


def _supply(rng, available, d, roots, capacity, count, style, reversal=False):
    active = available & (d >= 0)
    if count <= capacity:
        return _pick(rng, active, count, style)
    rr, cc = np.indices(d.shape)
    geometric = np.minimum.reduce([np.abs(rr - r) + np.abs(cc - c) for r, c in roots])
    candidates = [tuple(map(int, p)) for p in np.argwhere(active)]
    rng.shuffle(candidates)
    for a in candidates:
        near = active & (d <= d[a])
        far = active & (d > d[a])
        if near.sum() < capacity or far.sum() < count - capacity:
            continue
        b_pool = far
        if reversal:
            b_pool = b_pool & (geometric < geometric[a])
        if not b_pool.any():
            continue
        b_candidates = np.argwhere(b_pool)
        b = tuple(map(int, b_candidates[int(rng.integers(len(b_candidates)))]))
        left = _pick(rng, near, capacity, style, [a])
        right = _pick(rng, far, count - capacity, style, [b])
        return left + right
    return None


def _build_pair(rng, dimensions, double, roles, turns, obligation=None, complex_test=False, fallback_only=False):
    for attempt in range(96 if fallback_only else 0, 112):
        fallback = attempt >= 96
        if fallback:
            dimensions = (max(20, dimensions[0]), max(20, dimensions[1]))
        a = _blank(rng, dimensions, double, roles, witness=(obligation == 'paths' or complex_test), fallback=fallback)
        info = _analyze(a)
        kind = 'fallback' if fallback else str(rng.choice(['open', 'rails', 'turns', 'rooms'], p=[.12, .25, .32, .31]))
        if obligation == 'paths' and kind == 'open':
            kind = 'turns'
        wall = _walls(rng, info, kind, require_pocket=obligation == 'paths')
        if wall is None:
            continue
        a[wall] = roles[3]
        info = _analyze(a)
        available = ~_reserved(info) & ~wall
        # Destinations, axes, source stencil, and outer rim never carry loose supply.
        keys = list(info['collectors'])
        rng.shuffle(keys)
        special = None
        if obligation == 'paths':
            special = ('top', 'left') if fallback else keys[0]
        elif obligation == 'partial':
            for corner in keys:
                ts = sorted(_rotpoint(p, a.shape, turns) for p in info['collectors'][corner]['target'])
                ns = [n for n in range(1, len(ts)) if set(ts[:n]) != set(sorted(ts, key=lambda p: (p[1], p[0]))[:n])]
                if ns:
                    special, partial_loads = corner, ns
                    break
            if special is None:
                continue
        supply, failed = [], False
        for corner in keys:
            col = info['collectors'][corner]
            d = _distances(col['region'], wall, col['target'])
            cap = col['capacity']
            if corner == special and obligation == 'partial':
                count = int(rng.choice(partial_loads))
            elif corner == special and obligation == 'paths':
                count = cap + int(rng.integers(1, 7))
            else:
                draw = rng.random()
                count = 0 if draw < .10 else cap + int(rng.integers(1, 9)) if draw < .33 else int(rng.integers(1, cap + 1))
            chosen = _supply(rng, available & col['region'], d, col['target'], cap, count,
                             str(rng.choice(['dispersed', 'compact', 'line'])),
                             reversal=(corner == special and obligation == 'paths'))
            if chosen is None:
                failed = True
                break
            supply.extend(chosen)
        if failed:
            continue
        inactive = np.zeros(a.shape, dtype=bool)
        for col in info['collectors'].values():
            d = _distances(col['region'], wall, col['target'])
            inactive |= col['region'] & (d < 0)
        unreachable_pool = available & inactive
        minimum = 1 if obligation == 'paths' else 0
        if unreachable_pool.sum() < minimum:
            continue
        n = int(rng.integers(minimum, min(8, int(unreachable_pool.sum())) + 1))
        supply.extend(_pick(rng, unreachable_pool, n, str(rng.choice(['dispersed', 'compact', 'line']))))
        if double:
            unselected = next(corner for corner in info['groups'] if corner not in info['collectors'])
            rr, cc = np.indices(a.shape)
            region = ((rr < info['ar']) if unselected[0] == 'top' else (rr > info['ar']))
            region &= ((cc < info['ac']) if unselected[1] == 'left' else (cc > info['ac']))
            pool = available & region
            minimum = 1 if obligation == 'paths' else 0
            if pool.sum() < minimum:
                continue
            n = int(rng.integers(minimum, min(10, int(pool.sum())) + 1))
            supply.extend(_pick(rng, pool, n, str(rng.choice(['dispersed', 'compact', 'line']))))
        for p in supply:
            a[p] = roles[1]
        a = np.rot90(a, -turns).copy()
        f = pair_features(a)
        if obligation == 'paths' and not (f['path_beats_geometry'] and f['origin_sensitive'] and f['unreachable'] and f['internal_gap'] and f['fragmented']):
            continue
        if obligation == 'partial' and not f['order_sensitive']:
            continue
        return {'input': a, 'output': solve(a)}
    raise RuntimeError('Bounded corner-siphon construction failed')


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    if not _integer(num_train) or not _integer(num_test):
        raise ValueError('Pair counts must be integers')
    nt, ne = int(num_train), int(num_test)
    if not (2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne):
        raise ValueError('Need 2..10 train, 1..4 test, and more train than test')
    rng = np.random.default_rng(int(seed))
    roles = tuple(map(int, rng.choice(10, 4, replace=False)))
    varying_palette = rng.random() < .65
    turn = int(rng.integers(4))
    shift = int(rng.integers(3))
    requests = [(False, 'partial'), (True, 'paths')]
    requests += [(bool(rng.integers(2)), None) for _ in range(nt - 2)]
    rng.shuffle(requests)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        specs = requests if split == 'train' else [(bool(rng.integers(2)), None) for _ in range(ne)]
        for double, obligation in specs:
            if shift == 0:
                dims = tuple(map(int, rng.integers(20, 26, 2) if split == 'train' else rng.integers(26, 31, 2)))
                turns = turn if obligation else int(rng.integers(4))
            elif shift == 1:
                dims = tuple(map(int, rng.integers(20, 31, 2)))
                turns = turn if split == 'train' else (turn + int(rng.integers(1, 4))) % 4
            else:
                dims = tuple(map(int, rng.integers(20, 31, 2)))
                turns = int(rng.integers(4))
            if varying_palette:
                roles = tuple(map(int, rng.choice(10, 4, replace=False)))
            for _ in range(16):
                pair = _build_pair(rng, dims, double, roles, turns, obligation,
                                   complex_test=(split == 'test' and shift == 2))
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('Unable to construct distinct inputs')
    validate_generated_puzzle(puzzle)
    return puzzle


def pair_features(input_grid):
    """Task-specific measurements reconstructed entirely from a visible input."""
    info = _analyze(input_grid)
    collections = _collections(info)
    old = _collections(info, original_roots=True)
    rr, cc = np.indices(info['grid'].shape)
    loads, capacities, excess, unreachable = [], [], 0, 0
    path_beats_geometry = origin_sensitive = order_sensitive = False
    max_distance = 0
    for corner, col in info['collectors'].items():
        record = collections[corner]
        selected, retained, d = record['selected'], record['excess'], record['distance']
        loads.append(len(selected))
        capacities.append(col['capacity'])
        excess += len(retained)
        unreachable += int((info['supply'] & col['region'] & (d < 0)).sum())
        metric = np.minimum.reduce([np.abs(rr - r) + np.abs(cc - c) for r, c in col['target']])
        if selected and retained:
            path_beats_geometry |= max(metric[p] for p in selected) > min(metric[p] for p in retained)
        origin_sensitive |= set(selected) != set(old[corner]['selected'])
        targets = col['target']
        order_sensitive |= set(targets[:len(selected)]) != set(sorted(targets, key=lambda p: (p[1], p[0]))[:len(selected)])
        if np.any(d >= 0):
            max_distance = max(max_distance, int(d.max()))
    inactive_supply = info['supply'].copy()
    for col in info['collectors'].values():
        inactive_supply &= ~col['region']
    return dict(shape=list(info['grid'].shape), roles=[info[k] for k in ('bg', 'ink', 'slot', 'wall_color')],
                orientation='+'.join(sorted(info['sides'])), marker_count=len(info['sides']),
                stencil_size=len(info['stencil']), capacities=capacities, loads=loads,
                excess=excess, unreachable=unreachable, inactive=int(inactive_supply.sum()),
                internal_gap=bool(info['internal_gaps']), gap_cells=len(info['internal_gaps']),
                fragmented=any(len(_components(col['source'])) > 1 for col in info['collectors'].values()),
                partition_components=[len(_components(col['source'])) for col in info['collectors'].values()],
                path_beats_geometry=bool(path_beats_geometry), origin_sensitive=bool(origin_sensitive),
                order_sensitive=bool(order_sensitive), wall_count=int(info['wall'].sum()),
                max_distance=max_distance, deleted_slots=sum(len(v) for k, v in info['groups'].items() if k not in info['collectors']))


def validate_generated_puzzle(puzzle):
    _require(type(puzzle) is dict and set(puzzle) == {'train', 'test'}, 'Invalid puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'Splits must be lists')
    _require(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4 and len(puzzle['train']) > len(puzzle['test']), 'Invalid counts')
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(type(pair) is dict and set(pair) == {'input', 'output'}, 'Invalid pair schema')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape, 'Dimensions must be preserved')
            _require(_key(a) not in seen, 'Duplicate input')
            seen.add(_key(a))
            info = _analyze(a)
            collections = _collections(info)
            _require(np.array_equal(_render(info, collections), b), 'Incorrect corner-siphon output')
            _require(np.array_equal(a[info['wall']], b[info['wall']]), 'Walls must be preserved')
            _require(np.array_equal(a[info['axes'] & (a == info['slot'])], b[info['axes'] & (a == info['slot'])]), 'Axial stencil must be preserved')
            _require(b[info['center']] == info['ink'], 'Key must be preserved')
            _require(np.count_nonzero(a == info['ink']) + sum(sum(a[p] == info['bg'] for p in path) for path in info['guides']) == np.count_nonzero(b == info['ink']), 'Ink must be conserved except new guide cells')
            selected = {p for record in collections.values() for p in record['selected']}
            deleted_slots = sum(len(cells) for corner, cells in info['groups'].items() if corner not in info['collectors'])
            _require(np.count_nonzero(b == info['slot']) == len(info['stencil']) - deleted_slots - len(selected),
                     'Stencil cells must be moved, discarded, or filled exactly once')
            for p in map(tuple, np.argwhere(info['supply'])):
                _require(b[p] == (info['bg'] if p in selected else info['ink']), 'Selected supply alone may disappear')
            for corner, col in info['collectors'].items():
                record = collections[corner]
                _require(len(record['selected']) <= col['capacity'], 'Capacity exceeded')
                _require(len(record['selected']) == min(col['capacity'], len(record['selected']) + len(record['excess'])), 'Wrong collection count')
                for i, p in enumerate(col['target']):
                    _require(b[p] == (info['ink'] if i < len(record['selected']) else info['slot']), 'Wrong displayed row-major prefix')
            features[split].append(pair_features(a))
    train = features['train']
    _require(any(f['marker_count'] == 1 and f['order_sensitive'] for f in train), 'Missing partial one-marker ordering demonstration')
    _require(any(f['marker_count'] == 2 and f['deleted_slots'] and f['inactive'] and f['excess'] and f['unreachable'] and
                 f['internal_gap'] and f['fragmented'] and f['path_beats_geometry'] and f['origin_sensitive'] for f in train),
             'Missing integrated axes/path/capacity demonstration')


def self_test():
    """Bounded 42-seed semantic checks, structural variation, and diagnostic mutations."""
    puzzles, fs, signatures, normalized = [], [], set(), set()
    orientations = {'train': set(), 'test': set()}
    for seed in SELF_TEST_SEEDS:
        p = generate_puzzle(seed)
        puzzles.append(p)
        signatures.add(tuple(_key(pair['input']) for s in ('train', 'test') for pair in p[s]))
        for split in ('train', 'test'):
            for pair in p[split]:
                a, b = pair['input'], pair['output']
                frozen = a.copy()
                _require(np.array_equal(solve(a), b) and np.array_equal(a, frozen), 'Input-only solve/mutation check failed')
                f = pair_features(a)
                fs.append(f)
                orientations[split].add(f['orientation'])
                info = _analyze(a)
                role_map = {info['bg']: 0, info['ink']: 1, info['slot']: 2}
                if info['wall_color'] is not None:
                    role_map[info['wall_color']] = 3
                normalized.add((a.shape, bytes(role_map[int(c)] for c in a.ravel())))
        # Check both inherited mechanisms on actual demonstrations.
        integrated = next(pair for pair in p['train'] if pair_features(pair['input'])['path_beats_geometry'])
        info = _analyze(integrated['input'])
        _require(not np.array_equal(_render(info, _collections(info, original_roots=True)), integrated['output']), 'Original-footprint shortcut survived')
        _require(not np.array_equal(_render(info, _collections(info, unlimited=True)), integrated['output']), 'Unlimited collection shortcut survived')
        _require(not np.array_equal(_render(info, _collections(info, ignore_walls=True)), integrated['output']), 'Wall-free shortcut survived')
        ordering = next(pair for pair in p['train'] if pair_features(pair['input'])['order_sensitive'])
        info = _analyze(ordering['input'])
        _require(not np.array_equal(_render(info, _collections(info), column_order=True), ordering['output']), 'Column-major shortcut survived')
    _require(len(signatures) == 42 and len(normalized) >= 190, 'Structural diversity collapsed')
    _require(all(len(v) == 8 for v in orientations.values()), 'Missing directions in a split')
    _require(len({tuple(f['shape']) for f in fs}) >= 65, 'Too few canvas geometries')
    _require(len({c for f in fs for c in f['capacities']}) >= 20, 'Capacity geometry collapsed')
    _require(len({n for f in fs for n in f['loads']}) >= 18, 'Count variation collapsed')
    _require(max(f['max_distance'] for f in fs) >= 25, 'No substantial wall paths')
    _require(any(f['wall_count'] == 0 for f in fs), 'No open-field examples')
    _require(any(0 in f['loads'] for f in fs), 'No empty collectors')
    _require(sum(f['fragmented'] for f in fs) >= 42, 'Fragmented cuts missing')
    _require(sum(len({tuple(pair_features(x['input'])['roles']) for s in ('train', 'test') for x in p[s]}) > 1 for p in puzzles) >= 15, 'Local palette variation missing')
    for role in range(3):
        _require({f['roles'][role] for f in fs} == set(range(10)), 'Role color variation missing')
    p1, p2 = generate_puzzle(121212, 7, 3), generate_puzzle(121212, 7, 3)
    _require(all(np.array_equal(a[k], b[k]) for s in p1 for a, b in zip(p1[s], p2[s]) for k in a), 'Determinism failed')
    for nt, ne in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(606060 + nt, nt, ne)
        _require(len(p['train']) == nt and len(p['test']) == ne, 'Counts ignored')
    # Exercise construction when all random geometry proposals are bypassed.
    fallback_case = 0
    for shape in ((20, 20), (20, 30), (30, 20), (30, 30)):
        for turns in range(4):
            for double, obligation in ((False, 'partial'), (True, 'paths')):
                pair = _build_pair(np.random.default_rng(1000 + fallback_case), shape, double,
                                   (0, 2, 8, 5), turns, obligation, fallback_only=True)
                _require(np.array_equal(solve(pair['input']), pair['output']), 'Fallback label failed')
                fallback_case += 1
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    # A frontier tie must be rejected, independently of construction.
    supply = np.zeros((3, 3), dtype=bool)
    supply[0, :2] = True
    try:
        _select(supply, np.zeros((3, 3), dtype=int), 1)
    except ValueError:
        pass
    else:
        raise AssertionError('Ambiguous capacity tie accepted')
    damaged = generate_puzzle(919191, 2, 1)
    damaged['test'][0]['output'] = damaged['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError('Corrupt label accepted')


if __name__ == '__main__':
    self_test()
    print('Corner siphons: 42-seed self-test passed')
