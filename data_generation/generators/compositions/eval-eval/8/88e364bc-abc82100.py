"""Registered rigid stencils sliding in one shared, wall-bounded chamber.

Cards are decoded geometrically; no seed or training pair is needed by solve.
Only neutral stencil pixels are solid. In particular, a hole registration may
travel across a wall cell without painting or colliding at that virtual point.
"""
from collections import deque
import numpy as np


DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
              (0, 1), (1, -1), (1, 0), (1, 1))
CARDINAL = tuple(d for d in DIRECTIONS if 0 in d)
DIAGONAL = tuple(d for d in DIRECTIONS if 0 not in d)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(name + ' is out of range')
    return int(value)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'invalid grid dimensions')
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             'grid must contain integer colors 0..9')
    return a.astype(int, copy=False)


def _fingerprint(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _components(mask, diagonal=True):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    neighbors = DIRECTIONS if diagonal else CARDINAL
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append(cells)
    return result


def _bbox(cells):
    cells = list(cells)
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _decode(input_grid):
    """Find filled carrier cards; their 2/1 label stroke determines all roles."""
    a = _grid(input_grid)
    counts = np.bincount(a.ravel(), minlength=10)
    _require(np.count_nonzero(counts == counts.max()) == 1, 'background must be the unique mode')
    bg = int(counts.argmax())
    records, card_cells = [], set()
    for cells in _components(a != bg):
        r0, r1, c0, c1 = _bbox(cells)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if min(h, w) < 4 or len(cells) != h * w:
            continue
        sub = a[r0:r1 + 1, c0:c1 + 1]
        carrier = int(sub[0, 0])
        border = np.concatenate((sub[0], sub[-1], sub[:, 0], sub[:, -1]))
        if np.any(border != carrier) or len(np.unique(sub)) != 4:
            continue
        roles = {int(x): set(map(tuple, np.argwhere(sub == x)))
                 for x in np.unique(sub) if int(x) != carrier}
        sources = [x for x, pts in roles.items() if len(pts) == 2]
        targets = [x for x, pts in roles.items() if len(pts) == 1]
        if len(sources) != 1 or len(targets) != 1:
            continue
        source, target = sources[0], targets[0]
        neutral = next(x for x in roles if x not in (source, target))
        if len(roles[neutral]) < 3 or len(_components(sub == neutral)) != 1:
            continue
        tip = next(iter(roles[target]))
        readings = []
        for dr, dc in DIRECTIONS:
            if roles[source] != {(tip[0] - dr, tip[1] - dc),
                                 (tip[0] - 2 * dr, tip[1] - 2 * dc)}:
                continue
            q = tip[0] + dr, tip[1] + dc
            if not (0 < q[0] < h - 1 and 0 < q[1] < w - 1):
                continue
            if sub[q] not in (carrier, neutral):
                continue
            # Registration belongs to the shape's drawing, including a hole
            # inside its box or immediately next to an exact mask cell.
            mr0, mr1, mc0, mc1 = _bbox(roles[neutral])
            near = min(max(abs(q[0] - r), abs(q[1] - c)) for r, c in roles[neutral]) <= 1
            inside_box = mr0 <= q[0] <= mr1 and mc0 <= q[1] <= mc1
            if not (near or inside_box):
                continue
            readings.append((q, (dr, dc)))
        _require(len(readings) == 1, 'card must have a unique label direction and registration')
        q, direction = readings[0]
        records.append(dict(source=source, target=target, neutral=neutral,
                            carrier=carrier, direction=direction,
                            q=(q[0] + r0, q[1] + c0), bbox=(r0, r1, c0, c1),
                            offsets=tuple(sorted((r - q[0], c - q[1])
                                                 for r, c in roles[neutral])),
                            card=frozenset(cells)))
        card_cells |= cells
    _require(1 <= len(records) <= 2, 'input needs one or two readable cards')
    _require(len({r['source'] for r in records}) == len(records), 'source has two instructions')
    _require(len({r['carrier'] for r in records}) == 1, 'cards must share a wall carrier')
    _require(len({r['neutral'] for r in records}) == 1, 'cards must share a neutral role')
    wall_color, neutral = records[0]['carrier'], records[0]['neutral']
    wall = a == wall_color
    for p in card_cells:
        wall[p] = False
    _require(np.any(wall), 'missing enclosure')
    interior = _enclosed_space(wall)
    _require(np.any(interior), 'walls do not enclose a chamber')
    all_anchors = set()
    for rec in records:
        rec['anchors'] = tuple(sorted(p for p in map(tuple, np.argwhere(a == rec['source']))
                                      if p not in card_cells))
        _require(rec['anchors'], 'every card needs external anchors')
        for p in rec['anchors']:
            _require(interior[p], 'anchor must be inside the enclosure')
            _require(all(0 <= p[0] + r < a.shape[0] and 0 <= p[1] + c < a.shape[1]
                         and interior[p[0] + r, p[1] + c] for r, c in rec['offsets']),
                     'initial registered stencil does not fit')
        all_anchors.update(rec['anchors'])
    for p in map(tuple, np.argwhere(a == neutral)):
        _require(p in card_cells, 'neutral material is only drawn on cards')
    inert = {p for p in map(tuple, np.argwhere(a != bg))
             if p not in card_cells and not wall[p] and p not in all_anchors}
    _require(all(not interior[p] for p in inert), 'inert content must be outside the chamber')
    return dict(grid=a, bg=bg, wall=wall, interior=interior, cards=card_cells,
                records=records, anchors=all_anchors, inert=inert)


def _enclosed_space(wall):
    """Four-connected flood from the canvas edges separates inside from outside."""
    h, w = wall.shape
    outside = np.zeros_like(wall)
    queue = deque()
    edges = {(r, c) for r in (0, h - 1) for c in range(w)}
    edges |= {(r, c) for r in range(h) for c in (0, w - 1)}
    for p in sorted(edges):
        if not wall[p]:
            outside[p] = True
            queue.append(p)
    while queue:
        r, c = queue.popleft()
        for dr, dc in CARDINAL:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not wall[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                queue.append((nr, nc))
    return ~outside & ~wall


def _legal(wall, offsets, q):
    return all(0 <= q[0] + r < wall.shape[0] and 0 <= q[1] + c < wall.shape[1]
               and not wall[q[0] + r, q[1] + c] for r, c in offsets)


def _trace(wall, offsets, start, direction, corner=True):
    """Input-side solver: test actual translated pixels, never a bounding box."""
    _require(_legal(wall, offsets, start), 'invalid starting footprint')
    dr, dc = direction
    q, path = start, [start]
    for _ in range(31):
        nxt = q[0] + dr, q[1] + dc
        if not _legal(wall, offsets, nxt):
            return path
        if corner and dr and dc:
            if (not _legal(wall, offsets, (q[0] + dr, q[1]))
                    or not _legal(wall, offsets, (q[0], q[1] + dc))):
                return path
        q = nxt
        path.append(q)
    raise AssertionError('straight rigid motion failed to terminate')


def _predict(info, mode='rigid'):
    a, bg, wall = info['grid'], info['bg'], info['wall']
    out = a.copy()
    for p in info['anchors']:
        out[p] = bg
    paint, moves = {}, []
    for rec in info['records']:
        offsets, direction = rec['offsets'], rec['direction']
        for start in rec['anchors']:
            if mode == 'initial':
                path = [start]
            elif mode == 'point':
                path = _trace(wall, ((0, 0),), start, direction)
            else:
                path = _trace(wall, offsets, start, direction, corner=(mode != 'no_corner'))
            endpoint = path[-1]
            footprint = {(endpoint[0] + r, endpoint[1] + c) for r, c in offsets}
            if mode == 'point':
                # A charitable point-then-stamp shortcut clips at walls and edges.
                footprint = {p for p in footprint if 0 <= p[0] < a.shape[0]
                             and 0 <= p[1] < a.shape[1] and not wall[p]}
            for p in footprint:
                if mode == 'rigid':
                    _require(p not in info['cards'] and p not in info['inert'], 'paint crosses fixed content')
                    _require(p not in paint or paint[p] == rec['target'], 'conflicting final paint')
                paint[p] = rec['target']
            moves.append(dict(source=rec['source'], target=rec['target'], start=start,
                              end=endpoint, path=path, offsets=offsets,
                              direction=direction, footprint=footprint))
    for p, color in paint.items():
        out[p] = color
    return out.astype(np.int8), moves


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode this input's cards and slide their registered rigid copies."""
    return _predict(_decode(input_grid))[0]


def _fit_space(free, offsets):
    """Construction-side configuration space, computed by Boolean intersection."""
    h, w = free.shape
    fit = np.ones_like(free)
    for dr, dc in offsets:
        shifted = np.zeros_like(free)
        r0, r1 = max(0, -dr), min(h, h - dr)
        c0, c1 = max(0, -dc), min(w, w - dc)
        if r0 < r1 and c0 < c1:
            shifted[r0:r1, c0:c1] = free[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        fit &= shifted
    return fit


def _config_path(fit, start, direction, corner=True):
    def at(p):
        return 0 <= p[0] < fit.shape[0] and 0 <= p[1] < fit.shape[1] and fit[p]
    q, path = start, [start]
    dr, dc = direction
    while at((q[0] + dr, q[1] + dc)):
        if corner and dr and dc and (not at((q[0] + dr, q[1])) or not at((q[0], q[1] + dc))):
            break
        q = q[0] + dr, q[1] + dc
        path.append(q)
    return path


def _geometry(offsets):
    variants = []
    for flip in (1, -1):
        pts = [(r, flip * c) for r, c in offsets]
        for _ in range(4):
            r0, _, c0, _ = _bbox(pts)
            variants.append(tuple(sorted((r - r0, c - c0) for r, c in pts)))
            pts = [(c, -r) for r, c in pts]
    return min(variants)


def _random_shape(rng, large=False):
    h, w = int(rng.integers(2, 5 + large)), int(rng.integers(2, 5 + large))
    family = _choose(rng, ('growth', 'growth', 'branch', 'outline', 'c', 'line', 'zigzag'))
    if family == 'line':
        cells = {(0, c) for c in range(int(rng.integers(3, 6 + large)))}
    elif family == 'branch':
        rr, cc = int(rng.integers(h)), int(rng.integers(w))
        cells = {(r, cc) for r in range(h)} | {(rr, c) for c in range(w)}
        if rng.random() < .6:
            cells |= {(h - 1, c) for c in range(cc, w)}
    elif family in ('outline', 'c'):
        h, w = max(3, h), max(3, w)
        cells = {(r, c) for r in range(h) for c in range(w)
                 if r in (0, h - 1) or c in (0, w - 1)}
        if family == 'c':
            cells -= {(r, 0) for r in range(1, h - 1)}
    elif family == 'zigzag':
        cells = {(r, r % 2) for r in range(max(3, h))}
        cells |= {(0, c) for c in range(w)}
    else:
        cells = {(int(rng.integers(h)), int(rng.integers(w)))}
        desired = int(rng.integers(3, max(4, min(h * w, 13 + 3 * large))))
        for _ in range(100):
            if len(cells) >= desired:
                break
            r, c = _choose(rng, sorted(cells))
            dr, dc = _choose(rng, CARDINAL)
            if 0 <= r + dr < h and 0 <= c + dc < w:
                cells.add((r + dr, c + dc))
    # Independently turn/reflect the stencil, not the entire scene.
    for _ in range(int(rng.integers(4))):
        cells = {(c, -r) for r, c in cells}
    if rng.random() < .5:
        cells = {(r, -c) for r, c in cells}
    return cells


def _registered_shape(rng, direction, kind=None, large=False, unseen=None):
    dr, dc = direction
    for _ in range(80):
        cells = _random_shape(rng, large)
        if len(cells) < 3 or unseen is not None and _geometry(cells) in unseen:
            continue
        near = {(r + x, c + y) for r, c in cells for x in (-1, 0, 1) for y in (-1, 0, 1)}
        r0, r1, c0, c1 = _bbox(cells)
        near |= {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        choices = []
        for q in sorted(near):
            occupied = q in cells
            if kind == 'occupied' and not occupied or kind == 'hole' and occupied:
                continue
            if any((q[0] - k * dr, q[1] - k * dc) in cells for k in (1, 2, 3)):
                continue
            offsets = tuple(sorted((r - q[0], c - q[1]) for r, c in cells))
            if kind == 'occupied' and sum(r for r, c in offsets) == sum(c for r, c in offsets) == 0:
                continue
            choices.append(offsets)
        if choices:
            return _choose(rng, choices)
    raise ValueError('could not sample a registered shape')


def _make_card(rng, offsets, direction, bg, wall, neutral, source, target):
    dr, dc = direction
    points = set(offsets) | {(0, 0), (-dr, -dc), (-2 * dr, -2 * dc), (-3 * dr, -3 * dc)}
    r0, r1, c0, c1 = _bbox(points)
    pad = [int(rng.choice([1, 1, 1, 2])) for _ in range(4)]
    a = np.full((r1 - r0 + 1 + pad[0] + pad[1],
                 c1 - c0 + 1 + pad[2] + pad[3]), wall, dtype=np.int8)
    qr, qc = -r0 + pad[0], -c0 + pad[2]
    for r, c in offsets:
        a[qr + r, qc + c] = neutral
    for k in (2, 3):
        a[qr - k * dr, qc - k * dc] = source
    a[qr - dr, qc - dc] = target
    return a


def _room(rng, h, w, teaching=False, plain=False):
    footprint = np.ones((h, w), bool)
    kind = 0 if plain else int(rng.integers(4))
    if kind == 1:
        for corner in rng.choice(4, int(rng.integers(1, 4)), replace=False):
            hh, ww = int(rng.integers(1, 4)), int(rng.integers(1, 4))
            rs = slice(0, hh) if corner < 2 else slice(h - hh, h)
            cs = slice(0, ww) if corner % 2 == 0 else slice(w - ww, w)
            footprint[rs, cs] = False
    elif kind == 2:
        yy, xx = np.indices((h, w))
        cut = int(rng.integers(2, 5))
        footprint &= (yy + xx >= cut) & (yy + (w - 1 - xx) >= cut)
        footprint &= ((h - 1 - yy) + xx >= cut) & ((h - 1 - yy) + (w - 1 - xx) >= cut)
    elif kind == 3:
        left = right = 0
        for r in range(h):
            left = int(np.clip(left + rng.integers(-1, 2), 0, 2))
            right = int(np.clip(right + rng.integers(-1, 2), 0, 2))
            footprint[r, :left] = False
            if right:
                footprint[r, -right:] = False
    padded = np.pad(footprint, 1)
    inside = footprint.copy()
    for dr, dc in CARDINAL:
        inside &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    wall = footprint & ~inside
    if not teaching:
        for _ in range(int(rng.integers(0, 5))):
            r, c = int(rng.integers(3, h - 3)), int(rng.integers(3, w - 3))
            orientation = _choose(rng, CARDINAL)
            length = int(rng.integers(1, 4))
            for k in range(length):
                nr, nc = r + k * orientation[0], c + k * orientation[1]
                if 1 <= nr < h - 1 and 1 <= nc < w - 1 and inside[nr, nc]:
                    wall[nr, nc] = True
        if rng.random() < .45:
            # A border-connected baffle, in addition to freestanding islands.
            r, c = int(rng.integers(3, h - 3)), int(rng.integers(3, w - 3))
            if rng.random() < .5:
                wall[:r, c] |= footprint[:r, c]
            else:
                wall[r, :c] |= footprint[r, :c]
    return wall, footprint


def _teaching_hole(rng, h, w, direction, wall):
    """Parametric C and island: exact-mask clearance, virtual q, corner sweep.

    In the southeast basis an island at q+(-R+1,-1) is in the C's
    empty space. The next diagonal position fits but the vertical swept
    position hits the top-left stencil pixel. Starting two steps earlier
    exposes motion as well as this corner-only collision.
    """
    dr, dc = direction
    radius, width = int(rng.integers(2, 4)), int(rng.integers(2, 5))
    base = {(r, c) for r in (-radius, radius) for c in range(-1, width + 1)}
    base |= {(r, width) for r in range(-radius + 1, radius)}
    offsets = tuple(sorted((dr * r, dc * c) for r, c in base))
    q = (int(rng.integers(radius + 3, h - radius - 1)),
         int(rng.integers(5, w - width - 1)))
    # Reflect the room and q along with the basis, so all four diagonals
    # obtain the same parameterized teaching geometry.
    if dr < 0:
        q = h - 1 - q[0], q[1]
    if dc < 0:
        q = q[0], w - 1 - q[1]
    island = q[0] + dr * (-radius + 1), q[1] - dc
    wall[island] = True
    start = q[0] - 2 * dr, q[1] - 2 * dc
    return offsets, start


def _pack(rng, room_shape, cards):
    """Randomly pack a room and freely placed cards, with a guaranteed tail."""
    h, w = room_shape
    for attempt in range(24):
        shape = (30, 30) if attempt >= 8 else (int(rng.integers(max(22, h + 2), 31)),
                                                int(rng.integers(max(22, w + 2), 31)))
        occupied = np.zeros(shape, bool)
        rr, cc = int(rng.integers(shape[0] - h + 1)), int(rng.integers(shape[1] - w + 1))
        occupied[max(0, rr - 1):min(shape[0], rr + h + 1),
                 max(0, cc - 1):min(shape[1], cc + w + 1)] = True
        placements, valid = [None] * len(cards), True
        for i in rng.permutation(len(cards)):
            ch, cw = cards[i].shape
            windows = np.lib.stride_tricks.sliding_window_view(occupied, (ch, cw))
            candidates = np.argwhere(~windows.any(axis=(-1, -2)))
            if not len(candidates):
                valid = False
                break
            r, c = map(int, candidates[int(rng.integers(len(candidates)))])
            placements[i] = (r, c)
            occupied[max(0, r - 1):min(shape[0], r + ch + 1),
                     max(0, c - 1):min(shape[1], c + cw + 1)] = True
        if valid:
            return shape, (rr, cc), placements
    # One band is only a bounded fitting fallback, never a hidden convention.
    order = list(map(int, rng.permutation(len(cards))))
    total_w = sum(cards[i].shape[1] for i in order) + len(cards) - 1
    max_h = max(card.shape[0] for card in cards)
    if total_w <= 30 and h + 1 + max_h <= 30:
        placements, c = [None] * len(cards), 0
        for i in order:
            placements[i] = (h + 1, c)
            c += cards[i].shape[1] + 1
        return (30, 30), (0, 0), placements
    total_h = sum(cards[i].shape[0] for i in order) + len(cards) - 1
    max_w = max(card.shape[1] for card in cards)
    if total_h <= 30 and w + 1 + max_w <= 30:
        placements, r = [None] * len(cards), 0
        for i in order:
            placements[i] = (r, w + 1)
            r += cards[i].shape[0] + 1
        return (30, 30), (0, 0), placements
    raise ValueError('room and cards cannot fit')


def _make_pair(rng, palette, plan, unseen):
    bg, wall_color, neutral = palette[:3]
    teaching = plan['teaching']
    n = plan['count']
    mappings = plan['mapping']
    for attempt in range(100):
        try:
            minimum = 14 if teaching == 'hole' else 11
            maximum = 19 if n == 2 else 21
            h, w = int(rng.integers(minimum, maximum)), int(rng.integers(minimum, maximum))
            wall, footprint = _room(rng, h, w, teaching == 'hole', plain=attempt > 65)
            specs, forced = [], None
            for i in range(n):
                direction = plan['directions'][i]
                if teaching == 'hole' and i == 0:
                    offsets, forced = _teaching_hole(rng, h, w, direction, wall)
                else:
                    kind = 'occupied' if teaching == 'occupied' else _choose(rng, ('occupied', 'hole', None))
                    offsets = _registered_shape(rng, direction, kind, plan['test'], unseen if plan['test'] else None)
                specs.append(dict(offsets=offsets, direction=direction,
                                  source=mappings[i][0], target=mappings[i][1]))
            free = _enclosed_space(wall)
            selected, paint, starts = [], {}, set()
            for i, rec in enumerate(specs):
                fit = _fit_space(free, rec['offsets'])
                candidates = []
                for p in map(tuple, np.argwhere(fit & free)):
                    path = _config_path(fit, p, rec['direction'])
                    if len(path) <= 1:
                        continue
                    if teaching == 'occupied' and i == 0:
                        point_path = _config_path(free, p, rec['direction'])
                        if path[-1] == point_path[-1]:
                            continue
                    candidates.append((p, path))
                if forced is not None and i == 0:
                    candidates = [(p, path) for p, path in candidates if p == forced]
                    if not candidates:
                        raise ValueError('diagnostic start does not fit')
                    path = candidates[0][1]
                    if len(path) != 3 or len(_config_path(fit, forced, rec['direction'], False)) <= 3:
                        raise ValueError('missing corner witness')
                else:
                    rng.shuffle(candidates)
                desired = int(_choose(rng, (1, 1, 2, 2, 3, 4)))
                anchors, paths = [], []
                for p, path in candidates:
                    if p in starts:
                        continue
                    final = {(path[-1][0] + r, path[-1][1] + c) for r, c in rec['offsets']}
                    if any(q in paint and paint[q] != rec['target'] for q in final):
                        continue
                    anchors.append(p)
                    paths.append(path)
                    starts.add(p)
                    paint.update({q: rec['target'] for q in final})
                    if len(anchors) >= desired:
                        break
                if not anchors:
                    raise ValueError('no compatible moving copies')
                if forced is not None and i == 0 and anchors[0] != forced:
                    raise ValueError('diagnostic anchor lost')
                # A stationary additional launch is allowed, but not needed
                # to establish the rule. All cards already have a moving copy.
                if teaching is None and rng.random() < .12:
                    for p in map(tuple, np.argwhere(fit & free)):
                        if p in starts or len(_config_path(fit, p, rec['direction'])) > 1:
                            continue
                        final = {(p[0] + r, p[1] + c) for r, c in rec['offsets']}
                        if any(q in paint and paint[q] != rec['target'] for q in final):
                            continue
                        anchors.append(p)
                        paths.append([p])
                        starts.add(p)
                        paint.update({q: rec['target'] for q in final})
                        break
                selected.append((rec, anchors, paths))
            cards = [_make_card(rng, rec['offsets'], rec['direction'], bg, wall_color,
                                neutral, rec['source'], rec['target']) for rec in specs]
            shape, (rr, cc), placements = _pack(rng, (h, w), cards)
            a = np.full(shape, bg, dtype=np.int8)
            a[rr:rr + h, cc:cc + w][wall] = wall_color
            fixed = np.zeros(shape, bool)
            fixed[rr:rr + h, cc:cc + w] = True
            for card, (r, c) in zip(cards, placements):
                a[r:r + card.shape[0], c:c + card.shape[1]] = card
                fixed[max(0, r - 1):min(shape[0], r + card.shape[0] + 1),
                      max(0, c - 1):min(shape[1], c + card.shape[1] + 1)] = True
            for rec, anchors, paths in selected:
                for r, c in anchors:
                    a[rr + r, cc + c] = rec['source']
            b = a.copy()
            for r, c in starts:
                b[rr + r, cc + c] = bg
            for (r, c), color in paint.items():
                b[rr + r, cc + c] = color
            # Optional unrelated content outside the shared chamber, with a
            # background gap from cards so it cannot corrupt their topology.
            active = {x for m in mappings for x in m}
            inert_colors = [x for x in palette[3:] if x not in active]
            for _ in range(int(rng.integers(0, 5))):
                cells = np.argwhere(~fixed & (a == bg))
                if not len(cells) or not inert_colors:
                    break
                r, c = map(int, cells[int(rng.integers(len(cells)))])
                color = int(_choose(rng, inert_colors))
                a[r, c] = b[r, c] = color
            if rng.random() < .8:
                r0, r1, c0, c1 = _bbox(map(tuple, np.argwhere(a != bg)))
                r0, c0 = max(0, r0 - int(rng.integers(3))), max(0, c0 - int(rng.integers(3)))
                r1, c1 = min(shape[0] - 1, r1 + int(rng.integers(3))), min(shape[1] - 1, c1 + int(rng.integers(3)))
                a, b = a[r0:r1 + 1, c0:c1 + 1].copy(), b[r0:r1 + 1, c0:c1 + 1].copy()
            info = _decode(a)
            predicted, _ = _predict(info)
            if info['bg'] != bg or not np.array_equal(predicted, b):
                continue
            if teaching == 'occupied' and np.array_equal(_predict(info, 'point')[0], b):
                continue
            if teaching == 'hole' and np.array_equal(_predict(info, 'no_corner')[0], b):
                continue
            return {'input': a, 'output': b}
        except (ValueError, AssertionError):
            continue
    raise AssertionError('bounded pair construction exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    nt = _integer(num_train, 2, 10, 'num_train')
    nq = _integer(num_test, 1, 4, 'num_test')
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    pool = palette[3:]
    first_dir, diag_dir = _choose(rng, CARDINAL), _choose(rng, DIAGONAL)
    plans = [dict(teaching='occupied', count=1, directions=[first_dir],
                  mapping=[(pool[0], pool[1])], test=False),
             dict(teaching='hole', count=2, directions=[diag_dir, _choose(rng, CARDINAL)],
                  mapping=[(pool[0], pool[2]), (pool[2], pool[0])], test=False)]
    for i in range(2, nt + nq):
        test = i >= nt
        count = int(_choose(rng, (1, 2, 2) if test else (1, 1, 2)))
        shuffled = list(map(int, rng.permutation(pool)))
        mode = _choose(rng, ('independent', 'shared', 'chain', 'reciprocal'))
        if count == 1:
            mapping = [(shuffled[0], shuffled[1])]
        elif mode == 'shared':
            mapping = [(shuffled[0], shuffled[2]), (shuffled[1], shuffled[2])]
        elif mode == 'reciprocal':
            mapping = [(shuffled[0], shuffled[1]), (shuffled[1], shuffled[0])]
        elif mode == 'chain':
            mapping = [(shuffled[0], shuffled[1]), (shuffled[1], shuffled[2])]
        else:
            mapping = [(shuffled[0], shuffled[2]), (shuffled[1], shuffled[3])]
        dirs = [_choose(rng, DIRECTIONS) for _ in range(count)]
        if test:
            # A novel orientation is a new combination of demonstrated
            # cardinal/diagonal geometry, rather than an extra motion rule.
            dirs[0] = _choose(rng, [d for d in DIRECTIONS if d not in (first_dir, diag_dir)])
        plans.append(dict(teaching=None, count=count, directions=dirs, mapping=mapping, test=test))
    puzzle, seen, training_geometry = {'train': [], 'test': []}, set(), set()
    # Teaching roles are fixed, but their order is not a clue to the rule.
    train_order = list(map(int, rng.permutation(nt)))
    for i in train_order + list(range(nt, nt + nq)):
        plan = plans[i]
        for _ in range(8):
            pair = _make_pair(rng, palette, plan, training_geometry)
            key = _fingerprint(pair['input'])
            if key not in seen:
                break
        _require(key not in seen, 'duplicate input')
        seen.add(key)
        split = 'test' if plan['test'] else 'train'
        puzzle[split].append(pair)
        if split == 'train':
            training_geometry |= {_geometry(r['offsets']) for r in _decode(pair['input'])['records']}
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info, moves):
    corner = 0
    negative = virtual = overlap = 0
    visited = set()
    for move in moves:
        offsets, q = move['offsets'], move['end']
        dr, dc = move['direction']
        if dr and dc and _legal(info['wall'], offsets, (q[0] + dr, q[1] + dc)):
            corner += 1
        r0, r1, c0, c1 = _bbox(offsets)
        for pos in move['path']:
            negative += any(info['wall'][pos[0] + r, pos[1] + c]
                            for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))
            virtual += bool(info['wall'][pos])
        overlap += len(move['footprint'] & visited)
        visited |= move['footprint']
    wall_pts = list(map(tuple, np.argwhere(info['wall'])))
    r0, r1, c0, c1 = _bbox(wall_pts)
    sub = info['wall'][r0:r1 + 1, c0:c1 + 1]
    irregular = not (sub[0].all() and sub[-1].all() and sub[:, 0].all() and sub[:, -1].all())
    return dict(shape=info['grid'].shape, cards=len(info['records']),
                directions=tuple(r['direction'] for r in info['records']),
                stencil_sizes=tuple(len(r['offsets']) for r in info['records']),
                holes=sum((0, 0) not in r['offsets'] for r in info['records']),
                distances=tuple(len(m['path']) - 1 for m in moves),
                anchors=len(moves), corner=corner, negative=negative,
                virtual=virtual, overlap=overlap, irregular=irregular,
                geometry=tuple(_geometry(r['offsets']) for r in info['records']))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'illegal pair counts')
    seen, trained, training = set(), set(), []
    roles = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape, 'dimensions changed')
            key = _fingerprint(a)
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            info = _decode(a)
            expected, moves = _predict(info)
            _require(np.array_equal(b, expected), 'output violates rigid registered motion')
            _require(not np.array_equal(a, b), 'unchanged example')
            roles.add((info['bg'], info['records'][0]['carrier'], info['records'][0]['neutral']))
            fixed = info['cards'] | info['inert'] | set(map(tuple, np.argwhere(info['wall'])))
            _require(all(a[p] == b[p] for p in fixed), 'fixed content changed')
            _require(all(any(len(m['path']) > 1 for m in moves if m['source'] == r['source'])
                         for r in info['records']), 'each card must launch a moving copy')
            for move in moves:
                _require(_legal(info['wall'], move['offsets'], move['end']), 'illegal endpoint')
                _require(all(info['interior'][p] for p in move['footprint']), 'copy escaped its enclosure')
            f = _features(info, moves)
            if split == 'train':
                training.append((info, moves, f, b))
                trained |= set(f['geometry'])
            else:
                _require(all(g not in trained for g in f['geometry']), 'test must contain new shape geometry')
    _require(len(roles) == 1, 'background/wall/stencil conventions changed within puzzle')
    dirs = {d for info, moves, f, b in training for d in f['directions']}
    _require(any(d in CARDINAL for d in dirs) and any(d in DIAGONAL for d in dirs),
             'missing cardinal/diagonal teaching')
    _require(any(f['corner'] for info, moves, f, b in training), 'missing corner-only stop evidence')
    _require(any(f['negative'] for info, moves, f, b in training), 'missing exact-mask negative space evidence')
    _require(any(f['holes'] for info, moves, f, b in training), 'missing hole registration evidence')
    _require(any((0, 0) in r['offsets'] and (sum(x for x, y in r['offsets']) != 0
                 or sum(y for x, y in r['offsets']) != 0)
                 for info, moves, f, b in training for r in info['records']),
             'missing off-center occupied registration')
    _require(any(not np.array_equal(_predict(info, 'point')[0], b)
                 for info, moves, f, b in training), 'point-then-stamp shortcut not distinguished')
    _require(any(not np.array_equal(_predict(info, 'initial')[0], b)
                 for info, moves, f, b in training), 'motion omission not distinguished')
    _require(any(not np.array_equal(_predict(info, 'no_corner')[0], b)
                 for info, moves, f, b in training), 'diagonal shortcut not distinguished')
    targets = {}
    chain = False
    for info, moves, f, b in training:
        local_sources = {r['source'] for r in info['records']}
        for r in info['records']:
            targets.setdefault(r['source'], set()).add(r['target'])
            chain |= r['target'] in local_sources
    _require(any(len(ts) > 1 for ts in targets.values()), 'missing local mapping change')
    _require(chain, 'missing nonrecursive color-role chaining evidence')


def self_test():
    """Bounded 42-seed audit plus count, determinism, and corruption checks."""
    # A hand-specified regression has an independently stated endpoint:
    # anchor (5,5) -> virtual q (7,7), crossing island (6,6). The next
    # diagonal footprint fits, but a swept axial footprint hits that island.
    manual = np.zeros((26, 24), dtype=np.int8)
    manual[1, 1:15] = manual[14, 1:15] = 6
    manual[1:15, 1] = manual[1:15, 14] = 6
    manual[6, 6], manual[5, 5] = 6, 1
    offsets = tuple(sorted({(r, c) for r in (-2, 2) for c in range(-1, 3)}
                           | {(r, 2) for r in range(-1, 2)}))
    card = _make_card(np.random.default_rng(8), offsets, (1, 1), 0, 6, 8, 1, 2)
    manual[17:17 + card.shape[0], 2:2 + card.shape[1]] = card
    expected = manual.copy()
    expected[5, 5] = 0
    for r, c in offsets:
        expected[7 + r, 7 + c] = 2
    _require(np.array_equal(solve(manual), expected), 'manual rigid/corner/hole regression')
    _require(expected[6, 6] == 6, 'virtual registration must not paint its island')
    for turns in range(4):
        _require(np.array_equal(solve(np.rot90(manual, turns)), np.rot90(expected, turns)),
                 'rotation equivariance')
    recolor = np.array([9, 4, 7, 2, 0, 3, 5, 8, 1, 6])
    _require(np.array_equal(solve(recolor[manual]), recolor[expected]), 'semantic palette equivariance')
    puzzles, features, shapes, geometries, fingerprints = [], [], set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        fingerprints.add(tuple(_fingerprint(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        local_sizes, local_geometries = set(), set()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                out, moves = _predict(info)
                _require(np.array_equal(out, pair['output']), 'solve mismatch')
                f = _features(info, moves)
                f['split'] = split
                features.append(f)
                shapes.add(f['shape'])
                geometries |= set(f['geometry'])
                local_sizes.add(f['shape'])
                local_geometries |= set(f['geometry'])
        _require(len(local_sizes) >= 2 and len(local_geometries) >= 3, 'within-puzzle structural variation')
    _require(len(fingerprints) == 42, 'puzzle seed diversity')
    _require(len(shapes) >= 35 and len(geometries) >= 45, 'geometry diversity')
    for split in ('train', 'test'):
        fs = [f for f in features if f['split'] == split]
        _require({d for f in fs for d in f['directions']} == set(DIRECTIONS), 'orientation coverage')
        _require({f['cards'] for f in fs} == {1, 2}, 'card-count coverage')
        _require(any(f['holes'] for f in fs), 'hole registration coverage')
        _require(any(f['irregular'] for f in fs), 'irregular boundaries coverage')
        _require(any(f['overlap'] for f in fs), 'compatible merged copies coverage')
    _require(any(f['virtual'] for f in features), 'virtual registration passing a wall')
    _require(any(0 in f['distances'] for f in features), 'stationary launch support')
    _require(any(1 in f['distances'] for f in features), 'one-step launches')
    _require(any(30 in f['shape'] for f in features), 'full-size canvas')
    a = generate_puzzle(93, 7, 3)
    b = generate_puzzle(93, 7, 3)
    _require(all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s])
                 for k in ('input', 'output')), 'nondeterminism')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(9000 + 10 * nt + nq, nt, nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, 'count contract')
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(9, nt, nq)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    bad['test'][0]['output'] = bad['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    return {'seeds': 42, 'pairs': len(features), 'canvas_shapes': len(shapes),
            'stencil_geometries': len(geometries),
            'corner_witness_pairs': sum(f['corner'] > 0 for f in features),
            'negative_space_pairs': sum(f['negative'] > 0 for f in features),
            'virtual_registration_pairs': sum(f['virtual'] > 0 for f in features),
            'overlap_pairs': sum(f['overlap'] > 0 for f in features)}
