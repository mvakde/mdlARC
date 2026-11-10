"""Keyed reservoirs: conserve pigment while filling variable-capacity cavities.

Only NumPy and the standard library are used. 8 is a wall, 9 an inlet;
all other colors have input-visible roles. No parent module is imported.
"""
from collections import deque

import numpy as np


STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "Invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "Grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "Invalid color")
    return a


def _components(mask, diagonal=False):
    h, w = mask.shape
    floor = mask.tolist()
    seen = [[False]*w for _ in range(h)]
    steps = STEPS + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else STEPS
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r][c]:
            continue
        seen[r][c] = True
        todo = [(r, c)]
        for rr, cc in todo:
            for dr, dc in steps:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and floor[nr][nc] and not seen[nr][nc]:
                    seen[nr][nc] = True
                    todo.append((nr, nc))
        result.append(todo)
    return result


def _dilate(mask, diagonal=True):
    a = np.pad(mask, 1)
    result = a.copy()
    for dr, dc in STEPS + (((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else ()):
        result[1 + dr:1 + dr + mask.shape[0], 1 + dc:1 + dc + mask.shape[1]] |= mask
    return result[1:-1, 1:-1]


def _background(a):
    colors, counts = np.unique(a, return_counts=True)
    _require(np.count_nonzero(counts == counts.max()) == 1, "Background must be uniquely dominant")
    bg = int(colors[np.argmax(counts)])
    _require(bg not in (8, 9), "Reserved wall/inlet cannot be background")
    return bg


def _key(a, bg):
    candidates = []
    h, w = a.shape
    for cells in _components(a != bg, diagonal=True):
        rr, cc = zip(*cells)
        r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
        sh, sw = r1 - r0 + 1, c1 - c0 + 1
        if len(cells) != sh * sw or len(set(int(a[q]) for q in cells)) != len(cells):
            continue
        if any(int(a[q]) in (8, 9) for q in cells):
            continue
        lanes = []
        if sh == 2 and 2 <= sw <= 3 and 0 < c0 and c1 < w - 1:
            if r0 == 0:
                lanes.append(("top", a[0, c0:c1+1], a[1, c0:c1+1]))
            if r1 == h - 1:
                lanes.append(("bottom", a[-1, c0:c1+1], a[-2, c0:c1+1]))
        if sw == 2 and 2 <= sh <= 3 and 0 < r0 and r1 < h - 1:
            if c0 == 0:
                lanes.append(("left", a[r0:r1+1, 0], a[r0:r1+1, 1]))
            if c1 == w - 1:
                lanes.append(("right", a[r0:r1+1, -1], a[r0:r1+1, -2]))
        for side, outer, inner in lanes:
            candidates.append((side, dict(zip(map(int, outer), map(int, inner))), cells))
    _require(len(candidates) == 1, "Expected one isolated directed edge key")
    side, mapping, cells = candidates[0]
    mask = np.zeros(a.shape, bool)
    for q in cells:
        mask[q] = True
    return side, mapping, mask


def _parse(a):
    bg = _background(a)
    side, mapping, keymask = _key(a, bg)
    keycells = set(zip(*np.nonzero(keymask)))
    pigments = set(mapping.values())
    # Pigment is traversable even when it forms a solid cluster.
    passable = (a == bg) | np.isin(a, list(pigments))
    passable[keymask] = False
    reservoirs = []
    h, w = a.shape
    for cells in _components(passable):
        if any(r in (0, h-1) or c in (0, w-1) for r, c in cells):
            continue
        boundary = set()
        for r, c in cells:
            for dr, dc in STEPS:
                q = (r+dr, c+dc)
                if not passable[q]:
                    boundary.add(q)
        gates = [q for q in boundary if a[q] == 9]
        colors = {int(a[q]) for q in boundary if a[q] != 9}
        if len(gates) != 1 or len(colors) != 1 or 8 in colors or boundary & keycells:
            continue
        rim = next(iter(colors))
        _require(rim not in pigments and rim != bg, "Rim and pigment roles must be disjoint")
        _require(all(a[q] == bg for q in cells), "Input reservoir must be empty")
        gate = gates[0]
        inside = set(cells)
        interior_neighbors = [(gate[0]+dr, gate[1]+dc) for dr, dc in STEPS
                              if (gate[0]+dr, gate[1]+dc) in inside]
        _require(len(interior_neighbors) == 1, "Inlet needs one interior neighbor")
        q = interior_neighbors[0]
        start = (2*gate[0]-q[0], 2*gate[1]-q[1])
        _require(0 <= start[0] < h and 0 <= start[1] < w and passable[start], "Inlet must face exterior floor")
        reservoirs.append(dict(cells=sorted(cells), boundary=boundary, rim=rim, gate=gate,
                               start=start, pigment=mapping.get(rim)))
    _require(reservoirs, "No marked closed reservoirs")
    _require(len({r['rim'] for r in reservoirs}) == len(reservoirs), "Reservoir rim identities must be unique")
    all_gates = set(zip(*np.where(a == 9)))
    _require(all_gates == {r['gate'] for r in reservoirs}, "Every inlet must belong to one closed reservoir")
    cavity = np.zeros(a.shape, bool)
    for reservoir in reservoirs:
        for q in reservoir['cells']:
            cavity[q] = True
    exterior = passable & ~cavity
    _require(any(r['pigment'] is not None for r in reservoirs), "No keyed reservoir")
    return dict(bg=bg, side=side, mapping=mapping, keymask=keymask,
                reservoirs=reservoirs, exterior=exterior, cavity=cavity)


def _distances(exterior, start):
    _require(exterior[start], "Inlet exterior is blocked")
    h, w = exterior.shape
    floor = exterior.tolist()
    result = [[-1]*w for _ in range(h)]
    result[start[0]][start[1]] = 1  # the port itself is distance zero
    todo = deque([start])
    while todo:
        r, c = todo.popleft()
        for dr, dc in STEPS:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and floor[nr][nc] and result[nr][nc] < 0:
                result[nr][nc] = result[r][c] + 1
                todo.append((nr, nc))
    return np.array(result, dtype=np.int16)


def _selected(a, reservoir, distance):
    pigment = reservoir['pigment']
    if pigment is None:
        return [], []
    supply = [tuple(map(int, q)) for q in np.argwhere((a == pigment) & (distance >= 0))]
    supply.sort(key=lambda q: (int(distance[q]), q))
    capacity = len(reservoir['cells'])
    if len(supply) > capacity:
        _require(distance[supply[capacity-1]] < distance[supply[capacity]], "Capacity boundary has a distance tie")
    return supply[:capacity], supply[capacity:]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read all conventions from this input; never consult generation or examples."""
    a = _grid(input_grid)
    scene = _parse(a)
    return _transform(a, scene)


def _transform(a, scene):
    out = a.copy()
    for reservoir in scene['reservoirs']:
        if reservoir['pigment'] is None:
            continue
        distance = _distances(scene['exterior'], reservoir['start'])
        selected, _ = _selected(a, reservoir, distance)
        for q in selected:
            out[q] = scene['bg']
        for q in reservoir['cells'][:len(selected)]:
            out[q] = reservoir['pigment']
    return out


def _cavity_shape(rng, kind=None):
    if kind is None:
        kind = str(rng.choice(['rectangle', 'profile', 'diamond', 'elbow']))
    h, w = int(rng.integers(2, 5)), int(rng.integers(2, 6))
    a = np.ones((h, w), bool)
    if kind == 'profile':
        left, right = 0, w-1
        for r in range(h):
            anchor = int(rng.integers(left, right+1))
            left, right = int(rng.integers(anchor+1)), int(rng.integers(anchor, w))
            a[r] = False
            a[r, left:right+1] = True
    elif kind == 'diamond':
        radius = int(rng.integers(1, 3))
        rr, cc = np.indices((2*radius+1, 2*radius+1))
        a = abs(rr-radius) + abs(cc-radius) <= radius
    elif kind == 'elbow':
        a[1:, 1:] = False
    a = np.rot90(a, int(rng.integers(4)))
    if rng.random() < .5:
        a = np.fliplr(a)
    rr, cc = np.where(a)
    return a[rr.min():rr.max()+1, cc.min():cc.max()+1].copy()


def _reservoir_stamp(rng, rim, kind=None):
    cavity = np.pad(_cavity_shape(rng, kind), 1)
    outline = _dilate(cavity) & ~cavity
    stamp = np.zeros(cavity.shape, np.int8)
    stamp[outline] = rim
    candidates = []
    h, w = cavity.shape
    for r, c in np.argwhere(outline):
        r, c = int(r), int(c)
        for dr, dc in STEPS:
            ir, ic, er, ec = r+dr, c+dc, r-dr, c-dc
            if not (0 <= ir < h and 0 <= ic < w and cavity[ir, ic]):
                continue
            if 0 <= er < h and 0 <= ec < w and (outline[er, ec] or cavity[er, ec]):
                continue
            neighbors = [(r+sr, c+sc) for sr, sc in STEPS]
            if sum(0 <= nr < h and 0 <= nc < w and cavity[nr, nc] for nr, nc in neighbors) == 1:
                candidates.append((r, c))
    _require(candidates, "Shape has no inlet site")
    gate = candidates[int(rng.integers(len(candidates)))]
    stamp[gate] = 9
    return stamp


def _put_key(a, side, sources, fills, start):
    for i, (source, fill) in enumerate(zip(sources, fills)):
        p = start+i
        if side == 'top':
            a[0, p], a[1, p] = source, fill
        elif side == 'bottom':
            a[-1, p], a[-2, p] = source, fill
        elif side == 'left':
            a[p, 0], a[p, 1] = source, fill
        else:
            a[p, -1], a[p, -2] = source, fill


def _walls(rng, shape, kind):
    h, w = shape
    wall = np.zeros(shape, bool)
    if kind == 'baffles':
        vertical = bool(rng.integers(2))
        b = wall.T if vertical else wall
        span, length = b.shape
        offset = int(rng.integers(3, 6))
        step = int(rng.integers(3, 6))
        for i, r in enumerate(range(offset, span-2, step)):
            gap = int(rng.integers(1, 5))
            b[r] = True
            if i % 2:
                b[r, :gap] = False
            else:
                b[r, -gap:] = False
    elif kind == 'partitions':
        boxes = [(0, 0, h, w)]
        for _ in range(int(rng.integers(3, 9))):
            if not boxes:
                break
            k = int(rng.integers(len(boxes)))
            r0, c0, r1, c1 = boxes.pop(k)
            if r1-r0 >= 7 and (c1-c0 < 7 or rng.random() < .5):
                cut = int(rng.integers(r0+3, r1-3))
                wall[cut, c0:c1] = True
                if rng.random() < .8:
                    gap = int(rng.integers(c0, c1))
                    wall[cut, gap:min(c1, gap+int(rng.integers(1, 4)))] = False
                boxes.extend([(r0, c0, cut, c1), (cut+1, c0, r1, c1)])
            elif c1-c0 >= 7:
                cut = int(rng.integers(c0+3, c1-3))
                wall[r0:r1, cut] = True
                if rng.random() < .8:
                    gap = int(rng.integers(r0, r1))
                    wall[gap:min(r1, gap+int(rng.integers(1, 4))), cut] = False
                boxes.extend([(r0, c0, r1, cut), (r0, cut+1, r1, c1)])
    elif kind == 'segments':
        for _ in range(int(rng.integers(3, 10))):
            r0, c0 = int(rng.integers(h-4)), int(rng.integers(w-4))
            r1, c1 = int(rng.integers(r0+3, h)), int(rng.integers(c0+3, w))
            wall[r0:r1+1, c0] = True
            wall[r0, c0:c1+1] = True
            if rng.random() < .5:
                wall[r0:r1+1, c1] = True
            if rng.random() < .25:
                wall[r1, c0:c1+1] = True
    return wall


def _empty_scene(rng, side, rich=False, lesson=''):
    h = int(rng.integers(20 if rich else 17, 31 if rich else 28))
    w = int(rng.integers(20 if rich else 17, 31 if rich else 28))
    n = int(rng.integers(2, 4))
    active = int(rng.integers(1 if rich else 2, n+1))
    inert = 1 if lesson == 'partial' else int(rng.random() < .4)
    sources, fills = list(range(1, n+1)), list(range(n+1, 2*n+1))
    order = rng.permutation(n)
    a = np.zeros((h, w), np.int8)
    axis = w if side in ('top', 'bottom') else h
    _put_key(a, side, [sources[i] for i in order], [fills[i] for i in order], int(rng.integers(1, axis-n)))
    protected = _dilate(a != 0)
    rims = [int(x) for x in rng.choice(sources, active, replace=False)] + ([7] if inert else [])
    for index, rim in enumerate(rims):
        stamp = _reservoir_stamp(rng, rim, 'elbow' if lesson == 'path' and index == 0 else None)
        sh, sw = stamp.shape
        positions = [(r, c) for r in range(1, h-sh) for c in range(1, w-sw)
                     if not np.any(protected[r:r+sh, c:c+sw])]
        if not positions:
            return None
        r, c = positions[int(rng.integers(len(positions)))]
        a[r:r+sh, c:c+sw] = stamp
        # Clear a halo so every inlet points to real exterior floor.
        protected[max(0, r-1):r+sh+1, max(0, c-1):c+sw+1] = True
    kind = str(rng.choice(['open', 'segments', 'partitions', 'baffles'], p=[.12, .23, .25, .40] if rich else [.18, .32, .25, .25]))
    wall = _walls(rng, a.shape, kind)
    wall[protected] = False
    a[wall] = 8
    return a


def _fallback_scene(side):
    # A U-shaped path detour plus a closed, independent supply pocket.
    # Used only after bounded random attempts; payload/labels are still derived
    # from the resulting input topology, never from these coordinates.
    a = np.zeros((24, 28), np.int8)
    _put_key(a, 'top', [1, 2], [3, 4], 21)
    a[3, 4:13] = 8
    a[3:21, 4] = 8
    a[3:21, 12] = 8
    a[0:4, 0] = a[0:4, 3] = 8
    a[0, 0:4] = a[3, 0:4] = 8
    a[5:9, 6:11] = 1
    a[6:8, 7:10] = 0
    a[7, 9] = 1
    a[6, 6] = 9
    a[10:15, 17:23] = 2
    a[11:14, 18:22] = 0
    a[11, 17] = 9
    a[18:22, 20:26] = 7
    a[19:21, 21:25] = 0
    a[20, 20] = 9
    return np.rot90(a, {'top': 0, 'left': 1, 'bottom': 2, 'right': 3}[side]).copy()


def _pick(rng, choices, count, style='scattered', forced=()):
    pool = sorted(set(choices) - set(forced))
    chosen = list(forced)
    _require(count <= len(pool)+len(chosen), "Not enough supply space")
    if count == len(chosen):
        return chosen
    if style == 'scattered':
        ids = rng.choice(len(pool), count-len(chosen), replace=False)
        return chosen + [pool[int(i)] for i in ids]
    anchor = chosen[0] if chosen else pool[int(rng.integers(len(pool)))]
    if style == 'line':
        horizontal = bool(rng.integers(2))
        score = lambda q: (abs(q[0]-anchor[0])*40+abs(q[1]-anchor[1])) if horizontal else (abs(q[1]-anchor[1])*40+abs(q[0]-anchor[0]))
    else:
        score = lambda q: (q[0]-anchor[0])**2 + (q[1]-anchor[1])**2
    # Shuffle before stable sorting to randomize equal geometric distances.
    pool = [pool[int(i)] for i in rng.permutation(len(pool))]
    pool.sort(key=score)
    return chosen + pool[:count-len(chosen)]


def _place_load(rng, a, scene, reservoir, mode, allowed, need_detour=False):
    distance = _distances(scene['exterior'], reservoir['start'])
    pool = [tuple(map(int, q)) for q in np.argwhere(allowed & (distance >= 0) & (a == 0))]
    cap = len(reservoir['cells'])
    style = str(rng.choice(['scattered', 'cluster', 'line']))
    if mode == 'partial':
        counts = [n for n in range(1, min(cap, 8))
                  if set(reservoir['cells'][:n]) != set(sorted(reservoir['cells'], key=lambda q: (q[1], q[0]))[:n])]
        if not counts:
            return False
        count = int(rng.choice(counts))
        if len(pool) < count:
            return False
        chosen = _pick(rng, pool, count, style)
    elif mode == 'excess':
        extra = int(rng.integers(1, 7))
        if len(pool) < cap+extra:
            return False
        chosen = None
        gate = reservoir['gate']
        manhattan = lambda q: abs(q[0]-gate[0])+abs(q[1]-gate[1])
        for k in rng.permutation(len(pool)):
            near_end = pool[int(k)]
            cutoff = int(distance[near_end])
            near = [q for q in pool if distance[q] <= cutoff]
            far = [q for q in pool if distance[q] > cutoff]
            if len(near) < cap or len(far) < extra:
                continue
            if need_detour:
                alternatives = [q for q in far if manhattan(q) < manhattan(near_end)]
                if not alternatives:
                    continue
                retained = alternatives[int(rng.integers(len(alternatives)))]
                chosen = (_pick(rng, near, cap, style, [near_end])
                          + _pick(rng, far, extra, style, [retained]))
            else:
                chosen = _pick(rng, near, cap, style) + _pick(rng, far, extra, style)
            break
        if chosen is None:
            return False
    else:
        count = int(rng.integers(0, min(cap, len(pool))+1)) if mode == 'random' else min(cap, len(pool))
        chosen = _pick(rng, pool, count, style)
    for q in chosen:
        a[q] = reservoir['pigment']
    return True


def _features(a, scene=None):
    if scene is None:
        scene = _parse(a)
    records = []
    for r in scene['reservoirs']:
        cells = r['cells']
        rr, cc = zip(*cells)
        f = dict(capacity=len(cells), rim=r['rim'], pigment=r['pigment'], gate=r['gate'],
                 irregular=len(cells) != (max(rr)-min(rr)+1)*(max(cc)-min(cc)+1),
                 shape=tuple((x-min(rr), y-min(cc)) for x, y in cells),
                 selected=0, reachable=0, excess=0, unreachable=0, reversal=False,
                 prefix_witness=False, max_distance=0)
        if r['pigment'] is not None:
            d = _distances(scene['exterior'], r['start'])
            picked, excess = _selected(a, r, d)
            f.update(selected=len(picked), reachable=len(picked)+len(excess), excess=len(excess),
                     unreachable=int(np.count_nonzero((a == r['pigment']) & ~scene['keymask'] & (d < 0))),
                     max_distance=int(d.max()))
            f['prefix_witness'] = bool(0 < len(picked) < len(cells) and
                set(cells[:len(picked)]) != set(sorted(cells, key=lambda q: (q[1], q[0]))[:len(picked)]))
            if picked and excess:
                manhattan = lambda q: abs(q[0]-r['gate'][0])+abs(q[1]-r['gate'][1])
                f['reversal'] = max(map(manhattan, picked)) > min(map(manhattan, excess))
        records.append(f)
    return dict(side=scene['side'], background=scene['bg'], shape=a.shape,
                key_length=len(scene['mapping']), records=records,
                walls=int(np.count_nonzero(a == 8)))


def _generate_pair(rng, side, lesson, rich, random_attempts=80):
    for attempt in range(random_attempts+1):
        fallback = attempt == random_attempts
        a = _fallback_scene(side) if fallback else _empty_scene(rng, side, rich, lesson)
        if a is None:
            continue
        try:
            scene = _parse(a)
        except AssertionError:
            continue
        active = [r for r in scene['reservoirs'] if r['pigment'] is not None]
        if lesson and len(active) < 2:
            continue
        allowed = scene['exterior'] & ~_dilate(scene['keymask'])
        success = True
        targets = list(range(len(active)))
        if lesson == 'path':
            targets = [i for i, r in enumerate(active) if len(r['cells']) !=
                       (max(q[0] for q in r['cells'])-min(q[0] for q in r['cells'])+1)*
                       (max(q[1] for q in r['cells'])-min(q[1] for q in r['cells'])+1)]
        if not targets:
            continue
        target = targets[int(rng.integers(len(targets)))]
        for i, r in enumerate(active):
            if i == target and lesson == 'partial':
                mode = 'partial'
            elif i == target and lesson == 'path':
                mode = 'excess'
            elif fallback:
                mode = 'full'
            else:
                mode = str(rng.choice(['partial', 'full', 'random', 'excess'], p=[.2, .2, .35, .25]))
            if not _place_load(rng, a, scene, r, mode, allowed, need_detour=(i == target and lesson == 'path')):
                success = False
                break
        if not success:
            continue
        # Unreachable supply has the SAME pigment as a real receiver. It is
        # separated by walls, never made inactive just by using another color.
        unreachable_added = False
        for r in active:
            d = _distances(scene['exterior'], r['start'])
            pool = [tuple(map(int, q)) for q in np.argwhere(allowed & (d < 0) & (a == 0))]
            if pool:
                count = int(rng.integers(1, min(5, len(pool))+1))
                for q in _pick(rng, pool, count, str(rng.choice(['cluster', 'scattered', 'line']))):
                    a[q] = r['pigment']
                unreachable_added = True
        if lesson == 'path' and not unreachable_added:
            continue
        # Key entries without a receiver may still have exterior supply.
        used = {r['pigment'] for r in active}
        unused = set(scene['mapping'].values()) - used
        for pigment in sorted(unused):
            pool = [tuple(map(int, q)) for q in np.argwhere(allowed & (a == 0))]
            for q in _pick(rng, pool, min(len(pool), int(rng.integers(1, 5)))):
                a[q] = pigment
        palette = np.array(list(map(int, rng.permutation(8))) + [8, 9], dtype=np.int8)
        a = palette[a]
        try:
            final_scene = _parse(a)
            out = _transform(a, final_scene)
            f = _features(a, final_scene)
        except AssertionError:
            continue
        records = f['records']
        if not any(r['selected'] for r in records):
            continue
        if lesson and sum(r['selected'] > 0 for r in records) < 2:
            continue
        if lesson == 'partial' and not any(r['prefix_witness'] for r in records):
            continue
        if lesson == 'path' and not (any(r['excess'] and r['reversal'] and r['irregular'] for r in records)
                                    and any(r['unreachable'] for r in records)):
            continue
        return {'input': a, 'output': out}
    raise AssertionError('Constructive fallback did not satisfy lesson')


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def _signature(a):
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    sides = [str(s) for s in rng.permutation(['top', 'left', 'bottom', 'right'])]
    lessons = ['partial', 'path'] + ['']*(num_train-2)
    lessons = [lessons[int(i)] for i in rng.permutation(num_train)]
    result = {'train': [], 'test': []}
    seen = set()
    for split, count in [('train', num_train), ('test', num_test)]:
        for i in range(count):
            # At least two key directions are demonstrated. Tests use a third
            # direction, with longer/different exterior networks and cavities.
            side = sides[i % 2] if split == 'train' else sides[2 + i % 2]
            for _ in range(12):
                pair = _generate_pair(rng, side, lessons[i] if split == 'train' else '', split == 'test')
                sig = _signature(pair['input'])
                if sig not in seen:
                    seen.add(sig)
                    result[split].append(pair)
                    break
            else:
                raise AssertionError('Unable to construct a distinct input')
    validate_generated_puzzle(result)
    return result


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'Invalid puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'Splits must be lists')
    _require(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
             and len(puzzle['train']) > len(puzzle['test']), 'Illegal counts')
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'Invalid pair schema')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape, 'Shape changed')
            sig = (a.shape, tuple(map(int, a.flat)))
            _require(sig not in seen, 'Duplicate input')
            seen.add(sig)
            scene = _parse(a)
            _require(np.array_equal(_transform(a, scene), b), 'Wrong reservoir transfer')
            _require(np.array_equal(np.bincount(a.ravel(), minlength=10), np.bincount(b.ravel(), minlength=10)),
                     'Pigment/background conservation failed')
            structural = ~scene['exterior'] & ~scene['cavity']
            _require(np.array_equal(a[structural], b[structural]), 'Key, walls, rim, or inlet changed')
            for r in scene['reservoirs']:
                if r['pigment'] is None:
                    _require(all(b[q] == scene['bg'] for q in r['cells']), 'Unlisted reservoir filled')
                    continue
                d = _distances(scene['exterior'], r['start'])
                selected, retained = _selected(a, r, d)
                n = len(selected)
                _require(all(b[q] == r['pigment'] for q in r['cells'][:n]), 'Non-prefix fill')
                _require(all(b[q] == scene['bg'] for q in r['cells'][n:]), 'Unfilled slot changed')
                _require(all(b[q] == a[q] for q in retained), 'Reachable overflow erased')
                outside = scene['exterior'] & (a == r['pigment']) & (d < 0)
                _require(np.array_equal(a[outside], b[outside]), 'Unreachable pigment erased')
            if split == 'train':
                training.append(_features(a, scene))
    records = [r for f in training for r in f['records']]
    _require(any(r['prefix_witness'] for r in records), 'Missing partial/order demonstration')
    _require(any(r['excess'] and r['reversal'] and r['irregular'] for r in records), 'Missing irregular capacity/path demonstration')
    _require(any(r['unreachable'] for r in records), 'Missing unreachable supply demonstration')
    _require(any(r['pigment'] is None for r in records), 'Missing unlisted-reservoir demonstration')
    _require(any(sum(r['selected'] > 0 for r in f['records']) >= 2 for f in training),
             'Need two differently keyed reservoirs collecting in one scene')
    _require(len({f['side'] for f in training}) >= 2, 'Need two demonstrated key directions')


def self_test():
    """Bounded structural and semantic checks over 42 reproducible seeds."""
    # Hand-specified transport: two directed matches, partial variable-size
    # cavities, and an unreachable cell of the correct pigment.
    hand = np.zeros((14, 19), np.int8)
    hand[:2, 1:3] = [[1, 2], [3, 4]]
    hand[4:8, 2:7] = 1
    hand[5:7, 3:6] = 0
    hand[5, 2] = 9
    hand[4:9, 11:16] = 2
    hand[5:8, 12:15] = 0
    hand[6, 11] = 9
    hand[5:8, 0] = 3
    hand[6, 7:11] = 4
    hand[9:13, 0] = hand[9:13, 3] = 8
    hand[9, :4] = hand[12, :4] = 8
    hand[10, 1] = 3
    expected = hand.copy()
    expected[5:8, 0] = 0
    expected[6, 7:11] = 0
    expected[5, 3:6] = 3
    expected[5, 12:15] = 4
    expected[6, 12] = 4
    _require(np.array_equal(solve(hand), expected), 'Hand-specified color/partial/reachability regression')
    # A straight supply line is split by a six-cell capacity frontier. Its
    # five excess cells and one wall-separated cell must remain in place.
    line = np.zeros((8, 22), np.int8)
    line[:2, 3:5] = [[1, 2], [3, 4]]
    line[:, 1] = 8
    line[3:7, 14:19] = 1
    line[4:6, 15:18] = 0
    line[4, 14] = 9
    line[4, 3:14] = 3
    line[4, 0] = 3
    line[7, 21] = 4
    expected = line.copy()
    expected[4, 8:14] = 0
    expected[4:6, 15:18] = 3
    _require(np.array_equal(solve(line), expected), 'Hand-specified capacity/cluster/matching regression')
    # Exercise the final construction path directly instead of hoping a
    # random seed happens to exhaust its scene attempts. Its other receivers
    # always receive positive loads, so fallback success is constructive.
    for side in ('top', 'bottom', 'left', 'right'):
        for lesson in ('', 'partial', 'path'):
            pair = _generate_pair(np.random.default_rng(612), side, lesson, False, random_attempts=0)
            _require(np.array_equal(solve(pair['input']), pair['output']), 'Fallback regression')
    puzzles, features = [], []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _features(pair['input'])
                f['split'] = split
                features.append(f)
    signatures = [tuple(_signature(p['input']) for s in ('train', 'test') for p in puzzle[s]) for puzzle in puzzles]
    _require(len(set(signatures)) == 42, 'Seed diversity collapsed')
    _require(len({f['shape'] for f in features}) >= 60, 'Canvas diversity collapsed')
    _require({f['side'] for f in features} == {'top', 'bottom', 'left', 'right'}, 'Key orientation missing')
    _require({f['background'] for f in features} == set(range(8)), 'Background palette range collapsed')
    _require({f['key_length'] for f in features} == {2, 3}, 'Key lengths collapsed')
    records = [r for f in features for r in f['records']]
    _require(len({r['shape'] for r in records}) >= 40, 'Cavity shapes collapsed')
    _require(len({r['capacity'] for r in records}) >= 10, 'Capacities collapsed')
    _require(len({len(f['records']) for f in features}) >= 3, 'Receiver counts collapsed')
    _require(any(r['pigment'] is None for r in records), 'Unlisted reservoirs absent')
    _require(sum(r['irregular'] and r['selected'] > 0 for r in records) >= 30, 'Irregular capacities absent')
    _require(any(f['walls'] == 0 for f in features), 'Simple open fields absent')
    _require(max(r['max_distance'] for r in records) >= 40, 'Long path geometries absent')
    _require(len({_signature(p['input']) for p in puzzles[0]['train']}) == 4, 'Within-puzzle inputs collapsed')
    a = generate_puzzle(173, 3, 2)
    b = generate_puzzle(173, 3, 2)
    _require(all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x), 'Nondeterminism')
    for nt, ne in [(2, 1), (10, 4), (7, 3)]:
        p = generate_puzzle(901+nt*10+ne, nt, ne)
        _require((len(p['train']), len(p['test'])) == (nt, ne), 'Requested counts ignored')
    for nt, ne in [(1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3., 1), (4, None)]:
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError('Illegal count accepted')
    # Tie regression: two cells at distance one cannot fit a one-slot cavity.
    tie = np.zeros((3, 3), np.int8)
    tie[0, 1] = tie[1, 0] = 2
    d = np.full((3, 3), -1, np.int16)
    d[0, 1] = d[1, 0] = 1
    try:
        _selected(tie, {'pigment': 2, 'cells': [(2, 2)]}, d)
    except AssertionError:
        pass
    else:
        raise AssertionError('Capacity tie accepted')
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    broken['test'][0]['output'] = broken['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('Corrupted output accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds, targeted regressions, and boundary counts)')
