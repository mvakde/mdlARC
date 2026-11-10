"""Stencil broadcast chambers. Only NumPy and the Python standard library."""
from collections import deque
import hashlib
import numpy as np

D4 = ((-1, 0), (0, -1), (0, 1), (1, 0))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _counts(a, b):
    if not (_integer(a) and _integer(b) and 2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("Require integer counts 2..10 / 1..4 with train > test")
    return int(a), int(b)


def _grid(a):
    a = np.asarray(a)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "noninteger cells")
    _need(bool(np.all((a >= 0) & (a <= 9))), "bad colors")
    return a.astype(np.int8, copy=False)


def _components(cells, steps=D4):
    left = set(cells)
    groups = []
    while left:
        p = min(left)
        left.remove(p)
        group, todo = {p}, [p]
        for r, c in todo:
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in left:
                    left.remove(q)
                    group.add(q)
                    todo.append(q)
        groups.append(group)
    return groups


def _near(cells, steps=D4):
    return {(r + dr, c + dc) for r, c in cells for dr, dc in steps} - set(cells)


def _box(cells):
    cells = tuple(cells)
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells) + 1, max(c for r, c in cells) + 1)


def _rect(box):
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1) for c in range(c0, c1)}


def _shell(box):
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1) for c in range(c0, c1)
            if r in (r0, r1 - 1) or c in (c0, c1 - 1)}


def _crop(mask):
    r0, c0, r1, c1 = _box(zip(*np.where(mask)))
    return mask[r0:r1, c0:c1].copy()


def _analyze(input_grid):
    """Recover rectangles, payloads, token components and physical wire ownership."""
    a = _grid(input_grid)
    frames = []
    for color in range(1, 10):
        for comp in _components(zip(*np.where(a == color))):
            box = _box(comp)
            r0, c0, r1, c1 = box
            if min(r1-r0, c1-c0) < 4 or comp != _shell(box):
                continue
            inside = a[r0+1:r1-1, c0+1:c1-1]
            colors = set(map(int, np.unique(inside)))
            frame = dict(box=box, shell=comp, color=color, shape=inside.shape)
            rim = np.concatenate((inside[0], inside[-1], inside[:, 0], inside[:, -1]))
            if colors <= {0, color} and colors == {0, color} and np.all(rim == 0):
                tokens = _components(zip(*np.where(inside == color)))
                _need(len(tokens) == len(_components(zip(*np.where(inside == color)), D8)),
                      "tokens touch diagonally")
                _need(max(map(len, tokens)) <= 6, "token exceeds six cells")
                frame.update(kind="target", count=len(tokens), area=sum(map(len, tokens)),
                             sizes=tuple(sorted(map(len, tokens))))
            elif 0 not in colors and color not in colors and len(colors) == 2 and np.all(rim == rim[0]):
                secondary = int(rim[0])
                primary = next(iter(colors - {secondary}))
                mask = inside == primary
                _need(len(_components(zip(*np.where(mask)))) == 1, "disconnected source ink")
                frame.update(kind="source", tile=_crop(mask), primary=primary, secondary=secondary)
            else:
                continue
            frames.append(frame)
    _need(frames and len({f['color'] for f in frames}) == 1, "missing or mixed frame convention")
    sources = [i for i, f in enumerate(frames) if f['kind'] == 'source']
    targets = [i for i, f in enumerate(frames) if f['kind'] == 'target']
    _need(sources and targets, "need sources and targets")
    occupied = set()
    shell_owner = {}
    for i, f in enumerate(frames):
        cells = _rect(f['box'])
        _need(not occupied & cells, "overlapping frames")
        occupied |= cells
        shell_owner.update({p: i for p in f['shell']})
    wire_cells = set(zip(*np.where(a != 0))) - occupied
    routes = []
    source_routes = {i: [] for i in sources}
    target_routes = {i: [] for i in targets}
    for color in sorted({int(a[p]) for p in wire_cells}):
        for cells in _components(p for p in wire_cells if a[p] == color):
            contacts = {}
            for r, c in cells:
                for dr, dc in D4:
                    q = (r + dr, c + dc)
                    if q not in shell_owner:
                        continue
                    i = shell_owner[q]
                    r0, c0, r1, c1 = frames[i]['box']
                    side = 'top' if r < r0 else 'bottom' if r >= r1 else 'left' if c < c0 else 'right'
                    contacts.setdefault(i, set()).add(side)
            src = set(contacts) & set(sources)
            dst = set(contacts) & set(targets)
            _need(len(src) == 1 and dst, "wire must link one source to targets")
            owner = next(iter(src))
            route_index = len(routes)
            source_routes[owner].append(route_index)
            for i in dst:
                _need(len(contacts[i]) == 1, "target has multiple entry sides")
                side = next(iter(contacts[i]))
                target_routes[i].append((owner, 1 if side in ('left', 'right') else 0, side))
            routes.append(dict(source=owner, targets=dst, cells=cells, color=color))
    _need(all(len(v) == 1 for v in source_routes.values()), "source must have one route")
    _need(all(len(v) <= 1 for v in target_routes.values()), "conflicting target supplies")
    jobs = []
    for i in targets:
        if not target_routes[i]:
            continue
        owner, axis, side = target_routes[i][0]
        t, s = frames[i], frames[owner]
        h, w = s['tile'].shape
        ph, pw = (h, t['count']*(w+1)-1) if axis == 1 else (t['count']*(h+1)-1, w)
        _need(ph <= t['shape'][0] and pw <= t['shape'][1], "printing exceeds chamber")
        jobs.append(dict(target=i, source=owner, axis=axis, side=side, print_shape=(ph, pw)))
    _need(jobs, "no active target")
    return dict(frames=frames, sources=sources, targets=targets, routes=routes, jobs=jobs)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solution; no generation parameters or examples are consulted."""
    a = _grid(input_grid)
    info = _analyze(a)
    out = a.copy()
    for job in info['jobs']:
        f, s = info['frames'][job['target']], info['frames'][job['source']]
        r0, c0, r1, c1 = f['box']
        region = out[r0+1:r1-1, c0+1:c1-1]
        region[:] = 0
        ph, pw = job['print_shape']
        top, left = (region.shape[0]-ph)//2, (region.shape[1]-pw)//2
        h, w = s['tile'].shape
        # Modular indexing is separate from the generator's concatenation.
        for r in range(ph):
            for c in range(pw):
                tr, tc = (r, c % (w+1)) if job['axis'] == 1 else (r % (h+1), c)
                ink = tr < h and tc < w and s['tile'][tr, tc]
                region[top+r, left+c] = s['primary'] if ink else s['secondary']
    return out


def _ri(rng, lo, hi):
    return int(rng.integers(lo, hi+1))


def _pick(rng, values):
    return values[_ri(rng, 0, len(values)-1)]


def _mask(rng, h, w):
    """Connected random growth, cropped afterwards; no forced corner path."""
    if h >= 3 and w >= 3 and rng.random() < .18:
        m = np.ones((h, w), dtype=bool)
        m[1:-1, 1:-1] = False
        return m
    cells = {(_ri(rng, 0, h-1), _ri(rng, 0, w-1))}
    amount = _ri(rng, max(1, (h*w)//3), h*w)
    while len(cells) < amount:
        frontier = sorted(p for p in _near(cells) if 0 <= p[0] < h and 0 <= p[1] < w)
        cells.add(_pick(rng, frontier))
    m = np.zeros((h, w), dtype=bool)
    for p in cells:
        m[p] = True
    return _crop(m)


def _tokens(rng, h, w, n, area=None):
    """Small irregular objects with a blank rim and no diagonal contacts."""
    canvas = {(r, c) for r in range(1, h-1) for c in range(1, w-1)}
    for attempt in range(50):
        objects, occupied = [], set()
        sizes = [1] * n
        total = area if area is not None else _ri(rng, n, min(6*n, max(n, len(canvas)//3)))
        for _ in range(total-n):
            choices = [i for i in range(n) if sizes[i] < 6]
            sizes[_pick(rng, choices)] += 1
        good = True
        for size in sorted(sizes, reverse=True):
            allowed = canvas - occupied - _near(occupied, D8)
            if not allowed:
                good = False
                break
            cells = {_pick(rng, sorted(allowed))}
            for _ in range(size-1):
                choices = sorted(_near(cells) & allowed)
                if not choices:
                    good = False
                    break
                cells.add(_pick(rng, choices))
            if not good:
                break
            occupied |= cells
            objects.append(cells)
        if good:
            result = np.zeros((h, w), dtype=bool)
            for p in occupied:
                result[p] = True
            return result
    return None


def _source(mask, primary, secondary, frame_color, rng, padding=None):
    margins = tuple(_ri(rng, 1, 2) for _ in range(4)) if padding is None else padding
    top, bottom, left, right = margins
    h, w = mask.shape
    inside = np.full((h+top+bottom, w+left+right), secondary, dtype=np.int8)
    inside[top:top+h, left:left+w][mask] = primary
    array = np.pad(inside, 1, constant_values=frame_color)
    return dict(kind='source', array=array, tile=mask, primary=primary, secondary=secondary)


def _target(h, w, n, owner, axis, frame_color, rng, area=None):
    tokens = _tokens(rng, h, w, n, area)
    if tokens is None:
        return None
    inside = np.where(tokens, frame_color, 0).astype(np.int8)
    return dict(kind='target', array=np.pad(inside, 1, constant_values=frame_color),
                count=n, owner=owner, axis=axis, area=int(tokens.sum()))


def _tiles(rng, lesson):
    if lesson == 0:
        return [np.rot90(np.array([[1, 1], [1, 0]], dtype=bool), _ri(rng, 0, 3))]
    if lesson == 1:
        return [np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
                np.array([[1, 0], [1, 0], [1, 1]], dtype=bool)]
    return [_mask(rng, _ri(rng, 1, 4), _ri(rng, 1, 4))
            for _ in range(_pick(rng, (1, 1, 2)))]


def _specs(rng, lesson, test, fixed=False):
    colors = list(map(int, rng.permutation(np.arange(1, 10))))
    frame_color, wire_colors = colors[0], colors[1:3]
    tiles = _tiles(rng, lesson)
    if fixed and lesson < 0:
        tiles = [np.array([[1, 1], [1, 0]], dtype=bool)]
    specs = []
    for i, m in enumerate(tiles):
        p, s = map(int, rng.choice(colors[3:], 2, replace=False))
        specs.append(_source(m, p, s, frame_color, rng, (1, 1, 1, 1) if fixed else None))
    if lesson == 0:
        axis = 1 if fixed else _ri(rng, 0, 1)
        jobs = [(0, 2, axis, 9, 9, 6), (0, 3, axis, 9, 9, 6),
                (None, 2, 1, 5, 6, None)]
    elif lesson == 1:
        jobs = [(0, 2, 0, 6, 5, None), (1, 2, 1, 5, 6, None),
                (None, 1, 1, 5, 6, None)]
    else:
        jobs = []
        for owner, tile in enumerate(tiles):
            for j in range(1 if fixed else _pick(rng, (1, 1, 2))):
                n = _ri(rng, 4, 5) if test and owner == 0 and j == 0 else _ri(rng, 1, 3)
                axis = 1 if fixed else _ri(rng, 0, 1)
                th, tw = tile.shape
                ph, pw = (th, n*(tw+1)-1) if axis == 1 else (n*(th+1)-1, tw)
                if max(ph, pw) > 19:
                    return None
                h, w = max(5, ph+_ri(rng, 0, 3)), max(5, pw+_ri(rng, 0, 3))
                if n >= 4:
                    h, w = max(h, 7), max(w, 7)
                jobs.append((owner, n, axis, h, w, None))
        if fixed:
            owner, n, axis, h, w, area = jobs[0]
            jobs[0] = (owner, n, axis, 7, max(7, 3*n+1), area)
        if fixed or rng.random() < .65:
            jobs.append((None, _ri(rng, 1, 3), 1, 5, 6, None))
    for owner, n, axis, h, w, area in jobs:
        obj = _target(h, w, n, owner, axis, frame_color, rng, area)
        if obj is None:
            return None
        specs.append(obj)
    return specs, wire_colors


def _place(rng, specs, shape):
    h, w = shape
    order = sorted(range(len(specs)), key=lambda i: -specs[i]['array'].size)
    occupied, placed = set(), {}
    for i in order:
        sh, sw = specs[i]['array'].shape
        if sh > h or sw > w:
            return None
        positions = [(r, c) for r in range(h-sh+1) for c in range(w-sw+1)]
        rng.shuffle(positions)
        blocked = occupied | _near(occupied, D8)
        for r, c in positions[:300]:
            box = (r, c, r+sh, c+sw)
            cells = _rect(box)
            if not cells & blocked:
                placed[i] = box
                occupied |= cells
                break
        else:
            return None
    return [placed[i] for i in range(len(specs))]


def _ports(box, shape, axis=None):
    r0, c0, r1, c1 = box
    h, w = shape
    p = []
    if axis in (None, 0):
        p += [(r0-1, c) for c in range(c0+1, c1-1) if r0 > 0]
        p += [(r1, c) for c in range(c0+1, c1-1) if r1 < h]
    if axis in (None, 1):
        p += [(r, c0-1) for r in range(r0+1, r1-1) if c0 > 0]
        p += [(r, c1) for r in range(r0+1, r1-1) if c1 < w]
    return p


def _path(rng, starts, goal, allowed):
    starts = sorted(starts)
    rng.shuffle(starts)
    queue = deque(starts)
    previous = {p: None for p in starts}
    steps = [D4[i] for i in rng.permutation(4)]
    while queue:
        p = queue.popleft()
        if p == goal:
            out = set()
            while p is not None:
                out.add(p)
                p = previous[p]
            return out
        r, c = p
        for dr, dc in steps:
            q = r+dr, c+dc
            if q in allowed and q not in previous:
                previous[q] = p
                queue.append(q)
    return None


def _route(rng, specs, boxes, shape):
    canvas = _rect((0, 0, *shape))
    frame_cells = set().union(*(_rect(b) for b in boxes))
    forbidden = frame_cells | _near(frame_cells)
    sources = [i for i, s in enumerate(specs) if s['kind'] == 'source']
    chosen = {}
    for i, s in enumerate(specs):
        if s['kind'] == 'source' or s['owner'] is not None:
            options = _ports(boxes[i], shape, None if s['kind'] == 'source' else s['axis'])
            # A port may touch only its own rectangle.
            other = frame_cells - _rect(boxes[i])
            options = [p for p in options if p not in frame_cells and not _near({p}) & other]
            if not options:
                return None
            chosen[i] = _pick(rng, options)
    routes, used = {}, set()
    for owner in rng.permutation(len(sources)):
        owner = int(owner)
        root = sources[owner]
        destinations = [i for i, s in enumerate(specs) if s['kind'] == 'target' and s['owner'] == owner]
        rng.shuffle(destinations)
        tree = {chosen[root]}
        if tree & (used | _near(used)):
            return None
        for i in destinations:
            allowed = (canvas - forbidden - used - _near(used)) | tree | {chosen[i]}
            if chosen[i] in used | _near(used):
                return None
            path = _path(rng, tree, chosen[i], allowed)
            if path is None:
                return None
            tree |= path
        routes[owner] = tree
        used |= tree
    return routes


def _render(specs, boxes, shape, routes, wire_colors):
    a = np.zeros(shape, dtype=np.int8)
    for s, (r0, c0, r1, c1) in zip(specs, boxes):
        a[r0:r1, c0:c1] = s['array']
    for owner, cells in routes.items():
        for p in cells:
            a[p] = wire_colors[owner]
    b = a.copy()
    sources = [s for s in specs if s['kind'] == 'source']
    for s, box in zip(specs, boxes):
        if s['kind'] != 'target' or s['owner'] is None:
            continue
        source = sources[s['owner']]
        tile = np.where(source['tile'], source['primary'], source['secondary']).astype(np.int8)
        axis = s['axis']
        sep_shape = (tile.shape[0], 1) if axis == 1 else (1, tile.shape[1])
        sep = np.full(sep_shape, source['secondary'], dtype=np.int8)
        parts = []
        for i in range(s['count']):
            if i:
                parts.append(sep)
            parts.append(tile)
        printed = np.concatenate(parts, axis=axis)
        r0, c0, r1, c1 = box
        region = b[r0+1:r1-1, c0+1:c1-1]
        region[:] = 0
        dr, dc = (region.shape[0]-printed.shape[0])//2, (region.shape[1]-printed.shape[1])//2
        if dr < 0 or dc < 0:
            return None
        region[dr:dr+printed.shape[0], dc:dc+printed.shape[1]] = printed
    return {'input': a, 'output': b}


def _center(f):
    r0, c0, r1, c1 = f['box']
    return (r0+r1-1)/2, (c0+c1-1)/2


def _distance(a, b):
    x, y = _center(a), _center(b)
    return abs(x[0]-y[0]) + abs(x[1]-y[1])


def _nonnearest(info):
    fs = info['frames']
    return any(any(_distance(fs[j['target']], fs[s]) < _distance(fs[j['target']], fs[j['source']])
                       for s in info['sources'] if s != j['source']) for j in info['jobs'])


def _fallback(rng, lesson, test):
    # Sparse, hand-routed members of the same rule after bounded packing attempts.
    result = _specs(rng, lesson, test, fixed=True)
    _need(result is not None, "fallback token construction failed")
    specs, wires = result
    if lesson == 0:
        boxes = [(0, 0, 6, 6), (0, 18, 11, 29), (18, 18, 29, 29), (21, 0, 28, 8)]
        routes = {0: {(2, c) for c in range(6, 13)} | {(r, 12) for r in range(2, 23)} |
                  {(4, c) for c in range(12, 18)} | {(22, c) for c in range(12, 18)}}
    elif lesson == 1:
        boxes = [(23, 0, 29, 7), (0, 0, 7, 6), (0, 18, 8, 25),
                 (15, 18, 22, 26), (11, 0, 18, 8)]
        routes = {
            0: {(25, c) for c in range(7, 13)} | {(r, 12) for r in range(25, 30)} |
               {(29, c) for c in range(12, 30)} | {(r, 29) for r in range(10, 30)} |
               {(10, c) for c in range(21, 30)} | {(r, 21) for r in range(8, 11)},
            1: {(3, c) for c in range(6, 15)} | {(r, 14) for r in range(3, 18)} |
               {(17, c) for c in range(14, 18)}}
    else:
        sh, sw = specs[0]['array'].shape
        th, tw = specs[1]['array'].shape
        boxes = [(23, 0, 23+sh, sw), (0, 30-tw, th, 30), (12, 0, 19, 8)]
        # Count <=5 gives target left edge >=12.
        routes = {0: {(25, c) for c in range(sw, 10)} | {(r, 9) for r in range(2, 26)} |
                  {(2, c) for c in range(9, 30-tw)}}
    pair = _render(specs, boxes, (30, 30), routes, wires)
    _need(pair is not None and np.array_equal(solve(pair['input']), pair['output']), "bad fallback")
    # Transposition also transposes the payload; centering remains well defined.
    if rng.random() < .5:
        pair = {k: v.T.copy() for k, v in pair.items()}
    return pair


def _pair(rng, lesson, test):
    for attempt in range(70):
        made = _specs(rng, lesson, test)
        if made is None:
            continue
        specs, wire_colors = made
        low = 25 if lesson >= 0 or len(specs) >= 5 else 18 if len(specs) >= 4 else 12
        shape = (_ri(rng, low, 30), _ri(rng, low, 30))
        boxes = _place(rng, specs, shape)
        if boxes is None:
            continue
        routes = _route(rng, specs, boxes, shape)
        if routes is None:
            continue
        pair = _render(specs, boxes, shape, routes, wire_colors)
        if pair is None:
            continue
        try:
            info = _analyze(pair['input'])
            if lesson == 1 and not _nonnearest(info):
                continue
            if not np.array_equal(solve(pair['input']), pair['output']):
                continue
        except AssertionError:
            continue
        return pair
    return _fallback(rng, lesson, test)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ne)):
        for i in range(count):
            lesson = i if split == 'train' and i < 2 else -1
            for retry in range(8):
                pair = _pair(rng, lesson, split == 'test')
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise AssertionError("distinct-input retry budget exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "split types")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair keys")
            a, b = _grid(pair['input']), _grid(pair['output'])
            _need(a.shape == b.shape and not np.array_equal(a, b), "identity or resized output")
            _need(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            info = _analyze(a)
            _need(np.array_equal(solve(a), b), "output disagrees with input-derived rule")
            allowed = set()
            for j in info['jobs']:
                r0, c0, r1, c1 = info['frames'][j['target']]['box']
                allowed |= _rect((r0+1, c0+1, r1-1, c1-1))
            _need(set(zip(*np.where(a != b))) <= allowed, "changed source, wire, boundary or inactive chamber")
            analyses[split].append(info)
    train = analyses['train']
    _need({j['axis'] for d in train for j in d['jobs']} == {0, 1}, "missing entry-axis evidence")
    _need(any(len(d['targets']) > len(d['jobs']) for d in train), "no inactive chamber witness")
    _need(any(len(r['targets']) >= 2 for d in train for r in d['routes']), "no branching broadcast witness")
    _need(any(_nonnearest(d) for d in train), "nearest-source shortcut not contradicted")
    _need(any(len(d['sources']) >= 2 for d in train), "no multiple-source witness")
    _need(any((d['frames'][j['source']]['tile'].shape[1-j['axis']] >
               d['frames'][j['source']]['tile'].shape[j['axis']])
              for d in train for j in d['jobs']), "missing short-axis repetition witness")
    _need(any(a['source'] == b['source'] and a['axis'] == b['axis'] and
              d['frames'][a['target']]['shape'] == d['frames'][b['target']]['shape'] and
              d['frames'][a['target']]['area'] == d['frames'][b['target']]['area'] and
              d['frames'][a['target']]['count'] != d['frames'][b['target']]['count']
              for d in train for a in d['jobs'] for b in d['jobs']), "missing equal-size/area different-count witness")
    counts = {d['frames'][j['target']]['count'] for d in train for j in d['jobs']}
    _need({2, 3} <= counts and max(counts) <= 3, "training quantity contract")
    for d in analyses['test']:
        _need(max(d['frames'][j['target']]['count'] for j in d['jobs']) > max(counts), "missing count shift")
    for transform in range(1, 8):
        _need(any(not np.array_equal(f['tile'], np.rot90(f['tile'][:, ::-1] if transform >= 4 else f['tile'], transform % 4))
                  for d in train for f in d['frames'] if f['kind'] == 'source'), "stencil orientation ambiguous")


def self_test():
    """42 seeds, streamed metrics, counterfactuals and bounded corruption checks."""
    shapes, tiles, layouts, colors, signatures = set(), set(), set(), set(), set()
    axes, counts, sizes, source_counts, fans, ports = set(), set(), set(), set(), set(), set()
    metrics = dict(seeds=42, pairs=0, nonnearest_pairs=0, inactive_pairs=0,
                   equal_area_count_witness_puzzles=42, count_shift_puzzles=42)
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = pair['input']
                d = _analyze(a)
                fs = d['frames']
                metrics['pairs'] += 1
                metrics['nonnearest_pairs'] += int(_nonnearest(d))
                metrics['inactive_pairs'] += int(len(d['targets']) > len(d['jobs']))
                digest.update(a.tobytes())
                shapes.add(a.shape)
                colors.add(tuple(sorted(map(int, np.unique(a)))))
                source_counts.add(len(d['sources']))
                fans.update(len(r['targets']) for r in d['routes'])
                for f in fs:
                    if f['kind'] == 'source':
                        tiles.add(_key(f['tile']))
                    else:
                        counts.add(f['count'])
                        sizes.update(f['sizes'])
                for j in d['jobs']:
                    axes.add(j['axis'])
                    ports.add(j['side'])
                    layouts.add((a.shape, fs[j['target']]['box'], fs[j['source']]['box'], j['axis']))
        signatures.add(digest.hexdigest())
        if seed == 0:
            first = puzzle
    _need(len(signatures) == 42 and len(shapes) >= 25, "scene diversity")
    _need(len(tiles) >= 20 and len(layouts) >= 100 and len(colors) >= 20, "structural/palette diversity")
    _need(axes == {0, 1} and counts >= {1, 2, 3, 4, 5}, "axis/count coverage")
    _need(sizes >= {1, 2, 3, 4, 5, 6} and source_counts >= {1, 2} and max(fans) >= 2, "object/network coverage")
    _need(len(ports) == 4, "entry-side coverage")
    other = generate_puzzle(0)
    _need(all(_key(x[k]) == _key(y[k]) for s in first for x, y in zip(first[s], other[s])
              for k in ('input', 'output')), "nondeterminism")
    # Every legal pair count is checked once, with no puzzle archive retained.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            generate_puzzle(1000+nt*10+ne, nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    for lesson in (0, 1, -1):
        for seed in range(3):
            p = _fallback(np.random.default_rng(seed), lesson, lesson < 0)
            _need(np.array_equal(solve(p['input']), p['output']), "fallback regression")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    bad['test'][0]['output'].flat[0] = (int(bad['test'][0]['output'].flat[0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    metrics.update(grid_shapes=len(shapes), binary_stencils=len(tiles), layouts=len(layouts),
                   input_palettes=len(colors), source_counts=sorted(source_counts),
                   counts=sorted(counts), token_sizes=sorted(sizes), entry_sides=sorted(ports))
    return metrics
