"""Remote registration: arrow-selected handles impose rigid-motion constraints.

Only the standard library and NumPy are used. Prediction has no puzzle state.
Construction maintains footprints; decoding reconstructs objects from pixels.
"""
from collections import Counter
import numpy as np

DISTANCE = 5
DIRS = ((-1, 0), (-1, 1), (0, 1), (1, 1),
        (1, 0), (1, -1), (0, -1), (-1, -1))
ORTHO = ((-1, 0), (1, 0), (0, -1), (0, 1))
KING = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(grid.dtype, np.integer), "integer grid required")
    _require(bool(np.all((0 <= grid) & (grid <= 9))), "colors must be 0..9")
    return grid.astype(np.int8, copy=False)


def _components(points, neighbors=KING):
    remaining = set(points)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        found, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    found.add(p)
                    stack.append(p)
        result.append(found)
    return result


def _arrow(tip, direction):
    r, c = tip
    dr, dc = direction
    if dr == 0:
        offsets = ((0, 0), (-1, -dc), (1, -dc))
    elif dc == 0:
        offsets = ((0, 0), (-dr, -1), (-dr, 1))
    else:
        offsets = ((0, 0), (-dr, 0), (0, -dc))
    return {(r + a, c + b) for a, b in offsets}


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _decode(input_grid, distance=DISTANCE):
    grid = _grid(input_grid)
    h, w = grid.shape
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(counts.argmax())
    _require(counts[bg] > grid.size / 2, "background must occupy over half the input")
    parts = _components(map(tuple, np.argwhere(grid != bg).tolist()))
    anchors, other = {}, []
    for cells in parts:
        p = min(cells)
        if len(cells) == 1 and (p[0] in (0, h - 1) or p[1] in (0, w - 1)):
            anchors.setdefault(int(grid[p]), set()).add(p)
        else:
            other.append(cells)
    _require(1 <= len(anchors) <= 4, "need one to four guide identities")
    systems = {}
    for color, cells in anchors.items():
        rows, cols = {r for r, _ in cells}, {c for _, c in cells}
        horizontal = len(rows) == 1 and all(c in (0, w - 1) for _, c in cells)
        vertical = len(cols) == 1 and all(r in (0, h - 1) for r, _ in cells)
        _require(horizontal != vertical, "anchors must define an unambiguous line")
        axis = 0 if horizontal else 1
        guide = next(iter(rows if horizontal else cols))
        _require(0 < guide < grid.shape[axis] - 1, "corner anchors are forbidden")
        _require(not any(s['axis'] == axis and s['guide'] == guide for s in systems.values()),
                 "same-axis guides must have distinct coordinates")
        systems[color] = {'axis': axis, 'guide': guide, 'anchors': cells}
    arrows, carriers = [], []
    for cells in other:
        colors = {int(grid[p]) for p in cells}
        if colors & systems.keys():
            _require(len(colors) == 1 and len(cells) == 3,
                     "guide-colored objects must be monochrome arrowheads")
            matches = [(tip, d) for tip in sorted(cells) for d in DIRS
                       if _arrow(tip, d) == cells]
            _require(len(matches) == 1, "unrecognized arrowhead")
            tip, direction = matches[0]
            endpoint = tuple(tip[k] + distance * direction[k] for k in (0, 1))
            _require(_inside(endpoint, grid.shape), "shot endpoint leaves the canvas")
            arrows.append({'cells': cells, 'tip': tip, 'direction': direction,
                           'endpoint': endpoint, 'color': next(iter(colors))})
        else:
            _require(len(cells) >= 4 and len(_components(cells, ORTHO)) == 1,
                     "carriers must be four-connected and have at least four cells")
            carriers.append({'cells': cells, 'constraints': [set(), set()], 'hits': []})
    _require(arrows and carriers, "need controllers and carriers")
    _require(set(a['color'] for a in arrows) == set(systems), "each guide needs a controller")
    owner = {p: i for i, obj in enumerate(carriers) for p in obj['cells']}
    for arrow in arrows:
        point = arrow['endpoint']
        if point in owner:
            obj = carriers[owner[point]]
            s = systems[arrow['color']]
            obj['constraints'][s['axis']].add(s['guide'] - point[s['axis']])
            obj['hits'].append(arrow)
    for obj in carriers:
        _require(all(len(values) <= 1 for values in obj['constraints']),
                 "inconsistent displacement constraints")
        obj['delta'] = tuple(next(iter(values)) if values else 0
                             for values in obj['constraints'])
        obj['target'] = {(r + obj['delta'][0], c + obj['delta'][1]): int(grid[r, c])
                         for r, c in obj['cells']}
        obj['outside'] = {p for p in obj['target'] if not _inside(p, grid.shape)}
    return grid, bg, systems, arrows, carriers


def _render(decoded):
    grid, bg, systems, arrows, carriers = decoded
    out = np.full_like(grid, bg)
    occupied = set()
    for color, system in systems.items():
        for p in system['anchors']:
            out[p] = color
            occupied.add(p)
    for arrow in arrows:
        for p in arrow['cells']:
            out[p] = arrow['color']
            occupied.add(p)
    for obj in carriers:
        for p, color in obj['target'].items():
            if p in obj['outside']:
                continue
            _require(p not in occupied, "final footprints overlap")
            out[p] = color
            occupied.add(p)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer guides, remote handles and their joint translation from input pixels."""
    return _render(_decode(input_grid))


def _choose(rng, items):
    return items[int(rng.integers(len(items)))]


def _shape(rng, small=False):
    kind = int(rng.integers(5))
    if small:
        kind = 0
    if kind == 0:
        n = int(rng.integers(4, 9 if small else 18))
        cells = {(0, 0)}
        for _ in range(n - 1):
            frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in ORTHO} - cells)
            candidates = [p for p in frontier if
                          max(r for r, _ in cells | {p}) - min(r for r, _ in cells | {p}) <= 5 and
                          max(c for _, c in cells | {p}) - min(c for _, c in cells | {p}) <= 5]
            cells.add(_choose(rng, candidates))
    elif kind in (1, 2):
        h, w = int(rng.integers(2, 6)), int(rng.integers(2, 7))
        if kind == 2:
            h, w = max(3, h), max(3, w)
        cells = {(r, c) for r in range(h) for c in range(w)
                 if kind == 1 or r in (0, h - 1) or c in (0, w - 1)}
        if kind == 2 and rng.random() < .4:
            cells.remove((0, int(rng.integers(1, w - 1))))
    else:
        length = int(rng.integers(4, 8))
        cells = {(r, 0) for r in range(length)}
        for r in rng.choice(length, size=int(rng.integers(1, 4)), replace=False):
            for c in range(1, int(rng.integers(2, 4))):
                cells.add((int(r), c * int((-1, 1)[kind - 3])))
    for _ in range(int(rng.integers(4))):
        cells = {(c, -r) for r, c in cells}
    if rng.random() < .5:
        cells = {(r, -c) for r, c in cells}
    low = tuple(min(p[k] for p in cells) for k in (0, 1))
    return sorted((r - low[0], c - low[1]) for r, c in cells)


def _blocked(inp, bg):
    occupied = inp != bg
    padded = np.pad(occupied, 1)
    return np.logical_or.reduce([padded[1 + r:1 + r + inp.shape[0],
                                       1 + c:1 + c + inp.shape[1]]
                                 for r in (-1, 0, 1) for c in (-1, 0, 1)])


def _separated(groups):
    occupied = set()
    for cells in groups:
        if any((r + dr, c + dc) in occupied for r, c in cells
               for dr in (-1, 0, 1) for dc in (-1, 0, 1)):
            return False
        occupied.update(cells)
    return True


def _place_bundle(rng, inp, out, bg, systems, requests, directions, payload,
                  *, clip=None, teaching=False, dual=False, redundant=False,
                  novel=(), nearest_witness=False, small=False):
    """Place a carrier and its aiming arrows as a jointly constrained module.

    Candidate origins are exhaustively enumerated for each bounded shape proposal.
    Generated labels are drawn from these construction footprints, not solve().
    """
    h, w = inp.shape
    block = _blocked(inp, bg)
    fixed = inp != bg
    # Previous carrier sources are not fixed obstacles at the destination.
    fixed = np.zeros_like(fixed)
    for system in systems:
        for p in system['anchors']:
            fixed[p] = True
    guide_colors = {s['color'] for s in systems}
    fixed |= np.isin(inp, list(guide_colors))
    for _ in range(100):
        points = _shape(rng, small)
        cells = set(points)
        handles, shots, constraint = [], [], {}
        for index, (j, multiplicity) in enumerate(requests):
            s = systems[j]
            options = points
            if teaching:
                lo, hi = min(p[s['axis']] for p in points), max(p[s['axis']] for p in points)
                options = [p for p in points if 2 * p[s['axis']] != lo + hi and
                           p[s['axis']] != (lo + hi) // 2]
            if dual and handles:
                options = [p for p in options if p != handles[0]]
            if s['axis'] in constraint:
                # Different guides may register different parallel slices of the
                # same object, provided they imply the very same displacement.
                required_coordinate = s['guide'] - constraint[s['axis']]
                options = [p for p in options if p[s['axis']] == required_coordinate]
            if not options:
                break
            handle = _choose(rng, options)
            handles.append(handle)
            constraint[s['axis']] = s['guide'] - handle[s['axis']]
            pool = list(directions)
            if teaching and not dual:
                pool = [d for d in pool if d[s['axis']] != 0]
            if len(pool) < multiplicity:
                break
            selected = [pool[int(k)] for k in rng.choice(len(pool), size=multiplicity, replace=False)]
            if redundant and index == 0:
                if not any(a * b == 0 for a, b in selected) or not any(a * b != 0 for a, b in selected):
                    break
            if novel and index == 0:
                selected[0] = _choose(rng, list(novel))
                if len(set(selected)) != len(selected):
                    break
            for d in selected:
                tip = tuple(handle[k] - DISTANCE * d[k] for k in (0, 1))
                shots.append((_arrow(tip, d), s['color']))
        else:
            if not _separated([cells] + [a for a, _ in shots]):
                continue
            bundle = cells | set().union(*(a for a, _ in shots)) if shots else cells
            low, high = [min(p[k] for p in bundle) for k in (0, 1)], [max(p[k] for p in bundle) for k in (0, 1)]
            if any(high[k] - low[k] >= inp.shape[k] for k in (0, 1)):
                continue
            rr, cc = np.meshgrid(np.arange(-low[0], h - high[0]),
                                 np.arange(-low[1], w - high[1]), indexing='ij')
            rr, cc = rr.ravel(), cc.ravel()
            valid = np.ones(len(rr), bool)
            for r, c in bundle:
                valid &= ~block[rr + r, cc + c]
            delta = [(constraint[k] - (rr if k == 0 else cc)) if k in constraint
                     else np.zeros_like(rr) for k in (0, 1)]
            if requests:
                valid &= (delta[0] != 0) | (delta[1] != 0)
            if teaching and dual:
                valid &= (delta[0] != 0) & (delta[1] != 0)
            if nearest_witness:
                s = systems[requests[0][0]]
                other = next(t for t in systems if t['axis'] == s['axis'] and t != s)
                coord = (rr if s['axis'] == 0 else cc) + handles[0][s['axis']]
                valid &= abs(coord - other['guide']) < abs(coord - s['guide'])
            clipped = np.zeros(len(rr), bool)
            visible = np.zeros(len(rr), bool)
            arrow_cells = set().union(*(a for a, _ in shots)) if shots else set()
            for ar, ac in arrow_cells:
                valid &= out[rr + ar, cc + ac] == bg
            for r, c in cells:
                tr, tc = rr + r + delta[0], cc + c + delta[1]
                inside = (tr >= 0) & (tr < h) & (tc >= 0) & (tc < w)
                clipped |= ~inside
                visible |= inside
                br, bc = np.clip(tr, 0, h - 1), np.clip(tc, 0, w - 1)
                valid &= ~inside | ((out[br, bc] == bg) & ~fixed[br, bc])
                for ar, ac in arrow_cells:
                    valid &= (tr != rr + ar) | (tc != cc + ac)
            valid &= visible
            if clip is not None:
                valid &= clipped if clip else ~clipped
            choices = np.flatnonzero(valid)
            if not len(choices):
                continue
            k = int(rng.choice(choices))
            origin = int(rr[k]), int(cc[k])
            shift = int(delta[0][k]), int(delta[1][k])
            body = _choose(rng, payload)
            colors = {p: body for p in points}
            if len(points) >= 6 and rng.random() < .55:
                for ix in rng.choice(len(points), size=int(rng.integers(1, min(4, len(points)) + 1)), replace=False):
                    colors[points[int(ix)]] = _choose(rng, payload)
            for (r, c), color in colors.items():
                inp[origin[0] + r, origin[1] + c] = color
                target = origin[0] + r + shift[0], origin[1] + c + shift[1]
                if _inside(target, inp.shape):
                    out[target] = color
            for cells_a, color in shots:
                for r, c in cells_a:
                    p = origin[0] + r, origin[1] + c
                    inp[p] = out[p] = color
            return True
    return False


def _miss(rng, inp, out, bg, systems, directions):
    block = _blocked(inp, bg)
    options = []
    for s in systems:
        for d in directions:
            for r in range(inp.shape[0]):
                for c in range(inp.shape[1]):
                    endpoint = r + 5 * d[0], c + 5 * d[1]
                    cells = _arrow((r, c), d)
                    if not _inside(endpoint, inp.shape) or inp[endpoint] != bg:
                        continue
                    if all(_inside(p, inp.shape) and not block[p] and out[p] == bg for p in cells):
                        options.append((cells, s['color']))
    if not options:
        return False
    cells, color = _choose(rng, options)
    for p in cells:
        inp[p] = out[p] = color
    return True


def _anchors(rng, dims, palette, axes, edge_axis=None, parallel=False):
    h, w = dims
    systems, used = [], set()
    for j, axis in enumerate(axes):
        options = list(range(1, dims[axis] - 1))
        rng.shuffle(options)
        if edge_axis is not None and j == edge_axis:
            options = [1, dims[axis] - 2]
            rng.shuffle(options)
        for guide in options:
            if any(s['axis'] == axis and (abs(s['guide'] - guide) < (5 if parallel else 2)) for s in systems):
                continue
            ends = (0, 1) if rng.random() < .35 else (int(rng.integers(2)),)
            cells = {(guide, 0 if e == 0 else w - 1) if axis == 0 else
                     (0 if e == 0 else h - 1, guide) for e in ends}
            if not _separated([used, cells]):
                continue
            systems.append({'color': palette[j + 1], 'axis': axis, 'guide': guide, 'anchors': cells})
            used |= cells
            break
        else:
            return None
    return systems


def _scene(rng, params, kind, clipping=False):
    teaching = kind in ('single_demo', 'dual_demo')
    palette = params['palette'] if params['stable'] else tuple(map(int, rng.permutation(10)))
    bg = palette[0]
    if kind == 'single_demo':
        dims = tuple(map(int, rng.integers(22, 31, size=2)))
        axes = [params['axis']] * 2
        requests = [[(0, 2)], [(1, 1)]]
    elif kind == 'dual_demo':
        dims = tuple(map(int, rng.integers(20, 31, size=2)))
        axes = [params['axis'], 1 - params['axis']]
        requests = [[(0, 1), (1, 1)], []]
    else:
        lo = 19 if kind == 'test' else 14
        dims = tuple(map(int, rng.integers(lo, 31, size=2)))
        n = int(rng.integers(1, 5))
        axes = list(map(int, rng.integers(2, size=n)))
        count = int(rng.integers(1, 5))
        requests = []
        for i in range(count):
            j = int(rng.integers(n))
            other = [k for k in range(n) if k != j]
            if other and rng.random() < .45:
                requests.append([(j, 1), (_choose(rng, other), 1)])
            else:
                requests.append([(j, int(rng.choice((1, 2, 3), p=(.72, .23, .05))))])
        if rng.random() < .4:
            requests.append([])
    first_guide = requests[0][0][0]
    systems = _anchors(rng, dims, palette, axes, edge_axis=first_guide if clipping else None,
                       parallel=kind == 'single_demo')
    if systems is None:
        return None
    # Shape span bounds make widely separated parallel handle constraints
    # impossible; choose a single registration instead of a futile placement.
    for request in requests:
        if len(request) == 2:
            x, y = systems[request[0][0]], systems[request[1][0]]
            if x['axis'] == y['axis'] and abs(x['guide'] - y['guide']) > 6:
                request.pop()
    inp, out = np.full(dims, bg, np.int8), np.full(dims, bg, np.int8)
    for s in systems:
        for p in s['anchors']:
            inp[p] = out[p] = s['color']
    payload = palette[len(systems) + 1:]
    pool = DIRS if kind == 'test' else params['train_dirs']
    for i, request in enumerate(requests):
        succeeded = _place_bundle(rng, inp, out, bg, systems, request, pool, payload,
                                  clip=(clipping if i == 0 else False) if teaching else
                                       (False if not clipping else True if i == 0 else None),
                                  teaching=teaching, dual=len(request) == 2,
                                  redundant=kind == 'single_demo' and i == 0,
                                  novel=params['novel'] if kind == 'test' and i == 0 else (),
                                  nearest_witness=kind == 'single_demo' and i == 0,
                                  small=not request or (not teaching and rng.random() < .25))
        if not succeeded and (teaching or i == 0):
            return None
    # Unused guides get background shots; these have the same rule and stay fixed.
    for s in systems:
        if np.sum(inp == s['color']) == len(s['anchors']):
            if not _miss(rng, inp, out, bg, [s], pool):
                return None
    if kind == 'dual_demo' or rng.random() < .3:
        if not _miss(rng, inp, out, bg, systems, pool) and teaching:
            return None
    if np.mean(inp == bg) <= .5:
        return None
    try:
        decoded = _decode(inp)
        if not np.array_equal(_render(decoded), out):
            return None
        if kind == 'single_demo' and _alternative_fits(decoded, out, 'first_obstacle'):
            return None
    except ValueError:
        return None
    return {'input': inp, 'output': out}


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def _fingerprint(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def _count(value, lo, hi, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not lo <= int(value) <= hi:
        raise ValueError(f'{name} must be {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a = _count(num_train, 2, 10, 'num_train')
    b = _count(num_test, 1, 4, 'num_test')
    if a <= b:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    # Rotate/reflection of a teaching direction subset, never a color-to-vector map.
    turns, mirror = int(rng.integers(4)), bool(rng.integers(2))
    def transform(d):
        r, c = d
        if mirror:
            c = -c
        for _ in range(turns):
            r, c = c, -r
        return r, c
    train_dirs = tuple(transform(d) for d in ((-1, 0), (0, 1), (-1, 1), (1, -1)))
    params = {'palette': tuple(map(int, rng.permutation(10))), 'stable': bool(rng.random() < .3),
              'axis': int(rng.integers(2)), 'train_dirs': train_dirs,
              'novel': tuple(d for d in DIRS if d not in train_dirs)}
    teach_clip = bool(rng.random() < .55)
    plans = [('single_demo', False), ('dual_demo', teach_clip)]
    plans += [('extra', bool(teach_clip and rng.random() < .25)) for _ in range(a - 2)]
    order = rng.permutation(a)
    plans = [plans[int(i)] for i in order]
    plans += [('test', bool(teach_clip and rng.random() < .6)) for _ in range(b)]
    puzzle, seen = {'train': [], 'test': []}, set()
    for i, (kind, clip) in enumerate(plans):
        for _ in range(100):
            pair = _scene(rng, params, kind, clip)
            if pair is not None and _key(pair['input']) not in seen:
                seen.add(_key(pair['input']))
                puzzle['train' if i < a else 'test'].append(pair)
                break
        else:
            raise RuntimeError('bounded scene construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    grid, bg, systems, arrows, carriers = _decode(pair['input'])
    activated = [c for c in carriers if c['hits']]
    nearest_wrong = False
    for c in activated:
        for arrow in c['hits']:
            s = systems[arrow['color']]
            distance = abs(arrow['endpoint'][s['axis']] - s['guide'])
            nearest_wrong |= any(t['axis'] == s['axis'] and
                                 abs(arrow['endpoint'][s['axis']] - t['guide']) < distance
                                 for t in systems.values())
    return {'shape': grid.shape, 'background': bg, 'guides': len(systems),
            'axes': tuple(sorted({s['axis'] for s in systems.values()})),
            'carriers': len(carriers), 'sizes': tuple(len(c['cells']) for c in carriers),
            'arrows': len(arrows), 'directions': tuple(sorted({a['direction'] for a in arrows})),
            'dual': any(all(c['constraints']) and all(c['delta']) for c in activated),
            'distinct_handles': any(len({a['endpoint'] for a in c['hits']}) >= 2 and all(c['constraints'])
                                    for c in activated),
            'parallel_handles': any(len({a['color'] for a in c['hits']}) >= 2 and
                                    sum(bool(x) for x in c['constraints']) == 1 for c in activated),
            'redundant': any(max(Counter(a['endpoint'] for a in c['hits']).values()) >= 2 for c in activated),
            'unhit': any(not c['hits'] for c in carriers),
            'miss': any(not any(a in c['hits'] for c in carriers) for a in arrows),
            'clips': sum(bool(c['outside']) for c in carriers), 'nearest_wrong': nearest_wrong,
            'full_move': any(c['delta'] != (0, 0) and not c['outside'] for c in activated),
            'shape_keys': tuple(tuple(sorted((r - min(x for x, _ in c['cells']),
                                             col - min(y for _, y in c['cells']))
                                            for r, col in c['cells'])) for c in carriers)}


def _alternative_output(decoded, mode):
    """Named shortcut diagnostics, not an independent correctness oracle."""
    grid, bg, systems, arrows, carriers = decoded
    if mode == 'no_guides':
        out = grid.copy()
        for obj in carriers:
            if obj['hits']:
                for p in obj['cells']:
                    out[p] = obj['hits'][0]['color']
        return out
    changed = []
    owner = {p: i for i, obj in enumerate(carriers) for p in obj['cells']}
    first_hits = [[] for _ in carriers]
    if mode == 'first_obstacle':
        for arrow in arrows:
            for step in range(1, DISTANCE + 1):
                p = tuple(arrow['tip'][k] + step * arrow['direction'][k] for k in (0, 1))
                if p in owner:
                    first_hits[owner[p]].append(dict(arrow, endpoint=p))
                    break
    for i, obj in enumerate(carriers):
        center = tuple((min(p[k] for p in obj['cells']) + max(p[k] for p in obj['cells'])) // 2
                       for k in (0, 1))
        if mode == 'no_arrows':
            s = min(systems.values(), key=lambda s: (abs(s['guide'] - center[s['axis']]),
                                                      s['axis'], s['guide']))
            delta = tuple(s['guide'] - center[k] if k == s['axis'] else 0 for k in (0, 1))
        elif mode == 'one_axis':
            delta = (obj['delta'][0], 0) if all(obj['constraints']) else obj['delta']
        else:
            values = [[], []]
            hits = first_hits[i] if mode == 'first_obstacle' else obj['hits']
            for arrow in hits:
                s = systems[arrow['color']]
                point = center if mode == 'centers' else arrow['endpoint']
                if mode == 'nearest':
                    s = min((x for x in systems.values() if x['axis'] == s['axis']),
                            key=lambda x: (abs(x['guide'] - point[x['axis']]), x['guide']))
                values[s['axis']].append(s['guide'] - point[s['axis']])
            if mode == 'sum_hits':
                delta = tuple(sum(v) for v in values)
            else:
                _require(all(len(set(v)) <= 1 for v in values), 'shortcut has conflicting constraints')
                delta = tuple(v[0] if v else 0 for v in values)
        target = {(r + delta[0], c + delta[1]): int(grid[r, c]) for r, c in obj['cells']}
        changed.append(dict(obj, delta=delta, target=target,
                            outside={p for p in target if not _inside(p, grid.shape)}))
    return _render((grid, bg, systems, arrows, changed))


def _alternative_fits(decoded, output, mode):
    try:
        return np.array_equal(_alternative_output(decoded, mode), output)
    except ValueError:
        return False


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'wrong puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'illegal counts')
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'wrong pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'input/output dimensions differ')
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            decoded = _decode(inp)
            _require(np.array_equal(_render(decoded), out), 'wrong remote registration output')
            _require(not np.array_equal(inp, out), 'unchanged generated example')
            # Every payload cell's color and offset survive unless geometrically clipped.
            _, bg, systems, arrows, carriers = decoded
            for s in systems.values():
                _require(all(inp[p] == out[p] for p in s['anchors']), 'anchor moved')
            for arrow in arrows:
                _require(all(inp[p] == out[p] for p in arrow['cells']), 'controller changed')
            for obj in carriers:
                for p, color in obj['target'].items():
                    if _inside(p, inp.shape):
                        _require(int(out[p]) == color, 'payload cell lost or recolored')
            features[split].append(_features(pair))
    train = features['train']
    _require(any(f['full_move'] and f['redundant'] and f['nearest_wrong'] for f in train),
             'need complete, redundant, color-selected single-axis registration evidence')
    _require(any(f['dual'] and f['distinct_handles'] and f['unhit'] and f['miss'] for f in train),
             'need distinct-handle dual-axis movement, selection and misses in training')
    _require(any(f['clips'] for f in train) or not any(f['clips'] for f in features['test']),
             'test clipping must have a demonstration')
    train_dirs = {d for f in train for d in f['directions']}
    for f in features['test']:
        _require(bool(set(f['directions']) - train_dirs), 'test must extend compass coverage')
    # The teaching shots must locate their handles at five, not a competing distance.
    for mode in ('no_arrows', 'no_guides', 'centers', 'nearest', 'sum_hits',
                 'one_axis', 'first_obstacle'):
        _require(not all(_alternative_fits(_decode(p['input']), p['output'], mode)
                         for p in puzzle['train']), f'training fails to reject {mode}')
    for distance in range(1, 30):
        if distance == DISTANCE:
            continue
        fits = True
        for pair in puzzle['train']:
            try:
                fits &= np.array_equal(_render(_decode(pair['input'], distance)), pair['output'])
            except ValueError:
                fits = False
            if not fits:
                break
        _require(not fits, f'training also fits distance {distance}')


def _fixture_tests():
    """Small hand-constructed checks of joint constraints, redundancy and clipping."""
    def fixture(parallel=False, clipped=False):
        g = np.zeros((25, 25), np.int8)
        row = 1 if clipped else 4
        g[row, 0] = 1
        g[6, 24] = 2 if parallel else 0
        if not parallel:
            g[24, 15] = 2
        g[11:15, 5:9] = 6
        g[11, 5] = 7
        # A boundary-touching payload is not an isolated controller anchor.
        g[0:2, 21:24] = 8
        hit = (13 if clipped else 12, 6)
        shots = [((hit[0] + 5, 6), (-1, 0), 1),
                 ((hit[0] + 5, 1), (-1, 1), 1)]
        shots += [((9, 7), (1, 0), 2)] if parallel else [((13, 3), (0, 1), 2)]
        for tip, d, color in shots:
            for p in _arrow(tip, d):
                g[p] = color
        return g
    for parallel, clipped in ((False, False), (True, False), (False, True)):
        g = fixture(parallel, clipped)
        decoded = _decode(g)
        out = solve(g)
        carrier = next(c for c in decoded[-1] if len(c['cells']) == 16)
        expected_delta = (-12, 7) if clipped else (-8, 0) if parallel else (-8, 7)
        _require(carrier['delta'] == expected_delta, 'joint displacement fixture')
        _require(len(carrier['outside']) == (4 if clipped else 0), 'clipping fixture')
        _require(np.all(out[0:2, 21:24] == 8), 'unhit edge carrier must remain')
        _require(np.all(out[11:15, 5:9] == 0), 'source erasure fixture')
        for p, color in carrier['target'].items():
            if _inside(p, g.shape):
                _require(int(out[p]) == color, 'rigid colored-offset fixture')
        if not parallel:
            for mode in ('no_arrows', 'no_guides', 'sum_hits', 'one_axis', 'first_obstacle'):
                _require(not _alternative_fits(decoded, out, mode), f'fixture fails ablation {mode}')
    inconsistent = fixture(True, False)
    inconsistent[6, 24], inconsistent[7, 24] = 0, 2
    try:
        solve(inconsistent)
    except ValueError:
        pass
    else:
        raise AssertionError('contradictory parallel constraints accepted')
    aligned = fixture(False, False)
    aligned[4, 0], aligned[12, 0] = 0, 1
    aligned[24, 15], aligned[24, 8] = 0, 2
    _require(np.array_equal(solve(aligned), aligned), 'already registered carrier must stay')
    swap = np.zeros((23, 23), np.int8)
    swap[4, 0], swap[12, 0] = 1, 2
    swap[11:14, 10:13], swap[3:6, 10:13] = 6, 7
    swap[3, 10] = 8
    for tip, direction, color in (((17, 11), (-1, 0), 1), ((4, 6), (0, 1), 2)):
        for point in _arrow(tip, direction):
            swap[point] = color
    swapped = solve(swap)
    _require(np.all(swapped[3:6, 10:13] == 6), 'simultaneous source reuse')
    _require(int(swapped[11, 10]) == 8 and int(swapped[12, 11]) == 7,
             'simultaneous patterned carrier exchange')
    vanished = np.zeros((30, 30), np.int8)
    vanished[0:21, 0], vanished[0, 0:21] = 6, 6
    vanished[1, 29], vanished[29, 1] = 1, 2
    for tip, direction, color in (((20, 5), (0, -1), 1), ((5, 20), (-1, 0), 2)):
        for point in _arrow(tip, direction):
            vanished[point] = color
    _require(not np.any(solve(vanished) == 6), 'fully clipped compatible carrier fixture')


def self_test() -> None:
    """Bounded 42-seed structural/semantic checks plus metamorphic mutations."""
    _fixture_tests()
    records, signatures, shapes, motifs, all_directions = [], set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_fingerprint(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _features(pair)
                records.append(f)
                shapes.add(f['shape'])
                motifs.update(f['shape_keys'])
                all_directions.update(f['directions'])
                before = pair['input'].copy()
                _require(np.array_equal(solve(before), pair['output']), 'prediction mismatch')
                _require(np.array_equal(before, pair['input']), 'solve mutated its input')
    _require(len(signatures) == 42, 'seed diversity')
    _require(len(shapes) >= 45 and len(motifs) >= 70, 'geometry diversity')
    _require(all_directions == set(DIRS), 'eight-direction coverage')
    _require({1, 2, 3, 4} <= {f['guides'] for f in records}, 'guide count diversity')
    _require(len({f['carriers'] for f in records}) >= 4, 'carrier count diversity')
    _require(len({f['arrows'] for f in records}) >= 5, 'controller count diversity')
    _require(len({f['background'] for f in records}) == 10, 'palette-role diversity')
    _require(any(f['clips'] for f in records) and any(f['unhit'] for f in records), 'boundary/selection coverage')
    p = generate_puzzle(101, 2, 1)
    _require(_fingerprint(p) == _fingerprint(generate_puzzle(101, 2, 1)), 'determinism')
    rng = np.random.default_rng(87)
    for turns in (1, 2, 3):
        q = {'train': [], 'test': []}
        for split in q:
            for pair in p[split]:
                palette = rng.permutation(10)
                q[split].append({k: palette[np.rot90(v, turns)].astype(np.int8) for k, v in pair.items()})
        validate_generated_puzzle(q)
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(200 + 10 * a + b, a, b)
            _require(len(puzzle['train']) == a and len(puzzle['test']) == b, 'count mismatch')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')


if __name__ == '__main__':
    self_test()
    print('42-seed self-test passed')
