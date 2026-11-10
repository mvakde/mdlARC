"""Twin-pattern couplers: a local visual analogy defines rail partnership.

Only stdlib and NumPy. No files, examples, seeds or sampled metadata are read
by solve. Experiments stream in memory; this module writes no artifacts.
"""
from __future__ import annotations

from collections import deque
from numbers import Integral

import numpy as np


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _need(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "grid dimensions")
    _need(np.issubdtype(g.dtype, np.integer), "integer colors required")
    _need(bool(np.all((0 <= g) & (g <= 9))), "color bounds")
    return g


def _key(g):
    return g.shape, g.astype(np.uint8).tobytes()


def _inside(g, p):
    return 0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1]


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _normalise(inp):
    candidates = []
    for k in range(4):
        g = np.rot90(inp, k)
        token = int(g[0, 0])
        if (g.shape[1] >= 2 and np.all(g[:, 0] == token)
                and np.count_nonzero(g == token) == g.shape[0]):
            candidates.append((g, k, token))
    _need(len(candidates) == 1, "unique edge strip required")
    g, k, token = candidates[0]
    bg = int(g[0, 1])
    _need(bg != token and np.all(g[:, 1] == bg), "clear approach column required")
    return g, k, token, bg


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask)))
    result = []
    while todo:
        first = todo.pop()
        cells, queue = {first}, deque([first])
        while queue:
            p = queue.popleft()
            for d in DIRS:
                q = _add(p, d)
                if q in todo:
                    todo.remove(q)
                    cells.add(q)
                    queue.append(q)
        result.append(cells)
    return result


def _reference(g, token, bg):
    candidates = []
    for color in map(int, np.unique(g)):
        if color in (token, bg):
            continue
        parts = _components(g == color)
        if len(parts) != 2:
            continue
        boxes = []
        for cells in parts:
            r, c = min(p[0] for p in cells), min(p[1] for p in cells)
            b, e = max(p[0] for p in cells), max(p[1] for p in cells)
            outline = {(y, x) for y in range(r, b + 1) for x in range(c, e + 1)
                       if y in (r, b) or x in (c, e)}
            if b - r < 3 or e - c < 3 or cells != outline:
                break
            boxes.append((r, c, b + 1, e + 1))
        if len(boxes) != 2:
            continue
        boxes.sort()
        interiors = [g[r+1:b-1, c+1:e-1] for r, c, b, e in boxes]
        a, b = interiors
        if a.shape != b.shape or token in a or token in b:
            continue
        guides = set(map(int, np.unique(g))) - {token, bg, color}
        if not 3 <= len(guides) <= 6:
            continue
        if any(set(map(int, np.unique(x))) != guides | {bg} for x in interiors):
            continue
        mapping = {bg: bg}
        valid = True
        for x, y in zip(a.flat, b.flat):
            x, y = int(x), int(y)
            if x in mapping and mapping[x] != y:
                valid = False
                break
            mapping[x] = y
        if (not valid or set(mapping) != guides | {bg}
                or set(mapping.values()) != guides | {bg}
                or any(mapping.get(mapping[x]) != x for x in mapping)
                or not any(mapping[x] != x for x in guides)
                or not any(mapping[x] == x for x in guides)):
            continue
        candidates.append(dict(frame=color, boxes=boxes, mapping=mapping, guides=guides))
    _need(len(candidates) == 1, "unique complete involutive motif analogy required")
    return candidates[0]


def _flanks(g, p, d, guides):
    n = (-d[1], d[0])
    a, b = _add(p, n), _add(p, (-n[0], -n[1]))
    if _inside(g, a) and _inside(g, b):
        x, y = int(g[a]), int(g[b])
        if x in guides and y in guides:
            return x, y
    return None


def _trace(g, token, bg, guides, mapping):
    """Directional input reader; flanks select entrances and blocked turns."""
    routes, openings = [], []
    for r in range(1, g.shape[0] - 1):
        entrance = None
        for c in range(1, g.shape[1]):
            if int(g[r, c]) != bg:
                break
            sides = _flanks(g, (r, c), (0, 1), guides)
            if sides is not None:
                accepted = mapping[sides[0]] == sides[1]
                openings.append((r, c, sides[0], sides[1], accepted))
                if accepted:
                    entrance = c
                break
        if entrance is None:
            continue
        p, d = (r, 1), (0, 1)
        path, directions, seen = [p], [], set()
        turns = locked = 0
        support = []
        while True:
            _need((p, d) not in seen, "cyclic route")
            seen.add((p, d))
            q = _add(p, d)
            if not _inside(g, q):
                break
            if int(g[q]) == bg:
                p = q
                path.append(p)
                directions.append(d)
                sides = _flanks(g, p, d, guides)
                support.append(sides is not None and mapping[sides[0]] == sides[1])
                continue
            choices, wrong = [], False
            for nd in ((-d[1], d[0]), (d[1], -d[0])):
                q = _add(p, nd)
                if not _inside(g, q) or int(g[q]) != bg:
                    continue
                sides = _flanks(g, q, nd, guides)
                if sides is not None:
                    if mapping[sides[0]] == sides[1]:
                        choices.append(nd)
                    else:
                        wrong = True
            _need(len(choices) <= 1, "ambiguous turn")
            if not choices:
                locked += int(wrong)
                break
            d = choices[0]
            turns += 1
        _need(len(path) == len(set(path)), "self-intersecting route")
        _need(len(path) >= 3, "route too short")
        last = max((i for i, v in enumerate(support) if v), default=-1)
        gap = False
        for i in range(1, max(1, len(support) - 1)):
            if (not support[i] and any(support[:i]) and any(support[i+1:])
                    and directions[i-1] == directions[i] == directions[i+1]):
                gap = True
        routes.append(dict(start=(r, 0), end=p, path=path, turns=turns,
                           locked=locked, gap=gap, tail=len(support)-1-last,
                           open=p[0] in (0, g.shape[0]-1) or p[1] == g.shape[1]-1,
                           entrance=entrance))
    paths = [p for route in routes for p in route['path']]
    _need(len(paths) == len(set(paths)), "routes merge or cross")
    _need(len({x['end'] for x in routes}) == len(routes), "destination conflict")
    return routes, openings


def _analyse(inp):
    g, k, token, bg = _normalise(_grid(inp))
    ref = _reference(g, token, bg)
    routes, openings = _trace(g, token, bg, ref['guides'], ref['mapping'])
    return dict(grid=g, rotation=k, token=token, bg=bg, routes=routes,
                openings=openings, **ref)


def _answer(a):
    out = a['grid'].copy()
    for route in a['routes']:
        out[route['start']] = a['bg']
        out[route['end']] = a['token']
    return np.rot90(out, -a['rotation']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from pixels alone; also accepts a nested integer list."""
    return _answer(_analyse(input_grid))


def _polyline(points):
    path = [points[0]]
    for target in points[1:]:
        p = path[-1]
        d = (int(np.sign(target[0]-p[0])), int(np.sign(target[1]-p[1])))
        if (d[0] == 0) == (d[1] == 0):
            return None
        while p != target:
            p = _add(p, d)
            path.append(p)
    cells = set(path)
    if len(path) != len(cells):
        return None
    for i, p in enumerate(path):
        degree = sum(_add(p, d) in cells for d in DIRS)
        if degree != (1 if i in (0, len(path)-1) else 2):
            return None
    return path


def _random_path(rng, g, bends, closed):
    h, w = g.shape
    p = (int(rng.integers(1, h-1)), 1)
    points, d = [p], (0, 1)
    for i in range(bends + 1):
        p = points[-1]
        if i == bends and not closed and d != (0, -1):
            target = ((h-1 if d[0] > 0 else 0), p[1]) if d[0] else (p[0], w-1)
        else:
            choices = []
            for length in range(3, max(h, w)):
                q = p[0]+length*d[0], p[1]+length*d[1]
                if 1 <= q[0] < h-1 and 4 <= q[1] < w-1:
                    choices.append(q)
            if not choices:
                return None
            target = choices[int(rng.integers(len(choices)))]
        points.append(target)
        d = (-d[1], d[0]) if rng.integers(2) else (d[1], -d[0])
    return _polyline(points)


def _walls(path, g, left, right, margin, cap):
    cells, walls = set(path), {}
    dirs = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:])]
    for i, p in enumerate(path):
        for d in {dirs[max(0, i-1)], dirs[min(i, len(dirs)-1)]}:
            for n, color in (((-d[1], d[0]), left), ((d[1], -d[0]), right)):
                q = _add(p, n)
                if not _inside(g, q) or q[1] < margin or q in cells:
                    continue
                if q in walls and walls[q] != color:
                    return None
                walls[q] = color
    q = _add(path[-1], dirs[-1])
    if _inside(g, q) and q[1] > 0:
        if q in cells:
            return None
        walls[q] = cap
    return walls


def _decorate(rng, path, walls, gap, tail):
    dirs = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:])]
    eligible = []
    for i in range(3, len(path)-3):
        if len(set(dirs[i-2:i+2])) != 1:
            continue
        d, p = dirs[i], path[i]
        sides = (_add(p, (-d[1], d[0])), _add(p, (d[1], -d[0])))
        if all(q in walls for q in sides):
            eligible.append((i, sides))
    if gap and eligible:
        # Keep visible support on both sides, including when the tail is also
        # trimmed. A gap at the last support cell would merely extend the tail.
        interior = [x for x in eligible
                    if eligible[0][0]+2 <= x[0] <= len(path)-7]
        choices = interior or eligible
        start = choices[int(rng.integers(len(choices)))][0]
        length = int(rng.integers(1, min(3, max(1, len(path)-5-start))+1))
        for i, sides in eligible:
            if start <= i < start+length:
                for q in sides:
                    walls.pop(q, None)
    if tail:
        for i in range(len(path)-1, max(2, len(path)-int(rng.integers(2, 5))), -1):
            if i >= 2 and dirs[i-2] != dirs[-1]:
                break
            p, d = path[i], dirs[-1]
            for n in ((-d[1], d[0]), (d[1], -d[0])):
                walls.pop(_add(p, n), None)


def _motif(rng, shape, guides, bg):
    size = int(np.prod(shape))
    for _ in range(80):
        values = list(guides) + [bg]
        values += list(map(int, rng.choice(list(guides)+[bg, bg], size-len(values))))
        rng.shuffle(values)
        a = np.array(values, dtype=np.int8).reshape(shape)
        mask = a != bg
        views = [mask[::-1], mask[:, ::-1], mask[::-1, ::-1]]
        if shape[0] == shape[1]:
            views += [mask.T, mask.T[::-1], mask.T[:, ::-1], mask.T[::-1, ::-1]]
        if all(not np.array_equal(mask, view) for view in views):
            return a
    # A single background corner gives an asymmetric rectangular occupancy.
    # The normal sampled sizes always allow a random asymmetric mask.
    raise AssertionError("bounded motif sampling exhausted")


def _plaques(rng, g, guides, bg, frame, mapping, fallback=False):
    if fallback:
        shape = (2, 4)
        positions = [(1, g.shape[1]-14), (1, g.shape[1]-7)]
    else:
        shapes = [(h, w) for h in (2, 3) for w in (3, 4, 5) if h*w >= len(guides)+1]
        shape = shapes[int(rng.integers(len(shapes)))]
        ph, pw = shape[0]+2, shape[1]+2
        positions = []
        for _ in range(100):
            r = int(rng.integers(g.shape[0]-ph+1))
            c = int(rng.integers(5, g.shape[1]-pw+1))
            if all(r+ph < y or y+ph < r or c+pw < x or x+pw < c for y, x in positions):
                positions.append((r, c))
                if len(positions) == 2:
                    break
        if len(positions) < 2:
            return False
    a = _motif(rng, shape, guides, bg)
    b = np.array([[mapping[int(x)] for x in row] for row in a], dtype=np.int8)
    if rng.integers(2):
        a, b = b, a
    for (r, c), content in zip(positions, (a, b)):
        ph, pw = shape[0]+2, shape[1]+2
        g[r:r+ph, c:c+pw] = frame
        g[r+1:r+ph-1, c+1:c+pw-1] = content
    return True


def _features(a):
    routes = a['routes']
    return {name for name, present in (
        ('turn', any(x['turns'] for x in routes)),
        ('locked', any(x['locked'] for x in routes)),
        ('gap', any(x['gap'] for x in routes)),
        ('tail', any(x['tail'] for x in routes)),
        ('open', any(x['open'] for x in routes)),
        ('closed', any(not x['open'] for x in routes))) if present}


def _make_pair(rng, palette, rotation, index, is_test, fallback=False):
    bg, token, frame = map(int, palette[:3])
    count = int(rng.integers(3, 7))
    guides = list(map(int, palette[3:3+count]))
    rng.shuffle(guides)
    mapping = {x: x for x in guides+[bg]}
    swaps = int(rng.integers(1, (count-1)//2+1))
    for i in range(swaps):
        x, y = guides[2*i:2*i+2]
        mapping[x], mapping[y] = y, x
    nonfixed = [x for x in guides if mapping[x] != x]
    fixed = [x for x in guides if mapping[x] == x]
    width = (28 if is_test else 22) if fallback else int(rng.integers(24 if is_test else 18, 31 if is_test else 24))
    height = 30 if fallback else int(rng.integers(23 if is_test else 20, 31))
    g = np.full((height, width), bg, dtype=np.int8)
    g[:, 0] = token
    if not _plaques(rng, g, guides, bg, frame, mapping, fallback):
        return None
    reserved, expected, expected_paths = set(), {}, {}

    def place(path, a, b, active, lock=False, force_gap=False, force_tail=False):
        nonlocal g, reserved
        if path is None or len(path) < 5:
            return False
        cells = set(path)
        if cells & reserved or any(g[p] != bg for p in cells):
            return False
        first_turn_col = next((p[1] for p, q in zip(path, path[1:]) if p[0] != q[0]), path[-1][1])
        margin = int(rng.integers(2, min(4, first_turn_col-2)+1)) if first_turn_col >= 4 else 2
        walls = _walls(path, g, a, b, margin, int(rng.choice(guides)))
        if walls is None:
            return False
        _decorate(rng, path, walls, force_gap or rng.random() < .35, force_tail or rng.random() < .45)
        wanted_path = path
        if lock:
            turns = [i for i in range(1, len(path)-1)
                     if (path[i][0]-path[i-1][0], path[i][1]-path[i-1][1])
                     != (path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])]
            if not turns:
                return False
            i = turns[0]
            d = (path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])
            # Two identical exchanged colors make a convincing locked turn.
            for n in ((-d[1], d[0]), (d[1], -d[0])):
                q = _add(path[i+1], n)
                if q not in walls:
                    return False
                walls[q] = nonfixed[0]
            wanted_path = path[:i+1]
        if any(q in reserved or (g[q] != bg and g[q] != v) for q, v in walls.items()):
            return False
        candidate = g.copy()
        for q, v in walls.items():
            candidate[q] = v
        wanted = dict(expected)
        paths = dict(expected_paths)
        if active:
            start = (path[0][0], 0)
            wanted[start] = wanted_path[-1]
            paths[start] = wanted_path
        try:
            routes, openings = _trace(candidate, token, bg, set(guides), mapping)
        except AssertionError:
            return False
        if {x['start']: x['end'] for x in routes} != wanted:
            return False
        if any(x['path'] != paths[x['start']] for x in routes):
            return False
        if not active and not any(o[0] == path[0][0] and not o[4] for o in openings):
            return False
        g, reserved = candidate, reserved | cells
        expected.clear()
        expected.update(wanted)
        expected_paths.clear()
        expected_paths.update(paths)
        return True

    demonstration = not is_test and index == 0
    bent_demo = not is_test and index == 1
    # The first two training pairs must teach the conventions even at (2,1).
    requirements = [(nonfixed[0], mapping[nonfixed[0]], True, False),
                    (fixed[0], fixed[0], True, bent_demo),
                    (nonfixed[0], nonfixed[0], False, False),
                    (nonfixed[0], fixed[0], False, False),
                    (mapping[nonfixed[0]], mapping[nonfixed[0]], False, False)]
    if fallback:
        if demonstration:
            paths = [_polyline([(7, 1), (7, width-1)]),
                     _polyline([(12, 1), (12, width-4)]),
                     _polyline([(17, 1), (17, width-3)]),
                     _polyline([(22, 1), (22, width-1)]),
                     _polyline([(27, 1), (27, width-4)])]
        else:
            paths = [_polyline([(7, 1), (7, width-5), (10, width-5), (10, width-1)]),
                     _polyline([(15, 1), (15, width//2), (18, width//2)]),
                     _polyline([(21, 1), (21, width-3)]),
                     _polyline([(25, 1), (25, width-1)]),
                     _polyline([(28, 1), (28, width-4)])]
        for j, (a, b, active, lock) in enumerate(requirements):
            if not place(paths[j], a, b, active, lock,
                         force_gap=j in (0, 3), force_tail=j in (0, 3)):
                return None
    else:
        for j, (a, b, active, lock) in enumerate(requirements):
            for _ in range(150):
                bends = (0 if demonstration or not active else
                         int(rng.integers(1 if bent_demo else 0, 4)))
                path = _random_path(rng, g, bends, closed=(j in (1, 4) or (j == 2 and rng.random() < .5)))
                if place(path, a, b, active, lock, force_gap=demonstration and j in (0, 3),
                         force_tail=demonstration and j in (0, 3)):
                    break
            else:
                return None
        for _ in range(int(rng.integers(0, 4))):
            for _ in range(30):
                a = int(rng.choice(guides))
                active = bool(rng.random() < .75)
                b = mapping[a] if active else int(rng.choice([x for x in guides if x != mapping[a]]))
                bends = 0 if demonstration else int(rng.integers(4))
                path = _random_path(rng, g, bends, closed=bool(rng.integers(2)))
                if place(path, a, b, active):
                    break
    inp = np.rot90(g, rotation).copy()
    try:
        analysis = _analyse(inp)
    except AssertionError:
        return None
    features = _features(analysis)
    if demonstration and not {'gap', 'tail', 'open', 'closed'} <= features:
        return None
    if bent_demo and not {'turn', 'locked'} <= features:
        return None
    # Ground truth is constructed from intended transfers, not solve's answer.
    out = g.copy()
    for start, end in expected.items():
        out[start], out[end] = bg, token
    out = np.rot90(out, rotation).copy()
    if not np.array_equal(out, _answer(analysis)):
        return None
    return {'input': inp, 'output': out}


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n, m = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    palette, base_rotation = rng.permutation(10), int(rng.integers(4))
    mixed_palette, mixed_rotation = bool(rng.integers(2)), bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', n), ('test', m)):
        for i in range(count):
            colors = rng.permutation(10) if mixed_palette else palette
            rotation = (base_rotation+i) % 4 if mixed_rotation else base_rotation
            for attempt in range(25):
                pair = _make_pair(rng, colors, rotation, i, split == 'test', fallback=attempt >= 20)
                if pair is not None and _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError('bounded scene generation exhausted')
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _need(all(isinstance(puzzle[k], list) for k in puzzle), 'split lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, coverage = set(), set()
    for split in ('train', 'test'):
        for i, pair in enumerate(puzzle[split]):
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(inp.shape == out.shape, 'shape preservation')
            _need(_key(inp) not in seen, 'distinct inputs required')
            seen.add(_key(inp))
            a = _analyse(inp)
            _need(np.array_equal(out, _answer(a)), 'incorrect token transfers')
            openings = a['openings']
            _need(any(o[4] and o[2] != o[3] for o in openings), 'unequal accepted witness')
            _need(any(o[4] and o[2] == o[3] for o in openings), 'fixed accepted witness')
            _need(any(not o[4] and o[2] == o[3] for o in openings), 'equal rejected witness')
            _need(any(not o[4] and o[2] != o[3] for o in openings), 'unequal rejected witness')
            canonical_out = np.rot90(out, a['rotation'])
            fixed = (a['grid'] != a['bg']) & (a['grid'] != a['token'])
            _need(np.array_equal(a['grid'][fixed], canonical_out[fixed]), 'rails and motifs preserved')
            _need(np.count_nonzero(inp == a['token']) == np.count_nonzero(out == a['token']), 'conservation')
            _need(np.count_nonzero(inp != out) == 2*len(a['routes']), 'only starts and terminals change')
            features = _features(a)
            if split == 'train':
                coverage |= features
                if i == 0:
                    _need({'gap', 'tail', 'open', 'closed'} <= features, 'first demo evidence')
                if i == 1:
                    _need({'turn', 'locked'} <= features, 'second demo evidence')
            else:
                _need(features <= coverage, 'test convention missing from training')
                _need(a['grid'].shape[1] >= 24, 'test depth shift')
    _need({'gap', 'tail', 'open', 'closed', 'turn', 'locked'} <= coverage, 'demonstration coverage')


def _normalised_structure(g):
    roles = {}
    return tuple(roles.setdefault(int(x), len(roles)) for x in g.flat)


def self_test():
    """Bounded, streaming 42-seed suite plus metamorphic and contract checks."""
    shapes, structures, mappings, rotations, active_counts = set(), set(), set(), set(), set()
    lengths, turns, palettes, plaque_shapes, signatures = set(), set(), set(), set(), set()
    features, pair_count = set(), 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_key(puzzle['train'][0]['input']))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                before = inp.copy()
                _need(np.array_equal(solve(inp), out) and np.array_equal(before, inp), 'pure input-only solve')
                a = _analyse(inp)
                shapes.add(inp.shape)
                structures.add((inp.shape, _normalised_structure(inp)))
                guides = sorted(a['guides'])
                mappings.add(tuple(guides.index(a['mapping'][x]) for x in guides))
                rotations.add(a['rotation'])
                active_counts.add(len(a['routes']))
                lengths.update(len(x['path']) for x in a['routes'])
                turns.update(x['turns'] for x in a['routes'])
                palettes.add(len(guides))
                r, c, b, e = a['boxes'][0]
                plaque_shapes.add((b-r-2, e-c-2))
                features |= _features(a)
                pair_count += 1
        if seed < 3:
            for k in range(4):
                pair = puzzle['test'][0]
                _need(np.array_equal(solve(np.rot90(pair['input'], k)), np.rot90(pair['output'], k)), 'rotation equivariance')
            rename = np.random.default_rng(seed+901).permutation(10)
            _need(np.array_equal(solve(rename[pair['input']]), rename[pair['output']]), 'color equivariance')
            a = _analyse(pair['input'])
            g = a['grid'].copy()
            r, c, b, e = a['boxes'][0]
            y, x, z, w = a['boxes'][1]
            first = g[r+1:b-1, c+1:e-1].copy()
            g[r+1:b-1, c+1:e-1] = g[y+1:z-1, x+1:w-1]
            g[y+1:z-1, x+1:w-1] = first
            swapped = solve(np.rot90(g, -a['rotation']))
            _need(np.array_equal(swapped == a['token'], pair['output'] == a['token']), 'unordered reference')
    _need(len(signatures) == 42 and len(structures) >= 200, 'structural diversity')
    _need(len(shapes) >= 30 and len(mappings) >= 8, 'shape and analogy diversity')
    _need(rotations == {0, 1, 2, 3} and palettes == {3, 4, 5, 6}, 'palette/orientation diversity')
    _need({0, 1, 2} <= turns and len(lengths) >= 20 and len(active_counts) >= 3, 'route diversity')
    _need(len(plaque_shapes) >= 4, 'mosaic size diversity')
    for seed in range(6):
        for index, is_test in ((0, False), (1, False), (0, True)):
            fallback = _make_pair(np.random.default_rng(seed), np.arange(10),
                                  seed % 4, index, is_test, fallback=True)
            _need(fallback is not None, 'constructive fallback must retain evidence')
            _need(np.array_equal(solve(fallback['input']), fallback['output']), 'fallback answer')
    # Concrete counterfactual: same target geometry, a different reference map.
    p = generate_puzzle(0, 2, 1)
    fixture = _make_pair(np.random.default_rng(103), np.arange(10), 0, 0, False, fallback=True)
    _need(fixture is not None, 'reference counterfactual fixture')
    a = _analyse(fixture['input'])
    old = _answer(a)
    nonfixed = next(o[2] for o in a['openings'] if o[4] and o[2] != o[3])
    mate = a['mapping'][nonfixed]
    fixed_color = next(o[2] for o in a['openings'] if o[4] and o[2] == o[3])
    changed_map = dict(a['mapping'])
    changed_map[nonfixed], changed_map[fixed_color], changed_map[mate] = fixed_color, nonfixed, mate
    g = a['grid'].copy()
    r, c, b, e = a['boxes'][0]
    y, x, z, w = a['boxes'][1]
    g[y+1:z-1, x+1:w-1] = np.array([[changed_map[int(v)] for v in row] for row in g[r+1:b-1, c+1:e-1]])
    counterfactual = np.rot90(g, -a['rotation']).copy()
    changed_answer = solve(counterfactual)
    _need(not np.array_equal(changed_answer == a['token'], old == a['token']),
          'reference must affect token transfers, beyond copying the changed motif')
    # Both references must produce legitimate generated-family demonstrations,
    # including all witnesses; a zero-mover out-of-family twin would not suffice.
    for inp, out in ((fixture['input'], old), (counterfactual, changed_answer)):
        twin_puzzle = {'train': [{'input': inp, 'output': out}, p['train'][1]], 'test': p['test']}
        validate_generated_puzzle(twin_puzzle)
    # Omitting B's directional continuation gives the wrong terminal on the
    # second demonstration. Merely selecting movers using the mosaics cannot
    # recover the missing turn geometry.
    b = _analyse(p['train'][1]['input'])
    straight_ends = []
    for route in b['routes']:
        r, c = route['start'][0], 1
        while c+1 < b['grid'].shape[1] and b['grid'][r, c+1] == b['bg']:
            c += 1
        straight_ends.append((r, c))
    _need(straight_ends != [x['end'] for x in b['routes']], 'B geometry must affect terminals')
    # All explicit legal combinations. No file payloads or retained puzzle list.
    for n in range(2, 11):
        for m in range(1, min(4, n-1)+1):
            q = generate_puzzle(57, n, m)
            validate_generated_puzzle(q)
    for n, m in ((1, 1), (11, 1), (2, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    q = generate_puzzle(91, 2, 1)
    r = generate_puzzle(91, 2, 1)
    _need(all(np.array_equal(a[k], b[k]) for s in q for a, b in zip(q[s], r[s])
              for k in ('input', 'output')), 'determinism')
    q['test'][0]['output'] = q['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(q)
    except AssertionError:
        pass
    else:
        raise AssertionError('incorrect output accepted')
    return dict(seeds=42, pairs=pair_count, shapes=len(shapes), structures=len(structures),
                relation_patterns=len(mappings), active_counts=sorted(active_counts),
                route_lengths=len(lengths), bends=sorted(turns), guide_counts=sorted(palettes),
                mosaic_shapes=[list(map(int, shape)) for shape in sorted(plaque_shapes)],
                features=sorted(features))
