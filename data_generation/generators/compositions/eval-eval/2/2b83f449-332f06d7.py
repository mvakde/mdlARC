"""Rotary sieves: footprint reachability selects a gravity-ranked delivery bay.

Only standard library and NumPy. No files, examples, seed or construction
metadata are consulted by solve(). No execution traces are retained.
"""
from collections import deque
import hashlib

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'invalid grid dimensions')
    _require(np.issubdtype(a.dtype, np.integer), 'grid must contain integers')
    _require(bool(np.all((a >= 0) & (a <= 9))), 'invalid color')
    return a.astype(np.int8, copy=True)


def _neighbors(p, shape):
    r, c = p
    for x, y in ((r-1, c), (r, c-1), (r, c+1), (r+1, c)):
        if 0 <= x < shape[0] and 0 <= y < shape[1]:
            yield x, y


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        component, queue = [p], deque([p])
        while queue:
            for q in _neighbors(queue.popleft(), mask.shape):
                if q in unseen:
                    unseen.remove(q)
                    component.append(q)
                    queue.append(q)
        result.append(component)
    return result


def _box(cells):
    points = np.asarray(cells)
    r, c = map(int, points.min(axis=0))
    rr, cc = map(int, points.max(axis=0) + 1)
    return r, c, rr-r, cc-c


def _square(cells):
    r, c, h, w = _box(cells)
    _require(h == w and len(cells) == h*w, 'marker is not a solid square')
    return (r, c), h


def _roles(inp):
    """Infer roles from a wall frame, thin bars and solid marker components."""
    border = np.concatenate((inp[0], inp[-1], inp[:, 0], inp[:, -1]))
    _require(len(np.unique(border)) == 1, 'a uniform wall frame is required')
    wall = int(border[0])
    colors = set(map(int, np.unique(inp))) - {wall}
    _require(len(colors) == 4, 'five distinct visible roles required')
    parts = {color: _components(inp == color) for color in sorted(colors)}
    candidates = []
    for rotor in sorted(colors):
        boxes = [_box(x) for x in parts[rotor]]
        if not all(min(h, w) == 1 and max(h, w) >= 3 and max(h, w) % 2
                   and len(cells) == h*w
                   for cells, (_, _, h, w) in zip(parts[rotor], boxes)):
            continue
        for mover in sorted(colors - {rotor}):
            if len(parts[mover]) != 1:
                continue
            try:
                start, k = _square(parts[mover][0])
            except AssertionError:
                continue
            for goal in sorted(colors - {rotor, mover}):
                if len(parts[goal]) < 2:
                    continue
                try:
                    docks = [_square(cells) for cells in parts[goal]]
                    _require(all(size == k for _, size in docks), 'different bay sizes')
                    _require(len({p[0] for p, _ in docks}) == len(docks), 'bay depths must differ')
                except AssertionError:
                    continue
                floor = next(iter(colors - {rotor, mover, goal}))
                candidates.append((wall, floor, mover, goal, rotor, start, k,
                                   sorted(p for p, _ in docks), boxes))
    _require(len(candidates) == 1, 'visible roles must have a unique interpretation')
    return candidates[0]


def _turn(inp, roles):
    wall, floor, mover, goal, rotor, start, k, docks, boxes = roles
    out = inp.copy()
    writes = {}
    for r, c, h, w in boxes:
        size = max(h, w)
        a = size // 2
        cr, cc = r + h//2, c + w//2
        _require(a <= cr < inp.shape[0]-a and a <= cc < inp.shape[1]-a,
                 'rotor tile crosses boundary')
        tile = inp[cr-a:cr+a+1, cc-a:cc+a+1]
        _require(bool(np.all(np.isin(tile, [floor, rotor]))), 'rotor tile is not an empty square chamber')
        for x in range(cr-a, cr+a+1):
            for y in range(cc-a, cc+a+1):
                value = rotor if (y == cc if h == 1 else x == cr) else floor
                _require((x, y) not in writes or writes[x, y] == value, 'conflicting rotor writes')
                writes[x, y] = value
    for p, value in writes.items():
        out[p] = value
    return out


def _valid(mask, k):
    if k > min(mask.shape):
        return np.zeros((0, 0), dtype=bool)
    return np.lib.stride_tricks.sliding_window_view(mask, (k, k)).all(axis=(-1, -2))


def _distances(mask, start):
    result = np.full(mask.shape, -1, dtype=np.int16)
    if mask.size == 0 or not mask[start]:
        return result
    result[start] = 0
    queue = deque([start])
    while queue:
        p = queue.popleft()
        for q in _neighbors(p, mask.shape):
            if mask[q] and result[q] < 0:
                result[q] = result[p] + 1
                queue.append(q)
    return result


def _destination(scene, roles, point=False):
    wall, floor, mover, goal, rotor, start, k, docks, boxes = roles
    allowed = (scene != wall) & (scene != rotor)
    distance = _distances(_valid(allowed, 1 if point else k), start)
    eligible = [p for p in docks if distance[p] >= 0]
    return max(eligible, key=lambda p: p[0]) if eligible else start, distance, eligible


def _analyze(inp):
    roles = _roles(inp)
    wall, floor, mover, goal, rotor, start, k, docks, boxes = roles
    turned = _turn(inp, roles)
    destination, distance, eligible = _destination(turned, roles)
    out = turned.copy()
    out[inp == mover] = floor
    r, c = destination
    out[r:r+k, c:c+k] = mover
    before, _, _ = _destination(inp, roles)
    point, _, _ = _destination(turned, roles, point=True)
    info = {'roles': roles[:5], 'k': k, 'start': start, 'docks': docks,
            'destination': destination, 'eligible': eligible, 'before': before,
            'point': point, 'rotors': len(boxes), 'bars': tuple(max(h, w) for _, _, h, w in boxes),
            'axes': tuple(int(h == 1) for _, _, h, w in boxes),
            'travel': int(distance[destination]),
            'nearest': min((int(distance[p]) for p in eligible), default=0),
            'shape': inp.shape}
    return out, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Turn the visible baffles; choose the lowest fully reachable marked bay.

    Gravity ranks destinations, not intermediate steps: sideways and upward
    detours are allowed. All translations preserve the entire square footprint.
    """
    return _analyze(_grid(input_grid))[0]


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(name + ' must be an integer')
        if not low <= int(value) <= high:
            raise ValueError(name + ' outside legal range')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _layout(rng, k, columns, small_rotor):
    """A branched chamber graph with one opening gate and one size filter.

    Room/passage geometry is painted directly, never via parent generators.
    Extra chambers grow as a random tree; optional narrow edges create loops.
    """
    q = 2*k + 1 + (2 if k == 2 and columns <= 3 and rng.random() < .3 else 0)
    row_count = 4 if q == 5 and rng.random() < .4 else 3
    gap_max = min(3, (28-row_count*q)//(row_count-1),
                  (28-columns*q)//max(1, columns-1))
    gap_max = max(1, gap_max)
    rg = rng.integers(1, gap_max+1, size=row_count-1)
    cg = rng.integers(1, gap_max+1, size=columns-1)
    rows = [1 + q//2]
    cols = [1 + q//2]
    for gap in rg:
        rows.append(rows[-1] + q + int(gap))
    for gap in cg:
        cols.append(cols[-1] + q + int(gap))
    h, w = rows[-1] + q//2 + 2, cols[-1] + q//2 + 2
    _require(h <= 30 and w <= 30, 'layout budget')
    spine = int(rng.integers(columns))
    side = spine + (1 if spine == 0 else -1 if spine == columns-1 else int(rng.choice([-1, 1])))
    base_row = int(rng.integers(row_count-2))
    S, R, D = (base_row, spine), (base_row+1, spine), (base_row+2, spine)
    T, V, U = (base_row, side), (base_row+1, side), (base_row+2, side)
    base = {S, R, D, T, V, U}
    edges = [(S, R, k), (R, D, k), (S, T, k),
             (T, V, k if small_rotor else int(rng.integers(1, k))),
             (V, U, k)]
    if rng.random() < .65:
        edges.append((D, U, int(rng.integers(1, k))))
    active = set(base)
    remaining = {(i, j) for i in range(row_count) for j in range(columns)} - active
    # All additional nodes attach to this same physical routing problem.
    while remaining:
        choices = [(a, b) for a in sorted(active) if a != R
                   for b in sorted(remaining) if abs(a[0]-b[0])+abs(a[1]-b[1]) == 1]
        _require(bool(choices), 'growth stalled')
        a, b = choices[int(rng.integers(len(choices)))]
        edges.append((a, b, int(rng.integers(1, k+2))))
        active.add(b)
        remaining.remove(b)
    existing = {frozenset((a, b)) for a, b, _ in edges}
    for a in sorted(active):
        for b in sorted(active):
            if a < b and a != R and b != R and frozenset((a, b)) not in existing \
                    and abs(a[0]-b[0])+abs(a[1]-b[1]) == 1 and rng.random() < .25:
                edges.append((a, b, int(rng.integers(1, k))))
    boxes = {}
    for node in sorted(active):
        hh, ww = map(int, rng.integers(k+1, q+1, size=2))
        if node == R:
            hh = ww = q
        if node == V and small_rotor:
            hh = ww = 2*int(rng.integers(1, k)) + 1
        cr, cc = rows[node[0]], cols[node[1]]
        boxes[node] = (cr-hh//2, cc-ww//2, hh, ww)
    grid = np.zeros((h, w), dtype=np.int8)
    for r, c, hh, ww in boxes.values():
        grid[r:r+hh, c:c+ww] = 1
    half = int(rng.integers(2))
    for a, b, width in edges:
        ar, ac, ah, aw = boxes[a]
        br, bc, bh, bw = boxes[b]
        if a[0] != b[0]:
            lo, hi = max(ac, bc), min(ac+aw, bc+bw)
            if R in (a, b):
                center = cols[spine]
                lo, hi = (lo, min(hi, center)) if half == 0 else (max(lo, center+1), hi)
            width = min(width, hi-lo)
            _require(width >= (k if R in (a, b) else 1), 'main port does not fit')
            offset = int(rng.integers(lo, hi-width+1))
            top, bottom = (ar+ah, br) if ar < br else (br+bh, ar)
            grid[top:bottom, offset:offset+width] = 1
        else:
            lo, hi = max(ar, br), min(ar+ah, br+bh)
            width = min(width, hi-lo)
            offset = int(rng.integers(lo, hi-width+1))
            left, right = (ac+aw, bc) if ac < bc else (bc+bw, ac)
            grid[offset:offset+width, left:right] = 1
    # 0 wall, 1 floor, 2 mover, 3 bays, 4 baffles before palette permutation.
    for node in (R, V) if small_rotor else (R,):
        r, c, hh, ww = boxes[node]
        grid[r+hh//2, c:c+ww] = 4
    return grid, boxes, S, R, D, T, V, U


def _make_pair(rng, k, columns, small_rotor, palette, goal_limit, need_rank=False, need_depth=False):
    for attempt in range(160 if need_rank or need_depth else 100):
        try:
            inp, boxes, S, R, D, T, V, U = _layout(rng, k, columns, small_rotor)
            bay_zone = np.zeros(inp.shape, dtype=bool)
            for node, (r, c, hh, ww) in boxes.items():
                if node != R and (node != V or not small_rotor):
                    bay_zone[r:r+hh, c:c+ww] = True
            r, c, hh, ww = boxes[S]
            start = (r+int(rng.integers(hh-k+1)), c+int(rng.integers(ww-k+1)))
            inp[start[0]:start[0]+k, start[1]:start[1]+k] = 2
            # Rotate geometry BEFORE selecting destinations. Gravity is always
            # screen-down, so this is not a silent rotation of the convention.
            turns = int(rng.choice(4, p=[.45, .25, .05, .25]))
            inp, bay_zone = np.rot90(inp, turns).copy(), np.rot90(bay_zone, turns).copy()
            start, _ = _square(list(map(tuple, np.argwhere(inp == 2))))
            bars = [_box(cells) for cells in _components(inp == 4)]
            roles = (0, 1, 2, 3, 4, start, k, [], bars)
            turned = _turn(inp, roles)
            prevalid = _valid((inp != 0) & (inp != 4), k)
            postvalid = _valid((turned != 0) & (turned != 4), k)
            pre = _distances(prevalid, start)
            post = _distances(postvalid, start)
            point = _distances((turned != 0) & (turned != 4), start)
            candidates = [tuple(map(int, p)) for p in np.argwhere(
                prevalid & postvalid & _valid(bay_zone & (inp != 2), k))]
            rng.shuffle(candidates)

            def separated(p, q):
                return p[0]+k < q[0] or q[0]+k < p[0] or p[1]+k < q[1] or q[1]+k < p[1]

            goals = None
            for enabled in candidates:
                if post[enabled] < 0 or pre[enabled] >= 0:
                    continue
                shallow = next((p for p in candidates if post[p] >= 0 and pre[p] >= 0
                                and p[0] < enabled[0] and post[p] < post[enabled]
                                and separated(p, enabled)), None)
                if shallow is None:
                    continue
                blocked = next((p for p in candidates if post[p] < 0 and point[p] >= 0
                                and p[0] > enabled[0] and separated(p, enabled)
                                and separated(p, shallow)), None)
                if blocked is not None:
                    witnesses = [shallow, enabled, blocked]
                    if need_depth:
                        another = next((p for p in candidates if post[p] < 0 and point[p] >= 0
                                        and p[0] > enabled[0]
                                        and p[0] not in {q[0] for q in witnesses}
                                        and all(separated(p, q) for q in witnesses)), None)
                        if another is None:
                            continue
                        witnesses.append(another)
                    if need_rank:
                        farther = next((p for p in candidates if post[p] > post[enabled]
                                        and p[0] < enabled[0]
                                        and p[0] not in {q[0] for q in witnesses}
                                        and all(separated(p, q) for q in witnesses)), None)
                        if farther is None:
                            continue
                        witnesses.append(farther)
                    goals = witnesses
                    break
            _require(goals is not None, 'no geometrically distinguished three-bay witness')
            for p in candidates:
                if len(goals) >= goal_limit:
                    break
                if p[0] not in {q[0] for q in goals} and all(separated(p, q) for q in goals):
                    goals.append(p)
            _require(len(goals) == goal_limit, 'insufficient separated bay positions')
            for r, c in goals:
                inp[r:r+k, c:c+k] = 3
            # Only horizontal reflection preserves the fixed downward ranking.
            if rng.random() < .5:
                inp = np.fliplr(inp).copy()
            # Extra padding translates the graph and changes canvas dimensions.
            pads = []
            for dimension in inp.shape:
                total = int(rng.integers(min(3, 30-dimension)+1))
                before = int(rng.integers(total+1))
                pads.append((before, total-before))
            inp = np.pad(inp, pads)
            inp = np.asarray(palette, dtype=np.int8)[inp]
            out, info = _analyze(inp)
            _require(len(info['eligible']) >= 2, 'need competing reachable bays')
            _require(info['destination'] != info['before'], 'rotation must change the selected bay')
            _require(info['destination'] != info['point'], 'footprint must change the selected bay')
            _require(info['travel'] > info['nearest'] > 0, 'nearest-bay shortcut must fail')
            _require(len(info['docks']) >= 3, 'need inaccessible deeper bay')
            if need_rank:
                _, distances, _ = _destination(_turn(inp, _roles(inp)), _roles(inp))
                _require(info['travel'] < max(int(distances[p]) for p in info['eligible']),
                         'farthest-bay shortcut must fail in the rank demonstration')
            if need_depth:
                _require(sum(p[0] > info['destination'][0] for p in info['docks']) >= 2,
                         'depth demonstration needs at least two excluded lower bays')
            return {'input': inp, 'output': out}
        except AssertionError:
            continue
    raise RuntimeError('bounded rotary-sieve construction exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.choice(10, 5, replace=False)))
    size_shift = bool(rng.integers(2))
    specs = [(2, True, False, True), (2 if size_shift else 3, False, True, False)]
    specs += [(2 if size_shift else int(rng.integers(2, 4)), bool(rng.integers(2)), False, False)
              for _ in range(num_train-2)]
    rng.shuffle(specs)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            if split == 'train':
                k, small_rotor, need_rank, need_depth = specs[index]
                columns, goals = (3, 4) if need_rank or need_depth else (int(rng.integers(2, 4)), 3)
            else:
                need_rank = need_depth = False
                k = 3 if size_shift else int(rng.integers(2, 4))
                small_rotor = bool(rng.integers(2))
                columns = 3 if k == 3 else int(rng.integers(3, 5))
                goals = int(rng.integers(4, 6))
            for _ in range(12):
                pair = _make_pair(rng, k, columns, small_rotor, palette, goals, need_rank, need_depth)
                signature = (pair['input'].shape, pair['input'].tobytes())
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct-input budget exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, palettes, training = set(), set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'canvas changed')
            signature = (inp.shape, inp.tobytes())
            _require(signature not in seen, 'duplicate input')
            seen.add(signature)
            expected, info = _analyze(inp)
            _require(np.array_equal(out, expected), 'incorrect rotation, delivery or preservation')
            palettes.add(info['roles'])
            wall, floor, mover, goal, rotor = info['roles']
            k = info['k']
            _require(int((out == mover).sum()) == k*k, 'square area was not conserved')
            _square(list(map(tuple, np.argwhere(out == mover))))
            _require(int((inp == goal).sum()) - int((out == goal).sum()) == k*k,
                     'exactly one bay must be replaced')
            _require(int((inp == rotor).sum()) == int((out == rotor).sum()), 'baffle area changed')
            _require(len(info['eligible']) >= 2 and len(info['docks']) >= 3, 'missing competing bays')
            _require(info['before'] != info['destination'], 'A ablation has no destination effect')
            _require(info['point'] != info['destination'], 'B ablation has no destination effect')
            _require(info['travel'] > info['nearest'] > 0, 'nearest bay is not distinguished')
            if split == 'train':
                _, distances, _ = _destination(_turn(inp, _roles(inp)), _roles(inp))
                info['farthest'] = max(int(distances[p]) for p in info['eligible'])
                info['depth_rank'] = 1 + sum(p[0] > info['destination'][0] for p in info['docks'])
                training.append(info)
    _require(len(palettes) == 1, 'palette roles must be shared within a puzzle')
    _require(any(x['rotors'] >= 2 for x in training) and any(x['rotors'] == 1 for x in training),
             'training must show both a rotary filter and an ordinary narrow passage')
    _require(any(x['travel'] < x['farthest'] for x in training), 'farthest-bay shortcut is not distinguished')
    _require(len({x['depth_rank'] for x in training}) >= 2, 'fixed bay-depth ordinal is not distinguished')


def self_test():
    """Stream 42 seeds: bounded memory, no files, deterministic geometry metrics."""
    # A literal geometry with analytically fixed destinations, independently
    # of the sampler and its chosen goals. A 2-square can use the left half
    # of a length-5 rotor, but cannot use a length-3 rotor or a width-1 edge.
    fixture = np.zeros((22, 16), dtype=np.int8)
    for r in (1, 15):
        for c in (1, 9):
            fixture[r:r+5, c:c+5] = 1
    fixture[8:13, 1:6] = 1
    fixture[9:12, 10:13] = 1
    fixture[6:8, 1:3] = fixture[13:15, 1:3] = 1
    fixture[1:3, 6:9] = 1
    fixture[6:9, 10:12] = fixture[12:15, 10:12] = 1
    fixture[17, 6:9] = 1
    fixture[10, 1:6] = fixture[10, 10:13] = 4
    for k, destination in ((2, (15, 1)), (1, (17, 11))):
        inp = fixture.copy()
        inp[3:3+k, 1:1+k] = 2
        for r, c in ((1, 9), (15, 1), (17, 11)):
            inp[r:r+k, c:c+k] = 3
        expected = inp.copy()
        expected[8:13, 1:6] = expected[9:12, 10:13] = 1
        expected[8:13, 3] = expected[9:12, 11] = 4
        expected[3:3+k, 1:1+k] = 1
        r, c = destination
        expected[r:r+k, c:c+k] = 2
        _require(np.array_equal(solve(inp), expected), 'literal footprint/capacity example failed')
        if k == 2:
            # Transposing changes screen-down ranking: (1,9) becomes (9,1),
            # lower than the other reachable bay (1,15). Recompute delivery,
            # rather than rotating a completed answer and silently gravity.
            other = inp.T.copy()
            expected_t = inp.copy()
            expected_t[8:13, 1:6] = expected_t[9:12, 10:13] = 1
            expected_t[8:13, 3] = expected_t[9:12, 11] = 4
            expected_t[3:5, 1:3] = 1
            expected_t[1:3, 9:11] = 2
            _require(np.array_equal(solve(other), expected_t.T), 'screen-down ranking changed with rotor axis')
    shapes, layouts, signatures, palettes, sizes, bars, goals, rotors = (set() for _ in range(8))
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        repeat = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for i, pair in enumerate(puzzle[split]):
                inp, out = pair['input'], pair['output']
                _require(np.array_equal(inp, repeat[split][i]['input']) and
                         np.array_equal(out, repeat[split][i]['output']), 'nondeterminism')
                saved = inp.copy()
                _require(np.array_equal(solve(inp), out) and np.array_equal(inp, saved), 'solver changed input')
                _, info = _analyze(inp)
                normalized = np.zeros(inp.shape, dtype=np.int8)
                for code, color in enumerate(info['roles']):
                    normalized[inp == color] = code
                digest.update(inp.tobytes())
                shapes.add(inp.shape)
                layouts.add(hashlib.sha256(normalized.tobytes()).hexdigest())
                palettes.add(info['roles'])
                sizes.add(info['k'])
                bars.update(info['bars'])
                goals.add(len(info['docks']))
                rotors.add(info['rotors'])
        signatures.add(digest.hexdigest())
    _require(len(signatures) == 42 and len(layouts) >= 180, 'structure diversity collapsed')
    _require(len(shapes) >= 40 and len(palettes) >= 35, 'geometry or palette diversity collapsed')
    _require(sizes == {2, 3} and {3, 5, 7} <= bars and {3, 4, 5} <= goals and rotors == {1, 2},
             'capacity, bay or rotor variation missing')
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            generate_puzzle(901 + 10*a+b, a, b)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                   (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Reject supplied wrong answers rather than trusting construction state.
    bad = generate_puzzle(0)
    roles = _roles(bad['test'][0]['input'])
    wall, floor = roles[:2]
    bad['test'][0]['output'][0, 0] = floor
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('altered scenery accepted')
