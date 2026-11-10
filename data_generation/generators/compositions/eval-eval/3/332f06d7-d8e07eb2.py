"""Single-use key courier: exact glyph lookup interleaved with square motion.

The solver parses pixels only. Generation uses a separate ordered route sweep.
Structural colors 2/3/8/9 are fixed; wall, floor, and glyph inks vary visibly.
"""

from collections import deque
import numpy as np


HOST_SEEDS = (0, 438089067, 678975651, 603118693, 556889966, 37137902,
              468112178, 769642315, 741942986, 171900398, 630118144,
              878249096, 240382498, 81689692, 578416913, 724630224,
              83231860, 709336743, 421609407, 341307942, 387689545,
              199638044, 634546607, 478306327, 296528892, 92914062,
              131229728, 374850219, 341691567, 532812621, 829789083,
              735006196, 820678092, 608505986, 331440558, 759296028,
              642957491, 835458715, 902623365, 419047232, 286855550,
              317650491)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid size")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=True)


def _neighbors(p, shape):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        q = p[0] + dr, p[1] + dc
        if 0 <= q[0] < shape[0] and 0 <= q[1] < shape[1]:
            yield q


def _distances(mask, starts):
    d = np.full(mask.shape, -1, dtype=np.int16)
    queue = deque()
    for p in starts:
        p = tuple(map(int, p))
        if mask[p] and d[p] < 0:
            d[p] = 0
            queue.append(p)
    while queue:
        p = queue.popleft()
        for q in _neighbors(p, mask.shape):
            if mask[q] and d[q] < 0:
                d[q] = d[p] + 1
                queue.append(q)
    return d


def _rect(mask):
    points = np.argwhere(mask)
    _require(len(points) > 0, "empty rectangle")
    r, c = map(int, points.min(0))
    rr, cc = map(int, points.max(0) + 1)
    _require(len(points) == (rr - r) * (cc - c), "not a solid rectangle")
    return r, c, rr, cc


def _token(array, background):
    mask = array != background
    rr, cc = np.where(mask)
    _require(len(rr) > 0, "empty token")
    crop = array[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    occupied = crop != background
    inks = np.unique(crop[occupied])
    _require(len(inks) == 1, "token must have one ink")
    return (int(inks[0]), crop.shape, tuple(map(int, occupied.ravel())))


def _mask(token):
    return np.array(token[2], bool).reshape(token[1])


def _shape(token):
    return token[1:]


def _orient(grid, orientation):
    a = np.fliplr(grid) if orientation >= 4 else grid
    return np.rot90(a, orientation % 4).copy()


def _parse_canonical(g):
    _require(g.shape[0] >= 11 and g.shape[1] >= 8, "scene too small")
    dividers = np.flatnonzero(np.all(g == 9, axis=1))
    _require(tuple(dividers) == (6,), "tray divider must be the sole full 9 row")
    wall = int(g[0, 0])
    _require(wall not in (2, 3, 8, 9), "invalid wall role")
    _require(np.all(g[[0, 5]] == wall), "tray margin damaged")
    columns = np.flatnonzero(g[1] == 9)
    _require(len(columns) >= 7, "missing tray frame")
    lo, hi = int(columns.min()), int(columns.max())
    _require(0 < lo < hi < g.shape[1] - 1 and (hi - lo) % 3 == 0,
             "invalid tray extent")
    slots = (hi - lo) // 3
    _require(2 <= slots <= 7, "invalid tray capacity")
    expected = np.full((6, g.shape[1]), wall, np.int8)
    expected[1, lo:hi + 1] = expected[4, lo:hi + 1] = 9
    expected[1:5, lo:hi + 1:3] = 9
    keys = {}
    for i in range(slots):
        c = lo + 3 * i + 1
        inner = g[2:4, c:c + 2]
        expected[2:4, c:c + 2] = inner
        if np.any(inner != wall):
            t = _token(inner, wall)
            _require(t[0] not in (2, 3, 8, 9), "invalid key ink")
            _require(t not in keys, "keys must be unique; locks may repeat")
            keys[t] = c
    _require(len(keys) >= 2 and np.array_equal(expected, g[:6]), "invalid key tray")
    arena = g[7:]
    sr, sc, er, ec = _rect(arena == 2)
    k = er - sr
    _require(k == ec - sc and 2 <= k <= 4, "mover must be a 2..4 square")
    gr, gc, ger, gec = _rect(arena == 3)
    _require(ger - gr == gec - gc == k, "goal size differs")
    neighbors = set()
    for p in map(tuple, np.argwhere(arena == 2)):
        for q in _neighbors(p, arena.shape):
            if int(arena[q]) not in (wall, 2, 3, 8, 9):
                neighbors.add(int(arena[q]))
    _require(len(neighbors) == 1, "start must have a visible floor approach")
    floor = neighbors.pop()
    _require(all(t[0] != floor for t in keys), "floor cannot be key ink")
    base = (arena == floor) | (arena == 2) | (arena == 3)
    remaining = ~base & (arena != wall)
    gates = []
    labels = np.full(arena.shape, -1, dtype=np.int8)
    while remaining.any():
        start = tuple(np.argwhere(remaining)[0])
        component = _distances(remaining, [start]) >= 0
        r, c, rr, cc = _rect(component)
        panel = arena[r:rr, c:cc]
        _require(8 in panel and min(panel.shape) >= 2, "invalid locked panel")
        t = _token(panel, 8)
        _require(t[0] not in (wall, floor, 2, 3, 8, 9), "invalid lock ink")
        _require(max(t[1]) <= 2, "lock glyph exceeds 2x2")
        _require((panel.shape[0] == k and 4 <= panel.shape[1] <= 5) or
                 (panel.shape[1] == k and 4 <= panel.shape[0] <= 5),
                 "panel must span a full-width passage")
        labels[component] = len(gates)
        gates.append((t, (r, c, rr, cc)))
        remaining[component] = False
    _require(2 <= len(gates) <= 8, "expected two to eight lock panels")
    all_floor = base | (labels >= 0)
    cell_d = _distances(all_floor, np.argwhere(arena == 3))
    _require(np.all(cell_d[all_floor] >= 0), "corridor must be cell-connected")
    # A corridor loop would surround a wall island. Wide square elbows do not.
    wall_mask = arena == wall
    boundary = np.zeros(arena.shape, bool)
    boundary[[0, -1]] = True
    boundary[:, [0, -1]] = True
    exterior = _distances(wall_mask, np.argwhere(wall_mask & boundary)) >= 0
    _require(np.all(exterior[wall_mask]), "route has a shortcut loop")
    return dict(g=g, wall=wall, floor=floor, keys=keys, slots=slots,
                arena=arena, k=k, start=(sr, sc), goal=(gr, gc),
                gates=gates, labels=labels, base=base, all_floor=all_floor,
                cell_d=cell_d)


def _parse(input_grid):
    grid = _grid(input_grid)
    candidates = []
    for rotation in range(4):
        try:
            info = _parse_canonical(np.rot90(grid, rotation))
        except AssertionError:
            continue
        info['rotation'] = rotation
        candidates.append(info)
    _require(len(candidates) == 1, "scene orientation is ambiguous or invalid")
    return candidates[0]


def _placements(mask, k):
    return np.lib.stride_tricks.sliding_window_view(mask, (k, k)).all(axis=(-1, -2))


def _run(info):
    """Reachability and local activation, without route or generation metadata."""
    k = info['k']
    allowed = info['base'].copy()
    all_valid = _placements(info['all_floor'], k)
    available = set(info['keys'])
    opened, consumed = [], set()
    touchable = set()
    while True:
        valid = _placements(allowed, k)
        distance = _distances(valid, [info['start']])
        reachable = distance >= 0
        touchable = set()
        for p in map(tuple, np.argwhere(reachable)):
            for q in _neighbors(p, valid.shape):
                if not valid[q] and all_valid[q]:
                    r, c = q
                    ids = set(map(int, info['labels'][r:r+k, c:c+k].ravel()))
                    ids.discard(-1)
                    ids.difference_update(opened)
                    _require(len(ids) <= 1, "two locks contacted simultaneously")
                    touchable.update(ids)
        _require(len(touchable) <= 1, "competing forward locks")
        if not touchable:
            break
        gate = next(iter(touchable))
        token, (r, c, rr, cc) = info['gates'][gate]
        if token not in available:
            break
        available.remove(token)
        consumed.add(token)
        opened.append(gate)
        allowed[r:rr, c:cc] = True
    success = bool(reachable[info['goal']])
    if success:
        destination = info['goal']
    else:
        windows = np.lib.stride_tricks.sliding_window_view(info['cell_d'], (k, k))
        destinations = []
        for score in (windows.min(axis=(-1, -2)), windows.sum(axis=(-1, -2))):
            choices = np.argwhere(reachable & (score == score[reachable].min()))
            _require(len(choices) == 1, "ambiguous stopping alignment")
            destinations.append(tuple(map(int, choices[0])))
        _require(len(set(destinations)) == 1, "ambiguous stopping front")
        destination = destinations[0]
    out = info['g'].copy()
    for token in consumed:
        c = info['keys'][token]
        out[2:4, c:c+2] = info['wall']
    arena = out[7:]
    for gate in opened:
        _, (r, c, rr, cc) = info['gates'][gate]
        arena[r:rr, c:cc] = info['floor']
    arena[info['arena'] == 2] = info['floor']
    r, c = destination
    arena[r:r+k, c:c+k] = 2
    stop = 'goal' if success else ('lock' if touchable else 'width')
    repeated = bool(touchable and info['gates'][next(iter(touchable))][0] in consumed)
    matching_stranded = sum(t in available and i not in opened
                            for i, (t, _) in enumerate(info['gates']))
    return np.rot90(out, -info['rotation']).copy(), dict(
        success=success, stop=stop, repeated=repeated, destination=destination,
        opened=tuple(opened), consumed=consumed, available=available,
        travel=int(distance[destination]), stranded=matching_stranded)


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _run(_parse(input_grid))[0]


def _shape_bank():
    bank = {}
    for code in range(1, 16):
        a = np.array([(code >> i) & 1 for i in range(4)], np.int8).reshape(2, 2)
        t = _token(a, 0)
        bank[_shape(t)] = _mask(t)
    return tuple(bank.values())


_SHAPES = _shape_bank()


def _vocabulary(rng, inks):
    ids = rng.choice(len(_SHAPES), 4, replace=False)
    masks = [_SHAPES[int(i)] for i in ids]
    def t(mask, ink):
        return int(ink), mask.shape, tuple(map(int, mask.ravel()))
    x, y, z = t(masks[0], inks[0]), t(masks[1], inks[0]), t(masks[0], inks[1])
    extra = [t(masks[2], inks[2]), t(masks[3], inks[3])]
    missing = t(masks[1], inks[1])
    return x, y, z, extra, missing


def _walk(rng, k, count):
    # Fit the arena beneath the six-row key tray and its divider.
    choices = [(nr, nc) for nr in range(1, 4) for nc in range(1, 5)
               if nr * nc >= count and nr*k + (nr-1)*6 <= 21
               and nc*k + (nc-1)*6 <= 28]
    nr, nc = choices[int(rng.integers(len(choices)))]
    best = []
    for _ in range(48):
        p = int(rng.integers(nr)), int(rng.integers(nc))
        path, used = [p], {p}
        while len(path) < count:
            options = [q for q in _neighbors(path[-1], (nr, nc)) if q not in used]
            if not options:
                break
            q = options[int(rng.integers(len(options)))]
            path.append(q)
            used.add(q)
        if len(path) > len(best):
            best = path
        if len(path) == count:
            break
    if len(best) < count:
        snake = [(r, c if r % 2 == 0 else nc-1-c)
                 for r in range(nr) for c in range(nc)]
        begin = int(rng.integers(len(snake)-count+1))
        best = snake[begin:begin+count]
        if rng.integers(2):
            best.reverse()
    def coordinates(n, limit):
        gaps = [6] * (n-1)
        budget = limit - n*k - sum(gaps)
        for i in rng.permutation(n-1):
            extra = int(rng.integers(min(2, budget)+1))
            gaps[int(i)] += extra
            budget -= extra
        return [0] + list(np.cumsum([k+v for v in gaps]))
    rows, cols = coordinates(nr, 21), coordinates(nc, 28)
    return [(int(rows[r]), int(cols[c])) for r, c in best]


def _route_anchors(nodes):
    route = [nodes[0]]
    for a, b in zip(nodes, nodes[1:]):
        dr, dc = int(np.sign(b[0]-a[0])), int(np.sign(b[1]-a[1]))
        p = a
        while p != b:
            p = p[0]+dr, p[1]+dc
            route.append(p)
    return route


def _draw_token(block, token, background):
    block[:] = background
    mask = _mask(token)
    r, c = (block.shape[0]-mask.shape[0])//2, (block.shape[1]-mask.shape[1])//2
    sub = block[r:r+mask.shape[0], c:c+mask.shape[1]]
    sub[mask] = token[0]


def _make_pair(rng, k, lesson, test, wall, floor, inks, orientation):
    x, y, z, extra, missing = _vocabulary(rng, inks)
    keys = [x, y, z] + extra[:int(rng.integers(3))]
    count = int(rng.integers(6, 10 if k == 2 else 7)) if test else int(rng.integers(4, 6))
    if lesson == 'repeat':
        count = max(count, 5)
    nodes = _walk(rng, k, count)
    edges = count-1
    events = {}
    if lesson == 'success':
        positions = sorted(map(int, rng.choice(edges, 2, replace=False)))
        events.update(zip(positions, (('gate', x), ('gate', y))))
        if test:
            others = [i for i in range(edges) if i not in events]
            for i, token in zip(rng.permutation(others), keys[2:]):
                events[int(i)] = ('gate', token)
    else:
        first = int(rng.integers(edges-3 if lesson == 'repeat' else edges-2))
        failure = int(rng.integers(first+(2 if lesson == 'repeat' else 1), edges-1))
        events[first] = ('gate', x)
        if lesson == 'narrow':
            events[failure] = ('narrow', int(rng.integers(1, k)))
        elif lesson == 'repeat':
            events[failure] = ('gate', x)
            between = list(range(first+1, failure))
            if between:
                events[int(rng.choice(between))] = ('gate', y)
        else:
            events[failure] = ('gate', missing)
        events[failure+1] = ('gate', z if lesson != 'narrow' else y)
        # Further inaccessible structure is real input, not discarded suffix.
        if test:
            for i in range(failure+2, edges):
                if rng.random() < .5:
                    events[i] = ('gate', keys[int(rng.integers(len(keys)))])
                elif rng.random() < .5:
                    events[i] = ('narrow', int(rng.integers(1, k)))
    arena = np.full((23, 30), wall, np.int8)
    for r, c in nodes:
        arena[r:r+k, c:c+k] = floor
    panels = []
    open_edges = [i for i in range(edges) if i not in events]
    bulge_lesson = int(rng.choice(open_edges)) if open_edges and lesson == 'success' else -1
    for i, (a, b) in enumerate(zip(nodes, nodes[1:])):
        horizontal = a[0] == b[0]
        if horizontal:
            start, end, transverse = min(a[1], b[1])+k, max(a[1], b[1]), a[0]
        else:
            start, end, transverse = min(a[0], b[0])+k, max(a[0], b[0]), a[1]
        def region(low, high, offset=0, width=k):
            return ((slice(transverse+offset, transverse+offset+width), slice(low, high))
                    if horizontal else
                    (slice(low, high), slice(transverse+offset, transverse+offset+width)))
        arena[region(start, end)] = floor
        event = events.get(i)
        if event is None and (i == bulge_lesson or rng.random() < .55):
            # A passable widening followed by a narrowing back to k defeats
            # "stop at the first width reduction" without changing the rail.
            room = (21 if horizontal else 28) - transverse - k
            if room > 0:
                extra_width = int(rng.integers(1, min(2, room)+1))
                arena[region(start+1, end-1, 0, k+extra_width)] = floor
        if event and event[0] == 'narrow':
            width = event[1]
            arena[region(start+1, end-1)] = wall
            offset = int(rng.integers(k-width+1))
            arena[region(start+1, end-1, offset, width)] = floor
        elif event:
            depth = int(rng.integers(4, min(5, end-start-2)+1))
            low = int(rng.integers(start+1, end-depth))
            rs, cs = region(low, low+depth)
            _draw_token(arena[rs, cs], event[1], 8)
            panels.append((event[1], (rs.start, cs.start, rs.stop, cs.stop)))
    for position, color in ((nodes[0], 2), (nodes[-1], 3)):
        r, c = position
        arena[r:r+k, c:c+k] = color
    # Construction oracle: sweep the explicitly built route, opening only a
    # panel contacted by the next square footprint. It never calls solve().
    state = arena.copy()
    available, consumed, opened = set(keys), set(), set()
    destination = nodes[0]
    for r, c in _route_anchors(nodes)[1:]:
        window = state[r:r+k, c:c+k]
        if np.any(window == wall):
            break
        hits = [j for j, (_, (pr, pc, rr, cc)) in enumerate(panels)
                if j not in opened and r < rr and r+k > pr and c < cc and c+k > pc]
        _require(len(hits) <= 1, "construction overlaps locks")
        if hits:
            j = hits[0]
            token, (pr, pc, rr, cc) = panels[j]
            if token not in available:
                break
            available.remove(token)
            consumed.add(token)
            opened.add(j)
            state[pr:rr, pc:cc] = floor
        destination = r, c
    state[arena == 2] = floor
    r, c = destination
    state[r:r+k, c:c+k] = 2
    # Crop empty lattice rows/columns, then add independently sampled margins.
    rr, cc = np.where(arena != wall)
    r0, r1, c0, c1 = int(rr.min()), int(rr.max()+1), int(cc.min()), int(cc.max()+1)
    arena, state = arena[r0:r1, c0:c1], state[r0:r1, c0:c1]
    slots = len(keys) + int(rng.integers(2))
    width = max(arena.shape[1]+2, 3*slots+3)
    height = arena.shape[0]+2
    height += int(rng.integers(min(3, 23-height)+1))
    width += int(rng.integers(min(3, 30-width)+1))
    top = int(rng.integers(1, height-arena.shape[0]))
    left = int(rng.integers(1, width-arena.shape[1]))
    inp = np.full((7+height, width), wall, np.int8)
    inp[6] = 9
    lo = int(rng.integers(1, width-(3*slots+1)))
    hi = lo+3*slots
    inp[1, lo:hi+1] = inp[4, lo:hi+1] = 9
    inp[1:5, lo:hi+1:3] = 9
    assignment = list(keys) + [None]*(slots-len(keys))
    rng.shuffle(assignment)
    locations = {}
    for i, token in enumerate(assignment):
        if token is not None:
            c = lo+3*i+1
            _draw_token(inp[2:4, c:c+2], token, wall)
            locations[token] = c
    inp[7+top:7+top+arena.shape[0], left:left+arena.shape[1]] = arena
    out = inp.copy()
    out[7+top:7+top+state.shape[0], left:left+state.shape[1]] = state
    for token in consumed:
        c = locations[token]
        out[2:4, c:c+2] = wall
    return {'input': _orient(inp, orientation), 'output': _orient(out, orientation)}


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError('counts must be integers')
        if not low <= int(value) <= high:
            raise ValueError('counts out of range')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation([0, 1, 4, 5, 6, 7])))
    wall, floor, *inks = palette
    orientation = int(rng.integers(8))
    mixed_orientation = bool(rng.integers(2))
    scale_shift = bool(rng.integers(2))
    lessons = ['success', 'narrow']
    if num_train >= 3:
        lessons.append('repeat')
    if num_train >= 4:
        lessons.append('missing')
    while len(lessons) < num_train:
        lessons.append(str(rng.choice(['success', 'narrow', 'repeat', 'missing'])))
    tests = ['success', 'narrow'] if num_train == 2 else ['success', 'narrow', 'repeat', 'missing']
    puzzle, seen = {'train': [], 'test': []}, set()
    schedule = [('train', lesson, 2 if i == 0 else 3 if i == 1 else int(rng.integers(2, 4)))
                for i, lesson in enumerate(lessons)]
    schedule += [('test', str(rng.choice(tests)), 4 if scale_shift else int(rng.integers(2, 4)))
                 for _ in range(num_test)]
    for split, lesson, k in schedule:
        for _ in range(32):
            o = int(rng.integers(8)) if mixed_orientation else orientation
            pair = _make_pair(rng, k, lesson, split == 'test', wall, floor, inks, o)
            # An asymmetric throat beside an elbow can have competing forward
            # alignments. Sample a different scene instead of adding a tie rule.
            try:
                prediction = solve(pair['input'])
            except AssertionError:
                continue
            _require(np.array_equal(prediction, pair['output']), 'route sweep and pixel solver disagree')
            signature = pair['input'].shape, pair['input'].tobytes()
            if signature not in seen:
                seen.add(signature)
                puzzle[split].append(pair)
                break
        else:
            raise AssertionError('distinct-input construction exhausted')
    for split in puzzle:
        puzzle[split] = [puzzle[split][int(i)] for i in rng.permutation(len(puzzle[split]))]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    try:
        nt, _ = _counts(len(puzzle['train']), len(puzzle['test']))
    except ValueError as error:
        raise AssertionError(str(error)) from error
    seen, rows = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            signature = inp.shape, inp.tobytes()
            _require(signature not in seen, 'duplicate input')
            seen.add(signature)
            info = _parse(inp)
            expected, result = _run(info)
            _require(np.array_equal(expected, out), 'incorrect key consumption, opening, motion, or scenery')
            _require(result['travel'] > 0 and len(result['opened']) > 0, 'degenerate example')
            _require(len(result['consumed']) == len(result['opened']), 'keys are single-use')
            _require(len(result['available']) + len(result['consumed']) == len(info['keys']), 'key accounting')
            rows.append((split, info, result))
    training = rows[:nt]
    _require(any(r['success'] and len(r['opened']) >= 2 for _, _, r in training), 'need multi-lock success')
    _require(any(r['stop'] == 'width' and r['stranded'] for _, _, r in training), 'need width stop with stranded keys')
    if any(r['stop'] == 'lock' for split, _, r in rows if split == 'test'):
        _require(any(r['stop'] == 'lock' for _, _, r in training), 'lock failure must be taught')
    if any(r['repeated'] for split, _, r in rows if split == 'test'):
        _require(any(r['repeated'] for _, _, r in training), 'single-use repetition must be taught')
    for same_shape in (False, True):
        witnessed = False
        for _, info, result in training:
            for used in result['consumed']:
                for unused in result['available']:
                    witnessed |= ((_shape(used) == _shape(unused) and used[0] != unused[0])
                                  if same_shape else
                                  (used[0] == unused[0] and _shape(used) != _shape(unused)))
        _require(witnessed, 'training must distinguish both shape and ink')


def _fingerprint(puzzle):
    return tuple((p[k].shape, p[k].tobytes()) for s in ('train', 'test')
                 for p in puzzle[s] for k in ('input', 'output'))


def _controlled_fixture(k, kind):
    """Straight scenes with destinations specified by coordinates, not solve."""
    x = (4, (2, 2), (1, 0, 1, 1))
    y = (4, (2, 2), (1, 0, 0, 1))
    z = (5, (2, 2), (1, 0, 1, 1))
    second = x if kind == 'repeat' else (5, y[1], y[2]) if kind == 'missing' else y
    inp = np.zeros((13, 30), np.int8)
    inp[6] = 9
    inp[1, 8:18] = inp[4, 8:18] = 9
    inp[1:5, 8:18:3] = 9
    for index, token in enumerate((x, y, z)):
        _draw_token(inp[2:4, 9+3*index:11+3*index], token, 0)
    arena = inp[7:]
    arena[1:1+k, 1:27] = 1
    if kind == 'width':
        arena[3:1+k, 13:15] = 0  # the same two-cell throat for both sizes
    _draw_token(arena[1:1+k, 5:9], x, 8)
    _draw_token(arena[1:1+k, 19:23], second, 8)
    arena[1:1+k, 1:1+k] = 2
    arena[1:1+k, 27-k:27] = 3
    expected = inp.copy()
    expected[2:4, 9:11] = 0
    out = expected[7:]
    out[1:1+k, 5:9] = 1
    out[arena == 2] = 1
    if kind == 'width' and k == 3:
        destination = (1, 10)
    elif kind in ('missing', 'repeat'):
        destination = (1, 19-k)
    else:
        destination = (1, 27-k)
        out[1:1+k, 19:23] = 1
        expected[2:4, 12:14] = 0
    r, c = destination
    out[r:r+k, c:c+k] = 2
    return inp, expected


def _route_features(info):
    """Observable cell-route length and turns, for diversity measurement only."""
    p, old, turns = info['start'], None, 0
    while info['cell_d'][p] > 0:
        options = [q for q in _neighbors(p, info['arena'].shape)
                   if info['cell_d'][q] == info['cell_d'][p]-1]
        _require(bool(options), 'cell route disappeared')
        if old is not None:
            straight = (p[0]+old[0], p[1]+old[1])
            q = straight if straight in options else options[0]
        else:
            q = options[0]
        direction = q[0]-p[0], q[1]-p[1]
        turns += old is not None and old != direction
        old, p = direction, q
    return int(info['cell_d'][info['start']]), int(turns)


def self_test():
    """Bounded semantic, mutation, determinism, and 42-seed diversity checks."""
    # These fixtures make the inherited concepts separately necessary. A
    # two-wide throat passes k=2 but stops k=3 at column 10; the next usable
    # key and lock remain intact. Changing only a lock's ink stops k=2 at
    # column 17. Reusing an already consumed key also stops it there.
    for k, kind in ((2, 'width'), (3, 'width'), (2, 'missing'), (2, 'repeat')):
        inp, expected = _controlled_fixture(k, kind)
        for orientation in range(8):
            _require(np.array_equal(solve(_orient(inp, orientation)), _orient(expected, orientation)),
                     'controlled width, exact-match, or single-use fixture failed')
    inp, expected = _controlled_fixture(3, 'width')
    for array in (inp, expected):
        a, b = array[2:4, 9:11].copy(), array[2:4, 15:17].copy()
        array[2:4, 9:11], array[2:4, 15:17] = b, a
    _require(np.array_equal(solve(inp), expected), 'tray order affected encounter order')
    inp, expected = _controlled_fixture(2, 'width')
    _require(np.array_equal(solve(inp[:-2]), expected[:-2]), 'compact 11-row scene rejected')
    fingerprints, shapes, palettes, tokens, rows = set(), set(), set(), set(), []
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), 'nondeterminism')
        fingerprints.add(_fingerprint(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _parse(pair['input'])
                _, result = _run(info)
                shapes.add(pair['input'].shape)
                palettes.add((info['wall'], info['floor']))
                tokens.update(info['keys'])
                rows.append((split, info, result))
                # Global pre-unlocking is wrong on every narrow lesson: at
                # least one still-usable matching lock remains downstream.
                if result['stop'] == 'width':
                    _require(result['stranded'] > 0, 'width concept has no effect on lookup')
    _require(len(fingerprints) == 42 and len(shapes) >= 35, 'insufficient structural variation')
    _require(len(palettes) >= 15 and len(tokens) >= 30, 'palette or token variation collapsed')
    _require({i['k'] for _, i, _ in rows} == {2, 3, 4}, 'size coverage')
    _require({r['stop'] for s, _, r in rows if s == 'test'} == {'goal', 'width', 'lock'}, 'fixed test outcome')
    _require(any(r['repeated'] for s, _, r in rows if s == 'test'), 'repeated-lock test missing')
    _require(len({len(i['gates']) for _, i, _ in rows}) >= 4, 'lock count collapsed')
    _require(len({len(i['keys']) for _, i, _ in rows}) >= 3, 'key count collapsed')
    route_features = {_route_features(i) for _, i, _ in rows}
    _require(len(route_features) >= 45, 'route geometry collapsed')
    _require(max(t for _, t in route_features) >= 5 and any(t == 0 for _, t in route_features),
             'straight or winding routes missing')
    wide_motion = 0
    for _, info, result in rows:
        if result['success']:
            valid = _placements(info['all_floor'], info['k'])
            wide_motion += any(sum(bool(valid[q]) for q in _neighbors(tuple(p), valid.shape)) > 2
                               for p in np.argwhere(valid))
    _require(wide_motion >= 10, 'passable wide sections missing')
    baseline = generate_puzzle(0)
    # Whole-scene rotations/reflections must commute with the input-only rule.
    for o in range(8):
        for pair in baseline['train'] + baseline['test']:
            _require(np.array_equal(solve(_orient(pair['input'], o)), _orient(pair['output'], o)),
                     'scene symmetry changed rule')
    # Reject semantic damage, with valid shape and palette unchanged.
    for defect in ('ink', 'wall', 'square', 'duplicate'):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in baseline[s]] for s in baseline}
        pair = bad['train'][0]
        if defect == 'duplicate':
            bad['train'][1] = {k: v.copy() for k, v in pair.items()}
        else:
            info = _parse(pair['input'])
            out = np.rot90(pair['output'], info['rotation'])
            if defect == 'ink':
                token = next(iter(info['keys']))
                c = info['keys'][token]
                out[2, c] = 8
            elif defect == 'wall':
                out[0, 0] = 8
            else:
                p = tuple(np.argwhere(out == 2)[0])
                out[p] = info['floor']
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('validator accepted ' + defect)
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(123, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return {'seeds': 42, 'pairs': len(rows), 'grid_shapes': len(shapes),
            'wall_floor_assignments': len(palettes), 'glyph_tokens': len(tokens),
            'route_length_turn_profiles': len(route_features),
            'successful_examples_with_placement_freedom': wide_motion}
