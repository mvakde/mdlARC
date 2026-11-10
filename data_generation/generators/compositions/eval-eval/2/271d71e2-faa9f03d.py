"""Fuelled wire transfer: reconstruct, retract from the tip, and grow a ray.

Only NumPy and the standard library are used.  The solver reads one grid; the
sampler's latent geometry is used solely to construct and cross-check examples.
"""
from __future__ import annotations

import copy
import numpy as np

BG, FUEL, USED, REPAIR, DIRECTION = 0, 2, 8, 5, 4
PAYLOADS = (1, 3, 6, 7, 9)
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'invalid grid shape')
    _need(np.issubdtype(a.dtype, np.integer), 'grid must contain integers')
    _need(bool(np.all((0 <= a) & (a <= 9))), 'invalid color')
    return a.astype(np.int8, copy=True)


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _neighbors(p, cells):
    return [_add(p, d) for d in DIRS if _add(p, d) in cells]


def _components(cells):
    remaining, result = set(cells), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, found = [start], {start}
        while todo:
            for q in _neighbors(todo.pop(), remaining):
                remaining.remove(q)
                todo.append(q)
                found.add(q)
        result.append(found)
    return result


def _line(a, b):
    _need(a[0] == b[0] or a[1] == b[1], 'nonorthogonal line')
    d = (int(np.sign(b[0]-a[0])), int(np.sign(b[1]-a[1])))
    return [(a[0]+i*d[0], a[1]+i*d[1])
            for i in range(max(abs(b[0]-a[0]), abs(b[1]-a[1]))+1)]


def _polyline(points):
    cells = [points[0]]
    for a, b in zip(points, points[1:]):
        cells.extend(_line(a, b)[1:])
    return cells


def _ray(p, d, shape):
    result = []
    p = _add(p, d)
    while _inside(p, shape):
        result.append(p)
        p = _add(p, d)
    return result


def _turns(walk):
    ds = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(walk, walk[1:])]
    return sum(a != b for a, b in zip(ds, ds[1:]))


def _deleted_bends(v):
    walk = [v['pivot']]+v['old']
    removed = set(v['old'][-v['moved']:]) if v['moved'] else set()
    return sum(b in removed and (b[0]-a[0], b[1]-a[1]) != (c[0]-b[0], c[1]-b[1])
               for a, b, c in zip(walk, walk[1:], walk[2:]))


def _counter_order(bounds, direction):
    t, b, l, r = bounds
    if direction == (-1, 0):
        return [(y, x) for y in range(t+1, b) for x in range(l+1, r)]
    if direction == (1, 0):
        return [(y, x) for y in range(b-1, t, -1) for x in range(r-1, l, -1)]
    if direction == (0, -1):
        return [(y, x) for x in range(l+1, r) for y in range(b-1, t, -1)]
    return [(y, x) for x in range(r-1, l, -1) for y in range(t+1, b)]


def _analyze(input_grid):
    """Recover frames, repaired simple paths, pivots, and ordered old branches."""
    inp = _grid(input_grid)
    state_cells = {tuple(map(int, p)) for p in np.argwhere(np.isin(inp, (FUEL, USED)))}
    _need(bool(state_cells), 'missing counters')
    counters, reserved = [], set()
    for component in _components(state_cells):
        ys, xs = zip(*component)
        t, b, l, r = min(ys)-1, max(ys)+1, min(xs)-1, max(xs)+1
        _need(t >= 0 and l >= 0 and b < inp.shape[0] and r < inp.shape[1], 'open counter')
        interior = {(y, x) for y in range(t+1, b) for x in range(l+1, r)}
        block = {(y, x) for y in range(t, b+1) for x in range(l, r+1)}
        ring = block-interior
        _need(component == interior, 'counter interior must be a filled rectangle')
        colors = {int(inp[p]) for p in ring}
        _need(len(colors) == 1 and next(iter(colors)) in PAYLOADS, 'invalid frame')
        _need(not reserved & block, 'overlapping counters')
        reserved.update(block)
        counters.append(dict(bounds=(t, b, l, r), block=block, ring=ring,
                             color=next(iter(colors))))
    _need(len({v['color'] for v in counters}) == len(counters),
          'each device needs a distinct payload color within a grid')
    work = inp.copy()
    for p in reserved:
        work[p] = BG
    # Ordinary markers can be on endpoints, bends, or the directional pivot.
    pending = {tuple(map(int, p)) for p in np.argwhere(work == REPAIR)}
    while pending:
        changes = []
        for p in sorted(pending):
            colors = {int(work[q]) for d in DIRS
                      if _inside(q := _add(p, d), inp.shape) and int(work[q]) in PAYLOADS}
            if len(colors) == 1:
                changes.append((p, next(iter(colors))))
        _need(bool(changes), 'unresolved repair marker')
        for p, color in changes:
            work[p] = color
            pending.remove(p)
    # Unmarked omissions are isolated straight cells, never missing corners.
    while True:
        changes = []
        for y, x in np.argwhere(work == BG):
            p = int(y), int(x)
            if p in reserved:
                continue
            colors = set()
            for d in ((1, 0), (0, 1)):
                a, b = _add(p, d), _add(p, (-d[0], -d[1]))
                if _inside(a, inp.shape) and _inside(b, inp.shape):
                    if int(work[a]) == int(work[b]) and int(work[a]) in PAYLOADS:
                        colors.add(int(work[a]))
            _need(len(colors) <= 1, 'ambiguous crossing gap')
            if colors:
                changes.append((p, next(iter(colors))))
        if not changes:
            break
        for p, color in changes:
            work[p] = color
    all_path = {tuple(map(int, p)) for p in np.argwhere(np.isin(work, PAYLOADS))}
    all_markers = {tuple(map(int, p)) for p in np.argwhere(inp == DIRECTION)}
    assigned, assigned_markers, devices = set(), set(), []
    for counter in counters:
        color = counter['color']
        cells = {p for p in all_path if int(work[p]) == color}
        attachments = {q for p in counter['ring'] for q in _neighbors(p, cells)}
        _need(len(attachments) == 1, 'counter must have one external wire port')
        start = next(iter(attachments))
        component = next(c for c in _components(cells) if start in c)
        _need(not component & assigned, 'wire shared by counters')
        assigned.update(component)
        _need(len(_neighbors(start, component)) == 1, 'wire does not start at its counter')
        walk, previous, current = [], None, start
        while True:
            _need(current not in walk, 'cyclic wire')
            walk.append(current)
            following = [q for q in _neighbors(current, component) if q != previous]
            if not following:
                break
            _need(len(following) == 1, 'wire must not branch')
            previous, current = current, following[0]
        _need(set(walk) == component, 'unexplained wire geometry')
        _need(current[0] in (0, inp.shape[0]-1) or current[1] in (0, inp.shape[1]-1),
              'old terminal tip must reach a boundary')
        candidates = []
        positions = {p: i for i, p in enumerate(walk)}
        for q in sorted(all_markers):
            for d in DIRS:
                p = q[0]-d[0], q[1]-d[1]
                i = positions.get(p, -1)
                if not 0 < i < len(walk)-1:
                    continue
                if walk[i+1] != (p[0]-d[0], p[1]-d[1]):
                    continue
                incoming = walk[i-1][0]-p[0], walk[i-1][1]-p[1]
                if incoming[0]*d[0] + incoming[1]*d[1] != 0:
                    continue
                ray = _ray(p, d, inp.shape)
                if (set(ray) & (all_path | reserved)) or (set(ray) & all_markers != {q}):
                    continue
                candidates.append((q, d, i, ray))
        _need(len(candidates) == 1, 'direction marker has no unique pivot and old branch')
        q, d, index, ray = candidates[0]
        _need(q not in assigned_markers, 'shared direction marker')
        assigned_markers.add(q)
        order = _counter_order(counter['bounds'], d)
        values = [int(inp[p]) for p in order]
        advanced = values.count(USED)
        _need(values == [USED]*advanced + [FUEL]*(len(values)-advanced), 'invalid unary prefix')
        old = walk[index+1:]
        fuel = len(values)-advanced
        moved = min(fuel, len(old), len(ray))
        devices.append(dict(counter, walk=walk, path=component, pivot=walk[index], marker=q,
                            direction=d, old=old, ray=ray, order=order, advanced=advanced,
                            fuel=fuel, moved=moved, old_bends=_turns([walk[index]]+old)))
    _need(assigned == all_path and assigned_markers == all_markers, 'unassigned trace or marker')
    _need(len(devices) >= 1, 'no devices')
    for i, a in enumerate(devices):
        for b in devices[i+1:]:
            _need(not set(a['ray']) & set(b['ray']), 'new rays may not cross')
    explained = reserved | all_path | all_markers
    _need(all(tuple(map(int, p)) in explained for p in np.argwhere(inp != BG)), 'unexplained color')
    return inp, work, devices


def _render_solution(inp, repaired, devices):
    out = np.full(inp.shape, BG, dtype=np.int8)
    for v in devices:
        for p in v['block']:
            out[p] = inp[p]
        k = v['moved']
        removed = set(v['old'][-k:]) if k else set()
        for p in v['path']-removed:
            out[p] = v['color']
        for p in v['ray'][:k]:
            out[p] = v['color']
        for j, p in enumerate(v['order']):
            out[p] = USED if j < v['advanced']+k else FUEL
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from input alone; color conventions are fixed and documented."""
    return _render_solution(*_analyze(input_grid))


def _rotate_cell(p, shape, k):
    y, x = p
    h, w = shape
    for _ in range(k):
        y, x, h, w = w-1-x, y, w, h
    return y, x


def _simple(walk):
    cells = set(walk)
    return len(cells) == len(walk) and all(len(_neighbors(p, cells)) ==
           (1 if i in (0, len(walk)-1) else 2) for i, p in enumerate(walk))


def _candidate(rng, shape, color, bends, lesson, direction):
    """A randomly spaced, potentially backtracking old branch and a counter trunk.

    Construct facing right, then rotate coordinates.  Decreasing old-branch
    columns prevent crossings while arbitrary vertical reversals vary arclength.
    No invisible slots or output labels are used by the parser.
    """
    rotation = ((0, 1), (-1, 0), (0, -1), (1, 0)).index(direction)
    canonical_shape = shape if rotation % 2 == 0 else shape[::-1]
    h, w = canonical_shape
    min_c = max(4, 3*(bends//2+1))
    if min_c >= w-2:
        return None
    r = int(rng.integers(2, h-2))
    c = int(rng.integers(min_c, w-2))
    pivot = (r, c)
    points = [pivot]
    for segment in range(bends+1):
        y, x = points[-1]
        last = segment == bends
        if segment % 2 == 0:
            if last:
                q = y, 0
            else:
                future_horizontal = (bends-segment)//2
                low = max(1, 3*future_horizontal)
                if x-2 <= low:
                    return None
                q = y, int(rng.integers(low, x-2))
        elif last:
            q = (0 if rng.random() < .5 else h-1), x
        else:
            choices = [yy for yy in range(1, h-1) if abs(yy-y) >= 3]
            q = int(rng.choice(choices)), x
        points.append(q)
    old_walk = _polyline(points)
    if not _simple(old_walk):
        return None
    ih = int(rng.integers(2 if lesson else 1, 7))
    iw = int(rng.integers(2 if lesson else 1, 7))
    oh, ow = ih+2, iw+2
    side = int(rng.choice((-1, 1)))
    length = int(rng.integers(2, 6))
    # A straight trunk or a dogleg shifts the counter independently of the pivot.
    offset = int(rng.integers(3, 7)) if rng.random() < .4 else 0
    if offset:
        length = max(length, 4)
    port_col = c+offset
    border_row = r+side*(length+1)
    t = border_row if side == 1 else border_row-oh+1
    l = port_col-int(rng.integers(1, ow-1))
    b, rr = t+oh-1, l+ow-1
    if t < 0 or l < 0 or b >= h or rr >= w:
        return None
    if offset:
        turn_row = r+3*side
        trunk_points = [(border_row-side, port_col), (turn_row, port_col), (turn_row, c), pivot]
    else:
        trunk_points = [(border_row-side, c), pivot]
    trunk = _polyline(trunk_points)
    walk = trunk+old_walk[1:]
    if not _simple(walk):
        return None
    block = {(y, x) for y in range(t, b+1) for x in range(l, rr+1)}
    if block & set(walk):
        return None
    contacts = {p for p in walk if _neighbors(p, block)}
    if contacts != {walk[0]}:
        return None
    ray = _ray(pivot, (0, 1), canonical_shape)
    if set(ray) & (set(walk) | block):
        return None
    capacity, limit = ih*iw, min(len(old_walk)-1, len(ray))
    if lesson == 'fuel':
        # Retract through the final bend while retaining an initial old stub.
        tail_length = len(_line(points[-2], points[-1]))-1
        low, high = tail_length+1, min(limit-1, capacity-1)
        if low > high:
            return None
        fuel = int(rng.integers(low, high+1))
        advanced = capacity-fuel
    elif lesson in ('edge', 'old'):
        if lesson == 'edge' and not len(ray) < len(old_walk)-1:
            return None
        if lesson == 'old' and not len(old_walk)-1 < len(ray):
            return None
        # Leave some fuel, cross a counter row, and stop inside a counter row.
        choices = [a for a in range(1, capacity-limit)
                   if a//ih < (a+limit)//ih and (a+limit) % ih != 0]
        if not choices:
            return None
        advanced = int(rng.choice(choices))
    elif lesson == 'winding':
        # The query's extra bends must affect the transfer, rather than merely
        # decorate an untouched old arm.  Reach and remove two old bends.
        past_two_bends = (len(_line(points[-3], points[-2]))-1
                          + len(_line(points[-2], points[-1]))-1 + 1)
        if min(capacity, limit) < past_two_bends:
            return None
        fuel = int(rng.integers(past_two_bends, capacity+1))
        advanced = capacity-fuel
    else:
        mode = int(rng.integers(5))
        if mode == 0:
            advanced = 0  # entirely fresh
        elif mode == 1:
            advanced = capacity  # exhausted, but repairs still take effect
        elif mode == 2 and limit <= capacity:
            advanced = capacity-limit
        else:
            advanced = int(rng.integers(capacity+1))
    transform = lambda p: _rotate_cell(p, canonical_shape, rotation)
    global_block = {transform(p) for p in block}
    ys, xs = zip(*global_block)
    bounds = min(ys), max(ys), min(xs), max(xs)
    return dict(color=color, bounds=bounds, block=global_block,
                walk=[transform(p) for p in walk], pivot=transform(pivot),
                old=[transform(p) for p in old_walk[1:]], ray=[transform(p) for p in ray],
                direction=direction, advanced=advanced)


def _make_pair(rng, lessons, split, desired_count, first_direction):
    # Random whole-canvas packing, with complete swept rays reserved.
    for scene_attempt in range(40):
        shape = (int(rng.integers(16, 31)), int(rng.integers(16, 31)))
        if desired_count > 1:
            shape = tuple(max(n, 24) for n in shape)
        colors = list(map(int, rng.choice(PAYLOADS, desired_count, replace=False)))
        devices, occupied = [], set()
        for j, color in enumerate(colors):
            found = None
            lesson = lessons[j] if j < len(lessons) else None
            for _ in range(250):
                bends = int(rng.integers(2, 5)) if split == 'test' else int(rng.integers(0, 2))
                if lesson and lesson != 'winding':
                    bends = 1
                direction = first_direction if j == 0 else DIRS[int(rng.integers(4))]
                v = _candidate(rng, shape, color, bends, lesson, direction)
                if v is None:
                    continue
                footprint = v['block'] | set(v['walk']) | set(v['ray'])
                if footprint & occupied or any(_neighbors(p, occupied) for p in footprint):
                    continue
                found = v
                occupied.update(footprint)
                devices.append(v)
                break
            if found is None:
                break
        if len(devices) != desired_count:
            continue
        inp = np.full(shape, BG, dtype=np.int8)
        expected = np.full(shape, BG, dtype=np.int8)
        for v in devices:
            for p in v['block'] | set(v['walk']):
                inp[p] = v['color']
            order = _counter_order(v['bounds'], v['direction'])
            k = min(len(order)-v['advanced'], len(v['old']), len(v['ray']))
            removed = set(v['old'][-k:]) if k else set()
            for p in v['block'] | (set(v['walk'])-removed) | set(v['ray'][:k]):
                expected[p] = v['color']
            for i, p in enumerate(order):
                inp[p] = USED if i < v['advanced'] else FUEL
                expected[p] = USED if i < v['advanced']+k else FUEL
            inp[v['ray'][0]] = DIRECTION
            # The pivot repair always survives, so repair is observable even
            # when the whole old branch is transferred.
            inp[v['pivot']] = REPAIR
            candidates = [p for p in v['walk'] if p != v['walk'][0] and p != v['pivot']]
            rng.shuffle(candidates)
            marked = {v['pivot']}
            for p in candidates[:int(rng.integers(1, 5))]:
                if not _neighbors(p, marked):
                    inp[p] = REPAIR
                    marked.add(p)
            gaps = [p for i, p in enumerate(v['walk'][1:-1], 1)
                    if v['walk'][i-1][0]+v['walk'][i+1][0] == 2*p[0]
                    and v['walk'][i-1][1]+v['walk'][i+1][1] == 2*p[1]
                    and p not in marked and not _neighbors(p, marked)]
            rng.shuffle(gaps)
            selected_gaps = set()
            for p in gaps[:int(rng.integers(0, 4))]:
                if not _neighbors(p, selected_gaps):
                    inp[p] = BG
                    selected_gaps.add(p)
        try:
            parsed_inp, repaired, parsed = _analyze(inp)
            _need(len(parsed) == len(devices), 'wrong device recovery')
            for v, pv in zip(sorted(devices, key=lambda v: v['bounds']),
                             sorted(parsed, key=lambda v: v['bounds'])):
                _need(pv['walk'] == v['walk'], 'ambiguous path completion')
                _need(pv['old'] == v['old'], 'wrong old branch')
            _need(np.array_equal(_render_solution(parsed_inp, repaired, parsed), expected),
                  'constructed label disagrees with input reconstruction')
            for v in parsed:
                mask = expected == v['color']
                _need(not np.any(mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]),
                      'transfer merged thin strokes into a thick block')
        except AssertionError:
            continue
        return {'input': inp, 'output': expected}
    raise RuntimeError('bounded scene construction failed')


def _counts(num_train, num_test):
    for name, value in (('num_train', num_train), ('num_test', num_test)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(name+' must be an integer')
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError('require 2..10 train, 1..4 test, and num_train > num_test')


def _signature(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    directions = [DIRS[int(i)] for i in rng.permutation(4)]
    anchors = ['fuel', 'edge', 'old']
    rng.shuffle(anchors)
    lessons = [anchors[:2], anchors[2:]] + [[] for _ in range(num_train-2)]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            lesson = lessons[i] if split == 'train' else ['winding']
            desired = len(lesson) if split == 'train' and lesson else int(rng.choice((1, 2), p=(.65, .35)))
            direction = directions[i % 4] if split == 'train' else DIRS[int(rng.integers(4))]
            for attempt in range(12):
                pair = _make_pair(rng, lesson, split, desired, direction)
                key = _signature(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('could not construct distinct inputs')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle schema')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(inp.shape == out.shape, 'dimensions changed')
            key = _signature(inp)
            _need(key not in seen, 'duplicate input')
            seen.add(key)
            _, repaired, vs = _analyze(inp)
            _need(np.array_equal(_render_solution(inp, repaired, vs), out), 'incorrect output')
            _need(not np.any(np.isin(out, (REPAIR, DIRECTION))), 'annotation survived')
            for v in vs:
                mask = out == v['color']
                _need(not np.any(mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]),
                      'output wire has a thick block')
                k = v['moved']
                # Independent invariant: a transfer conserves physical wire
                # cells, and spends exactly the same number of counter cells.
                outside = np.ones(inp.shape, dtype=bool)
                for p in v['block']:
                    outside[p] = False
                original_count = len(v['path'])
                _need(int(np.count_nonzero((out == v['color']) & outside)) == original_count,
                      'wire length was not conserved')
                _need(sum(out[p] == USED for p in v['order']) == v['advanced']+k,
                      'counter expenditure differs from transfer')
                if split == 'train':
                    training.append(v)
            if split == 'test':
                _need(any(_deleted_bends(v) >= 2 for v in vs),
                      'query does not exercise transfer through extra bends')
    _need(any(0 < v['moved'] == v['fuel'] < min(len(v['old']), len(v['ray']))
              and _deleted_bends(v) >= 1 for v in training), 'missing fuel-limited lesson')
    _need(any(0 < v['moved'] == len(v['ray']) < min(v['fuel'], len(v['old']))
              for v in training), 'missing boundary-limited lesson')
    _need(any(0 < v['moved'] == len(v['old']) < min(v['fuel'], len(v['ray']))
              for v in training), 'missing old-wire-limited lesson')
    _need(len({v['direction'] for v in training if v['moved']}) >= 2, 'rotation not demonstrated')


def self_test() -> dict:
    """Bounded 42-seed checks, ablations, rotation tests, and corrupt-label tests."""
    puzzles, shapes, structures, features = [], set(), set(), []
    for seed in range(42):
        p = generate_puzzle(seed)
        puzzles.append(p)
        for split in ('train', 'test'):
            for pair in p[split]:
                inp, out = pair['input'], pair['output']
                _, repaired, vs = _analyze(inp)
                shapes.add(inp.shape)
                roles = {}
                normalized = tuple(roles.setdefault(int(c), len(roles)) for c in inp.flat)
                structures.add((inp.shape, normalized))
                _need(np.array_equal(solve(inp.copy()), out), 'solver mismatch')
                # The rule rotates with the marker and the counter ordering.
                _need(np.array_equal(solve(np.rot90(inp)), np.rot90(out)), 'rotation mismatch')
                for v in vs:
                    features.append((split, v['direction'], v['old_bends'], len(v['old']),
                                     len(v['order']), v['fuel'], v['moved'], len(vs)))
        # Omitting A gives an unlimited-fuel transfer, observably wrong in train.
        no_a_differs, no_b_differs = False, False
        for pair in p['train']:
            inp, repaired, vs = _analyze(pair['input'])
            unlimited = copy.deepcopy(vs)
            for v in unlimited:
                v['moved'] = min(len(v['old']), len(v['ray']))
            no_a_differs |= not np.array_equal(_render_solution(inp, repaired, unlimited), pair['output'])
            # Whole-branch deletion discards wire that the budget cannot move.
            wrong = pair['output'].copy()
            for v in vs:
                for point in v['old']:
                    wrong[point] = BG
            no_b_differs |= not np.array_equal(wrong, pair['output'])
        _need(no_a_differs and no_b_differs, 'one parent became dispensable')
    _need(len(shapes) >= 40 and len(structures) >= 200, 'structural diversity collapsed')
    _need({f[1] for f in features} == set(DIRS), 'missing orientations')
    _need({f[2] for f in features if f[0] == 'test'} >= {2, 3, 4}, 'missing winding queries')
    _need(any(f[5] == 0 for f in features) and any(f[5] == f[4] for f in features),
          'missing full/empty counter range')
    _need({f[7] for f in features} >= {1, 2}, 'missing device count variation')
    _need(any(f[3] > 20 for f in features), 'old branch lengths collapsed')
    _need(any(f[6] >= 10 for f in features), 'missing long transfers')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000+10*nt+nq, nt, nq)
            _need((len(p['train']), len(p['test'])) == (nt, nq), 'counts ignored')
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    repeated = generate_puzzle(17)
    _need(all(_signature(a[k]) == _signature(b[k])
              for s in repeated for a, b in zip(repeated[s], puzzles[17][s])
              for k in ('input', 'output')), 'nondeterminism')
    for p in puzzles[:3]:
        broken = copy.deepcopy(p)
        out = broken['test'][0]['output']
        out[0, 0] = (int(out[0, 0])+1) % 10
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError('wrong output was accepted')
    return dict(seeds=42, pairs=sum(len(p[s]) for p in puzzles for s in p),
                shapes=len(shapes), color_normalized_inputs=len(structures),
                directions=4, device_counts=sorted({f[7] for f in features}),
                old_bends=sorted({f[2] for f in features}),
                max_old_length=max(f[3] for f in features), max_transfer=max(f[6] for f in features))
