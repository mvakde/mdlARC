"""Self-assembling slalom: gate projection and unary routing in one cascade.

Only Python's standard library and NumPy are used. Solving decodes the input;
construction records are never passed to solve or the puzzle validator.
"""
from __future__ import annotations

import numpy as np

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "bad grid colors")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _puzzle_key(p):
    return tuple(tuple((_key(x['input']), _key(x['output'])) for x in p[s])
                 for s in ('train', 'test'))


def _to_forward(a, direction):
    # Coordinates are (forward, lateral); the seed is at forward coordinate 0.
    return (a, a.T, a[::-1], a.T[::-1])[direction].copy()


def _from_forward(a, direction):
    return (a, a.T, a[::-1], a[::-1].T)[direction].copy()


def _components(a, bg):
    remaining = {tuple(map(int, p)) for p in np.argwhere(a != bg)}
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        points, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    points.add(q)
                    stack.append(q)
        result.append(points)
    return result


def _read_input(value):
    grid = _grid(value)
    colors, counts = np.unique(grid, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    _require(int(counts.max()) > grid.size / 2, "background must occupy a majority")
    components = _components(grid, bg)
    h, w = grid.shape
    seeds = [next(iter(p)) for p in components if len(p) == 1 and
             any((r in (0, h - 1) or c in (0, w - 1)) for r, c in p)]
    _require(len(seeds) == 1, "expected one isolated boundary seed")
    r, c = seeds[0]
    hits = [r == 0, c == 0, r == h - 1, c == w - 1]
    _require(sum(hits) == 1, "corner seed has no unique direction")
    direction = hits.index(True)
    route = int(grid[r, c])
    a = _to_forward(grid, direction)
    gates = []
    for points in _components(a, bg):
        if len(points) == 1 and next(iter(points))[0] == 0:
            continue
        marks = [p for p in points if int(a[p]) == route]
        _require(len(marks) == 1, "gate must have one seed-colored registration")
        f = min(p[0] for p in points)
        end = max(p[0] for p in points)
        lo, hi = min(p[1] for p in points), max(p[1] for p in points)
        _require(0 < f <= end < a.shape[0] - 1, "gate touches a forward edge")
        _require(marks[0][0] == f and lo < marks[0][1] < hi,
                 "registration must be inside the front bar")
        _require(all((f, col) in points for col in range(lo, hi + 1)),
                 "front bar must span the entire lateral envelope")
        front = a[f, lo:hi + 1]
        roles = set(map(int, front)) - {route}
        _require(len(roles) == 2, "front must contain material, instructions, registration")
        candidates = []
        for marker in sorted(roles):
            body = next(iter(roles - {marker}))
            indices = tuple(map(int, np.flatnonzero(front == marker)))
            n = len(indices)
            if not 1 <= n <= 4 or len(front) < 2 * n + 3:
                continue
            low = indices == tuple(2 * j for j in range(n))
            high = indices == tuple(sorted(len(front) - 1 - 2 * j for j in range(n)))
            if low == high:
                continue
            if not any(front[j] == body == front[j + 1] for j in range(len(front) - 1)):
                continue
            if any(int(a[p]) in (marker, route) for p in points if p[0] != f):
                continue
            candidates.append((marker, body, n, 1 if low else -1))
        _require(len(candidates) == 1, "front roles or endpoint code are ambiguous")
        marker, body, n, side = candidates[0]
        gates.append({'points': points, 'f': f, 'end': end, 'lo': lo, 'hi': hi,
                      'reg': marks[0][1], 'marker': marker, 'body': body,
                      'n': n, 'side': side})
    gates.sort(key=lambda g: g['f'])
    _require(gates, "no gates")
    _require(all(x['end'] < y['f'] for x, y in zip(gates, gates[1:])),
             "gates must have disjoint forward bands")
    _require(len({g['marker'] for g in gates}) == 1, "one instruction color per input")
    initial = int(np.argwhere(a[0] == route)[0, 0])
    return grid, a, bg, route, direction, initial, gates


def _assemble(a, bg, route, initial, gates, mode='correct'):
    """Reconstruct the cascade; modes are only explicit shortcut diagnostics."""
    height, width = a.shape
    out = np.full_like(a, bg)
    path = np.zeros(a.shape, dtype=bool)
    gate_cells, envelope = set(), set()
    lane, start, previous_end = initial, 0, -1
    records = []
    for i, g in enumerate(gates):
        guide = initial if mode == 'static_guide' else lane
        shift = 0 if mode == 'no_alignment' else guide - g['reg']
        lo, hi = g['lo'] + shift, g['hi'] + shift
        n = 1 if mode == 'fixed_clearance' else g['n']
        d = n + 1
        side = g['side']
        if mode == 'alternate' and i:
            side = -records[-1]['side']
        if mode == 'wrong_end':
            side = -side
        target = hi + d if side == 1 else lo - d
        turn = g['f'] - d
        _require(0 <= lo <= hi < width and 0 <= target < width, "cascade leaves canvas")
        _require(turn >= 1 and turn > previous_end and turn >= start,
                 "turn must follow the preceding envelope")
        _require(lo <= lane <= hi, "aligned gate must intercept incoming line")
        for r, c in g['points']:
            q = r, c + shift
            _require(q not in gate_cells, "target gates overlap")
            value = int(a[r, c])
            out[q] = g['body'] if value == g['marker'] else value
            gate_cells.add(q)
        envelope.update((r, c) for r in range(g['f'], g['end'] + 1)
                        for c in range(lo, hi + 1))
        path[start:turn + 1, lane] = True
        path[turn, min(lane, target):max(lane, target) + 1] = True
        records.append(dict(g, lane=lane, shift=shift, target=target, turn=turn,
                            target_lo=lo, target_hi=hi, side=side))
        lane, start, previous_end = target, turn, g['end']
    path[start:, lane] = True
    _require(not any(path[p] for p in envelope), "route crosses a gate envelope")
    out[path] = route
    return out, path, records


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all roles, directions, translations, and route cells from this input."""
    _, a, bg, route, direction, initial, gates = _read_input(input_grid)
    out, _, _ = _assemble(a, bg, route, initial, gates)
    return _from_forward(out, direction)


def _shortcut_changes(value, mode):
    _, a, bg, route, _, initial, gates = _read_input(value)
    expected = _assemble(a, bg, route, initial, gates)[0]
    try:
        other = _assemble(a, bg, route, initial, gates, mode)[0]
    except AssertionError:
        return True  # The shortcut cannot even construct a valid answer.
    return not np.array_equal(expected, other)


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return int(value)


def _transition_options(width, lane, n, sides):
    options = []
    for side in sides:
        for length in range(2 * n + 3, min(width, 17) + 1):
            indices = {2 * j if side == 1 else length - 1 - 2 * j for j in range(n)}
            for k in range(1, length - 1):
                if k in indices:
                    continue
                # An adjacent material pair visibly distinguishes material from unary markers.
                body = set(range(length)) - indices - {k}
                if not any(j + 1 in body for j in body):
                    continue
                lo, hi = lane - k, lane + length - 1 - k
                target = hi + n + 1 if side == 1 else lo - n - 1
                if 0 <= lo <= hi < width and 1 <= target < width - 1:
                    options.append((length, k, side, lo, target))
    return options


def _chain(rng, width, counts, repeated, eccentric=False, interior=False):
    failed = set()
    repeat_side = int(rng.choice((-1, 1)))

    def search(index, lane):
        if index == len(counts):
            return []
        state = index, lane
        if state in failed:
            return None
        sides = (repeat_side,) if repeated and index < 2 else (-1, 1)
        options = _transition_options(width, lane, counts[index], sides)
        if eccentric and index == 0:
            options = [v for v in options if v[1] not in ((v[0] - 1) // 2, v[0] // 2)]
        if interior:
            options = [v for v in options if v[3] >= 1 and v[3] + v[0] - 1 < width - 1]
        for j in rng.permutation(len(options)):
            option = options[int(j)]
            rest = search(index + 1, option[4])
            if rest is not None:
                return [option] + rest
        failed.add(state)
        return None

    for lane in rng.permutation(np.arange(1, width - 1)):
        plan = search(0, int(lane))
        if plan is not None:
            return int(lane), plan
    return None


def _shape(rng, length, depth):
    """A full front spine, with flat/solid/frame/comb/stepped rear geometry."""
    points = {(0, c) for c in range(length)}
    if depth == 0:
        return points
    kind = int(rng.integers(4))
    if kind == 0:
        points.update((r, c) for r in range(1, depth + 1) for c in range(length))
    elif kind == 1:
        points.update((r, c) for r in range(1, depth + 1) for c in (0, length - 1))
        points.update((depth, c) for c in range(length))
    elif kind == 2:
        teeth = rng.choice(length, size=int(rng.integers(1, min(length, 5) + 1)), replace=False)
        for i, c in enumerate(teeth):
            reach = depth if i == 0 else int(rng.integers(1, depth + 1))
            points.update((r, int(c)) for r in range(1, reach + 1))
    else:
        left, right = 0, length - 1
        for r in range(1, depth + 1):
            if right - left > 1:
                if rng.random() < .5:
                    left += 1
                else:
                    right -= 1
            points.update((r, c) for c in range(left, right + 1))
    return points


def _make_pair(rng, counts, direction, palette, repeated=False, force_depth=False,
               no_alignment_witness=False):
    bg, route, marker = palette[:3]
    # Narrow proposals widen only when the whole cascade is infeasible.
    proposed = int(rng.integers(12, 31))
    for width in dict.fromkeys((proposed, min(30, proposed + 5), 30)):
        answer = _chain(rng, width, counts, repeated, eccentric=force_depth,
                        interior=no_alignment_witness)
        if answer is not None:
            break
    else:
        raise AssertionError('no feasible chain even at width 30')
    initial, transitions = answer
    depths = list(map(int, rng.integers(0, 4, size=len(counts))))
    if force_depth:
        depths[0] = max(2, depths[0])
    # Count sequences always fit at depth zero; reduce optional depth to fit.
    minimum = sum(n + 2 + depth for n, depth in zip(counts, depths)) + 2
    while minimum > 30:
        candidates = [i for i, d in enumerate(depths) if d > (2 if force_depth and i == 0 else 0)]
        _require(candidates, 'required demonstration exceeds bounds')
        i = int(rng.choice(candidates))
        depths[i] -= 1
        minimum -= 1
    height = int(rng.integers(minimum, 31))
    slack = rng.multinomial(height - minimum, np.full(len(counts) + 1, 1 / (len(counts) + 1)))
    inp = np.full((height, width), bg, dtype=np.int8)
    out = np.full_like(inp, bg)
    inp[0, initial] = route
    lane, start, previous_end = initial, 0, -1
    common_offset = int(rng.choice((-1, 1))) if no_alignment_witness else None
    for i, (n, depth, transition) in enumerate(zip(counts, depths, transitions)):
        length, k, side, target_lo, target = transition
        turn = (1 if i == 0 else previous_end + 1) + int(slack[i])
        f = turn + n + 1
        shape = _shape(rng, length, depth)
        body = int(rng.choice(palette[3:]))
        cells = {p: body for p in shape}
        rear = sorted(p for p in shape if p[0] > 0)
        if rear and rng.random() < .65:
            decoration = int(rng.choice([v for v in palette[3:] if v != body]))
            for j in rng.choice(len(rear), size=int(rng.integers(1, min(3, len(rear)) + 1)), replace=False):
                cells[rear[int(j)]] = decoration
        for j in range(n):
            cells[0, 2 * j if side == 1 else length - 1 - 2 * j] = marker
        cells[0, k] = route
        origins = [v for v in range(width - length + 1) if v != target_lo]
        _require(origins, 'no displaced source location')
        raw_lo = target_lo + common_offset if no_alignment_witness else int(rng.choice(origins))
        for (r, c), value in sorted(cells.items()):
            inp[f + r, raw_lo + c] = value
            out[f + r, target_lo + c] = body if value == marker else value
        out[start:turn + 1, lane] = route
        out[turn, min(lane, target):max(lane, target) + 1] = route
        lane, start, previous_end = target, turn, f + depth
    out[start:, lane] = route
    if np.count_nonzero(inp == bg) <= inp.size / 2:
        return None
    return {'input': _from_forward(inp, direction), 'output': _from_forward(out, direction)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    palette = tuple(map(int, rng.permutation(10)))
    stable = bool(rng.random() < .35)
    maximum = int(rng.integers(2, 5))
    base = int(rng.integers(4))
    # Training includes perpendicular directions; tests reserve the fourth direction.
    train_dirs = [base, (base + 1) % 4, (base + 2) % 4]
    test_direction = (base + 3) % 4
    # Random slots, then shuffled presentation; no fixed first-example template.
    plans = []
    demo_counts = [1, maximum]
    rng.shuffle(demo_counts)
    other_counts = list(range(2, maximum)) or [int(rng.integers(1, maximum + 1))]
    rng.shuffle(other_counts)
    plans.append((demo_counts, train_dirs[0], True, True, False))
    plans.append((other_counts, train_dirs[1], False, True, True))
    for _ in range(num_train - 2):
        count = int(rng.integers(1, 5))
        ns = list(map(int, rng.integers(1, maximum + 1, size=count)))
        plans.append((ns, int(rng.choice(train_dirs)), False, False, False))
    for _ in range(num_test):
        count = int(rng.integers(1, 5))
        ns = list(map(int, rng.integers(1, maximum + 1, size=count)))
        plans.append((ns, test_direction, False, False, False))
    puzzle, seen = {'train': [], 'test': []}, set()
    for i, (ns, direction, repeated, depth, no_alignment_witness) in enumerate(plans):
        split = 'train' if i < num_train else 'test'
        for _ in range(32):
            local_palette = palette if stable else tuple(map(int, rng.permutation(10)))
            pair = _make_pair(rng, ns, direction, local_palette, repeated, depth,
                              no_alignment_witness)
            if pair is None:
                continue
            key = _key(pair['input'])
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise AssertionError('unique pair sampling exhausted')
    # Teach both endpoint meanings even if all sampled commands happened to agree.
    sides = {g['side'] for p in puzzle['train'] for g in _read_input(p['input'])[-1]}
    if len(sides) == 1:
        pair = puzzle['train'][1]
        direction = _read_input(pair['input'])[4]
        for role in ('input', 'output'):
            pair[role] = _from_forward(_to_forward(pair[role], direction)[:, ::-1], direction)
    for split in puzzle:
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'bad puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'bad example counts')
    seen, stats = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'bad pair schema')
            inp, supplied = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == supplied.shape, 'canvas must be preserved')
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            _, a, bg, route, direction, initial, gates = _read_input(inp)
            expected, path, records = _assemble(a, bg, route, initial, gates)
            b = _to_forward(supplied, direction)
            _require(np.array_equal(b, expected), 'output disagrees with the cascade')
            _require(b[0, initial] == route, 'seed changed')
            _require(not np.array_equal(inp, supplied), 'identity pair')
            _require(np.count_nonzero(path[0]) == 1 and np.count_nonzero(path[-1]) == 1,
                     'route must enter and exit once')
            # Check objectwise invariants, including cells in holes/old footprints.
            for g in records:
                _require(g['reg'] + g['shift'] == g['lane'], 'registration did not meet guide')
                for r, c in g['points']:
                    value = g['body'] if a[r, c] == g['marker'] else a[r, c]
                    _require(b[r, c + g['shift']] == value, 'rigid payload or local normalization changed')
                _require(g['target'] - (g['target_hi'] if g['side'] == 1 else g['target_lo'])
                         == g['side'] * (g['n'] + 1), 'wrong lateral clearance')
                _require(g['f'] - g['turn'] == g['n'] + 1, 'wrong approach clearance')
            stats[split].append((direction, records))
    train_records = [g for _, gs in stats['train'] for g in gs]
    test_records = [g for _, gs in stats['test'] for g in gs]
    _require({1, 2} <= {g['n'] for g in train_records}, 'need distinct unary demonstrations')
    _require({g['n'] for g in test_records} <= {g['n'] for g in train_records}, 'unseen clearance count')
    _require({g['side'] for g in train_records} == {-1, 1}, 'both endpoint meanings must be taught')
    _require(any(x['side'] == y['side'] for _, gs in stats['train'] for x, y in zip(gs, gs[1:])),
             'need a repeated-side cascade demonstration')
    _require(any(g['end'] > g['f'] and g['shift'] != 0 for g in train_records),
             'need complete two-dimensional rigid movement')
    _require(any(g['end'] > g['f'] and g['shift'] != 0 and
                 g['reg'] - g['lo'] not in ((g['hi'] - g['lo']) // 2,
                                           (g['hi'] - g['lo'] + 1) // 2)
                 for g in train_records), 'need an eccentric rigid movement demonstration')
    for mode in ('no_alignment', 'static_guide', 'fixed_clearance', 'alternate', 'wrong_end'):
        _require(any(_shortcut_changes(p['input'], mode) for p in puzzle['train']),
                 f'training fails to distinguish {mode}')
    # In at least one demonstration, omitting A yields a legal, different B-style
    # route rather than merely violating the input geometry of the shortcut.
    valid_no_alignment = False
    for pair in puzzle['train']:
        _, a, bg, route, _, initial, gates = _read_input(pair['input'])
        try:
            other = _assemble(a, bg, route, initial, gates, 'no_alignment')[0]
        except AssertionError:
            continue
        if not np.array_equal(other, _assemble(a, bg, route, initial, gates)[0]):
            valid_no_alignment = True
            break
    _require(valid_no_alignment, 'need a valid answer-changing no-alignment example')


def _fixture():
    """Hand-specified 19x24 cascade with a frame and an asymmetric comb."""
    a = np.zeros((19, 24), dtype=np.int8)
    a[0, 6] = 1
    first = {(0, c): 2 for c in range(7)}
    first.update({(r, c): 2 for r in (1, 2) for c in (0, 6)})
    first.update({(2, c): 2 for c in range(7)})
    first[0, 0], first[0, 3], first[1, 6] = 3, 1, 4
    second = {(0, c): 2 for c in range(9)}
    second.update({(1, c): 2 for c in (1, 4, 7)})
    second.update({(2, c): 2 for c in (1, 7)})
    second[0, 8], second[0, 6], second[0, 3], second[1, 4] = 3, 3, 1, 4
    expected = np.zeros_like(a)
    for f, raw, target, cells in ((5, 8, 3, first), (13, 2, 8, second)):
        for (r, c), value in cells.items():
            a[f + r, raw + c] = value
            expected[f + r, target + c] = 2 if value == 3 else value
    expected[0:4, 6] = 1
    expected[3, 6:12] = 1
    expected[3:11, 11] = 1
    expected[10, 5:12] = 1
    expected[10:19, 5] = 1
    return a, expected


def self_test():
    """Bounded 42-seed variation, regression, metamorphic and negative checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, 'seed diversity')
    _require(_puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0)), 'determinism')
    shapes, palettes, geometries, directions, counts, signs, depths, gate_counts = (set() for _ in range(8))
    shifts = set()
    for p in puzzles:
        for split in ('train', 'test'):
            for pair in p[split]:
                _, a, bg, route, direction, initial, gates = _read_input(pair['input'])
                _, _, records = _assemble(a, bg, route, initial, gates)
                shapes.add(pair['input'].shape)
                palettes.add((bg, route, tuple((g['marker'], g['body']) for g in gates)))
                directions.add((direction, split))
                gate_counts.add((len(gates), split))
                for g in records:
                    counts.add(g['n'])
                    signs.add(g['side'])
                    depths.add(g['end'] - g['f'])
                    shifts.add(int(np.sign(g['shift'])))
                    geometries.add(tuple(sorted((r - g['f'], c - g['lo']) for r, c in g['points'])))
    _require(len(shapes) >= 60 and len(geometries) >= 60 and len(palettes) >= 35,
             'insufficient geometry/palette variation')
    _require(directions == {(d, s) for d in range(4) for s in ('train', 'test')}, 'direction coverage')
    _require(counts == {1, 2, 3, 4} and signs == {-1, 1} and shifts == {-1, 1}, 'instruction/movement coverage')
    _require(depths == {0, 1, 2, 3}, 'rear-depth diversity')
    _require(all((n, s) in gate_counts for n in (1, 2, 3, 4) for s in ('train', 'test')), 'gate-count coverage')
    a, expected = _fixture()
    _require(np.array_equal(solve(a), expected), 'literal feedback/frame regression')
    rng = np.random.default_rng(740)
    for direction in range(4):
        colors = rng.permutation(10)
        query = colors[_from_forward(a, direction)]
        _require(np.array_equal(solve(query), colors[_from_forward(expected, direction)]),
                 'rotation/recoloring regression')
        _require(np.array_equal(solve(query[:, ::-1]), colors[_from_forward(expected, direction)][:, ::-1]),
                 'reflection regression')
    # A marker-count edit changes the second gate's destination by one cell.
    edited = a.copy()
    edited[5, 10] = 3  # second unary marker; registration and geometry stay fixed
    _, edited_a, bg, route, _, initial, gates = _read_input(edited)
    moved = _assemble(edited_a, bg, route, initial, gates)[2]
    _require(moved[0]['target'] == 12 and moved[1]['target_lo'] == 9
             and moved[1]['target'] == 6, 'count must affect later registration')
    # Already aligned sources still normalize markers and draw the same route.
    _, forward, bg, route, _, initial, gates = _read_input(a)
    records = _assemble(forward, bg, route, initial, gates)[2]
    aligned = np.zeros_like(a)
    aligned[0, initial] = route
    for g in records:
        for r, c in g['points']:
            aligned[r, c + g['shift']] = a[r, c]
    _require(np.array_equal(solve(aligned), expected), 'zero-displacement support')
    # A deeper frame and a final guide on the lateral boundary, outside sampling bands.
    boundary = np.zeros((15, 12), dtype=np.int8)
    boundary[0, 6] = 1
    deep = {(0, c): 2 for c in range(7)}
    deep.update({(r, c): 2 for r in range(1, 6) for c in (0, 6)})
    deep.update({(5, c): 2 for c in range(7)})
    deep[0, 0], deep[0, 3], deep[1, 6] = 3, 1, 4
    boundary_expected = np.zeros_like(boundary)
    for (r, c), value in deep.items():
        boundary[5 + r, c] = value
        boundary_expected[5 + r, c + 3] = 2 if value == 3 else value
    boundary_expected[:4, 6] = 1
    boundary_expected[3, 6:12] = 1
    boundary_expected[3:, 11] = 1
    _require(np.array_equal(solve(boundary), boundary_expected), 'deep/boundary-envelope support')
    # Five gates: the solver has no four-gate or rear-depth cap.
    longer = np.zeros((28, 20), dtype=np.int8)
    longer_expected = np.zeros_like(longer)
    longer[0, 10] = 1
    lane, start = 10, 0
    for i, f in enumerate((3, 8, 13, 18, 23)):
        spine = {(0, c): 2 for c in range(5)}
        spine.update({(r, c): 2 for r in (1, 2) for c in (0, 4)})
        spine.update({(2, c): 2 for c in range(5)})
        spine[0, 2] = 1
        spine[0, 0 if i % 2 == 0 else 4] = 3
        for (r, c), value in spine.items():
            longer[f + r, c] = value
            longer_expected[f + r, lane - 2 + c] = 2 if value == 3 else value
        turn, target = f - 2, 14 if i % 2 == 0 else 10
        longer_expected[start:turn + 1, lane] = 1
        longer_expected[turn, min(lane, target):max(lane, target) + 1] = 1
        lane, start = target, turn
    longer_expected[start:, lane] = 1
    _require(np.array_equal(solve(longer), longer_expected), 'longer cascade support')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(900 + nt * 10 + nq, nt, nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, 'explicit count ignored')
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = {s: [{k: np.asarray(v).copy() for k, v in x.items()} for x in puzzles[0][s]] for s in puzzles[0]}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged answer accepted')
    ambiguous = a.copy()
    ambiguous[5, 12] = 1
    try:
        solve(ambiguous)
    except AssertionError:
        pass
    else:
        raise AssertionError('duplicate registration accepted')
    return {'seeds': 42, 'distinct_puzzles': 42, 'grid_shapes': len(shapes),
            'carrier_geometries': len(geometries), 'palette_assignments': len(palettes),
            'direction_split_cells': len(directions), 'gate_count_split_cells': len(gate_counts),
            'unary_counts': sorted(counts), 'rear_depths': sorted(depths),
            'semantic_verification': 'generator-owned checks only'}


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
