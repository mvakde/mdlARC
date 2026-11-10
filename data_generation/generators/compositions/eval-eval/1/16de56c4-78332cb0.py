"""Rhythmic trail jigsaw: geometric gluing establishes a path-distance cadence.

Only standard library and NumPy are used. No file access or persistent state is
needed by generation or solving. Construction labels are painted from sampled
geometry, then checked against a reconstruction from the returned input.
"""

from collections import Counter
import hashlib
import numpy as np


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected an integer ARC grid of dimensions 1..30")
    return grid.astype(np.int8, copy=True)


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return (grid.shape, grid.tobytes())


def _neighbors(point):
    r, c = point
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _partition(grid):
    """Read full divider lines; corner background and absent divider are visible."""
    candidates = []
    height, width = grid.shape
    for color in np.unique(grid):
        rows = list(map(int, np.flatnonzero(np.all(grid == color, axis=1))))
        cols = list(map(int, np.flatnonzero(np.all(grid == color, axis=0))))
        if not rows and not cols:
            continue
        rr, cc = [-1] + rows + [height], [-1] + cols + [width]
        hs = [b - a - 1 for a, b in zip(rr, rr[1:])]
        ws = [b - a - 1 for a, b in zip(cc, cc[1:])]
        if min(hs + ws) < 3 or len(set(hs)) != 1 or len(set(ws)) != 1:
            continue
        panels = [grid[r + 1:rr[i + 1], c + 1:cc[j + 1]].copy()
                  for i, r in enumerate(rr[:-1]) for j, c in enumerate(cc[:-1])]
        if not 2 <= len(panels) <= 6 or any(np.any(p == color) for p in panels):
            continue
        corners = {int(p[r, c]) for p in panels for r, c in
                   ((0, 0), (0, -1), (-1, 0), (-1, -1))}
        if len(corners) != 1:
            continue
        candidates.append(dict(panels=panels, divider=int(color),
                               background=corners.pop(), layout=(len(hs), len(ws))))
    if len(candidates) != 1:
        raise ValueError("one equal-piece partition is required")
    return candidates[0]


def _orders(ports):
    """Count up to two directed Hamiltonian chains; repeats need global checking."""
    starts = [i for i, (a, b) in enumerate(ports) if not a and b]
    ends = [i for i, (a, b) in enumerate(ports) if a and not b]
    if len(starts) != 1 or len(ends) != 1 or any(not a and not b for a, b in ports):
        return []
    answers = []

    def visit(order, remaining):
        if len(answers) == 2:
            return
        if not remaining:
            if order[-1] == ends[0]:
                answers.append(tuple(order))
            return
        outgoing = ports[order[-1]][1]
        if outgoing:
            for i in remaining:
                if ports[i][0] == outgoing:
                    visit(order + [i], tuple(j for j in remaining if j != i))

    visit([starts[0]], tuple(i for i in range(len(ports)) if i != starts[0]))
    return answers


def _trace(panel, background, incoming, outgoing):
    """Trace one induced, unbranched path in horizontal piece coordinates."""
    cells = {tuple(map(int, q)) for q in np.argwhere(panel != background)}
    adjacent = {q: [p for p in _neighbors(q) if p in cells] for q in cells}
    if len(cells) < 2 or any(len(v) not in (1, 2) for v in adjacent.values()):
        raise ValueError("trail must be an unbranched path")
    endpoints = [q for q in cells if len(adjacent[q]) == 1]
    if len(endpoints) != 2:
        raise ValueError("trail needs exactly two endpoints")
    if incoming:
        start = (incoming[0], 0)
    else:
        far = (outgoing[0], panel.shape[1] - 1)
        start = next((q for q in endpoints if q != far), None)
    if start not in endpoints:
        raise ValueError("incoming port is not a path endpoint")
    route, previous, current = [], None, start
    while current is not None:
        route.append(current)
        following = [q for q in adjacent[current] if q != previous]
        previous, current = current, (following[0] if following else None)
    if len(route) != len(cells):
        raise ValueError("piece has disconnected trail cells")
    if outgoing and route[-1] != (outgoing[0], panel.shape[1] - 1):
        raise ValueError("outgoing port is not the other endpoint")
    return route


def _beads(values, wire, metric=None):
    """Infer reference hue, exact period/phase, terminal, and off-phase extras."""
    values = np.asarray(values)
    metric = np.arange(len(values)) if metric is None else np.asarray(metric)
    counts = Counter(int(c) for c in values if c != wire)
    repeated = [c for c, n in counts.items() if n == 2]
    if len(repeated) != 1 or any(n > 2 for n in counts.values()):
        raise ValueError("one bead color must occur twice; all other beads once")
    ref = repeated[0]
    a, b = map(int, np.flatnonzero(values == ref))
    period = int(metric[b] - metric[a])
    if period < 1:
        raise ValueError("reference distance must be positive")
    extras = [i for i, c in enumerate(values) if int(c) not in (wire, ref)]
    aligned = [i for i in extras if (int(metric[i]) - int(metric[a])) % period == 0]
    if len(aligned) > 1:
        raise ValueError("multiple on-phase terminals")
    terminal = aligned[0] if aligned else None
    return dict(a=a, b=b, period=period, reference=ref, terminal=terminal,
                offbeat=tuple(i for i in extras if i != terminal))


def _decode(input_grid):
    grid = _grid(input_grid)
    decoded = _partition(grid)
    panels, bg = decoded['panels'], decoded['background']
    counts = Counter(int(c) for p in panels for c in p.ravel() if int(c) != bg)
    wires = [c for c, n in counts.items() if n > 2]
    if len(wires) != 1:
        raise ValueError("plain trail color must be the only foreground hue occurring over twice")
    wire = wires[0]
    candidates = []
    for axis in ('H', 'V'):
        oriented = [p if axis == 'H' else p.T for p in panels]
        if any(np.any(p[0] != bg) or np.any(p[-1] != bg) for p in oriented):
            continue
        ports = [(tuple(map(int, np.flatnonzero(p[:, 0] != bg))),
                  tuple(map(int, np.flatnonzero(p[:, -1] != bg)))) for p in oriented]
        if any(len(a) > 1 or len(b) > 1 for a, b in ports):
            continue
        orders = _orders(ports)
        if len(orders) != 1:
            continue
        try:
            routes = [_trace(p, bg, a, b) for p, (a, b) in zip(oriented, ports)]
        except ValueError:
            continue
        candidates.append((axis, ports, orders[0], routes))
    if len(candidates) != 1:
        raise ValueError("geometry must specify exactly one directed assembly")
    axis, ports, order, routes = candidates[0]
    if axis == 'V':
        routes = [[(c, r) for r, c in route] for route in routes]
    route = [(i, point) for i in order for point in routes[i]]
    values = np.array([panels[i][q] for i, q in route], dtype=np.int8)
    beads = _beads(values, wire)
    decoded.update(axis=axis, ports=ports, order=order, routes=routes,
                   route=route, values=values, wire=wire, beads=beads)
    return decoded


def _paint(values, wire, metric=None):
    metric = np.arange(len(values)) if metric is None else np.asarray(metric)
    bead = _beads(values, wire, metric)
    a, b, p, terminal = (bead[k] for k in ('a', 'b', 'period', 'terminal'))
    mask = (metric - metric[a]) % p == 0
    paint = bead['reference']
    if terminal is not None:
        low, high = min(metric[a], metric[terminal]), max(metric[b], metric[terminal])
        mask &= (metric >= low) & (metric <= high)
        paint = int(values[terminal])
    result = values.copy()
    result[mask] = paint
    return result


def _render(decoded, colors=None, order=None):
    panels = [p.copy() for p in decoded['panels']]
    if colors is not None:
        for (i, point), color in zip(decoded['route'], colors):
            panels[i][point] = color
    order = decoded['order'] if order is None else order
    output = np.concatenate([panels[i] for i in order], axis=1 if decoded['axis'] == 'H' else 0)
    if max(output.shape) > 30:
        raise ValueError("assembled grid exceeds ARC bounds")
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the answer solely from this input; does not mutate its argument."""
    decoded = _decode(input_grid)
    return _render(decoded, _paint(decoded['values'], decoded['wire']))


def _layout(panels, order, shape, divider):
    nr, nc = shape
    h, w = panels[0].shape
    grid = np.full((nr * (h + 1) - 1, nc * (w + 1) - 1), divider, dtype=np.int8)
    for slot, i in enumerate(order):
        r, c = divmod(slot, nc)
        grid[r * (h + 1):r * (h + 1) + h, c * (w + 1):c * (w + 1) + w] = panels[i]
    return grid


def _make_links(rng, n, height):
    pool = list(range(1, height - 1))
    for _ in range(40):
        links = [int(_pick(rng, pool)) for _ in range(n - 1)]
        ports = list(zip([()] + [(x,) for x in links], [(x,) for x in links] + [()]))
        if len(_orders(ports)) == 1:
            return links
    # Distinct blocks, each once or twice, force at most one self-loop per join.
    k = n // 2
    chosen = list(map(int, rng.choice(pool, k, replace=False)))
    doubled = set(map(int, rng.choice(k, n - 1 - k, replace=False)))
    return [row for i, row in enumerate(chosen) for _ in range(2 if i in doubled else 1)]


def _make_route(rng, h, w, incoming, outgoing):
    """Random x-monotone induced route, allowing separated vertical detours."""
    left = 0 if incoming is not None else int(rng.integers(1, w - 1))
    right = w - 1 if outgoing is not None else int(rng.integers(1, w - 1))
    start = incoming if incoming is not None else int(rng.integers(1, h - 1))
    finish = outgoing if outgoing is not None else int(rng.integers(1, h - 1))
    choices = list(range(max(1, left), min(w - 2, right) + 1))
    rng.shuffle(choices)
    bends = []
    for c in choices:
        if all(abs(c - x) >= 2 for x in bends) and (not bends or rng.random() < .65):
            bends.append(c)
    bends.sort()
    # At least one internal bend column is feasible for every supported tile.
    route, row = [(int(start), left)], int(start)
    targets = [int(rng.integers(1, h - 1)) for _ in bends]
    targets[-1] = int(finish)
    target_by_col = dict(zip(bends, targets))
    for c in range(left, right + 1):
        if c != left:
            route.append((row, c))
        target = target_by_col.get(c, row)
        while row != target:
            row += 1 if target > row else -1
            route.append((row, c))
    return route


def _configurations(n, axis):
    options = []
    for h in range(max(4, (n // 2) + 2), 11):
        for w in range(3, min(10, 30 // n) + 1):
            ph, pw = (h, w) if axis == 'H' else (w, h)
            layouts = [(r, n // r) for r in range(1, n + 1) if n % r == 0
                       and r * (ph + 1) - 1 <= 30 and (n // r) * (pw + 1) - 1 <= 30]
            if layouts:
                options.append((h, w, layouts))
    return options


def _geometry(rng, route, output_points, mode, witness, min_piece_gap=0):
    n = len(route)
    options = []
    for period in range(1 if mode == 'terminal' else 2, (n - 1) // 3 + 1):
        for a in range(n - period):
            b = a + period
            if route[b][0] - route[a][0] < min_piece_gap:
                continue
            if witness and route[a][0] == route[b][0]:
                continue
            if mode == 'boundary':
                if not (a >= period and b + period < n):
                    continue
                # The strong witness measures a bend, rather than axial projection.
                if witness and period == abs(output_points[b][1] - output_points[a][1]):
                    continue
                options.append((a, b, None))
            else:
                for t in range(a % period, n, period):
                    if not (t <= a - 2 * period or t >= b + 2 * period):
                        continue
                    low, high = min(a, t), max(b, t)
                    if a % period < low or list(range(a % period, n, period))[-1] > high:
                        options.append((a, b, t))
    if not options:
        return None
    # Sample the period independently so small periods do not dominate by count.
    p = int(_pick(rng, sorted({b - a for a, b, _ in options})))
    return _pick(rng, [g for g in options if g[1] - g[0] == p])


def _build_pair(rng, axis, mode, n, roles, witness, min_piece_gap=0):
    h, w, layouts = _pick(rng, _configurations(n, axis))
    links = _make_links(rng, n, h)
    routes = [_make_route(rng, h, w, None if i == 0 else links[i - 1],
                          None if i == n - 1 else links[i]) for i in range(n)]
    bg, divider, wire = roles
    route = [(i, q) for i, piece in enumerate(routes) for q in piece]
    points = [(r, i * w + c) for i, (r, c) in route]
    geometry = _geometry(rng, route, points, mode, witness, min_piece_gap)
    if geometry is None:
        return None
    a, b, terminal = geometry
    period = b - a
    palette = list(map(int, rng.permutation([c for c in range(10) if c not in roles])))
    ref = palette.pop()
    panels = [np.full((h, w), bg, dtype=np.int8) for _ in range(n)]
    for i, q in route:
        panels[i][q] = wire
    for x in (a, b):
        i, q = route[x]
        panels[i][q] = ref
    paint = ref
    if terminal is not None:
        paint = palette.pop()
        i, q = route[terminal]
        panels[i][q] = paint
    options = [x for x in range(len(route)) if (x - a) % period]
    capacity = len(route) - 2 - (terminal is not None) - 3
    if capacity < 0:
        return None
    extra = min(len(options), int(rng.integers(3)), capacity)
    if witness and mode == 'boundary':
        # An off-beat bead beyond the references contrasts with a real terminal.
        options = [x for x in options if x < a or x > b]
        extra = max(1, extra)
    for x in rng.choice(options, extra, replace=False):
        i, q = route[int(x)]
        panels[i][q] = palette.pop()
    output_panels = [p.copy() for p in panels]
    low, high = (0, len(route) - 1) if terminal is None else (min(a, terminal), max(b, terminal))
    for x in range(a % period, len(route), period):
        if low <= x <= high:
            i, q = route[x]
            output_panels[i][q] = paint
    if axis == 'V':
        panels = [p.T.copy() for p in panels]
        output_panels = [p.T.copy() for p in output_panels]
    layout = _pick(rng, layouts)
    order = list(map(int, rng.permutation(n)))
    if witness and order == list(range(n)):
        order = order[1:] + order[:1]
    pair = dict(input=_layout(panels, order, layout, divider),
                output=np.concatenate(output_panels, axis=1 if axis == 'H' else 0))
    return pair


def _counterfactuals(decoded):
    """Concrete rival rules, used as teaching checks, never as solving state."""
    values, wire, bead = decoded['values'], decoded['wire'], decoded['beads']
    correct = _paint(values, wire)
    local, tile_indices = [], []
    for j, i in enumerate(decoded['order']):
        length = len(decoded['routes'][i])
        local.extend(range(length))
        tile_indices.extend([j] * length)
    local = np.asarray(local)
    reset = values.copy()
    mask = (local - bead['a']) % bead['period'] == 0
    if bead['terminal'] is not None:
        low, high = min(bead['a'], bead['terminal']), max(bead['b'], bead['terminal'])
        mask &= (np.arange(len(values)) >= low) & (np.arange(len(values)) <= high)
    reset[mask] = bead['reference'] if bead['terminal'] is None else values[bead['terminal']]
    try:
        counted_seams = _paint(values, wire, np.arange(len(values)) + tile_indices)
        seam_diff = not np.array_equal(counted_seams, correct)
    except ValueError:
        seam_diff = True
    # Incorrectly use geometric packing order to concatenate each directed piece.
    reading_route = [(i, q) for i in range(len(decoded['panels'])) for q in decoded['routes'][i]]
    reading_values = np.array([decoded['panels'][i][q] for i, q in reading_route], dtype=np.int8)
    try:
        reading_period = _beads(reading_values, wire)['period']
    except ValueError:
        reading_period = -1
    return dict(reset_diff=not np.array_equal(reset, correct), seam_diff=seam_diff,
                reading_period_diff=reading_period != bead['period'])


def _features(pair):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    d = _decode(inp)
    expected = _render(d, _paint(d['values'], d['wire']))
    assert np.array_equal(out, expected), "wrong geometric assembly or trail rhythm"
    bead, values = d['beads'], d['values']
    n = len(d['panels'])
    axis_index = 1 if d['axis'] == 'H' else 0
    side = d['panels'][0].shape[axis_index]
    assert d['divider'] not in out, "packing dividers must disappear"
    assert out.shape[axis_index] == n * side
    chunks = np.split(out, n, axis=axis_index)
    for chunk, i in zip(chunks, d['order']):
        assert np.array_equal(chunk != d['background'], d['panels'][i] != d['background'])
    # Literal off-beat cells survive. No paint can appear outside the old route.
    painted = _paint(values, d['wire'])
    assert all(painted[i] == values[i] for i in bead['offbeat'])
    assert not np.array_equal(values, painted), "every example must change trail colors"
    p, a, b, t = (bead[k] for k in ('period', 'a', 'b', 'terminal'))
    phase = list(range(a % p, len(values), p))
    stop = t is not None and any(x < min(a, t) or x > max(b, t) for x in phase)
    if stop:
        for x in phase:
            if x < min(a, t) or x > max(b, t):
                assert painted[x] == d['wire'], "terminal must stop the rhythm"
    bends = 0
    for route in d['routes']:
        steps = [(q[0] - r[0], q[1] - r[1]) for r, q in zip(route, route[1:])]
        bends += sum(x != y for x, y in zip(steps, steps[1:]))
    ports = [d['ports'][i][1] for i in d['order'][:-1]]
    f = dict(axis=d['axis'], pieces=n, panel_shape=d['panels'][0].shape,
             layout=d['layout'], background=d['background'], divider=d['divider'], wire=d['wire'],
             length=len(values), period=p, phase=a % p, terminal=t is not None,
             terminal_direction=0 if t is None else (-1 if t < a else 1),
             stopping=bool(stop), offbeat=len(bead['offbeat']), bends=bends,
             cross_piece=d['route'][a][0] != d['route'][b][0],
             bidirectional=t is None and phase[0] < a and phase[-1] > b,
             shuffled=d['order'] != tuple(range(n)), repeated_ports=len(set(ports)) < len(ports))
    f.update(_counterfactuals(d))
    return f


def validate_generated_puzzle(puzzle):
    """Validate returned cells, route conservation, uniqueness, and teaching."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, features = set(), {}
    for split in ('train', 'test'):
        features[split] = []
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp = _grid(pair['input'])
            _grid(pair['output'])
            assert _key(inp) not in seen, "inputs must be distinct"
            seen.add(_key(inp))
            features[split].append(_features(pair))
    train = features['train']
    assert any(not f['terminal'] and f['bidirectional'] for f in train)
    assert any(f['terminal'] and f['stopping'] for f in train)
    assert any(f['offbeat'] for f in train)
    assert any(f['cross_piece'] and f['shuffled'] and f['bends'] for f in train)
    for name in ('reset_diff', 'seam_diff', 'reading_period_diff'):
        assert any(f[name] for f in train), "training must distinguish " + name
    assert min(f['pieces'] for f in features['test']) > max(f['pieces'] for f in train)


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    roles = list(map(int, rng.choice(10, 3, replace=False)))
    stable_roles = bool(rng.random() < .5)
    regime = _pick(rng, ('mixed', 'mixed', 'homogeneous', 'shift'))
    first_axis = _pick(rng, ('H', 'V'))
    witness_slots = list(map(int, rng.choice(num_train, 2, replace=False)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            witness = split == 'train' and index in witness_slots
            if witness:
                mode = 'boundary' if index == witness_slots[0] else 'terminal'
            else:
                mode = _pick(rng, ('boundary', 'terminal'))
            axis = _pick(rng, ('H', 'V')) if regime == 'mixed' else first_axis
            if regime == 'mixed' and witness:
                axis = first_axis if mode == 'boundary' else ('V' if first_axis == 'H' else 'H')
            if regime == 'shift' and split == 'test':
                axis = 'V' if first_axis == 'H' else 'H'
            for attempt in range(100):
                n = int(_pick(rng, (5, 6))) if split == 'test' else int(_pick(rng, (3, 4) if witness and mode == 'boundary' else (2, 3, 4)))
                pair_roles = roles if stable_roles else list(map(int, rng.choice(10, 3, replace=False)))
                pair = _build_pair(rng, axis, mode, n, pair_roles, witness,
                                   min_piece_gap=2 if split == 'test' else 0)
                if pair is None or _key(pair['input']) in seen:
                    continue
                f = _features(pair)
                if witness and mode == 'boundary' and not all(f[k] for k in
                        ('reset_diff', 'seam_diff', 'reading_period_diff')):
                    continue
                break
            else:
                raise RuntimeError('bounded route construction exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                g = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(g.shape))
                digest.update(g.tobytes())
    return digest.hexdigest()


def _nonmonotone_regression():
    """Hand-built backward-turning route outside the sampler's vocabulary."""
    routes = [
        [(1, 1), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4), (3, 3),
         (3, 2), (4, 2), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6)],
        [(5, 0), (5, 1), (5, 2), (4, 2), (3, 2), (3, 3), (3, 4), (2, 4), (1, 4)],
    ]
    panels = [np.zeros((7, 7), dtype=np.int8) for _ in routes]
    route = [(i, q) for i, piece in enumerate(routes) for q in piece]
    for i, q in route:
        panels[i][q] = 5
    # References 12/16 imply period four across the join. Terminal four lies
    # two steps before the first reference; singleton seventeen is off phase.
    for x, color in ((12, 2), (16, 2), (4, 3), (17, 8)):
        i, q = route[x]
        panels[i][q] = color
    expected = [p.copy() for p in panels]
    # Explicit label: no dependency on _paint, _decode, or solve.
    for x in (4, 8, 12, 16):
        i, q = route[x]
        expected[i][q] = 3
    return dict(input=_layout(panels, [1, 0], (2, 1), 9),
                output=np.concatenate(expected, axis=1))


def self_test():
    """42 seeds, bounded in-memory metrics, count boundaries and negative checks."""
    seen, records, shapes = set(), [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        seen.add(_puzzle_digest(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input'].copy()
                prediction = solve(inp)
                assert np.array_equal(inp, pair['input']), 'solve mutated its input'
                assert np.array_equal(prediction, pair['output'])
                f = _features(pair)
                records.append(f)
                shapes.add(inp.shape)
                # Changing palette bijectively cannot change geometry or rhythm.
                assert np.array_equal(solve(9 - inp), 9 - prediction)
                assert np.array_equal(solve(inp.T), prediction.T)
    assert len(seen) == 42
    assert {f['pieces'] for f in records} == {2, 3, 4, 5, 6}
    assert {f['axis'] for f in records} == {'H', 'V'}
    assert {f['background'] for f in records} == set(range(10))
    assert {f['wire'] for f in records} == set(range(10))
    assert len(shapes) >= 35 and len({f['panel_shape'] for f in records}) >= 25
    assert len({f['period'] for f in records}) >= 10
    assert {f['terminal_direction'] for f in records} == {-1, 0, 1}
    assert {f['offbeat'] for f in records} >= {0, 1, 2}
    assert sum(f['repeated_ports'] for f in records) >= 10
    assert sum(f['bends'] >= 6 for f in records) >= 30
    assert _puzzle_digest(generate_puzzle(17)) == _puzzle_digest(generate_puzzle(17))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(700 + a * 10 + b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    assert len(_orders([((), (1,)), ((1,), (1,)), ((1,), ())])) == 1
    assert len(_orders([((), (1,)), ((1,), (1,)), ((1,), (1,)), ((1,), ())])) == 2
    manual = _nonmonotone_regression()
    assert np.array_equal(solve(manual['input']), manual['output'])
    assert np.array_equal(solve(manual['input'].T), manual['output'].T)
    damaged = generate_puzzle(0)
    d = _decode(damaged['train'][0]['input'])
    i, q = d['route'][d['beads']['a']]
    # A changed required output cell must fail validation.
    ax = 1 if d['axis'] == 'H' else 0
    slot = d['order'].index(i)
    target = list(q)
    target[ax] += slot * d['panels'][0].shape[ax]
    damaged['train'][0]['output'][tuple(target)] = d['background']
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('route erasure accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed')
