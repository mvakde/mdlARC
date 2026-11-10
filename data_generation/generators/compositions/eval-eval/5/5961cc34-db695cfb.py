"""Latent relay lattice: finite diagonal bars form an opaque signal network.

Only Python's standard library and NumPy are used. The solver decodes the
launcher legend and all geometry from the supplied input, including when the
port color is absent. No parent generator is imported or executed.
"""
from collections import deque

import numpy as np


_DIAGONALS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "Grids must be two-dimensional with axes in 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and
             bool(np.all((0 <= a) & (a <= 9))), "Colors must be integers in 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _span(a, b):
    dr = 1 if b[0] > a[0] else -1
    dc = 1 if b[1] > a[1] else -1
    return tuple((a[0] + k * dr, a[1] + k * dc)
                 for k in range(abs(b[0] - a[0]) + 1))


def _slope(bar):
    return 1 if bar[-1][1] > bar[0][1] else -1


def _inside(p, shape, interior=False):
    lo = int(interior)
    return lo <= p[0] < shape[0] - lo and lo <= p[1] < shape[1] - lo


def _owner(bars):
    owner = {}
    for i, bar in enumerate(bars):
        for p in bar:
            owner.setdefault(p, []).append(i)
    return owner


def _decode(value):
    a = _grid(value)
    h, w = a.shape
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    _require(int(counts.max()) > a.size // 2, "Background must occupy over half the input")
    boundary = [(r, c) for r in range(h) for c in range(w)
                if (r in (0, h - 1) or c in (0, w - 1)) and a[r, c] != bg]
    _require(len(boundary) == 1, "Exactly one original foreground cell touches the boundary")
    start = boundary[0]
    _require(sum((start[0] in (0, h - 1), start[1] in (0, w - 1))) == 1,
             "Launcher must start away from corners")
    signal = int(a[start])
    signal_cells = {tuple(map(int, p)) for p in np.argwhere(a == signal)}
    _require(2 <= len(signal_cells) <= 5, "Launcher tail has 2..5 signal cells")
    possibilities = []
    for dr, dc in _DIAGONALS:
        tail = tuple((start[0] + k * dr, start[1] + k * dc)
                     for k in range(len(signal_cells)))
        cap = (start[0] + len(tail) * dr, start[1] + len(tail) * dc)
        if set(tail) == signal_cells and _inside(cap, a.shape, True):
            possibilities.append((tail, cap, (dr, dc)))
    _require(len(possibilities) == 1, "Launcher must be a straight diagonal tail")
    tail, cap, direction = possibilities[0]
    primary = int(a[cap])
    _require(primary not in (bg, signal), "Cap supplies a distinct endpoint color")
    others = set(map(int, colors)) - {bg, signal, primary}
    _require(len(others) <= 1, "There is at most one port-marker color")
    port_color = next(iter(others)) if others else None
    primary_points = sorted({tuple(map(int, p)) for p in np.argwhere(a == primary)} - {cap})
    markers = ({tuple(map(int, p)) for p in np.argwhere(a == port_color)}
               if port_color is not None else set())
    dots = set(primary_points) | markers
    _require(all(_inside(p, a.shape, True) for p in dots), "All relay dots are interior")
    _require(all(max(abs(r - x), abs(c - y)) > 1
                 for r, c in tail + (cap,) for x, y in dots),
             "Launcher is separated from the other visible dots")
    degrees = {p: 0 for p in primary_points}
    bars = []
    for j, p in enumerate(primary_points):
        for q in primary_points[j + 1:]:
            if abs(p[0] - q[0]) == abs(p[1] - q[1]):
                _require(abs(p[0] - q[0]) >= 2, "Paired spans have at least three cells")
                degrees[p] += 1
                degrees[q] += 1
                bars.append(_span(p, q))
    _require(bars and all(n <= 1 for n in degrees.values()), "Endpoint partners must be unique")
    owner = _owner(bars)
    _require(not (set(tail) | {cap}) & set(owner), "No virtual bar overlaps the launcher")
    _require(all(len(owner.get(p, ())) <= 1 for p in markers),
             "A marker cannot belong to two crossing bars")
    ports = [tuple(sorted(markers & set(bar[1:-1]))) for bar in bars]
    return dict(grid=a, background=bg, signal=signal, primary=primary,
                port_color=port_color, tail=tail, cap=cap, direction=direction,
                bars=bars, ports=ports, owner=owner, markers=markers,
                endpoints={p for bar in bars for p in (bar[0], bar[-1])},
                unpaired={p for p, n in degrees.items() if n == 0},
                inert=markers - set(owner))


def _cast(shape, origin, direction, owner, transparent=False):
    r, c = origin
    path, hits = [], []
    while True:
        r += direction[0]
        c += direction[1]
        if not _inside((r, c), shape):
            return tuple(path), tuple(sorted(set(hits)))
        path.append((r, c))
        if (r, c) in owner:
            hits.extend(owner[r, c])
            if not transparent:
                return tuple(path), tuple(sorted(set(hits)))


def _trace(scene, *, visible_only=False, infinite_ports=False,
           transparent=False, activate_all=False):
    """Reachability closure; original bars remain opaque after activation.

    Keyword switches are documented shortcut counterfactuals used in validation,
    never latent conventions used to solve a family member.
    """
    bars = scene['bars']
    ports = [list(ps) for ps in scene['ports']]
    if infinite_ports:
        for p in scene['inert']:
            for i, bar in enumerate(bars):
                if p[1] - bar[0][1] == _slope(bar) * (p[0] - bar[0][0]):
                    ports[i].append(p)
    owner = scene['owner']
    if visible_only:
        owner = _owner([(bar[0], bar[-1]) for bar in bars])
    trace = set(scene['tail']) | {scene['cap']}
    active = set()
    queue = deque([(scene['cap'], scene['direction'])])
    records = []

    def activate(i):
        if i in active:
            return
        active.add(i)
        trace.update(bars[i])
        slope = _slope(bars[i])
        for p in ports[i]:
            queue.extend(((p, (1, -slope)), (p, (-1, slope))))

    if activate_all:
        for i in range(len(bars)):
            activate(i)
    while queue:
        origin, d = queue.popleft()
        path, hits = _cast(scene['grid'].shape, origin, d, owner, transparent)
        trace.update(path)
        records.append((origin, d, path, hits))
        for i in hits:
            activate(i)
    return trace, active, records


def _render(scene, trace):
    output = np.full_like(scene['grid'], scene['background'])
    for p in trace:
        output[p] = scene['signal']
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using only the input and the fixed, visible launcher convention."""
    scene = _decode(input_grid)
    return _render(scene, _trace(scene)[0])


def _facts(scene, result=None):
    trace, active, rays = _trace(scene) if result is None else result
    visible = scene['endpoints'] | scene['markers']
    shadow = set()
    feedback = False
    encountered = set()
    for origin, d, path, hits in rays:
        if hits:
            feedback |= bool(encountered & set(hits))
            encountered.update(hits)
            _, later = _cast(scene['grid'].shape, path[-1], d, scene['owner'], True)
            shadow.update(set(later) - active)
    return dict(active=active, count=len(scene['bars']), trace=trace,
                hidden=any(path and hits and path[-1] not in visible for _, _, path, hits in rays),
                shadow=shadow, feedback=feedback,
                absorber=any(not scene['ports'][i] for i in active),
                max_ports=max((len(scene['ports'][i]) for i in active), default=0),
                active_slopes={_slope(scene['bars'][i]) for i in active},
                unused_triggered=any(scene['ports'][i] for i in range(len(scene['bars'])) if i not in active),
                active_length=max((len(scene['bars'][i]) for i in active), default=0),
                lengths=tuple(len(bar) for bar in scene['bars']),
                ports=tuple(len(ps) for ps in scene['ports']),
                inert_count=len(scene['inert']), unpaired_count=len(scene['unpaired']),
                shape=scene['grid'].shape, direction=scene['direction'],
                palette=(scene['background'], scene['signal'], scene['primary'], scene['port_color']))


def _new_bars(rng, shape, count, max_length, max_ports):
    h, w = shape
    bars, endpoints = [], []
    for _ in range(count):
        for attempt in range(160):
            length = int(rng.integers(3, min(max_length, h - 2, w - 2) + 1))
            if rng.random() < .45:
                length = min(length, int(rng.integers(3, 10)))
            steps = length - 1
            slope = int(rng.choice((-1, 1)))
            r = int(rng.integers(1, h - steps - 1))
            c = int(rng.integers(1, w - steps - 1)) + (steps if slope < 0 else 0)
            bar = _span((r, c), (r + steps, c + slope * steps))
            if any(abs(p[0] - q[0]) == abs(p[1] - q[1])
                   for p in (bar[0], bar[-1]) for q in endpoints):
                continue
            bars.append(bar)
            endpoints.extend((bar[0], bar[-1]))
            break
        else:
            return None
    owner = _owner(bars)
    ports = []
    for bar in bars:
        eligible = [p for p in bar[1:-1] if len(owner[p]) == 1]
        support, weights = ((0, 1, 2), (.25, .45, .30)) if max_ports == 2 else (
            (0, 1, 2, 3), (.20, .35, .30, .15))
        n = min(len(eligible), int(rng.choice(support, p=weights)))
        ports.append(tuple(eligible[int(k)] for k in rng.permutation(len(eligible))[:n]))
    return bars, ports


def _launcher_options(rng, shape, bars, ports):
    dots = {p for bar in bars for p in (bar[0], bar[-1])} | {p for ps in ports for p in ps}
    owner = _owner(bars)
    h, w = shape
    starts = [(r, c) for r in range(h) for c in range(w)
              if sum((r in (0, h - 1), c in (0, w - 1))) == 1]
    options = [(p, d, length) for p in starts for d in _DIAGONALS
               for length in (2, 3, 4)]
    for k in rng.permutation(len(options)):
        p, d, n = options[int(k)]
        cells = tuple((p[0] + j * d[0], p[1] + j * d[1]) for j in range(n + 1))
        if not all(_inside(q, shape, True) for q in cells[1:]):
            continue
        if set(cells) & set(owner):
            continue
        if any(max(abs(r - x), abs(c - y)) <= 1 for r, c in cells for x, y in dots):
            continue
        yield cells[:-1], cells[-1], d


def _meets(f, witness):
    if witness == 'network':
        return (len(f['active']) >= 2 and f['hidden'] and f['max_ports'] >= 2
                and f['unused_triggered'] and f['active_slopes'] == {-1, 1})
    if witness == 'occlusion':
        return len(f['active']) >= 2 and f['hidden'] and f['shadow'] and f['absorber']
    if witness == 'long':
        return f['active_length'] >= 11 and f['hidden']
    if witness == 'branch':
        return f['max_ports'] >= 3 and f['hidden']
    if witness == 'absorber':
        return len(f['active']) == 1 and f['max_ports'] == 0 and f['hidden']
    return True


def _finite_witness(rng, scene):
    """Add an inert dot beyond a reached span that would change an infinite rule."""
    original = _trace(scene)
    candidates = []
    for i in sorted(original[1]):
        bar = scene['bars'][i]
        slope = _slope(bar)
        for anchor, sign in ((bar[0], -1), (bar[-1], 1)):
            for k in range(1, 30):
                p = (anchor[0] + sign * k, anchor[1] + sign * k * slope)
                if not _inside(p, scene['grid'].shape, True):
                    break
                if p in scene['owner'] or scene['grid'][p] != scene['background']:
                    continue
                if any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= 1
                       for q in scene['tail'] + (scene['cap'],)):
                    continue
                candidates.append(p)
    for k in rng.permutation(len(candidates)):
        p = candidates[int(k)]
        scene['inert'].add(p)
        scene['markers'].add(p)
        extra = _trace(scene, infinite_ports=True)[0]
        if extra != original[0]:
            scene['grid'][p] = scene['port_color']
            return True
        scene['inert'].remove(p)
        scene['markers'].remove(p)
    return False


def _make_scene(shape, bars, ports, tail, cap, direction, palette):
    bg, signal, primary, pc = palette
    a = np.full(shape, bg, dtype=np.int8)
    for bar in bars:
        a[bar[0]] = a[bar[-1]] = primary
    for ps in ports:
        for p in ps:
            a[p] = pc
    for p in tail:
        a[p] = signal
    a[cap] = primary
    # Keep the reserved port role even in a two-role foreground scene, for
    # construction of optional distractors. The public decoder needs no reserve.
    markers = {p for ps in ports for p in ps}
    return dict(grid=a, background=bg, signal=signal, primary=primary,
                port_color=pc, tail=tail, cap=cap, direction=direction,
                bars=bars, ports=ports, owner=_owner(bars), markers=markers,
                endpoints={p for bar in bars for p in (bar[0], bar[-1])},
                unpaired=set(), inert=set())


def _add_dots(rng, scene):
    a, bg = scene['grid'], scene['background']
    # Independently optional unpaired endpoints and off-span port markers.
    primary_count = int(rng.choice((0, 0, 0, 1, 1, 2)))
    inert_count = int(rng.choice((0, 0, 1, 1, 2, 3, 5)))
    for kind, target in (('primary', primary_count), ('port', inert_count)):
        added = 0
        for flat in rng.permutation(a.size):
            p = (int(flat) // a.shape[1], int(flat) % a.shape[1])
            if added >= target:
                break
            if not _inside(p, a.shape, True) or a[p] != bg or p in scene['owner']:
                continue
            if any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= 1
                   for q in scene['tail'] + (scene['cap'],)):
                continue
            if kind == 'primary':
                if any(abs(p[0] - q[0]) == abs(p[1] - q[1])
                       for q in scene['endpoints'] | scene['unpaired']):
                    continue
                a[p] = scene['primary']
                scene['unpaired'].add(p)
            else:
                a[p] = scene['port_color']
                scene['markers'].add(p)
                scene['inert'].add(p)
            added += 1


def _fallback_geometry(rng, witness):
    """Emergency geometry, invented here; never a stored input/output example."""
    if witness == 'long':
        shape = (30, 30)
        ends = (((2, 23), (23, 2)), ((10, 19), (15, 24)),
                ((3, 20), (9, 26)), ((2, 25), (5, 28)))
        ports = [((8, 17), (17, 8)), ((11, 20),), (), ((3, 26),)]
    else:
        shape = (int(rng.integers(16, 31)), int(rng.integers(24, 31)))
        ends = (((4, 9), (10, 3)), ((8, 11), (11, 14)),
                ((3, 14), (8, 19)), ((3, 18), (7, 22)))
        ports = [((5, 8), (8, 5)), ((9, 12),), (), ((4, 19),)]
        if witness == 'branch':
            ports[0] += ((7, 6),)
        if witness == 'absorber':
            ports[0] = ()
    bars = [_span(a, b) for a, b in ends]
    # Apply any square symmetry to the geometry; dimensions transform too.
    reflect = bool(rng.integers(2))
    turns = int(rng.integers(4))
    if reflect:
        bars = [tuple((r, shape[1] - 1 - c) for r, c in b) for b in bars]
        ports = [tuple((r, shape[1] - 1 - c) for r, c in ps) for ps in ports]
    for _ in range(turns):
        bars = [tuple((shape[1] - 1 - c, r) for r, c in b) for b in bars]
        ports = [tuple((shape[1] - 1 - c, r) for r, c in ps) for ps in ports]
        shape = shape[::-1]
    # Canonical upper-to-lower ordering is needed by the slope formula.
    bars = [tuple(sorted(b)) for b in bars]
    return shape, bars, ports


def _generate_pair(rng, palette, witness, length_limit, port_limit, used, count_band):
    for attempt in range(360):
        fallback = attempt >= 320
        if fallback:
            shape, bars, ports = _fallback_geometry(rng, witness)
        else:
            if witness in ('network', 'occlusion'):
                count = int(rng.integers(4, 8 if count_band else 7))
                shape = tuple(map(int, rng.integers(17, 31, 2)))
            elif witness in ('long', 'branch'):
                count = int(rng.integers(2, 9))
                shape = tuple(map(int, rng.integers(18, 31, 2)))
            else:
                count = int(rng.integers(1, 9 if count_band else 6))
                shape = tuple(map(int, rng.integers(11, 31, 2)))
                if rng.random() < .25:
                    shape = (int(rng.integers(9, 16)), int(rng.integers(23, 31)))
            geometry = _new_bars(rng, shape, count, length_limit, port_limit)
            if geometry is None:
                continue
            bars, ports = geometry
        for index, (tail, cap, direction) in enumerate(_launcher_options(rng, shape, bars, ports)):
            if index >= 160:
                break
            scene = _make_scene(shape, bars, ports, tail, cap, direction, palette)
            result = _trace(scene)
            facts = _facts(scene, result)
            if not _meets(facts, witness):
                continue
            if witness == 'network' and not _finite_witness(rng, scene):
                continue
            _add_dots(rng, scene)
            if _key(scene['grid']) in used:
                continue
            decoded = _decode(scene['grid'])
            output = _render(decoded, _trace(decoded)[0])
            return {'input': scene['grid'], 'output': output}
    raise RuntimeError("Bounded placement exhausted, including emergency geometry")


def _count(value, name, lower, upper):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or not lower <= int(value) <= upper):
        raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    train_count = _count(num_train, 'num_train', 2, 10)
    test_count = _count(num_test, 'num_test', 1, 4)
    _require(train_count > test_count, "num_train must exceed num_test")
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and int(seed) >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    base = tuple(map(int, rng.permutation(10)[:4]))
    varying_palette = bool(rng.integers(2))
    count_band = int(rng.integers(2))
    shift = str(rng.choice(('long', 'branch', 'absorber')))
    puzzle, used = {'train': [], 'test': []}, set()
    for split, count in (('train', train_count), ('test', test_count)):
        for index in range(count):
            witness = ('network', 'occlusion')[index] if split == 'train' and index < 2 else (
                shift if split == 'test' else '')
            length_limit = 8 if shift == 'long' and split == 'train' else 25
            port_limit = 2 if split == 'train' or shift != 'branch' else 3
            palette = base
            if varying_palette:
                palette = (tuple(np.roll(base, 1)) if split == 'train' and index == 1
                           else base if split == 'train' and index == 0
                           else tuple(map(int, rng.permutation(10)[:4])))
            pair = _generate_pair(rng, palette, witness, length_limit, port_limit, used, count_band)
            used.add(_key(pair['input']))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "Wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    a = _count(len(puzzle['train']), 'num_train', 2, 10)
    b = _count(len(puzzle['test']), 'num_test', 1, 4)
    _require(a > b, "Training count must exceed test count")
    seen, training, test = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "Wrong pair keys")
            scene = _decode(pair['input'])
            out = _grid(pair['output'])
            result = _trace(scene)
            _require(out.shape == scene['grid'].shape and
                     np.array_equal(out, _render(scene, result[0])), "Output disagrees with the input rule")
            _require(set(map(int, np.unique(out))) <= {scene['background'], scene['signal']},
                     "Output can contain only background and signal")
            key = _key(scene['grid'])
            _require(key not in seen, "Duplicate input")
            seen.add(key)
            _require(not np.array_equal(scene['grid'], out), "Unchanged example")
            f = _facts(scene, result)
            (training if split == 'train' else test).append((scene, f))
    _require(any(_meets(f, 'network') for _, f in training), "Missing branching network demonstration")
    _require(any(_meets(f, 'occlusion') for _, f in training), "Missing first-hit/absorber demonstration")
    for switch in ('activate_all', 'visible_only', 'infinite_ports', 'transparent'):
        _require(any(_trace(scene, **{switch: True})[0] != f['trace'] for scene, f in training),
                 f"Training fails to distinguish shortcut {switch}")
    train_facts = [f for _, f in training]
    test_facts = [f for _, f in test]
    # At least one observable shift, without generation metadata or seed.
    long_shift = (max(max(f['lengths']) for f in train_facts) <= 8 and
                  all(f['active_length'] >= 11 for f in test_facts))
    branch_shift = (all(f['max_ports'] <= 2 for f in train_facts) and
                    all(f['max_ports'] >= 3 for f in test_facts))
    absorber_shift = all(len(f['active']) == 1 and f['max_ports'] == 0 for f in test_facts)
    _require(long_shift or branch_shift or absorber_shift, "Missing intended train/test shift")


def _puzzle_key(puzzle):
    return tuple((_key(_grid(p['input'])), _key(_grid(p['output'])))
                 for s in ('train', 'test') for p in puzzle[s])


def self_test():
    """Bounded diversity and invariant audit over exactly 42 default puzzles."""
    keys, shapes, palettes, geometry, facts = set(), set(), set(), set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        keys.add(_puzzle_key(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                scene = _decode(pair['input'])
                f = _facts(scene)
                facts.append((split, f))
                shapes.add(f['shape'])
                palettes.add(f['palette'])
                geometry.add((f['shape'], f['lengths'], f['ports'], len(f['active']), f['direction']))
                before = scene['grid'].copy()
                assert np.array_equal(solve(before), pair['output'])
                assert np.array_equal(before, scene['grid']), "Solver mutated input"
    assert len(keys) == 42
    assert len(shapes) >= 50 and len(palettes) >= 25 and len(geometry) >= 150
    rows = [f for _, f in facts]
    assert {f['direction'] for f in rows} == set(_DIAGONALS)
    assert len({f['count'] for f in rows}) >= 6
    assert {0, 1, 2, 3} <= {n for f in rows for n in f['ports']}
    assert any(f['feedback'] for f in rows) and any(f['shadow'] for f in rows)
    assert any(f['unpaired_count'] for f in rows) and any(f['inert_count'] == 0 for f in rows)
    assert any(max(f['shape']) / min(f['shape']) >= 2 for f in rows)
    assert max(max(f['lengths']) for f in rows) >= 20
    assert len({n for f in rows for n in f['lengths']}) >= 15
    # Exact minimum request, maximum request, and reproducibility.
    for counts in ((2, 1), (10, 4)):
        p = generate_puzzle(1073, *counts)
        assert _puzzle_key(p) == _puzzle_key(generate_puzzle(1073, *counts))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count request accepted")
    # A known cyclic graph: parallel opaque bars send a port-aligned ray back.
    bars = [_span((4, 9), (10, 3)), _span((8, 15), (14, 9))]
    scene = _make_scene((18, 18), bars, [((5, 8),), ((10, 13),)],
                        ((0, 1), (1, 2)), (2, 3), (1, 1), (0, 9, 2, 4))
    decoded = _decode(scene['grid'])
    result = _trace(decoded)
    assert result[1] == {0, 1} and _facts(decoded, result)['feedback']
    assert len(result[2]) == 5, "Each reached port emits exactly two rays once"
    # No port color is needed to complete and absorb a hidden first hit.
    empty_ports = _make_scene((12, 12), [_span((4, 7), (8, 3))], [()],
                             ((0, 1), (1, 2)), (2, 3), (1, 1), (0, 9, 2, 4))
    assert _decode(empty_ports['grid'])['port_color'] is None
    expected_cells = {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
                      (4, 7), (6, 5), (7, 4), (8, 3)}
    expected = np.zeros((12, 12), dtype=np.int8)
    for cell in expected_cells:
        expected[cell] = 9
    assert np.array_equal(solve(empty_ports['grid']), expected)
    # Exercise every emergency witness directly, even when no random sample
    # exhausts its placement budget. No generated seed is special-cased.
    emergency_rng = np.random.default_rng(9017)
    for witness in ('network', 'occlusion', 'long', 'branch', 'absorber'):
        for _ in range(3):
            shape, bars, ports = _fallback_geometry(emergency_rng, witness)
            for tail, cap, direction in _launcher_options(emergency_rng, shape, bars, ports):
                scene = _make_scene(shape, bars, ports, tail, cap, direction, (0, 9, 2, 4))
                if not _meets(_facts(scene), witness):
                    continue
                if witness == 'network' and not _finite_witness(emergency_rng, scene):
                    continue
                decoded = _decode(scene['grid'])
                assert _meets(_facts(decoded), witness)
                assert np.array_equal(solve(scene['grid']), _render(scene, _trace(scene)[0]))
                break
            else:
                raise AssertionError('Emergency construction lost its teaching witness')
    # Crossing spans are allowed, but a marker at their intersection is not.
    crossing = _make_scene((18, 18), [_span((4, 5), (10, 11)), _span((5, 12), (11, 6))],
                           [(), ()], ((0, 2), (1, 3)), (2, 4), (1, 1), (0, 9, 2, 4))
    _decode(crossing['grid'])
    crossing['grid'][8, 9] = 4
    try:
        _decode(crossing['grid'])
    except ValueError:
        pass
    else:
        raise AssertionError("Ambiguous crossing marker accepted")
    # Deliberate output corruption exercises semantic rejection, not just dtype.
    p = generate_puzzle(0)
    p['test'][0]['output'][0, 0] = (int(p['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except ValueError:
        pass
    else:
        raise AssertionError("Corrupted output accepted")


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds, input-derived routing and shortcut witnesses')
