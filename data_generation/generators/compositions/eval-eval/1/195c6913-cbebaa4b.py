"""Phase-locked staircase workshop: a cyclic brush chooses geometric docks.

Only standard-library and NumPy dependencies. No parent code or task data is
loaded. The header encodes every palette role needed by the input-only solver.
"""
from __future__ import annotations

from collections import Counter
import json
import numpy as np

N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
KEYS = tuple((gap, offset) for gap in range(2, 7) for offset in (-1, 0, 1))
REQUIRED = {"cycling", "multiplicity", "bend_phase", "panel_phase",
            "marker_phase", "seam_phase", "color_gate", "geometry_gate", "movement"}


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _counts(a, b):
    if not (_integer(a) and _integer(b) and 2 <= a <= 10 and
            1 <= b <= 4 and a > b):
        raise ValueError("Require integer counts 2..10 / 1..4, with train > test")
    return int(a), int(b)


def _components(cells):
    left = set(cells)
    result = []
    while left:
        q = min(left)
        left.remove(q)
        found, stack = {q}, [q]
        while stack:
            r, c = stack.pop()
            for dr, dc in N4:
                p = r + dr, c + dc
                if p in left:
                    left.remove(p)
                    found.add(p)
                    stack.append(p)
        result.append(found)
    return result


def _shift(cells, delta):
    return {(r + delta[0], c + delta[1]) for r, c in cells}


def _box(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells), max(c for r, c in cells))


def _key(pins):
    if len(pins) != 2:
        raise ValueError("A dock has two pins")
    upper, main = sorted(pins)
    key = main[0] - upper[0], upper[1] - main[1]
    if key not in KEYS:
        raise ValueError("Invalid docking geometry")
    return key, main


def _pins(main, key):
    gap, offset = key
    return {main, (main[0] - gap, main[1] + offset)}


def _trace(body, start, outlet):
    """Walk maximal right/up runs. An outlet takes precedence over a turn."""
    path, collisions, legs = [], [], []
    q, direction = start, 0
    for _ in range(60):
        dr, dc = ((0, 1), (-1, 0))[direction]
        length = 0
        while q in body:
            if q in path:
                raise ValueError("Repeated route cell")
            path.append(q)
            length += 1
            q = q[0] + dr, q[1] + dc
        if not length:
            raise ValueError("Missing route continuation")
        legs.append(length)
        if q == outlet:
            if direction != 0:
                raise ValueError("Outlet must finish a rightward leg")
            return dict(path=tuple(path), collisions=tuple(collisions), legs=tuple(legs))
        collisions.append(q)
        direction = 1 - direction
        dr, dc = ((0, 1), (-1, 0))[direction]
        q = path[-1][0] + dr, path[-1][1] + dc
    raise ValueError("Route does not reach its outlet")


def _array(value):
    g = np.asarray(value)
    if not (g.ndim == 2 and all(1 <= n <= 30 for n in g.shape) and
            np.issubdtype(g.dtype, np.integer) and np.all((0 <= g) & (g <= 9))):
        raise ValueError("Expected an integer ARC grid")
    return g.astype(np.int8, copy=True)


def _decode(value):
    g = _array(value)
    h, w = g.shape
    if h < 8 or w < 15:
        raise ValueError("Missing workshop header")
    bg = int(g[0, 0])

    def swatch(c):
        a = g[1:3, c:c + 2]
        if a.shape != (2, 2) or not np.all(a == a[0, 0]):
            raise ValueError("Broken header swatch")
        return int(a[0, 0])

    terrain, marker = swatch(1), swatch(4)
    word = []
    for c in range(9, min(w - 1, 25) + 1, 3):
        color = swatch(c)
        if color == bg:
            break
        word.append(color)
    if not 2 <= len(word) <= 6 or len({bg, terrain, marker}) != 3:
        raise ValueError("Missing palette roles or cyclic word")
    if set(word) & {bg, terrain, marker} or len(set(word)) < 2:
        raise ValueError("Program colors must be separate from terrain and pins")
    header = np.full((5, w), bg, dtype=np.int8)
    for c, color in [(1, terrain), (4, marker)] + list(zip(range(9, 27, 3), word)):
        header[1:3, c:c + 2] = color
    if not np.array_equal(g[:5], header):
        raise ValueError("Unrecognized header or occupied separator")
    work = g.copy()
    work[:5] = bg
    pieces, anchors = [], []
    for cells in _components(map(tuple, np.argwhere(work != bg).tolist())):
        body = {q for q in cells if int(g[q]) == terrain}
        outpins = {q for q in cells if int(g[q]) == marker}
        colored = cells - body - outpins
        if not colored or len({int(g[q]) for q in colored}) != 1:
            raise ValueError("Missing inlet or start")
        hue = int(g[min(colored)])
        if hue not in word:
            raise ValueError("Inlet color absent from program")
        outkey, outmain = _key(outpins)
        if len(colored) == 1:
            if hue != word[0]:
                raise ValueError("Start must show the first phase")
            start = next(iter(colored))
            body.add(start)
            inkey, inmain, inpins = None, None, set()
            anchors.append(len(pieces))
        else:
            inkey, inmain = _key(colored)
            inpins = colored
            start = inmain[0], inmain[1] + 1
        if start not in body or len(_components(body)) != 1:
            raise ValueError("Broken terrain panel")
        record = _trace(body, start, outmain)
        if set(record['collisions']) & cells:
            raise ValueError("A bend collision must be empty")
        pieces.append(dict(body=body, inpins=inpins, outpins=outpins,
                           cells=cells, hue=hue, inkey=inkey, inmain=inmain,
                           outkey=outkey, outmain=outmain, start=start, **record))
    if len(anchors) != 1 or len(pieces) < 2:
        raise ValueError("Require one visible anchored start and loose panels")
    # Input panels may not even touch diagonally: all port ownership is visible.
    for i, p in enumerate(pieces):
        buffer = {(r + dr, c + dc) for r, c in p['cells']
                  for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        if any(buffer & q['cells'] for q in pieces[:i]):
            raise ValueError("Scattered panels touch")
    return dict(grid=g, bg=bg, terrain=terrain, marker=marker,
                word=tuple(word), pieces=pieces, anchor=anchors[0])


def _fits(info, piece, delta, occupied, gate, exact=True):
    cells = _shift(piece['cells'], delta)
    collisions = _shift(piece['collisions'], delta)
    h, w = info['grid'].shape
    if any(not (5 <= r < h and 0 <= c < w) for r, c in cells | collisions):
        return False
    # For the geometry ablation only, ignore the satellite-pin requirement.
    if exact:
        if cells & occupied != gate:
            return False
    elif ((cells & occupied) - gate or
          _shift(piece['body'] | piece['outpins'], delta) & occupied):
        return False
    if collisions & occupied or collisions & cells:
        return False
    return True


def _assemble(info):
    pieces, word = info['pieces'], info['word']
    current = info['anchor']
    positions, order = {current: (0, 0)}, [current]
    p = pieces[current]
    occupied = p['cells'] | set(p['collisions'])
    phase = len(p['path'])
    gate_evidence = set()
    while True:
        p, d = pieces[current], positions[current]
        main = p['outmain'][0] + d[0], p['outmain'][1] + d[1]
        gate = _shift(p['outpins'], d)
        needed = word[phase % len(word)]
        choices = []
        for j, q in enumerate(pieces):
            if j in positions:
                continue
            delta = main[0] - q['inmain'][0], main[1] - q['inmain'][1]
            same_shape, same_hue = q['inkey'] == p['outkey'], q['hue'] == needed
            if same_shape and _fits(info, q, delta, occupied, gate):
                if same_hue:
                    choices.append((j, delta))
                else:
                    gate_evidence.add('color_gate')
            elif same_hue and not same_shape and _fits(info, q, delta, occupied, gate, False):
                gate_evidence.add('geometry_gate')
        if len(choices) > 1:
            raise ValueError("Ambiguous phase-compatible dock")
        if not choices:
            break
        current, delta = choices[0]
        positions[current] = delta
        order.append(current)
        q = pieces[current]
        occupied |= _shift(q['cells'], delta) | _shift(q['collisions'], delta)
        phase += len(q['path'])
    return dict(positions=positions, order=order, gate_evidence=gate_evidence)


def _draw(info, assembly, variant=None):
    out = np.full(info['grid'].shape, info['bg'], dtype=np.int8)
    word = info['word']
    if variant == 'multiplicity':
        word = tuple(dict.fromkeys(word))
    phase = 0
    for i, j in enumerate(assembly['order']):
        p, d = info['pieces'][j], assembly['positions'][j]
        for q in _shift(p['body'], d):
            out[q] = info['terrain']
        for q in _shift(p['inpins'] | p['outpins'] | set(p['collisions']), d):
            out[q] = info['marker']
        if variant == 'panel_phase':
            phase = 0
        local = 0
        for leg_index, leg in enumerate(p['legs']):
            if variant == 'bend_phase':
                phase = 0
            for q in p['path'][local:local + leg]:
                out[q[0] + d[0], q[1] + d[1]] = word[phase % len(word)]
                phase += 1
            local += leg
            if variant == 'marker_phase' and leg_index < len(p['legs']) - 1:
                phase += 1
        if variant == 'seam_phase':
            phase += len(p['outpins'])
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """All roles, the anchor, program, terrain and choices come from this input."""
    info = _decode(input_grid)
    return _draw(info, _assemble(info))


def _coverage(info, assembly):
    correct = _draw(info, assembly)
    evidence = set(assembly['gate_evidence'])
    for name in ('multiplicity', 'bend_phase', 'panel_phase', 'marker_phase', 'seam_phase'):
        if not np.array_equal(correct, _draw(info, assembly, name)):
            evidence.add(name)
    if sum(len(info['pieces'][j]['path']) for j in assembly['order']) >= 2 * len(info['word']):
        evidence.add('cycling')
    if any(d != (0, 0) for d in assembly['positions'].values()):
        evidence.add('movement')
    return evidence


def _plan(horizontal, vertical):
    path, collisions, legs = [], [], []
    r = c = 0
    for i, distance in enumerate(horizontal):
        segment = [(r, k) for k in range(c, c + distance + 1)]
        if i:
            segment = segment[1:]  # Bend is counted only once.
        path.extend(segment)
        legs.append(len(segment))
        c += distance
        if i < len(vertical):
            collisions.append((r, c + 1))
            distance_up = vertical[i]
            path.extend((k, c) for k in range(r - 1, r - distance_up - 1, -1))
            legs.append(distance_up)
            r -= distance_up
            collisions.append((r - 1, c))
    return dict(path=tuple(path), collisions=tuple(collisions), legs=tuple(legs)), (r, c)


def _panel(rng, inkey, outkey, horizontal, vertical, hue):
    record, end = _plan(horizontal, vertical)
    inpins = _pins((0, -1), inkey) if inkey is not None else set()
    outpins = _pins((end[0], end[1] + 1), outkey)
    blocked = inpins | outpins | set(record['collisions'])
    body = set(record['path'])
    if inkey is not None:
        gap, offset = inkey
        body |= {(r, c) for r in range(-gap, 1) for c in (0, 1)} - blocked
        body.add((-gap, offset))
    gap, offset = outkey
    body |= {(r, c) for r in range(end[0] - gap, end[0] + 1)
             for c in (end[1] - 1, end[1])} - blocked
    body.add((end[0] - gap, end[1] + offset))
    # Grow off-route terrain without filling any collision cell. Supports make
    # every sparse pin visibly belong to its panel; added cells do not change runs.
    top = min(r for r, c in body)
    available = {(r, c) for r in range(top, 1) for c in range(end[1] + 1)} - blocked
    target = max(len(body), int(len(available) * rng.uniform(.30, .92)))
    while len(body) < target:
        frontier = sorted({(r + dr, c + dc) for r, c in body for dr, dc in N4}
                          & available - body)
        if not frontier:
            break
        body.add(frontier[int(rng.integers(len(frontier)))])
    if body & blocked or len(_components(body)) != 1:
        raise ValueError("Unsupported panel geometry")
    actual = _trace(body, (0, 0), (end[0], end[1] + 1))
    if actual != record:
        raise ValueError("Terrain changed planned maximal runs")
    return dict(body=body, inpins=inpins, outpins=outpins, hue=hue,
                start=(0, 0), inmain=(0, -1) if inkey else None,
                outmain=(end[0], end[1] + 1), inkey=inkey,
                outkey=outkey, cells=body | inpins | outpins, **record)


def _word(rng, length, colors, repeated=False):
    distinct = int(rng.integers(2, min(4, length) + 1))
    if repeated:
        distinct = min(distinct, length - 1)
    palette = list(map(int, rng.choice(colors, distinct, replace=False)))
    word = palette + [palette[int(rng.integers(distinct))] for _ in range(length - distinct)]
    rng.shuffle(word)
    return tuple(word)


def _scatter(rng, shape, panels, anchor, bg, terrain, marker, word):
    h, w = shape
    coordinates = [np.array(sorted(p['cells']), dtype=int) for p in panels]
    boxes = [_box(p['cells']) for p in panels]
    order = sorted(range(1, len(panels)), key=lambda i: -len(panels[i]['cells']))
    for _ in range(30):
        inp = np.full(shape, bg, dtype=np.int8)
        forbidden = np.zeros(shape, dtype=bool)

        def reserve(cells):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = cells[:, 0] + dr, cells[:, 1] + dc
                    valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
                    forbidden[rr[valid], cc[valid]] = True

        reserve(coordinates[0])
        positions = {0: (0, 0)}
        for i in order:
            p = panels[i]
            t, l, b, r = boxes[i]
            for _ in range(350):
                d = (int(rng.integers(5, h - (b - t))) - t,
                     int(rng.integers(0, w - (r - l))) - l)
                cells = coordinates[i] + np.array(d)
                if d != (0, 0) and not np.any(forbidden[cells[:, 0], cells[:, 1]]):
                    positions[i] = d
                    reserve(cells)
                    break
            else:
                break
        if len(positions) != len(panels):
            continue
        for i, p in enumerate(panels):
            d = positions[i]
            for q in _shift(p['body'], d):
                inp[q] = terrain
            for q in _shift(p['outpins'], d):
                inp[q] = marker
            for q in _shift(p['inpins'], d):
                inp[q] = p['hue']
            if i == anchor:
                q = p['start'][0] + d[0], p['start'][1] + d[1]
                inp[q] = word[0]
        for c, color in [(1, terrain), (4, marker)] + list(zip(range(9, 27, 3), word)):
            inp[1:3, c:c + 2] = color
        return inp
    return None


def _make_pair(rng, roles, regime, test=False, witness=False):
    bg, terrain, marker = roles
    colors = [c for c in range(10) if c not in roles]
    for attempt in range(180):
        length = int(rng.integers(3, 6)) if witness else int(rng.integers(2, 6))
        if regime == 'word' and test:
            length = 6
        word = _word(rng, length, colors, witness)
        count = 4 if regime == 'chain' and test else int(rng.integers(2, 4))
        keys = [KEYS[int(x)] for x in rng.choice(len(KEYS), count + 2, replace=False)]
        # Sometimes two active inlet geometries are identical. Their different
        # accepted phases, rather than a unique geometric order, select the
        # next panel. Leave terminal and diagnostic key types separate.
        if count >= 3 and len(set(word)) >= 3 and rng.random() < .55:
            keys[int(rng.integers(1, count - 1))] = keys[0]
        panels, cursor, phase = [], (0, 0), 0
        offsets = []
        try:
            for i in range(count):
                bends = 2 if regime == 'bends' and test and i == 0 else int(rng.integers(0, 2))
                if witness and i == 0:
                    bends = 1
                # Smaller panels make room for more docks; width is not a lattice.
                horizontal = [int(rng.integers(2, 5)) for _ in range(bends + 1)]
                if count == 4:
                    horizontal = [int(rng.integers(2, 4)) for _ in range(bends + 1)]
                vertical = [int(rng.integers(2, 5)) for _ in range(bends)]
                p = _panel(rng, keys[i - 1] if i else None, keys[i],
                           horizontal, vertical, word[phase % length])
                panels.append(p)
                offsets.append(cursor)
                phase += len(p['path'])
                main = p['outmain']
                cursor = cursor[0] + main[0], cursor[1] + main[1] + 1
            # Two diagnostic unused panels compete at the first outlet: one
            # matches its shape, one its phase. They have different terrain.
            needed = panels[1]['hue']
            used_hues = {p['hue'] for p in panels[1:] if p['inkey'] == keys[0]}
            wrong_choices = sorted(set(word) - used_hues)
            if not wrong_choices:
                continue
            wrong_hue = int(rng.choice(wrong_choices))
            for key, hue in ((keys[0], wrong_hue), (keys[count], needed)):
                p = _panel(rng, key, keys[count + 1],
                           [int(rng.integers(2, 5)), int(rng.integers(2, 5))],
                           [int(rng.integers(2, 5))], hue)
                panels.append(p)
        except ValueError:
            continue
        cells = set().union(*(_shift(p['cells'], d) | _shift(p['collisions'], d)
                             for p, d in zip(panels[:count], offsets)))
        t, l, b, r = _box(cells)
        extent_h, extent_w = b - t + 1, r - l + 1
        if extent_h > 23 or extent_w > 29:
            continue
        h = int(rng.integers(max(22, extent_h + 5), 31))
        w = int(rng.integers(max(23, extent_w, 3 * length + 8), 31))
        delta = (int(rng.integers(5, h - extent_h + 1)) - t,
                 int(rng.integers(0, w - extent_w + 1)) - l)
        # Active panels use assembly coordinates; loose-panel coordinates may
        # be any convenient reference before all of them are scattered.
        placed = []
        for i, p in enumerate(panels):
            d = (offsets[i][0] + delta[0], offsets[i][1] + delta[1]) if i < count else delta
            q = dict(p)
            for name in ('body', 'inpins', 'outpins', 'cells', 'path', 'collisions'):
                moved = [(a + d[0], c + d[1]) for a, c in p[name]]
                q[name] = tuple(moved) if name in ('path', 'collisions') else set(moved)
            for name in ('start', 'inmain', 'outmain'):
                q[name] = None if p[name] is None else (p[name][0] + d[0], p[name][1] + d[1])
            placed.append(q)
        inp = _scatter(rng, (h, w), placed, 0, bg, terrain, marker, word)
        if inp is None:
            continue
        # Labels are constructed from the planned route and rigid positions,
        # not obtained by calling solve. The decoder must recover them.
        planned_info = dict(grid=inp, bg=bg, terrain=terrain, marker=marker,
                            word=word, pieces=placed)
        out = _draw(planned_info, dict(order=list(range(count)), positions={i: (0, 0) for i in range(count)}))
        try:
            info = _decode(inp)
            assembly = _assemble(info)
        except ValueError:
            continue
        if (len(assembly['order']) != count or
                any(d == (0, 0) for j, d in assembly['positions'].items() if j != info['anchor']) or
                not np.array_equal(out, _draw(info, assembly))):
            continue
        evidence = _coverage(info, assembly)
        if witness and not REQUIRED <= evidence:
            continue
        return {'input': inp, 'output': out}
    raise RuntimeError("Bounded workshop construction failed")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice(10, 3, replace=False)))
    regime = ('chain', 'bends', 'word')[int(rng.integers(3))]
    witness_index = int(rng.integers(a))
    puzzle, used = {'train': [], 'test': []}, set()
    for split, count in (('train', a), ('test', b)):
        for i in range(count):
            for _ in range(12):
                pair = _make_pair(rng, roles, regime, split == 'test',
                                  split == 'train' and i == witness_index)
                key = pair['input'].shape, pair['input'].tobytes()
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not create distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, evidence, roles = set(), set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _array(pair['input']), _array(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = inp.shape, inp.tobytes()
            assert key not in seen
            seen.add(key)
            info = _decode(inp)
            assembly = _assemble(info)
            roles.add((info['bg'], info['terrain'], info['marker']))
            assert np.array_equal(out, _draw(info, assembly))
            assert 2 <= len(assembly['order']) <= 4
            assert len(info['pieces']) - len(assembly['order']) == 2
            assert assembly['positions'][info['anchor']] == (0, 0)
            assert all(d != (0, 0) for j, d in assembly['positions'].items() if j != info['anchor'])
            # Rigid terrain-cell conservation, including unpainted lobes, and
            # shared inlet/outlet ownership. Painting changes hue, not geometry.
            bodies, pins, collisions = set(), Counter(), set()
            for j in assembly['order']:
                p, d = info['pieces'][j], assembly['positions'][j]
                moved = _shift(p['body'], d)
                assert not moved & bodies
                bodies |= moved
                pins.update(_shift(p['inpins'] | p['outpins'], d))
                collisions |= _shift(p['collisions'], d)
            assert not bodies & (set(pins) | collisions)
            assert Counter(pins.values())[2] == 2 * (len(assembly['order']) - 1)
            assert set(pins.values()) <= {1, 2}
            assert np.count_nonzero(out != info['bg']) == len(bodies | set(pins) | collisions)
            assert np.all(out[:5] == info['bg'])
            if split == 'train':
                evidence |= _coverage(info, assembly)
    assert len(roles) == 1
    assert REQUIRED <= evidence, REQUIRED - evidence


def _signature(puzzle):
    return tuple((s, p['input'].shape, np.asarray(p['input']).tobytes(),
                  np.asarray(p['output']).tobytes()) for s in ('train', 'test') for p in puzzle[s])


def self_test():
    """Bounded 42-seed structural/semantic diversity and negative regressions."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_signature(p) for p in puzzles}) == 42
    assert _signature(puzzles[0]) == _signature(generate_puzzle(0))
    metrics = Counter()
    shapes, masks, roles, keys, patterns, leg_counts, counts = set(), set(), set(), set(), set(), set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                assembly = _assemble(info)
                shapes.add(info['grid'].shape)
                roles.add((info['bg'], info['terrain'], info['marker']))
                counts.add(len(assembly['order']))
                names = {}
                patterns.add(tuple(names.setdefault(x, len(names)) for x in info['word']))
                metrics['pairs'] += 1
                metrics['selected_panels'] += len(assembly['order'])
                metrics['unused_panels'] += len(info['pieces']) - len(assembly['order'])
                active_keys = [info['pieces'][j]['inkey'] for j in assembly['order'][1:]]
                metrics['repeated_active_keys'] += len(active_keys) > len(set(active_keys))
                for p in info['pieces']:
                    t, l, b, r = _box(p['body'])
                    masks.add(tuple(sorted((a - t, c - l) for a, c in p['body'])))
                    keys.add(p['outkey'])
                    leg_counts.add(len(p['legs']))
    assert counts == {2, 3, 4}
    assert leg_counts >= {1, 3, 5}
    assert len(shapes) >= 25 and len(masks) >= 200
    assert len(roles) >= 30 and len(keys) == 15 and len(patterns) >= 20
    assert metrics['repeated_active_keys'] >= 5
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            generate_puzzle(71 + a * 7 + b, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted")
    # Verify failure on each incorrect phase model and on an erased dock pin.
    base = puzzles[0]
    for variant in ('multiplicity', 'bend_phase', 'panel_phase', 'marker_phase', 'seam_phase'):
        for index, pair in enumerate(base['train']):
            info = _decode(pair['input'])
            assembly = _assemble(info)
            bad_out = _draw(info, assembly, variant)
            if not np.array_equal(pair['output'], bad_out):
                bad = {s: [dict(p) for p in base[s]] for s in base}
                bad['train'][index] = dict(input=pair['input'], output=bad_out)
                try:
                    validate_generated_puzzle(bad)
                except AssertionError:
                    break
                raise AssertionError("Incorrect phase output accepted")
        else:
            raise AssertionError("Missing diagnostic phase witness")
    pair = base['test'][0]
    bad = {s: [dict(p) for p in base[s]] for s in base}
    modified = pair['output'].copy()
    info = _decode(pair['input'])
    assembly = _assemble(info)
    j = assembly['order'][0]
    pin = min(info['pieces'][j]['outpins'])
    modified[pin] = info['bg']
    bad['test'][0] = dict(input=pair['input'], output=modified)
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Erased docking pin accepted")
    metrics.update(shapes=len(shapes), body_masks=len(masks), role_palettes=len(roles),
                   key_geometries=len(keys), word_patterns=len(patterns))
    return dict(metrics)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
