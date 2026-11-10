"""Recruitable relays: first contact installs a keyed, rotatable outlet.

Only NumPy and the standard library are used. No files, seed, demonstrations,
or generation metadata are consulted by solve. All coordinates are (row, col).
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
NEIGHBORS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
HOST_SEEDS = (0, 294456548, 48292877, 644051770, 72934351, 283745054,
              680744786, 147166269, 366136900, 450992962, 80203053,
              108578921, 130279465, 202638235, 570832768, 935886584,
              378182323, 620640856, 90801311, 824497385, 875563290,
              335597499, 105853294, 665198778, 679359187, 777195222,
              114346871, 765816288, 278612435, 479667644, 469255904,
              155443436, 510159947, 664742023, 73136720, 601981903,
              90063779, 518860182, 66496864, 658063927, 870780706, 642597408)


def _step(p, direction, distance=1):
    dr, dc = DIRS[direction]
    return p[0] + dr * distance, p[1] + dc * distance


def _translate(cells, offset):
    return {(r + offset[0], c + offset[1]) for r, c in cells}


def _expanded(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _inside(cells, shape, margin=0):
    return all(margin <= r < shape[0] - margin and
               margin <= c < shape[1] - margin for r, c in cells)


def _components(cells):
    remaining, result = set(cells), []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, component = [p], {p}
        while stack:
            r, c = stack.pop()
            for dr, dc in NEIGHBORS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    stack.append(q)
        result.append(component)
    return result


def _ring(r=0, c=0):
    return {(r + a, c + b) for a in range(4) for b in range(4)
            if a in (0, 3) or b in (0, 3)}


def _socket(side, r=0, c=0, outside=False):
    d = 1 if outside else 0
    return ({(r - d, c + 1), (r - d, c + 2)},
            {(r + 1, c + 3 + d), (r + 2, c + 3 + d)},
            {(r + 3 + d, c + 1), (r + 3 + d, c + 2)},
            {(r + 1, c - d), (r + 2, c - d)})[side]


def _port_side(part):
    h, w = part.shape
    ports = set(map(tuple, np.argwhere(part == 2)))
    assert 1 <= len(ports) <= 2
    sides = [s for s, yes in enumerate((all(r == 0 for r, c in ports),
             all(c == w - 1 for r, c in ports),
             all(r == h - 1 for r, c in ports), all(c == 0 for r, c in ports))) if yes]
    assert len(sides) == 1, 'outlet must identify exactly one extreme side'
    side = sides[0]
    span = sorted(c if side % 2 == 0 else r for r, c in ports)
    assert span == list(range(span[0], span[0] + len(span)))
    assert all(part[_step(p, (side + 2) % 4)] == 1 for p in ports)
    return side


def _canonicalize(part):
    """The sole centered two-body-cell extreme edge is the inward neck."""
    assert len(_components(map(tuple, np.argwhere(part != 0)))) == 1
    candidates = []
    for turn in range(4):
        a = np.rot90(part, turn).copy()
        h, w = a.shape
        if h < 3 or w < 4 or w % 2:
            continue
        neck = np.zeros(w, dtype=np.int8)
        neck[w // 2 - 1:w // 2 + 1] = 1
        if not np.array_equal(a[-1], neck):
            continue
        try:
            side = _port_side(a)
        except (AssertionError, IndexError):
            continue
        if side != 2:
            candidates.append((a, turn, side))
    assert len(candidates) == 1, 'neck orientation must be unique'
    return candidates[0]


def _dock(part, side, r, c):
    neck = np.zeros(part.shape, dtype=bool)
    neck[-1, part.shape[1] // 2 - 1:part.shape[1] // 2 + 1] = True
    a, neck = np.rot90(part, -side), np.rot90(neck, -side)
    target = np.array(sorted(_socket(side, r, c, True)))
    shift = target.min(axis=0) - np.argwhere(neck).min(axis=0)
    cells = {tuple(map(int, p + shift)) for p in np.argwhere(a != 0)}
    ports = {tuple(map(int, p + shift)) for p in np.argwhere(a == 2)}
    assert _translate(set(map(tuple, np.argwhere(neck))), shift) == set(map(tuple, target))
    return cells, ports


def _array(grid):
    a = np.asarray(grid)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _decode(input_grid):
    a = _array(input_grid)
    h, w = a.shape
    bg = int(a[0, 0])
    assert all(int(a[p]) == bg for p in ((0, w - 1), (h - 1, 0), (h - 1, w - 1)))
    assert np.count_nonzero(a == bg) > a.size // 2
    components = _components(map(tuple, np.argwhere(a != bg)))
    boundary = [x for x in components if not _inside(x, a.shape, 1)]
    assert len(boundary) == 1, 'exactly one boundary launcher'
    launcher = boundary[0]
    edge = [p for p in launcher if not _inside({p}, a.shape, 1)]
    assert len(edge) == 1
    start = edge[0]
    direction = 2 if start[0] == 0 else 0 if start[0] == h - 1 else 1 if start[1] == 0 else 3
    line = [_step(start, direction, k) for k in range(len(launcher))]
    assert 3 <= len(line) <= 5 and set(line) == launcher
    signal, marker = int(a[start]), int(a[line[-1]])
    assert signal != marker and all(int(a[p]) == signal for p in line[:-1])
    stations, parts, frame_colors = [], {}, set()
    for component in components:
        if component == launcher:
            continue
        colors = {int(a[p]) for p in component}
        assert len(colors) == 2
        r0, c0 = min(r for r, c in component), min(c for r, c in component)
        if marker in colors:
            key = next(iter(colors - {marker}))
            assert key not in parts, 'one loose part per key'
            rh, cw = max(r for r, c in component) - r0 + 1, max(c for r, c in component) - c0 + 1
            mask = np.zeros((rh, cw), dtype=np.int8)
            for r, c in component:
                mask[r - r0, c - c0] = 2 if int(a[r, c]) == marker else 1
            canonical, undo, port_side = _canonicalize(mask)
            parts[key] = dict(cells=component, canonical=canonical, undo=undo,
                              source_shape=mask.shape, port_side=port_side)
        else:
            assert component == _ring(r0, c0), 'station must be a hollow 4 by 4 frame'
            counts = {color: sum(int(a[p]) == color for p in component) for color in colors}
            assert sorted(counts.values()) == [2, 10]
            key = next(color for color in colors if counts[color] == 2)
            frame_color = next(iter(colors - {key}))
            sockets = [s for s in range(4) if {p for p in component if int(a[p]) == key} == _socket(s, r0, c0)]
            assert len(sockets) == 1
            stations.append(dict(cells=component, key=key, side=sockets[0], r=r0, c=c0))
            frame_colors.add(frame_color)
    assert 1 <= len(stations) <= 4 and len(frame_colors) == 1
    frame = next(iter(frame_colors))
    keys = {s['key'] for s in stations}
    assert len(keys) == len(stations) and keys <= set(parts)
    assert 0 <= len(parts) - len(stations) <= 1
    assert not set(parts) & {bg, signal, marker, frame}
    assert len({bg, signal, marker, frame}) == 4
    potential, used = [], set()
    for station in stations:
        part = parts[station['key']]
        cells, ports = _dock(part['canonical'], station['side'], station['r'], station['c'])
        whole = cells | station['cells']
        assert _inside(cells, a.shape) and not (whole & used)
        assert not (cells & station['cells']) and not (whole & launcher)
        used |= whole
        potential.append(dict(cells=cells, ports=ports, whole=whole,
                              direction=(part['port_side'] + station['side']) % 4,
                              rotation=(part['undo'] - station['side']) % 4,
                              canonical=part['canonical']))
    return dict(grid=a, bg=bg, signal=signal, marker=marker, frame=frame,
                tail=set(line[:-1]), cap=line[-1], initial=direction,
                stations=stations, parts=parts, potential=potential)


def _run(data):
    """Operate on the decoded original frames; stock parts are collected first."""
    shape = data['grid'].shape
    owner = {p: i for i, s in enumerate(data['stations']) for p in s['cells']}
    trace = data['tail'] | {data['cap']}
    rays, installed, reached, transitions, gaps = set(), set(), [], [], []
    origins, direction = {data['cap']}, data['initial']
    for _ in range(len(data['stations']) + 1):
        for distance in range(1, 32):
            front = {_step(p, direction, distance) for p in origins}
            if not _inside(front, shape):
                assert all(not _inside({p}, shape) for p in front)
                output = np.full(shape, data['bg'], dtype=np.int8)
                for p in trace:
                    output[p] = data['signal']
                return output, dict(reached=reached, transitions=transitions, gaps=gaps,
                                    trace=trace, rays=rays, installed=installed)
            assert not front & installed, 'ray must not re-enter an installed part'
            hits = {owner[p] for p in front if p in owner}
            if hits:
                assert len(hits) == 1
                index = next(iter(hits))
                assert all(owner.get(p) == index for p in front), 'full-width first contact required'
                assert index not in reached, 'cycles are outside the family'
                assembly = data['potential'][index]
                assert not assembly['cells'] & trace, 'installation cannot overwrite an earlier ray'
                reached.append(index)
                gaps.append(distance - 1)
                transitions.append((direction, assembly['direction'], len(assembly['ports'])))
                trace |= assembly['whole']
                installed |= assembly['cells']
                origins, direction = assembly['ports'], assembly['direction']
                break
            trace |= front
            rays |= front
        else:
            raise AssertionError('ray failed to leave a 30-cell axis')
    raise AssertionError('unterminated route')


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all roles, installations, rotations, and first hits from input."""
    return _run(_decode(input_grid))[0]


def _make_part(rng, outlet_side, outlet_width, width=None, stem=None):
    assert outlet_side in (0, 1, 3)
    for _ in range(90):
        w = int(rng.choice((4, 6), p=(.75, .25))) if width is None else width
        nstem = int(rng.integers(1, 3)) if stem is None else stem
        body = int(rng.integers(2, 5))
        h, middle = body + nstem, w // 2
        part = np.zeros((h, w), dtype=np.int8)
        part[:, middle - 1:middle + 1] = 1
        style = int(rng.integers(4))
        for r in range(body):
            left = 0 if style < 2 else int(rng.integers(0, middle))
            right = w if style < 2 else int(rng.integers(middle + 1, w + 1))
            part[r, left:right] = 1
        part[body - 1] = 1
        if style == 1:
            part[0, int(rng.choice((0, w - 1)))] = 0
        if style == 3 and body >= 3:
            # A real hole or asymmetric notch, preserving the central stem.
            r = int(rng.integers(body - 1))
            c = int(rng.choice([j for j in range(w) if j not in (middle - 1, middle)]))
            part[r, c] = 0
        if not np.any(part[:, 0]) or not np.any(part[:, -1]) or not np.any(part[0]):
            continue
        options = []
        if outlet_side == 0:
            options = [{(0, start + k) for k in range(outlet_width)}
                       for start in range(1, w - outlet_width)]
        else:
            col = 0 if outlet_side == 3 else w - 1
            options = [{(start + k, col) for k in range(outlet_width)}
                       for start in range(1, h - outlet_width)]
        options = [p for p in options if all(part[q] == 1 and
                   part[_step(q, (outlet_side + 2) % 4)] == 1 for q in p)]
        if not options:
            continue
        port = options[int(rng.integers(len(options)))]
        for q in port:
            part[q] = 2
        try:
            canonical, undo, side = _canonicalize(part)
        except AssertionError:
            continue
        if undo == 0 and side == outlet_side:
            return canonical
    return _fallback_part(outlet_side, outlet_width, width, stem)


def _fallback_part(outlet_side, outlet_width, width=None, stem=None):
    # A full chamber always supplies the same requested neck, outlet, and
    # width if random silhouettes exhaust their small bounded search.
    w = 4 if width is None else width
    nstem = 1 if stem is None else stem
    part = np.zeros((3 + nstem, w), dtype=np.int8)
    part[:3] = 1
    part[3:, w // 2 - 1:w // 2 + 1] = 1
    points = ([(0, 1 + k) for k in range(outlet_width)] if outlet_side == 0 else
              [(1 + k, 0 if outlet_side == 3 else w - 1) for k in range(outlet_width)])
    for p in points:
        part[p] = 2
    canonical, undo, side = _canonicalize(part)
    assert undo == 0 and side == outlet_side
    return canonical


def _final_ray(origins, direction, shape, occupied):
    ray = set()
    for distance in range(1, 32):
        front = {_step(p, direction, distance) for p in origins}
        if not _inside(front, shape):
            return ray
        if front & occupied:
            return None
        ray |= front
    return None


def _layout(rng, n, decoys, requirements, long_stem=False):
    total = n + decoys
    low = 16 if total == 1 else 21 if total == 2 else 25
    shape = tuple(map(int, rng.integers(low, 31, size=2)))
    initial = int(rng.integers(4))
    if initial % 2:
        lane = int(rng.integers(3, shape[0] - 3))
        start = (lane, 0 if initial == 1 else shape[1] - 1)
    else:
        lane = int(rng.integers(3, shape[1] - 3))
        start = (0 if initial == 2 else shape[0] - 1, lane)
    length = int(rng.integers(2, 5))
    tail = {_step(start, initial, k) for k in range(length)}
    cap = _step(start, initial, length)
    launcher = tail | {cap}
    budget = [110]

    def grow(stations, occupied, rays, origins, incoming):
        index = len(stations)
        if index == n:
            final = _final_ray(origins, incoming, shape, occupied)
            return None if final is None else (stations, rays | final)
        req = requirements[index]
        clearance = _expanded(occupied)
        for _ in range(18):
            if budget[0] <= 0:
                return None
            budget[0] -= 1
            side = int(rng.integers(4))
            if side == (incoming + 2) % 4:
                continue
            relative = req.get('port', int(rng.choice((0, 1, 3))))
            outgoing = (side + relative) % 4
            if outgoing == (incoming + 2) % 4:
                continue
            try:
                part = _make_part(rng, relative, req['beam'], req.get('width'),
                                  3 if long_stem and index == 0 else None)
            except RuntimeError:
                continue
            choices = []
            gap_order = rng.permutation(np.arange(2, 25))
            for gap in gap_order:
                approach = {_step(p, incoming, k) for p in origins for k in range(1, int(gap))}
                if not _inside(approach, shape) or approach & occupied:
                    continue
                first = min(_step(p, incoming, int(gap)) for p in origins)
                for entry in range(5 - len(origins)):
                    r, c = (first[0] - 3, first[1] - entry) if incoming == 0 else (
                            (first[0] - entry, first[1]) if incoming == 1 else (
                            (first[0], first[1] - entry) if incoming == 2 else
                            (first[0] - entry, first[1] - 3)))
                    ring = _ring(r, c)
                    cells, ports = _dock(part, side, r, c)
                    whole = ring | cells
                    if not _inside(ring, shape, 1) or not _inside(whole, shape):
                        continue
                    if whole & clearance or whole & rays or cells & approach:
                        continue
                    choices.append((r, c, cells, ports, whole, approach))
                if len(choices) >= 8:
                    break
            if not choices:
                continue
            rng.shuffle(choices)
            for r, c, cells, ports, whole, approach in choices[:5]:
                station = dict(r=r, c=c, side=side, part=part, ports=ports,
                               cells=cells, whole=whole, outgoing=outgoing)
                result = grow(stations + [station], occupied | whole, rays | approach,
                              ports, outgoing)
                if result is not None:
                    return result
        return None

    result = grow([], launcher, set(), {cap}, initial)
    if result is None:
        return None
    stations, rays = result
    occupied = launcher | set().union(*(s['whole'] for s in stations))
    for _ in range(decoys):
        clearance = _expanded(occupied)
        for _ in range(180):
            side = int(rng.integers(4))
            relative = int(rng.choice((0, 1, 3)))
            part = _make_part(rng, relative, int(rng.integers(1, 3)))
            r = int(rng.integers(1, shape[0] - 4))
            c = int(rng.integers(1, shape[1] - 4))
            cells, ports = _dock(part, side, r, c)
            whole = cells | _ring(r, c)
            if not _inside(whole, shape) or whole & clearance or whole & rays:
                continue
            stations.append(dict(r=r, c=c, side=side, part=part, ports=ports,
                                 cells=cells, whole=whole, outgoing=(side + relative) % 4))
            occupied |= whole
            break
        else:
            return None
    return shape, tail, cap, stations, rays, initial


def _pack(rng, layout, colors, unmatched, rotations):
    shape, tail, cap, stations, rays, initial = layout
    bg, signal, marker, frame = colors[:4]
    keys = colors[4:]
    inp = np.full(shape, bg, dtype=np.int8)
    occupied = tail | {cap}
    parts = []
    for i, station in enumerate(stations):
        ring = _ring(station['r'], station['c'])
        for p in ring:
            inp[p] = frame
        for p in _socket(station['side'], station['r'], station['c']):
            inp[p] = keys[i]
        occupied |= ring
        turn = int((-station['side'] - rotations[i]) % 4) if i < len(rotations) else int(rng.integers(4))
        parts.append((np.rot90(station['part'], turn), keys[i]))
    if unmatched:
        part = _make_part(rng, int(rng.choice((0, 1, 3))), int(rng.integers(1, 3)))
        parts.append((np.rot90(part, int(rng.integers(4))), keys[len(stations)]))
    for p in tail:
        inp[p] = signal
    inp[cap] = marker
    blocked = _expanded(occupied)
    order = list(map(int, rng.permutation(len(parts))))
    order.sort(key=lambda j: -parts[j][0].size)
    for index in order:
        part, key = parts[index]
        local = set(map(tuple, np.argwhere(part != 0)))
        options = [(r, c) for r in range(1, shape[0] - part.shape[0])
                   for c in range(1, shape[1] - part.shape[1])]
        rng.shuffle(options)
        forbidden = blocked | rays
        for offset in options:
            cells = _translate(local, offset)
            if cells & forbidden:
                continue
            for r, c in local:
                inp[r + offset[0], c + offset[1]] = marker if part[r, c] == 2 else key
            blocked |= _expanded(cells)
            break
        else:
            return None
    return inp


def _pair(rng, n, decoys, unmatched, colors, requirements, rotations,
          long_stem=False, require_turn=False):
    # Counts and witnesses stay fixed while geometry is retried. No simpler
    # specifications are substituted after a difficult placement attempt.
    for _ in range(70):
        layout = _layout(rng, n, decoys, requirements, long_stem)
        if layout is None:
            continue
        inp = _pack(rng, layout, colors, unmatched, rotations)
        if inp is None:
            continue
        try:
            data = _decode(inp)
            out, facts = _run(data)
        except AssertionError:
            continue
        has_turn = any(a != b for a, b, width in facts['transitions'])
        if len(facts['reached']) == n and (has_turn or not require_turn):
            return dict(input=inp, output=out)
    raise RuntimeError('bounded scene construction exhausted without changing its specification')


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain = _count(num_train, 'num_train', 2, 10)
    ntest = _count(num_test, 'num_test', 1, 4)
    if ntrain <= ntest:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    base = list(map(int, rng.permutation(10)))
    vary_palette = bool(rng.integers(2))
    length_shift = bool(rng.integers(2))
    rotation_witnesses = list(map(int, rng.permutation(4)))
    side_port_witnesses = list(map(int, rng.permutation((1, 3))))
    puzzle, seen = dict(train=[], test=[]), set()
    for split, count in (('train', ntrain), ('test', ntest)):
        for index in range(count):
            witness = split == 'train' and index < 2
            if witness:
                n, decoys, unmatched = 2, int(index == 0), int(index == 1)
                widths = list(map(int, rng.permutation((1, 2))))
                port_kinds = [0, side_port_witnesses[index]]
                rng.shuffle(port_kinds)
                requirements = [dict(beam=widths[j], port=port_kinds[j]) for j in range(n)]
                requirements[0]['width'] = 4 if index == 0 else 6
                rotations = rotation_witnesses[2 * index:2 * index + 2]
            else:
                n = 3 if split == 'test' and length_shift else int(rng.choice((0, 1, 2, 3), p=(.08, .23, .40, .29)))
                if split == 'train' and length_shift:
                    n = min(n, 2)
                decoys = int(rng.integers(0 if n else 1, min(2, 4 - n) + 1))
                unmatched = int(rng.integers(2)) if n + decoys < 5 else 0
                requirements = [dict(beam=int(rng.integers(1, 3))) for _ in range(n)]
                rotations = []
            long_stem = split == 'test' and not length_shift
            # A span-shift test contains an installed long stem, never just a decoy.
            if long_stem and n == 0:
                n = 1
                requirements = [dict(beam=int(rng.integers(1, 3)))]
                decoys = min(decoys, 3)
            if vary_palette:
                colors = list(np.roll(base, index)) if split == 'train' and index < 2 else list(map(int, rng.permutation(10)))
            else:
                colors = base.copy()
            for _ in range(5):
                pair = _pair(rng, n, decoys, unmatched, colors, requirements, rotations,
                             long_stem, require_turn=witness)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('failed to generate distinct inputs')
            seen.add(key)
            puzzle[split].append(pair)
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(ntrain)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _stem(part):
    row = np.zeros(part.shape[1], dtype=np.int8)
    row[part.shape[1] // 2 - 1:part.shape[1] // 2 + 1] = 1
    count = 0
    for values in part[::-1]:
        if not np.array_equal(values, row):
            break
        count += 1
    return count


def _facts(pair):
    inp, out = _array(pair['input']), _array(pair['output'])
    assert inp.shape == out.shape
    data = _decode(inp)
    expected, run = _run(data)
    assert np.array_equal(out, expected), 'incorrect recruited assembly or signal route'
    assert set(map(int, np.unique(out))) == {data['bg'], data['signal']}
    reached = run['reached']
    active = [data['potential'][i] for i in reached]
    inactive = [i for i in range(len(data['stations'])) if i not in reached]
    # Input-only invariants and two parent-omission witnesses.
    assert all(not (data['stations'][i]['cells'] & run['trace']) for i in inactive)
    assert all(p['cells'] <= run['trace'] for p in active)
    assert all(not (part['cells'] & run['rays']) for part in data['parts'].values())
    assert not np.array_equal(inp, out)
    signatures = []
    for p in active:
        part = p['canonical']
        signatures.append((part.shape, part.tobytes()))
    return dict(shape=inp.shape, initial=data['initial'], active=len(reached),
                decoys=len(inactive), unmatched=len(data['parts']) - len(data['stations']),
                rotations={p['rotation'] for p in active},
                relative_ports={_port_side(p['canonical']) for p in active},
                widths={len(p['ports']) for p in active},
                body_widths={p['canonical'].shape[1] for p in active},
                stems={_stem(p['canonical']) for p in active},
                socket_sides={data['stations'][i]['side'] for i in reached},
                gaps=run['gaps'], signatures=signatures,
                turns={(b - a) % 4 for a, b, width in run['transitions']},
                bg=data['bg'], palette=(data['bg'], data['signal'], data['marker'], data['frame']))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, facts = set(), dict(train=[], test=[])
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            key = _key(_array(pair['input']))
            assert key not in seen, 'duplicate input'
            seen.add(key)
            facts[split].append(_facts(pair))
    train = facts['train']
    assert any(f['active'] >= 2 for f in train), 'demonstrate causal chaining'
    assert any(f['decoys'] >= 1 for f in train), 'reachable-only assembly witness'
    assert any(f['unmatched'] == 1 for f in train), 'color filtering witness'
    assert set().union(*(f['rotations'] for f in train)) == set(range(4))
    assert set().union(*(f['widths'] for f in train)) == {1, 2}
    assert set().union(*(f['relative_ports'] for f in train)) == {0, 1, 3}
    assert set().union(*(f['body_widths'] for f in train)) == {4, 6}
    assert any(1 in f['turns'] or 3 in f['turns'] for f in train)
    # Each puzzle implements one of two visible geometric recombinations.
    test = facts['test']
    assert (max(f['active'] for f in train) <= 2 and all(f['active'] == 3 for f in test)) or (
            all(max(f['stems'], default=0) <= 2 for f in train) and all(3 in f['stems'] for f in test))


def self_test():
    """42 seeds, streamed; no puzzle dumps, logs, or retained execution traces."""
    shapes, masks, initial, sockets, rotations, ports, widths = set(), set(), set(), set(), set(), set(), set()
    active_counts, decoy_counts, unmatched_counts, stems, turns = set(), set(), set(), set(), set()
    puzzle_hashes, backgrounds, palettes = set(), set(), set()
    longest_gap, length_shifts, stem_shifts = 0, 0, 0
    for side in (0, 1, 3):
        for beam in (1, 2):
            for width in (4, 6):
                for stem in (1, 2, 3):
                    part = _fallback_part(side, beam, width, stem)
                    assert _port_side(part) == side and np.count_nonzero(part == 2) == beam
                    assert _stem(part) == stem and part.shape[1] == width
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        records = {}
        for split in ('train', 'test'):
            records[split] = []
            for pair in puzzle[split]:
                for role in ('input', 'output'):
                    digest.update(repr(pair[role].shape).encode())
                    digest.update(pair[role].tobytes())
                f = _facts(pair)
                records[split].append(f)
                shapes.add(f['shape'])
                masks.update(f['signatures'])
                initial.add(f['initial'])
                sockets.update(f['socket_sides'])
                rotations.update(f['rotations'])
                ports.update(f['relative_ports'])
                widths.update(f['widths'])
                stems.update(f['stems'])
                turns.update(f['turns'])
                active_counts.add(f['active'])
                decoy_counts.add(f['decoys'])
                unmatched_counts.add(f['unmatched'])
                backgrounds.add(f['bg'])
                palettes.add(f['palette'])
                longest_gap = max(longest_gap, max(f['gaps'], default=0))
        length_shifts += all(f['active'] == 3 for f in records['test']) and max(f['active'] for f in records['train']) <= 2
        stem_shifts += all(3 in f['stems'] for f in records['test'])
        puzzle_hashes.add(digest.hexdigest())
    assert len(puzzle_hashes) == 42
    assert len(shapes) >= 25 and len(masks) >= 70
    assert initial == sockets == rotations == set(range(4))
    assert ports == {0, 1, 3} and widths == {1, 2} and stems == {1, 2, 3}
    assert active_counts == {0, 1, 2, 3} and decoy_counts == {0, 1, 2}
    assert unmatched_counts == {0, 1} and {0, 1, 3} <= turns
    assert len(backgrounds) >= 8 and len(palettes) >= 35 and longest_gap >= 10
    assert length_shifts >= 8 and stem_shifts >= 8
    a, b = generate_puzzle(991, 2, 1), generate_puzzle(991, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in ('train', 'test') for x, y in zip(a[s], b[s]) for k in ('input', 'output'))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(73 + nt * 4 + nq, nt, nq)
            assert (len(p['train']), len(p['test'])) == (nt, nq)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in a[s]] for s in ('train', 'test')}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged answer accepted')
    return dict(seeds=42, distinct_puzzles=len(puzzle_hashes), grid_shapes=len(shapes),
                installed_part_masks=len(masks), max_approach_gap=longest_gap,
                chain_extension_witness_puzzles=length_shifts,
                long_stem_witness_puzzles=stem_shifts)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
