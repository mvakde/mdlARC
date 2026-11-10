"""Routed seams: flood-defined payload partitions move to entry-edge corners.

Only NumPy and the standard library are used. No parent files or generation
metadata are consulted by solve. Invalid/ambiguous scenes raise ValueError.
"""
from __future__ import annotations

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger cells")
    _require(np.all((a >= 0) & (a <= 9)), "invalid colors")
    return a.astype(np.int8, copy=True)


def _neighbors(p):
    r, c = p
    return ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1))


def _components(cells):
    todo, result = set(cells), []
    while todo:
        start = min(todo)
        todo.remove(start)
        found, stack = {start}, [start]
        while stack:
            for q in _neighbors(stack.pop()):
                if q in todo:
                    todo.remove(q)
                    found.add(q)
                    stack.append(q)
        result.append(found)
    return result


def _box(cells):
    rs, cs = zip(*cells)
    return min(rs), max(rs), min(cs), max(cs)


def _step(p, d):
    return p[0] + DIRS[d][0], p[1] + DIRS[d][1]


def _decode(a):
    h, w = a.shape
    bg = int(a[0, 0])
    _require(all(a[p] == bg for p in ((0, w - 1), (h - 1, 0), (h - 1, w - 1))),
             "input corners must be background")
    comps = _components(map(tuple, np.argwhere(a != bg)))
    launches = [s for s in comps if any(r in (0, h - 1) or c in (0, w - 1) for r, c in s)]
    _require(len(launches) == 1, "need one boundary launcher")
    launcher = launches[0]
    boundary = [p for p in launcher if p[0] in (0, h - 1) or p[1] in (0, w - 1)]
    _require(len(boundary) == 1, "launcher touches exactly one edge cell")
    start = boundary[0]
    initial = 2 if start[0] == 0 else 0 if start[0] == h - 1 else 1 if start[1] == 0 else 3
    line = [start]
    for _ in range(len(launcher) - 1):
        line.append(_step(line[-1], initial))
    _require(3 <= len(line) <= 5 and set(line) == launcher, "launcher must be straight")
    signal, cap = int(a[line[0]]), int(a[line[-1]])
    _require(signal != cap and all(a[p] == signal for p in line[:-1]), "tail/cap convention")
    payloads, relays = [], []
    for s in comps:
        if s == launcher:
            continue
        colors, counts = np.unique([a[p] for p in sorted(s)], return_counts=True)
        if len(colors) == 1:
            payloads.append((int(colors[0]), s))
            continue
        _require(len(colors) == 2 and counts[0] != counts[1], "relay has body and minority port")
        port_color = int(colors[np.argmin(counts)])
        carrier = int(colors[np.argmax(counts)])
        port = {p for p in s if a[p] == port_color}
        body = s - port
        _require(1 <= len(port) <= 2 and len(_components(body)) == 1, "invalid relay port/body")
        r0, r1, c0, c1 = _box(s)
        sides = [d for d, ok in enumerate((all(r == r0 for r, c in port),
                 all(c == c1 for r, c in port), all(r == r1 for r, c in port),
                 all(c == c0 for r, c in port))) if ok]
        _require(len(sides) == 1, "port must indicate one extreme side")
        d = sides[0]
        span = sorted(c if d % 2 == 0 else r for r, c in port)
        _require(span == list(range(span[0], span[0] + len(span))), "port is contiguous")
        _require(all(_step(p, (d + 2) % 4) in body for p in port), "port needs inward body neighbors")
        relays.append(dict(cells=s, port=port, direction=d, carrier=carrier, mark=port_color))
    _require(len(payloads) == 1 and relays, "need one connected monochrome payload and relays")
    payload_color, payload = payloads[0]
    carriers, marks = {o['carrier'] for o in relays}, {o['mark'] for o in relays}
    _require(len(carriers) == len(marks) == 1, "relays share body/port roles")
    _require(len({bg, signal, cap, payload_color, *carriers, *marks}) == 6, "six distinct visible roles")
    return dict(grid=a, bg=bg, signal=signal, cap=cap, payload_color=payload_color,
                payload=payload, relays=relays, launcher=launcher, start=start, initial=initial,
                cap_cell=line[-1])


def _route(info):
    """Payload is transparent. Every contact is with an ORIGINAL relay."""
    h, w = info['grid'].shape
    objects = info['relays']
    owner = {p: i for i, o in enumerate(objects) for p in o['cells']}
    trace, reached, widths, directions, gaps = set(info['launcher']), [], [], [], []
    d, front, gap = info['initial'], [_step(info['cap_cell'], info['initial'])], 0
    # Every activation is unique, and each straight segment takes <=30 steps.
    for _ in range(31 * (len(objects) + 1)):
        inside = [0 <= r < h and 0 <= c < w for r, c in front]
        if not any(inside):
            return trace, reached, widths, directions, gaps
        _require(all(inside), "partial beam exit")
        hits = {owner[p] for p in front if p in owner}
        if hits:
            _require(len(hits) == 1, "tied relay collision")
            i = next(iter(hits))
            _require(all(owner.get(p) == i for p in front), "partial-width contact")
            _require(i not in reached, "relay cycle or reactivation")
            o = objects[i]
            _require(o['direction'] != (d + 2) % 4, "reverse emission")
            reached.append(i)
            widths.append(len(o['port']))
            directions.append((d, o['direction']))
            gaps.append(gap)
            trace.update(o['cells'])
            d, gap = o['direction'], 0
            front = sorted(_step(p, d) for p in o['port'])
        else:
            trace.update(front)
            front = [_step(p, d) for p in front]
            gap += 1
    raise ValueError("unterminated route")


def _entry_corners(shape, d):
    h, w = shape
    return (((h - 1, 0), (h - 1, w - 1)) if d == 0 else
            ((0, 0), (h - 1, 0)) if d == 1 else
            ((0, 0), (0, w - 1)) if d == 2 else
            ((0, w - 1), (h - 1, w - 1)))


def _analyze(input_grid):
    info = _decode(_grid(input_grid))
    trace, reached, widths, turns, gaps = _route(info)
    h, w = info['grid'].shape
    free = {(r, c) for r in range(h) for c in range(w)} - trace
    regions = _components(free)  # Payload and unreached relays are traversable here.
    _require(len(regions) == 2, "the seam must separate exactly two regions")
    corners = _entry_corners((h, w), info['initial'])
    groups, region_sets = [], []
    for corner in corners:
        matches = [s for s in regions if corner in s]
        _require(len(matches) == 1, "entry corner is on seam")
        region_sets.append(matches[0])
        groups.append(info['payload'] & matches[0])
    _require(region_sets[0] != region_sets[1], "entry corners must lie on opposite sides")
    _require(all(groups), "both seam sides need payload")
    fixed = info['payload'] & trace
    _require(fixed, "need payload on the seam")
    info.update(trace=trace, reached=reached, widths=widths, turns=turns, gaps=gaps,
                groups=groups, corners=corners, fixed=fixed)
    return info


def _translated(cells, corner):
    r0, r1, c0, c1 = _box(cells)
    dr = -r0 if corner[0] == 0 else corner[0] - r1
    dc = -c0 if corner[1] == 0 else corner[1] - c1
    return {(r + dr, c + dc) for r, c in cells}


def _render(info, mode='correct', strict=True):
    out = np.full_like(info['grid'], info['bg'])
    for p in info['trace']:
        out[p] = info['signal']
    for p in info['fixed']:
        out[p] = info['payload_color']
    groups = info['groups']
    if mode == 'straight_partition':
        axis = 1 if info['initial'] % 2 == 0 else 0
        coordinate = info['start'][axis]
        moving = info['payload'] - info['fixed']
        groups = [{p for p in moving if p[axis] < coordinate},
                  {p for p in moving if p[axis] >= coordinate}]
    if mode == 'leave_payload':
        for p in info['payload']:
            out[p] = info['payload_color']
        return out
    if mode == 'erase_on_seam':
        for p in info['fixed']:
            out[p] = info['signal']
    targets = []
    for cells, corner in zip(groups, info['corners']):
        pieces = _components(cells) if mode == 'separate_fragments' else [cells]
        for piece in pieces:
            if not piece:
                continue
            target = _translated(piece, corner)
            if strict:
                _require(not target & info['trace'], "translated payload collides with seam")
                _require(not any(target & other for other in targets), "translated partitions overlap")
            targets.append(target)
            for p in target:
                out[p] = info['payload_color']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Trace the route, flood its complement, and translate whole side groups."""
    return _render(_analyze(input_grid))


def _relay(rng, r, c, h, w, outgoing, width, incoming, lanes):
    cells = {(rr, cc) for rr in range(r, r + h) for cc in range(c, c + w)}
    span_start = r + 1 if outgoing % 2 else c + 1
    port = ({(span_start + k, c + (w - 1 if outgoing == 1 else 0)) for k in range(width)}
            if outgoing % 2 else
            {(r + (h - 1 if outgoing == 2 else 0), span_start + k) for k in range(width)})
    # Clip optional corners without removing any entering lane or port neighbor.
    for p in ((r, c), (r, c + w - 1), (r + h - 1, c), (r + h - 1, c + w - 1)):
        protected = p[1] in lanes if incoming % 2 == 0 else p[0] in lanes
        if not protected and rng.random() < .4:
            cells.remove(p)
    return dict(cells=cells, port=port, direction=outgoing, incoming=incoming, lanes=lanes)


def _layout(rng, n, widths):
    """Construct a simple stair seam; randomize its spacing and relay masks."""
    h = int(rng.integers(28, 31)) if n >= 3 else int(rng.integers(22, 31))
    w = int(rng.integers(26, 31))
    r0 = int(rng.integers(12, 15 if n >= 3 else min(18, h - 7)))
    c0 = int(rng.integers(10, w - 13))
    first_h = max(3, widths[0] + 2)
    first_w = int(rng.integers(3, 5))
    lane = c0 + 1
    objects = [_relay(rng, r0, c0, first_h, first_w, 1, widths[0], 2, [lane])]
    if n >= 2:
        c2 = w - int(rng.integers(7, 9))
        r2 = r0
        second_h = max(3, widths[0] + 2)
        second_w = max(3, widths[1] + 2)
        objects.append(_relay(rng, r2, c2, second_h, second_w, 2, widths[1], 1,
                              list(range(r0 + 1, r0 + 1 + widths[0]))))
    if n >= 3:
        r3 = r2 + second_h + int(rng.integers(3, 6))
        third_h = max(3, widths[2] + 2)
        third_w = max(3, widths[1] + 2)
        objects.append(_relay(rng, r3, c2, third_h, third_w, 3, widths[2], 2,
                              list(range(c2 + 1, c2 + 1 + widths[1]))))
    if n >= 4:
        c4 = int(rng.integers(4, 8))
        fourth_h = max(3, widths[2] + 2)
        fourth_w = max(3, widths[3] + 2)
        objects.append(_relay(rng, r3, c4, fourth_h, fourth_w, 2, widths[3], 3,
                              list(range(r3 + 1, r3 + 1 + widths[2]))))
    length = int(rng.integers(2, 5))
    tail = {(r, lane) for r in range(length)}
    return h, w, tail, (length, lane), objects


def _payload(rng, target, objects, comb, winding):
    r0, r1, c0, c1 = _box(target['cells'])
    # A loop can cross both legs of a bend; a bridge adds detached cut fragments.
    outer = (r0 - 2, r1 + 2, c0 - 2, c1 + 2)
    a, b, x, y = outer
    ring = ({(r, c) for r in range(a, b + 1) for c in (x, y)} |
            {(r, c) for r in (a, b) for c in range(x, y + 1)})
    cells = set(ring)
    if comb:
        lanes = target['lanes']
        lo, hi = min(lanes), max(lanes)
        d = target['incoming']
        if d in (0, 2):
            row = r0 - 4 if d == 2 else r1 + 4
            near = r0 - 2 if d == 2 else r1 + 2
            cells |= {(r, lo - 1) for r in range(min(row, near), max(row, near) + 1)}
            cells |= {(row, c) for c in range(lo - 1, hi + 2)}
        else:
            col = c0 - 4 if d == 1 else c1 + 4
            near = c0 - 2 if d == 1 else c1 + 2
            cells |= {(lo - 1, c) for c in range(min(col, near), max(col, near) + 1)}
            cells |= {(r, col) for r in range(lo - 1, hi + 2)}
    bounds = _box(cells)
    forbidden = {q for o in objects for r, c in o['cells']
                 for q in ((r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))}
    allowed = {(r, c) for r in range(bounds[0], bounds[1] + 1)
               for c in range(bounds[2], bounds[3] + 1)} - forbidden
    if winding and not comb:
        # Frontier growth, rather than a required loop, also produces irregular
        # trees, thin bridges, filled silhouettes and incomplete rings.
        choices = sorted(allowed)
        if not choices:
            raise ValueError('no payload space')
        cells = {choices[int(rng.integers(len(choices)))]}
        target_size = int(rng.integers(20, max(21, min(65, len(allowed)) + 1)))
    else:
        _require(cells <= allowed, 'payload ring too close to another relay')
        target_size = len(cells) + int(rng.integers(0, 19))
    for _ in range(target_size - len(cells)):
        frontier = sorted({q for p in cells for q in _neighbors(p) if q in allowed and q not in cells})
        if not frontier:
            break
        cells.add(frontier[int(rng.integers(len(frontier)))])
    return cells


def _make_pair(rng, n, widths, roles, rotation, mirror, witness, decoys):
    for attempt in range(96):
        try:
            h, w, tail, cap, objects = _layout(rng, n, widths)
            target_index = int(rng.integers(n))
            # The last attempts retain complexity and use the guaranteed first-bend bridge.
            if attempt >= 80:
                target_index = 0
            payload = _payload(rng, objects[target_index], objects, witness or rng.random() < .45,
                               not witness and attempt < 80 and rng.random() < .6)
            _require(all(0 < r < h - 1 and 0 < c < w - 1 for r, c in payload), 'payload reaches border')
            _require(max(_box(payload)[1] - _box(payload)[0], _box(payload)[3] - _box(payload)[2]) < 11,
                     'payload exceeds corner reserve')
            bg, signal, cap_color, carrier, port_color, payload_color = roles
            a = np.full((h, w), bg, dtype=np.int8)
            for o in objects:
                for p in o['cells']:
                    a[p] = port_color if p in o['port'] else carrier
            for p in tail:
                a[p] = signal
            a[cap] = cap_color
            for p in payload:
                a[p] = payload_color
            info = _analyze(a)
            _require(len(info['reached']) == n, 'intended route lost')
            if witness:
                _require(max(len(_components(s)) for s in info['groups']) >= 2, 'fragment witness lost')
                expected = _render(info)
                _require(not np.array_equal(expected, _render(info, 'straight_partition', False)),
                         'missing topology witness')
                _require(not np.array_equal(expected, _render(info, 'separate_fragments', False)),
                         'missing collective movement witness')
            # Decoys may be behind a hit or elsewhere; they must not alter the seam.
            for j in range(decoys):
                placed = False
                for trial in range(120):
                    dh, dw = map(int, rng.integers(3, 7, size=2))
                    if trial == 0 and j == 0:
                        rr, cc = _box(objects[0]['cells'])[1] + 4, info['start'][1] - 1
                    else:
                        rr, cc = int(rng.integers(2, h - dh)), int(rng.integers(2, w - dw))
                    if rr + dh >= h - 1 or cc < 1 or cc + dw >= w - 1:
                        continue
                    o = _relay(rng, rr, cc, dh, dw, int(rng.integers(4)), 1, 2, [])
                    expanded = {(r + dr, c + dc) for r, c in o['cells']
                                for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
                    if any(a[p] != bg for p in expanded if 0 <= p[0] < h and 0 <= p[1] < w):
                        continue
                    if o['cells'] & info['trace']:
                        continue
                    for p in o['cells']:
                        a[p] = port_color if p in o['port'] else carrier
                    placed = True
                    break
                _require(placed, 'could not place required decoy')
            a = np.fliplr(a).copy() if mirror else a
            a = np.rot90(a, rotation).copy()
            final_info = _analyze(a)
            _require(len(final_info['reached']) == n, 'decoy changed route')
            return {'input': a, 'output': _render(final_info)}
        except ValueError:
            continue
    raise RuntimeError('bounded scene construction failed without lowering requested complexity')


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    return int(value)


def _key(a):
    return tuple(a.shape), a.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    _require(num_train > num_test, 'num_train must exceed num_test')
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_)) and seed >= 0,
             'seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    base = list(map(int, rng.permutation(10)))
    vary_palette, fixed_side = rng.random() < .7, rng.random() < .3
    side = int(rng.integers(4))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            if split == 'train':
                n = 1 if i == 0 else 2 if i == 1 else int(rng.integers(1, 3))
                widths = [1] if i == 0 else [2, 1] if i == 1 else list(map(int, rng.integers(1, 3, size=n)))
            else:
                n = int(rng.integers(3, 5))
                widths = list(map(int, rng.integers(1, 3, size=n)))
            if vary_palette:
                palette = base if split == 'train' and i == 0 else list(np.roll(base, 1)) if split == 'train' and i == 1 else list(map(int, rng.permutation(10)))
            else:
                palette = base
            rotation = side if fixed_side else int(rng.integers(4))
            mirror = bool(rng.integers(2))
            decoys = 1 if split == 'train' and i == 1 else int(rng.integers(3))
            for _ in range(16):
                pair = _make_pair(rng, n, widths, palette[:6], rotation, mirror,
                                  split == 'train' and i == 1, decoys)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('could not produce distinct inputs')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(type(puzzle) is dict and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits are lists')
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, 'illegal counts')
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(type(pair) is dict and set(pair) == {'input', 'output'}, 'pair schema')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape and _key(a) not in seen, 'shape or duplicate input')
            seen.add(_key(a))
            info = _analyze(a)
            expected = _render(info)
            _require(np.array_equal(expected, b), 'incorrect output')
            _require(not np.array_equal(a, b), 'identity pair')
            _require(set(map(int, np.unique(b))) == {info['bg'], info['signal'], info['payload_color']},
                     'output role leakage')
            _require(np.count_nonzero(b == info['payload_color']) == len(info['payload']), 'payload conservation')
            _require(all(b[p] == info['payload_color'] for p in info['fixed']), 'on-seam payload moved')
            _require(all(b[p] == info['signal'] for p in info['trace'] - info['fixed']), 'seam rendering')
            # Six input roles distinguish every generated pair from the 3-role A
            # and 5-role selected/original B pairs; no source fixtures are loaded.
            _require(len(np.unique(a)) == 6, 'input role count')
            info['fragmented'] = max(len(_components(s)) for s in info['groups']) >= 2
            info['topology_matters'] = not np.array_equal(b, _render(info, 'straight_partition', False))
            info['collective_matters'] = not np.array_equal(b, _render(info, 'separate_fragments', False))
            _require(not np.array_equal(b, _render(info, 'leave_payload', False)), 'A omission unchanged')
            records[split].append(info)
    train, test = records['train'], records['test']
    _require(any(f['fragmented'] and f['topology_matters'] and f['collective_matters'] for f in train),
             'training needs joint topology and collective-translation witness')
    _require(any(len(f['relays']) > len(f['reached']) for f in train), 'training needs decoy erasure')
    _require({1, 2} <= {w for f in train for w in f['widths']}, 'training needs both port widths')
    _require(any(len(f['reached']) >= 2 for f in train), 'training needs chaining')
    _require(all(set(f['widths']) <= {w for t in train for w in t['widths']} for f in test), 'undemonstrated width')
    _require(min(len(f['reached']) for f in test) > max(len(f['reached']) for f in train),
             'missing route-length generalization')


def self_test():
    """Bounded 42-seed diversity audit plus legal-count and equivariance checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures, structures, dimensions, directions, role_colors = set(), set(), set(), set(), [set() for _ in range(6)]
    active, decoys, sizes, fragmentation, payload_counts = set(), set(), set(), set(), set()
    witness_count = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a, b = pair['input'], pair['output']
                info = _analyze(a)
                assert np.array_equal(solve(a.tolist()), b)
                dimensions.add(a.shape)
                directions.add(info['initial'])
                active.add(len(info['reached']))
                decoys.add(len(info['relays']) - len(info['reached']))
                payload_counts.add(len(info['payload']))
                fragmentation.update(len(_components(s)) for s in info['groups'])
                sizes.update((r1-r0+1, c1-c0+1) for r0, r1, c0, c1 in map(lambda o: _box(o['cells']), info['relays']))
                roles = [info['bg'], info['signal'], info['cap'], info['relays'][0]['carrier'],
                         info['relays'][0]['mark'], info['payload_color']]
                for bucket, color in zip(role_colors, roles):
                    bucket.add(color)
                lookup = {color: i for i, color in enumerate(roles)}
                structures.add((a.shape, tuple(lookup[int(v)] for v in a.flat)))
                witness_count += not np.array_equal(b, _render(info, 'straight_partition', False))
    assert len(signatures) == 42 and len(structures) > 150
    assert len(dimensions) >= 25 and directions == set(range(4))
    assert active == {1, 2, 3, 4} and decoys == {0, 1, 2}
    assert len(sizes) >= 8 and len(payload_counts) >= 20
    assert max(fragmentation) >= 2 and witness_count >= 42
    assert all(len(colors) >= 8 for colors in role_colors)
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(90210 + nt * 10 + ns, nt, ns)
            assert (len(p['train']), len(p['test'])) == (nt, ns)
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    assert all(_key(a[k]) == _key(b[k]) for s in ('train', 'test')
               for a, b in zip(puzzles[0][s], generate_puzzle(0)[s]) for k in ('input', 'output'))
    pair = puzzles[0]['train'][0]
    for k in range(4):
        assert np.array_equal(solve(np.rot90(pair['input'], k)), np.rot90(pair['output'], k))
    assert np.array_equal(solve(np.fliplr(pair['input'])), np.fliplr(pair['output']))
    permutation = np.random.default_rng(818).permutation(10)
    assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
    # Hand-specified valid scenes outside the stair sampler: straight emission
    # and a complete miss. The expected masks below are explicit translations,
    # not another invocation of the generator or its rendering helpers.
    small = np.zeros((16, 16), dtype=np.int8)
    small[0:2, 7], small[2, 7] = 2, 3
    payload_cells = {(4, 5), (4, 6), (4, 7), (4, 8), (4, 9),
                     (5, 5), (5, 7), (5, 9), (6, 5), (6, 6),
                     (6, 7), (6, 8), (6, 9)}
    for cell in payload_cells:
        small[cell] = 6
    explicit = np.zeros_like(small)
    explicit[:, 7] = 2
    for cell in ((4, 7), (5, 7), (6, 7), (0, 0), (0, 1), (1, 0),
                 (2, 0), (2, 1), (0, 14), (0, 15), (1, 15), (2, 14), (2, 15)):
        explicit[cell] = 6
    miss = small.copy()
    miss[9:12, 10:13], miss[11, 11] = 5, 4
    assert np.array_equal(solve(miss), explicit)
    straight = small.copy()
    straight[8:11, 6:9], straight[10, 7] = 5, 4
    explicit[8:11, 6:9] = 2
    assert np.array_equal(solve(straight), explicit)
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ('train', 'test')}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    return dict(seeds=42, pairs=sum(len(p[s]) for p in puzzles for s in ('train', 'test')),
                structures=len(structures), shapes=len(dimensions), active_counts=sorted(active),
                decoy_counts=sorted(decoys), relay_sizes=len(sizes), payload_counts=len(payload_counts),
                topology_counterexamples=int(witness_count))


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
