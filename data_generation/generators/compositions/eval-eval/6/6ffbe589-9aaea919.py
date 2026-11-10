"""Rotary switchboard: color-layer routing controls multi-source stack addition.

Only NumPy and the standard library are used. No parent files are read at runtime.
Directions are numbered north, east, south, west, in clockwise order.
"""
from __future__ import annotations

import itertools
import numpy as np


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("grids must be integer 2D arrays, size 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _coord(center, direction, radius, tangent=0):
    dr, dc = -radius, tangent
    for _ in range(direction % 4):
        dr, dc = dc, -dr
    return center + dr, center + dc


def _rod(a, center, direction, radius, width, color):
    for t in range(-(width // 2), width // 2 + 1):
        a[_coord(center, direction, radius, t)] = color


def _frame(a, radius, color):
    c = len(a) // 2
    lo, hi = c - radius, c + radius
    a[lo, lo:hi + 1] = a[hi, lo:hi + 1] = color
    a[lo:hi + 1, lo] = a[lo:hi + 1, hi] = color


def _base_board(center, ring, bg, frame, hub):
    a = np.full((2 * center + 1, 2 * center + 1), bg, dtype=np.int8)
    _frame(a, center, frame)
    _frame(a, ring, frame)
    a[center, center] = hub
    return a


def _read_stack(board, bg, direction, start, stop, source):
    c = len(board) // 2
    rods = []
    for radius in range(start, stop + 1):
        points = [_coord(c, direction, radius, t) for t in range(-2, 3)]
        values = np.array([board[p] for p in points])
        if values[2] == bg:
            continue
        # Near the hub another spoke may intersect this sampled row, but its
        # rod is separated from the axis. Read only the axis-centered run.
        lo = hi = 2
        while lo > 0 and values[lo - 1] == values[2]:
            lo -= 1
        while hi < 4 and values[hi + 1] == values[2]:
            hi += 1
        occupied = np.arange(lo, hi + 1)
        width = len(occupied)
        assert width in ((3, 5) if source else (1, 3, 5)), 'invalid rod width'
        assert occupied.tolist() == list(range(2 - width // 2, 3 + width // 2))
        assert len(set(int(v) for v in values[occupied])) == 1
        rods.append((radius, width, int(values[occupied[0]])))
    if not rods:
        return None
    assert len({(w, col) for _, w, col in rods}) == 1, 'stack tokens must repeat'
    radii = [r for r, _, _ in rods]
    assert (radii[0] == start if source else radii[-1] == stop), 'wrong baseline'
    differences = np.diff(radii)
    if source:
        assert len(set(differences.tolist())) <= 1
        assert not len(differences) or int(differences[0]) in (2, 3)
    else:
        assert all(int(d) == 2 for d in differences), 'receiver pitch is two'
        assert radii[0] >= rods[0][1] // 2 + 1, 'receiver touches another spoke'
    return dict(direction=direction, radii=radii, width=rods[0][1],
                color=rods[0][2], count=len(rods))


def _parse(input_grid):
    a = _grid(input_grid)
    values, counts = np.unique(a, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1, 'background must be dominant'
    bg = int(values[counts.argmax()])
    candidates = []
    for color in values:
        if color == bg:
            continue
        points = np.argwhere(a == color)
        r, col = points.min(axis=0).tolist()
        er, ec = (points.max(axis=0) + 1).tolist()
        n = er - r
        if n != ec - col or n % 2 != 1 or not 15 <= n <= 27:
            continue
        b = a[r:er, col:ec]
        if (np.all(b[0] == color) and np.all(b[-1] == color)
                and np.all(b[:, 0] == color) and np.all(b[:, -1] == color)):
            candidates.append((r, col, n, int(color)))
    assert len(candidates) == 1, 'one outer square frame is required'
    r, col, n, frame = candidates[0]
    board = a[r:r + n, col:col + n].copy()
    c = n // 2
    frame_inside = board == frame
    frame_inside[[0, -1], :] = False
    frame_inside[:, [0, -1]] = False
    points = np.argwhere(frame_inside)
    assert len(points)
    low, high = points.min(axis=0), points.max(axis=0)
    ring = c - int(low[0])
    assert 4 <= ring <= c - 2
    assert np.array_equal(low, [c - ring, c - ring])
    assert np.array_equal(high, [c + ring, c + ring])
    hub = int(board[c, c])
    assert len({bg, frame, hub}) == 3
    reconstructed = _base_board(c, ring, bg, frame, hub)
    sources, receivers = [], {}
    for d in range(4):
        s = _read_stack(board, bg, d, ring + 1, c - 1, True)
        t = _read_stack(board, bg, d, 1, ring - 1, False)
        if s:
            sources.append(s)
        if t:
            receivers[d] = t
        for stack in (s, t):
            if stack:
                assert stack['color'] not in (bg, frame, hub)
                for radius in stack['radii']:
                    _rod(reconstructed, c, d, radius, stack['width'], stack['color'])
    assert sources and receivers
    assert np.array_equal(board, reconstructed), 'unexpected board cells or touching rods'
    source_colors = {s['color'] for s in sources}
    assert not source_colors & {s['color'] for s in receivers.values()}
    exterior = a != bg
    exterior[r:r + n, col:col + n] = False
    markers, blocked = {}, set()
    for color in sorted(source_colors):
        points = np.argwhere(exterior & (a == color))
        if not len(points):
            continue
        rows, cols = points.T
        horizontal = np.ptp(rows) == 0 and np.ptp(cols) + 1 == len(points)
        vertical = np.ptp(cols) == 0 and np.ptp(rows) + 1 == len(points)
        assert horizontal or vertical, 'one straight counter per source color'
        assert 1 <= len(points) <= 9
        coords = [tuple(p) for p in points.tolist()]
        for y, x in coords:
            assert not (r - 1 <= y <= r + n and col - 1 <= x <= col + n)
            assert (y, x) not in blocked, 'counters must be detached'
        for y, x in coords:
            blocked.update((y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        markers[color] = dict(length=len(points), coords=coords,
                              orientation='single' if len(points) == 1 else
                              ('horizontal' if horizontal else 'vertical'))
    assert set(int(x) for x in a[exterior]) <= source_colors, 'unmatched counter color'
    for s in sources:
        s['turns'] = markers.get(s['color'], {}).get('length', 0)
        s['target'] = (s['direction'] + s['turns']) % 4
    return dict(board=board, background=bg, frame=frame, hub=hub, center=c,
                ring=ring, sources=sources, receivers=receivers, markers=markers,
                position=(r, col))


def _answer(info, route=None, contribution=None):
    """Coordinate-based addition/recoloring; optional rivals are used in checks."""
    out = info['board'].copy()
    c, receivers = info['center'], info['receivers']
    totals = {d: 0 for d in receivers}
    for s in info['sources']:
        target = s['target'] if route is None else route(s)
        if target not in receivers:
            continue
        totals[target] += s['count'] if contribution is None else contribution(s)
        for radius in s['radii']:
            _rod(out, c, s['direction'], radius, s['width'], info['hub'])
    for d, receiver in receivers.items():
        for k in range(receiver['count'], receiver['count'] + totals[d]):
            radius = info['ring'] - 1 - 2 * k
            assert radius >= receiver['width'] // 2 + 1, 'insufficient inward room'
            _rod(out, c, d, radius, receiver['width'], receiver['color'])
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive routing by rotating each entire source-color mask from the input."""
    info = _parse(input_grid)
    board, c = info['board'], info['center']
    rotations = {}
    for color in {s['color'] for s in info['sources']}:
        mask = board == color
        turns = info['markers'].get(color, {}).get('length', 0)
        rotations[color] = np.rot90(mask, -(turns % 4))
    # A source's innermost rod center identifies the spoke reached by its mask.
    def routed(s):
        radius = s['radii'][0]
        y, x = _coord(c, s['direction'], radius)
        for _ in range(s['turns'] % 4):
            y, x = x, len(board) - 1 - y
        assert rotations[s['color']][y, x]
        if y < c and x == c:
            return 0
        if y == c and x > c:
            return 1
        if y > c and x == c:
            return 2
        assert y == c and x < c
        return 3
    return _answer(info, route=routed)


def _capacity(ring, width):
    return 1 + (ring - 1 - (width // 2 + 1)) // 2


def _sample_descriptor(rng, witness=None, long_counter=False):
    if witness is not None:
        center = int(rng.choice([12, 13]))
        ring = int(rng.integers(8, center - 3))
        # These evidence constraints are only for the two demonstrations.
        # Directions, palettes, widths, other operands, and layouts are sampled.
        kind, first, second = witness
        if kind == 0:
            plans = []
            for t in range(4):
                donor = (t - first) % 4
                for b in [d for d in range(4) if d not in (t, donor)]:
                    a = next(d for d in range(4) if d not in (t, donor, b))
                    q = (a + second) % 4
                    if q not in (t, (b + first) % 4):
                        plans.append((t, donor, b, a, q))
            t, donor, b, a, q = plans[int(rng.integers(len(plans)))]
            source_dirs = [t, donor, b, a]
            ks, heights, color_ids = [0, first, first, second], [1, 2, int(rng.integers(1, 3)), 1], [0, 1, 1, 2]
            destinations = {t: 1, q: int(rng.integers(1, 3))}
            idle = next(d for d in range(4) if d not in (t, q, (b + first) % 4))
            destinations[idle] = 1
        else:
            missing = next(iter({1, 2, 3} - {first, second}))
            plans = []
            for t in range(4):
                donor = (t - missing) % 4
                for a in [d for d in range(4) if d not in (t, donor)]:
                    b = next(d for d in range(4) if d not in (t, donor, a))
                    for odd in (1, 3):
                        q = (a + odd) % 4
                        if q not in (t, b):
                            plans.append((t, donor, a, b, odd, q))
            t, donor, a, b, odd, q = plans[int(rng.integers(len(plans)))]
            source_dirs = [t, donor, a, b]
            ks, heights, color_ids = [4, missing, odd, 0], [1, 1, int(rng.integers(1, 3)), int(rng.integers(1, 3))], [0, 1, 2, 3]
            destinations = {t: 2, q: int(rng.integers(1, 3))}
        source_count = max(color_ids) + 1
        raw = {i: ks[color_ids.index(i)] for i in range(source_count)}
        pitches = [int(rng.integers(2, min(3, center - ring - 2) + 1)) for _ in range(4)]
    else:
        for _ in range(100):
            center = int(rng.integers(7, 14))
            ring = int(rng.integers(4, center - 1))
            count = int(rng.integers(1, 5))
            source_dirs = rng.choice(4, count, replace=False).tolist()
            source_count = int(rng.integers(1, count + 1))
            color_ids = list(range(source_count)) + rng.integers(0, source_count, count - source_count).tolist()
            rng.shuffle(color_ids)
            raw = {i: int(rng.integers(0, 10 if long_counter else 5)) for i in range(source_count)}
            if long_counter:
                forced = int(rng.integers(source_count))
                raw[forced] = int(rng.integers(5, 10))
            pitches = rng.integers(2, 4, count).tolist()
            heights = [int(rng.integers(1, min(3, 1 + (center - ring - 2) // p) + 1)) for p in pitches]
            totals = [0] * 4
            for d, i, h in zip(source_dirs, color_ids, heights):
                totals[(d + raw[i]) % 4] += h
            cap = _capacity(ring, 1)
            eligible = [d for d in range(4) if totals[d] < cap]
            active = [d for d in eligible if totals[d] > 0]
            if long_counter:
                active = [d for d in active if any(i == forced and (s + raw[i]) % 4 == d
                                                  for s, i in zip(source_dirs, color_ids))]
            if not active:
                continue
            port_count = int(rng.integers(1, len(eligible) + 1))
            ports = [int(rng.choice(active))]
            other = [d for d in eligible if d not in ports]
            if port_count > 1:
                ports += rng.choice(other, port_count - 1, replace=False).tolist()
            destinations = {d: int(rng.integers(1, min(3, cap - totals[d]) + 1)) for d in ports}
            break
        else:
            # Guaranteed simple member; there is still a visible long-count shift.
            center, ring, source_dirs, source_count = 9, 6, [0], 1
            color_ids, heights, pitches = [0], [1], [2]
            raw = {0: 5 if long_counter else 1}
            destinations = {1: 1}
    palette = rng.permutation(10).tolist()
    bg, frame, hub = palette[:3]
    colors = palette[3:3 + source_count]
    receiver_pool = palette[3 + source_count:]
    sources = [dict(direction=d, count=h, width=int(rng.choice([3, 5])),
                    pitch=p, color=colors[i], turns=raw[i])
               for d, h, p, i in zip(source_dirs, heights, pitches, color_ids)]
    totals = {d: sum(s['count'] for s in sources if (s['direction'] + s['turns']) % 4 == d)
              for d in destinations}
    receivers = {}
    for d, initial in destinations.items():
        widths = [w for w in (1, 3, 5) if _capacity(ring, w) >= initial + totals[d]]
        receivers[d] = dict(count=initial, width=int(rng.choice(widths)), color=int(rng.choice(receiver_pool)))
    counters = {colors[i]: k for i, k in raw.items() if k}
    return dict(center=center, ring=ring, background=bg, frame=frame, hub=hub,
                sources=sources, receivers=receivers, counters=counters)


def _draw_descriptor(desc):
    c, ring = desc['center'], desc['ring']
    inp = _base_board(c, ring, desc['background'], desc['frame'], desc['hub'])
    out = inp.copy()
    totals = {d: 0 for d in desc['receivers']}
    for s in desc['sources']:
        target = (s['direction'] + s['turns']) % 4
        consumed = target in totals
        if consumed:
            totals[target] += s['count']
        for k in range(s['count']):
            radius = ring + 1 + k * s['pitch']
            assert radius < c
            _rod(inp, c, s['direction'], radius, s['width'], s['color'])
            _rod(out, c, s['direction'], radius, s['width'], desc['hub'] if consumed else s['color'])
    for d, t in desc['receivers'].items():
        for k in range(t['count'] + totals[d]):
            radius = ring - 1 - 2 * k
            assert radius >= t['width'] // 2 + 1
            _rod(out, c, d, radius, t['width'], t['color'])
            if k < t['count']:
                _rod(inp, c, d, radius, t['width'], t['color'])
    return inp, out


def _pack_counters(rng, board, bg, counters, forced=None):
    n = len(board)
    specs = sorted(counters.items(), key=lambda item: -item[1])
    forced = forced or {}
    for _ in range(24):
        h, w = rng.integers(n + 3, 31, 2).tolist()
        r, c = int(rng.integers(h - n + 1)), int(rng.integers(w - n + 1))
        inp = np.full((h, w), bg, dtype=np.int8)
        inp[r:r + n, c:c + n] = board
        blocked = np.zeros((h, w), dtype=bool)
        blocked[max(0, r - 1):min(h, r + n + 1), max(0, c - 1):min(w, c + n + 1)] = True
        success = True
        for color, length in specs:
            groups = []
            for orient in ('horizontal', 'vertical'):
                if color in forced and forced[color] != orient:
                    continue
                bh, bw = (1, length) if orient == 'horizontal' else (length, 1)
                yy, xx = np.indices((h - bh + 1, w - bw + 1))
                clear = np.ones(yy.shape, dtype=bool)
                for k in range(length):
                    clear &= ~blocked[yy + (k if bh > 1 else 0), xx + (k if bw > 1 else 0)]
                starts = np.argwhere(clear)
                if len(starts):
                    groups.append((orient, starts))
            if not groups:
                success = False
                break
            orient, starts = groups[int(rng.integers(len(groups)))]
            y, x = starts[int(rng.integers(len(starts)))].tolist()
            for k in range(length):
                yy, xx = y + (k if orient == 'vertical' else 0), x + (k if orient == 'horizontal' else 0)
                inp[yy, xx] = color
                blocked[max(0, yy - 1):min(h, yy + 2), max(0, xx - 1):min(w, xx + 2)] = True
        if success:
            return inp
    # Bounded packing fallback. Four length-nine counters always fit.
    inp = np.full((30, 30), bg, dtype=np.int8)
    inp[:n, :n] = board
    if n <= 21:
        for i, (color, length) in enumerate(specs):
            inp[:length, n + 1 + 2 * i] = color
    else:
        cursors = [0, 0]
        for color, length in specs:
            lane = int(np.argmin(cursors))
            start = cursors[lane]
            assert start + length <= n
            if lane == 0:
                inp[start:start + length, n + 1] = color
            else:
                inp[n + 1, start:start + length] = color
            cursors[lane] += length + 1
    return inp


def _make_pair(rng, witness, long_counter, orientation):
    desc = _sample_descriptor(rng, witness, long_counter)
    board, output = _draw_descriptor(desc)
    forced = {}
    if witness is not None:
        length = max(witness[1:]) if witness[0] == 0 else 4
        color = next(c for c, k in desc['counters'].items() if k == length)
        forced[color] = orientation if witness[0] == 0 else ('vertical' if orientation == 'horizontal' else 'horizontal')
    inp = _pack_counters(rng, board, desc['background'], desc['counters'], forced)
    return dict(input=inp, output=output)


def _unique(pair, seen):
    if _key(pair['input']) not in seen:
        return pair
    # Reassign visible colors rather than inserting a hidden example identifier.
    info = _parse(pair['input'])
    old = [info['background'], info['frame'], info['hub']]
    old += [c for c in range(10) if c not in old]
    for roles in itertools.permutations(range(10), 3):
        new = list(roles) + [c for c in range(10) if c not in roles]
        lut = np.empty(10, dtype=np.int8)
        lut[old] = new
        result = {name: lut[a] for name, a in pair.items()}
        if _key(result['input']) not in seen:
            return result
    raise AssertionError('720 distinct visible palettes exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if (not _integer(num_train) or not _integer(num_test) or
            not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError('require 2..10 training, 1..4 test, and num_train > num_test')
    rng = np.random.default_rng(seed)
    orientation = str(rng.choice(['horizontal', 'vertical']))
    witnesses = rng.permutation(2).tolist()
    angles = [(1, 3), (2, 1), (2, 3), (3, 1)][int(rng.integers(4))]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(count):
            witness = (witnesses[index], *angles) if split == 'train' and index < 2 else None
            pair = _unique(_make_pair(rng, witness, split == 'test', orientation), seen)
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _totals(info):
    return {d: [s for s in info['sources'] if s['target'] == d]
            for d in info['receivers']}


def _rival_differs(info, output, route=None, contribution=None):
    try:
        return not np.array_equal(_answer(info, route, contribution), output)
    except AssertionError:  # a rival's clipped sum also contradicts the answer
        return True


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            info = _parse(inp)
            assert np.array_equal(_answer(info), out), 'wrong routing, sum, highlighting, or crop'
            assert np.array_equal(solve(inp), out)
            for d, t in info['receivers'].items():
                add = sum(s['count'] for s in info['sources'] if s['target'] == d)
                width = t['width']
                # Receiver-color area includes other ports of that color, so
                # measure its own spoke rather than group pixels globally.
                output_stack = _read_stack(out, info['background'], d, 1, info['ring'] - 1, False)
                assert output_stack['count'] == t['count'] + add
                assert output_stack['width'] == width and output_stack['color'] == t['color']
            for s in info['sources']:
                output_stack = _read_stack(out, info['background'], s['direction'], info['ring'] + 1,
                                           info['center'] - 1, True)
                assert output_stack['radii'] == s['radii'] and output_stack['width'] == s['width']
                assert output_stack['color'] == (info['hub'] if s['target'] in info['receivers'] else s['color'])
            analyses[split].append(info)
    train = analyses['train']
    active = [s for p in train for s in p['sources'] if s['target'] in p['receivers']]
    assert {0, 1, 2, 3, 4} <= {s['turns'] for s in active}, 'unary and full-cycle evidence'
    assert any(s['target'] not in p['receivers'] for p in train for s in p['sources']), 'closed-port evidence'
    assert any(not v for p in train for v in _totals(p).values()), 'untouched-receiver evidence'
    assert any(a['color'] == b['color'] and
               (a['target'] in p['receivers']) != (b['target'] in p['receivers'])
               for p in train for a in p['sources'] for b in p['sources']), 'whole-layer/partial participation evidence'
    assert any(len(v) >= 2 and len({s['count'] for s in v}) >= 2
               for p in train for v in _totals(p).values()), 'unequal source aggregation evidence'
    assert any(sum(s['count'] for s in v) > len({r for s in v for r in s['radii']})
               for p in train for v in _totals(p).values()), 'overlapping virtual tokens must retain multiplicity'
    operands = [(sum(s['count'] for s in v), p['receivers'][d]['count'])
                for p in train for d, v in _totals(p).items() if v]
    assert len({s for s, d in operands}) >= 2 and len({d for s, d in operands}) >= 2
    assert len({(s + d) / d for s, d in operands}) >= 2, 'destination multiplier ambiguity'
    assert any(t['count'] > 1 for p in train for t in p['receivers'].values()), 'visible pitch evidence'
    for rival in ('no_rotation', 'counterclockwise', 'one_per_stack', 'pixel_count'):
        different = False
        for pair, p in zip(puzzle['train'], train):
            if rival == 'no_rotation':
                different |= _rival_differs(p, pair['output'], route=lambda s: s['direction'])
            elif rival == 'counterclockwise':
                different |= _rival_differs(p, pair['output'], route=lambda s: (s['direction'] - s['turns']) % 4)
            elif rival == 'one_per_stack':
                different |= _rival_differs(p, pair['output'], contribution=lambda s: 1)
            else:
                different |= _rival_differs(p, pair['output'], contribution=lambda s: s['count'] * s['width'])
        assert different, 'training does not distinguish ' + rival
    assert all(max(p['markers'].get(s['color'], {}).get('length', 0)
                   for s in p['sources'] if s['target'] in p['receivers']) >= 5
               for p in analyses['test']), 'test must exercise a contributing long counter'
    return None


def self_test():
    """Bounded semantic, variation, mutation, determinism, and count checks."""
    shapes, boards, rings, palettes, widths, pitches, sources, ports = (set() for _ in range(8))
    fingerprints, structures, lengths, directions = set(), set(), set(), set()
    same_layer = inactive_port = blocked = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        fingerprints.add(tuple(_key(q['input']) for split in p for q in p[split]))
        for split in p:
            for pair in p[split]:
                a = _parse(pair['input'])
                shapes.add(pair['input'].shape)
                boards.add(len(a['board']))
                rings.add(a['ring'])
                palettes.add((a['background'], a['frame'], a['hub']))
                sources.add(len(a['sources']))
                ports.add(len(a['receivers']))
                widths.update(s['width'] for s in a['sources'] + list(a['receivers'].values()))
                for s in a['sources']:
                    if len(s['radii']) > 1:
                        pitches.add(s['radii'][1] - s['radii'][0])
                    directions.add(s['target'])
                lengths.update(m['length'] for m in a['markers'].values())
                blocked += any(s['target'] not in a['receivers'] for s in a['sources'])
                inactive_port += any(not v for v in _totals(a).values())
                same_layer += len({s['color'] for s in a['sources']}) < len(a['sources'])
                role_ids = {}
                structures.add((a['board'].shape, tuple(role_ids.setdefault(int(v), len(role_ids)) for v in a['board'].flat)))
    assert len(fingerprints) == 42 and len(structures) >= 100
    assert len(shapes) >= 25 and len(boards) >= 6 and len(rings) >= 5
    assert len(palettes) >= 35 and widths == {1, 3, 5} and pitches == {2, 3}
    assert sources == {1, 2, 3, 4} and ports == {1, 2, 3, 4}
    assert lengths == set(range(1, 10)) and directions == set(range(4))
    assert blocked >= 42 and same_layer >= 42 and inactive_port >= 5
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + 10 * nt + nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.0, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p, q = generate_puzzle(777, 2, 1), generate_puzzle(777, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for split in p for x, y in zip(p[split], q[split]) for k in x)
    # Erase a received token, fail to highlight a contributing source, alter an
    # untouched source, retain a counter, and supply the no-rotation shortcut.
    for defect in ('sum', 'highlight', 'untouched', 'crop', 'rotation'):
        broken = {split: [{k: a.copy() for k, a in pair.items()} for pair in p[split]] for split in p}
        for pair in broken['train']:
            a = _parse(pair['input'])
            if defect == 'sum':
                d = next(d for d, v in _totals(a).items() if v)
                t = a['receivers'][d]
                _rod(pair['output'], a['center'], d, a['ring'] - 1 - 2 * t['count'], t['width'], a['background'])
            elif defect in ('highlight', 'untouched'):
                contributing = defect == 'highlight'
                s = next((s for s in a['sources'] if (s['target'] in a['receivers']) == contributing), None)
                if s is None:
                    continue
                _rod(pair['output'], a['center'], s['direction'], s['radii'][0], s['width'],
                     s['color'] if contributing else a['hub'])
            elif defect == 'crop':
                pair['output'] = pair['input'].copy()
            else:
                if not _rival_differs(a, pair['output'], route=lambda s: s['direction']):
                    continue
                pair['output'] = _answer(a, route=lambda s: s['direction'])
            break
        try:
            validate_generated_puzzle(broken)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError('mutation accepted: ' + defect)
    return dict(seeds=42, distinct_puzzles=len(fingerprints), structures=len(structures),
                input_shapes=len(shapes), board_sizes=sorted(boards), ring_radii=sorted(rings),
                source_counts=sorted(sources), receiver_counts=sorted(ports),
                token_widths=sorted(widths), source_pitches=sorted(pitches),
                counter_lengths=sorted(lengths))


if __name__ == '__main__':
    print(self_test())
