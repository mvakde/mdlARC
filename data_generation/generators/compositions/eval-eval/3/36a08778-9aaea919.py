"""Ancestry buses: obstacle-routed, duplicate-free unary broadcast addition.

Only standard-library modules and NumPy are used. No parent data or generator
is needed at runtime. The two training witnesses constrain events, not grids.
"""
from __future__ import annotations

import itertools
import json
import numpy as np


def _check(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError('expected an integer ARC grid with sides 1..30')
    return a.astype(np.int8, copy=True)


def _runs(bits):
    d = np.diff(np.r_[False, bits, False].astype(int))
    return list(zip(np.flatnonzero(d == 1).tolist(),
                    np.flatnonzero(d == -1).tolist()))


def _connected(mask):
    todo = set(zip(*np.nonzero(mask)))
    if not todo:
        return False
    queue = [todo.pop()]
    while queue:
        y, x = queue.pop()
        for p in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
            if p in todo:
                todo.remove(p)
                queue.append(p)
    return not todo


def _x_at(port, token_width, width):
    return min(max(0, port - token_width // 2), width - token_width)


def _compatible_ports(ports, token_width, width):
    xs = sorted(_x_at(c, token_width, width) for c in ports)
    return all(b >= a + token_width + 1 for a, b in zip(xs, xs[1:]))


def _draw(a, y, x, mask, color):
    h, w = mask.shape
    _check(0 <= y and y+h <= a.shape[0] and 0 <= x and x+w <= a.shape[1],
           'token would be clipped')
    a[y:y+h, x:x+w][mask] = color


def _parse_band(band, offset, ports, bg, forbidden, width):
    lanes = _runs(np.any(band != bg, axis=0))
    _check(len(lanes) == len(ports), 'one stack per port is required')
    stacks = []
    for x, end in lanes:
        matching = [c for c in ports if x <= c < end]
        _check(len(matching) == 1, 'ambiguous port-to-stack alignment')
        port = matching[0]
        _check(x == _x_at(port, end-x, width), 'stack is not centered on its port')
        groups = _runs(np.any(band[:, x:end] != bg, axis=1))
        blocks = [band[y:stop, x:end] for y, stop in groups]
        _check(bool(blocks), 'empty stack')
        mask = blocks[0] != bg
        h, w = mask.shape
        _check(1 <= h <= 2 and 1 <= w <= 3 and _connected(mask), 'invalid token mask')
        _check(np.all(np.any(mask, axis=0)) and np.all(np.any(mask, axis=1)),
               'token must fill its bounding box')
        colors = set(int(c) for block in blocks for c in np.unique(block[block != bg]))
        _check(len(colors) == 1 and not colors & forbidden, 'invalid payload color')
        _check(all(np.array_equal(block != bg, mask) for block in blocks),
               'stack tokens differ')
        stacks.append(dict(port=port, x=x, ys=[offset+y for y, _ in groups],
                           count=len(groups), mask=mask, color=colors.pop()))
    return sorted(stacks, key=lambda s: s['port'])


def _parse_canonical(a, bg):
    h, w = a.shape
    rows = np.flatnonzero(np.all(a != bg, axis=1))
    _check(len(rows) == 2, 'expected two complete port rails')
    top, bottom = map(int, rows)
    _check(top >= 2 and bottom-top >= 7 and bottom <= h-3, 'invalid band sizes')
    colors, counts = np.unique(a[[top, bottom]], return_counts=True)
    _check(len(colors) == 2 and sum(counts == max(counts)) == 1, 'ambiguous rail roles')
    barrier = int(colors[np.argmax(counts)])
    guide = next(int(c) for c in colors if int(c) != barrier)
    starts = np.flatnonzero(a[top] == guide).tolist()
    sinks = np.flatnonzero(a[bottom] == guide).tolist()
    _check(1 <= len(starts) <= 4 and 1 <= len(sinks) <= 4, 'invalid port count')
    field = a[top+1:bottom]
    _check(set(map(int, np.unique(field))) <= {bg, guide, barrier}, 'unexpected field color')
    stub_mask = np.zeros(field.shape, bool)
    lengths = []
    for c in starts:
        ys = np.flatnonzero(field[:, c] == guide)
        _check(2 <= len(ys) <= 3 and np.array_equal(ys, np.arange(len(ys))),
               'launch port must have a straight inward stub')
        stub_mask[:len(ys), c] = True
        lengths.append(len(ys))
    _check(np.array_equal(stub_mask, field == guide), 'extra field guides')
    src = _parse_band(a[:top], 0, starts, bg, {guide, barrier}, w)
    dst = _parse_band(a[bottom+1:], bottom+1, sinks, bg, {guide, barrier}, w)
    _check(all(s['ys'][-1]+s['mask'].shape[0] == top for s in src), 'unaligned sources')
    _check(all(s['ys'][0] == bottom+1 for s in dst), 'unaligned receivers')
    _check(np.all(a[0] == bg) and np.all(a[-1] == bg), 'outer blank margins required')
    masks = [s['mask'] for s in src+dst]
    _check(all(np.array_equal(masks[0], m) for m in masks), 'one local token silhouette required')
    pitches = {b-y for s in src+dst for y, b in zip(s['ys'], s['ys'][1:])}
    _check(len(pitches) == 1, 'input must visibly reveal one common pitch')
    pitch = pitches.pop()
    _check(masks[0].shape[0] < pitch <= masks[0].shape[0]+2, 'invalid token pitch')
    return dict(a=a, background=bg, guide=guide, barrier=barrier, top=top,
                bottom=bottom, field=field, sources=src, receivers=dst,
                mask=masks[0], pitch=pitch, stub_lengths=lengths)


def _parse(value):
    a = _grid(value)
    bg = int(a[0, 0])
    _check(all(int(a[y, x]) == bg for y, x in
               ((0, 0), (0, -1), (-1, 0), (-1, -1))), 'corners must be background')
    colors, counts = np.unique(a, return_counts=True)
    _check(int(colors[np.argmax(counts)]) == bg and sum(counts == max(counts)) == 1,
           'background must be uniquely dominant')
    candidates = []
    for k in range(4):
        canonical = np.rot90(a, k).copy()
        try:
            parsed = _parse_canonical(canonical, bg)
            parsed['rotation'] = k
            candidates.append(parsed)
        except (AssertionError, IndexError, ValueError):
            pass
    if len(candidates) != 1:
        raise ValueError('input does not have a unique ancestry-bus interpretation')
    return candidates[0]


def _merge(target, labels, multiplicity=False):
    overlap = bool(set(target) & set(labels))
    for identity, count in labels.items():
        target[identity] = target.get(identity, 0)+count if multiplicity else 1
    return overlap


def _routing(field, starts, bg, barrier, multiplicity=False):
    """Labels denote source positions, never payload colors or token counts.

    All collisions at a depth use the immutable input layer. Painted crosspieces
    are a trace of the directed event; they do not create lateral/backward flow.
    """
    depth, width = field.shape
    active = {c: {i: 1} for i, c in enumerate(starts)}
    painted = np.zeros(field.shape, bool)
    events, multi, repeated, simultaneous, clipped, merges = [], 0, 0, 0, 0, 0
    incoming = []
    for r in range(depth):
        incoming.append(set(active))
        hits, following = {}, {}
        for c, labels in sorted(active.items()):
            if field[r, c] != barrier:
                following[c] = labels.copy()
                painted[r, c] = True
                continue
            left = right = c
            while left > 0 and field[r, left-1] == barrier:
                left -= 1
            while right+1 < width and field[r, right+1] == barrier:
                right += 1
            hits.setdefault((left, right), []).append(labels)
        simultaneous += len(hits) > 1
        for (left, right), incident in sorted(hits.items()):
            union = {}
            for labels in incident:
                repeated += _merge(union, labels, multiplicity)
            multi += len(incident) > 1
            clipped += left == 0 or right == width-1
            events.append((r, left, right, tuple(sorted(union))))
            if r > 0:
                painted[r-1, max(0, left-1):min(width, right+2)] = True
            for c in (left-1, right+1):
                if 0 <= c < width:
                    merges += _merge(following.setdefault(c, {}), union, multiplicity)
                    painted[r, c] = True
        active = following
    all_runs = [(r, left, stop-1) for r, row in enumerate(field)
                for left, stop in _runs(row == barrier)]
    reached = {(r, left, right) for r, left, right, _ in events}
    stats = dict(events=events, collision_count=len(events), simultaneous=simultaneous,
                 multi_hit=multi, repeated_ancestry=repeated, clipped=clipped,
                 merges=merges, inert=len(all_runs)-len(reached), incoming=incoming)
    return active, painted, stats


def _received(parsed, multiplicity=False, straight=False):
    starts = [s['port'] for s in parsed['sources']]
    if straight:
        terminal = {c: {i: 1} for i, c in enumerate(starts)}
    else:
        terminal, _, _ = _routing(parsed['field'], starts, parsed['background'],
                                   parsed['barrier'], multiplicity)
    return [terminal.get(s['port'], {}) for s in parsed['receivers']]


def _growth(parsed, received, weights=None):
    if weights is None:
        weights = [s['count'] for s in parsed['sources']]
    return [sum(weights[i]*copies for i, copies in labels.items()) for labels in received]


def _render(parsed):
    a = parsed['a']
    out = a.copy()
    bg, guide = parsed['background'], parsed['guide']
    starts = [s['port'] for s in parsed['sources']]
    terminal, painted, _ = _routing(parsed['field'], starts, bg, parsed['barrier'])
    region = out[parsed['top']+1:parsed['bottom']]
    region[painted & (parsed['field'] == bg)] = guide
    for r in (parsed['top'], parsed['bottom']):
        out[r, a[r] == guide] = bg
    for s in parsed['sources']:
        for y in s['ys']:
            _draw(out, y, s['x'], s['mask'], guide)
    labels = [terminal.get(s['port'], {}) for s in parsed['receivers']]
    for s, growth in zip(parsed['receivers'], _growth(parsed, labels)):
        for i in range(s['count'], s['count']+growth):
            y = s['ys'][0]+i*parsed['pitch']
            _check(y+s['mask'].shape[0] < len(out), 'receiver has insufficient headroom')
            _draw(out, y, s['x'], s['mask'], s['color'])
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from visible rails, launch stubs, token repetition, and obstacles."""
    p = _parse(input_grid)
    return np.rot90(_render(p), -p['rotation']).copy()


def _mask(rng, h, w):
    mask = np.ones((h, w), bool)
    if h == 2 and w > 1 and rng.random() < .7:
        # Connected full-span silhouettes built from overlapping row intervals.
        cut = int(rng.integers(1, w+1))
        mask[0, cut:] = False
        if rng.integers(2):
            mask = mask[:, ::-1]
        if rng.integers(2):
            mask = mask[::-1]
    return mask.copy()


def _inert(rng, field, starts, bg, guide, barrier, budget):
    """Place bars outside current incident rays without changing their topology."""
    for _ in range(budget):
        _, _, stats = _routing(field, starts, bg, barrier)
        options = []
        for r in range(len(field)):
            eligible = np.array([field[r, c] == bg and c not in stats['incoming'][r]
                                 and (c == 0 or field[r, c-1] != barrier)
                                 and (c+1 == field.shape[1] or field[r, c+1] != barrier)
                                 for c in range(field.shape[1])])
            options.extend((r, left, end) for left, end in _runs(eligible))
        if not options:
            break
        r, left, end = options[int(rng.integers(len(options)))]
        n = int(rng.integers(1, min(end-left, 6)+1))
        x = int(rng.integers(left, end-n+1))
        field[r, x:x+n] = barrier


def _witness_field(rng, width, depth, tw, kind):
    field = np.zeros((depth, width), np.int8)
    if kind == 'rejoin':
        a = int(rng.integers(2, max(3, width//3)))
        b = int(rng.integers(max(a+2*tw+3, 2*width//3), width-2))
        middle = (a+b)//2
        starts = [int(rng.integers(a, middle)), int(rng.integers(middle+1, b+1))]
        if not _compatible_ports(starts, tw, width):
            starts = [a, b]
        first = int(rng.integers(2, depth-3))
        second = int(rng.integers(first+2, depth))
        field[first, a:b+1] = 2
        a2 = int(rng.integers(0, a))
        b2 = int(rng.integers(b+1, width))
        if a2 == 0 and b2 == width-1:
            a2 = 1
        field[second, a2:b2+1] = 2
    else:
        middle = int(rng.integers(tw+3, width-tw-3))
        a = int(rng.integers(1, middle-max(tw, 2)+1))
        b = int(rng.integers(middle+max(tw, 2), width-1))
        starts = [int(rng.integers(a, middle-1)),
                  int(rng.integers(middle+2, b+1))]
        r = int(rng.integers(2, depth))
        field[r, a:middle] = 2
        field[r, middle+1:b+1] = 2
    field[:2, starts] = 1
    _inert(rng, field, starts, 0, 1, 2, int(rng.integers(1, 4)))
    return field, starts


def _random_field(rng, width, depth, tw, nsrc):
    field = np.zeros((depth, width), np.int8)
    candidates = list(map(int, rng.permutation(width)))
    starts = []
    for c in candidates:
        if _compatible_ports(starts+[c], tw, width):
            starts.append(c)
            if len(starts) == nsrc:
                break
    _check(len(starts) == nsrc, 'source ports do not fit')
    starts.sort()
    stub = int(rng.integers(2, 4))
    field[:stub, starts] = 1
    active = set(starts)
    for r in range(stub, depth):
        if rng.random() < .8:
            pivots = list(map(int, rng.permutation(sorted(active))))
            quota = int(rng.integers(1, min(3, len(pivots))+1))
            for c in pivots[:quota]:
                length = int(rng.integers(1, width)) if rng.random() < .25 else int(rng.integers(1, min(7, width)))
                left = int(rng.integers(max(0, c-length+1), min(c, width-length)+1))
                end = left+length
                if (np.all(field[r, left:end] == 0)
                        and (left == 0 or field[r, left-1] != 2)
                        and (end == width or field[r, end] != 2)):
                    field[r, left:end] = 2
        # Only the current prefix is needed to schedule subsequent obstacles.
        terminal, _, _ = _routing(field[:r+1], starts, 0, 2)
        active = set(terminal)
    _inert(rng, field, starts, 0, 1, 2, int(rng.integers(0, 5)))
    return field, starts


def _choose_sinks(rng, terminal, tw, width, kind):
    ports = []
    if kind in ('rejoin', 'shared'):
        ports = sorted(terminal)
        if not _compatible_ports(ports, tw, width):
            return None
    else:
        priority = sorted(terminal, key=lambda c: (-len(terminal[c]), float(rng.random())))
        quota = int(rng.integers(1, 4))
        for c in priority:
            if _compatible_ports(ports+[c], tw, width):
                ports.append(c)
                if len(ports) == quota:
                    break
    if not ports:
        return None
    if kind == 'rejoin' or (len(ports) < 4 and rng.random() < .4):
        unused = [c for c in range(width) if c not in terminal
                  and _compatible_ports(ports+[c], tw, width)]
        if unused:
            ports.append(int(rng.choice(unused)))
        elif kind == 'rejoin':
            return None
    return sorted(ports)


def _operands(rng, nsrc, terminals, sinks, h, pitch, depth, kind):
    feasible = []
    for counts in itertools.product(range(1, 4), repeat=nsrc):
        if kind == 'rejoin' and len(set(counts)) == 1:
            continue
        source_height = h+(max(counts)-1)*pitch
        space = 30-source_height-depth-4  # two rails and two blank margins
        capacity = 1+(space-h)//pitch
        growth = [sum(counts[i] for i in terminals.get(c, {})) for c in sinks]
        bounds = [min(3, capacity-g) for g in growth]
        if min(bounds) < 1:
            continue
        if max(counts) == 1 and max(bounds) < 2:
            continue  # no local visible pitch
        feasible.append((counts, growth, bounds))
    if not feasible:
        return None
    counts, growth, bounds = feasible[int(rng.integers(len(feasible)))]
    dests = [int(rng.integers(1, b+1)) for b in bounds]
    if max(counts) == 1 and max(dests) == 1:
        eligible = [i for i, b in enumerate(bounds) if b >= 2]
        dests[int(rng.choice(eligible))] = 2
    return list(counts), dests, growth


def _build(rng, settings, kind, test=False):
    for attempt in range(160):
        if settings['local_geometry']:
            h = 2 if kind == 'shared' else int(rng.integers(1, 3))
            tw = int(rng.integers(2, 4)) if kind == 'shared' else int(rng.integers(1, 4))
        else:
            h, tw = settings['h'], settings['tw']
        pitch = h+int(rng.integers(1, 3))
        width = int(rng.integers(max(14, 4*tw+5), 31))
        depth = int(rng.integers(9, 13)) if test else int(rng.integers(6, 10))
        # Wider geometry/count choices fit more readily at unit pitch gaps.
        if attempt >= 80:
            h, pitch, depth = 1, 2, 9 if test else 6
        mask = _mask(rng, h, tw)
        nsrc = 3 if test else int(rng.integers(1, 4))
        if kind in ('rejoin', 'shared'):
            field, starts = _witness_field(rng, width, depth, tw, kind)
        else:
            field, starts = _random_field(rng, width, depth, tw, nsrc)
        terminal, painted, stats = _routing(field, starts, 0, 2)
        if test and (stats['collision_count'] < 3 or not any(len(x) > 1 for x in terminal.values())):
            continue
        sinks = _choose_sinks(rng, terminal, tw, width, kind)
        if sinks is None:
            continue
        operands = _operands(rng, len(starts), terminal, sinks, h, pitch, depth, kind)
        if operands is None:
            continue
        counts, dests, growth = operands
        source_height = h+(max(counts)-1)*pitch
        dest_height = h+(max(d+g for d, g in zip(dests, growth))-1)*pitch
        minimum = source_height+depth+dest_height+4
        spare = 30-minimum
        top_margin = 1+int(rng.integers(0, min(3, spare)+1))
        remaining = spare-(top_margin-1)
        bottom_margin = 1+int(rng.integers(0, min(3, remaining)+1))
        height = minimum+top_margin+bottom_margin-2
        top = top_margin+source_height
        bottom = top+depth+1
        roles = tuple(map(int, rng.choice(10, 3, replace=False))) if settings['local_roles'] else settings['roles']
        bg, guide, barrier = roles
        pool = [c for c in range(10) if c not in roles]
        source_colors = rng.choice(pool, len(starts)).tolist()
        if kind == 'rejoin':
            source_colors[1] = source_colors[0]  # identity is spatial, not color
        dest_colors = rng.choice(pool, len(sinks)).tolist()
        a = np.full((height, width), bg, np.int8)
        b = a.copy()
        for r, ports in ((top, starts), (bottom, sinks)):
            a[r] = barrier
            b[r] = barrier
            a[r, ports] = guide
            b[r, ports] = bg
        lookup = np.array(roles, np.int8)
        a[top+1:bottom] = lookup[field]
        b[top+1:bottom] = lookup[field]
        b[top+1:bottom][painted & (field == 0)] = guide
        for c, n, color in zip(starts, counts, source_colors):
            x = _x_at(c, tw, width)
            for i in range(n):
                y = top-h-i*pitch
                _draw(a, y, x, mask, color)
                _draw(b, y, x, mask, guide)
        for c, n, g, color in zip(sinks, dests, growth, dest_colors):
            x = _x_at(c, tw, width)
            for i in range(n+g):
                y = bottom+1+i*pitch
                if i < n:
                    _draw(a, y, x, mask, color)
                _draw(b, y, x, mask, color)
        if np.count_nonzero(a == bg) <= max(np.count_nonzero(a == color) for color in range(10) if color != bg):
            continue
        rotation = int(rng.integers(4)) if settings['local_rotation'] else settings['rotation']
        pair = {name: np.rot90(value, rotation).copy() for name, value in (('input', a), ('output', b))}
        try:
            _parse(pair['input'])
        except (ValueError, AssertionError):
            continue
        return pair
    raise RuntimeError('bounded ancestry-bus construction exhausted')


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _evidence(pairs):
    flags = {name: False for name in ('rejoin', 'shared', 'unreached', 'unequal',
                                     'counting', 'routing', 'area', 'same_color')}
    observations = []
    for pair in pairs:
        p = _parse(pair['input'])
        received = _received(p)
        growth = _growth(p, received)
        flags['rejoin'] |= growth != _growth(p, _received(p, multiplicity=True))
        flags['unreached'] |= any(not labels for labels in received)
        flags['unequal'] |= any(len({p['sources'][i]['count'] for i in labels}) > 1 for labels in received)
        flags['shared'] |= (any(len(labels) >= 2 for labels in received)
                            and any(0 < len(labels) < len(p['sources']) for labels in received))
        flags['counting'] |= growth != _growth(p, received, [1]*len(p['sources']))
        flags['routing'] |= growth != _growth(p, _received(p, straight=True))
        flags['area'] |= bool(np.any(growth)) and p['mask'].sum() > 1
        flags['same_color'] |= len({s['color'] for s in p['sources']}) < len(p['sources'])
        observations.extend((s['count'], g) for s, g in zip(p['receivers'], growth))
    _check(all(flags[n] for n in flags if n != 'area'), 'training lacks a required ancestry/counting witness')
    _check(len({g for d, g in observations}) >= 3, 'training increments are not informative')
    _check(any(d != g and g > 0 for d, g in observations), 'destination doubling remains plausible')
    _check(any(d > 1 and g > 0 for d, g in observations), 'replacement/product shortcut remains plausible')
    return flags


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, lower, upper, name in ((num_train, 2, 10, 'num_train'),
                                     (num_test, 1, 4, 'num_test')):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lower <= value <= upper:
            raise ValueError(f'{name} must be an integer in {lower}..{upper}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    settings = dict(roles=tuple(map(int, rng.choice(10, 3, replace=False))),
                    h=int(rng.integers(1, 3)), tw=int(rng.integers(1, 4)),
                    local_geometry=bool(rng.integers(2)), local_roles=bool(rng.integers(2)),
                    rotation=int(rng.integers(4)), local_rotation=bool(rng.integers(2)))
    # Jointly choose two informative witnesses. The order is not a split code.
    for _ in range(80):
        anchors = [_build(rng, settings, 'rejoin'), _build(rng, settings, 'shared')]
        try:
            _evidence(anchors)
        except AssertionError:
            continue
        break
    else:
        raise RuntimeError('bounded informative-training construction exhausted')
    if rng.integers(2):
        anchors.reverse()
    puzzle = {'train': anchors, 'test': []}
    seen = {_key(p['input']) for p in anchors}
    for split, amount in (('train', num_train-2), ('test', num_test)):
        for _ in range(amount):
            for attempt in range(80):
                pair = _build(rng, settings, 'random', test=split == 'test')
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('bounded unique-input construction exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _check(all(isinstance(puzzle[s], list) for s in puzzle), 'split schema')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _check(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'pair counts')
    seen = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _check(a.shape == b.shape and _key(a) not in seen, 'shape or duplicate input')
            seen.add(_key(a))
            p = _parse(a)
            canonical_b = np.rot90(b, p['rotation'])
            _check(np.array_equal(canonical_b, _render(p)), 'incorrect routed stack transfer')
            _check(not np.array_equal(a, b), 'identity pair')
            barrier, guide, bg = p['barrier'], p['guide'], p['background']
            ca = p['a']
            _check(np.all(canonical_b[ca == barrier] == barrier), 'barrier overwritten')
            for row in (p['top'], p['bottom']):
                _check(np.all(canonical_b[row, ca[row] == guide] == bg), 'retained control ports')
            for s in p['sources']:
                for y in s['ys']:
                    patch = canonical_b[y:y+s['mask'].shape[0], s['x']:s['x']+s['mask'].shape[1]]
                    _check(np.all(patch[s['mask']] == guide), 'source not highlighted')
            received = _received(p)
            for s, g in zip(p['receivers'], _growth(p, received)):
                band = canonical_b[p['bottom']+1:, s['x']:s['x']+s['mask'].shape[1]]
                _check(len(_runs(np.any(band != bg, axis=1))) == s['count']+g, 'wrong receiver token count')
                if not g:
                    _check(np.array_equal(band, ca[p['bottom']+1:, s['x']:s['x']+s['mask'].shape[1]]),
                           'unreached receiver changed')
    flags = _evidence(puzzle['train'])
    if any(_parse(p['input'])['mask'].sum() > 1 for p in puzzle['test']):
        _check(flags['area'], 'test token area not distinguished in training')


def _signature(puzzle):
    return tuple((_key(p['input']), _key(p['output'])) for s in ('train', 'test') for p in puzzle[s])


def family_measurements(puzzle):
    records = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            p = _parse(pair['input'])
            received = _received(p)
            _, _, stats = _routing(p['field'], [s['port'] for s in p['sources']], p['background'], p['barrier'])
            records.append(dict(split=split, shape=pair['input'].shape, rotation=p['rotation'],
                                roles=(p['background'], p['guide'], p['barrier']),
                                mask=(p['mask'].shape, p['mask'].tobytes().hex()), pitch=p['pitch'],
                                depth=len(p['field']), sources=len(p['sources']), receivers=len(p['receivers']),
                                source_counts=[s['count'] for s in p['sources']],
                                destination_counts=[s['count'] for s in p['receivers']],
                                ancestry=[sorted(labels) for labels in received], growth=_growth(p, received),
                                collisions=stats['collision_count'], simultaneous=stats['simultaneous'],
                                inert=stats['inert'], clipped=stats['clipped'],
                                dedup_visible=_growth(p, received) != _growth(p, _received(p, multiplicity=True))))
    return records


def self_test():
    """Bounded 42-seed semantic/variation checks plus targeted mutations."""
    # Hand-built family members outside the sampler's current operand ranges.
    # Expected ancestry is explicit: one broad run gathers every source, forks,
    # and a broader run reconverges both copies, leaving only edge terminals.
    # This regression does not call the routing helper to construct its labels.
    for starts, counts in (([3, 7, 11, 15], [1, 1, 1, 1]), ([3, 15], [4, 1])):
        mask = np.ones((1, 2), bool)
        width, pitch, depth = 19, 2, 6
        source_height = 1+(max(counts)-1)*pitch
        dest_height = 1+(1+sum(counts))*pitch
        top, bottom = source_height+1, source_height+depth+2
        a = np.zeros((source_height+depth+dest_height+4, width), np.int8)
        sinks = [0, 6, 12, 18]
        a[top] = a[bottom] = 2
        a[top, starts] = a[bottom, sinks] = 1
        a[top+1:top+3, starts] = 1
        a[top+3, 2:17] = 2
        a[top+5, 1:18] = 2
        for c, n in zip(starts, counts):
            for i in range(n):
                _draw(a, top-1-i*pitch, _x_at(c, 2, width), mask, 3)
        for c in sinks:
            for i in range(2):
                _draw(a, bottom+1+i*pitch, _x_at(c, 2, width), mask, 4)
        expected = a.copy()
        expected[top, starts] = expected[bottom, sinks] = 0
        for c, n in zip(starts, counts):
            for i in range(n):
                _draw(expected, top-1-i*pitch, _x_at(c, 2, width), mask, 1)
        for c in (0, 18):
            for i in range(2, 2+sum(counts)):
                _draw(expected, bottom+1+i*pitch, _x_at(c, 2, width), mask, 4)
        for r, columns in ((1, range(1, 18)), (2, (1, 17)),
                           (3, range(19)), (4, (0, 18)), (5, (0, 18))):
            for c in columns:
                if a[top+1+r, c] == 0:
                    expected[top+1+r, c] = 1
        for rotation in range(4):
            _check(np.array_equal(solve(np.rot90(a, rotation)), np.rot90(expected, rotation)),
                   'four-port / extended-count regression')
    records, signatures, structures = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _check(_signature(puzzle) == _signature(generate_puzzle(seed)), 'seed not reproducible')
        signatures.add(_signature(puzzle))
        records.extend(family_measurements(puzzle))
        for pair in puzzle['train']+puzzle['test']:
            _check(np.array_equal(solve(pair['input']), pair['output']), 'input-only solve mismatch')
            roles = {}
            structures.add((pair['input'].shape,
                            tuple(roles.setdefault(int(c), len(roles)) for c in pair['input'].flat)))
    _check(len(signatures) == 42 and len(structures) >= 150, 'structural variation collapsed')
    _check(len({r['shape'] for r in records}) >= 45, 'canvas variation collapsed')
    _check({r['rotation'] for r in records} == set(range(4)), 'rotation coverage missing')
    _check({r['sources'] for r in records} >= {1, 2, 3}, 'source count coverage missing')
    _check({r['receivers'] for r in records} >= {1, 2, 3}, 'receiver count coverage missing')
    _check(len({r['mask'] for r in records}) >= 8, 'token geometry coverage missing')
    _check(len({r['roles'] for r in records}) >= 30, 'palette role coverage missing')
    _check(max(r['collisions'] for r in records) >= 6, 'collision composition coverage missing')
    for split in ('train', 'test'):
        local = [r for r in records if r['split'] == split]
        _check(any(r['simultaneous'] for r in local), 'simultaneous collision coverage missing')
        _check(any(r['clipped'] for r in local), 'clipped branch coverage missing')
        _check(any(r['dedup_visible'] for r in local), 'duplicate ancestry coverage missing')
        _check(any(not ids for r in local for ids in r['ancestry']), 'unreached receiver coverage missing')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            puzzle = generate_puzzle(314159, nt, nq)
            _check(len(puzzle['train']) == nt and len(puzzle['test']) == nq, 'counts not honored')
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    base = generate_puzzle(0, 2, 1)
    for defect in ('port', 'source', 'receiver', 'barrier'):
        broken = {s: [{k: v.copy() for k, v in p.items()} for p in base[s]] for s in base}
        pair = broken['train'][0]
        p = _parse(pair['input'])
        out = np.rot90(pair['output'], p['rotation']).copy()
        if defect == 'port':
            out[p['top'], p['sources'][0]['port']] = p['guide']
        elif defect == 'source':
            s = p['sources'][0]
            _draw(out, s['ys'][0], s['x'], s['mask'], s['color'])
        elif defect == 'receiver':
            growth = _growth(p, _received(p))
            i = next(i for i, g in enumerate(growth) if g)
            s = p['receivers'][i]
            _draw(out, s['ys'][0]+s['count']*p['pitch'], s['x'], s['mask'], p['background'])
        else:
            y, x = np.argwhere(p['a'] == p['barrier'])[0]
            out[y, x] = p['guide']
        pair['output'] = np.rot90(out, -p['rotation']).copy()
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError(f'validator accepted broken {defect}')
    return dict(seeds=42, pairs=len(records), shapes=len({r['shape'] for r in records}),
                structures=len(structures), masks=len({r['mask'] for r in records}),
                palettes=len({r['roles'] for r in records}),
                semantic_verification='generator-owned invariants and counterfactuals')


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
