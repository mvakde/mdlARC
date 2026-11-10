"""Dock-and-echo: geometric receivers turn patterned motion into reflected copies.

All conventions are visible. There is no puzzle-wide hidden trail color.
The solver parses the input; construction does not supply it with metadata.
"""

from itertools import product
import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # Outward from a fork's base.


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid color")
    return a


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                p = r + dr, c + dc
                if p in unseen:
                    unseen.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append(cells)
    return result


def _bounds(cells):
    r, c = zip(*cells)
    return min(r), min(c), max(r) + 1, max(c) + 1


def _rect(k, q, length, span):
    return set(product(range(k, k + length), range(q, q + span)))


def _expand(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _paint(a, cells, color):
    for p in cells:
        a[p] = color


def _parse_fork(cells):
    """Recognize a full base and three separated positive constant-depth runs."""
    parses = []
    for out in _DIRS:
        dr, dc = out
        perp = (-dc, dr)
        kmin = min(r * dr + c * dc for r, c in cells)
        qmin = min(r * perp[0] + c * perp[1] for r, c in cells)
        local = {(r * dr + c * dc - kmin,
                  r * perp[0] + c * perp[1] - qmin) for r, c in cells}
        width = max(q for _, q in local) + 1
        if {q for k, q in local if k == 0} != set(range(width)):
            continue
        lengths = [sum((k, q) in local for k in range(1, 31)) for q in range(width)]
        expected = {(0, q) for q in range(width)} | {
            (k, q) for q, n in enumerate(lengths) for k in range(1, n + 1)}
        if expected != local:
            continue
        runs, q = [], 0
        while q < width:
            if lengths[q] == 0:
                q += 1
                continue
            start, n = q, lengths[q]
            while q < width and lengths[q] == n:
                q += 1
            runs.append((start, q, n))
        if len(runs) != 3 or any(runs[i][1] >= runs[i + 1][0] for i in (0, 1)):
            continue
        qa, qb, n = runs[1]
        if qa + qb != width or (qb - qa) % 2 != 1:
            continue
        parses.append(dict(out=out, perp=perp, kmin=kmin, qmin=qmin,
                           width=width, q0=qa, span=qb-qa, stop=n+1,
                           lengths=tuple(t[2] for t in runs), runs=tuple(runs), cells=cells))
    _require(len(parses) == 1, "fork must have a unique base and three positive prongs")
    return parses[0]


def _point(fork, k, q):
    k, q = k + fork['kmin'], q + fork['qmin']
    return (k * fork['out'][0] + q * fork['perp'][0],
            k * fork['out'][1] + q * fork['perp'][1])


def _candidate(fork, cells, a, bg):
    dr, dc = fork['out']
    vr, vc = fork['perp']
    ks = [r * dr + c * dc - fork['kmin'] for r, c in cells]
    qs = {r * vr + c * vc - fork['qmin'] for r, c in cells}
    if qs != set(range(fork['q0'], fork['q0'] + fork['span'])):
        return None
    start, length = min(ks), max(ks) - min(ks) + 1
    if start <= fork['stop']:
        return None
    route = {_point(fork, k, q) for k in range(fork['stop'], start + length) for q in qs}
    if any(int(a[p]) != bg for p in route - cells):
        return None
    return dict(start=start, length=length, distance=start-fork['stop'], route=route,
                final={_point(fork, k, q) for k in range(fork['stop'], fork['stop']+length)
                       for q in qs})


def _analyze(input_grid):
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    _require(int(counts.max()) * 2 > a.size, "background must occupy a strict majority")
    forks, tokens, occupied = [], [], set()
    for cells in _components(a != bg):
        _require(occupied.isdisjoint(_expand(cells)), "input objects must be separated")
        occupied.update(cells)
        r0, c0, r1, c1 = _bounds(cells)
        if len(cells) == (r1-r0) * (c1-c0):
            tokens.append(dict(cells=cells, box=(r0, c0, r1, c1), tile=a[r0:r1, c0:c1].copy()))
        else:
            _require(len({int(a[p]) for p in cells}) == 1, "fork must be monochrome")
            fork = _parse_fork(cells)
            fork['color'] = int(a[min(cells)])
            forks.append(fork)
    _require(forks and tokens, "expected forks and rectangular tokens")
    matches, used, assigned = [], set(), set()
    for fi, fork in enumerate(forks):
        candidates = [(ti, m) for ti, token in enumerate(tokens)
                      if (m := _candidate(fork, token['cells'], a, bg)) is not None]
        _require(len(candidates) <= 1, "two tokens qualify for one fork")
        if not candidates:
            continue
        ti, match = candidates[0]
        _require(ti not in used, "one token qualifies for two forks")
        d, length = match['distance'], match['length']
        _require(d >= length + 1, "echo must be separated from the swept strip")
        _require(match['start'] >= max(fork['lengths']) + 2, "token begins inside the guards")
        dr, dc = fork['out']
        echo = {(r+d*dr, c+d*dc) for r, c in tokens[ti]['cells']}
        _require(all(0 <= r < a.shape[0] and 0 <= c < a.shape[1] for r, c in echo),
                 "echo outside canvas")
        _require(all(int(a[p]) == bg for p in echo), "echo destination must be empty")
        region = match['route'] | echo
        _require(assigned.isdisjoint(region), "successful constructions overlap")
        assigned.update(region)
        used.add(ti)
        match.update(fi=fi, ti=ti, echo=echo,
                     ink=int(a[_point(fork, match['start'], fork['q0'] + fork['span']//2)]))
        matches.append(match)
    _require(matches, "at least one successful dock is required")
    return dict(background=bg, forks=forks, tokens=tokens, matches=matches,
                unmatched=set(range(len(tokens))) - used)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the answer using only the supplied cells; never mutate the input."""
    a = _grid(input_grid)
    info = _analyze(a)
    b = a.copy()
    for token in info['tokens']:
        _paint(b, token['cells'], info['background'])
    for m in info['matches']:
        token, fork = info['tokens'][m['ti']], info['forks'][m['fi']]
        dr, dc = fork['out']
        d = m['distance']
        _paint(b, m['route'] - m['final'], m['ink'])
        r0, c0, r1, c1 = token['box']
        # Translation preserves the docked pattern. Reflection of both coordinates
        # about the original tile center, then +d outward, constructs its echo.
        for r, c in token['cells']:
            b[r-d*dr, c-d*dc] = a[r, c]
            b[r0+r1-1-r+d*dr, c0+c1-1-c+d*dc] = a[r, c]
    return b


def _partition(rng, h, w, count, minimum=10):
    if count == 1:
        return [(0, 0, h, w)]
    horizontal = bool(rng.integers(2))
    n = h if horizontal else w
    cut = int(rng.integers(minimum, n-minimum))
    if count == 2:
        boxes = ([(0, 0, cut, w), (cut+1, 0, h-cut-1, w)] if horizontal else
                 [(0, 0, h, cut), (0, cut+1, h, w-cut-1)])
    else:
        other = w if horizontal else h
        cross = int(rng.integers(minimum, other-minimum))
        boxes = ([(0, 0, cut, w), (cut+1, 0, h-cut-1, cross),
                  (cut+1, cross+1, h-cut-1, w-cross-1)] if horizontal else
                 [(0, 0, h, cut), (0, cut+1, cross, w-cut-1),
                  (cross+1, cut+1, h-cross-1, w-cut-1)])
    if rng.integers(2):
        boxes = [(h-r-bh, w-c-bw, bh, bw) for r, c, bh, bw in boxes]
    return [boxes[int(j)] for j in rng.permutation(count)]


def _module(rng, box, spec):
    r0, c0, bh, bw = box
    dr, dc = spec['out']
    vr, vc = -dc, dr
    cross, depth = (bw, bh) if dr else (bh, bw)
    active = spec['active']
    span_options = [s for s in (1, 3, 5) if s+4 <= cross]
    span = 3 if spec.get('witness') else int(rng.choice(span_options))
    widths = [(g, gap, tail) for g in (1, 2) for gap in (1, 2, 3) for tail in (0, 1)
              if span + 2*(g+gap+tail) <= cross]
    g, gap, tail = widths[int(rng.integers(len(widths)))]
    width = span + 2*(g+gap+tail)
    q0 = (width-span)//2
    if active:
        options = [n for n in range(1, 5) if 3*n+4 <= depth]
        length = 2 if spec.get('witness') else int(rng.choice(options))
        cap = min(6, depth-3*length-3)
        mode = spec.get('mode', 'random')
        if mode == 'long':
            center = int(rng.integers(2, cap+1))
        elif mode == 'short':
            center = int(rng.integers(1, min(cap, 3)+1))
        else:
            center = int(rng.integers(1, cap+1))
        stop = center+1
        d = int(rng.integers(length+1, (depth-stop-length)//2+1))
        start = stop+d
        guard_cap = min(6, start-2)
        if mode == 'long':
            left, right = map(int, rng.integers(1, center, size=2))
        elif mode == 'short':
            left, right = map(int, rng.integers(center+1, guard_cap+1, size=2))
        else:
            left, right = map(int, rng.integers(1, guard_cap+1, size=2))
        original = _rect(start, q0, length, span)
        final = _rect(stop, q0, length, span)
        route = _rect(stop, q0, d+length, span)
        echo = _rect(stop+2*d, q0, length, span)
        token_span = span
    else:
        cap = min(6, depth-4)
        left, center, right = map(int, rng.integers(1, cap+1, size=3))
        length = int(rng.integers(1, min(2, depth-max(left, center, right)-2)+1))
        token_span = span+2  # Collinear but the wrong cross-section.
        original = _rect(max(left, center, right)+2, q0-1, length, token_span)
        final, route, echo = set(), set(), set()
    structure = {(0, q) for q in range(width)}
    for qa, qb, n in ((tail, tail+g, left), (q0, q0+span, center),
                      (width-tail-g, width-tail, right)):
        structure.update(_rect(1, qa, n, qb-qa))
    reserve = structure | original | route | echo
    raw = {(k*dr+q*vr, k*dc+q*vc) for k, q in reserve}
    min_r, min_c, max_r, max_c = _bounds(raw)
    offset_r = r0 + int(rng.integers(bh-(max_r-min_r)+1)) - min_r
    offset_c = c0 + int(rng.integers(bw-(max_c-min_c)+1)) - min_c

    def place(cells):
        return {(k*dr+q*vr+offset_r, k*dc+q*vc+offset_c) for k, q in cells}

    return dict(structure=place(structure), original=place(original),
                final=place(final), route=place(route), echo=place(echo),
                reserve=place(reserve), active=active, length=length, span=token_span,
                source_order=sorted(original), out=(dr, dc), perp=(vr, vc),
                offset=(offset_r, offset_c), witness=spec.get('witness', False),
                repeat=spec.get('repeat', False))


def _pattern(rng, length, span, bg, ink, witness=False, repeat=False, common=None):
    carrier = (common if common is not None and common != ink else
               int(rng.choice([x for x in range(10) if x not in (bg, ink)])))
    t = np.full((length, span), carrier, dtype=np.int8)
    inward = (0, span//2)
    t[inward] = ink
    budget = (t.size-1)//2  # Reserve fill cells; tiny tiles need no majority role.
    others = [(r, c) for r in range(length) for c in range(span) if (r, c) != inward]
    rng.shuffle(others)
    if witness:
        p = (length-1, 0)
        auxiliary = int(rng.choice([x for x in range(10) if x not in (bg, carrier, ink)]))
        t[p] = ink if repeat else auxiliary
        return t
    if budget > 1:
        count = int(rng.integers(0, budget))
        mode = int(rng.integers(3))
        if mode == 1:
            axis = int(rng.integers(2))
            fixed = int(rng.integers(length if axis == 0 else span))
            others.sort(key=lambda p: p[axis] != fixed)
        elif mode == 2:
            others.sort(key=lambda p: abs(p[0]-inward[0])+abs(p[1]-inward[1]))
        detail = int(rng.choice([x for x in range(10) if x not in (bg, carrier)]))
        for p in others[:count]:
            t[p] = (int(rng.choice([ink, detail])) if mode == 0 else detail)
    return t


def _draw_token(a, module, pattern):
    ks, qs = zip(*module['source_order'])
    k0, q0 = min(ks), min(qs)
    dr, dc = module['out']
    vr, vc = module['perp']
    rr, cc = module['offset']
    for k, q in module['source_order']:
        a[k*dr+q*vr+rr, k*dc+q*vc+cc] = pattern[k-k0, q-q0]


def _new_pair(rng, specs, anchor=False):
    count = len(specs)
    low = 23 if anchor else 21 if count > 1 else 15
    h, w = map(int, rng.integers(low, 31, size=2))
    bg = int(rng.integers(10))
    common = int(rng.choice([x for x in range(10) if x != bg])) if rng.random() < .35 else None
    for attempt in range(120):
        boxes = _partition(rng, h, w, count, minimum=11 if anchor else 10)
        modules = [_module(rng, box, spec) for box, spec in zip(boxes, specs)]
        a = np.full((h, w), bg, dtype=np.int8)
        b = a.copy()
        palette = list(map(int, rng.permutation([x for x in range(10) if x != bg])))
        # Witnesses teach independent inks. Other examples may share some or all
        # inward colors, without tying them to fork orientation or carrier color.
        inks = (palette if anchor else [int(rng.choice(palette[:int(rng.integers(1, 5))]))
                                      for _ in modules])
        shared_fork = int(rng.choice(palette)) if rng.random() < .3 else None
        blocked = set()
        for j, module in enumerate(modules):
            fcolor = shared_fork if shared_fork is not None else int(rng.choice(palette))
            _paint(a, module['structure'], fcolor)
            _paint(b, module['structure'], fcolor)
            t = _pattern(rng, module['length'], module['span'], bg, inks[j],
                         module['witness'], module['repeat'], common)
            _draw_token(a, module, t)
            blocked.update(_expand(module['reserve']))
            if module['active']:
                # Ground truth comes directly from sampled local geometry.
                _paint(b, module['route']-module['final'], int(t[0, module['span']//2]))
                r, c, r1, c1 = _bounds(module['final'])
                ro, co, ro1, co1 = _bounds(module['original'])
                global_tile = a[ro:ro1, co:co1].copy()
                b[r:r1, c:c1] = global_tile
                er, ec, er1, ec1 = _bounds(module['echo'])
                b[er:er1, ec:ec1] = global_tile[::-1, ::-1]
        try:
            info = _analyze(a)
        except AssertionError:
            continue
        if len(info['matches']) != sum(s['active'] for s in specs) or not np.array_equal(solve(a), b):
            continue
        # Extra rectangular distractors never occupy a successful route or echo.
        for _ in range(int(rng.integers(1, 5))):
            dh, dw = map(int, rng.integers(1, 4, size=2))
            accepted = None
            for small in (False, True):
                if small:
                    dh = dw = 1
                for index in rng.permutation((h-dh+1)*(w-dw+1)):
                    r, c = divmod(int(index), w-dw+1)
                    cells = _rect(r, c, dh, dw)
                    if not blocked.isdisjoint(cells):
                        continue
                    if any(_candidate(f, cells, a, bg) is not None for f in info['forks']):
                        continue
                    accepted = cells
                    break
                if accepted is not None:
                    break
            _require(accepted is not None, "no space for a distractor")
            r, c, r1, c1 = _bounds(accepted)
            ink = int(rng.choice(palette))
            a[r:r1, c:c1] = _pattern(rng, r1-r, c1-c, bg, ink, common=common)
            blocked.update(_expand(accepted))
        return dict(input=a, output=b)
    raise RuntimeError("could not pack separated, uniquely matched constructions")


def _count(value, lo, hi, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n, m = _count(num_train, 2, 10, 'num_train'), _count(num_test, 1, 4, 'num_test')
    if n <= m:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    omitted = int(rng.integers(4))
    familiar = [d for j, d in enumerate(_DIRS) if j != omitted]
    # The first witness contains one horizontal and one vertical movement.
    horizontal = [d for d in familiar if d[0] == 0]
    vertical = [d for d in familiar if d[1] == 0]
    first = [horizontal[int(rng.integers(len(horizontal)))], vertical[int(rng.integers(len(vertical)))]]
    third = next(d for d in familiar if d not in first)
    recipes = [
        [dict(out=d, active=True, witness=True, mode='short') for d in first],
        [dict(out=third, active=True, witness=True, repeat=True, mode='long'),
         dict(out=_DIRS[omitted], active=False)],
    ]
    for _ in range(n-2):
        count = int(rng.integers(1, 3))
        dirs = [familiar[int(j)] for j in rng.choice(3, size=count, replace=False)]
        recipes.append([dict(out=d, active=(j == 0 or bool(rng.integers(2)))) for j, d in enumerate(dirs)])
    puzzle, seen = dict(train=[], test=[]), set()
    for split, count in (('train', n), ('test', m)):
        for i in range(count):
            specs = recipes[i] if split == 'train' else [
                dict(out=d, active=True) for d in [_DIRS[omitted]] +
                [familiar[int(j)] for j in rng.choice(3, size=2, replace=False)]]
            for _ in range(30):
                pair = _new_pair(rng, specs, anchor=split == 'train' and i < 2)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct-input retries exhausted')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
    a, b = _grid(pair['input']), _grid(pair['output'])
    _require(a.shape == b.shape, 'dimensions changed')
    info = _analyze(a)
    _require(np.array_equal(solve(a), b), 'solve disagrees')
    changed = set()
    for token in info['tokens']:
        changed.update(token['cells'])
    for fork in info['forks']:
        _require(all(a[p] == b[p] for p in fork['cells']), 'fork changed')
    for ti in info['unmatched']:
        _require(all(int(b[p]) == info['background'] for p in info['tokens'][ti]['cells']),
                 'unmatched token survived')
    for match in info['matches']:
        token, fork = info['tokens'][match['ti']], info['forks'][match['fi']]
        changed.update(match['route'] | match['echo'])
        d = match['distance']
        dr, dc = fork['out']
        for p in match['route'] - match['final']:
            _require(int(b[p]) == match['ink'], 'wrong trail ink')
        # Slice-based validation differs from the scalar solver construction.
        r, c, r1, c1 = _bounds(match['final'])
        er, ec, er1, ec1 = _bounds(match['echo'])
        _require(np.array_equal(b[r:r1, c:c1], token['tile']), 'docked tile was rotated')
        _require(np.array_equal(b[er:er1, ec:ec1], token['tile'][::-1, ::-1]), 'echo is not a half-turn')
        ar, ac, ar1, ac1 = token['box']
        _require((ar-r, ac-c) == (d*dr, d*dc) and (er-ar, ec-ac) == (d*dr, d*dc),
                 'original tile is not the midpoint')
    mask = np.ones(a.shape, dtype=bool)
    for p in changed:
        mask[p] = False
    _require(np.array_equal(a[mask], b[mask]), 'unassigned cells changed')
    _require(not np.array_equal(a, b), 'identity pair')
    return info


def validate_generated_puzzle(puzzle):
    """Check schema, local geometry, midpoint echoes, erasure and teaching coverage."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'splits must be lists')
    n, m = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, 'illegal counts')
    seen, analyses = set(), dict(train=[], test=[])
    operations = set(range(4))
    modes, active_dirs, inactive_wrong_span = set(), set(), False
    sample_witnesses, repeated_sample, different_inks = set(), False, False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info = _check_pair(pair)
            analyses[split].append(info)
            key = _key(pair['input'])
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            _require(info['unmatched'], 'distractor erasure is not demonstrated')
            if split != 'train':
                continue
            different_inks |= len({t['ink'] for t in info['matches']}) > 1
            for fi, fork in enumerate(info['forks']):
                if fi not in {t['fi'] for t in info['matches']}:
                    for token in info['tokens']:
                        # A genuinely aligned rectangle with a different span.
                        dr, dc = fork['out']
                        vr, vc = fork['perp']
                        qs = {r*vr+c*vc-fork['qmin'] for r, c in token['cells']}
                        ks = {r*dr+c*dc-fork['kmin'] for r, c in token['cells']}
                        inactive_wrong_span |= (min(qs)+max(qs) == fork['width']-1
                                                and len(qs) != fork['span']
                                                and min(ks) > max(fork['lengths']))
            for match in info['matches']:
                fork, token = info['forks'][match['fi']], info['tokens'][match['ti']]
                active_dirs.add(fork['out'])
                l, c, r = fork['lengths']
                modes.add('long' if c > max(l, r) else 'short' if c < min(l, r) else 'other')
                er, ec, er1, ec1 = _bounds(match['echo'])
                target = np.asarray(pair['output'])[er:er1, ec:ec1]
                t = token['tile']
                views = (t, t[::-1, :], t[:, ::-1], t[::-1, ::-1])
                operations.intersection_update(j for j, v in enumerate(views) if np.array_equal(v, target))
                a = np.asarray(pair['input'])
                q0, k0, span = fork['q0'], match['start'], fork['span']
                alternatives = {'outer_edge': _point(fork, k0+match['length']-1, q0+span//2),
                                'left_corner': _point(fork, k0, q0),
                                'right_corner': _point(fork, k0, q0+span-1)}
                sample_witnesses.update(name for name, p in alternatives.items() if a[p] != match['ink'])
                colors, counts = np.unique(t, return_counts=True)
                if sum(counts == counts.max()) == 1 and int(colors[np.argmax(counts)]) != match['ink']:
                    sample_witnesses.add('majority')
                repeated_sample |= np.count_nonzero(t == match['ink']) > 1
    _require(operations == {3}, 'training does not distinguish the half-turn from reflection')
    _require({'short', 'long'} <= modes, 'central-tip stopping lacks contrasting evidence')
    _require(inactive_wrong_span, 'missing collinear wrong-width rejection evidence')
    _require(sample_witnesses == {'outer_edge', 'left_corner', 'right_corner', 'majority'},
             'inward midpoint sampling lacks contrast')
    _require(repeated_sample and different_inks, 'sampling is confused with unique colors or a fixed trail')
    _require(len(active_dirs) == 3, 'training must show three successful orientations')
    _require(all(len(i['forks']) <= 2 for i in analyses['train']), 'training count regime changed')
    _require(all(len(i['forks']) == len(i['matches']) == 3 for i in analyses['test']),
             'test must combine three successful docks')
    _require(all(len({f['out'] for f in i['forks']} - active_dirs) == 1 for i in analyses['test']),
             'test must activate the fourth outward orientation')


def self_test():
    """Bounded 42-seed diversity checks plus count and semantic negative cases."""
    shapes, geometries, patterns, layouts, distances, spans, lengths = (set() for _ in range(7))
    backgrounds, inks, modes, signatures = set(), set(), set(), set()
    repeated = symmetric = shared_ink = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(tuple(_key(t['input']) for s in ('train', 'test') for t in p[s]))
        for split in ('train', 'test'):
            for pair in p[split]:
                a = pair['input']
                info = _analyze(a)
                shapes.add(a.shape)
                backgrounds.add(info['background'])
                layouts.add((a.shape, tuple(tuple(sorted(f['cells'])) for f in info['forks'])))
                local_inks = []
                for fork in info['forks']:
                    geometries.add((fork['width'], fork['runs']))
                    l, c, r = fork['lengths']
                    modes.add('long' if c > max(l, r) else 'short' if c < min(l, r) else 'other')
                for match in info['matches']:
                    t = info['tokens'][match['ti']]['tile']
                    distances.add(match['distance'])
                    spans.add(info['forks'][match['fi']]['span'])
                    lengths.add(match['length'])
                    inks.add(match['ink'])
                    local_inks.append(match['ink'])
                    repeated += np.count_nonzero(t == match['ink']) > 1
                    symmetric += np.array_equal(t, t[::-1, ::-1])
                    norm = {}
                    patterns.add((t.shape, tuple(norm.setdefault(int(x), len(norm)) for x in t.flat)))
                shared_ink += len(set(local_inks)) < len(local_inks)
    _require(len(signatures) == 42 and len(shapes) >= 40 and len(layouts) >= 200, 'layout diversity collapsed')
    _require(len(geometries) >= 100 and len(patterns) >= 50, 'geometry or motif diversity collapsed')
    _require(spans == {1, 3, 5} and lengths == {1, 2, 3, 4}, 'rectangle dimensions missing')
    _require(len(distances) >= 8 and backgrounds == inks == set(range(10)), 'distance or palette gap')
    _require({'short', 'long', 'other'} <= modes and repeated >= 42 and symmetric >= 10,
             'contrasting motifs or fork modes missing')
    _require(shared_ink >= 10, 'repeated inks across receivers missing')
    # Exercise valid geometry beyond the sampler's bounds, then all D4 views.
    # Seven-cell span, nine-cell guard, asymmetric pattern, and no palette map.
    wide = np.zeros((25, 15), dtype=np.int8)
    structure = _rect(0, 2, 1, 11) | _rect(1, 2, 9, 1) | _rect(1, 12, 3, 1) | _rect(1, 4, 4, 7)
    _paint(wide, structure, 8)
    tile = np.full((2, 7), 5, dtype=np.int8)
    tile[0, 3], tile[1, 0], tile[1, 5] = 2, 7, 2
    wide[12:14, 4:11] = tile
    expected = wide.copy()
    expected[12:14, 4:11] = 0
    expected[5:7, 4:11] = tile
    expected[7:14, 4:11] = 2
    expected[19:21, 4:11] = tile[::-1, ::-1]
    for reflection in (False, True):
        for k in range(4):
            a = np.rot90(wide[:, ::-1] if reflection else wide, k)
            b = np.rot90(expected[:, ::-1] if reflection else expected, k)
            _require(np.array_equal(solve(a), b), 'broad-family or D4 equivariance regression')
    p, q = generate_puzzle(731, 6, 3), generate_puzzle(731, 6, 3)
    _require(all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s])
                 for k in ('input', 'output')), 'nondeterminism')
    for n in range(2, 11):
        for m in range(1, min(4, n-1)+1):
            p = generate_puzzle(1000+10*n+m, n, m)
            _require(len(p['train']) == n and len(p['test']) == m, 'counts ignored')
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (4.0, 1),
                 (True, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            continue
        raise AssertionError('illegal count accepted')
    p = generate_puzzle(17, 2, 1)
    a = p['test'][0]['input']
    before = a.copy()
    solve(a)
    _require(np.array_equal(a, before), 'solve mutated input')
    info = _analyze(a)
    for where in ('trail', 'echo', 'final', 'fork', 'unused'):
        broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        match = info['matches'][0]
        touched = set().union(*(t['cells'] for t in info['tokens']),
                              *(f['cells'] for f in info['forks']),
                              *(m['route'] | m['echo'] for m in info['matches']))
        point = {'trail': min(match['route']-match['final']), 'echo': min(match['echo']),
                 'final': min(match['final']), 'fork': min(info['forks'][0]['cells']),
                 'unused': next(p for p in product(range(a.shape[0]), range(a.shape[1])) if p not in touched)}[where]
        b = broken['test'][0]['output']
        b[point] = (int(b[point])+1) % 10
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            continue
        raise AssertionError('semantic corruption accepted: '+where)


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds, all legal counts, semantic negative cases)')
