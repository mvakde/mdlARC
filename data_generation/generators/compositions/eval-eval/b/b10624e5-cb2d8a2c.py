"""Stencil convoy: reflected, scaled cargo determines a bar-avoiding lane.

All conventions are visible in one input. No parent module, saved pair, seed,
or training label is used by solve. Coordinates used internally are forward
(away from the seed boundary) and lateral (the other array coordinate).
"""
from __future__ import annotations

from collections import deque
import numpy as np


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}: {value}")
    return value


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(int, copy=True)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def _puzzle_key(p):
    return tuple(tuple((_key(x['input']), _key(x['output'])) for x in p[s])
                 for s in ('train', 'test'))


def _forward(a, direction):
    # down, right, up, left. Lateral increases right for vertical headings,
    # and down for horizontal headings; these are orthogonal reflections.
    if direction == 0:
        return a.copy()
    if direction == 1:
        return a.T.copy()
    if direction == 2:
        return a[::-1].copy()
    return a.T[::-1].copy()


def _physical(a, direction):
    if direction in (0, 2):
        return _forward(a, direction)
    if direction == 1:
        return a.T.copy()
    return a[::-1].T.copy()


def _components(a, bg):
    unseen = set(map(tuple, np.argwhere(a != bg).tolist()))
    result = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        part, queue = {first}, deque([first])
        while queue:
            r, c = queue.popleft()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in unseen:
                    unseen.remove(q)
                    part.add(q)
                    queue.append(q)
        result.append(part)
    return result


def _square(points):
    rs, cs = zip(*points)
    r, c = min(rs), min(cs)
    side = max(rs)-r+1
    assert side == max(cs)-c+1 and len(points) == side*side
    return r, c, side


def _scale_pixels(motif, reference_side, side, bg):
    """Exact rational scaling, including sub-anchor detail and background gaps.

    Integer subpixels give a destination cell a color only if its whole area
    agrees. Color boundaries that would straddle a cell are rejected.
    """
    rs, cs = zip(*motif)
    r0 = min(rs)//reference_side*reference_side
    c0 = min(cs)//reference_side*reference_side
    r1 = (max(rs)//reference_side+1)*reference_side
    c1 = (max(cs)//reference_side+1)*reference_side
    raw = np.full((r1-r0, c1-c0), bg, dtype=int)
    for (r, c), color in motif.items():
        raw[r-r0, c-c0] = color
    subpixels = raw.repeat(side, axis=0).repeat(side, axis=1)
    h, w = subpixels.shape[0]//reference_side, subpixels.shape[1]//reference_side
    blocks = subpixels.reshape(h, reference_side, w, reference_side)
    lo, hi = blocks.min(axis=(1, 3)), blocks.max(axis=(1, 3))
    assert np.array_equal(lo, hi), 'scaled color boundary straddles a pixel'
    offset_r, offset_c = r0*side//reference_side, c0*side//reference_side
    return {(int(r)+offset_r, int(c)+offset_c): int(lo[r, c])
            for r, c in np.argwhere(lo != bg)}


def _reflect(motif, side, sign):
    return {(r, c if sign == 1 else side-1-c): color
            for (r, c), color in motif.items()}


def _extent(motif):
    rs, cs = zip(*motif)
    return min(rs), max(rs), min(cs), max(cs)


def _decode(input_grid):
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[counts.argmax()])
    assert int(counts.max()) > a.size//2, 'background must be a strict majority'
    components = _components(a, bg)
    seeds = []
    h, w = a.shape
    for part in components:
        if len({int(a[p]) for p in part}) != 1:
            continue
        try:
            r, c, side = _square(part)
        except AssertionError:
            continue
        edges = [r == 0, c == 0, r+side == h, c+side == w]
        if sum(edges) == 1:
            seeds.append((part, r, c, side, edges.index(True)))
    assert len(seeds) == 1, 'one isolated square must identify the launch edge'
    seed, _, _, side, direction = seeds[0]
    anchor = int(a[next(iter(seed))])
    a = _forward(a, direction)
    components = _components(a, bg)
    sources = [p for p in components if any(a[q] == anchor for q in p)
               and len({int(a[q]) for q in p}) > 1]
    assert len(sources) == 1, 'one complete prototype with a matching square'
    source = sources[0]
    anchor_points = {p for p in source if a[p] == anchor}
    sr, sc, reference_side = _square(anchor_points)
    motif = {(r-sr, c-sc): int(a[r, c]) for r, c in source}
    assert min(r for r, c in source) > 0
    assert max(r for r, c in source) < a.shape[0]-1
    assert min(c for r, c in source) > 0 and max(c for r, c in source) < a.shape[1]-1
    assert len({r for r, c in source}) > 1 and len({c for r, c in source}) > 1
    seed_parts = [p for p in components if all(a[q] == anchor for q in p)]
    assert len(seed_parts) == 1
    r, initial, decoded_side = _square(seed_parts[0])
    assert r == 0 and decoded_side == side and 0 < initial < a.shape[1]-side
    source_left, source_right = min(c for r, c in source), max(c for r, c in source)
    assert source_right < initial or source_left >= initial+side
    reference_sign = -1 if source_right < initial else 1
    bars, palette = [], set()
    for part in components:
        if part == source or part == seed_parts[0]:
            continue
        rs, cs = zip(*part)
        assert len(set(rs)) == 1, 'bars must be one-cell-thick and transverse'
        f, lo, hi = rs[0], min(cs), max(cs)
        assert len(part) == hi-lo+1
        values, frequency = np.unique(a[f, lo:hi+1], return_counts=True)
        assert len(values) == 2 and frequency[0] != frequency[1]
        bar = int(values[frequency.argmax()])
        marker = int(values[frequency.argmin()])
        palette.add((bar, marker))
        positions = tuple(int(c) for c in np.flatnonzero(a[f] == marker))
        # The source uses neither bar color, so the whole row can be counted.
        n = len(positions)
        assert 1 <= n <= 3 and hi-lo+1 >= 2*n+2
        low = positions == tuple(lo+2*j for j in range(n))
        high = positions == tuple(hi-2*j for j in reversed(range(n)))
        assert low != high, 'an asymmetric endpoint command is required'
        bars.append({'f': f, 'lo': lo, 'hi': hi, 'n': n,
                     'sign': 1 if low else -1})
    assert 1 <= len(bars) <= 4 and len(palette) == 1
    bar, marker = palette.pop()
    assert len({bg, anchor, bar, marker}) == 4
    assert not ({bar, marker, bg} & set(motif.values()))
    bars.sort(key=lambda b: b['f'])
    scaled = _scale_pixels(motif, reference_side, side, bg)
    return {'input': a, 'bg': bg, 'anchor': anchor, 'bar': bar,
            'marker': marker, 'direction': direction, 'side': side,
            'reference_side': reference_side, 'source': source, 'motif': motif,
            'scaled': scaled, 'initial': initial, 'bars': bars,
            'reference_sign': reference_sign}


def _trace(data, footprint=True, count=True, mirror=True, scale=True,
           relative=True, strict=True):
    side = data['side']
    stencil = data['scaled'] if scale else data['motif']
    mirror_side = side if scale else data['reference_side']
    current, previous = data['initial'], -1
    h, w = data['input'].shape
    plan = []
    for bar in data['bars']:
        sign = bar['sign']
        reflection = sign*data['reference_sign'] if relative else sign
        copy = _reflect(stencil, mirror_side, reflection if mirror else 1)
        rmin, rmax, cmin, cmax = _extent(copy)
        if not footprint:
            rmin, rmax, cmin, cmax = 0, side-1, 0, side-1
        d = bar['n']+1 if count else 2
        turn = bar['f']-d-rmax
        target = (bar['hi']+d-cmin if sign == 1 else bar['lo']-d-cmax)
        if strict:
            assert 0 < turn <= h-side
            assert turn+rmin > previous, 'reflection/turn must clear the preceding bar'
            assert bar['lo'] <= current and current+side-1 <= bar['hi']
            assert 0 <= current+cmin and current+cmax < w
            assert 0 <= target+cmin and target+cmax < w
            assert bar['f']+rmin >= 0 and bar['f']+rmax < h
            assert target != current
        plan.append({**bar, 'turn': turn, 'target': target, 'copy': copy})
        previous, current = bar['f'], target
    return plan


def _block(a, top, left, height, width, color):
    r0, c0 = max(0, top), max(0, left)
    r1, c1 = min(a.shape[0], top+height), min(a.shape[1], left+width)
    if r0 < r1 and c0 < c1:
        a[r0:r1, c0:c1] = color


def _render(data, plan):
    """Paint the swept square lane first, then the colored station impressions."""
    result = data['input'].copy()
    result[result == data['marker']] = data['bar']
    current, start, side = data['initial'], 0, data['side']
    for station in plan:
        turn, target = station['turn'], station['target']
        _block(result, start, current, turn-start+side, side, data['anchor'])
        _block(result, turn, min(current, target), side,
               abs(target-current)+side, data['anchor'])
        start, current = turn, target
    _block(result, start, current, result.shape[0]-start, side, data['anchor'])
    for station in plan:
        for (r, c), color in station['copy'].items():
            _block(result, station['f']+r, station['target']+c, 1, 1, color)
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    data = _decode(input_grid)
    plan = _trace(data)
    return _physical(_render(data, plan), data['direction'])


def _base_motif(rng, anchor, details, witness=False, small=False, fine=False):
    """Connected growth, arms, hooks and forks; no anchor-colored extra cells."""
    if fine:
        a = 2
        motif = {(r, c): anchor for r in range(a) for c in range(a)}
        # Thin details beside a two-pixel square. Equal-scale copies preserve
        # their sub-anchor boundaries, rather than converting them into tiles.
        motif.update({(2, 0): details[0], (3, 0): details[-1],
                      (3, 1): details[-1], (0, -1): details[-1]})
        if rng.random() < .5:
            motif[2, -1] = details[0]
        return motif, a
    motif = {(0, 0): anchor}
    if witness:
        motif[1, 0] = details[0]
        motif[0, -1] = details[-1]
    style = int(rng.integers(4))
    limit = 2 if small else int(rng.integers(2, 5))
    steps = int(rng.integers(2, 5 if small else 10))
    if style == 1:
        length = int(rng.integers(1, limit+1))
        for r in range(1, length+1):
            motif[r, 0] = details[0]
        for c in range(1, int(rng.integers(2, limit+2))):
            motif[length, -c] = details[-1]
    elif style == 2:
        for c in range(-limit, limit+1):
            if c:
                motif[0, c] = details[0]
        for c in (-limit, limit):
            for r in range(1, int(rng.integers(2, limit+2))):
                motif[r, c] = details[-1]
    else:
        for step in range(steps):
            frontier = sorted({q for r, c in motif for q in
                               ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
                               if q not in motif and -limit <= q[0] <= limit
                               and -limit <= q[1] <= limit})
            if not frontier:
                break
            q = frontier[int(rng.integers(len(frontier)))]
            motif[q] = details[step % len(details)]
    if not witness:
        for _ in range(int(rng.integers(4))):
            motif = {(c, -r): color for (r, c), color in motif.items()}
        if rng.random() < .5:
            motif = {(r, -c): color for (r, c), color in motif.items()}
    if len({r for r, c in motif}) < 2 or len({c for r, c in motif}) < 2:
        motif[1, 0] = details[0]
        motif[0, -1] = details[-1]
    return motif, 1


def _unit_pixels(base, unit):
    return {(r*unit+i, c*unit+j): color for (r, c), color in base.items()
            for i in range(unit) for j in range(unit)}


def _span_choices(width, current, side, n, sign, extent):
    _, _, cmin, cmax = extent
    choices = []
    if not (0 <= current+cmin and current+cmax < width):
        return choices
    for lo in range(current+1):
        for hi in range(max(current+side-1, lo+2*n+1), width):
            target = hi+n+1-cmin if sign == 1 else lo-n-1-cmax
            if 0 <= target+cmin and target+cmax < width:
                choices.append((lo, hi, target))
    return choices


def _find_spans(rng, width, initial, side, ns, signs, stencils):
    failed = set()

    def search(i, current):
        if i == len(ns):
            return []
        if (i, current) in failed:
            return None
        choices = _span_choices(width, current, side, ns[i], signs[i], _extent(stencils[i]))
        for index in rng.permutation(len(choices)):
            lo, hi, target = choices[int(index)]
            rest = search(i+1, target)
            if rest is not None:
                return [(lo, hi, target)]+rest
        failed.add((i, current))
        return None
    return search(0, initial)


def _make_pair(rng, direction, ns, signs, reference_unit, destination_unit,
               witness=False, small=False, fine=False, reference_sign=1,
               minimal=False):
    roles = list(map(int, rng.permutation(10)))
    bg, anchor, bar, marker = roles[:4]
    details = roles[4:4+int(rng.integers(1, 4))]
    if minimal:
        base = {(0, 0): anchor, (1, 0): details[0], (0, -1): details[-1]}
        base_anchor = 1
    else:
        base, base_anchor = _base_motif(rng, anchor, details, witness, small, fine)
    source = _unit_pixels(base, reference_unit)
    scaled = _unit_pixels(base, destination_unit)
    reference_side, side = base_anchor*reference_unit, base_anchor*destination_unit
    if witness and _reflect(scaled, side, -1) == scaled:
        return None
    copies = [_reflect(scaled, side, sign*reference_sign) for sign in signs]
    srmin, srmax, scmin, scmax = _extent(source)
    rmin, rmax, _, _ = _extent(scaled)
    source_height, source_width = srmax-srmin+1, scmax-scmin+1
    # Source is in a launch band; each later turn is beyond the previous bar
    # even for the trailing-most stencil pixel. Slack varies all segment gaps.
    minimum_positions, previous = [], -1
    for i, n in enumerate(ns):
        turn = source_height-rmin+2 if i == 0 else previous-rmin+1
        previous = turn+n+1+rmax
        minimum_positions.append(previous)
    minimum_height = previous+max(rmax+1, side+1)+1
    if minimum_height > 30 or source_width+(_extent(copies[0])[3]-_extent(copies[0])[2]+1)+3 > 30:
        return None
    height = 30 if minimal else int(rng.integers(minimum_height, 31))
    slack = height-minimum_height
    allocation = rng.multinomial(slack, np.full(len(ns)+1, 1/(len(ns)+1)))
    positions = []
    accumulated = 0
    for i, f in enumerate(minimum_positions):
        accumulated += int(allocation[i])
        positions.append(f+accumulated)
    min_width = max(8, source_width+_extent(copies[0])[3]-_extent(copies[0])[2]+4,
                    2*max(ns)+2+_extent(copies[0])[3]-_extent(copies[0])[2]+max(ns)+1)
    if min_width > 30:
        return None
    width = 30 if minimal else int(rng.integers(min_width, 31))
    if rng.random() < .18 and height >= min_width:
        width = height
    if rng.random() < .12:
        width = 30
    initial_candidates = list(map(int, rng.permutation(np.arange(1, width-side))))
    plan = placement = None
    cmin, cmax = _extent(copies[0])[2:]
    for initial in initial_candidates:
        if not (0 <= initial+cmin and initial+cmax < width):
            continue
        source_columns = [left-scmin for left in range(1, width-source_width)
                          if (left+source_width <= initial+cmin-1 if reference_sign == -1
                              else left >= initial+cmax+2)]
        if not source_columns:
            continue
        spans = _find_spans(rng, width, initial, side, ns, signs, copies)
        if spans is not None:
            placement = (1-srmin, int(rng.choice(source_columns)))
            plan = spans
            break
    if plan is None:
        return None
    a = np.full((height, width), bg, dtype=int)
    a[:side, initial:initial+side] = anchor
    sr, sc = placement
    for (r, c), color in source.items():
        a[sr+r, sc+c] = color
    stations = []
    for f, n, sign, (lo, hi, target), copy in zip(positions, ns, signs, plan, copies):
        a[f, lo:hi+1] = bar
        for j in range(n):
            a[f, lo+2*j if sign == 1 else hi-2*j] = marker
        stations.append({'f': f, 'lo': lo, 'hi': hi, 'n': n, 'sign': sign,
                         'turn': f-n-1-rmax, 'target': target, 'copy': copy})
    if np.count_nonzero(a == bg) <= a.size//2:
        return None
    # Labels come from the sampled base geometry and geometric construction.
    # The input-only decoder never sees this plan.
    data = {'input': a, 'anchor': anchor, 'bar': bar, 'marker': marker,
            'side': side, 'initial': initial}
    out = _render(data, stations)
    pair = {'input': _physical(a, direction), 'output': _physical(out, direction)}
    try:
        _analyze(pair)
    except AssertionError:
        return None
    return pair


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    num_train = _integer(num_train, 'num_train', 2, 10)
    num_test = _integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    # A curriculum controls distributions only; all output conventions are fixed.
    shift = int(rng.integers(3))  # held-out heading / longer sequence / larger cargo
    directions = list(map(int, rng.permutation(4)))
    train_dirs = directions[:3] if shift == 0 else directions
    test_dirs = directions[3:] if shift == 0 else directions
    first_sign = int(rng.choice((-1, 1)))
    first_reference = int(rng.choice((-1, 1)))
    puzzle, seen = {'train': [], 'test': []}, set()

    def add(split, obligation=None):
        for attempt in range(160):
            direction = int(rng.choice(train_dirs if split == 'train' else test_dirs))
            witness = obligation is not None
            fine = False
            reference_sign = int(rng.choice((-1, 1)))
            if obligation == 'routing':
                count = 2 if shift == 1 or attempt > 40 else int(rng.choice((2, 3)))
                ns = list(map(int, rng.permutation([1, 2])))
                if count == 3:
                    ns.append(int(rng.integers(1, 4)))
                signs = [first_sign]*2 + ([-first_sign] if count == 3 else [])
                reference_unit = destination_unit = 1
                reference_sign = first_reference
            elif obligation == 'scale':
                ns = [3, int(rng.integers(1, 3))]
                if rng.random() < .5:
                    ns.reverse()
                signs = [-first_sign, first_sign]
                reference_unit, destination_unit = ((1, 2) if shift == 2 or rng.random() < .5 else (2, 1))
                reference_sign = -first_reference
            else:
                count = (3 if split == 'test' and shift == 1 else
                         int(rng.choice((1, 2) if shift == 1 else (1, 1, 2, 2, 3, 4))))
                ns = list(map(int, rng.integers(1, 4, size=count)))
                signs = list(map(int, rng.choice((-1, 1), size=count)))
                destination_unit = 3 if split == 'test' and shift == 2 else int(rng.choice((1, 1, 2)))
                reference_unit = int(rng.choice((1, 1, 2, 3)))
                fine = (count == 1 and destination_unit != 3 and rng.random() < .20)
                if fine:
                    reference_unit = destination_unit = 1
            small = len(ns) >= 2 or destination_unit == 3 or attempt > 40
            pair = _make_pair(rng, direction, ns, signs, reference_unit,
                              destination_unit, witness, small, fine, reference_sign)
            if pair is not None and _key(pair['input']) not in seen:
                puzzle[split].append(pair)
                seen.add(_key(pair['input']))
                return
        # A small L has a feasible 30x30 plan in every curriculum. Enumerating
        # 90 background/anchor assignments guarantees a unique input against
        # at most 13 existing pairs. This fallback follows the same rule.
        direction = int(rng.choice(train_dirs if split == 'train' else test_dirs))
        if obligation == 'routing':
            ns, signs, ru, du, ref = [1, 2], [first_sign]*2, 1, 1, first_reference
        elif obligation == 'scale':
            ns, signs, ru, du, ref = [3, 1], [-first_sign, first_sign], 1, 2, -first_reference
        elif split == 'test' and shift == 1:
            ns, signs, ru, du, ref = [1, 2, 1], [1, 1, -1], 1, 1, 1
        else:
            ns, signs, ru, du, ref = [2], [1], 1, 3 if split == 'test' and shift == 2 else 1, 1
        pair = _make_pair(rng, direction, ns, signs, ru, du, reference_sign=ref, minimal=True)
        assert pair is not None, 'small-L fallback must be feasible'
        d = _decode(pair['input'])
        old_roles = [d['bg'], d['anchor']]
        old_roles += [c for c in range(10) if c not in old_roles]
        for bg in range(10):
            for anchor in range(10):
                if bg == anchor:
                    continue
                new_roles = [bg, anchor]+[c for c in range(10) if c not in (bg, anchor)]
                palette = np.empty(10, dtype=int)
                palette[old_roles] = new_roles
                candidate = {k: palette[v] for k, v in pair.items()}
                if _key(candidate['input']) not in seen:
                    seen.add(_key(candidate['input']))
                    puzzle[split].append(candidate)
                    return
        raise AssertionError('finite fallback palette space exhausted')

    add('train', 'routing')
    add('train', 'scale')
    for _ in range(num_train-2):
        add('train')
    for _ in range(num_test):
        add('test')
    for split in puzzle:
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    a, b = _grid(pair['input']), _grid(pair['output'])
    assert a.shape == b.shape
    data = _decode(a)
    plan = _trace(data)
    canonical = _forward(b, data['direction'])
    expected = _render(data, plan)
    assert np.array_equal(canonical, expected), 'incorrect convoy, station, or normalization'
    assert not np.array_equal(a, b)
    # Verify preservation and the actual empty-lane geometry, not just hashes.
    source = data['source']
    assert all(canonical[p] == data['input'][p] for p in source)
    structure = (data['input'] == data['bar']) | (data['input'] == data['marker'])
    assert np.all(canonical[structure] == data['bar'])
    assert data['marker'] not in set(map(int, np.unique(canonical)))
    side, current, start = data['side'], data['initial'], 0
    lane = np.zeros(data['input'].shape, dtype=int)
    for p in plan:
        _block(lane, start, current, p['turn']-start+side, side, 1)
        _block(lane, p['turn'], min(current, p['target']), side,
               abs(p['target']-current)+side, 1)
        current, start = p['target'], p['turn']
        rmin, rmax, cmin, cmax = _extent(p['copy'])
        assert p['f']-(p['turn']+rmax) == p['n']+1
        nearest = (p['target']+cmin-p['hi'] if p['sign'] == 1 else
                   p['lo']-(p['target']+cmax))
        assert nearest == p['n']+1
        for (r, c), color in p['copy'].items():
            assert canonical[p['f']+r, p['target']+c] == color
    _block(lane, start, current, lane.shape[0]-start, side, 1)
    assert not np.any(lane[structure]) and not any(lane[p] for p in source)
    # Check the carried shape throughout each motion, including its reflection
    # at a bend. Forward launch/exit clipping is allowed for the virtual cargo;
    # all painted station impressions are complete and in bounds.
    current, start, carried = data['initial'], 0, plan[0]['copy']

    def check_pose(row, col, stencil):
        for dr, dc in stencil:
            r, c = row+dr, col+dc
            assert 0 <= c < data['input'].shape[1]
            if 0 <= r < data['input'].shape[0]:
                assert not structure[r, c] and (r, c) not in source

    for p in plan:
        for row in range(start, p['turn']+1):
            check_pose(row, current, carried)
        carried = p['copy']
        for col in range(min(current, p['target']), max(current, p['target'])+1):
            check_pose(p['turn'], col, carried)
        start, current = p['turn'], p['target']
    for row in range(start, data['input'].shape[0]-side+1):
        check_pose(row, current, carried)
    # The lane and all impressions form one component, excluding bars/source.
    moving = (canonical != data['bg']) & ~structure
    for p in source:
        moving[p] = False
    assert len(_components(moving.astype(int), 0)) == 1
    data['plan'] = plan
    return data


def _ablation_differs(data, **options):
    plan = _trace(data, strict=False, **options)
    return not np.array_equal(_render(data, plan), _render(data, data['plan']))


def _fine_geometry(data):
    motif, side, bg = data['motif'], data['reference_side'], data['bg']
    for (r, c), color in motif.items():
        for dr, dc, boundary in ((-1, 0, r), (1, 0, r+1),
                                 (0, -1, c), (0, 1, c+1)):
            if motif.get((r+dr, c+dc), bg) != color and boundary % side:
                return True
    return False


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            key = _key(_grid(pair['input']))
            assert key not in seen, 'duplicate input'
            seen.add(key)
            analyses[split].append(_analyze(pair))
    train = analyses['train']
    assert {b['n'] for a in train for b in a['bars']} == {1, 2, 3}
    assert {b['sign'] for a in train for b in a['bars']} == {-1, 1}
    assert any(any(x['sign'] == y['sign'] for x, y in zip(a['bars'], a['bars'][1:])) for a in train)
    assert any(a['side'] != a['reference_side'] for a in train)
    assert {a['reference_sign'] for a in train} == {-1, 1}
    for options in ({'footprint': False}, {'count': False}, {'mirror': False},
                    {'scale': False}, {'relative': False}):
        assert any(_ablation_differs(a, **options) for a in train), f'undemonstrated distinction: {options}'
    test = analyses['test']
    assert (bool({a['direction'] for a in test}-{a['direction'] for a in train})
            or max(len(a['bars']) for a in test) > max(len(a['bars']) for a in train)
            or max(a['side'] for a in test) > max(a['side'] for a in train)), 'missing structural train/test shift'


def self_test() -> dict:
    puzzles, shapes, layouts, ratios, headings, bar_counts, marker_counts = set(), set(), set(), set(), set(), set(), set()
    transitions, spans, anchors, fine_count = set(), set(), set(), 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.add(_puzzle_key(p))
        if seed < 3:
            assert _puzzle_key(p) == _puzzle_key(generate_puzzle(seed))
        for split in p:
            for pair in p[split]:
                a = _analyze(pair)
                assert np.array_equal(solve(pair['input']), pair['output'])
                shapes.add(pair['input'].shape)
                headings.add((a['direction'], split))
                bar_counts.add((len(a['bars']), split))
                marker_counts.update(b['n'] for b in a['bars'])
                transitions.update((x['sign'], y['sign']) for x, y in zip(a['bars'], a['bars'][1:]))
                ratios.add((a['side'], a['reference_side']))
                anchors.add(a['side'])
                spans.add(_extent(a['scaled']))
                color_map = {a['bg']: 0, a['anchor']: 1, a['bar']: 2, a['marker']: 3}
                normalized = np.empty_like(pair['input'])
                for color in pair['input'].flat:
                    color_map.setdefault(int(color), len(color_map))
                for color, role in color_map.items():
                    normalized[pair['input'] == color] = role
                layouts.add(_key(normalized))
                fine_count += _fine_geometry(a)
    assert len(puzzles) == 42 and len(shapes) >= 35 and len(layouts) >= 180
    assert headings == {(d, split) for d in range(4) for split in ('train', 'test')}
    assert {(n, split) for n in (1, 2, 3) for split in ('train', 'test')} <= bar_counts
    assert marker_counts == {1, 2, 3} and anchors >= {1, 2, 3}
    assert transitions == {(-1, -1), (-1, 1), (1, -1), (1, 1)}
    assert len(ratios) >= 6 and len(spans) >= 25 and fine_count > 0
    # Geometric fallback fixtures across reference polarity and both sides.
    rng = np.random.default_rng(404)
    for ref in (-1, 1):
        for sign in (-1, 1):
            for direction in range(4):
                for ns, signs, ru, du in (([1, 2], [sign, sign], 1, 1),
                                          ([3, 1], [sign, -sign], 1, 2),
                                          ([1, 2, 1], [sign, sign, -sign], 1, 1),
                                          ([2], [sign], 1, 3)):
                    pair = _make_pair(rng, direction, ns, signs, ru, du,
                                      reference_sign=ref, minimal=True)
                    assert pair is not None
                    _analyze(pair)
    # Literal cell-level regressions: a 1-cell source is carried at scale two.
    # Its leading arm moves the turn from row 7 to row 5; the opposite-side
    # arm changes the right station from column 16 to 18. No tracing helper
    # supplies these expected coordinates or rectangle sizes.
    for sign in (-1, 1):
        inp = np.zeros((25, 28), dtype=int)
        inp[0:2, 10:12] = 1
        inp[2, 23], inp[2, 22], inp[3, 23] = 1, 4, 6
        inp[10, 8:15] = 5
        inp[10, 8 if sign == 1 else 14] = 8
        expected = inp.copy()
        expected[10, 8:15] = 5
        expected[0:7, 10:12] = 1
        if sign == 1:
            expected[5:7, 10:20] = 1
            expected[5:25, 18:20] = 1
            expected[10:12, 16:18] = 4
            expected[12:14, 18:20] = 6
        else:
            expected[5:7, 3:12] = 1
            expected[5:25, 3:5] = 1
            expected[10:12, 5:7] = 4
            expected[12:14, 3:5] = 6
        for direction in range(4):
            palette = np.array([7, 2, 0, 9, 3, 1, 4, 6, 5, 8])
            a, b = palette[_physical(inp, direction)], palette[_physical(expected, direction)]
            assert np.array_equal(solve(a), b)
            _analyze({'input': a, 'output': b})
    # Every legal explicit count pair, and robust type/range rejections.
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            p = generate_puzzle(800+10*train+test, train, test)
            assert (len(p['train']), len(p['test'])) == (train, test)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3, 1.5), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    for seed in (-1, True, 1.5):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal seed accepted')
    # Exact scaling accepts aligned reductions and refuses ambiguous ones.
    motif = {(r, c): 1 for r in range(2) for c in range(2)}
    motif[2, 0] = 4
    assert _scale_pixels(motif, 2, 4, 0)[4, 0] == 4
    try:
        _scale_pixels(motif, 2, 1, 0)
    except AssertionError:
        pass
    else:
        raise AssertionError('ambiguous reduction accepted')
    p = generate_puzzle(91, 2, 1)
    broken = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    q = broken['test'][0]
    changed = np.argwhere(q['input'] != q['output'])[0]
    q['output'][tuple(changed)] = q['input'][tuple(changed)]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged output accepted')
    return {'seeds': 42, 'distinct_puzzles': len(puzzles), 'shapes': len(shapes),
            'color_normalized_inputs': len(layouts), 'side_pairs': sorted(ratios),
            'stencil_extents': len(spans), 'fine_geometry_pairs': fine_count}


if __name__ == '__main__':
    print(self_test())
