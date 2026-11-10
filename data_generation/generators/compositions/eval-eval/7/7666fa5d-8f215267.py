"""Cross-section elections: diagonal interval geometry meets object voting.

No parent code, files, latent seed state, or palette lookup is used by solve.
Generation and self_test write no files. Diagnostic serialization, if requested
by a caller, must use compact JSON; digit rows are preferable for inspection.
"""
from __future__ import annotations

import hashlib
import numpy as np


_FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
_EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name):
    _require(isinstance(value, (int, np.integer)) and
             not isinstance(value, (bool, np.bool_)), name + " must be an integer")
    return int(value)


def _counts(num_train, num_test):
    a, b = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal train/test counts")
    return a, b


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= x <= 30 for x in g.shape), "grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer) and
             bool(np.all((g >= 0) & (g <= 9))), "grid colors")
    return g.astype(np.int8, copy=False)


def _connected(cells, offsets):
    todo = set(cells)
    stack = [todo.pop()]
    while stack:
        r, c = stack.pop()
        for dr, dc in offsets:
            p = (r + dr, c + dc)
            if p in todo:
                todo.remove(p)
                stack.append(p)
    return not todo


def _parse(value):
    """Identify straight diagonal rails and all other monochromatic objects."""
    g = _grid(value)
    colors, frequencies = np.unique(g, return_counts=True)
    bg = int(colors[np.argmax(frequencies)])
    _require(int(frequencies.max()) * 2 > g.size, "background must be a strict majority")
    unseen = set(map(tuple, np.argwhere(g != bg).tolist()))
    parts = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        color = int(g[start])
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in _EIGHT:
                p = (r + dr, c + dc)
                if p in unseen and int(g[p]) == color:
                    unseen.remove(p)
                    stack.append(p)
        parts.append(dict(color=color, cells=tuple(sorted(cells))))
    rails, objects = [], []
    for part in parts:
        cells = part['cells']
        r0, c0 = cells[0]
        slope = None
        if len(cells) >= 3:
            for s in (-1, 1):
                if cells == tuple((r0 + i, c0 + s*i) for i in range(len(cells))):
                    slope = s
        if slope is None:
            _require(1 <= len(cells) <= 8 and _connected(cells, _FOUR),
                     "objects must be orthogonally connected, with area 1..8")
            objects.append(part)
        else:
            ts = [r + slope*c for r, c in cells]
            rails.append(dict(**part, slope=slope, q=c0-slope*r0,
                              lo=min(ts), hi=max(ts)))
    _require(2 <= len(rails) <= 5, "expected two through five rails")
    _require(len({r['slope'] for r in rails}) == 1, "rails must be parallel")
    _require(len({r['color'] for r in rails}) == len(rails), "one rail per category color")
    rails.sort(key=lambda r: r['q'])
    _require(all(b['q']-a['q'] >= 3 for a, b in zip(rails, rails[1:])),
             "rails need distinct, separated transverse positions")
    # Separate objects remain unambiguous even under the stronger 8-neighbor
    # segmentation; in particular a diagonal line cannot be an ordinary object.
    owner = {p: i for i, part in enumerate(parts) for p in part['cells']}
    _require(all(owner.get((r+dr, c+dc), i) == i
                 for (r, c), i in owner.items() for dr, dc in _EIGHT),
             "different components must have a background moat")
    votes = {r['color']: sum(o['color'] == r['color'] for o in objects) for r in rails}
    areas = {r['color']: sum(len(o['cells']) for o in objects if o['color'] == r['color'])
             for r in rails}
    return dict(grid=g, background=bg, rails=rails, objects=objects,
                votes=votes, areas=areas, slope=rails[0]['slope'])


def _coordinates(shape, rails):
    rows, cols = np.indices(shape)
    s = rails[0]['slope']
    return cols-s*rows, rows+s*cols


def _envelope(shape, rails):
    q, t = _coordinates(shape, rails)
    mask = np.zeros(shape, dtype=bool)
    # Union of pairwise interval intersections is exactly the outer-active
    # envelope. Used for diagnostics/placement, not for choosing a winner.
    for i, a in enumerate(rails):
        for b in rails[i+1:]:
            mask |= ((t >= max(a['lo'], b['lo'])) & (t <= min(a['hi'], b['hi'])) &
                     (q > a['q']) & (q < b['q']))
    return mask


def _paint(shape, bg, rails, weights, alternative=''):
    """Sweep continuous endpoint intervals; erase objects, preserve rails."""
    out = np.full(shape, bg, dtype=np.int8)
    rail_mask = np.zeros(shape, dtype=bool)
    for rail in rails:
        rr, cc = zip(*rail['cells'])
        out[rr, cc] = rail['color']
        rail_mask[rr, cc] = True
    q, t = _coordinates(shape, rails)
    for level in range(int(t.min()), int(t.max())+1):
        active = [r for r in rails if r['lo'] <= level <= r['hi']]
        if alternative == 'all_boundaries':
            active = rails if min(r['lo'] for r in rails) <= level <= max(r['hi'] for r in rails) else []
        elif alternative == 'pixel_centers':
            active = [r for r in active if (level-r['lo']) % 2 == 0]
        elif alternative == 'zero_boundaries':
            active = [r for r in active if weights[r['color']] > 0]
        if len(active) < 2:
            continue
        electorate = rails if alternative == 'global' else active
        largest = max(weights[r['color']] for r in electorate)
        winners = [r for r in electorate if weights[r['color']] == largest]
        if len(winners) != 1 and alternative != 'tie_first':
            continue
        winner = winners[0]['color']
        band = (q > active[0]['q']) & (q < active[-1]['q'])
        if alternative == 'neighbors':
            band = np.zeros(shape, dtype=bool)
            for a, b in zip(rails, rails[1:]):
                if a in active and b in active:
                    band |= (q > a['q']) & (q < b['q'])
        out[(t == level) & band & ~rail_mask] = winner
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """All conventions and vote counts are recovered from this input only."""
    d = _parse(input_grid)
    return _paint(d['grid'].shape, d['background'], d['rails'], d['votes'])


def _shape(rng, area):
    """Geometry is sampled independently of category multiplicity."""
    kind = int(rng.integers(4))
    if kind == 0:
        cells = {(0, c) for c in range(area)}
    elif kind == 1:
        width = int(rng.integers(1, min(area, 4)+1))
        cells = {(i//width, i % width) for i in range(area)}
    elif kind == 2:
        arm = int(rng.integers(1, area+1))
        cells = {(0, c) for c in range(arm)} | {(r, 0) for r in range(area-arm+1)}
    else:
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _FOUR}-cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    for _ in range(int(rng.integers(4))):
        cells = {(c, -r) for r, c in cells}
    if rng.random() < .5:
        cells = {(r, -c) for r, c in cells}
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _abstract_rails(rng, colors, witness, slope):
    n = len(colors)
    qs = [0]
    for _ in range(n-1):
        qs.append(qs[-1] + int(rng.integers(6 if witness else 4, 10)))
    intervals = []
    mode = int(rng.integers(5))
    order = list(map(int, rng.permutation(n)))
    stagger_rank = {index: rank for rank, index in enumerate(order)}
    for i, q in enumerate(qs):
        if witness:
            if i in (0, n-1):
                lo, hi = -int(rng.integers(8, 12)), int(rng.integers(8, 12))
                lo -= (lo-q) % 2
                hi -= (hi-q) % 2
            else:
                lo = -int(rng.integers(1, 5))
                lo -= (lo-q) % 2
                hi = lo + 2*int(rng.integers(2, 5))
        else:
            length = int(rng.integers(3, 11))
            if mode == 0:
                lo = -length+1
            elif mode == 1:
                lo = 0
            elif mode == 2:
                lo = -2*(length-1)
            elif mode == 3:
                lo = 4*stagger_rank[i]-2*n
            else:
                lo = int(rng.integers(-12, 8))
            lo -= (lo-q) % 2
            hi = lo+2*(length-1)
        intervals.append((lo, hi))
    s = slope
    rails = []
    for color, q, (lo, hi) in zip(colors, qs, intervals):
        cells = tuple(((t-s*q)//2, (q+s*t)//2) for t in range(lo, hi+1, 2))
        rails.append(dict(color=color, cells=cells, q=q, lo=lo, hi=hi, slope=s))
    return rails


def _place(rng, shape, bg, rails, specs, votes):
    g = np.full(shape, bg, dtype=np.int8)
    blocked = np.zeros(shape, dtype=bool)
    for rail in rails:
        for r, c in rail['cells']:
            g[r, c] = rail['color']
            blocked[max(0, r-1):r+2, max(0, c-1):c+2] = True
    fill = _paint(shape, bg, rails, votes) != bg
    envelope = _envelope(shape, rails)
    # Restrained witnesses are placed before unconstrained shapes; every legal
    # integer origin is a candidate. There is no hidden placement lattice.
    order = sorted(range(len(specs)), key=lambda i: (specs[i][2] == 'any', -len(specs[i][1]), i))
    for i in order:
        color, cells, zone = specs[i]
        h, w = 1+max(r for r, c in cells), 1+max(c for r, c in cells)
        if h > shape[0] or w > shape[1]:
            return None
        allowed = ~blocked
        if zone == 'inside':
            allowed = allowed & fill
        elif zone == 'outside':
            allowed = allowed & ~envelope
        candidates = np.ones((shape[0]-h+1, shape[1]-w+1), dtype=bool)
        for dr, dc in cells:
            candidates &= allowed[dr:dr+candidates.shape[0], dc:dc+candidates.shape[1]]
        rr, cc = np.nonzero(candidates)
        if not len(rr):
            return None
        j = int(rng.integers(len(rr)))
        r0, c0 = int(rr[j]), int(cc[j])
        for dr, dc in cells:
            r, c = r0+dr, c0+dc
            g[r, c] = color
            blocked[max(0, r-1):r+2, max(0, c-1):c+2] = True
    return g


def _translated(rails, dr, dc):
    result = []
    for a in rails:
        s = a['slope']
        result.append(dict(a, cells=tuple((r+dr, c+dc) for r, c in a['cells']),
                           q=a['q']+dc-s*dr, lo=a['lo']+dr+s*dc, hi=a['hi']+dr+s*dc))
    return result


def _alternative(d, name):
    weights = d['areas'] if name == 'area' else (
        {c: int(n > 0) for c, n in d['votes'].items()} if name == 'presence' else d['votes'])
    if name in ('inside_only', 'outside_only'):
        envelope = _envelope(d['grid'].shape, d['rails'])
        weights = {c: sum(o['color'] == c and
                         all(bool(envelope[p]) == (name == 'inside_only') for p in o['cells'])
                         for o in d['objects']) for c in d['votes']}
    return _paint(d['grid'].shape, d['background'], d['rails'], weights,
                  '' if name in ('area', 'presence', 'inside_only', 'outside_only') else name)


_ALTERNATIVES = ('area', 'presence', 'global', 'neighbors', 'pixel_centers',
                 'tie_first', 'all_boundaries', 'zero_boundaries', 'inside_only', 'outside_only')


def _make_pair(rng, bg, colors, counts, forced, distractors, witness, slope, seen, focus=None):
    votes = dict(zip(colors, counts))
    for attempt in range(64):
        abstract = _abstract_rails(rng, colors, witness, slope)
        all_cells = [p for a in abstract for p in a['cells']]
        r0, c0 = min(r for r, c in all_cells), min(c for r, c in all_cells)
        span = (max(r for r, c in all_cells)-r0+1, max(c for r, c in all_cells)-c0+1)
        if max(span) > 30:
            continue
        specs = []
        for color, count in zip(colors, counts):
            areas = forced.get(color, [int(rng.integers(1, 9)) for _ in range(count)])
            for j, area in enumerate(areas):
                zone = ('inside' if j == 0 else 'outside' if j == 1 else 'any') if witness == 1 and count == 3 else 'any'
                specs.append((color, _shape(rng, area), zone))
        for j, color in enumerate(distractors):
            zone = 'inside' if witness == 1 and j == 0 else 'any'
            specs.append((color, _shape(rng, 1 if zone == 'inside' else int(rng.integers(1, 9))), zone))
        dims = tuple(max(s, int(rng.integers(18, 26))) for s in span)
        for expansion in range(3):
            shape = tuple(min(30, d+3*expansion) for d in dims)
            if attempt >= 40:
                shape = (30, 30)
            top = int(rng.integers(shape[0]-span[0]+1))
            left = int(rng.integers(shape[1]-span[1]+1))
            rails = _translated(abstract, top-r0, left-c0)
            grid = _place(rng, shape, bg, rails, specs, votes)
            if grid is None or np.count_nonzero(grid == bg)*2 <= grid.size:
                continue
            key = (grid.shape, grid.tobytes())
            if key in seen:
                continue
            output = _paint(shape, bg, rails, votes)
            if focus is not None:
                # The new count must actually matter: four defeats a visible
                # three-vote competitor on at least one painted cross-section.
                contenders = [a for a in rails if a['color'] in focus]
                if not np.any((output == focus[0]) & _envelope(shape, contenders)):
                    continue
            if witness:
                d = _parse(grid)
                needed = ('area', 'presence', 'global', 'neighbors', 'pixel_centers',
                          'inside_only', 'outside_only') if witness == 1 else ('tie_first', 'zero_boundaries')
                if any(np.array_equal(output, _alternative(d, name)) for name in needed):
                    continue
            seen.add(key)
            return dict(input=grid, output=output)
    raise RuntimeError("bounded layout/packing attempts exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed')
    _require(seed >= 0, 'seed must be nonnegative')
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    anchor, herald, old, novel = map(int, rng.permutation(10)[:4])
    collision_area = int(rng.integers(4, 8))
    cut = int(rng.integers(1, collision_area))
    puzzle, seen = dict(train=[], test=[]), set()
    first_bg = None
    first_slope = int(rng.choice((-1, 1)))
    for index in range(nt+nq):
        train = index < nt
        local = index if train else index-nt
        witness = index+1 if index < 2 else 0
        slope = first_slope if index == 0 else -first_slope if index == 1 else int(rng.choice((-1, 1)))
        forced = {}
        focus = None
        if witness:
            excluded = {anchor, herald, old, novel}
            bg = int(rng.choice([c for c in range(10) if c not in excluded and c != first_bg]))
            if index == 0:
                first_bg = bg
            n = int(rng.choice((3, 3, 4)))
            other = old if index == 0 else int(rng.choice([c for c in range(10) if c not in excluded | {bg}]))
            outer = [anchor, other]
            rng.shuffle(outer)
            middle = [herald]
            if n == 4:
                middle.append(int(rng.choice([c for c in range(10) if c not in set(outer+middle+[bg, novel, old])])) )
                rng.shuffle(middle)
            colors = [outer[0]]+middle+[outer[1]]
            row = {anchor: 1 if index == 0 else 2,
                   herald: 3 if index == 0 else 2, other: 2 if index == 0 else 0}
            counts = [row.get(c, int(rng.integers(0, 2))) for c in colors]
            if index == 0:
                forced = {anchor: [collision_area], herald: [1, 1, 1],
                          other: [int(rng.integers(3, 6)), int(rng.integers(3, 6))]}
                distractors = [novel]
            else:
                forced = {anchor: [cut, collision_area-cut],
                          herald: [4, collision_area-3]}
                distractors = [old]
        else:
            bg = int(rng.integers(10))
            n = int(rng.integers(2, 6))
            if not train and local == 0 and bg == novel:
                bg = (bg+1) % 10
            pool = [c for c in range(10) if c != bg and (not train or c != novel)]
            colors = list(map(int, rng.choice(pool, n, replace=False)))
            counts = list(map(int, rng.integers(0, 4 if train else 5, n)))
            if rng.random() < .08:
                counts = [0]*n
            if not train and local == 0:
                if novel not in colors:
                    colors[int(rng.integers(n))] = novel
                counts = [min(x, 3) for x in counts]
                counts[colors.index(novel)] = 4
                competitor = int(rng.choice([c for c in colors if c != novel]))
                counts[colors.index(competitor)] = 3
                focus = (novel, competitor)
            while sum(counts) > 14:
                choices = [i for i, c in enumerate(colors) if c != novel and
                           (focus is None or c != focus[1]) and counts[i] > 0]
                counts[int(rng.choice(choices))] -= 1
            dpool = [c for c in range(10) if c != bg and c not in colors]
            nd = max(int(not sum(counts)), int(rng.integers(0, 4)))
            distractors = list(map(int, rng.choice(dpool, nd, replace=True)))
        pair = _make_pair(rng, bg, colors, counts, forced, distractors, witness, slope, seen, focus)
        puzzle['train' if train else 'test'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def puzzle_metrics(puzzle):
    """Compact semantic measurements recovered from pixels, without seed data."""
    records = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            d = _parse(pair['input'])
            g, rails, bg = d['grid'], d['rails'], d['background']
            envelope = _envelope(g.shape, rails)
            rail_cells = {p for a in rails for p in a['cells']}
            fill_colors = {int(pair['output'][r, c]) for r, c in np.argwhere(envelope)
                           if (int(r), int(c)) not in rail_cells and int(pair['output'][r, c]) != bg}
            masks = []
            for o in d['objects']:
                r0, c0 = min(r for r, c in o['cells']), min(c for r, c in o['cells'])
                masks.append(tuple((r-r0, c-c0) for r, c in o['cells']))
            records.append(dict(split=split, shape=g.shape, background=bg, slope=d['slope'],
                rails=len(rails), colors=[a['color'] for a in rails], votes=list(d['votes'].values()),
                areas=list(d['areas'].values()), lengths=[len(a['cells']) for a in rails],
                gaps=[b['q']-a['q'] for a, b in zip(rails, rails[1:])],
                endpoints=[(a['q'], a['lo'], a['hi']) for a in rails],
                objects=len(d['objects']), object_areas=[len(o['cells']) for o in d['objects']],
                masks=masks, painted_colors=sorted(fill_colors),
                distractors=sum(o['color'] not in d['votes'] for o in d['objects']),
                outside_objects=sum(all(not envelope[p] for p in o['cells']) for o in d['objects']),
                inside_objects=sum(all(envelope[p] for p in o['cells']) for o in d['objects']),
                edge=any(r in (0, g.shape[0]-1) or c in (0, g.shape[1]-1) for r, c in rail_cells),
                alternative_differences={name: int(np.sum(pair['output'] != _alternative(d, name)))
                                         for name in _ALTERNATIVES}))
    return records


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split schema')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'output dimensions')
            key = (inp.shape, inp.tobytes())
            _require(key not in seen, 'duplicate inputs')
            seen.add(key)
            d = _parse(inp)
            _require(np.array_equal(out, solve(inp)), 'wrong election, erasure or rail preservation')
            _require(not np.array_equal(inp, out), 'unchanged pair')
            if split == 'train':
                training.append((d, out))
    # These checks concern demonstrations, not hidden sampler parameters.
    for name in _ALTERNATIVES:
        _require(any(not np.array_equal(out, _alternative(d, name)) for d, out in training),
                 'missing training evidence against ' + name)
    examples = [(c, d['votes'][c], d['areas'][c]) for d, out in training for c in d['votes']]
    _require(any(c == cc and a == aa and n != nn for c, n, a in examples for cc, nn, aa in examples),
             'missing same-color equal-area different-count witness')
    _require(any(n == nn and a != aa for c, n, a in examples for cc, nn, aa in examples),
             'missing equal-count different-area witness')
    categories = {c for d, out in training for c in d['votes']}
    discarded = {o['color'] for d, out in training for o in d['objects'] if o['color'] not in d['votes']}
    _require(bool(categories & discarded), 'missing local color-role evidence')
    _require(len({d['background'] for d, out in training}) >= 2, 'missing background-role variation')
    changing = False
    for d, out in training:
        cells = {p for rail in d['rails'] for p in rail['cells']}
        painted = {int(out[r, c]) for r, c in np.argwhere(_envelope(out.shape, d['rails']))
                   if (int(r), int(c)) not in cells and int(out[r, c]) != d['background']}
        changing |= len(painted) >= 2
    _require(changing, 'missing changing local winners')


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                g = _grid(pair[role])
                h.update(bytes(g.shape))
                h.update(g.tobytes())
    return h.hexdigest()


def self_test():
    """42 streamed seeds, all count combinations, semantic negative checks.

    At most two puzzles plus small feature sets are retained. No disk I/O,
    execution histories, logs, payload copies, workers or unbounded retries.
    """
    # Analytical 17x17 fixture, independent of the random input constructor.
    # Outer rails: q=8,20 and -6<=t<=6; short middle: q=14, 0<=t<=4.
    # Votes 1/3/2 imply paint 2 at t=0..4, otherwise 3 at t=-6..6.
    fixture = np.zeros((17, 17), dtype=np.int8)
    rail_cells = []
    for color, q, lo, hi in ((1, 8, -6, 6), (2, 14, 0, 4), (3, 20, -6, 6)):
        for t in range(lo, hi+1, 2):
            r, c = (t+q)//2, (q-t)//2
            fixture[r, c] = color
            rail_cells.append((r, c, color))
    fixture[0:2, 0:2] = 1
    fixture[(0, 3, 15), (12, 15, 0)] = 2
    fixture[(0, 16), (16, 16)] = 3
    expected_fixture = np.zeros_like(fixture)
    for r in range(17):
        for c in range(17):
            if 8 < r+c < 20 and -6 <= r-c <= 6:
                expected_fixture[r, c] = 2 if 0 <= r-c <= 4 else 3
    for r, c, color in rail_cells:
        expected_fixture[r, c] = color
    assert np.array_equal(solve(fixture), expected_fixture)
    assert expected_fixture[4, 7] == 3  # odd t: between pixel centers
    assert expected_fixture[6, 8] == 3  # inactive inner rail is bypassed
    assert expected_fixture[6, 6] == 2 and expected_fixture[0, 0] == 0
    split = fixture.copy()
    split[0:2, 0:2] = 0
    split[0, 0:2] = split[3, 0:2] = 1
    assert np.count_nonzero(split == 1) == np.count_nonzero(fixture == 1)
    assert _parse(split)['votes'][1] == 2 and solve(split)[4, 7] == 0
    tie = fixture.copy()
    tie[15, 0] = 0
    assert solve(tie)[6, 6] == 0 and solve(tie)[6, 8] == 3
    zero = fixture.copy()
    zero[0, 16] = zero[16, 16] = 0
    assert solve(zero)[4, 7] == 1  # zero-vote color 3 still bounds the span
    fingerprints, shapes, layouts, masks, backgrounds, slopes, rail_counts = (set() for _ in range(7))
    counts, lengths, gaps = set(), set(), set()
    alternative_hits = dict.fromkeys(_ALTERNATIVES, 0)
    local_changes = edges = zero_categories = inside = outside = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        if seed in (0, 21, 41):
            assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        rs = puzzle_metrics(puzzle)
        for r in rs:
            shapes.add(r['shape'])
            backgrounds.add(r['background'])
            slopes.add(r['slope'])
            rail_counts.add(r['rails'])
            counts.update(r['votes'])
            lengths.update(r['lengths'])
            gaps.update(r['gaps'])
            masks.update(r['masks'])
            q0, t0 = min(q for q, lo, hi in r['endpoints']), min(lo for q, lo, hi in r['endpoints'])
            layouts.add(tuple((q-q0, lo-t0, hi-t0) for q, lo, hi in r['endpoints']))
            local_changes += len(r['painted_colors']) >= 2
            edges += r['edge']
            zero_categories += 0 in r['votes']
            inside += r['inside_objects'] > 0
            outside += r['outside_objects'] > 0
            for name in alternative_hits:
                alternative_hits[name] += r['alternative_differences'][name] > 0
    assert len(fingerprints) == 42 and len(shapes) >= 35 and len(layouts) >= 150
    assert len(masks) >= 75 and backgrounds == set(range(10)) and slopes == {-1, 1}
    assert rail_counts == {2, 3, 4, 5} and counts >= {0, 1, 2, 3, 4}
    assert min(lengths) == 3 and max(lengths) >= 10 and gaps >= {4, 5, 6, 7, 8, 9}
    assert min(local_changes, edges, zero_categories, inside, outside) >= 10
    assert min(alternative_hits.values()) >= 42
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000+nt*10+nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, False)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Same connected object, same color, different area: no vote change.
    p = generate_puzzle(0, 2, 1)
    g = p['train'][0]['input']
    d = _parse(g)
    for o in d['objects']:
        for r, c in o['cells']:
            for dr, dc in _FOUR:
                rr, cc = r+dr, c+dc
                if not (0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]) or g[rr, cc] != d['background']:
                    continue
                if len(o['cells']) >= 8:
                    continue
                grown = g.copy()
                grown[rr, cc] = o['color']
                try:
                    new = _parse(grown)
                except ValueError:
                    continue
                if new['votes'] == d['votes'] and new['rails'] == d['rails']:
                    assert np.array_equal(solve(grown), solve(g))
                    break
            else:
                continue
            break
        else:
            continue
        break
    else:
        raise AssertionError('no valid growth metamorphism found')
    # Rotate/reflection and arbitrary bijective palette relabeling commute.
    expected = solve(g)
    for k in range(4):
        assert np.array_equal(solve(np.rot90(g, k)), np.rot90(expected, k))
    assert np.array_equal(solve(np.fliplr(g)), np.fliplr(expected))
    palette = np.array([8, 2, 6, 1, 9, 3, 0, 5, 7, 4], dtype=np.int8)
    assert np.array_equal(solve(palette[g]), palette[expected])
    # Corruption of a tied interior cell must be caught, beyond schema checks.
    bad = {s: [dict(input=x['input'].copy(), output=x['output'].copy()) for x in p[s]] for s in p}
    td = _parse(bad['train'][1]['input'])
    tied_prediction = _alternative(td, 'tie_first')
    wrong = np.argwhere(tied_prediction != bad['train'][1]['output'])[0]
    bad['train'][1]['output'][tuple(wrong)] = tied_prediction[tuple(wrong)]
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        return
    raise AssertionError('corrupt tie output accepted')
