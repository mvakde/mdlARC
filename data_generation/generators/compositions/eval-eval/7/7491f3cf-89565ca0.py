"""Splice two noisy floor plans, then count rooms incident to each control side.

Only Python's standard library and NumPy are used. No parent module, file,
seed, or demonstration is consulted by solve().
"""
from __future__ import annotations

import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(np.all((0 <= a) & (a <= 9)), "ARC colors required")
    return a.astype(np.int8, copy=False)


def _components(mask, diagonal=False):
    """Return components as flat cell indices; ordinary Python flood fill."""
    h, w = mask.shape
    remaining = mask.ravel().tolist()
    groups = []
    for start in range(h * w):
        if not remaining[start]:
            continue
        remaining[start] = False
        stack, cells = [start], []
        while stack:
            p = stack.pop()
            cells.append(p)
            r, c = divmod(p, w)
            neighbors = []
            if r:
                neighbors.append(p - w)
            if r + 1 < h:
                neighbors.append(p + w)
            if c:
                neighbors.append(p - 1)
            if c + 1 < w:
                neighbors.append(p + 1)
            if diagonal:
                if r and c:
                    neighbors.append(p - w - 1)
                if r and c + 1 < w:
                    neighbors.append(p - w + 1)
                if r + 1 < h and c:
                    neighbors.append(p + w - 1)
                if r + 1 < h and c + 1 < w:
                    neighbors.append(p + w + 1)
            for q in neighbors:
                if remaining[q]:
                    remaining[q] = False
                    stack.append(q)
        groups.append(cells)
    return groups


def _mask(shape, cells):
    a = np.zeros(shape, dtype=bool)
    a.ravel()[cells] = True
    return a


def _regions(cut):
    groups = _components(~cut)
    _require(len(groups) == 2, "cut must have two complementary regions")
    return [_mask(cut.shape, cells) for cells in groups]


def _near(mask):
    a = np.pad(mask, 1)
    h, w = mask.shape
    return np.logical_or.reduce([a[y:y+h, x:x+w] for y in range(3) for x in range(3)])


def _marker_sites(cut, regions):
    away = ~_near(cut)
    return [list(map(int, np.flatnonzero(away & r))) for r in regions]


def _check_cut(cut):
    h, w = cut.shape
    groups = _components(cut)
    _require(len(groups) == 1 and len(groups[0]) >= 3, "one connected cut required")
    degree = np.zeros(cut.shape, dtype=np.int8)
    degree[1:] += cut[:-1]
    degree[:-1] += cut[1:]
    degree[:, 1:] += cut[:, :-1]
    degree[:, :-1] += cut[:, 1:]
    ends = cut & (degree == 1)
    _require(int(ends.sum()) == 2 and np.all((degree[cut] == 1) | (degree[cut] == 2)),
             "cut must be a simple orthogonal path")
    border = np.zeros(cut.shape, bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    _require(np.array_equal(cut & border, ends), "only endpoints may touch panel edge")
    regions = _regions(cut)
    _require(all(_marker_sites(cut, regions)), "both sides need separated marker sites")
    return regions


def _repair(panel, wall, noise):
    observed = panel == wall
    bracket = np.zeros(panel.shape, bool)
    bracket[:, 1:-1] |= observed[:, :-2] & observed[:, 2:]
    bracket[1:-1, :] |= observed[:-2, :] & observed[2:, :]
    return observed | ((panel == noise) & bracket)


def _incidence(walls, marked, other):
    """Count bounded free components once for each region they intersect."""
    h, w = walls.shape
    m, o = marked.ravel().tolist(), other.ravel().tolist()
    count = [0, 0]
    memberships, sizes = [], []
    for room in _components(~walls):
        if any(p // w in (0, h-1) or p % w in (0, w-1) for p in room):
            continue
        bits = (any(m[p] for p in room), any(o[p] for p in room))
        count[0] += bits[0]
        count[1] += bits[1]
        memberships.append(bits)
        sizes.append(len(room))
    return tuple(count), memberships, sizes


def _hybrid(a, b, cut, marked, other):
    return (a & marked) | (b & other) | ((a | b) & cut)


def _encode(counts, colors, noise):
    _require(max(counts) >= 1, "no reportable rooms")
    result = np.full((2, max(counts)), noise, dtype=np.int8)
    for i in range(2):
        result[i, :counts[i]] = colors[i]
    return result


def _pack(panels, frame, vertical=False):
    h, w = panels[0].shape
    if vertical:
        grid = np.full((3*h+4, w+2), frame, dtype=np.int8)
        for i, p in enumerate(panels):
            grid[1+i*(h+1):1+i*(h+1)+h, 1:-1] = p
    else:
        grid = np.full((h+2, 3*w+4), frame, dtype=np.int8)
        for i, p in enumerate(panels):
            grid[1:-1, 1+i*(w+1):1+i*(w+1)+w] = p
    return grid


def _unpack(grid):
    frame = int(grid[0, 0])
    rows = np.flatnonzero(np.all(grid == frame, axis=1)).tolist()
    cols = np.flatnonzero(np.all(grid == frame, axis=0)).tolist()
    height, width = grid.shape
    if len(rows) == 2 and len(cols) == 4:
        _require((width-4) % 3 == 0, "horizontal panel dimensions")
        w = (width-4)//3
        _require(rows == [0, height-1] and cols == [i*(w+1) for i in range(4)], "frame spacing")
        panels = [grid[1:-1, 1+i*(w+1):1+i*(w+1)+w] for i in range(3)]
        vertical = False
    elif len(rows) == 4 and len(cols) == 2:
        _require((height-4) % 3 == 0, "vertical panel dimensions")
        h = (height-4)//3
        _require(cols == [0, width-1] and rows == [i*(h+1) for i in range(4)], "frame spacing")
        panels = [grid[1+i*(h+1):1+i*(h+1)+h, 1:-1] for i in range(3)]
        vertical = True
    else:
        raise AssertionError("three congruent framed panels required")
    _require(all(frame not in p for p in panels), "frame color inside panel")
    return panels, frame, vertical


def _source_geometry(walls):
    """Check a complete rectangular partition, not a product of line counts."""
    h, w = walls.shape
    _require(np.all(walls[[0, -1], :]) and np.all(walls[:, [0, -1]]), "open source border")
    reconstructed = np.zeros(walls.shape, bool)
    rooms = _components(~walls)
    _require(rooms, "source has no room")
    dimensions = []
    for room in rooms:
        ys, xs = zip(*(divmod(p, w) for p in room))
        t, b, l, r = min(ys), max(ys), min(xs), max(xs)
        _require(len(room) == (b-t+1)*(r-l+1), "source room must be rectangular")
        _require(0 < t <= b < h-1 and 0 < l <= r < w-1, "unbounded source room")
        reconstructed[t-1, l-1:r+2] = reconstructed[b+1, l-1:r+2] = True
        reconstructed[t-1:b+2, l-1] = reconstructed[t-1:b+2, r+1] = True
        dimensions.append((b-t+1, r-l+1))
    _require(np.array_equal(reconstructed, walls), "source has unexplained wall fragments")
    return dimensions


def _decode(value):
    grid = _grid(value)
    panels, frame, vertical = _unpack(grid)
    control, p, q = panels
    colors, freq = np.unique(control, return_counts=True)
    _require(len(colors) == 2 and freq[0] != freq[1], "control color roles")
    bg = int(colors[np.argmax(freq)])
    ink = int(colors[np.argmin(freq)])
    groups = sorted(_components(control != bg, True), key=len)
    _require(len(groups) == 2 and len(groups[0]) == 1 and len(groups[1]) > 1,
             "isolated marker and cut required")
    marker = groups[0][0]
    cut = _mask(control.shape, groups[1])
    regions = _check_cut(cut)
    _require(not _near(cut).ravel()[marker], "marker touches cut")
    side = 0 if regions[0].ravel()[marker] else 1
    pc, qc = set(map(int, np.unique(p)))-{bg}, set(map(int, np.unique(q)))-{bg}
    common = pc & qc
    _require(len(common) == 1 and len(pc-common) == 1 and len(qc-common) == 1, "source color roles")
    noise, ca, cb = next(iter(common)), next(iter(pc-common)), next(iter(qc-common))
    _require(len({frame, bg, ink, noise, ca, cb}) == 6, "distinct semantic roles required")
    a, b = _repair(p, ca, noise), _repair(q, cb, noise)
    sizes_a, sizes_b = _source_geometry(a), _source_geometry(b)
    # Covered walls must be isolated straight pixels with visible support.
    for observed, wall, color in ((p, a, ca), (q, b, cb)):
        damage = wall & (observed == noise)
        for y, x in np.argwhere(damage):
            degree = sum(0 <= yy < wall.shape[0] and 0 <= xx < wall.shape[1] and wall[yy, xx]
                         for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)))
            _require(degree == 2, "damage at corner or junction")
        _require(np.any((observed == noise) & ~wall), "noise must also occur away from walls")
    return dict(grid=grid, panels=panels, frame=frame, vertical=vertical, background=bg,
                control=ink, noise=noise, colors=(ca, cb), cut=cut, regions=regions,
                side=side, marker=marker, a=a, b=b, source_sizes=(sizes_a, sizes_b))


def _answer(d, side=None, repair=True):
    side = d['side'] if side is None else side
    marked, other = d['regions'][side], d['regions'][1-side]
    if repair:
        a, b = d['a'], d['b']
    else:
        a, b = d['panels'][1] == d['colors'][0], d['panels'][2] == d['colors'][1]
    walls = _hybrid(a, b, d['cut'], marked, other)
    counts, membership, sizes = _incidence(walls, marked, other)
    return counts, membership, sizes


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all roles, repaired walls and room memberships from this input."""
    d = _decode(input_grid)
    counts, _, _ = _answer(d)
    return _encode(counts, d['colors'], d['noise'])


def _has_partial(walls):
    degree = np.zeros(walls.shape, np.int8)
    degree[1:] += walls[:-1]
    degree[:-1] += walls[1:]
    degree[:, 1:] += walls[:, :-1]
    degree[:, :-1] += walls[:, 1:]
    return bool(np.any(walls[1:-1, 1:-1] & (degree[1:-1, 1:-1] == 3)))


_WITNESSES = {'descending', 'padding', 'shared', 'private', 'cut_not_wall',
              'seam_union', 'splice_before_count', 'not_union', 'not_first',
              'not_second', 'marker', 'partial', 'not_pixels', 'repair', 'noise_empty'}


def _observed_flags(d):
    """Distinguish ignoring damage from treating every noise pixel as wall."""
    flags = set()
    correct = _answer(d)[0]
    if _answer(d, repair=False)[0] != correct:
        flags.add('repair')
    m, o = d['regions'][d['side']], d['regions'][1-d['side']]
    a = d['a'] | (d['panels'][1] == d['noise'])
    b = d['b'] | (d['panels'][2] == d['noise'])
    all_noise_walls = _hybrid(a, b, d['cut'], m, o)
    if _incidence(all_noise_walls, m, o)[0] != correct:
        flags.add('noise_empty')
    return flags


def _geometry_flags(a, b, cut, regions):
    """Counterfactual comparisons over the two opposite-marker controls."""
    flags, results = set(), []
    if _has_partial(a) or _has_partial(b):
        flags.add('partial')
    first = _incidence(a, *regions)[0]
    second = _incidence(b, *regions)[0]
    union = _incidence(a | b, *regions)[0]
    for side in (0, 1):
        m, o = regions[side], regions[1-side]
        wall = _hybrid(a, b, cut, m, o)
        counts, members, _ = _incidence(wall, m, o)
        results.append(counts)
        if counts[0] > counts[1]:
            flags.add('descending')
        if min(counts) > 0 and counts[0] != counts[1]:
            flags.add('padding')
        if (True, True) in members:
            flags.add('shared')
        if (True, False) in members or (False, True) in members:
            flags.add('private')
        if counts != _incidence(wall | cut, m, o)[0]:
            flags.add('cut_not_wall')
        no_b_seam = (a & m) | (b & o) | (a & cut)
        if counts != _incidence(no_b_seam, m, o)[0]:
            flags.add('seam_union')
        order = [side, 1-side]
        fa, fb, fu = [tuple(v[i] for i in order) for v in (first, second, union)]
        if counts != (fa[0], fb[1]):
            flags.add('splice_before_count')
        if counts != fa:
            flags.add('not_first')
        if counts != fb:
            flags.add('not_second')
        if counts != fu:
            flags.add('not_union')
        pixels = (int(np.count_nonzero(~wall & m)), int(np.count_nonzero(~wall & o)))
        if counts != pixels:
            flags.add('not_pixels')
    if results[0] != results[1]:
        flags.add('marker')
    return flags, results


def _capacity(box):
    t, l, b, r = box
    return ((b-t)//2)*((r-l)//2)


def _offsets(rng, parts, span):
    gaps = [2]*parts
    for _ in range(span-2*parts):
        gaps[int(rng.integers(parts))] += 1
    return [0] + list(np.cumsum(gaps).astype(int))


def _network(rng, h, w, count, style):
    outer = (0, 0, h-1, w-1)
    _require(count <= _capacity(outer), "partition capacity")
    factors = [(y, count//y) for y in range(1, count+1)
               if count % y == 0 and 2*y < h and 2*(count//y) < w
               and (style == 'grid' or y == 1 or y == count)]
    if style != 'recursive' and factors:
        nr, nc = factors[int(rng.integers(len(factors)))]
        ys, xs = _offsets(rng, nr, h-1), _offsets(rng, nc, w-1)
        boxes = [(ys[y], xs[x], ys[y+1], xs[x+1]) for y in range(nr) for x in range(nc)]
    else:
        boxes = [outer]
        while len(boxes) < count:
            choices = []
            for i, (t, l, b, r) in enumerate(boxes):
                for axis, start, end in ((0, t, b), (1, l, r)):
                    for p in range(start+2, end-1):
                        halves = ((t, l, p, r), (p, l, b, r)) if axis == 0 else ((t, l, b, p), (t, p, b, r))
                        capacity = sum(_capacity(box) for j, box in enumerate(boxes) if i != j)
                        if capacity + sum(_capacity(box) for box in halves) >= count:
                            choices.append((i, halves))
            _require(choices, "no feasible recursive split")
            i, halves = choices[int(rng.integers(len(choices)))]
            boxes[i:i+1] = halves
    walls = np.zeros((h, w), bool)
    for t, l, b, r in boxes:
        walls[t, l:r+1] = walls[b, l:r+1] = True
        walls[t:b+1, l] = walls[t:b+1, r] = True
    _require(len(_components(~walls)) == count, "partition lost rooms")
    return walls


def _sample_cut(rng, h, w, kind):
    across = bool(rng.integers(2))
    ch, cw = (w, h) if across else (h, w)
    cut = np.zeros((ch, cw), bool)
    if kind == 'straight':
        cut[:, int(rng.integers(2, cw-2))] = True
    elif kind == 'elbow':
        y, x = int(rng.integers(2, ch-2)), int(rng.integers(2, cw-2))
        cut[:y+1, x] = True
        cut[y, x:] = True
    elif kind == 'stepped':
        candidates = list(range(2, ch-2, 2))
        steps = int(rng.integers(1, min(2, len(candidates))+1))
        ys = sorted(map(int, rng.choice(candidates, steps, replace=False)))
        x, last = int(rng.integers(2, cw-2)), 0
        for y in ys:
            cut[last:y+1, x] = True
            options = [c for c in range(2, cw-2) if c != x]
            nx = int(rng.choice(options))
            cut[y, min(x, nx):max(x, nx)+1] = True
            x, last = nx, y
        cut[last:, x] = True
    else:
        raise ValueError("unknown cut kind")
    if across:
        cut = cut.T.copy()
    if rng.integers(2):
        cut = cut[::-1, :].copy()
    if rng.integers(2):
        cut = cut[:, ::-1].copy()
    _check_cut(cut)
    return cut


def _cut_kind(cut):
    ys, xs = np.where(cut)
    if len(set(ys)) == 1 or len(set(xs)) == 1:
        return 'straight'
    # A single elbow's endpoints are on adjacent edges. Stepped cuts connect
    # opposite edges; simple paths never make intermediate boundary contact.
    h, w = cut.shape
    sides = (bool(np.any(cut[0])), bool(np.any(cut[-1])),
             bool(np.any(cut[:, 0])), bool(np.any(cut[:, -1])))
    return 'stepped' if (sides[0] and sides[1]) or (sides[2] and sides[3]) else 'elbow'


def _corrupt(rng, walls, bg, color, noise, witness=False):
    h, w = walls.shape
    degree = np.zeros(walls.shape, np.int8)
    degree[1:] += walls[:-1]
    degree[:-1] += walls[1:]
    degree[:, 1:] += walls[:, :-1]
    degree[:, :-1] += walls[:, 1:]
    straight = np.zeros(walls.shape, bool)
    straight[:, 1:-1] |= walls[:, :-2] & walls[:, 2:]
    straight[1:-1] |= walls[:-2] & walls[2:]
    sites = list(map(int, rng.permutation(np.flatnonzero(walls & straight & (degree == 2)))))
    wanted = int(rng.integers(2, 6)) if witness else int(rng.integers(0, 5))
    damage = np.zeros(walls.shape, bool)
    if wanted:
        for p in sites:
            if not _near(damage).ravel()[p]:
                damage.ravel()[p] = True
                if int(damage.sum()) >= wanted:
                    break
    panel = np.full(walls.shape, bg, np.int8)
    panel[walls] = color
    panel[damage] = noise
    observed = panel == color
    bracket = np.zeros(walls.shape, bool)
    bracket[:, 1:-1] |= observed[:, :-2] & observed[:, 2:]
    bracket[1:-1] |= observed[:-2] & observed[2:]
    offwall = np.flatnonzero(~walls & ~bracket)
    _require(len(offwall), "need off-wall noise site")
    amount = min(len(offwall), int(rng.integers(1, 6)))
    panel.ravel()[rng.choice(offwall, amount, replace=False)] = noise
    _require(np.array_equal(_repair(panel, color, noise), walls), "unrecoverable corruption")
    return panel


def _make_pair(rng, a, b, cut, regions, side, palette, vertical, witness=False, panels=None):
    frame, bg, ink, ca, cb, noise = palette
    ctrl = np.full(cut.shape, bg, np.int8)
    ctrl[cut] = ink
    sites = _marker_sites(cut, regions)[side]
    ctrl.ravel()[int(rng.choice(sites))] = ink
    if panels is None:
        pa = _corrupt(rng, a, bg, ca, noise, witness)
        pb = _corrupt(rng, b, bg, cb, noise, witness)
    else:
        pa, pb = panels
    source_panels = [ctrl, pa, pb]
    if vertical:
        source_panels = [p.T.copy() for p in source_panels]
    inp = _pack(source_panels, frame, vertical)
    counts = _incidence(_hybrid(a, b, cut, regions[side], regions[1-side]),
                        regions[side], regions[1-side])[0]
    _require(min(counts) > 0, "both census rows must be nonempty")
    return {'input': inp, 'output': _encode(counts, (ca, cb), noise)}


def _palette(rng, base, fixed):
    return base if fixed else tuple(map(int, rng.choice(10, 6, replace=False)))


def _sample_geometry(rng, kind, dense=False, witness=False):
    h = int(rng.integers(12 if dense else 8, 21 if dense else 17))
    w = int(rng.integers(7, 9))
    cut = _sample_cut(rng, h, w, kind)
    regions = _regions(cut)
    maximum = min(12 if dense else 7, _capacity((0, 0, h-1, w-1)))
    minimum = 8 if dense else 3 if witness else 1
    counts = [int(rng.integers(minimum, maximum+1)) for _ in range(2)]
    styles = [str(rng.choice(['recursive', 'recursive', 'recursive', 'strip', 'grid'])) for _ in range(2)]
    a, b = [_network(rng, h, w, c, s) for c, s in zip(counts, styles)]
    # A partition made entirely of one-cell-wide rooms cannot contain visible
    # irrelevant noise: every free cell would be interpreted as a covered wall.
    # Reject it before corruption, including in the witness search.
    for walls in (a, b):
        bracket = np.zeros(walls.shape, bool)
        bracket[:, 1:-1] |= walls[:, :-2] & walls[:, 2:]
        bracket[1:-1] |= walls[:-2] & walls[2:]
        _require(np.any(~walls & ~bracket), 'source needs a distinguishable empty-noise site')
    return a, b, cut, regions


def _key(a):
    a = np.asarray(a, np.int8)
    return a.shape, a.tobytes()


def _resolve_counts(num_train, num_test):
    for value, name, low, high in ((num_train, 'num_train', 2, 10), (num_test, 'num_test', 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _resolve_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    base = tuple(map(int, rng.choice(10, 6, replace=False)))
    fixed = bool(rng.random() < .35)
    regime = str(rng.choice(['step', 'elbow', 'density']))
    initial_kind = 'straight' if regime == 'elbow' else str(rng.choice(['straight', 'elbow']))
    # Whole random partitions are retained only when their counterfactuals teach
    # the rule. The two controls isolate dot movement without altering walls.
    for _ in range(3000):
        try:
            a, b, cut, regions = _sample_geometry(rng, initial_kind, witness=True)
        except AssertionError:
            continue
        flags, counts = _geometry_flags(a, b, cut, regions)
        if not _WITNESSES-{'repair', 'noise_empty'} <= flags or not all(min(v) > 0 for v in counts):
            continue
        palette, vertical = _palette(rng, base, fixed), bool(rng.integers(2))
        first = _make_pair(rng, a, b, cut, regions, 0, palette, vertical, True)
        d = _decode(first['input'])
        # Preserve the exact observed noisy sources in the twin.
        source_panels = d['panels'][1:]
        if vertical:
            source_panels = [p.T.copy() for p in source_panels]
        second = _make_pair(rng, a, b, cut, regions, 1, palette, vertical, True, source_panels)
        observation_flags = _observed_flags(d) | _observed_flags(_decode(second['input']))
        if {'repair', 'noise_empty'} <= observation_flags:
            train = [first, second]
            break
    else:
        raise RuntimeError('bounded witness search exhausted')
    puzzle = {'train': train, 'test': []}
    seen = {_key(p['input']) for p in train}
    for split, number in (('train', nt-2), ('test', ne)):
        for _ in range(number):
            for attempt in range(300):
                if split == 'train':
                    if regime == 'elbow':
                        kind = 'straight'
                    elif regime == 'step':
                        kind = str(rng.choice(['straight', 'elbow']))
                    else:
                        kind = str(rng.choice(['straight', 'elbow', 'stepped']))
                else:
                    kind = 'stepped' if regime == 'step' else 'elbow' if regime == 'elbow' else initial_kind
                try:
                    a, b, cut, regions = _sample_geometry(rng, kind, dense=split == 'test' and regime == 'density')
                    pair = _make_pair(rng, a, b, cut, regions, int(rng.integers(2)),
                                      _palette(rng, base, fixed), bool(rng.integers(2)))
                except AssertionError:
                    continue
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded example search exhausted')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split lists required')
    _resolve_counts(len(puzzle['train']), len(puzzle['test']))
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            inp, output = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            d = _decode(inp)
            counts, _, _ = _answer(d)
            _require(min(counts) > 0, 'empty incidence row')
            _require(np.array_equal(output, _encode(counts, d['colors'], d['noise'])), 'incorrect census')
            infos[split].append(d)
    # The witness pair is found from returned grids, not an index or seed.
    taught = set()
    twins = False
    for i, d in enumerate(infos['train']):
        flags, _ = _geometry_flags(d['a'], d['b'], d['cut'], d['regions'])
        # Only flags of a genuine opposite-marker twin establish both controls.
        for e in infos['train'][i+1:]:
            same = (d['frame'] == e['frame'] and d['vertical'] == e['vertical']
                    and d['background'] == e['background'] and d['control'] == e['control']
                    and d['colors'] == e['colors'] and d['noise'] == e['noise']
                    and d['side'] != e['side'] and np.array_equal(d['cut'], e['cut'])
                    and all(np.array_equal(p, q) for p, q in zip(d['panels'][1:], e['panels'][1:])))
            if same:
                twins = True
                taught |= flags
        taught |= _observed_flags(d)
    _require(twins and _WITNESSES <= taught, 'training lacks semantic witnesses')


def _puzzle_key(puzzle):
    return tuple((s, tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s])) for s in ('train', 'test'))


def self_test():
    """Bounded semantic, metamorphic, API and variation checks over 42 seeds."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, outputs, partitions, controls, palettes = set(), set(), set(), set(), set()
    kinds, orientations, source_counts, incidence_counts = set(), set(), set(), set()
    damaged = intact = partial = shared = 0
    for puzzle in puzzles:
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                d = _decode(pair['input'])
                counts, members, _ = _answer(d)
                shapes.add(d['grid'].shape)
                outputs.add(pair['output'].shape)
                controls.add(_key(d['cut']))
                partitions.add((_key(d['a']), _key(d['b'])))
                palettes.add((d['frame'], d['background'], d['control'], *d['colors'], d['noise']))
                kinds.add(_cut_kind(d['cut']))
                orientations.add(d['vertical'])
                source_counts.update(map(len, d['source_sizes']))
                incidence_counts.update(counts)
                shared += (True, True) in members
                partial += _has_partial(d['a']) or _has_partial(d['b'])
                for i, wall in enumerate((d['a'], d['b'])):
                    damage = np.any(wall & (d['panels'][i+1] == d['noise']))
                    damaged += bool(damage)
                    intact += not damage
        # Recoloring and aligned within-panel reflection preserve the semantics.
        pair = puzzle['train'][0]
        d = _decode(pair['input'])
        colors = np.array([9, 7, 5, 3, 1, 8, 6, 4, 2, 0], np.int8)
        _require(np.array_equal(solve(colors[pair['input']]), colors[pair['output']]), 'recoloring')
        for axis in (0, 1):
            reflected = _pack([np.flip(p, axis).copy() for p in d['panels']], d['frame'], d['vertical'])
            _require(np.array_equal(solve(reflected), pair['output']), 'within-panel reflection')
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, 'seed diversity')
    _require(_puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0)), 'determinism')
    _require(len(shapes) >= 20 and len(controls) >= 50 and len(partitions) >= 100, 'geometry diversity')
    _require(len(palettes) >= 30 and len(outputs) >= 4, 'role/count diversity')
    _require(kinds == {'straight', 'elbow', 'stepped'} and orientations == {False, True}, 'cut/layout coverage')
    _require(partial >= 60 and shared >= 50 and damaged >= 100 and intact >= 5, 'structural coverage')
    _require(min(source_counts) == 1 and max(source_counts) >= 10, 'source count variation')
    _require(len(incidence_counts) >= 6, 'incidence count variation')
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(123 + nt*7 + ne, nt, ne)
            _require(len(p['train']) == nt and len(p['test']) == ne, 'explicit counts')
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    bad['train'][0]['output'][0, 0] = (int(bad['train'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('wrong answer accepted')
    return dict(seeds=42, pairs=210, input_shapes=len(shapes), output_shapes=len(outputs),
                distinct_source_pairs=len(partitions), distinct_controls=len(controls),
                role_palettes=len(palettes), source_counts=sorted(source_counts),
                incidence_counts=sorted(incidence_counts), partial_pairs=partial,
                shared_room_pairs=shared, damaged_sources=damaged, intact_sources=intact)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
