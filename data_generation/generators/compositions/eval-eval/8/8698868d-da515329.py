"""Wind-up pinwheel shuttles: topology controls finite winding and deposition.

No parent modules, stored examples, external dependencies, or puzzle state are
used by solve. See task_spec.md for the intended family and sampling scope.
"""
from itertools import combinations, product
import numpy as np


_N4 = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError('expected an integer ARC grid with sides 1..30')
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    steps = _N4 + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else _N4
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        pending, cells = [start], []
        while pending:
            p = pending.pop()
            cells.append(p)
            for dy, dx in steps:
                n = p[0] + dy, p[1] + dx
                if n in unseen:
                    unseen.remove(n)
                    pending.append(n)
        result.append(cells)
    return result


def _rotate(y, x, turns):
    for _ in range(turns % 4):
        y, x = x, -y
    return y, x


def _endpoint(radius, budget, direction):
    """Closed form derived from B's four-stage cycle, also used for packing."""
    q, s = divmod(budget, 4)
    y, x = ((-radius-8*q, -8*q), (-2-8*q, radius+2+8*q),
            (radius+4+8*q, 4+8*q), (6+8*q, -radius-6-8*q))[s]
    return _rotate(y, x, direction)


def _walk(radius, budget, direction):
    """Finite geometric version of B v2's stepped winding, in relative cells."""
    cells = {(-d, 0) for d in range(1, radius + 1)}
    y, x, dy, dx = -radius, 0, 0, 1
    for stage in range(budget):
        for vy, vx, length in ((dy, dx, 2+4*stage),
                               (dx, -dy, radius-2), (dy, dx, radius)):
            for _ in range(length):
                y, x = y + vy, x + vx
                cells.add((y, x))
        dy, dx = dx, -dy
    return {_rotate(a, b, direction) for a, b in cells}, _rotate(y, x, direction)


def _analytic_mask(shape, center, radius, budget, direction):
    """Cell membership in finite line intervals; does not walk a path."""
    yy, xx = np.indices(shape)
    yy, xx = yy-center[0], xx-center[1]
    yy, xx = _rotate(yy, xx, -direction)
    mask = (xx == 0) & (yy <= -1) & (yy >= -radius)
    for j in range(budget):
        y, x = _endpoint(radius, j, 0)
        dy, dx = ((0, 1), (1, 0), (0, -1), (-1, 0))[j % 4]
        along = (yy-y)*dy + (xx-x)*dx
        across = (yy-y)*dx - (xx-x)*dy
        run = 2+4*j
        mask |= (across == 0) & (along >= 0) & (along <= run)
        mask |= (along == run) & (across >= 0) & (across <= radius-2)
        mask |= (across == radius-2) & (along >= run) & (along <= run+radius)
    return mask


def _boxes_disjoint(points, side):
    # Even corner overlap is forbidden; edge adjacency is allowed.
    return all(max(abs(a-c), abs(b-d)) >= side
               for (a, b), (c, d) in combinations(points, 2))


def _cross(shape, center, length, background, thread):
    a = np.full(shape, background, dtype=np.int8)
    y, x = center
    for d in range(1, length + 1):
        a[y-d, x] = a[y+d, x] = a[y, x-d] = a[y, x+d] = thread
    return a


def _decode(input_grid):
    a = _grid(input_grid)
    candidates = []
    for color in map(int, np.unique(a)):
        mask = a == color
        count = int(mask.sum())
        if count % 4 or not 1 <= count//4 <= 4:
            continue
        length = count//4
        rows = np.flatnonzero(mask.sum(axis=1) == 2*length)
        cols = np.flatnonzero(mask.sum(axis=0) == 2*length)
        if len(rows) != 1 or len(cols) != 1:
            continue
        center = int(rows[0]), int(cols[0])
        y, x = center
        if min(y, x, a.shape[0]-1-y, a.shape[1]-1-x) < length:
            continue
        if a[center] == color:
            continue
        seed = _cross(a.shape, center, length, int(a[center]), color)
        if np.array_equal(seed == color, mask):
            candidates.append((color, length, center))
    if len(candidates) != 1:
        raise ValueError('expected one distinct-color hollow equal-spoke cross')
    thread, length, center = candidates[0]
    background = int(a[center])
    bodies = _components((a != background) & (a != thread))
    if not 2 <= len(bodies) <= 4:
        raise ValueError('expected two to four connected motif bodies')
    shuttles, sides, directions = [], set(), set()
    expected_input = _cross(a.shape, center, length, background, thread)
    for cells in bodies:
        rs, cs = zip(*cells)
        r0, r1, c0, c1 = min(rs), max(rs)+1, min(cs), max(cs)+1
        motif = a[r0:r1, c0:c1].copy()
        s = r1-r0
        if s not in (5, 7) or c1-c0 != s:
            raise ValueError('motifs must be equal-sized 5x5 or 7x7 squares')
        color = int(motif[0, 0])
        perimeter = np.concatenate((motif[0], motif[-1], motif[:, 0], motif[:, -1]))
        if (color in (background, thread) or not np.all(perimeter == color)
                or not set(map(int, np.unique(motif))) <= {color, background}):
            raise ValueError('each motif must have a complete single-color perimeter')
        q = s//2
        radius = length+q+1
        my, mx = r0+q, c0+q
        offsets = [_rotate(-radius, 0, d) for d in range(4)]
        offset = my-center[0], mx-center[1]
        if offset not in offsets:
            raise ValueError('motifs must dock immediately beyond cross tips')
        direction = offsets.index(offset)
        if direction in directions:
            raise ValueError('at most one shuttle may load an arm')
        directions.add(direction)
        sides.add(s)
        holes = motif == background
        budget = len(_components(holes))
        if budget > 4:
            raise ValueError('winding budgets are zero through four cavities')
        expected_input[r0:r1, c0:c1] = motif
        shuttles.append({'direction': direction, 'budget': budget, 'motif': motif,
                         'color': color, 'holes': holes, 'source': (r0, c0),
                         'area': int(holes.sum())})
    if len(sides) != 1 or not np.array_equal(a, expected_input):
        raise ValueError('input must contain only the cross and its common-size shuttles')
    s = next(iter(sides))
    radius = length+s//2+1
    starts = [_rotate(-radius, 0, m['direction']) for m in shuttles]
    ends = [_endpoint(radius, m['budget'], m['direction']) for m in shuttles]
    if not _boxes_disjoint(starts, s) or not _boxes_disjoint(ends, s+2):
        raise ValueError('source motifs and full-plane destination footprints must be disjoint')
    shuttles.sort(key=lambda m: m['direction'])
    return {'grid': a, 'background': background, 'thread': thread, 'length': length,
            'center': center, 'size': s, 'radius': radius, 'shuttles': shuttles}


def _paste_clipped(out, patch, top, left):
    y0, x0 = max(0, top), max(0, left)
    y1, x1 = min(out.shape[0], top+patch.shape[0]), min(out.shape[1], left+patch.shape[1])
    if y0 < y1 and x0 < x1:
        out[y0:y1, x0:x1] = patch[y0-top:y1-top, x0-left:x1-left]


def _render(info, *, analytic=False, alternative=None):
    a, center = info['grid'], info['center']
    bg, thread, radius = info['background'], info['thread'], info['radius']
    out = _cross(a.shape, center, info['length'], bg, thread)
    destinations = []
    for m in info['shuttles']:
        k, d = m['budget'], m['direction']
        if alternative == 'ignore_topology':
            k = 0
        if alternative == 'area':
            k = m['area']
        if alternative == 'eight_connectivity':
            k = len(_components(m['holes'], diagonal=True))
        if alternative == 'straight':
            distance = radius + sum(2+4*j + radius-2 + radius for j in range(k))
            cells = {_rotate(-v, 0, d) for v in range(1, distance+1)}
            end = _rotate(-distance, 0, d)
        elif alternative == 'ordinary_spiral':
            cells = {(-v, 0) for v in range(1, radius+1)}
            y, x, dy, dx = -radius, 0, 0, 1
            for j in range(k):
                for _ in range(radius+2+4*j):
                    y, x = y+dy, x+dx
                    cells.add((y, x))
                dy, dx = dx, -dy
            cells = {_rotate(y, x, d) for y, x in cells}
            end = _rotate(y, x, d)
        elif analytic:
            out[_analytic_mask(a.shape, center, radius, k, d)] = thread
            end = _endpoint(radius, k, d)
            cells = ()
        else:
            cells, end = _walk(radius, k, d)
        for y, x in cells:
            y, x = y+center[0], x+center[1]
            if 0 <= y < a.shape[0] and 0 <= x < a.shape[1]:
                out[y, x] = thread
        turns = 0 if alternative in ('straight', 'no_rotation') else k
        motif = np.rot90(m['motif'], -turns)
        replacement = bg if alternative == 'no_substitution' else thread
        patch = np.full((info['size']+2, info['size']+2), thread, dtype=np.int8)
        patch[1:-1, 1:-1] = np.where(motif == bg, replacement, motif)
        half = patch.shape[0]//2
        destinations.append((patch, center[0]+end[0]-half, center[1]+end[1]-half))
    # Deposits lie above threads; destination boxes never overlap in valid inputs.
    for patch, top, left in destinations:
        _paste_clipped(out, patch, top, left)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the visible scene, walk finite windings, and deposit rotated cargo."""
    return _render(_decode(input_grid))


def _transforms(mask):
    return [np.rot90(mask, k) for k in range(4)] + [np.rot90(mask.T, k) for k in range(4)]


def _cavity_mask(rng, size, count, witness=None):
    holes = np.zeros((size, size), dtype=bool)
    if count == 0:
        return holes
    if witness == 'ring':
        holes[1:-1, 1:-1] = True
        return holes
    if witness == 'diagonal':
        holes[1, 1] = holes[2, 2] = holes[2, 3] = True
        return _transforms(holes)[int(rng.integers(8))].copy()
    n = size-2
    mode = int(rng.integers(4))
    if count == 1 and mode == 0:
        holes[1:-1, 1:-1] = True
        return holes
    if count == 4 and mode == 0:
        q = int(rng.integers(1, (n-1)//2+1))
        for r in (1, size-1-q):
            for c in (1, size-1-q):
                holes[r:r+q, c:c+q] = True
        return holes
    # Arbitrary isolated starts. Growth is allowed to be long and irregular,
    # but cannot merge cavities or disconnect the colored motif body.
    for _ in range(16):
        trial = np.zeros_like(holes)
        for index in rng.permutation(n*n):
            r, c = 1+int(index)//n, 1+int(index)%n
            if any(trial[r+dy, c+dx] for dy, dx in _N4):
                continue
            trial[r, c] = True
            if len(_components(~trial)) != 1:
                trial[r, c] = False
            if int(trial.sum()) == count:
                holes = trial
                break
        if int(holes.sum()) == count:
            break
    else:
        # Four corners are a constructive lower bound for both supported sizes.
        for r, c in list(product((1, size-2), repeat=2))[:count]:
            holes[r, c] = True
    target = count if mode == 1 else int(rng.integers(count, n*n+1))
    for index in rng.permutation(n*n):
        if int(holes.sum()) >= target:
            break
        r, c = 1+int(index)//n, 1+int(index)%n
        if holes[r, c] or not any(holes[r+dy, c+dx] for dy, dx in _N4):
            continue
        trial = holes.copy()
        trial[r, c] = True
        if len(_components(trial)) == count and len(_components(~trial)) == 1:
            holes = trial
    return _transforms(holes)[int(rng.integers(8))].copy()


def _relative_geometry(size, length, directions, budgets):
    radius = length+size//2+1
    starts = [_rotate(-radius, 0, d) for d in directions]
    ends = [_endpoint(radius, k, d) for d, k in zip(directions, budgets)]
    if not _boxes_disjoint(starts, size) or not _boxes_disjoint(ends, size+2):
        return None
    input_cells = {(dy*d, dx*d) for dy, dx in _N4 for d in range(1, length+1)}
    all_cells = set(input_cells)
    q, p = size//2, size//2+1
    for (y, x), end, direction, budget in zip(starts, ends, directions, budgets):
        input_cells.update((y+a, x+b) for a in range(-q, q+1) for b in range(-q, q+1))
        path, walked_end = _walk(radius, budget, direction)
        assert end == walked_end
        all_cells.update(path)
        all_cells.update((end[0]+a, end[1]+b)
                         for a in range(-p, p+1) for b in range(-p, p+1))
    all_cells.update(input_cells)
    def bounds(cells):
        rows, cols = zip(*cells)
        return min(rows), max(rows), min(cols), max(cols)
    return bounds(input_cells), bounds(all_cells)


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _sample_plan(rng, *, required=(), count=None, radius=None, exclude_radius=None,
                 full=True, size=None, length=None, budgets_pool=range(5)):
    """Bounded search over a finite feasible geometry set, with an exhaustive fallback."""
    options = []
    for s in ((5, 7) if size is None else (size,)):
        for l in (range(1, 5) if length is None else (length,)):
            r = l+s//2+1
            if (radius is not None and r != radius) or r == exclude_radius:
                continue
            for n in ((2, 3, 4) if count is None else (count,)):
                if n < len(required):
                    continue
                for dirs in combinations(range(4), n):
                    options.append((s, l, dirs))
    if not options:
        raise ValueError('no feasible geometry choices')
    for _ in range(100):
        s, l, dirs = _choose(rng, options)
        ks = list(required) + [int(_choose(rng, list(budgets_pool))) for _ in range(len(dirs)-len(required))]
        rng.shuffle(ks)
        if not any(ks):
            continue
        geom = _relative_geometry(s, l, dirs, ks)
        if geom is None:
            continue
        ib, ob = geom
        b = ob if full else ib
        if max(b[1]-b[0]+1, b[3]-b[2]+1) <= 30:
            # Even cropped scenes use a full configuration that fits 30 cells,
            # leaving bounded, readable clipping of final deposits.
            if max(ob[1]-ob[0]+1, ob[3]-ob[2]+1) <= 30:
                return s, l, dirs, tuple(ks), ib, ob
    # Exhaustive finite fallback avoids reliance on lucky random retries.
    for s, l, dirs in options:
        for ks in product(tuple(budgets_pool), repeat=len(dirs)):
            if not any(ks):
                continue
            need = list(ks)
            for k in required:
                if k not in need:
                    break
                need.remove(k)
            else:
                geom = _relative_geometry(s, l, dirs, ks)
                if geom:
                    ib, ob = geom
                    if max(ob[1]-ob[0]+1, ob[3]-ob[2]+1) <= 30:
                        return s, l, dirs, ks, ib, ob
    raise ValueError('no compatible disjoint destinations in the requested plan')


def _place_window(rng, ib, ob, clipped):
    low_y, high_y, low_x, high_x = ob
    if clipped:
        # Trim where original motifs remain fully visible. At most three
        # cells per side keeps each endpoint motif partly visible.
        choices = []
        if ib[0] > low_y:
            choices.append(0)
        if ib[1] < high_y:
            choices.append(1)
        if ib[2] > low_x:
            choices.append(2)
        if ib[3] < high_x:
            choices.append(3)
        if not choices:
            raise ValueError('configuration cannot exhibit clipping')
        side = int(_choose(rng, choices))
        trim = int(rng.integers(1, min(3, (ib[0]-low_y, high_y-ib[1],
                                         ib[2]-low_x, high_x-ib[3])[side])+1))
        if side == 0:
            low_y += trim
        elif side == 1:
            high_y -= trim
        elif side == 2:
            low_x += trim
        else:
            high_x -= trim
    # Unrestricted independent blank margins and rectangular dimensions.
    span_y, span_x = high_y-low_y+1, high_x-low_x+1
    extra_y, extra_x = int(rng.integers(31-span_y)), int(rng.integers(31-span_x))
    before_y, before_x = int(rng.integers(extra_y+1)), int(rng.integers(extra_x+1))
    # Padding a trimmed edge would hide the train/test shift, so keep it tight.
    if clipped:
        if side == 0:
            before_y = 0
        elif side == 1:
            before_y = extra_y
        elif side == 2:
            before_x = 0
        else:
            before_x = extra_x
    return (span_y+extra_y, span_x+extra_x), (-low_y+before_y, -low_x+before_x)


def _make_pair(rng, plan, palette, *, witness=False, clipped=False):
    size, length, dirs, ks, ib, ob = plan
    shape, center = _place_window(rng, ib, ob, clipped)
    bg, thread = palette
    colors = [c for c in range(10) if c not in palette]
    a = _cross(shape, center, length, bg, thread)
    radius, q = length+size//2+1, size//2
    for d, k in zip(dirs, ks):
        mode = 'ring' if witness and k == 1 else 'diagonal' if witness and k == 2 else None
        holes = _cavity_mask(rng, size, k, mode)
        color = int(_choose(rng, colors))
        motif = np.where(holes, bg, color).astype(np.int8)
        y, x = _rotate(-radius, 0, d)
        y, x = y+center[0]-q, x+center[1]-q
        a[y:y+size, x:x+size] = motif
    info = _decode(a)
    return {'input': a, 'output': _render(info)}


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _check_counts(num_train, num_test):
    for name, v, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not low <= v <= high:
            raise ValueError(name+' has an illegal count')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    regime = str(_choose(rng, ('load', 'clip', 'scale')))
    base = tuple(map(int, rng.choice(10, 2, replace=False)))
    puzzle, used = {'train': [], 'test': []}, set()
    # Two demonstrations already teach zero, three distinct nonzero budgets,
    # area-vs-topology, four-vs-eight connectivity, rotation, and radius changes.
    first_radius = int(_choose(rng, (5, 6)))
    second_radius = first_radius+1
    for split, n in (('train', int(num_train)), ('test', int(num_test))):
        for i in range(n):
            if split == 'train' and i == 0:
                kwargs = {'required': (1, 2), 'count': 2, 'radius': first_radius}
            elif split == 'train' and i == 1:
                kwargs = {'required': (0, 3), 'count': 2, 'radius': second_radius}
            elif split == 'test' and regime == 'load':
                kwargs = {'count': 4, 'required': (2,)}
            elif split == 'test' and regime == 'scale':
                kwargs = {'radius': second_radius+1, 'count': int(_choose(rng, (2, 3))), 'required': (2,)}
            elif split == 'test':
                kwargs = {'count': int(_choose(rng, (2, 3))), 'required': (2,)}
            else:
                kwargs = {'count': int(_choose(rng, (2, 3))),
                          'radius': int(_choose(rng, range(4, second_radius+1))) if regime == 'scale' else None,
                          'required': (1,)}
                if kwargs['radius'] == 4:
                    kwargs['count'] = 2
            clipped = split == 'test' and regime == 'clip'
            # Changing the two roles between the first demonstrations is visible
            # in the input. No literal color is an output selector.
            palette = base if split == 'train' and i == 0 else base[::-1] if split == 'train' and i == 1 else tuple(map(int, rng.choice(10, 2, replace=False)))
            plan = _sample_plan(rng, **kwargs)
            for attempt in range(16):
                pair = _make_pair(rng, plan, palette, witness=split == 'train' and i == 0, clipped=clipped)
                key = _key(pair['input'])
                if key not in used:
                    break
            else:
                # At most 13 prior inputs; 90 visible ordered role palettes
                # provide a finite guarantee without changing geometry.
                original = pair['input']
                source_roles = list(palette) + [c for c in range(10) if c not in palette]
                for new_roles in ((a, b) for a in range(10) for b in range(10) if a != b):
                    target = list(new_roles) + [c for c in range(10) if c not in new_roles]
                    lut = np.empty(10, dtype=np.int8)
                    lut[source_roles] = target
                    a = lut[original]
                    if _key(a) not in used:
                        pair = {'input': a, 'output': solve(a)}
                        key = _key(a)
                        break
                else:
                    raise AssertionError('finite color-permutation bound failed')
            used.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _fully_visible(info):
    h, w = info['grid'].shape
    p = info['size']//2+1
    cy, cx = info['center']
    for m in info['shuttles']:
        y, x = _endpoint(info['radius'], m['budget'], m['direction'])
        if min(cy+y-p, cx+x-p, h-1-cy-y-p, w-1-cx-x-p) < 0:
            return False
        cells, _ = _walk(info['radius'], m['budget'], m['direction'])
        if any(not (0 <= cy+a < h and 0 <= cx+b < w) for a, b in cells):
            return False
    return True


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, info = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == b.shape
            assert _key(a) not in seen, 'duplicate input'
            seen.add(_key(a))
            decoded = _decode(a)
            expected = _render(decoded, analytic=True)
            assert np.array_equal(expected, b), 'incorrect finite winding or motif deposition'
            assert set(map(int, np.unique(b))) <= set(map(int, np.unique(a)))
            assert np.all(b[a == decoded['thread']] == decoded['thread']), 'seed must be retained'
            assert b[decoded['center']] == decoded['background'], 'hollow center must remain open'
            assert not np.array_equal(a, b)
            if split == 'train':
                assert _fully_visible(decoded), 'training must show complete paths and deposits'
            info[split].append(decoded)
    train = info['train']
    budgets = {m['budget'] for a in train for m in a['shuttles']}
    assert {0, 1, 2, 3} <= budgets, 'training must teach the finite turn budget'
    assert len({a['radius'] for a in train}) >= 2, 'radius-dependent inward jog is not taught'
    assert any(any(m['area'] > 1 for m in a['shuttles']) for a in train)
    assert any(any(m['budget'] > len(_components(m['holes'], True)) for m in a['shuttles']) for a in train), 'four/eight connectivity witness missing'
    assert any(any(u['budget'] < v['budget'] and u['area'] > v['area']
                   for u, v in combinations(a['shuttles'], 2))
               or any(v['budget'] < u['budget'] and v['area'] > u['area']
                      for u, v in combinations(a['shuttles'], 2)) for a in train), 'area inversion missing'
    for shortcut in ('ignore_topology', 'area', 'eight_connectivity', 'straight',
                     'ordinary_spiral', 'no_rotation', 'no_substitution'):
        assert any(not np.array_equal(_render(a, alternative=shortcut), pair['output'])
                   for a, pair in zip(train, puzzle['train'])), 'training permits shortcut: '+shortcut
    test = info['test']
    # The sampler guarantees at least one of three visible structural shifts.
    assert (any(not _fully_visible(a) for a in test)
            or max(len(a['shuttles']) for a in test) > max(len(a['shuttles']) for a in train)
            or min(a['radius'] for a in test) > max(a['radius'] for a in train)), 'train/test structural shift missing'
    return None


def _equal(p, q):
    return all(len(p[s]) == len(q[s]) and all(np.array_equal(a[k], b[k])
               for a, b in zip(p[s], q[s]) for k in ('input', 'output')) for s in ('train', 'test'))


def self_test():
    """Bounded checks on 42 seeds, plus interface and targeted geometry tests."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    measures = {'input_shapes': set(), 'radii': set(), 'sizes': set(), 'loads': set(),
                'budgets': set(), 'backgrounds': set(), 'threads': set(),
                'layouts': set(), 'cavity_shapes': set(), 'cavity_areas': set()}
    signatures, clipped_tests = set(), 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = _decode(pair['input'])
                assert np.array_equal(solve(pair['input'].copy()), pair['output'])
                assert np.array_equal(solve(np.rot90(pair['input'])), np.rot90(pair['output']))
                # A source-tight crop can heavily clip trails or remove a
                # destination altogether; solving must commute with cropping.
                rs, cs = np.nonzero(pair['input'] != a['background'])
                window = (slice(int(rs.min()), int(rs.max())+1),
                          slice(int(cs.min()), int(cs.max())+1))
                assert np.array_equal(solve(pair['input'][window]), pair['output'][window])
                measures['input_shapes'].add(pair['input'].shape)
                measures['radii'].add(a['radius'])
                measures['sizes'].add(a['size'])
                measures['loads'].add(len(a['shuttles']))
                measures['backgrounds'].add(a['background'])
                measures['threads'].add(a['thread'])
                measures['layouts'].add((a['size'], a['length'], tuple((m['direction'], m['budget']) for m in a['shuttles'])))
                for m in a['shuttles']:
                    measures['budgets'].add(m['budget'])
                    measures['cavity_areas'].update(map(len, _components(m['holes'])))
                    measures['cavity_shapes'].add((m['holes'].shape, min(v.tobytes() for v in _transforms(m['holes']))))
                clipped_tests += split == 'test' and not _fully_visible(a)
    assert len(signatures) == 42
    assert len(measures['input_shapes']) >= 35
    assert measures['radii'] == set(range(4, 9))
    assert measures['sizes'] == {5, 7} and measures['loads'] == {2, 3, 4}
    assert measures['budgets'] == set(range(5))
    assert len(measures['backgrounds']) == len(measures['threads']) == 10
    assert len(measures['layouts']) >= 80 and len(measures['cavity_shapes']) >= 35
    assert max(measures['cavity_areas']) >= 25 and clipped_tests >= 5
    for radius in range(4, 9):
        for budget in range(5):
            for direction in range(4):
                cells, end = _walk(radius, budget, direction)
                assert end == _endpoint(radius, budget, direction)
                for shape, center in (((30, 30), (4, 20)), ((23, 15), (18, 5))):
                    expected = np.zeros(shape, dtype=bool)
                    for y, x in cells:
                        y, x = y+center[0], x+center[1]
                        if 0 <= y < shape[0] and 0 <= x < shape[1]:
                            expected[y, x] = True
                    assert np.array_equal(expected, _analytic_mask(shape, center, radius, budget, direction))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(100+nt*10+ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    assert _equal(generate_puzzle(719917181, 2, 1), generate_puzzle(719917181, 2, 1))
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('mutated output accepted')
    return {k: len(v) for k, v in measures.items()} | {'puzzles': 42, 'clipped_tests': clipped_tests}


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
