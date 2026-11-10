"""Anchored textured archipelago: one registered sheet seen through copied masks.

Only Python's standard library and NumPy are used. No parent code is imported.
"""

from __future__ import annotations

import numpy as np


def _pick(rng, choices):
    return choices[int(rng.integers(len(choices)))]


def _key(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return a.shape, a.tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = []
    while remaining:
        start = remaining.pop()
        todo, group = [start], [start]
        while todo:
            r, c = todo.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
                    group.append(q)
        groups.append(group)
    return groups


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int64, copy=False)


def _parse(input_grid):
    """Read every convention from this input's pixels and fixed card geometry."""
    g = _grid(input_grid)
    bg, frame = int(g[0, 0]), int(g[1, 1])
    assert frame != bg and np.all(g[0] == bg) and np.all(g[:, 0] == bg)
    end = 1
    while end < g.shape[1] and g[1, end] == frame:
        end += 1
    width = end - 1
    assert width >= 7 and (width-3) % 2 == 0
    n = (width-3)//2
    assert 2 <= n <= 5 and g.shape[0] > n+4
    card = g[1:n+3, 1:2*n+4]
    assert card.shape == (n+2, 2*n+3)
    assert np.all(card[0] == frame) and np.all(card[-1] == frame)
    assert np.all(card[:, 0] == frame) and np.all(card[:, -1] == frame)
    assert np.all(card[:, n+1] == frame)
    carrier = g[2:n+2, 2:n+2].copy()
    stencil = g[2:n+2, n+3:2*n+3].copy()
    assert np.all((carrier != bg) & (carrier != frame))
    assert not np.any(stencil == frame)
    outside = g[:n+4].copy()
    outside[1:n+3, 1:2*n+4] = bg
    assert np.all(outside == bg)
    stage = g[n+4:]
    colors = [int(v) for v in np.unique(stage) if v != bg]
    assert len(colors) == 2 and frame not in colors
    groups = {v: _components(stage == v) for v in colors}
    eligible = [v for v in colors if len(groups[v]) == 2
                and sorted(map(len, groups[v]))[0] == 1
                and sorted(map(len, groups[v]))[1] >= 3]
    assert len(eligible) == 1
    ink = eligible[0]
    marker = next(v for v in colors if v != ink)
    assert all(len(c) == 1 for c in groups[marker])
    glyph = np.array(next(c for c in groups[ink] if len(c) > 1), dtype=int)
    anchor_pixel = next(c[0] for c in groups[ink] if len(c) == 1)
    tr, tc = map(int, glyph.min(axis=0))
    br, bc = map(int, glyph.max(axis=0))
    h, w = br-tr+1, bc-tc+1
    assert 2 <= h <= 5 and 2 <= w <= 5
    mask = stage[tr:br+1, tc:bc+1] == ink
    points = [anchor_pixel] + [c[0] for c in groups[marker]]
    pr, pc = map(int, np.min(points, axis=0))
    er, ec = map(int, np.max(points, axis=0))
    assert br+1 < pr or er+1 < tr or bc+1 < pc or ec+1 < tc
    assert all((r-pr) % 2 == 0 and (c-pc) % 2 == 0 for r, c in points)
    m, k = (er-pr)//2+1, (ec-pc)//2+1
    assert 1 <= m <= 6 and 1 <= k <= 6 and len(points) >= 2
    layout = np.zeros((m, k), dtype=bool)
    for r, c in points:
        layout[(r-pr)//2, (c-pc)//2] = True
    anchor = ((int(anchor_pixel[0])-pr)//2, (int(anchor_pixel[1])-pc)//2)
    origin = (tr+n+4-anchor[0]*h, tc-anchor[1]*w)
    H, W = m*h, k*w
    assert origin[0] >= n+4 and origin[1] >= 0
    assert origin[0]+H <= g.shape[0] and origin[1]+W <= g.shape[1]
    vertical = not np.all(carrier == carrier[:1])
    horizontal = not np.all(carrier == carrier[:, :1])
    assert not vertical or H >= n and H % n == 0
    assert not horizontal or W >= n and W % n == 0
    return dict(shape=g.shape, bg=bg, frame=frame, n=n, ink=ink, marker=marker,
                mask=mask, layout=layout, anchor=anchor, origin=origin,
                carrier=carrier, stencil=stencil, H=H, W=W,
                template_top=(tr+n+4, tc), map_points=points)


def _sheet(info, mode="correct"):
    n, H, W = info['n'], info['H'], info['W']
    rr, cc = np.arange(H)[:, None], np.arange(W)[None, :]
    sr, sc = rr*n//H, cc*n//W
    carrier, stencil, bg = info['carrier'], info['stencil'], info['bg']
    h, w = info['mask'].shape
    ar, ac = info['anchor']
    base = carrier[sr, sc]
    native = stencil[(rr-ar*h) % n, (cc-ac*w) % n]
    if mode == 'tile_carrier':
        base = carrier[rr % n, cc % n]
    elif mode == 'resize_stencil':
        native = stencil[sr, sc]
    elif mode == 'assembly_phase':
        native = stencil[rr % n, cc % n]
    elif mode == 'assembly_phase_rows':
        native = stencil[rr % n, (cc-ac*w) % n]
    elif mode == 'assembly_phase_columns':
        native = stencil[(rr-ar*h) % n, cc % n]
    elif mode == 'reset_islands':
        base = carrier[(rr % h)*n//h, (cc % w)*n//w]
        native = stencil[(rr % h) % n, (cc % w) % n]
    elif mode == 'fuse_before_resize':
        fused = np.where(stencil == bg, carrier, stencil)
        return fused[sr, sc]
    elif mode == 'ignore_stencil':
        return base
    elif mode == 'opaque_stencil':
        return native
    return np.where(native == bg, base, native)


def _support(info, solid=False):
    return np.kron(info['layout'], np.ones_like(info['mask']) if solid else info['mask'])


def _put(info, sheet, support):
    out = np.full(info['shape'], info['bg'], dtype=np.int64)
    r, c = info['origin']
    out[r:r+info['H'], c:c+info['W']] = np.where(support, sheet, info['bg'])
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only decoding, global texture composition, and aperture clipping."""
    info = _parse(input_grid)
    return _put(info, _sheet(info), _support(info))


_REQUIRED = frozenset(('full_rectangle', 'solid_mask', 'monochrome', 'tile_carrier',
                       'resize_stencil', 'fuse_before_resize', 'ignore_stencil',
                       'opaque_stencil', 'reset_islands', 'assembly_phase',
                       'assembly_phase_rows', 'assembly_phase_columns',
                       'center_anchor', 'pitch_two', 'swapped_pitch',
                       'flip_mask_rows', 'flip_mask_columns'))


def _geometric_support(info, row_pitch, col_pitch, anchor=None):
    support = np.zeros(info['shape'], dtype=bool)
    ar, ac = info['anchor'] if anchor is None else anchor
    tr, tc = info['template_top']
    mr, mc = np.nonzero(info['mask'])
    for r, c in np.argwhere(info['layout']):
        rows, cols = tr+(r-ar)*row_pitch+mr, tc+(c-ac)*col_pitch+mc
        valid = (rows >= 0) & (rows < support.shape[0]) & (cols >= 0) & (cols < support.shape[1])
        support[rows[valid], cols[valid]] = True
    return support


def _evidence(info, out):
    support, sheet = _support(info), _sheet(info)
    covered = set()
    alternatives = {
        'full_rectangle': _put(info, sheet, np.ones_like(support)),
        'solid_mask': _put(info, sheet, _support(info, solid=True)),
        'monochrome': _put(info, np.full_like(sheet, info['ink']), support),
    }
    for mode in ('tile_carrier', 'resize_stencil', 'fuse_before_resize', 'ignore_stencil',
                 'opaque_stencil', 'reset_islands', 'assembly_phase',
                 'assembly_phase_rows', 'assembly_phase_columns'):
        alternatives[mode] = _put(info, _sheet(info, mode), support)
    for name, flipped in (('flip_mask_rows', np.flipud(info['mask'])),
                          ('flip_mask_columns', np.fliplr(info['mask']))):
        alternatives[name] = _put(info, sheet, np.kron(info['layout'], flipped))
    for name, alt in alternatives.items():
        if not np.array_equal(out, alt):
            covered.add(name)
    h, w = info['mask'].shape
    actual = out != info['bg']
    for name, rp, cp, anchor in (
        ('center_anchor', h, w, tuple((v-1)//2 for v in info['layout'].shape)),
        ('pitch_two', 2, 2, None), ('swapped_pitch', w, h, None)):
        if not np.array_equal(actual, _geometric_support(info, rp, cp, anchor)):
            covered.add(name)
    return covered


def _grow_mask(rng, h, w):
    """A spanning corner-to-corner path followed by variable frontier growth."""
    mask = np.zeros((h, w), dtype=bool)
    r = c = 0
    mask[0, 0] = True
    steps = [0]*(h-1) + [1]*(w-1)
    rng.shuffle(steps)
    for axis in steps:
        r += axis == 0
        c += axis == 1
        mask[r, c] = True
    target = max(int(mask.sum()), round(h*w*float(rng.uniform(.3, 1))))
    while mask.sum() < target:
        frontier = set()
        for r, c in np.argwhere(mask):
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= q[0] < h and 0 <= q[1] < w and not mask[q]:
                    frontier.add(q)
        mask[_pick(rng, sorted(frontier))] = True
    if rng.integers(2):
        mask = np.fliplr(mask)
    if rng.integers(2):
        mask = np.flipud(mask)
    return mask.copy()


def _layout(rng, m, k, allow_gaps=(True, True)):
    density = float(rng.uniform(.25, .9))
    layout = rng.random((m, k)) < density
    # Tight bounds plus explicit witnesses of the physical lattice pitch.
    for r in (0, m-1):
        layout[r, int(rng.integers(k))] = True
    for c in (0, k-1):
        layout[int(rng.integers(m)), c] = True
    if m > 1:
        r, c = int(rng.integers(m-1)), int(rng.integers(k))
        layout[r:r+2, c] = True
    if k > 1:
        r, c = int(rng.integers(m)), int(rng.integers(k-1))
        layout[r, c:c+2] = True
    for axis, permitted in enumerate(allow_gaps):
        sums = layout.sum(axis=1-axis)
        if not permitted:
            for v in np.flatnonzero(sums == 0):
                layout[(v, int(rng.integers(k))) if axis == 0
                       else (int(rng.integers(m)), v)] = True
    return layout


def _textures(rng, n, colors, informative):
    bg, frame, ink, marker = colors
    pool = [v for v in range(10) if v not in (bg, frame)]
    count = int(rng.integers(2 if informative else 1, 5))
    inks = rng.choice(pool, count, replace=False)
    rr, cc = np.indices((n, n))
    kind = int(rng.integers(2 if informative else 0, 6))
    if kind == 0:
        labels = np.zeros((n, n), dtype=int)
    elif kind == 1:
        line = rng.integers(count, size=n)
        labels = np.tile(line, (n, 1))
        if rng.integers(2):
            labels = labels.T
    elif kind == 2:
        labels = (rr+cc) % count
    elif kind == 3:
        labels = (rr//int(rng.integers(1, n)) + cc//int(rng.integers(1, n))) % count
    elif kind == 4:
        centers = rng.choice(n*n, count, replace=False)
        dist = abs(rr[..., None]-centers//n) + abs(cc[..., None]-centers % n)
        labels = np.argmin(dist, axis=2)
    else:
        labels = rng.integers(count, size=(n, n))
    carrier = inks[labels]
    stencil = np.full((n, n), bg, dtype=np.int64)
    if informative or rng.random() > .12:
        kind = int(rng.integers(5))
        density = float(rng.uniform(.15, .8))
        if kind == 0:
            opaque = rng.random((n, n)) < density
        elif kind == 1:
            opaque = rr == int(rng.integers(n))
            if rng.integers(2):
                opaque = opaque.T
        elif kind == 2:
            opaque = np.minimum.reduce([rr, cc, n-1-rr, n-1-cc]) == 0
            opaque &= rng.random((n, n)) > .2
        elif kind == 3:
            center = (int(rng.integers(n)), int(rng.integers(n)))
            opaque = abs(rr-center[0])+abs(cc-center[1]) <= int(rng.integers(1, n))
        else:
            opaque = rng.random((n, n)) > density
        if not opaque.any():
            opaque.ravel()[int(rng.integers(n*n))] = True
        if opaque.all():
            opaque.ravel()[int(rng.integers(n*n))] = False
        stencil_inks = rng.choice(pool, int(rng.integers(1, min(4, int(opaque.sum()))+1)), replace=False)
        stencil[opaque] = rng.choice(stencil_inks, int(opaque.sum()))
    return carrier.astype(np.int64), stencil


def _assemble(rng, mask, layout, anchor, carrier, stencil, colors):
    """Place the input on a plane; generate labels by an explicit pixel loop."""
    bg, frame, ink, marker = colors
    n = carrier.shape[0]
    h, w = mask.shape
    m, k = layout.shape
    H, W, ph, pw = m*h, k*w, 2*m-1, 2*k-1
    tr, tc = anchor[0]*h, anchor[1]*w
    positions = []
    for pr in range(-ph-1, H+2):
        for pc in range(-pw-1, W+2):
            if not (pr+ph < tr or tr+h < pr or pc+pw < tc or tc+w < pc):
                continue
            minr, minc = min(0, pr), min(0, pc)
            bh, bw = max(H, pr+ph)-minr, max(W, pc+pw)-minc
            # Outer column zero is reserved for background.
            if bh <= 26-n and bw <= 29:
                positions.append((pr, pc, minr, minc, bh, bw))
    assert positions, 'no valid stage placement'
    # Some tight arrangements and some padded ones; no fixed object quadrant.
    if rng.random() < .4:
        area = min(p[4]*p[5] for p in positions)
        positions = [p for p in positions if p[4]*p[5] <= area+12]
    pr, pc, minr, minc, bh, bw = _pick(rng, positions)
    stage_h = int(rng.integers(bh, 27-n))
    canvas_w = int(rng.integers(max(bw+1, 2*n+5), 31))
    top = n+4+int(rng.integers(stage_h-bh+1))-minr
    left = 1+int(rng.integers(canvas_w-bw))-minc
    g = np.full((n+4+stage_h, canvas_w), bg, dtype=np.int64)
    g[1:n+3, 1:2*n+4] = frame
    g[2:n+2, 2:n+2] = carrier
    g[2:n+2, n+3:2*n+3] = stencil
    mr, mc = np.nonzero(mask)
    g[top+tr+mr, left+tc+mc] = ink
    for r, c in np.argwhere(layout):
        g[top+pr+2*r, left+pc+2*c] = ink if (r, c) == anchor else marker
    # Unlike solve's vectorized sheet construction, this only visits apertures.
    out = np.full_like(g, bg)
    for r, c in np.argwhere(layout):
        for a, b in zip(mr, mc):
            y, x = int(r*h+a), int(c*w+b)
            payload = stencil[(y-tr) % n, (x-tc) % n]
            if payload == bg:
                payload = carrier[y*n//H, x*n//W]
            out[top+y, left+x] = payload
    return {'input': g, 'output': out}


def _sample(rng, periods, orientation, colors, informative=False, allow_gaps=(True, True)):
    n = int(_pick(rng, periods))
    carrier, stencil = _textures(rng, n, colors, informative)
    vertical = not np.all(carrier == carrier[:1])
    horizontal = not np.all(carrier == carrier[:, :1])
    options = []
    for h in range(2, 6):
        for w in range(2, 6):
            for m in range(1, 7):
                for k in range(1, 7):
                    H, W = h*m, w*k
                    if m*k < 2 or H > 24-n or W > 27:
                        continue
                    if informative and (m < 2 or k < 2 or h == w):
                        continue
                    if orientation and np.sign(k-m) not in (orientation, 0):
                        continue
                    if vertical and (H < n or H % n):
                        continue
                    if horizontal and (W < n or W % n):
                        continue
                    options.append((h, w, m, k))
    assert options
    h, w, m, k = _pick(rng, options)
    mask = _grow_mask(rng, h, w)
    layout = _layout(rng, m, k, allow_gaps)
    points = [tuple(map(int, p)) for p in np.argwhere(layout)]
    center = ((m-1)//2, (k-1)//2)
    anchors = [p for p in points if not informative or p != center]
    anchor = _pick(rng, anchors or points)
    return _assemble(rng, mask, layout, anchor, carrier, stencil, colors)


def _training_diverse(infos):
    return (len({i['mask'].shape[0] for i in infos}) >= 2
            and len({i['mask'].shape[1] for i in infos}) >= 2
            and len({i['anchor'][0] for i in infos}) >= 2
            and len({i['anchor'][1] for i in infos}) >= 2
            and len({(i['H'], i['W']) for i in infos}) >= 2
            and any(i['mask'].shape[0] != i['mask'].shape[1] for i in infos))


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f'{name} must be an integer')
        if not low <= int(value) <= high:
            raise ValueError(f'{name} must be in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(3))
    train_periods = (2, 3) if shift == 0 else (2, 3, 4)
    test_periods = (4, 5) if shift == 0 else (2, 3, 4, 5)
    train_orientation = (0, -1, 1)[shift]
    test_orientation = (0, 1, -1)[shift]
    consistent = rng.random() < .25
    base_colors = tuple(map(int, rng.choice(10, 4, replace=False)))

    def palette():
        return base_colors if consistent else tuple(map(int, rng.choice(10, 4, replace=False)))

    candidates, infos, covers = [], [], []
    chosen = None
    for _ in range(120):
        pair = _sample(rng, train_periods, train_orientation, palette(), informative=True)
        info = _parse(pair['input'])
        cover = _evidence(info, pair['output'])
        for j in rng.permutation(len(candidates)):
            j = int(j)
            if (_REQUIRED <= covers[j] | cover and _training_diverse([infos[j], info])
                    and _key(candidates[j]['input']) != _key(pair['input'])):
                chosen = [candidates[j], pair]
                break
        candidates.append(pair)
        infos.append(info)
        covers.append(cover)
        if chosen is not None:
            break
    if chosen is None:
        # A finite, constructive witness search, retaining this seed's palette.
        # This removes dependence on a probabilistic teaching-selection success.
        chosen = _fallback_training(rng, palette())
    puzzle = {'train': chosen, 'test': []}
    used = {_key(p['input']) for p in chosen}
    for _ in range(nt-2):
        pair = _unique_sample(rng, train_periods, train_orientation, palette, used)
        puzzle['train'].append(pair)
    training = [_parse(p['input']) for p in puzzle['train']]
    gaps = tuple(any(not np.all(i['layout'].any(axis=1-axis)) for i in training) for axis in (0, 1))
    for _ in range(nq):
        # Shifted aspect is strict in tests; period shift is visibly encoded.
        pair = _unique_sample(rng, test_periods, test_orientation, palette, used, gaps,
                              strict_orientation=bool(test_orientation))
        puzzle['test'].append(pair)
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _unique_sample(rng, periods, orientation, palette, used, gaps=(True, True), strict_orientation=False):
    for _ in range(100):
        pair = _sample(rng, periods, orientation, palette(), allow_gaps=gaps)
        info = _parse(pair['input'])
        m, k = info['layout'].shape
        if strict_orientation and np.sign(k-m) != orientation:
            continue
        key = _key(pair['input'])
        if key not in used:
            used.add(key)
            return pair
    # At most 14 inputs are used. At least 32 visibly different single-mark
    # cards guarantee a fresh member even after an unlucky bounded sample.
    colors = palette()
    bg, frame, ink, marker = colors
    n = int(periods[0])
    m, k = (4, 2) if orientation < 0 else (2, 4) if orientation > 0 else (2, 3)
    mask = np.array([[1, 1, 0], [1, 1, 1]], dtype=bool)
    layout = np.ones((m, k), dtype=bool)
    payloads = [v for v in range(10) if v not in (bg, frame)]
    carrier = np.full((n, n), payloads[0], dtype=int)
    for location in range(n*n):
        for payload in payloads:
            stencil = np.full((n, n), bg, dtype=int)
            stencil.ravel()[location] = payload
            pair = _assemble(rng, mask, layout, (m-1, k-1), carrier, stencil, colors)
            key = _key(pair['input'])
            if key not in used:
                used.add(key)
                return pair
    raise AssertionError('finite uniqueness construction exhausted')


def _fallback_training(rng, colors):
    """Constructive teaching examples, used only after the random pool fails."""
    n = 3
    bg, frame, ink, marker = colors
    payload = [c for c in range(10) if c not in (bg, frame)]
    a, b, c = map(int, rng.choice(payload, 3, replace=False))
    carrier = np.array([[a, b, c], [c, a, b], [b, c, a]])
    stencil = np.full((n, n), bg, dtype=int)
    stencil[0, 1], stencil[1, 0], stencil[2, 2] = c, b, a
    first_mask = np.ones((2, 4), dtype=bool)
    first_mask[0, 0] = False
    second_mask = np.ones((4, 2), dtype=bool)
    second_mask[-1, -1] = False
    layout = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=bool)
    pairs = [_assemble(rng, first_mask, layout, (0, 1), carrier, stencil, colors),
             _assemble(rng, second_mask, layout, (2, 2), carrier[::-1].copy(), stencil.T.copy(), colors)]
    assert _REQUIRED <= set().union(*(_evidence(_parse(p['input']), p['output']) for p in pairs))
    return pairs


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, coverage = set(), set()
    decoded = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _key(inp)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            info = _parse(inp)
            decoded[split].append(info)
            assert inp.shape == out.shape
            assert np.array_equal(solve(inp), out), 'incorrect output'
            assert not np.array_equal(inp, out)
            support = _support(info)
            assert np.count_nonzero(out != info['bg']) == int(info['layout'].sum())*int(info['mask'].sum())
            r, c = info['origin']
            assert np.array_equal(out[r:r+info['H'], c:c+info['W']] != info['bg'], support)
            exterior = out.copy()
            exterior[r:r+info['H'], c:c+info['W']] = info['bg']
            assert np.all(exterior == info['bg']), 'unpainted regions or card retained'
            if split == 'train':
                coverage |= _evidence(info, out)
    assert _REQUIRED <= coverage, f'training misses {_REQUIRED-coverage}'
    assert _training_diverse(decoded['train']), 'training geometry insufficient'
    for axis in (0, 1):
        if any(not np.all(i['layout'].any(axis=1-axis)) for i in decoded['test']):
            assert any(not np.all(i['layout'].any(axis=1-axis)) for i in decoded['train'])


def diversity_metrics(puzzles):
    """Actual decoded structural properties; no sampler-mode labels are used."""
    metrics = dict(periods=set(), glyphs=set(), layouts=set(), anchors=set(),
                   shapes=set(), carriers=set(), stencils=set(), roles=set(),
                   counts=set(), densities=set(), phases=set(), aspect_shifts=set())
    for puzzle in puzzles:
        for split in ('train', 'test'):
            for p in puzzle[split]:
                i = _parse(p['input'])
                metrics['periods'].add(i['n'])
                metrics['glyphs'].add(_key(i['mask']))
                metrics['layouts'].add(_key(i['layout']))
                metrics['anchors'].add(i['anchor'])
                metrics['shapes'].add(p['input'].shape)
                metrics['carriers'].add(_normalized(i['carrier']))
                metrics['stencils'].add(_normalized(i['stencil']))
                metrics['roles'].add((i['bg'], i['frame'], i['ink'], i['marker']))
                metrics['counts'].add(int(i['layout'].sum()))
                metrics['densities'].add(round(float(i['mask'].mean()), 2))
                h, w = i['mask'].shape
                metrics['phases'].add((i['anchor'][0]*h % i['n'], i['anchor'][1]*w % i['n']))
        signs = []
        for split in ('train', 'test'):
            signs.append(tuple(sorted({int(np.sign(i['layout'].shape[1]-i['layout'].shape[0]))
                                       for i in map(lambda p: _parse(p['input']), puzzle[split])})))
        metrics['aspect_shifts'].add(tuple(signs))
    return {k: len(v) for k, v in metrics.items()}


def _normalized(grid):
    mapping = {}
    return grid.shape, bytes(mapping.setdefault(int(c), len(mapping)) for c in grid.flat)


def self_test():
    """Bounded 42-seed variation, counterfactuals, counts, and corruption checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures = [tuple((_key(p['input']), _key(p['output']))
                        for s in ('train', 'test') for p in puzzle[s]) for puzzle in puzzles]
    assert len(set(signatures)) == 42
    replay = generate_puzzle(17)
    assert signatures[17] == tuple((_key(p['input']), _key(p['output']))
                                  for s in ('train', 'test') for p in replay[s])
    metrics = diversity_metrics(puzzles)
    for name, minimum in (('periods', 4), ('glyphs', 55), ('layouts', 70), ('anchors', 15),
                          ('shapes', 55), ('carriers', 55), ('stencils', 55),
                          ('roles', 60), ('counts', 12), ('densities', 12), ('phases', 12)):
        assert metrics[name] >= minimum, (name, metrics[name])
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(6000+10*nt+nq, nt, nq))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (2.0, 1), (True, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Exercise the constructive witness path even when normal selection succeeds.
    rng = np.random.default_rng(90210)
    witnesses = _fallback_training(rng, (0, 9, 1, 2))
    validate_generated_puzzle({'train': witnesses, 'test': [generate_puzzle(91)['test'][0]]})
    # Move the anchor by swapping two map inks; output must change, without labels.
    # Use the large-canvas witness, so this check cannot be skipped for lack of room.
    p = witnesses[0]
    old = p['input']
    i = _parse(old)
    changed = np.full((30, 30), i['bg'], dtype=int)
    changed[:old.shape[0], :old.shape[1]] = old
    # Reposition the two known witness objects with safe margins on all sides.
    stage_start = i['n']+4
    changed[stage_start:] = i['bg']
    mr, mc = np.nonzero(i['mask'])
    changed[stage_start+6+mr, 10+mc] = i['ink']
    for r, c in np.argwhere(i['layout']):
        changed[stage_start+1+2*r, 2+2*c] = (i['ink'] if (r, c) == i['anchor']
                                                          else i['marker'])
    i = _parse(changed)
    original_answer = solve(changed)
    ar, ac = next((r+stage_start, c) for r, c in i['map_points']
                  if changed[r+stage_start, c] == i['ink'])
    ordinary = [(r+stage_start, c) for r, c in i['map_points']
                if changed[r+stage_start, c] == i['marker']]
    moved = False
    for br, bc in ordinary:
        candidate = changed.copy()
        candidate[ar, ac], candidate[br, bc] = i['marker'], i['ink']
        try:
            answer = solve(candidate)
        except AssertionError:  # Some moves exceed this particular canvas.
            continue
        assert not np.array_equal(answer, original_answer)
        moved = True
        break
    assert moved, 'no valid anchor counterfactual exercised'
    damaged = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in puzzles[0][s]]
               for s in ('train', 'test')}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt label accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed')
