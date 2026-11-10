"""Fault echoes across tiled seams.

No parent code, source pairs, seeds, or training labels are consulted by solve.
The intended glyph consists of aligned 2x2 uniform blocks, each with at most
one flipped pixel. Damage is classified before substitution; its halos are
computed AFTER assembling all copies, in the final pixel coordinate system.
"""
from __future__ import annotations

import numpy as np

HIGHLIGHT = 7
PALETTE = tuple(c for c in range(10) if c != HIGHLIGHT)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid of size 1..30")
    return a.astype(np.int8, copy=True)


def _box(points):
    a = np.asarray(points, dtype=int)
    lo, hi = a.min(axis=0), a.max(axis=0) + 1
    return tuple(map(int, (*lo, *hi)))  # top, left, bottom-exclusive, right-exclusive


def _connected(a):
    points = set(map(tuple, np.argwhere(a)))
    if not points:
        return False
    seen = {min(points)}
    stack = list(seen)
    while stack:
        r, c = stack.pop()
        for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if q in points and q not in seen:
                seen.add(q)
                stack.append(q)
    return len(seen) == len(points)


def _dilate(a):
    h, w = a.shape
    padded = np.pad(a, 1)
    return np.logical_or.reduce([padded[r:r+h, c:c+w]
                                 for r in range(3) for c in range(3)])


def _decode(inp):
    """Identify the map by its ordinary points, then decode the glyph blocks."""
    colors, counts = np.unique(inp, return_counts=True)
    if len(colors) != 3 or HIGHLIGHT in colors:
        raise ValueError("input must have three colors, excluding highlight 7")
    if np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("background must be uniquely most frequent")
    bg = int(colors[np.argmax(counts)])
    foreground = [int(c) for c in colors if c != bg]
    candidates = []
    for ink, marker in (foreground, foreground[::-1]):
        ordinary = np.argwhere(inp == marker)
        if len(ordinary) < 3:
            continue
        if any(inp[rr, cc] == marker
               for r, c in ordinary
               for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
               if 0 <= rr < inp.shape[0] and 0 <= cc < inp.shape[1]):
            continue
        pr, pc, pb, pe = _box(ordinary)
        ink_points = np.argwhere(inp == ink)
        inside = ((ink_points[:, 0] >= pr) & (ink_points[:, 0] < pb)
                  & (ink_points[:, 1] >= pc) & (ink_points[:, 1] < pe))
        if inside.sum() != 1 or (~inside).sum() < 8:
            continue
        anchor_pixel = ink_points[inside][0]
        tr, tc, tb, te = _box(ink_points[~inside])
        h, w = tb-tr, te-tc
        if h % 2 or w % 2 or not (4 <= h <= 8 and 4 <= w <= 8):
            continue
        # An entire background row/column separates the two bounding boxes.
        if not (tb < pr or pb < tr or te < pc or pe < tc):
            continue
        observed = inp[tr:tb, tc:te] == ink
        if np.any(inp[tr:tb, tc:te] == marker):
            continue
        block_counts = observed.reshape(h//2, 2, w//2, 2).sum(axis=(1, 3))
        if np.any(block_counts == 2):
            continue
        coarse = block_counts >= 3
        if (not _connected(coarse) or coarse.all()
                or not coarse[0].any() or not coarse[-1].any()
                or not coarse[:, 0].any() or not coarse[:, -1].any()):
            continue
        base = np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1)
        holes = base & ~observed
        extras = ~base & observed
        if not holes.any():
            continue
        points = np.vstack((ordinary, anchor_pixel))
        if np.any((points - [pr, pc]) % 2):
            continue
        mh, mw = (pb-pr+1)//2, (pe-pc+1)//2
        layout = np.zeros((mh, mw), dtype=bool)
        indices = (points - [pr, pc])//2
        layout[indices[:, 0], indices[:, 1]] = True
        anchor = tuple(map(int, (anchor_pixel - [pr, pc])//2))
        top, left = tr-anchor[0]*h, tc-anchor[1]*w
        if not (0 <= top <= inp.shape[0]-mh*h
                and 0 <= left <= inp.shape[1]-mw*w):
            continue
        candidates.append(dict(bg=bg, ink=ink, marker=marker, base=base,
                               holes=holes, extras=extras, coarse=coarse,
                               layout=layout, anchor=anchor, tr=tr, tc=tc,
                               top=top, left=left, map_box=(pr, pc, pb, pe)))
    if len(candidates) != 1:
        raise ValueError("input does not uniquely encode a family member")
    return candidates[0]


def _assembly_masks(shape, m):
    """Vectorized substitution, used by the input-only solver."""
    base, holes, layout = m['base'], m['holes'], m['layout']
    rr, cc = m['top'], m['left']
    hh, ww = base.shape[0]*layout.shape[0], base.shape[1]*layout.shape[1]
    solid = np.zeros(shape, dtype=bool)
    missing = np.zeros(shape, dtype=bool)
    anchor = np.zeros(shape, dtype=bool)
    solid[rr:rr+hh, cc:cc+ww] = np.kron(layout, base)
    missing[rr:rr+hh, cc:cc+ww] = np.kron(layout, holes)
    anchor[m['tr']:m['tr']+base.shape[0], m['tc']:m['tc']+base.shape[1]] = base
    return solid, missing, anchor


def solve(input_grid: np.ndarray) -> np.ndarray:
    inp = _grid(input_grid)
    m = _decode(inp)
    solid, missing, anchor = _assembly_masks(inp.shape, m)
    out = np.full_like(inp, m['bg'])
    out[solid & ~missing] = m['ink']
    out[anchor & ~missing] = m['marker']
    out[_dilate(missing) & ~missing] = HIGHLIGHT
    return out


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _counts(num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10),
                                ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not lo <= value <= hi):
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _template(rng, dims):
    """Variable connected coarse scaffold, with intact block-size evidence."""
    h, w = dims
    coarse = rng.random(dims) < rng.uniform(.25, .75)
    full_row, full_col = int(rng.integers(h)), int(rng.integers(w))
    coarse[full_row] = True
    coarse[:, full_col] = True
    # Keep only cells attached to the spanning cross, then add random frontier.
    seen = {(full_row, full_col)}
    stack = list(seen)
    while stack:
        r, c = stack.pop()
        for a, b in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= a < h and 0 <= b < w and coarse[a, b] and (a, b) not in seen:
                seen.add((a, b)); stack.append((a, b))
    coarse[:] = False
    for p in seen:
        coarse[p] = True
    if coarse.all():
        choices = [(r, c) for r in range(h) for c in range(w)
                   if r != full_row and c != full_col]
        coarse[_pick(rng, choices)] = False
    return np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1), full_row, full_col


def _damage(rng, base, row, col, style, teaching=False):
    h, w = base.shape
    holes, extras = np.zeros_like(base), np.zeros_like(base)
    used = set()

    def add(r, c):
        key = (r//2, c//2)
        if base[r, c] and key not in used:
            holes[r, c] = True
            used.add(key)

    if style == 'horizontal':
        # One-sided damage makes neighboring copies acquire NEW halo pixels.
        add(2*row + int(rng.integers(2)), w-1)
    elif style == 'vertical':
        x = 2*col + int(rng.integers(2))
        add(0, x); add(h-1, x)
    elif style == 'both':
        y, x = 2*row + int(rng.integers(2)), 2*col + int(rng.integers(2))
        add(y, w-1); add(0, x); add(h-1, x)
    elif style == 'cluster':
        # Four flips around a coarse-block intersection: one per block.
        intersections = [(r, c) for r in range(2, h, 2) for c in range(2, w, 2)
                         if base[r-1:r+1, c-1:c+1].all()]
        if intersections:
            r, c = _pick(rng, intersections)
            for a, b in ((r-1, c-1), (r-1, c), (r, c-1), (r, c)):
                add(a, b)
    wanted = int(rng.integers(max(1, int(holes.sum())),
                               min(6, int(base.sum())//4)+1))
    available = list(map(tuple, np.argwhere(base)))
    rng.shuffle(available)
    for r, c in available:
        if holes.sum() >= wanted:
            break
        # Keep the first horizontal witness one-sided at that same row.
        if style == 'horizontal' and c == 0:
            continue
        add(int(r), int(c))
    blank_blocks = np.argwhere(~base[::2, ::2])
    rng.shuffle(blank_blocks)
    budget = int(rng.integers(1 if teaching else 0, min(4, len(blank_blocks))+1))
    for r, c in blank_blocks[:budget]:
        extras[2*r+int(rng.integers(2)), 2*c+int(rng.integers(2))] = True
    return holes, extras


def _layout(rng, dims, gapped=False, sparse=False, teaching=False):
    h, w = dims
    rows, cols = list(range(h)), list(range(w))
    if gapped:
        if h >= 4: rows.remove(int(rng.integers(2, h-1)))
        if w >= 4: cols.remove(int(rng.integers(2, w-1)))
    layout = np.zeros(dims, dtype=bool)
    for r in rows:
        for c in cols:
            layout[r, c] = True
    corners = {(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)}
    available = [p for p in map(tuple, np.argwhere(layout)) if p not in corners]
    rng.shuffle(available)
    target = max(4, int(round(layout.sum()*rng.uniform(.25, 1.0))))
    if teaching:
        target = min(target, int(layout.sum())-1)
    for p in available:
        if layout.sum() <= target:
            break
        layout[p] = False
        vertical = (layout[1:] & layout[:-1]).any()
        horizontal = (layout[:, 1:] & layout[:, :-1]).any()
        if not (vertical and horizontal and (sparse or gapped or _connected(layout))):
            layout[p] = True
    return layout


def _place(rng, base, holes, extras, layout, anchor, colors, edge=False):
    """Fit the anchored assembly and a separated input constellation on a plane."""
    h, w = base.shape
    mh, mw = layout.shape
    ah, aw, ph, pw = h*mh, w*mw, 2*mh-1, 2*mw-1
    tr, tc = anchor[0]*h, anchor[1]*w
    positions = []
    for r in range(-ph-1, ah+2):
        for c in range(-pw-1, aw+2):
            if not (r+ph < tr or tr+h < r or c+pw < tc or tc+w < c):
                continue
            lowr, lowc = min(0, r), min(0, c)
            highr, highc = max(ah, r+ph), max(aw, c+pw)
            if highr-lowr <= 30 and highc-lowc <= 30:
                if not edge or (0 <= r and r+ph <= ah and 0 <= c and c+pw <= aw):
                    positions.append((r, c, lowr, lowc, highr, highc))
    if not positions:
        raise ValueError('no valid constellation placement')
    r, c, lowr, lowc, highr, highc = _pick(rng, positions)
    min_h, min_w = highr-lowr, highc-lowc
    gh = min_h if edge else int(rng.integers(min_h, min(30, min_h+5)+1))
    gw = min_w if edge else int(rng.integers(min_w, min(30, min_w+5)+1))
    offset_r = 0 if edge else int(rng.integers(gh-min_h+1))
    offset_c = 0 if edge else int(rng.integers(gw-min_w+1))
    origin_r, origin_c = offset_r-lowr, offset_c-lowc
    tr += origin_r; tc += origin_c
    bg, ink, marker = colors
    inp = np.full((gh, gw), bg, dtype=np.int8)
    glyph = np.where(base, ink, bg).astype(np.int8)
    glyph[holes] = bg
    glyph[extras] = ink
    inp[tr:tr+h, tc:tc+w] = glyph
    for a, b in np.argwhere(layout):
        inp[origin_r+r+2*a, origin_c+c+2*b] = ink if (a, b) == anchor else marker
    # Construction renderer is a pixel loop, independent of Kronecker solve.
    out = np.full_like(inp, bg)
    missing = set()
    for a, b in np.argwhere(layout):
        rr, cc = origin_r+a*h, origin_c+b*w
        color = marker if (a, b) == anchor else ink
        for y, x in np.argwhere(base):
            if holes[y, x]:
                missing.add((int(rr+y), int(cc+x)))
            else:
                out[rr+y, cc+x] = color
    for y, x in sorted(missing):
        for yy in range(max(0, y-1), min(gh, y+2)):
            for xx in range(max(0, x-1), min(gw, x+2)):
                if (yy, xx) not in missing:
                    out[yy, xx] = HIGHLIGHT
    return {'input': inp, 'output': out}


def _cropped_halos(inp, m):
    """Plausible incorrect shortcut: clip each halo to its own template box."""
    solid, missing, anchor = _assembly_masks(inp.shape, m)
    out = np.full_like(inp, m['bg'])
    out[solid & ~missing] = m['ink']
    out[anchor & ~missing] = m['marker']
    local = _dilate(m['holes']) & ~m['holes']
    h, w = m['base'].shape
    for r, c in np.argwhere(m['layout']):
        y, x = m['top']+r*h, m['left']+c*w
        out[y:y+h, x:x+w][local] = HIGHLIGHT
    return out


def _features(inp, m):
    solid, missing, anchor = _assembly_masks(inp.shape, m)
    halo = _dilate(missing) & ~missing
    layout, holes = m['layout'], m['holes']
    h, w = m['base'].shape
    repeated_extras = np.zeros_like(solid)
    rr, cc = m['top'], m['left']
    repeated_extras[rr:rr+h*layout.shape[0], cc:cc+w*layout.shape[1]] = np.kron(layout, m['extras'])
    return dict(shape=inp.shape, template=(h, w), map=layout.shape,
                anchor=m['anchor'], holes=int(holes.sum()), extras=int(m['extras'].sum()),
                copies=int(layout.sum()), glyph=(h, w, m['coarse'].tobytes()),
                constellation=(layout.shape, layout.tobytes()),
                palette=(m['bg'], m['ink'], m['marker']),
                gap_rows=bool(np.any(~layout.any(axis=1))),
                gap_cols=bool(np.any(~layout.any(axis=0))),
                connected=_connected(layout),
                horizontal=bool((layout[:, 1:] & layout[:, :-1]).any()),
                vertical=bool((layout[1:] & layout[:-1]).any()),
                clipped=bool(missing[0].any() or missing[-1].any()
                             or missing[:, 0].any() or missing[:, -1].any()),
                visible_erasure=bool((repeated_extras & ~halo).any()),
                seam=not np.array_equal(solve(inp), _cropped_halos(inp, m)),
                halo_in_blank=bool((halo & ~solid).any()),
                halo_in_anchor=bool((halo & anchor & ~missing).any()),
                joined_holes=bool((missing[1:] & missing[:-1]).any()
                                 or (missing[:, 1:] & missing[:, :-1]).any()),
                edge_contacts=(m['top'] == 0, m['left'] == 0,
                               m['top']+h*layout.shape[0] == inp.shape[0],
                               m['left']+w*layout.shape[1] == inp.shape[1]))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(4))
    stable_palette = bool(rng.integers(2))
    shared_colors = tuple(map(int, rng.choice(PALETTE, 3, replace=False)))
    puzzle = {'train': [], 'test': []}
    seen, previous_dims, previous_anchor = set(), None, None
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            teaching = split == 'train' and index < 2
            gapped = regime == 2 and (split == 'test' or index == 1)
            if regime == 0:
                maps = ((2, 3), (2, 4)) if split == 'train' else ((3, 2), (4, 2))
            elif regime == 1:
                maps = ((3, 2), (4, 2)) if split == 'train' else ((2, 3), (2, 4))
            elif regime == 2:
                maps = ((4, 4),) if gapped and split == 'train' else (
                    ((4, 4), (4, 5), (5, 4)) if split == 'test' else ((2, 3), (3, 2)))
            else:
                maps = ((2, 3), (3, 2), (3, 3)) if split == 'train' else (
                    (3, 3), (2, 4), (4, 2), (3, 4), (4, 3), (4, 4), (4, 5), (5, 4))
            if not teaching and not gapped and rng.random() < .18:
                horizontal = (split == 'train') == (regime == 0)
                if regime >= 2:
                    horizontal = bool(rng.integers(2))
                maps = tuple((1, n) if horizontal else (n, 1) for n in (4, 5, 6))
            for attempt in range(100):
                mh, mw = _pick(rng, maps)
                options = [(a, b) for a in range(2, 5) for b in range(2, 5)
                           if 2*a*mh <= 30 and 2*b*mw <= 30
                           and (not teaching or index != 0 or a != b)
                           and (not teaching or index != 1
                                or (a != previous_dims[0] and b != previous_dims[1]))]
                if not options:
                    continue
                dims = _pick(rng, options)
                base, row, col = _template(rng, dims)
                if teaching and index == 0 and any(np.array_equal(base, other)
                        for other in (np.flipud(base), np.fliplr(base), np.rot90(base, 2))):
                    continue
                style = ('horizontal' if index == 0 else 'vertical') if teaching else (
                    'both' if split == 'test' and regime == 3 else
                    _pick(rng, ('random', 'horizontal', 'vertical', 'both', 'cluster')))
                holes, extras = _damage(rng, base, row, col, style, teaching)
                layout = _layout(rng, (mh, mw), gapped,
                                 sparse=(not teaching and regime == 2), teaching=teaching)
                # The ordinary points alone must retain the map's tight box.
                anchors = []
                for p in map(tuple, np.argwhere(layout)):
                    remainder = layout.copy()
                    remainder[p] = False
                    if _box(np.argwhere(remainder)) == (0, 0, mh, mw):
                        anchors.append(p)
                if teaching and index == 0:
                    anchors = [p for p in anchors if p != ((mh-1)//2, (mw-1)//2)]
                if teaching and index == 1:
                    anchors = [p for p in anchors
                               if p[0] != previous_anchor[0] and p[1] != previous_anchor[1]]
                if not anchors:
                    continue
                anchor = _pick(rng, anchors)
                colors = shared_colors if stable_palette else tuple(
                    map(int, rng.choice(PALETTE, 3, replace=False)))
                pair = _place(rng, base, holes, extras, layout, anchor, colors,
                              edge=teaching and index == 0)
                inp = pair['input']
                # Demonstrations retain visible surviving ink in both roles;
                # too much damage can hide the anchor inversion under halos.
                if set(map(int, np.unique(pair['output']))) != set(colors) | {HIGHLIGHT}:
                    continue
                key = (inp.shape, inp.tobytes())
                if key in seen:
                    continue
                m = _decode(inp)
                if teaching and index == 0:
                    f = _features(inp, m)
                    if not (f['seam'] and f['halo_in_blank'] and f['visible_erasure']):
                        continue
                seen.add(key)
                puzzle[split].append(pair)
                if teaching and index == 0:
                    previous_dims, previous_anchor = dims, anchor
                break
            else:
                raise RuntimeError('bounded construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, features = set(), {'train': [], 'test': []}
    for split in puzzle:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            key = (inp.shape, inp.tobytes())
            assert key not in seen
            seen.add(key)
            m = _decode(inp)
            solid, missing, anchor = _assembly_masks(inp.shape, m)
            halo = _dilate(missing) & ~missing
            # Family invariants are checked separately from full solver equality.
            assert np.all(out[missing] == m['bg']), 'all repeated holes stay empty'
            assert np.all(out[halo] == HIGHLIGHT), 'exact global Moore halo'
            assert np.all(out[~solid & ~halo] == m['bg']), 'specks and map are erased'
            assert np.all(out[solid & ~missing & ~halo & ~anchor] == m['ink'])
            assert np.all(out[anchor & ~missing & ~halo] == m['marker'])
            assert missing.sum() == m['holes'].sum()*m['layout'].sum()
            assert np.array_equal(out, solve(inp))
            assert not np.array_equal(inp, out)
            # Parents A inputs have two colors; parent B outputs have three.
            # Every new pair has three input and four output colors.
            assert set(map(int, np.unique(out))) == {m['bg'], m['ink'], m['marker'], HIGHLIGHT}
            features[split].append(_features(inp, m))
    train, test = features['train'], features['test']
    assert len({f['anchor'][0] for f in train}) >= 2
    assert len({f['anchor'][1] for f in train}) >= 2
    assert len({f['template'][0] for f in train}) >= 2
    assert len({f['template'][1] for f in train}) >= 2
    assert any(f['seam'] for f in train), 'teach assembly-wide halos'
    assert any(f['extras'] for f in train) and all(f['holes'] for f in train)
    assert any(f['visible_erasure'] for f in train), 'show erased specks away from halos'
    assert any(f['horizontal'] and f['vertical'] for f in train)
    assert any(f['halo_in_blank'] for f in train)
    assert any(f['joined_holes'] for f in train), 'teach exclusion of neighboring hole centers'
    for flag in ('gap_rows', 'gap_cols'):
        assert not any(f[flag] for f in test) or any(f[flag] for f in train)


def self_test():
    """Bounded 42-seed suite, semantic counterfactuals, and legal-count checks."""
    # A valid connected scaffold without a complete coarse row or column:
    # solve is broader than the construction sampler's spanning-cross method.
    staircase = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=bool)
    base = np.repeat(np.repeat(staircase, 2, axis=0), 2, axis=1)
    holes, extras = np.zeros_like(base), np.zeros_like(base)
    holes[1, 1] = True
    extras[0, 6] = True
    layout = np.array([[1, 1, 1], [1, 0, 1]], dtype=bool)
    fixture = _place(np.random.default_rng(4567), base, holes, extras, layout,
                     (1, 0), (0, 5, 9), edge=True)
    assert np.array_equal(_decode(fixture['input'])['base'], base)
    assert np.array_equal(solve(fixture['input']), fixture['output'])
    for k in range(4):
        assert np.array_equal(solve(np.rot90(fixture['input'], k).T),
                              np.rot90(fixture['output'], k).T)
    # Hand-constructed anchor swap: only two map pixels change. Both anchored
    # assemblies fit, and an extra next to a fault must become warning color.
    inp = np.zeros((30, 30), dtype=np.int8)
    coarse = np.array([[1, 1, 1], [1, 1, 0]], dtype=bool)
    clean = np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1)
    glyph = clean.astype(np.int8)
    glyph[2, 3] = 0
    glyph[2, 4] = 1
    inp[10:14, 12:18] = glyph
    for r, c in np.argwhere(layout):
        inp[2*r, 2*c] = 1 if (r, c) == (0, 0) else 4
    answer = solve(inp)
    assert answer[12, 15] == 0 and answer[12, 16] == HIGHLIGHT
    swapped = inp.copy()
    swapped[0, 0], swapped[2, 4] = 4, 1
    assert not np.array_equal(solve(swapped), answer)
    features, signatures = [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(tuple((s, pair['input'].shape, pair['input'].tobytes(),
                              pair['output'].tobytes())
                             for s in p for pair in p[s]))
        for split in p:
            for pair in p[split]:
                inp, out = pair['input'], pair['output']
                m = _decode(inp)
                features.append(_features(inp, m))
                # Omitting A: replicate the observed dirty glyph, no classification/halo.
                raw = inp[m['tr']:m['tr']+m['base'].shape[0],
                          m['tc']:m['tc']+m['base'].shape[1]] == m['ink']
                bad = np.full_like(inp, m['bg'])
                h, w = raw.shape
                for r, c in np.argwhere(m['layout']):
                    y, x = m['top']+r*h, m['left']+c*w
                    bad[y:y+h, x:x+w][raw] = m['marker'] if (r, c) == m['anchor'] else m['ink']
                assert not np.array_equal(bad, out)
                # Omitting B: process damage once, with no map substitution.
                only = dict(m, layout=np.ones((1, 1), dtype=bool), anchor=(0, 0),
                            top=m['tr'], left=m['tc'])
                solid, missing, anchor = _assembly_masks(inp.shape, only)
                bad = np.full_like(inp, m['bg'])
                bad[solid & ~missing] = m['ink']
                bad[_dilate(missing) & ~missing] = HIGHLIGHT
                assert not np.array_equal(bad, out)
                # The rule commutes with every whole-grid quarter turn/reflection.
                if seed < 3:
                    for k in range(4):
                        a, b = np.rot90(inp, k).T, np.rot90(out, k).T
                        assert np.array_equal(solve(a), b)
        if seed < 3:
            replay = generate_puzzle(seed)
            assert all(np.array_equal(pair[k], other[k])
                       for s in p for pair, other in zip(p[s], replay[s]) for k in pair)
    assert len(signatures) == 42
    assert len({f['shape'] for f in features}) >= 45
    assert len({f['glyph'] for f in features}) >= 45
    assert len({f['constellation'] for f in features}) >= 35
    assert len({f['template'] for f in features}) >= 7
    assert len({f['anchor'] for f in features}) >= 9
    assert len({f['copies'] for f in features}) >= 9
    assert len({f['holes'] for f in features}) >= 4
    assert len({f['extras'] for f in features}) >= 3
    assert len({f['palette'] for f in features}) >= 30
    assert any(f['gap_rows'] and f['gap_cols'] for f in features)
    assert any(not f['connected'] for f in features)
    assert all(any(f['edge_contacts'][i] for f in features) for i in range(4))
    assert any(f['clipped'] for f in features) and any(not f['clipped'] for f in features)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(1000+nt*10+nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p = generate_puzzle(0)
    p['test'][0]['output'] = p['test'][0]['output'].copy()
    p['test'][0]['output'][0, 0] = (int(p['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds, count checks, symmetry and parent ablations')
