"""Registered mosaic completion: exact observation matching determines an anchor.

Only NumPy and the standard library are used. Solving never reads examples,
files, a seed, or generation state. The complete reference is in the input.
"""

import numpy as np


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def _d4(grid, index):
    # Four rotations, then a left/right reflection followed by four rotations.
    return np.rot90(np.fliplr(grid) if index >= 4 else grid, index % 4).copy()


def _pick(rng, sequence):
    return sequence[int(rng.integers(len(sequence)))]


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = []
    while remaining:
        start = remaining.pop()
        group, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    group.append(q)
        groups.append(group)
    return groups


def _bbox(mask):
    points = np.argwhere(mask)
    if not len(points):
        raise ValueError("empty object")
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    return tuple(map(int, (*lo, *hi)))


def _matches(orbit, patch):
    matches = []
    h, w = patch.shape
    for i, full in enumerate(orbit):
        H, W = full.shape
        if h > H or w > W:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(full, (h, w))
        offsets = np.argwhere(np.all(windows == patch, axis=(-1, -2)))
        matches.extend((i, int(r), int(c)) for r, c in offsets)
    return matches


def _parse(input_grid):
    inp = np.asarray(input_grid)
    if (inp.ndim != 2 or not np.issubdtype(inp.dtype, np.integer)
            or not all(1 <= side <= 30 for side in inp.shape)
            or np.any((inp < 0) | (inp > 9))):
        raise ValueError("invalid ARC grid")
    values, counts = np.unique(inp, return_counts=True)
    if len(values) != 4 or np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("need four roles and a uniquely dominant background")
    bg = int(values[np.argmax(counts)])
    foreground = [int(v) for v in values if v != bg]
    frames = []
    for color in foreground:
        r, c, b, e = _bbox(inp == color)
        h, w = b-r, e-c
        if min(h, w) < 5 or h % 2 != 1 or w % 2 != 1:
            continue
        expected = np.zeros(inp.shape, dtype=bool)
        expected[r, c:e] = expected[b-1, c:e] = True
        expected[r:b, c] = expected[r:b, e-1] = True
        if np.array_equal(inp == color, expected):
            frames.append((color, (r, c, b, e)))
    if len(frames) != 1:
        raise ValueError("need one complete observation frame")
    frame, rect = frames[0]
    r, c, b, e = rect
    inside = np.zeros(inp.shape, dtype=bool)
    inside[r+1:b-1, c+1:e-1] = True
    outside = np.ones(inp.shape, dtype=bool)
    outside[r:b, c:e] = False
    candidates = []
    for color in foreground:
        if color == frame or np.count_nonzero(inside & (inp == color)) != 1:
            continue
        groups = _components(outside & (inp == color))
        if len(groups) == 1 and len(groups[0]) >= 3:
            tr, tc, tb, te = _bbox(outside & (inp == color))
            if min(tb-tr, te-tc) >= 2:
                candidates.append((color, (tr, tc, tb, te)))
    if len(candidates) != 1:
        raise ValueError("glyph and exceptional point are not identifiable")
    ink, glyph_rect = candidates[0]
    marker = next(v for v in foreground if v not in (frame, ink))
    tr, tc, tb, te = glyph_rect
    glyph_region = inp[tr:tb, tc:te]
    if np.any((glyph_region != ink) & (glyph_region != bg)):
        raise ValueError("glyph box contains another object")
    glyph = glyph_region == ink
    atlas_points = outside & (inp == marker)
    if any(len(group) != 1 for group in _components(atlas_points)):
        raise ValueError("atlas points must be isolated")
    pr, pc, pb, pe = _bbox(atlas_points)
    if (pb-pr) % 2 != 1 or (pe-pc) % 2 != 1:
        raise ValueError("atlas extent does not use pitch two")
    coords = np.argwhere(atlas_points)
    if np.any((coords - (pr, pc)) % 2):
        raise ValueError("atlas coordinates do not use pitch two")
    atlas = (inp[pr:pb:2, pc:pe:2] == marker)
    if len({_key(_d4(atlas, i)) for i in range(8)}) != 8:
        raise ValueError("atlas must be asymmetric")
    sample = inp[r+1:b-1:2, c+1:e-1:2]
    if np.any(~np.isin(sample, (bg, marker, ink))):
        raise ValueError("invalid observation site")
    patch = sample != bg
    anchors = np.argwhere(sample == ink)
    if len(anchors) != 1 or int(patch.sum()) < 2 or patch.all():
        raise ValueError("observation needs one anchor, occupancy, and empty evidence")
    anchor_local = tuple(map(int, anchors[0]))
    expected_interior = np.full((b-r-2, e-c-2), bg, dtype=inp.dtype)
    expected_interior[::2, ::2] = sample
    if not np.array_equal(expected_interior, inp[r+1:b-1, c+1:e-1]):
        raise ValueError("observation has off-lattice cells")
    atlas_rect = (pr, pc, pb, pe)
    for a, other in ((rect, atlas_rect), (rect, glyph_rect),
                     (atlas_rect, glyph_rect)):
        if not _separate(a, other):
            raise ValueError("input object boxes must have a blank separation")
    # Check the complete input, not only cells convenient for decoding.
    rebuilt = np.full_like(inp, bg)
    rebuilt[pr:pb:2, pc:pe:2][atlas] = marker
    rebuilt[tr:tb, tc:te][glyph] = ink
    rebuilt[r, c:e] = rebuilt[b-1, c:e] = frame
    rebuilt[r:b, c] = rebuilt[r:b, e-1] = frame
    rebuilt[r+1:b-1:2, c+1:e-1:2] = sample
    if not np.array_equal(rebuilt, inp):
        raise ValueError("unrelated input cells")
    orbit = [_d4(atlas, i) for i in range(8)]
    matches = _matches(orbit, patch)
    if len(matches) != 1:
        raise ValueError("observation is ambiguous or has no exact match")
    orientation, row, col = matches[0]
    layout = orbit[orientation]
    if patch.size >= layout.size:
        raise ValueError("observation must be a proper crop")
    ar, ac = row + anchor_local[0], col + anchor_local[1]
    gh, gw = glyph.shape
    top, left = tr-ar*gh, tc-ac*gw
    if (top < 0 or left < 0 or top+layout.shape[0]*gh > inp.shape[0]
            or left+layout.shape[1]*gw > inp.shape[1]):
        raise ValueError("complete registered assembly does not fit")
    return dict(shape=inp.shape, roles=(bg, marker, ink, frame), atlas=atlas,
                layout=layout, patch=patch, glyph=glyph, orientation=orientation,
                offset=(row, col), local_anchor=anchor_local, anchor=(ar, ac),
                top=top, left=left, glyph_rect=glyph_rect, frame_rect=rect,
                atlas_rect=atlas_rect)


def _separate(a, b):
    # Rectangles use exclusive bottom/right; one full blank band is required.
    return (a[2]+1 <= b[0] or b[2]+1 <= a[0]
            or a[3]+1 <= b[1] or b[3]+1 <= a[1])


def _answer(info, *, crop_only=False, invert=True, solid=False, local=False):
    """Expansion independent of the generation painting loop."""
    bg, marker, ink, _ = info['roles']
    mask = np.ones_like(info['glyph']) if solid else info['glyph']
    layout = info['patch'] if crop_only else info['layout']
    anchor = info['local_anchor'] if crop_only or local else info['anchor']
    gh, gw = mask.shape
    tr, tc, _, _ = info['glyph_rect']
    top, left = tr-anchor[0]*gh, tc-anchor[1]*gw
    stamp = np.full((layout.shape[0]*gh, layout.shape[1]*gw), bg, dtype=np.int8)
    stamp[np.kron(layout, mask)] = ink if invert else marker
    exceptional = np.zeros_like(layout)
    exceptional[anchor] = True
    stamp[np.kron(exceptional, mask)] = marker if invert else ink
    out = np.full(info['shape'], bg, dtype=np.int8)
    # Alternatives may be out of bounds; their visible clipped predictions
    # suffice for comparing shortcuts. Valid intended answers are never clipped.
    r0, c0 = max(0, top), max(0, left)
    r1, c1 = min(out.shape[0], top+stamp.shape[0]), min(out.shape[1], left+stamp.shape[1])
    if r0 < r1 and c0 < c1:
        out[r0:r1, c0:c1] = stamp[r0-top:r1-top, c0-left:c1-left]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete registered mosaic solely from input pixels."""
    return _answer(_parse(input_grid))


def _connected_mask(rng, h, w, density):
    mask = np.zeros((h, w), dtype=bool)
    center = int(rng.integers(h)), int(rng.integers(w))
    mask[center] = True
    endpoints = [(0, int(rng.integers(w))), (h-1, int(rng.integers(w))),
                 (int(rng.integers(h)), 0), (int(rng.integers(h)), w-1)]
    rng.shuffle(endpoints)
    for endpoint in endpoints:
        r, c = center
        while (r, c) != endpoint:
            axes = [axis for axis in (0, 1) if (r, c)[axis] != endpoint[axis]]
            axis = _pick(rng, axes)
            if axis == 0:
                r += 1 if endpoint[0] > r else -1
            else:
                c += 1 if endpoint[1] > c else -1
            mask[r, c] = True
    target = max(int(mask.sum()), int(round(density*h*w)))
    while int(mask.sum()) < target:
        frontier = set()
        for r, c in np.argwhere(mask):
            for a, b in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= a < h and 0 <= b < w and not mask[a, b]:
                    frontier.add((int(a), int(b)))
        mask[_pick(rng, sorted(frontier))] = True
    return mask


def _crop_bank(reference):
    orbit = [_d4(reference, i) for i in range(8)]
    H, W = reference.shape
    bank, visited = [], set()
    for h in range(2, H+1):
        for w in range(2, W+1):
            if (h, w) == (H, W):
                continue
            for r in range(H-h+1):
                for c in range(W-w+1):
                    patch = reference[r:r+h, c:c+w]
                    key = _key(patch)
                    if key in visited or patch.sum() < 2 or patch.all():
                        continue
                    visited.add(key)
                    if len(_matches(orbit, patch)) == 1:
                        bank.append((r, c, h, w))
    return bank


def _reference(rng):
    H, W = int(rng.integers(4, 8)), int(rng.integers(4, 8))
    scattered = bool(rng.integers(2))
    gaps = scattered and rng.random() < .65
    for _ in range(120):
        if not scattered:
            ref = _connected_mask(rng, H, W, float(rng.uniform(.3, .8)))
        else:
            ref = rng.random((H, W)) < float(rng.uniform(.28, .7))
            if gaps:
                if H >= 5:
                    ref[int(rng.integers(1, H-1)), :] = False
                if W >= 5:
                    ref[:, int(rng.integers(1, W-1))] = False
            # Tight support. Preserve empty bands, but provide physical-step cues.
            ref[0, :2] = ref[-1, -2:] = True
            ref[:2, 0] = ref[-2:, -1] = True
        if len({_key(_d4(ref, i)) for i in range(8)}) != 8:
            continue
        bank = _crop_bank(ref)
        areas = sorted({h*w for _, _, h, w in bank})
        if len(areas) >= 3 and len(bank) >= 12:
            cut = areas[len(areas)//2]
            large = [p for p in bank if p[2]*p[3] >= cut]
            small = [p for p in bank if p[2]*p[3] < cut]
            if len(large) >= 5 and small:
                return ref, large, small
    # A bounded procedural retry rather than a memorized parent image.
    # This fallback remains random and has the already sampled dimensions.
    for _ in range(120):
        ref = rng.random((H, W)) < .5
        ref[0, 0] = ref[-1, -1] = True
        if len({_key(_d4(ref, i)) for i in range(8)}) != 8:
            continue
        bank = _crop_bank(ref)
        areas = sorted({h*w for _, _, h, w in bank})
        if len(areas) >= 3 and len(bank) >= 12:
            cut = areas[len(areas)//2]
            large = [p for p in bank if p[2]*p[3] >= cut]
            small = [p for p in bank if p[2]*p[3] < cut]
            if len(large) >= 5 and small:
                return ref, large, small
    raise RuntimeError("bounded asymmetric atlas construction failed")


def _window(reference, crop, orientation):
    r, c, h, w = crop
    support = np.zeros_like(reference)
    support[r:r+h, c:c+w] = True
    row, col, bottom, right = _bbox(_d4(support, orientation))
    layout = _d4(reference, orientation)
    return layout, layout[row:bottom, col:right].copy(), (row, col)


def _positions(dims, canvas, glyph_rect):
    h, w = dims
    rows, cols = np.meshgrid(np.arange(canvas[0]-h+1),
                             np.arange(canvas[1]-w+1), indexing='ij')
    points = np.column_stack((rows.ravel(), cols.ravel()))
    tr, tc, tb, te = glyph_rect
    valid = ((points[:, 0]+h+1 <= tr) | (tb+1 <= points[:, 0])
             | (points[:, 1]+w+1 <= tc) | (te+1 <= points[:, 1]))
    return points[valid]


def _pack(rng, ref, layout, patch, anchor, glyph, colors):
    gh, gw = glyph.shape
    ah, aw = layout.shape[0]*gh, layout.shape[1]*gw
    atlas_dims = tuple(2*s-1 for s in ref.shape)
    frame_dims = tuple(2*s+1 for s in patch.shape)
    min_h, min_w = max(ah, atlas_dims[0], frame_dims[0]), max(aw, atlas_dims[1], frame_dims[1])
    for attempt in range(32):
        canvas = ((int(rng.integers(min_h, 31)), int(rng.integers(min_w, 31)))
                  if attempt < 16 else (30, 30))
        top, left = int(rng.integers(canvas[0]-ah+1)), int(rng.integers(canvas[1]-aw+1))
        tr, tc = top+anchor[0]*gh, left+anchor[1]*gw
        glyph_rect = (tr, tc, tr+gh, tc+gw)
        atlas_places = _positions(atlas_dims, canvas, glyph_rect)
        frame_places = _positions(frame_dims, canvas, glyph_rect)
        if not len(atlas_places) or not len(frame_places):
            continue
        selected = None
        for j in rng.permutation(len(atlas_places))[:160]:
            pr, pc = map(int, atlas_places[j])
            valid = ((frame_places[:, 0]+frame_dims[0]+1 <= pr)
                     | (pr+atlas_dims[0]+1 <= frame_places[:, 0])
                     | (frame_places[:, 1]+frame_dims[1]+1 <= pc)
                     | (pc+atlas_dims[1]+1 <= frame_places[:, 1]))
            possible = frame_places[valid]
            if len(possible):
                fr, fc = map(int, possible[int(rng.integers(len(possible)))])
                selected = pr, pc, fr, fc
                break
        if selected is None:
            continue
        pr, pc, fr, fc = selected
        bg, marker, ink, frame = colors
        inp = np.full(canvas, bg, dtype=np.int8)
        rr, cc = np.nonzero(ref)
        inp[pr+2*rr, pc+2*cc] = marker
        fh, fw = frame_dims
        inp[fr, fc:fc+fw] = inp[fr+fh-1, fc:fc+fw] = frame
        inp[fr:fr+fh, fc] = inp[fr:fr+fh, fc+fw-1] = frame
        rr, cc = np.nonzero(patch)
        inp[fr+1+2*rr, fc+1+2*cc] = marker
        # The local exceptional coordinate is calculated from the full anchor.
        offsets = _matches([layout], patch)
        if len(offsets) != 1:
            raise AssertionError("a unique D4 crop must also have a unique local offset")
        _, r, c = offsets[0]
        local = anchor[0]-r, anchor[1]-c
        inp[fr+1+2*local[0], fc+1+2*local[1]] = ink
        rr, cc = np.nonzero(glyph)
        inp[tr+rr, tc+cc] = ink
        vals, counts = np.unique(inp, return_counts=True)
        if counts[np.flatnonzero(vals == bg)[0]] <= max(counts[vals != bg]):
            continue
        out = np.full_like(inp, bg)
        for u, v in np.argwhere(layout):
            out[top+u*gh+rr, left+v*gw+cc] = marker if (u, v) == anchor else ink
        return {'input': inp, 'output': out}
    return None


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10),
                                    ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(f"illegal {name}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    reference, large, small = _reference(rng)
    stable_palette = rng.random() < .25
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    puzzle, seen = {'train': [], 'test': []}, set()
    previous_anchor = previous_dims = None
    for split, count, bank in (('train', nt, large), ('test', nq, small)):
        for index in range(count):
            teaching = split == 'train' and index < 2
            for attempt in range(100):
                orientation = (int(rng.choice((1, 3))) if teaching and index == 0
                               else int(rng.integers(4, 8)) if teaching else int(rng.integers(8)))
                crop = _pick(rng, bank)
                layout, patch, offset = _window(reference, crop, orientation)
                if teaching and index == 0 and offset == (0, 0):
                    continue
                points = [tuple(map(int, p)) for p in np.argwhere(patch)]
                candidates = []
                for local in points:
                    anchor = offset[0]+local[0], offset[1]+local[1]
                    if teaching and index == 0 and anchor == ((layout.shape[0]-1)//2, (layout.shape[1]-1)//2):
                        continue
                    if teaching and index == 1 and (anchor[0] == previous_anchor[0] or anchor[1] == previous_anchor[1]):
                        continue
                    candidates.append(anchor)
                if not candidates:
                    continue
                anchor = _pick(rng, candidates)
                options = [(h, w) for h in range(2, min(4, 24//layout.shape[0])+1)
                           for w in range(2, min(4, 24//layout.shape[1])+1)
                           if (not teaching or index != 0 or h != w)
                           and (not teaching or index != 1 or
                                (h != previous_dims[0] and w != previous_dims[1]))]
                dims = _pick(rng, options)
                glyph = _connected_mask(rng, *dims, float(rng.uniform(.35, 1.0)))
                if teaching and glyph.all():
                    # A single corner notch is connected, tight, and nonsolid.
                    glyph[:] = True
                    glyph[_pick(rng, ((0, 0), (0, dims[1]-1), (dims[0]-1, 0),
                                      (dims[0]-1, dims[1]-1)))] = False
                colors = palette if stable_palette else tuple(map(int, rng.choice(10, 4, replace=False)))
                pair = _pack(rng, reference, layout, patch, anchor, glyph, colors)
                if pair is None or _key(pair['input']) in seen:
                    continue
                seen.add(_key(pair['input']))
                puzzle[split].append(pair)
                if teaching and index == 0:
                    previous_anchor, previous_dims = anchor, dims
                break
            else:
                raise RuntimeError("bounded distinct pair construction failed")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check returned pixels, semantic invariants, and demonstration coverage."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = np.asarray(pair['input']), np.asarray(pair['output'])
            assert out.ndim == 2 and np.issubdtype(out.dtype, np.integer)
            assert all(1 <= n <= 30 for n in out.shape) and np.all((out >= 0) & (out <= 9))
            info = _parse(inp)
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            assert np.array_equal(_answer(info), out)
            bg, marker, ink, frame = info['roles']
            population, glyph_size = int(info['layout'].sum()), int(info['glyph'].sum())
            assert np.count_nonzero(out != bg) == population*glyph_size
            assert np.count_nonzero(out == marker) == glyph_size
            assert np.count_nonzero(out == ink) == (population-1)*glyph_size
            assert not np.any(out == frame)
            assert int(info['patch'].sum()) < population
            assert not np.array_equal(_answer(info, crop_only=True), out), "A must change the answer"
            assert not np.array_equal(_answer(info, invert=False), out), "B's inversion must change the answer"
            infos[split].append(info)
    rows = infos['train'] + infos['test']
    assert all(np.array_equal(rows[0]['atlas'], m['atlas']) for m in rows)
    first, second = infos['train'][:2]
    assert first['orientation'] in (1, 3) and second['orientation'] >= 4
    assert first['offset'] != (0, 0)
    assert first['anchor'] != tuple((s-1)//2 for s in first['layout'].shape)
    assert all(first['anchor'][axis] != second['anchor'][axis] for axis in (0, 1))
    assert all(first['glyph'].shape[axis] != second['glyph'].shape[axis] for axis in (0, 1))
    assert first['glyph'].shape[0] != first['glyph'].shape[1]
    assert not first['glyph'].all() and not second['glyph'].all()
    assert not np.array_equal(_answer(first, local=True), puzzle['train'][0]['output'])
    for m, pair in zip(infos['train'][:2], puzzle['train'][:2]):
        assert not np.array_equal(_answer(m, solid=True), pair['output'])
    assert max(m['patch'].size for m in infos['test']) < min(m['patch'].size for m in infos['train'])


def _signature(puzzle):
    return tuple((split, _key(p['input']), _key(p['output']))
                 for split in ('train', 'test') for p in puzzle[split])


def diversity_metrics(puzzles):
    """Small JSON-friendly measurements obtained by parsing actual input grids."""
    rows = []
    for puzzle in puzzles:
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                m = _parse(pair['input'])
                bg = m['roles'][0]
                atlas = m['atlas']
                patch = m['patch']
                pr, pc, pb, pe = m['atlas_rect']
                fr, fc, fb, fe = m['frame_rect']
                rows.append(dict(split=split, canvas=list(m['shape']),
                    atlas_shape=list(atlas.shape), atlas_population=int(atlas.sum()),
                    atlas_components=len(_components(atlas)),
                    empty_rows=int(np.count_nonzero(~atlas.any(axis=1))),
                    empty_cols=int(np.count_nonzero(~atlas.any(axis=0))),
                    crop_shape=list(patch.shape), crop_area=int(patch.size),
                    crop_coverage=float(patch.size/atlas.size),
                    empty_crop_edge=bool(not patch[0].any() or not patch[-1].any()
                                         or not patch[:, 0].any() or not patch[:, -1].any()),
                    orientation=m['orientation'], offset=list(m['offset']),
                    anchor=list(m['anchor']), glyph_shape=list(m['glyph'].shape),
                    glyph_population=int(m['glyph'].sum()), roles=list(m['roles']),
                    old_atlas_repainted=int(np.count_nonzero(pair['output'][pr:pb:2, pc:pe:2][atlas] != bg)),
                    old_frame_repainted=int(np.count_nonzero(pair['output'][fr:fb, fc:fe][pair['input'][fr:fb, fc:fe] == m['roles'][3]] != bg)),
                    output_touches_edge=bool(m['top'] == 0 or m['left'] == 0
                        or m['top']+m['layout'].shape[0]*m['glyph'].shape[0] == m['shape'][0]
                        or m['left']+m['layout'].shape[1]*m['glyph'].shape[1] == m['shape'][1])))
    return rows


def self_test():
    """Bounded 42-seed structural, semantic, count, and counterfactual checks."""
    puzzles = []
    for seed in range(42):
        nt = 2+seed % 9
        nq = 1+seed % min(4, nt-1)
        puzzles.append(generate_puzzle(seed, nt, nq))
    assert _signature(puzzles[0]) == _signature(generate_puzzle(0, 2, 1))
    assert len({_signature(p) for p in puzzles}) == 42
    refs = [_parse(p['train'][0]['input'])['atlas'] for p in puzzles]
    assert len({min(_key(_d4(ref, i)) for i in range(8)) for ref in refs}) >= 40
    rows = diversity_metrics(puzzles)
    assert len({tuple(r['atlas_shape']) for r in rows}) >= 10
    assert len({tuple(r['canvas']) for r in rows}) >= 40
    assert len({tuple(r['crop_shape']) for r in rows}) >= 15
    assert len({tuple(r['anchor']) for r in rows}) >= 20
    assert len({tuple(r['glyph_shape']) for r in rows}) >= 8
    assert len({r['atlas_population'] for r in rows}) >= 12
    assert {r['orientation'] for r in rows} == set(range(8))
    assert set(v for r in rows for v in r['roles']) == set(range(10))
    assert any(r['atlas_components'] > 1 for r in rows)
    assert any(r['empty_rows'] for r in rows) and any(r['empty_cols'] for r in rows)
    assert any(r['empty_crop_edge'] for r in rows)
    assert any(r['old_atlas_repainted'] for r in rows)
    assert any(r['old_frame_repainted'] for r in rows)
    assert any(r['output_touches_edge'] for r in rows)
    assert min(r['crop_coverage'] for r in rows) < .4
    assert max(r['crop_coverage'] for r in rows) > .75
    masks = {_key(_parse(pair['input'])['glyph']) for p in puzzles for s in p for pair in p[s]}
    assert len(masks) >= 35
    # Exercise all legal counts on one seed; the host also exercises these.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(91, nt, nq)
            assert (len(p['train']), len(p['test'])) == (nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                    (True, 1), (4.0, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
           for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupted answer accepted")
    # Swapping only the anchor color with another observed occupied point leaves
    # exact matching unchanged, but translates and recolors the whole assembly.
    counterfactual_checked = False
    for puzzle in puzzles:
        if counterfactual_checked:
            break
        for pair in puzzle['train']:
            m = _parse(pair['input'])
            fr, fc, _, _ = m['frame_rect']
            old = m['local_anchor']
            for r, c in np.argwhere(m['patch']):
                new = int(r), int(c)
                if new == old:
                    continue
                changed = pair['input'].copy()
                a = fr+1+2*old[0], fc+1+2*old[1]
                b = fr+1+2*new[0], fc+1+2*new[1]
                changed[a], changed[b] = int(changed[b]), int(changed[a])
                try:
                    info = _parse(changed)
                except ValueError:
                    continue  # Only counterfactuals that still fit the canvas.
                assert info['orientation'] == m['orientation'] and info['offset'] == m['offset']
                assert np.array_equal(info['patch'], m['patch'])
                assert not np.array_equal(solve(changed), pair['output'])
                counterfactual_checked = True
                break
            if counterfactual_checked:
                break
    assert counterfactual_checked


if __name__ == '__main__':
    self_test()
    print('self_test passed')
