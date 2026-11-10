"""Anchored elastic stamp tables. All semantic data are visible in the input.

Only NumPy and the standard library are used. No parent code, stored grids,
seed state, or training examples are consulted by solve().
"""

from bisect import bisect_right
from itertools import permutations
import numpy as np


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((a >= 0) & (a <= 9))
    return a


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        part, stack = [p], [p]
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    part.append(q)
                    stack.append(q)
        result.append(part)
    return result


def _apart(a, b):
    """Half-open boxes with at least one intervening row or column."""
    return (a[2]+1 <= b[0] or b[2]+1 <= a[0]
            or a[3]+1 <= b[1] or b[3]+1 <= a[1])


def _card(grid, component, bg, sep):
    rr, cc = np.array(component).T
    t, l, b, r = int(rr.min()), int(cc.min()), int(rr.max())+1, int(cc.max())+1
    h, w = b-t-4, r-l-2
    assert 2 <= h <= 4 and 2 <= w <= 4
    tile = grid[t:b, l:r]
    frame = np.zeros(tile.shape, bool)
    frame[[0, h+1, h+3], :] = True
    frame[:, [0, w+1]] = True
    assert np.all(tile[frame] == sep)
    glyph = tile[1:h+1, 1:w+1]
    colors = np.unique(glyph[glyph != bg])
    assert len(colors) == 1 and colors[0] != sep
    source = int(colors[0])
    mask = glyph == source
    assert mask[0].any() and mask[-1].any()
    assert mask[:, 0].any() and mask[:, -1].any()
    swatch = tile[h+2, 1:w+1]
    values = swatch[swatch != bg]
    assert len(values) == 1 and values[0] != sep
    expected = np.full(tile.shape, bg, dtype=int)
    expected[frame] = sep
    expected[1:h+1, 1:w+1] = glyph
    expected[h+2, 1:w+1] = swatch
    assert np.array_equal(tile, expected)
    assert len(component) == int(frame.sum())
    return source, dict(mask=mask.copy(), target=int(values[0]),
                        pos=(t+1, l+1), box=(t, l, b, r))


def _decode_with_frame(grid, bg, sep):
    parts = _components(grid == sep)
    large = [p for p in parts if len(p) >= 18]
    assert 2 <= len(large) <= 4
    cards = {}
    for part in large:
        source, card = _card(grid, part, bg, sep)
        assert source not in cards
        cards[source] = card
    boxes = [card['box'] for card in cards.values()]
    assert all(_apart(a, b) for i, a in enumerate(boxes) for b in boxes[i+1:])
    remaining = grid.copy()
    for t, l, b, r in boxes:
        remaining[t:b, l:r] = bg
    ring = list(map(tuple, np.argwhere(remaining == sep)))
    assert len(ring) == 4
    center = np.mean(ring, axis=0)
    assert np.all(center == np.floor(center))
    ar, ac = map(int, center)
    assert set(ring) == {(ar-1, ac), (ar+1, ac), (ar, ac-1), (ar, ac+1)}
    assert int(remaining[ar, ac]) in cards
    point_mask = (remaining != bg) & (remaining != sep)
    points = np.argwhere(point_mask)
    assert len(points) >= 2
    assert set(map(int, remaining[point_mask])) == set(cards)
    pr, pc = map(int, points.min(axis=0))
    er, ec = map(int, points.max(axis=0))
    assert np.all((points[:, 0]-pr) % 2 == 0)
    assert np.all((points[:, 1]-pc) % 2 == 0)
    rows, cols = (er-pr)//2+1, (ec-pc)//2+1
    assert 1 <= rows <= 7 and 1 <= cols <= 7
    mapping = np.full((rows, cols), -1, dtype=np.int8)
    for r, c in points:
        mapping[(int(r)-pr)//2, (int(c)-pc)//2] = remaining[r, c]
    anchor = (ar-pr)//2, (ac-pc)//2
    # The one-cell map apron reserves room for any legal halo location.
    map_box = (pr-1, pc-1, er+2, ec+2)
    assert 0 <= map_box[0] and 0 <= map_box[1]
    assert map_box[2] <= grid.shape[0] and map_box[3] <= grid.shape[1]
    assert all(_apart(box, map_box) for box in boxes)
    meta = dict(bg=bg, sep=sep, cards=cards, mapping=mapping,
                anchor=anchor, map_pos=(pr, pc), map_box=map_box,
                shape=grid.shape)
    rh, cw, rp, cp, origin = _geometry(meta)
    assert 0 <= origin[0] and origin[0]+rp[-1] <= grid.shape[0]
    assert 0 <= origin[1] and origin[1]+cp[-1] <= grid.shape[1]
    return meta


def _decode(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    assert all(np.all(edge == bg) for edge in
               (grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    candidates = []
    for sep in np.unique(grid):
        if sep == bg:
            continue
        try:
            candidates.append(_decode_with_frame(grid, bg, int(sep)))
        except (AssertionError, IndexError, ValueError):
            continue
    assert len(candidates) == 1, 'expected one valid card/map interpretation'
    return candidates[0]


def _geometry(meta, mode='correct'):
    mapping, cards = meta['mapping'], meta['cards']
    ar, ac = meta['anchor']
    if mode == 'first_anchor':
        ar, ac = map(int, np.argwhere(mapping >= 0)[0])
    source = int(mapping[ar, ac])
    h0, w0 = cards[source]['mask'].shape
    rh = [h0] * mapping.shape[0]
    cw = [w0] * mapping.shape[1]
    for r, c in np.argwhere(mapping >= 0):
        h, w = cards[int(mapping[r, c])]['mask'].shape
        rh[r], cw[c] = max(rh[r], h), max(cw[c], w)
    if mode == 'uniform':
        rh, cw = [h0]*len(rh), [w0]*len(cw)
    elif mode == 'global':
        rh, cw = [max(rh)]*len(rh), [max(cw)]*len(cw)
    elif mode == 'padded':
        rh, cw = [v+4 for v in rh], [v+2 for v in cw]
    elif mode in ('occupied_only', 'occupied_only_h', 'occupied_only_w'):
        # A plausible alternative uses the anchor's minimum only for empty
        # bands. Training must show that occupied smaller bands retain it too.
        if mode != 'occupied_only_w':
            rh = [max([cards[int(v)]['mask'].shape[0] for v in row if v >= 0],
                      default=h0) for row in mapping]
        if mode != 'occupied_only_h':
            cw = [max([cards[int(v)]['mask'].shape[1] for v in col if v >= 0],
                      default=w0) for col in mapping.T]
    elif mode in ('compress', 'compress_h', 'compress_w'):
        if mode != 'compress_w':
            rh = [v if np.any(mapping[r] >= 0) else 0 for r, v in enumerate(rh)]
        if mode != 'compress_h':
            cw = [v if np.any(mapping[:, c] >= 0) else 0 for c, v in enumerate(cw)]
    rp, cp = [0], [0]
    for v in rh:
        rp.append(rp[-1]+v)
    for v in cw:
        cp.append(cp[-1]+v)
    t, l = cards[source]['pos']
    origin = (t-rp[ar], l-cp[ac])
    if mode == 'map_origin':
        origin = meta['map_pos']
    elif mode == 'center':
        origin = (origin[0]-(rh[ar]-h0)//2, origin[1]-(cw[ac]-w0)//2)
    return rh, cw, rp, cp, origin


def _render(meta, mode='correct'):
    """Forward copy loop, also used for explicitly named wrong-rule probes."""
    result = np.full(meta['shape'], meta['bg'], dtype=np.int8)
    rh, cw, rp, cp, (top, left) = _geometry(meta, mode)
    mapping, cards = meta['mapping'], meta['cards']
    anchor_source = int(mapping[meta['anchor']])
    for r, c in np.argwhere(mapping >= 0):
        source = int(mapping[r, c])
        card = cards[anchor_source if mode == 'one_template' else source]
        mask = card['mask']
        if mode == 'solid':
            mask = np.ones_like(mask)
        elif mode == 'flip_h':
            mask = np.fliplr(mask)
        elif mode == 'flip_v':
            mask = np.flipud(mask)
        color = source if mode == 'no_key' else card['target']
        dr = (rh[r]-mask.shape[0])//2 if mode == 'center' else 0
        dc = (cw[c]-mask.shape[1])//2 if mode == 'center' else 0
        rr, cc = np.nonzero(mask)
        rr, cc = top+rp[r]+dr+rr, left+cp[c]+dc+cc
        # Wrong hypotheses may overlap or leave the canvas. Only the correct
        # rule is required to be a legal tiling; probes paint their in-bounds part.
        valid = (rr >= 0) & (rr < result.shape[0]) & (cc >= 0) & (cc < result.shape[1])
        result[rr[valid], cc[valid]] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using only the grid's cards, keys, coordinates, and anchor halo."""
    return _render(_decode(input_grid))


def _inverse_check(meta, output):
    """Check every pixel by inverse cell lookup rather than repeating stamping."""
    rh, cw, rp, cp, (top, left) = _geometry(meta)
    mapping, cards, bg = meta['mapping'], meta['cards'], meta['bg']
    for r in range(output.shape[0]):
        for c in range(output.shape[1]):
            y, x = r-top, c-left
            expected = bg
            if 0 <= y < rp[-1] and 0 <= x < cp[-1]:
                mr, mc = bisect_right(rp, y)-1, bisect_right(cp, x)-1
                source = int(mapping[mr, mc])
                if source >= 0:
                    card = cards[source]
                    dy, dx = y-rp[mr], x-cp[mc]
                    mask = card['mask']
                    if dy < mask.shape[0] and dx < mask.shape[1] and mask[dy, dx]:
                        expected = card['target']
            assert int(output[r, c]) == expected, f'wrong output pixel {(r, c)}'
    population = sum(int(cards[int(mapping[r, c])]['mask'].sum())
                     for r, c in np.argwhere(mapping >= 0))
    assert int(np.sum(output != bg)) == population
    assert set(map(int, np.unique(output))) <= {bg, *(d['target'] for d in cards.values())}
    ar, ac = meta['anchor']
    card = cards[int(mapping[ar, ac])]
    assert (top+rp[ar], left+cp[ac]) == card['pos']
    for r in range(len(rh)):
        assert rh[r] == max([card['mask'].shape[0]] +
                           [cards[int(v)]['mask'].shape[0] for v in mapping[r] if v >= 0])
    for c in range(len(cw)):
        assert cw[c] == max([card['mask'].shape[1]] +
                           [cards[int(v)]['mask'].shape[1] for v in mapping[:, c] if v >= 0])


_SHORTCUTS = ('one_template', 'no_key', 'map_origin', 'first_anchor',
              'uniform', 'global', 'padded', 'compress', 'compress_h',
              'compress_w', 'occupied_only', 'occupied_only_h',
              'occupied_only_w', 'solid',
              'flip_h', 'flip_v', 'center')


def _training_coverage(metas, outputs):
    assert len({m['anchor'][0] for m in metas}) >= 2
    assert len({m['anchor'][1] for m in metas}) >= 2
    assert any(np.any(np.all(m['mapping'] == -1, axis=0)) and
               np.any(np.all(m['mapping'] == -1, axis=1)) for m in metas)
    assert any(len(set(_geometry(m)[0])) > 1 and len(set(_geometry(m)[1])) > 1 for m in metas)
    for mode in _SHORTCUTS:
        assert any(not np.array_equal(out, _render(m, mode))
                   for m, out in zip(metas, outputs)), f'undistinguished shortcut: {mode}'
    # Direct pitch-two step witnesses on both axes.
    assert all(any(np.any((m['mapping'][:-1] >= 0) & (m['mapping'][1:] >= 0))
                   if axis == 0 else
                   np.any((m['mapping'][:, :-1] >= 0) & (m['mapping'][:, 1:] >= 0))
                   for m in metas) for axis in (0, 1))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, metadata, outputs = set(), {'train': [], 'test': []}, []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            meta = _decode(inp)
            _inverse_check(meta, out)
            assert np.array_equal(solve(inp), out)
            metadata[split].append(meta)
            if split == 'train':
                outputs.append(out)
    _training_coverage(metadata['train'], outputs)
    assert all(max(m['mapping'].shape) <= 4 for m in metadata['train'])
    assert all(5 in m['mapping'].shape and len(m['cards']) == 4 for m in metadata['test'])


def _pick(rng, choices):
    return choices[int(rng.integers(len(choices)))]


def _transform_mask(a, turns, flip):
    return np.rot90(np.fliplr(a) if flip else a, turns).copy()


def _sample_mask(rng, h, w, asymmetric=False):
    if asymmetric:
        mask = np.ones((h, w), bool)
        mask[_pick(rng, [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)])] = False
        return mask
    style = int(rng.integers(6))
    if style == 0:
        return np.ones((h, w), bool)
    if style == 1:
        mask = np.ones((h, w), bool)
        mask[1:-1, 1:-1] = False
        return mask
    if style == 2:
        mask = np.zeros((h, w), bool)
        mask[int(rng.integers(h)), :] = True
        mask[:, int(rng.integers(w))] = True
        return mask
    if style == 3:
        rr, cc = np.indices((h, w))
        mask = (rr+cc) % 2 == int(rng.integers(2))
    else:
        mask = rng.random((h, w)) < float(rng.uniform(.12, .88))
    # Tight bounds, including for disconnected/sparse masks.
    mask[0, int(rng.integers(w))] = True
    mask[-1, int(rng.integers(w))] = True
    mask[int(rng.integers(h)), 0] = True
    mask[int(rng.integers(h)), -1] = True
    return mask


def _random_map(rng, rows, cols, k):
    eligible = np.ones((rows, cols), bool)
    if rows >= 4 and rng.random() < .35:
        eligible[int(rng.integers(1, rows-1)), :] = False
    if cols >= 4 and rng.random() < .35:
        eligible[:, int(rng.integers(1, cols-1))] = False
    available = list(map(tuple, np.argwhere(eligible)))
    if len(available) < k:
        return None
    mandatory = {(0, int(_pick(rng, np.flatnonzero(eligible[0])))),
                 (rows-1, int(_pick(rng, np.flatnonzero(eligible[-1])))),
                 (int(_pick(rng, np.flatnonzero(eligible[:, 0]))), 0),
                 (int(_pick(rng, np.flatnonzero(eligible[:, -1]))), cols-1)}
    style = int(rng.integers(5))
    if style == 0:
        order = list(map(tuple, rng.permutation(available)))
    elif style == 1:
        order = sorted(available, key=lambda p: (p[0], p[1] if p[0] % 2 == 0 else -p[1]))
    elif style == 2:
        center = _pick(rng, available)
        order = sorted(available, key=lambda p: (abs(p[0]-center[0])+abs(p[1]-center[1]), p))
    elif style == 3:
        order = sorted(available, key=lambda p: (sum(p) % 2, sum(p), p))
    else:
        order = sorted(available, key=lambda p: (min(p[0], rows-1-p[0], p[1], cols-1-p[1]), p))
    count = int(rng.integers(max(k, len(mandatory)), len(available)+1))
    occupied = list(sorted(mandatory))
    occupied += [p for p in order if p not in mandatory][:count-len(mandatory)]
    rng.shuffle(occupied)
    weights = rng.dirichlet(np.full(k, float(rng.uniform(.2, 2.0))))
    labels = list(map(int, rng.permutation(k)))
    labels += list(map(int, rng.choice(k, count-k, p=weights)))
    mapping = np.full((rows, cols), -1, dtype=np.int8)
    for p, identity in zip(occupied, labels):
        mapping[p] = identity
    return mapping


def _command_data(rng, teaching, test, extension_axis):
    if teaching == 0:
        dims = [(2, 2), (2, int(rng.integers(3, 5))),
                (int(rng.integers(3, 5)), 2)]
        # Every layout retains independent height/width maxima, but the
        # demonstration is not restricted to one five-command silhouette.
        mapping = np.array(_pick(rng, [
            [[0, 1, -1], [2, 0, 0]],
            [[0, 1, 0], [2, 0, 0]],
            [[0, 1, 0], [2, -1, -1], [-1, 0, 0]],
            [[0, 1, -1], [2, 0, 0], [-1, 0, -1]],
            [[0, 1, -1, -1], [2, 0, 0, 0]],
        ]), dtype=np.int8)
        mapping = _transform_mask(mapping, int(rng.integers(4)), bool(rng.integers(2)))
    elif teaching == 1:
        dims = [(int(rng.integers(3, 5)), int(rng.integers(3, 5))),
                (2, 2), (int(rng.integers(2, 5)), int(rng.integers(2, 5)))]
        # Identity 1 alone occupies an internal row and internal column.
        # Both have to retain the larger anchor's minimum, in addition to
        # the completely empty row/column inserted below. Internal witnesses
        # still change visible placements after any dihedral map transform.
        small = np.array([[0, 1, 2], [-1, 1, -1], [0, -1, 0]], dtype=np.int8)
        for point in ((1, 0), (1, 2), (2, 1)):
            if rng.random() < .35:
                small[point] = 1
        mapping = np.full((4, 4), -1, dtype=np.int8)
        mapping[np.ix_([0, 1, 3], [0, 1, 3])] = small
        mapping = _transform_mask(mapping, int(rng.integers(4)), bool(rng.integers(2)))
    else:
        k = 4 if test else int(rng.integers(2, 5))
        dims = [(int(rng.integers(2, 5)), int(rng.integers(2, 5))) for _ in range(k)]
        if test:
            shape = (5, int(rng.integers(2, 6))) if extension_axis == 0 else (int(rng.integers(2, 6)), 5)
        else:
            shape = _pick(rng, [(a, b) for a in range(1, 5) for b in range(1, 5) if a*b >= k])
        mapping = _random_map(rng, *shape, k)
        if mapping is None:
            return None
    masks = [_sample_mask(rng, *d, asymmetric=teaching == 0 and i == 1)
             for i, d in enumerate(dims)]
    if teaching is None and rng.random() < .18:
        matches = [(a, b) for a in range(len(dims)) for b in range(a+1, len(dims)) if dims[a] == dims[b]]
        if matches:
            a, b = _pick(rng, matches)
            masks[b] = masks[a].copy()
    return mapping, masks


def _pack(rng, mapping, masks, teaching, previous_anchor, scatter):
    """Construct in a 28×28 core, reserving a background border after cropping."""
    k = len(masks)
    anchors = list(map(tuple, np.argwhere(mapping == 0 if teaching is not None else mapping >= 0)))
    first = tuple(map(int, np.argwhere(mapping >= 0)[0]))
    if teaching == 0:
        anchors = [a for a in anchors if a != first]
    if teaching == 1:
        anchors = [a for a in anchors if a[0] != previous_anchor[0] and a[1] != previous_anchor[1]]
    rng.shuffle(anchors)
    orders = list(permutations(range(k)))
    rng.shuffle(orders)
    gh = max(m.shape[0]+4 for m in masks)
    gw = sum(m.shape[1]+2 for m in masks)+k-1
    mh, mw = 2*mapping.shape[0]+1, 2*mapping.shape[1]+1
    chosen = None
    for anchor in anchors:
        source = int(mapping[anchor])
        provisional = dict(mapping=mapping, cards={i: dict(mask=m, pos=(0, 0)) for i, m in enumerate(masks)}, anchor=anchor)
        rh, cw, rp, cp, _ = _geometry(provisional)
        ys = [y for y in range(28-gh+1) if 0 <= y+1-rp[anchor[0]]
              and y+1-rp[anchor[0]]+rp[-1] <= 28
              and (y >= mh+1 or y+gh+1+mh <= 28)]
        if not ys:
            continue
        for order in orders:
            offsets, offset = {}, 0
            for identity in order:
                offsets[identity] = offset
                offset += masks[identity].shape[1]+3
            ax = offsets[source]+1
            xs = [x for x in range(28-gw+1) if 0 <= x+ax-cp[anchor[1]]
                  and x+ax-cp[anchor[1]]+cp[-1] <= 28]
            if xs:
                chosen = anchor, order, offsets, int(_pick(rng, ys)), int(_pick(rng, xs)), rp, cp
                break
        if chosen is not None:
            break
    if chosen is None:
        return None
    anchor, order, offsets, y, x, rp, cp = chosen
    source = int(mapping[anchor])
    positions = [(y, x+offsets[i]) for i in range(k)]
    map_ys = list(range(0, y-mh)) + list(range(y+gh+1, 28-mh+1))
    my, mx = int(_pick(rng, map_ys)), int(rng.integers(29-mw))
    map_box = (my, mx, my+mh, mx+mw)
    boxes = [(t, l, t+m.shape[0]+4, l+m.shape[1]+2) for (t, l), m in zip(positions, masks)]
    # Scatter nonanchor cards without altering the output's anchor position.
    for i in map(int, rng.permutation(k)):
        if i == source or rng.random() >= scatter:
            continue
        h, w = masks[i].shape[0]+4, masks[i].shape[1]+2
        options = [(t, l) for t in range(29-h) for l in range(29-w)
                   if _apart((t, l, t+h, l+w), map_box)
                   and all(j == i or _apart((t, l, t+h, l+w), box) for j, box in enumerate(boxes))]
        if options:
            t, l = _pick(rng, options)
            positions[i], boxes[i] = (t, l), (t, l, t+h, l+w)
    top, left = positions[source][0]+1-rp[anchor[0]], positions[source][1]+1-cp[anchor[1]]
    return positions, boxes, map_box, anchor, (top, left), rp, cp


def _make_pair(rng, teaching, test, axis, previous_anchor, scatter):
    data = _command_data(rng, teaching, test, axis)
    if data is None:
        return None
    mapping, masks = data
    packed = _pack(rng, mapping, masks, teaching, previous_anchor, scatter)
    if packed is None:
        return None
    positions, boxes, map_box, anchor, origin, rp, cp = packed
    k = len(masks)
    colors = list(map(int, rng.choice(10, k+2, replace=False)))
    bg, sep, sources = colors[0], colors[1], colors[2:]
    target_pool = [c for c in range(10) if c not in (bg, sep)]
    targets = [int(_pick(rng, [c for c in target_pool if c != sources[i]]))
               if teaching is not None else int(_pick(rng, target_pool)) for i in range(k)]
    if teaching == 0 and len(set(targets)) == 1:
        targets[1] = int(_pick(rng, [c for c in target_pool if c not in (targets[0], sources[1])]))
    inp = np.full((28, 28), bg, dtype=np.int8)
    out = np.full_like(inp, bg)
    for i, ((t, l), mask) in enumerate(zip(positions, masks)):
        h, w = mask.shape
        inp[t:t+h+4, l:l+w+2] = sep
        inp[t+1:t+h+1, l+1:l+w+1] = np.where(mask, sources[i], bg)
        inp[t+h+2, l+1:l+w+1] = bg
        inp[t+h+2, l+1+int(rng.integers(w))] = targets[i]
    my, mx = map_box[:2]
    for r, c in np.argwhere(mapping >= 0):
        i = int(mapping[r, c])
        inp[my+1+2*r, mx+1+2*c] = sources[i]
        # Generated ground truth comes from command data, not solve/decode.
        rr, cc = np.nonzero(masks[i])
        out[origin[0]+rp[r]+rr, origin[1]+cp[c]+cc] = targets[i]
    ar, ac = my+1+2*anchor[0], mx+1+2*anchor[1]
    for r, c in ((ar-1, ac), (ar+1, ac), (ar, ac-1), (ar, ac+1)):
        inp[r, c] = sep
    all_boxes = boxes + [map_box, (origin[0], origin[1], origin[0]+rp[-1], origin[1]+cp[-1])]
    t = min(b[0] for b in all_boxes)
    l = min(b[1] for b in all_boxes)
    b = max(b[2] for b in all_boxes)
    r = max(b[3] for b in all_boxes)
    ch, cw = b-t, r-l
    extra_h, extra_w = int(rng.integers(2, 31-ch)), int(rng.integers(2, 31-cw))
    pt, pl = int(rng.integers(1, extra_h)), int(rng.integers(1, extra_w))
    padding = ((pt, extra_h-pt), (pl, extra_w-pl))
    return dict(input=np.pad(inp[t:b, l:r], padding, constant_values=bg),
                output=np.pad(out[t:b, l:r], padding, constant_values=bg))


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    axis, scatter = int(rng.integers(2)), float(_pick(rng, [0.0, .4, 1.0]))
    puzzle, seen, teaching_metas, teaching_outputs = {'train': [], 'test': []}, set(), [], []
    previous_anchor = None
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            teaching = index if split == 'train' and index < 2 else None
            for attempt in range(96):
                pair = _make_pair(rng, teaching, split == 'test', axis, previous_anchor, scatter)
                if pair is None or _key(pair['input']) in seen:
                    continue
                meta = _decode(pair['input'])
                if teaching == 1:
                    try:
                        _training_coverage(teaching_metas+[meta], teaching_outputs+[pair['output']])
                    except AssertionError:
                        continue
                puzzle[split].append(pair)
                seen.add(_key(pair['input']))
                if teaching is not None:
                    teaching_metas.append(meta)
                    teaching_outputs.append(pair['output'])
                    previous_anchor = meta['anchor']
                break
            else:
                raise RuntimeError('bounded construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def measure_puzzle(puzzle):
    """Actual structural measurements from returned inputs, not sampler flags."""
    records = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            m = _decode(pair['input'])
            rh, cw, rp, cp, origin = _geometry(m)
            cards, mapping = m['cards'], m['mapping']
            anchor_mask = cards[int(mapping[m['anchor']])]['mask']
            floor_rows = sum(bool(np.any(row >= 0)) and
                             max([cards[int(v)]['mask'].shape[0] for v in row if v >= 0],
                                 default=anchor_mask.shape[0]) < anchor_mask.shape[0]
                             for row in mapping)
            floor_cols = sum(bool(np.any(col >= 0)) and
                             max([cards[int(v)]['mask'].shape[1] for v in col if v >= 0],
                                 default=anchor_mask.shape[1]) < anchor_mask.shape[1]
                             for col in mapping.T)
            records.append(dict(split=split, shape=m['shape'], k=len(cards),
                map_shape=mapping.shape, anchor=m['anchor'], rows=rh, cols=cw,
                floor_rows=int(floor_rows), floor_cols=int(floor_cols),
                occupied=int(np.sum(mapping >= 0)), empty_rows=int(np.sum(np.all(mapping == -1, axis=1))),
                empty_cols=int(np.sum(np.all(mapping == -1, axis=0))),
                card_positions=tuple(d['pos'] for d in cards.values()),
                dims=tuple(d['mask'].shape for d in cards.values()),
                duplicate_masks=len({_key(d['mask']) for d in cards.values()}) < len(cards),
                shared_keys=len({d['target'] for d in cards.values()}) < len(cards),
                self_key=any(s == d['target'] for s, d in cards.items()),
                disconnected=any(len(_components(d['mask'])) > 1 for d in cards.values()),
                hollow=any(d['mask'].shape[0] >= 3 and d['mask'].shape[1] >= 3
                           and d['mask'][0].all() and d['mask'][-1].all()
                           and d['mask'][:, 0].all() and d['mask'][:, -1].all()
                           and not d['mask'].all() for d in cards.values()),
                roles=(m['bg'], m['sep']), origin=origin,
                overlap=int(np.sum((np.asarray(pair['input']) != m['bg']) &
                                   (np.asarray(pair['output']) != m['bg'])))))
    return records


def self_test():
    """Bounded structural diversity, count, consistency, and mutation checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures, shapes, maps, motifs, anchors, dimensions, roles = set(), set(), set(), set(), set(), set(), set()
    records = []
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        records.extend(measure_puzzle(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                m = _decode(pair['input'])
                shapes.add(m['shape'])
                anchors.add(m['anchor'])
                roles.add((m['bg'], m['sep']))
                canonical = np.full(m['mapping'].shape, -1, dtype=np.int8)
                for i, (s, d) in enumerate(m['cards'].items()):
                    canonical[m['mapping'] == s] = i
                    motifs.add(_key(d['mask']))
                    dimensions.add(d['mask'].shape)
                maps.add(_key(canonical))
    assert len(signatures) == 42 and len(shapes) >= 25 and len(maps) >= 80
    assert len(motifs) >= 60 and len(anchors) >= 12 and len(roles) >= 30
    assert dimensions == {(h, w) for h in range(2, 5) for w in range(2, 5)}
    assert {r['k'] for r in records} == {2, 3, 4}
    assert min(r['occupied'] for r in records) <= 4 and max(r['occupied'] for r in records) >= 15
    assert any(1 in r['map_shape'] for r in records)
    assert any(r['duplicate_masks'] for r in records)
    for prop in ('shared_keys', 'self_key', 'disconnected', 'hollow', 'overlap'):
        assert sum(bool(r[prop]) for r in records) >= 5, prop
    assert sum(bool(r['floor_rows'] and r['floor_cols']) for r in records) >= 42
    assert any(len({p[0] for p in r['card_positions']}) > 1 for r in records)
    first = generate_puzzle(0)
    again = generate_puzzle(0)
    assert all(np.array_equal(a[role], b[role]) for s in first for a, b in zip(first[s], again[s])
               for role in ('input', 'output'))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000+nt*10+nq, nt, nq)
            assert (len(p['train']), len(p['test'])) == (nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (3.5, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid counts accepted')
    corrupted = {s: [{role: a.copy() for role, a in p.items()} for p in first[s]] for s in first}
    corrupted['test'][0]['output'][0, 0] = (int(corrupted['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError('incorrect output accepted')
    inp = first['train'][0]['input'].copy()
    m = _decode(inp)
    # Literal colors never carry hidden roles. Apply a bijection to the
    # complete scene and require exactly the same transformation in that palette.
    permutation = np.array([3, 7, 0, 6, 9, 2, 5, 8, 4, 1], dtype=np.int8)
    for split in first:
        for pair in first[split]:
            assert np.array_equal(solve(permutation[pair['input']]),
                                  permutation[pair['output']])
    # A changed visible key recolors just its identity's copies; its geometry
    # and every other output pixel stay fixed, even if targets share colors.
    source = next(iter(m['cards']))
    card = m['cards'][source]
    t, l, b, r = card['box']
    new_color = next(c for c in range(10) if c not in
                     (m['bg'], m['sep'], card['target']))
    swatch = inp[b-2, l+1:r-1]
    swatch[swatch != m['bg']] = new_color
    changed = solve(inp)
    expected = first['train'][0]['output'].copy()
    _, _, rp, cp, (top, left) = _geometry(m)
    rr, cc = np.nonzero(card['mask'])
    for row, col in np.argwhere(m['mapping'] == source):
        expected[top+rp[row]+rr, left+cp[col]+cc] = new_color
    assert np.array_equal(changed, expected)
    assert np.count_nonzero(changed != first['train'][0]['output']) == (
        int(card['mask'].sum()) * int(np.sum(m['mapping'] == source)))
    inp = first['train'][0]['input'].copy()
    ar, ac = m['anchor']
    pr, pc = m['map_pos']
    inp[pr+2*ar-1, pc+2*ac] = m['bg']
    try:
        solve(inp)
    except AssertionError:
        pass
    else:
        raise AssertionError('broken anchor halo accepted')
    # Lists are legal returned-grid representations too.
    validate_generated_puzzle({s: [{k: a.tolist() for k, a in p.items()} for p in first[s]] for s in first})
    # Family members outside the sampler's five-site extent, including outputs
    # touching the top/left boundary. Construct pixels directly; solve must
    # never enforce the training/test sampling schedule on an individual input.
    masks = (np.eye(2, dtype=bool), np.array([[1, 1], [1, 0]], dtype=bool))
    for rows, cols in ((7, 1), (1, 7), (6, 6), (7, 7)):
        sample = np.zeros((30, 30), dtype=np.int8)
        positions = ((11, 11), (11, 16)) if rows == 1 else ((11, 17), (11, 22))
        for source, ((t, l), mask) in enumerate(zip(positions, masks), 1):
            sample[t:t+6, l:l+4] = 9
            sample[t+1:t+3, l+1:l+3] = np.where(mask, source, 0)
            sample[t+4, l+1:l+3] = (source+2, 0)
        anchor = ((0, 6) if rows == 1 else (6, 0) if cols == 1 else (3, 3))
        commands = {(0, 0): 2, (rows-1, cols-1): 2, anchor: 1}
        if min(rows, cols) > 1:
            commands[(0, cols-1)] = 1
            commands[(rows-1, 0)] = 2
        for (r, c), source in commands.items():
            sample[2+2*r, 2+2*c] = source
        ar, ac = 2+2*anchor[0], 2+2*anchor[1]
        for r, c in ((ar-1, ac), (ar+1, ac), (ar, ac-1), (ar, ac+1)):
            sample[r, c] = 9
        top, left = positions[0][0]+1-2*anchor[0], positions[0][1]+1-2*anchor[1]
        expected = np.zeros_like(sample)
        for (r, c), source in commands.items():
            rr, cc = np.nonzero(masks[source-1])
            expected[top+2*r+rr, left+2*c+cc] = source+2
        assert np.array_equal(solve(sample), expected)
        if cols == 1:
            assert np.any(expected[0] != 0)
        if rows == 1:
            assert np.any(expected[:, 0] != 0)


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds, all legal counts, ablations, and mutations')
