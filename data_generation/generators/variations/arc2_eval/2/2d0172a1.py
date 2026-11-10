import numpy as np


# The invariant is geometric; literal colors never encode a hidden convention.
# Layout groups are not objects: only actual enclosing components add frames.

def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    h, w = mask.shape
    for r, c in zip(*np.nonzero(mask)):
        if seen[r, c]:
            continue
        stack = [(int(r), int(c))]
        seen[r, c] = True
        cells = []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(cells)
    return result


def _holes(mask):
    padded = np.pad(mask, 1)
    outside = np.zeros(padded.shape, dtype=bool)
    outside[0, 0] = True
    stack = [(0, 0)]
    h, w = padded.shape
    while stack:
        y, x = stack.pop()
        for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
            if 0 <= yy < h and 0 <= xx < w and not padded[yy, xx] and not outside[yy, xx]:
                outside[yy, xx] = True
                stack.append((yy, xx))
    return (~padded & ~outside)[1:-1, 1:-1]


def _parse_input(grid, background=None):
    values, counts = np.unique(grid, return_counts=True)
    assert len(values) == 2
    if background is None:
        background = int(values[np.argmax(counts)])
    foreground = int(values[values != background][0])
    records = []
    for cells in _components(grid == foreground):
        coords = np.array(cells)
        lo, hi = coords.min(axis=0), coords.max(axis=0)
        mask = np.zeros(grid.shape, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        hole = _holes(mask)
        kind = 'C' if hole.any() else 'B'
        if kind == 'C':
            assert len(_components(hole)) == 1, 'contour has multiple enclosed regions'
        else:
            # Compact, connected, filled leaves; no fixed blob dimensions.
            assert mask.sum() >= np.prod(hi-lo+1) * .35
        records.append(dict(kind=kind, cells=coords, lo=lo, hi=hi,
                            mask=mask, hole=hole, area=len(cells)))
    assert any(o['kind'] == 'C' for o in records)
    labels = np.zeros(grid.shape, dtype=int)
    for i, obj in enumerate(records):
        labels[obj['mask']] = i+1
    for a, b in ((labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:]),
                 (labels[:-1, :-1], labels[1:, 1:]), (labels[:-1, 1:], labels[1:, :-1])):
        assert not np.any((a > 0) & (b > 0) & (a != b)), 'touching objects'
    parents = []
    for i, child in enumerate(records):
        containing = []
        for j, obj in enumerate(records):
            if i == j or obj['kind'] != 'C':
                continue
            inside = obj['hole'][child['cells'][:, 0], child['cells'][:, 1]]
            assert inside.all() or not inside.any(), 'crossing components'
            if inside.all():
                containing.append(j)
        parents.append(min(containing, key=lambda j: int(records[j]['hole'].sum()))
                       if containing else None)
    children = {i: [] for i in range(len(records))}
    roots = []
    for i, parent in enumerate(parents):
        (roots if parent is None else children[parent]).append(i)
    for i, obj in enumerate(records):
        assert bool(children[i]) == (obj['kind'] == 'C'), 'empty contour or enclosing blob'
    return dict(background=background, foreground=foreground, components=records,
                parents=parents, children=children, roots=roots)


def _tile(mask, leaves):
    return dict(mask=mask, leaves=leaves)


def _place(tiles, axis, offsets, gaps):
    """Compose tiles at supplied cross-axis offsets, returning translated leaves."""
    starts = []
    position = 0
    cross = 1-axis
    low = min(offsets)
    shape = [0, 0]
    for i, tile in enumerate(tiles):
        start = [0, 0]
        start[axis], start[cross] = position, offsets[i]-low
        starts.append(start)
        position += tile['mask'].shape[axis]
        if i < len(gaps):
            position += gaps[i]
        shape[cross] = max(shape[cross], start[cross]+tile['mask'].shape[cross])
    shape[axis] = position
    mask = np.zeros(shape, dtype=bool)
    leaves = []
    for tile, (y, x) in zip(tiles, starts):
        h, w = tile['mask'].shape
        mask[y:y+h, x:x+w] |= tile['mask']
        leaves.extend((key, row+y, col+x) for key, row, col in tile['leaves'])
    return _tile(mask, leaves)


def _visible_layout(info):
    """Recover a layout from whitespace cuts and aligned leaf projections.

    Try both slicing directions. If two legal decompositions disagree, reject
    the placement instead of resolving a diagonal by a center-span heuristic.
    """
    objects = info['components']
    cache = {}

    def align(tiles, axis):
        cross = 1-axis
        edges = []
        for i in range(len(tiles)):
            for j in range(i+1, len(tiles)):
                shifts = set()
                for a, ay, ax in tiles[i]['leaves']:
                    for b, by, bx in tiles[j]['leaves']:
                        if max(objects[a]['lo'][cross], objects[b]['lo'][cross]) <= min(objects[a]['hi'][cross], objects[b]['hi'][cross]):
                            shifts.add((ay, ax)[cross]-(by, bx)[cross])
                assert len(shifts) <= 1, 'competing alignment anchors'
                if shifts:
                    edges.append((i, j, shifts.pop()))
        offsets = {0: 0}
        for _ in tiles:
            for i, j, delta in edges:
                if i in offsets and j in offsets:
                    assert offsets[j] == offsets[i]+delta, 'inconsistent alignment'
                elif i in offsets:
                    offsets[j] = offsets[i]+delta
                elif j in offsets:
                    offsets[i] = offsets[j]-delta
        assert len(offsets) == len(tiles), 'unanchored diagonal placement'
        return [offsets[i] for i in range(len(tiles))]

    def group(ids):
        ids = tuple(sorted(ids))
        if ids in cache:
            return cache[ids]
        if len(ids) == 1:
            i = ids[0]
            if objects[i]['kind'] == 'B':
                result = _tile(np.ones((1, 1), dtype=bool), [(i, 0, 0)])
            else:
                content = group(info['children'][i])
                mask = np.pad(content['mask'], 2)
                mask[0, :] = mask[-1, :] = True
                mask[:, 0] = mask[:, -1] = True
                result = _tile(mask, [(k, y+2, x+2) for k, y, x in content['leaves']])
        else:
            candidates = []
            for axis in (0, 1):
                ordered = sorted(ids, key=lambda i: int(objects[i]['lo'][axis]))
                partitions = []
                end = -100
                for i in ordered:
                    start = int(objects[i]['lo'][axis])
                    if start > end+1:
                        partitions.append([])
                    partitions[-1].append(i)
                    end = max(end, int(objects[i]['hi'][axis]))
                if len(partitions) < 2:
                    continue
                try:
                    tiles = [group(part) for part in partitions]
                    candidate = _place(tiles, axis, align(tiles, axis), [1]*(len(tiles)-1))
                    candidates.append(candidate)
                except AssertionError:
                    continue
            assert candidates, 'no unambiguous cardinal decomposition'
            result = candidates[0]
            for candidate in candidates[1:]:
                assert np.array_equal(candidate['mask'], result['mask'])
                assert sorted(candidate['leaves']) == sorted(result['leaves'])
        cache[ids] = result
        return result

    result = group(info['roots'])
    # Exterior points need a halo, while separate frames need an actual gap.
    pad = [0, 0, 0, 0]
    h, w = result['mask'].shape
    exterior = {i for i in info['roots'] if objects[i]['kind'] == 'B'}
    for i, y, x in result['leaves']:
        if i in exterior:
            pad = [max(pad[0], 1-y), max(pad[1], y+2-h),
                   max(pad[2], 1-x), max(pad[3], x+2-w)]
    return np.pad(result['mask'], ((pad[0], pad[1]), (pad[2], pad[3])))


def solve(train_pairs, test_input):
    """Input-only geometric prediction; demonstrations establish this rule."""
    info = _parse_input(np.asarray(test_input))
    mask = _visible_layout(info)
    return np.where(mask, info['foreground'], info['background']).astype(np.int8)


def _leaf_count(node):
    if node[0] == 'B':
        return 1
    if node[0] == 'C':
        return _leaf_count(node[1])
    return sum(_leaf_count(child) for child in node[2])


def _group(rng, children, axis=None):
    if len(children) == 1:
        return children[0]
    if axis is None:
        axis = int(rng.integers(2))
    return ('G', axis, tuple(children), tuple(int(rng.integers(_leaf_count(c))) for c in children))


def _random_node(rng, depth=1, rich=False):
    n = int(rng.choice([1, 2, 3, 4], p=[.32, .39, .23, .06]))
    children = []
    for _ in range(n):
        if depth < 3 and rng.random() < (.35 if depth == 1 else .12):
            children.append(_random_node(rng, depth+1))
        else:
            children.append(('B',))
    rng.shuffle(children)
    axis = int(rng.integers(2))
    if n >= 3 and (rich or rng.random() < .65):
        cut = int(rng.integers(1, n))
        a = _group(rng, children[:cut], 1-axis)
        b = _group(rng, children[cut:], 1-axis)
        content = _group(rng, [a, b], axis)
    else:
        content = _group(rng, children, axis)
    return ('C', content)


def _sample_scene(rng, witness=False):
    if witness:
        # A compositional witness for nesting, mixed siblings, and cross-frame
        # alignment. The nesting position, arms, counts, directions, and anchor
        # are resampled, and its position in the training list is shuffled.
        inner = ('C', _group(rng, [('B',)] * int(rng.choice([1, 1, 2]))))
        siblings = [inner, ('B',), ('B',)]
        if rng.random() < .25:
            siblings.append(('B',))
        rng.shuffle(siblings)
        axis = int(rng.integers(2))
        cut = int(rng.integers(1, len(siblings)))
        content = _group(rng, [_group(rng, siblings[:cut], 1-axis),
                               _group(rng, siblings[cut:], 1-axis)], axis)
        outer = ('C', content)
        roots = [outer, ('B',)]
        rng.shuffle(roots)
        return _group(rng, roots, int(rng.integers(2)))
    principal = _random_node(rng)
    roots = [principal]
    if rng.random() < .47:
        roots.append(_random_node(rng, 2) if rng.random() < .19 else ('B',))
    if rng.random() < .09:
        roots.append(('B',))
    rng.shuffle(roots)
    if len(roots) == 3 and rng.random() < .5:
        return _group(rng, [roots[0], _group(rng, roots[1:])])
    return _group(rng, roots)


def _blob_mask(rng, small):
    cap = 3 if small else 4
    h, w = (int(rng.integers(1, cap+1)) for _ in range(2))
    if max(h, w) > 2*min(h, w)+1:
        h = w = int(rng.integers(2, cap+1))
    mask = np.ones((h, w), dtype=bool)
    if rng.random() < .65:
        # Peel boundary pixels, retaining connected, hole-free filled clusters.
        for _ in range(int(rng.integers(1, max(2, h*w//2)))):
            y, x = int(rng.integers(h)), int(rng.integers(w))
            if not mask[y, x] or mask.sum() <= max(1, h*w*.55):
                continue
            candidate = mask.copy()
            candidate[y, x] = False
            if len(_components(candidate)) == 1 and not _holes(candidate).any():
                mask = candidate
    ys, xs = np.nonzero(mask)
    return mask[ys.min():ys.max()+1, xs.min():xs.max()+1]


def _profile(rng, length, cap, style):
    if cap == 0 or style == 0:
        return np.zeros(length, dtype=int)
    t = np.linspace(0, 1, length)
    if style == 1:  # broad middle bulge, narrow ends
        values = np.abs(2*t-1)**int(rng.integers(1, 4))
    elif style == 2:  # taper, with independently chosen direction
        values = t if rng.random() < .5 else 1-t
    elif style == 3:  # genuine inward bay along an edge
        center = float(rng.uniform(.3, .7))
        values = np.maximum(0, 1-np.abs(t-center)/float(rng.uniform(.18, .4)))
    else:  # irregular stepped edge, with independent plateau widths
        knots = int(rng.integers(3, min(8, length)+1))
        values = np.interp(t, np.linspace(0, 1, knots), rng.random(knots))
    out = np.rint(values*cap).astype(int)
    if rng.random() < .35:
        out = cap-out
    return out


def _outline(rng, h, w, padding, style):
    if style == 0:
        mask = np.zeros((h, w), dtype=bool)
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
        return mask
    top, bottom, left, right = padding
    caps = [v-2 for v in padding]
    tcap, bcap, lcap, rcap = caps
    styles = [style] + [int(rng.integers(1, 6)) if style else 0 for _ in range(3)]
    t = _profile(rng, w, tcap, styles[0])
    b = h-1-_profile(rng, w, bcap, styles[1])
    l = _profile(rng, h, lcap, styles[2])
    r = w-1-_profile(rng, h, rcap, styles[3])
    # Four disjoint side corridors surround the children and their moat.
    # Orthogonal connectors make every staircase 4-connected, even at corners.
    x0, x1, y0, y1 = lcap, w-1-rcap, tcap, h-1-bcap
    path = [(int(t[x]), x) for x in range(x0, x1+1)]
    path += [(y0, x1), (y0, int(r[y0]))]
    path += [(y, int(r[y])) for y in range(y0, y1+1)]
    path += [(int(b[x1]), int(r[y1])), (int(b[x1]), x1)]
    path += [(int(b[x]), x) for x in range(x1, x0-1, -1)]
    path += [(y1, x0), (y1, int(l[y1]))]
    path += [(y, int(l[y])) for y in range(y1, y0-1, -1)]
    path += [(int(t[x0]), int(l[y0])), path[0]]
    mask = np.zeros((h, w), dtype=bool)
    for (y, x), (yy, xx) in zip(path, path[1:]):
        mask[min(y, yy):max(y, yy)+1, x] = True
        mask[yy, min(x, xx):max(x, xx)+1] = True
    return mask


def _draw_scene(node, rng, small=False):
    serial = [0]
    exterior = set()

    def draw(obj, depth):
        if obj[0] == 'B':
            key = serial[0]
            serial[0] += 1
            if depth == 0:
                exterior.add(key)
            mask = _blob_mask(rng, small)
            h, w = mask.shape
            return (_tile(mask, [(key, (h-1)//2, (w-1)//2)]),
                    _tile(np.ones((1, 1), dtype=bool), [(key, 0, 0)]))
        if obj[0] == 'G':
            pairs = [draw(child, depth) for child in obj[2]]
            axis, cross = obj[1], 1-obj[1]
            tiles = []
            for version in (0, 1):
                parts = [p[version] for p in pairs]
                offsets = [-p['leaves'][index][1+cross] for p, index in zip(parts, obj[3])]
                if version == 0 and rng.random() < .4:
                    offsets = [v+int(rng.integers(-1, 2)) for v in offsets]
                gaps = ([int(rng.integers(1, 4 if small else 6)) for _ in parts[:-1]]
                        if version == 0 else [1]*(len(parts)-1))
                tiles.append(_place(parts, axis, offsets, gaps))
            return tuple(tiles)
        content, canonical = draw(obj[1], depth+1)
        padding = [int(rng.integers(2, 5 if small else 7)) for _ in range(4)]
        top, bottom, left, right = padding
        h, w = content['mask'].shape
        style = int(rng.choice(6, p=[.09, .2, .16, .22, .17, .16]))
        mask = _outline(rng, h+top+bottom, w+left+right, padding, style)
        mask[top:top+h, left:left+w] |= content['mask']
        inp = _tile(mask, [(k, y+top, x+left) for k, y, x in content['leaves']])
        out = np.pad(canonical['mask'], 2)
        out[0, :] = out[-1, :] = True
        out[:, 0] = out[:, -1] = True
        return inp, _tile(out, [(k, y+2, x+2) for k, y, x in canonical['leaves']])

    inp, out = draw(node, 0)
    h, w = out['mask'].shape
    pads = [0, 0, 0, 0]
    for k, y, x in out['leaves']:
        if k in exterior:
            pads = [max(pads[0], 1-y), max(pads[1], y+2-h),
                    max(pads[2], 1-x), max(pads[3], x+2-w)]
    return inp['mask'], np.pad(out['mask'], ((pads[0], pads[1]), (pads[2], pads[3])))


def _fingerprint(grid):
    value = 1469598103934665603
    for cell in grid.ravel().tolist():
        value = ((value ^ (int(cell)+1))*1099511628211) & ((1 << 64)-1)
    return (*grid.shape, value)


def _key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _features(grid):
    info = _parse_input(grid)
    objs = info['components']
    depths = []
    for i, obj in enumerate(objs):
        depth, parent = 0, info['parents'][i]
        while parent is not None:
            depth += 1
            parent = info['parents'][parent]
        if obj['kind'] == 'C':
            depths.append(depth+1)
    out = _visible_layout(info)
    normalized = _parse_input(out.astype(np.int8), 0)
    no = normalized['components']
    # Mixed sibling arrangements cannot be separated into a single aligned row
    # or column: their normalized component centers vary in both coordinates.
    mixed = False
    max_siblings = 0
    for ids in list(normalized['children'].values())+[normalized['roots']]:
        max_siblings = max(max_siblings, len(ids))
        if len(ids) >= 3:
            centers = np.array([(no[i]['lo']+no[i]['hi']) for i in ids])
            mixed |= bool(np.ptp(centers[:, 0]) and np.ptp(centers[:, 1]))
    ext = [i for i in normalized['roots'] if no[i]['kind'] == 'B']
    frames = [i for i in normalized['roots'] if no[i]['kind'] == 'C']
    offcenter = False
    sides = set()
    for b in ext:
        for c in frames:
            p = no[b]['lo']
            lo, hi = no[c]['lo'], no[c]['hi']
            for axis, lower, upper in ((0, 'above', 'below'), (1, 'left', 'right')):
                if p[axis] < lo[axis] or p[axis] > hi[axis]:
                    sides.add(lower if p[axis] < lo[axis] else upper)
                    offcenter |= bool(2*p[1-axis] != lo[1-axis]+hi[1-axis])
    contour_shapes, blob_shapes, areas, nonrect, concave = [], [], [], 0, 0
    boundary_missing, clearances, gaps = [], [], []
    for obj in objs:
        dims = tuple((obj['hi']-obj['lo']+1).tolist())
        if obj['kind'] == 'C':
            contour_shapes.append(dims)
            y, x = obj['lo']; yy, xx = obj['hi']
            local = obj['mask'][y:yy+1, x:xx+1]
            nonrect += not (local[0].all() and local[-1].all() and local[:, 0].all() and local[:, -1].all())
            boundary_missing.append(int((~local[0]).sum()+(~local[-1]).sum()+
                                        (~local[1:-1, 0]).sum()+(~local[1:-1, -1]).sum()))
            filled = local | obj['hole'][y:yy+1, x:xx+1]
            # A bay has an edge lying inward of the chord between two edge cells.
            for scan in (filled, filled.T):
                lefts = [int(np.nonzero(row)[0][0]) for row in scan]
                rights = [int(np.nonzero(row)[0][-1]) for row in scan]
                if any(lefts[k] > max(lefts[0], lefts[-1]) or rights[k] < min(rights[0], rights[-1]) for k in range(1, len(scan)-1)):
                    concave += 1
                    break
        else:
            blob_shapes.append(dims)
            areas.append(obj['area'])
    for i, ids in info['children'].items():
        if ids:
            lo = np.min([objs[j]['lo'] for j in ids], axis=0)
            hi = np.max([objs[j]['hi'] for j in ids], axis=0)
            clearances.extend((lo-objs[i]['lo']-1).tolist())
            clearances.extend((objs[i]['hi']-hi-1).tolist())
    axes = set()
    for ids in list(info['children'].values())+[info['roots']]:
        for a in ids:
            for b in ids:
                if a >= b:
                    continue
                for axis in (0, 1):
                    gap = max(objs[b]['lo'][axis]-objs[a]['hi'][axis],
                              objs[a]['lo'][axis]-objs[b]['hi'][axis])-1
                    if gap > 0:
                        gaps.append(int(gap))
                        axes.add(axis)
    return dict(contours=len(depths), blobs=len(areas), depth=max(depths),
                mixed=bool(mixed), offcenter=bool(offcenter), exterior=len(ext),
                sides=sorted(sides), top_contours=len(frames), siblings=max_siblings,
                nonrect=int(nonrect), concave=concave, contour_shapes=contour_shapes,
                blob_shapes=blob_shapes, blob_areas=areas,
                boundary_missing=boundary_missing, clearances=clearances,
                gaps=gaps, axes=sorted(axes), topology=_tree_signature(info),
                palette=(info['background'], info['foreground']),
                edge=bool(np.any(grid[0] == info['foreground']) or np.any(grid[-1] == info['foreground']) or
                          np.any(grid[:, 0] == info['foreground']) or np.any(grid[:, -1] == info['foreground'])),
                output_key=_key(out), input_shape=grid.shape)


def _make_pair(rng, palette, size_policy, witness=False):
    background, foreground = palette
    # Bounded stochastic search, progressively using compact metric realizations.
    # The final attempts use a small compositional witness, still randomized.
    for attempt in range(160):
        scene = _sample_scene(rng, witness)
        if attempt >= 100:
            b = ('B',)
            inner = ('C', b)
            mixed = _group(rng, [b, _group(rng, [b, inner], 1)], 0)
            scene = _group(rng, [('C', mixed), b], 0) if witness else ('C', _group(rng, [b, b]))
        inp, expected = _draw_scene(scene, rng, small=attempt >= 8 or size_policy == 'compact')
        h, w = inp.shape
        if max(h, w, *expected.shape) > 30:
            continue
        if not (h > expected.shape[0] and w > expected.shape[1]):
            continue
        if size_policy == 'large':
            height, width = (int(rng.integers(max(v, 25), 31)) for v in (h, w))
        elif size_policy == 'compact':
            height, width = min(30, h+int(rng.integers(0, 5))), min(30, w+int(rng.integers(0, 5)))
        else:
            height, width = int(rng.integers(h, 31)), int(rng.integers(w, 31))
        if rng.random() < .09:
            height = width = 30
        y, x = int(rng.integers(height-h+1)), int(rng.integers(width-w+1))
        grid = np.full((height, width), background, dtype=np.int8)
        grid[y:y+h, x:x+w][inp] = foreground
        if (grid == background).sum() <= (grid == foreground).sum():
            continue
        output = np.where(expected, foreground, background).astype(np.int8)
        try:
            inferred = solve([], grid)
            assert np.array_equal(inferred, output)
            assert np.array_equal(_visible_layout(_parse_input(output, background)), expected)
            if witness:
                f = _features(grid)
                assert f['mixed'] and f['offcenter'] and f['depth'] >= 2
        except AssertionError:
            continue
        if _fingerprint(grid) in _SOURCE_INPUT_FINGERPRINTS:
            continue
        return dict(input=grid, output=output)
    # Proven-fit procedural fallback, independent of the source fixtures.
    point = lambda key: _tile(np.ones((1, 1), bool), [(key, 0, 0)])
    inner_mask = np.ones((5, 5), bool)
    inner_mask[1:4, 1:4] = False
    inner_mask[2, 2] = True
    inner = _tile(inner_mask, [(0, 2, 2)])
    content = _place([point(1), inner], 1, [2, 0], [1])
    content = _place([point(2), content], 0, [4, 0], [1])
    mask = np.pad(content['mask'], 2)
    mask[0, :] = mask[-1, :] = True
    mask[:, 0] = mask[:, -1] = True
    outer = _tile(mask, [(k, y+2, x+2) for k, y, x in content['leaves']])
    mask = np.pad(_place([outer, point(3)], 0, [0, 6], [1])['mask'], ((0, 1), (0, 0)))
    mask = np.rot90(mask, int(rng.integers(4)))
    if rng.random() < .5:
        mask = mask[:, ::-1]
    grid = np.where(mask.repeat(2, 0).repeat(2, 1), foreground, background).astype(np.int8)
    return dict(input=grid, output=np.where(mask, foreground, background).astype(np.int8))


def _resolve_counts(rng, num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError(name+' is outside the allowed integer range')
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train-1 if num_train is not None else 4)+1))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test+1), 11))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    # Puzzle-level policies control sampling, never the transformation itself.
    palette_mode = int(rng.integers(4))  # stable, swapping roles, or freely varying
    base_palette = tuple(int(c) for c in rng.choice(10, 2, replace=False))
    regime = int(rng.integers(4))  # comparable, larger test, smaller test, mixed
    policies = [('any', 'any'), ('compact', 'large'), ('large', 'compact'), ('any', 'any')][regime]
    puzzle = dict(train=[], test=[])
    seen = set()
    index = 0
    for split, count, policy in (('train', num_train, policies[0]), ('test', num_test, policies[1])):
        for i in range(count):
            if palette_mode == 0:
                palette = base_palette
            elif palette_mode == 1:
                palette = base_palette if index % 2 == 0 else base_palette[::-1]
            else:
                palette = tuple(int(c) for c in rng.choice(10, 2, replace=False))
            pair = _make_pair(rng, palette, policy, witness=split == 'train' and i == 0)
            # Different canvas placements produce a finite deterministic path
            # out of the extraordinarily unlikely duplicate-input case.
            for retry in range(32):
                if _key(pair['input']) not in seen:
                    break
                pair = _make_pair(rng, palette, policy, witness=split == 'train' and i == 0)
            assert _key(pair['input']) not in seen
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
            index += 1
        rng.shuffle(puzzle[split])
    return puzzle


def _tree_signature(info, ids=None):
    if ids is None:
        ids = info['roots']
    return tuple(sorted(('B',) if info['components'][i]['kind'] == 'B'
                        else ('C', _tree_signature(info, info['children'][i])) for i in ids))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen = set()
    witnesses = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= d <= 30 for d in grid.shape)
                assert 0 <= grid.min() <= grid.max() <= 9
            inp, out = pair['input'], pair['output']
            assert _key(inp) not in seen
            seen.add(_key(inp))
            assert _fingerprint(inp) not in _SOURCE_INPUT_FINGERPRINTS
            assert inp.shape[0] > out.shape[0] and inp.shape[1] > out.shape[1]
            info = _parse_input(inp)
            assert np.array_equal(out, solve(puzzle['train'], inp))
            output_info = _parse_input(out, info['background'])
            assert _tree_signature(info) == _tree_signature(output_info)
            for i, obj in enumerate(output_info['components']):
                if obj['kind'] == 'B':
                    assert obj['area'] == 1
                else:
                    y, x = obj['lo']; yy, xx = obj['hi']
                    assert obj['area'] == 2*(yy-y+xx-x)
                    children = [output_info['components'][j] for j in output_info['children'][i]]
                    assert np.array_equal(np.min([c['lo'] for c in children], axis=0), obj['lo']+2)
                    assert np.array_equal(np.max([c['hi'] for c in children], axis=0), obj['hi']-2)
            assert np.array_equal(_visible_layout(output_info), out == info['foreground'])
            if split == 'train':
                witnesses.append(_features(inp))
    assert any(f['mixed'] and f['offcenter'] and f['depth'] >= 2 for f in witnesses)
    # No split-specific literal palette or size rule exists. The witness exposes
    # both spatial axes, recursive containment, and asymmetric exterior alignment.


# Exact source regressions are fixtures, never sampling templates.
_SOURCE_REGRESSIONS = (
    (
        """3334444444333333333333333
4444333334443333333333333
4333333333344433333333333
4333333333333443333333333
4333333333333343333333333
4333444444333344333333333
4334433334433334334433333
4334333333433334334443333
4334344433433334333443333
4344334433433334333333333
4334333333433344333333333
4334433334433343333333333
4333444444333343333333333
4333333333333343333333333
4333333333333443333333333
4433344433333433333333333
3433344433334433333333333
3433333333334333333333333
3433333333344333333333333
3444444444443333333333333""",
        """444444444333
433333334333
434444434333
434333434333
434343434343
434333434333
434444434333
433333334333
433343334333
433333334333
444444444333""",
    ),
    (
        """1111111111111111
1111111111111111
1111111111111111
1111111111111111
1111111111111111
1111144444411111
1114441111411111
1144111411411111
1141144411411111
1141144411411111
1141114111411111
1141111114411111
1144111444111111
1114444411111111
1111111111111111
1111111111111111
1111111111111111
1111111111111111
1111111111111111
1111111111111111""",
        """44444
41114
41414
41114
44444""",
    ),
    (
        """4444999999444444
4449944449444444
4999444449944444
4944444444944444
4944499444944444
4944999444944444
4944994444944444
4944444444944444
4944449444944444
4944999449944444
4944499449444444
4994444449444444
4494444449444444
4994444449444444
4944444499444444
4999944994444444
4444999944444444
4444444444444444
4444444444444444
4444444444444444""",
        """99999
94449
94949
94449
94949
94449
99999""",
    ),
    (
        """2222222222222222222222222
2222222222222222222222222
2222222222444444422222222
2222444444422222422222222
2222422222222222442222222
2222422244444422244422222
2222422442222422222442222
2222422422442422222242222
2222422424442424442242222
2222442424442424422242222
2222242422222422222442222
2222242444224422244422222
2222242224444222442222222
2222244222222224422222222
2222224444444444222222222
2222222222222222222222222
2222222222222222222222222
2222222222222222222222222
2222222222222222222222222
2222222222222222222222222""",
        """44444444444
42222222224
42444442224
42422242224
42424242424
42422242224
42444442224
42222222224
44444444444""",
    ),
    (
        """7777777777799977777777777
7777999999997999997777777
7799977777777777799777777
7997777777777777779997777
7977777777777777777799977
7977779999999777777777997
7977799777779999997777799
7977997777777777797777779
7977977777777777799799779
7977977997779999779799779
7977977999779999779777779
7977977799779997799777779
7977977777777777997777779
7977999777777799977777799
7997779999999997777777797
7799777777777777777779997
7779777777777777799999777
7779977777779999997777777
7777999999999777777777777
7777777777777777777777777
7777777777777777777777777
7777777777777777777777777
7777777777777777777777777
7777777777777777777777777
7777777777777777777777777""",
        """9999999999999
9777777777779
9799999997779
9797777797779
9797979797979
9797777797779
9799999997779
9777777777779
9999999999999""",
    ),
    (
        """8888666666666666668888888
8666688888888888866688888
6688888888888888888668888
6888888888666888888868888
6888888886666888888868888
6888888888888888888868888
6888888888888888888866888
6888888866666668888886888
6888888868888866888886888
6866688868666886888886888
6866688868666886888866888
6886688868868866888868888
6888888866888668888868888
6888888886666688888668888
6688888888888888888688888
8668888888888888886688888
8866888888888888886888888
8886666888888888866888888
8888886666666666668888888
8888888888888888888888888
8888888888888888888888888
8888888888666888888888888
8888888886666888888888888
8888888888668888888888888
8888888888888888888888888""",
        """66666666666
68888888886
68888868886
68888888886
68886666686
68886888686
68686868686
68886888686
68886666686
68888888886
66666666666
88888888888
88888868888
88888888888""",
    ),
)
_SOURCE_INPUT_FINGERPRINTS = {(20,25,4252032328188928144), (20,16,12739395773425212587), (20,16,137190695620593818), (20,25,11660064614137846301), (25,25,2865180405025538449), (25,25,7630536321869844588)}


def review_metrics(puzzles):
    """Measure returned grids, without consulting the sampler's latent choices."""
    rows = []
    for index, puzzle in enumerate(puzzles):
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                rows.append(dict(_features(pair['input']), puzzle=index, split=split))

    def hist(values):
        result = {}
        for value in values:
            key = str(value)
            result[key] = result.get(key, 0)+1
        return dict(sorted(result.items()))

    groups = [[r for r in rows if r['puzzle'] == i] for i in range(len(puzzles))]
    total_contours = sum(r['contours'] for r in rows)
    within = {}
    for name in ('input_shape', 'output_key', 'contours', 'blobs', 'depth', 'palette', 'topology'):
        within[name] = sum(len({r[name] for r in group}) > 1 for group in groups)
    stable = sum(len({r['palette'] for r in group}) == 1 for group in groups)
    swaps = sum(any(r['palette'][::-1] in {q['palette'] for q in group} for r in group) for group in groups)
    shape_sets = [set(r['blob_shapes']) for r in rows]
    regimes = dict(larger_test=0, smaller_test=0, comparable=0)
    for puzzle in puzzles:
        areas = {s: np.mean([p['input'].size for p in puzzle[s]]) for s in puzzle}
        ratio = areas['test']/areas['train']
        regimes['larger_test' if ratio > 1.2 else 'smaller_test' if ratio < 1/1.2 else 'comparable'] += 1
    splits = {}
    for split in ('train', 'test'):
        selected = [r for r in rows if r['split'] == split]
        splits[split] = dict(pairs=len(selected), mixed=sum(r['mixed'] for r in selected),
                             offcenter=sum(r['offcenter'] for r in selected),
                             no_exterior=sum(r['exterior'] == 0 for r in selected),
                             depths=hist(r['depth'] for r in selected),
                             top_contours=hist(r['top_contours'] for r in selected),
                             input_shapes=len({r['input_shape'] for r in selected}),
                             normalized_outputs=len({r['output_key'] for r in selected}),
                             exterior_sides=hist(s for r in selected for s in r['sides']))
    # Reproduce the review's corpus shortcut on disjoint halves of the suite.
    lookup = {}
    for r in rows:
        if r['split'] == 'test' and r['puzzle'] < len(puzzles)//2:
            key = tuple(sorted(r['contour_shapes']))
            counts = lookup.setdefault(key, {})
            counts[r['output_key']] = counts.get(r['output_key'], 0)+1
    targets = [r for r in rows if r['split'] == 'test' and r['puzzle'] >= len(puzzles)//2]
    hits = known = 0
    for r in targets:
        choices = lookup.get(tuple(sorted(r['contour_shapes'])), {})
        if choices:
            known += 1
            hits += max(choices, key=choices.get) == r['output_key']
    realizations = {}
    box_collisions = {}
    for r in rows:
        realizations.setdefault(r['output_key'], set()).add(tuple(sorted(r['contour_shapes'])))
        box_collisions.setdefault(tuple(sorted(r['contour_shapes'])), set()).add(r['output_key'])
    return dict(puzzles=len(puzzles), pairs=len(rows), splits=splits,
                train_counts=hist(len(p['train']) for p in puzzles),
                test_counts=hist(len(p['test']) for p in puzzles),
                normalized_outputs=len({r['output_key'] for r in rows}),
                topologies=len({r['topology'] for r in rows}),
                input_shapes=len({r['input_shape'] for r in rows}),
                dimensions_range=[min(min(r['input_shape']) for r in rows), max(max(r['input_shape']) for r in rows)],
                square_30=sum(r['input_shape'] == (30, 30) for r in rows),
                compact=sum(max(r['input_shape']) <= 18 for r in rows),
                edge_touching=sum(r['edge'] for r in rows),
                contours=total_contours, nonrectangular=sum(r['nonrect'] for r in rows),
                concave=sum(r['concave'] for r in rows),
                contour_counts=hist(r['contours'] for r in rows),
                blob_counts=hist(r['blobs'] for r in rows), depths=hist(r['depth'] for r in rows),
                sibling_counts=hist(r['siblings'] for r in rows),
                blob_shapes=hist(s for r in rows for s in r['blob_shapes']),
                blob_areas=hist(a for r in rows for a in r['blob_areas']),
                clearances=hist(c for r in rows for c in r['clearances']),
                sibling_gaps=hist(g for r in rows for g in r['gaps']),
                boundary_missing=hist(m for r in rows for m in r['boundary_missing']),
                same_pair_blob_size_variation=sum(len(s) > 1 for s in shape_sets),
                same_pair_boundary_variation=sum(len(set(r['boundary_missing'])) > 1 for r in rows),
                combined_mixed_distorted_varied_blobs=sum(r['mixed'] and r['nonrect'] > 0 and len(set(r['blob_shapes'])) > 1 for r in rows),
                within_puzzle_variation=within, stable_palettes=stable, role_swaps=swaps,
                palettes=len({r['palette'] for r in rows}), measured_size_regimes=regimes,
                output_layouts_with_multiple_metric_realizations=sum(len(v) > 1 for v in realizations.values()),
                contour_size_keys_with_multiple_outputs=sum(len(v) > 1 for v in box_collisions.values()),
                bbox_shortcut=dict(targets=len(targets), known_keys=known, correct=hits))


def _puzzle_key(puzzle):
    return tuple((s, tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s])) for s in ('train', 'test'))


def self_test() -> None:
    arr = lambda text: np.array([[int(c) for c in row] for row in text.split()], dtype=np.int8)
    for raw_input, raw_output in _SOURCE_REGRESSIONS:
        inp, out = arr(raw_input), arr(raw_output)
        assert np.array_equal(solve([], inp), out), 'source regression'
        assert _tree_signature(_parse_input(inp)) == _tree_signature(_parse_input(out, _parse_input(inp)['background']))
    # Same two contour bounding boxes, different visible topology and answer.
    # This directly defeats the reported shortcut even when the key is known.
    counterexamples = []
    for count in (1, 2):
        grid = np.zeros((25, 25), dtype=np.int8)
        grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 1
        grid[5, 5:20] = grid[19, 5:20] = 1
        grid[5:20, 5] = grid[5:20, 19] = 1
        grid[10:12, 8:10] = 1
        if count == 2:
            grid[10:12, 15:17] = 1
        info = _parse_input(grid)
        boxes = sorted(tuple((o['hi']-o['lo']+1).tolist()) for o in info['components'] if o['kind'] == 'C')
        prediction = solve([], grid)
        assert prediction.shape == (9, 7+2*count)
        counterexamples.append((boxes, _key(prediction)))
    assert counterexamples[0][0] == counterexamples[1][0]
    assert counterexamples[0][1] != counterexamples[1][1]
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    metrics = review_metrics(puzzles)
    assert len(metrics['train_counts']) >= 4 and len(metrics['test_counts']) >= 3
    assert metrics['normalized_outputs'] >= 100 and metrics['topologies'] >= 30
    assert metrics['input_shapes'] >= 100 and metrics['palettes'] >= 60
    assert metrics['depths'].get('3', 0) >= 2 and len(metrics['contour_counts']) >= 3
    assert len(metrics['blob_counts']) >= 5 and len(metrics['sibling_counts']) >= 4
    assert metrics['nonrectangular']/metrics['contours'] >= .75
    assert metrics['concave'] >= 20
    assert len(metrics['blob_shapes']) >= 12 and len(metrics['blob_areas']) >= 12
    assert len(metrics['clearances']) >= 5 and len(metrics['sibling_gaps']) >= 6
    assert metrics['same_pair_blob_size_variation'] >= 150
    assert metrics['same_pair_boundary_variation'] >= 80
    assert metrics['combined_mixed_distorted_varied_blobs'] >= 40
    assert metrics['compact'] >= 20 and metrics['square_30'] >= 20 and metrics['edge_touching'] >= 100
    assert metrics['stable_palettes'] >= 4 and metrics['role_swaps'] >= 4
    assert all(metrics['measured_size_regimes'][s] >= 4 for s in metrics['measured_size_regimes'])
    assert metrics['within_puzzle_variation']['input_shape'] >= 40
    assert metrics['within_puzzle_variation']['output_key'] >= 40
    assert metrics['output_layouts_with_multiple_metric_realizations'] >= 35
    assert metrics['contour_size_keys_with_multiple_outputs'] >= 15
    for split in ('train', 'test'):
        f = metrics['splits'][split]
        assert f['mixed'] >= (42 if split == 'train' else 8)
        assert f['offcenter'] >= (42 if split == 'train' else 8)
        assert f['no_exterior'] >= 10 and len(f['exterior_sides']) == 4
        assert f['top_contours'].get('2', 0) >= 2
    shortcut = metrics['bbox_shortcut']
    assert shortcut['correct'] < shortcut['targets']*.35
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(generate_puzzle(19)) == _puzzle_key(puzzles[19])
    assert _puzzle_key(generate_puzzle(47, 7, 3)) == _puzzle_key(generate_puzzle(47, 7, 3))
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(921, train, test)
        validate_generated_puzzle(p)
        assert train is None or len(p['train']) == train
        assert test is None or len(p['test']) == test
    for train, test in ((1, 1), (11, 1), (2, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid count request accepted')
    # Metamorphic checks use the visible input alone. They enlarge arbitrary
    # rows/columns, so the old fixed-size blob parser could not pass these.
    rng = np.random.default_rng(8021)
    for puzzle in puzzles:
        pair = puzzle['test'][0]
        inp = pair['input'].copy()
        prediction = solve(puzzle['train'], inp)
        assert np.array_equal(prediction, pair['output'])
        bg = _parse_input(inp)['background']
        permutation = rng.permutation(10)
        assert np.array_equal(solve([], permutation[inp]), permutation[prediction])
        shifted = np.pad(inp, ((2, 1), (1, 3)), constant_values=bg)
        assert np.array_equal(solve([], shifted), prediction)
        distorted = np.repeat(np.repeat(inp, rng.integers(1, 3, inp.shape[0]), axis=0),
                              rng.integers(1, 3, inp.shape[1]), axis=1)
        assert np.array_equal(solve([], distorted), prediction)
    # Independent metric redraws of fixed abstract scenes must normalize alike.
    realized = 0
    for _ in range(20):
        scene = _sample_scene(rng)
        answers, drawings = set(), set()
        for _ in range(12):
            inp, expected = _draw_scene(scene, rng, small=True)
            if max(inp.shape) > 30 or inp.sum()*2 >= inp.size:
                continue
            try:
                actual = solve([], inp.astype(np.int8))
                assert np.array_equal(actual, expected)
            except AssertionError:
                continue
            answers.add(_key(expected))
            drawings.add(_key(inp))
        assert len(answers) <= 1
        realized += len(drawings) >= 2
    assert realized >= 8
    # A wrong point, crop, or open frame must be rejected by the validator.
    for mode in range(3):
        p = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in ('train', 'test')}
        target = p['test'][0]
        bg = _parse_input(target['input'])['background']
        if mode == 0:
            info = _parse_input(target['output'], bg)
            blob = next(o for o in info['components'] if o['kind'] == 'B')
            target['output'][tuple(blob['lo'])] = bg
        elif mode == 1:
            target['output'] = np.pad(target['output'], 1, constant_values=bg)
        else:
            info = _parse_input(target['output'], bg)
            contour = next(o for o in info['components'] if o['kind'] == 'C')
            y, x = contour['lo']
            target['output'][y, x+1] = bg
        try:
            validate_generated_puzzle(p)
        except AssertionError:
            pass
        else:
            raise AssertionError('corrupted answer accepted')
