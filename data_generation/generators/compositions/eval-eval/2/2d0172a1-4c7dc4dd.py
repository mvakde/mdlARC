"""Membrane release: infer a local topological edit, then compact its result.

Only Python and NumPy are used. No parent module, file, seed or demonstration
is consulted by solve(). Internal colors 0/1/2 mean empty/ink/marked atom.
"""
from itertools import permutations

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _components(mask):
    seen = np.zeros(mask.shape, bool)
    result = []
    h, w = mask.shape
    for r, c in zip(*np.nonzero(mask)):
        if seen[r, c]:
            continue
        todo = [(int(r), int(c))]
        seen[r, c] = True
        cells = []
        while todo:
            y, x = todo.pop()
            cells.append((y, x))
            for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    todo.append((yy, xx))
        result.append(np.array(cells, dtype=int))
    return result


def _hole(mask):
    padded = np.pad(mask, 1)
    outside = np.zeros(padded.shape, bool)
    outside[0, 0] = True
    todo = [(0, 0)]
    h, w = padded.shape
    while todo:
        y, x = todo.pop()
        for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
            if 0 <= yy < h and 0 <= xx < w and not padded[yy, xx] and not outside[yy, xx]:
                outside[yy, xx] = True
                todo.append((yy, xx))
    return (~padded & ~outside)[1:-1, 1:-1]


def _perimeter(shape):
    mask = np.zeros(shape, bool)
    mask[0] = mask[-1] = True
    mask[:, 0] = mask[:, -1] = True
    return mask


def _inside(a, b):
    return bool(np.all(a[0] > b[0]) and np.all(a[1] < b[1]))


def _panels(grid):
    """Find the three outermost monochrome rectangular carrier components."""
    bg = int(grid[0, 0])
    _require(all(np.all(edge == bg) for edge in (grid[0], grid[-1], grid[:, 0], grid[:, -1])),
             "a blank exterior border is required")
    rectangles = []
    for color in np.unique(grid):
        if color == bg:
            continue
        for cells in _components(grid == color):
            lo, hi = cells.min(0), cells.max(0)
            dims = hi-lo+1
            if min(dims) < 5:
                continue
            mask = np.zeros(tuple(dims), bool)
            mask[tuple((cells-lo).T)] = True
            if np.array_equal(mask, _perimeter(tuple(dims))):
                rectangles.append((lo, hi, int(color)))
    carriers = [a for a in rectangles if not any(_inside(a, b) for b in rectangles)]
    _require(len(carriers) == 3, "expected three unambiguous carriers")
    for i, a in enumerate(carriers):
        for b in carriers[i+1:]:
            _require(bool(np.any(a[1] < b[0]) or np.any(b[1] < a[0])), "overlapping carriers")
    colors = [a[2] for a in carriers]
    repeated = [c for c in set(colors) if colors.count(c) == 2]
    _require(len(repeated) == 1 and len(set(colors)) == 2, "two reference frames and one target frame required")
    interiors, boxes = [], []
    for lo, hi, color in carriers:
        y, x = lo
        yy, xx = hi
        interiors.append(grid[y+1:yy, x+1:xx].copy())
        boxes.append((int(y), int(x), int(yy-y+1), int(xx-x+1), color))
    reference_ids = [i for i, color in enumerate(colors) if color == repeated[0]]
    target_id = next(i for i in range(3) if i not in reference_ids)
    outside = grid.copy()
    for y, x, h, w, _ in boxes:
        outside[y:y+h, x:x+w] = bg
    _require(np.all(outside == bg), "unassociated exterior drawing")
    _require(all(c not in set(np.unique(p)) for c in set(colors) for p in interiors),
             "carrier colors cannot occur in specimens")
    return bg, [interiors[i] for i in reference_ids], interiors[target_id], boxes, target_id


def _roles(panel, bg):
    colors = set(map(int, np.unique(panel))) - {bg}
    _require(len(colors) == 2, "specimen needs ink and marked roles")
    hollow = []
    for c in colors:
        for cells in _components(panel == c):
            mask = np.zeros(panel.shape, bool)
            mask[tuple(cells.T)] = True
            if _hole(mask).any():
                hollow.append(c)
    _require(len(set(hollow)) == 1, "ink role must be identified by an enclosure")
    ink = hollow[0]
    return ink, next(c for c in colors if c != ink)


def _parse(panel, bg, ink, marked):
    """Recover actual enclosure, rather than bounding-box containment."""
    _require(set(map(int, np.unique(panel))) <= {bg, ink, marked}, "unexpected specimen color")
    records = []
    for color, role in ((ink, 1), (marked, 2)):
        for cells in _components(panel == color):
            lo, hi = cells.min(0), cells.max(0)
            mask = np.zeros(panel.shape, bool)
            mask[tuple(cells.T)] = True
            hole = _hole(mask)
            contour = bool(hole.any())
            _require(not contour or role == 1, "marked atom cannot be hollow")
            _require(not contour or len(_components(hole)) == 1, "multiple-hole contour")
            if not contour:
                _require(len(cells) >= .4*np.prod(hi-lo+1), "noncompact terminal")
            records.append(dict(cells=cells, lo=lo, hi=hi, mask=mask, hole=hole,
                                contour=contour, role=role))
    _require(sum(o['role'] == 2 for o in records) == 1, "exactly one marked atom required")
    labels = np.zeros(panel.shape, int)
    for i, o in enumerate(records):
        labels[o['mask']] = i+1
    for a, b in ((labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:]),
                 (labels[:-1, :-1], labels[1:, 1:]), (labels[:-1, 1:], labels[1:, :-1])):
        _require(not np.any((a != b) & (a > 0) & (b > 0)), "touching specimen objects")
    parents = []
    for i, child in enumerate(records):
        containing = []
        for j, parent in enumerate(records):
            if i == j or not parent['contour']:
                continue
            flags = parent['hole'][tuple(child['cells'].T)]
            _require(flags.all() or not flags.any(), "crossing contour")
            if flags.all():
                containing.append(j)
        parents.append(min(containing, key=lambda j: int(records[j]['hole'].sum())) if containing else None)
    children = [[] for _ in records]
    for i, p in enumerate(parents):
        if p is not None:
            children[p].append(i)
    _require(all(bool(children[i]) == o['contour'] for i, o in enumerate(records)), "empty contour")
    marker_id = next(i for i, o in enumerate(records) if o['role'] == 2)
    ancestors = []
    p = parents[marker_id]
    while p is not None:
        ancestors.append(p)
        p = parents[p]
    return dict(objects=records, parents=parents, marker=marker_id, ancestors=ancestors)


def _join(tiles, axis, offsets, gaps):
    cross = 1-axis
    low = min(offsets)
    shape = [0, 0]
    starts = []
    position = 0
    for i, (a, leaves) in enumerate(tiles):
        start = [0, 0]
        start[axis], start[cross] = position, offsets[i]-low
        starts.append(start)
        shape[cross] = max(shape[cross], start[cross]+a.shape[cross])
        position += a.shape[axis] + (gaps[i] if i < len(gaps) else 0)
    shape[axis] = position
    result = np.zeros(shape, np.int8)
    result_leaves = []
    for (a, leaves), (y, x) in zip(tiles, starts):
        h, w = a.shape
        result[y:y+h, x:x+w] = a
        result_leaves.extend((key, r+y, c+x) for key, r, c in leaves)
    return result, result_leaves


def _normal(info, removed=()):
    """Compact surviving components using whitespace cuts and terminal anchors.

    Removing a membrane changes which components are siblings. Layout is then
    inferred again from those components, so erased frame sizes leave no gaps.
    """
    objects, old_parents = info['objects'], info['parents']
    removed = set(removed)
    live = [i for i in range(len(objects)) if i not in removed]
    children = {i: [] for i in live}
    roots = []
    for i in live:
        p = old_parents[i]
        while p in removed:
            p = old_parents[p]
        (roots if p is None else children[p]).append(i)
    cache = {}

    def offsets_for(tiles, axis):
        cross = 1-axis
        links = []
        for i in range(len(tiles)):
            for j in range(i+1, len(tiles)):
                shifts = set()
                for a, ay, ax in tiles[i][1]:
                    for b, by, bx in tiles[j][1]:
                        if max(objects[a]['lo'][cross], objects[b]['lo'][cross]) <= min(objects[a]['hi'][cross], objects[b]['hi'][cross]):
                            shifts.add((ay, ax)[cross]-(by, bx)[cross])
                _require(len(shifts) <= 1, "competing terminal alignments")
                if shifts:
                    links.append((i, j, shifts.pop()))
        offsets = {0: 0}
        for _ in tiles:
            for i, j, delta in links:
                if i in offsets and j in offsets:
                    _require(offsets[j] == offsets[i]+delta, "inconsistent alignment")
                elif i in offsets:
                    offsets[j] = offsets[i]+delta
                elif j in offsets:
                    offsets[i] = offsets[j]-delta
        _require(len(offsets) == len(tiles), "unanchored diagonal layout")
        return [offsets[i] for i in range(len(tiles))]

    def group(ids):
        key = tuple(sorted(ids))
        if key in cache:
            return cache[key]
        if len(key) == 1:
            i = key[0]
            o = objects[i]
            if not o['contour']:
                tile = (np.array([[o['role']]], np.int8), [(i, 0, 0)])
            else:
                a, leaves = group(children[i])
                a = np.pad(a, 2)
                a[_perimeter(a.shape)] = 1
                tile = a, [(i, y+2, x+2) for i, y, x in leaves]
        else:
            options = []
            for axis in (0, 1):
                partitions = []
                end = -2
                for i in sorted(key, key=lambda j: int(objects[j]['lo'][axis])):
                    if objects[i]['lo'][axis] > end+1:
                        partitions.append([])
                    partitions[-1].append(i)
                    end = max(end, int(objects[i]['hi'][axis]))
                if len(partitions) < 2:
                    continue
                try:
                    tiles = [group(part) for part in partitions]
                    options.append(_join(tiles, axis, offsets_for(tiles, axis), [1]*(len(tiles)-1)))
                except ValueError:
                    continue
            _require(bool(options), "no cardinal decomposition")
            tile = options[0]
            for other in options[1:]:
                _require(np.array_equal(tile[0], other[0]) and sorted(tile[1]) == sorted(other[1]),
                         "ambiguous normalization")
        cache[key] = tile
        return tile

    a, leaves = group(roots)
    # Frames crop flush; every terminal exposed in the exterior gets a halo.
    pads = [0, 0, 0, 0]
    exterior = {i for i in roots if not objects[i]['contour']}
    h, w = a.shape
    for i, y, x in leaves:
        if i in exterior:
            pads = [max(pads[0], 1-y), max(pads[1], y+2-h),
                    max(pads[2], 1-x), max(pads[3], x+2-w)]
    return np.pad(a, ((pads[0], pads[1]), (pads[2], pads[3])))


def _analyze(input_grid):
    grid = _grid(input_grid)
    bg, references, target, boxes, target_id = _panels(grid)
    # The reference with more membranes is the before-state, wherever located.
    counts = []
    for panel in references:
        counts.append(sum(_hole(o_mask).any() for color in np.unique(panel) if color != bg
                          for cells in _components(panel == color)
                          for o_mask in [_mask_for(panel.shape, cells)]))
    _require(counts[0] != counts[1], "reference must demonstrate membrane loss")
    before, after = references if counts[0] > counts[1] else references[::-1]
    ri, rm = _roles(before, bg)
    bi = _parse(before, bg, ri, rm)
    ai = _parse(after, bg, ri, rm)
    _normal(bi)  # Original states must also have unique cardinal layouts.
    k = len(bi['ancestors'])-len(ai['ancestors'])
    _require(1 <= k <= len(bi['ancestors']), "invalid release depth")
    _require(np.array_equal(_normal(bi, bi['ancestors'][:k]), _normal(ai)),
             "reference does not release its nearest marked ancestors")
    ti, tm = _roles(target, bg)
    target_info = _parse(target, bg, ti, tm)
    _require(len(target_info['ancestors']) >= k, "target has too few membranes")
    _normal(target_info)
    return dict(bg=bg, reference=bi, partner=ai, target=target_info, k=k,
                ink=ti, marked=tm, target_panel=target, boxes=boxes, target_id=target_id)


def _mask_for(shape, cells):
    mask = np.zeros(shape, bool)
    mask[tuple(cells.T)] = True
    return mask


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _analyze(input_grid)
    target = info['target']
    normalized = _normal(target, target['ancestors'][:info['k']])
    return np.array([info['bg'], info['ink'], info['marked']], dtype=np.int8)[normalized]


# The constructive renderer below builds outputs from abstract trees. It does
# not call the visible-input layout decoder to construct ground-truth labels.
def _leaf(role=1):
    return ('B', role)


def _leaves(node):
    if node[0] == 'B':
        return 1
    if node[0] == 'C':
        return _leaves(node[1])
    return sum(_leaves(child) for child in node[2])


def _group(rng, children, axis=None):
    if len(children) == 1:
        return children[0]
    axis = int(rng.integers(2)) if axis is None else axis
    return ('G', axis, tuple(children), tuple(int(rng.integers(_leaves(c))) for c in children))


def _mixed_group(rng, children):
    rng.shuffle(children)
    axis = int(rng.integers(2))
    if len(children) >= 3 and rng.random() < .7:
        cut = int(rng.integers(1, len(children)))
        return _group(rng, [_group(rng, children[:cut], 1-axis),
                            _group(rng, children[cut:], 1-axis)], axis)
    return _group(rng, children, axis)


def _has_marker(node):
    if node[0] == 'B':
        return node[1] == 2
    if node[0] == 'C':
        return _has_marker(node[1])
    return any(_has_marker(c) for c in node[2])


def _release(node, k):
    """Tree contraction, preserving grouping/anchors and every terminal."""
    path = []
    def locate(n):
        if n[0] == 'C':
            path.append(n)
            locate(n[1])
        elif n[0] == 'G':
            locate(next(c for c in n[2] if _has_marker(c)))
    locate(node)
    removed = {id(n) for n in path[-k:]}
    def edit(n):
        if n[0] == 'B':
            return n
        if n[0] == 'C':
            content = edit(n[1])
            return content if id(n) in removed else ('C', content)
        return ('G', n[1], tuple(edit(c) for c in n[2]), n[3])
    return edit(node)


def _blob(rng, compact):
    h, w = map(int, rng.integers(1, 3 if compact else 4, 2))
    a = np.ones((h, w), bool)
    if min(h, w) > 1 and rng.random() < .6:
        corners = [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]
        y, x = corners[int(rng.integers(4))]
        a[y, x] = False
    return a


def _outline(rng, shape, padding):
    """A closed stepped wall in safe corridors, sometimes a plain rectangle."""
    h, w = shape
    pt, pb, pl, pr = padding
    if rng.random() < .2 or max(padding) == 2:
        return _perimeter(shape)
    # Independent plateau profiles remove stretches of the rectangular sides.
    left, right = np.zeros(h, int), np.full(h, w-1, int)
    for values, cap, direction in ((left, pl-2, 1), (right, pr-2, -1)):
        if cap:
            cuts = sorted(set([1, h-1] + list(map(int, rng.integers(1, h, 3)))))
            for lo, hi in zip(cuts, cuts[1:]):
                values[lo:hi] += direction*int(rng.integers(cap+1))
    path = [(0, 0), (0, w-1)]
    path += [(y, int(right[y])) for y in range(h)]
    path += [(h-1, w-1), (h-1, 0)]
    path += [(y, int(left[y])) for y in range(h-1, -1, -1)]
    path.append(path[0])
    mask = np.zeros(shape, bool)
    for (y, x), (yy, xx) in zip(path, path[1:]):
        mask[min(y, yy):max(y, yy)+1, x] = True
        mask[yy, min(x, xx):max(x, xx)+1] = True
    # Rotate the wall construction's side profiles to both axes.
    return mask


def _draw(node, rng=None, compact=True):
    serial = [0]
    exterior = set()
    def draw(n, depth):
        if n[0] == 'B':
            key = serial[0]
            serial[0] += 1
            if depth == 0:
                exterior.add(key)
            mask = np.ones((1, 1), bool) if rng is None else _blob(rng, compact)
            a = np.where(mask, n[1], 0).astype(np.int8)
            return a, [(key, (a.shape[0]-1)//2, (a.shape[1]-1)//2)]
        if n[0] == 'G':
            tiles = [draw(c, depth) for c in n[2]]
            cross = 1-n[1]
            offsets = [-tile[1][anchor][1+cross] for tile, anchor in zip(tiles, n[3])]
            gaps = [1 if rng is None else int(rng.integers(1, 3 if compact else 5)) for _ in tiles[:-1]]
            return _join(tiles, n[1], offsets, gaps)
        a, leaves = draw(n[1], depth+1)
        if rng is None:
            pads = [2]*4
        elif compact:
            # Compact geometry is essential for packing a three-level target
            # beside the reference. Still vary sides independently; avoid four
            # compulsory half-probability expansions at every nesting level.
            pads = [2+int(rng.random() < .2) for _ in range(4)]
        else:
            pads = list(map(int, rng.integers(2, 5, 4)))
        pt, pb, pl, pr = pads
        shape = (a.shape[0]+pt+pb, a.shape[1]+pl+pr)
        if rng is None:
            wall = _perimeter(shape)
        elif rng.integers(2):
            wall = _outline(rng, shape[::-1], [pl, pr, pt, pb]).T
        else:
            wall = _outline(rng, shape, pads)
        result = wall.astype(np.int8)
        result[pt:pt+a.shape[0], pl:pl+a.shape[1]] = a
        return result, [(key, y+pt, x+pl) for key, y, x in leaves]
    a, leaves = draw(node, 0)
    if rng is not None:
        return a  # Metric specimens need no exterior-point crop convention.
    pads = [0, 0, 0, 0]
    for key, y, x in leaves:
        if key in exterior:
            pads = [max(pads[0], 1-y), max(pads[1], y+2-a.shape[0]),
                    max(pads[2], 1-x), max(pads[3], x+2-a.shape[1])]
    return np.pad(a, ((pads[0], pads[1]), (pads[2], pads[3])))


def _reference(rng):
    # An ordinary bead at the outer level distinguishes nearest from outermost
    # deletion, even though both alternatives delete one contour.
    inner = ('C', _leaf(2))
    content = [_leaf(), inner]
    if rng.random() < .3:
        content.append(_leaf())
    return ('C', _mixed_group(rng, content))


def _target(rng, depth, witness=False):
    n = _leaf(2)
    for level in range(depth):
        children = [n]
        if level > 0 or rng.random() < .35:
            extra = ('C', _leaf()) if (witness and level == depth-1) or rng.random() < .23 else _leaf()
            children.append(extra)
        if rng.random() < (.55 if witness else .3):
            children.append(_leaf())
        n = ('C', _mixed_group(rng, children))
    if rng.random() < .2 and not witness:
        n = _mixed_group(rng, [n, _leaf()])
    return n


def _fitting_trees(rng, depth, witness, axis=None):
    """Procedural final fallback with a proven 30-cell three-carrier packing.

    A shared grouping axis lets the compact three-level target fit beside the
    reference pair. Orders still vary; these are trees, not stored grid pairs.
    """
    axis = int(rng.integers(2)) if axis is None else axis
    def row(children):
        rng.shuffle(children)
        return _group(rng, children, axis)
    reference = ('C', row([_leaf(), ('C', _leaf(2))]))
    target = ('C', _leaf(2))
    for level in range(1, depth):
        sibling = ('C', _leaf()) if witness and level == depth-1 else _leaf()
        target = ('C', row([target, sibling]))
    return reference, target


def _pack(rng, shapes):
    """Pack three unequal panels in any orientation of a two-plus-one shelf."""
    candidates = []
    for axis in (0, 1):
        cross = 1-axis
        for order in permutations(range(3)):
            a, b, c = order
            for gap in (1, 2, 3):
                dims = [0, 0]
                dims[axis] = max(shapes[a][axis], shapes[b][axis])+gap+shapes[c][axis]
                dims[cross] = max(shapes[a][cross]+gap+shapes[b][cross], shapes[c][cross])
                if max(dims) <= 28:
                    candidates.append((axis, order, gap, dims))
    _require(bool(candidates), "specimens do not fit")
    axis, (a, b, c), gap, dims = candidates[int(rng.integers(len(candidates)))]
    cross = 1-axis
    positions = [[0, 0] for _ in range(3)]
    shelf_size = max(shapes[a][axis], shapes[b][axis])
    pair_width = shapes[a][cross]+gap+shapes[b][cross]
    shift = int(rng.integers(dims[cross]-pair_width+1))
    positions[a][cross] = shift
    positions[b][cross] = shift+shapes[a][cross]+gap
    for i in (a, b):
        positions[i][axis] = int(rng.integers(shelf_size-shapes[i][axis]+1))
    positions[c][axis] = shelf_size+gap
    positions[c][cross] = int(rng.integers(dims[cross]-shapes[c][cross]+1))
    canvas = [int(rng.integers(d+2, min(30, d+5)+1)) for d in dims]
    shifts = [int(rng.integers(1, canvas[j]-dims[j])) for j in range(2)]
    return tuple(canvas), [tuple(v+shifts[j] for j, v in enumerate(p)) for p in positions]


def _palette(rng, base, stable):
    colors = base if stable else rng.permutation(10)
    bg, rf, tf, ri, rm, ti, tm = map(int, colors[:7])
    mode = int(rng.integers(3))
    if mode == 0:
        ti, tm = ri, rm
    elif mode == 1:
        tm = rm
    return bg, rf, tf, ri, rm, ti, tm


def _scene_pair(rng, palette, depth, ks, witness=False):
    """Generate one scene or a diagnostic pair with identical occupied target."""
    bg, rf, tf, ri, rm, ti, tm = palette
    for attempt in range(180):
        if attempt < 170:
            reference = _reference(rng)
            target = _target(rng, depth, witness)
        else:
            reference, target = _fitting_trees(rng, depth, witness)
        rb = _draw(reference)
        partners = [_draw(_release(reference, k)) for k in ks]
        outputs = [_draw(_release(target, k)) for k in ks]
        if any(max(o.shape) > 30 for o in outputs):
            continue
        # Later attempts retain the sampled structural tree but remove metric
        # expansion. No source pair or stored output serves as a fallback.
        raw = _draw(target, rng, compact=attempt >= 3) if attempt < 130 else _draw(target)
        if max(raw.shape) > 26:  # Two carrier cells and two exterior cells needed.
            continue
        try:
            target_info = _parse(raw, 0, 1, 2)
            if len(target_info['ancestors']) != depth:
                continue
            if not np.array_equal(_normal(target_info), _draw(target)):
                continue
            if not all(np.array_equal(_normal(target_info, target_info['ancestors'][:k]), out)
                       for k, out in zip(ks, outputs)):
                continue
            if not all(np.array_equal(_normal(_parse(out, 0, 1, 2)), out) for out in outputs):
                continue
            ph = max(p.shape[0] for p in partners)
            pw = max(p.shape[1] for p in partners)
            shapes = [(rb.shape[0]+2, rb.shape[1]+2), (ph+2, pw+2),
                      (raw.shape[0]+2, raw.shape[1]+2)]
            canvas, positions = _pack(rng, shapes)
        except ValueError:
            continue
        # Common partner carrier stays fixed in twins; only its contents change.
        centered = []
        for partner in partners:
            centered.append(np.pad(partner, (((ph-partner.shape[0])//2, (ph-partner.shape[0]+1)//2),
                                             ((pw-partner.shape[1])//2, (pw-partner.shape[1]+1)//2))))
        pairs = []
        for p, out in zip(centered, outputs):
            grid = np.full(canvas, bg, np.int8)
            for i, (content, frame, ink, marker) in enumerate(((rb, rf, ri, rm), (p, rf, ri, rm), (raw, tf, ti, tm))):
                y, x = positions[i]
                h, w = shapes[i]
                grid[y:y+h, x:x+w][_perimeter((h, w))] = frame
                grid[y+1:y+h-1, x+1:x+w-1] = np.array([bg, ink, marker], np.int8)[content]
            output = np.array([bg, ti, tm], np.int8)[out]
            try:
                if not np.array_equal(solve(grid), output):
                    break
            except ValueError:
                break
            pairs.append(dict(input=grid, output=output))
        if len(pairs) == len(ks):
            return pairs
    raise RuntimeError("bounded construction could not find an unambiguous scene")


def _counts(num_train, num_test):
    for value, lo, hi, name in ((num_train, 2, 10, 'num_train'), (num_test, 1, 4, 'num_test')):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(name+' must be an integer in its legal range')
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    base = rng.permutation(10)
    stable = bool(rng.integers(2))
    palette = _palette(rng, base, stable)
    # These two demonstrations prove reference dependence with the SAME target.
    twins = _scene_pair(rng, palette, 2, (1, 2), witness=True)
    puzzle = dict(train=twins, test=[])
    seen = {_key(p['input']) for p in twins}
    for split, count in (('train', num_train-2), ('test', num_test)):
        for i in range(count):
            depth = 2 if split == 'train' else 3
            for _ in range(20):
                pair = _scene_pair(rng, _palette(rng, base, stable), depth,
                                   (int(rng.integers(1, 3)),))[0]
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct scene search exhausted")
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    return puzzle


def _topology(info):
    objects, parents = info['objects'], info['parents']
    def children(p):
        return tuple(sorted(('C', children(i)) if o['contour'] else ('B', o['role'])
                            for i, o in enumerate(objects) if parents[i] == p))
    return children(None)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen = set()
    targets = {}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _analyze(inp)
            ti = info['target']
            _require(np.array_equal(solve(inp), out), "incorrect output")
            oi = _parse(out, info['bg'], info['ink'], info['marked'])
            _require(sum(o['contour'] for o in oi['objects']) == sum(o['contour'] for o in ti['objects'])-info['k'],
                     "wrong membrane count")
            _require(sum(not o['contour'] for o in oi['objects']) == sum(not o['contour'] for o in ti['objects']),
                     "lost terminal")
            _require(len(oi['ancestors']) == len(ti['ancestors'])-info['k'], "wrong marked depth")
            for i, o in enumerate(oi['objects']):
                if not o['contour']:
                    _require(len(o['cells']) == 1, "terminal is not a point")
                else:
                    lo, hi = o['lo'], o['hi']
                    local = o['mask'][lo[0]:hi[0]+1, lo[1]:hi[1]+1]
                    _require(np.array_equal(local, _perimeter(local.shape)), "nonrectangular output wall")
                    cs = [c for j, c in enumerate(oi['objects']) if oi['parents'][j] == i]
                    _require(np.array_equal(np.min([c['lo'] for c in cs], 0), lo+2) and
                             np.array_equal(np.max([c['hi'] for c in cs], 0), hi-2), "incorrect frame moat")
            normalized_output = np.array([info['bg'], info['ink'], info['marked']], np.int8)[_normal(oi)]
            _require(np.array_equal(normalized_output, out), "incorrect crop or spacing")
            _require(not np.array_equal(_normal(ti), _normal(oi)), "reference edit has no effect")
            if split == 'train':
                target_key = _key(info['target_panel'])
                targets.setdefault(target_key, []).append((info['k'], _key(out), info))
    witnesses = [entries for entries in targets.values()
                 if len({k for k, _, _ in entries}) >= 2 and len({answer for _, answer, _ in entries}) >= 2]
    _require(bool(witnesses), "training must show reference dependence on an identical target")
    _require(any(sum(o['contour'] for o in info['target']['objects']) > len(info['target']['ancestors'])
                 for entries in witnesses for _, _, info in entries), "training needs an unaffected side enclosure")


def self_test():
    """Bounded 42-seed structural, semantic, and metamorphic checks."""
    puzzles = [generate_puzzle(s) for s in range(42)]
    shapes, answers, trees, blob_shapes, palettes, depths, commands = set(), set(), set(), set(), set(), set(), set()
    nonrectangular = 0
    for seed, puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _analyze(pair['input'])
                ti = info['target']
                shapes.add(pair['input'].shape)
                answers.add(_key(_normal(_parse(pair['output'], info['bg'], info['ink'], info['marked']))))
                trees.add(_topology(ti))
                palettes.add((info['bg'], info['ink'], info['marked']))
                depths.add(len(ti['ancestors']))
                commands.add(info['k'])
                for o in ti['objects']:
                    dims = tuple(o['hi']-o['lo']+1)
                    if not o['contour']:
                        blob_shapes.add(dims)
                    else:
                        lo, hi = o['lo'], o['hi']
                        local = o['mask'][lo[0]:hi[0]+1, lo[1]:hi[1]+1]
                        nonrectangular += not np.array_equal(local, _perimeter(local.shape))
        # Color permutation preserves geometric roles, including frame pairing.
        p = puzzle['test'][0]
        permutation = np.random.default_rng(seed+9201).permutation(10)
        _require(np.array_equal(solve(permutation[p['input']]), permutation[p['output']]), "recolor failure")
    _require(len(shapes) >= 20 and len(answers) >= 35 and len(trees) >= 15, "insufficient structural diversity")
    _require(len(blob_shapes) >= 4 and len(palettes) >= 20 and nonrectangular >= 20, "insufficient metric diversity")
    _require(depths == {2, 3} and commands == {1, 2}, "missing depth/command coverage")
    # Both orientations of the procedural final fallback fit and normalize.
    for axis in (0, 1):
        for depth, witness in ((2, True), (3, False)):
            rng = np.random.default_rng(821+axis+depth)
            ref, target = _fitting_trees(rng, depth, witness, axis)
            rb, raw = _draw(ref), _draw(target)
            ti = _parse(raw, 0, 1, 2)
            _require(np.array_equal(_normal(ti), raw), "fallback input ambiguity")
            for k in (1, 2):
                partner, output = _draw(_release(ref, k)), _draw(_release(target, k))
                _require(np.array_equal(_normal(ti, ti['ancestors'][:k]), output), "fallback release mismatch")
                _pack(rng, [(rb.shape[0]+2, rb.shape[1]+2), (partner.shape[0]+2, partner.shape[1]+2),
                            (raw.shape[0]+2, raw.shape[1]+2)])
    for a, b in ((2, 1), (7, 3), (10, 4)):
        validate_generated_puzzle(generate_puzzle(1000+a*10+b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    repeated = generate_puzzle(0)
    _require(all(_key(x[k]) == _key(y[k]) for s in repeated for x, y in zip(repeated[s], puzzles[0][s])
                 for k in ('input', 'output')), "nondeterminism")
    # Wrong crop, lost atom, and erased surviving side wall must be rejected.
    for mode in range(3):
        broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
        p = broken['train'][0]
        info = _analyze(p['input'])
        if mode == 0:
            p['output'] = np.pad(p['output'], 1, constant_values=info['bg'])
        elif mode == 1:
            p['output'][p['output'] == info['marked']] = info['bg']
        else:
            oi = _parse(p['output'], info['bg'], info['ink'], info['marked'])
            shell = next(o for o in oi['objects'] if o['contour'])
            y, x = shell['lo']
            p['output'][y, x+1] = info['bg']
        try:
            validate_generated_puzzle(broken)
        except ValueError:
            pass
        else:
            raise AssertionError("corrupted result accepted")
    return dict(seeds=42, pairs=210, input_shapes=len(shapes), normalized_answers=len(answers),
                target_topologies=len(trees), blob_shapes=len(blob_shapes), target_palettes=len(palettes),
                nonrectangular_target_contours=nonrectangular, marked_depths=sorted(depths), release_depths=sorted(commands))
