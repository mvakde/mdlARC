"""Scope-aware stamping inside geometrically normalized enclosures.

Only standard-library modules and NumPy are used. No parent code is imported.
"""
from collections import Counter
import numbers
import numpy as np


_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(grid.dtype, np.integer), "cells must be integers")
    _require(np.all((0 <= grid) & (grid <= 9)), "colors must be 0..9")
    return grid.astype(np.int8, copy=True)


def _components(mask):
    seen = np.zeros(mask.shape, bool)
    h, w = mask.shape
    result = []
    for y, x in np.argwhere(mask):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack, cells = [(y, x)], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in _NEIGHBORS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        result.append(np.array(cells, dtype=int))
    return result


def _hole(mask):
    padded = np.pad(mask, 1)
    outside = np.zeros(padded.shape, bool)
    outside[0, 0] = True
    stack = [(0, 0)]
    h, w = padded.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in _NEIGHBORS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not padded[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                stack.append((rr, cc))
    return (~padded & ~outside)[1:-1, 1:-1]


def _key(array):
    array = np.asarray(array, dtype=np.int8)
    return array.shape, array.tobytes()


def _nearest_prototype_conflict(info):
    """Whether examples reject distance lookup by centers and visible pixels.

    This is a demonstration-coverage check, never part of solve's rule.
    Every equally nearest candidate must have the wrong motif for a conflict.
    """
    recs = info['records']
    prototypes = [i for i, rec in enumerate(recs) if rec['kind'] == 'P']
    rejected = [False, False]
    for i, rec in enumerate(recs):
        if rec['kind'] != 'B':
            continue
        assigned = _key(recs[info['templates'][i]]['patch'])
        for metric in (0, 1):
            distances = []
            for j in prototypes:
                positions = (recs[j]['center'][None, :] if metric == 0 else recs[j]['cells'])
                distances.append(int(np.min(np.sum((positions-rec['center'])**2, axis=1))))
            closest = min(distances)
            rejected[metric] |= all(_key(recs[j]['patch']) != assigned
                                    for j, distance in zip(prototypes, distances)
                                    if distance == closest)
    return all(rejected)


def _parse(grid, output=False, background=None):
    """Read actual enclosure, without using bounding boxes as containment."""
    grid = _grid(grid)
    colors, counts = np.unique(grid, return_counts=True)
    if background is None:
        background = int(colors[np.argmax(counts)])
        _require(max(counts) * 2 > grid.size, "input needs a majority background")
    records, labels = [], np.zeros(grid.shape, dtype=int)
    drawing_colors = set()
    for cells in _components(grid != background):
        lo, hi = cells.min(axis=0), cells.max(axis=0)
        support = np.zeros(grid.shape, bool)
        support[cells[:, 0], cells[:, 1]] = True
        palette = set(map(int, grid[cells[:, 0], cells[:, 1]]))
        hole = _hole(support)
        if len(palette) > 1:
            kind = 'P'
            patch = grid[lo[0]:hi[0]+1, lo[1]:hi[1]+1].copy()
            _require(all(d % 2 == 1 for d in patch.shape), "stamp dimensions must be odd")
            _require(np.array_equal(patch != background,
                                    support[lo[0]:hi[0]+1, lo[1]:hi[1]+1]),
                     "a stamp rectangle may not contain another object")
        elif hole.any():
            kind, patch = 'C', None
            _require(len(_components(hole)) == 1, "contours must enclose one region")
            drawing_colors.update(palette)
        else:
            kind, patch = 'B', None
            _require(len(palette) == 1, "blob must be monochrome")
            _require(all(d % 2 == 1 for d in hi-lo+1), "blob boxes must have odd dimensions")
            _require(len(cells) >= .4 * int(np.prod(hi-lo+1)), "blob must be compact")
        index = len(records)
        labels[support] = index + 1
        records.append(dict(kind=kind, lo=lo, hi=hi, cells=cells, support=support,
                            hole=hole, palette=palette, patch=patch, center=(lo+hi)//2))
    _require(len(drawing_colors) == 1, "need one contour drawing color")
    drawing = drawing_colors.pop()
    for rec in records:
        if rec['kind'] == 'P':
            _require(drawing not in rec['palette'], "stamp cannot use the drawing color")
        else:
            _require(rec['palette'] == {drawing}, "all contours and blobs use one drawing color")
    for a, b in ((labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:]),
                 (labels[:-1, :-1], labels[1:, 1:]), (labels[:-1, 1:], labels[1:, :-1])):
        _require(not np.any((a > 0) & (b > 0) & (a != b)), "objects may not touch")
    parents, children = [], [[] for _ in records]
    for i, child in enumerate(records):
        enclosing = []
        for j, parent in enumerate(records):
            if i == j or parent['kind'] != 'C':
                continue
            inside = parent['hole'][child['cells'][:, 0], child['cells'][:, 1]]
            _require(inside.all() or not inside.any(), "crossing enclosures")
            if inside.all():
                enclosing.append(j)
        parents.append(min(enclosing, key=lambda j: int(records[j]['hole'].sum()))
                       if enclosing else None)
    roots = []
    for i, parent in enumerate(parents):
        (roots if parent is None else children[parent]).append(i)
    _require(len(roots) == 1 and records[roots[0]]['kind'] == 'C',
             "one principal contour must contain the entire scene")
    for i, rec in enumerate(records):
        _require(bool(children[i]) == (rec['kind'] == 'C'), "no empty contours")
    templates, scope_templates = {}, {}

    def scopes(i, inherited):
        rec = records[i]
        if rec['kind'] != 'C':
            templates[i] = i if rec['kind'] == 'P' else inherited
            _require(templates[i] is not None, "every blob needs an enclosing template")
            return
        local = [j for j in children[i] if records[j]['kind'] == 'P']
        if not output:
            _require(len(local) <= 1, "at most one immediate prototype per scope")
            current = local[0] if local else inherited
            _require(current is not None, "principal contour needs a prototype")
            scope_templates[i] = current
            for j in children[i]:
                scopes(j, current)
        else:
            for j in children[i]:
                if records[j]['kind'] == 'C':
                    scopes(j, None)
                else:
                    _require(records[j]['kind'] == 'P', "normalized output has no drawing-colored blobs")
                    templates[j] = j
    scopes(roots[0], None)
    return dict(grid=grid, bg=int(background), fg=int(drawing), records=records,
                parents=parents, children=children, root=roots[0],
                templates=templates, scope_templates=scope_templates)


def _tile(grid, anchors):
    return grid, anchors


def _join(tiles, axis, offsets, gaps, bg):
    cross = 1-axis
    low = min(offsets)
    starts, along, shape = [], 0, [0, 0]
    for i, (grid, _) in enumerate(tiles):
        pos = [0, 0]
        pos[axis], pos[cross] = along, offsets[i]-low
        starts.append(pos)
        along += grid.shape[axis]
        if i < len(gaps):
            along += gaps[i]
        shape[cross] = max(shape[cross], pos[cross]+grid.shape[cross])
    shape[axis] = along
    result, anchors = np.full(shape, bg, np.int8), []
    for (grid, leaves), (y, x) in zip(tiles, starts):
        h, w = grid.shape
        result[y:y+h, x:x+w] = grid
        anchors.extend((i, r+y, c+x) for i, r, c in leaves)
    return _tile(result, anchors)


def _frame(tile, fg, bg):
    grid, leaves = tile
    result = np.pad(grid, 2, constant_values=bg)
    result[0, :] = result[-1, :] = fg
    result[:, 0] = result[:, -1] = fg
    return _tile(result, [(i, r+2, c+2) for i, r, c in leaves])


def _normalize(info, *, global_stamp=False, point_blobs=False):
    """Reconstruct using cardinal cuts and exact, visible terminal-center anchors.

    The optional flags are ablation checks only. Public solve always uses the
    scoped, exact-stamp rule.
    """
    recs, cache = info['records'], {}
    bg, fg = info['bg'], info['fg']
    global_id = info['scope_templates'][info['root']]

    def align(tiles, axis):
        cross, edges = 1-axis, []
        for i in range(len(tiles)):
            for j in range(i+1, len(tiles)):
                shifts = set()
                for a, ar, ac in tiles[i][1]:
                    for b, br, bc in tiles[j][1]:
                        if recs[a]['center'][cross] == recs[b]['center'][cross]:
                            shifts.add((ar, ac)[cross] - (br, bc)[cross])
                _require(len(shifts) <= 1, "competing alignment anchors")
                if shifts:
                    edges.append((i, j, shifts.pop()))
        offsets = {0: 0}
        for _ in tiles:
            for i, j, delta in edges:
                if i in offsets and j in offsets:
                    _require(offsets[j] == offsets[i]+delta, "inconsistent alignment")
                elif i in offsets:
                    offsets[j] = offsets[i]+delta
                elif j in offsets:
                    offsets[i] = offsets[j]-delta
        _require(len(offsets) == len(tiles), "unanchored diagonal layout")
        return [offsets[i] for i in range(len(tiles))]

    def group(ids):
        ids = tuple(sorted(ids))
        if ids in cache:
            return cache[ids]
        if len(ids) == 1:
            i = ids[0]
            rec = recs[i]
            if rec['kind'] == 'C':
                result = _frame(group(info['children'][i]), fg, bg)
            else:
                if rec['kind'] == 'B' and point_blobs:
                    patch = np.array([[fg]], np.int8)
                else:
                    template = (global_id if global_stamp and rec['kind'] == 'B'
                                else info['templates'][i])
                    patch = recs[template]['patch'].copy()
                h, w = patch.shape
                result = _tile(patch, [(i, h//2, w//2)])
        else:
            candidates = []
            for axis in (0, 1):
                # A terminal encodes a center site: its footprint must not
                # determine the spatial decomposition. Contours retain their
                # enclosing intervals until their contents are reconstructed.
                def bounds(i):
                    rec = recs[i]
                    if rec['kind'] == 'C':
                        return int(rec['lo'][axis]), int(rec['hi'][axis])
                    center = int(rec['center'][axis])
                    return center, center

                ordered = sorted(ids, key=lambda i: bounds(i)[0])
                parts, end = [], -100
                for i in ordered:
                    start, stop = bounds(i)
                    if start > end+1:
                        parts.append([])
                    parts[-1].append(i)
                    end = max(end, stop)
                if len(parts) < 2:
                    continue
                try:
                    tiles = [group(part) for part in parts]
                    candidates.append(_join(tiles, axis, align(tiles, axis),
                                            [1]*(len(tiles)-1), bg))
                except ValueError:
                    continue
            _require(bool(candidates), "no consistent cardinal layout")
            result = candidates[0]
            for other in candidates[1:]:
                _require(np.array_equal(result[0], other[0]) and
                         sorted(result[1]) == sorted(other[1]), "ambiguous layout")
        cache[ids] = result
        return result
    return group([info['root']])[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Compute the output from visible pixels alone."""
    result = _normalize(_parse(input_grid))
    _require(all(1 <= n <= 30 for n in result.shape), "normalized result exceeds 30")
    return result


def _symmetries(grid):
    return [f(np.rot90(grid, k)) for k in range(4) for f in (lambda x: x, np.fliplr)]


def _asymmetric(grid):
    return all(not np.array_equal(grid, other) for other in _symmetries(grid)[1:])


def _stamp(rng, shape, bg, payload, witness=False):
    h, w = shape
    rr, cc = np.indices(shape)
    for _ in range(40):
        style = int(rng.integers(5))
        if style == 0:
            labels = rr+cc
        elif style == 1:
            labels = rr if rng.integers(2) else cc
        elif style == 2:
            labels = np.minimum.reduce([rr, cc, h-1-rr, w-1-cc])
        elif style == 3:
            labels = abs(rr-h//2)+abs(cc-w//2)
        else:
            labels = rng.integers(len(payload), size=shape)
        patch = np.asarray(payload, np.int8)[(labels+int(rng.integers(len(payload)))) % len(payload)].copy()
        if min(shape) >= 3 and (witness or rng.random() < .65):
            patch[h//2, w//2] = bg
            if not witness and rng.random() < .35:
                patch[int(rng.integers(h)), int(rng.integers(w))] = bg
        visible = patch != bg
        if len(_components(visible)) != 1 or not all(s.any() for s in
                (visible[0], visible[-1], visible[:, 0], visible[:, -1])):
            continue
        if len(set(map(int, patch[visible]))) < 2:
            continue
        if witness and not _asymmetric(patch):
            # Perturb an actual motif, with no compulsory palette-advertising row.
            patch[0, 1] = payload[-1]
            patch[-1, 0] = payload[0]
            if not _asymmetric(patch):
                continue
        return patch
    # A constructive exact motif with the same requested dimensions.
    patch = rng.choice(payload, size=shape).astype(np.int8)
    patch[0, 0], patch[-1, -1] = payload[0], payload[1]
    if witness:
        patch[h//2, w//2] = bg
        # Witness calls use 3x3. This fixed fallback only covers an exhausted
        # motif search, and has no hidden semantic role.
        patch = np.array([[payload[0], payload[0], payload[1]],
                          [payload[1], bg, payload[0]],
                          [payload[0], payload[1], payload[1]]], dtype=np.int8)
        assert _asymmetric(patch)
    return patch


def _blob(rng, compact=False):
    shapes = [(1, 1), (1, 3), (3, 1), (3, 3)]
    if not compact:
        shapes += [(3, 5), (5, 3)]
    shape = shapes[int(rng.integers(len(shapes)))]
    mask = np.ones(shape, bool)
    if min(shape) > 1 and rng.random() < .65:
        # Corner peeling changes shape, while retaining odd tight dimensions.
        for r, c in ((0, 0), (0, shape[1]-1), (shape[0]-1, 0), (shape[0]-1, shape[1]-1)):
            if rng.random() < .5:
                mask[r, c] = False
    return mask


def _outline(rng, shape, padding, simple=False):
    h, w = shape
    caps = [n-2 for n in padding]
    top, bottom, left, right = caps

    def profile(length, cap):
        if cap == 0 or simple:
            return np.zeros(length, int)
        # Independent straight, tapered, bulging, and stepped sides.
        t = np.linspace(0, 1, length)
        style = int(rng.integers(4))
        values = (t if style == 0 else abs(2*t-1) if style == 1 else
                  1-abs(2*t-1) if style == 2 else
                  np.interp(t, np.linspace(0, 1, 4), rng.random(4)))
        return np.rint(values*cap).astype(int)

    t = profile(w, top)
    b = h-1-profile(w, bottom)
    l = profile(h, left)
    r = w-1-profile(h, right)
    x0, x1, y0, y1 = left, w-1-right, top, h-1-bottom
    path = [(int(t[x]), x) for x in range(x0, x1+1)]
    path += [(y0, x1), (y0, int(r[y0]))]
    path += [(y, int(r[y])) for y in range(y0, y1+1)]
    path += [(int(b[x1]), int(r[y1])), (int(b[x1]), x1)]
    path += [(int(b[x]), x) for x in range(x1, x0-1, -1)]
    path += [(y1, x0), (y1, int(l[y1]))]
    path += [(y, int(l[y])) for y in range(y1, y0-1, -1)]
    path += [(int(t[x0]), int(l[y0])), path[0]]
    mask = np.zeros(shape, bool)
    for (y, x), (yy, xx) in zip(path, path[1:]):
        mask[min(y, yy):max(y, yy)+1, x] = True
        mask[yy, min(x, xx):max(x, xx)+1] = True
    return mask


def _leaf_count(node):
    if node[0] in ('B', 'P'):
        return 1
    if node[0] == 'C':
        return _leaf_count(node[1])
    return sum(_leaf_count(child) for child in node[2])


def _group(rng, nodes, mixed=False):
    nodes = list(nodes)
    rng.shuffle(nodes)
    if len(nodes) == 1:
        return nodes[0]
    axis = int(rng.integers(2))

    def join(parts, direction):
        if len(parts) == 1:
            return parts[0]
        return ('G', direction, tuple(parts),
                tuple(int(rng.integers(_leaf_count(c))) for c in parts))
    if len(nodes) >= 3 and (mixed or rng.random() < .5):
        cut = int(rng.integers(1, len(nodes)))
        nodes = [join(nodes[:cut], 1-axis), join(nodes[cut:], 1-axis)]
    return join(nodes, axis)


def _immediate_prototypes(node):
    if node[0] == 'P':
        return [node[1]]
    if node[0] in ('B', 'C'):
        return []
    return [p for c in node[2] for p in _immediate_prototypes(c)]


def _scene(rng, bg, payload, kind):
    """Generate scope structure, independent of distorted metric rendering."""
    shape_options = [(1, 3), (3, 1), (3, 3), (3, 5), (5, 3)]

    def proto(witness=False, small=False):
        shape = (3, 3) if witness else shape_options[int(rng.integers(3 if small else len(shape_options)))]
        return ('P', _stamp(rng, shape, bg, payload, witness))

    def contour(children, mixed=False):
        return ('C', _group(rng, children, mixed))

    def distinct(parent):
        for _ in range(20):
            child = proto(small=True)
            if _key(parent[1]) != _key(child[1]):
                return child
        patch = parent[1].copy()
        colors, counts = np.unique(patch[patch != bg], return_counts=True)
        old = int(colors[np.argmax(counts)])
        new = int(next(c for c in colors if c != old))
        y, x = map(int, np.argwhere(patch == old)[0])
        patch[y, x] = new
        return ('P', patch)

    b = ('B',)
    if kind == 'inherit':
        child = contour([b] * int(rng.integers(1, 3)))
        return contour([proto(True), b, child], True)
    if kind == 'override':
        parent = proto(small=True)
        child_proto = distinct(parent)
        child = contour([child_proto, b, b], True)
        return contour([parent, b, child], True)
    if kind == 'branch_test':
        p0 = proto(small=True)
        p1 = distinct(p0)
        inheritor = contour([b] * int(rng.integers(1, 3)))
        override = contour([p1, b])
        return contour([p0, b, inheritor, override], True)
    if kind == 'deep_test':
        p0 = proto(small=True)
        p1 = distinct(p0)
        inner = contour([p1, b])
        middle = contour([b, inner])
        return contour([p0, b, middle], True)
    # Extra demonstrations vary leaf counts, nesting position, mixed groups,
    # prototype size, and whether a nested region inherits or overrides.
    nodes = [proto(), *([b] * int(rng.integers(1, 4)))]
    if rng.random() < .7:
        child_nodes = [b] * int(rng.integers(1, 4))
        if rng.random() < .55:
            child_nodes.append(proto())
        nodes.append(contour(child_nodes))
    return contour(nodes, bool(rng.integers(2)))


def _draw(scene, rng, bg, fg, compact=False):
    serial = [0]

    def visit(node, inherited):
        kind = node[0]
        if kind in ('B', 'P'):
            i = serial[0]
            serial[0] += 1
            if kind == 'P':
                inp = out = node[1].copy()
            else:
                mask = _blob(rng, compact)
                inp = np.where(mask, fg, bg).astype(np.int8)
                out = inherited.copy()
            return (_tile(inp, [(i, inp.shape[0]//2, inp.shape[1]//2)]),
                    _tile(out, [(i, out.shape[0]//2, out.shape[1]//2)]))
        if kind == 'G':
            pairs = [visit(child, inherited) for child in node[2]]
            results = []
            for version in (0, 1):
                parts = [pair[version] for pair in pairs]
                offsets = [-part[1][a][1+(1-node[1])] for part, a in zip(parts, node[3])]
                gaps = ([int(rng.integers(1, 3 if compact else 5)) for _ in parts[:-1]]
                        if version == 0 else [1]*(len(parts)-1))
                results.append(_join(parts, node[1], offsets, gaps, bg))
            return tuple(results)
        local = _immediate_prototypes(node[1])
        current = local[0] if local else inherited
        inp, out = visit(node[1], current)
        padding = [int(rng.integers(2, 4 if compact else 6)) for _ in range(4)]
        t, b, l, r = padding
        h, w = inp[0].shape
        outline = _outline(rng, (h+t+b, w+l+r), padding, rng.random() < .12)
        grid = np.where(outline, fg, bg).astype(np.int8)
        grid[t:t+h, l:l+w] = inp[0]
        metric = _tile(grid, [(i, y+t, x+l) for i, y, x in inp[1]])
        return metric, _frame(out, fg, bg)
    metric, canonical = visit(scene, None)
    return metric[0], canonical[0]


def _make_pair(rng, bg, fg, payload, kind, seen):
    for attempt in range(240):
        scene = _scene(rng, bg, payload, kind)
        inp, out = _draw(scene, rng, bg, fg, compact=attempt >= 8)
        if max(*inp.shape, *out.shape) > 30:
            continue
        # Canvas padding and placement are independent of the rule and split.
        h, w = inp.shape
        eh, ew = int(rng.integers(min(5, 30-h)+1)), int(rng.integers(min(5, 30-w)+1))
        y, x = int(rng.integers(eh+1)), int(rng.integers(ew+1))
        canvas = np.full((h+eh, w+ew), bg, np.int8)
        canvas[y:y+h, x:x+w] = inp
        if _key(canvas) in seen or (canvas == bg).sum()*2 <= canvas.size:
            continue
        try:
            info = _parse(canvas)
            if not np.array_equal(_normalize(info), out):
                continue
            if kind == 'override' and np.array_equal(_normalize(info, global_stamp=True), out):
                continue
            if kind == 'override' and not _nearest_prototype_conflict(info):
                continue
        except ValueError:
            continue
        return dict(input=canvas, output=out)
    raise RuntimeError('bounded construction failed for '+kind)


def _counts(num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or not lo <= int(value) <= hi:
            raise ValueError(name+' is outside its legal integer range')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    base_palette = rng.permutation(10)
    palette_mode = int(rng.integers(3))  # stable, role exchange, or independent
    shift = int(rng.integers(2))
    puzzle, seen, index = dict(train=[], test=[]), set(), 0
    for split, count in (('train', ntrain), ('test', ntest)):
        for i in range(count):
            palette = (base_palette if palette_mode == 0 else np.roll(base_palette, index)
                       if palette_mode == 1 else rng.permutation(10))
            bg, fg = map(int, palette[:2])
            payload = [int(c) for c in palette[2:2+int(rng.integers(2, 5))]]
            kind = ('inherit' if i == 0 else 'override' if i == 1 else 'free') if split == 'train' else (
                'branch_test' if (i+shift) % 2 == 0 else 'deep_test')
            pair = _make_pair(rng, bg, fg, payload, kind, seen)
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
            index += 1
        rng.shuffle(puzzle[split])
    return puzzle


def _tree_contents(info):
    recs = info['records']

    def visit(i):
        if recs[i]['kind'] == 'C':
            return ('C', tuple(sorted((visit(j) for j in info['children'][i]), key=repr)))
        patch = recs[info['templates'][i]]['patch']
        return ('S', _key(patch))
    return visit(info['root'])


def _features(info):
    recs = info['records']
    contours = [i for i, r in enumerate(recs) if r['kind'] == 'C']
    blobs = [i for i, r in enumerate(recs) if r['kind'] == 'B']
    prototypes = [i for i, r in enumerate(recs) if r['kind'] == 'P']
    depths, inherited, overridden, nonrect, mixed = [], False, False, 0, False
    for i in contours:
        parent, depth = info['parents'][i], 1
        while parent is not None:
            depth += 1
            parent = info['parents'][parent]
        depths.append(depth)
        local = [j for j in info['children'][i] if recs[j]['kind'] == 'P']
        direct_blobs = [j for j in info['children'][i] if recs[j]['kind'] == 'B']
        inherited |= bool(not local and direct_blobs)
        parent = info['parents'][i]
        if local and parent is not None and direct_blobs:
            overridden |= _key(recs[local[0]]['patch']) != _key(recs[info['scope_templates'][parent]]['patch'])
        r = recs[i]
        y, x = r['lo']; yy, xx = r['hi']
        m = r['support'][y:yy+1, x:xx+1]
        nonrect += not (m[0].all() and m[-1].all() and m[:, 0].all() and m[:, -1].all())
        ids = info['children'][i]
        if len(ids) >= 3:
            centers = np.array([recs[j]['center'] for j in ids])
            mixed |= bool(np.ptp(centers[:, 0]) and np.ptp(centers[:, 1]))
    return dict(contours=len(contours), blobs=len(blobs), prototypes=len(prototypes),
                depth=max(depths), inherited=bool(inherited), overridden=bool(overridden),
                nonrect=int(nonrect), mixed=bool(mixed),
                holes=any(np.any(recs[i]['patch'] == info['bg']) for i in prototypes),
                asymmetric=any(_asymmetric(recs[i]['patch']) for i in prototypes),
                nearest_conflict=_nearest_prototype_conflict(info),
                stamp_shapes=tuple(recs[i]['patch'].shape for i in prototypes),
                blob_shapes=tuple(tuple(map(int, recs[i]['hi']-recs[i]['lo']+1)) for i in blobs))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'exact puzzle keys required')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, train = set(), []
    omit_a, omit_b = False, False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'exact pair keys required')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            info = _parse(inp)
            expected = _normalize(info)
            _require(np.array_equal(out, expected), 'incorrect scoped substitution or layout')
            normalized = _parse(out, output=True, background=info['bg'])
            _require(_tree_contents(info) == _tree_contents(normalized), 'wrong motif multiplicities or scope tree')
            root = normalized['records'][normalized['root']]
            _require(np.array_equal(root['lo'], (0, 0)) and
                     np.array_equal(root['hi'], np.array(out.shape)-1), 'output crop must be tight')
            for i, rec in enumerate(normalized['records']):
                if rec['kind'] != 'C':
                    continue
                y, x = rec['lo']; yy, xx = rec['hi']
                mask = rec['support'][y:yy+1, x:xx+1]
                _require(mask[0].all() and mask[-1].all() and mask[:, 0].all() and mask[:, -1].all() and
                         int(mask.sum()) == 2*(yy-y+xx-x), 'output contour must be a thin rectangle')
                kids = [normalized['records'][j] for j in normalized['children'][i]]
                _require(np.array_equal(np.min([k['lo'] for k in kids], axis=0), rec['lo']+2) and
                         np.array_equal(np.max([k['hi'] for k in kids], axis=0), rec['hi']-2),
                         'frame must fit its contents with one background cell of clearance')
            if split == 'train':
                train.append(_features(info))
                try:
                    omit_a |= not np.array_equal(out, _normalize(info, global_stamp=True))
                except ValueError:
                    omit_a = True
                try:
                    omit_b |= not np.array_equal(out, _normalize(info, point_blobs=True))
                except ValueError:
                    omit_b = True
    _require(any(f['inherited'] for f in train), 'training must demonstrate inheritance')
    _require(any(f['overridden'] for f in train), 'training must demonstrate an override')
    _require(any(f['nearest_conflict'] for f in train), 'training must reject nearest-stamp shortcuts')
    _require(any(f['mixed'] for f in train), 'training must demonstrate both layout axes')
    _require(any(f['holes'] and f['asymmetric'] for f in train), 'training must fix hole and orientation conventions')
    _require(omit_a and omit_b, 'training must distinguish both parent omissions')


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s]) for s in ('train', 'test'))


def self_test():
    """Bounded 42-seed checks, semantic counterexamples, and metamorphic checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    dimensions, layouts, stamp_shapes, blob_shapes, depths, counts = set(), set(), set(), set(), set(), set()
    nonrect, stable, varied = 0, 0, 0
    for puzzle in puzzles:
        palettes = set()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _parse(pair['input'])
                f = _features(info)
                dimensions.add(pair['input'].shape)
                layouts.add(_key(pair['output'] == info['fg']))
                stamp_shapes.update(f['stamp_shapes'])
                blob_shapes.update(f['blob_shapes'])
                depths.add(f['depth'])
                counts.add((f['contours'], f['blobs'], f['prototypes']))
                palettes.add((info['bg'], info['fg']))
                nonrect += f['nonrect']
        stable += len(palettes) == 1
        varied += len(palettes) > 1
    assert len(dimensions) >= 35 and len(layouts) >= 50
    assert len(stamp_shapes) >= 5 and len(blob_shapes) >= 5
    assert depths == {1, 2, 3} and len(counts) >= 10
    assert nonrect >= 60 and stable >= 3 and varied >= 10
    # Recoloring and translation must preserve the role-based answer.
    rng = np.random.default_rng(1607)
    for puzzle in puzzles[:8]:
        pair = puzzle['test'][0]
        permutation = rng.permutation(10)
        assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
        inp, bg = pair['input'], _parse(pair['input'])['bg']
        if max(inp.shape) < 30:
            shifted = np.pad(inp, ((1, 0), (0, 1)), constant_values=bg)
            assert np.array_equal(solve(shifted), pair['output'])
        # Blob area is discarded; keep its center, scope, and stamp unchanged.
        collapsed = inp.copy()
        info = _parse(inp)
        for rec in info['records']:
            if rec['kind'] == 'B':
                collapsed[rec['support']] = bg
                collapsed[tuple(rec['center'])] = info['fg']
        assert np.array_equal(solve(collapsed), pair['output'])
    # Pixel-based ablations are witnessed on demonstrations, not just on tests.
    witness = puzzles[0]
    assert any(not np.array_equal(_normalize(_parse(p['input']), global_stamp=True), p['output'])
               for p in witness['train'])
    assert any(not np.array_equal(_normalize(_parse(p['input']), point_blobs=True), p['output'])
               for p in witness['train'])
    # The supplied host verifier covers every legal combination. Here keep
    # the expensive semantic self-test bounded while exercising boundaries.
    for a, b in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(8128, a, b)
        validate_generated_puzzle(p)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # A changed motif cell must fail even if dimensions and palettes stay valid.
    damaged = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in witness[s]] for s in witness}
    cell = np.argwhere(damaged['test'][0]['output'] != _parse(damaged['test'][0]['input'])['fg'])[0]
    y, x = map(int, cell)
    damaged['test'][0]['output'][y, x] = (int(damaged['test'][0]['output'][y, x])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError('incorrect output accepted')
    return dict(seeds=42, input_shapes=len(dimensions), frame_layouts=len(layouts),
                stamp_shapes=sorted(stamp_shapes), blob_shapes=sorted(blob_shapes),
                depth_values=sorted(depths), count_combinations=len(counts),
                nonrectangular_contours=nonrect, stable_palettes=stable, varied_palettes=varied)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
