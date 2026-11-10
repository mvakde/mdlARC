"""Inherited stencil rooms. Standard-library/NumPy, with input-only decoding."""
from __future__ import annotations

import json
from collections import deque
import numpy as np


def _components(mask):
    seen = np.zeros(mask.shape, bool)
    groups = []
    h, w = mask.shape
    for r, c in zip(*np.nonzero(mask)):
        if seen[r, c]:
            continue
        seen[r, c] = True
        todo, cells = [(int(r), int(c))], []
        while todo:
            y, x = todo.pop()
            cells.append((y, x))
            for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    todo.append((yy, xx))
        groups.append(np.asarray(cells, dtype=int))
    return groups


def _holes(mask):
    padded = np.pad(mask, 1)
    outside = np.zeros(padded.shape, bool)
    outside[0, 0] = True
    todo = deque([(0, 0)])
    h, w = padded.shape
    while todo:
        y, x = todo.popleft()
        for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
            if 0 <= yy < h and 0 <= xx < w and not padded[yy, xx] and not outside[yy, xx]:
                outside[yy, xx] = True
                todo.append((yy, xx))
    return (~padded & ~outside)[1:-1, 1:-1]


def _record(cells, shape):
    lo, hi = cells.min(0), cells.max(0)
    mask = np.zeros(shape, bool)
    mask[cells[:, 0], cells[:, 1]] = True
    y, x = lo
    yy, xx = hi
    local = mask[y:yy+1, x:xx+1]
    local_hole = _holes(local)
    hole = np.zeros(shape, bool)
    hole[y:yy+1, x:xx+1] = local_hole
    return dict(cells=cells, lo=lo, hi=hi, mask=mask, hole=hole,
                local=local, hole_count=len(_components(local_hole)))


def _owner(obj, loops, skip=None):
    candidates = []
    for i, room in enumerate(loops):
        if i == skip:
            continue
        inside = room['hole'][obj['cells'][:, 0], obj['cells'][:, 1]]
        assert inside.all() or not inside.any(), 'crossing an enclosure'
        if inside.all():
            candidates.append(i)
    return min(candidates, key=lambda i: int(loops[i]['hole'].sum())) if candidates else None


def _try_roles(grid, background, contour, tag, records):
    loops, plates = records[contour], records[tag]
    assert loops and plates and len(loops) == len(plates)
    assert all(o['hole_count'] == 1 for o in loops)
    for plate in plates:
        local = plate['local']
        assert min(local.shape) >= 3
        border = np.ones(local.shape, bool)
        border[1:-1, 1:-1] = False
        assert np.array_equal(local, border), 'swatch border must be a rectangle'
        y, x = plate['lo']
        yy, xx = plate['hi']
        plate['stencil'] = grid[y+1:yy, x+1:xx].copy()
        assert not np.any(np.isin(plate['stencil'], [contour, tag]))
    parents = [_owner(o, loops, i) for i, o in enumerate(loops)]
    plate_owners = [_owner(o, loops) for o in plates]
    assert None not in plate_owners and sorted(plate_owners) == list(range(len(loops)))
    remaining = (grid != background) & (grid != contour) & (grid != tag)
    for plate in plates:
        y, x = plate['lo']
        yy, xx = plate['hi']
        remaining[y:yy+1, x:xx+1] = False
    leaves = []
    for cells in _components(remaining):
        leaf = _record(cells, grid.shape)
        assert leaf['local'].all(), 'carrier must be a solid rectangular miniature'
        y, x = leaf['lo']
        yy, xx = leaf['hi']
        leaf['carrier'] = grid[y:yy+1, x:xx+1].copy()
        assert not np.any(np.isin(leaf['carrier'], [background, contour, tag]))
        leaves.append(leaf)
    assert leaves
    objects = [dict(o, kind='C') for o in loops] + [dict(o, kind='L') for o in leaves]
    parents += [_owner(o, loops) for o in leaves]
    children = {i: [] for i in range(len(loops))}
    roots = []
    for i, parent in enumerate(parents):
        (roots if parent is None else children[parent]).append(i)
    assert all(children.values()), 'every room has a carrier or child room'
    for plate, owner in zip(plates, plate_owners):
        objects[owner]['stencil'] = plate['stencil']
    # Objects must remain distinguishable even with diagonal adjacency.
    labels = np.zeros(grid.shape, int)
    for i, obj in enumerate(objects + plates, 1):
        mask = obj['mask'].copy()
        if i > len(objects):
            y, x = obj['lo']
            yy, xx = obj['hi']
            mask[y:yy+1, x:xx+1] = True
        assert not np.any(labels[mask])
        labels[mask] = i
    for a, b in ((labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:]),
                 (labels[:-1, :-1], labels[1:, 1:]), (labels[:-1, 1:], labels[1:, :-1])):
        assert not np.any((a != 0) & (b != 0) & (a != b)), 'touching objects'
    depths = []
    for i in range(len(objects)):
        path, parent = [], parents[i]
        while parent is not None:
            assert parent not in path, 'containment cycle'
            path.append(parent)
            parent = parents[parent]
        depths.append(len(path))
    return dict(background=background, contour=contour, tag=tag, objects=objects,
                parents=parents, children=children, roots=roots, depths=depths,
                plates=plates, input_shape=grid.shape)


def _decode(input_grid):
    grid = np.asarray(input_grid)
    assert grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer)
    assert all(1 <= n <= 30 for n in grid.shape) and np.all((grid >= 0) & (grid <= 9))
    bg = int(grid[0, 0])
    assert np.all(grid[0] == bg) and np.all(grid[-1] == bg)
    assert np.all(grid[:, 0] == bg) and np.all(grid[:, -1] == bg)
    records = {int(c): [_record(p, grid.shape) for p in _components(grid == c)]
               for c in np.unique(grid) if c != bg}
    closed = [c for c, objects in records.items() if all(o['hole_count'] == 1 for o in objects)]
    possible = []
    for contour in closed:
        for tag in closed:
            if tag == contour:
                continue
            try:
                info = _try_roles(grid, bg, contour, tag, records)
                _layout(info)  # Spatial roles must also be unambiguous.
                possible.append(info)
            except AssertionError:
                continue
    assert len(possible) == 1, 'input must uniquely identify room and swatch roles'
    return possible[0]


def _layout(info, flat_depth=False):
    """Return normalized geometry, without painting any pattern."""
    objects = info['objects']
    cache = {}

    def compose(parts, axis):
        shape = [max(p[0][0] for p in parts), max(p[0][1] for p in parts)]
        shape[axis] = sum(p[0][axis] for p in parts) + len(parts)-1
        boxes, start = {}, 0
        for dims, positions in parts:
            for i, (r, c, h, w) in positions.items():
                boxes[i] = (r + (start if axis == 0 else 0),
                            c + (start if axis == 1 else 0), h, w)
            start += dims[axis]+1
        return tuple(shape), boxes

    def group(ids):
        ids = tuple(sorted(ids))
        if ids in cache:
            return cache[ids]
        if len(ids) == 1:
            i = ids[0]
            obj = objects[i]
            if obj['kind'] == 'L':
                scale = 1 if flat_depth else max(1, info['depths'][i])
                h, w = (v*scale for v in obj['carrier'].shape)
                result = ((h, w), {i: (0, 0, h, w)})
            else:
                (h, w), inner = group(info['children'][i])
                boxes = {j: (r+2, c+2, hh, ww) for j, (r, c, hh, ww) in inner.items()}
                boxes[i] = (0, 0, h+4, w+4)
                result = ((h+4, w+4), boxes)
        else:
            candidates = []
            for axis in (0, 1):
                ordered = sorted(ids, key=lambda i: int(objects[i]['lo'][axis]))
                partitions, end = [], -1
                for i in ordered:
                    if not partitions or objects[i]['lo'][axis] > end+1:
                        partitions.append([])
                    partitions[-1].append(i)
                    end = max(end, int(objects[i]['hi'][axis]))
                if len(partitions) < 2:
                    continue
                anchors = [min(int(objects[i]['lo'][1-axis]) for i in part) for part in partitions]
                if len(set(anchors)) != 1:
                    continue
                try:
                    candidates.append(compose([group(part) for part in partitions], axis))
                except AssertionError:
                    continue
            assert candidates, 'no aligned cardinal decomposition'
            result = candidates[0]
            assert all(candidate == result for candidate in candidates[1:]), 'competing layouts'
        cache[ids] = result
        return result

    shape, boxes = group(info['roots'])
    assert max(shape) <= 30, 'normalized diagram does not fit'
    return shape, boxes


def _ancestors(info, i):
    path, parent = [], info['parents'][i]
    while parent is not None:
        path.append(parent)
        parent = info['parents'][parent]
    return path[::-1]


def _predict(info, mode='normal'):
    shape, boxes = _layout(info, flat_depth=mode == 'flat_depth')
    out = np.full(shape, info['background'], dtype=int)
    for i, obj in enumerate(info['objects']):
        r, c, h, w = boxes[i]
        if obj['kind'] == 'C':
            out[r, c:c+w] = out[r+h-1, c:c+w] = info['contour']
            out[r:r+h, c] = out[r:r+h, c+w-1] = info['contour']
            continue
        carrier = obj['carrier']
        scale = 1 if mode == 'flat_depth' else max(1, info['depths'][i])
        for y in range(h):
            for x in range(w):
                if mode == 'tile_carrier':
                    ink = int(carrier[y % carrier.shape[0], x % carrier.shape[1]])
                else:
                    ink = int(carrier[y//scale, x//scale])
                path = _ancestors(info, i)
                if mode == 'no_stencil':
                    path = []
                elif mode == 'nearest_only':
                    path = path[-1:]
                elif mode == 'outer_only':
                    path = path[:1]
                elif mode == 'global_scope':
                    path = sorted(info['children'], key=lambda j: (info['depths'][j], j))
                elif mode == 'reverse_layers':
                    path = path[::-1]
                for j in path:
                    stencil = info['objects'][j]['stencil']
                    rr, cc, _, _ = boxes[j]
                    yy, xx = r+y-rr-1, c+x-cc-1
                    if mode == 'leaf_phase':
                        yy, xx = y, x
                    elif mode == 'metric_phase':
                        yy = r+y-int(info['objects'][j]['lo'][0])-1
                        xx = c+x-int(info['objects'][j]['lo'][1])-1
                    elif mode == 'enlarge_stencil':
                        factor = info['depths'][j]+1
                        yy, xx = yy//factor, xx//factor
                    overlay = int(stencil[yy % stencil.shape[0], xx % stencil.shape[1]])
                    if overlay != info['background'] or mode == 'opaque_background':
                        ink = overlay
                out[r+y, c+x] = ink
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover roles, actual enclosure, layout and layers solely from input pixels."""
    return _predict(_decode(input_grid))


def _join_arrays(parts, axis, gap, background):
    shape = [max(p.shape[0] for p in parts), max(p.shape[1] for p in parts)]
    shape[axis] = sum(p.shape[axis] for p in parts) + gap*(len(parts)-1)
    result = np.full(shape, background, int)
    starts, start = [], 0
    for part in parts:
        r, c = (start, 0) if axis == 0 else (0, start)
        result[r:r+part.shape[0], c:c+part.shape[1]] = part
        starts.append((r, c))
        start += part.shape[axis]+gap
    return result, starts


def _group(rng, parts, axis=None):
    return parts[0] if len(parts) == 1 else dict(kind='G', axis=int(rng.integers(2)) if axis is None else axis, parts=parts)


def _image(rng, shape, inks):
    h, w = shape
    k = int(rng.integers(1, min(4, len(inks), h*w)+1))
    selected = rng.choice(inks, k, replace=False)
    rr, cc = np.indices(shape)
    style = int(rng.integers(5))
    if style == 0:
        labels = (rr+cc) % k
    elif style == 1:
        labels = np.broadcast_to(rng.integers(k, size=(h, 1)), shape)
    elif style == 2:
        labels = np.broadcast_to(rng.integers(k, size=(1, w)), shape)
    elif style == 3:
        labels = (rr//int(rng.integers(1, h+1)) + cc//int(rng.integers(1, w+1))) % k
    else:
        labels = rng.integers(k, size=shape)
    return selected[labels].astype(int)


def _stencil(rng, shape, colors, informative=False):
    bg, _, _, inks = colors
    stencil = _image(rng, shape, inks)
    rr, cc = np.indices(shape)
    style = int(rng.integers(5))
    if style == 0:
        mask = (rr+cc+int(rng.integers(2))) % 2 == 0
    elif style == 1:
        mask = rr == int(rng.integers(shape[0]))
    elif style == 2:
        mask = cc == int(rng.integers(shape[1]))
    else:
        mask = rng.random(shape) < float(rng.uniform(.2, .8))
    if not informative and rng.random() < .13:
        mask[:] = False
    if informative and mask.size > 1:
        if not mask.any():
            mask.flat[int(rng.integers(mask.size))] = True
        if mask.all():
            mask.flat[int(rng.integers(mask.size))] = False
    return np.where(mask, stencil, bg)


def _scene(rng, colors, witness=False, shift=False):
    serial = [0]
    small = witness or rng.random() < .5

    def leaf(cap=4, shape=None):
        if shape is None:
            cap = min(cap, 2) if small else cap
            shape = tuple(int(rng.integers(1, cap+1)) for _ in range(2))
        i = serial[0]
        serial[0] += 1
        return dict(kind='L', id=i, carrier=_image(rng, shape, colors[3]))

    def room(parts, compact=False, period=None):
        i = serial[0]
        serial[0] += 1
        if period is None:
            period = tuple(int(rng.integers(1, (2 if compact or small else 3)+1)) for _ in range(2))
        return dict(kind='C', id=i, parts=_group(rng, parts),
                    stencil=_stencil(rng, period, colors, informative=witness))

    def random_room(depth):
        count = int(rng.choice([1, 2, 3], p=[.3, .5, .2]))
        parts = [random_room(depth+1) if depth < 3 and rng.random() < .25
                 else leaf(3 if depth > 1 else 4) for _ in range(count)]
        rng.shuffle(parts)
        if count == 3 and rng.random() < .6:
            axis = int(rng.integers(2))
            cut = int(rng.integers(1, 3))
            parts = [_group(rng, parts[:cut], 1-axis), _group(rng, parts[cut:], 1-axis)]
            node = room(parts)
            node['parts']['axis'] = axis
            return node
        return room(parts)

    if witness:
        inner_parts = [leaf(2) for _ in range(1 if rng.random() < .75 else 2)]
        inner = room(inner_parts, compact=True)
        middle_parts = [inner]
        if rng.random() < .8:
            middle_parts.append(leaf(shape=(2, 2)))
        rng.shuffle(middle_parts)
        middle = room(middle_parts, compact=True)
        outer_parts = [middle]
        if rng.random() < .8:
            outer_parts.append(leaf(2))
        rng.shuffle(outer_parts)
        scene = room(outer_parts, compact=True)
    else:
        roots = [random_room(1)]
        if rng.random() < .35:
            roots.append(leaf(3) if rng.random() < .65 else random_room(1))
        rng.shuffle(roots)
        scene = _group(rng, roots)
    if shift:
        # A native period outside training support, on a visibly encoded swatch.
        first = scene if scene['kind'] == 'C' else next(p for p in scene['parts'] if p['kind'] == 'C')
        period = (4, int(rng.integers(1, 4))) if rng.random() < .5 else (int(rng.integers(1, 4)), 4)
        first['stencil'] = _stencil(rng, period, colors, informative=True)
    return scene


def _outline(rng, shape, padding, simple):
    h, w = shape
    filled = np.ones(shape, bool)
    rr, cc = np.indices(shape)
    if not simple:
        top, bottom, left, right = padding
        cuts = [int(rng.integers(0, max(1, min(a, b)-1)))
                for a, b in ((top, left), (top, right), (bottom, left), (bottom, right))]
        filled &= (rr+cc >= cuts[0]) & (rr+w-1-cc >= cuts[1])
        filled &= (h-1-rr+cc >= cuts[2]) & (h-1-rr+w-1-cc >= cuts[3])
        # Shallow bays occupy only the safe corridors outside the contents.
        for side, pad in enumerate(padding):
            if pad < 3 or rng.random() < .45:
                continue
            axis_length = w if side < 2 else h
            a = int(rng.integers(2, axis_length-2))
            b = int(rng.integers(a+1, axis_length-1))
            amount = int(rng.integers(1, pad-1))
            if side == 0:
                filled[:amount, a:b] = False
            elif side == 1:
                filled[h-amount:, a:b] = False
            elif side == 2:
                filled[a:b, :amount] = False
            else:
                filled[a:b, w-amount:] = False
    # An 8-neighbor erosion makes stair corners 4-connected rather than diagonal.
    padded = np.pad(filled, 1)
    interior = np.ones(shape, bool)
    for dy in range(3):
        for dx in range(3):
            interior &= padded[dy:dy+h, dx:dx+w]
    return filled & ~interior


def _metric(node, rng, colors, compact):
    bg, contour, tag, _ = colors
    if node['kind'] == 'L':
        return node['carrier'].copy()
    if node['kind'] == 'G':
        gap = int(rng.integers(1, 3 if compact else 5))
        return _join_arrays([_metric(p, rng, colors, compact) for p in node['parts']], node['axis'], gap, bg)[0]
    child = _metric(node['parts'], rng, colors, compact)
    swatch = np.pad(node['stencil'], 1, constant_values=tag)
    order = [swatch, child] if rng.random() < .5 else [child, swatch]
    axis = int(rng.integers(2))
    content = _join_arrays(order, axis, int(rng.integers(1, 3)), bg)[0]
    padding = [int(rng.choice([2, 3], p=[.78, .22])) if compact else int(rng.integers(2, 5)) for _ in range(4)]
    top, bottom, left, right = padding
    result = np.pad(content, ((top, bottom), (left, right)), constant_values=bg)
    result[_outline(rng, result.shape, padding, simple=rng.random() < .18)] = contour
    return result


def _construct_output(node, colors):
    """Tree-based constructor; prediction separately reconstructs the visible input."""
    bg, contour, _, _ = colors

    def draw(obj, depth, ancestors):
        if obj['kind'] == 'L':
            scale = max(1, depth)
            result = obj['carrier'].repeat(scale, 0).repeat(scale, 1)
            return result, {obj['id']: (0, 0, *result.shape)}, [(obj['id'], ancestors)], {}
        if obj['kind'] == 'G':
            parts = [draw(p, depth, ancestors) for p in obj['parts']]
            result, starts = _join_arrays([p[0] for p in parts], obj['axis'], 1, bg)
            boxes, paths, swatches = {}, [], {}
            for (_, positions, leaves, stencils), (y, x) in zip(parts, starts):
                boxes.update({i: (r+y, c+x, h, w) for i, (r, c, h, w) in positions.items()})
                paths.extend(leaves)
                swatches.update(stencils)
            return result, boxes, paths, swatches
        child, boxes, paths, swatches = draw(obj['parts'], depth+1, ancestors+[obj['id']])
        result = np.pad(child, 2, constant_values=bg)
        result[0, :] = result[-1, :] = contour
        result[:, 0] = result[:, -1] = contour
        boxes = {i: (r+2, c+2, h, w) for i, (r, c, h, w) in boxes.items()}
        boxes[obj['id']] = (0, 0, *result.shape)
        swatches[obj['id']] = obj['stencil']
        return result, boxes, paths, swatches

    result, boxes, paths, stencils = draw(node, 0, [])
    for i, path in paths:
        r, c, h, w = boxes[i]
        tile = result[r:r+h, c:c+w]
        for j in path:
            rr, cc, _, _ = boxes[j]
            stencil = stencils[j]
            layer = stencil[(np.arange(r, r+h)-rr-1)[:, None] % stencil.shape[0],
                            (np.arange(c, c+w)-cc-1)[None, :] % stencil.shape[1]]
            tile[layer != bg] = layer[layer != bg]
    return result


_MUTANTS = ('flat_depth', 'no_stencil', 'tile_carrier', 'nearest_only', 'outer_only',
            'leaf_phase', 'enlarge_stencil', 'reverse_layers', 'opaque_background',
            'global_scope', 'metric_phase')


def _evidence(pair, info=None):
    if info is None:
        info = _decode(pair['input'])
    out, evidence = np.asarray(pair['output']), set()
    for mode in _MUTANTS:
        try:
            alternative = _predict(info, mode)
            if not np.array_equal(out, alternative):
                evidence.add(mode)
        except AssertionError:
            # The variant has no consistent cardinal output with its wrong sizes.
            evidence.add(mode)
    _, boxes = _layout(info)
    for i, obj in enumerate(info['objects']):
        if obj['kind'] != 'L':
            continue
        evidence.add('depth_'+str(max(1, info['depths'][i])))
        r, c, h, w = boxes[i]
        for y in range(h):
            for x in range(w):
                layers = []
                for j in _ancestors(info, i):
                    rr, cc, _, _ = boxes[j]
                    stencil = info['objects'][j]['stencil']
                    layers.append(int(stencil[(r+y-rr-1) % stencil.shape[0], (c+x-cc-1) % stencil.shape[1]]))
                if layers and all(ink == info['background'] for ink in layers):
                    evidence.add('visible_base')
                if len(layers) > 1 and layers[-1] == info['background'] and any(ink != info['background'] for ink in layers[:-1]):
                    evidence.add('inherited_through_hole')
    return evidence


_REQUIRED = frozenset(_MUTANTS + ('depth_1', 'depth_2', 'depth_3', 'visible_base', 'inherited_through_hole'))


def _key(grid):
    array = np.asarray(grid, dtype=np.int8)
    return array.shape, array.tobytes()


def _normalized(grid):
    roles = {}
    array = np.asarray(grid)
    return array.shape, bytes(roles.setdefault(int(c), len(roles)) for c in array.flat)


def _shift_evidence(info):
    """A new period must actually matter, with visible opacity and transparency."""
    normal = _predict(info)
    _, boxes = _layout(info)
    for j, obj in enumerate(info['objects']):
        if obj['kind'] != 'C' or 4 not in obj['stencil'].shape:
            continue
        stencil = obj['stencil']
        if not np.any(stencil == info['background']) or not np.any(stencil != info['background']):
            continue
        axis = stencil.shape.index(4)
        descendants = [i for i, leaf in enumerate(info['objects'])
                       if leaf['kind'] == 'L' and j in _ancestors(info, i)]
        if not any(boxes[i][2+axis] >= 4 for i in descendants):
            continue
        modified = dict(info, objects=[dict(o) for o in info['objects']])
        modified['objects'][j]['stencil'] = np.full(stencil.shape, info['background'], int)
        if np.array_equal(_predict(modified), normal):
            continue
        slices = [slice(None), slice(None)]
        slices[axis] = slice(0, 3)
        modified['objects'][j]['stencil'] = stencil[tuple(slices)]
        if not np.array_equal(_predict(modified), normal):
            return True
    return False


def _palette(rng):
    values = rng.permutation(10).tolist()
    return (values[0], values[1], values[2], values[3:3+int(rng.integers(3, 8))])


def _make_pair(rng, colors, witness=False, shift=False):
    for attempt in range(240):
        node = _scene(rng, colors, witness=witness, shift=shift)
        expected = _construct_output(node, colors)
        if max(expected.shape) > 30:
            continue
        core = _metric(node, rng, colors, compact=witness or attempt >= 4)
        if max(core.shape) > 28:
            continue
        padding = [(1, int(rng.integers(1, min(4, 30-d)+1))) for d in core.shape]
        # Draw asymmetric canvas margins, always with a uniform exterior border.
        pairs = []
        for d, (_, extra) in zip(core.shape, padding):
            total = min(30-d, 1+extra)
            before = int(rng.integers(1, total))
            pairs.append((before, total-before))
        grid = np.pad(core, pairs, constant_values=colors[0])
        try:
            info = _decode(grid)
            if not np.array_equal(_predict(info), expected):
                continue
            if shift and not _shift_evidence(info):
                continue
        except AssertionError:
            continue
        return dict(input=grid, output=expected)
    raise RuntimeError('bounded room sampling exhausted')


def _counts(num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, 'num_train'), (num_test, 1, 4, 'num_test')):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(name+' must be an integer in the legal range')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    colors = _palette(rng)
    stable = rng.random() < .65
    candidates, covers, train = [], [], None
    # Two jointly informative demonstrations suffice even at the count boundary.
    for _ in range(80):
        pair = _make_pair(rng, colors if stable else _palette(rng), witness=rng.random() < .7)
        cover = _evidence(pair)
        for j in rng.permutation(len(candidates)):
            j = int(j)
            if (_REQUIRED <= cover | covers[j]
                    and _normalized(pair['output']) != _normalized(candidates[j]['output'])
                    and _key(pair['input']) != _key(candidates[j]['input'])):
                train = [candidates[j], pair]
                break
        candidates.append(pair)
        covers.append(cover)
        if train is not None:
            break
    if train is None:
        raise RuntimeError('bounded demonstration selection exhausted')
    puzzle, seen = {'train': train, 'test': []}, {_key(p['input']) for p in train}
    for split, count in (('train', num_train), ('test', num_test)):
        while len(puzzle[split]) < count:
            for retry in range(60):
                pair = _make_pair(rng, colors if stable else _palette(rng), shift=split == 'test' and not puzzle['test'])
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('bounded unique input sampling exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, evidence, periods = set(), set(), {'train': set(), 'test': set()}
    visible_shift = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = np.asarray(pair['input']), np.asarray(pair['output'])
            for grid in (inp, out):
                assert grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            assert np.array_equal(out, _predict(info)), 'incorrect printed diagram'
            shape, boxes = _layout(info)
            assert out.shape == shape and not np.array_equal(inp, out)
            occupied = np.zeros(out.shape, bool)
            for i, obj in enumerate(info['objects']):
                r, c, h, w = boxes[i]
                if obj['kind'] == 'C':
                    periods[split].add(obj['stencil'].shape)
                    assert np.all(out[r, c:c+w] == info['contour'])
                    assert np.all(out[r+h-1, c:c+w] == info['contour'])
                    assert np.all(out[r:r+h, c] == info['contour'])
                    assert np.all(out[r:r+h, c+w-1] == info['contour'])
                    occupied[r, c:c+w] = occupied[r+h-1, c:c+w] = True
                    occupied[r:r+h, c] = occupied[r:r+h, c+w-1] = True
                    direct = [boxes[j] for j in info['children'][i]]
                    assert min(b[0] for b in direct) == r+2 and min(b[1] for b in direct) == c+2
                    assert max(b[0]+b[2] for b in direct) == r+h-2
                    assert max(b[1]+b[3] for b in direct) == c+w-2
                else:
                    scale = max(1, info['depths'][i])
                    assert (h, w) == tuple(d*scale for d in obj['carrier'].shape)
                    assert not np.any(out[r:r+h, c:c+w] == info['tag'])
                    occupied[r:r+h, c:c+w] = True
            assert np.all(out[~occupied] == info['background']), 'printing escaped the carriers'
            assert not np.any(out == info['tag']), 'swatch borders survive in answer'
            if split == 'train':
                evidence |= _evidence(pair, info)
            else:
                visible_shift |= _shift_evidence(info)
    assert _REQUIRED <= evidence, 'missing demonstration evidence: '+str(_REQUIRED-evidence)
    assert periods['test']-periods['train'], 'test must introduce a native stencil period'
    assert visible_shift, 'the new test period must visibly affect carrier printing'


def self_test():
    """Bounded semantic, metamorphic, count and diversity checks on 42 seeds."""
    # Hand-worked fixture: independent numeric answer, not a sampler template.
    manual = np.zeros((22, 28), dtype=int)
    def frame(array, r, c, h, w, ink):
        array[r, c:c+w] = array[r+h-1, c:c+w] = ink
        array[r:r+h, c] = array[r:r+h, c+w-1] = ink
    frame(manual, 1, 1, 20, 26, 5)
    frame(manual, 8, 3, 10, 18, 5)
    frame(manual, 3, 3, 3, 5, 8)
    manual[4, 4:7] = [2, 0, 6]
    frame(manual, 10, 5, 3, 4, 8)
    manual[11, 6:8] = [0, 3]
    manual[10, 11:13] = [1, 4]
    expected = np.zeros((10, 12), dtype=int)
    frame(expected, 0, 0, 10, 12, 5)
    frame(expected, 2, 2, 6, 8, 5)
    expected[4:6, 4:8] = [3, 1, 3, 2]
    assert np.array_equal(solve(manual), expected), 'hand-worked inheritance regression'
    md = _decode(manual)
    assert not np.array_equal(_predict(md, 'nearest_only'), expected)
    assert not np.array_equal(_predict(md, 'no_stencil'), expected)
    assert not np.array_equal(_predict(md, 'leaf_phase'), expected)
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, structures, trees, periods, scales, palettes = set(), set(), set(), set(), set(), set()
    room_counts, leaf_counts, nonrect, holes = set(), set(), 0, 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                shapes.add(pair['input'].shape)
                structures.add(_normalized(pair['output']))
                palettes.add((info['background'], info['contour'], info['tag']))
                room_counts.add(len(info['children']))
                leaf_counts.add(len(info['objects'])-len(info['children']))
                scales.update(max(1, info['depths'][i]) for i, o in enumerate(info['objects']) if o['kind'] == 'L')
                periods.update(o['stencil'].shape for o in info['objects'] if o['kind'] == 'C')
                def tree(i):
                    return ('L',) if info['objects'][i]['kind'] == 'L' else ('C', tuple(sorted(tree(j) for j in info['children'][i])))
                trees.add(tuple(sorted(tree(i) for i in info['roots'])))
                for obj in info['objects']:
                    if obj['kind'] == 'C':
                        local = obj['local']
                        nonrect += not (local[0].all() and local[-1].all() and local[:, 0].all() and local[:, -1].all())
                        holes += np.any(obj['stencil'] == info['background'])
    assert len(shapes) >= 30 and len(structures) >= 80
    assert len(trees) >= 12 and len(periods) >= 10 and len(palettes) >= 35
    assert {1, 2, 3} <= scales and len(room_counts) >= 3 and len(leaf_counts) >= 5
    assert nonrect >= 50 and holes >= 100
    assert all(_key(a['input']) == _key(b['input']) and _key(a['output']) == _key(b['output'])
               for s in ('train', 'test') for a, b in zip(puzzles[0][s], generate_puzzle(0)[s]))
    # Recoloring and translating exterior margins cannot alter the recovered rule.
    rng = np.random.default_rng(17303)
    for puzzle in puzzles[:8]:
        pair = puzzle['test'][0]
        inp, out = pair['input'], pair['output']
        permutation = rng.permutation(10)
        assert np.array_equal(solve(permutation[inp]), permutation[out])
        info = _decode(inp)
        # Reconstruct a compact exterior, then add a different safe margin.
        ys, xs = np.nonzero(inp != info['background'])
        cropped = inp[ys.min():ys.max()+1, xs.min():xs.max()+1]
        shifted = np.pad(cropped, ((1, 1), (1, 1)), constant_values=info['background'])
        assert np.array_equal(solve(shifted), out)
        # Changing a plate's ink affects every eligible carrier, not frame geometry.
        altered = inp.copy()
        plate = info['plates'][0]
        y, x = plate['lo']+1
        old = int(altered[y, x])
        replacement = next(c for c in range(10) if c not in (info['background'], info['contour'], info['tag'], old))
        altered[y, x] = replacement
        new_info = _decode(altered)
        assert _layout(new_info) == _layout(info)
    # Redraw fixed topology, carriers and swatches with genuinely different
    # contours, metric gaps and annotation locations; only geometry may change.
    realized = 0
    for _ in range(8):
        colors = _palette(rng)
        node = _scene(rng, colors)
        expected = _construct_output(node, colors)
        if max(expected.shape) > 30:
            continue
        drawings = set()
        for _ in range(12):
            core = _metric(node, rng, colors, compact=rng.random() < .6)
            if max(core.shape) > 28:
                continue
            drawn = np.pad(core, 1, constant_values=colors[0])
            try:
                prediction = solve(drawn)
            except AssertionError:
                continue
            assert np.array_equal(prediction, expected), 'metric geometry changed printing'
            drawings.add(_key(drawn))
        realized += len(drawings) >= 2
    assert realized >= 3, 'insufficient geometry-invariance redraws'
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(892+a*10+b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    damaged = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ('train', 'test')}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output passed')
    return {'seeds': 42, 'input_shapes': len(shapes), 'normalized_outputs': len(structures),
            'containment_trees': len(trees), 'stencil_periods': len(periods),
            'role_palettes': len(palettes), 'nonrectangular_rooms': nonrect}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
