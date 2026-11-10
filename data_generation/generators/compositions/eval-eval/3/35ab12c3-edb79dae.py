"""Sparse routes cast stencil-shaped, marker-displaced companion shadows.

Only Python's standard library and NumPy are used. No parent module or fixture
is needed at runtime. See task_spec.md for the fixed input-only conventions.
"""
from __future__ import annotations

import json
from itertools import combinations

import numpy as np


OFFSETS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _box(points):
    a = np.asarray(list(points), dtype=int)
    assert len(a)
    return (int(a[:, 0].min()), int(a[:, 0].max()) + 1,
            int(a[:, 1].min()), int(a[:, 1].max()) + 1)


def _line(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    length = max(abs(dr), abs(dc))
    if not length or not (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
        return ()
    sr, sc = int(np.sign(dr)), int(np.sign(dc))
    return tuple((a[0] + i * sr, a[1] + i * sc) for i in range(length + 1))


def _cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def _intersect(a, b, c, d):
    if (max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0])
            or max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1])):
        return False
    return _cross(a, b, c)*_cross(a, b, d) <= 0 and _cross(c, d, a)*_cross(c, d, b) <= 0


def _route(points):
    """Infer an undirected edge set: unique simple cycle, else unique simple path.

    Continuous segment crossings (including half-pixel diagonal crossings) are
    forbidden. Reversed traversal and different cycle starting points are ties
    for the same answer, not competing topologies.
    """
    points = sorted(set(tuple(map(int, p)) for p in points))
    n = len(points)
    assert 2 <= n <= 10, "routes need 2..10 distinct landmarks"
    landmarks = set(points)
    edges, rasters = [], []
    for i, j in combinations(range(n), 2):
        line = _line(points[i], points[j])
        if line and not (set(line[1:-1]) & landmarks):
            edges.append((i, j))
            rasters.append(set(line))
    neighbors = [[] for _ in points]
    for e, (i, j) in enumerate(edges):
        neighbors[i].append((j, e))
        neighbors[j].append((i, e))
    assert all(neighbors), "isolated source landmark"
    conflict = [set() for _ in edges]
    for e, (a, b) in enumerate(edges):
        for f, (c, d) in enumerate(edges[:e]):
            if not ({a, b} & {c, d}) and _intersect(points[a], points[b], points[c], points[d]):
                conflict[e].add(f)
                conflict[f].add(e)

    def search(closed):
        found = {}

        def visit(order, used, selected):
            if len(found) > 1:
                return
            if len(order) == n:
                chosen = selected
                if closed:
                    closing = [e for j, e in neighbors[order[-1]] if j == order[0]]
                    if not closing or conflict[closing[0]] & set(selected):
                        return
                    chosen = selected + [closing[0]]
                found[tuple(sorted(chosen))] = tuple(order)
                return
            for j, e in neighbors[order[-1]]:
                if j not in used and not (conflict[e] & set(selected)):
                    visit(order + [j], used | {j}, selected + [e])

        for start in ([0] if closed else range(n)):
            visit([start], {start}, [])
            if len(found) > 1:
                break
        return found

    cycles = search(True) if n >= 3 else {}
    assert len(cycles) <= 1, "multiple simple full contours"
    solutions = cycles or search(False)
    assert len(solutions) == 1, "route has no unique preferred completion"
    selected, order = next(iter(solutions.items()))
    pixels = set().union(*(rasters[e] for e in selected))
    all_pixels = set().union(*rasters)
    polygon = [points[i] for i in order]
    turns = [_cross(polygon[i-1], polygon[i], polygon[(i+1) % n]) for i in range(n)] if cycles else []
    return dict(points=points, pixels=pixels, kind="cycle" if cycles else "path",
                edges=[edges[e] for e in selected], order=polygon,
                rejected=all_pixels-pixels, concave=any(x < 0 for x in turns) and any(x > 0 for x in turns),
                lengths=[len(rasters[e])-1 for e in selected],
                diagonal=any(points[i][0] != points[j][0] and points[i][1] != points[j][1]
                             for i, j in (edges[e] for e in selected)))


def _sweep(route, mask, offset):
    radius = mask.shape[0] // 2
    support = [(int(r)-radius, int(c)-radius) for r, c in np.argwhere(mask)]
    dr, dc = offset
    return {(r+dr+a, c+dc+b) for r, c in route for a, b in support}


def _frame(grid):
    candidates = []
    for color in np.unique(grid):
        cells = np.argwhere(grid == color)
        r, e, c, d = _box(cells)
        h, w = e-r, d-c
        if h < 5 or w < 5 or len(cells) != 2*h+2*w-4:
            continue
        if (np.all(grid[r, c:d] == color) and np.all(grid[e-1, c:d] == color)
                and np.all(grid[r:e, c] == color) and np.all(grid[r:e, d-1] == color)):
            candidates.append(((r, e, c, d), int(color)))
    assert len(candidates) == 1, "need one exclusive frame color"
    return candidates[0]


def _parse(input_grid):
    grid = _grid(input_grid)
    frame, frame_color = _frame(grid)
    r, e, c, d = frame
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[np.argmax(counts)])
    assert int(max(counts)) > grid.size // 2, "background must be a strict majority"
    assert background != frame_color
    exterior = np.ones(grid.shape, dtype=bool)
    exterior[r:e, c:d] = False
    sources = sorted(int(x) for x in np.unique(grid[exterior]) if x != background)
    assert 1 <= len(sources) <= 3
    brushes, brush_boxes = {}, {}
    for source in sources:
        a, b, x, y = _box(np.argwhere(exterior & (grid == source)))
        assert b-a == y-x and b-a in (1, 3, 5)
        assert b < r or a > e or y < c or x > d, "brush needs a blank gap from frame"
        tile = grid[a:b, x:y]
        assert np.all(np.isin(tile, [source, background]))
        mask = tile == source
        assert mask[len(mask)//2, len(mask)//2]
        assert mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()
        brushes[source], brush_boxes[source] = mask.copy(), (a, b, x, y)
    boxes = list(brush_boxes.values())
    for i, (a, b, x, y) in enumerate(boxes):
        for aa, bb, xx, yy in boxes[i+1:]:
            assert b < aa or bb < a or y < xx or yy < x, "legend bounds need blank gaps"
    interior = grid[r+1:e-1, c+1:d-1]
    routes = []
    for source in sources:
        points = [(int(a)+r+1, int(b)+c+1) for a, b in np.argwhere(interior == source)]
        route = _route(points)
        route.update(source=source, brush=brushes[source])
        assert len(route['pixels']-set(points)) > 0, "each route must have a gap"
        routes.append(route)
    markers = [(int(a)+r+1, int(b)+c+1) for a, b in np.argwhere(
        ~np.isin(interior, [background] + sources))]
    assert len(markers) == len(routes), "one dot per route"
    assigned = set()
    for marker in markers:
        matches = [(i, p) for i, route in enumerate(routes) for p in route['points']
                   if max(abs(p[0]-marker[0]), abs(p[1]-marker[1])) == 1]
        assert len(matches) == 1, "dot must identify exactly one source vertex"
        i, anchor = matches[0]
        assert i not in assigned
        assigned.add(i)
        delta = marker[0]-anchor[0], marker[1]-anchor[1]
        routes[i].update(marker=marker, offset=delta, target=int(grid[marker]))
        routes[i]['shadow'] = _sweep(routes[i]['pixels'], routes[i]['brush'], delta)
    source_pixels = set()
    for route in routes:
        assert not source_pixels & route['pixels'], "source routes may not intersect"
        source_pixels |= route['pixels']
    for route in routes:
        assert route['marker'] not in source_pixels
        assert route['marker'] in route['shadow']
        assert all(r < a < e-1 and c < b < d-1 for a, b in route['pixels'] | route['shadow']), "no clipping"
    return dict(grid=grid, frame=frame, frame_color=frame_color, background=background,
                routes=routes, brushes=brushes, brush_boxes=brush_boxes, source_pixels=source_pixels)


def _paint(analysis, *, center_only=False, zero_offset=False, all_alignments=False,
           source_paint=False, rotate_brush=False, shadow_front=False,
           vertices_only=False, xor_stamps=False):
    grid = analysis['grid']
    bg = analysis['background']
    r, e, c, d = analysis['frame']
    result = np.full((e-r, d-c), bg, dtype=np.int8)
    result[0, :] = result[-1, :] = analysis['frame_color']
    result[:, 0] = result[:, -1] = analysis['frame_color']
    claimed = {}
    centerlines = []
    counterfactual = any((center_only, zero_offset, all_alignments, source_paint,
                          rotate_brush, shadow_front, vertices_only, xor_stamps))
    for route in analysis['routes']:
        pixels = route['pixels'] | route['rejected'] if all_alignments else route['pixels']
        centerlines.append((pixels, route['source']))
        mask = np.ones((1, 1), bool) if center_only else route['brush']
        if rotate_brush:
            mask = np.rot90(mask)
        centers = set(route['points']) if vertices_only else pixels
        offset = (0, 0) if zero_offset else route['offset']
        shadow = _sweep(centers, mask, offset)
        if xor_stamps:
            parity = set()
            radius = len(mask)//2
            for a, b in centers:
                for u, v in np.argwhere(mask):
                    point = a+offset[0]+int(u)-radius, b+offset[1]+int(v)-radius
                    if point in parity:
                        parity.remove(point)
                    else:
                        parity.add(point)
            shadow = parity
        paint = route['source'] if source_paint else route['target']
        for a, b in shadow:
            if not (r < a < e-1 and c < b < d-1):
                assert counterfactual, 'valid brush leaves the interior'
                continue
            if (a, b) not in analysis['source_pixels'] and grid[a, b] == bg:
                assert counterfactual or (a, b) not in claimed or claimed[a, b] == paint, "different shadows collide"
                claimed[a, b] = paint
            result[a-r, b-c] = paint
    if not shadow_front:
        for pixels, source in centerlines:
            for a, b in pixels:
                result[a-r, b-c] = source
    # The supplied dot is an existing sample of the shadow paint.
    for route in analysis['routes']:
        a, b = route['marker']
        result[a-r, b-c] = route['target']
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct the answer from only the visible input and fixed conventions."""
    return _paint(_parse(input_grid))


def _transformed(points, code):
    result = []
    for r, c in points:
        if code >= 4:
            c = -c
        for _ in range(code % 4):
            r, c = c, -r
        result.append((r, c))
    a, _, x, _ = _box(result)
    return [(r-a, c-x) for r, c in result]


def _sample_route(rng, family, scale):
    for _ in range(220):
        h, w = int(rng.integers(3, scale+1)), int(rng.integers(3, scale+1))
        if family == 'segment':
            points, closed = [(0, 0), (0, w)], False
        elif family == 'walk':
            points, closed = [(0, 0)], False
            for _ in range(int(rng.integers(2, 6))):
                dr, dc = OFFSETS[int(rng.integers(8))]
                length = int(rng.integers(1, max(3, scale//2)+1))
                points.append((points[-1][0]+dr*length, points[-1][1]+dc*length))
        elif family == 'diamond':
            points, closed = [(0, w), (w, 2*w), (2*w, w), (w, 0)], True
        elif family == 'notch':
            h, w = max(h, 5), max(w, 5)
            left = int(rng.integers(1, w-2))
            right = int(rng.integers(left+1, w))
            depth = int(rng.integers(1, h-1))
            points = [(0, 0), (0, left), (depth, left), (depth, right),
                      (0, right), (0, w), (h, w), (h, 0)]
            closed = True
        elif family == 'clipped':
            cut = int(rng.integers(1, min(h, w)))
            points, closed = [(0, 0), (0, w-cut), (cut, w), (h, w), (h, cut), (h-cut, 0)], True
        else:
            if rng.random() < .4:
                w = h
            points, closed = [(0, 0), (0, w), (h, w), (h, 0)], True
        if len(set(points)) != len(points):
            continue
        points = _transformed(points, int(rng.integers(8)))
        a, b, x, y = _box(points)
        if b-a > 18 or y-x > 18:
            continue
        try:
            info = _route(points)
        except AssertionError:
            continue
        expected = set()
        for p, q in zip(points, points[1:] + ([points[0]] if closed else [])):
            expected.update(_line(p, q))
        if info['pixels'] != expected or info['kind'] != ('cycle' if closed else 'path'):
            continue
        if family == 'notch' and not info['concave']:
            continue
        if not info['pixels']-set(points):
            continue
        return points, expected
    # Finite fallback remains in the requested open/closed class.
    points = [(0, 0), (0, 5), (3, 5)] if family in ('walk', 'segment') else [(0, 0), (0, 5), (5, 5), (5, 0)]
    info = _route(points)
    return points, info['pixels']


def _sample_brush(rng, n, asymmetric=False):
    for _ in range(200):
        y, x = np.indices((n, n))
        kind = int(rng.integers(7))
        if kind == 0:
            mask = rng.random((n, n)) < rng.uniform(.2, .8)
        elif kind == 1:
            mask = (x == n//2) | (y == n//2)
        elif kind == 2:
            mask = (x == y) | (x+y == n-1)
        elif kind == 3:
            mask = (x == 0) | (y == 0) | (x == n-1) | (y == n-1)
        elif kind == 4:
            mask = (x+y) % 2 == 0
        elif kind == 5:
            mask = y % 2 == 0
        else:
            mask = np.ones((n, n), bool)
        mask[n//2, n//2] = True
        for edge in (mask[0], mask[-1], mask[:, 0], mask[:, -1]):
            if not edge.any():
                edge[int(rng.integers(n))] = True
        mask = np.rot90(mask, int(rng.integers(4))).copy()
        if not asymmetric or all(not np.array_equal(mask, z) for z in
                                 [np.rot90(mask, k) for k in (1, 2, 3)] +
                                 [np.rot90(np.fliplr(mask), k) for k in range(4)]):
            return mask
    mask = np.zeros((n, n), bool)
    mask[0, :] = mask[:, 0] = True
    mask[n//2, n//2] = mask[-1, 1] = True
    return mask


def _object(rng, family, scale, n, offsets, informative=False):
    for _ in range(80):
        points, pixels = _sample_route(rng, family, scale)
        brush = _sample_brush(rng, n, asymmetric=informative)
        choices = []
        for dr, dc in offsets:
            for vertex in points:
                marker = vertex[0]+dr, vertex[1]+dc
                if marker in pixels:
                    continue
                if sum(max(abs(marker[0]-p[0]), abs(marker[1]-p[1])) == 1 for p in points) != 1:
                    continue
                shadow = _sweep(pixels, brush, (dr, dc))
                if informative:
                    if not (shadow & (pixels-set(points))) or len(shadow-pixels-{marker}) < 3:
                        continue
                    if (shadow-pixels) == (_sweep(pixels, brush, (0, 0))-pixels):
                        continue
                choices.append((marker, (dr, dc), shadow))
        if choices:
            marker, offset, shadow = choices[int(rng.integers(len(choices)))]
            footprint = pixels | shadow
            a, b, x, y = _box(footprint)
            shift = lambda values: {(r-a, c-x) for r, c in values}
            return dict(points=[(r-a, c-x) for r, c in points], pixels=shift(pixels),
                        shadow=shift(shadow), marker=(marker[0]-a, marker[1]-x),
                        offset=offset, brush=brush, h=b-a, w=y-x)
    raise AssertionError("could not make an observable brush companion")


def _place_objects(rng, objects):
    # Footprints, rather than bounding rectangles, permit genuine nesting.
    placed, blocked = [], set()
    for obj in sorted(objects, key=lambda o: -o['h']*o['w']):
        options = [(r, c) for r in range(22-obj['h']) for c in range(22-obj['w'])]
        rng.shuffle(options)
        for dr, dc in options:
            footprint = {(r+dr, c+dc) for r, c in obj['pixels'] | obj['shadow']}
            if not footprint & blocked:
                item = dict(obj)
                for key in ('points', 'pixels', 'shadow'):
                    shifted = [(r+dr, c+dc) for r, c in obj[key]]
                    item[key] = shifted if key == 'points' else set(shifted)
                item['marker'] = obj['marker'][0]+dr, obj['marker'][1]+dc
                placed.append(item)
                blocked |= {(r+a, c+b) for r, c in footprint for a in (-1, 0, 1) for b in (-1, 0, 1)}
                break
        else:
            return None
    footprint = set().union(*(o['pixels'] | o['shadow'] for o in placed))
    a, b, x, y = _box(footprint)
    # A one-cell frame surrounds the footprint. Empty slack can be uneven,
    # and the footprint itself may touch the first interior row or column.
    extra_r = int(rng.integers(0, min(3, 22-(b-a))+1))
    extra_c = int(rng.integers(0, min(3, 22-(y-x))+1))
    before_r, before_c = int(rng.integers(extra_r+1)), int(rng.integers(extra_c+1))
    for obj in placed:
        dr, dc = 1+before_r-a, 1+before_c-x
        for key in ('points', 'pixels', 'shadow'):
            shifted = [(r+dr, c+dc) for r, c in obj[key]]
            obj[key] = shifted if key == 'points' else set(shifted)
        obj['marker'] = obj['marker'][0]+dr, obj['marker'][1]+dc
    return placed, b-a+extra_r+2, y-x+extra_c+2


def _draw_frame(grid, r, c, h, w, color):
    grid[r, c:c+w] = grid[r+h-1, c:c+w] = color
    grid[r:r+h, c] = grid[r:r+h, c+w-1] = color


def _layout(rng, objects, h, w, background, frame_color, sources, targets):
    for attempt in range(35):
        if attempt == 34:
            fr, fc = 0, 0  # Guaranteed bottom shelf: h,w <= 24, max_n <= 5.
        else:
            fr, fc = int(rng.integers(31-h)), int(rng.integers(31-w))
        occupied = np.zeros((30, 30), bool)
        occupied[max(0, fr-1):min(30, fr+h+1), max(0, fc-1):min(30, fc+w+1)] = True
        positions = []
        for obj in objects:
            n = len(obj['brush'])
            positions.append(None)
            if attempt == 34:
                rr, cc = h+1, sum(len(o['brush'])+1 for o in objects[:len(positions)-1])
                options = [(rr, cc)]
            else:
                options = [(r, c) for r in range(31-n) for c in range(31-n)
                           if not occupied[r:r+n, c:c+n].any()]
                rng.shuffle(options)
            if not options:
                break
            rr, cc = options[0]
            if rr+n > 30 or cc+n > 30:
                break
            positions[-1] = (rr, cc)
            occupied[max(0, rr-1):min(30, rr+n+1), max(0, cc-1):min(30, cc+n+1)] = True
        if len(positions) != len(objects) or any(p is None for p in positions):
            continue
        grid = np.full((30, 30), background, dtype=np.int8)
        _draw_frame(grid, fr, fc, h, w, frame_color)
        output = grid[fr:fr+h, fc:fc+w].copy()
        for obj, source, target, (rr, cc) in zip(objects, sources, targets, positions):
            n = len(obj['brush'])
            grid[rr:rr+n, cc:cc+n][obj['brush']] = source
            for r, c in obj['points']:
                grid[fr+r, fc+c] = source
            mr, mc = obj['marker']
            grid[fr+mr, fc+mc] = target
            for r, c in obj['shadow']:
                output[r, c] = target
        for obj, source in zip(objects, sources):
            for r, c in obj['pixels']:
                output[r, c] = source
        a, b, x, y = _box(np.argwhere(grid != background))
        a, x = max(0, a-int(rng.integers(3))), max(0, x-int(rng.integers(3)))
        b, y = min(30, b+int(rng.integers(3))), min(30, y+int(rng.integers(3)))
        grid = grid[a:b, x:y].copy()
        if np.count_nonzero(grid == background) <= grid.size//2:
            continue
        return dict(input=grid, output=output)
    return None


def _make_pair(rng, count, offsets, sizes, demo=None, required_offset=None):
    for attempt in range(120):
        objects = []
        for i in range(count):
            family = ('rectangle' if demo == 'contour' else 'walk') if i == 0 and demo else str(
                rng.choice(['segment', 'walk', 'rectangle', 'diamond', 'clipped', 'notch'],
                           p=[.16, .24, .17, .12, .15, .16]))
            n = int(rng.choice(sizes))
            if i == 0 and (demo or max(sizes) == 5):
                n = max(sizes)
            permitted = (required_offset,) if i == 0 and required_offset is not None else offsets
            objects.append(_object(rng, family, int(rng.integers(5, 10)), n, permitted,
                                   informative=i == 0 and demo is not None))
        packed = _place_objects(rng, objects)
        if packed is None:
            continue
        objects, h, w = packed
        if max(h, w) > 24:
            continue
        colors = list(map(int, rng.permutation(10)))
        bg, frame_color = colors[:2]
        sources = colors[2:2+count]
        palette = colors[2+count:]
        targets = list(map(int, rng.choice(palette, count, replace=True)))
        pair = _layout(rng, objects, h, w, bg, frame_color, sources, targets)
        if pair is None:
            continue
        try:
            analysis = _parse(pair['input'])
            assert np.array_equal(_paint(analysis), pair['output'])
            if demo:
                f = _features(analysis)
                assert f['spread_difference'] > 0 and f['offset_difference'] > 0 and f['priority_difference'] > 0
                if demo == 'contour':
                    assert f['chord_difference']
                else:
                    assert any(o['kind'] == 'path' and len(o['points']) >= 3 for o in analysis['routes'])
                    assert f['orientation_difference']
                    assert f['vertex_difference'] > 0 and f['union_difference'] > 0
            return pair
        except AssertionError:
            continue
    raise RuntimeError("bounded pair generation exhausted")


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    nt, nq = _count('num_train', num_train, 2, 10), _count('num_test', num_test, 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    mode = int(rng.integers(3))  # direction, brush extent, simultaneous sources
    held = OFFSETS[int(rng.integers(8))]
    train_offsets = tuple(d for d in OFFSETS if mode != 0 or d != held)
    train_sizes = (1, 3) if mode == 1 else (1, 3, 5)
    demos = list(map(int, rng.choice(nt, 2, replace=False)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            training = split == 'train'
            demo = ('contour' if i == demos[0] else 'open') if training and i in demos else None
            k = int(rng.integers(1, (2 if mode == 2 and training else 3)+1))
            if mode == 2 and not training:
                k = 3
            offsets = train_offsets if training else OFFSETS
            sizes = train_sizes if training else ((5,) if mode == 1 else (1, 3, 5))
            for _ in range(12):
                pair = _make_pair(rng, k, offsets, sizes, demo,
                                  required_offset=held if mode == 0 and not training else None)
                fingerprint = _key(pair['input'])
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('could not generate distinct inputs')
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(analysis):
    answer = _paint(analysis)
    def different(**kwargs):
        return int(np.count_nonzero(_paint(analysis, **kwargs) != answer))
    return dict(spread_difference=different(center_only=True),
                offset_difference=different(zero_offset=True),
                chord_difference=different(all_alignments=True),
                paint_difference=different(source_paint=True),
                orientation_difference=different(rotate_brush=True),
                priority_difference=different(shadow_front=True),
                vertex_difference=different(vertices_only=True),
                union_difference=different(xor_stamps=True))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            fingerprint = _key(inp)
            assert fingerprint not in seen
            seen.add(fingerprint)
            a = _parse(inp)
            assert np.array_equal(_paint(a), out), 'incorrect brush sweep or spine priority'
            r, e, c, d = a['frame']
            original = inp[r:e, c:d]
            assert out.shape == original.shape
            occupied = original != a['background']
            assert np.array_equal(out[occupied], original[occupied]), 'existing board cells changed'
            assert np.any(out != original)
            assert set(map(int, np.unique(out))) <= set(map(int, np.unique(inp)))
            infos[split].append(a)
    train = infos['train']
    assert any(o['kind'] == 'cycle' and o['rejected'] for a in train for o in a['routes'])
    assert any(o['kind'] == 'path' and len(o['points']) >= 3 for a in train for o in a['routes'])
    evidence = [_features(a) for a in train]
    for name in ('spread_difference', 'offset_difference', 'chord_difference',
                 'paint_difference', 'priority_difference', 'orientation_difference',
                 'vertex_difference', 'union_difference'):
        assert any(f[name] > 0 for f in evidence), f'training does not distinguish {name}'
    # The distribution shift is visible in inputs, independent of labels.
    tr_offsets = {o['offset'] for a in train for o in a['routes']}
    te_offsets = {o['offset'] for a in infos['test'] for o in a['routes']}
    tr_sizes = {len(o['brush']) for a in train for o in a['routes']}
    te_sizes = {len(o['brush']) for a in infos['test'] for o in a['routes']}
    assert (te_offsets-tr_offsets or max(te_sizes) > max(tr_sizes)
            or max(len(a['routes']) for a in infos['test']) > max(len(a['routes']) for a in train))


def measure_puzzle(puzzle):
    rows = []
    for split in ('train', 'test'):
        for index, pair in enumerate(puzzle[split]):
            a = _parse(pair['input'])
            r, e, c, d = a['frame']
            sides = set()
            for b, end, x, right in a['brush_boxes'].values():
                sides.add('above' if end < r else 'below' if b > e else 'left' if right < c else 'right')
            rows.append(dict(split=split, index=index, input_shape=list(np.asarray(pair['input']).shape),
                             output_shape=list(np.asarray(pair['output']).shape), background=a['background'],
                             frame_color=a['frame_color'], sources=len(a['routes']),
                             brush_sizes=[len(o['brush']) for o in a['routes']],
                             vertices=[len(o['points']) for o in a['routes']],
                             kinds=[o['kind'] for o in a['routes']],
                             concave=sum(o['concave'] for o in a['routes']),
                             diagonal=sum(o['diagonal'] for o in a['routes']),
                             unit_links=sum(o['lengths'].count(1) for o in a['routes']),
                             offsets=[list(o['offset']) for o in a['routes']],
                             same_paint=len({o['target'] for o in a['routes']}) < len(a['routes']),
                             sides=sorted(sides), **_features(a)))
    return rows


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records, structures, shapes, brush_shapes, palettes = [], set(), set(), set(), set()
    signatures = set()
    for seed, puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        records.extend(measure_puzzle(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = _parse(pair['input'])
                shapes.add(_key(pair['input'])[0])
                palettes.add(a['background'])
                roles = {}
                normalized = tuple(roles.setdefault(int(v), len(roles)) for v in np.asarray(pair['input']).flat)
                structures.add((np.asarray(pair['input']).shape, normalized))
                for o in a['routes']:
                    brush_shapes.add(_key(o['brush'].astype(np.int8)))
                assert np.array_equal(solve(pair['input']), pair['output'])
                # Arbitrary recoloring and all dihedral transforms preserve the
                # visible rule, including the brush orientation and marker.
                if split == 'test' and seed < 8:
                    permutation = np.random.default_rng(1000+seed).permutation(10)
                    assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
                    for k in range(4):
                        for reflect in (False, True):
                            inp, out = np.rot90(pair['input'], k), np.rot90(pair['output'], k)
                            if reflect:
                                inp, out = np.fliplr(inp), np.fliplr(out)
                            assert np.array_equal(solve(inp), out)
    assert len(signatures) == 42 and len(structures) >= 200 and len(shapes) >= 40
    assert palettes == set(range(10)) and len(brush_shapes) >= 40
    assert {r['sources'] for r in records} == {1, 2, 3}
    assert {n for r in records for n in r['brush_sizes']} == {1, 3, 5}
    assert {tuple(d) for r in records for d in r['offsets']} == set(OFFSETS)
    assert sum(r['concave'] for r in records) and sum(r['unit_links'] for r in records)
    assert sum(r['same_paint'] for r in records) and sum(r['diagonal'] for r in records)
    assert set.union(*(set(r['sides']) for r in records)) == {'above', 'below', 'left', 'right'}
    repeat = generate_puzzle(17)
    assert all(_key(x[k]) == _key(y[k]) for s in ('train', 'test')
               for x, y in zip(puzzles[17][s], repeat[s]) for k in ('input', 'output'))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(7000+nt*10+nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Reject a removed shadow pixel: validation is not merely shape checking.
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ('train', 'test')}
    pair = bad['test'][0]
    a = _parse(pair['input'])
    r, e, c, d = a['frame']
    location = tuple(np.argwhere((pair['output'] != pair['input'][r:e, c:d]) &
                                (pair['output'] != a['background']))[0])
    pair['output'][location] = a['background']
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted sweep accepted')
    return dict(seeds=42, pairs=len(records), input_shapes=len(shapes),
                color_normalized_structures=len(structures), brush_patterns=len(brush_shapes))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
