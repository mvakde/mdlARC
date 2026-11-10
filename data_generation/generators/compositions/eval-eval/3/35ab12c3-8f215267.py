"""Census-displaced contours: input-only geometry, local counts and vector motion.

Only NumPy and the standard library are used. No evidence files are loaded.
Self-tests stream bounded puzzles and do not save payloads or execution traces.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


DIRECTIONS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
CARDINAL = tuple(d for d in DIRECTIONS if not d[0] * d[1])
DIAGONAL = tuple(d for d in DIRECTIONS if d[0] * d[1])


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name):
    _require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)),
             name + ' must be an integer')
    return int(value)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'grid dimensions')
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)), 'grid colors')
    return a.astype(np.int8, copy=False)


def _line(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    steps = max(abs(dr), abs(dc))
    _require(steps > 0 and (dr == 0 or dc == 0 or abs(dr) == abs(dc)), 'unaligned edge')
    sr, sc = int(np.sign(dr)), int(np.sign(dc))
    return {(a[0] + k * sr, a[1] + k * sc) for k in range(steps + 1)}


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _intersects(a, b, c, d):
    if (max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0])
            or max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1])):
        return False
    return _cross(a, b, c) * _cross(a, b, d) <= 0 and _cross(c, d, a) * _cross(c, d, b) <= 0


def _boundary(polygon):
    return set().union(*(_line(a, b) for a, b in zip(polygon, polygon[1:] + polygon[:1])))


def _inside(point, polygon):
    """Strict containment of a cell center, including exact diagonal boundaries."""
    r, c = point
    inside = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if (_cross(a, b, point) == 0 and min(a[0], b[0]) <= r <= max(a[0], b[0])
                and min(a[1], b[1]) <= c <= max(a[1], b[1])):
            return False
        if (a[0] > r) != (b[0] > r):
            crossing = a[1] + (r - a[0]) * (b[1] - a[1]) / (b[0] - a[0])
            if c < crossing:
                inside = not inside
    return inside


def _cycle(points):
    """Unique noncrossing Hamiltonian cycle on visible aligned landmarks.

    At ten vertices and a fixed start, even a complete graph has fewer than one
    million partial paths. Edge-set signatures remove reversal symmetry.
    """
    points = list(points)
    n = len(points)
    _require(4 <= n <= 10 and len(set(points)) == n, 'landmark count')
    _require(all(max(abs(a[0]-b[0]), abs(a[1]-b[1])) >= 2
                 for i, a in enumerate(points) for b in points[:i]), 'landmarks touch')
    vertices = set(points)
    edges, rasters = [], []
    for i, a in enumerate(points):
        for j in range(i + 1, n):
            b = points[j]
            dr, dc = abs(a[0]-b[0]), abs(a[1]-b[1])
            if dr == 0 or dc == 0 or dr == dc:
                pixels = _line(a, b)
                if not (pixels - {a, b}) & vertices:
                    edges.append((i, j))
                    rasters.append(pixels)
    adjacency = [[] for _ in points]
    for k, (i, j) in enumerate(edges):
        adjacency[i].append((j, k))
        adjacency[j].append((i, k))
    _require(all(len(a) >= 2 for a in adjacency), 'no complete cycle')
    conflicts = [0] * len(edges)
    for k, (a, b) in enumerate(edges):
        for l, (c, d) in enumerate(edges[:k]):
            if not {a, b} & {c, d} and _intersects(points[a], points[b], points[c], points[d]):
                conflicts[k] |= 1 << l
                conflicts[l] |= 1 << k
    solutions, calls = {}, 0

    def visit(order, used, selected, edge_ids):
        nonlocal calls
        calls += 1
        _require(calls <= 1_000_000, 'cycle search budget')
        if len(solutions) > 1:
            return
        if len(order) == n:
            for neighbor, edge in adjacency[order[-1]]:
                if neighbor == 0 and not conflicts[edge] & selected:
                    signature = tuple(sorted(edge_ids + [edge]))
                    solutions[signature] = order
            return
        for neighbor, edge in adjacency[order[-1]]:
            if not used & (1 << neighbor) and not conflicts[edge] & selected:
                visit(order + [neighbor], used | (1 << neighbor), selected | (1 << edge), edge_ids + [edge])

    visit([0], 1, 0, [])
    _require(len(solutions) == 1, 'ambiguous or absent cycle')
    selected, order = next(iter(solutions.items()))
    polygon = [points[i] for i in order]
    return dict(polygon=polygon, pixels=set().union(*(rasters[k] for k in selected)),
                all_pixels=set().union(*rasters), chords=len(edges)-len(selected))


def _components(grid, bg):
    seen, result = set(), []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if grid[r, c] == bg or (r, c) in seen:
                continue
            color = int(grid[r, c])
            cells, stack = set(), [(r, c)]
            seen.add((r, c))
            while stack:
                p = stack.pop()
                cells.add(p)
                for dr, dc in CARDINAL:
                    q = p[0]+dr, p[1]+dc
                    if (0 <= q[0] < h and 0 <= q[1] < w and q not in seen and grid[q] == color):
                        seen.add(q)
                        stack.append(q)
            result.append(dict(color=color, cells=cells))
    return result


def _inspect(input_grid):
    grid = _grid(input_grid)
    values, frequencies = np.unique(grid, return_counts=True)
    bg = int(values[int(np.argmax(frequencies))])
    _require(int(max(frequencies)) > grid.size // 2, 'background must occupy a strict majority')
    parts = _components(grid, bg)
    groups = {int(c): [p for p in parts if p['color'] == c] for c in values if c != bg}
    source_colors = [c for c, ps in groups.items() if 4 <= len(ps) <= 10
                     and all(len(p['cells']) == 1 for p in ps)]
    _require(1 <= len(source_colors) <= 2, 'one or two source colors required')
    sources = []
    for color in source_colors:
        points = sorted(set().union(*(p['cells'] for p in groups[color])))
        info = _cycle(points)
        info.update(color=color, points=points)
        sources.append(info)
    if len(sources) == 2:
        a, b = sources
        _require(not a['pixels'] & b['pixels'], 'source boundaries overlap')
        _require(not any(_inside(p, b['polygon']) for p in a['points'])
                 and not any(_inside(p, a['polygon']) for p in b['points']), 'nested/intersecting sources')
        _require(not any(_intersects(p, q, r, s)
                         for p, q in zip(a['polygon'], a['polygon'][1:]+a['polygon'][:1])
                         for r, s in zip(b['polygon'], b['polygon'][1:]+b['polygon'][:1])), 'sources cross')
    loose = [p for p in parts if p['color'] not in source_colors]
    markers = [p for p in loose if len(p['cells']) == 1]
    objects = [p for p in loose if len(p['cells']) != 1]
    _require(len(markers) == len(sources), 'one marker per source required')
    for marker in markers:
        p = next(iter(marker['cells']))
        anchors = [(i, v) for i, source in enumerate(sources) for v in source['points']
                   if max(abs(p[0]-v[0]), abs(p[1]-v[1])) == 1]
        _require(len(anchors) == 1, 'ambiguous direction marker')
        i, v = anchors[0]
        _require('marker' not in sources[i], 'multiple markers on a source')
        _require(not any(p in s['pixels'] for s in sources), 'marker lies on boundary')
        sources[i].update(marker=p, category=marker['color'], direction=(p[0]-v[0], p[1]-v[1]))
    _require(all('marker' in s for s in sources), 'unmarked source')
    labels = {}
    for i, p in enumerate(loose):
        for cell in p['cells']:
            labels[cell] = i
    for p, i in labels.items():
        for dr, dc in DIRECTIONS:
            q = p[0]+dr, p[1]+dc
            _require(q not in labels or labels[q] == i, 'objects/markers touch')
    source_pixels = set().union(*(s['pixels'] for s in sources))
    for obj in objects:
        _require(2 <= len(obj['cells']) <= 9, 'object area')
        _require(not obj['cells'] & source_pixels, 'object lies on boundary')
        memberships = []
        for source in sources:
            inside = [_inside(p, source['polygon']) for p in obj['cells']]
            _require(all(inside) or not any(inside), 'object straddles a boundary')
            memberships.append(all(inside))
        _require(sum(memberships) <= 1, 'object belongs to multiple windows')
        obj['memberships'] = memberships
    for i, source in enumerate(sources):
        matched = [o for o in objects if o['memberships'][i] and o['color'] == source['category']]
        source['count'] = len(matched)
        source['area'] = sum(len(o['cells']) for o in matched)
        _require(source['count'] <= 4, 'population exceeds four')
    detail = dict(background=bg, sources=sources, objects=objects, markers=markers, shape=grid.shape)
    _render(detail)  # Bounds and overlap validation, with no generation metadata.
    return detail


def _render(detail, alternative=None, strict=True):
    bg, shape = detail['background'], detail['shape']
    output = np.full(shape, bg, dtype=np.int8)
    claimed = {}
    source_pixels = set().union(*(s['pixels'] for s in detail['sources']))
    for i, source in enumerate(detail['sources']):
        n = source['count']
        if alternative == 'unit':
            n = int(n > 0)
        elif alternative == 'area':
            n = source['area']
        elif alternative == 'global':
            n = sum(o['color'] == source['category'] for o in detail['objects'])
        elif alternative == 'colorless':
            n = sum(o['memberships'][i] for o in detail['objects'])
        elif alternative == 'box':
            rs, cs = zip(*source['polygon'])
            n = sum(o['color'] == source['category'] and all(
                min(rs) < r < max(rs) and min(cs) < c < max(cs) for r, c in o['cells'])
                for o in detail['objects'])
        pixels = source['all_pixels'] if alternative == 'chords' else source['pixels']
        dr, dc = source['direction']
        distances = range(1, n+1) if alternative == 'repeated' else (n,)
        for distance in distances:
            for r, c in pixels:
                q = r+distance*dr, c+distance*dc
                in_bounds = 0 <= q[0] < shape[0] and 0 <= q[1] < shape[1]
                if strict:
                    _require(in_bounds, 'companion would be clipped')
                if not in_bounds or q in source_pixels:
                    continue
                if strict:
                    _require(q not in claimed or claimed[q] == source['category'], 'companion colors conflict')
                claimed[q] = source['category']
                output[q] = source['category']
    for source in detail['sources']:
        pixels = source['all_pixels'] if alternative == 'chords' else source['pixels']
        for p in pixels:
            output[p] = source['color']
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every output cell from the given input; no seeds or global state."""
    return _render(_inspect(input_grid))


def _transform(points, code):
    result = []
    for r, c in points:
        if code & 4:
            c = -c
        for _ in range(code % 4):
            r, c = c, -r
        result.append((r, c))
    r0, c0 = min(r for r, c in result), min(c for r, c in result)
    return [(r-r0, c-c0) for r, c in result]


def _polygon(rng, small, family=None):
    """Procedural geometry; sample independently from token shapes and colors."""
    kind = family or str(rng.choice(('rectangle', 'square', 'chamfer', 'diamond', 'notch')))
    for _ in range(70):
        if small:
            h, w = (int(rng.integers(7, 10)) for _ in range(2))
        else:
            h, w = (int(rng.integers(10, 16)) for _ in range(2))
        if kind == 'square':
            w = h
        if kind in ('rectangle', 'square'):
            points = [(0, 0), (0, w), (h, w), (h, 0)]
        elif kind == 'diamond':
            a = int(rng.integers(4, 6)) if small else int(rng.integers(6, 10))
            b = a if rng.random() < .5 else a + int(rng.integers(0, 2))
            # Each edge must remain a 45-degree line: an oblique diamond is a
            # rotated rectangle rather than an unequal-axis rhombus.
            points = [(0, a), (b, a+b), (a+b, b), (a, 0)]
        elif kind == 'notch':
            a, b = 2, h-2
            depth = int(rng.integers(2, min(4, w-4)+1))
            points = [(0, 0), (0, w), (a, w), (a, w-depth),
                      (b, w-depth), (b, w), (h, w), (h, 0)]
        else:
            cuts = [int(rng.choice((0, 2, 3))) for _ in range(4)]
            a, b, c, d = cuts
            points = [(0, a), (0, w-b), (b, w), (h-c, w),
                      (h, w-c), (h, d), (h-d, 0), (a, 0)]
            points = [p for i, p in enumerate(points) if p != points[i-1]]
        points = _transform(points, int(rng.integers(8)))
        try:
            inferred = _cycle(sorted(points))
            if inferred['pixels'] == _boundary(points):
                return points
        except ValueError:
            continue
    raise ValueError('bounded polygon proposals exhausted')


def _token(rng, area=None):
    area = int(rng.choice((2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9))) if area is None else area
    kind = int(rng.integers(4))
    if kind == 0:
        cells = {(0, c) for c in range(area)}
    elif kind == 1:
        width = int(rng.integers(2, min(4, area)+1))
        cells = {(k//width, k % width) for k in range(area)}
    else:
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in CARDINAL} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    return tuple(sorted(_transform(sorted(cells), int(rng.integers(8)))))


def _place_token(rng, grid, blocked, allowed, color, area=None):
    for _ in range(18):
        mask = _token(rng, area)
        h, w = 1+max(r for r, c in mask), 1+max(c for r, c in mask)
        if h > grid.shape[0] or w > grid.shape[1]:
            continue
        valid = np.ones((grid.shape[0]-h+1, grid.shape[1]-w+1), dtype=bool)
        available = allowed & ~blocked
        for dr, dc in mask:
            valid &= available[dr:dr+valid.shape[0], dc:dc+valid.shape[1]]
        rr, cc = np.nonzero(valid)
        if not len(rr):
            continue
        j = int(rng.integers(len(rr)))
        cells = [(int(rr[j])+dr, int(cc[j])+dc) for dr, dc in mask]
        for r, c in cells:
            grid[r, c] = color
            blocked[max(0, r-1):r+2, max(0, c-1):c+2] = True
        return
    raise ValueError('bounded token placement exhausted')


def _scene(rng, polygon, count, direction, bg, source_color, category, noise_color,
           areas=None, notch_witness=False, noisy=True):
    pixels = _boundary(polygon)
    dr, dc = direction
    copies = {(r+count*dr, c+count*dc) for r, c in pixels}
    anchors = []
    for v in polygon:
        m = v[0]+dr, v[1]+dc
        if (m not in pixels and not _inside(m, polygon)
                and sum(max(abs(m[0]-q[0]), abs(m[1]-q[1])) == 1 for q in polygon) == 1):
            anchors.append(m)
    _require(bool(anchors), 'no unambiguous exterior marker')
    marker = anchors[int(rng.integers(len(anchors)))]
    extent = pixels | copies | {marker}
    r0, c0 = min(r for r, c in extent)-2, min(c for r, c in extent)-2
    h, w = max(r for r, c in extent)-r0+3, max(c for r, c in extent)-c0+3
    _require(h <= 30 and w <= 30, 'scene too large')
    poly = [(r-r0, c-c0) for r, c in polygon]
    marker = marker[0]-r0, marker[1]-c0
    source = {(r-r0, c-c0) for r, c in pixels}
    copy = {(r-r0, c-c0) for r, c in copies}
    grid = np.full((h, w), bg, dtype=np.int8)
    blocked = np.zeros((h, w), dtype=bool)
    for p in poly:
        grid[p] = source_color
    for p in source:
        blocked[p] = True
    grid[marker] = category
    blocked[max(0, marker[0]-1):marker[0]+2, max(0, marker[1]-1):marker[1]+2] = True
    inside = np.array([[_inside((r, c), poly) for c in range(w)] for r in range(h)], dtype=bool)
    outside = ~inside
    for p in source:
        outside[p] = False
    if notch_witness:
        rs, cs = zip(*poly)
        notch = outside.copy()
        for r in range(h):
            for c in range(w):
                notch[r, c] &= min(rs) < r < max(rs) and min(cs) < c < max(cs)
        _place_token(rng, grid, blocked, notch, category, 2)
    # Forced witnesses are exceptions only to area sampling, never to the rule.
    for j in range(count):
        _place_token(rng, grid, blocked, inside, category, None if areas is None else areas[j])
    if noisy:
        _place_token(rng, grid, blocked, inside, noise_color, 2 if count >= 3 else None)
    # Every local scene has a category-colored exterior object; other scenes'
    # matching populations also remain exterior to this scene after packing.
    _place_token(rng, grid, blocked, outside, category, 2 if count >= 3 else None)
    if rng.random() < .3:
        _place_token(rng, grid, blocked, outside, noise_color)
    out = np.full_like(grid, bg)
    for p in copy:
        out[p] = category
    for p in source:
        out[p] = source_color
    # Crop around all input and intended output pixels, preserving all copies.
    occupied = (grid != bg) | (out != bg)
    rr, cc = np.nonzero(occupied)
    sl = np.s_[int(rr.min()):int(rr.max())+1, int(cc.min()):int(cc.max())+1]
    return {'input': grid[sl].copy(), 'output': out[sl].copy()}


def _pack_scenes(rng, scenes, bg):
    if len(scenes) == 1:
        a = scenes[0]
        additions = [int(rng.integers(min(4, 31-n))) for n in a['input'].shape]
        starts = [int(rng.integers(n+1)) for n in additions]
        h, w = (n+x for n, x in zip(a['input'].shape, additions))
        pair = {key: np.full((h, w), bg, dtype=np.int8) for key in a}
        for key in pair:
            pair[key][starts[0]:starts[0]+a[key].shape[0], starts[1]:starts[1]+a[key].shape[1]] = a[key]
        return pair
    options = []
    for axis in (0, 1):
        for gap in (1, 2, 3):
            along = sum(s['input'].shape[axis] for s in scenes)+gap
            across = max(s['input'].shape[1-axis] for s in scenes)
            if max(along, across) <= 30:
                options.append((axis, gap, along, across))
    _require(bool(options), 'two scenes do not fit')
    axis, gap, along, across = options[int(rng.integers(len(options)))]
    along += int(rng.integers(min(3, 31-along)))
    across += int(rng.integers(min(4, 31-across)))
    shape = (along, across) if axis == 0 else (across, along)
    pair = {k: np.full(shape, bg, dtype=np.int8) for k in ('input', 'output')}
    cursor = int(rng.integers(along-sum(s['input'].shape[axis] for s in scenes)-gap+1))
    for scene in scenes:
        cross_start = int(rng.integers(across-scene['input'].shape[1-axis]+1))
        r, c = (cursor, cross_start) if axis == 0 else (cross_start, cursor)
        h, w = scene['input'].shape
        for key in pair:
            pair[key][r:r+h, c:c+w] = scene[key]
        cursor += scene['input'].shape[axis]+gap
    return pair


def _pair(rng, counts, directions, bg, category=None, demo=None):
    for _ in range(160):
        palette = [int(x) for x in rng.permutation(10) if x != bg and x != category]
        cats = ([category]*len(counts) if category is not None else
                [palette.pop()]*len(counts) if rng.random() < .6 else [palette.pop() for _ in counts])
        source_colors = [palette.pop() for _ in counts]
        noise = palette.pop()
        scenes = []
        try:
            for i, count in enumerate(counts):
                family = 'square' if demo == 0 and i == 0 else 'notch' if demo == 1 and i == 0 else None
                polygon = _polygon(rng, len(counts) == 2, family)
                areas = [4] if demo == 0 and i == 0 else [2, 2] if demo == 1 and i == 0 else [2] if demo == 1 and i == 1 else None
                scenes.append(_scene(rng, polygon, count, directions[i], bg, source_colors[i], cats[i], noise,
                                     areas=areas, notch_witness=demo == 1 and i == 0))
            pair = _pack_scenes(rng, scenes, bg)
            info = _inspect(pair['input'])
            _require(np.array_equal(pair['output'], _render(info)), 'construction/reconstruction mismatch')
            return pair
        except ValueError:
            continue
    raise RuntimeError('bounded scene proposals exhausted')


def _fingerprint(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed')
    _require(seed >= 0, 'seed must be nonnegative')
    nt, nq = _integer(num_train, 'num_train'), _integer(num_test, 'num_test')
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'illegal train/test counts')
    rng = np.random.default_rng(seed)
    category = int(rng.integers(10))
    held = DIRECTIONS[int(rng.integers(8))]
    allowed = [d for d in DIRECTIONS if d != held]
    teaching_directions = [CARDINAL[int(rng.integers(4))], DIAGONAL[int(rng.integers(4))]]
    for i, domain in enumerate((CARDINAL, DIAGONAL)):
        if teaching_directions[i] == held:
            teaching_directions[i] = [d for d in domain if d != held][int(rng.integers(3))]
    puzzle, seen = {'train': [], 'test': []}, set()
    previous_bg = None
    for index in range(nt+nq):
        train = index < nt
        local = index if train else index-nt
        split = 'train' if train else 'test'
        demo = local if train and local < 2 else None
        bg = int(rng.choice([c for c in range(10) if c != previous_bg and (demo is None or c != category)]))
        if demo == 0:
            counts = [1, 0]
        elif demo == 1:
            counts = [2, 1]
        elif not train and local == 0:
            counts = [int(rng.integers(3, 5))]
        else:
            number = int(rng.integers(1, 3))
            maximum = 2 if train or number == 2 else 4
            counts = [int(rng.integers(maximum+1)) for _ in range(number)]
        domain = allowed if train else DIRECTIONS
        directions = [domain[int(rng.integers(len(domain)))] for _ in counts]
        if demo is not None:
            directions[0] = teaching_directions[demo]
        elif not train and local == 0:
            directions[0] = held
        for _ in range(12):
            pair = _pair(rng, counts, directions, bg, category if demo is not None else None, demo)
            key = _fingerprint(pair['input'])
            if key not in seen:
                seen.add(key)
                break
        else:
            raise RuntimeError('distinct input proposals exhausted')
        puzzle[split].append(pair)
        previous_bg = bg
    validate_generated_puzzle(puzzle)
    return puzzle


def _different_alternative(detail, actual, alternative):
    return not np.array_equal(actual, _render(detail, alternative, strict=False))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split schema')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'pair counts')
    seen, details = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'output dimensions changed')
            key = _fingerprint(inp)
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            detail = _inspect(inp)
            _require(np.array_equal(out, _render(detail)), 'incorrect census displacement, completion or erasure')
            _require(np.any(inp != out), 'unchanged pair')
            details[split].append(detail)
    training = details['train']
    sources = [s for d in training for s in d['sources']]
    _require({s['count'] for s in sources} == {0, 1, 2}, 'training population witnesses')
    _require(any(a['category'] == b['category'] and a['area'] == b['area'] and a['count'] != b['count']
                 for a in sources for b in sources), 'same-category equal-area/different-count witness')
    _require(any(a['category'] == b['category'] and a['count'] == b['count'] and a['area'] != b['area']
                 for a in sources for b in sources), 'same-count/different-area witness')
    _require(len({d['background'] for d in training}) >= 2, 'backgrounds must vary')
    _require(any(len(d['sources']) == 2 and len({s['category'] for s in d['sources']}) == 1
                 and len({s['count'] for s in d['sources']}) > 1 for d in training), 'local shared-category witness')
    informative = [s['direction'] for s in sources if s['count'] > 0]
    _require(any(d in CARDINAL for d in informative) and any(d in DIAGONAL for d in informative),
             'cardinal/diagonal teaching evidence')
    for alternative in ('chords', 'unit', 'area', 'global', 'box', 'colorless', 'repeated'):
        _require(any(_different_alternative(d, np.asarray(p['output']), alternative)
                     for d, p in zip(training, puzzle['train'])), 'training shortcut survives: '+alternative)
    first_query = details['test'][0]['sources']
    _require(any(s['count'] >= 3 for s in first_query), 'query population extrapolation')
    _require(any(s['direction'] not in {t['direction'] for t in sources} for s in first_query),
             'query direction must be held out')


def puzzle_metrics(puzzle):
    """Small reconstructible feature records, not a second copy of any grid."""
    records = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            d = _inspect(pair['input'])
            masks = []
            for obj in d['objects']:
                r0, c0 = min(r for r, c in obj['cells']), min(c for r, c in obj['cells'])
                masks.append(tuple(sorted((r-r0, c-c0) for r, c in obj['cells'])))
            turns = [[int(np.sign(_cross(p[i-1], p[i], p[(i+1) % len(p)]))) for i in range(len(p))]
                     for p in (s['polygon'] for s in d['sources'])]
            records.append(dict(split=split, shape=d['shape'], background=d['background'],
                windows=len(d['sources']), counts=tuple(s['count'] for s in d['sources']),
                directions=tuple(s['direction'] for s in d['sources']),
                vertices=tuple(len(s['points']) for s in d['sources']),
                concave=sum(-1 in t and 1 in t for t in turns),
                chords=sum(s['chords'] for s in d['sources']),
                geometry=tuple(tuple(sorted(_transform(s['points'], 0))) for s in d['sources']),
                areas=tuple(len(o['cells']) for o in d['objects']), masks=tuple(masks),
                shared=len(d['sources']) == 2 and len({s['category'] for s in d['sources']}) == 1,
                overlap=sum(len({(r+s['count']*s['direction'][0], c+s['count']*s['direction'][1])
                                 for r, c in s['pixels']} & s['pixels']) for s in d['sources']),
                diagonal=sum(any(a[0] != b[0] and a[1] != b[1] for a, b in
                                 zip(s['polygon'], s['polygon'][1:]+s['polygon'][:1])) for s in d['sources'])))
    return records


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for key in ('input', 'output'):
                a = np.asarray(pair[key], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 streamed seeds; bounded contract, semantic, diversity and mutation checks."""
    signatures, shapes, geometries, masks, directions, backgrounds, areas = (set() for _ in range(7))
    counters, counts, vertices = Counter(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = _digest(puzzle)
        _require(signature not in signatures, 'cross-seed duplicate')
        signatures.add(signature)
        for record in puzzle_metrics(puzzle):
            shapes.add(record['shape'])
            geometries.add(record['geometry'])
            masks.update(record['masks'])
            directions.update(record['directions'])
            backgrounds.add(record['background'])
            areas.update(record['areas'])
            counts.update(record['counts'])
            vertices.update(record['vertices'])
            for name in ('concave', 'chords', 'shared', 'overlap', 'diagonal'):
                counters[name] += record[name]
            counters['one' if record['windows'] == 1 else 'two'] += 1
        # A second call is transient; at most two small puzzles are held here.
        if seed in (0, 21, 41):
            _require(signature == _digest(generate_puzzle(seed)), 'nondeterministic seed')
    _require(len(signatures) == 42 and len(shapes) >= 35, 'puzzle/canvas variation')
    _require(len(geometries) >= 80 and len(masks) >= 45, 'geometry/token vocabulary')
    _require(directions == set(DIRECTIONS) and backgrounds == set(range(10)), 'direction/background variation')
    _require(counts == set(range(5)) and {4, 8} <= vertices, 'population/vertex variation')
    _require(areas == set(range(2, 10)), 'object areas')
    _require(min(counters.values()) > 0, 'missing structural feature')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000+nt*10+nq, nt, nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, 'explicit counts')
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    good = generate_puzzle(4, 2, 1)
    # Whole-input transformations exercise the solver beyond sampled layouts.
    original = good['test'][0]
    for turns in range(4):
        for reflect in (False, True):
            inp = np.rot90(original['input'], turns)
            out = np.rot90(original['output'], turns)
            if reflect:
                inp, out = np.fliplr(inp), np.fliplr(out)
            _require(np.array_equal(solve(inp), out), 'dihedral covariance')
    palette = np.random.default_rng(9041).permutation(10).astype(np.int8)
    _require(np.array_equal(solve(palette[original['input']]), palette[original['output']]),
             'local palette covariance')
    for corruption in ('identity', 'pixel'):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in good[s]] for s in good}
        pair = bad['test'][0]
        if corruption == 'identity':
            pair['output'] = pair['input'].copy()
        else:
            p = tuple(np.argwhere(pair['input'] != pair['output'])[0])
            pair['output'][p] = (int(pair['output'][p])+1) % 10
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            continue
        raise AssertionError('semantic corruption accepted')
