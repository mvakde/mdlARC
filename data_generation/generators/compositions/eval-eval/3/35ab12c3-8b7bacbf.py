"""Registered paint apertures: a routed paint supplies intersecting stencils.

Only stdlib and NumPy are used. The solver reconstructs geometry and sources
from the input. Generation constructs labels from its intended polygons, then
checks them against that reconstruction. No parent generator is imported.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
import numpy as np

D4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))
DIAGONALS = D8[4:]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "bad grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must have integer cells")
    _require(bool(np.all((a >= 0) & (a <= 9))), "bad color")
    return a.astype(np.int8, copy=True)


def _move(cells, d):
    return {(r + d[0], c + d[1]) for r, c in cells}


def _halo(cells, steps=D8):
    return {(r + dr, c + dc) for r, c in cells for dr, dc in steps}


def _components(cells, steps=D8):
    left = set(cells)
    result = []
    while left:
        start = min(left)
        left.remove(start)
        component, queue = {start}, [start]
        for r, c in queue:
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in left:
                    left.remove(q)
                    component.add(q)
                    queue.append(q)
        result.append(component)
    return result


def _aligned(a, b):
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dr, dc) > 0 and (not dr or not dc or dr == dc)


def _segment(a, b):
    _require(_aligned(a, b), "unaligned polygon edge")
    dr, dc = b[0] - a[0], b[1] - a[1]
    sr, sc = int(np.sign(dr)), int(np.sign(dc))
    return {(a[0] + k * sr, a[1] + k * sc) for k in range(max(abs(dr), abs(dc)) + 1)}


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _intersects(a, b, c, d):
    if any(max(a[k], b[k]) < min(c[k], d[k]) or max(c[k], d[k]) < min(a[k], b[k]) for k in (0, 1)):
        return False
    return _cross(a, b, c) * _cross(a, b, d) <= 0 and _cross(c, d, a) * _cross(c, d, b) <= 0


def _cycle(points):
    """Unique simple spanning cycle, considering and rejecting aligned chords."""
    points = tuple(sorted(points))
    n = len(points)
    _require(4 <= n <= 8, "a stencil needs 4..8 vertices")
    vertices = set(points)
    _require(not (_halo(vertices) & vertices), "vertices must not touch")
    edges = []
    adjacent = [[] for _ in points]
    for i, a in enumerate(points):
        for j in range(i + 1, n):
            b = points[j]
            if _aligned(a, b) and not ((_segment(a, b) - {a, b}) & vertices):
                edges.append((i, j))
                adjacent[i].append(j)
                adjacent[j].append(i)
    _require(all(len(v) >= 2 for v in adjacent), "not a closed contour")
    answers = {}
    visits = 0

    def search(order, selected):
        nonlocal visits
        visits += 1
        _require(visits <= 12000, "cycle search bound")
        if len(answers) > 1:
            return
        i = order[-1]
        choices = [0] if len(order) == n else adjacent[i]
        for j in choices:
            if j not in adjacent[i] or (j in order and not (j == 0 and len(order) == n)):
                continue
            if any(not ({i, j} & {u, v}) and _intersects(points[i], points[j], points[u], points[v])
                   for u, v in selected):
                continue
            chosen = selected + [(i, j)]
            if len(order) == n:
                key = tuple(sorted(tuple(sorted(e)) for e in chosen))
                answers[key] = tuple(points[k] for k in order)
            else:
                search(order + [j], chosen)

    search([0], [])
    _require(len(answers) == 1, "ambiguous or absent simple contour")
    order = next(iter(answers.values()))
    boundary = set().union(*(_segment(a, b) for a, b in zip(order, order[1:] + order[:1])))
    return order, boundary, len(edges) - n


def _interior(boundary):
    """Four-connected exterior flood on a padded local canvas, even at an edge."""
    r0, r1 = min(r for r, c in boundary) - 1, max(r for r, c in boundary) + 1
    c0, c1 = min(c for r, c in boundary) - 1, max(c for r, c in boundary) + 1
    exterior, queue = {(r0, c0)}, [(r0, c0)]
    for r, c in queue:
        for dr, dc in D4:
            q = (r + dr, c + dc)
            if r0 <= q[0] <= r1 and c0 <= q[1] <= c1 and q not in boundary and q not in exterior:
                exterior.add(q)
                queue.append(q)
    return {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)} - boundary - exterior


def _polygon_interior(vertices, boundary):
    """Generation's continuous point-in-polygon construction of integer cells."""
    cells = set()
    for r in range(min(p[0] for p in vertices), max(p[0] for p in vertices) + 1):
        for c in range(min(p[1] for p in vertices), max(p[1] for p in vertices) + 1):
            if (r, c) in boundary:
                continue
            odd = False
            for a, b in zip(vertices, vertices[1:] + vertices[:1]):
                if (a[0] > r) != (b[0] > r):
                    x = a[1] + (r - a[0]) * (b[1] - a[1]) / (b[0] - a[0])
                    if c < x:
                        odd = not odd
            if odd:
                cells.add((r, c))
    return cells


def _analyze(input_grid):
    a = _grid(input_grid)
    values, counts = np.unique(a, return_counts=True)
    bg = int(values[np.argmax(counts)])
    _require(int(max(counts)) > a.size // 2, "background must occupy a strict majority")
    groups = {int(v): {tuple(map(int, p)) for p in np.argwhere(a == v)} for v in values if v != bg}
    wire_colors = [v for v, cells in groups.items() if _halo(cells) & cells]
    _require(len(wire_colors) == 1, "one scene wire color must be recognizable by contiguous runs")
    wire_color = wire_colors[0]
    wires = _components(groups[wire_color])
    targets = []
    for color, points in groups.items():
        if color == wire_color or len(points) < 4:
            continue
        order, boundary, chords = _cycle(points)
        inside = _interior(boundary)
        _require(bool(inside), "empty stencil")
        targets.append(dict(color=color, vertices=points, order=order, boundary=boundary,
                            inside=inside, chords=chords))
    _require(1 <= len(targets) <= 3, "need 1..3 stencils")
    source_colors = {t['color'] for t in targets}
    used_markers = set()
    for i, t in enumerate(targets):
        candidates = []
        for color, points in groups.items():
            if color in source_colors or color == wire_color or len(points) != 1:
                continue
            p = next(iter(points))
            associations = [(j, v) for j, other in enumerate(targets) for v in other['vertices'] if p in _halo({v})]
            if len(associations) == 1 and associations[0][0] == i:
                vertex = associations[0][1]
                candidates.append((color, p, (p[0] - vertex[0], p[1] - vertex[1])))
        _require(len(candidates) == 1, "each contour needs one unambiguous companion marker")
        color, marker, offset = candidates[0]
        _require(color not in used_markers and marker not in t['boundary'], "bad companion marker")
        used_markers.add(color)
        copy_boundary, copy_inside = _move(t['boundary'], offset), _move(t['inside'], offset)
        _require(all(0 <= r < a.shape[0] and 0 <= c < a.shape[1] for r, c in t['boundary'] | copy_boundary),
                 "clipped stencils are outside the family")
        aperture = t['inside'] & copy_inside
        _require(bool(aperture), "empty registered aperture")
        t.update(marker_color=color, marker=marker, offset=offset,
                 copy_boundary=copy_boundary, copy_inside=copy_inside, aperture=aperture)
    paint_by_wire = {}
    sources = []
    for color, cells in groups.items():
        if color in source_colors | used_markers | {wire_color}:
            continue
        for p in sorted(cells):
            neighbors = _halo({p})
            incident = []
            for i, wire in enumerate(wires):
                cup = (wire & neighbors) | {p}
                if (len(cup) == 4 and max(r for r, c in cup)-min(r for r, c in cup) == 1
                        and max(c for r, c in cup)-min(c for r, c in cup) == 1):
                    incident.append(i)
            _require(len(incident) == 1, "paint must sit in a three-neighbor wire cup")
            i = incident[0]
            _require(i not in paint_by_wire, "one source per wire component")
            _require(all(a[q] in (bg, wire_color) for q in neighbors if 0 <= q[0] < a.shape[0] and 0 <= q[1] < a.shape[1]),
                     "paint cup touches unrelated structure")
            paint_by_wire[i] = color
            sources.append((p, color, i))
    _require(1 <= len(sources) <= 2 and len(wires) == len(sources), "one or two supplied wire components")
    all_bodies = set()
    for t in targets:
        body = t['boundary'] | t['inside'] | t['copy_boundary'] | t['copy_inside']
        _require(not body & (all_bodies | _halo(all_bodies)), "different stencil pairs must be separated")
        all_bodies |= body
        _require(not groups[wire_color] & body, "wire enters a stencil body")
        incident = [i for i, w in enumerate(wires) if w & _halo(t['boundary'])]
        _require(len(incident) <= 1, "conflicting contour supply")
        t['owner'] = incident[0] if incident else None
        t['paint'] = paint_by_wire[incident[0]] if incident else None
    _require(any(t['paint'] is not None for t in targets), "no productive supply")
    # All original strokes precede all companion strokes. Existing cells survive.
    originals = set().union(*(t['boundary'] for t in targets))
    out = a.copy()
    for t in targets:
        for p in t['copy_boundary'] - originals:
            if a[p] == bg:
                out[p] = t['marker_color']
    for t in targets:
        for p in t['boundary']:
            _require(a[p] in (bg, t['color']), "original stroke overlaps an input marker")
            out[p] = t['color']
        if t['paint'] is not None:
            for p in t['aperture']:
                _require(a[p] == bg, "occluded aperture")
                out[p] = t['paint']
    return out, dict(background=bg, wire_color=wire_color, wires=wires, sources=sources, targets=targets)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only inference; does not read files or generation state."""
    return _analyze(input_grid)[0]


def _ri(rng, lo, hi):
    return int(rng.integers(lo, hi + 1))


def _pick(rng, values):
    return values[_ri(rng, 0, len(values) - 1)]


def _shape(rng, force=None):
    for _ in range(48):
        kind = force or _pick(rng, ('box', 'diamond', 'bevel', 'slant', 'elbow', 'notch'))
        h, w = _ri(rng, 5, 9), _ri(rng, 5, 9)
        if kind == 'box':
            points = [(0, 0), (0, w), (h, w), (h, 0)]
        elif kind == 'diamond':
            x, y = _ri(rng, 3, 5), _ri(rng, 3, 5)
            if force == 'diamond':
                y = x  # Guaranteed nonconsecutive aligned chords in the lesson.
            points = [(0, y), (x, x + y), (x + y, x), (y, 0)]
        elif kind == 'bevel':
            h, w = _ri(rng, 7, 10), _ri(rng, 7, 10)
            cuts = [_pick(rng, (0, 2)) for _ in range(4)]
            x, y, z, q = cuts
            points = [(0, x), (0, w-y), (y, w), (h-z, w), (h, w-z), (h, q), (h-q, 0), (x, 0)]
            points = [p for i, p in enumerate(points) if p != points[i-1]]
        elif kind == 'slant':
            h, w = _ri(rng, 3, 5), _ri(rng, 4, 7)
            points = [(0, h), (0, h+w), (h, w), (h, 0)]
        elif kind == 'elbow':
            h, w = _ri(rng, 7, 10), _ri(rng, 7, 10)
            x, y = _ri(rng, 3, h-3), _ri(rng, 3, w-3)
            points = [(0, 0), (0, w), (x, w), (x, y), (h, y), (h, 0)]
        else:
            h, w = _ri(rng, 7, 9), _ri(rng, 8, 10)
            x, y, depth = 3, w-3, _ri(rng, 2, h-3)
            points = [(0, 0), (0, x), (depth, x), (depth, y), (0, y), (0, w), (h, w), (h, 0)]
        code = _ri(rng, 0, 7)
        transformed = []
        for r, c in points:
            if code & 4:
                c = -c
            for _ in range(code % 4):
                r, c = c, -r
            transformed.append((r, c))
        mr, mc = min(r for r, c in transformed), min(c for r, c in transformed)
        transformed = [(r-mr, c-mc) for r, c in transformed]
        try:
            _, boundary, _ = _cycle(transformed)
        except ValueError:
            continue
        inside = _polygon_interior(transformed, boundary)
        if inside and inside == _interior(boundary):
            return transformed, boundary, inside, kind
    raise ValueError('bounded shape sampling exhausted')


def _path(rng, starts, goals, allowed):
    starts = sorted(starts)
    rng.shuffle(starts)
    queue = deque(starts)
    previous = {p: None for p in starts}
    directions = [D8[int(i)] for i in rng.permutation(8)]
    end = None
    while queue:
        p = queue.popleft()
        if p in goals:
            end = p
            break
        for dr, dc in directions:
            q = (p[0]+dr, p[1]+dc)
            if q in allowed and q not in previous:
                previous[q] = p
                queue.append(q)
    if end is None:
        return None
    result = set()
    while end is not None:
        result.add(end)
        end = previous[end]
    return result


def _scene(rng, owners, offsets, witness, concave_demo=False):
    n = len(owners)
    low = 14 if n == 1 else 23 if n == 2 else 28
    h, w = _ri(rng, low, 30), _ri(rng, low, 30)
    canvas = {(r, c) for r in range(h) for c in range(w)}
    objects, occupied = [], set()
    for i in range(n):
        forced = 'diamond' if witness and i == 0 else 'elbow' if concave_demo and i == 0 else None
        points, boundary, inside, kind = _shape(rng, forced)
        d = offsets[i]
        copy_boundary, copy_inside = _move(boundary, d), _move(inside, d)
        aperture = inside & copy_inside
        body = boundary | inside | copy_boundary | copy_inside
        if not aperture:
            return None
        candidates = [p for p in sorted(_move(set(points), d)) if p not in boundary
                      and sum(p in _halo({v}) for v in points) == 1]
        if not candidates:
            return None
        marker = _pick(rng, candidates)
        rh = max(r for r, c in body) - min(r for r, c in body) + 1
        cw = max(c for r, c in body) - min(c for r, c in body) + 1
        if rh > h or cw > w:
            return None
        placed = False
        for _ in range(80):
            dplace = (_ri(rng, 0, h-rh)-min(r for r, c in body), _ri(rng, 0, w-cw)-min(c for r, c in body))
            shifted = _move(body, dplace)
            if shifted & (occupied | _halo(occupied)):
                continue
            shift = lambda cells: _move(cells, dplace)
            objects.append(dict(vertices=shift(points), boundary=shift(boundary), inside=shift(inside),
                                copy_boundary=shift(copy_boundary), copy_inside=shift(copy_inside),
                                body=shifted, aperture=shift(aperture), marker=next(iter(shift({marker}))),
                                owner=owners[i], offset=d, kind=kind))
            occupied |= shifted
            placed = True
            break
        if not placed:
            return None
    nroutes = max(x for x in owners if x is not None) + 1
    wire_sets, source_points = [], []
    cups_blocked = occupied | _halo(occupied)
    for owner in range(nroutes):
        placed = False
        for _ in range(100):
            # A three-neighbor L cup rotates independently of the contours.
            square = {(0, 0), (0, 1), (1, 0), (1, 1)}
            source = _pick(rng, sorted(square))
            shift = (_ri(rng, 0, h-2), _ri(rng, 0, w-2))
            cup = _move(square, shift)
            if cup & cups_blocked:
                continue
            p = (source[0]+shift[0], source[1]+shift[1])
            source_points.append(p)
            wire_sets.append(cup - {p})
            cups_blocked |= cup | _halo(cup)
            placed = True
            break
        if not placed:
            return None
    markers = {t['marker'] for t in objects}
    for owner in rng.permutation(nroutes):
        owner = int(owner)
        own = wire_sets[owner]
        other = set().union(*(wire_sets[j] for j in range(nroutes) if j != owner))
        forbidden = occupied | other | _halo(other) | set(source_points) | _halo(set(source_points))
        forbidden |= markers | _halo(markers)
        for t in objects:
            if t['owner'] != owner:
                forbidden |= _halo(t['boundary'])
        allowed = (canvas - forbidden) | own
        indices = [i for i, t in enumerate(objects) if t['owner'] == owner]
        rng.shuffle(indices)
        for i in indices:
            goals = _halo(objects[i]['boundary']) & allowed
            if not goals:
                return None
            path = _path(rng, own, goals, allowed)
            if path is None:
                return None
            own |= path
        wire_sets[owner] = own
    palette = [int(x) for x in rng.permutation(10)]
    bg, wire = palette[:2]
    paints = palette[2:2+nroutes]
    remaining = palette[2+nroutes:]
    a = np.full((h, w), bg, dtype=np.int8)
    for i, t in enumerate(objects):
        t['color'], t['marker_color'] = remaining[2*i:2*i+2]
        for p in t['vertices']:
            a[p] = t['color']
        a[t['marker']] = t['marker_color']
    for cells in wire_sets:
        for p in cells:
            a[p] = wire
    for owner, p in enumerate(source_points):
        a[p] = paints[owner]
    # Construct labels from the sampled polygons and intended ownership.
    b = a.copy()
    all_original = set().union(*(t['boundary'] for t in objects))
    for t in objects:
        for p in t['copy_boundary'] - all_original:
            if a[p] == bg:
                b[p] = t['marker_color']
        for p in t['boundary']:
            b[p] = t['color']
        if t['owner'] is not None:
            for p in t['aperture']:
                b[p] = paints[t['owner']]
    # Fit to actual drawings/sources, retaining independently sampled margins.
    rr, cc = np.where(b != bg)
    r0, r1 = max(0, int(rr.min())-_ri(rng, 0, 2)), min(h, int(rr.max())+1+_ri(rng, 0, 2))
    c0, c1 = max(0, int(cc.min())-_ri(rng, 0, 2)), min(w, int(cc.max())+1+_ri(rng, 0, 2))
    a, b = a[r0:r1, c0:c1].copy(), b[r0:r1, c0:c1].copy()
    try:
        predicted, info = _analyze(a)
        if not np.array_equal(predicted, b):
            return None
        if witness and not any(_exclusive_cells(t) for t in info['targets'] if t['paint'] is not None):
            return None
        if concave_demo and (not _wrong_nearest(info) or not _gap_attachment(info)):
            return None
    except ValueError:
        return None
    return {'input': a, 'output': b}


def _exclusive_cells(t):
    strokes = t['boundary'] | t['copy_boundary']
    return (t['inside'] ^ t['copy_inside']) - strokes


def _wrong_nearest(info):
    """A closer, differently colored source cannot replace tracing the wire."""
    for t in info['targets']:
        if t['paint'] is None:
            continue
        distances = [(min(abs(p[0]-q[0])+abs(p[1]-q[1]) for q in t['boundary']), color, owner)
                     for p, color, owner in info['sources']]
        actual = next(d for d, color, owner in distances if owner == t['owner'])
        if any(d < actual and color != t['paint'] for d, color, owner in distances):
            return True
    return False


def _gap_attachment(info):
    return any(t['owner'] is not None and not (info['wires'][t['owner']] & _halo(t['vertices']))
               for t in info['targets'])


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(_integer(seed) and seed >= 0, 'seed must be a nonnegative integer')
    _require(_integer(num_train) and 2 <= num_train <= 10, 'num_train must be an integer in 2..10')
    _require(_integer(num_test) and 1 <= num_test <= 4, 'num_test must be an integer in 1..4')
    _require(num_train > num_test, 'num_train must exceed num_test')
    rng = np.random.default_rng(int(seed))
    held = _pick(rng, DIAGONALS)
    train_offsets = tuple(d for d in D8 if d != held)
    lessons = [int(x) for x in rng.permutation(int(num_train))[:2]]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(count):
            lesson = lessons.index(index) if split == 'train' and index in lessons else None
            if lesson == 0:
                owners = [0, 0]
            elif lesson == 1:
                owners = [0, 1, None]
            else:
                n = _ri(rng, 1, 3)
                owners = [0] + [_pick(rng, (0, 1, None)) for _ in range(n-1)]
                rng.shuffle(owners)
            offsets = [_pick(rng, train_offsets if split == 'train' else D8) for _ in owners]
            witness = lesson == 0 or split == 'test'
            if witness:
                offsets[0] = held if split == 'test' else _pick(rng, tuple(d for d in DIAGONALS if d != held))
                # The query's held-out registration always affects a supplied stencil.
                if owners[0] is None:
                    owners[0] = 0
            if lesson == 1:
                offsets[0] = _pick(rng, D4)
            for _ in range(96):
                pair = _scene(rng, owners, offsets, witness, concave_demo=lesson == 1)
                if pair is not None and _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('bounded scene construction exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def measure_pair(pair):
    predicted, info = _analyze(pair['input'])
    _require(np.array_equal(predicted, pair['output']), 'wrong output')
    ts = info['targets']
    return {'shape': list(predicted.shape), 'background': info['background'], 'wire': info['wire_color'],
            'targets': len(ts), 'sources': len(info['sources']),
            'offsets': [list(t['offset']) for t in ts], 'vertices': [len(t['vertices']) for t in ts],
            'aperture_areas': [len(t['aperture']) for t in ts],
            'supplied': sum(t['paint'] is not None for t in ts),
            'unsupplied': sum(t['paint'] is None for t in ts),
            'multi_target_route': any(sum(t['owner'] == i for t in ts) >= 2 for i in range(len(info['wires']))),
            'exclusive_witness': sum(len(_exclusive_cells(t)) for t in ts if t['paint'] is not None),
            'chords': sum(t['chords'] for t in ts),
            'wrong_nearest': _wrong_nearest(info), 'gap_attachment': _gap_attachment(info),
            'concave': sum(len({int(np.sign(_cross(t['order'][i-1], p, t['order'][(i+1) % len(t['order'])])))
                                for i, p in enumerate(t['order'])} - {0}) > 1 for t in ts),
            'wire_cells': sum(len(w) for w in info['wires']),
            'boundary_contact': any(r in (0, predicted.shape[0]-1) or c in (0, predicted.shape[1]-1)
                                    for t in ts for r, c in t['boundary'] | t['copy_boundary'])}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'bad puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'splits must be lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'bad pair counts')
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'bad pair keys')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape, 'shape changed')
            _require(_key(a) not in seen, 'duplicate input')
            seen.add(_key(a))
            expected, info = _analyze(a)
            _require(np.array_equal(expected, b), 'completion, registration, or supply mismatch')
            _require(bool(np.any(a != b)), 'identity pair')
            _require(np.array_equal(a[a != info['background']], b[a != info['background']]), 'input foreground overwritten')
            # Check interiors via a second geometric definition, and priority at overlaps.
            for t in info['targets']:
                _require(t['inside'] == _polygon_interior(t['order'], t['boundary']), 'raster/polygon interior disagreement')
                _require(all(b[p] == t['color'] for p in t['boundary']), 'source priority lost')
                if t['paint'] is not None:
                    _require(all(b[p] == t['paint'] for p in t['aperture']), 'partial aperture fill')
                _require(all(b[p] == info['background'] for p in _exclusive_cells(t)), 'paint leaked into an exclusive crescent')
            infos[split].append(info)
    training = infos['train']
    _require(any(len(i['sources']) == 2 for i in training), 'demonstrate separate paint supplies')
    _require(any(t['paint'] is None for i in training for t in i['targets']), 'demonstrate an unsupplied stencil')
    _require(any(sum(t['owner'] == j for t in i['targets']) >= 2 for i in training for j in range(len(i['wires']))),
             'demonstrate propagation to several stencils')
    _require(any(t['paint'] is not None and _exclusive_cells(t) for i in training for t in i['targets']),
             'demonstrate intersection rather than whole-contour fill')
    _require(any(t['chords'] for i in training for t in i['targets']), 'demonstrate chord rejection')
    _require(any(_wrong_nearest(i) for i in training), 'demonstrate routing over paint proximity')
    _require(any(_gap_attachment(i) for i in training), 'demonstrate attachment to an interpolated side')
    _require(any(len({int(np.sign(_cross(t['order'][j-1], p, t['order'][(j+1) % len(t['order'])])))
                      for j, p in enumerate(t['order'])} - {0}) > 1 for i in training for t in i['targets']),
             'demonstrate a concave stencil')
    train_offsets = {t['offset'] for i in training for t in i['targets']}
    _require(bool(train_offsets & set(D4)) and bool(train_offsets & set(DIAGONALS)), 'teach axial and diagonal registration')
    _require(any(t['paint'] is not None and t['offset'] not in train_offsets for i in infos['test'] for t in i['targets']),
             'query must transfer to an unseen registration direction')


def self_test():
    """42 in-memory seeds, symmetry checks, count boundaries, and bad-label probes."""
    measurements, hashes, shapes, offsets, geometries = [], set(), set(), set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in p[split]:
                a = pair['input']
                digest.update(bytes(a.shape) + a.tobytes() + pair['output'].tobytes())
                f = measure_pair(pair)
                measurements.append(f)
                shapes.add(tuple(f['shape']))
                offsets.update(tuple(d) for d in f['offsets'])
                _, info = _analyze(a)
                for t in info['targets']:
                    r0, c0 = min(r for r, c in t['vertices']), min(c for r, c in t['vertices'])
                    geometries.add(tuple(sorted((r-r0, c-c0) for r, c in t['vertices'])))
        hashes.add(digest.hexdigest())
    _require(len(hashes) == 42 and len(shapes) >= 40 and len(geometries) >= 60, 'insufficient structural variety')
    _require(offsets == set(D8), 'offset coverage')
    _require({f['targets'] for f in measurements} == {1, 2, 3}, 'target count coverage')
    _require({f['sources'] for f in measurements} == {1, 2}, 'source count coverage')
    _require(len({f['background'] for f in measurements}) == 10, 'background coverage')
    _require(sum(f['concave'] for f in measurements) > 10, 'concave coverage')
    _require(sum(f['boundary_contact'] for f in measurements) > 10, 'edge coverage')
    original = generate_puzzle(0)
    again = generate_puzzle(0)
    _require(all(np.array_equal(x[k], y[k]) for s in original for x, y in zip(original[s], again[s])
                 for k in ('input', 'output')), 'nondeterministic')
    for pair in original['train'] + original['test']:
        for k in range(4):
            for reflect in (False, True):
                a, b = np.rot90(pair['input'], k), np.rot90(pair['output'], k)
                if reflect:
                    a, b = np.fliplr(a), np.fliplr(b)
                _require(np.array_equal(solve(a), b), 'symmetry failure')
    for nt, nq in ((2, 1), (10, 1), (10, 4), (5, 4)):
        p = generate_puzzle(1000+nt*10+nq, nt, nq)
        _require(len(p['train']) == nt and len(p['test']) == nq, 'count mismatch')
    for nt, nq in ((1, 1), (11, 1), (4, 0), (10, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Omission of either parent must be visible: wrong color, empty aperture,
    # and full-source-interior fill are all rejected.
    bad_index = next(i for i, pair in enumerate(original['train'])
                     if any(t['paint'] is not None and _exclusive_cells(t) for t in _analyze(pair['input'])[1]['targets']))
    pair = original['train'][bad_index]
    _, info = _analyze(pair['input'])
    for i in range(3):
        bad = {s: [{k: np.array(v, copy=True) for k, v in q.items()} for q in original[s]] for s in original}
        target = next(t for t in info['targets'] if t['paint'] is not None)
        location = next(iter(target['aperture']))
        wrong = info['background'] if i == 0 else target['marker_color']
        if i == 2:
            options = [(t, q) for t in info['targets'] if t['paint'] is not None for q in _exclusive_cells(t)]
            target, location = options[0]
            wrong = target['paint']
        bad['train'][bad_index]['output'][location] = wrong
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('wrong answer accepted')
    return {'seeds': 42, 'pairs': len(measurements), 'distinct_puzzles': len(hashes),
            'grid_shapes': len(shapes), 'stencil_geometries': len(geometries), 'offsets': sorted(offsets),
            'concave_stencils': sum(f['concave'] for f in measurements),
            'exclusive_witness_cells': sum(f['exclusive_witness'] for f in measurements),
            'boundary_pairs': sum(f['boundary_contact'] for f in measurements),
            'semantic_checks': 'generator-owned; no independent solver or blind prediction'}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
