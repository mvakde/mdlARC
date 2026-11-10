"""Silhouette-keyed embroidery: input-only geometric dictionary completion."""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import numpy as np

D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _normal(cells):
    cells = tuple(cells)
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _rotate_point(r, c, k):
    for _ in range(k % 4):
        r, c = -c, r
    return r, c


def _rotated(cells, k):
    return tuple(sorted(_rotate_point(r, c, k) for r, c in cells))


def _canonical(cells):
    cells = tuple(cells)
    return min(_normal(_rotated(cells, k)) for k in range(4))


def _mirror_key(cells):
    return _canonical((r, -c) for r, c in cells)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in D8:
                p = r+dr, c+dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(tuple(sorted(cells)))
    return result


def _connected(cells, directions=D8):
    remaining = set(cells)
    if not remaining:
        return False
    stack = [min(remaining)]
    remaining.remove(stack[0])
    while stack:
        r, c = stack.pop()
        for dr, dc in directions:
            p = r+dr, c+dc
            if p in remaining:
                remaining.remove(p)
                stack.append(p)
    return not remaining


def _transform(motif, k=0, reflect=False):
    def point(r, c):
        return _rotate_point(r, -c if reflect else c, k)
    return {'body': tuple(sorted(point(r, c) for r, c in motif['body'])),
            'payload': tuple(sorted((*point(r, c), color)
                                    for r, c, color in motif['payload']))}


def _bounds(motif):
    cells = list(motif['body']) + [(r, c) for r, c, _ in motif['payload']]
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return r0, c0, max(r for r, c in cells)-r0+1, max(c for r, c in cells)-c0+1


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(int, copy=False)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _frames(grid):
    """Recognize rectangular borders and their empty inner rim geometrically."""
    h, w = grid.shape
    down, right = np.ones((h, w), int), np.ones((h, w), int)
    for r in range(h-2, -1, -1):
        down[r] += (grid[r] == grid[r+1]) * down[r+1]
    for c in range(w-2, -1, -1):
        right[:, c] += (grid[:, c] == grid[:, c+1]) * right[:, c+1]
    panels = []
    for r in range(h-6):
        for end_r in range(r+6, h):
            cols = np.flatnonzero(down[r] >= end_r-r+1)
            for index, c in enumerate(cols):
                for end_c in cols[index+1:]:
                    width = int(end_c-c+1)
                    if width < 7 or right[r, c] < width or right[end_r, c] < width:
                        continue
                    rectangle = grid[r:end_r+1, c:end_c+1]
                    inner = rectangle[1:-1, 1:-1]
                    fill, frame = int(inner[0, 0]), int(rectangle[0, 0])
                    if fill == frame:
                        continue
                    if not (np.all(inner[0] == fill) and np.all(inner[-1] == fill)
                            and np.all(inner[:, 0] == fill) and np.all(inner[:, -1] == fill)):
                        continue
                    if np.count_nonzero(inner == fill)*2 <= inner.size:
                        continue
                    components = _components(inner != fill)
                    if components:
                        panels.append({'box': (r, int(c), end_r-r+1, width),
                                       'fill': fill, 'frame': frame,
                                       'inner': inner, 'components': components})
    return panels


def _disjoint(a, b):
    r, c, h, w = a
    y, x, v, u = b
    return r+h <= y or y+v <= r or c+w <= x or x+u <= c


def _read_object(panel, cells, pivot, complete):
    inner, frame = panel['inner'], panel['frame']
    anchors = [(r, c) for r, c in cells if inner[r, c] == pivot]
    assert len(anchors) == 1
    ar, ac = anchors[0]
    body = tuple(sorted((r-ar, c-ac) for r, c in cells if inner[r, c] in (frame, pivot)))
    payload = tuple(sorted((r-ar, c-ac, int(inner[r, c])) for r, c in cells
                           if inner[r, c] not in (frame, pivot)))
    assert len(body) >= 3 and _connected(body)
    # The pivot must make the marked carrier's orientation unique.
    assert sum(_rotated(body, k) == body for k in range(4)) == 1
    assert bool(payload) == complete
    assert all(color != panel['fill'] for _, _, color in payload)
    return {'body': body, 'payload': payload, 'anchor': (ar, ac),
            'class': _canonical(body)}


def _dictionary(reference, pivot):
    templates, counts = {}, Counter()
    for cells in reference['components']:
        obj = _read_object(reference, cells, pivot, True)
        key = obj['class']
        if key in templates:
            old = templates[key]
            turns = [k for k in range(4) if _rotated(old['body'], k) == obj['body']]
            assert len(turns) == 1, 'inconsistent reference pivot'
            assert _transform(old, turns[0])['payload'] == obj['payload'], 'conflicting embroidery'
        else:
            templates[key] = obj
        counts[key] += 1
    return templates, counts


def _completion(info, mode='normal'):
    target, templates = info['target'], info['templates']
    inner = target['inner']
    out = np.full((inner.shape[0]+2, inner.shape[1]+2), target['frame'], dtype=int)
    out[1:-1, 1:-1] = inner
    matches = []
    for obj in info['objects']:
        proto = templates.get(obj['class'])
        if proto is None and mode == 'reflection':
            candidates = [t for t in templates.values() if _mirror_key(t['body']) == obj['class']]
            if candidates:
                proto = _transform(candidates[0], reflect=True)
        if proto is None:
            matches.append(None)
            continue
        turns = [k for k in range(4) if _rotated(proto['body'], k) == obj['body']]
        if mode == 'reflection' and obj['class'] not in templates and len(turns) != 1:
            # A mirror-only silhouette with a differently placed pivot does
            # not satisfy this marked-reflection alternative either.
            matches.append(None)
            continue
        assert len(turns) == 1, 'shape and pivot must have one alignment'
        k = turns[0]
        if mode == 'identity' and k != 0:
            matches.append(None)
            continue
        motif = _transform(proto, k)
        ar, ac = obj['anchor']
        for dr, dc, color in motif['payload']:
            r, c = ar+dr, ac+dc
            assert 1 <= r < inner.shape[0]-1 and 1 <= c < inner.shape[1]-1, 'clipped embroidery'
            assert color not in (info['reference']['fill'], target['fill'], target['frame'], info['pivot'])
            assert out[r+1, c+1] == target['fill'], 'embroidery overwrites another cell'
            out[r+1, c+1] = color
        matches.append(k)
    if mode == 'normal':
        assert any(k is not None for k in matches), 'at least one dictionary request'
        assert len(_components(out[1:-1, 1:-1] != target['fill'])) == len(info['objects']), 'completed motifs touch'
    return out, matches


def _interpret(input_grid):
    grid = _grid(input_grid)
    panels = _frames(grid)
    interpretations = []
    for target in panels:
        frame, inner = target['frame'], target['inner']
        nonframe = [[int(inner[r, c]) for r, c in comp if inner[r, c] != frame]
                    for comp in target['components']]
        if not all(len(values) == 1 for values in nonframe):
            continue
        pivots = {values[0] for values in nonframe}
        if len(pivots) != 1:
            continue
        pivot = pivots.pop()
        for reference in panels:
            if reference['frame'] != frame or not _disjoint(target['box'], reference['box']):
                continue
            try:
                assert pivot not in (reference['fill'], target['fill'], frame)
                templates, repetitions = _dictionary(reference, pivot)
                objects = [_read_object(target, comp, pivot, False) for comp in target['components']]
                info = {'reference': reference, 'target': target, 'pivot': pivot,
                        'templates': templates, 'repetitions': repetitions, 'objects': objects}
                out, matches = _completion(info)
                info['matches'] = matches
                interpretations.append((out, info))
            except AssertionError:
                continue
    assert len(interpretations) == 1, 'input must have a unique reference/target interpretation'
    return interpretations[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer from visible cells only; never consult generation state."""
    return _interpret(input_grid)[0]


def _sample_body(rng, limit=4, area=None, forbidden=(), chiral=False, diagonal=False):
    forbidden = set(forbidden)
    for _ in range(400):
        n = int(area if area is not None else rng.integers(3, min(12, limit*limit)+1))
        style = int(rng.integers(6))
        if area is None and style == 0:
            cells = {(0, c) for c in range(int(rng.integers(3, limit+1)))}
        elif area is None and style == 1 and limit >= 3:
            h, w = int(rng.integers(3, limit+1)), int(rng.integers(3, limit+1))
            cells = {(r, c) for r in range(h) for c in range(w)
                     if r in (0, h-1) or c in (0, w-1)}
        elif area is None and style == 2:
            h, w = int(rng.integers(2, limit+1)), int(rng.integers(2, limit+1))
            cells = {(r, 0) for r in range(h)} | {(h-1, c) for c in range(w)}
        else:
            cells = {(0, 0)}
            directions = D8 if style == 5 else D4
            while len(cells) < n:
                frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in directions} - cells)
                candidates = []
                for p in frontier:
                    trial = cells | {p}
                    if max(r for r, c in trial)-min(r for r, c in trial) < limit and max(c for r, c in trial)-min(c for r, c in trial) < limit:
                        candidates.append(p)
                if not candidates:
                    break
                cells.add(candidates[int(rng.integers(len(candidates)))])
        key = _canonical(cells)
        if (len(cells) >= 3 and key not in forbidden
                and (not diagonal or not _connected(cells, D4))
                and (not chiral or key != _mirror_key(cells))):
            return _normal(cells)
    # Deterministic finite fallback: all eight-connected 3x3 masks.
    for bits in range(1, 512):
        cells = tuple((r, c) for r in range(3) for c in range(3) if bits & (1 << (3*r+c)))
        if len(cells) < 3 or (area is not None and len(cells) != area) or not _connected(cells):
            continue
        if (_canonical(cells) not in forbidden
                and (not diagonal or not _connected(cells, D4))
                and (not chiral or _canonical(cells) != _mirror_key(cells))):
            return _normal(cells)
    # An eight-cell diagonal carrier needs more than a 3x3 envelope.
    if diagonal and area == 8 and limit >= 4:
        for core in combinations([(r, c) for r in range(1, 4) for c in range(1, 4)], 7):
            cells = ((0, 0),) + core
            if (_connected(cells) and not _connected(cells, D4)
                    and _canonical(cells) not in forbidden
                    and (not chiral or _canonical(cells) != _mirror_key(cells))):
                return _normal(cells)
    raise RuntimeError('carrier sampling exhausted')


def _motif(rng, body, decor_colors, small=False):
    pivots = []
    for ar, ac in body:
        relative = tuple(sorted((r-ar, c-ac) for r, c in body))
        if sum(_rotated(relative, k) == relative for k in range(4)) == 1:
            pivots.append((ar, ac))
    ar, ac = pivots[int(rng.integers(len(pivots)))]
    carrier = set((r-ar, c-ac) for r, c in body)
    occupied, embroidery = set(carrier), []
    n = int(rng.integers(1, 4 if small else 7))
    for _ in range(n):
        candidates = sorted({(r+dr, c+dc) for r, c in occupied for dr, dc in D8} - occupied)
        candidates = [p for p in candidates
                      if max(r for r, c in occupied | {p})-min(r for r, c in occupied | {p}) < 7
                      and max(c for r, c in occupied | {p})-min(c for r, c in occupied | {p}) < 7]
        # Favor edge-connected embellishments; exclusively diagonal cells remain legal.
        weights = np.array([3 if any((p[0]+dr, p[1]+dc) in occupied for dr, dc in D4) else 1
                            for p in candidates], dtype=float)
        p = candidates[int(rng.choice(len(candidates), p=weights/weights.sum()))]
        occupied.add(p)
        embroidery.append(p)
    color_count = int(rng.integers(1, min(3, len(embroidery), len(decor_colors))+1))
    palette = list(rng.choice(decor_colors, size=color_count, replace=False))
    colors = palette + [palette[int(rng.integers(color_count))] for _ in range(len(embroidery)-color_count)]
    rng.shuffle(colors)
    return {'body': tuple(sorted(carrier)),
            'payload': tuple(sorted((r, c, int(color)) for (r, c), color in zip(embroidery, colors)))}


def _shelf(motifs, width):
    row, col, shelf_height = 2, 2, 0
    placements = []
    for motif in motifs:
        r0, c0, h, w = _bounds(motif)
        if w+4 > width:
            return None
        if col+w > width-2:
            row += shelf_height+1
            col, shelf_height = 2, 0
        placements.append((row-r0, col-c0))
        col += w+1
        shelf_height = max(shelf_height, h)
    return max(7, row+shelf_height+2), placements


def _panel_options(motifs):
    result = []
    cells = sum(len(m['body'])+len(m['payload']) for m in motifs)
    for w in range(7, 29):
        packed = _shelf(motifs, w)
        if packed:
            h, places = packed
            if h <= 28 and (h-2)*(w-2) > 2*cells:
                result.append((h, w, places))
    return result


def _geometry(rng, references, targets, preferred):
    ro, to = _panel_options(references), _panel_options(targets)
    feasible = [(axis, a, b) for axis in (0, 1) for a in ro for b in to
                if a[axis]+b[axis]+3 <= 30]
    if not feasible:
        return None
    filtered = [v for v in feasible if v[0] == preferred]
    feasible = filtered or feasible
    areas = np.array([(a[axis]+b[axis]+3)*(max(a[1-axis], b[1-axis])+2)
                      for axis, a, b in feasible], dtype=float)
    temperature = float(rng.choice([70, 180, 600]))
    weights = np.exp(-(areas-areas.min())/temperature)
    axis, a, b = feasible[int(rng.choice(len(feasible), p=weights/weights.sum()))]
    shapes = [list(a[:2]), list(b[:2])]
    for i in (0, 1):
        for d in (0, 1):
            slack = 27-shapes[0][d]-shapes[1][d] if d == axis else 28-shapes[i][d]
            shapes[i][d] += int(rng.integers(min(3, slack)+1))
    gap = int(rng.integers(1, min(3, 28-shapes[0][axis]-shapes[1][axis])+1))
    required = [max(s[0] for s in shapes)+2, max(s[1] for s in shapes)+2]
    required[axis] = shapes[0][axis]+shapes[1][axis]+gap+2
    canvas = [int(rng.integers(n, min(30, n+4)+1)) for n in required]
    if rng.random() < .08:
        canvas = [30, 30]
    positions = [[0, 0], [0, 0]]
    order = [0, 1] if rng.random() < .5 else [1, 0]
    start = int(rng.integers(1, canvas[axis]-shapes[0][axis]-shapes[1][axis]-gap))
    positions[order[0]][axis] = start
    positions[order[1]][axis] = start+shapes[order[0]][axis]+gap
    other = 1-axis
    for i in (0, 1):
        positions[i][other] = int(rng.integers(1, canvas[other]-shapes[i][other]))
    placements = []
    for motifs, option, shape in ((references, a, shapes[0]), (targets, b, shapes[1])):
        # Free scattering uses the shelf as a finite, guaranteed fallback.
        occupied, places = set(), []
        for motif in motifs:
            r0, c0, h, w = _bounds(motif)
            candidates = [(r, c) for r in range(2, shape[0]-h-1) for c in range(2, shape[1]-w-1)]
            rng.shuffle(candidates)
            for r, c in candidates:
                box = {(y, x) for y in range(r, r+h) for x in range(c, c+w)}
                halo = {(y+dr, x+dc) for y, x in box for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
                if not occupied & halo:
                    occupied.update(box)
                    places.append((r-r0, c-c0))
                    break
            else:
                dr = int(rng.integers(shape[0]-option[0]+1))
                dc = int(rng.integers(shape[1]-option[1]+1))
                places = [(r+dr, c+dc) for r, c in option[2]]
                break
        placements.append(places)
    return tuple(canvas), [tuple(positions[i]+shapes[i]) for i in (0, 1)], placements


def _draw(grid, motif, anchor, body_color, pivot, payload):
    ar, ac = anchor
    for r, c in motif['body']:
        grid[ar+r, ac+c] = pivot if (r, c) == (0, 0) else body_color
    if payload:
        for r, c, color in motif['payload']:
            grid[ar+r, ac+c] = color


def _make_pair(rng, profile, config):
    for attempt in range(100):
        palette = config['palette'].copy()
        if not config['stable']:
            rng.shuffle(palette)
        body_color, pivot, rf, tf = map(int, palette[:4])
        if profile != 'contrast' and rng.random() < .35:
            tf = rf
        decor_colors = [int(c) for c in palette if c not in (body_color, pivot, rf, tf)]
        small = attempt >= 20 or rng.random() < .35
        limit = 3 if attempt >= 20 else int(rng.choice([3, 4, 5], p=[.35, .5, .15]))
        ntypes = 2 if profile in ('contrast', 'pivot') else 3 if profile == 'test' else int(rng.integers(1, 4))
        bodies, forbidden = [], set()
        if profile == 'pivot':
            bodies.append(((0, 0), (0, 1), (1, 0), (1, 1)))
        elif profile == 'contrast':
            # A 3x3 mask with eight cells is always achiral. Leave at least
            # two vacancies when demanding a reflection-negative witness.
            bodies.append(_sample_body(rng, limit, area=int(rng.integers(4, min(8, limit*limit-2)+1)), chiral=True))
        else:
            bodies.append(_sample_body(rng, limit))
        forbidden.add(_canonical(bodies[0]))
        if profile == 'contrast':
            forbidden.add(_mirror_key(bodies[0]))
        for i in range(1, ntypes):
            b = _sample_body(rng, limit, area=len(bodies[0]) if profile == 'contrast' else None,
                             forbidden=forbidden, diagonal=profile == 'contrast')
            bodies.append(b)
            forbidden.add(_canonical(b))
        templates = [_motif(rng, b, decor_colors, small) for b in bodies]
        refs, targets = [], []
        first_turns = []
        for i, template in enumerate(templates):
            k = int(rng.integers(4))
            first_turns.append(k)
            refs.append(_transform(template, k))
            if profile not in ('contrast', 'pivot') and rng.random() < .22:
                refs.append(_transform(template, int(rng.integers(4))))
        if profile == 'contrast':
            targets = [_transform(templates[0], (first_turns[0]+int(rng.choice([1, 3]))) % 4),
                       _transform(templates[1], int(rng.integers(4))),
                       _transform(templates[0], int(rng.integers(4)), reflect=True)]
            authorized = [True, True, False]
        elif profile == 'pivot':
            unknown_forbidden = forbidden | {_mirror_key(b) for b in bodies}
            unknown_body = _sample_body(rng, limit, forbidden=unknown_forbidden)
            u = _motif(rng, unknown_body, decor_colors, True)
            u['payload'] = ()
            targets = [_transform(templates[0], first_turns[0]),
                       _transform(templates[0], (first_turns[0]+1) % 4),
                       u, _transform(u, int(rng.integers(4)))]
            authorized = [True, True, False, False]
        else:
            count = int(rng.integers(4, 7) if profile == 'test' else rng.integers(2, 6))
            unknown_count = int(rng.integers(0, min(2, count-1)+1))
            for _ in range(count-unknown_count):
                i = int(rng.integers(ntypes))
                targets.append(_transform(templates[i], int(rng.integers(4))))
            authorized = [True]*len(targets)
            for _ in range(unknown_count):
                ub = _sample_body(rng, limit, forbidden=forbidden)
                u = _motif(rng, ub, decor_colors, True)
                u['payload'] = ()
                targets.append(_transform(u, int(rng.integers(4))))
                authorized.append(False)
        # Shuffle complete occurrences and query carriers independently.
        refs = [refs[int(i)] for i in rng.permutation(len(refs))]
        order = [int(i) for i in rng.permutation(len(targets))]
        targets, authorized = [targets[i] for i in order], [authorized[i] for i in order]
        preferred = int(rng.integers(2)) if config['layout'] is None else config['layout']
        geometry = _geometry(rng, refs, targets, preferred)
        if geometry is None:
            continue
        shape, boxes, places = geometry
        inp = np.full(shape, rf, dtype=int)
        for box, fill in zip(boxes, (rf, tf)):
            r, c, h, w = box
            inp[r:r+h, c:c+w] = body_color
            inp[r+1:r+h-1, c+1:c+w-1] = fill
        for motif, anchor in zip(refs, places[0]):
            r, c, _, _ = boxes[0]
            _draw(inp, motif, (r+anchor[0], c+anchor[1]), body_color, pivot, True)
        r, c, h, w = boxes[1]
        for motif, anchor in zip(targets, places[1]):
            _draw(inp, motif, (r+anchor[0], c+anchor[1]), body_color, pivot, False)
        out = inp[r:r+h, c:c+w].copy()
        for motif, anchor, allowed in zip(targets, places[1], authorized):
            if allowed:
                _draw(out, motif, anchor, body_color, pivot, True)
        try:
            predicted, info = _interpret(inp)
            assert np.array_equal(predicted, out)
            f = _features(info, out)
            if profile == 'contrast':
                assert f['mirror_negative'] and f['quarter_turn'] and f['transparency'] and f['equal_area_keys']
                assert f['different_templates'] and f['matched_diagonal']
            if profile == 'pivot':
                assert f['pivot_matters'] and f['duplicate_unknown'] and f['unused_keys']
            return {'input': inp, 'output': out}
        except AssertionError:
            continue
    raise RuntimeError('bounded scene construction exhausted')


def _features(info, output):
    templates, objects, matches = info['templates'], info['objects'], info['matches']
    unknown = [o for o, k in zip(objects, matches) if k is None]
    requested = {o['class'] for o, k in zip(objects, matches) if k is not None}
    unknown_counts = Counter(o['class'] for o in unknown)
    symmetric = {key for key, t in templates.items()
                 if any(_normal(_rotated(t['body'], k)) == _normal(t['body']) for k in (1, 2, 3))}
    pivot_matters = any(len({k for o, k in zip(objects, matches) if o['class'] == key}) > 1
                        and len({_transform(templates[key], k)['payload'] for o, k in zip(objects, matches)
                                 if o['class'] == key and k is not None}) > 1 for key in symmetric)
    transparency = False
    for o, k in zip(objects, matches):
        if k is None:
            continue
        t = _transform(templates[o['class']], k)
        _, _, h, w = _bounds(t)
        transparency |= h*w > len(t['body'])+len(t['payload'])
    rbox, tbox = info['reference']['box'], info['target']['box']
    return {'reference_types': len(templates), 'target_count': len(objects),
            'matched_count': sum(k is not None for k in matches), 'unknown_count': len(unknown),
            'quarter_turn': any(k in (1, 3) for k in matches),
            'mirror_negative': any(o['class'] == _mirror_key(t['body']) and
                                   _canonical(t['body']) != _mirror_key(t['body']) and
                                   any(_rotated(_transform(t, reflect=True)['body'], k) == o['body'] for k in range(4))
                                   for o in unknown for t in templates.values()),
            'duplicate_unknown': any(n >= 2 for n in unknown_counts.values()),
            'unused_keys': bool(set(templates)-requested), 'pivot_matters': pivot_matters,
            'transparency': transparency and info['reference']['fill'] != info['target']['fill'],
            'equal_area_keys': len({len(t['body']) for t in templates.values()}) < len(templates),
            'different_templates': len({t['payload'] for t in templates.values()}) >= 2,
            'layout': 0 if rbox[0]+rbox[2] <= tbox[0] or tbox[0]+tbox[2] <= rbox[0] else 1,
            'reference_first': rbox[0] < tbox[0] if rbox[0] != tbox[0] and
                               (rbox[0]+rbox[2] <= tbox[0] or tbox[0]+tbox[2] <= rbox[0]) else rbox[1] < tbox[1],
            'repetitions': tuple(info['repetitions'].values()),
            'body_shapes': tuple(_canonical(t['body']) for t in templates.values()),
            'body_sizes': tuple(len(t['body']) for t in templates.values()),
            'payload_sizes': tuple(len(t['payload']) for t in templates.values()),
            'payload_colors': tuple(len({color for _, _, color in t['payload']}) for t in templates.values()),
            'diagonal': any(not _connected(t['body'], D4) for t in templates.values()),
            'matched_diagonal': any(k is not None and not _connected(o['body'], D4)
                                    for o, k in zip(objects, matches)),
            'symmetric_key': bool(symmetric), 'rotations': tuple(k for k in matches if k is not None),
            'role_profile': (info['reference']['frame'], info['pivot'], info['reference']['fill'], info['target']['fill']),
            'output_shape': output.shape}


def _check_counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
            raise ValueError('counts must be integers: train 2..10, test 1..4')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a non-negative integer')
    rng = np.random.default_rng(seed)
    config = {'palette': rng.permutation(10), 'stable': rng.random() < .2,
              'layout': None if rng.random() < .5 else int(rng.integers(2))}
    puzzle, seen = {'train': [], 'test': []}, set()
    profiles = ['contrast', 'pivot'] + ['random']*(int(num_train)-2)
    for split, sequence in (('train', profiles), ('test', ['test']*int(num_test))):
        for profile in sequence:
            for _ in range(10):
                pair = _make_pair(rng, profile, config)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct input construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, evidence = set(), []
    rotation_witness = transfer_witness = reflection_witness = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            predicted, info = _interpret(inp)
            assert np.array_equal(out, predicted), 'incorrect completed target'
            target = info['target']
            r, c, h, w = target['box']
            bare = inp[r:r+h, c:c+w]
            assert out.shape == (h, w) and not np.array_equal(out, bare)
            # Exact preservation of every original target cell, including frame and pivots.
            original = bare != target['fill']
            assert np.array_equal(out[original], bare[original])
            expected_added = sum(len(info['templates'][o['class']]['payload'])
                                 for o, k in zip(info['objects'], info['matches']) if k is not None)
            assert np.count_nonzero(out != bare) == expected_added
            for panel in (info['reference'], target):
                pr, pc, ph, pw = panel['box']
                assert pr >= 1 and pc >= 1 and pr+ph < inp.shape[0] and pc+pw < inp.shape[1]
            exterior = np.ones(inp.shape, dtype=bool)
            for panel in (info['reference'], target):
                pr, pc, ph, pw = panel['box']
                exterior[pr:pr+ph, pc:pc+pw] = False
            assert np.all(inp[exterior] == info['reference']['fill'])
            forbidden_colors = {info['reference']['fill'], target['fill'], target['frame'], info['pivot']}
            assert all(color not in forbidden_colors for t in info['templates'].values()
                       for _, _, color in t['payload'])
            if split == 'train':
                f = _features(info, out)
                evidence.append(f)
                rotation_witness |= not np.array_equal(_completion(info, 'identity')[0], out)
                transfer_witness |= not np.array_equal(bare, out)
                if f['mirror_negative']:
                    reflection_witness |= not np.array_equal(_completion(info, 'reflection')[0], out)
    assert any(f['quarter_turn'] and f['mirror_negative'] and f['transparency']
               and f['equal_area_keys'] and f['different_templates'] and f['matched_diagonal'] for f in evidence)
    assert any(f['pivot_matters'] and f['duplicate_unknown'] and f['unused_keys'] for f in evidence)
    assert rotation_witness and transfer_witness and reflection_witness


def self_test():
    """Bounded cell-derived checks, including diversity over exactly 42 seeds."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    fingerprints, records, input_shapes, body_shapes, roles = set(), [], set(), set(), set()
    for seed, puzzle in enumerate(puzzles):
        fingerprint = tuple((_key(p['input']), _key(p['output'])) for s in ('train', 'test') for p in puzzle[s])
        fingerprints.add(fingerprint)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                original = pair['input'].copy()
                prediction, info = _interpret(pair['input'])
                assert np.array_equal(prediction, pair['output'])
                assert np.array_equal(original, pair['input']), 'solver mutated its input'
                f = _features(info, prediction)
                records.append(f)
                input_shapes.add(pair['input'].shape)
                body_shapes.update(f['body_shapes'])
                roles.add(f['role_profile'])
        again = generate_puzzle(seed)
        assert fingerprint == tuple((_key(p['input']), _key(p['output'])) for s in ('train', 'test') for p in again[s])
    assert len(fingerprints) == 42
    assert len(input_shapes) >= 35 and len(body_shapes) >= 100 and len(roles) >= 80
    assert {r['reference_types'] for r in records} == {1, 2, 3}
    assert {r['target_count'] for r in records} >= {2, 3, 4, 5, 6}
    assert {r['layout'] for r in records} == {0, 1}
    assert {r['reference_first'] for r in records} == {False, True}
    assert {k for r in records for k in r['rotations']} == {0, 1, 2, 3}
    assert {n for r in records for n in r['payload_sizes']} >= {1, 2, 3, 4, 5, 6}
    assert {n for r in records for n in r['payload_colors']} == {1, 2, 3}
    assert any(r['diagonal'] for r in records) and any(r['symmetric_key'] for r in records)
    assert any(2 in r['repetitions'] for r in records)
    assert any(r['role_profile'][2] == r['role_profile'][3] for r in records)
    assert any(r['unknown_count'] == 0 for r in records)
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(1000+a*10+b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # A corrupted target pigment must be rejected by the family validator.
    bad = {s: [{name: grid.copy() for name, grid in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
