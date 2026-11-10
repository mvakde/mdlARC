"""Exchange wallpaper: two local states woven in opaque, ordered L fields.

No parent module, file, seed, or training label is used by the input-only solver.
Reference motifs are a visible legend. Their shared scaffold fixes pattern phase.
"""
from math import gcd, isqrt
from functools import reduce
import numpy as np


HOST_SEEDS = (0, 905255776, 956901903, 233105573, 323519590, 192676092,
    724518931, 903279629, 442747748, 942073958, 419946433, 663203421,
    787551069, 145308166, 266554754, 389039890, 913619662, 413951747,
    638012082, 238005214, 457944374, 730500905, 76565875, 728537353,
    471240899, 400510104, 928071247, 146585772, 181819919, 210343506,
    133886998, 625260729, 871996463, 792301296, 528841156, 365332364,
    783732969, 973450789, 15251929, 253836756, 541600478, 101511057)


def _integer(v):
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


def _counts(nt, ns):
    if not _integer(nt) or not 2 <= nt <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(ns) or not 1 <= ns <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if nt <= ns:
        raise ValueError("num_train must be greater than num_test")
    return int(nt), int(ns)


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or any(n < 1 or n > 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer grid with colors 0..9")
    return g.astype(int, copy=True)


def _key(g):
    return g.shape, tuple(map(int, g.flat))


def _bounds(points):
    return (min(y for y, x in points), min(x for y, x in points),
            max(y for y, x in points), max(x for y, x in points))


def _inside(y, x, b):
    return b[0] <= y <= b[2] and b[1] <= x <= b[3]


def _overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _marker_cells(b, s, diagonal):
    y0, x0, y1, x1 = b
    if diagonal == 0:
        starts = [(y0, x0), (y0, x0+s), (y0+s, x0),
                  (y1-s+1, x1-s+1), (y1-s+1, x1-2*s+1),
                  (y1-2*s+1, x1-s+1)]
    else:
        starts = [(y0, x1-s+1), (y0, x1-2*s+1), (y0+s, x1-s+1),
                  (y1-s+1, x0), (y1-s+1, x0+s), (y1-2*s+1, x0)]
    return {(y+dy, x+dx) for y, x in starts
            for dy in range(s) for dx in range(s)}


def _recognize_corners(points):
    if not points or len(points) % 6:
        return None
    s = isqrt(len(points) // 6)
    if s < 1 or 6*s*s != len(points):
        return None
    b = _bounds(points)
    if min(b[2]-b[0]+1, b[3]-b[1]+1) < 4*s:
        return None
    for diagonal in (0, 1):
        if points == _marker_cells(b, s, diagonal):
            return dict(bounds=b, scale=s, diagonal=diagonal, markers=points)
    return None


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, todo = {start}, [start]
        while todo:
            y, x = todo.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not (dy or dx):
                        continue
                    p = y+dy, x+dx
                    if p in remaining:
                        remaining.remove(p)
                        group.add(p)
                        todo.append(p)
        groups.append(group)
    return groups


def _primitive(crop):
    lengths = []
    for arr in (crop, crop.T):
        for row in arr:
            edges = [0] + (np.flatnonzero(row[1:] != row[:-1])+1).tolist() + [len(row)]
            lengths.extend(b-a for a, b in zip(edges, edges[1:]))
    s = reduce(gcd, lengths)
    small = crop[::s, ::s]
    assert np.array_equal(crop, np.repeat(np.repeat(small, s, 0), s, 1))
    return s, small


def _infer_templates(scene, bg):
    groups = _components(scene != bg)
    assert len(groups) == 2, "exactly two reference motifs are required"
    maps, boxes, scales = [], [], []
    for group in groups:
        b = _bounds(group)
        crop = scene[b[0]:b[2]+1, b[1]:b[3]+1]
        assert np.count_nonzero(crop != bg) == len(group), "interleaved reference motifs"
        scale, small = _primitive(crop)
        maps.append({(int(y), int(x)): int(small[y, x])
                     for y, x in np.argwhere(small != bg)})
        boxes.append(b)
        scales.append(scale)
    assert scales[0] == scales[1], "reference states must share a scale"
    a, b = maps
    shifts = set()
    for color in set(a.values()) & set(b.values()):
        pa = {p for p, c in a.items() if c == color}
        pb = {p for p, c in b.items() if c == color}
        aa, bb = _bounds(pa), _bounds(pb)
        shift = aa[0]-bb[0], aa[1]-bb[1]
        if pa == {(y+shift[0], x+shift[1]) for y, x in pb}:
            shifts.add(shift)
    candidates = []
    for dy, dx in sorted(shifts):
        aligned = {(y+dy, x+dx): c for (y, x), c in b.items()}
        common = {p for p, c in a.items() if aligned.get(p) == c}
        if len(common) >= 3:
            candidates.append((len(common), aligned, common))
    assert candidates, "no common scaffold"
    score = max(c[0] for c in candidates)
    best = [c for c in candidates if c[0] == score]
    assert len(best) == 1, "tied scaffold registration"
    _, b, common = best[0]
    y0, x0, _, _ = _bounds(common)
    templates = [{(y-y0, x-x0): c for (y, x), c in m.items()} for m in (a, b)]
    assert templates[0] != templates[1], "reference states must differ"
    union = set(templates[0]) | set(templates[1])
    ub = _bounds(union)
    period = ub[2]-ub[0]+2, ub[3]-ub[1]+2  # union extent plus one blank seam
    exclusive = [set(templates[i].values()) - set(templates[1-i].values()) for i in (0, 1)]
    assert all(exclusive), "each state needs an exclusive selector color"
    return templates, period, exclusive, boxes, scales[0]


def _read_input(g):
    bg = int(np.argmax(np.bincount(g.ravel(), minlength=10)))
    assert np.count_nonzero(g == bg)*2 > g.size, "input background must be a strict majority"
    candidates = []
    h, w = g.shape
    # Table position is not assumed; each cue must decode to one full L pair.
    for x in range(w-1):
        rows = np.flatnonzero(g[:, x] != bg)
        for run in np.split(rows, np.flatnonzero(np.diff(rows) != 1)+1):
            if not 1 <= len(run) <= 5:
                continue
            y0, y1 = int(run[0]), int(run[-1])
            cues = list(map(int, g[y0:y1+1, x]))
            selectors = list(map(int, g[y0:y1+1, x+1]))
            if len(set(cues)) != len(cues) or bg in selectors or set(cues) & set(selectors):
                continue
            table = (y0, x, y1, x+1)
            if any(g[y, xx] != bg
                   for y in range(max(0, y0-1), min(h, y1+2))
                   for xx in range(max(0, x-1), min(w, x+3))
                   if not _inside(y, xx, table)):
                continue
            fields = []
            valid = True
            for cue, selector in zip(cues, selectors):
                points = {tuple(map(int, p)) for p in np.argwhere(g == cue)
                          if not _inside(int(p[0]), int(p[1]), table)}
                field = _recognize_corners(points)
                if field is None or _overlap(field['bounds'], table):
                    valid = False
                    break
                fields.append(dict(field, cue=cue, selector=selector))
            if not valid:
                continue
            scene = g.copy()
            scene[y0:y1+1, x:x+2] = bg
            for field in fields:
                for p in field['markers']:
                    scene[p] = bg
            try:
                templates, period, exclusive, refs, ref_scale = _infer_templates(scene, bg)
                assert not set(cues) & (set(templates[0].values()) | set(templates[1].values()))
                assert all(not _overlap(ref, f['bounds']) for ref in refs for f in fields)
                for f in fields:
                    states = [i for i in (0, 1) if f['selector'] in exclusive[i]]
                    assert len(states) == 1, "selector does not identify exactly one state"
                    f['state'] = states[0]
                candidates.append(dict(background=bg, table=table, fields=fields,
                    templates=templates, period=period, reference_boxes=refs, reference_scale=ref_scale))
            except AssertionError:
                continue
    assert len(candidates) == 1, "input has no unique table/reference interpretation"
    return candidates[0]


def _pattern_maps(info, bbox_phase=False):
    ph, pw = info['period']
    maps = []
    for template in info['templates']:
        by, bx = (0, 0)
        if bbox_phase:
            by, bx, _, _ = _bounds(template)
        maps.append({((y-by) % ph, (x-bx) % pw): c for (y, x), c in template.items()})
    return maps


def _predict(g, info, *, exchange=True, reverse=False, transparent=False,
             bbox_phase=False, unit_scale=False):
    """Pixel-first reader; each pixel takes its foremost covering layer."""
    result = g.copy()
    bg = info['background']
    maps = _pattern_maps(info, bbox_phase)
    ph, pw = info['period']
    fields = info['fields'][::-1] if reverse else info['fields']
    for y in range(g.shape[0]):
        for x in range(g.shape[1]):
            for field in fields:
                b = field['bounds']
                if not _inside(y, x, b):
                    continue
                if (y, x) in field['markers']:
                    value = field['cue']
                else:
                    s = 1 if unit_scale else field['scale']
                    p = ((y-b[0])//s % ph, (x-b[1])//s % pw)
                    state = 1-field['state'] if exchange else field['state']
                    value = maps[state].get(p, bg)
                if transparent and value == bg:
                    continue
                result[y, x] = value
                break
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using only the input: discover keys, align states, weave and layer."""
    g = _grid(input_grid)
    return _predict(g, _read_input(g))


def _draw(g, mapping, anchor, s=1):
    for (y, x), color in mapping.items():
        yy, xx = anchor[0]+y*s, anchor[1]+x*s
        g[yy:yy+s, xx:xx+s] = color


def _compose(g, bg, fields, templates, period):
    """Ground-truth construction: render translated motifs, then whole layers.

    This uses lattice copies rather than the solver's per-pixel modular lookup.
    """
    out = g.copy()
    ph, pw = period
    for field in reversed(fields):
        y0, x0, y1, x1 = field['bounds']
        h, w, s = y1-y0+1, x1-x0+1, field['scale']
        # Padding lets negative logical endpoints enter from earlier tile copies.
        pad = 2*max(ph, pw)*s
        layer = np.full((h+2*pad, w+2*pad), bg, dtype=int)
        for iy in range(-2, (h+s*ph-1)//(s*ph)+2):
            for ix in range(-2, (w+s*pw-1)//(s*pw)+2):
                _draw(layer, templates[1-field['state']],
                      (pad+iy*ph*s, pad+ix*pw*s), s)
        out[y0:y1+1, x0:x1+1] = layer[pad:pad+h, pad:pad+w]
        for p in field['markers']:
            out[p] = field['cue']
    return out


def _design(rng, change):
    """Procedural connected scaffolds with pair-local colors and decorations."""
    h, w = int(rng.integers(2, 5)), int(rng.integers(2, 5))
    cy, cx = h//2, w//2
    kind = int(rng.integers(4))
    if kind == 0:
        core = {(cy, x) for x in range(w)} | {(y, cx) for y in range(h)}
    elif kind == 1:
        core = {(y, 0) for y in range(h)} | {(h-1, x) for x in range(w)}
    elif kind == 2:
        core = {(y, min(y, w-1)) for y in range(h)} | {(h-1, x) for x in range(w)}
    else:
        core = {(0, x) for x in range(w)} | {(y, cx) for y in range(h)}
    while len(core) < 4:
        frontier = sorted({(y+dy, x+dx) for y, x in core
                           for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                           if 0 <= y+dy < h and 0 <= x+dx < w} - core)
        core.add(frontier[int(rng.integers(len(frontier)))])
    for _ in range(int(rng.integers(3))):
        frontier = sorted({(y+dy, x+dx) for y, x in core
                           for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                           if 0 <= y+dy < h and 0 <= x+dx < w} - core)
        if frontier:
            core.add(frontier[int(rng.integers(len(frontier)))])
    def endpoint(side):
        b = _bounds(core)
        choices = sorted(p for p in core if
            (p[0] == b[0] if side == 0 else p[0] == b[2] if side == 1
             else p[1] == b[1] if side == 2 else p[1] == b[3]))
        root = choices[int(rng.integers(len(choices)))]
        dy, dx = ((-1, 0), (1, 0), (0, -1), (0, 1))[side]
        result = {(root[0]+dy, root[1]+dx)}
        if rng.random() < .45:
            lateral = (0, 1) if dy else (1, 0)
            result.add((root[0]+dy+lateral[0], root[1]+dx+lateral[1]))
        return result
    side = int(rng.integers(4))
    ea = endpoint(side)
    eb = ea if change == 'recolor' else endpoint((side+int(rng.integers(1, 4))) % 4)
    # A common interior location recolors only in the combined mode.
    mutable = {sorted(core)[int(rng.integers(len(core)))]} if change == 'combined' else set()
    common = core-mutable
    palette = list(map(int, rng.permutation(10)))
    bg, carrier, ca, cb = palette[:4]
    common_map = {p: carrier for p in common}
    next_color = 4
    if len(common) >= 5 and rng.random() < .5:
        common_map[sorted(common)[int(rng.integers(len(common)))]] = palette[next_color]
        next_color += 1
    maps = []
    for points, accent in ((ea | mutable, ca), (eb | mutable, cb)):
        mapping = dict(common_map)
        mapping.update({p: accent for p in points})
        if len(points) > 1 and rng.random() < .4:
            # Keep the selector present while adding a second changing accent.
            mapping[max(points)] = palette[next_color]
            next_color += 1
        maps.append(mapping)
    rotation, reflection = int(rng.integers(4)), bool(rng.integers(2))
    def transform(p):
        y, x = p
        for _ in range(rotation):
            y, x = x, -y
        return y, -x if reflection else x
    maps = [{transform(p): c for p, c in m.items()} for m in maps]
    same = {p for p, c in maps[0].items() if maps[1].get(p) == c}
    y0, x0, _, _ = _bounds(same)
    maps = [{(y-y0, x-x0): c for (y, x), c in m.items()} for m in maps]
    ub = _bounds(set(maps[0]) | set(maps[1]))
    period = ub[2]-ub[0]+2, ub[3]-ub[1]+2
    used = {bg} | set(maps[0].values()) | set(maps[1].values())
    cues = [c for c in palette if c not in used]
    selectors = [ca, cb]
    assert all(c in m.values() for c, m in zip(selectors, maps))
    return bg, maps, period, cues, selectors


def _layout(rng, templates, n):
    """A freely ordered legend strip on any side, outside all field bounds."""
    ub = _bounds(set(templates[0]) | set(templates[1]))
    rh, rw = ub[2]-ub[0]+1, ub[3]-ub[1]+1
    side = int(rng.integers(4))
    sizes = [(rh, rw), (rh, rw), (n, 2)]
    order = list(map(int, rng.permutation(3)))
    gaps = list(map(int, rng.integers(1, 3, size=2)))
    if side < 2:
        kh = max(rh, n)+2
        width = max(22, 2*rw+2+sum(gaps)+2)
        width = int(rng.integers(width, 31))
        sh = int(rng.integers(16, 30-kh))
        height = sh+kh+1
        sy = kh+1 if side == 0 else 0
        ky = 0 if side == 0 else sh+1
        total = sum(sizes[i][1] for i in order)+sum(gaps)
        cursor = int(rng.integers(0, width-total+1))
        positions = {}
        for j, i in enumerate(order):
            hh, ww = sizes[i]
            positions[i] = ky+int(rng.integers(0, kh-hh+1)), cursor
            cursor += ww+(gaps[j] if j < 2 else 0)
        scene = (sy, 0, sy+sh-1, width-1)
    else:
        kw = max(rw, 2)+2
        height = max(22, 2*rh+n+sum(gaps)+2)
        height = int(rng.integers(height, 31))
        sw = int(rng.integers(16, 30-kw))
        width = sw+kw+1
        sx = kw+1 if side == 2 else 0
        kx = 0 if side == 2 else sw+1
        total = sum(sizes[i][0] for i in order)+sum(gaps)
        cursor = int(rng.integers(0, height-total+1))
        positions = {}
        for j, i in enumerate(order):
            hh, ww = sizes[i]
            positions[i] = cursor, kx+int(rng.integers(0, kw-ww+1))
            cursor += hh+(gaps[j] if j < 2 else 0)
        scene = (0, sx, height-1, sx+sw-1)
    anchors = [(positions[i][0]-ub[0], positions[i][1]-ub[1]) for i in (0, 1)]
    return (height, width), scene, anchors, positions[2]


def _rectangles(rng, region, n, scales, intro=False):
    y0, x0, y1, x1 = region
    h, w = y1-y0+1, x1-x0+1
    fields, used = [], set()
    topology = str(rng.choice(['free', 'partial', 'nested', 'bands', 'chain', 'disjoint']))
    for i, s in enumerate(scales):
        placed = False
        for _ in range(100):
            if intro:
                # Split along whichever axis gives two generous fields.
                axis = 1 if w >= h else 0
                lo, hi = (x0, x1) if axis else (y0, y1)
                mid = (lo+hi)//2
                a, b = (lo, mid-1) if i == 0 else (mid+1, hi)
                rr = (y0, a, y1, b) if axis else (a, x0, b, x1)
                ah, aw = rr[2]-rr[0]+1, rr[3]-rr[1]+1
                if min(ah, aw) < 4*s:
                    return None
                hh = int(rng.integers(4*s, ah+1))
                ww = int(rng.integers(4*s, aw+1))
                yy = int(rng.integers(rr[0], rr[2]-hh+2))
                xx = int(rng.integers(rr[1], rr[3]-ww+2))
            elif topology == 'bands':
                horizontal = bool((i+int(rng.integers(2))) % 2)
                hh = int(rng.integers(4*s, min(h, 6*s)+1)) if horizontal else int(rng.integers(max(4*s, h-4), h+1))
                ww = int(rng.integers(max(4*s, w-4), w+1)) if horizontal else int(rng.integers(4*s, min(w, 6*s)+1))
                yy, xx = int(rng.integers(y0, y1-hh+2)), int(rng.integers(x0, x1-ww+2))
            else:
                rr = region
                if topology == 'nested' and fields and rng.random() < .7:
                    b = fields[0]['bounds']
                    if min(b[2]-b[0]-1, b[3]-b[1]-1) >= 4*s:
                        rr = (b[0]+1, b[1]+1, b[2]-1, b[3]-1)
                hh = int(rng.integers(4*s, rr[2]-rr[0]+2))
                ww = int(rng.integers(4*s, rr[3]-rr[1]+2))
                yy, xx = int(rng.integers(rr[0], rr[2]-hh+2)), int(rng.integers(rr[1], rr[3]-ww+2))
            b = (yy, xx, yy+hh-1, xx+ww-1)
            diagonal = int(rng.integers(2))
            points = _marker_cells(b, s, diagonal)
            if points & used or any(b == f['bounds'] for f in fields):
                continue
            if not intro and fields:
                intersections = [_overlap(b, f['bounds']) for f in fields]
                if topology == 'disjoint' and any(intersections):
                    continue
                if topology in ('partial', 'chain') and not any(intersections):
                    continue
                if topology == 'chain' and i > 1 and any(intersections[:-1]):
                    continue
            fields.append(dict(bounds=b, scale=s, diagonal=diagonal, markers=points))
            used |= points
            placed = True
            break
        if not placed:
            return None
    return fields


def _witnesses(g, info, expected=None):
    out = _predict(g, info) if expected is None else expected
    result = set()
    alternatives = {'priority': dict(reverse=True), 'opaque': dict(transparent=True),
        'exchange': dict(exchange=False), 'phase': dict(bbox_phase=True), 'scaled': dict(unit_scale=True)}
    for name, options in alternatives.items():
        if not np.array_equal(out, _predict(g, info, **options)):
            result.add(name)
    if any(out[p] != f['cue'] for f in info['fields'] for p in f['markers']):
        result.add('marker_occlusion')
    if any(out[p] == f['cue'] for f in info['fields'] for p in f['markers']):
        result.add('marker_retention')
    a, b = info['templates']
    if any(a[p] != b[p] for p in set(a) & set(b)):
        result.add('recolor')
    if set(a) != set(b):
        result.add('relocate')
    maps = _pattern_maps(info)
    common = {p for p, c in a.items() if b.get(p) == c}
    ph, pw = info['period']
    copies = {}
    def value(f, y, x, state):
        if (y, x) in f['markers']:
            return f['cue']
        yy, xx = (y-f['bounds'][0])//f['scale'], (x-f['bounds'][1])//f['scale']
        return maps[state].get((yy % ph, xx % pw), info['background'])
    for y in range(g.shape[0]):
        for x in range(g.shape[1]):
            stack = [(rank, f) for rank, f in enumerate(info['fields'])
                     if _inside(y, x, f['bounds'])]
            if not stack:
                continue
            rank, f = stack[0]
            state = f['state']
            if (y, x) not in f['markers']:
                if out[y, x] != value(f, y, x, state):
                    result.add('direction'+str(state))
                yy, xx = (y-f['bounds'][0])//f['scale'], (x-f['bounds'][1])//f['scale']
                residue = yy % ph, xx % pw
                if residue in common:
                    copies.setdefault((rank, residue), set()).add((yy//ph, xx//pw))
                if out[y, x] == info['background'] and any(
                        value(q, y, x, 1-q['state']) != info['background']
                        and (y, x) not in q['markers'] for _, q in stack[1:]):
                    result.add('gap_erasure')
    if any(len(v) > 1 for v in copies.values()):
        result.add('repeat')
    return result


def _make_pair(rng, split, index):
    intro = split == 'train' and index == 0
    teaching = split == 'train' and index == 1
    change = 'recolor' if intro else 'combined' if teaching or split == 'test' else str(rng.choice(['recolor', 'relocate', 'combined']))
    for _ in range(240):
        bg, templates, period, cues, selectors = _design(rng, change)
        n = 2 if intro else int(rng.integers(2, min(3, len(cues))+1)) if teaching else int(rng.integers(1, min(4, len(cues))+1))
        shape, region, anchors, table_pos = _layout(rng, templates, n)
        limit = min(region[2]-region[0]+1, region[3]-region[1]+1)//4
        scales = [1, 2] if intro else list(map(int, rng.choice(
            list(range(1, min(limit, 3 if split == 'test' else 2)+1)), size=n)))
        if teaching:
            scales[0], scales[1] = 1, 2
        if split == 'test':
            # A guaranteed scale generalization; other fields need not enlarge.
            scales[0] = 3
        fields = _rectangles(rng, region, n, scales, intro)
        if fields is None:
            continue
        states = [0, 1] + list(map(int, rng.integers(2, size=max(0, n-2)))) if n > 1 else [int(rng.integers(2))]
        rng.shuffle(states)
        for f, cue, state in zip(fields, cues[:n], states):
            f.update(cue=cue, state=state, selector=selectors[state])
        fields = [fields[int(i)] for i in rng.permutation(n)]
        g = np.full(shape, bg, dtype=int)
        for template, anchor in zip(templates, anchors):
            _draw(g, template, anchor)
        for f in fields:
            for p in f['markers']:
                g[p] = f['cue']
        ty, tx = table_pos
        for rank, f in enumerate(fields):
            g[ty+rank, tx:tx+2] = f['cue'], f['selector']
        try:
            info = _read_input(g)
            out = _compose(g, bg, fields, templates, period)
            assert np.array_equal(out, _predict(g, info))
            witnesses = _witnesses(g, info, out)
            needed = {'exchange', 'marker_retention'}
            if intro:
                needed |= {'scaled', 'repeat', 'direction0', 'direction1', 'recolor'}
            if teaching:
                needed |= {'priority', 'opaque', 'gap_erasure', 'marker_occlusion', 'phase', 'relocate'}
            if split == 'test':
                needed.add('scaled')
            if not needed <= witnesses:
                continue
            return dict(input=g, output=out)
        except AssertionError:
            continue
    raise RuntimeError('bounded readable field construction exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ns = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ns)):
        for index in range(count):
            for _ in range(12):
                pair = _make_pair(rng, split, index)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('unique input construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            g, out = _grid(pair['input']), _grid(pair['output'])
            assert g.shape == out.shape
            key = _key(g)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            info = _read_input(g)
            expected = _predict(g, info)
            assert np.array_equal(expected, out), 'incorrect exchange wallpaper'
            assert not np.array_equal(g, out), 'vacuous example'
            witnesses = _witnesses(g, info, out)
            if split == 'train':
                training |= witnesses
            else:
                assert {'exchange', 'scaled', 'priority', 'opaque', 'phase', 'repeat',
                        'recolor', 'relocate', 'direction0', 'direction1'} <= training
            # Explicitly check all data outside the construction, including keys.
            mask = np.zeros(g.shape, dtype=bool)
            for f in info['fields']:
                y0, x0, y1, x1 = f['bounds']
                mask[y0:y1+1, x0:x1+1] = True
            assert np.array_equal(g[~mask], out[~mask]), 'legend or outside scene changed'
    assert {'exchange', 'scaled', 'priority', 'opaque', 'phase', 'repeat', 'recolor',
            'relocate', 'direction0', 'direction1', 'marker_retention', 'marker_occlusion', 'gap_erasure'} <= training


def measure_pair(pair):
    g = _grid(pair['input'])
    info = _read_input(g)
    fields = info['fields']
    a, b = info['templates']
    same = {p for p, c in a.items() if b.get(p) == c}
    return {'shape': tuple(g.shape), 'background': info['background'],
        'palette': tuple(map(int, np.unique(g))), 'fields': len(fields),
        'scales': tuple(sorted(f['scale'] for f in fields)),
        'diagonals': tuple(sorted({f['diagonal'] for f in fields})),
        'period': info['period'], 'geometry': tuple(sorted(same)),
        'sizes': tuple((f['bounds'][2]-f['bounds'][0]+1, f['bounds'][3]-f['bounds'][1]+1) for f in fields),
        'overlaps': sum(_overlap(f['bounds'], q['bounds']) for i, f in enumerate(fields) for q in fields[i+1:]),
        'nested': any(_overlap(f['bounds'], q['bounds']) and
            ((f['bounds'][0] <= q['bounds'][0] and f['bounds'][1] <= q['bounds'][1] and f['bounds'][2] >= q['bounds'][2] and f['bounds'][3] >= q['bounds'][3]) or
             (q['bounds'][0] <= f['bounds'][0] and q['bounds'][1] <= f['bounds'][1] and q['bounds'][2] >= f['bounds'][2] and q['bounds'][3] >= f['bounds'][3]))
             for i, f in enumerate(fields) for q in fields[i+1:]),
        'clipped_blocks': any((f['bounds'][2]-f['bounds'][0]+1) % f['scale'] or (f['bounds'][3]-f['bounds'][1]+1) % f['scale'] for f in fields),
        'edge': any(f['bounds'][0] == 0 or f['bounds'][1] == 0 or f['bounds'][2] == g.shape[0]-1 or f['bounds'][3] == g.shape[1]-1 for f in fields),
        'unequal_support': len(a) != len(b), 'table': info['table'][:2],
        'witnesses': tuple(sorted(_witnesses(g, info, _grid(pair['output']))))}


def self_test():
    """Bounded 42-seed semantic and structural sweep plus mutation checks."""
    records, signatures = [], set()
    within_geometry = within_palette = 0
    for index, seed in enumerate(HOST_SEEDS):
        nt = 2+index % 9
        ns = 1+index % min(4, nt-1)
        puzzle = generate_puzzle(seed, nt, ns)
        validate_generated_puzzle(puzzle)
        again = generate_puzzle(seed, nt, ns)
        signature = tuple(_key(p[r]) for s in ('train', 'test') for p in puzzle[s] for r in ('input', 'output'))
        assert signature == tuple(_key(p[r]) for s in ('train', 'test') for p in again[s] for r in ('input', 'output'))
        signatures.add(signature)
        local = []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                original = _grid(pair['input'])
                assert np.array_equal(solve(pair['input']), pair['output'])
                assert np.array_equal(original, pair['input']), 'solver mutated input'
                m = measure_pair(pair)
                m['split'] = split
                records.append(m)
                local.append(m)
        within_geometry += len({m['geometry'] for m in local}) > 1
        within_palette += len({m['palette'] for m in local}) > 1
    assert len(signatures) == 42 and within_geometry >= 35 and within_palette >= 35
    assert len({m['shape'] for m in records}) >= 60
    assert len({m['geometry'] for m in records}) >= 40
    assert {m['background'] for m in records} == set(range(10))
    assert {m['fields'] for m in records} >= {1, 2, 3, 4}
    assert {s for m in records for s in m['scales']} == {1, 2, 3}
    assert len({m['scales'] for m in records}) >= 12
    assert len({m['period'] for m in records}) >= 10
    for feature in ('nested', 'clipped_blocks', 'edge', 'unequal_support'):
        assert sum(bool(m[feature]) for m in records) >= 10, feature
    for split in ('train', 'test'):
        assert any(m['diagonals'] == (0, 1) for m in records if m['split'] == split)
        assert any(m['overlaps'] == 0 for m in records if m['split'] == split)
        assert any(m['overlaps'] for m in records if m['split'] == split)
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Exercise reader-supported members outside the sampler: five fields,
    # a scale-four field, scale-two legends, and irregularly placed keys.
    common = {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 2): 1,
              (2, 0): 1, (2, 1): 1, (2, 2): 4}
    templates = [dict(common), dict(common)]
    templates[0].update({(-1, 1): 2, (1, 1): 2})
    templates[1].update({(3, 2): 3, (1, 1): 3})
    manual = np.zeros((30, 30), dtype=int)
    _draw(manual, templates[0], (2, 0), 2)
    _draw(manual, templates[1], (0, 10), 2)
    fields = []
    boxes = [(10, 0, 25, 15), (14, 4, 21, 11), (11, 18, 17, 24),
             (17, 20, 25, 29), (22, 17, 29, 24)]
    for i, box in enumerate(boxes):
        s, diagonal = (4 if i == 0 else 1), i % 2
        markers = _marker_cells(box, s, diagonal)
        fields.append(dict(bounds=box, scale=s, diagonal=diagonal,
                           markers=markers, cue=5+i, state=i % 2, selector=2+i % 2))
        for p in markers:
            assert manual[p] == 0, 'manual cue collision'
            manual[p] = 5+i
        manual[i, 22:24] = 5+i, 2+i % 2
    parsed = _read_input(manual)
    assert parsed['reference_scale'] == 2 and len(parsed['fields']) == 5
    assert max(f['scale'] for f in parsed['fields']) == 4
    assert np.array_equal(solve(manual.tolist()), _compose(manual, 0, fields, templates, (6, 4)))
    sample = generate_puzzle(0, 2, 1)
    # A changed output pixel and an unchanged legend pixel must both be checked.
    for point in (tuple(np.argwhere(sample['test'][0]['input'] != sample['test'][0]['output'])[0]),
                  _read_input(sample['test'][0]['input'])['table'][:2]):
        bad = {s: [{r: p[r].copy() for r in ('input', 'output')} for p in sample[s]] for s in sample}
        g = bad['test'][0]['output']
        g[point] = (int(g[point])+1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('corrupted output accepted')
    malformed = sample['test'][0]['input'].copy()
    p = next(iter(_read_input(malformed)['fields'][0]['markers']))
    malformed[p] = _read_input(malformed)['background']
    try:
        solve(malformed)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged L markers accepted')
    return {'seeds': 42, 'pairs': len(records), 'shapes': len({m['shape'] for m in records}),
            'scaffolds': len({m['geometry'] for m in records}),
            'scale_schedules': len({m['scales'] for m in records}),
            'periods': len({m['period'] for m in records}),
            'within_puzzle_geometry': within_geometry, 'within_puzzle_palette': within_palette}


if __name__ == '__main__':
    print(self_test())
