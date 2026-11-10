"""Elastic loom: a color census stretches strips beneath a native-period stencil."""

from collections import Counter
import numpy as np


def _check(ok, message):
    if not ok:
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(num_train, num_test):
    a, b = _integer(num_train, 'num_train'), _integer(num_test, 'num_test')
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError('require 2..10 train, 1..4 test, and train > test')
    return a, b


def _array(value):
    g = np.asarray(value)
    _check(g.ndim == 2 and all(1 <= x <= 30 for x in g.shape), 'grid dimensions')
    _check(np.issubdtype(g.dtype, np.integer), 'integer grid required')
    _check(bool(np.all((g >= 0) & (g <= 9))), 'color bounds')
    return g


def _components(g, background, diagonal=False):
    remaining = {(int(r), int(c)) for r, c in zip(*np.nonzero(g != background))}
    result = []
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    if diagonal:
        directions += ((1, 1), (1, -1), (-1, 1), (-1, -1))
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        color = int(g[start])
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for dr, dc in directions:
                q = r + dr, c + dc
                if q in remaining and int(g[q]) == color:
                    remaining.remove(q)
                    todo.append(q)
        rs, cs = zip(*cells)
        result.append(dict(color=color, cells=tuple(sorted(cells)),
                           box=(min(rs), min(cs), max(rs)+1, max(cs)+1)))
    return result


def _frame_kind(part):
    r, c, rr, cc = part['box']
    if rr-r < 3 or cc-c < 3:
        return None
    perimeter = {(i, j) for i in range(r, rr) for j in range(c, cc)
                 if i in (r, rr-1) or j in (c, cc-1)}
    cells = set(part['cells'])
    if cells == perimeter:
        return 'stencil'
    if cells == perimeter | {(r+1, c+1)}:
        return 'carrier'
    return None


def _decode(input_grid):
    """Recover background, frames, rails, and external census from this input."""
    g = _array(input_grid)
    bg = int(g[0, 0])
    _check(np.all(g[0] == bg) and np.all(g[-1] == bg)
           and np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg), 'background margin')
    parts = _components(g, bg)
    candidates = []
    def has_moat(part):
        r, c, rr, cc = part['box']
        if r < 1 or c < 1 or rr >= g.shape[0] or cc >= g.shape[1]:
            return False
        return (np.all(g[r-1, c-1:cc+1] == bg) and np.all(g[rr, c-1:cc+1] == bg)
                and np.all(g[r-1:rr+1, c-1] == bg) and np.all(g[r-1:rr+1, cc] == bg))
    for color in sorted({p['color'] for p in parts}):
        own = [p for p in parts if p['color'] == color]
        if (len(own) == 2 and all(has_moat(p) for p in own)
                and sorted(str(_frame_kind(p)) for p in own) == ['carrier', 'stencil']):
            candidates.append((color, own))
    _check(len(candidates) == 1, 'need exactly one pinned carrier and one stencil frame')
    marker, frames = candidates[0]
    by_kind = {_frame_kind(p): p for p in frames}
    boxes = [p['box'] for p in frames]
    for r, c, rr, cc in boxes:
        _check(r >= 1 and c >= 1 and rr < g.shape[0] and cc < g.shape[1], 'frame margin')
    a, b = boxes
    _check(a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1],
           'frame rectangles need a background gap')
    r, c, rr, cc = by_kind['carrier']['box']
    interior = g[r+1:rr-1, c+1:cc-1]
    _check(interior.shape[0] >= 2 and interior.shape[1] >= 2, 'missing rails or carrier')
    rows = interior[1:, 0].copy()
    cols = interior[0, 1:].copy()
    carrier = interior[1:, 1:].copy()
    r, c, rr, cc = by_kind['stencil']['box']
    stencil = g[r+1:rr-1, c+1:cc-1].copy()
    _check(all(1 <= n <= 6 for n in carrier.shape + stencil.shape), 'source dimensions 1..6')
    _check(np.all(carrier != bg) and np.all(carrier != marker), 'carrier must be payload')
    _check(np.all(rows != bg) and np.all(rows != marker)
           and np.all(cols != bg) and np.all(cols != marker), 'invalid rail label')
    _check(not np.any(stencil == marker), 'stencil cannot contain frame marker')
    outside = g.copy()
    for r, c, rr, cc in boxes:
        outside[r:rr, c:cc] = bg
    objects = _components(outside, bg)
    _check(all(p['color'] != marker and 1 <= len(p['cells']) <= 12 for p in objects),
           'external objects must be small, monochromatic, and not markers')
    _check(all(_frame_kind(p) is None for p in objects), 'external object imitates a source frame')
    for a, b in ((outside[:-1, :], outside[1:, :]), (outside[:, :-1], outside[:, 1:])):
        _check(not np.any((a != bg) & (b != bg) & (a != b)), 'external objects touch orthogonally')
    # Objects cannot touch a frame rectangle, even at a diagonal corner.
    for p in objects:
        for i, j in p['cells']:
            _check(all(not (r-1 <= i <= rr and c-1 <= j <= cc)
                       for r, c, rr, cc in boxes), 'object touches frame')
    categories = set(map(int, rows)) | set(map(int, cols))
    census = Counter(p['color'] for p in objects)
    counts = {color: census[color] for color in categories}
    _check(all(0 <= n <= 5 for n in counts.values()), 'category count 0..5')
    row_runs = [counts[int(color)] for color in rows]
    col_runs = [counts[int(color)] for color in cols]
    _check(1 <= sum(row_runs) <= 30 and 1 <= sum(col_runs) <= 30, 'empty/oversize output')
    return dict(background=bg, marker=marker, frames=frames, rows=rows, cols=cols,
                carrier=carrier, stencil=stencil, objects=objects, outside=outside,
                categories=categories, counts=counts, row_runs=row_runs, col_runs=col_runs)


def _render(carrier, stencil, row_runs, col_runs, background):
    base = np.repeat(np.repeat(carrier, row_runs, axis=0), col_runs, axis=1)
    rr, cc = np.indices(base.shape)
    tiled = stencil[rr % stencil.shape[0], cc % stencil.shape[1]]
    return np.where(tiled == background, base, tiled).astype(np.int64)


def solve(input_grid: np.ndarray) -> np.ndarray:
    d = _decode(input_grid)
    return _render(d['carrier'], d['stencil'], d['row_runs'], d['col_runs'], d['background'])


def _shape(rng, area=None):
    # Shapes are drawn without seeing the category's count.
    if area is None:
        area = int(rng.choice([1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]))
    style = int(rng.integers(4))
    if style == 0 and area <= 8:
        cells = {(0, j) for j in range(area)}
    elif style == 1:
        w = int(rng.integers(1, min(4, area)+1))
        cells = {(j//w, j % w) for j in range(area)}
    elif style == 2:
        arm = int(rng.integers(1, area+1))
        cells = {(0, j) for j in range(arm)} | {(j, 0) for j in range(area-arm+1)}
    else:
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = sorted({(r+dr, c+dc) for r, c in cells
                               for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    for _ in range(int(rng.integers(4))):
        cells = {(c, -r) for r, c in cells}
    if rng.random() < .5:
        cells = {(r, -c) for r, c in cells}
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    shape = tuple(sorted((r-r0, c-c0) for r, c in cells))
    # Reserve exact plain/pinned rectangular outlines for the source frames.
    part = dict(cells=shape, box=(0, 0, 1+max(r for r, c in shape),
                                 1+max(c for r, c in shape)))
    if _frame_kind(part) is not None:
        shape = tuple((0, c) for c in range(area))
    return shape


def _carrier(rng, h, w, palette, teaching):
    k = int(rng.integers(2 if teaching else 1, min(5, len(palette), h*w)+1))
    inks = rng.choice(palette, k, replace=False)
    kind = int(rng.integers(6))
    rr, cc = np.indices((h, w))
    if kind == 0:
        labels = np.zeros((h, w), dtype=int)
    elif kind == 1:
        labels = np.broadcast_to(rng.integers(k, size=(h, 1)), (h, w)).copy()
    elif kind == 2:
        labels = np.broadcast_to(rng.integers(k, size=(1, w)), (h, w)).copy()
    elif kind == 3:
        labels = (rr//int(rng.integers(1, h+1)) + cc//int(rng.integers(1, w+1))) % k
    elif kind == 4:
        labels = np.zeros((h, w), dtype=int)
        for ink in range(1, k):
            a, b = int(rng.integers(h)), int(rng.integers(w))
            labels[a:int(rng.integers(a+1, h+1)), b:int(rng.integers(b+1, w+1))] = ink
    else:
        labels = rng.integers(k, size=(h, w))
    if teaching:
        # Random distinct pixels make varying axes available; evidence below
        # decides whether their differences actually survive the overlay.
        labels.ravel()[rng.choice(h*w, k, replace=False)] = np.arange(k)
    return inks[labels].astype(np.int64)


def _stencil(rng, p, q, palette, bg, teaching):
    result = np.full((p, q), bg, dtype=np.int64)
    if not teaching and rng.random() < .16:
        return result
    rr, cc = np.indices((p, q))
    kind = int(rng.integers(6))
    density = float(rng.uniform(.15, .75))
    if kind == 0:
        mask = rng.random((p, q)) < density
    elif kind == 1:
        mask = (rr + cc + int(rng.integers(2))) % 2 == 0
    elif kind == 2:
        mask = np.broadcast_to(rng.random((p, 1)) < density, (p, q)).copy()
    elif kind == 3:
        mask = np.broadcast_to(rng.random((1, q)) < density, (p, q)).copy()
    elif kind == 4:
        mask = (rr == 0) | (cc == 0) | (rr == p-1) | (cc == q-1)
        mask &= rng.random((p, q)) > .15
    else:
        r, c = int(rng.integers(p)), int(rng.integers(q))
        mask = abs(rr-r) + abs(cc-c) <= int(rng.integers(1, max(p, q)+1))
    if teaching or p*q > 1:
        if not mask.any():
            mask.ravel()[int(rng.integers(p*q))] = True
        if mask.all():
            mask.ravel()[int(rng.integers(p*q))] = False
    n = int(mask.sum())
    if n:
        k = 1 if rng.random() < .55 else int(rng.integers(1, min(4, len(palette), n)+1))
        inks = rng.choice(palette, k, replace=False)
        result[mask] = rng.choice(inks, n)
    return result


def _draw_frame(g, box, marker, inside):
    r, c, rr, cc = box
    g[r:rr, c:cc] = marker
    g[r+1:rr-1, c+1:cc-1] = inside


def _place_objects(rng, g, boxes, specs, diagonal_color=None, greedy=False):
    """Free pixel placements, excluding four-neighbor contact; diagonal is legal."""
    h, w = g.shape
    bg = int(g[0, 0])
    blocked = np.zeros(g.shape, dtype=bool)
    blocked[[0, -1], :] = True
    blocked[:, [0, -1]] = True
    for r, c, rr, cc in boxes:
        blocked[r-1:rr+1, c-1:cc+1] = True
    # The diagonal witness is placed first, so it is never lost to packing.
    special = [i for i, (color, _) in enumerate(specs) if color == diagonal_color]
    order = special + sorted([i for i in range(len(specs)) if i not in special],
                             key=lambda i: (-len(specs[i][1]), float(rng.random())))
    anchor_cells = None
    for index in order:
        color, shape = specs[index]
        sh, sw = 1+max(r for r, c in shape), 1+max(c for r, c in shape)
        if sh > h or sw > w:
            return None
        allowed = np.ones((h-sh+1, w-sw+1), dtype=bool)
        for dr, dc in shape:
            allowed &= ~blocked[dr:dr+h-sh+1, dc:dc+w-sw+1]
        if color == diagonal_color and anchor_cells is not None:
            contact = np.zeros(g.shape, dtype=bool)
            for r, c in anchor_cells:
                for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                    if 0 <= r+dr < h and 0 <= c+dc < w:
                        contact[r+dr, c+dc] = True
            touching = np.zeros_like(allowed)
            for dr, dc in shape:
                touching |= contact[dr:dr+h-sh+1, dc:dc+w-sw+1]
            allowed &= touching
        rr, cc = np.nonzero(allowed)
        if not len(rr):
            return None
        j = 0 if greedy else int(rng.integers(len(rr)))
        r0, c0 = int(rr[j]), int(cc[j])
        cells = [(r0+dr, c0+dc) for dr, dc in shape]
        for r, c in cells:
            g[r, c] = color
            for nr, nc in ((r, c), (r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True
        if color == diagonal_color:
            anchor_cells = cells
    return g


def _pack(rng, carrier_interior, stencil, marker, bg, specs, diagonal_color):
    sizes = [(carrier_interior.shape[0]+2, carrier_interior.shape[1]+2),
             (stencil.shape[0]+2, stencil.shape[1]+2)]
    area = sum(len(s) for _, s in specs) + sum(h*w for h, w in sizes)
    minimum = min(27, max(14, int(np.ceil(np.sqrt(area*2.1)))))
    for attempt in range(30):
        h = int(rng.integers(minimum, 31)) if attempt < 20 else 30
        w = int(rng.integers(minimum, 31)) if attempt < 20 else 30
        boxes = []
        for sh, sw in sizes:
            for _ in range(40):
                r, c = int(rng.integers(1, h-sh)), int(rng.integers(1, w-sw))
                box = (r, c, r+sh, c+sw)
                if all(box[2] < b[0] or b[2] < box[0] or box[3] < b[1] or b[3] < box[1]
                       for b in boxes):
                    boxes.append(box)
                    break
            else:
                break
        if len(boxes) != 2:
            continue
        g = np.full((h, w), bg, dtype=np.int64)
        _draw_frame(g, boxes[0], marker, carrier_interior)
        _draw_frame(g, boxes[1], marker, stencil)
        result = _place_objects(rng, g, boxes, specs, diagonal_color)
        if result is not None:
            return result
    # Roomy fallback retains all counts, teaching areas and diagonal contact.
    # Only optional object masks become bars. Frame coordinates are not a rule.
    boxes = [(1, 1, 1+sizes[0][0], 1+sizes[0][1]),
             (1, 12, 1+sizes[1][0], 12+sizes[1][1])]
    simple_specs = [(color, tuple((0, c) for c in range(len(shape))))
                    for color, shape in specs]
    for _ in range(20):
        g = np.full((30, 30), bg, dtype=np.int64)
        _draw_frame(g, boxes[0], marker, carrier_interior)
        _draw_frame(g, boxes[1], marker, stencil)
        result = _place_objects(rng, g, boxes, simple_specs, diagonal_color)
        if result is not None:
            return result
    raise AssertionError('bounded packing failed')


def _make_pair(rng, bg, marker, cats, counts, rows, cols, periods,
               forced=None, diagonal_color=None, distractors=(), teaching=False):
    palette = [c for c in range(10) if c not in (bg, marker)]
    carrier = _carrier(rng, len(rows), len(cols), palette, teaching)
    stencil = _stencil(rng, *periods, palette, bg, teaching)
    inside = np.empty((len(rows)+1, len(cols)+1), dtype=np.int64)
    inside[0, 0] = marker
    inside[1:, 0], inside[0, 1:], inside[1:, 1:] = rows, cols, carrier
    forced = {} if forced is None else forced
    specs = [(color, _shape(rng, forced[color][i] if color in forced else None))
             for color in cats for i in range(counts[color])]
    specs.extend((color, _shape(rng)) for color in distractors)
    inp = _pack(rng, inside, stencil, marker, bg, specs, diagonal_color)
    # Construction uses sampled counts, while solve/validation resegment input.
    out = _render(carrier, stencil, [counts[int(c)] for c in rows],
                  [counts[int(c)] for c in cols], bg)
    return dict(input=inp, output=out)


_REQUIRED = frozenset(('area_counts', 'bbox_counts', 'unit_counts', 'pooled_counts', 'swap_axes',
    'global_resize', 'carrier_tile', 'stencil_resize', 'overlay_before_stretch',
    'restart_stencil', 'opaque_background', 'ignore_stencil', 'phase_rows',
    'phase_columns', 'visible_carrier', 'visible_overwrite', 'diagonal_components',
    'zero_strip', 'shared_label', 'distractor', 'sort_strips', 'keep_zero_gaps',
    'split_shared_count'))


def _evidence(pair, d=None):
    d = _decode(pair['input']) if d is None else d
    carrier, stencil, bg = d['carrier'], d['stencil'], d['background']
    rows, cols = d['row_runs'], d['col_runs']
    out, evidence = np.asarray(pair['output']), set()
    h, w = out.shape
    rr, cc = np.indices((h, w))
    base = np.repeat(np.repeat(carrier, rows, axis=0), cols, axis=1)
    tile = stencil[rr % stencil.shape[0], cc % stencil.shape[1]]
    def compare(name, alt):
        if not np.array_equal(out, alt):
            evidence.add(name)
    def overlay(base_image, thread=tile):
        return np.where(thread == bg, base_image, thread)
    areas = {c: sum(len(o['cells']) for o in d['objects'] if o['color'] == c)
             for c in d['categories']}
    bounds = {}
    for color in d['categories']:
        points = [point for o in d['objects'] if o['color'] == color for point in o['cells']]
        bounds[color] = ((max(r for r, c in points)-min(r for r, c in points)+1)
                         * (max(c for r, c in points)-min(c for r, c in points)+1)) if points else 0
    for name, values in (('area_counts', areas), ('bbox_counts', bounds),
                         ('unit_counts', {c: 1 for c in areas}),
                         ('pooled_counts', {c: len(d['objects']) for c in areas})):
        shape = (sum(values[int(c)] for c in d['rows']), sum(values[int(c)] for c in d['cols']))
        if shape != out.shape:
            evidence.add(name)
    if sum(rows) != sum(cols):
        evidence.add('swap_axes')
    if 0 in rows or 0 in cols:
        evidence.add('keep_zero_gaps')
    divided = (sum(d['counts'][int(c)] // list(d['rows']).count(c) for c in d['rows']),
               sum(d['counts'][int(c)] // list(d['cols']).count(c) for c in d['cols']))
    if divided != out.shape:
        evidence.add('split_shared_count')
    ri, ci = np.argsort(rows, kind='stable'), np.argsort(cols, kind='stable')
    compare('sort_strips', _render(carrier[ri[:, None], ci[None, :]], stencil,
                                  [rows[int(i)] for i in ri], [cols[int(i)] for i in ci], bg))
    compare('global_resize', overlay(carrier[rr*carrier.shape[0]//h, cc*carrier.shape[1]//w]))
    compare('carrier_tile', overlay(carrier[rr % carrier.shape[0], cc % carrier.shape[1]]))
    enlarged = stencil[rr*stencil.shape[0]//h, cc*stencil.shape[1]//w]
    compare('stencil_resize', overlay(base, enlarged))
    sr, sc = np.indices(carrier.shape)
    pre_thread = stencil[sr % stencil.shape[0], sc % stencil.shape[1]]
    pre = np.where(pre_thread == bg, carrier, pre_thread)
    compare('overlay_before_stretch', np.repeat(np.repeat(pre, rows, axis=0), cols, axis=1))
    local_r = np.concatenate([np.arange(n) for n in rows])
    local_c = np.concatenate([np.arange(n) for n in cols])
    restarted = stencil[local_r[:, None] % stencil.shape[0], local_c[None, :] % stencil.shape[1]]
    compare('restart_stencil', overlay(base, restarted))
    compare('opaque_background', tile)
    compare('ignore_stencil', base)
    compare('phase_rows', overlay(base, stencil[(rr+1) % stencil.shape[0], cc % stencil.shape[1]]))
    compare('phase_columns', overlay(base, stencil[rr % stencil.shape[0], (cc+1) % stencil.shape[1]]))
    if np.any(tile == bg):
        evidence.add('visible_carrier')
    if np.any((tile != bg) & (tile != base)):
        evidence.add('visible_overwrite')
    eight = Counter(p['color'] for p in _components(d['outside'], bg, diagonal=True))
    if any(eight[c] != d['counts'][c] for c in d['categories']):
        evidence.add('diagonal_components')
    if 0 in rows or 0 in cols:
        evidence.add('zero_strip')
    if any(c in set(map(int, d['rows'])) & set(map(int, d['cols'])) and n > 0
           for c, n in d['counts'].items()) and any(
            list(d['rows']).count(c) > 1 or list(d['cols']).count(c) > 1
            for c, n in d['counts'].items() if n > 0):
        evidence.add('shared_label')
    if any(p['color'] not in d['categories'] for p in d['objects']):
        evidence.add('distractor')
    return evidence


def _key(grid):
    g = np.asarray(grid, dtype=np.uint8)
    return g.shape, g.tobytes()


def _normalized(grid):
    g, palette = np.asarray(grid), {}
    return g.shape, bytes(palette.setdefault(int(v), len(palette)) for v in g.flat)


def _teaching(rng):
    anchor, b, c, z, zz = map(int, rng.choice(10, 5, replace=False))
    collision = int(rng.integers(4, 9))
    split = int(rng.integers(1, collision))
    other = int(rng.choice([n for n in range(3, 11) if n != collision]))
    other_split = int(rng.integers(1, other))
    # Two randomized count/area witnesses, with no fixed pictures or positions.
    for _ in range(64):
        pairs = []
        for index in range(2):
            cats = [anchor, b, z] if index == 0 else [anchor, c, zz]
            counts = dict(zip(cats, (1, 2, 0) if index == 0 else (2, 3, 0)))
            bg, marker = map(int, rng.choice([x for x in range(10) if x not in cats
                                             and not (index == 1 and x == b)], 2, replace=False))
            rows = [cats[0], cats[1], cats[2], int(rng.choice(cats[:2]))]
            cols = cats[:]
            if rng.random() < .5:
                rows, cols = cols, rows
            rng.shuffle(rows)
            rng.shuffle(cols)
            forced = ({anchor: [collision], b: [other_split, other-other_split]}
                      if index == 0 else {anchor: [split, collision-split]})
            dpool = [x for x in range(10) if x not in cats and x not in (bg, marker)]
            distractors = ([b] if index == 1 else []) + [int(rng.choice(dpool))
                                                       for _ in range(int(rng.integers(3)))]
            periods = tuple(map(int, rng.integers(2, 5, size=2)))
            pairs.append(_make_pair(rng, bg, marker, cats, counts, rows, cols, periods,
                                    forced, anchor if index == 1 else None, distractors, True))
        ds = [_decode(p['input']) for p in pairs]
        if (_REQUIRED <= set().union(*(_evidence(p, d) for p, d in zip(pairs, ds)))
                and ds[0]['background'] != ds[1]['background']
                and _normalized(ds[0]['carrier']) != _normalized(ds[1]['carrier'])):
            return pairs
    raise AssertionError('bounded teaching selection failed')


def _ordinary(rng, shift=None):
    bg, marker = map(int, rng.choice(10, 2, replace=False))
    palette = [x for x in range(10) if x not in (bg, marker)]
    k = int(rng.integers(1, 6))
    cats = list(map(int, rng.choice(palette, k, replace=False)))
    count_values = rng.integers(0, 4, size=k)
    if not np.any(count_values):
        count_values[int(rng.integers(k))] = int(rng.integers(1, 4))
    if shift == 'count':
        count_values[0] = int(rng.integers(4, 6))
    counts = dict(zip(cats, map(int, count_values)))
    h, w = map(int, rng.integers(1, 5, size=2))
    if shift == 'carrier':
        if rng.random() < .5:
            h = int(rng.integers(5, 7))
        else:
            w = int(rng.integers(5, 7))
    # Every category must actually appear on a rail; this is a local legend.
    while h+w < k:
        h = min(4, h+1)
        w = min(4, w+1)
    labels = cats + list(map(int, rng.choice(cats, h+w-k)))
    rng.shuffle(labels)
    rows, cols = labels[:h], labels[h:]
    positive = [c for c in cats if counts[c] > 0]
    if not sum(counts[c] for c in rows):
        rows[int(rng.integers(h))] = int(rng.choice(positive))
    if not sum(counts[c] for c in cols):
        cols[int(rng.integers(w))] = int(rng.choice(positive))
    # A repair might remove a zero label, whose count was zero anyway.
    cats = sorted(set(rows+cols))
    if shift == 'count' and not any(counts[c] >= 4 for c in cats):
        rows[0] = next(c for c in counts if counts[c] >= 4)
        cats = sorted(set(rows+cols))
    p, q = map(int, rng.integers(1, 5, size=2))
    new_axis = None
    if shift == 'stencil':
        if rng.random() < .5:
            p = int(rng.integers(5, 7))
            new_axis = 0
        else:
            q = int(rng.integers(5, 7))
            new_axis = 1
        # Expose at least one wrap of the new native period. Its longer axis
        # is supported by familiar counts, not a simultaneous count extension.
        direction = rows if new_axis == 0 else cols
        period = p if new_axis == 0 else q
        high = max(cats, key=lambda c: counts[c])
        if counts[high] < 2:
            counts[high] = int(rng.integers(2, 4))
        while len(direction) < 4:
            direction.append(high)
        while sum(counts[c] for c in direction) <= period:
            low = min(range(len(direction)), key=lambda i: counts[direction[i]])
            direction[low] = high
        cats = sorted(set(rows+cols))
    dpool = [x for x in palette if x not in cats]
    distractors = [int(rng.choice(dpool)) for _ in range(int(rng.integers(4)))]
    for _ in range(32):
        pair = _make_pair(rng, bg, marker, cats, counts, rows, cols, (p, q),
                          distractors=distractors, teaching=new_axis is not None)
        if new_axis is None:
            return pair
        stencil = _decode(pair['input'])['stencil']
        period = stencil.shape[new_axis]
        if not any(np.array_equal(stencil, np.roll(stencil, n, axis=new_axis))
                   for n in range(1, period)):
            return pair
    raise AssertionError('bounded informative stencil shift failed')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed')
    if seed < 0:
        raise ValueError('seed must be nonnegative')
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    puzzle = {'train': _teaching(rng), 'test': []}
    seen = {_key(p['input']) for p in puzzle['train']}
    shift = str(rng.choice(['carrier', 'count', 'stencil']))
    for split, count in (('train', nt), ('test', nq)):
        while len(puzzle[split]) < count:
            for _ in range(20):
                pair = _ordinary(rng, shift if split == 'test' else None)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise AssertionError('could not make distinct input')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _check(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, details = set(), {'train': [], 'test': []}
    evidence = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            inp, out = _array(pair['input']), _array(pair['output'])
            _check(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            d = _decode(inp)
            _check(np.array_equal(solve(inp), out), 'incorrect census/stretch/overlay output')
            # Direct pixel provenance and global phase, including every boundary.
            source_r = [i for i, n in enumerate(d['row_runs']) for _ in range(n)]
            source_c = [j for j, n in enumerate(d['col_runs']) for _ in range(n)]
            _check(out.shape == (len(source_r), len(source_c)), 'strip dimensions')
            for r, sr in enumerate(source_r):
                for c, sc in enumerate(source_c):
                    ink = int(d['stencil'][r % d['stencil'].shape[0], c % d['stencil'].shape[1]])
                    expected = int(d['carrier'][sr, sc]) if ink == d['background'] else ink
                    _check(int(out[r, c]) == expected, 'pixel provenance')
            details[split].append(d)
            if split == 'train':
                evidence |= _evidence(pair, d)
            elif (max(d['stencil'].shape) > 4 and max(d['carrier'].shape) <= 4
                  and max(d['counts'].values()) <= 3):
                _check(any(period > 4 and out.shape[axis] > period and not any(
                    np.array_equal(d['stencil'], np.roll(d['stencil'], k, axis=axis))
                    for k in range(1, period))
                    for axis, period in enumerate(d['stencil'].shape)),
                    'larger-period test must expose its actual wrap')
    _check(_REQUIRED <= evidence, 'training misses: ' + str(sorted(_REQUIRED-evidence)))
    tr, te = details['train'], details['test']
    records = [(c, n, sum(len(o['cells']) for o in d['objects'] if o['color'] == c))
               for d in tr for c, n in d['counts'].items() if n > 0]
    _check(any(c == cc and a == aa and n != nn for c, n, a in records
               for cc, nn, aa in records), 'same-color area collision missing')
    _check(any(n == nn and a != aa for c, n, a in records for cc, nn, aa in records),
           'same-count area independence missing')
    _check(len({d['background'] for d in tr}) >= 2, 'local background evidence')
    categories = set().union(*(d['categories'] for d in tr))
    discarded = {o['color'] for d in tr for o in d['objects'] if o['color'] not in d['categories']}
    _check(bool(categories & discarded), 'local matching versus permanent color dictionary')
    _check(any(max(d['carrier'].shape) > 4 or max(d['stencil'].shape) > 4
               or max(d['counts'].values()) > 3 for d in te), 'train/test shift missing')


def self_test():
    """Bounded 42-seed semantic, variation, metamorphic, and API checks."""
    # Small hand-computed regression: repeats [2,1,2,0] by [1,2,0],
    # two diagonally touching blue objects, a red bar, and an unmatched ink.
    hand = np.zeros((16, 18), dtype=np.int64)
    inside = np.array([[9, 2, 1, 3], [1, 4, 5, 6], [2, 7, 8, 4],
                       [1, 6, 4, 5], [3, 8, 7, 6]])
    _draw_frame(hand, (1, 1, 8, 7), 9, inside)
    _draw_frame(hand, (2, 10, 6, 14), 9, np.array([[0, 6], [5, 0]]))
    hand[11, 3] = hand[12, 4] = 1
    hand[10, 11:14] = 2
    hand[13, 9] = 8
    expected = np.array([[4, 6, 5], [5, 5, 5], [7, 6, 8], [5, 4, 5], [6, 6, 4]])
    _check(np.array_equal(solve(hand.tolist()), expected), 'hand-computed local stretch/phase')
    hand[13, 9] = 0
    _check(np.array_equal(solve(hand), expected), 'unmatched object is irrelevant')
    # Exercise the stated upper boundary and unsampled fully opaque support.
    large = np.zeros((24, 26), dtype=np.int64)
    inside = np.ones((7, 7), dtype=np.int64)
    inside[0, 0] = 9
    rr, cc = np.indices((6, 6))
    inside[1:, 1:] = 2 + (rr+cc) % 2
    _draw_frame(large, (1, 1, 10, 10), 9, inside)
    _draw_frame(large, (1, 14, 9, 22), 9, np.zeros((6, 6), dtype=np.int64))
    for point in ((13, 2), (13, 6), (17, 3), (19, 12), (20, 20)):
        large[point] = 1
    rr, cc = np.indices((30, 30))
    _check(np.array_equal(solve(large), 2+(rr//5+cc//5) % 2), '30x30 shared-key stretch')
    large[2:8, 15:21] = 7
    _check(np.array_equal(solve(large), np.full((30, 30), 7)), 'fully opaque stencil')
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    features, signatures, masks, census_values, periods = [], set(), set(), set(), set()
    shifts, backgrounds, markers, category_counts = set(), set(), set(), set()
    for puzzle in puzzles:
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                d = _decode(pair['input'])
                features.append((split, d, pair))
                backgrounds.add(d['background'])
                markers.add(d['marker'])
                category_counts.add(len(d['categories']))
                census_values.update(d['counts'].values())
                periods.add(d['stencil'].shape)
                for o in d['objects']:
                    r0, c0, _, _ = o['box']
                    masks.add(tuple((r-r0, c-c0) for r, c in o['cells']))
                if split == 'test':
                    if max(d['carrier'].shape) > 4:
                        shifts.add('carrier')
                    if max(d['stencil'].shape) > 4:
                        shifts.add('stencil')
                    if max(d['counts'].values()) > 3:
                        shifts.add('count')
    _check(len(signatures) == 42, 'cross-seed repetition')
    _check(backgrounds == set(range(10)) and markers == set(range(10)), 'palette role variation')
    _check(census_values == set(range(6)), 'count/zero diversity')
    _check({1, 2, 3, 4, 5} <= category_counts, 'category diversity')
    _check(shifts == {'carrier', 'stencil', 'count'}, 'shift diversity')
    _check(len(masks) >= 60 and max(len(m) for m in masks) == 12, 'object geometry diversity')
    _check(len(periods) >= 15, 'native rectangular periods')
    for split in ('train', 'test'):
        fs = [(d, p) for s, d, p in features if s == split]
        _check(len({_key(p['output'])[0] for _, p in fs}) >= 15, 'output shape diversity')
        _check(len({_normalized(d['carrier']) for d, _ in fs}) >= 25, 'carrier structure diversity')
        _check(len({_normalized(d['stencil']) for d, _ in fs}) >= 20, 'stencil structure diversity')
        _check(any(np.all(d['stencil'] == d['background']) for d, _ in fs), 'empty stencil support')
        _check(any(np.unique(d['carrier']).size == 1 for d, _ in fs), 'solid carrier support')
    # Every legal explicit size and strict illegal type/count boundary.
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(700 + 10*a + b, a, b)
            _check(len(p['train']) == a and len(p['test']) == b, 'explicit counts')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('invalid counts accepted')
    replay = generate_puzzle(137, 7, 3)
    again = generate_puzzle(137, 7, 3)
    _check(all(_key(a[k]) == _key(b[k]) for s in replay
               for a, b in zip(replay[s], again[s]) for k in a), 'deterministic replay')
    # Color bijection includes background and marker: no literal ink semantics.
    rng = np.random.default_rng(913)
    permutation = rng.permutation(10)
    for pair in puzzles[0]['train'] + puzzles[0]['test']:
        _check(np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']]),
               'palette-equivariance')
    # Add a connected pixel to a counted object without merging objects.
    pair = puzzles[0]['train'][0]
    d = _decode(pair['input'])
    mutated = pair['input'].copy()
    grown = False
    for o in d['objects']:
        if o['color'] not in d['categories'] or len(o['cells']) >= 12:
            continue
        for r, c in o['cells']:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r+dr, c+dc
                if not (1 <= nr < mutated.shape[0]-1 and 1 <= nc < mutated.shape[1]-1):
                    continue
                if mutated[nr, nc] != d['background']:
                    continue
                trial = mutated.copy()
                trial[nr, nc] = o['color']
                try:
                    dd = _decode(trial)
                except AssertionError:
                    continue
                if dd['counts'] == d['counts']:
                    _check(np.array_equal(solve(trial), pair['output']), 'area must be irrelevant')
                    grown = True
                    break
            if grown:
                break
        if grown:
            break
    _check(grown, 'area mutation witness not found')
    damaged = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in puzzles[0][s]]
               for s in puzzles[0]}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        return
    raise AssertionError('validator accepted corrupt output')
