"""Parity shadows: registered stencils on an extrapolated growing support.

Only standard-library modules and NumPy are used. The public solver reads its
input alone. Construction paints full-canvas latent supports; solving decodes
visible objects and counts translated footprints directly in the marked crop.
"""
from collections import Counter
import numpy as np


HOST_SEEDS = (0, 471156767, 288419514, 466734022, 366514895, 952842995,
              769115929, 474615491, 124030158, 512381267, 221511523, 997820734,
              778865405, 208538419, 43654328, 514907215, 732011972, 228483701,
              864218831, 367798500, 403914463, 533277457, 371369078, 428301529,
              169618596, 560911419, 950318011, 579633623, 91424207, 549963083,
              887241482, 885665683, 162063022, 229950392, 82952347, 130152348,
              735401599, 94158346, 48316666, 666925034, 432364913, 785749456)
DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), 'invalid dimensions')
    _need(np.issubdtype(a.dtype, np.integer) and not np.issubdtype(a.dtype, np.bool_),
          'grids must contain integers')
    _need(bool(np.all((0 <= a) & (a <= 9))), 'invalid colors')
    return a


def _key(a):
    a = np.asarray(a)
    return a.shape, tuple(map(int, a.flat))


def _puzzle_key(p):
    return tuple((_key(x['input']), _key(x['output']))
                 for split in ('train', 'test') for x in p[split])


def _corner_cells(bounds):
    r0, r1, c0, c1 = bounds
    return {(r0, c0), (r0, c0 + 1), (r0 + 1, c0),
            (r1, c1), (r1, c1 - 1), (r1 - 1, c1)}


def _corners(a):
    options = []
    for color in map(int, np.unique(a)):
        pos = np.argwhere(a == color)
        if len(pos) != 6:
            continue
        r0, c0 = map(int, pos.min(axis=0))
        r1, c1 = map(int, pos.max(axis=0))
        if r1 <= r0 or c1 <= c0:
            continue
        tl = {(r0, c0), (r0, c0 + 1), (r0 + 1, c0)}
        br = {(r1, c1), (r1, c1 - 1), (r1 - 1, c1)}
        if any(abs(r - s) + abs(c - d) <= 1 for r, c in tl for s, d in br):
            continue
        bb = (r0, r1, c0, c1)
        if set(map(tuple, pos)) == _corner_cells(bb):
            options.append((color, bb))
    _need(len(options) == 1, 'missing or ambiguous L annotations')
    return options[0]


def _components(a, color):
    todo = set(map(tuple, np.argwhere(a == color)))
    result = []
    while todo:
        p = min(todo)
        todo.remove(p)
        group, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in todo:
                        todo.remove(q)
                        group.add(q)
                        stack.append(q)
        result.append(group)
    return result


def _instructions(a, bg, marker, neutral):
    masks = _components(a, neutral)
    _need(1 <= len(masks) <= 2, 'one or two separate stencils required')
    near = {}
    for k, mask in enumerate(masks):
        for r, c in mask:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    near.setdefault((r + dr, c + dc), set()).add(k)
    choices = [[] for _ in masks]
    banned = {bg, marker, neutral}
    for sr, sc in np.argwhere(~np.isin(a, list(banned))):
        sr, sc = int(sr), int(sc)
        source = int(a[sr, sc])
        for dr, dc in DIRS:
            qr, qc = sr + 2 * dr, sc + 2 * dc
            if not (0 <= qr < a.shape[0] and 0 <= qc < a.shape[1]):
                continue
            tr, tc = sr + dr, sc + dc
            target = int(a[tr, tc])
            if target in banned or target == source or int(a[qr, qc]) not in (bg, neutral):
                continue
            for k in sorted(near.get((qr, qc), ())):
                choices[k].append(dict(source=source, target=target, q=(qr, qc),
                                       direction=(dr, dc), mask=masks[k],
                                       labels=((sr, sc), (tr, tc)),
                                       offsets=tuple(sorted((int(r) - qr, int(c) - qc)
                                                            for r, c in masks[k]))))
    _need(all(len(x) == 1 for x in choices), 'stencil labels are not unique')
    records = [x[0] for x in choices]
    labels = [p for rec in records for p in rec['labels']]
    _need(len(set(labels)) == 2 * len(records), 'shared labels')
    _need(len({x['source'] for x in records}) == len(records), 'repeated local source')
    return records


def _linear(values):
    values = list(map(int, values))
    _need(len(values) >= 2, 'insufficient arithmetic evidence')
    d = values[1] - values[0]
    _need(all(x == values[0] + i * d for i, x in enumerate(values)),
          'non-arithmetic evidence')
    return values[0], d


def _fits(clean, bg, sources):
    models = []
    for axis in (0, 1):
        a = clean if axis == 0 else clean.T
        rows = np.flatnonzero(np.any(a != bg, axis=1))
        groups = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
        if not 4 <= len(groups) <= 5 or not len(rows):
            continue
        try:
            starts, widths, transverse, colors, thick = [], [], [], [], []
            for group in groups:
                cols = np.flatnonzero(np.any(a[group] != bg, axis=0))
                _need(len(cols) > 0, 'empty bar')
                patch = a[group[0]:group[-1] + 1, cols[0]:cols[-1] + 1]
                _need(len(np.unique(patch)) == 1 and int(patch[0, 0]) in sources,
                      'bar is not a solid source rectangle')
                starts.append(int(group[0]))
                transverse.append(int(cols[0]))
                widths.append(len(cols))
                thick.append(len(group))
                colors.append(int(patch[0, 0]))
            _need(len(set(thick)) == 1 and thick[0] in (1, 2), 'invalid bar thickness')
            s0, ds = _linear(np.diff(starts))
            _need(min(np.diff(starts)) > thick[0], 'touching bar slots')
            u0, du = _linear(transverse)
            w0, dw = _linear(widths)
            periods = [p for p in (1, 2) if len(colors) >= p + 1 and
                       len(set(colors[:p])) == p and set(colors[:p]) == sources and
                       all(c == colors[i % p] for i, c in enumerate(colors))]
            _need(bool(periods), 'unestablished color cycle')
            period = min(periods)
            models.append(dict(axis=axis, v0=starts[0], s0=s0, ds=ds, u0=u0,
                               du=du, w0=w0, dw=dw, t=thick[0], n=len(groups),
                               cycle=tuple(colors[:period])))
        except ValueError:
            continue
    return models


def _bar(model, i):
    return (model['v0'] + i * model['s0'] + i * (i - 1) // 2 * model['ds'],
            model['u0'] + i * model['du'], model['w0'] + i * model['dw'])


def _indices(model, extent):
    indices = list(range(model['n']))
    for direction, initial in ((-1, -1), (1, model['n'])):
        for j in range(35):
            i = initial + direction * j
            v, u, w = _bar(model, i)
            gap_i = i if direction < 0 else i - 1
            if w <= 0 or model['s0'] + gap_i * model['ds'] <= model['t']:
                break
            if (direction < 0 and v + model['t'] <= 0) or (direction > 0 and v >= extent):
                break
            indices.append(i)
    return sorted(indices)


def _support_cells(shape, model, frozen=False, one_per_bar=False):
    extent, width = shape if model['axis'] == 0 else shape[::-1]
    cells = []
    # The frozen baseline repeats the nearest observed bar independently all
    # the way to the canvas edge, rather than borrowing the true stop indices.
    indices = range(-35, model['n'] + 35) if frozen else _indices(model, extent)
    for i in indices:
        v, u, w = _bar(model, i)
        if frozen and i >= model['n']:
            k = model['n'] - 1
            v_last, u, w = _bar(model, k)
            v = v_last + (i - k) * (model['s0'] + (k - 1) * model['ds'])
        if frozen and i < 0:
            v, u, w = model['v0'] + i * model['s0'], model['u0'], model['w0']
        color = model['cycle'][i % len(model['cycle'])]
        pixels = [(v, u)] if one_per_bar else [(r, c)
                   for r in range(max(0, v), min(extent, v + model['t']))
                   for c in range(max(0, u), min(width, u + w))]
        for r, c in pixels:
            if 0 <= r < extent and 0 <= c < width:
                p = (r, c) if model['axis'] == 0 else (c, r)
                cells.append((p, color))
    return cells


def _coverage(cells, records, bounds):
    r0, r1, c0, c1 = bounds
    counts = {x['target']: np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=np.int16)
              for x in records}
    by_source = {x['source']: x for x in records}
    for (ar, ac), color in cells:
        rec = by_source[color]
        for dr, dc in rec['offsets']:
            r, c = ar + dr, ac + dc
            if r0 <= r <= r1 and c0 <= c <= c1:
                counts[rec['target']][r - r0, c - c0] += 1
    return counts


def _paint(counts, bg, parity=True):
    shape = next(iter(counts.values())).shape
    out = np.full(shape, bg, dtype=np.int8)
    active = np.zeros(shape, dtype=np.int8)
    for target, count in counts.items():
        hit = count % 2 == 1 if parity else count > 0
        active += hit
        out[hit] = target
    _need(bool(np.all(active <= 1)), 'different odd target colors collide')
    return out


def _decode(input_grid):
    a = _grid(input_grid)
    marker, bounds = _corners(a)
    freq = np.bincount(a.ravel().astype(int), minlength=10)
    _need(np.count_nonzero(freq == freq.max()) == 1, 'tied background')
    bg = int(freq.argmax())
    _need(bg != marker, 'marker equals background')
    r0, r1, c0, c1 = bounds
    inside = a[r0:r1 + 1, c0:c1 + 1]
    _need(bool(np.all((inside == bg) | (inside == marker))), 'nonempty erased window')
    interpretations = []
    for neutral in map(int, np.unique(a)):
        if neutral in (bg, marker):
            continue
        try:
            records = _instructions(a, bg, marker, neutral)
            clean = a.copy()
            clean[(clean == marker) | (clean == neutral)] = bg
            for rec in records:
                for p in rec['labels']:
                    clean[p] = bg
            sources = {rec['source'] for rec in records}
            _need(set(map(int, np.unique(clean))) <= sources | {bg}, 'unassigned content')
            for model in _fits(clean, bg, sources):
                cells = _support_cells(a.shape, model)
                counts = _coverage(cells, records, bounds)
                out = _paint(counts, bg)
                interpretations.append(dict(bg=bg, marker=marker, neutral=neutral,
                                            bounds=bounds, records=records, model=model,
                                            cells=cells, clean=clean, counts=counts, output=out,
                                            shape=a.shape))
        except ValueError:
            continue
    _need(len(interpretations) == 1, 'input has no unique role/axis interpretation')
    return interpretations[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the shadow exclusively from this input, without puzzle state."""
    return _decode(input_grid)['output'].copy()


def _different(x, y):
    return x.shape != y.shape or not np.array_equal(x, y)


def _crop_distance(d):
    m, bb = d['model'], d['bounds']
    lo, hi = (bb[0], bb[1]) if m['axis'] == 0 else (bb[2], bb[3])
    extent = d['shape'][m['axis']]
    distances = []
    for i in _indices(m, extent):
        if 0 <= i < m['n']:
            continue
        v, u, w = _bar(m, i)
        if v <= hi and v + m['t'] > lo:
            distances.append(i - m['n'] if i >= m['n'] else -i - 1)
    return min(distances) if distances else None


def _features(d):
    bb, bg, out = d['bounds'], d['bg'], d['output']
    r0, r1, c0, c1 = bb
    restricted = [(p, c) for p, c in d['cells'] if r0 <= p[0] <= r1 and c0 <= p[1] <= c1]
    observed = [(tuple(map(int, p)), int(d['clean'][tuple(p)]))
                for p in np.argwhere(d['clean'] != bg)]
    def differs(cells):
        try:
            alternative = _paint(_coverage(cells, d['records'], bb), bg)
        except ValueError:
            # A shortcut yielding contradictory colors cannot match the answer.
            return True
        return _different(out, alternative)

    before = differs(restricted)
    explicit = differs(observed)
    frozen = differs(_support_cells(d['shape'], d['model'], frozen=True))
    single = differs(_support_cells(d['shape'], d['model'], one_per_bar=True))
    identity = [dict(rec, offsets=((0, 0),)) for rec in d['records']]
    try:
        identity_out = _paint(_coverage(d['cells'], identity, bb), bg)
        identity_diff = _different(out, identity_out)
    except ValueError:
        identity_diff = True
    raw = np.full(out.shape, bg, dtype=np.int8)
    for (r, c), color in restricted:
        raw[r - r0, c - c0] = color
    even = sum(int(np.count_nonzero((x > 0) & (x % 2 == 0))) for x in d['counts'].values())
    return dict(omitting_A=explicit, freezing_geometry=frozen,
                omitting_B=_different(out, raw), omitting_B_geometry=identity_diff, crop_first=before,
                per_object=single, even_cells=even,
                holes=sum((0, 0) not in x['offsets'] for x in d['records']),
                occupied=sum((0, 0) in x['offsets'] for x in d['records']),
                painted=int(np.count_nonzero(out != bg)), skipped_bars=_crop_distance(d))


def _turn(p, k):
    r, c = p
    for _ in range(k):
        r, c = c, -r
    return int(r), int(c)


def _shape(rng, hole=False):
    h, w = int(rng.integers(2, 6)), int(rng.integers(2, 6))
    style = int(rng.integers(6))
    if hole:
        h = int(rng.integers(3, 6))
        cells = {(r, w - 1) for r in range(h)}
        cells |= {(r, c) for r in (0, h - 1) for c in range(w)}
        q = (1, 0)  # An edge hole, with empty label cells to its left.
    else:
        if style == 0:
            cells = {(r, c) for r in range(h) for c in range(w)
                     if r in (0, h - 1) or c in (0, w - 1)}
        elif style == 1:
            cells = {(r, 0) for r in range(h)} | {(h - 1, c) for c in range(w)}
        elif style == 2:
            cells = {(r, c) for r in range(h) for c in (min(r, w - 1), min(r + 1, w - 1))}
        elif style == 3:
            cells = {(r, 0) for r in range(h)} | {(h // 2, c) for c in range(w)}
            cells |= {(r, w - 1) for r in range(h // 2, h)}
        elif style == 4:
            cells = {(r, c) for r in range(h) for c in range(w)}
            cells -= {(r, c) for r in range(1, h - 1) for c in range(1, w - 1)
                      if rng.random() < .7}
        else:
            cells = {(0, 0), (0, 1), (1, 0)}
            for _ in range(int(rng.integers(3, 14))):
                r, c = sorted(cells)[int(rng.integers(len(cells)))]
                dr, dc = DIRS[int(rng.integers(4))]
                if 0 <= r + dr < h and 0 <= c + dc < w:
                    cells.add((r + dr, c + dc))
        left = min(c for r, c in cells)
        qs = sorted(p for p in cells if p[1] == left)
        q = qs[0] if rng.random() < .5 else qs[-1]
    offsets = tuple(sorted((r - q[0], c - q[1]) for r, c in cells))
    if rng.random() < .5:
        offsets = tuple((-r, c) for r, c in offsets)
    rotation = int(rng.integers(4))
    return tuple(sorted(_turn(p, rotation) for p in offsets)), _turn((0, 1), rotation)


def _place(rng, a, bg, marker, neutral, bounds, specifications):
    forbidden = {(r, c) for r in range(bounds[0], bounds[1] + 1)
                 for c in range(bounds[2], bounds[3] + 1)}
    forbidden |= set(map(tuple, np.argwhere(a != bg)))
    halo, records = set(), []
    for source, target, offsets, direction in specifications:
        dr, dc = direction
        relative = set(offsets) | {(0, 0), (-dr, -dc), (-2 * dr, -2 * dc)}
        rr0, rr1 = min(r for r, c in relative), max(r for r, c in relative)
        cc0, cc1 = min(c for r, c in relative), max(c for r, c in relative)
        qs = [(r, c) for r in range(-rr0, a.shape[0] - rr1)
              for c in range(-cc0, a.shape[1] - cc1)]
        for k in rng.permutation(len(qs)):
            qr, qc = qs[int(k)]
            mask = {(qr + r, qc + c) for r, c in offsets}
            sl, tl = (qr - 2 * dr, qc - 2 * dc), (qr - dr, qc - dc)
            occupied = mask | {(qr, qc), sl, tl}
            if occupied & forbidden or mask & halo:
                continue
            candidate = a.copy()
            for p in mask:
                candidate[p] = neutral
            candidate[sl], candidate[tl] = source, target
            try:
                decoded = _instructions(candidate, bg, marker, neutral)
            except ValueError:
                continue
            rec = next((x for x in decoded if x['source'] == source), None)
            if len(decoded) != len(records) + 1 or rec is None or rec['q'] != (qr, qc):
                continue
            if rec['offsets'] != offsets:
                continue
            a = candidate
            records.append(rec)
            forbidden |= occupied
            halo |= {(r + u, c + v) for r, c in mask for u in (-1, 0, 1) for v in (-1, 0, 1)}
            break
        else:
            raise ValueError('instructions do not fit')
    return a, records


def _latent_grid(shape, p):
    """Construction path: explicit latent rectangles, including clipped ones."""
    latent = np.full(shape, p['bg'], dtype=np.int8)
    n = p['n']
    for i in range(-30, 41):
        w = p['w0'] + i * p['dw']
        lo, hi = min(0, i), max(n - 2, i - 1)
        # All steps connecting this slot to the observed interval must exist.
        valid_gaps = [p['s0'] + k * p['ds'] for k in range(lo, hi + 1)]
        if w <= 0 or min(valid_gaps) <= p['t']:
            continue
        if min(p['w0'], w) <= 0:
            continue
        r = p['v0'] + i * p['s0'] + i * (i - 1) // 2 * p['ds']
        c = p['u0'] + i * p['du']
        r0, r1 = max(0, r), min(shape[0], r + p['t'])
        c0, c1 = max(0, c), min(shape[1], c + w)
        if r0 < r1 and c0 < c1:
            latent[r0:r1, c0:c1] = p['cycle'][i % len(p['cycle'])]
    return latent


def _truth(latent, records, bg, bounds):
    """Construct the entire shadow by array translations, then extract it."""
    count = {x['target']: np.zeros(latent.shape, dtype=np.int16) for x in records}
    for rec in records:
        rows, cols = np.where(latent == rec['source'])
        for dr, dc in rec['offsets']:
            rr, cc = rows + dr, cols + dc
            valid = (rr >= 0) & (rr < latent.shape[0]) & (cc >= 0) & (cc < latent.shape[1])
            count[rec['target']][rr[valid], cc[valid]] += 1
    r0, r1, c0, c1 = bounds
    # Conflicts outside the requested window are irrelevant to its answer.
    cropped = {c: a[r0:r1 + 1, c0:c1 + 1] for c, a in count.items()}
    return _paint(cropped, bg)


def _reflect(a, out, bounds, flip_r, flip_c, transpose, marker):
    r0, r1, c0, c1 = bounds
    a = a.copy()
    # The caller supplies an unannotated canvas; redraw Ls only at the end.
    if flip_r:
        r0, r1 = a.shape[0] - 1 - r1, a.shape[0] - 1 - r0
        a, out = a[::-1].copy(), out[::-1].copy()
    if flip_c:
        c0, c1 = a.shape[1] - 1 - c1, a.shape[1] - 1 - c0
        a, out = a[:, ::-1].copy(), out[:, ::-1].copy()
    if transpose:
        a, out = a.T.copy(), out.T.copy()
        r0, r1, c0, c1 = c0, c1, r0, r1
    for p in _corner_cells((r0, r1, c0, c1)):
        a[p] = marker
    return {'input': a, 'output': out}


def _sample_pair(rng, palette, test, index, axis_shift, axis):
    bg, marker, neutral = palette[:3]
    teaching = not test and index < 2
    count = 1 if teaching and index == 0 else 2 if teaching else int(rng.integers(1, 3))
    colors = palette[3:]
    if teaching and index == 0:
        mappings = [(colors[0], colors[1])]
    elif teaching:
        mappings = [(colors[0], colors[2]), (colors[2], colors[1])]
    else:
        c = [int(x) for x in rng.permutation(colors)]
        if count == 1:
            mappings = [(c[0], c[1])]
        else:
            mode = int(rng.integers(4))
            mappings = ([(c[0], c[2]), (c[1], c[2])] if mode == 0 else
                        [(c[0], c[1]), (c[1], c[2])] if mode == 1 else
                        [(c[0], c[1]), (c[1], c[0])] if mode == 2 else
                        [(c[0], c[2]), (c[1], c[3])])
    n = 4 if teaching else int(rng.integers(4, 6))
    t = int(rng.integers(1, 3))
    s0 = t + int(rng.integers(1, 4))
    ds = 0 if rng.random() < .72 else int(rng.choice((-1, 1)))
    skip = int(rng.integers(1, 3)) if test else 0
    span = 2 if teaching and index == 1 else int(rng.integers(1, 3))
    last = n + skip + span - 1
    if s0 + min(0, (last - 1) * ds) <= t:
        raise ValueError('invalid gaps')
    dw = 1 if teaching and index == 0 else int(rng.choice((-1, 0, 1, 1, 2)))
    w0 = int(rng.integers(2, 8)) + (last * -dw if dw < 0 else 0)
    du = int(rng.choice((-2, -1, 0, 0, 1, 2)))
    if min(w0, w0 + last * dw) < 2:
        raise ValueError('invalid lengths')
    v0 = int(rng.integers(0, 4))
    umin = min(0, last * du)
    umax = max(i * du + w0 + i * dw for i in range(last + 1))
    min_width = max(14, umax - umin + 2)
    if min_width > 30:
        raise ValueError('transverse span too large')
    width = int(rng.integers(min_width, min(30, min_width + 8) + 1))
    u0 = int(rng.integers(-umin, width - umax + 1))
    cycle = tuple(s for s, target in mappings)
    if not teaching and count == 2 and rng.random() < .5:
        cycle = cycle[::-1]
    p = dict(bg=bg, axis=0, v0=v0, s0=s0, ds=ds, u0=u0, du=du,
             w0=w0, dw=dw, t=t, n=n, cycle=cycle)
    first = n + skip
    fv, fu, fw = _bar(p, first)
    lv, lu, lw = _bar(p, last)
    last_visible = _bar(p, n - 1)[0] + t - 1
    previous_end = _bar(p, first - 1)[0] + t - 1
    r0 = max(last_visible + 1, previous_end + 1, fv - int(rng.integers(0, 3)))
    r1 = max(r0 + 2, lv + t - 1 + int(rng.integers(0, 4)))
    min_height = max(12, r1 + 1)
    if min_height > 30:
        raise ValueError('axial span too large')
    height = int(rng.integers(min_height, min(30, min_height + 6) + 1))
    c0 = max(0, min(fu, lu) - int(rng.integers(0, 3)))
    c1 = min(width - 1, max(fu + fw - 1, lu + lw - 1) + int(rng.integers(0, 3)))
    if not teaching and rng.random() < .45 and c1 - c0 > 6:
        c0 += int(rng.integers(0, min(5, c1 - c0 - 3)))
        c1 -= int(rng.integers(0, min(4, c1 - c0 - 3)))
    if c1 - c0 < 3:
        c0, c1 = max(0, c0 - 1), min(width - 1, c0 + 3)
    bb = (r0, r1, c0, c1)
    a = np.full((height, width), bg, dtype=np.int8)
    for i in range(n):
        v, u, w = _bar(p, i)
        if not (0 <= v and v + t <= height and 0 <= u and u + w <= width):
            raise ValueError('visible bar clipped')
        a[v:v + t, u:u + w] = cycle[i % count]
    specs = []
    for k, (source, target) in enumerate(mappings):
        hole = (teaching and index == 1 and k == 0) or (not teaching and rng.random() < .45)
        offsets, direction = _shape(rng, hole)
        specs.append((source, target, offsets, direction))
    a, records = _place(rng, a, bg, marker, neutral, bb, specs)
    latent = _latent_grid(a.shape, p)
    out = _truth(latent, records, bg, bb)
    if not np.any(out != bg):
        raise ValueError('empty shadow')
    transpose = bool(axis ^ int(test and axis_shift))
    pair = _reflect(a, out, bb, bool(rng.integers(2)), bool(rng.integers(2)), transpose, marker)
    d = _decode(pair['input'])
    if not np.array_equal(d['output'], pair['output']):
        raise ValueError('latent construction and input prediction disagree')
    f = _features(d)
    if not all(f[x] for x in ('omitting_A', 'omitting_B', 'omitting_B_geometry')):
        raise ValueError('one parent does not affect the answer')
    if teaching:
        if not all(f[x] for x in ('omitting_A', 'freezing_geometry', 'omitting_B', 'omitting_B_geometry',
                                 'crop_first', 'per_object', 'even_cells')):
            raise ValueError('teaching counterfactual not distinguished')
        if index == 0 and not f['occupied']:
            raise ValueError('missing occupied registration')
        if index == 0 and not any(sum(r for r, c in rec['offsets']) != 0 or
                                  sum(c for r, c in rec['offsets']) != 0
                                  for rec in d['records']):
            raise ValueError('missing off-center registration')
        if index == 1 and not f['holes']:
            raise ValueError('missing hole registration')
        if index == 1 and not all(rec['target'] in out for rec in records):
            raise ValueError('one alternating source is not demonstrated')
    return pair


def _count(value, low, high, name):
    _need(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)),
          name + ' must be an integer')
    _need(low <= int(value) <= high, name + ' out of range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate a deterministic puzzle with stable role conventions."""
    _need(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_))
          and int(seed) >= 0, 'seed must be a nonnegative integer')
    nt = _count(num_train, 2, 10, 'num_train')
    ne = _count(num_test, 1, 4, 'num_test')
    _need(nt > ne, 'num_train must exceed num_test')
    rng = np.random.default_rng(int(seed))
    palette = tuple(map(int, rng.permutation(10)))
    shift, axis = bool(rng.integers(2)), int(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ne)):
        for index in range(count):
            for _ in range(500):
                try:
                    pair = _sample_pair(rng, palette, split == 'test', index, shift, axis)
                except ValueError:
                    continue
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded distinct-pair sampling exhausted')
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _need(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'invalid splits')
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, 'invalid counts')
    seen, decoded = set(), []
    for pair in puzzle['train'] + puzzle['test']:
        _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
        a, out = _grid(pair['input']), _grid(pair['output'])
        key = _key(a)
        _need(key not in seen, 'duplicate inputs')
        seen.add(key)
        d = _decode(a)
        _need(np.array_equal(d['output'], out), 'incorrect shadow output')
        _need(d['marker'] not in out and d['neutral'] not in out, 'annotation leaked')
        _need(np.any(out != d['bg']), 'empty shadow')
        _need(set(map(int, np.unique(out))) <= {d['bg']} | {x['target'] for x in d['records']},
              'unexpected output color')
        f = _features(d)
        _need(f['omitting_A'] and f['omitting_B'] and f['omitting_B_geometry'],
              'parent contribution lost')
        decoded.append(d)
    _need(len({(d['bg'], d['neutral'], d['marker']) for d in decoded}) == 1, 'role convention changed')
    for d in decoded[:2]:
        f = _features(d)
        _need(all(f[x] for x in ('omitting_A', 'omitting_B', 'omitting_B_geometry', 'freezing_geometry',
                                'crop_first', 'per_object', 'even_cells')),
              'insufficient teaching demonstration')
    _need(all(_crop_distance(d) == 0 for d in decoded[:nt]),
          'training is not immediate continuation')
    _need(all(_crop_distance(d) in (1, 2) for d in decoded[nt:]),
          'test does not skip unseen bars')
    _need(_features(decoded[0])['occupied'] > 0, 'missing occupied registration')
    _need(any(sum(r for r, c in rec['offsets']) != 0 or
              sum(c for r, c in rec['offsets']) != 0 for rec in decoded[0]['records']),
          'missing off-center registration')
    _need(_features(decoded[1])['holes'] > 0 and len(decoded[1]['records']) == 2,
          'missing hole/multiple-source demonstration')
    _need(all(rec['target'] in decoded[1]['output'] for rec in decoded[1]['records']),
          'one alternating source is not demonstrated')
    maps = [{r['source']: r['target'] for r in d['records']} for d in decoded[:2]]
    _need(any(c in maps[1] and maps[0][c] != maps[1][c] for c in maps[0]),
          'local mapping change not demonstrated')


def self_test():
    """Bounded semantic counterfactuals, diversity, schema and count tests."""
    # Hand-calculated spatial-difference regressions, without generated labels.
    # Run lengths 2,3,4,5 imply lengths 6 and 7 at rows 13 and 15.
    # A two-cell stencil cancels interiors; three cells retain odd triple hits.
    manual = ((False, 2, ('200000200', '000000000', '200000020', '000000000')),
              (True, 2, ('020000020', '000000000', '020000002', '000000000')),
              (False, 3, ('202222020', '000000000', '202222202', '000000000')))
    for hole, size, rows in manual:
        a = np.zeros((18, 16), dtype=np.int8)
        a[3, 1], a[2, 1] = 1, 2
        a[1, 1 + int(hole):1 + int(hole) + size] = 8
        for i in range(4):
            a[5 + 2 * i, 4:6 + i] = 1
        for q in _corner_cells((13, 16, 4, 12)):
            a[q] = 6
        expected = np.asarray([[int(c) for c in row] for row in rows], dtype=np.int8)
        assert np.array_equal(solve(a), expected)
        snapshot = a.copy()
        solve(a)
        assert np.array_equal(a, snapshot), 'solver mutated input'
    features, structures, shapes, masks, directions, palettes, regimes = [], set(), set(), set(), set(), set(), set()
    totals = Counter()
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        structures.add(_puzzle_key(p))
        for split in ('train', 'test'):
            for pair in p[split]:
                d = _decode(pair['input'])
                m, f = d['model'], _features(d)
                features.append(m)
                shapes.add((pair['input'].shape, pair['output'].shape))
                palettes.add((d['bg'], d['neutral'], d['marker']))
                regimes.add((m['axis'], int(np.sign(m['dw']))))
                for rec in d['records']:
                    masks.add(rec['offsets'])
                    directions.add(rec['direction'])
                    totals['holes'] += (0, 0) not in rec['offsets']
                totals['growth'] += m['dw'] != 0
                totals['drift'] += m['du'] != 0
                totals['gap_change'] += m['ds'] != 0
                totals['even'] += f['even_cells'] > 0
                totals['outside'] += f['crop_first']
                totals['two_sources'] += len(d['records']) == 2
    assert len(structures) == 42 and len(shapes) >= 60 and len(masks) >= 60
    assert len(palettes) >= 35 and len(directions) == 4 and len(regimes) >= 5
    assert all(totals[k] >= v for k, v in dict(holes=40, growth=100, drift=50,
                gap_change=5, even=100, outside=100, two_sources=70).items()), totals
    assert {m['t'] for m in features} == {1, 2}
    assert {m['n'] for m in features} == {4, 5}
    assert len({m['w0'] for m in features}) >= 5
    assert _puzzle_key(generate_puzzle(771, 3, 2)) == _puzzle_key(generate_puzzle(771, 3, 2))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(812 + nt * 10 + ne, nt, ne)
            validate_generated_puzzle(p)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    broken = generate_puzzle(913, 2, 1)
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError('corrupted output accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed')
