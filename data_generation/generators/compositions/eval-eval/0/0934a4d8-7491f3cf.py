"""Reconstruct an erased window in a region-selected, two-layer mirror mosaic.

Only standard-library modules and NumPy are used. No files are read or written.
The sampler renders complete latent layers; the solver instead enumerates the
small set of layer values compatible with visible pixels in each mirror orbit.
"""
from collections import Counter
import hashlib

import numpy as np


HOST_SEEDS = (0, 889574166, 631229316, 270916295, 432114396, 763574095,
              300421726, 714308490, 625793198, 386769470, 271430270, 865613832,
              688938394, 930547135, 815383217, 37967746, 709556767, 273070165,
              133749473, 811852504, 243096340, 532393536, 27222136, 69355725,
              29749384, 698912258, 788929099, 661591130, 557586742, 479140398,
              352381055, 32347263, 858393761, 630152667, 891587749, 269009133,
              291067130, 888116920, 569038770, 919703156, 116546, 906488362)
_EVIDENCE = frozenset(('collision', 'b_only', 'a_only', 'blank',
                       'marked_fg', 'other_fg', 'naive_wrong', 'flat_wrong'))


def _require(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'invalid grid shape')
    _require(np.issubdtype(a.dtype, np.integer), 'grid must contain integers')
    _require(np.all((a >= 0) & (a <= 9)), 'invalid ARC color')
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask, diagonal=False):
    h, w = mask.shape
    # Use Python booleans and flat integer offsets inside flood fill. Repeated
    # NumPy scalar indexing here dominated rejection-sampling runtime.
    flat, seen, groups = mask.ravel().tolist(), bytearray(h * w), []
    directions = [(r, c, r * w + c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                  if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    for origin, active in enumerate(flat):
        if not active or seen[origin]:
            continue
        seen[origin] = 1
        stack, group = [origin], []
        while stack:
            current = stack.pop()
            r, c = divmod(current, w)
            group.append((r, c))
            for dr, dc, offset in directions:
                nr, nc, q = r + dr, c + dc, current + offset
                if 0 <= nr < h and 0 <= nc < w and flat[q] and not seen[q]:
                    seen[q] = 1
                    stack.append(q)
        groups.append(group)
    return groups


def _mask(shape, cells):
    a = np.zeros(shape, dtype=bool)
    for p in cells:
        a[p] = True
    return a


def _cut_info(cut):
    """Simple digital path; no diagonal shortcut around an orthogonal elbow."""
    h, w = cut.shape
    cells = {tuple(map(int, p)) for p in np.argwhere(cut)}
    graph = {p: [] for p in cells}
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                q = r + dr, c + dc
                if (dr or dc) and q in cells:
                    if dr and dc and ((r + dr, c) in cells or (r, c + dc) in cells):
                        continue
                    graph[r, c].append(q)
    ends = [p for p, v in graph.items() if len(v) == 1]
    _require(len(cells) >= 3 and len(ends) == 2 and
             all(1 <= len(v) <= 2 for v in graph.values()), 'cut must be a simple path')
    boundary = lambda p: p[0] in (0, h - 1) or p[1] in (0, w - 1)
    _require(all(boundary(p) for p in ends) and
             all(not boundary(p) for p in cells if p not in ends), 'invalid cut endpoints')
    order, previous, p = [], None, min(ends)
    while p is not None and p not in order:
        order.append(p)
        following = [q for q in graph[p] if q != previous]
        previous, p = p, following[0] if following else None
    _require(len(order) == len(cells), 'cut has a disconnected loop')
    regions = [_mask(cut.shape, g) for g in _components(~cut)]
    _require(len(regions) == 2, 'cut must leave exactly two four-connected regions')
    near = cut.copy()
    for r, c in cells:
        near[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True
    sites = [[tuple(map(int, p)) for p in np.argwhere(region & ~near)] for region in regions]
    _require(all(sites), 'both sides need a separated marker site')
    return regions, sites, order


def _pack(control, mosaic, frame, vertical=False):
    h, w = control.shape
    result = np.full((2 * h + 3, w + 2) if vertical else (h + 2, 2 * w + 3),
                     frame, dtype=np.int8)
    result[1:h + 1, 1:w + 1] = control
    origin = (h + 2, 1) if vertical else (1, w + 2)
    r, c = origin
    result[r:r + h, c:c + w] = mosaic
    return result


def _decode(input_grid):
    g = _grid(input_grid)
    frame, (gh, gw) = int(g[0, 0]), g.shape
    _require(np.all(g[[0, -1], :] == frame) and np.all(g[:, [0, -1]] == frame),
             'broken outer frame')
    # Frame color never occurs in an interior, so exactly one complete divider exists.
    rows = [r for r in range(1, gh - 1) if np.all(g[r] == frame)]
    cols = [c for c in range(1, gw - 1) if np.all(g[:, c] == frame)]
    _require(len(rows) + len(cols) == 1, 'need exactly two congruent panels')
    vertical = bool(rows)
    if vertical:
        h, w = (gh - 3) // 2, gw - 2
        _require(gh == 2 * h + 3 and rows == [h + 1], 'unequal stacked panels')
        control, mosaic = g[1:h + 1, 1:-1], g[h + 2:-1, 1:-1]
    else:
        h, w = gh - 2, (gw - 3) // 2
        _require(gw == 2 * w + 3 and cols == [w + 1], 'unequal side panels')
        control, mosaic = g[1:-1, 1:w + 1], g[1:-1, w + 2:-1]
    _require(min(h, w) >= 5 and frame not in control and frame not in mosaic,
             'interior dimensions or frame role invalid')
    colors = list(map(int, np.unique(control)))
    _require(len(colors) == 2, 'control must have two colors')
    interpretations = []
    for ink in colors:
        ink_mask = control == ink
        padded = np.pad(ink_mask, 1)
        neighbors = np.zeros(ink_mask.shape, dtype=bool)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr or dc:
                    neighbors |= padded[1 + dr:h + 1 + dr, 1 + dc:w + 1 + dc]
        # A valid ink interpretation must have exactly one singleton marker.
        # Reject the usual background interpretation without flood-filling it.
        if np.count_nonzero(ink_mask & ~neighbors) != 1:
            continue
        groups = sorted(_components(ink_mask, True), key=len)
        if len(groups) != 2 or len(groups[0]) != 1 or len(groups[1]) < 3:
            continue
        try:
            cut = _mask(control.shape, groups[1])
            regions, sites, path = _cut_info(cut)
            marker = groups[0][0]
            side = 0 if regions[0][marker] else 1
            _require(marker in sites[side], 'marker touches cut')
            interpretations.append((ink, cut, regions, sites, path, marker, side))
        except AssertionError:
            continue
    _require(len(interpretations) == 1, 'ambiguous control diagram')
    ink, cut, regions, sites, path, marker, side = interpretations[0]
    bg = next(c for c in colors if c != ink)
    _require(ink not in mosaic, 'control color is not content')
    rectangles = []
    for color in map(int, np.unique(mosaic)):
        if color == bg:
            continue
        rr, cc = np.nonzero(mosaic == color)
        r0, r1, c0, c1 = int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1
        if len(rr) >= 2 and len(rr) == (r1 - r0) * (c1 - c0):
            rectangles.append((color, (r0, r1, c0, c1)))
    _require(len(rectangles) == 1, 'erasure must be the unique solid rectangular color')
    eraser, box = rectangles[0]
    _require(len({frame, bg, ink, eraser}) == 4, 'structural roles must differ')
    marked, other = regions[side], regions[1 - side]
    palettes = [tuple(sorted(set(map(int, mosaic[region])) - {bg, eraser}))
                for region in (marked, other)]
    _require(all(1 <= len(p) <= 2 for p in palettes) and not set(palettes[0]) & set(palettes[1]),
             'layers need disjoint foreground palettes visible off the cut')
    _require(set(map(int, mosaic.flat)) <= {bg, eraser, *palettes[0], *palettes[1]},
             'a content color lacks an off-cut role witness')
    return dict(grid=g, control=control, mosaic=mosaic, frame=frame, bg=bg, ink=ink,
                eraser=eraser, box=box, cut=cut, marked=marked, other=other, side=side,
                palettes=palettes, sites=sites, path=path, marker=marker, vertical=vertical)


def _orbits(shape):
    h, w = shape
    return [[(r, c) for r in sorted({i, h - 1 - i}) for c in sorted({j, w - 1 - j})]
            for i in range((h + 1) // 2) for j in range((w + 1) // 2)]


def _value(a, b, bg, cut, marked, p):
    return (a if a != bg else b) if cut[p] else (a if marked[p] else b)


def _infer(input_grid):
    d = _decode(input_grid)
    m, bg, eraser = d['mosaic'], d['bg'], d['eraser']
    r0, r1, c0, c1 = d['box']
    answer = np.full((r1 - r0, c1 - c0), -1, dtype=np.int8)
    records = []
    for cells in _orbits(m.shape):
        candidates = [(a, b) for a in (bg, *d['palettes'][0]) for b in (bg, *d['palettes'][1])
                      if all(m[p] == eraser or
                             int(m[p]) == _value(a, b, bg, d['cut'], d['marked'], p)
                             for p in cells)]
        _require(candidates, 'visible pixels violate within-layer mirror symmetry')
        for p in cells:
            if m[p] == eraser:
                values = {_value(a, b, bg, d['cut'], d['marked'], p) for a, b in candidates}
                _require(len(values) == 1, 'erased pixel has multiple compatible completions')
                answer[p[0] - r0, p[1] - c0] = next(iter(values))
        records.append((cells, candidates))
    _require(np.all(answer >= 0), 'incomplete crop')
    d['orbits'] = records
    return answer, d


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the target crop solely from the input, without generation state."""
    return _infer(input_grid)[0]


def _evidence(d, answer):
    m, bg, eraser, cut = d['mosaic'], d['bg'], d['eraser'], d['cut']
    r0, r1, c0, c1 = d['box']
    flags = set()
    if np.any(d['marked'][r0:r1, c0:c1] & (answer != bg)):
        flags.add('marked_fg')
    if np.any(d['other'][r0:r1, c0:c1] & (answer != bg)):
        flags.add('other_fg')
    for cells, candidates in d['orbits']:
        # Priority evidence must be known from STRICT regional observations,
        # independently of any assumed cut priority.
        av = {int(m[p]) for p in cells if m[p] != eraser and d['marked'][p]}
        bv = {int(m[p]) for p in cells if m[p] != eraser and d['other'][p]}
        if len(av) == len(bv) == 1:
            a, b = next(iter(av)), next(iter(bv))
            for p in cells:
                if cut[p]:
                    name = ('collision' if a != bg and b != bg else
                            'a_only' if a != bg else 'b_only' if b != bg else 'blank')
                    # Collision and lower-layer-through-background must affect a label.
                    if name in ('a_only', 'blank') or m[p] == eraser:
                        flags.add(name)
        for p in cells:
            if m[p] != eraser:
                continue
            visible = [q for q in cells if m[q] != eraser]
            if visible:
                q = min(visible, key=lambda q: (abs(p[0] - q[0]) + abs(p[1] - q[1]), q))
                if m[q] != answer[p[0] - r0, p[1] - c0]:
                    flags.add('naive_wrong')
    # Regionwise modal filling discards the inherited reflection mechanism.
    for region in (d['marked'], d['other']):
        visible = m[region & (m != eraser)]
        if visible.size:
            counts = Counter(map(int, visible))
            modal = min(counts, key=lambda x: (-counts[x], x))
            target = region[r0:r1, c0:c1]
            if np.any(target & (answer != modal)):
                flags.add('flat_wrong')
    return flags


def _line(p, q):
    r, c = p
    r1, c1 = q
    dr, dc = abs(r1 - r), abs(c1 - c)
    sr, sc, error = (1 if r < r1 else -1), (1 if c < c1 else -1), dc - dr
    cells = []
    while True:
        cells.append((r, c))
        if (r, c) == (r1, c1):
            return cells
        twice = 2 * error
        if twice > -dr:
            error -= dr
            c += sc
        if twice < dc:
            error += dc
            r += sr


def _sample_cut(rng, h, w, family):
    cut = np.zeros((h, w), dtype=bool)
    if family == 'straight':
        cut[:, int(rng.integers(2, w - 2))] = True
    elif family == 'diagonal':
        c0, c1 = rng.choice(np.arange(1, w - 1), 2, replace=False)
        cut = _mask((h, w), _line((0, int(c0)), (h - 1, int(c1))))
    elif family == 'elbow':
        r, c = int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2))
        cut[:r + 1, c] = True
        cut[r, c:] = True
    else:
        steps = list(range(2, h - 2, 3))
        n = int(rng.integers(1, min(3, len(steps)) + 1))
        rows = sorted(map(int, rng.choice(steps, n, replace=False)))
        last, col = 0, int(rng.integers(2, w - 2))
        for row in rows:
            cut[last:row + 1, col] = True
            choices = [c for c in range(1, w - 1) if c != col]
            new = int(rng.choice(choices))
            cut[row, min(col, new):max(col, new) + 1] = True
            last, col = row, new
        cut[last:, col] = True
    if rng.random() < .5:
        cut = cut[::-1]
    if rng.random() < .5:
        cut = cut[:, ::-1]
    return cut.copy()


def _pattern(rng, shape, bg, colors, style):
    h, w = shape
    bh, bw = (h + 1) // 2, (w + 1) // 2
    rr, cc = np.indices((bh, bw))
    if style == 0:
        mask = rng.random((bh, bw)) < float(rng.uniform(.25, .75))
    elif style == 1:
        period = int(rng.integers(2, 4))
        mask = (rr + cc + int(rng.integers(period))) % period != 0
    elif style == 2:
        axis = rr if rng.random() < .5 else cc
        mask = (axis + int(rng.integers(3))) % int(rng.integers(2, 4)) == 0
    elif style == 3:
        a, b = sorted(map(int, rng.choice(bh + 1, 2, replace=False)))
        c, e = sorted(map(int, rng.choice(bw + 1, 2, replace=False)))
        mask = (rr >= a) & (rr < b) & (cc >= c) & (cc < e)
        if rng.random() < .5:
            mask &= (rr == a) | (rr == b - 1) | (cc == c) | (cc == e - 1)
    elif style == 4:
        mask = (rr == int(rng.integers(bh))) | (cc == int(rng.integers(bw)))
    else:
        mask = np.zeros((bh, bw), dtype=bool)
        for _ in range(int(rng.integers(2, 6))):
            r, c = int(rng.integers(bh)), int(rng.integers(bw))
            mask[r:r + int(rng.integers(1, 4)), c:c + int(rng.integers(1, 4))] = True
    base = np.full((bh, bw), bg, dtype=np.int8)
    paint = np.asarray(colors, dtype=np.int8)[(rr + cc) % len(colors)]
    base[mask] = paint[mask]
    rows, cols = np.arange(h), np.arange(w)
    return base[np.minimum(rows, h - 1 - rows)[:, None],
                np.minimum(cols, w - 1 - cols)[None, :]].copy()


def _composite(a, b, bg, cut, marked):
    # Ground truth from the actual complete layers, separate from orbit inference.
    result = np.where(marked, a, b).astype(np.int8)
    result[cut] = np.where(a[cut] != bg, a[cut], b[cut])
    return result


def _edges(shape, box):
    r, s, c, e = box
    return {i for i, yes in enumerate((r == 0, e == shape[1], s == shape[0], c == 0)) if yes}


def _candidate_has_witnesses(a, b, mosaic, bg, eraser, cut, marked, box, required):
    """Cheap necessary rejection checks, followed by unchanged full inference.

    This consumes no randomness. Every tested condition was already required by
    palette validation or _evidence. It cannot reject a formerly accepted pair.
    Complete latent arrays are used only during generation, never by solve.
    """
    visible, other = mosaic != eraser, ~cut & ~marked
    for layer, region in ((a, marked), (b, other)):
        palette = set(map(int, layer.flat)) - {bg}
        observed = set(map(int, mosaic[visible & region])) - {bg}
        if not palette or palette != observed:
            return False
    if not required:
        return True
    hidden = ~visible
    if 'marked_fg' in required and not np.any(hidden & marked & (a != bg)):
        return False
    if 'other_fg' in required and not np.any(hidden & other & (b != bg)):
        return False
    seam_requirements = required & {'collision', 'b_only', 'a_only', 'blank'}
    if seam_requirements:
        def witnessed(region):
            seen = visible & region
            return seen | seen[::-1] | seen[:, ::-1] | seen[::-1, ::-1]
        strict = cut & witnessed(marked) & witnessed(other)
        states = {'collision': hidden & (a != bg) & (b != bg),
                  'b_only': hidden & (a == bg) & (b != bg),
                  'a_only': (a != bg) & (b == bg),
                  'blank': (a == bg) & (b == bg)}
        if any(not np.any(strict & states[name]) for name in seam_requirements):
            return False
    return True


def _sample_pair(rng, params, side, held_edge, test, required, seen):
    for attempt in range(1200):
        # Evidence examples use a bent/slanted cut; easy straight cases remain
        # available after the defining seam interaction has been demonstrated.
        families = ('elbow', 'diagonal', 'stepped') if required else ('straight', 'elbow', 'diagonal', 'stepped')
        family = str(rng.choice(families))
        h, w = int(rng.integers(9, 15)), int(rng.integers(8, 14))
        if rng.random() < .17:
            h = int(rng.integers(15, 27))
        if rng.random() < .5:
            # Transpose the entire local problem to sample cuts with both endpoint directions.
            h, w = w, h
        vertical = w > 13 or (h <= 13 and rng.random() < .5)
        if (vertical and h > 13) or (not vertical and w > 13):
            continue
        try:
            if rng.random() < .5:
                cut = _sample_cut(rng, w, h, family).T.copy()
            else:
                cut = _sample_cut(rng, h, w, family)
            regions, sites, path = _cut_info(cut)
        except AssertionError:
            continue
        marker = sites[side][int(rng.integers(len(sites[side])))]
        selected = regions[side]
        colors = params['colors'] if not params['vary_colors'] else tuple(map(int, rng.permutation(10)))
        frame, bg, ink, eraser = colors[:4]
        na, nb = params['layer_colors']
        ca, cb = colors[4:4 + na], colors[4 + na:4 + na + nb]
        styles = [params['style'] if rng.random() < .45 else int(rng.integers(6)) for _ in range(2)]
        a, b = _pattern(rng, (h, w), bg, ca, styles[0]), _pattern(rng, (h, w), bg, cb, styles[1])
        full = _composite(a, b, bg, cut, selected)
        hh, ww = int(rng.integers(2, min(7, h - 2))), int(rng.integers(2, min(7, w - 2)))
        focus = path[int(rng.integers(len(path)))] if required or rng.random() < .7 else (int(rng.integers(h)), int(rng.integers(w)))
        r0 = int(np.clip(focus[0] - int(rng.integers(hh)), 0, h - hh))
        c0 = int(np.clip(focus[1] - int(rng.integers(ww)), 0, w - ww))
        if test:
            if held_edge == 0:
                r0 = 0
            elif held_edge == 1:
                c0 = w - ww
            elif held_edge == 2:
                r0 = h - hh
            else:
                c0 = 0
        box = r0, r0 + hh, c0, c0 + ww
        if not test and held_edge in _edges((h, w), box):
            continue
        if hh * ww > .26 * h * w:
            continue
        control = np.full((h, w), bg, dtype=np.int8)
        control[cut] = ink
        control[marker] = ink
        mosaic = full.copy()
        mosaic[r0:r0 + hh, c0:c0 + ww] = eraser
        if not _candidate_has_witnesses(a, b, mosaic, bg, eraser, cut, selected, box, required):
            continue
        inp = _pack(control, mosaic, frame, vertical)
        if _key(inp) in seen:
            continue
        out = full[r0:r0 + hh, c0:c0 + ww].copy()
        try:
            predicted, d = _infer(inp)
            if (set(map(int, a.flat)) - {bg} != set(d['palettes'][0]) or
                    set(map(int, b.flat)) - {bg} != set(d['palettes'][1])):
                continue
            if not np.array_equal(predicted, out) or not required <= _evidence(d, out):
                continue
        except AssertionError:
            continue
        return {'input': inp, 'output': out}
    raise RuntimeError('bounded sampling exhausted while constructing a readable layer witness')


def _opposite_twin(pair):
    """Only a legal twin when reversing cut priority preserves EVERY visible cell."""
    answer, d = _infer(pair['input'])
    m, bg, eraser = d['mosaic'], d['bg'], d['eraser']
    r0, r1, c0, c1 = d['box']
    # Require that both layers are independently determined for every seam cell.
    # A twin is optional, so failure simply leaves the ordinary example in place.
    reversed_full = m.copy()
    reversed_full[r0:r1, c0:c1] = answer
    for cells, candidates in d['orbits']:
        for p in cells:
            if not d['cut'][p]:
                continue
            possible = {b if b != bg else a for a, b in candidates}
            if len(possible) != 1:
                return None
            value = next(iter(possible))
            if m[p] != eraser and m[p] != value:
                return None
            if m[p] == eraser:
                reversed_full[p] = value
    control = d['control'].copy()
    control[d['marker']] = bg
    marker = d['sites'][1 - d['side']][0]
    control[marker] = d['ink']
    inp = _pack(control, m, d['frame'], d['vertical'])
    out = reversed_full[r0:r1, c0:c1].copy()
    try:
        predicted = solve(inp)
    except AssertionError:
        return None
    if not np.array_equal(out, predicted) or np.array_equal(out, answer):
        return None
    return {'input': inp, 'output': out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for v, low, high, name in ((num_train, 2, 10, 'num_train'), (num_test, 1, 4, 'num_test')):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not low <= int(v) <= high:
            raise ValueError(name + ' is not a legal integer count')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed))
    params = dict(colors=tuple(map(int, rng.permutation(10))), vary_colors=bool(rng.random() < .7),
                  layer_colors=(int(rng.integers(1, 3)), int(rng.integers(1, 3))), style=int(rng.integers(6)))
    held_edge, first_side = int(rng.integers(4)), int(rng.integers(2))
    puzzle, seen, covered = {'train': [], 'test': []}, set(), set()
    for i in range(int(num_train)):
        side = first_side if i == 0 else 1 - first_side if i == 1 else int(rng.integers(2))
        # First teach a genuine two-layer seam collision. The second fills any
        # remaining evidence obligations even for the minimum two demonstrations.
        required = {'collision', 'marked_fg', 'other_fg', 'naive_wrong', 'flat_wrong'} if i == 0 else (
            set(_EVIDENCE) - covered if i == 1 else set())
        twin = _opposite_twin(puzzle['train'][0]) if i == 1 else None
        if twin is not None:
            twin_answer, twin_d = _infer(twin['input'])
            if not required <= _evidence(twin_d, twin_answer):
                twin = None
        pair = twin if twin is not None else _sample_pair(rng, params, side, held_edge, False, required, seen)
        puzzle['train'].append(pair)
        seen.add(_key(pair['input']))
        out, d = _infer(pair['input'])
        covered |= _evidence(d, out)
    for _ in range(int(num_test)):
        pair = _sample_pair(rng, params, int(rng.integers(2)), held_edge, True,
                            {'collision', 'b_only', 'marked_fg', 'other_fg', 'naive_wrong', 'flat_wrong'}, seen)
        puzzle['test'].append(pair)
        seen.add(_key(pair['input']))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'invalid example counts')
    seen, covered, sides = set(), set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            predicted, d = _infer(inp)
            _require(np.array_equal(out, predicted), 'incorrect target-oriented completion')
            _require(d['eraser'] not in out and d['frame'] not in out and d['ink'] not in out,
                     'structural color leaked into answer')
            if split == 'train':
                covered |= _evidence(d, out)
                sides.add(d['side'])
    _require(covered >= _EVIDENCE, 'demonstrations omit an essential distinction')
    _require(sides == {0, 1}, 'demonstrations must mark both region choices')


def self_test():
    """Bounded in-memory checks on the supplied 42 seeds; returns small metrics."""
    shapes, windows, cuts, palettes, structures, signatures = set(), set(), set(), set(), set(), set()
    colors, orientations, pair_count, twins = set(), set(), 0, 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signature = hashlib.sha256()
        train_edges, test_edges = set(), set()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                pair_count += 1
                out, d = _infer(pair['input'])
                _require(np.array_equal(out, pair['output']), 'solver disagrees')
                shapes.add(d['mosaic'].shape)
                windows.add(out.shape)
                cuts.add((d['cut'].shape, d['cut'].tobytes()))
                palettes.add(tuple(d['palettes']))
                colors.add(d['eraser'])
                orientations.add(d['vertical'])
                mapping = {}
                normalized = bytes(mapping.setdefault(int(x), len(mapping)) for x in pair['input'].flat)
                structures.add((pair['input'].shape, normalized))
                signature.update(pair['input'].tobytes())
                signature.update(out.tobytes())
                (train_edges if split == 'train' else test_edges).update(_edges(d['mosaic'].shape, d['box']))
                # A color permutation must preserve solving, independently per pair.
                perm = np.roll(np.arange(10, dtype=np.int8), 3)
                _require(np.array_equal(solve(perm[pair['input']]), perm[out]), 'color-role dependence')
                if split == 'train' and pair is puzzle['train'][0]:
                    for axis in (0, 1):
                        reflected = _pack(np.flip(d['control'], axis), np.flip(d['mosaic'], axis),
                                          d['frame'], d['vertical'])
                        _require(np.array_equal(solve(reflected), np.flip(out, axis)),
                                 'within-panel reflection changed the rule')
                    transposed = _pack(d['control'].T, d['mosaic'].T, d['frame'], not d['vertical'])
                    _require(np.array_equal(solve(transposed), out.T), 'transpose changed the rule')
        _require(bool(test_edges - train_edges), 'held-out erasure boundary missing')
        digest = signature.digest()
        signatures.add(digest)
        if (puzzle['train'][0]['input'].shape == puzzle['train'][1]['input'].shape and
                np.count_nonzero(puzzle['train'][0]['input'] != puzzle['train'][1]['input']) == 2):
            twins += 1
        if seed in (0, 889574166):
            again = generate_puzzle(seed)
            _require(all(np.array_equal(x[k], y[k]) for s in puzzle
                         for x, y in zip(puzzle[s], again[s]) for k in ('input', 'output')),
                     'seed not reproducible')
    _require(len(signatures) == 42 and len(shapes) >= 20 and len(windows) >= 15 and
             len(cuts) >= 100 and len(structures) >= 180 and len(palettes) >= 20,
             'structural diversity collapsed')
    _require(colors == set(range(10)) and orientations == {False, True}, 'role/layout variation missing')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(123456 + 10 * nt + nq, nt, nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, 'counts ignored')
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.5, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError('illegal count accepted')
    p = generate_puzzle(0)
    p['test'][0]['output'][0, 0] = (int(p['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted label accepted')
    pair = generate_puzzle(0)['train'][0]
    answer, d = _infer(pair['input'])
    full = d['mosaic'].copy()
    r0, r1, c0, c1 = d['box']
    full[r0:r1, c0:c1] = answer
    h, w = full.shape
    erased = full.copy()
    erased[h // 2 - 1:h // 2 + 1, w // 2 - 1:w // 2 + 1] = d['eraser']
    invalid_inputs = [_pack(d['control'], erased, d['frame'], d['vertical'])]
    broken_frame = pair['input'].copy()
    broken_frame[0, 0] = d['bg']
    invalid_inputs.append(broken_frame)
    # Two contradictory strict observations in the same layer's mirror orbit.
    for cells, _ in d['orbits']:
        selected = [p for p in cells if d['marked'][p] and d['mosaic'][p] != d['eraser']]
        if len(selected) >= 2:
            corrupt = d['mosaic'].copy()
            p = selected[0]
            corrupt[p] = d['bg'] if corrupt[p] != d['bg'] else d['palettes'][0][0]
            invalid_inputs.append(_pack(d['control'], corrupt, d['frame'], d['vertical']))
            break
    _require(len(invalid_inputs) == 3, 'missing contradictory-witness fixture')
    touching = d['control'].copy()
    touching[d['marker']] = d['bg']
    p = next(tuple(map(int, p)) for p in np.argwhere(~d['cut'])
             if any(max(abs(int(p[0]) - r), abs(int(p[1]) - c)) == 1 for r, c in d['path']))
    touching[p] = d['ink']
    invalid_inputs.append(_pack(touching, d['mosaic'], d['frame'], d['vertical']))
    for invalid in invalid_inputs:
        try:
            solve(invalid)
        except AssertionError:
            continue
        raise AssertionError('malformed or ambiguous input accepted')
    for name in ('branch', 'loop', 'three_regions'):
        cut = np.zeros((9, 9), dtype=bool)
        if name == 'branch':
            cut[:, 4] = True
            cut[4, 4:7] = True
        elif name == 'loop':
            cut[2:7, 2] = cut[2:7, 6] = True
            cut[2, 2:7] = cut[6, 2:7] = True
        else:
            cut[:, 2] = cut[:, 6] = True
        try:
            _cut_info(cut)
        except AssertionError:
            continue
        raise AssertionError('malformed cut accepted')
    return dict(seeds=42, pairs=pair_count, panel_shapes=len(shapes), window_shapes=len(windows),
                cut_structures=len(cuts), normalized_inputs=len(structures), layer_palettes=len(palettes),
                marker_only_twins=twins)
