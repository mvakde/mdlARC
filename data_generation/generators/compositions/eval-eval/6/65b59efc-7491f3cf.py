"""Paired identity stamps crossed by a lifted, one-pixel seam.

Only NumPy and Python's standard library are used. No parent code, source
examples, files, seed state, or training examples are consulted by solve().
"""
from __future__ import annotations

import numpy as np


def _need(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
          "grid dimensions must be 1..30")
    _need(np.issubdtype(a.dtype, np.integer) and not np.issubdtype(a.dtype, np.bool_),
          "grid cells must be integers")
    _need(np.all((a >= 0) & (a <= 9)), "colors must be 0..9")
    return a.astype(np.int8, copy=False)


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}")
    return value


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask, diagonal=False):
    directions = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                  if (dr or dc) and (diagonal or abs(dr) + abs(dc) == 1)]
    cells = set(map(tuple, np.argwhere(mask)))
    result = []
    while cells:
        first = min(cells)
        cells.remove(first)
        stack, group = [first], []
        while stack:
            r, c = stack.pop()
            group.append((int(r), int(c)))
            for dr, dc in directions:
                q = r + dr, c + dc
                if q in cells:
                    cells.remove(q)
                    stack.append(q)
        result.append(group)
    return result


def _mask(shape, cells):
    a = np.zeros(shape, dtype=bool)
    for p in cells:
        a[p] = True
    return a


def _path_geometry(cut):
    """Return canonical path order, two four-connected regions, marker sites."""
    cells = set(map(tuple, np.argwhere(cut)))
    _need(len(cells) >= 3, "cut too short")
    graph = {}
    for r, c in cells:
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                p = r + dr, c + dc
                if not (dr or dc) or p not in cells:
                    continue
                if dr and dc and ((r + dr, c) in cells or (r, c + dc) in cells):
                    continue
                neighbors.append(p)
        graph[r, c] = neighbors
    ends = sorted(p for p, ns in graph.items() if len(ns) == 1)
    _need(len(ends) == 2 and all(1 <= len(ns) <= 2 for ns in graph.values()),
          "cut must have two ends and no branch/loop")
    h, w = cut.shape
    border = lambda p: p[0] in (0, h - 1) or p[1] in (0, w - 1)
    _need(all(border(p) for p in ends) and
          all(not border(p) for p in cells if p not in ends),
          "only cut endpoints may meet a boundary")
    order, previous, current = [], None, ends[0]
    while current is not None and current not in order:
        order.append(tuple(map(int, current)))
        ns = [p for p in graph[current] if p != previous]
        previous, current = current, ns[0] if ns else None
    _need(len(order) == len(cells), "cut contains an extra component or loop")
    groups = _components(~cut)
    _need(len(groups) == 2, "cut must leave exactly two regions")
    regions = [_mask(cut.shape, group) for group in groups]
    near = cut.copy()
    for r, c in cells:
        near[max(0, r - 1):min(h, r + 2), max(0, c - 1):min(w, c + 2)] = True
    sites = [list(map(tuple, np.argwhere(region & ~near))) for region in regions]
    _need(all(sites), "both sides need a separated marker site")
    return order, regions, sites


def _lift(cut, s, marker):
    order, _, _ = _path_geometry(cut)
    q = s // 2
    shape = tuple(n * s for n in cut.shape)
    fine = np.zeros(shape, dtype=bool)
    centers = [(r * s + q, c * s + q) for r, c in order]
    for p, end in zip(centers, centers[1:]):
        dr = int(np.sign(end[0] - p[0]))
        dc = int(np.sign(end[1] - p[1]))
        steps = max(abs(end[0] - p[0]), abs(end[1] - p[1]))
        for t in range(steps + 1):
            fine[p[0] + t * dr, p[1] + t * dc] = True
    for end, neighbor in ((centers[0], centers[1]), (centers[-1], centers[-2])):
        dr = int(np.sign(end[0] - neighbor[0]))
        dc = int(np.sign(end[1] - neighbor[1]))
        r, c = end
        while 0 <= r < shape[0] and 0 <= c < shape[1]:
            fine[r, c] = True
            r, c = r + dr, c + dc
    _, regions, _ = _path_geometry(fine)
    point = marker[0] * s + q, marker[1] * s + q
    _need(not fine[point], "lifted marker lies on cut")
    side = 0 if regions[0][point] else 1
    return fine, regions[side], regions[1 - side]


def _prefix(row, color):
    i = 0
    while i < len(row) and row[i] == color:
        i += 1
    return i


def _decode(value):
    a = _grid(value)
    frame = int(a[0, 0])
    lw = _prefix(a[0], frame)
    _need(lw >= 7, "missing library frame")
    middle = next((r for r in range(1, a.shape[0])
                   if np.all(a[r, :lw] == frame)), None)
    _need(middle is not None, "missing library middle divider")
    s = middle - 2
    _need(s in (3, 5) and (lw - 1) % (s + 1) == 0, "invalid stamp scale")
    k = (lw - 1) // (s + 1)
    _need(2 <= k <= (6 if s == 3 else 4), "invalid identity count")
    lh = 2 * s + 5
    _need(a.shape[0] >= lh + 8, "missing control/map section")
    bg = int(a[lh, 0])
    _need(bg != frame and np.all(a[lh] == bg), "missing background gap")
    _need(np.all(a[:lh, lw:] == bg), "nonbackground outside library")
    _need(np.all(a[[0, s + 2, 2 * s + 4], :lw] == frame), "broken library frame")
    _need(np.all(a[:lh, :lw:s + 1] == frame), "broken library dividers")
    cards = []
    for face, top in enumerate((1, s + 3)):
        row = []
        for j in range(k):
            c0 = 1 + j * (s + 1)
            stamp = a[top:top + s, c0:c0 + s]
            values = np.unique(stamp[stamp != bg])
            _need(len(values) == 1, "stamp needs one nonempty identity")
            label = int(values[0])
            _need(label != frame and label not in [t[0] for t in row],
                  "stamp identities must be distinct and not frame")
            keyrow = a[top + s, c0:c0 + s]
            target = int(keyrow[s // 2])
            _need(target != bg, "key must be visible")
            expected = np.full(s, bg, dtype=np.int8)
            expected[s // 2] = target
            _need(np.array_equal(keyrow, expected), "invalid centered swatch")
            row.append((label, stamp == label, target))
        cards.append(row)
    labels = {t[0] for t in cards[0]}
    _need(labels == {t[0] for t in cards[1]}, "library faces have different identities")
    boardtop = lh + 1
    bw = _prefix(a[boardtop], frame)
    _need(bw >= 13 and (bw - 3) % 2 == 0, "invalid control/map frame width")
    w = (bw - 3) // 2
    bottom = next((r for r in range(boardtop + 1, a.shape[0])
                   if np.all(a[r, :bw] == frame)), None)
    _need(bottom is not None, "missing control/map bottom frame")
    h = bottom - boardtop - 1
    _need(5 <= h <= 30 // s and 5 <= w <= 30 // s, "invalid board dimensions")
    _need(np.all(a[boardtop:bottom + 1, [0, w + 1, 2 * w + 2]] == frame),
          "broken board divider")
    _need(np.all(a[boardtop:bottom + 1, bw:] == bg) and
          np.all(a[bottom + 1:] == bg), "invalid right/bottom margin")
    ctrl = a[boardtop + 1:bottom, 1:w + 1]
    placement = a[boardtop + 1:bottom, w + 2:2 * w + 2]
    colors = np.unique(ctrl[ctrl != bg])
    _need(len(colors) == 1, "control requires one cut/marker color")
    control = int(colors[0])
    _need(control not in {bg, frame, *labels}, "control role overlaps identity/frame")
    _need(set(map(int, np.unique(placement))) <= {bg, *labels}, "unknown map identity")
    _need(all(np.any(placement == label) for label in labels), "unused map identity")
    groups = sorted(_components(ctrl != bg, diagonal=True), key=len)
    _need(len(groups) == 2 and len(groups[0]) == 1 and len(groups[1]) >= 3,
          "control must have singleton marker and one separate cut")
    marker = groups[0][0]
    cut = _mask(ctrl.shape, groups[1])
    path, regions, sites = _path_geometry(cut)
    side = 0 if regions[0][marker] else 1
    _need(marker in sites[side], "marker touches cut")
    fine, marked, other = _lift(cut, s, marker)
    return dict(s=s, k=k, h=h, w=w, background=bg, frame=frame, control=control,
                cards=cards, placement=placement, cut=cut, marker=marker,
                path=path, regions=regions, sites=sites, side=side,
                fine=fine, marked=marked, other=other, boardtop=boardtop,
                margins=(a.shape[0] - bottom - 1, a.shape[1] - max(lw, bw)))


def _symmetry(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _plane(d, face, *, solid=False, raw=False, correspondence='identity', transform=0):
    bg, s = d['background'], d['s']
    out = np.full((d['h'] * s, d['w'] * s), bg, dtype=np.int8)
    for j, (label, silhouette, target) in enumerate(d['cards'][face]):
        if correspondence == 'position' and face == 1:
            label = d['cards'][0][j][0]
        elif correspondence == 'cyclic' and face == 1:
            upper = [t[0] for t in d['cards'][0]]
            shift = upper.index(d['cards'][1][0][0])
            label = upper[(shift + j) % len(upper)]
        silhouette = np.ones((s, s), dtype=bool) if solid else _symmetry(silhouette, transform)
        occupied = np.kron(d['placement'] == label, silhouette)
        out[occupied] = label if raw else target
    return out


def _render(d, alternative=None, transform=0):
    u = _plane(d, 0, solid=alternative == 'solid', raw=alternative == 'raw', transform=transform)
    l = _plane(d, 1, solid=alternative == 'solid', raw=alternative == 'raw',
               correspondence=alternative if alternative in ('position', 'cyclic') else 'identity',
               transform=transform)
    cut, marked, other, bg = d['fine'], d['marked'], d['other'], d['background']
    if alternative == 'upper':
        return u
    seam = np.where(u != bg, u, l)
    if alternative == 'union':
        return seam
    if alternative == 'lower_priority':
        seam = np.where(l != bg, l, u)
    if alternative == 'block_cut':
        expand = lambda a: np.repeat(np.repeat(a, d['s'], axis=0), d['s'], axis=1)
        cut = expand(d['cut'])
        marked = expand(d['regions'][d['side']])
        other = expand(d['regions'][1 - d['side']])
    return np.where(cut, seam, np.where(marked, u, l)).astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only pixel lookup; no generation or puzzle state is consulted."""
    d = _decode(input_grid)
    bg, s = d['background'], d['s']
    faces = [{label: (mask, target) for label, mask, target in row} for row in d['cards']]
    out = np.full((d['h'] * s, d['w'] * s), bg, dtype=np.int8)
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            label = int(d['placement'][r // s, c // s])
            if label == bg:
                continue
            um, uk = faces[0][label]
            lm, lk = faces[1][label]
            u = uk if um[r % s, c % s] else bg
            l = lk if lm[r % s, c % s] else bg
            if d['fine'][r, c]:
                out[r, c] = u if u != bg else l
            else:
                out[r, c] = u if d['marked'][r, c] else l
    return out


def _pack(s, cards, placement, cut, marker, bg, frame, control, margins=(0, 0)):
    h, w = placement.shape
    k = len(cards[0])
    lh, lw, bw = 2 * s + 5, k * (s + 1) + 1, 2 * w + 3
    bottom_margin, right_margin = margins
    a = np.full((lh + 1 + h + 2 + bottom_margin,
                 max(lw, bw) + right_margin), bg, dtype=np.int8)
    a[[0, s + 2, 2 * s + 4], :lw] = frame
    a[:lh, :lw:s + 1] = frame
    for face, top in enumerate((1, s + 3)):
        for j, (label, silhouette, target) in enumerate(cards[face]):
            c = 1 + j * (s + 1)
            a[top:top + s, c:c + s] = np.where(silhouette, label, bg)
            a[top + s, c + s // 2] = target
    r0 = lh + 1
    a[[r0, r0 + h + 1], :bw] = frame
    a[r0:r0 + h + 2, [0, w + 1, 2 * w + 2]] = frame
    ctrl = np.where(cut, control, bg).astype(np.int8)
    ctrl[marker] = control
    a[r0 + 1:r0 + h + 1, 1:w + 1] = ctrl
    a[r0 + 1:r0 + h + 1, w + 2:2 * w + 2] = placement
    return a


def _repack(d, **changes):
    args = {name: d[name] for name in ('s', 'cards', 'placement', 'cut', 'marker', 'frame', 'control', 'margins')}
    args['bg'] = d['background']
    args.update(changes)
    return _pack(**args)


def _cut(rng, h, w, family):
    for attempt in range(80):
        a = np.zeros((h, w), dtype=bool)
        if family == 'horizontal':
            a[int(rng.integers(2, h - 2)), :] = True
        elif family == 'vertical':
            a[:, int(rng.integers(2, w - 2))] = True
        elif family == 'diagonal':
            _need(h == w, "diagonal sampling uses squares")
            np.fill_diagonal(a, True)
        elif family == 'elbow':
            r, c = int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2))
            a[:r + 1, c] = True
            a[r, c:] = True
        elif family == 'stepped':
            switches = [int(rng.integers(2, h - 2))]
            if h >= 7 and rng.random() < .5:
                switches = [2, h - 3]
            col, last = int(rng.integers(1, w - 1)), 0
            for row in switches:
                a[last:row + 1, col] = True
                choices = [c for c in range(1, w - 1) if c != col]
                nxt = int(rng.choice(choices))
                a[row, min(col, nxt):max(col, nxt) + 1] = True
                col, last = nxt, row
            a[last:, col] = True
        else:
            raise ValueError("unknown cut family")
        if rng.random() < .5:
            a = np.fliplr(a).copy()
        if rng.random() < .5:
            a = np.flipud(a).copy()
        if family == 'stepped' and h == w and rng.random() < .5:
            a = a.T.copy()
        try:
            _path_geometry(a)
            return a
        except ValueError:
            continue
    # Constructive two-bend path keeps the requested stepped family.
    _need(family == 'stepped', "cut construction exhausted")
    a = np.zeros((h, w), dtype=bool)
    a[:h // 2 + 1, 1] = True
    a[h // 2, 1:w - 1] = True
    a[h // 2:, w - 2] = True
    _path_geometry(a)
    return a


def _stamp(rng, s, richness, witness=False):
    style = int(rng.integers(10))
    if witness and rng.random() < .65:
        style = 9
    rr, cc = np.indices((s, s))
    a = np.zeros((s, s), dtype=bool)
    if style == 0:
        a[:] = True
    elif style == 1:
        a[0] = a[-1] = True
        a[:, 0] = a[:, -1] = True
    elif style == 2:
        a[s // 2] = True
        a[:, s // 2] = True
    elif style == 3:
        a = np.eye(s, dtype=bool)
        if rng.random() < .5:
            a |= np.fliplr(a)
    elif style == 4:
        a = np.repeat((np.arange(s) % 2 == int(rng.integers(2)))[None, :], s, axis=0)
    elif style == 5:
        a = (rr + cc) % 2 == int(rng.integers(2))
    elif style == 6:
        r0, r1 = sorted(map(int, rng.choice(s, 2, replace=False)))
        c0, c1 = sorted(map(int, rng.choice(s, 2, replace=False)))
        a[r0:r1 + 1, c0:c1 + 1] = True
    elif style == 7:
        a[0, :int(rng.integers(1, s + 1))] = True
        a[-1, -int(rng.integers(1, s + 1)):] = True
    elif style == 8:
        a.flat[rng.choice(s * s, int(rng.integers(1, s + 1)), replace=False)] = True
    else:
        density = rng.uniform(.2, .8) if witness or richness else rng.uniform(.12, .65)
        a = rng.random((s, s)) < density
        if not a.any():
            a.flat[int(rng.integers(s * s))] = True
    return _symmetry(a, int(rng.integers(8)))


def _map(rng, h, w, labels, bg, witness=False):
    total, k = h * w, len(labels)
    density = float(rng.uniform(.65, 1)) if witness else float(rng.choice([0, .2, .4, .7, 1]))
    count = max(k, int(round(total * density)))
    coords = [(r, c) for r in range(h) for c in range(w)]
    style = int(rng.integers(6))
    if style == 0 or witness:
        positions = rng.permutation(total).tolist()
    elif style == 1:
        positions = [r * w + c for r in range(h)
                     for c in (range(w) if r % 2 == 0 else range(w - 1, -1, -1))]
    elif style == 2:
        positions = [r * w + c for r, c in sorted(coords, key=lambda p: (sum(p), p))]
    elif style == 3:
        cr, cc = int(rng.integers(h)), int(rng.integers(w))
        positions = [r * w + c for r, c in sorted(coords,
                     key=lambda p: (abs(p[0] - cr) + abs(p[1] - cc), p))]
    elif style == 4:
        positions = [r * w + c for r, c in sorted(coords,
                     key=lambda p: (min(p[0], p[1], h - 1 - p[0], w - 1 - p[1]), p))]
    else:
        positions = [r * w + c for r, c in sorted(coords, key=lambda p: (p[1] % 2, p))]
    counts = 1 + rng.multinomial(count - k, rng.dirichlet(np.full(k, .5)))
    commands = np.repeat(rng.permutation(labels), counts)
    if style == 0 or witness:
        rng.shuffle(commands)
    a = np.full((h, w), bg, dtype=np.int8)
    a.flat[np.array(positions[:count])] = commands
    if rng.random() < .5:
        a = a[:, ::-1].copy()
    if rng.random() < .5:
        a = a[::-1].copy()
    return a


_ALTERNATIVES = ('solid', 'upper', 'union', 'raw', 'position', 'cyclic', 'block_cut', 'lower_priority')
_FLAGS = ('marked_upper', 'marked_suppresses_lower', 'other_lower', 'other_suppresses_upper',
          'collision', 'lower_only_seam', 'upper_only_seam', 'blank_seam', 'split_tile')


def _evidence(d):
    u, l, bg = _plane(d, 0), _plane(d, 1), d['background']
    um, lm, cut, marked, other = u != bg, l != bg, d['fine'], d['marked'], d['other']
    vals = [np.any(marked & um), np.any(marked & ~um & lm),
            np.any(other & lm), np.any(other & um & ~lm),
            np.any(cut & um & lm & (u != l)), np.any(cut & ~um & lm),
            np.any(cut & um & ~lm), np.any(cut & ~um & ~lm)]
    split = False
    s = d['s']
    for r, c in np.argwhere(d['cut'] & (d['placement'] != bg)):
        sl = np.s_[r * s:(r + 1) * s, c * s:(c + 1) * s]
        split |= bool(np.any((marked & um)[sl]) and np.any((other & lm)[sl]))
    return {name for name, present in zip(_FLAGS, vals + [split]) if present}


def _noncyclic(d):
    upper = [t[0] for t in d['cards'][0]]
    order = [upper.index(t[0]) for t in d['cards'][1]]
    return any(p != (order[0] + i) % len(order) for i, p in enumerate(order))


def _is_witness(d):
    if _evidence(d) != set(_FLAGS) or not _noncyclic(d):
        return False
    out = _render(d)
    if not all(not np.array_equal(out, _render(d, alternative=a)) for a in _ALTERNATIVES):
        return False
    return all(not np.array_equal(out, _render(d, transform=t)) for t in range(1, 8))


def _new_pair(rng, params, s, h, w, k, family, side, witness=False):
    cut = _cut(rng, h, w, family)
    _, _, sites = _path_geometry(cut)
    marker = tuple(map(int, sites[side][int(rng.integers(len(sites[side])))]))
    palette = params['palette'] if params['fixed_palette'] or rng.random() < .25 else rng.permutation(10)
    bg, frame, control = map(int, palette[:3])
    labels = list(map(int, palette[3:3 + k]))
    silhouettes = [[_stamp(rng, s, params['richness'], witness) for _ in labels] for _ in range(2)]
    if not witness and rng.random() < .25:
        face = int(rng.integers(2))
        i, j = map(int, rng.choice(k, 2, replace=False))
        silhouettes[face][j] = silhouettes[face][i].copy()
    pool = [c for c in range(10) if c != bg]
    if witness:
        targets = None
        for _ in range(80):
            candidate = list(map(int, rng.choice(pool, 2 * k, replace=False)))
            if all(candidate[i] != labels[i % k] for i in range(2 * k)):
                targets = candidate
                break
        _need(targets is not None, "key construction exhausted")
    else:
        targets = list(map(int, rng.choice(pool, 2 * k)))
        if rng.random() < .3:
            targets[int(rng.integers(2 * k))] = labels[int(rng.integers(k))]
    upper_order = list(map(int, rng.permutation(k)))
    lower_order = list(map(int, rng.permutation(k)))
    if witness:
        lower_order = upper_order.copy()
        i, j = map(int, rng.choice(k, 2, replace=False))
        lower_order[i], lower_order[j] = lower_order[j], lower_order[i]
    cards = [[(labels[i], silhouettes[face][i], targets[face * k + i]) for i in order]
             for face, order in enumerate((upper_order, lower_order))]
    placement = _map(rng, h, w, labels, bg, witness)
    margins = tuple(map(int, rng.integers(0, 4, size=2))) if rng.random() < .5 else (0, 0)
    inp = _pack(s, cards, placement, cut, marker, bg, frame, control, margins)
    d = _decode(inp)
    out = _render(d)
    return {'input': inp, 'output': out}, d


def _dimensions(rng, s, regime, test, family):
    hi = 30 // s
    if regime == 'extent':
        if not test:
            return 5, 5
        if family == 'diagonal':
            n = int(rng.integers(6, hi + 1))
            return n, n
        dims = [int(rng.integers(5, hi + 1)), int(rng.integers(5, hi + 1))]
        dims[int(rng.integers(2))] = int(rng.integers(6, hi + 1))
        return tuple(dims)
    if family == 'diagonal':
        n = int(rng.integers(5, hi + 1))
        return n, n
    return tuple(map(int, rng.integers(5, hi + 1, size=2)))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt, nq = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    regime = str(rng.choice(['scale', 'extent', 'library', 'bends']))
    base_s = 3 if regime == 'scale' else int(rng.choice([3, 5]))
    params = dict(palette=rng.permutation(10), fixed_palette=bool(rng.random() < .25),
                  richness=bool(rng.integers(2)))
    first_side, twin = int(rng.integers(2)), bool(rng.random() < .55)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        test = split == 'test'
        for index in range(count):
            if not test and index == 1 and twin:
                d = _decode(puzzle['train'][0]['input'])
                sites = d['sites'][1 - d['side']]
                marker = tuple(map(int, sites[int(rng.integers(len(sites)))]))
                inp = _repack(d, marker=marker)
                pair = {'input': inp, 'output': _render(_decode(inp))}
            else:
                witness = not test and index == 0
                s = 5 if test and regime == 'scale' else base_s
                if regime == 'bends':
                    family = 'stepped' if test else ('elbow' if witness else str(rng.choice(['horizontal', 'vertical', 'elbow'])))
                else:
                    family = str(rng.choice(['horizontal', 'vertical', 'diagonal', 'elbow', 'stepped']))
                h, w = _dimensions(rng, s, regime, test, family)
                if witness:
                    k = 3
                elif regime == 'library':
                    k = int(rng.integers(4, (6 if s == 3 else 4) + 1)) if test else int(rng.integers(2, 4))
                else:
                    k = int(rng.integers(2, (6 if s == 3 else 4) + 1))
                side = first_side if not test and index == 0 else (1 - first_side if not test and index == 1 else int(rng.integers(2)))
                for attempt in range(240):
                    pair, d = _new_pair(rng, params, s, h, w, k, family, side, witness)
                    if _key(pair['input']) in seen or not np.any(pair['output'] != d['background']):
                        continue
                    if witness and not _is_witness(d):
                        continue
                    if not test and index == 1 and not np.any((_plane(d, 0) != _plane(d, 1)) & ~d['fine']):
                        continue
                    break
                else:
                    raise RuntimeError("bounded example sampling exhausted")
            key = _key(pair['input'])
            _need(key not in seen, "duplicate input")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "splits must be lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid pair counts")
    seen, records = set(), {'train': [], 'test': []}
    flags, disproved, symmetries, sides = set(), set(), set(), set()
    noncyclic = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            d = _decode(inp)
            _need(out.shape == (d['h'] * d['s'], d['w'] * d['s']), "incorrect expanded shape")
            _need(np.array_equal(out, solve(inp)), "incorrect solved pixels")
            _need(np.any(out != d['background']), "empty generated result")
            targets = {t[2] for row in d['cards'] for t in row}
            _need(set(map(int, np.unique(out))) <= {d['background'], *targets}, "output contains metadata color")
            records[split].append(d)
            if split == 'train':
                flags |= _evidence(d)
                disproved |= {a for a in _ALTERNATIVES if not np.array_equal(out, _render(d, alternative=a))}
                symmetries |= {t for t in range(1, 8) if not np.array_equal(out, _render(d, transform=t))}
                noncyclic |= _noncyclic(d) and not np.array_equal(out, _render(d, alternative='position'))
                if np.any((_plane(d, 0) != _plane(d, 1)) & ~d['fine']):
                    sides.add(d['side'])
    _need(flags == set(_FLAGS), "training lacks regional/seam/split-tile evidence")
    _need(disproved == set(_ALTERNATIVES) and symmetries == set(range(1, 8)) and noncyclic,
          "training permits an inherited-concept shortcut")
    _need(sides == {0, 1}, "training must effectively select both regions")
    tr, te = records['train'], records['test']
    scale_shift = {d['s'] for d in tr} == {3} and {d['s'] for d in te} == {5}
    extent_shift = all(d['h'] > max(x['h'] for x in tr) or d['w'] > max(x['w'] for x in tr) for d in te)
    library_shift = min(d['k'] for d in te) > max(d['k'] for d in tr)
    bend_shift = min(_bends(d) for d in te) > max(_bends(d) for d in tr)
    _need(scale_shift or extent_shift or library_shift or bend_shift, "missing observable train/test shift")


def _bends(d):
    steps = [(q[0] - p[0], q[1] - p[1]) for p, q in zip(d['path'], d['path'][1:])]
    return sum(p != q for p, q in zip(steps, steps[1:]))


def _family(d):
    rows, cols = zip(*d['path'])
    if len(set(rows)) == 1:
        return 'horizontal'
    if len(set(cols)) == 1:
        return 'vertical'
    if _bends(d) == 0:
        return 'diagonal'
    return 'elbow' if _bends(d) == 1 else 'stepped'


def _equal(p, q):
    return all(len(p[s]) == len(q[s]) and
               all(np.array_equal(a[n], b[n]) for a, b in zip(p[s], q[s]) for n in ('input', 'output'))
               for s in ('train', 'test'))


def _expect_rejected(call):
    try:
        call()
    except (ValueError, AssertionError):
        return
    raise AssertionError("malformed data was accepted")


def self_test():
    """Bounded consistency, intervention, and recovered diversity checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert _equal(puzzles[0], generate_puzzle(0))
    shapes, scales, counts, cuts, layouts, motifs, palettes, families = set(), set(), set(), set(), set(), set(), set(), set()
    densities, empty_seams, repeated_silhouettes, twin_count = [], 0, 0, 0
    for puzzle in puzzles:
        local_structures = set()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                d = _decode(pair['input'])
                assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
                shapes.add((pair['input'].shape, pair['output'].shape))
                scales.add(d['s'])
                counts.add(d['k'])
                families.add(_family(d))
                cuts.add(_key(d['cut']))
                palettes.add((d['background'], d['frame'], d['control']))
                upper = [t[0] for t in d['cards'][0]]
                normalized = np.full(d['placement'].shape, -1, dtype=np.int8)
                for i, label in enumerate(upper):
                    normalized[d['placement'] == label] = i
                layouts.add(_key(normalized))
                local_structures.add((_key(d['cut']), _key(normalized), d['marker']))
                density = float(np.mean(d['placement'] != d['background']))
                densities.append(density)
                for row in d['cards']:
                    motifs.update(_key(t[1]) for t in row)
                    repeated_silhouettes += len({_key(t[1]) for t in row}) < d['k']
                empty_seams += not np.any(d['fine'] & ((_plane(d, 0) != d['background']) | (_plane(d, 1) != d['background'])))
        assert len(local_structures) >= 3
        first, second = puzzle['train'][:2]
        if first['input'].shape == second['input'].shape:
            twin_count += np.count_nonzero(first['input'] != second['input']) == 2
        d = _decode(first['input'])
        marker = tuple(map(int, d['sites'][1 - d['side']][0]))
        moved = _repack(d, marker=marker)
        swapped = solve(moved)
        assert not np.array_equal(swapped, first['output'])
        assert np.array_equal(swapped[d['fine']], first['output'][d['fine']])
        assert np.array_equal(swapped[d['marked']], _plane(d, 1)[d['marked']])
        assert np.array_equal(swapped[d['other']], _plane(d, 0)[d['other']])
        # Move complete lower cards and their keys: no column correspondence.
        cards = [d['cards'][0], d['cards'][1][::-1]]
        assert np.array_equal(solve(_repack(d, cards=cards)), first['output'])
        # Local scene reflection preserves row roles and output axes.
        for axis in (0, 1):
            cards = [[(lab, np.flip(mask, axis), key) for lab, mask, key in row] for row in d['cards']]
            marker = (d['h'] - 1 - d['marker'][0], d['marker'][1]) if axis == 0 else (d['marker'][0], d['w'] - 1 - d['marker'][1])
            mirrored = _repack(d, cards=cards, placement=np.flip(d['placement'], axis),
                               cut=np.flip(d['cut'], axis), marker=marker)
            assert np.array_equal(solve(mirrored), np.flip(first['output'], axis))
        rename = np.array([7, 0, 8, 3, 1, 9, 5, 2, 6, 4], dtype=np.int8)
        assert np.array_equal(solve(rename[first['input']]), rename[first['output']])
    assert scales == {3, 5} and counts >= {2, 3, 4, 5, 6}
    assert families == {'horizontal', 'vertical', 'diagonal', 'elbow', 'stepped'}
    assert len(shapes) >= 35 and len(cuts) >= 35 and len(layouts) >= 150 and len(motifs) >= 100
    assert len(palettes) >= 30
    assert min(densities) <= .12 and max(densities) == 1
    assert repeated_silhouettes >= 15 and empty_seams >= 5 and twin_count >= 8
    for i, (nt, nq) in enumerate(((2, 1), (3, 2), (5, 4), (10, 1), (10, 4))):
        p = generate_puzzle(700 + i, nt, nq)
        assert len(p['train']) == nt and len(p['test']) == nq
    assert _equal(generate_puzzle(91, 7, 3), generate_puzzle(91, 7, 3))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        _expect_rejected(lambda nt=nt, nq=nq: generate_puzzle(0, nt, nq))
    for seed in (-1, True, 2.5):
        _expect_rejected(lambda seed=seed: generate_puzzle(seed))
    bad = {s: [{n: a.copy() for n, a in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    _expect_rejected(lambda: validate_generated_puzzle(bad))
    d = _decode(puzzles[0]['train'][0]['input'])
    cut_only = puzzles[0]['train'][0]['input'].copy()
    cut_only[d['boardtop'] + 1 + d['marker'][0], 1 + d['marker'][1]] = d['background']
    _expect_rejected(lambda: solve(cut_only))
    # Fixed geometric regressions directly establish the centerline convention.
    cut = np.zeros((5, 5), dtype=bool)
    cut[2, :] = True
    for s in (3, 5):
        fine, marked, other = _lift(cut, s, (0, 0))
        expected = np.zeros((5 * s, 5 * s), dtype=bool)
        expected[2 * s + s // 2, :] = True
        assert np.array_equal(fine, expected)
        assert marked[:2 * s + s // 2].all() and other[2 * s + s // 2 + 1:].all()
    for malformed in ('branch', 'loop', 'three_regions'):
        cut = np.zeros((7, 7), dtype=bool)
        if malformed == 'branch':
            cut[:, 3] = True
            cut[3, 3:6] = True
        elif malformed == 'loop':
            cut[2:5, 2] = cut[2:5, 4] = True
            cut[2, 2:5] = cut[4, 2:5] = True
        else:
            cut[:, 2] = cut[:, 4] = True
        _expect_rejected(lambda cut=cut: _path_geometry(cut))


if __name__ == '__main__':
    self_test()
    print('self_test passed')
