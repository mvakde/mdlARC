"""Perforated shutters: material-attached defects in an occluding continuation.

Only NumPy and the standard library are used. Labels are drawn from sampled
geometry; solve fits the observed panels and exact covers of visible defects.
No parent generator, files, seed, or demonstration outputs are used by solve.
"""
from itertools import product

import numpy as np


HOST_SEEDS = (0, 848443378, 722943271, 721264461, 530658820, 149380097,
              373203974, 47914563, 894571295, 938187442, 480594612, 388699046,
              879652542, 611064083, 918203447, 783920389, 176721838, 564466719,
              918193154, 230152869, 538653110, 241901661, 403246198, 346509179,
              326105014, 103541074, 230415543, 148868820, 530045727, 944851238,
              953377118, 470010977, 139493613, 708544910, 488888073, 828376195,
              842297677, 801676770, 482405099, 446103251, 707879398, 105060495)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(name + ' is out of range')
    return value


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), 'invalid dimensions'
    assert np.issubdtype(a.dtype, np.integer), 'noninteger grid'
    assert np.all((0 <= a) & (a <= 9)), 'invalid color'
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a)
    return a.shape, tuple(map(int, a.flat))


def _corners(bb):
    r0, r1, c0, c1 = bb
    return {(r0, c0), (r0, c0 + 1), (r0 + 1, c0),
            (r1, c1), (r1, c1 - 1), (r1 - 1, c1)}


def _find_query(a):
    found = []
    for raw in np.unique(a):
        positions = np.argwhere(a == raw)
        if len(positions) != 6:
            continue
        r0, c0 = map(int, positions.min(axis=0))
        r1, c1 = map(int, positions.max(axis=0))
        bb = (r0, r1, c0, c1)
        if r1 <= r0 or c1 <= c0 or r1 - r0 + c1 - c0 < 4:
            continue
        if set(map(tuple, positions)) == _corners(bb):
            found.append((int(raw), bb))
    assert len(found) == 1, 'query must be two disconnected canonical Ls'
    return found[0]


def _linear(values):
    values = tuple(map(int, values))
    assert len(values) >= 3
    d = values[1] - values[0]
    assert all(v == values[0] + i * d for i, v in enumerate(values)), 'nonarithmetic geometry'
    return values[0], d


def _geom(m, i):
    y = m['y'][0] + i * m['y'][1]
    h = m['h'][0] + i * m['h'][1]
    v = m['v'][0] + i * m['v'][1]
    t = m['t'][0] + i * m['t'][1]
    x = m['x'] + i * m['step']
    assert 7 <= h <= 30 and t >= 3 and 1 <= v and v + t < h, 'undefined panel'
    return y, x, h, v, t


def _site(m, template, i):
    layer, anchor, offset, c = template
    y, x, h, v, t = _geom(m, i)
    length, top = (h, 0) if layer == 0 else (t, v)
    r = top + (offset if anchor == 0 else length - 1 - offset)
    assert 0 <= offset < length and 0 <= c < m['w']
    if layer:
        assert 1 <= c < m['w'] - 1
    hidden = layer == 0 and 1 <= c < m['w'] - 1 and v <= r < v + t
    return y + r, x + c, hidden


def _fit_patch(p, bg):
    """Fit a solid rear rectangle plus its nested front; bg cells are erasures.

    The rim has no ordinary noise, but can have holes. It establishes the rear
    fill even when the front occupies most of the panel. All front intervals
    are scored against surviving cells, not a supplied output.
    """
    h, w = p.shape
    assert 7 <= h <= 30 and w in (5, 6)
    rim = np.concatenate((p[0], p[-1], p[1:-1, 0], p[1:-1, -1]))
    rear = set(map(int, rim[rim != bg]))
    assert len(rear) == 1, 'rim must identify one rear fill'
    rear = next(iter(rear))
    other = set(map(int, np.unique(p))) - {bg, rear}
    assert len(other) == 1
    front = next(iter(other))
    known = p != bg
    costs = []
    for v in range(1, h - 3):
        for t in range(3, h - v):
            latent = np.full_like(p, rear)
            latent[v:v + t, 1:w - 1] = front
            cost = int(np.count_nonzero(known & (p != latent)))
            costs.append((cost, v, t, latent))
    costs.sort(key=lambda q: q[:3])
    assert costs and costs[0][0] <= 2
    assert len(costs) == 1 or costs[1][0] > costs[0][0], 'ambiguous inner rectangle'
    _, v, t, latent = costs[0]
    return rear, front, v, t, latent, costs[0][0]


def _fit_axis(a, bb, bg):
    r0, r1, c0, c1 = bb
    columns = np.flatnonzero(np.any(a != bg, axis=0))
    assert len(columns)
    groups = np.split(columns, np.flatnonzero(np.diff(columns) > 1) + 1)
    assert 3 <= len(groups) <= 4
    assert columns[-1] < c0 or columns[0] > c1, 'query must be beyond evidence'
    patches, starts, tops, heights, widths, vs, ts, colors, scenes = [], [], [], [], [], [], [], [], []
    wrong = 0
    for cols in groups:
        rows = np.flatnonzero(np.any(a[:, cols] != bg, axis=1))
        p = a[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        rear, front, v, t, scene, damage = _fit_patch(p, bg)
        patches.append(p)
        starts.append(int(cols[0])); tops.append(int(rows[0]))
        heights.append(p.shape[0]); widths.append(p.shape[1])
        vs.append(v); ts.append(t); colors.append((rear, front)); scenes.append(scene)
        wrong += damage
    assert len(set(widths)) == 1
    x, step = _linear(starts)
    assert 1 <= step - widths[0] <= 2
    palette = colors[0]
    assert all(cs == (palette[i % 2], palette[1 - i % 2]) for i, cs in enumerate(colors))
    m = dict(x=x, step=step, w=widths[0], y=_linear(tops), h=_linear(heights),
             v=_linear(vs), t=_linear(ts), palette=palette, n=len(patches))
    holes = set()
    for i, p in enumerate(patches):
        for r, c in np.argwhere(p == bg):
            holes.add((i, int(r), int(c)))
    assert holes, 'no material perforations'
    # Include the one-panel halo on both sides of the query. Validate every
    # intermediate panel as well; a skipped panel cannot have undefined sizes.
    indices = []
    for i in range(-10, 15):
        xx = x + i * step
        if xx - 1 <= c1 and xx + m['w'] >= c0:
            indices.append(i)
    assert indices
    lo, hi = min(0, min(indices)), max(m['n'] - 1, max(indices))
    for i in range(lo, hi + 1):
        _geom(m, i)
    return m, patches, scenes, holes, indices, wrong


def _template_covers(m, scenes, holes):
    candidates = {}
    for i, r, c in sorted(holes):
        layer = int(scenes[i][r, c] == m['palette'][1 - i % 2])
        _, _, h, v, t = _geom(m, i)
        length, local = (h, r) if layer == 0 else (t, r - v)
        for anchor in (0, 1):
            offset = local if anchor == 0 else length - 1 - local
            template = (layer, anchor, offset, c)
            witnessed = set()
            good = True
            for j in range(m['n']):
                try:
                    rr, cc, hidden = _site(m, template, j)
                except AssertionError:
                    good = False; break
                yy, xx, _, _, _ = _geom(m, j)
                point = (j, rr - yy, cc - xx)
                if not hidden:
                    if point not in holes:
                        good = False; break
                    witnessed.add(point)
            if good and len(witnessed) >= 2:
                candidates[template] = frozenset(witnessed)
    assert candidates, 'defects need at least two visible witnesses'
    items = sorted(candidates.items())
    by_point = {p: [k for k, (_, seen) in enumerate(items) if p in seen] for p in holes}
    assert all(by_point.values()), 'unexplained erasure'
    covers = []

    def visit(remaining, chosen):
        assert len(covers) < 128, 'excessive ambiguous defect covers'
        if not remaining:
            covers.append(tuple(items[k][0] for k in chosen))
            return
        p = min(remaining, key=lambda q: len([k for k in by_point[q] if items[k][1] <= remaining]))
        for k in by_point[p]:
            seen = items[k][1]
            if seen <= remaining:
                visit(remaining - seen, chosen + [k])

    visit(frozenset(holes), [])
    assert covers, 'no consistent material correspondence'
    return covers


def _predict(m, templates, indices, bb, bg, ink, *, diagnostics=True, gate=True):
    """Input-side rendering uses coordinate predicates and simultaneous masks."""
    r0, r1, c0, c1 = bb
    rr, cc = np.indices((r1 - r0 + 1, c1 - c0 + 1))
    rr += r0; cc += c0
    out = np.full(rr.shape, bg, dtype=np.int8)
    demands = {c: np.zeros(rr.shape, dtype=bool) for c in m['palette']}
    center_mask = np.zeros(rr.shape, dtype=bool)
    centers = []
    for i in indices:
        y, x, h, v, t = _geom(m, i)
        rear, front = m['palette'][i % 2], m['palette'][1 - i % 2]
        inside = (y <= rr) & (rr < y + h) & (x <= cc) & (cc < x + m['w'])
        shutter = inside & (y + v <= rr) & (rr < y + v + t) & (x + 1 <= cc) & (cc < x + m['w'] - 1)
        out[inside] = rear; out[shutter] = front
        for template in templates:
            r, c, hidden = _site(m, template, i)
            if gate and hidden:
                continue
            fill = front if template[0] else rear
            centers.append((r, c, fill))
    if diagnostics:
        for r, c, fill in centers:
            center = (rr == r) & (cc == c)
            ring = (np.abs(rr - r) <= 1) & (np.abs(cc - c) <= 1) & ~center
            center_mask |= center
            demands[next(color for color in m['palette'] if color != fill)] |= ring
        a, b = m['palette']
        assert not np.any(demands[a] & demands[b] & ~center_mask), 'incompatible rings'
        for color, mask in demands.items():
            out[mask] = color
        out[center_mask] = ink
    return out, centers


def _analyze(input_grid):
    a = _grid(input_grid)
    ink, bb = _find_query(a)
    values, counts = np.unique(a, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1
    bg = int(values[counts.argmax()])
    assert bg != ink and len(values) == 4
    evidence = a.copy(); evidence[evidence == ink] = bg
    assert np.all(evidence[bb[0]:bb[1] + 1, bb[2]:bb[3] + 1] == bg)
    answers = []
    for transposed in (False, True):
        canonical = evidence.T if transposed else evidence
        bounds = (bb[2], bb[3], bb[0], bb[1]) if transposed else bb
        try:
            m, patches, scenes, holes, indices, wrong = _fit_axis(canonical, bounds, bg)
            covers = _template_covers(m, scenes, holes)
        except AssertionError:
            continue
        for templates in covers:
            try:
                # Attachments must remain defined through skipped panels, even
                # when the particular site is occluded in one of those panels.
                for i in range(min(0, min(indices)), max(m['n'] - 1, max(indices)) + 1):
                    for template in templates:
                        _site(m, template, i)
                answer, centers = _predict(m, templates, indices, bounds, bg, ink)
            except AssertionError:
                continue
            pred = answer.T.copy() if transposed else answer
            answers.append((pred, m, templates, indices, bounds, transposed, holes, wrong, centers))
    assert answers, 'no supported shutter continuation'
    expected = answers[0][0]
    assert all(np.array_equal(expected, item[0]) for item in answers), 'ambiguous predicted crop'
    item = answers[0]
    return expected, dict(model=item[1], templates=item[2], indices=item[3], bounds=item[4],
                          transposed=item[5], holes=item[6], wrong=item[7], centers=item[8],
                          background=bg, ink=ink, original_bounds=bb, alternatives=len(answers))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every output cell solely from the supplied input."""
    return _analyze(input_grid)[0]


def _latent_fields(m, templates, count, bg, ink, shape):
    """Label path: explicit rectangle slices, then constraint-map ring strokes."""
    clean = np.full(shape, bg, dtype=np.int8)
    damaged = clean.copy()
    centers = []
    for i in range(count):
        y, x, h, v, t = _geom(m, i)
        rear, front = m['palette'][i % 2], m['palette'][1 - i % 2]
        clean[y:y + h, x:x + m['w']] = rear
        clean[y + v:y + v + t, x + 1:x + m['w'] - 1] = front
        for layer, anchor, offset, column in templates:
            length = t if layer else h
            local_row = offset if anchor == 0 else length - offset - 1
            row = y + (v if layer else 0) + local_row
            col = x + column
            if not layer and 1 <= column < m['w'] - 1 and y + v <= row < y + v + t:
                continue
            assert 0 <= local_row < length
            centers.append((row, col, front if layer else rear))
    damaged[:] = clean
    for r, c, _ in centers:
        damaged[r, c] = bg
    center_set = {(r, c) for r, c, _ in centers}
    demands = {}
    for r, c, fill in centers:
        other = next(value for value in m['palette'] if value != fill)
        for dr, dc in product((-1, 0, 1), repeat=2):
            q = r + dr, c + dc
            if q in center_set or not (0 <= q[0] < shape[0] and 0 <= q[1] < shape[1]):
                continue
            assert q not in demands or demands[q] == other, 'conflicting latent rings'
            demands[q] = other
    diagnostic = clean.copy()
    for q, color in demands.items():
        diagnostic[q] = color
    for r, c in center_set:
        diagnostic[r, c] = ink
    return clean, damaged, diagnostic, centers


def _make_pair(rng, bg, ink, palette, kind, test, shift, axis):
    n = 3 if kind or test else int(rng.choice((3, 3, 4)))
    skip = int(test and shift in (1, 3))
    span = 2 if test and shift == 2 and rng.random() < .55 else 1
    w = 5 if n + skip + span >= 5 else int(rng.choice((5, 6)))
    gap = 1 if n + skip + span >= 5 else int(rng.choice((1, 1, 2)))
    total = n + skip + span
    extent = total * (w + gap) - gap
    assert extent <= 30
    if kind == 1:
        h0, dh = int(rng.integers(7, 10)), int(rng.choice((1, 2)))
        v0, dv, t0, dt = 2, 1, 3, 0
        templates = ((0, 0, 2, 1), (1, 1, 0, w - 3), (1, 1, 0, w - 2))
    elif kind == 2:
        h0, dh = int(rng.integers(11, 14)), int(rng.choice((-1, 1, 2)))
        v0, dv, t0, dt = 5, -1, 5, 0
        templates = ((0, 0, 3, 1), (0, 0, 1, 0), (1, 1, 0, w - 2))
    else:
        dh = int(rng.integers(-2, 3)); dt = int(rng.integers(-1, 2)); dv = int(rng.integers(-1, 2))
        h0 = int(rng.integers(7, 13)) + max(0, -dh * (total - 1))
        t0 = int(rng.integers(3, 6)) + max(0, -dt * (total - 1))
        v0 = int(rng.integers(1, 5)) + max(0, -dv * (total - 1))
        templates = None
    drift = int(rng.integers(-1, 2))
    tops = [i * drift for i in range(total)]
    heights = [h0 + i * dh for i in range(total)]
    ymin, ymax = min(tops), max(y + h for y, h in zip(tops, heights))
    assert ymax - ymin <= 28 and min(heights) >= 7
    pad_top = int(rng.integers(0, min(2, 30 - (ymax - ymin)) + 1))
    pad_bottom = int(rng.integers(0, min(2, 30 - (ymax - ymin) - pad_top) + 1))
    pad_left = int(rng.integers(0, min(2, 30 - extent) + 1))
    pad_right = int(rng.integers(0, min(2, 30 - extent - pad_left) + 1))
    m = dict(x=pad_left, step=w + gap, w=w, y=(pad_top - ymin, drift),
             h=(h0, dh), v=(v0, dv), t=(t0, dt), palette=palette, n=n)
    for i in range(total):
        _geom(m, i)
    if templates is None:
        rear_sites, front_sites = [], []
        for layer in (0, 1):
            length = min(heights) if not layer else min(t0 + i * dt for i in range(total))
            for anchor in (0, 1):
                for offset in range(length):
                    for c in range(w) if not layer else range(1, w - 1):
                        site = (layer, anchor, offset, c)
                        visible = sum(not _site(m, site, i)[2] for i in range(n))
                        if visible >= 2:
                            (front_sites if layer else rear_sites).append(site)
        assert rear_sites and front_sites
        templates = [rear_sites[int(rng.integers(len(rear_sites)))],
                     front_sites[int(rng.integers(len(front_sites)))]]
        if rng.random() < .45:
            layer, anchor, offset, c = templates[int(rng.integers(2))]
            candidates = [(layer, anchor, offset, c + d) for d in (-1, 1)
                          if 0 <= c + d < w and (not layer or 1 <= c + d < w - 1)]
            if candidates:
                templates.append(candidates[int(rng.integers(len(candidates)))])
        templates = tuple(templates)
    shape = (ymax - ymin + pad_top + pad_bottom, extent + pad_left + pad_right)
    clean, damaged, diagnostic, centers = _latent_fields(m, templates, total, bg, ink, shape)
    a = np.full(shape, bg, dtype=np.int8)
    for i in range(n):
        y, x, h, v, t = _geom(m, i)
        a[y:y + h, x:x + w] = damaged[y:y + h, x:x + w]
        noise = 1 if kind == 1 else int(rng.choice((0, 1, 1, 2)))
        candidates = [(r, c) for r in range(y + 1, y + h - 1) for c in range(x + 1, x + w - 1)
                      if a[r, c] != bg]
        rng.shuffle(candidates)
        for r, c in candidates[:noise]:
            a[r, c] = next(value for value in palette if value != clean[r, c])
    targets = list(range(n + skip, total))
    r0 = min(_geom(m, i)[0] for i in targets)
    r1 = max(_geom(m, i)[0] + _geom(m, i)[2] - 1 for i in targets)
    c0 = m['x'] + targets[0] * m['step']
    c1 = m['x'] + targets[-1] * m['step'] + w - 1
    # Optional margins are actual query geometry, not obligatory blank fringes.
    if rng.random() < .35:
        r0 = max(0, r0 - 1); r1 = min(shape[0] - 1, r1 + 1)
    if rng.random() < .35:
        c0 = max(c0 - 1, m['x'] + (n - 1) * m['step'] + w)
        c1 = min(shape[1] - 1, c1 + 1)
    if kind == 2:
        # The rear edge center is outside the crop, but its ring enters it.
        # This teaches painting before cropping even for two-example puzzles.
        c0 = max(c0, m['x'] + targets[0] * m['step'] + 1)
    if (test and shift in (2, 3)) or (not kind and not test and rng.random() < .25):
        r0 += int(rng.integers(0, min(4, r1 - r0 - 2) + 1))
        r1 -= int(rng.integers(0, min(4, r1 - r0 - 2) + 1))
        c0 += int(rng.integers(1, min(3, c1 - c0) + 1)) if test else int(rng.integers(2))
        if test:
            c0 = max(c0, m['x'] + targets[0] * m['step'] + 2)
        c1 -= int(rng.integers(0, min(2, c1 - c0 - 1) + 1))
    assert r1 > r0 and c1 > c0 and r1 - r0 + c1 - c0 >= 4
    answer = diagnostic[r0:r1 + 1, c0:c1 + 1].copy()
    assert np.any(answer == ink), 'crop must expose a diagnostic center'
    bb = (r0, r1, c0, c1)
    if rng.integers(2):
        a = a[::-1].copy(); answer = answer[::-1].copy()
        bb = (shape[0] - 1 - r1, shape[0] - 1 - r0, c0, c1)
    if rng.integers(2):
        a = a[:, ::-1].copy(); answer = answer[:, ::-1].copy()
        bb = (bb[0], bb[1], shape[1] - 1 - bb[3], shape[1] - 1 - bb[2])
    transpose = (1 - axis if test and shift in (0, 3) else axis)
    if transpose:
        a = a.T.copy(); answer = answer.T.copy()
        bb = (bb[2], bb[3], bb[0], bb[1])
    for r, c in _corners(bb):
        a[r, c] = ink
    predicted, info = _analyze(a)
    assert np.array_equal(answer, predicted), 'latent and input-fit paths disagree'
    if kind == 1:
        assert info['wrong'] > 0
        assert len({fill for _, _, fill in info['centers']}) == 2
    if kind == 2:
        ungated, _ = _predict(info['model'], info['templates'], info['indices'], info['bounds'], bg, ink, gate=False)
        if info['transposed']:
            ungated = ungated.T
        assert not np.array_equal(answer, ungated), 'concealing demo must distinguish occlusion'
    return {'input': a, 'output': answer}


def _fallback_pair(bg, ink, palette, kind, test, shift, axis, variant):
    """Sixteen explicit legal geometries per role; no random search or labels.

    All sites have sufficient witnesses and all ring demands are compatible.
    Height and top margin make this pool larger than the maximum 13 preceding
    inputs, so duplicate avoidance cannot exhaust it.
    """
    skip = int(test and shift in (1, 3))
    total = 4 + skip
    concealing = kind == 2
    h0 = (11 if concealing else 7) + variant // 4
    top = variant % 4
    m = dict(x=0, step=6, w=5, y=(top, 0), h=(h0, 1),
             v=(5, -1) if concealing else (2, 1),
             t=(5, 0) if concealing else (3, 0), palette=palette, n=3)
    templates = (((0, 0, 3, 1), (0, 0, 1, 0), (1, 1, 0, 3)) if concealing else
                 ((0, 0, 2, 1), (1, 1, 0, 2), (1, 1, 0, 3)))
    shape = (top + h0 + total, total * 6)
    clean, damaged, diagnostic, _ = _latent_fields(m, templates, total, bg, ink, shape)
    inp = np.full(shape, bg, dtype=np.int8)
    for i in range(3):
        y, x, h, _, _ = _geom(m, i)
        inp[y:y + h, x:x + 5] = damaged[y:y + h, x:x + 5]
        if kind == 1:
            inp[y + 1, x + 3] = next(c for c in palette if c != clean[y + 1, x + 3])
    i = total - 1
    y, x, h, _, _ = _geom(m, i)
    cut = 2 if test and shift in (2, 3) else int(concealing)
    bb = (y, y + h - 1, x + cut, x + 4)
    out = diagnostic[bb[0]:bb[1] + 1, bb[2]:bb[3] + 1].copy()
    if (1 - axis if test and shift in (0, 3) else axis):
        inp, out = inp.T.copy(), out.T.copy()
        bb = (bb[2], bb[3], bb[0], bb[1])
    for r, c in _corners(bb):
        inp[r, c] = ink
    assert np.array_equal(solve(inp), out)
    return {'input': inp, 'output': out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    num_train = _integer(num_train, 'num_train', 2, 10)
    num_test = _integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    bg, ink, a, b = map(int, rng.choice(10, 4, replace=False))
    stable_palette = bool(rng.integers(2))
    shift = int(rng.integers(4)); axis = int(rng.integers(2))
    puzzle = {'train': [], 'test': []}; seen = set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            kind = index + 1 if split == 'train' and index < 2 else 0
            for _ in range(240):
                palette = (a, b) if stable_palette else tuple(map(int, rng.choice([c for c in range(10) if c not in (bg, ink)], 2, replace=False)))
                try:
                    pair = _make_pair(rng, bg, ink, palette, kind, split == 'test', shift, axis)
                except AssertionError:
                    continue
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key); puzzle[split].append(pair)
                    break
            else:
                for variant in range(16):
                    pair = _fallback_pair(bg, ink, (a, b), kind, split == 'test', shift, axis, variant)
                    key = _key(pair['input'])
                    if key not in seen:
                        seen.add(key); puzzle[split].append(pair)
                        break
                else:
                    raise AssertionError('distinct fallback pool exhausted')
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, roles, analyses = set(), set(), []
    for pair in puzzle['train'] + puzzle['test']:
        assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
        inp, out = _grid(pair['input']), _grid(pair['output'])
        assert _key(inp) not in seen; seen.add(_key(inp))
        expected, info = _analyze(inp)
        assert np.array_equal(out, expected), 'incorrect output'
        bb = info['original_bounds']
        assert out.shape == (bb[1] - bb[0] + 1, bb[3] - bb[2] + 1)
        assert np.any(out == info['ink'])
        assert set(map(int, np.unique(out))) <= set(map(int, np.unique(inp)))
        roles.add((info['background'], info['ink']))
        clean, _ = _predict(info['model'], info['templates'], info['indices'], info['bounds'], info['background'], info['ink'], diagnostics=False)
        if info['transposed']:
            clean = clean.T
        assert not np.array_equal(out, clean), 'B contribution is inert'
        analyses.append(info)
    assert len(roles) == 1, 'shared background and diagnostic ink convention'
    training = analyses[:len(puzzle['train'])]
    assert any(len({fill for _, _, fill in i['centers']}) == 2 for i in training)
    assert any(i['wrong'] > 0 for i in training)
    assert any(i['model']['h'][1] != 0 for i in training)
    assert any(any(fill == other and max(abs(r - rr), abs(c - cc)) == 1
                   for j, (r, c, fill) in enumerate(i['centers'])
                   for rr, cc, other in i['centers'][:j]) for i in training)
    # Check an explicit alternative: transporting rear defects through a front
    # layer must change at least one training answer.
    occlusion_matters = False
    for pair, info in zip(puzzle['train'], training):
        try:
            ungated, _ = _predict(info['model'], info['templates'], info['indices'], info['bounds'], info['background'], info['ink'], gate=False)
            if info['transposed']:
                ungated = ungated.T
            occlusion_matters |= not np.array_equal(pair['output'], ungated)
        except AssertionError:
            occlusion_matters = True
    assert occlusion_matters, 'no demonstration of defect occlusion'
    assert any(any(not (i['bounds'][0] <= r <= i['bounds'][1] and i['bounds'][2] <= c <= i['bounds'][3])
                       and i['bounds'][0] - 1 <= r <= i['bounds'][1] + 1
                       and i['bounds'][2] - 1 <= c <= i['bounds'][3] + 1 for r, c, _ in i['centers'])
               for i in training), 'no training halo from an outside center'
    # A constant-copy alternative retains B's diagnosis and color cycling but
    # ignores A's changing geometry. It must disagree on a training answer.
    growth_matters = False
    for pair, info in zip(puzzle['train'], training):
        m = info['model']
        nearest = 0 if info['bounds'][3] < m['x'] else m['n'] - 1
        y, _, h, v, t = _geom(m, nearest)
        fixed = dict(m, y=(y, 0), h=(h, 0), v=(v, 0), t=(t, 0))
        try:
            copied, _ = _predict(fixed, info['templates'], info['indices'], info['bounds'], info['background'], info['ink'])
            if info['transposed']:
                copied = copied.T
            growth_matters |= not np.array_equal(pair['output'], copied)
        except AssertionError:
            growth_matters = True
    assert growth_matters, 'constant repetition matches all demonstrations'


def self_test():
    """Bounded structural and semantic probes on all 42 published host seeds."""
    signatures, shapes, growth, shutter_motion, widths, counts = set(), set(), set(), set(), set(), set()
    directions, backgrounds, inks, crops, templates_seen = set(), set(), set(), set(), set()
    hidden, halo, clusters, partial, skipped, normalized = 0, 0, 0, 0, 0, set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple((_key(p['input']), _key(p['output'])) for p in puzzle['train'] + puzzle['test']))
        # Restoring ordinary wrong-fill damage changes the input evidence but
        # must leave the output unchanged. The query Ls and erasures stay put.
        pair = puzzle['train'][0]
        _, first = _analyze(pair['input'])
        canonical = pair['input'].T.copy() if first['transposed'] else pair['input'].copy()
        scene, _ = _predict(first['model'], (), range(first['model']['n']),
                            (0, canonical.shape[0] - 1, 0, canonical.shape[1] - 1),
                            first['background'], first['ink'], diagnostics=False)
        wrong_cells = (canonical != first['background']) & (canonical != first['ink']) & (scene != first['background']) & (canonical != scene)
        assert np.any(wrong_cells)
        canonical[wrong_cells] = scene[wrong_cells]
        restored = canonical.T if first['transposed'] else canonical
        assert np.array_equal(solve(restored), pair['output'])
        reversed_rings, _ = _predict(first['model'], tuple(reversed(first['templates'])), first['indices'], first['bounds'], first['background'], first['ink'])
        if first['transposed']:
            reversed_rings = reversed_rings.T
        assert np.array_equal(reversed_rings, pair['output'])
        for pair in puzzle['train'] + puzzle['test']:
            _, info = _analyze(pair['input'])
            m, bb = info['model'], info['bounds']
            shapes.add((np.asarray(pair['input']).shape, np.asarray(pair['output']).shape))
            growth.add(m['h'][1]); shutter_motion.add((m['v'][1], m['t'][1]))
            widths.add(m['w']); counts.add(m['n'])
            backgrounds.add(info['background']); inks.add(info['ink']); crops.add(pair['output'].shape)
            direction = ('left' if bb[3] < m['x'] else 'right')
            directions.add((('up' if direction == 'left' else 'down') if info['transposed'] else direction))
            templates_seen.update(info['templates'])
            bodies = [i for i in info['indices'] if _geom(m, i)[1] <= bb[3] and _geom(m, i)[1] + m['w'] - 1 >= bb[2]
                      and _geom(m, i)[0] <= bb[1] and _geom(m, i)[0] + _geom(m, i)[2] - 1 >= bb[0]]
            hidden += any(_site(m, t, i)[2] for t in info['templates'] for i in bodies)
            partial += any(bb[0] > _geom(m, i)[0] or bb[1] < _geom(m, i)[0] + _geom(m, i)[2] - 1
                           or bb[2] > _geom(m, i)[1] or bb[3] < _geom(m, i)[1] + m['w'] - 1 for i in bodies)
            skipped += min(bodies) >= m['n'] + 1 or max(bodies) <= -2
            centers = info['centers']
            clusters += any(fill == other and max(abs(r - rr), abs(c - cc)) == 1
                            for j, (r, c, fill) in enumerate(centers) for rr, cc, other in centers[:j])
            halo += any(not (bb[0] <= r <= bb[1] and bb[2] <= c <= bb[3]) and
                        bb[0] - 1 <= r <= bb[1] + 1 and bb[2] - 1 <= c <= bb[3] + 1 for r, c, _ in centers)
            color_ids = {}; normalized.add(tuple(tuple(color_ids.setdefault(int(v), len(color_ids)) for v in row) for row in pair['output']))
    assert len(signatures) == 42 and len(shapes) >= 60 and len(normalized) >= 100
    assert {-2, -1, 0, 1, 2} <= growth and len(shutter_motion) >= 6
    assert widths == {5, 6} and counts == {3, 4} and len(crops) >= 35
    assert directions == {'left', 'right', 'up', 'down'}
    assert len(backgrounds) >= 8 and len(inks) >= 8 and len(templates_seen) >= 35
    assert hidden >= 42 and clusters >= 42 and partial >= 20 and skipped >= 5 and halo >= 2
    p = generate_puzzle(0)
    q = generate_puzzle(0)
    assert all(_key(x[k]) == _key(y[k]) for s in p for x, y in zip(p[s], q[s]) for k in ('input', 'output'))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(19, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    for kind, test, shift, axis in product((0, 1, 2), (False, True), range(4), range(2)):
        fallback_keys = set()
        for variant in range(16):
            pair = _fallback_pair(0, 8, (2, 7), kind, test, shift, axis, variant)
            fallback_keys.add(_key(pair['input']))
        assert len(fallback_keys) == 16
    conflict_model = dict(x=0, step=6, w=5, y=(0, 0), h=(9, 0), v=(2, 0), t=(5, 0), palette=(2, 7), n=3)
    conflict_sites = ((0, 0, 1, 1), (1, 0, 0, 1))
    for path in ('predicate', 'strokes'):
        try:
            if path == 'predicate':
                _predict(conflict_model, conflict_sites, [0], (0, 8, 0, 4), 0, 8)
            else:
                _latent_fields(conflict_model, conflict_sites, 1, 0, 8, (9, 5))
        except AssertionError:
            pass
        else:
            raise AssertionError('conflicting complementary rings accepted')
    bad = generate_puzzle(7, 2, 1)
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted label accepted')
    return dict(seeds=42, distinct_puzzles=len(signatures), shape_pairs=len(shapes),
                normalized_outputs=len(normalized), directions=sorted(directions),
                growth=sorted(growth), shutter_motion=sorted(shutter_motion),
                hidden_defect_pairs=hidden, adjacent_center_pairs=clusters,
                partial_pairs=partial, skipped_pairs=skipped, outside_center_halos=halo)
