"""Contour printing through island stencils; all semantics are input-only."""
from __future__ import annotations

import numpy as np

N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
MARKER = 9


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return g.astype(np.int8, copy=True)


def _flood(allowed, start):
    reached = np.zeros(allowed.shape, bool)
    if not allowed[start]:
        return reached
    reached[start] = True
    queue = [start]
    for r, c in queue:
        for dr, dc in N4:
            x, y = r + dr, c + dc
            if (0 <= x < allowed.shape[0] and 0 <= y < allowed.shape[1]
                    and allowed[x, y] and not reached[x, y]):
                reached[x, y] = True
                queue.append((x, y))
    return reached


def _components(mask):
    remaining = mask.copy()
    result = []
    while remaining.any():
        p = tuple(map(int, np.argwhere(remaining)[0]))
        component = _flood(remaining, p)
        result.append(component)
        remaining[component] = False
    return result


def _dilate(mask, diagonal=True):
    h, w = mask.shape
    padded = np.pad(mask, 1)
    result = mask.copy()
    for dr, dc in N8 if diagonal else N4:
        result |= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return result


def _exterior(selected):
    return _flood(np.pad(~selected, 1, constant_values=True), (0, 0))


def _depth(mask, diagonal=True):
    """Successive erosion of a filled silhouette, including off-canvas edges."""
    h, w = mask.shape
    current = mask.copy()
    depth = np.full(mask.shape, -1, np.int16)
    level = 0
    while current.any():
        padded = np.pad(current, 1)
        inner = current.copy()
        for dr, dc in N8 if diagonal else N4:
            inner &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        depth[current & ~inner] = level
        current = inner
        level += 1
    return depth


def _roles(g):
    counts = np.bincount(g.ravel(), minlength=10)
    if counts[MARKER] != 1:
        raise ValueError("color 9 must be the unique probe head")
    order = sorted(range(9), key=lambda c: (-int(counts[c]), c))
    bg, st = order[:2]
    if not counts[bg] > counts[st] > max(counts[order[2]], 1):
        raise ValueError("background and structure require strict frequency ranks")
    return bg, st


def _analyze(g):
    bg, st = _roles(g)
    marker = tuple(map(int, np.argwhere(g == MARKER)[0]))
    selected = _flood(g != st, marker)
    outside = _exterior(selected)
    hull = ~outside[1:-1, 1:-1]
    depth = _depth(hull)
    cues = np.argwhere((g != bg) & (g != st) & (g != MARKER))
    if not len(cues):
        raise ValueError("the probe needs at least one visible palette entry")
    direction = None
    length = 0
    for r, c in cues:
        dr, dc = int(r) - marker[0], int(c) - marker[1]
        if abs(dr) != abs(dc) or not dr:
            raise ValueError("palette cues must share one diagonal ray from the head")
        d = (1 if dr > 0 else -1, 1 if dc > 0 else -1)
        if direction is not None and direction != d:
            raise ValueError("palette cues lie on different rays")
        direction = d
        length = max(length, abs(dr))
    if not 2 <= length <= 4:
        raise ValueError("palette period must have length 2..4")
    m = int(depth[marker])
    positions, period = [], []
    for t in range(1, length + 1):
        r, c = marker[0] + t * direction[0], marker[1] + t * direction[1]
        if (not 0 <= r < g.shape[0] or not 0 <= c < g.shape[1]
                or not selected[r, c] or depth[r, c] != m + t):
            raise ValueError("probe entries must advance one exterior depth at a time")
        if int(g[r, c]) in (st, MARKER):
            raise ValueError("palette cannot use structure or marker")
        positions.append((r, c))
        period.append(int(g[r, c]))
    assert period[-1] != bg
    islands = (g == st) & hull
    keep = np.zeros(g.shape, bool)
    neighbor = _dilate(selected, diagonal=False)
    for component in _components(g == st):
        if np.any(component & neighbor):
            keep |= component
    return dict(bg=bg, st=st, marker=marker, selected=selected,
                outside=outside, hull=hull, depth=depth, islands=islands,
                keep=keep, erased=(g == st) & ~keep, period=tuple(period),
                phase=m + 1, direction=direction, positions=positions)


def _render(g, a, *, phase=None, depth=None, all_islands=False, solid=False):
    out = g.copy()
    out[a['erased']] = a['bg']
    mask = a['islands']
    if all_islands:
        # Counterfactual: apply the same registered cycle to every compartment.
        # It retains A's computation on the selected islands, isolating B's role.
        out = g.copy()
        palette = np.array(a['period'], np.int8)
        for region in _components(g != a['st']):
            hull = ~_exterior(region)[1:-1, 1:-1]
            dep = _depth(hull)
            stencil = (g == a['st']) & hull
            out[stencil] = palette[(dep[stencil] - a['phase']) % len(palette)]
    elif solid:
        out[mask] = a['period'][0]
    else:
        dep = a['depth'] if depth is None else depth
        ph = a['phase'] if phase is None else phase
        palette = np.array(a['period'], np.int8)
        out[mask] = palette[(dep[mask] - ph) % len(palette)]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Select, fill holes for depth measurement, then print only the islands."""
    g = _grid(input_grid)
    return _render(g, _analyze(g))


def _paths(open_region, hull, needed):
    dep = _depth(hull)
    h, w = hull.shape
    paths = []
    for r, c in np.argwhere(open_region & (dep == 0)):
        r, c = int(r), int(c)
        for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            def inside(x, y):
                return 0 <= x < h and 0 <= y < w and hull[x, y]
            if inside(r - dr, c) or inside(r, c - dc):
                continue
            positions = [(r + t * dr, c + t * dc) for t in range(needed)]
            if all(0 <= x < h and 0 <= y < w and open_region[x, y]
                   and dep[x, y] == t for t, (x, y) in enumerate(positions)):
                paths.append((r, c, dr, dc))
    return paths


def _geometry_ok(st):
    # Avoid diagonal-only structure joins and enclosed free rooms.
    a, b, c, d = st[:-1, :-1], st[:-1, 1:], st[1:, :-1], st[1:, 1:]
    if np.any((a == d) & (b == c) & (a != b)):
        return False
    regions = _components(~st)
    return bool(regions) and all(m.sum() >= 8 and
        (m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())
        for m in regions)


def _polyomino(rng, limit):
    # Random connected growth, including bars, blocks, bent and branched objects.
    kind = int(rng.integers(4))
    h, w = int(rng.integers(1, limit + 1)), int(rng.integers(1, limit + 1))
    if kind == 0:
        return np.ones((h, w), bool)
    if kind == 1:
        return np.ones((1, int(rng.integers(2, limit * 2 + 1))), bool)
    if kind == 2:
        shape = np.zeros((h, w), bool)
        shape[:, 0] = shape[-1, :] = True
        return np.rot90(shape, int(rng.integers(4))).copy()
    shape = np.zeros((h, w), bool)
    shape[h // 2, w // 2] = True
    for _ in range(int(rng.integers(1, h * w + 1))):
        fringe = _dilate(shape, False) & ~shape
        if not fringe.any():
            break
        r, c = np.argwhere(fringe)[int(rng.integers(fringe.sum()))]
        shape[r, c] = True
    rr, cc = np.nonzero(shape)
    return shape[rr.min():rr.max() + 1, cc.min():cc.max() + 1]


def _place(rng, st, region, reserved, count, edge=False):
    for _ in range(count):
        for attempt in range(35):
            motif = _polyomino(rng, 4 if attempt < 25 else 2)
            if rng.integers(2):
                motif = motif.T
            sh, sw = motif.shape
            if sh > st.shape[0] or sw > st.shape[1]:
                continue
            r = int(rng.integers(st.shape[0] - sh + 1))
            c = int(rng.integers(st.shape[1] - sw + 1))
            cells = np.zeros(st.shape, bool)
            cells[r:r + sh, c:c + sw] = motif
            if not np.all(region[cells]) or np.any(cells & reserved):
                continue
            if edge:
                if not (cells[0].any() or cells[-1].any()
                        or cells[:, 0].any() or cells[:, -1].any()):
                    continue
                halo = _dilate(cells)
            else:
                # Two free cells separate islands from the outer silhouette.
                if r < 2 or c < 2 or r + sh > st.shape[0] - 2 or c + sw > st.shape[1] - 2:
                    continue
                halo = _dilate(_dilate(cells))
            if not np.all(region[halo]) or np.any(halo & st) or np.any(halo & reserved):
                continue
            trial = st | cells
            if not _geometry_ok(trial):
                continue
            st[cells] = True
            region[cells] = False
            break


def _partition(rng, h, w, target):
    st = np.zeros((h, w), bool)
    rooms = [(0, h, 0, w)]
    for _ in range(target - 1):
        options = [(i, axis) for i, (a, b, c, d) in enumerate(rooms)
                   for axis, length in ((0, b - a), (1, d - c)) if length >= 11]
        if not options:
            break
        i, axis = options[int(rng.integers(len(options)))]
        a, b, c, d = rooms.pop(i)
        if axis == 0:
            q = int(rng.integers(a + 4, b - 4))
            st[q, c:d] = True
            rooms.extend(((a, q, c, d), (q + 1, b, c, d)))
        else:
            q = int(rng.integers(c + 4, d - 4))
            st[a:b, q] = True
            rooms.extend(((a, b, c, q), (a, b, q + 1, d)))
    return st


def _intrusions(rng, st, budget, needed):
    """Attach uneven solid lobes to walls or canvas edges; keep usable probes."""
    for _ in range(budget):
        for _attempt in range(15):
            h, w = st.shape
            if st.any() and rng.random() < .65:
                r, c = map(int, np.argwhere(st)[int(rng.integers(st.sum()))])
            else:
                edge = int(rng.integers(4))
                r = 0 if edge == 0 else h - 1 if edge == 1 else int(rng.integers(h))
                c = 0 if edge == 2 else w - 1 if edge == 3 else int(rng.integers(w))
            rh, cw = int(rng.integers(1, 5)), int(rng.integers(1, 5))
            a, b = max(0, r - rh // 2), min(h, r + rh)
            x, y = max(0, c - cw // 2), min(w, c + cw)
            trial = st.copy()
            trial[a:b, x:y] = True
            if not _geometry_ok(trial):
                continue
            if any(_paths(m, ~_exterior(m)[1:-1, 1:-1], needed)
                   for m in _components(~trial)):
                st[:] = trial
                break


def _base_scene(rng, witness, test, length, marker_depth):
    needed = marker_depth + length + 1
    reserved = None
    if witness == 1:
        h, w = int(rng.integers(22, 31)), int(rng.integers(26, 31))
        s = int(rng.integers(11, (w - 2) // 2 + 1))
        st = np.zeros((h, w), bool)
        st[:, s] = True
        notch = int(rng.integers(9, h - 7))
        st[notch:notch + 2, s - 3:s] = True
        marker_path = (0, 0, 1, 1)
        # This pixel distinguishes corner-inclusive depth from L1 depth.
        st[notch - int(rng.integers(4, 7)):notch - 2, s - 6] = True
        start = (0, 0)
    elif witness == 2:
        h, w = int(rng.integers(22, 31)), int(rng.integers(25, 31))
        s = int(rng.integers(max(14, w // 2 + 1), w - 5))
        cut = w - s - 1
        st = np.zeros((h, w), bool)
        st[:, cut] = True
        row = int(rng.integers(8, h - 7))
        st[row:row + int(rng.integers(1, 4)), cut + 3:cut + s // 2 + 2] = True
        marker_path = (h - 1, w - 1, -1, -1)
        start = (h - 1, w - 1)
    else:
        h = int(rng.integers(16 if test else 12, 31))
        w = int(rng.integers(16 if test else 14, 31))
        target = int(rng.integers(1, 6))
        st = _partition(rng, h, w, target)
        _intrusions(rng, st, int(rng.integers(0, 6)), needed)
        options = []
        for m in _components(~st):
            hull = ~_exterior(m)[1:-1, 1:-1]
            for path in _paths(m, hull, needed):
                options.append((m, path))
        if not options:
            return None
        selected, marker_path = options[int(rng.integers(len(options)))]
        start = tuple(map(int, np.argwhere(selected)[0]))
    selected = _flood(~st, start)
    if not _paths(selected, ~_exterior(selected)[1:-1, 1:-1], needed):
        return None
    r, c, dr, dc = marker_path
    reserved = np.zeros(st.shape, bool)
    for t in range(needed):
        reserved[r + t * dr, c + t * dc] = True
    _place(rng, st, selected, reserved, int(rng.integers(0 if witness else 1, 5)))
    # Objects beyond the chosen boundary provide erasure evidence.
    for region in _components(~st):
        if region[start]:
            continue
        _place(rng, st, region, reserved, int(rng.integers(1, 4)))
        if rng.random() < .45:
            _place(rng, st, region, reserved, 1, edge=True)
    selected = _flood(~st, start)
    if not _geometry_ok(st):
        return None
    return st, selected, marker_path


def _make_pair(rng, witness, test, bg, st_color, rotation, reflect, ordinal):
    length = 2 if witness == 1 else 4 if witness == 2 else int(rng.integers(2, 5))
    m = 0 if witness == 1 else 1 if witness == 2 else int(rng.integers(2, 4)) if test else int(rng.integers(0, 3))
    for attempt in range(80):
        base = _base_scene(rng, witness, test, length, m)
        if base is None:
            continue
        structure, selected, path = base
        r, c, dr, dc = path
        colors = [v for v in range(9) if v not in (bg, st_color)]
        period = [int(v) for v in rng.choice(colors, length, replace=False)]
        if witness == 2:
            period[1] = bg
        elif witness == 0 and rng.random() < .4:
            period[int(rng.integers(length - 1))] = bg
        g = np.full(structure.shape, bg, np.int8)
        g[structure] = st_color
        g[r + m * dr, c + m * dc] = MARKER
        for t, color in enumerate(period, 1):
            g[r + (m + t) * dr, c + (m + t) * dc] = color
        a = _analyze(g)
        out = _render(g, a)
        if not a['islands'].any() or (witness and not a['erased'].any()):
            continue
        sizes = [int(region.sum()) for region in _components(g != st_color)]
        if witness == 1 and int(a['selected'].sum()) >= max(sizes):
            continue
        if witness == 2 and int(a['selected'].sum()) <= min(sizes):
            continue
        printed = out[a['islands']]
        if len(set(map(int, printed))) < 2:
            continue
        if witness == 1:
            if np.array_equal(out, _render(g, a, depth=_depth(a['hull'], False))):
                continue
        if witness == 2 and not np.any(printed == bg):
            continue
        for key, array in (('input', g), ('output', out)):
            array = np.rot90(array, rotation)
            if reflect:
                array = np.fliplr(array)
            if key == 'input':
                inp = array.copy()
            else:
                output = array.copy()
        return {'input': inp, 'output': output}
    # A deterministic constructive geometry also supports deeper test probes.
    # It keeps the same rule and does not depend on an output lookup table.
    h, w = 26 + ordinal % 5, 28 + ordinal // 5 % 3
    structure = np.zeros((h, w), bool)
    structure[:, w - 5] = True
    structure[h // 2, 2:w // 2] = True
    structure[4:6, w - 3] = True
    g = np.full((h, w), bg, np.int8)
    g[structure] = st_color
    g[m, m] = MARKER
    colors = [v for v in range(9) if v not in (bg, st_color)]
    period = [int(v) for v in rng.choice(colors, length, replace=False)]
    for t, v in enumerate(period, 1):
        g[m + t, m + t] = v
    out = solve(g)
    if witness:
        raise RuntimeError("teaching witness construction unexpectedly exhausted")
    g, out = np.rot90(g, rotation), np.rot90(out, rotation)
    if reflect:
        g, out = np.fliplr(g), np.fliplr(out)
    return {'input': g.copy(), 'output': out.copy()}


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count('num_train', num_train, 2, 10), _count('num_test', num_test, 1, 4)
    if nt <= ne:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle = {'train': [], 'test': []}
    seen = set()
    # Pair roles are visible. A shuffled schedule prevents colors from naming roles.
    backgrounds = [int(v) for v in rng.permutation(9)]
    shared_rotation, shared_reflect = int(rng.integers(4)), bool(rng.integers(2))
    for i in range(nt + ne):
        witness = i + 1 if i < 2 else 0
        test = i >= nt
        bg = backgrounds[i % 9]
        st = int(rng.choice([v for v in range(9) if v != bg]))
        rotation = shared_rotation if witness else int(rng.integers(4))
        reflect = shared_reflect if witness else bool(rng.integers(2))
        for retry in range(12):
            pair = _make_pair(rng, witness, test, bg, st, rotation, reflect, i + retry)
            key = _key(pair['input'])
            if key not in seen:
                break
        else:
            raise RuntimeError("could not construct a distinct input")
        seen.add(key)
        puzzle['test' if test else 'train'].append(pair)
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    return puzzle


def _distance_depth(hull, exterior):
    """Check erosion against direct Chebyshev distance to the padded exterior."""
    result = np.full(hull.shape, -1, np.int16)
    outside_points = np.argwhere(exterior)
    for r, c in np.argwhere(hull):
        difference = np.abs(outside_points - (int(r) + 1, int(c) + 1))
        result[r, c] = int(np.max(difference, axis=1).min()) - 1
    return result


def _features(g, out):
    a = _analyze(g)
    islands = _components(a['islands'])
    free = _components(g != a['st'])
    area = int(a['selected'].sum())
    sizes = [int(m.sum()) for m in free]
    corners = tuple(i for i, v in enumerate((a['selected'][0, 0], a['selected'][0, -1],
                                            a['selected'][-1, 0], a['selected'][-1, -1])) if v)
    stencil_shapes = []
    for mask in islands:
        rr, cc = np.nonzero(mask)
        local = mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
        stencil_shapes.append(_key(local))
    manhattan = _render(g, a, depth=_depth(a['hull'], False))
    local = out.copy()
    for mask in islands:
        dep = _depth(mask)
        local[mask] = np.array(a['period'])[(dep[mask] - a['phase']) % len(a['period'])]
    rr, cc = np.nonzero(a['hull'])
    box = np.full(g.shape, -1, np.int16)
    box[rr, cc] = np.minimum.reduce((rr - rr.min(), rr.max() - rr, cc - cc.min(), cc.max() - cc))
    return dict(shape=g.shape, background=a['bg'], structure=a['st'],
                compartments=len(free), islands=len(islands), erased=len(_components(a['erased'])),
                period=len(a['period']), marker_depth=a['phase'] - 1,
                direction=a['direction'], corners=corners,
                smaller=area < max(sizes), larger=len(sizes) > 1 and area > min(sizes),
                selection_changed=not np.array_equal(out, _render(g, a, all_islands=True)),
                cycle_changed=not np.array_equal(out, _render(g, a, solid=True)),
                phase_changed=not np.array_equal(out, _render(g, a, phase=0)),
                diagonal_changed=not np.array_equal(out, manhattan),
                local_changed=not np.array_equal(out, local),
                box_changed=not np.array_equal(out, _render(g, a, depth=box)),
                repeat=int(a['depth'][a['islands']].max()) - int(a['depth'][a['islands']].min()) >= len(a['period']),
                neutral_print=bool(np.any(out[a['islands']] == a['bg'])),
                internal_gap=a['bg'] in a['period'][1:-1],
                stencil_shapes=stencil_shapes)


def validate_generated_puzzle(puzzle):
    """Validate arrays, printing semantics, and the combined teaching evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, train_features, training = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            a = _analyze(inp)
            assert _geometry_ok(inp == a['st'])
            assert a['islands'].any()
            assert np.array_equal(a['depth'], _distance_depth(a['hull'], a['outside']))
            assert np.array_equal(out, _render(inp, a))
            # The negative-space field is latent: only structural cells can change.
            assert np.array_equal(out[inp != a['st']], inp[inp != a['st']])
            assert np.all(out[a['keep'] & ~a['islands']] == a['st'])
            assert np.all(out[a['erased']] == a['bg'])
            assert set(map(int, out[a['islands']])).issubset(set(a['period']))
            assert len(set(map(int, out[a['islands']]))) >= 2
            assert out[a['marker']] == MARKER and not np.array_equal(inp, out)
            if split == 'train':
                train_features.append(_features(inp, out))
                training.append((inp, out, a))
    assert {2, 4}.issubset({f['period'] for f in train_features})
    assert {0, 1}.issubset({f['marker_depth'] for f in train_features})
    for feature in ('smaller', 'larger', 'selection_changed', 'cycle_changed', 'phase_changed',
                    'diagonal_changed', 'local_changed', 'box_changed', 'repeat',
                    'neutral_print', 'internal_gap'):
        assert any(f[feature] for f in train_features), f"missing training evidence: {feature}"
    assert any(f['erased'] for f in train_features)
    assert not set.intersection(*(set(f['corners']) for f in train_features))
    # LCM(2, 3, 4) is 12: these exhaust all fixed-phase shortcuts.
    assert not any(all(np.array_equal(out, _render(inp, a, phase=phase))
                       for inp, out, a in training) for phase in range(12))


def _puzzle_key(puzzle):
    return tuple((s, _key(np.asarray(p['input'])), _key(np.asarray(p['output'])))
                 for s in ('train', 'test') for p in puzzle[s])


def _regressions():
    # A literal fixture checks the intended rule independently of scene sampling.
    g = np.zeros((17, 24), np.int8)
    g[:, 16] = 8
    g[7:10, 3:9] = 8
    g[5:7, 20:22] = 8
    g[0, 0], g[1, 1], g[2, 2] = 9, 2, 4
    expected = g.copy()
    expected[5:7, 20:22] = 0
    for r in range(7, 10):
        for c in range(3, 9):
            d = min(r, 16 - r, c, 15 - c)
            expected[r, c] = (2, 4)[(d - 1) % 2]
    assert np.array_equal(solve(g), expected)
    for rotation in range(4):
        for reflection in (False, True):
            inp, out = np.rot90(g, rotation), np.rot90(expected, rotation)
            if reflection:
                inp, out = np.fliplr(inp), np.fliplr(out)
            assert np.array_equal(solve(inp), out)
    remap = np.array([3, 4, 5, 6, 7, 8, 0, 1, 2, 9], np.int8)
    assert np.array_equal(solve(remap[g]), remap[expected])
    # Moving the same probe one depth inward rotates all existing island colors.
    shifted = g.copy()
    shifted[0, 0] = shifted[1, 1] = shifted[2, 2] = 0
    shifted[1, 1], shifted[2, 2], shifted[3, 3] = 9, 2, 4
    shifted_expected = shifted.copy()
    shifted_expected[5:7, 20:22] = 0
    shifted_expected[7:10, 3:9] = np.where(expected[7:10, 3:9] == 2, 4, 2)
    assert np.array_equal(solve(shifted), shifted_expected)
    # A new disconnected stencil cannot perturb ink already printed elsewhere.
    added = g.copy()
    added[12:14, 10:12] = 8
    assert np.array_equal(solve(added)[7:10, 3:9], expected[7:10, 3:9])
    distractors = g.copy()
    distractors[0:2, 19:21] = 8
    distractors[12:14, 19:21] = 8
    distractors[15:17, 23] = 8
    assert np.array_equal(solve(distractors), expected)
    # Repositioning the probe across the divider selects the opposite compartment.
    across = g.copy()
    across[0, 0] = across[1, 1] = across[2, 2] = 0
    across[0, 23], across[1, 22], across[2, 21] = 9, 2, 4
    across_out = solve(across)
    assert np.all(across_out[7:10, 3:9] == 0)
    assert np.any(across_out[5:7, 20:22] != 8)
    # Connecting a formerly transparent island to the divider changes its role.
    attached = g.copy()
    attached[8, 9:17] = 8
    attached_out = solve(attached)
    assert np.all(attached_out[7:10, 3:9] == 8)
    original = g.copy()
    solve(g)
    assert np.array_equal(g, original)


def self_test():
    """Bounded semantic regressions and actual structural variation over 42 seeds."""
    _regressions()
    complete, shapes, geometries, stencils = set(), set(), set(), set()
    features = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        complete.add(_puzzle_key(puzzle))
        local_features = []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                a = _analyze(inp)
                f = _features(inp, out)
                features.append(f)
                local_features.append(f)
                shapes.add(inp.shape)
                geometries.add(_key((inp == a['st']).astype(np.int8)))
                stencils.update(f['stencil_shapes'])
                assert np.array_equal(solve(inp.tolist()), out)
        assert len({f['background'] for f in local_features}) >= 3
        assert len({f['marker_depth'] for f in local_features}) >= 3
        assert len({f['shape'] for f in local_features}) >= 2
    assert len(complete) == 42 and len(shapes) >= 50 and len(geometries) >= 150
    assert len(stencils) >= 30
    assert {1, 2, 3}.issubset({f['compartments'] for f in features})
    assert {1, 2, 3}.issubset({f['islands'] for f in features})
    assert {0, 1, 2}.issubset({f['erased'] for f in features})
    assert {0, 1, 2, 3}.issubset({f['marker_depth'] for f in features})
    assert {2, 3, 4} == {f['period'] for f in features}
    assert len({f['direction'] for f in features}) == 4
    assert len({f['background'] for f in features}) == 9
    assert len({f['structure'] for f in features}) == 9
    # The supplied verifier exhausts all legal counts; keep this test bounded
    # by exercising the count extremes alongside the 42 structural seeds.
    for nt, ne in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(73, nt, ne)
        validate_generated_puzzle(p)
    assert _puzzle_key(generate_puzzle(91)) == _puzzle_key(generate_puzzle(91))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    damaged = generate_puzzle(55, 2, 1)
    a = _analyze(damaged['test'][0]['input'])
    r, c = map(int, np.argwhere(a['islands'])[0])
    damaged['test'][0]['output'][r, c] = a['st']
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted printing accepted")


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds, count boundaries, semantic regressions)')
