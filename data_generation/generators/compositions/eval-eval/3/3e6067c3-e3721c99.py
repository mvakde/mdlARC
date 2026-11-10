"""Laser itinerary: route-controlled surgery followed by fragment topology.

Only standard-library code and NumPy are used. Parsing and solving receive pixels
only; sampling preferences never enter the decoder.
"""
import numpy as np


def _components(mask):
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for r, c in zip(*np.nonzero(mask)):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        queue = [(r, c)]
        seen[r, c] = True
        cells = []
        while queue:
            y, x = queue.pop()
            cells.append((y, x))
            for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1]
                        and mask[yy, xx] and not seen[yy, xx]):
                    seen[yy, xx] = True
                    queue.append((yy, xx))
        result.append(cells)
    return result


def _mask(cells):
    a = np.asarray(cells, dtype=int)
    lo, hi = a.min(0), a.max(0) + 1
    mask = np.zeros(tuple(hi-lo), dtype=bool)
    mask[tuple((a-lo).T)] = True
    return mask, tuple(int(x) for x in lo)


def _holes(mask):
    # A canvas edge is never an extra wall. Only four-neighbor connectivity
    # matters; a diagonal escape does not open a hole.
    padded = np.pad(np.asarray(mask, dtype=bool), 1)
    return len(_components(~padded)) - 1


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _band(a, b, size, patch, shape):
    result = np.zeros(shape, dtype=bool)
    off = (size-patch)//2
    if a[0] == b[0] and abs(a[1]-b[1]) > size:
        result[a[0]+off:a[0]+off+patch,
               min(a[1], b[1])+size:max(a[1], b[1])] = True
    elif a[1] == b[1] and abs(a[0]-b[0]) > size:
        result[min(a[0], b[0])+size:max(a[0], b[0]),
               a[1]+off:a[1]+off+patch] = True
    else:
        return None
    return result


def _routes(graph, labels, key):
    answers = []
    def visit(path, used):
        if len(answers) == 2:
            return
        if len(path) == len(key):
            answers.append(tuple(path))
            return
        choices = range(len(labels)) if not path else graph[path[-1]]
        for j in choices:
            if j not in used and labels[j] == key[len(path)]:
                visit(path+[j], used | {j})
    visit([], set())
    return answers


def _parse(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    assert np.all(grid[0] == bg) and np.all(grid[-1] == bg)
    separators = [(r, int(grid[r, 0])) for r in range(1, grid.shape[0]-1)
                  if grid[r, 0] != bg and np.all(grid[r] == grid[r, 0])]
    assert len(separators) == 2
    (top, carrier), (bottom, carrier2) = separators
    assert carrier == carrier2 and bottom > top+4
    mapping, glyph_order = {}, []
    for color in np.unique(grid[1:top]):
        color = int(color)
        if color == bg:
            continue
        assert color != carrier
        comps = _components(grid[1:top] == color)
        assert len(comps) == 1
        glyph, pos = _mask(comps[0])
        k = _holes(glyph)
        assert k not in mapping
        mapping[k] = color
        glyph_order.append((pos[1], k))
    assert mapping
    tokens = [(r, c) for r, c in zip(*np.nonzero(grid[bottom+1:] != bg))]
    assert 3 <= len(tokens) <= 8
    assert len({r for r, c in tokens}) == 1
    tokens.sort(key=lambda p: p[1])
    assert all(b[1]-a[1] == 2 for a, b in zip(tokens, tokens[1:]))
    key = [int(grid[bottom+1+r, c]) for r, c in tokens]
    body = grid[top+1:bottom].copy()
    stations = []
    geometry = None
    occupied = np.zeros(body.shape, dtype=bool)
    for cells in _components(body == carrier):
        carrier_mask, (r, c) = _mask(cells)
        size = carrier_mask.shape[0]
        assert size >= 3 and carrier_mask.shape[1] == size
        block = body[r:r+size, c:c+size]
        yy, xx = np.nonzero(block != carrier)
        assert len(yy)
        off = int(yy.min())
        patch = int(yy.max())-off+1
        assert off > 0 and 2*off+patch == size
        assert int(xx.min()) == off and int(xx.max()) == off+patch-1
        assert len(yy) == patch*patch
        label = int(block[off, off])
        assert label not in (bg, carrier)
        assert np.all(block[off:off+patch, off:off+patch] == label)
        assert geometry is None or geometry == (size, patch)
        geometry = (size, patch)
        stations.append((r, c, label))
        occupied[r:r+size, c:c+size] = True
    assert geometry is not None and 3 <= len(stations) <= 9
    stations.sort()
    labels = [s[2] for s in stations]
    query_colors = set(int(c) for c in np.unique(body[~occupied])) - {bg}
    assert len(query_colors) == 1
    query = query_colors.pop()
    assert query not in labels and query not in mapping.values()
    assert set(key) <= set(labels) and carrier not in key and query not in key
    assert not np.any(body[:, 0] != bg) and not np.any(body[:, -1] != bg)
    size, patch = geometry
    graph = [[] for _ in stations]
    bands = {}
    for i, a in enumerate(stations):
        for j, b in enumerate(stations):
            if i == j:
                continue
            band = _band(a, b, size, patch, body.shape)
            if band is not None and np.all(np.isin(body[band], (bg, query))):
                graph[i].append(j)
                bands[i, j] = band
    routes = _routes(graph, labels, key)
    assert len(routes) == 1, "the key must select one distinct-station route"
    route = routes[0]
    cut = np.zeros(body.shape, dtype=bool)
    for i, j in zip(route, route[1:]):
        assert not np.any(cut & bands[i, j]), "laser bands must not overlap"
        cut |= bands[i, j]
    original = body == query
    assert original.any()
    # No foreign pixels inside an original material object's cavity.
    for cells in _components(original):
        m, pos = _mask(cells)
        for hole in _components(~np.pad(m, 1)):
            if all(0 < r < m.shape[0]+1 and 0 < c < m.shape[1]+1 for r, c in hole):
                assert all(body[pos[0]+r-1, pos[1]+c-1] == bg for r, c in hole)
    fragments = []
    for cells in _components(original & ~cut):
        m, pos = _mask(cells)
        fragments.append((cells, _holes(m)))
    return dict(grid=grid, body=body, background=bg, carrier=carrier,
                query=query, size=size, patch=patch, stations=stations,
                labels=labels, key=key, top=top, bottom=bottom, mapping=mapping,
                legend_order=tuple(k for x, k in sorted(glyph_order)),
                graph=graph, bands=bands, route=route, cut=cut,
                original=original, fragments=fragments)


def _render(info, *, pre_cut=False, ignore_key=False):
    body = info['body'].copy()
    cut = info['cut']
    if ignore_key:
        cut = np.zeros(body.shape, dtype=bool)
        for band in info['bands'].values():
            cut |= band
    material = info['original'] if pre_cut else info['original'] & ~cut
    for cells in _components(material):
        m, _ = _mask(cells)
        color = info['mapping'].get(_holes(m), info['background'])
        body[tuple(np.asarray(cells).T)] = color
    # Beam cells replace both background and removed material. Their colors are
    # annotations of cuts, never material walls in the topology calculation.
    edges = list(info['bands']) if ignore_key else list(zip(info['route'], info['route'][1:]))
    for i, j in edges:
        body[info['bands'][i, j]] = info['labels'][i]
    out = info['grid'].copy()
    out[info['top']+1:info['bottom']] = body
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the visible itinerary, cut the material, and classify fragments."""
    return _render(_parse(input_grid))


def _compact(k):
    mask = np.ones((3, max(3, 2*k+1)), dtype=bool)
    if k:
        mask[1, 1:2*k:2] = False
    return mask


def _object(k, rng):
    # Connected material with separated cavities. Exterior carving and growth
    # alter silhouettes only after checking connectivity and the intended count.
    h, w = int(rng.integers(3, 8)), max(2*k+1, int(rng.integers(3, 9)))
    m = np.ones((h, w), dtype=bool)
    if k:
        parts = np.array_split(np.arange(1, w-1), k)
        for part in parts:
            x = int(rng.choice(part))
            y = int(rng.integers(1, h-1))
            m[y, x] = False
            # Grow a bent or elongated hole while retaining the desired topology.
            for _ in range(int(rng.integers(0, 5))):
                holes = np.argwhere(~m)
                yy, xx = holes[int(rng.integers(len(holes)))]
                dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[int(rng.integers(4))]
                yy, xx = int(yy+dr), int(xx+dc)
                if 0 < yy < h-1 and 0 < xx < w-1:
                    candidate = m.copy()
                    candidate[yy, xx] = False
                    if len(_components(candidate)) == 1 and _holes(candidate) == k:
                        m = candidate
        if _holes(m) != k:
            m = _compact(k)
    elif rng.random() < .6:
        # U/C shapes have a large open bay but zero holes.
        m[:] = False
        m[-1] = True
        m[:, 0] = True
        m[:, -1] = True
    for _ in range(8):
        y, x = int(rng.integers(m.shape[0])), int(rng.integers(m.shape[1]))
        if not (y in (0, m.shape[0]-1) or x in (0, m.shape[1]-1)):
            continue
        candidate = m.copy()
        candidate[y, x] = False
        if candidate.any() and len(_components(candidate)) == 1 and _holes(candidate) == k:
            m = candidate
    return np.rot90(m, int(rng.integers(4))).copy()


def _bridge(patch, rng, rich=False, width=None):
    # Two lobes joined by a thin neck. The selected beam crosses the neck,
    # separating the cavities into independently classified fragments.
    lobe_h = 3 if patch == 1 else 4
    gap = 1 if patch == 1 else 3
    w = int(width if width is not None else
            rng.integers(5, 8) if rich else rng.integers(3, 6))
    m = np.zeros((2*lobe_h+gap, w), dtype=bool)
    for start in (0, lobe_h+gap):
        m[start:start+lobe_h] = True
        m[start+1:start+lobe_h-1, 1:w-1] = False
    if rich:
        # One lobe has two cavities. Splitting yields classes 1 and 2 from an
        # original class-3 object, a test shift from the symmetric teaching case.
        m[lobe_h+gap+1:2*lobe_h+gap-1, w//2] = True
    x = int(rng.integers(w))
    m[lobe_h:lobe_h+gap, x] = True
    assert len(_components(m)) == 1 and _holes(m) == (3 if rich else 2)
    return m


def _reserve(free, r, c, mask):
    h, w = mask.shape
    free[max(0, r-1):min(free.shape[0], r+h+1),
         max(0, c-1):min(free.shape[1], c+w+1)] = False


def _locations(free, h, w):
    if h > free.shape[0] or w > free.shape[1]:
        return []
    sums = np.pad((~free).astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    totals = sums[h:, w:] - sums[:-h, w:] - sums[h:, :-w] + sums[:-h, :-w]
    return [tuple(int(x) for x in p) for p in np.argwhere(totals == 0)]


def _path(rng, positions, size, patch, shape, primary, length):
    stations_mask = np.zeros(shape, dtype=bool)
    for r, c in positions:
        stations_mask[r:r+size, c:c+size] = True
    graph, bands = [[] for _ in positions], {}
    for i, a in enumerate(positions):
        for j, b in enumerate(positions):
            if i == j:
                continue
            band = _band(a, b, size, patch, shape)
            if band is not None and not np.any(band & stations_mask):
                graph[i].append(j)
                bands[i, j] = band
    a, b = primary
    if (a, b) not in bands:
        return None
    budget = 500
    def extend(path, painted):
        nonlocal budget
        budget -= 1
        if budget < 0:
            return None
        if len(path) == length:
            return path
        for j in rng.permutation(graph[path[-1]]):
            j = int(j)
            band = bands[path[-1], j]
            if j not in path and not np.any(painted & band):
                result = extend(path+[j], painted | band)
                if result is not None:
                    return result
        return None
    path = extend([a, b], bands[a, b])
    return None if path is None else (path, graph, bands)


def _labels(rng, graph, path, pool, allow_repeat=True):
    if not allow_repeat and len(path) > len(pool):
        return None
    for attempt in range(100):
        n = min(len(pool), len(path))
        key = list(int(x) for x in rng.permutation(pool)[:n])
        key += [int(rng.choice(key)) for _ in range(len(path)-n)]
        if allow_repeat and attempt < 60 and len(key) >= 4 and rng.random() < .6:
            key[int(rng.integers(1, len(key)))] = key[0]
        rng.shuffle(key)
        if len(set(key)) < 2:
            continue
        labels = [int(rng.choice(pool)) for _ in graph]
        for j, color in zip(path, key):
            labels[j] = color
        if _routes(graph, labels, key) == [tuple(path)]:
            return labels, key
    return None


def _layout(rng, anchor, test):
    if anchor == 0:
        size, patch, bh = 5, 1, 19
        bw = int(rng.integers(26, 31))
        ys, xs = [1, 7, 13], [1, bw-6]
        cells = [(r, c) for r in ys for c in xs]
        primary = (2, 3)
    elif anchor == 1:
        size, patch, bh = 5, 3, 19
        bw = int(rng.integers(25, 31))
        ys, xs = [1, 13], [1, int(rng.integers(8, bw-11)), bw-6]
        cells = [(r, c) for r in ys for c in xs]
        primary = (1, 4)
    else:
        size = int(rng.choice((3, 3, 5)))
        patch = 1 if size == 3 else int(rng.choice((1, 3)))
        bh, bw = int(rng.integers(13, 20)), int(rng.integers(20, 31))
        nr, nc = int(rng.integers(2, 4)), int(rng.integers(2, 4))
        if nr*size+nr-1+2 > bh or nc*size+nc-1+2 > bw:
            return None
        def coords(n, s, extent):
            gaps = np.ones(n-1, dtype=int)
            spare = extent-n*s-(n-1)-2
            for j in rng.permutation(n-1):
                extra = int(rng.integers(spare+1))
                gaps[j] += extra
                spare -= extra
            start = int(rng.integers(1, spare+2))
            return [start]+[start+int(x) for x in np.cumsum(s+gaps)]
        ys, xs = coords(nr, size, bh), coords(nc, size, bw)
        cells = [(r, c) for r in ys for c in xs]
        if len(cells) > 4 and rng.random() < .6:
            remove = set(int(i) for i in rng.choice(len(cells), int(rng.integers(1, 3)), replace=False))
            cells = [p for i, p in enumerate(cells) if i not in remove]
        choices = [(i, j) for i, a in enumerate(cells) for j, b in enumerate(cells)
                   if i != j and (a[0] == b[0] or a[1] == b[1])]
        if not choices:
            return None
        primary = choices[int(rng.integers(len(choices)))]
    if rng.random() < .5:
        primary = primary[::-1]
    length = int(rng.integers(3 if not test else 4,
                             min(len(cells), 4 if anchor is not None else 8)+1))
    route_info = _path(rng, cells, size, patch, (bh, bw), primary, length)
    if route_info is None:
        return None
    return size, patch, (bh, bw), cells, primary, route_info


def _sample_pair(rng, anchor, test, roles, allow_repeat=True):
    layout = _layout(rng, anchor, test)
    if layout is None:
        return None
    size, patch, shape, positions, primary, (path, graph, bands) = layout
    bg, carrier, query = roles
    pool = [c for c in range(10) if c not in roles]
    assigned = _labels(rng, graph, path, pool, allow_repeat)
    if assigned is None:
        return None
    labels, key = assigned
    body = np.full(shape, bg, dtype=np.int8)
    free = np.ones(shape, dtype=bool)
    free[:, 0] = free[:, -1] = False
    off = (size-patch)//2
    for (r, c), color in zip(positions, labels):
        body[r:r+size, c:c+size] = carrier
        body[r+off:r+off+patch, c+off:c+off+patch] = color
        _reserve(free, r, c, np.ones((size, size), dtype=bool))
    main = bands[primary]
    total_cut = np.zeros(shape, dtype=bool)
    for a, b in zip(path, path[1:]):
        total_cut |= bands[a, b]
    other = total_cut & ~main
    horizontal = positions[primary[0]][0] == positions[primary[1]][0]
    bridge = _bridge(patch, rng, rich=test or (anchor is None and rng.random() < .35))
    if not horizontal:
        bridge = bridge.T
    # Primary surgery is centered on the joining neck, not on either cavity.
    candidates = []
    for r, c in _locations(free, *bridge.shape):
        local = main[r:r+bridge.shape[0], c:c+bridge.shape[1]]
        if horizontal:
            centered = np.all(local[(bridge.shape[0]-patch)//2:(bridge.shape[0]+patch)//2])
        else:
            centered = np.all(local[:, (bridge.shape[1]-patch)//2:(bridge.shape[1]+patch)//2])
        if centered and not other[r:r+bridge.shape[0], c:c+bridge.shape[1]].any():
            candidates.append((r, c))
    if not candidates:
        return None
    r, c = candidates[int(rng.integers(len(candidates)))]
    body[r:r+bridge.shape[0], c:c+bridge.shape[1]][bridge] = query
    _reserve(free, r, c, bridge)
    # First demonstration also opens a single ring: size/area is not a class.
    if anchor == 0:
        ring = np.ones((5, 5), dtype=bool)
        ring[1:4, 1:4] = False
        locations = [(r, c) for r, c in _locations(free, 5, 5)
                     if main[r+2, c+2] and not other[r:r+5, c:c+5].any()]
        if not locations:
            return None
        r, c = locations[int(rng.integers(len(locations)))]
        body[r:r+5, c:c+5][ring] = query
        _reserve(free, r, c, ring)
    classes = ([1, 0] if anchor == 0 else [0, 1]) if anchor is not None else [0, 1]+[
        k for k in (2, 3) if rng.random() < (.65 if test else .35)]
    if anchor is None and rng.random() < .2:
        classes = [0, 2] if rng.random() < .5 else [1, 3]
    if anchor is None:
        rng.shuffle(classes)
    class_colors = [int(x) for x in rng.permutation(pool)[:len(classes)]]
    # Required uncut two-hole erasure and zero-hole recoloring in both anchors.
    mandatory = (2, 0) if anchor is not None else ()
    counts = list(mandatory)+[int(rng.integers(4)) for _ in range(int(rng.integers(1, 5)))]
    for index, k in enumerate(counts):
        required = index < len(mandatory)
        masks = [_compact(k), _compact(k).T] if required else [_object(k, rng), _compact(k)]
        placed = False
        for m in masks:
            choices = _locations(free, *m.shape)
            if required:
                choices = [(r, c) for r, c in choices if not total_cut[r:r+m.shape[0], c:c+m.shape[1]].any()]
            if choices:
                r, c = choices[int(rng.integers(len(choices)))]
                body[r:r+m.shape[0], c:c+m.shape[1]][m] = query
                _reserve(free, r, c, m)
                placed = True
                break
        if required and not placed:
            return None
    grid = _assemble(body, classes, class_colors, key, roles, rng)
    if grid is None:
        return None
    try:
        info = _parse(grid)
    except AssertionError:
        return None
    out = _render(info)
    if np.array_equal(out, _render(info, pre_cut=True)):
        return None
    return {'input': grid, 'output': out}


def _assemble(body, classes, class_colors, key, roles, rng):
    # One compact legend, two carrier separators, and a detached itinerary.
    shape = body.shape
    bg, carrier, query = roles
    glyphs = [_compact(k) for k in classes]
    legend_width = sum(g.shape[1] for g in glyphs)+len(glyphs)-1
    if legend_width+2 > shape[1]:
        return None
    grid = np.full((shape[0]+11, shape[1]), bg, dtype=np.int8)
    left = int(rng.integers(1, shape[1]-legend_width))
    for glyph, color in zip(glyphs, class_colors):
        grid[1:4, left:left+glyph.shape[1]][glyph] = color
        left += glyph.shape[1]+1
    grid[4] = carrier
    grid[6:6+shape[0]] = body
    bottom = 7+shape[0]
    grid[bottom] = carrier
    key_left = int(rng.integers(1, shape[1]-(2*len(key)-1)))
    grid[bottom+2, key_left:key_left+2*len(key):2] = key
    return grid


def _fallback_pair(roles, anchor, test, rng):
    """Direct guaranteed-fitting construction; no probabilistic retry loop."""
    bg, carrier, query = roles
    body = np.full((19, 26), bg, dtype=np.int8)
    vertical = anchor == 1
    patch = 3 if vertical else 1
    positions = ([(r, c) for r in (1, 13) for c in (1, 10, 20)] if vertical
                 else [(r, c) for r in (1, 7, 13) for c in (1, 20)])
    pool = [int(c) for c in rng.permutation([c for c in range(10) if c not in roles])]
    labels = pool[:6]  # Unique labels give an immediately unique route.
    path = [1, 4, 5] if vertical else [2, 3, 5, 4] if test else [2, 3, 5]
    off = (5-patch)//2
    for (r, c), color in zip(positions, labels):
        body[r:r+5, c:c+5] = carrier
        body[r+off:r+off+patch, c+off:c+off+patch] = color
    bridge = _bridge(patch, rng, rich=test, width=5)
    if vertical:
        bridge = bridge.T
        r, c = 7, 7
    else:
        r, c = 6, 7
    body[r:r+bridge.shape[0], c:c+bridge.shape[1]][bridge] = query
    if not vertical:
        ring = np.ones((5, 5), dtype=bool)
        ring[1:4, 1:4] = False
        body[7:12, 14:19][ring] = query
    # Untouched listed and unlisted query classes, clear of every route edge.
    r, c = (7, 19) if vertical else (0, 10)
    m = _compact(2)
    body[r:r+3, c:c+5][m] = query
    r, c = (0, 7) if vertical else (16, 9)
    body[r:r+3, c:c+(2 if vertical else 3)] = query
    grid = _assemble(body, [0, 1] if vertical else [1, 0], pool[-2:],
                     [labels[i] for i in path], roles, rng)
    return {'input': grid, 'output': solve(grid)}


def _counts(rng, a, b):
    for value, name, low, high in ((a, 'num_train', 2, 10), (b, 'num_test', 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high):
            raise ValueError(f'{name} has an illegal count')
    if a is None:
        a = int(rng.integers(max(2, (b or 1)+1), 11))
    if b is None:
        b = int(rng.integers(1, min(4, a-1)+1))
    if a <= b:
        raise ValueError('num_train must exceed num_test')
    return int(a), int(b)


def _signature(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    nt, nq = _counts(rng, num_train, num_test)
    stable = bool(rng.random() < .4)
    roles = tuple(int(x) for x in rng.permutation(10)[:3])
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        allow_repeat = split == 'train' or any(
            len(set(key)) < len(key)
            for key in (_parse(pair['input'])['key'] for pair in puzzle['train']))
        for index in range(count):
            local_roles = roles if stable else tuple(int(x) for x in rng.permutation(10)[:3])
            anchor = index if split == 'train' and index < 2 else None
            pair = None
            for attempt in range(48):
                pair = _sample_pair(rng, anchor, split == 'test', local_roles, allow_repeat)
                if pair is not None and _signature(pair['input']) not in seen:
                    break
            else:
                pair = _fallback_pair(local_roles, anchor, split == 'test', rng)
                if _signature(pair['input']) in seen:
                    # Forty affine permutations exceed the thirteen possible
                    # prior inputs; all ten colors are valid semantic roles.
                    found = None
                    for mul in (1, 3, 7, 9):
                        for shift in range(10):
                            remap = (mul*np.arange(10)+shift) % 10
                            candidate = {k: remap[v].astype(np.int8) for k, v in pair.items()}
                            if _signature(candidate['input']) not in seen:
                                found = candidate
                                break
                        if found is not None:
                            break
                    if found is not None:
                        pair = found
                    else:
                        raise RuntimeError('distinct fallback exhausted')
            seen.add(_signature(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    p = _parse(pair['input'])
    originals, surgeries = [], []
    for cells in _components(p['original']):
        original_k = _holes(_mask(cells)[0])
        originals.append(original_k)
        remainder = np.zeros(p['body'].shape, dtype=bool)
        remainder[tuple(np.asarray(cells).T)] = True
        children = tuple(sorted(_holes(_mask(c)[0]) for c in _components(remainder & ~p['cut'])))
        surgeries.append((original_k, children))
    axes = tuple('h' if p['stations'][a][0] == p['stations'][b][0] else 'v'
                 for a, b in zip(p['route'], p['route'][1:]))
    pieces = [k for cells, k in p['fragments']]
    return dict(shape=p['grid'].shape, geometry=(p['size'], p['patch']),
                layout=tuple(s[:2] for s in p['stations']), length=len(p['key']),
                unused=len(p['stations'])-len(p['key']), axes=axes,
                classes=tuple(sorted(p['mapping'])), original_holes=tuple(originals),
                legend_order=p['legend_order'],
                fragment_holes=tuple(pieces), fragments=len(pieces),
                surgeries=tuple(surgeries),
                split=any(len(children) > 1 for k, children in surgeries),
                opening=any(sum(children) < k for k, children in surgeries),
                erased=sum(k not in p['mapping'] for k in pieces),
                roles=(p['background'], p['carrier'], p['query']),
                repeated=len(set(p['key'])) < len(p['key']))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, train, key_witnesses, test_repeated = set(), [], [], False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            signature = _signature(inp)
            assert signature not in seen
            seen.add(signature)
            p = _parse(inp)
            expected = _render(p)
            assert np.array_equal(out, expected), 'wrong surgery or fragment recoloring'
            assert np.any(p['original'] & p['cut']), 'route must actually cut material'
            assert not np.array_equal(out, _render(p, pre_cut=True)), 'original-object shortcut survives'
            # Exact preservation outside laser/material includes stations and panels.
            changed = p['original'] | p['cut']
            obody = out[p['top']+1:p['bottom']]
            assert np.array_equal(obody[~changed], p['body'][~changed])
            for cells, k in p['fragments']:
                assert np.all(obody[tuple(np.asarray(cells).T)] == p['mapping'].get(k, p['background']))
            if split == 'train':
                train.append(_features(pair))
                key_witnesses.append(not np.array_equal(out, _render(p, ignore_key=True)))
            else:
                test_repeated |= len(set(p['key'])) < len(p['key'])
    assert {axis for t in train for axis in t['axes']} == {'h', 'v'}
    assert any(a['geometry'][0] == b['geometry'][0] and a['geometry'][1] != b['geometry'][1]
               for a in train for b in train)
    assert any(t['split'] for t in train) and any(t['opening'] for t in train)
    assert any((2, (1, 1)) in t['surgeries'] for t in train)
    assert any(k == 1 and children and set(children) == {0}
               for t in train for k, children in t['surgeries'])
    assert any(t['erased'] for t in train)
    assert any(key_witnesses), 'training must distinguish routing from all alignments'
    assert {(0, 1), (1, 0)} <= {t['legend_order'] for t in train}
    assert not test_repeated or any(t['repeated'] for t in train)
    assert {k for t in train for k in t['fragment_holes'] if k in t['classes']} >= {0, 1}
    return True


def self_test():
    # Hand topology cases separate diagonal and orthogonal openings.
    m = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)
    assert _holes(m) == 1
    m[0, 1] = False
    assert _holes(m) == 0
    assert _holes(np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)) == 0
    metrics, signatures = [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(tuple(_signature(pair['input']) for s in p for pair in p[s]))
        metrics += [_features(pair) for s in p for pair in p[s]]
    assert len(signatures) == 42
    assert len({m['shape'] for m in metrics}) >= 15
    assert len({m['layout'] for m in metrics}) >= 30
    assert {m['geometry'] for m in metrics} >= {(3, 1), (5, 1), (5, 3)}
    assert len({m['roles'] for m in metrics}) >= 30
    assert len({m['length'] for m in metrics}) >= 3
    assert len({m['classes'] for m in metrics}) >= 3
    assert any(m['repeated'] for m in metrics)
    assert any(m['unused'] == 0 for m in metrics) and any(m['unused'] >= 2 for m in metrics)
    assert any(m['split'] for m in metrics) and any(m['opening'] for m in metrics)
    rng = np.random.default_rng(819)
    fallback = {'train': [_fallback_pair((0, 8, 5), i, False, rng) for i in range(2)],
                'test': [_fallback_pair((0, 8, 5), None, True, rng)]}
    assert validate_generated_puzzle(fallback)
    p, q = generate_puzzle(0), generate_puzzle(0)
    assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
    # Geometric symmetry and full color permutation preserve input-only semantics.
    remap = np.array([7, 2, 9, 0, 5, 8, 1, 6, 4, 3], dtype=np.int8)
    for s in p:
        for pair in p[s]:
            assert np.array_equal(solve(remap[pair['input']]), remap[pair['output']])
            # The key is still read left-to-right, so mirror only the work field.
            a = _parse(pair['input'])
            inp, out = pair['input'].copy(), pair['output'].copy()
            region = slice(a['top']+1, a['bottom'])
            inp[region] = inp[region, ::-1]
            out[region] = out[region, ::-1]
            assert np.array_equal(solve(inp), out)
    # Reversing the key reverses departure colors while preserving the cut mask.
    pair = p['train'][0]
    info = _parse(pair['input'])
    reversed_input = pair['input'].copy()
    row = info['bottom']+1+int(np.nonzero(reversed_input[info['bottom']+1:] != info['background'])[0][0])
    xs = np.flatnonzero(reversed_input[row] != info['background'])
    reversed_input[row, xs] = reversed_input[row, xs][::-1]
    reversed_info = _parse(reversed_input)
    assert np.array_equal(info['cut'], reversed_info['cut'])
    assert not np.array_equal(solve(reversed_input), pair['output'])
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            assert validate_generated_puzzle(generate_puzzle(4242+a*10+b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.2, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Corrupt both a preserved cell and a surviving fragment color.
    for loc in ((0, 0), None):
        bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        if loc is None:
            info = _parse(bad['train'][0]['input'])
            r, c = info['fragments'][0][0][0]
            loc = (r+info['top']+1, c)
        out = bad['train'][0]['output']
        out[loc] = (int(out[loc])+1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('output corruption accepted')
    return {'seeds_checked': 42, 'pairs_checked': len(metrics),
            'shapes': len({m['shape'] for m in metrics}),
            'layouts': len({m['layout'] for m in metrics})}
