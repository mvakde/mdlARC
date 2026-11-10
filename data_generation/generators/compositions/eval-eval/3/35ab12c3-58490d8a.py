"""Moving-fence census: whole-object departures and arrivals.

Only Python's standard library and NumPy are used.  The solver reconstructs all
roles, the fence, the displacement, and the chart orientation from one input.
"""
import numpy as np


OFFSETS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
HOST_SEEDS = (0, 517351590, 824377316, 335867564, 585275519, 677058505,
              362884294, 957047929, 53763491, 707929897, 94493389, 457788058,
              425272780, 709428398, 133658365, 723577259, 901232635, 93493833,
              320977141, 301625350, 813227269, 776896328, 836831513, 222389942,
              635893779, 725036952, 351237387, 286402599, 375666878, 155759687,
              844830922, 424885052, 4661455, 257094976, 229504461, 72067903,
              657796117, 952433680, 565961999, 671817707, 252674677, 846887245)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), 'bad grid dimensions')
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)),
             'grid must have integer colors 0..9')
    return a


def _key(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _oriented(a, turns, reflect):
    b = np.rot90(a, turns)
    return np.fliplr(b) if reflect else b


def _unoriented(a, turns, reflect):
    return np.rot90(np.fliplr(a) if reflect else a, -turns).copy()


def _vector(v, turns, reflect):
    r, c = v
    for _ in range(turns):
        r, c = -c, r
    return r, -c if reflect else c


def _aligned(a, b):
    dr, dc = abs(a[0]-b[0]), abs(a[1]-b[1])
    return max(dr, dc) > 0 and (dr == 0 or dc == 0 or dr == dc)


def _segment(a, b):
    n = max(abs(a[0]-b[0]), abs(a[1]-b[1]))
    return [(a[0]+i*(b[0]-a[0])//n, a[1]+i*(b[1]-a[1])//n)
            for i in range(n+1)]


def _cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def _intersect(a, b, c, d):
    if (max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0])
            or max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1])):
        return False
    return _cross(a, b, c)*_cross(a, b, d) <= 0 and _cross(c, d, a)*_cross(c, d, b) <= 0


def _infer_polygon(vertices):
    """The unique simple cycle through all landmarks, using visible alignment.

    Edge-set deduplication removes reverse traversal. Crossings are tested in
    continuous geometry, including crossings between diagonal lattice cells.
    """
    pts = sorted(vertices)
    _require(4 <= len(pts) <= 10, 'fence needs 4..10 landmarks')
    edges = [(i, j) for i, a in enumerate(pts) for j, b in enumerate(pts)
             if i < j and _aligned(a, b)
             and not set(_segment(a, b)[1:-1]) & set(pts)]
    adjacency = [[] for _ in pts]
    for k, (i, j) in enumerate(edges):
        adjacency[i].append((j, k))
        adjacency[j].append((i, k))
    conflicts = [set() for _ in edges]
    for k, (a, b) in enumerate(edges):
        for l, (c, d) in enumerate(edges[:k]):
            if not {a, b} & {c, d} and _intersect(pts[a], pts[b], pts[c], pts[d]):
                conflicts[k].add(l)
                conflicts[l].add(k)
    solutions = {}

    def walk(order, used, selected):
        if len(solutions) > 1:
            return
        if len(order) == len(pts):
            for neighbor, edge in adjacency[order[-1]]:
                if neighbor == 0 and not conflicts[edge] & selected:
                    solutions[tuple(sorted(selected | {edge}))] = order
            return
        for neighbor, edge in adjacency[order[-1]]:
            if neighbor not in used and not conflicts[edge] & selected:
                walk(order+[neighbor], used | {neighbor}, selected | {edge})

    walk([0], {0}, set())
    _require(len(solutions) == 1, 'fence must have exactly one simple full cycle')
    chosen, order = next(iter(solutions.items()))
    polygon = [pts[i] for i in order]
    return polygon, len(edges)-len(chosen)


def _inside_mask(shape, polygon):
    """Strict Euclidean interior at lattice-cell centers; boundary is outside."""
    rr, cc = np.indices(shape)
    inside, boundary = np.zeros(shape, bool), np.zeros(shape, bool)
    for a, b in zip(polygon, polygon[1:]+polygon[:1]):
        cross = (b[0]-a[0])*(cc-a[1])-(b[1]-a[1])*(rr-a[0])
        boundary |= ((cross == 0) & (rr >= min(a[0], b[0])) & (rr <= max(a[0], b[0]))
                     & (cc >= min(a[1], b[1])) & (cc <= max(a[1], b[1])))
        if a[0] != b[0]:
            x = a[1]+(rr-a[0])*(b[1]-a[1])/(b[0]-a[0])
            inside ^= ((a[0] > rr) != (b[0] > rr)) & (cc < x)
    return inside & ~boundary


def _components(points, diagonal=True):
    remaining, result = set(points), []
    steps = OFFSETS if diagonal else ((1, 0), (-1, 0), (0, 1), (0, -1))
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, group = [p], {p}
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = (r+dr, c+dc)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    group.add(q)
        result.append(group)
    return result


def _find_chart(grid):
    """A field-only rectangle, two scene-linked header tags, and aligned keys."""
    candidates = []
    for field in np.unique(grid):
        rr, cc = np.where(grid == field)
        r0, r1, c0, c1 = int(rr.min()), int(rr.max())+1, int(cc.min()), int(cc.max())+1
        panel = grid[r0:r1, c0:c1]
        if min(panel.shape) < 5 or not all(np.all(edge == field) for edge in
                                          (panel[0], panel[-1], panel[:, 0], panel[:, -1])):
            continue
        outside = np.ones(grid.shape, bool)
        outside[r0:r1, c0:c1] = False
        if not outside.any():
            continue
        bg = int(np.argmax(np.bincount(grid[outside], minlength=10)))
        if bg == field or np.mean(grid[outside] == bg) < .6:
            continue
        ring = np.zeros(grid.shape, bool)
        ring[max(0, r0-1):r1+1, max(0, c0-1):c1+1] = True
        ring[r0:r1, c0:c1] = False
        if not np.all(grid[ring] == bg):
            continue
        for turns in range(4):
            for reflect in (False, True):
                p = _oriented(panel, turns, reflect)
                positions = [tuple(map(int, q)) for q in np.argwhere(p != field)]
                if not 3 <= len(positions) <= 5:
                    continue
                colors = [int(p[q]) for q in positions]
                if len(set(colors)) != len(colors) or bg in colors:
                    continue
                header_row = min(r for r, c in positions)
                tags = sorted(q for q in positions if q[0] == header_row)
                keys = sorted(q for q in positions if q[0] != header_row)
                if len(tags) != 2 or not keys or tags[1][1]-tags[0][1] != 4:
                    continue
                mid = tags[0][1]+2
                if (any(c != mid for r, c in keys) or keys[0][0]-header_row < 2
                        or any(b[0]-a[0] < 2 for a, b in zip(keys, keys[1:]))):
                    continue
                source, companion = int(p[tags[0]]), int(p[tags[1]])
                if not 4 <= np.count_nonzero((grid == source) & outside) <= 10:
                    continue
                if np.count_nonzero((grid == companion) & outside) != 1:
                    continue
                candidates.append(dict(field=int(field), bg=bg, bounds=(r0, r1, c0, c1),
                                       outside=outside, canonical=p.copy(), turns=turns,
                                       reflect=reflect, source=source, companion=companion,
                                       keys=[(q, int(p[q])) for q in keys]))
    _require(len(candidates) == 1, 'chart and its orientation must be unique')
    return candidates[0]


def _inspect(grid):
    m = _find_chart(grid)
    ext, source, companion = m['outside'], m['source'], m['companion']
    vertices = [tuple(map(int, p)) for p in np.argwhere((grid == source) & ext)]
    polygon, chords = _infer_polygon(vertices)
    marker = tuple(map(int, np.argwhere((grid == companion) & ext)[0]))
    anchors = [v for v in vertices if max(abs(v[0]-marker[0]), abs(v[1]-marker[1])) == 1]
    _require(len(anchors) == 1, 'translation marker must have one adjacent vertex')
    offset = tuple(marker[i]-anchors[0][i] for i in (0, 1))
    moved = [(r+offset[0], c+offset[1]) for r, c in polygon]
    _require(all(0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] for r, c in moved),
             'translated fence is clipped')
    sm, tm = _inside_mask(grid.shape, polygon), _inside_mask(grid.shape, moved)
    # Reference box masks support a meaningful geometry-ablation check.
    r0, r1 = min(r for r, c in polygon), max(r for r, c in polygon)
    c0, c1 = min(c for r, c in polygon), max(c for r, c in polygon)
    box = [(r0, c0), (r0, c1), (r1, c1), (r1, c0)]
    bm = _inside_mask(grid.shape, box)
    bmt = _inside_mask(grid.shape, [(r+offset[0], c+offset[1]) for r, c in box])
    objects, occupied = [], set(vertices) | {marker}
    colors = sorted(set(map(int, grid[ext]))-{m['bg'], source, companion})
    for color in colors:
        for cells in _components(map(tuple, np.argwhere((grid == color) & ext))):
            _require(not cells & {(r+dr, c+dc) for r, c in occupied
                                  for dr in (-1, 0, 1) for dc in (-1, 0, 1)},
                     'payload objects or landmarks touch')
            occupied.update(cells)
            state = all(sm[p] for p in cells), all(tm[p] for p in cells)
            bbox_state = all(bm[p] for p in cells), all(bmt[p] for p in cells)
            any_state = any(sm[p] for p in cells), any(tm[p] for p in cells)
            objects.append(dict(color=color, cells=cells, state=state,
                                bbox_state=bbox_state, any_state=any_state))
    keycolors = {color for q, color in m['keys']}
    _require(keycolors <= set(colors), 'every key must have exterior payload')
    counts = {c: [0, 0] for c in keycolors}
    for obj in objects:
        if obj['color'] in counts and obj['state'] in ((True, False), (False, True)):
            counts[obj['color']][int(obj['state'][1])] += 1
    out = m['canonical'].copy()
    for (r, c), color in m['keys']:
        for side, n in enumerate(counts[color]):
            for i in range(1, n+1):
                dest = r, c+(-2 if side == 0 else 2)*i
                _require(0 < dest[1] < out.shape[1]-1, 'tally capacity exceeded')
                out[dest] = color
    m.update(vertices=vertices, polygon=polygon, marker=marker, offset=offset,
             chords=chords, objects=objects, counts=counts,
             expected=_unoriented(out, m['turns'], m['reflect']))
    return m


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve one input, with no training pairs, seed, caches, or generation state."""
    return _inspect(_grid(input_grid))['expected'].copy()


def _normalize(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _shape(rng):
    glyphs = (((0, 0),), ((0, 0), (0, 1)), ((0, 0), (1, 1)),
              ((0, 0), (0, 1), (1, 0), (1, 1)),
              ((0, 1), (1, 0), (1, 2), (2, 1)),
              ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2)),
              ((0, 0), (0, 1), (0, 2)))
    if rng.random() < .6:
        cells = glyphs[int(rng.integers(len(glyphs)))]
    else:
        cells = {(1, 1)}
        for _ in range(int(rng.integers(1, 7))):
            choices = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in OFFSETS
                              if 0 <= r+dr < 4 and 0 <= c+dc < 4}-cells)
            cells.add(choices[int(rng.integers(len(choices)))])
    turns, reflect = int(rng.integers(4)), bool(rng.integers(2))
    return _normalize([_vector(p, turns, reflect) for p in cells])


def _polygon(rng, h, w, mode):
    if mode == 'rectangle':
        return [(0, 0), (0, w), (h, w), (h, 0)]
    if mode == 'diamond':
        d = min(h, w)//2
        return [(0, d), (d, 2*d), (2*d, d), (d, 0)]
    if mode == 'clipped':
        hi = min(h//2-1, w//2-1, 6)
        a, b, c, d = map(int, rng.integers(3, hi+1, size=4))
        return [(0, a), (0, w-b), (b, w), (h-c, w),
                (h, w-c), (h, d), (h-d, 0), (a, 0)]
    if mode in ('elbow', 'clipped_elbow'):
        a, b = int(rng.integers(4, h-3)), int(rng.integers(5, w-4))
        if mode == 'clipped_elbow':
            return [(0, 3), (0, w), (a, w), (a, b), (h, b), (h, 0), (3, 0)]
        return [(0, 0), (0, w), (a, w), (a, b), (h, b), (h, 0)]
    a, b = h//3, 2*h//3
    depth = int(rng.integers(3, max(4, w//2)))
    return [(0, 0), (0, w), (a, w), (a, w-depth),
            (b, w-depth), (b, w), (h, w), (h, 0)]


def _place(rng, grid, blocked, masks, color, state, forced=None, special=None):
    sm, tm, bm, bmt = masks
    for attempt in range(18):
        cells = forced if forced is not None else _shape(rng) if attempt < 14 else ((0, 0),)
        sh, sw = max(r for r, c in cells)+1, max(c for r, c in cells)+1
        nh, nw = grid.shape[0]-sh+1, grid.shape[1]-sw+1
        if min(nh, nw) <= 0:
            continue
        valid = np.ones((nh, nw), bool)
        memberships = []
        for mask in masks:
            all_in, any_in = np.ones((nh, nw), bool), np.zeros((nh, nw), bool)
            for r, c in cells:
                cut = mask[r:r+nh, c:c+nw]
                all_in &= cut
                any_in |= cut
            memberships.append((all_in, any_in))
        for r, c in cells:
            valid &= ~blocked[r:r+nh, c:c+nw]
        for i, flag in enumerate(state):
            valid &= memberships[i][0] if flag else ~memberships[i][0]
        if special == 'partial':
            valid &= memberships[0][1] ^ memberships[1][1]
        elif special == 'false_box':
            valid &= memberships[2][0] ^ memberships[3][0]
        elif special == 'far':
            valid &= ~memberships[2][1] & ~memberships[3][1]
        positions = np.argwhere(valid)
        if len(positions):
            r0, c0 = map(int, positions[int(rng.integers(len(positions)))])
            absolute = {(r0+r, c0+c) for r, c in cells}
            for r, c in absolute:
                grid[r, c] = color
                blocked[max(0, r-1):r+2, max(0, c-1):c+2] = True
            return dict(color=color, cells=absolute, state=state)
        if forced is not None:
            break
    return None


def _make_pair(rng, k, offset, mode, witness=0, extrapolate=False):
    """Labels use placed component records, then are checked by the input parser."""
    for attempt in range(100):
        palette = list(map(int, rng.permutation(10)))
        bg, field, source, companion = palette[:4]
        keys, ignored = palette[4:4+k], palette[4+k]
        losses = list(map(int, rng.integers(0, 3, size=k)))
        gains = list(map(int, rng.integers(0, 3, size=k)))
        if witness == 1:
            losses[0], gains[0] = 2, 1
            losses[1], gains[1] = 0, 1
        elif witness == 2:
            losses[0], gains[0] = 1, 2
        elif extrapolate:
            losses[0], gains[0] = int(rng.integers(4, 6)), 1
        elif not sum(losses+gains):
            gains[0] = 1
        for i in range(k):
            if losses[i] == gains[i] == 0:
                gains[i] = 1
        # Keep either swept band packable; category colors with zero departures
        # can still have arrivals. These are sampling limits, not solver rules.
        for side in (losses, gains):
            while sum(side) > 5:
                eligible = [i for i in range(1, k) if side[i] > 0]
                side[eligible[int(rng.integers(len(eligible)))]] -= 1
        for i in range(k):
            if losses[i] == gains[i] == 0:
                gains[i] = 1
        # Limit crowding without changing protected witnesses or extrapolation.
        while sum(losses+gains) > 10:
            candidates = [(a, i) for a in (losses, gains) for i in range(1, k)
                          if a[i] > 0 and losses[i]+gains[i] > 1]
            a, i = candidates[int(rng.integers(len(candidates)))]
            a[i] -= 1
        header = int(rng.integers(1, 3))
        rows = [header+2+2*i for i in range(k)]
        ph = rows[-1]+int(rng.integers(2, 4))
        lc, rc = max(1, max(losses)), max(1, max(gains))
        mid = 2*lc+int(rng.integers(1, 3))
        pw = mid+2*rc+int(rng.integers(2, 4))
        panel = np.full((ph, pw), field, np.int8)
        panel[header, mid-2], panel[header, mid+2] = source, companion
        for r, color in zip(rows, keys):
            panel[r, mid] = color
        pad = int(rng.integers(1, 3))
        max_h = min(17, 28-ph-2*pad)
        low_h = min(11 if k == 1 else 12, max_h)
        h = int(rng.integers(low_h, max_h+1))
        w = int(rng.integers(16, 30-2*pad))
        local = _polygon(rng, h, w, mode)
        intended = [(r+pad, c+pad) for r, c in local]
        try:
            inferred, _ = _infer_polygon(intended)
        except ValueError:
            continue
        shape = (max(r for r, c in intended)+pad+1, max(c for r, c in intended)+pad+1)
        sm = _inside_mask(shape, intended)
        if not np.array_equal(sm, _inside_mask(shape, inferred)):
            continue
        shift = [(r+offset[0], c+offset[1]) for r, c in intended]
        tm = _inside_mask(shape, shift)
        edges = set().union(*(set(_segment(a, b)) for a, b in zip(intended, intended[1:]+intended[:1])))
        marker_options = []
        for r, c in intended:
            p = (r+offset[0], c+offset[1])
            if p not in edges and sum(max(abs(p[0]-a), abs(p[1]-b)) == 1 for a, b in intended) == 1:
                marker_options.append(p)
        if not marker_options:
            continue
        marker = marker_options[int(rng.integers(len(marker_options)))]
        grid = np.full(shape, bg, np.int8)
        blocked = np.zeros(shape, bool)
        for r, c in intended+[marker]:
            grid[r, c] = companion if (r, c) == marker else source
            blocked[max(0, r-1):r+2, max(0, c-1):c+2] = True
        r0, r1 = min(r for r, c in intended), max(r for r, c in intended)
        c0, c1 = min(c for r, c in intended), max(c for r, c in intended)
        box = [(r0, c0), (r0, c1), (r1, c1), (r1, c0)]
        masks = sm, tm, _inside_mask(shape, box), _inside_mask(shape, [(r+offset[0], c+offset[1]) for r, c in box])
        requests = []
        for i, color in enumerate(keys):
            for side, n in enumerate((losses[i], gains[i])):
                for j in range(n):
                    if witness == 1 and i == side == j == 0:
                        forced = ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2))
                    elif witness == 1 and i == side == 0 and j == 1:
                        forced = ((0, 0),)
                    elif sum((losses, gains)[side]) >= 4:
                        # Bars extend across the narrow band rather than along
                        # it, retaining non-unit object area in crowded cases.
                        forced = ((0, 0), (1, 0)) if offset[0] else ((0, 0), (0, 1))
                    else:
                        forced = None
                    requests.append((color, (side == 0, side == 1), forced, None))
        # Prioritize the forced connectivity witness; shuffle remaining movers.
        if witness == 1:
            first, rest = requests[:1], requests[1:]
            rng.shuffle(rest)
            requests = first+rest
        else:
            rng.shuffle(requests)
        requests += [(keys[0], (True, True), None, None),
                     (keys[0], (False, False), ((0, 0),), 'far')]
        if witness == 2:
            requests += [(keys[0], (False, False), ((0, 0),), 'false_box')]
        if witness or rng.random() < .6:
            # A straight three-cell glyph crosses a virtual fence; its middle
            # or some cells being inside must not make the whole object count.
            partial = ((0, 0), (1, 0), (2, 0)) if offset[0] else ((0, 0), (0, 1), (0, 2))
            requests += [(keys[0], (False, False), partial, 'partial')]
        if witness == 2 or rng.random() < .65:
            requests += [(ignored, (True, False) if rng.random() < .5 else (False, True), None, None)]
        records = []
        for color, state, forced, special in requests:
            obj = _place(rng, grid, blocked, masks, color, state, forced, special)
            if obj is None:
                break
            records.append(obj)
        if len(records) != len(requests):
            continue
        output = panel.copy()
        for r, color, loss, gain in zip(rows, keys, losses, gains):
            for side, n in enumerate((loss, gain)):
                for i in range(1, n+1):
                    output[r, mid+(-2 if side == 0 else 2)*i] = color
        gap = int(rng.integers(1, min(3, 31-grid.shape[0]-ph)))
        height = grid.shape[0]+gap+ph
        width = max(grid.shape[1], pw)
        height += int(rng.integers(0, min(3, 31-height)))
        width += int(rng.integers(0, min(3, 31-width)))
        canvas = np.full((height, width), bg, np.int8)
        sc = int(rng.integers(width-grid.shape[1]+1))
        pc = int(rng.integers(width-pw+1))
        if rng.random() < .5:
            sr, pr = 0, height-ph
        else:
            pr, sr = 0, height-grid.shape[0]
        canvas[sr:sr+grid.shape[0], sc:sc+grid.shape[1]] = grid
        canvas[pr:pr+ph, pc:pc+pw] = panel
        meta = _inspect(canvas)
        _require(np.array_equal(meta['expected'], output), 'object-list label disagreement')
        alternatives = ('area', 'four', 'any') if witness == 1 else ('box',) if witness == 2 else ()
        if any(_counterfactual_counts(meta, alt) == meta['counts'] for alt in alternatives):
            continue
        # Scene and chart transform together. Requested offset is local here.
        return dict(input=canvas, output=output)
    raise RuntimeError('bounded placement failed for '+repr((k, offset, mode, witness, extrapolate)))


def _check_counts(num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(name+' is outside its legal integer range')
    _require(num_train > num_test, 'num_train must exceed num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)) and seed >= 0,
             'seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    regime = int(rng.integers(3))  # displacement, count, or category transfer
    held = OFFSETS[int(rng.integers(8))]
    train_offsets = [d for d in OFFSETS if d != held]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, n in (('train', num_train), ('test', num_test)):
        for i in range(n):
            witness = i+1 if split == 'train' and i < 2 else 0
            k = (2 if witness == 1 else 1 if witness == 2 else
                 3 if split == 'test' and regime == 2 else int(rng.integers(1, 3 if regime == 2 else 4)))
            mode = ('rectangle' if witness == 1 else 'clipped_elbow' if witness == 2 else
                    str(rng.choice(('rectangle', 'clipped', 'diamond', 'elbow', 'notch'))))
            desired = held if split == 'test' and regime == 0 else train_offsets[int(rng.integers(7))] if regime == 0 else OFFSETS[int(rng.integers(8))]
            turns, reflect = int(rng.integers(4)), bool(rng.integers(2))
            local = next(d for d in OFFSETS if _vector(d, turns, reflect) == desired)
            for retry in range(12):
                pair = _make_pair(rng, k, local, mode, witness, split == 'test' and regime == 1)
                pair = {role: _oriented(a, turns, reflect).copy() for role, a in pair.items()}
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('distinct input construction failed')
            seen.add(_key(pair['input']))
            # Compare the object-list labels with visible topology/segmentation.
            _require(np.array_equal(solve(pair['input']), pair['output']), 'construction/parser disagreement')
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _counterfactual_counts(meta, mode):
    result = {c: [0, 0] for q, c in meta['keys']}
    if mode == 'four':
        shape = meta['outside'].shape
        sm = _inside_mask(shape, meta['polygon'])
        dr, dc = meta['offset']
        tm = _inside_mask(shape, [(r+dr, c+dc) for r, c in meta['polygon']])
    for obj in meta['objects']:
        if obj['color'] not in result:
            continue
        if mode == 'four':
            for fragment in _components(obj['cells'], False):
                state = all(sm[p] for p in fragment), all(tm[p] for p in fragment)
                if state in ((True, False), (False, True)):
                    result[obj['color']][int(state[1])] += 1
            continue
        state = obj['bbox_state'] if mode == 'box' else obj['any_state'] if mode == 'any' else obj['state']
        if state in ((True, False), (False, True)):
            amount = len(obj['cells']) if mode == 'area' else 1
            result[obj['color']][int(state[1])] += amount
    return result


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'bad puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'splits must be lists')
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'bad pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            meta = _inspect(inp)
            _require(np.array_equal(out, meta['expected']), 'wrong census or crop')
            _require(sum(sum(v) for v in meta['counts'].values()) > 0, 'no moving keyed objects')
            if split == 'train':
                training.append(meta)
    # These are observed witnesses, reconstructed from inputs and outputs.
    for alt in ('box', 'any', 'area', 'four'):
        _require(any(_counterfactual_counts(m, alt) != m['counts'] for m in training),
                 'training fails to distinguish '+alt)
    _require(any(a != b for m in training for a, b in m['counts'].values()), 'no source/companion-side evidence')
    _require(any(0 in v for m in training for v in m['counts'].values()), 'no zero-side evidence')
    _require(len({len(m['keys']) for m in training}) >= 2, 'no category-count variation')
    _require(any(o['color'] not in m['counts'] and o['state'] in ((True, False), (False, True))
                 for m in training for o in m['objects']), 'no ignored moving object')
    _require(any(o['state'] == (True, True) for m in training for o in m['objects']), 'no stable interior control')
    _require(any(o['state'] == (False, False) for m in training for o in m['objects']), 'no exterior control')


def self_test():
    """Bounded structural, geometric, ablation, and diversity tests on 42 seeds."""
    puzzles = [generate_puzzle(seed) for seed in HOST_SEEDS]
    metas = [_inspect(p['input']) for q in puzzles for s in ('train', 'test') for p in q[s]]
    assert len({_key(q['train'][0]['input']) for q in puzzles}) == 42
    assert len({m['bounds'] for m in metas}) >= 35
    assert len({m['expected'].shape for m in metas}) >= 25
    assert len({len(m['vertices']) for m in metas}) >= 3
    assert {m['offset'] for m in metas} == set(OFFSETS)
    assert {m['field'] for m in metas} == set(range(10))
    assert {m['bg'] for m in metas} == set(range(10))
    assert {len(m['keys']) for m in metas} == {1, 2, 3}
    assert any(max(max(v) for v in m['counts'].values()) >= 5 for m in metas)
    assert len({_normalize(o['cells']) for m in metas for o in m['objects']}) >= 30
    assert any(m['chords'] > 0 for m in metas)
    # Test translation and strict boundary at known lattice points.
    box = [(1, 1), (1, 7), (7, 7), (7, 1)]
    mask = _inside_mask((10, 10), box)
    assert mask[2, 2] and not mask[1, 2] and not mask[0, 2]
    # The supplied verifier exhausts all legal explicit combinations. Here
    # retain representative boundaries so the 42-seed test stays bounded.
    for nt, nq in ((2, 1), (4, 2), (7, 3), (5, 4), (10, 1), (10, 4)):
        q = generate_puzzle(73, nt, nq)
        assert len(q['train']) == nt and len(q['test']) == nq
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    a, b = generate_puzzle(817), generate_puzzle(817)
    assert all(_key(x[r]) == _key(y[r]) for s in a for x, y in zip(a[s], b[s]) for r in x)
    # Equivariance includes panel crop orientation and all marker directions.
    p = puzzles[0]['test'][0]
    for turns in range(4):
        for reflect in (False, True):
            assert np.array_equal(solve(_oriented(p['input'], turns, reflect)),
                                  _oriented(p['output'], turns, reflect))
    bad = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in a[s]] for s in a}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('corrupt output accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds, count boundaries, geometry and ablations)')
