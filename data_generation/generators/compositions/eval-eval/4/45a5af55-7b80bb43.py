"""Gated contour weaving: dye a repaired drawing by topological nesting depth.

The color strip is read vertically, top to bottom. Its final sample duplicates
the preceding one. Background/ink are the two most frequent input colors;
everything else belongs to the rectangular color strip. No generation state is
used by solve. Construction uses polygon containment; solving uses flood-filled
regions and adjacency across ink, providing two different authored checks.
"""

from collections import deque
import copy
import numpy as np


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((a >= 0) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _at(a, r, c):
    return 0 <= r < a.shape[0] and 0 <= c < a.shape[1] and bool(a[r, c])


def _strong(mask):
    """Ink supported by a horizontal or vertical run of at least three cells."""
    out = np.zeros_like(mask)
    for m, s in ((mask, out), (mask.T, out.T)):
        triples = m[:, :-2] & m[:, 1:-1] & m[:, 2:]
        for k in range(3):
            s[:, k:k + triples.shape[1]] |= triples
    return out


def _gaps(mask):
    result = []
    for m, vertical in ((mask, False), (mask.T, True)):
        g = m[:, :-4] & m[:, 4:]
        g &= ~(m[:, 1:-3] | m[:, 2:-2] | m[:, 3:-1])
        for y, x in np.argwhere(g):
            r, c = (int(x), int(y)) if vertical else (int(y), int(x))
            result.append((r, c, int(vertical), int(not vertical)))
    return result


def _repair_evidence(mask):
    supported = _strong(mask)
    result = []
    for r, c, dr, dc in _gaps(mask):
        if not (_at(mask, r-dr, c-dc) or _at(mask, r+5*dr, c+5*dc)):
            continue
        matches = []
        for end in (0, 4):
            ar, ac = r + end*dr, c + end*dc
            toward = 1 if end == 0 else -1
            for side in (-1, 1):
                sr, sc = toward*dr + side*dc, toward*dc + side*dr
                spur = tuple((ar + k*sr, ac + k*sc) for k in (1, 2))
                if (all(_at(mask, rr, cc) and not supported[rr, cc]
                        for rr, cc in spur)
                        and not _at(mask, ar + 3*sr, ac + 3*sc)):
                    matches.append((end, side, spur))
        if matches:
            result.append(((r, c, dr, dc), tuple(matches)))
    return result


def _reconstruct(mask):
    evidence = _repair_evidence(mask)
    filled = mask.copy()
    for (r, c, dr, dc), matches in evidence:
        assert len(matches) == 1, "ambiguous spur attachment"
        for k in (1, 2, 3):
            filled[r + k*dr, c + k*dc] = True
    # In this family every retained outline cell has straight support. The only
    # unsupported cells are short diagonal spurs or detached diagonal clutter.
    return _strong(filled), evidence


def _short_diagonals(mask):
    """Erased components must really be separate straight diagonal fragments."""
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    components = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nxt = (r+dr, c+dc)
                    if (dr or dc) and nxt in remaining:
                        remaining.remove(nxt)
                        todo.append(nxt)
        cells.sort()
        if (len(cells) not in (2, 3)
                or any(cells[i+1][0]-cells[i][0] != 1 for i in range(len(cells)-1))
                or len({cells[i+1][1]-cells[i][1] for i in range(len(cells)-1)}) != 1
                or abs(cells[1][1]-cells[0][1]) != 1):
            return None
        components.append(tuple(cells))
    return components


def _parse(grid):
    a = _grid(grid)
    colors, counts = np.unique(a, return_counts=True)
    assert len(colors) >= 4
    order = np.argsort(counts)[::-1]
    assert counts[order[0]] > counts[order[1]] > counts[order[2]]
    bg, ink = map(int, colors[order[:2]])
    marker = (a != bg) & (a != ink)
    rr, cc = np.where(marker)
    r0, r1, c0, c1 = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())
    strip = a[r0:r1+1, c0:c1+1]
    assert 3 <= len(strip) <= 5 and 1 <= strip.shape[1] <= 3
    assert marker.sum() == strip.size and np.all(marker[r0:r1+1, c0:c1+1])
    assert np.all(strip == strip[:, :1])
    code = strip[:, 0].copy()
    assert code[-2] == code[-1] and len(np.unique(code)) >= 2
    mask = a == ink
    assert not mask[0].any() and not mask[-1].any()
    assert not mask[:, 0].any() and not mask[:, -1].any()
    # The strip is outside all contours, on either side of their bounding box.
    _, wc = np.where(mask)
    assert c1 < int(wc.min()) or c0 > int(wc.max())
    return a, bg, ink, code, mask, (r0, r1, c0, c1)


def _region_depths(walls):
    """Count barriers from exterior using the graph of four-connected air regions.

    A gate makes its two sides one vertex. A sealed contour gives an edge
    between two vertices. BFS distance is the number of enclosing sealed walls.
    This does not use bounding boxes or any polygon metadata.
    """
    h, w = walls.shape
    labels = np.full((h, w), -1, dtype=np.int16)
    count = 0
    for row in range(h):
        for col in range(w):
            if walls[row, col] or labels[row, col] >= 0:
                continue
            labels[row, col] = count
            q = deque([(row, col)])
            while q:
                r, c = q.popleft()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r+dr, c+dc
                    if (0 <= nr < h and 0 <= nc < w and not walls[nr, nc]
                            and labels[nr, nc] < 0):
                        labels[nr, nc] = count
                        q.append((nr, nc))
            count += 1
    graph = [set() for _ in range(count)]
    for r, c in np.argwhere(walls):
        r, c = int(r), int(c)
        for dr, dc in ((1, 0), (0, 1)):
            if 0 <= r-dr and r+dr < h and 0 <= c-dc and c+dc < w:
                u, v = int(labels[r-dr, c-dc]), int(labels[r+dr, c+dc])
                if u >= 0 and v >= 0 and u != v:
                    graph[u].add(v)
                    graph[v].add(u)
    distance = np.full(count, -1, dtype=np.int16)
    outside = int(labels[0, 0])
    assert outside >= 0
    distance[outside] = 0
    q = deque([outside])
    while q:
        u = q.popleft()
        for v in sorted(graph[u]):
            if distance[v] < 0:
                distance[v] = distance[u] + 1
                q.append(v)
    assert np.all(distance >= 0), "unsupported thick/intersecting boundary"
    depth = np.zeros((h, w), dtype=np.int8)
    air = labels >= 0
    depth[air] = distance[labels[air]]
    for r, c in np.argwhere(walls):
        neighbor_labels = labels[max(0, r-1):min(h, r+2), max(0, c-1):min(w, c+2)]
        adjacent = neighbor_labels[neighbor_labels >= 0]
        assert len(adjacent)
        # The outline belongs to the deeper region, including at convex corners.
        depth[r, c] = max(distance[adjacent])
    return depth, distance, graph


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solver; the color strip and geometry determine every cell."""
    _, _, _, code, mask, _ = _parse(input_grid)
    walls, _ = _reconstruct(mask)
    depth, _, _ = _region_depths(walls)
    assert int(depth.max()) < len(code)-1, "color strip too short"
    return code[depth].astype(np.int8)


def _minimum(levels):
    return 9 + 6*(levels-1)


def _size(box):
    r0, c0, r1, c1 = box
    return r1-r0+1, c1-c0+1


def _inset(box, n):
    r0, c0, r1, c1 = box
    return r0+n, c0+n, r1-n, c1-n


def _split_box(rng, box, first_min, second_min):
    """Split along a feasible axis with three blank positions between boxes."""
    h, w = _size(box)
    axes = [i for i, n in enumerate((h, w)) if n >= first_min + second_min + 3]
    if not axes or min(h, w) < max(first_min, second_min):
        return None
    axis = int(_pick(rng, axes))
    size = (h, w)[axis]
    cut = int(rng.integers(first_min, size-second_min-3+1))
    r0, c0, r1, c1 = box
    if axis == 0:
        parts = [(r0, c0, r0+cut-1, c1), (r0+cut+3, c0, r1, c1)]
    else:
        parts = [(r0, c0, r1, c0+cut-1), (r0, c0+cut+3, r1, c1)]
    return parts


def _polygon_masks(shape, box, notch=None):
    """A rectangle or a rectangle with one corner replaced by an inward step."""
    r0, c0, r1, c1 = box
    h, w = _size(box)
    boundary = np.zeros((h, w), dtype=bool)
    area = np.ones((h, w), dtype=bool)
    if notch is None:
        vertices = [(0, 0), (0, w-1), (h-1, w-1), (h-1, 0)]
    else:
        nh, nw, flip_r, flip_c = notch
        vertices = [(0, 0), (0, w-1-nw), (nh, w-1-nw),
                    (nh, w-1), (h-1, w-1), (h-1, 0)]
        area[:nh, w-nw:] = False
    for (ar, ac), (br, bc) in zip(vertices, vertices[1:] + vertices[:1]):
        boundary[min(ar, br):max(ar, br)+1, min(ac, bc):max(ac, bc)+1] = True
    if notch is not None:
        if flip_r:
            boundary, area = boundary[::-1], area[::-1]
        if flip_c:
            boundary, area = boundary[:, ::-1], area[:, ::-1]
    walls = np.zeros(shape, dtype=bool)
    filled = np.zeros(shape, dtype=bool)
    walls[r0:r1+1, c0:c1+1] = boundary
    filled[r0:r1+1, c0:c1+1] = area
    return walls, filled


def _node(rng, shape, region, levels, richness, nodes, parent=-1, force_fork=False):
    minimum = _minimum(levels)
    rh, rw = _size(region)
    assert min(rh, rw) >= minimum
    r0, c0, r1, c1 = region
    if force_fork:
        # A single parent with two children; a long side of 27 gives a 21-cell
        # interior, enough for 9 + 3 + 9. The short side is independently sampled.
        h = rh if rh >= 27 else int(rng.integers(minimum, rh+1))
        w = rw if rw >= 27 else int(rng.integers(minimum, rw+1))
    else:
        h, w = int(rng.integers(minimum, rh+1)), int(rng.integers(minimum, rw+1))
    top, left = int(rng.integers(r0, r1-h+2)), int(rng.integers(c0, c1-w+2))
    box = (top, left, top+h-1, left+w-1)
    safe = _inset(box, 3)
    notch = None
    if not force_fork and min(h, w) >= 13 and rng.random() < (0.15, 0.45, 0.7)[richness]:
        nh, nw = int(rng.integers(3, min(6, h-7)+1)), int(rng.integers(3, min(6, w-7)+1))
        fr, fc = bool(rng.integers(2)), bool(rng.integers(2))
        # Place children completely in one arm of the stepped polygon. Its other
        # arm remains free space; this changes region geometry, not layer order.
        below = (top+nh+3, left+3, top+h-4, left+w-4)
        beside = (top+3, left+3, top+h-4, left+w-nw-4)
        possible = [b for b in (below, beside)
                    if levels == 1 or min(_size(b)) >= _minimum(levels-1)]
        if possible:
            notch = (nh, nw, fr, fc)
            safe = _pick(rng, possible)
            if fr:
                safe = (2*top+h-1-safe[2], safe[1], 2*top+h-1-safe[0], safe[3])
            if fc:
                safe = (safe[0], 2*left+w-1-safe[3], safe[2], 2*left+w-1-safe[1])
    boundary, area = _polygon_masks(shape, box, notch)
    index = len(nodes)
    nodes.append({'boundary': boundary, 'area': area, 'parent': parent,
                  'box': box, 'notch': notch is not None})
    if levels > 1:
        fork = _split_box(rng, safe, _minimum(levels-1), 9)
        use_fork = force_fork or (fork is not None and rng.random() < 0.35*richness)
        if use_fork:
            assert fork is not None
            # Reflection changes which sibling carries the deeper subtree.
            if rng.integers(2):
                if _size(fork[1])[0] >= _minimum(levels-1) and _size(fork[1])[1] >= _minimum(levels-1):
                    fork = fork[::-1]
            _node(rng, shape, fork[0], levels-1, richness, nodes, index)
            _node(rng, shape, fork[1], 1, richness, nodes, index)
        else:
            _node(rng, shape, safe, levels-1, richness, nodes, index)
    return index


def _cut_candidates(mask, owner):
    result = []
    for m, o, vertical in ((mask, owner, False), (mask.T, owner.T, True)):
        valid = np.logical_and.reduce([m[:, k:m.shape[1]-4+k] & o[:, k:o.shape[1]-4+k]
                                      for k in range(5)])
        for y, x in np.argwhere(valid):
            r, c = (int(x), int(y)) if vertical else (int(y), int(x))
            dr, dc = int(vertical), int(not vertical)
            if _at(mask, r-dr, c-dc) or _at(mask, r+5*dr, c+5*dc):
                result.append((r, c, dr, dc))
    return result


def _cut(mask, gap):
    r, c, dr, dc = gap
    trial = mask.copy()
    for k in (1, 2, 3):
        trial[r+k*dr, c+k*dc] = False
    return trial


def _gate(rng, mask, owner):
    candidates = _cut_candidates(mask, owner)
    for i in rng.permutation(len(candidates)):
        trial = _cut(mask, candidates[int(i)])
        if np.array_equal(_strong(trial), trial) and not _repair_evidence(trial):
            return trial
    raise ValueError('no supported gate placement')


def _damage(rng, inp, expected, owner, prior, forbidden):
    candidates = _cut_candidates(inp, owner & expected)
    for i in rng.permutation(len(candidates)):
        r, c, dr, dc = gap = candidates[int(i)]
        trial = _cut(inp, gap)
        # Intact portions must retain the same three-cell support as the input
        # convention. This rules out weak contour bits masquerading as clutter.
        base_trial = _cut(expected, gap)
        if not np.array_equal(_strong(base_trial), base_trial):
            continue
        choices = [(end, side) for end in (0, 4) for side in (-1, 1)]
        rng.shuffle(choices)
        for end, side in choices:
            toward = 1 if end == 0 else -1
            ar, ac = r + end*dr, c + end*dc
            sr, sc = toward*dr + side*dc, toward*dc + side*dr
            cells = [(ar+k*sr, ac+k*sc) for k in (1, 2)]
            if not all(0 < rr < inp.shape[0]-1 and 0 < cc < inp.shape[1]-1
                       and not inp[rr, cc] and not expected[rr, cc] and not forbidden[rr, cc]
                       for rr, cc in cells):
                continue
            candidate = trial.copy()
            for rr, cc in cells:
                candidate[rr, cc] = True
            try:
                rebuilt, evidence = _reconstruct(candidate)
            except AssertionError:
                continue
            if (len(evidence) == prior+1 and np.array_equal(rebuilt, expected)
                    and _short_diagonals(candidate & ~expected) is not None):
                return candidate
    raise ValueError('no unambiguous repair placement')


def _scatter(rng, inp, expected, forbidden, count):
    for _ in range(count):
        for _ in range(80):
            length = int(rng.integers(2, 4))
            r, c = int(rng.integers(1, inp.shape[0]-1)), int(rng.integers(1, inp.shape[1]-1))
            dr, dc = int(_pick(rng, [-1, 1])), int(_pick(rng, [-1, 1]))
            cells = [(r+k*dr, c+k*dc) for k in range(length)]
            if not all(0 < rr < inp.shape[0]-1 and 0 < cc < inp.shape[1]-1
                       and not inp[rr, cc] and not expected[rr, cc] and not forbidden[rr, cc]
                       for rr, cc in cells):
                continue
            candidate = inp.copy()
            for rr, cc in cells:
                candidate[rr, cc] = True
            try:
                rebuilt, ev = _reconstruct(candidate)
            except AssertionError:
                continue
            if (np.array_equal(rebuilt, expected)
                    and len(ev) == len(_repair_evidence(inp))
                    and _short_diagonals(candidate & ~expected) is not None
                    and np.count_nonzero(candidate & ~_strong(candidate))
                    == np.count_nonzero(inp & ~_strong(inp)) + length):
                inp = candidate
                break
    return inp


def _make_pair(rng, params, kind, levels):
    richness = params['richness']
    fork = kind == 'fork'
    thickness = int(rng.integers(1, 4))
    gutter = int(rng.integers(1, 3))
    strip_space = thickness + gutter
    min_side = _minimum(levels)
    h = int(rng.integers(max(min_side+2, 29 if fork else 1), 31))
    w = int(rng.integers(min_side + 2 + strip_space, 31))
    shape = (h, w)
    nodes = []
    region = (1, strip_space+1, h-2, w-2)
    _node(rng, shape, region, levels, richness, nodes, force_fork=fork)
    # A separate root can arise from a recursive split of free scene space in
    # ordinary examples, without a fixed reserved quadrant or a second palette.
    if kind == 'free' and levels <= 2 and rng.random() < 0.3:
        spare = _split_box(rng, region, _minimum(levels), 9)
        if spare is not None:
            nodes = []
            _node(rng, shape, spare[0], levels, richness, nodes)
            _node(rng, shape, spare[1], 1, richness, nodes)
    full = np.logical_or.reduce([n['boundary'] for n in nodes])
    assert np.array_equal(_strong(full), full)
    assert not (full[:-1, :-1] & full[1:, :-1] & full[:-1, 1:] & full[1:, 1:]).any()
    # Polygons must be laminar and separated. The area check catches step/corner
    # choices that would protrude into a child's reserved space.
    for i, node in enumerate(nodes):
        if node['parent'] >= 0:
            assert np.all(~node['area'] | nodes[node['parent']]['area'])
        for other in nodes[i+1:]:
            assert not (node['boundary'] & other['boundary']).any()
    closed = [True]*len(nodes)
    if kind in ('gate', 'mixed_gate'):
        closed[0] = False
    elif kind == 'free':
        closed = [bool(rng.random() < 0.65) for _ in nodes]
        if not any(closed):
            closed[int(rng.integers(len(nodes)))] = True
    elif kind == 'fork':
        # Both sealed siblings and gated siblings occur. Otherwise every fork
        # would collapse to a chain in the region-adjacency graph.
        children = [i for i, n in enumerate(nodes) if n['parent'] == 0]
        if rng.random() < 0.5:
            closed[int(_pick(rng, children))] = False
    expected = full.copy()
    for i, sealed in enumerate(closed):
        if not sealed:
            gate_count = 2 if kind == 'mixed_gate' and i == 0 else 1
            for _ in range(gate_count):
                expected = _gate(rng, expected, nodes[i]['boundary'])
    forbidden = np.zeros(shape, dtype=bool)
    # Keep all noise, not only the outlines, out of the exterior strip lane.
    forbidden[:, :strip_space+1] = True
    sr = int(rng.integers(0, h-(levels+2)+1))
    sc = int(rng.integers(0, gutter+1))
    forbidden[sr:sr+levels+2, sc:sc+thickness] = True
    inp_mask = expected.copy()
    sealed_ids = [i for i, value in enumerate(closed) if value]
    required = int(_pick(rng, sealed_ids))
    damaged_ids = [required]
    for i in rng.permutation(len(nodes)):
        i = int(i)
        if i != required and rng.random() < (0.15, 0.4, 0.65)[richness]:
            damaged_ids.append(i)
    if kind == 'mixed_gate' and 0 not in damaged_ids:
        damaged_ids.append(0)
    repairs = 0
    for i in damaged_ids:
        try:
            inp_mask = _damage(rng, inp_mask, expected, nodes[i]['boundary'], repairs, forbidden)
            repairs += 1
        except ValueError:
            if i == required or (kind == 'mixed_gate' and i == 0):
                raise
    if richness and rng.random() < 0.5:
        try:
            i = int(_pick(rng, sealed_ids))
            inp_mask = _damage(rng, inp_mask, expected, nodes[i]['boundary'], repairs, forbidden)
        except ValueError:
            pass
    inp_mask = _scatter(rng, inp_mask, expected, forbidden,
                        int(rng.integers(1, richness+3)))
    if params['palette_mode'] == 'fixed':
        bg, ink = params['palette']
    elif params['palette_mode'] == 'swap':
        bg, ink = params['palette'][::int(_pick(rng, [-1, 1]))]
    else:
        bg, ink = map(int, rng.choice(10, 2, replace=False))
    palette = [c for c in range(10) if c not in (bg, ink)]
    palette = list(map(int, rng.permutation(palette)))
    if kind == 'repeat':
        # Repeated slots must occupy real depth before a later different color.
        slots = [palette[0], palette[0], palette[1]]
    elif kind == 'gate':
        slots = palette[:levels+1]
    else:
        k = int(rng.integers(2, levels+2))
        slots = [int(_pick(rng, palette[:k])) for _ in range(levels+1)]
        slots[0], slots[-1] = palette[0], palette[1]
    code = np.array(slots + [slots[-1]], dtype=np.int8)
    inp = np.where(inp_mask, ink, bg).astype(np.int8)
    inp[sr:sr+len(code), sc:sc+thickness] = code[:, None]
    # Label construction is polygon containment, independent of the solver's
    # flood-fill and shortest paths. Open outlines contribute zero everywhere.
    depth = np.zeros(shape, dtype=np.int8)
    for node, sealed in zip(nodes, closed):
        if sealed:
            depth += node['area']
    out = code[depth]
    if rng.integers(2):
        inp, out = inp[:, ::-1].copy(), out[:, ::-1].copy()
    pair = {'input': inp, 'output': out.astype(np.int8)}
    _analyze_pair(pair)
    return pair


def _count(value, lo, hi, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n = _count(num_train, 2, 10, 'num_train')
    t = _count(num_test, 1, 4, 'num_test')
    if n <= t:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    params = {'richness': int(rng.integers(3)),
              'palette_mode': str(rng.choice(['fixed', 'swap', 'varied', 'varied'])),
              'palette': tuple(map(int, rng.choice(10, 2, replace=False)))}
    shift = str(rng.choice(['deeper', 'fork', 'mixed_gate']))
    result, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', n), ('test', t)):
        for i in range(count):
            if split == 'train' and i < 2:
                kind, levels = ('repeat' if i == 0 else 'gate'), 2
            elif split == 'train':
                kind, levels = 'free', int(rng.integers(1, 3))
            else:
                # The first deeper test must exercise a new ACTUAL depth, not
                # merely an extra drawn outline whose gate cancels that depth.
                kind = ('deep' if i == 0 else 'free') if shift == 'deeper' else shift
                levels = 3 if shift == 'deeper' else 2
            for _ in range(100):
                try:
                    pair = _make_pair(rng, params, kind, levels)
                    key = _key(pair['input'])
                    if key not in seen:
                        seen.add(key)
                        result[split].append(pair)
                        break
                except (AssertionError, ValueError):
                    continue
            else:
                raise RuntimeError(f'could not construct {kind} at depth {levels}, seed {seed}, pair {split}/{i}')
    rng.shuffle(result['train'])
    validate_generated_puzzle(result)
    return result


def _paint_alternative(mask, code, fill_gaps=False, repair=False):
    if repair:
        walls, _ = _reconstruct(mask)
    else:
        walls = _strong(mask)
    if fill_gaps:
        for r, c, dr, dc in _gaps(walls):
            if _at(walls, r-dr, c-dc) or _at(walls, r+5*dr, c+5*dc):
                for k in (1, 2, 3):
                    walls[r+k*dr, c+k*dc] = True
        # This deliberately wrong policy can bridge unrelated outlines and make
        # thick ink. Evaluate it with generic cell-crossing costs, rather than
        # rejecting a valid family member because the WRONG policy left domain.
        depth = _crossing_cost_depth(walls)
        return code[np.minimum(depth, len(code)-1)]
    depth, _, _ = _region_depths(walls)
    return code[np.minimum(depth, len(code)-1)]


def _crossing_cost_depth(walls):
    """Total diagnostic for overfilled drawings; never used by solve or labels."""
    h, w = walls.shape
    cost = np.full((h, w), 1000, dtype=np.int16)
    q = deque()
    for r in range(h):
        for c in range(w):
            if r in (0, h-1) or c in (0, w-1):
                cost[r, c] = int(walls[r, c])
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((1,0), (-1,0), (0,1), (0,-1)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w:
                step = int(walls[nr, nc])
                candidate = cost[r, c] + step
                if candidate < cost[nr, nc]:
                    cost[nr, nc] = candidate
                    if step:
                        q.append((nr, nc))
                    else:
                        q.appendleft((nr, nc))
    depth = cost.copy()
    for r, c in np.argwhere(walls):
        nearby_walls = walls[max(0,r-1):min(h,r+2), max(0,c-1):min(w,c+2)]
        nearby_cost = cost[max(0,r-1):min(h,r+2), max(0,c-1):min(w,c+2)]
        if (~nearby_walls).any():
            depth[r, c] = nearby_cost[~nearby_walls].max()
    return depth


def _analyze_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
    a, bg, ink, code, mask, strip = _parse(pair['input'])
    out = _grid(pair['output'])
    assert out.shape == a.shape
    walls, evidence = _reconstruct(mask)
    assert evidence and np.array_equal(_strong(walls), walls)
    fragments = _short_diagonals(mask & ~walls)
    assert fragments is not None
    assert all(matches[0][2] in fragments or matches[0][2][::-1] in fragments
               for _, matches in evidence)
    assert not (walls[:-1, :-1] & walls[1:, :-1] & walls[:-1, 1:] & walls[1:, 1:]).any()
    depth, distances, graph = _region_depths(walls)
    assert 1 <= depth.max() <= 3 and depth.max() < len(code)-1
    assert np.array_equal(out, code[depth]), 'incorrect region, wall, or strip recoloring'
    assert not np.isin(out, [bg, ink]).any()
    assert not np.array_equal(out, _paint_alternative(mask, code)), 'repair has no effect on dye'
    # At least one real region boundary changes color, rather than a palette
    # that erases every distinction in this particular scene.
    assert len(np.unique(out)) >= 2
    all_filled = _paint_alternative(mask, code, fill_gaps=True, repair=True)
    slots = code[:-1]
    runs = slots[np.r_[True, slots[1:] != slots[:-1]]]
    collapsed = runs[np.minimum(depth, len(runs)-1)]
    return {'shape': a.shape, 'roles': (bg, ink), 'code': tuple(map(int, slots)),
            'repairs': len(evidence), 'axes': tuple(sorted({g[2] for g, _ in evidence})),
            'end_bends': tuple((m[0][0], m[0][1]) for _, m in evidence),
            'depth': int(depth.max()), 'regions': len(distances),
            'gates': sum(_at(walls, r-dr, c-dc) or _at(walls, r+5*dr, c+5*dc)
                         for r, c, dr, dc in _gaps(walls)),
            'gate_matters': not np.array_equal(out, all_filled),
            'repeat_matters': not np.array_equal(out, collapsed),
            'strip_width': strip[3]-strip[2]+1, 'strip_left': strip[2] < a.shape[1]//2,
            'structure': (a.shape, mask.tobytes(), strip),
            'clutter_cells': int(np.count_nonzero(mask & ~walls)) - 2*len(evidence),
            'branching': any(len(neighbors) >= 3 for neighbors in graph)}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    n, t = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= n <= 10 and 1 <= t <= 4 and n > t
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info = _analyze_pair(pair)
            k = _key(pair['input'])
            assert k not in seen, 'duplicate input'
            seen.add(k)
            if split == 'train':
                training.append(info)
    assert any(f['repeat_matters'] for f in training), 'no evidence against run compression'
    assert any(f['gate_matters'] for f in training), 'no evidence against filling every gap'


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def self_test():
    infos, signatures, roles, colors = [], set(), set(), set()
    deeper = forked = gated_tests = 0
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_key(puzzle))
        local = []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _analyze_pair(pair)
                infos.append(info)
                local.append(info)
                roles.add(info['roles'])
                colors.update(map(int, np.unique(pair['input'])))
                assert np.array_equal(solve(pair['input']), pair['output'])
                # Horizontal reflection preserves the fixed vertical code order.
                assert np.array_equal(solve(pair['input'][:, ::-1]), pair['output'][:, ::-1])
        test_info = local[-1]
        deeper += len(test_info['code']) == 4
        forked += test_info['branching']
        gated_tests += test_info['gates'] >= 2
        if first is None:
            first = puzzle
    assert len(signatures) == 42
    assert len({f['shape'] for f in infos}) >= 25
    assert len({f['structure'] for f in infos}) >= 190
    assert {f['depth'] for f in infos} == {1, 2, 3}
    assert {f['strip_width'] for f in infos} == {1, 2, 3}
    assert {f['strip_left'] for f in infos} == {False, True}
    assert set().union(*(set(f['axes']) for f in infos)) == {0, 1}
    assert set().union(*(set(f['end_bends']) for f in infos)) == {(0,-1), (0,1), (4,-1), (4,1)}
    assert len(roles) >= 25 and colors == set(range(10))
    assert max(f['repairs'] for f in infos) >= 3
    assert any(f['clutter_cells'] >= 3 for f in infos)
    assert deeper >= 4 and forked >= 1 and gated_tests >= 4
    assert _puzzle_key(first) == _puzzle_key(generate_puzzle(0))
    # Numeric colors have no semantics: permuting every color commutes with solve.
    perm = np.array([6, 3, 9, 0, 8, 2, 5, 1, 4, 7], dtype=np.int8)
    for pair in first['train'] + first['test']:
        assert np.array_equal(solve(perm[pair['input']]), perm[pair['output']])
    for n in range(2, 11):
        for t in range(1, min(4, n-1)+1):
            p = generate_puzzle(1100+n*10+t, n, t)
            assert len(p['train']) == n and len(p['test']) == t
    for n, t in [(1,1), (11,1), (4,0), (6,5), (2,2), (3,4),
                 (True,1), (4,False), (3.5,1), (None,1)]:
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Full-label checks reject a wrong region as well as a single wrong cell.
    bad = copy.deepcopy(first)
    pair = bad['test'][0]
    _, _, _, code, mask, _ = _parse(pair['input'])
    pair['output'] = _paint_alternative(mask, code)
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted deletion without repair')
    bad = copy.deepcopy(first)
    bad['train'][0]['output'][0, 0] = (int(bad['train'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted a wrong dye cell')
    return {'seeds': 42, 'pairs': len(infos), 'shapes': len({f['shape'] for f in infos}),
            'structures': len({f['structure'] for f in infos}), 'roles': len(roles),
            'deeper_tests': deeper, 'branching_tests': forked, 'multi_supported_break_tests': gated_tests}


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
