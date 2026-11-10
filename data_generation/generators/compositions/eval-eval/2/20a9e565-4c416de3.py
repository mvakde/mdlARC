"""Size-ranked corner garlands. No I/O, parent imports, or latent solver state."""
from collections import deque
import hashlib
import numpy as np


DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(points):
    remaining = set(points)
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        component, todo = {p}, deque([p])
        while todo:
            r, c = todo.popleft()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    todo.append(q)
        result.append(component)
    return result


def _marker_cells(bounds):
    t, l, b, r = bounds
    return {(t, l), (t, l+1), (t+1, l), (b, r), (b, r-1), (b-1, r)}


def _markers(a):
    found = []
    for raw in np.unique(a):
        pos = np.argwhere(a == raw)
        if len(pos) != 6:
            continue
        t, l = map(int, pos.min(axis=0))
        b, r = map(int, pos.max(axis=0))
        bounds = t, l, b, r
        if b <= t or r <= l:
            continue
        points = set(map(tuple, pos))
        if points == _marker_cells(bounds) and len(_components(points)) == 2:
            found.append((int(raw), bounds))
    assert len(found) == 1, 'expected two unique disconnected L annotations'
    return found[0]


def _outline(rect):
    t, l, b, r = rect
    return ({(t, x) for x in range(l, r+1)} |
            {(b, x) for x in range(l, r+1)} |
            {(y, l) for y in range(t, b+1)} |
            {(y, r) for y in range(t, b+1)})


def _corner(rect, q):
    t, l, b, r = rect
    return ((t, l), (t, r), (b, l), (b, r))[q]


def _global(rect, q, point):
    y, x = _corner(rect, q)
    dy, dx = DIRS[q]
    u, v = point
    return y + dy*u, x + dx*v


def _local(rect, q, point):
    y, x = _corner(rect, q)
    dy, dx = DIRS[q]
    r, c = point
    return (r-y)*dy, (c-x)*dx


def _inside(p, bounds):
    t, l, b, r = bounds
    return t <= p[0] <= b and l <= p[1] <= r


def _beads(kernel, stride, count):
    return {(u+j*stride[0], v+j*stride[1])
            for j in range(count) for u, v in kernel}


def _parse(input_grid):
    """Recover geometry, template, growth and crop exclusively from observations."""
    a = _grid(input_grid)
    marker, bounds = _markers(a)
    vals, counts = np.unique(a, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1
    bg = int(vals[np.argmax(counts)])
    assert bg != marker
    t, l, b, r = bounds
    assert np.all(np.isin(a[t:b+1, l:r+1], (bg, marker))), 'crop must be erased'
    possibilities = []
    for fc in map(int, vals):
        if fc in (bg, marker):
            continue
        comps = _components(map(tuple, np.argwhere(a == fc)))
        if not 3 <= len(comps) <= 5:
            continue
        rects = []
        for comp in comps:
            ys, xs = zip(*comp)
            rect = min(ys), min(xs), max(ys), max(xs)
            if min(rect[2]-rect[0]+1, rect[3]-rect[1]+1) < 5 or comp != _outline(rect):
                break
            rects.append(rect)
        if len(rects) != len(comps):
            continue
        rects.sort(key=lambda x: (x[2]-x[0]+1)*(x[3]-x[1]+1))
        areas = [(x[2]-x[0]+1)*(x[3]-x[1]+1) for x in rects]
        if len(set(areas)) != len(areas):
            continue
        owners = [i for i, (y0, x0, y1, x1) in enumerate(rects)
                  if y0 < t <= b < y1 and x0 < l <= r < x1]
        if len(owners) != 1:
            continue
        # Rectangles and their interiors must be disjoint, with one blank-cell gap.
        if any(not (u[2]+1 < v[0] or v[2]+1 < u[0] or
                    u[3]+1 < v[1] or v[3]+1 < u[1])
               for i, u in enumerate(rects) for v in rects[:i]):
            continue
        possibilities.append((fc, rects, owners[0]))
    assert len(possibilities) == 1, 'frame color/geometry is ambiguous'
    frame, rects, query = possibilities[0]
    scene_pixels = set().union(*(_outline(rect) for rect in rects)) | _marker_cells(bounds)
    observations = []
    for i, rect in enumerate(rects):
        y0, x0, y1, x1 = rect
        pixels = {(y, x) for y in range(y0+1, y1) for x in range(x0+1, x1)
                  if int(a[y, x]) not in (bg, marker)}
        assert pixels and all(a[p] != frame for p in pixels)
        colors = {int(a[p]) for p in pixels}
        assert len(colors) == 1, 'one local accent per frame'
        options = []
        for q in range(4):
            points = {_local(rect, q, p) for p in pixels}
            # Each ornament starts one cell inward on both axes. The opposing
            # edges retain at least one blank interior row/column in full copies.
            if min(u for u, v in points) == min(v for u, v in points) == 1:
                options.append((q, points))
        assert len(options) == 1, 'corner association is ambiguous'
        q, points = options[0]
        scene_pixels |= pixels
        observations.append(dict(rect=rect, q=q, color=colors.pop(), points=points))
    assert scene_pixels == set(map(tuple, np.argwhere(a != bg))), 'unexplained pixels'
    kernels, strides, known_counts = set(), set(), {}
    for i, obs in enumerate(observations):
        if i == query:
            continue
        comps = _components(obs['points'])
        comps.sort(key=lambda c: (min(p[0] for p in c), min(p[1] for p in c)))
        starts = []
        for comp in comps:
            u0, v0 = min(p[0] for p in comp), min(p[1] for p in comp)
            starts.append((u0, v0))
            kernels.add(frozenset((u-u0+1, v-v0+1) for u, v in comp))
        assert starts[0] == (1, 1)
        for j in range(1, len(starts)):
            strides.add((starts[j][0]-starts[j-1][0], starts[j][1]-starts[j-1][1]))
        known_counts[i] = len(comps)
    assert len(kernels) == 1 and len(strides) == 1, 'copies disagree or stride unobserved'
    kernel, stride = next(iter(kernels)), next(iter(strides))
    kh, kw = max(u for u, v in kernel), max(v for u, v in kernel)
    assert 2 <= kh <= 3 and 2 <= kw <= 3 and len(kernel) >= 3
    assert any(u >= 2 and v >= 2 for u, v in kernel)
    assert stride[0] >= kh and stride[1] >= kw
    indices = sorted(known_counts)
    i, j = indices[:2]
    difference = known_counts[j]-known_counts[i]
    assert difference % (j-i) == 0
    delta = difference // (j-i)
    base = known_counts[i]-i*delta
    assert delta > 0 and base > 0
    assert all(known_counts[k] == base+k*delta for k in indices), 'nonlinear count progression'
    bead_counts = [base+k*delta for k in range(len(rects))]
    for i, obs in enumerate(observations):
        complete = _beads(kernel, stride, bead_counts[i])
        h, w = obs['rect'][2]-obs['rect'][0]+1, obs['rect'][3]-obs['rect'][1]+1
        assert all(1 <= u < h-1 and 1 <= v < w-1 for u, v in complete)
        visible = {p for p in complete if not _inside(_global(obs['rect'], obs['q'], p), bounds)}
        assert obs['points'] == visible, 'clues are not the predicted unmasked ornament'
        if i == query:
            assert 0 < len(visible) < len(complete), 'query needs an actual partial ornament'
    return a, dict(background=bg, marker=marker, frame=frame, bounds=bounds,
                   observations=observations, query=query, kernel=kernel, stride=stride,
                   counts=bead_counts, base=base, delta=delta)


def _prediction(a, info, mode='correct'):
    t, l, b, r = info['bounds']
    out = np.full((b-t+1, r-l+1), info['background'], dtype=np.int8)
    obs = info['observations'][info['query']]
    count = info['counts'][info['query']]
    kernel = info['kernel']
    if mode == 'copy_largest':
        count = max(c for i, c in enumerate(info['counts']) if i != info['query'])
    elif mode == 'repeat_forever':
        count = 30
    elif mode == 'fragment_template':
        # Ignore B: the surviving first-row/column clue is the whole template.
        kernel = frozenset(obs['points'])
    elif mode == 'rotation':
        # Alternative convention maps adjacent corners by quarter turns. Anchor
        # its stencil at the lowest-area complete reference corner.
        ref = next(o for i, o in enumerate(info['observations']) if i != info['query'])
        if (ref['q'] in (1, 2)) != (obs['q'] in (1, 2)):
            kernel = frozenset((v, u) for u, v in kernel)
            stride = info['stride'][::-1]
        else:
            stride = info['stride']
        for p in _beads(kernel, stride, count):
            y, x = _global(obs['rect'], obs['q'], p)
            if t <= y <= b and l <= x <= r:
                out[y-t, x-l] = obs['color']
        return out
    for p in _beads(kernel, info['stride'], count):
        y, x = _global(obs['rect'], obs['q'], p)
        if t <= y <= b and l <= x <= r:
            out[y-t, x-l] = obs['color']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the inclusive marked crop; all parameters come from this input."""
    a, info = _parse(input_grid)
    return _prediction(a, info)


def _kernel(rng, kh, kw, asymmetric=False):
    domain = [(u, v) for u in range(1, kh+1) for v in range(1, kw+1)]
    candidates = []
    for mask in range(1, 1 << len(domain)):
        s = frozenset(p for j, p in enumerate(domain) if mask & (1 << j))
        if (len(s) >= 3 and len(_components(s)) == 1 and
                min(u for u, v in s) == min(v for u, v in s) == 1 and
                max(u for u, v in s) == kh and max(v for u, v in s) == kw and
                any(u >= 2 and v >= 2 for u, v in s) and
                (not asymmetric or s != frozenset((v, u) for u, v in s))):
            candidates.append(s)
    return candidates[int(rng.integers(len(candidates)))]


def _pack(rng, sizes):
    """Bounded free rectangle packing; no retained attempts or per-step traces."""
    for mask in range(1, 1 << len(sizes)):
        group = [s for i, s in enumerate(sizes) if mask & (1 << i)]
        for axis in (0, 1):
            # If every pair is too wide to share this axis, they must stack on
            # the other axis. Reject impossible cliques before random trials.
            if (all(a[axis]+b[axis]+1 > 30 for i, a in enumerate(group) for b in group[:i])
                    and sum(s[1-axis] for s in group)+len(group)-1 > 30):
                return None
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0]*sizes[i][1], reverse=True)
    for _ in range(24):
        placed = {}
        for i in order:
            h, w = sizes[i]
            found = False
            for _ in range(65):
                t = int(rng.integers(31-h))
                l = int(rng.integers(31-w))
                rect = t, l, t+h-1, l+w-1
                if all(rect[2]+1 < z[0] or z[2]+1 < rect[0] or
                       rect[3]+1 < z[1] or z[3]+1 < rect[1] for z in placed.values()):
                    placed[i] = rect
                    found = True
                    break
            if not found:
                break
        if len(placed) == len(sizes):
            return [placed[i] for i in range(len(sizes))]
    return None


def _inversion(info, attribute):
    values = []
    for obs in info['observations']:
        t, l, b, r = obs['rect']
        h, w = b-t+1, r-l+1
        values.append({'height': h, 'width': w, 'perimeter': h+w,
                       'reading': (t, l)}[attribute])
    return any(values[i] > values[j] for i in range(len(values))
               for j in range(i+1, len(values)))


def _construct(rng, stage, witness=False, fallback_variant=None):
    # Stage zero teaches extrapolation, stage one interior-rank completion;
    # tests shift to a longer initial garland or a fifth frame.
    test = stage == 2
    nframes = (int(rng.integers(3, 5)) if stage == 3 else
               5 if test and rng.random() < .4 else 4)
    base = 1 if not test or nframes == 5 else 2
    query = (int(rng.integers(1, nframes)) if stage == 3 else
             nframes-1 if stage != 1 else nframes-2)
    if fallback_variant is not None:
        nframes, base, query = 4, (2 if test else 1), (2 if stage == 1 else 3)
        kh = kw = 2
        stride = (2, 2)
    else:
        # Large square kernels with five stages cannot coexist with all other
        # frames on a 30-cell canvas. Rectangular kernels still allow long runs.
        choices = [(2, 2)] if witness or stage == 1 else [(2, 2), (2, 3), (3, 2)]
        if base == 1 and nframes == 4 and not witness and stage != 1:
            choices += [(3, 3)]
        kh, kw = choices[int(rng.integers(len(choices)))]
        stride = kh, kw
        # A possible extra diagonal gap changes spacing without changing the
        # primitive stencil. Only one axis is enlarged to preserve packing room.
        if not witness and rng.random() < .25:
            axis = int(rng.integers(2))
            stride = (kh+int(axis == 0), kw+int(axis == 1))
    kernel = _kernel(rng, kh, kw, asymmetric=witness)
    counts = [base+i for i in range(nframes)]
    padding = stride if witness else (int(rng.integers(stride[0]+1)), int(rng.integers(stride[1]+1)))
    ends = (kh+(counts[query]-1)*stride[0]+padding[0],
            kw+(counts[query]-1)*stride[1]+padding[1])
    sizes = []
    for i, count in enumerate(counts):
        h, w = kh+(count-1)*stride[0]+3, kw+(count-1)*stride[1]+3
        if i == query:
            h, w = max(h, ends[0]+2), max(w, ends[1]+2)
        if fallback_variant is None:
            h += int(rng.integers(3))
            w += int(rng.integers(3))
        sizes.append((h, w))
    # The first two examples dissociate area from one-dimensional size and
    # perimeter. These are aspect-ratio constraints, not position templates.
    if witness:
        sizes = ([(5, 15), (8, 10), (10, 12), (12, 12)] if fallback_variant is not None else
                 [(5, int(rng.integers(15, 19))),
                  (int(rng.integers(8, 11)), int(rng.integers(10, 13))),
                  (int(rng.integers(10, 13)), int(rng.integers(11, 14))), sizes[3]])
    elif stage == 1:
        sizes = ([(15, 5), (10, 8), (10, 11), (12, 12)] if fallback_variant is not None else
                 [(int(rng.integers(15, 19)), 5),
                  (int(rng.integers(10, 13)), int(rng.integers(8, 11))),
                  (sizes[2][0], int(rng.integers(11, 14))),
                  (int(rng.integers(12, 15)), int(rng.integers(12, 15)))])
    areas = [h*w for h, w in sizes]
    if (areas != sorted(set(areas)) or any(max(s) > 28 for s in sizes)
            or sizes[query][0] < ends[0]+2 or sizes[query][1] < ends[1]+2):
        return None
    if fallback_variant is None:
        rects = _pack(rng, sizes)
        if rects is None:
            return None
    else:
        # Compact deterministic rescue placements; colors and corners still vary.
        slots = (((10, 0), (0, 0), (0, 17), (17, 0)) if witness else
                 ((0, 0), (19, 0), (19, 10), (0, 7)) if stage == 1 else
                 ((0, 0), (0, 15), (15, 0), (15, 15)))
        rects = [(t, l, t+h-1, l+w-1) for (t, l), (h, w) in zip(slots, sizes)]
    # Trim empty margins independently, then restore small optional margins.
    t0, l0 = min(z[0] for z in rects), min(z[1] for z in rects)
    h0, w0 = max(z[2] for z in rects)-t0+1, max(z[3] for z in rects)-l0+1
    extra_r = int(rng.integers(min(3, 30-h0)+1))
    extra_c = int(rng.integers(min(3, 30-w0)+1))
    dr = int(rng.integers(extra_r+1))-t0
    dc = int(rng.integers(extra_c+1))-l0
    rects = [(t+dr, l+dc, b+dr, r+dc) for t, l, b, r in rects]
    palette = list(map(int, rng.permutation(10)))
    bg, frame, marker = palette[:3]
    accents = palette[3:3+int(rng.integers(1, 5))]
    qs = list(map(int, rng.integers(4, size=nframes)))
    # Query adjacent to the first complete reference makes the reflection rule
    # observable for asymmetric kernels in the first demonstration.
    if witness:
        qs[query] = qs[0] ^ 1
    if fallback_variant is not None:
        qs[query] = (qs[query] + fallback_variant) % 4
    colors = [int(rng.choice(accents)) for _ in rects]
    full = np.full((h0+extra_r, w0+extra_c), bg, dtype=np.int8)
    for rect in rects:
        for p in _outline(rect):
            full[p] = frame
    # Labels are explicitly painted from sampled objects, never solve().
    for i, rect in enumerate(rects):
        for j in range(counts[i]):
            for u, v in kernel:
                p = _global(rect, qs[i], (u+j*stride[0], v+j*stride[1]))
                full[p] = colors[i]
    p0 = _global(rects[query], qs[query], (2, 2))
    p1 = _global(rects[query], qs[query], ends)
    bounds = min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1])
    t, l, b, r = bounds
    inp = full.copy()
    inp[t:b+1, l:r+1] = bg
    for p in _marker_cells(bounds):
        inp[p] = marker
    pair = {'input': inp, 'output': full[t:b+1, l:r+1].copy()}
    try:
        _, info = _parse(inp)
        assert np.array_equal(_prediction(inp, info), pair['output'])
        assert not np.array_equal(_prediction(inp, info, 'copy_largest'), pair['output'])
        assert not np.array_equal(_prediction(inp, info, 'fragment_template'), pair['output'])
        if witness:
            assert not np.array_equal(_prediction(inp, info, 'repeat_forever'), pair['output'])
            assert not np.array_equal(_prediction(inp, info, 'rotation'), pair['output'])
            assert _inversion(info, 'width') and _inversion(info, 'perimeter')
            assert _inversion(info, 'reading')
        if stage == 1:
            assert _inversion(info, 'height')
    except AssertionError:
        return None
    return pair


def _count(name, value, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count('num_train', num_train, 2, 10), _count('num_test', num_test, 1, 4)
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ne)):
        for index in range(count):
            stage = 2 if split == 'test' else index if index < 2 else 3
            witness = split == 'train' and index == 0
            for attempt in range(80):
                pair = _construct(rng, stage, witness, None if attempt < 64 else attempt-64)
                if pair is not None and _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('bounded construction exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            a, info = _parse(inp.copy())
            assert np.array_equal(_prediction(a, info), out), 'wrong reconstruction'
            assert info['marker'] not in out
            assert set(map(int, np.unique(out))) <= {info['background'], info['observations'][info['query']]['color']}
            assert not np.array_equal(_prediction(a, info, 'copy_largest'), out), 'A ablation does not change answer'
            assert not np.array_equal(_prediction(a, info, 'fragment_template'), out), 'B ablation does not change answer'
            infos[split].append(info)
    # Required demonstrations teach extrapolation, an interior query, finite
    # termination and reflected asymmetric geometry without consulting tests.
    assert any(i['query'] == len(i['counts'])-1 for i in infos['train'])
    assert any(i['query'] < len(i['counts'])-1 for i in infos['train'])
    assert any(not np.array_equal(_prediction(None, i, 'repeat_forever'), p['output'])
               for i, p in zip(infos['train'], puzzle['train']))
    assert any(not np.array_equal(_prediction(None, i, 'rotation'), p['output'])
               for i, p in zip(infos['train'], puzzle['train']))
    assert all(any(_inversion(i, attribute) for i in infos['train'])
               for attribute in ('height', 'width', 'perimeter', 'reading'))
    return None


def self_test():
    """Stream 42 seeds, accumulating only small summaries; no file writes."""
    hashes, shapes, kernels, strides, roles, positions = set(), set(), set(), set(), set(), set()
    queries, counts, repeats = set(), set(), set()
    n_pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                before = inp.copy()
                predicted = solve(inp)
                assert np.array_equal(predicted, out) and np.array_equal(inp, before)
                _, info = _parse(inp)
                digest.update(bytes(inp.shape) + inp.tobytes() + bytes(out.shape) + out.tobytes())
                shapes.add((inp.shape, out.shape))
                kernels.add(tuple(sorted(info['kernel'])))
                strides.add(info['stride'])
                roles.add((info['background'], info['frame'], info['marker']))
                positions.add(tuple(o['rect'] for o in info['observations']))
                queries.add(info['query'])
                counts.add(tuple(info['counts']))
                repeats.add(len({o['color'] for o in info['observations']}) < len(info['observations']))
                n_pairs += 1
        hashes.add(digest.hexdigest())
    assert len(hashes) == 42 and len(shapes) >= 60 and len(kernels) >= 12
    assert len(strides) >= 3 and len(roles) >= 100 and len(positions) >= 150
    assert {2, 3, 4} <= queries and len(counts) >= 3 and True in repeats
    # Every legal count, strict rejection of coercible noninteger values.
    for nt in range(2, 11):
        for ne in range(1, min(nt-1, 4)+1):
            validate_generated_puzzle(generate_puzzle(700+nt*10+ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    a, b = generate_puzzle(802, 2, 1), generate_puzzle(802, 2, 1)
    assert all(_key(p[k]) == _key(q[k]) for s in a for p, q in zip(a[s], b[s]) for k in p)
    a['test'][0]['output'][0, 0] = (int(a['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(a)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt label accepted')
    return {'seeds': 42, 'pairs': n_pairs, 'unique_puzzles': len(hashes),
            'input_output_shapes': len(shapes), 'kernels': len(kernels),
            'strides': [list(map(int, s)) for s in sorted(strides)], 'role_assignments': len(roles),
            'layouts': len(positions), 'count_progressions': sorted(counts)}
