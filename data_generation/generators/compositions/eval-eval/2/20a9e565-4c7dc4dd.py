"""Calibrated wavefronts: local analogy governs a cyclic spatial continuation.

Only NumPy and the standard library. No file access, stored puzzles, or generation
state is used by solve. Ground-truth distances use graph search; the input reader
uses coordinate metrics. JSON written by callers must use compact separators.
"""
import hashlib
import heapq
import numpy as np


# (metric kind, row cost, column cost). Costs are relative to displayed axes.
_METRICS = (("taxicab", 1, 1), ("square", 1, 1),
            ("taxicab", 1, 2), ("taxicab", 2, 1))
_HOST_SEEDS = (0, 295649563, 630577739, 376503444, 11405610, 688690918,
               64933362, 861416345, 222816821, 19756653, 49373240,
               862368728, 304914465, 812460897, 284641246, 238366236,
               769427626, 35863866, 500387694, 200240188, 771870027,
               691987244, 666766318, 286533339, 215646480, 539688676,
               755291456, 411814017, 931245775, 237879078, 879697911,
               197773645, 122155610, 301724265, 768730697, 607576862,
               852646970, 100313873, 205006326, 172701539, 53584725, 266147555)


def _check_grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi):
        raise ValueError(name + " is outside its legal integer range")
    return int(value)


def _key(a):
    a = np.asarray(a, dtype=np.uint8)
    return a.shape, a.tobytes()


def _fingerprint(a):
    return hashlib.sha256(repr(a.shape).encode() + a.astype(np.uint8).tobytes()).hexdigest()


def _corners(bounds):
    r0, r1, c0, c1 = bounds
    return {(r0, c0), (r0, c0 + 1), (r0 + 1, c0),
            (r1, c1), (r1, c1 - 1), (r1 - 1, c1)}


def _find_marker(a):
    found = []
    for color in np.unique(a):
        positions = np.argwhere(a == color)
        if len(positions) != 6:
            continue
        r0, c0 = map(int, positions.min(axis=0))
        r1, c1 = map(int, positions.max(axis=0))
        if min(r1 - r0, c1 - c0) < 2:
            continue
        bounds = r0, r1, c0, c1
        if set(map(tuple, positions)) == _corners(bounds):
            found.append((int(color), bounds))
    if len(found) != 1:
        raise ValueError("expected two unique disconnected canonical corner Ls")
    return found[0]


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask)))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, cells = [start], {start}
        while stack:
            r, c = stack.pop()
            for nxt in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if nxt in todo:
                    todo.remove(nxt)
                    cells.add(nxt)
                    stack.append(nxt)
        result.append(cells)
    return result


def _extract(a):
    marker, bounds = _find_marker(a)
    bg = int(a[0, 0])
    if any(int(a[p]) != bg for p in ((0, -1), (-1, 0), (-1, -1))):
        raise ValueError("canvas corners must identify the common background")
    candidates = []
    for raw in np.unique(a):
        color = int(raw)
        if color in (bg, marker):
            continue
        groups = _components(a == color)
        if len(groups) != 3:
            continue
        boxes = []
        for cells in groups:
            r0 = min(r for r, c in cells)
            r1 = max(r for r, c in cells)
            c0 = min(c for r, c in cells)
            c1 = max(c for r, c in cells)
            expected = {(r, c) for r in range(r0, r1+1) for c in range(c0, c1+1)
                        if r in (r0, r1) or c in (c0, c1)}
            if min(r1-r0, c1-c0) < 4 or cells != expected:
                break
            boxes.append((r0, r1, c0, c1))
        if len(boxes) != 3:
            continue
        containing = [b for b in boxes if b[0] < bounds[0] <= bounds[1] < b[1]
                      and b[2] < bounds[2] <= bounds[3] < b[3]]
        if len(containing) != 1:
            continue
        target_box = containing[0]
        refs = [b for b in boxes if b != target_box]
        if (refs[0][1]-refs[0][0], refs[0][3]-refs[0][2]) != (
                refs[1][1]-refs[1][0], refs[1][3]-refs[1][2]):
            continue
        interiors = [a[r0+1:r1, c0+1:c1].copy() for r0, r1, c0, c1 in refs]
        sparse = [i for i, x in enumerate(interiors)
                  if len(set(map(int, np.unique(x))) - {bg}) == 1]
        if len(sparse) != 1:
            continue
        seed_ref = interiors[sparse[0]]
        full_ref = interiors[1-sparse[0]]
        source = next(iter(set(map(int, np.unique(seed_ref))) - {bg}))
        ref_sources = np.argwhere(seed_ref == source)
        if not 2 <= len(ref_sources) <= 3:
            continue
        r0, r1, c0, c1 = target_box
        target = a[r0+1:r1, c0+1:c1].copy()
        crop = (bounds[0]-r0-1, bounds[1]-r0-1,
                bounds[2]-c0-1, bounds[3]-c0-1)
        target_sources = np.argwhere(target == source)
        if not 2 <= len(target_sources) <= 4:
            continue
        if not np.array_equal(full_ref == source, seed_ref == source):
            continue
        colors = set(map(int, np.unique(full_ref))) - {bg, source}
        target_colors = set(map(int, np.unique(target))) - {bg, source, marker}
        if not 2 <= len(colors) <= 3 or len(colors) != len(target_colors):
            continue
        if {color, marker} & (colors | target_colors | {source}):
            continue
        # No unaccounted content outside the three frame rectangles.
        outside = np.ones(a.shape, dtype=bool)
        for y0, y1, x0, x1 in boxes:
            outside[y0:y1+1, x0:x1+1] = False
        if not np.all(a[outside] == bg):
            continue
        candidates.append(dict(bg=bg, marker=marker, frame=color, source=source,
                               reference=interiors, seed_ref=seed_ref, full_ref=full_ref,
                               ref_sources=ref_sources, target=target,
                               sources=target_sources, crop=crop, bounds=bounds,
                               boxes=boxes, target_box=target_box))
    if len(candidates) != 1:
        raise ValueError("expected one association of two calibration panels and a marked chamber")
    return candidates[0]


def _coordinate_distances(shape, sources, metric):
    rr, cc = np.indices(shape)
    kind, wr, wc = metric
    fields = []
    for r, c in sources:
        dr, dc = abs(rr-int(r)), abs(cc-int(c))
        fields.append(np.maximum(dr, dc) if kind == "square" else wr*dr + wc*dc)
    return np.stack(fields)


def _nearest(fields):
    d = fields.min(axis=0)
    return d, (fields == d).sum(axis=0) >= 2


def _cycle_from_cells(a, d, mask, p):
    palette = []
    for residue in range(p):
        values = np.unique(a[mask & ((d-1) % p == residue)])
        if len(values) != 1:
            return None
        palette.append(int(values[0]))
    return tuple(palette) if len(set(palette)) == p else None


def _render(d, tied, bg, source, palette, cancel):
    out = np.asarray(palette, dtype=np.int8)[(d-1) % len(palette)].copy()
    out[d == 0] = source
    if cancel:
        out[tied & (d > 0)] = bg
    return out


def _reference_fits(scene):
    bg, source = scene['bg'], scene['source']
    a = scene['full_ref']
    p = len(set(map(int, np.unique(a))) - {bg, source})
    fits = []
    for metric in _METRICS:
        d, tied = _nearest(_coordinate_distances(a.shape, scene['ref_sources'], metric))
        if not np.any(tied & (d > 0)):
            continue
        for cancel in (False, True):
            mask = (d > 0) & (~tied if cancel else True)
            palette = _cycle_from_cells(a, d, mask, p)
            if palette is None or set(palette) & {bg, source}:
                continue
            if np.array_equal(_render(d, tied, bg, source, palette, cancel), a):
                fits.append((metric, cancel, palette))
    return fits


def _analyze(a):
    a = _check_grid(a)
    scene = _extract(a)
    bg, source, marker = scene['bg'], scene['source'], scene['marker']
    target = scene['target']
    p = len(set(map(int, np.unique(target))) - {bg, source, marker})
    fits = _reference_fits(scene)
    if len(fits) != 1:
        raise ValueError("calibration must uniquely determine the metric and collision behavior")
    metric, cancel, ref_palette = fits[0]
    target_fields = _coordinate_distances(target.shape, scene['sources'], metric)
    d, tied = _nearest(target_fields)
    visible = (target != bg) & (target != source) & (target != marker)
    if not visible.any():
        raise ValueError("target has no visible cycle")
    radius = int(d[visible].max())
    if not p <= radius <= p+1:
        raise ValueError("target must expose a full cycle and optionally one repeated band")
    # Initial fronts are disjoint: source cells cannot be confused with seams.
    if np.any(tied & (d <= radius)):
        raise ValueError("observed target fronts already collide")
    palette = _cycle_from_cells(target, d, visible, p)
    if palette is None or set(palette) & {bg, source, marker, scene['frame']}:
        raise ValueError("target cycle is incomplete or inconsistent")
    expected_visible = _render(d, tied, bg, source, palette, cancel)
    expected_visible[d > radius] = bg
    r0, r1, c0, c1 = scene['crop']
    if not np.all(d[r0:r1+1, c0:c1+1] > radius):
        raise ValueError("the requested crop must lie beyond the observed fronts")
    clean = target.copy()
    clean[clean == marker] = bg
    if not np.array_equal(clean, expected_visible):
        raise ValueError("target contains unsupported evidence or incomplete early bands")
    full = _render(d, tied, bg, source, palette, cancel)
    answer = full[r0:r1+1, c0:c1+1].copy()
    crop_ties = tied[r0:r1+1, c0:c1+1]
    unique = ~crop_ties
    owners = set(map(int, target_fields[:, r0:r1+1, c0:c1+1].argmin(axis=0)[unique]))
    if int(crop_ties.sum()) < 3 or not unique.any():
        raise ValueError("crop must show both collision and noncollision cells")
    if len(np.unique(answer[unique])) < 2:
        raise ValueError("crop must require continuation of multiple color phases")
    # These same cells are colored under either policy, including an annotation.
    if not any(unique[r-r0, c-c0] for r, c in _corners(scene['crop'])):
        raise ValueError("at least one marker must hide continuing wave content")
    info = dict(metric=metric, cancel=cancel, palette=palette, ref_palette=ref_palette,
                radius=radius, sources=len(scene['sources']),
                ref_sources=len(scene['ref_sources']),
                crop_distance=int(d[r0:r1+1, c0:c1+1].min()),
                tie_cells=int(crop_ties.sum()), output_shape=answer.shape,
                background=bg, frame=scene['frame'], marker=marker,
                source_color=source, target_shape=target.shape,
                reference_shape=scene['full_ref'].shape,
                crop_owners=len(owners), partial=True, bounds=scene['bounds'])
    return answer, info, scene


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer from the three chambers and their pixel evidence only."""
    return _analyze(input_grid)[0]


def _graph_distances(shape, sources, metric):
    """Construction path: shortest travel times on an explicit finite lattice."""
    kind, wr, wc = metric
    moves = [(-1, 0, wr), (1, 0, wr), (0, -1, wc), (0, 1, wc)]
    if kind == 'square':
        moves += [(-1, -1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, 1)]
    fields = []
    h, w = shape
    for y, x in sources:
        a = np.full(shape, 10000, dtype=np.int16)
        a[y, x] = 0
        pending = [(0, int(y), int(x))]
        while pending:
            cost, r, c = heapq.heappop(pending)
            if cost != int(a[r, c]):
                continue
            for dr, dc, weight in moves:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w and cost+weight < int(a[nr, nc]):
                    a[nr, nc] = cost+weight
                    heapq.heappush(pending, (cost+weight, nr, nc))
        fields.append(a)
    return np.stack(fields)


def _paint_from_graph(fields, bg, source, palette, cancel):
    # Explicit cell painting; does not call the input-side renderer.
    out = np.full(fields.shape[1:], bg, dtype=np.int8)
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            times = [int(t) for t in fields[:, r, c]]
            first = min(times)
            if first == 0:
                out[r, c] = source
            elif not cancel or times.count(first) == 1:
                out[r, c] = palette[(first-1) % len(palette)]
    return out


def _outline(a, box, color):
    r0, r1, c0, c1 = box
    a[r0, c0:c1+1] = a[r1, c0:c1+1] = color
    a[r0:r1+1, c0] = a[r0:r1+1, c1] = color


def _put(a, interior, r, c, frame):
    h, w = interior.shape
    _outline(a, (r, r+h+1, c, c+w+1), frame)
    a[r+1:r+h+1, c+1:c+w+1] = interior


def _transform(a, transpose, flip_r, flip_c):
    if transpose:
        a = a.T
    if flip_r:
        a = a[::-1]
    if flip_c:
        a = a[:, ::-1]
    return a.copy()


def _make_design(rng, count, transpose, metric=None):
    """Sample one local calibration plus two alternative collision labels."""
    metric = _METRICS[int(rng.integers(len(_METRICS)))] if metric is None else metric
    colors = list(map(int, rng.permutation(10)))
    bg, source, frame, marker = colors[:4]
    p = int(rng.integers(2, 4))
    ref_palette = tuple(colors[4:4+p])
    remap = bool(rng.integers(2))
    palette = tuple(colors[4+p:4+2*p]) if remap else ref_palette
    radius = p + int(rng.integers(2))
    # Close sources in the calibration expose collisions; their positions do
    # not match those of the larger target chamber.
    for _ in range(100):
        rh, rw = int(rng.integers(5, 8)), int(rng.integers(5, 8))
        nref = int(rng.integers(2, 4))
        selected = rng.choice(rh*rw, nref, replace=False)
        ref_sources = np.array([divmod(int(i), rw) for i in selected], dtype=int)
        ref_seed = np.full((rh, rw), bg, dtype=np.int8)
        for point in ref_sources:
            ref_seed[tuple(point)] = source
        fields = _graph_distances((rh, rw), ref_sources, metric)
        refs = [_paint_from_graph(fields, bg, source, ref_palette, cancel) for cancel in (False, True)]
        trial = dict(bg=bg, source=source, full_ref=refs[0], ref_sources=ref_sources)
        first = _reference_fits(trial)
        trial['full_ref'] = refs[1]
        second = _reference_fits(trial)
        if (len(first) == len(second) == 1 and first[0][:2] == (metric, False)
                and second[0][:2] == (metric, True)):
            break
    else:
        raise ValueError("could not sample a uniquely informative calibration")
    h = int(rng.integers(2*radius+6, 16))
    w = int(rng.integers(max(16, 2*radius+7), 27))
    # Jittered corners allow skewed bisectors and 2/3/4-way meetings. The lower
    # size bound keeps all initial fronts disjoint, including square waves.
    corner_ids = rng.choice(4, count, replace=False)
    sources = []
    for k in corner_ids:
        y, x = int(rng.integers(3)), int(rng.integers(3))
        sources.append((y if k < 2 else h-1-y, x if k % 2 == 0 else w-1-x))
    fields = _graph_distances((h, w), sources, metric)
    d, tied = _nearest(fields)
    if np.any(tied & (d <= radius)):
        raise ValueError("initial fronts intersect")
    full = [_paint_from_graph(fields, bg, source, palette, cancel) for cancel in (False, True)]
    for _ in range(160):
        candidates = np.argwhere(tied & (d > radius))
        if not len(candidates):
            raise ValueError("target has no future collision")
        tr, tc = map(int, candidates[int(rng.integers(len(candidates)))])
        oh, ow = int(rng.integers(3, min(10, h)+1)), int(rng.integers(3, min(12, w)+1))
        r0 = int(np.clip(tr-int(rng.integers(oh)), 0, h-oh))
        c0 = int(np.clip(tc-int(rng.integers(ow)), 0, w-ow))
        r1, c1 = r0+oh-1, c0+ow-1
        region = np.s_[r0:r1+1, c0:c1+1]
        ties = tied[region]
        owners = set(map(int, fields[:, r0:r1+1, c0:c1+1].argmin(axis=0)[~ties]))
        if (np.any(d[region] <= radius) or int(ties.sum()) < 3 or ties.all()
                or (count >= 3 and len(owners) < 3)
                or len(np.unique(full[0][region][~ties])) < 2
                or not any(not tied[r, c] for r, c in _corners((r0, r1, c0, c1)))):
            continue
        crop = r0, r1, c0, c1
        break
    else:
        raise ValueError("could not find an informative continuation crop")
    target = full[0].copy()
    target[d > radius] = bg
    gap = int(rng.integers(1, 3))
    panel_gap = int(rng.integers(1, 4))
    min_width = max(w+2, 2*(rw+2)+panel_gap)
    height = rh+2 + gap + h+2 + 2
    if height > 30 or min_width+2 > 30:
        raise ValueError("layout exceeds ARC bounds")
    width = int(rng.integers(min_width+2, 31))
    canvas = np.full((height, width), bg, dtype=np.int8)
    start = int(rng.integers(1, width-(2*(rw+2)+panel_gap)))
    ref_pos = [(1, start), (1, start+rw+2+panel_gap)]
    if rng.integers(2):
        ref_pos.reverse()
    ty = 1+rh+2+gap
    tx = int(rng.integers(1, width-w-2))
    _put(canvas, ref_seed, *ref_pos[0], frame)
    _put(canvas, refs[0], *ref_pos[1], frame)
    _put(canvas, target, ty, tx, frame)
    bounds = (ty+1+crop[0], ty+1+crop[1], tx+1+crop[2], tx+1+crop[3])
    mask = np.zeros(canvas.shape, bool)
    mask[bounds[0]:bounds[1]+1, bounds[2]:bounds[3]+1] = True
    flip_r, flip_c = bool(rng.integers(2)), bool(rng.integers(2))
    transformed_mask = _transform(mask, transpose, flip_r, flip_c)
    positions = np.argwhere(transformed_mask)
    lo, hi = positions.min(axis=0), positions.max(axis=0)
    final_bounds = int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1])
    pairs = []
    for cancel in (False, True):
        a = canvas.copy()
        _put(a, refs[int(cancel)], *ref_pos[1], frame)
        a = _transform(a, transpose, flip_r, flip_c)
        for point in _corners(final_bounds):
            a[point] = marker
        out = _transform(full[int(cancel)][crop[0]:crop[1]+1, crop[2]:crop[3]+1],
                         transpose, flip_r, flip_c)
        predicted, _, _ = _analyze(a)
        if not np.array_equal(predicted, out):
            raise ValueError("construction and input-reading paths disagree")
        pairs.append({'input': a, 'output': out})
    return pairs


def _sample_design(rng, count, transpose, metric=None):
    # Retries stay in RAM; unsuccessful proposals do not create files or logs.
    for _ in range(80):
        try:
            return _make_design(rng, count, transpose, metric)
        except ValueError:
            continue
    # A finite pool of same-family fallback designs, visited in seeded order.
    # No weakened validation or changed rule. Its order can change between
    # requests, so fallback does not force identical extra examples.
    for variant in rng.permutation(32):
        fallback_rng = np.random.default_rng(9117+variant)
        try:
            return _make_design(fallback_rng, count, transpose, _METRICS[0])
        except ValueError:
            continue
    raise RuntimeError("bounded design construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 'num_train', 2, 10)
    ne = _count(num_test, 'num_test', 1, 4)
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    train_axis = bool(rng.integers(2))
    axis_shift = bool(rng.integers(2))
    twins = _sample_design(rng, 2, train_axis)
    if rng.integers(2):
        twins.reverse()
    puzzle = {'train': twins, 'test': []}
    seen = {_key(pair['input']) for pair in twins}
    for split, count in (('train', nt-2), ('test', ne)):
        for _ in range(count):
            for _ in range(32):
                nsource = 2 if split == 'train' else int(rng.integers(3, 5))
                axis = train_axis if split == 'train' else train_axis ^ axis_shift
                pair = _sample_design(rng, nsource, axis)[int(rng.integers(2))]
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('could not construct distinct inputs')
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, infos, scenes = set(), {'train': [], 'test': []}, []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _check_grid(pair['input']), _check_grid(pair['output'])
            assert _key(a) not in seen, 'duplicate input'
            seen.add(_key(a))
            expected, info, scene = _analyze(a)
            assert np.array_equal(expected, b), 'incorrect continuation crop'
            assert not np.any(b == info['marker'])
            assert set(map(int, np.unique(b))) <= set(map(int, np.unique(a)))
            assert info['crop_distance'] > info['radius']
            assert info['tie_cells'] > 0 and len(np.unique(b)) >= 2
            # Parent A ablations: freezing the observed waves leaves a blank
            # crop; holding one color cannot explain the noncollision phases.
            assert np.any(b != info['background'])
            assert len(set(info['palette']) & set(map(int, np.unique(b)))) >= 2
            infos[split].append(info)
            if split == 'train':
                scenes.append(scene)
    assert {i['cancel'] for i in infos['train'][:2]} == {False, True}
    # The mandatory controlled training contrast changes only calibration.
    assert np.array_equal(scenes[0]['target'], scenes[1]['target'])
    assert np.array_equal(scenes[0]['seed_ref'], scenes[1]['seed_ref'])
    assert scenes[0]['bounds'] == scenes[1]['bounds']
    assert not np.array_equal(puzzle['train'][0]['output'], puzzle['train'][1]['output'])
    assert all(i['sources'] == 2 for i in infos['train'])
    assert all(i['sources'] in (3, 4) for i in infos['test'])
    assert all(i['crop_owners'] >= 3 for i in infos['test'])


def _puzzle_signature(p):
    return tuple((_key(pair['input']), _key(pair['output']))
                 for split in ('train', 'test') for pair in p[split])


def _diagnostic_test():
    # B correspondence is content based: exchanging the complete members
    # cannot reverse an irreversible cancellation relation.
    pair = generate_puzzle(0, 2, 1)['train'][0]
    a = pair['input'].copy()
    _, _, scene = _analyze(a)
    ref_boxes = [b for b in scene['boxes'] if b != scene['target_box']]
    contents = [a[y0+1:y1, x0+1:x1].copy() for y0, y1, x0, x1 in ref_boxes]
    for box, content in zip(ref_boxes, contents[::-1]):
        y0, y1, x0, x1 = box
        a[y0+1:y1, x0+1:x1] = content
    assert np.array_equal(solve(a), pair['output'])
    # Every literal role, including background and annotation, can be permuted.
    mapping = np.array([6, 3, 8, 1, 9, 0, 4, 7, 2, 5], dtype=np.int8)
    assert np.array_equal(solve(mapping[a]), mapping[pair['output']])
    # Equality of two farther distances is not a first-arrival collision.
    # The other cells respectively witness a two-way and a three-way minimum.
    fields = np.array([[[1, 3, 3]], [[3, 3, 3]], [[3, 7, 3]]])
    d, tied = _nearest(fields)
    assert tied.tolist() == [[False, True, True]]
    assert _render(d, tied, 0, 9, (1, 2), True).tolist() == [[1, 0, 0]]
    # Altering the input's last visible band cannot silently choose a new rule.
    broken = pair['input'].copy()
    target = scene['target']
    colors = scene['source'], scene['bg'], scene['marker']
    pos = np.argwhere(~np.isin(target, colors))[0]
    y0, _, x0, _ = scene['target_box']
    broken[y0+1+pos[0], x0+1+pos[1]] = scene['bg']
    try:
        solve(broken)
    except ValueError:
        pass
    else:
        raise AssertionError('incomplete early wave evidence was accepted')


def self_test():
    """Stream 42 supplied seeds; retain only small counters and hashes."""
    metrics, palettes, backgrounds, shapes, crop_shapes = set(), set(), set(), set(), set()
    sources, cycles, radii, source_layouts, policies, signatures = set(), set(), set(), set(), set(), set()
    twin_changes = remapped = square = skewed = edge_sources = total = 0
    for seed in _HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(hashlib.sha256(repr(_puzzle_signature(puzzle)).encode()).hexdigest())
        assert _puzzle_signature(puzzle) == _puzzle_signature(generate_puzzle(seed))
        twin_changes += not np.array_equal(puzzle['train'][0]['output'], puzzle['train'][1]['output'])
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                _, info, scene = _analyze(pair['input'])
                total += 1
                metrics.add(info['metric'])
                palettes.add(info['palette'])
                backgrounds.add(info['background'])
                shapes.add(pair['input'].shape)
                crop_shapes.add(pair['output'].shape)
                sources.add(info['sources'])
                cycles.add(len(info['palette']))
                radii.add(info['radius'])
                policies.add(info['cancel'])
                source_layouts.add((scene['target'].shape, tuple(map(tuple, scene['sources']))))
                remapped += not bool(set(info['palette']) & set(info['ref_palette']))
                square += info['metric'][0] == 'square'
                skewed += info['metric'][1] != info['metric'][2]
                h, w = scene['target'].shape
                edge_sources += any(r in (0, h-1) or c in (0, w-1) for r, c in scene['sources'])
                # Independent geometry identity: graph travel times vs analytic distances.
                assert np.array_equal(_graph_distances(scene['target'].shape, scene['sources'], info['metric']),
                                      _coordinate_distances(scene['target'].shape, scene['sources'], info['metric']))
    assert len(signatures) == 42 and twin_changes == 42 and total == 210
    assert metrics == set(_METRICS) and policies == {False, True}
    assert sources == {2, 3, 4} and cycles == {2, 3} and radii == {2, 3, 4}
    assert len(backgrounds) >= 8 and len(palettes) >= 25
    assert len(shapes) >= 20 and len(crop_shapes) >= 20 and len(source_layouts) >= 80
    assert min(remapped, square, skewed, edge_sources) > 10
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            puzzle = generate_puzzle(42+nt*10+ne, nt, ne)
            validate_generated_puzzle(puzzle)
            assert len(puzzle['train']) == nt and len(puzzle['test']) == ne
    for nt, ne in ((1,1), (11,1), (4,0), (6,5), (2,2), (3,4),
                   (True,1), (4,1.5), (4,None), (None,1)):
        try:
            generate_puzzle(1, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    damaged = generate_puzzle(321, 2, 1)
    damaged['test'][0]['output'][0,0] = (int(damaged['test'][0]['output'][0,0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('incorrect output accepted')
    _diagnostic_test()
    return dict(seeds=42, pairs=total, controlled_contrasts=twin_changes,
                metrics=len(metrics), source_counts=sorted(sources), cycle_lengths=sorted(cycles),
                visible_radii=sorted(radii), input_shapes=len(shapes), crop_shapes=len(crop_shapes),
                source_layouts=len(source_layouts), backgrounds=len(backgrounds),
                target_cycles=len(palettes), fully_remapped_pairs=remapped,
                square_metric_pairs=square, stretched_metric_pairs=skewed,
                pairs_with_edge_emitters=edge_sources)
