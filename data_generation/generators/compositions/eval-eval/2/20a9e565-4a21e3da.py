"""An extrapolated colored fault cuts a shape before its fragments move.

Only standard-library Python and NumPy are used. Generation paints labels from
explicit coordinates; solve recovers the post progression from the input.
"""
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid colors")
    return a.astype(np.int8, copy=True)


def _corners(bounds):
    r0, r1, c0, c1 = bounds
    return {(r0, c0), (r0, c0 + 1), (r0 + 1, c0),
            (r1, c1), (r1, c1 - 1), (r1 - 1, c1)}


def _window(a):
    found = []
    for color in np.unique(a):
        cells = np.argwhere(a == color)
        if len(cells) != 6:
            continue
        lo, hi = cells.min(axis=0), cells.max(axis=0)
        b = (int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1]))
        if b[1] - b[0] >= 4 and b[3] - b[2] >= 4:
            if set(map(tuple, cells)) == _corners(b):
                found.append((int(color), b))
    _require(len(found) == 1, "need two unique opposite L annotations")
    return found[0]


def _components(cells):
    todo, result = set(cells), []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, component = [start], {start}
        while stack:
            r, c = stack.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in todo:
                    todo.remove(q)
                    component.add(q)
                    stack.append(q)
        result.append(component)
    return result


def _parse(input_grid):
    """Read annotations, orient toward the evidence, then fit finite differences."""
    original = _grid(input_grid)
    marker, original_bounds = _window(original)
    values, counts = np.unique(original, return_counts=True)
    _require(np.count_nonzero(counts == counts.max()) == 1, "ambiguous background")
    bg = int(values[counts.argmax()])
    _require(bg != marker, "annotation is not background")
    options = []
    for k in range(4):
        a = np.rot90(original, k).copy()
        pos = np.argwhere(a == marker)
        lo, hi = pos.min(axis=0), pos.max(axis=0)
        r0, r1, c0, c1 = int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1])
        a[a == marker] = bg
        outside = a != bg
        outside[r0:r1 + 1, c0:c1 + 1] = False
        evidence = np.argwhere(outside)
        if len(evidence) and np.all(evidence[:, 0] < r0) and np.all(
                (evidence[:, 1] >= c0) & (evidence[:, 1] <= c1)):
            options.append((k, a, (r0, r1, c0, c1), evidence))
    _require(len(options) == 1, "evidence must lie on exactly one entry side")
    k, a, bounds, evidence = options[0]
    r0, r1, c0, c1 = bounds
    first, last = int(evidence[:, 0].min()), int(evidence[:, 0].max())
    _require(len(evidence) == last - first + 1, "one guide cell per evidence row required")
    samples = []
    for row in range(first, last + 1):
        cols = np.flatnonzero(a[row] != bg)
        _require(len(cols) == 1, "malformed post")
        col = int(cols[0])
        samples.append((col - c0, int(a[row, col])))
    runs = []
    for sample in samples:
        if runs and sample == runs[-1][0]:
            runs[-1][1] += 1
        else:
            runs.append([sample, 1])
    unit = runs[0][1]
    _require(1 <= unit <= 3 and all(n == unit for _, n in runs), "unequal post heights")
    _require(len(runs) >= 4, "at least four visible posts required")
    xs = [item[0][0] for item in runs]
    cs = [item[0][1] for item in runs]
    d, acceleration = xs[1] - xs[0], xs[2] - 2 * xs[1] + xs[0]
    _require(acceleration in (-1, 0, 1), "unsupported second difference")
    _require(all(x == xs[0] + i * d + acceleration * i * (i - 1) // 2
                 for i, x in enumerate(xs)), "inconsistent post geometry")
    cycles = [tuple(cs[:p]) for p in (2, 3)
              if len(cs) >= p + 1 and len(set(cs[:p])) == p
              and all(c == cs[i % p] for i, c in enumerate(cs))]
    _require(len(cycles) == 1, "color cycle not established")
    palette = cycles[0]
    crop = a[r0:r1 + 1, c0:c1 + 1].copy()
    colors = set(map(int, np.unique(crop))) - {bg}
    _require(len(colors) == 1, "window must contain one payload color")
    payload = colors.pop()
    _require(payload not in palette and bg not in palette and marker not in palette,
             "roles must be distinct")
    _require(len(_components(map(tuple, np.argwhere(crop == payload)))) == 1,
             "payload must be connected before the cut")
    _require(np.all(crop[0] == bg) and np.all(crop[-1] == bg)
             and np.all(crop[:, 0] == bg) and np.all(crop[:, -1] == bg),
             "payload must be strictly inside the window")
    height, width = crop.shape
    indices = (np.arange(r0, r1 + 1) - first) // unit
    path = xs[0] + d * indices + acceleration * indices * (indices - 1) // 2
    _require(bool(np.all((path > 0) & (path < width - 1))), "fault leaves the window")
    ink = np.asarray([palette[int(i) % len(palette)] for i in indices], dtype=np.int8)
    mask = crop == payload
    hits = np.flatnonzero(mask[np.arange(height), path])
    _require(len(hits) >= 2, "need multiple fault contacts")
    rr, cc = np.indices(crop.shape)
    groups = [set(map(tuple, np.argwhere(mask & (cc < path[:, None])))),
              set(map(tuple, np.argwhere(mask & (cc > path[:, None]))))]
    _require(all(groups), "both sides must contain payload")
    return dict(original=original, bounds=original_bounds, rotation=k, crop=crop,
                bg=bg, marker=marker, payload=payload, palette=palette,
                unit=unit, visible=len(runs), first=first, top=r0,
                gap=r0 - last - 1, d=d, acceleration=acceleration,
                xs=xs, indices=indices, path=path, ink=ink, hits=hits,
                groups=groups)


def _raster_solution(info):
    """Input-side solution using masks and collective bounding-box crops."""
    crop = info['crop']
    h, w = crop.shape
    bg, payload, path = info['bg'], info['payload'], info['path']
    mask = crop == payload
    out = np.full_like(crop, bg)
    hitrows = info['hits']
    out[hitrows, path[hitrows]] = payload
    for r in range(int(hitrows[-1]) + 1):
        if not mask[r, path[r]]:
            out[r, path[r]] = info['ink'][r]
    for side, cells in enumerate(info['groups']):
        coords = np.asarray(sorted(cells), dtype=int)
        lo, hi = coords.min(axis=0), coords.max(axis=0)
        piece = np.zeros(tuple(hi - lo + 1), dtype=bool)
        piece[coords[:, 0] - lo[0], coords[:, 1] - lo[1]] = True
        start = 0 if side == 0 else w - piece.shape[1]
        target = out[:piece.shape[0], start:start + piece.shape[1]]
        _require(bool(np.all(target[piece] == bg)), "translated piece collides with fault")
        target[piece] = payload
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer everything from this input; preserve its output orientation."""
    info = _parse(input_grid)
    return np.rot90(_raster_solution(info), -info['rotation']).copy()


def _paint_truth(shape, cells, path, ink, bg, payload, fragments=False,
                 nearest=False, relocate=True):
    """Generation-side coordinate renderer; no input fitting or solve call."""
    h, w = shape
    out = np.full(shape, bg, dtype=np.int8)
    fixed = {(r, c) for r, c in cells if c == int(path[r])}
    _require(bool(fixed), "no fault contact")
    stop = (min if nearest else max)(r for r, _ in fixed)
    for r in range(stop + 1):
        out[r, int(path[r])] = ink[r]
    for r, c in fixed:
        out[r, c] = payload
    if not relocate:
        for r, c in cells:
            out[r, c] = payload
        return out
    for side in (0, 1):
        group = {(r, c) for r, c in cells
                 if (c < int(path[r]) if side == 0 else c > int(path[r]))}
        pieces = _components(group) if fragments else [group]
        for piece in pieces:
            if not piece:
                continue
            dr = -min(r for r, _ in piece)
            dc = -min(c for _, c in piece) if side == 0 else w - 1 - max(c for _, c in piece)
            for r, c in piece:
                if not fragments:
                    _require(out[r + dr, c + dc] == bg, "latent placement collision")
                out[r + dr, c + dc] = payload
    return out


def _payload_shape(rng, h, w, path, span, witness):
    """A winding trunk, unequal crossing branches, then irregular growth."""
    lo, hi = span + 1, w - span - 2
    left_trunk = bool(rng.integers(2))
    low, high = (lo, int(min(path)) - 1) if left_trunk else (int(max(path)) + 1, hi)
    if low > high:
        return None
    col = int(rng.integers(low, high + 1))
    cells, trunk = set(), {}
    for row in range(1, h - 1):
        new = int(np.clip(col + int(rng.integers(-1, 2)), low, high))
        cells.update((row, c) for c in range(min(col, new), max(col, new) + 1))
        trunk[row] = new
        col = new
    if witness:
        rows = [1, h - 3]
    else:
        rows = [1, h - 2]
        if h >= 8 and rng.random() < .65:
            rows.append(int(rng.integers(3, h - 3)))
    for row in sorted(set(rows)):
        p = int(path[row])
        if left_trunk:
            end = int(rng.integers(p + 1, min(hi, p + 5) + 1))
        else:
            end = int(rng.integers(max(lo, p - 5), p))
        cells.update((row, c) for c in range(min(trunk[row], end), max(trunk[row], end) + 1))
    # Connected frontier growth varies density, thickness, gaps and cut fragments.
    area = (h - 2) * (hi - lo + 1)
    extra = int(rng.integers(0, max(2, area // (7 if witness else 3))))
    for _ in range(extra):
        frontier = sorted({q for r, c in cells
                           for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                           if 1 <= q[0] < h - 1 and lo <= q[1] <= hi and q not in cells})
        if not frontier:
            break
        cells.add(frontier[int(rng.integers(len(frontier)))])
    return cells


def _make_pair(rng, kind, test, turn, roles, witness):
    bg, marker, payload, *pool = roles
    p = int(rng.integers(2, 4))
    palette = pool[:p]
    unit = int(rng.integers(2, 4)) if kind != 'straight' else int(rng.integers(1, 4))
    visible = 4 if kind == 'accelerated' else int(rng.integers(4, 6))
    gap = int(rng.integers(unit)) + (unit if test else 0)
    h = int(rng.integers(6 if witness else 5, 11))
    total_rows = visible * unit + gap + h
    if total_rows > 28:
        return None
    n = (total_rows + unit - 1) // unit
    acceleration = int(rng.choice([-1, 1])) if kind == 'accelerated' else 0
    ds = list(range(-5, 6)) if acceleration else ([0] if kind == 'straight' else [-1, 1])
    candidates = []
    for d in ds:
        xs = [d * i + acceleration * i * (i - 1) // 2 for i in range(n)]
        span = max(xs) - min(xs)
        if span <= 7:
            candidates.append((d, xs, span))
    if not candidates:
        return None
    d, xs, span = candidates[int(rng.integers(len(candidates)))]
    minw = max(12, 3 * span + 5)
    if minw > 28:
        return None
    w = int(rng.integers(minw, 29))
    base = int(rng.integers(span + 2, w - 2 * span - 2)) - min(xs)
    xs = [base + x for x in xs]
    indices = (visible * unit + gap + np.arange(h)) // unit
    path = np.asarray([xs[int(i)] for i in indices], dtype=int)
    ink = np.asarray([palette[int(i) % p] for i in indices], dtype=np.int8)
    cells = _payload_shape(rng, h, w, path, span, witness)
    if cells is None:
        return None
    hits = sorted(r for r, c in cells if c == path[r])
    groups = [{(r, c) for r, c in cells if c < path[r]},
              {(r, c) for r, c in cells if c > path[r]}]
    if len(hits) < 2 or not all(groups):
        return None
    if witness:
        gap_hit = any(r not in hits for r in range(min(hits), max(hits) + 1))
        fragmented = max(len(_components(g)) for g in groups) >= 2
        changes = any(np.sign(c - path[r]) != np.sign(c - xs[visible - 1]) for r, c in cells)
        if not (gap_hit and fragmented and changes and max(hits) < max(r for r, c in cells)):
            return None
    answer = _paint_truth((h, w), cells, path, ink, bg, payload)
    spare_h, spare_w = 30 - total_rows, 30 - w
    top = int(rng.integers(1, min(3, spare_h - 1) + 1))
    bottom = int(rng.integers(1, min(3, spare_h - top) + 1))
    left = int(rng.integers(1, min(3, spare_w - 1) + 1))
    right = int(rng.integers(1, min(3, spare_w - left) + 1))
    grid = np.full((top + total_rows + bottom, left + w + right), bg, dtype=np.int8)
    for i in range(visible):
        grid[top + i * unit:top + (i + 1) * unit, left + xs[i]] = palette[i % p]
    r0 = top + visible * unit + gap
    bounds = (r0, r0 + h - 1, left, left + w - 1)
    for r, c in cells:
        grid[r0 + r, left + c] = payload
    # Rotate the complete scene and answer; canonical L annotations are redrawn.
    tag = np.zeros_like(grid, dtype=bool)
    tag[bounds[0]:bounds[1] + 1, bounds[2]:bounds[3] + 1] = True
    grid, tag = np.rot90(grid, turn).copy(), np.rot90(tag, turn)
    pos = np.argwhere(tag)
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    bounds = (int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1]))
    for point in _corners(bounds):
        _require(grid[point] == bg, "marker overwrites payload")
        grid[point] = marker
    return {'input': grid, 'output': np.rot90(answer, turn).copy()}


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(name + ' out of range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """All conventions are visible or family-wide; seed controls sampling only."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    nt = _count(num_train, 'num_train', 2, 10)
    ne = _count(num_test, 'num_test', 1, 4)
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.permutation(10)[:6]))
    variable_palette = bool(rng.integers(2))
    base_turn = int(rng.integers(4))
    held_out_turn = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ne)):
        for j in range(count):
            test = split == 'test'
            kind = ('accelerated' if j == 0 else 'linear') if not test and j < 2 else str(
                rng.choice(['accelerated', 'linear', 'straight'], p=[.45, .35, .20]))
            if test and j == 0 and kind == 'straight':
                kind = 'accelerated'
            turn = ((base_turn + (j % 2)) % 4 if not test else
                    (base_turn + 2 + j % 2) % 4 if held_out_turn else (base_turn + j % 2) % 4)
            pair_roles = tuple(map(int, rng.permutation(10)[:6])) if variable_palette else roles
            for _ in range(600):
                pair = _make_pair(rng, kind, test, turn, pair_roles, not test and j == 0)
                if pair is not None and _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded geometry sampling exhausted')
    return puzzle


def pair_features(input_grid):
    """Input-derived measurements, including actual cut and contact geometry."""
    q = _parse(input_grid)
    cells = set(map(tuple, np.argwhere(q['crop'] == q['payload'])))
    hits, path = q['hits'], q['path']
    return dict(entry=('top', 'right', 'bottom', 'left')[q['rotation']],
                shape=list(q['original'].shape), crop_shape=list(q['crop'].shape),
                background=q['bg'], marker=q['marker'], payload=q['payload'],
                palette=list(q['palette']), post_height=q['unit'], visible=q['visible'],
                first_difference=q['d'], second_difference=q['acceleration'],
                skipped_posts=q['gap'] // q['unit'], crop_phase=int(q['indices'][0] % len(q['palette'])),
                partial_post=q['gap'] % q['unit'] != 0,
                path_span=int(max(path) - min(path)),
                partition_sizes=[len(g) for g in q['groups']],
                partition_components=[len(_components(g)) for g in q['groups']],
                internal_gap=any(r not in hits for r in range(int(hits[0]), int(hits[-1]) + 1)),
                short_endpoint=bool(int(hits[-1]) < max(r for r, c in cells)),
                membership_changes=int(sum(np.sign(c - path[r]) != np.sign(c - q['xs'][-1]) for r, c in cells)))


def _mutant_diff(q, name):
    cells = set(map(tuple, np.argwhere(q['crop'] == q['payload'])))
    try:
        options = dict(fragments=name == 'fragments', nearest=name == 'nearest', relocate=name != 'no_B')
        path = np.full_like(q['path'], q['xs'][-1]) if name == 'no_A' else q['path']
        ink = np.full_like(q['ink'], q['palette'][(q['visible'] - 1) % len(q['palette'])]) if name == 'no_A' else q['ink']
        alternate = _paint_truth(q['crop'].shape, cells, path, ink, q['bg'], q['payload'], **options)
    except ValueError:
        return True
    return not np.array_equal(alternate, _raster_solution(q))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, features, train_queries = set(), {'train': [], 'test': []}, []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            q = _parse(inp)
            expected = np.rot90(_raster_solution(q), -q['rotation'])
            assert np.array_equal(out, expected)
            assert q['marker'] not in out
            assert set(map(int, np.unique(out))) <= set(map(int, np.unique(inp)))
            assert np.count_nonzero(out == q['payload']) == np.count_nonzero(q['crop'] == q['payload'])
            # Nothing is deleted. Only unoccupied path cells up to the last hit gain ink.
            assert np.count_nonzero(np.isin(out, q['palette'])) == int(q['hits'][-1]) + 1 - len(q['hits'])
            assert out.shape == (q['bounds'][1] - q['bounds'][0] + 1,
                                 q['bounds'][3] - q['bounds'][2] + 1)
            features[split].append(pair_features(inp))
            if split == 'train':
                train_queries.append(q)
    tr, te = features['train'], features['test']
    assert any(f['second_difference'] != 0 for f in tr)
    assert any(f['second_difference'] == 0 and f['first_difference'] != 0 for f in tr)
    assert all(f['skipped_posts'] == 0 for f in tr)
    assert all(f['skipped_posts'] == 1 for f in te)
    assert any(f['internal_gap'] and f['short_endpoint'] and max(f['partition_components']) > 1
               and f['membership_changes'] > 0 for f in tr)
    for name in ('no_A', 'no_B', 'nearest', 'fragments'):
        assert any(_mutant_diff(q, name) for q in train_queries), name + ' shortcut unexposed'


def self_test():
    """Bounded 42-seed audit, legal counts, determinism and negative checks."""
    features, structures, puzzle_keys = [], set(), set()
    directions = {'train': set(), 'test': set()}
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzle_keys.add(tuple(_key(pair['input']) for s in p for pair in p[s]))
        for split in ('train', 'test'):
            for pair in p[split]:
                q = _parse(pair['input'])
                f = pair_features(pair['input'])
                features.append(f)
                directions[split].add(f['entry'])
                # Normalize semantic roles, retaining all locations and dimensions.
                normal = np.full_like(q['original'], 0)
                normal[q['original'] == q['payload']] = 1
                normal[q['original'] == q['marker']] = 2
                for i, color in enumerate(q['palette']):
                    normal[q['original'] == color] = 3 + i
                structures.add(_key(normal))
    assert len(puzzle_keys) == 42 and len(structures) >= 180
    assert all(len(v) == 4 for v in directions.values())
    assert {f['second_difference'] for f in features} == {-1, 0, 1}
    assert {f['post_height'] for f in features} == {1, 2, 3}
    assert {len(f['palette']) for f in features} == {2, 3}
    assert len({tuple(f['shape']) for f in features}) >= 35
    assert len({tuple(f['crop_shape']) for f in features}) >= 25
    assert {f['background'] for f in features} == set(range(10))
    assert sum(f['membership_changes'] > 0 for f in features) >= 80
    assert sum(max(f['partition_components']) > 1 for f in features) >= 80
    assert sum(f['internal_gap'] for f in features) >= 80
    assert sum(f['short_endpoint'] for f in features) >= 40
    assert sum(f['partial_post'] for f in features) >= 50
    # All 30 legal explicit combinations, with the smallest demonstration budget.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(700 + nt * 10 + ne, nt, ne)
            validate_generated_puzzle(p)
            assert len(p['train']) == nt and len(p['test']) == ne
    p, again = generate_puzzle(91, 2, 1), generate_puzzle(91, 2, 1)
    assert all(_key(x[k]) == _key(y[k]) for s in p for x, y in zip(p[s], again[s]) for k in x)
    for nt, ne in [(1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.0, 1), (4, False), (4, None)]:
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p['test'][0]['output'][0, 0] = (int(p['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('corrupt label accepted')
    return {'seeds': 42, 'pairs': len(features), 'structures': len(structures),
            'input_shapes': len({tuple(f['shape']) for f in features}),
            'crop_shapes': len({tuple(f['crop_shape']) for f in features}),
            'curved': sum(f['second_difference'] != 0 for f in features),
            'fragmented': sum(max(f['partition_components']) > 1 for f in features),
            'internal_gaps': sum(f['internal_gap'] for f in features),
            'short_endpoints': sum(f['short_endpoint'] for f in features),
            'membership_changes': sum(f['membership_changes'] > 0 for f in features)}
