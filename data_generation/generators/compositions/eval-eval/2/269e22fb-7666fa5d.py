"""Recognition-activated loom. No files, examples, or seed are used by solve.

All experiments are in memory. The bounded self-test returns aggregate metrics;
it never writes puzzles, logs, traces, or caches of generated examples.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _d4(a, k):
    return np.rot90(np.fliplr(a) if k >= 4 else a, k % 4).copy()


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError('expected a 1..30 integer grid with colors 0..9')
    return a.astype(np.int8, copy=True)


def _frame(grid, bg):
    """The outermost solid rectangular card has a uniform one-cell border."""
    candidates = []
    for color in np.unique(grid):
        if color == bg:
            continue
        rr, cc = np.where(grid == color)
        r, c, h, w = int(rr.min()), int(cc.min()), int(rr.max()-rr.min()+1), int(cc.max()-cc.min()+1)
        if h < 4 or w < 4:
            continue
        rect = grid[r:r+h, c:c+w]
        if (np.all(rect != bg) and np.all(rect[0] == color)
                and np.all(rect[-1] == color) and np.all(rect[:, 0] == color)
                and np.all(rect[:, -1] == color)
                and int(np.sum(grid == color)) == 2*h+2*w-4):
            inner = rect[1:-1, 1:-1]
            if len(np.unique(inner)) == 2 and color not in inner:
                candidates.append((h*w, r, c, h, w, int(color)))
    if not candidates:
        raise ValueError('missing framed two-color swatch')
    candidates.sort(reverse=True)
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        raise ValueError('ambiguous card')
    _, r, c, h, w, color = candidates[0]
    return r, c, h, w, color


def _strokes(grid, mask):
    cells = np.argwhere(mask)
    candidates = []
    for slope in (-1, 1):
        lines = {}
        for r, c in cells:
            lines.setdefault(int(c-slope*r), []).append(int(r))
        specs, labels = [], np.zeros(grid.shape, dtype=np.int16)
        valid = True
        for q, rows in sorted(lines.items()):
            rows.sort()
            runs = []
            for r in rows:
                if not runs or r != runs[-1][-1]+1:
                    runs.append([])
                runs[-1].append(r)
            for run in runs:
                colors = {int(grid[r, q+slope*r]) for r in run}
                if len(run) < 2 or len(colors) != 1:
                    valid = False
                    break
                specs.append((q, 2*run[0]+slope*q, 2*run[-1]+slope*q, colors.pop()))
                for r in run:
                    labels[r, q+slope*r] = len(specs)
            if not valid:
                break
        if valid and len(specs) >= 2:
            candidates.append((slope, specs, labels))
    if len(candidates) != 1:
        raise ValueError('source strokes must have a unique common slope')
    return candidates[0]


def _matches(mask, forbidden, patch):
    """Deduplicate symmetry-equivalent matches of the same physical rectangle."""
    colors = np.unique(patch)
    if len(colors) != 2:
        raise ValueError('swatch needs two color roles')
    binary = (patch == colors[1]).astype(np.int8)
    found, variants = set(), set()
    for k in range(8):
        transformed = _d4(binary, k)
        for exchange in (0, 1):
            p = transformed ^ exchange
            signature = _key(p)
            if signature in variants:
                continue
            variants.add(signature)
            h, w = p.shape
            if h > mask.shape[0] or w > mask.shape[1]:
                continue
            windows = np.lib.stride_tricks.sliding_window_view(mask, (h, w))
            blocked = np.lib.stride_tricks.sliding_window_view(forbidden, (h, w))
            good = np.all(windows == p, axis=(-2, -1)) & ~np.any(blocked, axis=(-2, -1))
            for r, c in np.argwhere(good):
                found.add((int(r), int(c), h, w))
    return sorted(found)


def _inspect(input_grid):
    grid = _grid(input_grid)
    values, counts = np.unique(grid, return_counts=True)
    if counts.max()*2 <= grid.size:
        raise ValueError('field must occupy more than half the grid')
    bg = int(values[np.argmax(counts)])
    r, c, h, w, paint = _frame(grid, bg)
    card = np.zeros(grid.shape, dtype=bool)
    card[r:r+h, c:c+w] = True
    query = grid[r+1:r+h-1, c+1:c+w-1]
    source = (grid != bg) & ~card
    if not np.any(source):
        raise ValueError('missing reference')
    if set(int(x) for x in np.unique(grid[source])) & ({paint} | set(int(x) for x in np.unique(query))):
        raise ValueError('card colors and source ink colors must be distinct')
    slope, specs, labels = _strokes(grid, source)
    matches = _matches(source, card, query)
    if len(matches) != 1:
        raise ValueError('swatch must locate exactly one reference rectangle')
    y, x, hh, ww = matches[0]
    selected = sorted(int(v)-1 for v in np.unique(labels[y:y+hh, x:x+ww]) if v)
    if len(selected) < 2:
        raise ValueError('swatch must touch at least two complete strokes')
    return dict(grid=grid, bg=bg, paint=paint, card=card, query=query,
                source=source, slope=slope, specs=specs, labels=labels,
                window=matches[0], selected=selected)


def _fill(shape, slope, specs, selected, alternative=''):
    """Union of pairwise interval brackets equals the outermost-active sweep."""
    rows, cols = np.indices(shape)
    q, t = cols-slope*rows, rows+slope*cols
    selected = sorted(selected, key=lambda i: (specs[i][0], specs[i][1]))
    filled = np.zeros(shape, dtype=bool)
    for rank, i in enumerate(selected):
        others = selected[rank+1:rank+2] if alternative == 'consecutive' else selected[rank+1:]
        for j in others:
            qa, la, ha, ca = specs[i]
            qb, lb, hb, cb = specs[j]
            if alternative == 'same_color' and ca != cb:
                continue
            part = ((t >= max(la, lb)) & (t <= min(ha, hb))
                    & (q > min(qa, qb)) & (q < max(qa, qb)))
            if alternative == 'pixel_centers':
                part &= ((t-qa) % 2 == 0) & ((t-qb) % 2 == 0)
            filled |= part
    return filled


def _paint(info, alternative=''):
    specs, selected = info['specs'], info['selected']
    if alternative == 'all_strokes':
        selected = list(range(len(specs)))
    elif alternative == 'crop_clipped':
        y, x, h, w = info['window']
        clipped = []
        for i in selected:
            rr, cc = np.where(info['labels'][y:y+h, x:x+w] == i+1)
            ts = rr+y+info['slope']*(cc+x)
            q, _, _, color = specs[i]
            clipped.append((q, int(ts.min()), int(ts.max()), color))
        specs, selected = clipped, list(range(len(clipped)))
    result = info['grid'].copy()
    mask = _fill(result.shape, info['slope'], specs, selected, alternative)
    result[mask & (result == info['bg']) & ~info['card']] = info['paint']
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recognize the key, activate touched whole strokes, and fill their loom."""
    return _paint(_inspect(input_grid))


def _query_operations(info):
    y, x, h, w = info['window']
    patch = info['source'][y:y+h, x:x+w].astype(np.int8)
    query = (info['query'] == np.unique(info['query'])[1]).astype(np.int8)
    return {k for k in range(8) for exchange in (0, 1)
            if np.array_equal(_d4(patch, k) ^ exchange, query)}


def _sweep(info):
    """Label construction uses active extrema, separately from pairwise solve."""
    rows, cols = np.indices(info['grid'].shape)
    q, t = cols-info['slope']*rows, rows+info['slope']*cols
    out = info['grid'].copy()
    for level in range(int(t.min()), int(t.max())+1):
        active = [info['specs'][i][0] for i in info['selected']
                  if info['specs'][i][1] <= level <= info['specs'][i][2]]
        if len(active) >= 2:
            paint = ((t == level) & (q > min(active)) & (q < max(active))
                     & (out == info['bg']) & ~info['card'])
            out[paint] = info['paint']
    return out


def _geometry(rng, witness):
    n = int(rng.integers(4 if witness else 3, 8))
    qs = np.r_[0, np.cumsum(rng.integers(3, 8, size=n-1))]
    mode = int(rng.integers(4))
    lengths = rng.integers(2, 11, size=n)
    if mode == 0:
        lengths[:] = int(rng.integers(3, 9))
    specs = []
    for q, length in zip(qs, lengths):
        lo = int(rng.integers(-10, 5))
        lo -= (lo-int(q)) % 2
        if mode == 1:
            lo = int(q) % 2-8
        elif mode == 2:
            lo = int(q) % 2-2*int(length)+2
        specs.append([int(q), lo, lo+2*(int(length)-1), 1])
    if witness:
        i = int(rng.integers(1, n-1))
        short_lo = int(qs[i]) % 2-2
        specs[i][1:3] = [short_lo, short_lo+2]
        for j in (i-1, i+1):
            lo = int(qs[j]) % 2-int(rng.choice([8, 10, 12]))
            specs[j][1:3] = [lo, lo+2*int(rng.integers(8, 12))]
    slope = int(rng.choice([-1, 1]))
    strokes = []
    for q, lo, hi, _ in specs:
        cells = []
        for t in range(lo, hi+1, 2):
            cells.append(((t-slope*q)//2, (q+slope*t)//2))
        strokes.append(cells)
    cells = np.array([p for stroke in strokes for p in stroke])
    r0, c0 = cells.min(axis=0)
    rh, cw = cells.max(axis=0)-cells.min(axis=0)+1
    if min(rh, cw) < 5 or max(rh, cw) > 24:
        return None
    H = int(rng.integers(max(22, rh), min(30, max(22, rh+10))+1))
    W = int(rng.integers(max(22, cw), min(30, max(22, cw+10))+1))
    top, left = int(rng.integers(H-rh+1)), int(rng.integers(W-cw+1))
    labels = np.zeros((H, W), dtype=np.int16)
    for i, stroke in enumerate(strokes):
        for r, c in stroke:
            labels[int(r-r0)+top, int(c-c0)+left] = i+1
    # Read the actual translated intervals, so no off-canvas state exists.
    grid = (labels > 0).astype(np.int8)
    actual_slope, actual_specs, labels = _strokes(grid, labels > 0)
    assert actual_slope == slope
    return labels, actual_specs, slope, (top, left, int(rh), int(cw))


def _place_card(rng, source, height, width):
    # One blank cell around the card separates its role from the source ink.
    h, w = height+2, width+2
    blocked = np.pad(source, 1, constant_values=False)
    halo = np.lib.stride_tricks.sliding_window_view(blocked, (h+2, w+2))
    valid = np.argwhere(~np.any(halo, axis=(-2, -1)))
    if not len(valid):
        return None
    r, c = valid[int(rng.integers(len(valid)))]
    return int(r), int(c), h, w


def _make_pair(rng, query_transform, exchange, witness, small, area_limit, seen,
               forbidden_paint=None, teach_geometry=False):
    """Finite rejection over readable strokes and exact identifying swatches."""
    for attempt in range(160):
        geom = _geometry(rng, witness)
        if geom is None:
            continue
        labels, specs, slope, (top, left, rh, cw) = geom
        source = labels > 0
        palette = [int(v) for v in rng.permutation(10)]
        if forbidden_paint is not None and palette[1] == forbidden_paint:
            palette[1], palette[7] = palette[7], palette[1]
        bg, paint, qa, qb = palette[:4]
        marker_count = 2 if witness else int(rng.integers(1, 4))
        marker_colors = palette[4:4+marker_count]
        base = np.full(labels.shape, bg, dtype=np.int8)
        for i in range(len(specs)):
            color = marker_colors[i % marker_count]
            specs[i] = (*specs[i][:3], color)
            base[labels == i+1] = color
        blank_card = np.zeros(base.shape, dtype=bool)
        for trial in range(72):
            low, high = (3, 8) if small else (5, 11)
            h = int(rng.integers(low, min(rh, high)+1))
            w = int(rng.integers(low, min(cw, high)+1))
            if h*w >= area_limit:
                continue
            y = top+int(rng.integers(rh-h+1))
            x = left+int(rng.integers(cw-w+1))
            crop_labels = labels[y:y+h, x:x+w]
            selected = sorted(int(v)-1 for v in np.unique(crop_labels) if v)
            if not 2 <= len(selected) < len(specs):
                continue
            # At least two selected objects continue outside the visible crop.
            if sum(np.sum(crop_labels == i+1) < np.sum(labels == i+1) for i in selected) < 2:
                continue
            info = dict(grid=base, bg=bg, paint=paint, card=blank_card,
                        labels=labels, source=source, specs=specs, slope=slope,
                        selected=selected, window=(y, x, h, w))
            expected = _paint(info)
            if np.array_equal(expected, base):
                continue
            outside = np.ones(base.shape, dtype=bool)
            outside[y:y+h, x:x+w] = False
            if not np.any((expected != base) & outside):
                continue
            required = ['all_strokes', 'crop_clipped']
            if witness:
                required += ['consecutive', 'same_color', 'pixel_centers']
            if any(np.array_equal(expected, _paint(info, alt)) for alt in required):
                continue
            binary = source[y:y+h, x:x+w].astype(np.int8)
            if teach_geometry and len({_key(_d4(binary, k) ^ swap)
                                       for k in range(8) for swap in (0, 1)}) != 16:
                continue
            query = _d4(binary ^ exchange, query_transform)
            placement = _place_card(rng, source, *query.shape)
            if placement is None:
                continue
            r, c, ch, cw_card = placement
            grid = base.copy()
            grid[r:r+ch, c:c+cw_card] = paint
            grid[r+1:r+ch-1, c+1:c+cw_card-1] = np.where(query, qb, qa)
            if _key(grid) in seen:
                continue
            # Card placement cannot occlude or introduce a possible match.
            try:
                final = _inspect(grid)
            except ValueError:
                continue
            if final['window'] != (y, x, h, w):
                continue
            out = _sweep(final)
            if not np.any((out != grid) & outside):
                continue
            if any(np.array_equal(out, _paint(final, alt)) for alt in required):
                continue
            seen.add(_key(grid))
            return {'input': grid, 'output': out}
    raise RuntimeError('bounded loom construction exhausted')


def _integer(value, name, lower, upper=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or int(value) < lower or (upper is not None and int(value) > upper)):
        raise ValueError(f'illegal {name}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    nt = _integer(num_train, 'num_train', 2, 10)
    nq = _integer(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    turn, reflection = int(rng.choice([1, 3])), int(rng.integers(4, 8))
    train_pool = [0, turn, reflection]
    if nt >= 4 and rng.random() < .6:
        train_pool.append(int(rng.choice([k for k in range(8) if k not in train_pool])))
    test_pool = [k for k in range(8) if k not in train_pool]
    first_paint = None
    for i in range(nt):
        # A quarter turn and a reflection teach generators of D4 even at 2/1.
        transform = (turn, reflection)[i] if i < 2 else int(rng.choice(train_pool))
        pair = _make_pair(rng, transform, i % 2 if i < 2 else int(rng.integers(2)),
                          i == 1, False, 122, seen, first_paint if i == 1 else None,
                          teach_geometry=i < 2)
        puzzle['train'].append(pair)
        if i == 0:
            first_paint = _inspect(pair['input'])['paint']
    train_areas = [_inspect(p['input'])['query'].size for p in puzzle['train']]
    for i in range(nq):
        # Smaller identifying keys recombine already taught whole-object filling.
        transform = int(rng.choice(test_pool))
        puzzle['test'].append(_make_pair(rng, transform, int(rng.integers(2)), False,
                                          True, max(train_areas), seen))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, infos, witnesses = set(), {'train': [], 'test': []}, {'train': set(), 'test': set()}
    alternatives = ('all_strokes', 'crop_clipped', 'consecutive', 'pixel_centers', 'same_color')
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _inspect(inp)
            infos[split].append(info)
            assert len(info['selected']) < len(info['specs']), 'generated examples need a distractor'
            assert np.array_equal(out, _paint(info))
            assert np.array_equal(out, _sweep(info))
            changed = inp != out
            assert np.any(changed) and np.all(inp[changed] == info['bg'])
            assert np.all(out[changed] == info['paint'])
            assert np.array_equal(out[info['card'] | info['source']], inp[info['card'] | info['source']])
            y, x, h, w = info['window']
            window_mask = np.zeros(inp.shape, dtype=bool)
            window_mask[y:y+h, x:x+w] = True
            assert np.any(changed & ~window_mask), 'whole strokes must act beyond the crop'
            crop_labels = info['labels'][y:y+h, x:x+w]
            assert sum(np.sum(crop_labels == i+1) < np.sum(info['labels'] == i+1)
                       for i in info['selected']) >= 2
            for alt in alternatives:
                if not np.array_equal(out, _paint(info, alt)):
                    witnesses[split].add(alt)
            assert not np.array_equal(out, _paint(info, 'all_strokes')), 'A must change boundary membership'
            assert not np.array_equal(out, _paint(info, 'crop_clipped')), 'activation must use whole objects'
    assert set(alternatives) <= witnesses['train'], 'training must distinguish all documented shortcuts'
    assert witnesses['test'] <= witnesses['train']
    assert len({i['paint'] for i in infos['train']}) >= 2, 'frame selects the paint role'
    demonstrated = [next(iter(ops)) for i in infos['train']
                    if len(ops := _query_operations(i)) == 1]
    assert any(k in (1, 3) for k in demonstrated), 'need an unambiguous quarter-turn demonstration'
    assert any(k >= 4 for k in demonstrated), 'need an unambiguous reflection demonstration'
    assert all(i['query'].size < max(j['query'].size for j in infos['train']) for i in infos['test'])


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                grid = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeds, aggregate-only retention, analytical and equivariance checks."""
    # Two length-three NE/SW strokes bracket seven cells, four between centers.
    fixture = np.zeros((5, 5), dtype=np.int8)
    fixture[(0, 1, 2, 2, 3, 4), (2, 1, 0, 4, 3, 2)] = 3
    slope, specs, labels = _strokes(fixture, fixture != 0)
    mask = _fill(fixture.shape, slope, specs, [0, 1]) & (fixture == 0)
    expected = np.zeros_like(mask)
    expected[(1, 1, 2, 2, 2, 3, 3), (2, 3, 1, 2, 3, 1, 2)] = True
    assert np.array_equal(mask, expected)
    assert int(np.sum(mask)) == 7
    assert not np.array_equal(mask, _fill(fixture.shape, slope, specs, [0, 1], 'pixel_centers'))
    fingerprints, shapes, geometries, crops, counts, selected_counts, slopes, paints = (set() for _ in range(8))
    nonconsecutive = edge = changed_cells = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _inspect(pair['input'])
                shapes.add(info['grid'].shape)
                crops.add(info['query'].shape)
                counts.add(len(info['specs']))
                selected_counts.add(len(info['selected']))
                slopes.add(info['slope'])
                paints.add(info['paint'])
                rr, cc = np.where(info['source'])
                binary = info['source'][rr.min():rr.max()+1, cc.min():cc.max()+1].astype(np.int8)
                geometries.add(min(_key(_d4(binary, k)) for k in range(8)))
                selected = info['selected']
                nonconsecutive += any(b-a > 1 for a, b in zip(selected, selected[1:]))
                edge += bool(np.any(info['source'][0]) or np.any(info['source'][-1])
                             or np.any(info['source'][:, 0]) or np.any(info['source'][:, -1]))
                changed_cells += int(np.sum(pair['input'] != pair['output']))
        if seed == 0:
            assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
            inp, out = puzzle['test'][0]['input'], puzzle['test'][0]['output']
            for k in range(8):
                assert np.array_equal(solve(_d4(inp, k)), _d4(out, k))
            rename = np.array([7, 4, 8, 0, 2, 9, 1, 6, 3, 5], dtype=np.int8)
            assert np.array_equal(solve(rename[inp]), rename[out])
            bad = {s: [{r: a.copy() for r, a in p.items()} for p in puzzle[s]] for s in puzzle}
            row, col = np.argwhere(inp != out)[0]
            bad['test'][0]['output'][row, col] = inp[row, col]
            try:
                validate_generated_puzzle(bad)
            except AssertionError:
                pass
            else:
                raise AssertionError('corrupt output accepted')
    assert len(fingerprints) == 42
    assert len(shapes) >= 20 and len(geometries) >= 160 and len(crops) >= 25
    assert {3, 4, 5, 6, 7} <= counts and {2, 3, 4} <= selected_counts
    assert slopes == {-1, 1} and len(paints) >= 8
    assert nonconsecutive > 0 and edge > 0
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(12345, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return dict(seeds=42, pairs=210, distinct_puzzles=len(fingerprints), grid_shapes=len(shapes),
                reference_geometry_classes=len(geometries), crop_shapes=len(crops),
                stroke_counts=sorted(counts), selected_counts=sorted(selected_counts),
                slopes=sorted(slopes), paint_colors=len(paints),
                nonconsecutive_selections=nonconsecutive, source_edge_contacts=edge,
                changed_cells=changed_cells, independent_solver=False)
