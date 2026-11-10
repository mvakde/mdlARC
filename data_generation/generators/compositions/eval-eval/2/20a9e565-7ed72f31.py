"""Mirror-echo orbits with extrapolated symmetry gates and an inclusive L crop.

Generation renders explicit latent tiles with array flips. The input reader
recovers gates and verifies cellwise transformations; solve uses only that input.
"""
from __future__ import annotations

import numpy as np


HOST_SEEDS = (0, 63849537, 778955804, 619380952, 882247517, 402520677,
              176941504, 606181469, 996841958, 595108341, 395547284,
              634950357, 483827698, 760699239, 414942468, 661265453,
              361308622, 523515323, 926364870, 787703902, 327557153,
              431424604, 832905900, 727526025, 414939741, 737672378,
              445885623, 581822949, 915760104, 358228616, 348447694,
              271126368, 829561710, 337279327, 348287992, 498464209,
              644562571, 239916772, 472293752, 543212975, 269056894,
              21128446)


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(name + ' is out of range')
    return int(value)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError('Expected an integer ARC grid of dimensions 1..30')
    return a.astype(np.int64, copy=True)


def _key(a):
    return a.shape, tuple(map(int, a.flat))


def _corners(bounds):
    r0, r1, c0, c1 = bounds
    return {(r0, c0), (r0, c0 + 1), (r0 + 1, c0),
            (r1, c1), (r1, c1 - 1), (r1 - 1, c1)}


def _find_window(a):
    found = []
    for color in np.unique(a):
        cells = np.argwhere(a == color)
        if len(cells) != 6:
            continue
        r0, c0 = map(int, cells.min(axis=0))
        r1, c1 = map(int, cells.max(axis=0))
        if r1 <= r0 or c1 <= c0 or r1-r0+c1-c0 < 4:
            continue
        bounds = r0, r1, c0, c1
        if set(map(tuple, cells)) == _corners(bounds):
            found.append((int(color), bounds))
    if len(found) != 1:
        raise ValueError('The two canonical L corners must have a unique color')
    return found[0]


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, comp = [start], []
        while stack:
            r, c = stack.pop()
            comp.append((int(r), int(c)))
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(comp)
    return result


def _bounds(cells):
    return (min(r for r, c, v in cells), max(r for r, c, v in cells),
            min(c for r, c, v in cells), max(c for r, c, v in cells))


def _reflect(cells, axis, baseline, kind):
    return sorted((2*baseline-r if kind == 'P' else r, 2*axis-c, v)
                  for r, c, v in cells)


def _translate_to_bbox(cells, dest):
    """Competing rule: move an unchanged glyph into the true destination box."""
    r0, _, c0, _ = _bounds(cells)
    rr, _, cc, _ = _bounds(dest)
    return sorted((r+rr-r0, c+cc-c0, v) for r, c, v in cells)


def _canonical_model(a, bounds, bg, crop_color, gate_color):
    """Read a horizontal chain. Return None unless every input cell fits it."""
    components = _components(a == gate_color)
    if len(components) < 3:
        return None
    gates = []
    for comp in components:
        rows = sorted(r for r, c in comp)
        cols = {c for r, c in comp}
        if len(cols) != 1 or len(rows) % 2 != 1 or rows != list(range(rows[0], rows[-1]+1)):
            return None
        gates.append((next(iter(cols)), (rows[0]+rows[-1])//2,
                      'P' if len(rows) == 1 else 'L', len(rows)))
    gates.sort()
    if len({g[0] for g in gates}) != len(gates) or len({g[1] for g in gates}) != 1:
        return None
    axes = [g[0] for g in gates]
    kinds = [g[2] for g in gates]
    period = 1 if len(set(kinds)) == 1 else 2
    if any(k != kinds[i % period] for i, k in enumerate(kinds)):
        return None
    increments = np.diff(axes)
    curvature = int(increments[1] - increments[0])
    if not np.all(np.diff(increments) == curvature):
        return None
    content = (a != bg) & (a != crop_color) & (a != gate_color)
    active = np.flatnonzero(np.any(content, axis=0))
    if not len(active):
        return None
    runs = np.split(active, np.flatnonzero(np.diff(active) != 1)+1)
    if len(runs) != len(gates)+1:
        return None
    echoes = []
    for run in runs:
        pos = np.argwhere(content[:, run[0]:run[-1]+1])
        echoes.append(sorted((int(r), int(c+run[0]), int(a[r, c+run[0]])) for r, c in pos))
    widths = [int(run[-1]-run[0]+1) for run in runs]
    if len(set(widths)) != 1:
        return None
    for i, g in enumerate(gates):
        if not int(runs[i][-1]) < g[0] < int(runs[i+1][0]):
            return None
        if _reflect(echoes[i], g[0], g[1], g[2]) != echoes[i+1]:
            return None
    r0, r1, c0, c1 = bounds
    if not (int(runs[-1][-1]) < c0 or int(runs[0][0]) > c1):
        return None
    return dict(grid=a, bounds=bounds, background=bg, crop_color=crop_color,
                gate_color=gate_color, axes=axes, kinds=kinds[:period], period=period,
                baseline=gates[0][1], curvature=curvature, first_step=int(increments[0]),
                echoes=echoes, lengths=[g[3] for g in gates], width=widths[0],
                forward=int(runs[-1][-1]) < c0)


def _axis_at(model, i, linear=False):
    axes = model['axes']
    if linear:
        anchor = len(axes)-1 if model['forward'] else 0
        step = axes[-1]-axes[-2] if model['forward'] else axes[1]-axes[0]
        return axes[anchor] + (i-anchor)*step
    return axes[0] + i*model['first_step'] + i*(i-1)//2*model['curvature']


def _project(model, *, linear=False, unturned=False, fixed_kind=None, copy_initial=False):
    r0, r1, c0, c1 = model['bounds']
    answer = np.full((r1-r0+1, c1-c0+1), model['background'], dtype=np.int64)
    forward = model['forward']
    cells = model['echoes'][-1] if forward else model['echoes'][0]
    original = (model['echoes'][0] if forward else model['echoes'][-1]) if copy_initial else cells
    i = len(model['axes']) if forward else -1
    records = []
    painted = set()
    for _ in range(64):
        axis = _axis_at(model, i, linear)
        kind = fixed_kind or model['kinds'][i % model['period']]
        dest = _reflect(cells, axis, model['baseline'], kind)
        before, after = _bounds(cells), _bounds(dest)
        if (forward and after[2] <= before[3]) or (not forward and after[3] >= before[2]):
            # Extrapolation stops when the spatial progression ceases to be valid.
            if linear:
                break
            raise ValueError('The continuation overlaps or reverses direction')
        if (forward and after[2] > c1) or (not forward and after[3] < c0):
            break
        shown = _translate_to_bbox(original, dest) if unturned or copy_initial else dest
        touched = False
        for r, c, v in shown:
            if r0 <= r <= r1 and c0 <= c <= c1:
                if (r, c) in painted:
                    raise ValueError('Echoes collide in the crop')
                painted.add((r, c))
                answer[r-r0, c-c0] = v
                touched = True
        records.append(dict(index=i+1 if forward else i, kind=kind,
                            axis=axis, bounds=after, cells=dest, intersects=touched))
        cells = dest
        if (forward and after[3] >= c1) or (not forward and after[2] <= c0):
            break
        i += 1 if forward else -1
    else:
        raise ValueError('Continuation did not reach the requested window')
    return answer, records


def _read(input_grid):
    a = _grid(input_grid)
    crop_color, bounds = _find_window(a)
    colors, counts = np.unique(a, return_counts=True)
    if np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError('Background must be uniquely dominant')
    bg = int(colors[counts.argmax()])
    if bg == crop_color:
        raise ValueError('Background and annotations must differ')
    r0, r1, c0, c1 = bounds
    region = a[r0:r1+1, c0:c1+1]
    if not np.all((region == bg) | (region == crop_color)):
        raise ValueError('The entire requested rectangle must be erased')
    models = []
    for transposed in (False, True):
        canonical = a.T if transposed else a
        bb = (c0, c1, r0, r1) if transposed else bounds
        for color in colors:
            color = int(color)
            if color in (bg, crop_color):
                continue
            model = _canonical_model(canonical, bb, bg, crop_color, color)
            if model is None:
                continue
            try:
                predicted, records = _project(model)
            except ValueError:
                continue
            if not np.any(predicted != bg):
                continue
            model['transposed'] = transposed
            model['records'] = records
            models.append(model)
    if len(models) != 1:
        raise ValueError('Input must uniquely identify one symmetry orbit')
    return models[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct the requested crop from one input, without any generation state."""
    model = _read(input_grid)
    answer, _ = _project(model)
    return answer.T.copy() if model['transposed'] else answer.copy()


def _tile(rng, bg, palette, width=0):
    h = int(rng.integers(2, 8))
    w = width or int(rng.integers(2, 5))
    mask = np.zeros((h, w), dtype=bool)
    if rng.random() < .22:
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
        mask[int(rng.choice([0, h-1])), int(rng.choice([0, w-1]))] = False
    else:
        # An eight-connected spanning walk, then local growth. Internal holes,
        # sparse diagonal links and dense irregular bodies all remain possible.
        row = int(rng.integers(h))
        for col in range(w):
            mask[row, col] = True
            row = max(0, min(h-1, row + int(rng.integers(-1, 2))))
        target = int(rng.integers(max(w, h*w//4), h*w+1))
        for _ in range(target):
            frontier = sorted({(int(r)+dr, int(c)+dc)
                               for r, c in np.argwhere(mask)
                               for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                               if 0 <= r+dr < h and 0 <= c+dc < w and not mask[r+dr, c+dc]})
            if not frontier:
                break
            mask[frontier[int(rng.integers(len(frontier)))]] = True
    active = np.flatnonzero(np.any(mask, axis=1))
    mask = mask[active[0]:active[-1]+1]
    if mask.shape[0] < 2:
        return None
    a = np.full(mask.shape, bg, dtype=np.int64)
    p = int(rng.integers(1, min(3, len(palette))+1))
    chosen = rng.choice(palette, size=p, replace=False)
    a[mask] = rng.choice(chosen, size=int(mask.sum()))
    # Geometric/color witnesses for both operations, including homogeneous chains.
    if np.array_equal(a, a[:, ::-1]) or np.array_equal(a, a[::-1, ::-1]):
        return None
    return a


def _make_pair(rng, config, plan, fallback=None):
    bg = config['backgrounds'][0] if fallback is not None else int(rng.choice(config['backgrounds']))
    crop_color, gate_color = config['crop'], config['gate']
    if fallback is None:
        tile = _tile(rng, bg, config['payloads'], width=2 if plan['growth'] else 0)
        if tile is None:
            return None
        nvisible = plan.get('visible') or (4 if plan['growth'] else int(rng.integers(4, 7)))
        delta = 2 if plan['growth'] else int(rng.choice([0, 0, 0, 2, -2]))
        s = tile.shape[1]+1 + 2*int(rng.integers(0, 3))
        if delta < 0:
            s += 2*(nvisible+plan['skip']+plan['span']-2)
        if plan['growth']:
            s = 3
        y = int(rng.integers(-tile.shape[0]-1, 2))
    else:
        tile = np.array([[config['payloads'][0], config['payloads'][0]],
                         [config['payloads'][0], bg]], dtype=np.int64)
        nvisible, delta, s, y = plan.get('visible') or 4, 2 if plan['growth'] else 0, 3, -2
    word = list(config['word'])
    # Choose a phase that makes the first hidden operation the requested witness.
    if plan['witness'] and len(word) == 2:
        if word[(nvisible-1) % 2] != plan['witness']:
            word.reverse()
    elif fallback is None and not plan['witness'] and rng.random() < .5:
        word.reverse()
    h, w = tile.shape
    total = nvisible + plan['skip'] + plan['span']
    xs = [i*s + delta*i*(i-1)//2 for i in range(total)]
    if any(xs[i+1]-xs[i] < w+1 for i in range(total-1)):
        return None
    tiles, gates = [(y, xs[0], tile.copy())], []
    current, top = tile.copy(), y
    for i in range(total-1):
        twice_axis = xs[i]+xs[i+1]+w-1
        if twice_axis % 2:
            return None
        kind = word[i % len(word)]
        # Latent rendering uses local array flips, not solver cell mappings.
        current = current[:, ::-1].copy()
        if kind == 'P':
            current = current[::-1].copy()
            top = -top-h+1
        length = 1 if kind == 'P' else (3 if fallback is not None else int(rng.choice([3, 5, 7, 9])))
        gates.append((twice_axis//2, kind, length))
        tiles.append((top, xs[i+1], current.copy()))
    requested = tiles[nvisible+plan['skip']:]
    c0, c1 = requested[0][1], requested[-1][1]+w-1
    r0 = min(t[0] for t in requested)
    r1 = max(t[0]+h-1 for t in requested)
    if plan['slice'] and len(requested) > 1:
        c0 += 1 if fallback is not None else int(rng.integers(0, w))
        c1 -= 0 if fallback is not None else int(rng.integers(0, w))
        if fallback is None and r1-r0 >= 3:
            a = int(rng.integers(r0, r1-1))
            r0, r1 = a, int(rng.integers(a+1, r1+1))
    elif fallback is None:
        r0 -= int(rng.integers(0, 2))
        r1 += int(rng.integers(0, 2))
    while r1-r0+c1-c0 < 4:
        r0 -= 1
    visible = tiles[:nvisible]
    ymin = min([r0] + [t[0] for t in visible] + [-g[2]//2+1 for g in gates[:nvisible-1]])
    ymax = max([r1] + [t[0]+h-1 for t in visible] + [g[2]//2 for g in gates[:nvisible-1]])
    xmin, xmax = 0, c1
    height, width = ymax-ymin+1, xmax-xmin+1
    if max(height, width) > 30:
        return None
    if fallback is None:
        pad_top = int(rng.integers(0, min(3, 30-height)+1))
        pad_left = int(rng.integers(0, min(3, 30-width)+1))
        pad_bottom = int(rng.integers(0, min(3, 30-height-pad_top)+1))
        pad_right = int(rng.integers(0, min(3, 30-width-pad_left)+1))
    else:
        pad_top, pad_left = fallback//4, fallback % 4
        pad_bottom = pad_right = 0
    if height+pad_top+pad_bottom > 30 or width+pad_left+pad_right > 30:
        return None
    canvas = np.full((height+pad_top+pad_bottom, width+pad_left+pad_right), bg, dtype=np.int64)
    dr, dc = pad_top-ymin, pad_left
    for yy, xx, a in visible:
        canvas[yy+dr:yy+dr+h, xx+dc:xx+dc+w] = a
    for x, kind, length in gates[:nvisible-1]:
        canvas[dr-length//2:dr+length//2+1, x+dc] = gate_color
    bb = r0+dr, r1+dr, c0+dc, c1+dc
    # Labels are crops of explicit latent states, not calls to solve().
    answer = np.full((r1-r0+1, c1-c0+1), bg, dtype=np.int64)
    for yy, xx, a in requested:
        for rr, cc in np.argwhere(a != bg):
            r, c = yy+int(rr), xx+int(cc)
            if r0 <= r <= r1 and c0 <= c <= c1:
                answer[r-r0, c-c0] = a[rr, cc]
    if not np.any(answer != bg):
        return None
    if plan['reverse']:
        canvas = canvas[:, ::-1].copy()
        answer = answer[:, ::-1].copy()
        bb = bb[0], bb[1], canvas.shape[1]-1-bb[3], canvas.shape[1]-1-bb[2]
    if plan['axis']:
        canvas, answer = canvas.T.copy(), answer.T.copy()
        bb = bb[2], bb[3], bb[0], bb[1]
    for q in _corners(bb):
        canvas[q] = crop_color
    try:
        model = _read(canvas)
    except ValueError:
        return None
    predicted = _project(model)[0]
    if model['transposed']:
        predicted = predicted.T
    if not np.array_equal(predicted, answer):
        raise AssertionError('Latent tile rendering disagrees with the input model')
    if plan['growth'] and np.array_equal(_project(model, linear=True)[0], _project(model)[0]):
        return None
    if plan['witness']:
        true = _project(model)[0]
        other = 'P' if plan['witness'] == 'L' else 'L'
        if (np.array_equal(_project(model, unturned=True)[0], true)
                or np.array_equal(_project(model, fixed_kind=other)[0], true)):
            return None
    if plan.get('visible') == 5 and np.array_equal(_project(model, copy_initial=True)[0], _project(model)[0]):
        return None
    return {'input': canvas, 'output': answer}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    colors = list(map(int, rng.permutation(10)))
    word = (('L',), ('P',), ('L', 'P'))[int(rng.choice(3, p=[.18, .18, .64]))]
    config = dict(backgrounds=colors[:1 if rng.random() < .65 else 2],
                  crop=colors[2], gate=colors[3], payloads=colors[4:], word=word)
    shift = str(rng.choice(['span', 'skip', 'reverse', 'axis', 'slice']))
    axis, reverse = bool(rng.integers(2)), bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            test = split == 'test'
            plan = dict(growth=not test and i == 0,
                        visible=5 if not test and i == 1 else None,
                        witness=word[i % len(word)] if not test and i < len(word) else None,
                        axis=not axis if test and shift == 'axis' else axis,
                        reverse=not reverse if test and shift == 'reverse' else reverse,
                        span=2 if test and shift in ('span', 'slice') else 1,
                        skip=1 if test and shift == 'skip' else 0,
                        slice=test and shift == 'slice')
            # Additional demonstrations can vary count and shape while the first
            # two establish curvature and every active symmetry operation.
            if not test and i >= 2:
                plan['span'] = int(rng.choice([1, 1, 2]))
                plan['skip'] = int(rng.choice([0, 0, 1]))
                plan['slice'] = plan['span'] == 2 and rng.random() < .3
                if shift != 'axis' and rng.random() < .3:
                    plan['axis'] = not axis
                if shift != 'reverse' and rng.random() < .3:
                    plan['reverse'] = not reverse
            pair = None
            for attempt in range(216):
                candidate = _make_pair(rng, config, plan,
                                       fallback=attempt-200 if attempt >= 200 else None)
                if candidate is not None and _key(candidate['input']) not in seen:
                    pair = candidate
                    break
            if pair is None:
                raise RuntimeError('Bounded construction exhausted its fallback pool')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {'train', 'test'}:
        raise AssertionError('Puzzle must have exactly train and test')
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, roles, taught, witnessed, classified = set(), set(), set(), set(), set()
    curvature_witness, initial_witness = False, False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'Duplicate input'
            seen.add(_key(inp))
            model = _read(inp)
            roles.add((model['crop_color'], model['gate_color']))
            expected, records = _project(model)
            canon_out = out.T if model['transposed'] else out
            assert np.array_equal(expected, canon_out), 'Incorrect echo crop'
            assert model['crop_color'] not in out and model['gate_color'] not in out
            assert set(map(int, np.unique(out))) <= set(map(int, np.unique(inp)))
            assert len(model['echoes']) >= 4
            assert np.any(out != model['background'])
            assert inp.shape != out.shape
            if split == 'train':
                taught.update(model['kinds'])
                no_turn, _ = _project(model, unturned=True)
                for rec in records:
                    if rec['intersects'] and not np.array_equal(no_turn, expected):
                        witnessed.add(rec['kind'])
                    other = 'P' if rec['kind'] == 'L' else 'L'
                    if rec['intersects'] and not np.array_equal(_project(model, fixed_kind=other)[0], expected):
                        classified.add(rec['kind'])
                curvature_witness |= not np.array_equal(_project(model, linear=True)[0], expected)
                initial_witness |= not np.array_equal(_project(model, copy_initial=True)[0], expected)
            else:
                assert set(model['kinds']) <= taught, 'Test introduces an undemonstrated gate type'
    assert len(roles) == 1, 'Annotation roles must remain stable inside a puzzle'
    assert taught <= witnessed, 'Training must distinguish orientation-preserving translation'
    assert taught <= classified, 'Training must distinguish line reflection from point rotation'
    assert curvature_witness, 'Training must distinguish fixed spacing from changing placement'
    assert initial_witness, 'Training must distinguish an initial-glyph copy from the accumulated orbit'


def _puzzle_key(p):
    return tuple((_key(np.asarray(pair['input'])), _key(np.asarray(pair['output'])))
                 for s in ('train', 'test') for pair in p[s])


def self_test():
    """A bounded 42-seed suite, including actual geometry and omission witnesses."""
    stats = dict(puzzles=set(), shapes=set(), silhouettes=set(), curves=set(),
                 directions=set(), words=set(), backgrounds=set(), roles=set(),
                 counts=set(), widths=set(), lengths=set(), spans=set())
    totals = dict(partial=0, skipped=0, multicolor=0, multi_echo=0, positive_curvature=0, negative_curvature=0)
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        stats['puzzles'].add(_puzzle_key(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                m = _read(pair['input'])
                true, records = _project(m)
                assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
                source = m['echoes'][0]
                rr0, rr1, cc0, cc1 = _bounds(source)
                normalized = tuple((r-rr0, c-cc0) for r, c, v in source)
                stats['silhouettes'].add(normalized)
                stats['shapes'].add((pair['input'].shape, pair['output'].shape))
                stats['curves'].add(m['curvature'])
                stats['directions'].add((m['transposed'], m['forward']))
                stats['words'].add(tuple(m['kinds']))
                stats['backgrounds'].add(m['background'])
                stats['roles'].add((m['crop_color'], m['gate_color']))
                stats['counts'].add(len(m['echoes']))
                stats['widths'].add(m['width'])
                stats['lengths'].update(m['lengths'])
                touched = [rec for rec in records if rec['intersects']]
                stats['spans'].add(len(touched))
                totals['multi_echo'] += len(touched) > 1
                totals['skipped'] += bool(records and not records[0]['intersects'])
                totals['multicolor'] += len(set(map(int, np.unique(true)))-{m['background']}) > 1
                totals['positive_curvature'] += m['curvature'] > 0
                totals['negative_curvature'] += m['curvature'] < 0
                r0, r1, c0, c1 = m['bounds']
                totals['partial'] += any(r0 > rec['bounds'][0] or r1 < rec['bounds'][1]
                                         or c0 > rec['bounds'][2] or c1 < rec['bounds'][3]
                                         for rec in touched)
                # Cellwise input relationships are separately checked against
                # cropped array flips, using arbitrary colored silhouettes.
                for i, axis in enumerate(m['axes']):
                    old, new = m['echoes'][i], m['echoes'][i+1]
                    old_bb, new_bb = _bounds(old), _bounds(new)
                    patch = np.full((old_bb[1]-old_bb[0]+1, old_bb[3]-old_bb[2]+1), m['background'])
                    for r, c, v in old:
                        patch[r-old_bb[0], c-old_bb[2]] = v
                    patch = np.fliplr(patch)
                    if m['kinds'][i % m['period']] == 'P':
                        patch = np.flipud(patch)
                    compare = np.full(patch.shape, m['background'])
                    for r, c, v in new:
                        compare[r-new_bb[0], c-new_bb[2]] = v
                    assert np.array_equal(patch, compare)
    assert len(stats['puzzles']) == 42
    assert len(stats['shapes']) >= 80 and len(stats['silhouettes']) >= 50
    assert stats['directions'] == {(False, False), (False, True), (True, False), (True, True)}
    assert {('L',), ('P',), ('L', 'P'), ('P', 'L')} <= stats['words']
    assert len(stats['backgrounds']) >= 7 and len(stats['roles']) >= 25
    assert len(stats['counts']) >= 2 and len(stats['widths']) >= 3
    assert {1, 3, 5, 7, 9} <= stats['lengths'] and {1, 2} <= stats['spans']
    assert all(totals[k] > 0 for k in totals), totals
    # Directly exercise every deterministic fallback member for each operator
    # pool and each crop regime. Distinct dimensions prove duplicate exhaustion
    # impossible, even for maximum-count puzzles with 13 preceding inputs.
    rng = np.random.default_rng(991)
    for word in (('L',), ('P',), ('L', 'P')):
        config = dict(backgrounds=[0], crop=5, gate=6, payloads=[2, 3], word=word)
        for regime in ('growth', 'initial', 'span', 'skip', 'slice', 'reverse'):
            plan = dict(growth=regime == 'growth', witness=word[0] if regime in ('growth', 'initial') else None,
                        visible=5 if regime == 'initial' else None,
                        axis=regime in ('span', 'slice'), reverse=regime == 'reverse',
                        span=2 if regime in ('span', 'slice') else 1,
                        skip=int(regime == 'skip'), slice=regime == 'slice')
            fallback_keys = set()
            for variant in range(16):
                pair = _make_pair(rng, config, plan, fallback=variant)
                assert pair is not None
                assert np.array_equal(solve(pair['input']), pair['output'])
                fallback_keys.add(_key(pair['input']))
            assert len(fallback_keys) == 16
    # All count combinations, determinism at the largest boundary, illegal types.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(77, nt, ne))
    assert _puzzle_key(generate_puzzle(1234, 10, 4)) == _puzzle_key(generate_puzzle(1234, 10, 4))
    for nt, ne in ((1, 1), (11, 1), (2, 2), (3, 4), (4, 0), (6, 5),
                   (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    # Label corruption and gate inconsistency must fail validation.
    p = generate_puzzle(31, 2, 1)
    p['test'][0]['output'][0, 0] = (int(p['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('Corrupt label accepted')
    p = generate_puzzle(32, 2, 1)
    m = _read(p['train'][0]['input'])
    cells = m['echoes'][1]
    r, c, _ = cells[0]
    damaged = p['train'][0]['input'].copy()
    if m['transposed']:
        r, c = c, r
    damaged[r, c] = m['background']
    try:
        solve(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError('Inconsistent visible echoes accepted')
    return {k: len(v) for k, v in stats.items()} | {'features': totals}
