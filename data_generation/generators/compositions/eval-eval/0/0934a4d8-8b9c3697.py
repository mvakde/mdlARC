"""Mirror-locked docks: reconstruct a cap to choose geometrically dockable cargo.

Only the standard library and NumPy are used. The input parser has no dependency
on RNG state or construction metadata. Ground truth is painted separately from
the constructed shaft and chosen original/final footprints.
"""
from __future__ import annotations

from collections import Counter
import numpy as np


_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))  # outward from the base


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append(cells)
    return result


def _rectangle(cells):
    rr, cc = zip(*cells)
    return len(cells) == (max(rr) - min(rr) + 1) * (max(cc) - min(cc) + 1)


def _local(point, dock):
    r, c = point
    dr, dc = dock['out']
    pr, pc = dock['perp']
    return r * dr + c * dc - dock['kmin'], r * pr + c * pc - dock['qmin']


def _cell(dock, k, q):
    k, q = k + dock['kmin'], q + dock['qmin']
    dr, dc = dock['out']
    pr, pc = dock['perp']
    return k * dr + q * pr, k * dc + q * pc


def _recover_cap(shaft, marker, t):
    """Compare actual visible witnesses, then recover in target coordinates.

    A relation applies to this shaft only. Both models may fit if they predict
    the same cap. Conflicting predictions or damaged witnesses are rejected.
    """
    n, s = shaft.shape
    assert n >= 2 * t + 2 and np.all(shaft[-t:] == marker)
    visible = shaft != marker
    assert visible[:-t].all() and not visible[-t:].any()
    predictions, modes, witnesses = [], [], []
    for mode in ('mirror', 'half_turn'):
        reflected = shaft[::-1] if mode == 'mirror' else shaft[::-1, ::-1]
        paired = visible & (reflected != marker)
        if np.count_nonzero(paired) < 2 * s:
            continue
        if not np.array_equal(shaft[paired], reflected[paired]):
            continue
        prediction = reflected[-t:].copy()
        if marker in prediction:
            continue
        predictions.append(prediction)
        modes.append(mode)
        witnesses.append(int(paired.sum()))
    assert predictions, 'No visibly supported shaft symmetry'
    assert all(np.array_equal(predictions[0], b) for b in predictions), 'Ambiguous cap'
    assert len(np.unique(predictions[0])) >= 2, 'Generated keys carry a nonconstant code'
    return predictions[0], tuple(modes), tuple(witnesses)


def _parse_dock(cells, a):
    parses = []
    for dr, dc in _DIRS:
        perp = (-dc, dr)
        kmin = min(r * dr + c * dc for r, c in cells)
        qmin = min(r * perp[0] + c * perp[1] for r, c in cells)
        dock = dict(out=(dr, dc), perp=perp, kmin=kmin, qmin=qmin, cells=cells)
        local = {_local(p, dock) for p in cells}
        width = max(q for _, q in local) + 1
        if {q for k, q in local if k == 0} != set(range(width)):
            continue
        frame = int(a[_cell(dock, 0, 0)])
        if any(int(a[_cell(dock, 0, q)]) != frame for q in range(width)):
            continue
        lengths = [sum((k, q) in local for k in range(1, 31)) for q in range(width)]
        expected = {(0, q) for q in range(width)} | {
            (k, q) for q, length in enumerate(lengths) for k in range(1, length + 1)}
        if local != expected:
            continue
        runs, q = [], 0
        while q < width:
            if lengths[q] == 0:
                q += 1
                continue
            start, length = q, lengths[q]
            while q < width and lengths[q] == length:
                q += 1
            runs.append((start, q, length))
        if len(runs) != 3:
            continue
        if runs[0][0] != 0 or runs[2][1] != width:
            continue
        if runs[0][1] >= runs[1][0] or runs[1][1] >= runs[2][0]:
            continue
        q0, q1, n = runs[1]
        if q0 + q1 != width or runs[0][1] != width - runs[2][0]:
            continue
        if any(int(a[_cell(dock, k, q)]) != frame
               for start, end, length in (runs[0], runs[2])
               for k in range(1, length + 1) for q in range(start, end)):
            continue
        shaft = np.array([[a[_cell(dock, k, q)] for q in range(q0, q1)]
                          for k in range(1, n + 1)], dtype=np.int8)
        if frame in shaft or len(np.unique(shaft)) < 3:
            continue
        marker = int(shaft[-1, 0])
        if not np.all(shaft[-1] == marker):
            continue
        t = 0
        for row in shaft[::-1]:
            if not np.all(row == marker):
                break
            t += 1
        try:
            code, modes, witnesses = _recover_cap(shaft, marker, t)
        except AssertionError:
            continue
        dock.update(width=width, runs=tuple(runs), q0=q0, span=q1-q0,
                    shaft_length=n, cap_length=t, stop=n-t+1,
                    frame=frame, marker=marker, code=code,
                    modes=modes, witnesses=witnesses, shaft=shaft)
        parses.append(dock)
    assert len(parses) == 1, 'A dock must have one supported orientation and cap'
    return parses[0]


def _analyze(value):
    a = _grid(value)
    corners = [int(a[p]) for p in ((0, 0), (0, -1), (-1, 0), (-1, -1))]
    assert len(set(corners)) == 1, 'Background is visible at all four corners'
    background = corners[0]
    docks, tokens = [], []
    for cells in _components(a != background):
        if _rectangle(cells):
            tokens.append(cells)
        else:
            docks.append(_parse_dock(cells, a))
    assert docks and tokens
    markers = {d['marker'] for d in docks}
    assert len(markers) == 1
    marker = markers.pop()
    assert marker != background
    assert all(marker not in [int(a[p]) for p in token] for token in tokens)
    cap_cells = set()
    for d in docks:
        cap_cells |= {_cell(d, k, q) for k in range(d['stop'], d['shaft_length'] + 1)
                      for q in range(d['q0'], d['q0'] + d['span'])}
    assert set(map(tuple, np.argwhere(a == marker))) == cap_cells, 'Reserved marker appears outside caps'
    clean = a.copy()
    for cells in tokens:
        for p in cells:
            clean[p] = background
    matches, used, routes = [], set(), set()
    geometric_impostors, off_axis_keys = 0, 0
    for di, dock in enumerate(docks):
        accepted = []
        for ti, cells in enumerate(tokens):
            local = {_local(p, dock) for p in cells}
            ks, qs = sorted({k for k, _ in local}), sorted({q for _, q in local})
            if len(ks) != dock['cap_length'] or len(qs) != dock['span']:
                continue
            tile = np.array([[a[_cell(dock, k, q)] for q in qs] for k in ks], dtype=np.int8)
            same_code = np.array_equal(tile, dock['code'])
            aligned = qs == list(range(dock['q0'], dock['q0'] + dock['span']))
            if same_code and not aligned and ks[0] > dock['shaft_length']:
                off_axis_keys += 1
            # Original cargo must lie strictly beyond the entire patterned shaft.
            if not aligned or ks[0] <= dock['shaft_length']:
                continue
            route = {_cell(dock, k, q) for k in range(dock['stop'], ks[-1] + 1) for q in qs}
            final = {_cell(dock, k, q) for k in range(dock['stop'], dock['shaft_length'] + 1) for q in qs}
            if any(not (0 <= p[0] < a.shape[0] and 0 <= p[1] < a.shape[1])
                   or (p not in final and int(clean[p]) != background)
                   for p in route):
                continue
            if not same_code:
                geometric_impostors += 1
                continue
            accepted.append(dict(di=di, ti=ti, original=cells, final=final,
                                 route=route, distance=ks[0]-dock['stop'],
                                 start=ks[0], code=tile))
        assert len(accepted) <= 1, 'More than one aligned exact key'
        if accepted:
            match = accepted[0]
            assert match['ti'] not in used, 'Cargo assigned twice'
            assert routes.isdisjoint(match['route']), 'Intersecting accepted sweeps'
            used.add(match['ti'])
            routes |= match['route']
            matches.append(match)
    return dict(input=a, background=background, marker=marker, docks=docks,
                tokens=tokens, clean=clean, matches=matches, cap_cells=cap_cells,
                geometric_impostors=geometric_impostors, off_axis_keys=off_axis_keys)


def _render(record):
    out = record['clean'].copy()
    for m in record['matches']:
        dock = record['docks'][m['di']]
        for p in m['route'] - m['final']:
            out[p] = record['marker']
        for i, row in enumerate(dock['code']):
            for j, color in enumerate(row):
                out[_cell(dock, dock['stop'] + i, dock['q0'] + j)] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Determine background, geometry, cap code, candidates and trail from input."""
    return _render(_analyze(input_grid))


def _shaft(rng, n, s, t, palette, mode, style, require_mode=False):
    colors = np.asarray(palette, dtype=np.int8)
    for _ in range(100):
        h = (n + 1) // 2
        if style == 'random':
            top = rng.choice(colors, size=(h, s))
        elif style == 'bands':
            top = rng.choice(colors, size=(h, 1)) * np.ones((1, s), dtype=np.int8)
            top[:, -1] = rng.choice(colors, size=h)
        else:
            motif = rng.choice(colors, size=(int(rng.integers(2, 4)), s))
            top = np.tile(motif, ((h + len(motif) - 1) // len(motif), 1))[:h].copy()
        if mode == 'half_turn' and n % 2:
            top[-1] = top[-1][np.minimum(np.arange(s), s-1-np.arange(s))]
        full = np.empty((n, s), dtype=np.int8)
        for r in range(n):
            row = top[min(r, n-1-r)]
            full[r] = row[::-1] if mode == 'half_turn' and r >= h else row
        cap = full[-t:]
        if len(np.unique(cap)) < 2:
            continue
        # The false key has the same histogram but the wrong orientation/order.
        wrong = cap[::-1, ::-1].copy()
        if np.array_equal(wrong, cap):
            wrong = cap.copy()
            different = np.flatnonzero(wrong.ravel() != wrong.flat[0])
            j = int(different[0])
            wrong.flat[0], wrong.flat[j] = int(wrong.flat[j]), int(wrong.flat[0])
        masked = full.copy()
        marker = next(c for c in range(10) if c not in palette)
        masked[-t:] = marker
        try:
            recovered, supported, _ = _recover_cap(masked, marker, t)
        except AssertionError:
            continue
        if require_mode:
            if supported != (mode,) or np.array_equal(cap, full[:t]):
                continue
            if mode == 'half_turn' and np.array_equal(cap, full[:t][::-1]):
                continue
        if np.array_equal(recovered, cap):
            return full, wrong
    raise AssertionError('Bounded pattern sampling did not yield an unambiguous cap')


def _module(rng, max_width, colors, active, mode, style, strong=False, demonstrate=False):
    background, marker, frames, palette = colors
    max_span = min(4, (max_width - 2) // 3)
    assert max_span >= 1
    s = 2 if strong else int(rng.integers(2 if demonstrate else 1, max_span + 1))
    t = 2 if strong else (1 if demonstrate else int(rng.integers(2 if s == 1 else 1, 4)))
    n = 2*t + 2 + int(rng.integers(0, 4))
    extra = int(rng.integers(0, min(2, (max_width - (3*s+2))//2) + 1))
    q0 = s + 1 + extra
    width = 2*q0 + s
    guard = int(rng.integers(1, min(2, q0-1) + 1))
    lengths = [int(rng.integers(1, n + 1)), int(rng.integers(1, n + 1))]
    # Both equal and unequal arms, including arms longer than the visible stop.
    if rng.random() < .25:
        lengths[1] = lengths[0]
    frame = int(rng.choice(frames))
    near = n + int(rng.integers(2, 4))
    far = near + t + int(rng.integers(1, 4))
    height = far + t + int(rng.integers(1, 3))
    inp = np.full((height, width), background, dtype=np.int8)
    inp[0] = frame
    inp[1:lengths[0]+1, :guard] = frame
    inp[1:lengths[1]+1, width-guard:] = frame
    full, wrong = _shaft(rng, n, s, t, palette, mode, style, require_mode=strong or demonstrate)
    cap = full[-t:].copy()
    inp[1:n+1, q0:q0+s] = full
    inp[n-t+1:n+1, q0:q0+s] = marker
    # Nearest axial tile is a histogram-preserving impostor.
    inp[near:near+t, q0:q0+s] = wrong
    # A nearer exact key is off axis and must be erased.
    off_q = q0 - s - 1
    inp[near:near+t, off_q:off_q+s] = cap
    if active:
        inp[far:far+t, q0:q0+s] = cap
    # Optional extra rectangle has the wrong perpendicular span or axial length.
    if rng.random() < .5:
        inp[far:far+1, width-1:width] = int(rng.choice(palette))
    out = inp.copy()
    out[near:near+t, q0:q0+s] = background
    out[near:near+t, off_q:off_q+s] = background
    out[far:far+1, width-1:width] = background
    if active:
        out[n-t+1:far+t, q0:q0+s] = marker
        out[n-t+1:n+1, q0:q0+s] = cap
    if rng.integers(2):
        inp, out = np.rot90(inp, 2).copy(), np.rot90(out, 2).copy()
    return inp, out


def _palette(rng):
    roles = list(map(int, rng.permutation(10)))
    background, marker = roles[:2]
    frames = roles[2:4]
    palette = roles[4:4+int(rng.integers(2, 5))]
    return background, marker, frames, palette


def _make_pair(rng, index, test, axis, fixed_colors, style):
    colors = fixed_colors if fixed_colors is not None else _palette(rng)
    if test:
        count = int(rng.integers(2, 4))
    elif index == 0:
        count = 1
    elif index == 1:
        count = 2
    else:
        count = int(rng.integers(1, 3))
    spacing = int(rng.integers(1, 3))
    margin = int(rng.integers(1, 3))
    width_budget = (30 - 2*margin - spacing*(count-1)) // count
    modules = []
    for mi in range(count):
        active = not (not test and index == 1 and mi == 1)
        if (test or index >= 2) and mi > 0 and rng.random() < .25:
            active = False
        mode = 'mirror' if not test and index == 0 else ('half_turn' if not test and index == 1 and mi == 0
                                                       else str(rng.choice(['mirror', 'half_turn'])))
        modules.append(_module(rng, width_budget, colors, active, mode,
                               style if rng.random() < .6 else str(rng.choice(['random', 'bands', 'motif'])),
                               strong=not test and index == 0,
                               demonstrate=not test and index == 1 and mi == 0))
    max_height = max(a.shape[0] for a, _ in modules)
    h = min(30, max_height + 2*margin + int(rng.integers(0, 3)))
    w = sum(a.shape[1] for a, _ in modules) + spacing*(count-1) + 2*margin
    assert h <= 30 and w <= 30
    inp = np.full((h, w), colors[0], dtype=np.int8)
    out = inp.copy()
    c = margin
    for a, b in modules:
        r = int(rng.integers(1, h-a.shape[0]))
        inp[r:r+a.shape[0], c:c+a.shape[1]] = a
        out[r:r+a.shape[0], c:c+a.shape[1]] = b
        c += a.shape[1] + spacing
    turn = (axis + (int(rng.integers(2))*2) + (1 if test else 0)) % 4
    return dict(input=np.rot90(inp, turn).copy(), output=np.rot90(out, turn).copy())


def _count(value, name, low, high):
    if (not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
            or not low <= int(value) <= high):
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must be greater than num_test')
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed) if int(seed) >= 0 else int(seed) % (1 << 64))
    axis = int(rng.integers(4))
    style = str(rng.choice(['random', 'bands', 'motif']))
    colors = _palette(rng) if rng.random() < .35 else None
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            for _ in range(50):
                pair = _make_pair(rng, index, split == 'test', axis, colors, style)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError('No distinct scene in bounded sampling')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, training_modes, exclusive_training_modes = set(), set(), set()
    inactive_training, training_axes, test_axes = False, set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == b.shape
            assert _key(a) not in seen
            seen.add(_key(a))
            record = _analyze(a)
            assert np.array_equal(_render(record), b), 'Incorrect docking output'
            assert record['matches'], 'Every scene demonstrates a successful docking'
            assert record['geometric_impostors'] >= len(record['docks'])
            assert record['off_axis_keys'] >= len(record['docks'])
            assert len(record['tokens']) > len(record['matches'])
            for d in record['docks']:
                assert min(run[2] for run in d['runs']) >= 1
                # Visible shaft, all guards, and the full base are preserved.
                protected = d['cells'] - record['cap_cells']
                assert all(int(a[p]) == int(b[p]) for p in protected)
                if split == 'train':
                    training_modes.update(d['modes'])
                    if len(d['modes']) == 1:
                        exclusive_training_modes.update(d['modes'])
                    training_axes.add(0 if d['out'][0] else 1)
                else:
                    test_axes.add(0 if d['out'][0] else 1)
            active_ids = {m['di'] for m in record['matches']}
            if split == 'train':
                inactive_training |= len(active_ids) < len(record['docks'])
            for di, d in enumerate(record['docks']):
                final = {_cell(d, k, q) for k in range(d['stop'], d['shaft_length']+1)
                         for q in range(d['q0'], d['q0']+d['span'])}
                if di not in active_ids:
                    assert all(b[p] == record['marker'] for p in final)
            for m in record['matches']:
                d = record['docks'][m['di']]
                assert m['distance'] > d['cap_length']
                assert all(b[p] == record['marker'] for p in m['route'] - m['final'])
                # Completed shaft satisfies each supported relation, including the inserted cap.
                shaft = np.array([[b[_cell(d, k, q)] for q in range(d['q0'], d['q0']+d['span'])]
                                  for k in range(1, d['shaft_length']+1)])
                for mode in d['modes']:
                    reflected = shaft[::-1] if mode == 'mirror' else shaft[::-1, ::-1]
                    assert np.array_equal(shaft, reflected)
                # A closer aligned impostor exists with exactly the same histogram.
                witnesses = []
                for ti, cells in enumerate(record['tokens']):
                    local = {_local(p, d) for p in cells}
                    ks, qs = sorted({k for k, _ in local}), sorted({q for _, q in local})
                    if (len(ks) == d['cap_length'] and qs == list(range(d['q0'], d['q0']+d['span']))
                            and d['shaft_length'] < ks[0] < m['start']):
                        tile = np.array([[a[_cell(d, k, q)] for q in qs] for k in ks])
                        if Counter(map(int, tile.ravel())) == Counter(map(int, d['code'].ravel())):
                            witnesses.append(not np.array_equal(tile, d['code']))
                assert any(witnesses), 'Nearest/histogram shortcut lacks a counterexample'
    assert {'mirror', 'half_turn'} <= training_modes
    assert exclusive_training_modes == {'mirror', 'half_turn'}
    assert inactive_training
    assert len(training_axes) == len(test_axes) == 1 and training_axes.isdisjoint(test_axes)


def _puzzle_key(puzzle):
    return tuple(tuple((_key(np.asarray(p['input'])), _key(np.asarray(p['output']))) for p in puzzle[s])
                 for s in ('train', 'test'))


def self_test():
    """Bounded 42-seed variation census, equivariance and corruption fixtures."""
    signatures, shapes, geometries, caps, palettes, codes = set(), set(), set(), set(), set(), set()
    modes, directions, dock_counts, active_counts, distances, normalized = set(), set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_key(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = pair['input']
                record = _analyze(a)
                shapes.add(a.shape)
                palettes.add((record['background'], record['marker']))
                dock_counts.add(len(record['docks']))
                active_counts.add(len(record['matches']))
                distances.update(m['distance'] for m in record['matches'])
                mapping = {}
                norm = bytes(mapping.setdefault(int(c), len(mapping)) for c in a.ravel())
                normalized.add((a.shape, norm))
                for d in record['docks']:
                    geometries.add((d['width'], d['runs'], d['cap_length']))
                    caps.add(d['code'].shape)
                    codes.add(_key(d['code']))
                    modes.update(d['modes'])
                    directions.add(d['out'])
        if seed in (0, 21, 41):
            assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
    assert len(signatures) == 42 and len(normalized) >= 180
    assert len(shapes) >= 35 and len(geometries) >= 100
    assert len(caps) >= 8 and len(codes) >= 120 and len(palettes) >= 25
    assert modes == {'mirror', 'half_turn'} and directions == set(_DIRS)
    assert dock_counts == {1, 2, 3} and active_counts == {1, 2, 3} and len(distances) >= 6
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000 + 10*nt+nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.5, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError('Illegal counts accepted')
    p = generate_puzzle(17, 2, 1)
    # Explicit answer-changing ablations on the one-dock demonstration.
    one = next(_analyze(pair['input']) for pair in p['train']
               if len(_analyze(pair['input'])['docks']) == 1)
    d = one['docks'][0]
    expected = _render(one)
    for omitted in ('A', 'B'):
        candidates = []
        for cells in one['tokens']:
            local = {_local(point, d) for point in cells}
            ks, qs = sorted({k for k, _ in local}), sorted({q for _, q in local})
            if len(ks) != d['cap_length'] or len(qs) != d['span'] or ks[0] <= d['shaft_length']:
                continue
            tile = np.array([[one['input'][_cell(d, k, q)] for q in qs] for k in ks])
            if omitted == 'A' and qs != list(range(d['q0'], d['q0']+d['span'])):
                continue
            if omitted == 'B' and not np.array_equal(tile, d['code']):
                continue
            candidates.append((ks[0], ks, qs, tile))
        _, ks, qs, tile = min(candidates, key=lambda choice: choice[0])
        ablated = one['clean'].copy()
        final = {_cell(d, k, q) for k in range(d['stop'], d['stop']+d['cap_length']) for q in qs}
        route = {_cell(d, k, q) for k in range(d['stop'], ks[-1]+1) for q in qs}
        for point in route - final:
            ablated[point] = one['marker']
        for i in range(len(ks)):
            for j, q in enumerate(qs):
                ablated[_cell(d, d['stop']+i, q)] = tile[i, j]
        assert not np.array_equal(ablated, expected), f'Omitting {omitted} failed to change the answer'
    for pair in p['train'] + p['test']:
        a, b = pair['input'], pair['output']
        for turn in range(4):
            assert np.array_equal(solve(np.rot90(a, turn)), np.rot90(b, turn))
        permutation = np.random.default_rng(321).permutation(10)
        assert np.array_equal(solve(permutation[a]), permutation[b])
    # Mixed orthogonal approaches obey the same rule despite not being sampled
    # by the parallel-lane packing preference. Construct labels before parsing.
    rng = np.random.default_rng(8107)
    colors = _palette(rng)
    mixed_input = np.full((30, 30), colors[0], dtype=np.int8)
    mixed_output = mixed_input.copy()
    for turn, r, c in ((0, 1, 1), (1, 21, 14)):
        aa, bb = _module(rng, 8, colors, True, 'half_turn', 'random', demonstrate=True)
        aa, bb = np.rot90(aa, turn), np.rot90(bb, turn)
        mixed_input[r:r+aa.shape[0], c:c+aa.shape[1]] = aa
        mixed_output[r:r+bb.shape[0], c:c+bb.shape[1]] = bb
    mixed = _analyze(mixed_input)
    assert len(mixed['matches']) == 2
    assert {bool(d['out'][0]) for d in mixed['docks']} == {False, True}
    assert np.array_equal(solve(mixed_input), mixed_output)
    # Break an intact pair of shaft witnesses while preserving geometry/roles.
    damaged = p['train'][0]['input'].copy()
    rec = _analyze(damaged)
    d = next(d for d in rec['docks'] if len(d['modes']) == 1 and d['span'] >= 2)
    point = _cell(d, d['cap_length']+1, d['q0'])
    old = int(damaged[point])
    damaged[point] = next(int(c) for c in np.unique(d['shaft'][:-d['cap_length']]) if int(c) != old)
    try:
        solve(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('Broken visible symmetry witnesses accepted')
    corrupted = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    m = _analyze(corrupted['test'][0]['input'])['matches'][0]
    point = min(m['final'])
    corrupted['test'][0]['output'][point] = (int(corrupted['test'][0]['output'][point]) + 1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError('Wrong cap label accepted')
    # Duplicate a valid cargo on the same ray: there must be no tie-breaking by proximity.
    a = p['train'][0]['input'].copy()
    record = _analyze(a)
    m = record['matches'][0]
    d = record['docks'][m['di']]
    for cells in record['tokens']:
        local = {_local(point, d) for point in cells}
        ks, qs = sorted({k for k, _ in local}), sorted({q for _, q in local})
        if (len(ks) == d['cap_length'] and qs == list(range(d['q0'], d['q0']+d['span']))
                and d['shaft_length'] < ks[0] < m['start']):
            for i, k in enumerate(ks):
                for j, q in enumerate(qs):
                    a[_cell(d, k, q)] = d['code'][i, j]
            break
    try:
        solve(a)
    except AssertionError:
        pass
    else:
        raise AssertionError('Ambiguous pair of exact keys accepted')
    return {'seeds': 42, 'puzzles': len(signatures), 'shapes': len(shapes),
            'color_normalized_inputs': len(normalized), 'dock_geometries': len(geometries),
            'cap_shapes': len(caps), 'distinct_cap_codes': len(codes),
            'background_marker_assignments': len(palettes), 'travel_distances': sorted(map(int, distances))}


if __name__ == '__main__':
    print(self_test())
