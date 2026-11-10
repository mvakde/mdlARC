"""Register loose observations of a native-period overprint on a scaled carrier.

Only Python's standard library and NumPy are used. No files, examples, seeds or
construction coordinates are consulted by solve(). Experiments belong in scratch/.
"""
from __future__ import annotations

import hashlib
import numpy as np


MARKER = 5
SEARCH_LIMIT = 100000
REQUIRED = frozenset((
    'ignore_print', 'tile_rows', 'tile_columns', 'stretch_print',
    'opaque_transparency', 'local_phases', 'input_phases',
    'row_phase', 'column_phase', 'swap_rulers',
))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= x <= 30 for x in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError('expected a 1..30 integer grid with colors 0..9')
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _components(a, bg):
    todo = set(zip(*np.nonzero(a != bg)))
    parts = []
    while todo:
        p = min(todo)
        todo.remove(p)
        stack, points = [p], [p]
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in todo:
                    todo.remove(q)
                    stack.append(q)
                    points.append(q)
        r0, c0 = map(min, zip(*points))
        r1, c1 = map(max, zip(*points))
        part = np.full((r1-r0+1, c1-c0+1), bg, dtype=np.int8)
        for r, c in points:
            part[r-r0, c-c0] = a[r, c]
        parts.append(((int(r0), int(c0)), part))
    return parts


def _frame(carrier, sy, sx, bg):
    h, w = carrier.shape
    a = np.full((sy+h+2, sx+w+2), bg, dtype=np.int8)
    a[sy:, sx] = MARKER
    a[sy:, -1] = MARKER
    a[sy, sx:] = MARKER
    a[-1, sx:] = MARKER
    a[:sy, sx] = MARKER
    a[sy, :sx] = MARKER
    a[sy+1:-1, sx+1:-1] = carrier
    return a


def _decode(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    if bg == MARKER or not (np.all(a[0] == bg) and np.all(a[-1] == bg)
                            and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)):
        raise ValueError('background must occupy the complete outer border')
    parts = _components(a, bg)
    frames = [p for p in parts if np.any(p[1] == MARKER)]
    if len(frames) != 1:
        raise ValueError('need exactly one gray frame')
    frame_origin, f = frames[0]
    marker = f == MARKER
    sy = int(np.argmax(marker.sum(axis=1)))
    sx = int(np.argmax(marker.sum(axis=0)))
    carrier = f[sy+1:-1, sx+1:-1].copy()
    if (not 2 <= sy <= 4 or not 2 <= sx <= 4
            or not all(3 <= n <= 6 for n in carrier.shape)
            or len(np.unique(carrier)) < 2
            or np.any((carrier == bg) | (carrier == MARKER))
            or not np.array_equal(_frame(carrier, sy, sx, bg), f)):
        raise ValueError('invalid carrier, frame or rulers')
    origins, windows = [], []
    for origin, part in parts:
        if origin == frame_origin:
            continue
        if np.any((part == bg) | (part == MARKER)):
            raise ValueError('loose observations must be filled rectangles')
        visible = np.isin(part, np.unique(carrier))
        if (np.count_nonzero(visible) < 3 or len(np.unique(part[visible])) < 2
                or np.all(visible)):
            raise ValueError('each window needs a multicolor key and print ink')
        origins.append(origin)
        windows.append(part)
    if not 2 <= len(windows) <= 4:
        raise ValueError('need two to four observations')
    return bg, carrier, sy, sx, windows, origins


def _base(carrier, sy, sx):
    return np.repeat(np.repeat(carrier, sy, axis=0), sx, axis=1)


def _render(carrier, stencil, sy, sx, bg):
    base = _base(carrier, sy, sx)
    h, w = carrier.shape
    r, c = np.indices(base.shape)
    screen = stencil[r % h, c % w]
    return np.where(screen == bg, base, screen).astype(np.int8)


def _observation(f, phase, h, w, carrier_colors, bg):
    """Record both transparent and opaque residues; -1 means unobserved."""
    obs = np.full(h*w, -1, dtype=np.int8)
    a, b = phase
    for r in range(f.shape[0]):
        for c in range(f.shape[1]):
            j = ((a+r) % h)*w + (b+c) % w
            color = int(f[r, c])
            value = bg if color in carrier_colors else color
            if obs[j] >= 0 and obs[j] != value:
                return None
            obs[j] = value
    return obs


def _domains(carrier, sy, sx, windows, bg):
    h, w = carrier.shape
    base = _base(carrier, sy, sx)
    colors = set(map(int, carrier.flat))
    domains = []
    for f in windows:
        ph, pw = f.shape
        if ph > base.shape[0] or pw > base.shape[1] or (ph >= h and pw >= w):
            raise ValueError('each observation must fit and leave a residue unobserved')
        visible = np.isin(f, list(colors))
        found, examined = {}, set()
        for r in range(base.shape[0]-ph+1):
            for c in range(base.shape[1]-pw+1):
                phase = r % h, c % w
                if phase in examined:
                    continue
                if np.any(base[r:r+ph, c:c+pw][visible] != f[visible]):
                    continue
                examined.add(phase)
                obs = _observation(f, phase, h, w, colors, bg)
                if obs is not None:
                    found[phase] = (obs, (r, c))
        if not found:
            raise ValueError('no feature-supported registration')
        domains.append([(phase, obs, position)
                        for phase, (obs, position) in sorted(found.items())])
    return domains


def _recover(decoded):
    bg, carrier, sy, sx, windows, _ = decoded
    h, w = carrier.shape
    domains = _domains(carrier, sy, sx, windows, bg)
    order = sorted(range(len(windows)), key=lambda i: (len(domains[i]), i))
    solutions, selected = {}, {}
    visits = 0

    def search(depth, observed):
        nonlocal visits
        visits += 1
        if visits > SEARCH_LIMIT:
            raise ValueError('registration search exceeds family bound')
        if len(solutions) > 1:
            return
        if depth == len(order):
            if np.all(observed >= 0):
                solutions.setdefault(observed.tobytes(), (observed.copy(), dict(selected)))
            return
        i = order[depth]
        for candidate in domains[i]:
            _, obs, _ = candidate
            known = obs >= 0
            if np.any((observed[known] >= 0) & (observed[known] != obs[known])):
                continue
            merged = observed.copy()
            merged[known] = obs[known]
            selected[i] = candidate
            search(depth+1, merged)
            del selected[i]
            if len(solutions) > 1:
                return

    search(0, np.full(h*w, -1, dtype=np.int8))
    if len(solutions) != 1:
        raise ValueError('the complete periodic print is absent or ambiguous')
    stencil, selected = next(iter(solutions.values()))
    return stencil.reshape(h, w), domains, selected, visits


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer exclusively from this input's visible observations."""
    decoded = _decode(input_grid)
    bg, carrier, sy, sx, _, _ = decoded
    stencil, _, _, _ = _recover(decoded)
    # Deliberately use a cell loop, separate from the constructor's renderer.
    result = _base(carrier, sy, sx)
    h, w = carrier.shape
    for r in range(result.shape[0]):
        for c in range(result.shape[1]):
            ink = int(stencil[r % h, c % w])
            if ink != bg:
                result[r, c] = ink
    return result


def _naive_screen(decoded, phases):
    bg, carrier, _, _, windows, _ = decoded
    h, w = carrier.shape
    colors = set(map(int, carrier.flat))
    screen = np.full((h, w), bg, dtype=np.int8)
    for f, (a, b) in zip(windows, phases):
        for r in range(f.shape[0]):
            for c in range(f.shape[1]):
                color = int(f[r, c])
                if color not in colors:
                    screen[(a+r) % h, (b+c) % w] = color
    return screen


def _evidence(decoded, stencil):
    bg, carrier, sy, sx, windows, origins = decoded
    h, w = carrier.shape
    base = _base(carrier, sy, sx)
    r, c = np.indices(base.shape)
    tiled = stencil[r % h, c % w]
    correct = np.where(tiled == bg, base, tiled)
    alternatives = {
        'ignore_print': base,
        'tile_rows': np.where(tiled == bg, carrier[r % h, c//sx], tiled),
        'tile_columns': np.where(tiled == bg, carrier[r//sy, c % w], tiled),
        'stretch_print': np.where(_base(stencil, sy, sx) == bg,
                                  base, _base(stencil, sy, sx)),
        'opaque_transparency': tiled,
        'local_phases': _render(carrier, _naive_screen(decoded, [(0, 0)]*len(windows)), sy, sx, bg),
        'input_phases': _render(carrier, _naive_screen(decoded, origins), sy, sx, bg),
        'row_phase': np.where(stencil[(r+1) % h, c % w] == bg,
                              base, stencil[(r+1) % h, c % w]),
        'column_phase': np.where(stencil[r % h, (c+1) % w] == bg,
                                 base, stencil[r % h, (c+1) % w]),
        'swap_rulers': _render(carrier, stencil, sx, sy, bg),
    }
    return {name for name, alt in alternatives.items() if not np.array_equal(correct, alt)}


def _analyze(pair):
    decoded = _decode(pair['input'])
    bg, carrier, sy, sx, windows, _ = decoded
    stencil, domains, selected, visits = _recover(decoded)
    expected = _render(carrier, stencil, sy, sx, bg)
    assert np.array_equal(expected, np.asarray(pair['output'])), 'incorrect overprint'
    observations = np.array([selected[i][1] for i in range(len(windows))])
    assert np.all(np.any(observations >= 0, axis=0)), 'unseen periodic residues'
    opaque = stencil.ravel() != bg
    assert np.any(opaque) and np.any(~opaque), 'need ink and transparency'
    assert all(not np.all(obs[opaque] >= 0) for obs in observations), 'one donor supplies all print'
    shared_ink = np.sum(observations[:, opaque] >= 0, axis=0) >= 2
    shared_paper = np.sum(observations[:, ~opaque] >= 0, axis=0) >= 2
    assert np.any(shared_ink) and np.any(shared_paper), 'need shared ink and transparent evidence'
    return dict(decoded=decoded, stencil=stencil, domains=domains, selected=selected,
                visits=visits, evidence=_evidence(decoded, stencil))


def _pack(objects, bg, rng):
    maxh = max(a.shape[0] for a in objects)+2
    maxw = max(a.shape[1] for a in objects)+2
    canvases = [(int(rng.integers(max(14, maxh), 31)),
                 int(rng.integers(max(14, maxw), 31))) for _ in range(12)]
    canvases += [(30, 30)]*4
    for height, width in canvases:
        occupied = np.zeros((height, width), dtype=bool)
        occupied[0] = occupied[-1] = True
        occupied[:, 0] = occupied[:, -1] = True
        grid = np.full((height, width), bg, dtype=np.int8)
        order = sorted(range(len(objects)),
                       key=lambda i: -(objects[i].size + float(rng.random())*15))
        for i in order:
            a = objects[i]
            ph, pw = a.shape
            view = np.lib.stride_tricks.sliding_window_view(occupied, (ph, pw))
            free = np.argwhere(~view.any(axis=(-2, -1)))
            if not len(free):
                break
            r, c = map(int, free[int(rng.integers(len(free)))])
            grid[r:r+ph, c:c+pw] = a
            occupied[max(0, r-1):min(height, r+ph+1),
                     max(0, c-1):min(width, c+pw+1)] = True
        else:
            return grid
    raise ValueError('could not pack separated observations')


def _carrier(rng, h, w, colors):
    k = len(colors)
    r, c = np.indices((h, w))
    kind = int(rng.integers(5))
    if kind == 0:
        labels = rng.integers(k, size=(h, w))
    elif kind == 1:
        rows = sorted(rng.choice(np.arange(1, h), int(rng.integers(1, h)), replace=False))
        cols = sorted(rng.choice(np.arange(1, w), int(rng.integers(1, w)), replace=False))
        blocks = rng.integers(k, size=(len(rows)+1, len(cols)+1))
        labels = blocks[np.searchsorted(rows, r, side='right'), np.searchsorted(cols, c, side='right')]
    elif kind == 2:
        line = rng.integers(k, size=w)
        shifts = rng.integers(w, size=h)
        labels = line[(c+shifts[r]) % w]
    elif kind == 3:
        labels = ((r//int(rng.integers(1, h)) + c//int(rng.integers(1, w))) % k)
        mask = rng.random((h, w)) < .18
        labels[mask] = rng.integers(k, size=int(mask.sum()))
    else:
        labels = np.zeros((h, w), dtype=int)
        for ink in range(1, k):
            a, b = int(rng.integers(h)), int(rng.integers(w))
            labels[abs(r-a)+abs(c-b) <= int(rng.integers(1, max(h, w)))] = ink
    # A small independent patch prevents an invisible declared carrier ink.
    locations = rng.choice(h*w, k, replace=False)
    labels.ravel()[locations] = rng.permutation(k)
    return np.asarray(colors, dtype=np.int8)[labels]


def _stencil(rng, h, w, inks, bg):
    r, c = np.indices((h, w))
    kind = int(rng.integers(6))
    density = float(rng.uniform(.18, .55))
    if kind == 0:
        mask = rng.random((h, w)) < density
    elif kind == 1:
        mask = (r+c+int(rng.integers(2))) % 2 == 0
        mask &= rng.random((h, w)) > .15
    elif kind == 2:
        mask = (r == int(rng.integers(h))) | (c == int(rng.integers(w)))
    elif kind == 3:
        mask = np.minimum.reduce((r, c, h-1-r, w-1-c)) == 0
        mask &= rng.random((h, w)) > .25
    elif kind == 4:
        mask = np.zeros((h, w), dtype=bool)
        points = {(int(rng.integers(h)), int(rng.integers(w)))}
        while len(points) < max(2, round(density*h*w)):
            frontier = sorted({((a+dr) % h, (b+dc) % w)
                               for a, b in points
                               for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))}-points)
            points.add(frontier[int(rng.integers(len(frontier)))])
        for p in points:
            mask[p] = True
    else:
        mask = ((r+int(rng.integers(h))) % int(rng.integers(2, h+1)) == 0)
        mask |= rng.random((h, w)) < .12
    stencil = np.full((h, w), bg, dtype=np.int8)
    stencil[mask] = rng.choice(inks, size=int(mask.sum()))
    return stencil


def _cut_windows(rng, out, h, w, axis, count):
    # Two cyclic bands jointly observe the entire torus, each omitting a band.
    n = w if axis == 0 else h
    cut = int(rng.integers(1, n-1))
    overlap = int(rng.integers(1, n-cut))
    start = int(rng.integers(n))
    intervals = ((start, cut+overlap), (start+cut, n-cut))
    windows = []
    for start, length in intervals:
        if axis == 0:
            ph = int(rng.integers(h+1, min(out.shape[0], h+3)+1))
            pw = length
            choices = [c for c in range(out.shape[1]-pw+1) if c % w == start % w]
            if not choices:
                return None
            a = int(rng.integers(out.shape[0]-ph+1))
            b = int(rng.choice(choices))
        else:
            ph = length
            pw = int(rng.integers(w+1, min(out.shape[1], w+3)+1))
            choices = [r for r in range(out.shape[0]-ph+1) if r % h == start % h]
            if not choices:
                return None
            a = int(rng.choice(choices))
            b = int(rng.integers(out.shape[1]-pw+1))
        windows.append(out[a:a+ph, b:b+pw].copy())
    for _ in range(count-2):
        if rng.random() < .3:
            ph, pw = int(rng.integers(2, h)), int(rng.integers(2, w))
        elif axis == 0:
            ph = int(rng.integers(1, h))
            pw = int(rng.integers(w+1, min(out.shape[1], w+3)+1))
        else:
            pw = int(rng.integers(1, w))
            ph = int(rng.integers(h+1, min(out.shape[0], h+3)+1))
        a, b = int(rng.integers(out.shape[0]-ph+1)), int(rng.integers(out.shape[1]-pw+1))
        windows.append(out[a:a+ph, b:b+pw].copy())
    return windows


def _make_pair(rng, bg, pool, axis, count):
    for _ in range(240):
        h, w = int(rng.integers(3, 7)), int(rng.integers(3, 7))
        sy, sx = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        shuffled = rng.permutation(pool)
        k = int(rng.integers(2, 5))
        colors = shuffled[:k]
        inks = shuffled[k:k+int(rng.integers(1, min(3, len(pool)-k)+1))]
        carrier = _carrier(rng, h, w, colors)
        stencil = _stencil(rng, h, w, inks, bg)
        out = _render(carrier, stencil, sy, sx, bg)
        windows = _cut_windows(rng, out, h, w, axis, count)
        if windows is None:
            continue
        if any(np.count_nonzero(np.isin(f, colors)) < 3
               or len(np.unique(f[np.isin(f, colors)])) < 2
               or np.all(np.isin(f, colors)) for f in windows):
            continue
        try:
            # Pixel reconstruction is checked before and after random packing.
            decoded = bg, carrier, sy, sx, windows, [(0, 0)]*count
            recovered, _, _, _ = _recover(decoded)
            if not np.array_equal(recovered, stencil):
                continue
            inp = _pack([_frame(carrier, sy, sx, bg)]+windows, bg, rng)
            pair = {'input': inp, 'output': out}
            _analyze(pair)
            return pair
        except (ValueError, AssertionError):
            continue
    raise RuntimeError('bounded observation sampler exhausted')


def _integer(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _integer(num_train, 2, 10, 'num_train')
    nq = _integer(num_test, 1, 4, 'num_test')
    if nt <= nq:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    bg = int(rng.choice([c for c in range(10) if c != MARKER]))
    pool = np.array([c for c in range(10) if c not in (bg, MARKER)])
    ceiling = int(rng.integers(2, 4))
    for _ in range(48):
        first = _make_pair(rng, bg, pool, 0, ceiling)
        second = _make_pair(rng, bg, pool, 1, int(rng.integers(2, ceiling+1)))
        infos = [_analyze(first), _analyze(second)]
        if (REQUIRED <= infos[0]['evidence'] | infos[1]['evidence']
                and infos[0]['decoded'][1].shape != infos[1]['decoded'][1].shape
                and first['output'].shape != second['output'].shape):
            break
    else:
        raise RuntimeError('bounded teaching-pair selection exhausted')
    puzzle = {'train': [first, second], 'test': []}
    seen = {_key(p['input']) for p in puzzle['train']}
    for split, size in (('train', nt), ('test', nq)):
        while len(puzzle[split]) < size:
            for _ in range(24):
                count = int(rng.integers(2, ceiling+1)) if split == 'train' else ceiling+1
                pair = _make_pair(rng, bg, pool, int(rng.integers(2)), count)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('could not obtain distinct inputs')
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, backgrounds, evidence, train_shapes, train_out_shapes = set(), set(), set(), set(), set()
    counts = {'train': [], 'test': []}
    training_directions = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            info = _analyze(pair)
            bg, carrier, sy, sx, windows, _ = info['decoded']
            backgrounds.add(bg)
            counts[split].append(len(windows))
            assert np.array_equal(solve(inp), out)
            if split == 'train':
                evidence |= info['evidence']
                train_shapes.add(carrier.shape)
                train_out_shapes.add(out.shape)
                h, w = carrier.shape
                training_directions |= {0 for f in windows if f.shape[0] > h and f.shape[1] < w}
                training_directions |= {1 for f in windows if f.shape[0] < h and f.shape[1] > w}
    assert len(backgrounds) == 1, 'background changes inside puzzle'
    assert REQUIRED <= evidence, 'missing teaching evidence: ' + str(REQUIRED-evidence)
    assert len(train_shapes) >= 2 and len(train_out_shapes) >= 2
    assert training_directions == {0, 1}, 'demonstrate periodic continuation on both axes'
    assert min(counts['test']) > max(counts['train']), 'missing fragment-count shift'


def _normalized(a):
    mapping = {}
    return a.shape, bytes(mapping.setdefault(int(x), len(mapping)) for x in a.flat)


def self_test():
    """Stream a bounded 42-seed cohort; retain signatures, never grid corpora."""
    structures, shapes, carriers, masks, backgrounds, scales, phases, counts = (set() for _ in range(8))
    signatures, ambiguous, max_visits = set(), 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                digest.update(pair['input'].tobytes())
                digest.update(pair['output'].tobytes())
                info = _analyze(pair)
                bg, carrier, sy, sx, windows, _ = info['decoded']
                stencil = info['stencil']
                structures.add(_normalized(pair['output']))
                shapes.add(pair['output'].shape)
                carriers.add(_normalized(carrier))
                masks.add((stencil.shape, (stencil != bg).tobytes()))
                backgrounds.add(bg)
                scales.add((sy, sx))
                counts.add(len(windows))
                phases.update(candidate[0] for candidate in info['selected'].values())
                ambiguous += any(len(d) > 1 for d in info['domains'])
                max_visits = max(max_visits, info['visits'])
        signatures.add(digest.digest())
    assert len(signatures) == 42
    assert len(structures) >= 150 and len(shapes) >= 45
    assert len(carriers) >= 100 and len(masks) >= 100
    assert len(backgrounds) >= 8 and len(scales) == 9
    assert counts == {2, 3, 4} and len(phases) >= 20
    assert ambiguous > 0 and max_visits < SEARCH_LIMIT
    p = generate_puzzle(0, 2, 1)
    q = generate_puzzle(0, 2, 1)
    assert all(_key(a[k]) == _key(b[k]) for split in p
               for a, b in zip(p[split], q[split]) for k in ('input', 'output'))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            generated = generate_puzzle(7000+10*nt+nq, nt, nq)
            assert len(generated['train']) == nt and len(generated['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (2, 2), (3, 4), (4, 0), (6, 5),
                   (True, 1), (4, False), (3.2, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    pair = p['test'][0]
    # Whole-example transpose swaps the two visible ruler directions.
    assert np.array_equal(solve(pair['input'].T), pair['output'].T)
    permutation = np.array([9, 3, 7, 1, 0, 5, 8, 4, 6, 2], dtype=np.int8)
    assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
    bg = int(pair['input'][0, 0])
    objects = [part for _, part in _components(pair['input'], bg)]
    repacked = _pack(objects, bg, np.random.default_rng(81))
    assert np.array_equal(solve(repacked), pair['output'])
    damaged = {split: [{k: v.copy() for k, v in a.items()} for a in p[split]] for split in p}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds, all legal counts, invariants and mutations)')
