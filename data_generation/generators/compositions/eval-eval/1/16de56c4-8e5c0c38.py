"""Clocked symmetry sieve: an edge cadence admits candidate reflection axes.

Two equal ticks specify an arithmetic lattice of axes; a terminal bounds it
and paints the retained subsets. Labels are constructed from intact cores.
The public solver recovers everything from input cells, using convolution.
"""

import numpy as np


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _key(grid):
    a = np.asarray(grid)
    return a.shape, a.astype(np.uint8).tobytes()


def _background(grid):
    counts = np.bincount(grid.ravel().astype(int), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] * 2 <= grid.size:
        raise ValueError("a strict majority background is required")
    return bg


def _clock_top(grid, bg):
    """Parse a boundary clock. Marker colors occur nowhere else in the grid."""
    row = grid[0]
    hues = [int(c) for c in np.unique(row) if c != bg]
    if not 1 <= len(hues) <= 2:
        return None
    candidates = []
    for ref in hues:
        positions = np.flatnonzero(row == ref).tolist()
        q = len(positions) // 2
        if q not in (1, 2) or len(positions) != 2 * q:
            continue
        if np.count_nonzero(grid == ref) != 2 * q:
            continue
        a, b = int(positions[0]), int(positions[q])
        if positions != list(range(a, a + q)) + list(range(b, b + q)):
            continue
        p = b - a
        if p < q or (q == 2 and p < 3):
            continue
        extras = [c for c in hues if c != ref]
        terminal = paint = None
        if extras:
            paint = extras[0]
            ts = np.flatnonzero(row == paint).tolist()
            if len(ts) != q or np.count_nonzero(grid == paint) != q:
                continue
            terminal = int(ts[0])
            if ts != list(range(terminal, terminal + q)):
                continue
            if (terminal - a) % p or a <= terminal <= b:
                continue
        sites = tuple(range(a % p, grid.shape[1] - q + 1, p))
        if terminal is None:
            admitted = sites
        else:
            lo, hi = min(a, terminal), max(b, terminal)
            admitted = tuple(x for x in sites if lo <= x <= hi)
        body_colors = tuple(int(c) for c in np.unique(grid) if c not in (bg, ref, paint))
        if not body_colors or len(sites) < 3 or len(admitted) < 2:
            continue
        candidates.append(dict(q=q, a=a, b=b, period=p, terminal=terminal,
                               paint=paint, reference=ref, sites=sites,
                               axes=tuple(2 * x + q - 1 for x in admitted),
                               full_axes=tuple(2 * x + q - 1 for x in sites),
                               colors=body_colors, background=bg))
    return candidates[0] if len(candidates) == 1 else None


def _parse(grid):
    grid = np.asarray(grid)
    if grid.ndim != 2 or not all(1 <= d <= 30 for d in grid.shape):
        raise ValueError("grid dimensions must be 1..30")
    if not np.issubdtype(grid.dtype, np.integer) or np.any((grid < 0) | (grid > 9)):
        raise ValueError("grid must contain integer colors 0..9")
    bg = _background(grid)
    choices = []
    for turn in range(4):
        work = np.rot90(grid, -turn)
        clock = _clock_top(work, bg)
        if clock is not None:
            choices.append((turn, work, clock))
    if len(choices) != 1:
        raise ValueError("exactly one boundary must contain a complete clock")
    return choices[0]


def _scores(mask):
    return sum((np.convolve(row.astype(int), row.astype(int)) for row in mask),
               np.zeros(2 * mask.shape[1] - 1, dtype=int))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Retain the unique best supported subset at an admitted axis, per hue."""
    turn, work, clock = _parse(input_grid)
    out = np.full_like(work, clock['background'])
    out[0] = work[0]
    cols = np.arange(work.shape[1])
    for color in clock['colors']:
        mask = work == color
        scores = _scores(mask)
        maximum = max(int(scores[x]) for x in clock['axes'])
        winners = [x for x in clock['axes'] if scores[x] == maximum]
        if len(winners) != 1 or maximum == 0:
            raise ValueError("figure needs one uniquely best admitted axis")
        reflected = winners[0] - cols
        valid = (reflected >= 0) & (reflected < work.shape[1])
        keep = np.zeros_like(mask)
        keep[:, valid] = mask[:, valid] & mask[:, reflected[valid]]
        out[keep] = color if clock['paint'] is None else clock['paint']
    return np.rot90(out, turn).copy()


def _support(cells, axis):
    return {p for p in cells if (p[0], axis - p[1]) in cells}


def _core_is_best(cells, core, axis, axes):
    return (_support(cells, axis) == core and
            all(len(_support(cells, x)) < len(core) for x in axes if x != axis))


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _tick_geometry(rng, width, q, p, bounded, boundary_witness):
    choices = []
    for a in range(width - q - p + 1):
        b = a + p
        sites = tuple(range(a % p, width - q + 1, p))
        usable = [x for x in sites if 2 <= x <= width - q - 3]
        if len(sites) < 4 or len(usable) < 2:
            continue
        if not bounded:
            if boundary_witness and not (any(x < a for x in usable) and any(x > b for x in usable)):
                continue
            choices.append((a, b, None, sites))
        else:
            for t in sites:
                if a <= t <= b:
                    continue
                lo, hi = min(a, t), max(b, t)
                admitted = [x for x in usable if lo <= x <= hi]
                outside = [x for x in sites if not lo <= x <= hi and 1 <= x <= width - q - 1]
                if len(admitted) >= 2 and outside:
                    choices.append((a, b, t, sites))
    return _pick(rng, choices) if choices else None


def _make_core(rng, shape, axis, occupied):
    h, w = shape
    q = 1 if axis % 2 == 0 else 2
    x = axis // 2
    radius = min(3, x + (q == 2), w - 1 - x)
    if radius < 1:
        return None
    height = int(rng.integers(3, 7))
    top = int(rng.integers(1, h - height + 1))
    style = int(rng.integers(4))
    cells = set()
    for r in range(top, top + height):
        if style == 0:
            ds = [int(rng.integers(1, radius + 1))]
        elif style == 1:
            level = 1 + min(r - top, top + height - 1 - r) % radius
            ds = list(range(1, level + 1))
        elif style == 2:
            ds = [d for d in range(1, radius + 1) if rng.random() < .55]
        else:
            ds = [radius] if r in (top, top + height - 1) else [1]
        for d in ds:
            cells.add((r, x - d + (q == 2)))
            cells.add((r, x + d))
        if q == 1 and rng.random() < .45:
            cells.add((r, x))
    if not 6 <= len(cells) <= 18 or cells & occupied:
        return None
    return cells


def _decoy(rng, shape, core, occupied, axis, axes, full_axes, kind):
    """A stronger symmetric island that cannot supply an admitted axis.

    Two columns on rows disjoint from the core give strong rejected symmetry,
    while all accepted-axis scores are explicitly checked after construction.
    """
    h, w = shape
    core_rows = {r for r, _ in core}
    candidates = []
    lo, hi = min(axes), max(axes)
    for left in range(w - 1):
        for right in range(left + 1, w):
            center = left + right
            if center in axes:
                continue
            if kind == 'outside' and center not in set(full_axes) - set(axes):
                continue
            if kind == 'phase' and (center % 2 != axis % 2 or not lo <= center <= hi):
                continue
            rows = [r for r in range(1, h) if r not in core_rows and
                    (r, left) not in occupied and (r, right) not in occupied]
            n = len(core) // 2 + 1
            if len(rows) >= n:
                candidates.append((left, right, rows, n))
    for j in rng.permutation(len(candidates))[:80]:
        left, right, rows, n = candidates[int(j)]
        chosen = rng.choice(rows, n, replace=False)
        decoy = {(int(r), c) for r in chosen for c in (left, right)}
        if _core_is_best(core | decoy, core, axis, axes):
            return decoy
    return None


def _noise(rng, shape, cells, core, occupied, axis, axes, amount):
    h, w = shape
    result = set(cells)
    for _ in range(amount):
        near = sorted({(r + dr, c + dc) for r, c in result
                       for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                       if 1 <= r + dr < h and 0 <= c + dc < w} - result - occupied)
        if near and rng.random() < .65:
            candidates = [near[int(j)] for j in rng.permutation(len(near))[:30]]
        else:
            candidates = [(int(rng.integers(1, h)), int(rng.integers(w))) for _ in range(30)]
        for p in candidates:
            if p in occupied or p in result:
                continue
            if _core_is_best(result | {p}, core, axis, axes):
                result.add(p)
                break
    return result


def _build_pair(rng, q, limit, is_test, witness, turn, rescue=False):
    p = int(rng.integers(limit + 1, min(7, limit + 2) + 1)) if is_test else int(rng.integers(3 if q == 2 else 2, limit + 1))
    if witness is None and not is_test and q == 1 and rng.random() < .2:
        p = 1
    bounded = witness == 'bounded' or (witness is None and rng.random() < .5)
    count = 2 if witness else int(rng.integers(1, 5))
    width = int(rng.integers(max(12, 4 * p + q), 31))
    height = int(rng.integers(12 if witness else 10 if is_test else 8, 31))
    if rescue:
        width = height = 30
    geom = _tick_geometry(rng, width, q, p, bounded, witness == 'boundary')
    if geom is None:
        return None
    a, b, t, sites = geom
    admitted = sites if t is None else tuple(x for x in sites if min(a, t) <= x <= max(b, t))
    axes = tuple(2 * x + q - 1 for x in admitted)
    full_axes = tuple(2 * x + q - 1 for x in sites)
    usable = [axis for axis in axes if 2 <= axis // 2 <= width - q - 3]
    colors = rng.choice(10, count + 3, replace=False).tolist()
    bg, ref, paint = colors[:3]
    inp = np.full((height, width), bg, dtype=np.int8)
    out = inp.copy()
    for left in (a, b):
        inp[0, left:left + q] = ref
    if t is not None:
        inp[0, t:t + q] = paint
    out[0] = inp[0]
    occupied = set()
    for index, color in enumerate(colors[3:]):
        placed = False
        for attempt in range(100):
            axis = int(_pick(rng, usable))
            if witness == 'boundary':
                options = [x for x in usable if (x < 2 * a + q - 1 if index == 0 else x > 2 * b + q - 1)]
                axis = int(_pick(rng, options))
            core = _make_core(rng, inp.shape, axis, occupied)
            if core is None:
                continue
            kind = None
            if witness == 'bounded':
                kind = 'outside' if index == 0 else 'phase'
            elif witness == 'boundary' and index == 0:
                kind = 'phase'
            elif witness is None and rng.random() < .55:
                kind = 'outside' if bounded and rng.random() < .4 else 'phase'
            cells = set(core)
            if kind:
                decoy = _decoy(rng, inp.shape, core, occupied, axis, axes, full_axes, kind)
                if decoy is None:
                    continue
                cells |= decoy
            amount = int(rng.integers(1, 6)) if witness or rng.random() < .8 else 0
            cells = _noise(rng, inp.shape, cells, core, occupied, axis, axes, amount)
            if len(occupied | cells) + np.count_nonzero(inp[0] != bg) >= inp.size * .48:
                continue
            if witness and cells == core:
                continue
            for r, c in cells:
                inp[r, c] = color
            for r, c in core:
                out[r, c] = color if t is None else paint
            occupied |= cells
            placed = True
            break
        if not placed:
            return None
    return {'input': np.rot90(inp, turn).copy(), 'output': np.rot90(out, turn).copy()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate distinct examples; tests use a longer cadence than all training."""
    num_train = _count('num_train', num_train, 2, 10)
    num_test = _count('num_test', num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    rng = np.random.default_rng(seed)
    q = int(rng.integers(1, 3))
    limit = int(rng.integers(3, 6))
    mixed_edges = bool(rng.random() < .5)
    fixed_turn = int(rng.integers(4))
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            witness = ('boundary', 'bounded')[index] if split == 'train' and index < 2 else None
            turn = int(rng.integers(4)) if mixed_edges else fixed_turn
            for attempt in range(80):
                pair = _build_pair(rng, q, limit, split == 'test', witness, turn, rescue=attempt >= 60)
                if pair is not None and _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('bounded geometric construction exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    turn, work, clock = _parse(pair['input'])
    out = np.rot90(np.asarray(pair['output']), -turn)
    bg = clock['background']
    expected = np.full_like(work, bg)
    expected[0] = work[0]
    metrics = dict(clock)
    metrics.update(turn=turn, shape=work.shape, figures=[], deleted=0,
                   off_cadence=False, beyond_terminal=False, left=False, right=False)
    for color in clock['colors']:
        cells = set(map(tuple, np.argwhere(work == color).tolist()))
        subsets = {axis: _support(cells, axis) for axis in clock['axes']}
        best = max(map(len, subsets.values()))
        winners = [axis for axis, s in subsets.items() if len(s) == best]
        assert len(winners) == 1 and best > 0, 'tied or empty admitted support'
        axis = winners[0]
        core = subsets[axis]
        assert any(2 * c != axis for _, c in core), 'axis needs a genuine mirrored pair'
        for r, c in core:
            expected[r, c] = color if clock['paint'] is None else clock['paint']
        deleted = len(cells - core)
        metrics['deleted'] += deleted
        # Ablations distinguish phase restriction from terminal restriction.
        same_parity_in_span = range(min(clock['axes']), max(clock['axes']) + 1, 2)
        phase_max = max(len(_support(cells, x)) for x in same_parity_in_span)
        full_max = max(len(_support(cells, x)) for x in clock['full_axes'])
        metrics['off_cadence'] |= phase_max > best
        metrics['beyond_terminal'] |= full_max > best
        metrics['left'] |= axis < 2 * clock['a'] + clock['q'] - 1
        metrics['right'] |= axis > 2 * clock['b'] + clock['q'] - 1
        rows, cols = zip(*core)
        metrics['figures'].append(dict(color=color, axis=axis, cells=best, deleted=deleted,
                                      extent=(max(rows) - min(rows) + 1, max(cols) - min(cols) + 1),
                                      axis_cells=sum(2 * c == axis for _, c in core),
                                      row_patterns=len({tuple(sorted(c for rr, c in core if rr == r)) for r in set(rows)})))
    assert np.array_equal(out, expected), 'incorrect retained subset or terminal painting'
    assert np.array_equal(solve(np.asarray(pair['input'])), pair['output'])
    assert np.array_equal(out[0], work[0]), 'clock must be preserved'
    assert np.all((work != bg) | (out == bg)), 'no foreground may be added'
    return metrics


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen = set()
    records = {'train': [], 'test': []}
    for split in records:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                a = np.asarray(grid)
                assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
                assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
            assert np.asarray(pair['input']).shape == np.asarray(pair['output']).shape
            key = _key(pair['input'])
            assert key not in seen, 'duplicate input'
            seen.add(key)
            records[split].append(_analyze(pair))
    training = records['train']
    assert any(a['terminal'] is None and a['left'] and a['right'] for a in training), 'teach both boundary directions'
    assert any(a['terminal'] is not None and a['beyond_terminal'] for a in training), 'teach terminal stopping'
    assert any(a['off_cadence'] for a in training), 'teach cadence-constrained symmetry'
    assert all(all(f['deleted'] for f in a['figures']) for a in training[:2]), 'teach deletion, not reflection filling'
    assert len({a['q'] for split in records.values() for a in split}) == 1
    assert min(a['period'] for a in records['test']) > max(a['period'] for a in training), 'longer-cadence test shift'


def _puzzle_key(puzzle):
    return tuple((_key(p['input']), _key(p['output'])) for s in ('train', 'test') for p in puzzle[s])


def self_test() -> None:
    """Bounded geometric coverage, equivariance, ablations and mutation checks."""
    records, fingerprints = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_puzzle_key(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                records.append(_analyze(pair))
                # The same input-only rule works after every rotation and hue permutation.
                if seed < 3:
                    for turn in range(4):
                        inp = 9 - np.rot90(pair['input'], turn)
                        expected = 9 - np.rot90(pair['output'], turn)
                        assert np.array_equal(solve(inp), expected)
        if seed < 2:
            assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
    assert len(fingerprints) == 42
    assert {a['q'] for a in records} == {1, 2}
    assert {a['turn'] for a in records} == {0, 1, 2, 3}
    assert len({a['shape'] for a in records}) >= 35
    assert {a['background'] for a in records} == set(range(10))
    assert {len(a['figures']) for a in records} == {1, 2, 3, 4}
    assert {1, 2, 3, 4, 5, 6, 7} <= {a['period'] for a in records}
    assert any(a['terminal'] is not None and a['terminal'] < a['a'] for a in records)
    assert any(a['terminal'] is not None and a['terminal'] > a['b'] for a in records)
    figures = [f for a in records for f in a['figures']]
    assert len({f['extent'] for f in figures}) >= 15
    assert any(f['axis_cells'] for f in figures) and any(f['row_patterns'] >= 3 for f in figures)
    assert any(f['deleted'] == 0 for f in figures)
    assert sum(a['off_cadence'] for a in records) >= 42
    assert sum(a['beyond_terminal'] for a in records) >= 42
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            puzzle = generate_puzzle(1000 + train * 10 + test, train, test)
            assert len(puzzle['train']) == train and len(puzzle['test']) == test
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4., 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = generate_puzzle(123, 2, 1)
    pair = bad['train'][0]
    r, c = np.argwhere(pair['output'] != pair['input'])[0]
    pair['output'][r, c] = pair['input'][r, c]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted an unsupported cell')
