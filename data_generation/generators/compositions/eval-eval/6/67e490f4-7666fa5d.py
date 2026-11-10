"""Diagonal juries: finite diagonal brackets enfranchise whole shape objects.

Standard library and NumPy only. No files are read or written by this module.
Generation labels use planned ballots; solve reconstructs visible ballots.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from hashlib import sha256
import numpy as np


def _norm(cells):
    cells = tuple((int(r), int(c)) for r, c in cells)
    y, x = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-y, c-x) for r, c in cells))


def _rot(cells):
    return _norm((c, -r) for r, c in cells)


def _mirror(cells):
    return _norm((r, -c) for r, c in cells)


@lru_cache(maxsize=1024)
def _variants(cells, reflection=True):
    found = set()
    for start in (cells, _mirror(cells)) if reflection else (cells,):
        cur = _norm(start)
        for _ in range(4):
            found.add(cur)
            cur = _rot(cur)
    return tuple(sorted(found))


def _canon(cells):
    return min(_variants(_norm(cells)))


def _extent(cells):
    return max(r for r, c in cells)+1, max(c for r, c in cells)+1


@lru_cache(maxsize=1)
def _pool():
    level = {((0, 0),)}
    found = set()
    for size in range(2, 7):
        grown = set()
        for shape in sorted(level):
            near = {(r+dr, c+dc) for r, c in shape
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))}
            for point in sorted(near-set(shape)):
                grown.add(_canon((*shape, point)))
        level = grown
        if size >= 3:
            found.update(s for s in level if max(_extent(s)) <= 4)
    return tuple(sorted(found))


def _components(mask):
    pending = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while pending:
        start = min(pending)
        pending.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for point in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if point in pending:
                    pending.remove(point)
                    stack.append(point)
        result.append(tuple(sorted(cells)))
    return result


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'grid dimensions'
    assert np.issubdtype(a.dtype, np.integer), 'integer colors required'
    assert np.all((a >= 0) & (a <= 9)), 'color range'
    return a


def _runs(grid, color, slope):
    pixels = grid == color
    # A polyomino can accidentally partition into several diagonal runs
    # (notably an S tetromino). Its orthogonal joins disqualify a rail role.
    if np.any(pixels[1:] & pixels[:-1]) or np.any(pixels[:, 1:] & pixels[:, :-1]):
        return None
    lines = defaultdict(list)
    for r, c in np.argwhere(grid == color):
        lines[int(c-slope*r)].append(int(r))
    result = []
    for q, rows in sorted(lines.items()):
        start = last = rows[0]
        for r in rows[1:] + [None]:
            if r is not None and r == last+1:
                last = r
                continue
            if last-start+1 < 2:
                return None
            result.append((q, 2*start+slope*q, 2*last+slope*q))
            start = last = r
    return result


def _parse(grid):
    a = _grid(grid)
    values, counts = np.unique(a, return_counts=True)
    bg = int(values[np.argmax(counts)])
    assert counts.max() > a.size/2, 'unique majority background required'
    candidates = []
    for color in values:
        if int(color) == bg:
            continue
        for slope in (-1, 1):
            runs = _runs(a, int(color), slope)
            if runs is not None and len(runs) >= 2:
                candidates.append((int(color), slope, runs))
    assert len(candidates) == 1, 'unique diagonal rail color and slope required'
    rail, slope, runs = candidates[0]
    objects = []
    for color in values:
        if int(color) in (bg, rail):
            continue
        for cells in _components(a == color):
            assert len(cells) >= 2, 'objects need orthogonal adjacency'
            objects.append(dict(color=int(color), cells=cells, shape=_canon(cells)))
    assert objects, 'objects required'
    return dict(background=bg, rail=rail, slope=slope, runs=runs, objects=objects)


def _field(shape, slope, runs, alternative=''):
    rr, cc = np.indices(shape)
    q, t = cc-slope*rr, rr+slope*cc
    mask = np.zeros(shape, dtype=bool)
    ordered = sorted(runs)
    if alternative == 'infinite':
        return (q > min(v[0] for v in runs)) & (q < max(v[0] for v in runs))
    for level in range(int(t.min()), int(t.max())+1):
        active = [v for v in runs if v[1] <= level <= v[2]]
        if alternative == 'pixel_centers':
            active = [v for v in active if (level-v[0]) % 2 == 0]
        if alternative == 'consecutive':
            for a, b in zip(ordered, ordered[1:]):
                if a in active and b in active:
                    mask |= (t == level) & (q > a[0]) & (q < b[0])
        elif len(active) >= 2:
            mask |= ((t == level) & (q > min(v[0] for v in active))
                     & (q < max(v[0] for v in active)))
    return mask


def _pairwise_field(shape, slope, runs):
    """Validation formulation: union all pairwise overlap parallelograms."""
    rr, cc = np.indices(shape)
    q, t = cc-slope*rr, rr+slope*cc
    mask = np.zeros(shape, dtype=bool)
    for i, a in enumerate(runs):
        for b in runs[i+1:]:
            mask |= ((q > min(a[0], b[0])) & (q < max(a[0], b[0]))
                     & (t >= max(a[1], b[1])) & (t <= min(a[2], b[2])))
    return mask


def _predict(grid, alternative='', parsed=None):
    a = np.asarray(grid)
    info = _parse(a) if parsed is None else parsed
    mask = _field(a.shape, info['slope'], info['runs'], alternative)
    ballots = defaultdict(Counter)
    keys, eligible = [], []
    for obj in info['objects']:
        raw = _norm(obj['cells'])
        key = obj['shape']
        if alternative == 'area':
            key = len(raw)
        elif alternative == 'bbox':
            key = tuple(sorted(_extent(raw)))
        elif alternative == 'translation':
            key = raw
        elif alternative == 'rotation':
            key = min(_variants(raw, False))
        inside = sum(bool(mask[r, c]) for r, c in obj['cells'])
        admitted = inside == len(raw)
        if alternative == 'all':
            admitted = True
        elif alternative == 'any':
            admitted = inside > 0
        elif alternative == 'majority':
            admitted = inside*2 > len(raw)
        if admitted:
            ballots[key][obj['color']] += 1
        keys.append(key)
        eligible.append(admitted)
    winners = {}
    for key in keys:
        counts = ballots[key]
        assert counts, 'unsupported shape class'
        high = max(counts.values())
        best = [color for color, n in counts.items() if n == high]
        assert len(best) == 1, 'tied eligible ballot'
        winners[key] = best[0]
    out = a.copy()
    for obj, key in zip(info['objects'], keys):
        for r, c in obj['cells']:
            out[r, c] = winners[key]
    return out, dict(info=info, mask=mask, ballots=dict(ballots), winners=winners,
                     eligible=eligible)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer rail geometry, whole-object franchise and exact-shape elections."""
    return _predict(_grid(input_grid))[0]


def _different(pair, alternative, data=None, require_valid=False):
    try:
        out, _ = _predict(pair['input'], alternative,
                          None if data is None else data['info'])
    except AssertionError:
        return not require_valid
    return not np.array_equal(out, pair['output'])


def _block(available, cells):
    h, w = available.shape
    for r, c in cells:
        available[max(0, r-1):min(h, r+2), max(0, c-1):min(w, c+2)] = False


def _place(rng, available, mask, variants, category):
    for j in rng.permutation(len(variants)):
        raw = variants[int(j)]
        h, w = _extent(raw)
        nh, nw = available.shape[0]-h+1, available.shape[1]-w+1
        if min(nh, nw) <= 0:
            continue
        valid = np.ones((nh, nw), dtype=bool)
        inside = np.zeros((nh, nw), dtype=np.int16)
        for r, c in raw:
            valid &= available[r:r+nh, c:c+nw]
            inside += mask[r:r+nh, c:c+nw]
        if category == 'inside':
            valid &= inside == len(raw)
        elif category == 'outside':
            valid &= inside == 0
        elif category == 'partial':
            # A majority-inside witness also refutes centroid-like heuristics.
            valid &= (inside*2 > len(raw)) & (inside < len(raw))
        positions = np.argwhere(valid)
        if len(positions):
            r, c = map(int, positions[int(rng.integers(len(positions)))])
            cells = tuple((r+dr, c+dc) for dr, dc in raw)
            _block(available, cells)
            return cells
    return None


def _geometry(rng, teaching, load, simple=False):
    compact = simple and load <= 3 and rng.random() < .65
    spanq = int(rng.integers(10, 21)) if compact else (
        int(rng.integers(20, 29)) if load > 6 else int(rng.integers(17, 27)))
    spant = int(rng.integers(12, 23)) if compact else (
        int(rng.integers(22, 31)) if load > 6 else int(rng.integers(18, 29)))
    count = int(rng.integers(3, 5)) if teaching else int(rng.integers(2, 6))
    qs = [0, spanq]
    options = [q for q in range(5, spanq-4)]
    rng.shuffle(options)
    for q in options:
        if len(qs) == count:
            break
        if min(abs(q-v) for v in qs) >= 4:
            qs.append(q)
    qs.sort()
    runs = []
    for rank, q in enumerate(qs):
        if teaching and rank not in (0, len(qs)-1):
            lo = int(rng.integers(spant//3, spant//2+1))
            hi = lo+2*int(rng.integers(1, 3))
        else:
            lo = int(rng.integers(0, 5))
            hi = spant-int(rng.integers(0, 5))
            if not teaching and rank not in (0, len(qs)-1):
                lo -= int(rng.integers(0, 7))
                hi += int(rng.integers(0, 7))
        lo += (q-lo) % 2
        hi -= (hi-q) % 2
        runs.append((q, lo, hi))
    # Render at slope +1, then reflect the complete pair for slope -1.
    cells = [((t-q)//2, (q+t)//2) for q, lo, hi in runs for t in range(lo, hi+1, 2)]
    minr, minc = np.min(cells, axis=0)
    maxr, maxc = np.max(cells, axis=0)
    h, w = int(maxr-minr+1), int(maxc-minc+1)
    if max(h, w) > 30:
        return None
    # Room for remote, nonvoting recipients is sampled independently.
    lower = 16 if compact else (25 if load > 6 else 23)
    upper = 25 if compact else 30
    H = int(rng.integers(max(h, lower), max(h, upper)+1))
    W = int(rng.integers(max(w, lower), max(w, upper)+1))
    dy, dx = int(rng.integers(H-h+1))-int(minr), int(rng.integers(W-w+1))-int(minc)
    shifted = [(q+dx-dy, lo+dy+dx, hi+dy+dx) for q, lo, hi in runs]
    actual = [(r+dy, c+dx) for r, c in cells]
    return (H, W), shifted, actual


def _choose(rng, sequence):
    return sequence[int(rng.integers(len(sequence)))]


def _make_pair(rng, shapes, mode, slope, seen, forbidden_winner=None):
    teaching = mode in ('teach', 'rich')
    for attempt in range(120):
        palette = list(map(int, rng.permutation(10)))
        bg, rail = palette[:2]
        colors = palette[2:]
        winners = {s: colors[i] for i, s in enumerate(shapes)}
        if not teaching and len(shapes) > 1 and rng.random() < .35:
            # Independent species may reach the same consensus color.
            shared_pool = colors[:int(rng.integers(1, len(shapes)+1))]
            winners = {s: _choose(rng, shared_pool) for s in shapes}
        if forbidden_winner is not None and winners[shapes[0]] == forbidden_winner:
            swap = next(c for c in colors if c not in winners.values()
                        and c != forbidden_winner)
            winners[shapes[0]] = swap
        specs = []
        for i, shape in enumerate(shapes):
            win = winners[shape]
            losers = [c for c in colors if c != win]
            rng.shuffle(losers)
            if mode == 'rich' and i == 0:
                ballot = (3, 2, 2)
            elif mode == 'rich':
                ballot = (1,)
            elif mode == 'teach':
                ballot = (2, 1)
            else:
                ballot = _choose(rng, ((1,), (2, 1), (2, 1), (3, 1), (2, 1, 1), (3, 2)))
            for color, count in zip([win]+losers, ballot):
                variants = _variants(shape)
                if teaching and i == 0:
                    variants = (_mirror(shape),) if color == win else (shape,)
                specs.extend((shape, color, variants, 'inside') for _ in range(count))
            # Nonvoters reverse the global census and crossing-object ballot.
            # Their colors are visible data, never an output-only highlight.
            if teaching and i == 0:
                loser = losers[0]
                partial_count = 2
                specs.extend((shape, loser, _variants(shape), 'partial')
                             for _ in range(partial_count))
                outside_count = (ballot[0]+1) - (ballot[1] if len(ballot) > 1 else 0) - partial_count
                specs.extend((shape, loser, _variants(shape), 'outside')
                             for _ in range(max(1, outside_count)))
            else:
                n = int(rng.integers(1, 4))
                specs.extend((shape, losers[0], _variants(shape), 'outside') for _ in range(n))
                if not teaching and rng.random() < .5:
                    specs.append((shape, losers[0], _variants(shape), 'partial'))
        geom = _geometry(rng, teaching, sum(v[3] == 'inside' for v in specs), len(shapes) == 1)
        if geom is None:
            continue
        dims, runs, rail_cells = geom
        mask = _field(dims, 1, runs)
        available = np.ones(dims, dtype=bool)
        _block(available, rail_cells)
        # Hard-to-place boundary objects precede interior objects and remotes.
        order = list(map(int, rng.permutation(len(specs))))
        order.sort(key=lambda j: ({'partial': 0, 'inside': 1, 'outside': 2}[specs[j][3]],
                                  -len(specs[j][0])))
        placed = []
        for j in order:
            shape, color, variants, category = specs[j]
            cells = _place(rng, available, mask, variants, category)
            if cells is None:
                break
            placed.append((shape, color, cells))
        else:
            inp = np.full(dims, bg, dtype=np.int8)
            for r, c in rail_cells:
                inp[r, c] = rail
            out = inp.copy()
            for shape, color, cells in placed:
                for r, c in cells:
                    inp[r, c], out[r, c] = color, winners[shape]
            if slope == -1:
                inp, out = np.fliplr(inp).copy(), np.fliplr(out).copy()
            pair = dict(input=inp, output=out)
            if _key(inp) in seen:
                continue
            try:
                expected, data = _predict(inp)
                if not np.array_equal(out, expected):
                    continue
                if teaching:
                    required = ('all', 'area', 'bbox', 'translation', 'rotation',
                                'any', 'majority', 'consecutive', 'pixel_centers', 'infinite')
                    if not all(_different(pair, alt, data) for alt in required):
                        continue
                    if not _different(pair, 'all', data, require_valid=True):
                        continue
            except AssertionError:
                continue
            return pair
    raise RuntimeError('bounded object/rail packing exhausted')


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _counts(num_train, num_test):
    for name, n, lo, hi in (('num_train', num_train, 2, 10),
                            ('num_test', num_test, 1, 4)):
        if (not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_))
                or not lo <= n <= hi):
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    pool = _pool()
    anchors = [s for s in pool if len(s) in (4, 5) and max(_extent(s)) <= 3
               and _mirror(s) not in _variants(s, False)]
    anchor = _choose(rng, anchors)
    companions = [s for s in pool if s != anchor and len(s) == len(anchor)
                  and sorted(_extent(s)) == sorted(_extent(anchor))]
    companion = _choose(rng, companions)
    puzzle, seen, known = dict(train=[], test=[]), set(), set()
    base_slope = int(rng.choice([-1, 1]))
    first_winner = None
    for split, count in (('train', nt), ('test', ne)):
        for i in range(count):
            if split == 'train' and i < 2:
                shapes, mode = [anchor, companion], 'teach' if i == 0 else 'rich'
            else:
                k = int(rng.integers(1, 4))
                if split == 'test':
                    unseen = [s for s in pool if s not in known]
                    first = _choose(rng, unseen)
                    others = [s for s in pool if s != first]
                    shapes = [first] + [others[int(j)] for j in rng.choice(len(others), k-1, replace=False)]
                else:
                    shapes = [pool[int(j)] for j in rng.choice(len(pool), k, replace=False)]
                mode = 'ordinary'
            slope = base_slope*(-1 if i % 2 else 1) if split == 'train' else int(rng.choice([-1, 1]))
            pair = _make_pair(rng, shapes, mode, slope, seen,
                              first_winner if split == 'train' and i == 1 else None)
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
            if split == 'train':
                known.update(shapes)
                if i == 0:
                    first_winner = _predict(pair['input'])[1]['winners'][anchor]
    validate_generated_puzzle(puzzle)
    return puzzle


_ALTERNATIVES = ('all', 'area', 'bbox', 'translation', 'rotation', 'any',
                 'majority', 'consecutive', 'pixel_centers', 'infinite')


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, data_by_split = set(), dict(train=[], test=[])
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape, 'preserve canvas'
            key = _key(inp)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            expected, data = _predict(inp)
            info, mask = data['info'], data['mask']
            assert np.array_equal(mask, _pairwise_field(inp.shape, info['slope'], info['runs']))
            assert np.array_equal(expected, out), 'wrong election or changed geometry'
            preserve = (inp == info['background']) | (inp == info['rail'])
            assert np.array_equal(inp[preserve], out[preserve]), 'preserve field and rails'
            owner = np.full(inp.shape, -1, dtype=np.int16)
            for i, obj in enumerate(info['objects']):
                for r, c in obj['cells']:
                    owner[r, c] = i
                assert np.all(out[tuple(np.array(obj['cells']).T)] == data['winners'][obj['shape']])
            # Distinct objects and rails must have an eight-neighbor moat.
            for i, obj in enumerate(info['objects']):
                for r, c in obj['cells']:
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            rr, cc = r+dr, c+dc
                            if 0 <= rr < inp.shape[0] and 0 <= cc < inp.shape[1]:
                                assert inp[rr, cc] != info['rail'], 'object touches rail'
                                assert owner[rr, cc] in (-1, i), 'objects touch'
            data_by_split[split].append(data)
    # A valid, wrong global census proves that B changes the actual answer.
    assert any(_different(p, 'all', d, require_valid=True)
               for p, d in zip(puzzle['train'], data_by_split['train'])), 'missing spatial election witness'
    for alt in _ALTERNATIVES:
        if any(_different(p, alt, d) for p, d in zip(puzzle['test'], data_by_split['test'])):
            assert any(_different(p, alt, d) for p, d in zip(puzzle['train'], data_by_split['train'])), f'missing {alt} training witness'
    assert any(not np.array_equal(p['input'], p['output']) for p in puzzle['train'])


def _fingerprint(puzzle):
    digest = sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                key = _key(pair[role])
                digest.update(bytes(key[0]))
                digest.update(key[1])
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds; no file writes, unbounded logs or retained puzzles."""
    # Analytical B fixture: three bracket cells, including between-pixel ones.
    grid = np.zeros((5, 5), dtype=np.int8)
    runs = [(2, 2, 6), (6, 2, 6)]
    expected = np.zeros((5, 5), dtype=bool)
    expected[(0, 0, 1), (3, 4, 4)] = True
    # In q=c-r,t=r+c coordinates these are the actual overlap parallelogram.
    direct = _field(grid.shape, 1, runs)
    assert np.array_equal(direct, expected)
    assert np.array_equal(direct, _pairwise_field(grid.shape, 1, runs))
    # Hand-derived values separately guard strict q and inclusive t endpoints.
    assert direct[0, 3] and direct[0, 4] and direct[1, 4]
    assert not direct[0, 2] and not direct[1, 3] and not direct[2, 4]
    # Hand-placed new-rule fixture. Two blue eligible dominos beat one red;
    # a partly enclosed red domino and a remote red domino have no vote.
    # All five recipients must become blue, though the global census is red.
    fixture = np.zeros((15, 15), dtype=np.int8)
    for r in range(6, 13):
        fixture[r, r-6] = 9
    for r in range(7):
        fixture[r, r+6] = 9
    blue = [(5, 4), (5, 5), (8, 7), (9, 7)]
    red = [(5, 8), (6, 8), (2, 3), (3, 3), (13, 10), (13, 11)]
    for r, c in blue:
        fixture[r, c] = 2
    for r, c in red:
        fixture[r, c] = 3
    answer = fixture.copy()
    for r, c in red:
        answer[r, c] = 2
    assert np.array_equal(solve(fixture), answer)
    assert _different(dict(input=fixture, output=answer), 'all', require_valid=True)
    # Reflection and palette bijections preserve the geometry/elections.
    remap = np.array([4, 0, 7, 1, 3, 5, 6, 2, 9, 8], dtype=np.int8)
    assert np.array_equal(solve(np.fliplr(remap[fixture])), np.fliplr(remap[answer]))
    fingerprints, dims, layouts, vocab, ballots, slopes, palettes = (set() for _ in range(7))
    class_counts, object_counts, rail_counts, partial_counts = (set() for _ in range(4))
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        fp = _fingerprint(p)
        assert fp == _fingerprint(generate_puzzle(seed))
        fingerprints.add(fp)
        for split in ('train', 'test'):
            for pair in p[split]:
                assert np.array_equal(solve(pair['input']), pair['output'])
                _, d = _predict(pair['input'])
                a = d['info']
                dims.add(pair['input'].shape)
                slopes.add(a['slope'])
                palettes.add((a['background'], a['rail']))
                vocab.update(o['shape'] for o in a['objects'])
                ballots.update(tuple(sorted(v.values(), reverse=True)) for v in d['ballots'].values())
                class_counts.add(len(d['ballots']))
                object_counts.add(len(a['objects']))
                rail_counts.add(len(a['runs']))
                partial_counts.add(sum(0 < sum(d['mask'][r, c] for r, c in o['cells']) < len(o['cells']) for o in a['objects']))
                q0, t0 = min(v[0] for v in a['runs']), min(v[1] for v in a['runs'])
                layouts.add(tuple((q-q0, lo-t0, hi-t0) for q, lo, hi in a['runs']))
    assert len(fingerprints) == 42
    assert len(dims) >= 15 and len(layouts) >= 150
    assert len(vocab) >= 20 and len(palettes) >= 40
    assert slopes == {-1, 1} and {1, 2, 3} <= class_counts
    assert len(object_counts) >= 10 and {2, 3, 4} <= rail_counts
    assert {(1,), (2, 1), (3, 2, 2)} <= ballots
    assert max(partial_counts) >= 2
    for nt, ne in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(100+nt, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    broken = generate_puzzle(0, 2, 1)
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    return dict(seeds=42, distinct_puzzles=len(fingerprints), canvas_shapes=len(dims),
                rail_layouts=len(layouts), shape_classes=len(vocab), palettes=len(palettes),
                ballot_patterns=sorted(ballots), object_counts=sorted(object_counts),
                rail_counts=sorted(rail_counts), semantic_check='authored checks only')
