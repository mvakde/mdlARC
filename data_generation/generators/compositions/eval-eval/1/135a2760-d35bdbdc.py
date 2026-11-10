"""Misprints in a periodic relation mosaic become simultaneous two-hop links.

Only standard-library modules and NumPy are used. The solver parses the input;
generation builds labels from its clean construction, then validates that parsing
and unique periodic recovery rediscover that construction.
"""
from collections import Counter
import numpy as np


_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_PERIODS = ((2, 2), (1, 4), (4, 1), (2, 3), (3, 2), (3, 3))


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"invalid {name}")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= d <= 30 for d in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("invalid ARC grid")
    return a.astype(np.int8, copy=True)


def _support(center, kind):
    r, c = center
    return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if kind == 'square' or dr == 0 or dc == 0}


def _parse(grid):
    h, w = grid.shape
    if min(h, w) < 7:
        raise ValueError("missing frame and token mosaic")
    frame, bg = int(grid[0, 0]), int(grid[1, 1])
    if frame == bg:
        raise ValueError("frame and padding must differ")
    if not (np.all(grid[0] == frame) and np.all(grid[-1] == frame)
            and np.all(grid[:, 0] == frame) and np.all(grid[:, -1] == frame)
            and np.all(grid[1, 1:-1] == bg) and np.all(grid[-2, 1:-1] == bg)
            and np.all(grid[1:-1, 1] == bg) and np.all(grid[1:-1, -2] == bg)):
        raise ValueError("invalid frame or inner padding ring")
    if np.any(grid[1:-1, 1:-1] == frame):
        raise ValueError("frame color reserved for the outer border")
    todo = set(zip(*np.where((grid != frame) & (grid != bg))))
    tokens = {}
    mask = np.zeros(grid.shape, dtype=bool)
    while todo:
        p = min(todo)
        todo.remove(p)
        cells, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in _D4:
                q = (r + dr, c + dc)
                if q in todo:
                    todo.remove(q)
                    cells.add(q)
                    stack.append(q)
        rs, cs = zip(*cells)
        if max(rs) - min(rs) != 2 or max(cs) - min(cs) != 2:
            raise ValueError("token must have a centered 3 by 3 support")
        center = (min(rs) + 1, min(cs) + 1)
        kinds = [k for k in ('square', 'cross') if cells == _support(center, k)]
        bodies = {int(grid[q]) for q in cells if q != center}
        if len(kinds) != 1 or len(bodies) != 1:
            raise ValueError("ambiguous token body or support")
        tokens[center] = dict(center=center, cells=cells, kind=kinds[0],
                              body=bodies.pop(), value=int(grid[center]))
        for q in cells:
            mask[q] = True
    rows = sorted({r for r, c in tokens})
    cols = sorted({c for r, c in tokens})
    if len(rows) < 2 or len(cols) < 2 or len(tokens) != len(rows) * len(cols):
        raise ValueError("tokens must form a complete rectangular lattice")
    for coordinates in (rows, cols):
        gaps = np.diff(coordinates)
        if not np.all(gaps == gaps[0]) or gaps[0] < 4:
            raise ValueError("token centers must be regularly spaced and separated")
    ordered = [[tokens[(r, c)] for c in cols] for r in rows]
    relations = np.array([[(t['body'], t['value']) for t in row] for row in ordered],
                         dtype=np.int8)
    return dict(frame=frame, bg=bg, rows=rows, cols=cols, tokens=ordered,
                relations=relations, mask=mask)


def _recover(relations):
    """All supported periods must agree on the complete denoised relation field.

    A candidate needs at least three observations in EVERY phase, a strict
    majority of identical ordered pairs, unchanged bodies and <= N//8 misprints.
    Period bounds come from the input size, not the sampled construction period.
    """
    nr, nc, _ = relations.shape
    candidates = []
    for pr in range(1, nr + 1):
        for pc in range(1, nc + 1):
            if (nr // pr) * (nc // pc) < 3:
                continue
            template = np.empty((pr, pc, 2), dtype=np.int8)
            supported = True
            for r in range(pr):
                for c in range(pc):
                    observations = relations[r::pr, c::pc].reshape(-1, 2)
                    counts = Counter(map(tuple, observations.tolist()))
                    winner, count = counts.most_common(1)[0]
                    if count * 2 <= len(observations):
                        supported = False
                        break
                    template[r, c] = winner
                if not supported:
                    break
            if not supported:
                continue
            clean = template[np.arange(nr)[:, None] % pr, np.arange(nc)[None, :] % pc]
            if not np.array_equal(clean[:, :, 0], relations[:, :, 0]):
                continue
            errors = int(np.count_nonzero(clean[:, :, 1] != relations[:, :, 1]))
            if 1 <= errors <= nr * nc // 8:
                candidates.append(((pr, pc), clean))
    if not candidates:
        raise ValueError("no supported periodic repair")
    candidates.sort(key=lambda x: (x[0][0] * x[0][1], x[0]))
    clean = candidates[0][1]
    if any(not np.array_equal(clean, c[1]) for c in candidates):
        raise ValueError("periodic explanations imply different repairs")
    return clean, candidates[0][0], tuple(p for p, a in candidates)


def _compose(clean, defects):
    flat = clean.reshape(-1, 2)
    results = {}
    for r, c in zip(*np.where(defects)):
        body, target = map(int, clean[r, c])
        donors = [int(v) for i, (b, v) in enumerate(flat)
                  if int(b) == target and i != r * clean.shape[1] + c]
        if not donors:
            continue
        if len(set(donors)) != 1:
            raise ValueError("a requested identity has conflicting recovered donors")
        results[(int(r), int(c))] = donors[0]
    return results


def _paint(grid, parsed, results):
    result = grid.copy()
    result[parsed['mask']] = parsed['bg']
    for (r, c), value in results.items():
        token = parsed['tokens'][r][c]
        for q in token['cells']:
            result[q] = token['body']
        result[token['center']] = value
    return result


def _analyze(input_grid):
    grid = _grid(input_grid)
    parsed = _parse(grid)
    clean, period, candidates = _recover(parsed['relations'])
    defects = clean[:, :, 1] != parsed['relations'][:, :, 1]
    results = _compose(clean, defects)
    return grid, parsed, clean, defects, results, period, candidates


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid, parsed, clean, defects, results, period, candidates = _analyze(input_grid)
    return _paint(grid, parsed, results)


def _features(input_grid):
    grid, parsed, clean, defects, results, period, candidates = _analyze(input_grid)
    nr, nc = defects.shape
    mappings = {}
    for b, v in clean.reshape(-1, 2):
        mappings.setdefault(int(b), set()).add(int(v))
    retained_donor = nonrecursive = 0
    for r, c in results:
        target = int(clean[r, c, 1])
        for rr, cc in results:
            if (rr, cc) != (r, c) and int(clean[rr, cc, 0]) == target:
                retained_donor += 1
                nonrecursive += results[rr, cc] != int(clean[rr, cc, 1])
    ambiguous_sources = {b for b, vals in mappings.items() if len(vals) > 1}
    periodic_witness = sum(int(clean[r, c, 0]) in ambiguous_sources for r, c in results)
    return dict(shape=grid.shape, lattice=(nr, nc), period=period,
                pitches=(int(parsed['rows'][1] - parsed['rows'][0]),
                         int(parsed['cols'][1] - parsed['cols'][0])),
                frame=parsed['frame'], background=parsed['bg'], tokens=nr*nc,
                defects=int(defects.sum()), survivors=len(results),
                failed=int(defects.sum())-len(results),
                partial=bool(nr % period[0] or nc % period[1]),
                fork_bodies=len(ambiguous_sources), periodic_witness=periodic_witness,
                retained_donor=retained_donor, nonrecursive=nonrecursive,
                kinds=tuple(sorted({t['kind'] for row in parsed['tokens'] for t in row})),
                monochrome=int(np.count_nonzero(clean[:, :, 0] == clean[:, :, 1])),
                composition_changes=sum(v != int(clean[r, c, 1]) for (r, c), v in results.items()),
                identity_colors=len(set(map(int, clean.ravel()))))


def _tile(rng, period, mode):
    # A forked source identity makes periodic phase essential: X has TWO
    # original destinations. Only requested donor identities must be functional.
    size = period[0] * period[1]
    relations = [(0, 1), (0, 2), (1, 2), (2, 3)]
    # The chain X->Y->W->Z teaches simultaneous recovered-original lookup.
    # Failure mode deliberately leaves Z without a body token.
    with_z = mode == 'random' and size > 4 and rng.random() < .65
    donor_map = {1: 2, 2: 3}
    if with_z:
        donor_map[3] = int(rng.choice([1, 3]))
        relations.append((3, donor_map[3]))
    while len(relations) < size:
        body = int(rng.choice([0, 1, 2, 4, 5]))
        if body == 0:
            value = int(rng.choice([1, 2]))
        elif body in donor_map:
            value = donor_map[body]
        else:
            if body not in donor_map:
                donor_map[body] = int(rng.choice([1, 2, 3, body]))
            value = donor_map[body]
        relations.append((body, value))
    permutation = list(map(int, rng.permutation(size)))
    tile = np.array([relations[i] for i in permutation], dtype=np.int8).reshape(*period, 2)
    core = {i: divmod(permutation.index(i), period[1]) for i in range(4)}
    return tile, core


def _geometry(rng, nr, nc):
    coordinates, dimensions = [], []
    for count in (nr, nc):
        pitches = [p for p in (4, 5) if 7 + p * (count - 1) <= 30]
        pitch = int(rng.choice(pitches))
        extra = int(rng.integers(0, min(4, 30 - (7 + pitch * (count - 1))) + 1))
        before = int(rng.integers(extra + 1))
        coordinates.append([3 + before + pitch*i for i in range(count)])
        dimensions.append(7 + extra + pitch * (count - 1))
    return tuple(dimensions), coordinates


def _make_pair(rng, parameters, mode, test=False):
    if mode == 'chain':
        period_choices = [(2, 2), (1, 4), (4, 1)]
    elif mode == 'failure':
        period_choices = [(2, 3), (3, 2)]
    else:
        period_choices = list(_PERIODS)
    for attempt in range(96):
        period = period_choices[int(rng.integers(len(period_choices)))]
        pr, pc = period
        sizes = [(r, c) for r in range(4, 7) for c in range(4, 7)
                 if (r // pr)*(c // pc) >= 3 and (not test or r*c >= 24)]
        nr, nc = sizes[int(rng.integers(len(sizes)))]
        tile, core = _tile(rng, period, mode)
        clean = tile[np.arange(nr)[:, None] % pr, np.arange(nc)[None, :] % pc].copy()
        defects = np.zeros((nr, nc), dtype=bool)
        if mode == 'chain':
            phases = [core[0], core[2]]
        elif mode == 'failure':
            phases = [core[1], core[3]]
        else:
            count = int(rng.integers(3, min(4, nr*nc // 8, pr*pc) + 1)) if test else 2
            # Include a fork query in every example; other errors are independent.
            phases = [core[int(rng.integers(2))]]
            available = [(r, c) for r in range(pr) for c in range(pc) if (r, c) not in phases]
            phases += [available[int(i)] for i in rng.choice(len(available), count-1, replace=False)]
        observed = clean.copy()
        for phase in phases:
            r = int(rng.choice(np.arange(phase[0], nr, pr)))
            c = int(rng.choice(np.arange(phase[1], nc, pc)))
            defects[r, c] = True
            wrong = 3 if mode == 'chain' and phase == core[0] else int(rng.choice(
                [v for v in range(6) if v != int(clean[r, c, 1])]))
            observed[r, c, 1] = wrong
        try:
            recovered, actual_period, candidates = _recover(observed)
            if not np.array_equal(clean, recovered):
                continue
            # Construction label uses the sampled clean field, never solve().
            answers = _compose(clean, defects)
            if not answers:
                continue
        except ValueError:
            continue
        palette = list(map(int, rng.permutation(10)))
        if not parameters['varied_roles']:
            frame, bg = parameters['roles']
            palette = [frame, bg] + list(map(int, rng.permutation(
                [x for x in range(10) if x not in (frame, bg)])))
        frame, bg = palette[:2]
        identities = palette[2:]
        shape, (rows, cols) = _geometry(rng, nr, nc)
        grid = np.full(shape, bg, dtype=np.int8)
        grid[0, :] = grid[-1, :] = frame
        grid[:, 0] = grid[:, -1] = frame
        output = grid.copy()
        for r, row in enumerate(rows):
            for c, col in enumerate(cols):
                kind = parameters['morphology']
                if kind == 'mixed':
                    kind = str(rng.choice(['square', 'cross']))
                cells = _support((row, col), kind)
                body, value = map(int, observed[r, c])
                for q in cells:
                    grid[q] = identities[body]
                grid[row, col] = identities[value]
                if (r, c) in answers:
                    for q in cells:
                        output[q] = identities[body]
                    output[row, col] = identities[answers[(r, c)]]
        rotation = int(rng.integers(4))
        grid, output = np.rot90(grid, rotation), np.rot90(output, rotation)
        if rng.random() < .5:
            grid, output = np.fliplr(grid), np.fliplr(output)
        return dict(input=grid.copy(), output=output.copy())
    raise RuntimeError("bounded unambiguous mosaic construction exhausted")


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    nt = _integer(num_train, 'num_train', 2, 10)
    ne = _integer(num_test, 'num_test', 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    parameters = dict(varied_roles=bool(rng.random() < .65),
                      roles=tuple(map(int, rng.choice(10, 2, replace=False))),
                      morphology=str(rng.choice(['square', 'cross', 'mixed'], p=[.35, .2, .45])))
    modes = ['chain', 'failure'] + ['random']*(nt-2)
    rng.shuffle(modes)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ne)):
        for index in range(count):
            mode = modes[index] if split == 'train' else 'random'
            for attempt in range(32):
                pair = _make_pair(rng, parameters, mode, test=split == 'test')
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, statistics = set(), {}
    for split in ('train', 'test'):
        statistics[split] = []
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            grid, parsed, clean, defects, results, period, candidates = _analyze(inp)
            assert np.array_equal(out, _paint(grid, parsed, results))
            assert np.array_equal(inp[~parsed['mask']], out[~parsed['mask']])
            assert 0 < len(results) <= int(defects.sum()) <= defects.size // 8
            assert not np.array_equal(inp, out)
            # Every survivor was an actual misprint, with exact original shape.
            for r, row in enumerate(parsed['tokens']):
                for c, token in enumerate(row):
                    if (r, c) in results:
                        assert defects[r, c]
                        assert out[token['center']] == results[(r, c)]
                        assert all(out[q] == token['body'] for q in token['cells']
                                   if q != token['center'])
                    else:
                        assert all(out[q] == parsed['bg'] for q in token['cells'])
            f = _features(inp)
            assert f['fork_bodies'] and f['periodic_witness']
            assert f['composition_changes']
            assert (f['defects'] == 2 if split == 'train' else 3 <= f['defects'] <= 4)
            statistics[split].append(f)
    train = statistics['train']
    assert any(f['retained_donor'] and f['nonrecursive'] for f in train)
    assert any(f['failed'] for f in train)
    assert len({f['period'] for f in train}) >= 2
    assert all(f['tokens'] >= 24 for f in statistics['test'])
    return None


def self_test():
    """Bounded semantic, corruption, reproducibility and 42-seed diversity checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signature = lambda p: tuple((_key(q['input']), _key(q['output']))
                                for s in ('train', 'test') for q in p[s])
    assert len({signature(p) for p in puzzles}) == 42
    assert signature(puzzles[0]) == signature(generate_puzzle(0))
    features = [_features(q['input']) for p in puzzles for s in ('train', 'test') for q in p[s]]
    assert len({f['shape'] for f in features}) >= 25
    assert len({f['lattice'] for f in features}) >= 7
    assert set(_PERIODS) <= {f['period'] for f in features}
    assert {4, 5} <= {int(p) for f in features for p in f['pitches']}
    assert len({f['survivors'] for f in features}) >= 3
    assert len({f['identity_colors'] for f in features}) >= 3
    assert {('square',), ('cross',), ('cross', 'square')} <= {f['kinds'] for f in features}
    assert any(f['partial'] for f in features) and any(not f['partial'] for f in features)
    assert len({(f['frame'], f['background']) for f in features}) >= 25
    assert any(30 in f['shape'] for f in features)
    for f in features:
        assert f['periodic_witness'] and f['composition_changes']
    # Specification-valid inputs outside the generation schedule: a 1x3
    # relation tile, two errors in the SAME phase, and a faulty monochrome
    # relation that composes to a solid token. Expected answers are written
    # directly here, without calling the generator's construction or composer.
    diagnostic = np.zeros((19, 27), dtype=np.int8)
    diagnostic[0, :] = diagnostic[-1, :] = 9
    diagnostic[:, 0] = diagnostic[:, -1] = 9
    expected = diagnostic.copy()
    relation_word = ((1, 2), (2, 3), (3, 3))
    faults = {(0, 0), (2, 3), (3, 5)}
    for r in range(4):
        for c in range(6):
            center = (3+4*r, 3+4*c)
            body, value = relation_word[c % 3]
            for q in _support(center, 'cross'):
                diagnostic[q] = body
            diagnostic[center] = 4 if (r, c) in faults else value
            if (r, c) in faults:
                for q in _support(center, 'cross'):
                    expected[q] = body
                expected[center] = 3
    assert np.array_equal(solve(diagnostic), expected)
    # Omitting B: keep the true faults but only repair their one-hop centers.
    # Omitting A: treat every observed token with any matching raw body as a
    # query. Neither variant fits any puzzle's whole training set.
    for puzzle in puzzles:
        differs_one_hop = differs_raw = False
        for pair in puzzle['train']:
            grid, parsed, clean, defects, results, _, _ = _analyze(pair['input'])
            repaired_requests = {(int(r), int(c)): int(clean[r, c, 1])
                                 for r, c in zip(*np.where(defects))}
            differs_one_hop |= not np.array_equal(_paint(grid, parsed, repaired_requests), pair['output'])
            raw = parsed['relations']
            try:
                raw_results = _compose(raw, np.ones(defects.shape, dtype=bool))
                differs_raw |= not np.array_equal(_paint(grid, parsed, raw_results), pair['output'])
            except ValueError:
                differs_raw = True  # noisy raw donors conflict; decoded ones do not
        assert differs_one_hop and differs_raw
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(101, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Validate rejects a changed scaffold and an uncomposed repaired center.
    original = puzzles[0]
    for damage in ('frame', 'one_hop'):
        bad = {s: [{k: np.asarray(v).copy() for k, v in q.items()} for q in original[s]]
               for s in ('train', 'test')}
        if damage == 'frame':
            bad['train'][0]['output'][0, 0] = (int(bad['train'][0]['output'][0, 0])+1) % 10
        else:
            changed = False
            for pair in bad['train']:
                _, parsed, clean, defects, results, _, _ = _analyze(pair['input'])
                for (r, c), value in results.items():
                    if value != int(clean[r, c, 1]):
                        pair['output'][parsed['tokens'][r][c]['center']] = clean[r, c, 1]
                        changed = True
                        break
                if changed:
                    break
            assert changed
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("semantic corruption accepted")
    return None
