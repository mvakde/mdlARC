"""Elastic ribbons with printing registered to their original tips.

Only NumPy and the standard library are used. Generation constructs labels from
sampled ribbon data; solve reconstructs that data from the input cells alone.
No files, global examples, or random state are consulted by solve.
"""
from __future__ import annotations

import hashlib
import numpy as np


REFERENCES = (6, 8, 9, 10, 12, 15, 16, 18, 20, 24)
SELF_TEST_SEEDS = (0, 85669303, 879957299, 811535761, 176169198, 936711691,
    754048170, 822364386, 930978253, 964652773, 959319205, 84110011,
    680126089, 271963826, 573172032, 145630403, 657001973, 845499352,
    75162778, 348608941, 289846914, 905157711, 217434740, 847824585,
    962685751, 615552607, 844645352, 163103642, 372418070, 445224990,
    734684849, 519341140, 218738239, 722894275, 258149031, 459873535,
    813759140, 260709223, 964769803, 21370286, 414477891, 646801223)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("a grid must be an integer 1..30 rectangle with colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return (a.shape, a.tobytes())


def _orient(a, turns, mirror):
    a = np.rot90(a, turns)
    return np.fliplr(a).copy() if mirror else a.copy()


def _undo(a, turns, mirror):
    if mirror:
        a = np.fliplr(a)
    return np.rot90(a, -turns).copy()


def _try_decode(g, background, marker):
    """Read one possible coordinate frame; return None if its grammar fails."""
    rows = np.flatnonzero(np.any(g == marker, axis=1))
    if len(rows) not in (3, 5, 7) or np.any(np.diff(rows) < 4):
        return None
    if rows[0] < 1 or rows[-1] >= g.shape[0] - 1:
        return None
    starts, lengths = [], []
    for r in rows:
        positions = np.flatnonzero(g[r] == marker)
        if len(positions) < 2 or np.any(np.diff(positions) != 1):
            return None
        starts.append(int(positions[0]))
        lengths.append(len(positions))
    if len(set(starts)) != 1:
        return None
    anchor = starts[0]
    target = lengths[len(rows)//2]
    if not 2 <= target <= 26 or anchor + target >= g.shape[1]:
        return None
    occupied = np.zeros(g.shape, dtype=bool)
    ribbons = []
    for r0, length in zip(rows, lengths):
        r = int(r0)
        cells = np.flatnonzero(g[r-1] != background)
        n = len(cells)
        if (not 2 <= n <= 6 or length < n or length > 26
                or target < n or target % n != 0
                or not np.array_equal(cells, np.arange(anchor, anchor+n))):
            return None
        carrier = g[r-1, anchor:anchor+n].copy()
        start = anchor + length - n
        stencil = g[r+1, start:anchor+length].copy()
        if (len(stencil) != n or np.all(stencil != background)
                or np.any(carrier == marker) or np.any(stencil == marker)):
            return None
        occupied[r, anchor:anchor+length] = True
        occupied[r-1, anchor:anchor+n] = True
        occupied[r+1, start:anchor+length] = True
        ribbons.append(dict(row=r, length=length, n=n,
                            carrier=carrier, stencil=stencil))
    if np.any((g != background) & ~occupied):
        return None
    return dict(grid=g, background=background, marker=marker, anchor=anchor,
                target=target, ribbons=ribbons)


def _decode(input_grid):
    g = _grid(input_grid)
    bg = int(g[0, 0])
    if not (np.all(g[0] == bg) and np.all(g[-1] == bg)
            and np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg)):
        raise ValueError("the input needs a uniform one-cell background border")
    candidates = []
    colors = [int(c) for c in np.unique(g) if c != bg]
    for turns in range(4):
        for mirror in (False, True):
            canonical = _orient(g, turns, mirror)
            for marker in colors:
                decoded = _try_decode(canonical, bg, marker)
                if decoded is not None:
                    decoded.update(turns=turns, mirror=mirror)
                    candidates.append(decoded)
    if len(candidates) != 1:
        raise ValueError("input must have a unique ribbon coordinate frame")
    return candidates[0]


def _render(decoded, alternative=None):
    """Cell-loop solver and explicit competing hypotheses used for evidence."""
    g, bg = decoded['grid'], decoded['background']
    out = g.copy()
    ribbons, a = decoded['ribbons'], decoded['anchor']
    target = decoded['target']
    if alternative == 'median_length':
        target = sorted(d['length'] for d in ribbons)[len(ribbons)//2]
    elif alternative == 'first_length':
        target = ribbons[0]['length']
    elif alternative == 'last_length':
        target = ribbons[-1]['length']
    elif alternative == 'geometric_middle':
        midpoint2 = ribbons[0]['row'] + ribbons[-1]['row']
        closest = min(ribbons, key=lambda d: (abs(2*d['row']-midpoint2), d['row']))
        target = closest['length']
    for d in ribbons:
        r, old, n = d['row'], d['length'], d['n']
        width = old if alternative == 'keep_lengths' else target
        if a + width >= g.shape[1]:
            return None  # The competitor would run out of canvas.
        out[r, a:a+old] = bg
        for x in range(width):
            index = x*n//width
            base = int(d['carrier'][x % n if alternative == 'tile_carrier' else index])
            if alternative == 'stretch_stencil':
                ink = int(d['stencil'][index])
            elif alternative == 'root_phase':
                ink = int(d['stencil'][x % n])
            else:
                # Its visible swatch occupies [old-n,old) in root coordinates.
                ink = int(d['stencil'][(x-old) % n])
            if alternative == 'normalize_only':
                value = decoded['marker']
            elif alternative == 'ignore_stencil':
                value = base
            elif alternative == 'opaque_background':
                value = ink
            else:
                value = base if ink == bg else ink
            out[r, a+x] = value
    return _undo(out, decoded['turns'], decoded['mirror'])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer root, direction, swatches, guide ink and reference from the input."""
    return _render(_decode(input_grid))


ALTERNATIVES = frozenset(('median_length', 'first_length', 'last_length',
    'geometric_middle', 'keep_lengths', 'normalize_only', 'tile_carrier',
    'stretch_stencil', 'root_phase', 'ignore_stencil', 'opaque_background'))


def _evidence(pair, decoded=None):
    d = _decode(pair['input']) if decoded is None else decoded
    expected = np.asarray(pair['output'])
    return {name for name in ALTERNATIVES
            if not np.array_equal(_render(d, name), expected)}


def _carrier(rng, n, palette):
    k = int(rng.integers(1, min(4, n, len(palette))+1))
    colors = rng.choice(palette, k, replace=False)
    # Unequal runs, alternating stripes and arbitrary sequences all occur.
    kind = int(rng.integers(3))
    if kind == 0:
        cuts = sorted(rng.choice(np.arange(1, n), k-1, replace=False))
        labels = np.searchsorted(cuts, np.arange(n), side='right')
    elif kind == 1:
        labels = np.arange(n) % k
    else:
        labels = rng.integers(k, size=n)
        labels[rng.choice(n, k, replace=False)] = np.arange(k)
    if rng.integers(2):
        labels = labels[::-1]
    return colors[labels].astype(np.int8)


def _stencil(rng, n, palette, background):
    out = np.full(n, background, dtype=np.int8)
    if rng.random() < .15:
        return out
    count = int(rng.integers(1, n))
    kind = int(rng.integers(3))
    if kind == 0:
        positions = rng.choice(n, count, replace=False)
    elif kind == 1:
        start = int(rng.integers(n-count+1))
        positions = np.arange(start, start+count)
    else:
        positions = np.arange(int(rng.integers(2)), n, 2)
    k = int(rng.integers(1, min(3, len(palette), len(positions))+1))
    inks = rng.choice(palette, k, replace=False)
    out[positions] = rng.choice(inks, len(positions))
    return out


def _construct(rng, bg, marker, palette, reference, count, orientation,
               witness=False, mode=None):
    periods = [n for n in range(2, 7) if reference % n == 0 and 2*n <= reference]
    ns = [int(rng.choice(periods)) for _ in range(count)]
    if witness and len(periods) > 1:
        chosen = rng.choice(count, 2, replace=False)
        ns[int(chosen[0])], ns[int(chosen[1])] = periods[0], periods[-1]
    cap = int(rng.integers(reference+1, min(26, reference+8)+1))
    lengths = [reference] * count
    middle = count//2
    others = [int(i) for i in rng.permutation(count) if i != middle]
    mode = str(rng.choice(('mixed', 'extend', 'trim', 'one_extend', 'one_trim'))) if mode is None else mode
    short = lambda i: int(rng.integers(ns[i], reference))
    long = lambda: int(rng.integers(reference+1, cap+1))
    if witness:
        majority_short = bool(rng.integers(2))
        for i in others:
            lengths[i] = short(i) if majority_short else long()
        i = others[0]
        lengths[i] = long() if majority_short else short(i)
    elif mode.startswith('one_'):
        i = others[0]
        lengths[i] = short(i) if mode == 'one_extend' else long()
    else:
        for i in others:
            if rng.random() < .23:
                continue
            lengths[i] = short(i) if mode == 'extend' or (mode == 'mixed' and rng.random() < .5) else long()
        if mode == 'mixed':
            lengths[others[0]], lengths[others[1]] = short(others[0]), long()
        elif len(set(lengths)) == 1:
            lengths[others[0]] = short(others[0]) if mode == 'extend' else long()
    gaps = [4] * (count-1)
    if witness:
        # An outer large gap separates ordinal from geometric centrality.
        gaps[0 if rng.integers(2) else -1] += int(rng.choice((5, 7)))
    else:
        budget = int(rng.integers(0, min(7, 25-sum(gaps))+1))
        for _ in range(budget):
            gaps[int(rng.integers(count-1))] += 1
    height_min = sum(gaps)+5
    width_min = max(reference, *lengths)+2
    height = int(rng.integers(height_min, 31))
    width = int(rng.integers(width_min, 31))
    first_row = int(rng.integers(2, height-sum(gaps)-2))
    anchor = int(rng.integers(1, width-max(reference, *lengths)))
    rows = first_row + np.r_[0, np.cumsum(gaps)]
    inp = np.full((height, width), bg, dtype=np.int8)
    out = inp.copy()
    for row0, length, n in zip(rows, lengths, ns):
        r = int(row0)
        carrier, stencil = _carrier(rng, n, palette), _stencil(rng, n, palette, bg)
        inp[r-1, anchor:anchor+n] = carrier
        inp[r, anchor:anchor+length] = marker
        inp[r+1, anchor+length-n:anchor+length] = stencil
        out[r-1, anchor:anchor+n] = carrier
        out[r+1, anchor+length-n:anchor+length] = stencil
        # Constructor is a vectorized material projection, separate from solve.
        base = np.repeat(carrier, reference//n)
        inks = stencil[(np.arange(reference)-length) % n]
        out[r, anchor:anchor+reference] = np.where(inks == bg, base, inks)
    turns, mirror = orientation//2, bool(orientation % 2)
    return {'input': _orient(inp, turns, mirror),
            'output': _orient(out, turns, mirror)}


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high):
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate one coherent puzzle; all stochastic choices use this RNG."""
    train_count = _count(num_train, 'num_train', 2, 10)
    test_count = _count(num_test, 'num_test', 1, 4)
    if train_count <= test_count:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed) % (1 << 64))
    colors = rng.permutation(10)
    bg, marker = int(colors[0]), int(colors[1])
    palette = colors[2:2+int(rng.integers(3, 9))]
    witness_ref = int(rng.choice((12, 18, 20, 24)))
    other_refs = [r for r in REFERENCES if r != witness_ref]
    train_refs = [witness_ref] + [int(x) for x in rng.choice(other_refs, 2, replace=False)]
    orientation = int(rng.integers(8))
    seen, puzzle = set(), {'train': [], 'test': []}
    evidence = set()

    def add(split, reference, count, orient, witness=False, need=frozenset()):
        # Bounded retries retain no failed grids or logs. The geometric witness
        # already forces A's two selection counterexamples; patterns supply B.
        for _ in range(128):
            pair = _construct(rng, bg, marker, palette, reference, count, orient, witness)
            if _key(pair['input']) in seen:
                continue
            d = _decode(pair['input'])
            cover = _evidence(pair, d)
            if not need <= cover:
                continue
            assert np.array_equal(_render(d), pair['output'])
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
            return cover
        raise RuntimeError('bounded ribbon sampler could not supply teaching evidence')

    # Both selection witnesses fit within a five-ribbon array. No color,
    # reference, period, pattern, phase or absolute direction is fixed by index.
    evidence |= add('train', witness_ref, 5, orientation, witness=True,
                    need=frozenset(('median_length', 'geometric_middle')))
    second_orientation = (orientation + int(rng.choice((2, 6)))) % 8
    evidence |= add('train', train_refs[1], int(rng.choice((3, 7))), second_orientation,
                    need=ALTERNATIVES-evidence)
    for _ in range(2, train_count):
        evidence |= add('train', int(rng.choice(train_refs)), int(rng.choice((3, 5, 7))),
                        int(rng.integers(8)))
    held_refs = [r for r in REFERENCES if r not in train_refs]
    # A new visible reference length is always the test shift. Orientations and
    # material complexity otherwise share support with the demonstrations.
    for _ in range(test_count):
        add('test', int(rng.choice(held_refs)), int(rng.choice((3, 5, 7))),
            int(rng.integers(8)))
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    """Validate returned cells, material semantics, and demonstration evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, roles = set(), set()
    decoded = {'train': [], 'test': []}
    evidence = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            d = _decode(inp)
            expected = _render(d)
            assert np.array_equal(out, expected), 'incorrect material or normalization'
            roles.add((d['background'], d['marker']))
            decoded[split].append(d)
            canon_out = _orient(out, d['turns'], d['mirror'])
            change_region = np.zeros(d['grid'].shape, dtype=bool)
            a, target, bg = d['anchor'], d['target'], d['background']
            for item in d['ribbons']:
                r, old = item['row'], item['length']
                change_region[r, a:a+max(old, target)] = True
                assert np.all(canon_out[r, a:a+target] != bg)
                assert np.all(canon_out[r, a+target:a+old] == bg)
            assert np.array_equal(canon_out[~change_region], d['grid'][~change_region]), 'swatches changed'
            assert not np.any(canon_out == d['marker']), 'unprinted guide ink remains'
            if split == 'train':
                evidence |= _evidence(pair, d)
    assert len(roles) == 1, 'background and guide convention changed within puzzle'
    assert ALTERNATIVES <= evidence, 'demonstrations miss: ' + str(ALTERNATIVES-evidence)
    train = decoded['train']
    assert len({d['target'] for d in train}) >= 2
    assert len({len(d['ribbons']) for d in train}) >= 2
    assert any(len({x['n'] for x in d['ribbons']}) > 1 for d in train), 'no mixed-period demonstration'
    assert any(any(x['length'] < d['target'] for x in d['ribbons'])
               and any(x['length'] > d['target'] for x in d['ribbons']) for d in train)
    assert {d['target'] for d in decoded['test']}.isdisjoint({d['target'] for d in train}), 'missing length shift'


def _features(d):
    ribbons = d['ribbons']
    return dict(shape=d['grid'].shape, count=len(ribbons), reference=d['target'],
        gaps=tuple(np.diff([x['row'] for x in ribbons]).tolist()),
        lengths=tuple(x['length'] for x in ribbons),
        periods=tuple(x['n'] for x in ribbons),
        direction=(d['turns'], d['mirror']), background=d['background'],
        marker=d['marker'], empty=sum(np.all(x['stencil'] == d['background']) for x in ribbons),
        carriers=tuple(tuple(x['carrier'].tolist()) for x in ribbons),
        masks=tuple(tuple((x['stencil'] != d['background']).tolist()) for x in ribbons))


def self_test():
    """Bounded 42-seed campaign; no files or execution traces are written."""
    # Hand-calculated material regression: identical carriers, two different
    # stencils, short/central/long guides. These expected rows are literal.
    example = np.zeros((14, 12), dtype=np.int8)
    for row, length, stamp in ((2, 3, (3, 0)), (6, 6, (0, 4)), (10, 8, (3, 0))):
        example[row-1, 1:3] = (1, 2)
        example[row, 1:1+length] = 5
        example[row+1, 1+length-2:1+length] = stamp
    expected_rows = ((1, 3, 1, 3, 2, 3), (1, 4, 1, 4, 2, 4), (3, 1, 3, 2, 3, 2))
    answer = solve(example)
    for row, values in zip((2, 6, 10), expected_rows):
        assert tuple(answer[row, 1:7]) == values
    assert np.all(answer[10, 7:9] == 0)
    # Change the middle gauge and move its visible tip swatch with it.
    reference_change = example.copy()
    reference_change[6, 1:7], reference_change[7, :] = 0, 0
    reference_change[6, 1:5], reference_change[7, 3:5] = 5, (0, 4)
    shorter = solve(reference_change)
    for row, values in zip((2, 6, 10), ((1, 3, 2, 3), (1, 4, 2, 4), (3, 1, 3, 2))):
        assert tuple(shorter[row, 1:5]) == values
        assert np.all(shorter[row, 5:9] == 0)
    # The outer gauge changes only its own printing phase, while the common
    # six-cell finished length remains fixed by the untouched middle gauge.
    phase_change = example.copy()
    phase_change[2, 1:5], phase_change[3, :] = 5, 0
    phase_change[3, 3:5] = (3, 0)
    shifted = solve(phase_change)
    assert tuple(shifted[2, 1:7]) == (3, 1, 3, 2, 3, 2)
    assert np.array_equal(shifted[6], answer[6]) and np.array_equal(shifted[10], answer[10])
    signatures, features = set(), {'train': [], 'test': []}
    rng = np.random.default_rng(19427)
    last = None
    for seed in SELF_TEST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in p[split]:
                d = _decode(pair['input'])
                features[split].append(_features(d))
                for k in ('input', 'output'):
                    digest.update(bytes(pair[k].shape))
                    digest.update(pair[k].tobytes())
                assert np.array_equal(solve(pair['input']), pair['output'])
            # Colors are roles derived from geometry, including black payload.
            pair = p[split][0]
            perm = rng.permutation(10)
            assert np.array_equal(solve(perm[pair['input']]), perm[pair['output']])
            for turns in range(4):
                for mirror in (False, True):
                    assert np.array_equal(solve(_orient(pair['input'], turns, mirror)),
                                          _orient(pair['output'], turns, mirror))
        signatures.add(digest.hexdigest())
        if seed == 0:
            again = generate_puzzle(seed)
            assert all(_key(a[k]) == _key(b[k]) for s in p
                       for a, b in zip(p[s], again[s]) for k in ('input', 'output'))
        last = p
    assert len(signatures) == 42
    for split, fs in features.items():
        assert {f['count'] for f in fs} == {3, 5, 7}
        assert len({f['reference'] for f in fs}) >= 7
        assert len({f['shape'] for f in fs}) >= 15
        assert len({f['direction'] for f in fs}) >= 6
        assert len({(f['count'], f['reference'], f['gaps'], f['lengths'], f['periods']) for f in fs}) >= 35
        assert len({f['background'] for f in fs}) >= 7
        assert any(f['empty'] for f in fs)
        assert len({m for f in fs for m in f['masks']}) >= 12
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(27000+10*a+b, a, b))
    for args in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.5, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, *args)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal explicit counts accepted')
    # Preserve neither normalization nor phase as hidden generation metadata:
    # change just a visible carrier, then just an opaque stencil cell.
    d = _decode(last['train'][0]['input'])
    source = d['grid']
    for role in ('carrier', 'stencil'):
        changed = False
        for x in d['ribbons']:
            for i, color in enumerate(x[role]):
                if role == 'stencil' and color == d['background']:
                    continue
                trial = source.copy()
                r = x['row'] - 1 if role == 'carrier' else x['row'] + 1
                c = d['anchor'] + i if role == 'carrier' else d['anchor']+x['length']-x['n']+i
                trial[r, c] = next(c0 for c0 in range(10) if c0 not in (int(color), d['background'], d['marker']))
                oriented = _undo(trial, d['turns'], d['mirror'])
                if not np.array_equal(solve(oriented), solve(last['train'][0]['input'])):
                    changed = True
                    break
            if changed:
                break
        assert changed, f'no visible dependence on {role}'
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in last[s]] for s in last}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return {'seeds': 42, 'distinct_puzzles': len(signatures),
            'split_pairs': {s: len(fs) for s, fs in features.items()},
            'checks': 'material invariants, eleven alternatives, legal counts, D4, palette, mutation'}


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
