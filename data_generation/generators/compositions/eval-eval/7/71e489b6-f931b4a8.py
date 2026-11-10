"""Torn periodic screen: asymmetric defects open windows into a scaled swatch.

No parent code, stored examples, or generation state is used by solve().
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def _normalized(a):
    roles = {}
    return a.shape, bytes(roles.setdefault(int(x), len(roles)) for x in a.flat)


def _integer(value, low, high, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _halo(mask, diagonal=True, wrap=False):
    h, w = mask.shape
    result = np.zeros_like(mask, dtype=bool)
    padded = np.pad(mask, 1)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if not (dy or dx) or (not diagonal and abs(dy)+abs(dx) != 1):
                continue
            if wrap:
                result |= np.roll(mask, (dy, dx), axis=(0, 1))
            else:
                result |= padded[1+dy:1+dy+h, 1+dx:1+dx+w]
    return result & ~mask


def _decode(input_grid):
    """Parse roles and infer the stencil using only visible phase votes."""
    a = np.asarray(input_grid)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= x <= 30 for x in a.shape)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("invalid ARC grid")
    marker = int(a[0, 0])
    dividers = np.flatnonzero(np.all(a == marker, axis=0))
    if len(dividers) != 1:
        raise ValueError("one full-height frame-colored divider is required")
    divider = int(dividers[0])
    n = divider - 2
    h, w = a.shape[0], a.shape[1] - divider - 1
    if not 2 <= n <= 5 or not (3*n <= h <= 30 and 3*n <= w <= 30-n-3):
        raise ValueError("swatch or screen dimensions outside this family")
    field = int(a[-1, 0])
    if field == marker:
        raise ValueError("blank field and frame must differ")
    service = a[:, :divider]
    expected_service = np.full(service.shape, field, dtype=int)
    expected_service[:n+2, :n+2] = marker
    carrier = a[1:n+1, 1:n+1].copy()
    expected_service[1:n+1, 1:n+1] = carrier
    if not np.array_equal(service, expected_service):
        raise ValueError("malformed swatch frame or service margin")
    screen = a[:, divider+1:].copy()
    colors = set(map(int, np.unique(screen)))
    if field not in colors or len(colors) != 2 or marker in colors:
        raise ValueError("screen must use field plus one opaque ink")
    solid = next(iter(colors - {field}))
    if np.any(np.isin(carrier, [field, solid, marker])):
        raise ValueError("swatch inks must differ from the three structural roles")
    if not np.all(carrier == carrier[:1]) and h % n:
        raise ValueError("varying swatch rows require integral enlargement")
    if not np.all(carrier == carrier[:, :1]) and w % n:
        raise ValueError("varying swatch columns require integral enlargement")
    observed = screen == solid
    motif = np.zeros((n, n), dtype=bool)
    for r in range(n):
        for c in range(n):
            votes = observed[r::n, c::n]
            ink_count = int(votes.sum())
            if 2*ink_count == votes.size:
                raise ValueError("tied phase votes are outside the family")
            motif[r, c] = 2*ink_count > votes.size
    if not motif.any() or motif.all():
        raise ValueError("the intact stencil must contain both opacity roles")
    rr, cc = np.indices((h, w))
    clean = motif[rr % n, cc % n]
    missing = clean & ~observed
    extra = ~clean & observed
    backing = carrier[rr*n//h, cc*n//w]
    return dict(n=n, h=h, w=w, marker=marker, field=field, solid=solid,
                carrier=carrier, screen=screen, motif=motif, clean=clean,
                missing=missing, extra=extra, backing=backing)


def _compose(backing, clean, missing, halo, solid, field):
    out = np.where(clean, solid, backing).astype(np.int64)
    out[halo] = backing[halo]
    out[missing] = field
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover periodic material, erase surplus, preserve and outline tears."""
    d = _decode(input_grid)
    return _compose(d['backing'], d['clean'], d['missing'],
                    _halo(d['missing']), d['solid'], d['field'])


def _orient(a, rng):
    a = np.rot90(a, int(rng.integers(4)))
    if rng.random() < .5:
        a = np.fliplr(a)
    return a.copy()


def _swatch(n, payload, rng):
    k = int(rng.integers(1, min(4, len(payload), n*n)+1))
    inks = rng.choice(payload, k, replace=False)
    rr, cc = np.indices((n, n))
    kind = int(rng.integers(7))
    if k == 1:
        labels = np.zeros((n, n), dtype=int)
    elif kind == 0:
        # Unequal runs as well as single-cell stripes.
        line = rng.integers(k, size=n)
        line[0], line[-1] = 0, 1
        labels = np.tile(line, (n, 1))
    elif kind == 1:
        labels = (rr//int(rng.integers(1, n)) + cc//int(rng.integers(1, n))) % k
    elif kind == 2:
        cuts = sorted(rng.choice(np.arange(1, n), min(k-1, n-1), replace=False))
        labels = np.searchsorted(cuts, rr, side='right') % k
        labels = (labels + np.searchsorted(cuts, cc, side='right')) % k
    elif kind == 3:
        centers = rng.choice(n*n, k, replace=False)
        distance = abs(rr[..., None]-centers//n) + abs(cc[..., None]-centers % n)
        labels = np.argmin(distance, axis=2)
    elif kind == 4:
        labels = (rr + int(rng.integers(1, k+1))*cc) % k
    elif kind == 5:
        labels = np.zeros((n, n), dtype=int)
        for ink in range(1, k):
            r, c = int(rng.integers(n)), int(rng.integers(n))
            labels[r:, c:] = ink
    else:
        labels = rng.integers(k, size=(n, n))
    return _orient(inks[labels].astype(np.int64), rng)


def _motif(n, rng):
    rr, cc = np.indices((n, n))
    kind = int(rng.integers(7))
    if kind == 0:
        density = rng.uniform(.22, .82)
        mask = rng.random((n, n)) < density
    elif kind == 1:
        mask = np.zeros((n, n), dtype=bool)
        r, c = int(rng.integers(n)), int(rng.integers(n))
        v, u = int(rng.integers(r+1, n+1)), int(rng.integers(c+1, n+1))
        mask[r:v, c:u] = True
    elif kind == 2:
        mask = np.minimum.reduce([rr, cc, n-1-rr, n-1-cc]) < int(rng.integers(1, 3))
    elif kind == 3:
        mask = (rr + cc + int(rng.integers(2))) % 2 == 0
    elif kind == 4:
        mask = (rr == int(rng.integers(n))) | (cc == int(rng.integers(n)))
    elif kind == 5:
        line = rng.random(n) < rng.uniform(.25, .8)
        mask = np.tile(line, (n, 1))
    else:
        mask = np.ones((n, n), dtype=bool)
        r, c = int(rng.integers(n)), int(rng.integers(n))
        mask[r:, c:] = False
    if not mask.any():
        mask.flat[int(rng.integers(n*n))] = True
    if mask.all():
        mask.flat[int(rng.integers(n*n))] = False
    if rng.random() < .3:
        mask = ~mask
    return _orient(mask, rng)


_DEFECT_MOTIFS = (
    ((0, 0),), ((0, 0), (0, 1)), ((0, 0), (1, 1)),
    ((0, 0), (0, 2)), ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (1, 1), (2, 2)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
)


def _add_defect(rng, available, selected, phase_load, phase_limit, n, motif, edge=False):
    points = np.asarray(motif, dtype=int).copy()
    if rng.random() < .5:
        points = points[:, ::-1]
    points *= rng.choice([-1, 1], size=2)
    points -= points.min(axis=0)
    h, w = available.shape
    mh, mw = points.max(axis=0)+1
    candidates = []
    for r in range(h-int(mh)+1):
        for c in range(w-int(mw)+1):
            q = points + (r, c)
            if not np.all(available[q[:, 0], q[:, 1]] & ~selected[q[:, 0], q[:, 1]]):
                continue
            if edge and not np.any((q[:, 0] == 0) | (q[:, 0] == h-1)
                                   | (q[:, 1] == 0) | (q[:, 1] == w-1)):
                continue
            additions = np.zeros((n, n), dtype=int)
            np.add.at(additions, (q[:, 0] % n, q[:, 1] % n), 1)
            if np.all(phase_load + additions <= phase_limit):
                candidates.append((r, c))
    if not candidates:
        return False
    r, c = candidates[int(rng.integers(len(candidates)))]
    q = points + (r, c)
    selected[q[:, 0], q[:, 1]] = True
    np.add.at(phase_load, (q[:, 0] % n, q[:, 1] % n), 1)
    return True


def _construct(carrier, motif, h, w, missing, extra, roles):
    """Label renderer uses construction masks and explicit cell operations."""
    field, solid, marker = roles
    n = carrier.shape[0]
    screen = np.empty((h, w), dtype=np.int64)
    output = np.empty((h, w), dtype=np.int64)
    for r in range(h):
        for c in range(w):
            opaque = bool(motif[r % n, c % n])
            screen[r, c] = solid if opaque else field
            output[r, c] = solid if opaque else carrier[r*n//h, c*n//w]
    screen[missing], screen[extra] = field, solid
    for r, c in np.argwhere(missing):
        for y in range(max(0, int(r)-1), min(h, int(r)+2)):
            for x in range(max(0, int(c)-1), min(w, int(c)+2)):
                output[y, x] = carrier[y*n//h, x*n//w]
    output[missing] = field
    inp = np.full((h, n+3+w), field, dtype=np.int64)
    inp[:n+2, :n+2] = marker
    inp[1:n+1, 1:n+1] = carrier
    inp[:, n+2] = marker
    inp[:, n+3:] = screen
    return {'input': inp, 'output': output}


def _sample_pair(rng, roles, payload, sizes, split, regime, teaching=False):
    n = int(rng.choice(sizes))
    carrier = _swatch(n, payload, rng)
    vertical = not np.all(carrier == carrier[:1])
    horizontal = not np.all(carrier == carrier[:, :1])
    if regime == 0 and split == 'test':
        # Guarantee an observable new vertical enlargement, not a solid swatch.
        for _ in range(64):
            if vertical:
                break
            carrier = _swatch(n, payload, rng)
            vertical = not np.all(carrier == carrier[:1])
        else:
            carrier = np.full((n, n), int(payload[0]), dtype=np.int64)
            carrier[n//2:] = int(payload[1])
            vertical = True
        horizontal = not np.all(carrier == carrier[:, :1])
    row_factors = [5, 6] if regime == 0 and split == 'test' else [3, 4]
    if regime != 0:
        row_factors = list(range(3, min(6, 30//n)+1))
    h = n*int(rng.choice(row_factors))
    cap_w = 30-n-3
    w = n*int(rng.integers(3, cap_w//n+1))
    if not vertical and rng.random() < .5:
        high = min(30, 4*n) if regime == 0 else 30
        h = int(rng.integers(3*n, high+1))
    if not horizontal and rng.random() < .5:
        w = int(rng.integers(3*n, cap_w+1))
    rr, cc = np.indices((h, w))
    observations = np.zeros((n, n), dtype=int)
    np.add.at(observations, (rr.ravel() % n, cc.ravel() % n), 1)
    limit = (observations-1)//2
    motif = _motif(n, rng)
    if regime == 2 and split == 'test':
        # Count shift is guaranteed by visible defects, even for a sparse n=2 motif.
        for _ in range(64):
            if int(limit[motif].sum()) >= 7:
                break
            motif = _motif(n, rng)
        else:
            motif[:] = True
            motif.flat[int(rng.integers(n*n))] = False
    clean = motif[rr % n, cc % n]
    missing, extra = np.zeros((h, w), dtype=bool), np.zeros((h, w), dtype=bool)
    load = np.zeros((n, n), dtype=int)
    max_holes = min(14, max(3, int(clean.sum())//9))
    if regime == 2 and split == 'test':
        max_holes = min(18, int(limit[motif].sum()), max(7, int(clean.sum())//7))
        wanted = int(rng.integers(7, max_holes+1))
    else:
        if regime == 2:
            max_holes = min(6, max_holes)
        wanted = int(rng.integers(1, max_holes+1))
    if not teaching and not (regime == 2 and split == 'test') and rng.random() < .12:
        wanted = 0
    if wanted and (teaching or rng.random() < .4):
        _add_defect(rng, clean, missing, load, limit, n, _DEFECT_MOTIFS[0], edge=True)
    for _ in range(18):
        left = wanted-int(missing.sum())
        if left <= 0:
            break
        support = _DEFECT_MOTIFS if regime == 2 and split == 'test' else _DEFECT_MOTIFS[:6]
        choices = [m for m in support if len(m) <= left]
        motif_holes = choices[int(rng.integers(len(choices)))]
        if not _add_defect(rng, clean, missing, load, limit, n, motif_holes):
            _add_defect(rng, clean, missing, load, limit, n, _DEFECT_MOTIFS[0])
    if regime == 2 and split == 'test' and missing.sum() < 7:
        raise RuntimeError('failed constructive defect-count shift')
    max_extras = min(7, max(1, int((~clean).sum())//12))
    extra_count = int(rng.integers(1 if teaching or not missing.any() else 0, max_extras+1))
    for _ in range(extra_count):
        m = _DEFECT_MOTIFS[int(rng.choice([0, 0, 0, 1, 2]))]
        if len(m) > extra_count-int(extra.sum()):
            m = _DEFECT_MOTIFS[0]
        if extra.sum() >= extra_count:
            break
        if not _add_defect(rng, ~clean, extra, load, limit, n, m):
            _add_defect(rng, ~clean, extra, load, limit, n, _DEFECT_MOTIFS[0])
    return _construct(carrier, motif, h, w, missing, extra, roles)


_REQUIRED_EVIDENCE = frozenset((
    'omit_A', 'tile_carrier', 'scaled_stencil', 'no_halo', 'four_neighbors',
    'fill_holes', 'retain_extras', 'constant_highlight', 'per_hole_color',
    'repeat_tears', 'wrap_halo', 'halo_includes_holes', 'opaque_field',
    'overlapping_halos', 'phase_rows', 'phase_columns',
    'multicolor_opaque_halo',
    'trust_first_tile',
))


def _evidence(pair):
    d = _decode(pair['input'])
    out = np.asarray(pair['output'])
    n, h, w = d['n'], d['h'], d['w']
    rr, cc = np.indices((h, w))
    holes, clean, backing = d['missing'], d['clean'], d['backing']
    halo = _halo(holes)
    evidence = set()
    def compare(name, alternative):
        if not np.array_equal(out, alternative):
            evidence.add(name)
    def render(b=backing, mask=clean, tears=holes, ring=halo):
        return _compose(b, mask, tears, ring, d['solid'], d['field'])
    compare('omit_A', np.where(clean, d['solid'], backing))
    compare('tile_carrier', render(b=d['carrier'][rr % n, cc % n]))
    compare('scaled_stencil', render(mask=d['motif'][rr*n//h, cc*n//w]))
    compare('no_halo', render(ring=np.zeros_like(halo)))
    compare('four_neighbors', render(ring=_halo(holes, diagonal=False)))
    filled = out.copy()
    filled[holes] = backing[holes]
    compare('fill_holes', filled)
    retained = out.copy()
    retained[d['extra'] & ~halo] = d['solid']
    compare('retain_extras', retained)
    constant = render()
    constant[halo] = int(d['carrier'][0, 0])
    compare('constant_highlight', constant)
    single_color = np.where(clean, d['solid'], backing)
    coverage = np.zeros((h, w), dtype=int)
    for r, c in np.argwhere(holes):
        center = np.zeros_like(holes)
        center[r, c] = True
        ring = _halo(center) & ~holes
        single_color[ring] = backing[r, c]
        coverage[ring] += 1
        if len(np.unique(backing[ring & clean])) >= 2:
            evidence.add('multicolor_opaque_halo')
    single_color[holes] = d['field']
    compare('per_hole_color', single_color)
    if np.any(coverage >= 2):
        evidence.add('overlapping_halos')
    damaged_phases = np.zeros((n, n), dtype=bool)
    damaged_phases[rr[holes] % n, cc[holes] % n] = True
    repeated = damaged_phases[rr % n, cc % n] & clean
    compare('repeat_tears', render(tears=repeated, ring=_halo(repeated)))
    compare('wrap_halo', render(ring=_halo(holes, wrap=True)))
    includes = out.copy()
    includes[holes] = backing[holes]
    compare('halo_includes_holes', includes)
    opaque = render(b=np.full_like(backing, d['field']))
    compare('opaque_field', opaque)
    compare('phase_rows', render(mask=d['motif'][(rr+1) % n, cc % n]))
    compare('phase_columns', render(mask=d['motif'][rr % n, (cc+1) % n]))
    first_motif = d['screen'][:n, :n] == d['solid']
    first_clean = first_motif[rr % n, cc % n]
    first_missing = first_clean & (d['screen'] == d['field'])
    compare('trust_first_tile', render(mask=first_clean, tears=first_missing,
                                      ring=_halo(first_missing)))
    return evidence


def _fallback_teaching(roles, payload, sizes):
    """Constructive witnesses for a rare exhausted random pool, same family rule."""
    result = []
    for n in (min(sizes), max(sizes)):
        rr, cc = np.indices((n, n))
        carrier = np.asarray(payload[:3])[(rr+2*cc) % 3]
        motif = np.ones((n, n), dtype=bool)
        motif[0, 1] = False
        h, w = 3*n, 4*n
        missing, extra = np.zeros((h, w), dtype=bool), np.zeros((h, w), dtype=bool)
        missing[0, 0] = missing[n, n] = missing[n, n+2] = True
        extra[h-n, w-n+1] = True
        result.append(_construct(carrier, motif, h, w, missing, extra, roles))
    return result


def _distinct_fallback(pair, seen):
    """At least 36 two-speck layouts compete with at most 13 used inputs."""
    d = _decode(pair['input'])
    r, c = np.argwhere(~d['motif'])[0]
    points = [(y, x) for y in range(int(r), d['h'], d['n'])
              for x in range(int(c), d['w'], d['n'])]
    for first, second in combinations(points, 2):
        extra = np.zeros((d['h'], d['w']), dtype=bool)
        extra[first] = extra[second] = True
        candidate = _construct(d['carrier'], d['motif'], d['h'], d['w'],
                               d['missing'], extra, (d['field'], d['solid'], d['marker']))
        if _key(candidate['input']) not in seen:
            return candidate
    raise AssertionError('two-speck uniqueness bound was violated')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    train_count = _integer(num_train, 2, 10, 'num_train')
    test_count = _integer(num_test, 1, 4, 'num_test')
    if train_count <= test_count:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    palette = rng.permutation(10)
    roles = tuple(map(int, palette[:3]))
    payload = palette[3:]
    regime = int(rng.integers(3))
    held_size = int(rng.integers(2, 6))
    train_sizes = [n for n in range(2, 6) if regime != 1 or n != held_size]
    test_sizes = [held_size] if regime == 1 else list(range(2, 6))
    candidates, covers = [], []
    train = None
    for _ in range(96):
        pair = _sample_pair(rng, roles, payload, train_sizes, 'train', regime, teaching=True)
        evidence = _evidence(pair)
        for j in rng.permutation(len(candidates)):
            other = candidates[int(j)]
            a, b = _decode(other['input']), _decode(pair['input'])
            if (a['n'] != b['n'] and (a['h'], a['w']) != (b['h'], b['w'])
                    and _normalized(a['carrier']) != _normalized(b['carrier'])
                    and _REQUIRED_EVIDENCE <= (covers[int(j)] | evidence)):
                train = [other, pair]
                break
        if train is not None:
            break
        candidates.append(pair)
        covers.append(evidence)
    if train is None:
        train = _fallback_teaching(roles, payload, train_sizes)
    puzzle = {'train': train, 'test': []}
    seen = {_key(p['input']) for p in train}
    for split, count, sizes in [('train', train_count, train_sizes), ('test', test_count, test_sizes)]:
        while len(puzzle[split]) < count:
            for _ in range(64):
                pair = _sample_pair(rng, roles, payload, sizes, split, regime)
                if _key(pair['input']) not in seen:
                    break
            else:
                pair = _distinct_fallback(pair, seen)
            puzzle[split].append(pair)
            seen.add(_key(pair['input']))
    for split in puzzle:
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check decoded semantics, local precedence, teaching evidence, and diversity."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, role_sets, training, testing = set(), set(), [], []
    evidence = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for value in pair.values():
                a = np.asarray(value)
                assert a.ndim == 2 and np.issubdtype(a.dtype, np.integer)
                assert all(1 <= side <= 30 for side in a.shape)
                assert np.all((a >= 0) & (a <= 9))
            inp, out = np.asarray(pair['input']), np.asarray(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            d = _decode(inp)
            role_sets.add((d['field'], d['solid'], d['marker']))
            assert out.shape == (d['h'], d['w'])
            assert d['missing'].any() or d['extra'].any(), 'undamaged example'
            halo = _halo(d['missing'])
            exposed = (~d['clean'] | halo) & ~d['missing']
            intact = d['clean'] & ~halo & ~d['missing']
            assert np.all(out[d['missing']] == d['field']), 'hole centers must survive'
            assert np.count_nonzero(out == d['field']) == int(d['missing'].sum())
            assert np.array_equal(out[exposed], d['backing'][exposed]), 'incorrect exposed backing'
            assert np.all(out[intact] == d['solid']), 'intact opaque material changed'
            assert not np.any(out == d['marker']), 'frame leaked into answer'
            assert np.array_equal(solve(inp), out)
            # All phase classes have at least nine observations and a strict vote.
            for r in range(d['n']):
                for c in range(d['n']):
                    block = (d['screen'] == d['solid'])[r::d['n'], c::d['n']]
                    errors = np.count_nonzero(block != d['motif'][r, c])
                    assert block.size >= 9 and 2*errors < block.size
            if split == 'train':
                training.append(d)
                evidence |= _evidence(pair)
            else:
                testing.append(d)
    assert len(role_sets) == 1, 'structural colors must be consistent within a puzzle'
    assert len({d['n'] for d in training}) >= 2
    assert len({(d['h'], d['w']) for d in training}) >= 2
    assert len({_normalized(d['carrier']) for d in training}) >= 2
    assert _REQUIRED_EVIDENCE <= evidence, 'missing teaching evidence: ' + str(_REQUIRED_EVIDENCE-evidence)
    scale_shift = (max(d['h']/d['n'] for d in training) <= 4
                   and min(d['h']/d['n'] for d in testing) >= 5)
    size_shift = not ({d['n'] for d in training} & {d['n'] for d in testing})
    count_shift = (max(int(d['missing'].sum()) for d in training)
                   < min(int(d['missing'].sum()) for d in testing))
    assert scale_shift or size_shift or count_shift, 'no realized train/test shift'


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(p[k], q[k]) for p, q in zip(a[s], b[s]) for k in p)
        for s in ('train', 'test'))


def self_test():
    """Bounded 42-seed structural and semantic checks, plus targeted boundary probes."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records = [_decode(p['input']) for task in puzzles for s in task for p in task[s]]
    assert len({_key(task['test'][0]['input']) for task in puzzles}) == 42
    assert len({_normalized(task['test'][0]['output']) for task in puzzles}) >= 38
    assert {d['n'] for d in records} == {2, 3, 4, 5}
    assert len({(d['h'], d['w']) for d in records}) >= 35
    assert len({_key(d['motif']) for d in records}) >= 60
    assert len({_normalized(d['carrier']) for d in records}) >= 65
    assert len({int(d['missing'].sum()) for d in records}) >= 10
    assert len({int(d['extra'].sum()) for d in records}) >= 6
    assert len({(d['field'], d['solid'], d['marker']) for d in records}) >= 35
    assert any(d['field'] != 0 and np.any(d['carrier'] == 0) for d in records)
    assert any(d['h'] % d['n'] or d['w'] % d['n'] for d in records)
    assert any(not d['missing'].any() for d in records)
    assert any(not d['extra'].any() for d in records)
    assert _same(generate_puzzle(87123, 7, 3), generate_puzzle(87123, 7, 3))
    # Frozen, hand-worked n=2 example. This expectation does not call the renderer.
    def digits(rows):
        return np.array([[int(c) for c in row] for row in rows.split('/')])
    fixture = digits('99999011111/92399101010/94599110111/99999101010/00009111111/00009101011')
    expected = digits('021111/222313/120311/144515/111111/141515')
    assert np.array_equal(solve(fixture), expected)
    for held in range(2, 6):
        fallback = _fallback_teaching((0, 1, 9), np.array([2, 3, 4, 5, 6, 7, 8]),
                                      [n for n in range(2, 6) if n != held])
        assert _REQUIRED_EVIDENCE <= set().union(*(_evidence(p) for p in fallback))
        for p in fallback:
            assert np.array_equal(solve(p['input']), p['output'])
    # Force the finite uniqueness fallback beyond the maximum used-input count.
    used = {_key(fixture)}
    base_pair = {'input': fixture, 'output': expected}
    for _ in range(14):
        candidate = _distinct_fallback(base_pair, used)
        assert _key(candidate['input']) not in used
        used.add(_key(candidate['input']))
        assert np.array_equal(solve(candidate['input']), candidate['output'])
    # Every supported explicit count, and the smallest teaching set on new seeds.
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            task = generate_puzzle(1000+a*10+b, a, b)
            assert (len(task['train']), len(task['test'])) == (a, b)
    for seed in range(8):
        validate_generated_puzzle(generate_puzzle(8000+seed, 2, 1))
    for a, b in [(1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.0, 1), (4, None)]:
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Halo morphology: corner clipping, diagonal adjacency, and hole exclusion.
    holes = np.zeros((5, 5), dtype=bool)
    holes[0, 0] = holes[1, 1] = True
    ring = _halo(holes)
    assert not ring[0, 0] and not ring[1, 1] and ring[2, 2]
    assert not ring[-1, -1] and _halo(holes, wrap=True)[-1, -1]
    # Color permutation is a symmetry, including black in any role.
    perm = np.array([7, 9, 4, 1, 0, 8, 2, 6, 3, 5])
    for pair in puzzles[0]['train'] + puzzles[0]['test']:
        assert np.array_equal(solve(perm[pair['input']]), perm[pair['output']])
    # Reject a held label mutation and malformed ambiguous phase evidence.
    tied = _construct(np.array([[2, 3], [4, 5]]), np.array([[True, True], [True, False]]),
                      6, 8, np.zeros((6, 8), dtype=bool), np.zeros((6, 8), dtype=bool),
                      (0, 1, 9))['input']
    screen = tied[:, 5:]
    phase = screen[::2, ::2]
    phase[:1] = 0
    phase[1, :2] = 0  # Six of twelve phase observations are now blank.
    try:
        solve(tied)
    except ValueError:
        pass
    else:
        raise AssertionError('tied phase votes accepted')
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]]
               for s in ('train', 'test')}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted label accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42-seed variation and family invariants')
