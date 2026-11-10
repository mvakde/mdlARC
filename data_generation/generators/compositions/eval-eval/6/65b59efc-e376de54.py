"""Cyclic loom: middle-lane span and phase-corrected keyed motif weaving.

Only Python's standard library and NumPy are used. No parent code or data is read.
"""
from __future__ import annotations

import numpy as np


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _transform(a, code):
    if code >= 4:
        a = np.fliplr(a)
    return np.rot90(a, code % 4).copy()


def _read_core(core, n, bg):
    """Canonical reading: keyed cards above a left-aligned loom frame."""
    sep = int(core[0, 0])
    assert sep != bg
    top = np.flatnonzero(core[0] != sep)
    lib_width = int(top[0]) if len(top) else core.shape[1]
    assert (lib_width - 1) % (n + 1) == 0
    k = (lib_width - 1) // (n + 1)
    assert 2 <= k <= min(8, 29 // (n + 1))
    assert core.shape[0] >= n + 12
    assert np.all(core[:n + 4, lib_width:] == bg)
    for r in (0, n + 1, n + 3):
        assert np.all(core[r, :lib_width] == sep)
    assert np.all(core[n + 4] == bg)
    for c in range(0, lib_width, n + 1):
        assert np.all(core[:n + 4, c] == sep)

    order, masks, targets = [], {}, {}
    for j in range(k):
        c = 1 + j * (n + 1)
        tile = core[1:n + 1, c:c + n]
        colors = np.unique(tile[tile != bg])
        assert len(colors) == 1
        label = int(colors[0])
        assert label != sep and label not in order
        order.append(label)
        masks[label] = tile == label
        swatch = core[n + 2, c:c + n]
        assert np.all(swatch == swatch[0]) and swatch[0] != bg
        targets[label] = int(swatch[0])

    start = n + 5
    line = core[start]
    ends = np.flatnonzero(line != sep)
    loom_width = int(ends[0]) if len(ends) else core.shape[1]
    h, w = core.shape[0] - start - 2, loom_width - 2
    assert h >= 5 and w >= 2 and n * h <= 30 and n * w <= 30
    assert core.shape[1] == max(lib_width, loom_width)
    assert np.all(core[start:, loom_width:] == bg)
    assert np.all(core[-1, :loom_width] == sep)
    assert np.all(core[start:, 0] == sep)
    assert np.all(core[start:, loom_width - 1] == sep)
    loom = core[start + 1:-1, 1:loom_width - 1].copy()
    assert set(map(int, np.unique(loom))) <= {bg, *order}
    rows = np.flatnonzero(np.any(loom != bg, axis=1)).tolist()
    assert len(rows) in (3, 5, 7)
    assert all(b - a >= 2 for a, b in zip(rows, rows[1:]))
    spans = [np.flatnonzero(loom[r] != bg).tolist() for r in rows]
    assert all(s == list(range(s[0], s[-1] + 1)) for s in spans)
    common_left = len({s[0] for s in spans}) == 1
    common_right = len({s[-1] for s in spans}) == 1
    assert common_left != common_right, "anchored side must be unique"
    direction = 1 if common_left else -1
    anchor = spans[0][0] if common_left else spans[0][-1]
    words = [tuple(int(loom[r, c]) for c in (s if common_left else s[::-1]))
             for r, s in zip(rows, spans)]
    reference = words[len(rows) // 2]
    period = len(set(reference))
    assert period >= 2
    cycle = reference[:period]
    assert len(set(cycle)) == period
    assert all(x == cycle[i % period] for i, x in enumerate(reference))
    assert all(word[0] in cycle for word in words)
    length = len(reference)
    assert 0 <= anchor + (length - 1) * direction < w
    return dict(n=n, k=k, bg=bg, sep=sep, order=tuple(order), masks=masks,
                targets=targets, loom=loom, rows=rows, words=words,
                direction=direction, anchor=anchor, cycle=cycle, length=length)


def _parse(input_grid):
    grid = _grid(input_grid)
    candidates = []
    for turns in range(4):
        oriented = np.rot90(grid, turns)
        for bg in map(int, np.unique(oriented)):
            rr, cc = np.nonzero(oriented != bg)
            if not len(rr):
                continue
            core = oriented[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
            for reflected in (False, True):
                reading = np.fliplr(core) if reflected else core
                for n in range(2, 6):
                    try:
                        p = _read_core(reading, n, bg)
                    except (AssertionError, IndexError):
                        continue
                    p.update(turns=turns, reflected=reflected)
                    candidates.append(p)
    assert candidates, 'no framed interpretation'
    # Equal-width frames permit equivalent mirrored readings. Both must give
    # the same answer in the input's axes; this is not an arbitrary tie break.
    answer = _expand(candidates[0], _macro(candidates[0]))
    assert all(np.array_equal(answer, _expand(p, _macro(p))) for p in candidates[1:]), 'ambiguous framed input'
    return candidates[0]


def _macro(p, *, length=None, cycle=None, mode='phase'):
    result = np.full(p['loom'].shape, p['bg'], dtype=np.int8)
    length = p['length'] if length is None else int(length)
    cycle = p['cycle'] if cycle is None else tuple(cycle)
    for r, word in zip(p['rows'], p['words']):
        phase = cycle.index(word[0]) if mode == 'phase' else 0
        for step in range(length):
            c = p['anchor'] + step * p['direction']
            assert 0 <= c < result.shape[1]
            result[r, c] = word[0] if mode == 'uniform' else cycle[(phase + step) % len(cycle)]
    return result


def _expand(p, macro, *, use_key=True, stamp_transform=0):
    n = p['n']
    out = np.full((n * macro.shape[0], n * macro.shape[1]), p['bg'], dtype=np.int8)
    for r, c in zip(*np.nonzero(macro != p['bg'])):
        label = int(macro[r, c])
        target = p['targets'][label] if use_key else label
        mask = _transform(p['masks'][label], stamp_transform)
        out[r * n:(r + 1) * n, c * n:(c + 1) * n] = np.where(mask, target, p['bg'])
    if p.get('reflected', False):
        out = np.fliplr(out)
    return np.rot90(out, -p.get('turns', 0)).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """All scale, palette, span, phase, and presentation data come from the input."""
    p = _parse(input_grid)
    return _expand(p, _macro(p))


def _motif(rng, n):
    style = int(rng.integers(9))
    rr, cc = np.indices((n, n))
    if style == 0:
        mask = np.ones((n, n), dtype=bool)
    elif style == 1:
        mask = (rr == 0) | (rr == n - 1) | (cc == 0) | (cc == n - 1)
    elif style == 2:
        mask = rr == cc
    elif style == 3:
        mask = (rr == int(rng.integers(n))) | (cc == int(rng.integers(n)))
    elif style == 4:
        mask = (rr + cc) % 2 == int(rng.integers(2))
    elif style == 5:
        mask = np.zeros((n, n), dtype=bool)
        a, b = sorted(map(int, rng.choice(n, 2, replace=False)))
        c, d = sorted(map(int, rng.choice(n, 2, replace=False)))
        mask[a:b + 1, c:d + 1] = True
    else:
        mask = rng.random((n, n)) < float(rng.uniform(.15, .85))
    if not mask.any():
        mask.flat[int(rng.integers(n * n))] = True
    return _transform(mask, int(rng.integers(8)))


def _sample_geometry(rng, n, count, witness):
    maximum = 30 // n
    if witness:
        gaps = [5, 2, 2, 2] if rng.integers(2) else [2, 2, 2, 5]
    else:
        gaps = [2] * (count - 1)
        spare = maximum - (sum(gaps) + 1)
        for _ in range(int(rng.integers(min(5, spare) + 1))):
            gaps[int(rng.integers(count - 1))] += 1
    used = sum(gaps) + 1
    padding = int(rng.integers(min(3, maximum - used) + 1))
    before = int(rng.integers(padding + 1))
    return [before] + [before + int(v) for v in np.cumsum(gaps)], used + padding


def _input(rng, n, order, masks, targets, loom, bg, sep):
    width = len(order) * (n + 1) + 1
    h, w = loom.shape
    core = np.full((n + h + 7, max(width, w + 2)), bg, dtype=np.int8)
    core[:n + 4, :width] = sep
    for j, label in enumerate(order):
        c = 1 + j * (n + 1)
        core[1:n + 1, c:c + n] = np.where(masks[label], label, bg)
        core[n + 2, c:c + n] = targets[label]
    start = n + 5
    core[start:, :w + 2] = sep
    core[start + 1:-1, 1:w + 1] = loom
    row_pad = int(rng.integers(min(4, 30 - core.shape[0]) + 1))
    col_pad = int(rng.integers(min(4, 30 - core.shape[1]) + 1))
    top, left = int(rng.integers(row_pad + 1)), int(rng.integers(col_pad + 1))
    return np.pad(core, ((top, row_pad - top), (left, col_pad - left)), constant_values=bg)


def _draw_pair(rng, n, k, count, length, period, right, presentation, kind):
    witness = kind == 'witness'
    bg, sep = map(int, rng.choice(10, 2, replace=False))
    labels = [int(x) for x in rng.choice([c for c in range(10) if c not in (bg, sep)], k, replace=False)]
    cycle = tuple(int(x) for x in rng.choice(labels, period, replace=False))
    # Informative libraries reverse the cycle and place any unused cards freely.
    # Thus following the physical library order cannot reconstruct the master.
    if kind in ('witness', 'rival'):
        order = list(cycle[::-1]) + [x for x in labels if x not in cycle]
        order = np.roll(order, int(rng.integers(k))).tolist()
    else:
        order = [int(x) for x in rng.permutation(labels)]
    masks = {x: _motif(rng, n) for x in labels}
    if kind == 'ordinary' and rng.random() < .22:
        a, b = map(int, rng.choice(labels, 2, replace=False))
        masks[b] = masks[a].copy()
    if kind in ('witness', 'rival'):
        for _ in range(64):
            values = [int(x) for x in rng.choice([c for c in range(10) if c != bg], k, replace=False)]
            if all(x != y for x, y in zip(labels, values)):
                break
        else:
            values = np.roll(labels, 1).tolist()
    else:
        values = [int(x) for x in rng.choice([c for c in range(10) if c != bg], k)]
    targets = dict(zip(labels, values))
    rows, height = _sample_geometry(rng, n, count, witness)
    capacity = min(30 // n, length + int(rng.integers(1, 7)))
    middle = count // 2
    lengths = [length] * count
    others = [i for i in range(count) if i != middle]
    if witness:
        for i in others:
            lengths[i] = int(rng.integers(1, length))
        lengths[others[0]] = length - 1
        lengths[others[-1]] = int(rng.integers(length + 1, capacity + 1))
    elif kind == 'rival':
        # A noncentral lane has the same span and a different valid cycle.
        lengths[others[0]] = length
        lengths[others[1]] = length - 1 if rng.integers(2) else int(rng.integers(length + 1, capacity + 1))
    else:
        mode = int(rng.integers(5))
        for i in others:
            if rng.random() < .25:
                continue
            short = mode == 0 or (mode == 2 and rng.integers(2))
            lengths[i] = int(rng.integers(1, length)) if short else int(rng.integers(length + 1, capacity + 1))
        if mode in (3, 4):
            lengths = [length] * count
            lengths[others[0]] = length - 1 if mode == 3 else capacity
        if len(set(lengths)) == 1:
            lengths[others[0]] = length - 1
    used_width = max(lengths)
    padding = int(rng.integers(min(3, 30 // n - used_width) + 1))
    width = used_width + padding
    before = int(rng.integers(padding + 1))
    anchor = width - 1 - before if right else before
    direction = -1 if right else 1
    loom = np.full((height, width), bg, dtype=np.int8)
    phases = [int(rng.integers(period)) for _ in range(count)]
    phases[middle] = 0
    if witness:
        phases[others[0]], phases[others[1]] = 1, period - 1
    for i, (r, span) in enumerate(zip(rows, lengths)):
        first = cycle[phases[i]]
        if i == middle:
            word = [cycle[t % period] for t in range(span)]
        elif kind == 'rival' and i == others[0]:
            rival = cycle[::-1]
            phase = rival.index(first)
            word = [rival[(phase + t) % period] for t in range(span)]
        else:
            mode = int(rng.integers(3))
            if mode == 0:
                word = [first] * span
            elif mode == 1:
                word = [cycle[(phases[i] + t) % period] for t in range(span)]
            else:
                word = [first] + [int(x) for x in rng.choice(labels, span - 1)]
            if witness and i == others[0]:
                word[1] = first  # Definite error in a retained position.
        for t, label in enumerate(word):
            loom[r, anchor + direction * t] = label

    # Generation uses per-identity Kronecker masks; solve uses block substitution.
    repaired = np.full(loom.shape, bg, dtype=np.int8)
    for i, r in enumerate(rows):
        for t in range(length):
            repaired[r, anchor + direction * t] = cycle[(phases[i] + t) % period]
    output = np.full((n * height, n * width), bg, dtype=np.int8)
    for label in labels:
        output[np.kron(repaired == label, masks[label])] = targets[label]
    inp = _input(rng, n, order, masks, targets, loom, bg, sep)
    return {'input': _transform(inp, presentation), 'output': _transform(output, presentation)}


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError('counts must be integers: train 2..10, test 1..4')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _evidence(p):
    true = _expand(p, _macro(p))
    def differs(macro, **kwargs):
        return not np.array_equal(true, _expand(p, macro, **kwargs))
    lengths = [len(w) for w in p['words']]
    distances = [abs(2 * r - p['rows'][0] - p['rows'][-1]) for r in p['rows']]
    near = [i for i, d in enumerate(distances) if d == min(distances)]
    full = p['words'][len(p['rows']) // 2]
    active_order = [x for x in p['order'] if x in p['cycle']]
    solid = dict(p, masks={x: np.ones_like(m) for x, m in p['masks'].items()})
    positional = dict(p,
        masks={x: p['masks'][p['order'][i]] for i, x in enumerate(p['cycle'])},
        targets={x: p['targets'][p['order'][i]] for i, x in enumerate(p['cycle'])})
    key_count_fits = 0 <= p['anchor'] + (p['k'] - 1) * p['direction'] < p['loom'].shape[1]
    return dict(
        no_B=differs(p['loom']), no_key=differs(_macro(p), use_key=False),
        no_motifs=not np.array_equal(true, _expand(solid, _macro(p))),
        positional_lookup=not np.array_equal(true, _expand(positional, _macro(p))),
        uniform=differs(_macro(p, mode='uniform')),
        no_phase=differs(_macro(p, mode='copy')),
        full_word=differs(_macro(p, cycle=full)),
        library_order=differs(_macro(p, cycle=active_order)),
        reverse_cycle=differs(_macro(p, cycle=p['cycle'][::-1])),
        key_count=key_count_fits and p['length'] != p['k'] and differs(_macro(p, length=p['k'])),
        period_length=differs(_macro(p, length=len(p['cycle']))),
        median=differs(_macro(p, length=sorted(lengths)[len(lengths) // 2])),
        geometric=len(near) == 1 and differs(_macro(p, length=lengths[near[0]])),
        extension=any(x < p['length'] for x in lengths),
        truncation=any(x > p['length'] for x in lengths),
        interior=any(p['loom'][r, p['anchor'] + t * p['direction']] != _macro(p)[r, p['anchor'] + t * p['direction']]
                     for r, word in zip(p['rows'], p['words']) for t in range(min(len(word), p['length']))),
        transforms=[differs(_macro(p), stamp_transform=t) for t in range(1, 8)])


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    shift = int(rng.integers(3))  # Library, stamp scale, or lane count transfer.
    presentation = int(rng.integers(8))
    first_k = int(rng.choice((3, 4)))
    first_length = first_k + int(rng.integers(1, 3))
    first_right = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, total in (('train', a), ('test', b)):
        for i in range(total):
            kind = 'witness' if split == 'train' and i == 0 else 'rival' if split == 'train' and i == 1 else 'ordinary'
            if kind == 'witness':
                n, k, count, length, period = 2, first_k, 5, first_length, first_k
            elif kind == 'rival':
                n = 3 if shift == 1 else int(rng.integers(2, 6))
                k, count = 7 - first_k, 3
                delta = first_length - first_k
                options = [(l, p) for p in range(3, k + 1) for l in range(p, 30 // n)
                           if l != first_length and l - k != delta and l - p != delta]
                length, period = options[int(rng.integers(len(options)))]
            else:
                if split == 'test' and shift == 0:
                    n = int(rng.choice((2, 3, 4)))
                    k = int(rng.integers(5, min(8, 29 // (n + 1)) + 1))
                    count = int(rng.choice((3, 5, 7) if n == 2 else (3, 5) if n == 3 else (3,)))
                elif split == 'test' and shift == 1:
                    n, count = int(rng.choice((4, 5))), 3
                    k = int(rng.integers(2, 29 // (n + 1) + 1))
                elif split == 'test' and shift == 2:
                    n, count = 2, 7
                    k = int(rng.integers(2, 9))
                else:
                    n = int(rng.choice((2, 3))) if shift == 1 else int(rng.integers(2, 6))
                    count = int(rng.choice((3, 5) if n <= 3 else (3,)))
                    k = int(rng.integers(2, min(4, 29 // (n + 1)) + 1))
                period = int(rng.integers(2, min(k, 30 // n - 1) + 1))
                length = int(rng.integers(period, 30 // n))
            right = first_right if kind == 'witness' else not first_right if kind == 'rival' else bool(rng.integers(2))
            for _ in range(128):
                pair = _draw_pair(rng, n, k, count, length, period, right, presentation, kind)
                p = _parse(pair['input'])
                if kind == 'witness':
                    e = _evidence(p)
                    if not all(e[x] for x in ('no_B', 'no_key', 'no_motifs', 'positional_lookup', 'uniform', 'no_phase', 'full_word', 'library_order', 'reverse_cycle', 'key_count', 'period_length', 'median', 'geometric', 'interior')) or not all(e['transforms']):
                        continue
                if _key(pair['input']) in seen:
                    continue
                seen.add(_key(pair['input']))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError('bounded informative-pair construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check returned-cell structure, keyed blocks, phase, and training evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training, tests = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            p = _parse(inp)
            macro = _macro(p)
            canonical = np.rot90(out, p['turns'])
            if p['reflected']:
                canonical = np.fliplr(canonical)
            n = p['n']
            assert canonical.shape == (n * macro.shape[0], n * macro.shape[1])
            # Inspect every tile, including erased tails and all motif holes.
            for r in range(macro.shape[0]):
                for c in range(macro.shape[1]):
                    label = int(macro[r, c])
                    block = canonical[n * r:n * (r + 1), n * c:n * (c + 1)]
                    if label == p['bg']:
                        assert np.all(block == p['bg'])
                    else:
                        mask = p['masks'][label]
                        assert np.all(block[mask] == p['targets'][label])
                        assert np.all(block[~mask] == p['bg'])
            assert set(map(int, np.unique(out))) <= {p['bg'], *(p['targets'][x] for x in p['cycle'])}
            for r, word in zip(p['rows'], p['words']):
                assert macro[r, p['anchor']] == word[0]
                assert np.count_nonzero(macro[r] != p['bg']) == p['length']
            central = len(p['rows']) // 2
            for t, x in enumerate(p['words'][central]):
                assert macro[p['rows'][central], p['anchor'] + t * p['direction']] == x
            if split == 'train':
                training.append(p)
            else:
                tests.append(p)
    evidence = [_evidence(p) for p in training]
    for name in ('no_B', 'no_key', 'no_motifs', 'positional_lookup', 'uniform', 'no_phase', 'full_word', 'library_order', 'reverse_cycle', 'key_count', 'period_length', 'median', 'geometric', 'interior', 'extension', 'truncation'):
        assert any(e[name] for e in evidence), f'missing training evidence: {name}'
    assert all(any(e['transforms'][t] for e in evidence) for t in range(7))
    assert len({p['length'] for p in training}) >= 2
    assert len({len(p['rows']) for p in training}) >= 2
    assert len({p['length'] - p['k'] for p in training}) >= 2
    assert len({p['length'] - len(p['cycle']) for p in training}) >= 2
    assert (max(p['k'] for p in training) < min(p['k'] for p in tests)
            or max(p['n'] for p in training) < min(p['n'] for p in tests)
            or max(len(p['rows']) for p in training) < min(len(p['rows']) for p in tests)), 'missing structural transfer'
    # A competing, complete cycle occurs away from the ordinal middle.
    assert any(any(len(w) == p['length'] and len(set(w)) == len(p['cycle'])
                   and any(w[t] != p['cycle'][(p['cycle'].index(w[0]) + t) % len(p['cycle'])] for t in range(len(w)))
                   for i, w in enumerate(p['words']) if i != len(p['rows']) // 2) for p in training)


def self_test():
    """Bounded 42-seed campaign, metamorphic checks, counts, and corruptions."""
    rng = np.random.default_rng(800777)
    records, signatures, motifs, plans, shapes = [], set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(pair['input']) for s in puzzle for pair in puzzle[s]))
        if seed < 3:
            again = generate_puzzle(seed)
            assert all(np.array_equal(puzzle[s][i][v], again[s][i][v]) for s in puzzle for i in range(len(puzzle[s])) for v in ('input', 'output'))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                p = _parse(pair['input'])
                assert np.array_equal(solve(pair['input']), pair['output'])
                records.append((split, p['n'], p['k'], len(p['rows']), p['length'], len(p['cycle']), p['direction'], p['turns'], p['bg']))
                motifs.update((p['n'], m.tobytes()) for m in p['masks'].values())
                normalized = np.full(p['loom'].shape, -1, dtype=np.int8)
                for j, x in enumerate(p['order']):
                    normalized[p['loom'] == x] = j
                plans.add(_key(normalized))
                shapes.add(pair['output'].shape)
            # Recoloring and whole-scene geometry must commute with solving.
            pair = puzzle[split][0]
            palette = rng.permutation(10).astype(np.int8)
            assert np.array_equal(solve(palette[pair['input']]), palette[pair['output']])
            for code in range(8):
                assert np.array_equal(solve(_transform(pair['input'], code)), _transform(pair['output'], code))
    assert len(signatures) == 42
    assert {r[1] for r in records} == {2, 3, 4, 5}
    assert {r[2] for r in records} >= {2, 3, 4, 5, 6, 7}
    assert {r[3] for r in records} == {3, 5, 7}
    assert len({r[4] for r in records}) >= 8
    assert len({r[5] for r in records}) >= 4
    assert {r[6] for r in records} == {-1, 1}
    assert {r[7] for r in records} == {0, 1, 2, 3}
    assert {r[8] for r in records} == set(range(10))
    assert len(motifs) >= 120 and len(plans) >= 150 and len(shapes) >= 40
    assert any(r[4] != r[2] and r[4] != r[5] for r in records if r[0] == 'test')
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(90210 + a * 10 + b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    puzzle = generate_puzzle(0)
    broken = {s: [{v: np.asarray(g).copy() for v, g in pair.items()} for pair in puzzle[s]] for s in puzzle}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    # Input-only counterfactuals: alter just the master order, then the key.
    p = _parse(puzzle['train'][0]['input'])
    master = list(p['cycle'])
    master[1], master[2] = master[2], master[1]
    changed_loom = p['loom'].copy()
    r = p['rows'][len(p['rows']) // 2]
    for t in range(p['length']):
        changed_loom[r, p['anchor'] + t * p['direction']] = master[t % len(master)]
    changed = _input(rng, p['n'], p['order'], p['masks'], p['targets'], changed_loom, p['bg'], p['sep'])
    original = np.rot90(puzzle['train'][0]['output'], p['turns'])
    if p['reflected']:
        original = np.fliplr(original)
    assert not np.array_equal(solve(changed), original)
    # Shortening the visible master leaves its period intact but changes span.
    changed_loom = p['loom'].copy()
    changed_loom[r, p['anchor'] + (p['length'] - 1) * p['direction']] = p['bg']
    changed = _input(rng, p['n'], p['order'], p['masks'], p['targets'], changed_loom, p['bg'], p['sep'])
    assert _parse(changed)['length'] == p['length'] - 1
    assert not np.array_equal(solve(changed), original)
    # Nonanchor data in another lane is corruption, not a second instruction.
    changed_loom = p['loom'].copy()
    r_other = next(row for row, word in zip(p['rows'], p['words']) if row != r and len(word) >= 2)
    c = p['anchor'] + p['direction']
    changed_loom[r_other, c] = next(x for x in p['order'] if x != changed_loom[r_other, c])
    changed = _input(rng, p['n'], p['order'], p['masks'], p['targets'], changed_loom, p['bg'], p['sep'])
    assert np.array_equal(solve(changed), original)
    changed = _input(rng, p['n'], p['order'][::-1], p['masks'], p['targets'], p['loom'], p['bg'], p['sep'])
    assert np.array_equal(solve(changed), original), 'card position leaked into identity lookup'
    changed_targets = dict(p['targets'])
    x = p['cycle'][0]
    changed_targets[x] = next(c for c in range(10) if c not in (p['bg'], p['targets'][x]))
    changed = _input(rng, p['n'], p['order'], p['masks'], changed_targets, p['loom'], p['bg'], p['sep'])
    assert not np.array_equal(solve(changed), original)
    return {'seeds': 42, 'pairs': len(records), 'motifs': len(motifs),
            'macro_plans': len(plans), 'output_shapes': len(shapes),
            'semantic_verification': 'generator-owned invariants and metamorphic checks'}


if __name__ == '__main__':
    print(self_test())
