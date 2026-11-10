"""Reflected losses: sort a window's clipped fragments into fixed slots.

Standard library and NumPy only. Solving never reads generation state or files.
Half-open intervals are used internally; None denotes an empty fragment.
"""
from __future__ import annotations

import copy
import json
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)),
             "grid colors must be integers 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _puzzle_key(p):
    return tuple(tuple((_key(np.asarray(x['input'])), _key(np.asarray(x['output'])))
                       for x in p[s]) for s in ('train', 'test'))


def _length(interval):
    return 0 if interval is None else interval[1] - interval[0]


def _intersection(top, end, window_top, window_end):
    lo, hi = max(top, window_top), min(end, window_end)
    return None if lo >= hi else (lo - window_top, hi - window_top)


def _check_ties(intervals):
    by_length = {}
    for span in intervals:
        length = _length(span)
        _require(length not in by_length or by_length[length] == span,
                 "equal fragment lengths must have equal coordinates")
        by_length[length] = span


def _analyze(input_grid):
    """Recover the complete scene and queried intervals solely from pixels."""
    a = _grid(input_grid)
    bg = int(a[0, 0])
    _require(all(int(a[r, c]) == bg for r, c in
                 ((0, 0), (0, -1), (-1, 0), (-1, -1))),
             "corners must identify background")
    rows = np.flatnonzero(np.all(a != bg, axis=1))
    cols = np.flatnonzero(np.all(a != bg, axis=0))
    _require(len(rows) + len(cols) == 1, "carrier must be unique")
    vertical = bool(len(cols))
    work = a.T.copy() if vertical else a.copy()
    row = int(cols[0] if vertical else rows[0])
    h, w = work.shape
    carrier = int(work[row, 0])
    _require(carrier != bg and int(work[row, -1]) == carrier,
             "carrier endpoints must agree")
    _require(np.all(np.nonzero(work == carrier)[0] == row),
             "carrier color occurs only on its line")
    candidates = []
    for color in np.unique(work):
        color = int(color)
        if color in (bg, carrier):
            continue
        rr, cc = np.nonzero(work == color)
        r, s = int(rr.min()), int(rr.max()) + 1
        c, d = int(cc.min()), int(cc.max()) + 1
        if d - c >= 5 and len(rr) == (s-r)*(d-c):
            candidates.append((color, (r, s, c, d)))
    _require(len(candidates) == 1, "mask must be uniquely rectangular and broad")
    mask, (r, s, c, d) = candidates[0]
    _require(1 <= c < d <= w-1, "mask must leave carrier endpoints visible")
    left = d <= w // 2
    right = c >= (w+1) // 2
    _require(left or right, "mask must remain in one reflection wing")
    donor = work[r:s, w-d:w-c][:, ::-1]
    _require(not np.any(donor == mask), "reflected donors must remain intact")
    full = work.copy()
    full[r:s, c:d] = donor
    _require(not np.any(full == mask), "mask color cannot be content")
    _require(np.array_equal(full, full[:, ::-1]), "completed field must reflect")
    line = full[row]
    rebuilt = np.full_like(full, bg)
    rebuilt[row] = carrier
    records = []
    x = 0
    while x < w:
        color = int(line[x])
        end = x + 1
        while end < w and int(line[end]) == color:
            end += 1
        if color != carrier:
            _require(color not in (bg, mask), "invalid slot color")
            _require(1 <= end-x <= 3, "slot width must be 1..3")
            occupied = np.flatnonzero(np.any(full[:, x:end] != bg, axis=1))
            top, stop = int(occupied[0]), int(occupied[-1]) + 1
            _require(top <= row < stop and np.all(full[top:stop, x:end] == color),
                     "bars must be solid rectangles crossing the carrier")
            rebuilt[top:stop, x:end] = color
            records.append((x, end, color, top, stop))
        x = end
    _require(np.array_equal(full, rebuilt), "unexplained scene cells")
    _require(4 <= len(records) <= 10 and len(records) % 2 == 0,
             "two wings with 2..5 slots each required")
    _require(len({v[2] for v in records}) >= 2, "at least two accent colors required")
    target = [v for v in records if c <= v[0] and v[1] <= d]
    _require(len(target) * 2 == len(records), "mask must cover exactly one wing")
    for start, end, color, top, stop in records:
        intersects = start < d and c < end
        _require(not intersects or (c <= start and end <= d),
                 "partial slot-width masking is unsupported")
        _require((w-end, w-start, color, top, stop) in records,
                 "every slot must have an intact reflected twin")
    n = len(target)
    _require(all(v[1] <= w//2 for v in records[:n]) and
             all(v[0] >= (w+1)//2 for v in records[n:]),
             "a central gap must separate the wings")
    _require(records[n][0] > records[n-1][1], "central gap must be nonempty")
    intervals = [_intersection(v[3], v[4], r, s) for v in target]
    _check_ties(intervals)
    _require(len({_length(v) for v in intervals}) >= 2,
             "query must have at least two fragment lengths")
    return {'vertical': vertical, 'background': bg, 'carrier': carrier,
            'mask': mask, 'row': row, 'box': (r, s, c, d), 'full': full,
            'target': target, 'intervals': intervals}


def _render_window(record, assigned):
    r, s, c, d = record['box']
    out = np.full((s-r, d-c), record['background'], dtype=np.int8)
    if r <= record['row'] < s:
        out[record['row']-r] = record['carrier']
    for slot, span in zip(record['target'], assigned):
        if span is not None:
            start, end, color = slot[:3]
            out[span[0]:span[1], start-c:end-c] = color
    return out.T.copy() if record['vertical'] else out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only reflection, clipping, and shortest-first interval selection."""
    record = _analyze(input_grid)
    remaining = record['intervals'].copy()
    assigned = []
    while remaining:
        shortest = min(remaining, key=_length)
        remaining.remove(shortest)
        assigned.append(shortest)
    return _render_window(record, assigned)


def _check_pair(pair):
    """Relational output check: no call to solve or its rendering helper."""
    a, out = _grid(pair['input']), _grid(pair['output'])
    record = _analyze(a)
    b = out.T if record['vertical'] else out
    r, s, c, d = record['box']
    _require(b.shape == (s-r, d-c), "output must be exactly the mask window")
    bg = record['background']
    expected_gaps = record['full'][r:s, c:d]
    gaps = np.ones(d-c, dtype=bool)
    remaining = record['intervals'].copy()
    previous, final = -1, []
    for start, end, color, _, _ in record['target']:
        x, y = start-c, end-c
        gaps[x:y] = False
        strip = b[:, x:y]
        occupied = np.flatnonzero(np.any(strip != bg, axis=1))
        span = None
        rebuilt = np.full_like(strip, bg)
        if len(occupied):
            lo, hi = int(occupied[0]), int(occupied[-1])+1
            span = (lo, hi)
            rebuilt[lo:hi] = color
        _require(np.array_equal(strip, rebuilt),
                 "destination slots must keep color, width, and solid fragments")
        length = _length(span)
        _require(length >= previous, "output fragment lengths must increase")
        _require(span in remaining, "clipped interval multiset must be conserved")
        remaining.remove(span)
        previous = length
        final.append(span)
    _require(not remaining, "fragment omitted")
    _require(np.array_equal(b[:, gaps], expected_gaps[:, gaps]),
             "carrier and background gaps must stay fixed")
    _require(not np.any(out == record['mask']), "mask cannot appear in answer")
    record['final'] = final
    return record


def _features(record):
    spans = record['intervals']
    final = record['final']
    slots = record['target']
    widths = [v[1]-v[0] for v in slots]
    lost = [_length(v) for v in spans]
    complete = [v[4]-v[3] for v in slots]
    visible = [length-fragment for length, fragment in zip(complete, lost)]
    def by(values):
        return [spans[i] for i in sorted(range(len(spans)), key=lambda i: values[i])]
    r, s, c, d = record['box']
    return {'count': len(slots), 'vertical': record['vertical'],
            'shape': tuple(reversed(record['full'].shape)) if record['vertical']
                     else record['full'].shape,
            'output_shape': (d-c, s-r) if record['vertical'] else (s-r, d-c),
            'role_colors': (record['background'], record['carrier'], record['mask']),
            'left': d <= record['full'].shape[1]//2,
            'crossing': r <= record['row'] < s,
            'widths': tuple(widths), 'lost': tuple(lost),
            'intervals': tuple(spans), 'changed': spans != final,
            'whole_disagrees': by(complete) != final,
            'visible_disagrees': by(visible) != final,
            'area_disagrees': by([a*b for a, b in zip(widths, lost)]) != final,
            'reversal_disagrees': spans[::-1] != final,
            'empty': 0 in lost, 'ties': len(set(lost)) < len(lost),
            'touching': any(b[0] == a[1] for a, b in zip(slots, slots[1:])),
            'repeat': len({v[2] for v in slots}) < len(slots),
            'offset_move': any(span is not None and span != old and
                               span[0]+span[1]-1 != 2*(record['row']-r)
                               for span, old in zip(final, spans)) if r <= record['row'] < s else False,
            'edge': r == 0 or s == record['full'].shape[0]}


def validate_generated_puzzle(puzzle):
    """Validate structure, semantic relations, and demonstration evidence."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'},
             "puzzle must have exactly train/test keys")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')),
             "splits must be lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _counts(nt, nq)
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'},
                     "pairs must have exactly input/output keys")
            key = _key(_grid(pair['input']))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            features[split].append(_features(_check_pair(pair)))
    train = features['train']
    _require(sum(f['changed'] for f in train) >= 2,
             "two training examples must distinguish repair from redistribution")
    _require(any(not f['crossing'] and f['whole_disagrees'] and
                 f['visible_disagrees'] and f['area_disagrees'] and
                 f['reversal_disagrees'] for f in train),
             "off-carrier witness must defeat whole/visible/area/reversal rules")
    _require(any(f['crossing'] and f['offset_move'] for f in train),
             "training must demonstrate biased interval transfer")
    _require(len({f['count'] for f in train}) >= 2, "training must vary slot count")
    axes = {f['vertical'] for f in train if f['changed']}
    _require(all(f['vertical'] in axes for f in features['test']),
             "test orientation must be demonstrated")
    # The two ablations are checked from actual cells, not sampled parameters.
    ordering_ablations = 0
    for pair in puzzle['train']:
        rec = _analyze(pair['input'])
        r, s, c, d = rec['box']
        faithful = rec['full'][r:s, c:d]
        if rec['vertical']:
            faithful = faithful.T
        ordering_ablations += not np.array_equal(faithful, pair['output'])
        _require(np.any(np.isin(pair['output'], [v[2] for v in rec['target']])),
                 "answer must contain reflected material; background ablation fails")
    _require(ordering_ablations >= 2, "ordering ablation must fail on two demonstrations")


def _counts(num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10),
                                ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f'{name} must be an integer')
        if not lo <= int(value) <= hi:
            raise ValueError(f'{name} must be between {lo} and {hi}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')


def _colors(rng, config, n):
    if config['fixed_roles']:
        bg, carrier, mask = config['roles']
    else:
        bg, carrier, mask = [int(v) for v in rng.choice(10, 3, replace=False)]
    pool = [v for v in range(10) if v not in (bg, carrier, mask)]
    preferred = [v for v in config['palette'] if v in pool]
    size = int(rng.integers(2, min(n, 5)+1))
    if len(preferred) >= size and rng.random() < .5:
        palette = preferred[:size]
    else:
        palette = [int(v) for v in rng.choice(pool, size, replace=False)]
    accents = palette + [int(v) for v in rng.choice(palette, n-size)]
    rng.shuffle(accents)
    return bg, carrier, mask, accents


def _geometry(rng, spans, window, widths, accents):
    """Sample a compact mirrored layout subject only to visible constraints."""
    n = len(spans)
    central = int(rng.integers(1, 5))
    cap = (30-central)//2
    margin = int(rng.integers(1, 4))
    inner = int(rng.integers(0, 3))
    gaps = [int(rng.integers(0, 3)) for _ in range(n-1)]
    for i in range(n-1):
        gaps[i] = max(gaps[i], int(accents[i] == accents[i+1]))
    # Witness widths always fit; shrink extra gaps/margins first.
    minimum_gaps = [int(accents[i] == accents[i+1]) for i in range(n-1)]
    while margin + sum(widths) + sum(gaps) + inner > cap:
        extra = [i for i in range(n-1) if gaps[i] > minimum_gaps[i]]
        if extra:
            gaps[extra[-1]] -= 1
        elif inner:
            inner -= 1
        elif margin > 1:
            margin -= 1
        else:
            options = [i for i in range(n) if widths[i] > 1]
            _require(bool(options), "layout does not fit")
            widths[options[-1]] -= 1
    if sum(widths)+sum(gaps) < 5:
        gaps[0] += 5-sum(widths)-sum(gaps)
    q = sum(widths)+sum(gaps)
    _require(margin+q+inner <= cap, "window does not fit")
    slack = int(rng.integers(0, cap-(margin+q+inner)+1))
    inner += slack
    half = margin+q+inner
    width = 2*half+central
    starts, x = [], margin
    for i, thickness in enumerate(widths):
        starts.append(x)
        x += thickness + (gaps[i] if i < n-1 else 0)
    c = margin-int(rng.integers(0, margin))
    d = margin+q+int(rng.integers(0, inner+1))
    low = min(-1, window[0], min(v[0] for v in spans))
    high = max(1, window[1]-1, max(v[1]-1 for v in spans))
    _require(high-low+1 <= 30, "perpendicular spans do not fit")
    budget = 30-(high-low+1)
    before = int(rng.integers(0, min(3, budget)+1))
    after = int(rng.integers(0, min(3, budget-before)+1))
    row = before-low
    height = high-low+1+before+after
    return height, width, row, starts, (row+window[0], row+window[1], c, d)


def _ordinary_spans(rng, n):
    spans = []
    for _ in range(n):
        length = int(rng.integers(1, 17))
        up = int(rng.integers(0, length))
        spans.append((-up, length-up))
    if rng.random() < .28 and n > 2:
        spans[-1] = spans[0]
    lo, hi = min(v[0] for v in spans), max(v[1] for v in spans)
    if rng.random() < .45:
        r = int(rng.integers(lo-2, 1))
        s = int(rng.integers(1, hi+3))
    elif rng.random() < .5:
        r = int(rng.integers(min(lo, -2)-2, -1))
        s = int(rng.integers(r+1, 1))
    else:
        s = int(rng.integers(max(hi, 2)+1, max(hi, 2)+4))
        r = int(rng.integers(1, s))
    return spans, (r, s)


def _make_pair(rng, config, left, vertical, witness=0, test=False):
    for _ in range(250):
        if witness == 1:
            k = int(rng.integers(2, 5))
            losses = [k+2, k, k+1]
            spans = [(-losses[0]-1, int(rng.integers(0, 3))+1),
                     (-losses[1]-1, int(rng.integers(k+6, k+10))+1),
                     (-losses[2]-1, int(rng.integers(0, 2))+1), (0, 1)]
            window = (-k-4, -1)
            widths = [1, 3, 1, int(rng.integers(1, 3))]
            if rng.random() < .5:
                spans = [(1-hi, 1-lo) for lo, hi in spans]
                window = (1-window[1], 1-window[0])
        elif witness == 2:
            e = int(rng.integers(2, 5))
            spans = [(1-e, e+1), (-e-1, e+1), (0, 1)]
            window = (-e-1-int(rng.integers(0, 3)), e+1+int(rng.integers(0, 3)))
            widths = [int(v) for v in rng.integers(1, 4, size=3)]
        else:
            n = 5 if test and rng.random() < .6 else int(rng.integers(2, 6))
            spans, window = _ordinary_spans(rng, n)
            widths = [int(v) for v in rng.integers(1, 4, size=n)]
            if rng.random() < .2:
                widths = [int(rng.integers(1, 4))] * n
        n = len(spans)
        # Coordinates below are always the target wing's physical order. Reflect
        # their layout when querying the right wing, but never reverse the rule.
        clipped = [_intersection(lo, hi, *window) for lo, hi in spans]
        try:
            _check_ties(clipped)
            _require(len({_length(v) for v in clipped}) >= 2, "uniform losses")
            bg, carrier, mask, accents = _colors(rng, config, n)
            layout_accents = accents if left else accents[::-1]
            layout_widths = widths.copy() if left else widths[::-1].copy()
            layout_spans = spans if left else spans[::-1]
            h, w, row, starts, box = _geometry(rng, layout_spans, window,
                                               layout_widths, layout_accents)
            full = np.full((h, w), bg, dtype=np.int8)
            full[row] = carrier
            left_records = []
            for x, thickness, color, (lo, hi) in zip(starts, layout_widths,
                                                    layout_accents, layout_spans):
                full[row+lo:row+hi, x:x+thickness] = color
                full[row+lo:row+hi, w-x-thickness:w-x] = color
                left_records.append((x, x+thickness, color, row+lo, row+hi))
            r, s, c, d = box
            if not left:
                c, d = w-d, w-c
                box = (r, s, c, d)
                target = [(w-end, w-start, color, top, stop)
                          for start, end, color, top, stop in left_records[::-1]]
            else:
                target = left_records
            inp = full.copy()
            inp[r:s, c:d] = mask
            # Label path uses the true scene, not solver inference.
            fragments = [_intersection(v[3], v[4], r, s) for v in target]
            ordered = sorted(fragments, key=_length)
            out = np.full((s-r, d-c), bg, dtype=np.int8)
            if r <= row < s:
                out[row-r] = carrier
            for dest, interval in zip(target, ordered):
                if interval is not None:
                    out[interval[0]:interval[1], dest[0]-c:dest[1]-c] = dest[2]
            if vertical:
                inp, out = inp.T.copy(), out.T.copy()
            pair = {'input': inp, 'output': out}
            feature = _features(_check_pair(pair))
            if witness == 1:
                _require(feature['whole_disagrees'] and feature['visible_disagrees']
                         and feature['area_disagrees'] and feature['reversal_disagrees']
                         and feature['changed'], "witness lost its counterexample")
            elif witness == 2:
                _require(feature['offset_move'] and feature['changed'],
                         "offset witness must move biased intervals")
            return pair
        except ValueError:
            continue
    raise RuntimeError('bounded construction exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    seed = int(seed)
    if seed < 0:
        seed %= 1 << 64
    rng = np.random.default_rng(seed)
    roles = tuple(int(v) for v in rng.choice(10, 3, replace=False))
    config = {'fixed_roles': bool(rng.random() < .25), 'roles': roles,
              'palette': [int(v) for v in rng.choice([c for c in range(10) if c not in roles],
                                                    4, replace=False)]}
    mode = str(rng.choice(['horizontal', 'vertical', 'mixed']))
    if mode == 'mixed':
        axes = [False, True] + [bool(rng.integers(2)) for _ in range(num_train-2)]
        test_axes = [bool(rng.integers(2)) for _ in range(num_test)]
    else:
        axes = [mode == 'vertical'] * num_train
        test_axes = [mode == 'vertical'] * num_test
    training_left = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, orientations in (('train', axes), ('test', test_axes)):
        for i, vertical in enumerate(orientations):
            witness = i+1 if split == 'train' and i < 2 else 0
            left = training_left if split == 'train' else not training_left
            for _ in range(40):
                pair = _make_pair(rng, config, left, vertical, witness, split == 'test')
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('duplicate avoidance exhausted')
            seen.add(key)
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test():
    """Bounded 42-seed sweep, count contract, metamorphic and mutation checks."""
    signatures, features, structures = set(), [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_key(puzzle))
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                original = pair['input'].copy()
                prediction = solve(pair['input'])
                assert np.array_equal(prediction, pair['output'])
                assert np.array_equal(pair['input'], original), 'solve mutated input'
                f = _features(_check_pair(pair))
                f['split'] = split
                features.append(f)
                roles = {}
                normalized = tuple(tuple(roles.setdefault(int(v), len(roles)) for v in line)
                                   for line in pair['input'])
                structures.add(normalized)
        if seed in (0, 17, 41):
            assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
    assert len(signatures) == 42
    assert _puzzle_key(generate_puzzle(-1)) == _puzzle_key(generate_puzzle(-1))
    assert len(structures) >= 150
    assert len({f['shape'] for f in features}) >= 50
    assert len({f['output_shape'] for f in features}) >= 35
    assert {f['count'] for f in features} == set(range(2, 6))
    assert {f['vertical'] for f in features} == {False, True}
    assert {f['left'] for f in features} == {False, True}
    assert {c for f in features for c in f['role_colors']} == set(range(10))
    assert {w for f in features for w in f['widths']} == {1, 2, 3}
    for name in ('empty', 'ties', 'touching', 'repeat', 'edge'):
        assert any(f[name] for f in features), name
    assert any(1 in f['lost'] for f in features)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(314159+nt*10+nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (4, 1.5)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p = generate_puzzle(91, 2, 1)
    transposed = {s: [{k: v.T.copy() for k, v in pair.items()} for pair in p[s]]
                  for s in p}
    validate_generated_puzzle(transposed)
    mapping = np.array([4, 8, 1, 9, 3, 0, 7, 2, 6, 5], dtype=np.int8)
    renamed = {s: [{k: mapping[v] for k, v in pair.items()} for pair in p[s]] for s in p}
    validate_generated_puzzle(renamed)
    for q in (transposed, renamed):
        for pair in q['train']+q['test']:
            assert np.array_equal(solve(pair['input']), pair['output'])
    bad = copy.deepcopy(p)
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('wrong label accepted')
    # Explicit repair-only and area mutations must fail relational checking.
    defeated = set()
    for pair in p['train']:
        rec = _analyze(pair['input'])
        r, s, c, d = rec['box']
        faithful = rec['full'][r:s, c:d]
        if rec['vertical']:
            faithful = faithful.T
        candidates = {'repair': faithful}
        slots, spans = rec['target'], rec['intervals']
        area = sorted(range(len(slots)),
                      key=lambda i: (slots[i][1]-slots[i][0])*_length(spans[i]))
        candidates['area'] = _render_window(rec, [spans[i] for i in area])
        for name, out in candidates.items():
            if np.array_equal(out, pair['output']):
                continue
            try:
                _check_pair({'input': pair['input'], 'output': out})
            except ValueError:
                defeated.add(name)
            else:
                raise AssertionError(f'{name} shortcut accepted')
    assert defeated == {'repair', 'area'}
    return {'seeds': 42, 'distinct_puzzles': len(signatures),
            'color_normalized_inputs': len(structures),
            'input_shapes': len({f['shape'] for f in features}),
            'output_shapes': len({f['output_shape'] for f in features}),
            'pairs': len(features),
            'empty_fragment_pairs': sum(f['empty'] for f in features),
            'tied_fragment_pairs': sum(f['ties'] for f in features)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
