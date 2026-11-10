"""Cyclic ribbon repair: match a periodic swatch word, then resize to the middle.

Only NumPy and the standard library are used. No parent implementation is loaded.
The sampler draws labels directly; the solver reconstructs geometry from pixels.
"""
from functools import lru_cache
import json
import numpy as np


HOST_SEEDS = (0, 958342514, 111865447, 771200534, 91066356, 87150421,
              388169869, 977321091, 530597089, 890906063, 929934482, 612580506,
              87335771, 409960242, 83120939, 345620415, 137976476, 189055278,
              429313693, 677632180, 163948148, 262843129, 149913075, 209018344,
              871960637, 489993790, 209113911, 118640535, 620079180, 213250904,
              887623871, 37272048, 75573282, 299431920, 768039262, 342111624,
              986141209, 566621044, 677475586, 957442012, 3042810, 100368469)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    seen = set()
    groups = []
    for rr, cc in np.argwhere(mask):
        p = int(rr), int(cc)
        if p in seen:
            continue
        seen.add(p)
        stack, cells = [p], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (0 <= q[0] < mask.shape[0] and 0 <= q[1] < mask.shape[1]
                        and mask[q] and q not in seen):
                    seen.add(q)
                    stack.append(q)
        groups.append(cells)
    return groups


def _word(row, phase, length):
    return tuple(int(row[(phase + k) % len(row)]) for k in range(length))


def _matches(sequence, motifs):
    return [(r, p) for r, row in enumerate(motifs) for p in range(len(row))
            if _word(row, p, len(sequence)) == tuple(sequence)]


def _overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _parse(input_grid):
    """Return a single input-only interpretation, including cyclic phase matches."""
    inp = _grid(input_grid)
    counts = np.bincount(inp.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    assert counts[bg] > np.partition(counts, -2)[-2], 'background must be unique'
    swatches, beads = [], []
    for cells in _components(inp != bg):
        if len(cells) == 1:
            swatches.append(cells[0])
            continue
        points = np.array(cells)
        y, x = (int(v) for v in points.min(axis=0))
        bottom, right = (int(v) + 1 for v in points.max(axis=0))
        s = bottom - y
        assert 2 <= s <= 4 and right - x == s and len(cells) == s * s
        color = int(inp[y, x])
        assert np.all(inp[y:bottom, x:right] == color), 'beads must be solid squares'
        beads.append((y, x, s, color))
    assert len(swatches) >= 6 and len(beads) >= 6
    points = np.array(swatches)
    mr, mc = (int(v) for v in points.min(axis=0))
    er, ec = (int(v) + 1 for v in points.max(axis=0))
    assert (er - mr) % 2 == 1 and (ec - mc) % 2 == 1
    mini = inp[mr:er:2, mc:ec:2].copy()
    assert np.all(mini != bg) and mini.size == len(swatches)
    mask = np.zeros((er - mr, ec - mc), dtype=bool)
    mask[::2, ::2] = True
    assert np.all(inp[mr:er, mc:ec][~mask] == bg)
    mini_rect = mr, mc, er, ec
    side = beads[0][2]
    assert all(b[2] == side for b in beads)
    candidates = []
    for axis in (1, 0):
        # axis is the direction of bead progression, cross is ribbon ordering.
        cross = 1 - axis
        groups = {}
        for bead in beads:
            groups.setdefault(bead[cross], []).append(bead)
        positions = sorted(groups)
        n = len(positions)
        if n not in (3, 5, 7, 9) or min(np.diff(positions)) < side + 1:
            continue
        ordered = [sorted(groups[p], key=lambda b: b[axis]) for p in positions]
        if any(len(g) < 2 for g in ordered):
            continue
        differences = [b[axis] - a[axis] for g in ordered for a, b in zip(g, g[1:])]
        stride = differences[0]
        if not side < stride <= side + 3 or any(d != stride for d in differences):
            continue
        common_min = len({g[0][axis] for g in ordered}) == 1
        common_max = len({g[-1][axis] for g in ordered}) == 1
        if common_min == common_max:
            continue
        sign = 1 if common_min else -1
        directed = [g if sign == 1 else g[::-1] for g in ordered]
        motifs = mini if axis == 1 else mini.T
        if sign == -1:
            motifs = motifs[:, ::-1]
        if not (2 <= motifs.shape[0] <= 4 and 3 <= motifs.shape[1] <= 6):
            continue
        matches = [_matches([b[3] for b in g], motifs) for g in directed]
        if any(len(m) != 1 for m in matches):
            continue
        lengths = tuple(len(g) for g in directed)
        reference = lengths[n // 2]
        if not 3 <= reference <= 9:
            continue
        valid = True
        for g in directed:
            anchor = g[0]
            for k in range(reference):
                loc = [anchor[0], anchor[1]]
                loc[axis] += sign * k * stride
                y, x = loc
                rect = y, x, y + side, x + side
                if not (0 <= y <= inp.shape[0] - side and 0 <= x <= inp.shape[1] - side):
                    valid = False
                elif _overlap(rect, mini_rect):
                    valid = False
                elif k >= len(g) and np.any(inp[y:y + side, x:x + side] != bg):
                    valid = False
        if valid:
            candidates.append(dict(background=bg, side=side, stride=stride,
                                   axis=axis, sign=sign, groups=directed,
                                   positions=tuple(positions), lengths=lengths,
                                   count=n, reference=reference, motifs=motifs,
                                   matches=tuple(m[0] for m in matches),
                                   mini_rect=mini_rect, canvas=inp.shape))
    assert len(candidates) == 1, 'input must have one ribbon interpretation'
    return candidates[0]


def _finish(inp, a, reference=None, constant_tail=False):
    out = np.asarray(inp, dtype=np.int8).copy()
    target = a['reference'] if reference is None else reference
    s = a['side']
    for g, (r, p) in zip(a['groups'], a['matches']):
        anchor = g[0]
        for b in g[target:]:
            out[b[0]:b[0] + s, b[1]:b[1] + s] = a['background']
        for k in range(len(g), target):
            loc = [anchor[0], anchor[1]]
            loc[a['axis']] += a['sign'] * k * a['stride']
            y, x = loc
            assert 0 <= y <= out.shape[0] - s and 0 <= x <= out.shape[1] - s
            color = g[-1][3] if constant_tail else int(a['motifs'][r, (p + k) % a['motifs'].shape[1]])
            out[y:y + s, x:x + s] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from this input alone; neither training labels nor sampling state enter."""
    inp = _grid(input_grid)
    return _finish(inp, _parse(inp))


@lru_cache(maxsize=3)
def _options(kind):
    result = []
    for n in ((5,) if kind == 'witness' else (3, 5, 7, 9)):
        for s in (2, 3, 4):
            cross_box = (n - 1) * (s + 1) + s
            if kind == 'witness':
                cross_box += 2 * (s + 1)
            for gutter in (1, 2, 3):
                for h in (2, 3, 4):
                    for w in (3, 4, 5, 6):
                        refs = range(w + 1, min(w + 3, 9) + 1) if kind == 'test' else range(3, w + 1)
                        for ref in refs:
                            if kind == 'witness' and ref == w:
                                continue
                            for growth in (1, 2, 3):
                                cap = ref + growth
                                axial_box = (cap - 1) * (s + gutter) + s
                                for sep in (0, 1):
                                    needed = [max(cross_box, 2 * h - 1), max(axial_box, 2 * w - 1)]
                                    needed[sep] = (cross_box + 2 * h if sep == 0 else axial_box + 2 * w)
                                    if max(needed) <= 30:
                                        result.append((n, s, gutter, h, w, ref, cap, sep))
    return tuple(result)


def _lengths(rng, n, ref, cap, kind, mode):
    result = [ref] * n
    others = [int(i) for i in rng.permutation(n) if i != n // 2]
    short = lambda: int(rng.integers(2, ref))
    long = lambda: int(rng.integers(ref + 1, cap + 1))
    if kind == 'witness':
        for i in others[:3]:
            result[i] = short()
        result[others[3]] = long()
    elif mode in ('one_extend', 'one_trim'):
        result[others[0]] = short() if mode == 'one_extend' else long()
    else:
        for i in others:
            if rng.random() < .20:
                continue
            result[i] = short() if mode == 'extend' or (mode == 'mixed' and rng.random() < .5) else long()
        if mode == 'mixed':
            result[others[0]], result[others[1]] = short(), long()
        elif result == [ref] * n:
            result[others[0]] = short() if mode == 'extend' else long()
    return result


def _phase_choices(motifs, length):
    return [(r, p) for r, row in enumerate(motifs) for p in range(len(row))
            if len(_matches(_word(row, p, length), motifs)) == 1]


def _sample_motifs(rng, h, w, colors, lengths, ref, teaching):
    # At least one unique bigram per motif guarantees a usable clue for any
    # allowed length. Other phases may need longer clues to become unique.
    short_index = next((i for i, length in enumerate(lengths) if length < ref), None)
    for attempt in range(161):
        if attempt == 160:
            # Constructive fallback: row colors alternate with distinct column
            # colors. All cyclic bigrams are unique, including odd-width seams.
            motifs = np.empty((h, w), dtype=np.int8)
            for r in range(h):
                for c in range(w):
                    motifs[r, c] = colors[r] if c % 2 == 0 else colors[h + c // 2]
        else:
            size = int(rng.integers(max(3, int(np.ceil(np.sqrt(h * w)))), 9))
            motifs = rng.choice(colors[:size], size=(h, w)).astype(np.int8)
        pairs = _phase_choices(motifs, 2)
        if {r for r, p in pairs} != set(range(h)):
            continue
        choices = {length: _phase_choices(motifs, length) for length in set(lengths)}
        extension_choices = []
        if short_index is not None:
            length = lengths[short_index]
            extension_choices = [(r, p) for r, p in choices[length]
                                 if p > 0 and p + ref > w
                                 and any(motifs[r, (p + k) % w] != motifs[r, (p + length - 1) % w]
                                         for k in range(length, ref))]
        if teaching and not extension_choices:
            continue
        selections = [pool[int(rng.integers(len(pool)))] for pool in (choices[k] for k in lengths)]
        if teaching:
            selections[short_index] = extension_choices[int(rng.integers(len(extension_choices)))]
            protected_row = selections[short_index][0]
            # No other large clue reveals this short ribbon's unseen word
            # positions. Its continuation therefore has to use the miniature.
            for i, length in enumerate(lengths):
                if i != short_index and selections[i][0] == protected_row:
                    alternatives = [q for q in choices[length] if q[0] != protected_row]
                    selections[i] = alternatives[int(rng.integers(len(alternatives)))]
        # Force two source motifs to participate without imposing physical order.
        if len({r for r, p in selections}) == 1:
            i = next(i for i in range(len(lengths)) if i != short_index)
            alternatives = [q for q in choices[lengths[i]] if q[0] != selections[0][0]]
            selections[i] = alternatives[int(rng.integers(len(alternatives)))]
        return motifs, selections
    raise AssertionError('constructive motif fallback failed')


def _place(rng, mini, box, sep):
    gap = int(rng.integers(1, min(4, 30 - mini[sep] - box[sep]) + 1))
    need = [max(mini[d], box[d]) for d in (0, 1)]
    need[sep] = mini[sep] + box[sep] + gap
    mode = int(rng.choice(4, p=(.15, .05, .40, .40)))
    if mode == 0:
        n = int(rng.integers(max(need), 31))
        shape = n, n
    elif mode == 1:
        shape = 30, 30
    else:
        shape = tuple(int(rng.integers(v, min(30, v + (2 if mode == 2 else 12)) + 1)) for v in need)
    base, anchor = [0, 0], [0, 0]
    start = int(rng.integers(shape[sep] - need[sep] + 1))
    if rng.integers(2):
        base[sep], anchor[sep] = start, start + box[sep] + gap
    else:
        anchor[sep], base[sep] = start, start + mini[sep] + gap
    other = 1 - sep
    base[other] = int(rng.integers(shape[other] - box[other] + 1))
    anchor[other] = int(rng.integers(shape[other] - mini[other] + 1))
    return shape, anchor, base


def _make_pair(rng, kind, turns, previous=None, teaching=False):
    options = _options(kind)
    if previous is not None:
        options = [o for o in options if o[0] != previous['count']
                   and o[1] != previous['side'] and o[5] != previous['reference']]
    assert options
    # Choose count and scale before detailed fitting. Uniformly choosing from
    # all fitting tuples would greatly overrepresent small squares and arrays.
    chosen_count = int(rng.choice(sorted({o[0] for o in options})))
    options = [o for o in options if o[0] == chosen_count]
    chosen_side = int(rng.choice(sorted({o[1] for o in options})))
    options = [o for o in options if o[1] == chosen_side]
    shapes = sorted({(o[3], o[4]) for o in options})
    chosen_shape = shapes[int(rng.integers(len(shapes)))]
    options = [o for o in options if (o[3], o[4]) == chosen_shape]
    chosen_gutter = int(rng.choice(sorted({o[2] for o in options})))
    options = [o for o in options if o[2] == chosen_gutter]
    n, s, gutter, h, w, ref, cap, sep = options[int(rng.integers(len(options)))]
    gaps = [s + 1] * (n - 1)
    if kind == 'witness':
        gaps[0 if rng.integers(2) else -1] += 2 * (s + 1)
    else:
        slack = 30 - s - sum(gaps) - (2 * h if sep == 0 else 0)
        if rng.integers(2):
            extra = int(rng.integers(min(3, slack // (n - 1)) + 1))
            gaps = [g + extra for g in gaps]
        elif slack:
            for _ in range(int(rng.integers(1, min(9, slack) + 1))):
                gaps[int(rng.integers(n - 1))] += 1
    offsets = [0] + [int(v) for v in np.cumsum(gaps)]
    stride = s + gutter
    mode = 'mixed' if teaching or kind == 'test' else str(rng.choice(('mixed', 'extend', 'trim', 'one_extend', 'one_trim')))
    lengths = _lengths(rng, n, ref, cap, kind, mode)
    palette = rng.permutation(10)
    if previous is not None and palette[0] == previous['background']:
        palette[0], palette[1] = palette[1], palette[0]
    bg = int(palette[0])
    motifs, selections = _sample_motifs(rng, h, w, palette[1:], lengths, ref, teaching or kind == 'test')
    box = offsets[-1] + s, (max(max(lengths), ref) - 1) * stride + s
    shape, mini, base = _place(rng, (2 * h - 1, 2 * w - 1), box, sep)
    inp = np.full(shape, bg, dtype=np.int8)
    inp[mini[0]:mini[0] + 2 * h:2, mini[1]:mini[1] + 2 * w:2] = motifs
    out = inp.copy()
    for yoff, length, (r, p) in zip(offsets, lengths, selections):
        y = base[0] + yoff
        for grid, span in ((inp, length), (out, ref)):
            for k in range(span):
                x = base[1] + k * stride
                grid[y:y + s, x:x + s] = motifs[r, (p + k) % w]
    if rng.integers(2):
        inp, out = np.flipud(inp), np.flipud(out)
    inp, out = np.rot90(inp, turns).copy(), np.rot90(out, turns).copy()
    return {'input': inp, 'output': out}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(name + ' outside legal range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 'num_train', 2, 10)
    ns = _count(num_test, 'num_test', 1, 4)
    if nt <= ns:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    seed = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed)
    result, seen = {'train': [], 'test': []}, set()
    common = bool(rng.random() < .30)
    direction = int(rng.integers(4))
    previous = None
    for split, count in (('train', nt), ('test', ns)):
        for index in range(count):
            kind = 'test' if split == 'test' else 'witness' if index == 0 else 'train'
            teaching = split == 'train' and index < 2
            for attempt in range(32):
                pair = _make_pair(rng, kind, direction if common else int(rng.integers(4)),
                                  previous if split == 'train' and index == 1 else None, teaching)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise AssertionError('unable to draw a distinct input')
            seen.add(key)
            result[split].append(pair)
            if split == 'train' and index == 0:
                previous = _parse(pair['input'])
        rng.shuffle(result[split])
    validate_generated_puzzle(result)
    return result


def _features(a):
    ref, w = a['reference'], a['motifs'].shape[1]
    gaps = tuple(int(v) for v in np.diff(a['positions']))
    distances = [abs(2 * p - a['positions'][0] - a['positions'][-1]) for p in a['positions']]
    nearest = [i for i, d in enumerate(distances) if d == min(distances)]
    color_witness = any(length < ref and any(a['motifs'][r, (p + k) % w] != g[-1][3]
                        for k in range(length, ref))
                        for length, g, (r, p) in zip(a['lengths'], a['groups'], a['matches']))
    wrapped_extension = any(length < ref and any(p + k >= w for k in range(length, ref))
                            for length, (r, p) in zip(a['lengths'], a['matches']))
    longer_match = any(len(_matches([b[3] for b in g[:2]], a['motifs'])) > 1
                       for g in a['groups'])
    unseen_word_tail = any(length < ref and length < w
                          and sum(other_r == r for other_r, other_p in a['matches']) == 1
                          for length, (r, p) in zip(a['lengths'], a['matches']))
    return dict(count=a['count'], reference=ref, lengths=a['lengths'], side=a['side'],
                gutter=a['stride'] - a['side'], gaps=gaps, motif_shape=a['motifs'].shape,
                background=a['background'], direction=(a['axis'], a['sign']),
                canvas=a['canvas'], phases=tuple(p for r, p in a['matches']),
                used_rows=len({r for r, p in a['matches']}),
                extend=min(a['lengths']) < ref, trim=max(a['lengths']) > ref,
                nonmedian=sorted(a['lengths'])[a['count'] // 2] != ref,
                geometric_witness=len(nearest) == 1 and nearest[0] != a['count'] // 2
                and a['lengths'][nearest[0]] != ref,
                color_witness=color_witness, wrapped_extension=wrapped_extension,
                longer_match=longer_match, unseen_word_tail=unseen_word_tail)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            a = _parse(inp)
            assert np.array_equal(_finish(inp, a), out), 'incorrect ribbon repair'
            assert not np.array_equal(inp, out)
            s, ref = a['side'], a['reference']
            mr, mc, er, ec = a['mini_rect']
            assert np.array_equal(inp[mr:er, mc:ec], out[mr:er, mc:ec])
            for g in a['groups']:
                for b in g[:ref]:
                    assert np.all(out[b[0]:b[0] + s, b[1]:b[1] + s] == b[3])
                for b in g[ref:]:
                    assert np.all(out[b[0]:b[0] + s, b[1]:b[1] + s] == a['background'])
            f = _features(a)
            assert f['used_rows'] >= 2
            features[split].append(f)
    train = features['train']
    for name in ('count', 'reference', 'side', 'background'):
        assert len({f[name] for f in train}) >= 2, name
    assert any(f['nonmedian'] and f['geometric_witness'] for f in train)
    assert sum(f['color_witness'] and f['wrapped_extension'] and f['extend'] and f['trim']
               for f in train) >= 2
    assert sum(f['unseen_word_tail'] for f in train) >= 2
    assert all(f['reference'] <= f['motif_shape'][1] for f in train)
    assert any(f['reference'] < f['motif_shape'][1] for f in train)
    assert all(f['reference'] > f['motif_shape'][1] and f['extend'] and f['trim']
               and f['color_witness'] and f['wrapped_extension'] for f in features['test'])


def _signature(puzzle):
    return tuple((_key(p['input']), _key(p['output']))
                 for split in ('train', 'test') for p in puzzle[split])


def self_test():
    """Bounded 42-seed audit, family-specific ablations and symmetry checks."""
    rng = np.random.default_rng(18091)
    puzzles, records = [], {'train': [], 'test': []}
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                a = _parse(inp)
                f = _features(a)
                records[split].append(f)
                assert np.array_equal(solve(inp), out)
                if f['color_witness']:
                    assert not np.array_equal(_finish(inp, a, constant_tail=True), out)
                if f['nonmedian']:
                    median = sorted(a['lengths'])[a['count'] // 2]
                    assert not np.array_equal(_finish(inp, a, median), out)
                if split == 'train' and f['unseen_word_tail']:
                    w = a['motifs'].shape[1]
                    i = next(i for i, (length, (r, p)) in enumerate(zip(a['lengths'], a['matches']))
                             if length < a['reference'] and length < w
                             and sum(other_r == r for other_r, other_p in a['matches']) == 1)
                    r, p = a['matches'][i]
                    q = (p + a['lengths'][i]) % w
                    q = q if a['sign'] == 1 else w - 1 - q
                    mr, mc, er, ec = a['mini_rect']
                    y, x = (mr + 2 * r, mc + 2 * q) if a['axis'] == 1 else (mr + 2 * q, mc + 2 * r)
                    mutated = inp.copy()
                    fresh = next(c for c in range(10) if c not in set(int(v) for v in inp.flat))
                    mutated[y, x] = fresh
                    # The observed large clues and middle count are identical.
                    # A fresh symbol in an unobserved position cannot create a
                    # duplicate match of any existing observed sequence.
                    changed = solve(mutated)
                    differences = changed != out
                    differences[mr:er, mc:ec] = False
                    assert np.any(differences), 'miniature intervention had no ribbon effect'
        # Every seed gets a palette permutation and all eight plane symmetries.
        pair = puzzle['test'][0]
        a = _parse(pair['input'])
        changed_reference = pair['input'].copy()
        b = a['groups'][a['count'] // 2][-1]
        s = a['side']
        changed_reference[b[0]:b[0] + s, b[1]:b[1] + s] = a['background']
        altered = _parse(changed_reference)
        assert altered['reference'] == a['reference'] - 1
        assert not np.array_equal(solve(changed_reference), pair['output'])
        perm = rng.permutation(10).astype(np.int8)
        assert np.array_equal(solve(perm[pair['input']]), perm[pair['output']])
        for k in range(4):
            for mirror in (False, True):
                left, right = np.rot90(pair['input'], k), np.rot90(pair['output'], k)
                if mirror:
                    left, right = np.fliplr(left), np.fliplr(right)
                assert np.array_equal(solve(left), right)
    assert len({_signature(p) for p in puzzles}) == 42
    assert _signature(generate_puzzle(0)) == _signature(puzzles[0])
    for rows in records.values():
        assert {f['count'] for f in rows} == {3, 5, 7, 9}
        assert {f['side'] for f in rows} == {2, 3, 4}
        assert {f['gutter'] for f in rows} == {1, 2, 3}
        assert len({f['direction'] for f in rows}) == 4
        assert len({f['motif_shape'] for f in rows}) >= 8
        assert len({f['canvas'] for f in rows}) >= 15
        assert len({(f['count'], f['lengths'], f['gaps'], f['phases']) for f in rows}) >= 35
        assert any(len(set(f['gaps'])) == 1 for f in rows)
        assert any(len(set(f['gaps'])) > 1 for f in rows)
        assert any(f['longer_match'] for f in rows)
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            generate_puzzle(9271, nt, ns)
    class ForcedHomogeneous:
        def integers(self, *args, **kwargs):
            return rng.integers(*args, **kwargs)

        def choice(self, colors, size):
            return np.full(size, colors[0], dtype=np.int8)

    for h, w in ((2, 3), (4, 6)):
        motifs, selections = _sample_motifs(ForcedHomogeneous(), h, w,
                                           np.arange(1, 10), [2, 2, 3, 2, 4], 3, True)
        assert len({r for r, p in selections}) >= 2
        assert all(len(_matches(_word(motifs[r], p, length), motifs)) == 1
                   for length, (r, p) in zip([2, 2, 3, 2, 4], selections))
    for args in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *args)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    broken = {split: [{k: g.copy() for k, g in p.items()} for p in puzzles[0][split]]
              for split in ('train', 'test')}
    pair = broken['test'][0]
    a = _parse(pair['input'])
    b = a['groups'][a['count'] // 2][0]
    pair['output'][b[0], b[1]] = a['background']
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted damage to the central ribbon')
    # Useful compact measurements; they are generator-owned evidence, not an
    # independent semantic certification.
    return {'seeds': 42, 'pairs': {s: len(rows) for s, rows in records.items()},
            'distinct_puzzles': 42,
            'by_split': {s: {'counts': sorted({f['count'] for f in rows}),
                             'references': sorted({f['reference'] for f in rows}),
                             'shapes': len({f['canvas'] for f in rows}),
                             'motif_shapes': len({f['motif_shape'] for f in rows}),
                             'longer_match_cases': sum(f['longer_match'] for f in rows),
                             'wrapped_extensions': sum(f['wrapped_extension'] for f in rows)}
                         for s, rows in records.items()}}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
