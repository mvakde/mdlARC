"""Expanding pulse fans; all semantic parameters are recovered from input pixels.

Only standard-library modules and NumPy are used. Labels are painted from latent
coordinates; the solver segments pixels and independently fits the propagation
law. No parent implementation, examples, files, or generation state are used.
"""
from itertools import permutations

import numpy as np


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(int, copy=False)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _orient(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _point(r, c, length, t):
    """Construction-side affine isometry, including points outside the bar."""
    if t >= 4:
        c = length - 1 - c
    k = t % 4
    if k == 0:
        return r, c
    if k == 1:
        return length - 1 - c, r
    if k == 2:
        return 1 - r, length - 1 - c
    return c, 1 - r


def _corners(bb):
    r0, r1, c0, c1 = bb
    return {(r0, c0), (r0, c0 + 1), (r0 + 1, c0),
            (r1, c1), (r1, c1 - 1), (r1 - 1, c1)}


def _find_markers(a):
    found = []
    for color in np.unique(a):
        p = np.argwhere(a == color)
        if len(p) != 6:
            continue
        r0, c0 = map(int, p.min(axis=0))
        r1, c1 = map(int, p.max(axis=0))
        if r1 <= r0 or c1 <= c0 or r1 - r0 + c1 - c0 < 4:
            continue
        bb = r0, r1, c0, c1
        if set(map(tuple, p)) == _corners(bb):
            found.append((int(color), bb))
    assert len(found) == 1, 'missing or ambiguous opposite L corners'
    return found[0]


def _components(a, background):
    unseen = set(map(tuple, np.argwhere(a != background)))
    result = []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        stack, cells = [p], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for v in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if v in unseen:
                    unseen.remove(v)
                    stack.append(v)
        result.append(cells)
    return result


def _canonical(patch):
    fits = []
    for t in range(8):
        b = _orient(patch, t)
        if b.shape[0] != 2 or not 3 <= b.shape[1] <= 5:
            continue
        color = int(b[0, 0])
        if (np.all(b[0] == color) and b[1, 0] == color
                and np.all(b[1, 1:] != color)):
            fits.append(b)
    assert len(fits) == 1, 'emitter does not define a unique intrinsic frame'
    return fits[0]


def _decode(value):
    a = _grid(value)
    marker, bb = _find_markers(a)
    colors, counts = np.unique(a, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1
    bg = int(colors[np.argmax(counts)])
    assert bg != marker
    clean = a.copy()
    clean[clean == marker] = bg
    r0, r1, c0, c1 = bb
    assert np.all(clean[r0:r1 + 1, c0:c1 + 1] == bg), 'query must be erased'
    emitters, fronts = [], []
    for cells in _components(clean, bg):
        p = np.asarray(cells, dtype=int)
        lo, hi = p.min(axis=0), p.max(axis=0)
        patch = clean[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1]
        assert patch.size == len(cells), 'objects must be solid rectangles or lines'
        if min(patch.shape) == 2 and 3 <= max(patch.shape) <= 5:
            emitters.append(dict(pos=lo, patch=patch, cells=set(cells)))
        else:
            assert min(patch.shape) == 1 and max(patch.shape) >= 3
            fronts.append(p)
    assert 2 <= len(emitters) <= 4 and fronts
    template = _canonical(emitters[0]['patch'])
    length = template.shape[1]
    anchor = (length - 1) // 2
    structure = set(map(int, template.flat))
    assert 2 <= len(structure) <= 3 and not {bg, marker} & structure
    rail = int(template[0, 0])
    labels = np.arange(2 * length).reshape(2, length)
    occupied = set().union(*(e['cells'] for e in emitters))
    for e in emitters:
        matches = [t for t in range(8) if np.array_equal(_orient(template, t), e['patch'])]
        assert len(matches) == 1, 'emitters must be exact transformed copies'
        t = matches[0]
        lab = _orient(labels, t)
        edge = np.argwhere(lab == 0)[0]
        inner = np.argwhere(lab == length)[0]
        next_slot = np.argwhere(lab == 1)[0]
        e.update(origin=e['pos'] + edge, normal=edge - inner,
                 tangent=next_slot - edge, transform=t, observed=[], clues={})
    observations, code = [], {}
    for p in fronts:
        matches = []
        width = len(p)
        if (width - 1) % (length - 1):
            raise AssertionError('front width is not an integral expansion')
        q = (width - 1) // (length - 1)
        assert q >= 1
        left = -(q - 1) * anchor
        for index, e in enumerate(emitters):
            delta = p - e['origin']
            depths = delta @ e['normal']
            slots = delta @ e['tangent']
            if (len(set(map(int, depths))) != 1 or depths[0] < 2
                    or set(map(int, slots)) != set(range(left, left + width))):
                continue
            # A nearer bar blocks association to a farther collinear emitter.
            center = e['origin'] + anchor * e['tangent']
            if any(tuple(center + d * e['normal']) in occupied
                   for d in range(1, int(depths[0]))):
                continue
            matches.append((index, int(depths[0]), slots))
        assert len(matches) == 1, 'a front must have one emitter interpretation'
        index, distance, slots = matches[0]
        e = emitters[index]
        assert q not in e['observed'], 'duplicate scale observation'
        e['observed'].append(q)
        observations.append((q, distance))
        for point, offset in zip(p, slots):
            color = int(clean[tuple(point)])
            if color == rail:
                continue
            assert color not in structure | {bg, marker}, 'invalid signal role'
            assert (int(offset) - left) % q == 0, 'signal is between homologous slots'
            s = (int(offset) - left) // q
            assert 0 <= s < length
            assert s not in code or code[s] == color, 'contradictory pooled signals'
            code[s] = color
            e['clues'][s] = color
    assert code and len(code) <= 4
    assert all(len(e['observed']) >= 3 for e in emitters), 'insufficient geometric evidence'
    unique = sorted(set(observations))
    by_q = {}
    for q, d in unique:
        assert q not in by_q or by_q[q] == d, 'propagation differs between copies'
        by_q[q] = d
    assert len(by_q) >= 3
    qs = sorted(by_q)
    difference = by_q[qs[1]] - by_q[qs[0]]
    assert difference % (qs[1] - qs[0]) == 0
    step = difference // (qs[1] - qs[0])
    first = by_q[qs[0]] - (qs[0] - 1) * step
    assert 2 <= step <= 3 and 2 <= first <= 4
    assert all(d == first + (q - 1) * step for q, d in by_q.items())
    return dict(grid=a, bg=bg, marker=marker, bounds=bb, template=template,
                rail=rail, length=length, anchor=anchor, first=first, step=step,
                emitters=emitters, code=code)


def _prediction(info, *, fixed_width=False, local_code=False):
    r0, r1, c0, c1 = info['bounds']
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), info['bg'], dtype=int)
    owners = np.full(out.shape, -1, dtype=int)
    queried = set()
    for index, e in enumerate(info['emitters']):
        for q in range(1, 31):
            d = info['first'] + (q - 1) * info['step']
            if d > max(info['grid'].shape):
                break
            scale = 1 if fixed_width else q
            left = -(scale - 1) * info['anchor']
            width = scale * (info['length'] - 1) + 1
            code = e['clues'] if local_code else info['code']
            for j in range(width):
                p = e['origin'] + d * e['normal'] + (left + j) * e['tangent']
                r, c = map(int, p)
                if not (r0 <= r <= r1 and c0 <= c <= c1):
                    continue
                rr, cc = r - r0, c - c0
                assert owners[rr, cc] == -1, 'fronts overlap in the query'
                owners[rr, cc] = index
                out[rr, cc] = code.get(j // scale, info['rail']) if j % scale == 0 else info['rail']
                queried.add((index, q))
    return out, owners, queried


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct the inclusive crop using only the supplied grid."""
    return _prediction(_decode(input_grid))[0]


def _shelf_layout(rng, sizes):
    """Pack variable rectangles; no fixed locations, axes, or canvas dimensions."""
    options = []
    n = len(sizes)
    for order in permutations(range(n)):
        for breaks in range(1 << (n - 1)):
            rows, row = [], []
            for k, i in enumerate(order):
                row.append(i)
                if k == n - 1 or breaks & (1 << k):
                    rows.append(row)
                    row = []
            heights = [max(sizes[i][0] for i in row) for row in rows]
            widths = [sum(sizes[i][1] for i in row) + len(row) - 1 for row in rows]
            h, w = sum(heights) + len(rows) - 1, max(widths)
            if h <= 30 and w <= 30:
                options.append((rows, heights, widths, h, w))
    if not options:
        return None
    rows, heights, widths, h, w = options[int(rng.integers(len(options)))]
    h += int(rng.integers(min(5, 30 - h) + 1))
    w += int(rng.integers(min(5, 30 - w) + 1))
    total_h = sum(heights) + len(rows) - 1
    y = int(rng.integers(h - total_h + 1))
    positions = {}
    for row, rh, rw in zip(rows, heights, widths):
        x = int(rng.integers(w - rw + 1))
        for i in row:
            positions[i] = (y + int(rng.integers(rh - sizes[i][0] + 1)), x)
            x += sizes[i][1] + 1
        y += rh + 1
    return (h, w), positions


def _construct(rng, split, index, variant=None):
    demonstration = split == 'train' and index < 2
    fallback = variant is not None
    length = 3 if fallback else int(rng.integers(3, 6))
    count = 2 if demonstration or fallback else int(rng.choice([2, 3, 4], p=[.65, .25, .10]))
    palette = list(map(int, rng.permutation(10)))
    bg, marker, rail = palette[:3]
    structural_count = 2 if fallback else int(rng.integers(2, 4))
    structural = palette[2:2 + structural_count]
    template = np.full((2, length), rail, dtype=int)
    template[1, 1:] = rng.choice(structural[1:], length - 1)
    if structural_count == 3:
        template[1, -1] = structural[2]
        template[1, 1] = structural[1]
    slot_count = 2 if demonstration or fallback else int(rng.integers(1, min(4, length) + 1))
    slots = sorted(map(int, rng.choice(length, slot_count, replace=False)))
    if demonstration and 0 not in slots and length - 1 not in slots:
        slots[0] = 0
        slots.sort()
    signals = palette[2 + structural_count:]
    code = dict(zip(slots, signals[:slot_count]))
    if not demonstration and not fallback and slot_count > 1 and rng.random() < .3:
        code[slots[-1]] = code[slots[0]]
    first, step = (2, 2) if fallback else (int(rng.integers(2, 5)), int(rng.integers(2, 4)))
    start = 1 if demonstration or fallback else int(rng.choice([1, 1, 2]))
    visible = [list(range(start, start + (3 if demonstration or fallback else int(rng.choice([3, 3, 4])))))
               for _ in range(count)]
    skip = 1 if split == 'test' else 0
    if not fallback and split == 'test' and rng.random() < .25:
        skip = 2
    span = 1 if fallback or (demonstration and index == 0) else int(rng.choice([1, 1, 2]))
    target_first = max(visible[0]) + 1 + skip
    target_last = target_first + span - 1
    qmax = max(target_last, max(map(max, visible)))
    base = 0 if fallback else int(rng.integers(8))
    if demonstration and index == 0:
        transforms = [base, base ^ 4]
    elif demonstration and index == 1:
        transforms = [base, (base // 4) * 4 + (base + 3) % 4]
    else:
        transforms = [base if rng.random() < .4 else int(rng.integers(8)) for _ in range(count)]
    if fallback:
        transforms = [0, 3] if demonstration and index == 1 else [0, 4]
    # Allocate partial observations of the intrinsic slot code.
    hints = [dict() for _ in range(count)]
    mode = 'pool' if demonstration and index == 0 else ('full' if demonstration else str(rng.choice(['pool', 'full', 'redundant'])))
    full_source = 1 if demonstration and index == 1 else (int(rng.integers(count)) if mode == 'full' else 0)
    for k, s in enumerate(slots):
        emitter = k % count if mode == 'pool' else (full_source if mode == 'full' else int(rng.integers(count)))
        q = visible[emitter][k % len(visible[emitter])] if mode == 'full' else int(rng.choice(visible[emitter]))
        if mode == 'full':
            q = visible[emitter][0]
        hints[emitter][q, s] = code[s]
        if mode == 'redundant' and rng.random() < .6:
            other = (emitter + 1) % count
            hints[other][int(rng.choice(visible[other])), s] = code[s]
    anchor = (length - 1) // 2
    def front_cells(q, local_code):
        left = -(q - 1) * anchor
        d = first + (q - 1) * step
        cells = {(-d, left + j): rail for j in range(q * (length - 1) + 1)}
        for s, color in local_code.items():
            cells[-d, left + q * s] = color
        return cells
    widest = front_cells(target_last, {})
    min_c, max_c = min(c for r, c in widest), max(c for r, c in widest)
    top, bottom = -(first + (target_last - 1) * step), -(first + (target_first - 1) * step)
    if fallback:
        top -= 1
        bottom += 1
    else:
        # Margins cannot accidentally include the deliberately skipped front.
        top -= int(rng.integers(step))
        bottom += int(rng.integers(step))
        if not (demonstration and index == 0) and rng.random() < .65:
            trim = max(1, (max_c - min_c) // 3)
            min_c += int(rng.integers(trim + 1))
            max_c -= int(rng.integers(trim + 1))
    while bottom - top < 2:
        upper_margin = -(first + (target_last - 1) * step) - top
        lower_margin = bottom + first + (target_first - 1) * step
        if upper_margin < step - 1 and (rng.integers(2) or lower_margin >= step - 1):
            top -= 1
        else:
            bottom += 1
    if max_c - min_c < 2:
        return None
    if split == 'test':
        # Make cross-emitter correspondence necessary in the actual crop,
        # including single-slot and heavily trimmed test windows.
        needed = [s for s in slots if any(min_c <= q * s - (q - 1) * anchor <= max_c
                                         for q in range(target_first, target_last + 1))]
        if not needed:
            return None
        s = int(rng.choice(needed))
        hints[0] = {key: color for key, color in hints[0].items() if key[1] != s}
        if not any(any(key[1] == s for key in hint) for hint in hints[1:]):
            other = int(rng.integers(1, count))
            hints[other][int(rng.choice(visible[other])), s] = code[s]
    patches, lowers, sizes = [], [], []
    for i, t in enumerate(transforms):
        cells = {(r, c): int(template[r, c]) for r in range(2) for c in range(length)}
        for q in visible[i]:
            clues = {s: color for (qq, s), color in hints[i].items() if qq == q}
            cells.update(front_cells(q, clues))
        observed = {_point(r, c, length, t): color for (r, c), color in cells.items()}
        extent = list(observed)
        extent += [_point(r, c, length, t) for r, c in front_cells(qmax, {})]
        if i == 0:
            extent += [_point(r, c, length, t) for r in (top, bottom) for c in (min_c, max_c)]
        points = np.asarray(extent)
        lo, hi = points.min(axis=0), points.max(axis=0)
        patches.append(observed)
        lowers.append(lo)
        sizes.append(tuple(map(int, hi - lo + 1)))
    if fallback:
        # Sixteen distinct fallback shapes exceed the maximum 13 preceding inputs.
        shape = (18 + variant // 4, 26 + variant % 4)
        places = {0: (1, 1), 1: (1, sizes[0][1] + 2)}
    else:
        layout = _shelf_layout(rng, sizes)
        if layout is None:
            return None
        shape, places = layout
    offsets = [np.asarray(places[i]) - lowers[i] for i in range(count)]
    inp = np.full(shape, bg, dtype=int)
    for i, patch in enumerate(patches):
        for p, color in patch.items():
            v = np.asarray(p) + offsets[i]
            inp[tuple(v)] = color
    query_points = np.asarray([_point(r, c, length, transforms[0])
                               for r in (top, bottom) for c in (min_c, max_c)]) + offsets[0]
    lo, hi = query_points.min(axis=0), query_points.max(axis=0)
    bb = int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1])
    r0, r1, c0, c1 = bb
    if not np.all(inp[r0:r1 + 1, c0:c1 + 1] == bg):
        return None
    # Explicit latent painting, not a call to solve or the fitting path.
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    painted = set()
    contributors = set()
    for i, t in enumerate(transforms):
        for q in range(1, 31):
            for (r, c), color in front_cells(q, code).items():
                pr, pc = np.asarray(_point(r, c, length, t)) + offsets[i]
                if r0 <= pr <= r1 and c0 <= pc <= c1:
                    p = int(pr - r0), int(pc - c0)
                    if p in painted:
                        return None
                    painted.add(p)
                    contributors.add(i)
                    out[p] = color
    if contributors != {0} or not any(np.any(out == color) for color in code.values()):
        return None
    for p in _corners(bb):
        inp[p] = marker
    try:
        info = _decode(inp)
        prediction, _, _ = _prediction(info)
        if not np.array_equal(prediction, out):
            return None
        if demonstration and index == 0:
            if np.array_equal(out, _prediction(info, local_code=True)[0]):
                return None
            if np.array_equal(out, _prediction(info, fixed_width=True)[0]):
                return None
        if split == 'test' and np.array_equal(out, _prediction(info, local_code=True)[0]):
            return None
    except AssertionError:
        return None
    return {'input': inp, 'output': out}


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(name + ' is out of range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count('num_train', num_train, 2, 10)
    num_test = _count('num_test', num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            for attempt in range(96):
                variant = attempt - 80 if attempt >= 80 else None
                pair = _construct(rng, split, index, variant)
                if pair is not None and _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError('bounded layout construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert _key(a) not in seen
            seen.add(_key(a))
            info = _decode(a)
            expected, owners, query = _prediction(info)
            assert np.array_equal(expected, b), 'incorrect expanded-front reconstruction'
            assert info['marker'] not in b
            assert set(map(int, b.flat)) <= set(map(int, a.flat))
            assert np.any(b != info['bg']) and query
            assert len(set(owners[owners >= 0])) == 1, 'query must select one fan'
            for index, q in query:
                assert q > max(info['emitters'][index]['observed']), 'query must extrapolate'
            if split == 'test':
                assert min(q - max(info['emitters'][i]['observed']) for i, q in query) >= 2
                assert not np.array_equal(expected, _prediction(info, local_code=True)[0]), 'test must require transfer'
            if split == 'train':
                training.append(info)
    assert any(not np.array_equal(_prediction(f)[0], _prediction(f, fixed_width=True)[0])
               for f in training), 'training must distinguish growth from fixed slots'
    assert any(not np.array_equal(_prediction(f)[0], _prediction(f, local_code=True)[0])
               for f in training), 'training must require pooling across emitters'
    assert any(len({e['transform'] % 2 for e in f['emitters']}) == 2
               and not np.array_equal(_prediction(f)[0], _prediction(f, local_code=True)[0])
               for f in training), 'training must show quarter-turn code transfer'
    assert any(any(e['transform'] % 4 == other['transform'] % 4
                   and (e['transform'] >= 4) != (other['transform'] >= 4)
                   for e in f['emitters'] for other in f['emitters'])
               for f in training), 'training must show longitudinal reflection'


def self_test():
    signatures, shapes, lengths, counts, steps, palettes, transforms = (set() for _ in range(7))
    codes, expansions, normalized = set(), set(), set()
    pooled = repeated = multiple = skipped = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for p in puzzle['train'] + puzzle['test']))
        for pair in puzzle['train'] + puzzle['test']:
            f = _decode(pair['input'])
            answer, owner, query = _prediction(f)
            shapes.add((pair['input'].shape, answer.shape))
            lengths.add(f['length'])
            counts.add(len(f['emitters']))
            steps.add((f['first'], f['step']))
            palettes.add((f['bg'], f['marker'], tuple(f['template'].flat), tuple(f['code'].items())))
            transforms.update(e['transform'] for e in f['emitters'])
            codes.add(tuple(sorted(f['code'])))
            expansions.update(q for i, q in query)
            normalized.add((pair['input'].shape, (pair['input'] != f['bg']).tobytes()))
            pooled += all(len(e['clues']) < len(f['code']) for e in f['emitters'])
            repeated += len(set(f['code'].values())) < len(f['code'])
            multiple += len(query) > 1
            skipped += any(q > max(f['emitters'][i]['observed']) + 1 for i, q in query)
            assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
    assert len(signatures) == 42 and len(normalized) >= 150
    assert len(shapes) >= 50 and lengths == {3, 4, 5}
    assert {2, 3, 4} <= counts and len(steps) >= 5
    assert len(palettes) >= 150 and transforms == set(range(8))
    assert len(codes) >= 10 and len(expansions) >= 3
    assert pooled >= 42 and repeated >= 5 and multiple >= 10 and skipped >= 42
    # A compatible unsampled arrangement: nonconsecutive observations, an
    # observed front touching a border, and a future front beginning off-canvas.
    custom = np.zeros((18, 30), dtype=int)
    custom[15, 3:6] = 1
    custom[16, 3:6] = [1, 2, 2]
    custom[15, 23:26] = 1
    custom[16, 23:26] = [2, 2, 1]
    for q in (1, 3, 4):
        r = 15 - 2 * q
        custom[r, 4 - q:5 + q] = 1
        custom[r, 24 - q:25 + q] = 1
    custom[9, 1] = 3  # slot zero of the first emitter, scale three
    custom[9, 21] = 4  # slot two of the reflected emitter, scale three
    for p in _corners((2, 4, 0, 10)):
        custom[p] = 9
    expected = np.zeros((3, 11), dtype=int)
    expected[1] = [1] * 10 + [4]
    assert np.array_equal(solve(custom), expected)
    # Directly exercise every fallback, including the quarter-turn demonstration.
    for split, index in (('train', 0), ('train', 1), ('test', 0)):
        fallback_shapes = set()
        for variant in range(16):
            pair = _construct(np.random.default_rng(3000 + variant), split, index, variant)
            assert pair is not None
            assert np.array_equal(solve(pair['input']), pair['output'])
            fallback_shapes.add(pair['input'].shape)
        assert len(fallback_shapes) == 16
    # Input-only invariance under role permutations and whole-grid isometries.
    p = generate_puzzle(701, 2, 1)
    for pair in p['train'] + p['test']:
        for t in range(8):
            f = _decode(pair['input'])
            a = pair['input'].copy()
            a[a == f['marker']] = f['bg']
            bb = f['bounds']
            labels = np.arange(a.size).reshape(a.shape)
            moved = _orient(labels, t)
            marked = [np.argwhere(moved == labels[r, c])[0] for r, c in _corners(bb)]
            pts = np.asarray(marked)
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            new_bb = int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1])
            a = _orient(a, t)
            for cell in _corners(new_bb):
                a[cell] = f['marker']
            permutation = np.roll(np.arange(10), t + 1)
            assert np.array_equal(solve(permutation[a]), permutation[_orient(pair['output'], t)])
    for nt, ne in ((2, 1), (10, 1), (10, 4), (5, 4), (3, 2)):
        q = generate_puzzle(900 + nt + ne, nt, ne)
        other = generate_puzzle(900 + nt + ne, nt, ne)
        assert all(_key(a[k]) == _key(b[k]) for s in q for a, b in zip(q[s], other[s]) for k in a)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = generate_puzzle(951, 2, 1)
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
