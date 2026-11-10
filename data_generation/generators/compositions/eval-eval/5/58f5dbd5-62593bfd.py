"""Floating cutouts: color selects a cavity; rotational company sets its edge.

No files, parent modules, seed state, or examples are read by solve(). Generation
uses bounded rectangle packing. Tests and self-tests stream their measurements;
this module writes no artifacts or execution traces.
"""
from collections import Counter
from hashlib import sha256

import numpy as np


_D8 = tuple((y, x) for y in (-1, 0, 1) for x in (-1, 0, 1) if (y, x) != (0, 0))


def _key(mask):
    return mask.shape, mask.astype(np.uint8).tobytes()


def _trim(mask):
    yy, xx = np.nonzero(mask)
    assert len(yy), 'empty stencil'
    return mask[yy.min():yy.max()+1, xx.min():xx.max()+1].copy()


def _canon(mask, mode='rotation'):
    if mode == 'area':
        return int(mask.sum())
    turns = (0,) if mode == 'identity' else (0, 2) if mode == 'half' else range(4)
    forms = [_key(np.rot90(mask, t)) for t in turns]
    if mode == 'dihedral':
        forms += [_key(np.rot90(np.fliplr(mask), t)) for t in range(4)]
    return min(forms)


def _connected(mask, directions=_D8):
    cells = set(map(tuple, np.argwhere(mask)))
    if not cells:
        return False
    seen = {min(cells)}
    todo = list(seen)
    while todo:
        y, x = todo.pop()
        for dy, dx in directions:
            p = y+dy, x+dx
            if p in cells and p not in seen:
                seen.add(p)
                todo.append(p)
    return seen == cells


def _random_mask(rng):
    kind = int(rng.integers(9))
    h, w = (int(rng.integers(2, 6)) for _ in range(2))
    m = np.zeros((h, w), dtype=bool)
    if kind == 0:
        m = np.ones((1, int(rng.integers(1, 6))), dtype=bool)
    elif kind == 1:
        m[:] = True
    elif kind == 2:
        m[:, 0] = m[-1, :] = True
    elif kind == 3:
        m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
    elif kind == 4:
        m = np.eye(int(rng.integers(2, 6)), dtype=bool)
        if rng.random() < .4:
            m |= np.fliplr(m)
    elif kind == 5:
        m[int(rng.integers(h)), :] = True
        m[:, int(rng.integers(w))] = True
    else:
        m = rng.random((h, w)) < float(rng.uniform(.25, .85))
        if not m.any():
            m[0, 0] = True
    return _trim(np.rot90(m, int(rng.integers(4))))


def _shape(rng, forbidden, *, chiral=False):
    def allowed(m):
        return (_canon(m) not in forbidden and _connected(m)
                and (not chiral or (_canon(m) != _canon(np.fliplr(m))
                     and len({_key(np.rot90(m, t)) for t in range(4)}) == 4
                     and 2*int(m.sum()) != m.size)))

    for _ in range(240):
        m = _random_mask(rng)
        if allowed(m):
            return m
    # Small finite fallback, also valid for asymmetric teaching shapes.
    for bits in range(1, 512):
        m = _trim(np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3))
        if allowed(m):
            return m
    raise RuntimeError('shape vocabulary exhausted')


def _design(rng, witness):
    """Return masks, intended class labels, and colors' owner indices."""
    masks, labels = [], []

    def add_group(base, count, label, turns=None):
        turns = [int(rng.integers(4)) for _ in range(count)] if turns is None else turns
        for t in turns:
            masks.append(np.rot90(base, t).copy())
            labels.append(label)

    if witness == 'contrast':
        base = _shape(rng, set(), chiral=True)
        add_group(base, 2, 0, [0, int(rng.choice([1, 3]))])
        add_group(np.fliplr(base), 1, 1, [0])
        forbidden = {_canon(base), _canon(np.fliplr(base))}
        ordinary = _shape(rng, forbidden)
        add_group(ordinary, 1, 2)
        forbidden |= {_canon(ordinary), _canon(np.fliplr(ordinary))}
        owners = [0, 2, 3]
        if rng.random() < .5:
            owners.append(1)
        extras = int(rng.integers(3))
    elif witness == 'scope':
        base = _shape(rng, set(), chiral=True)
        add_group(base, 3, 0, [0, 0, 2])
        forbidden = {_canon(base), _canon(np.fliplr(base))}
        ordinary = _shape(rng, forbidden)
        add_group(ordinary, 1, 1)
        forbidden |= {_canon(ordinary), _canon(np.fliplr(ordinary))}
        # The two other class-0 stencils have NO chambers. They still lift 0.
        owners, extras = [0, 3], int(rng.integers(4))
    else:
        target = int(rng.integers(3, 9))
        forbidden, label = set(), 0
        while len(masks) < target:
            remaining = target-len(masks)
            count = int(rng.choice([1, 2, 3], p=[.45, .35, .20]))
            count = min(count, remaining)
            # Either handedness can have its own independent repeated class.
            mirrors = [np.fliplr(m) for m in masks if _canon(np.fliplr(m)) not in forbidden]
            base = (mirrors[int(rng.integers(len(mirrors)))].copy()
                    if mirrors and rng.random() < .2 else _shape(rng, forbidden))
            add_group(base, count, label)
            forbidden.add(_canon(base))
            label += 1
        count = int(rng.integers(1, min(4, target)+1))
        owners = [int(i) for i in rng.choice(target, count, replace=False)]
        return masks, labels, owners

    for _ in range(extras):
        m = _shape(rng, forbidden)
        add_group(m, 1, max(labels)+1)
        forbidden |= {_canon(m), _canon(np.fliplr(m))}
        if len(owners) < 4 and rng.random() < .6:
            owners.append(len(masks)-1)
    return masks, labels, owners


def _pack(rng, items):
    """At most 24 greedy rectangle placements; no unbounded packing search."""
    area = sum((h+1)*(w+1) for _, _, h, w in items)
    maxh, maxw = max(q[2] for q in items)+2, max(q[3] for q in items)+2
    for attempt in range(24):
        if attempt >= 12:
            height = width = 30
        else:
            height = int(rng.integers(max(maxh, 14), 31))
            width = int(rng.integers(max(maxw, 14), 31))
            if height*width < area*1.15:
                continue
        blocked = np.zeros((height, width), dtype=bool)
        order = sorted(range(len(items)), key=lambda i: (-items[i][2]*items[i][3], float(rng.random())))
        placements = {}
        for i in order:
            kind, index, h, w = items[i]
            cs = np.pad(blocked.astype(np.int16), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
            sums = cs[h:, w:]-cs[:-h, w:]-cs[h:, :-w]+cs[:-h, :-w]
            locations = np.argwhere(sums[1:-1, 1:-1] == 0)+1
            if not len(locations):
                break
            chosen = 0 if attempt == 23 else int(rng.integers(len(locations)))
            y, x = map(int, locations[chosen])
            placements[kind, index] = (y, x)
            blocked[y-1:y+h+1, x-1:x+w+1] = True
        else:
            return height, width, placements
    return None


def _make_pair(rng, witness, test, background):
    for _ in range(48):
        masks, labels, owners = _design(rng, witness)
        if test:
            frequencies = Counter(labels)
            repeated = [i for i, label in enumerate(labels) if frequencies[label] > 1 and int(masks[i].sum()) >= 3]
            singletons = [i for i, label in enumerate(labels) if frequencies[label] == 1]
            if not repeated or not singletons:
                continue
            owners = [int(rng.choice(repeated)), int(rng.choice(singletons))]
            remaining = [i for i in range(len(masks)) if i not in owners]
            extra = int(rng.integers(min(2, len(remaining))+1))
            owners += [int(i) for i in rng.choice(remaining, extra, replace=False)]
        colors = [int(c) for c in rng.choice([c for c in range(10) if c != background], len(masks), replace=False)]
        chamber_sizes = {}
        for i in owners:
            mh, mw = masks[i].shape
            slack = int(rng.integers(5, 7)) if test else max(5-mh, int(rng.integers(2, 5)))
            chamber_sizes[i] = (mh+slack+2, mw+2)
        if witness == 'contrast':
            # Same-sized chambers with opposite destinations reject size sorting.
            chamber_sizes[2] = chamber_sizes[0]
        items = [('plate', i, *chamber_sizes[i]) for i in owners]
        items += [('stencil', i, *m.shape) for i, m in enumerate(masks)]
        packed = _pack(rng, items)
        if packed is None:
            continue
        h, w, positions = packed
        inp = np.full((h, w), background, dtype=np.int8)
        out = inp.copy()
        frequencies = Counter(labels)
        for i in owners:
            y, x = positions['plate', i]
            ph, pw = chamber_sizes[i]
            inp[y:y+ph, x:x+pw] = colors[i]
            out[y:y+ph, x:x+pw] = colors[i]
            mh, mw = masks[i].shape
            dy = 1 if frequencies[labels[i]] > 1 else ph-1-mh
            out[y+dy:y+dy+mh, x+1:x+1+mw][masks[i]] = background
        for i, m in enumerate(masks):
            y, x = positions['stencil', i]
            inp[y:y+m.shape[0], x:x+m.shape[1]][m] = colors[i]
        # Crop BOTH canvases identically to input content plus a background frame.
        yy, xx = np.nonzero(inp != background)
        box = (slice(int(yy.min())-1, int(yy.max())+2), slice(int(xx.min())-1, int(xx.max())+2))
        inp, out = inp[box].copy(), out[box].copy()
        pad = []
        for dimension in inp.shape:
            total = int(rng.integers(min(4, 30-dimension)+1))
            if rng.random() < .15:
                total = 30-dimension
            a = int(rng.integers(total+1))
            pad.append((a, total-a))
        inp = np.pad(inp, pad, constant_values=background)
        out = np.pad(out, pad, constant_values=background)
        return {'input': inp, 'output': out}
    raise RuntimeError('bounded packing budget exhausted')


def _components(grid, bg):
    seen = np.zeros(grid.shape, dtype=bool)
    result = []
    h, w = grid.shape
    for yy, xx in np.argwhere(grid != bg):
        y, x = int(yy), int(xx)
        if seen[y, x]:
            continue
        color = int(grid[y, x])
        todo, cells = [(y, x)], []
        seen[y, x] = True
        while todo:
            cy, cx = todo.pop()
            cells.append((cy, cx))
            for dy, dx in _D8:
                ny, nx = cy+dy, cx+dx
                if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and int(grid[ny, nx]) == color:
                    seen[ny, nx] = True
                    todo.append((ny, nx))
        result.append((color, np.array(cells, dtype=int)))
    return result


def _check_grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= d <= 30 for d in g.shape)
    assert np.issubdtype(g.dtype, np.integer) and np.all((0 <= g) & (g <= 9))
    return g


def _decode(input_grid):
    g = _check_grid(input_grid)
    bg = int(g[0, 0])
    assert all(np.all(edge == bg) for edge in (g[0], g[-1], g[:, 0], g[:, -1])), 'background frame missing'
    plates, plate_cells = {}, np.zeros(g.shape, dtype=bool)
    for color, cells in _components(g, bg):
        y, x = map(int, cells.min(0))
        y1, x1 = map(int, cells.max(0)+1)
        h, w = y1-y, x1-x
        if h >= 7 and 3 <= w <= 7 and len(cells) == h*w:
            assert color not in plates, 'multiple chambers for one color'
            plates[color] = (y, x, h, w)
            plate_cells[y:y1, x:x1] = True
    assert plates, 'no solid chamber'
    patterns, boxes = {}, []
    external = (g != bg) & ~plate_cells
    for color in map(int, np.unique(g[external])):
        cells = np.argwhere(external & (g == color))
        y, x = map(int, cells.min(0))
        y1, x1 = map(int, cells.max(0)+1)
        assert 1 <= y1-y <= 5 and 1 <= x1-x <= 5, 'stencil bounds'
        region = g[y:y1, x:x1]
        assert not plate_cells[y:y1, x:x1].any()
        assert np.all((region == bg) | (region == color)), 'mixed stencil register'
        pattern = region == color
        assert _connected(pattern), 'stencil must be eight-connected'
        patterns[color] = pattern
        boxes.append((y, x, y1-y, x1-x))
    assert set(plates) <= set(patterns), 'missing same-colored stencil'
    for color, (y, x, h, w) in plates.items():
        mh, mw = patterns[color].shape
        assert w-2 == mw and h-2-mh >= 2, 'stencil does not fit chamber'
    # Each visible register/chamber is separated by at least one blank row or
    # column. This is a geometric parsing contract, not a class-membership cue.
    rects = list(plates.values())+boxes
    for i, (y, x, h, w) in enumerate(rects):
        for v, u, a, b in rects[i+1:]:
            assert y+h < v or v+a < y or x+w < u or u+b < x, 'registers touch'
    return {'shape': g.shape, 'bg': bg, 'plates': plates, 'patterns': patterns}


def _render(info, mode='rotation', *, scope='all', destination='normal', polarity='subtract', mirror_holes=False):
    patterns, plates = info['patterns'], info['plates']
    out = np.full(info['shape'], info['bg'], dtype=np.int8)
    if mode == 'chambers':
        keys = {c: (h, w) for c, (_, _, h, w) in plates.items()}
    else:
        keys = {c: _canon(m, mode) for c, m in patterns.items()}
    counted = list(plates) if scope == 'owners' or mode == 'chambers' else list(patterns)
    counts = Counter(keys[c] for c in counted)
    for c, (y, x, h, w) in plates.items():
        m = patterns[c]
        repeated = counts[keys[c]] == 2 if destination == 'exactly_two' else counts[keys[c]] >= 2
        up = (not repeated if destination == 'reversed' else True if destination == 'top'
              else False if destination == 'bottom' else repeated)
        dy = 1 if up else h-1-m.shape[0]
        out[y:y+h, x:x+w] = c
        m = np.fliplr(m) if mirror_holes else m
        m = ~m if polarity == 'complement' else m
        out[y+dy:y+dy+m.shape[0], x+1:x+1+m.shape[1]][m] = info['bg']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct all stencil masks and their rotational classes from input."""
    return _render(_decode(input_grid))


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    stable_bg = int(rng.integers(10)) if rng.random() < .2 else None
    witnesses = ['contrast', 'scope']+['free']*(nt-2)
    rng.shuffle(witnesses)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ne)):
        for index in range(count):
            for _ in range(16):
                background = int(rng.integers(10)) if stable_bg is None else stable_bg
                pair = _make_pair(rng, witnesses[index] if split == 'train' else 'free', split == 'test', background)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct input budget exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    p, owners = info['patterns'], info['plates']
    keys = {c: _canon(m) for c, m in p.items()}
    freq = Counter(keys.values())
    ownfreq = Counter(keys[c] for c in owners)
    statuses = {c: freq[keys[c]] > 1 for c in owners}
    repeated = [c for c in owners if statuses[c]]
    single = [c for c in owners if not statuses[c]]
    return {
        'stencils': len(p), 'chambers': len(owners), 'background': info['bg'],
        'multiplicities': sorted(freq.values()), 'mixed': bool(repeated and single),
        'ownerless_lift': any(ownfreq[keys[c]] == 1 for c in repeated),
        'quarter_turn': any(keys[a] == keys[b] and _key(p[a]) != _key(p[b])
                            and any(_key(np.rot90(p[a], t)) == _key(p[b]) for t in (1, 3))
                            for a in repeated for b in p if a != b),
        'reflection_negative': any(keys[c] != keys[d] and _canon(np.fliplr(p[c])) == keys[d]
                                   for c in single for d in p if c != d),
        'ordinary_singleton': any(_canon(np.fliplr(p[c])) == keys[c]
                                  or _canon(np.fliplr(p[c])) not in freq for c in single),
        'same_chamber_opposite': any(owners[a][2:] == owners[b][2:] for a in repeated for b in single),
        'triplet_ownerless': any(freq[keys[c]] == 3 and ownfreq[keys[c]] == 1 for c in repeated),
        'diagonal': any(not _connected(m, ((0, 1), (0, -1), (1, 0), (-1, 0))) for m in p.values()),
        'slacks': [h-2-p[c].shape[0] for c, (_, _, h, w) in owners.items()],
    }


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _check_grid(pair['input']), _check_grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            assert np.array_equal(_render(info), out), 'cavity shape, destination, frame, or erasure is wrong'
            assert not np.array_equal(inp, out)
            occupied = np.zeros(inp.shape, dtype=bool)
            for c, (y, x, h, w) in info['plates'].items():
                tile = out[y:y+h, x:x+w]
                assert all(np.all(edge == c) for edge in (tile[0], tile[-1], tile[:, 0], tile[:, -1]))
                assert np.all((tile == c) | (tile == info['bg']))
                assert int((tile == info['bg']).sum()) == int(info['patterns'][c].sum())
                occupied[y:y+h, x:x+w] = True
            assert np.all(out[~occupied] == info['bg']), 'stencils were not removed'
            features = _features(info)
            assert 3 <= features['stencils'] <= 8 and 1 <= features['chambers'] <= 4
            low, high = (2, 4) if split == 'train' else (5, 6)
            assert all(low <= slack <= high for slack in features['slacks']), 'curriculum slack range'
            if split == 'test':
                assert features['mixed'], 'test must exercise both destinations'
            if split == 'train':
                training.append((info, out, features))
    fs = [f for _, _, f in training]
    assert sum(f['mixed'] for f in fs) >= 2
    for feature in ('ownerless_lift', 'quarter_turn', 'reflection_negative', 'ordinary_singleton',
                    'same_chamber_opposite', 'triplet_ownerless'):
        assert any(f[feature] for f in fs), f'missing teaching witness: {feature}'
    # Finite counterfactuals measure demonstration coverage. They are NOT an
    # independent solver or a proof against every conceivable alternative rule.
    alternatives = [dict(mode=m) for m in ('identity', 'half', 'dihedral', 'area', 'chambers')]
    alternatives += [dict(scope='owners'), dict(polarity='complement'), dict(mirror_holes=True)]
    alternatives += [dict(destination=d) for d in ('exactly_two', 'top', 'bottom', 'reversed')]
    for hypothesis in alternatives:
        assert any(not np.array_equal(_render(info, **hypothesis), out) for info, out, _ in training), str(hypothesis)


def _digest(puzzle):
    digest = sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                g = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(g.shape))
                digest.update(g.tobytes())
    return digest.hexdigest()


def self_test():
    """42 streamed seeds, actual-grid diversity, count boundaries, mutations."""
    hashes, shapes, normalized = set(), set(), set()
    backgrounds, stencil_counts, chamber_counts, mask_shapes, multiplicities = set(), set(), set(), set(), set()
    slacks = {'train': set(), 'test': set()}
    feature_counts = Counter()
    pair_count = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        hashes.add(_digest(puzzle))
        if seed == 0:
            assert _digest(puzzle) == _digest(generate_puzzle(seed))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                pair_count += 1
                info = _decode(pair['input'])
                f = _features(info)
                backgrounds.add(f['background'])
                stencil_counts.add(f['stencils'])
                chamber_counts.add(f['chambers'])
                multiplicities.update(f['multiplicities'])
                slacks[split].update(f['slacks'])
                mask_shapes.update(m.shape for m in info['patterns'].values())
                shapes.add(pair['input'].shape)
                roles = {}
                norm = bytes(roles.setdefault(int(c), len(roles)) for c in pair['input'].flat)
                normalized.add(sha256(bytes(pair['input'].shape)+norm).hexdigest())
                for name in ('ownerless_lift', 'quarter_turn', 'reflection_negative', 'ordinary_singleton',
                             'same_chamber_opposite', 'triplet_ownerless', 'diagonal'):
                    feature_counts[name] += bool(f[name])
                assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
    assert len(hashes) == 42 and len(normalized) > 190
    assert len(shapes) >= 20 and backgrounds == set(range(10))
    assert {3, 4, 5, 6, 7, 8} <= stencil_counts and chamber_counts == {1, 2, 3, 4}
    assert {1, 2, 3} <= multiplicities and len(mask_shapes) >= 15
    assert slacks['train'] == {2, 3, 4} and slacks['test'] == {5, 6}
    assert min(feature_counts.values()) >= 10
    # Actual INPUT mutations demonstrate ownership and nonlocal companionship.
    ownership_changed = company_changed = False
    for pair in generate_puzzle(0)['train']:
        inp = pair['input']
        info = _decode(inp)
        patterns, plates = info['patterns'], info['plates']
        keys = {c: _canon(m) for c, m in patterns.items()}
        freq = Counter(keys.values())
        external = np.ones(inp.shape, dtype=bool)
        for y, x, h, w in plates.values():
            external[y:y+h, x:x+w] = False
        for a in plates:
            for b in plates:
                if (plates[a][2:] == plates[b][2:] and patterns[a].shape == patterns[b].shape
                        and (freq[keys[a]] > 1) != (freq[keys[b]] > 1)):
                    swapped = inp.copy()
                    swapped[external & (inp == a)] = b
                    swapped[external & (inp == b)] = a
                    assert not np.array_equal(solve(swapped), pair['output'])
                    ownership_changed = True
            companions = [c for c in patterns if c != a and keys[c] == keys[a]]
            if companions and all(c not in plates for c in companions):
                removed = inp.copy()
                for c in companions:
                    removed[external & (inp == c)] = info['bg']
                changed = solve(removed)
                y, x, h, w = plates[a]
                assert not np.array_equal(changed[y:y+h, x:x+w], pair['output'][y:y+h, x:x+w])
                company_changed = True
    assert ownership_changed and company_changed
    # Fitting family members outside the sampler: nine-way companionship and a
    # tall one-stencil singleton, both with 17 surplus interior rows.
    wide_class = np.zeros((30, 30), dtype=np.int8)
    wide_class[1:21, 1:4] = 1
    wide_class[1:21, 6:9] = 2
    expected = wide_class.copy()
    expected[2, 2] = expected[2, 7] = 0
    for c in range(1, 10):
        wide_class[26, 2*c-1] = c
    assert np.array_equal(solve(wide_class), expected)
    lone = np.zeros((30, 10), dtype=np.int8)
    lone[1:21, 1:4] = 1
    expected_lone = lone.copy()
    lone[26, 1] = 1
    expected_lone[19, 2] = 0
    assert np.array_equal(solve(lone), expected_lone)
    # All legal explicit count combinations, without retaining their payloads.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(123456+nt*10+ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('accepted illegal counts')
    bad = generate_puzzle(314159, 2, 1)
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('accepted corrupted output')
    return {'seeds': 42, 'pairs': pair_count, 'distinct_puzzles': len(hashes),
            'input_shapes': len(shapes), 'normalized_inputs': len(normalized),
            'backgrounds': sorted(backgrounds), 'stencil_counts': sorted(stencil_counts),
            'chamber_counts': sorted(chamber_counts), 'mask_dimensions': sorted(mask_shapes),
            'multiplicities': sorted(multiplicities), 'train_slack': sorted(slacks['train']),
            'test_slack': sorted(slacks['test']), 'witness_pairs': dict(feature_counts),
            'outside_sampler_fixtures': ['nine_way_class', 'tall_singleton'],
            'input_mutations': {'ownership_changes_answer': ownership_changed,
                                'ownerless_company_changes_answer': company_changed}}
