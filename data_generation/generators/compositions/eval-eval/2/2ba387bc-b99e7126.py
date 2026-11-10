"""Rank-fed recursive mosaics: shape queues supply a stencil's two channels.

Only NumPy and the standard library are used. No examples, seed state, file
reads, or generation metadata are consulted by solve.
"""
from __future__ import annotations

import numpy as np


def _integer(value, name, lo, hi=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < lo or (hi is not None and value > hi):
        raise ValueError(f"illegal {name}")
    return value


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((a >= 0) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    """Eight-connected components, counting instances rather than colors."""
    todo = set(map(tuple, np.argwhere(mask)))
    result = []
    while todo:
        p = min(todo)
        todo.remove(p)
        stack, comp = [p], [p]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in todo:
                        todo.remove(q)
                        stack.append(q)
                        comp.append(q)
        result.append(comp)
    return result


def _transforms(mask):
    rotations = [np.rot90(mask, k) for k in range(4)]
    return rotations + [np.fliplr(m) for m in rotations]


def _asymmetric(mask):
    return all(not np.array_equal(mask, m) for m in _transforms(mask)[1:])


def _infer(value):
    a = _grid(value)
    bg = int(a[0, 0])
    assert np.all(a[0] == bg) and np.all(a[-1] == bg)
    assert np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)
    objects, size = [], None
    for comp in _components(a != bg):
        r = min(p[0] for p in comp)
        c = min(p[1] for p in comp)
        bottom = max(p[0] for p in comp) + 1
        right = max(p[1] for p in comp) + 1
        n = bottom - r
        assert 3 <= n <= 5 and n == right - c
        assert size is None or size == n
        size = n
        tile = a[r:bottom, c:right].copy()
        colors = np.unique(tile[tile != bg])
        assert len(colors) == 1, 'every source object is monochrome'
        color = int(colors[0])
        mask = tile != bg
        rim = np.ones((n, n), dtype=bool)
        rim[1:-1, 1:-1] = False
        if mask.all():
            kind = 'S'
        elif np.array_equal(mask, rim):
            kind = 'H'
        else:
            kind = 'G'
        # Complete square footprints have a one-cell background moat.
        halo = a[r-1:bottom+1, c-1:right+1].copy()
        halo[1:-1, 1:-1] = bg
        assert np.all(halo == bg), 'touching/overlapping footprints'
        objects.append(dict(top=int(r), left=int(c), kind=kind,
                            color=color, mask=mask))
    glyphs = [o for o in objects if o['kind'] == 'G']
    assert len(glyphs) == 1, 'one irregular stencil is required'
    glyph = glyphs[0]
    queues = [[o for o in sorted(objects, key=lambda o: (o['top'], o['left']))
               if o['kind'] == k] for k in ('H', 'S')]
    assert all(queues), 'both donor geometries must be present'
    active = list(map(tuple, np.argwhere(glyph['mask'])))
    assert max(map(len, queues)) == len(active), 'one rank per active address'
    return dict(grid=a, background=bg, size=size, objects=objects,
                glyph=glyph, mask=glyph['mask'], active=active, queues=queues)


def _mosaic(info, queues=None, outer_mask=None, inner_mask=None, addresses=None):
    """Each address holds the same geometry, with its own two color donors."""
    n, bg = info['size'], info['background']
    inner_mask = info['mask'] if inner_mask is None else inner_mask
    outer_mask = info['mask'] if outer_mask is None else outer_mask
    queues = info['queues'] if queues is None else queues
    out = np.full((n*n, n*n), bg, dtype=np.int8)
    addresses = np.argwhere(outer_mask) if addresses is None else addresses
    for rank, (r, c) in enumerate(addresses):
        hcolor = queues[0][rank]['color'] if rank < len(queues[0]) and queues[0][rank] is not None else bg
        scolor = queues[1][rank]['color'] if rank < len(queues[1]) and queues[1][rank] is not None else bg
        out[r*n:(r+1)*n, c*n:(c+1)*n] = np.where(inner_mask, hcolor, scolor)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the stencil and shape queues from this input alone."""
    return _mosaic(_infer(input_grid))


def _tie_witness(info):
    return any(a['top'] == b['top'] and a['color'] != b['color']
               for q in info['queues'] for j, a in enumerate(q) for b in q[j+1:])


def _alternatives(info):
    """Concrete shortcut diagnostics, not a search over every possible rule."""
    q = info['queues']
    alt = {}
    for label, key, reverse in (
        ('horizontal', lambda o: (o['left'], o['top']), False),
        ('horizontal_reverse', lambda o: (o['left'], o['top']), True),
    ):
        alt[label] = _mosaic(info, [sorted(xs, key=key, reverse=reverse) for xs in q])
    alt['reverse_H'] = _mosaic(info, [q[0][::-1], q[1]])
    alt['reverse_S'] = _mosaic(info, [q[0], q[1][::-1]])
    alt['swap_channels'] = _mosaic(info, q[::-1])
    alt['column_addresses'] = _mosaic(info, addresses=sorted(info['active'], key=lambda p: (p[1], p[0])))
    k = len(info['active'])
    alt['repeat_last'] = _mosaic(info, [xs + [xs[-1]]*(k-len(xs)) for xs in q])
    # Greedily match each hollow with its closest unused solid, then append
    # leftover solids. This preserves colors/counts but changes ordinal pairs.
    remaining, near = list(q[1]), []
    for hollow in q[0]:
        if remaining:
            donor = min(remaining, key=lambda o: (abs(o['top']-hollow['top']) +
                        abs(o['left']-hollow['left']), o['top'], o['left']))
            remaining = [o for o in remaining if o is not donor]
            near.append(donor)
    near.extend(remaining)
    alt['nearest_pairing'] = _mosaic(info, [q[0], near])
    # The review's greedy scan opens a fresh rank when its current shape slot
    # is full; it never returns to an earlier empty slot.
    rows = [[None, None]]
    for o in sorted((o for o in info['objects'] if o['kind'] != 'G'),
                    key=lambda o: (o['top'], o['left'])):
        col = int(o['kind'] == 'S')
        if rows[-1][col] is not None:
            rows.append([None, None])
        rows[-1][col] = o
    alt['greedy'] = _mosaic(info, [[row[j] for row in rows] for j in (0, 1)])
    alt['dense_outer'] = _mosaic(info, outer_mask=np.ones_like(info['mask']))
    alt['solid_inner'] = _mosaic(info, inner_mask=np.ones_like(info['mask']))
    # Plausible interpretations of the two scales' relative orientation.
    for k, m in enumerate(_transforms(info['mask'])[1:], 1):
        alt[f'outer_transform_{k}'] = _mosaic(info, outer_mask=m)
        alt[f'inner_transform_{k}'] = _mosaic(info, inner_mask=m)
    return alt


def _stencil_ok(m, minimum):
    n = m.shape[0]
    k = int(m.sum())
    capacity = ((29 // (n + 1)) ** 2 - 1) // 2
    rim = np.ones_like(m)
    rim[1:-1, 1:-1] = False
    return (minimum <= k <= min(n*n-1, capacity)
            and all(m.any(axis=axis)[[0, -1]].all() for axis in (0, 1))
            and len(_components(m)) == 1 and not np.array_equal(m, rim))


def _witness_mask(m):
    # A prefix-shaped blueprint cannot distinguish sparse addressing from
    # simply putting the donor ranks into the first K dense raster slots.
    dense_prefix = np.arange(m.size).reshape(m.shape) < int(m.sum())
    active = list(map(tuple, np.argwhere(m)))
    different_address_orders = active != sorted(active, key=lambda p: (p[1], p[0]))
    return _asymmetric(m) and not np.array_equal(m, dense_prefix) and different_address_orders


def _sample_stencil(rng, n, excluded, witness, minimum):
    cap = min(n*n-1, ((29 // (n+1))**2 - 1)//2)
    lo = max(minimum, n)
    for _ in range(384):
        k = int(rng.integers(lo, cap+1))
        m = np.zeros((n, n), dtype=bool)
        # Mix free subsets and grown patterns; neither split has an alphabet.
        if rng.random() < .6:
            m.flat[rng.choice(n*n, k, replace=False)] = True
        else:
            m[int(rng.integers(n)), int(rng.integers(n))] = True
            while int(m.sum()) < k:
                frontier = {(int(r)+dr, int(c)+dc) for r, c in np.argwhere(m)
                            for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                            if 0 <= r+dr < n and 0 <= c+dc < n and not m[r+dr, c+dc]}
                choices = sorted(frontier)
                r, c = choices[int(rng.integers(len(choices)))]
                m[r, c] = True
        if (_stencil_ok(m, minimum) and _key(m) not in excluded
                and (not witness or _witness_mask(m))):
            return m
    # Exhaustive, tiny finite fallback. More than ten eligible 3x3 designs
    # remain after any possible puzzle history. Geometry is always input-visible.
    for code in range(1, 511):
        m = np.array([(code >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        if (_stencil_ok(m, minimum) and _key(m) not in excluded
                and (not witness or _witness_mask(m))):
            return m
    raise RuntimeError('no stencil in the bounded fallback')


def _positions(rng, n, count, witness):
    cap = 29 // (n+1)
    legal_cols = [c for c in range(2 if witness else 1, cap+1)
                  if (count+c-1)//c <= cap]
    cols = int(rng.choice(legal_cols))
    rows = (count + cols - 1)//cols
    height = int(rng.integers(rows*(n+1)+1, 31))
    width = int(rng.integers(cols*(n+1)+1, 31))
    gaps = rng.multinomial(height-(rows*(n+1)+1), np.full(rows+1, 1/(rows+1)))
    points = []
    top = 1 + int(gaps[0])
    amounts = [cols]*rows
    amounts[-1] = count - cols*(rows-1)
    rng.shuffle(amounts)
    for row, amount in enumerate(amounts):
        hgap = rng.multinomial(width-(amount*(n+1)+1), np.full(amount+1, 1/(amount+1)))
        left = 1 + int(hgap[0])
        for j in range(amount):
            points.append((top, left))
            left += n+1+int(hgap[j+1])
        top += n+1+int(gaps[row+1])
    # Free moves remove the construction shelves when space permits. Footprint
    # separation remains mandatory even when a stencil has empty corner cells.
    rr, cc = np.indices((height-n-1, width-n-1))
    rr, cc = rr+1, cc+1
    for _ in range(2):
        for idx in rng.permutation(count):
            legal = np.ones(rr.shape, dtype=bool)
            for j, (r, c) in enumerate(points):
                if j != idx:
                    legal &= (np.abs(rr-r) > n) | (np.abs(cc-c) > n)
            sites = np.argwhere(legal)
            y, x = sites[int(rng.integers(len(sites)))]
            points[int(idx)] = (int(y)+1, int(x)+1)
    return (height, width), sorted(points)


def _draw(n, color, bg, kind, mask):
    if kind == 'G':
        return np.where(mask, color, bg).astype(np.int8)
    tile = np.full((n, n), color, dtype=np.int8)
    if kind == 'H':
        tile[1:-1, 1:-1] = bg
    return tile


def _make_pair(rng, mask, counts, bg, demonstration, forced_kind, witness):
    n, (hc, sc) = mask.shape[0], counts
    shape, positions = _positions(rng, n, hc+sc+1, witness)
    kinds = list(rng.permutation(['H']*hc + ['S']*sc))
    if witness:
        first = str(rng.choice(['H', 'S']))
        kinds.remove(first)
        kinds.remove(first)
        kinds = [first, first] + kinds
    glyph_index = int(rng.integers(len(kinds)+1))
    kinds.insert(glyph_index, 'G')
    palette = [c for c in range(10) if c != bg]
    if rng.random() < .5:
        palette = list(map(int, rng.choice(palette, int(rng.integers(2, 7)), replace=False)))
    colors = list(map(int, rng.choice(palette, len(kinds))))
    if forced_kind is not None:
        colors[kinds.index(forced_kind)] = demonstration
    if witness:
        for kind in ('H', 'S'):
            indices = [i for i, k in enumerate(kinds) if k == kind]
            if colors[indices[0]] == colors[indices[-1]]:
                colors[indices[-1]] = int(rng.choice([c for c in range(10)
                                                    if c not in (bg, colors[indices[0]])]))
    a = np.full(shape, bg, dtype=np.int8)
    for (r, c), kind, color in zip(positions, kinds, colors):
        a[r:r+n, c:c+n] = _draw(n, color, bg, kind, mask)
    info = _infer(a)
    return dict(input=a, output=_mosaic(info))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    num_train = _integer(num_train, 'num_train', 2, 10)
    num_test = _integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    rng = np.random.default_rng(seed)
    stable_size = bool(rng.random() < .25)
    stable_bg = bool(rng.random() < .25)
    base_size = int(rng.choice([3, 4, 5], p=[.45, .4, .15]))
    demonstration = int(rng.integers(10))
    base_bg = int(rng.choice([c for c in range(10) if c != demonstration]))
    first_sign = int(rng.choice([-1, 1]))
    puzzle, used, training_masks = dict(train=[], test=[]), set(), set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            first_two = split == 'train' and index < 2
            witness = split == 'train' and index == 0
            n = base_size if stable_size else int(rng.choice([3, 4, 5], p=[.45, .4, .15]))
            # First test asks for a new blueprint and a longer absent channel.
            minimum = 5 if split == 'test' and index == 0 else (4 if first_two else 3)
            excluded = training_masks if split == 'test' or (split == 'train' and index == 1) else set()
            mask = _sample_stencil(rng, n, excluded, witness, minimum)
            k = int(mask.sum())
            if first_two:
                gap = int(rng.integers(1, min(2, k-2)+1))
                sign = first_sign if index == 0 else -first_sign
                counts = (k-gap, k) if sign > 0 else (k, k-gap)
            elif split == 'test' and index == 0:
                gap = int(rng.integers(3, k))
                counts = (k-gap, k) if rng.integers(2) else (k, k-gap)
            elif rng.random() < .25:
                counts = (k, k)
            else:
                small = int(rng.integers(1, k+1))
                counts = (small, k) if rng.integers(2) else (k, small)
            bg = base_bg if stable_bg else int(rng.choice([c for c in range(10) if c != demonstration]))
            forced = ('H' if index == 0 else 'S') if first_two else None
            for _ in range(96):
                pair = _make_pair(rng, mask, counts, bg, demonstration, forced, witness)
                if _key(pair['input']) in used:
                    continue
                if witness:
                    info = _infer(pair['input'])
                    if not _tie_witness(info) or any(np.array_equal(pair['output'], alt) for alt in _alternatives(info).values()):
                        continue
                break
            else:
                raise RuntimeError('bounded layout/color witness sampling exhausted')
            used.add(_key(pair['input']))
            puzzle[split].append(pair)
            if split == 'train':
                training_masks.add(_key(mask))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check visible semantics, preservation, channel padding, and evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert _key(a) not in seen
            seen.add(_key(a))
            info = _infer(a)
            n, bg = info['size'], info['background']
            assert b.shape == (n*n, n*n)
            assert np.array_equal(b, _mosaic(info))
            # Check both scales and padding directly, with no generated metadata.
            q = info['queues']
            for r in range(n):
                for c in range(n):
                    tile = b[r*n:(r+1)*n, c*n:(c+1)*n]
                    if not info['mask'][r, c]:
                        assert np.all(tile == bg)
                    else:
                        rank = info['active'].index((r, c))
                        for channel, pixels in enumerate((info['mask'], ~info['mask'])):
                            expected = q[channel][rank]['color'] if rank < len(q[channel]) else bg
                            assert np.all(tile[pixels] == expected)
            if split == 'train':
                training.append(info)
    assert any(_tie_witness(i) for i in training), 'visible left-to-right tie convention'
    assert len({_key(i['mask']) for i in training}) >= 2, 'two demonstrated designs'
    differences = [len(i['queues'][1])-len(i['queues'][0]) for i in training]
    assert min(differences) < 0 < max(differences), 'both missing-channel directions must be taught'
    diagnostics = [(_mosaic(i), _alternatives(i)) for i in training]
    for name in diagnostics[0][1]:
        assert any(not np.array_equal(out, alternatives[name])
                   for out, alternatives in diagnostics), name
    roles = {}
    for info in training:
        for o in info['objects']:
            if o['kind'] != 'G':
                roles.setdefault(o['color'], set()).add(o['kind'])
    assert any(v == {'H', 'S'} for v in roles.values()), 'geometry/color role switch'
    train_masks = {_key(i['mask']) for i in training}
    tests = [_infer(p['input']) for p in puzzle['test']]
    assert _key(tests[0]['mask']) not in train_masks, 'new test blueprint'
    assert abs(len(tests[0]['queues'][0])-len(tests[0]['queues'][1])) >= 3


def self_test():
    """Bounded 42-seed structural variation and family-specific controls."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures, shapes, masks, layouts, counts, backgrounds, sizes = (set() for _ in range(7))
    ties = repeats = balanced = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        # A donor-color intervention must affect exactly its own rank/channel.
        first = puzzle['train'][0]
        info = _infer(first['input'])
        n = info['size']
        ar, ac = info['active'][0]
        for channel, q in enumerate(info['queues']):
            donor = q[0]
            changed = np.asarray(first['input']).copy()
            color = next(c for c in range(10) if c not in (info['background'], donor['color']))
            tile = changed[donor['top']:donor['top']+n, donor['left']:donor['left']+n]
            tile[tile != info['background']] = color
            prediction = solve(changed)
            actual_changes = set(map(tuple, np.argwhere(prediction != first['output'])))
            pixels = info['mask'] if channel == 0 else ~info['mask']
            expected_changes = {(ar*n+int(r), ac*n+int(c)) for r,c in np.argwhere(pixels)}
            assert actual_changes == expected_changes
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = np.asarray(pair['input'])
                snapshot = a.copy()
                assert np.array_equal(solve(a.copy()), pair['output'])
                assert np.array_equal(a, snapshot)
                i = _infer(a)
                shapes.add(a.shape)
                masks.add(_key(i['mask']))
                layouts.add((i['size'], tuple((o['top'], o['left'], o['kind']) for o in i['objects'])))
                counts.add(tuple(map(len, i['queues'])))
                backgrounds.add(i['background'])
                sizes.add(i['size'])
                balanced += len(i['queues'][0]) == len(i['queues'][1])
                for q in i['queues']:
                    ties += len({o['top'] for o in q}) < len(q)
                    repeats += len({o['color'] for o in q}) < len(q)
                # Recoloring a stencil-only ink does not change its meaning.
                changed = a.copy()
                g = i['glyph']
                r, c, n = g['top'], g['left'], i['size']
                tile = changed[r:r+n, c:c+n]
                tile[i['mask']] = next(x for x in range(10) if x not in (i['background'], g['color']))
                assert np.array_equal(solve(changed), pair['output'])
    assert len(signatures) == 42
    assert sizes == {3, 4, 5} and len(backgrounds) >= 8
    assert len(shapes) >= 35 and len(layouts) >= 150
    assert len(masks) >= 55 and len(counts) >= 20
    assert ties >= 15 and repeats >= 40 and balanced >= 10
    assert any(len({_infer(p['input'])['size'] for s in ('train', 'test') for p in puzzle[s]}) == 1 for puzzle in puzzles)
    assert any(len({_infer(p['input'])['size'] for p in puzzle['train']}) > 1 for puzzle in puzzles)
    # All 30 legal explicit count combinations, with repeatability at boundaries.
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            p = generate_puzzle(701 + train*10 + test, train, test)
            assert len(p['train']) == train and len(p['test']) == test
    for train, test in ((2, 1), (10, 4)):
        p, q = generate_puzzle(99, train, test), generate_puzzle(99, train, test)
        assert all(_key(a[k]) == _key(b[k]) for s in ('train', 'test')
                   for a, b in zip(p[s], q[s]) for k in ('input', 'output'))
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Concrete negative controls for output channel semantics and source parsing.
    p = puzzles[0]
    damaged = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in p[s]] for s in ('train', 'test')}
    i = _infer(damaged['test'][0]['input'])
    missing_channel = int(len(i['queues'][0]) > len(i['queues'][1]))
    rank = min(map(len, i['queues']))
    r, c = i['active'][rank]
    y, x = np.argwhere(i['mask'] if missing_channel == 0 else ~i['mask'])[0]
    damaged['test'][0]['output'][r*i['size']+y, c*i['size']+x] = i['queues'][1-missing_channel][0]['color']
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('incorrect missing-channel fill accepted')
    bad = np.asarray(p['train'][0]['input']).copy()
    bad[0, 0] = (int(bad[0, 0])+1) % 10
    try:
        solve(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('broken background margin accepted')
    return {'seeds': 42, 'pairs': sum(len(p[s]) for p in puzzles for s in ('train', 'test')),
            'distinct_puzzles': len(signatures), 'input_shapes': len(shapes),
            'stencils': len(masks), 'layouts': len(layouts), 'count_pairs': len(counts),
            'sizes': list(map(int, sorted(sizes))), 'backgrounds': sorted(backgrounds),
            'same_class_row_ties': ties, 'repeated_color_queues': repeats,
            'balanced_examples': balanced}
