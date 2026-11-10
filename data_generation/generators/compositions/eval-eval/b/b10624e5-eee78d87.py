"""Registered moire: mirrored periodic prints, with overlap-sensitive ink.

The divider color doubles as the overlap ink. Everything needed by solve is
visible in one input; there are no puzzle-specific learned color or size rules.
Construction stamps repeated blocks; solve pulls phase coordinates instead.
"""

from itertools import product
import numpy as np


def _full_box(m):
    return bool(m[1, 1] and 3 <= int(m.sum()) <= 8 and
                m[0].any() and m[2].any() and m[:, 0].any() and m[:, 2].any())


_MASKS = tuple(m for code in range(512)
              if _full_box(m := np.array([(code >> k) & 1 for k in range(9)],
                                        dtype=bool).reshape(3, 3)))


def _asymmetric(m):
    return all(not np.array_equal(m, other) for other in
               (m[::-1], m[:, ::-1], m[::-1, ::-1], m.T,
                m.T[::-1], m.T[:, ::-1], m.T[::-1, ::-1]))


def _key(grid):
    g = np.asarray(grid)
    return g.shape, g.astype(np.uint8).tobytes()


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(rng, train, test):
    for value, name, low, high in ((train, 'num_train', 2, 10),
                                   (test, 'num_test', 1, 4)):
        if value is not None:
            value = _integer(value, name)
            if not low <= value <= high:
                raise ValueError(f'{name} must be in {low}..{high}')
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(test + 1, 11))
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError('num_train must exceed num_test')
    return int(train), int(test)


def _square_components(panel, color):
    pending = {tuple(map(int, p)) for p in np.argwhere(panel == color)}
    squares = []
    while pending:
        start = min(pending)
        pending.remove(start)
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in pending:
                    pending.remove(p)
                    component.add(p)
                    stack.append(p)
        rows, cols = zip(*component)
        top, left, side = min(rows), min(cols), max(rows)-min(rows)+1
        assert max(cols)-left+1 == side and len(component) == side*side
        squares.append((top, left, side))
    return sorted(squares)


def _decode(input_grid):
    g = np.asarray(input_grid)
    assert g.ndim == 2 and all(1 <= n <= 30 for n in g.shape)
    assert np.issubdtype(g.dtype, np.integer) and np.all((g >= 0) & (g <= 9))
    colors, frequencies = np.unique(g, return_counts=True)
    bg = int(colors[np.argmax(frequencies)])
    assert int(frequencies.max()) > g.size // 2, 'background must be a strict majority'
    lines = [('h', r, int(g[r, 0])) for r in range(1, g.shape[0]-1)
             if g[r, 0] != bg and np.all(g[r] == g[r, 0])]
    lines += [('v', c, int(g[0, c])) for c in range(1, g.shape[1]-1)
              if g[0, c] != bg and np.all(g[:, c] == g[0, c])]
    assert len(lines) == 1, 'one full non-background divider is required'
    axis, index, overlap = lines[0]
    if axis == 'v':
        bounds = ((0, g.shape[0], 0, index),
                  (0, g.shape[0], index+1, g.shape[1]))
    else:
        bounds = ((0, index, 0, g.shape[1]),
                  (index+1, g.shape[0], 0, g.shape[1]))
    panels = [g[a:b, c:d] for a, b, c, d in bounds]
    sets = [set(map(int, np.unique(p))) for p in panels]
    common = (sets[0] & sets[1]) - {bg}
    assert len(common) == 1
    marker = next(iter(common))
    refs = [i for i, values in enumerate(sets) if len(values) == 3]
    assert len(refs) == 1
    ref = refs[0]
    assert sets[1-ref] == {bg, marker}
    ink, = sets[ref] - {bg, marker}
    assert len({bg, marker, ink, overlap}) == 4
    reference = panels[ref]
    ref_squares = _square_components(reference, marker)
    assert len(ref_squares) == 1
    r, c, unit = ref_squares[0]
    top, left = r-unit, c-unit
    assert top >= 0 and left >= 0
    patch = reference[top:top+3*unit, left:left+3*unit]
    assert patch.shape == (3*unit, 3*unit)
    mask = np.zeros((3, 3), dtype=bool)
    for y, x in product(range(3), repeat=2):
        block = patch[y*unit:(y+1)*unit, x*unit:(x+1)*unit]
        assert np.all(block == block[0, 0])
        value = int(block[0, 0])
        assert value in ({marker} if (y, x) == (1, 1) else {bg, ink})
        mask[y, x] = value != bg
    assert _full_box(mask)
    reconstructed = np.full(reference.shape, bg, dtype=g.dtype)
    reconstructed[top:top+3*unit, left:left+3*unit] = patch
    assert np.array_equal(reference, reconstructed), 'unrelated swatch detail'
    target = panels[1-ref]
    anchors = _square_components(target, marker)
    assert 2 <= len(anchors) <= 4
    # A carrier cell separates every pair of registration marks, even diagonally.
    for i, (ar, ac, s) in enumerate(anchors):
        for br, bc, t in anchors[:i]:
            assert ar > br+t or br > ar+s or ac > bc+t or bc > ac+s
    return dict(grid=g, bg=bg, marker=marker, ink=ink, overlap=overlap,
                axis=axis, ref=ref, bounds=bounds, reference=reference,
                target=target, mask=mask, unit=unit, anchors=anchors)


def _pull_counts(info, *, reflection=True, scaling=True, registration=True,
                 periodic=True):
    """Inverse-coordinate field evaluation, including negative floor division."""
    h, w = info['target'].shape
    rows, cols = np.indices((h, w))
    result = np.zeros((h, w), dtype=np.int8)
    for r, c, side in info['anchors']:
        if not scaling:
            side = info['unit']
        if not registration:
            r = c = 0
        y, x = (rows-r)//side, (cols-c)//side
        iy, ix = (y+1) % 3, (x+1) % 3
        if reflection:
            if info['axis'] == 'h':
                iy = 2-iy
            else:
                ix = 2-ix
        occupancy = info['mask'][iy, ix]
        if not periodic:
            occupancy &= (y >= -1) & (y <= 1) & (x >= -1) & (x <= 1)
        result += occupancy
    return result


def _stamp_counts(mask, shape, anchors):
    """Construction in the canonical left-swatch layout: stamp mirrored tiles."""
    counts = np.zeros(shape, dtype=np.int8)
    mirrored = mask[:, ::-1]
    for r, c, side in anchors:
        period = 3*side
        for dy, dx in np.argwhere(mirrored):
            y0 = (r+(int(dy)-1)*side) % period
            x0 = (c+(int(dx)-1)*side) % period
            for y in range(y0-period, shape[0], period):
                for x in range(x0-period, shape[1], period):
                    a, b = max(0, y), min(shape[0], y+side)
                    d, e = max(0, x), min(shape[1], x+side)
                    if a < b and d < e:
                        counts[a:b, d:e] += 1
    return counts


def _inks(counts, bg, ink, overlap):
    out = np.full(counts.shape, bg, dtype=np.int8)
    out[counts == 1] = ink
    out[counts >= 2] = overlap
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve only from the supplied grid. No generator state is consulted."""
    info = _decode(input_grid)
    result = info['grid'].copy()
    a, b, c, d = info['bounds'][1-info['ref']]
    result[a:b, c:d] = _inks(_pull_counts(info), info['bg'], info['ink'], info['overlap'])
    return result


def _ablations(info):
    correct = _inks(_pull_counts(info), info['bg'], info['ink'], info['overlap'])
    differences = {}
    for name, kwargs in (('mirror', {'reflection': False}),
                         ('scale', {'scaling': False}),
                         ('phase', {'registration': False}),
                         ('repeat', {'periodic': False})):
        wrong = _inks(_pull_counts(info, **kwargs), info['bg'], info['ink'], info['overlap'])
        differences[name] = int(np.count_nonzero(correct != wrong))
    count = _pull_counts(info)
    differences['overlap'] = int(np.count_nonzero(count >= 2))
    differences['source'] = min(
        int(np.count_nonzero(np.minimum(count, 2) != np.minimum(
            count-_pull_counts(dict(info, anchors=[anchor])), 2)))
        for anchor in info['anchors'])
    # The old B error: an independent central rectangle overwrites carrier gaps.
    wrong = _inks(count, info['bg'], info['ink'], info['ink'])
    h, w = wrong.shape
    wrong[(h-6)//2:(h-6)//2+6, (w-6)//2:(w-6)//2+6] = info['overlap']
    differences['fixed_highlight'] = int(np.count_nonzero(correct != wrong))
    return differences


def _informative(info, witness):
    counts = _pull_counts(info)
    area = counts.size
    # Keep the three states legible, rather than accepting almost-solid moire.
    if not (np.count_nonzero(counts == 0) >= max(2, .03*area) and
            np.count_nonzero(counts == 1) >= .12*area and
            np.count_nonzero(counts >= 2) >= .10*area):
        return False
    if any(np.array_equal(np.minimum(counts, 2), np.minimum(
            counts-_pull_counts(dict(info, anchors=[anchor])), 2))
           for anchor in info['anchors']):
        return False
    if not witness:
        return True
    return (np.any(counts == 0) and all(_ablations(info).values()) and
            (len(info['anchors']) < 3 or np.any(counts >= 3)))


def _position(rng, h, w, side):
    # Interior and boundary placement are independent of motif and split.
    r, c = int(rng.integers(h-side+1)), int(rng.integers(w-side+1))
    if rng.random() < .35:
        r = int(rng.choice([0, h-side]))
    if rng.random() < .35:
        c = int(rng.choice([0, w-side]))
    return r, c, side


def _construct(rng, mask, unit, number, larger, palette, rotation):
    bg, divider, marker, ink = palette
    strip = 3*unit + int(rng.integers(2, 6))
    h = int(rng.integers(11, 30))
    w = int(rng.integers(11, 30-strip))
    sides = [1, 2] + [int(rng.integers(1, 3)) for _ in range(number-2)]
    if larger:
        sides[-1] = 3
    rng.shuffle(sides)
    anchors = []
    for side in sorted(sides, reverse=True):
        candidates = [_position(rng, h, w, side) for _ in range(80)]
        # Complete finite enumeration handles unlucky boundary draws.
        candidates += [(r, c, side) for r in range(h-side+1) for c in range(w-side+1)]
        for r, c, s in candidates:
            if all(r > y+t or y > r+s or c > x+t or x > c+s for y, x, t in anchors):
                anchors.append((r, c, s))
                break
        else:
            raise AssertionError('registration placement exhausted')
    inp = np.full((h, strip+1+w), bg, dtype=np.int8)
    inp[:, strip] = divider
    top = int(rng.integers(h-3*unit+1))
    left = int(rng.integers(strip-3*unit+1))
    tile = np.where(mask, ink, bg).astype(np.int8)
    tile[1, 1] = marker
    inp[top:top+3*unit, left:left+3*unit] = np.repeat(np.repeat(tile, unit, 0), unit, 1)
    for r, c, side in anchors:
        inp[r:r+side, strip+1+c:strip+1+c+side] = marker
    out = inp.copy()
    out[:, strip+1:] = _inks(_stamp_counts(mask, (h, w), anchors), bg, ink, divider)
    return {'input': np.rot90(inp, rotation).copy(),
            'output': np.rot90(out, rotation).copy()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed')
    if seed < 0:
        raise ValueError('seed must be nonnegative')
    rng = np.random.default_rng(seed)
    ntrain, ntest = _counts(rng, num_train, num_test)
    palette = tuple(map(int, rng.permutation(10)[:4]))
    mixed_palette = bool(rng.integers(2))
    count_shift = bool(rng.integers(2))
    first_rotation = int(rng.integers(4))
    puzzle = {'train': [], 'test': []}
    used_inputs, used_masks = set(), set()
    first_density = None
    for split, number_pairs in (('train', ntrain), ('test', ntest)):
        for index in range(number_pairs):
            witness = split == 'train' and index < 2
            number = (2+index if witness else 4 if split == 'test' and count_shift
                      else int(rng.integers(2, 4)))
            unit = 1+index if witness else int(rng.integers(1, 3))
            rotation = ((first_rotation+index) % 4 if witness else int(rng.integers(4)))
            larger = split == 'test' and not count_shift
            eligible = [m for m in _MASKS if m.tobytes() not in used_masks and
                        (not witness or _asymmetric(m)) and
                        (not witness or index == 0 or int(m.sum()) != first_density)]
            local_palette = tuple(map(int, rng.permutation(10)[:4])) if mixed_palette else palette
            for attempt in range(512):
                densities = sorted({int(m.sum()) for m in eligible})
                density = int(rng.choice(densities))
                choices = [m for m in eligible if m.sum() == density]
                mask = choices[int(rng.integers(len(choices)))]
                pair = _construct(rng, mask, unit, number, larger, local_palette, rotation)
                if _key(pair['input']) in used_inputs:
                    continue
                info = _decode(pair['input'])
                if not _informative(info, witness):
                    continue
                assert np.array_equal(solve(pair['input']), pair['output'])
                break
            else:
                raise RuntimeError('bounded informative-example search exhausted')
            puzzle[split].append(pair)
            used_inputs.add(_key(pair['input']))
            # Mask novelty is modulo the actual swatch orientation, not color.
            used_masks.update(np.rot90(mask, k).tobytes() for k in range(4))
            if witness and index == 0:
                first_density = int(mask.sum())
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check returned arrays, geometric roles, overlap arithmetic, and evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, witnesses, train_infos, test_infos = set(), [], [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = (np.asarray(pair[k]) for k in ('input', 'output'))
            for grid in (inp, out):
                assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            assert np.array_equal(solve(inp), out)
            a, b, c, d = info['bounds'][info['ref']]
            assert np.array_equal(out[a:b, c:d], inp[a:b, c:d])
            a, b, c, d = info['bounds'][1-info['ref']]
            target = out[a:b, c:d]
            counts = _pull_counts(info)
            assert np.array_equal(target == info['bg'], counts == 0)
            assert np.array_equal(target == info['ink'], counts == 1)
            assert np.array_equal(target == info['overlap'], counts >= 2)
            assert not np.any(target == info['marker'])
            assert _informative(info, False)
            if split == 'train':
                train_infos.append(info)
                if _asymmetric(info['mask']) and _informative(info, True):
                    witnesses.append(info)
            else:
                test_infos.append(info)
    assert {i['axis'] for i in witnesses} == {'h', 'v'}, 'both divider directions need evidence'
    assert len({int(i['mask'].sum()) for i in witnesses}) >= 2
    assert any(len(i['anchors']) == 3 and np.any(_pull_counts(i) >= 3) for i in witnesses)
    assert {i['unit'] for i in witnesses} == {1, 2}
    assert all(s <= 2 for i in train_infos for _, _, s in i['anchors'])
    # Every held-out pair increases source count OR introduces scale three.
    assert all(len(i['anchors']) > max(len(t['anchors']) for t in train_infos) or
               any(s == 3 for _, _, s in i['anchors']) for i in test_infos)
    train_masks = {np.rot90(i['mask'], k).tobytes() for i in train_infos for k in range(4)}
    assert all(i['mask'].tobytes() not in train_masks for i in test_infos)


def _fingerprint(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def self_test():
    """Bounded semantic/variation checks across 42 seeds and all count pairs."""
    masks, shapes, palettes, sides, locations, sources, outputs = (set() for _ in range(7))
    shifts, fingerprints = set(), set()
    for seed in range(42):
        ntest = 1+seed % 4
        ntrain = ntest+1+(seed//4) % (10-ntest)
        puzzle = generate_puzzle(seed, ntrain, ntest)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        if seed < 3:
            assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed, ntrain, ntest))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                counts = _pull_counts(info)
                # Independently stamp after mapping horizontal layouts back to vertical.
                if info['axis'] == 'v':
                    stamp = _stamp_counts(info['mask'], counts.shape, info['anchors'])
                else:
                    stamp = _stamp_counts(info['mask'].T, counts.T.shape,
                                          [(c, r, s) for r, c, s in info['anchors']]).T
                assert np.array_equal(counts, stamp)
                masks.add(info['mask'].tobytes())
                shapes.add(pair['input'].shape)
                palettes.add((info['bg'], info['overlap'], info['marker'], info['ink']))
                sources.add(len(info['anchors']))
                locations.add((info['axis'], info['ref']))
                sides.update(s for r, c, s in info['anchors'])
                outputs.add(_key(pair['output']))
                if split == 'test':
                    shifts.add('scale' if any(s == 3 for r, c, s in info['anchors']) else 'count')
    assert len(fingerprints) == 42 and len(masks) >= 70
    assert len(shapes) >= 70 and len(palettes) >= 35 and len(outputs) >= 200
    assert sides == {1, 2, 3} and sources == {2, 3, 4}
    assert len(locations) == 4 and shifts == {'scale', 'count'}
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            p = generate_puzzle(1000+10*train+test, train, test)
            assert (len(p['train']), len(p['test'])) == (train, test)
    for args in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, False), ('4', 1)):
        try:
            generate_puzzle(99, *args)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p = generate_puzzle(2026, 2, 1)
    assert _fingerprint(p) == _fingerprint(generate_puzzle(2026, 2, 1))
    # Return-label corruption must be detected, including carrier holes.
    info = _decode(p['test'][0]['input'])
    a, b, c, d = info['bounds'][1-info['ref']]
    r, x = map(int, np.argwhere(_pull_counts(info) == 0)[0])
    p['test'][0]['output'][a+r, c+x] = info['marker']
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted label was accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed')
