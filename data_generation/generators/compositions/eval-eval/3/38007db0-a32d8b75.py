"""Differential toggle mosaics. All decoding conventions are visible or fixed.

The repeated tile supplies the texture. Differences in each odd tile supply a
small stencil, controlled by that band's two bounded glyphs. Stencils toggle
whole texture blocks; an even number of requests cancels. No parent imports,
fixtures, file access, hidden palette dictionary, or seed-dependent solving.
"""

from collections import Counter

import numpy as np


CORNERS = ((0, 0), (0, 1), (1, 1), (1, 0))
NEIGHBORS = ((0, 1), (1, 2), (2, 1), (1, 0))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(int, copy=False)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def _crop(mask):
    points = np.argwhere(mask)
    if not len(points):  # Used only when testing competing interpretations.
        return np.zeros((1, 1), bool)
    lo, hi = points.min(0), points.max(0) + 1
    return mask[lo[0]:hi[0], lo[1]:hi[1]].copy()


def _decode(input_grid):
    a = _grid(input_grid)
    lattice = int(a[0, 0])
    rows = np.flatnonzero(np.all(a == lattice, axis=1))
    cols = np.flatnonzero(np.all(a == lattice, axis=0))
    assert len(rows) >= 2 and len(cols) >= 6
    assert rows[0] == cols[0] == 0
    assert rows[-1] == a.shape[0] - 1 and cols[-1] == a.shape[1] - 1
    n = int(rows[1] - 1)
    assert 3 <= n <= 5 and np.all(np.diff(rows) == n + 1)
    copies = len(cols) - 3
    assert copies >= 3
    assert np.all(np.diff(cols[:-2]) == n + 1)
    assert np.all(np.diff(cols[-3:]) == 4)
    table, controls, palette = [], [], None
    for top, bottom in zip(rows, rows[1:]):
        tiles = [a[top + 1:bottom, left + 1:right].copy()
                 for left, right in zip(cols[:copies], cols[1:copies + 1])]
        assert all(t.shape == (n, n) and np.all(t != lattice) for t in tiles)
        glyphs = {}
        for left, right in zip(cols[-3:], cols[-2:]):
            zone = a[top + 1:bottom, left + 1:right]
            points = np.argwhere(zone != lattice)
            assert len(points) == 9
            lo, hi = points.min(0), points.max(0) + 1
            box = zone[lo[0]:hi[0], lo[1]:hi[1]]
            assert box.shape == (3, 3) and np.all(box != lattice)
            values, counts = np.unique(box, return_counts=True)
            assert len(values) == 2 and max(counts) in (7, 8)
            carrier = int(values[np.argmax(counts)])
            ink = int(values[np.argmin(counts)])
            if palette is None:
                palette = (carrier, ink)
            assert palette == (carrier, ink)
            mark = set(map(tuple, np.argwhere(box == ink)))
            if len(mark) == 1:
                position = next(iter(mark))
                corners = tuple((2 * r, 2 * c) for r, c in CORNERS)
                assert position in corners and 'anchor' not in glyphs
                glyphs['anchor'] = corners.index(position)
            else:
                assert len(mark) == 2 and (1, 1) in mark and 'turn' not in glyphs
                position = next(iter(mark - {(1, 1)}))
                assert position in NEIGHBORS
                glyphs['turn'] = NEIGHBORS.index(position)
        assert set(glyphs) == {'turn', 'anchor'}
        table.append(tiles)
        controls.append((glyphs['turn'], glyphs['anchor']))
    carrier, ink = palette
    assert len({lattice, carrier, ink}) == 3
    assert set(map(int, np.unique(a))) == {lattice, carrier, ink}
    counts = Counter(_key(t) for band in table for t in band)
    best = max(counts.values())
    winners = [key for key, count in counts.items() if count == best]
    assert len(winners) == 1, 'the repeated template must be unique'
    master = next(t for band in table for t in band if _key(t) == winners[0])
    assert set(map(int, np.unique(master))) == {carrier, ink}
    entries = []
    for tiles, (turn, anchor) in zip(table, controls):
        odd = [i for i, tile in enumerate(tiles) if not np.array_equal(tile, master)]
        assert len(odd) == 1, 'each band needs exactly one visible exception'
        variant = tiles[odd[0]]
        assert set(map(int, np.unique(variant))) == {carrier, ink}
        delta = variant != master
        assert 1 <= delta.sum() <= int(.65 * n * n)
        entries.append(dict(variant=variant, delta=delta, mask=_crop(delta),
                            turn=turn, anchor=anchor, column=odd[0]))
    return dict(n=n, copies=copies, bands=len(table), lattice=lattice,
                carrier=carrier, ink=ink, master=master, table=table,
                entries=entries)


def _forward_counts(n, instructions):
    counts = np.zeros((n, n), int)
    for mask, turn, anchor in instructions:
        rotated = np.rot90(mask, -turn)
        h, w = rotated.shape
        assert h <= n and w <= n
        ar, ac = CORNERS[anchor]
        top, left = ar * (n - h), ac * (n - w)
        counts[top:top + h, left:left + w] += rotated
    return counts


def _expand(master, counts, carrier, ink, union=False):
    n = len(master)
    base = np.tile(master, (n, n))
    active = counts > 0 if union else counts % 2 == 1
    pixels = np.repeat(np.repeat(active, n, axis=0), n, axis=1)
    swapped = np.where(base == carrier, ink, carrier)
    return np.where(pixels, swapped, base).astype(int)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from pixels only, independently of sampling and demonstrations."""
    info = _decode(input_grid)
    instructions = [(e['mask'], e['turn'], e['anchor']) for e in info['entries']]
    counts = _forward_counts(info['n'], instructions)
    return _expand(info['master'], counts, info['carrier'], info['ink'])


def _corner_map(anchor, transform):
    r, c = CORNERS[anchor]
    if transform >= 4:
        c = 1 - c
    for _ in range(transform % 4):
        r, c = c, 1 - r
    return r, c


def _reference_counts(info, convention=(1, 0, 0), behavior='normal'):
    """Inverse coordinate check; does not call the forward rotation/placement.

    Also supports explicit plausible alternatives for training evidence checks.
    The alternatives are tests, never branches of the family's solver.
    """
    sign, offset, transform = convention
    n = info['n']
    counts = np.zeros((n, n), int)
    for index, e in enumerate(info['entries']):
        mask = e['mask']
        if behavior == 'ink_stencil':
            mask = _crop(e['variant'] == info['ink'])
        elif behavior == 'only_additions':
            mask = _crop(e['delta'] & (info['master'] == info['carrier']))
        elif behavior == 'only_removals':
            mask = _crop(e['delta'] & (info['master'] == info['ink']))
        elif behavior == 'first_column':
            mask = _crop(info['table'][index][0] != info['master'])
        elif behavior == 'last_column':
            mask = _crop(info['table'][index][-1] != info['master'])
        elif behavior == 'no_crop':
            mask = e['delta']
        elif behavior == 'fill_holes':
            mask = np.ones(mask.shape, bool)
        elif behavior == 'last_band' and index != info['bands'] - 1:
            continue
        k = (sign * e['turn'] + offset) % 4
        sh, sw = mask.shape
        h, w = (sw, sh) if k % 2 else (sh, sw)
        ar, ac = _corner_map(e['anchor'], transform)
        top, left = ar * (n - h), ac * (n - w)
        for r in range(n):
            for c in range(n):
                y, x = r - top, c - left
                if not (0 <= y < h and 0 <= x < w):
                    continue
                if k == 0:
                    sy, sx = y, x
                elif k == 1:
                    sy, sx = sh - 1 - x, y
                elif k == 2:
                    sy, sx = sh - 1 - y, sw - 1 - x
                else:
                    sy, sx = x, sw - 1 - y
                counts[r, c] += int(mask[sy, sx])
    return counts


def _scalar_output(info, counts):
    n, master = info['n'], info['master']
    out = np.empty((n * n, n * n), int)
    for r in range(n * n):
        for c in range(n * n):
            value = int(master[r % n, c % n])
            if counts[r // n, c // n] % 2:
                value = info['ink'] if value == info['carrier'] else info['carrier']
            out[r, c] = value
    return out


def _wrong_texture(info, rotate=False):
    """Two competing substitution rules, combined with the same XOR convention.

    A stamp's pixel changes relative to the master are composed by XOR. Using
    the exception as texture, or rotating the texture with the stencil, would
    produce these outputs. The correct stamp exchanges the unrotated master.
    """
    master, n = info['master'], info['n']
    master_bits = master == info['ink']
    bits = np.tile(master_bits, (n, n))
    for e in info['entries']:
        footprint = _forward_counts(n, [(e['mask'], e['turn'], e['anchor'])]) > 0
        texture = np.rot90(master, -e['turn']) if rotate else e['variant']
        target_bits = texture == info['carrier']
        bits ^= np.kron(footprint, master_bits != target_bits)
    return np.where(bits, info['ink'], info['carrier'])


def _training_evidence(infos):
    parities = [_reference_counts(i) % 2 for i in infos]
    survivors = []
    for sign in (1, -1):
        for offset in range(4):
            for transform in range(8):
                convention = (sign, offset, transform)
                if all(np.array_equal(_reference_counts(i, convention) % 2, p)
                       for i, p in zip(infos, parities)):
                    survivors.append(convention)
    assert survivors == [(1, 0, 0)], 'demonstrations leave control meanings ambiguous'
    for behavior in ('ink_stencil', 'only_additions', 'only_removals',
                     'first_column', 'last_column', 'no_crop', 'fill_holes', 'last_band'):
        assert any(not np.array_equal(_reference_counts(i, behavior=behavior) % 2, p)
                   for i, p in zip(infos, parities)), f'undistinguished shortcut: {behavior}'
    counts = [_reference_counts(i) for i in infos]
    assert any(np.any((c > 0) & (c % 2 == 0)) for c in counts), 'show cancellation'
    assert any(c.max() >= 3 for c in counts), 'show three-way overlap'
    assert any(not np.array_equal(c > 0, p) for c, p in zip(counts, parities))
    assert any(np.any(p) for p in parities), 'show exchange, not just cancellation'
    edits = [(int(np.sum(e['delta'] & (i['master'] == i['carrier']))),
              int(np.sum(e['delta'] & (i['master'] == i['ink']))))
             for i in infos for e in i['entries']]
    assert any(a == r > 0 for a, r in edits), 'show a count-preserving exception'
    assert any(len({e['column'] for e in i['entries']}) > 1 for i in infos)
    assert any(not np.array_equal(i['master'], np.rot90(i['master'])) for i in infos)
    for rotate in (False, True):
        assert any(not np.array_equal(_wrong_texture(i, rotate),
                                      _scalar_output(i, c))
                   for i, c in zip(infos, counts)), 'undistinguished texture substitution'


def _master(rng, n, carrier, ink, witness):
    for _ in range(40):
        bits = rng.random((n, n)) < rng.uniform(.20, .80)
        if not bits.any() or bits.all():
            continue
        if witness and np.array_equal(bits, np.rot90(bits)):
            continue
        if not witness and rng.random() < .25:
            bits |= bits[:, ::-1]
            if bits.all():
                continue
        return np.where(bits, ink, carrier).astype(int)
    bits = np.zeros((n, n), bool)
    bits[0, 1] = True
    return np.where(bits, ink, carrier).astype(int)


def _delta(rng, n, witness=False):
    for _ in range(40):
        low, high = (2, n) if witness else (1, n + 1)
        h, w = int(rng.integers(low, high)), int(rng.integers(low, high))
        rr, cc = np.indices((h, w))
        style = int(rng.integers(5))
        if style == 0:
            mask = rng.random((h, w)) < rng.uniform(.20, .85)
        elif style == 1:
            mask = (rr == int(rng.integers(h))) | (cc == int(rng.integers(w)))
        elif style == 2:
            mask = (rr == 0) | (rr == h - 1) | (cc == 0) | (cc == w - 1)
            mask &= rng.random((h, w)) > .15
        elif style == 3:
            mask = ((rr + cc) % 2 == int(rng.integers(2)))
        else:
            mask = np.ones((h, w), bool)
            mask[int(rng.integers(h)), int(rng.integers(w))] = False
        mask = _crop(mask)
        if not 1 <= mask.sum() <= int(.65 * n * n):
            continue
        if witness and (min(mask.shape) < 2 or mask.all()
                        or np.array_equal(mask, np.rot90(mask, 2))):
            continue
        h, w = mask.shape
        top, left = int(rng.integers(n - h + 1)), int(rng.integers(n - w + 1))
        result = np.zeros((n, n), bool)
        result[top:top + h, left:left + w] = mask
        return result
    result = np.zeros((n, n), bool)
    r, c = int(rng.integers(n - 1)), int(rng.integers(n - 1))
    result[r:r + 2, c:c + 2] = [[1, 1], [1, 0]]
    return result


def _assemble(master, variants, columns, controls, copies, palette, rng):
    n, bands = len(master), len(variants)
    lattice, carrier, ink = palette
    data_width = copies * (n + 1) + 1
    a = np.full((bands * (n + 1) + 1, data_width + 8), lattice, int)
    for b, (variant, col, (turn, anchor)) in enumerate(zip(variants, columns, controls)):
        top = 1 + b * (n + 1)
        for c in range(copies):
            left = 1 + c * (n + 1)
            a[top:top + n, left:left + n] = variant if c == col else master
        corner = np.full((3, 3), carrier, int)
        ar, ac = CORNERS[anchor]
        corner[2 * ar, 2 * ac] = ink
        direction = np.full((3, 3), carrier, int)
        direction[1, 1] = direction[NEIGHBORS[turn]] = ink
        glyphs = [corner, direction]
        if rng.random() < .5:
            glyphs.reverse()
        for j, glyph in enumerate(glyphs):
            row = top + int(rng.integers(n - 2))
            left = data_width + 4 * j
            a[row:row + 3, left:left + 3] = glyph
    return a


def _make_pair(rng, n, controls, palette, witness=False, balanced=False):
    lattice, carrier, ink = palette
    bands = len(controls)
    copies = int(rng.integers(3, 21 // (n + 1) + 1))
    master = _master(rng, n, carrier, ink, witness)
    variants, deltas = [], []
    for b in range(bands):
        for _ in range(50):
            if balanced and b == 1:
                delta = np.zeros((n, n), bool)
                delta.flat[int(rng.choice(np.flatnonzero(master == carrier)))] = True
                delta.flat[int(rng.choice(np.flatnonzero(master == ink)))] = True
            elif not witness and deltas and rng.random() < .16:
                delta = deltas[int(rng.integers(len(deltas)))].copy()
            else:
                delta = _delta(rng, n, witness)
            variant = master.copy()
            variant[delta] = np.where(master[delta] == carrier, ink, carrier)
            if len(np.unique(variant)) == 2:
                break
        else:
            # A binary master always permits a one-cell flip retaining both roles.
            delta = np.zeros((n, n), bool)
            majority = carrier if (master == carrier).sum() >= (master == ink).sum() else ink
            delta.flat[int(rng.choice(np.flatnonzero(master == majority)))] = True
            variant = np.where(delta, carrier + ink - master, master)
        deltas.append(delta)
        variants.append(variant)
    columns = list(map(int, rng.integers(copies, size=bands)))
    inp = _assemble(master, variants, columns, controls, copies, palette, rng)
    instructions = [(_crop(d), k, a) for d, (k, a) in zip(deltas, controls)]
    counts = _forward_counts(n, instructions)
    out = _expand(master, counts, carrier, ink)
    return {'input': inp, 'output': out}


def _count_arg(name, value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not lo <= int(value) <= hi:
        raise ValueError(f'{name} must be in {lo}..{hi}')
    return int(value)


def _conservative_witnesses(rng, palettes):
    """Finite generation-only fallback if random teaching search is exhausted.

    These small designed bit patterns are not parent fixtures. They implement
    exactly the ordinary rule, with palette and visible key positions supplied
    by the current RNG. The input-only solver has no access to this helper.
    """
    designs = (
        ('001/010/100', ('000/001/011', '000/010/011', '110/010/000'),
         ((1, 3), (1, 3), (0, 3)), (1, 2, 3)),
        ('0101/0010/1010/0110',
         ('0000/0101/0111/0111', '0000/0100/1000/0000', '0000/1000/1110/0000'),
         ((1, 2), (0, 3), (1, 3)), (2, 2, 2)),
    )
    pairs = []
    for (text, edits, controls, columns), palette in zip(designs, palettes):
        lattice, carrier, ink = palette
        bits = np.array([[int(c) for c in row] for row in text.split('/')], bool)
        master = np.where(bits, ink, carrier)
        deltas = [np.array([[int(c) for c in row] for row in t.split('/')], bool) for t in edits]
        variants = [np.where(d, carrier + ink - master, master) for d in deltas]
        inp = _assemble(master, variants, columns, controls, 4, palette, rng)
        counts = _forward_counts(len(master), [(_crop(d), k, a) for d, (k, a) in zip(deltas, controls)])
        pairs.append({'input': inp, 'output': _expand(master, counts, carrier, ink)})
    return pairs


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count_arg('num_train', num_train, 2, 10)
    nq = _count_arg('num_test', num_test, 1, 4)
    if nt <= nq:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    seed = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed)
    shared_palette = bool(rng.random() < .3)
    palette = tuple(map(int, rng.choice(10, 3, replace=False)))

    def next_palette():
        return palette if shared_palette else tuple(map(int, rng.choice(10, 3, replace=False)))

    q, even = int(rng.choice([1, 3])), int(rng.choice([0, 2]))
    a = int(rng.integers(4))
    b = (a + int(rng.choice([1, 3]))) % 4
    teaching_controls = [[(q, a), (q, a), (even, a)],
                         [(q, b), (even, a), (q, a)]]
    used = set(teaching_controls[0] + teaching_controls[1])
    allowed_held = [(k, an) for k in range(4) for an in range(4) if (k, an) not in used]
    held = allowed_held[int(rng.integers(len(allowed_held)))]
    allowed_train = [(k, an) for k in range(4) for an in range(4) if (k, an) != held]
    for _ in range(128):
        first = _make_pair(rng, 3, teaching_controls[0], next_palette(), witness=True)
        second_controls = teaching_controls[1] + [allowed_train[int(rng.integers(15))]
                                                 for _ in range(int(rng.integers(3)))]
        second = _make_pair(rng, 4, second_controls, next_palette(), witness=True, balanced=True)
        try:
            _training_evidence([_decode(first['input']), _decode(second['input'])])
        except AssertionError:
            continue
        break
    else:
        first, second = _conservative_witnesses(rng, [next_palette(), next_palette()])
        # Replace only sampling preferences; every convention remains visible.
        used = {(e['turn'], e['anchor']) for pair in (first, second)
                for e in _decode(pair['input'])['entries']}
        pool = [(k, an) for k in range(4) for an in range(4) if (k, an) not in used]
        held = pool[int(rng.integers(len(pool)))]
        allowed_train = [(k, an) for k in range(4) for an in range(4) if (k, an) != held]
    train = [first, second]
    if rng.random() < .5:
        train.reverse()
    puzzle, seen = {'train': train, 'test': []}, {_key(p['input']) for p in train}
    for split, count in (('train', nt - 2), ('test', nq)):
        for index in range(count):
            for _ in range(64):
                n = 5 if split == 'test' and index == 0 else int(rng.choice([3, 4] if split == 'train' else [3, 4, 5]))
                bands = int(rng.integers(1, 29 // (n + 1) + 1))
                pool = allowed_train if split == 'train' else [(k, an) for k in range(4) for an in range(4)]
                controls = [pool[int(rng.integers(len(pool)))] for _ in range(bands)]
                if split == 'test' and index == 0:
                    controls[int(rng.integers(bands))] = held
                pair = _make_pair(rng, n, controls, next_palette())
                if _key(pair['input']) not in seen:
                    break
            else:
                raise AssertionError('bounded distinct input construction failed')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            counts = _reference_counts(info)
            expected = _scalar_output(info, counts)
            assert np.array_equal(out, expected), 'label violates toggle mosaic rule'
            assert np.array_equal(solve(inp), expected)
            n, master = info['n'], info['master']
            assert out.shape == (n * n, n * n)
            swapped = info['carrier'] + info['ink'] - master
            for r in range(n):
                for c in range(n):
                    block = out[r*n:(r+1)*n, c*n:(c+1)*n]
                    assert np.array_equal(block, swapped if counts[r, c] % 2 else master)
            infos[split].append(info)
    _training_evidence(infos['train'])
    assert max(i['n'] for i in infos['train']) < infos['test'][0]['n']
    train_controls = {(e['turn'], e['anchor']) for i in infos['train'] for e in i['entries']}
    assert any((e['turn'], e['anchor']) not in train_controls for e in infos['test'][0]['entries'])
    return None


def self_test():
    """Bounded structural, semantic, ablation, mutation and 42-seed diversity checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records, signatures, shapes, palettes, columns = [], set(), set(), set(), set()
    for seed, puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        signature = tuple(_key(p[r]) for s in ('train', 'test') for p in puzzle[s] for r in ('input', 'output'))
        signatures.add(signature)
        if seed < 3:
            other = generate_puzzle(seed)
            assert signature == tuple(_key(p[r]) for s in ('train', 'test') for p in other[s] for r in ('input', 'output'))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                counts = _reference_counts(info)
                records.append(info)
                shapes.add(pair['input'].shape)
                palettes.add((info['lattice'], info['carrier'], info['ink']))
                columns.add(tuple(e['column'] for e in info['entries']))
                assert np.array_equal(_forward_counts(info['n'], [(e['mask'], e['turn'], e['anchor']) for e in info['entries']]), counts)
                # Reordering complete bands (including their controls) cannot change a commutative toggle.
                reordered = dict(info, entries=info['entries'][::-1])
                assert np.array_equal(_reference_counts(reordered), counts)
                stride = info['n'] + 1
                chunks = [pair['input'][b*stride:(b+1)*stride] for b in range(info['bands'])]
                reversed_input = np.vstack(chunks[::-1] + [pair['input'][-1:]])
                assert np.array_equal(solve(reversed_input), pair['output'])
                # Two identical extra requests cancel for every member, independent of geometry.
                duplicate = dict(info, entries=info['entries'] + [info['entries'][0]] * 2)
                assert np.array_equal(_reference_counts(duplicate) % 2, counts % 2)
    assert len(signatures) == 42 and len(shapes) >= 12 and len(palettes) >= 60
    assert {i['n'] for i in records} == {3, 4, 5}
    assert len({i['bands'] for i in records}) >= 5 and len({i['copies'] for i in records}) == 3
    assert {i['lattice'] for i in records} == set(range(10))
    assert {(e['turn'], e['anchor']) for i in records for e in i['entries']} == {(k, a) for k in range(4) for a in range(4)}
    assert len({_key(i['master'] == i['ink']) for i in records}) >= 100
    stencil_shapes = {e['mask'].shape for i in records for e in i['entries']}
    assert len(stencil_shapes) >= 12
    assert any(e['mask'].all() for i in records for e in i['entries'])
    assert any(not e['mask'].all() for i in records for e in i['entries'])
    assert any(i['bands'] == 1 for i in records)
    assert len(columns) >= 60
    fallback = _conservative_witnesses(np.random.default_rng(7), [(0, 1, 2), (9, 8, 7)])
    _training_evidence([_decode(p['input']) for p in fallback])
    for pair in fallback:
        assert np.array_equal(solve(pair['input']), pair['output'])
    # A legal input whose two identical requests cancel everywhere.
    first_band = fallback[0]['input'][:4]
    cancellation = np.vstack([first_band, first_band, fallback[0]['input'][-1:]])
    ci = _decode(cancellation)
    assert np.array_equal(solve(cancellation), np.tile(ci['master'], (3, 3)))
    nested = {s: [{r: v.tolist() for r, v in pair.items()} for pair in puzzles[0][s]]
              for s in puzzles[0]}
    validate_generated_puzzle(nested)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(202600 + nt * 10 + nq, nt, nq)
            validate_generated_puzzle(puzzle)
            assert len(puzzle['train']) == nt and len(puzzle['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = {s: [{r: np.array(v, copy=True) for r, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    damaged = puzzles[0]['train'][0]['input'].copy()
    info = _decode(damaged)
    c = info['entries'][0]['column']
    damaged[1:1 + info['n'], 1 + c * (info['n'] + 1):1 + c * (info['n'] + 1) + info['n']] = info['master']
    try:
        solve(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('missing exception accepted')
    return dict(seeds=42, pairs=len(records), input_shapes=len(shapes),
                role_palettes=len(palettes), stencil_shapes=len(stencil_shapes),
                normalized_masters=len({_key(i['master'] == i['ink']) for i in records}),
                geometric_conventions_checked=64, independent_semantic_approval=False)
