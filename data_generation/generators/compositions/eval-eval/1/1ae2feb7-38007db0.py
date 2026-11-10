"""Rhythmic fault broadcast: a difference run clocks its own exact patch.

No files, parent code, labels, or generation state are consulted by solve.
The only dependency beyond Python's standard library is NumPy.
"""
from collections import Counter
import hashlib
import numbers

import numpy as np


HOST_SEEDS = (0,154190155,718417110,689015228,229288235,4125563,789881793,
              433684809,995848075,148516652,207592542,264937125,189897362,
              641986714,633164784,56735672,951056191,244947916,148397792,
              538645823,966559455,931409480,970518649,405473972,231904984,
              338960235,897283215,388473353,900293747,738559855,409775106,
              566500412,295801107,884924957,536273043,85357038,327540586,
              202572899,808943769,160097509,321673518,140706830)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), 'grid dimensions'
    assert np.issubdtype(a.dtype, np.integer), 'integer cells required'
    assert np.all((a >= 0) & (a <= 9)), 'invalid color'
    return a


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _runs(mask):
    """Maximal True runs in each row, with half-open column bounds."""
    result = []
    for r, row in enumerate(mask):
        c = 0
        while c < len(row):
            if not row[c]:
                c += 1
                continue
            start = c
            while c < len(row) and row[c]:
                c += 1
            result.append((r, start, c))
    return result


def _assemble(rows, lattice):
    bands, copies = len(rows), len(rows[0])
    ph, pw = rows[0][0].shape
    out = np.full((bands * (ph + 1) + 1, copies * (pw + 1) + 1),
                  lattice, dtype=np.int8)
    for b, row in enumerate(rows):
        for c, tile in enumerate(row):
            r0, c0 = b * (ph + 1) + 1, c * (pw + 1) + 1
            out[r0:r0 + ph, c0:c0 + pw] = tile
    return out


def _read_input(input_grid):
    """Recover all semantic roles from the original input only."""
    a = _grid(input_grid)
    lattice = int(a[0, 0])
    rr = np.flatnonzero(np.all(a == lattice, axis=1))
    cc = np.flatnonzero(np.all(a == lattice, axis=0))
    assert len(rr) >= 2 and len(cc) >= 4, 'need framed bands and >=3 copies'
    assert rr[0] == cc[0] == 0
    assert rr[-1] == a.shape[0] - 1 and cc[-1] == a.shape[1] - 1
    assert np.all(np.diff(rr) == rr[1]) and np.all(np.diff(cc) == cc[1]), 'regular lattice'
    ph, pw = int(rr[1] - 1), int(cc[1] - 1)
    assert ph >= 1 and pw >= 1
    tiles = [[a[r + 1:s, c + 1:d] for c, d in zip(cc, cc[1:])]
             for r, s in zip(rr, rr[1:])]
    assert all(np.all(t != lattice) for row in tiles for t in row), 'lattice in tile'
    counts = Counter(_key(t) for row in tiles for t in row)
    maximum = max(counts.values())
    winners = [k for k, v in counts.items() if v == maximum]
    assert len(winners) == 1, 'ambiguous master'
    master = next(t for row in tiles for t in row if _key(t) == winners[0])
    sources, patches = [], []
    for row in tiles:
        different = [j for j, t in enumerate(row) if not np.array_equal(t, master)]
        assert len(different) == 1, 'need exactly one visible exception per band'
        s = different[0]
        sources.append(s)
        patches.append(_runs(row[s] != master))
    return dict(grid=a, lattice=lattice, ph=ph, pw=pw, bands=len(tiles),
                copies=len(tiles[0]), tiles=tiles, master=master,
                sources=sources, patches=patches)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Apply tile-distance clocks inferred from maximal disagreement runs."""
    info = _read_input(input_grid)
    out = info['grid'].copy()
    ph, pw = info['ph'], info['pw']
    for b, s in enumerate(info['sources']):
        source = info['tiles'][b][s]
        r0 = 1 + b * (ph + 1)
        for j in range(info['copies']):
            if j == s:
                continue
            distance = abs(j - s)
            c0 = 1 + j * (pw + 1)
            for r, left, right in info['patches'][b]:
                if (distance - 1) % (right - left) == 0:
                    out[r0 + r, c0 + left:c0 + right] = source[r, left:right]
    return out


def _master_mask(rng, h, w):
    """A mixture of genuine structures, not just recolorings of one tile."""
    style = int(rng.integers(6))
    rr, cc = np.indices((h, w))
    if style == 0:
        m = rng.random((h, w)) < float(rng.uniform(.2, .8))
    elif style == 1:
        axis = rr if rng.random() < .5 else cc
        m = axis % int(rng.integers(2, 4)) == int(rng.integers(2))
    elif style == 2:
        m = rng.random((h, w)) < .4
        m |= m[:, ::-1] if rng.random() < .5 else m[::-1, :]
    elif style == 3:
        m = np.zeros((h, w), dtype=bool)
        spine = int(rng.integers(w))
        m[:, spine] = True
        for r in rng.choice(h, int(rng.integers(1, h + 1)), replace=False):
            end = int(rng.integers(w))
            m[r, min(spine, end):max(spine, end) + 1] = True
    elif style == 4:
        m = (rr == 0) | (rr == h - 1) | (cc == 0) | (cc == w - 1)
        if rng.random() < .5:
            m = ~m
    else:
        m = np.zeros((h, w), dtype=bool)
        m[int(rng.integers(h)), int(rng.integers(w))] = True
        for _ in range(int(rng.integers(1, h * w))):
            points = [(r, c) for r in range(h) for c in range(w)
                      if not m[r, c] and any(0 <= x < h and 0 <= y < w and m[x, y]
                      for x, y in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)))]
            if not points:
                break
            m[points[int(rng.integers(len(points)))]] = True
    if not np.any(m):
        m[int(rng.integers(h)), int(rng.integers(w))] = True
    if np.all(m):
        m[int(rng.integers(h)), int(rng.integers(w))] = False
    return m


def _delta(rng, h, w):
    d = np.zeros((h, w), dtype=bool)
    selected = rng.choice(h, int(rng.integers(1, h + 1)), replace=False)
    for r in selected:
        length = int(rng.integers(1, min(3, w) + 1))
        start = int(rng.integers(w - length + 1))
        d[r, start:start + length] = True
        if rng.random() < .5:
            # Optional independent run, with at least one unchanged cell between.
            possibilities = [(s, s + n) for n in range(1, min(3, w) + 1)
                             for s in range(w - n + 1)
                             if s + n < start or s > start + length]
            if possibilities:
                left, right = possibilities[int(rng.integers(len(possibilities)))]
                d[r, left:right] = True
    return d


def _paint_pair(rng, geometry, palette, lesson=None, reach_min=None):
    ph, pw, bands, copies = geometry
    lattice, carrier, ink = palette
    master_mask = _master_mask(rng, ph, pw)
    deltas = [_delta(rng, ph, pw) for _ in range(bands)]
    sources = [int(rng.integers(copies)) for _ in range(bands)]
    witness = int(rng.integers(bands))
    if lesson == 'periods':
        r3, r2, r1 = map(int, rng.choice(ph, 3, replace=False))
        phase = int(rng.integers(2))
        master_mask[r3] = (np.arange(pw) + phase) % 2 == 1
        master_mask[r2] = (np.arange(pw) + int(rng.integers(2))) % 2 == 1
        start2 = int(rng.integers(2))
        single = int(rng.integers(pw))
        # The length-3 flip has a +/-1 color imbalance; cancel it with L=1.
        master_mask[r1, single] = not bool(phase)
        d = np.zeros((ph, pw), dtype=bool)
        d[r3, :] = True
        d[r2, start2:start2 + 2] = True
        d[r1, single] = True
        deltas[witness] = d
        sources[witness] = int(rng.choice([0, copies - 1]))
    elif lesson == 'split':
        r, r2 = map(int, rng.choice(ph, 2, replace=False))
        master_mask[r] = np.arange(pw) % 2 == int(rng.integers(2))
        master_mask[r2] = np.arange(pw) % 2 == int(rng.integers(2))
        d = np.zeros((ph, pw), dtype=bool)
        d[r, 0] = True
        d[r, 2:4] = True
        d[r2, 1:3] = True
        if rng.random() < .5:
            d = d[:, ::-1].copy()
        deltas[witness] = d
        sources[witness] = int(rng.choice([1, copies - 2]))
    elif lesson == 'wide':
        r = int(rng.integers(ph))
        # At least two independently clocked patches, with a visible gap.
        d = deltas[witness]
        d[r] = False
        options = [(1, 2), (2, 1), (1, 3), (3, 1)]
        left_length, right_length = options[int(rng.integers(len(options)))]
        d[r, :left_length] = True
        d[r, pw - right_length:] = True
        sources[witness] = int(rng.choice([0, copies - 1]))
    elif lesson == 'reach':
        d = deltas[witness]
        d[int(rng.integers(ph)), :] = True  # visibly exercise period 2
    if reach_min is not None:
        eligible = [s for s in range(copies) if max(s, copies - 1 - s) > reach_min]
        assert eligible
        sources = [int(rng.choice(eligible)) for _ in range(bands)]
    master = np.where(master_mask, ink, carrier).astype(np.int8)
    inputs, outputs = [], []
    for d, s in zip(deltas, sources):
        variant = master.copy()
        variant[d] = np.where(master_mask[d], carrier, ink)
        row = [master.copy() for _ in range(copies)]
        row[s] = variant
        answer = [t.copy() for t in row]
        # Construction uses known run coordinates and outward range stepping.
        # solve instead re-infers them and tests each receiver's divisibility.
        for r, left, right in _runs(d):
            length = right - left
            for sign in (-1, 1):
                for j in range(s + sign, copies if sign == 1 else -1, sign * length):
                    answer[j][r, left:right] = variant[r, left:right]
        inputs.append(row)
        outputs.append(answer)
    return {'input': _assemble(inputs, lattice), 'output': _assemble(outputs, lattice)}


def _legal_counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral)
                or not low <= int(value) <= high):
            raise ValueError('counts must be integers: train 2..10, test 1..4')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _legal_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(['reach', 'wide']))
    shared_palette = bool(rng.random() < .3)
    palette = tuple(map(int, rng.choice(10, 3, replace=False)))
    lessons = list(rng.permutation(['periods', 'split'])) + [None] * (num_train - 2)
    puzzle, seen, max_reach = {'train': [], 'test': []}, set(), 0
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            lesson = lessons[index] if split == 'train' else regime
            if lesson == 'periods':
                ph, pw, copies = int(rng.integers(3, 6)), 3, int(rng.integers(6, 8))
            elif lesson == 'split':
                ph, pw, copies = int(rng.integers(3, 6)), 4, 5
            elif lesson == 'reach':
                ph, pw, copies = int(rng.integers(2, 6)), 2, int(rng.integers(8, 10))
            elif lesson == 'wide':
                ph, pw, copies = int(rng.integers(2, 6)), int(rng.integers(5, 7)), 4
            else:
                ph, pw = int(rng.integers(2, 6)), int(rng.integers(2, 5))
                copies = int(rng.integers(3, min(7, 29 // (pw + 1)) + 1))
            cap = min(4 if lesson in ('periods', 'split') else 5, 29 // (ph + 1))
            bands = int(rng.integers(1, cap + 1))
            geometry = ph, pw, bands, copies
            for attempt in range(32):
                colors = palette if shared_palette else tuple(map(int, rng.choice(10, 3, replace=False)))
                pair = _paint_pair(rng, geometry, colors, lesson,
                                   max_reach if lesson == 'reach' else None)
                k = _key(pair['input'])
                if k not in seen:
                    break
            else:
                raise RuntimeError('bounded unique-input attempts exhausted')
            seen.add(k)
            puzzle[split].append(pair)
            if split == 'train':
                info = _read_input(pair['input'])
                max_reach = max(max_reach, max(max(s, copies - 1 - s) for s in info['sources']))
    validate_generated_puzzle(puzzle)
    return puzzle


def _counterfactual(info, kind):
    """Named rival rules used only to audit demonstration evidence."""
    result = info['grid'].copy()
    for b, s in enumerate(info['sources']):
        source = info['tiles'][b][s]
        patches = info['patches'][b]
        if kind == 'first':
            s, source = 0, info['tiles'][b][0]
            patches = _runs(source != info['master'])
        if kind == 'raw':
            divided = []
            for r, left, right in patches:
                start = left
                for c in range(left + 1, right + 1):
                    if c == right or source[r, c] != source[r, start]:
                        divided.append((r, start, c))
                        start = c
            patches = divided
        if kind == 'whole_row':
            patches = [(r, min(c for rr, c, e in patches if rr == r),
                        max(e for rr, c, e in patches if rr == r))
                       for r in sorted({r for r, c, e in patches})]
        for j in range(info['copies']):
            if j == s:
                continue
            for r, left, right in patches:
                distance = abs(j - s)
                length = right - left
                eligible = (kind == 'all' or ((distance if kind == 'zero_phase' else distance - 1) % length == 0))
                if eligible:
                    r0 = b * (info['ph'] + 1) + 1
                    c0 = j * (info['pw'] + 1) + 1
                    result[r0 + r, c0 + left:c0 + right] = source[r, left:right]
    return result


def _training_evidence(records):
    recurrence, balanced, split_row, interior = set(), False, False, False
    for pair, info in records:
        for b, (s, patches) in enumerate(zip(info['sources'], info['patches'])):
            source = info['tiles'][b][s]
            balanced |= np.array_equal(np.sort(source, axis=None), np.sort(info['master'], axis=None))
            interior |= 0 < s < info['copies'] - 1
            per_row = Counter(r for r, left, right in patches)
            split_row |= any(n >= 2 for n in per_row.values())
            reach = max(s, info['copies'] - 1 - s)
            recurrence.update(right - left for r, left, right in patches if reach >= 1 + right - left)
    assert {1, 2, 3} <= recurrence, 'teach numerical recurrence'
    assert balanced, 'need a count-preserving exceptional bitmap'
    assert split_row and interior, 'need independent patches and bilateral origin'
    for kind in ('all', 'first', 'raw', 'whole_row', 'zero_phase'):
        assert any(not np.array_equal(pair['output'], _counterfactual(info, kind))
                   for pair, info in records), 'rival fits all demonstrations: ' + kind


def validate_generated_puzzle(puzzle):
    """Check returned grids, exact preservation, schedules, and teaching evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _legal_counts(len(puzzle['train']), len(puzzle['test']))
    records, seen = {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            info = _read_input(inp)
            assert len(np.unique(inp)) == 3, 'sampler uses three visible roles'
            assert not np.array_equal(inp, out), 'must broadcast at least one patch'
            lattice_mask = inp == info['lattice']
            assert np.array_equal(inp[lattice_mask], out[lattice_mask]), 'changed lattice'
            for b, (row, s) in enumerate(zip(info['tiles'], info['sources'])):
                source = row[s]
                r0 = b * (info['ph'] + 1) + 1
                patch_lengths = np.zeros(source.shape, dtype=int)
                for r, left, right in info['patches'][b]:
                    patch_lengths[r, left:right] = right - left
                for j in range(info['copies']):
                    c0 = j * (info['pw'] + 1) + 1
                    tile = out[r0:r0 + info['ph'], c0:c0 + info['pw']]
                    if j == s:
                        assert np.array_equal(tile, source), 'source modified'
                    else:
                        for r in range(info['ph']):
                            for c in range(info['pw']):
                                length = patch_lengths[r, c]
                                selected = length and (abs(j - s) - 1) % length == 0
                                expected = source[r, c] if selected else info['master'][r, c]
                                assert tile[r, c] == expected, 'patch, phase, clipping or preservation'
            assert np.array_equal(solve(inp), out), 'input-only solve disagreement'
            records[split].append((pair, info))
    _training_evidence(records['train'])
    train_infos = [i for p, i in records['train']]
    assert len({i['pw'] for i in train_infos}) >= 2
    max_width = max(i['pw'] for i in train_infos)
    max_copies = max(i['copies'] for i in train_infos)
    max_distance = max(max(s, i['copies'] - 1 - s) for i in train_infos for s in i['sources'])
    for pair, info in records['test']:
        if info['pw'] > max_width:
            assert any(len({e - a for rr, a, e in patches if rr == r}) >= 2
                       for patches in info['patches'] for r, a, e in patches), 'wider tests combine clocks'
        else:
            assert info['copies'] > max_copies, 'test must extend taught geometry'
            assert all(max(s, info['copies'] - 1 - s) > max_distance for s in info['sources'])


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded streaming checks; returns measurements and creates no files."""
    master = np.array([[0, 2, 0], [2, 0, 2], [0, 0, 2]], dtype=np.int8)
    source = np.array([[2, 0, 2], [0, 2, 2], [0, 2, 2]], dtype=np.int8)
    inp = _assemble([[source] + [master] * 6], 8)
    # Explicit, hand-calculated schedules: L3 at 1,4; L2 at 1,3,5; L1 everywhere.
    expected_tiles = []
    for j in range(7):
        tile = master.copy()
        if j in (0, 1, 4):
            tile[0] = source[0]
        if j in (0, 1, 3, 5):
            tile[1] = source[1]
        tile[2] = source[2]
        expected_tiles.append(tile)
    expected = _assemble([expected_tiles], 8)
    assert np.array_equal(solve(inp), expected)
    assert np.array_equal(solve(inp[:, ::-1]), expected[:, ::-1]), 'horizontal mirror'
    centered = _assemble([[master] * 3 + [source] + [master] * 3], 8)
    centered_answer = []
    for j in range(7):
        tile = master.copy()
        if j in (2, 3, 4):
            tile[0] = source[0]
        if j in (0, 2, 3, 4, 6):
            tile[1] = source[1]
        tile[2] = source[2]
        centered_answer.append(tile)
    assert np.array_equal(solve(centered), _assemble([centered_answer], 8))
    # Two separated disagreement runs in one row have independent clocks.
    m = np.array([[0, 2, 0, 2], [2, 0, 2, 0]], dtype=np.int8)
    e = np.array([[2, 2, 2, 0], [2, 0, 2, 0]], dtype=np.int8)
    partial = np.array([[2, 2, 0, 2], [2, 0, 2, 0]], dtype=np.int8)
    separated = _assemble([[m, e, m, m, m]], 8)
    assert np.array_equal(solve(separated), _assemble([[e, e, e, partial, e]], 8))
    # The input-only rule is not capped at the sampler's period 3.
    long_m = np.array([[0, 2, 0, 2], [2, 0, 2, 0]], dtype=np.int8)
    long_e = long_m.copy()
    long_e[0] = [2, 0, 2, 0]
    longer = _assemble([[long_e, long_m, long_m, long_m, long_m]], 8)
    assert np.array_equal(solve(longer), _assemble([[long_e, long_e, long_m, long_m, long_m]], 8))
    seen, shapes, masters, widths, copies, periods, positions = set(), set(), set(), set(), set(), set(), set()
    palettes, multi, balanced, band_counts, normalized = set(), 0, 0, set(), set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = _puzzle_digest(puzzle)
        assert digest == _puzzle_digest(generate_puzzle(seed)), 'determinism'
        seen.add(digest)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                i = _read_input(pair['input'])
                shapes.add(pair['input'].shape)
                widths.add(i['pw'])
                copies.add(i['copies'])
                band_counts.add(i['bands'])
                palettes.add(tuple(sorted(map(int, np.unique(pair['input'])))))
                roles = {}
                norm = bytes(roles.setdefault(int(c), len(roles)) for c in pair['input'].flat)
                normalized.add((pair['input'].shape, norm))
                m = i['master']
                masters.add((m.shape, bytes((m == m[0, 0]).astype(np.uint8).flat)))
                for b, patches in enumerate(i['patches']):
                    periods.update(e - a for r, a, e in patches)
                    multi += any(n >= 2 for n in Counter(r for r, a, e in patches).values())
                    s = i['sources'][b]
                    positions.add('edge' if s in (0, i['copies'] - 1) else 'interior')
                    balanced += np.array_equal(np.sort(i['tiles'][b][s], axis=None), np.sort(m, axis=None))
        # A color permutation must not change semantic roles or arithmetic.
        p = puzzle['test'][0]
        perm = np.arange(9, -1, -1, dtype=np.int8)
        assert np.array_equal(solve(perm[p['input']]), perm[p['output']])
        assert np.array_equal(solve(p['input'][::-1]), p['output'][::-1])
    assert len(seen) == 42 and len(normalized) >= 180
    assert len(shapes) >= 25 and len(masters) >= 100
    assert widths == {2, 3, 4, 5, 6} and copies >= {3, 4, 5, 6, 7, 8, 9}
    assert periods == {1, 2, 3} and positions == {'edge', 'interior'}
    assert len(palettes) >= 30 and len(band_counts) >= 4
    assert multi >= 42 and balanced >= 42
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + 10 * nt + ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    base = generate_puzzle(0, 2, 1)
    for defect in ('source_absent', 'output_cell', 'lattice'):
        bad = {s: [{k: a.copy() for k, a in pair.items()} for pair in base[s]] for s in base}
        p = bad['train'][0]
        i = _read_input(p['input'])
        if defect == 'source_absent':
            s = i['sources'][0]
            c0 = 1 + s * (i['pw'] + 1)
            p['input'][1:1 + i['ph'], c0:c0 + i['pw']] = i['master']
        elif defect == 'output_cell':
            p['output'][1, 1] = (int(p['output'][1, 1]) + 1) % 10
        else:
            p['output'][0, 0] = (int(p['output'][0, 0]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError('mutation accepted: ' + defect)
    return dict(seeds=42, pairs=210, distinct_puzzles=len(seen),
                input_shapes=len(shapes), normalized_inputs=len(normalized),
                master_masks=len(masters), tile_widths=sorted(widths),
                copy_counts=sorted(copies), periods=sorted(periods),
                palettes=len(palettes), multi_run_bands=int(multi),
                balanced_bands=int(balanced), independent_solver=False)
