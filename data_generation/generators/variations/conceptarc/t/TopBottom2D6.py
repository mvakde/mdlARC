"""Select and crop the connected color reaching the lowest input row."""

import numpy as np

_SOURCE_INPUTS = {
    ("00202000", "00202000", "00222000", "00000000", "00000404", "00000404", "00000444", "00000000"),
    ("0008880", "7778000", "0070000", "0000000", "0000000", "0000000", "0000000"),
    ("000000", "003030", "003330", "808300", "888000", "080000"),
    ("0100", "0000", "0050", "0000"),
    ("00000000", "00330000", "00330000", "00660000", "00660000", "00000000", "00000000", "00000000"),
    ("304000", "304000", "304000", "304444", "333300", "000000"),
    ("0000000000", "0777777700", "0777777700", "0777777700", "0000000000", "0000222222", "0000200002", "0000200002", "0000200002", "0000222222"),
}


def _sig(a):
    return tuple(''.join(str(int(x)) for x in row) for row in a)


def _counts(seed, rng, nt, ns):
    for name, x, lo, hi in (("num_train", nt, 2, 10), ("num_test", ns, 1, 4)):
        if x is not None and (not isinstance(x, (int, np.integer)) or isinstance(x, bool) or not lo <= x <= hi):
            raise ValueError(name)
    if nt is None and ns is None:
        ns = seed % 4 + 1
        nt = ns + 1 + (seed // 4) % (10 - ns)
    elif nt is None:
        nt = int(rng.integers(ns + 1, 11))
    elif ns is None:
        ns = int(rng.integers(1, min(4, nt - 1) + 1))
    if nt <= ns:
        raise ValueError('num_train must exceed num_test')
    return int(nt), int(ns)


def _mask(rng, h, w, kind):
    a = np.zeros((h, w), bool)
    if kind == 'solid':
        a[:] = True
    elif kind == 'frame' and h >= 3 and w >= 3:
        a[[0, -1], :] = True
        a[:, [0, -1]] = True
    elif kind == 'tip':
        a[0, :] = True
        a[:, int(rng.integers(w))] = True
        if h > 2 and w > 2 and rng.integers(2):
            a[int(rng.integers(1, h - 1)), :] = True
    elif kind == 'branch':
        c = int(rng.integers(w))
        a[:, c] = True
        a[int(rng.integers(h)), :c + 1] = True
        a[int(rng.integers(h)), c:] = True
        for _ in range(int(rng.integers(0, 3))):
            r, end = int(rng.integers(h)), int(rng.integers(w))
            a[r, min(c, end):max(c, end) + 1] = True
    else:
        a[int(rng.integers(h)), int(rng.integers(w))] = True
        target = int(rng.integers(max(2, h * w // 4), max(3, 3 * h * w // 4) + 1))
        while int(a.sum()) < min(target, h * w):
            choices = set()
            for r, c in np.argwhere(a):
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < h and 0 <= cc < w and not a[rr, cc]:
                        choices.add((rr, cc))
            if not choices:
                break
            a[sorted(choices)[int(rng.integers(len(choices)))]] = True
    rr, cc = np.where(a)
    return a[rr.min():rr.max() + 1, cc.min():cc.max() + 1]


def _shape(rng, h, w):
    kinds = ['solid', 'tip', 'branch', 'irregular']
    if h >= 3 and w >= 3:
        kinds.append('frame')
    return _mask(rng, h, w, kinds[int(rng.integers(len(kinds)))])


def _solve(inp, bg):
    colors = [int(c) for c in np.unique(inp) if c != bg]
    winner = max(colors, key=lambda c: np.where(inp == c)[0].max())
    rr, cc = np.where(inp == winner)
    crop = inp[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    return np.where(crop == winner, winner, bg).astype(np.int8)


def _placements(a, b, ta, tb):
    ca = [(ta + int(r), int(c)) for r, c in np.argwhere(a)]
    cb = {(tb + int(r), int(c)) for r, c in np.argwhere(b)}
    choices = {k: [] for k in ('separate', 'touch', 'zero', 'one', 'many')}
    for dx in range(-a.shape[1] - 3, b.shape[1] + 4):
        aa = {(r, c + dx) for r, c in ca}
        if aa & cb:
            continue
        inside = sum(ta <= r < ta + a.shape[0] and dx <= c < dx + a.shape[1] for r, c in cb)
        touch = any((r - 1, c) in cb or (r + 1, c) in cb or (r, c - 1) in cb or (r, c + 1) in cb for r, c in aa)
        overlap = max(ta, tb) <= min(ta + a.shape[0] - 1, tb + b.shape[0] - 1) and max(dx, 0) <= min(dx + a.shape[1] - 1, b.shape[1] - 1)
        category = 'many' if inside >= 2 else 'one' if inside == 1 else 'touch' if touch else 'zero' if overlap else 'separate'
        choices[category].append(dx)
    return choices


def _compose(rng, a, b, ta, tb, dx, bg, colors, compact=False):
    top, left = min(ta, tb), min(dx, 0)
    h = max(ta + a.shape[0], tb + b.shape[0]) - top
    w = max(dx + a.shape[1], b.shape[1]) - left
    margins = [int(rng.integers(0, 2 if compact else 4)) for _ in range(4)]
    margins[1] += max(0, 4 - h - margins[0] - margins[1])
    margins[3] += max(0, 4 - w - margins[2] - margins[3])
    grid = np.full((h + margins[0] + margins[1], w + margins[2] + margins[3]), bg, np.int8)
    ar, ac = np.where(a)
    br, bc = np.where(b)
    grid[margins[0] + ta - top + ar, margins[2] + dx - left + ac] = colors[0]
    grid[margins[0] + tb - top + br, margins[2] - left + bc] = colors[1]
    return grid


def _pair(rng, bg, colors, mode):
    if mode == 'tiny':
        side = int(rng.integers(1, 3))
        a = b = np.ones((side, side), bool)
        ta, tb = side + int(rng.integers(0, 3)), 0
        dx = int(rng.integers(-side - 2, side + 3))
        inp = _compose(rng, a, b, ta, tb, dx, bg, colors, True)
        return inp, _solve(inp, bg)
    if mode == 'small':
        hb, wb = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        a = np.ones((1, int(rng.integers(1, 3))), bool)
        b = np.ones((hb, wb), bool)
        ta, tb = hb + int(rng.integers(0, 3)), 0
        categories = ('separate', 'touch')
    elif mode == 'level':
        ha = int(rng.integers(4, 7))
        a = np.ones((ha, int(rng.integers(2, 5))), bool)
        hb = int(rng.integers(2, min(ha, 4)))
        b = np.ones((hb, int(rng.integers(1, 3))), bool)
        ta = tb = 0
        categories = ('touch', 'one', 'many', 'separate')
    elif mode == 'contain':
        ha, wa = int(rng.integers(4, 8)), int(rng.integers(4, 8))
        a = _mask(rng, ha, wa, 'frame')
        hb, wb = int(rng.integers(1, ha - 1)), int(rng.integers(1, wa - 1))
        b = _shape(rng, hb, wb)
        if b.shape[0] > ha - 2 or b.shape[1] > wa - 2:
            return None
        ta, tb = 0, int(rng.integers(1, ha - b.shape[0]))
        dx = -int(rng.integers(1, wa - b.shape[1]))
        inp = _compose(rng, a, b, ta, tb, dx, bg, colors)
        return inp, _solve(inp, bg)
    elif mode == 'zero':
        ha, wa = int(rng.integers(4, 8)), int(rng.integers(4, 8))
        a = _mask(rng, ha, wa, 'frame' if rng.integers(2) else 'branch')
        b = np.zeros((ha, wa), bool)
        b[0, :] = True
        b[:, -1] = True
        ta, tb = 2, 0
        dx = -(wa - 2)
        inp = _compose(rng, a, b, ta, tb, dx, bg, colors)
        return inp, _solve(inp, bg)
    else:
        equal = mode == 'equal'
        relation = 'lower' if equal else ('lower', 'same', 'higher')[int(rng.integers(3))]
        if equal:
            ha = hb = int(rng.integers(1, 7))
            wa = wb = int(rng.integers(1, 7))
        elif relation == 'lower':
            ha, hb = int(rng.integers(1, 8)), int(rng.integers(1, 8))
            wa, wb = int(rng.integers(1, 8)), int(rng.integers(1, 8))
        else:
            ha = int(rng.integers(3, 8))
            hb = int(rng.integers(1, ha))
            wa, wb = int(rng.integers(1, 8)), int(rng.integers(1, 8))
        a = _shape(rng, ha, wa)
        b = a.copy() if equal and rng.integers(2) else _shape(rng, hb, wb)
        ha, hb = a.shape[0], b.shape[0]
        if relation == 'lower':
            tb, ta = 0, max(1, hb - ha + 1) + int(rng.integers(0, 4))
        elif relation == 'same':
            if ha <= hb:
                return None
            ta = tb = 0
        else:
            if ha <= hb + 1:
                return None
            ta, tb = 0, int(rng.integers(1, ha - hb))
        categories = ('separate', 'touch', 'zero', 'one', 'many')
    if ta + a.shape[0] <= tb + b.shape[0]:
        return None
    placements = _placements(a, b, ta, tb)
    category = categories[int(rng.integers(len(categories)))]
    if not placements[category]:
        return None
    dx = placements[category][int(rng.integers(len(placements[category])))]
    inp = _compose(rng, a, b, ta, tb, dx, bg, colors, mode == 'small')
    return inp, _solve(inp, bg)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    nt, ns = _counts(int(seed), rng, num_train, num_test)
    bg = int(rng.choice([0, 0, 0, 0, 0, 0, 0, 1, 2, 5]))
    base = tuple(int(c) for c in rng.choice([c for c in range(10) if c != bg], 2, replace=False))
    puzzle = {'train': [], 'test': []}
    seen = set()
    first_side = 0
    first_right_side = 0
    for regime, count in (('train', nt), ('test', ns)):
        for i in range(count):
            mode = ('small', 'level')[i] if regime == 'train' and i < 2 else ('random', 'random', 'random', 'random', 'random', 'equal', 'equal', 'contain', 'zero', 'tiny')[int(rng.integers(10))]
            for _ in range(250):
                prng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                colors = (base if i == 0 else base[::-1]) if regime == 'train' and i < 2 else tuple(int(c) for c in prng.choice([c for c in range(10) if c != bg], 2, replace=False))
                pair = _pair(prng, bg, colors, mode)
                if pair is None:
                    continue
                inp, out = pair
                sig = _sig(inp)
                counts = np.bincount(inp.ravel(), minlength=10)
                if sig in seen or sig in _SOURCE_INPUTS or counts[bg] <= max(counts[colors[0]], counts[colors[1]]) or max(inp.shape) > 30 or out.shape[0] >= inp.shape[0] or out.shape[1] >= inp.shape[1]:
                    continue
                if regime == 'train' and i < 2:
                    side = int(np.sign(np.where(inp == colors[0])[1].min() - np.where(inp == colors[1])[1].min()))
                    right_side = int(np.sign(np.where(inp == colors[0])[1].max() - np.where(inp == colors[1])[1].max()))
                    if side == 0 or right_side != side or (i == 1 and (side != -first_side or right_side != -first_right_side)):
                        continue
                    if i == 0:
                        first_side = side
                        first_right_side = right_side
                seen.add(sig)
                puzzle[regime].append({'input': inp, 'output': out})
                break
            else:
                raise RuntimeError('could not construct a unique pair')
    validate_generated_puzzle(puzzle)
    return puzzle


def _connected(inp, color):
    cells = set(map(tuple, np.argwhere(inp == color)))
    found = {next(iter(cells))}
    todo = list(found)
    while todo:
        r, c = todo.pop()
        for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if q in cells and q not in found:
                found.add(q)
                todo.append(q)
    return found == cells


def _facts(inp, out):
    bg = int(np.argmax(np.bincount(inp.ravel(), minlength=10)))
    colors = [int(c) for c in np.unique(inp) if c != bg]
    assert len(colors) == 2 and all(_connected(inp, c) for c in colors)
    boxes = {}
    for c in colors:
        rr, cc = np.where(inp == c)
        boxes[c] = (int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max()), len(rr))
    assert boxes[colors[0]][1] != boxes[colors[1]][1]
    w = max(colors, key=lambda c: boxes[c][1])
    l = next(c for c in colors if c != w)
    r0, r1, c0, c1, _ = boxes[w]
    crop = inp[r0:r1 + 1, c0:c1 + 1]
    assert np.array_equal(out, np.where(crop == w, w, bg))
    assert out.shape[0] < inp.shape[0] and out.shape[1] < inp.shape[1]
    return bg, w, l, boxes[w], boxes[l]


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[k], list) for k in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, bgs, training = set(), set(), []
    for regime in ('train', 'test'):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for a in pair.values():
                assert isinstance(a, np.ndarray) and a.ndim == 2 and np.issubdtype(a.dtype, np.integer)
                assert 1 <= a.shape[0] <= 30 and 1 <= a.shape[1] <= 30
                assert bool(np.all((a >= 0) & (a <= 9)))
            inp, out = pair['input'], pair['output']
            key = _sig(inp)
            assert key not in seen and key not in _SOURCE_INPUTS
            seen.add(key)
            bg, w, l, wb, lb = _facts(inp, out)
            bgs.add(bg)
            if regime == 'train':
                training.append((w, l, wb, lb))
    assert len(bgs) == 1
    assert any(wb[4] < lb[4] for _, _, wb, lb in training)
    assert any(wb[0] == lb[0] for _, _, wb, lb in training)
    assert any(wb[0] > lb[0] and wb[1] - wb[0] < lb[1] - lb[0] for _, _, wb, lb in training)
    assert training[0][0] == training[1][1] and training[0][1] == training[1][0]
    assert training[0][2][4] < training[0][3][4]
    assert training[1][2][4] > training[1][3][4]
    assert (training[0][2][2] - training[0][3][2]) * (training[1][2][2] - training[1][3][2]) < 0
    assert (training[0][2][3] - training[0][3][3]) * (training[1][2][3] - training[1][3][3]) < 0


def _equal_puzzles(a, b):
    return all(len(a[k]) == len(b[k]) and all(np.array_equal(x[t], y[t]) for x, y in zip(a[k], b[k]) for t in ('input', 'output')) for k in ('train', 'test'))


def self_test() -> None:
    train_counts, test_counts, sizes, outputs, layouts, palettes = (set() for _ in range(6))
    relations = set()
    smaller = 0
    features = {k: set() for k in ('contact', 'inside', 'top', 'shape', 'area', 'box', 'background')}
    varied_palettes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _equal_puzzles(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        pair_palettes = set()
        for regime in ('train', 'test'):
            for pair in puzzle[regime]:
                inp, out = pair['input'], pair['output']
                bg, w, l, wb, lb = _facts(inp, out)
                sizes.add(inp.shape)
                outputs.add(_sig(out))
                palettes.add(tuple(sorted((w, l))))
                pair_palettes.add(tuple(sorted((w, l))))
                relations.add(int(np.sign(wb[0] - lb[0])))
                smaller += wb[4] < lb[4]
                features['background'].add(bg)
                features['top'].add(int(np.sign(wb[0] - lb[0])))
                features['area'].add(int(np.sign(wb[4] - lb[4])))
                features['box'].add((wb[1] - wb[0], wb[3] - wb[2]) == (lb[1] - lb[0], lb[3] - lb[2]))
                winner_cells = set(map(tuple, np.argwhere(inp == w)))
                loser_cells = set(map(tuple, np.argwhere(inp == l)))
                contact = any((r - 1, c) in loser_cells or (r + 1, c) in loser_cells or (r, c - 1) in loser_cells or (r, c + 1) in loser_cells for r, c in winner_cells)
                inside = sum(wb[0] <= r <= wb[1] and wb[2] <= c <= wb[3] for r, c in loser_cells)
                features['contact'].add(contact)
                features['inside'].add(0 if inside == 0 else 1 if inside == 1 else 2)
                mask = out == w
                shape = 'pixel' if wb[4] == 1 else 'line' if min(mask.shape) == 1 else 'solid' if mask.all() else 'frame' if mask.shape[0] >= 3 and mask.shape[1] >= 3 and mask[0, :].all() and mask[-1, :].all() and mask[:, 0].all() and mask[:, -1].all() and not mask[1:-1, 1:-1].any() else 'other'
                features['shape'].add(shape)
                rr, cc = np.where(inp != bg)
                sub = inp[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
                remap = {bg: 0, w: 1, l: 2}
                layouts.add(tuple(tuple(remap[int(x)] for x in row) for row in sub))
        varied_palettes += len(pair_palettes) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(sizes) >= 30 and len(outputs) >= 50 and len(layouts) >= 90 and len(palettes) >= 20
    assert relations == {-1, 0, 1} and smaller >= 40
    assert features['contact'] == {False, True}
    assert features['inside'] == {0, 1, 2}
    assert features['top'] == {-1, 0, 1}
    assert features['area'] == {-1, 0, 1}
    assert features['box'] == {False, True}
    assert features['shape'] == {'pixel', 'line', 'solid', 'frame', 'other'}
    assert len(features['background']) >= 3 and varied_palettes >= 35
    for seed, nt, ns in ((100, 2, 1), (101, 10, 1), (102, 10, 4), (103, None, 1), (104, None, 4), (105, 2, None), (106, 10, None)):
        p = generate_puzzle(seed, nt, ns)
        validate_generated_puzzle(p)
        assert _equal_puzzles(p, generate_puzzle(seed, nt, ns))
        assert nt is None or len(p['train']) == nt
        assert ns is None or len(p['test']) == ns
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(123, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
