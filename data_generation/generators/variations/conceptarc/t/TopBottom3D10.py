import numpy as np


_SOURCE_ROWS = (
    '111111000000000/122221000000000/122221333333300/122221000000300/111111333333300/003000000000300/003333333333300/003000000000300/003333333333300/003000000000300/003333333333300/003000000000300/003333333333300/003000000000300/003333333333300/000000000000000',
    '00000088888000000/00000084448000000/06666684448666660/06060688888606060/06060606060606060/06060606060606060/06060606060606060/06060606060606060/06666666666666660/00000000000000000/00000000000000000',
    '00000055555500000/07777753333577770/07070753333507070/07070753333507070/07070755555507070/07070707070707070/07070707070707070/07777777777777770/00000000000000000',
    '000000/099990/090090/099222/090222/099222/090090/099990/090090/099990/090090/099990/000000',
    '000000000000000/000000000000000/004444444444400/004033330000400/004434434444400/004030030000400/004434434444400/004033330000400/004444444444400/004000000000400/004444444444400/004000000000400/004444444444400/000000000000000/000000000000000',
)
_SOURCE_INPUTS = tuple(np.array([[int(c) for c in row] for row in s.split('/')], dtype=np.int8) for s in _SOURCE_ROWS)


def _counts(rng, num_train, num_test):
    for value, lo, hi, name in ((num_train, 2, 10, 'num_train'), (num_test, 1, 4, 'num_test')):
        if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
            raise ValueError(f'{name} must be an integer in [{lo}, {hi}]')
    choices = [(a, b) for a in range(2, 11) for b in range(1, 5)
               if a > b and (num_train is None or num_train == a) and (num_test is None or num_test == b)]
    if not choices:
        raise ValueError('num_train must exceed num_test')
    return choices[int(rng.integers(len(choices)))]


def _lattice_mask(h, w, bars, conns):
    mask = np.zeros((h, w), dtype=bool)
    mask[np.asarray(bars), conns[0]:conns[-1] + 1] = True
    mask[bars[0]:bars[-1] + 1, np.asarray(conns)] = True
    return mask


def _recover_horizontal(inp):
    h, w = inp.shape
    border = np.concatenate((inp[0], inp[-1], inp[:, 0], inp[:, -1]))
    bg = int(np.argmax(np.bincount(border, minlength=10)))
    candidates = []
    for color in np.unique(inp):
        color = int(color)
        if color == bg:
            continue
        ys, xs = np.where(inp == color)
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
        if y1 - y0 < 2 or x1 - x0 < 2:
            continue
        for dy in range(2, 9):
            if (y1 - y0) % dy:
                continue
            bars = tuple(range(y0, y1 + 1, dy))
            if not 2 <= len(bars) <= 10:
                continue
            for dx in range(1, 7):
                if (x1 - x0) % dx:
                    continue
                conns = tuple(range(x0, x1 + 1, dx))
                if not 2 <= len(conns) <= 14:
                    continue
                mask = _lattice_mask(h, w, bars, conns)
                missing = mask & (inp != color)
                count = int(np.count_nonzero(missing))
                if count < 2 or count > 0.45 * np.count_nonzero(mask):
                    continue
                if np.any((inp == color) & ~mask) or np.any(inp[missing] == bg):
                    continue
                if any(np.count_nonzero(inp[y, x0:x1 + 1] == color) < 2 for y in bars):
                    continue
                if any(np.count_nonzero(inp[y0:y1 + 1, x] == color) < 2 for x in conns):
                    continue
                candidates.append((int(np.count_nonzero(mask)), color, mask, bars, conns, bg))
    return candidates


def _recover(inp):
    candidates = [(score, color, mask, bars, conns, bg, 'horizontal')
                  for score, color, mask, bars, conns, bg in _recover_horizontal(inp)]
    candidates += [(score, color, mask.T, bars, conns, bg, 'vertical')
                   for score, color, mask, bars, conns, bg in _recover_horizontal(inp.T)]
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0]
    for score, color, mask, *_ in candidates[1:]:
        if color == best[1] and np.array_equal(mask, best[2]):
            continue
        if color == best[1] or 2 * score >= best[0]:
            return None
    return best


def _geometry(rng, compact=False):
    if compact:
        nb = int(rng.integers(4, 9))
        top = int(rng.integers(1, 3))
        bottom = int(rng.integers(1, 3))
        bars = tuple(top + 2 * i for i in range(nb))
        return bars[-1] + bottom + 1, 6, bars, (1, 4)
    for _ in range(100):
        kind = int(rng.integers(3))
        if kind == 0:
            nb, dy = 2, int(rng.integers(4, 9))
            nc, dx = int(rng.integers(5, 12)), int(rng.integers(2, 4))
        elif kind == 1:
            nb, dy = int(rng.integers(5, 10)), int(rng.integers(2, 5))
            nc, dx = int(rng.integers(2, 6)), int(rng.integers(2, 6))
        else:
            nb, dy = int(rng.integers(3, 9)), int(rng.integers(2, 6))
            nc, dx = int(rng.integers(3, 10)), int(rng.integers(2, 5))
        height, width = (nb - 1) * dy + 1, (nc - 1) * dx + 1
        if height > 25 or width > 25:
            continue
        top, bottom, left, right = [int(rng.integers(1, 5)) for _ in range(4)]
        if height + top + bottom > 30 or width + left + right > 30:
            continue
        h, w = height + top + bottom, width + left + right
        bars = tuple(top + i * dy for i in range(nb))
        conns = tuple(left + i * dx for i in range(nc))
        return h, w, bars, conns
    raise RuntimeError('Could not place lattice')


def _clip(value, length, limit):
    return max(0, min(int(value), limit - length))


def _object(rng, inp, bars, conns, style, zone, primary, secondary):
    h, w = inp.shape
    if style == 'frame':
        oh, ow = int(rng.integers(3, 8)), int(rng.integers(4, 10))
    elif style == 'solid':
        oh, ow = int(rng.integers(2, 7)), int(rng.integers(2, 8))
    else:
        oh, ow = int(rng.integers(3, 8)), int(rng.integers(3, 9))
    oh, ow = min(oh, h), min(ow, w)
    if zone == 'top':
        target_y = bars[0]
    elif zone == 'bottom':
        target_y = bars[-1]
    else:
        target_y = int(rng.integers(bars[0], bars[-1] + 1))
    if zone == 'left':
        target_x = conns[0]
    elif zone == 'right':
        target_x = conns[-1]
    else:
        target_x = int(rng.integers(conns[0], conns[-1] + 1))
    yt = _clip(target_y - int(rng.integers(oh)), oh, h)
    xl = _clip(target_x - int(rng.integers(ow)), ow, w)
    if zone == 'top' and rng.integers(3) == 0:
        yt = 0
    if zone == 'left' and rng.integers(3) == 0:
        xl = 0
    if zone == 'right' and rng.integers(3) == 0:
        xl = w - ow
    if zone == 'bottom' and rng.integers(3) == 0:
        yt = h - oh
    patch = np.zeros((oh, ow), dtype=bool)
    if style == 'solid':
        patch[:] = True
        inp[yt:yt + oh, xl:xl + ow][patch] = primary
    elif style == 'frame':
        local = inp[yt:yt + oh, xl:xl + ow]
        fill = int(rng.integers(4))
        for y in range(1, oh - 1):
            for x in range(1, ow - 1):
                if fill == 0 or (fill == 1 and (x + y) % 2 == 0) or (fill == 2 and y % 2 == 1) or (fill == 3 and x % 2 == 1):
                    local[y, x] = secondary
        local[0, :] = primary
        local[-1, :] = primary
        local[:, 0] = primary
        local[:, -1] = primary
    else:
        motif = int(rng.integers(4))
        for y in range(oh):
            for x in range(ow):
                if motif == 0:
                    patch[y, x] = (x + y) % 2 == 0 or y in (0, oh - 1)
                elif motif == 1:
                    patch[y, x] = y in (0, oh - 1) or x in (0, ow - 1) and y % 2 == 0
                elif motif == 2:
                    patch[y, x] = y in (0, oh - 1) or (x + 2 * y) % 3 == 0
                else:
                    patch[y, x] = x in (0, ow - 1) or (x + y) % 3 == 0
        inp[yt:yt + oh, xl:xl + ow][patch] = primary


def _pair(rng, style, zone, palette, vertical):
    bg, lattice, primary, secondary = palette
    for _ in range(300):
        h, w, bars, conns = _geometry(rng, style == 'solid' and rng.integers(5) == 0)
        mask = _lattice_mask(h, w, bars, conns)
        inp = np.full((h, w), bg, dtype=np.int8)
        inp[mask] = lattice
        _object(rng, inp, bars, conns, style, zone, primary, secondary)
        out = inp.copy()
        out[mask] = lattice
        if np.count_nonzero(inp != out) < 2:
            continue
        if not np.any((inp != bg) & ~mask):
            continue
        if style == 'frame' and not (np.any(out == primary) and np.any(out == secondary)):
            continue
        if vertical:
            inp, out = inp.T.copy(), out.T.copy()
        recovered = _recover(inp)
        if recovered is None or recovered[1] != lattice or recovered[5] != bg or not np.array_equal(recovered[2], mask.T if vertical else mask):
            continue
        if any(inp.shape == source.shape and np.array_equal(inp, source) for source in _SOURCE_INPUTS):
            continue
        return {'input': inp, 'output': out}
    raise RuntimeError('Could not construct an unambiguous pair')


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    ntrain, ntest = _counts(rng, num_train, num_test)
    regime = int(rng.integers(4))
    shared_palette = bool(rng.integers(5) == 0)
    fixed_bg = bool(rng.integers(3) != 0)
    vertical = bool(rng.integers(4) == 0)
    digits = list(int(x) for x in rng.permutation(10))
    if fixed_bg:
        base = (0,) + tuple(x for x in digits if x != 0)[:3]
    else:
        base = tuple(digits[:4])
    if regime == 0:
        train_styles = ['frame'] * ntrain
        test_styles = ['solid', 'motif'][:ntest] if ntest >= 2 else [('frame', 'solid', 'motif')[int(rng.integers(3))]]
        while len(test_styles) < ntest:
            test_styles.append(('frame', 'solid', 'motif')[int(rng.integers(3))])
    elif regime == 1:
        train_styles = ['frame', 'solid'] + [('frame', 'solid', 'motif')[int(rng.integers(3))] for _ in range(ntrain - 2)]
        test_styles = ['motif'] + [('frame', 'solid', 'motif')[int(rng.integers(3))] for _ in range(ntest - 1)]
    elif regime == 2:
        train_styles = ['solid', 'motif'] + [('frame', 'solid', 'motif')[int(rng.integers(3))] for _ in range(ntrain - 2)]
        test_styles = ['frame'] + [('frame', 'solid', 'motif')[int(rng.integers(3))] for _ in range(ntest - 1)]
    else:
        train_styles = ['frame', 'motif'] + [('frame', 'solid', 'motif')[int(rng.integers(3))] for _ in range(ntrain - 2)]
        test_styles = [('frame', 'solid', 'motif')[int(rng.integers(3))] for _ in range(ntest)]
    zones = ('top', 'bottom', 'left', 'right', 'interior')
    def palette(i):
        if shared_palette:
            return base
        if fixed_bg:
            a = base[1:]
            if i >= 3 and rng.integers(3) == 0:
                a = tuple(int(x) for x in rng.choice([d for d in range(10) if d != 0], size=3, replace=False))
            return (0, a[i % 3], a[(i + 1) % 3], a[(i + 2) % 3])
        return tuple(base[(i + k) % 4] for k in range(4))
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, styles in (('train', train_styles), ('test', test_styles)):
        for j, style in enumerate(styles):
            i = len(seen)
            if regime == 0 and split == 'train':
                zone = 'top'
            elif style == 'solid' and regime == 0 and split == 'test':
                zone = ('left', 'right')[int(rng.integers(2))]
            elif style == 'motif' and regime == 0 and split == 'test':
                zone = 'interior'
            else:
                zone = zones[int(rng.integers(len(zones)))]
            for _ in range(30):
                pair = _pair(np.random.default_rng(int(rng.integers(0, 2**63))), style, zone, palette(i), vertical)
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('Duplicate inputs')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen_inputs, seen_pairs = set(), set()
    demonstrated_overlap = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            inp, out = pair['input'], pair['output']
            assert inp.shape == out.shape
            ikey = (inp.shape, inp.tobytes())
            pkey = (ikey, out.tobytes())
            assert ikey not in seen_inputs and pkey not in seen_pairs
            seen_inputs.add(ikey)
            seen_pairs.add(pkey)
            assert not any(inp.shape == source.shape and np.array_equal(inp, source) for source in _SOURCE_INPUTS)
            recovered = _recover(inp)
            assert recovered is not None, 'Input lattice is ambiguous'
            _, color, mask, bars, conns, bg, orientation = recovered
            assert np.array_equal(out, np.where(mask, color, inp))
            diff = inp != out
            assert np.count_nonzero(diff) >= 2 and np.all(mask[diff])
            assert np.any((inp != bg) & ~mask)
            assert np.any(inp == bg)
            assert len(bars) >= 2 and len(conns) >= 2
            if split == 'train':
                demonstrated_overlap = True
    assert demonstrated_overlap


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(pa[k], pb[k]) for pa, pb in zip(a[s], b[s]) for k in ('input', 'output')) for s in ('train', 'test'))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    shapes, layouts, palettes, topologies, orientations, puzzle_keys = set(), set(), set(), set(), set(), set()
    variable_lattice = variable_bg = shared_roles = role_swaps = 0
    two_many = multi_few = narrow = broad = seven_bars = 0
    bar_only = connector_only = both = edge_contact = sparse = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signatures = []
        local_lattice, local_bg = set(), set()
        local_colors = []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                _, color, mask, bars, conns, bg, orientation = _recover(inp)
                assert np.array_equal(out, np.where(mask, color, inp))
                assert np.count_nonzero(inp != out) >= 2
                shapes.add(inp.shape)
                layouts.add((len(bars), bars[1] - bars[0], len(conns), conns[1] - conns[0]))
                topologies.add((len(bars), len(conns)))
                palettes.add((bg, color, tuple(sorted(set(int(x) for x in np.unique(inp)) - {bg, color}))))
                orientations.add(orientation)
                local_lattice.add(color)
                local_bg.add(bg)
                local_colors.append(set(int(x) for x in np.unique(inp)) - {bg, color})
                canonical_input = inp if orientation == 'horizontal' else inp.T
                canonical_diff = (inp != out) if orientation == 'horizontal' else (inp != out).T
                canonical_mask = mask if orientation == 'horizontal' else mask.T
                bar_hit = any(np.any(canonical_diff[y]) for y in bars)
                connector_hit = any(np.any(canonical_diff[:, x]) for x in conns)
                bar_only += bar_hit and not connector_hit
                connector_only += connector_hit and not bar_hit
                both += bar_hit and connector_hit
                obj = np.isin(canonical_input, tuple(local_colors[-1]))
                ys, xs = np.where(obj)
                y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
                edge_contact += y0 == 0 or x0 == 0 or y1 == canonical_input.shape[0] - 1 or x1 == canonical_input.shape[1] - 1
                sparse += np.count_nonzero(obj) < 0.8 * (y1 - y0 + 1) * (x1 - x0 + 1)
                two_many += len(bars) == 2 and len(conns) >= 5
                multi_few += len(bars) >= 5 and len(conns) <= 5
                seven_bars += len(bars) >= 7
                width = conns[-1] - conns[0] + 1
                narrow += width <= 6
                broad += width >= 18
                signatures.append((inp.shape, inp.tobytes(), out.tobytes()))
        puzzle_keys.add(tuple(signatures))
        variable_lattice += len(local_lattice) > 1
        variable_bg += len(local_bg) > 1
        shared_roles += len(local_lattice) == 1 and len(local_bg) == 1
        colors = list(local_lattice)
        role_swaps += any(c in objects for c in colors for objects in local_colors)
    assert len(shapes) >= 35 and len(layouts) >= 45 and len(topologies) >= 12
    assert len(palettes) >= 40 and len(puzzle_keys) == 42
    assert {2, 7} <= {x[0] for x in topologies}
    assert variable_lattice >= 20 and variable_bg >= 5 and shared_roles >= 2 and role_swaps >= 20
    assert two_many >= 30 and multi_few >= 30 and seven_bars >= 20
    assert narrow >= 10 and broad >= 20 and edge_contact >= 30 and sparse >= 20
    assert bar_only >= 3 and connector_only >= 5 and both >= 50
    assert orientations == {'horizontal', 'vertical'}
    for seed, train, test in ((9, 2, 1), (10, 10, 1), (11, 10, 4), (12, None, 1), (13, None, 4), (14, 2, None), (15, 10, None)):
        a, b = generate_puzzle(seed, train, test), generate_puzzle(seed, train, test)
        assert _equal(a, b)
        if train is not None:
            assert len(a['train']) == train
        if test is not None:
            assert len(a['test']) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(123, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError((train, test))
