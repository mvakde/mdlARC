"""Anomaly interference: registered discrepancy parity controls exact stamping.

No parent imports, external files, learned dictionaries, or generation state are
used by solve. Sampling preferences are separate from the input-observable rule.
"""

from collections import Counter
from functools import lru_cache
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _bbox(mask):
    rr, cc = np.nonzero(mask)
    if not len(rr):
        raise ValueError("empty visible encoding")
    return int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1


def _separated(a, b):
    # Half-open boxes: a complete blank line must intervene.
    return a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2]


def _transforms(a):
    return [f(np.rot90(a, k)) for k in range(4)
            for f in (lambda v: v, np.fliplr)]


def _asymmetric(a):
    return all(not np.array_equal(a, t) for t in _transforms(a)[1:])


def _touches_sides(mask):
    return all(np.any(v) for v in (mask[0], mask[-1], mask[:, 0], mask[:, -1]))


def _read_input(input_grid):
    """Recover the unique lattice, master, deviations, and stamp from cells."""
    g = _grid(input_grid)
    bg = int(g[0, 0])
    if not (np.all(g[[0, -1], :] == bg) and np.all(g[:, [0, -1]] == bg)):
        raise ValueError("the source canvas needs a uniform background rim")
    interpretations = []
    for lattice in map(int, np.unique(g)):
        if lattice == bg:
            continue
        r0, r1, c0, c1 = _bbox(g == lattice)
        board = g[r0:r1, c0:c1]
        if not (np.all(board[[0, -1], :] == lattice)
                and np.all(board[:, [0, -1]] == lattice)):
            continue
        rs = np.flatnonzero(np.all(board == lattice, axis=1))
        cs = np.flatnonzero(np.all(board == lattice, axis=0))
        if len(rs) < 3 or len(cs) < 4:
            continue
        if (not np.all(np.diff(rs) == rs[1])
                or not np.all(np.diff(cs) == cs[1])):
            continue
        ph, pw = int(rs[1] - 1), int(cs[1] - 1)
        if min(ph, pw) < 1:
            continue
        panels = [[board[r+1:s, c+1:d].copy()
                   for c, d in zip(cs, cs[1:])]
                  for r, s in zip(rs, rs[1:])]
        if any(np.any(p == lattice) for row in panels for p in row):
            continue
        colors = set(map(int, np.unique(board))) - {lattice, bg}
        if len(colors) != 1:
            continue
        signal = next(iter(colors))
        panel_counts = Counter(_key(p) for row in panels for p in row)
        maximum = max(panel_counts.values())
        winners = [k for k, n in panel_counts.items() if n == maximum]
        if len(winners) != 1:
            continue
        master = next(p for row in panels for p in row if _key(p) == winners[0])
        if not (np.any(master == bg) and np.any(master == signal)):
            continue
        columns, selected, deltas = [], [], []
        valid = True
        for row in panels:
            exceptions = [c for c, p in enumerate(row) if not np.array_equal(p, master)]
            if len(exceptions) != 1:
                valid = False
                break
            c = exceptions[0]
            columns.append(c)
            selected.append(row[c])
            deltas.append(row[c] != master)
        if not valid:
            continue
        outside = g != bg
        outside[r0:r1, c0:c1] = False
        payload = set(map(int, g[outside]))
        if len(payload) < 2 or payload & {bg, lattice, signal}:
            continue
        sb = _bbox(outside)
        if not _separated((r0, r1, c0, c1), sb):
            continue
        stamp = g[sb[0]:sb[1], sb[2]:sb[3]].copy()
        if not _touches_sides(stamp != bg):
            continue
        if ph * stamp.shape[0] > 30 or pw * stamp.shape[1] > 30:
            continue
        multiplicity = np.sum(deltas, axis=0, dtype=int)
        interpretations.append(dict(
            background=bg, lattice=lattice, signal=signal, payload=payload,
            panel_shape=(ph, pw), bands=len(panels), copies=len(panels[0]),
            board_box=(r0, r1, c0, c1), stamp_box=sb, panels=panels,
            master=master.copy(), selected=selected, columns=tuple(columns),
            deltas=np.array(deltas, dtype=bool), multiplicity=multiplicity,
            parity=(multiplicity % 2 == 1), stamp=stamp,
        ))
    if len(interpretations) != 1:
        raise ValueError("input must have one unambiguous lattice/stamp interpretation")
    return interpretations[0]


def _expanded(mask, stamp, bg):
    """Cell-index prediction, separate from the generator's block writes."""
    sh, sw = stamp.shape
    rr, cc = np.indices((mask.shape[0] * sh, mask.shape[1] * sw))
    return np.where(mask[rr // sh, cc // sw], stamp[rr % sh, cc % sw], bg).astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _read_input(input_grid)
    return _expanded(info['parity'], info['stamp'], info['background'])


def _block_write(mask, stamp, bg):
    sh, sw = stamp.shape
    out = np.full((mask.shape[0] * sh, mask.shape[1] * sw), bg, dtype=np.int8)
    for r, c in np.argwhere(mask):
        out[r*sh:(r+1)*sh, c*sw:(c+1)*sw] = stamp
    return out


def _fits(ph, pw, bands, copies, sh, sw):
    bh, bw = bands * (ph + 1) + 1, copies * (pw + 1) + 1
    return (ph * sh <= 30 and pw * sw <= 30
            and ((max(bh, sh) + 2 <= 30 and bw + sw + 3 <= 30)
                 or (bh + sh + 3 <= 30 and max(bw, sw) + 2 <= 30)))


@lru_cache(maxsize=12)
def _geometries(regime, split, witness=0):
    choices = []
    for ph in range(2, 9):
        for pw in range(2, 9):
            if split == 'train' and max(ph, pw) > 4:
                continue
            if split == 'test' and regime != 'panel_growth' and max(ph, pw) > 5:
                continue
            if split == 'test' and regime == 'panel_growth' and max(ph, pw) < 5:
                continue
            if witness == 1 and min(ph, pw) < 3:
                continue
            for bands in range(2, 7):
                if split == 'train' and bands > 4:
                    continue
                if split == 'test' and regime == 'band_growth' and bands < 5:
                    continue
                if split == 'test' and regime != 'band_growth' and bands > 4:
                    continue
                if witness and bands != (4 if witness == 1 else 3):
                    continue
                for copies in range(3, 7):
                    if witness == 2 and copies % 2 == 0:
                        continue
                    for sh in range(2, 7):
                        for sw in range(2, 7):
                            if split == 'train' and max(sh, sw) > (3 if regime == 'stamp_growth' else 5):
                                continue
                            if split == 'test' and regime == 'stamp_growth' and max(sh, sw) < 4:
                                continue
                            if witness == 1 and min(sh, sw) < 3:
                                continue
                            if _fits(ph, pw, bands, copies, sh, sw):
                                choices.append((ph, pw, bands, copies, sh, sw))
    if not choices:
        raise RuntimeError("empty feasible geometry domain")
    return tuple(choices)


def _choose(rng, items):
    return items[int(rng.integers(len(items)))]


def _master_pattern(rng, shape):
    h, w = shape
    r, c = np.indices(shape)
    style = int(rng.integers(7))
    if style == 0:
        mask = (r if rng.random() < .5 else c) % int(rng.integers(2, 4)) == 0
    elif style == 1:
        mask = (r + c) % int(rng.integers(2, 4)) == 0
    elif style == 2:
        mask = (r == 0) | (r == h-1) | (c == 0) | (c == w-1)
    elif style == 3:
        mask = (r == int(rng.integers(h))) | (c == int(rng.integers(w)))
    elif style == 4:
        mask = (r % 2 == 0) & (c % 2 == 0)
    else:
        mask = rng.random(shape) < float(rng.uniform(.2, .8))
    if rng.random() < .25:
        mask |= mask[::-1, ::-1]
    if rng.random() < .25:
        mask = ~mask
    if not np.any(mask):
        mask.flat[int(rng.integers(mask.size))] = True
    if np.all(mask):
        mask.flat[int(rng.integers(mask.size))] = False
    return mask


def _delta_pattern(rng, shape):
    h, w = shape
    r, c = np.indices(shape)
    style = int(rng.integers(8))
    mask = np.zeros(shape, dtype=bool)
    if style == 0:
        mask.flat[int(rng.integers(mask.size))] = True
    elif style == 1:
        if rng.random() < .5:
            mask[int(rng.integers(h)), :] = True
        else:
            mask[:, int(rng.integers(w))] = True
    elif style == 2:
        mask = (r + int(rng.integers(w))) % w == c
    elif style in (3, 4):
        r0, r1 = sorted(map(int, rng.choice(h + 1, 2, replace=False)))
        c0, c1 = sorted(map(int, rng.choice(w + 1, 2, replace=False)))
        mask[r0:r1, c0:c1] = True
        if style == 4 and r1-r0 >= 3 and c1-c0 >= 3:
            mask[r0+1:r1-1, c0+1:c1-1] = False
    elif style == 5:
        mask = (r % 2 == int(rng.integers(2))) & (c % 2 == int(rng.integers(2)))
    else:
        count = int(rng.integers(1, mask.size + 1))
        mask.flat[rng.choice(mask.size, count, replace=False)] = True
    if not np.any(mask):
        mask.flat[int(rng.integers(mask.size))] = True
    if rng.random() < .5:
        mask = mask[::-1, :]
    if rng.random() < .5:
        mask = mask[:, ::-1]
    return mask.copy()


def _principal_witness(rng, master):
    """Construct double cancellation, triple survival, and both edit signs."""
    h, w = master.shape
    # The final mask leaves a complete outer line blank. This visibly prohibits
    # tight-cropping the placement field while retaining a fully asymmetric mask.
    for _ in range(200):
        available = np.ones((h, w), dtype=bool)
        side = int(rng.integers(4))
        if side < 2:
            available[0 if side == 0 else -1, :] = False
        else:
            available[:, 0 if side == 2 else -1] = False
        survivors = rng.choice(np.flatnonzero(available), 4, replace=False)
        p = np.zeros(master.shape, dtype=bool)
        p.flat[survivors] = True
        if _asymmetric(p):
            break
    else:
        # All witness shapes are 3..4 in each axis; this fixed geometry is only
        # a bounded fallback, with random permutation and master/stamp content.
        survivors = np.array([0, 1, w, 2*w+1])
        p = np.zeros(master.shape, dtype=bool)
        p.flat[survivors] = True
        if not _asymmetric(p):
            raise RuntimeError("could not construct asymmetric parity witness")
    b, c, d, e = map(int, rng.permutation(survivors))
    a = int(rng.choice(np.flatnonzero(~p)))
    master.flat[[a, c, e]] = False
    master.flat[[b, d]] = True
    masks = []
    for cells in ((a, b), (a, b, c), (b, d), (e,)):
        mask = np.zeros(master.shape, dtype=bool)
        mask.flat[list(cells)] = True
        masks.append(mask)
    rng.shuffle(masks)
    return masks


def _deltas(rng, master, bands, companion=False):
    shape = master.shape
    mode = int(rng.integers(5))
    common = _delta_pattern(rng, shape)
    masks = []
    for b in range(bands):
        if mode == 1 and b % 2 == 1:
            mask = masks[-1].copy()
        elif mode == 2:
            mask = common.copy()  # Complete or partial cancellation by parity.
        elif mode == 3:
            mask = _delta_pattern(rng, shape) ^ common
        elif mode == 4:
            mask = np.zeros(shape, dtype=bool)
            # Translation is clipped, not toroidal.
            dr, dc = int(rng.integers(-1, 2)), int(rng.integers(-1, 2))
            for r, c in np.argwhere(common):
                if 0 <= r+dr < shape[0] and 0 <= c+dc < shape[1]:
                    mask[r+dr, c+dc] = True
        else:
            mask = _delta_pattern(rng, shape)
        if not np.any(mask):
            mask.flat[int(rng.integers(mask.size))] = True
        masks.append(mask)
    if companion and not np.any(np.logical_xor.reduce(masks)):
        # Change one event while keeping the last exception visibly nonempty.
        candidates = [i for i in range(master.size)
                      if not masks[-1].flat[i] or masks[-1].sum() > 1]
        index = int(_choose(rng, candidates))
        masks[-1].flat[index] = not masks[-1].flat[index]
    return masks


def _stamp(rng, shape, bg, payload, witness=False):
    h, w = shape
    r, c = np.indices(shape)
    for _ in range(120):
        style = int(rng.integers(6))
        if witness:
            style = 5
        if style == 0:
            labels = r if rng.random() < .5 else c
        elif style == 1:
            labels = np.minimum.reduce([r, c, h-1-r, w-1-c])
        elif style == 2:
            labels = r + c
        elif style == 3:
            labels = np.minimum(r, h-1-r) + np.minimum(c, w-1-c)
        elif style == 4:
            tile = rng.integers(len(payload), size=(2, 2))
            labels = tile[r % 2, c % 2]
        else:
            labels = rng.integers(len(payload), size=shape)
        stamp = payload[(labels + int(rng.integers(len(payload)))) % len(payload)].copy()
        holes = np.zeros(shape, dtype=bool)
        if witness:
            holes[int(rng.integers(1, h-1)), int(rng.integers(1, w-1))] = True
            boundary = [(i, j) for i in range(h) for j in range(w)
                        if i in (0, h-1) or j in (0, w-1)]
            holes[_choose(rng, boundary)] = True
        elif rng.random() < .65:
            holes = rng.random(shape) < float(rng.uniform(.08, .35))
        stamp[holes] = bg
        if (not _touches_sides(stamp != bg)
                or len(set(map(int, np.unique(stamp))) - {bg}) < 2):
            continue
        if witness and not _asymmetric(stamp):
            continue
        return stamp
    # A guaranteed tight multicolor shape; asymmetric for every witness size.
    stamp = np.full(shape, payload[0], dtype=np.int8)
    stamp[0, 1] = payload[1]
    stamp[-1, 0] = payload[1]
    if witness:
        stamp[1, 1] = bg
        stamp[0, -1] = bg
    return stamp


def _board(master, masks, columns, copies, bg, lattice, signal):
    ph, pw = master.shape
    board = np.full((len(masks)*(ph+1)+1, copies*(pw+1)+1), lattice, dtype=np.int8)
    for b, mask in enumerate(masks):
        for c in range(copies):
            bits = master ^ mask if c == columns[b] else master
            r0, c0 = 1+b*(ph+1), 1+c*(pw+1)
            board[r0:r0+ph, c0:c0+pw] = np.where(bits, signal, bg)
    return board


def _layout(rng, board, stamp, bg):
    bh, bw = board.shape
    sh, sw = stamp.shape
    layouts = []
    for gap in range(1, 4):
        if max(bh, sh)+2 <= 30 and bw+sw+gap+2 <= 30:
            layouts += [('left', gap), ('right', gap)]
        if bh+sh+gap+2 <= 30 and max(bw, sw)+2 <= 30:
            layouts += [('above', gap), ('below', gap)]
    direction, gap = _choose(rng, layouts)
    horizontal = direction in ('left', 'right')
    base_h = max(bh, sh)+2 if horizontal else bh+sh+gap+2
    base_w = bw+sw+gap+2 if horizontal else max(bw, sw)+2
    pad = int(_choose(rng, (1, 3, 30)))
    h = base_h + int(rng.integers(min(30-base_h, pad)+1))
    w = base_w + int(rng.integers(min(30-base_w, pad)+1))
    if horizontal:
        start = 1 + int(rng.integers(w-base_w+1))
        bc, sc = (start, start+bw+gap) if direction == 'right' else (start+sw+gap, start)
        br = int(rng.integers(1, h-bh))
        sr = int(rng.integers(1, h-sh))
    else:
        start = 1 + int(rng.integers(h-base_h+1))
        br, sr = (start, start+bh+gap) if direction == 'below' else (start+sh+gap, start)
        bc = int(rng.integers(1, w-bw))
        sc = int(rng.integers(1, w-sw))
    g = np.full((h, w), bg, dtype=np.int8)
    g[br:br+bh, bc:bc+bw] = board
    g[sr:sr+sh, sc:sc+sw] = stamp
    return g


def _pair(rng, geometry, palette, witness=0):
    ph, pw, bands, copies, sh, sw = geometry
    bg, lattice, signal = map(int, palette[:3])
    payload = np.array(palette[3:3+int(rng.integers(2, 6))], dtype=np.int8)
    master = _master_pattern(rng, (ph, pw))
    masks = (_principal_witness(rng, master) if witness == 1
             else _deltas(rng, master, bands, witness == 2))
    columns = list(map(int, rng.integers(copies, size=bands)))
    if witness == 1:
        # Repeated columns and varying positions prohibit permutation/fixed-column
        # inference. Their coordinates have no role in registered interference.
        x = int(rng.integers(copies))
        y = (x + int(rng.integers(1, copies))) % copies
        columns = [x, x, y, int(rng.integers(copies))]
        rng.shuffle(columns)
    stamp = _stamp(rng, (sh, sw), bg, payload, witness == 1)
    board = _board(master, masks, columns, copies, bg, lattice, signal)
    parity = np.logical_xor.reduce(masks)
    return {'input': _layout(rng, board, stamp, bg),
            'output': _block_write(parity, stamp, bg)}


def _check_count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _check_count('num_train', num_train, 2, 10)
    ne = _check_count('num_test', num_test, 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    regime = str(_choose(rng, ('band_growth', 'panel_growth', 'stamp_growth')))
    palette_mode = str(_choose(rng, ('stable', 'varied', 'varied', 'exchange')))
    base_palette = rng.permutation(10)
    first = _choose(rng, _geometries(regime, 'train', 1))
    second_choices = [g for g in _geometries(regime, 'train', 2)
                      if g[:2] != first[:2] and g[4:] != first[4:]]
    second = _choose(rng, second_choices)
    configurations = [('train', first, 1), ('train', second, 2)]
    configurations += [('train', _choose(rng, _geometries(regime, 'train')), 0)
                       for _ in range(nt-2)]
    configurations += [('test', _choose(rng, _geometries(regime, 'test')), 0)
                       for _ in range(ne)]
    puzzle, seen = {'train': [], 'test': []}, set()
    for index, (split, geometry, witness) in enumerate(configurations):
        palette = (rng.permutation(10) if palette_mode == 'varied'
                   else np.roll(base_palette, index) if palette_mode == 'exchange'
                   else base_palette)
        for _ in range(40):
            pair = _pair(rng, geometry, palette, witness)
            if _key(pair['input']) not in seen:
                seen.add(_key(pair['input']))
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("failed to construct a distinct input")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative_predictions(info):
    bg, stamp, count = info['background'], info['stamp'], info['multiplicity']
    ds, master, signal = info['deltas'], info['master'], info['signal']
    additions = ds & (master == bg)
    removals = ds & (master == signal)
    masks = {
        'union': count > 0, 'intersection': count == info['bands'],
        'exactly_once': count == 1, 'first_delta': ds[0], 'last_delta': ds[-1],
        'additions_only': np.logical_xor.reduce(additions),
        'removals_only': np.logical_xor.reduce(removals),
        'exception_ink_union': np.logical_or.reduce([p == signal for p in info['selected']]),
        'exception_ink_xor': np.logical_xor.reduce([p == signal for p in info['selected']]),
        'master_ink': master == signal,
        'xor_every_panel': np.logical_xor.reduce([p == signal for row in info['panels'] for p in row]),
        'all_cells': np.ones(master.shape, dtype=bool),
    }
    predictions = {name: _expanded(mask, stamp, bg) for name, mask in masks.items()}
    parity = info['parity']
    if np.any(parity):
        r0, r1, c0, c1 = _bbox(parity)
        predictions['crop_parity'] = _expanded(parity[r0:r1, c0:c1], stamp, bg)
    else:
        predictions['crop_parity'] = np.array([[bg]], dtype=np.int8)
    filled_stamp = np.where(stamp == bg, next(iter(sorted(info['payload']))), stamp)
    predictions['fill_stamp_holes'] = _expanded(parity, filled_stamp, bg)
    predictions['flat_blocks'] = _expanded(parity, np.full(stamp.shape, stamp.flat[0]), bg)
    predictions['coarse_mask'] = np.where(parity, stamp[stamp != bg][0], bg).astype(np.int8)
    for i, t in enumerate(_transforms(parity)[1:], 1):
        predictions[f'mask_transform_{i}'] = _expanded(t, stamp, bg)
    for i, t in enumerate(_transforms(stamp)[1:], 1):
        predictions[f'stamp_transform_{i}'] = _expanded(parity, t, bg)
    for c in range(info['copies']):
        fixed = np.logical_xor.reduce([row[c] != master for row in info['panels']])
        predictions[f'fixed_column_{c}'] = _expanded(fixed, stamp, bg)
    return predictions


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {'train', 'test'}:
        raise ValueError("puzzle must contain exactly train/test")
    if not all(isinstance(puzzle[s], list) for s in puzzle):
        raise ValueError("splits must be lists")
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _check_count('num_train', nt, 2, 10)
    _check_count('num_test', ne, 1, 4)
    assert nt > ne
    seen, infos = set(), {'train': [], 'test': []}
    rejected = set()
    common_columns = None
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _key(inp)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            info = _read_input(inp)
            infos[split].append(info)
            ph, pw = info['panel_shape']
            sh, sw = info['stamp'].shape
            assert out.shape == (ph*sh, pw*sw), 'full panel extent must be expanded'
            assert np.array_equal(out, solve(inp)), 'incorrect solved output'
            # Directly inspect all macro blocks, including inactive and hole cells.
            for r in range(ph):
                for c in range(pw):
                    block = out[r*sh:(r+1)*sh, c*sw:(c+1)*sw]
                    if info['multiplicity'][r, c] % 2:
                        assert np.array_equal(block, info['stamp'])
                    else:
                        assert np.all(block == info['background'])
            assert not ({info['lattice'], info['signal']} & set(map(int, np.unique(out))))
            assert np.all(info['deltas'].sum(axis=(1, 2)) > 0)
            if split == 'train':
                for name, prediction in _alternative_predictions(info).items():
                    if not np.array_equal(prediction, out):
                        rejected.add(name)
                compatible_columns = {c for c in range(info['copies'])
                                      if f'fixed_column_{c}' not in rejected}
                common_columns = (compatible_columns if common_columns is None
                                  else common_columns & compatible_columns)
    train = infos['train']
    required = {'union', 'intersection', 'exactly_once', 'first_delta', 'last_delta',
                'additions_only', 'removals_only', 'exception_ink_union', 'exception_ink_xor',
                'master_ink', 'xor_every_panel', 'all_cells', 'crop_parity',
                'fill_stamp_holes', 'flat_blocks', 'coarse_mask'}
    required |= {f'{role}_transform_{i}' for role in ('mask', 'stamp') for i in range(1, 8)}
    assert required <= rejected, 'training must reject documented simpler alternatives'
    assert not common_columns, 'training must reject fixed-column selection'
    assert len({i['panel_shape'] for i in train}) >= 2
    assert len({i['stamp'].shape for i in train}) >= 2
    assert any(np.any((i['multiplicity'] >= 2) & (i['multiplicity'] % 2 == 0)) for i in train)
    assert any(np.any((i['multiplicity'] >= 3) & (i['multiplicity'] % 2 == 1)) for i in train)
    assert any(_asymmetric(i['parity']) and _asymmetric(i['stamp']) for i in train)
    assert any(len(set(i['columns'])) > 1 and len(set(i['columns'])) < i['bands'] for i in train)
    assert any(np.any(i['stamp'][1:-1, 1:-1] == i['background']) for i in train)
    assert any(any(np.any(side == i['background']) for side in
                   (i['stamp'][0], i['stamp'][-1], i['stamp'][:, 0], i['stamp'][:, -1])) for i in train)
    edits = [(int(np.sum(d & (i['master'] == i['background']))),
              int(np.sum(d & (i['master'] == i['signal']))))
             for i in train for d in i['deltas']]
    assert any(a == r and a > 0 for a, r in edits), 'balanced edit evidence'
    assert any(a > 0 and r == 0 for a, r in edits), 'pure addition evidence'
    assert any(r > 0 and a == 0 for a, r in edits), 'pure removal evidence'
    # Verify an actual geometric shift, not a latent regime label.
    for i in infos['test']:
        assert (i['bands'] > max(t['bands'] for t in train)
                or max(i['panel_shape']) > max(max(t['panel_shape']) for t in train)
                or max(i['stamp'].shape) > max(max(t['stamp'].shape) for t in train))


def _puzzle_key(puzzle):
    return tuple(tuple((_key(np.asarray(p['input'])), _key(np.asarray(p['output'])))
                       for p in puzzle[s]) for s in ('train', 'test'))


def _normalized(a):
    mapping = {}
    return a.shape, tuple(mapping.setdefault(int(v), len(mapping)) for v in a.ravel())


def self_test():
    """Bounded deterministic suite: 42 seeds, legal counts, and negative cases."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    for seed in (0, 7, 41):
        assert _puzzle_key(puzzles[seed]) == _puzzle_key(generate_puzzle(seed))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000 + nt*10 + ne, nt, ne)
            assert (len(p['train']), len(p['test'])) == (nt, ne)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    for seed in (-1, True, 1.5):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal seed accepted')
    geometry, panels, stamps, masks, shapes, layouts, palettes = (set() for _ in range(7))
    bands, copies, zero_outputs, boundary_edits = set(), set(), 0, 0
    for puzzle in puzzles:
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                i = _read_input(pair['input'])
                geometry.add((i['panel_shape'], i['stamp'].shape, i['bands'], i['copies']))
                panels.add(_normalized(i['master']))
                stamps.add(_normalized(i['stamp']))
                masks.add(_key(i['parity']))
                shapes.add((pair['input'].shape, pair['output'].shape))
                bands.add(i['bands'])
                copies.add(i['copies'])
                palettes.add((i['background'], i['lattice'], i['signal']))
                a, b = i['board_box'], i['stamp_box']
                layouts.add('left' if b[3] < a[2] else 'right' if a[3] < b[2]
                            else 'above' if b[1] < a[0] else 'below')
                zero_outputs += int(not np.any(i['parity']))
                boundary_edits += int(np.any(i['deltas'][:, [0, -1], :])
                                      or np.any(i['deltas'][:, :, [0, -1]]))
    assert len(geometry) >= 80 and len(shapes) >= 60
    assert len(panels) >= 60 and len(stamps) >= 100 and len(masks) >= 60
    assert len(palettes) >= 50
    assert bands == set(range(2, 7)) and len(copies) >= 3
    assert layouts == {'left', 'right', 'above', 'below'}
    assert zero_outputs > 0 and boundary_edits > 50
    # One corrupted label and one deleted visible exception must be rejected.
    for defect in ('output', 'missing_exception'):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]]
               for s in ('train', 'test')}
        pair = bad['train'][0]
        if defect == 'output':
            pair['output'][0, 0] = (int(pair['output'][0, 0])+1) % 10
        else:
            i = _read_input(pair['input'])
            ph, pw = i['panel_shape']
            r0 = i['board_box'][0] + 1
            c0 = i['board_box'][2] + 1 + i['columns'][0]*(pw+1)
            pair['input'][r0:r0+ph, c0:c0+pw] = i['master']
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError(f'accepted {defect}')


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds and all legal count combinations')
