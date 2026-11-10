"""Patch atlas: enlarge the union of edits in rowwise exceptional panels.

No parent code, files, seeds, or labels are consulted by the input-only solver.
Generation and tests keep grids in memory; this module writes no artifacts.
"""
from collections import Counter
import hashlib

import numpy as np


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _segments(plan, bg, axis):
    """Unique occupied contiguous sequences, in the visible clue direction."""
    h, w = plan.shape
    result = []
    for length in range(2, plan.shape[axis] + 1):
        positions = {}
        for r in range(h - (length - 1 if axis == 0 else 0)):
            for c in range(w - (length - 1 if axis == 1 else 0)):
                seq = tuple(int(plan[r + (axis == 0) * k,
                                     c + (axis == 1) * k]) for k in range(length))
                if bg not in seq:
                    positions.setdefault(seq, []).append((r, c))
        for seq, locations in positions.items():
            if len(locations) == 1:
                result.append((locations[0], length, seq))
    return result


def _asymmetric(a):
    transforms = [a[::-1], a[:, ::-1], a[::-1, ::-1]]
    if a.shape[0] == a.shape[1]:
        transforms += [a.T, a.T[::-1, ::-1], np.rot90(a), np.rot90(a, 3)]
    return all(not np.array_equal(a, b) for b in transforms)


def _patch(master, masks, bg, colors):
    counts = np.sum(masks, axis=0)
    plan = np.full(master.shape, bg, dtype=np.int8)
    plan[counts > 0] = np.where(master[counts > 0], colors[0], colors[1])
    return plan, counts


def _recover(value):
    inp = _grid(value)
    bg = int(inp[0, 0])
    assert all(np.all(edge == bg) for edge in (inp[0], inp[-1], inp[:, 0], inp[:, -1]))
    # The frame color has exactly one rectangular lattice bounding all its cells.
    candidates = []
    for color in sorted(set(map(int, np.unique(inp))) - {bg}):
        pts = np.argwhere(inp == color)
        top, left = map(int, pts.min(axis=0))
        bottom, right = map(int, pts.max(axis=0) + 1)
        region = inp[top:bottom, left:right]
        rows = np.flatnonzero(np.all(region == color, axis=1))
        cols = np.flatnonzero(np.all(region == color, axis=0))
        if len(rows) < 3 or len(cols) < 4:
            continue
        if (rows[0] != 0 or cols[0] != 0 or rows[-1] != len(region) - 1
                or cols[-1] != region.shape[1] - 1):
            continue
        if not (np.all(np.diff(rows) == rows[1]) and np.all(np.diff(cols) == cols[1])):
            continue
        frame = np.zeros(region.shape, dtype=bool)
        frame[rows, :] = True
        frame[:, cols] = True
        if np.array_equal(region == color, frame):
            candidates.append((color, top, left, bottom, right, rows, cols))
    assert len(candidates) == 1, 'one unambiguous framed atlas is required'
    frame, top, left, bottom, right, rows, cols = candidates[0]
    ph, pw = int(rows[1] - 1), int(cols[1] - 1)
    bands, copies = len(rows) - 1, len(cols) - 1
    assert min(ph, pw) >= 1 and bands >= 2 and copies >= 3
    panels = [[inp[top + r + 1:top + s, left + c + 1:left + d]
               for c, d in zip(cols, cols[1:])] for r, s in zip(rows, rows[1:])]
    frequencies = Counter(p.tobytes() for row in panels for p in row)
    ranking = frequencies.most_common()
    assert ranking[0][1] > ranking[1][1]
    master = next(p for row in panels for p in row if p.tobytes() == ranking[0][0]).copy()
    colors = tuple(sorted(map(int, np.unique(master))))
    assert len(colors) == 2 and bg not in colors and frame not in colors
    masks, columns, variants = [], [], []
    for row in panels:
        assert all(set(map(int, np.unique(p))) <= set(colors) for p in row)
        odd = [(c, p) for c, p in enumerate(row) if not np.array_equal(p, master)]
        assert len(odd) == 1, 'each band has one visible exceptional bitmap'
        c, variant = odd[0]
        columns.append(c)
        variants.append(variant.copy())
        masks.append(variant != master)
    # Every changed pixel is the other one of the two atlas colors. Thus
    # overlapping edits agree and are combined once, without parity cancellation.
    masks = np.array(masks, dtype=bool)
    plan = np.full((ph, pw), bg, dtype=np.int8)
    for variant, mask in zip(variants, masks):
        assert np.all((plan[mask] == bg) | (plan[mask] == variant[mask]))
        plan[mask] = variant[mask]
    outside = inp != bg
    outside[top:bottom, left:right] = False
    squares = []
    for cells in _components(outside):
        points = np.array(cells)
        y, x = map(int, points.min(axis=0))
        ey, ex = map(int, points.max(axis=0) + 1)
        side = ey - y
        assert side >= 2 and ex - x == side and len(cells) == side * side
        color = int(inp[y, x])
        assert color in colors and np.all(inp[y:ey, x:ex] == color)
        squares.append((y, x, side, color))
    assert len(squares) >= 2
    if len({s[0] for s in squares}) == 1:
        axis = 1
    else:
        assert len({s[1] for s in squares}) == 1
        axis = 0
    squares.sort(key=lambda s: s[axis])
    side = squares[0][2]
    assert all(s[2] == side for s in squares)
    gaps = np.diff([s[axis] for s in squares])
    stride = int(gaps[0])
    assert np.all(gaps == stride) and stride > side
    seq = tuple(s[3] for s in squares)
    matches = [(p, length) for p, length, values in _segments(plan, bg, axis) if values == seq]
    assert len(matches) == 1, 'the combined patch must uniquely locate the seed'
    match, length = matches[0]
    base = (squares[0][0] - match[0] * stride, squares[0][1] - match[1] * stride)
    by, bx = base
    ey, ex = by + (ph - 1) * stride + side, bx + (pw - 1) * stride + side
    assert 1 <= by < ey < inp.shape[0] and 1 <= bx < ex < inp.shape[1]
    assert ey < top or bottom < by or ex < left or right < bx, 'representations need a blank gutter'
    seed_sites = np.zeros(plan.shape, dtype=bool)
    for k in range(length):
        seed_sites[match[0] + (axis == 0) * k, match[1] + (axis == 1) * k] = True
    # Recover exactly the visible input seed, rejecting extra squares or pixels.
    expected_input = np.full(inp.shape, bg, dtype=np.int8)
    expected_input[top:bottom, left:right] = inp[top:bottom, left:right]
    expected = expected_input.copy()
    for (r, c), color in np.ndenumerate(plan):
        y, x = by + r * stride, bx + c * stride
        if seed_sites[r, c]:
            expected_input[y:y + side, x:x + side] = color
        if color != bg:
            expected[y:y + side, x:x + side] = color
    assert np.array_equal(inp, expected_input)
    count_map = masks.sum(axis=0)
    exclusive = tuple(int(np.count_nonzero(m & (count_map == 1) & ~seed_sites)) for m in masks)
    balanced = tuple(np.count_nonzero(v == colors[0]) == np.count_nonzero(master == colors[0])
                     for v in variants)
    added_colors = tuple(sorted(set(map(int, plan[(count_map > 0) & ~seed_sites]))))
    meta = dict(bg=bg, frame=frame, colors=colors, panel_shape=(ph, pw), bands=bands,
                copies=copies, columns=tuple(columns), edits=tuple(map(int, masks.sum(axis=(1, 2)))),
                side=side, gutter=stride - side, axis=axis, match=match, length=length,
                cross=match[1 - axis], base=base, atlas=(top, left), shape=inp.shape,
                plan=plan, masks=masks, master=master, variants=variants, seed_sites=seed_sites,
                exclusive=exclusive, balanced=balanced, added_colors=added_colors,
                overlap=int(np.count_nonzero((count_map > 1) & ~seed_sites)),
                holes=int(np.count_nonzero(count_map == 0)), asymmetric=_asymmetric(plan))
    return expected, meta


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive atlas, difference union, scale, and alignment solely from pixels."""
    return _recover(input_grid)[0]


def _geometry_options():
    result = []
    for ph in range(3, 6):
        for pw in range(3, 6):
            for bands in range(2, 5):
                for copies in range(3, 6):
                    bank = (bands * (ph + 1) + 1, copies * (pw + 1) + 1)
                    for side in range(2, 5):
                        for gutter in range(1, 4):
                            box = tuple((n - 1) * (side + gutter) + side for n in (ph, pw))
                            for sep in (0, 1):
                                need = [max(bank[k], box[k]) + 2 for k in (0, 1)]
                                need[sep] = bank[sep] + box[sep] + 3
                                if max(need) <= 30:
                                    result.append((ph, pw, bands, copies, side, gutter, sep))
    return tuple(result)


_GEOMETRIES = _geometry_options()


def _sample_master(rng, shape):
    h, w = shape
    rr, cc = np.indices(shape)
    style = int(rng.integers(6))
    if style == 0:
        mask = rng.random(shape) < float(rng.uniform(.25, .70))
    elif style == 1:
        mask = (rr + cc) % 2 == 0
    elif style == 2:
        mask = (rr == 0) | (cc == 0) | (rr == h - 1) | (cc == w - 1)
    elif style == 3:
        mask = (rr == int(rng.integers(h))) | (cc == int(rng.integers(w)))
    elif style == 4:
        mask = rr == cc
        mask |= rng.random(shape) < .15
    else:
        mask = (rr % int(rng.integers(2, 4))) == 0
        if rng.integers(2):
            mask = (cc % int(rng.integers(2, 4))) == 0
    if rng.integers(2):
        mask = ~mask
    if not 2 <= int(mask.sum()) <= mask.size - 2:
        mask = (rr + cc) % 2 == 0
    return mask


def _sample_masks(rng, master, bands, witness):
    area = master.size
    ones, zeros = np.flatnonzero(master), np.flatnonzero(~master)
    masks = []
    shared = int(rng.integers(area))
    for b in range(bands):
        mask = np.zeros(master.shape, dtype=bool)
        mode = 'balanced' if witness == 'balanced' else str(rng.choice(['balanced', 'add', 'remove', 'scatter', 'line']))
        if mode == 'balanced':
            high = min(len(ones), len(zeros), area // 4, 3)
            k = int(rng.integers(1, high + 1))
            mask.flat[rng.choice(ones, k, replace=False)] = True
            mask.flat[rng.choice(zeros, k, replace=False)] = True
        elif mode in ('add', 'remove'):
            cells = zeros if mode == 'add' else ones
            k = int(rng.integers(1, min(len(cells) - 1, area // 2) + 1))
            mask.flat[rng.choice(cells, k, replace=False)] = True
        elif mode == 'line':
            dim = int(rng.integers(2))
            index = int(rng.integers(master.shape[dim]))
            if dim == 0:
                mask[index, :] = True
            else:
                mask[:, index] = True
        else:
            k = int(rng.integers(1, area // 2 + 1))
            mask.flat[rng.choice(area, k, replace=False)] = True
        if witness == 'overlap':
            mask.flat[shared] = True
        if mask.sum() > area // 2:
            cells = np.flatnonzero(mask)
            mask[:] = False
            mask.flat[rng.choice(cells, area // 2, replace=False)] = True
        if witness is None and masks and rng.random() < .12:
            mask = masks[int(rng.integers(len(masks)))].copy()
        masks.append(mask)
    return np.array(masks)


def _candidate_clues(master, masks, bg, colors, axis, interior, witness, boundary):
    plan, counts = _patch(master, masks, bg, colors)
    if not np.any(counts == 0):
        return []
    if any(not 1 <= int(m.sum()) <= master.size // 2 for m in masks):
        return []
    if any(len(np.unique(master ^ m)) != 2 for m in masks):
        return []
    if witness and not _asymmetric(plan):
        return []
    clues = []
    for match, length, seq in _segments(plan, bg, axis):
        cross = match[1 - axis]
        n = plan.shape[1 - axis]
        if interior:
            if not (0 < cross < n - 1 and length < plan.shape[axis]):
                continue
        elif cross not in (0, n - 1) or (boundary is not None and cross != (0 if boundary == 0 else n - 1)):
            continue
        sites = np.zeros(plan.shape, dtype=bool)
        for k in range(length):
            sites[match[0] + (axis == 0) * k, match[1] + (axis == 1) * k] = True
        added = set(map(int, plan[(counts > 0) & ~sites]))
        if added != set(colors):
            continue
        if witness and any(not np.any(m & (counts == 1) & ~sites) for m in masks):
            continue
        if witness == 'overlap' and (not np.any((counts > 1) & ~sites) or match == (0, 0)):
            continue
        clues.append((match, length))
    return clues


def _fallback(rng, shape, bands, bg, colors, axis, interior, witness, boundary):
    """Finite constructive patch: unique XY clue, YX and Y.X contributions.

    Two masks have exclusive cells and a shared pair. Additional bands may repeat
    a mask; overlap is idempotent. Transposition gives vertical clues.
    """
    h, w = shape if axis == 1 else shape[::-1]
    seed_rows = list(range(1, h - 1)) if interior else ([0 if boundary == 0 else h - 1] if boundary is not None else [0, h - 1])
    for r in seed_rows:
        other = [v for v in range(h) if v != r]
        for a in other:
            for b in other:
                if a == b:
                    continue
                for c in range(w - 1):
                    s0, s1 = (r, c), (r, c + 1)
                    u0, u1, o0, o1 = (a, 0), (a, 1), (b, 0), (b, w - 1)
                    master = np.zeros((h, w), dtype=bool)
                    for p in (s0, u1, o1):
                        master[p] = True
                    first = [s0, u0, o0, o1]
                    if witness == 'overlap':
                        first.remove(o0)
                    sets = [first, [s1, u1, o0, o1]]
                    masks = np.zeros((bands, h, w), dtype=bool)
                    for k in range(bands):
                        for p in sets[k % 2]:
                            masks[k][p] = True
                    if axis == 0:
                        master, masks = master.T, masks.transpose(0, 2, 1)
                    choices = _candidate_clues(master, masks, bg, colors, axis, interior, witness, boundary)
                    if choices:
                        match, length = choices[int(rng.integers(len(choices)))]
                        return master, masks, match, length
    raise AssertionError('constructive patch fallback exhausted')


def _sample_atlas(rng, shape, bands, bg, colors, axis, interior, witness, boundary):
    for _ in range(100):
        master = _sample_master(rng, shape)
        masks = _sample_masks(rng, master, bands, witness)
        choices = _candidate_clues(master, masks, bg, colors, axis, interior, witness, boundary)
        if choices:
            match, length = choices[int(rng.integers(len(choices)))]
            return master, masks, match, length
    return _fallback(rng, shape, bands, bg, colors, axis, interior, witness, boundary)


def _place(rng, bank, box, sep):
    other = 1 - sep
    max_gap = min(3, 28 - bank[sep] - box[sep])
    gap = int(rng.integers(1, max_gap + 1))
    needed = [max(bank[k], box[k]) + 2 for k in (0, 1)]
    needed[sep] = bank[sep] + box[sep] + gap + 2
    if rng.random() < .15:
        shape = (30, 30)
    elif rng.random() < .2:
        size = int(rng.integers(max(needed), 31))
        shape = (size, size)
    else:
        shape = tuple(int(rng.integers(n, min(30, n + 4) + 1)) for n in needed)
    atlas, base = [0, 0], [0, 0]
    start = int(rng.integers(1, shape[sep] - needed[sep] + 2))
    if rng.integers(2):
        atlas[sep], base[sep] = start, start + bank[sep] + gap
    else:
        base[sep], atlas[sep] = start, start + box[sep] + gap
    atlas[other] = int(rng.integers(1, shape[other] - bank[other]))
    base[other] = int(rng.integers(1, shape[other] - box[other]))
    return shape, tuple(atlas), tuple(base)


def _make_pair(rng, geometry, axis, interior, witness=None, boundary=None, avoid_bg=None):
    ph, pw, bands, copies, side, gutter, sep = geometry
    palette = list(map(int, rng.choice(10, 4, replace=False)))
    if palette[0] == avoid_bg:
        palette[0], palette[1] = palette[1], palette[0]
    bg, frame, *colors = palette
    master, masks, match, length = _sample_atlas(rng, (ph, pw), bands, bg, colors, axis, interior, witness, boundary)
    bank_shape = (bands * (ph + 1) + 1, copies * (pw + 1) + 1)
    stride = side + gutter
    box = ((ph - 1) * stride + side, (pw - 1) * stride + side)
    shape, atlas, base = _place(rng, bank_shape, box, sep)
    bank = np.full(bank_shape, frame, dtype=np.int8)
    columns = rng.integers(copies, size=bands)
    if witness and len(set(map(int, columns))) == 1:
        columns[1] = (columns[1] + int(rng.integers(1, copies))) % copies
    for b in range(bands):
        for c in range(copies):
            pixels = master ^ masks[b] if c == columns[b] else master
            y, x = b * (ph + 1) + 1, c * (pw + 1) + 1
            bank[y:y + ph, x:x + pw] = np.where(pixels, colors[1], colors[0])
    inp = np.full(shape, bg, dtype=np.int8)
    y, x = atlas
    inp[y:y + bank_shape[0], x:x + bank_shape[1]] = bank
    plan, _ = _patch(master, masks, bg, colors)
    for k in range(length):
        r, c = match[0] + (axis == 0) * k, match[1] + (axis == 1) * k
        y, x = base[0] + r * stride, base[1] + c * stride
        inp[y:y + side, x:x + side] = plan[r, c]
    out = inp.copy()
    for (r, c), color in np.ndenumerate(plan):
        if color != bg:
            y, x = base[0] + r * stride, base[1] + c * stride
            out[y:y + side, x:x + side] = color
    return {'input': inp, 'output': out}


def _count(value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError('example counts must be integers')
    if not lo <= value <= hi:
        raise ValueError('example count out of range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count(num_train, 2, 10), _count(num_test, 1, 4)
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    mode = int(rng.integers(3))
    axes = [1] if mode == 0 else [0] if mode == 1 else [0, 1]
    rng.shuffle(axes)
    # In a mixed puzzle the two explanatory witnesses cover different axes.
    # The completed train list is shuffled after constructing these witnesses.
    train_axes = [axes[k % len(axes)] for k in range(nt)]
    witnesses = ['balanced', 'overlap'] + [None] * (nt - 2)
    first_boundary = int(rng.integers(2))
    seen, previous = set(), None
    result = {'train': [], 'test': []}
    for split, count in (('train', nt), ('test', ne)):
        for k in range(count):
            axis = train_axes[k] if split == 'train' else int(rng.choice(axes))
            witness = witnesses[k] if split == 'train' else None
            options = [g for g in _GEOMETRIES if (witness is None or g[2] == 2)]
            if split == 'train' and k == 1:
                options = [g for g in options if g[:2] != previous['panel_shape'] and g[4] != previous['side']]
            geometry = options[int(rng.integers(len(options)))]
            boundary = (first_boundary if k == 0 else 1 - first_boundary) if witness else None
            pair = _make_pair(rng, geometry, axis, split == 'test', witness, boundary,
                              previous['bg'] if split == 'train' and k == 1 else None)
            # A collision needs no unbounded retries: an injective palette cycle
            # supplies 90 distinct possibilities against at most 13 seen inputs.
            if _key(pair['input']) in seen:
                active = sorted(map(int, np.unique(pair['input'])))
                found = False
                for a in range(10):
                    for b in range(10):
                        if a == b:
                            continue
                        targets = [a, b] + [v for v in range(10) if v not in (a, b)]
                        lookup = np.arange(10, dtype=np.int8)
                        lookup[active] = targets[:len(active)]
                        candidate = {name: lookup[grid] for name, grid in pair.items()}
                        if _key(candidate['input']) not in seen:
                            pair, found = candidate, True
                            break
                    if found:
                        break
                assert found
            seen.add(_key(pair['input']))
            result[split].append(pair)
            if split == 'train' and k == 0:
                previous = _recover(pair['input'])[1]
        rng.shuffle(result[split])
    validate_generated_puzzle(result)
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, metas = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            expected, m = _recover(inp)
            assert inp.shape == out.shape and np.array_equal(out, expected)
            assert np.count_nonzero(out != inp) >= 2 * m['side'] ** 2
            assert len(np.unique(inp)) == 4
            assert 3 <= min(m['panel_shape']) <= max(m['panel_shape']) <= 5
            assert 2 <= m['bands'] <= 4 and 3 <= m['copies'] <= 5
            assert 2 <= m['side'] <= 4 and 1 <= m['gutter'] <= 3
            assert m['holes'] >= 1 and len(m['added_colors']) == 2
            assert all(1 <= n <= np.prod(m['panel_shape']) // 2 for n in m['edits'])
            assert all(len(np.unique(v)) == 2 for v in m['variants'])
            # Output preserves the complete atlas, existing squares and border.
            assert np.all(out[inp != m['bg']] == inp[inp != m['bg']])
            metas[split].append(m)
    train = metas['train']
    assert any(all(m['balanced']) and min(m['exclusive']) > 0 and m['asymmetric'] for m in train)
    assert any(m['overlap'] > 0 and min(m['exclusive']) > 0 and m['match'] != (0, 0)
               and m['asymmetric'] for m in train)
    assert any(len(set(m['columns'])) > 1 for m in train)
    for key in ('panel_shape', 'side', 'bg'):
        assert len({m[key] for m in train}) >= 2
    taught = {m['axis'] for m in train if m['asymmetric'] and min(m['exclusive']) > 0}
    assert {m['axis'] for m in metas['train'] + metas['test']} <= taught
    for m in train:
        assert m['cross'] in (0, m['panel_shape'][1 - m['axis']] - 1)
    for m in metas['test']:
        assert 0 < m['cross'] < m['panel_shape'][1 - m['axis']] - 1
        assert m['length'] < m['panel_shape'][m['axis']]
    return True


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 streamed seeds, legal counts, negative checks and metamorphic checks."""
    records, signatures, structures = [], set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(_signature(p))
        for split in ('train', 'test'):
            for pair in p[split]:
                assert np.array_equal(solve(pair['input']), pair['output'])
                _, m = _recover(pair['input'])
                records.append((split, {k: m[k] for k in ('panel_shape', 'bands', 'copies', 'side',
                               'gutter', 'axis', 'match', 'length', 'bg', 'shape', 'atlas', 'base', 'edits', 'overlap')}))
                roles = {}
                normalized = bytes(roles.setdefault(int(v), len(roles)) for v in pair['input'].flat)
                structures.add((pair['input'].shape, hashlib.sha256(normalized).digest()))
        # Injective recoloring must commute with the input-only transformation.
        pair = p['train'][0]
        lookup = np.array([3, 8, 0, 9, 2, 7, 4, 1, 6, 5], dtype=np.int8)
        assert np.array_equal(solve(lookup[pair['input']]), lookup[pair['output']])
        _, m = _recover(pair['input'])
        changed_in, changed_out = pair['input'].copy(), pair['output'].copy()
        top, left = m['atlas']
        ph, pw = m['panel_shape']
        # Reorder band observations and cyclically relocate all panel copies.
        # The hidden patch and target origin must remain exactly the same.
        for b in range(m['bands']):
            old_y = top + (m['bands'] - 1 - b) * (ph + 1) + 1
            new_y = top + b * (ph + 1) + 1
            for c in range(m['copies']):
                old_x = left + ((c + 1) % m['copies']) * (pw + 1) + 1
                new_x = left + c * (pw + 1) + 1
                pixels = pair['input'][old_y:old_y + ph, old_x:old_x + pw]
                changed_in[new_y:new_y + ph, new_x:new_x + pw] = pixels
                changed_out[new_y:new_y + ph, new_x:new_x + pw] = pixels
        assert np.array_equal(solve(changed_in), changed_out)
        # The omission and overlap counterfactuals below compare patch contents
        # directly. They are author-owned ablations, not an independent solver.
        for pair in p['train']:
            _, m = _recover(pair['input'])
            counts = m['masks'].sum(axis=0)
            if min(m['exclusive']) > 0:
                for band, mask in enumerate(m['masks']):
                    remaining = counts - mask
                    lost = (counts > 0) & (remaining == 0) & ~m['seed_sites']
                    assert np.any(lost), ('missing band has no consequence', band)
            if m['overlap']:
                assert np.any((counts > 1) & ~m['seed_sites'] & (m['plan'] != m['bg']))
            # Copying any full exceptional tile paints unchanged coordinates.
            assert np.any((counts == 0) & (m['variants'][0] != m['bg']))
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('output corruption was accepted')
    assert len(signatures) == 42 and len(structures) >= 200
    for split in ('train', 'test'):
        group = [m for s, m in records if s == split]
        for key, minimum in (('panel_shape', 6), ('bands', 3), ('copies', 3), ('side', 3),
                             ('gutter', 3), ('axis', 2), ('match', 8), ('bg', 9), ('shape', 20)):
            assert len({m[key] for m in group}) >= minimum, (split, key)
        assert any(m['overlap'] for m in group)
        assert any(len(set(m['edits'])) > 1 for m in group)
    assert _signature(generate_puzzle(931, 7, 3)) == _signature(generate_puzzle(931, 7, 3))
    p = generate_puzzle(0, 2, 1)
    pair = p['train'][0]
    _, m = _recover(pair['input'])
    broken = pair['input'].copy()
    ph, pw = m['panel_shape']
    y = m['atlas'][0] + 1
    x = m['atlas'][1] + m['columns'][0] * (pw + 1) + 1
    broken[y:y + ph, x:x + pw] = m['master']
    try:
        solve(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('missing exceptional panel was accepted')
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            generate_puzzle(1000 + 10 * nt + ne, nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts were accepted')
    # Exercise the deterministic construction independently of random successes.
    for h in range(3, 6):
        for w in range(3, 6):
            for axis in (0, 1):
                for interior, witness in ((False, 'balanced'), (False, 'overlap'), (True, None)):
                    _fallback(np.random.default_rng(5), (h, w), 2, 0, [2, 7], axis,
                              interior, witness, 1 if witness == 'overlap' else 0)
    return True
