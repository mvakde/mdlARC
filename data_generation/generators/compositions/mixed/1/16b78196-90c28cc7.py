"""Pack translation-only band tabs, crop, and collapse equal-vector runs.

No parent module, file, seed, or generated example is consulted by solve().
The packing search is exhaustive within the specified full-width profile-chain
family, including both sides, every alignment, subsets, and global groupings.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


HOST_SEEDS = (0, 916054765, 814250577, 674416437, 456913220, 759669385,
              394281347, 452832667, 125108072, 901067334, 592471769,
              561370584, 664788505, 322970563, 623633882, 91144666,
              238737829, 389265740, 27909923, 920035375, 440963037,
              357335769, 489636566, 702744111, 758447619, 40912215,
              357227401, 23175958, 537676671, 795948509, 101215728,
              371473863, 88046977, 857693326, 24576024, 23054769,
              72564829, 703688196, 462440493, 318690035, 797368063,
              431713006)


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _counts(train, test):
    if not _integer(train) or not 2 <= train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(test) or not 1 <= test <= 4 or train <= test:
        raise ValueError("num_test must be in 1..4 and less than num_train")
    return int(train), int(test)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _crop(a, bg):
    cells = np.argwhere(a != bg)
    if not len(cells):
        raise ValueError("empty assembly")
    lo, hi = cells.min(axis=0), cells.max(axis=0) + 1
    return a[lo[0]:hi[0], lo[1]:hi[1]].copy()


def _collapse(a):
    # Both comparisons use the SAME uncompressed rectangle. This is equivalent
    # to either axis order: deleting duplicate vectors cannot hide a difference.
    rs = np.r_[0, 1 + np.flatnonzero(np.any(a[1:] != a[:-1], axis=1))]
    cs = np.r_[0, 1 + np.flatnonzero(np.any(a[:, 1:] != a[:, :-1], axis=0))]
    result = a[np.ix_(rs, cs)].copy()
    rh = tuple(map(int, np.diff(np.r_[rs, a.shape[0]])))
    cw = tuple(map(int, np.diff(np.r_[cs, a.shape[1]])))
    return result, rh, cw


def _components(a, bg):
    todo = set(map(tuple, np.argwhere(a != bg)))
    objects = []
    while todo:
        first = min(todo)
        todo.remove(first)
        color = int(a[first])
        stack, cells = [first], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in todo and a[p] == color:
                    todo.remove(p)
                    stack.append(p)
        objects.append((np.array(sorted(cells), dtype=int), color))
    return objects


def _dilate(mask):
    result = np.zeros_like(mask)
    h, w = mask.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r0, r1 = max(0, -dr), min(h, h-dr)
            c0, c1 = max(0, -dc), min(w, w-dc)
            result[r0+dr:r1+dr, c0+dc:c1+dc] |= mask[r0:r1, c0:c1]
    return result


def _scene(a):
    bg = int(a[0, 0])
    if not a[0, -1] == a[-1, 0] == a[-1, -1] == bg:
        raise ValueError("all four corners must identify the background")
    objects = _components(a, bg)
    spanning = []
    for i, (cells, color) in enumerate(objects):
        if len(np.unique(cells[:, 1])) == a.shape[1]:
            spanning.append((i, False, color))
        if len(np.unique(cells[:, 0])) == a.shape[0]:
            spanning.append((i, True, color))
    if len(spanning) != 1:
        raise ValueError("expected one connected spanning band")
    index, transposed, band = spanning[0]
    canonical = a.T.copy() if transposed else a.copy()
    band_mask = canonical == band
    edges = []
    for c in range(canonical.shape[1]):
        rows = np.flatnonzero(band_mask[:, c])
        if not len(rows) or not np.array_equal(rows, np.arange(rows[0], rows[-1]+1)):
            raise ValueError("band must have interval cross sections")
        edges.append((int(rows[0]), int(rows[-1])+1))
    edges = np.array(edges, dtype=int)
    if (not np.all(edges[1:, 0] < edges[:-1, 1])
            or not np.all(edges[:-1, 0] < edges[1:, 1])):
        raise ValueError("band must be connected")
    pieces = []
    forbidden = _dilate(band_mask)
    for i, (cells, color) in enumerate(objects):
        if i == index:
            continue
        if color == band:
            raise ValueError("the structural color is reserved for the band")
        cells = cells[:, ::-1] if transposed else cells
        if np.any(forbidden[cells[:, 0], cells[:, 1]]):
            raise ValueError("input components must have one-cell clearance")
        mask = np.zeros_like(band_mask)
        mask[cells[:, 0], cells[:, 1]] = True
        forbidden |= _dilate(mask)
        pieces.append((cells, color))
    if not 1 <= len(pieces) <= 8:
        raise ValueError("expected 1..8 detached pieces")
    return canonical, bg, band, transposed, edges, pieces


def _describe(cells, sign):
    x = cells[:, 1] - cells[:, 1].min()
    y = sign*cells[:, 0]
    lower, upper = [], []
    for c in range(int(x.max())+1):
        values = np.sort(y[x == c])
        if not len(values) or not np.array_equal(values, np.arange(values[0], values[-1]+1)):
            raise ValueError("pieces must be full-width, column-convex layers")
        lower.append(int(values[0]))
        upper.append(int(values[-1])+1)
    origin = min(lower)
    return (tuple(v-origin for v in lower), tuple(v-origin for v in upper),
            np.column_stack((y-origin, x)))


def _packing_solutions(input_grid):
    a, bg, band, transposed, edges, pieces = _scene(input_grid)
    h, w = a.shape
    base = np.full_like(a, bg)
    base[a == band] = band
    candidates = []
    # An attachment is not selected by notch size or nearest source position.
    # Enumerate both boundary sides, all widths, and every possible alignment.
    for sign in (-1, 1):
        desc = [_describe(cells, sign) for cells, _ in pieces]
        lookup = {}
        for i, (lower, _, _) in enumerate(desc):
            lookup.setdefault(lower, []).append(i)
        boundary = 1-edges[:, 0] if sign == -1 else edges[:, 1]
        for width in sorted({len(d[0]) for d in desc}):
            for start in range(w-width+1):
                initial = tuple(map(int, boundary[start:start+width]))

                def extend(profile, bits, placed):
                    if bits and len(set(profile)) == 1:
                        occupied = frozenset((int(r), int(c))
                                             for _, cells in placed for r, c in cells)
                        candidates.append((bits, occupied, placed, sign, start, width))
                    offset = min(profile)
                    normalized = tuple(v-offset for v in profile)
                    for i in lookup.get(normalized, ()):
                        if bits & (1 << i):
                            continue
                        _, upper, local = desc[i]
                        cells = np.column_stack((sign*(local[:, 0]+offset), local[:, 1]+start))
                        if np.any(cells[:, 0] < 0) or np.any(cells[:, 0] >= h):
                            continue
                        if np.any(base[cells[:, 0], cells[:, 1]] != bg):
                            continue
                        if np.array_equal(cells.min(axis=0), pieces[i][0].min(axis=0)):
                            continue  # Every detached component actually moves.
                        extend(tuple(offset+v for v in upper), bits | (1 << i),
                               placed+[(i, cells)])

                extend(initial, 0, [])
    by_piece = [[] for _ in pieces]
    for candidate in candidates:
        for i in range(len(pieces)):
            if candidate[0] & (1 << i):
                by_piece[i].append(candidate)
    full = (1 << len(pieces))-1
    solutions = {}

    def cover(bits, occupied, chosen):
        if len(solutions) >= 2:
            return
        if bits == full:
            result = base.copy()
            for candidate in chosen:
                for i, cells in candidate[2]:
                    result[cells[:, 0], cells[:, 1]] = pieces[i][1]
            solutions.setdefault(result.tobytes(), (result, chosen))
            return
        options = [(i, [c for c in by_piece[i]
                         if not c[0] & bits and not c[1] & occupied])
                   for i in range(len(pieces)) if not bits & (1 << i)]
        _, choices = min(options, key=lambda item: len(item[1]))
        for candidate in choices:
            cover(bits | candidate[0], occupied | candidate[1], chosen+[candidate])

    cover(0, frozenset(), [])
    return list(solutions.values()), dict(background=bg, band=band, edges=edges,
                                         pieces=pieces, transposed=transposed,
                                         candidate_count=len(candidates))


def _recover(input_grid):
    a = _grid(input_grid)
    solutions, info = _packing_solutions(a)
    if len(solutions) != 1:
        raise ValueError("expected exactly one distinct colored assembly")
    assembled, tabs = solutions[0]
    if info['transposed']:
        assembled = assembled.T.copy()
    cropped = _crop(assembled, info['background'])
    output, rows, columns = _collapse(cropped)
    info.update(assembled=assembled, cropped=cropped, tabs=tabs,
                row_runs=rows, column_runs=columns)
    return output, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the compressed, completed assembly using only visible input."""
    return _recover(input_grid)[0]


def _profile(rng, width, amplitude):
    mode = int(rng.integers(3))
    if mode == 0:
        p = rng.integers(amplitude+1, size=width)
    elif mode == 1:
        p = np.floor(np.linspace(0, amplitude, width)).astype(int)
        if rng.integers(2):
            p = p[::-1]
    else:
        p = np.clip(np.cumsum(rng.integers(-1, 2, size=width)), -amplitude, amplitude)
        p = p-p.min()
    if p.max() == 0:
        p[int(rng.integers(width))] = 1
    return p-p.min()


def _cuts(rng, width, count):
    cuts = [_profile(rng, width, int(rng.integers(1, 3)))]
    for _ in range(count-1):
        p = _profile(rng, width, int(rng.integers(1, 3)))
        prev = cuts[-1]
        offset = max(int(np.max(prev+1-p)),
                     int(np.max(np.maximum(prev[:-1], prev[1:])+1-
                                np.minimum(p[:-1], p[1:]))))
        cuts.append(p+offset+int(rng.integers(2)))
    cuts.append(np.full(width, int(cuts[-1].max())+int(rng.integers(1, 3))))
    return cuts


def _scatter(rng, base, pieces, bg):
    h, w = base.shape
    for _ in range(18):
        result = base.copy()
        forbidden = _dilate(base != bg)
        forbidden[0, 0] = forbidden[0, -1] = True
        forbidden[-1, 0] = forbidden[-1, -1] = True
        order = sorted(range(len(pieces)), key=lambda i: -len(pieces[i][0])+rng.random()*5)
        for i in order:
            target, color = pieces[i]
            local = target-target.min(axis=0)
            ph, pw = local.max(axis=0)+1
            if ph > h or pw > w:
                break
            available = np.ones((h-ph+1, w-pw+1), dtype=bool)
            for r, c in local:
                available &= ~forbidden[r:r+h-ph+1, c:c+w-pw+1]
            origin = target.min(axis=0)
            available[tuple(origin)] = False
            locations = np.argwhere(available)
            if not len(locations):
                break
            cells = local+locations[int(rng.integers(len(locations)))]
            result[cells[:, 0], cells[:, 1]] = color
            mask = np.zeros_like(forbidden)
            mask[cells[:, 0], cells[:, 1]] = True
            forbidden |= _dilate(mask)
        else:
            return result
    raise ValueError("insufficient scatter clearance")


def _make_pair(rng, kind, transpose, palette):
    bg, band, available_colors = palette
    cross = int(rng.integers(11, 16))
    width = int(rng.integers(3, 6))
    if kind == 'single':
        widths, sides, counts = [width], [int(rng.choice((-1, 1)))], [3]
    elif kind == 'witness':
        widths, sides, counts = [width, width], [-1, 1], [2, 2]
    elif kind == 'extra':
        if rng.integers(2):
            widths, sides, counts = [width], [int(rng.choice((-1, 1)))], [int(rng.integers(2, 4))]
        else:
            widths = [width, width if rng.random() < .6 else int(rng.integers(3, 6))]
            sides = [-1, 1]
            counts = [int(rng.integers(1, 3)), 2]
    else:
        widths = [width, width if rng.random() < .75 else int(rng.integers(3, 6))]
        sides = [int(rng.choice((-1, 1)))]*2
        if rng.random() < .5:
            sides[1] *= -1
        counts = [2, 3] if rng.random() < .8 else [3, 3]
        rng.shuffle(counts)
    starts, reserved = [], {-1: set(), 1: set()}
    for i, (tw, side) in enumerate(zip(widths, sides)):
        choices = [s for s in range(1, cross-tw)
                   if not set(range(s-1, s+tw+1)) & reserved[side]]
        if kind == 'witness' and i:
            choices = [starts[0]]
        if not choices:
            raise ValueError("insufficient tab separation")
        start = int(rng.choice(choices))
        starts.append(start)
        reserved[side].update(range(start, start+tw))
    cuts = [_cuts(rng, tw, n) for tw, n in zip(widths, counts)]
    extents = {side: max([int(p[-1][0]) for s, p in zip(sides, cuts) if s == side]+[0])
               for side in (-1, 1)}
    core = int(rng.integers(2, 4))
    center = extents[-1]
    logical_h = extents[-1]+core+extents[1]
    if logical_h > 24:
        raise ValueError("assembly too deep")
    base = np.full((logical_h, cross), bg, dtype=np.int8)
    base[center:center+core, :] = band
    label = base.copy()
    destinations = []
    colors = list(map(int, rng.choice(available_colors, size=sum(counts), replace=True)))
    # Same cap color makes complete top and bottom row vectors recur, while
    # independent interlocks still determine which physical cap goes where.
    if kind == 'witness':
        colors[counts[0]-1] = colors[-1]
    color_index = 0
    for tw, start, side, interfaces in zip(widths, starts, sides, cuts):
        for c in range(tw):
            for u in range(int(interfaces[0][c])):
                r = center-1-u if side == -1 else center+core+u
                base[r, start+c] = label[r, start+c] = band
        for lo, hi in zip(interfaces, interfaces[1:]):
            cells = np.array([(center-1-u if side == -1 else center+core+u, start+c)
                              for c in range(tw) for u in range(int(lo[c]), int(hi[c]))])
            color = colors[color_index]
            color_index += 1
            label[cells[:, 0], cells[:, 1]] = color
            destinations.append((cells, color))
    # Replicate the WHOLE assembly, before extracting and scattering its pieces.
    # Thus expansion is global; individual objects are never independently scaled.
    row_scale = np.ones(logical_h, dtype=int)
    budget = min(24, logical_h+int(rng.integers(2, 7)))
    for _ in range(budget-logical_h):
        row_scale[int(rng.integers(logical_h))] += 1
    col_scale = rng.integers(1, 3, size=cross)
    while col_scale.sum() > 28:
        index = int(rng.choice(np.flatnonzero(col_scale > 1)))
        col_scale[index] -= 1
    expanded_base = np.repeat(np.repeat(base, row_scale, axis=0), col_scale, axis=1)
    expanded_label = np.repeat(np.repeat(label, row_scale, axis=0), col_scale, axis=1)
    ch, cw = expanded_base.shape
    height = int(rng.integers(max(20, ch+2), 31))
    offset = int(rng.integers(1, height-ch))
    canvas = np.full((height, cw), bg, dtype=np.int8)
    canvas[offset:offset+ch] = expanded_base
    assembled = np.full_like(canvas, bg)
    assembled[offset:offset+ch] = expanded_label
    pieces = []
    for cells, color in destinations:
        mask = np.zeros(base.shape, dtype=bool)
        mask[cells[:, 0], cells[:, 1]] = True
        expanded_mask = np.repeat(np.repeat(mask, row_scale, axis=0), col_scale, axis=1)
        pieces.append((np.argwhere(expanded_mask)+np.array((offset, 0)), color))
    inp = _scatter(rng, canvas, pieces, bg)
    if rng.integers(2):
        inp, assembled = inp[:, ::-1], assembled[:, ::-1]
    if rng.integers(2):
        inp, assembled = inp[::-1], assembled[::-1]
    if transpose:
        inp, assembled = inp.T, assembled.T
    expected = _collapse(_crop(assembled, bg))[0]
    predicted, info = _recover(inp)
    if (not np.array_equal(info['assembled'], assembled)
            or not np.array_equal(predicted, expected)):
        raise ValueError("construction disagrees with unique assembly")
    if kind == 'witness' and not all(_recurrences(expected)):
        raise ValueError("recurrence witness was hidden by its interlock")
    pair = {'input': inp.copy(), 'output': expected.copy()}
    _validate_pair(pair)
    return pair


def _recurrences(a):
    return (len({tuple(row) for row in a}) < a.shape[0],
            len({tuple(col) for col in a.T}) < a.shape[1])


def _validate_pair(pair):
    if not isinstance(pair, dict) or set(pair) != {'input', 'output'}:
        raise ValueError("pair must have exactly input/output")
    a, b = _grid(pair['input']), _grid(pair['output'])
    predicted, info = _recover(a)
    assert np.array_equal(predicted, b), 'incorrect output'
    assembled, crop = info['assembled'], info['cropped']
    bg, band = info['background'], info['band']
    assert np.array_equal(a == band, assembled == band), 'band changed before collapse'
    assert np.array_equal(np.bincount(a.ravel(), minlength=10),
                          np.bincount(assembled.ravel(), minlength=10)), 'packing recolored/lost cells'
    canonical = assembled.T if info['transposed'] else assembled
    seen = set()
    for tab in info['tabs']:
        all_target = np.concatenate([cells for _, cells in tab[2]], axis=0)
        side, start, width = tab[3:]
        window = info['edges'][start:start+width]
        if side == 1:
            lo, hi = int(window[:, 1].min()), int(all_target[:, 0].max())+1
        else:
            lo, hi = int(all_target[:, 0].min()), int(window[:, 0].max())
        assert np.all(canonical[lo:hi, start:start+width] != bg), 'tab has a gap'
        for i, target in tab[2]:
            assert i not in seen
            seen.add(i)
            source, color = info['pieces'][i]
            shifted = source+target.min(axis=0)-source.min(axis=0)
            assert set(map(tuple, shifted)) == set(map(tuple, target)), 'nontranslation'
            assert np.all(canonical[target[:, 0], target[:, 1]] == color)
    assert seen == set(range(len(info['pieces'])))
    assert np.array_equal(np.repeat(np.repeat(b, info['row_runs'], axis=0),
                                    info['column_runs'], axis=1), crop)
    assert b.shape[0] < crop.shape[0] and b.shape[1] < crop.shape[1], 'both axes must matter'
    assert not np.array_equal(_collapse(_crop(a, bg))[0], b), 'packing omission did not matter'
    assert set(np.unique(crop)) == set(np.unique(b)), 'collapse lost a logical color'
    assert not np.any(np.all(b[1:] == b[:-1], axis=1))
    assert not np.any(np.all(b[:, 1:] == b[:, :-1], axis=0))
    return info


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {'train', 'test'}:
        raise ValueError("puzzle must have exactly train/test")
    if not all(isinstance(puzzle[s], list) for s in ('train', 'test')):
        raise ValueError("splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, train_info = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info = _validate_pair(pair)
            key = _key(np.asarray(pair['input']))
            assert key not in seen, 'duplicate inputs'
            seen.add(key)
            if split == 'train':
                train_info.append(info)
    assert any(all(_recurrences(np.asarray(p['output']))) for p in puzzle['train']), \
        'training must distinguish consecutive-only collapse in both axes'
    assert {i['transposed'] for i in train_info} == {False, True}, \
        'training must demonstrate both packing directions'


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    train, test = _counts(num_train, num_test)
    if not _integer(seed):
        raise TypeError("seed must be an integer")
    rng = np.random.default_rng(seed)
    # Stable versus local palettes are sampling preferences, not hidden rules.
    stable = bool(rng.integers(2))
    shared = tuple(map(int, rng.permutation(10)))
    orientation = bool(rng.integers(2))
    training = [('single', orientation), ('witness', not orientation)]
    training += [('extra', bool(rng.integers(2))) for _ in range(train-2)]
    rng.shuffle(training)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, specs in (('train', training),
                         ('test', [('challenge', bool(rng.integers(2))) for _ in range(test)])):
        for kind, transpose in specs:
            colors = shared if stable else tuple(map(int, rng.permutation(10)))
            palette = (colors[0], colors[1], colors[2:2+int(rng.integers(3, 9))])
            for _ in range(160):
                try:
                    pair = _make_pair(rng, kind, transpose, palette)
                except (ValueError, AssertionError):
                    continue
                signature = _key(pair['input'])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded construction could not find a unique packing")
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(puzzle):
    return tuple((split, tuple((_key(p['input']), _key(p['output'])) for p in puzzle[split]))
                 for split in ('train', 'test'))


def _normalize(a):
    colors = {}
    return a.shape, bytes(colors.setdefault(int(c), len(colors)) for c in a.flat)


def self_test():
    """Stream 42 seeds; retain feature sets and hashes, never a corpus or traces."""
    # Explicit discriminator: adjacent duplicates disappear; A,B,A survives.
    latent = np.array([[2, 4, 2], [7, 7, 7], [2, 4, 2]], dtype=np.int8)
    assert _recurrences(latent) == (True, True)
    enlarged = np.repeat(np.repeat(latent, [2, 1, 3], axis=0), [1, 3, 2], axis=1)
    assert np.array_equal(_collapse(enlarged)[0], latent)
    reverse_order = _collapse(enlarged.T)[0].T
    assert np.array_equal(reverse_order, latent)
    assert np.unique(latent, axis=0).shape[0] < latent.shape[0]
    assert np.unique(latent, axis=1).shape[1] < latent.shape[1]
    # Distinct colored assemblies, rather than object permutations or compressed
    # answers, determine uniqueness. Never silently pick the first exact cover.
    fixture = np.zeros((18, 18), dtype=np.int8)
    fixture[6:9, :] = 8
    fixture[9, [4, 11]] = 8
    cap = np.array([(r, c) for c, lo in enumerate((0, 1, 0)) for r in range(lo, 3)])
    fixture[cap[:, 0]+1, cap[:, 1]+1] = 2
    assert len(_packing_solutions(fixture)[0]) == 2
    try:
        solve(fixture)
    except ValueError:
        pass
    else:
        raise AssertionError('ambiguous packing accepted')
    fixture[cap[:, 0]+13, cap[:, 1]+8] = 3
    assert len(_packing_solutions(fixture)[0]) == 2
    fixture[fixture == 3] = 2
    assert len(_packing_solutions(fixture)[0]) == 1
    shapes, output_shapes, normalized, fingerprints = set(), set(), set(), set()
    tab_counts, piece_counts, widths, palettes, run_profiles, roles = set(), set(), set(), set(), set(), set()
    layouts, recurrences = set(), [0, 0]
    pair_count = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signature = _signature(puzzle)
        fingerprints.add(hashlib.sha256(repr(signature).encode()).hexdigest())
        if seed == 0:
            assert signature == _signature(generate_puzzle(seed))
            pair = puzzle['train'][0]
            original_input = pair['input'].copy()
            permutation = np.array([7, 2, 9, 0, 5, 8, 3, 1, 4, 6], dtype=np.int8)
            assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
            for turns in (1, 2, 3):
                assert np.array_equal(solve(np.rot90(pair['input'], turns)),
                                      np.rot90(pair['output'], turns))
            assert np.array_equal(pair['input'], original_input), 'solver mutated input'
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _validate_pair(pair)
                pair_count += 1
                shapes.add(pair['input'].shape)
                output_shapes.add(pair['output'].shape)
                normalized.add(hashlib.sha256(repr(_normalize(pair['input'])).encode()).hexdigest())
                tab_counts.add(len(info['tabs']))
                piece_counts.add(len(info['pieces']))
                widths.update(t[5] for t in info['tabs'])
                palettes.add(len(np.unique(pair['output'])))
                run_profiles.add((info['row_runs'], info['column_runs']))
                roles.add((info['background'], info['band']))
                signs = tuple(sorted(t[3] for t in info['tabs']))
                layouts.add(signs)
                for axis, flag in enumerate(_recurrences(pair['output'])):
                    recurrences[axis] += int(flag)
    assert len(fingerprints) == 42
    assert len(shapes) >= 35 and len(output_shapes) >= 15
    assert len(normalized) >= int(.95*pair_count)
    assert tab_counts >= {1, 2} and piece_counts >= {2, 3, 4, 5, 6}
    assert len(widths) >= 5 and len(run_profiles) >= 100 and len(roles) >= 20
    assert len(palettes) >= 4 and len(layouts) >= 4
    assert min(recurrences) >= 42
    # Count endpoints and malformed requests, including non-integer counts.
    for train, test in ((2, 1), (10, 1), (10, 4), (5, 4)):
        validate_generated_puzzle(generate_puzzle(783+train+test, train, test))
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    original = generate_puzzle(73, 2, 1)
    broken = {s: [{'input': p['input'].copy(), 'output': p['output'].copy()}
                  for p in original[s]] for s in original}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong output accepted")
    return dict(seeds=42, pairs=pair_count, input_shapes=len(shapes),
                output_shapes=len(output_shapes), normalized_inputs=len(normalized),
                piece_counts=sorted(piece_counts), tab_counts=sorted(tab_counts),
                run_profiles=len(run_profiles), color_roles=len(roles),
                separated_row_recurrences=recurrences[0],
                separated_column_recurrences=recurrences[1])


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(",", ":")))
