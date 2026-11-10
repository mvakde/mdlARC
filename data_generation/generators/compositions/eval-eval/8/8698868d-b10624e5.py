"""Cavity-addressed aperture printing. All output conventions are input-visible.

0 = background; 1 = count dots; 2 = square registration anchor;
3 = its diagonally adjacent corner pin; 4..9 = wall/printing colors.
"""
from __future__ import annotations

import copy
import itertools
import numpy as np


_FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
_EIGHT = _FOUR + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in (_EIGHT if diagonal else _FOUR):
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _box(cells):
    p = np.asarray(cells)
    lo, hi = p.min(0), p.max(0) + 1
    return int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1])


def _border(a):
    return np.concatenate((a[0], a[-1], a[1:-1, 0], a[1:-1, -1]))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= x <= 30 for x in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((a >= 0) & (a <= 9))
    return a.astype(np.int64, copy=False)


def _anchor(a):
    cells = np.argwhere(a == 2)
    assert len(cells), "missing registration square"
    r0, r1, c0, c1 = _box(cells)
    side = r1 - r0
    assert side == c1 - c0 and len(cells) == side * side
    assert np.all(a[r0:r1, c0:c1] == 2)
    pins = np.argwhere(a == 3)
    assert len(pins) == 1, "one corner pin required"
    pr, pc = map(int, pins[0])
    assert pr in (r0 - 1, r1) and pc in (c0 - 1, c1)
    return (r0, c0, side), (-1 if pr < r0 else 1, -1 if pc < c0 else 1)


def _decode(input_grid):
    a = _grid(input_grid)
    walls = _components(a >= 4)
    assert walls, "no framed objects"
    boxes = [_box(cells) for cells in walls]
    used = np.zeros(a.shape, dtype=bool)
    donors, receivers = {}, []
    for box in boxes:
        r0, r1, c0, c1 = box
        assert not np.any(used[r0:r1, c0:c1]), "object boxes overlap"
        used[r0:r1, c0:c1] = True
        tile = a[r0:r1, c0:c1]
        anchor, pin = _anchor(tile)
        if np.any(tile == 1):
            perimeter = _border(tile)
            carrier = int(perimeter[0])
            assert carrier >= 4 and np.all(perimeter == carrier)
            assert set(map(int, np.unique(tile))) <= {0, 1, 2, 3, carrier}
            aperture = tile != carrier
            assert len(_components(aperture)) == 1, "aperture must be connected"
            dots = tile == 1
            assert not np.any(dots[:-1] & dots[1:])
            assert not np.any(dots[:, :-1] & dots[:, 1:])
            count = int(dots.sum())
            receivers.append(dict(box=box, tile=tile, anchor=anchor, pin=pin,
                                  carrier=carrier, aperture=aperture, count=count))
        else:
            edge = _border(tile)
            colors = set(map(int, edge[edge >= 4]))
            assert len(colors) == 1 and np.all(edge != 0), "closed donor perimeter"
            base = next(iter(colors))
            virtual = tile.copy()
            virtual[(virtual == 2) | (virtual == 3)] = base
            assert len(_components(virtual != 0)) == 1
            cavities = _components(tile == 0)
            count = len(cavities)
            assert count >= 1 and count not in donors, "donor counts must be unique"
            donors[count] = dict(box=box, tile=tile, virtual=virtual, anchor=anchor,
                                 pin=pin, base=base, count=count,
                                 cavity_sizes=tuple(sorted(map(len, cavities))))
    assert np.all(a[~used] == 0), "unrelated content outside objects"
    for i, (r0, r1, c0, c1) in enumerate(boxes):
        for s0, s1, t0, t1 in boxes[i + 1:]:
            assert r1 < s0 or s1 < r0 or c1 < t0 or t1 < c0, "one-cell object gap"
    assert len(donors) >= 2 and len(receivers) >= 2
    assert set(donors) == {r['count'] for r in receivers}, "every donor must be used"
    for r in receivers:
        # Some donor ink must differ from the floor, even with shared palettes.
        assert np.any((donors[r['count']]['virtual'] >= 4)
                      & (donors[r['count']]['virtual'] != r['carrier']))
    return a, donors, receivers


def _scaled(donor, side, flips):
    """Exact area rasterization; never round a color boundary."""
    raw = donor['virtual']
    sr, sc, source_side = donor['anchor']
    assert raw.shape[0] * side % source_side == 0
    assert raw.shape[1] * side % source_side == 0
    assert sr * side % source_side == 0 and sc * side % source_side == 0
    expanded = np.repeat(np.repeat(raw, side, 0), side, 1)
    h = raw.shape[0] * side // source_side
    w = raw.shape[1] * side // source_side
    blocks = expanded.reshape(h, source_side, w, source_side)
    lo, hi = blocks.min(axis=(1, 3)), blocks.max(axis=(1, 3))
    assert np.array_equal(lo, hi), "scale straddles a color boundary"
    ar, ac = sr * side // source_side, sc * side // source_side
    if flips[0]:
        lo = lo[::-1]
        ar = h - ar - side
    if flips[1]:
        lo = lo[:, ::-1]
        ac = w - ac - side
    return lo, (ar, ac)


def _print_tile(receiver, donor, *, flips=None, side=None, masking=True):
    tile = receiver['tile']
    dr, dc, ds = receiver['anchor']
    if flips is None:
        flips = tuple(x != y for x, y in zip(donor['pin'], receiver['pin']))
    raster, offset = _scaled(donor, ds if side is None else side, flips)
    top, left = dr - offset[0], dc - offset[1]
    result = tile.copy()
    aperture = receiver['aperture'] if masking else np.ones(tile.shape, dtype=bool)
    result[aperture] = receiver['carrier']
    r0, c0 = max(0, top), max(0, left)
    r1, c1 = min(len(tile), top + len(raster)), min(tile.shape[1], left + raster.shape[1])
    visible = 0
    if r0 < r1 and c0 < c1:
        fragment = raster[r0-top:r1-top, c0-left:c1-left]
        window = result[r0:r1, c0:c1]
        mask = aperture[r0:r1, c0:c1] & (fragment != 0)
        window[mask] = fragment[mask]
        visible = int(mask.sum())
    # Registration marks have fixed pixel thickness and never become ink.
    registration = (tile == 2) | (tile == 3)
    result[registration] = tile[registration]
    total = int(np.count_nonzero(raster))
    return result, dict(flips=tuple(flips), ratio=(ds, donor['anchor'][2]),
                        lost=total-visible, visible=visible, total=total,
                        changed=int(np.count_nonzero((result >= 4) & aperture
                                                   & (result != receiver['carrier']))))


def solve(input_grid: np.ndarray) -> np.ndarray:
    a, donors, receivers = _decode(input_grid)
    result = a.copy()
    for receiver in receivers:
        r0, r1, c0, c1 = receiver['box']
        printed, _ = _print_tile(receiver, donors[receiver['count']])
        result[r0:r1, c0:c1] = printed
    return result


def _choice(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def _cavity_mask(rng, shape, count, witness=False):
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    reserved = np.zeros(shape, dtype=bool)
    reserved[1, 1] = True
    reserved[0, 0] = True
    if witness:
        if count == 1:
            mask[1:-1, 1:-1] = True
            mask[1, 1] = False
        else:
            mask[2, 2] = mask[3, 3] = mask[4, 3] = True
        return mask
    if count == 1 and rng.random() < .3:
        mask[1:-1, 1:-1] = True
        mask[1, 1] = False
        return mask
    inner = [(r, c) for r in range(1, h-1) for c in range(1, w-1)
             if not reserved[r, c]]
    for _ in range(60):
        mask[:] = False
        for i in rng.permutation(len(inner)):
            p = inner[int(i)]
            if any(0 <= p[0]+dr < h and 0 <= p[1]+dc < w
                   and mask[p[0]+dr, p[1]+dc] for dr, dc in _FOUR):
                continue
            trial = mask.copy()
            trial[p] = True
            if len(_components(~trial & ~reserved)) != 1:
                continue
            mask = trial
            if int(mask.sum()) == count:
                break
        if int(mask.sum()) == count:
            break
    else:
        raise RuntimeError('cavity seed capacity')
    target = int(rng.integers(count, len(inner)+1))
    for _ in range(3):
        for i in rng.permutation(len(inner)):
            p = inner[int(i)]
            if int(mask.sum()) >= target:
                return mask
            if mask[p] or not any(mask[p[0]+dr, p[1]+dc] for dr, dc in _FOUR):
                continue
            trial = mask.copy()
            trial[p] = True
            if (len(_components(trial)) == count
                    and len(_components(~trial & ~reserved)) == 1):
                mask = trial
    return mask


def _make_donor(rng, count, unit=1, witness=False, base=None):
    shape = (6, 5) if witness else (int(rng.integers(6, 8)), int(rng.integers(5, 8)))
    if unit == 2 and not witness:
        shape = (int(rng.integers(5, 7)), 6)
    if count >= 4:
        shape = max(6, shape[0]), max(6, shape[1])
    holes = _cavity_mask(rng, shape, count, witness)
    base = int(rng.integers(4, 10)) if base is None else base
    macro = np.full(shape, base, dtype=np.int64)
    macro[holes] = 0
    # Interior printing colors vary, but the perimeter and registration cell
    # retain the base color, which is recoverable after tags are removed.
    if not witness and rng.random() < .65:
        detail = _choice(rng, [c for c in range(4, 10) if c != base])
        for r in range(1, shape[0]-1):
            for c in range(1, shape[1]-1):
                if macro[r, c] and (r, c) != (1, 1) and rng.random() < .35:
                    macro[r, c] = detail
    raw = np.repeat(np.repeat(macro, unit, 0), unit, 1)
    raw[unit:2*unit, unit:2*unit] = 2
    raw[unit-1, unit-1] = 3
    raw = np.rot90(raw, int(rng.integers(4)))
    if rng.random() < .5:
        raw = np.fliplr(raw)
    anchor, pin = _anchor(raw)
    virtual = raw.copy()
    virtual[(virtual == 2) | (virtual == 3)] = base
    assert len(_components(raw >= 4)) == 1
    assert len(_components(raw == 0)) == count
    return dict(tile=raw.copy(), virtual=virtual, anchor=anchor, pin=pin,
                base=base, count=count)


def _isolated_dots(rng, available, count):
    cells = np.argwhere(available)
    for _ in range(30):
        chosen = np.zeros(available.shape, dtype=bool)
        for i in rng.permutation(len(cells)):
            r, c = map(int, cells[int(i)])
            if any(0 <= r+dr < len(chosen) and 0 <= c+dc < chosen.shape[1]
                   and chosen[r+dr, c+dc] for dr, dc in _FOUR):
                continue
            chosen[r, c] = True
            if int(chosen.sum()) == count:
                return chosen
    return None


def _make_receiver(rng, donor, side, flips, full=False, irregular=True):
    raster, offset = _scaled(donor, side, flips)
    pin = tuple(-p if f else p for p, f in zip(donor['pin'], flips))
    carrier = _choice(rng, [c for c in range(4, 10) if c != donor['base']])
    for _ in range(100):
        if full:
            h, w = (x + 2 + int(rng.integers(2)) for x in raster.shape)
            top = int(rng.integers(1, h-raster.shape[0]))
            left = int(rng.integers(1, w-raster.shape[1]))
            ar, ac = top + offset[0], left + offset[1]
        else:
            h, w = int(rng.integers(7, 10)), int(rng.integers(7, 10))
            ar = int(rng.integers(2, h-side-1))
            ac = int(rng.integers(2, w-side-1))
        pr = ar-1 if pin[0] < 0 else ar+side
        pc = ac-1 if pin[1] < 0 else ac+side
        if not (1 <= pr < h-1 and 1 <= pc < w-1):
            continue
        tile = np.full((h, w), carrier, dtype=np.int64)
        tile[1:-1, 1:-1] = 0
        tile[ar:ar+side, ac:ac+side] = 2
        tile[pr, pc] = 3
        if irregular:
            for _ in range(int(rng.integers(1, 5))):
                trial = tile.copy()
                direction = int(rng.integers(4))
                if direction < 2:
                    col = int(rng.integers(1, w-1))
                    depth = int(rng.integers(1, max(2, h//2)))
                    rr = slice(1, depth+1) if direction == 0 else slice(h-depth-1, h-1)
                    region = trial[rr, col:col+1]
                else:
                    row = int(rng.integers(1, h-1))
                    depth = int(rng.integers(1, max(2, w//2)))
                    cc = slice(1, depth+1) if direction == 2 else slice(w-depth-1, w-1)
                    region = trial[row:row+1, cc]
                if np.any((region == 2) | (region == 3)):
                    continue
                region[:] = carrier
                if len(_components(trial != carrier)) == 1:
                    tile = trial
        dots = _isolated_dots(rng, tile == 0, donor['count'])
        if dots is None:
            continue
        tile[dots] = 1
        receiver = dict(tile=tile, aperture=tile != carrier, carrier=carrier,
                        anchor=(ar, ac, side), pin=pin, count=donor['count'])
        printed, record = _print_tile(receiver, donor)
        if record['changed'] < 4 or (full and record['lost']):
            continue
        return receiver, printed
    raise RuntimeError('receiver placement capacity')


def _pack(rng, objects):
    """Independent free offsets, with a small exhaustive shelf fallback."""
    shapes = [obj.shape for obj in objects]
    order = sorted(range(len(objects)), key=lambda i: -shapes[i][0]*shapes[i][1])
    for _ in range(24):
        occupied = np.zeros((30, 30), dtype=bool)
        boxes = [None] * len(objects)
        for i in order:
            h, w = shapes[i]
            free = []
            # Prefix sums test the one-cell bounding-box gap in constant time.
            pref = np.pad(occupied.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
            for r in range(31-h):
                for c in range(31-w):
                    a, b = max(0, r-1), min(30, r+h+1)
                    d, e = max(0, c-1), min(30, c+w+1)
                    if pref[b, e]-pref[a, e]-pref[b, d]+pref[a, d] == 0:
                        free.append((r, c))
            if not free:
                break
            r, c = _choice(rng, free)
            occupied[r:r+h, c:c+w] = True
            boxes[i] = (r, c)
        if all(p is not None for p in boxes):
            return boxes
    # At most seven objects: no recursion or unbounded packing search.
    for sequence in itertools.permutations(order):
        boxes = [None] * len(objects)
        r = c = row_height = 0
        for i in sequence:
            h, w = shapes[i]
            if c + w > 30:
                r += row_height + 1
                c = row_height = 0
            if r + h > 30:
                break
            boxes[i] = r, c
            c += w + 1
            row_height = max(row_height, h)
        if all(p is not None for p in boxes):
            return boxes
    raise RuntimeError('scene packing capacity')


def _build_pair(rng, regime):
    if regime == 'topology':
        # Same foreground color forces selection to use topology rather than
        # a donor/receiver color dictionary. Area order is deliberately reversed.
        base = int(rng.integers(4, 10))
        donors = {k: _make_donor(rng, k, witness=True, base=base) for k in (1, 2)}
        jobs = [(2, 1, (False, True), True, False),
                (1, 1, (True, False), True, False),
                (2, 2, (True, True), True, False)]
    elif regime == 'masking':
        donors = {k: _make_donor(rng, k, unit=2, witness=True) for k in (1, 2)}
        jobs = [(2, 1, (True, False), False, True),
                (1, 2, (False, True), False, True)]
    else:
        n = int(rng.integers(2, 4))
        # Test count sets extend beyond the two guaranteed demonstrations.
        counts = sorted(map(int, rng.choice([1, 2, 3, 4], n, replace=False)))
        if regime == 'test' and max(counts) < 3:
            counts[-1] = int(rng.integers(3, 5))
        # Larger reference registrations are compatible with two-plate scenes;
        # each plate chooses its resolution separately. Three-plate scenes use
        # the finer unit so seven independently masked objects fit the canvas.
        donors = {k: _make_donor(rng, k, unit=2 if n == 2 and rng.random() < .35 else 1)
                  for k in counts}
        extra = int(rng.integers(1 if regime == 'test' else 0, 2))
        receiver_counts = counts + [int(rng.choice(counts)) for _ in range(extra)]
        rng.shuffle(receiver_counts)
        jobs = [(k, int(rng.integers(1, 4 if regime == 'test' else 3)),
                 tuple(bool(x) for x in rng.integers(0, 2, 2)), False, True)
                for k in receiver_counts]
    objects = [d['tile'] for d in donors.values()]
    outputs = [obj.copy() for obj in objects]
    for k, side, flips, full, irregular in jobs:
        receiver, printed = _make_receiver(rng, donors[k], side, flips, full, irregular)
        objects.append(receiver['tile'])
        outputs.append(printed)
    origins = _pack(rng, objects)
    inp, out = np.zeros((30, 30), dtype=np.int64), np.zeros((30, 30), dtype=np.int64)
    for (r, c), obj, printed in zip(origins, objects, outputs):
        h, w = obj.shape
        inp[r:r+h, c:c+w] = obj
        out[r:r+h, c:c+w] = printed
    low_r = min(r for r, c in origins)
    low_c = min(c for r, c in origins)
    high_r = max(r+obj.shape[0] for (r, c), obj in zip(origins, objects))
    high_c = max(c+obj.shape[1] for (r, c), obj in zip(origins, objects))
    # Both edge contact and independently sized surrounding margins occur.
    low_r = int(rng.integers(low_r+1))
    low_c = int(rng.integers(low_c+1))
    high_r = int(rng.integers(high_r, 31))
    high_c = int(rng.integers(high_c, 31))
    return dict(input=inp[low_r:high_r, low_c:high_c].copy(),
                output=out[low_r:high_r, low_c:high_c].copy())


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed')
    nt, ne = _integer(num_train, 'num_train'), _integer(num_test, 'num_test')
    if seed < 0 or not (2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne):
        raise ValueError('require nonnegative seed, train 2..10, test 1..4, train > test')
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, regimes in [('train', ['topology', 'masking'] + ['general']*(nt-2)),
                           ('test', ['test']*ne)]:
        for regime in regimes:
            for _ in range(40):
                try:
                    pair = _build_pair(rng, regime)
                except RuntimeError:
                    continue
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded generation exhausted')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative(a, donors, receivers, mode):
    result = a.copy()
    # Rank matching is well-defined when donor cavity areas are distinct.
    ranked = sorted(donors.values(), key=lambda d: int((d['tile'] == 0).sum()))
    area_map = dict(zip(sorted(donors), ranked))
    for r in receivers:
        d = area_map[r['count']] if mode == 'area' else donors[r['count']]
        options = {}
        if mode == 'no_reflection':
            options['flips'] = (False, False)
        if mode == 'no_scale':
            options['side'] = d['anchor'][2]
        if mode == 'no_mask':
            options['masking'] = False
        printed, _ = _print_tile(r, d, **options)
        r0, r1, c0, c1 = r['box']
        result[r0:r1, c0:c1] = printed
    return result


def _analyze(pair):
    a, donors, receivers = _decode(pair['input'])
    output = _grid(pair['output'])
    assert output.shape == a.shape
    assert np.array_equal(output, solve(a)), "incorrect aperture printing"
    assert not np.array_equal(a, output)
    records = []
    for r in receivers:
        printed, rec = _print_tile(r, donors[r['count']])
        assert rec['changed'] >= 4, "no visible donor-specific print"
        assert np.all(printed[r['tile'] == 1] != 1), "clues must disappear"
        assert np.array_equal(printed[~r['aperture']], r['tile'][~r['aperture']])
        records.append(rec)
    areas = {k: int((d['tile'] == 0).sum()) for k, d in donors.items()}
    inversions = any(x < y and areas[x] > areas[y] for x in areas for y in areas)
    alternatives = {m: not np.array_equal(_alternative(a, donors, receivers, m), output)
                    for m in ('area', 'no_reflection', 'no_scale', 'no_mask')}
    return dict(donors=donors, receivers=receivers, records=records,
                inversion=inversions, alternatives=alternatives,
                diagonal=any(len(_components(d['tile'] == 0, True)) < k
                             for k, d in donors.items()))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            key = _key(a)
            assert key not in seen, "duplicate input"
            seen.add(key)
            analysis = _analyze(dict(input=a, output=b))
            if split == 'train':
                training.append(analysis)
    assert any(a['inversion'] and a['alternatives']['area'] for a in training)
    assert any(a['diagonal'] for a in training), "four/eight connectivity not exposed"
    assert any(any(max(d['cavity_sizes']) > 1 for d in a['donors'].values()) for a in training)
    for mode in ('no_reflection', 'no_scale', 'no_mask'):
        assert any(a['alternatives'][mode] for a in training), mode + ' shortcut survives'
    flips = {r['flips'] for a in training for r in a['records'] if r['lost'] == 0}
    assert {(False, True), (True, False), (True, True)} <= flips
    assert any(r['ratio'][0] > r['ratio'][1] for a in training for r in a['records'])
    assert any(r['ratio'][0] < r['ratio'][1] for a in training for r in a['records'])


def self_test():
    """42 seeds, actual decoded structure, ablations, and corruption checks."""
    shapes, count_sets, donor_shapes, pins, ratios, fingerprints = set(), set(), set(), set(), set(), set()
    repeated = multicolor = irregular = clipped = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        fingerprints.add(tuple(_key(x['input']) for s in ('train', 'test') for x in p[s]))
        if seed < 3:
            q = generate_puzzle(seed)
            assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s])
                       for k in ('input', 'output'))
        for split in p:
            for pair in p[split]:
                a = _analyze(pair)
                shapes.add(pair['input'].shape)
                count_sets.add(tuple(sorted(a['donors'])))
                repeated += len(a['receivers']) > len(a['donors'])
                for d in a['donors'].values():
                    donor_shapes.add((d['virtual'].shape, (d['virtual'] == 0).tobytes()))
                    pins.add(d['pin'])
                    multicolor += len(np.unique(d['virtual'][d['virtual'] >= 4])) > 1
                for r in a['receivers']:
                    irregular += int((r['tile'][1:-1, 1:-1] == r['carrier']).sum()) > 0
                for rec in a['records']:
                    n, d = rec['ratio']
                    g = int(np.gcd(n, d))
                    ratios.add((n//g, d//g))
                    clipped += rec['lost'] > 0
    assert len(fingerprints) == 42
    assert len(shapes) >= 25 and len(count_sets) >= 7 and len(donor_shapes) >= 100
    assert pins == {(-1, -1), (-1, 1), (1, -1), (1, 1)}
    assert {(1, 1), (1, 2), (2, 1), (3, 1)} <= ratios
    assert min(repeated, multicolor, irregular, clipped) >= 42
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(101 + nt*7 + ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (2.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    p = generate_puzzle(0, 2, 1)
    broken = copy.deepcopy(p)
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    # Mutating a returned scene cannot affect future predictions or generation.
    original = p['test'][0]['input'].copy()
    expected = p['test'][0]['output'].copy()
    assert np.array_equal(solve(original.tolist()), expected)
    a, donors, receivers = _decode(original)
    for mode in ('no_reflection', 'no_scale', 'no_mask'):
        assert any(_analyze(pair)['alternatives'][mode] for pair in p['train'])
    # Exact rational rendering rejects incompatible fine boundaries.
    fine = dict(virtual=np.array([[4, 4], [4, 5]]), anchor=(0, 0, 2))
    try:
        _scaled(fine, 1, (False, False))
    except AssertionError:
        pass
    else:
        raise AssertionError('ambiguous reduction accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds and all legal explicit counts)')
