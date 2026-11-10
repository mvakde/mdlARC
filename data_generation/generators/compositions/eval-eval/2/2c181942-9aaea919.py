"""Perforated keys feed a printer: docking controls negative-space addition.

All solving conventions are fixed geometry or visible in the query. No parent
code, files, seed, demonstrations, or generation metadata are used by solve.
"""
from __future__ import annotations

import itertools
import json

import numpy as np


OUT = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=int)
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer 2D grid, sides 1..30, colours 0..9")
    return a.astype(int, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(active, colours=None, diagonal=True):
    """Components of a mask, optionally separated by their pixel colours."""
    remaining = set(map(tuple, np.argwhere(active)))
    result = []
    neighbours = N8 if diagonal else N4
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for dr, dc in neighbours:
                p = (r + dr, c + dc)
                if p in remaining and (colours is None or colours[p] == colours[start]):
                    remaining.remove(p)
                    todo.append(p)
        coords = np.array(sorted(cells), dtype=int)
        lo, hi = coords.min(axis=0), coords.max(axis=0)
        mask = np.zeros(tuple(hi - lo + 1), dtype=bool)
        local = coords - lo
        mask[tuple(local.T)] = True
        result.append(dict(coords=coords, origin=lo, mask=mask,
                           colour=None if colours is None else int(colours[start])))
    return result


def _holes(mask):
    h, w = mask.shape
    holes = []
    for component in _components(~mask, diagonal=False):
        rr, cc = component['coords'].T
        if np.all((rr > 0) & (rr < h - 1) & (cc > 0) & (cc < w - 1)):
            holes.append(component)
    return holes


def _canonical(mask):
    candidates = []
    for turns in range(4):
        m = np.rot90(mask, turns).copy()
        h, w = m.shape
        if h < 2 or w < 2 or w % 2:
            continue
        edge = np.zeros(w, dtype=bool)
        edge[w // 2 - 1:w // 2 + 1] = True
        if np.array_equal(m[-1], edge):
            candidates.append((m, turns))
    assert len(candidates) == 1, "attachment must have one geometric orientation"
    return candidates[0]


def _hub(grid, r, c, labels, destination, ink):
    grid[r, c + 2:c + 4] = labels[0]
    grid[r + 2:r + 4, c + 5] = labels[1]
    grid[r + 5, c + 2:c + 4] = labels[2]
    grid[r + 2:r + 4, c] = labels[3]
    if destination == 0:
        grid[r + 1, c + 2:c + 4] = labels[0]
    elif destination == 1:
        grid[r + 2:r + 4, c + 4] = labels[1]
    elif destination == 2:
        grid[r + 4, c + 2:c + 4] = labels[2]
    else:
        grid[r + 2:r + 4, c + 1] = labels[3]
    grid[r + 2:r + 4, c + 2:c + 4] = ink


def _find_hub(a, bg):
    hubs = []
    if min(a.shape) < 6:
        return hubs
    possible = (a[:-5, 2:-3] != bg) & (a[:-5, 2:-3] == a[:-5, 3:-2])
    for r, c in np.argwhere(possible):
        b = a[r:r + 6, c:c + 6]
        pairs = (b[0, 2:4], b[2:4, 5], b[5, 2:4], b[2:4, 0])
        if any(p[0] != p[1] for p in pairs):
            continue
        labels = tuple(int(p[0]) for p in pairs)
        ink = int(b[2, 2])
        if len({bg, ink, *labels}) != 6 or not np.all(b[2:4, 2:4] == ink):
            continue
        for destination in range(4):
            expected = np.full((6, 6), bg, dtype=int)
            _hub(expected, 0, 0, labels, destination, ink)
            if np.array_equal(b, expected):
                hubs.append((np.array([r, c]), labels, destination, ink))
    return hubs


def _dock(canonical, side):
    """Oriented mask and its origin relative to the hub's top-left corner."""
    neck = np.zeros_like(canonical)
    neck[-1, canonical.shape[1] // 2 - 1:canonical.shape[1] // 2 + 1] = True
    oriented = np.rot90(canonical, -side).copy()
    local_neck = np.argwhere(np.rot90(neck, -side))
    targets = (np.array([[-1, 2], [-1, 3]]), np.array([[2, 6], [3, 6]]),
               np.array([[6, 2], [6, 3]]), np.array([[2, -1], [3, -1]]))
    origin = targets[side].min(axis=0) - local_neck.min(axis=0)
    assert set(map(tuple, local_neck + origin)) == set(map(tuple, targets[side]))
    return oriented, origin


def _inspect_input(a):
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    assert counts[bg] > a.size // 2, "background must be the unique majority"
    hubs = _find_hub(a, bg)
    assert len(hubs) == 1, "one visibly encoded hub is required"
    origin, labels, destination, ink = hubs[0]
    active = a != bg
    r, c = origin
    active[r:r + 6, c:c + 6] = False
    objects = _components(active, a)
    assert objects and all(o['colour'] != ink for o in objects)
    # Every separate object has an eight-neighbour blank clearance.
    owner = np.full(a.shape, -1, dtype=int)
    owner[r:r + 6, c:c + 6] = -2
    for index, obj in enumerate(objects):
        owner[tuple(obj['coords'].T)] = index
    for index, obj in enumerate(objects):
        for rr, cc in obj['coords']:
            nearby = owner[max(0, rr - 1):rr + 2, max(0, cc - 1):cc + 2]
            assert np.all((nearby == -1) | (nearby == index)), "objects touch"

    receiver = [o for o in objects if o['colour'] == labels[destination]]
    assert receiver
    stamps = [_canonical(o['mask']) for o in receiver]
    stamp, turns = stamps[0]
    assert not _holes(stamp) and len(_components(stamp, diagonal=False)) == 1
    assert all(t == turns and np.array_equal(m, stamp) for m, t in stamps)
    positions = sorted((o['origin'] for o in receiver), key=lambda p: int(p @ OUT[turns]))
    pitch = stamp.shape[0] + 1
    assert all(np.array_equal(p, positions[0] + i * pitch * OUT[turns])
               for i, p in enumerate(positions)), "receiver stack has a missing/irregular gap"

    keys, unmatched = [], []
    before = 0
    for side, colour in enumerate(labels):
        if side == destination:
            continue
        members = [o for o in objects if o['colour'] == colour]
        assert len(members) <= 1, "a source port has two keys"
        if members:
            obj = members[0]
            m, t = _canonical(obj['mask'])
            assert _holes(m), "source keys must contain enclosed cavities"
            before += sum(np.array_equal(h['mask'], receiver[0]['mask']) for h in _holes(obj['mask']))
            keys.append(dict(side=side, colour=colour, canonical=m, turns=t,
                             delta=(t - side) % 4))
    assert keys
    unmatched = [o for o in objects if o['colour'] not in labels]
    return dict(background=bg, hub=origin, labels=labels, destination=destination,
                ink=ink, stamp=stamp, initial=len(receiver), keys=keys,
                receiver_delta=(turns - destination) % 4, before=before,
                unmatched=unmatched)


def _layout(stamp, initial, destination, keys):
    """Compute the entire assembly and its charge before choosing a canvas."""
    template, receiver_origin = _dock(stamp, destination)
    source_plans, total = [], 0
    reserved = set(itertools.product(range(6), repeat=2))
    all_coords = [np.array(sorted(reserved), dtype=int)]
    for key in keys:
        oriented, origin = _dock(key['canonical'], key['side'])
        cavities = _holes(oriented)
        charge = 0
        footprint = oriented.copy()
        for hole in cavities:
            hole['eligible'] = np.array_equal(hole['mask'], template)
            hole['target'] = hole['coords'] + origin
            footprint[tuple(hole['coords'].T)] = True  # Protect even empty cavities.
            charge += int(hole['eligible'])
        cells = np.argwhere(oriented) + origin
        keep_clear = np.argwhere(footprint) + origin
        assert reserved.isdisjoint(map(tuple, keep_clear)), "assembled keys overlap"
        reserved.update(map(tuple, keep_clear))
        all_coords.append(keep_clear)
        source_plans.append(dict(key=key, cells=cells, cavities=cavities, charge=charge))
        total += charge
    assert total > 0, "at least one cavity must contribute"
    copies = []
    for i in range(initial + total):
        coords = np.argwhere(template) + receiver_origin + i * (stamp.shape[0] + 1) * OUT[destination]
        assert reserved.isdisjoint(map(tuple, coords)), "receiver intersects a key"
        reserved.update(map(tuple, coords))
        copies.append(coords)
        all_coords.append(coords)
    extent = np.concatenate(all_coords)
    return dict(sources=source_plans, copies=copies, total=total, extent=extent)


def _render(shape, bg, hub, labels, destination, ink, layout):
    answer = np.full(shape, bg, dtype=int)
    coords = layout['extent'] + hub
    assert np.all(coords >= 0) and np.all(coords < np.array(shape)), "assembly clips"
    _hub(answer, *hub, labels, destination, ink)
    for source in layout['sources']:
        answer[tuple((source['cells'] + hub).T)] = source['key']['colour']
        for hole in source['cavities']:
            if hole['eligible']:
                answer[tuple((hole['target'] + hub).T)] = ink
    for copy in layout['copies']:
        answer[tuple((copy + hub).T)] = labels[destination]
    assert np.count_nonzero(answer == bg) > answer.size // 2
    return answer


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the assembly and arithmetic solely from this input grid."""
    a = _grid(input_grid)
    p = _inspect_input(a)
    layout = _layout(p['stamp'], p['initial'], p['destination'], p['keys'])
    return _render(a.shape, p['background'], p['hub'], p['labels'],
                   p['destination'], p['ink'], layout)


def _sample_stamp(rng):
    h = int(rng.choice([2, 3]))
    w = int(rng.choice([4, 6]))
    for _ in range(30):
        mask = np.zeros((h, w), dtype=bool)
        mask[:, w // 2 - 1:w // 2 + 1] = True
        mask[0] = True
        if h == 3:
            for c in range(w):
                if c not in (w // 2 - 1, w // 2) and rng.random() < .65:
                    mask[1, c] = True
            if rng.random() < .55:
                c = int(rng.integers(1, w - 1))
                mask[0, c] = False
        if len(_components(mask, diagonal=False)) == 1 and not _holes(mask):
            return _canonical(mask)[0]
    mask = np.zeros((2, w), dtype=bool)
    mask[0] = True
    mask[1, w // 2 - 1:w // 2 + 1] = True
    return mask


def _bad_cavity(rng, stamp, desired, forbidden_delta=None, rotated_only=False):
    if rotated_only or rng.random() < .8:
        alternatives = [d for d in (1, 2, 3) if d != forbidden_delta]
        delta = int(rng.choice(alternatives))
        return np.rot90(stamp, desired + delta).copy()
    # Distinct connected masks make exact silhouette, rather than merely angle,
    # necessary too. They need not possess an attachment.
    for _ in range(12):
        h, w = map(int, rng.integers(2, 5, size=2))
        mask = np.ones((h, w), dtype=bool)
        if rng.random() < .6:
            mask[0, -1] = False
        if not np.array_equal(mask, np.rot90(stamp, desired)):
            return mask
    return np.ones((2, 2), dtype=bool)


def _make_key(rng, cavities, compact=False):
    rim = 1 if compact else int(rng.choice([1, 1, 2]))
    gap = 1 if compact else int(rng.choice([1, 1, 2]))
    stem = 1 if compact else int(rng.integers(1, 4))
    transverse = len(cavities) > 1 and rng.random() < .4
    extra = 0 if compact else int(rng.choice([0, 0, 2]))
    if transverse:
        body_h = max(m.shape[0] for m in cavities) + 2 * rim
        body_w = sum(m.shape[1] for m in cavities) + (len(cavities) - 1) * gap + 2 * rim + extra
    else:
        body_h = sum(m.shape[0] for m in cavities) + (len(cavities) - 1) * gap + 2 * rim
        body_w = max(m.shape[1] for m in cavities) + 2 * rim + extra
    body_w += body_w % 2
    mask = np.zeros((body_h + stem, body_w), dtype=bool)
    mask[:body_h] = True
    mask[body_h:, body_w // 2 - 1:body_w // 2 + 1] = True
    cursor = rim
    for cavity in cavities:
        h, w = cavity.shape
        if transverse:
            row = int(rng.integers(rim, body_h - rim - h + 1))
            col = cursor
            cursor += w + gap
        else:
            row = cursor
            col = int(rng.integers(rim, body_w - rim - w + 1))
            cursor += h + gap
        block = mask[row:row + h, col:col + w]
        block[cavity] = False
    if rim == 2 and not compact and rng.random() < .65:
        # Exterior notches cannot open a cavity through its retained inner rim.
        c = int(rng.integers(1, body_w - 1))
        mask[0, c] = False
    assert len(_holes(mask)) == len(cavities)
    assert len(_components(mask, diagonal=False)) == 1
    return _canonical(mask)[0]


def _stack_mask(stamp, count):
    h, w = stamp.shape
    result = np.zeros((count * h + count - 1, w), dtype=bool)
    for i in range(count):
        result[i * (h + 1):i * (h + 1) + h] = stamp
    return result


def _pack(rng, shape, bg, hub, labels, destination, ink, loose):
    a = np.full(shape, bg, dtype=int)
    _hub(a, *hub, labels, destination, ink)
    blocked = np.zeros(shape, dtype=bool)
    r, c = hub
    blocked[max(0, r - 1):r + 7, max(0, c - 1):c + 7] = True
    order = list(rng.permutation(len(loose)))
    order.sort(key=lambda i: -loose[i][0].size)
    for index in order:
        mask, colour = loose[index]
        h, w = mask.shape
        if h > shape[0] or w > shape[1]:
            return None
        # Reserve complete rectangles so receiver gaps and key cavities cannot
        # hide other objects. Integral images enumerate positions cheaply.
        sums = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        occupied = sums[h:, w:] - sums[:-h, w:] - sums[h:, :-w] + sums[:-h, :-w]
        choices = np.argwhere(occupied == 0)
        if not len(choices):
            return None
        top, left = map(int, choices[int(rng.integers(len(choices)))])
        a[top:top + h, left:left + w][mask] = colour
        blocked[max(0, top - 1):top + h + 1, max(0, left - 1):left + w + 1] = True
    return a


def _build_pair(rng, stamp, destination, initial, charges, anchor=False,
                compact=False, distractor_count=None):
    palette = list(map(int, rng.permutation(10)))
    bg, ink, labels = palette[0], palette[1], tuple(palette[2:6])
    sides = list(map(int, rng.choice([s for s in range(4) if s != destination],
                                   len(charges), replace=False)))
    keys, loose = [], []
    receiver_delta = 0 if anchor else int(rng.integers(4))
    loose.append((np.rot90(_stack_mask(stamp, initial), -destination - receiver_delta).copy(),
                  labels[destination]))
    for index, (side, charge) in enumerate(zip(sides, charges)):
        desired = side - destination
        delta = index + 1 if anchor else int(rng.integers(4))
        holes = [np.rot90(stamp, desired).copy() for _ in range(charge)]
        if charge == 0 or (charge == 1 and not anchor and rng.random() < .55):
            holes.append(_bad_cavity(rng, stamp, desired,
                                     forbidden_delta=delta if anchor else None,
                                     rotated_only=anchor))
        canonical = _make_key(rng, holes, compact)
        keys.append(dict(side=side, colour=labels[side], canonical=canonical, delta=delta))
        loose.append((np.rot90(canonical, -side - delta).copy(), labels[side]))
    if distractor_count is None:
        distractor_count = 1 if anchor else int(rng.choice([0, 1, 2], p=[.5, .4, .1]))
    for index in range(distractor_count):
        if anchor or index == 0 or rng.random() < .6:
            mask = _make_key(rng, [np.rot90(stamp, int(rng.integers(4))).copy()], compact=True)
        else:
            mask = np.ones((int(rng.integers(2, 5)), int(rng.integers(2, 5))), dtype=bool)
            mask[0, -1] = False
        loose.append((np.rot90(mask, int(rng.integers(4))).copy(), int(rng.choice(palette[6:]))))
    try:
        layout = _layout(stamp, initial, destination, keys)
    except AssertionError:
        return None
    assert [s['charge'] for s in layout['sources']] == list(charges)
    low, high = layout['extent'].min(axis=0), layout['extent'].max(axis=0)
    minimum = high - low + 1
    if np.any(minimum > 30):
        return None
    # Sample headroom from actual assembled geometry, not a split size template.
    minimum = np.maximum(minimum, 16 if len(keys) <= 1 else 20)
    shape = tuple(int(rng.integers(int(v), 31)) for v in minimum)
    for attempt in range(10):
        if attempt == 4:
            shape = (30, 30)
        lo, hi = -low, np.array(shape) - 1 - high
        if rng.random() < .25:
            hub = np.array([int(rng.choice([x, y])) for x, y in zip(lo, hi)])
        else:
            hub = np.array([int(rng.integers(x, y + 1)) for x, y in zip(lo, hi)])
        a = _pack(rng, shape, bg, hub, labels, destination, ink, loose)
        if a is None or np.count_nonzero(a == bg) <= a.size // 2:
            continue
        try:
            b = _render(shape, bg, hub, labels, destination, ink, layout)
            parsed = _inspect_input(a)
            recovered = _layout(parsed['stamp'], parsed['initial'], destination, parsed['keys'])
            assert recovered['total'] == sum(charges)
            assert np.array_equal(solve(a), b)
            if anchor:
                assert parsed['before'] == 0
        except AssertionError:
            continue
        return dict(input=a, output=b)
    return None


def _arithmetic_evidence(operands):
    totals = [sum(s) for s, d in operands]
    dests = [d for s, d in operands]
    if len(set(totals)) < 2 or len(set(dests)) < 2:
        return False
    if len({s + d for s, d in zip(totals, dests)}) < 2:
        return False
    if not any(totals[0] * d != s * dests[0] for s, d in zip(totals, dests)):
        return False
    rivals = (lambda s, d: 2 * d, lambda s, d: 2 * sum(s),
              lambda s, d: d * sum(s), lambda s, d: max(d, sum(s)),
              lambda s, d: sum(s), lambda s, d: d + len(s),
              lambda s, d: d + sum(x > 0 for x in s), lambda s, d: d + max(s))
    return all(any(rival(s, d) != d + sum(s) for s, d in operands) for rival in rivals)


def _counts(num_train, num_test):
    for v, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not lo <= v <= hi:
            raise ValueError("illegal training/test count")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    stamp = _sample_stamp(rng)
    train_directions = list(map(int, rng.choice(4, 2, replace=False)))
    test_directions = [s for s in range(4) if s not in train_directions]
    second_options = [(s, d) for s in ((1,), (2,), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2), (0, 1, 2))
                      for d in range(1, 4) if sum(s) + d <= 6]
    anchors = None
    for attempt in range(160):
        # Required witness properties, not a fixed order/count progression.
        first_charges = (2, int(rng.integers(2)), 0)
        first_initial = int(rng.integers(1, 3))
        compatible = [(s, d) for s, d in second_options
                      if _arithmetic_evidence([(first_charges, first_initial), (s, d)])]
        second_charges, second_initial = compatible[int(rng.integers(len(compatible)))]
        compact = attempt >= 10
        if attempt >= 100:
            # This compact witness combination is feasible for all sampled
            # stamp sizes: a full [2,1,0] assembly, then one adjacent key.
            first_charges, first_initial = (2, 1, 0), 1
            second_charges, second_initial = (2,), 3
        first = _build_pair(rng, stamp, train_directions[0], first_initial,
                            first_charges, anchor=True, compact=compact)
        if first is None:
            continue
        second = _build_pair(rng, stamp, train_directions[1], second_initial,
                             second_charges, compact=compact)
        if second is not None:
            anchors = [first, second]
            break
    if anchors is None:
        raise RuntimeError("bounded compact witness construction exhausted")
    puzzle = {'train': anchors, 'test': []}
    seen = {_key(p['input']) for p in anchors}
    for split, count, directions in (('train', num_train - 2, train_directions),
                                      ('test', num_test, test_directions)):
        for _ in range(count):
            pair = None
            for attempt in range(140):
                source_count = int(rng.integers(1, 4))
                charges = tuple(map(int, rng.integers(0, 3, size=source_count)))
                initial = int(rng.integers(1, 4))
                if not 1 <= sum(charges) <= 4 or initial + sum(charges) > 6:
                    continue
                if attempt >= 90:
                    charges, initial = (1,), int(rng.integers(1, 4))
                candidate = _build_pair(rng, stamp, int(rng.choice(directions)), initial,
                                        charges, compact=attempt >= 12,
                                        distractor_count=0 if attempt >= 90 else None)
                if candidate is not None and _key(candidate['input']) not in seen:
                    pair = candidate
                    break
            if pair is None:
                raise RuntimeError("bounded pair construction exhausted")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    a, b = _grid(pair['input']), _grid(pair['output'])
    p = _inspect_input(a)
    layout = _layout(p['stamp'], p['initial'], p['destination'], p['keys'])
    expected = _render(a.shape, p['background'], p['hub'], p['labels'],
                       p['destination'], p['ink'], layout)
    assert np.array_equal(b, expected), "answer violates docking/charge/copying"
    charges = tuple(s['charge'] for s in layout['sources'])
    bad = sum(not h['eligible'] for s in layout['sources'] for h in s['cavities'])
    template = np.rot90(p['stamp'], -p['destination'])
    wrong_angle = sum(not h['eligible'] and any(np.array_equal(np.rot90(h['mask'], t), template)
                      for t in (1, 2, 3)) for s in layout['sources'] for h in s['cavities'])
    # Pixel-level conservation checks supplement exact solver agreement.
    for source in layout['sources']:
        assert np.all(b[tuple((source['cells'] + p['hub']).T)] == source['key']['colour'])
        for hole in source['cavities']:
            wanted = p['ink'] if hole['eligible'] else p['background']
            assert np.all(b[tuple((hole['target'] + p['hub']).T)] == wanted)
    receiver_pixels = int(np.count_nonzero(b == p['labels'][p['destination']])) - 4
    assert receiver_pixels == (p['initial'] + sum(charges)) * int(p['stamp'].sum())
    assert np.count_nonzero(b == p['ink']) == 4 + sum(charges) * int(p['stamp'].sum())
    assert np.array_equal(a[p['hub'][0]:p['hub'][0] + 6, p['hub'][1]:p['hub'][1] + 6],
                          b[p['hub'][0]:p['hub'][0] + 6, p['hub'][1]:p['hub'][1] + 6])
    assert len(layout['copies']) == p['initial'] + sum(charges)
    return dict(parsed=p, layout=layout, charges=charges, bad=bad, wrong_angle=wrong_angle,
                stamp=_key(p['stamp']), shape=a.shape)


def _perforated_distractor_witness(obj):
    if not _holes(obj['mask']):
        return False
    try:
        _canonical(obj['mask'])
        return True
    except AssertionError:
        return False


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    analyses, seen = {}, set()
    for split in ('train', 'test'):
        analyses[split] = []
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == b.shape and not np.array_equal(a, b)
            assert _key(a) not in seen, "duplicate input"
            seen.add(_key(a))
            analyses[split].append(_features(pair))
    train = analyses['train']
    operands = [(f['charges'], f['parsed']['initial']) for f in train]
    assert _arithmetic_evidence(operands), "arithmetic shortcuts fit training"
    sides, turns = set(), set()
    for f in train:
        p = f['parsed']
        sides.update([p['destination']] + [k['side'] for k in p['keys']])
        turns.update([p['receiver_delta']] + [k['delta'] for k in p['keys']])
    assert sides == turns == set(range(4)), "training lacks docking witnesses"
    assert any(f['bad'] for f in train), "demonstrate orientation rejection"
    assert any(f['wrong_angle'] for f in train), "distinguish matching up to rotation"
    assert any(0 in f['charges'] for f in train), "demonstrate an uncharged docked key"
    assert any(max(f['charges']) >= 2 for f in train), "count holes, not keys"
    assert any(sum(f['charges']) > max(f['charges']) for f in train), "teach aggregation"
    assert any(f['parsed']['before'] != sum(f['charges']) for f in train), "docking must affect the sum"
    assert any(any(_perforated_distractor_witness(o) for o in f['parsed']['unmatched'])
               for f in train), "teach colour filtering with an attachable key"
    train_destinations = {f['parsed']['destination'] for f in train}
    assert len(train_destinations) == 2
    assert all(f['parsed']['destination'] not in train_destinations for f in analyses['test'])
    # Omitting B (assembly without growth) changes every answer by S stamps.
    assert all(sum(f['charges']) > 0 for aa in analyses.values() for f in aa)


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def self_test():
    """Bounded 42-seed variation, evidence, determinism and mutation checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    rows = [_features(pair) for p in puzzles for split in ('train', 'test') for pair in p[split]]
    shapes = {f['shape'] for f in rows}
    stamps = {f['stamp'] for f in rows}
    key_shapes = {_key(k['canonical']) for f in rows for k in f['parsed']['keys']}
    palettes = {(f['parsed']['background'], f['parsed']['ink'], f['parsed']['labels']) for f in rows}
    charges = {f['charges'] for f in rows}
    operands = {(f['parsed']['initial'], sum(f['charges'])) for f in rows}
    required_turns = {k['delta'] for f in rows for k in f['parsed']['keys']}
    assert len(shapes) >= 25 and len(stamps) >= 8 and len(key_shapes) >= 60
    assert len(palettes) >= 100 and len(charges) >= 15 and len(operands) >= 10
    assert required_turns == set(range(4))
    assert {1, 2, 3} <= {len(f['charges']) for f in rows}
    assert {0, 1, 2} <= {len(f['parsed']['unmatched']) for f in rows}
    # Counts: all 30 legal combinations, including maximum and minimum budgets.
    legal = 0
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(700 + 10 * a + b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
            legal += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Corrupt selected semantic features; validation must reject each answer.
    original = puzzles[0]
    for defect in ('growth', 'filling', 'material', 'hub'):
        broken = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in original[s]]
                  for s in ('train', 'test')}
        pair = broken['train'][0]
        f = _features(pair)
        p, layout = f['parsed'], f['layout']
        if defect == 'growth':
            pair['output'][tuple((layout['copies'][-1] + p['hub']).T)] = p['background']
            coord = layout['copies'][-1][0] + p['hub']
        elif defect == 'filling':
            hole = next(h for source in layout['sources'] for h in source['cavities'] if h['eligible'])
            coord = hole['target'][0] + p['hub']
        elif defect == 'material':
            coord = layout['sources'][0]['cells'][0] + p['hub']
        else:
            coord = p['hub'] + [2, 2]
        pair['output'][tuple(coord)] = p['background']
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError("corrupt " + defect + " accepted")
    # Turning an existing key in place is a different input, not hidden state:
    # individual mask turns enumerate the attachment parser and hole rotation.
    for f in rows[:12]:
        for key in f['parsed']['keys']:
            for t in range(4):
                recovered, undo = _canonical(np.rot90(key['canonical'], t))
                assert np.array_equal(recovered, key['canonical']) and undo == (-t) % 4
    # Whole-input rotation and arbitrary palette permutation preserve the rule.
    pair = original['train'][0]
    for t in (1, 2, 3):
        assert np.array_equal(solve(np.rot90(pair['input'], t)), np.rot90(pair['output'], t))
    lookup = np.random.default_rng(918).permutation(10)
    assert np.array_equal(solve(lookup[pair['input']]), lookup[pair['output']])

    # A valid input mutation: turn one cavity 180 degrees without moving its
    # chamber. Its area/number stay fixed, but its contribution drops by one.
    pair = next(p for p in original['train']
                if _features(p)['bad'] and sum(_features(p)['charges']) >= 2)
    f = _features(pair)
    p = f['parsed']
    a = pair['input'].copy()
    key = next(k for k in p['keys'] if any(np.array_equal(np.rot90(h['mask'], -k['side']),
                      np.rot90(p['stamp'], -p['destination'])) for h in _holes(k['canonical'])))
    changed = key['canonical'].copy()
    hole = next(h for h in _holes(changed) if np.array_equal(np.rot90(h['mask'], -key['side']),
                                      np.rot90(p['stamp'], -p['destination'])))
    y, x = hole['origin']
    h, w = hole['mask'].shape
    block = changed[y:y + h, x:x + w]
    block[:] = True
    block[np.rot90(hole['mask'], 2)] = False
    outside = a != p['background']
    r, c = p['hub']
    outside[r:r + 6, c:c + 6] = False
    obj = next(o for o in _components(outside, a) if o['colour'] == key['colour'])
    rr, cc = obj['origin']
    oriented = np.rot90(changed, -key['turns'])
    a[rr:rr + oriented.shape[0], cc:cc + oriented.shape[1]] = np.where(oriented, key['colour'], p['background'])
    updated = _features({'input': a, 'output': solve(a)})
    assert sum(updated['charges']) == sum(f['charges']) - 1
    assert updated['bad'] == f['bad'] + 1

    # Input-only solving extends beyond the sampler's two-hole keys.
    rng = np.random.default_rng(4321)
    stamp = np.array([[1, 1, 1, 1], [0, 1, 1, 0]], dtype=bool)
    extra = None
    for _ in range(20):
        extra = _build_pair(rng, stamp, 0, 1, (3,), compact=True, distractor_count=0)
        if extra is not None:
            break
    assert extra is not None and sum(_features(extra)['charges']) == 3
    try:
        _canonical(np.ones((2, 2), dtype=bool))
    except AssertionError:
        pass
    else:
        raise AssertionError("symmetric attachment silently disambiguated")
    return dict(seeds=42, pairs=len(rows), legal_count_combinations=legal,
                grid_shapes=len(shapes), receiver_masks=len(stamps), key_masks=len(key_shapes),
                palettes=len(palettes), charge_vectors=len(charges), operand_pairs=len(operands),
                semantic_verification="generator-owned checks only")


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
