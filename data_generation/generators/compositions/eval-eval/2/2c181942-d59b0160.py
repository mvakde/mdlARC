"""Missing-color docking: unordered membership produces a rigid-motion address.

No parent generator, source fixture, seed, or generation state is used by solve.
Only Python's standard library and NumPy are required.
"""
from __future__ import annotations

import numpy as np


_NEIGHBORS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "Invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "Grid cells must be integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "Grid colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _hub_points(r, c):
    return np.array(((r, c+1), (r, c+2), (r+1, c+3), (r+2, c+3),
                     (r+3, c+1), (r+3, c+2), (r+1, c), (r+2, c)), dtype=int)


def _draw_hub(a, r, c, labels):
    points = _hub_points(r, c)
    a[points[:, 0], points[:, 1]] = np.repeat(labels, 2)


def _find_hub(a, bg):
    h, w = a.shape
    allowed = np.zeros((4, 4), dtype=bool)
    allowed[0, 1:3] = allowed[3, 1:3] = True
    allowed[1:3, 0] = allowed[1:3, 3] = True
    candidates = []
    for r in range(h-3):
        for c in range(w-3):
            if a[r, c+1] == bg or a[r, c+1] != a[r, c+2]:
                continue
            b = a[r:r+4, c:c+4]
            if np.any(b[~allowed] != bg):
                continue
            bars = (b[0, 1:3], b[1:3, 3], b[3, 1:3], b[1:3, 0])
            labels = tuple(int(x[0]) for x in bars)
            if bg not in labels and len(set(labels)) == 4 and all(x[0] == x[1] for x in bars):
                candidates.append((r, c, labels))
    _require(len(candidates) == 1, "Expected one hollow four-bar hub")
    return candidates[0]


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells = [start]
        for r, c in cells:
            for dr, dc in _NEIGHBORS:
                p = (r+dr, c+dc)
                if p in remaining:
                    remaining.remove(p)
                    cells.append(p)
        result.append(np.asarray(cells, dtype=int))
    return result


def _attachment_options(mask):
    options = []
    for k in range(4):
        m = np.rot90(mask, k)
        h, w = m.shape
        if h < 2 or w < 4 or w % 2:
            continue
        edge = np.zeros(w, dtype=bool)
        edge[w//2-1:w//2+1] = True
        if np.array_equal(m[-1], edge):
            options.append(k)
    return options


def _canonical(block, bg):
    options = _attachment_options(block != bg)
    results = {_key(np.rot90(block, k)): k for k in options}
    _require(len(results) == 1, "Missing or inequivalent attachment orientations")
    k = next(iter(results.values()))
    return np.rot90(block, k).copy(), tuple(options)


def _dock(block, bg, side, r, c):
    """Return destination coordinates and colors; canonical connector points down."""
    neck = np.zeros(block.shape, dtype=bool)
    neck[-1, block.shape[1]//2-1:block.shape[1]//2+1] = True
    oriented = np.rot90(block, -side)
    attachment = np.argwhere(np.rot90(neck, -side))
    targets = (
        ((r-1, c+1), (r-1, c+2)),
        ((r+1, c+4), (r+2, c+4)),
        ((r+4, c+1), (r+4, c+2)),
        ((r+1, c-1), (r+2, c-1)),
    )[side]
    shift = np.asarray(targets).min(axis=0) - attachment.min(axis=0)
    local = np.argwhere(oriented != bg)
    return local + shift, oriented[local[:, 0], local[:, 1]]


def _analyze(input_grid):
    a = _grid(input_grid)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(int(counts[bg]) * 2 > a.size, "Background must be a strict majority")
    r, c, labels = _find_hub(a, bg)
    mask = a != bg
    hp = _hub_points(r, c)
    mask[hp[:, 0], hp[:, 1]] = False
    objects = []
    used = set()
    fillers = set()
    for coords in _components(mask):
        lo, hi = coords.min(axis=0), coords.max(axis=0)+1
        b = a[lo[0]:hi[0], lo[1]:hi[1]].copy()
        # Distinct eight-connected components must not be interleaved through
        # background holes in another object's bounding box.
        local = coords-lo
        isolated = np.full(b.shape, bg, dtype=np.int8)
        isolated[local[:, 0], local[:, 1]] = a[coords[:, 0], coords[:, 1]]
        _require(np.array_equal(b, isolated), "Interleaved object bounding boxes")
        canonical, options = _canonical(b, bg)
        filler_pair = canonical[-1, canonical.shape[1]//2-1:canonical.shape[1]//2+1]
        _require(filler_pair[0] == filler_pair[1], "Connector must have one filler color")
        filler = int(filler_pair[0])
        _require(filler != bg and filler not in labels, "Filler must be a non-hub color")
        fillers.add(filler)
        present = frozenset(map(int, np.unique(b[b != bg])))
        intersection = present & set(labels)
        side = labels.index(next(iter(set(labels)-intersection))) if len(intersection) == 3 else None
        obj = dict(coords=coords, origin=lo, block=b, canonical=canonical,
                   mask=canonical != bg, options=options, filler=filler,
                   present=present, intersection=intersection, side=side)
        if side is not None:
            _require(side not in used, "Duplicate eligible address")
            used.add(side)
            target, values = _dock(canonical, bg, side, r, c)
            _require(bool(np.all((target >= 0) & (target < np.array(a.shape)))), "Docking clips canvas")
            obj.update(target=target, values=values,
                       turns=frozenset((k-side) % 4 for k in options))
        objects.append(obj)
    _require(bool(objects) and len(fillers) == 1, "Expected loose records with common connector filler")
    # The hub's empty corner/interior region is reserved, with a one-cell gap.
    for obj in objects:
        rr, cc = obj['coords'].T
        _require(not bool(np.any((rr >= r-1) & (rr <= r+4) & (cc >= c-1) & (cc <= c+4))),
                 "Loose record touches hub region")
    out = a.copy()
    for obj in objects:
        if obj['side'] is not None:
            rr, cc = obj['coords'].T
            out[rr, cc] = bg
    for obj in objects:
        if obj['side'] is not None:
            rr, cc = obj['target'].T
            _require(bool(np.all(out[rr, cc] == bg)), "Destination overlaps retained content")
            out[rr, cc] = obj['values']
    return out, dict(background=bg, filler=next(iter(fillers)), hub=(r, c),
                     labels=labels, objects=objects, sides=frozenset(used))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete answer from this input alone, returning a fresh array."""
    return _analyze(input_grid)[0]


def _mask(rng, minimum_marks):
    for _ in range(80):
        w = int(rng.choice((4, 6, 8), p=(.43, .38, .19)))
        h = int(rng.integers(2 if w < 8 else 3, 8))
        stem = int(rng.integers(1 if w < 8 else 2, h))
        body = h-stem
        mid = w//2
        kind = int(rng.integers(4))
        m = np.zeros((h, w), dtype=bool)
        m[:, mid-1:mid+1] = True
        if kind == 0:
            m[:body] = True
        elif kind == 1:
            for row in range(body):
                reach = max(2, mid-row)
                m[row, mid-reach:mid+reach] = True
        elif kind == 2:
            m[:body] = True
            for row in range(body):
                for col in range(w):
                    if col not in (mid-1, mid) and rng.random() < .24:
                        m[row, col] = False
            # Support both transverse extremes; later connectivity/ambiguity
            # checks reject detached or competing shoulders.
            m[int(rng.integers(body)), 0] = True
            m[int(rng.integers(body)), -1] = True
        else:
            wing = mid-1
            if body < wing:
                continue
            m[:wing] = False
            for row in range(wing):
                m[row, row] = m[row, w-1-row] = True
                if rng.random() < .4:
                    m[row, row:w-row] = True
        if not m[:, 0].any() or not m[:, -1].any():
            continue
        if len(_components(m)) != 1 or _attachment_options(m) != [0]:
            continue
        if int(m.sum())-2 >= minimum_marks:
            return m
    # Enough body cells for every teaching package, still directionally unique.
    rows = max(2, (minimum_marks+3)//4)
    m = np.ones((rows+1, 4), dtype=bool)
    m[-1] = (False, True, True, False)
    return m


def _paint(rng, m, marks, bg, filler):
    b = np.full(m.shape, bg, dtype=np.int8)
    b[m] = filler
    available = m.copy()
    available[-1] = False
    coords = np.argwhere(available)
    _require(len(coords) >= len(marks), "Insufficient marker capacity")
    selected = coords[rng.choice(len(coords), len(marks), replace=False)]
    if len(marks):
        b[selected[:, 0], selected[:, 1]] = rng.permutation(marks)
    return b


def _free_marks(rng, labels, extras, side):
    if side is None:
        n = int(rng.choice((0, 1, 2, 4), p=(.22, .18, .42, .18)))
        marks = list(map(int, rng.choice(labels, n, replace=False)))
    else:
        marks = [v for s, v in enumerate(labels) if s != side]
    n = int(rng.integers(len(extras)+1))
    marks += list(map(int, rng.choice(extras, n, replace=False)))
    if marks:
        marks += list(map(int, rng.choice(marks, int(rng.integers(4)))))
    return marks


def _objects(rng, labels, extras, sides, deltas, package):
    objects = []
    for i, side in enumerate(sides):
        marks = [v for s, v in enumerate(labels) if s != side]
        if package == 'extras':
            marks += [extras[0], marks[0], extras[0]]
        elif package == 'free':
            marks = _free_marks(rng, labels, extras, side)
        objects.append(dict(side=side, marks=marks, mask=_mask(rng, len(marks)),
                            input_turns=(-side-deltas[i]) % 4))
    if package == 'identity':
        present = objects[0]['marks']
        negatives = [([present[0], present[1], extras[0]], objects[0]['mask'].copy()),
                     ([labels[0], labels[1], extras[0], extras[1], labels[0], extras[0]], None),
                     (list(labels), None)]
    elif package == 'extras':
        negatives = [([labels[0], labels[1], extras[0], extras[1], labels[0], extras[0]], None),
                     ([], None)]
    else:
        count = int(rng.integers(4))
        if not sides:
            count = max(1, count)
        negatives = [(_free_marks(rng, labels, extras, None), None) for _ in range(count)]
    for marks, mask in negatives:
        objects.append(dict(side=None, marks=marks,
                            mask=_mask(rng, len(marks)) if mask is None else mask,
                            input_turns=int(rng.integers(4))))
    return objects


def _halo(blocked, coords):
    h, w = blocked.shape
    for r, c in coords:
        blocked[max(0, r-1):min(h, r+2), max(0, c-1):min(w, c+2)] = True


def _packing(rng, shape, hub, labels, bg, objects):
    a = np.full(shape, bg, dtype=np.int8)
    r, c = hub
    _draw_hub(a, r, c, labels)
    blocked = np.zeros(shape, dtype=bool)
    blocked[max(0, r-1):r+5, max(0, c-1):c+5] = True
    # Reserve the future assembly as empty working space for every loose source.
    for obj in objects:
        if obj['side'] is not None:
            coords, _ = _dock(obj['colored'], bg, obj['side'], r, c)
            _halo(blocked, coords)
    order = list(map(int, rng.permutation(len(objects))))
    order.sort(key=lambda i: -objects[i]['colored'].size)
    for i in order:
        obj = objects[i]
        b = np.rot90(obj['colored'], obj['input_turns'])
        h, w = b.shape
        local = np.argwhere(b != bg)
        tops, lefts = np.indices((shape[0]-h+1, shape[1]-w+1))
        rr = tops.ravel()[:, None] + local[:, 0]
        cc = lefts.ravel()[:, None] + local[:, 1]
        good = np.flatnonzero(~np.any(blocked[rr, cc], axis=1))
        if not len(good):
            return None
        if rng.random() < .35:
            edge = good[(tops.ravel()[good] == 0) | (lefts.ravel()[good] == 0)
                        | (tops.ravel()[good]+h == shape[0]) | (lefts.ravel()[good]+w == shape[1])]
            good = edge if len(edge) else good
        k = int(rng.choice(good))
        # Reserve the whole bounding box too: no interleaving through holes.
        top, left = int(tops.ravel()[k]), int(lefts.ravel()[k])
        a[rr[k], cc[k]] = b[local[:, 0], local[:, 1]]
        box = np.argwhere(np.ones(b.shape, dtype=bool)) + (top, left)
        _halo(blocked, box)
    return a


def _make_pair(rng, sides, deltas, package, previous_bg):
    palette = list(map(int, rng.permutation(10)))
    if palette[0] == previous_bg:
        palette[0], palette[1] = palette[1], palette[0]
    bg, filler = palette[:2]
    labels = tuple(palette[2:6])
    extras = palette[6:6+int(rng.integers(2 if package != 'free' else 0, 5))]
    objs = _objects(rng, labels, extras, sides, deltas, package)
    for obj in objs:
        obj['colored'] = _paint(rng, obj['mask'], obj['marks'], bg, filler)
    relative = [_hub_points(0, 0)]
    for obj in objs:
        if obj['side'] is not None:
            relative.append(_dock(obj['colored'], bg, obj['side'], 0, 0)[0])
    coords = np.concatenate(relative)
    lower, upper = coords.min(axis=0), coords.max(axis=0)
    extent = upper-lower+1
    shape = np.maximum(rng.integers(16, 31, size=2), extent)
    for attempt in range(24):
        low, high = -lower, shape-1-upper
        hub = tuple(int(rng.choice((x, y))) if rng.random() < .35
                    else int(rng.integers(x, y+1)) for x, y in zip(low, high))
        a = _packing(rng, tuple(map(int, shape)), hub, labels, bg, objs)
        if a is not None and int(np.count_nonzero(a == bg))*2 > a.size:
            try:
                output, _ = _analyze(a)
            except ValueError:
                pass
            else:
                return dict(input=a, output=output)
        shape = np.minimum(30, shape+1)
    # A compact packing fallback retains the same sides, turns, and multisets.
    a = np.full((30, 30), bg, dtype=np.int8)
    _draw_hub(a, 13, 13, labels)
    slots = ((0, 0), (0, 7), (0, 14), (0, 21), (24, 0), (24, 7), (24, 14))
    slot_order = rng.permutation(len(slots))
    for i, obj in enumerate(objs):
        m = np.ones((max(2, (len(obj['marks'])+3)//4)+1, 4), dtype=bool)
        m[-1] = (False, True, True, False)
        b = np.rot90(_paint(rng, m, obj['marks'], bg, filler), obj['input_turns'])
        r, c = slots[int(slot_order[i])]
        a[r:r+b.shape[0], c:c+b.shape[1]] = b
    return dict(input=a, output=solve(a))


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return value


def _bits(sides):
    return sum(1 << s for s in sides)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    _require(num_train > num_test, 'num_train must exceed num_test')
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and seed >= 0, 'seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    sides = list(map(int, rng.permutation(4)))
    rotations = list(map(int, rng.permutation(4)))
    configs = [(sides[:2], rotations[:2], 'identity'),
               (sides[2:], rotations[2:], 'extras')]
    for _ in range(num_train-2):
        n = int(rng.integers(5))
        ss = list(map(int, rng.choice(4, n, replace=False)))
        configs.append((ss, list(map(int, rng.integers(4, size=n))), 'free'))
    train_bits = {_bits(ss) for ss, _, _ in configs}
    available = [b for b in range(1, 16) if b not in train_bits]
    for _ in range(num_test):
        bit = int(rng.choice(available))
        ss = [s for s in range(4) if bit & (1 << s)]
        configs.append((ss, list(map(int, rng.integers(4, size=len(ss)))), 'free'))
    puzzle = {'train': [], 'test': []}
    seen = set()
    previous_bg = None
    for i, (ss, deltas, package) in enumerate(configs):
        for _ in range(16):
            pair = _make_pair(rng, ss, deltas, package, previous_bg)
            if _key(pair['input']) not in seen:
                break
        else:
            raise RuntimeError('Distinct-input sampling exhausted')
        seen.add(_key(pair['input']))
        previous_bg = int(np.argmax(np.bincount(pair['input'].ravel(), minlength=10)))
        puzzle['train' if i < num_train else 'test'].append(pair)
    order = rng.permutation(num_train)
    puzzle['train'] = [puzzle['train'][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def _training_coverage(infos):
    _require(set().union(*(set(i['sides']) for i in infos)) == set(range(4)), 'Teach every destination')
    positives = [(i, o) for i in infos for o in i['objects'] if o['side'] is not None]
    negatives = [(i, o) for i in infos for o in i['objects'] if o['side'] is None]
    turns = set().union(*(set(o['turns']) for _, o in positives))
    _require(turns == set(range(4)), 'Teach every rotation amount')
    def marks(i, o):
        return o['canonical'][o['mask'] & (o['canonical'] != i['filler'])]
    _require(any(len(marks(i, o)) == 3 for i, o in positives), 'Need exact three-mark positive')
    _require(any(len(o['intersection']) == 4 for _, o in negatives), 'Need all-four negative')
    _require(any(len(marks(i, o)) == 0 for i, o in negatives), 'Need unmarked negative')
    _require(any(len(o['present']-{i['background'], i['filler']}) >= 4
                 and len(o['intersection']) == 2 and len(marks(i, o)) > 3 for i, o in negatives),
             'Need richer incomplete negative')
    for group in (positives, negatives):
        _require(any(len(marks(i, o)) > len(set(map(int, marks(i, o))))
                     and bool(o['present']-set(i['labels'])-{i['filler']}) for i, o in group),
                 'Teach irrelevant repetition and extra identities')
    # Same shape, same three-mark count and distinct-color count, opposite action.
    _require(any(len(marks(i, p)) == len(marks(i, n)) == 3
                 and len(set(map(int, marks(i, n)))) == 3
                 and np.array_equal(p['mask'], n['mask'])
                 for i in infos for p in i['objects'] if p['side'] is not None
                 for n in i['objects'] if n['side'] is None), 'Missing matched-shape identity witness')


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'Invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'Splits must be lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'Invalid pair counts')
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'Invalid pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'Output dimensions changed')
            _require(_key(inp) not in seen, 'Duplicate input')
            seen.add(_key(inp))
            expected, info = _analyze(inp)
            _require(np.array_equal(expected, out), 'Incorrect complementary assembly')
            _require(np.array_equal(np.bincount(inp.ravel(), minlength=10),
                                   np.bincount(out.ravel(), minlength=10)), 'Colored-cell conservation failed')
            r, c = info['hub']
            _require(np.array_equal(inp[r:r+4, c:c+4], out[r:r+4, c:c+4]), 'Hub changed')
            for obj in info['objects']:
                rr, cc = obj['coords'].T
                if obj['side'] is None:
                    _require(np.array_equal(inp[rr, cc], out[rr, cc]), 'Rejected object damaged')
                else:
                    _require(bool(np.all(out[rr, cc] == info['background'])), 'Moved source not cleared')
                    tr, tc = obj['target'].T
                    _require(np.array_equal(out[tr, tc], obj['values']), 'Rigid colored shape not preserved')
            infos[split].append(info)
    _training_coverage(infos['train'])
    observed = {_bits(i['sides']) for i in infos['train']}
    _require(all(i['sides'] and _bits(i['sides']) not in observed for i in infos['test']),
             'Test must recombine demonstrated sides in an unseen subset')


def _puzzle_key(p):
    return tuple(tuple((_key(pair['input']), _key(pair['output'])) for pair in p[s]) for s in ('train', 'test'))


def _counterfactuals(pair):
    inp, expected = pair['input'], pair['output']
    _, info = _analyze(inp)
    bg = info['background']
    # Geometry and palettes covary with the rule, not with hidden seed choices.
    for k in range(4):
        assert np.array_equal(solve(np.rot90(inp, k)), np.rot90(expected, k))
    permutation = np.array((9, 3, 5, 0, 7, 1, 8, 4, 6, 2), dtype=np.int8)
    assert np.array_equal(solve(permutation[inp]), permutation[expected])
    for obj in info['objects']:
        if obj['side'] is None:
            continue
        b = obj['canonical']
        marks = b[obj['mask'] & (b != info['filler'])]
        if len(marks) != 3:
            continue
        def check_address(candidate):
            changed = inp.copy()
            source_block = np.rot90(candidate, -obj['options'][0])
            r0, c0 = obj['origin']
            changed[r0:r0+source_block.shape[0], c0:c0+source_block.shape[1]] = source_block
            solved, altered_info = _analyze(changed)
            assert [o['side'] for o in altered_info['objects']] == [o['side'] for o in info['objects']]
            for old, new in zip(info['objects'], altered_info['objects']):
                if old['side'] is not None:
                    assert np.array_equal(old['target'], new['target'])
            assert np.array_equal(np.bincount(changed.ravel(), minlength=10),
                                  np.bincount(solved.ravel(), minlength=10))
        points = np.argwhere(obj['mask'] & (b != info['filler']))
        moved_marks = b.copy()
        moved_marks[points[:, 0], points[:, 1]] = np.roll(marks, 1)
        check_address(moved_marks)
        unmarked = obj['mask'] & (b == info['filler'])
        unmarked[-1] = False  # Connector convention remains visible.
        spare = tuple(np.argwhere(unmarked)[0])
        repeated = b.copy()
        repeated[spare] = marks[0]
        check_address(repeated)
        extra = next(c for c in range(10) if c not in set(info['labels']) | {bg, info['filler']})
        with_extra = b.copy()
        with_extra[spare] = extra
        check_address(with_extra)
        # Same silhouette, number of marks, and connector; remove one identity.
        changed = inp.copy()
        source = obj['coords']
        marked = [tuple(p) for p in source if int(inp[tuple(p)]) in info['labels']]
        changed[marked[0]] = extra
        out, changed_info = _analyze(changed)
        assert len(changed_info['sides']) == len(info['sides'])-1
        rr, cc = obj['coords'].T
        assert np.array_equal(out[rr, cc], changed[rr, cc])
        tr, tc = obj['target'].T
        assert np.all(out[tr, tc] == bg)
        # Erasure without A's assembly is a concrete distinct answer.
        ablated = inp.copy()
        for p in info['objects']:
            if p['side'] is not None:
                ar, ac = p['coords'].T
                ablated[ar, ac] = bg
        assert not np.array_equal(ablated, expected)
        return


def _hub_swap_check():
    # Roomy, controlled counterfactual: records unchanged, address bars swapped.
    a = np.zeros((24, 24), dtype=np.int8)
    labels = (1, 2, 3, 4)
    _draw_hub(a, 10, 10, labels)
    m = np.array(((1, 1, 1, 1), (1, 1, 1, 1), (0, 1, 1, 0)), dtype=bool)
    rng = np.random.default_rng(123)
    for side, origin in enumerate(((0, 0), (0, 19), (19, 0), (19, 19))):
        b = _paint(rng, m, [x for s, x in enumerate(labels) if s != side], 0, 5)
        b = np.rot90(b, side)
        r, c = origin
        a[r:r+b.shape[0], c:c+b.shape[1]] = b
    before, first = _analyze(a)
    altered = a.copy()
    _draw_hub(altered, 10, 10, (2, 1, 3, 4))
    after, second = _analyze(altered)
    assert [o['intersection'] for o in first['objects']] == [o['intersection'] for o in second['objects']]
    assert [o['side'] for o in first['objects']] != [o['side'] for o in second['objects']]
    before[10:14, 10:14] = after[10:14, 10:14]
    assert not np.array_equal(before, after)


def self_test():
    """Bounded 42-seed structural, causal, covariance, and contract checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    shapes, masks, geometries, palettes, subsets = set(), set(), set(), set(), set()
    counts, label_counts, edge_hits = set(), set(), 0
    diagonal, asymmetric = 0, 0
    for seed, puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        if seed < 3:
            assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input']
                _, info = _analyze(inp)
                shapes.add(inp.shape)
                palettes.add((info['background'], info['filler'], info['labels']))
                subsets.add(_bits(info['sides']))
                counts.add((len(info['sides']), len(info['objects'])-len(info['sides'])))
                for obj in info['objects']:
                    mask = obj['mask']
                    masks.add(_key(mask.astype(np.int8)))
                    neck = np.zeros(mask.shape[1], dtype=bool)
                    neck[mask.shape[1]//2-1:mask.shape[1]//2+1] = True
                    stem = 0
                    for row in mask[::-1]:
                        if not np.array_equal(row, neck):
                            break
                        stem += 1
                    geometries.add((mask.shape[0], mask.shape[1], stem))
                    label_counts.add(len(obj['intersection']))
                    asymmetric += not np.array_equal(mask, np.fliplr(mask))
                    # Count four-connected components without adding another solver.
                    cells = set(map(tuple, np.argwhere(mask)))
                    queue = [cells.pop()]
                    for r, c in queue:
                        for point in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                            if point in cells:
                                cells.remove(point)
                                queue.append(point)
                    diagonal += bool(cells)
                foreground = np.argwhere(pair['output'] != info['background'])
                edge_hits += bool(np.any((foreground == 0) | (foreground == np.array(inp.shape)-1)))
                if split == 'train':
                    _counterfactuals(pair)
    assert len(shapes) >= 40 and len(masks) >= 60 and len(geometries) >= 25
    assert len(palettes) >= 100 and len(subsets) >= 12 and len(counts) >= 12
    assert label_counts == {0, 1, 2, 3, 4}
    assert diagonal >= 10 and asymmetric >= 10 and edge_hits >= 20
    _hub_swap_check()
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(2026+a*10+b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.5, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('Corrupt output accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds, all legal counts, causal and variation checks')
