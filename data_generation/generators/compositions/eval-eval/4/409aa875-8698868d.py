"""Cavity-powered stencils: topology supplies distance; arrows supply direction.

Only standard library and NumPy are used. No files, labels, seed, or generation
state are consulted by solve. Diagnostics are streamed, never archived in bulk.
"""
from __future__ import annotations

import hashlib
import numpy as np


FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAGONAL = ((-1, -1), (-1, 1), (1, -1), (1, 1))
EIGHT = FOUR + DIAGONAL
HOST_SEEDS = (0, 742323826, 276808768, 646988347, 358994001, 638467452,
              993723441, 391217716, 772409480, 898675089, 66121621, 970923171,
              833823800, 983258279, 617529321, 715547749, 263733308, 602088710,
              662448370, 468356347, 904995065, 756188097, 751063045, 271125828,
              946166325, 958042350, 208146095, 5937605, 502816992, 167388066,
              449461708, 417066389, 575110749, 374725289, 8673590, 367360300,
              284900273, 632793637, 728497539, 995090361, 111114310, 461947963)


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _components(mask, diagonal=False):
    pending = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while pending:
        start = min(pending)
        pending.remove(start)
        cells, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in EIGHT if diagonal else FOUR:
                p = (r + dr, c + dc)
                if p in pending:
                    pending.remove(p)
                    cells.append(p)
                    stack.append(p)
        result.append(cells)
    return result


def _arrow(tip, direction):
    r, c = tip
    dr, dc = direction
    if dr == 0:
        wings = ((-1, -dc), (1, -dc))
    elif dc == 0:
        wings = ((-dr, -1), (-dr, 1))
    else:
        wings = ((-dr, 0), (0, -dc))
    return ((r, c),) + tuple((r + a, c + b) for a, b in wings)


def _arrow_options(holes, direction):
    h, w = holes.shape
    options = []
    for r in range(h):
        for c in range(w):
            cells = _arrow((r, c), direction)
            if all(0 <= a < h and 0 <= b < w and not holes[a, b] for a, b in cells):
                options.append(cells)
    return options


def _valid_grid(value):
    grid = np.asarray(value)
    assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((0 <= grid) & (grid <= 9))
    return grid.astype(np.int8, copy=False)


def _read(input_grid):
    grid = _valid_grid(input_grid)
    bg = int(grid[0, 0])
    assert np.all(grid[0] == bg) and np.all(grid[-1] == bg)
    assert np.all(grid[:, 0] == bg) and np.all(grid[:, -1] == bg), 'blank input frame required'
    objects, roles = [], set()
    for cells in _components(grid != bg):
        rs, cs = zip(*cells)
        r, c, h, w = min(rs), min(cs), max(rs) - min(rs) + 1, max(cs) - min(cs) + 1
        assert 5 <= h <= 8 and 5 <= w <= 8
        box = grid[r:r + h, c:c + w]
        assert np.all(box[0] != bg) and np.all(box[-1] != bg)
        assert np.all(box[:, 0] != bg) and np.all(box[:, -1] != bg)
        colors, counts = np.unique(box[box != bg], return_counts=True)
        assert len(colors) == 2 and np.count_nonzero(counts == 3) == 1
        marker = int(colors[np.argmin(counts)])
        ink = int(colors[np.argmax(counts)])
        marker_cells = set(map(tuple, np.argwhere(box == marker).tolist()))
        matches = [(tip, d) for tip in sorted(marker_cells) for d in EIGHT
                   if set(_arrow(tip, d)) == marker_cells]
        assert len(matches) == 1, 'each stamp has exactly one recognizable arrow'
        tip, direction = matches[0]
        holes = box == bg
        k = len(_components(holes))
        dr, dc = direction
        rr, cc = r + k * dr, c + k * dc
        assert 0 <= rr <= grid.shape[0] - h and 0 <= cc <= grid.shape[1] - w
        assert len(cells) == int((~holes).sum()), 'stamp ink must be four-connected'
        objects.append({'r': r, 'c': c, 'holes': holes.copy(), 'k': k,
                        'direction': direction, 'arrow': _arrow(tip, direction)})
        roles.add((ink, marker))
    assert 1 <= len(objects) <= 6 and len(roles) == 1
    boxes = [(o['r'], o['c'], *o['holes'].shape) for o in objects]
    assert all(_separate(a, b) for i, a in enumerate(boxes) for b in boxes[:i])
    ink, marker = roles.pop()
    assert len({bg, ink, marker}) == 3
    return grid, bg, ink, marker, objects


def _coverage(objects, shape, distance=None, solid=False):
    count = np.zeros(shape, dtype=np.uint8)
    for obj in objects:
        mask = np.ones_like(obj['holes']) if solid else ~obj['holes']
        k = obj['k'] if distance is None else distance(obj)
        dr, dc = obj['direction']
        r, c = obj['r'] + k * dr, obj['c'] + k * dc
        h, w = mask.shape
        if not (0 <= r <= shape[0] - h and 0 <= c <= shape[1] - w):
            return None
        count[r:r + h, c:c + w] += mask
    return count


def _paint(count, bg, ink, mode='parity'):
    if mode == 'parity':
        keep = count % 2 == 1
    elif mode == 'union':
        keep = count > 0
    elif mode == 'single':
        keep = count == 1
    else:
        raise ValueError('unknown paint mode')
    return np.where(keep, ink, bg).astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid, bg, ink, _, objects = _read(input_grid)
    return _paint(_coverage(objects, grid.shape), bg, ink)


def _seed_holes(rng, h, w, k):
    """Independent hole seeds with connected ink; a geometric fallback is finite."""
    inner = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    for _ in range(18):
        holes = np.zeros((h, w), dtype=bool)
        for i in rng.permutation(len(inner)):
            r, c = inner[int(i)]
            if any(holes[r + dr, c + dc] for dr, dc in FOUR):
                continue
            holes[r, c] = True
            if len(_components(~holes)) != 1:
                holes[r, c] = False
            if holes.sum() == k:
                return holes
    # Alternating cells along the inner perimeter leave connected ink. The
    # complete 3x3 checkerboard is also valid, and realizes five cavities.
    candidates = []
    for parity in (0, 1):
        for boundary_only in (True, False):
            mask = np.zeros((h, w), dtype=bool)
            for r, c in inner:
                if ((r + c) % 2 == parity and
                    (not boundary_only or r in (1, h - 2) or c in (1, w - 2))):
                    mask[r, c] = True
            if mask.sum() >= k and len(_components(~mask)) == 1:
                candidates.append(mask)
    assert candidates, ('no seed capacity', h, w, k)
    cells = np.argwhere(_pick(rng, candidates))
    holes = np.zeros((h, w), dtype=bool)
    for i in rng.permutation(len(cells))[:k]:
        holes[tuple(cells[int(i)])] = True
    return holes


def _capacity(h, w):
    return 5 if (h, w) == (5, 5) else min(8, h + w - 6)


def _stamp(rng, h, w, k, direction, witness=None):
    if k == 0:
        holes = np.zeros((h, w), dtype=bool)
    elif witness == 'connected_area':
        full = np.zeros((h, w), dtype=bool)
        full[1:-1, 1:-1] = True
        options = []
        for cells in _arrow_options(np.zeros_like(full), direction):
            trial = full.copy()
            for p in cells:
                trial[p] = False
            if trial.sum() >= 8 and len(_components(trial)) == len(_components(~trial)) == 1:
                options.append((trial, cells))
        assert options
        holes, arrow = _pick(rng, options)
        return {'holes': holes, 'k': k, 'direction': direction, 'arrow': arrow}
    elif witness == 'diagonal_contact':
        holes = np.zeros((h, w), dtype=bool)
        holes[1, 1] = holes[2, 2] = holes[2, 3] = True
        if rng.integers(2):
            holes = np.flipud(holes).copy()
        if rng.integers(2):
            holes = np.fliplr(holes).copy()
    else:
        holes = _seed_holes(rng, h, w, k)
    options = _arrow_options(holes, direction)
    assert options, ('no marker position', h, w, k, direction)
    arrow = _pick(rng, options)
    # Grow seeds into unequal, irregular connected cavities. Counts cannot
    # change, the ink stays connected, and arrow cells are reserved beforehand.
    if k and witness is None:
        target = int(rng.integers(k, (h - 2) * (w - 2) + 1))
        inner = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
        for _ in range(3):
            for i in rng.permutation(len(inner)):
                if holes.sum() >= target:
                    break
                r, c = inner[int(i)]
                if holes[r, c] or (r, c) in arrow or not any(holes[r + a, c + b] for a, b in FOUR):
                    continue
                holes[r, c] = True
                if len(_components(holes)) != k or len(_components(~holes)) != 1:
                    holes[r, c] = False
    return {'holes': holes, 'k': k, 'direction': direction, 'arrow': arrow}


def _separate(a, b):
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return ar + ah < br or br + bh < ar or ac + aw < bc or bc + bw < ac


def _assemble(rng, objects):
    """Direct construction from masks, independently reread by solve/validator."""
    extents = []
    for obj in objects:
        r, c = obj['r'], obj['c']
        h, w = obj['holes'].shape
        dr, dc = obj['direction']
        for a, b in ((r, c), (r + obj['k'] * dr, c + obj['k'] * dc)):
            extents.append((a, b, a + h, b + w))
    r0, c0 = min(e[0] for e in extents), min(e[1] for e in extents)
    h = max(e[2] for e in extents) - r0
    w = max(e[3] for e in extents) - c0
    assert h <= 28 and w <= 28
    pads = []
    for size in (h, w):
        total = int(rng.integers(2, min(6, 30 - size) + 1))
        before = int(rng.integers(1, total))
        pads.append((before, total - before))
    bg, ink, marker = map(int, rng.choice(10, size=3, replace=False))
    inp = np.full((h + sum(pads[0]), w + sum(pads[1])), bg, dtype=np.int8)
    moved = []
    for obj in objects:
        obj = dict(obj, r=obj['r'] - r0 + pads[0][0], c=obj['c'] - c0 + pads[1][0])
        r, c = obj['r'], obj['c']
        hh, ww = obj['holes'].shape
        box = np.where(obj['holes'], bg, ink).astype(np.int8)
        for p in obj['arrow']:
            box[p] = marker
        inp[r:r + hh, c:c + ww] = box
        moved.append(obj)
    out = _paint(_coverage(moved, inp.shape), bg, ink)
    return {'input': inp, 'output': out}


def _turn_pair(pair, rng):
    # Rotating/reflection acts on both the masks and their directions visibly.
    turns, reflect = int(rng.integers(4)), bool(rng.integers(2))
    return {key: np.rot90(np.fliplr(g) if reflect else g, turns).copy()
            for key, g in pair.items()}


def _teaching_pair(rng, triple=False):
    objects = []
    if triple:
        # Three separated sources converge to a common top-left destination.
        # Different sizes prevent this from being three identical solid masks.
        sizes = list(map(int, rng.choice((5, 6, 7), size=3)))
        if len(set(sizes)) == 1:
            sizes[-1] = 5 if sizes[0] != 5 else 6
        directions = [DIAGONAL[int(i)] for i in rng.permutation(4)[:3]]
        for s, d in zip(sizes, directions):
            k = s if s < 7 else int(rng.integers(7, 9))
            obj = _stamp(rng, s, s, k, d)
            objects.append(dict(obj, r=-k * d[0], c=-k * d[1]))
    else:
        ah, aw = map(int, rng.integers(6, 9, size=2))
        bh, bw = map(int, rng.integers(6, 9, size=2))
        a = _stamp(rng, ah, aw, 1, (0, 1), 'connected_area')
        b = _stamp(rng, bh, bw, 2, (0, -1), 'diagonal_contact')
        objects = [dict(a, r=0, c=0), dict(b, r=int(rng.integers(-1, 2)), c=aw + 1)]
        zh, zw = map(int, rng.integers(5, 9, size=2))
        z = _stamp(rng, zh, zw, 0, _pick(rng, ((0, -1), (0, 1))))
        objects.append(dict(z, r=max(ah, bh + objects[1]['r']) + 2,
                            c=int(rng.integers(0, aw + bw + 2 - zw))))
    return _turn_pair(_assemble(rng, objects), rng)


def _random_pair(rng, mixed):
    for _ in range(80):
        n = int(rng.integers(3 if mixed else 2, 6))
        pool = EIGHT if mixed else _pick(rng, (FOUR, DIAGONAL))
        directions = [_pick(rng, pool) for _ in range(n)]
        if mixed:
            directions[0], directions[1] = _pick(rng, FOUR), _pick(rng, DIAGONAL)
        objects, boxes = [], []
        for direction in directions:
            h, w = map(int, rng.integers(5, 9, size=2))
            k = int(rng.integers(0, _capacity(h, w) + 1))
            obj = _stamp(rng, h, w, k, direction)
            dr, dc = direction
            for _ in range(90):
                r = int(rng.integers(1 + max(0, -k * dr), 30 - h - max(0, k * dr)))
                c = int(rng.integers(1 + max(0, -k * dc), 30 - w - max(0, k * dc)))
                if all(_separate((r, c, h, w), b) for b in boxes):
                    objects.append(dict(obj, r=r, c=c))
                    boxes.append((r, c, h, w))
                    break
            else:
                break
        if len(objects) != n:
            continue
        count = _coverage(objects, (30, 30))
        if count is not None and np.any(count == 2) and np.any(count == 1):
            return _assemble(rng, objects)
    # Guaranteed mixed-class partial collision, with a distant stationary stamp.
    # The fallback obeys the same general rule and is not needed by solve.
    a = _stamp(rng, 6, 6, 3, (0, 1))
    b = _stamp(rng, 6, 6, 4, (-1, -1) if mixed else (0, -1))
    z = _stamp(rng, 5, 5, 0, (0, -1))
    return _turn_pair(_assemble(rng, [dict(a, r=0, c=0), dict(b, r=0, c=7),
                                    dict(z, r=9, c=2)]), rng)


def _resolve_counts(rng, num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                  not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
            raise ValueError('counts must be integers in their legal ranges')
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1 if num_train is not None else 5))
    if num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _fingerprint(grid):
    g = np.asarray(grid, dtype=np.int8)
    return g.shape, hashlib.sha256(g.tobytes()).hexdigest()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    nt, ne = _resolve_counts(rng, num_train, num_test)
    puzzle = {'train': [], 'test': []}
    used = set()
    for split, count in (('train', nt), ('test', ne)):
        for i in range(count):
            for _ in range(20):
                pair = (_teaching_pair(rng, triple=bool(i)) if split == 'train' and i < 2
                        else _random_pair(rng, mixed=split == 'test'))
                key = _fingerprint(pair['input'])
                if key not in used:
                    puzzle[split].append(pair)
                    used.add(key)
                    break
            else:
                raise RuntimeError('distinct-input retry bound exhausted')
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(nt)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _analysis(inp):
    grid, bg, ink, marker, objects = _read(inp)
    cover = _coverage(objects, grid.shape)
    answer = _paint(cover, bg, ink)
    def differs(distance=None, solid=False, mode='parity'):
        alternative = _coverage(objects, grid.shape, distance, solid)
        return alternative is None or not np.array_equal(_paint(alternative, bg, ink, mode), answer)
    areas = [(o['k'], int(o['holes'].sum())) for o in objects]
    classes = {int(all(o['direction'])) for o in objects}
    return {'objects': objects, 'shape': grid.shape, 'roles': (bg, ink, marker),
            'mixed': len(classes) == 2, 'zero': any(k == 0 for k, _ in areas),
            'area_inversion': any(k < l and a > b for k, a in areas for l, b in areas),
            'connectivity': differs(lambda o: len(_components(o['holes'], diagonal=True))),
            'no_direction': differs(lambda o: 0),
            'area_distance': differs(lambda o: int(o['holes'].sum())),
            'union': differs(mode='union'), 'erase_all_collisions': differs(mode='single'),
            'solid': differs(solid=True), 'double': int(np.count_nonzero(cover == 2)),
            'triple': int(np.count_nonzero(cover == 3)),
            'constant_models': {d for d in range(30) if not differs(lambda o, d=d: d)}}


def _focused_tests():
    rng = np.random.default_rng(410)
    # Higher-order cancellation and the six-stamp bound, using legal separated
    # sources on a ring. All six translated upper-left corners coincide.
    objects = []
    for d in EIGHT[:6]:
        o = _stamp(rng, 7, 7, 8, d)
        objects.append(dict(o, r=-8 * d[0], c=-8 * d[1]))
    for n in (4, 5, 6):
        pair = _assemble(rng, objects[:n])
        grid, bg, ink, _, parsed = _read(pair['input'])
        counts = _coverage(parsed, grid.shape)
        assert counts.max() == n
        assert np.all(solve(grid)[counts == n] == (ink if n % 2 else bg))
        assert np.array_equal(solve(grid), pair['output'])
    # A destination may touch the canvas edge even though the input frame is
    # blank. A feasible count beyond the sampled 0..8 range uses the same rule.
    for s, k in ((7, 1), (8, 9)):
        o = _stamp(rng, s, s, k, (0, 1))
        inp = np.zeros((s + 2, s + k + 1), dtype=np.int8)
        box = np.where(o['holes'], 0, 4).astype(np.int8)
        for p in o['arrow']:
            box[p] = 7
        inp[1:s + 1, 1:s + 1] = box
        expected = np.zeros_like(inp)
        expected[1:s + 1, 1 + k:s + 1 + k] = np.where(o['holes'], 0, 4)
        assert np.array_equal(solve(inp), expected) and np.any(expected[:, -1] == 4)
        # Arrow location is annotation only, not a paste anchor.
        clean = inp.copy()
        for p in o['arrow']:
            clean[1 + p[0], 1 + p[1]] = 4
        alternative = _arrow_options(o['holes'], (0, 1))[-1]
        for p in alternative:
            clean[1 + p[0], 1 + p[1]] = 7
        assert np.array_equal(solve(clean), expected)
        lookup = np.array([8, 3, 0, 5, 1, 6, 4, 2, 9, 7], dtype=np.int8)
        assert np.array_equal(solve(lookup[inp]), lookup[expected])


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    used, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _valid_grid(pair['input']), _valid_grid(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = _fingerprint(inp)
            assert key not in used
            used.add(key)
            assert np.array_equal(solve(inp.copy()), out), 'wrong movement, stencil, or parity'
            a = _analysis(inp)
            assert set(map(int, np.unique(out))) <= set(a['roles'][:2])
            analyses[split].append(a)
    tr = analyses['train']
    for feature in ('zero', 'area_inversion', 'connectivity', 'no_direction',
                    'area_distance', 'union', 'erase_all_collisions', 'solid'):
        assert any(a[feature] for a in tr), 'missing training witness: ' + feature
    assert not set.intersection(*(a['constant_models'] for a in tr)), 'fixed distance shortcut survives'
    assert all(not a['mixed'] for a in tr), 'training curriculum uses one direction class per scene'
    assert all(a['mixed'] for a in analyses['test']), 'test combines the two demonstrated direction classes'


def self_test():
    """42 seeds streamed; no retained grids or logs. Returns compact measurements."""
    _focused_tests()
    shapes, geometries, directions, cavity_shapes, roles, signatures = set(), set(), set(), set(), set(), set()
    counts, strides, sizes, totals = set(), set(), set(), {'pairs': 0, 'double_pairs': 0, 'triple_pairs': 0}
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signature = []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = _analysis(pair['input'])
                signature.append(_fingerprint(pair['input']))
                shapes.add(a['shape'])
                roles.add(a['roles'])
                counts.add(len(a['objects']))
                totals['pairs'] += 1
                totals['double_pairs'] += a['double'] > 0
                totals['triple_pairs'] += a['triple'] > 0
                for obj in a['objects']:
                    directions.add(obj['direction'])
                    strides.add(obj['k'])
                    sizes.add(obj['holes'].shape)
                    geometries.add((obj['holes'].shape, obj['direction'], obj['k']))
                    cavity_shapes.add(_fingerprint(obj['holes']))
                assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
        signatures.add(tuple(signature))
    assert len(signatures) == 42 and len(shapes) >= 30 and len(geometries) >= 60
    assert directions == set(EIGHT) and strides == set(range(9)) and len(counts) >= 3
    assert len(cavity_shapes) >= 100 and len(sizes) >= 10 and len(roles) >= 50
    assert totals['triple_pairs'] >= 42 and totals['double_pairs'] >= 84
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(31, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.0, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    first, second = generate_puzzle(17, 2, 1), generate_puzzle(17, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in first for a, b in zip(first[s], second[s])
               for k in ('input', 'output'))
    # Rotation/reflection preserve the whole rule and curriculum.
    rotated = {s: [{k: np.rot90(v).copy() for k, v in p.items()} for p in first[s]] for s in first}
    validate_generated_puzzle(rotated)
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    return dict(totals, seeds=42, distinct_puzzles=len(signatures), input_shapes=len(shapes),
                stamp_geometries=len(geometries), cavity_masks=len(cavity_shapes),
                rectangle_shapes=len(sizes), palette_role_triples=len(roles),
                object_counts=sorted(counts), strides=sorted(strides), directions=len(directions))


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
