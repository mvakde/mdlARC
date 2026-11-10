"""Fit reflection-cleaned stamps to cavity-count sockets by a unique bijection.

No files, parent generators, seed state, or training labels enter solve().
All searches and diagnostics are bounded; generation writes nothing to disk.
"""
from __future__ import annotations

from itertools import product
import hashlib
import numpy as np

_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask, diagonal=False):
    cells = set(map(tuple, np.argwhere(mask).tolist()))
    steps = _STEPS + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else _STEPS
    result = []
    while cells:
        first = min(cells)
        cells.remove(first)
        stack, component = [first], []
        while stack:
            r, c = stack.pop()
            component.append((r, c))
            for dr, dc in steps:
                p = (r + dr, c + dc)
                if p in cells:
                    cells.remove(p)
                    stack.append(p)
        result.append(component)
    return result


def _rim(a):
    return np.concatenate((a[0], a[-1], a[1:-1, 0], a[1:-1, -1]))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError('expected a 1..30 by 1..30 integer color grid')
    return a


def _branches(raw):
    """V/H are centered on the complete rectangular rim, including half axes."""
    result = []
    for direction, reflected in (('V', raw[:, ::-1]), ('H', raw[::-1, :])):
        keep = raw & reflected
        if not any(np.array_equal(keep, b['keep']) for b in result):
            result.append({'direction': direction, 'keep': keep,
                           'count': len(_components(~keep))})
    return result


def _assignments(options, demands=None):
    """At most 2**4 branches. Identical V/H results were already collapsed."""
    result = {}
    for selected in product(*options):
        counts = tuple(sorted(b['count'] for b in selected))
        if len(set(counts)) != len(counts):
            continue
        if demands is None or counts == demands:
            result.setdefault(counts, []).append(selected)
    return result


def _objects(grid):
    # The blank outer rim identifies background. Flood only background connected
    # to it; its complement groups enclosed islands with their rectangular rim.
    border = _rim(grid)
    if not np.all(border == border[0]):
        raise ValueError('the canvas must have a uniform blank outer rim')
    bg = int(border[0])
    h, w = grid.shape
    exterior = np.zeros((h, w), dtype=bool)
    exterior[0, 0] = True
    stack = [(0, 0)]
    while stack:
        r, c = stack.pop()
        for dr, dc in _STEPS:
            rr, cc = r + dr, c + dc
            if (0 <= rr < h and 0 <= cc < w and not exterior[rr, cc]
                    and grid[rr, cc] == bg):
                exterior[rr, cc] = True
                stack.append((rr, cc))
    objects = []
    for component in _components(~exterior):
        rs, cs = zip(*component)
        r, c, hh, ww = min(rs), min(cs), max(rs) - min(rs) + 1, max(cs) - min(cs) + 1
        if len(component) != hh * ww:
            raise ValueError('each object must enclose a full rectangle')
        obj = grid[r:r+hh, c:c+ww]
        color = int(obj[0, 0])
        if color == bg or not np.all(_rim(obj) == color):
            raise ValueError('object rim must be complete and monochromatic')
        if not set(map(int, np.unique(obj))) <= {bg, color}:
            raise ValueError('objects are binary')
        objects.append({'box': (r, c, hh, ww), 'color': color, 'grid': obj})
    shapes = sorted({o['grid'].shape for o in objects}, key=lambda s: s[0] * s[1])
    if len(shapes) != 2 or shapes[1] != (shapes[0][0] + 2, shapes[0][1] + 2):
        raise ValueError('socket dimensions must exceed stamp dimensions by two')
    stamps = [o for o in objects if o['grid'].shape == shapes[0]]
    sockets = [o for o in objects if o['grid'].shape == shapes[1]]
    if not (2 <= len(stamps) <= 4 and len(stamps) == len(sockets)):
        raise ValueError('need two to four stamps and equally many sockets')
    if not all(5 <= d <= 8 for d in shapes[0]):
        raise ValueError('stamp dimensions must be 5..8')
    for i, a in enumerate(objects):
        r, c, hh, ww = a['box']
        for b in objects[i+1:]:
            rr, cc, bh, bw = b['box']
            if not (r + hh < rr or rr + bh < r or c + ww < cc or cc + bw < c):
                raise ValueError('object boxes require a complete blank row or column gap')
    counts = []
    for socket in sockets:
        clue = socket['grid'][1:-1, 1:-1] == bg
        if np.any(clue[:-1] & clue[1:]) or np.any(clue[:, :-1] & clue[:, 1:]):
            raise ValueError('socket clues must be four-neighbor isolated')
        socket['count'] = int(clue.sum())
        if not 0 <= socket['count'] <= 8:
            raise ValueError('socket counts must be 0..8')
        counts.append(socket['count'])
    if len(set(counts)) != len(counts):
        raise ValueError('socket counts must be distinct')
    return bg, stamps, sockets


def _decode(input_grid):
    grid = _grid(input_grid)
    bg, stamps, sockets = _objects(grid)
    options = [_branches(stamp['grid'] != bg) for stamp in stamps]
    demands = tuple(sorted(s['count'] for s in sockets))
    solutions = _assignments(options, demands).get(demands, [])
    if len(solutions) != 1:
        raise ValueError('the complete stamp/socket assignment must be unique')
    selected = solutions[0]
    by_count = {b['count']: (stamp, b) for stamp, b in zip(stamps, selected)}
    output = np.full(grid.shape, bg, dtype=np.int8)
    for socket in sockets:
        stamp, branch = by_count[socket['count']]
        if stamp['color'] == socket['color']:
            raise ValueError('a fitted stamp must differ from its carrier')
        r, c, h, w = socket['box']
        output[r:r+h, c:c+w] = socket['color']
        output[r+1:r+h-1, c+1:c+w-1] = np.where(branch['keep'], stamp['color'], socket['color'])
    return output, (bg, stamps, sockets, options, selected)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only the visible boxes, reflection partners, and cavity-count bijection."""
    return _decode(input_grid)[0]


def _features(stamps, options, selected, demands):
    masks = [b['keep'] for b in selected]
    counts = [b['count'] for b in selected]
    areas = [int((~m).sum()) for m in masks]
    return {
        'coupled': any(sum(b['count'] in demands for b in bs) > 1 for bs in options),
        'nongreedy': any(any(int(b['keep'].sum()) > int(s['keep'].sum()) for b in bs)
                         for bs, s in zip(options, selected)),
        'area_inversion': any(k < l and a > b for k, a in zip(counts, areas)
                              for l, b in zip(counts, areas)),
        'connectivity': any(len(_components(~m, True)) < k for m, k in zip(masks, counts)),
        'topology_changed': any(len(_components(~a)) != k for a, k in zip(stamps, counts)),
        'both_deleted': all(np.any(raw != m) for raw, m in zip(stamps, masks)),
        'directions': {b['direction'] for bs, b in zip(options, selected) if len(bs) == 2},
    }


def _sample_raw(rng, h, w, witness=False):
    a = np.ones((h, w), dtype=bool)
    mode = 0 if witness else int(rng.integers(6))
    if mode <= 2:
        a[1:-1, 1:-1] = rng.random((h-2, w-2)) < rng.uniform(.55, .96)
    elif mode == 3:
        # Rectangular cavities and separating walls, then asymmetric additions.
        a[1:-1, 1:-1] = False
        if rng.random() < .7:
            for c in range(2, w-2, int(rng.integers(2, 4))):
                a[1:-1, c] = True
        if rng.random() < .7:
            for r in range(2, h-2, int(rng.integers(2, 4))):
                a[r, 1:-1] = True
        a[1:-1, 1:-1] |= rng.random((h-2, w-2)) < rng.uniform(.05, .30)
    elif mode == 4:
        # Irregular multi-cell empty regions, rather than only singleton holes.
        for _ in range(int(rng.integers(1, 5))):
            r, c = int(rng.integers(1, h-1)), int(rng.integers(1, w-1))
            for _ in range(int(rng.integers(1, h*w//2))):
                a[r, c] = False
                dr, dc = _STEPS[int(rng.integers(4))]
                r, c = int(np.clip(r+dr, 1, h-2)), int(np.clip(c+dc, 1, w-2))
    else:
        # Unchanged stamps are legitimate: solid, hollow, paired bars, four holes.
        kind = int(rng.integers(4))
        if kind == 1:
            a[1:-1, 1:-1] = False
        elif kind == 2:
            a[1:-1, 1] = a[1:-1, -2] = False
        elif kind == 3:
            for r in (1, h-2):
                for c in (1, w-2):
                    a[r, c] = False
    return a


def _fallback_family(h, w, n):
    # A constructive, new-family fixture; not a parent motif. The first stamp
    # offers {V:3,H:1}, the second {V:2,H:1}; demands {1,2} force H then V.
    rows = (('111111','101111','111111','110111','100001','111111'),
            ('111111','111111','110001','101111','111011','111111'))
    raw = [np.array([[x == '1' for x in row] for row in shape]) for shape in rows]
    # Symmetric row/column duplication preserves reflection and connectivity.
    h, w = (8 if h >= 7 else 6), (8 if w >= 7 else 6)
    if n >= 3:
        h = w = 6
    row_ix = [0, 1, 1, 2, 3, 4, 4, 5]
    col_ix = [0, 1, 2, 2, 3, 3, 4, 5]
    raw = [a[row_ix, :] if h == 8 else a for a in raw]
    raw = [a[:, col_ix] if w == 8 else a for a in raw]
    if n >= 3:
        raw.append(np.ones((h, w), dtype=bool))
    if n == 4:
        a = np.ones((h, w), dtype=bool)
        for r in (1, h-2):
            for c in (1, w-2):
                a[r, c] = False
        raw.append(a)
    options = [_branches(a) for a in raw]
    demands = tuple(sorted([1, 2] + ([0] if n >= 3 else []) + ([4] if n == 4 else [])))
    choices = _assignments(options, demands)[demands]
    assert len(choices) == 1
    return raw, options, choices[0]


def _make_family(rng, h, w, n, witness=False, coupled=False):
    for _ in range(1200 if witness else 160):
        raw = [_sample_raw(rng, h, w, witness) for _ in range(n)]
        options = [_branches(a) for a in raw]
        groups = _assignments(options)
        eligible = []
        for demands, choices in groups.items():
            if len(choices) != 1 or max(demands) > 8:
                continue
            selected = choices[0]
            f = _features(raw, options, selected, demands)
            if coupled and not f['coupled']:
                continue
            if witness and not (f['coupled'] and f['nongreedy'] and f['area_inversion']
                                and f['connectivity'] and f['both_deleted']
                                and f['topology_changed']
                                and f['directions'] == {'V', 'H'}):
                continue
            eligible.append(selected)
        if eligible:
            return raw, options, eligible[int(rng.integers(len(eligible))) ]
    return _fallback_family(h, w, n)


def _clues(rng, h, w, count):
    for _ in range(16):
        a = np.zeros((h, w), dtype=bool)
        for i in rng.permutation(h*w):
            r, c = divmod(int(i), w)
            if any(0 <= r+dr < h and 0 <= c+dc < w and a[r+dr, c+dc] for dr, dc in _STEPS):
                continue
            if int(a.sum()) == count:
                return a
            a[r, c] = True
        if int(a.sum()) == count:
            return a
    a = np.zeros((h, w), dtype=bool)
    available = [(r, c) for r in range(h) for c in range(w) if (r+c) % 2 == 0]
    for i in rng.permutation(len(available))[:count]:
        a[available[int(i)]] = True
    return a


def _origins(boxes, h, w):
    rr, cc = np.indices((29-h, 29-w))
    rr, cc = rr+1, cc+1
    free = np.ones(rr.shape, dtype=bool)
    for r, c, bh, bw in boxes:
        free &= (rr+h < r) | (r+bh < rr) | (cc+w < c) | (c+bw < cc)
    return np.argwhere(free) + 1


def _pack(rng, shapes, n):
    for _ in range(20):
        boxes = []
        for h, w in shapes:
            positions = _origins(boxes, h, w)
            if not len(positions):
                break
            r, c = map(int, positions[int(rng.integers(len(positions)))])
            boxes.append((r, c, h, w))
        if len(boxes) == len(shapes):
            return boxes
    # Proven shelf packing in at most 27x27 cells for n>=3, 21x19 for n=2.
    rows = [[0, n], [1, n+1]] if n == 2 else (
        [[0, 1, n], [2, n+1, n+2]] if n == 3 else
        [[0, 1, n], [2, 3, n+1], [n+2, n+3]])
    boxes = [None] * (2*n)
    top = 1
    for row in rows:
        left, height = 1, max(shapes[i][0] for i in row)
        for i in row:
            h, w = shapes[i]
            boxes[i] = (top, left, h, w)
            left += w+1
        top += height+1
    assert all(r+h <= 29 and c+w <= 29 for r, c, h, w in boxes)
    # Translate the packing; each object's visible orientation stays put.
    rh = max(r+h for r, c, h, w in boxes)
    cw = max(c+w for r, c, h, w in boxes)
    dr, dc = int(rng.integers(30-rh)), int(rng.integers(30-cw))
    return [(r+dr, c+dc, h, w) for r, c, h, w in boxes]


def _make_pair(rng, h, w, n, witness=False, coupled=False):
    raw, options, selected = _make_family(rng, h, w, n, witness, coupled)
    h, w = raw[0].shape
    bg = int(rng.integers(10))
    colors = list(map(int, rng.permutation([c for c in range(10) if c != bg])))
    pool = colors[:int(rng.integers(2, 10))]
    donors = [int(rng.choice(pool)) for _ in range(n)]
    carriers = [int(rng.choice([c for c in pool if c != donor])) for donor in donors]
    sockets, inserts = [], []
    for donor, carrier, branch in zip(donors, carriers, selected):
        socket = np.full((h+2, w+2), carrier, dtype=np.int8)
        socket[1:-1, 1:-1][_clues(rng, h, w, branch['count'])] = bg
        insert = np.full_like(socket, carrier)
        insert[1:-1, 1:-1] = np.where(branch['keep'], donor, carrier)
        sockets.append(socket)
        inserts.append(insert)
    # The spatial orders are independent; count, not order, joins the objects.
    order = list(map(int, rng.permutation(n)))
    sockets, inserts = [sockets[i] for i in order], [inserts[i] for i in order]
    shapes = [(h+2, w+2)]*n + [(h, w)]*n
    boxes = _pack(rng, shapes, n)
    inp = np.full((30, 30), bg, dtype=np.int8)
    out = inp.copy()
    objects = sockets + [np.where(a, color, bg).astype(np.int8) for a, color in zip(raw, donors)]
    for i, (obj, (r, c, hh, ww)) in enumerate(zip(objects, boxes)):
        inp[r:r+hh, c:c+ww] = obj
        if i < n:
            out[r:r+hh, c:c+ww] = inserts[i]
    # Independent blank margins, including occasional full 30x30 canvases.
    r0 = min(r for r, c, hh, ww in boxes)
    c0 = min(c for r, c, hh, ww in boxes)
    r1 = max(r+hh for r, c, hh, ww in boxes)
    c1 = max(c+ww for r, c, hh, ww in boxes)
    low_r, low_c = int(rng.integers(r0)), int(rng.integers(c0))
    high_r, high_c = int(rng.integers(r1+1, 31)), int(rng.integers(c1+1, 31))
    return {'input': inp[low_r:high_r, low_c:high_c].copy(),
            'output': out[low_r:high_r, low_c:high_c].copy()}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def _key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(('count', 'scale', 'aspect')))
    puzzle, used = {'train': [], 'test': []}, set()
    for split, length in (('train', num_train), ('test', num_test)):
        for i in range(length):
            witness = split == 'train' and i == 0
            if shift == 'count':
                n = 2 if split == 'train' else int(rng.integers(3, 5))
                h, w = map(int, rng.integers(5, 8, size=2))
            elif shift == 'scale':
                n = 2
                h, w = map(int, rng.integers(5, 7, size=2) if split == 'train' else rng.integers(7, 9, size=2))
            else:
                n = 2
                h, w = (int(rng.integers(5, 7)), int(rng.integers(7, 9)))
                if split == 'test':
                    h, w = w, h
            for _ in range(8):
                pair = _make_pair(rng, h, w, n, witness, coupled=(split == 'test' or witness))
                key = _key(pair['input'])
                if key not in used:
                    break
            else:
                # Any pair has >=2 colors. Ninety maps of these two colors
                # exceed all thirteen previous inputs, giving a finite fallback.
                present = list(map(int, np.unique(pair['input'])))
                source = present[:2] + [c for c in range(10) if c not in present[:2]]
                found = False
                for a in range(10):
                    for b in range(10):
                        if a == b:
                            continue
                        target = [a, b] + [c for c in range(10) if c not in (a, b)]
                        lut = np.empty(10, dtype=np.int8)
                        lut[source] = target
                        candidate = {k: lut[v] for k, v in pair.items()}
                        if _key(candidate['input']) not in used:
                            pair, found = candidate, True
                            break
                    if found:
                        break
                assert found
            used.add(_key(pair['input']))
            puzzle[split].append(pair)
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect(pair):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    expected, data = _decode(inp)
    assert np.array_equal(out, expected), 'incorrect cleanup, assignment, substitution, or transfer'
    bg, stamps, sockets, options, selected = data
    assert inp.shape == out.shape
    by_count = {s['count']: s for s in sockets}
    for stamp, branch in zip(stamps, selected):
        raw = stamp['grid'] != bg
        # Explicit coordinate reconstruction checks the vectorized reflection.
        h, w = raw.shape
        axis = branch['direction']
        retained = np.array([[bool(raw[r, c] and raw[r, w-1-c] if axis == 'V'
                                   else raw[r, c] and raw[h-1-r, c])
                              for c in range(w)] for r in range(h)])
        assert np.array_equal(retained, branch['keep'])
        socket = by_count[branch['count']]
        r, c, hh, ww = socket['box']
        patch = out[r+1:r+hh-1, c+1:c+ww-1]
        assert np.array_equal(patch == stamp['color'], retained)
        assert len(_components(patch == socket['color'])) == socket['count']
        sr, sc, sh, sw = stamp['box']
        assert np.all(out[sr:sr+sh, sc:sc+sw] == bg), 'donor was not consumed'
    raw = [s['grid'] != bg for s in stamps]
    demands = tuple(sorted(s['count'] for s in sockets))
    f = _features(raw, options, selected, demands)
    f.update({'shape': inp.shape, 'stamp_shape': raw[0].shape, 'n': len(stamps), 'bg': bg,
              'counts': demands, 'palette': tuple(map(int, np.unique(inp))),
              'nonpositional': any(not np.array_equal(s['grid'][1:-1, 1:-1] == bg,
                                       ~b['keep']) for b in selected for s in sockets
                                   if b['count'] == s['count']),
              'deleted': sum(int((a & ~b['keep']).sum()) for a, b in zip(raw, selected)),
              'clean': sum(np.array_equal(a, b['keep']) for a, b in zip(raw, selected)),
              'cavity_sizes': [len(c) for b in selected for c in _components(~b['keep'])],
              'parities': {(0 if b['direction'] == 'V' else 1,
                            (a.shape[1] if b['direction'] == 'V' else a.shape[0]) % 2)
                           for a, b in zip(raw, selected) if len(_branches(a)) == 2}})
    return f


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            key = _key(_grid(pair['input']))
            assert key not in seen, 'duplicate input'
            seen.add(key)
            f = _inspect(pair)
            if split == 'train':
                training.append(f)
    # Enforce evidence, not a hidden output convention or fixed slot schedule.
    for name in ('coupled', 'nongreedy', 'area_inversion', 'connectivity',
                 'topology_changed', 'nonpositional'):
        assert any(f[name] for f in training), 'training lacks ' + name + ' witness'
    assert set.union(*(f['directions'] for f in training)) == {'V', 'H'}
    assert any(f['both_deleted'] for f in training)
    assert max(n for f in training for n in f['cavity_sizes']) > 1


def self_test():
    """Stream 42 seeds; retain small feature sets/hashes, never full trace payloads."""
    shapes, stamp_shapes, counts, palettes, backgrounds, hashes, normalized = (set() for _ in range(7))
    directions, parities, object_counts, sizes = set(), set(), set(), set()
    totals = {'pairs': 0, 'coupled': 0, 'nongreedy': 0, 'clean': 0,
              'connectivity': 0, 'topology_changed': 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                assert np.array_equal(solve(np.asarray(pair['input']).copy()), pair['output'])
                f = _inspect(pair)
                shapes.add(f['shape']); stamp_shapes.add(f['stamp_shape']); counts.add(f['counts'])
                palettes.add(f['palette']); backgrounds.add(f['bg']); object_counts.add(f['n'])
                directions |= f['directions']; parities |= f['parities']; sizes.update(f['cavity_sizes'])
                totals['pairs'] += 1
                for name in ('coupled', 'nongreedy', 'clean', 'connectivity', 'topology_changed'):
                    totals[name] += int(f[name])
                digest.update(bytes(pair['input'].shape)); digest.update(pair['input'].tobytes())
                roles = {}; a = pair['input']
                norm = bytes(roles.setdefault(int(c), len(roles)) for c in a.flat)
                normalized.add(hashlib.sha256(bytes(a.shape)+norm).digest())
        hashes.add(digest.digest())
        if seed < 2:
            other = generate_puzzle(seed)
            assert all(np.array_equal(a[k], b[k]) for split in puzzle
                       for a, b in zip(puzzle[split], other[split]) for k in a)
    assert len(hashes) == 42 and len(normalized) >= 190
    assert len(shapes) >= 25 and len(stamp_shapes) >= 12 and len(counts) >= 12
    assert backgrounds == set(range(10)) and len(palettes) >= 40
    assert object_counts == {2, 3, 4} and directions == {'V', 'H'}
    assert parities == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert max(sizes) >= 12 and totals['clean'] > 0 and totals['coupled'] >= 84
    for h, w, n in ((6, 6, 2), (8, 6, 2), (6, 8, 2), (8, 8, 2), (6, 6, 3), (6, 6, 4)):
        raw, options, selected = _fallback_family(h, w, n)
        f = _features(raw, options, selected, tuple(sorted(b['count'] for b in selected)))
        assert f['coupled'] and f['nongreedy'] and f['directions'] == {'V', 'H'}
        assert f['area_inversion'] and f['connectivity'] and f['topology_changed']
        if n == 2:
            assert f['both_deleted']
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(123+a*10+b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    broken = generate_puzzle(902, 2, 1)
    pair = broken['test'][0]
    _, (_, _, sockets, _, _) = _decode(pair['input'])
    socket = sockets[0]
    r, c, h, w = socket['box']
    # Erase a retained corner of the inserted stamp to its carrier: all schema,
    # color, bounding-box, and unchanged-background checks would still pass.
    pair['output'][r+1, c+1] = socket['color']
    try:
        validate_generated_puzzle(broken)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('corrupt output accepted')
    # Identical V/H masks are one interpretation; different masks with the same
    # count are two, and must not acquire an arbitrary direction tie-break.
    raw = np.ones((6, 6), bool); raw[1, 1] = False
    options = _branches(raw)
    assert len(options) == 2 and options[0]['count'] == options[1]['count']
    solid = _branches(np.ones((6, 6), bool))
    assert len(_assignments([options, solid], (0, options[0]['count']))[(0, options[0]['count'])]) == 2
    return {'seeds': 42, 'pairs': totals['pairs'], 'input_shapes': len(shapes),
            'stamp_shapes': len(stamp_shapes), 'count_sets': len(counts),
            'palettes': len(palettes), 'color_normalized_inputs': len(normalized),
            'object_counts': sorted(object_counts), 'largest_cavity': max(sizes),
            'features': totals}
