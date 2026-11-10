"""Registered stencil collisions: macro completion and simultaneous micro stamping.

Only NumPy and the Python standard library are used. Every solving convention is
fixed here or encoded by a visible lattice, stencil, or two-color key.
"""
from itertools import combinations

import numpy as np


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= x <= 30 for x in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _runs(indices):
    groups = []
    for x in map(int, indices):
        if groups and x == groups[-1][1] + 1:
            groups[-1] = (groups[-1][0], x)
        else:
            groups.append((x, x))
    return groups


def _box(cells):
    a = np.asarray(cells)
    return (int(a[:, 0].min()), int(a[:, 0].max()) + 1,
            int(a[:, 1].min()), int(a[:, 1].max()) + 1)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        todo, cells = [min(remaining)], []
        remaining.remove(todo[0])
        while todo:
            r, c = todo.pop()
            cells.append((int(r), int(c)))
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in remaining:
                    remaining.remove(p)
                    todo.append(p)
        result.append(cells)
    return result


def _origins(mask, shape, supplied):
    n = mask.shape[0]
    return [(r, c) for r in range(shape[0] - n + 1)
            for c in range(shape[1] - n + 1)
            if all(0 <= y - r < n and 0 <= x - c < n and mask[y - r, x - c]
                   for y, x in supplied)]


def _active(mask, origin):
    return {(origin[0] + int(y), origin[1] + int(x))
            for y, x in np.argwhere(mask)}


def _asymmetric(mask):
    alternatives = [np.rot90(mask, q) for q in (1, 2, 3)]
    alternatives += [np.rot90(np.fliplr(mask), q) for q in range(4)]
    return all(not np.array_equal(mask, a) for a in alternatives)


def _parse(input_grid):
    """Read the board, keys, masks and unique registrations from input alone."""
    grid = _grid(input_grid)
    candidates = []
    for color in map(int, np.unique(grid)):
        box = _box(np.argwhere(grid == color))
        r, e, c, d = box
        block = grid[r:e, c:d]
        rr = _runs(np.flatnonzero(np.all(block == color, axis=1)))
        cc = _runs(np.flatnonzero(np.all(block == color, axis=0)))
        if len(rr) < 3 or len(cc) < 3:
            continue
        if (rr[0][0], cc[0][0], rr[-1][1], cc[-1][1]) != (0, 0, e-r-1, d-c-1):
            continue
        widths = {b-a+1 for a, b in rr + cc}
        gaps = {rr[i+1][0]-rr[i][1]-1 for i in range(len(rr)-1)}
        gaps |= {cc[i+1][0]-cc[i][1]-1 for i in range(len(cc)-1)}
        if len(widths) != 1 or len(gaps) != 1:
            continue
        t, n = next(iter(widths)), next(iter(gaps))
        if t not in (1, 2) or n not in (2, 3, 4):
            continue
        expected = np.zeros(block.shape, bool)
        for a, b in rr:
            expected[a:b+1, :] = True
        for a, b in cc:
            expected[:, a:b+1] = True
        if np.array_equal(block == color, expected):
            candidates.append((box, color, n, t, len(rr)-1, len(cc)-1))
    assert len(candidates) == 1, "one exclusive regular divider lattice is required"
    box, divider, n, t, rows, cols = candidates[0]
    r0, r1, c0, c1 = box
    exterior = np.ones(grid.shape, bool)
    exterior[r0:r1, c0:c1] = False
    # The uniform exterior background is strictly modal outside the board.
    colors, counts = np.unique(grid[exterior], return_counts=True)
    assert len(counts) and np.sum(counts == counts.max()) == 1
    bg = int(colors[np.argmax(counts)])
    assert bg != divider
    exterior &= grid != bg
    mapping, key_boxes, removed = {}, {}, np.zeros(grid.shape, bool)
    for cells in _components(exterior):
        values = {int(grid[y, x]) for y, x in cells}
        if len(values) == 1:
            continue
        a, b, x, y = _box(cells)
        tile = grid[a:b, x:y]
        assert tile.shape in ((2, 3), (3, 2)) and len(cells) == 6
        values, frequencies = np.unique(tile, return_counts=True)
        assert len(values) == 2 and set(map(int, frequencies)) == {2, 4}
        source = int(values[np.argmin(frequencies)])
        target = int(values[np.argmax(frequencies)])
        source_mask = tile == source
        patterns = [np.rot90(np.array([[1, 0, 0], [1, 0, 0]], bool), q)
                    for q in range(4)]
        assert any(np.array_equal(source_mask, m) for m in patterns)
        assert source not in mapping and divider not in (source, target)
        mapping[source], key_boxes[source] = target, (a, b, x, y)
        removed[a:b, x:y] = True
    assert 1 <= len(mapping) <= 3
    remaining = exterior & ~removed
    assert set(map(int, np.unique(grid[remaining]))) == set(mapping)
    masks, stencil_boxes = {}, {}
    for source in mapping:
        a, b, x, y = _box(np.argwhere(remaining & (grid == source)))
        tile = grid[a:b, x:y]
        assert tile.shape == (n, n) and np.all(np.isin(tile, (bg, source)))
        masks[source], stencil_boxes[source] = tile == source, (a, b, x, y)
        assert 3 <= int(masks[source].sum()) < n*n
    boxes = list(stencil_boxes.values()) + list(key_boxes.values())
    for i, (a, b, x, y) in enumerate(boxes):
        assert b <= r0 or a >= r1 or y <= c0 or x >= c1
        for aa, bb, xx, yy in boxes[i+1:]:
            assert b < aa or bb < a or y < xx or yy < x
    supplied = {s: set() for s in mapping}
    for row in range(rows):
        for col in range(cols):
            y, x = r0+t+row*(n+t), c0+t+col*(n+t)
            tile = grid[y:y+n, x:x+n]
            assert np.all(tile == tile[0, 0]), "input cells are solid seeds or blank"
            source = int(tile[0, 0])
            if source != bg:
                assert source in mapping
                supplied[source].add((row, col))
    origins, complete = {}, {}
    for source in mapping:
        assert len(supplied[source]) >= 2
        choices = _origins(masks[source], (rows, cols), supplied[source])
        assert len(choices) == 1, "no tie-breaking between competing registrations"
        origins[source] = choices[0]
        complete[source] = _active(masks[source], choices[0])
        assert supplied[source] < complete[source]
    return dict(grid=grid, box=box, divider=divider, background=bg, n=n, t=t,
                shape=(rows, cols), masks=masks, mapping=mapping, supplied=supplied,
                complete=complete, origins=origins, key_boxes=key_boxes,
                stencil_boxes=stencil_boxes)


def _stamp(shape, n, masks, mapping, destinations):
    """Count SOURCE hits before mapping; any multiplicity >= 2 clears a pixel."""
    hits = np.zeros((*shape, n, n), dtype=np.int8)
    paint = np.zeros(hits.shape, dtype=np.int8)
    for source, places in destinations.items():
        for row, col in places:
            hits[row, col] += masks[source]
            paint[row, col][masks[source]] = mapping[source]
    return hits, paint


def _answer(info, destinations=None, cancel=True):
    r, e, c, d = info['box']
    result = info['grid'][r:e, c:d].copy()
    n, t, bg = info['n'], info['t'], info['background']
    hits, paint = _stamp(info['shape'], n, info['masks'], info['mapping'],
                         info['complete'] if destinations is None else destinations)
    for row in range(info['shape'][0]):
        for col in range(info['shape'][1]):
            y, x = t+row*(n+t), t+col*(n+t)
            tile = result[y:y+n, x:x+n]
            tile[:] = bg
            keep = hits[row, col] == 1 if cancel else hits[row, col] > 0
            tile[keep] = paint[row, col][keep]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _answer(_parse(input_grid))


def _mask(rng, n, asymmetric=False):
    for _ in range(80):
        kind = int(rng.integers(6))
        y, x = np.indices((n, n))
        if kind == 0:
            mask = (y == 0) | (x == 0) | (y == n-1)
        elif kind == 1:
            mask = (x == y) | (x+y == n-1)
        elif kind == 2:
            mask = (x+y) % 2 == 0
        else:
            mask = rng.random((n, n)) < rng.uniform(.3, .8)
        for edge in (mask[0], mask[-1], mask[:, 0], mask[:, -1]):
            if not edge.any():
                edge[int(rng.integers(n))] = True
        if mask.all():
            mask[int(rng.integers(n)), int(rng.integers(n))] = False
        mask = np.rot90(mask, int(rng.integers(4))).copy()
        if mask.sum() >= 3 and (not asymmetric or _asymmetric(mask)):
            return mask
    # A bounded asymmetric fallback for n >= 3.
    mask = np.zeros((n, n), bool)
    mask[0, :] = mask[:, 0] = True
    mask[-1, 1] = True
    return mask


def _fragment(rng, mask, shape, origin, occupied, regime):
    active = sorted(_active(mask, origin) - occupied)
    limit = min(len(active), int(mask.sum()) - 1)
    if limit < 2:
        return None
    if regime == 'sparse':
        sizes = list(range(2, min(limit, (int(mask.sum())-1)//2) + 1))
    elif regime == 'dense':
        sizes = list(range(max(2, (int(mask.sum())+1)//2), limit+1))
    else:
        sizes = list(range(2, limit+1))
    if not sizes:
        return None
    for _ in range(64):
        count = int(rng.choice(sizes))
        chosen = {active[int(i)] for i in rng.choice(len(active), count, replace=False)}
        if _origins(mask, shape, chosen) == [origin]:
            return chosen
    # Exhaustive small fallback, in randomized size order, with a finite cap.
    checked = 0
    for count in rng.permutation(sizes):
        for chosen in combinations(active, int(count)):
            checked += 1
            if _origins(mask, shape, chosen) == [origin]:
                return set(chosen)
            if checked == 4096:
                return None
    return None


def _make_pair(rng, n, k, direction, mapping_kind, fragment_regime, informative=False,
               require_triple=False):
    t = int(rng.choice((1, 2))) if n < 4 else 1
    max_rows, max_cols = (26-n-t)//(n+t), (30-t)//(n+t)
    min_rows = max(n, min(4, max_rows) if k == 3 else n)
    min_cols = max(n, 4 if k == 3 else n)
    rows, cols = int(rng.integers(min_rows, max_rows+1)), int(rng.integers(min_cols, max_cols+1))
    palette = list(map(int, rng.permutation(10)))
    bg, divider = palette[:2]
    sources, spare = palette[2:2+k], palette[2+k:]
    if mapping_kind == 'chain' and k >= 2:
        order = list(map(int, rng.permutation(sources)))
        mapping = {s: order[i+1] if i+1 < k else spare[0] for i, s in enumerate(order)}
    elif mapping_kind == 'cycle' and k >= 2:
        order = list(map(int, rng.permutation(sources)))
        mapping = {s: order[(i+1) % k] for i, s in enumerate(order)}
    elif mapping_kind == 'many':
        mapping = {s: spare[0] for s in sources}
    else:
        mapping = {s: spare[i] for i, s in enumerate(sources)}
    masks = {s: _mask(rng, n, informative and i == 0) for i, s in enumerate(sources)}
    origins = {s: (int(rng.integers(rows-n+1)), int(rng.integers(cols-n+1))) for s in sources}
    complete = {s: _active(masks[s], origins[s]) for s in sources}
    hits, paint = _stamp((rows, cols), n, masks, mapping, complete)
    if require_triple and not np.any(hits >= 3):
        return None
    if k >= 2:
        if not np.any(hits >= 2):
            return None
        # Teach intact stamps as well as their clipped collision versions.
        for s in sources:
            others = set().union(*(complete[o] for o in sources if o != s))
            if not complete[s] - others:
                return None
        # A mixed tile must retain ink and also erase actual intersecting ink.
        if not any(np.any(hits[r, c] == 1) and np.any(hits[r, c] >= 2)
                   for r in range(rows) for c in range(cols)):
            return None
    supplied, occupied = {}, set()
    for s in map(int, rng.permutation(sources)):
        fragment = _fragment(rng, masks[s], (rows, cols), origins[s], occupied, fragment_regime)
        if fragment is None:
            return None
        supplied[s] = fragment
        occupied |= fragment
    # Completion, rather than merely substituting existing blocks, must matter.
    old_hits, old_paint = _stamp((rows, cols), n, masks, mapping, supplied)
    if np.array_equal(np.where(hits == 1, paint, bg), np.where(old_hits == 1, old_paint, bg)):
        return None
    bh, bw = rows*(n+t)+t, cols*(n+t)+t
    # Separate, independently shuffled stencil and key banks above the board.
    width = max(bw, k*(n+1)-1, k*4-1)
    base_height = bh+n+4
    margin_y = int(rng.integers(min(2, 30-base_height)+1))
    margin_x = int(rng.integers(min(2, 30-width)+1))
    grid = np.full((base_height+margin_y, width+margin_x), bg, np.int8)
    top, left = n+4+int(rng.integers(margin_y+1)), int(rng.integers(width+margin_x-bw+1))
    board = grid[top:top+bh, left:left+bw]
    board[:] = divider
    for r in range(rows):
        for c in range(cols):
            board[t+r*(n+t):t+r*(n+t)+n, t+c*(n+t):t+c*(n+t)+n] = bg
    for s, places in supplied.items():
        for r, c in places:
            board[t+r*(n+t):t+r*(n+t)+n, t+c*(n+t):t+c*(n+t)+n] = s
    for i, s in enumerate(map(int, rng.permutation(sources))):
        x = i*(n+1)
        grid[:n, x:x+n][masks[s]] = s
    for i, s in enumerate(map(int, rng.permutation(sources))):
        grid[n+1:n+3, i*4:i*4+3] = np.array([[s, mapping[s], mapping[s]]]*2)
    info = dict(grid=grid, box=(top, top+bh, left, left+bw), n=n, t=t,
                background=bg, shape=(rows, cols), masks=masks, mapping=mapping,
                complete=complete)
    output = _answer(info)
    return {'input': np.rot90(grid, direction).copy(), 'output': np.rot90(output, direction).copy()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, name, low, high in ((num_train, 'num_train', 2, 10), (num_test, 'num_test', 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(('scale', 'mapping', 'sparse', 'side')))
    directions = list(map(int, rng.permutation(4)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(count):
            for _ in range(600):
                is_train = split == 'train'
                n = 3 if is_train and index == 0 else int(rng.choice((3, 4) if is_train and index == 1 else (2, 3, 4)))
                k = (3 if is_train and index == 1 else
                     (int(rng.choice((1, 2, 3))) if is_train and index >= 2 else int(rng.choice((2, 3)))))
                if is_train and index == 0:
                    k = 2
                if shift == 'scale':
                    n = 3 if is_train and index < 2 else (int(rng.choice((2, 3))) if is_train else 4)
                elif shift == 'sparse':
                    n = 3 if is_train and index == 0 else int(rng.choice((3, 4)))
                direction = int(rng.choice(directions[:2] if is_train else directions[2:])) if shift == 'side' else int(rng.integers(4))
                kind = 'chain' if is_train and index == 0 else ('many' if is_train and index == 1 else str(rng.choice(('disjoint', 'chain', 'cycle', 'many'))))
                if shift == 'mapping':
                    if not is_train:
                        kind = 'cycle'
                    elif index >= 2:
                        kind = str(rng.choice(('disjoint', 'chain', 'many')))
                regime = ('dense' if is_train else 'sparse') if shift == 'sparse' else 'any'
                pair = _make_pair(rng, n, k, direction, kind, regime,
                                  is_train and index == 0, is_train and index == 1)
                if pair is not None and _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _parse(inp)
            expected = _answer(info)
            assert np.array_equal(expected, out), 'incorrect completion, mapping, or cancellation'
            assert set(map(int, np.unique(out))) <= set(map(int, np.unique(inp)))
            assert not np.array_equal(out, _answer(info, info['supplied'])), 'A completion has no effect'
            identity = dict(info, mapping={s: s for s in info['mapping']})
            assert not np.array_equal(out, _answer(identity)), 'B paint mapping has no effect'
            r, e, c, d = info['box']
            assert out.shape == (e-r, d-c)
            div = inp[r:e, c:d] == info['divider']
            assert np.array_equal(out[div], inp[r:e, c:d][div])
            hits, _ = _stamp(info['shape'], info['n'], info['masks'], info['mapping'], info['complete'])
            if len(info['mapping']) >= 2:
                assert np.any(hits >= 2) and not np.array_equal(out, _answer(info, cancel=False))
                for s in info['mapping']:
                    others = set().union(*(info['complete'][o] for o in info['mapping'] if o != s))
                    assert info['complete'][s] - others, 'each source needs an intact destination'
            if split == 'train':
                training.append(info)
    assert any(_asymmetric(m) for a in training for m in a['masks'].values())
    assert any(a['mapping'][s] in a['mapping'] for a in training for s in a['mapping'])
    assert any(len(set(a['mapping'].values())) < len(a['mapping']) for a in training)
    assert any(np.any(_stamp(a['shape'], a['n'], a['masks'], a['mapping'], a['complete'])[0] >= 3)
               for a in training), 'training must distinguish triple clearing from parity'
    assert any(not np.array_equal(_answer(a), _answer(dict(a, mapping={
        s: a['mapping'].get(paint, paint) for s, paint in a['mapping'].items()})))
        for a in training), 'training must distinguish direct from recursive paint'


def self_test():
    """Bounded 42-seed diversity, count, regression and metamorphic checks."""
    puzzles = [generate_puzzle(s) for s in range(42)]
    records = [_parse(p['input']) for puzzle in puzzles for split in ('train', 'test') for p in puzzle[split]]
    assert len({_key(p['train'][0]['input']) for p in puzzles}) == 42
    assert len({_key(m) for a in records for m in a['masks'].values()}) >= 45
    assert len({a['grid'].shape for a in records}) >= 30
    assert {a['n'] for a in records} == {2, 3, 4}
    assert {a['t'] for a in records} == {1, 2}
    assert {len(a['mapping']) for a in records} == {1, 2, 3}
    assert {a['background'] for a in records} == set(range(10))
    assert {a['divider'] for a in records} == set(range(10))
    assert any(len(_components(m)) > 1 for a in records for m in a['masks'].values())
    assert any(not _asymmetric(m) for a in records for m in a['masks'].values())
    assert any(2*len(a['supplied'][s]) < int(a['masks'][s].sum()) for a in records for s in a['mapping'])
    assert any(np.any(_stamp(a['shape'], a['n'], a['masks'], a['mapping'], a['complete'])[0] >= 3) for a in records)
    for seed in (0, 17, 41):
        repeated = generate_puzzle(seed)
        for split in ('train', 'test'):
            for p, q in zip(puzzles[seed][split], repeated[split]):
                assert all(np.array_equal(p[k], q[k]) for k in ('input', 'output'))
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            p = generate_puzzle(103+train*5+test, train, test)
            assert len(p['train']) == train and len(p['test']) == test
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Hand-computed two-source pixel collision, including equal mapped colors.
    masks = {1: np.array([[1, 1], [1, 0]], bool), 2: np.array([[1, 0], [1, 1]], bool)}
    places = {1: {(0, 0)}, 2: {(0, 0)}}
    hits, paint = _stamp((1, 1), 2, masks, {1: 7, 2: 7}, places)
    assert np.array_equal(hits[0, 0], [[2, 1], [2, 1]])
    assert np.array_equal(np.where(hits[0, 0] == 1, paint[0, 0], 0), [[0, 7], [0, 7]])
    # Triple hits also clear; this is not odd/even parity cancellation.
    masks[3] = np.ones((2, 2), bool)
    hits, _ = _stamp((1, 1), 2, masks, {1: 7, 2: 8, 3: 9}, places | {3: {(0, 0)}})
    assert hits[0, 0, 0, 0] == 3
    # Palette permutations and whole-grid rotations preserve the same rule.
    permutation = np.random.default_rng(716).permutation(10)
    for pair in puzzles[0]['train'] + puzzles[0]['test']:
        for q in range(4):
            inp = np.rot90(permutation[pair['input']], q).copy()
            expected = np.rot90(permutation[pair['output']], q).copy()
            before = inp.copy()
            assert np.array_equal(solve(inp), expected) and np.array_equal(inp, before)
    # Moving keys or complete stencil squares changes proximity, not association.
    moved_key = moved_stencil = False
    for a in records:
        sources = list(a['mapping'])
        if len(sources) < 2:
            continue
        b1, b2 = [a['key_boxes'][s] for s in sources[:2]]
        r, e, c, d = b1
        rr, ee, cc, dd = b2
        if (e-r, d-c) != (ee-rr, dd-cc):
            continue
        changed = a['grid'].copy()
        changed[r:e, c:d], changed[rr:ee, cc:dd] = a['grid'][rr:ee, cc:dd], a['grid'][r:e, c:d]
        assert np.array_equal(solve(changed), _answer(a))
        moved_key = True
        r, e, c, d = a['stencil_boxes'][sources[0]]
        rr, ee, cc, dd = a['stencil_boxes'][sources[1]]
        changed = a['grid'].copy()
        changed[r:e, c:d], changed[rr:ee, cc:dd] = a['grid'][rr:ee, cc:dd], a['grid'][r:e, c:d]
        assert np.array_equal(solve(changed), _answer(a))
        moved_stencil = True
        break
    assert moved_key and moved_stencil
    # Two left-column clues of a U admit several translations. Equal overlap
    # is not grounds to pick one of them, even though all clues fit each.
    ambiguous = np.zeros((28, 21), dtype=np.int8)
    u = np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], bool)
    ambiguous[:3, :3][u] = 7
    ambiguous[4:6, :3] = [[7, 9, 9], [7, 9, 9]]
    board = ambiguous[7:, :]
    board[:] = 2
    for r in range(5):
        for c in range(5):
            board[1+r*4:4+r*4, 1+c*4:4+c*4] = 0
    for r, c in ((1, 1), (2, 1)):
        board[1+r*4:4+r*4, 1+c*4:4+c*4] = 7
    assert len(_origins(u, (5, 5), {(1, 1), (2, 1)})) > 1
    try:
        solve(ambiguous)
    except AssertionError as error:
        assert 'registration' in str(error)
    else:
        raise AssertionError('ambiguous registration accepted')
    damaged = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ('train', 'test')}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt divider accepted')
