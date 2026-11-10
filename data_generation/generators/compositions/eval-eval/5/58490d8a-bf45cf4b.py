"""Ink-budget route: component counts gate the colored layers of a block stamp.

Only NumPy and the standard library are used. Generation, validation and tests
write no files. Run this module to print one bounded, compact self-test report.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


HOST_SEEDS = (0, 260775159, 236157378, 396595103, 934706788, 100282564,
              911891119, 903359578, 937399645, 160857169, 968175052,
              599090263, 420141132, 170162751, 825373644, 41060445,
              972245328, 921849086, 372503605, 830456933, 685541459,
              145503467, 68633737, 979746373, 217241474, 564487686,
              532704009, 749028803, 241441827, 733900785, 911811857,
              188672415, 953377664, 236864589, 74692685, 679996289,
              593464097, 718006272, 149109789, 71533320, 444474505, 194651372)
_ORTH = ((-1, 0), (0, -1), (0, 1), (1, 0))
_EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
_DIAMOND = ((0, 1), (1, 0), (1, 2), (2, 1))
_SQUARE = ((0, 0), (0, 1), (1, 0), (1, 1))
_SINGLE = ((0, 0),)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid bounds"
    assert np.issubdtype(a.dtype, np.integer), "integer grids required"
    assert np.all((0 <= a) & (a <= 9)), "color bounds"
    return a


def _key(a):
    a = np.asarray(a)
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _normalize(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple((int(r-r0), int(c-c0)) for r, c in cells)


def _bounds(cells):
    return (min(r for r, c in cells), max(r for r, c in cells)+1,
            min(c for r, c in cells), max(c for r, c in cells)+1)


def _components(mask, diagonal=True):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    steps = _EIGHT if diagonal else _ORTH
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, stack = [], [start]
        while stack:
            r, c = stack.pop()
            component.append((int(r), int(c)))
            for dr, dc in steps:
                p = (r+dr, c+dc)
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(component)
    return result


def _asymmetric(a):
    for k in range(4):
        b = np.rot90(a, k)
        if k and np.array_equal(a, b):
            return False
        if np.array_equal(a, np.fliplr(b)):
            return False
    return True


def _parse(input_grid):
    """Recover every parameter from one input; no generation state is read."""
    g = _grid(input_grid)
    colors, frequencies = np.unique(g, return_counts=True)
    bg = int(colors[np.argmax(frequencies)])
    assert frequencies.max() > g.size/2 and bg < 8, "majority background"
    border = np.concatenate((g[0], g[-1], g[:, 0], g[:, -1]))
    assert np.all(border == bg), "blank input border"
    route_mask = (g == 8) | (g == 9)
    starts = np.argwhere(g == 9)
    assert len(starts) == 1, "one marked starting endpoint"
    cells = set(map(tuple, np.argwhere(route_mask)))
    assert len(cells) >= 3
    neighbors = {p: [(p[0]+dr, p[1]+dc) for dr, dc in _ORTH
                     if (p[0]+dr, p[1]+dc) in cells] for p in cells}
    assert sum(len(v) == 1 for v in neighbors.values()) == 2
    assert all(len(v) in (1, 2) for v in neighbors.values()), "unbranched route"
    start = tuple(map(int, starts[0]))
    assert len(neighbors[start]) == 1, "start is an endpoint"
    path, previous, current = [], None, start
    while current is not None:
        assert current not in path, "route cycle"
        path.append(current)
        following = [p for p in neighbors[current] if p != previous]
        previous, current = current, following[0] if following else None
    assert len(path) == len(cells), "all route cells must be reached"
    rb = _bounds(path)
    r0, r1, c0, c1 = rb
    assert np.all(np.isin(g[r0:r1, c0:c1], (bg, 8, 9))), "clean route crop"
    # Its whole crop plus a blank halo is kept free of unrelated encodings.
    halo = g[r0-1:r1+1, c0-1:c1+1]
    assert np.all(np.isin(halo, (bg, 8, 9)))
    records = []
    for comp in _components((g != bg) & ~route_mask):
        palette = {int(g[r, c]) for r, c in comp}
        records.append((comp, palette))
    multi = [(comp, palette) for comp, palette in records if len(palette) >= 2]
    assert len(multi) == 1, "one multicolor stamp, monochrome supply objects"
    stamp_cells, keys = multi[0]
    sb = _bounds(stamp_cells)
    sr0, sr1, sc0, sc1 = sb
    stamp = g[sr0:sr1, sc0:sc1].copy()
    assert min(stamp.shape) >= 2 and keys <= set(range(8))
    assert np.count_nonzero(stamp != bg) == len(stamp_cells), "unique stamp crop"
    ring = g[sr0-1:sr1+1, sc0-1:sc1+1].copy()
    ring[1:-1, 1:-1] = bg
    assert np.all(ring == bg), "stamp's rectangular halo"
    objects, counts = {}, {}
    for comp, palette in records:
        if len(palette) != 1:
            continue
        color = next(iter(palette))
        assert all(not (sr0 <= r < sr1 and sc0 <= c < sc1) for r, c in comp)
        objects.setdefault(color, []).append(tuple(sorted(_normalize(comp))))
        counts[color] = counts.get(color, 0)+1
    assert all(counts.get(c, 0) > 0 for c in keys), "positive supply per stamp color"
    shape = (r1-r0, c1-c0)
    assert shape[0]*stamp.shape[0] <= 30 and shape[1]*stamp.shape[1] <= 30
    return dict(background=bg, path=_normalize(path), route_shape=shape,
                stamp=stamp, keys=tuple(sorted(keys)), counts=counts,
                objects=objects, route_bounds=rb, stamp_bounds=sb,
                distractors=tuple(sorted(set(counts)-keys)))


def _pixel_prediction(meta, counts=None, path=None, stamp=None):
    """Input-only prediction by rank lookup and fine/coarse coordinate indexing."""
    stamp = meta['stamp'] if stamp is None else stamp
    counts = meta['counts'] if counts is None else counts
    path = meta['path'] if path is None else path
    sh, sw = stamp.shape
    rank = np.zeros(meta['route_shape'], dtype=int)
    for i, (r, c) in enumerate(path, 1):
        rank[r, c] = i
    budgets = np.zeros(10, dtype=int)
    for color in meta['keys']:
        budgets[color] = counts.get(color, 0)
    rr, cc = np.indices((rank.shape[0]*sh, rank.shape[1]*sw))
    ink = stamp[rr % sh, cc % sw]
    step = rank[rr // sh, cc // sw]
    active = (step > 0) & (step <= budgets[ink]) & (ink != meta['background'])
    return np.where(active, ink, meta['background']).astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve any input satisfying the family grammar, without demonstrations."""
    return _pixel_prediction(_parse(input_grid))


def _block_output(path, stamp, bg, counts):
    """Construction labels use explicit block writes, not the solver indexing."""
    mh, mw = 1+max(r for r, c in path), 1+max(c for r, c in path)
    sh, sw = stamp.shape
    output = np.full((mh*sh, mw*sw), bg, dtype=np.int8)
    for index, (r, c) in enumerate(path, 1):
        tile = stamp.copy()
        for color in np.unique(stamp):
            if color != bg and index > counts.get(int(color), 0):
                tile[stamp == color] = bg
        output[r*sh:(r+1)*sh, c*sw:(c+1)*sw] = tile
    return output


def _make_route(rng, length, stamp_shape, witness=False):
    caps = tuple(min(10, 30//n) for n in stamp_shape)
    style = int(rng.integers(3))
    path = None
    if style == 2:
        for _ in range(40):
            trial = [(int(rng.integers(caps[0])), int(rng.integers(caps[1])))]
            occupied = set(trial)
            while len(trial) < length:
                r, c = trial[-1]
                choices = []
                for dr, dc in _ORTH:
                    q = (r+dr, c+dc)
                    if not (0 <= q[0] < caps[0] and 0 <= q[1] < caps[1]) or q in occupied:
                        continue
                    if sum((q[0]+a, q[1]+b) in occupied for a, b in _ORTH) == 1:
                        choices.append(q)
                if not choices:
                    break
                q = choices[int(rng.integers(len(choices)))]
                trial.append(q)
                occupied.add(q)
            if len(trial) == length:
                path = trial
                break
    if path is None and style == 1:
        width = (length+1)//2
        trial = ([(0, c) for c in range(width)] + [(1, width-1)] +
                 [(2, c) for c in range(width-1, -1, -1)])[:length]
        if 3 <= caps[0] and width <= caps[1]:
            path = trial
    if path is None:
        # Constructive bounded fallback: a random monotone staircase.
        feasible = [h for h in range(1, caps[0]+1) if 1 <= length-h+1 <= caps[1]]
        assert feasible, "compatible stamp and route dimensions"
        h = feasible[int(rng.integers(len(feasible)))]
        w = length-h+1
        moves = [(1, 0)]*(h-1)+[(0, 1)]*(w-1)
        rng.shuffle(moves)
        path = [(0, 0)]
        for dr, dc in moves:
            r, c = path[-1]
            path.append((r+dr, c+dc))
    path = list(_normalize(path))
    if rng.random() < .5:
        path = list(_normalize([(r, -c) for r, c in path]))
    if rng.random() < .5:
        path = list(_normalize([(-r, c) for r, c in path]))
    if rng.random() < .5:
        path.reverse()
    if witness and path[0] == min(path):
        path.reverse()
    return tuple(path)


def _make_stamp(rng, shape, bg, keys, witness):
    h, w = shape
    rr, cc = np.indices(shape)
    for _ in range(60):
        mode = int(rng.integers(5))
        if mode == 0:
            labels = rr+cc
        elif mode == 1:
            labels = rr if rng.random() < .5 else cc
        elif mode == 2:
            labels = np.minimum.reduce((rr, cc, h-1-rr, w-1-cc))
        elif mode == 3:
            labels = abs(2*rr-h+1)+abs(2*cc-w+1)
        else:
            labels = rng.integers(len(keys), size=shape)
        stamp = np.array(keys, dtype=np.int8)[(labels+int(rng.integers(len(keys)))) % len(keys)]
        # Supply all keys at independently sampled positions, never a palette row.
        locations = rng.choice(h*w, len(keys), replace=False)
        for color, index in zip(keys, locations):
            stamp.flat[int(index)] = color
        holes = rng.random(shape) < (.12 if witness else float(rng.choice((0, .08, .22))))
        if witness:
            holes[1, 1] = True
        stamp[holes] = bg
        visible = stamp != bg
        if set(map(int, np.unique(stamp)))-{bg} != set(keys):
            continue
        if not all(np.any(edge) for edge in (visible[0], visible[-1], visible[:, 0], visible[:, -1])):
            continue
        if len(_components(visible)) != 1:
            continue
        if witness and (not _asymmetric(stamp) or not _asymmetric(stamp == keys[0])
                        or not np.any(~visible)):
            continue
        return stamp
    # Witness is 3x3 with two colors, so this explicit asymmetric hole fallback
    # is always legal there. Other stamps use a full, connected rectangle.
    stamp = np.full(shape, keys[0], dtype=np.int8)
    for i, color in enumerate(keys[1:], 1):
        stamp.flat[i] = color
    if witness:
        stamp[1, 1] = bg
        stamp[-1, 0] = keys[1]
    return stamp


def _glyph(rng):
    mode = int(rng.integers(7))
    if mode == 0:
        return _DIAMOND
    if mode == 1:
        return _SQUARE
    if mode == 2:
        return ((0, 0), (1, 1), (2, 2))
    if mode == 3:
        return ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2))
    if mode == 4:
        return tuple((0, c) for c in range(int(rng.integers(1, 4))))
    target = int(rng.integers(1, 10))
    cells = {(1, 1)}
    while len(cells) < target:
        fringe = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _EIGHT
                         if 0 <= r+dr < 3 and 0 <= c+dc < 3}-cells)
        cells.add(fringe[int(rng.integers(len(fringe)))])
    return tuple(sorted(_normalize(sorted(cells))))


def _block_rect(blocked, top, left, height, width):
    blocked[max(0, top-1):top+height+1, max(0, left-1):left+width+1] = True


def _place_rect(grid, blocked, value, rng):
    h, w = value.shape
    nr, nc = grid.shape[0]-h, grid.shape[1]-w
    free = np.ones((nr+1, nc+1), dtype=bool)
    for r in range(h):
        for c in range(w):
            free &= ~blocked[r:r+nr+1, c:c+nc+1]
    points = np.argwhere(free)
    if not len(points):
        return False
    r, c = map(int, points[int(rng.integers(len(points)))])
    grid[r:r+h, c:c+w] = value
    _block_rect(blocked, r, c, h, w)
    return True


def _place_glyph(grid, blocked, color, shape, rng):
    sh, sw = 1+max(r for r, c in shape), 1+max(c for r, c in shape)
    nr, nc = grid.shape[0]-sh+1, grid.shape[1]-sw+1
    free = np.ones((nr, nc), dtype=bool)
    for r, c in shape:
        free &= ~blocked[r:r+nr, c:c+nc]
    points = np.argwhere(free)
    if not len(points):
        return False
    r0, c0 = map(int, points[int(rng.integers(len(points)))])
    for dr, dc in shape:
        r, c = r0+dr, c0+dc
        grid[r, c] = color
        blocked[max(0, r-1):r+2, max(0, c-1):c+2] = True
    return True


def _compose_scene(rng, bg, path, stamp, objects):
    route = np.full((1+max(r for r, c in path), 1+max(c for r, c in path)), bg, dtype=np.int8)
    for i, (r, c) in enumerate(path):
        route[r, c] = 9 if i == 0 else 8
    objects = list(objects)
    rng.shuffle(objects)
    objects.sort(key=lambda item: -len(item[1]))
    estimate = route.size+stamp.size+sum((2+max(r for r, c in s))*(2+max(c for r, c in s)) for _, s in objects)
    lower = max(12, int(np.sqrt(estimate*1.5)))
    for attempt in range(6):
        if attempt == 5:
            # Guaranteed packing: corner crops and 3x3 supply slots in 30x30.
            g = np.full((30, 30), bg, dtype=np.int8)
            blocked = np.zeros(g.shape, dtype=bool)
            rh, rw = route.shape
            sh, sw = stamp.shape
            g[1:1+rh, 1:1+rw] = route
            g[1:1+sh, 29-sw:29] = stamp
            _block_rect(blocked, 1, 1, rh, rw)
            _block_rect(blocked, 1, 29-sw, sh, sw)
            slots = [(r, c) for r in range(1, 27, 4) for c in range(1, 27, 4)
                     if not blocked[r:r+3, c:c+3].any()]
            assert len(slots) >= len(objects), "fallback slot capacity"
            rng.shuffle(slots)
            for (color, shape), (r0, c0) in zip(objects, slots):
                for dr, dc in shape:
                    g[r0+dr, c0+dc] = color
            return g
        h = min(30, max(route.shape[0]+4, stamp.shape[0]+4,
                       int(rng.integers(lower, max(lower+1, 29)))+attempt))
        w = min(30, max(route.shape[1]+stamp.shape[1]+5,
                       int(rng.integers(max(12, estimate//h), 31))+attempt))
        g = np.full((h, w), bg, dtype=np.int8)
        blocked = np.zeros(g.shape, dtype=bool)
        blocked[[0, -1], :] = True
        blocked[:, [0, -1]] = True
        encodings = [route, stamp]
        if rng.random() < .5:
            encodings.reverse()
        if not all(_place_rect(g, blocked, a, rng) for a in encodings):
            continue
        if all(_place_glyph(g, blocked, color, shape, rng) for color, shape in objects):
            return g
    raise AssertionError("unreachable packing fallthrough")


def _pair(rng, regime, train, index, anchor, first_bg):
    witness, role_witness = train and index == 0, train and index == 1
    length = int(rng.integers(5, 8) if train or regime != 0 else rng.integers(9, 13))
    if not train and regime != 0:
        length = int(rng.integers(5, 9))
    if witness:
        k, stamp_shape = 2, (3, 3)
    else:
        k = 2 if train and regime == 2 else int(rng.integers(3, 5) if not train and regime == 2 else rng.integers(2, 4))
        if role_witness:
            stamp_shape = (2, 3) if rng.random() < .5 else (3, 2)
        elif not train and regime == 1:
            options = ((4, 3), (3, 4), (5, 2), (2, 5), (4, 4), (5, 3), (3, 5))
            stamp_shape = options[int(rng.integers(len(options)))]
        else:
            cap = 3 if train and regime == 1 else 4
            stamp_shape = (int(rng.integers(2, cap+1)), int(rng.integers(2, cap+1)))
    palette = list(map(int, rng.permutation(8)))
    if witness:
        bg = first_bg
        keys = [anchor]+[c for c in palette if c not in (bg, anchor)][:k-1]
    elif role_witness:
        bg = next(c for c in palette if c not in (anchor, first_bg))
        keys = [first_bg]+[c for c in palette if c not in (bg, first_bg, anchor)][:k-1]
    else:
        bg, keys = palette[0], palette[1:1+k]
    available = [c for c in palette if c not in keys and c != bg]
    if role_witness:
        available.remove(anchor)
        available.insert(0, anchor)
    nd = max(1, int(rng.integers(3))) if role_witness else int(rng.integers(3))
    distractors = available[:nd]
    counts = {c: int(rng.integers(1, min(6, length)+1)) for c in keys}
    if witness:
        counts[keys[0]], counts[keys[1]] = 3, 1
    else:
        # Sometimes ink survives the entire route, including actual surplus.
        if rng.random() < .28:
            counts[keys[0]] = length+1
        if min(counts.values()) >= length:
            counts[keys[-1]] = max(1, length-2)
    dcounts = {c: int(rng.integers(1, 5)) for c in distractors}
    while sum(counts.values())+sum(dcounts.values()) > 24:
        pool = [c for c in counts if counts[c] > 1]
        counts[max(pool, key=lambda c: counts[c])] -= 1
    path = _make_route(rng, length, stamp_shape, witness)
    if witness:
        # Demonstrate a turn while the three-object color is still available.
        # Otherwise a rule that rotates stamps only on turns can evade training.
        for _ in range(24):
            a, b, c = path[:3]
            if (b[0]-a[0], b[1]-a[1]) != (c[0]-b[0], c[1]-b[1]):
                break
            path = _make_route(rng, length, stamp_shape, True)
        else:
            path = ((0, 1), (0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (3, 2))[:length]
    stamp = _make_stamp(rng, stamp_shape, bg, keys, witness)
    objects = []
    for color, count in list(counts.items())+list(dcounts.items()):
        prototype = _glyph(rng)
        for j in range(count):
            shape = prototype if rng.random() < .25 else _glyph(rng)
            if witness and color == keys[0]:
                shape = (_DIAMOND, _SQUARE, _SINGLE)[j]
            objects.append((color, shape))
    g = _compose_scene(rng, bg, path, stamp, objects)
    expected = _block_output(path, stamp, bg, counts)
    assert np.array_equal(solve(g), expected), "construction/visible input disagreement"
    return {'input': g, 'output': expected}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, n, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not low <= n <= high:
            raise ValueError(name+' is outside its legal integer range')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    regime = int(rng.integers(3))
    first_bg, anchor = map(int, rng.choice(8, 2, replace=False))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, n in (('train', int(num_train)), ('test', int(num_test))):
        for i in range(n):
            for _ in range(8):
                pair = _pair(rng, regime, split == 'train', i, anchor, first_bg)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded distinct-input retries exhausted')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _shortcut_predictions(m):
    """Deliberately wrong alternatives used as demonstration coverage checks."""
    g_counts = m['counts']
    area = {c: sum(map(len, m['objects'][c])) for c in m['keys']}
    orth = {}
    for c in m['keys']:
        total = 0
        for shape in m['objects'][c]:
            mask = np.zeros((1+max(r for r, s in shape), 1+max(s for r, s in shape)), dtype=bool)
            for r, s in shape:
                mask[r, s] = True
            total += len(_components(mask, False))
        orth[c] = total
    pixel_budget = {c: g_counts[c]//int(np.sum(m['stamp'] == c)) for c in m['keys']}
    alternatives = {
        'unlimited_ink': _pixel_prediction(m, {c: len(m['path']) for c in m['keys']}),
        'pixel_area_counts': _pixel_prediction(m, area),
        'four_neighbor_objects': _pixel_prediction(m, orth),
        'include_template': _pixel_prediction(m, {c: g_counts[c]+1 for c in m['keys']}),
        'one_token_per_pixel': _pixel_prediction(m, pixel_budget),
        'row_major_route': _pixel_prediction(m, path=sorted(m['path'])),
        'reverse_route': _pixel_prediction(m, path=m['path'][::-1]),
        'mirror_stamp': _pixel_prediction(m, stamp=np.fliplr(m['stamp'])),
        'turn_stamp': _pixel_prediction(m, stamp=np.rot90(m['stamp'], 2)),
        'solid_stamp': _pixel_prediction(m, stamp=np.full(m['stamp'].shape, m['keys'][0], dtype=np.int8)),
        'fill_stamp_holes': _pixel_prediction(m, stamp=np.where(m['stamp'] == m['background'],
                                                              m['keys'][0], m['stamp'])),
    }
    # Square stamps make this plausible route-relative interpretation directly
    # comparable. First placement keeps its orientation; each turn rotates it.
    if m['stamp'].shape[0] == m['stamp'].shape[1]:
        turned = np.full_like(alternatives['turn_stamp'], m['background'])
        tile = m['stamp'].copy()
        previous_direction = None
        headings = {(0, 1): 0, (1, 0): 1, (0, -1): 2, (-1, 0): 3}
        size = tile.shape[0]
        for i, (r, c) in enumerate(m['path'], 1):
            if i > 1:
                pr, pc = m['path'][i-2]
                direction = headings[(r-pr, c-pc)]
                if previous_direction is not None:
                    tile = np.rot90(tile, previous_direction-direction)
                previous_direction = direction
            filtered = tile.copy()
            for color in m['keys']:
                if i > g_counts[color]:
                    filtered[tile == color] = m['background']
            turned[r*size:(r+1)*size, c*size:(c+1)*size] = filtered
        alternatives['rotate_on_turns'] = turned
    return alternatives


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, training, testing = set(), [], []
    rejected = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            g, output = _grid(pair['input']), _grid(pair['output'])
            assert _key(g) not in seen, "duplicate input"
            seen.add(_key(g))
            m = _parse(g)
            expected = _pixel_prediction(m)
            assert np.array_equal(output, expected), "incorrect ink-budget stamping"
            assert not np.any((output == 8) | (output == 9))
            sh, sw = m['stamp'].shape
            assert output.shape == (m['route_shape'][0]*sh, m['route_shape'][1]*sw)
            assert np.array_equal(output[m['path'][0][0]*sh:(m['path'][0][0]+1)*sh,
                                         m['path'][0][1]*sw:(m['path'][0][1]+1)*sw], m['stamp'])
            for c in m['keys']:
                assert int(np.sum(output == c)) == int(np.sum(m['stamp'] == c))*min(m['counts'][c], len(m['path']))
            (training if split == 'train' else testing).append(m)
            if split == 'train':
                rejected.update(name for name, pred in _shortcut_predictions(m).items()
                                if not np.array_equal(pred, output))
    assert rejected == (set(_shortcut_predictions(training[0])) | {'rotate_on_turns'}), "missing shortcut witness"
    # A single color has a diamond, square and singleton: equal-area connectivity
    # differences and unequal areas jointly establish actual object counting.
    assert any(any({_DIAMOND, _SQUARE, _SINGLE} <= set(m['objects'][c])
                   and m['counts'][c] == 3 < len(m['path']) for c in m['keys']) for m in training)
    assert any(np.any(m['stamp'] == m['background']) and _asymmetric(m['stamp']) for m in training)
    assert len({m['stamp'].shape for m in training}) >= 2
    assert any(set(a['keys']) & set(b['distractors']) for a in training for b in training)
    assert any(a['background'] in b['keys'] for a in training for b in training)
    # At least one controlled shift is visible; these are sampling regimes only.
    assert (min(len(m['path']) for m in testing) > max(len(m['path']) for m in training)
            or min(m['stamp'].size for m in testing) > max(m['stamp'].size for m in training)
            or min(len(m['keys']) for m in testing) > max(len(m['keys']) for m in training))


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Streaming checks on 42 specified seeds; retain features, never grid archives."""
    signatures, shapes, routes, stamps, glyphs, palettes = set(), set(), set(), set(), set(), set()
    shifts, counts_seen, categories, backgrounds = set(), set(), set(), set()
    surplus = ties = holes = backtracking = 0
    pairs = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = _fingerprint(puzzle)
        assert signature == _fingerprint(generate_puzzle(seed))
        signatures.add(signature)
        by_split = {}
        for split in ('train', 'test'):
            by_split[split] = []
            for pair in puzzle[split]:
                pairs += 1
                original = np.asarray(pair['input']).copy()
                m = _parse(original)
                predicted = solve(original)
                assert np.array_equal(original, pair['input']), "solve mutates input"
                assert np.array_equal(predicted, pair['output'])
                by_split[split].append(m)
                shapes.add((tuple(original.shape), tuple(predicted.shape)))
                routes.add((m['route_shape'], m['path']))
                roles = {m['background']: 0}
                canonical = tuple(roles.setdefault(int(c), len(roles)) for c in m['stamp'].flat)
                stamps.add((m['stamp'].shape, canonical))
                palettes.add((m['background'], m['keys']))
                backgrounds.add(m['background'])
                categories.add(len(m['keys']))
                ns = [m['counts'][c] for c in m['keys']]
                counts_seen.update(ns)
                surplus += max(ns) > len(m['path'])
                ties += len(set(ns)) < len(ns)
                holes += bool(np.any(m['stamp'] == m['background']))
                backtracking += m['path'] != tuple(sorted(m['path']))
                for objects in m['objects'].values():
                    glyphs.update(objects)
        train, test = by_split['train'], by_split['test']
        shifts.add((min(len(m['path']) for m in test) > max(len(m['path']) for m in train),
                    min(m['stamp'].size for m in test) > max(m['stamp'].size for m in train),
                    min(len(m['keys']) for m in test) > max(len(m['keys']) for m in train)))
    assert len(signatures) == 42
    assert len(shapes) >= 100 and len(routes) >= 120 and len(stamps) >= 100
    assert len(glyphs) >= 50 and len(palettes) >= 80
    assert backgrounds == set(range(8)) and categories == {2, 3, 4}
    assert max(counts_seen) >= 8 and surplus and ties and holes >= 42 and backtracking >= 100
    assert len(shifts) >= 3
    legal_counts = 0
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(0, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
            legal_counts += 1
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    p = generate_puzzle(0)
    p['test'][0]['output'][0, 0] = (int(p['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return dict(valid=True, seeds=42, pairs=pairs, legal_count_combinations=legal_counts,
                distinct_puzzles=len(signatures), shape_pairs=len(shapes), ordered_routes=len(routes),
                normalized_stamps=len(stamps), glyph_shapes=len(glyphs), palettes=len(palettes),
                categories=sorted(categories), component_counts=sorted(counts_seen),
                surplus_pairs=int(surplus), tied_budget_pairs=int(ties), hole_pairs=int(holes),
                non_row_major_pairs=int(backtracking), shift_signatures=sorted(shifts),
                independent_solver=False, blind_prediction=False)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
