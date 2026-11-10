"""Sew count-selected stamps, then classify the assembled topology.

All answer conventions are visible in each input. Only standard-library and
NumPy facilities are used. Source-card frames and the legend divider share a
color; no literal color is assigned a fixed semantic role.
"""
from __future__ import annotations

import numbers
from functools import lru_cache
import numpy as np


_FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask, diagonal=False):
    mask = np.asarray(mask, dtype=bool)
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    steps = _FOUR + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else _FOUR
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in steps:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _box(cells):
    p = np.asarray(cells, dtype=int)
    r, c = p.min(axis=0)
    b, e = p.max(axis=0) + 1
    return int(r), int(b), int(c), int(e)


def _mask_from_cells(cells):
    r, b, c, e = _box(cells)
    mask = np.zeros((b-r, e-c), dtype=bool)
    for rr, cc in cells:
        mask[rr-r, cc-c] = True
    return mask


def _holes(mask, diagonal=False):
    inverse = ~np.pad(np.asarray(mask, dtype=bool), 1)
    h, w = inverse.shape
    return [[(r-1, c-1) for r, c in cells]
            for cells in _components(inverse, diagonal)
            if all(0 < r < h-1 and 0 < c < w-1 for r, c in cells)]


def _connected(mask):
    return len(_components(mask)) == 1


def _perimeter(grid):
    return np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or any(not 1 <= x <= 30 for x in a.shape)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError('expected an integer ARC grid with dimensions 1..30')
    return a.astype(np.int8, copy=True)


def _classify(assembled, mapping, bg):
    out = np.full(assembled.shape, bg, dtype=np.int8)
    objects = []
    for cells in _components(assembled):
        mask = _mask_from_cells(cells)
        k = len(_holes(mask))
        for r, c in cells:
            out[r, c] = mapping.get(k, bg)
        objects.append((cells, mask, k))
    return out, objects


def _read_oriented(grid, line):
    """Parse a scene with its legend above a full-width divider."""
    bg = int(grid[0, 0])
    frame = int(grid[line, 0])
    assert frame != bg and 3 <= line < grid.shape[0]-3
    panel, work = grid[:line], grid[line+1:]
    assert not np.any(panel == frame)
    card_cells = _components(work == frame)
    assert 2 <= len(card_cells) <= 4
    cards, sizes, query_colors = [], set(), set()
    clean = work.copy()
    for cells in card_cells:
        r, b, c, e = _box(cells)
        card = work[r:b, c:e]
        assert b-r == e-c and 7 <= b-r <= 10
        n = b-r-2
        assert np.all(_perimeter(card) == frame)
        assert len(cells) == 4*(n+2)-4
        inner = card[1:-1, 1:-1]
        values = set(map(int, np.unique(inner))) - {bg}
        assert len(values) == 1 and frame not in values
        q = next(iter(values))
        mask = inner == q
        assert _connected(mask)
        assert set(map(int, np.unique(inner))) <= {q, bg}
        cards.append((r, b, c, e, mask.copy()))
        sizes.add(n)
        query_colors.add(q)
        clean[r:b, c:e] = bg
    assert len(sizes) == len(query_colors) == 1
    n, query = next(iter(sizes)), next(iter(query_colors))
    carriers = set(map(int, np.unique(clean))) - {bg}
    assert len(carriers) == 1
    carrier = next(iter(carriers))
    assert carrier not in (bg, frame, query)
    stamps = {}
    for r, b, c, e, mask in cards:
        k = len(_holes(mask))
        assert 0 <= k <= 4 and k not in stamps
        stamps[k] = mask
    tiles = []
    for cells in _components(clean == carrier):
        r, b, c, e = _box(cells)
        assert (b-r, e-c) == (n, n)
        tile = clean[r:b, c:e]
        assert np.all(_perimeter(tile) == carrier)
        assert set(map(int, np.unique(tile))) <= {carrier, bg}
        clues = tile == bg
        assert not np.any(clues[:-1] & clues[1:])
        assert not np.any(clues[:, :-1] & clues[:, 1:])
        k = int(clues.sum())
        assert k in stamps
        tiles.append((r, c, k, clues.copy()))
    assert 2 <= len(tiles) <= 6
    rs = sorted({r for r, c, k, clue in tiles})
    cs = sorted({c for r, c, k, clue in tiles})
    assert len(rs)*len(cs) == len(tiles)
    assert len({(r, c) for r, c, k, clue in tiles}) == len(tiles)
    gaps = set(int(x-n) for origins in (rs, cs) for x in np.diff(origins))
    assert gaps and gaps <= {1, 2}
    assert all(len(set(map(int, np.diff(origins)))) <= 1 for origins in (rs, cs))
    assert set(stamps) <= {k for r, c, k, clue in tiles}
    # All work foreground is accounted for; foreign pixels inside boxes fail.
    occupied = np.zeros(work.shape, dtype=bool)
    for r, b, c, e, mask in cards:
        assert not occupied[r:b, c:e].any()
        occupied[r:b, c:e] = True
    for r, c, k, clues in tiles:
        assert not occupied[r:r+n, c:c+n].any()
        occupied[r:r+n, c:c+n] = True
    assert not np.any((work != bg) & ~occupied)
    boxes = [(r, b, c, e) for r, b, c, e, mask in cards]
    boxes += [(r, r+n, c, c+n) for r, c, k, clues in tiles]
    # Cards and tiles are separated by background, including their box corners.
    for i, (r, b, c, e) in enumerate(boxes):
        for rr, bb, cc, ee in boxes[i+1:]:
            assert b < rr or bb < r or e < cc or ee < c
    mapping, entries = {}, []
    for color in set(map(int, np.unique(panel))) - {bg}:
        assert color not in (frame, carrier, query)
        comps = _components(panel == color)
        assert len(comps) == 1
        mask = _mask_from_cells(comps[0])
        k = len(_holes(mask))
        assert 0 <= k <= 6 and k not in mapping
        mapping[k] = color
        entries.append((k, color, mask))
    assert 1 <= len(mapping) <= 3
    assembled = np.zeros((len(rs)*n, len(cs)*n), dtype=bool)
    selected = np.empty((len(rs), len(cs)), dtype=int)
    for r, c, k, clues in tiles:
        i, j = rs.index(r), cs.index(c)
        assembled[i*n:(i+1)*n, j*n:(j+1)*n] = stamps[k]
        selected[i, j] = k
    output, objects = _classify(assembled, mapping, bg)
    for cells, mask, k in objects:
        r0, b, c0, e = _box(cells)
        assert all(not assembled[r0+r, c0+c]
                   for hole in _holes(mask) for r, c in hole), 'foreign foreground in a cavity'
    assert any(k in mapping for cells, mask, k in objects)
    return dict(background=bg, frame=frame, query=query, carrier=carrier,
                size=n, layout=(len(rs), len(cs)), stamps=stamps, tiles=tiles,
                selected=selected, assembled=assembled, mapping=mapping,
                entries=entries, objects=objects, output=output)


def _parse(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    assert all(int(grid[r, c]) == bg for r, c in
               ((0, 0), (0, -1), (-1, 0), (-1, -1)))
    candidates = []
    for rotation in range(4):
        oriented = np.rot90(grid, rotation)
        for line in range(1, oriented.shape[0]-1):
            if oriented[line, 0] == bg or not np.all(oriented[line] == oriented[line, 0]):
                continue
            try:
                p = _read_oriented(oriented, line)
            except (AssertionError, IndexError, ValueError):
                continue
            p['rotation'] = rotation
            p['output'] = np.rot90(p['output'], -rotation).copy()
            candidates.append(p)
    assert len(candidates) == 1, 'input needs a unique frame/legend/board interpretation'
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer using only input pixels, including its local legend."""
    return _parse(input_grid)['output'].copy()


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _locations(free, h, w):
    if h > free.shape[0] or w > free.shape[1]:
        return np.empty((0, 2), dtype=int)
    summed = np.pad((~free).astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    bad = summed[h:, w:] - summed[:-h, w:] - summed[h:, :-w] + summed[:-h, :-w]
    return np.argwhere(bad == 0)


def _reserve(free, r, c, h, w):
    free[max(0, r-1):r+h+1, max(0, c-1):c+w+1] = False


def _fallback_pack(n, rows, cols, count, gap):
    h, w = rows*n+(rows-1)*gap, cols*n+(cols-1)*gap
    if h > 22 or w > 28:
        return None
    for r, c in ((0, 0), (0, 28-w), (22-h, 0), (22-h, 28-w)):
        free = np.ones((22, 28), dtype=bool)
        _reserve(free, r, c, h, w)
        boxes = [(r, c, h, w)]
        for _ in range(count):
            positions = _locations(free, n+2, n+2)
            if not len(positions):
                break
            rr, cc = map(int, positions[0])
            boxes.append((rr, cc, n+2, n+2))
            _reserve(free, rr, cc, n+2, n+2)
        if len(boxes) == count+1:
            return boxes
    return None


@lru_cache(maxsize=4)
def _configs(layouts):
    return [(n, r, c, count, gap) for n in range(5, 9)
            for r, c in layouts for count in range(2, min(4, r*c)+1)
            for gap in (1, 2) if _fallback_pack(n, r, c, count, gap) is not None]


def _place(rng, config):
    n, rows, cols, count, gap = config
    h, w = rows*n+(rows-1)*gap, cols*n+(cols-1)*gap
    for _ in range(12):
        free = np.ones((22, 28), dtype=bool)
        r, c = int(rng.integers(23-h)), int(rng.integers(29-w))
        boxes = [(r, c, h, w)]
        _reserve(free, r, c, h, w)
        for _ in range(count):
            positions = _locations(free, n+2, n+2)
            if not len(positions):
                break
            rr, cc = map(int, positions[int(rng.integers(len(positions)))])
            boxes.append((rr, cc, n+2, n+2))
            _reserve(free, rr, cc, n+2, n+2)
        if len(boxes) == count+1:
            return boxes
    boxes = _fallback_pack(*config)
    assert boxes is not None
    return boxes


def _bay_stamps(n, rng):
    """Two open bays close across a seam, adding one previously absent hole."""
    left, right = np.zeros((n, n), bool), np.zeros((n, n), bool)
    left[1:-1, 1:] = True
    right[1:-1, :-1] = True
    r = n//2
    depth = int(rng.integers(1, 3))
    height = int(rng.integers(1, min(2, n-2-r)+1))
    left[r:r+height, n-depth:] = False
    right[r:r+height, :depth] = False
    right[2, n-3] = False
    assert _connected(left) and _connected(right)
    assert len(_holes(left)) == 0 and len(_holes(right)) == 1
    return {0: left, 1: right}


def _topology_stamps(rng=None):
    """Procedural teaching shapes: open bay, large hole, diagonal two holes."""
    base = np.zeros((6, 6), bool)
    base[1:5, :] = True
    a, b, c = base.copy(), base.copy(), base.copy()
    a[2:4, :2] = False
    b[2:4, 1:5] = False
    c[2, 1] = c[3, 2] = c[3, 3] = c[1, 0] = False
    if rng is not None:
        # Keep the teaching inequalities while varying the actual contours.
        # Top and bottom background rows make every variant portless along
        # the selector strip's vertical assembly direction.
        for _ in range(int(rng.integers(4))):
            cavity = _holes(b)[0]
            trial = b.copy()
            trial[_choose(rng, cavity)] = True
            if _connected(trial) and len(_holes(trial)) == 1:
                b = trial
        depth = int(rng.integers(1, 4))
        a = base.copy()
        a[2:4, :depth] = False
        if rng.random() < .5:
            c = base.copy()
            c[2, 2] = c[3, 3] = c[3, 4] = False
            c[1, :2] = False
        variants = []
        for m in (a, b, c):
            if rng.random() < .5:
                m = m[::-1]
            if rng.random() < .5:
                m = m[:, ::-1]
            variants.append(m.copy())
        a, b, c = variants
    assert [len(_holes(m)) for m in (a, b, c)] == [0, 1, 2]
    assert all(_connected(m) for m in (a, b, c))
    return {0: a, 1: b, 2: c}


def _random_stamp(rng, n, k):
    # An intact rectangular core permits controlled hole growth. Exterior bays
    # and edge margins independently change whether neighboring copies sew.
    for _ in range(12):
        margins = list(map(int, rng.integers(0, 2, 4)))
        r, b, c, e = margins[0], n-margins[1], margins[2], n-margins[3]
        mask = np.zeros((n, n), bool)
        mask[r:b, c:e] = True
        if k == 1 and rng.random() < .25:
            mask[r+1:b-1, c+1:e-1] = False
        else:
            cells = [(rr, cc) for rr in range(r+1, b-1) for cc in range(c+1, e-1)]
            for index in rng.permutation(len(cells)):
                if len(_holes(mask)) == k:
                    break
                trial = mask.copy()
                trial[cells[int(index)]] = False
                if _connected(trial) and len(_holes(trial)) == len(_holes(mask))+1:
                    mask = trial
        if len(_holes(mask)) != k:
            continue
        # Grow cavities without merging them or disconnecting the stamp.
        for _ in range(int(rng.integers(n*n//3+1))):
            holes = {p for cells in _holes(mask) for p in cells}
            candidates = sorted({(rr+dr, cc+dc) for rr, cc in holes for dr, dc in _FOUR
                                 if r < rr+dr < b-1 and c < cc+dc < e-1
                                 and mask[rr+dr, cc+dc]})
            if not candidates:
                break
            trial = mask.copy()
            trial[_choose(rng, candidates)] = False
            if _connected(trial) and len(_holes(trial)) == k:
                mask = trial
        for _ in range(int(rng.integers(5))):
            trial = mask.copy()
            side, reach = int(rng.integers(4)), int(rng.integers(1, n//2+1))
            p = int(rng.integers(n))
            if side == 0:
                trial[:reach, p] = False
            elif side == 1:
                trial[n-reach:, p] = False
            elif side == 2:
                trial[p, :reach] = False
            else:
                trial[p, n-reach:] = False
            if _connected(trial) and len(_holes(trial)) == k:
                mask = trial
        return np.rot90(mask, int(rng.integers(4))).copy()
    # A full square with spaced single-cell holes is feasible for n>=5, k<=4.
    mask = np.ones((n, n), bool)
    spots = [(r, c) for r in range(1, n-1, 2) for c in range(1, n-1, 2)]
    for index in rng.permutation(len(spots))[:k]:
        mask[spots[int(index)]] = False
    assert _connected(mask) and len(_holes(mask)) == k
    return mask


def _clues(rng, n, k):
    if k == 0:
        return np.zeros((n, n), bool)
    cells = [(r, c) for r in range(1, n-1) for c in range(1, n-1)]
    for _ in range(16):
        mask = np.zeros((n, n), bool)
        for index in rng.permutation(len(cells)):
            r, c = cells[int(index)]
            if any(mask[r+dr, c+dc] for dr, dc in _FOUR):
                continue
            trial = mask.copy()
            trial[r, c] = True
            if _connected(~trial):
                mask = trial
            if int(mask.sum()) == k:
                return mask
    mask = np.zeros((n, n), bool)
    for r, c in [(1, 1), (1, n-2), (n-2, 1), (n-2, n-2)][:k]:
        mask[r, c] = True
    assert int(mask.sum()) == k and _connected(~mask)
    return mask


def _glyph(k):
    mask = np.ones((3, max(3, 2*k+1)), bool)
    mask[1, 1:2*k:2] = False
    return mask


def _assemble(stamps, selection):
    n = next(iter(stamps.values())).shape[0]
    return np.block([[stamps[int(k)] for k in row] for row in selection])


def _render(rng, config, stamps, selection, keys, roles, rotation):
    n, rows, cols, count, gap = config
    bg, frame, query, carrier = roles[:4]
    colors = roles[4:4+len(keys)]
    mapping = dict(zip(keys, colors))
    # Labels are constructed from the chosen source masks and board selection.
    output, _ = _classify(_assemble(stamps, selection), mapping, bg)
    body = np.full((22, 28), bg, np.int8)
    boxes = _place(rng, config)
    r, c, h, w = boxes[0]
    for i in range(rows):
        for j in range(cols):
            rr, cc = r+i*(n+gap), c+j*(n+gap)
            tile = np.full((n, n), carrier, np.int8)
            tile[_clues(rng, n, int(selection[i, j]))] = bg
            body[rr:rr+n, cc:cc+n] = tile
    order = list(stamps)
    rng.shuffle(order)
    for (r, c, h, w), k in zip(boxes[1:], order):
        body[r:r+h, c:c+w] = frame
        body[r+1:r+h-1, c+1:c+w-1] = np.where(stamps[k], query, bg)
    r0 = min(x[0] for x in boxes)
    c0 = min(x[1] for x in boxes)
    r1 = max(x[0]+x[2] for x in boxes)
    c1 = max(x[1]+x[3] for x in boxes)
    r0 = int(rng.integers(r0+1))
    c0 = int(rng.integers(c0+1))
    r1 = int(rng.integers(r1, 23))
    c1 = int(rng.integers(c1, 29))
    body = body[r0:r1, c0:c1]
    glyphs = [_glyph(k) for k in keys]
    key_width = sum(m.shape[1] for m in glyphs)+len(keys)-1
    width = max(body.shape[1]+2, key_width+2)
    grid = np.full((body.shape[0]+8, width), bg, np.int8)
    grid[5] = frame
    start = int(rng.integers(width-body.shape[1]-1))+1
    grid[7:7+body.shape[0], start:start+body.shape[1]] = body
    c = int(rng.integers(width-key_width-1))+1
    for glyph, color in zip(glyphs, colors):
        r = int(rng.integers(1, 3))
        grid[r:r+3, c:c+glyph.shape[1]][glyph] = color
        c += glyph.shape[1]+1
    return {'input': np.rot90(grid, rotation).copy(),
            'output': np.rot90(output, rotation).copy()}


def _random_pair(rng, config, rotation, roles):
    n, rows, cols, count, gap = config
    counts = [0]+list(map(int, rng.choice([1, 2, 3, 4], count-1, replace=False)))
    for attempt in range(16):
        stamps = {k: _random_stamp(rng, n, k) for k in counts}
        if attempt == 15:
            # Isolated zero-hole copies guarantee a listed class can survive.
            stamps[0] = np.pad(np.ones((n-2, n-2), bool), 1)
        flat = counts+[int(_choose(rng, counts)) for _ in range(rows*cols-count)]
        rng.shuffle(flat)
        selection = np.array(flat).reshape(rows, cols)
        _, objects = _classify(_assemble(stamps, selection), {}, roles[0])
        present = sorted({k for cells, mask, k in objects})
        eligible = [k for k in present if k <= 6]
        if eligible:
            break
    assert eligible
    rng.shuffle(eligible)
    keys = eligible[:int(rng.integers(1, min(3, len(eligible))+1))]
    if len(keys) < 3 and rng.random() < .35:
        unused = [k for k in range(7) if k not in present and k not in keys]
        if unused:
            keys.append(int(_choose(rng, unused)))
    # Keep the miniature legend within 28 cells, without changing its meaning.
    while sum(_glyph(k).shape[1] for k in keys)+len(keys)-1 > 28:
        keys.pop()
    rng.shuffle(keys)
    return _render(rng, config, stamps, selection, keys, roles, rotation)


def _check_counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10),
                                   ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral)
                or not low <= value <= high):
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _unique_recolor(pair, seen):
    # A finite fallback: all 90 images of two present colors, more than 13
    # earlier inputs, preserve the complete spatial and topological rule.
    first = list(map(int, np.unique(pair['input'])))[:2]
    order = first+[c for c in range(10) if c not in first]
    for a in range(10):
        for b in range(10):
            if a == b:
                continue
            target = [a, b]+[c for c in range(10) if c not in (a, b)]
            lookup = np.empty(10, np.int8)
            lookup[order] = target
            new = {k: lookup[v] for k, v in pair.items()}
            if _key(new['input']) not in seen:
                return new
    raise AssertionError('finite uniqueness bound failed')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    train_configs = _configs(((1, 2), (2, 1), (1, 3), (3, 1), (2, 2)))
    # The test introduces a new adjacency layout, either a four-stamp strip
    # or a six-stamp rectangle. No output convention changes with the split.
    shift = int(rng.integers(2))
    test_configs = _configs(((1, 4), (4, 1)) if shift == 0 else ((2, 3), (3, 2)))
    base_roles = list(map(int, rng.permutation(10)))
    stable_roles = rng.random() < .25
    base_rotation = int(rng.integers(4))
    stable_rotation = rng.random() < .3
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, total in (('train', num_train), ('test', num_test)):
        for i in range(total):
            roles = base_roles if stable_roles else list(map(int, rng.permutation(10)))
            rotation = base_rotation if stable_rotation else int(rng.integers(4))
            if split == 'train' and i == 0:
                n = int(rng.integers(6, 9))
                config = (n, 2, 2, 2, 1)
                stamps = _bay_stamps(n, rng)
                pair = _render(rng, config, stamps, np.array([[0, 1], [1, 0]]),
                               list(map(int, rng.permutation([0, 2]))), roles, rotation)
            elif split == 'train' and i == 1:
                selection = rng.permutation(3).reshape(3, 1)
                keys = list(map(int, rng.permutation([0, 2])))
                pair = _render(rng, (6, 3, 1, 3, 1), _topology_stamps(rng),
                               selection, keys, roles, rotation)
            else:
                config = _choose(rng, train_configs if split == 'train' else test_configs)
                pair = _random_pair(rng, config, rotation, roles)
            if _key(pair['input']) in seen:
                pair = _unique_recolor(pair, seen)
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    # Shuffle teaching examples, preserving their guaranteed aggregate evidence.
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _training_evidence(parsed):
    flags = dict(seam=False, erasure=False, retained=set(), area=False,
                 diagonal=False, orientation=False, nonpositional=False)
    for p in parsed:
        n = p['size']
        for cells, mask, k in p['objects']:
            tile_ids = {(r//n, c//n) for r, c in cells}
            local_sum = sum(int(p['selected'][i, j]) for i, j in tile_ids)
            if len(tile_ids) > 1 and k > local_sum and k in p['mapping']:
                flags['seam'] = True
            if k not in p['mapping']:
                flags['erasure'] = True
            else:
                flags['retained'].add(k)
        areas = {k: sum(map(len, _holes(m))) for k, m in p['stamps'].items()}
        flags['area'] |= any(a < b and areas[a] > areas[b] for a in areas for b in areas)
        for k, m in p['stamps'].items():
            flags['diagonal'] |= len(_holes(m, True)) != k
            transforms = [np.rot90(m, i) for i in range(4)]
            transforms += [np.rot90(m.T, i) for i in range(4)]
            flags['orientation'] |= all(not np.array_equal(m, t) for t in transforms[1:])
        for r, c, k, clues in p['tiles']:
            hole_mask = np.zeros((n, n), bool)
            for hole in _holes(p['stamps'][k]):
                for rr, cc in hole:
                    hole_mask[rr, cc] = True
            flags['nonpositional'] |= k > 0 and not np.array_equal(hole_mask, clues)
    return flags


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, train_parsed = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            p = _parse(inp)
            assert np.array_equal(out, p['output']), 'incorrect sewing, topology, erasure, or crop'
            assert out.size < inp.size
            # Classification colors only foreground; it never fills a cavity.
            oriented = np.rot90(out, p['rotation'])
            assert np.all(oriented[~p['assembled']] == p['background'])
            for cells, mask, k in p['objects']:
                assert all(oriented[r, c] == p['mapping'].get(k, p['background']) for r, c in cells)
            if split == 'train':
                train_parsed.append(p)
    flags = _training_evidence(train_parsed)
    assert all(flags[k] for k in ('seam', 'erasure', 'area', 'diagonal', 'orientation', 'nonpositional')), flags
    assert len(flags['retained']) >= 2


def diversity_features(pair):
    p = _parse(pair['input'])
    return dict(shape=tuple(np.asarray(pair['input']).shape), size=p['size'],
                layout=p['layout'], rotation=p['rotation'],
                roles=(p['background'], p['frame'], p['query'], p['carrier']),
                source_counts=tuple(sorted(p['stamps'])),
                selected=tuple(map(tuple, p['selected'].tolist())),
                classes=tuple(sorted(k for cells, mask, k in p['objects'])),
                keys=tuple(sorted(p['mapping'])),
                erased=sum(k not in p['mapping'] for cells, mask, k in p['objects']),
                stamp_shapes=tuple((k, m.tobytes()) for k, m in sorted(p['stamps'].items())),
                cavity_areas=tuple(sorted(len(hole) for m in p['stamps'].values() for hole in _holes(m))))


def self_test():
    """Bounded semantic regressions and structural variation over 42 seeds."""
    # Hand-calculated seam fixture: [0,1] makes TWO holes, not 0+1.
    stamps = _bay_stamps(6, np.random.default_rng(19))
    joined = _assemble(stamps, np.array([[0, 1], [1, 0]]))
    out, objects = _classify(joined, {0: 4, 2: 7}, 0)
    assert sorted(k for cells, mask, k in objects) == [0, 1, 2]
    assert np.all(out[:6][joined[:6]] == 7)
    assert not np.any(out[6:, :6]) and np.any(out[6:, 6:] == 4)
    t = _topology_stamps()
    assert sum(map(len, _holes(t[1]))) == 8
    assert sum(map(len, _holes(t[2]))) == 3
    assert len(_holes(t[2], True)) != 2
    features, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for s in puzzle for p in puzzle[s]))
        local = []
        for split in puzzle:
            for pair in puzzle[split]:
                before = np.asarray(pair['input']).copy()
                assert np.array_equal(solve(before), pair['output'])
                assert np.array_equal(before, pair['input'])
                local.append(diversity_features(pair))
        assert len({f['layout'] for f in local}) >= 3
        features.extend(local)
    assert len(signatures) == 42
    assert len({f['shape'] for f in features}) >= 20
    assert len({f['size'] for f in features}) == 4
    assert len({f['layout'] for f in features}) >= 7
    assert len({f['roles'] for f in features}) >= 30
    assert len({f['source_counts'] for f in features}) >= 8
    assert len({f['classes'] for f in features}) >= 15
    assert len({f['stamp_shapes'] for f in features}) >= 100
    assert len({f['selected'] for f in features}) >= 50
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(91, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (2.5, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    a, b = generate_puzzle(99, 2, 1), generate_puzzle(99, 2, 1)
    assert all(np.array_equal(p[k], q[k]) for s in a for p, q in zip(a[s], b[s]) for k in p)
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('wrong answer accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds and every legal count combination)')
