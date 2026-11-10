"""Counted spillways: exterior object counts construct a shared liquid network.

Only NumPy and the standard library are used. No files or parent code are read.
Generation retains one puzzle at a time; self_test writes no artifacts or traces.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np


_DIAMOND = ((0, 1), (1, 0), (1, 2), (2, 1))
_SQUARE = ((0, 0), (0, 1), (1, 0), (1, 1))
_SINGLE = ((0, 0),)
_X = ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2))


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape):
        raise ValueError("grids must have two dimensions in 1..30")
    if not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9)):
        raise ValueError("grid colors must be integers in 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
             if (dr or dc) and (diagonal or abs(dr) + abs(dc) == 1)]
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, part = [start], []
        while stack:
            r, c = stack.pop()
            part.append((int(r), int(c)))
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(part)
    return result


def _parse(input_grid):
    """Recover every semantic role from the one visible input."""
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    if not all(np.all(edge == bg) for edge in (grid[0], grid[-1], grid[:, 0], grid[:, -1])):
        raise ValueError("the exterior margin must be blank")
    candidates = []
    for color in np.unique(grid):
        if color == bg:
            continue
        rr, cc = np.where(grid == color)
        r0, r1 = int(rr.min()), int(rr.max()) + 1
        c0, c1 = int(cc.min()), int(cc.max()) + 1
        tank = grid[r0:r1, c0:c1]
        if min(tank.shape) < 7 or not (r0 > 0 and c0 > 0 and r1 < grid.shape[0] and c1 < grid.shape[1]):
            continue
        if all(np.all(edge == color) for edge in (tank[0], tank[-1], tank[:, 0], tank[:, -1])):
            candidates.append((int(color), (r0, r1, c0, c1), tank.copy()))
    if len(candidates) != 1:
        raise ValueError("reservoir frame must be unique")
    wall, bounds, tank = candidates[0]
    field = int(tank[1, 1])
    if field in (bg, wall) or tank[1, -2] != field:
        raise ValueError("both upper interior corners must show the field")
    keys, liquids = {}, []
    for color in np.unique(tank):
        color = int(color)
        if color in (field, wall):
            continue
        points = np.argwhere(tank == color)
        if len(points) == 1:
            keys[color] = tuple(map(int, points[0]))
        else:
            liquids.append(color)
    if len(liquids) != 1 or not 2 <= len(keys) <= 4 or bg in keys or bg in liquids:
        raise ValueError("need singleton dam keys and one repeated liquid color")
    water = liquids[0]
    r0, r1, c0, c1 = bounds
    outside = np.ones(grid.shape, bool)
    outside[r0:r1, c0:c1] = False
    ring = np.zeros(grid.shape, bool)
    ring[r0-1:r1+1, c0-1:c1+1] = True
    ring[r0:r1, c0:c1] = False
    if np.any(grid[ring] != bg) or np.any(grid[outside] == field) or np.any(grid[outside] == water):
        raise ValueError("reservoir roles must be distinct from exterior objects")
    counts, objects = {}, {}
    for color in sorted(set(map(int, grid[outside])) - {bg}):
        objects[color] = _components((grid == color) & outside, diagonal=True)
        counts[color] = len(objects[color])
    built = tank.copy()
    columns = []
    for color, (r, c) in keys.items():
        n = counts.get(color, 0)
        if not 1 <= n <= 6 or r - n + 1 < 2 or c in columns:
            raise ValueError("invalid dam count, height, or column")
        if not np.all(tank[r+1:, c] == wall) or not np.all(tank[r-n+1:r, c] == field):
            raise ValueError("a dam needs a grounded foundation and clear extension")
        built[r-n+1:r+1, c] = color
        columns.append(c)
    columns.sort()
    intervals = [(a + 1, b) for a, b in zip([0] + columns, columns + [tank.shape[1] - 1])]
    if any(a >= b for a, b in intervals):
        raise ValueError("compartments must have positive width")
    source = tank == water
    source_sections = [i for i, (a, b) in enumerate(intervals) if source[:, a:b].any()]
    if len(source_sections) != 1:
        raise ValueError("all supply must start in one compartment")
    solid = (built != field) & (built != water)
    return dict(grid=grid, background=bg, bounds=bounds, tank=tank, wall=wall,
                field=field, water=water, keys=keys, counts=counts, objects=objects,
                built=built, solid=solid, source=source, intervals=intervals,
                source_section=source_sections[0])


def _equilibrium(solid, columns, source_section, volume):
    """Exact basin-capacity filling, with every equal-height spill tree checked.

    A leaf is a flat-bottom compartment. The highest internal dam splits a
    range into two subbasins. Fill the entry side up to that spill height,
    route excess into the other side, then raise the joined basin. Equal-height
    dams have up to 14 binary spill trees (four dams); all must give the same
    complete-level answer. This avoids intermediate partial-cell pour artifacts.
    """
    h, w = solid.shape
    columns = sorted(columns)
    intervals = [(a+1, b) for a, b in zip([0]+columns, columns+[w-1])]
    ground = solid[1:].sum(axis=0)
    if np.any(solid[1:-1] & ~solid[2:]):
        raise ValueError("terrain must be grounded")
    for a, b in intervals:
        if len(set(map(int, ground[a:b]))) != 1:
            raise ValueError("each compartment must have a flat floor")
    heights = [int(ground[c]) for c in columns]
    for i, (a, b) in enumerate(intervals):
        adjacent = heights[max(0, i-1):min(len(heights), i+1)]
        if adjacent and int(ground[a]) > min(adjacent):
            raise ValueError("a floor cannot stand above an adjacent dam")
    cache = {}

    def trees(lo, hi):
        if (lo, hi) in cache:
            return cache[lo, hi]
        if lo == hi:
            result = [(lo, hi, None)]
        else:
            tallest = max(heights[lo:hi])
            result = [(lo, hi, i, left, right) for i in range(lo, hi) if heights[i] == tallest
                      for left in trees(lo, i) for right in trees(i+1, hi)]
        cache[lo, hi] = result
        return result

    masks = {}

    def full(lo, hi, head):
        key = (lo, hi, head)
        if key not in masks:
            result = np.zeros(solid.shape, bool)
            result[max(1, h-head):, intervals[lo][0]:intervals[hi][1]] = True
            masks[key] = result & ~solid
        return masks[key]

    def fill(tree, entry, amount, ceiling):
        lo, hi, split = tree[:3]
        if amount == 0:
            return np.zeros(solid.shape, bool)
        cap = int(full(lo, hi, ceiling).sum())
        if amount > cap:
            raise ValueError("reservoir overflow")
        if split is None:
            for head in range(ceiling+1):
                mask = full(lo, hi, head)
                if int(mask.sum()) == amount:
                    return mask.copy()
            raise ValueError("partial water level")
        head = heights[split]
        left, right = tree[3:]
        if entry <= split:
            first, second, other_entry = left, right, split+1
        else:
            first, second, other_entry = right, left, split
        fcap = int(full(first[0], first[1], head).sum())
        scap = int(full(second[0], second[1], head).sum())
        if amount <= fcap:
            return fill(first, entry, amount, head)
        first_mask = full(first[0], first[1], head)
        if amount <= fcap+scap:
            return first_mask | fill(second, other_entry, amount-fcap, head)
        for level in range(head+1, ceiling+1):
            mask = full(lo, hi, level)
            if int(mask.sum()) == amount:
                return mask.copy()
        raise ValueError("partial joined-basin level")

    answers = [fill(tree, source_section, int(volume), h-1)
               for tree in trees(0, len(columns))]
    if not answers or any(not np.array_equal(answers[0], a) for a in answers[1:]):
        raise ValueError("equal-height spillways have competing allocations")
    return answers[0].copy()


def _settle(source, solid, columns):
    columns = sorted(columns)
    intervals = [(a+1, b) for a, b in zip([0]+columns, columns+[solid.shape[1]-1])]
    entries = [i for i, (a, b) in enumerate(intervals) if source[:, a:b].any()]
    if len(entries) != 1:
        raise ValueError("supply must occupy one compartment")
    return _equilibrium(solid, columns, entries[0], int(source.sum()))


def _closed(pool, solid):
    """A pool fills its entire reachable basin below its flat surface."""
    rows = np.arange(solid.shape[0])[:, None]
    for cells in _components(pool):
        top = min(r for r, c in cells)
        available = ~solid & (rows >= top)
        seen, stack = {cells[0]}, [cells[0]]
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= q[0] < solid.shape[0] and 0 <= q[1] < solid.shape[1] and available[q] and q not in seen:
                    seen.add(q)
                    stack.append(q)
        if seen != set(cells):
            return False
    return True


def solve(input_grid: np.ndarray) -> np.ndarray:
    m = _parse(input_grid)
    water = _settle(m['source'], m['solid'], [c for r, c in m['keys'].values()])
    if not _closed(water, m['solid']):
        raise ValueError("ambiguous partial water level outside the family")
    result = m['built'].copy()
    result[result == m['water']] = m['field']
    result[water] = m['water']
    return result


def _shape(rng):
    mode = int(rng.integers(6))
    if mode == 0:
        cells = _DIAMOND
    elif mode == 1:
        cells = _X
    elif mode == 2:
        cells = _SQUARE
    else:
        cells = {(1, 1)}
        target = int(rng.integers(1, 9))
        while len(cells) < target:
            frontier = sorted({(r+dr, c+dc) for r, c in cells
                               for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                               if 0 <= r+dr < 3 and 0 <= c+dc < 3} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    cells = list(cells)
    for _ in range(int(rng.integers(4))):
        cells = [(-c, r) for r, c in cells]
    if rng.integers(2):
        cells = [(r, -c) for r, c in cells]
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _tank(rng, colors, keys, counts, mode, fallback=False):
    bg, field, wall, water = colors[:4]
    k = len(keys)
    h = 12 if fallback else int(rng.integers(11, 15))
    widths = [2] * (k + 1) if fallback else list(map(int, rng.integers(1, 5, size=k+1)))
    source_section = 0 if mode == 'retain' or fallback else int(rng.integers(k+1))
    widths[source_section] = max(2, widths[source_section])
    w = sum(widths) + k + 2
    # Bound width while retaining all compartments and a useful source width.
    while w > 27:
        i = int(np.argmax(widths))
        widths[i] -= 1
        w -= 1
    tank = np.full((h, w), field, np.int8)
    tank[0] = tank[-1] = wall
    tank[:, 0] = tank[:, -1] = wall
    intervals, columns, left = [], [], 1
    for i, width in enumerate(widths):
        intervals.append((left, left+width))
        floor = h-1 if fallback or (mode == 'retain' and i == 0) else h-1-int(rng.integers(3))
        for c in range(left, left+width):
            tank[floor:, c] = wall
        left += width
        if i < k:
            columns.append(left)
            base = h-2 if fallback else h-3-int(rng.integers(2))
            tank[base+1:, left] = wall
            tank[base, left] = keys[i]
            left += 1
    built = tank.copy()
    for color, n, c in zip(keys, counts, columns):
        r = int(np.where(tank[:, c] == color)[0][0])
        built[r-n+1:r+1, c] = color
    solid = built != field
    a, b = intervals[source_section]
    allowed = np.zeros(tank.shape, bool)
    allowed[:, a:b] = ~solid[:, a:b]
    # Both upper interior corners explicitly encode the field role.
    allowed[1, 1] = allowed[1, -2] = False
    feed = (1, (a+b-1)//2)
    if feed in ((1, 1), (1, w-2)):
        feed = (1, a+1 if a == 1 else a)
    bare_solid = tank != field
    candidates = []
    for volume in range(2, int(allowed.sum()) + 1):
        try:
            target = _equilibrium(solid, columns, source_section, volume)
        except ValueError:
            continue
        if not _closed(target, solid) or _closed(target, bare_solid):
            continue
        sections = sum(bool(target[:, x:y].any()) for x, y in intervals)
        if (mode == 'retain' and sections != 1) or (mode == 'spill' and sections < 2):
            continue
        candidates.append((volume, target.copy()))
    if not candidates:
        return None
    volume, target = candidates[int(rng.integers(len(candidates)))]
    points = np.argwhere(allowed)
    anchor = np.array(feed)
    dr, dc = np.abs(points-anchor).T
    style = int(rng.integers(4))
    jitter = rng.random(len(points))
    score = (dr+dc+jitter if style == 0 else jitter if style == 1 else
             3*dc+.1*dr+jitter if style == 2 else 3*(~target[points[:, 0], points[:, 1]])+jitter)
    chosen = points[np.argsort(score)[:volume]].copy()
    if not np.any(np.all(chosen == anchor, axis=1)):
        chosen[-1] = anchor
    source = np.zeros(tank.shape, bool)
    source[chosen[:, 0], chosen[:, 1]] = True
    assert np.array_equal(_settle(source, solid, columns), target)
    tank[source] = water
    output = built.copy()
    output[target] = water
    return tank, output


def _scene(rng, tank, output, colors, keys, counts, witness, distract_count):
    bg, field, wall, water = colors[:4]
    objects = []
    for i, (color, n) in enumerate(zip(keys, counts)):
        for j in range(n):
            shape = (_DIAMOND, _SQUARE, _SINGLE)[j] if witness and i == 0 and j < 3 else _shape(rng)
            objects.append((color, shape))
    distractor = int(rng.choice(colors[4+len(keys):]))
    for _ in range(distract_count):
        objects.append((distractor, _shape(rng)))
    h, w = tank.shape
    possible = [cols for cols in range(3, 8)
                if 4*math.ceil(len(objects)/cols) + h + 3 <= 30]
    cols = int(rng.choice(possible))
    rows = math.ceil(len(objects)/cols)
    band = 4*rows
    canvas_h = h + band + 3
    canvas_w = max(w+2, 4*cols+1)
    canvas_h += int(rng.integers(min(2, 30-canvas_h)+1))
    canvas_w += int(rng.integers(min(2, 30-canvas_w)+1))
    grid = np.full((canvas_h, canvas_w), bg, np.int8)
    above = bool(rng.integers(2))
    r0 = band+2 if above else 1
    c0 = int(rng.integers(1, canvas_w-w))
    grid[r0:r0+h, c0:c0+w] = tank
    band_start = 1 if above else h+2
    slots = [(band_start+4*r, 1+4*c) for r in range(rows) for c in range(cols)]
    rng.shuffle(slots)
    for (color, shape), (r, c) in zip(objects, slots):
        sh, sw = 1+max(x for x, y in shape), 1+max(y for x, y in shape)
        r += int(rng.integers(4-sh))
        c += int(rng.integers(4-sw))
        for dr, dc in shape:
            grid[r+dr, c+dc] = color
    if rng.integers(2):
        grid, output = np.fliplr(grid).copy(), np.fliplr(output).copy()
    return {'input': grid, 'output': output}


def _make_pair(rng, colors, k, cap, mode, witness=False, novel_count=False):
    keys = colors[4:4+k]
    counts = list(map(int, rng.integers(1, cap+1, size=k)))
    counts[0] = max(3 if witness else 2, counts[0])
    if novel_count:
        counts[int(rng.integers(k))] = 5
    while sum(counts) > 16:
        eligible = [i for i, n in enumerate(counts) if n > (3 if witness and i == 0 else 2 if i == 0 else 1)]
        counts[int(rng.choice(eligible))] -= 1
    for attempt in range(25):
        result = _tank(rng, colors, keys, counts, mode, fallback=attempt == 24)
        if result is not None:
            tank, output = result
            return _scene(rng, tank, output, colors, keys, counts, witness,
                          int(rng.integers(3)))
    raise AssertionError("constructive reservoir fallback failed")


def _resolve_counts(rng, num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (not _integer(value) or not lo <= int(value) <= hi):
            raise ValueError("illegal example count")
    if num_train is None:
        num_train = int(rng.integers(max(2, (num_test or 0)+1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train-1)+1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _key(grid):
    return grid.shape, np.ascontiguousarray(grid).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    nt, nq = _resolve_counts(rng, num_train, num_test)
    # All roles are encoded locally. A palette is shared for visual economy;
    # category membership can change per pair without a color dictionary.
    palette = list(map(int, rng.permutation(10)))
    shift = int(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, n in (('train', nt), ('test', nq)):
        for i in range(n):
            witness = split == 'train' and i == 0
            mode = 'retain' if witness else 'spill' if split == 'test' or i == 1 else str(rng.choice(['retain', 'spill']))
            k = 2 if witness else 3 if split == 'train' and i == 1 else int(rng.integers(2, 4))
            if split == 'test' and shift == 0:
                k = 4
            colors = palette.copy()
            # Stable field/liquid roles within a puzzle, locally visible keys.
            tail = colors[4:9]
            rng.shuffle(tail)
            colors[4:9] = tail
            for _ in range(8):
                pair = _make_pair(rng, colors, k, 5 if split == 'test' and shift == 1 else 4,
                                  mode, witness, novel_count=split == 'test' and shift == 1)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise AssertionError("could not construct a distinct input")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(inp, out=None):
    m = _parse(inp)
    expected = solve(inp)
    if out is not None:
        assert np.array_equal(_grid(out), expected), "incorrect reservoir output"
    settled = expected == m['water']
    assert int(settled.sum()) == int(m['source'].sum()) >= 2, "volume changed"
    assert np.any(settled != m['source']), "liquid did not move"
    assert int(np.argwhere(settled)[:, 0].sum()) > int(np.argwhere(m['source'])[:, 0].sum())
    old_terrain = (m['tank'] != m['field']) & (m['tank'] != m['water'])
    assert np.array_equal(expected[old_terrain], m['tank'][old_terrain]), "old terrain changed"
    changed = expected != m['tank']
    growth = (m['tank'] == m['field']) & (m['built'] != m['field'])
    assert np.all(~changed | growth | (expected == m['water']) | (m['tank'] == m['water']))
    assert not np.any(growth & m['source']), "a dam erased liquid"
    assert _closed(settled, m['solid'])
    for color in m['keys']:
        assert int((expected == color).sum()) == m['counts'][color], "incorrect dam height"
    bare = (m['tank'] != m['field']) & (m['tank'] != m['water'])
    assert not _closed(settled, bare), "extended dams are not needed to contain these pools"
    # Exterior components of different colors also have a blank halo.
    owner = np.full(m['grid'].shape, -1, int)
    serial = 0
    for parts in m['objects'].values():
        for part in parts:
            for point in part:
                owner[point] = serial
            serial += 1
    for r, c in np.argwhere(owner >= 0):
        neighbors = owner[max(0, r-1):r+2, max(0, c-1):c+2]
        assert np.all((neighbors < 0) | (neighbors == owner[r, c])), "exterior objects touch"
    areas = [[len(part) for part in m['objects'][color]] for color in m['keys']]
    diagonal = any(len(_components((m['grid'] == color) & _outside(m), False)) != m['counts'][color]
                   for color in m['keys'])
    sections = sum(bool(settled[:, a:b].any()) for a, b in m['intervals'])
    glyphs = []
    for color in m['keys']:
        for part in m['objects'][color]:
            r0, c0 = min(r for r, c in part), min(c for r, c in part)
            glyphs.append(tuple(sorted((r-r0, c-c0) for r, c in part)))
    return dict(meta=m, sections=sections, volume=int(settled.sum()), diagonal=diagonal,
                mixed=any(len(set(a)) > 1 for a in areas), glyphs=glyphs,
                pools=len(_components(settled)), levels=tuple(sorted({min(r for r, c in part) for part in _components(settled)})))


def _outside(m):
    result = np.ones(m['grid'].shape, bool)
    r0, r1, c0, c1 = m['bounds']
    result[r0:r1, c0:c1] = False
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, train, tests, roles = set(), [], [], set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            f = _features(inp, out)
            m = f['meta']
            roles.add((m['background'], m['field'], m['wall'], m['water']))
            (train if split == 'train' else tests).append(f)
    assert len(roles) == 1, "inconsistent puzzle palette roles"
    assert any(f['sections'] == 1 for f in train), "missing retention demonstration"
    assert any(f['sections'] >= 2 for f in train), "missing overflow demonstration"
    assert any(f['diagonal'] and f['mixed'] and set((_DIAMOND, _SQUARE, _SINGLE)) <= set(f['glyphs'])
               for f in train), "missing object-count witness"
    assert all(f['sections'] >= 2 for f in tests)
    assert len({len(f['meta']['keys']) for f in train}) >= 2


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        h.update(bytes([len(puzzle[split])]))
        for p in puzzle[split]:
            for name in ('input', 'output'):
                a = _grid(p[name])
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 sequential seeds; bounded memory, no files, no saved execution traces."""
    # Hand-calculated overflow: a two-cell-wide source cup holds six cells
    # below its height-four rim; the remaining two fill its neighbor's floor.
    solid = np.zeros((12, 10), bool)
    solid[0] = solid[-1] = True
    solid[:, 0] = solid[:, -1] = True
    solid[8:, 3] = True
    solid[10:, 6] = True
    expected = np.zeros(solid.shape, bool)
    expected[8:11, 1:3] = True
    expected[10, 4:6] = True
    assert np.array_equal(_equilibrium(solid, [3, 6], 0, 8), expected)
    # Equal rims with insufficient supply to fill both sides are rejected.
    solid[8:, 6] = True
    try:
        _equilibrium(solid, [3, 6], 1, 8)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous equal-rim allocation accepted")
    equal = _equilibrium(solid, [3, 6], 1, 18)
    assert int(equal.sum()) == 18 and _closed(equal, solid)
    # Exercise the guaranteed construction directly, including fitting count
    # six, even though ordinary puzzle sampling currently extrapolates to five.
    for k in (2, 3, 4):
        for first_count in range(2, 7):
            for mode in ('retain', 'spill'):
                rng = np.random.default_rng(1000*k + 10*first_count + (mode == 'spill'))
                colors = list(range(10))
                result = _tank(rng, colors, colors[4:4+k],
                               [first_count] + [1]*(k-1), mode, fallback=True)
                assert result is not None, "flat-floor fallback failed"
                tank, output = result
                source, solid = tank == 3, (output != 1) & (output != 3)
                assert _closed(output == 3, solid)
                columns = [int(np.where(tank == color)[1][0]) for color in colors[4:4+k]]
                assert np.array_equal(_settle(source, solid, columns), output == 3)
    signatures, shapes, structures, glyphs = set(), set(), set(), set()
    volumes, levels, counts, keysizes, roles, source_positions = set(), set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        if seed in (0, 17):
            assert _digest(generate_puzzle(seed)) == _digest(puzzle)
        for split in ('train', 'test'):
            for p in puzzle[split]:
                f = _features(p['input'], p['output'])
                m = f['meta']
                columns = [c for r, c in m['keys'].values()]
                assert np.array_equal(_settle(m['source'], m['solid'], columns), p['output'] == m['water'])
                # Repack the same volume within the same compartment. Source
                # shape and vertical placement do not alter its liquid amount.
                a, b = m['intervals'][m['source_section']]
                available = np.zeros(m['source'].shape, bool)
                available[:, a:b] = ~m['solid'][:, a:b]
                available[1, 1] = available[1, -2] = False
                points = np.argwhere(available)
                rng = np.random.default_rng(seed + 100)
                points = points[rng.permutation(len(points))[:int(m['source'].sum())]]
                repacked = np.zeros_like(available)
                repacked[points[:, 0], points[:, 1]] = True
                assert np.array_equal(_settle(repacked, m['solid'], columns), p['output'] == m['water'])
                changed_input = p['input'].copy()
                r0, r1, c0, c1 = m['bounds']
                changed_tank = changed_input[r0:r1, c0:c1]
                changed_tank[changed_tank == m['water']] = m['field']
                changed_tank[repacked] = m['water']
                assert np.array_equal(solve(changed_input), p['output'])
                shapes.add(p['input'].shape)
                # Normalize by visible roles and actual first-occurrence keys.
                normalized = np.full(m['tank'].shape, 0, np.int8)
                normalized[m['tank'] == m['wall']] = 1
                normalized[m['source']] = 2
                for i, color in enumerate(sorted(m['keys'], key=lambda c: m['keys'][c][1])):
                    normalized[m['tank'] == color] = 3+i
                structures.add(hashlib.sha256(normalized.tobytes()+bytes(normalized.shape)).hexdigest())
                glyphs.update(f['glyphs'])
                volumes.add(f['volume'])
                levels.add(f['levels'])
                counts.update(m['counts'][c] for c in m['keys'])
                keysizes.add(len(m['keys']))
                roles.add((m['background'], m['field'], m['wall'], m['water']))
                source_positions.add(m['source_section'])
    assert len(signatures) == 42 and len(structures) >= 180
    assert len(shapes) >= 20 and len(glyphs) >= 40 and len(volumes) >= 15
    assert counts >= {1, 2, 3, 4, 5} and keysizes == {2, 3, 4}
    assert len(levels) >= 20 and len(roles) >= 35 and len(source_positions) >= 4
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(123, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), ('4', 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Parser/solver recover locally permuted roles without any training state.
    p = generate_puzzle(19, 2, 1)
    perm = np.random.default_rng(99).permutation(10)
    for pair in p['train'] + p['test']:
        assert np.array_equal(solve(perm[pair['input']]), perm[pair['output']])
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupted output accepted")
